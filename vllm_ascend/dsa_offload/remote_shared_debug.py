# SPDX-License-Identifier: Apache-2.0
"""有限步eager诊断；只读副本，不修改模型选择、mirror、ticket或APPLIED。

额外LI、同步、D2H和落盘会改变调度；本模块输出不得用于性能比较。
数值差异/诊断失败记录到report，由generate后的worker RPC统一汇总。
"""
import csv
import json
from pathlib import Path

import torch


def difference(actual, reference, limit=16):
    a, b = actual.tolist(), reference.tolist()
    sa, sb = set(a), set(b)
    return dict(position_diff=sum(x != y for x, y in zip(a, b)),
                sorted_diff=sum(x != y for x, y in zip(sorted(a), sorted(b))),
                intersection=len(sa & sb), actual_only_count=len(sa - sb), reference_only_count=len(sb - sa),
                actual_only_head=sorted(sa - sb)[:limit], reference_only_head=sorted(sb - sa)[:limit],
                actual_duplicates=len(a)-len(sa), reference_duplicates=len(b)-len(sb))


class RemoteSharedDebugSession:
    def __init__(self, owner, config, dp, tp, physical):
        self.owner, self.config = owner, config
        self.dp, self.tp, self.physical = dp, tp, physical
        self.enabled = (dp, tp) in config.ranks
        self.epochs = {}
        self.errors = []
        self.io_rows = self.io_layers = self.li_comparisons = 0
        self.topk_rows_seen = set()
        self.directory = None

    def _sample(self):
        if not self.enabled:
            return False
        epoch = self.owner.runtime._forward_epoch
        if epoch not in self.epochs and len(self.epochs) < self.config.steps:
            self.epochs[epoch] = len(self.epochs)
        return epoch in self.epochs

    def _emit(self, name, value):
        print(name + " " + json.dumps(value, ensure_ascii=False), flush=True)

    def _error(self, phase, layer, error):
        value = dict(dp=self.dp, tp=self.tp, epoch=self.owner.runtime._forward_epoch,
                     layer=layer, phase=phase, error=str(error))
        self.errors.append(value)
        self._emit("DSA_REMOTE_DEBUG_ERROR", value)

    def _directory(self):
        if self.directory is None:
            shadow = self.owner.runtime.remote_store_shadow
            self.directory = (Path(self.config.directory) / f"store{shadow.store.store_id}_ns{shadow.namespace}"
                              / f"dp{self.dp}_tp{self.tp}")
            self.directory.mkdir(parents=True, exist_ok=False)
        return self.directory

    def select(self, context, data, query, key, weights, scale, key_scale, actual_q,
               rows, candidates, visible, table):
        data["debug"] = None
        if not self._sample():
            return
        try:
            b = rows.numel()
            sample = dict(epoch=self.owner.runtime._forward_epoch, source_layer=context.layer_id,
                          rows=rows.cpu().tolist(), candidates=candidates.cpu().tolist(),
                          visible=visible.cpu().tolist(), logical=data["logical"].detach().cpu().reshape(b, 4, 2048).clone())
            data["debug"] = sample
            if self.config.compare_li:
                from vllm_ascend.dsa_offload.ops import quant_lightning_indexer_topk
                common = dict(query=query, key=key, weights=weights, query_dequant_scale=scale,
                              key_dequant_scale=key_scale, actual_seq_lengths_query=actual_q, block_table=table)
                # 保持完整batch/TND，不为了观察一个row裁成BS1。两次调用均只读，独立输出。
                for name, lens, mode in (("li_same_domain", candidates, 0), ("li_full", visible, 3)):
                    sample[name] = quant_lightning_indexer_topk(
                        **common, candidate_lens=lens, sparse_mode=mode).detach().cpu().reshape(b, 4, 2048)
            self._directory()
            for i, row in enumerate(sample["rows"]):
                if row not in self.config.topk_rows:
                    continue
                self.topk_rows_seen.add(row)
                for q in range(4):
                    logical = sample["logical"][i, q]
                    value = dict(step=self.epochs[sample["epoch"]], epoch=sample["epoch"], dp=self.dp, tp=self.tp,
                                 layer=context.layer_id, row=row, request_id=self.owner.request_ids[row], query=q,
                                 candidate_D=sample["candidates"][i], visible_V=sample["visible"][i],
                                 lissend_head=logical[:self.config.topk_print_limit].tolist(),
                                 lissend_out_of_range=int(((logical < 0) | (logical >= sample["candidates"][i])).sum()))
                    for name in ("li_same_domain", "li_full"):
                        if name not in sample:
                            continue
                        ref = sample[name][i, q]
                        diff = difference(logical, ref, self.config.topk_print_limit)
                        value[name+"_head"] = ref[:self.config.topk_print_limit].tolist()
                        value[name+"_diff"] = diff
                        if name == "li_same_domain":
                            self.li_comparisons += 1
                            if diff["sorted_diff"] or diff["actual_duplicates"] or diff["reference_duplicates"] or value["lissend_out_of_range"]:
                                self._error("li_same_domain", context.layer_id, f"row={row} q={q} {diff}")
                        else:
                            value["li_full_tail_tokens"] = int((ref >= sample["candidates"][i]).sum())
                            value["li_full_out_of_range"] = int(((ref < -1) | (ref >= sample["visible"][i]-3+q)).sum())
                    self._emit("DSA_LI_COMPARE", value)
        except Exception as error:
            self._error("li_reference", context.layer_id, error)

    def consume(self, context, data, payload):
        sample = data.get("debug")
        if not self.enabled or sample is None or sample["epoch"] != self.owner.runtime._forward_epoch:
            return
        try:
            self._consume(context, data, payload, sample)
        except Exception as error:
            self._error("reply_snapshot", context.layer_id, error)

    def _consume(self, context, data, payload, sample):
        rows = data["rows"].cpu().tolist()
        if rows != sample["rows"]:
            raise ValueError("debug batch order changed since LISSend")
        status = payload["status"].cpu()
        generations = payload["generation"].cpu().tolist()
        shadow = self.owner.runtime.remote_store_shadow
        per_request = []
        operation = None
        for group, generation in enumerate(generations):
            begin = group*16
            size = min(16, len(rows)-begin)
            reply = self.owner.native.lis_send_pool_debug_read(self.owner.handle, generation, size)
            descriptors = reply["descriptors"].tolist()
            expected = [[row, int(self.owner.bound[row, 0]), int(self.owner.bound[row, 1])]
                        for row in rows[begin:begin+size]]
            if descriptors != expected or reply["layer"] != context.layer_id or reply["group"] != group:
                raise ValueError("reply command identity does not match active layer/group/rows/binding")
            wanted_op = "DU_AND_GATHER" if context.layer_id == context.execution_plan.first_payload_layer else "GATHER_ONLY"
            if reply["operation"] != wanted_op or reply["selection_group"] != context.execution_plan.group:
                raise ValueError("reply operation/selection group mismatch")
            plans = sample.setdefault("plan_generations", {})
            if wanted_op == "DU_AND_GATHER":
                plans[group] = generation
            if reply["plan_generation"] != plans.get(group):
                raise ValueError("reply refers to a different selection plan generation")
            operation = reply["operation"]
            offsets = reply["offsets"].tolist()
            for local, (row, handle, version) in enumerate(descriptors):
                i = begin+local
                records = reply["records"][offsets[local]:offsets[local+1]]
                misses = len(records)
                logical = sample["logical"][i]
                tokens = set(logical.flatten().tolist())
                if any(t < 0 or t >= sample["candidates"][i] for t in tokens):
                    raise ValueError(f"row={row}: logical TopK outside durable candidate range")
                miss_tokens = records[:, 0].tolist()
                if len(set(miss_tokens)) != misses or not set(miss_tokens) <= tokens:
                    raise ValueError(f"row={row}: reply miss tokens are not a unique subset of TopK")
                if status[i, 0].item() != generation or status[i, 1].item() != 0:
                    raise ValueError("debug CopySFA status mismatch")
                sent = reply["topk"][local] if reply["topk"] is not None else None
                sent_diff = int((sent != logical).sum()) if sent is not None else None
                if sent_diff:
                    self._error("sent_topk", context.layer_id, f"row={row} mismatch={sent_diff}")
                record = dict(step=self.epochs[sample["epoch"]], epoch=sample["epoch"], dp=self.dp, tp=self.tp,
                              physical_device=self.physical, source_layer=sample["source_layer"],
                              payload_layer=context.layer_id, selection_group=context.execution_plan.group,
                              command_group=group, generation=generation, plan_generation=reply["plan_generation"],
                              operation=operation, batch_position=i,
                              pool_row=row, request_id=self.owner.request_ids[row], store_id=shadow.store.store_id,
                              request_handle=handle, kv_version=version, union_topk=len(tokens),
                              union_hits_derived=len(tokens)-misses if sent is not None else None,
                              fetch_tokens=misses, payload_bytes=misses*656,
                              victims=int((records[:, 2] >= 0).sum()), group_metadata_bytes=reply["metadata_bytes"],
                              group_payload_bytes=reply["payload_bytes"], completion=reply["completion"],
                              status=int(status[i, 1]), sent_topk_mismatch=sent_diff)
                path = self._directory() / "summary.csv"
                header = not path.exists()
                with path.open("a", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(output, fieldnames=list(record))
                    if header:
                        writer.writeheader()
                    writer.writerow(record)
                self.io_rows += 1
                per_request.append(dict(row=row, request_handle=handle, fetch_tokens=misses))
                if row in self.config.topk_rows:
                    saved = dict(metadata=record, logical_topk=logical.clone(), records=records.clone())
                    if sent is not None:
                        saved["sent_topk"] = sent.clone()
                    for name in ("li_same_domain", "li_full"):
                        if name in sample:
                            saved[name] = sample[name][i].clone()
                    name = f"step{record['step']}_layer{context.layer_id}_row{row}_gen{generation}"
                    torch.save(saved, self.directory / f"{name}.pt")
                    (self.directory / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
        self.io_layers += 1
        total = sum(r["fetch_tokens"] for r in per_request)
        self._emit("DSA_REMOTE_IO", dict(dp=self.dp, tp=self.tp, step=self.epochs[sample["epoch"]],
                   layer=context.layer_id, operation=operation, batch=len(rows), generations=generations,
                   fetch_tokens=total, payload_bytes=total*656, per_request=per_request, performance_valid=False))

    def report(self):
        covered = (bool(self.io_rows) and len(self.epochs) == self.config.steps
                   and set(self.config.topk_rows) <= self.topk_rows_seen
                   and (not self.config.compare_li or self.li_comparisons > 0))
        return dict(dp=self.dp, tp=self.tp, enabled=self.enabled, passed=not self.enabled or (covered and not self.errors),
                    sampled_steps=len(self.epochs), required_steps=self.config.steps, io_rows=self.io_rows,
                    io_layers=self.io_layers, li_comparisons=self.li_comparisons, errors=self.errors,
                    directory=str(self.directory) if self.directory else None, performance_valid=False)


def remote_shared_debug_report(worker):
    """generate后统一取回；不在层内因数值不一致退出TP collective。"""
    runtime = worker.model_runner.dsa_offload_runtime
    remote = runtime.remote_shared if runtime is not None else None
    if remote is None or remote.debug is None:
        return dict(passed=False, error="remote shared debug is not initialized")
    return remote.debug.report()
