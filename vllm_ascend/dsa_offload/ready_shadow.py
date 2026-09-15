# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""真实LIDU/KSC/SFA轨迹上的有限ready对拍。所有采集/D2H均在显式eager调试模式。

候选永远不写模型状态，不在层内抛精度异常（避免另一TP rank卡在后续collective）。
generate结束后由collective_rpc汇总全部rank，失败或覆盖不足才令入口失败。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from vllm_ascend.dsa_offload.contracts import DSA_ROW_MODE_SPARSE

TOPK, QUERIES, TAIL, BLOCK, ROW_BYTES = 2048, 4, 256, 128, 656


class ReadyShadow:
    def __init__(self, config, runtime, rank: int):
        self.config, self.runtime, self.rank = config, runtime, rank
        self.layers: list[int] = []
        self.full_layer: int | None = None
        self.selected = None
        self.pending: dict[int, dict] = {}
        self.counts: dict[int, int] = {}
        self.attempts = 0
        self.errors: list[dict] = []
        self.skips: dict[str, int] = {}

    def register_layer(self, layer, state, source):
        if state == self.config.group:
            self.full_layer = layer
        if state == self.config.group or (source is not None and source == self.full_layer):
            self.layers.append(layer)
            self.counts[layer] = 0

    def _skip(self, reason):
        self.skips[reason] = self.skips.get(reason, 0) + 1

    def _failure(self, phase, layer, error, sample=None):
        result = dict(rank=self.rank, layer=layer, epoch=self.runtime._forward_epoch,
                      phase=phase, error=str(error), passed=False)
        if sample is not None:
            try:
                directory = Path(self.config.directory)
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"rank{self.rank}_step{self.runtime._forward_epoch}_layer{layer}.pt"
                torch.save(sample, path)
                result["snapshot"] = str(path)
            except Exception as save_error:
                result["snapshot_error"] = str(save_error)
        self.errors.append(result)
        print("DSA_READY_SHADOW_FAILURE " + json.dumps(result), flush=True)

    def before_selection(self, context, cache_slots, rows, row_modes, actual_q):
        if context.layer_id != self.full_layer:
            return
        self.selected = None
        if self.attempts >= self.config.steps or self.errors:
            return
        try:
            if rows.numel() != 1 or actual_q.cpu().tolist() != [QUERIES]:
                self._skip("not_bs1_mtp3")
                return
            if row_modes.reshape(-1).cpu().tolist() != [DSA_ROW_MODE_SPARSE]:
                self._skip("not_sparse")
                return
            row = int(rows.cpu()[0])
            before = cache_slots[row].detach().cpu().clone()
            budget = int(before[-1])
            if budget <= 0:
                self._skip("first_fill")
                return
            self.attempts += 1
            self.selected = dict(version=1, epoch=self.runtime._forward_epoch, pool_row=row,
                                 group=self.config.group, rank=self.rank, budget=budget,
                                 mirror_before=before[:-1].clone())
        except Exception as error:
            self._failure("before_lidu", context.layer_id, error)

    @staticmethod
    def _pages(packed, table):
        # 只采本请求使用的HBM页，不clone模型整层的全局cache allocation。
        return packed.view(torch.uint8).index_select(0, table.to(packed.device).long()).cpu().clone()

    def before_copy(self, context, packed, table, arena, dram_table, outputs):
        selected = self.selected
        layer = context.layer_id
        if layer not in self.layers or selected is None or selected["epoch"] != self.runtime._forward_epoch:
            return
        try:
            budget = selected["budget"]
            page_ids = table[0, :(budget + TAIL) // BLOCK].detach().cpu().clone()
            if torch.any(page_ids < 0) or page_ids.unique().numel() != page_ids.numel():
                raise ValueError("shadow requires unique resident pages covering sparse budget + tail")
            count = int(outputs.miss_count[0].cpu())
            sources = outputs.topk_index[0].reshape(-1)[:count].cpu().clone()
            destinations = outputs.topk_slots[0].reshape(-1)[:count].cpu().clone()
            # 用模型已有DRAM页表定位，只搬实际miss字节，不复制整个DRAM arena。
            logical_table = dram_table[0].cpu()
            source_pages = logical_table[sources.long() // BLOCK].long()
            if torch.any(source_pages <= 0):
                raise ValueError("miss references unpublished/null local DRAM page")
            physical = source_pages * BLOCK + sources.long() % BLOCK
            payload = arena.view(torch.uint8).reshape(-1, ROW_BYTES).index_select(
                0, physical.to(arena.device)).cpu().clone()
            state = self.runtime.resident_token_pool.get_cache_slots(self.config.group)
            sample = dict(selected, layer=layer, page_ids=page_ids,
                          mirror_after=state[selected["pool_row"], :-1].cpu().clone(),
                          copy_sources=sources, copy_destinations=destinations, payload=payload,
                          topk=self.runtime._a5_attention_src_ids[:QUERIES, 0, :TOPK].cpu().clone(),
                          tail_slots=self.runtime._a5_attention_slots[:QUERIES, 0, TOPK:].cpu().clone(),
                          packed_before=self._pages(packed, page_ids))
            self.pending[layer] = sample
        except Exception as error:
            self._failure("before_ksc", layer, error)

    def after_attention(self, context, query, packed, reference, scale):
        sample = self.pending.pop(context.layer_id, None)
        if sample is None:
            return
        try:
            sample.update(query=query.detach().cpu().clone(), reference=reference.detach().cpu().clone(),
                          scale=float(scale), packed_after=self._pages(packed, sample["page_ids"]))
            from kvcache_scatter_copy_custom_ops.ready_shadow import validate_ready

            result = validate_ready(sample, str(query.device))
            result.update(rank=self.rank, epoch=sample["epoch"], layer=context.layer_id,
                          group=self.config.group, inference="local_lidu_ksc_sfa", selection="reference_lidu")
            self.counts[context.layer_id] += 1
            print("DSA_READY_SHADOW_RESULT " + json.dumps(result), flush=True)
            if not result["passed"]:
                self._failure("compare", context.layer_id, result, sample)
        except Exception as error:
            self._failure("candidate", context.layer_id, error, sample)

    def report(self):
        covered = bool(self.layers) and all(n == self.config.steps for n in self.counts.values())
        return dict(rank=self.rank, passed=covered and not self.errors, source="ready", path="remote",
                    counts=self.counts, required_per_layer=self.config.steps, attempts=self.attempts,
                    skips=self.skips, errors=self.errors, real_pool=False, selection="reference_lidu")


def ready_shadow_report(worker):
    """供LLM.collective_rpc在generate完成后调用；不在层内增加TP collective。"""
    runtime = worker.model_runner.dsa_offload_runtime
    if runtime is None or runtime.ready_shadow is None:
        return dict(passed=False, error="ready shadow not initialized on worker")
    return runtime.ready_shadow.report()
