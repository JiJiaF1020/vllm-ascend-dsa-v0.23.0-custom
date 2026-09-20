# SPDX-License-Identifier: Apache-2.0
"""hbm_full_remote_shared首轮整网功能链；连接/PUT/刷新限定eager图外。

注册command/reply/completion只由算子runtime持有。这里的Tensor均为普通caller-owned
输入输出；不把远端GVA变成Tensor。首次decode绑定一批真实MTP3请求，允许缩批/重排，
不支持padding、新请求补入或row复用。结束请求的绑定保留到worker shutdown。
"""
from __future__ import annotations

import logging

import torch

from vllm_ascend.dsa_offload.contracts import DSA_ROW_MODE_SPARSE
from vllm_ascend.dsa_offload.ops import DSAOffloadSelectionOutput

logger = logging.getLogger(__name__)


class RemoteShared:
    def __init__(self, config, runtime, plan, layers, *, dp_rank=0, tp_rank=0, physical_device=None):
        import kvcache_scatter_copy_custom_ops  # noqa: F401
        from kvcache_scatter_copy_custom_ops import custom_ops_lib

        self.native, self.config, self.runtime, self.plan = custom_ops_lib, config, runtime, plan
        self.identity = f"dp={dp_rank} tp={tp_rank} physical={physical_device}"
        self.layers = layers  # (context, packed KV)；只有真实target层，不含MTP proposer。
        self.dump_epoch = -1
        for name in ("lis_send_pool_connect_selection", "lis_send_pool_refresh", "lis_send_pool_close"):
            if not hasattr(self.native, name):
                raise RuntimeError(f"missing {name}; rebuild operator extension and C8 OPP together")
        self.handle = None
        self.rows = None
        self.active_rows = None
        self.request_ids = {}
        self.retired = set()
        self.bound = None
        self.buffers = {}
        self.current = None
        self.previous_out = None
        self.calls = self.fetches = self.sfas = 0

    def validate_requests(self, request_ids, rows):
        """不把同名新请求或新row静默绑定到仍存活的旧session。"""
        for request_id, row in zip(request_ids, rows, strict=True):
            row = int(row)
            if row in self.retired or (self.handle is not None and self.request_ids.get(row) != request_id):
                raise ValueError("remote shared supports one offline cohort: no new requests or retired-row reuse after BIND")

    def retire(self, row):
        self.retired.add(int(row))
        logger.info("DSA_SHARED_RETIRED %s row=%s connection_retained=1 store_retained=1 row_reusable=0",
                    self.identity, row)

    def dump_before_write(self, metadata):
        runtime = self.runtime
        if (not runtime.dump_before_cache_write or not runtime.dump_job_count
                or self.dump_epoch == runtime._forward_epoch):
            return
        # 已确认满块来自上一轮KV；必须在任一层prolog覆盖双tail之前保存。
        # 首轮eager集中完成所有层PUT，避免源full等待尚未执行的shared层而死锁。
        logger.info("DSA_SHARED_PREWRITE_BEGIN epoch=%s pages_per_layer=%s",
                    runtime._forward_epoch, runtime.dump_job_count)
        for context, packed in self.layers:
            context.dump_full_blocks(
                resident_cache=(packed,), before_cache_write=True,
                full_dump_table=metadata.dsa_indexer_block_table,
                full_dump_rows=metadata.dsa_resident_pool_indices,
            )
        self.dump_epoch = runtime._forward_epoch
        logger.info("DSA_SHARED_PREWRITE_END epoch=%s", self.dump_epoch)

    def _connect(self, rows, candidates, budgets):
        host_rows = rows.cpu().tolist()
        b = len(host_rows)
        if host_rows != list(range(b)) or not 1 <= b <= 32 or len(set(budgets)) != 1:
            raise ValueError("first remote shared run requires fixed dense pool rows 0..B-1 and one HBM budget")
        shadow = self.runtime.remote_store_shadow
        wm = shadow.watermarks
        done = wm.done[:b] * 128
        if bool((done < candidates.cpu()).any()) or bool((shadow.handles[:b] == 0).any()):
            raise RuntimeError("shared remote GET precedes all-layer PUT_DONE; complete prefill PUT first")
        bound = torch.stack((shadow.handles[:b], wm.done_versions[:b], done.to(torch.int64)), dim=1).contiguous()
        owners = torch.full((len(self.plan.groups), b, budgets[0]), -1, dtype=torch.int32)
        for group in self.plan.groups:
            if group.remote_layers:
                mirror = self.runtime.resident_token_pool.get_cache_slots(group.group)
                mirror[:b, :self.runtime.max_model_len].fill_(-1)
        host, port = self.config.du_endpoint.rsplit(":", 1)
        logger.info("DSA_SHARED_CONNECT_BEGIN %s endpoint=%s rows=%s slots=%s handles=%s namespace=%s",
                    self.identity, self.config.du_endpoint, host_rows, budgets[0],
                    bound[:, 0].tolist(), shadow.namespace)
        self.handle = self.native.lis_send_pool_connect_selection(
            host, int(port), self.config.device_eid, str(self.runtime.device), owners,
            self.runtime.max_logical_blocks * 128, b, self.runtime.num_layers,
            [group.source_layer for group in self.plan.groups], shadow.store.store_id, bound,
            False, self.config.eid_index, self.config.pool_eid)
        wm.acknowledge_binding(torch.arange(b), bound[:, 1], (bound[:, 2] // 128).int())
        self.rows, self.bound, self.budget = host_rows, bound, budgets[0]
        self.active_rows = host_rows
        self.request_ids = {row: shadow.request_ids[row] for row in host_rows}
        self.groups = (b + 15) // 16
        self.period = ((self.runtime.num_layers * self.groups + 7) // 8) * 8
        self.arange_tail = torch.arange(256, dtype=torch.int32, device=self.runtime.device)
        self.causal_query = torch.arange(4, dtype=torch.int32, device=self.runtime.device)
        logger.info("DSA_SHARED_CONNECT_OK %s wire=5 backend=real_store groups=%s", self.identity, len(self.plan.groups))

    def _buffer(self, context, batch):
        # 仅保存各SG当前活跃shape；上一轮已完整消费，不累计32种batch的scratch。
        key = context.execution_plan.group
        if key in self.buffers and self.buffers[key]["status"].shape[0] != batch:
            del self.buffers[key]
        if key not in self.buffers:
            device = self.runtime.device
            empty = lambda shape, dtype: torch.empty(shape, dtype=dtype, device=device)
            self.buffers[key] = dict(
                prepared=empty((batch*4, 1, 2304), torch.int32),
                logical=empty((batch*4, 1, 2048), torch.int32),
                full=empty((batch*4, 1, 2304), torch.int32),
                generation=empty(((batch+15)//16,), torch.int64),
                ticket=torch.zeros(((batch+15)//16, 2), dtype=torch.int64, device=device),
                status=empty((batch, 4), torch.int64),
                maximum=empty((1,), torch.float32), total=empty((1,), torch.float32),
                tail_info=empty((batch, 2), torch.int32))
        return self.buffers[key]

    def _generation(self, data, layer):
        start = self.runtime._forward_epoch * self.period + layer * self.groups + 1
        # 图外eager验证；后续图版将固定shape的generation H2D放到step边界。
        data["generation"].copy_(torch.arange(start, start+data["generation"].numel(), dtype=torch.int64,
                                            device=self.runtime.device))

    @staticmethod
    def _check_ticket(data, phase):
        ticket = data["ticket"].cpu()
        if not torch.equal(ticket[:, 0], data["generation"].cpu()) or bool(ticket[:, 1].any()):
            raise RuntimeError(f"{phase} submission failed: {ticket.tolist()}")

    def select(self, context, query, weights, scale, actual_q, rows, candidates, visible, row_modes, table, budgets):
        if torch.compiler.is_compiling():
            raise RuntimeError("remote shared first integration requires eager, not graph capture")
        b = rows.numel()
        if actual_q.cpu().tolist() != [4*(i+1) for i in range(b)] or query.shape != (4*b, 32, 128):
            raise ValueError("remote shared decode currently requires real MTP3 Q=4 per request, no padding")
        if row_modes.cpu().tolist() != [DSA_ROW_MODE_SPARSE]*b:
            raise ValueError("remote shared decode requires all rows SPARSE; use a prompt exceeding resident budget")
        pool = self.runtime.resident_token_pool
        # full不再运行LIDU；budget来自scheduler，不从尚未初始化的mirror metadata猜测。
        budgets = budgets.cpu().tolist()
        if self.handle is None:
            self._connect(rows, candidates, budgets)
        host_rows = rows.cpu().tolist()
        if (not 1 <= b <= len(self.rows) or len(set(host_rows)) != b or
                any(row not in self.request_ids or row in self.retired for row in host_rows) or
                budgets != [self.budget]*b):
            raise ValueError("remote shared requires live rows from the initial cohort and one unchanged HBM budget")
        # generation的layer/group步幅固定为连接容量；只改变本次调用的活跃group数。
        # 缩批/重排会让旧plan中部分row不再出现在下一条命令，必须先用全row APPLIED
        # 快照确认旧plan，否则Pool仍等待已经退出请求的确认而无法接受下一轮。
        changed = host_rows != self.active_rows
        if changed or bool((candidates.cpu() > self.bound[host_rows, 2]).any()):
            shadow = self.runtime.remote_store_shadow
            wm = shadow.watermarks
            done = wm.done[self.rows].to(torch.int64) * 128
            if bool((candidates.cpu() > done[host_rows]).any()):
                raise RuntimeError("remote candidate D exceeds all-layer PUT_DONE; this eager PoC cannot GET a page still being dumped by later layers")
            bound = torch.stack((shadow.handles[self.rows], wm.done_versions[self.rows], done), dim=1).contiguous()
            # 所有上一轮shared已完成APPLIED；刷新会同步设备并等待Host CQE/commit。
            self.native.lis_send_pool_refresh(self.handle, bound)
            self.bound = bound
            wm.acknowledge_binding(torch.tensor(self.rows), bound[:, 1], (bound[:, 2] // 128).int())
            logger.info("DSA_SHARED_REFRESH_OK bound=%s active_rows=%s batch_changed=%s connection_retained=1",
                        bound[:, 2].tolist(), host_rows, changed)
        self.active_rows = host_rows
        tail_length = visible.cpu() - candidates.cpu()
        if bool(((tail_length < 4) | (tail_length > 256)).any()):
            raise ValueError("framework causal tail must fit 256 tokens including four queries")
        data = self._buffer(context, b)
        self._generation(data, context.execution_plan.first_payload_layer)
        key, key_scale = context.indexer_cache
        torch.ops.nanovllm_dsa.lis_send(
            query.contiguous(), key, weights.contiguous(), scale.contiguous(), key_scale.squeeze(-1),
            actual_q, rows, candidates, table.contiguous(), data["generation"], data["ticket"],
            pool.get_cache_slots(context.execution_plan.group), data["prepared"], None, data["logical"])
        self._check_ticket(data, "LISSend")
        tokens = candidates[:, None, None] + self.arange_tail[None, None, :]
        valid = tokens < (visible[:, None, None] - 3 + self.causal_query[None, :, None])
        tail = torch.where(valid, tokens.expand(b, 4, 256), -1).reshape(4*b, 1, 256)
        data["full"][:, :, :2048].copy_(data["logical"])
        data["full"][:, :, 2048:].copy_(tail)
        data["prepared"][:, :, 2048:].copy_(torch.where(tail >= 0, self.budget + (tail & 255), -1))
        data.update(rows=rows, actual_q=actual_q, context=context, epoch=self.runtime._forward_epoch)
        self.current = data
        self.runtime._selection_source_layer = context.layer_id
        self.calls += data["generation"].numel()
        logger.info("DSA_SHARED_SELECT source=%s payload=%s epoch=%s commands=%s", context.layer_id,
                    context.execution_plan.first_payload_layer, self.runtime._forward_epoch, self.calls)
        return DSAOffloadSelectionOutput(data["full"], data["tail_info"], visible)

    def fetch(self, context):
        data = self.current
        if (data is None or data["epoch"] != self.runtime._forward_epoch or
                data["context"].layer_id != context.selection_source_layer_id):
            raise RuntimeError("shared layer has no current source selection plan")
        if context.layer_id != context.execution_plan.first_payload_layer:
            self._generation(data, context.layer_id)
            torch.ops.custom.pool_fetch_shared_out(self.previous_out, data["rows"], data["generation"],
                                                  data["status"], data["ticket"])
            self._check_ticket(data, "GATHER_ONLY")
            self.fetches += data["generation"].numel()
        return DSAOffloadSelectionOutput(data["prepared"], data["tail_info"])

    def attention(self, context, query, packed, table, actual_q, scale):
        data = self.current
        out = torch.empty((*query.shape[:-1], 512), dtype=query.dtype, device=query.device)
        torch.ops.nanovllm_dsa.copy_sfa_mtp_c8_out(
            query.contiguous(), packed, data["prepared"], table.contiguous(), actual_q, data["rows"],
            data["generation"], data["ticket"],
            self.runtime.resident_token_pool.get_cache_slots(context.execution_plan.group),
            scale, out, data["maximum"], data["total"], data["status"],
            context.execution_plan.mirror_writer == "scatter_sfa")
        self.previous_out = out
        if context.execution_plan.applied:
            torch.ops.custom.remote_du_applied_out(out, data["rows"], data["generation"],
                                                  data["status"], data["ticket"])
            self._check_ticket(data, "APPLIED")
        # 功能验收逐层检查，不把这里的同步端到端时间作为性能成绩。
        status = data["status"].cpu()
        expected = data["generation"].cpu().repeat_interleave(16)[:status.shape[0]]
        if not torch.equal(status[:, 0], expected) or bool(status[:, 1].any()):
            raise RuntimeError(f"ScatterSFA failed at layer {context.layer_id}: {status.tolist()}")
        self.sfas += 1
        logger.info("DSA_SHARED_CONSUMED %s layer=%s generation=%s mirror=%s applied=%s", self.identity, context.layer_id,
                    expected.tolist(), context.execution_plan.mirror_writer, context.execution_plan.applied)
        return out

    def close(self):
        if self.handle is None:
            return
        self.native.lis_send_pool_close(self.handle)
        self.handle = None
        self.rows = self.active_rows = self.bound = self.current = self.previous_out = None
        self.buffers.clear()
        logger.info("DSA_SHARED_DRAIN_OK %s select=%s gather_only=%s scatter_sfa=%s real_store=1",
                    self.identity, self.calls, self.fetches, self.sfas)
