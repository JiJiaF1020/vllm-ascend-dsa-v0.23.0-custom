# SPDX-License-Identifier: Apache-2.0
"""hbm_full_remote_shared功能链；连接/PUT/刷新/检查均在decode图外。

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
    def __init__(self, config, runtime, plan, layers, *, dp_rank=0, tp_rank=0, physical_device=None,
                 graph_enabled=False, debug_config=None):
        import kvcache_scatter_copy_custom_ops  # noqa: F401
        from kvcache_scatter_copy_custom_ops import custom_ops_lib

        self.native, self.config, self.runtime, self.plan = custom_ops_lib, config, runtime, plan
        self.identity = f"dp={dp_rank} tp={tp_rank} physical={physical_device}"
        self.layers = layers  # (context, packed KV)；只有真实target层，不含MTP proposer。
        self.contexts = {context.layer_id: context for context, _ in layers}
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
        self.previous_status = None
        self.calls = self.fetches = self.sfas = 0
        self.graph_enabled = graph_enabled
        self.prepared_epoch = -1
        self.warmed_batches = set()
        self.layer_buffers = {}
        self.step_data = None
        self.graph_inputs = {}
        self.debug = None
        if debug_config is not None:
            if graph_enabled:
                raise ValueError("remote_shared debug requires eager")
            if not hasattr(self.native, "lis_send_pool_debug_read"):
                raise RuntimeError("remote_shared debug requires updated operator extension")
            from vllm_ascend.dsa_offload.remote_shared_debug import RemoteSharedDebugSession
            self.debug = RemoteSharedDebugSession(self, debug_config, dp_rank, tp_rank, physical_device)

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
        # 图外集中完成所有层PUT，避免源full等待尚未执行的shared层而死锁。
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
        binding_rows = list(range(b))
        if (sorted(host_rows) != binding_rows or not 1 <= b <= 32
                or len(budgets) != b or len(set(budgets)) != 1):
            raise ValueError(
                "first remote shared run requires the pool row set 0..B-1 (any batch order) "
                f"and one HBM budget; {self.identity} rows={host_rows} budgets={budgets}")
        shadow = self.runtime.remote_store_shadow
        wm = shadow.watermarks
        done = wm.done[:b] * 128
        # PUT_DONE/binding按稳定pool row存放；candidate与query按本轮batch顺序存放。
        if bool((done[host_rows] < candidates.cpu()).any()) or bool((shadow.handles[:b] == 0).any()):
            raise RuntimeError("shared remote GET precedes all-layer PUT_DONE; complete prefill PUT first")
        bound = torch.stack((shadow.handles[:b], wm.done_versions[:b], done.to(torch.int64)), dim=1).contiguous()
        owners = torch.full((len(self.plan.groups), b, budgets[0]), -1, dtype=torch.int32)
        for group in self.plan.groups:
            if group.remote_layers:
                mirror = self.runtime.resident_token_pool.get_cache_slots(group.group)
                mirror[:b, :self.runtime.max_model_len].fill_(-1)
        host, port = self.config.du_endpoint.rsplit(":", 1)
        logger.info("DSA_SHARED_CONNECT_BEGIN %s endpoint=%s active_rows=%s bound_rows=%s slots=%s handles=%s namespace=%s",
                    self.identity, self.config.du_endpoint, host_rows, binding_rows, budgets[0],
                    bound[:, 0].tolist(), shadow.namespace)
        self.handle = self.native.lis_send_pool_connect_selection(
            host, int(port), self.config.device_eid, str(self.runtime.device), owners,
            self.runtime.max_logical_blocks * 128, b, self.runtime.num_layers,
            [group.source_layer for group in self.plan.groups], shadow.store.store_id, bound,
            False, self.config.eid_index, self.config.pool_eid)
        wm.acknowledge_binding(torch.arange(b), bound[:, 1], (bound[:, 2] // 128).int())
        # REFRESH仍按稳定row生成全量快照，不能继承首次batch的排列。
        self.rows, self.bound, self.budget = binding_rows, bound, budgets[0]
        self.active_rows = host_rows
        self.request_ids = {row: shadow.request_ids[row] for row in host_rows}
        self.groups = (b + 15) // 16
        self.period = ((self.runtime.num_layers * self.groups + 7) // 8) * 8
        self.arange_tail = torch.arange(256, dtype=torch.int32, device=self.runtime.device)
        self.causal_query = torch.arange(4, dtype=torch.int32, device=self.runtime.device)
        logger.info("DSA_SHARED_CONNECT_OK %s wire=5 backend=real_store groups=%s", self.identity, len(self.plan.groups))

    def _buffer(self, context, batch):
        # 图持有每种batch的地址；缩批不能删除已经捕获的buffer。
        key = (context.execution_plan.group, batch)
        if key not in self.buffers:
            device = self.runtime.device
            empty = lambda shape, dtype: torch.empty(shape, dtype=dtype, device=device)
            self.buffers[key] = dict(
                prepared=empty((batch*4, 1, 2304), torch.int32),
                logical=empty((batch*4, 1, 2048), torch.int32),
                full=empty((batch*4, 1, 2304), torch.int32),
                ticket=torch.zeros(((batch+15)//16, 2), dtype=torch.int64, device=device),
                maximum=empty((1,), torch.float32), total=empty((1,), torch.float32),
                tail_info=empty((batch, 2), torch.int32))
        return self.buffers[key]

    def _payload(self, context, batch):
        key = (context.layer_id, batch)
        if key not in self.layer_buffers:
            group = self._buffer(context, batch)
            self.layer_buffers[key] = dict(
                generation=torch.empty(((batch+15)//16,), dtype=torch.int64, device=self.runtime.device),
                status=torch.empty((batch, 4), dtype=torch.int64, device=self.runtime.device),
                ticket=group["ticket"],
            )
        return self.layer_buffers[key]

    def prepare_decode(self, metadata):
        """在整模型forward/capture之外完成PUT、binding和本轮generation更新。"""
        self.step_data = None
        if not metadata.dsa_decode_forward:
            return
        rows = metadata.dsa_resident_pool_indices
        candidates = metadata.dsa_candidate_lens
        budgets = metadata.dsa_sparse_budget_tokens.cpu().tolist()
        b = rows.numel()
        if metadata.cum_query_lens.cpu().tolist() != [4*(i+1) for i in range(b)]:
            raise ValueError("remote shared decode requires four queries per request")
        if metadata.dsa_row_modes.cpu().tolist() != [DSA_ROW_MODE_SPARSE]*b:
            raise ValueError("remote shared decode requires all rows SPARSE")
        tail = metadata.seq_lens.cpu() - candidates.cpu()
        if bool(((tail < 4) | (tail > 256)).any()):
            raise ValueError("framework causal tail must fit 256 tokens including four queries")
        self.dump_before_write(metadata)
        self._prepare_binding(rows, candidates, budgets)
        for context, _ in self.layers:
            if context.full_cache is None:
                self._generation(self._payload(context, b), context.layer_id)
            elif context.execution_plan.first_payload_layer is not None:
                self._buffer(context, b)
        self.prepared_epoch = self.runtime._forward_epoch
        self.step_data = (b, metadata)

    def finish_decode(self):
        """图外检查真实status；不把Host诊断算进设备FULL graph时间。"""
        if self.step_data is None:
            return
        b, _ = self.step_data
        for context, _ in self.layers:
            if context.full_cache is not None:
                continue
            data = self._payload(context, b)
            self._check_status(data, context.layer_id)
            if context.execution_plan.applied:
                self._check_ticket(data, "APPLIED")
        self.warmed_batches.add(b)
        self.calls += sum(bool(g.remote_layers) for g in self.plan.groups) * ((b+15)//16)
        self.fetches += sum(max(0, len(g.remote_layers)-1) for g in self.plan.groups) * ((b+15)//16)
        self.sfas += sum(len(g.remote_layers) for g in self.plan.groups)
        logger.info("DSA_SHARED_STEP_OK %s epoch=%s batch=%s checked=ScatterSFA,APPLIED",
                    self.identity, self.prepared_epoch, b)
        self.step_data = None

    def validate_graph_inputs(self, key, metadata, args, kwargs):
        """仅授权已预热真实decode捕获，并拒绝replay偷偷更换输入地址。"""
        if (not self.graph_enabled or self.step_data is None or self.handle is None
                or self.prepared_epoch != self.runtime._forward_epoch
                or self.step_data[0] not in self.warmed_batches):
            raise RuntimeError("remote FULL graph requires prepared, warmed real decode; dummy capture is forbidden")
        signature = []

        def record(name, value):
            if isinstance(value, torch.Tensor):
                signature.append((name, value.data_ptr(), tuple(value.shape), value.stride(), value.dtype))
            elif isinstance(value, (tuple, list)):
                for i, item in enumerate(value):
                    record(f"{name}.{i}", item)
            elif isinstance(value, dict):
                for field, item in sorted(value.items()):
                    record(f"{name}.{field}", item)

        record("args", args)
        record("kwargs", kwargs)
        for layer, meta in sorted(metadata.items()):
            for field in ("seq_lens", "cum_query_lens", "block_table", "slot_mapping", "cos", "sin",
                          "dsa_indexer_block_table", "dsa_indexer_slot_mapping", "dsa_row_modes",
                          "dsa_resident_pool_indices", "dsa_sparse_budget_tokens", "dsa_candidate_lens"):
                record(f"{layer}.{field}", getattr(meta, field, None))
        previous = self.graph_inputs.setdefault(key, signature)
        if previous != signature:
            raise RuntimeError("remote FULL graph input address/shape changed; do not replay stale metadata")

    def _prepare_binding(self, rows, candidates, budgets):
        if self.handle is None:
            self._connect(rows, candidates, budgets)
        host_rows = rows.cpu().tolist()
        if (not 1 <= len(host_rows) <= len(self.rows) or len(set(host_rows)) != len(host_rows) or
                any(row not in self.request_ids or row in self.retired for row in host_rows) or
                budgets != [self.budget]*len(host_rows)):
            raise ValueError("remote shared requires live rows from the initial cohort and one unchanged HBM budget")
        changed = host_rows != self.active_rows
        if changed or bool((candidates.cpu() > self.bound[host_rows, 2]).any()):
            shadow = self.runtime.remote_store_shadow
            wm = shadow.watermarks
            done = wm.done[self.rows].to(torch.int64) * 128
            if bool((candidates.cpu() > done[host_rows]).any()):
                raise RuntimeError("remote candidate D exceeds all-layer PUT_DONE")
            bound = torch.stack((shadow.handles[self.rows], wm.done_versions[self.rows], done), dim=1).contiguous()
            # 缩批/重排用全row APPLIED快照确认旧plan；REFRESH内部同步设备及Host commit。
            self.native.lis_send_pool_refresh(self.handle, bound)
            self.bound = bound
            wm.acknowledge_binding(torch.tensor(self.rows), bound[:, 1], (bound[:, 2] // 128).int())
            logger.info("DSA_SHARED_REFRESH_OK bound=%s active_rows=%s batch_changed=%s connection_retained=1",
                        bound[:, 2].tolist(), host_rows, changed)
        self.active_rows = host_rows

    @staticmethod
    def _check_status(data, layer):
        status = data["status"].cpu()
        expected = data["generation"].cpu().repeat_interleave(16)[:status.shape[0]]
        if not torch.equal(status[:, 0], expected) or bool(status[:, 1].any()):
            raise RuntimeError(f"ScatterSFA failed at layer {layer}: {status.tolist()}")

    def _generation(self, data, layer):
        start = self.runtime._forward_epoch * self.period + layer * self.groups + 1
        # 每层独立地址，数值在step边界H2D；不可在图内固化epoch或跨层复用地址。
        data["generation"].copy_(torch.arange(start, start+data["generation"].numel(), dtype=torch.int64,
                                            device=self.runtime.device))

    @staticmethod
    def _check_ticket(data, phase):
        ticket = data["ticket"].cpu()
        if not torch.equal(ticket[:, 0], data["generation"].cpu()) or bool(ticket[:, 1].any()):
            raise RuntimeError(f"{phase} submission failed: {ticket.tolist()}")

    def select(self, context, query, weights, scale, actual_q, rows, candidates, visible, row_modes, table, budgets):
        b = rows.numel()
        if query.shape != (4*b, 32, 128):
            raise ValueError("remote shared decode currently requires real MTP3 Q=4 per request, no padding")
        pool = self.runtime.resident_token_pool
        if self.graph_enabled:
            if self.prepared_epoch != self.runtime._forward_epoch:
                raise RuntimeError("remote decode must be prepared before model forward/capture")
        else:
            if actual_q.cpu().tolist() != [4*(i+1) for i in range(b)]:
                raise ValueError("remote shared requires four queries per request")
            if row_modes.cpu().tolist() != [DSA_ROW_MODE_SPARSE]*b:
                raise ValueError("remote shared decode requires all rows SPARSE")
            self._prepare_binding(rows, candidates, budgets.cpu().tolist())
            tail_length = visible.cpu() - candidates.cpu()
            if bool(((tail_length < 4) | (tail_length > 256)).any()):
                raise ValueError("framework causal tail must fit 256 tokens including four queries")
        data = self._buffer(context, b)
        payload_context = self.contexts[context.execution_plan.first_payload_layer]
        payload = self._payload(payload_context, b)
        if not self.graph_enabled:
            self._generation(payload, context.execution_plan.first_payload_layer)
        key, key_scale = context.indexer_cache
        torch.ops.nanovllm_dsa.lis_send(
            query.contiguous(), key, weights.contiguous(), scale.contiguous(), key_scale.squeeze(-1),
            actual_q, rows, candidates, table.contiguous(), payload["generation"], data["ticket"],
            pool.get_cache_slots(context.execution_plan.group), data["prepared"], None, data["logical"])
        if not self.graph_enabled:
            self._check_ticket(payload, "LISSend")
        if self.debug is not None:
            self.debug.select(context, data, query, key, weights, scale, key_scale, actual_q,
                              rows, candidates, visible, table)
        tokens = candidates[:, None, None] + self.arange_tail[None, None, :]
        valid = tokens < (visible[:, None, None] - 3 + self.causal_query[None, :, None])
        tail = torch.where(valid, tokens.expand(b, 4, 256), -1).reshape(4*b, 1, 256)
        data["full"][:, :, :2048].copy_(data["logical"])
        data["full"][:, :, 2048:].copy_(tail)
        data["prepared"][:, :, 2048:].copy_(torch.where(tail >= 0, self.budget + (tail & 255), -1))
        data.update(rows=rows, actual_q=actual_q, context=context, epoch=self.runtime._forward_epoch)
        self.current = data
        self.runtime._selection_source_layer = context.layer_id
        if not self.graph_enabled:
            self.calls += payload["generation"].numel()
            logger.info("DSA_SHARED_SELECT source=%s payload=%s epoch=%s commands=%s", context.layer_id,
                        context.execution_plan.first_payload_layer, self.runtime._forward_epoch, self.calls)
        return DSAOffloadSelectionOutput(data["full"], data["tail_info"], visible)

    def fetch(self, context):
        data = self.current
        if (data is None or data["epoch"] != self.runtime._forward_epoch or
                data["context"].layer_id != context.selection_source_layer_id):
            raise RuntimeError("shared layer has no current source selection plan")
        if context.layer_id != context.execution_plan.first_payload_layer:
            payload = self._payload(context, data["rows"].numel())
            if not self.graph_enabled:
                self._generation(payload, context.layer_id)
            torch.ops.custom.pool_fetch_shared_out(self.previous_out, data["rows"], payload["generation"],
                                                  self.previous_status, data["ticket"])
            if not self.graph_enabled:
                self._check_ticket(payload, "GATHER_ONLY")
                self.fetches += payload["generation"].numel()
        return DSAOffloadSelectionOutput(data["prepared"], data["tail_info"])

    def attention(self, context, query, packed, table, actual_q, scale):
        data = self.current
        payload = self._payload(context, data["rows"].numel())
        out = torch.empty((*query.shape[:-1], 512), dtype=query.dtype, device=query.device)
        torch.ops.nanovllm_dsa.copy_sfa_mtp_c8_out(
            query.contiguous(), packed, data["prepared"], table.contiguous(), actual_q, data["rows"],
            payload["generation"], data["ticket"],
            self.runtime.resident_token_pool.get_cache_slots(context.execution_plan.group),
            scale, out, data["maximum"], data["total"], payload["status"],
            context.execution_plan.mirror_writer == "scatter_sfa")
        self.previous_out = out
        self.previous_status = payload["status"]
        if context.execution_plan.applied:
            torch.ops.custom.remote_du_applied_out(out, data["rows"], payload["generation"],
                                                  payload["status"], data["ticket"])
            if not self.graph_enabled:
                self._check_ticket(payload, "APPLIED")
        # 功能验收逐层检查，不把这里的同步端到端时间作为性能成绩。
        if not self.graph_enabled:
            self._check_status(payload, context.layer_id)
            if self.debug is not None:
                self.debug.consume(context, data, payload, packed, table)
            self.sfas += 1
            logger.info("DSA_SHARED_CONSUMED %s layer=%s mirror=%s applied=%s", self.identity, context.layer_id,
                        context.execution_plan.mirror_writer, context.execution_plan.applied)
        return out

    def close(self):
        if self.handle is None:
            return
        self.native.lis_send_pool_close(self.handle)
        self.handle = None
        self.rows = self.active_rows = self.bound = self.current = self.previous_out = None
        self.previous_status = None
        self.buffers.clear()
        self.layer_buffers.clear()
        logger.info("DSA_SHARED_DRAIN_OK %s select=%s gather_only=%s scatter_sfa=%s real_store=1",
                    self.identity, self.calls, self.fetches, self.sfas)
