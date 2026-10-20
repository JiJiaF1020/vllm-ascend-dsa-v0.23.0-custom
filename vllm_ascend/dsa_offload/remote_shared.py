# SPDX-License-Identifier: Apache-2.0
"""hbm_full_remote_shared功能链；连接/源注册/首次BIND在图外，直通PUT在图内提交。

注册command/reply/completion只由算子runtime持有。这里的Tensor均为普通caller-owned
输入输出；不把远端GVA变成Tensor。首次decode绑定一批真实MTP3请求，允许缩批/重排，
模型允许padding，远端节点只处理真实请求前缀；不支持新请求补入或row复用。
图仅按模型BatchDescriptor缓存；设备rows=-1标识PAD，结束请求绑定保留到worker shutdown。

remote_shared配置中validate默认False；显式True或开启debug时恢复Host验收检查。
关闭验收不取消设备completion/APPLIED、PUT_DONE/绑定和图地址生命周期检查。
"""
from __future__ import annotations

import logging
import socket
from contextlib import contextmanager

import torch
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend import envs
from vllm_ascend.dsa_offload.contracts import DSA_ROW_MODE_SPARSE
from vllm_ascend.dsa_offload.ops import DSAOffloadSelectionOutput

logger = logging.getLogger(__name__)


class RemoteShared:
    def __init__(self, config, runtime, plan, layers, *, dp_rank=0, tp_rank=0, physical_device=None,
                 graph_enabled=False, debug_config=None, validate=False, worker_parking=False,
                 startup_pad=False, reserved=None, put_transport="urma_read", put_visibility="next_step_local"):
        import kvcache_scatter_copy_custom_ops  # noqa: F401
        from kvcache_scatter_copy_custom_ops import custom_ops_lib

        self.native, self.config, self.runtime, self.plan = custom_ops_lib, config, runtime, plan
        if getattr(self.native, "lis_send_fused_output_version", lambda: 0)() < 1:
            raise RuntimeError("remote shared requires fused-output LISSend extension and matching C8 OPP; rebuild both")
        if getattr(self.native, "lis_send_weight_stride_version", lambda: 0)() < 1:
            raise RuntimeError("remote shared requires strided-weight LISSend extension and matching C8 OPP; rebuild both")
        if graph_enabled and getattr(self.native, "lis_send_padding_version", lambda: 0)() < 1:
            raise RuntimeError("remote fixed-capacity graph requires PAD-aware extension and matching C8/ordinary OPP")
        self.identity = f"host={socket.gethostname()} dp={dp_rank} tp={tp_rank} physical={physical_device}"
        self.host_mstx = None
        if envs.VLLM_ASCEND_DSA_HOST_MSTX:
            from torch_npu.npu import mstx

            self.host_mstx = mstx
        self.layers = layers  # (context, packed KV)；只有真实target层，不含MTP proposer。
        self.contexts = {context.layer_id: context for context, _ in layers}
        self.dump_epoch = -1
        for name in ("lis_send_pool_connect_selection", "lis_send_pool_refresh", "lis_send_pool_close"):
            if not hasattr(self.native, name):
                raise RuntimeError(f"missing {name}; rebuild operator extension and C8 OPP together")
        self.handle = reserved[3] if reserved else None
        self.startup_pad = startup_pad
        self.put_transport = put_transport
        self.deferred_put = put_visibility == "next_step_local"
        self.rows_changed = False
        self.real_bound = False
        if reserved:
            self.groups = (reserved[4] + 15) // 16
            self.period = ((runtime.num_layers * self.groups + 7) // 8) * 8
            self.budget = reserved[5]
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
        self.validate = validate or debug_config is not None
        self.worker_parking = worker_parking
        self.notified_epoch = -1
        self.last_payload_layer = max((layer for group in plan.groups for layer in group.remote_layers), default=None)
        if worker_parking and not hasattr(self.native, "lis_send_pool_begin_step"):
            raise RuntimeError("worker_parking requires updated operator extension and Pool")
        self.prepared_epoch = -1
        self.warmed_batches = set()
        self.layer_buffers = {}
        self.generation_owner = None
        self.generation_epoch = -1
        self.step_data = None
        self.graph_inputs = {}
        self.debug = None
        self.direct_put = None
        if put_transport == "urma_read":
            if self.handle is None or getattr(self.native, "lis_send_direct_put_version", lambda: 0)() < 2:
                raise RuntimeError("direct PUT requires reserved connection, updated Pool/Store/extension and ordinary/C8 OPP")
            from vllm_ascend.dsa_offload.remote_put import RemotePut

            self.direct_put = RemotePut(self, reserved[6] if reserved and len(reserved) > 6 else None)
            runtime.remote_store_shadow.direct_put = self.direct_put
        if debug_config is not None:
            if graph_enabled:
                raise ValueError("remote_shared debug requires eager")
            if not hasattr(self.native, "lis_send_pool_debug_read"):
                raise RuntimeError("remote_shared debug requires updated operator extension")
            from vllm_ascend.dsa_offload.remote_shared_debug import RemoteSharedDebugSession
            self.debug = RemoteSharedDebugSession(self, debug_config, dp_rank, tp_rank, physical_device)

    @contextmanager
    def host_range(self, stage, batch):
        """仅供图外调用；标签不读Tensor值，stream=None不向设备插入打点任务。"""
        if self.host_mstx is None:
            yield
            return
        label = (f"DSA_HOST/{stage} {self.identity} "
                 f"epoch={self.runtime._forward_epoch} batch={batch}")
        range_id = self.host_mstx.range_start(label, stream=None)
        try:
            yield
        finally:
            if range_id:
                self.host_mstx.range_end(range_id)

    def validate_requests(self, request_ids, rows):
        """不把同名新请求或新row静默绑定到仍存活的旧session。"""
        for request_id, row in zip(request_ids, rows, strict=True):
            row = int(row)
            if row in self.retired or (self.real_bound and self.request_ids.get(row) != request_id):
                raise ValueError("remote shared supports one offline cohort: no new requests or retired-row reuse after BIND")

    def retire(self, row):
        self.retired.add(int(row))
        logger.info("DSA_SHARED_RETIRED %s row=%s connection_retained=1 store_retained=1 row_reusable=0",
                    self.identity, row)

    def dump_before_write(self, metadata):
        if self.direct_put is not None:
            return  # 模型首层在图内提交全层plan；不再图外逐层D2H/TCP PUT。
        runtime = self.runtime
        if (not runtime.dump_before_cache_write or not runtime.dump_job_count
                or self.dump_epoch == runtime._forward_epoch):
            logger.debug(
                "DSA_SHARED_PREWRITE_SKIP %s epoch=%s enabled=%s pages_per_layer=%s last_dump_epoch=%s",
                self.identity, runtime._forward_epoch, runtime.dump_before_cache_write,
                runtime.dump_job_count, self.dump_epoch,
            )
            return
        # 已确认满块来自上一轮KV；必须在任一层prolog覆盖双tail之前保存。
        # 图外集中完成所有层PUT，避免源full等待尚未执行的shared层而死锁。
        logger.info("DSA_SHARED_PREWRITE_BEGIN %s epoch=%s pages_per_layer=%s",
                    self.identity, runtime._forward_epoch, runtime.dump_job_count)
        for context, packed in self.layers:
            context.dump_full_blocks(
                resident_cache=(packed,), before_cache_write=True,
                full_dump_table=metadata.dsa_indexer_block_table,
                full_dump_rows=metadata.dsa_resident_pool_indices,
            )
        self.dump_epoch = runtime._forward_epoch
        logger.info("DSA_SHARED_PREWRITE_END %s epoch=%s", self.identity, self.dump_epoch)

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
        if self.handle is None:
            self.handle = self.native.lis_send_pool_connect_selection(
                host, int(port), self.config.device_eid, str(self.runtime.device), owners,
                self.runtime.max_logical_blocks * 128, b, self.runtime.num_layers,
                [group.source_layer for group in self.plan.groups], shadow.store.store_id, bound,
                False, self.config.eid_index, self.config.pool_eid)
            self.groups = (b + 15) // 16
            self.period = ((self.runtime.num_layers * self.groups + 7) // 8) * 8
        else:
            if budgets != [self.budget] * b:
                raise ValueError("real HBM budget differs from startup reservation")
            capacity = self.runtime.resident_token_pool.max_num_reqs
            full_bound = torch.zeros((capacity, 3), dtype=torch.int64)
            full_bound[:b].copy_(bound)
            self.native.lis_send_pool_refresh(self.handle, full_bound)
            bound = full_bound
        self.real_bound = True
        wm.acknowledge_binding(torch.arange(b), bound[:b, 1], (bound[:b, 2] // 128).int())
        # REFRESH仍按稳定row生成全量快照，不能继承首次batch的排列。
        self.rows, self.bound, self.budget = binding_rows, bound, budgets[0]
        self.active_rows = host_rows
        self.request_ids = {row: shadow.request_ids[row] for row in host_rows}
        if self.worker_parking:
            self.native.lis_send_pool_begin_step(self.handle, 0, [])
        logger.info("DSA_SHARED_CONNECT_OK %s wire=%s backend=real_store groups=%s", self.identity,
                    6 if self.deferred_put else 5, len(self.plan.groups))

    def _buffer(self, context, batch):
        # 按图容量而非active B持有地址；同容量缩批只更新输入值。
        key = (context.execution_plan.group, batch)
        if key not in self.buffers:
            device = self.runtime.device
            empty = lambda shape, dtype: torch.empty(shape, dtype=dtype, device=device)
            self.buffers[key] = dict(
                prepared=empty((batch*4, 1, 2304), torch.int32),
                full=empty((batch*4, 1, 2304), torch.int32),
                ticket=torch.zeros(((batch+15)//16, 2), dtype=torch.int64, device=device),
                maximum=empty((1,), torch.float32), total=empty((1,), torch.float32),
                rows=empty((batch,), torch.int32),
                tail_info=empty((batch, 2), torch.int32))
        return self.buffers[key]

    def _payload(self, context, batch):
        key = (context.layer_id, batch)
        if key not in self.layer_buffers:
            if self.generation_owner is None:
                self.generation_owner = CpuGpuBuffer(self.runtime.num_layers, self.groups, dtype=torch.int64,
                                                     device=self.runtime.device, pin_memory=self.runtime.pin_memory)
            group = self._buffer(context, batch)
            self.layer_buffers[key] = dict(
                generation=self.generation_owner.gpu[context.layer_id, :(batch+15)//16],
                status=torch.empty((batch, 4), dtype=torch.int64, device=self.runtime.device),
                ticket=group["ticket"],
            )
        return self.layer_buffers[key]

    def prepare_decode(self, metadata, *, confirmed_tokens=None, full_table_cpu=None, cache_layout=None):
        """图外准备plan/普通输入与generation；TCP回退仍在这里同步PUT/REFRESH。"""
        self.step_data = None
        if not metadata.dsa_decode_forward:
            return
        with self.host_range("prepare_decode", metadata.dsa_resident_pool_indices.numel()):
            self._prepare_decode(metadata, confirmed_tokens=confirmed_tokens, full_table_cpu=full_table_cpu, cache_layout=cache_layout)

    def _prepare_decode(self, metadata, *, confirmed_tokens=None, full_table_cpu=None, cache_layout=None):
        b = self.runtime.active_num_reqs
        graph_b = metadata.dsa_resident_pool_indices.numel()
        if not 0 <= b <= graph_b:
            raise RuntimeError(f"remote active batch {b} exceeds metadata rows {graph_b}")
        self.current = self.previous_out = self.previous_status = None
        logger.debug("DSA_SHARED_GRAPH_ROWS %s epoch=%s active=%s graph_rows=%s pad=%s remote_commands_per_layer=%s",
                     self.identity, self.runtime._forward_epoch, b, graph_b, graph_b-b, (b+15)//16)
        if b == 0:
            self.rows_changed = bool(self.active_rows)
            self.active_rows = []
            if self.direct_put is not None:
                self.direct_put.prepare(metadata, None)
            # 已连接rank可在同容量图中replay全PAD；从未建链的idle仍仅走eager。
            if self.handle is not None:
                for context, _ in self.layers:
                    if context.full_cache is None:
                        self._generation(self._payload(context, graph_b), context.layer_id)
                    elif context.execution_plan.first_payload_layer is not None:
                        self._buffer(context, graph_b)
            self.prepared_epoch = self.runtime._forward_epoch
            self.step_data = (0, metadata)
            return
        rows = metadata.dsa_resident_pool_indices[:b]
        candidates = metadata.dsa_candidate_lens[:b]
        with self.host_range("metadata_prepare", b):
            if cache_layout is None:
                raise RuntimeError("remote prepare_decode requires existing Host cache-layout columns")
            host_rows = cache_layout.resident_pool_indices_cpu[:b].tolist()
            budgets = cache_layout.sparse_budget_tokens_cpu[:b].tolist()
            host_candidates = torch.from_numpy(cache_layout.candidate_lens_cpu[:b])
            if self.validate:
                if metadata.cum_query_lens[:b].cpu().tolist() != [4*(i+1) for i in range(b)]:
                    raise ValueError("remote shared decode requires four queries per request")
                if metadata.dsa_row_modes[:b].cpu().tolist() != [DSA_ROW_MODE_SPARSE]*b:
                    raise ValueError("remote shared decode requires all rows SPARSE")
                tail = metadata.seq_lens[:b].cpu() - candidates.cpu()
                if bool(((tail < 4) | (tail > 256)).any()):
                    raise ValueError("framework causal tail must fit 256 tokens including four queries")
        with self.host_range("dump_before_write", b):
            self.dump_before_write(metadata)
        with self.host_range("prepare_binding", b):
            self._prepare_binding(rows, candidates, budgets, host_rows, host_candidates)
        if self.direct_put is not None:
            self.direct_put.prepare(metadata, confirmed_tokens, full_table_cpu)
        with self.host_range("generation_update", b):
            for context, _ in self.layers:
                if context.full_cache is None:
                    self._generation(self._payload(context, graph_b), context.layer_id)
                elif context.execution_plan.first_payload_layer is not None:
                    self._buffer(context, graph_b)
        self.prepared_epoch = self.runtime._forward_epoch
        self.step_data = (b, metadata)

    def notify_step_submitted(self):
        """图外、model forward下发后调用；不读取设备Tensor，不等待设备完成或TCP ACK。"""
        if not self.worker_parking or self.handle is None or self.last_payload_layer is None:
            return
        epoch = self.runtime._forward_epoch
        if epoch == self.notified_epoch:
            return
        if self.graph_enabled:
            if self.step_data is None or self.prepared_epoch != epoch:
                return
            batch = self.step_data[0]
        else:
            if self.current is None or self.current["epoch"] != epoch:
                return
            batch = self.current["rows"].numel()
        if batch == 0:
            return
        start = epoch * self.period + self.last_payload_layer * self.groups + 1
        self.native.lis_send_pool_begin_step(self.handle, epoch + 1, list(range(start, start + (batch + 15) // 16)))
        self.notified_epoch = epoch

    def finish_decode(self):
        """维护步末记账；仅显式验收模式同步读取设备status/ticket。"""
        if self.step_data is None:
            return
        with self.host_range("finish_decode", self.step_data[0]):
            self._finish_decode()

    def _finish_decode(self):
        b, metadata = self.step_data
        if b == 0:
            self.step_data = None
            return
        if self.validate:
            for context, _ in self.layers:
                if context.full_cache is not None:
                    continue
                data = self._payload(context, metadata.dsa_resident_pool_indices.numel())
                self._check_status(data, context.layer_id)
                if context.execution_plan.applied:
                    self._check_ticket(data, "APPLIED")
        self.warmed_batches.add(b)
        self.calls += sum(bool(g.remote_layers) for g in self.plan.groups) * ((b+15)//16)
        self.fetches += sum(max(0, len(g.remote_layers)-1) for g in self.plan.groups) * ((b+15)//16)
        self.sfas += sum(len(g.remote_layers) for g in self.plan.groups)
        if self.validate:
            logger.info("DSA_SHARED_STEP_OK %s epoch=%s batch=%s checked=ScatterSFA,APPLIED",
                        self.identity, self.prepared_epoch, b)
        self.step_data = None

    def validate_graph_inputs(self, key, metadata, args, kwargs):
        """授权已预热真实decode或无IO的idle图，拒绝replay更换输入地址。"""
        if (not self.graph_enabled or self.step_data is None
                or self.prepared_epoch != self.runtime._forward_epoch):
            raise RuntimeError("remote FULL graph requires prepared decode/PAD metadata")
        b = self.step_data[0]
        if b != self.runtime.active_num_reqs:
            raise RuntimeError("remote prepared metadata does not match active batch")
        if self.handle is None or (not self.startup_pad and not self.warmed_batches):
            raise RuntimeError("remote graph requires an existing connection and one warmed real decode")
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
        if self.direct_put is not None:
            for field in ("plan", "count", "confirmed", "generation", "step_control"):
                record(f"put.{field}", getattr(self.direct_put, field))
        for layer, meta in sorted(metadata.items()):
            for field in ("seq_lens", "cum_query_lens", "block_table", "slot_mapping", "cos", "sin",
                          "dsa_indexer_block_table", "dsa_indexer_slot_mapping", "dsa_row_modes",
                          "dsa_resident_pool_indices", "dsa_sparse_budget_tokens", "dsa_candidate_lens"):
                record(f"{layer}.{field}", getattr(meta, field, None))
        previous = self.graph_inputs.setdefault(key, signature)
        if previous != signature:
            raise RuntimeError("remote FULL graph input address/shape changed; do not replay stale metadata")

    def _prepare_binding(self, rows, candidates, budgets, host_rows=None, host_candidates=None):
        if not self.real_bound:
            with self.host_range("connect", rows.numel()):
                self._connect(rows, candidates, budgets)
        host_rows = rows.cpu().tolist() if host_rows is None else host_rows
        if (not 1 <= len(host_rows) <= len(self.rows) or len(set(host_rows)) != len(host_rows) or
                any(row not in self.request_ids or row in self.retired for row in host_rows) or
                budgets != [self.budget]*len(host_rows)):
            raise ValueError("remote shared requires live rows from the initial cohort and one unchanged HBM budget")
        changed = host_rows != self.active_rows
        self.rows_changed = changed
        if self.deferred_put:
            self.active_rows = host_rows
            return  # 图内step控制携带新APPLIED；缩批/重排不再同步设备或发Host REFRESH。
        host_candidates = candidates.cpu() if host_candidates is None else host_candidates
        growth = self.direct_put is None and bool((host_candidates > self.bound[host_rows, 2]).any())
        if changed or growth:
            shadow = self.runtime.remote_store_shadow
            wm = shadow.watermarks
            if self.direct_put is not None:
                # 仅缩批/重排的控制面刷新；普通满页增长由Pool在图内gate前发布。
                # 先收尾上一图，再读CURRENT_DONE；否则Host可在上一PUT gate前读到旧版本。
                torch.npu.synchronize(self.runtime.device)
                snapshot = shadow.store._client.current_done(shadow.handles[self.rows].contiguous())
                from vllm_ascend.dsa_offload.remote_store import PutDoneBatch

                wm.complete(torch.tensor(self.rows), PutDoneBatch(*snapshot))
            done = wm.done[self.rows].to(torch.int64) * 128
            if bool((host_candidates > done[host_rows]).any()):
                raise RuntimeError("remote candidate D exceeds all-layer PUT_DONE")
            active_bound = torch.stack((shadow.handles[self.rows], wm.done_versions[self.rows], done), dim=1).contiguous()
            bound = self.bound.clone()
            bound[self.rows] = active_bound
            # 缩批/重排用全row APPLIED快照确认旧plan；REFRESH内部同步设备及Host commit。
            with self.host_range("pool_refresh", rows.numel()):
                self.native.lis_send_pool_refresh(self.handle, bound)
            self.bound = bound
            wm.acknowledge_binding(torch.tensor(self.rows), bound[self.rows, 1], (bound[self.rows, 2] // 128).int())
            logger.info("DSA_SHARED_REFRESH_OK bound=%s active_rows=%s batch_changed=%s connection_retained=1",
                        bound[:, 2].tolist(), host_rows, changed)
        self.active_rows = host_rows

    def _check_status(self, data, layer):
        if not self.validate:
            return
        status = data["status"].cpu()[:self.runtime.active_num_reqs]
        expected = data["generation"].cpu().repeat_interleave(16)[:status.shape[0]]
        if not torch.equal(status[:, 0], expected) or bool(status[:, 1].any()):
            raise RuntimeError(f"ScatterSFA failed at layer {layer}: {status.tolist()}")

    def _generation(self, data, layer):
        if self.generation_epoch == self.runtime._forward_epoch:
            return
        # 固定owner中逐层互不重叠的view，整个模型每步一次H2D，不逐层启动arange/copy。
        base = self.runtime._forward_epoch * self.period + 1
        for index in range(self.runtime.num_layers):
            for group in range(self.groups):
                self.generation_owner.np[index, group] = base + index * self.groups + group
        self.generation_owner.copy_to_gpu()
        self.generation_epoch = self.runtime._forward_epoch

    def _check_ticket(self, data, phase):
        if not self.validate:
            return
        groups = (self.runtime.active_num_reqs + 15) // 16
        ticket = data["ticket"].cpu()[:groups]
        if not torch.equal(ticket[:, 0], data["generation"].cpu()[:groups]) or bool(ticket[:, 1].any()):
            raise RuntimeError(f"{phase} submission failed: {ticket.tolist()}")

    def select(self, context, query, weights, scale, actual_q, rows, candidates, visible, row_modes, table, budgets):
        b = rows.numel()
        if not 0 < b <= 32 or query.shape != (4*b, 32, 128):
            raise ValueError("remote shared decode requires fixed MTP3 Q=4 capacity B=1..32")
        pool = self.runtime.resident_token_pool
        if self.graph_enabled:
            if self.prepared_epoch != self.runtime._forward_epoch:
                raise RuntimeError("remote decode must be prepared before model forward/capture")
        else:
            if self.validate:
                if actual_q.cpu().tolist() != [4*(i+1) for i in range(b)]:
                    raise ValueError("remote shared requires four queries per request")
                if row_modes.cpu().tolist() != [DSA_ROW_MODE_SPARSE]*b:
                    raise ValueError("remote shared decode requires all rows SPARSE")
            if self.direct_put is None:
                self._prepare_binding(rows, candidates, budgets.cpu().tolist())
            elif self.prepared_epoch != self.runtime._forward_epoch:
                raise RuntimeError("direct PUT decode must be prepared before model forward")
            if self.validate:
                tail_length = visible.cpu() - candidates.cpu()
                if bool(((tail_length < 4) | (tail_length > 256)).any()):
                    raise ValueError("framework causal tail must fit 256 tokens including four queries")
        data = self._buffer(context, b)
        # LISSend直接读row_modes做通信门禁，并写出下游共用的固定地址PAD哨兵rows。
        # 不读取Host active B，不把本地padding_pool_index当作远端请求。
        payload_context = self.contexts[context.execution_plan.first_payload_layer]
        payload = self._payload(payload_context, b)
        if not self.graph_enabled:
            self._generation(payload, context.execution_plan.first_payload_layer)
        key, key_scale = context.indexer_cache
        torch.ops.nanovllm_dsa.lis_send(
            query.contiguous(), key, weights, scale.contiguous(), key_scale.squeeze(-1),
            actual_q, rows, candidates, table.contiguous(), payload["generation"], data["ticket"],
            pool.get_cache_slots(context.execution_plan.group), data["prepared"],
            final_seq_lengths_kv=visible, row_modes=row_modes,
            full_attention_indices=data["full"], normalized_req_pool_entries=data["rows"])
        rows = data["rows"]
        if not self.graph_enabled:
            self._check_ticket(payload, "LISSend")
        if self.debug is not None:
            self.debug.select(context, data, query, key, weights, scale, key_scale, actual_q,
                              rows, candidates, visible, table)
        data.update(actual_q=actual_q, context=context, epoch=self.runtime._forward_epoch)
        self.current = data
        self.runtime._selection_source_layer = context.layer_id
        if not self.graph_enabled:
            self.calls += payload["generation"].numel()
            logger.debug("DSA_SHARED_SELECT source=%s payload=%s epoch=%s commands=%s", context.layer_id,
                        context.execution_plan.first_payload_layer, self.runtime._forward_epoch, self.calls)
        return DSAOffloadSelectionOutput(data["full"], data["tail_info"], visible)

    def idle_selection(self, context, rows, visible):
        """零active专用图：保留层plan关系，禁止调用任何通信或索引更新算子。"""
        data = self._buffer(context, 0)
        data.update(rows=rows[:0], context=context, epoch=self.runtime._forward_epoch)
        self.current = data
        self.runtime._selection_source_layer = context.layer_id
        return DSAOffloadSelectionOutput(data["full"], data["tail_info"], visible[:0])

    def fetch(self, context):
        data = self.current
        if (data is None or data["epoch"] != self.runtime._forward_epoch or
                data["context"].layer_id != context.selection_source_layer_id):
            raise RuntimeError("shared layer has no current source selection plan")
        if data["rows"].numel() == 0:
            return DSAOffloadSelectionOutput(data["prepared"], data["tail_info"])
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
        b = data["rows"].numel()
        if b == 0:
            return torch.zeros((*query.shape[:-1], 512), dtype=query.dtype, device=query.device)
        payload = self._payload(context, b)
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
        # 显式验收才逐层同步检查；设备侧status/APPLIED依赖始终保留。
        if not self.graph_enabled:
            self._check_status(payload, context.layer_id)
            if self.debug is not None:
                self.debug.consume(context, data, payload, packed, table)
            self.sfas += 1
            logger.debug("DSA_SHARED_CONSUMED %s layer=%s mirror=%s applied=%s", self.identity, context.layer_id,
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
