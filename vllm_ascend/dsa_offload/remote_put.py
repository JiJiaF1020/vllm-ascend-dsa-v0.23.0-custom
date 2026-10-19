# SPDX-License-Identifier: Apache-2.0
"""直通PUT的固定地址普通控制输入；HBM源注册和通信区由native runtime隐藏持有。"""
from __future__ import annotations

import numpy as np
import torch
from vllm.v1.utils import CpuGpuBuffer

PAGE_TOKENS = 128
MAX_PAGES = 4096


class RemotePut:
    @staticmethod
    def allocate(runtime, confirmed_tokens=None):
        """在模型memory profile前分配普通控制Tensor；此时不需要KV源地址。"""
        rows = runtime.resident_token_pool.max_num_reqs
        capacity = min(MAX_PAGES, rows * runtime.num_layers)
        device = runtime.device
        plan_buffer = CpuGpuBuffer(capacity, 4, dtype=torch.int32, device=device, pin_memory=runtime.pin_memory)
        control_buffer = CpuGpuBuffer(5, dtype=torch.int64, device=device, pin_memory=runtime.pin_memory)
        control_buffer.np.fill(0)
        control_buffer.copy_to_gpu()
        return dict(
            plan_buffer=plan_buffer, control_buffer=control_buffer,
            plan=plan_buffer.gpu,
            count=control_buffer.gpu[4:5].view(torch.int32)[:1],
            confirmed=(confirmed_tokens if confirmed_tokens is not None else
                       torch.zeros((rows,), dtype=torch.int32, device=device)),
            write_end=torch.zeros((rows,), dtype=torch.int32, device=device),
            generation=control_buffer.gpu[:1], step_control=control_buffer.gpu[1:4],
            status=torch.zeros((4,), dtype=torch.int64, device=device),
            ticket=torch.zeros((2,), dtype=torch.int64, device=device))

    def __init__(self, remote, buffers=None):
        self.remote = remote
        self.runtime = remote.runtime
        self.rows = self.runtime.resident_token_pool.max_num_reqs
        for name, tensor in (buffers if buffers is not None else self.allocate(self.runtime)).items():
            setattr(self, name, tensor)
        self.capacity = self.plan.shape[0]
        self.serial = 0
        self.bound_handles = None
        self._full_sources = np.empty(self.rows, dtype=np.int32)
        self.first_group = next(group for group in remote.plan.groups if group.remote_layers)
        remote.native.lis_send_pool_register_put_sources(
            remote.handle, [packed for _, packed in remote.layers],
            [context.full_cache is None for context, _ in remote.layers])

    def bind_producers(self, handles):
        snapshot = handles[:self.rows].contiguous()
        if self.bound_handles is None or not torch.equal(snapshot, self.bound_handles):
            self.remote.native.lis_send_pool_bind_producers(self.remote.handle, snapshot)
            self.bound_handles = snapshot.clone()

    def _next(self):
        self.serial += 1
        self.control_buffer.np[0] = self.serial

    def put_layer(self, layer, source_blocks, rows, pages):
        """Prefill：在该层producer之后提交；仅等该批READ_DONE，不等尚未执行层。"""
        self.bind_producers(self.runtime.remote_store_shadow.handles)
        for start in range(0, len(pages), self.capacity):
            end = min(len(pages), start + self.capacity)
            host = self.plan_buffer.np[:end-start]
            host[:, 0] = rows[start:end]
            host[:, 1] = layer
            host[:, 2] = source_blocks[start:end]
            host[:, 3] = pages[start:end]
            self.plan_buffer.copy_to_gpu(end-start)
            self.control_buffer.np[4] = end-start
            self._next()
            self.control_buffer.copy_to_gpu()
            torch.ops.custom.pool_put_full_c8_out(self.plan, self.count, self.confirmed, self.write_end,
                                                self.generation, self.ticket, self.status, False)
            # 冷路径正确性边界：READ_DONE不是全层PUT_DONE，也不是DU BIND。
            result = self.status.cpu()
            if int(result[1]) != 0 or int(result[0]) != self.serial:
                raise RuntimeError(f"prefill direct PUT failed: {result.tolist()}")

    def prepare(self, metadata, confirmed_tokens, full_table_cpu=None):
        """图外只准备小型plan；模型入口的固定PUT节点提交，正常增长不等Host REFRESH。"""
        n = self.runtime.dump_job_count if self.runtime.active_num_reqs else 0
        self.control_buffer.np[1:4] = (self.runtime._forward_epoch + 1, self.runtime.active_num_reqs,
                                       int(self.remote.rows_changed))
        self.control_buffer.np[4] = 0
        # 直接消费本轮caller-owned batch顺序输入；不做to/index_copy或构造stable-row GPU副本。
        if confirmed_tokens is not None and (confirmed_tokens.data_ptr() != self.confirmed.data_ptr() or
                                             confirmed_tokens.dtype != self.confirmed.dtype):
            raise RuntimeError("direct PUT confirmed input must retain its startup-captured device address/dtype")
        self.input_confirmed = self.confirmed
        self.input_end = metadata.seq_lens
        self.input_rows = metadata.dsa_resident_pool_indices
        if not n or not self.runtime.dump_before_cache_write:
            self.control_buffer.copy_to_gpu()
            return
        if confirmed_tokens is None:
            raise RuntimeError("decode direct PUT requires authoritative device confirmed lengths")
        rows = self.runtime._dump_pool_indices[:n]
        pages = self.runtime._dump_logical_indices[:n]
        sources = self.runtime._dump_source_blocks[:n]
        active_rows = self.remote.active_rows
        total = n * len(self.remote.layers)
        if total > self.capacity or n > self.rows:
            raise RuntimeError("decode PUT exceeds preallocated all-layer plan capacity")
        full_sources = None
        if any(context.full_cache is not None for context, _ in self.remote.layers):
            # 与原dump相同的物理表；只在边界step读取一次，不逐层重复D2H。
            if full_table_cpu is None:
                raise RuntimeError("direct PUT needs the existing Host Indexer block table")
            table = full_table_cpu
            positions = {row: i for i, row in enumerate(active_rows)}
            full_sources = self._full_sources[:n]
            for i, (row, page) in enumerate(zip(rows, pages, strict=True)):
                full_sources[i] = table[positions[int(row)], int(page)]
        for index, (context, _) in enumerate(self.remote.layers):
            host = self.plan_buffer.np[index*n:(index+1)*n]
            host[:, 0] = rows
            host[:, 1] = context.layer_id
            host[:, 2] = full_sources if context.full_cache is not None else sources
            host[:, 3] = pages
        self.plan_buffer.copy_to_gpu(total)
        self.control_buffer.np[4] = total
        self._next()
        self.control_buffer.copy_to_gpu()

    def submit(self, batch):
        # 复用首个LISSend的普通mutable ticket，保留图编译器可见的副作用依赖。
        context = self.remote.contexts[self.first_group.source_layer]
        ticket = self.remote._buffer(context, batch)["ticket"][0]
        torch.ops.custom.pool_put_full_c8_out(self.plan, self.count, self.input_confirmed, self.input_end,
                                            self.generation, ticket, self.status, True,
                                            request_rows=self.input_rows, step_control=self.step_control)
