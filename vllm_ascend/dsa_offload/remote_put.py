# SPDX-License-Identifier: Apache-2.0
"""直通PUT的固定地址普通控制输入；HBM源注册和通信区由native runtime隐藏持有。"""
from __future__ import annotations

import numpy as np
import torch

PAGE_TOKENS = 128
MAX_PAGES = 4096


class RemotePut:
    @staticmethod
    def allocate(runtime):
        """在模型memory profile前分配普通控制Tensor；此时不需要KV源地址。"""
        rows = runtime.resident_token_pool.max_num_reqs
        capacity = min(MAX_PAGES, rows * runtime.num_layers)
        device = runtime.device
        return dict(
            plan=torch.empty((capacity, 4), dtype=torch.int32, device=device),
            count=torch.zeros((1,), dtype=torch.int32, device=device),
            confirmed=torch.zeros((rows,), dtype=torch.int32, device=device),
            write_end=torch.zeros((rows,), dtype=torch.int32, device=device),
            generation=torch.zeros((1,), dtype=torch.int64, device=device),
            status=torch.zeros((4,), dtype=torch.int64, device=device),
            ticket=torch.zeros((2,), dtype=torch.int64, device=device),
            row_indices=torch.empty((rows,), dtype=torch.int64, device=device))

    def __init__(self, remote, buffers=None):
        self.remote = remote
        self.runtime = remote.runtime
        self.rows = self.runtime.resident_token_pool.max_num_reqs
        for name, tensor in (buffers if buffers is not None else self.allocate(self.runtime)).items():
            setattr(self, name, tensor)
        self.capacity = self.plan.shape[0]
        self.serial = 0
        self.bound_handles = None
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
        self.generation.fill_(self.serial)

    def put_layer(self, layer, source_blocks, rows, pages):
        """Prefill：在该层producer之后提交；仅等该批READ_DONE，不等尚未执行层。"""
        self.bind_producers(self.runtime.remote_store_shadow.handles)
        for start in range(0, len(pages), self.capacity):
            end = min(len(pages), start + self.capacity)
            host = np.column_stack((rows[start:end], np.full(end-start, layer),
                                    source_blocks[start:end], pages[start:end])).astype(np.int32)
            self.plan[:end-start].copy_(torch.from_numpy(host))
            self.count.fill_(end-start)
            self._next()
            torch.ops.custom.pool_put_full_c8_out(self.plan, self.count, self.confirmed, self.write_end,
                                                self.generation, self.ticket, self.status, False)
            # 冷路径正确性边界：READ_DONE不是全层PUT_DONE，也不是DU BIND。
            result = self.status.cpu()
            if int(result[1]) != 0 or int(result[0]) != self.serial:
                raise RuntimeError(f"prefill direct PUT failed: {result.tolist()}")

    def prepare(self, metadata, confirmed_tokens, full_table_cpu=None):
        """图外只准备小型plan；模型入口的固定PUT节点提交，正常增长不等Host REFRESH。"""
        n = self.runtime.dump_job_count
        if not n or not self.runtime.dump_before_cache_write:
            self.count.zero_()
            return
        if confirmed_tokens is None:
            raise RuntimeError("decode direct PUT requires authoritative device confirmed lengths")
        rows = self.runtime._dump_pool_indices[:n]
        pages = self.runtime._dump_logical_indices[:n]
        sources = self.runtime._dump_source_blocks[:n]
        active = self.runtime.active_num_reqs
        active_rows = self.remote.active_rows
        # 满页step才执行小型元数据拷贝；不读取KV、不执行D2H、TCP PUT或绑定等待。
        self.row_indices[:active].copy_(torch.tensor(active_rows, dtype=torch.int64))
        indices = self.row_indices[:active]
        self.confirmed.index_copy_(0, indices, confirmed_tokens[:active].to(torch.int32))
        self.write_end.index_copy_(0, indices, metadata.seq_lens[:active].to(torch.int32))
        full_sources = None
        if any(context.full_cache is not None for context, _ in self.remote.layers):
            # 与原dump相同的物理表；只在边界step读取一次，不逐层重复D2H。
            if full_table_cpu is None:
                raise RuntimeError("direct PUT needs the existing Host Indexer block table")
            table = full_table_cpu
            positions = {row: i for i, row in enumerate(active_rows)}
            full_sources = np.asarray([table[positions[int(row)], int(page)]
                                       for row, page in zip(rows, pages, strict=True)], dtype=np.int32)
        chunks = []
        for context, _ in self.remote.layers:
            source = full_sources if context.full_cache is not None else sources
            chunks.append(np.column_stack((rows, np.full(n, context.layer_id), source, pages)))
        host = np.concatenate(chunks).astype(np.int32)
        if len(host) > self.capacity:
            raise RuntimeError("decode PUT exceeds preallocated all-layer plan capacity")
        self.plan[:len(host)].copy_(torch.from_numpy(host))
        self.count.fill_(len(host))
        self._next()

    def submit(self, batch):
        # 复用首个LISSend的普通mutable ticket，保留图编译器可见的副作用依赖。
        context = self.remote.contexts[self.first_group.source_layer]
        ticket = self.remote._buffer(context, batch)["ticket"][0]
        torch.ops.custom.pool_put_full_c8_out(self.plan, self.count, self.confirmed, self.write_end,
                                            self.generation, ticket, self.status, True)
