# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""图外 Store facade：复用 packed C8 页，不伪造可访问的本地 DRAM Tensor。

仅承担 CREATE/PUT/PUT_DONE。DU BIND ACK 之前不会推进 bound 水位；本模块
不替换现有 local_all runtime，也不把 native 控制 RPC 放入被捕获的图中。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

PAGE_TOKENS = 128
TOKEN_BYTES = 656


@dataclass(frozen=True)
class RemotePackedArena:
    """容量外观，不能传给本地 KSC，也不分配同尺寸的本地 backing。"""

    shape: tuple[int, ...]
    dtype: torch.dtype = torch.uint8
    device: torch.device = torch.device("cpu")

    def data_ptr(self) -> int:
        raise RuntimeError("remote arena has no local data pointer; use PUT/Get")


@dataclass(frozen=True)
class PutDoneBatch:
    handles: torch.Tensor  # CPU I64[N]
    versions: torch.Tensor  # CPU I64[N]
    full_blocks: torch.Tensor  # CPU I32[N]


class RemoteKvStore:
    """一个 worker 的图外控制客户端；Store 身份与 DU endpoint 分开管理。"""

    def __init__(self, endpoint: str, *, payload_layers: int, capacity: int) -> None:
        # 延迟导入，保持 local_all 不依赖远端算子包、socket 或 HCCP。
        from kvcache_scatter_copy_custom_ops import custom_ops_lib

        self._client = custom_ops_lib.RemoteKvStoreClient(endpoint)
        layers, rows, source_capacity, page_tokens, token_bytes = self._client.layout
        if (layers, source_capacity, page_tokens, token_bytes) != (payload_layers, capacity, PAGE_TOKENS, TOKEN_BYTES):
            raise ValueError("remote Store packed layout does not match worker")
        self.store_id = self._client.store_id
        self.capacity = capacity
        self.payload_layers = payload_layers
        self.max_requests = rows
        self.arena = RemotePackedArena((payload_layers, rows, capacity // PAGE_TOKENS, PAGE_TOKENS, 1, TOKEN_BYTES))

    def create_requests(self, keys: torch.Tensor, capacities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """CPU I64[N,2] 的稳定请求键、CPU I32[N] 容量；不使用 batch 下标作为身份。"""
        return self._client.create_requests(keys, capacities), capacities.clone()

    def dump_packed_pages(
        self,
        layer_id: int,
        packed_hbm: torch.Tensor,
        source_block_ids: torch.Tensor,
        request_handles: torch.Tensor,
        logical_block_ids: torch.Tensor,
    ) -> None:
        """从 producer 当前流同步提取完整页，按原始字节上传。

        图外功能路径，明确包含阻塞 D2H；不称为异步 PUT 或零同步推理。
        caller 必须在真实 producer 流调用，或先建立 producer→当前流依赖。
        返回后源页可以被本次 PUT 释放，但完整请求水位仍需 wait_put_done。
        """
        if not 0 <= layer_id < self.payload_layers:
            raise ValueError("invalid payload layer")
        if source_block_ids.device.type != "cpu" or source_block_ids.dtype != torch.int32 or source_block_ids.ndim != 1:
            raise ValueError("source_block_ids must be CPU int32 [N]")
        if packed_hbm.ndim != 4 or tuple(packed_hbm.shape[1:]) != (PAGE_TOKENS, 1, TOKEN_BYTES):
            raise ValueError("packed HBM must have shape [P,128,1,656]")
        if packed_hbm.element_size() != 1 or not packed_hbm.is_contiguous():
            raise ValueError("packed HBM must be contiguous one-byte storage")
        count = source_block_ids.numel()
        if request_handles.numel() != count or logical_block_ids.numel() != count:
            raise ValueError("PUT control columns must have matching lengths")
        if count == 0:
            return
        if bool(((source_block_ids < 0) | (source_block_ids >= packed_hbm.shape[0])).any()):
            raise ValueError("source block outside HBM allocation")
        # view(uint8) 是位解释，不对 FP8/INT8 做数值转换；只搬选中的页，不搬整个 arena。
        raw = packed_hbm.view(torch.uint8).reshape(-1, PAGE_TOKENS, TOKEN_BYTES)
        for start in range(0, count, 64):
            end = min(start + 64, count)
            ids = source_block_ids[start:end].to(device=packed_hbm.device, dtype=torch.int64)
            selected = raw.index_select(0, ids).to(device="cpu", non_blocking=False).contiguous()
            self._client.put_packed_pages(layer_id, request_handles[start:end], logical_block_ids[start:end], selected)

    def put_cpu_pages(self, layer_id: int, handles: torch.Tensor, pages: torch.Tensor, raw: torch.Tensor) -> None:
        """已有 CPU packed dump 可直接复用，避免多做一次 H2D→D2H。"""
        self._client.put_packed_pages(layer_id, handles, pages, raw)

    def wait_put_done(self, handles: torch.Tensor, targets: torch.Tensor) -> PutDoneBatch:
        return PutDoneBatch(*self._client.wait_put_done(handles, targets))

    def close_requests(self, handles: torch.Tensor) -> None:
        """必须先停止调度并获得 DU drain ACK；仍有 binding 时服务端明确拒绝。"""
        self._client.close_requests(handles)


class RemoteWatermarks:
    """固定 row 容量的 planned/done/bound 三水位，不能借用本地 durable 的单一字段。"""

    def __init__(self, rows: int) -> None:
        self.planned = torch.zeros(rows, dtype=torch.int32)
        self.done = torch.zeros_like(self.planned)
        self.bound = torch.zeros_like(self.planned)
        self.done_versions = torch.zeros(rows, dtype=torch.int64)
        self.bound_versions = torch.zeros_like(self.done_versions)

    def plan(self, rows: torch.Tensor, pages: torch.Tensor) -> None:
        if bool((pages < self.planned[rows]).any()):
            raise ValueError("planned watermark cannot decrease")
        self.planned[rows] = pages

    def complete(self, rows: torch.Tensor, result: PutDoneBatch) -> None:
        if bool(((result.full_blocks < self.done[rows]) | (result.full_blocks > self.planned[rows])).any()):
            raise ValueError("PUT_DONE must remain inside the planned prefix")
        if bool((result.versions < self.done_versions[rows]).any()):
            raise ValueError("PUT_DONE version cannot decrease")
        self.done[rows] = result.full_blocks
        self.done_versions[rows] = result.versions

    def acknowledge_binding(self, rows: torch.Tensor, versions: torch.Tensor, pages: torch.Tensor) -> None:
        """仅在真实 BIND/REFRESH ACK 后调用，不在发出 RPC 时预先推进。"""
        if bool(((pages < self.bound[rows]) | (pages > self.done[rows])).any()):
            raise ValueError("bound watermark must remain inside PUT_DONE")
        if bool(((versions < self.bound_versions[rows]) | (versions > self.done_versions[rows])).any()):
            raise ValueError("bound version must remain inside PUT_DONE")
        self.bound[rows] = pages
        self.bound_versions[rows] = versions
