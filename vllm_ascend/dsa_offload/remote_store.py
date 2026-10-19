# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""图外 Store facade：复用 packed C8 页，不伪造可访问的本地 DRAM Tensor。

仅承担 CREATE/PUT/PUT_DONE。DU BIND ACK 之前不会推进 bound 水位；本模块
不替换现有 local_all runtime，也不把 native 控制 RPC 放入被捕获的图中。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import numpy as np
import torch

PAGE_TOKENS = 128
TOKEN_BYTES = 656
logger = logging.getLogger(__name__)


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

        if not hasattr(custom_ops_lib, "RemoteKvStoreClient"):
            raise RuntimeError("operator package has no RemoteKvStoreClient; rebuild the matching Torch extension")
        logger.info("DSA_STORE_OPEN_BEGIN endpoint=%s native_library=%s", endpoint, custom_ops_lib.__file__)
        self._client = custom_ops_lib.RemoteKvStoreClient(endpoint)
        layers, rows, source_capacity, page_tokens, token_bytes = self._client.layout
        if (layers, page_tokens, token_bytes) != (payload_layers, PAGE_TOKENS, TOKEN_BYTES) or source_capacity < capacity:
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


class RemoteStoreShadow:
    """真实框架生命周期的写侧穿刺；保留本地计算，不建 DU binding。

    所有操作限定图外/eager。row 是框架稳定行，handle 是 Store 身份；每个
    worker 使用独立 namespace。同一个 row 在 CLOSE 后可绑定新请求，但当前
    Store 不回收旧 handle/物理页。PUT_DONE 只在全部 target 层到齐后查询。
    """

    def __init__(self, store: RemoteKvStore, rows: int, *, inference: str = "local_all") -> None:
        self.store = store
        self.inference = inference
        self.namespace = uuid.uuid4().int & ((1 << 63) - 1)
        self.next_request = 1
        self.handles = torch.zeros(rows, dtype=torch.int64)
        self.request_ids: dict[int, str] = {}
        self.watermarks = RemoteWatermarks(rows)
        self.layer_pages = np.zeros((store.payload_layers, rows), dtype=np.int32)
        self.put_pages = 0
        self.closed_requests = 0
        self.direct_put = None
        logger.info("DSA_STORE_OPEN inference=%s store_id=%s layers=%s capacity=%s",
                    inference, store.store_id, store.payload_layers, store.capacity)

    def ensure_requests(self, request_ids: list[str], pool_rows: np.ndarray) -> None:
        for request_id, row_value in zip(request_ids, pool_rows, strict=True):
            row = int(row_value)
            if row in self.request_ids:
                if self.request_ids[row] != request_id:
                    raise RuntimeError("Store row reused before CLOSE")
                continue
            # 单个 CREATE 成功后立即记录，后续 CREATE 失败也不会丢失已分配的 handle。
            keys = torch.tensor([[self.namespace, self.next_request]], dtype=torch.int64)
            capacities = torch.tensor([self.store.capacity], dtype=torch.int32)
            handles, _ = self.store.create_requests(keys, capacities)
            self.handles[row] = handles[0]
            self.request_ids[row] = request_id
            self.next_request += 1
            logger.info("DSA_STORE_CREATE row=%s request=%s handle=%s", row, request_id, int(handles[0]))
        if self.direct_put is not None:
            self.direct_put.bind_producers(self.handles)

    def plan_dump(self, pool_rows: np.ndarray, logical_pages: np.ndarray) -> None:
        if not pool_rows.size:
            return
        planned = self.watermarks.planned.numpy()
        np.maximum.at(planned, pool_rows, logical_pages + 1)

    def put_layer(self, layer: int, packed_hbm: torch.Tensor, source_blocks: np.ndarray,
                  pool_rows: np.ndarray, logical_pages: np.ndarray, *, before_cache_write: bool) -> None:
        if not pool_rows.size:
            return
        if torch.compiler.is_compiling():
            raise RuntimeError("store_shadow PUT must not enter a compiled graph")
        rows = torch.from_numpy(pool_rows.astype(np.int64, copy=False))
        if bool((self.handles[rows] == 0).any()):
            raise RuntimeError("Store PUT requires CREATE for each real request row")
        # 原 dump plan 已按 request/logical page 排序。只接受连续满块，不把洞当作前缀。
        unique_rows = np.unique(pool_rows)
        for row in unique_rows:
            pages = logical_pages[pool_rows == row]
            start = int(self.layer_pages[layer, row])
            if not np.array_equal(pages, np.arange(start, start + pages.size)):
                raise RuntimeError(f"Store PUT prefix mismatch: layer={layer} row={row} next_page={start}")
        logger.info("DSA_STORE_PUT_BEGIN layer=%s pages=%s phase=%s", layer, pool_rows.size,
                    "prewrite" if before_cache_write else "post_attention")
        if self.direct_put is not None:
            self.direct_put.put_layer(layer, source_blocks, pool_rows, logical_pages)
        else:
            self.store.dump_packed_pages(
                layer, packed_hbm, torch.from_numpy(source_blocks), self.handles[rows],
                torch.from_numpy(logical_pages.astype(np.int32, copy=False)),
            )
        np.maximum.at(self.layer_pages[layer], pool_rows, logical_pages + 1)
        self.put_pages += int(pool_rows.size)
        logger.info("DSA_STORE_PUT_END layer=%s pages=%s bytes=%s", layer, pool_rows.size,
                    pool_rows.size * PAGE_TOKENS * TOKEN_BYTES)
        # Store 的 READY 前缀跨所有 payload 层取 min；第一层 PUT 后直接 WAIT 会死等后续层。
        prefix = self.layer_pages[:, unique_rows].min(axis=0)
        ready = prefix > self.watermarks.done.numpy()[unique_rows]
        if np.any(ready):
            done_rows = torch.from_numpy(unique_rows[ready].astype(np.int64, copy=False))
            targets = torch.from_numpy(prefix[ready].copy())
            result = self.store.wait_put_done(self.handles[done_rows], targets)
            if not torch.equal(result.handles, self.handles[done_rows]) or not torch.equal(result.full_blocks, targets):
                raise RuntimeError("Store PUT_DONE does not match this worker's handles/pages")
            self.watermarks.complete(done_rows, result)
            logger.info("DSA_STORE_PUT_DONE rows=%s pages=%s versions=%s bound=0",
                        done_rows.tolist(), result.full_blocks.tolist(), result.versions.tolist())

    def release(self, row: int) -> None:
        if row not in self.request_ids:
            return
        handle = self.handles[row:row + 1]
        # shadow 没有 DU lease，不伪造 DRAIN/APPLIED；同步 PUT 返回后已无在途读者。
        self.store.close_requests(handle)
        logger.info("DSA_STORE_CLOSE row=%s handle=%s done_pages=%s binding=none", row, int(handle[0]),
                    int(self.watermarks.done[row]))
        self.closed_requests += 1
        del self.request_ids[row]
        self.handles[row] = 0
        self.layer_pages[:, row] = 0
        for values in (self.watermarks.planned, self.watermarks.done, self.watermarks.bound,
                       self.watermarks.done_versions, self.watermarks.bound_versions):
            values[row] = 0

    def close(self) -> None:
        for row in tuple(self.request_ids):
            self.release(row)
        logger.info("DSA_STORE_CLOSED requests=%s put_pages=%s inference=%s remote_get=%s",
                    self.closed_requests, self.put_pages, self.inference, int(self.inference != "local_all"))
