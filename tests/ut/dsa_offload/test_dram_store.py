# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm_ascend.dsa_offload.dram_store import (
    DSAHotDRAMStore,
    calculate_dram_usable_blocks,
)


def _cpu_arena(
    block_shape: tuple[int, ...],
    dtype: torch.dtype,
    capacity: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.zeros(
        (capacity, *block_shape),
        dtype=dtype,
        device=device,
    )


def _make_store(*, metadata_only: bool = False) -> DSAHotDRAMStore:
    return DSAHotDRAMStore(
        usable_blocks=8,
        storage_rows=3,
        max_logical_blocks=6,
        device=torch.device("cpu"),
        arena_factory=_cpu_arena,
        metadata_only=metadata_only,
    )


def test_dram_capacity_uses_fractional_multiplier_with_ceil() -> None:
    assert calculate_dram_usable_blocks(7, 1.5) == 11


def test_layer_arenas_follow_resident_cache_block_shapes() -> None:
    store = _make_store()
    resident_nope = torch.empty(4, 128, 1, 512)
    resident_rope = torch.empty(4, 128, 1, 64)

    store.add_layer(
        layer_id=0,
        resident_nope_cache=resident_nope,
        resident_rope_cache=resident_rope,
    )
    arenas = store.get_layer_arenas(0)

    assert arenas.nope.shape == (9, 128, 1, 512)
    assert arenas.rope.shape == (9, 128, 1, 64)
    assert torch.count_nonzero(arenas.nope[0]).item() == 0
    assert torch.count_nonzero(arenas.rope[0]).item() == 0


def test_packed_c8_layer_uses_an_opaque_byte_arena() -> None:
    store = _make_store()
    resident_packed = torch.empty(4, 128, 1, 656, dtype=torch.int8)

    store.add_packed_layer(
        layer_id=0,
        resident_packed_cache=resident_packed,
    )
    arenas = store.get_layer_arenas(0)

    assert arenas.nope is None
    assert arenas.rope is None
    assert arenas.packed is not None
    assert arenas.packed.shape == (9, 128, 1, 656)
    assert arenas.packed.dtype == torch.int8
    assert torch.count_nonzero(arenas.packed[0]).item() == 0


def test_metadata_only_store_forbids_local_payload_arenas() -> None:
    # 不提供 arena_factory；远端元数据路径不应导入 torch_npu 或创建 payload。
    store = DSAHotDRAMStore(
        usable_blocks=8,
        storage_rows=3,
        max_logical_blocks=6,
        device=torch.device("cpu"),
        metadata_only=True,
    )
    assert store.metadata_only
    packed = torch.empty(4, 128, 1, 656, dtype=torch.int8)
    nope = torch.empty(4, 128, 1, 512)
    rope = torch.empty(4, 128, 1, 64)
    with pytest.raises(RuntimeError, match="metadata-only store has no local payload"):
        store.add_packed_layer(layer_id=0, resident_packed_cache=packed)
    with pytest.raises(RuntimeError, match="metadata-only store has no local payload"):
        store.add_layer(layer_id=0, resident_nope_cache=nope, resident_rope_cache=rope)
    with pytest.raises(RuntimeError, match="metadata-only store has no local payload"):
        store.get_layer_arenas(0)


@pytest.mark.parametrize("metadata_only", (False, True))
def test_reservation_is_idempotent_and_release_reclaims_whole_row(metadata_only: bool) -> None:
    store = _make_store(metadata_only=metadata_only)
    pool_rows = np.array([0, 0, 1], dtype=np.intp)
    logical_blocks = np.array([0, 2, 0], dtype=np.intp)

    first = store.reserve_blocks(
        pool_indices=pool_rows,
        logical_block_indices=logical_blocks,
    )
    assert first.new_mask.tolist() == [True, True, True]
    assert np.all(first.physical_block_ids > 0)
    assert store.num_free_blocks == 5

    second = store.reserve_blocks(
        pool_indices=pool_rows,
        logical_block_indices=logical_blocks,
    )
    assert second.new_mask.tolist() == [False, False, False]
    assert second.physical_block_ids.tolist() == (
        first.physical_block_ids.tolist()
    )

    active = np.empty((2, 6), dtype=np.int32)
    store.gather_rows(
        pool_indices=np.array([1, 0], dtype=np.intp),
        output=active,
    )
    assert active[0, 0] == first.physical_block_ids[2]
    assert active[1, 0] == first.physical_block_ids[0]
    assert active[1, 2] == first.physical_block_ids[1]

    store.commit_durable_full_block_counts(
        pool_indices=np.array([0, 1], dtype=np.intp),
        full_block_counts=np.array([3, 1], dtype=np.int32),
    )
    durable = np.empty(2, dtype=np.int32)
    store.gather_durable_full_block_counts(
        pool_indices=np.array([1, 0], dtype=np.intp),
        output=durable,
    )
    assert durable.tolist() == [1, 3]

    store.release_pool_index(0)
    assert store.num_free_blocks == 7
    assert np.count_nonzero(store.logical_block_table[0]) == 0
    assert store.durable_full_block_counts[0] == 0
    assert store.logical_block_table[1, 0] != 0
