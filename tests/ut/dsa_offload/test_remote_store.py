# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm_ascend.dsa_offload.remote_store import PutDoneBatch, RemotePackedArena, RemoteWatermarks


def test_remote_arena_has_no_dereferenceable_pointer():
    arena = RemotePackedArena((2, 8, 64, 128, 1, 656))
    assert arena.dtype == torch.uint8
    with pytest.raises(RuntimeError, match="no local data pointer"):
        arena.data_ptr()


def test_remote_watermarks_do_not_publish_before_ack():
    state = RemoteWatermarks(4)
    rows = torch.tensor([1, 3])
    pages = torch.tensor([4, 8], dtype=torch.int32)
    versions = torch.tensor([1, 2], dtype=torch.int64)
    state.plan(rows, pages)
    assert state.done.sum() == state.bound.sum() == 0
    with pytest.raises(ValueError, match="PUT_DONE"):
        state.acknowledge_binding(rows, versions, pages)
    state.complete(rows, PutDoneBatch(torch.tensor([41, 87]), versions, pages))
    assert state.bound.sum() == 0
    state.acknowledge_binding(rows, versions, pages)
    assert state.bound.tolist() == [0, 4, 0, 8]
    state.plan(rows, pages + 1)
    assert torch.equal(state.bound[rows], pages)
    with pytest.raises(ValueError, match="decrease"):
        state.plan(rows, pages - 1)
