# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_ascend.dsa_offload.graph_gate import (
    DSARowModeGraphDecision,
    evaluate_dsa_row_mode_decode_graph,
    evaluate_remote_decode_graph,
)
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)


def _state(*stages: DSARequestCacheStage):
    return SimpleNamespace(
        valid=True,
        row_count=len(stages),
        stages_cpu=np.asarray(stages, dtype=np.int32),
    )


@pytest.mark.parametrize(
    "stages",
    [
        (DSARequestCacheStage.DENSE_DECODE,),
        (DSARequestCacheStage.ENTER_SPARSE_DECODE,),
        (DSARequestCacheStage.SPARSE_DECODE,),
        (
            DSARequestCacheStage.DENSE_DECODE,
            DSARequestCacheStage.ENTER_SPARSE_DECODE,
            DSARequestCacheStage.SPARSE_DECODE,
        ),
    ],
)
def test_all_uniform_decode_stages_share_full_graph(stages) -> None:
    decision = evaluate_dsa_row_mode_decode_graph(
        state=_state(*stages),
        num_reqs=len(stages),
        total_num_scheduled_tokens=len(stages),
        max_num_scheduled_tokens=1,
        uniform_decode_query_len=1,
        max_capture_size=8,
    )

    assert decision.use_full_graph
    assert not decision.is_expected_eager


@pytest.mark.parametrize(
    ("state", "num_reqs", "total_tokens", "max_tokens", "capture", "reason"),
    [
        (
            _state(DSARequestCacheStage.PREFILL),
            1,
            1,
            1,
            8,
            "non_decode_stage",
        ),
        (
            _state(DSARequestCacheStage.DENSE_DECODE),
            1,
            2,
            2,
            8,
            "non_uniform_decode",
        ),
        (
            _state(
                DSARequestCacheStage.DENSE_DECODE,
                DSARequestCacheStage.SPARSE_DECODE,
            ),
            2,
            2,
            1,
            1,
            "capture_size_miss",
        ),
    ],
)
def test_normal_non_graph_phases_are_expected_eager(
    state,
    num_reqs: int,
    total_tokens: int,
    max_tokens: int,
    capture: int,
    reason: str,
) -> None:
    decision = evaluate_dsa_row_mode_decode_graph(
        state=state,
        num_reqs=num_reqs,
        total_num_scheduled_tokens=total_tokens,
        max_num_scheduled_tokens=max_tokens,
        uniform_decode_query_len=1,
        max_capture_size=capture,
    )

    assert not decision.use_full_graph
    assert decision.is_expected_eager
    assert decision.reason == reason


def test_stale_input_batch_state_is_not_a_fallback() -> None:
    state = _state(DSARequestCacheStage.DENSE_DECODE)
    state.row_count = 2

    decision = evaluate_dsa_row_mode_decode_graph(
        state=state,
        num_reqs=1,
        total_num_scheduled_tokens=1,
        max_num_scheduled_tokens=1,
        uniform_decode_query_len=1,
        max_capture_size=8,
    )

    assert not decision.use_full_graph
    assert not decision.is_expected_eager
    assert decision.reason == "input_batch_row_mismatch"


def test_enabled_graph_without_capture_sizes_is_not_a_fallback() -> None:
    decision = evaluate_dsa_row_mode_decode_graph(
        state=_state(DSARequestCacheStage.DENSE_DECODE),
        num_reqs=1,
        total_num_scheduled_tokens=1,
        max_num_scheduled_tokens=1,
        uniform_decode_query_len=1,
        max_capture_size=0,
    )

    assert not decision.use_full_graph
    assert not decision.is_expected_eager
    assert decision.reason == "missing_capture_sizes"


def test_uniform_mtp3_decode_uses_the_same_full_graph_gate() -> None:
    decision = evaluate_dsa_row_mode_decode_graph(
        state=_state(
            DSARequestCacheStage.DENSE_DECODE,
            DSARequestCacheStage.SPARSE_DECODE,
        ),
        num_reqs=2,
        total_num_scheduled_tokens=8,
        max_num_scheduled_tokens=4,
        uniform_decode_query_len=4,
        max_capture_size=8,
    )

    assert decision.use_full_graph


@pytest.mark.parametrize("batch", [1, 8, 16, 32])
def test_remote_mtp3_exact_batch_requires_real_warmup(batch) -> None:
    decision = evaluate_dsa_row_mode_decode_graph(
        state=_state(*([DSARequestCacheStage.SPARSE_DECODE] * batch)),
        num_reqs=batch, total_num_scheduled_tokens=batch * 4,
        max_num_scheduled_tokens=4, uniform_decode_query_len=4, max_capture_size=32,
    )
    sizes = [4, 32, 64, 128]
    warmup = evaluate_remote_decode_graph(decision, num_tokens=batch * 4, capture_sizes=sizes, warmed=False)
    assert warmup.reason == "remote_decode_warmup" and warmup.is_expected_eager
    assert evaluate_remote_decode_graph(decision, num_tokens=batch * 4, capture_sizes=sizes, warmed=True).use_full_graph


@pytest.mark.parametrize("tokens", [4, 12, 20, 24, 28, 32, 68, 124, 128])
def test_remote_allows_padding_after_real_batch_warmup(tokens) -> None:
    eligible = DSARowModeGraphDecision(True, "allow_row_mode_decode")
    assert evaluate_remote_decode_graph(eligible, num_tokens=tokens, capture_sizes=[32, 64, 128], warmed=True) is eligible
    result = evaluate_remote_decode_graph(eligible, num_tokens=tokens, capture_sizes=[32, 64, 128], warmed=False)
    assert result.reason == "remote_decode_warmup"


def test_remote_padding_does_not_override_prefill_or_capacity_gate() -> None:
    eligible = DSARowModeGraphDecision(True, "allow_row_mode_decode")
    result = evaluate_remote_decode_graph(eligible, num_tokens=132, capture_sizes=[32, 64, 128], warmed=True)
    assert result.reason == "capture_size_miss" and result.is_expected_eager
    prefill = DSARowModeGraphDecision(False, "non_decode_stage")
    assert evaluate_remote_decode_graph(prefill, num_tokens=128, capture_sizes=[128], warmed=True) is prefill


def test_mixed_mtp_query_lengths_fall_back_to_true_eager() -> None:
    decision = evaluate_dsa_row_mode_decode_graph(
        state=_state(
            DSARequestCacheStage.DENSE_DECODE,
            DSARequestCacheStage.SPARSE_DECODE,
        ),
        num_reqs=2,
        total_num_scheduled_tokens=5,
        max_num_scheduled_tokens=4,
        uniform_decode_query_len=4,
        max_capture_size=8,
    )

    assert not decision.use_full_graph
    assert decision.is_expected_eager
    assert decision.reason == "non_uniform_decode"
