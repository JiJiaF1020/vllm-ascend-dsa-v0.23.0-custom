# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.dsa_offload import kv_cache as dsa_kv_cache
from vllm_ascend.dsa_offload.kv_cache import (
    DSAIndexerKVSpec,
    DSAResidentMLAAttentionSpec,
    bind_dsa_full_kv_cache_views,
    build_dsa_kv_cache_config,
    build_dsa_kv_cache_groups,
    dsa_max_memory_usage_bytes,
    dsa_pool_bytes_per_base_block,
    get_dsa_full_kv_aliases,
    get_dsa_group_num_blocks,
    get_dsa_kv_cache_binding_order,
    get_dsa_kv_cache_group_ids,
    validate_dsa_kv_cache_config,
)
from vllm_ascend.dsa_offload.kv_cache_coordinator import (
    DSABlockPoolView,
    DSAKVCacheCoordinator,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_get_kv_cache_groups,
)


def _make_specs(
    num_layers: int = 2,
) -> dict[str, DSAIndexerKVSpec | DSAResidentMLAAttentionSpec]:
    specs: dict[
        str,
        DSAIndexerKVSpec | DSAResidentMLAAttentionSpec,
    ] = {}
    for layer_idx in range(num_layers):
        specs[f"model.layers.{layer_idx}.self_attn.indexer.k_cache"] = DSAIndexerKVSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        )
        specs[f"model.layers.{layer_idx}.self_attn.attn"] = DSAResidentMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            sparse_head_dim=(512, 64, 0),
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
        )
    return specs


def _make_vllm_config(
    *,
    max_model_len: int = 32768,
    override: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=override),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )


def _make_mtp_spec() -> AscendMLAAttentionSpec:
    return AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=704,
        sparse_head_dim=(512, 64, 128),
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
        cache_sparse_sfa_c8=False,
        cache_sparse_li_c8=False,
    )


def _make_remote_specs(
    num_layers: int = 5,
    full_layers: tuple[int, ...] = (0, 1, 2),
) -> dict[str, DSAIndexerKVSpec | DSAResidentMLAAttentionSpec]:
    specs = _make_specs(num_layers)
    for layer in range(num_layers):
        resident_name = f"model.layers.{layer}.self_attn.attn"
        specs[resident_name] = replace(
            specs[resident_name],
            head_size=656,
            sparse_head_dim=(656, 0, 0),
            cache_sparse_sfa_c8=True,
        )
        indexer_name = f"model.layers.{layer}.self_attn.indexer.k_cache"
        if layer in full_layers:
            specs[indexer_name] = replace(specs[indexer_name], cache_sparse_li_c8=True, full_kv_token_bytes=656)
        else:
            del specs[indexer_name]
    return specs


@pytest.mark.parametrize("full_layers", [(0,), (0, 1, 2), tuple(range(5))])
def test_remote_full_uses_only_indexer_owned_kv_storage(monkeypatch, full_layers) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 4)
    groups = build_dsa_kv_cache_groups(_make_remote_specs(full_layers=full_layers))
    aliases = get_dsa_full_kv_aliases(groups)
    assert aliases == {
        f"model.layers.{layer}.self_attn.attn": f"model.layers.{layer}.self_attn.indexer.k_cache"
        for layer in full_layers
    }
    physical_resident_layers = 5 - len(full_layers)
    expected_cost = 128 * (physical_resident_layers * 656 + len(full_layers) * 4 * (128 + 4 + 656))
    assert dsa_pool_bytes_per_base_block(groups) == expected_cost
    config = build_dsa_kv_cache_config(_make_vllm_config(override=128), groups, available_memory=1)
    assert len(config.kv_cache_tensors) == 5
    assert sum(t.size for t in config.kv_cache_tensors) == 128 * expected_cost
    assert get_dsa_group_num_blocks(config, groups[0]) == 512
    assert get_dsa_group_num_blocks(config, groups[1]) == 128
    for tensor in config.kv_cache_tensors:
        if any(name in aliases for name in tensor.shared_by):
            assert len(tensor.shared_by) == 2
            indexer, resident = tensor.shared_by
            assert aliases[resident] == indexer


def test_remote_full_aliases_keep_mtp_native_and_cross_rank_capacity(monkeypatch) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 4)
    specs = _make_remote_specs()
    specs["model.layers.5.self_attn.attn"] = _make_mtp_spec()
    groups = build_dsa_kv_cache_groups(specs)
    config = build_dsa_kv_cache_config(_make_vllm_config(override=256), groups, available_memory=1)
    assert config.kv_cache_tensors[-1].shared_by == ["model.layers.5.self_attn.attn"]
    assert get_dsa_group_num_blocks(config, groups[2]) == 1024
    config.num_blocks = 128
    for tensor in config.kv_cache_tensors:
        tensor.size //= 2
    validate_dsa_kv_cache_config(config)
    assert get_dsa_group_num_blocks(config, groups[0]) == 512
    assert get_dsa_group_num_blocks(config, groups[1]) == 128
    assert get_dsa_group_num_blocks(config, groups[2]) == 512


def test_remote_full_savings_match_78_layer_capacity_report(monkeypatch) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 4)
    specs = _make_remote_specs(num_layers=78, full_layers=tuple(range(21)))
    specs["model.layers.78.self_attn.attn"] = _make_mtp_spec()
    groups = build_dsa_kv_cache_groups(specs)
    old_cost = 128 * (78 * 656 + 21 * 4 * 788 + 4 * 1408)
    saved = 21 * 1024 * 128 * 656
    assert saved == 1_805_647_872
    assert 1024 * (old_cost - dsa_pool_bytes_per_base_block(groups)) == saved
    config = build_dsa_kv_cache_config(_make_vllm_config(), groups, available_memory=1024 * old_cost)
    assert config.num_blocks == 1152
    assert get_dsa_group_num_blocks(config, groups[0]) * 128 == 589824
    assert config.num_blocks * 128 // 12544 == 11
    assert dsa_max_memory_usage_bytes(_make_vllm_config(max_model_len=131072), groups) == (
        1024 * dsa_pool_bytes_per_base_block(groups)
    )


def test_remote_full_capacity_report_counts_only_physical_resident(monkeypatch) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 4)
    settings = SimpleNamespace(
        max_resident_budget_tokens=12288, sparse_tail_block_count=2, mtp_enabled=True, mtp_cache_layer_count=1
    )
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.get_ascend_config", lambda: SimpleNamespace(dsa_offload_config=settings)
    )
    logs = []
    monkeypatch.setattr(dsa_kv_cache.logger, "info_once", lambda fmt, *args: logs.append(fmt % args))
    specs = _make_remote_specs(num_layers=78, full_layers=tuple(range(21)))
    specs["model.layers.78.self_attn.attn"] = _make_mtp_spec()
    groups = build_dsa_kv_cache_groups(specs)
    config = build_dsa_kv_cache_config(_make_vllm_config(override=1024), groups, available_memory=1)
    vconfig = _make_vllm_config(max_model_len=40960)
    vconfig.scheduler_config = SimpleNamespace(max_num_seqs=7)
    dsa_kv_cache.report_dsa_kv_cache_config(vconfig, config)
    assert "indexer=21, resident=57, mtp=1" in logs[0]
    assert "21 aliases (no extra HBM; 78 logical MLA layers)" in logs[0]
    assert "1,805,647,872 bytes" in logs[0]
    assert "10 requests (12,544 resident slots/request)" in logs[0]
    assert "7 requests (max_num_seqs=7)" in logs[0]


def test_remote_full_raw_views_reuse_payload_without_copy_or_clear(monkeypatch) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 4)
    groups = build_dsa_kv_cache_groups(_make_remote_specs())
    config = build_dsa_kv_cache_config(_make_vllm_config(override=1), groups, available_memory=1)
    aliases = get_dsa_full_kv_aliases(groups)
    raw = {
        source: (
            torch.empty(4 * 128 * 128, dtype=torch.int8),
            torch.empty(4 * 128 * 4, dtype=torch.int8),
            torch.full((4 * 128 * 656,), 42, dtype=torch.int8),
        )
        for source in aliases.values()
    }
    for layer in (3, 4):
        raw[f"model.layers.{layer}.self_attn.attn"] = (torch.full((128 * 656,), 17, dtype=torch.int8),)
    shared = raw["model.layers.3.self_attn.attn"][0]
    bind_dsa_full_kv_cache_views(config, raw)
    for name, source in aliases.items():
        view = raw[name][0]
        payload = raw[source][2]
        assert view.numel() == 128 * 656
        assert view.data_ptr() == payload.data_ptr()
        assert view.untyped_storage().data_ptr() == payload.untyped_storage().data_ptr()
        assert torch.all(view == 42)
        view.fill_(9)
        assert torch.all(payload[: view.numel()] == 9)
        assert torch.all(payload[view.numel() :] == 42)
    assert raw["model.layers.3.self_attn.attn"][0] is shared
    assert torch.all(shared == 17)


def test_remote_full_rejects_independent_resident_and_missing_payload(monkeypatch) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 4)
    groups = build_dsa_kv_cache_groups(_make_remote_specs())
    config = build_dsa_kv_cache_config(_make_vllm_config(override=1), groups, available_memory=1)
    name, source = next(iter(get_dsa_full_kv_aliases(groups).items()))
    with pytest.raises(RuntimeError, match="independent resident storage"):
        bind_dsa_full_kv_cache_views(config, {name: (torch.empty(1, dtype=torch.int8),)})
    with pytest.raises(RuntimeError, match="key/scale/full payload"):
        bind_dsa_full_kv_cache_views(config, {source: (torch.empty(1, dtype=torch.int8),)})
    tensor = next(tensor for tensor in config.kv_cache_tensors if name in tensor.shared_by)
    tensor.shared_by.remove(source)
    with pytest.raises(RuntimeError, match="share its Indexer allocation"):
        get_dsa_group_num_blocks(config, groups[1])


def test_remote_full_rejects_incompatible_resident_layout() -> None:
    groups = build_dsa_kv_cache_groups(_make_remote_specs())
    groups[1] = replace(groups[1], kv_cache_spec=replace(groups[1].kv_cache_spec, sparse_head_dim=(512, 64, 0)))
    with pytest.raises(RuntimeError, match="matching packed-C8 layouts"):
        get_dsa_full_kv_aliases(groups)


def test_local_full_keeps_independent_resident_storage(monkeypatch) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 4)
    specs = _make_remote_specs()
    for name, spec in tuple(specs.items()):
        if isinstance(spec, DSAIndexerKVSpec):
            specs[name] = replace(spec, full_kv_token_bytes=0)
    groups = build_dsa_kv_cache_groups(specs)
    assert not get_dsa_full_kv_aliases(groups)
    config = build_dsa_kv_cache_config(_make_vllm_config(override=128), groups, available_memory=1)
    assert len(config.kv_cache_tensors) == 8
    assert all(len(tensor.shared_by) == 1 for tensor in config.kv_cache_tensors)
    assert dsa_pool_bytes_per_base_block(groups) == 128 * (5 * 656 + 3 * 4 * 132)
    raw = {}
    bind_dsa_full_kv_cache_views(config, raw)
    assert raw == {}


def test_indexer_spec_accounts_for_one_vector_per_token() -> None:
    spec = next(spec for spec in _make_specs(num_layers=1).values() if isinstance(spec, DSAIndexerKVSpec))

    assert spec.page_size_bytes == 128 * 1 * 128 * 2


@pytest.mark.parametrize("full_kv_token_bytes", [0, 656])
def test_c8_indexer_spec_merge_preserves_split_cache_layout(full_kv_token_bytes: int) -> None:
    specs = [
        DSAIndexerKVSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_sparse_li_c8=True,
            full_kv_token_bytes=full_kv_token_bytes,
        )
        for _ in range(2)
    ]

    merged = DSAIndexerKVSpec.merge(specs)

    assert merged.cache_sparse_li_c8
    assert merged.c8_k_cache_dtype == torch.float8_e4m3fn
    assert merged.c8_k_scale_cache_dtype == torch.float32
    assert merged.full_kv_token_bytes == full_kv_token_bytes
    assert merged.page_size_bytes == 128 * (128 + 4 + full_kv_token_bytes)


def test_c8_resident_spec_merge_preserves_packed_cache_layout() -> None:
    specs = [
        DSAResidentMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=656,
            sparse_head_dim=(656, 0, 0),
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
            cache_sparse_sfa_c8=True,
            cache_sparse_li_c8=False,
        )
        for _ in range(2)
    ]

    merged = DSAResidentMLAAttentionSpec.merge(specs)

    assert type(merged) is DSAResidentMLAAttentionSpec
    assert merged.cache_sparse_sfa_c8
    assert not merged.cache_sparse_li_c8
    assert merged.sparse_head_dim == (656, 0, 0)
    assert merged.page_size_bytes == 128 * 656


def test_split_groups_keep_stable_plane_order() -> None:
    groups = build_dsa_kv_cache_groups(_make_specs())

    assert len(groups) == 2
    assert isinstance(groups[0].kv_cache_spec, DSAIndexerKVSpec)
    assert isinstance(
        groups[1].kv_cache_spec,
        DSAResidentMLAAttentionSpec,
    )
    assert len(groups[0].layer_names) == len(groups[1].layer_names) == 2


def test_mtp_draft_layer_forms_native_bf16_full_cache_group() -> None:
    specs = _make_specs(num_layers=2)
    specs["model.layers.78.self_attn.attn"] = _make_mtp_spec()

    groups = build_dsa_kv_cache_groups(specs)

    assert len(groups) == 3
    assert not any("layers.78" in name for name in groups[0].layer_names)
    assert not any("layers.78" in name for name in groups[1].layer_names)
    assert groups[2].layer_names == ["model.layers.78.self_attn.attn"]
    assert type(groups[2].kv_cache_spec) is AscendMLAAttentionSpec
    assert groups[2].kv_cache_spec.dtype == torch.bfloat16
    assert groups[2].kv_cache_spec.sparse_head_dim == (512, 64, 128)
    assert groups[2].is_eagle_group


@pytest.mark.parametrize(
    "mtp_spec",
    [
        replace(_make_mtp_spec(), cache_sparse_li_c8=True),
        replace(
            _make_mtp_spec(),
            head_size=576,
            sparse_head_dim=(512, 64, 0),
        ),
        replace(_make_mtp_spec(), compress_ratio=2),
        replace(_make_mtp_spec(), cache_dtype_str="fp8_ds_mla"),
    ],
)
def test_mtp_full_group_rejects_non_native_cache_layout(
    mtp_spec: AscendMLAAttentionSpec,
) -> None:
    specs = _make_specs(num_layers=1)
    specs["model.layers.78.self_attn.attn"] = mtp_spec

    with pytest.raises(RuntimeError, match="three-entry BF16"):
        build_dsa_kv_cache_groups(specs)


def test_engine_core_grouping_initializes_process_local_ascend_config(
    monkeypatch,
) -> None:
    initialized_with: list[object] = []
    vllm_config = object()
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.init_ascend_config",
        initialized_with.append,
    )

    groups = _ascend_get_kv_cache_groups(  # type: ignore[arg-type]
        vllm_config,
        _make_specs(),
    )

    assert initialized_with == [vllm_config]
    assert len(groups) == 2


def test_split_group_ids_follow_spec_identity_not_position() -> None:
    specs = _make_specs()
    specs["model.layers.78.self_attn.attn"] = _make_mtp_spec()
    groups = list(reversed(build_dsa_kv_cache_groups(specs)))
    config = SimpleNamespace(kv_cache_groups=groups)

    group_ids = get_dsa_kv_cache_group_ids(config)  # type: ignore[arg-type]

    assert group_ids.indexer == 2
    assert group_ids.resident_mla == 1
    assert group_ids.mtp_full == 0


def test_split_cache_binding_orders_each_layer_resident_then_indexer() -> None:
    groups = build_dsa_kv_cache_groups(_make_specs())
    config = SimpleNamespace(kv_cache_groups=groups)

    assert get_dsa_kv_cache_binding_order(config) == [
        "model.layers.0.self_attn.attn",
        "model.layers.0.self_attn.indexer.k_cache",
        "model.layers.1.self_attn.attn",
        "model.layers.1.self_attn.indexer.k_cache",
    ]


def test_split_cache_binding_leaves_mtp_full_group_to_baseline() -> None:
    specs = _make_specs(num_layers=1)
    specs["model.layers.78.self_attn.attn"] = _make_mtp_spec()
    groups = build_dsa_kv_cache_groups(specs)
    config = SimpleNamespace(kv_cache_groups=groups)

    assert get_dsa_kv_cache_binding_order(config) == [
        "model.layers.0.self_attn.attn",
        "model.layers.0.self_attn.indexer.k_cache",
    ]


def test_split_cache_binding_rejects_duplicate_layer_name() -> None:
    groups = build_dsa_kv_cache_groups(_make_specs(num_layers=1))
    groups[1].layer_names.append(groups[0].layer_names[0])
    config = SimpleNamespace(kv_cache_groups=groups)

    with pytest.raises(RuntimeError, match="more than one group"):
        get_dsa_kv_cache_binding_order(config)


def test_ratio_is_expressed_by_final_tensor_sizes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())
    bytes_per_base_block = dsa_pool_bytes_per_base_block(groups)

    config = build_dsa_kv_cache_config(
        _make_vllm_config(),
        groups,
        available_memory=bytes_per_base_block * 256,
    )

    assert config.num_blocks == 256
    assert get_dsa_group_num_blocks(config, groups[0]) == 768
    assert get_dsa_group_num_blocks(config, groups[1]) == 256
    assert not hasattr(groups[0], "dsa_num_blocks")
    assert not hasattr(groups[1], "dsa_num_blocks")


def test_mtp_full_group_uses_dense_three_x_capacity(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    specs = _make_specs()
    specs["model.layers.78.self_attn.attn"] = _make_mtp_spec()
    groups = build_dsa_kv_cache_groups(specs)
    bytes_per_base_block = dsa_pool_bytes_per_base_block(groups)

    config = build_dsa_kv_cache_config(
        _make_vllm_config(),
        groups,
        available_memory=bytes_per_base_block * 256,
    )

    group_ids = get_dsa_kv_cache_group_ids(config)
    assert group_ids.indexer == 0
    assert group_ids.resident_mla == 1
    assert group_ids.mtp_full == 2
    assert config.num_blocks == 256
    assert get_dsa_group_num_blocks(config, groups[0]) == 768
    assert get_dsa_group_num_blocks(config, groups[1]) == 256
    assert get_dsa_group_num_blocks(config, groups[2]) == 768


def test_mtp_full_group_requires_target_indexer_token_coverage(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    specs = _make_specs()
    specs["model.layers.78.self_attn.attn"] = replace(
        _make_mtp_spec(),
        block_size=64,
    )
    groups = build_dsa_kv_cache_groups(specs)

    with pytest.raises(RuntimeError, match="same token range"):
        build_dsa_kv_cache_config(
            _make_vllm_config(override=256),
            groups,
            available_memory=1,
        )


def test_num_gpu_blocks_override_keeps_base_block_semantics(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())

    config = build_dsa_kv_cache_config(
        _make_vllm_config(override=320),
        groups,
        available_memory=dsa_pool_bytes_per_base_block(groups) * 256,
    )

    assert config.num_blocks == 320
    assert get_dsa_group_num_blocks(config, groups[0]) == 960
    assert get_dsa_group_num_blocks(config, groups[1]) == 320


def test_cross_rank_base_capacity_shrink_preserves_plane_ratio(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    specs = _make_specs()
    specs["model.layers.78.self_attn.attn"] = _make_mtp_spec()
    groups = build_dsa_kv_cache_groups(specs)
    config = build_dsa_kv_cache_config(
        _make_vllm_config(),
        groups,
        available_memory=dsa_pool_bytes_per_base_block(groups) * 256,
    )

    old_base_blocks = config.num_blocks
    config.num_blocks = 128
    for tensor in config.kv_cache_tensors:
        tensor.size = tensor.size // old_base_blocks * config.num_blocks

    validate_dsa_kv_cache_config(config)
    assert get_dsa_group_num_blocks(config, groups[0]) == 384
    assert get_dsa_group_num_blocks(config, groups[1]) == 128
    assert get_dsa_group_num_blocks(config, groups[2]) == 384


def test_prefill_admission_uses_weighted_base_block_cost(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())
    vllm_config = _make_vllm_config(max_model_len=2048)

    assert dsa_max_memory_usage_bytes(
        vllm_config,
        groups,
    ) == 16 * dsa_pool_bytes_per_base_block(groups)


def test_split_groups_accept_indexer_subset_for_shared_topology() -> None:
    specs = _make_specs(num_layers=2)
    del specs["model.layers.1.self_attn.indexer.k_cache"]

    # GLM-5.2 共享 indexer 拓扑下 indexer 层是 resident 层的真子集，
    # 不再要求每层都有独立 indexer，分组应通过。
    groups = build_dsa_kv_cache_groups(specs)
    assert len(groups) == 2
    assert len(groups[0].layer_names) == 1  # 仅 layer 0 有 indexer
    assert len(groups[1].layer_names) == 2


def test_split_groups_reject_orphan_indexer_without_resident() -> None:
    # indexer 指向一个不存在 resident 层的 transformer 下标仍属硬错误。
    specs = _make_specs(num_layers=1)
    del specs["model.layers.0.self_attn.indexer.k_cache"]
    orphan = DSAIndexerKVSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    specs["model.layers.7.self_attn.indexer.k_cache"] = orphan

    with pytest.raises(RuntimeError, match="no matching resident MLA layer"):
        build_dsa_kv_cache_groups(specs)


def test_split_groups_reject_more_indexer_than_resident() -> None:
    specs = _make_specs(num_layers=1)
    specs["model.layers.1.self_attn.indexer.k_cache"] = DSAIndexerKVSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    # 保留 layer 0 resident，使入口先命中明确的数量合同。

    with pytest.raises(RuntimeError, match="more Indexer caches than resident"):
        build_dsa_kv_cache_groups(specs)


def test_block_pool_view_resets_every_physical_pool() -> None:
    class _Pool:
        num_gpu_blocks = 4

        def __init__(self, result: bool) -> None:
            self.result = result
            self.reset_calls = 0

        def reset_prefix_cache(self) -> bool:
            self.reset_calls += 1
            return self.result

    first = _Pool(False)
    second = _Pool(True)
    view = DSABlockPoolView([first, second])  # type: ignore[list-item]

    assert view.reset_prefix_cache() is False
    assert first.reset_calls == 1
    assert second.reset_calls == 1


def test_component_admission_checks_each_physical_pool() -> None:
    class _Pool:
        def __init__(self, free_blocks: int) -> None:
            self.free_blocks = free_blocks

        def get_num_free_blocks(self) -> int:
            return self.free_blocks

    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.physical_block_pools = (  # type: ignore[attr-defined]
        _Pool(6),
        _Pool(2),
        _Pool(5),
    )

    assert coordinator.can_allocate((6, 2, 5))
    assert not coordinator.can_allocate((7, 1, 1))
    assert not coordinator.can_allocate((1, 3, 1))
    assert not coordinator.can_allocate((1, 1, 6))
    with pytest.raises(RuntimeError, match="reservations"):
        coordinator.can_allocate((1, 1, 1), reserved_blocks=1)
