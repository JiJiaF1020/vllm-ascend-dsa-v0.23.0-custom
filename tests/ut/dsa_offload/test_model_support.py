# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm_ascend.dsa_offload.execution_plan import (
    DsaKvPlacement, LayerExecutionPath, SelectionDomain, build_execution_plan,
)
from vllm_ascend.dsa_offload.model_support import (
    get_dsa_offload_model_capabilities,
    require_dsa_offload_model_support,
)


def test_hbm_full_remote_shared_execution_plan():
    plan = build_execution_plan(["full", "full", "full", "shared", "shared"], 5,
                                DsaKvPlacement.HBM_FULL_REMOTE_SHARED)
    assert [layer.ops for layer in plan.layers] == [
        ("li", "sfa_native"), ("li", "sfa_native"), ("lis_send", "sfa_native"),
        ("scatter_sfa",), ("scatter_sfa", "applied"),
    ]
    assert [(c.submit_layer, c.payload_layer, c.action, c.plan_command) for c in plan.commands] == [
        (2, 3, "DU_AND_GATHER", 0), (2, 4, "AUTO_GATHER", 0),
    ]
    assert [layer.mirror_writer for layer in plan.layers] == ["none"] * 4 + ["scatter_sfa"]
    assert plan.groups[2].mirror_commit_layer == 4
    assert plan.layers[3].send_command is None  # 首层 payload 已由 full 提前请求。
    assert plan.layers[4].send_command is None  # 后续payload也由Full触发，不再逐层发请求。
    assert "full_triggered_group_gather" in plan.required_capabilities
    assert [layer.path for layer in plan.layers] == [
        LayerExecutionPath.NATIVE_FULL, LayerExecutionPath.NATIVE_FULL,
        LayerExecutionPath.LIS_SEND_FULL, LayerExecutionPath.REMOTE_SHARED,
        LayerExecutionPath.REMOTE_SHARED,
    ]
    assert [layer.selection_domain for layer in plan.layers] == [
        SelectionDomain.VISIBLE_CAUSAL, SelectionDomain.VISIBLE_CAUSAL,
        SelectionDomain.DURABLE_WITH_CAUSAL_TAIL, SelectionDomain.SOURCE_PLAN,
        SelectionDomain.SOURCE_PLAN,
    ]
    plan.validate_binding((0, 1, 2), (0, 1, 2, 3, 4))
    with pytest.raises(RuntimeError, match="refusing silent local fallback"):
        plan.require_capabilities(set())
    plan.require_capabilities(set(plan.required_capabilities))


def test_local_full_remote_shared_keeps_du_on_npu():
    plan = build_execution_plan(["full", "full", "full", "shared", "shared"], 5,
                                DsaKvPlacement.LOCAL_FULL_REMOTE_SHARED)
    assert [layer.ops for layer in plan.layers] == [
        ("lidu", "ksc", "sfa"), ("lidu", "ksc", "sfa"),
        ("lidu", "miss_plan_send", "ksc", "sfa"),
        ("scatter_sfa",), ("gather_only", "scatter_sfa", "applied"),
    ]
    assert plan.commands[0].action == "MISS_PLAN_AND_GATHER"
    assert all(layer.du_owner == "npu" for layer in plan.layers)
    assert [layer.mirror_writer for layer in plan.layers] == ["lidu"] * 3 + ["none"] * 2
    assert "pool_du" not in plan.required_capabilities
    assert plan.groups[2].mirror_commit_layer == 2
    assert plan.groups[2].release_layer == 4


@pytest.mark.parametrize("placement", list(DsaKvPlacement))
def test_execution_plan_topology_and_generation(placement):
    roles = ["full", "shared", "full", "shared", "shared", "full"]
    plan = build_execution_plan(roles, len(roles), placement)
    assert [layer.source_layer for layer in plan.layers] == [0, 0, 2, 2, 2, 5]
    assert [group.layers for group in plan.groups] == [(0, 1), (2, 3, 4), (5,)]
    generations = [plan.generation(step, command.index, group, 2)
                   for step in range(4) for command in plan.commands for group in range(2)]
    assert generations == list(range(1, len(generations) + 1))
    truncated = build_execution_plan(roles[:4], 4, placement)
    assert truncated.groups[1].release_layer == 3
    assert plan.groups[-1].remote_layers == ()
    with pytest.raises(ValueError, match="binding mismatch"):
        plan.validate_binding((0, 1, 2), tuple(range(6)))


def test_local_execution_plan_preserves_baseline_and_rejects_invalid_topology():
    plan = build_execution_plan(None, 3)
    plan.require_capabilities(set())
    assert not plan.commands
    assert all(layer.ops == ("lidu", "ksc", "sfa") for layer in plan.layers)
    for roles, count in [(["shared"], 1), (["full"], 2), (["full", "typo"], 2), (None, 0)]:
        with pytest.raises(ValueError):
            build_execution_plan(roles, count)

def _model_config(
    architecture: str,
    *,
    use_mla: bool = True,
    index_topk: int = 2048,
    compress_ratios=None,
):
    hf_text_config = SimpleNamespace(
        index_topk=index_topk,
        index_head_dim=128,
        index_n_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    hf_config = SimpleNamespace()
    if compress_ratios is not None:
        hf_text_config.compress_ratios = compress_ratios
        hf_config.compress_ratios = compress_ratios
    return SimpleNamespace(
        architecture=architecture,
        use_mla=use_mla,
        hf_text_config=hf_text_config,
        hf_config=hf_config,
    )


@pytest.mark.parametrize(
    "architecture",
    ["DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"],
)
def test_model_support_is_capability_based(architecture: str) -> None:
    capabilities = get_dsa_offload_model_capabilities(_model_config(architecture))

    assert capabilities.supported
    assert capabilities.architecture == architecture


def test_unknown_architecture_with_required_capabilities_is_supported() -> None:
    capabilities = get_dsa_offload_model_capabilities(_model_config("FutureMlaIndexerForCausalLM"))

    assert capabilities.supported


@pytest.mark.parametrize(
    ("model_config", "missing"),
    [
        (_model_config("NoMla", use_mla=False), "MLA attention"),
        (_model_config("WrongTopK", index_topk=1024), "index_topk=2048"),
        (
            _model_config("CompressedDSA", compress_ratios=[2]),
            "sparse indexer",
        ),
    ],
)
def test_missing_capability_is_reported(model_config, missing: str) -> None:
    capabilities = get_dsa_offload_model_capabilities(model_config)

    assert not capabilities.supported
    assert any(missing in item for item in capabilities.missing_requirements)
    with pytest.raises(ValueError, match=missing):
        require_dsa_offload_model_support(model_config)


def _glm52_model_config() -> SimpleNamespace:
    # 真实 GLM-5.2 拓扑（zai-org/GLM-5.2 config.json）：78 层 = 21 full + 57
    # shared，full 在 0,1,2 后每隔 4 层一个（6,10,...,74）。
    full_layers = {0, 1, 2, *range(6, 78, 4)}
    assert len(full_layers) == 21
    indexer_types = ["full" if i in full_layers else "shared" for i in range(78)]
    hf_text_config = SimpleNamespace(
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        indexer_types=indexer_types,
        index_topk_freq=4,
        num_hidden_layers=78,
    )
    return SimpleNamespace(
        architecture="GlmMoeDsaForCausalLM",
        use_mla=True,
        hf_text_config=hf_text_config,
        hf_config=SimpleNamespace(),
    )


def test_shared_indexer_topology_is_parsed() -> None:
    capabilities = get_dsa_offload_model_capabilities(_glm52_model_config())

    assert capabilities.supported  # 拓扑不影响 supported 判定
    assert capabilities.has_shared_indexer_layers
    full = capabilities.full_indexer_layer_indices
    shared = capabilities.shared_indexer_layer_indices
    assert full is not None and shared is not None
    assert len(full) == 21
    assert len(shared) == 57
    assert full[:3] == (0, 1, 2)
    assert 6 in full and 74 in full and 3 not in full
    assert len(full) + len(shared) == 78


def test_all_full_topology_is_default_and_not_shared() -> None:
    capabilities = get_dsa_offload_model_capabilities(_model_config("DeepseekV32ForCausalLM"))

    assert capabilities.supported
    assert not capabilities.has_shared_indexer_layers
    assert capabilities.full_indexer_layer_indices is None
    assert capabilities.shared_indexer_layer_indices is None


def test_malformed_indexer_types_is_rejected() -> None:
    config = _model_config("GlmMoeDsaForCausalLM")
    config.hf_text_config.indexer_types = ["full", 128, "shared"]  # 非全字符串

    with pytest.raises(ValueError, match="entries must be strings"):
        get_dsa_offload_model_capabilities(config)


def test_unknown_indexer_type_is_rejected() -> None:
    config = _model_config("GlmMoeDsaForCausalLM")
    config.hf_text_config.indexer_types = ["full", "reuse"]

    with pytest.raises(ValueError, match="must be 'full' or 'shared'"):
        get_dsa_offload_model_capabilities(config)


def test_undeclared_topk_freq_topology_is_not_treated_as_shared() -> None:
    # DeepSeek IndexCache 风格：用 index_topk_freq（无 indexer_types）造 skip_topk
    # 层。未声明 indexer_types → 不识别为 shared 拓扑，spec/bind 门会拒绝其
    # 缺 indexer 的层（不进入共享复用路径）。
    config = _model_config("DeepseekV32ForCausalLM")
    config.hf_text_config.index_topk_freq = 4

    capabilities = get_dsa_offload_model_capabilities(config)

    assert capabilities.indexer_types is None
    assert not capabilities.has_shared_indexer_layers
    assert capabilities.full_indexer_layer_indices is None
