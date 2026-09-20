# SPDX-License-Identifier: Apache-2.0
"""[TO BE REMOVE] 无MTP权重时的四query IO穿刺；不是MTP模型或性能基线。"""


def synthetic_draft_enabled(vllm_config) -> bool:
    return (getattr(vllm_config, "additional_config", None) or {}).get("dsa_synthetic_draft", False)


def validate_synthetic_draft(vllm_config, *, resolved_graph: bool = False) -> None:
    enabled = synthetic_draft_enabled(vllm_config)
    if type(enabled) is not bool:
        raise ValueError("dsa_synthetic_draft must be a boolean")
    if not enabled:
        return
    from vllm.config import CompilationMode, CUDAGraphMode

    spec = vllm_config.speculative_config
    if (spec is None or spec.method != "ngram" or spec.num_speculative_tokens != 3
            or spec.parallel_drafting or spec.disable_padded_drafter_batch):
        raise ValueError("synthetic draft requires ngram transport with exactly three drafts, no parallel drafting")
    if (not vllm_config.model_config.use_mla
            or vllm_config.scheduler_config.async_scheduling is not False
            or vllm_config.parallel_config.pipeline_parallel_size != 1):
        raise ValueError("synthetic draft supports MLA, PP1 and synchronous scheduling only")
    compilation = vllm_config.compilation_config
    if vllm_config.model_config.enforce_eager:
        if compilation.mode != CompilationMode.NONE or compilation.cudagraph_mode != CUDAGraphMode.NONE:
            raise ValueError("synthetic eager requires compilation/capture disabled")
    elif (compilation.mode != CompilationMode.VLLM_COMPILE
          or not compilation.cudagraph_mode.has_full_cudagraphs()
          or (resolved_graph and compilation.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY)):
        raise ValueError("synthetic graph requires target FULL_DECODE_ONLY; proposer stays outside capture")
    if getattr(spec, "rejection_sample_method", "standard") != "standard":
        raise ValueError("synthetic draft keeps real rejection sampling; forced acceptance is not supported")
