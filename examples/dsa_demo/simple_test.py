#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""离线测试底座；用户配置和启动入口见 simple_test_1p.py / simple_test_2p.py。

卸载、target 图、MTP 图分别配置。配置显式传给 spawn 子进程，不修改模块全局状态。
沿用离线 SPMD DP、跨机前端 barrier 和异常清理逻辑；每个 TP 组完整落在一台机器上。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import os
import runpy
import socket
import time
from dataclasses import dataclass, fields, replace
from datetime import timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SmokeConfig:
    model_path: str
    tensor_parallel_size: int = 2
    data_parallel_size: int = 4  # 全局 DP 数，不是每机 DP 数。
    enable_offload: bool = True
    run_mode: str = "graph"  # cache-init / eager / graph；不卸载用 enable_offload=False。
    prompts: tuple[str, ...] | list[str] = ("请用一句话介绍你自己。",)
    max_num_seqs: int = 1  # 每个 DP rank 的 BS；prompt 模板循环填满 DP * BS 个请求。
    max_model_len: int = 25600
    max_num_batched_tokens: int = 25600
    enable_chunked_prefill: bool = False
    max_tokens: int = 64
    gpu_memory_utilization: float = 0.85
    quantization: str = "ascend"
    enable_expert_parallel: bool = True
    enable_a5_packed_c8: bool = True
    enable_cpu_binding: bool = True  # 共用宿主机、不希望改变绑核时设 False。
    enable_mtp: bool = True
    synthetic_draft: bool = False  # [TO BE REMOVE] 无MTP权重，合成三token，仅target/IO功能穿刺。
    mtp_num_speculative_tokens: int = 3
    enable_mtp_graph: bool = True  # 本地/远端卸载支持独立MTP3图；baseline与synthetic仍用eager。
    enable_profile: bool = False
    profile_dir: str = "./profiles"
    result_dir: str | None = "./results"  # 自动区分拓扑、卸载、图模式和 rank；None 不保存。
    process_timeout_seconds: int = 3600
    dsa_indexer_mla_block_ratio: int = 3
    dsa_max_active_reqs: int = 256
    dsa_hot_cpu_block_multiple: float = 1.0
    dsa_prompt_budget_thresholds: tuple[int, ...] = (32768, 65536)
    dsa_mtp_resident_budget_tokens: tuple[int, ...] = (8192, 12288, 12288)  # 仅卸载 + MTP 生效。
    graph_capture_sizes: tuple[int, ...] = (1, 2, 4, 8, 16)  # 请求数；自动补入 max_num_seqs，再转换成 token 数。
    store_endpoint: str | None = None  # 仅 CLI --store-endpoint；真实写侧穿刺，推理仍是 local_all。
    remote_shared: dict[str, Any] | None = None  # CLI JSON；真实shared Get，不是store_shadow。
    ready_shadow_dir: str | None = None  # 仅CLI；使用真实LIDU轨迹，候选不改变模型输出。
    ready_shadow_steps: int = 3
    ready_shadow_group: int = 0
    metrics_summary: bool = True  # generate 返回后汇总 TTFT/TPOT/吞吐；host 时钟，不计时不扰动。
    bench: bool = False  # bench 口径：定长 decode（min_tokens=max_tokens, ignore_eos）+ 官方 bench 风格报告。

    @property
    def mtp_graph_enabled(self) -> bool:
        return (not self.synthetic_draft and self.enable_offload and self.run_mode == "graph"
                and self.enable_mtp and self.enable_mtp_graph)

    @property
    def run_name(self) -> str:
        mtp = "mtp_graph" if self.mtp_graph_enabled else ("mtp_eager" if self.enable_mtp else "no_mtp")
        if self.synthetic_draft:
            mtp = "synthetic_draft3_NOT_MTP"
        return (
            f"dp{self.data_parallel_size}_tp{self.tensor_parallel_size}"
            f"_{'offload' if self.enable_offload else 'baseline'}_{self.run_mode}_{mtp}"
            f"{'_store_shadow' if self.store_endpoint else ''}"
            f"{'_hbm_full_remote_shared' if self.remote_shared else ''}"
        )


NATIVE_RUNTIME_ENV_OVERRIDES = {
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_BUFFSIZE": "512",
    "OMP_NUM_THREADS": "10",
    "OMP_PROC_BIND": "false",
    "PYTHONHASHSEED": "114514",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "VLLM_ASCEND_ENABLE_FUSED_MC2": "0",
    "VLLM_ASCEND_ENABLE_MLAPO": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}


def read_smoke_environment() -> dict[str, str]:
    # Load the stdlib-only registry without package __init__/logger importing
    # vLLM before the child's rank/device environment has been configured.
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/envs.py"
    spec = importlib.util.spec_from_file_location("smoke_envs", path)
    envs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(envs)
    known = {envs.SMOKE_ENV_PREFIX + field.upper(): field for field in envs.SMOKE_ENV_FIELDS}
    unknown = sorted(name for name in os.environ if name.startswith(envs.SMOKE_ENV_PREFIX) and name not in known)
    if unknown:
        raise ValueError(f"Unknown smoke environment variables: {', '.join(unknown)}")
    return {field: envs.env_variables[name]() for name, field in known.items() if name in os.environ}


def config_from_environment(defaults: SmokeConfig, values: dict[str, str]) -> SmokeConfig:
    overrides = {}
    for field in fields(defaults):
        if field.name not in values:
            continue
        # Budget 字段排在两个开关之后，使用本次环境变量覆盖后的开关值。
        if field.name == "dsa_mtp_resident_budget_tokens" and not (
            overrides.get("enable_mtp", defaults.enable_mtp)
            and overrides.get("enable_offload", defaults.enable_offload)
        ):
            print("[dsa-smoke] ignoring DSA_MTP_RESIDENT_BUDGET_TOKENS: requires offload + MTP", flush=True)
            continue
        raw = values[field.name]
        default = getattr(defaults, field.name)
        try:
            if field.name == "prompts":
                value = json.loads(raw)
                if not isinstance(value, list):
                    raise ValueError("expected a JSON list of strings")
            elif isinstance(default, bool):
                choices = {
                    "1": True,
                    "true": True,
                    "yes": True,
                    "on": True,
                    "0": False,
                    "false": False,
                    "no": False,
                    "off": False,
                }
                value = choices[raw.strip().lower()]
            elif isinstance(default, tuple):
                value = tuple(int(item.strip()) for item in raw.split(","))
            elif isinstance(default, (int, float)):
                value = type(default)(raw)
            elif field.name == "result_dir" and raw.strip().lower() in {"none", "null", ""}:
                value = None
            else:
                value = raw
        except (ValueError, KeyError) as error:
            raise ValueError(f"Invalid environment value for {field.name}: {raw!r}") from error
        overrides[field.name] = value
    if "dataset" in values:
        if "prompts" in values:
            raise ValueError("Set only one of DATASET and PROMPTS smoke overrides")
        # Lazy import: arbitrary short/custom prompts do not need the novel data.
        import novel_dataset

        datasets = {
            name: getattr(novel_dataset, name)
            for name in ("chinese_5k", "chinese_10k", "chinese_23k", "chinese_40k", "chinese_64k", "chinese_104k")
        }
        datasets["short"] = ["请用一句话介绍你自己。"]
        name = values["dataset"].strip()
        if name not in datasets:
            raise ValueError(f"Unknown DATASET {name!r}; choose {', '.join(datasets)}")
        overrides["prompts"] = datasets[name]
    return replace(defaults, **overrides)


def main_from_env(defaults: SmokeConfig, *, node_size: int) -> None:
    values = read_smoke_environment()
    node = {}
    if node_size > 1:
        for key, default in (("node_rank", 0), ("master_addr", ""), ("master_port", 29500), ("sync_port", 29600)):
            raw = values.get(key, default)
            try:
                node[key] = type(default)(raw)
            except ValueError as error:
                raise ValueError(f"Invalid environment value for {key}: {raw!r}") from error
    main(config_from_environment(defaults, values), node_size=node_size, **node)


def validate_config(config: SmokeConfig) -> None:
    if config.synthetic_draft and (not config.enable_mtp or config.mtp_num_speculative_tokens != 3
                                  or config.run_mode not in ("eager", "graph")):
        raise ValueError("synthetic draft requires ENABLE_MTP=1, three drafts and RUN_MODE=eager/graph")
    if config.remote_shared is not None:
        if (not config.enable_offload or not config.enable_a5_packed_c8 or not config.enable_mtp
                or config.mtp_num_speculative_tokens != 3 or config.run_mode not in ("eager", "graph")
                or config.tensor_parallel_size not in (1, 2)
                or not 1 <= config.max_num_seqs <= 32
                or config.store_endpoint or config.ready_shadow_dir):
            raise ValueError(
                "remote shared smoke requires TP1/TP2, BS1..32 per DP, A5 C8 MTP3 eager/graph; "
                "one offline cohort, no row reuse or shadow modes"
            )
        # 只加载stdlib配置模块；dry-run与spawn之前不能初始化torch/vLLM/NPU。
        path = Path(__file__).resolve().parents[2] / "vllm_ascend/dsa_offload/remote_shared_config.py"
        remote = runpy.run_path(str(path))["DSARemoteSharedConfig"].from_value(config.remote_shared)
        remote.validate_topology(config.data_parallel_size, config.tensor_parallel_size)
        if remote.debug is not None and config.run_mode != "eager":
            raise ValueError("remote_shared debug requires RUN_MODE=eager; not a performance run")
    if config.ready_shadow_dir is not None and (
        not config.enable_offload or not config.enable_a5_packed_c8 or not config.enable_mtp
        or config.mtp_num_speculative_tokens != 3 or config.run_mode != "eager"
        or config.data_parallel_size != 1 or config.tensor_parallel_size not in (1, 2)
        or config.max_num_seqs != 1 or config.enable_mtp_graph or config.enable_profile
        or config.ready_shadow_steps < 1 or config.ready_shadow_group < 0
    ):
        raise ValueError("ready shadow requires DP1 TP1/2 BS1, local A5 offload, MTP3 eager; no graph/profile")
    if config.store_endpoint and (
        not config.enable_offload or not config.enable_a5_packed_c8 or config.run_mode != "eager"
    ):
        raise ValueError("--store-endpoint requires offload + A5 C8 + run_mode=eager; inference remains local_all")
    if config.run_mode not in {"cache-init", "eager", "graph"}:
        raise ValueError("run_mode must be cache-init/eager/graph; use enable_offload=False for baseline")
    for name in (
        "tensor_parallel_size",
        "data_parallel_size",
        "max_num_seqs",
        "max_model_len",
        "max_num_batched_tokens",
        "max_tokens",
        "process_timeout_seconds",
    ):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not config.prompts or isinstance(config.prompts, str) or any(not isinstance(p, str) for p in config.prompts):
        raise ValueError("prompts must be a nonempty list/tuple of strings")
    if not 0 < config.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if not config.enable_chunked_prefill and config.max_num_batched_tokens < config.max_model_len:
        raise ValueError("chunked prefill is off: max_num_batched_tokens must be >= max_model_len")
    if any(size < 1 for size in config.graph_capture_sizes):
        raise ValueError("graph_capture_sizes must contain positive request counts")
    if config.enable_offload and config.data_parallel_size > 1 and not config.enable_expert_parallel:
        raise ValueError("DSA DP requires enable_expert_parallel=True")
    if config.enable_mtp:
        if config.mtp_num_speculative_tokens != 3:
            raise ValueError("This smoke test supports MTP3 only")
        if config.enable_offload and not config.enable_a5_packed_c8:
            raise ValueError("DSA MTP3 requires enable_a5_packed_c8=True")
        if config.enable_offload:
            budgets = config.dsa_mtp_resident_budget_tokens
            if len(budgets) != len(config.dsa_prompt_budget_thresholds) + 1:
                raise ValueError(
                    "DSA_MTP_RESIDENT_BUDGET_TOKENS must have one more value than DSA_PROMPT_BUDGET_THRESHOLDS"
                )
            if any(budget not in (8192, 10240, 12288) for budget in budgets):
                raise ValueError("DSA MTP resident budgets must be 8192, 10240 or 12288")
            if any(left > right for left, right in zip(budgets, budgets[1:])):
                raise ValueError("DSA MTP resident budgets must be non-decreasing")


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def resolve_node_launch(
    config: SmokeConfig, args: argparse.Namespace
) -> tuple[str, int, int | None, list[tuple[int, int]]]:
    validate_config(config)
    if config.remote_shared is not None and config.remote_shared.get("node_count", 1) != args.node_size:
        raise ValueError("remote_shared.node_count must match launcher node-size")
    if args.node_size < 1 or not 0 <= args.node_rank < args.node_size:
        raise ValueError("node-size must be positive and node-rank must be in [0, node-size)")
    if config.data_parallel_size % args.node_size:
        raise ValueError("data_parallel_size must be divisible by node-size; cross-node TP is not supported here")
    dp_per_node = config.data_parallel_size // args.node_size
    ranks = [(local, args.node_rank * dp_per_node + local) for local in range(dp_per_node)]
    if args.node_size == 1:
        return "127.0.0.1", get_open_port(), None, ranks
    if not args.master_addr.strip():
        raise ValueError("Set MASTER_ADDR in simple_test_2p.py or pass --master-addr with node 0's IP")
    if not all(1 <= p <= 65535 for p in (args.master_port, args.sync_port)):
        raise ValueError("master-port and sync-port must be in [1, 65535]")
    if args.master_port == args.sync_port:
        raise ValueError("sync-port must differ from master-port")
    return args.master_addr.strip(), args.master_port, args.sync_port, ranks


def build_llm_kwargs(config: SmokeConfig, multi_node: bool = False) -> dict[str, Any]:
    """纯配置构建：不导入 vLLM/torch、不初始化设备、不创建 profile 目录。"""
    validate_config(config)
    graph = config.run_mode == "graph"
    additional_config: dict[str, Any] = {"enable_cpu_binding": config.enable_cpu_binding}
    if config.synthetic_draft:
        additional_config["dsa_synthetic_draft"] = True
    if config.data_parallel_size > 1:
        additional_config["dp_allreduce_on_npu"] = True
    if multi_node:
        # Ascend 950 使用已验证的默认 MC2 算法，不强制 hierarchy。
        additional_config["enable_mc2_hierarchy_comm"] = False
    if config.enable_a5_packed_c8:
        additional_config.update(enable_sparse_sfa_c8=True, enable_sparse_li_c8=True)
    if config.enable_offload:
        additional_config["dsa_sparse_config"] = {
            "enabled": True,
            "split_indexer_cache": True,
            "indexer_mla_block_ratio": config.dsa_indexer_mla_block_ratio,
            "sparse_activation_tokens": 8192 if config.enable_mtp else 6144,
            "prompt_budget_thresholds": list(config.dsa_prompt_budget_thresholds),
            "resident_budget_tokens": list(config.dsa_mtp_resident_budget_tokens)
            if config.enable_mtp
            else [10240, 10240, 10240],
            "max_active_reqs": config.dsa_max_active_reqs,
            "hot_cpu_block_multiple": config.dsa_hot_cpu_block_multiple,
            "enable_row_mode_decode_graph": graph,
            "trace_points": {"enabled": False, "points": ["first_sample"], "ranks": [0]},
        }
        if config.store_endpoint:
            additional_config["dsa_sparse_config"]["remote_store"] = {
                "mode": "store_shadow", "endpoint": config.store_endpoint,
            }
        if config.remote_shared is not None:
            additional_config["dsa_sparse_config"].update(
                placement="hbm_full_remote_shared", remote_shared=config.remote_shared)
        if config.ready_shadow_dir is not None:
            additional_config["dsa_sparse_config"]["ready_shadow"] = {
                "directory": config.ready_shadow_dir, "steps": config.ready_shadow_steps,
                "group": config.ready_shadow_group,
            }
    kwargs: dict[str, Any] = {
        "model": config.model_path,
        "tensor_parallel_size": config.tensor_parallel_size,
        "pipeline_parallel_size": 1,
        "quantization": config.quantization,
        "seed": 1024,
        "enable_expert_parallel": config.enable_expert_parallel,
        "max_num_seqs": config.max_num_seqs,
        "max_model_len": config.max_model_len,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "trust_remote_code": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": config.enable_chunked_prefill,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "block_size": 128,
        "async_scheduling": False,
        "enforce_eager": not graph,
        "disable_log_stats": False,
        "additional_config": additional_config,
    }
    if config.enable_mtp:
        kwargs["speculative_config"] = {
            "method": "ngram" if config.synthetic_draft else "mtp",
            "num_speculative_tokens": config.mtp_num_speculative_tokens,
            "draft_tensor_parallel_size": config.tensor_parallel_size,
            "enforce_eager": not config.mtp_graph_enabled,
        }
        # graph/eager、baseline/offload 使用相同的 hidden/collective 布局。
        additional_config.update(
            enable_flashcomm1=False,
            enable_shared_expert_dp=False,
            enable_flashcomm2_parallel_size=0,
            enable_reduce_sample=False,
        )
    if graph:
        query_len = 1 + config.mtp_num_speculative_tokens if config.enable_mtp else 1
        kwargs["compilation_config"] = {
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": sorted(
                {
                    size * query_len
                    for size in (*config.graph_capture_sizes, config.max_num_seqs)
                    if size <= config.max_num_seqs
                }
            ),
            "pass_config": {"enable_sp": False},
        }
    return kwargs


def shard_prompts(config: SmokeConfig, dp_rank: int) -> list[tuple[int, str]]:
    start = dp_rank * config.max_num_seqs
    return [(i, config.prompts[i % len(config.prompts)]) for i in range(start, start + config.max_num_seqs)]


def write_result(config: SmokeConfig, payload: dict[str, Any], dp_rank: int) -> None:
    if config.result_dir is None:
        return
    path = Path(config.result_dir) / f"{config.run_name}.dp{dp_rank}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[dsa-smoke] wrote result: {path}", flush=True)


def _percentile(ordered: list[float], q: float) -> float:
    n = len(ordered)
    if n == 1:
        return ordered[0]
    rank = min(n - 1, max(0, int(round(q * (n - 1)))))
    return ordered[rank]


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return {
        "avg": round(sum(ordered) / n, 3),
        "min": round(ordered[0], 3),
        "p50": round(median, 3),
        "p90": round(_percentile(ordered, 0.9), 3),
        "p99": round(_percentile(ordered, 0.99), 3),
        "max": round(ordered[-1], 3),
    }


def _bench_report(scope: str, requests: int, output_tokens: int, wall_s: float,
                  ttft: dict[str, float] | None, tpot: dict[str, float] | None) -> str:
    """vllm bench serve 风格的报告块，指标同名同公式，离线口径。"""
    def row(label, stats, unit="ms"):
        if not stats:
            return f"{label}: unavailable"
        return (f"{label}: avg={stats['avg']}{unit} p50={stats['p50']} p90={stats['p90']} "
                f"p99={stats['p99']} max={stats['max']}")
    lines = [
        f"============ DSA Smoke Benchmark Result ({scope}, offline) ============",
        f"Successful requests: {requests}",
        f"Benchmark duration (s): {round(wall_s, 2)}",
        f"Total generated tokens: {output_tokens}",
        f"Request throughput (req/s): {round(requests / wall_s, 3) if wall_s > 0 else None}",
        f"Output token throughput (tok/s): {round(output_tokens / wall_s, 2) if wall_s > 0 else None}",
        row("TTFT (ms)", ttft),
        row("TPOT (ms, excl. 1st token)", tpot),
        "=" * 72,
    ]
    return "\n".join(lines)


def summarize_request_metrics(
    outputs: list[Any],
    gen_start: float,
    gen_end: float,
) -> dict[str, Any]:
    """每请求 TTFT/TPOT + rank 级吞吐。全部在 generate 返回后计算，用 host 时钟，不加设备同步。"""
    wall_s = gen_end - gen_start
    per_request = []
    ttfts: list[float] = []
    tpots: list[float] = []
    total_tokens = 0
    for output in outputs:
        n = len(output.outputs[0].token_ids)
        total_tokens += n
        metrics = output.metrics
        if metrics is None or metrics.first_token_ts is None or metrics.last_token_ts is None:
            per_request.append({"request_id": output.request_id, "output_tokens": n, "metrics": "unavailable"})
            continue
        ttft_ms = (metrics.first_token_ts - metrics.arrival_time) * 1000
        tpot_ms = (metrics.last_token_ts - metrics.first_token_ts) / (n - 1) * 1000 if n > 1 else None
        ttfts.append(ttft_ms)
        if tpot_ms is not None:
            tpots.append(tpot_ms)
        per_request.append(
            {
                "request_id": output.request_id,
                "output_tokens": n,
                "ttft_ms": round(ttft_ms, 3),
                "tpot_ms": round(tpot_ms, 3) if tpot_ms is not None else None,
            }
        )
    avg_tpot = sum(tpots) / len(tpots) if tpots else None
    return {
        "requests": len(outputs),
        "output_tokens": total_tokens,
        "wall_s": round(wall_s, 3),
        "gen_start_epoch": gen_start,
        "gen_end_epoch": gen_end,
        "ttft_ms": _stats(ttfts),
        "tpot_ms": _stats(tpots),
        "decode_tps_per_seq": round(1000 / avg_tpot, 2) if avg_tpot else None,
        "rank_output_tps": round(total_tokens / wall_s, 2) if wall_s > 0 else None,
        "per_request": per_request,
    }


def merge_global_metrics(config: SmokeConfig, dp_rank: int) -> None:
    """rank0 在 shutdown barrier 后合并各 rank 已落盘的 summary；失败不影响跑程结论。"""
    if dp_rank != 0 or config.result_dir is None or not config.metrics_summary:
        return
    try:
        summaries = []
        for rank in range(config.data_parallel_size):
            path = Path(config.result_dir) / f"{config.run_name}.dp{rank}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "metrics_summary" in payload:
                summaries.append(payload["metrics_summary"])
        if len(summaries) != config.data_parallel_size:
            print(f"[dsa-smoke][metrics] global merge skipped: {len(summaries)}/{config.data_parallel_size} "
                  "rank summaries present", flush=True)
            return
        start = min(s["gen_start_epoch"] for s in summaries)
        end = max(s["gen_end_epoch"] for s in summaries)
        tokens = sum(s["output_tokens"] for s in summaries)
        wall_s = end - start
        # 全局分位数从各 rank per_request 明细重算，不做 rank 平均的平均。
        all_ttft = [r["ttft_ms"] for s in summaries for r in s["per_request"] if "ttft_ms" in r]
        all_tpot = [r["tpot_ms"] for s in summaries for r in s["per_request"] if r.get("tpot_ms") is not None]
        ttft_stats, tpot_stats = _stats(all_ttft), _stats(all_tpot)
        requests = sum(s["requests"] for s in summaries)
        global_summary = {
            "dp_size": config.data_parallel_size,
            "requests": requests,
            "output_tokens": tokens,
            "global_wall_s": round(wall_s, 3),
            "request_throughput_rps": round(requests / wall_s, 3) if wall_s > 0 else None,
            "global_output_tps": round(tokens / wall_s, 2) if wall_s > 0 else None,
            "ttft_ms": ttft_stats,
            "tpot_ms": tpot_stats,
            "tpot_under_50ms": (tpot_stats["p99"] < 50) if tpot_stats else None,
        }
        out = Path(config.result_dir) / f"{config.run_name}.metrics.json"
        out.write_text(json.dumps({"global": global_summary, "ranks": summaries}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print("[dsa-smoke][metrics][global] " + json.dumps(global_summary, ensure_ascii=False), flush=True)
        print(_bench_report("global", requests, tokens, wall_s, ttft_stats, tpot_stats), flush=True)
        print(f"[dsa-smoke][metrics] wrote {out}", flush=True)
    except Exception as exc:
        print(f"[dsa-smoke][metrics] global merge failed (non-fatal): {exc}", flush=True)


def wait_frontend_barrier(store: Any, phase: str, dp_rank: int, dp_size: int) -> None:
    store.set(f"{phase}/{dp_rank}", "1")
    store.wait([f"{phase}/{rank}" for rank in range(dp_size)])


def run_dp_rank(
    config: SmokeConfig,
    dp_rank: int,
    local_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    shutdown_barrier: Any,
    sync_port: int | None,
) -> None:
    # 必须先设置 rank/device 环境，再导入任何 vLLM/torch 模块。
    os.environ.update(NATIVE_RUNTIME_ENV_OVERRIDES)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    os.environ.update(
        {
            "VLLM_DP_RANK": str(dp_rank),
            "VLLM_DP_RANK_LOCAL": str(local_dp_rank),
            "VLLM_DP_SIZE": str(config.data_parallel_size),
            "VLLM_DP_MASTER_IP": dp_master_ip,
            "VLLM_DP_MASTER_PORT": str(dp_master_port),
        }
    )
    store = None
    llm = None
    try:
        if sync_port is not None:
            from torch.distributed import TCPStore

            store = TCPStore(
                dp_master_ip,
                sync_port,
                config.data_parallel_size,
                dp_rank == 0,
                timedelta(seconds=config.process_timeout_seconds),
                True,
            )
            wait_frontend_barrier(store, "ready", dp_rank, config.data_parallel_size)
        from vllm import LLM, SamplingParams

        kwargs = build_llm_kwargs(config, multi_node=sync_port is not None)
        if config.enable_profile:
            from vllm.config import ProfilerConfig

            profile_dir = Path(config.profile_dir).resolve() / config.run_name / f"dp_rank_{dp_rank}"
            profile_dir.mkdir(parents=True, exist_ok=True)
            kwargs["profiler_config"] = ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=str(profile_dir),
                torch_profiler_with_stack=False,
                torch_profiler_record_shapes=False,
                torch_profiler_with_memory=False,
            )
        print(
            f"[dsa-smoke] rank={dp_rank}/{config.data_parallel_size} local_rank={local_dp_rank} "
            f"mode={config.run_mode} offload={config.enable_offload} model={config.model_path!r} "
            f"tp={config.tensor_parallel_size} bs={config.max_num_seqs} max_model_len={config.max_model_len} "
            f"mtp={config.enable_mtp} mtp_graph_enabled={config.mtp_graph_enabled}",
            flush=True,
        )
        llm = LLM(**kwargs)
        payload = {
            "mode": config.run_mode,
            "offload": config.enable_offload,
            "model": config.model_path,
            "dp_rank": dp_rank,
            "mtp_graph_enabled": config.mtp_graph_enabled,
            "synthetic_draft": config.synthetic_draft,
            "physical_drafter": not config.synthetic_draft and config.enable_mtp,
            "status": "passed",
        }
        if config.run_mode == "cache-init":
            payload["validated"] = "LLM/cache initialization only; no generate()"
            print(f"[dsa-smoke][dp={dp_rank}] cache initialized; offload={config.enable_offload}", flush=True)
        else:
            items = shard_prompts(config, dp_rank)
            sampling = SamplingParams(temperature=0.0, max_tokens=config.max_tokens,
                                      min_tokens=config.max_tokens if config.bench else 0,
                                      ignore_eos=config.bench)
            profiling_started = False
            try:
                if config.enable_profile:
                    llm.start_profile()
                    profiling_started = True
                gen_start = time.time()
                outputs = llm.generate([prompt for _, prompt in items], sampling, use_tqdm=False)
                gen_end = time.time()
            finally:
                if profiling_started:
                    llm.stop_profile()
            records = []
            for (index, _), output in zip(items, outputs, strict=True):
                completion = output.outputs[0]
                record = {
                    "request_index": index,
                    "dp_rank": dp_rank,
                    "prompt": output.prompt,
                    "token_ids": list(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "text": completion.text,
                }
                records.append(record)
                print(
                    f"\nDP {dp_rank} Req {index}: token_ids={record['token_ids']} "
                    f"finish_reason={record['finish_reason']!r}\nOutput: {record['text']!r}\n{'-' * 60}",
                    flush=True,
                )
            payload["outputs"] = records
            if config.metrics_summary:
                summary = summarize_request_metrics(outputs, gen_start, gen_end)
                payload["metrics_summary"] = summary
                brief = {k: v for k, v in summary.items() if k != "per_request"}
                print(f"[dsa-smoke][metrics][dp={dp_rank}] " + json.dumps(brief, ensure_ascii=False), flush=True)
                if config.bench:
                    print(_bench_report(f"dp={dp_rank}", summary["requests"], summary["output_tokens"],
                                        summary["wall_s"], summary["ttft_ms"], summary["tpot_ms"]), flush=True)
            if config.remote_shared and config.remote_shared.get("debug") is not None:
                from vllm_ascend.dsa_offload.remote_shared_debug import remote_shared_debug_report

                reports = llm.collective_rpc(remote_shared_debug_report)
                payload["remote_debug"] = reports
                print("DSA_REMOTE_DEBUG_SUMMARY " + json.dumps(reports), flush=True)
                if not reports or not all(report["passed"] for report in reports):
                    payload["status"] = "failed_remote_debug"
            if config.ready_shadow_dir is not None:
                from vllm_ascend.dsa_offload.ready_shadow import ready_shadow_report

                reports = llm.collective_rpc(ready_shadow_report)
                payload["ready_shadow"] = reports
                print("DSA_READY_SHADOW_SUMMARY " + json.dumps(reports), flush=True)
                if not reports or not all(report["passed"] for report in reports):
                    payload["status"] = "failed_ready_shadow"
                    write_result(config, payload, dp_rank)
                    raise RuntimeError("ready shadow mismatch or insufficient coverage; see per-rank reports")
        write_result(config, payload, dp_rank)
        # 等所有 DP 完成后才退出，防止提前销毁仍在参与 EP collective 的 worker。
        if store is not None:
            wait_frontend_barrier(store, "shutdown", dp_rank, config.data_parallel_size)
        else:
            shutdown_barrier.wait(timeout=config.process_timeout_seconds)
        # 各 rank 的结果文件已在 barrier 前落盘；rank0 合并全局吞吐/TPOT。
        merge_global_metrics(config, dp_rank)
        if payload.get("status") == "failed_remote_debug":
            raise RuntimeError("remote debug mismatch or insufficient coverage; see DSA_REMOTE_DEBUG_SUMMARY")
    except BaseException:
        if shutdown_barrier is not None:
            shutdown_barrier.abort()
        raise
    finally:
        if llm is not None:
            del llm


def stop_processes(processes: list[multiprocessing.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        if process.pid is not None:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)


def run_node_dp(config: SmokeConfig, args: argparse.Namespace) -> None:
    master_ip, master_port, sync_port, ranks = resolve_node_launch(config, args)
    # PYTHONHASHSEED 等需要在 spawn 启动解释器前生效；导入底座时不修改环境。
    os.environ.update(NATIVE_RUNTIME_ENV_OVERRIDES)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(config.data_parallel_size) if args.node_size == 1 else None
    ep_size = config.tensor_parallel_size * config.data_parallel_size if config.enable_expert_parallel else 1
    print(
        f"[dsa-smoke] node={args.node_rank}/{args.node_size} global topology "
        f"TP={config.tensor_parallel_size} DP={config.data_parallel_size} EP={ep_size}; "
        f"local_dp_ranks={len(ranks)} global_dp_ranks={[global_rank for _, global_rank in ranks]} "
        f"required_local_npus={config.tensor_parallel_size * len(ranks)} "
        f"dp_master={master_ip}:{master_port} sync_port={sync_port}",
        flush=True,
    )
    if config.data_parallel_size == 1:
        run_dp_rank(config, 0, 0, master_ip, master_port, barrier, sync_port)
        return
    processes = [
        context.Process(
            target=run_dp_rank,
            args=(config, global_rank, local, master_ip, master_port, barrier, sync_port),
            name=f"dsa-dp-{global_rank}",
        )
        for local, global_rank in ranks
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + config.process_timeout_seconds
        while True:
            failed = [p for p in processes if p.exitcode not in (None, 0)]
            if failed:
                details = ", ".join(f"{p.name}={p.exitcode}" for p in failed)
                raise RuntimeError(f"offline DP rank failed: {details}")
            if all(p.exitcode == 0 for p in processes):
                for process in processes:
                    process.join()
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"offline DP did not finish within {config.process_timeout_seconds} seconds")
            time.sleep(1)
    finally:
        if any(p.is_alive() for p in processes):
            if barrier is not None:
                barrier.abort()
            stop_processes(processes)


def main(
    config: SmokeConfig,
    *,
    node_size: int,
    node_rank: int = 0,
    master_addr: str = "",
    master_port: int = 29500,
    sync_port: int = 29600,
) -> None:
    parser = argparse.ArgumentParser(description="Offline TP/DP test; configure with VLLM_ASCEND_SMOKE_* environment")
    if node_size > 1:
        parser.add_argument("--node-rank", type=int, default=node_rank)
        parser.add_argument("--master-addr", default=master_addr)
        parser.add_argument("--master-port", type=int, default=master_port)
        parser.add_argument("--sync-port", type=int, default=sync_port)
    else:
        parser.set_defaults(node_rank=0, master_addr="", master_port=0, sync_port=0)
    parser.set_defaults(node_size=node_size)
    parser.add_argument("--mtp-eager", action="store_true", help="MTP eager 对照；不改变 target 模式")
    parser.add_argument("--store-endpoint", help="host:port；真实 CREATE/PUT/PUT_DONE/CLOSE，推理仍走本地，不启用远端 Get")
    parser.add_argument("--remote-shared-config", help="本机JSON文件；Store/DU/EID配置，各节点分别加载自己的Pool配置")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置和 rank 分配，不加载模型或初始化 NPU")
    parser.add_argument("--ready-shadow-dir", help="有限ready旁路对拍；失败落盘，推理输出仍使用本地LIDU/KSC/SFA")
    parser.add_argument("--ready-shadow-steps", type=int, default=3)
    parser.add_argument("--ready-shadow-group", type=int, default=0, help="full层对应的selection-state id")
    args = parser.parse_args()
    if args.mtp_eager:
        config = replace(config, enable_mtp_graph=False)
    if args.remote_shared_config:
        with open(args.remote_shared_config, encoding="utf-8") as source:
            config = replace(config, remote_shared=json.load(source))
    if config.remote_shared is not None:
        # 在父进程校验和spawn之前固定本机配置；节点数必须原样传到框架worker。
        remote_shared = dict(config.remote_shared)
        remote_shared.setdefault("node_count", args.node_size)
        config = replace(config, remote_shared=remote_shared)
        print(f"[dsa-smoke] remote_shared host={socket.gethostname()} "
              f"node={args.node_rank}/{args.node_size} config={args.remote_shared_config} "
              f"store={remote_shared.get('store_endpoint')}", flush=True)
        print("[dsa-smoke] hbm_full_remote_shared real_store=1 real_get=1; target=" + config.run_mode +
              "; PUT/REFRESH/status checks outside decode graph", flush=True)
    if args.store_endpoint:
        config = replace(config, store_endpoint=args.store_endpoint)
        print("[dsa-smoke] mode=store_shadow inference=local_all remote_get=0; synchronous PUT, not a performance run", flush=True)
    if args.ready_shadow_dir is not None:
        config = replace(config, ready_shadow_dir=args.ready_shadow_dir,
                         ready_shadow_steps=args.ready_shadow_steps, ready_shadow_group=args.ready_shadow_group)
        print("[dsa-smoke] ready shadow: reference=LIDU/KSC/SFA candidate=CopySFA; real_pool=0; NOT a performance run", flush=True)
    # 在创建任何子进程之前发现无效配置。
    _, _, _, ranks = resolve_node_launch(config, args)
    if config.synthetic_draft:
        print("[TO BE REMOVE] synthetic draft3: no MTP model/weights; target mode=" + config.run_mode +
              "; proposer=host; rejection=standard; NOT MTP performance/accuracy validation", flush=True)
    elif config.enable_mtp and config.enable_mtp_graph and not config.enable_offload:
        print("[dsa-smoke] baseline MTP graph is not supported; using MTP eager, target mode unchanged", flush=True)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "node_rank": args.node_rank,
                    "rank_map": ranks,
                    "required_local_npus": config.tensor_parallel_size * len(ranks),
                    "total_requests": config.data_parallel_size * config.max_num_seqs,
                    "run_name": config.run_name,
                    "llm_kwargs": build_llm_kwargs(config, node_size > 1),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    run_node_dp(config, args)
