# SPDX-License-Identifier: Apache-2.0
"""按模型拓扑生成静态下发计划；不导入 Torch，不在 forward 中执行规划。

本模块描述所需能力，不以生成计划代替设备实现。绑定前必须检查 capabilities；
generation 是每个 step 的逻辑偏移，不是可直接复用的 wire generation/物理 ring slot。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class DsaKvPlacement(str, Enum):
    LOCAL_ALL = "local_all"
    HBM_FULL_REMOTE_SHARED = "hbm_full_remote_shared"
    LOCAL_FULL_REMOTE_SHARED = "local_full_remote_shared"


class LayerExecutionPath(str, Enum):
    NATIVE_FULL = "native_full"
    LIS_SEND_FULL = "lis_send_full"
    LOCAL_LIDU_FULL = "local_lidu_full"
    LOCAL_LIDU_SEND_FULL = "local_lidu_send_full"
    LOCAL_SHARED = "local_shared"
    REMOTE_SHARED = "remote_shared"


class SelectionDomain(str, Enum):
    VISIBLE_CAUSAL = "visible_causal"
    DURABLE_WITH_CAUSAL_TAIL = "durable_with_causal_tail"
    SOURCE_PLAN = "source_plan"


@dataclass(frozen=True)
class LayerExecutionPlan:
    layer: int
    group: int
    source_layer: int
    role: str
    kv_backend: str
    du_owner: str
    ops: tuple[str, ...]
    first_payload_layer: int | None
    plan_command: int | None
    transfer_command: int | None
    send_command: int | None
    mirror_writer: str
    applied: bool
    path: LayerExecutionPath
    selection_domain: SelectionDomain


@dataclass(frozen=True)
class CommandPlan:
    index: int
    submit_layer: int
    payload_layer: int
    group: int
    plan_command: int
    action: str


@dataclass(frozen=True)
class SelectionGroupPlan:
    group: int
    source_layer: int
    layers: tuple[int, ...]
    remote_layers: tuple[int, ...]
    du_owner: str
    mirror_commit_layer: int | None
    release_layer: int


@dataclass(frozen=True)
class DsaExecutionPlan:
    placement: DsaKvPlacement
    layers: tuple[LayerExecutionPlan, ...]
    groups: tuple[SelectionGroupPlan, ...]
    commands: tuple[CommandPlan, ...]
    required_capabilities: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)

    def require_capabilities(self, available: set[str]) -> None:
        missing = sorted(set(self.required_capabilities) - available)
        if missing:
            raise RuntimeError(
                f"DSA placement={self.placement.value} execution backend is not ready: "
                f"missing={missing}; refusing silent local fallback"
            )

    def validate_binding(self, full_layers: tuple[int, ...], target_layers: tuple[int, ...]) -> None:
        """模型绑定阶段核对真实模块；不接受仅层数相同但来源关系错误的模型。"""
        expected_full = tuple(layer.layer for layer in self.layers if layer.role == "full")
        expected_target = tuple(layer.layer for layer in self.layers)
        if tuple(sorted(full_layers)) != expected_full or tuple(sorted(target_layers)) != expected_target:
            raise ValueError(
                f"DSA execution-plan binding mismatch: full={full_layers}/{expected_full}, "
                f"target={target_layers}/{expected_target}"
            )

    def generation(self, step: int, command: int, network_group: int, network_groups: int) -> int:
        """每次通信独立代次；共享 plan 始终引用其源 command 的同轮代次。"""
        if step < 0 or network_groups <= 0 or not 0 <= network_group < network_groups:
            raise ValueError("invalid step/network group")
        if not 0 <= command < len(self.commands):
            raise ValueError("invalid command index")
        return (step * len(self.commands) + command) * network_groups + network_group + 1

    def table(self) -> str:
        rows = ["layer group source kv du path selection ops plan transfer mirror applied"]
        for layer in self.layers:
            rows.append(
                f"{layer.layer} {layer.group} {layer.source_layer} {layer.kv_backend} "
                f"{layer.du_owner} {layer.path.value} {layer.selection_domain.value} "
                f"{'->'.join(layer.ops)} {layer.plan_command} "
                f"{layer.transfer_command} {layer.mirror_writer} {int(layer.applied)}"
            )
        return "\n".join(rows)


def build_execution_plan(
    indexer_types: tuple[str, ...] | list[str] | None,
    num_hidden_layers: int,
    placement: DsaKvPlacement | str = DsaKvPlacement.LOCAL_ALL,
) -> DsaExecutionPlan:
    """每个 full 开组，紧随的 shared 归入该组；MTP proposer 不在 target 拓扑内。

    无 indexer_types 的旧模型保持 all-full。配置改变必须重新分配/绑定/capture，
    不能给已有图替换此对象。全局 group 编号不因单组没有远端消费者而重编号。
    """
    placement = DsaKvPlacement(placement)
    if isinstance(num_hidden_layers, bool) or not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be a positive integer")
    if isinstance(indexer_types, (str, bytes)):
        raise ValueError("indexer_types must be a sequence of full/shared")
    roles = tuple(indexer_types) if indexer_types is not None else ("full",) * num_hidden_layers
    if len(roles) != num_hidden_layers or any(role not in ("full", "shared") for role in roles):
        raise ValueError("indexer_types must contain full/shared for every target layer")
    if roles[0] != "full":
        raise ValueError("shared layer must have a preceding full source")

    sources = [i for i, role in enumerate(roles) if role == "full"]
    layers: list[LayerExecutionPlan] = []
    groups: list[SelectionGroupPlan] = []
    commands: list[CommandPlan] = []
    capabilities: set[str] = set()
    for group, source in enumerate(sources):
        end = sources[group + 1] if group + 1 < len(sources) else len(roles)
        members = tuple(range(source, end))
        remote = members[1:] if placement != DsaKvPlacement.LOCAL_ALL else ()
        hbm_full = placement == DsaKvPlacement.HBM_FULL_REMOTE_SHARED
        owner = ("pool" if remote else "none") if hbm_full else "npu"
        first_payload = remote[0] if remote else None
        plan_command = len(commands) if remote else None
        transfer_commands: dict[int, int] = {}
        for ordinal, payload in enumerate(remote):
            action = ("DU_AND_GATHER" if hbm_full else "MISS_PLAN_AND_GATHER") if ordinal == 0 else "GATHER_ONLY"
            index = len(commands)
            commands.append(CommandPlan(index, source if ordinal == 0 else payload,
                                        payload, group, plan_command, action))
            transfer_commands[payload] = index
        if remote:
            capabilities.update(("shared_fetch", "scatter_sfa", "group_applied", "grouped_transport"))
            capabilities.add("pool_du" if hbm_full else "external_miss_plan")
        if hbm_full:
            capabilities.add("full_hbm_cache")
            if remote:
                capabilities.add("lis_send_durable_dual_output")

        for layer in members:
            is_full = layer == source
            last = layer == end - 1
            if placement == DsaKvPlacement.LOCAL_ALL:
                backend = "local_dram"
                path = LayerExecutionPath.LOCAL_LIDU_FULL if is_full else LayerExecutionPath.LOCAL_SHARED
                ops = ("lidu", "ksc", "sfa") if is_full else ("ksc", "sfa")
            elif is_full and hbm_full:
                backend = "hbm"
                path = LayerExecutionPath.LIS_SEND_FULL if remote else LayerExecutionPath.NATIVE_FULL
                ops = ("lis_send", "sfa_native") if remote else ("li", "sfa_native")
            elif is_full:
                backend = "local_dram"
                path = LayerExecutionPath.LOCAL_LIDU_SEND_FULL if remote else LayerExecutionPath.LOCAL_LIDU_FULL
                ops = ("lidu", "miss_plan_send", "ksc", "sfa") if remote else ("lidu", "ksc", "sfa")
            else:
                backend = "remote_pool"
                path = LayerExecutionPath.REMOTE_SHARED
                ops = ("scatter_sfa",) if layer == first_payload else ("gather_only", "scatter_sfa")
            domain = (
                SelectionDomain.SOURCE_PLAN if not is_full else
                SelectionDomain.VISIBLE_CAUSAL if path == LayerExecutionPath.NATIVE_FULL else
                SelectionDomain.DURABLE_WITH_CAUSAL_TAIL
            )
            applied = bool(remote) and last
            if applied:
                ops += ("applied",)
            mirror = "lidu" if owner == "npu" and is_full else (
                "scatter_sfa" if owner == "pool" and last else "none")
            send = plan_command if is_full else (transfer_commands.get(layer) if layer != first_payload else None)
            layers.append(LayerExecutionPlan(
                layer, group, source, roles[layer], backend, owner, ops, first_payload,
                plan_command, transfer_commands.get(layer), send, mirror, applied, path, domain,
            ))
        commit = end - 1 if owner == "pool" else (source if owner == "npu" else None)
        groups.append(SelectionGroupPlan(group, source, members, remote, owner, commit, end - 1))
    return DsaExecutionPlan(placement, tuple(layers), tuple(groups), tuple(commands), tuple(sorted(capabilities)))


def main() -> None:
    """无 Torch 的配置预检入口；只打印计划，不代表 backend 已支持执行。"""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="从模型 config.json 自动生成 DSA 下发计划（不执行模型）")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--placement", choices=[p.value for p in DsaKvPlacement], required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config = config.get("text_config", config)
    plan = build_execution_plan(config.get("indexer_types"), config["num_hidden_layers"], args.placement)
    print(json.dumps(plan.as_dict(), ensure_ascii=False, indent=2) if args.json else plan.table())


if __name__ == "__main__":
    main()
