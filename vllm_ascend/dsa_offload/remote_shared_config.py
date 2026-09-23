# SPDX-License-Identifier: Apache-2.0
"""本机图外连接配置：按HCCP物理device ID选址；不同节点分别加载自己的表。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace


def _endpoint(value: str) -> None:
    host, sep, port = value.rpartition(":")
    if not host or not sep or not port.isdecimal() or not 0 < int(port) < 65536:
        raise ValueError("remote_shared endpoints must be host:port")


@dataclass(frozen=True)
class RemoteSharedDevice:
    du_endpoint: str
    device_eid: str
    eid_index: int = -1
    pool_eid: str = ""
    # 可覆盖全局默认Store；同一worker的CREATE/PUT与DU必须属于同一Pool。
    store_endpoint: str | None = None

    @classmethod
    def from_value(cls, value):
        if not isinstance(value, Mapping) or set(value) - {
            "du_endpoint", "device_eid", "eid_index", "pool_eid", "store_endpoint"
        } or not {"du_endpoint", "device_eid"} <= value.keys():
            raise ValueError("remote_shared device requires du_endpoint/device_eid")
        result = cls(**value)
        _endpoint(result.du_endpoint)
        if result.store_endpoint is not None:
            _endpoint(result.store_endpoint)
        for eid in (result.device_eid, result.pool_eid):
            if eid and (len(eid) != 32 or any(c not in "0123456789abcdefABCDEF" for c in eid)
                        or int(eid, 16) == 0):
                raise ValueError("remote_shared requires real 32-digit hexadecimal EIDs")
        if not result.device_eid or type(result.eid_index) is not int or result.eid_index < -1:
            raise ValueError("remote_shared requires a device EID and valid EID index")
        return cls(result.du_endpoint, result.device_eid.lower(), result.eid_index, result.pool_eid.lower(),
                   result.store_endpoint)


@dataclass(frozen=True)
class RemoteSharedDebug:
    directory: str
    steps: int = 3
    topk_rows: tuple[int, ...] = (0,)
    ranks: tuple[tuple[int, int], ...] = ((0, 0),)
    compare_li: bool = True
    topk_print_limit: int = 16
    capture_kv: bool = False

    @classmethod
    def from_value(cls, value):
        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) - {
            "directory", "steps", "topk_rows", "ranks", "compare_li", "topk_print_limit", "capture_kv"
        } or not isinstance(value.get("directory"), str) or not value["directory"].strip():
            raise ValueError("remote_shared.debug requires directory and supported debug fields")
        result = cls(**value)
        if (type(result.steps) is not int or result.steps < 1 or type(result.compare_li) is not bool
                or type(result.capture_kv) is not bool
                or type(result.topk_print_limit) is not int or not 1 <= result.topk_print_limit <= 2048
                or not isinstance(result.topk_rows, (list, tuple)) or not result.topk_rows
                or any(type(r) is not int or r < 0 for r in result.topk_rows)
                or not isinstance(result.ranks, (list, tuple)) or not result.ranks
                or any(not isinstance(r, (list, tuple)) or len(r) != 2
                       or any(type(v) is not int or v < 0 for v in r) for r in result.ranks)):
            raise ValueError("invalid remote_shared debug steps/rows/ranks/print limit")
        return cls(result.directory, result.steps, tuple(result.topk_rows),
                   tuple(tuple(r) for r in result.ranks), result.compare_li, result.topk_print_limit, result.capture_kv)


@dataclass(frozen=True)
class DSARemoteSharedConfig:
    store_endpoint: str
    devices: tuple[tuple[int, RemoteSharedDevice], ...] = ()
    single: RemoteSharedDevice | None = None
    debug: RemoteSharedDebug | None = None
    # 显式验收模式：恢复逐层status/ticket及decode输入值的同步D2H检查。
    validate: bool = False
    # 全局DP均分到各节点；devices只列本机卡，TP组不跨节点。
    node_count: int = 1

    @classmethod
    def from_value(cls, value):
        if value is None:
            return None
        if not isinstance(value, Mapping) or "store_endpoint" not in value:
            raise ValueError("remote_shared requires store_endpoint")
        _endpoint(value["store_endpoint"])
        debug = RemoteSharedDebug.from_value(value.get("debug"))
        validate = value.get("validate", False)
        if type(validate) is not bool:
            raise ValueError("remote_shared.validate must be a boolean")
        node_count = value.get("node_count", 1)
        if type(node_count) is not int or node_count < 1:
            raise ValueError("remote_shared.node_count must be a positive integer")
        if "devices" not in value:
            # 兼容已有DP1/TP1单EID文件，多rank不得静默复用它。
            single = RemoteSharedDevice.from_value({k: v for k, v in value.items()
                                                   if k not in ("store_endpoint", "debug", "validate", "node_count")})
            return cls(value["store_endpoint"], single=single, debug=debug, validate=validate, node_count=node_count)
        if set(value) - {"store_endpoint", "devices", "debug", "validate", "node_count"} or not isinstance(value["devices"], Mapping):
            raise ValueError("remote_shared accepts store_endpoint and devices keyed by physical device ID")
        devices = []
        for key, item in value["devices"].items():
            if not isinstance(key, str) or not key.isdecimal() or key != str(int(key)):
                raise ValueError("devices keys must be physical device IDs, e.g. '2', not logical 'npu:0'")
            devices.append((int(key), RemoteSharedDevice.from_value(item)))
        if not devices:
            raise ValueError("remote_shared devices cannot be empty")
        for name in ("du_endpoint", "device_eid"):
            entries = [getattr(device, name) for _, device in devices]
            if len(set(entries)) != len(entries):
                raise ValueError(f"remote_shared requires one independent {name} per physical device")
        if any(device.eid_index < 0 or not device.pool_eid for _, device in devices):
            raise ValueError("device table requires exact eid_index and pool_eid from probe")
        return cls(value["store_endpoint"], tuple(sorted(devices)), debug=debug, validate=validate, node_count=node_count)

    def validate_topology(self, dp: int, tp: int) -> None:
        if self.debug and any(d >= dp or t >= tp for d, t in self.debug.ranks):
            raise ValueError("remote_shared debug rank outside DP/TP topology")
        if dp < 1 or tp not in (1, 2):
            raise ValueError("remote_shared supports TP1/TP2 and positive DP size")
        if dp % self.node_count:
            raise ValueError("remote_shared DP must be divisible by node_count; TP groups must stay within a node")
        if dp * tp > 1 and not self.devices:
            raise ValueError("multi-rank remote_shared requires a physical-device table; single EID is TP1/DP1 only")
        local_devices = (dp // self.node_count) * tp
        if self.devices and len(self.devices) < local_devices:
            raise ValueError(
                f"remote_shared device table is smaller than local DP * TP: "
                f"configured={len(self.devices)} required={local_devices} node_count={self.node_count}"
            )

    def for_device(self, inventory) -> tuple[int, RemoteSharedDevice]:
        """inventory来自当前worker的aiv_urma_query_eids，不猜测ASCEND_RT_VISIBLE_DEVICES映射。"""
        physical = {int(row["phy_id"]) for row in inventory}
        if len(physical) != 1:
            raise ValueError("NPU EID inventory must identify exactly one physical device")
        phy_id = physical.pop()
        selected = dict(self.devices).get(phy_id) if self.devices else self.single
        if selected is None:
            raise ValueError(f"remote_shared has no configuration for physical NPU {phy_id}")
        if not any(row["eid"].lower() == selected.device_eid and
                   (selected.eid_index == -1 or int(row["eid_index"]) == selected.eid_index)
                   for row in inventory):
            raise ValueError(f"remote_shared EID/index does not belong to current physical NPU {phy_id}; rerun probe")
        return phy_id, replace(selected, store_endpoint=selected.store_endpoint or self.store_endpoint)
