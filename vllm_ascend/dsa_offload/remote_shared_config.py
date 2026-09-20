# SPDX-License-Identifier: Apache-2.0
"""图外连接配置：按HCCP返回的物理device ID选址，不依赖DP/TP编号或可见卡顺序。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


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

    @classmethod
    def from_value(cls, value):
        if not isinstance(value, Mapping) or set(value) - {
            "du_endpoint", "device_eid", "eid_index", "pool_eid"
        } or not {"du_endpoint", "device_eid"} <= value.keys():
            raise ValueError("remote_shared device requires du_endpoint/device_eid")
        result = cls(**value)
        _endpoint(result.du_endpoint)
        for eid in (result.device_eid, result.pool_eid):
            if eid and (len(eid) != 32 or any(c not in "0123456789abcdefABCDEF" for c in eid)
                        or int(eid, 16) == 0):
                raise ValueError("remote_shared requires real 32-digit hexadecimal EIDs")
        if not result.device_eid or type(result.eid_index) is not int or result.eid_index < -1:
            raise ValueError("remote_shared requires a device EID and valid EID index")
        return cls(result.du_endpoint, result.device_eid.lower(), result.eid_index, result.pool_eid.lower())


@dataclass(frozen=True)
class DSARemoteSharedConfig:
    store_endpoint: str
    devices: tuple[tuple[int, RemoteSharedDevice], ...] = ()
    single: RemoteSharedDevice | None = None

    @classmethod
    def from_value(cls, value):
        if value is None:
            return None
        if not isinstance(value, Mapping) or "store_endpoint" not in value:
            raise ValueError("remote_shared requires store_endpoint")
        _endpoint(value["store_endpoint"])
        if "devices" not in value:
            # 兼容已有DP1/TP1单EID文件，多rank不得静默复用它。
            single = RemoteSharedDevice.from_value({k: v for k, v in value.items() if k != "store_endpoint"})
            return cls(value["store_endpoint"], single=single)
        if set(value) != {"store_endpoint", "devices"} or not isinstance(value["devices"], Mapping):
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
        return cls(value["store_endpoint"], tuple(sorted(devices)))

    def validate_topology(self, dp: int, tp: int) -> None:
        if dp < 1 or tp not in (1, 2):
            raise ValueError("remote_shared supports TP1/TP2 and positive DP size")
        if dp * tp > 1 and not self.devices:
            raise ValueError("multi-rank remote_shared requires a physical-device table; single EID is TP1/DP1 only")
        if self.devices and len(self.devices) < dp * tp:
            raise ValueError("remote_shared device table is smaller than DP * TP")

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
        return phy_id, selected
