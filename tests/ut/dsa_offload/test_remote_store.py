# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import shutil
import socket
import subprocess
import time

import numpy as np
import pytest
import torch

from vllm_ascend.dsa_offload.remote_store import (
    PutDoneBatch,
    RemoteKvStore,
    RemotePackedArena,
    RemoteStoreShadow,
    RemoteWatermarks,
)


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


def test_store_shadow_real_control_lifecycle(tmp_path):
    """真实生产服务和 native binding；依赖未部署时跳过，不替换成 RPC mock。"""
    pytest.importorskip("kvcache_scatter_copy_custom_ops.custom_ops_lib")
    binary = shutil.which("urma_pool_server")
    if binary is None:
        pytest.skip("add the built urma_pool_server directory to PATH")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        server = subprocess.Popen([
            binary, "--backing", str(tmp_path / "backing"), "--store-ports", str(port),
            "--payload-layers", "2", "--requests", "8", "--source-capacity", "512",
            "--store-bytes", str(64 * 128 * 656), "--trace-lifecycle",
        ], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 10
            while "MEMPOOL_READY" not in log_path.read_text():
                assert server.poll() is None and time.monotonic() < deadline, log_path.read_text()
                time.sleep(0.05)
            endpoint = f"127.0.0.1:{port}"
            a = RemoteStoreShadow(RemoteKvStore(endpoint, payload_layers=2, capacity=512), 2)
            b = RemoteStoreShadow(RemoteKvStore(endpoint, payload_layers=2, capacity=512), 2)
            a.ensure_requests(["a", "b"], np.array([0, 1]))
            b.ensure_requests(["a"], np.array([0]))
            assert a.handles[0] != b.handles[0]
            handles = a.handles.clone()
            a.ensure_requests(["b", "a"], np.array([1, 0]))
            assert torch.equal(a.handles, handles)
            rows, pages = np.array([1, 0]), np.array([0, 0])
            src = np.array([1, 0], dtype=np.int32)
            raw = torch.arange(2 * 128 * 656, dtype=torch.int64).to(torch.uint8).reshape(2, 128, 1, 656)
            a.plan_dump(rows, pages)
            a.put_layer(0, raw, src, rows, pages, before_cache_write=False)
            assert a.watermarks.done.tolist() == [0, 0]
            a.put_layer(1, raw, src, rows, pages, before_cache_write=False)
            assert a.watermarks.done.tolist() == [1, 1]
            assert a.watermarks.bound.tolist() == [0, 0]
            with (tmp_path / "backing").open("rb") as stored:
                assert stored.read(raw.numel()) == raw.flip(0).numpy().tobytes()
            a.plan_dump(rows, pages + 1)
            for layer in range(2):
                a.put_layer(layer, raw, src, rows, pages + 1, before_cache_write=True)
            assert a.watermarks.done.tolist() == [2, 2]
            with pytest.raises(RuntimeError, match="prefix mismatch"):
                a.put_layer(0, raw, src, rows, pages + 1, before_cache_write=True)
            a.release(0)
            a.ensure_requests(["a"], np.array([0]))
            assert a.handles[0] != handles[0] and a.watermarks.done[0] == 0
            a.close()
            b.close()
            logs = log_path.read_text()
            assert "op=PUT_DONE phase=READY" in logs and "op=CLOSE phase=CLOSED" in logs
        finally:
            server.terminate()
            server.wait(timeout=35)
