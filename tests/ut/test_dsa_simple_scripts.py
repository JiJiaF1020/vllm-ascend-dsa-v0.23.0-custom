# SPDX-License-Identifier: Apache-2.0
"""Smoke configuration/launcher tests without vLLM or NPU dependencies."""

import argparse
import importlib
import io
import itertools
import json
import multiprocessing
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

DEMO = Path(__file__).resolve().parents[2] / "examples/dsa_demo"


def load_smoke():
    with patch.object(sys, "path", [str(DEMO), *sys.path]):
        return importlib.import_module("simple_test")


def inspect_spawn_config(config, queue):
    queue.put((config.enable_mtp_graph, load_smoke().build_llm_kwargs(config)))


class TestSimpleScripts(unittest.TestCase):
    def setUp(self):
        self.smoke = load_smoke()
        self.config = self.smoke.SmokeConfig(model_path="test-model")

    def test_dataset_and_entry_defaults(self):
        with patch.object(sys, "path", [str(DEMO), *sys.path]):
            dataset = importlib.import_module("novel_dataset")
            one = importlib.import_module("simple_test_1p").CONFIG
            two = importlib.import_module("simple_test_2p").CONFIG
        self.assertFalse(hasattr(dataset, "chinese_20k"))
        self.assertEqual(dataset.chinese_48k, [dataset.chinese_23k[0] * 2])
        self.assertEqual(dataset.chinese_64k, [(dataset.chinese_23k[0] * 3)[:92800]])
        self.assertEqual(dataset.chinese_10k, [dataset.chinese_23k[0][:15000]])
        self.assertEqual((one.tensor_parallel_size, one.data_parallel_size), (2, 4))
        self.assertEqual((two.tensor_parallel_size, two.data_parallel_size), (1, 16))
        for config in (one, two):
            self.assertEqual(config.prompts, dataset.chinese_23k)
            self.assertFalse(config.enable_profile)
            self.assertFalse(config.enable_chunked_prefill)
            self.assertTrue(config.mtp_graph_enabled)

    def test_configuration_matrix(self):
        for (tp, dp), offload, mode, mtp, graph in itertools.product(
            [(8, 1), (2, 4), (4, 2), (1, 16)],
            [False, True],
            ["cache-init", "eager", "graph"],
            [False, True],
            [False, True],
        ):
            with self.subTest(tp=tp, dp=dp, offload=offload, mode=mode, mtp=mtp, graph=graph):
                config = replace(
                    self.config,
                    tensor_parallel_size=tp,
                    data_parallel_size=dp,
                    enable_offload=offload,
                    run_mode=mode,
                    enable_mtp=mtp,
                    enable_mtp_graph=graph,
                    max_num_seqs=4,
                )
                kwargs = self.smoke.build_llm_kwargs(config, multi_node=True)
                additional = kwargs["additional_config"]
                self.assertEqual("dsa_sparse_config" in additional, offload)
                self.assertTrue(additional["enable_sparse_li_c8"])
                self.assertFalse(additional["enable_mc2_hierarchy_comm"])
                self.assertEqual("dp_allreduce_on_npu" in additional, dp > 1)
                self.assertEqual(kwargs["enforce_eager"], mode != "graph")
                self.assertEqual("compilation_config" in kwargs, mode == "graph")
                self.assertEqual("speculative_config" in kwargs, mtp)
                if mode == "graph":
                    compilation = kwargs["compilation_config"]
                    self.assertEqual(compilation["cudagraph_mode"], "FULL_DECODE_ONLY")
                    self.assertEqual(compilation["cudagraph_capture_sizes"], [4, 8, 16] if mtp else [1, 2, 4])
                if mtp:
                    spec = kwargs["speculative_config"]
                    self.assertEqual(spec["draft_tensor_parallel_size"], tp)
                    self.assertEqual(spec["enforce_eager"], not (offload and graph and mode == "graph"))

    def test_environment_overrides(self):
        values = {
            "model_path": "/models/custom",
            "tensor_parallel_size": "1",
            "data_parallel_size": "16",
            "max_num_seqs": "8",
            "enable_offload": "false",
            "enable_mtp_graph": "0",
            "gpu_memory_utilization": "0.83",
            "graph_capture_sizes": "1, 2, 8",
            "enable_profile": "yes",
            "profile_dir": "/tmp/profile",
            "result_dir": "none",
            "prompts": '["你好", "hello"]',
            "enable_chunked_prefill": "on",
            "max_num_batched_tokens": "16384",
        }
        with patch.dict(os.environ, {"VLLM_ASCEND_SMOKE_" + k.upper(): v for k, v in values.items()}, clear=True):
            self.assertEqual(self.smoke.read_smoke_environment(), values)
            config = self.smoke.config_from_environment(self.config, values)
        self.assertEqual(config.tensor_parallel_size, 1)
        self.assertEqual(config.data_parallel_size, 16)
        self.assertEqual(config.max_num_seqs, 8)
        self.assertFalse(config.enable_offload)
        self.assertFalse(config.enable_mtp_graph)
        self.assertEqual(config.gpu_memory_utilization, 0.83)
        self.assertEqual(config.graph_capture_sizes, (1, 2, 8))
        self.assertEqual(config.prompts, ["你好", "hello"])
        self.assertTrue(config.enable_profile)
        self.assertIsNone(config.result_dir)
        self.smoke.build_llm_kwargs(config)

    def test_environment_dataset_and_rejections(self):
        with patch.object(sys, "path", [str(DEMO), *sys.path]):
            for name in ("short", "chinese_10k", "chinese_23k", "chinese_48k", "chinese_64k"):
                config = self.smoke.config_from_environment(self.config, {"dataset": name})
                self.assertEqual(len(config.prompts), 1)
            for values in (
                {"dataset": "typo"},
                {"dataset": "short", "prompts": '["hello"]'},
                {"enable_offload": "maybe"},
                {"tensor_parallel_size": "1.5"},
                {"graph_capture_sizes": "1,,8"},
                {"prompts": '"not a list"'},
            ):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    self.smoke.config_from_environment(self.config, values)
        with patch.dict(os.environ, {"VLLM_ASCEND_SMOKE_MTYPO": "1"}, clear=True), self.assertRaises(ValueError):
            self.smoke.read_smoke_environment()

    def test_node_environment_and_cli_precedence(self):
        values = {
            "VLLM_ASCEND_SMOKE_NODE_RANK": "1",
            "VLLM_ASCEND_SMOKE_MASTER_ADDR": "192.0.2.1",
            "VLLM_ASCEND_SMOKE_MASTER_PORT": "29501",
            "VLLM_ASCEND_SMOKE_SYNC_PORT": "29601",
            "VLLM_ASCEND_SMOKE_DATA_PARALLEL_SIZE": "16",
            "VLLM_ASCEND_SMOKE_ENABLE_MTP_GRAPH": "1",
        }
        with (
            patch.dict(os.environ, values, clear=True),
            patch.object(sys, "argv", ["smoke", "--node-rank", "0", "--mtp-eager"]),
            patch.object(self.smoke, "run_node_dp") as launch,
        ):
            self.smoke.main_from_env(self.config, node_size=2)
        config, args = launch.call_args.args
        self.assertEqual(config.data_parallel_size, 16)
        self.assertFalse(config.enable_mtp_graph)
        self.assertEqual(
            (args.node_rank, args.master_addr, args.master_port, args.sync_port), (0, "192.0.2.1", 29501, 29601)
        )

    def test_all_config_fields_have_registered_overrides(self):
        from dataclasses import fields

        values = {"VLLM_ASCEND_SMOKE_" + field.name.upper(): "test" for field in fields(self.config)}
        with patch.dict(os.environ, values, clear=True):
            self.assertEqual(set(self.smoke.read_smoke_environment()), {field.name for field in fields(self.config)})

    def test_topology_and_prompt_assignment(self):
        for nodes, dp, tp in [(1, 1, 8), (1, 4, 2), (1, 16, 1), (2, 16, 1), (2, 4, 2)]:
            config = replace(
                self.config, data_parallel_size=dp, tensor_parallel_size=tp, max_num_seqs=3, prompts=["a", "b"]
            )
            global_ranks = []
            for node in range(nodes):
                args = argparse.Namespace(
                    node_size=nodes, node_rank=node, master_addr="192.0.2.1", master_port=29500, sync_port=29600
                )
                _, _, sync_port, ranks = self.smoke.resolve_node_launch(config, args)
                self.assertEqual(len(ranks), dp // nodes)
                self.assertEqual(sync_port, None if nodes == 1 else 29600)
                global_ranks.extend(rank for _, rank in ranks)
            self.assertEqual(global_ranks, list(range(dp)))
            requests = [item for rank in global_ranks for item in self.smoke.shard_prompts(config, rank)]
            self.assertEqual(requests, [(i, ["a", "b"][i % 2]) for i in range(dp * 3)])

    def test_invalid_configuration(self):
        for changes in (
            {"run_mode": "disabled"},
            {"tensor_parallel_size": 0},
            {"prompts": []},
            {"max_num_batched_tokens": 4096},
            {"enable_expert_parallel": False},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.smoke.build_llm_kwargs(replace(self.config, **changes))
        args = argparse.Namespace(node_size=2, node_rank=1, master_addr="", master_port=29500, sync_port=29600)
        with self.assertRaises(ValueError):
            self.smoke.resolve_node_launch(self.config, args)
        args.master_addr = "192.0.2.1"
        args.sync_port = args.master_port
        with self.assertRaises(ValueError):
            self.smoke.resolve_node_launch(self.config, args)

    def test_dry_run_has_no_runtime_or_filesystem_side_effects(self):
        with tempfile.TemporaryDirectory() as folder:
            config = replace(
                self.config, enable_profile=True, profile_dir=folder + "/profile", result_dir=folder + "/result"
            )
            output = io.StringIO()
            environment = dict(os.environ)
            with (
                patch.object(sys, "argv", ["smoke", "--dry-run"]),
                redirect_stdout(output),
                patch.object(self.smoke, "run_node_dp") as launch,
            ):
                self.smoke.main(config, node_size=1)
            launch.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])
            self.assertEqual(dict(os.environ), environment)
            self.assertEqual(json.loads(output.getvalue())["required_local_npus"], 8)

    def test_mtp_eager_override_survives_spawn(self):
        with patch.object(sys, "argv", ["smoke", "--mtp-eager"]), patch.object(self.smoke, "run_node_dp") as launch:
            self.smoke.main(self.config, node_size=1)
        config = launch.call_args.args[0]
        self.assertFalse(config.enable_mtp_graph)
        self.assertTrue(self.config.enable_mtp_graph)
        # Real spawn serializes CONFIG rather than re-reading the entry's globals.
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        with patch.object(sys, "path", [str(DEMO), *sys.path]):
            process = context.Process(target=inspect_spawn_config, args=(config, queue))
            process.start()
            try:
                enabled, kwargs = queue.get(timeout=20)
                process.join(timeout=20)
                self.assertEqual(process.exitcode, 0)
                self.assertFalse(enabled)
                self.assertTrue(kwargs["speculative_config"]["enforce_eager"])
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
                queue.close()
                queue.join_thread()

    def test_cache_init_does_not_generate(self):
        fake_vllm = Mock()
        barrier = Mock()
        config = replace(self.config, run_mode="cache-init", enable_offload=False, result_dir=None)
        with (
            patch.dict(sys.modules, {"vllm": fake_vllm}),
            patch.dict(os.environ),
            patch.object(self.smoke, "write_result") as result,
            redirect_stdout(io.StringIO()),
        ):
            self.smoke.run_dp_rank(config, 1, 1, "127.0.0.1", 29500, barrier, None)
            self.assertEqual(os.environ["VLLM_DP_RANK"], "1")
        fake_vllm.LLM.return_value.generate.assert_not_called()
        barrier.wait.assert_called_once()
        self.assertFalse(result.call_args.args[1]["offload"])
        self.assertNotIn("dsa_sparse_config", fake_vllm.LLM.call_args.kwargs["additional_config"])


if __name__ == "__main__":
    unittest.main()
