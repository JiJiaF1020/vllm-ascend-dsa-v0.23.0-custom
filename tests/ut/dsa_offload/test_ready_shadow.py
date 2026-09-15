# SPDX-License-Identifier: Apache-2.0
"""验证实际shadow采集器的快照所有权、group生命周期和失败汇总；不模拟NPU kernel。"""
from types import SimpleNamespace
import unittest

import torch

from vllm_ascend.dsa_offload.config import DSAReadyShadowConfig
from vllm_ascend.dsa_offload.ready_shadow import ReadyShadow


class TestReadyShadow(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(_forward_epoch=1)
        self.shadow = ReadyShadow(DSAReadyShadowConfig(steps=2), self.runtime, 0)
        self.shadow.register_layer(0, 0, None)
        self.shadow.register_layer(1, None, 0)
        self.shadow.register_layer(2, 1, None)
        self.shadow.register_layer(3, None, 2)
        self.context = SimpleNamespace(layer_id=0)
        self.mapping = torch.full((1, 8193), -1, dtype=torch.int32)
        self.mapping[0, :4096] = torch.arange(4096)
        self.mapping[0, -1] = 4096

    def begin(self):
        self.shadow.before_selection(self.context, self.mapping, torch.tensor([0]),
                                     torch.tensor([2]), torch.tensor([4]))

    def test_selection_snapshot_precedes_mutation_and_is_independent(self):
        self.begin()
        self.mapping[0, 7] = -1
        self.assertEqual(int(self.shadow.selected["mirror_before"][7]), 7)
        self.assertEqual(self.shadow.layers, [0, 1])
        self.assertEqual(self.shadow.attempts, 1)

    def test_first_fill_and_short_mtp_do_not_count_as_passed(self):
        self.mapping[0, -1] = -4096
        self.begin()
        self.assertIsNone(self.shadow.selected)
        self.assertEqual(self.shadow.skips, {"first_fill": 1})
        self.assertFalse(self.shadow.report()["passed"])

    def test_shared_layers_use_own_payload_and_no_global_cache_clone(self):
        self.begin()
        ids = torch.arange(2048).reshape(1, 1, -1).repeat(4, 1, 1).int()
        self.runtime._a5_attention_src_ids = ids
        self.runtime._a5_attention_slots = torch.cat((ids, torch.full((4, 1, 256), -1)), dim=2).int()
        self.runtime.resident_token_pool = SimpleNamespace(get_cache_slots=lambda _: self.mapping)
        # 真实源token5000，目标slot3000；校验抓到的是KSC之前的字节。
        self.runtime._a5_attention_src_ids[:, 0, 0] = 5000
        outputs = SimpleNamespace(topk_index=torch.tensor([[[5000]]]),
                                  topk_slots=torch.tensor([[[3000]]]), miss_count=torch.tensor([1]))
        pages = torch.arange(34).reshape(1, -1).int()
        dram_table = torch.ones((1, 64), dtype=torch.int32)
        packed = torch.zeros((40, 128, 1, 656), dtype=torch.int8)
        for layer, value in ((0, 17), (1, 23)):
            arena = torch.full((2, 128, 1, 656), value, dtype=torch.int8)
            self.shadow.before_copy(SimpleNamespace(layer_id=layer), packed, pages, arena, dram_table, outputs)
        packed.fill_(99)
        for layer, value in ((0, 17), (1, 23)):
            sample = self.shadow.pending[layer]
            self.assertEqual(sample["packed_before"].shape[0], 34)
            self.assertEqual(int(sample["packed_before"].max()), 0)
            self.assertTrue(torch.all(sample["payload"] == value))

    def test_failure_is_reported_after_inference_and_coverage_is_required(self):
        self.shadow._failure("test", 0, "mismatch")
        self.assertFalse(self.shadow.report()["passed"])
        self.shadow.counts = {0: 2, 1: 2}
        self.assertFalse(self.shadow.report()["passed"])
        self.shadow.errors.clear()
        self.assertTrue(self.shadow.report()["passed"])


if __name__ == "__main__":
    unittest.main()
