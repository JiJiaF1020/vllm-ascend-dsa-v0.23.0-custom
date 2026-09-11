#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""单机默认 8 卡 DP4TP2EP8；通过 VLLM_ASCEND_SMOKE_* 环境变量覆盖配置，见 README.md。"""

from novel_dataset import chinese_23k
from simple_test import SmokeConfig
from simple_test import main_from_env as main

CONFIG = SmokeConfig(
    model_path="/home/models/GLM-5.2-w4a4c8-mxfp4_99",
    tensor_parallel_size=2,
    data_parallel_size=4,
    prompts=chinese_23k,
    profile_dir="./profiles/1p",
    result_dir="./results/1p",
)

if __name__ == "__main__":
    main(CONFIG, node_size=1)
