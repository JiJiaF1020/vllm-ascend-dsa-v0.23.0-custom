# History

蓝绿协作轮次档案。只追加，不修改。每轮至少记录：派单 commit、执行环境、原始数据出处、判读结论与置信度。

## 轮次记录格式

```
## §R<n> <标题>（<日期>）

**派单**（commit <id>）：<一句话任务 + 判读通过标准>
**交付**（commit <id>）：<一句话结果 + 数据出处>
**判读**（蓝区）：<阳性/阴性/无效 + 依据 + 下一步>
```

## §R1 remote-offload 单机整网首发（2026-10-26）

**派单**（commit 见本节首次提交）：框架安装（含 catlass 本机拷贝修复）→ 算子仓构建 → 8 卡 EID 探测 → embedded Pool → DP1/TP1 eager BS1 实跑；全绿则放大 DP4/TP2。通过标准写死于 pingpong.md §R1：capacity report 须含 `indexer=21, resident=57, mtp=1`，日志须依次命中 EXECUTION_PLAN/STORE_PUT_DONE/CONNECT_BEGIN/CONNECT_OK(real_store)/≥2 轮 CONSUMED/DRAIN_OK 非零，进程 RC=0。
**交付**：待绿区回传。
**判读**（蓝区）：待定。

背景存档：算子仓公开镜像 `JiJiaF1020/a5-remote-offload-ops`（main=6e85590db）；本分支公开镜像在 `JiJiaF1020/vllm-ascend-dsa-v0.23.0-custom` 的 `vllm-ascend-v0.23.0-custom-remote-offload`。LIM 1..4 行校验在本分支未变（csrc/.../vllm_a5_li_manage_c8_torch_adpt.h:98-100），首发 eager + BS1 规避；切 graph 前须另行核对 capture 档。

## §R1 判读与收尾（2026-10-27）

**交付**（commit 9b09cf51e）：B/C/D/E1/E2 通过；E3 阻塞——GLM-5.2 权重 391GB 在 TP1/DP1 及 TP2/DP1（EP2）下物理放不进 96GB/卡，NPU OOM，非链路/驱动/代码 bug。数据出处：pingpong.md §R1 交付 + 绿区 /home/fjj/pp_r1_*.log。
**判读**（蓝区）：链路部分阳性（安装/构建/EID 探测/Pool/dry-run 全部实测通过）；整网推理路径未触达（模型未装载，未走到 CONNECT/CONSUMED）。绿区"F（DP4/TP2）必然 OOM"结论不采纳：EP=tp×dp（simple_test.py:40,549），DP4/TP2→EP8，每卡权重 ≈54GB 可装——置信度：反推（代码+量化算术），R2 实测验证。D 阶段分组偏差（占位符 UB_A/UB_B→udmac1d1e2/udmac0d1e3）认定为蓝区派单缺陷，绿区处置接受，分组沿用。TP8 硬拒确认为本分支硬约束（remote_shared_config.py:161）。下一步：R2 直接 DP4/TP2/EP8 首发。

## §R2 DP4/TP2（EP8）整网首发（2026-10-27）

**派单**（commit 见本节首次提交）：复用 R1 Pool（pid 2167590），eager BS1，DP4/TP2（EP8）跑 chinese_23k。通过标准写死于 pingpong.md §R2：dry-run RC=0；实跑 RC=0 有生成文本；日志序列 EXECUTION_PLAN→STORE_PUT_DONE→CONNECT_BEGIN→CONNECT_OK(real_store)→≥2 轮 CONSUMED→DRAIN_OK 非零；CONNECT rows=4；capacity report 原文记录（数值偏差蓝区判）。
**交付**：待绿区回传。
**判读**（蓝区）：待定。
