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

## §R2 EI0006 卡点（2026-10-27）

**交付**（commit 1fd0d9471）：dry-run 通过；实跑在 profile_run 首次 HcclAllGather 处 EI0006 建链超时，remote-shared 路径未触达（Pool 侧仅 OPEN 无 PUT）。EP8 权重装载成功（每卡 61.3GB）。绿区修两缺陷：per-device store_endpoint（配置补全，符合 remote_store_lifecycle_validation.md 2026-10-20 节）+ DU sibling CPU 6→5（R1 派单命令缺陷）。裸 8 卡 HCCL 探针 8/8 PASS。数据出处：pingpong.md §R2 交付 + 绿区 /home/fjj/pp_r2b_*.log。
**判读**（蓝区）：阴性。增量：蓝区"DP4/TP2→EP8 可装 391G 模型"反推被实测证实（61.3GB/卡）；两修复均接受并入库（sibling 后续一律用 5；JSON 沿用 per-device store_endpoint 版）。EI0006 归框架/配置层（裸探针过 + 本机本地卸载分支同拓扑曾跑通至 decode，均实测）；最可疑机制为 4 EngineCore 进程树启动偏斜超 HCCL 建链超时（估计）。下一步：R3 对照实验。

## §R3 EI0006 定位与越过（2026-10-27）

**派单**（commit 见本节首次提交）：R3-A offload 关闭对照（同 DP4/TP2 eager BS1）→ R3-B offload 开 + HCCL_CONNECT_TIMEOUT=1800 → 仍 EI0006 则 R3-C（ASCEND_LAUNCH_BLOCKING=1 + HCCL_ENTRY_LOG_ENABLE=1）抓精确栈。通过标准写死于 pingpong.md §R3（R3-B 沿用 R2 的日志序列/rows=4/capacity report 标准）。
**交付**：待绿区回传。
**判读**（蓝区）：待定。

## §R4 作者新流程跑通优化版（2026-10-28）

**派单**（commit 见本节首次提交）：R3 派单 superseded。依据：① 作者 10-28 给新跑通流程（profile_pool_inference.py 一键双 Pool + 模型，graph+MTP graph、40K、BS1 为作者验证配置）并亲口确认 probe 需手工补后四卡 store_endpoint（绿区 R2 修复方向获作者证实）；② 算子仓 main 更新到 f63a5bfed（ScatterSFA 回退修 217.654→265.659μs 退化 + 新增 --shared-port-gather），绿区旧版 6e85590db 必须重编。R4-0 停旧 Pool + /dev/shm remount 384G（已授权）；R4-3 作者原配置首发（BS1），EI0006 则 R4-3b 加 HCCL_CONNECT_TIMEOUT=1800；R4-4 条件放大 BS16（40K×16 用户验收口径方向）。判读预案写死于 pingpong.md §R4。
**交付**：待绿区回传。
**判读**（蓝区）：待定。

**蓝区开发（R4 期间并行）**：simple_test.py 新增 metrics 汇总（每 rank TTFT/TPOT/吞吐 + rank0 全局合并 `<run_name>.metrics.json`，host 时钟、generate 返回后计算、零计时扰动；开关 VLLM_ASCEND_SMOKE_METRICS_SUMMARY，默认开）。纯增量，不影响 R4 执行；绿区下次 pull 后 R4-4/后续跑程自动带出 metrics 行。

## §R4 作者新流程跑通优化版 — 交付与判读（2026-10-29）

**交付**（commit 177f9acd3）：R4-0/1/2 全过（f63a5bf 重编、双 Pool shared-gather 就绪、384G shm、store 路由正确）；R4-3 在 load_model 阶段 MoE EP8 建链崩 EI0015（RootInfoDetect failed，7/8 worker，PLAN 后 2 秒）；R4-3b/4 条件不触发。附 R3-A 数据：offload=0 也崩（HCCL dispatch timeout EmptyQueue）。
**判读**（蓝区）：阴性，但 R3-A 证明故障与 remote-offload 无关，收敛到本分支 EP8 跨 DP HCCL 建链层；崩溃阶段三轮漂移（profile_run→dummy_run→load_model）指向 rank 时序竞争。最可疑：重 Pool 轮询线程（du-workers 7×8）与框架绑核叠加致建链期 CPU 饥饿（估计）；作者 va.env 内容未知（已向用户索取）。代理变更为 90.255.129.35:3128。下一步：R5 三级梯子（HCCL_CONNECT_TIMEOUT=1800 → 关 CPU 绑核 → HCCL 抓栈）。

**R5 修订**（同日）：用户确认作者（w00919836）就是在绿区本机跑通的——差分框架从"作者机 vs 绿区机"修正为"同机不同账号/环境"。R5 前置加 R5-0 作者现场取证（va.env / dp4tp2.json / scp-opt、scp-va 版本 / 成功 run 日志 / 作者 EID 表，全只读），并授权绿区：va.env 里发现未设的 HCCL 变量可直接补上当级重跑。
