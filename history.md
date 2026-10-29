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

## §R5 EI0015 重试梯子 — 交付与判读（2026-10-29）

**交付**（commit 13ac84c36）：R5-A（HCCL_CONNECT_TIMEOUT=1800）、R5-B（+关绑核）均复现 EI0015×7 失败；R5-C 因 ASCEND_LAUNCH_BLOCKING 与 ACL graph 互斥即败（蓝区派单缺陷），ascend/log tgz 已留存。R5-0 取证：作者 va.env 无 HCCL 变量；作者参数 CHUNKED_PREFILL=1/BS16/MAX_MODEL_LEN=40960/du 19090 段/binding=none。
**判读**（蓝区）：阴性。超时与绑核两方向实测排除；"作者有 HCCL 调优"假说作废。蓝区比对 GitHub 原始仓后修正误判：作者 scp-va HEAD cac7e2c4e 即分支 tip，与绿区同一份代码；真正差异只剩 smoke 未提交本地改动（min_tokens=max_tokens、ignore_eos=True）+ 执行参数 + 执行上下文（容器 vs 宿主，未知）。下一步：R6 逐字复现作者现场（作者目录/环境/配置原样），成则差分二分，败则查机器环境漂移。

## §R6 逐字复现作者现场 — 交付与判读（2026-10-29）

**交付**（commit f50e3884d）：作者命令逐字复现（同 scp-opt/scp-va/va.env/dp4tp2.json/19090 段）仍 EI0015×7；首跑撞他人 write-bench 占 udmac1d1e3/19090（hold-999，未杀，自然释放后重跑）。补漏入库：宿主机无 CANN toolkit；驱动 25.6.rc2/固件 7.0.9.100.203；作者无 bash_history；作者 smoke 改动=min_tokens=max_tokens+ignore_eos=True、gpu_mem=0.97。
**判读**（蓝区）：阴性，但故障性质改写为间歇性——时间线"绿区败（10-27/28）→作者过（10-28 04:07）→绿区败（10-29）"，非单向漂移。头号嫌疑：16 卡共享机多租户干扰（10-29 新增他人 v0.30 容器 + 当场抓到他人占用我方 UB 设备）。下一步：R7 间歇性鉴别（占用取证 + HCCL 裸探针 + 复现×3），探针败即机器层实锤。

## §R7 间歇性鉴别 — 交付与判读（2026-10-30）

**交付**（commit 7b5805939）：8 卡 HCCL 裸探针 3 次 8/8 全败 EI0015（00:24/00:29/00:34），同脚本 10-27 曾 8/8 PASS；三次失败时 NPU 全空闲、无租户进程、无端口占用；R7-2 按预案未执行。线索：URMA 设备节点 d1e2~5 系 10-30 00:32 重建（e6 系仍 10-22 旧）。
**判读**（蓝区）：机器层实锤。代码/配置/流程全部排除（作者 10-28 04:07 同机同命令 RUN_OK 为证）。"多租户运行时干扰"修正为"机器在 10-28 04:07 后、10-29 05:22 前被改动且故障持续"。修复需机器管理员介入。下一步：R8 只读证据采集（dmesg/npu-smi health/error-code/uburma 节点/登录痕迹），给用户提交管理员用；恢复验收 = 探针 8/8 PASS → R6-1 复现。

## §R8 机器层证据采集 + R9 杀占用恢复验收（2026-10-30）

**R8 交付**（commit 44686ea）：6 组只读证据全部执行——udmac d1e2~d1e5 八个节点 01:48/01:49 批量重建（内核/驱动层周期性重建）；dmesg UDMA/ubase 层 01:45-01:46 高频活动；唯一 /dev/uburma holder = kvaf_cpu_service（udmac0d1e6, --port 28999, PID 182921）；npu-smi -t health/error-code 本机 25.6.rc2 不支持已用 npu-smi info 替代；8 卡 Health 全 OK/0% util/无进程。
**R8 补充**（commit 1aab4f9）：/etc/hccl_rootinfo.json 被重新生成（02:07:16，19496B 合法 JSON）→ 探针 EI0015 消失 → 前移 EI0020（NPU 网络端口 16666 被占用 8/8）；用户指示有人占用先不跑。
**R9 交付**（本 commit）：按用户授权杀占用容器 deepseek-v41-rc2-a5-20261029-175627（他人 vLLM 作业，27 个 VLLM 进程）→ 裸 HCCL 探针 8/8 PASS（04:34）→ R6 整网复现 RUN_OK model_exit=0（05:09，8 worker 全 DSA_SHARED_DRAIN_OK + DSA_STORE_CLOSED put_pages=389376）。首次 R6 曾遇 RaGetDevEidInfoNum code=1004（physical 0/2/7，瞬时竞争，EID 顺序/并发/pool 运行三态探测全 OK，重跑即过）。作者跑通现场逐字复现成功。

## §R9 跑通落袋 + §R10 规范化 bench 派单（2026-10-30）

**R9 交付**（commit 262d70a16）：用户授权杀占用容器（deepseek-v41-rc2-a5，他人 vLLM 作业占 NPU 16666 端口=EI0020 根因；rootinfo 空目录 02:07 重新生成=EI0015 根因）→ 探针 8/8 PASS → R6 逐字复现 RUN_OK（8 worker DRAIN_OK，put_pages=389376）。RaGetDevEidInfoNum code=1004 定性为瞬时竞争（重跑即过）。
**R9 判读**（蓝区）：阳性落袋，remote-offload 整网链路正式跑通。
**蓝区开发**：bench 口径入库（commit 717c396fc）——VLLM_ASCEND_SMOKE_BENCH=1 定长 decode（min_tokens=max_tokens, ignore_eos）+ vllm bench serve 风格报告块 + 全局分位数（per_request 明细重算 p50/p90/p99）+ request_throughput_rps + tpot_under_50ms 验收判定。mock 全过。
**R10 派单**：规范化 bench 首发：DP4/TP2/EP8、BS16/DP（64 请求）、40K 输入、定长 1024 输出、offload 开、graph+MTP graph。前置=占用取证+探针；LIM 命中则 BS 减半梯子（16→8→4）。判读预案写死。

## §R10 规范化 bench 首发 — 交付（2026-10-30 07:24~07:43）

**交付**（commit 待回填）：bench 首发失败 EI0006。前置核验全过（框架 0180ceb 含 717c396fc bench 口径；占用取证 8 卡空闲；探针 8/8 PASS；清场干净）。双 pool-a/b READY（POOL_ROI_POOL_READY groups=['a','b']，store 19080-19087/du 20000-20007），8 worker 全 DSA_EXECUTION_PLAN（权重装载 + MTP draft 加载成功），07:29:18 **8 worker 同时 EI0006**（Communication_Error_Get_Socket: Getting socket times out），栈 = 运行时 MoE 前向 `npu_dynamic_mx_quant → HcclAllGather`（EP8 all-gather）。工具 RC=0 但 summary 未生成、无 DRAIN_OK、无 metrics/results。EI0006 不在瞬时重跑白名单，按「其它新报错 → 回传」处置。日志：/home/fjj/a5-remote-offload-ops/tmp/r10-bench-bs16/ + /home/fjj/pp_r10_bench.log。
