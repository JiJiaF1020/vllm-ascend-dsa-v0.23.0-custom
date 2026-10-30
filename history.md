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

**交付**（commit 3b84df4）：bench 首发失败 EI0006。前置核验全过（框架 0180ceb 含 717c396fc bench 口径；占用取证 8 卡空闲；探针 8/8 PASS；清场干净）。双 pool-a/b READY（POOL_ROI_POOL_READY groups=['a','b']，store 19080-19087/du 20000-20007），8 worker 全 DSA_EXECUTION_PLAN（权重装载 + MTP draft 加载成功），07:29:18 **8 worker 同时 EI0006**（Communication_Error_Get_Socket: Getting socket times out），栈 = 运行时 MoE 前向 `npu_dynamic_mx_quant → HcclAllGather`（EP8 all-gather）。工具 RC=0 但 summary 未生成、无 DRAIN_OK、无 metrics/results。EI0006 不在瞬时重跑白名单，按「其它新报错 → 回传」处置。日志：/home/fjj/a5-remote-offload-ops/tmp/r10-bench-bs16/ + /home/fjj/pp_r10_bench.log。

## §R10 判读 + §R11 EI0006 隔离梯子派单（2026-10-30）

**R10 判读**（蓝区）：EI0006 = 8 worker 同一秒 socket 超时齐射 = 有 rank 未按时到达运行时 MoE 前向 EP8 all-gather（07:26:45 draft 就绪 → 07:29:18 齐射，~153s≈HCCL socket 超时量级）。不判机器层复发（探针 8/8、占用干净）。差分 R9 RUN_OK：R10 独有变量 = run_mode=graph+MTP graph 首次上机（R9 为 eager，主嫌疑）、数据面 udmac1d1e2/20000 段/eid-probe-r1 json（次嫌疑）、BENCH=1 仅 sampling 参数（嫌疑最低）。bench 基线未落袋。
**R11 派单**：隔离梯子——R11-0 只读取证（model.log 找没到的 rank + capture 标记 + URMA 行）→ R11-A 原样重跑（EI0006 入一次性白名单）→ R11-B RUN_MODE=eager 隔离 graph（过=eager bench 基线落袋）→ R11-C graph+ENABLE_MTP_GRAPH=0 分离 acl graph vs MTP graph → R11-D 回作者逐字配置验证数据面/机器漂移。白名单外新报错即停回传。
## §R11 隔离梯子（2026-10-30）

**派单**（d6bf390）：对 R10 EI0006 做隔离梯子，排除瞬时/graph/数据面/机器漂移。白名单=RaGetDevEidInfoNum 1004/EI0015/EI0020/EI0006 一次性重跑（仅 R11-A）。
**交付**（commit 3b84df4）：R11-A 原样重跑 EI0006 稳定复现（8 worker 08:26:50 齐射）→瞬时排除；R11-B eager EI0006 仍复现（8 worker 08:44:26 齐射, acl_graph 计数=0）→ graph 非触发面，跳过 R11-C；R11-D 作者逐字配置 2 跑 EI0006=0 但前移 DP1 专属 RaGetDevEidInfoNum code=1004（aiv_urma_query_eids, KV-cache 绑定阶段, 非瞬时），0 DRAIN，未到 generate。无 bench 通过步。数据出处：pp_r11a/b/d.log + tmp/r11a-bench-bs16/ + tmp/r11b-bench-bs16-eager/ + /home/w00919836/scp-opt/tmp/prefault-read-r11d/。

## §R11 交付判读 + §R12 归因拆解派单（2026-10-30）

**R11 交付**（commit 923d852ee）：梯子全执行——R11-A EI0006 稳定复现（非瞬时）；R11-B eager 同签名复现（acl_graph 计数=0）→ graph 面排除；R11-C 跳过；R11-D 作者配置 2 跑 EI0006=0 但前移 DP1 专属 1004（aiv_urma_query_eids，physical 2/3，两跑复现非瞬时），0 DRAIN。
**R11 判读**（蓝区）：① R10 主嫌疑 graph 证伪，自我修正；② EI0006 归因 fjj 数据面配置（e2/20000 段/eid-probe-r1 json，三选一未拆），fjj 配置 3/3 崩、作者配置 0 崩；③ 1004 与 R9 首跑交集 physical 2 + R7/R8 节点重建证据 → NPU 2/3 UB/EID 状态面嫌疑上升；④ 1004「重跑即过」修正为「取决于机器 UB 面状态，上限 2 次」。
**R12 派单**：R12-0 只读取证（EID 探测×3 轮重点 2/3 + uburma 节点 + dmesg + npu-smi + 端口快照）；R12-1 主实验 = fjj 框架仓 + BENCH=1 + 作者数据面配置逐字（scp-opt + dp4tp2.json + udmac1d1e3/e3 + 19090 段 + --framework-repo 混用），过则 EI0006 归因数据面且 bench 基线同轮落袋，复现则转 fjj 代码 diff。须先核验 vllm_ascend import 指向 fjj 仓。

## §R12 EI0006 归因拆解（fjj 框架+BENCH=1+作者数据面）（2026-10-30）

**派单**（commit 982340a）：R12-0 只读取证（NPU 2/3 UB/EID 状态面）→ R12-1 主实验 = fjj 框架仓 + BENCH=1 + 作者数据面配置逐字，过则 数据面归因且 bench 基线同轮落袋，复现则转 fjj 代码 diff。白名单=1004（≤2 次）；EI0006 复现即停回传。

**交付**（commit 309113d）：R12-0 五块全过（EID 两轮探测干净无 1004、UB 节点 09:42 批量重建与 R8 同型、端口全空闲）；R12-1 EI0006 复现（10:02:15 齐射，EI0006×20，0×1004，崩溃在 _initialize_kv_caches → determine_available_memory 内 MoE 前向 HcclAllGather，与 R10/R11 同签名同相位）→ 数据面配置排除，指向 fjj 框架/scp-va 之外代码或 smoke env 差异。数据出处：/home/fjj/pp_r12.log + tmp/r12-fjj-bench-authorplane/raw/model.log。

**判读**（蓝区）：待定（蓝区做 fjj vs scp-va 代码 diff）。

## §R12 交付判读 + §R13 绑核直测派单（2026-10-30）

**R12 交付**（commit 309113df0/e70df42ea）：R12-0 取证——NPU 2/3 EID 健康（19 EID×2 轮）、udmac 节点 09:42 周期性重建、dmesg 无错误码。R12-1 主实验（fjj 框架+BENCH=1+作者数据面）EI0006 复现（10:02:15 齐射，0×1004，0 DRAIN）→ 数据面配置排除。
**R12 判读**（蓝区）：第二次自我修正（数据面排除）；决定性矩阵收敛为 smoke env 集合 × 框架代码双变量。本地全量代码 diff（cac7e2c4e→HEAD）仅 bench 口径（simple_test.py +155/envs.py +4），不在 init 路径；**发现 enable_cpu_binding 默认 True**（simple_test.py:42/288→ascend_config.py:171→worker.py:786 bind_cpus→cpu_binding.py:792-798 线程绑核+NPU IRQ smp_affinity 机器级重定向），作者 scp-va 为 binding=none（R5-0），共享机争核致 HCCL 首连超时，机制吻合，升头号嫌疑。共享机纪律入库：fjj 仓 smoke 必须显式 ENABLE_CPU_BINDING=0。
**R13 派单**：R13-1 = R12-1 逐字 + VLLM_ASCEND_SMOKE_ENABLE_CPU_BINDING=0（头号嫌疑直测，过则根因坐实+bench 基线同轮落袋，须回传绑核关闭 grep 核验）；R13-2 = env 集合二分（BENCH=0 其余保留）预案。

## §R13 绑核嫌疑直测 + env 集合二分（2026-10-30）

**派单**（commit b34e7a3）：R13-1 = R12-1 逐字 + 唯一变量 ENABLE_CPU_BINDING=0（头号嫌疑直测），过则 绑核根因坐实 + bench 基线落袋；EI0006 复现 则 R13-2 env 二分（BENCH=0）。白名单=1004。

**交付**（commit 34a2b10）：R13-1 关绑核 EI0006 复现（10:54:05 齐射×20），绑核关闭核验通过（enable_cpu_binding=False 生效，无绑核执行痕迹）→ 绑核排除；R13-2 BENCH=0 仍 复现（11:03:16 ×20）→ bench env/代码路径排除，触发面在基础 env 集合或更底层。两轮均 0×1004，0 DRAIN，崩溃相位同签名（KV-cache init MoE 前向 HcclAllGather）。数据出处：/home/fjj/pp_r13.log + pp_r13b.log + tmp/r13-nobind-bench/ + tmp/r13-envbis-nobench/。

**判读**（蓝区）：待定（给定下一轮 逐项 unset 二分表）。

## §R13 交付判读 + §R14 env机制vs配置值正交派单（2026-10-30）

**R13 交付**（commit 34a2b10d5）：R13-1 关绑核（核验干净：additional_config False+无执行痕迹）EI0006 仍复现→绑核排除（第三次自我修正）；R13-2 BENCH=0 仍复现→bench env/代码排除；崩溃相位恒为 determine_available_memory 的 MoE EP8 all-gather 首连。0/7 轮到 generate。
**R13 判读**（蓝区）：发现配置层混淆——入口 simple_test_1p.py 无 env 时默认 BS1/23k/25600/chunked-off/0.85，而作者 diff 只加 7 开关不含 BS16/40960/chunked1；R5-0 记录的作者大参数只能来自其 shell 导出（bash_history）。「env 相关性」可能实为「配置尺寸相关性」；头号嫌疑转 MAX_NUM_BATCHED_TOKENS=40960（profile run 直接输入）。NATIVE_RUNTIME_ENV_OVERRIDES 无条件施加，非差分。
**R14 派单**：R14-0 只读（作者 bash_history smoke env 痕迹+scp-va 现状 diff+1p 脚本原文——判定 env 机制 vs 配置值的钥匙）；R14-1 正交实验=scp-va 代码+我们 env 值全集+作者数据面（禁导 BENCH，作者 envs.py 不认识会报错），崩=配置值轴/过 profile=代码轴；R14-2 预写五步逐项 unset 二分（batched_tokens→model_len→chunked→bs→gpu_mem）。

## §R14 env机制 vs 配置值正交拆解 + 逐项 unset 二分（2026-10-30）

**派单**（commit 3c4018f）：R14-0 只读取证（作者 bash_history 的 smoke env 导出痕迹=判定 env 机制 vs 配置值的钥匙）；R14-1 代码正交实验（scp-va 代码+我们 env 值全集+作者数据面），崩=配置值轴，过=代码轴；R14-2 五步逐项 unset 二分表预写，命中即停。EI0006 不在白名单。

**交付**（commit c526a38）：R14-0 ：/home/w00919836/.bash_history 不存在，实际 HOME=/root，/root/.bash_history 含 VLLM_ASCEND_SMOKE=0 命中（全部为装机/克隆命令），作者无 smoke env 导出痕迹；scp-va 未提交 diff=只加 profile_dir/7开关/gpu_mem0.97，不含 BS16/40960/chunked1。R14-1（scp-va+env 值全集+作者数据面）：EI0006 复现（11:34:45 齐射×20，0×1004，崩在 worker.py:564 determine_available_memory MoE 前向）→ 配置值轴坐实（代码轴排除）。R14-2 五步逐项 unset（batched_tokens→model_len→chunked→seqs→gpu_mem）全部 EI0006 复现（各×20，0×1004，同崩相位），5 候选配置值均非独立元凶。数据出处：/home/fjj/pp_r14.log + pp_r14_step{1..5}.log + tmp/r14-va-env-authorplane-step{1..5}/raw/model.log。

**判读**（蓝区）：待定（配置值轴坐实但非单一 env 变量，建议全部 unset 到纯默认对照或核查 scp-opt vs scp-va 残余差异）。

## §R14 交付判读 + §R15 MODEL_PATH 轴派单（2026-10-30）

**R14 交付**（commit c526a3860）：R14-0——作者 /root/.bash_history 无 smoke env 导出，scp-va diff 仍仅 2 文件 15 行（无大配置字段）⇒ 作者成功跑程=入口默认小配置。R14-1——scp-va 代码+我们 env 值全集+作者数据面 EI0006 复现（11:34:45）⇒ 配置值轴坐实、代码轴排除。R14-2——五步逐项 unset（batched_tokens/model_len/chunked/BS/gpu_mem）全复现 EI0006，无单一元凶；step1 profile 形状与 R11-D 相同仍崩。
**R14 判读**（蓝区）：剩余 4 个从未 unset 的 env 中，DATASET/MAX_TOKENS/ENABLE_PROFILE 本地判 inert（DSA budget 全来自默认值，profile 不用真实 prompts）→ 唯一活嫌疑 = MODEL_PATH（/data/model/...-A5-0731 vs R11-D 作者默认 /home/models/GLM-5.2-w4a4c8-mxfp4_99，可能不同 ckpt）；时间线旁证排除纯机器时段假说（R11-D 被崩溃轮夹在中间）。bench 0/8 到 generate。
**R15 派单**：R15-0 只读（两 ckpt config.json diff + 数据集 prompt 数——免费决定性）；R15-1 今日重放 R11-D（机器状态对照，EI0006=机器劣化转取证路线）；R15-2 R14-1 只留 MODEL_PATH（过=三 add-back 梯子；崩=依 R15-0 换 ckpt 路径一刀分 MODEL_PATH vs env 机制）。

## §R15 MODEL_PATH 轴排除 + 双对照（2026-10-30）

**派单**（commit a2f2353）：R15-0 只读取证（两 ckpt config.json diff + 数据集 prompt 数）；R15-1 今日重放 R11-D（机器状态对照，EI0006=机器劣化转取证）；R15-2 R14-1 只留 MODEL_PATH（过=三 add-back 梯子；崩=依 R15-0 换 ckpt 路径一刀分）。白名单=1004（≤2 次重跑）。

**交付**（commit a2f235346114c9a3b7ae59b7f1c5cc606fac1c4c）：R15-0：`/home/models/GLM-5.2-w4a4c8-mxfp4_99` 在本机/容器均不存在（CONFIG_MISSING_A），「两 ckpt 可能不同」前提失效；所有实际运行（R11-D/R14-1/R15-1/R15-2）经 va.env 覆盖为同一 `/data/model/...A5-0731/`；数据集 chinese_23k/40k 各仅 1 条超长 prompt。R15-1（va.env 全 env，EI0006=0）：DP1 专属 1004×4 + EngineCore patch_shm_broadcast 挂起，最终 Engine core initialization failed，与 R11-D 09:02 原跑（09:06:14 失败）签名一致 ⇒ 机器未劣化，R11-D 配方仍有效。R15-2（只留 MODEL_PATH，其余 15 个 smoke env 全 unset）：EI0006×13（13:16:09，HcclAllGather，worker.py:564，1004=0/RUN_OK=0）⇒ **MODEL_PATH 值排除（两轮同值仅 env 集不同）**，只剩 env 机制轴 → 蓝区定 R16。**执行要点**：蓝区称「va.env 已 unset 其余 9 个」不实，va.env 只 unset 4 项后 re-export 大部分（source 后残留 12 个 smoke env）；已显式 unset 全部 VLLM_ASCEND_SMOKE_* 兑现「只留 MODEL_PATH」。数据出处：/home/fjj/pp_r15a.log + pp_r15b.log + tmp/r15-r11d-replay/ + tmp/r15-only-modelpath/。

## §R15 交付判读 + §R16 va.env 融合配方派单（2026-10-30）

**R15 交付**（commit 84c304e45）：R15-0——/home/models/...99 不存在，va.env 第 23 行统一 MODEL_PATH=/data/model/...A5-0731 ⇒ 两 ckpt 前提失效；23k/40k 数据集各仅 1 条 prompt。R15-1——今日重放 R11-D 复现同签名（EI0006=0、DP1 1004×4、hang）=机器未劣化。R15-2——仅留 MODEL_PATH（source 后显式 unset 其余 15 个）仍 EI0006×13。
**R15 判读**（蓝区）：两前提坍塌——MODEL_PATH 轴排除；**va.env 本身 export 12 个 smoke env**（R12-1"unset 全部"系误报）⇒ R11-D 非零 env 组，"env 集合"轴坍塌；真实分界线 = va.env 独有变量集合 D（R15-1 D 在→profile 过；R15-2 D 不在→profile 崩；R9 RUN_OK 同为 va.env 配方）。流程教训：环境断言必须以 env|grep/文件原文为准，口头确认不作数（R12~R15 四轮梯子的错误前提，蓝绿各半责）。
**R16 派单**：R16-0 cat va.env 原文+残量清单+HCCL grep（D 集合内容，免费决定性）；R16-1 融合配方=source va.env 后覆盖我们 10 个值（scp-va+作者数据面），过 profile ⇒ 配方成立；R17 预案=fjj 仓+BENCH=1 同配方落 bench 基线。

## §R16 va.env 原文取证 + 融合配方验证（2026-10-30）

**派单**（commit 02e7288）：R16-0 只读取证（va.env 全文/D 集合/HCCL grep）；R16-1 融合配方验证（source va.env + 我们的 10 值覆盖，scp-va+作者数据面）；R17 预案=fjj 仓+BENCH=1 同配方落 bench 基线。

**R16-0**：va.env 原文——13 个 smoke env（RUN_MODE=graph、MTP 3/1/1、DP4/TP2、MAX_NUM_BATCHED_TOKENS=2048、MAX_TOKENS=256、MODEL_PATH 尾斜杠、ASCEND_RT_VISIBLE_DEVICES=0-7）；D 集合无 DSA_RESIDENT_BUDGET/MAX_ACTIVE_REQS/PROMPT_BUDGET_THRESHOLDS/INDEXER_MLA_BLOCK_RATIO/GRAPH_CAPTURE_SIZES；非 smoke 独有 `VLLM_ASCEND_DSA_HOST_MSTX=0`（DSA 前缀）；**无 HCCL 变量**。

**R16-1**（交付 commit 12866a46b81d566486a6570886f5c8042384f0a2）：**EI0006（配方不成立）**——容器内跑（宿主无 vllm 修正），8 worker 权重装载成功（13:51:20，389.88 GiB）后 13:54:49 DP1_TP1_EP3 HcclAllGather 超时 EI0006，model.log 20 处，1004=0，summary RUN_FAILED。命中预案「EI0006 ⇒ D 非保护因子、保护机制在 va.env 别处（非 smoke 变量/启动顺序）→ 停回传」。与 R14-1/R15-2 同签名（worker.py:564 HcclAllGather），但相位在权重装载后。执行修正 3 项：宿主 python3 无 vllm→容器内跑；残留输出目录 FileExistsError→清理；/dev/shm 268G 残留 backing→清理。数据：/home/fjj/pp_r16.log + tmp/r16-vaenv-fusion/。

## §R16 交付判读 + §R17 保护因子直测派单（2026-10-30）

**R16 交付**（commit 12866a46b）：R16-0 va.env 全文——13 smoke env；关键值 MAX_NUM_BATCHED_TOKENS=$((2048))、MAX_TOKENS=256、RUN_MODE=graph、MTP×3、TP2/DP4、chinese_40k、MODEL_PATH=/data/...0731/；非 smoke 独有 VLLM_ASCEND_DSA_HOST_MSTX=0；无 HCCL 变量。R16-1 融合配方（va.env 全量+我们 10 值覆盖，scp-va+作者数据面）EI0006×20 → D 非保护因子、MSTX=0 排除（非充分）。
**R16 判读**（蓝区）：D 集合审计——MTP/TP/DP/RUN_MODE 与脚本默认值同值天然 inert；真不同值仅 batched_tokens/MAX_TOKENS/MSTX。**batched_tokens 相关性闭环：profile 过的 run（R9 RUN_OK/R15-1/作者现场）全=2048；崩的 10+ run 全≥25600（含 unset 默认 25600）→ 唯一完美相关变量**。机制：determine_available_memory dummy 前向 batch=batched_tokens，大 batch 首个 EP8 all-gather 首连超时。bench 口径预告：2048 下 prefill 分块、TPOT 不受影响。
**R17 派单**：R15-2 基线（仅 MODEL_PATH）逐个加回：①batched=2048 ②+chunked1 ③+MSTX=0 ④+MAX_TOKENS=256，命中即停（过 profile=EI0006=0）；每步 env 原文核验+默认值回显；R18 预案=命中组合+fjj 仓+BENCH=1 落 bench 基线。

## §R17 保护因子直测命中 + §R18 fjj bench 整网数据面跑通（2026-10-30）

**R17 阶梯**：R17-1（batched2048 单设）撞配置校验 `chunked prefill is off: max_num_batched_tokens must be >= max_model_len`（simple_test.py:231，batched<model_len 非法）未到 profile；R17-2（+chunked1）**EI0006=0 命中**，1004=76（DP3_TP1_EP7 KV-cache 绑定）、DRAIN_OK×1（DP3_TP0_EP6）、EngineCore DP0/1/2 patch_shm_broadcast 挂起 → 保护因子=batched2048+chunked1，阶梯即停，R16 判读（profile 过=2048）坐实。
**R18 bench**（fjj 仓+命中组合+BENCH=1，tmp/r18-fjj-bench）：首跑 `worker.py:475 Free memory 2.48/91.75 GiB`（R17-2 残留 worker 占 HBM 环境时序）→ 清场重跑。重跑 **EI0006=0/1004=0**、64 请求 decode 完成、`DSA_STORE_CLOSED requests=16 put_pages=389376 remote_get=1`、DRAIN_OK×2（DP0）、overlap CSV 14992 行（classify_union_mean≈75µs）——**整网数据面首次真跑通**。但 **3/4 EngineCore patch_shm_broadcast 挂起**（21 条，R17-2 同签名）→ `[dsa-smoke][metrics]`/metrics.json 未产出 → §R10 通过标准第 3/4 条不满足，bench 基线未落袋。新阻塞点=推理完成后收尾广播挂起，交蓝区判读。

## §R17+R18 交付判读 + §R19 止血基线派单（2026-10-30）

**R17 交付**：R17-1 batched2048 单设撞校验（chunked off 时 batched≥model_len，simple_test.py:231）；R17-2 +chunked1 **EI0006=0 命中** → 保护因子=batched2048+chunked1，R16 因果坐实。
**R18 交付**（fjj+BENCH=1 完整配方）：**整网数据面真跑通**——64 请求满 decode(~957 tok)、PUT/GET 真实（put_pages=389376、remote_get=1）、DRAIN_OK×2(DP0)、overlap CSV 14992 行(mean≈75µs)；但 3/4 EngineCore 收尾 `patch_shm_broadcast` 挂起 → metrics 未产出，基线未落袋（0/11）。
**R18 判读**（蓝区）：本地读 patch_shm_broadcast.py——系 vllm shm_broadcast **读侧**补丁（5s 防 lost-notify），挂起为 **enqueue 写侧 60s 无空闲块**=某 reader 停消费→块不回收→写侧死等。泄漏∝请求数×decode 步数（R9 16req/256tok 净 / R18 64req/957tok 挂 / R17-2 teardown 挂）。止血优先：退回 R9 包线（BS4×DP4=16 请求）跑 BENCH=1 先落基线；块泄漏根修走代码线。
**R19 派单**：R19-0 挂起上下文只读三块；R19-1 BS4 止血基线（过=基线落袋 BS4 口径）；R19-2 BS4+MT256（挂=泄漏∝步数）；R19-3 BS1+MT64（R9 精确包线）；全挂→代码线修 shm 块生命周期。

## §R19 止血线两档：BS4×1024 挂起复现 + BS4×256 8/8 DRAIN 消除挂起（2026-10-30）

**R19-1（BS4×1024，R18 逐字仅 BS16→BS4）**：EI0006=0/1004=0，DRAIN 仅 2/8（DP2），patch_shm_broadcast 挂起复现（DP0/1/3，首挂 17:47:28），泄漏∝decode 步数（1024）而非请求数（16）坐实。
**R19-2（BS4×256）**：EI0006=0/1004=0，**8/8 worker 全 DRAIN_OK + STORE_CLOSED（requests=4 put_pages=97344 remote_get=1）+ patch_shm 挂起 0 次**——短 decode 彻底消除收尾挂起，坐实泄漏∝步数；overlap CSV 8 端口 7239-7581 行（classify_union_mean≈38-39µs）。**但收尾撞框架仓统计 bug**：simple_test.py:451 `first_token_time` AttributeError（vLLM 0.23.0 已改名 `first_token_ts`，stats.py:208），4 DP 全崩 → metrics 未产出、基线未落袋。该函数是框架仓 R10 bench 口径新提交 717c396/e6fd53b 引入（作者 va 仓无），改字段名即可收 baseline。R19-3（BS1×64）不执行（挂起已消除，非挂起问题）。
## §用户经验入库：profile×decode 挂起与 EI0006 配方边界（2026-10-30）

用户反馈：开 vLLM profile 采数据时 decode 阶段会挂。蓝区核实：bench 配方 ENABLE_PROFILE=0 + --no-perf，未采 vLLM 侧 profile（pool 侧 overlap CSV 为固有遥测，R9 干净 run 同开）；该经验与 R17-2/R18 patch_shm 写侧块耗尽同根嫌疑（profile 增大每步广播流量→加速块耗尽），R17-2 曾因 unset 落作者默认 enable_profile=True 实采 profile 后挂起，同型旁证。纪律：bench/验收轮 profile 关；采 profile 先修块泄漏。同日入库 EI0006 配方边界：batched2048+chunked1 成对必需。


## §R19 判读 + 统计 bug 修复 + §R20 基线落袋派单（2026-10-30）

**R19 判读**（蓝区）：R19-1（BS4×1024）挂起复现、R19-2（BS4×256）8/8 DRAIN + 挂起 0 次 ⇒ **shm 块泄漏∝decode 步数坐实**（与请求数无关），当前可用包线 decode ≤256 步。新阻塞 = 蓝区 bench 代码 bug：`simple_test.py:451/454/455` 引用 vLLM 旧字段 `first_token_time/last_token_time`，0.23.0 已改名 `first_token_ts/last_token_ts`（绿区 model.log 定位）。
**蓝区修复**（本 commit）：3 处字段改名；mock（/tmp/test_smoke_metrics.py）同步改并全过。
**R20 派单**：绿区 pull 修复后逐字重跑 R19-2（BS4×256+BENCH=1），过 = §R10 第 3/4 条 metrics 基线落袋。

## §R20 统计 bug 修复生效，bench 基线首次落袋（2026-10-30）

pull c461a78（first_token_time→first_token_ts ×3，simple_test.py:451/454/455）后逐字重跑 R19-2（BS4×256 BENCH=1，tmp/r20-bench-bs4-mt256）。**summary RUN_OK model_exit=0（首次整网干净收尾）**、8/8 DRAIN_OK+STORE_CLOSED（requests=4 put_pages=97344 remote_get=1）、patch_shm 挂起 0 次、EI0006=0/1004=0；4×[dsa-smoke][metrics][dp=N]+[global] 行全产出，results/1p/dp4_tp2_offload_graph_mtp_graph_hbm_full_remote_shared.metrics.json 存在（global 含 requests/分位数/吞吐/tpot_under_50ms 全字段）。global 原文：requests=16/output_tokens=4096/global_wall_s=81.54/request_throughput_rps=0.196/global_output_tps=50.23/tpot_ms avg=136.055/tpot_under_50ms=false。overlap CSV 8 端口 59394 行 classify_union_mean_us≈12.7-13.0µs。**通过标准 4 条全满足 → bench 基线 1/11 落袋**。⚠️ 数据质量告警：ttft_ms 巨大负值（first_token_ts 与 arrival_time 时钟基准不一致，tpot 正常）交蓝区修口径。bench 1/11。

## §R20 判读 + TTFT 单位修复 + §R21 派单（2026-10-30）

**R20 交付**（commit a4f2b6429）：BS4×256 BENCH=1 逐字重跑，统计 bug 修复生效——**bench 基线首次落袋**：RUN_OK、8/8 DRAIN+STORE_CLOSED、挂起 0 次、4×metrics 行+global+metrics.json。数值：requests=16/output_tokens=4096/wall=81.54s/rps=0.196/output_tps=50.23/tpot avg=136.055 p50=135.997 p90=241.9 p99=242.1 min=29.899/tpot_under_50ms=false；ttft 巨负（待修）。overlap classify_union_mean≈12.7-13.0µs。
**R20 判读**（蓝区）：基线口径 = chunked-prefill 混合相位（prefill 肩覆盖 decode 期），TPOT avg/p50 不可用于 TPOT<50ms 判定，min=29.9ms 代表干净 decode 步；稳态口径需 prefill 分段或 100K×1K 长 decode 尾段（待 shm 泄漏根修）。TTFT bug 定性：arrival_time=epoch µs、first/last_token_ts=epoch s → **已修**（>1e12 归一为 µs，mock 真实单位用例过）。
**R21 派单**：R21-0 只读（stats.py 源码权威确认单位 + R19-1 挂起上下文=代码线输入）；R21-1 TTFT 修复逐字复跑 R20（ttft 量级合理为过标准）。R22 预案：offload-off 对照（+20% 口径）+ shm 块泄漏根修。

## §R21 交付：TTFT 复跑仍负，时钟域错配根因坐实（2026-10-30）

pull 15d4610（TTFT 单位归一修复）后逐字重跑 R20（BS4×256 BENCH=1，tmp/r21-bench-bs4-mt256）。首跑 23:42:27 撞 DP3_TP1_EP7 RaGetDevEidInfoNum code=1004（白名单瞬时竞争，本次落 DP3），清场重跑 23:53:18 成功：**summary RUN_OK model_exit=0**、8/8 DRAIN_OK+STORE_CLOSED（requests=4 put_pages=97344 remote_get=1）、patch_shm 挂起 0 次、EI=0/E4=0/MET=6/HANG=0；4×[dsa-smoke][metrics][dp=N]+[global] 全产出，metrics.json 存在（23:51 复跑版本）。global 原文：requests=16/output_tokens=4096/wall=81.481s/rps=0.196/output_tps=50.27/tpot avg=135.234/p50=134.919/min=28.725/tpot_under_50ms=false；**ttft_ms 仍负 avg=-1792611306857.886（-1.79e12）**。
**失败处置命中**（TTFT 仍负 ⇒ 回传 R21-0 stats 源码原文）：根因=**时钟域错配坐实**——`EngineCoreOutputs.__post_init__`（vllm/v1/engine/__init__.py:230-231）`self.timestamp = time.monotonic()` 使 first/last_token_ts 为 **engine-core 单调时钟**；`arrival_time = time.time()`（async_llm.py:742/input_processor.py:278,289）为 **wall-clock epoch 秒**；两域相减必无意义（数学闭合：764077−1793375411=−1792611334，×1000 与实测逐位吻合）。15d4610 的 arrival_s 归一只修量级未修时钟域。**建议修复：改用 `metrics.first_token_latency`**（stats.py:369-371 官方同域 wall-clock TTFT=iteration_timestamp−arrival_time，已写入 RequestStateStats 并经 output_processor.py:373 随 RequestOutput 透出）×1000 即 ms。R21-0 三块取证原文（stats.py 时钟域注释+arrival_time 全链 grep+R19-1 挂起上下文 18 条报文/DP2 DRAIN+三组挂起最后活动）已回传 pingpong.md。overlap 8 端口 59394 行 classify_union_mean_us≈12.7-13.0µs。机器已清场。bench 基线保持 R20 值，TTFT 待蓝区按真实时钟域修后复取。

## §R21 判读 + §R22 派单：TTFT 时钟域修复 + offload-off 对照 + shm 取证收口（2026-10-30）

**判读**：bench 管道验收（RUN_OK/8×DRAIN/metrics 全产出，TTFT 负值=统计链路 bug 非引擎问题）；绿区时钟域证据链完整（monotonic vs wall-clock，数学逐位闭合 −1.7926e12），蓝区认错 15d461035 只修量级，采纳 `metrics.first_token_latency`（stats.py:369-371 同域 wall-clock）方案——simple_test.py 已改（first_token_latency 优先、≤0/None 回退旧路径），本地 mock 三用例全过。shm 挂起机制认知更新：R19-1 停摆窗口证据指向 worker 在 DSA_SHARED_CONNECT_OK 之后同步点停摆、shm 写侧等待为下游症状，头号嫌疑从"块泄漏"修正为"connect 后停摆"。
**派单**：R22-0 只读取证（shm_broadcast.py 写侧源码 + R19-1 停摆窗口 17:46:09→17:47:28 worker 输出 + 各 EngineCore 告警计数）；R22-1 offload-off 对照轮（R13-1 骨架逐字，唯一变量 ENABLE_OFFLOAD=0，BS4×256 配方不变，输出 tmp/r22a-bench-bs4-mt256-nooffload），通过标准含 TTFT 区间验证（0<ttft<wall×1000）。
