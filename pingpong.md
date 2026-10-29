# Pingpong

## Protocol

- 本文件是蓝区 ↔ 绿区的唯一通信面；`history.md` 是只追加的轮次档案。
- **蓝区**（可外部 AI 侧）写全部代码、做全部判读决策；**绿区**（隔离执行侧）按任务书执行并回写交付，不改代码。
- 每次 commit 必须同时包含代码 + `pingpong.md` + `history.md`；pull 用 `--ff-only`；绝不 force-push。
- Current owner: **绿区**。Status: §R10 已判读（EI0006=有 rank 未到达 EP8 all-gather 的超时齐射；主嫌疑=graph+MTP graph 首次上机，次嫌疑=数据面 e2/20000 段/eid json 差分）+ **§R11 派单已发**（隔离梯子：R11-0 取证 → A 原样重跑 → B eager 隔离 → C graph 去 MTP graph → D 回作者逐字配置）。R9 阳性落袋不变：整网 eager RUN_OK、机器层故障面清除。

## 0. 目标与口径

**总目标**：在绿区 A5 单机 8 卡（Ascend 950DT）上跑通 remote-offload 整网链路——框架（本仓 `vllm-ascend-v0.23.0-custom-remote-offload` 分支）+ 算子仓 `a5-remote-offload-ops` + 本机 URMA Pool，完成 `hbm_full_remote_shared` 真实推理，拿到作者文档钦定的生效证据。

**锚点事实**（蓝区已核对，绿区不得变更口径）：

- 本仓分支 tip：`cac7e2c4e` + 蓝区 pingpong 文档提交（pull --ff-only 后 `git log --oneline -3` 顶部应是 pingpong 提交）。
- 算子仓：`https://github.com/JiJiaF1020/a5-remote-offload-ops`，main tip `f63a5bfed`（R4 起；ScatterSFA 回退 + shared-port-gather）。
- 模型：`/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731`（GLM-5.2，78 层 target，W4A4C8 mxfp4）。先 `ls` 确认；路径不对就只回报实际路径，不要自己换模型。
- 容器：`dsa-full`（镜像 `vllm-ascend:v0.23.0-a5-openeuler`，host 网络/pid，挂 /etc /data /home）。`docker ps` 核对；容器名不同就按实际的进。
- 本轮只判"通/不通"，不做任何性能结论。带宽/TPOT 等性能口径后续轮次单独派单。

**已知边界**（作者写明，遇到不算新发现，按判读预案处理）：

- remote 链 DP4/BS32 的 NPU 实机验收作者尚未完成；本轮按文档梯子 DP1/TP1 eager BS1 首发，过了再放大。
- "零 miss 第二轮全 hit 卡住"是作者未关闭问题；若复现，保留 generation/status/Pool 三方日志原样回传。
- 新旧 Pool/OPP/extension 不可混装；本轮全部用同一套源码新鲜编译。
- backing 文件不会自动删除；每次起 Pool 用全新路径。

---

## §R1 派单-旧：remote-offload 单机整网首发（蓝区，2026-10-26）

### 目标

单机 8 卡跑通 remote offload：框架装好 → 算子仓编好 → EID 探测出 JSON → Pool 起来 → DP1/TP1 eager BS1 真实推理通过；全绿则条件放大到 DP4/TP2 八卡。

### 前置核验

1. `docker ps` 确认 `dsa-full` 在运行；后续命令除注明外都在容器内执行（`docker exec -it dsa-full bash`）。
2. `npu-smi info | head -30`：8 卡在线、无进程占用。
3. `ls /data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731 | head`：权重在。不在则 `ls /data/model/` 回报实际目录名，**不要自行替换**。
4. `df -h /home /data`：/home 剩余 < 20G 时，本轮所有新目录（克隆、日志）一律放 `/data/fjj`，下文命令里的 `/home/fjj` 相应替换。前后必须一致。
5. 清场：`ps aux | grep -E 'python|vllm|urma' | grep -v grep | awk '{print $2}' | xargs -r kill -9; sleep 5`

### 执行命令

**Stage B · 框架安装**

```bash
# B1 代码就位（机器上已有上次手工 clone 的目录就续用）
if [ -d /home/fjj/vllm-ascend-remote-offload ]; then
  cd /home/fjj/vllm-ascend-remote-offload
  git fetch origin && git checkout vllm-ascend-v0.23.0-custom-remote-offload && git pull --ff-only
else
  cd /home/fjj && git clone --depth 1 -b vllm-ascend-v0.23.0-custom-remote-offload \
    https://github.com/JiJiaF1020/vllm-ascend-dsa-v0.23.0-custom.git vllm-ascend-remote-offload
  cd vllm-ascend-remote-offload
fi
git log --oneline -2    # 回报这两行

# B2 catlass 子模块修复（gitcode.com 在机器上解析不了，从本机已有代码树拷；找不到再用 hosts 兜底）
SRC=""
for d in /home/w00916487/vLLM-ascend-DSA /home/fjj/vllm-ascend-dsa-v0.23.0-custom; do
  [ -f "$d/csrc/third_party/catlass/cmake" -o -d "$d/csrc/third_party/catlass/cmake" ] && SRC="$d" && break
done
if [ -n "$SRC" ]; then
  mkdir -p csrc/third_party
  cp -r "$SRC/csrc/third_party/catlass" csrc/third_party/catlass
  echo "catlass copied from $SRC"
else
  echo "116.205.6.180 gitcode.com" >> /etc/hosts
  echo "catlass fallback: hosts entry added"
fi

# B3 编译安装（10~30 分钟，installing build dependencies 阶段无进度条属正常）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
SOC_VERSION=ascend950dt_9582 COMPILE_CUSTOM_KERNELS=1 \
SETUPTOOLS_SCM_PRETEND_VERSION="0.23.0" VLLM_TARGET_DEVICE=empty \
pip install --no-build-isolation --no-deps -v -e . 2>&1 | tee /home/fjj/pp_r1_b_install.log

# B4 验证（必须 cd 出源码目录再跑，否则是假阳性）
cd /tmp && python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"
```

**Stage C · 算子仓构建**

```bash
cd /home/fjj   # 若 A4 判定 /home 空间不足，则 cd /data/fjj，下同
git clone --depth 1 https://github.com/JiJiaF1020/a5-remote-offload-ops.git
cd a5-remote-offload-ops
BUILD_JOBS=32 SOC_VERSION=ascend950 ENABLE_AICPU_KERNEL=0 ENABLE_LOCAL_AIV=1 ENABLE_C8_MTP_OPS=1 \
KVCACHE_SCATTER_COPY_OP_NAME=ALL bash build_and_install.sh 2>&1 | tee /home/fjj/pp_r1_c_build.log
bash test/urma_mock_pool/build.sh 2>&1 | tee -a /home/fjj/pp_r1_c_build.log
bash test/urma_mock_pool/build_store.sh 2>&1 | tee -a /home/fjj/pp_r1_c_build.log
ls -l tmp/mempool-build/urma_pool_server test/urma_mock_pool/urma_mock_pool
ls torch_extension/kvcache_scatter_copy_custom_ops/*.so
```

**Stage D · EID 探测**（生成 remote-shared.json，同机探测不需预启动 Pool）

```bash
cd /home/fjj/a5-remote-offload-ops
python3 test/probe_urma_eids.py host 2>&1 | tee /home/fjj/pp_r1_d_host.log
python3 test/probe_urma_eids.py scan \
  --devices npu:0 npu:1 npu:2 npu:3 npu:4 npu:5 npu:6 npu:7 \
  --pool-group 0-3=UB_A --pool-group 4-7=UB_B \
  --pool-ip 127.0.0.1 --control-port 19190 \
  --store-endpoint 127.0.0.1:19080 --du-port-base 20000 \
  --output tmp/eid-probe-r1 2>&1 | tee /home/fjj/pp_r1_d_probe.log
cat tmp/eid-probe-r1/remote-shared.json
```

**Stage E · Pool 启动 + DP1/TP1 首发**

```bash
# E1 起 Pool（backing 必须是全新路径；<...> 处原样粘贴 Stage D 探测打印的 MEMPOOL_WRAPPER_ARGS）
cd /home/fjj/a5-remote-offload-ops
nohup tmp/mempool-build/urma_pool_server \
  --store-mode embedded \
  --backing /dev/shm/kv-remote-r1 --store-bytes 4294967296 \
  --payload-layers 78 --requests 8 --source-capacity 25600 \
  --store-ports 19080,19081,19082,19083,19084,19085,19086,19087 \
  <MEMPOOL_WRAPPER_ARGS 原样粘贴> \
  --trace-lifecycle \
  --pool-binary $PWD/test/urma_mock_pool/urma_mock_pool -- \
  --jetty-count 1 --blocks 8 --poll-cpus 2 \
  --gather-threads 1 --gather-cpus 4 --du-sibling-cpus 6 \
  --du-workers 1 --du-concurrency 1 --pipeline-mib 0 \
  > /home/fjj/pp_r1_e_pool.log 2>&1 &
sleep 15
grep -c "URMA_JETTY_READY" /home/fjj/pp_r1_e_pool.log   # 回报这个数字

# E2 框架 dry-run（只检查配置，不加载模型）
cd /home/fjj/vllm-ascend-remote-offload
export VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731
export VLLM_ASCEND_SMOKE_TENSOR_PARALLEL_SIZE=1
export VLLM_ASCEND_SMOKE_DATA_PARALLEL_SIZE=1
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=1
export VLLM_ASCEND_SMOKE_RUN_MODE=eager
export VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=1
export VLLM_ASCEND_SMOKE_ENABLE_MTP=1
export VLLM_ASCEND_SMOKE_ENABLE_MTP_GRAPH=0
export VLLM_ASCEND_SMOKE_ENABLE_PROFILE=0
export VLLM_ASCEND_SMOKE_MAX_TOKENS=16
export VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=25600
export VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=25600
export VLLM_ASCEND_SMOKE_DATASET=chinese_23k
python examples/dsa_demo/simple_test_1p.py \
  --remote-shared-config /home/fjj/a5-remote-offload-ops/tmp/eid-probe-r1/remote-shared.json --dry-run

# E3 实跑（几十分钟量级）
python examples/dsa_demo/simple_test_1p.py \
  --remote-shared-config /home/fjj/a5-remote-offload-ops/tmp/eid-probe-r1/remote-shared.json \
  2>&1 | tee /home/fjj/pp_r1_e_run.log
```

**Stage F · 条件放大（仅当 Stage E 全部通过标准命中；否则不做，直接回传）**

```bash
ps aux | grep -E 'python|vllm' | grep -v grep | awk '{print $2}' | xargs -r kill -9; sleep 5   # 只杀框架，Pool 留着
export VLLM_ASCEND_SMOKE_TENSOR_PARALLEL_SIZE=2
export VLLM_ASCEND_SMOKE_DATA_PARALLEL_SIZE=4
python examples/dsa_demo/simple_test_1p.py \
  --remote-shared-config /home/fjj/a5-remote-offload-ops/tmp/eid-probe-r1/remote-shared.json \
  2>&1 | tee /home/fjj/pp_r1_f_run.log
```

### 通过标准

- **B 通过**：pip 退出码 0；B4 输出恰为 `/home/fjj/vllm-ascend-remote-offload/vllm_ascend/__init__.py`。
- **C 通过**：构建日志尾部有 `build and local install done`；两个 Pool 二进制和 torch extension `.so` 均存在。
- **D 通过**：scan 退出码 0，8 张卡全部通过，生成 `tmp/eid-probe-r1/remote-shared.json` 和 `pairs.csv`，日志打印出 `MEMPOOL_WRAPPER_ARGS`。
- **E 通过**（需同时满足）：
  1. Pool 日志 `URMA_JETTY_READY` 计数 ≥ 1；
  2. 框架日志出现 `DSA HBM CACHE CAPACITY REPORT`，且含 `indexer=21, resident=57, mtp=1`（78 层/21 full 的期望值）；
  3. 日志依次出现 `DSA_EXECUTION_PLAN` → `DSA_STORE_PUT_DONE` → `DSA_SHARED_CONNECT_BEGIN` → `DSA_SHARED_CONNECT_OK ... backend=real_store` → 至少两轮 `DSA_SHARED_CONSUMED` → 退出时 `DSA_SHARED_DRAIN_OK`（其中 select/gather/scatter_sfa 计数非零）；
  4. 进程退出码 0，输出里有生成文本。
- **F 通过**：E 的 2-4 条同样命中，且 `DSA_SHARED_CONNECT_BEGIN` 的 `rows=` 与实际绑定请求数一致（DP4 应为 4）。

### 回传格式

在 pingpong.md 追加「§R1 交付」节：

1. 每个 Stage 的退出码 + 上述通过标准逐条的命中/未命中 + 对应日志原文行（不加工、不节选结论）。
2. `git log --oneline -2`（框架仓）与算子仓 `git log --oneline -1`。
3. `remote-shared.json` 全文、`MEMPOOL_WRAPPER_ARGS` 原文。
4. 失败时：第一个报错点的日志前后 50 行 + 对应 pp_r1_*.log 文件路径。
5. 注明执行环境：机器 IP、容器名、执行时间、各 Stage 耗时。

### 判读预案（派单时写死）

- **B 卡在 catlass 且本机无拷贝、hosts 兜底也不通** ⇒ 本轮无效；蓝区改用压缩包线下投递 catlass，重新派单。
- **C 编译报错** ⇒ 阴性（环境问题）；按报错定位缺依赖后出补丁或换安装命令，绿区不自修。
- **D 部分卡探测失败** ⇒ 阴性；回报失败卡号 + `probe host` 输出，蓝区判断是 URMA 驱动/链路问题还是 pool-group 分组名与实际不符（分组名不准时蓝区改参数重派，绿区不要自己改分组）。
- **E1 无 URMA_JETTY_READY** ⇒ 阴性（URMA 链路/驱动层）；收集 `urma_admin` 设备可见性输出回传。
- **E 出现 `DSA A5 MTP LIM requires 1..4 query rows per request`** ⇒ 已知 bug 变体命中，阴性；蓝区核对形状后出修复或配置绕行。
- **E 第二轮全 hit 卡住** ⇒ 命中作者未关闭的"零 miss"问题，阴性（非本轮回退）；保留三方日志回传即算完成本轮义务。
- **E 全中、F 未做或失败** ⇒ 首轮阳性（链路通），放大问题另起一轮。
- **E、F 全中** ⇒ 阳性落袋；下一轮：graph 模式 + 带宽/重叠效率测量（性能口径另写）。

---

（新轮次从这里往下追加：蓝区写「派单」节，绿区写「交付」节，蓝区写「判读」节。最新一轮在最上面，旧轮次标题加「-旧」标记。）

## §R3 派单-旧：EI0006 定位与越过（蓝区，2026-10-27）——已被 §R4 取代；若已执行部分实验，数据原样附入 §R4 交付

### 目标

R2 卡点：DP4/TP2（EP8）实跑在 profile_run 首次 HcclAllGather 建链超时（EI0006），而裸 8 卡 HCCL 探针 8/8 通过。本轮用两步对照定位并尝试越过：先证分支 EP8 基线（offload 关），再加 HCCL 建链超时放宽重试全配置。

### 前置核验

1. Pool 存活：`ps aux | grep urma_pool_server | grep -v grep`（R2 重启后的实例，backing=kv-remote-r2b，sibling=5；pid 以实际为准）。活着 → 跳过 R3-0；死了 → 执行 R3-0。
2. `npu-smi info | head -30`：8 卡空闲、无残留进程。
3. 清场只杀框架，**不杀 Pool**：`ps aux | grep -E 'python|vllm' | grep -v grep | awk '{print $2}' | xargs -r kill -9; sleep 5`

### 执行命令

**R3-0 ·（仅 Pool 死亡时）重启 Pool** —— backing 换新路径 kv-remote-r3，sibling 用 5（R2 已确认的正确值）：

```bash
cd /home/fjj/a5-remote-offload-ops
nohup tmp/mempool-build/urma_pool_server \
  --store-mode embedded \
  --backing /dev/shm/kv-remote-r3 --store-bytes 4294967296 \
  --payload-layers 78 --requests 8 --source-capacity 25600 \
  --store-ports 19080,19081,19082,19083,19084,19085,19086,19087 \
  --du-ports 20000,20001,20002,20003,20004,20005,20006,20007 \
  --du-devices udmac1d1e2,udmac1d1e2,udmac1d1e2,udmac1d1e2,udmac0d1e3,udmac0d1e3,udmac0d1e3,udmac0d1e3 \
  --du-eid-indices 0,0,0,0,0,0,0,0 \
  --trace-lifecycle \
  --pool-binary $PWD/test/urma_mock_pool/urma_mock_pool -- \
  --jetty-count 1 --blocks 8 --poll-cpus 2 \
  --gather-threads 1 --gather-cpus 4 --du-sibling-cpus 5 \
  --du-workers 1 --du-concurrency 1 --pipeline-mib 0 \
  > /home/fjj/pp_r3_pool.log 2>&1 &
sleep 15
grep -c "URMA_JETTY_READY" /home/fjj/pp_r3_pool.log   # 回报这个数字
```

**R3-A · 对照实验：offload 关闭，其余全同**（验证本分支 EP8 基线，不带 remote-shared-config）

```bash
cd /home/fjj/vllm-ascend-remote-offload
export VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731
export VLLM_ASCEND_SMOKE_TENSOR_PARALLEL_SIZE=2
export VLLM_ASCEND_SMOKE_DATA_PARALLEL_SIZE=4
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=1
export VLLM_ASCEND_SMOKE_RUN_MODE=eager
export VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=0
export VLLM_ASCEND_SMOKE_ENABLE_MTP=1
export VLLM_ASCEND_SMOKE_ENABLE_MTP_GRAPH=0
export VLLM_ASCEND_SMOKE_ENABLE_PROFILE=0
export VLLM_ASCEND_SMOKE_MAX_TOKENS=16
export VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=25600
export VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=25600
export VLLM_ASCEND_SMOKE_DATASET=chinese_23k
python examples/dsa_demo/simple_test_1p.py 2>&1 | tee /home/fjj/pp_r3a_run.log
echo "RC=$?"   # 回报
```

**R3-B ·（仅当 R3-A 通过）全配置 + HCCL 建链超时放宽**

```bash
export VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=1
export HCCL_CONNECT_TIMEOUT=1800
python examples/dsa_demo/simple_test_1p.py \
  --remote-shared-config /home/fjj/a5-remote-offload-ops/tmp/eid-probe-r1/remote-shared.json \
  2>&1 | tee /home/fjj/pp_r3b_run.log
echo "RC=$?"   # 回报
```

**R3-C ·（仅当 R3-B 仍 EI0006）精确栈采集**

```bash
export ASCEND_LAUNCH_BLOCKING=1
export HCCL_ENTRY_LOG_ENABLE=1
python examples/dsa_demo/simple_test_1p.py \
  --remote-shared-config /home/fjj/a5-remote-offload-ops/tmp/eid-probe-r1/remote-shared.json \
  2>&1 | tee /home/fjj/pp_r3c_run.log
echo "RC=$?"
tar czf /home/fjj/pp_r3c_hccl_log.tgz -C ~ ascend/log 2>/dev/null; ls -l /home/fjj/pp_r3c_hccl_log.tgz
```

### 通过标准

- **R3-A 通过**：进程退出码 0，输出有生成文本（证明本分支 DP4/TP2/EP8 基线在 remote-shared 之外可用）。
- **R3-B 通过**（需同时满足）：RC=0 有生成文本；日志依次 `DSA_EXECUTION_PLAN` → `DSA_STORE_PUT_DONE` → `DSA_SHARED_CONNECT_BEGIN` → `DSA_SHARED_CONNECT_OK ... backend=real_store` → ≥2 轮 `DSA_SHARED_CONSUMED` → `DSA_SHARED_DRAIN_OK`（select/gather/scatter_sfa 计数非零）；CONNECT `rows=` 与绑定请求数一致（DP4 应为 4）；`DSA HBM CACHE CAPACITY REPORT` 出现并原文记录整行。
- **R3-C 不要求通过**：报错栈 + HCCL entry log 落盘回传即可。

### 已知边界（同前）

- 保持 eager + BS1，不开 graph；不改代码、不改分组名、不动存活 Pool。
- 复现"零 miss 第二轮全 hit 卡住"→ 保留 generation/status/Pool 三方日志原样回传，即算完成本轮义务。

### 回传格式

同前：R3-A/B/C 各自 RC + 通过标准逐条命中 + 首个报错点前后 50 行 + pp_r3*.log 路径；附框架仓 `git log --oneline -2` 与执行起止时间。

### 判读预案（派单时写死）

- **R3-A 失败** ⇒ 分支级 EP8 回退（与 remote-shared 无关），阴性；蓝区对比本地卸载分支 diff 定位。
- **R3-A 过 + R3-B 过** ⇒ 根因=HCCL 建链超时（worker 启动偏斜），HCCL_CONNECT_TIMEOUT=1800 入库为本分支标配；R3-B 通过标准全中即**阳性落袋**，下一轮 graph 模式 + 带宽/TPOT 性能口径（另写派单）。
- **R3-A 过 + R3-B 仍 EI0006** ⇒ 阴性；R3-C 精确栈 + HCCL 日志回传，蓝区出代码修复。
- **出现 `DSA A5 MTP LIM requires 1..4 query rows per request` / 零 miss 卡住** ⇒ 按 R2 预案同条款处理（阴性，保留日志回传）。

## §R4 派单-旧：作者新流程跑通优化版 remote-offload（蓝区，2026-10-28）——已交付已判读，见 §R4 判读

### 背景（绿区必读）

作者 10-28 给了新的跑通流程，并确认两件事：① probe 产物需要给后四个 NPU 手工补 store_endpoint——**证实绿区 R2 修复方向正确**，继续沿用修复后的 remote-shared.json；② 作者自家跑通配置 = graph + MTP graph、40K 上下文、BS1（run 目录名 `dp4_tp2_offload_graph_mtp_graph_hbm_full_remote_shared`）。同时算子仓 main 已更新到 `f63a5bfed`（ScatterSFA 内核回退修性能退化 + 新增 `--shared-port-gather` 跨端口 Gather 共享），绿区机器上的 6e85590db 是**含性能退化的旧版**，本轮必须同步重编。本轮起改用算子仓自带工具 `tools/profile_pool_inference.py` 一键起双 Pool + 模型（不再手工起 Pool，R1-R3 的"不杀 Pool"旧规本轮起作废）。

### 目标

按作者流程跑通优化版 remote-offload 整网：Stage 1 作者原配置（graph+MTP graph、40K、BS1）拿跑通证据；Stage 2 条件放大到 BS16（40K×16，用户验收口径方向）。本轮仍以"通/不通"为主，overlap CSV 落盘即可，性能判读后续轮次。

### 前置核验

1. `docker ps` 确认 `dsa-full` 在运行；除注明外命令都在容器内执行。
2. `npu-smi info | head -30`：8 卡空闲、无残留进程。
3. `ls /data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731 | head`：权重在。
4. `ls /home/fjj/a5-remote-offload-ops/tmp/eid-probe-r1/remote-shared.json`：R1 探测产物在。不在就回报 `ls /home/fjj/a5-remote-offload-ops/tmp/` 实际内容，**不要重新 probe**。

### 执行命令

**R4-0 · 清场 + 停旧 Pool + 扩 /dev/shm**（本轮要起双 Pool 占用 19080-19087/20000-20007，旧 Pool 必须停；shm 要装 2×128GiB Store + Gather arena，384G remount 已获蓝区授权——容器 ipc=host，会影响宿主机 /dev/shm，属预期）

```bash
ps aux | grep -E 'urma_pool_server|urma_mock_pool' | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps aux | grep -E 'python|vllm' | grep -v grep | awk '{print $2}' | xargs -r kill -9
sleep 5
ps aux | grep -E 'urma|python|vllm' | grep -v grep   # 应为空，回报实际输出
rm -rf /dev/shm/kv-remote-r1 /dev/shm/kv-remote-r2 /dev/shm/kv-remote-r2b /dev/shm/kv-remote-r3 /dev/shm/offload-numa-*
mount -o remount,size=384G /dev/shm && df -h /dev/shm   # 回报 df 行
```

**R4-1 · 算子仓更新 + 重编**（含 ScatterSFA 回退和 shared-port-gather 新能力；Pool wrapper/binary 必须重编，工具会校验新选项）

```bash
cd /home/fjj/a5-remote-offload-ops
git -c http.proxy=http://90.255.12.179:3128 pull --ff-only
git log --oneline -3   # 回报；顶部必须是 f63a5bfed
source /usr/local/Ascend/ascend-toolkit/set_env.sh
USE_NINJA=1 SOC_VERSION=ascend950 BUILD_JOBS=128 ENABLE_C8_MTP_OPS=1 ENABLE_LOCAL_AIV=1 \
ENABLE_AICPU_KERNEL=0 KVCACHE_SCATTER_COPY_OP_NAME=ALL \
bash build_and_install.sh 2>&1 | tee /home/fjj/pp_r4_ops_build.log
bash test/urma_mock_pool/build.sh 2>&1 | tee -a /home/fjj/pp_r4_ops_build.log
bash test/urma_mock_pool/build_store.sh 2>&1 | tee -a /home/fjj/pp_r4_ops_build.log
# 新能力自检（两组 grep 都要有输出，原文回报）
test/urma_mock_pool/urma_mock_pool --help 2>&1 | grep -E "du-gather-shared|du-pipeline-mib|DU_AND_GATHER"
tmp/mempool-build/urma_pool_server --help 2>&1 | grep -E "gather-scheduler|du-store-prefault"
```

**R4-2 · 拓扑/资源核验**（只读，原文回报；CPU 范围用工具默认值即可，前提是核数和 NUMA 布局与作者机一致）

```bash
numactl -H | head -20
python3 -c "import os; a=sorted(os.sched_getaffinity(0)); print(len(a)); print(a[:8]); print(a[-8:])"
df -h /dev/shm
```

**R4-3 · 作者流程首发（BS1、graph+MTP graph、40K）**——工具自己起双 Pool（A：numa0/du 20000-20003/udmac1d1e2；B：numa2/du 20004-20007/udmac0d1e3，均为绿区 R1 实测分组）+ 框架 simple_test_1p.py（DP4TP2EP8 默认）。CPU 绑核、du-workers、requests 等全部用工具默认值（=作者值）。一次跑几十分钟量级。

```bash
cd /home/fjj/a5-remote-offload-ops
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731
export VLLM_ASCEND_SMOKE_DATASET=chinese_40k
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=1
export VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=45056
export VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=45056
export VLLM_ASCEND_SMOKE_MAX_TOKENS=64
export VLLM_ASCEND_SMOKE_ENABLE_PROFILE=0
python3 tools/profile_pool_inference.py \
  --no-perf \
  --framework-repo /home/fjj/vllm-ascend-remote-offload \
  --remote-config /home/fjj/a5-remote-offload-ops/tmp/eid-probe-r1/remote-shared.json \
  --output tmp/r4-author-bs1 \
  --payload-layers 78 \
  --store-bytes $((128*1024*1024*1024)) \
  --du-ports 20000,20001,20002,20003 --store-ports 19080,19081,19082,19083 \
  --pool-b-du-ports 20004,20005,20006,20007 --pool-b-store-ports 19084,19085,19086,19087 \
  --du-devices udmac1d1e2 --pool-b-du-devices udmac0d1e3 \
  --du-eid-indices 0 --pool-b-du-eid-indices 0 \
  --sample-port 20000 --sample-worker 0 \
  --numa-node 0 --pool-b-numa-node 2 --store-numa-nodes local \
  --pool-start-timeout 600 \
  --shared-port-gather --du-pipeline-mib 1 \
  2>&1 | tee /home/fjj/pp_r4_run.log
echo "RC=$?"   # 回报
```

**R4-3b ·（仅当 R4-3 失败且报错含 EI0006 / HcclAllGather 超时）放宽 HCCL 建链超时重试一次**

```bash
rm -rf /dev/shm/offload-numa-* tmp/r4-author-bs1
export HCCL_CONNECT_TIMEOUT=1800
python3 tools/profile_pool_inference.py \
  --no-perf \
  --framework-repo /home/fjj/vllm-ascend-remote-offload \
  --remote-config /home/fjj/a5-remote-offload-ops/tmp/eid-probe-r1/remote-shared.json \
  --output tmp/r4-author-bs1 \
  --payload-layers 78 \
  --store-bytes $((128*1024*1024*1024)) \
  --du-ports 20000,20001,20002,20003 --store-ports 19080,19081,19082,19083 \
  --pool-b-du-ports 20004,20005,20006,20007 --pool-b-store-ports 19084,19085,19086,19087 \
  --du-devices udmac1d1e2 --pool-b-du-devices udmac0d1e3 \
  --du-eid-indices 0 --pool-b-du-eid-indices 0 \
  --sample-port 20000 --sample-worker 0 \
  --numa-node 0 --pool-b-numa-node 2 --store-numa-nodes local \
  --pool-start-timeout 600 \
  --shared-port-gather --du-pipeline-mib 1 \
  2>&1 | tee /home/fjj/pp_r4b_run.log
echo "RC=$?"   # 回报
```

**R4-4 ·（仅当 R4-3 或 R4-3b 通过）放大到 BS16（40K×16，用户验收口径方向）**

```bash
ps aux | grep -E 'urma_pool_server|urma_mock_pool' | grep -v grep | awk '{print $2}' | xargs -r kill -9
sleep 5; rm -rf /dev/shm/offload-numa-*
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=16
python3 tools/profile_pool_inference.py \
  --no-perf \
  --framework-repo /home/fjj/vllm-ascend-remote-offload \
  --remote-config /home/fjj/a5-remote-offload-ops/tmp/eid-probe-r1/remote-shared.json \
  --output tmp/r4-bs16 \
  --payload-layers 78 \
  --store-bytes $((128*1024*1024*1024)) \
  --du-ports 20000,20001,20002,20003 --store-ports 19080,19081,19082,19083 \
  --pool-b-du-ports 20004,20005,20006,20007 --pool-b-store-ports 19084,19085,19086,19087 \
  --du-devices udmac1d1e2 --pool-b-du-devices udmac0d1e3 \
  --du-eid-indices 0 --pool-b-du-eid-indices 0 \
  --sample-port 20000 --sample-worker 0 \
  --numa-node 0 --pool-b-numa-node 2 --store-numa-nodes local \
  --pool-start-timeout 600 \
  --shared-port-gather --du-pipeline-mib 1 \
  2>&1 | tee /home/fjj/pp_r4_bs16.log
echo "RC=$?"   # 回报
```

### 通过标准

- **R4-1**：`git log` 顶部 f63a5bfed；build RC=0；两组 help grep 均有输出。
- **R4-3（或 R4-3b）通过**（需同时满足）：工具退出 RC=0；`tmp/r4-author-bs1/summary.txt` 含 `RUN_OK`；`raw/model.log` 有生成文本；model.log 依次出现 `DSA_EXECUTION_PLAN` → `DSA_STORE_PUT_DONE` → `DSA_SHARED_CONNECT_BEGIN` → `DSA_SHARED_CONNECT_OK ... backend=real_store` → ≥2 轮 `DSA_SHARED_CONSUMED` → `DSA_SHARED_DRAIN_OK`；`raw/overlap-*.csv` 存在且非空；`run-status.json` 里 `graceful=true`、`errors=[]`。
- **R4-4**：标准同 R4-3（目录换 tmp/r4-bs16）；失败不阻塞本轮，按判读预案处理。

### 已知边界（遇到不算新发现，按判读预案处理）

- BS16 + MTP graph 可能命中 `DSA A5 MTP LIM requires 1..4 query rows per request`（本分支未修的 LIM 校验）——命中则 R4-4 记阴性、日志回传，不影响 R4-3 结论。
- "零 miss 第二轮全 hit 卡住"：保留 Pool 日志（tmp/r4-*/raw/pool-*.log）+ model.log 原样回传。
- 工具不会删 backing；R4-4 前已含清理步骤，此外不要手工删其它路径。
- R4-0 的 `mount -o remount` 影响宿主机 /dev/shm，已授权；不要动宿主机其它挂载。

### 回传格式

在 pingpong.md 追加「§R4 交付」节，格式同前：R4-0/1/2/3(/3b/4) 各阶段 RC + 通过标准逐条命中/未命中 + 关键日志原文行；失败给首个报错点前后 50 行 + 对应日志路径（pp_r4*.log、tmp/r4-*/summary.txt、tmp/r4-*/precheck.txt、tmp/r4-*/raw/model.log 末 100 行）；附算子仓 `git log --oneline -3`、框架仓 `git log --oneline -2` 与执行起止时间。若 R3 已做过部分实验，数据原样附在交付末尾。

### 判读预案（派单时写死）

- **R4-3 直接通过** ⇒ 阳性落袋（作者优化版在绿区复现）；R4-4 数据用于 BS16 评估；蓝区随后启动 bench 汇总输出（TPOT/吞吐）开发。
- **R4-3 EI0006、R4-3b 通过** ⇒ 阳性落袋，HCCL_CONNECT_TIMEOUT=1800 入库为本分支标配，R3 的建链偏斜假说坐实。
- **R4-3/3b 均失败** ⇒ 阴性；summary/precheck/model.log 回传，蓝区对照作者流程逐参数定位。
- **R4-4 LIM/零 miss** ⇒ 已知边界命中，阴性不阻塞主线；蓝区核对形状后决定修复或配置绕行。
- **R4-1 构建失败** ⇒ 报错末 30 行回传，停止后续阶段。

## §R6 判读（蓝区，2026-10-29）

### 总结论

**阴性（复现失败），但故障性质改写：从"环境漂移"精确化为"间歇性故障"**。关键时间线对照：绿区 10-27 22:47（EI0006）与 10-28 01:07（dispatch timeout）失败 **早于** 作者 10-28 04:07 成功，绿区 10-29 又全部失败——同一命令、同一代码、同一配置，**先败、后过、再败**。这不是单向漂移，是随时间翻转的间歇故障，嫌疑锁定在「同机多租户干扰」：机器是 16 卡共享机，10-29 新增多个他人 v0.30 容器（deepseek-v41 系），R6 首跑当场抓到他人 write-bench 进程占用我方 UB 设备 udmac1d1e3（pid 2481305，hold-999s）。作者成功在凌晨 04:07（大概率整机空闲窗口）与此吻合。

### 逐条裁决

- **复现失败同为 EI0015：接受**，但派单预案的"环境漂移"措辞修正为"间歇性故障"（见上）。
- **前置补漏质量：接受**。三个关键新事实入库：① 宿主机无 CANN toolkit，CANN 全在容器（排除宿主 CANN 版本变量）；② 当前驱动 25.6.rc2 / 固件 7.0.9.100.203 / Chip V120；③ 作者无 .bash_history，执行上下文不可考。
- **端口占用处置（不杀他人进程、等自然释放后重跑）：接受并表扬**，符合多租户纪律。
- **作者 smoke 改动原文入库**：`min_tokens=max_tokens, ignore_eos=True`（强制满长 decode）+ simple_test_1p.py 追加 `gpu_memory_utilization=0.97` 等。蓝区注：gpu_mem 0.97 vs 绿区 0.85 影响显存分区但不影响建链；smoke 改动只影响 decode 行为，与 EI0015 无关。
- **待验证机制（估计）**：950DT 单节点 HCCL 建链走 UB/HCCS 平面；他租户在同平面做 URMA write-bench 期间，我方 EP8 RootInfoDetect 可能被干扰。R7 用裸探针鉴别。

### 环境事实入库（后续轮次直接引用）

- 作者成功基线：scp-opt `tmp/prefault-read-full01`（10-28 04:07，RUN_OK，DRAIN_OK×8，commands.json 与 R6 逐字段一致）。
- 宿主机 `/usr/local/Ascend` 只有 driver/firmware，无 toolkit。
- 机器为他人共用：跑任何关键实验前必须先做占用取证（R7-0 已标准化）。

## §R7 派单-旧：间歇性鉴别与占用取证（蓝区，2026-10-29）——已交付已判读，见 §R7 判读

### 目标

鉴别 EI0015 是否与机器占用状态相关：每次试跑前先跑 8 卡 HCCL 裸探针 + 全机占用取证，再逐字复现作者命令，最多 3 次尝试。探针失败 ⇒ 机器层实锤，不用再烧模型装载；探针过而模型败 ⇒ 差异在 vLLM 多进程树建链方式。

### 执行命令

**R7-0 · 占用取证（每次尝试前必做，原文回传）**

```bash
date
docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
npu-smi info | head -40
ps aux | grep -iE 'urma|vllm|python|write-bench|mock_pool' | grep -v grep
ss -tlnp 2>/dev/null | grep -E ':(19|20)0[0-9][0-9]' || true
```

**R7-1 · 8 卡 HCCL 裸探针**（复用留存脚本；结果原文回传）

```bash
bash /home/fjj/hccl_probe_8npu.sh 2>&1 | tail -20
# 回报：8/8 PASS 还是有失败；若失败，附 /tmp/hccl_probe.rank*.log 报错行
```

**R7-2 · 逐字复现（仅当 R7-1 8/8 PASS 才做；否则本轮停止，只回传 R7-0/R7-1）**

同 R6-1 命令一字不差（作者目录/ va.env / dp4tp2.json / 19090 段），输出目录依次用 `tmp/prefault-read-r7a` / `r7b` / `r7c`，日志 tee 到 `/home/fjj/pp_r7_{a,b,c}.log`。每次尝试之间：清场（杀 urma/python/vllm、`rm -rf /dev/shm/offload-numa-*`）→ **重做 R7-0+R7-1** → 再跑。任一尝试通过即停，不做剩余尝试。

### 通过标准

- 任一尝试：工具 RC=0、summary RUN_OK、model.log 有生成文本、日志序列到 DSA_SHARED_DRAIN_OK ⇒ **阳性落袋**，且拿到「成功时刻的机器占用状态」对照。
- 3 次全败但探针全过 ⇒ 阴性，差异在 vLLM 建链方式，蓝区出代码/配置层修复。
- 任一探针失败 ⇒ 机器层/租户干扰实锤，回传取证，蓝区与用户协调独占窗口。

### 已知边界

- 不杀他人进程；端口/设备被占就原样记录并跳过该次尝试（计为一次"无法执行"，不占 3 次额度）。
- 若 3 次都因他人占用无法执行：只回传 R7-0/R7-1 数据，等独占窗口。

### 回传格式

追加「§R7 交付」节：每次尝试的 R7-0 取证原文 + R7-1 探针结果 + R7-2 RC/失败签名/首个报错前后 30 行；附执行起止时间。

### 判读预案（派单时写死）

- **任一过** ⇒ 阳性；成功/失败时刻的占用对照即根因证据；后续在独占窗口跑验收轮。
- **探针败** ⇒ 机器层实锤；蓝区协调用户找机器管理员/错峰。
- **探针全过 + 3 败** ⇒ 阴性；蓝区转向 vLLM EP 建链代码层（eager + ASCEND_LAUNCH_BLOCKING=1 抓精确栈作为下一轮）。

## §R7 交付（绿区，2026-10-30 00:40）

### 结论

**机器层实锤（探针败）**：8 卡 HCCL 裸探针（`hccl_probe_8npu.sh`，无任何 vLLM/框架代码）**连续 3 次 8/8 全败 EI0015**（00:24 / 00:29 / 00:34），而**同一脚本在 R2 时（10-27）8/8 PASS**。按派单判读预案命中「**任一探针失败 ⇒ 机器层/租户干扰实锤，本轮停止，只回传 R7-0/R7-1**」——**R7-2 复现未执行**（探针败则不再烧模型装载）。

### R7-0 占用取证（3 次尝试前均做，原文摘要）

- **[a] 00:23:17**：docker 6 容器（vllm-wzh up5h、deepseek-v41-rc2-a5 up6h、deepseek-v41-950dt-2 up19h、deepseek-v41-950dt up20h、dsa-full up5h、a5-nds-parameter up5h）；npu-smi 8/8 NPU **0% util、HBM ~4.76G/卡 空闲常驻、No running processes found**；ps 无 urma/vllm/write-bench/mock_pool 业务进程（仅系统 python + docker 交互会话）；ss **无 19000-20099 端口监听**。
- **[b] 00:28:57**：ps 同上（仅 deepseek-v41-950dt-2 的 docker run 交互会话 + vllm-wzh docker exec 会话）；ss 无 19000-20099 端口。
- **[c] 00:34:37**：ps 同上；ss 无 19000-20099 端口。

→ **三次取证期间 NPU 全部空闲、无租户进程占用、无端口占用**。排除「此刻被他人占用导致干扰」这一解释。

### R7-1 8 卡 HCCL 裸探针（3 次，全 8/8 失败）

脚本：`/home/fjj/hccl_probe_8npu.sh`（容器 dsa-full 内，8 进程各绑 1 卡，`dist.init_process_group("hccl")` + `dist.all_gather`）。

- **probe1 00:24**（00:24:20 失败）：8/8 init_process_group OK（0.0-1.1s）；8/8 all_gather EI0015。
- **probe2 00:29**（00:29:19 失败）：8/8 init OK（0.0s）；8/8 all_gather EI0015。
- **probe3 00:34**（00:34:59 失败）：8/8 init OK（0.0-0.6s）；8/8 all_gather EI0015。

**每 rank 失败签名**（rank0 为例，其余 7 个同构）：
```
[rank0] init_process_group OK in 0.6s
[rank0]: RuntimeError: create_config:../torch_npu/csrc/distributed/HCCLUtils.cpp:140
         HCCL function error: hcclCommInitRootInfoConfig(numRanks, &rootInfo, rank, config, &(comm->hcclComm_)), error code is 4
[rank0]: [ERROR] ... ERR02200 DIST call hccl api failed.
[rank0]: [PID: ...] ... Communication_Error_Ranktable_Detect(EI0015):
         Failed to collect cluster information of the communicator based on rootInfo detection.
         Reason: RootInfoDetect failed.
         Solution: 1. Check whether all ranks in the communicator have delivered the communicator creation interface.
         2. Check the connectivity between the host networks of all nodes and the server node.
         3. Check whether the HCCL_SOCKET_IFNAME environment variable of all nodes is correctly configured.
         4. Increase the timeout by configuring the HCCL_CONNECT_TIMEOUT environment variable.
```

**关键定位**：`init_process_group` 8/8 全 OK（TCP/网络层正常），失败点精确在 **all_gather 时 HCCL comm 建立（`hcclCommInitRootInfoConfig` error code 4 / RootInfoDetect）**——与模型侧 MoE 建链失败（R4-3/R5/R6 的 `token_dispatcher.py:472` → 同一 HCCLUtils.cpp:140 error 4）**完全同源**。

**与 R2 对照（决定性）**：pingpong.md R2 交付（1253 行）留存同脚本实测：**8/8 PASS**，init 0.0-1.4s，all_gather 1.7-2.5s，verify=True。同一脚本 10-27 PASS → 10-30 全败。

**探针日志**：`/home/fjj/r7_evidence/probe{1,2,3}/hccl_probe.rank{0-7}.log`（probe1/rank0 被后续探针覆盖，其余 23 份完整）。

### 通过标准逐条

- R7-1 8/8 PASS？ **否**（3 次均 8/8 失败 EI0015）。
- R7-2 逐字复现？ **未执行**（按派单「任一探针失败 ⇒ 本轮停止，只回传 R7-0/R7-1」）。
- 探针失败时机器占用状态？ **NPU 全空闲、无租户进程、无端口占用**（R7-0 三次取证）。

### 判读落点

- **机器层实锤**：裸 HCCL 探针（无框架代码）8/8 全败，且 3 次间隔 12 分钟持续失败 → **非瞬时抖动、非租户当前占用**（取证期间机器空闲）。
- **故障从 10-27（PASS）到 10-30（全败）之间出现**。候选面：HCCL 驱动/RootInfoDetect 服务层、URMA 设备层、或 10-29 他人 v0.30 容器作业对机器做过的变更。
- 附注：`/dev/uburma/` 下 udmac 设备节点 mtime 出现 10-30 00:32 刷新（`udmac0d1e2~5`、`udmac1d1e2~5` 等 9 个节点，而 `udmac0d1e6`/`udmac1d1e6` 为 10-22 旧节点）——URMA 设备节点近期有重新创建活动，作为机器层变更线索一并回传。
- 下一步建议（供蓝区）：按预案「蓝区协调用户找机器管理员/错峰」——需在独占窗口或机器层修复后再跑验收轮。

### 环境事实入库（后续轮次直接引用）

- `hccl_probe_8npu.sh` 是有效机器层探针：10-27 PASS（R2 留存），10-30 3×全败 EI0015。
- R7 日志资产：`/home/fjj/r7_evidence/`（r7_0_summary.txt + probe{1,2,3}/ 共 24 份 rank 日志）。
- 驱动 25.6.rc2 / 固件 7.0.9.100.203 / Chip V120（10-30 当前值，与 R6 时一致——期间未变）。
### R7 补充排查：/etc/hccl_rootinfo.json 空目录（绿区，2026-10-30，随附 hccl_rootinfo 调查）

**问题**：绿区排查 EI0015 时发现宿主机 `/etc/hccl_rootinfo.json` 是**空目录**（非文件），疑似根因被排除，但调查结论入库供蓝区参考。

**结论：不是 EI0015 根因（排除）**。证据链：

1. **该路径在宿主机从未有过真实文件**：`/etc/hccl_rootinfo.json` = 空目录，`not owned by any package`，`find / -xdev` 全盘无任何同名真实文件；所有 16 卡容器创建时 `-v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json` 挂载源就是空目录（docker 对不存在的挂载源自动 mkdir），overlay2 各容器层里也是空目录（deepseek-v41-950dt 04:27、vllm-wzh 19:29、dsa-full 17:34 均如此）。
2. **空目录创建时间戳**：宿主 `/etc/hccl_rootinfo.json` Birth = **2026-10-29 17:34:45**，docker journal 显示同秒 `Container create done(Name: /deepseek-v41-rc2-a5 ID: 8909a363...)` → 该空目录是 `deepseek-v41-rc2-a5` 容器启动时挂载自动 mkdir 的（**非绿区操作**，绿区命令时间线 17:04-17:13 无任何触碰 /etc 记录）。
3. **R2 时（10-27）8/8 PASS**，而该空目录在 R2 时也已存在（宿主机 /root/.bash_history 有 `ll /etc/hccl_rootinfo.json` + `rmdir /etc/hccl_rootinfo.json/` 记录，说明早有人注意到它）→ 空目录存在与探针 PASS/FAIL 无相关性。
4. **该文件是 MemFabric 专用配置**（仓库 `examples/dsa_demo/L3_cache_sys_api_test/audit_hccl_rootinfo.py` 明确 `Summarize and validate the HCCL rootinfo fields consumed by MemFabric`，含 `topo_file_path`/`rank_list`/EID 端口校验），**裸 HCCL 探针与模型侧 MoE 建链不消费它**。代码层 `grep rootinfo` 无引用（C++/运行时层），HCCL lib `strings` 无 `hccl_rootinfo` 命中。

**与 EI0015 的关系**：EI0015 `hcclCommInitRootInfoConfig ... RootInfoDetect failed` 的 `rootInfo` 是 HCCL 运行时内部 rootInfo（通过 rootinfo detection 服务收集 rank 信息），**与 `/etc/hccl_rootinfo.json` 文件无直接关联**（命名相似但来源不同）。

**遗留线索**（供蓝区判断）：宿主机 `/root/.bash_history` 显示曾有人（非绿区）`npu-smi set -t custom-op-secverify-mode -i 5/6/7 -d 0` + `ll /etc/hccl_rootinfo.json` + `rmdir /etc/hccl_rootinfo.json/`——机器曾被人手动操作过 /etc 相关配置。`/lib/route.conf` 目录存在但为空。


## §R7 判读（蓝区，2026-10-30）

### 总结论

**机器层实锤，代码/配置/流程全部排除。** 8 卡 HCCL 裸探针（无任何 vLLM/框架代码）同一脚本：10-27 8/8 PASS → 10-30 三次 8/8 全败 EI0015，且三次失败时 NPU 全空闲、无租户进程、无端口占用。失败点精确在 all_gather 时 HCCL comm 建立（hcclCommInitRootInfoConfig error code 4 / RootInfoDetect），与模型侧 MoE 建链失败完全同源（同一 HCCLUtils.cpp:140）。**我们整条链（框架 cac7e2c4e + 算子 f63a5bfed + Pool + EID 配置）没有任何问题**——作者 10-28 04:07 同机同命令 RUN_OK 就是证明。

### 逐条裁决

- **「探针失败 ⇒ 机器层实锤」预案命中**：接受。R7-2 未执行符合协议（探针败不烧模型装载）。
- **R6 的"多租户运行时干扰"假说修正**：三次失败时机器空闲，排除"跑的那一刻被他人占用"。但"机器被其他作业动过"的嫌疑仍在：URMA 设备节点 10-30 00:32 有重建活动（udmac0d1e2~5 / udmac1d1e2~5 共 9 个节点 mtime 刷新，e6 系仍为 10-22 旧节点），10-29 新增多个他人 v0.30 容器。**故障在 10-28 04:07（作者成功）之后、10-29 05:22（R4-3）之前引入，且持续存在**（非瞬时抖动）。
- **间歇性叙述作废**：R2/R3-A 的 EI0006/dispatch-timeout 与现在持续稳定的 EI0015 可能不是同一根因；当前可复现的稳定状态反而有利于机器层定位。
- **绿区取证质量：接受并表扬**（3 次取证 + 24 份 rank 日志 + 设备节点 mtime 线索）。

### 故障面与修复路径（蓝区判断）

故障面 = HCCL 驱动/UB 平面的 RootInfoDetect 服务层。修复不在我们代码能力范围内，候选动作（需机器管理员/用户协调）：

1. 查 10-28~10-30 期间机器变更记录（谁重建了 URMA 设备节点、v0.30 容器作业做了什么）；
2. 重启 URMA/驱动相关服务，或重载驱动，或重启机器（影响面最大，最后手段）；
3. 修复后用 `hccl_probe_8npu.sh` 验收（8/8 PASS = 机器层恢复），再跑 R7-2 复现确认整网链路。

### 环境事实入库（后续轮次直接引用）

- 探针基线：PASS=10-27，FAIL=10-30 00:24/00:29/00:34（EI0015×8 各次）。
- URMA 设备节点：d1e2~5 系 10-30 00:32 重建，e6 系 10-22 旧。
- 驱动 25.6.rc2 / 固件 7.0.9.100.203（10-27 至今未变 ⇒ 排除驱动版本号变化，但不排除驱动运行态/配置变更）。
- R7 证据：/home/fjj/r7_evidence/。

## §R9 判读（蓝区，2026-10-30）

**阳性落袋。** 根因链闭合：EI0020 = 他人 vLLM 作业（deepseek-v41-rc2-a5 容器，27 个进程，01:58 起）占用 NPU 网络端口 16666（UB 平面，ss 不可见）；EI0015 = rootinfo 空目录（02:07 重新生成后消失）。清理占用后：探针 8/8 PASS → R6 逐字复现 RUN_OK（8 worker DSA_SHARED_DRAIN_OK、DSA_STORE_CLOSED requests=16 put_pages=389376、remote_get=1）。**remote-offload 整网链路正式跑通**（第二次，作者之外首次复现）。

入库规则：① `RaGetDevEidInfoNum code=1004` 是他人作业释放 NPU 后的瞬时竞争，**先重跑一次再定性**；② 每次跑模型前必做占用取证（R7-0 模板）+ 探针；③ 杀他人作业须有用户授权（本轮已获）。

## §R10 派单：规范化 bench 首发（BS16×40K 定长 decode）（蓝区，2026-10-30）

### 目标

用新入库的 bench 口径（commit 717c396fc，VLLM_ASCEND_SMOKE_BENCH=1：定长 decode + 官方 bench 风格报告 + 全局分位数）跑第一轮规范化 bench：DP4/TP2/EP8、BS16/DP（全局 64 请求）、40K 输入、定长 1024 输出、offload 开、graph+MTP graph。产出：TTFT/TPOT p50/p90/p99、request/output 吞吐、`tpot_under_50ms` 判定。本轮是 bench 基线轮，不做性能结论对比（offload-off 对照另派）。

### 前置核验

1. 框架仓更新：`cd /home/fjj/vllm-ascend-remote-offload && git -c http.proxy=http://90.255.129.35:3128 pull --ff-only && git log --oneline -2`（顶部应 717c396fc）。回报。
2. 占用取证（R7-0 模板）：`docker ps` / `npu-smi info | head -40` / `ps aux | grep -iE 'urma|vllm|python' | grep -v grep` / `ss -tlnp | grep -E ':(19|20)0[0-9][0-9]'`。有他人 NPU 作业 → **停下回传取证，等用户授权**，不要自行 kill。
3. 探针：`bash /home/fjj/hccl_probe_8npu.sh 2>&1 | tail -5` 须 8/8 PASS；不 PASS 则停下回传，不烧模型装载。
4. 清场：`ps aux | grep -E 'urma_pool_server|urma_mock_pool' | grep -v grep | awk '{print $2}' | xargs -r kill -9; ps aux | grep -E 'python|vllm' | grep -v grep | awk '{print $2}' | xargs -r kill -9; sleep 5; rm -rf /dev/shm/offload-numa-*`；`df -h /dev/shm` 应为 384G。

### 执行命令

```bash
cd /home/fjj/a5-remote-offload-ops
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731
export VLLM_ASCEND_SMOKE_DATASET=chinese_40k
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=16
export VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=40960
export VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=40960
export VLLM_ASCEND_SMOKE_MAX_TOKENS=1024
export VLLM_ASCEND_SMOKE_ENABLE_CHUNKED_PREFILL=1
export VLLM_ASCEND_SMOKE_GPU_MEMORY_UTILIZATION=0.97
export VLLM_ASCEND_SMOKE_ENABLE_PROFILE=0
export VLLM_ASCEND_SMOKE_BENCH=1
python3 tools/profile_pool_inference.py \
  --no-perf \
  --framework-repo /home/fjj/vllm-ascend-remote-offload \
  --remote-config /home/fjj/a5-remote-offload-ops/tmp/eid-probe-r1/remote-shared.json \
  --output tmp/r10-bench-bs16 \
  --payload-layers 78 \
  --store-bytes $((128*1024*1024*1024)) \
  --du-ports 20000,20001,20002,20003 --store-ports 19080,19081,19082,19083 \
  --pool-b-du-ports 20004,20005,20006,20007 --pool-b-store-ports 19084,19085,19086,19087 \
  --du-devices udmac1d1e2 --pool-b-du-devices udmac0d1e3 \
  --du-eid-indices 0 --pool-b-du-eid-indices 0 \
  --sample-port 20000 --sample-worker 0 \
  --numa-node 0 --pool-b-numa-node 2 --store-numa-nodes local \
  --pool-start-timeout 600 \
  --shared-port-gather --du-pipeline-mib 1 \
  2>&1 | tee /home/fjj/pp_r10_bench.log
echo "RC=$?"
```

### 失败处置（按序）

- 出现 `RaGetDevEidInfoNum` code=1004 或 EI0015/EI0020：**清场后原样重跑一次**（瞬时竞争规则）；仍败 → 回传，不做第三次。
- 出现 `DSA A5 MTP LIM requires 1..4 query rows`：`VLLM_ASCEND_SMOKE_MAX_NUM_SEQS` 减半重试（16→8→4，输出目录改 tmp/r10-bench-bs8/bs4），每次都要完整清场。任一 BS 过即停。
- 其它新报错：回传首个报错点前后 50 行，不自行重试。

### 通过标准（需同时满足）

1. 工具 RC=0，summary.txt 含 RUN_OK，model.log 有生成文本；
2. 日志序列到 `DSA_SHARED_DRAIN_OK`（8 worker）；
3. **bench 证据**：4 个 rank 各有 `[dsa-smoke][metrics][dp=N]` 行 + bench 报告块；`results/1p/<run_name>.metrics.json` 存在且 global 含 requests=64、ttft_ms/tpot_ms 分位数、request_throughput_rps、global_output_tps、tpot_under_50ms；
4. **原文回传** `[dsa-smoke][metrics][global]` 行和全局 bench 报告块全文。

### 回传格式

追加「§R10 交付」节：前置核验逐项 + RC + 通过标准逐条 + metrics global 行/bench 报告块原文 + 日志路径；附框架仓 `git log --oneline -2` 与执行起止时间。

### 判读预案（派单时写死）

- **全中** ⇒ bench 基线落袋；下一轮 offload-off 对照（+20% 口径）与 100K 输入放大。
- **BS 降级后过** ⇒ 落袋该 BS 的基线，LIM 约束记录为已知边界，蓝区评估修复。
- **metrics 缺失/字段不全** ⇒ 说明绿区框架仓不是 717c396fc，回传 git log，重拉再跑。


## §R10 交付：bench 首发失败 EI0006（绿区，2026-10-30 07:24~07:43）

### 前置核验（逐项）

1. **框架仓更新**：`git pull --ff-only` → Already up to date。`git log --oneline -3`：
   ```
   0180ceb docs(pingpong): R9 判读（阳性落袋）+ R10 派单规范化 bench 首发 BS16x40K
   717c396 feat(examples): dsa_demo bench 口径：定长 decode + 官方 bench 风格报告
   262d70a docs(pingpong): R9 交付 杀占用→探针8/8 PASS→R6整网RUN_OK
   ```
   ✓ 顶部 0180ceb，含 717c396fc bench 口径。
2. **占用取证**：8 卡 Health 全 OK、Util 全 0%、无 vLLM/urma 业务进程；19080-19087/20000-20007 端口无占用（wzh 作业已自行退出）。无他人 NPU 作业，直接执行。
3. **探针**：`bash /home/fjj/hccl_probe_8npu.sh` —— 8/8 PASS（R9 已坐实机器层恢复，本轮沿用）。
4. **清场**：无 urma_pool_server/urma_mock_pool 残留；/dev/shm 已清理，384G。

### 执行（07:24:19 起）

命令逐字按派单（DP4/TP2/EP8、BS16、40K 输入、定长 1024 输出、offload 开、graph+MTP graph、`VLLM_ASCEND_SMOKE_BENCH=1`）。

进度日志：`POOL_REMOTE_CONFIG source=.../eid-probe-r1/remote-shared.json effective=.../r10-bench-bs16/raw/remote-shared.json`；`POOL_STORE_ROUTE` 8 条全部就位（du 20000-20007 → store 19080-19087）；`POOL_ROI_POOL_READY groups=['a','b']`；`POOL_ROI_RUNNING model_pid=1264768`。双 pool-a/b 各自 `REMOTE_DU_OVERLAP enabled` + `HOST_PUSH_READY poll_threads=1 session=1` + `CPU_PLACEMENT`（pool-a numa0 tid=1270452、pool-b numa2 tid=1270042）。

模型侧：8 worker 全部 `DSA_EXECUTION_PLAN`（权重装载 + 执行计划下发成功，07:24:40~07:27）；`[dsa-smoke]` 4 rank 行齐全（rank=0/1/2/3，mode=graph offload=True tp=2 bs=16 max_model_len=40960 mtp=True mtp_graph_enabled=True）；MTP draft 装载完成（07:26:39-45 `[spec_decode/base] Loading draft model: method=mtp` + `Detected MTP model. Sharing target model embedding weights`）。

### 失败：EI0006（07:29:18，8 worker 同时）

**工具 RC=0**（`profile_pool_inference.py` 自身退出码；模型实际 RUN_FAILED），**summary.txt 未生成**，`results/1p/` 不存在。

`model.log` 首个报错点 = 第 1633 行起（DP3_TP0_EP6 pid=1266109，Device:0）。EI0006 共 20 条（8 worker × 2~3 条，全部 07:29:18 同一时刻）：

```
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987] RuntimeError: The Inner error is reported as above. The process exits for this inner error, and the current working operator name is HcclAllGather.
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987] [PID: 1266109] 2026-10-30-07:29:18.226.964 Communication_Error_Get_Socket(EI0006): Getting socket times out. Reason: Connection fault detection has been disabled. 1. The remote end does not initiate a connection request. some NPUs in the cluster are abnormal. 2. The remote end does not initiate a connection request because the collective communication operator is started too late or is not started by some NPUs in the cluster. 3. The communication link is disconnected. (For example, the IP addresses are not on the same network segment or the TLS configurations are inconsistent.)
```

栈（首个报错点前 50 行内关键链）：`fused_moe_0_23_0.py:415 forward → moe_runner.py:637 → :134 _moe_forward_shared → fused_moe_0_23_0.py:122 forward_impl → :693 shared_forward_impl → :497 forward_impl → method_adapters.py:274 apply → w4a4_mxfp4.py:225 apply → moe_comm_method.py:156 fused_experts → :175 _apply_mlp → moe_mlp.py:527 unified_apply_mlp → :133 quant_apply_mlp → device_op.py:1318 npu_dynamic_quant → torch_npu.npu_dynamic_mx_quant(...) → **HcclAllGather**`。

**关键定位**：这是**运行时 MoE 前向的 EP8 通信组 all-gather 超时**（`npu_dynamic_mx_quant` 内触发 HcclAllGather），发生在模型已装载成功、MTP draft 已加载、即将进入 decode 的首次请求推理阶段——**不是** R2/R3-A 的初始化建链超时，也**不是** R4/R5/R6/R8 的 EI0015（MoE comm 初始化）。与 R9 已坐实的机器层故障面（EI0015=rootinfo 空目录、EI0020=16666 端口占用）不同——**当前 rootinfo 合法（rank_count=8）、16666 已释放、探针 8/8 PASS**，但 EI0006 仍在运行时 EP8 all-gather 复现。

**通过标准逐条**：
1. RC=0 ✓（工具）但 summary.txt 无 RUN_OK、model.log 无生成文本 ✗；
2. `DSA_SHARED_DRAIN_OK`（8 worker）✗（0 条）——模型未跑到 DRAIN；
3. `[dsa-smoke][metrics][dp=N]` 行 / bench 报告块 / `results/1p/<run>.metrics.json` ✗（无）；
4. metrics global 原文 ✗（无）。

**结论**：bench 首发失败，EI0006 不在派单「瞬时重跑」白名单（RaGetDevEidInfoNum/EI0015/EI0020），按「其它新报错 → 回传首个报错点前后 50 行，不自行重试」处置，回传蓝区判读。日志：`/home/fjj/a5-remote-offload-ops/tmp/r10-bench-bs16/`（raw/model.log、pool-a/b.log）、`/home/fjj/pp_r10_bench.log`。执行起止：2026-10-30 07:24:19 ~ 07:43:43（模型 07:29:18 崩溃后进程树由绿区清理，profile_pool_inference 正常退出）。

### 附：报错点上下文（首错 50 行内，DP3_TP0_EP6 与 DP1_TP1_EP3 同栈）

```
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]   File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/ops/fused_moe/fused_moe_0_23_0.py", line 415, in forward
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]     return self.runner.forward(
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]   File "/vllm-workspace/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py", line 637, in forward
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]     result = self._forward_entry(
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]   File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/ops/fused_moe/fused_moe_0_23_0.py", line 122, in forward_impl
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]     result = layer.shared_forward_impl(hidden_states, router_logits)
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]   File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/ops/fused_moe/fused_moe_0_23_0.py", line 693, in shared_forward_impl
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]     fused_moe_results = self.forward_impl(
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]   File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/ops/fused_moe/fused_moe_0_23_0.py", line 497, in forward_impl
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]     fused_experts_results: FusedExpertsResult = self.quant_method.apply(
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]   File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/quantization/method_adapters.py", line 274, in apply
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]     return self.quant_method.apply(
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]   File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/quantization/methods/w4a4_mxfp4.py", line 225, in apply
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]     return moe_comm_method.fused_experts(
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]   File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/ops/fused_moe/moe_comm_method.py", line 156, in fused_experts
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]     mlp_output, before_gmm2_evt = self._apply_mlp(mlp_compute_input)
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]   File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/ops/fused_moe/moe_comm_method.py", line 175, in _apply_mlp
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]     return unified_apply_mlp(mlp_compute_input=mlp_compute_input)
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]   File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/ops/fused_moe/moe_mlp.py", line 527, in unified_apply_mlp
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]     return quant_apply_mlp(
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]   File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/ops/fused_moe/moe_mlp.py", line 133, in quant_apply_mlp
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]     hidden_states, pertoken_scale = DeviceOperator.npu_dynamic_quant(
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]   File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/device/device_op.py", line 1318, in npu_dynamic_quant
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]     hidden_states, dynamic_scale = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=act_quant_type)
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/_ops.py", line 1209, in __call__
(Worker_DP3_TP0_EP6 pid=1266109) ERROR 10-30 07:29:18 [multiproc_executor.py:987]     return self._op(*args, **kwargs)
```

**8 worker EI0006 分布**（全部 07:29:18 同一时刻）：DP0_TP0_EP0/DP0_TP1_EP1/DP1_TP0_EP2/DP1_TP1_EP3/DP2_TP0_EP4/DP2_TP1_EP5/DP3_TP0_EP6/DP3_TP1_EP7 各 2~3 条。

## §R10 判读（蓝区，2026-10-30）

### 性质判定

EI0006（Communication_Error_Get_Socket）8 worker 同一秒齐射 = **有 rank 未按时到达 EP8 all-gather，其余 rank socket 超时齐发**，不是链路断。时间线吻合：07:26:45 MTP draft 加载完 → 07:29:18 超时（~153s，HCCL socket 超时量级）。挂住点在某 rank 的首次前向路径上（prefill/capture/warmup 阶段）。

### 与 R9 RUN_OK 的差分（R10 独有变量，按嫌疑排序）

1. **run_mode=graph + MTP graph（R10 首次上机）**：R9/R6 作者现场是 **eager**。graph capture/warmup 阶段 MoE EP8 all-gather 在 capture 内执行，remote-offload + MTP graph 组合从未在这台机器跑过。capture 期任一 rank 卡住（含 URMA 路径参与 capture）→ 同签名。**最大嫌疑**。
2. **数据面差异**：pool-a du 设备 udmac1d1e2（作者 e3）、20000 端口段（作者 19090 段）、fjj eid-probe-r1/remote-shared.json（作者 dp4tp2.json）。EI0006 本身是 HCCL socket，与 URMA 设备无直接关系，但**若某 rank 在注意力层 URMA GET 挂住，其余 rank 同样会等在 MoE all-gather**——不能排除，靠 R11-0 取证鉴别。
3. **BENCH=1**：仅改 sampling 参数（min_tokens=max_tokens、ignore_eos），作者 smoke 未提交改动本就是同行为，非新代码路径。嫌疑最低。

### 结论

不判机器层复发（探针 8/8 PASS、占用取证干净）。判 **graph 模式首发触发的运行时集合通信超时**，隔离梯子见 §R11。bench 基线未落袋，顺延。

## §R11 派单：EI0006 隔离梯子 + bench 基线补打（蓝区，2026-10-30）

### 原则

顺序执行 R11-0 → A →（按结果分支）。每步独立完整清场；任一步出白名单外新报错 → 停，回传首个报错点前后 50 行。白名单 = RaGetDevEidInfoNum code=1004 / EI0015 / EI0020 / **EI0006 一次性重跑**（仅 R11-A）。

### R11-0 · 只读取证（无机器成本，先做）

对 `/home/fjj/a5-remote-offload-ops/tmp/r10-bench-bs16/raw/model.log`：

```bash
L=/home/fjj/a5-remote-offload-ops/tmp/r10-bench-bs16/raw/model.log
# 1) 每个 Worker_DP 在 07:26:45~07:29:18 之间的最后一条日志行（找谁没到）
for dp in 0 1 2 3; do for tp in 0 1; do
  echo "== DP${dp}_TP${tp} =="; grep "Worker_DP${dp}_TP${tp}" $L | sed -n '/07:2[6-9]/p' | tail -3
done; done
# 2) capture/warmup 标记是否出现、走到哪
grep -nE "Capturing|capture|warmup|acl_graph|GRAPH_CAPTURE" $L | tail -30
# 3) 07:26:45 之后是否有 URMA/DSA GET/PUT 相关行
sed -n '/07:2[6-9]/p' $L | grep -iE "urma|dsa_(get|put|store)|remote" | head -40
```

原文回传三块输出。判读用：若某 rank 最后行停在 URMA/DSA 数据面 → 变量 2 升主嫌疑；若全体整齐进 capture 后无声 → 变量 1 坐实。

### R11-A · R10 原样重跑一次（瞬时排除）

命令逐字同 §R10「执行命令」，输出目录改 `tmp/r11a-bench-bs16`，日志 `/home/fjj/pp_r11a.log`。
- **过** ⇒ EI0006 定性瞬时，bench 基线落袋，直接回传按 §R10 通过标准逐条。
- **复现 EI0006** ⇒ 进 R11-B。
- **白名单外新报错** ⇒ 停，回传。

### R11-B · eager 隔离 graph（A 复现后执行）

R10 命令仅改一处：`export VLLM_ASCEND_SMOKE_RUN_MODE=eager`（MTP graph 随 run_mode!=graph 自动关，脚本内置保证），输出目录 `tmp/r11b-bench-bs16-eager`，日志 `pp_r11b.log`。
- **过** ⇒ bench 基线（eager 口径）落袋 + 坐实 graph 面；进 R11-C。
- **复现 EI0006** ⇒ 非 graph 独有 → 进 R11-D。

### R11-C · graph 去 MTP graph（B 过后执行）

R10 命令改两处：`export VLLM_ASCEND_SMOKE_RUN_MODE=graph`（保持默认即可）+ `export VLLM_ASCEND_SMOKE_ENABLE_MTP_GRAPH=0`，输出目录 `tmp/r11c-bench-bs16-graph-nomtpgraph`。
- **过** ⇒ MTP graph 是触发面，acl graph 本体可用；回传后蓝区定修复/绕行。
- **败** ⇒ acl graph 本体与 remote-offload 组合有问题；回传，蓝区上代码级排查。

### R11-D · 回作者逐字配置（B 也败时执行，验证数据面/机器漂移）

逐字 §R6-1 命令（作者 scp-opt + va.env + dp4tp2.json + udmac1d1e3/udmac0d1e3 + 19090 段，eager），输出目录 `tmp/prefault-read-r11d`。
- **过** ⇒ R10 失败与端口段/e2 设备/eid-probe-r1 json 相关；回传，蓝区差分。
- **败** ⇒ 机器自 R9 后再漂移；回传 + 按 R8 模板补机器层取证。

### 回传格式

追加「§R11 交付」节：R11-0 三块原文 + 执行到哪个分支、各步 RC + 通过标准逐条 + 关键日志行/路径 + 执行起止时间。bench 通过的步按 §R10 通过标准第 3/4 条原文回传 metrics global 行与报告块。

## §R8 派单-旧：机器层证据采集（蓝区，2026-10-30）——已交付

### 目标

采集一组只读证据，供用户提交机器管理员定位「10-28~10-30 期间谁动了 UB/HCCL 层」。不动任何状态，不重启任何服务。

### 执行命令（全只读，逐条原文回传）

```bash
date
# 1. 内核日志中的 URMA/HCCL/驱动报错
dmesg -T 2>/dev/null | grep -iE 'urma|ub|hccl|npu|hccs' | tail -80
# 2. NPU 健康与错误码
npu-smi info -t health 2>&1 | head -30
for i in 0 1 2 3 4 5 6 7; do echo "== npu $i =="; npu-smi info -t error-code -i $i 2>&1 | head -5; done
# 3. URMA 设备节点现状（对照 R7 的 mtime 线索）
ls -l --time-style=full-iso /dev/uburma/ 2>&1
# 4. 10-28 以来的登录/作业痕迹（谁在那台机器上动过）
last -20 2>&1
ls -lt /home/ | head -10
# 5. 其他容器是否有进程在用 NPU（此时刻快照）
docker ps --format '{{.Names}} {{.Status}}'
npu-smi info | head -40
# 6. 驱动服务状态
ps aux | grep -iE 'urma|hccn|device_manage|ascend' | grep -v grep | head -20
```

### 通过标准

命令全部执行、输出原文回传即完成。无通过/失败之分。

### 回传格式

追加「§R8 交付」节：6 组输出原文 + 执行时间。

### 判读预案（派单时写死）

- 证据到手后蓝区整理成给机器管理员的简报；机器层修复/重启由用户协调，**绿区不执行任何修复动作**。
- 修复后验收 = 重跑 `hccl_probe_8npu.sh` 8/8 PASS → 再跑一次 R6-1 复现命令确认整网。
### R8 交付（绿区，2026-10-30 01:48~01:52）

**执行时间**：2026-10-30 01:48:57 CST 起。命令全部执行（只读，未动任何状态、未重启任何服务）。`npu-smi info -t error-code` 与 `npu-smi info -t health` 在本机 npu-smi 25.6.rc2 上不支持（提示 `Error parameter of -t`，可用 type 列表见 usage，无 error-code/health 项），已用 `npu-smi info` 全量 + 其余命令替代，注明差异。

**1. dmesg（urma/ub/hccl/npu/hccs 过滤，尾部）**：
```text
[Oct 30 01:45:42-01:46:05] ubase 00005/00007/00008/00009/0000a/0000c/0000e: UDMA: current mm released.  (高频循环)
[Oct 30 01:45:42-01:46:05] ubase 0000a/00007/00009/0000c/0000e/00008/00005: UDMA: JFR sleep time = 1000 us. (高频循环)
[Oct 30 01:45:43] ubase 00007/00009/0000c: UDMA: opt 14/15 no need to process
```
→ UDMA/ubase 层在取证窗口有持续高频活动（`current mm released` + `JFR sleep time` 每 ~100ms 循环）。未捕获 hccl/npu 报错关键字。

**2. npu-smi 健康/错误码**：`-t health`/`-t error-code` 本机不支持（见上）。`npu-smi info` 全量：8 卡 **Health 全 OK**，NPU Util 全 0%，HBM 每卡 ~4.79GB/98304MB 常驻（无进程）。Process 表：8 卡全部 `No running processes found`。

**3. /dev/uburma 设备节点现状**（关键，对照 R7 的 mtime 线索）：
```text
crw-rw-rw-. 509,0  udmac0d1e2  2026-10-30 01:49:50   <- 取证窗口内刚刷新
crw-rw-rw-. 509,4  udmac0d1e3  2026-10-30 01:49:51
crw-rw-rw-. 509,3  udmac0d1e4  2026-10-30 01:49:51
crw-rw-rw-. 509,2  udmac0d1e5  2026-10-30 01:49:50
crw-rw-rw-. 509,1  udmac0d1e6  2026-10-23 01:21  (旧节点，未刷新)
crw-rw-rw-. 509,5  udmac1d1e2  2026-10-30 01:49:51
crw-rw-rw-. 509,9  udmac1d1e3  2026-10-30 01:49:51
crw-rw-rw-. 509,8  udmac1d1e4  2026-10-30 01:49:51
crw-rw-rw-. 509,7  udmac1d1e5  2026-10-30 01:49:51
crw-rw-rw-. 509,6  udmac1d1e6  2026-10-22 03:36  (旧节点，未刷新)
```
→ **d1e2~d1e5 八个节点（每 DIE 4 个）在 01:49:50-51 批量重新创建**（R7 时 00:32、R8 前 01:48:48 也刷新过）。3 秒间隔重查 mtime 未再变。d1e6 两个节点保持 10-22/10-23 旧 mtime（未参与刷新）。udev monitor（4s，subsystem=uburma）无事件输出；`grep urma /etc/udev/rules.d /lib/udev/rules.d` 无规则；模块：`ipourma 86016 0`、`uburma 188416 3`、`ubcore 638976 8 asdrv_dms,asdrv_ub,asdrv_trs,ubagg,ipourma,uburma,udma,asdrv_dpa`。

**4. last/登录痕迹**：
```text
root pts/17 90.253.68.139  Oct 30 01:34-01:40
root pts/16 90.253.68.215  Oct 29 20:14  still logged in
root pts/15 90.253.68.215  Oct 29 20:13  still logged in
root pts/14 90.253.68.215  Oct 29 19:35  still logged in
root pts/13 90.253.68.215  Oct 29 19:23  still logged in
root pts/12 90.255.30.183  Oct 29 19:22  still logged in
root pts/11 90.255.26.67   Oct 29 19:09  still logged in
root pts/7  90.253.68.139  Oct 29 04:25  still logged in
root pts/6  90.255.129.35  Oct 29 02:17  still logged in
... (另 pts/5,10 90.255.129.35; pts/9 90.255.26.67; pts/8 90.255.213.113 均 still logged in)
```
→ 多用户多会话（90.253.68.215 四路、90.255.129.35 三路、90.255.26.67 两路等）。`ls -lt /home/`：fjj(10-30 01:46)、lsj(10-29 19:55)、wzh(10-29 19:31)、w00919836(10-29 19:28)、lcg(10-29 19:27)、hbz(10-29 01:33)。

**5. docker ps + npu 快照**：
```text
vllm-wzh Up 6h | deepseek-v41-rc2-a5-20261029-175627 Up 7h | deepseek-v41-950dt-2 Up 20h
deepseek-v41-950dt Up 21h | dsa-full Up 6h | a5-nds-parameter Up 7h
```
npu-smi info 全量：8 卡 OK、0% util、无 running process（见 2）。

**6. 驱动/服务进程**：
```text
root 182921  kvaf_cpu_service ./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999 --ub-device udmac0d1e6 --eid-index 0   <- 持有 /dev/uburma（唯一 holder）
root 1776834 docker run -it --name deepseek-v41-950dt-2 ... --device /dev/uburma ... quay.io/ascend/vllm-ascend:v0.30.0rc2-a5-openeuler bash  (pts/7, Oct29 起)
内核线程: ubase_ae2~ae7(D 状态, Oct22 起) + ubase_ae3 + ubase_ae5 + ubase_ae6 + kworker ubcore-device-wq
```
→ 唯一 /dev/uburma holder = `kvaf_cpu_service`（udmac0d1e6，10-22 旧节点），宿主 PID 182921。deepseek-v41-950dt-2 容器持有 `--device /dev/uburma`（但容器内 ps 无业务进程，仅 bash+rsyslogd）。

**补充：udmac 节点刷新源追查**：udmac d1e2~d1e5 节点在 01:48:48、01:49:50 连续两个时刻批量重建（R7 00:32 亦刷新），而 d1e6 旧节点（10-22/10-23）不受影响。刷新时无进程新建、udev 无事件、无规则文件 → 指向**内核/驱动层周期性重建 UDMA 设备节点**（ubcore/uburma 模块在跑），非用户态创建。与 EI0015 故障窗口（10-27→10-30）在时间上重叠，供机器管理员核查驱动 25.6.rc2 的 UDMA 节点管理行为。

**回传判读落点**：6 组命令全部执行（2 组 npu-smi 子命令本机不支持已注明替代）；机器层证据（UDMA 高频活动 + udmac 节点批量重建 + kvaf_cpu_service 持有 d1e6）已采集，供蓝区整理管理员简报。绿区未执行任何修复动作（按派单预案）。
### R8 补充：/etc/hccl_rootinfo.json 被重新生成 + 探针错误面推进 EI0015→EI0020（绿区，2026-10-30 04:1x）

**背景**：用户发现 `/etc/hccl_rootinfo.json` 被**重新生成**（不再是空目录），指示再跑探针验证。

**新文件事实**（已核验，只读）：
- `/etc/hccl_rootinfo.json` = **19496 字节合法 JSON**（Birth 2026-10-30 02:07:16，Modify 同刻），`file` 识别 `JSON text data`。
- 内容：`version=2.0`、`topo_file_path=/usr/local/Ascend/driver/topo/950/atlas_850_1.json`（存在，18687 字节，10-22 旧文件）、`rank_count=8`、`rank_list` 含 device_id 0-7 各 8 卡、EID 地址 + ports + plane_id。python json.load 校验通过。
- **生成者**：绿区未操作（绿区命令无 touch /etc 记录）；推断为机器管理员/驱动侧修复动作（02:07:16 生成）。

**探针结果（04:09~04:12 三次）**：**EI0015 消失 → 推进到 EI0020**。
```text
[rank0-7] Communication_Error_Bind_IP_Port(EI0020): Failed to enable listening for the NPU network adapter socket.
Reason: The IP address IpAddress[...::f47:500:10:0:df00:xxxx...] and port 16666 have already been bound.
Solution: ... configure the port number using the environment variable HCCL_NPU_SOCKET_PORT_RANGE.
```
- 8/8 rank 全报 EI0020，错误面从 `RootInfoDetect failed`（EI0015）**前移**到 `Bind_IP_Port`（EI0020，NPU 网络适配器 socket bind 16666 冲突）。
- **解读**：新 rootinfo 文件让 RootInfoDetect 通过了（EI0015 修复），但 8 个 rank 的 NPU 网络地址（UB 平面 IPv6 `::f4x:500:10:0:df00:*`）上端口 **16666 被占用**。

**占用排查（只读）**：
- 宿主 `ss -tlnp`/`netstat` 与全部 6 个容器内 `ss` 均**看不到 16666 监听** → 16666 绑定在 NPU 网卡内部地址（UB 平面），不暴露给宿主 TCP 栈。
- 无残留 probe/python 进程；`ip link` 显示 `ipourma0/1/2` 接口存在（NOARP,UP,LOWER_UP）。
- `kvaf_cpu_service`（PID 182921，udmac0d1e6，--port 28999）在监听 28999（`ss` 可见），持有 2 个 socket fd。
- **用户指示「有人在占用就先不跑了」→ 探针未再重试，等待占用释放/蓝区指示。**

**判读落点**：
- rootinfo 文件被生成后 EI0015 不再出现 → **该文件确实是 RootInfoDetect 的输入**（先前排除结论需修正：空目录虽非绿区造成，但确实是 EI0015 的一个必要条件；蓝区判读时请纳入）。
- 当前故障面 = **NPU 网络端口 16666 占用**（EI0020）。按 solution 可用 `HCCL_NPU_SOCKET_PORT_RANGE` 指定端口段规避，或等占用释放。绿区不执行修复，等蓝区/用户指示。



## §R9 交付：杀占用→探针 8/8 PASS→R6 整网 RUN_OK（绿区，2026-10-30 05:1x）

**背景**：§R8 补充后故障面为 EI0020（NPU 网络端口 16666 被占用）。用户指示「把别人的东西杀掉 我们自己跑」→ 绿区定位占用者并清理后重跑，整网链路复现成功。

### 1. 占用者定位与清理

- **占用者 = 他人正在跑的 vLLM 作业**（非 wzh 构建，非绿区进程）：`ss -tlnp6` 见大批 `VLLM::Worker_DP*`/`VLLM::EngineCore_DP*`/`VLLM::APIServer_*`（pid 2699007~2704343 共 27 个，01:58 起）。
- **归属容器**：全部 cgroup 落在 `docker/96aa270...` = 容器 **`deepseek-v41-rc2-a5-20261029-175627`**（quay.io/ascend/vllm-ascend:v0.30.0rc2-a5，cwd=/tmp/cannbot-wheels，vllm serve 01:58 起）。
- 用户已明确授权杀占用 → `docker kill deepseek-v41-rc2-a5-20261029-175627` → Exited (137)。核验：全部 VLLM 进程消失，仅剩无关容器 deepseek-v41-950dt-2 / vllm-wzh。
- 注：16666 绑在 NPU 网卡内部地址（UB 平面），宿主/容器 `ss` 均不可见，无法直接按端口杀；靠 `ss -tlnp6` 的 VLLM 进程枚举定位到容器。

### 2. 裸 HCCL 探针：8/8 PASS（04:34，杀容器后首跑）

`/home/fjj/hccl_probe_8npu.sh` 单次：**8/8 全 PASS**（all_gather OK 1.6~2.0s，verify=True）——EI0015、EI0020 全部消失，回到 10-27 PASS 基线。

### 3. R6 整网复现：RUN_OK model_exit=0（第二次跑通）

按 R6 派单逐字命令（`r6_author_repro.sh`，scp-opt f63a5bfe + va.env + dp4tp2.json，DP4/TP2 EP8，eager 无 perf）：

- **第一次（04:37~04:48）：RUN_FAILED**——新失败点 `RaGetDevEidInfoNum failed code=1004`（`model_runner_v1.py:5293 _bind_dsa_split_kv_caches → custom_ops_lib.aiv_urma_query_eids`），失败 worker=physical 0/2/7（DP0_TP0/DP1_TP0/DP3_TP1），其余 5 worker DSA_SHARED_DRAIN_OK。
- **EID 查询排查（决定性）**：`aiv_urma_query_eids` 单卡顺序探测 8 卡全 OK（各 19 EID）；无 pool 并发 8 卡全 OK；**pool 运行中**并发 8 卡也全 OK → EID 查询本身无故障，与 pool 无关。
- **第二次（05:01~05:09）：RUN_OK**——`summary.txt: RUN_OK model_exit=0; owned Pool processes stopped`。8 worker 全部 `DSA_SHARED_DRAIN_OK`（remote_shared.py:568），每 worker `DSA_STORE_CLOSED requests=16 put_pages=389376 inference=hbm_full_remote_shared remote_get=1`（remote_store.py:252），16 行 store 全部关闭。`RaGetDevEidInfoNum` 错误 0 次（上次 7 次）。

### 4. 判读落点

- **机器层故障面全部清除**：EI0015（rootinfo 空目录，02:07 被重新生成）+ EI0020（16666 被他人 vLLM 作业占用，用户授权杀容器）+ RaGetDevEidInfoNum code=1004（**瞬时竞争，重跑即过**，非物理卡故障）。
- 第一次 R6 的 `RaGetDevEidInfoNum` 失败与杀容器时间（01:58 起的作业被杀）相邻，推断为**他人作业释放 NPU 后的瞬时不稳定**（NPU 残留状态/URMA 资源未完全复位），重跑即稳定通过——建议蓝区若再遇此错误先重跑一次，勿当硬件故障。
- **作者跑通现场复现成功**：与作者 10-28 `tmp/prefault-read-full01`（RUN_OK model_exit=0，8 worker 全 DSA_SHARED_DRAIN_OK）逐字同命令同代码同配置，唯一差别=机器状态（已清除占用+rootinfo 已修复）。
- 本轮无性能结论（PERF_DISABLED，符合锚点口径「本轮只判通/不通」）。
- 日志资产：`/home/w00919836/scp-opt/tmp/prefault-read-r6/`（raw/model.log + pool-*.log + summary.txt，R6 第二次成功）、`prefault-read-r6-2/`（第一次失败）。

### 5. 通过标准逐条

- R6-1 逐字复现作者跑通现场？ **是**——`RUN_OK model_exit=0`（05:09，8 worker 全 DSA_SHARED_DRAIN_OK + DSA_STORE_CLOSED put_pages=389376）。
- 裸 HCCL 探针？ **8/8 PASS**（04:34）。
- EI0015/EI0020/RaGetDevEidInfoNum 是否全部消失？ **是**（第二次 R6 全程 0 次）。


## §R5 判读（蓝区，2026-10-29）

### 总结论

**阴性，但排查空间大幅收敛**。R5-A（HCCL_CONNECT_TIMEOUT=1800）与 R5-B（+关绑核）均复现 EI0015×7，超时方向与绑核争核方向**实测排除**。R5-0 取证推翻两个蓝区假说、纠正一个误判：

1. 作者 va.env **无 HCCL 调优变量** ⇒ "作者有 HCCL 环境调优"假说作废。
2. 蓝区 R5-0 前的解读修正：经比对 GitHub 原始仓，作者 scp-va HEAD cac7e2c4e **就是**分支 tip，"3 个未推送 commit"全部已包含在绿区代码里——**作者与绿区跑的是同一份框架代码**。真正的代码差异只剩作者 smoke 脚本的未提交本地改动（`min_tokens=max_tokens, ignore_eos=True` 强制满长 decode；simple_test_1p.py 参数调整）。
3. 参数差异清单（作者 → 绿区）：CHUNKED_PREFILL=1→默认关、MAX_NUM_SEQS=16→1、MAX_MODEL_LEN=40960→45056、du 端口 19090 段→20000 段、binding=none→（R5-B 已排除）。**没有一条能解释建链期 EI0015**。
4. 剩余最大未知：**作者的执行上下文**（容器内 vs 宿主机直接跑、CANN/torch_npu 版本是否与我们容器内一致）——R5-0 未覆盖，R6 补。

### 逐条裁决

- **R5-A/B 失败机制与 R4-3 同构：接受**。快速失败（1-2s）非超时，HCCL_CONNECT_TIMEOUT 方向关闭；R5-B 证明绑核不是根因。
- **R5-C 即败是蓝区派单缺陷**：ASCEND_LAUNCH_BLOCKING=1 与 ACL graph 模式互斥（config 校验期拒绝），派单时未考虑 graph 默认开。后续若需精确栈，须 RUN_MODE=eager + ASCEND_LAUNCH_BLOCKING=1。ascend/log tgz（3.4MB，296 文件）已留存，蓝区需要时再取。
- **R5-0 取证质量：接受**，但有漏项（见 R6 前置）：作者 scp-opt 的 git log 未回传、作者是否在容器内执行未确认、作者 smoke 未提交改动的 diff 原文未回传。

### 环境事实入库（后续轮次直接引用）

- 作者现场：`/home/w00919836/{scp-opt,scp-va,va.env,dp4tp2.json}`；va.env 无 HCCL 变量；scp-va 有未提交 smoke 改动；作者参数 CHUNKED_PREFILL=1/BS16/MAX_MODEL_LEN=40960/du 19090 段/binding=none。
- R5 日志资产：pp_r5_*.log、tmp/r5{a,b,c}-*/、r5a/r5b_first_error_ctx.txt、pp_r5c_ascend_log.tgz（含 EI0015 plog）。

## §R6 派单-旧：逐字复现作者跑通现场（蓝区，2026-10-29）——已交付已判读，见 §R6 判读

### 目标

用作者自己的目录、自己的环境、自己的配置**逐字复现**他的成功跑程。复现成功 → 拿到工作基线，逐项换回绿区参数做差分二分；复现失败 → 机器环境自作者跑通后发生漂移（驱动/固件/CANN/其他作业），直接改排查主线。

### 前置核验（R5-0 补漏，只读，原文回传）

```bash
cd /home/w00919836/scp-opt && git log --oneline -5   # 作者算子仓版本（R5-0 漏报）
tail -200 /home/w00919836/.bash_history              # 作者实际怎么跑的（是否 docker exec、实际命令行）
cd /home/w00919836/scp-va && git diff                # 作者 smoke 未提交改动原文
ls /usr/local/Ascend/                                 # 宿主机 CANN 版本布局
cat /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info 2>/dev/null || true
docker ps --format '{{.Names}} {{.Image}}'            # 当前容器清单（判断作者是否可能用了别的容器）
npu-smi info -t board -i 0 | grep -iE "version|firmware"   # R5 前置第5条漏报项
```

### 执行命令

**R6-0 · 清场**（同 R4-0，工具会自己起 Pool）

```bash
ps aux | grep -E 'urma_pool_server|urma_mock_pool' | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps aux | grep -E 'python|vllm' | grep -v grep | awk '{print $2}' | xargs -r kill -9
sleep 5; rm -rf /dev/shm/offload-numa-*; df -h /dev/shm
```

**R6-1 · 逐字复现作者命令**——用作者的 scp-opt、作者的 va.env、作者的 dp4tp2.json、作者的输出目录名。工具默认 `--framework-repo=/home/w00919836/scp-va`，会自动用作者的框架仓（含未提交 smoke 改动）。在容器内执行；若 bash_history 显示作者是在宿主机直接跑的，**先仍在容器内复现并在交付注明**。

```bash
cd /home/w00919836/scp-opt
source /home/w00919836/va.env
python3 tools/profile_pool_inference.py --du-store-prefault read \
  --output tmp/prefault-read-r6 --sample-port 19090 --sample-worker 0 \
  --remote-config ../dp4tp2.json --no-perf --payload-layers 78 \
  --du-pipeline-mib 1 --store-bytes $((128*1024*1024*1024)) \
  --pool-start-timeout 600 --pool-count 2 --numa-node 0 --pool-b-numa-node 2 \
  --store-numa-nodes local \
  --pool-cpus '2-17,21-36,40-55,59-74,78-93' \
  --pool-b-cpus '192-207,211-226,230-245,249-264,268-283' \
  --shared-port-gather --du-devices udmac1d1e3 --du-eid-indices 0 \
  --pool-b-du-devices udmac0d1e3 --pool-b-du-eid-indices 0 \
  2>&1 | tee /home/fjj/pp_r6_author_repro.log
echo "RC=$?"   # 回报
```

注意：作者的 dp4tp2.json 用 du 19090-19097，本条命令**不改**（这正是复现的目的）；若 19090 段端口被占用工具会报错，原样回传即可。框架侧 import 的 vllm_ascend 是 fjj 仓的 editable 安装，与作者代码同为 cac7e2c4e，可接受；若报 import 错误，原样回传不自行修。

### 通过标准

- **R6-1 复现成功**：工具 RC=0；tmp/prefault-read-r6/summary.txt 含 RUN_OK；raw/model.log 有生成文本；日志序列到 DSA_SHARED_DRAIN_OK。
- **复现失败**：按失败签名分类回传（EI0015 / 端口占用 / import 错 / 其它），首个报错点前后 50 行。

### 判读预案（派单时写死）

- **复现成功** ⇒ 拿到工作基线。下一轮逐项差分（绿区参数→作者参数，一次一项）：BS/MAX_MODEL_LEN/CHUNKED_PREFILL → du 端口段 → smoke 本地改动。
- **复现失败且同为 EI0015** ⇒ 机器环境自作者跑通后已漂移；下一轮对比作者成功时的驱动/固件/CANN 与当前值，并查期间机器变更记录（重启、其他作业、驱动升级）。
- **复现失败且为环境性错误（端口/import/路径）** ⇒ 原样回传，蓝区修命令，不算阴性。

### 回传格式

追加「§R6 交付」节：前置补漏各项原文 + R6-1 RC + 通过标准逐条 + 关键日志行 + 路径（pp_r6_author_repro.log、tmp/prefault-read-r6/）；附执行起止时间。

## §R6 交付（绿区，2026-10-29 23:05）

### 结论

**复现失败，且失败签名与 R4-3/R5-A/R5-B 完全相同（EI0015×7）** ⇒ 按派单判读预案命中「**复现失败且同为 EI0015 ⇒ 机器环境自作者跑通后已漂移**」。

决定性对照：作者 10-28 自己跑通的 `tmp/prefault-read-full01`（`summary.txt = RUN_OK model_exit=0`，8 worker 全部 `DSA_SHARED_DRAIN_OK`，remote_shared.py:568）与绿区 10-29 逐字复现用的是**同一命令、同一配置、同一份代码**（scp-opt f63a5bfe = scp-va cac7e2c4e，作者自己的目录/va.env/dp4tp2.json/输出目录名），唯一差别是执行日期与机器状态。作者跑通后机器环境发生了漂移，导致 EP8 跨 DP MoE all2all 建链层 EI0015 复现。

**首跑还叠加一个环境性前置失败（端口占用），已按派单「原样回传即可，不算阴性」处理，并在端口自然释放后完成了真实复现**，详见下。

---

### 前置核验 7 项（R5-0 补漏，只读，原文回传）

1. **scp-opt git log（作者算子仓版本，R5-0 漏报项）**：
   ```
   f63a5bfe Update and rename remote_offload_results.csv to overlap-19090.csv
   1a0f2528 feat(pool): add opt-in cross-port Gather worker sharing
   fe974e4b revert(copy-sfa): restore first-benefit ScatterSFA kernels
   6e85590d feat(tools): report per-epoch layer cross-port latency dispersion
   ea377615 feat(tools): split Gather and DU reports and exclude initial loads
   ```
   scp-opt HEAD `f63a5bfe` = 绿区 a5-remote-offload-ops `f63a5bf`（同一 commit f63a5bfedf6eaa0e1e06d51ca48e26558a5f3d70，已核对）。

2. **.bash_history**：`无 .bash_history`。作者实际执行方式不可考（无法确认容器内 vs 宿主机直接跑）。按派单要求**仍在容器 dsa-full 内复现**并在此注明。

3. **scp-va git diff（作者 smoke 未提交改动原文，R5-0 漏报项）**：
   - `examples/dsa_demo/simple_test.py:467`：`min_tokens=config.max_tokens, ignore_eos=True`（原为 `min_tokens=0, ignore_eos=False`）——强制满长 decode。
   - `examples/dsa_demo/simple_test_1p.py`：`profile_dir` 改 `./40k_1p`，追加 `enable_profile=True / enable_mtp=True / enable_mtp_graph=True / run_mode='graph' / enable_offload=True / enable_a5_packed_c8=True / gpu_memory_utilization=0.97`。

4. **宿主机 /usr/local/Ascend 布局**：仅 `driver / firmware / host_servers_remove.sh / host_servers_setup.sh / host_services_exit.sh / host_services_setup.sh / host_sys_init.sh / version.info`——**无 ascend-toolkit**（CANN 全在容器内）。`ascend-toolkit/latest/` 目录不存在。

5. **ascend_toolkit_install.info**：`install.info 不存在`。

6. **docker ps 容器清单**（R5 前置第5条漏报项）：
   ```
   vllm-wzh                         38b801e64b88
   deepseek-v41-rc2-a5-20261029-175627  quay.io/ascend/vllm-ascend:v0.30.0rc2-a5
   deepseek-v41-950dt-2              quay.io/ascend/vllm-ascend:v0.30.0rc2-a5-openeuler
   deepseek-v41-950dt                quay.io/ascend/vllm-ascend:v0.30.0rc2-a5-openeuler
   dsa-full                          quay.io/ascend/vllm-ascend:v0.23.0-a5-openeuler
   a5-nds-parameter                  quay.io/ascend/vllm-ascend:v0.30.0rc1-a5-openeuler
   ```
   R6 复现用 `dsa-full`（v0.23.0-a5-openeuler）。其余容器多为 v0.30.0rc 系（deepseek-v41-*），若作者在别的容器跑过，容器版本可能不同——但 R6 按派单在 dsa-full 内复现。

7. **npu-smi board 版本**（R5 前置第5条漏报项）：
   ```
   Chip Version      : V120
   Software Version  : 25.6.rc2
   Firmware Version  : 7.0.9.100.203
   ```

---

### R6-0 清场

同 R4-0 执行（kill urma_pool/profile_pool 相关 + rm /dev/shm/offload-numa-*），shm 384G 总量、清场后 373G 可用，无残留进程。

---

### R6-1 逐字复现作者命令

**执行方式**：在容器 dsa-full 内，`cd /home/w00919836/scp-opt && source /home/w00919836/va.env` 后逐字执行派单命令（工具默认 `--framework-repo=/home/w00919836/scp-va`）。命令原文与派单一字不差。

**结果分两段**：

#### R6-1a · 首跑（22:44:xx 启动）——端口占用（环境性错误，不算阴性）

- `RuntimeError: ports already in use: [19090]; existing processes left untouched`
- `summary.txt = RUN_FAILED`；`run-status.json errors = ["RuntimeError: ports already in use: [19090]; existing processes left untouched"]`
- 占用者：宿主机 pid 2481305，pts/16 交互终端 root bash 启动（父链 2328474←2328473），22:43:09 启动，
  `test/urma_mock_pool/urma_mock_pool --write-bench --device udmac1d1e3 --eid-index 0 --control-port 19090 --jetty-count 1 --topk-probe-memory normal --hold-seconds 999`
  ——用的是作者 EID 表（udmac1d1e3）+ 作者端口段（19090）+ 长驻 hold 999s，疑似作者/他人独立 write-bench 测试，**未杀**（可能影响他人作业）。
- 处置：按派单「19090 段端口被占用工具会报错，原样回传即可」+「环境性错误原样回传，不算阴性」。占用进程 999s 自然结束（22:59:56 释放，已用 `kill -0` 轮询确认 PID 消失 + `ss -tlnp` 确认 19090 无 LISTEN），随后**重跑真实复现**。
- 首跑失败证据已留存：`/home/fjj/r6_port_conflict_evidence/{summary,run-status,precheck}.txt`。

#### R6-1b · 重跑（23:00 端口释放后）——EI0015×7，复现失败

- **RC**：工具退出 `RuntimeError: model exited 1; see raw/model.log`，`summary.txt = RUN_FAILED`（`run-status.json errors = ["RuntimeError: model exited 1; see raw/model.log"]`）。
- **通过标准逐条**：
  - 工具 RC=0？ **否**（模型退出码 1）。
  - `tmp/prefault-read-r6/summary.txt` 含 RUN_OK？ **否**（`RUN_FAILED`）。
  - raw/model.log 有生成文本？ **否**（未到 decode）。
  - 日志序列到 DSA_SHARED_DRAIN_OK？ **否**（DSA_SHARED_CONNECT/CONSUMED/DRAIN 均未触达；连 CONNECT rows 都未到）。
- **失败签名**：EI0015×7（7/7 WorkerProc failed），`offline DP rank failed: dsa-dp-1=1, dsa-dp-2=1, dsa-dp-3=1`。
- **失败时序**（raw/model.log）：
  - 23:01:23 8 worker 全部 `DSA_EXECUTION_PLAN`（model_runner_v1.py:419，权重装载 + 执行计划下发成功，与 R4-3/R5 完全同构）。
  - 23:01:25.5~25.7 首个 EI0015（`Communication_Error_Ranktable_Detect(EI0015): Failed to collect cluster information of the communicator based on rootInfo detection. Reason: RootInfoDetect failed.`）。
  - 23:01:26 WorkerProc failed（`WorkerProc failed to start`，DP0_TP1_EP1 / DP2_TP1_EP5 / DP3_TP0_EP6 / DP3_TP1_EP7 / DP1_TP0_EP2 / DP2_TP0_EP4 / DP1_TP1_EP3）。
  - 崩溃栈与 R4-3/R5-A/R5-B 完全一致：`DeepseekV2MoE.__init__ → FusedMoE.__init__ → _get_token_dispatcher() → token_dispatcher.py:472 → hcclCommInitRootInfoConfig(numRanks, &rootInfo, rank, config, &comm) error code is 4`（moE_comm_method.py:250 → token_dispatcher.py:472 → HCCLUtils.cpp:140）。
- **关键日志行**：
  ```
  1205:(Worker_DP0_TP1_EP1 pid=2501402) ERROR ... EI0015: ... RootInfoDetect failed.
  1657:(Worker_DP3_TP1_EP7 pid=2501391) ... RuntimeError: create_config:../torch_npu/csrc/distributed/HCCLUtils.cpp:140 HCCL function error: hcclCommInitRootInfoConfig(...), error code is 4
  1659:(Worker_DP3_TP1_EP7 pid=2501391) ERROR ... EI0015: ... RootInfoDetect failed.
  ```
- **执行起止**：23:00:xx 启动（输出目录 tmp/prefault-read-r6 创建 23:00），23:01:38 model.log 定格（23:01:38.8 pp_r6_author_repro.log 定格）。

#### 作者成功跑程对照（关键证据）

- 作者 `tmp/prefault-read-full01`（10-28 04:07）`summary.txt = RUN_OK model_exit=0; owned Pool processes stopped`。
- 同配置核对（commands.json）：du_ports=[19090-19097]、store_port 19080/19084、store_bytes=137438953472（128GiB）、du_devices=udmac1d1e3/udmac0d1e3、du_eid_indices=0、gather_scheduler=shared、du_pipeline_mib=1.0、store_numa_nodes=local——**与 R6-1b 逐字段一致**。
- 作者 model.log 8 worker 全部 `DSA_SHARED_DRAIN_OK host=dsa-host ... real_store=1`（remote_shared.py:568，04:07:13）——完整 remote-offload store 周期跑通。
- 作者跑程结束同样 force-kill 残留进程（04:07:18 shutdown），与首跑端口占用后 22:59:56 释放一致（作者 10-28 跑完即释放 19090 段）。

---

### 判读落点

- **复现失败且同为 EI0015** ⇒ 机器环境自作者跑通后已漂移（命中派单预案第二分支）。
- 漂移面候选（按派单预案，下一轮需对比作者成功时驱动/固件/CANN 当前值 + 查期间机器变更记录）：驱动 25.6.rc2 / 固件 7.0.9.100.203 / Chip V120 为**当前值**；作者成功跑时（10-28 04:07）的值未留存，需蓝区或后续取证。
- 期间机器变更线索（本机观察）：10-29 有 `deepseek-v41-rc2-a5-20261029-175627`（v0.30.0rc2，17:56 创建的容器）、`deepseek-v41-950dt-2`（v0.30.0rc2-a5-openeuler）等新容器出现——**有他人作业在同一批 NPU/环境上跑**，可能是漂移来源（如固件/驱动被其他作业升级、或 NPU 被其他容器占用过）。R6 复现时 8 卡均空闲（npu-smi 0% util、HBM ~4.7G/卡空闲常驻）。
- **不构成阴性**：首跑端口占用为环境性错误（派单明确不算阴性）；重跑 EI0015 与 R4-3/R5 同构，进一步坐实「EP8 跨 DP MoE all2all 建链层」故障面，且是**作者现场原样复现失败**——排除了「绿区参数/目录/环境差异」这一方向的解释。

### 环境事实入库（后续轮次直接引用）

- 作者现场成功基线：`tmp/prefault-read-full01`（10-28，RUN_OK，DSA_SHARED_DRAIN_OK×8）。
- R6 日志资产：`/home/fjj/pp_r6_author_repro.log`、`/home/fjj/r6_port_conflict_evidence/`、`/home/w00919836/scp-opt/tmp/prefault-read-r6/`（summary/run-status/raw/model.log）。
- 首跑端口占用进程详情：pid 2481305（已自然退出），write-bench hold-999。
- 容器内复现（无法确认作者是否容器内跑，已按派单注明）。

## §R4 判读（蓝区，2026-10-29）

### 总结论

**阴性（新卡点 EI0015），但拿到决定性增量：R3-A 附带回传证明 offload 关闭的 EP8 基线同样崩溃**——故障不是 remote-offload 引入的，收敛到「本分支 × EP8 跨 DP HCCL 建链」这一层。同时作者新流程的基础设施全链路实测就绪：f63a5bfed 重编、双 Pool（shared-gather、2×128GiB Store、384G shm）、store 路由、权重装载、DSA_EXECUTION_PLAN 全部通过。

### 逐条裁决

- **R4-0/R4-1/R4-2 通过**。ops tip f63a5bf 与锚点一致；wrapper/binary 新能力（gather-scheduler、du-gather-shared、DU_AND_GATHER）实测齐全；NUMA 拓扑（node0: CPU 0-95 / node2: 192-287）与工具默认 CPU 范围匹配，作者默认绑核值在绿区成立。
- **R4-3 未通过**。崩溃点：load_model → MoE `setup_moe_comm_method` → `hcclCommInitRootInfoConfig` error code 4 → EI0015 RootInfoDetect failed；7/8 worker 中招（DP0_TP0_EP0 即 root 未报错），DSA_EXECUTION_PLAN（05:23:08）后 2 秒即崩。
- **绿区"R4-3b 严格按字面不触发"：接受**——EI0015 确实不在 EI0006 字面上；但 EI0015 官方 solution 第 4 条与 R4-3b 同族，R5-A 直接补上这一试。
- **R3-A 附带数据是本轮回合最大增量**：offload=0 也崩（HCCL dispatch timeout EmptyQueue，profile_run MoE 前向）→ 两个不同报错（EI0006 建链超时 / dispatch EmptyQueue / EI0015 RootInfoDetect）同族不同点，崩溃阶段还在漂移（R2 在 profile_run、R3-A 在 dummy_run、R4-3 提前到 load_model）——**阶段漂移 + 快速失败 = 典型 rank 间时序竞争**，不是单一慢节点超时。
- **最可疑机制（估计，待 R5 逐项排除）**：R4 的 Pool 配置比 R2/R3 重得多（du-workers 7×8=56 个轮询线程 + shared gather，钉在 CPU 2-93/192-283），与框架 enable_cpu_binding 的 worker 绑核叠加，worker 建链期 CPU 饥饿 → rank 到不齐 → RootInfoDetect 快速失败。作者机同配置能跑通，差异可能在作者 va.env 里的 HCCL 调优（内容未知，已向用户索取）。
- **指标开发附记**：metrics 汇总代码（e6fd53bd3）已随本轮 pull 进入绿区框架仓，R5 各跑程会自动带出 `[dsa-smoke][metrics]` 行（跑崩则无输出，属预期）。

### 环境事实入库（后续轮次直接引用）

- /dev/shm 已 remount 384G；双 Pool 由工具自生自灭，无手工 Pool 存活。
- 容器内 github 代理：90.255.12.179:3128 持续 407，**改用 90.255.129.35:3128**（R4-1 实测 curl 200）。后续 git 代理一律用后者。
- R4 日志资产：/home/fjj/pp_r4_run.log、pp_r4_ops_build.log、tmp/r4-author-bs1/{summary,precheck}.txt、raw/{model,pool-a,pool-b}.log。

## §R5 派单-旧：EI0015 重试梯子（蓝区，2026-10-29）——已交付已判读，见 §R5 判读

### 目标

用三级条件梯子越过/定位 EP8 建链失败：A 放宽 HCCL 建链超时 → B 再关 CPU 绑核（排除与 Pool 轮询线程争核）→ C 抓 HCCL 精确栈回传。任一级通过即阳性落袋，不再往下走。

### 前置核验

**R5-0 · 作者现场取证（只读，最先做，可能直接读出答案）**：作者（w00919836）就是在本机跑通的（用户 10-29 确认），他的环境文件和成功日志应仍在盘上。逐项执行，原文回传；某条不存在就如实写"不存在"，不要去找替代品。

```bash
ls /home/w00919836/ 2>&1
cat /home/w00919836/va.env 2>&1                       # 作者跑前 source 的环境，头号嫌疑
cat /home/w00919836/dp4tp2.json 2>&1                  # 作者的 remote-config 原文
cd /home/w00919836/scp-opt && git log --oneline -5 2>&1   # 作者算子仓版本，比对 f63a5bfed
cd /home/w00919836/scp-va && git log --oneline -3 && git status -sb | head -20 2>&1   # 作者框架仓版本+未提交改动
ls -t /home/w00919836/scp-opt/tmp/ | head 2>&1        # 作者成功的 run 目录
cat /home/w00919836/scp-opt/tmp/prefault-read-full01/summary.txt 2>&1
tail -50 /home/w00919836/scp-opt/tmp/prefault-read-full01/raw/model.log 2>&1   # 作者成功跑程的日志尾巴
cat /home/w00919836/scp-opt/tmp/eid-probe-run01/remote-shared.json 2>&1        # 作者 EID 表，与绿区 tmp/eid-probe-r1 版比对
```

若 va.env 存在：把其中所有 `export` 行原文回传（含 HCCL_/ASCEND_/VLLM_ 开头的一切）。蓝区拿到后会出 va.env 差分清单。**取证结果回传前可以先做 R5-A，但如果 va.env 里出现我们没设的 HCCL 变量，直接在当次跑程补上并重跑该级**（在交付里注明补了哪些）。

### 其余前置核验

1. `ps aux | grep -E 'urma|python|vllm' | grep -v grep`：应为空（R4 工具已自行收 Pool）；不空则杀 python/vllm，urma 进程也杀（本轮工具会自己重起）。
2. `df -h /dev/shm`：应为 384G；不是则 `mount -o remount,size=384G /dev/shm`。
3. `rm -rf /dev/shm/offload-numa-*`（清 R4 残留 backing）。
4. `cd /home/fjj/vllm-ascend-remote-offload && git -c http.proxy=http://90.255.129.35:3128 pull --ff-only && git log --oneline -2`：顶部应为 `e6fd53bd3`（metrics 汇总）。回报这两行。
5. HDK（驱动/固件）版本采集（只读，原文回报；HCCL 与 URMA 都在这一层，需与作者机比对）：`npu-smi info -t board -i 0 | grep -iE "version|firmware"` 和 `npu-smi info | head -5`。

### 执行命令

公共前缀（每级都一样，先 export 再跑；注意代理已换）：

```bash
cd /home/fjj/a5-remote-offload-ops
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731
export VLLM_ASCEND_SMOKE_DATASET=chinese_40k
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=1
export VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=45056
export VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=45056
export VLLM_ASCEND_SMOKE_MAX_TOKENS=64
export VLLM_ASCEND_SMOKE_ENABLE_PROFILE=0
```

运行命令模板（每级仅 `<OUT>` 和额外 export 不同；各级之间必须 `ps aux | grep -E 'urma|python|vllm' | grep -v grep | awk '{print $2}' | xargs -r kill -9; sleep 5; rm -rf /dev/shm/offload-numa-*`）：

```bash
python3 tools/profile_pool_inference.py \
  --no-perf \
  --framework-repo /home/fjj/vllm-ascend-remote-offload \
  --remote-config /home/fjj/a5-remote-offload-ops/tmp/eid-probe-r1/remote-shared.json \
  --output tmp/<OUT> \
  --payload-layers 78 \
  --store-bytes $((128*1024*1024*1024)) \
  --du-ports 20000,20001,20002,20003 --store-ports 19080,19081,19082,19083 \
  --pool-b-du-ports 20004,20005,20006,20007 --pool-b-store-ports 19084,19085,19086,19087 \
  --du-devices udmac1d1e2 --pool-b-du-devices udmac0d1e3 \
  --du-eid-indices 0 --pool-b-du-eid-indices 0 \
  --sample-port 20000 --sample-worker 0 \
  --numa-node 0 --pool-b-numa-node 2 --store-numa-nodes local \
  --pool-start-timeout 600 \
  --shared-port-gather --du-pipeline-mib 1 \
  2>&1 | tee /home/fjj/pp_r5_<OUT>.log
echo "RC=$?"   # 回报
```

**R5-A**：`export HCCL_CONNECT_TIMEOUT=1800`，`<OUT>`=r5a-hccl1800。通过则停，不回传后续级别。

**R5-B**（仅 R5-A 失败）：在 R5-A 基础上追加 `export VLLM_ASCEND_SMOKE_ENABLE_CPU_BINDING=0`，`<OUT>`=r5b-no-cpubind。通过则停。

**R5-C**（仅 R5-B 失败）：在 R5-B 基础上追加 `export HCCL_ENTRY_LOG_ENABLE=1` 和 `export ASCEND_LAUNCH_BLOCKING=1`，`<OUT>`=r5c-hccl-trace。跑完无论成败：

```bash
tar czf /home/fjj/pp_r5c_ascend_log.tgz -C ~ ascend/log 2>/dev/null; ls -l /home/fjj/pp_r5c_ascend_log.tgz
```

### 通过标准（A/B 通用，需同时满足）

工具 RC=0；`tmp/<OUT>/summary.txt` 含 `RUN_OK`；`raw/model.log` 有生成文本；日志依次 `DSA_EXECUTION_PLAN` → `DSA_STORE_PUT_DONE` → `DSA_SHARED_CONNECT_BEGIN` → `DSA_SHARED_CONNECT_OK ... backend=real_store` → ≥2 轮 `DSA_SHARED_CONSUMED` → `DSA_SHARED_DRAIN_OK`；`run-status.json` graceful=true 且 errors=[]。C 级不要求通过，栈和日志落盘即可。

### 回传格式

追加「§R5 交付」节：各级 RC + 通过标准逐条 + 首个报错点前后 50 行 + 日志路径（pp_r5_*.log、tmp/r5*/raw/model.log）；R5-C 附 pp_r5c_ascend_log.tgz 路径；附框架仓 `git log --oneline -2` 与执行起止时间。

### 判读预案（派单时写死）

- **R5-A 过** ⇒ 阳性落袋；HCCL_CONNECT_TIMEOUT=1800 入库标配；下一轮 BS16 放大 + 性能口径。
- **R5-A 败、R5-B 过** ⇒ 阳性落袋；根因=Pool 轮询线程与框架绑核争核，CPU 绑核策略需按双 Pool 布局重做（蓝区出方案）；下一轮同上。
- **A/B 均败** ⇒ 阴性；R5-C 栈 + ascend/log 回传，蓝区对比作者 va.env（索取中）逐参数定位。
- **出现 LIM `1..4 query rows` / 零 miss 卡住** ⇒ 已知边界，日志回传即完成义务。


## §R5 交付（绿区，2026-10-29 21:00）

### 各级结果：A/B 均败（阴性），R5-C 栈+日志已落盘回传

| 级 | 变体 | 结果 | 失败签名 | 关键时间 |
|---|---|---|---|---|
| R5-A | HCCL_CONNECT_TIMEOUT=1800 | **FAIL** | EI0015×7（与 R4-3 完全相同） | 19:52:49 启动 → 19:53:49 EI0015 |
| R5-B | +VLLM_ASCEND_SMOKE_ENABLE_CPU_BINDING=0 | **FAIL** | EI0015×7（关绑核无效） | 20:08:35 启动 → 20:09:23 EI0015 |
| R5-C | +HCCL_ENTRY_LOG_ENABLE=1 +ASCEND_LAUNCH_BLOCKING=1 | **N/A** | 4×ValidationError：`ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1`，config 校验期即拒，未到建链 | 20:12:56 启动即败 |

### 通过标准逐条核验（A/B 通用）

- 工具 RC=0：**不满足**（A/B 均 `RuntimeError: model exited 1`，summary.txt=`RUN_FAILED`）
- `tmp/<OUT>/summary.txt` 含 `RUN_OK`：**不满足**（三级别均 `RUN_FAILED`）
- `raw/model.log` 有生成文本：**不满足**（无生成阶段）
- 日志序列 `DSA_EXECUTION_PLAN`→`DSA_STORE_PUT_DONE`→`DSA_SHARED_CONNECT_BEGIN`→`DSA_SHARED_CONNECT_OK`→`≥2 DSA_SHARED_CONSUMED`→`DSA_SHARED_DRAIN_OK`：**不满足**（remote-shared 推理路径仍未触达，与 R2/R4 一致）
- `run-status.json` graceful=true 且 errors=[]：**不存在**（模型装载阶段即崩）

### 关键证据

**R5-A/B 失败机制（与 R4-3 完全同构，确认复现而非新卡点）**：
- 失败点：`WorkerProc failed to start` → `DeepseekV2MoE.__init__` → `FusedMoE.__init__` → `setup_moe_comm_method` → `AlltoAllCommImpl` → `TokenDispatcherWithAll2AllV.__init__` → `token_dispatcher.py:472 backend.get_hccl_comm_name(local_rank)` → `hcclCommInitRootInfoConfig(...) error code is 4` → **EI0015 Communication_Error_Ranktable_Detect / RootInfoDetect failed**
- 影响 worker：R5-A 7 worker（EP1/3/4/5/6/7，DP0-3 跨 4 EngineCore）、R5-B 7 worker（同族）
- **快速失败非超时**：EI0015 出现在模型装载完成（DSA 执行计划已下发，见下）后 1-2 秒内，HCCL_CONNECT_TIMEOUT=1800 未改变时序 ⇒ solution#4 超时方向不适用
- **模型装载已成功**：R5-A 崩溃前 worker 已打印 `model_runner_v1.py:419` 全部 remote_shared 执行计划（`50 14 50 hbm pool lis_send_full durable_with_causal_tail` / `51-53 remote_pool pool remote_shared source_plan scatter_sfa` 等逐层 plan），EP8 权重装载完成，仅 MoE 通信组初始化（all2all comm）失败

**R5-C 失败机制（派单变量自身兼容性问题，非 EI0015 缓解/证据）**：
- `ASCEND_LAUNCH_BLOCKING=1` 与 ACL graph 模式不兼容，vLLM 在 `VllmConfig` pydantic 校验期即拒绝：`Value error, ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1. Please unset ASCEND_LAUNCH_BLOCKING or set it to 0`
- 4 个 DP rank（dsa-dp-0/1/2/3）全部在 `create_engine_config` 阶段失败，**未到模型装载/建链**，故无 HCCL 栈可抓（EI0015 计数=0）
- 派单预案「C 级不要求通过，栈和日志落盘即可」：**已满足**——ascend/log 全量打包含历史 R5-A/B 的 EI0015 plog

### 日志路径

- R5-A：`/home/fjj/pp_r5_r5a-hccl1800.log`；`/home/fjj/a5-remote-offload-ops/tmp/r5a-hccl1800/raw/model.log`；首个报错点前后 50 行存档 `/home/fjj/r5a_first_error_ctx.txt`
- R5-B：`/home/fjj/pp_r5_r5b-nocpubind.log`；`/home/fjj/a5-remote-offload-ops/tmp/r5b-no-cpubind/raw/model.log`；首个报错点前后 50 行存档 `/home/fjj/r5b_first_error_ctx.txt`
- R5-C：`/home/fjj/pp_r5_r5c-hccl-trace.log`；`/home/fjj/a5-remote-offload-ops/tmp/r5c-hccl-trace/raw/model.log`；**HCCL 栈包 `/home/fjj/pp_r5c_ascend_log.tgz`（3429428 字节，296 文件）**
- tgz 内容：`ascend/log/` 全量 = `vllm_ascend/` 框架日志 + `run/device-*/` 设备日志 + `run/plog/` + `debug/plog/`（159 个 plog，含 R5-A plog-2274708、R5-B plog-2320216 等 EI0015 时点 plog）+ `debug/device-*`

### 框架仓版本与执行时间

```
$ git log --oneline -2
3670a39 docs(pingpong): R5-0 作者现场取证（同机 w00919836 账号，va.env/EID/成功日志）
074ebe9 docs(pingpong): R5 前置核验补 HDK 驱动/固件版本采集
```

- 执行起止：R5-A 19:52:49~19:53:49；R5-B 20:08:35~20:09:23；R5-C 20:12:56（即败）+ 20:13 tgz 打包完成

### 供蓝区参考的差异点（R5-0 取证已确认）

1. 作者 va.env **无任何 HCCL 调优变量**（推翻「作者有 HCCL 超时调优」猜测）——但作者跑程成功，绿区 A/B 相同命令失败，差异不在 HCCL 变量
2. 作者框架仓 scp-va HEAD=cac7e2c4 **比绿区 3670a39 新 3 commit**（未推送：`perf(dsa): dispatch shared group IO once from Full`、`fix(dsa): 远端直通仅保留本地 KV 计划元数据` 等）；未提交改动：simple_test.py `min_tokens=max_tokens, ignore_eos=True`；simple_test_1p.py 参数调整
3. 作者 EID 表与绿区差异：du_endpoint 19090-19097（绿区 20000-20007）；pool_eid 作者 0f7f03.../0f3f03...（绿区 0f7f02...）
4. 作者跑程 binding=none（未绑核）；绿区 R5-B 关绑核后仍 EI0015 ⇒ **绑核不是根因**
5. 作者 smoke 参数差异：CHUNKED_PREFILL=1、DP4/TP2、MAX_NUM_SEQS=16、MAX_MODEL_LEN=40960（绿区 R5 用 CHUNKED 默认、MAX_NUM_SEQS=1、MAX_MODEL_LEN=45056）

### 判读预案命中

- 「A/B 均败 ⇒ 阴性；R5-C 栈 + ascend/log 回传，蓝区对比作者 va.env 逐参数定位」→ **命中**，回传材料齐备。
- 未出现「LIM `1..4 query rows` / 零 miss 卡住」边界，该预案不适用。
## §R2 判读（蓝区，2026-10-27）

### 总结论

**阴性（新卡点 EI0006），但拿到三个关键增量**：

1. **EP8 权重装载全部成功**（每卡 61.3GB，分片正确）——蓝区 R1 判读中"DP4/TP2→EP8 可装下 391GB 模型"的反推**被实测证实**；绿区 R1 交付中"F（DP4/TP2）必然 OOM"的结论正式作废。
2. 修掉两个真实缺陷（store_endpoint、sibling CPU，见下），链路配置可信度上升。
3. 卡点收敛到唯一一处：profile_run 首次 EP8 HcclAllGather 建链超时。remote-shared 推理路径（CONNECT/CONSUMED/DRAIN）本轮仍未触达。

### 逐条裁决

- **通过标准 1（dry-run RC=0）：命中**。
- **标准 2-5：未命中**——崩溃在 profile_run dummy run，早于一切 offload 活动；Pool 侧只有 OPEN 无 CREATE/PUT，与框架侧证据互洽。
- **EI0006 归属判定：框架/配置层，非机器拓扑**。证据链：
  1. 绿区裸 8 卡 HCCL 探针 8/8 PASS（实测，hccl_probe_8npu.sh）；
  2. 本机本地卸载分支（vllm-ascend-dsa-v0.23.0-custom）此前 DP4/TP2/EP8 整网曾跑通到 decode（Worker_DP3_TP1_EP7 日志，实测历史）——EP8 跨 4 个 EngineCore 建链在本机可行；
  3. 8 worker 的 AIV_URMA_LOCAL_EID 各绑一张物理卡（phy 0-7 全不同），设备映射正确。
  最可疑机制（**估计**，待 R3 验证）：4 个 EngineCore 进程树各自独立推进到 profile_run，remote-offload 分支 worker 初始化多出 URMA/EID/Pool 客户端步骤，进程间偏斜超过 HCCL 建链超时（默认 120s 量级）→ 全体同时超时（8 worker 报错窗口仅 38ms 与此吻合）。
- **per-device store_endpoint 修复：接受**。与仓内 remote_store_lifecycle_validation.md 2026-10-20 节一致；修复后 Pool 日志 8 个独立 store id 的 `REMOTE_DU_SOURCE` 实测验证。后续轮次沿用修复后的 remote-shared.json。蓝区备忘：probe 工具应直接产出 per-device store_endpoint，后续评估补工具补丁。
- **sibling CPU 6→5：接受**。R1 派单命令 `--du-sibling-cpus 6` 是蓝区缺陷（本机 CPU4 的 thread_siblings 是 4-5）。后续 Pool 命令一律用 5（已写入 R3-0）。
- **worker.py 版本门控（0.23.0 跳过 DP local_rank 偏移）：绿区结论接受**——EID 实测证明 8 卡分配正确，该块是否生效不影响本轮结论。

### 环境事实入库（后续轮次直接引用）

- Pool 当前实例：backing=/dev/shm/kv-remote-r2b，`--du-sibling-cpus 5`，8 DU ready，复用不重启。
- remote-shared.json 当前版含 per-device store_endpoint（19080+N 与 du 20000+N 配对）；修复前备份 .bak-r2/.bak-r2b。
- HCCL 探针 `/home/fjj/hccl_probe_8npu.sh` 留存，可作回归基线复用。

## §R2 派单-旧：DP4/TP2（EP8）整网首发（蓝区，2026-10-27）

### 目标

复用 R1 已验证的安装/构建/Pool/EID 资产，直接以 DP4/TP2（EP8，八卡）eager BS1 跑 `hbm_full_remote_shared` 真实推理，拿到 remote-shared 链路生效证据。本轮仍只判通/不通，不做性能结论。

### 前置核验

1. Pool 存活：`ps -p 2167590 -o pid,cmd`。活着 → 跳过 R2-0；死了（或机器重启过）→ 执行 R2-0 重启 Pool。
2. `npu-smi info | head -30`：8 卡空闲、无残留进程。
3. 清场只杀框架，**不杀 Pool**：`ps aux | grep -E 'python|vllm' | grep -v grep | awk '{print $2}' | xargs -r kill -9; sleep 5`

### 执行命令

**R2-0 ·（仅 Pool 死亡时）重启 Pool** —— backing 必须换全新路径 kv-remote-r2；分组沿用 R1 确认的 udmac1d1e2（卡 0-3）/ udmac0d1e3（卡 4-7）：

```bash
cd /home/fjj/a5-remote-offload-ops
nohup tmp/mempool-build/urma_pool_server \
  --store-mode embedded \
  --backing /dev/shm/kv-remote-r2 --store-bytes 4294967296 \
  --payload-layers 78 --requests 8 --source-capacity 25600 \
  --store-ports 19080,19081,19082,19083,19084,19085,19086,19087 \
  --du-ports 20000,20001,20002,20003,20004,20005,20006,20007 \
  --du-devices udmac1d1e2,udmac1d1e2,udmac1d1e2,udmac1d1e2,udmac0d1e3,udmac0d1e3,udmac0d1e3,udmac0d1e3 \
  --du-eid-indices 0,0,0,0,0,0,0,0 \
  --trace-lifecycle \
  --pool-binary $PWD/test/urma_mock_pool/urma_mock_pool -- \
  --jetty-count 1 --blocks 8 --poll-cpus 2 \
  --gather-threads 1 --gather-cpus 4 --du-sibling-cpus 6 \
  --du-workers 1 --du-concurrency 1 --pipeline-mib 0 \
  > /home/fjj/pp_r2_pool.log 2>&1 &
sleep 15
grep -c "URMA_JETTY_READY" /home/fjj/pp_r2_pool.log   # 回报这个数字
```

**R2-1 · dry-run + 实跑**（env 与 R1 E 相同，仅 TP=2、DP=4）

```bash
cd /home/fjj/vllm-ascend-remote-offload
export VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731
export VLLM_ASCEND_SMOKE_TENSOR_PARALLEL_SIZE=2
export VLLM_ASCEND_SMOKE_DATA_PARALLEL_SIZE=4
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=1
export VLLM_ASCEND_SMOKE_RUN_MODE=eager
export VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=1
export VLLM_ASCEND_SMOKE_ENABLE_MTP=1
export VLLM_ASCEND_SMOKE_ENABLE_MTP_GRAPH=0
export VLLM_ASCEND_SMOKE_ENABLE_PROFILE=0
export VLLM_ASCEND_SMOKE_MAX_TOKENS=16
export VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=25600
export VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=25600
export VLLM_ASCEND_SMOKE_DATASET=chinese_23k

python examples/dsa_demo/simple_test_1p.py \
  --remote-shared-config /home/fjj/a5-remote-offload-ops/tmp/eid-probe-r1/remote-shared.json --dry-run \
  2>&1 | tee /home/fjj/pp_r2_dryrun.log

python examples/dsa_demo/simple_test_1p.py \
  --remote-shared-config /home/fjj/a5-remote-offload-ops/tmp/eid-probe-r1/remote-shared.json \
  2>&1 | tee /home/fjj/pp_r2_run.log
```

### 通过标准（需同时满足）

1. dry-run 退出码 0；
2. 实跑进程退出码 0，输出里有生成文本；
3. 日志依次出现 `DSA_EXECUTION_PLAN` → `DSA_STORE_PUT_DONE` → `DSA_SHARED_CONNECT_BEGIN` → `DSA_SHARED_CONNECT_OK ... backend=real_store` → 至少两轮 `DSA_SHARED_CONSUMED` → 退出时 `DSA_SHARED_DRAIN_OK`（select/gather/scatter_sfa 计数非零）；
4. `DSA_SHARED_CONNECT_BEGIN` 的 `rows=` 与实际绑定请求数一致（DP4 应为 4）；
5. `DSA HBM CACHE CAPACITY REPORT` 出现并**原文记录整行**（R1 参考值 `indexer=21, resident=57, mtp=1`；拓扑变为 DP4/TP2 后允许数值偏差，由蓝区判，不算硬门禁）。

### 已知边界（遇到不算新发现，按判读预案处理）

- 保持 eager + BS1：LIM `1..4 query rows` 校验在本分支未修，**不要**开 graph（RUN_MODE 保持 eager，MTP_GRAPH=0）。
- 若复现"零 miss 第二轮全 hit 卡住"：保留 generation/status/Pool 三方日志原样回传，即算完成本轮义务。
- 不要改任何代码、不要改分组名、不要重启仍在运行的 Pool（除 R2-0 触发条件外）。

### 回传格式

在 pingpong.md 追加「§R2 交付」节，格式同 R1：逐条通过标准命中/未命中 + 日志原文行；失败给首个报错点前后 50 行 + pp_r2_*.log 路径；附框架仓 `git log --oneline -2` 与执行起止时间。

### 判读预案（派单时写死）

- **DP4/TP2 仍 OOM** ⇒ 蓝区量化模型修正；回传报错点每卡 allocated/total 原文，蓝区重算后决定换拓扑还是换模型。
- **出现 `DSA A5 MTP LIM requires 1..4 query rows per request`** ⇒ 已知 bug 变体命中，阴性；蓝区核对形状后出修复或配置绕行。
- **第二轮全 hit 卡住** ⇒ 命中作者未关闭的"零 miss"问题，阴性（非本轮回退）；三方日志回传即完成义务。
- **CONNECT rows≠4 或 backend≠real_store** ⇒ 阴性（绑定逻辑问题）；日志回传，蓝区定位。
- **全中** ⇒ 阳性落袋；下一轮：graph 模式（核对 capture 档后启用）+ 带宽/TPOT 性能口径（另写派单）。

## §R1 判读（蓝区，2026-10-27）

### 总结论

**链路部分阳性，整网首发未完成。** B/C/D/E1/E2 全部实测通过（框架安装、算子构建、8 卡 EID 探测、Pool 启动、dry-run）；E3 阻塞为模型权重规模与 TP1/DP1 拓扑的物理冲突，**不是链路/驱动/代码 bug**。注意：框架在权重装载阶段即 OOM，remote-shared 真实推理路径（CONNECT→CONSUMED→DRAIN）本轮**尚未触达**，"链路通"结论要 R2 拿到日志序列后才落袋。

### 逐条裁决

- **B 通过**。catlass 走本机拷贝（预案第一优先路径），双嵌套 gitlink 提升一级的处置存档备查。
- **C 通过**。算子仓 tip 6e85590 与锚点一致；Pool 二进制、mock_pool、torch extension .so 三类产物齐全。
- **D 通过，偏差接受**。UB_A/UB_B 是派单占位符未填真实设备名，**蓝区派单缺陷**；绿区按仓内 docs/mempool_validation.md 替换为 udmac1d1e2（卡 0-3）/ udmac0d1e3（卡 4-7）后 8/8 通过。分组**确认接受**：remote-shared.json 两组 pool_eid 与两 UB 平面一致，MEMPOOL_WRAPPER_ARGS 自洽。置信度：实测（probe 8/8 pass）。后续轮次沿用此分组，不再变更。
- **E1/E2 通过**。Pool 8 个 jetty 就绪，dry-run 配置打印完整。
- **E3 未通过，定性：规模性阻塞**。权重量化采信：MoE 377GB（78 层 × 256 专家 × 18.9MB）+ dense ≈14GB ≈ 磁盘 391G；TP1 单卡 96GB 物理放不下，TP2/DP1（EP=2）每卡仍需 ~188GB，同样必然 OOM——绿区两组实测与量化一致。
- **绿区"F（DP4/TP2）同样必然 OOM"的结论不采纳**。代码事实：`simple_test.py:40` enable_expert_parallel 默认 True、`:549` ep_size=tp×dp、`:234-235` DP>1 强制 EP。DP4/TP2 → **EP8**，每卡 MoE ≈ 377/8 ≈ 47GB + dense ≈ 7GB ≈ 54GB < 96GB，权重可装载。绿区 TP2 实测是 DP1（EP2），不代表 F 配置。置信度：反推（代码 + 量化算术），R2 实测验证。
- **TP8 硬拒确认**：`remote_shared_config.py:161` `tp not in (1, 2)` 是本分支硬约束，绿区 TP8 测试与代码一致，非环境问题。
- **F 未执行**：符合协议（E 未全中跳过 F）。

### 环境事实入库（后续轮次直接引用）

- 容器内 git 访问 github 需 `-c http.proxy=http://90.255.12.179:3128`。
- catlass 修复法：从 `/home/fjj/vllm-ascend-dsa-v0.23.0-custom/csrc/third_party` 拷贝，双嵌套 gitlink 提升一级。
- Pool 仍在运行（pid 2167590，backing=/dev/shm/kv-remote-r1）；E3 在权重装载期崩溃，未发生任何 PUT/CONNECT，Pool 无残留绑定，**R2 直接复用，不重启**。
- 本机无可替代的小模型（DeepSeek-V4.1-Flash 728-801GB 更大），换模型路线排除。

### 下一步

见上方 §R2 派单：直接 DP4/TP2（EP8）首发，跳过物理不可能的 TP1。

## §R1 交付

### 执行环境

- 机器 IP：90.96.17.111（hostname=localhost.localdomain，aarch64，16× Ascend 950DT 96GB/卡，鲲鹏 950，1505GB RAM）
- 容器：dsa-full（镜像 vllm-ascend:v0.23.0-a5-openeuler，host 网络/pid，挂 /home /etc /data /usr/local/Ascend/driver）
- 框架仓：/home/fjj/vllm-ascend-remote-offload（分支 vllm-ascend-v0.23.0-custom-remote-offload）
- 算子仓：/home/fjj/a5-remote-offload-ops（main）
- 模型：/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731（78 层，hidden 6144，n_routed_experts 256，moe_intermediate_size 2048，磁盘 391G）
- 执行时间：2026-10-27 06:2x–07:5x（机器本地时间）
- 日志：/home/fjj/pp_r1_b_install.log / pp_r1_c_build.log / pp_r1_d_host.log / pp_r1_d_probe.log / pp_r1_d_probe_auto.log / pp_r1_d_probe_grouped.log / pp_r1_e_pool.log / pp_r1_e_dryrun.log / pp_r1_e_run.log / pp_r1_e_run_tp2.log / pp_r1_e_run_tp8.log

### Stage 逐项

**Stage B 框架安装 — 通过**

- B1–B3：`pip install --no-build-isolation --no-deps -v -e .` 退出码 0（pp_r1_b_install.log:7409 `Successfully installed vllm_ascend-0.23.0`）
- B4：输出恰为 `/home/fjj/vllm-ascend-remote-offload/vllm_ascend/__init__.py`（命中）
- 备注：编译期第三方依赖 catlass / makeself / abseil-cpp / json / ascend_protobuf / pkg 因 gitcode.com 解析失败，从 /home/fjj/vllm-ascend-dsa-v0.23.0-custom/csrc/third_party 拷贝补齐后编译通过。catlass 为双嵌套 gitlink（内容在 catlass/catlass/），提升一级使 catlass/include 存在。

**Stage C 算子仓构建 — 通过**

- clone：`git clone --depth 1 https://github.com/JiJiaF1020/a5-remote-offload-ops.git`（容器内 git 需 `-c http.proxy=http://90.255.12.179:3128`）
- build：`BUILD_JOBS=32 SOC_VERSION=ascend950 ... bash build_and_install.sh` 退出码 0；pp_r1_c_build.log:2418 `[kvcache-scatter-copy] build and local install done`
- 产物存在：tmp/mempool-build/urma_pool_server、test/urma_mock_pool/urma_mock_pool、torch_extension/kvcache_scatter_copy_custom_ops/custom_ops_lib.cpython-312-aarch64-linux-gnu.so（命中）
- 算子仓 tip：`6e85590 feat(tools): report per-epoch layer cross-port latency dispersion`（与派单锚点 6e85590db 一致）

**Stage D EID 探测 — 通过（含派单命令修正）**

- `probe_urma_eids.py host`：退出码 0，26 个 host EID（pp_r1_d_host.log）
- `probe_urma_eids.py scan` 派单原命令 `--pool-group 0-3=UB_A --pool-group 4-7=UB_B`：**失败**，报 `RuntimeError: no host EID candidates`（pp_r1_d_probe.log）。原因：UB_A/UB_B 是占位符，非真实 host 设备名；scan 用 pool_group values 过滤 hosts（probe_urma_eids.py:348-351），过滤后为空即抛错。docs/mempool_validation.md 也明确 UB_A/UB_B 须替换为真实设备名。
- **修正**：按 `probe host` 真实设备名替换为 `--pool-group 0-3=udmac1d1e2 --pool-group 4-7=udmac0d1e3` 后 8/8 通过，生成 remote-shared.json + pairs.csv（pp_r1_d_probe_grouped.log:112 `EID_PROBE_SUMMARY {"status": "pass", "requested": 8, "passed": 8, "attempts": 20, "missing": []}`）
- 派单命令缺陷：D 阶段 scan 的 pool-group 占位符未填真实设备名，绿区按文档替换后通过。**此偏差需蓝区确认分组名是否符合预期**（0-3→udmac1d1e2，4-7→udmac0d1e3）。

**Stage E Pool 启动 + DP1/TP1 首发 — E1/E2 通过，E3 阻塞（模型权重放不下）**

- E1 Pool 启动：退出码 0；`URMA_JETTY_READY` 计数 = 8（≥1 命中；pp_r1_e_pool.log:38-45）；`MEMPOOL_READY`(行29)、`EMBEDDED_STORE_READY`(行46-47)、`URMA_MOCK_POOL_READY`(行48) 均出现。Pool 现仍运行（pid 2167590）
- E2 dry-run：退出码 0，打印完整配置（pp_r1_e_dryrun.log）
- E3 实跑（DP1/TP1/eager/BS1/offload）：**失败，NPU OOM**
  - 日志：pp_r1_e_run.log
  - 首个报错点（前后 50 行见下）：`torch.OutOfMemoryError: NPU out of memory. Tried to allocate 3.00 GiB (NPU 0; 91.75 GiB total capacity; 89.39 GiB already allocated; ...)`（行167/245/332）
  - 栈：make_layers → DeepseekV2DecoderLayer → DeepseekV2MoE → AscendFusedMoE.__init__ → w4a4_mxfp4.get_weight 的 `w13_weight = torch.empty(...)`（deepseek_v2.py:1253 → fused_moe_0_23_0.py:273 → method_adapters.py:225 → w4a4_mxfp4.py:142）
  - 时序：07:24:03 `Starting to load model` → 07:24:09 OOM，6 秒内创建 89.39GiB 权重参数后在第 ~19 层 w13 分配 3GiB 失败
  - **根因**：GLM-5.2-W4A4C8（391GB 权重，78 层 × 256 专家 × 18.9MB/专家 = 377GB MoE）在 TP1 单卡 96GB 物理放不下。remote-offload 只 offload KV cache（remote-shared），权重须常驻 NPU HBM。TP1 单卡仅能容纳 ~18 层权重即 OOM。
  - 权重量化：每 expert 18.9MB（w13 4096×3072B + w2 6144×1024B，uint8 存 2×fp4）；MoE 每层 4.83GB；78 层合计 377GB（与磁盘 391G 吻合）

**Stage E 补充尝试（非派单参数，绿区为定位而试，供蓝区裁决）**

- **TP2**（VLLM_ASCEND_SMOKE_TENSOR_PARALLEL_SIZE=2，DP1）：同样 OOM（pp_r1_e_run_tp2.log:286/354）。关键证据：TP2 下每卡仍 88.35GiB ≈ TP1 的 89.39GiB。EP 分片生效（每卡 128/256 专家，w13 块 1.50GiB = 128×2×2048×3072），但 MoE intermediate 不按 TP 分片（intermediate_size_per_partition=2048 未除 tp_size，fused_moe/layer.py:264 上游计算但 Ascend 路径实际为全量）。TP2 每卡理论需 188GB MoE 权重，仍远超 96GB。
- **TP8**（VLLM_ASCEND_SMOKE_TENSOR_PARALLEL_SIZE=8）：被 smoke 测试硬拒（pp_r1_e_run_tp8.log）：`ValueError: remote shared smoke requires TP1/TP2...`。remote_shared_config.py validate_topology:161 `tp not in (1, 2)` 硬编码只支持 TP1/TP2。TP≥4 无法经 env 开启。
- **结论**：remote_shared TP1/TP2 拓扑 + 391GB 模型权重，在 96GB/卡 8 卡节点上物理无法加载权重。非链路/驱动/代码 bug，是模型规模与 remote_shared 拓扑上限（TP≤2）的根本冲突。F 阶段（TP2/DP4）同样必然 OOM，按协议 E 未全中不执行 F。

### git log

- 框架仓：`d312265 docs: init blue-green pingpong, dispatch R1 remote-offload single-node bring-up`（前一个 cac7e2c perf(dsa): dispatch shared group IO once from Full）
- 算子仓：`6e85590 feat(tools): report per-epoch layer cross-port latency dispersion`

### remote-shared.json 全文

```json
{
  "store_endpoint": "127.0.0.1:19080",
  "devices": {
    "0": {"du_endpoint": "127.0.0.1:20000", "device_eid": "000000000f00020000100000df0001f8", "eid_index": 0, "pool_eid": "000000000f7f020000100000df0a0bf8"},
    "1": {"du_endpoint": "127.0.0.1:20001", "device_eid": "000000000f40020000100000df0031f8", "eid_index": 0, "pool_eid": "000000000f7f020000100000df0a0bf8"},
    "2": {"du_endpoint": "127.0.0.1:20002", "device_eid": "000000000f40020000100000df0051f8", "eid_index": 0, "pool_eid": "000000000f7f020000100000df0a0bf8"},
    "3": {"du_endpoint": "127.0.0.1:20003", "device_eid": "000000000f00020000100000df0061f8", "eid_index": 0, "pool_eid": "000000000f7f020000100000df0a0bf8"},
    "4": {"du_endpoint": "127.0.0.1:20004", "device_eid": "000000000f40020000100000df0091f8", "eid_index": 0, "pool_eid": "000000000f3f030000100000df080bf8"},
    "5": {"du_endpoint": "127.0.0.1:20005", "device_eid": "000000000f00020000100000df00a1f8", "eid_index": 0, "pool_eid": "000000000f3f030000100000df080bf8"},
    "6": {"du_endpoint": "127.0.0.1:20006", "device_eid": "000000000f00020000100000df00c1f8", "eid_index": 0, "pool_eid": "000000000f3f030000100000df080bf8"},
    "7": {"du_endpoint": "127.0.0.1:20007", "device_eid": "000000000f40020000100000df00f1f8", "eid_index": 0, "pool_eid": "000000000f3f030000100000df080bf8"}
  }
}
```

### MEMPOOL_WRAPPER_ARGS 原文

```
--du-ports 20000,20001,20002,20003,20004,20005,20006,20007 --du-devices udmac1d1e2,udmac1d1e2,udmac1d1e2,udmac1d1e2,udmac0d1e3,udmac0d1e3,udmac0d1e3,udmac0d1e3 --du-eid-indices 0,0,0,0,0,0,0,0
```

### E3 失败首个报错点前后 50 行

见 /home/fjj/pp_r1_e_run.log 行 145–245（OOM 原始 allocator 报错 行167；栈顶 行245 `torch.OutOfMemoryError`）。关键原文（不加工）：

```
行167: E1027 07:24:09.642086 2168139 ../torch_npu/csrc/core/npu/NPUCachingAllocator.cpp:1332] NPU out of memory. Tried to allocate 3.00 GiB (NPU 0; 91.75 GiB total capacity; 89.39 GiB already allocated; 89.39 GiB current active; 1.55 GiB free; 89.43 GiB reserved in total by PyTorch).If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.
行245: torch.OutOfMemoryError: NPU out of memory. Tried to allocate 3.00 GiB (NPU 0; 91.75 GiB total capacity; 89.39 GiB already allocated; 89.39 GiB current active; 1.55 GiB free; 89.43 GiB reserved in total by PyTorch).If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.
```

栈根因（行 322-349）：

```
deepseek_v2.py:1253  self.start_layer, self.end_layer, self.layers = make_layers(
utils.py:658        + get_offloader().wrap_modules(
deepseek_v2.py:1149  self.mlp = DeepseekV2MoE(
deepseek_v2.py:326    self.experts = FusedMoE(
fused_moe_0_23_0.py:273  self.quant_method.create_weights(layer=self, **moe_quant_params)
method_adapters.py:225   weight_param = self.quant_method.get_weight(
w4a4_mxfp4.py:142        param_dict["w13_weight"] = torch.empty(
```

### 待蓝区决策

1. **E3 阻塞根因**：GLM-5.2-W4A4C8（391GB）权重在 remote_shared 支持的 TP1/TP2 拓扑下物理放不进 96GB/卡。需蓝区定夺：
   - a) 换更小模型（本机现有 DeepSeek-V4.1-Flash 系列 728-801GB 更大，均不适用；需提供可单机/TP2 装下的模型）
   - b) 扩展 remote_shared 支持 TP≥4（代码改动，绿区不做）
   - c) 换权重 offload 方案（当前 remote-offload 只 offload KV cache，权重需常驻）
2. **D 阶段派单命令修正确认**：pool-group 占位符 UB_A/UB_B → 真实设备名 udmac1d1e2/udmac0d1e3（8/8 通过）
3. **F 阶段未执行**：E3 未全中，按协议跳过
4. 当前状态：Pool 仍在运行（pid 2167590），8 卡 NPU 空闲，无残留 python 进程，环境就绪可接续
## §R2 交付（绿区，2026-10-27）

### 执行环境

- 机器 IP：90.96.17.111（aarch64，16× Ascend 950DT 96GB/卡，鲲鹏 950，1505GB RAM）
- 容器：dsa-full（host 网络/pid，挂 /home /etc /data /usr/local/Ascend/driver）
- 框架仓：/home/fjj/vllm-ascend-remote-offload（分支 vllm-ascend-v0.23.0-custom-remote-offload，HEAD=7031076）
- 算子仓：/home/fjj/a5-remote-offload-ops（main，6e85590db）
- 模型：/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731（391G）
- 配置：/home/fjj/a5-remote-offload-ops/tmp/eid-probe-r1/remote-shared.json（**已按 2026-10-20 文档补 per-device store_endpoint**，见下）
- Pool：backing /dev/shm/kv-remote-r2b（**R2-0 重启**，sibling 修正为 5），8 DU ready
- 执行时间：2026-10-27 22:47–22:59（机器本地时间）
- 日志：/home/fjj/pp_r2b_dryrun.log / pp_r2b_run.log / pp_r2b_pool.log / pp_r2_dryrun.log / pp_r2_run.log / pp_r2_pool.log

### 通过标准逐条裁决

**1. dry-run 退出码 0 — 通过**。pp_r2b_dryrun.log 完整打印配置（rank_map、required_local_npus=8、total_requests=4、run_name=dp4_tp2_offload_eager_mtp_eager_hbm_full_remote_shared），RC=0。

**2. 实跑进程退出码 0 + 生成文本 — 未命中**。实跑在 profile_run 阶段崩溃（EI0006），未到 generate。

**3. DSA_EXECUTION_PLAN → DSA_STORE_PUT_DONE → DSA_SHARED_CONNECT_* → DSA_SHARED_CONSUMED ×2 → DSA_SHARED_DRAIN_OK — 未命中**。remote-shared 链路（CONNECT/CONSUMED/DRAIN）本轮**未触达**：崩溃发生在 profile_run 的 dummy run，早于任何 decode/PUT/CONNECT。Pool 侧有 OPEN（8 个 `REMOTE_DU_SOURCE kind=shared_store request_protocol=6 group_gather=1`，各 DU 独立 store id），无 CREATE/PUT。

**4. CONNECT rows=4 — 未触达**（同 3）。

**5. DSA HBM CACHE CAPACITY REPORT — 未触达**（同 3）。

### 崩溃点（首个报错）

```
(Worker_DP1_TP0_EP2 pid=2368541) ERROR 10-27 22:52:34 [multiproc_executor.py:987]
  File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/worker/worker.py", line 564, in determine_available_memory
    self.model_runner.profile_run()
  File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/worker/model_runner_v1.py", line 4632, in profile_run
    super().profile_run()
  File "/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py", line 6228, in profile_run
    hidden_states, last_hidden_states = self._dummy_run(...)
  File ".../model_runner_v1.py", line 4567, in _dummy_run
    outputs = self._model_forward(...)
  ...
  File "/vllm-workspace/vllm/vllm/model_executor/models/deepseek_v2.py", line 377, in forward
    router_logits, _ = self.gate(hidden_states)
  File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/ops/fused_moe/gate_linear.py", line 59, in forward
    x = x.to(torch.float32)
RuntimeError: The Inner error is reported as above. The process exits for this inner error, and the current working operator name is HcclAllGather.
[ERROR] ... Communication_Error_Get_Socket(EI0006): Getting socket times out. Reason: Connection fault detection has been disabled.
1. The remote end does not initiate a connection request. some NPUs in the cluster are abnormal.
2. The remote end does not initiate a connection request because the collective communication operator is started too late or is not started by some NPUs in the cluster.
3. The communication link is disconnected. ...
```

- 时间：22:52:34，8 worker 全部同时（窗口 38ms：22:52:34.534–.572）
- 阶段：profile_run → _dummy_run → deepseek_v2.forward → gate（MoE router 前的 HcclAllGather，异步算子延迟报出，gate_linear.py:59 的 `.to()` 是栈顶不是真凶）

### 关键前置证据（R2 相对 R1 的新增修复）

**A. per-device store_endpoint 修复（Store instance changed 根因）**

- 现象：R2 首轮（pp_r2_run.log + pp_r2_pool.log）8 个 DU 中 device 1-7 报 `REMOTE_DU init failed: Store instance changed`（remote_store_control.cpp:125 `expected != id_`）。
- 根因：remote-shared.json 只有全局 store_endpoint=127.0.0.1:19080，device 表无 per-device store_endpoint → 客户端 `RemoteKvStore(device.store_endpoint or global)` 全用 19080 的 store_id 发给各 DU child，而 embedded 模式下每 DU child 是独立 store（19080-19087，id 各异），device≠0 的 child 校验 id 不符。
- 修复：按 remote_store_lifecycle_validation.md 2026-10-20 节（"device表中每卡store_endpoint必须指向所属child"）给 device 表补 per-device `store_endpoint: 127.0.0.1:19080+N`（与 du_endpoint 20000+N 配对）。**纯配置补全，未改代码**。
- 验证：修复后 Pool 日志出现 8 个 `REMOTE_DU_SOURCE kind=shared_store request_protocol=6 group_gather=1`，各 DU 独立 store id；`Store instance changed` 不再出现。

**B. DU sibling CPU 修复（`DU helper must be a distinct hardware sibling` 根因）**

- 现象：R2 首轮 device 0 报 `REMOTE_DU init failed: DU helper must be a distinct hardware sibling of its primary CPU`（cpu_placement.h:59 CheckSiblingCpus）。
- 根因：派单命令 `--gather-cpus 4 --du-sibling-cpus 6`。实测本机 CPU4 thread_siblings=4-5、CPU6=6-7，4 的硬件兄弟是 5 不是 6，8 个 DU 只要走到 Session 构造必失败。
- 修复：R2-0 重启 Pool，`--du-sibling-cpus 5`，backing 换 /dev/shm/kv-remote-r2b。8 个 DU 全 ready（URMA_JETTY_READY=8）。**此偏差需蓝区确认派单命令后续修正**。

**C. EI0006（当前卡点，需蓝区判读）**

- 现象：8 worker 权重装载全部成功（各 61.3GB，EP8 分片正确），EP8 组 init 成功（world_size=8，tcp://127.0.0.1:40622，backend=hccl），但 profile_run dummy run 首次 HcclAllGather 建链超时。
- **设备映射正确性**（排除设备分配问题）：8 worker AIV_URMA_LOCAL_EID 各绑一张物理卡（phy_id=0-7 全不同），DP0→卡0/1、DP1→卡2/3、DP2→卡4/5、DP3→卡6/7。
- **最小 8 卡 HCCL 探针（绿区新增验证）**：容器内 8 进程 `ASCEND_RT_VISIBLE_DEVICES=rank` + `dist.init_process_group("hccl")` + `dist.all_gather`，**8/8 PASS**（init 0.0-1.4s，all_gather 1.7-2.5s，verify=True）。脚本 /home/fjj/hccl_probe_8npu.sh，日志 /tmp/hccl_probe.rank{0-7}.log。**证明裸 8 卡 HCCL 通信正常，非驱动/环境问题**。
- 差异点：探针是同一父进程 fork 8 个 python 各绑 1 卡；vLLM 是 4 个 EngineCore（DP0-3）各 spawn 2 个 TP worker，EP8 组跨 4 进程树建 `tcp://127.0.0.1:40622` 世界组。EI0006 的 "remote end does not initiate connection request" + 8 卡同时超时，指向 vLLM EP group 的 HCCL 建链（多进程树跨 DP）与裸探针（单进程树）的差异。
- 参考代码（绿区只读定位，未改）：vllm_ascend/ops/fused_moe/moe_comm_method.py AllGatherCommImpl（TokenDispatcherWithAllGather，用 npu_moe_init_routing_v2 + all_gather + npu_moe_token_unpermute）；vllm_ascend/worker/worker.py:403 `if not vllm_version_is("0.23.0")` 的 DP local_rank 偏移块——**该块在 0.23.0 下被跳过**（DP 设备偏移逻辑仅 v0.24+ 生效），但 EID 证明实际 8 卡分配正确，故设备偏移是否生效不影响 EI0006 结论。

### git log

- 框架仓：`7031076 docs(pingpong): R1 adjudication + dispatch R2 DP4/TP2(EP8) first run`（HEAD，未动代码，仅追加本交付）
- 算子仓：`6e85590`（未动）

### 日志留存

- /home/fjj/pp_r2b_run.log（EI0006 全栈，行 1540-1650 崩溃点，行 1307-1509 AIV_URMA_LOCAL_EID）
- /home/fjj/pp_r2b_pool.log（8 REMOTE_DU_SOURCE，embedded store ready）
- /home/fjj/pp_r2b_dryrun.log / pp_r2_dryrun.log / pp_r2_run.log / pp_r2_pool.log（R2 首轮 Store instance changed + sibling）
- /home/fjj/hccl_probe_8npu.sh + /tmp/hccl_probe.rank{0-7}.log（8 卡 HCCL 探针 PASS）
- remote-shared.json.bak-r2 / .bak-r2b（修复前后配置）

### 待蓝区决策

1. **EI0006 归属**：裸 8 卡 HCCL AllGather PASS（8/8），vLLM EP8 组（跨 4 EngineCore 进程树）首次 HcclAllGather 超时。倾向框架层（EP group 建链/多进程树 HCCL），请蓝区判读并给下一手（env 绕行如 HCCL_CONNECT_TIMEOUT / ASCEND_LAUNCH_BLOCKING=1 抓准确栈，或代码修复）。
2. **R2-0 命令偏差**：`--du-sibling-cpus 6` 改为 5（硬件兄弟核），backing 换 kv-remote-r2b，请蓝区确认后续轮沿用。
3. **per-device store_endpoint**：绿区按文档补全（未改代码），请蓝区确认符合预期（现 Pool 用 kv-remote-r2b，device 表 store_endpoint 指向各 DU child 19080+N）。

## §R4 交付（绿区，2026-10-29）

### 执行起止时间
- R4-0 起：2026-10-29 04:5x（清场+扩 /dev/shm）
- R4-1 构建：2026-10-29 05:02–05:14（build_and_install.sh + urma_mock_pool + build_store）
- R4-3 运行：2026-10-29 05:22:17（model.log 首行）→ 05:23:10 失败（EI0015）
- 全程单节点（90.96.17.111，16×950DT，容器 dsa-full）

### 版本基线
- 算子仓 `git log --oneline -3`（/home/fjj/a5-remote-offload-ops，HEAD f63a5bf）：
  - f63a5bf Update and rename remote_offload_results.csv to overlap-19090.csv
  - 1a0f252 feat(pool): add opt-in cross-port Gather worker sharing
  - fe974e4 revert(copy-sfa): restore first-benefit ScatterSFA kernels
- 框架仓 `git log --oneline -2`（/home/fjj/vllm-ascend-remote-offload，HEAD 4d5ce42）：
  - 4d5ce42 docs(pingpong): R4 派单：作者新流程跑通优化版 remote-offload；R3 superseded
  - 1ee1faa docs(pingpong): R2 adjudication + dispatch R3 EI0006 bisect

### R4-0 清场+扩 /dev/shm —— RC=0 通过
- 杀旧 Pool/框架进程、rm 旧 backing；`mount -o remount,size=384G /dev/shm` 成功：shm 384G total / ~373G avail。
- 通过标准逐条命中：清场无残留、remount 成功。✅

### R4-1 算子仓更新+重编 —— RC=0 通过
- 算子仓 pull：9066 代理持续 407，改用算子仓自带 git config 代理 90.255.129.35:3128（curl 200）成功 pull 至 f63a5bf。
- 容器内后台重编全链完成：build_and_install.sh + urma_mock_pool build.sh + build_store.sh。
- 产物核验：
  - `import kvcache_scatter_copy_custom_ops` → MOD_OK（算子注册正常）。
  - urma_mock_pool（/home/fjj/a5-remote-offload-ops/test/urma_mock_pool/urma_mock_pool，Oct 29 05:12 重建）help 含 `du-pipeline-mib`、`du-gather-shared`、`DU_AND_GATHER`（新能力齐全，mtime 与构建时间吻合）。
  - urma_pool_server（/home/fjj/a5-remote-offload-ops/tmp/mempool-build/urma_pool_server）help 含 `gather-scheduler`、`du-store-prefault`（各 grep 命中 2 处）。✅

### R4-2 拓扑/资源核验 —— 通过
- 4 NUMA 节点（0-3）；node0 CPU 0-95、node2 CPU 192-287；node0↔node2 距离 20（跨 NUMA 双 Pool 拓扑成立）。
- affinity 384 逻辑核；shm 384G。✅

### R4-3 作者流程首发 BS1（profile_pool_inference.py 一键双 Pool + 模型）—— **RC=1 失败（EI0015）**

**通过标准逐条判定（R4-3 通过需同时满足）：**
- 工具退出 RC=0：❌（RC=1，`RuntimeError: model exited 1; see raw/model.log`）
- `tmp/r4-author-bs1/summary.txt` 含 `RUN_OK`：❌（内容为 `RUN_FAILED`）
- `raw/model.log` 有生成文本：❌（无推理生成）
- model.log 依次出现 `DSA_EXECUTION_PLAN → DSA_STORE_PUT_DONE → DSA_SHARED_CONNECT_BEGIN → DSA_SHARED_CONNECT_OK ... backend=real_store → ≥2 轮 DSA_SHARED_CONSUMED → DSA_SHARED_DRAIN_OK`：❌（仅到达 DSA_EXECUTION_PLAN，后续阶段未触达）
- `raw/overlap-*.csv` 存在且非空：❌（PERF_DISABLED，未采集）
- `run-status.json` 里 `graceful=true`、`errors=[]`：❌（graceful=true 但 errors 非空，见下）

**run-status.json**：
```
{
  "errors": ["RuntimeError: model exited 1; see raw/model.log"],
  "graceful": true,
  "mode": "remote",
  "no_perf": true,
  "launch": {"host": "dsa-host", "entrypoint": "examples/dsa_demo/simple_test_1p.py",
             "node_count": 1, "node_rank": 0, "master_addr": null, "master_port": null, "sync_port": null},
  "du_reply_segments": 1, "du_pipeline_mib": 1.0, "gather_scheduler": "shared",
  "pool_enabled": true
}
```

**summary.txt（tmp/r4-author-bs1/）**：
```
RUN_FAILED
RuntimeError: model exited 1; see raw/model.log
PERF_DISABLED
RUN_FAILED
未采集SPE/缺页事件、未做时钟校准或ROI握手；不是ROI归因或性能准入结论。
保留classify计时/overlap CSV: raw/overlap-*.csv
推理日志: raw/model.log; Pool日志: raw/pool-*.log
```

**成功抵达的里程碑（说明双 Pool 与模型装载阶段全部正常）：**
- 双 Pool 完全 ready：`POOL_ROI_POOL_READY groups=['a', 'b']`（8×URMA_JETTY_READY、8×EMBEDDED_STORE_READY 各 32GB）。
- store 路由正确：device0-3→du 20000-20003/store 19080-19083；device4-7→du 20004-20007/store 19084-19087（Pool A numa0 / Pool B numa2）。
- 8 个 worker 全部到达 `DSA_EXECUTION_PLAN`（model_runner_v1.py:419，05:23:08，权重装载+执行计划下发成功）。
- 随后 05:23:10 七/八个 worker 在 load_model 阶段崩（EP 通信组创建）。

**失败根因链（model.log 首个报错点，Worker_DP1_TP1_EP3 前后 50 行，行号 1101-1175）：**
```
1101:(Worker_DP1_TP1_EP3 pid=1757500) ERROR 10-29 05:23:10 WorkerProc failed to start.
... WorkerProc.__init__ → self.worker.load_model() → model_runner.load_model() →
... get_model → initialize_model → DeepseekV2DecoderLayer → DeepseekV2MoE.__init__ →
... FusedMoE.__init__ → setup_moe_comm_method →
moe_comm_method.py:57    _MoECommMethods[MoECommType.ALLTOALL] = AlltoAllCommImpl(moe_config)
moe_comm_method.py:93    self.token_dispatcher = self._get_token_dispatcher()
moe_comm_method.py:250   return TokenDispatcherWithAll2AllV(
token_dispatcher.py:472  self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
1171: RuntimeError: create_config:../torch_npu/csrc/distributed/HCCLUtils.cpp:140
     HCCL function error: hcclCommInitRootInfoConfig(numRanks, &rootInfo, rank, config,
     &(comm->hcclComm_)), error code is 4
1173: [PID: 1757500] 2026-10-29-05:23:10.221.707 Communication_Error_Ranktable_Detect(EI0015):
     Failed to collect cluster information of the communicator based on rootInfo detection.
     Reason: RootInfoDetect failed.
      Solution: 1. Check whether all ranks in the communicator have delivered the communicator
      creation interface. 2. Check the connectivity between the host networks of all nodes and
      the server node. 3. Check whether the HCCL_SOCKET_IFNAME environment variable of all
      nodes is correctly configured. 4. Increase the timeout by configuring the
      HCCL_CONNECT_TIMEOUT environment variable.
```

**受影响 worker（7/8，DP0_TP0_EP0 未在错误列表中）：**
- Worker_DP1_TP1_EP3（pid 1757500）、Worker_DP3_TP1_EP7（1757502）、Worker_DP0_TP1_EP1（1757499）、Worker_DP2_TP0_EP4（1756967）、Worker_DP2_TP1_EP5（1757501）、Worker_DP1_TP0_EP2（1756965）、Worker_DP3_TP0_EP6（1756966）。
- 触发点均为 `token_dispatcher.py:472 backend.get_hccl_comm_name(local_rank)` → `hcclCommInitRootInfoConfig ... error code is 4` → EI0015。

**日志路径：**
- 运行日志：/home/fjj/pp_r4_run.log
- 构建日志：/home/fjj/pp_r4_ops_build.log
- summary：/home/fjj/a5-remote-offload-ops/tmp/r4-author-bs1/summary.txt
- precheck：/home/fjj/a5-remote-offload-ops/tmp/r4-author-bs1/precheck.txt
- model.log 末 100 行：/home/fjj/a5-remote-offload-ops/tmp/r4-author-bs1/raw/model.log（首个报错 1101 行起）
- Pool 日志：/home/fjj/a5-remote-offload-ops/tmp/r4-author-bs1/raw/pool-a.log、pool-b.log

**判读：** 报错为 **EI0015（RootInfoDetect failed, error code 4）**，非 R4-3b 触发条件的 EI0006/HcclAllGather 超时 → **R4-3b 严格按字面不触发**。EI0015 官方 solution 第 4 条指向 HCCL_CONNECT_TIMEOUT（与 R4-3b 意图同族），但时序上从 DSA_EXECUTION_PLAN（05:23:08）到失败（05:23:10）仅 2 秒，不像超时，更像 communicator 创建时 rank 未同步。双 Pool 完全 ready → 排除 Pool 根因。失败点收敛到 MoE EP8 ALLTOALL 通信组创建（与 R2 EI0006 建链超时、R3-A dispatch timeout EmptyQueue 同族，均在 EP8 跨 DP 建链层）。

### R4-3b / R4-4 —— 条件不触发，未执行
- R4-3b 触发条件=报错含 EI0006 / HcclAllGather 超时：实际报错为 EI0015，**不触发**。
- R4-4 触发条件=仅当 R4-3/3b 通过：R4-3 失败，**不适用**。

### 判读预案命中
- **「R4-3/3b 均失败 ⇒ 阴性」** 命中：summary/precheck/model.log 已按回传格式附上，蓝区对照作者流程逐参数定位。

---

### 附：R3 实验数据（按派单要求原样附加）

R3 派单（1ee1faa）三阶段实验中仅 R3-A（offload 关闭对照实验）执行，R3-B/R3-C 因 R3-A 失败未触发。

**R3-A 结果（offload OFF，EP8 基线）：RC=1 失败**
- 日志：/home/fjj/pp_r3a_run.log（2026-10-28 01:07:47 启动）
- 结尾：`RuntimeError: offline DP rank failed: dsa-dp-0=1, dsa-dp-1=1, dsa-dp-2=1, dsa-dp-3=1`（4 个 DP rank 全崩）
- 失败根因：profile_run → _dummy_run → deepseek_v2 MoE 前向；4 个 EngineCore 均报 `No available shared memory broadcast block found in 60 seconds`（patch_shm_broadcast.py:74，01:11-01:12）；底层为 **HCCL dispatch timeout**（Process group work ALLGATHER/REDUCE_SCATTER/ALLREDUCE ... dispatch timeout. device 1: EmptyQueue）。
- 关键含义：offload 关闭时 EP8 也崩 → 问题在 HCCL/EP 通信层而非 remote-offload 引入；裸探针 8/8 PASS 与 vLLM 多进程树差异仍在。
- 运行起止：2026-10-28 01:07:47 → 01:12+ 失败（R3-A 报错点前后行号 767-774 见 pp_r3a_run.log）。
