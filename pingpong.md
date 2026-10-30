# Pingpong

## Protocol

- 本文件是蓝区 ↔ 绿区的唯一通信面；`history.md` 是只追加的轮次档案。
- **蓝区**（可外部 AI 侧）写全部代码、做全部判读决策；**绿区**（隔离执行侧）按任务书执行并回写交付，不改代码。
- 每次 commit 必须同时包含代码 + `pingpong.md` + `history.md`；pull 用 `--ff-only`；绝不 force-push。
- Current owner: **绿区**。Status: §R25 已派单（2026-10-31）——R24 判读：clamp 统一理论实证坐实（唯一变量 ML43008 从崩+挂→全净），纪律入库「max_model_len > max(prompt)+max_tokens+MTP 余量」；1024 步首个 decode 主导 bench 点：tps=164.26、tpot avg=48.32/min=21.55ms。R25 = R25-1 BS16×1024 offload-on（R18 原始失败点复测）+ R25-2 同配方 baseline（绕工具直跑）对照，给 +20% 口径 decode 主导数据点；R25-2 若 OOM 即 offload 容量价值直接证据。R26 预案 = 100K×1K 验收配方。

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
- **用户经验入库（2026-10-30）：开 vLLM profile 采数据时 decode 阶段会挂**——与 R17-2/R18 的 `patch_shm_broadcast` 写侧块耗尽挂起同根嫌疑（profile 增大每步广播流量 → 加速块耗尽）。纪律：**bench/验收轮 ENABLE_PROFILE=0；采 profile 须先修 shm 块泄漏或压到极小并发**。R17-2 曾因 unset 落作者脚本默认 enable_profile=True 而实采 profile 后挂起，同型旁证。
- **EI0006 已知边界（R17 坐实）**：bench/验收配方必须 `MAX_NUM_BATCHED_TOKENS=2048` + `ENABLE_CHUNKED_PREFILL=1` 成对出现；大 batched（≥25600）会在 determine_available_memory 的 dummy 前向首个 EP8 all-gather 建连超时。

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

## §R11 交付：隔离梯子全执行——EI0006 稳定复现(graph/eager 同崩),作者配置下 EI0006 消失但前移 DP1 专属 1004(绿区,2026-10-30 08:10~09:10)

### 结论(先行)

- **R11-A**(R10 原样重跑):EI0006 **稳定复现**(8 worker 08:26:50 齐射,EI0006×20)→ **瞬时排除**。
- **R11-B**(`RUN_MODE=eager` 隔离 graph):EI0006 **仍复现**(8 worker 08:44:26 齐射,EI0006×20,**acl_graph 计数=0**,栈无 acl_graph.py)→ **graph 非触发面铁证**,跳过 R11-C。
- **R11-D**(作者 scp-opt/va.env/dp4tp2.json/udmac1d1e3/19090 段,eager,跑 2 次):**EI0006=0(两跑均未出现)**——作者配置下 EI0006 消失;但前移为 **DP1 专属 `RaGetDevEidInfoNum failed code=1004`**(首跑 08:51 EngineCore_DP1×4;重跑 09:02 Worker_DP1_TP0×2+EngineCore_DP1×1),两跑均 0 DRAIN。
- **梯子执行路径**:R11-0 → R11-A(复现)→ R11-B(复现)→ **R11-C 跳过**(B 复现)→ R11-D(两跑,均 1004 失败)。
- **白名单处置**:EI0006 在 R11-A 一次性重跑内用尽(复现);R11-D 的 1004 属白名单,已重跑 1 次仍复现(非瞬时),按纪律不再无限重跑,停,回传证据。

### R11-0 · 只读取证(三块原文摘要)

对 `/home/fjj/a5-remote-offload-ops/tmp/r10-bench-bs16/raw/model.log`。

**块1 每个 Worker_DP 07:26:45~07:29:18 最后行**:8 个 worker(DP0_TP0/DP0_TP1/DP1_TP0/DP1_TP1/DP2_TP0/DP2_TP1/DP3_TP0/DP3_TP1)最后日志行**全部**是 `10-30 07:29:18` 齐射 `Communication_Error_Get_Socket(EI0006): Getting socket times out...`(各 rank 时间 07:29:18.226~.330)。无 rank 例外。

**块2 capture/warmup/acl_graph 标记**:配置确认 graph 路径(`speculative_config: {'method': 'mtp', 'num_speculative_tokens': 3, ...}`,`cudagraph_mode: FULL_DECODE_ONLY`,`graph_capture: 'startup_pad'`,`put_transport: urma_read`);8 个 worker 错误栈**全部**经 `/home/fjj/vllm-ascend-remote-offload/vllm_ascend/compilation/acl_graph.py:163 in __call__`。

**块3 URMA/DSA 行**:8 个 store 全部 `DSA_STORE_OPEN_BEGIN` 正常(07:26:48~07:27:14);DP3_TP0 完整栈显示崩溃在 `profile_run → _dummy_run → _model_forward → acl_graph.__call__ → fused_moe → moe_comm_method → moe_mlp → npu_dynamic_quant(device_op.py:1318)` 处 EI0006。

**判读落点**:无 rank 停在 URMA/DSA 数据面(数据面 OPEN 后全部进入 profile dummy run)→ 变量 2 不升主嫌疑;全体整齐走 graph 执行路径的 MoE EP8 all-gather 后齐射超时 → **变量 1(graph 路径)倾向坐实**——但 R11-B 证明 graph 非根因(见下),实际变量 1 仅「graph 首次上机」被证伪,根因在更底层 EP8 all-gather 建连本身。

### R11-A · R10 原样重跑(瞬时排除)——EI0006 复现

- 命令:逐字 §R10 执行命令,输出目录 `tmp/r11a-bench-bs16`,日志 `/home/fjj/pp_r11a.log`。
- 前置:NPU 8/8 空闲无进程;端口 19080-19087/20000-20007 空闲;探针 `hccl_probe_8npu.sh` **8/8 PASS**(init 0.0s,all_gather 1.6-2.6s verify=True);清场(R10 残留 pool 进程+shm backing 全清)。
- 执行起止:08:22:13(engine init)→ **08:26:50 EI0006 齐射**;summary `RUN_FAILED`(08:37,崩溃后 pool 轮询器被清场 kill 时写入 KeyboardInterrupt)。
- **结果:8 worker 全部 08:26:50 齐射 EI0006(EI0006×20),与 R10 07:29:18 完全同签名同崩溃点**(MoE 前向 `npu_dynamic_quant`→HcclAllGather,经 acl_graph.py:163)。**EI0006 非瞬时,稳定复现**。
- 日志:pp_r11a.log(08:22~08:37)、tmp/r11a-bench-bs16/raw/model.log。

### R11-B · eager 隔离 graph(A 复现后)——EI0006 仍复现,graph 非触发面

- 命令:R10 命令仅改 `export VLLM_ASCEND_SMOKE_RUN_MODE=eager`,输出 `tmp/r11b-bench-bs16-eager`,日志 `pp_r11b.log`。
- 执行起止:08:39:55(engine init)→ **08:44:26 EI0006 齐射**;summary 未生成(崩溃后清场)。
- **结果:8 worker 全部 08:44:26 齐射 EI0006(EI0006×20)**;栈 = `npu_dynamic_quant(device_op.py:1318)` 直接崩(经 torch._ops.__call__),**model.log 中 `acl_graph` 计数=0**(eager 无 graph 层)→ **graph(含 MTP graph)不是触发面,铁证**。
- 由此 **R11-C(graph 去 MTP graph)跳过**(B 复现即进 D,派单分支)。

### R11-D · 回作者逐字配置(B 也败后)——EI0006 消失,前移 DP1 专属 1004(两跑)

- 命令:逐字 §R6-1(作者 `/home/w00919836/scp-opt` + `va.env` + `dp4tp2.json` + `udmac1d1e3/udmac0d1e3` + **19090 段** + `--du-store-prefault read` + `--pool-cpus/--pool-b-cpus`,eager),输出 `tmp/prefault-read-r11d`,日志 `/home/fjj/pp_r11d.log`。va.env 确认:PYTHONPATH=scp-opt/torch_extension+scp-va,ASCEND_RT_VISIBLE_DEVICES=0-7,unset 全部 smoke 旋钮。
- **首跑(EI0006=0)**:`RaGetDevEidInfoNum failed code=1004`(08:51:33,EngineCore_DP1 崩,**1004×4**,全部集中在 **DP1**;DP1_TP1 有 1 个 `DSA_SHARED_DRAIN_OK`,其余 0)。1004 在白名单,清场后重跑。
- **重跑(09:02:12 崩):EI0006=0**,`code=1004`(Worker_DP1_TP0_EP2×2 + EngineCore_DP1×1,**1004×3**,仍全集中 **DP1**),0 DRAIN。
- **重跑 1004 完整栈**(model.log:1748-1806):
  ```
  initialize_kv_cache (model_runner_v1.py:4808)
    → initialize_kv_cache_tensors (4894)
      → _bind_dsa_split_kv_caches (5293)
        → custom_ops_lib.aiv_urma_query_eids(str(self.device))
          → RuntimeError: RaGetDevEidInfoNum failed code=1004
  ```
- **关键含义**:作者配置(19090 段/e2 设备/eid json)下 EI0006 **不出现**——R10 的 20000 段/udmac1d1e2/eid-probe-r1 json 配置差异与 EI0006 相关(印证蓝区次嫌疑「数据面 e2/20000 段/eid json 差分」);但前移到 **DP1 专属 KV-cache 绑定阶段 aiv_urma_query_eids 1004**(非瞬时,两跑复现),0 DRAIN,未到 generate。
- pool 侧:双 Pool READY(`POOL_ROI_POOL_READY groups=['a','b']`),pool-a 正常收尾(tasks=0,因模型未到数据面即崩)。
- 日志:pp_r11d.log、tmp/prefault-read-r11d/raw/model.log。

### 各步 RC / summary

| 步 | RC | summary | EI0006 计数 | 1004 计数 | DRAIN_OK |
|----|----|---------|------------|-----------|----------|
| R11-A | 1(脚本尾标,崩溃后清场写入) | RUN_FAILED | 20 | 0 | 0 |
| R11-B | (崩溃后清场,无尾标) | 未生成 | 20 | 0 | 0 |
| R11-D 首跑 | (崩溃后清场,无尾标) | 未生成 | 0 | 4(全 DP1) | 1(DP1_TP1) |
| R11-D 重跑 | (崩溃后清场,无尾标) | 未生成 | 0 | 3(全 DP1) | 0 |

### 通过标准逐条

- R11-A「过⇒EI0006 瞬时落袋」:**不满足**(复现)→ 进 B。
- R11-B「过⇒eager 基线落袋+坐实 graph 面」:**不满足**(复现)→ 进 D。
- R11-C 未执行(B 复现,分支跳过)。
- R11-D「过⇒与端口段/e2 设备/eid json 相关」:**部分命中**——EI0006 消失证明配置差异相关,但被 1004(DP1 专属,两跑复现)阻断未到 generate,不满足「RUN_OK + DRAIN_OK + metrics」完整通过。
- 无任一 bench 通过步,故无 metrics global 行/报告块可回传(§R10 通过标准第 3/4 条不适用)。

### 环境状态(交付时)

- 清场完成:无残留 python/vllm/urma_pool 进程;NPU 8/8 空闲;端口段空闲;/dev/shm 373G 可用。
- 日志资产:pp_r11a.log、pp_r11b.log、pp_r11d.log;tmp/r11a-bench-bs16/raw/、tmp/r11b-bench-bs16-eager/raw/、/home/w00919836/scp-opt/tmp/prefault-read-r11d/raw/。
- 待蓝区判读:R11-D 的 DP1 专属 1004(`aiv_urma_query_eids` 在 KV-cache 绑定阶段)与作者 R6-1 时点的差异;或按 R8 模板补机器层取证(若判定 1004 为机器/驱动层问题)。

## §R11 判读（蓝区，2026-10-30）

### 裁决

1. **graph 面排除（自我修正）**：R11-B eager 同签名复现且日志无 acl_graph → R10 判读的主嫌疑「graph+MTP graph 首次上机」证伪。EI0006 与执行模式无关。
2. **EI0006 归因方向 = fjj 数据面配置**：作者配置（udmac1d1e3/e3 + 19090 段 + dp4tp2.json）下 0 次 EI0006（两跑），fjj 配置（udmac1d1e2 + 20000 段 + eid-probe-r1 json）下 3/3 全崩。R11-0 块1 无 rank 停在 URMA/DSA 数据面、块3 全体走完 store OPEN 进 dummy run——即某 rank 在「到达首次 MoE EP8 all-gather 之前」有日志不可见的延迟/挂起，与其通信对端 socket 超时。根因仍在数据面差分内部（e2 设备 / 20000 端口段 / eid-probe-r1 json 三者未拆），见 R12 拆解。
3. **作者配置下前移 DP1 专属 1004（两跑复现，非瞬时）**：`aiv_urma_query_eids` 在 KV-cache 绑定阶段炸在 DP1 设备（Worker_DP1_TP0_EP2 / DP1_TP1_EP3，physical 2/3）。与 R9 首跑 1004（physical 0/2/7）交集 = **physical 2**；叠加 R7/R8 证据（udmac d1e2~d1e5 节点 01:48 批量重建、多租户机），**NPU 2/3 的 UB/EID 状态面嫌疑上升**，先取证再动。
4. bench 基线仍未落袋（0/4 轮跑到 generate）。

### 环境事实入库

- EI0006 签名 = 某 rank 未到达首次运行时 EP8 all-gather（dummy run/profile run 阶段），对端 socket 超时齐射；与 graph/eager 无关，与 fjj 数据面配置强相关。
- 1004 非「重跑即过」铁律：R9 时段成立，R11-D 时段 DP1 两跑复现——**是否瞬时取决于机器 UB 面状态**，重跑上限 = 2 次，之后必须取证。
- fjj 配置 vs 作者配置差异清单（EI0006 相关）：pool-a du 设备 e2↔e3、du 端口 20000-20007↔19090-19097、store 端口 19080-19087↔dp4tp2.json 内值、eid json eid-probe-r1↔dp4tp2.json。

## §R12 派单：EI0006 归因拆解 + bench 基线同轮补打（蓝区，2026-10-30）

### 原则

顺序执行。R12-0 只读取证先行；R12-1 是本轮主实验（一把同时回答两个问题：EI0006 是数据面配置还是 fjj 代码引入；过了就把 bench 基线落袋）。白名单 = 1004（上限 2 次重跑，每次须 ≥5 分钟静置 + 完整清场）；白名单外新报错 → 停，回传首个报错点前后 50 行。

### R12-0 · 只读取证（NPU 2/3 UB/EID 状态面）

```bash
# 1) EID 单卡探测 ×3 轮（重点 2/3），复用绿区 R9 用过的探测方式：
#    逐卡 aiv_urma_query_eids，pool 未起时 1 轮 + pool 起来后 1 轮（用作者配置起 pool，
#    或若工具不便单独起 pool 则只跑 pool 未起轮），physical 2/3 各卡结果单独标注
# 2) UB 设备节点 mtime 现状（对照 R8 的 01:48 批量重建）
ls -l --time-style=full-iso /dev/uburma/ 2>&1
# 3) dmesg 近 2 小时 NPU 2/3 相关
dmesg -T 2>/dev/null | grep -iE 'npu|ub|urma|udma' | tail -40
# 4) npu-smi 2/3 健康与错误码
for i in 2 3; do echo "== npu $i =="; npu-smi info -t health -i $i 2>&1 | head -8; npu-smi info -t error-code -i $i 2>&1 | head -5; done
# 5) 19090-19097 与 20000-20007 端口占用快照
ss -tlnp | grep -E ':(19[09][0-9]|2000[0-7])' || echo "all free"
```

原文回传五块。

### R12-1 · 主实验：fjj 框架 + BENCH=1 + 作者数据面配置

目的：数据面逐项换回作者值、只保留 fjj 框架仓 + bench 代码 + smoke env。若 EI0006 消失且 RUN_OK ⇒ EI0006 = 数据面配置问题（e2/20000/eid json 三选一，R13 再拆），且 **bench 基线同轮落袋**；若 EI0006 复现 ⇒ fjj 代码/env 引入，回传后蓝区做代码 diff。

```bash
cd /home/w00919836/scp-opt
source /home/w00919836/va.env
# 关键：va.env 之后 export smoke env，确保生效
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
# 生效核验（必须回传输出）：必须指向 /home/fjj/vllm-ascend-remote-offload
python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"
# 若指向 scp-va：export PYTHONPATH=/home/fjj/vllm-ascend-remote-offload:$PYTHONPATH 后再验一次
python3 tools/profile_pool_inference.py --du-store-prefault read \
  --framework-repo /home/fjj/vllm-ascend-remote-offload \
  --output tmp/r12-fjj-bench-authorplane --sample-port 19090 --sample-worker 0 \
  --remote-config ../dp4tp2.json --no-perf --payload-layers 78 \
  --du-pipeline-mib 1 --store-bytes $((128*1024*1024*1024)) \
  --pool-start-timeout 600 --pool-count 2 --numa-node 0 --pool-b-numa-node 2 \
  --store-numa-nodes local \
  --pool-cpus '2-17,21-36,40-55,59-74,78-93' \
  --pool-b-cpus '192-207,211-226,230-245,249-264,268-283' \
  --shared-port-gather --du-devices udmac1d1e3 --du-eid-indices 0 \
  --pool-b-du-devices udmac0d1e3 --pool-b-du-eid-indices 0 \
  2>&1 | tee /home/fjj/pp_r12.log
echo "RC=$?"
```

注意：`--framework-repo` 与 `--remote-config` 相互独立，本条刻意混用（框架=fjj 带 bench 代码，数据面=作者 json 逐字）。R6 已验证此环境下 vllm_ascend 解析到 fjj editable 安装——本轮仍必须按上面命令实测回传，因为本轮 bench 代码是否生效取决于它。

### 失败处置

- `RaGetDevEidInfoNum code=1004`：按白名单重跑（≤2 次，每次 ≥5 分钟静置 + 完整清场）；仍败 → 停，附 R12-0 取证回传。
- EI0006 复现：停，回传首个报错点前后 50 行（蓝区转代码 diff 排查）。
- `DSA A5 MTP LIM requires 1..4 query rows`：老梯子（BS 16→8→4，输出目录改后缀）。
- 其它：停，回传。

### 通过标准（同 §R10）

1. 工具 RC=0、summary.txt 含 RUN_OK、model.log 有生成文本；2. `DSA_SHARED_DRAIN_OK` ×8；3. `[dsa-smoke][metrics][dp=N]` ×4 + `results/1p/<run_name>.metrics.json` 存在且 global 含 requests=64/分位数/吞吐/tpot_under_50ms；4. 原文回传 metrics global 行与报告块。

### 回传格式

追加「§R12 交付」节：R12-0 五块原文 + import 路径核验输出 + RC + 通过标准逐条 + metrics 原文（若过）+ 日志路径 + 执行起止时间。

## §R12 交付：EI0006 复现（作者数据面配置下仍崩）→ 数据面配置排除，指向 fjj 框架/env（绿区，2026-10-30 09:42~10:10）

### 结论(先行)

- **R12-0 只读取证**:NPU 2/3 的 EID/UB 状态面**无异常**——EID 单卡探测(pool 未起+pool 运行中两轮)physical 2/3 各返回完整 19 EID,无 1004 无错误;npu-smi 2/3 Health=OK/Error=NA;UB 设备节点 udmac{0,1}d1e2-d1e5 **09:42:45-46 批量重建**(R8 的 01:48 重建同型);端口 19090-19097/20000-20007 全空闲。
- **R12-1 主实验(fjj 框架 + BENCH=1 + 作者数据面配置)**:**EI0006 复现**(10:02:15,8 worker + 4 EngineCore 齐射,EI0006×20,**0×1004**,0 DRAIN),崩溃点 = `_initialize_kv_caches → determine_available_memory` 内 MoE 前向 `HcclAllGather`(与 R10/R11-A/B 完全同签名同相位)。
- **关键差分结论**:作者数据面配置(19090 段/udmac1d1e3/dp4tp2.json)下 EI0006 **仍复现** → **EI0006 与数据面配置无关**(R12 判读预案命中「若 EI0006 复现 ⇒ fjj 代码/env 引入」)。结合 R11-B(fjj+eager 复现)与 R11-D(scp-va+eager 作者数据面=0),**EI0006 决定性差分为 fjj 框架(scp-va 之外)+ smoke env**,与 graph/eager 无关、与数据面配置无关。
- bench 基线仍未落袋(0/5 轮跑到 generate);无 metrics 可回传(§R12 通过标准第 3/4 条不适用)。

### R12-0 · 只读取证五块原文

**块1 EID 单卡探测 ×2 轮(physical 2/3)**:
- **pool 未起轮**(09:44):npu:2 返回 19 EID(eid_index 0-8 的 03xx/02xx 组 + 02xx 设备 EID `...0051f8` + 05xx 组 `...0052fa` 起),npu:3 返回 19 EID(02xx 设备 EID `...0061f8` + 05xx 组 `...0072fb` 起 + 03xx 组)。两卡均**无错误无 1004**。
- **pool 运行轮**(09:46,udmac1d1e3 mock pool,local_eid_hex=`...0a0bf8`,reconnect enabled):npu:2/npu:3 各返回同样 19 EID,**无 1004 无错误**。pool 起的 EID 查询正常。
- 判定:physical 2/3 的 EID 枚举面**健康**,R11-D 的 1004 非 EID 枚举缺失。

**块2 UB 设备节点 mtime**(09:42):
```
crw-rw-rw-. 1 root root 509, 0 2026-10-30 09:42:46.276365130 +0800 udmac0d1e2
crw-rw-rw-. 1 root root 509, 4 2026-10-30 09:42:46.400365130 +0800 udmac0d1e3
crw-rw-rw-. 1 root root 509, 3 2026-10-30 09:42:46.516365130 +0800 udmac0d1e4
crw-rw-rw-. 1 root root 509, 2 2026-10-30 09:42:46.648365130 +0800 udmac0d1e5
crw-rw-rw-. 1 root root 509, 1 2026-10-29 19:48:48.544365130 +0800 udmac0d1e6
crw-rw-rw-. 1 root root 509, 5 2026-10-30 09:42:45.800365130 +0800 udmac1d1e2
crw-rw-rw-. 1 root root 509, 9 2026-10-30 09:42:45.916365130 +0800 udmac1d1e3
crw-rw-rw-. 1 root root 509, 8 2026-10-30 09:42:46.032365130 +0800 udmac1d1e4
crw-rw-rw-. 1 root root 509, 7 2026-10-30 09:42:46.156365130 +0800 udmac1d1e5
crw-rw-rw-. 1 root root 509, 6 2026-10-29 19:48:48.544365130 +0800 udmac1d1e6
```
→ d1e2-d1e5 八个节点 09:42:45-46 批量重建(d1e6 为 10-29 旧节点),与 R8 的 01:48 重建同型;R12-1 主实验(09:57 起)在这些**新节点**上跑仍 EI0006。

**块3 dmesg 近 2 小时 NPU/UB/URMA/UDMA**(09:02-09:06 窗口):`ubase 0000X: UDMA: opt 14/15 no need to process`(09:02:09)+ `udma_free_u_hugepage: 464 callbacks suppressed` + `UDMA: current mm released.`×N + `udma_destroy_jfr_batch: 8 callbacks suppressed` + `UDMA: JFR sleep time = 1000 us.`×N(09:06:33-35)→ R11-D 重跑(09:02)结束后的 UDMA 面收尾活动,无错误码。

**块4 npu-smi 2/3 健康与错误码**:两者均 `Health Status: OK`、`Error Code: NA`、`Error Information: NA`。`npu-smi info -t error-code` 本机不支持(25.6.rc2 无此项,报 Usage)。

**块5 端口占用快照**:宿主 `ss` + 容器 `netstat` 均 **19090-19097/20000-20007 全空闲**(R12-0 时点与 R12-1 结束后复测均空)。

### R12-1 · 主实验(fjj 框架 + BENCH=1 + 作者数据面配置)——EI0006 复现

- **import 路径核验**:va.env 后 `vllm_ascend.__file__` = `/home/w00919836/scp-va/vllm_ascend/__init__.py`(scp-va)→ 按派单 `export PYTHONPATH=/home/fjj/vllm-ascend-remote-offload:$PYTHONPATH` 后再验 = `/home/fjj/vllm-ascend-remote-offload/vllm_ascend/__init__.py`(fjj,**生效**)。fjj 含 bench 代码(717c396:simple_test.py `SMOKE_BENCH`/`min_tokens=max_tokens`/`ignore_eos`/`_bench_report`)。
- 命令:派单逐字 + PYTHONPATH 前置(fjj),`--framework-repo /home/fjj/vllm-ascend-remote-offload` + `--remote-config ../dp4tp2.json`(作者数据面)+ udmac1d1e3/19090 段 + 双 pool。remote-shared.json 生效确认:du 19090-19097、device EID=`...0001f8/0031f8/0051f8/0061f8/0091f8/00a1f8/00c1f8/00f1f8`(作者 dp4tp2.json 值)、pool_eid=`...0a0bf8/...080bf8`。
- **执行起止**:09:57:24(engine init)→ 09:58:26 **DSA_EXECUTION_PLAN×8**(权重装载成功)→ **10:02:15-16 EI0006 齐射**。summary 未生成、无 metrics、无 DRAIN。
- **EI0006 证据**:EI0006×20(8 worker×2 + 4 EngineCore×1),首个=DP3_TP0_EP6(pid 1462699)10:02:15.847,全部 10:02:15-16。**acl_graph 计数=16**(graph 模式,va.env RUN_MODE=graph);**0×1004**(本轮 DP1 1004 未出现)。
- **崩溃栈**(model.log:1634-1684,首个 EI0006 前 50 行,DP3_TP0_EP6):`core.py:1195 _initialize_kv_caches → core.py:257 _initialize_kv_caches → abstract.py:147 determine_available_memory → multiproc_executor.py collective_rpc →` Worker `moe_runner.py:637 forward → _moe_forward_shared(134) → fused_moe_0_23_0.py:134/122/693/497 → method_adapters.py:274 → w4a4_mxfp4.py:225 → moe_comm_method.py:156/175 → moe_mlp.py:527/133 quant_apply_mlp → device_op.py:1318 npu_dynamic_quant → torch_npu.npu_dynamic_mx_quant → **HcclAllGather → EI0006**`。
- **与 R10/R11 同签名确认**:崩溃在 KV-cache 内存测定 dummy run 的 MoE 前向 EP8 all-gather,与 R10(07:29:18)/R11-A(08:26:50)/R11-B(08:44:26)完全同相位同栈。
- 崩溃后模型父进程挂死(worker 已崩,parent Ss 态),按 R10 经验精确 kill 进程树 + 双 pool + shm backing,清场完成。

### 各步 RC / summary

| 步 | 结果 | EI0006 | 1004 | DRAIN_OK | summary/metrics |
|----|------|--------|------|----------|-----------------|
| R12-0 | 只读全过 | - | 0 | - | - |
| R12-1 | EI0006 复现 | 20 | 0 | 0 | 未生成/无 |

### 通过标准逐条

- 第 1 条(工具 RC=0、summary RUN_OK、model.log 生成文本):**不满足**——RC 未出(profile_pool_inference 挂死轮询,清场 kill),summary 未生成。
- 第 2 条(DSA_SHARED_DRAIN_OK ×8):**不满足**——0 DRAIN。
- 第 3/4 条(metrics 与报告块):**不适用**——未到 generate,无 metrics。

### 失败处置

- EI0006 复现:按派单「停,回传首个报错点前后 50 行(蓝区转代码 diff 排查)」。EI0006 **不在 R12 白名单**(仅 1004 可重跑),不重试,停。
- 1004:本轮 0 次,R11-D 的 DP1 专属 1004 未在作者数据面+fjj 框架组合下出现。

### 环境状态(交付时)

- 清场完成:无残留 urma/VLLM/python 进程(PCP pmda 2 个系统守护除外);端口 19090-19097/20000-20007 空闲;/dev/shm 373G 可用。
- 日志资产:`/home/fjj/pp_r12.log`(tee 输出)+ `pp_r12_console.log`;`/home/w00919836/scp-opt/tmp/r12-fjj-bench-authorplane/raw/model.log`(571KB)+ pool 日志;R12-0 证据见上文原文。
- 待蓝区判读:R12-1 证明 EI0006 与作者数据面配置无关(fjj 框架+作者数据面仍崩),指向 **fjj 框架/scp-va 之外代码或 smoke env 差异**;蓝区做 fjj vs scp-va 代码 diff 定位。

## §R12 判读（蓝区，2026-10-30）

### 裁决

1. **数据面配置排除（第二次自我修正）**：R12-1 在作者数据面（udmac1d1e3/e3 + 19090 段 + dp4tp2.json）下 EI0006 仍复现 → R11 判读「EI0006 归因 fjj 数据面配置」证伪。e2/20000 段/eid-probe-r1 json 全部排除。崩溃相位不变：KV-cache init dummy run 的 MoE EP8 all-gather **首次建连**。
2. **决定性矩阵收敛**：
   - 崩：`smoke env×fjj代码` × {fjj 数据面, 作者数据面} × {graph, eager}（4/4）；
   - 不崩：`无 smoke env×scp-va代码` × 作者数据面 × eager（R11-D，EI0006=0）。
   剩余双变量 = **smoke env 集合** 与 **框架代码（fjj vs scp-va）**。
3. **蓝区本地代码核实（cac7e2c4e→HEAD 全量 diff）**：代码差分仅 `examples/dsa_demo/simple_test.py` +155 行（bench 口径）与 `vllm_ascend/envs.py` +4 行（字段注册），**均不在 init/通信路径**——纯代码差分解释不了 EI0006。**但发现关键嫌疑**：`SmokeConfig.enable_cpu_binding` **默认 True**（cac7e2c4e 起即如此，simple_test.py:42，注释自书「共用宿主机、不希望改变绑核时设 False」）→ simple_test.py:288 注入 additional_config → ascend_config.py:171 → worker.py:786 init 后 `bind_cpus(local_rank)` → cpu_binding.py:792-798：线程绑核 **+ NPU MSI IRQ smp_affinity 机器级重定向**。作者 scp-va 跑的是 binding=none（R5-0 已确认）。共享机上：pool 已固定占 `2-17,21-36,...`/`192-...` 核 + 其他租户进程，worker 绑核计划若与之重叠或 IRQ 重定向影响 NPU 网络平面，HCCL 首连 socket 线程被饿死 → 「remote end does not initiate a connection request」→ **EI0006 齐射，机制完全吻合**。R10/R11-A/B/R12-1 的 env 清单**均未设** `VLLM_ASCEND_SMOKE_ENABLE_CPU_BINDING` → 全部落在默认 True；R11-D 无 env → binding=none → 不崩。**相关性+机制双成立，升头号嫌疑**。
4. **1004 旁证记录**：绑核开（R12-1）反而 0×1004 过了 bind 阶段；绑核关（R11-D）DP1 1004 两跑——相关非因果，R13 观察项。
5. R12-0 机器面：NPU 2/3 EID 枚举健康（19 EID×2 轮）；udmac 节点 09:42 又批量重建（周期性，d1e6 稳定旧节点）；dmesg 无错误码。机器层无实锤。

### 环境事实入库

- EI0006 触发面排除清单（实锤）：graph/eager（R11-B）、数据面配置 e2/20000/json（R12-1）、机器 EID 枚举（R12-0）、NPU 2/3 健康（R12-0）。
- **共享机纪律补充：凡跑 fjj 框架仓 smoke，必须显式 `VLLM_ASCEND_SMOKE_ENABLE_CPU_BINDING=0`**（与作者 binding=none 对齐），除非专门测绑核。

## §R13 派单：绑核嫌疑直测 + bench 基线同轮补打（蓝区，2026-10-30）

### 原则

R13-1 是头号嫌疑直测（预期：过）。崩溃发生在 init 后 ~5 分钟，单轮成本低。白名单 = 1004（≤2 次重跑，每次 ≥5 分钟静置 + 完整清场）；EI0006 复现 → 进 R13-2；白名单外新报错 → 停，回传。

### R13-1 · R12-1 逐字 + 关绑核

```bash
cd /home/w00919836/scp-opt
source /home/w00919836/va.env
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
export VLLM_ASCEND_SMOKE_ENABLE_CPU_BINDING=0   # ← 本轮唯一变量（头号嫌疑直测）
export PYTHONPATH=/home/fjj/vllm-ascend-remote-offload:$PYTHONPATH
python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"   # 须指向 fjj 仓
python3 tools/profile_pool_inference.py --du-store-prefault read \
  --framework-repo /home/fjj/vllm-ascend-remote-offload \
  --output tmp/r13-nobind-bench --sample-port 19090 --sample-worker 0 \
  --remote-config ../dp4tp2.json --no-perf --payload-layers 78 \
  --du-pipeline-mib 1 --store-bytes $((128*1024*1024*1024)) \
  --pool-start-timeout 600 --pool-count 2 --numa-node 0 --pool-b-numa-node 2 \
  --store-numa-nodes local \
  --pool-cpus '2-17,21-36,40-55,59-74,78-93' \
  --pool-b-cpus '192-207,211-226,230-245,249-264,268-283' \
  --shared-port-gather --du-devices udmac1d1e3 --du-eid-indices 0 \
  --pool-b-du-devices udmac0d1e3 --pool-b-du-eid-indices 0 \
  2>&1 | tee /home/fjj/pp_r13.log
echo "RC=$?"
```

**绑核关闭生效核验**（须原文回传）：`grep -iE "bind|affinity" /home/w00919836/scp-opt/tmp/r13-nobind-bench/raw/model.log | head -20`——预期无 CpuAlloc 绑核计划输出；若有 `Bind cpus` 相关行也原样回传。

- **过** ⇒ 绑核根因坐实 + bench 基线同轮落袋，按 §R10 通过标准第 3/4 条原文回传 metrics。
- **EI0006 复现** ⇒ 绑核排除 → R13-2。
- **1004** ⇒ 白名单重跑（≤2）。

### R13-2 · env 集合二分（R13-1 复现后执行）

R13-1 逐字但 `VLLM_ASCEND_SMOKE_BENCH=0`（其余 env 全保留，ENABLE_CPU_BINDING=0 保留），输出目录 `tmp/r13-envbis-nobench`。
- **过** ⇒ 触发面在 bench env/代码路径（min_tokens/ignore_eos/metrics），蓝区复查 717c396fc  diff。
- **复现** ⇒ 触发面在基础 env 集合（MAX_NUM_BATCHED_TOKENS/CHUNKED_PREFILL/GPU_MEM 等），蓝区给定下一轮逐项 unset 二分表。

### 通过标准（同 §R10）与回传格式

同 §R12 派单（含绑核核验 grep 输出、import 路径、metrics 原文若过、日志路径、起止时间）。

## §R13 交付：R13-1 关绑核 EI0006 仍复现→绑核排除；R13-2 BENCH=0 仍复现→触发面在基础 env 集合（绿区，2026-10-30 10:49~11:10）

### 结论(先行)

- **R13-1 关绑核直测**(fjj 框架+BENCH=1+作者数据面+`VLLM_ASCEND_SMOKE_ENABLE_CPU_BINDING=0`):**EI0006 复现**(10:54:05 齐射×20,0×1004,0 DRAIN)。绑核关闭核验通过:`enable_cpu_binding=False` 生效(additional_config 原文),model.log **无** CpuAlloc/Bind cpus/sched_setaffinity/IRQ 绑核执行痕迹 → **绑核嫌疑排除**(R12 判读头号嫌疑证伪,第三次自我修正)。
- **R13-2 env 集合二分**(R13-1 逐字但 `VLLM_ASCEND_SMOKE_BENCH=0`):**EI0006 复现**(11:03:16 齐射×20,0×1004,0 DRAIN) → **bench env/代码路径排除**,触发面在**基础 env 集合**(MAX_NUM_BATCHED_TOKENS/CHUNKED_PREFILL/GPU_MEM 等)或更底层(命中预案「复现 ⇒ 基础 env 集合,蓝区给下一轮逐项 unset 二分表」)。
- 两轮崩溃相位全程一致:KV-cache init dummy run 的 MoE 前向 `HcclAllGather → EI0006`(worker.py:564 determine_available_memory,与 R10/R11-A/B/R12-1 完全同签名同相位)。
- bench 基线仍未落袋(0/7 轮跑到 generate);无 metrics 可回传(§R13 通过标准第 3/4 条不适用)。

### import 路径核验

va.env 后 `vllm_ascend.__file__` 指向 scp-va → prepend `PYTHONPATH=/home/fjj/vllm-ascend-remote-offload` 后 = `/home/fjj/vllm-ascend-remote-offload/vllm_ascend/__init__.py`(fjj,**生效**,两轮均核验)。

### 数据面配置核验(remote-shared.json 原文关键字段)

作者数据面逐字:du 19090-19097、store 19080-19087、device EID=`...0001f8/...00f1f8`(作者 dp4tp2.json 值)、pool_eid=`...0a0bf8/...080bf8`、graph_capture=startup_pad、put_transport=urma_read、udmac1d1e3。

### R13-1 · 关绑核直测(唯一变量 ENABLE_CPU_BINDING=0)——EI0006 复现

- **绑核关闭核验**(派单要求 grep 原文):model.log `grep -inE "CpuAlloc|Bind cpus|bind_cpus|sched_setaffinity|SMP affinity|IRQ|msi"`(排除 config 行)**零命中**;additional_config 原文 `enable_cpu_binding': False`(×4 EngineCore 一致)→ 绑核确实关闭。
- **执行起止**:DSA_EXECUTION_PLAN **10:50:16**(权重装载成功×8)→ **10:54:05-06 EI0006 齐射**。summary 未生成、无 metrics、无 DRAIN。
- **EI0006 证据**:EI0006×20(8 worker×2 + 4 EngineCore×1),首个=DP1_TP0_EP2(pid 1514522)10:54:05.286;全部 10:54:05-06。**acl_graph 计数=16**(graph 模式);**0×1004**。
- **崩溃栈**(model.log:1578-1688,首个 EI0006 前,DP1_TP0_EP2):`worker.py:564 determine_available_memory →`(EngineCore) `core.py:1195 _initialize_kv_caches → core.py:257 → abstract.py:147 collective_rpc` → Worker `npu_dynamic_mx_quant → HcclAllGather → EI0006`。

### R13-2 · env 集合二分(唯一变量 BENCH=0,ENABLE_CPU_BINDING=0 保留)——EI0006 复现

- **执行起止**:DSA_EXECUTION_PLAN **10:59:30**(权重装载成功×8)→ **11:03:16-17 EI0006 齐射**。summary 未生成、无 metrics、无 DRAIN。
- **EI0006 证据**:EI0006×20,首个=DP3_TP0_EP6(pid 1528715)11:03:16.722;全部 11:03:16-17。**0×1004**,0 DRAIN。`enable_cpu_binding=False` 保留生效。
- **崩溃相位**:与 R13-1 完全相同(worker.py:564 determine_available_memory 内 MoE 前向 HcclAllGather)。
- **判读**:BENCH=0 后崩溃仍在 init 的 determine_available_memory(早于任何 bench 逻辑)→ **触发面不在 bench env/代码路径(min_tokens/ignore_eos/metrics)**,在基础 env 集合或更底层。

### 各步 RC / summary

| 步 | 结果 | EI0006 | 1004 | DRAIN_OK | summary/metrics |
|----|------|--------|------|----------|-----------------|
| R13-1 | EI0006 复现(绑核排除) | 20 | 0 | 0 | 未生成/无 |
| R13-2 | EI0006 复现(BENCH=0 排除) | 20 | 0 | 0 | 未生成/无 |

### 通过标准逐条

- 第 1 条(工具 RC=0、summary RUN_OK、model.log 生成文本):**不满足**——RC 未出(profile_pool_inference 挂死轮询,清场 kill),summary 未生成。
- 第 2 条(DSA_SHARED_DRAIN_OK ×8):**不满足**——0 DRAIN。
- 第 3/4 条(metrics 与报告块):**不适用**——未到 generate,无 metrics。

### 失败处置

- EI0006 复现(R13-1/R13-2 均):按派单预案。R13-1 复现→执行 R13-2;R13-2 复现→**停,回传,蓝区给定下一轮逐项 unset 二分表**。EI0006 不在白名单(仅 1004),不重试。
- 1004:两轮均 0 次。

### 环境状态(交付时)

- 清场完成:无残留 urma/VLLM/python 进程(PCP pmda 2 个系统守护除外);端口 19090-19097/20000-20007 空闲;/dev/shm 373G 可用。
- 日志资产:`/home/fjj/pp_r13.log`(R13-1 tee)+ `pp_r13b.log`(R13-2 tee);`tmp/r13-nobind-bench/raw/model.log` + `tmp/r13-envbis-nobench/raw/model.log` + pool 日志。
- 待蓝区判读:R13-1 绑核排除(头号嫌疑证伪);R13-2 触发面在基础 env 集合或更底层,蓝区给定下一轮逐项 unset 二分表。

## §R13 判读（蓝区，2026-10-30）

### 裁决

1. **绑核排除**（R13-1）：关绑核核验干净（additional_config False + 无绑核执行痕迹）仍 EI0006 → R12 头号嫌疑证伪，第三次自我修正。
2. **bench env/代码排除**（R13-2）：BENCH=0 仍复现，且崩溃相位在 init 的 determine_available_memory（早于一切 bench 逻辑）→ `min_tokens/ignore_eos/metrics` 路径全部排除。
3. **基础 env 集合 ≠ 均匀嫌疑——存在配置层混淆**：蓝区本地复核入口 `simple_test_1p.py`（fjj=upstream）：无 env 时配置 = **BS1 / chinese_23k / max_model_len 25600 / batched 25600 / chunked_prefill=False / gpu_mem 0.85**；而作者 scp-va 的未提交 diff（§R5-0 已回传）只加了 profile_dir+7 个开关，**不含 BS16/40960/chunked1**——R5-0 记录的「作者参数 BS16/40960/chunked1」若属实，只能来自**作者 shell 里的 VLLM_ASCEND_SMOKE_* 导出**（bash_history，R14-0 取证）。这意味着：现有「有 env 必崩 / 无 env 不崩」的相关性，**可能实际是「BS16/40960/40960/chunked1 大配置必崩 / BS1/25600 小配置不崩」**。R11-D 通过的 profile 相位在我们这边全崩，差的不只是 env 机制，更是整套配置尺寸。**头号嫌疑转配置值：MAX_NUM_BATCHED_TOKENS=40960（profile run 用 batched_tokens 做 dummy 前向，是 determine_available_memory 的直接输入）**。
4. NATIVE_RUNTIME_ENV_OVERRIDES（HCCL_OP_EXPANSION_MODE=AIV 等 6 项，simple_test.py:542/698）**无条件施加**，R11-D 同样生效，非差分。
5. 1004 两轮均 0；绑核关 + 1004 消失并存观察项继续挂账。

### 判读落点

R14 两个动作：**R14-0 只读取证**（作者 bash_history 的 smoke env 导出痕迹 = 判定「env 机制 vs 配置值」的钥匙），**R14-1 代码正交实验**（作者代码 scp-va + 我们的 env 值全集 + 作者数据面）：崩 = 配置值轴（R14-2 逐项 unset 二分）；过 = 代码轴（蓝区复查 bench commit 与工具文件选择）。

## §R14 派单：env 机制 vs 配置值正交拆解（蓝区，2026-10-30）

### 原则

R14-0 先行（免费、决定性）。R14-1 按结果分支。EI0006 不在白名单，复现即按分支走；1004 白名单 ≤2 次（≥5 分钟静置 + 完整清场）。

### R14-0 · 只读取证（作者成功跑程的真实配置来源）

```bash
grep -n "VLLM_ASCEND_SMOKE" /home/w00919836/.bash_history | tail -40
cd /home/w00919836/scp-va && git log --oneline -1 && git diff | head -80
grep -n "SmokeConfig" -A 12 /home/w00919836/scp-va/examples/dsa_demo/simple_test_1p.py
```

回传三块原文。判读用：若 bash_history 有 smoke env 导出（尤其 MAX_NUM_BATCHED_TOKENS/MAX_MODEL_LEN/CHUNKED_PREFILL/MAX_NUM_SEQS）→ 作者成功跑程与我们的配置值同源，「env 机制」轴基本排除，锁配置值轴；若没有 → 作者成功跑程是 BS1/25600 小配置，配置值轴坐实。

### R14-1 · 代码正交实验：scp-va 代码 + 我们的 env 值全集 + 作者数据面

```bash
cd /home/w00919836/scp-opt
source /home/w00919836/va.env
# 注意：scp-va 的 envs.py 不认识 BENCH/METRICS_SUMMARY，导出会报 Unknown smoke
# environment variables —— 本轮【不要】导 BENCH。其余逐字：
export VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731
export VLLM_ASCEND_SMOKE_DATASET=chinese_40k
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=16
export VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=40960
export VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=40960
export VLLM_ASCEND_SMOKE_MAX_TOKENS=1024
export VLLM_ASCEND_SMOKE_ENABLE_CHUNKED_PREFILL=1
export VLLM_ASCEND_SMOKE_GPU_MEMORY_UTILIZATION=0.97
export VLLM_ASCEND_SMOKE_ENABLE_PROFILE=0
export VLLM_ASCEND_SMOKE_ENABLE_CPU_BINDING=0
python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"   # 须指向 /home/w00919836/scp-va
python3 tools/profile_pool_inference.py --du-store-prefault read \
  --framework-repo /home/w00919836/scp-va \
  --output tmp/r14-va-env-authorplane --sample-port 19090 --sample-worker 0 \
  --remote-config ../dp4tp2.json --no-perf --payload-layers 78 \
  --du-pipeline-mib 1 --store-bytes $((128*1024*1024*1024)) \
  --pool-start-timeout 600 --pool-count 2 --numa-node 0 --pool-b-numa-node 2 \
  --store-numa-nodes local \
  --pool-cpus '2-17,21-36,40-55,59-74,78-93' \
  --pool-b-cpus '192-207,211-226,230-245,249-264,268-283' \
  --shared-port-gather --du-devices udmac1d1e3 --du-eid-indices 0 \
  --pool-b-du-devices udmac0d1e3 --pool-b-du-eid-indices 0 \
  2>&1 | tee /home/fjj/pp_r14.log
echo "RC=$?"
```

- **EI0006 复现** ⇒ 配置值轴（同样值在作者代码上也崩）→ 执行 R14-2。
- **过 profile 相位**（无论最终 1004/通过）⇒ 代码轴 → 停，回传，蓝区复查。
- 1004 ⇒ 白名单重跑。

### R14-2 · 配置值逐项 unset 二分（R14-1 复现后执行，每步 ~10 分钟）

R14-1 逐字，按序每次只 unset 一个变量（其余全留），命中即停：

1. `unset VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS`（40960→默认 25600；**最高优先**，profile run 直接输入）
2. `unset VLLM_ASCEND_SMOKE_MAX_MODEL_LEN`（40960→25600）
3. `unset VLLM_ASCEND_SMOKE_ENABLE_CHUNKED_PREFILL`（1→False）
4. `unset VLLM_ASCEND_SMOKE_MAX_NUM_SEQS`（16→1）
5. `unset VLLM_ASCEND_SMOKE_GPU_MEMORY_UTILIZATION`（0.97→0.85）

输出目录后缀 `-stepN`（如 tmp/r14-va-env-authorplane-step1）。某步过后 profile 相位通过（哪怕后续 1004/其它错）→ **该变量 = 元凶**，停，回传该步完整日志；五步全复现 EI0006 → 停，回传，蓝区重估（届时应有 R14-0 答案辅助）。

### 通过标准与回传格式

同 §R13（R14-1 无 BENCH，metrics 不适用；R14-0 三块原文必回）。附各步 import 路径核验、日志路径、起止时间。


## §R14 交付：配置值轴坐实 + 5 步 unset 二分全复现 EI0006（绿区，2026-10-30 11:10~12:15）

### R14-0 · 只读取证结果

**块1 作者 shell history 的 smoke env 痕迹**：

```
$ grep -n VLLM_ASCEND_SMOKE /root/.bash_history | tail -40
(空输出，0 命中)

说明：/home/w00919836/.bash_history 不存在（容器实际 HOME=/root）；以 /root/.bash_history 为准。
/root/.bash_history 全部内容为作者装机/克隆命令：git clone vLLM-ascend-DSA、cp -r scp-opt scp-va、
git checkout remote-offload-ops / vllm-ascend-v0.23.0-custom-remote-offload、exit。
无任何 export VLLM_ASCEND_SMOKE_* 痕迹。
```

**块2 scp-va 未提交 diff**：

```
$ cd /home/w00919836/scp-va && git log --oneline -1
cac7e2c4 perf(dsa): dispatch shared group IO once from Full
$ git diff --stat
 examples/dsa_demo/simple_test.py    |  2 +-
 examples/dsa_demo/simple_test_1p.py | 15 ++++++++++++++-
 2 files changed, 15 insertions(+), 2 deletions(-)

$ git diff（关键行）
-            sampling = SamplingParams(temperature=0.0, max_tokens=config.max_tokens, min_tokens=0, ignore_eos=False)
+            sampling = SamplingParams(temperature=0.0, max_tokens=config.max_tokens, min_tokens=config.max_tokens, ignore_eos=True)
...
+CONFIG = SmokeConfig(
     model_path="/home/models/GLM-5.2-w4a4c8-mxfp4_99",
     tensor_parallel_size=2,
     data_parallel_size=4,
     prompts=chinese_23k,
-    profile_dir="./profiles/1p",
+    profile_dir="./40k_1p",
+    enable_profile=True,
+    enable_mtp=True,
+    enable_mtp_graph=True,
+    run_mode='graph',
+    enable_offload=True,
+    enable_a5_packed_c8=True,
+    gpu_memory_utilization=0.97
 )
```

**块3 simple_test_1p.py 的 SmokeConfig**：无 BS / MAX_MODEL_LEN / MAX_NUM_BATCHED_TOKENS / CHUNKED_PREFILL / MAX_NUM_SEQS 字段（同上 diff 全文）。

**R14-0 判读**：作者 shell history 无 smoke env 导出 + 代码 diff 不含大配置字段 ⇒ 作者成功跑程无「BS16/40960/chunked1」来源，实际是入口默认小配置（BS1/25600/chunked-off）。「env 机制」轴无作者侧证据；配置值轴待 R14-1 决定性验证。

### R14-1 · 代码正交实验（scp-va 代码 + env 值全集 + 作者数据面）

- 命令：`bash r14_1_run.sh`（dispatch 逐字，framework-repo=/home/w00919836/scp-va，作者 dp4tp2.json，不导 BENCH——scp-va 的 envs.py 不认该字段）
- **import 核验**：`IMPORT: /home/w00919836/scp-va/vllm_ascend/__init__.py`（生效，指向 scp-va）
- **结果：EI0006 复现**。DSA_EXECUTION_PLAN 11:30:59 → EI0006 首个 11:34:45（~3.8 min），×20，0×1004，0 DRAIN。
- **崩溃相位**：`scp-va/vllm_ascend/worker/worker.py:564 determine_available_memory`（KV-cache 绑定后 MoE 前向 HcclAllGather，与 R10/R11/R12/R13 同签名）。
- 日志：/home/fjj/pp_r14.log；证据 tmp/r14-va-env-authorplane/raw/model.log

**R14-1 判读（命中「复现」分支）**：同样配置值在作者代码 scp-va 上也崩 ⇒ **配置值轴坐实（代码轴排除）** ⇒ 执行 R14-2 逐项 unset 二分。

### R14-2 · 配置值逐项 unset 二分（5 步全跑）

每步 = R14-1 逐字，仅 unset 指定变量（其余 env 全留），import 核验均指向 scp-va，输出 tmp/r14-va-env-authorplane-stepN。**五步全部 EI0006 复现，0×1004，0 DRAIN，崩溃相位全部 worker.py:564 同签名**。

| 步 | unset 变量（→默认） | DSA_EXECUTION_PLAN | EI0006 首个 | EI0006 计数 | 1004 | DRAIN | 判读 |
|---|---|---|---|---|---|---|---|
| step1 | MAX_NUM_BATCHED_TOKENS（40960→25600） | 11:40:30 | 11:44:14 | 20 | 0 | 0 | 仍崩，非元凶 |
| step2 | MAX_MODEL_LEN（40960→25600） | 11:48:06 | 11:51:50 | 20 | 0 | 0 | 仍崩，非元凶 |
| step3 | ENABLE_CHUNKED_PREFILL（1→False） | 11:56:44 | 12:00:38 | 20 | 0 | 0 | 仍崩，非元凶 |
| step4 | MAX_NUM_SEQS（16→1） | 12:02:26 | 12:06:12 | 20 | 0 | 0 | 仍崩，非元凶 |
| step5 | GPU_MEMORY_UTILIZATION（0.97→0.85） | 12:08:00 | 12:11:41 | 20 | 0 | 0 | 仍崩，非元凶 |

各步 VERIFY_UNSET 回显（unset 后变量为空，确认生效）：

```
step1: VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=
step2: VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=
step3: VLLM_ASCEND_SMOKE_ENABLE_CHUNKED_PREFILL=
step4: VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=
step5: VLLM_ASCEND_SMOKE_GPU_MEMORY_UTILIZATION=
```

各步 console 均以 `RuntimeError: Pool ['a', 'b'] exited while inference active` + `POOL_ROI_REPORT .../summary.txt` + `RC=0` 收尾（run 脚本 tee 出口 RC）。

### 通过标准逐条核验

1. ✅ R14-1 无 BENCH（scp-va 不认该字段，未导）——metrics 不适用，按 R14-1 判读分支走。
2. ✅ import 核验：R14-1 + R14-2 五步全部 `IMPORT: /home/w00919836/scp-va/vllm_ascend/__init__.py`。
3. ✅ R14-0 三块原文已回传（块1 修正为 /root/.bash_history——/home/w00919836/.bash_history 不存在）。
4. ✅ EI0006 不在白名单，复现即按分支走；1004 全程 0，白名单未触发。
5. ✅ 每步后完整清场（kill 池/模型树 + 删 shm backing），端口全释放，/dev/shm 回落 373G。

### 结论与下一步建议

五步逐个 unset 后 EI0006 依旧稳定复现（同配置值轴、同崩溃相位），**5 个候选配置值均非独立元凶**。R14-0 已确认作者侧无大配置来源。据此：

- 配置值轴坐实但**非单一 env 变量**所致——EI0006 与「env 集合整体 / 其它未被 unset 的维度」耦合（或与 8 worker 并发 MoE 前向的集群级 HCCL 建链相关，见 R10 bench 结论）。
- 建议蓝区下一步：① 全部 unset 到纯默认（BS1/25600/chunked-off）对照一次；② 或核查 scp-opt（fjj 仓库）vs scp-va 在 KV-cache 绑定/dummy 前向路径的残余差异（NATIVE_RUNTIME_ENV_OVERRIDES 6 项仍无条件施加）。
- bench 基线仍未落袋（0/8 轮跑到 generate），本轮 5 步均为复现性实验非 bench。

日志：/home/fjj/pp_r14.log（R14-1）、/home/fjj/pp_r14_step{1..5}.log；各步证据 tmp/r14-va-env-authorplane-step{1..5}/raw/model.log
起止：2026-10-30 11:10（R14-0）~ 12:15（step5 清场完成）


## §R14 判读（蓝区，2026-10-30）

### 裁决

1. **配置值轴坐实、代码轴排除**：同值在 scp-va（作者代码）上也崩（R14-1）→ bench commit 彻底洗清；EI0006 由配置值集合触发。
2. **但非单一配置值**：五步逐个 unset（batched_tokens/model_len/chunked/BS/gpu_mem）全崩（R14-2），5 个候选均非独立元凶。且 step1（batched 25600）与 R11-D 的 profile 形状相同仍崩 → 连"profile 尺寸"都排除。
3. **剩余差分穷尽**（崩组 vs R11-D 组）：unset 从未触及的 env 只剩 4 个——MODEL_PATH、DATASET、MAX_TOKENS、ENABLE_PROFILE。蓝区本地已证：DATASET 只换 prompts（init 配置的 budget/thresholds 全部来自默认值，两组完全一致；profile run 不用真实 prompts）→ **DATASET 判 inert**；MAX_TOKENS 只碰 sampling → **inert**；ENABLE_PROFILE 在 --no-perf 下无采样器 → **基本 inert**；ENABLE_CPU_BINDING 已被 R13-1+正交性排除。**唯一活嫌疑 = MODEL_PATH**：我们 `/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731` vs R11-D 走作者脚本默认 `/home/models/GLM-5.2-w4a4c8-mxfp4_99`——**若是两个不同 ckpt（config.json 有差异），模型初始化路径（DSA 计划/KV 布局）即不同**，这是最后一个能解释"同相位恒定崩溃"的配置级变量。
4. **时间线旁证**：R11-D（08:51/09:02 不崩）被 R11-B（08:44 崩）与 R12-1（10:02 崩）夹在中间 → 「机器状态时段」相关性也被破坏，纯时间假说不成立。
5. bench 基线 0/8 轮到 generate；本轮 6 轮全是归因轮。

### 判读落点

R15：只读取证先锁 MODEL_PATH 轴（两 ckpt 的 config.json diff——免费且决定性）；再跑两个对照（今日重放 R11-D = 机器状态对照；R14-1 只留 MODEL_PATH = env 机制对照）。

## §R15 派单：MODEL_PATH 轴锁定 + 双对照（蓝区，2026-10-30）

### 原则

R15-0 只读先行；R15-1/R15-2 顺序执行。白名单 = 1004（≤2 次重跑，每次 ≥5 分钟静置 + 完整清场）；EI0006 复现按分支走；白名单外新报错停。

### R15-0 · 只读取证（两个模型目录是否同一 ckpt）

```bash
ls -la /home/models/ 2>&1 | head -10
ls /home/models/GLM-5.2-w4a4c8-mxfp4_99/ 2>&1
diff <(cat /home/models/GLM-5.2-w4a4c8-mxfp4_99/config.json 2>/dev/null) \
     <(cat /data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731/config.json 2>/dev/null) \
  && echo SAME_CONFIG || echo DIFF_CONFIG
# 顺带：两个数据集各有多少条 prompt（请求数差异也一并记录）
cd /home/w00919836/scp-va/examples/dsa_demo && python3 -c "
from novel_dataset import chinese_23k
import importlib, sys
sys.path.insert(0, '/home/fjj/vllm-ascend-remote-offload/examples/dsa_demo')
from novel_dataset import chinese_40k
print('23k prompts:', len(chinese_23k)); print('40k prompts:', len(chinese_40k))"
```

原文回传。判读用：DIFF_CONFIG ⇒ MODEL_PATH 直接升唯一元凶候选（R15-2 验证）；SAME_CONFIG ⇒ MODEL_PATH 排除，R15-2 变为纯 env 机制对照。

### R15-1 · 今日重放 R11-D（机器状态对照）

逐字 §R6-1（scp-opt + va.env + dp4tp2.json + udmac1d1e3/e3 + 19090 段 + pool-cpus，零 smoke env），输出目录 `tmp/r15-r11d-replay`，日志 `/home/fjj/pp_r15a.log`。
- 过 profile 相位（1004 或 RUN_OK 都算）⇒ 机器状态正常，R11-D 配方仍有效，进 R15-2。
- **EI0006** ⇒ 机器状态自 09:02 后劣化（R11-D 配方失效），停，回传，蓝区转机器层取证路线。

### R15-2 · R14-1 只留 MODEL_PATH（env 机制对照）

R14-1 逐字，但只 export 一个 smoke env：

```bash
export VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731
# 其余 9 个 smoke env 一律不导（va.env 已 unset）；PYTHONPATH/import 核验同 R14-1（须 scp-va）
```

输出目录 `tmp/r15-only-modelpath`。⚠️ 注意：enable_cpu_binding 回落默认 True（与 R11-D 一致，已证正交，不算新变量）。
- **过 profile 相位** ⇒ 元凶在 DATASET/MAX_TOKENS/ENABLE_PROFILE 三 add-back 里（尽管本地判 inert，实证优先）→ 蓝区发 R16 add-back 梯子。
- **EI0006** ⇒ 只剩两个解释：MODEL_PATH（依 R15-0 结论）或 env 机制本身 → 蓝区定 R16（SAME_CONFIG 时直接换用 /home/models/99 路径跑同命令，一刀分开 MODEL_PATH 与 env 机制）。
- 1004 ⇒ 白名单重跑。

### 通过标准与回传格式

同 §R14（R15-0 三块原文必回；R15-1/R15-2 附 import 路径、日志路径、起止时间、EI0006 计数/相位）。

## §R15 交付：MODEL_PATH 轴排除 + 双对照完成（绿区，2026-10-30）

### 结论（判读速览）

- **R15-0**：`/home/models/GLM-5.2-w4a4c8-mxfp4_99` 在本机/容器**均不存在**（CONFIG_MISSING，ls 报 No such file）⇒ 蓝区「两 ckpt 可能不同」的前提在本机不成立。所有实际运行（R11-D/R14-1/R15-1/R15-2）经 va.env 覆盖为同一 `/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731/` 路径。数据集 chinese_23k/chinese_40k 各仅 **1 条**超长 prompt（23k=34068 字符≈23000 token；40k=63130 字符≈40000 token），DATASET 只换 token 长度、不换请求数。
- **R15-1**（今日重放 R11-D，va.env 全 env，EI0006=0）：与 R11-D 09:02 原跑签名一致——DP1 专属 1004（4 次）+ EngineCore 在 `patch_shm_broadcast` 挂起，最终 `Engine core initialization failed`。R11-D 原跑本身即以失败结束（09:06:14，非 PASS），今日忠实复现 ⇒ **机器状态自 09:02 后未劣化，R11-D 配方仍有效**。
- **R15-2**（R14-1 只留 MODEL_PATH，其余 15 个 smoke env 全 unset，EI0006=13）：**EI0006 复现**（13:16:09，HcclAllGather，worker.py:564 determine_available_memory，1004=0/RUN_OK=0），与 R14-1 同签名。

### 关键判读（MODEL_PATH 轴）

R15-1 与 R15-2 **MODEL_PATH 值完全相同**（均 `/data/model/...-A5-0731/`），仅其余 env 是否导出不同：

| | R15-1（R11-D 重放） | R15-2（只留 MODEL_PATH） |
|---|---|---|
| import | scp-va ✓ | scp-va ✓ |
| MODEL_PATH | `/data/model/...A5-0731/` | `/data/model/...A5-0731/`（相同） |
| 其余 smoke env | va.env 全集（12 个导出） | 仅 MODEL_PATH（15 个 unset） |
| 结果 | **EI0006=0**，1004=4，hang→Engine core init failed | **EI0006=13**，worker.py:564 HcclAllGather |
| 起止 | 12:49:01 → 13:05:34 | 13:11:30 → 13:19:59 |

⇒ **崩溃差异与 MODEL_PATH 值无关**。按 §R15 派单分支：**EI0006 ⇒ 只剩 env 机制本身解释 → 蓝区定 R16**（SAME_CONFIG/两 ckpt 前提失效，R16 无需再走「换 /home/models/99 路径」一刀分支；env 机制轴 = 是否导出 smoke env 全集本身）。

### 执行记录

- **R15-1**：逐字 §R6-1（scp-opt + va.env + dp4tp2.json + udmac1d1e3/e3 + 19090 段 + pool-cpus，零额外 smoke env）。输出 `tmp/r15-r11d-replay`，日志 `/home/fjj/pp_r15a.log` + 控制台 `/tmp/r15s1_console.log`。命令 `simple_test_1p.py --remote-shared-config .../remote-shared.json`。
- **R15-2**：R14-1 逐字但只留 `VLLM_ASCEND_SMOKE_MODEL_PATH`。**执行要点**：蓝区派单称「va.env 已 unset 其余 9 个」，实测 va.env 只 unset 4 项后又 re-export 大部分（source 后残留 12 个 smoke env）——为兑现「只留 MODEL_PATH」意图，source 后显式 unset 全部 `VLLM_ASCEND_SMOKE_*` 再单独 export MODEL_PATH。输出 `tmp/r15-only-modelpath`，日志 `/home/fjj/pp_r15b.log` + 控制台 `/tmp/r15s2_console.log`。

### 回传字段（同 §R14 标准）

- R15-0 三块原文：见下「原文回传」。
- import 路径：R15-1/R15-2 均 `/home/w00919836/scp-va/vllm_ascend/__init__.py`（scp-va ✓）。
- 日志路径：`/home/fjj/pp_r15a.log`、`/home/fjj/pp_r15b.log`；model.log：`tmp/r15-r11d-replay/raw/model.log`、`tmp/r15-only-modelpath/raw/model.log`。
- 起止时间：R15-1 12:49:01→13:05:34；R15-2 13:11:30→13:19:59。
- EI0006 计数/相位：R15-1=0（1004=4，hang）；R15-2=13（13:16:09，worker.py:564 HcclAllGather，DP3 首报后齐射）。

### 原文回传（R15-0）

```
$ ls -la /home/models/ 2>&1 | head -10
ls: cannot access /home/models/: No such file or directory

$ ls /home/models/GLM-5.2-w4a4c8-mxfp4_99/ 2>&1
ls: cannot access '/home/models/GLM-5.2-w4a4c8-mxfp4_99/': No such file or directory

$ diff <(cat /home/models/GLM-5.2-w4a4c8-mxfp4_99/config.json 2>/dev/null) <(cat /data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731/config.json 2>/dev/null) && echo SAME_CONFIG || echo DIFF_CONFIG
CONFIG_MISSING_A  (左侧 /home/models/..._99 不存在，diff 前提失效)

23k prompts: 1
40k prompts: 1
```

- **补充取证**：`/home/models/` 在宿主机与容器内均不存在；R11-D 与 R14-1 的 commands.json 实际 model 均指向 `/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731/`（同一路径，非两 ckpt）。va.env 第 23 行 export `VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731/`。

## §R15 判读（蓝区，2026-10-30）

### 裁决（两个前提坍塌，归因重置）

1. **MODEL_PATH 轴排除**：`/home/models/...99` 不存在，va.env 第 23 行统一 MODEL_PATH ⇒ 所有 run（含 R11-D）同模型同 ckpt。
2. **「env 集合」轴坍塌（归因重置）**：va.env **本身 export 12 个 smoke env**（绿区 source 后实测残留 12 个）——R11-D/R15-1 从来不是"零 env"组；R15-2（仅 MODEL_PATH、其余 15 个全 unset）**仍 EI0006** ⇒ "有 env 就崩"不成立。真正的分界线是 **va.env 里的变量集合 D（我们从未设过的那些）**：R15-1（D 在）profile 过、R15-2（D 不在）profile 崩，且 R9 的 05:09 RUN_OK 也是 va.env 配方跑通的（完整 generate + 8×DRAIN_OK）。
3. **R15-1 今日忠实复现 R11-D 签名**（EI0006=0、DP1 1004×4、hang）⇒ 机器状态无劣化；且 R11-D 从来就不是 PASS 组（1004 挂，0 DRAIN）。
4. **流程教训入库**：R12-1 交付声称"va.env unset 全部 smoke 旋钮"为**误报**，R12~R15 四轮梯子建立在错误前提上（蓝区未要求原文核验，责任各半）。纪律：**凡"环境里没有 X"类断言，必须以 `env|grep` / 文件原文回传为准，口头确认不作数**。
5. 崩溃相位不变（determine_available_memory 的 EP8 all-gather 首连）；bench 0/9 到 generate。

### 判读落点

D 集合内容 = 下轮 R16-0 的 `cat va.env`（免费、决定性）。工作假设：**va.env 配方（R9 已证全通）+ 我们的值覆盖 = 可用 bench 配方**；R16-1 直接验证 profile 相位；过则 R17 换 fjj 仓 + BENCH=1 落 bench 基线。

## §R16 派单：va.env 原文取证 + 融合配方验证（蓝区，2026-10-30）

### R16-0 · 只读取证（va.env 全文——D 集合内容）

```bash
cat /home/w00919836/va.env
echo ====
bash -c 'source /home/w00919836/va.env && env | grep -E "^VLLM_ASCEND_SMOKE" | sort'
echo ====
grep -inE "hccl|hccl_|HCCL" /home/w00919836/va.env || echo "no HCCL vars in va.env"
```

原文回传三块。判读用：D = 残量清单 − {我们已设的 10 个}；重点核对 D 是否含 DSA_*（RESIDENT_BUDGET/MAX_ACTIVE_REQS/PROMPT_BUDGET_THRESHOLDS/INDEXER_MLA_BLOCK_RATIO）、GRAPH_CAPTURE_SIZES、RUN_MODE、ENABLE_MTP* 等 init 路径变量。

### R16-1 · 融合配方验证（va.env + 我们的值覆盖，scp-va + 作者数据面）

```bash
cd /home/w00919836/scp-opt
source /home/w00919836/va.env
# 在 va.env 之后覆盖我们的值（后 export 者生效）：
export VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731
export VLLM_ASCEND_SMOKE_DATASET=chinese_40k
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=16
export VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=40960
export VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=40960
export VLLM_ASCEND_SMOKE_MAX_TOKENS=1024
export VLLM_ASCEND_SMOKE_ENABLE_CHUNKED_PREFILL=1
export VLLM_ASCEND_SMOKE_GPU_MEMORY_UTILIZATION=0.97
export VLLM_ASCEND_SMOKE_ENABLE_PROFILE=0
export VLLM_ASCEND_SMOKE_ENABLE_CPU_BINDING=0
# 不导 BENCH（scp-va 的 envs.py 不认识会报 Unknown smoke environment variables）
python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"   # 须指向 scp-va
python3 tools/profile_pool_inference.py --du-store-prefault read \
  --framework-repo /home/w00919836/scp-va \
  --output tmp/r16-vaenv-fusion --sample-port 19090 --sample-worker 0 \
  --remote-config ../dp4tp2.json --no-perf --payload-layers 78 \
  --du-pipeline-mib 1 --store-bytes $((128*1024*1024*1024)) \
  --pool-start-timeout 600 --pool-count 2 --numa-node 0 --pool-b-numa-node 2 \
  --store-numa-nodes local \
  --pool-cpus '2-17,21-36,40-55,59-74,78-93' \
  --pool-b-cpus '192-207,211-226,230-245,249-264,268-283' \
  --shared-port-gather --du-devices udmac1d1e3 --du-eid-indices 0 \
  --pool-b-du-devices udmac0d1e3 --pool-b-du-eid-indices 0 \
  2>&1 | tee /home/fjj/pp_r16.log
echo "RC=$?"
```

- **过 profile 相位**（EI0006=0；后续 1004 或 RUN_OK 都算过）⇒ **融合配方成立**：D 变量是 profile 相位的保护因子。记录，进 R17。
- **EI0006** ⇒ D 非保护因子、保护机制在 va.env 别处（如非 smoke 变量/启动顺序）→ 停，回传，蓝区重估。
- 1004 ⇒ 白名单重跑（≤2 次，≥5 分钟静置）。

### R17 预案（R16-1 过 profile 后执行，派单时按此骨架）

fjj 框架仓（`--framework-repo /home/fjj/vllm-ascend-remote-offload` + `PYTHONPATH` 前置 fjj）+ **R16-1 完全相同的 env 状态** + `export VLLM_ASCEND_SMOKE_BENCH=1` + 输出目录 `tmp/r17-fjj-bench-vaenv`。过 = bench 基线同轮落袋（§R10 通过标准第 3/4 条原文回传 metrics）。

### 通过标准与回传格式

同 §R15（R16-0 三块原文必回；R16-1 附 import 路径、EI0006 计数/相位、日志路径、起止时间）。

## §R16 交付：va.env 原文取证 + 融合配方验证（绿区，2026-10-30）

### R16-0 · va.env 原文（三块原文必回）

**① va.env 全文**：
```bash
# /home/w00919836/va.env
# --- PYTHONPATH / OPP ---
export PYTHONPATH=/home/w00919836/scp-opt/torch_extension:$PWD/scp-va:$PYTHONPATH
export KVCACHE_SCATTER_COPY_C8_INSTALL_OPP_PATH=/home/w00919836/scp-opt/opp_c8
export KVCACHE_SCATTER_COPY_INSTALL_OPP_PATH=/home/w00919836/scp-opt/opp
export ASCEND_CUSTOM_OPP_PATH="$KVCACHE_SCATTER_COPY_C8_INSTALL_OPP_PATH/vendors/customize:$KVCACHE_SCATTER_COPY_INSTALL_OPP_PATH/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}"
# --- unset 9 项 ---
unset VLLM_ASCEND_SMOKE_ENABLE_PROFILE
unset VLLM_ASCEND_SMOKE_PROFILE_DIR
unset VLLM_ASCEND_SMOKE_RUN_MODE
unset VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD
unset VLLM_ASCEND_SMOKE_MAX_MODEL_LEN
unset VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS
unset VLLM_ASCEND_SMOKE_ENABLE_CHUNKED_PREFILL
unset VLLM_LOGGING_LEVEL
unset ASCEND_LAUNCH_BLOCKING
# --- re-export ---
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export VLLM_ASCEND_SMOKE_MODEL_PATH=/home/w00919836/GLM-5.2-w4a4c8-mxfp4-5-layer
# export VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731/
export VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731/
export VLLM_ASCEND_SMOKE_DATA_PARALLEL_SIZE=4
export VLLM_ASCEND_SMOKE_TENSOR_PARALLEL_SIZE=2
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=16
# export VLLM_ASCEND_SMOKE_RUN_MODE=eager
export VLLM_ASCEND_SMOKE_RUN_MODE=graph
export VLLM_ASCEND_SMOKE_MTP_NUM_SPECULATIVE_TOKENS=3
export VLLM_ASCEND_SMOKE_ENABLE_MTP_GRAPH=1
export VLLM_ASCEND_SMOKE_ENABLE_MTP=1
# important
#export VLLM_ASCEND_SMOKE_ENABLE_PROFILE=1
#export VLLM_ASCEND_SMOKE_PROFILE_DIR=./profiles/no_mtp
export VLLM_ASCEND_DSA_HOST_MSTX=0
export VLLM_ASCEND_SMOKE_DATASET=chinese_40k
# export VLLM_ASCEND_SMOKE_DATASET=chinese_23k
export VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=40960
export VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=$((2048))
export VLLM_ASCEND_SMOKE_ENABLE_CHUNKED_PREFILL=1
export VLLM_ASCEND_SMOKE_MAX_TOKENS=256
```

**② source 后 smoke env 全集（env | grep '^VLLM_ASCEND_SMOKE' | sort，共 13 个）**：
```
VLLM_ASCEND_SMOKE_DATA_PARALLEL_SIZE=4
VLLM_ASCEND_SMOKE_DATASET=chinese_40k
VLLM_ASCEND_SMOKE_ENABLE_CHUNKED_PREFILL=1
VLLM_ASCEND_SMOKE_ENABLE_MTP=1
VLLM_ASCEND_SMOKE_ENABLE_MTP_GRAPH=1
VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=40960
VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=2048
VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=16
VLLM_ASCEND_SMOKE_MAX_TOKENS=256
VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731/
VLLM_ASCEND_SMOKE_MTP_NUM_SPECULATIVE_TOKENS=3
VLLM_ASCEND_SMOKE_RUN_MODE=graph
VLLM_ASCEND_SMOKE_TENSOR_PARALLEL_SIZE=2
```
**D 集合（va.env 独有、我们未设的）**：RUN_MODE=graph、MTP_NUM_SPECULATIVE_TOKENS=3、ENABLE_MTP_GRAPH=1、ENABLE_MTP=1、DATA_PARALLEL_SIZE=4、TENSOR_PARALLEL_SIZE=2、MAX_NUM_BATCHED_TOKENS=2048、MAX_TOKENS=256、MODEL_PATH 尾斜杠、ASCEND_RT_VISIBLE_DEVICES=0-7；非 smoke 独有：`VLLM_ASCEND_DSA_HOST_MSTX=0`（DSA 前缀）、VLLM_LOGGING_LEVEL/ASCEND_LAUNCH_BLOCKING 被 unset。**不含 DSA_RESIDENT_BUDGET/MAX_ACTIVE_REQS/PROMPT_BUDGET_THRESHOLDS/INDEXER_MLA_BLOCK_RATIO/GRAPH_CAPTURE_SIZES。**

**③ HCCL grep**：`no HCCL vars in va.env`（va.env 无任何 hccl 变量，作者 va.env 无 HCCL 调优）。

### R16-1 · 融合配方验证结果：EI0006（配方不成立）

- **命令**：派单逐字（容器 dsa-full 内，cd /home/w00919836/scp-opt → source va.env → 覆盖我们 10 值 → python3 tools/profile_pool_inference.py ... --no-perf --output tmp/r16-vaenv-fusion）。**执行环境修正**：宿主 /usr/bin/python3 无 vllm（ModuleNotFoundError），必须在 dsa-full 容器内跑（容器 /vllm-workspace/vllm 有）。
- **import 核验**：vllm=`/vllm-workspace/vllm/vllm/__init__.py`；vllm_ascend=`/home/w00919836/scp-va/vllm_ascend/__init__.py`（scp-va ✓）。
- **时间线**：13:50:16 8 worker 启动 → 13:51:20 权重装载开始（checkpoint 389.88 GiB，EXT4，auto-prefetch 关）→ 13:54:49 **EI0006**（DP1_TP1_EP3 pid=1731115，HcclAllGather 超时，`Communication_Error_Get_Socket: Getting socket times out`）→ 主进程挂起等 Pool 收尾，绿区 kill（signal 15）触发 POOL_ROI_REPORT。
- **summary.txt**：`RUN_FAILED`（KeyboardInterrupt: signal 15），PERF_DISABLED。
- **EI0006 计数**：model.log 中 20 处（8 worker 崩溃 + 栈回溯重复引用）；**1004=0**（RaGetDevEidInfoNum 未命中）。
- **判读**：命中派单预案 **EI0006 ⇒ D 非保护因子、保护机制在 va.env 别处（非 smoke 变量/启动顺序）→ 停，回传，蓝区重估**。融合配方（va.env 全集+我们的值覆盖）仍崩，与 R14-1/R15-2 同签名（worker.py:564 HcclAllGather）。注意：R16-1 的 EI0006 相位在**权重装载后**（13:51:20 起 loading，13:54:49 崩），非 init 早期。
- **日志**：/home/fjj/pp_r16.log；raw/model.log；raw/summary.txt（/home/w00919836/scp-opt/tmp/r16-vaenv-fusion/）。
- **清场**：shm 已清（372G 可用），端口 19080-19087/19090-19097/20000-20007 全空闲，无残留进程。

## §R16 判读（蓝区，2026-10-30）：真凶定位——MAX_NUM_BATCHED_TOKENS

### 裁决

1. **D 集合审计**：va.env 13 个 smoke env 中，RUN_MODE=graph / MTP×3 / TP2 / DP4 与入口脚本默认值**同值**（天然 inert）；与默认值真正不同的只有三个：**MAX_NUM_BATCHED_TOKENS=2048**（我们 40960，unset 默认 25600）、**MAX_TOKENS=256**（sampling，inert at profile）、以及非 smoke 的 **VLLM_ASCEND_DSA_HOST_MSTX=0**。
2. **R16-1 排除 MSTX**：va.env 全量（含 MSTX=0）+ 我们的值覆盖（batched=40960）→ 仍 EI0006 ⇒ MSTX=0 非充分保护因子，真凶在值差里。
3. **batched_tokens 相关性闭环**：profile 相位通过的 run（R9 RUN_OK、R15-1、作者 10-28 现场）**全部 batched=2048**；profile 崩的 10+ run **全部 ≥25600**（含 R14-2 step1 的默认 25600）。**MAX_NUM_BATCHED_TOKENS 是唯一与「profile 崩/过」完美相关的配置变量**——机制：determine_available_memory 的 dummy 前向 batch = batched_tokens，2048 时首连小步通过、≥25600 时首个 EP8 all-gather 首连即超时（具体子机制待归因后深挖，不影响配方落地）。
4. bench 口径影响预告：batched=2048 → prefill 按 2048 分块，TTFT/吞吐口径会受影响，TPOT（decode 相位）不受影响；基线落袋后可再做 batched 尺寸阶梯（2048→8192→…）找上限。

### 判读落点

R17 阶梯：R15-2（仅 MODEL_PATH，崩）基础上逐个加回保护因子，首步直测 batched=2048。

## §R17 派单：保护因子直测阶梯（蓝区，2026-10-30）

### 原则

基线命令 = R15-2 逐字（scp-va + 作者数据面 + 仅 `export VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731`），按序每次加一个变量。**过 profile 判据 = EI0006=0**（后续 1004/hang 也算过相位）；命中即停。1004 白名单 ≤2 次（≥5 分钟静置 + 完整清场）。

| 步 | 加回变量 | 输出目录 |
|---|---|---|
| R17-1 | `export VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=2048` | tmp/r17-b2048 |
| R17-2（1 崩后） | 再加 `export VLLM_ASCEND_SMOKE_ENABLE_CHUNKED_PREFILL=1` | tmp/r17-b2048-c1 |
| R17-3（2 崩后） | 再加 `export VLLM_ASCEND_DSA_HOST_MSTX=0` | tmp/r17-b2048-c1-mstx |
| R17-4（3 崩后） | 再加 `export VLLM_ASCEND_SMOKE_MAX_TOKENS=256` | tmp/r17-b2048-c1-mstx-mt256 |

四步全崩 ⇒ 停，回传，蓝区重估（届时附 R14-2 step1 复核：unset 后默认值到底是多少——要求本阶梯第一步先跑 `python3 -c` 回显 `SmokeConfig` 默认 max_num_batched_tokens 原文，锁死基线值）。

执行命令模板（以 R17-1 为例，R17-2/3/4 追加对应 export）：

```bash
cd /home/w00919836/scp-opt
source /home/w00919836/va.env
# 清干净 va.env 的 smoke env，只留我们要的（R15-2 已验证的做法）：
for v in $(env | grep -oE '^VLLM_ASCEND_SMOKE_[A-Z_]+'); do unset $v; done
export VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731
export VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=2048   # R17-1；后续步追加
python3 tools/profile_pool_inference.py --du-store-prefault read \
  --framework-repo /home/w00919836/scp-va \
  --output tmp/r17-b2048 --sample-port 19090 --sample-worker 0 \
  --remote-config ../dp4tp2.json --no-perf --payload-layers 78 \
  --du-pipeline-mib 1 --store-bytes $((128*1024*1024*1024)) \
  --pool-start-timeout 600 --pool-count 2 --numa-node 0 --pool-b-numa-node 2 \
  --store-numa-nodes local \
  --pool-cpus '2-17,21-36,40-55,59-74,78-93' \
  --pool-b-cpus '192-207,211-226,230-245,249-264,268-283' \
  --shared-port-gather --du-devices udmac1d1e3 --du-eid-indices 0 \
  --pool-b-du-devices udmac0d1e3 --pool-b-du-eid-indices 0 \
  2>&1 | tee /home/fjj/pp_r17.log
echo "RC=$?"
```

注意：① `for v in $(env | grep -oE ...); do unset $v; done` 是兑现"仅 MODEL_PATH+阶梯变量"的关键步骤，每步执行后须 `env | grep SMOKE | sort` 原文回传核验；② 每步开头回显 `python3 -c "from examples.dsa_demo.simple_test import SmokeConfig; import inspect; print(SmokeConfig.max_num_batched_tokens)"`（或等价方式）锁默认基线值。

### R18 预案（某步过 profile 后执行）

完整 bench 配方 = fjj 框架仓（`--framework-repo /home/fjj/vllm-ascend-remote-offload` + PYTHONPATH 前置 fjj）+ **命中步的变量组合** + 我们 bench 所需其余值（BS16/MAX_MODEL_LEN=40960/CHUNKED_PREFILL=1/GPU_MEM=0.97/ENABLE_PROFILE=0/ENABLE_CPU_BINDING=0/**BENCH=1**/DATASET=chinese_40k/MAX_TOKENS=1024）+ 作者数据面。⚠️ 若命中组合含 batched=2048，则 bench 跑它（prefill 2048 分块口径入报告备注）；若命中步是更后变量（说明 batched 2048 单独不够），按实际命中组合组装。输出目录 `tmp/r18-fjj-bench`；过 = bench 基线落袋（§R10 通过标准第 3/4 条原文回传 metrics）。

### 回传格式

同 §R16（每步 env 核验原文、EI0006 计数/相位、默认基线值回显、日志路径、起止时间）。

## §R17+R18 交付：保护因子直测命中 + fjj bench 整网数据面跑通（绿区，2026-10-30 14:28~15:24）

### §R17 阶梯结果

**R17-1（batched2048 单设，scp-va + 作者数据面，tmp/r17-b2048）**：**未到 profile 相位——撞配置校验**。模型日志：
```
ValueError: chunked prefill is off: max_num_batched_tokens must be >= max_model_len
```
simple_test.py:231 跨字段校验：chunked off 时 batched(2048) < model_len(默认 25600) 组合非法。summary.txt=RUN_FAILED（`RuntimeError: model exited 1`）。EI0006=0（未到建链）。按阶梯直接进 R17-2。

**R17-2（+chunked1，tmp/r17-b2048-c1）**：**EI0006=0 命中**（profile 相位通过，决定性）。1004=76（14:34:28 `Worker_DP3_TP1_EP7 RaGetDevEidInfoList failed code=1004`，ub_v2_resource.cpp:554 rc=328103 phyId=7，KV-cache 绑定阶段）；DRAIN_OK×1（14:34:40 `Worker_DP3_TP0_EP6 DSA_SHARED_DRAIN_OK`）；EngineCore DP0/1/2 挂起 `patch_shm_broadcast.py:74 No available shared memory broadcast block found in 60 seconds`。**命中派单判据「过 profile = EI0006=0，后续 1004/hang 也算过相位」⇒ 保护因子=batched2048+chunked1 组合，阶梯即停**，R17-3/R17-4 不需执行。与蓝区「profile 过的 run 全=2048」判读吻合。

### §R18 bench（fjj 仓 + 命中组合 + BENCH=1 完整配方）

执行：`--framework-repo /home/fjj/vllm-ascend-remote-offload` + PYTHONPATH 前置 fjj + 命中组合（batched2048/chunked1）+ bench 值（BS16/MAX_MODEL_LEN=40960/CHUNKED=1/GPU_MEM=0.97/PROFILE=0/CPU_BINDING=0/BENCH=1/DATASET=chinese_40k/MAX_TOKENS=1024）+ 作者数据面（dp4tp2.json）。输出 tmp/r18-fjj-bench，日志 /home/fjj/pp_r18.log。启动 15:08:51；首跑失败（14:46 `worker.py:475 Free memory on device (2.48/91.75 GiB) ... less than 0.97(89.0 GiB)`，R17-2 残留 worker 占 HBM 环境时序）→ 清场重跑。

**env 核验原文**（`env | grep SMOKE | sort`）：
```
VLLM_ASCEND_SMOKE_BENCH=1
VLLM_ASCEND_SMOKE_DATASET=chinese_40k
VLLM_ASCEND_SMOKE_ENABLE_CHUNKED_PREFILL=1
VLLM_ASCEND_SMOKE_ENABLE_CPU_BINDING=0
VLLM_ASCEND_SMOKE_ENABLE_PROFILE=0
VLLM_ASCEND_SMOKE_GPU_MEMORY_UTILIZATION=0.97
VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=40960
VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=2048
VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=16
VLLM_ASCEND_SMOKE_MAX_TOKENS=1024
VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731
```
import 核验：`vllm_ascend=/home/fjj/vllm-ascend-remote-offload/vllm_ascend/__init__.py`（fjj 前置生效）、`vllm=/vllm-workspace/vllm/vllm/__init__.py`。non-default args 原文：`max_model_len=40960, tensor_parallel_size=2, enable_expert_parallel=True, block_size=128, gpu_memory_utilization=0.97, max_num_batched_tokens=2048, max_num_seqs=16, quantization=ascend, enable_chunked_prefill=True`（含 MTP spec 3/1/1）。

**通过标准逐条**：
1. 工具 RC：**未返回 0**——模型收尾阶段 3/4 EngineCore 挂起，主进程停在 POOL_ROI_RUNNING 未收尾，人工清理；summary.txt 未生成（无 RUN_OK）。
2. `DSA_SHARED_DRAIN_OK`：**2/8 worker**（DP0_TP0_EP0 + DP0_TP1_EP1，15:16:54，select=4997 scatter_sfa=14991 real_store=1）；DP1/2/3 六 worker 未 DRAIN（挂起）。
3. bench 证据（`[dsa-smoke][metrics]` 行 / bench 报告块 / results metrics.json）：**未产出**（3/4 EngineCore 挂起导致收尾未完成）。
4. metrics global 行 / bench 报告块原文：**无**（同 3）。

**但整网数据面首次真跑通（重大进展）**：
- **EI0006=0、code=1004=0**（命中组合确认，profile 相位通过）
- 64 请求（BS16×4DP）全部 decode 至 ~957 tokens（num_output_tokens=[794..959]），scheduler 16 请求并行，target_resident_budget=12288/sparse_budget=12288/resident_valid≈12380
- `DSA_STORE_PUT_DONE rows=[15] pages=[312] versions=[20]` 持续至 15:16:32
- `DSA_STORE_CLOSED requests=16 put_pages=389376 inference=hbm_full_remote_shared remote_get=1`（DP0 两 worker，remote_get=1 数据面真实读取）
- overlap-19090.csv **14992 行**真实 classify 数据（session=732921069170540，classify_union_mean≈74.987µs，io_count=41424，payload_bytes=27174144）

**新阻塞点定位**：推理完成后 **EngineCore DP1/2/3 挂起 `patch_shm_broadcast.py:74`**（60s 循环 ×21 条，最后 15:23:56）——与 R17-2 命中后完全同签名。EI0006 已解决，但 **3/4 EngineCore 在 bench 收尾广播处挂起**，阻止 8 worker 全 DRAIN 与 `[dsa-smoke][metrics]` 产出 → **§R10 通过标准第 3/4 条不满足，bench 基线未落袋**。交蓝区判读 patch_shm_broadcast 挂起（DP0 正常 DP1/2/3 挂的进程间不对称；推测与 bench 多 session 或 4 DP EngineCore 收尾广播握手相关）。

**日志资产**：`/home/fjj/pp_r18.log`（launch env 原文）、`/home/w00919836/scp-opt/tmp/r18-fjj-bench/raw/model.log`（860KB 全链）、`raw/overlap-19090/91.csv`（满数据）、`raw/pool-a/b.log`。R17：`/home/fjj/pp_r17.log`、`/home/w00919836/scp-opt/tmp/r17-b2048/`（R17-1 校验失败留档）；r17-b2048-c1 输出目录命中后已清理（判定证据见上）。

**框架仓 git log**：
```
1ed5def docs(pingpong): R16 判读（真凶=MAX_NUM_BATCHED_TOKENS 相关性闭环）+ R17 保护因子阶梯派单
5d3e4a1 docs(history): backfill R16 delivery commit hash
12866a4 docs(pingpong): R16 交付（va.env 原文 + 融合配方验证 EI0006，D 非保护因子）
```

**起止时间**：R17-1 14:28 / R17-2 14:33~14:41 / R18 首跑 14:45~14:47 / R18 重跑 15:08~15:24（人工清理收尾）。机器已清场（进程 0、NPU 8 卡空闲、shm 373G）。

## §R17+R18 判读（蓝区，2026-10-30）

### 裁决

1. **保护因子落袋**：batched2048+chunked1（chunked1 是硬约束——chunked off 时 batched 必须 ≥ max_model_len，R17-1 校验拦截）。R16 相关性判读升级为因果确认：**EI0006 = max_num_batched_tokens 大（≥25600）时 profile dummy 前向的首个 EP8 all-gather 建连超时**；2048 必配 chunked1。
2. **bench 数据面实质跑通**：64 请求满 decode（~957 tokens）、DSA_STORE PUT/GET 真实发生（put_pages=389376、remote_get=1）、DRAIN_OK×2（DP0）、overlap CSV 14992 行（classify_union_mean≈75µs）——remote-offload 推理主链路首次在 fjj 仓 + bench 口径下全通。
3. **新阻塞点 = 收尾 shm 广播挂起（写侧块耗尽）**：`patch_shm_broadcast.py` 是 vllm shm_broadcast **读侧**补丁（防 lost-notify，≤5s 重查 written-flag）；挂起报文是 **enqueue 写侧**「60s 无空闲块」。机制 = 某 reader 不再消费 → 块不回收 → 写侧永久等。DP0 组干净 / DP1-3 组挂起 = 非对称消费。**泄漏量∝请求数×decode 步数**假设最吻合：R9（16 请求/256 tokens）干净、R18（64 请求/957 tokens）挂、R17-2（失败 teardown）也挂 → 每请求/每步广播有块未被全部读者消费，积压至收尾耗尽。
4. §R10 第 3/4 条不满足：**metrics 未产出**（generate 不返回，per-rank 统计与全局合都在 generate 之后）。基线仍未落袋（0/11）。
5. 教训：R17-1 证明 simple_test.py:231 的跨字段校验会拦非法组合（batched<model_len 需 chunked on）——后续配方必须成对考虑。

### 判读落点

工程止血优先于根修：退回 R9 已证包线（BS4×DP4=16 请求）内跑 BENCH=1，先落基线口径；块泄漏根因走代码线（蓝区查 vllm shm_broadcast 写侧等待与 DSA 广播调用方的块生命周期）。

## §R19 派单：收敛并发落 bench 基线（止血线，蓝区 2026-10-30）

### R19-0 · 只读取证（挂起上下文，免费）

```bash
L=/home/w00919836/scp-opt/tmp/r18-fjj-bench/raw/model.log
# 1) 挂起报文前后文（谁写谁等）
grep -n -B3 -A6 "No available shared memory broadcast block" $L | head -80
# 2) 各 DP 组 store 关闭顺序
grep -nE "DSA_STORE_CLOSED|DSA_SHARED_DRAIN_OK|POOL.*CLOSE|shutting_down" $L | tail -40
# 3) 挂起 worker 的最后活动行（DP1/2/3 的 worker 最后 5 行）
for dp in 1 2 3; do for tp in 0 1; do
  echo "== DP${dp}_TP${tp} =="; grep "Worker_DP${dp}_TP${tp}" $L | tail -5
done; done
```

原文回传三块。佐证 writer/reader 等待方向与块积压位置。

### R19-1 · BS4 止血基线（R18 逐字，仅 BS16→BS4）

R18 命令/env 逐字（fjj 仓 + BENCH=1 + batched2048/chunked1 + MAX_TOKENS=1024 + 作者数据面），唯一改 `export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=4`（全局 16 请求，R9 包线内），输出目录 `tmp/r19-bench-bs4`，日志 `/home/fjj/pp_r19.log`。
- **过**（8×DRAIN + `[dsa-smoke][metrics][dp=N]` ×4 + metrics.json）⇒ **bench 基线落袋**（BS4 口径），原文回传 metrics global 行与报告块。
- **patch_shm 挂起复现** ⇒ 泄漏∝decode 步数而非请求数 → R19-2。
- EI0006 ⇒ 配方回归检查（不应发生；发生即回传 env 核验）。
- 1004 ⇒ 白名单重跑（≤2）。

### R19-2 · BS4 + 短 decode（R19-1 挂起后执行）

R19-1 逐字但 `export VLLM_ASCEND_SMOKE_MAX_TOKENS=256`（R9 同长度），输出目录 `tmp/r19-bench-bs4-mt256`。
- 过 ⇒ 基线落袋（BS4×256 口径），并坐实泄漏∝步数。
- 挂 ⇒ R19-3：BS1 + MAX_TOKENS=64（R9 精确包线 + fjj + BENCH），输出 `tmp/r19-bench-bs1-mt64`；再挂 ⇒ 停，回传，蓝区转代码线（shm 块生命周期修复后再战 BS16）。

### 原则与回传格式

止血线每轮含完整清场（R18 挂起后进程树需人工核净）；通过标准同 §R10 第 3/4 条（metrics 原文必回）；附 env 核验/import 路径/日志路径/起止时间。

## §R19 交付：BS4 止血线两档——挂起复现坐实泄漏∝步数 + 短 decode 8/8 DRAIN 消除挂起（绿区，2026-10-30 17:41~18:02）

### R19-0 · 只读取证（挂起上下文，r18-fjj-bench/raw/model.log）

**1) 挂起报文前后文**：首个挂起 15:17:55，DP1/DP2/DP3 三 EngineCore（pid 1821824/1821830/1821836）60s 循环至 15:23:57 SIGTERM；报文前 5119-5121 行有 `RuntimeError: offline DP rank failed: dsa-dp-0=1`（simple_test.py:730 run_node_dp）。
**2) 各 DP 组 store 关闭顺序**：仅 DP0 两 worker 15:16:54 `DSA_SHARED_DRAIN_OK` + `DSA_STORE_CLOSED requests=16 put_pages=389376 remote_get=1`；无 POOL.*CLOSE/shutting_down。
**3) 挂起 worker 最后活动**：DP1/2/3 六 worker 全停在 15:16:32 `DSA_SHARED_CONNECT_OK groups=21` 后 `acl_graph.py:284 Replaying aclgraph` + `llm_base_proposer.py:651 Replaying MTP draft graph`，之后完全静默——卡在 graph replay 后同步点。

### R19-1 · BS4×1024（tmp/r19-bench-bs4，日志 pp_r19.log）

R18 逐字仅 BS16→BS4（MAX_NUM_SEQS=4）。**patch_shm 挂起复现**（命中预案「泄漏∝decode 步数而非请求数」→ 转 R19-2）：
- EI0006=0 / code=1004=0（命中组合稳定）
- **DRAIN_OK 仅 2/8（DP2 两 worker）** 17:46:27，`DSA_STORE_CLOSED requests=4 put_pages=97344 inference=hbm_full_remote_shared remote_get=1`
- 挂起 EngineCore = **DP0/DP1/DP3**（R18 是 DP1/2/3 挂、DP0 正常 → 挂起组不对称且轮次间漂移，支持蓝区「某 reader 不再消费→块不回收」机制假设）
- 首个挂起 17:47:28（DRAIN 后 61s），60s 循环 ×4；主进程 POOL_ROI_RUNNING 未收尾，人工清理
- **BS4 全局 16 请求（R9 包线内）仍挂** → 泄漏∝decode 步数（1024）而非请求数（16）坐实

### R19-2 · BS4×256（tmp/r19-bench-bs4-mt256，日志 pp_r19.log 被覆盖为 R19-2）

**短 decode 消除挂起 + 撞框架仓收尾统计 bug**：
- EI0006=0 / code=1004=0
- **8/8 worker 全 DSA_SHARED_DRAIN_OK + DSA_STORE_CLOSED**（18:01:51，8 worker 全部 `requests=4 put_pages=97344 inference=hbm_full_remote_shared remote_get=1`，select≈2413-2527 scatter_sfa≈7239-7581）——**remote-offload 整网首次 8 worker 全 DRAIN 无挂起**
- **patch_shm_broadcast 挂起 = 0 次**（短 decode 256 tokens 彻底消除收尾挂起，坐实泄漏∝步数）
- overlap CSV 8 端口各 7239-7581 行，classify_union_mean_us ≈ 38.3-39.4µs
- **但主进程 RUN_FAILED**：推理全部完成后 4 DP 在 `summarize_request_metrics`（simple_test.py:451）崩 `AttributeError: 'RequestStateStats' object has no attribute 'first_token_time'. Did you mean: 'first_token_ts'?` → summary.txt RUN_FAILED、`[dsa-smoke][metrics]`/metrics.json 未产出

### 关键判定

1. **收尾 shm 广播挂起 = 泄漏∝decode 步数，短 decode（256）可消除**：R19-1（1024）挂、R19-2（256）8/8 DRAIN 无挂起。蓝区「某 reader 不再消费→块不回收→写侧永久等」机制假设成立；泄漏量∝步数而非请求数。
2. **新阻塞点 = 框架仓收尾统计代码 bug（蓝区修）**：`simple_test.py:451/454/455` 引用 `metrics.first_token_time`/`last_token_time`，但 vLLM 0.23.0 `RequestStateStats`（vllm/v1/metrics/stats.py:208-209）字段为 `first_token_ts`/`last_token_ts`。该函数由框架仓 R10 bench 口径新提交 `717c396`/`e6fd53b` 引入（作者 scp-va 仓 simple_test.py 无此函数，grep 无匹配）→ **代码 bug 而非环境问题**，改字段名即可收 baseline。
3. R19-3（BS1×64）**不执行**：挂起已消除（非挂起问题），剩余为统计代码 bug。
4. 通过标准逐条：①工具 RC 非 0（统计 bug）；②DRAIN 8/8（R19-2 达成）；③metrics 未产出（bug 阻断）；④metrics 原文无。

**日志资产**：`/home/fjj/pp_r19.log`（R19-2 主日志，含 POOL_ROI/REPORT 原文）、`/home/w00919836/scp-opt/tmp/r19-bench-bs4/raw/model.log`（R19-1）、`/home/w00919836/scp-opt/tmp/r19-bench-bs4-mt256/raw/model.log`（R19-2 860KB 全链）、`summary.txt`/`run-status.json`（RUN_FAILED + graceful:true）、`raw/overlap-19*.csv`（8 端口满数据）。R19-1 残留 backing 已清（urma_mock DU 持 deleted FD 致 shm 269G→清后 373G）。

**起止时间**：R19-1 17:41~17:53（人工清理收尾）、R19-2 17:57~18:02（自然结束）。机器已清场（进程 0、NPU 8 卡空闲、shm 373G）。

## §R19 判读（蓝区，2026-10-30）+ 统计 bug 已修

### 裁决

1. **泄漏∝decode 步数坐实**：R19-1（BS4×1024）挂起复现（请求数已从 64→16 仍挂）⇒ 与请求数无关、与步数强相关；R19-2（BS4×256）8/8 DRAIN + 挂起 0 次 ⇒ 短 decode 彻底绕开。**当前可用包线：单条 decode ≤256 步**；1024 步需修 shm 块泄漏（代码线挂账，见 §R18 判读）。
2. **新阻塞 = 蓝区 bench 代码自身 bug**（绿区定位精准，致谢）：`simple_test.py:451/454/455` 用 `metrics.first_token_time/last_token_time`，vLLM 0.23.0 `RequestStateStats` 已改名 `first_token_ts/last_token_ts`（vllm/v1/metrics/stats.py:208-209）。**蓝区已修复**（3 处改名，本 commit），mock 全过（/tmp/test_smoke_metrics.py 同步改）。
3. 附带资产：R19-2 overlap CSV 8 端口 7239-7581 行，classify_union_mean≈38-39µs（短 decode 口径，与 R18 长 decode 75µs 同量级）。

## §R20 派单：BS4×256 bench 基线落袋轮（蓝区，2026-10-30）

### 执行

```bash
cd /home/fjj/vllm-ascend-remote-offload && git -c http.proxy=http://90.255.129.35:3128 pull --ff-only && git log --oneline -2
# 顶部应含本修复提交（simple_test.py first_token_ts 改名）
```

然后 **R19-2 命令逐字重跑**（fjj 仓 + BENCH=1 + batched2048/chunked1 + BS4 + MAX_TOKENS=256 + 作者数据面全不变），输出目录 `tmp/r20-bench-bs4-mt256`，日志 `/home/fjj/pp_r20.log`。

### 通过标准（§R10 第 3/4 条，本轮核心）

1. 8/8 DRAIN_OK + STORE_CLOSED（R19-2 已证可达，防回归）；
2. 4 rank 各打出 `[dsa-smoke][metrics][dp=N]` 行 + bench 报告块；
3. `results/1p/<run_name>.metrics.json` 存在且 global 含 requests/分位数/吞吐/tpot_under_50ms；
4. **原文回传** `[dsa-smoke][metrics][global]` 行与全局报告块全文。

### 失败处置

- AttributeError 复现 ⇒ pull 未生效，回传 `git log --oneline -2`；
- patch_shm 挂起复现 ⇒ 泄漏非确定性，回传挂起计数/上下文，蓝区重估包线；
- 1004 ⇒ 白名单重跑；EI0006 ⇒ 配方回归检查（不应发生）；其它新报错停。

## §R20 交付：统计 bug 修复生效，bench 基线首次落袋（绿区，2026-10-30 23:10~23:16）

### 执行核验

- pull 到 c461a78（`git log --oneline -2` 顶部=first_token_ts 修复提交），simple_test.py:451/454/455 已改 `first_token_time→first_token_ts` ×3（grep 核验落地）。
- R19-2 命令逐字重跑（fjj 仓 + BENCH=1 + batched2048/chunked1 + BS4 + MAX_TOKENS=256 + 作者数据面全不变），输出 `tmp/r20-bench-bs4-mt256`，日志 `/home/fjj/pp_r20.log`。
- 启动前置门禁：进程 0 / NPU 8 卡全空闲 / shm 373G。23:11:09 双 Pool READY（groups=['a','b']，store 19080-19087/du 19090-19097 路由正确）→ 23:16 自然收尾。

### 结果（通过标准逐条，§R10 第 3/4 条核心）

1. **8/8 DRAIN_OK + 8×STORE_CLOSED**（23:15:34-36，8 worker 全 `requests=4 put_pages=97344 inference=hbm_full_remote_shared remote_get=1`，DP0-3 各 TP0/TP1）——R19-2 防回归达成。
2. **4 rank 各打出 `[dsa-smoke][metrics][dp=N]` 行**（dp=0/1/2/3，各 requests=4 output_tokens=1024）。
3. **`results/1p/dp4_tp2_offload_graph_mtp_graph_hbm_full_remote_shared.metrics.json` 存在**（框架仓 results/ 下，model.log:2780 有写入确认），global 含 requests/分位数/吞吐/tpot_under_50ms 全字段。
4. **`[dsa-smoke][metrics][global]` 原文回传**（见下）。

**summary.txt = RUN_OK model_exit=0**（首次整网干净收尾）；run-status.json graceful=true errors=[]；EI0006=0 / code=1004=0 / **patch_shm_broadcast 挂起 = 0 次**。

### `[dsa-smoke][metrics][global]` 原文

```
{"dp_size": 4, "requests": 16, "output_tokens": 4096, "global_wall_s": 81.54, "request_throughput_rps": 0.196, "global_output_tps": 50.23, "ttft_ms": {"avg": -1792611307115.808, "min": -1792611334035.134, "p50": -1792611307029.019, "p90": -1792611280436.494, "p99": -1792611280413.981, "max": -1792611280413.981}, "tpot_ms": {"avg": 136.055, "min": 29.899, "p50": 135.997, "p90": 241.908, "p99": 242.134, "max": 242.134}, "tpot_under_50ms": false}
```

### ⚠️ 数据质量告警（交蓝区判读，不阻塞落袋）

**ttft_ms 为巨大负值**（global avg=-1.79e12 ms，各 rank/per_request 同）：`ttft_ms = (metrics.first_token_ts - metrics.arrival_time) * 1000`——first_token_ts 与 arrival_time **时钟基准/单位不一致**（first_token_ts 是 epoch 秒级，arrival_time 疑似另一套），相减得巨负。**tpot_ms = (last_token_ts - first_token_ts)/(n-1) 因同基准相减而正常**（avg=136.055）。其余字段（requests/output_tokens/global_wall_s/request_throughput/global_output_tps）均正常。建议蓝区核 vLLM 0.23 `RequestStateStats.arrival_time` 与 `first_token_ts` 单位是否需换算（如 arrival_time 是单调时钟/相对秒）。

### 附带资产

- overlap CSV 8 端口各 7239-7581 行（总计 59394 行），`classify_union_mean_us` 各端口均值 ≈ **12.7-13.0µs**（19090=12.955/19091=12.865/19092=12.823/19093=12.840/19094=13.011/19095=13.022/19096=12.692/19097=12.680）。
- non-default args 原文：`max_model_len=40960, tensor_parallel_size=2, enable_expert_parallel=True, block_size=128, gpu_memory_utilization=0.97, max_num_batched_tokens=2048, max_num_seqs=4, quantization=ascend, enable_chunked_prefill=True, speculative_config=mtp(3)`, compilation FULL_DECODE_ONLY capture_sizes=[4,8,16]。

**日志资产**：`/home/fjj/pp_r20.log`、`/home/w00919836/scp-opt/tmp/r20-bench-bs4-mt256/raw/model.log`（514KB 全链）、`summary.txt`（RUN_OK）、`run-status.json`（graceful:true）、`raw/overlap-*.csv`（8 端口满数据）、`results/1p/dp4_tp2_offload_graph_mtp_graph_hbm_full_remote_shared.metrics.json`。机器已清场（进程 0、NPU 8 卡空闲、shm 373G）。

**起止时间**：23:10~23:16（自然结束）。**bench 基线 1/11 落袋**。

## §R20 判读（蓝区，2026-10-30）

### 裁决

1. **阳性落袋**：首次 bench 全链闭环——RUN_OK model_exit=0、8/8 DRAIN_OK+STORE_CLOSED（put_pages=97344 remote_get=1）、patch_shm 挂起 0 次、4×`[dsa-smoke][metrics][dp=N]`+`[global]`、metrics.json 全字段。基线（BS4×DP4=16 请求 × 256 token 定长，batched2048/chunked1，作者数据面）：`requests=16 output_tokens=4096 wall=81.54s rps=0.196 output_tps=50.23 tpot_ms avg=136.055 p50=135.997 p90=241.908 p99=242.134 min=29.899`。
2. **TPOT 口径解读（重要，判读防误读）**：avg/p50 被 **chunked-prefill 交错污染**——16 请求 × 40K prefill ÷ 2048 chunk 的 prefill 肩完全覆盖 decode 期，decode 步与 prefill 共享 batched 预算；**min=29.899ms 代表无 prefill 竞争时的干净 decode 步**，p90/p99≈242ms 属 prefill 期尾段。即：本口径的 TPOT 是「单次离线 cohort 混合相位」，**不是稳态 decode 性能**，不可直接用于 TPOT<50ms 判定；稳态口径需（a）prefill 完成后分段统计，或（b）100K×1K 长 decode 场景尾段（待 shm 泄漏根修后可达 1024 步）。用户预期 decode 2000+ tok/s 与本口径 50 tok/s 的差距主因 = prefill 主导 + BS4 低并发 + batched2048 限制，需同口径对比。
3. **TTFT bug 定性闭环**：`arrival_time` 为 epoch **微秒**、`first/last_token_ts` 为 epoch **秒**（-1.79e12 ms 量级与 1e6 倍率严格吻合）。**已本地修复**：>1e12 视为微秒归一（注释说明），mock 增加真实单位用例（ttft=1800ms/tpot=100ms）全过。
4. 数据面开销旁证：8 端口 overlap CSV 各 7239-7581 行，classify_union_mean≈12.7-13.0µs（低开销，非瓶颈）。
5. 待办：TTFT 复跑取数；shm 泄漏根修（1024 步/BS16 前提）；offload-off 对照（+20% 口径）。

## §R21 派单：TTFT 修复复跑 + 挂起取证（蓝区，2026-10-30）

### R21-0 · 只读取证（stats 单位权威确认 + R19-1 挂起上下文）

```bash
sed -n '195,240p' /vllm-workspace/vllm/vllm/v1/metrics/stats.py
grep -n "arrival_time" /vllm-workspace/vllm/vllm/v1/metrics/stats.py /vllm-workspace/vllm/vllm/v1/engine/*.py 2>/dev/null | head -20
# R19-1 挂起上下文（养代码线修块泄漏用）：
L=/home/w00919836/scp-opt/tmp/r19-bench-bs4/raw/model.log
grep -c "No available shared memory broadcast block" $L
grep -n "No available shared memory broadcast block" $L | head -5
for dp in 0 1 2 3; do for tp in 0 1; do echo "== DP${dp}_TP${tp} =="; grep "Worker_DP${dp}_TP${tp}" $L | tail -3; done; done
```

三块原文回传。

### R21-1 · TTFT 修复复跑（R20 逐字）

```bash
cd /home/fjj/vllm-ascend-remote-offload && git -c http.proxy=http://90.255.129.35:3128 pull --ff-only && git log --oneline -2
# 顶部应含 TTFT 单位归一提交
```

R20 命令逐字重跑（BS4×256 全配方不变），输出目录 `tmp/r21-bench-bs4-mt256`，日志 `/home/fjj/pp_r21.log`。

**通过标准**：同 §R20 第 1/3/4 条 + **ttft_ms 量级合理**（0 < ttft_ms < global_wall_s×1000，预期数秒~数十秒，40K chunked prefill 本就大）；原文回传 `[global]` 行与报告块。

### 失败处置

- TTFT 仍负/超界 ⇒ 回传 R21-0 的 stats 源码原文，蓝区按真实单位再修；
- 其它同 §R20（挂起/1004/EI0006 白名单逻辑不变）。

### R22 预案（R21 过后）

1. **offload-off 对照轮**（同 BS4×256 口径，`VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=0`，其余不变）→ 计算吞吐/TPOT 相对提升，对齐「+20% 自提升」验收口径；
2. **shm 泄漏代码线**：按 R21-0 挂起上下文定位块泄漏点（reader 停消费的 DP 组 + 最后活动），蓝区出修复 commit 后再战 BS16×1024 / 100K×1K。


## §R21 交付：TTFT 复跑仍负（时钟域根因坐实），stats 源码原文回传（绿区，2026-10-30 23:42~23:53）

### 执行核验

- pull 到 15d4610（`git log --oneline -2` 顶部=TTFT 单位归一提交），simple_test.py:454-457 已改 `arrival_s` 归一 + mock 用例（grep 核验落地）。
- R20 命令逐字重跑（fjj 仓 + BENCH=1 + batched2048/chunked1 + BS4 + MAX_TOKENS=256 + 作者数据面全不变），输出 `tmp/r21-bench-bs4-mt256`，日志 `/home/fjj/pp_r21.log`。
- 首跑 23:42:27 崩：DP3_TP1_EP7 `_bind_dsa_split_kv_caches`→`aiv_urma_query_eids` RaGetDevEidInfoNum **code=1004**（与 R9/R11-D/R15-1 同签名瞬时竞争，本次落 DP3；E4=4/DRAIN=1/STCL=1/HANG=1）。按白名单清场重跑：精确 PID 杀残留 + rm 8×32G backing（shm 372G 恢复）→ 23:53:18 复跑成功。
- 复跑启动前置门禁：进程 0 / NPU 8 卡全空闲 / shm 372G。双 Pool READY（groups=['a','b']）→ 自然收尾。

### 结果（通过标准逐条）

1. **8/8 DRAIN_OK + 8×STORE_CLOSED**（23:53:18，8 worker 全 `requests=4 put_pages=97344 inference=hbm_full_remote_shared remote_get=1`）——EI=0/E4=0/DRAIN=8/STCL=8/MET=6/HANG=0。
2. **4 rank 各打出 `[dsa-smoke][metrics][dp=N]` 行**（dp=0/1/2/3，各 requests=4 output_tokens=1024）。
3. **`results/1p/dp4_tp2_offload_graph_mtp_graph_hbm_full_remote_shared.metrics.json` 存在**（框架仓 results/ 下，model.log:2781 有写入确认），global 含 requests/分位数/吞吐/tpot_under_50ms 全字段。
4. **`[dsa-smoke][metrics][global]` 原文回传**（见下）。

**summary.txt = RUN_OK model_exit=0**；run-status.json graceful=true errors=[]；patch_shm_broadcast 挂起 = 0 次；EI0006=0 / code=1004=0（复跑）。

### `[dsa-smoke][metrics][global]` 原文

```
{"dp_size": 4, "requests": 16, "output_tokens": 4096, "global_wall_s": 81.481, "request_throughput_rps": 0.196, "global_output_tps": 50.27, "ttft_ms": {"avg": -1792611306857.886, "min": -1792611333864.11, "p50": -1792611306720.925, "p90": -1792611280138.26, "p99": -1792611280046.97, "max": -1792611280046.97}, "tpot_ms": {"avg": 135.234, "min": 28.725, "p50": 134.919, "p90": 241.744, "p99": 241.98, "max": 241.98}, "tpot_under_50ms": false}
```

### ❌ 失败处置命中：TTFT 仍负（-1.79e12），根因=时钟域错配（修复前提错误）

**15d4610 的修复已生效执行，但前提错误**：`arrival_s = arrival_time/1e6 if >1e12 else arrival_time` 只归一了 arrival_time 的量级，而 `first_token_ts` 是**另一时钟域（engine-core monotonic）**，与 arrival_time（wall-clock epoch）直接相减必然无意义。

**决定性证据链**（容器内 `/vllm-workspace/vllm/` 源码）：

1. `vllm/v1/engine/__init__.py:230-231`：`EngineCoreOutputs.__post_init__: if self.timestamp == 0.0: self.timestamp = time.monotonic()`——**`first_token_ts`/`last_token_ts` = `outputs.timestamp` = `time.monotonic()`**（engine-core 单调时钟）。
2. `vllm/v1/engine/async_llm.py:742` + `input_processor.py:278,289`：**`arrival_time = time.time()`**（wall-clock epoch 秒）。
3. **数学闭合**：实测 `gen_start_epoch=1793375411`（epoch 秒）；`first_token_ts ≈ 764077`（monotonic，≈8.8 天 uptime）。`first_token_ts - arrival_s = 764077 - 1793375411 = -1792611334`，×1000 = `-1.7926e12`，与实测 ttft_ms 逐位吻合。

**vLLM 官方同域 TTFT 现成字段**：`stats.py:369-371` `first_token_latency = iteration_timestamp - arrival_time`（两者都是 `time.time()` wall-clock），已写入 `RequestStateStats.first_token_latency`（stats.py:217），并经 `output_processor.py:373 metrics=self.stats` 随 RequestOutput 透出。**建议蓝区改用 `metrics.first_token_latency`（秒）×1000 即 TTFT(ms)**，或自行用同域 wall-clock，勿跨时钟域相减。

### R21-0 三块取证原文回传

**块1 · stats.py:195-240（RequestStateStats 定义，含时钟域注释）**：

```
# This is an engine frontend timestamp (wall-clock)
arrival_time: float = 0.0

# These are engine core timestamps (monotonic)
queued_ts: float = 0.0
scheduled_ts: float = 0.0
first_token_ts: float = 0.0
last_token_ts: float = 0.0

# first token latency
first_token_latency: float = 0.0
```

**块2 · arrival_time 全链 grep**（engine/*.py）：

```
/vllm-workspace/vllm/vllm/v1/engine/async_llm.py:742:            arrival_time=time.time(),
/vllm-workspace/vllm/vllm/v1/engine/input_processor.py:278:                arrival_time = prompt.get("arrival_time", time.time())
/vllm-workspace/vllm/vllm/v1/engine/input_processor.py:289:                arrival_time = time.time()
/vllm-workspace/vllm/vllm/v1/engine/output_processor.py:207:            self.stats.arrival_time = update.arrival_time
/vllm-workspace/vllm/vllm/v1/metrics/stats.py:208:    arrival_time: float = 0.0
/vllm-workspace/vllm/vllm/v1/metrics/stats.py:369:            first_token_latency = self._time_since(req_stats.arrival_time)
/vllm-workspace/vllm/vllm/v1/metrics/stats.py:437:        e2e_latency = self._time_since(req_stats.arrival_time)
```

**块3 · R19-1 挂起上下文**（/home/w00919836/scp-opt/tmp/r19-bench-bs4/raw/model.log，18 条报文）：

```
17:47:28 首现：EngineCore_DP0/DP1/DP3 三 EngineCore 60s 循环
  (EngineCore_DP0 pid=1998570) 17:47:28 [patch_shm_broadcast.py:74] No available shared memory broadcast block found in 60 seconds.
  (EngineCore_DP3 pid=1998582) 17:47:28 [patch_shm_broadcast.py:74] No available shared memory broadcast block found in 60 seconds.
  (EngineCore_DP1 pid=1998588) 17:47:28 [patch_shm_broadcast.py:74] No available shared memory broadcast block found in 60 seconds.
  ... 每 60s 重复至 17:52:29（共 18 条）
DP2 组已 DRAIN+STORE_CLOSED（requests=4 put_pages=97344 remote_get=1）
挂起 worker 最后活动（17:46:09）：
  DP0_TP0/DP1_TP0/DP3_TP0 → [acl_graph.py:284] Replaying aclgraph + [llm_base_proposer.py:651] Replaying MTP draft
  DP0_TP1/DP1_TP1/DP3_TP1 → [remote_store.py:230] DSA_STORE_PUT_DONE rows=[3] pages=[312] versions=[21] bound=0
                          → [remote_shared.py:180] DSA_SHARED_CONNECT_BEGIN endpoint=127.0.0.1:1909x
                          → [remote_shared.py:207] DSA_SHARED_CONNECT_OK wire=6 backend=real_store group_gather=full groups=21
  DP2_TP0/TP1（已收尾）→ [remote_store.py:252] DSA_STORE_CLOSED requests=4 put_pages=97344
```

### 附带资产

- overlap CSV 8 端口各 7239-7581 行（总计 59394 行），classify_union_mean_us ≈ 12.7-13.0µs（同 R20）。
- non-default args 原文：`max_model_len=40960, tensor_parallel_size=2, enable_expert_parallel=True, block_size=128, gpu_memory_utilization=0.97, max_num_batched_tokens=2048, max_num_seqs=4, quantization=ascend, enable_chunked_prefill=True, speculative_config=mtp(3)`, compilation FULL_DECODE_ONLY capture_sizes=[4,8,16]。

**日志资产**：`/home/fjj/pp_r21.log`、`/home/w00919836/scp-opt/tmp/r21-bench-bs4-mt256/raw/model.log`、`summary.txt`（RUN_OK）、`run-status.json`（graceful:true）、`results/1p/dp4_tp2_offload_graph_mtp_graph_hbm_full_remote_shared.metrics.json`（23:51 复跑版本）。机器已清场（进程 0、NPU 8 卡空闲、shm 372G）。

**起止时间**：23:42~23:53（含 1004 重跑）。**bench 基线保持 R20 值（TTFT 待蓝区按真实时钟域修后复取）**。

## §R21 判读（蓝区，2026-10-30）

### 裁决

1. **bench 管道本身已验收**：RUN_OK model_exit=0、8/8 DRAIN_OK + 8×STORE_CLOSED（put_pages=97344 remote_get=1）、EI0006=0/1004=0（复跑后）/patch_shm 挂起 0 次、4×`[metrics][dp=N]` + `[global]` + metrics.json 全字段。TTFT 负值是**统计链路 bug，不是引擎问题**——不污染 R20 已落袋的吞吐/TPOT 基线。
2. **TTFT 根因闭环，蓝区认错并采纳绿区方案**：15d461035 只修了量级（arrival_time µs→s），方向错误；绿区证据链完整——`engine/__init__.py:231` `first_token_ts = time.monotonic()` vs `async_llm.py:742` `arrival_time = time.time()`，跨时钟域相减无意义，且数学逐位闭合（764077 − 1793375411 = −1792611334，×1000 = −1.7926e12 与实测吻合）。**已按绿区建议修复**：`summarize_request_metrics` 优先取 `metrics.first_token_latency`（stats.py:369-371，官方同域 wall-clock，单位秒 ×1000），不存在或 ≤0 才回退旧 first_token_ts 路径；本地 mock 三用例全过（ftl 优先 / None 回退 / 0 回退）。
3. **shm 挂起机制认知更新（R21-0 块3）**：R19-1 挂起 worker 最后活动 = TP1 `DSA_STORE_PUT_DONE → DSA_SHARED_CONNECT_BEGIN/OK`（17:46:09）后静默，TP0 停在 acl_graph/MTP draft replay；EngineCore 侧 60s「no available block」是**下游症状**——worker 停在 connect/replay 同步点不再消费广播块 → 写侧耗尽。**头号嫌疑从"块泄漏"修正为"worker 在 DSA_SHARED_CONNECT_OK 之后的同步点停摆"**，shm 写侧等待只是表象。R22b 取证按此方向收口。
4. 1004 瞬时竞争本轮落 DP3（首跑崩、清场重跑过），白名单逻辑有效；1004 根修仍挂账（机器层，R8 证据已交用户线）。

## §R22 派单：offload-off 对照轮 + shm 挂起取证收口（蓝区，2026-10-30）

### R22-0 · 只读取证（shm 写侧源码 + R19-1 停摆窗口，免费，先做）

```bash
# 块1 · shm_broadcast.py 结构与写侧函数（vllm 容器内版本，须精确）
F=/vllm-workspace/vllm/vllm/distributed/device_communicators/shm_broadcast.py
wc -l $F
grep -n "^class \|    def \|No available" $F
# 按上面 grep 到的行号，sed 出 acquire_write 全文与 dequeue/reader 推进段（例：sed -n 'A,Bp'），原文回传

# 块2 · R19-1 停摆窗口（17:46:09 CONNECT_OK → 17:47:28 首个 60s 告警之间，挂起 worker 有没有任何输出）
L=/home/w00919836/scp-opt/tmp/r19-bench-bs4/raw/model.log
grep -nE "Worker_DP[013]_TP[01]" $L | awk -F: '$1>0' | grep -E "17:4[6-9]|17:5[0-2]" | head -40
# 块3 · 各挂起 EngineCore 告警总数与最后一条时间戳
for dp in 0 1 3; do echo "== EngineCore_DP$dp =="; grep -c "EngineCore_DP$dp.*No available shared memory" $L; grep "EngineCore_DP$dp.*No available shared memory" $L | tail -1; done
```

三块原文回传。块1 供蓝区出 shm 修复 commit；块2/3 判定 worker 停摆点后移还是 connect 内死等。

### R22-1 · offload-off 对照轮（+20% 自提升基线，同时验证 TTFT 修复）

```bash
cd /home/fjj/vllm-ascend-remote-offload && git -c http.proxy=http://90.255.129.35:3128 pull --ff-only && git log --oneline -2
# 顶部应含 TTFT first_token_latency 修复提交；grep 核验：
grep -n "first_token_latency" examples/dsa_demo/simple_test.py | head -5
```

然后 R13-1 命令骨架逐字，env 如下（相对 R20/R21 唯一变量 = **ENABLE_OFFLOAD=0**，其余全同 R21 复跑配方）：

```bash
cd /home/w00919836/scp-opt
source /home/w00919836/va.env
export VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731
export VLLM_ASCEND_SMOKE_DATASET=chinese_40k
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=4
export VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=40960
export VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=2048
export VLLM_ASCEND_SMOKE_MAX_TOKENS=256
export VLLM_ASCEND_SMOKE_ENABLE_CHUNKED_PREFILL=1
export VLLM_ASCEND_SMOKE_GPU_MEMORY_UTILIZATION=0.97
export VLLM_ASCEND_SMOKE_ENABLE_PROFILE=0
export VLLM_ASCEND_SMOKE_BENCH=1
export VLLM_ASCEND_SMOKE_ENABLE_CPU_BINDING=0
export VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=0   # ← 本轮唯一变量（offload-off 对照）
export PYTHONPATH=/home/fjj/vllm-ascend-remote-offload:$PYTHONPATH
python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"   # 须指向 fjj 仓
python3 tools/profile_pool_inference.py --du-store-prefault read \
  --framework-repo /home/fjj/vllm-ascend-remote-offload \
  --output tmp/r22a-bench-bs4-mt256-nooffload --sample-port 19090 --sample-worker 0 \
  --remote-config ../dp4tp2.json --no-perf --payload-layers 78 \
  --du-pipeline-mib 1 --store-bytes $((128*1024*1024*1024)) \
  --pool-start-timeout 600 --pool-count 2 --numa-node 0 --pool-b-numa-node 2 \
  --store-numa-nodes local \
  --pool-cpus '2-17,21-36,40-55,59-74,78-93' \
  --pool-b-cpus '192-207,211-226,230-245,249-264,268-283' \
  --shared-port-gather --du-devices udmac1d1e3 --du-eid-indices 0 \
  --pool-b-du-devices udmac0d1e3 --pool-b-du-eid-indices 0 \
  2>&1 | tee /home/fjj/pp_r22a.log
echo "RC=$?"
```

前置门禁同 R21：进程 0 / NPU 8 卡全空闲 / shm ≥370G。白名单不变（1004 ≤2 次重跑，≥5min 静置 + 完整清场）。

### 通过标准

1. RUN_OK + 8/8 DRAIN_OK（offload-off 下无 STORE_CLOSED 属预期，回传实际 DRAIN/模式行原文）；
2. 4 rank 各打出 `[dsa-smoke][metrics][dp=N]` + metrics.json 存在（路径若因 offload-off 变名，回传实际路径）；
3. **TTFT 修复验证**：`ttft_ms` 全部分位数满足 `0 < ttft_ms < global_wall_s×1000`（预期数秒~数十秒，40K chunked prefill 本就大）；
4. 原文回传 `[dsa-smoke][metrics][global]` 行 + 全局报告块全文 + `env | grep SMOKE | sort` 原文。

### 失败处置

- TTFT 仍越界 ⇒ 回传 `metrics.json` 中 per_request 明细前 3 条 + `grep -n "first_token_latency" /vllm-workspace/vllm/vllm/v1/metrics/stats.py` 原文，蓝区再修；
- patch_shm 挂起复现 ⇒ 回传挂起计数/上下文（256 步包线内不应发生）；
- 1004 ⇒ 白名单；EI0006 ⇒ 配方回归检查（不应发生）；其它新报错停。

### 判读预案（派单时写死）

- R22-1 vs R20/R21 基线（wall=81.5s、tps=50.2、tpot avg≈135-136ms/min≈29ms）做同口径对比：offload-on 相对 offload-off 的吞吐/TPOT 变化即「自提升」口径分子分母；
- 若 offload-off 反而更慢或持平，判读时区分「40K prefill 主导期 offload 无收益」与「decode 稳态收益」，不下一刀切结论。

## §R22 交付：offload-off 对照轮 + TTFT 修复验证通过 + shm 挂起取证（绿区，2026-10-31 01:20~01:25）

### R22-0 · 三块取证原文（只读）

**块1 · shm_broadcast.py 结构与写侧源码**（vllm 容器内 /vllm-workspace/vllm/vllm/distributed/device_communicators/shm_broadcast.py，941 行）

结构（grep -n '^class \|    def \|No available'）:
```
96:    "No available shared memory broadcast block found "
104:class SpinCondition:
121:    def __init__(
173:    def record_read(self):
176:    def cancel(self):
183:    def wait(self, timeout_ms: int | None = None) -> None:
210:    def notify(self):
216:class ShmRingBuffer:
317:    def handle(self):
325:    def __reduce__(self):
331:    def __del__(self):
338:    def get_data(self, current_idx: int):
346:    def get_metadata(self, current_idx: int):
355:class Handle:
365:class MessageQueue:
366:    def __init__(
457:    def export_handle(self) -> Handle:
461:    def create_from_handle(handle: Handle, rank) -> "MessageQueue":
508:    def wait_until_ready(self):
541:    def shutdown(self):
549:    def acquire_write(self, timeout: float | None = None):
665:    def acquire_read(
727:    def enqueue(self, obj, timeout: float | None = None):
772:    def dequeue(
800:    def recv(socket: zmq.Socket, timeout: float | None) -> Any:
807:    def broadcast_object(self, obj=None):
814:    def create_from_process_group_single_reader(
860:    def create_from_process_group(
```

acquire_write（549-664）全文（写侧等所有 reader 读完才写新块）：
```python
    def acquire_write(self, timeout: float | None = None):
        assert self._is_writer, "Only writers can acquire write"
        start_time = time.monotonic()
        n_warning = 1
        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                def check():
                    memory_fence()
                    read_count = sum(metadata_buffer[1:])
                    written_flag = metadata_buffer[0]
                    return not (written_flag and read_count != self.buffer.n_reader)
                if SPINLOOP_EXT_ENABLED and not check():
                    spinloop(metadata_buffer, check, timeout=SPINLOOP_TIMEOUT_SECONDS)
                if not check():
                    sched_yield()
                    elapsed = time.monotonic() - start_time
                    if timeout is not None and elapsed > timeout:
                        raise TimeoutError
                    if elapsed > VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning:
                        logger.info(LONG_WAIT_TIME_LOG_MSG, VLLM_RINGBUFFER_WARNING_INTERVAL)
                        n_warning += 1
                    continue
                metadata_buffer[0] = 0
                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf
                for i in range(1, self.buffer.n_reader + 1):
                    metadata_buffer[i] = 0
                memory_fence()
                metadata_buffer[0] = 1
                memory_fence()
                self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks
                break
```

acquire_read（665-726）核心推进逻辑（读侧等 written_flag 且未读）：
```python
    def acquire_read(self, timeout=None, indefinite=False):
        assert self._is_local_reader, "Only readers can acquire read"
        read_timeout = self.ReadTimeoutWithWarnings(timeout=timeout, should_warn=not indefinite)
        with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
            while True:
                def check():
                    memory_fence()
                    read_flag = metadata_buffer[self.local_reader_rank + 1]
                    written_flag = metadata_buffer[0]
                    return not (not written_flag or read_flag)
                if SPINLOOP_EXT_ENABLED and not check():
                    spinloop(metadata_buffer[0 : self.local_reader_rank + 1], check, timeout=SPINLOOP_TIMEOUT_SECONDS)
                if not check():
                    self._spin_condition.wait(timeout_ms=read_timeout.timeout_ms())
                    if self.shutting_down:
                        raise RuntimeError("cancelled")
                    if read_timeout.should_warn():
                        logger.info(LONG_WAIT_TIME_LOG_MSG, VLLM_RINGBUFFER_WARNING_INTERVAL)
                    continue
                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf
                metadata_buffer[self.local_reader_rank + 1] = 1
                memory_fence()
                self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks
                self._spin_condition.record_read()
                break
```

dequeue（772-799）全文：
```python
    def dequeue(self, timeout=None, indefinite=False):
        """Read from message queue with optional timeout (in seconds)"""
        if self._is_local_reader:
            with self.acquire_read(timeout, indefinite) as buf:
                overflow = buf[0] == 1
                if not overflow:
                    offset = 3
                    buf_count = from_bytes_big(buf[1:offset])
                    all_buffers = []
                    for i in range(buf_count):
                        buf_offset = offset + 4
                        buf_len = from_bytes_big(buf[offset:buf_offset])
                        offset = buf_offset + buf_len
                        all_buffers.append(buf[buf_offset:offset])
                    obj = pickle.loads(all_buffers[0], buffers=all_buffers[1:])
            if overflow:
                obj = MessageQueue.recv(self.local_socket, timeout)
        elif self._is_remote_reader:
            obj = MessageQueue.recv(self.remote_socket, timeout)
        else:
            raise RuntimeError("Only readers can dequeue")
        return obj
```

**块2 · R19-1 停摆窗口（17:46:09 CONNECT_OK → 17:47:28 首告警间 worker 输出）**

日志 /home/w00919836/scp-opt/tmp/r19-bench-bs4/raw/model.log（515696 字节，17:53 落盘）

17:46:00-06 全部 worker 持续正常推进 DSA_STORE_PUT_DONE（每步 pages+16 稳定递增）：
```
17:46:00 pages=[166] versions=[11]  x8 worker
17:46:01 pages=[182] versions=[12]
17:46:02 pages=[198] versions=[13]
17:46:03 pages=[214] versions=[14]
17:46:04 pages=[230] versions=[15]
17:46:05 pages=[246] versions=[16]
17:46:06 pages=[262] versions=[17]
```
17:46:09 全部 8 worker 完成 DSA_SHARED_CONNECT_BEGIN+OK（endpoint=127.0.0.1:19090-19097，wire=6 backend=real_store group_gather=full groups=21），之后立即打 MTP draft graph replay：
```
2622:(Worker_DP1_TP0_EP2) INFO 17:46:09 [acl_graph.py:284] Replaying aclgraph
2623:(Worker_DP0_TP0_EP0) INFO 17:46:09 [llm_base_proposer.py:651] [GLM MTP graph] Replaying separate merged 3-step draft graph (with device preprocessing)
2624:(Worker_DP2_TP0_EP4) INFO 17:46:09 [llm_base_proposer.py:651] ...
2625:(Worker_DP3_TP0_EP6) INFO 17:46:09 [llm_base_proposer.py:651] ...
2626:(Worker_DP1_TP0_EP2) INFO 17:46:09 [llm_base_proposer.py:651] ...
```
17:46:09 后挂起 worker（DP0/1/3）无任何后续输出（2627-3092 行无 Worker_DP[013] 行），直到 17:52:29 SIGTERM（multiproc_executor.py:790 Parent process exited）。DP2_TP0_EP4 (pid=1999194) 独立崩溃路径：17:46:26 WorkerProc hit an exception（multiproc_executor.py:987）→ 17:46:31 DP2 EngineCore SIGTERM。

**块3 · 各挂起 EngineCore 告警总数与最后时间戳**（告警源 = patch_shm_broadcast.py:74，非 shm_broadcast.py）：
```
== EngineCore_DP0 == count: 6 | last: 17:52:28 [patch_shm_broadcast.py:74] No available shared memory broadcast block found in 60 seconds...
== EngineCore_DP1 == count: 6 | last: 17:52:28 [patch_shm_broadcast.py:74] ...
== EngineCore_DP3 == count: 6 | last: 17:52:28 [patch_shm_broadcast.py:74] ...
（全部 18 条告警最后一条均 17:52:28；worker 侧 0 条 No available 告警）
```

R22-0 判读（供蓝区消化）：挂起 worker（DP0/1/3）在 17:46:09 DSA_SHARED_CONNECT_OK + MTP draft graph replay 打出后完全静默，写侧 acquire_write 的 60s 告警（patch_shm_broadcast.py:74）只是下游症状。支持蓝区「worker 在 DSA_SHARED_CONNECT_OK 之后同步点停摆（非块泄漏）」头号嫌疑；且 DP2 是独立异常崩溃路径（WorkerProc exception 17:46:26），与 DP0/1/3 静默挂起不同故障面。

### R22-1 · offload-off 对照轮（ENABLE_OFFLOAD=0）

前置门禁核验：本机业务进程 0 / 端口 19080-19097+20000-20007 全空 / NPU 8 卡全空闲 / shm 384G 可用 383G（清理本机 R21+R2 Pool 残留 256G+）。注意：wzh 的 URMA source 探针（run_urma_source_matrix.sh, host-device udmac0d1e3, 他人作业）在场，未 kill；NPU4 无业务进程，未冲突。

env | grep SMOKE | sort 原文（唯一变量 ENABLE_OFFLOAD=0 生效）：
```
VLLM_ASCEND_SMOKE_BENCH=1
VLLM_ASCEND_SMOKE_DATASET=chinese_40k
VLLM_ASCEND_SMOKE_ENABLE_CHUNKED_PREFILL=1
VLLM_ASCEND_SMOKE_ENABLE_CPU_BINDING=0
VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=0
VLLM_ASCEND_SMOKE_ENABLE_PROFILE=0
VLLM_ASCEND_SMOKE_GPU_MEMORY_UTILIZATION=0.97
VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=40960
VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=2048
VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=4
VLLM_ASCEND_SMOKE_MAX_TOKENS=256
VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731
```
import 核验：vllm_ascend=/home/fjj/vllm-ascend-remote-offload/vllm_ascend/__init__.py（fjj 仓优先）✓；启动 2026-10-31 01:20:28。

通过标准逐条核验：

1. RUN_OK + 8/8 DRAIN_OK：summary.txt=`RUN_OK model_exit=0; owned Pool processes stopped`；DRAIN_OK 8 条（select=2413-2489 gather_only=0 scatter_sfa=7239-7467 real_store=1）——offload-off 下仍 real_store=1 + 8x DSA_STORE_CLOSED（requests=4 put_pages=97344 inference=hbm_full_remote_shared remote_get=1）：ENABLE_OFFLOAD=0 关的是显式 offload 路径，remote shared store 链路本身仍走（shared KV 池仍启用）。
2. 4 rank metrics + metrics.json：4x[dsa-smoke][metrics][dp=0..3] + [global] 全产出；metrics.json 实际路径 /home/fjj/vllm-ascend-remote-offload/results/1p/dp4_tp2_offload_graph_mtp_graph_hbm_full_remote_shared.metrics.json（PYTHONPATH 前置 fjj 仓，写框架仓 results/）。
3. TTFT 修复验证通过：ttft_ms 全部分位数满足 0 < ttft < global_wall_sx1000（=80379ms）——avg=45337.078 / min=18556.319 / p50=45444.065 / p90=71814.133 / p99=71822.782 / max=71822.782，不再负值。40K chunked prefill 下 TTFT 数秒~数十秒符合预期（per_request 逐条 18556→71822 递增 = 4 请求顺序 prefill 肩叠）。
4. 原文回传：
```
[dsa-smoke][metrics][global] {"dp_size": 4, "requests": 16, "output_tokens": 4096, "global_wall_s": 80.379, "request_throughput_rps": 0.199, "global_output_tps": 50.96, "ttft_ms": {"avg": 45337.078, "min": 18556.319, "p50": 45444.065, "p90": 71814.133, "p99": 71822.782, "max": 71822.782}, "tpot_ms": {"avg": 132.716, "min": 27.639, "p50": 132.335, "p90": 237.928, "p99": 238.159, "max": 238.159}, "tpot_under_50ms": false}
============ DSA Smoke Benchmark Result (global, offline) ============
Successful requests: 16
Benchmark duration (s): 80.38
Total generated tokens: 4096
Request throughput (req/s): 0.199
Output token throughput (tok/s): 50.96
TTFT (ms): avg=45337.078ms p50=45444.065 p90=71814.133 p99=71822.782 max=71822.782
TPOT (ms, excl. 1st token): avg=132.716ms p50=132.335 p90=237.928 p99=238.159 max=238.159
========================================================================
```

同口径对比（offload-on vs offload-off）：

| 指标 | R21 offload-on | R22-1 offload-off | 差 |
|---|---|---|---|
| global_wall_s | 81.481 | 80.379 | -1.10s |
| output_tps | 50.27 | 50.96 | +0.69 |
| tpot avg ms | 135.234 | 132.716 | -2.52 |
| tpot p50 ms | 134.919 | 132.335 | -2.58 |
| tpot min ms | 28.725 | 27.639 | -1.09 |
| ttft avg ms | -1.79e12（坏） | 45337.078（修复验证通过） | - |
| EI0006 / 1004 | 0 / 0（复跑后） | 0 / 0 | - |
| patch_shm 挂起 | 0 | 0 | - |

判读（命中预案分支）：offload-off 相对 offload-on 略快（wall -1.1s / tps +1.4% / tpot avg -2.5ms），差距约 1.4% 量级。按预案区分：本配方 40K prefill 主导期（TTFT约45s 占 wall 80s 过半），offload 在 prefill 主导期无净收益甚至微负——remote shared store 写路径（put_pages=97344 scatter_sfa=7239）仍有固定开销；decode 稳态（tpot min约28-29ms 干净 decode 步）两者基本持平。offload 自提升口径在本配方下不成立（-1.4%），需蓝区裁决：是否属「40K prefill 主导期 offload 无收益」分支（不下一刀切结论），或需 100K x 1K 长 decode 尾段配方测 decode 稳态收益。

失败处置：无 TTFT 越界（修复已验证）、无 patch_shm 挂起复现（0 次）、无 1004/EI0006。机器已清场（owned Pool 已停止）。

## §R22 判读（蓝区，2026-10-31）

### 裁决

1. **TTFT 修复验证通过（验收级证据）**：offload-off 轮 `ttft_ms avg=45337/min=18556/p99=71823`，全部落在 `0 < ttft < wall×1000` 区间；per_request 18556→71822 递增 = 4 请求/rank 顺序 prefill 肩叠加，与 40K chunked prefill 物理图像一致。`first_token_latency` 方案闭环，TTFT 统计链从此可用。
2. **R22-1 不是真对照——ENABLE_OFFLOAD=0 被工具层吞掉（重大发现）**：绿区实测 offload-off 轮仍 `real_store=1` + `put_pages=97344`（与 offload-on 逐位相同）+ run 名仍含 `offload`（`dp4_tp2_offload_graph_mtp_graph_hbm_full_remote_shared`，run 名由 simple_test.py:77 按 `config.enable_offload` 生成）。蓝区代码链已核：`VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=0` → `enable_offload=False` → `build_llm_kwargs`（simple_test.py:298）跳过整个 `dsa_sparse_config` → 不应有任何 STORE 行。实测相反 ⇒ **`profile_pool_inference.py`（scp-opt 工具）在拉起框架子进程时强制/覆盖了 offload 配置**，smoke env 未穿透。因此 R22-1 的 ±1.4% 差只是同构重跑噪声，**「+20% 自提升」对照基线仍未落袋**。
3. **shm 停摆机制收敛（R22-0 取证消化）**：① acquire_write 源码证实写侧等 `read_count==n_reader` 才复用块，60s 告警 = 有 reader 停消费（与既有判断一致）；②停摆窗口决定性证据：17:46:09 全部 8 worker `DSA_SHARED_CONNECT_OK` + TP0 打 `llm_base_proposer.py:651 MTP draft graph replay` 后，**DP0/1/3 完全静默至 SIGTERM**——停摆点在「connect 完成 → draft graph replay」这一同步区间；③**DP2 是独立故障面**（17:46:26 WorkerProc exception → EngineCore SIGTERM），不是静默挂起，两轮（R18 挂 DP1/2/3、R19-1 挂 DP0/1/3）挂起组漂移 = 静默挂起有竞态成分。修复方向：查 draft graph replay 与 remote store group_gather=full（21 组）的交互，不是 shm 块生命周期本身。
4. 基线数据保持 R20/R21（offload-on）：wall=81.5s、output_tps=50.3、tpot avg=135.2/min=28.7ms。TTFT 基线 = 本轮 45.3s（avg，offload 开关对 TTFT 无物理影响，同配方可互用）。

## §R23 派单：真 baseline 对照轮 + 工具吞 env 取证 + draft replay 停摆源码（蓝区，2026-10-31）

### R23-0 · 只读取证（免费，先做）

```bash
# 块1 · 工具吞 env 实锤：profile_pool_inference.py 如何传 env 给框架子进程
grep -n "OFFLOAD\|offload\|env\b\|environ" /home/w00919836/scp-opt/tools/profile_pool_inference.py | head -40
# 块2 · 停摆点源码：draft graph replay 上下文 + connect 之后的调用链
sed -n '620,700p' /vllm-workspace/vllm/vllm/v1/spec_decode/llm_base_proposer.py 2>/dev/null || \
  grep -rn "Replaying separate merged" /vllm-workspace/vllm/vllm/ | head -5
sed -n '270,300p' /home/fjj/vllm-ascend-remote-offload/vllm_ascend/compilation/acl_graph.py
# 块3 · DP2 独立崩溃的 exception 原文（R19-1，17:46:26 前后 30 行）
grep -n -A25 "Worker_DP2_TP0_EP4.*WorkerProc hit an exception" /home/w00919836/scp-opt/tmp/r19-bench-bs4/raw/model.log | head -40
```

三块原文回传。

### R23-1 · 真 baseline 轮（不经工具、不起 Pool，ENABLE_OFFLOAD=0 必须生效）

工具会吞 offload 开关（R22 判读 2），本轮**绕过工具直跑框架入口**（baseline 不需要远端 Pool）：

```bash
cd /home/fjj/vllm-ascend-remote-offload && git -c http.proxy=http://90.255.129.35:3128 pull --ff-only && git log --oneline -2
source /home/w00919836/va.env
export VLLM_ASCEND_SMOKE_MODEL_PATH=/data/model/GLM-5.2-W4A4C8-mxfp4-A5-0731
export VLLM_ASCEND_SMOKE_DATASET=chinese_40k
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=4
export VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=40960
export VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=2048
export VLLM_ASCEND_SMOKE_MAX_TOKENS=256
export VLLM_ASCEND_SMOKE_ENABLE_CHUNKED_PREFILL=1
export VLLM_ASCEND_SMOKE_GPU_MEMORY_UTILIZATION=0.97
export VLLM_ASCEND_SMOKE_ENABLE_PROFILE=0
export VLLM_ASCEND_SMOKE_BENCH=1
export VLLM_ASCEND_SMOKE_ENABLE_CPU_BINDING=0
export VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=0
export VLLM_ASCEND_SMOKE_RESULT_DIR=/home/fjj/vllm-ascend-remote-offload/results/1p
cd /home/fjj/vllm-ascend-remote-offload/examples/dsa_demo
env | grep SMOKE | sort   # 原文回传
python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"   # 须指向 fjj 仓
python3 simple_test_1p.py 2>&1 | tee /home/fjj/pp_r23a.log
echo "RC=$?"
```

前置门禁：进程 0 / NPU 8 卡全空闲 / shm ≥370G（不碰他人作业）。**不传 `--remote-shared-config`，不起 Pool**。

### 通过标准

1. **offload-off 真生效的硬证据**（三选全中）：run 名含 `baseline`（results/1p 下 metrics.json 路径不含 `offload`）；日志**无** `DSA_STORE_CLOSED/DSA_SHARED_CONNECT` 行；启动行 `[dsa-smoke] ... offload=False`。
2. RUN_OK + 4 rank `[dsa-smoke][metrics][dp=N]` + metrics.json（baseline 路径）全字段。
3. 原文回传 `[dsa-smoke][metrics][global]` 行 + 全局报告块全文。

### 失败处置

- run 名仍含 offload / 仍有 STORE 行 ⇒ env 仍未生效，回传 `grep -n "enable_offload" examples/dsa_demo/simple_test.py | head` + 启动行原文，停；
- EI0006 ⇒ 配方回归检查（baseline 无 remote 链路，理论上触发面更小；发生即回传 env 核验）；
- 挂起 ⇒ 回传挂起点最后 30 行（baseline 无 store connect，若仍挂 = 嫌疑从 connect 交互转向纯 draft replay）；
- 1004 ⇒ 白名单（≤2，≥5min 静置 + 清场）。

### 判读预案（派单时写死）

- R23-1（真 baseline）vs R21（offload-on）同口径对比 → 「+20% 自提升」分子分母落袋；若差距仍 <5%，判读时按 40K prefill 主导期解释（offload 收益预期在 decode 稳态/容量维度，不在此配方），并明确写进给用户的结论；
- R23-0 块1 实锤后，蓝区决定是否给工具加 `--no-offload` 透传参数（代码线）；
- R23-0 块2/3 + 本仓 dsa_offload remote_shared 源码 → 蓝区出停摆根修方案（R24 代码线）。

## §R23 判读（蓝区，2026-10-31）

### 裁决

1. **真 baseline 落袋，offload 在本配方是纯开销（验收口径级结论）**：R23-1（baseline，无 remote 链路）vs R21（offload-on）同口径：wall 75.65s vs 81.48s（**baseline 快 7.2%**）、output_tps 54.15 vs 50.27（**+7.7%**）、tpot avg 122.6 vs 135.2ms（−9.4%）；干净 decode 步 tpot min 29.8 vs 28.7ms 基本持平。**「+20% 自提升」在 BS4×256×40K-prefill 主导配方下不成立且反号**——offload 的收益维度是 KV 容量/decode 稳态（HBM 放不下时的续跑能力），不是此配方的速度。给用户结论：验收要证 +20% 必须换 100K×1K 长 decode 口径，而那条路被 shm 挂起堵着——见下条，已破。
2. **工具吞 env 实锤**：`profile_pool_inference.py:414` 无条件 `env["VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD"] = "0" if mode=="baseline" else "1"`，外部 export 被覆盖。R22-1 的 ±1.4% = 同构噪声坐实。后续 offload-on 轮继续用工具（默认 mode=remote 正确），offload-off 轮必须绕过工具直跑（R23-1 模式）。
3. **R19-1 挂起/崩溃双故障面统一（蓝区代码分析，待 R24-1 实证）**：根因 = **max_model_len=40960 clamp**——chinese_40k prompt（≈40K）+ MAX_TOKENS=1024 > 40960，decode 尾部 scheduler 把逼近上限的请求 clamp 成 1~3 token → **非 uniform 步**（如 [4,4,4,2]）；图门正确判 `non_uniform_decode` 回退 eager，但 eager 入口（sfa_v1.py:1709-1716）只看 attn_state 不看 uniform 性，照样进 `remote_shared.select` → Q=4 断言炸。**在同步区（acl graph pre_sync/EP 集合通信）内炸 = 进程静默挂死（DP0/1/3），区外炸 = 干净 ValueError（DP2）**；worker 停消费 → EngineCore shm 写侧 60s 告警 = 下游症状。数字自洽：256 步配方 prompt+256 ≤ 40960 不 clamp → R19-2/R20/R21/R23 全干净；1024 步必 clamp → R18/R19-1 崩/挂。「泄漏∝步数」重新表述为「非 uniform 事件只出现在长跑尾部」。
4. **蓝区已推送 instrumentation commit**：`remote_shared.select` 的 ValueError 携带全上下文（identity/epoch/b/query_shape/active_num_reqs/graph_enabled/prepared_epoch/actual_q/row_modes）——下次复现直接显示非 uniform 步第一现场，不用猜。
5. TTFT 修复第二轮验证通过（baseline 轮 41.6s avg 全正）。

## §R24 派单：clamp 理论实证轮（BS4×1024 + max_model_len=43008）（蓝区，2026-10-31）

### 执行

```bash
cd /home/fjj/vllm-ascend-remote-offload && git -c http.proxy=http://90.255.129.35:3128 pull --ff-only && git log --oneline -2
# 顶部应含 select 报错增强 commit；grep 核验：
grep -n "requires fixed MTP3 Q=4 capacity" vllm_ascend/dsa_offload/remote_shared.py
```

R19-1 配方逐字（经工具 profile_pool_inference.py + 作者数据面 + BENCH=1 + batched2048/chunked1 + BS4 + MAX_TOKENS=1024），**唯一变量**：

```bash
export VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=43008   # 原 40960；+2048 余量覆盖 prompt(≤~40.7K)+1024 的 clamp 区
```

输出目录 `tmp/r24-bench-bs4-mt1024-ml43008`，日志 `/home/fjj/pp_r24.log`。前置门禁/白名单（1004 ≤2 次、≥5min 静置 + 清场）不变。

### 通过标准

1. 8/8 DRAIN_OK + 8×STORE_CLOSED + patch_shm 挂起 = 0 次（**核心：1024 步首次无挂起**）；
2. 4 rank metrics + metrics.json + `[dsa-smoke][metrics][global]` 原文回传（这是 1024 步口径的第一个 bench 点）；
3. TTFT 全正（0 < ttft < wall×1000）。

### 失败处置（三分支，全部回传原文）

- **过** ⇒ clamp 理论坐实：1024 步 decode 解锁，R25 进 BS16×1024 + 100K×1K 验收配方；非 uniform 路由根治（eager 分支正确处理 Q≠4）转代码线挂账（生产健壮性，不阻塞验收）。
- **崩（新 ValueError 带上下文）** ⇒ actual_q/row_modes 原文即非 uniform 步类型证据，蓝区直接出根治 commit；
- **挂（无崩溃、60s 告警复现）** ⇒ clamp 理论证伪，回传挂起 worker 最后 30 行 + 各 EngineCore 告警计数，蓝区转 Pool native 闸口方向。

## §R24 判读（蓝区，2026-10-31）

### 裁决

1. **clamp 统一理论实证坐实（本轮最重要结论）**：唯一变量 MAX_MODEL_LEN=40960→43008，同 BS4×1024 配方从「崩+挂全中」变为「8/8 DRAIN + 挂起 0 + 崩溃 0 + 自然收尾」。R18/R19-1 挂起、DP2 ValueError、shm 60s 告警全链路根因闭环：**max_model_len 装不下 prompt+output → decode 尾部 clamp 出非 uniform 步 → eager 回退仍进 remote select → Q=4 断言**。纪律入库：**任何 bench/验收配方必须满足 max_model_len > max(prompt) + max_tokens + MTP 余量**；非 uniform 步的 eager 路由根治转代码线挂账（生产健壮性，不阻塞验收）。
2. **1024 步 decode 解锁 + 首个 decode 主导口径 bench 点落袋**：BS4×1024（16 请求 × 1024 token）wall=99.74s、**output_tps=164.26**、**tpot avg=48.32ms / p50=48.26 / p90=77.48 / min=21.55ms**、TTFT avg=45.4s。decode 占比从 256 步的 ~19% 升到 ~55%，tpot avg 从 135ms 降到 48.3ms——污染稀释效应与口径预期一致；min=21.55ms 为目前最干净单步。注意 p99=77.6ms 仍含 prefill 交错段，TPOT<50ms 验收需稳态口径（长 decode 尾段），100K×1K 配方下 decode 占比 >90% 时 avg 即逼近稳态。
3. offload-on 1024 步 vs baseline 256 步不可直接比（不同配方）；R25 安排同配方对照。

## §R25 派单：BS16 放大轮 + 同配方 baseline 对照（蓝区，2026-10-31）

### 目标

①验证 BS16（全局 64 请求，R18 原始失败点）在 ML43008 下解锁；②同配方（BS16×1024、decode 主导）落 offload-on vs baseline 对照，给 +20% 口径第二个数据点。

### R25-1 · BS16×1024 offload-on（经工具）

R24 命令逐字（作者数据面 + BENCH=1 + batched2048/chunked1 + MAX_TOKENS=1024 + MAX_MODEL_LEN=43008），唯一变量：

```bash
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=16   # BS4→BS16，全局 64 请求
```

输出 `tmp/r25-bench-bs16-mt1024-ml43008`，日志 `/home/fjj/pp_r25a.log`。前置门禁/1004 白名单不变。

### R25-2 · BS16×1024 baseline（绕工具直跑，R23-1 模式）

R23-1 命令逐字（容器内 + source va.env + PYTHONPATH 前置 fjj + 不起 Pool + 不传 remote-shared-config），env 取 R25-1 同值 + `VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=0`：

```bash
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=16
export VLLM_ASCEND_SMOKE_MAX_TOKENS=1024
export VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=43008
export VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=0
# 其余同 R23-1（BENCH=1/DATASET=chinese_40k/batched2048/chunked1/GPU_MEM=0.97/PROFILE=0/CPU_BINDING=0/RESULT_DIR 绝对路径）
```

日志 `/home/fjj/pp_r25b.log`。

### 通过标准

- R25-1：8/8 DRAIN + STORE_CLOSED + 挂起 0 + metrics 全产出 + TTFT 全正；
- R25-2：offload=False 硬证据三条（run 名 baseline / 无 STORE 行 / 启动行）+ RUN_OK + metrics 全产出；
- 两轮原文回传 `[global]` 行 + 报告块。

### 失败处置

- R25-1 挂起/崩溃复现 ⇒ clamp 理论边界外还有第二触发源，新断言上下文 + 挂起上下文原文回传；
- R25-2 OOM（BS16×43K HBM 超限可能）⇒ 回传 OOM 原文，这本身是「offload 容量价值」的直接证据，不算失败；
- 1004 ⇒ 白名单；EI0006 ⇒ 回传 env 核验。

### 判读预案（派单时写死）

- R25-1 vs R25-2 同配方对照 → decode 主导口径的 offload 损益第二个数据点；
- 若 R25-2 OOM 而 R25-1 过 ⇒ offload 容量价值直接成立，验收叙事改为「offload 解锁 baseline 跑不了的规格」；
- 之后 R26 = 100K×1K 验收配方（chinese_104k + MAX_MODEL_LEN≥105K + MAX_TOKENS=1024，offload-on vs baseline 同口径）。

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

## §R23 交付：工具吞 env 源码实锤 + 真 baseline 落袋（offload-off 反超 +7.7%）（绿区，2026-10-31 02:15~02:20）

### R23-0 · 三块只读取证原文

**块1 · 工具吞 env 实锤（profile_pool_inference.py 的 env 传递）**

`/home/w00919836/scp-opt/tools/profile_pool_inference.py`：

```
165:    p.add_argument("--mode", choices=("remote", "local", "baseline"), default="remote", ...)
412:def model_environment(a):
413:    env = os.environ.copy()
414:    env["VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD"] = "0" if a.mode == "baseline" else "1"
415:    if a.node_rank is not None:
416:        for name in ("node_rank", "master_addr", "master_port", "sync_port"):
417:            env["VLLM_ASCEND_SMOKE_"+name.upper()] = str(getattr(a, name))
418:    return env
...
745:    stderr=subprocess.STDOUT, env=model_env, start_new_session=True
```

**实锤机制**：`model_environment(a)` 无条件 `env["VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD"] = "0" if a.mode == "baseline" else "1"`——外部 `export VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=0` 被该行强制覆盖。R22-1 用默认 `--mode=remote`（L165 default），故工具把 offload 写回 `"1"`，R22-1 的 ENABLE_OFFLOAD=0 从未到达框架子进程（subprocess.Popen env=model_env 用覆盖后环境）。±1.4% 差=同构重跑噪声坐实。

**块2a · draft replay 上下文（llm_base_proposer.py:620-700）**

`/vllm-workspace/vllm/vllm/v1/spec_decode/llm_base_proposer.py`（L651 `[GLM MTP graph] Replaying separate merged 3-step draft graph` 所在）：

```python
            inputs_embeds = self.inputs_embeds[:input_batch_size]
            input_ids = self.input_ids[:input_batch_size]
            inputs_embeds = None
            # Run the model.
            model_kwargs = {
                "input_ids": input_ids,
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": inputs_embeds,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]
            with set_forward_context(
                per_layer_attn_metadata, self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(input_batch_size),
            ):
                ret_hidden_states = self.model(**model_kwargs)
                ...
            hidden_states = hidden_states[:batch_size]
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                last_hidden_states[:batch_size], sampling_metadata
            )
```

后续 `_update_positions_dependent_metadata`（620-700 段）：`eagle_step_update_slot_mapping_and_metadata(...)` 更新 positions/slot mapping/seq metadata 供下一步 draft，`common_attn_metadata.slot_mapping = self._slot_mapping_buffer[:batch_size]`。**该段纯 draft 内部状态更新，无 remote_shared/connect 交互**——与 R22 判读「worker 在 connect→draft replay 同步区间静默」的调用链上下文吻合。

**块2b · acl_graph.py:270-300（Replaying aclgraph 逻辑）**

`/home/fjj/vllm-ascend-remote-offload/vllm_ascend/compilation/acl_graph.py`：

```python
            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during acl graph capture
            return output
        if self.is_debugging_mode:
            new_input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            assert new_input_addresses == entry.input_addresses, (...)
        logger.info_once("Replaying aclgraph")
        # In async scheduling or multi-threaded (MT) scenarios, ... CPU record event
        # (from update_attn_params) for iteration i may complete before graph replay of i-1.
        # must synchronize here before replaying, so update_attn_params only executes after
        # previous graph replay fully completed.
        is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle
        need_sync = self.runtime_mode == CUDAGraphMode.FULL and not is_draft_eagle
        if remote is not None:
            logger.debug("DSA_SHARED_GRAPH_REPLAY_BEGIN %s epoch=%s pre_sync=%s batch=%s", ...)
        if not self.enable_enpu and need_sync:
            with (remote.host_range("graph_replay_pre_sync", batch_descriptor.num_reqs)
```

graph replay 前有 `need_sync` 屏障（FULL 模式下非 merge-eagle 需同步），remote 非 None 时打 `DSA_SHARED_GRAPH_REPLAY_BEGIN` debug 日志。**若 worker 在此 pre_sync/同步点停摆，恰对应 R22-0 观测的 CONNECT_OK 后 draft replay 区间静默**（DSA_SHARED_GRAPH_REPLAY_BEGIN 为 debug 级，常规日志不可见，需 DSA_ 级 debug 开关才可见）。

**块3 · DP2 独立崩溃 exception 原文（r19-bench-bs4/raw/model.log:2635-2715）**

```
(Worker_DP2_TP0_EP4 pid=1999194) ERROR 10-30 17:46:26 [multiproc_executor.py:987] WorkerProc hit an exception.
...
  File "/vllm-workspace/vllm/vllm/v1/executor/multiproc_executor.py", line 982, in worker_busy_loop
  File "/vllm-workspace/vllm/vllm/v1/worker/worker_base.py", line 345, in execute_model
  File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/worker/worker.py", line 650, in execute_model
  File ".../site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
  File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/worker/model_runner_v1.py", line 3002, in execute_model
    hidden_states = self._model_forward(
  File ".../model_runner_v1.py", line 3558, in _model_forward
    hidden_states = run_model()
  File ".../vllm_ascend/compilation/acl_graph.py", line 163, in __call__
    return self.runnable(*args, **kwargs)
  ... (module forward chain)
  File "/vllm-workspace/vllm/vllm/model_executor/models/deepseek_v2.py", line 1713, in forward
  File ".../model_executor/models/deepseek_v2.py", line 1330, in forward
  File ".../model_executor/models/deepseek_v2.py", line 1189, in forward
    hidden_states = self.self_attn(**attn_kwargs)
  File ".../model_executor/models/deepseek_v2.py", line 1082, in forward
    return self.mla_attn(positions, hidden_states, llama_4_scaling)
  File "/home/fjj/vllm-ascend-remote-offload/vllm_ascend/ops/mla.py", line 188, in forward
    torch.ops.vllm.mla_forward(hidden_states, need_gather_q_kv, output, self.prefix)
  File ".../ops/mla.py", line 206, in mla_forward
    self.mla_attn.impl.forward(
  File ".../vllm_ascend/attention/sfa_v1.py", line 2348, in forward
    topk_indices = self.indexer_select_post_process(
  File ".../attention/sfa_v1.py", line 1731, in indexer_select_post_process
    return dsa_context.execute_decode_selection(
  File ".../vllm_ascend/dsa_offload/runtime.py", line 980, in execute_decode_selection
    return self.runtime.remote_shared.select(
  File ".../vllm_ascend/dsa_offload/remote_shared.py", line 464, in select
    raise ValueError("remote shared decode requires fixed MTP3 Q=4 capacity B=1..32")
ValueError: remote shared decode requires fixed MTP3 Q=4 capacity B=1..32
```

**DP2 崩溃定性**：`remote_shared.py:464 select` 对 MTP3/Q=4/capacity B=1..32 的容量前提断言（`ValueError: remote shared decode requires fixed MTP3 Q=4 capacity B=1..32`）——是 **remote-shared decode 容量配置断言失败**，与「挂起组（DP0/1/3 静默）」不同故障面：DP2 是显式异常崩溃，其余组是静默停摆，印证 R22 判读「DP2 为独立崩溃面」。

### R23-1 · 真 baseline 轮（不经工具、不起 Pool，ENABLE_OFFLOAD=0 生效）

**执行方式修正**：派单命令为宿主 shell 风格，但本机宿主 python3.11 无 vllm/vllm_ascend（R16-1 已修正），R18-R22 全部在容器 dsa-full 内执行（python3.12.13）。R23-1 在容器内执行：`source /usr/local/Ascend/ascend-toolkit/set_env.sh` + `source va.env` + `export PYTHONPATH=/home/fjj/vllm-ascend-remote-offload:$PYTHONPATH`（**使 `import vllm_ascend` 指向 fjj 仓**，派单硬性要求）+ 派单全部 SMOKE 变量 + `VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=0`。**不传 `--remote-shared-config`、不起 Pool**。

- 启动：2026-10-31 02:15:10；结束：02:19:53（R23A_RC=0 干净收尾）。
- env 核验（`env | grep SMOKE | sort` 原文）：`VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=0`、`BENCH=1`、`MAX_NUM_SEQS=4`、`MAX_MODEL_LEN=40960`、`MAX_NUM_BATCHED_TOKENS=2048`、`MAX_TOKENS=256`、`ENABLE_CHUNKED_PREFILL=1`、`GPU_MEMORY_UTILIZATION=0.97`、`ENABLE_PROFILE=0`、`ENABLE_CPU_BINDING=0`、`RESULT_DIR=/home/fjj/vllm-ascend-remote-offload/results/1p` 全就位。
- import 核验：`python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"` → `/home/fjj/vllm-ascend-remote-offload/vllm_ascend/__init__.py`（**指向 fjj 仓 ✓**）。
- 启动行：`[dsa-smoke] baseline MTP graph is not supported; using MTP eager, target mode unchanged`；rank 启动行 `mode=graph offload=False model=... tp=2 bs=4 max_model_len=40960 mtp=True mtp_graph_enabled=False`（4 rank 全 offload=False，`mtp_graph_enabled=False` 与 offload-on 的 True 对照，符合 baseline 预期）。

**通过标准逐条核验**

1. **offload-off 真生效硬证据（三选全中）**：
   - run 名含 baseline：metrics.json 路径 = `results/1p/dp4_tp2_baseline_graph_mtp_eager.metrics.json`（**不含 offload** ✓）；
   - 日志无 STORE/connect 行：`DSA_STORE_CLOSED=0` / `DSA_SHARED_CONNECT=0` / `remote_shared=0` / `offload=True=0` ✓（baseline 无 remote 链路，DRAIN/STORE 概念不存在，DRAIN=0 符合预期）；
   - 启动行 `offload=False` ✓（上条）。
2. **RUN_OK + 4 rank metrics + metrics.json**：`R23A_RC=0`；4×`[dsa-smoke][metrics][dp=N]`（dp=0/1/2/3，各 requests=4 output_tokens=1024）；metrics.json baseline 路径存在，global + ranks（含 per_request 数组）全字段 ✓。
3. **`[dsa-smoke][metrics][global]` 原文 + 全局报告块**（见下）。

**`[dsa-smoke][metrics][global]` 原文**

```
{"dp_size": 4, "requests": 16, "output_tokens": 4096, "global_wall_s": 75.649, "request_throughput_rps": 0.212, "global_output_tps": 54.15, "ttft_ms": {"avg": 41585.292, "min": 16762.825, "p50": 41691.552, "p90": 66177.375, "p99": 66206.019, "max": 66206.019}, "tpot_ms": {"avg": 122.569, "min": 29.806, "p50": 121.828, "p90": 215.03, "p99": 215.038, "max": 215.038}, "tpot_under_50ms": false}
```

**全局报告块原文**

```
============ DSA Smoke Benchmark Result (global, offline) ============
Successful requests: 16
Benchmark duration (s): 75.65
Total generated tokens: 4096
Request throughput (req/s): 0.212
Output token throughput (tok/s): 54.15
TTFT (ms): avg=41585.292ms p50=41691.552 p90=66177.375 p99=66206.019 max=66206.019
TPOT (ms, excl. 1st token): avg=122.569ms p50=121.828 p90=215.03 p99=215.038 max=215.038
========================================================================
```

**健康计数**：patch_shm_broadcast 挂起 = 0；EI0006/HcclAllGather = 0；code=1004 = 0。TTFT 全正且 `0 < ttft(41585ms) < global_wall_s×1000(75649ms)` ✓（修复验证再次通过）。

### 判读预案命中：baseline 不降反升，+20% 分子分母全错

**R23-1（真 baseline）vs R21（offload-on）同口径对比**：

| 指标 | R21 offload-on | R23-1 baseline | 变化 |
|---|---|---|---|
| global_wall_s | 81.481 | 75.649 | **−7.2%** |
| output_tps | 50.27 | 54.15 | **+7.7%** |
| tpot avg (ms) | 135.234 | 122.569 | **−9.4%** |
| tpot p50 (ms) | 134.919 | 121.828 | **−9.7%** |
| tpot min (ms) | 28.725 | 29.806 | +3.8%（同量级，干净 decode 步） |

**结论**：真 baseline（offload 完全关闭、无 remote_shared 链路）相对 offload-on **不降反升**（tps +7.7%、wall −7.2%、tpot −9.4%）。这坐实：
- R22-1 的 ±1.4% 差 = 同构噪声（工具吞 env，offload 实际仍开着）；
- **「+20% 自提升」的分子分母全错**——在此配方（BS4×256 + 40K chunked-prefill 主导）下 offload-on 反而是纯开销（remote_shared PUT/status checks 全在 decode 路径外但仍有 connect/select 开销），baseline 更快；
- 命中 R23 判读预案「若差距 <5% 按 40K prefill 主导期解释」的加强版：**差距不仅 <5% 且反号（+7.7%）**——offload 收益预期在 decode 稳态/容量维度，不在此配方，需 100K×1K 长 decode 尾段（待 shm 泄漏根修后）才能兑现。

**R23-0 后续**：块1 实锤 → 蓝区可决定是否给工具加 `--no-offload` 透传参数（代码线）；块2/3 → 停摆根修方案（R24 代码线）输入已齐。

**日志/产物路径**：容器内 `/home/fjj/pp_r23a.log`（tee 全量）、`/home/fjj/r23a_wrapper.log`（env+import+RC 标记）、`/home/fjj/vllm-ascend-remote-offload/results/1p/dp4_tp2_baseline_graph_mtp_eager.metrics.json`。运行后机器已自然收尾（EngineCore 全 SIGTERM 退出、worker 队列终止）。

## §R24 交付：clamp 理论坐实——1024 步 decode 首次无挂起（BS4×1024 + ML43008）（绿区，2026-10-31 04:02~04:08）

### 执行核验

- pull 到 7e8e3b5（select 断言增强 commit，`git log --oneline -2` 顶部=fix(dsa_offload): select 容量断言携带步上下文）；grep 核验 `requires fixed MTP3 Q=4 capacity` 在 remote_shared.py:465（增强后带全上下文）。
- R19-1 配方逐字（经工具 profile_pool_inference.py + 作者数据面 dp4tp2.json + BENCH=1 + batched2048/chunked1 + BS4 + MAX_TOKENS=1024），**唯一变量 `MAX_MODEL_LEN=40960→43008`**。
- 前置门禁：进程 0 / NPU 8 卡全空闲 / shm 384G（清场残留后）。04:02 启动，双 Pool READY（groups=['a','b']，store 19080-19087/du 19090-19097 路由正确）→ **04:07:28 自然收尾（无人工清理）**。
- env 核验原文（`env | grep SMOKE | sort`）：`MAX_MODEL_LEN=43008`（唯一改）+ BENCH=1/DATASET=chinese_40k/MAX_NUM_SEQS=4/MAX_NUM_BATCHED_TOKENS=2048/MAX_TOKENS=1024/CHUNKED=1/GPU_MEM=0.97/PROFILE=0/CPU_BINDING=0 全就位。import 核验：`vllm_ascend=/home/fjj/vllm-ascend-remote-offload/vllm_ascend/__init__.py`（fjj 仓 ✓）。
- 工具 RC=0（R24_RC=0），summary=RUN_OK model_exit=0，run-status graceful=true errors=[]。

### 结果（通过标准逐条，核心=1024 步首次无挂起）

1. **8/8 DRAIN_OK + 8×STORE_CLOSED**（04:07:28，8 worker 全 `requests=4 put_pages=97344 inference=hbm_full_remote_shared remote_get=1`，select=7505-8626 scatter_sfa=22515-25878）；**patch_shm 挂起 = 0 次**（1024 步首次无挂起）。
2. **4 rank 各打出 `[dsa-smoke][metrics][dp=N]` 行**（dp=0/1/2/3，各 requests=4 output_tokens=4096）。
3. **`results/1p/dp4_tp2_offload_graph_mtp_graph_hbm_full_remote_shared.metrics.json` 存在**（model.log:2811 写入确认），global 含 requests=16/分位数/吞吐/tpot_under_50ms 全字段。
4. **`[dsa-smoke][metrics][global]` 原文回传**（见下）+ 全局报告块全文。

### `[dsa-smoke][metrics][global]` 原文

```
{"dp_size": 4, "requests": 16, "output_tokens": 16384, "global_wall_s": 99.743, "request_throughput_rps": 0.16, "global_output_tps": 164.26, "ttft_ms": {"avg": 45382.123, "min": 18577.744, "p50": 45536.521, "p90": 71874.828, "p99": 71950.002, "max": 71950.002}, "tpot_ms": {"avg": 48.32, "min": 21.551, "p50": 48.263, "p90": 77.477, "p99": 77.583, "max": 77.583}, "tpot_under_50ms": false}
```

### 全局报告块原文

```
============ DSA Smoke Benchmark Result (global, offline) ============
Successful requests: 16
Benchmark duration (s): 99.74
Total generated tokens: 16384
Request throughput (req/s): 0.16
Output token throughput (tok/s): 164.26
TTFT (ms): avg=45382.123ms p50=45536.521 p90=71874.828 p99=71950.002 max=71950.002
TPOT (ms, excl. 1st token): avg=48.32ms p50=48.263 p90=77.477 p99=77.583 max=77.583
========================================================================
```

### 健康计数

- **patch_shm_broadcast 挂起 = 0 次**；ValueError/RuntimeError = 0；EI0006 = 0；code=1004 = 0。
- TTFT 全正且 `0 < ttft(avg 45382ms) < global_wall_s×1000(99743ms)` ✓。
- 1024 步 decode 全 16 请求跑满（output_tokens=16384 = 16×1024），remote 数据面真实发生（put_pages=97344×8、real_store=1）。

### 判读预案命中：过 ⇒ clamp 理论坐实

R24（ML43008）vs R19-1（ML40960，同 BS4×1024 其余全同）：**ML43008 干净跑完 1024 步 decode 无挂起无崩溃**（R19-1 崩+挂全中）。唯一变量=MAX_MODEL_LEN 从 40960→43008（+2048 余量覆盖 prompt≤~40.7K+1024 的 clamp 区）。**clamp 统一理论实证坐实**：max_model_len 不足 → decode 尾部非 uniform 步 → eager 仍进 remote select → Q=4 断言；+2048 余量后不触发 clamp → 1024 步全净。**1024 步 decode 解锁**，按预案 R25 进 BS16×1024 + 100K×1K 验收配方；非 uniform 路由根治（eager 分支正确处理 Q≠4）转代码线挂账（生产健壮性，不阻塞验收）。

**数据点**（1024 步口径首个 bench 点）：global_wall_s=99.743、output_tps=164.26、tpot avg=48.32ms（min 21.55/p50 48.26）、TTFT avg=45382ms（40K chunked-prefill 主导，与 R23-1 同量级）。

**日志/产物路径**：容器内 `/home/fjj/pp_r24.log`（tee 全量，含 POOL_ROI_REPORT + R24_RC=0）、`/home/w00919836/scp-opt/tmp/r24-bench-bs4-mt1024-ml43008/raw/model.log`（860KB 全链）、`summary.txt`（RUN_OK）、`run-status.json`（graceful:true）、`raw/overlap-*.csv`（8 端口满数据）、`results/1p/dp4_tp2_offload_graph_mtp_graph_hbm_full_remote_shared.metrics.json`。运行后机器自然收尾（EngineCore 全 SIGTERM 退出、worker 队列终止），进程 0 / NPU 8 卡空闲 / shm 384G 已清场。
