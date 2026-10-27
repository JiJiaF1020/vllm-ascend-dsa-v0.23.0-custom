# Pingpong

## Protocol

- 本文件是蓝区 ↔ 绿区的唯一通信面；`history.md` 是只追加的轮次档案。
- **蓝区**（可外部 AI 侧）写全部代码、做全部判读决策；**绿区**（隔离执行侧）按任务书执行并回写交付，不改代码。
- 每次 commit 必须同时包含代码 + `pingpong.md` + `history.md`；pull 用 `--ff-only`；绝不 force-push。
- Current owner: **绿区**。Status: R3 派单已发出（EI0006 定位与越过：offload-off 对照 + HCCL_CONNECT_TIMEOUT=1800），等绿区回传。R2 已判读：阴性，EP8 权重装载实测成功（61.3GB/卡），EI0006 归框架/配置层。

## 0. 目标与口径

**总目标**：在绿区 A5 单机 8 卡（Ascend 950DT）上跑通 remote-offload 整网链路——框架（本仓 `vllm-ascend-v0.23.0-custom-remote-offload` 分支）+ 算子仓 `a5-remote-offload-ops` + 本机 URMA Pool，完成 `hbm_full_remote_shared` 真实推理，拿到作者文档钦定的生效证据。

**锚点事实**（蓝区已核对，绿区不得变更口径）：

- 本仓分支 tip：`cac7e2c4e` + 蓝区 pingpong 文档提交（pull --ff-only 后 `git log --oneline -3` 顶部应是 pingpong 提交）。
- 算子仓：`https://github.com/JiJiaF1020/a5-remote-offload-ops`，main tip `6e85590db`。
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

## §R3 派单：EI0006 定位与越过（蓝区，2026-10-27）

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
