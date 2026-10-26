# Pingpong

## Protocol

- 本文件是蓝区 ↔ 绿区的唯一通信面；`history.md` 是只追加的轮次档案。
- **蓝区**（可外部 AI 侧）写全部代码、做全部判读决策；**绿区**（隔离执行侧）按任务书执行并回写交付，不改代码。
- 每次 commit 必须同时包含代码 + `pingpong.md` + `history.md`；pull 用 `--ff-only`；绝不 force-push。
- Current owner: **绿区**。Status: 第 1 轮派单已发出（remote-offload 单机整网首发），等绿区回传。

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

## §R1 派单：remote-offload 单机整网首发（蓝区，2026-10-26）

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
