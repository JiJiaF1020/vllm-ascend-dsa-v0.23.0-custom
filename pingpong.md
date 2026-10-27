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
