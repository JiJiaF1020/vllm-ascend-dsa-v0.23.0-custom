## 参数说明

### 有限ready旁路验证（新增，待A5实机验收）

`--ready-shadow-dir ./ready_shadow --ready-shadow-steps 3 --ready-shadow-group 0`
在本地LIDU/KSC/SFA轨迹上比较独立CopySFA副本，失败保存CPU快照；不使用真实Pool。
先用DP1/TP2、BS1、MTP3、`RUN_MODE=eager`、`ENABLE_MTP_GRAPH=0`和`--mtp-eager`。
需要配套算子仓新增的32/64头CopySFA与`kvcache_scatter_copy_custom_ops.ready_shadow`模块。
默认模型与DP4/TP2不变。只采稳定SPARSE、每请求四query调用，覆盖full及其shared层。
`generate`后汇总全部TP rank，数值失败或覆盖不足都会失败，模型始终返回本地结果。
验证有同步/复制，不可测性能；TP1实际128头、DP4、真实LISSend/PoolDU仍待后续。
详细命令及重放见算子仓`docs/glm52_ready_validation.md`。

环境变量均需加前缀 `VLLM_ASCEND_SMOKE_`，例如 `ENABLE_PROFILE` 写作 `VLLM_ASCEND_SMOKE_ENABLE_PROFILE`。未设置时使用默认值，布尔值用 `1` 开启、`0` 关闭。

| 参数（省略前缀） | 默认值 | 说明 |
|---|---|---|
| `MODEL_PATH` | `/home/models/GLM-5.2-w4a4c8-mxfp4_99` | 模型目录 |
| `TENSOR_PARALLEL_SIZE` | 单机 `2`，双机 `1` | TP 大小 |
| `DATA_PARALLEL_SIZE` | 单机 `4`，双机 `16` | 全局 DP 大小；双机须为偶数 |
| `MAX_NUM_SEQS` | `1` | 每个 DP rank 的 BS；总 BS = DP × 此值 |
| `DATASET` | `chinese_23k` | `short/chinese_5k/chinese_10k/chinese_23k/chinese_40k/chinese_64k/chinese_104k`；GLM-5.2 实测依次为（不含 short）5000/10000/23000/40000/64000/104000 token；104k 为40k与64k拼接 |
| `PROMPTS` | 未设置 | 自定义 JSON 数组，如 `'["你好"]'`；与 DATASET 二选一 |
| `MAX_MODEL_LEN` | `25600` | 单请求输入与输出的最大总 token 数 |
| `MAX_NUM_BATCHED_TOKENS` | `25600` | 每 DP rank 每步 token 预算；关闭 chunk 时须 ≥ MAX_MODEL_LEN |
| `MAX_TOKENS` | `64` | 最多生成的 token 数 |
| `ENABLE_CHUNKED_PREFILL` | `0` | Chunked prefill 开关；此前存在精度问题，默认关闭 |
| `ENABLE_OFFLOAD` | `1` | `0` 为不卸载基线，仍保留 C8；baseline 的 MTP 自动 eager |
| `RUN_MODE` | `graph` | `cache-init` 只初始化；`eager` 不入图；`graph` target FULL_DECODE_ONLY |
| `ENABLE_MTP` | `1` | MTP 开关 |
| `MTP_NUM_SPECULATIVE_TOKENS` | `3` | 当前开启 MTP 时仅支持 3 |
| `ENABLE_MTP_GRAPH` | `1` | 三轮 MTP 合并图；仅 offload + graph + MTP 时生效 |
| `GPU_MEMORY_UTILIZATION` | `0.85` | 显存利用率，范围 `(0,1]` |
| `QUANTIZATION` | `ascend` | 量化方式，本示例保持默认 |
| `ENABLE_EXPERT_PARALLEL` | `1` | EP 开关；开启时 EP=TP×DP；卸载且 DP>1 时必须开启 |
| `ENABLE_A5_PACKED_C8` | `1` | LI/SFA C8 开关；卸载 + MTP3 时必须开启 |
| `ENABLE_CPU_BINDING` | `1` | CPU 绑核开关 |
| `ENABLE_PROFILE` | `0` | Profile 开关，采集 generate 阶段；cache-init 不采集 |
| `PROFILE_DIR` | 单机 `./profiles/1p`，双机 `./profiles/2p` | 采集根目录，自动按配置/rank 分目录 |
| `RESULT_DIR` | 单机 `./results/1p`，双机 `./results/2p` | JSON 结果目录；`none` 禁用保存；重复实验请更换输出目录 |
| `GRAPH_CAPTURE_SIZES` | `1,2,4,8,16` + `MAX_NUM_SEQS` | 请求数档位；自动补入当前 BS、去重并过滤大于 BS 的值；MTP3 时自动乘 4 |
| `PROCESS_TIMEOUT_SECONDS` | `3600` | DP 进程及前端同步等待超时，单位秒 |
| `DSA_INDEXER_MLA_BLOCK_RATIO` | `4` | 卸载 Indexer / resident MLA block 比例 |
| `DSA_MAX_ACTIVE_REQS` | `256` | 卸载活跃请求容量上限 |
| `DSA_HOT_CPU_BLOCK_MULTIPLE` | `1.0` | 卸载 DRAM 热块容量倍率 |
| `DSA_PROMPT_BUDGET_THRESHOLDS` | `32768,65536` | 卸载 prompt 分档阈值，逗号分隔 |
| `DSA_MTP_RESIDENT_BUDGET_TOKENS` | `8192,12288,12288` | 仅卸载 + MTP 生效，依次对应 ≤32768、≤65536、>65536；可选 8192/10240/12288，须非递减；非 MTP 忽略并保持 10240,10240,10240 |
| `NODE_RANK` | `0` | 仅双机：第一台 0，第二台 1 |
| `MASTER_ADDR` | 空，双机必填 | 两台均填第一台 IP |
| `MASTER_PORT` | `29500` | 双机 DP rendezvous 端口，两台一致 |
| `SYNC_PORT` | `29600` | 双机前端同步端口，两台一致且不同于 MASTER_PORT |

## 单机多卡运行：simple_test_1p.py 使用样例

在仓库根目录运行：单机 8 卡 DP4TP2 + 卸载 + target graph + MTP3 graph。可在末尾加 `--dry-run` 只检查配置。

```bash
VLLM_ASCEND_SMOKE_MODEL_PATH=/home/models/GLM-5.2-w4a4c8-mxfp4_99 \
VLLM_ASCEND_SMOKE_TENSOR_PARALLEL_SIZE=2 \
VLLM_ASCEND_SMOKE_DATA_PARALLEL_SIZE=4 \
VLLM_ASCEND_SMOKE_GPU_MEMORY_UTILIZATION=0.92 \
VLLM_ASCEND_SMOKE_DATASET=chinese_64k \
VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=8 \
VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=80000 \
VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=16384 \
VLLM_ASCEND_SMOKE_ENABLE_CHUNKED_PREFILL=1 \
VLLM_ASCEND_SMOKE_MAX_TOKENS=256 \
VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=1 \
VLLM_ASCEND_SMOKE_ENABLE_MTP=1 \
VLLM_ASCEND_SMOKE_DSA_INDEXER_MLA_BLOCK_RATIO=4 \
VLLM_ASCEND_SMOKE_ENABLE_PROFILE=0 \
VLLM_ASCEND_SMOKE_PROFILE_DIR=./vllm_profile/dp4tp2_64k_bs8_offload  \
python examples/dsa_demo/simple_test_1p.py
```

## 双机多卡运行：simple_test_2p.py 使用样例

两台容器各运行一次下面的命令：每机 8 卡、全局 DP16TP1。

- 第一台 `NODE_RANK=0`，`MASTER_ADDR` 填第一台 IP
- 第二台 `NODE_RANK=1`，`MASTER_ADDR` 填第一台 IP

```bash
VLLM_ASCEND_SMOKE_NODE_RANK=0 \
VLLM_ASCEND_SMOKE_MASTER_ADDR=141.61.33.13 \
VLLM_ASCEND_SMOKE_MODEL_PATH=/home/models/GLM-5.2-w4a4c8-mxfp4_99 \
VLLM_ASCEND_SMOKE_TENSOR_PARALLEL_SIZE=1 \
VLLM_ASCEND_SMOKE_DATA_PARALLEL_SIZE=16 \
VLLM_ASCEND_SMOKE_GPU_MEMORY_UTILIZATION=0.92 \
VLLM_ASCEND_SMOKE_DATASET=chinese_64k \
VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=8 \
VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=80000 \
VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=16384 \
VLLM_ASCEND_SMOKE_ENABLE_CHUNKED_PREFILL=1 \
VLLM_ASCEND_SMOKE_MAX_TOKENS=256 \
VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=1 \
VLLM_ASCEND_SMOKE_ENABLE_MTP=1 \
VLLM_ASCEND_SMOKE_DSA_INDEXER_MLA_BLOCK_RATIO=4 \
VLLM_ASCEND_SMOKE_ENABLE_PROFILE=0 \
VLLM_ASCEND_SMOKE_PROFILE_DIR=./vllm_profile/2p_dp16_64k_bs8_offload  \
python examples/dsa_demo/simple_test_2p.py
```

## 双击HCCL连通性测试

在**两台机器的容器内、仓库根目录下，各运行同一条命令**。

```
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples/dsa_demo/check_two_node_dp16_ep16.sh --master-port 29688 --log-dir ./two_node_preflight_logs --node0-ip 141.61.33.13 --node1-ip 141.61.33.21
```

注意：

- 每台8张卡应空闲，容器使用 `--network=host`，具备 `ip`、`hostname` 命令和 `torch_npu` 环境。
- 两台 IP 参数的顺序、端口必须一致；脚本自动识别节点号及网卡。
- 两端尽快先后启动即可，不需要精确同时。不要额外套 `torchrun`。
- **不要加 `--skip-mc2`**，这样才会同时验证 HCCL 和 MC2。

检查内容包括16 rank 初始化、`all_reduce` 正确性、`all_to_all_single` 路由，以及 MC2 dispatch/combine。两台最后都出现：

```
[PASS] node=... completed DP16/EP16 communication preflight
[PASS] HCCL all_reduce + all_to_all + requested MC2 checks passed
```
