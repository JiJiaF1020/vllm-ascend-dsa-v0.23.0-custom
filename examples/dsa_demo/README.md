## 参数说明

### [TO BE REMOVE] 无MTP权重的合成draft调试

裁层模型没有MTP权重时，保持模型`num_nextn_predict_layers=0`；不要伪造三个物理层。
开启`VLLM_ASCEND_SMOKE_SYNTHETIC_DRAFT=1`后，配置使用无模型的ngram通道，但不做
ngram搜索：在target采样后重复最后一个有效token，提出至多三个draft。无有效输出、
不支持spec的请求不提议；接近上下文上限时缩短。保持正常rejection sampling，不强制接受。
开启时跳过ngram的Numba大矩阵预热，不加载任何draft权重，也不创建物理MTP cache层。

```bash
VLLM_ASCEND_SMOKE_MODEL_PATH=/你的五层模型目录 \
VLLM_ASCEND_SMOKE_TENSOR_PARALLEL_SIZE=1 \
VLLM_ASCEND_SMOKE_DATA_PARALLEL_SIZE=1 \
VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=1 \
VLLM_ASCEND_SMOKE_ENABLE_MTP=1 \
VLLM_ASCEND_SMOKE_SYNTHETIC_DRAFT=1 \
VLLM_ASCEND_SMOKE_ENABLE_MTP_GRAPH=0 \
VLLM_ASCEND_SMOKE_RUN_MODE=graph \
python examples/dsa_demo/simple_test_1p.py
```

同一组参数加`--dry-run`先检查配置。`RUN_MODE=eager`用于对应的eager测试；
`ENABLE_OFFLOAD=0`为原生target对照，`1`为本地卸载，二者都支持target图模式。
图模式沿用`VLLM_COMPILE + FULL_DECODE_ONLY`，capture大小为`4 × 请求数`。
CPU proposer和rejection/scheduler仍在图外；`ENABLE_MTP_GRAPH`对合成模式不创建draft图。
当前`--store-endpoint`和ready-shadow仍受原有eager门禁保护；真实`--remote-shared-config`
可以使用下文的延迟捕获decode FULL图（不是将Store调用放入图中）。
合成draft只提供target的Q=4输入，不能代替真实PUT/GET或精度验收。

验收不要只看配置或进程退出：

- `DSA_SYNTHETIC_DRAFT_ENABLED ... physical_mtp_layers=0`，没有MTP权重加载。
- `DSA_SYNTHETIC_DRAFT_PROPOSE ... draft_lengths=[3,...]`。
- `DSA_SYNTHETIC_DRAFT_TARGET ... query_lengths=[4,...]`，状态为`SpecDecoding`。
- 图模式在稳态看到`DSA_SYNTHETIC_DRAFT_DISPATCH mode=FULL ... uniform=True`，并以
  msprof确认实际replay；`GRAPH_READY`只证明capture配置，不代表每步都命中图。
- 首轮prefill、上下文/输出长度边界及DP空批不保证Q=4或命中图，不能强行填充这些请求。
- 本地卸载还要确认真实非零KSC搬运；远端eager还要确认真实回包/ScatterSFA，不以合成token代替IO。

结果JSON和文件名包含synthetic标识。这只验证target四路验证及IO/调度，不代表真实MTP
模型、接受率或整步性能；调试日志和CPU提议也引入Host开销。当前开发机只做Python级检查，
NPU capture/replay、精度及DP/TP并发须在服务器验收。无算子/wire变更，无需重新编译OPP/hcomm。

组图边界：合成draft是Host token列表，由原scheduler写入普通target输入buffer，
不新增注册HBM、GVA、跨流Event或Torch schema；不修改既有本地DRAM buffer生命周期。
实机仍需核对编译图参数、原算子源地址和流依赖，CPU检查不能替代FX/地址/可见性验收。
实现与本文集中在一个`[TO BE REMOVE]`提交，调试结束整体revert该提交，保留之前BS32基线。

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

真实`--remote-shared-config`链支持每DP BS1..32、TP1/TP2、MTP3 eager；
decode FULL图适配已加入，DP4/BS32的NPU实机验收尚未完成。
首次decode绑定一批请求，之后允许缩批/重排；请求结束保留row和连接，到worker退出统一释放。
暂不支持后补新请求或row复用。请给整批prefill足够token/HBM预算，并检查
`DSA_SHARED_CONNECT_BEGIN rows=...`是否确实包含期望的32请求；只设`MAX_NUM_SEQS=32`
不能保证首批实际BS32。该路径需最新C8 OPP+Torch extension，详细步骤见算子仓
`docs/shared_placement_review.md` §8.9，不沿用下面普通本地路径的graph设置。

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

### 双机分别使用本机 Pool：真实远端 shared

当前已适配配置与启动入口，双机真实通信/推理仍待实机验收。每台启动自己的
`urma_pool_server`（一个Store及本机各卡独立DU服务），沿用单机已验证的Pool启动参数。
两机Pool之间不通信、不共享backing；跨机HCCL/EP通信仍由框架负责。不改DU、gather、
wire或算子，无需为本次修改重编Pool、extension、C8 OPP或hcomm，只更新框架Python。

每台各准备一份`--remote-shared-config` JSON，保留原来的`store_endpoint`、`devices`、
可选`validate/debug`字段：

- `store_endpoint`指向本机Pool的Store；每个`du_endpoint`指向同一Pool对应的DU服务。
  容器内填写实际可达地址，不确定host网络时不要盲用`127.0.0.1`。
- `devices`只放本机8张卡的物理ID、真实EID、`eid_index`和`pool_eid`。
  两机物理ID均可为0～7；第二台不要改为8～15，也不要复制第一台的EID。
- `simple_test_2p.py`自动补入`node_count=2`，校验本机设备数为`全局DP / node_count * TP`。
  JSON显式提供的`node_count`必须与启动入口一致。单机入口默认1，旧配置继续有效。
  直接通过框架`additional_config`接入而不使用此脚本时，需要自己填写正确的`node_count`。
- 全局DP rank保持0～15；每个TP组必须在同一节点。debug的rank仍为全局DP/TP rank。
  worker建链前仍用HCCP inventory核对真实物理ID及EID/index，不通过校验就拒绝连接。

同一台机器按NUMA运行多个独立Pool时，每个`devices[物理ID]`可增加`store_endpoint`，
不填则沿用顶层默认值。例如NPU0/1的DU端口为19090/19091、Store为19080；NPU2/3的
DU端口为19092/19093、Store为19081，则在设备2/3的原有完整配置里补充：

```json
"store_endpoint": "127.0.0.1:19081"
```

这是一条新增字段示例，不是完整配置；保持各卡原来的真实EID/index/pool_eid和DU端口。
worker根据HCCP真实物理卡号选择Store后，CREATE、PUT、PUT_DONE和CLOSE都走选中的
Store；DU连接携带同一个Store ID及request handle/version，服务端会拒绝Store错配。
`DSA_SHARED_DEVICE`打印生效的Store/DU组合。每个wrapper使用独立backing，NPU到NUMA
关系由配置明确指定，不假定NPU编号等于DP rank或NUMA ID。未配置覆盖时单Store行为不变。

两台分别执行（第二台把rank改为1，配置路径换成自己的文件；模型、采样与其它参数一致）：

```bash
export VLLM_ASCEND_SMOKE_MODEL_PATH=/path/to/model
export VLLM_ASCEND_SMOKE_DATA_PARALLEL_SIZE=16
export VLLM_ASCEND_SMOKE_TENSOR_PARALLEL_SIZE=1
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=1
export VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=1
export VLLM_ASCEND_SMOKE_ENABLE_MTP=1
export VLLM_ASCEND_SMOKE_RUN_MODE=eager
export VLLM_ASCEND_SMOKE_ENABLE_PROFILE=0

python examples/dsa_demo/simple_test_2p.py \
  --node-rank 0 --master-addr <第一台主机IP> \
  --master-port 29500 --sync-port 29600 \
  --remote-shared-config /path/to/remote-shared-node0.json --dry-run
```

先分别dry-run：节点0的全局rank应为0～7，节点1为8～15，`required_local_npus=8`，
透传的`additional_config.dsa_sparse_config.remote_shared.node_count=2`。
dry-run不需要启动Pool，也不验证URMA连通性；确认后两台去掉`--dry-run`一起执行。
检查两台`DSA_SHARED_CONNECT_BEGIN`的host、全局dp、physical、endpoint和namespace，
确认各自访问本机服务。可以先跑`check_two_node_dp16_ep16.sh`隔离HCCL/MC2问题。

eager功能通过后再切`RUN_MODE=graph`；沿用原有真实数据预热、延迟capture和DP模式一致性，
不新增跨机远端内存或图输入。所有rank均须验证多步回放与退出drain，不能只看rank0。
`MAX_NUM_SEQS`是每个DP的batch，并非全局batch；提高前检查每卡HBM容量。每台Store按
本机8个worker的活跃请求及层/页数预留容量。当前仍限定离线固定请求批次，不支持请求
跨DP迁移、row复用或抢占恢复；Pool重启后需要重建连接，不能复用旧图和旧handle。

## 真实远端 shared 的 decode FULL 图（待实机验收）

先保持已经跑通的Store/DU/EID JSON及模型路径，只把target切到`graph`；drafter仍eager。
当前沿用vLLM 0.23的外层`ACLGraphWrapper`，直接捕获实际模型的算子调用，跳过Dynamo
编译forward；不是新建一套GraphEx多流调度，也不是仅打印FULL而执行eager。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ASCEND_SMOKE_DATA_PARALLEL_SIZE=4
export VLLM_ASCEND_SMOKE_TENSOR_PARALLEL_SIZE=2
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=32
export VLLM_ASCEND_SMOKE_RUN_MODE=graph
export VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=1
export VLLM_ASCEND_SMOKE_ENABLE_MTP=1
export VLLM_ASCEND_SMOKE_ENABLE_MTP_GRAPH=0
# 没有MTP权重时仅用于target/IO穿刺；有真实MTP权重则设0。
export VLLM_ASCEND_SMOKE_SYNTHETIC_DRAFT=1
python examples/dsa_demo/simple_test_1p.py --remote-shared-config /绝对路径/remote-shared.json --dry-run
python examples/dsa_demo/simple_test_1p.py --remote-shared-config /绝对路径/remote-shared.json
```

这是每DP BS32，共128个请求、8个DU连接；不是DP32。先DP1/BS1验证，再扩大。
首批prefill必须能容纳完整cohort（23k×32输入需相应prefill token预算和HBM）；仍不支持
新请求补入或row复用。JSON为每物理卡配置独立DU服务，Store容量至少256个rank-local请求。

执行次序：真实prefill/PUT → 首次真实decode eager预热 → 同shape下一步capture并执行一次
→ 后续replay。启动日志`DSA_SHARED_GRAPH_DEFERRED`表示跳过无连接的dummy capture，
不是禁用图；随后应出现`DSA_SHARED_GRAPH_CAPTURE_BEGIN/END first_replay=1`、
`Replaying aclgraph`。显式验收模式才输出逐步`DSA_SHARED_STEP_OK ... checked=ScatterSFA,APPLIED`。
需看msprof确认LISSend、GATHER_ONLY、ScatterSFA及APPLIED确实在图内。

每层generation/status独立，generation每step在图外H2D；已捕获shape的buffer保留到关闭。
PUT/PUT_DONE、REFRESH仍在图外，保留必要的Host同步。
所有DP均预热、均为Q=4且token数一致时才FULL；缩批只能命中精确capture size，
不一致时全体eager，不创建远端PAD请求。Q不足4、非SPARSE等原远端限制没有因此解除。
图内地址变化会明确报错，不会静默使用旧metadata。

`--remote-shared-config` JSON顶层可设置`"validate": true`恢复Host验收检查，默认False。
默认不做逐层ticket/status的同步D2H，也不做prepare/select中的累计query、SPARSE模式及
causal tail值检查；这些输入契约仍须满足，并非放开非Q4或超长tail支持。开启`debug`也会
恢复验收检查。性能测试应移除`debug`并关闭`validate`；否则检查本身会串行化设备下发。
默认模式保留设备completion/status/APPLIED依赖、PUT_DONE水位、row绑定与复用限制、
图输入地址检查。`finish_decode`仅记账，不代表Host已核验设备成功；错误不再保证逐层即时报告。
`prepare_decode`的budget/rows/candidate绑定读取仍有D2H，不能把本改动当成消除了全部步间同步。

延迟捕获的graph/workspace占用未计入启动期KV容量估算，需保留HBM余量，不能直接使用
启动日志建议的最大KV内存。此次只改框架Python，不要求重编Pool、hcomm或OPP；仍需已有
LISSend/ScatterSFA/APPLIED正确部署。仅完成静态检查，不能视为DP4/TP2设备验收已通过。

## eager远端请求对拍

在已通过的`--remote-shared-config` JSON中增加`debug`字段：

```json
"debug": {
  "directory": "./remote_debug_run1", "steps": 3,
  "topk_rows": [0], "ranks": [[0, 0]],
  "compare_li": true, "topk_print_limit": 16
}
```

设置`VLLM_ASCEND_SMOKE_RUN_MODE=eager`、`VLLM_ASCEND_SMOKE_ENABLE_MTP_GRAPH=0`，
其余模型、DP/TP及MTP3参数保持原样。`ranks`为DP/TP编号，`topk_rows`为稳定pool row。
所选rank打印前steps次远端forward中各层全部活跃请求的实际搬运token数（四路union miss），
并将指定row的LISSend、同候选范围原生LI、full原生LI保存为CPU `.pt`，终端默认仅打印前16项。

观察`DSA_LI_COMPARE`、`DSA_REMOTE_IO`和generate后的`DSA_REMOTE_DEBUG_SUMMARY`。
同候选范围LI集合不同视为失败；full LI含tail，集合不同仅作策略对照。IO摘要来自真实
reply，不是根据输入推测；GATHER_ONLY复用计划但加载当前层数据。目录中`summary.csv`
提供Store handle/version等身份，用于算子仓`tools/dump_remote_kv.py`手动读原始页。
`group_*_bytes`字段每请求重复，不能逐行相加。

需要新版算子extension的`lis_send_pool_debug_read`，不需因此重编OPP/hcomm。
手动页dump另需新版Store。完整部署、数据口径、dump命令见算子仓
`docs/shared_placement_review.md`的§8.11。
诊断会增加LI调用、同步、D2H和落盘，不能用于性能比较；未覆盖足够step/row或发生差异时，
等generate及DP结束屏障后统一报错，不在TP层内因数值差异提前退出。真实设备错误仍可能中断运行。
关闭诊断只需删除`debug`字段，原图模式不变。

## 可选Pool worker步间休眠

在原`--remote-shared-config` JSON顶层加入 `"worker_parking": true`，默认关闭。
先更新Pool和算子Torch extension（需包含`lis_send_pool_begin_step`），再更新框架Python；
此功能不需重编C8 OPP/hcomm，不新增端口。

唤醒通知位于`_model_forward()`返回后、`finish_decode()`/后处理之前：此时Host准备已完成、
NPU任务已提交，但不等待设备执行完。每rank用已有DU TCP连接单向发64B，无ACK、无D2H，
C++发送释放GIL。图内不增加节点。真实命令先于TCP到达时，poll也会兜底唤醒。
Pool在本步全部网络组最终CQE及索引收尾后让worker和merge helper休眠，poll仍忙轮询。
步内不会逐层睡醒；不改变APPLIED/plan/buffer复用语义。

第一轮建议保持现有绑核、batch和层数，只对比开/关此选项。确认Host准备阶段线程能够睡眠，
再看首层ScatterSFA等待和整步延迟是否改善；通知时机、OS唤醒仍可能引入首层额外等待。
旧Pool不能接收此控制帧，关闭选项并重启连接可恢复原模式。详细部署见算子仓
`test/urma_mock_pool/README.md`的“步间休眠”章节。

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
