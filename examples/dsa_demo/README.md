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
| `DSA_INDEXER_MLA_BLOCK_RATIO` | `3` | Indexer/full KV/MTP 与 resident MLA 的 block 容量比，不是 full/shared 层数比 |
| `DSA_MAX_ACTIVE_REQS` | `256` | 卸载活跃请求容量上限 |
| `DSA_HOT_CPU_BLOCK_MULTIPLE` | `1.0` | 卸载 DRAM 热块容量倍率 |
| `DSA_PROMPT_BUDGET_THRESHOLDS` | `32768,65536` | 卸载 prompt 分档阈值，逗号分隔 |
| `DSA_MTP_RESIDENT_BUDGET_TOKENS` | `8192,12288,12288` | 仅卸载 + MTP 生效，依次对应 ≤32768、≤65536、>65536；可选 8192/10240/12288，须非递减；非 MTP 忽略并保持 10240,10240,10240 |
| `NODE_RANK` | `0` | 仅双机：第一台 0，第二台 1 |
| `MASTER_ADDR` | 空，双机必填 | 两台均填第一台 IP |
| `MASTER_PORT` | `29500` | 双机 DP rendezvous 端口，两台一致 |
| `SYNC_PORT` | `29600` | 双机前端同步端口，两台一致且不同于 MASTER_PORT |

## 单机多卡运行：simple_test_1p.py 使用样例

远端`hbm_full_remote_shared`的full层仅分配完整packed KV，不再额外分配resident数据区；
本地卸载、shared各层resident和MTP full cache保持原样，无需新开关或JSON字段。
更新框架Python后必须重启推理并重新capture；此次不需重编OPP、extension或Pool。
检查启动的`DSA HBM CACHE CAPACITY REPORT`：78层/21 full时应为
`Physical cache layers: indexer=21, resident=57, mtp=1`，另列`Full MLA storage views: 21 aliases`。
逐full层`DSA_FULL_KV_STORAGE ... resident_extra_bytes=0`确认绑定地址与完整KV一致。
自动规划通常会将节省的预算重新用于扩池，因此总KV HBM不一定下降；比较base blocks、tokens及
`Sparse decode MLA limit`，不要只看`Dense Indexer limit`推断并发上限。
Prefill仍需完整上下文暂存，不能直接把shared物理池限制为`BS*(H+256)`。
单机/双机入口默认容量比例改为3:1，`profile_pool_inference.py`启动这些入口时也沿用该默认值；
若已有环境变量显式设置为4，需改为3或取消该覆盖。可用
`VLLM_ASCEND_SMOKE_DSA_INDEXER_MLA_BLOCK_RATIO=4`回退对照，不需重编算子或Pool，
但必须重启推理、重新分配cache并capture。比例不按full/shared层的数量或排列自动推导；
固定HBM预算下，降低比例会增加resident容量、减少完整历史容量，须同时检查Prefill准入和
MLA/Indexer/MTP并发上限，不能只看resident容量增加。

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
VLLM_ASCEND_SMOKE_DSA_INDEXER_MLA_BLOCK_RATIO=3 \
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
VLLM_ASCEND_SMOKE_DSA_INDEXER_MLA_BLOCK_RATIO=3 \
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
连接完成一次真实decode预热、真实query均为Q=4时允许向上匹配capture size，DP间真实B可以不同。
模型/MoE按原生dispatcher统一padding；LISSend、GATHER_ONLY、ScatterSFA、APPLIED
只使用真实请求前缀，不创建远端PAD请求，也不修改PAD mirror或等待PAD completion。
所有图内Tensor保留容量尺寸，设备rows=-1表示PAD，不再按active B切片。
source full的SFA也保留容量，沿用原生seq_lens/padding；普通full的原生LI/SFA保持原路径。
图缓存键仅为`BatchDescriptor`，同容量B=6/5/3/0共用同一张图；真实row重排也是设备输入值。
只有模型capture容量变化才可能产生新图，单纯缩批不再要求新的eager warmup。
已建链零请求DP在同容量图中跳过远端数学/IO/索引更新，保留prolog和o_proj/MoE集合通信；
从未建链的idle仍eager。Indexer/KV写入沿用原生PAD slot=-1语义，不冻结capture时token数。
启动期profile/capture不会因此建链，idle也不占用Pool row。
Q不足4、非SPARSE、新请求补入和row复用等原远端限制没有因此解除。
图内地址变化会明确报错，不会静默使用旧metadata。

验证padding时建议先用DP2/TP1，保持Q=4并让两DP分别出现B=6/5、6/3以及6/0，
再扩大到DP4/TP2。`VLLM_LOGGING_LEVEL=DEBUG`下检查
`DSA_SHARED_GRAPH_ROWS active=5 graph_rows=6 pad=1 remote_commands_per_layer=1`，
以及各rank的`DSA_GRAPH_DISPATCH final_mode=FULL`、`DSA_SHARED_GRAPH_REPLAY active=...`。
首次连接的`remote_decode_warmup`可以出现；其后不应仅因为同容量下B变化重复warmup/capture。
idle应为`active=0 ... remote_commands_per_layer=0`，且Pool没有该rank新增GET/APPLIED。
开`validate`核对真实请求status，eager/graph分别检查相同逻辑KV及输出；性能轮关闭validate/debug。
需要在设备上检查PAD slot、实际地址、流归属和completion代数，不能只看FULL日志。
本版须同时部署算子仓extension、C8 OPP和普通AIV OPP（APPLIED/GATHER_ONLY），Pool不变；
旧进程/图不可复用。详细设备合同及单算子动态B命令见算子仓`docs/dynamic_batch_graph.md`。

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
## Remote decode 图外 Host MSTX

排查步首同步或首个 MoE 等待时，在原启动命令前设置：

```bash
export VLLM_ASCEND_DSA_HOST_MSTX=1
python3 examples/dsa_demo/simple_test_1p.py --remote-shared-config /path/to/remote-shared.json
```

默认关闭；开关在 worker 创建 RemoteShared 时读取。使用框架内部 profiler 时，设置
`SmokeConfig.enable_profile=True`（或 `VLLM_ASCEND_SMOKE_ENABLE_PROFILE=1`）；
内部 `TorchNPUProfilerWrapper` 会用同一个 Host MSTX 开关联动 `msprof_tx` 采集选项，
无需外部 msprof，也不要再套另一套 profiler。仅设置 Host MSTX 环境变量不会启动
profiler，不改变 validate/debug 配置；输出位置仍沿用原 `profile_dir/run_name/dp_rank_*`。

从算子仓启动 Pool 和模型时，脚本会继承环境变量。例如只采框架内部 profiler：

```bash
export VLLM_ASCEND_DSA_HOST_MSTX=1
export VLLM_ASCEND_SMOKE_ENABLE_PROFILE=1
python3 tools/profile_pool_inference.py --no-perf \
  --framework-repo /home/w00919836/scp-va \
  --remote-config /path/to/remote-shared.json \
  --output tmp/pool-host-mstx-01
```

这里 `--no-perf` 只关闭 Pool CPU 的 perf/SPE 采样，不会关闭框架内部 profiler。
其它 Pool 容量、层数、设备、NUMA 参数继续沿用本次实际配置；框架日志在输出目录的
`raw/model.log`，框架 trace 仍在上面的 `profile_dir`，不是 Pool 的 overlap CSV。

新增区间使用 `torch_npu.npu.mstx.range_start(..., stream=None)`，只标记 Host，
不插入设备 Event、同步、D2H 或图内算子。接口语义见
[官方 MSTX API](https://www.hiascend.com/document/detail/zh/Pytorch/720/apiref/torchnpuCustomsapi/context/torch_npu-npu-mstx-range_start.md)。

时间线搜索 `DSA_HOST/`；标签包含 host、DP/TP、physical device、epoch 和 batch：

- `prepare_decode`：图外准备总区间。
  - `metadata_readback`：原预算读回与可选输入检查。
  - `dump_before_write`：满块检查/实际 PUT；无任务时也有短区间。
  - `prepare_binding`：原 rows/candidates 读回及绑定检查，内嵌实际触发的 `connect`、`pool_refresh`。
  - `generation_update`：各层 buffer 准备及 generation 下发。
- `graph_replay_pre_sync`：已有图每次 replay 前原本需要执行的同步；原路径跳过同步时不出现。
- `graph_replay_submit`：已有图的 Host replay 调用，不等于设备执行时间。
- `finish_decode`：步末记账，以及 validate 开启时原有的 status/ticket D2H。

这些区间用于 remote target；不在逐层 forward 或 draft 内加入标记。首次 capture 后的
首轮 replay 尚未使用上述两个 replay 区间，仍以原 CAPTURE_BEGIN/END 日志定位。
epoch 是 target forward 编号，不是生成 token 数。嵌套区间不能直接相加；
长 Host 同步应继续追踪它在等哪个先前设备任务，并对比同轮各 rank 到达第一次 MoE 的时刻。
默认关闭时不调用 MSTX；打开后仍有 Host 打点成本，需与关闭时对照。
只更新框架 Python，无需重编 OPP、Torch extension 或 Pool。

### 排查某一步未重放 target FULL 图

在启动 frontend/worker **之前**设置，重启进程生效：

```bash
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_ASCEND_DSA_HOST_MSTX=1
python3 examples/dsa_demo/simple_test_1p.py --remote-shared-config /path/to/remote-shared.json \
  > model-debug.log 2>&1
rg 'DSA_GRAPH_|DSA_SHARED_(CONNECT|PREWRITE|REFRESH|GRAPH_)|GLM MTP graph' model-debug.log
```

经 `profile_pool_inference.py` 启动时同样继承上述环境变量，查看其 `raw/model.log`。
demo 不再把显式设置的 DEBUG 覆盖成 INFO。Ascend 文件日志先初始化也不会再阻止
console handler 初始化；worker 只调用文件日志入口时也会配置 console。RemoteShared
的普通 `logger.info` 没有 rank0 限制；`CONNECT_OK` 每次连接打印，并非每步打印。
日志默认还写入 `~/ascend/log/vllm_ascend/`（可由 `ascend_log_path` 修改）。

按同一 DP/TP、epoch 关联下列记录，不要仅凭 stream ID 或 Python forward 标签认定断图：

| 标记 | 用途 |
| --- | --- |
| `DSA_GRAPH_GATE` | target 本地准入：实际每请求 query 长度、阶段、reason，以及 enforce-eager/cascade/encoder 阻断条件 |
| `DSA_GRAPH_DISPATCH` | 本地 `local_mode`、DP 协同的 `dp_mode`、最终 `final_mode` 与 capture descriptor；DP1 的 dp_mode 为 None |
| `DSA_SHARED_GRAPH_BYPASS` | 当前 wrapper 直接调用 runnable；NONE 表示未走图，wrapper mode 不匹配也会经过此分支，不能单独推断所有嵌套图都未执行 |
| `DSA_SHARED_GRAPH_WARMUP` / `CAPTURE_BEGIN/END` | eager warmup 或真实 capture；这时存在 Python forward 属于预期，END 的 first_replay=1 表示首次重放已提交 |
| `DSA_SHARED_GRAPH_REPLAY_BEGIN` / `DSA_SHARED_GRAPH_REPLAY` | 缓存图 replay 前/提交返回；前者含 pre_sync，夹在中间的长时间结合 MSTX 判断是在等同步还是提交；不是设备完成日志 |
| `DSA_SHARED_PREWRITE_SKIP` / `PREWRITE_BEGIN/END` | 区分没有 dump 任务、已处理本 epoch，与实际进入满块保存；包含 rank/epoch |

例如本 rank `local_mode=FULL`、`dp_mode=NONE`、`final_mode=NONE`，应继续找其它
DP rank 同轮的 gate/dispatch，而不是归因于“某个 rank 太慢”；协调规则按模式而非耗时降级。
`reason=non_uniform_decode` 应核对 query_lengths；`capture_size_miss` 核对
capture sizes；`remote_decode_warmup` 是该真实 batch shape 尚未预热。
`DSA_GRAPH_DISPATCH` 也可能来自 dummy 调度；真实 target 轮次还应有对应 GATE，
epoch 是本 rank 的 forward 计数，不保证不同 rank 永远一一相等。

本次只新增 Host 日志，未修改门禁、图内算子、stream/event、FX 参数或外部注册地址。
日志使用现有 Host 元数据，不增加 Tensor 回读；普通 batch DEBUG 日志也不再打印
`num_tokens_across_dp` 的设备内容，仅打印 shape，避免 Tensor repr 隐式同步。
DEBUG 本身会增加文件/console I/O，并启用原有 graph 输入地址检查，**只用于定位，
不能拿这一轮的时延作为性能基线**。本机无法验证实际 NPU replay，请以远端日志和 MSTX 为准。

## Full 一次触发全部 shared IO（2026-10-21）

`hbm_full_remote_shared` 固定使用整组 Gather：Full 的 LISSend 只发送一次
DU_AND_GATHER，Pool 根据启动拓扑自动聚合同组全部 shared 层，不再等待后继层发送
GATHER_ONLY 或逐层消费确认。各 shared 直接执行自己的 CopySFA，仍等待本层的
reply generation/completion，末层才更新 mirror、执行唯一 APPLIED。
APPLIED 一次检查整组所有层的 status，不能仅凭末层成功确认整个计划。

无需修改 remote-shared.json，也不增加回退配置。能力不足时直接报错，不静默恢复逐层
Gather。必须部署配套 Pool、Torch extension、普通 OPP（APPLIED）和 C8 OPP（CopySFA），
更新框架后重启 worker/Pool，重新捕获图。只更换框架 Python 或只重编 extension 不够。

启动后核对 Pool/NPU 的 `group_gather=1`，`DSA_EXECUTION_PLAN` 后继不再包含
`gather_only`；设备 timeline 只保留每组末端一次 APPLIED。eager IO 对拍中，后继层的
操作标为 `AUTO_GATHER`，仍读取各层真实回包，不是 ready 旁路。原 KV 对拍入口不变。
固定连接必须满足 `shared层数 × ceil(connection_batch/16) <= 8`，否则直接拒绝，防止
同组未消费的回包槽被覆盖；GLM 三 shared、BS32 需要 6 槽。

Pool 复用当前 context/worker 和分段发送流水；“一把触发”不表示只有一个 payload，
也不保证所有层同时 Gather。完整行为和重新编译命令见算子仓
`docs/full_triggered_group_gather.md`。本机 Host 回归不能替代 NPU/URMA、真实图与性能验收。
