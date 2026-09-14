# Store 真实生命周期接入：DP1/TP1 首验

本阶段 `store_shadow` 已连接真实框架调用点，但**推理仍走本地 LIDU/KSC/SFA**。
它把同一份真实 KV 满块同步上传远端，验证 CREATE / PUT / PUT_DONE / CLOSE，
不是 LISSend / PoolDU / CopySFA 的整网接入，也不是性能准入结果。
默认不启用，不改变 `local_all` 的 KV 放置或选择算法。

## 1. 已连接的调用点

| 框架节点 | 远端行为 |
| --- | --- |
| cache 初始化与逐层绑定 | OPEN；核对 target payload 层数、128 token/page、656B/token、容量；打印裁层后的物理层列表/full 层列表 |
| 首次真实 `prepare_forward` | 用稳定 pool row 绑定请求，CREATE；每个 worker 独立 namespace，不把 batch 下标当请求身份 |
| `_prepare_dump_plan` | 原计划负责源物理块与逻辑页；另记录 remote planned，不提前增加 done/bound |
| `DSALayerOffloadContext.dump_full_blocks` | 保留本地 dump，再从同一 HBM 源页做阻塞 D2H + TCP PUT；原始 U8 位解释，不转换 FP8 数值 |
| 最后一份所需 target 层数据到齐 | 查询 PUT_DONE，推进 done；没有 DU binding，所以 bound 始终为0 |
| `finished_req_ids` | CLOSE 成功后才归还稳定 row；下一请求使用新 handle |
| worker 正常 shutdown | CLOSE 最后一轮尚未收到 finished 通知的请求；不终止 Pool server |

请求暂时离开 active batch 不会 CLOSE；行重排不会重新 CREATE。MTP proposer 的完整
BF16 cache 与 Indexer cache 均不上传。模型失败时不要继续推进原请求；当前不实现断线恢复。

### PUT 时机

- Prefill：按既有逐层 attention 后 dump 的时机上传新满块。
- MTP decode：按 scheduler 已确认的前缀规划；下一 outer step 在本层覆盖 parity tail
  **之前**上传上一轮已确认的满块，不把未接受的 draft KV 当作已提交数据。
- 一页有效数据为 `128 × 656B`，只上传满页，不上传最终未满 tail。
- PUT_DONE 是跨全部 target payload 层的连续页前缀；不能在第一层 PUT 后就等待所有层。

此路径含真实 CPU/NPU 同步、D2H 和 TCP 调用。只支持 eager，并拒绝图编译/capture；
不是异步 PUT，也不会在 forward 中做 BIND/REFRESH/DRAIN。没有 lease 时无需伪造 APPLIED。

## 2. 内存池启动

在**算子仓**构建：

```bash
bash test/urma_mock_pool/build_store.sh
```

产物为 `tmp/mempool-build/urma_pool_server`。例如裁层后 **4个 target payload 层**、
max_model_len=32768，启动新的 backing：

```bash
./tmp/mempool-build/urma_pool_server \
  --backing /dev/shm/dsa-store-shadow-run1 \
  --store-bytes 1073741824 \
  --payload-layers 4 --requests 8 --source-capacity 32768 \
  --store-ports 19080 --trace-lifecycle
```

`--payload-layers` 必须等于裁层后实际 target KV 层数，**不是 full 层数或 selection group 数**。
不含独立 MTP proposer 层。此阶段只需 Store TCP 端口，无需 DU 端口、EID、HCCP 建链。
`--requests` 是 Store 累计创建容量；CLOSE 当前不回收物理页/handle，多次作业后需新 backing
或更大容量。不要覆盖已有 backing。上面内存量仅适用于这个小例子，不是完整模型配置。

## 3. NPU：同一裁层 checkpoint 做两次运行

在**框架仓**用现有 `examples/dsa_demo/simple_test_1p.py`。先保持现有 A5 算子包可用，
并确认其中提供 `custom_ops_lib.RemoteKvStoreClient`；此次框架修改是 Python，不需重编 OPP/hcomm。
服务端新增日志需要重编上面的 server。

```bash
export VLLM_ASCEND_SMOKE_MODEL_PATH=/path/to/cropped-model
export VLLM_ASCEND_SMOKE_DATA_PARALLEL_SIZE=1
export VLLM_ASCEND_SMOKE_TENSOR_PARALLEL_SIZE=1
export VLLM_ASCEND_SMOKE_MAX_NUM_SEQS=1
export VLLM_ASCEND_SMOKE_RUN_MODE=eager
export VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=1
export VLLM_ASCEND_SMOKE_ENABLE_MTP=1
export VLLM_ASCEND_SMOKE_ENABLE_MTP_GRAPH=0
export VLLM_ASCEND_SMOKE_MAX_MODEL_LEN=32768
export VLLM_ASCEND_SMOKE_MAX_NUM_BATCHED_TOKENS=32768
export VLLM_ASCEND_SMOKE_MAX_TOKENS=256

# 校验配置，不加载模型：
python examples/dsa_demo/simple_test_1p.py --store-endpoint <POOL_IP>:19080 --dry-run

# 先跑本地卸载基线：
python examples/dsa_demo/simple_test_1p.py

# 再启用真实满块上传，计算仍使用相同本地路径：
python examples/dsa_demo/simple_test_1p.py --store-endpoint <POOL_IP>:19080
```

模型路径、量化、prompt、seed、MTP、TP/DP 和本地预算保持一致。
两组结果目录分别使用原 run_name 和 `_store_shadow` 后缀，避免互相覆盖。
若还要原生不卸载基线，再单独设置 ENABLE_OFFLOAD=0；不能带 `--store-endpoint`。
不要把完整模型输出直接作为裁层模型的数值 oracle。

裁层时保留至少一个完整 full→shared selection group，配置的 indexer_types/层数必须与
实际模块一致。初始化沿用原 full/shared 拓扑校验，并打印 `DSA_STORE_SHADOW_TOPOLOGY`。
本阶段只验证组内各层的独立 payload 上传；远端 PlanSnapshot/FETCH/组级提交仍未实现。

## 4. 如何看完整流程

客户端正常顺序：

```text
DSA_STORE_SHADOW_TOPOLOGY ... selection_groups=... remote_selection_plan=0
DSA_STORE_CREATE row=0 ... handle=1
DSA_STORE_PUT_BEGIN layer=0 ... phase=post_attention
DSA_STORE_PUT_END layer=0 ...
... 所有 target 层 ...
DSA_STORE_PUT_DONE rows=[0] pages=[...] versions=[...] bound=0
... 后续满块 phase=prewrite ...
DSA_STORE_CLOSE ... binding=none
DSA_STORE_SHADOW_CLOSED ... inference=local_all remote_get=0
```

服务端 `MEMPOOL_LIFECYCLE` 使用同一台机器的 monotonic `time_us`，含 `port/store/rpc`：

```text
op=OPEN     phase=LAYOUT        layers=4 ...
op=CREATE   phase=CREATED       namespace=... key=... handle=1 ...
op=PUT      phase=COPY_BEGIN    layer=0 ... handle=1 page_min=0 page_max=... count=...
op=PUT      phase=LAYER_READY   layer=0 ...
op=PUT_DONE phase=WAIT_BEGIN    ...
op=PUT_DONE phase=READY         handle=1 version=... full_pages=...
op=CLOSE    phase=CLOSED        handle=1 backing_retained=1
```

每个 RPC 在接收完整请求前有 ACCEPT，收到后有 BEGIN，回 ACK 后有 ACK_SENT；错误有 ERROR。
`LAYER_READY` 只表示本次 PUT 已写入，不等于所有层都 READY；`ACK_SENT` 不是 NPU 消费确认。
未来 DU 控制客户端的 SNAPSHOT/LEASE 也会打印，但 LEASE 不是完整 BIND/DRAIN 成功的替代证明。
日志只打印身份/页范围/计数，不打印 KV 内容；正式性能测试关闭 `--trace-lifecycle`。

验收分别检查：

- 同一裁层模型两次运行的生成 token；需要定位误差时再抓同输入的 hidden states/logits。
- 实际 CREATE 数和关闭数匹配；row 重排/复用不混请求。
- 所有 target 层 PUT 后才出现对应 PUT_DONE。
- 至少出现一次 `phase=prewrite` 及 done 页数增长，才算覆盖 decode 满块；仅设 max_tokens=256
  不保证覆盖，模型可能提前 EOS。仅 prefill 成功不能算 decode 生命周期通过。
- 本阶段不能证明远端数据被 SFA 消费，也不能宣称节省本地 DRAM/HBM 或提升性能。

## 5. 本地验证边界与下一步

开发机已执行真实 C++ Store、双 TCP 控制客户端、真实 Torch CPU binding，验证逐层到齐、
递增 PUT、请求行重排/复用及 backing 字节一致性；不等价于框架/NPU 已通过。
现有 `tests/ut/dsa_offload/test_remote_store.py` 增加真实服务测试，需要将 server 加到 PATH
并部署 native binding；不会用 RPC mock 冒充服务。NPU 测试由服务器执行。

后续仍需完成真实动态候选长度/stride、LISSend 接入、selection-group 唯一计划/FETCH、
绑定刷新/消费确认以及长时队列回收。`store_shadow` 是明确的写侧验收点，不是这些能力的回退实现。

## 6. 算子包集成与后续 vendor 边界

当前两套算子入口不同，安装外部包不会自动替换框架的本地选择算法：

| 路径 | 入口/产物 | 本阶段用途 |
| --- | --- | --- |
| 框架本地 LIDU/KSC/dump | `vllm_ascend_C`、`torch.ops._C_ascend` | 原本地推理基线，不替换 |
| 原生 LI/QSFA | `torch_npu` 对应接口 | 保持框架现有调用 |
| 自研 LISSend/CopySFA | `kvcache_scatter_copy_custom_ops.custom_ops_lib` 注册 `torch.ops.nanovllm_dsa`，配套 C8 OPP | 待远端数据面适配，shadow 不调用 |
| Store 控制客户端 | 同一个 `custom_ops_lib.RemoteKvStoreClient` | shadow 已实际调用，初始化打印 `.so` 路径 |

近期分别构建安装。算子仓的 `build_and_install.sh` 构建 inplace extension 和本地 OPP，
**不等于把 Python 包安装到所有环境**。联调在启动框架前设置：

```bash
# 在算子仓构建（若已部署匹配版本则不必每次重建）：
BUILD_JOBS=32 SOC_VERSION=ascend950 ENABLE_C8_MTP_OPS=1 bash build_and_install.sh

# 保持以下路径为算子仓真实绝对路径，传给所有 worker：
export PYTHONPATH=/absolute/path/to/operator-repo/torch_extension:${PYTHONPATH:-}
python -c 'import kvcache_scatter_copy_custom_ops as p; print(p.custom_ops_lib.__file__); print(p.local_opapi_path())'
```

该包初始化会设置自己的 OPP 搜索路径，且存在历史 `torch_npu` 兼容别名；这不代表框架
`_C_ascend` 已切到新实现。部署要保留框架本地 OPP，核对实际自研 C8 `libcust_opapi.so`
路径和框架/算子/Pool 提交组合，不能只看 import 成功。复制 wheel 时还需单独部署 OPP，
不能只复制 Python 包中的 `.so`；可编辑引用适用于联调，不作为最终发布布局。

最终 vendor 建议按功能收敛，而不是复制整个实验仓：

- 将所需 LISSend、CopySFA 的 op_host/op_kernel/tiling、Torch launcher 与注册代码放进
  框架自己的 `csrc` 编译树；沿用标准 OPP 构建，来源仓库与固定 commit 写入来源清单。
- 同步引入精确版本的 wire headers、隐藏 runtime 与 Store 控制客户端；Pool server 仍为
  独立部署产物，使用同一协议版本。不把 endpoint/jetty/raw buffer 暴露成图输入。
- 框架 `dsa_offload` 保持薄适配层，runtime 不直接依赖实验脚本；迁移期间只加载一个
  新算子注册来源，避免外部 wheel 与 vendor 同名 schema 重复注册。
- 只引入明确需要的 kernel 与依赖；历史 AICPU 路径、性能实验开关、临时测试不随包复制。

vendor 本身留到动态长度与 selection-group ABI 稳定后执行；本轮没有复制算子源码到框架。
