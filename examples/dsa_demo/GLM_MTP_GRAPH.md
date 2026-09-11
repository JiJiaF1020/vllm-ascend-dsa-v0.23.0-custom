# GLM：target 图 + 三轮 MTP 合并图

离线入口统一为 `simple_test_1p.py` / `simple_test_2p.py`，共用 `simple_test.py`。
启动命令、数据集和输出目录见 [README](README.md)。脚本整理只涉及 Python，
不修改 LIM/QSFA/scatter 算子和 ABI，不需要因此重新编译算子。

## 开关

通过环境变量设置，无需修改 Python 文件：

```bash
export VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=1
export VLLM_ASCEND_SMOKE_RUN_MODE=graph
export VLLM_ASCEND_SMOKE_ENABLE_MTP=1
export VLLM_ASCEND_SMOKE_ENABLE_MTP_GRAPH=1
```

- 上述配置：target 使用 FULL_DECODE_ONLY，三轮 MTP 使用另一张合并图。
- `VLLM_ASCEND_SMOKE_ENABLE_MTP_GRAPH=0` 或命令行 `--mtp-eager`：target 图不变，MTP eager。
- `VLLM_ASCEND_SMOKE_RUN_MODE=eager`：target/MTP 都 eager。
- `VLLM_ASCEND_SMOKE_ENABLE_MTP=0`：不使用 speculative decoding。
- `VLLM_ASCEND_SMOKE_ENABLE_OFFLOAD=0`：不卸载；target 可继续使用图，MTP 自动 eager。

不卸载基线与 offload 的 MTP 层均保留全量 BF16 MLA/Indexer cache，target 层仍按
原来的 C8 设置运行。这样避免 BF16 MTP 投影把 packed FP8 cache 传给非量化 MLA Prolog。
这项 cache 修复不代表基线支持 MTP 图；当前合并图门禁仍要求 DSA offload。
比较原生基线与卸载方案时，先让两边 MTP 都 eager，并核对实际 cache/量化布局。

## 支持边界

限定 Ascend A5、`glm_moe_dsa`、DSA offload、MTP3，不按 TP/DP 数值设置白名单。
target/draft TP 必须相同；底座自动匹配。PP/PCP/DCP 均为 1、同步调度，
关闭 SP（含 compiler pass）、共享专家 DP、FlashComm2、reduce-sample。
不支持动态 EPLB/ENPU 或额外的 fine-grained lm-head TP；常规 TP 词表分片保留。
每个 TP 组必须完整落在一台机器内；新机器的拓扑、显存和 HCCL/MC2 仍需实测。

只对每请求 4 个 verification token 的纯 decode 批次选图；prefill、混合/非均匀批次
或任一 DP rank 不满足条件时，全 DP 组回退 eager。
每个 capture size 各有 target 图和 MTP 图，不是整个进程只创建两张图。
`graph_capture_sizes` 按请求数配置，底座过滤到 `max_num_seqs`，MTP3 时乘 4。

MTP 图包含 padding 清理、后两轮 position/seq_len/slot 更新、三轮 RoPE gather、
可选 C8 grouping metadata、三轮 forward、logits/greedy 采样及下一轮输入更新。
metadata 按 capture size 缓存，设备输出地址固定；CPU seq_len 原地更新。
DP metadata 同步、首轮输入整理和 staging copy、CPU bookkeeping、target rejection sampling
仍在图外。RoPE 使用 `index_select` 写入固定 cos/sin 缓冲区。

## 对照与验收

先确认模型、prompt、TP/DP、BS 和采样配置相同。默认 23k、BS1、关闭 chunked prefill/profile；
也可先设置 `VLLM_ASCEND_SMOKE_DATASET=short`。单机 TP8 设置
`VLLM_ASCEND_SMOKE_TENSOR_PARALLEL_SIZE=8` 和 `VLLM_ASCEND_SMOKE_DATA_PARALLEL_SIZE=1`。
其他拓扑见 README。依次执行：

```bash
unset ASCEND_LAUNCH_BLOCKING
set -o pipefail
python examples/dsa_demo/simple_test_1p.py --mtp-eager 2>&1 | tee mtp_eager.log
python examples/dsa_demo/simple_test_1p.py 2>&1 | tee mtp_graph.log
```

双机换为 `simple_test_2p.py`，两边传各自 `--node-rank` 和相同 `--master-addr`，
两组分别同时启动。网卡和本机 IP 使用各自环境，不要把第二台的本机 IP 配成 master IP。

1. 对比结果目录中每个 DP rank 的 token IDs 与 finish reason；eager/graph 文件名自动分开。
2. 图组应出现 `[GLM MTP graph] Enabled ...`、`Captured 3-step draft graph` 和
   `Replaying separate merged 3-step draft graph (with device preprocessing)`。
   仅有 target 的 `Replaying aclgraph` 不说明 MTP 已入图。
3. BS1 通过后增加并发/长度，覆盖不同接受长度、请求退出、空闲 rank、tail block 边界。
4. 最后开启 profile，比较同一稳态 step 边界的耗时、接受率和吞吐。
   不把 propose 变短或 take_draft 等待变长单独视为收益。
5. baseline/offload 对照时保持同样的 target 图和 MTP eager；长序列可能受全量 HBM 容量限制。

图失败时先保留第一条异常和完整 traceback，再用 `--mtp-eager` 做对照；
不要删除算子断言或把 padding 的 `-1` slot 强行改为 0。

## 本地检查范围

脚本配置与 spawn 参数传递可以在无 NPU 环境验证，`--dry-run` 不加载模型。
MTP CPU 测试覆盖固定地址 metadata、TP/BS mask 补齐、两路 hidden states、
批次缩小/空闲、DP 选图共识，以及 TP1DP8/TP2DP4/TP4DP2/TP8DP1 的 Gloo 对照。
这些检查不执行 CANN/HCCL/MC2，不能替代 Ascend 真机的精度和性能验收。
