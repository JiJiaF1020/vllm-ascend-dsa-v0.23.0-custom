# History

蓝绿协作轮次档案。只追加，不修改。每轮至少记录：派单 commit、执行环境、原始数据出处、判读结论与置信度。

## 轮次记录格式

```
## §R<n> <标题>（<日期>）

**派单**（commit <id>）：<一句话任务 + 判读通过标准>
**交付**（commit <id>）：<一句话结果 + 数据出处>
**判读**（蓝区）：<阳性/阴性/无效 + 依据 + 下一步>
```

## §R1 remote-offload 单机整网首发（2026-10-26）

**派单**（commit 见本节首次提交）：框架安装（含 catlass 本机拷贝修复）→ 算子仓构建 → 8 卡 EID 探测 → embedded Pool → DP1/TP1 eager BS1 实跑；全绿则放大 DP4/TP2。通过标准写死于 pingpong.md §R1：capacity report 须含 `indexer=21, resident=57, mtp=1`，日志须依次命中 EXECUTION_PLAN/STORE_PUT_DONE/CONNECT_BEGIN/CONNECT_OK(real_store)/≥2 轮 CONSUMED/DRAIN_OK 非零，进程 RC=0。
**交付**：待绿区回传。
**判读**（蓝区）：待定。

背景存档：算子仓公开镜像 `JiJiaF1020/a5-remote-offload-ops`（main=6e85590db）；本分支公开镜像在 `JiJiaF1020/vllm-ascend-dsa-v0.23.0-custom` 的 `vllm-ascend-v0.23.0-custom-remote-offload`。LIM 1..4 行校验在本分支未变（csrc/.../vllm_a5_li_manage_c8_torch_adpt.h:98-100），首发 eager + BS1 规避；切 graph 前须另行核对 capture 档。
