# MoE4All 启用 qwen35/qwen35moe MTP 投机解码方案

## 概述

本方案在 MoE4All（AMD Vulkan 推理引擎）上为 qwen35/qwen35moe 架构启用 MTP（Multi-Token Prediction）投机解码，支持 paged-MoE 专家流式分页。在 RX 7700 XT 12GB 上运行 Ornith-1.5-35B-A3B（22GB，Q6_K/Q5_K/IQ4_XS 混合专家），配合正确训练的 MTP 头，greedy 模式实现 **93.6 tok/s（基线 54.7 tok/s，1.71× 加速）**，接受率 α=1.0。

## 问题背景

MoE4All 的 MTP 实现（issue #33，`docs/mtp.md`）原本仅支持 qwen35（稠密 FFN），存在三个阻塞点使 qwen35moe（MoE）模型无法使用：

1. **头加载**：`load_mtp_head` 仅接受 `cfg.qwen35` 的稠密 FFN；qwen35moe 的头是 MoE FFN（路由专家 + 共享专家）
2. **验证路径**：`generate_dense_backend` 的 VERIFY 分支在 `c.moe.is_some()` 时直接跳过
3. **朴素权重上传**：MTP 驱动的 bind 闭包将主干全部权重原始上传到显存（模型大于显存即 OOM）

此外，Ornith-1.5 官方发布的 MTP 头从未训练过（随机初始化——权重统计 std≈0.02、kurt≈3.0，参见 [shisa-ai 的分析](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY)）。验证时使用了 shisa-ai 的 KL 蒸馏嫁接头（Qwen3.6 头移植 + 12K 轨迹蒸馏）。

## 改动内容

### 阶段 1：qwen35moe MTP 头支持（`crates/infr-llama/src/mtp/mod.rs`）

- `MtpFfn` 枚举：`Dense { gate, up, down }` | `Moe { gate_inp, gate_exps, up_exps, down_exps, shexp }`
- `load_mtp_head`：经 `cfg.moe.is_some()` 接受 qwen35moe，形状全部从 Config 派生
- `upload_mtp_head_bufs`：按 FFN 变体的可变权重列表
- `build_mtp_graph` + `build_mtp_draft_chain_graph`：MoE 变体发射 `Op::MoeFfn` + 共享专家 + `Op::MoeSharedExpertAdd`

### 阶段 2：验证路径 + 回滚

- `runner.rs` VERIFY 门禁放宽：qwen35moe 在 `moe_batched_ok`（所有专家 dtype ∈ `MOE_MMQ_DTYPES`）时准入
- DeltaNet 回滚过滤器对 qwen35moe 自动正确（`is_qwen35_attn_layer` 覆盖 interval=4 混合结构）
- 单元测试：`mtp_delta_filter_covers_qwen35moe_recurrent_layers`

### 阶段 3：分页 MoE verify 集成

- **`mtp/backends.rs`**：朴素全量上传 bind 替换为 `vulkan_moe_binder`（与正常路径相同的放置规划器 + 分页器安装器）；新增 `generate_mtp_spec_vulkan_timed_on_state` 支持冷/热绑定器分离
- **`chat/vulkan.rs`**：MTP 分支路由至 `ensure_session()` + `PlacementScope::enter()` 共享会话后端；`mtp_trunk: Option<SeamKv>` 跨周期持久化
- **`seam/mod.rs`**：VRAM 规划器中 MTP 头空间预留改为动态计算（根据 GGUF 头层张量实际字节数 + 词表嵌入表大小，替代硬编码 2 GiB）

### 阶段 4：贪心快路径

- GPU argmax 接受路径（`Op::Argmax`，m×4 字节回读）已存在；temp>0 时的全量 logits D2H 回退（m×vocab×4B，每 verify 4-11 MB，~25 MB/s）是主要开销
- 新增 `INFR_MTP_N_MAX` 环境变量调优草稿长度（默认 6；边缘头建议 4）

## 已知限制

1. **实践中仅 greedy 可用**：temp>0 使用 `run_verify_full` 下载完整 m×vocab×4B logits/周期（D2H ~25 MB/s over PCIe）。需要 GPU 端随机接受或持久 staging 缓冲才能支持采样 MTP。
2. **头会话每请求重建**："no cross-turn KV reuse" 设计（backends.rs）每次 `generate()` 重建 trunk+head 会话。跨轮持久化可消除 ~300 ms 请求间开销，但需解决自引用借用问题（backends.rs 有文档）。
3. **QkNormMrope 无融合 KV 写**：视觉 mrope 路径发射 `Op::QkNormMrope` + 显式 `Op::WriteKv`（无 peephole 融合）。mrope 图也不支持 decode replay。
4. **接受率是模型属性**：Ornith-1.5 发布的头从未训练过（随机初始化），需配合重训头（如 shisa-ai 的 KL 蒸馏嫁接）才能产生净加速。

## 性能

RX 7700 XT 12GB，Ornith-1.5-35B-A3B-APEX-MTP-I-Quality-MTPFIX（嫁接 shisa 蒸馏头），8K 上下文，greedy：

| 配置 | decode |
|------|--------|
| 基线（无 MTP）| 54.7 tok/s |
| MTP（n_max=6，α=1.0）| **93.6 tok/s（1.71×）** |

Serve 模式串行引擎（MTP 可用）：prefill 446 tok/s，decode 69.5 tok/s @12.5K prompt。

## 测试

- `cargo test -p infr-cpu` — 98+6 通过（含新增 QkNormMrope 文本坍缩 + 平面选择测试）
- `cargo test -p infr-vision` — 15 通过
- `cargo check --workspace` — 无错误
- 端到端：`INFR_MTP=1 infr run <qwen35moe-gguf> --temp 0`，MTP 摘要日志输出每周期 α + 聚合统计
