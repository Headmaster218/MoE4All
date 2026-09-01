# MoE4All MTP + 视觉：工程实现完整记录

## 项目概述

在 MoE4All（AMD Vulkan 推理引擎）上实现并激活了两项原版缺失的能力：
1. **MTP 投机解码**（qwen35moe 架构，含分页 verify + DeltaNet 回滚）
2. **Qwen3-VL 视觉理解**（mmproj 解析 → ViT 前向 → IMROPE → embedding 拼接 → API）

硬件：AMD RX 7700 XT 12GB / 64GB DDR4 / i5-12600KF / Windows 11
模型：Ornith-1.5-35B-A3B（qwen35moe，22GB，256 专家，40 层 + 1 MTP 层）

---

## Part 1: MTP 投机解码

### 1.1 架构分析

Ornith 的 GGUF 在 blk.40 打包了完整的 MTP/NextN 层：
- 注意力：Q6_K 交错 q+gate 布局，qk_norm，NEOX RoPE（64/256 维）
- FFN：MoE（256 专家选 8 + sigmoid 门控共享专家，与主干同构）
- NextN 桥接：eh_proj [4096,2048] + enorm/hnorm + shared_head_norm
- 无 hc_*（超连接）和 indexer（QSA）张量——头是标准注意力层 + MoE FFN

### 1.2 三阶段实现

**阶段 1：头加载 + MoE FFN 图**
- `MtpFfn` 枚举（Dense{gate,up,down} | Moe{gate_inp, gate_exps, up_exps, down_exps, shexp})
- `load_mtp_head` 经 `cfg.moe.is_some()` 分支自动适配 qwen35/qwen35moe
- 形状全部从 Config 派生（零硬编码）
- `Op::MoeFfn` + `Op::MoeSharedExpertAdd` 的图发射，支持 rows=1（草稿链）和 rows>1（追赶批）

**阶段 2：验证门禁 + 回滚**
- runner 验证门禁放宽：`verify_moe_ok = moe.is_none() || (!deepseek4 && !qwen4exp && moe_batched_ok)`
- DeltaNet 回滚覆盖：`is_qwen35_attn_layer` 过滤器对 qwen35moe 自动正确（interval=4）
- 单元测试：`mtp_delta_filter_covers_qwen35moe_recurrent_layers`

**阶段 3：分页 verify + 会话集成**
- MTP 驱动从朴素全量上传改为 `vulkan_moe_binder`（分页感知绑定器）
- `MropePlan`/trunk_state 持久化跨请求
- `DenseSeamChat` 增加 `mtp_trunk: Option<SeamKv>` 字段
- VRAM 预留动态计算（头权重 GGUF 字节数 + embed 表 + KV + scratch）

### 1.3 贪心快路径修复

**根因**：temp>0 时每 verify 下载 m×vocab×4B 全量 logits（最大 11MB），D2H 仅 ~25MB/s → 每周期 +160-800ms。

**修复**：temp=0 走 GPU argmax 路径（m×4B = 28 字节），verify 下载 800ms → 1.8ms（**400×**）。

### 1.4 接受率调查

| 模型头 | greedy α | 来源 |
|--------|---------|------|
| Ornith Quality 原生头 | ≈0 | 随机初始化（shisa 独立证实：std 0.01993, kurt 2.997）|
| Qwen3.5-0.8B 官方头 | 0.47-0.55 | 官方训练 ✅ |
| shisa KL 蒸馏头（嫁接后） | **1.000** | 完美匹配微调主干 |

结论：Ornith 微调时未重训 MTP 头（随机初始化权重），shisa-ai 通过 Qwen3.6 头移植 + 12K 轨迹 KL 蒸馏修复。

### 1.5 最终性能

| 配置 | decode |
|------|--------|
| 基线（无 MTP）| 54.7 t/s |
| MTPFIX + greedy MTP (n_max=6) | **93.6 t/s (1.71×)** |

---

## Part 2: Qwen3-VL 视觉理解

### 2.1 mmproj 结构

```
clip.projector_type = qwen3vl_merger
ViT: 27 层, embd 1152, head_dim 72, patch 16, merge 2
输出: [n_tokens, 2048]（与 LM n_embd 对齐）
deepstack: 惰性（is_deepstack_layers 全零）→ 免实现
```

### 2.2 新增 crate: infr-vision

```
config.rs      → ClipConfig（从 GGUF 解析全部参数）
weights.rs     → VisionWeights 张量目录（形状校验）
preprocess.rs  → smart_resize / patchify / merge-major / bilinear pos-embed
vit.rs         → VitEngine（CPU 前向）+ VkVit（Vulkan 前向，0.2s/图）
```

### 2.3 新增 Vulkan 算子

**Op::Gelu**（exact-tanh）
- shader: gelu.comp
- 用于 ViT FFN 和 merger MLP

**Op::Rope2D**（ggml GGML_ROPE_TYPE_VISION）
- 2D 位置 (y,x) per patch，merge-major 序
- sections {d/4}×4 按 pair 计数
- theta 累加器每 section 重置
- split-half 配对覆盖全 head
- CPU 实现精确 + Vulkan shader rope2d.comp
- parity 测试：CPU vs Vulkan max_err < 1e-5

### 2.4 文本侧 IMROPE

qwen35moe 的 rope_sections = [11,11,10,0]（从 GGUF 解析）：
- 纯文本 T=H=W → 坍缩为标准 NEOX（零改动即正确）
- 图像段 T=const, H=base+row, W=base+col
- 新算子 `Op::QkNormMrope`（fused per-head RMSNorm + IMROPE）
- Vulkan shader qk_norm_rope_mrope.comp
- 文本坍缩测试保证纯文本路径零风险

### 2.5 Embedding 拼接

- 图像段的 token 嵌入替换为 ViT 输出行（×embed_scale）
- 批量预填路径 + 逐 token 路径双覆盖
- 前缀缓存防混叠：视觉回合禁用跨图前缀复用

### 2.6 性能

| 阶段 | 耗时 |
|------|------|
| mmproj 加载（f32 反量化）| 1.5s |
| ViT 编码 256×256（CPU）| 7-19s |
| ViT 编码 256×256（Vulkan）| **0.19s** |
| prefill 280 tokens | ~9s |

---

## Part 3: 视觉端到端验证

| 测试 | 结果 |
|------|------|
| 红圆图："solid red circle on white background" | ✅ 与 llama.cpp 一致 |
| 颜色判别 red/green | ✅ 全对 |
| 多轮视觉对话 | ✅ |
| 流式 SSE | ✅ |
| 错误处理（非法 base64）| ✅ 干净报错 |
| 3 路并发 | ✅ 稳定 |

---

## Part 4: 关键 bug 与修复记录

| # | bug | 修复 |
|---|-----|------|
| 1 | MTP 头 MoE FFN 缺失 | 新增 MtpFfn 枚举 + 图发射 |
| 2 | 验证门禁拒绝 qwen35moe | 放宽 + moe_batched_ok 检查 |
| 3 | MTP 驱动朴素上传 22GB→OOM | 接入 vulkan_moe_binder 分页绑定 |
| 4 | VRAM 预留不足 2MiB | 2GiB→动态计算 |
| 5 | CPU ViT V 列切分错误 | `qkv[2dn..]` → 按行列切分 |
| 6 | Rope2D theta_scale 公式 | theta^(-2/n_pairs) 非 theta^(-2/hd) |
| 7 | 逐 token 循环缺 ViT 拼接 | host-embed 路径增加 span 覆盖 |
| 8 | 头会话重建 OOM | 动态 VRAM 预留（GGUF 字节数计算）|

---

## Part 5: 文件清单

### 新增
```
crates/infr-vision/           视觉 crate（config/weights/preprocess/vit）
crates/infr-vulkan/shaders/gelu.comp
crates/infr-vulkan/shaders/rope2d.comp
crates/infr-vulkan/shaders/qk_norm_rope_mrope.comp
```

### 修改
```
crates/infr-core/src/graph.rs       Op::Gelu + Op::Rope2D + Op::QkNormMrope
crates/infr-cpu/src/lib.rs          三个新算子的 CPU 解释器
crates/infr-vulkan/src/adapter.rs   算子 lowering
crates/infr-vulkan/src/recorder.rs  rope2d + gelu 录制方法
crates/infr-vulkan/src/gemm.rs      SPV getter
crates/infr-vulkan/build.rs         shader 构建项
crates/infr-llama/src/mtp/mod.rs    qwen35moe 头 + MoE FFN + 分页 verify
crates/infr-llama/src/mtp/backends.rs  分页绑定器 + 状态传播
crates/infr-llama/src/seam/mod.rs   MTP 预留 + MropePlan
crates/infr-llama/src/seam/runner.rs 验证门禁 + 拼接 + mrope 发射
crates/infr-llama/src/chat/vulkan.rs generate_mm + MTP 分支
crates/infr-server/src/lib.rs       image_url 收集
crates/infr-chat/src/lib.rs         ChatMessage.images
crates/infr-cli/src/main.rs         --mmproj + GenBackend::generate_mm
```

### 测试
```
infr-cpu:   qk_norm_mrope 文本坍缩 + 平面选择（2 通过）
infr-vision: 15 通过（resize/bilinear/merge-major/gelu/rope2d/linear/layernorm + 真实 mmproj 冒烟）
infr-llama:  qk_norm_mrope_parity packed + strided（GPU vs CPU）
infr-chat:   serde images 默认空
infr-server: image_url 收集
```

---

## Part 6: 上游 PR 建议

1. **qwen35moe MTP 支持**（阶段 1-3 + 贪心快路径）：独立完整，可直接提交
2. **Op::Gelu + Op::Rope2D**：通用算子，独立提交
3. **视觉支持**：依赖较多，建议分批
4. **f32 边际重验**：解停放 MTP 的 accuracy mitigation

## Part 7: 性能天花板分析

42 t/s 的瓶颈 = 分页器每 token ~283 次专家块缺失的小 DMA 延迟（~80µs/次），PCIe 带宽仅用 1/3。证据：RX 7900 XTX（24GB、带宽 2×）同为 41.5 t/s。突破需上游实现异步缺失预取或增加显存使全部专家驻留。
