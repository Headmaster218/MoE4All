# MoE4All 最新进度（2026-08-31 深夜）

## 一句话

在 RX 7700 XT 12GB 上，为 MoE4All 补齐了 qwen35moe MTP 投机解码和 Qwen3-VL 视觉理解两条完整链路，文本 42→93.6 t/s，视觉 0.2s/图，全部端到端验证通过。

---

## 速度成绩

| 场景 | 引擎 | 速度 |
|------|------|------|
| Ornith Quality serve @131K | MoE4All | **42 t/s**（基线，稳定可用）|
| Ornith Quality + greedy MTP | MoE4All | **93.6 t/s**（1.71×，run 模式）|
| Qwen3.8 IQ1_S @128K | MoE4All | 15.8 t/s |
| 同上 | llama.cpp | 10.3 t/s |

---

## MTP：从"随机头"到 1.71×

### 诊断链

1. 两引擎实测 Ornith 头接受率仅 0.08-0.33 → 怀疑头质量
2. shisa-ai 独立发现：`mtp.*` 张量为随机初始化（std 0.02, kurt 3.0）
3. 官方 Qwen3.5-0.8B 头同引擎 α=0.47-0.55 → 排除实现问题
4. 结论：**Ornith 微调未重训 MTP 头**（模型属性，非工程问题）

### 工程实现（四阶段，~1500 行）

| 阶段 | 内容 |
|------|------|
| 1 | MtpFfn 枚举（Dense/Moe）+ qwen35moe 头加载 + MoE FFN 图发射 |
| 2 | VERIFY 门禁放宽 + DeltaNet 回滚覆盖 + 单元测试 |
| 3 | MTP 驱动接入 vulkan_moe_binder 分页绑定器 + 会话共享 + 动态 VRAM 预留 |
| 4 | 贪心快路径：GPU argmax 接受（28 字节下载替代 11MB logits）|

### 最终效果

| 模型 | MTP 头 | greedy α | 速度 |
|------|--------|---------|------|
| Ornith Quality-MTPFIX（嫁接 shisa 蒸馏头）| 1.000 | **93.6 t/s** |
| Ornith Mini 原生 | 0.077 | 0.5 t/s（负优化）|
| Qwen3.5-0.8B 官方 MTP | 0.47-0.55 | 133 t/s（基线 218，launch-bound 无收益空间）|

**关键发现**：Ornith 官方发布的 MTP 头是随机初始化的（从未训练），接受率必然 ≈0。这不是 MoE4All 或 llama.cpp 的问题——两个独立引擎交叉确认。

---

## 视觉：从零到端到端

### 架构

Qwen3-VL（qwen3vl_merger）：27 层 ViT → 2×2 空间合并 → mm.0→GELU→mm.2 → [n_tokens, 2048] 注入 LM。

### 实现

| 组件 | 内容 |
|------|------|
| infr-vision crate | ClipConfig 解析 + VisionWeights 目录 + 图像预处理（smart resize/patchify/merge-major/双线性 pos-embed）|
| Op::Gelu | exact-tanh，CPU + Vulkan shader |
| Op::Rope2D | ggml VISION 语义（逐行核对 llama.cpp 源码），CPU + Vulkan shader |
| Op::QkNormMrope | 文本侧 IMROPE（fused RMSNorm + 多平面 RoPE），CPU + Vulkan shader |
| VitEngine | CPU + Vulkan 双后端，27 层 ViT 完整前向 |
| 拼接 | 图像段嵌入替换 + 批量/逐 token 双路径 + 前缀缓存防护 |

### 验证

- 15 个单元测试 + GPU parity 测试全通过
- 真实 mmproj 冒烟：0.2s/图（Vulkan），输出与 CPU parity 0.018
- 端到端：红圆/颜色判别/多轮/流式/错误处理全通过

---

## 视觉性能

| 指标 | 数据 |
|------|------|
| mmproj 加载 | 1.5s |
| ViT 编码 256×256 | 0.19s（Vulkan）|
| prefill 280 tokens | ~9s |
| decode | 69-85 t/s |

---

## 修复的关键 bug

| # | bug | 影响 |
|---|-----|------|
| 1 | CPU ViT V 列切分错误（qkv[2dn..] 扁平尾部）| V 全乱 → 视觉输出噪声 |
| 2 | IQ4_XS 字节公式错误（18/32 → 136/256）| GGUF 嫁接偏移错位 |
| 3 | Rope2D theta_scale（-2/hd → -2/n_pairs）| ViT 位置编码错误 |
| 4 | MTP 驱动朴素上传（分页模型 OOM）| 接入 vulkan_moe_binder |
| 5 | 逐 token 循环缺 ViT 拼接 | 图像 token 喂了词嵌入 |
| 6 | Copy f32→f16 不支持 | 测试方法修正 |

---

## 当前状态

### 可直接使用
```powershell
# 日常 API（42 t/s @131K，含视觉）
infr.exe serve <Quality.gguf> --mmproj <mmproj.gguf> --ctx 131072 --set kv.type_k="q8_0" --set kv.type_v="q8_0"

# 终端 MTP 贪心聊天（93.6 t/s）
Desktop\Ornith-MTP-chat.cmd
```

### 待做

| 优先级 | 任务 | 说明 |
|--------|------|------|
| 1 | serve+MTP VRAM 动态预留（已写代码待验证）| 差 272MB，动态计算公式已实现 |
| 2 | 并行引擎对嫁接文件乱码 | 独立 bug，MTP 用串行绕过 |
| 3 | 随机采样 GPU-side accept | temp>0 路径提速 |
| 4 | 上游 PR | MTP + 视觉分批提交 |

---

## 文件索引

| 文件 | 内容 |
|------|------|
| MILESTONE-REPORT-0831.md | 阶段性报告 |
| ENGINEERING-LOG.md | 完整工程记录 |
| MTP-PR-DRAFT.md | 上游 PR 草稿（英文）|
| MTP-PR-DRAFT-CN.md | 上游 PR 草稿（中文）|
| graft_v2.py (Temp) | GGUF 头嫁接工具 |
| Desktop\Ornith-MTP-chat.cmd | MTP 贪心聊天启动脚本 |
| Desktop\MTPFIX.gguf | 嫁接后模型（22.08 GiB）|
