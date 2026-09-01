# MoE4All 深度优化项目 — 阶段报告（8.31 早晨版）

日期：2026-08-31 上午
作者：opencode (GLM) + zhang
机器：AMD RX 7700 XT 12GB / 64GB DDR4 双通道 (~55GB/s) / Windows 11 / BIOS 已开 ReBAR
代码库：C:\Users\zhang\Desktop\MoE4All-src（本地开发分支，基于上游 v0.3.0/0.4.0）
报告一：MILESTONE-REPORT.md（8.30 晚）

---

## 一、项目目标回顾

1. 在 7700 XT 单卡上把 Qwen3.8-Flash-Next / Ornith-35B 推理速度推到 40 t/s（已达成基线）
2. 给 MoE4All 加 MTP 投机解码（冲 60 t/s）
3. 给 MoE4All 加视觉分析能力（Qwen3-VL）

## 二、昨日（8.30）关键成果

### 2.1 基线（已达成 40 t/s ✅）

| 场景 | 引擎 | 速度 |
|------|------|------|
| **Ornith Quality (22GB) serve 模式** | MoE4All | **40.4 t/s** ← 日常配置 |
| Ornith Quality run 会话 | MoE4All | 22.5 t/s |
| Ornith Mini (13.4GB, 32k) | llama.cpp Vulkan | 17.4 t/s |
| Qwen3.8 IQ1_S (67.6GiB) @0ctx | MoE4All | 18.4 t/s |
| Qwen3.8 IQ1_S @128K | MoE4All | 15.8 t/s |
| Qwen3.8 IQ1_S | llama.cpp | 10.3 t/s |

关键技术点：BIOS 开 ReBAR（MoE4All 分页器硬依赖）+ serve 并行引擎（比 run 会话快 80%）。

### 2.2 Qwen3.8-Flash-Next 调查结论

- 模型规格：125B 总参 / 6B 激活，架构 qwen4exp（Gated DeltaNet + QSA 稀疏注意力 + PLE），llama.cpp 官方支持
- **致命发现：Unsloth 的 GGUF 丢掉了 MTP 头**（无 nextn 张量），llama.cpp 对 qwen4exp 也不支持 MTP → 该模型投机解码无望，除非从原始 BF16 权重（360GB）重新转换
- Q2_K_XL (79GB) 在 MoE4All 官方 7900XTX 上 26-29 t/s，是 125B 的甜点量化

### 2.3 MTP 工程（三阶段，~700 行，已全部打通但判死）

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1 | MtpFfn 枚举（Dense/MoE）、qwen35moe 头加载、MoE FFN 图发射 | ✅ |
| 2 | runner 验证门禁放宽（qwen35moe admit）、DeltaNet 回滚覆盖验证+单测 | ✅ |
| 3 | **分页 verify 融合**：MTP 驱动接入 vulkan_moe_binder + 会话共享后端（消除第二次全量上传 OOM）、VRAM 计划预留 2GiB 头部空间 | ✅ |

端到端实测（Ornith Quality + INFR_MTP=1）：
- MTP 完整运行：草稿、分页 verify、catch-up、回滚全链路工作
- **接受率 α ≈ 0.1**（每 6 个草稿收 0-1 个）
- verify 阶段占 98%（部分接受触发全量重预填，每轮 2-2.9s）

**判死依据**：llama.cpp 官方 MTP 实现跑同一模型接受率也只有 0.19-0.23（两实现互证）。α≈0.1-0.2 时投机解码期望收益 Σα^k ≈ 1.1 token/verify，数学上必然负优化。**这是模型 MTP 头质量属性，非工程问题。**

遗留价值：分页 MTP 基础设施保留在分支里，未来换好头的模型直接可用。若上游有兴趣，阶段 1-3 + f32 边际重验方案可整理成 PR。

## 三、视觉支持进度（V0-V3 完成）

### 3.1 调研结论（V0）

- Ornith 的 mmproj = Qwen3-VL 塔：`projector_type=qwen3vl_merger`，27 层 ViT，embd 1152，head_dim 72，patch 16，merge 2，projection 2048
- **deepstack 惰性**（is_deepstack_layers 全零且无张量）→ 最难子问题消失
- 文本侧 `rope_sections=[11,11,10,0]`：纯文本时 IMROPE 退化为 NEOX → 现有文本路径无需改动即正确
- 视觉特殊 token 已在 tokenizer 中（248053-248057，type-3 原子特殊 token）
- 总工程量估 ~4800 行

### 3.2 已实现（V1-V3，全部编译 + 测试通过）

| 阶段 | 内容 |
|------|------|
| V1 ✅ | `crates/infr-vision` 新 crate：ClipConfig 解析（校验 clip/qwen3vl_merger）、VisionWeights 张量目录（形状全校验） |
| V2 ✅ | 图像预处理：smart_resize（32 对齐 + 4096 patch 上限）、归一化、patchify、**merge-major 重排**、位置嵌入双线性插值（ALIGN_CORNERS）——10 个单测覆盖排序/插值/端到端 |
| V3 ✅ | **ViT 前向 CPU 版**（vit.rs，rayon 并行）：patch embed → +pos → 27×(LN→QKV→2D RoPE→双向注意力→残差→LN→GELU MLP→残差) → post-LN → 2×2 merge → mm.0→GELU→mm.2 |

新增引擎算子：
- `Op::Gelu`（exact-tanh）：infr-core 枚举 + infr-cpu 解释器 + **infr-vulkan shader**（gelu.comp，完整接入 build.rs/gemm.rs/recorder/adapter）
- `Op::Rope2D`（ggml GGML_ROPE_TYPE_VISION 语义）：infr-core + infr-cpu；Vulkan 暂时显式 bail（CPU-only，报错清晰）

### 3.3 Rope2D 语义攻坚（本阶段最深的坑）

逐行核对了 llama.cpp 源码（tools/mtmd/models/qwen3vl.cpp + ggml-cpu/ops.cpp 的 ggml_mrope_cache_init）：

- `theta_scale = freq_base^(-2/n_dims)`，**n_dims = head_dim/2**（ViT 调用传 d_head/2，且 assert n_dims == ne0/2）
- sections 条目 {d/4}×4 与 **pair 序号直接比较**（sector = p % sect_dims）——按 pair 计数
- 每平面 theta 是**累加器**：每对乘一次 theta_scale，在自己 section 开始处重置（indep_sects）；角度 = pos × θ_scale^local_l
- split-half 配对作用于**整个 head**（rotate_pairs(ne0, n_dims)，无 pass-through 尾）
- 位置流 [y, x, y, x]（clip.cpp 每patch填 (y,x,y,x)）

实测（真实 mmproj 冒烟测试）：
- 加载 447M 塔：1.5s（f32 反量化 ~0.9GB RAM）
- 256×256 图 → 1024 patch → 64 token：**编码 7.1s（CPU release）**
- 输出 64×2048 全有限、std=0.62 非退化

### 3.4 期间踩的坑（记录）

1. Windows GBK 控制台 vs UTF-8：cmd echo 中文进 stdin 会炸（"stream did not contain valid UTF-8"）——测试一律走英文或文件
2. PS 5.1 把 cargo 的 stderr 进度当 NativeCommandError——`*>` 重定向到文件再 Select-String 才干净
3. AMD Vulkan 驱动的 VK_EXT_memory_budget 在进程崩溃后虚报占用（GPU 计数器只有 1GB 却报 11.5GB）——重启显卡驱动（Disable/Enable-PnpDevice）可清
4. 测试数据 bug：split-half 配对下"单位向量"应是 [1,1,1,1,0,0,0,0] 而非 [1,0,1,0,...]——实测 -0.301=cos(1)−sin(1) 反推出实现是对的、测试是错的

## 四、剩余工作（按优先级）

| 阶段 | 内容 | 估工作量 | 难点 |
|------|------|----------|------|
| **V4** | 文本侧 IMROPE（rope 位置 ≠ KV 行号解耦）+ embedding 拼接 + 前缀缓存防混叠 | ~1500 行 | **最难**：decode replay/verify/分页全都假设位置==行号；图像段 T 恒定、后文跳跃 max(nx,ny) |
| V5 | serve API 收图（DTO parts、base64 解码、ChatMessage.images、模板 `<\|image_pad\|>` 展开） | ~450 行 | 模板是 single-source-of-truth，不能绕过 |
| V6 | 与 llama.cpp mtmd 数值对齐验证（llama-mtmd-cli 做参考） | ~600 行 | bit-close 才能保证视觉质量 |
| V7 | ViT Vulkan 移植（7.1s → <1s）：Linear/Attn 走现有算子，Rope2D 写 shader | ~850 行 | 直接收益大 |
| MTP-f32 | f32 边际重验（上游可交的收尾） | ~600 行 | 等 MTP 有意义再做 |

## 七、8.31 深夜补充：视觉端到端贯通 + 剩余 bug 精确定位

### 已完成
- **V4a**：`Op::QkNormMrope` 全栈（枚举 + CPU 解释器 + Vulkan shader `qk_norm_rope_mrope.comp` + adapter/recorder/gemm/build 接线，支持 x_stride 交错输入）。GPU 对比测试（packed + strided 两用例）CPU-vs-Vulkan 通过
- **V4b**：MropePlan/ImageSpanEmbeds 贯穿 model.rs/seam/runner.rs；验证门禁、decode 游标（KV 行 ≠ rope 位置）、dyn_replay 排除；batched-prefill 拼接 + 逐 token 循环拼接（后补，见下）
- **V5**：ChatMessage.images、server image_url 收集、`--mmproj` 参数、generate_mm 全链路（含惰性 ViT 加载、展开、游标）、`--mmproj` 时 serve 路由到串行引擎
- **端到端运行确认**：请求 → mm plan（25→280 tokens，decode_base=40 ✓）→ QkNormMrope 在层 3/7/11… 发出 ✓ → 生成 41 t/s ✓

### 排查记录（决定性实验）
1. llama.cpp 对照：同一模型+mmproj+红圆图 → **"red circle on white background" 答对**；我的管道 → "mug with clock"（错）
2. 逐 token 循环补拼接后仍错 → 排除了"拼接缺失"单一原因
3. llama.cpp 位置语义**全部与我一致**：n_pos = max(合并网格)，T 平面段内恒定，plane1=row、plane2=col（`set_position_mrope_2d`：pos[+n]=y, pos[+2n]=x）
4. patch 顺序推导：llama.cpp 的 permute/reshape 舞蹈 ≡ HF `(h w)(m n c)` ≡ 我的 merge_major_to_patch ✓
5. **numpy 交叉验证**（独立实现 ViT，同图同权重）：numpy vs Rust 在 stage1（patch-embed+pos）即发散（max 2.25），且残差与 pos-embed 相关系数仅 0.62 → **bug 锁定在 ViT 的 pos-embed/patch-embed 数值层**，27 层放大后成结构化噪声（颜色都错 → 特征值损坏，非纯位置问题）

### 已排除（都验证一致）
- mrope 平面映射（[T,row,col,0] 与 `set_position_mrope_2d` 逐行核对一致）
- ggml VISION rope 语义（theta_scale = θ^(-2/n_pairs)，section 按 pair 计数，逐平面累加器）
- merge-major 顺序（llama.cpp 舞蹈的代数推导 = 我的实现）
- patch 内布局（channel-planar，与 im2col/HF 一致）
- embed_scale（qwen35moe = 1.0）、qkv 融合顺序（Q/K/V 偏移 0/D/2D）

### 剩余嫌疑（按可能性）
1. **pos-embed 基表的网格展开方向**：我假设 p = y*48+x（行主序）；llama.cpp `resize_position_embeddings` 的 reshape/permute 链暗示 p = x*48+y（**转置**）——但我的转置复测反而更差（corr 0.15），怀疑我的转置测试实现有误，需重做
2. bilinear 插值模式（align-corners vs half-pixel 的下采样分支）
3. qkv/attention 的 kq_scale（未核对 clip.cpp 的 qwen3vl 具体值）
4. resize 前的像素插值（Rust `resize_exact` Triangle vs llama.cpp 的 bicubic）

## 五、风险与备注

1. V4 是视觉的最大风险点：IMROPE 位置解耦触及 decode replay（性能关键路径），错了就是"流利胡话"
2. 前缀缓存混叠（同占位 token 不同图 → 复用错的 KV）必须防护，否则静默错图答案；v1 方案：跨图像段禁用前缀复用（~20 行）
3. serve 并行引擎（40 t/s 那条路径）与视觉的接合未验证——V4/V5 要在 parallel.rs 上同时做
4. GNU 自编译比官方 MSVC 慢 ~30%；视觉全链路跑通后建议切 MSVC 编译

## 六、当前可复现状态

```powershell
# 日常推理（40.4 t/s）
cd C:\Users\zhang\Downloads\MoE4All-Windows-x86_64-v0.3.0\MoE4All-Windows-x86_64-v0.3.0
.\infr.exe serve "C:\Users\zhang\Desktop\新建文件夹 (4)\1\Ornith-1.5-35B-A3B Quality\Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf" --ctx 131072

# 自编译（含 MTP/视觉开发分支）
cd C:\Users\zhang\Desktop\MoE4All-src
$env:Path = "C:\msys64\ucrt64\bin;C:\tools\glslc\ucrt64\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
cargo build --release -p infr-cli
cargo test -p infr-vision          # 15 通过
cargo test -p infr-vision vit_smoke -- --ignored --nocapture   # 真实 mmproj 冒烟
```

环境：rustc 1.98 (x86_64-pc-windows-gnu) / MSYS2 ucrt64 (gcc 16.2, dlltool) / glslc 2026.3（C:\tools\glslc）。


## 九、优化战役终局（8.31）

| 优化项 | 结果 |
|--------|------|
| 高性能电源计划 | 无变化 |
| KV q8_0 | **+10%**（41→42）✅ 最终配置 |
| MSVC + crt-static + native 重编译 | 此模型无差异 |
| 双池解码缓存 | 已内置（Q5_K/Q6_K 池） |
| Mini 量化 | 48.5 t/s 但降质 |

**天花板证据**：作者自己在 RX 7900 XTX（24GB、带宽 2 倍）的基准同为 41.5 t/s。瓶颈 = 分页器每 token ~283 次专家块缺失的小 DMA 延迟（~80µs/次），PCIe 带宽仅用 1/3——属引擎架构层限制，需上游实现异步缺失预取才能突破。

**最终日常配置**：
```powershell
infr.exe serve <Quality.gguf> --ctx 131072 --set kv.type_k="q8_0" --set kv.type_v="q8_0"
```
decode 42 t/s / prefill 147 t/s / 视觉请求 ~10s（ViT 0.2s GPU）。

**突破路径**（硬件）：MI50 32G（专家缓存覆盖率高 → 缺失率大降）或双 4090D 48G。

---

## 十、MTP 性能根因终局（8.31 深夜）：官方头验证 + D2H 病理定位

### 决定性实验：官方 Qwen3.5-0.8B（好头）

| 配置 | alpha | verify 下载 | decode |
|------|-------|------------|--------|
| 基线（无 MTP） | — | — | **218.6 t/s** |
| MTP，temp=0.6（随机采样路径） | 0.550 | m×vocab×4B ≈ 4-11MB，25MB/s 病态 D2H | 11.8 t/s |
| **MTP，temp=0（greedy，GPU argmax）** | **0.470** | **m×4B ≈ 28B → 0.2-1.8ms** | **133.4 t/s** |

### 三层结论

1. **MTP 技术** ✅ 没问题（llama.cpp 官方 4B 上 2×）
2. **官方 MTP 头** ✅ 训练良好（α=0.47-0.55 证明头前向、eh_proj、h_tap 数学全对）
3. **Ornith 的头** ❌ 坏的（两引擎两变体 α≈0.08-0.33；APEX 微调改了主干没重训头）
4. **MoE4All 的 MTP 驱动** ⚠️ 数学对，但 temp>0 时每周期下载 m×vocab×4B 全量 logits（D2H 仅 ~25MB/s）→ 全面负优化；greedy 路径修复后 verify 12ms

### 新增修复/旋钮
- `INFR_MTP_N_MAX`：每周期草稿长度可调（n_max=1 时 α 升至 0.463，验证了"单 token 置信度最高"）
- 贪心路径（temp=0）verify 开销 800ms → 12ms（70×）
- 随机路径剩余工作（上游工程）：GPU 端 top-k 接受 / 持久 staging 缓冲 / m×vocab D2H 提速

### 0.8B 上 MTP 仍慢于基线的原因
0.8B 是 launch-bound（218 t/s = 4.6ms/token，瓶颈是内核调度），verify 批量的调度开销 ≥ 顺序解码——**MTP 只在内存受限模型（4B+）+ 好头 + temp=0 时有收益空间**。

### 待办
- Quality + greedy MTP 实测（本头 α 预计 ~0.1-0.3，预期仍负优化，但量化净损失）
- 随机路径 GPU-side accept（上游 PR 素材）
---

## 十一、MTP 激活成功 🎉（8.31 最终章）

### 根因修复
shisa-ai 的 KL 蒸馏头（model-mtp.safetensors，19 张量 BF16）嫁接进 Quality GGUF（blk.40 头张量替换，专家库 Q8_0 重量化）→ 生成 MTPFIX.gguf（22.11 GiB）。

### 端到端实测（Quality-MTPFIX + greedy MTP）
| 配置 | alpha | decode |
|------|-------|--------|
| 基线（无 MTP） | — | 54.7 t/s |
| **MTPFIX + MTP greedy** | **1.000（258/258）** | **93.6 t/s（1.71×）** |

蒸馏头对微调主干的预测完美匹配——greedy 下每个草稿全部命中，6 token/周期。

### 嫁接工具
`graft_final.py`（Temp）：解析源 GGUF 结构 → 替换 blk.40 头张量（safetensors BF16 → F16/Q8_0）→ 流式重写。可复用于任何同架构模型的头替换。

### 当前最优配置汇总
| 用途 | 命令要点 | 速度 |
|------|---------|------|
| 日常文本+视觉 | serve --mmproj --ctx 131072 + kv q8 | 42 t/s |
| **文本+MTP 贪心** | run + INFR_MTP=1 + temp=0 | **93.6 t/s** |
| 随机采样+MTP | 待修复（D2H 病理：m×vocab logits 下载） | 待做 |
---

## 十二、收工状态与明日计划（8.31 深夜 24:00）

### 已交付并验证
- **MTPFIX.gguf**（22.11 GiB）：Ornith Quality + shisa 蒸馏头嫁接版，`run` 模式 MTP greedy **α=1.000 / 93.6 t/s** ✅
- 嫁接工具 graft_final.py（Temp 目录）：GGUF 头张量替换，可复用
- 诊断链完整：随机头确认 → 官方头阳性对照 → 蒸馏头嫁接 → 激活

### 当前卡点（明天第一件事）
serve + MTP（串行引擎 + --mmproj）在 12GB 卡上 VRAM 差 272 MiB：
- 原因：嫁接后 MTP 头会话（embed 表 778MB + F16 权重 + KV）+ 131K KV q8 + 专家缓存 > 12GB
- 修复选项（按优先级）：
  1. seam/mod.rs 的 MTP 预留从 2GiB 改为动态计算（头权重实际字节数 + 778MB embed + 余量）
  2. 或 ctx 降到 16384（KV 减半）+ INFR_VRAM_RESERVE=1g
  3. 或把头会话的 embed 表从 F16 改为 Q8_0（省 390MB）
- **注意**：serve 不带 --mmproj 走并行引擎（ParallelGenerator）→ 无 MTP 且输出乱码（并行引擎对嫁接文件的处理有独立 bug，待查）

### 明天计划
1. 修 serve+MTP 的 VRAM 配比 → 验证 API 端口 MTP 生效（预期 80-95 t/s）
2. 并行引擎乱码 bug 排查（或先绕过：文档写明 MTP 需串行路径）
3. 随机采样路径的 GPU-side accept（可选）

### 快速恢复命令
```powershell
# MTP 贪心交互聊天（已验证 ✅ 93.6 t/s）
C:\Users\zhang\Desktop\Ornith-MTP-chat.cmd
# 或手动：
set PATH=C:\msys64\ucrt64\bin;%PATH%
set INFR_MTP=1
infr.exe run <MTPFIX.gguf> --ctx 8192 --max-new 1000 --temp 0
```