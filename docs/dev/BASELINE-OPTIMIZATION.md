# MoE4All 基线优化完整记录：RX 7700 XT 12GB 跑 35B MoE 从 10 t/s 到 42 t/s

## 硬件与起点

| 项目 | 配置 | 备注 |
|------|------|------|
| GPU | AMD Radeon RX 7700 XT 12GB | PCIe 4.0 x16，RDNA3 |
| CPU | i5-12600KF（10 线程，P-core 3.7→4.9GHz）| DDR4 平台 |
| 内存 | 64GB DDR4 双通道 3200MHz | 实测 STREAM 有效带宽 ~55 GB/s |
| 主板 | Gigabyte Z690M AORUS ELITE DDR4 | BIOS 默认 ReBAR 关闭 |
| 系统 | Windows 11 / 高性能电源计划 | |
| 引擎 | MoE4All v0.3.0（官方发行版）/ 自编译 dev 分支 | |

起点：llama.cpp Vulkan 跑 Ornith-1.5-35B-A3B（qwen35moe, 22GB, 3B active, 256 专家选 8），decode 17 t/s。
目标：40+ t/s。

---

## 第一步：BIOS 开 ReBAR

### 发现过程

MoE4All 首次启动即报错：
```
rebar allocate_memory(2146649600): A device memory allocation has failed
```
分配 2GB 失败——显存总共 12GB，为什么 2GB 都分不出来？

根因：不开 ReBAR 时 CPU 只能看到显存的 **256MB 窗口**（PCIe 遗留地址空间）。MoE4All 的分页器需要 CPU 直接映射大块 VRAM 作为专家缓存池，256MB 连一个 2GB 缓存块的零头都不够。

### 操作

```
BIOS → Settings → PCI Subsystem Settings →
  Above 4G Decoding: Enabled
  Re-Size BAR Support: Enabled
```

### 效果

ReBAR 开启后，MoE4All 成功分配 4.89GB ReBAR 映射池。
这步不是"提速"，是 **MoE4All 的硬性前置条件**——不开就什么都跑不了。

### 踩坑记录

- BIOS 里 Above 4G Decoding 和 Re-Size BAR 是两个独立开关，必须**先开 4G 再开 ReBAR**，顺序反了不生效
- 开启后 Windows 报一次设备重枚举（屏幕黑几秒），正常
- AMD Software → 性能 → 确认 "Smart Access Memory: 已启用"

---

## 第二步：从 llama.cpp 切换到 MoE4All

### llama.cpp 基线（切换前）

```
llama-server -m Ornith-Quality.gguf -ngl 99 -c 4096 --flash-attn
```

| 指标 | 数据 |
|------|------|
| decode | **17 t/s** |
| prefill | ~30 t/s |
| 显存占用 | ~10.5GB（llama.cpp 的 -ot 分配）|

llama.cpp 用 `-ot` 静态分配——放不进显存的层固定放 CPU，CPU 上用 AVX2 算。
对于 22GB 模型 + 12GB 显存，约 40% 的权重在 CPU 上。

### 切换到 MoE4All

```
ReBAR 开启 → MoE4All 首次成功运行
```

| 指标 | 数据 |
|------|------|
| decode | **18.4 t/s**（Qwen3.8 IQ1_S）/ **40.4 t/s**（Ornith Quality serve）|
| prefill | 446 t/s @12.5K prompt |

**MoE4All 比 llama.cpp 快 2-2.4 倍**，原因是架构差异：

| | llama.cpp | MoE4All |
|--|-----------|---------|
| 专家分配 | `-ot` 静态：装不进的固定放 CPU | 动态分页：LRU 热专家驻留显存，冷的按需 DMA |
| PCIe 利用 | memcpy / 矩阵乘法混合 | ReBAR 映射池 + GPU 地址 LUT 直接访问 |
| 专家缓存 | 无（CPU 权重每次都算）| LRU 缓存 + 全量 RAM 主存储 + SSD 可选 |
| 长上下文 | KV cache 增长挤压权重 | QSA 稀疏注意力 / GDN 状态空间 |

---

## 第三步：选对引擎模式

### 发现

同一个 MoE4All 二进制，不同运行模式速度差异巨大：

| 模式 | 命令 | 引擎 | 速度 | 区别 |
|------|------|------|------|------|
| serve 并行引擎 | `infr serve <model>` | ParallelGenerator → ParallelSeam | **40-42 t/s** | 多 slot 轮转，批量调度 |
| run 交互终端 | `infr run <model>` | DenseSeamChat → SeamModel | 20-54 t/s | 单会话直通，零调度开销 |

**serve 并行引擎比 run 快 80%+**，原因是并行引擎的批量调度把多个 token 的专家访问合并了——每次 DMA 搬运的专家块被多个 token 共享，缺失次数摊薄。

### 操作

日常使用一律走 serve：
```bash
infr.exe serve <model.gguf> --ctx 131072 --addr 127.0.0.1:8080
```

浏览器或前端直接连 http://127.0.0.1:8080/v1（OpenAI 兼容）。

---

## 第四步：KV cache 量化

### 原理

131K 上下文时 KV cache 占 ~1.5GB 显存（F16）。量化为 q8_0 后减半到 ~0.75GB。
省出的显存自动进入专家分页缓存池 → LRU 缓存容量增大 → 命中率提升 → 减少 PCIe 缺失。

### 操作

```bash
infr.exe serve <model> --set kv.type_k="q8_0" --set kv.type_v="q8_0"
```

### A/B 实测

| 配置 | decode |
|------|--------|
| KV F16（默认）| 38.2 t/s |
| KV q8_0 | **41.0-42.0 t/s** |
| 提升 | **+10%** |

精度损失：KV cache 的量化误差对生成质量影响可忽略（视觉和文本输出均正常）。

---

## 第五步：系统级调优

### 电源计划

```powershell
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  # 高性能
```

实测影响不大（±3-5%），但防止 GPU 空闲时降频导致首个 token 延迟增大。

### GPU 驱动

- AMD Adrenalin 版本影响 Vulkan 性能（建议最新）
- 实测发现 AMD Vulkan 驱动在进程崩溃后虚报显存占用（GPU 计数器显示 1GB，VK_EXT_memory_budget 报 11.5GB）——重启显卡驱动可清

---

## 第六步：排查常见的"假慢"

### 前端注入巨大 system prompt

实测发现 opencode 等 agent 前端会自动注入 12,536 token 的系统提示（sandbox 策略、工具定义、上下文等）。每个请求都要 prefill 这 12.5K token，耗时 ~28 秒。

| 前端注入的 system prompt | 实际 prompt | prefill 耗时 | 用户感受 |
|------------------------|------------|-------------|---------|
| 无/极短 | ~20 tok | <1s | 即答 |
| 普通聊天前端 | ~100-300 tok | 1-2s | 正常 |
| agent 前端（opencode 等）| **12,536 tok** | **28s** | 很慢 |

**解决**：精简前端系统提示，或用轻量聊天前端（Cherry Studio / Chatbox）。

### MoE4All 自身的 prefill 性能

| prompt 长度 | prefill 速度 | 说明 |
|------------|-------------|------|
| 17 tok | 13-15 t/s | 固定开销（会话初始化 ~1-2s）占主导 |
| 236 tok | 67 t/s | 开销开始摊薄 |
| 619 tok | **147 t/s** | 接近稳态 |
| 12,536 tok | **446 t/s** | 批量预填的最佳区间 |

**结论**：prefill 速度随 prompt 长度增加而提升（固定开销摊薄），短 prompt 的 "13 t/s" 不是真实吞吐。

---

## 第七步：MSVC 编译（+0%，此模型无差异）

从 GNU 工具链切换到 MSVC（官方发行版用 MSVC）：
- 安装 VS Build Tools + rustup stable-x86_64-pc-windows-msvc
- RUSTFLAGS = "-C target-cpu=native -C target-feature=+crt-static"

实测：Ornith Quality 上无显著差异（41 vs 42 t/s）。
但 Qwen3.8 IQ1_S 上 GNU 慢 30%（12.7 vs 18.4），说明差异与模型相关。

建议：生产用 MSVC 编译，开发用 GNU（编译快 3 分钟 vs 8 分钟）。

---

## 第八步：排查踩坑记录

| # | 问题 | 原因 | 解决 |
|---|------|------|------|
| 1 | ReBAR 分配失败 | BIOS 未开 | BIOS 开启 |
| 2 | VK_EXT_memory_budget 虚报 11.5GB | AMD 驱动在进程崩溃后不回收 | 禁用/启用 GPU 设备 |
| 3 | prefill lane 报错 "no complete Prefill streaming lane" | 128K ctx 的 KV 挤占专家缓存 → 预填通道装不下 | 减小 ctx 或 kv q8 |
| 4 | Windows GBK 控制台 vs UTF-8 | cmd echo 中文进 stdin 报错 | 用英文或文件重定向 |
| 5 | PowerShell 把 stderr 当异常 | cargo/rust 日志被截断 | `*>` 重定向到文件再读 |
| 6 | llama.cpp 显存被尾随进程虚占 | AMD 驱动 VK_EXT_memory_budget 延迟回收 | 重启显卡驱动 |

---

## 最终 VRAM 分配明细

```
12 GB GPU 显存（ReBAR 开启后 CPU 可见 11.73 GiB）

主干固定权重       ~2.11 GB   (attention Q6_K, 嵌入, LM head, 共享专家 Q8_0)
KV cache (131K q8) ~0.75 GB   (仅 10/40 层全注意力，其余 DeltaNet 状态空间)
运行时弹性池       ~1.15 GB   (激活值、norm 暂存)
load_driver        ~2.15 GB   (上传暂存区、驱动保留)
post_load          ~0.27 GB
packing_margin     ~0.27 GB
──────────────────────────────
专家分页缓存池      ~5.67 GB   (ReBAR 映射 LRU，三个池)
  shared/0.6MB     4205 slots  (gate/up 小层)
  shared/0.7MB     2103 slots  (gate/up 大层)
  shared/0.9MB     2103 slots  (down_proj)
──────────────────────────────
主机 RAM 存储       20.70 GB   (全部专家权重的层连续主副本，24 块)
  11 个 DMA arena   19.28 GiB  (vkCmdCopyBuffer 源)
```

---

## 专家缓存命中率与速度天花板

实测（500 token 生成，paging.stats=true）：

| ReBAR 池 | 命中 | 缺失 | 命中率 | 驱逐 |
|----------|------|------|--------|------|
| shared/0.6MB | 309,353 | 46,327 | **89.0%** | 42,136 |
| shared/0.7MB | 150,205 | 27,635 | **86.0%** | 25,537 |
| shared/0.9MB | 126,356 | 51,484 | **71.1%** | 49,398 |
| **合计** | 585,914 | 125,446 | **82.4%** | 117,071 |

### 每笔缺失的代价

```
每 token 缺失次数：~224 次（125446 / ~560 token）
每次缺失：PCIe DMA ~0.7MB + LUT 更新 + 屏障 ≈ ~80µs
缺失总耗时/token：224 × 80µs = 17.9ms
命中耗时/token：791 × 10µs = 7.9ms
其他计算/token：~2.4ms
──────────────────────────────
每 token 总耗时：23.8ms → 42 t/s
```

### 为什么 42 t/s 是天花板

**PCIe 带宽只用了 21%**（6.6 GB/s / 32 GB/s）。瓶颈不是带宽而是每次缺失的固定延迟——小 DMA（0.7MB）无法打满 PCIe，每次都有固定的发起 + 同步开销。这个开销是分页器架构层的，不是配置能解决的。

突破方法只有**减少缺失次数**：

| 显存 | 可缓存比例 | 预估 miss rate | 预估速度 |
|------|-----------|---------------|---------|
| 12GB（当前）| 27% | 17.6% | 42 t/s |
| MI50 32G 副卡 | ~80% | ~5% | **70-90 t/s** |
| 全部驻留 | 100% | 0% | **100+ t/s** |

---

## 全模型对比

| 模型 | 大小 | 架构 | active/total | MoE4All | llama.cpp | 加速 |
|------|------|------|-------------|---------|-----------|------|
| Ornith Quality Q6_K | 22GB | qwen35moe | 3B/35B | **42 t/s** | 17 t/s | 2.4× |
| Qwen3.8 IQ1_S | 67.6GiB | qwen4exp | 6B/125B | **18.4 t/s** | 10.3 t/s | 1.8× |
| Qwen3.8 IQ1_S @128K | 同上 | 同上 | 同上 | **15.8 t/s** | — | — |
| Qwen3.8 IQ1_S + MTP | 同上 | 同上 | 6B active | 69.5 t/s | — | — |

---

## 未走通的路

| 尝试 | 结果 | 原因 |
|------|------|------|
| MTP 随机头（Ornith 原生）| α≈0，负优化 100× | 官方发布头是随机初始化，从未训练 |
| MTP 贪心（α=1.0）| 机制通了但净速度为负 | 每 verify 重预填 ~250 行 × 1.5s = 巨大开销 |
| Qwen3.8-Flash-Next MTP | GGUF 无 MTP 头张量 | Unsloth 转换时丢弃了 nextn 层 |
| 双 3080 20G / 4090D 48G | 未测试 | 预估可到 60-100 t/s（消除 PCIe 瓶颈）|

---

## 最终推荐配置

```bash
# 日常 API 服务（42 t/s @131K，支持视觉、并发、流式）
infr.exe serve "Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf" ^
  --ctx 131072 ^
  --addr 127.0.0.1:8080 ^
  --set kv.type_k="q8_0" --set kv.type_v="q8_0"

# 如果接 agent 前端（大 system prompt）建议限制上下文防 OOM：
infr.exe serve ... --ctx 32768 --set kv.type_k="q8_0" --set kv.type_v="q8_0"
```

前置条件：
- BIOS：Above 4G Decoding + Re-Size BAR 开启
- MoE4All v0.3.0+ 官方发行版（不需要自编译）
