# MoE4All 优化项目 — 阶段性报告

日期：2026-08-30
机器：AMD RX 7700 XT 12GB / 64GB DDR4 (55GB/s) / Windows 11
目标模型：Ornith-1.5-35B-A3B-APEX-MTP-I-Quality (qwen35moe, 22GB, 256 专家, 自带 MTP 头)
目标：本地推理 40+ t/s，冲击 60 t/s

---

## 一、已完成的里程碑

### 1. 基线调优（官方发行版）
- 开启 BIOS ReBAR（Above 4G Decoding + Resizable BAR）——MoE4All 分页器的前置条件
- llama.cpp Vulkan 对比组：10.3 t/s (IQ1_S 125B) / 17.4 t/s (Ornith Mini 32k)
- **MoE4All：Ornith Quality serve 模式 40.4 t/s，Qwen3.8-Flash-Next IQ1_S 18.4 t/s @0ctx / 15.8 t/s @128K**
- 结论：MoE4All 的专家分页 + ReBAR 架构完胜 llama.cpp，基线 40 t/s 已达成

### 2. 源码编译环境（从零）
- Rust 1.98 (GNU toolchain, rsproxy 镜像)
- glslc 2026.3（MSYS2 清华镜像单包提取，免 Vulkan SDK）
- MSYS2 ucrt64 toolchain（dlltool/gcc 16.2）
- 全量编译 `cargo build --release -p infr-cli` 成功（7 分钟），自编译 infr 0.4.0 验证可运行
- 已知问题：GNU 构建比官方 MSVC 构建慢 ~30%（12.7 vs 18.4 t/s @125B），发布级优化需 MSVC

### 3. MTP（投机解码）调研与前两阶段实现
关键发现：
- **Qwen3.8-Flash-Next 的 GGUF 没有 MTP 头**（Unsloth 转换时丢弃），此路不通
- **Ornith 自带 MTP 头**（blk.40.nextn.*，qwen35moe 架构）
- MoE4All 的 MTP 实现完整但被官方"停放"（int8 内核破坏了 greedy 逐 token 一致性）

已实现（自编译分支）：
- **阶段 1**：MTP 头 MoE FFN 支持
  - `MtpHeadWeights` 增加 `MtpFfn::{Dense, Moe}` 枚举（镜像 trunk 的 `FfnW`）
  - `load_mtp_head` 接受 qwen35moe（经 `cfg.moe`），形状全部从 cfg 派生
  - 上传/图发射（`Op::MoeFfn` + 共享专家 + `Op::MoeSharedExpertAdd`）双形状（草稿链 rows=1 / 追赶 rows>1）
- **阶段 2**：验证路径 + 回滚
  - 放宽 runner 验证门禁：qwen35moe 可跑 batched verify（qwen4exp/deepseek4 仍排除）
  - 回滚过滤器验证：qwen35moe 的 DeltaNet 层覆盖正确（含单元测试）
  - 解开 `mtp_enabled()` 停放开关（本地分支）

## 二、当前阻塞点（阶段 3 攻坚目标）

**MTP 驱动的架构缺陷**：`generate_mtp_spec_vulkan_timed_on` 的 `bind` 闭包把主干权重**朴素全量上传显存**（4B dense 时代的设计）。22GB 分页 MoE 模型一加载即 OOM（实测 8.76GB + 256MiB 失败点）。

**解法**：将 MTP 主干 verify 接到现有的分页 seam 会话上（共享 ReBAR 分页器 + 专家缓存），而非另起炉灶。涉及：
- run_verify / run_prime_last 的权重绑定改为分页感知
- 复用 DenseVulkanSession 的 pager/host store
- 之后才是 f32 边际重验（解决停放根因）

## 三、实测数据汇总

| 场景 | 引擎 | 速度 |
|------|------|------|
| Ornith Quality, serve, 基线 | MoE4All | **40.4 t/s** |
| Ornith Quality, run 会话, 基线 | MoE4All | 22.5 t/s |
| Ornith Mini 32k | llama.cpp Vulkan | 17.4 t/s |
| Qwen3.8 IQ1_S @0ctx | MoE4All | 18.4 t/s |
| Qwen3.8 IQ1_S @128K | MoE4All | 15.8 t/s |
| Qwen3.8 IQ1_S | llama.cpp Vulkan | 10.3 t/s |
| Ornith MTP (llama.cpp) | llama.cpp draft-mtp | 11.7 t/s（负优化，接受率 0.23）|

## 四、下一步

1. **阶段 3**：MTP 驱动主干接入分页会话（目标 50–60 t/s）
2. 阶段 4：f32 边际重验，恢复 greedy 一致性（可上游化）
3. 阶段 5：Qwen3-VL 视觉接入（mrope + deepstack，最大工程）
4. 可选：重新转换带 MTP 头的 Qwen3.8 GGUF（75GB，供未来双卡用）
