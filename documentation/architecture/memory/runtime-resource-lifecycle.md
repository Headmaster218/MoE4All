---
kind: architecture
status: current
scope: runtime-memory
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 运行时资源生命周期

## 启动顺序

```text
hardware probe
  -> resolve RAM/VRAM/context/ubatch policy
  -> allocate fixed model and optional MTP resources
  -> trim reclaimable upload pages
  -> measure real device room
  -> create unified VRAM shards
  -> place Expert filler and host tier
  -> allocate/fork session state
  -> warm up pipelines
```

预加载估算用于选择候选 Ubatch 和预算，但最终 unified arena 必须基于固定分配已经落地后的真实余量。
在 MoE arena 放置阶段，如果所选 ubatch 无法满足 scratch/reserve 条件，启动流程可以按较小 ubatch 重试；不能把预估阶段
选出的 ubatch 当作最终值。MTP 固定权重和 per-slot runtime 的启动期分配适用于 `ParallelSeam` 并发路径，传统单请求 MTP
路径则按首次请求初始化，详见 [Qwen3.8 MTP](../runtime/qwen38-mtp.md)。

## 统一显存所有者

统一 arena 是由多个 Vulkan shard 组成的逻辑地址空间。主要 owner：

| Owner | 生命周期 | 优先级 |
|---|---|---|
| KV / recurrent state | session 持久，动态增长 | 高 |
| Decode runtime | session 或 phase | 高 |
| Prefill ring/scratch | Prefill phase | 高 |
| Vision weights/runtime | image batch | 高 |
| Embedding weights/runtime | request 或 idle timeout | 高 |
| Expert cache | 填充剩余空间，可回收 | 最低 |

Expert slot 是 clean cache。被更高优先级 owner 覆盖时无需回写，之后可从 RAM 或 GGUF/SSD 恢复。

## 地址与安全点

Claim 不是简单地返回一段空闲字节。它可能需要淘汰或移动 Expert filler，并更新物理目录。发布新 lease 前必须：

1. 到达相关 GPU safe point。
2. 计算 victim 和目标 extent。
3. 保护 command stream 仍可引用的 slot。
4. 完成必要的 D2D move。
5. 原子发布新目录和 owner lease。

失败时不能留下半发布地址。

## 冻结 LUT

Decode/VERIFY 录制时会把每个 routed batch 的 LUT window 冻结到 append-only tape。之后 host-side pager 可继续规划，
但 tape 中的物理 slot 引用直到 command stream drain 都必须有效。只保护本批实际 routed 的 resident slot，避免把整个
pool 永久 pin 住。

该不变量曾被 lazy scratch/unified claim 破坏并表现为重复 token，见
[INC-20260925](../../evidence/incidents/INC-20260925-frozen-expert-lut-slot.md)。

## 主机内存层

- RAM 足够时建立 full layer-contiguous host store。
- RAM 受限时使用 bounded inclusive RAM/SSD cache。
- Host DMA 可将部分 RAM arena 导入 Vulkan；超出驱动 import limit 的范围回退到 staged/CPU push 路径。
- 传输方案在 session 建立时按硬件能力冻结，普通热路径不反复猜测 backend。

这里的判断是从总进程 RAM 预算扣除当时其他常驻占用后，以可供专家的预算 `R` 对比 routed expert payload `E`：`R >= E` 时完整 Host Store 覆盖专家、运行期不需 SSD demand read；`R < E` 时按池预加载可用容量，其余由只读 GGUF/SSD 按需填充。bounded tier 保留 inclusive shadow；clean expert 淘汰不做 GPU→RAM 回写。Host DMA 导入需要满足设备的 `minImportedHostPointerAlignment`，导入缓冲区只是原 RAM 的 Vulkan view，再以 `vkCmdCopyBuffer` 搬到显存；不能再为它预留一份完整的 Host Store 副本。

## 自动策略

自动 RAM 预算在启动时按当时系统状态冻结。显式预算类似手动配置，不应在运行中随系统 available 值漂移。
保守档使用启动时可用 RAM 减 3 GiB，激进档使用总物理 RAM 减 14 GiB，均写入进程总 RAM 预算。
保守 VRAM 在分配器 256 MiB guard 外再留 768 MiB；激进档将总进程显存限制为设备总量减 2 GiB，同时受实时可用量约束。两档都先真实分配固定资源，再测量剩余空间建立 arena；Windows 大型 ReBAR 模型还可能应用额外启动保留与失败重试。
离散 GPU 的默认 Prefill ubatch 分别从 2048/4096 行开始，放置不足时按档位下调；iGPU 使用独立默认值。

更完整的历史设计见 [分层统一内存](../../evidence/changes/memory/unified-memory.md) 和 [Tiered weight paging](../../evidence/changes/memory/tiered-weight-paging.md)。
