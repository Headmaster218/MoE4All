# 弹性统一显存验收 — 2026-08-24

## 结果

分页 MoE 专家、LLM 激活临时工作区以及原生 Embedding 权重/运行时现在使用同一个物理 Vulkan arena。
固定的稠密权重与 KV/持久状态保持在其外。专家槽位从低地址增长；可变大小的 LLM、Embedding、Vision
和 Draft 分配从高地址增长。当可变分配无法容纳时，分页器驱逐最冷的连续专家窗口，随后原位恢复每个
已释放的槽位。

运行时预留不再是第二个物理预留。它包含在 arena 内，并在相应运行时分配不存在时可供专家槽位使用。
原生 Embedding 权重仅为活跃请求从其 GGUF 加载，并在 GPU 执行和输出下载后释放。Embedding 激活分配是
瞬态的；缓存的执行计划仅保留小型主机可见输入/回读缓冲区。

加载后的自动上下文尺寸计算保持两个独立预算：KV/循环状态必须容纳于设备尚未提交的空间中，而激活临时
工作区可使用已提交的弹性 arena。这可防止完整的专家缓存令自动上下文尺寸报告为零，同时不会错误允许
持久 KV 占用专家槽位。

Vision 和 Draft 分配类别遵循同一高地址策略，已可供其引擎使用。动态 KV 分配仍有意不在本次范围内。

## 实机 GPU 验收

硬件：AMD Radeon RX 7900 XTX。配置：20 GiB 总显存预算、40 GiB RAM 预算、4096-token 上下文。

模型：

- `Qwen3.6-35B-A3B-APEX-I-Balanced.gguf`
- `nomic-embed-text-v1.5.f16.gguf`

两个端点均就绪后的初始 arena 记账：

| 类别 | 字节数 | MiB |
|---|---:|---:|
| 弹性 arena | 18,790,293,504 | 17,919.82 |
| 专家槽位 | 18,789,572,608 | 17,919.13 |
| Embedding 权重 | 0 | 0 |
| Embedding 运行时 | 0 | 0 |
| 空闲/槽位对齐尾部 | 720,896 | 0.69 |

初始不可用尾部占 arena 的 0.00384%。

一个双行 Embedding 请求暂时准入 273,530,880 字节（260.86 MiB）权重和一个 1,572,864 字节运行时分配。
执行后采样时，瞬态运行时已释放；所有 Embedding 类别在请求结束后立即归零。权重驻留时，专家槽位
对齐/碎片化留下 22,568,960 字节空闲，即 arena 的 0.120%。随后 Chat 请求恢复 351 个出借的槽位。
在其最终的小型 LLM 运行时分配之后，完整 arena 的记账为：

| 类别 | 字节数 |
|---|---:|
| 专家槽位 | 18,789,433,344 |
| LLM 运行时 | 340,224 |
| 空闲尾部 | 519,936 |
| 合计 | 18,790,293,504 |

最终空闲尾部占 arena 的 0.00277%。Embedding 权重和运行时均为零。

## 功能验收

- `/v1/embeddings`：包括按需加载 260.86 MiB 模型在内，2.88 s 返回一个 768 维向量。驱逐后重复相同
  请求会复用已编译计划、重新加载权重，在 0.13 s 返回，并逐位匹配（`max_abs = 0`）。
- 随后的 `/v1/chat/completions`：在 1.10 s 内返回 `PASS`，17 个提示 token 和 2 个完成 token。
- 另一对并发的 Chat + Embedding 也正确完成（`SAFE`、768 维向量），证明共享执行闸门会串行化冲突的
  arena 使用，不会发生死锁或陈旧的专家访问。
- 服务日志中没有错误、panic、设备丢失或内存不足事件。

原始最终服务日志：
`target/perf/unified-vram-20260824-033627.stderr.log`. The final release-build smoke test after the
split-budget context fix is `target/perf/unified-vram-20260824-040332.stderr.log`; it returned a
768-dimensional normalized Embedding and then two correct Qwen Chat responses (`OK.` / `OK`). Its
final free arena tail was 519,936 bytes (0.00277%).

## 构建与测试

- `cargo check -p infr-vulkan -p infr-embedding -p infr-llama -p infr-cli`
- `infr-vulkan` 统一分配器测试：7/7 通过。
- `infr-embedding` 测试：7/7 通过。
- `infr-llama` 内存计划预算测试：通过。
- `infr-llama` 加载后自动上下文拆分预算测试：2/2 通过。
- 原生服务器 `cargo build --release -p infr-cli`：使用真实 Vulkan SDK 通过。
