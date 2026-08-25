# 模型索引

[首页](../README.md) / 模型

本目录按“一个模型怎样推动架构演化”组织。通用机制在
[architecture](../architecture/README.md) 中只写一次。

| 模型 | 本 fork 的角色 | 最终阶段状态 |
|---|---|---|
| [Qwen3.6-35B-A3B Balanced](qwen36-35b.md) | 第一条完整性能主线；建立长上下文、Pager、Prefill/Decode 分治 | 生产力主力，深上下文 Q8 Decode/Prefill 均完成系统优化 |
| [Qwen3.5-122B-A10B Quality](qwen35-122b.md) | 验证 20 GiB VRAM + 45～51 GiB RAM 下的大 MoE | 可运行；Host DMA tg256 达 23.2 tok/s |
| [Ling 3.0 Flash](ling3-flash.md) | 第一条 KDA/MLA 混合架构和 RAM/SSD 超大模型证明 | 端到端可用，历史 Decode 约 36 tok/s |
| [DeepSeek V4 Flash](deepseek-v4-flash.md) | 压缩 KV、MXFP4、超大 expert payload 与 trace/simulation 试验场 | 正确可跑；约 4.3 tok/s，campaign 已收尾 |
| [Nomic Embed Text v1.5](nomic-embedding.md) | 第二执行图、原生 BERT 与统一显存的验收对象 | `/v1/embeddings` 原生 CPU/Vulkan 可用 |

## 阅读建议

- 想理解“怎么一步步优化出来”：从 [Qwen 35B](qwen36-35b.md) 开始。
- 想理解“超出 RAM 后为什么突然慢”：并读 [DeepSeek V4](deepseek-v4-flash.md) 和
  [Qwen 122B](qwen35-122b.md)。
- 想理解“两个模型怎样共享显存”：看 [Nomic Embedding](nomic-embedding.md) 与
  [统一 VRAM](../architecture/unified-vram.md)。
- 想理解“新架构移植而不是单纯调参”：看 [Ling](ling3-flash.md) 和
  [DeepSeek kernel](../kernels/deepseek-v4.md)。

---

[首页](../README.md) · [架构索引](../architecture/README.md) · [实验索引](../experiments/README.md)
