# Embedding API 与第二执行图

[首页](../README.md) / [Product](README.md) / Embedding API

## API 目标

提供 OpenAI-compatible `/v1/embeddings`，同时满足：

- 单独运行 Embedding；
- 与 Chat LLM 在同一个 INFR service endpoint 共存；
- API key、admission、日志和 lifecycle 统一；
- native engine 与 llama.cpp oracle 可对比；
- GPU 显存不由两个进程各自抢占。

## 两种实现阶段

### Managed llama.cpp

早期方案启动外部 runner，INFR 负责 proxy 和 supervision。优点是快速获得正确成熟的
Embedding；缺点是第二 Vulkan process/allocator，不可能与 LLM 真正统一 slot management。

### Native Nomic BERT

最终加入：

- GGUF config/tensor parser；
- BERT WordPiece tokenizer；
- CPU reference 与 Vulkan backend；
- pooling/L2 normalize；
- native engine boundary；
- parity harness；
- unified arena integration。

`serve-embedding` 可只启动 Embedding；`serve --embedding-model` 可在一个 endpoint 同时提供
Chat 和 Embedding。

## 请求生命周期

```text
HTTP parse/admission
      ↓
tokenize + batch shaping
      ↓
acquire auxiliary execution gate
      ↓
demand-load weights into high-address elastic range
      ↓
allocate transient activation/input/readback
      ↓
record + submit Vulkan graph
      ↓
download, pool, normalize, JSON response
      ↓
release runtime and GPU weights
```

compiled execution plan 可保留 host-side metadata/shader pipeline；不保留 260.86 MiB GPU
weights。因此第二次 demand reload 仍可仅约 0.13 s，而无需永久牺牲 Expert cache。

## 与 LLM 的互斥范围

不是整个 HTTP server 一次只能接一个请求，而是会改变统一 arena ownership 或提交同一 queue
的关键 GPU 段受 execution gate 协调。网络 parsing、tokenization、已有 plan 选择可在锁外。

这避免：

- Embedding 驱逐 LLM 正在读取的 Expert；
- LLM 在 LUT 尚未 restore 时提交；
- 同线程 read→write lock upgrade deadlock；
- 两个独立 allocator 都以为自己拥有完整 20 GiB budget。

## 数值与功能验收

- 768 dimensions；
- finite values；
- L2 norm = 1；
- 对 llama.cpp cosine `0.999955788`～`0.999974766`；
- long input、batch 8、中英文和 semantic pair 均覆盖；
- Embedding 后连续两个 Chat request 正确；
- 同时发 Chat + Embedding 请求可完成，无 device lost/OOM/deadlock。

## 为什么 Embedding 不需要 Draft KV

Embedding 是 encoder-style forward，不生成 token，没有 autoregressive KV cache。它需要：

- 模型权重；
- token/position/attention 临时 tensor；
- graph activation scratch；
- output readback。

`draft KV` 属于 speculative decoding 的 draft LLM，不属于 Embedding。Vision scratch 则是
图像 encoder/projector 的临时数据。统一 allocator 将三者作为不同 allocation class，只是
共享同一高地址策略。

## Vision 的复用点

未来 Vision 可复用：

- catalog 中 model/mmproj pairing；
- auxiliary execution gate；
- high-address weights/runtime allocation；
- cold Expert window loan/restore；
- request-scoped demand load；
- API/auth/service status。

不能直接复用的部分是 tokenizer、graph ops、image preprocessing 和 projector kernels。

---

[Product](README.md) · [Nomic 模型页](../models/nomic-embedding.md) ·
[统一 VRAM](../architecture/unified-vram.md)
