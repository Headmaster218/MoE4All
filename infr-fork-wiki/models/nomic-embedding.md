# Nomic Embed Text v1.5

[首页](../README.md) / [模型](README.md) / Nomic Embedding

## 为什么先做 Embedding

Embedding 模型本身只有约 260.86 MiB 权重，但它是验证多执行图、统一显存、生命周期和
服务 API 的理想最小对象。Vision 和 speculative draft 以后会更大、更复杂；如果连
Embedding 都只能单独启动一个进程，就无法验证真正的资源统一管理。

## 三阶段演化

### 1. Managed worker

最初由 INFR 管理外部 llama.cpp Embedding worker：INFR 负责 API、鉴权、admission、进程
生命周期和资源估算，计算仍交给成熟实现。它解决“能服务”，但 LLM 与 Embedding 不能
共享同一 Vulkan memory pool。

### 2. Native engine

依次加入：

- llama.cpp parity harness；
- 独立 Embedding engine boundary；
- BERT WordPiece tokenizer；
- Nomic BERT CPU reference；
- Nomic BERT Vulkan graph；
- 原生 `/v1/embeddings` serving。

权重 offset、F16 linear、graph-local attention KV 和 tokenizer 语义都按真实 GGUF 解析，
不是为一个文件写死 shape。

### 3. Unified elastic VRAM

第一版把 260.86 MiB weights 作为 unified arena 中的 persistent allocation，runtime 通过
loan 冷 Expert slot 获得空间。随后发现“偶尔调用的 Embedding 不该永久挤占 Expert”，
最终改为：

1. 请求到来时从 GGUF demand-load weights；
2. 从高地址分配 weights/runtime；
3. 必要时淘汰连续 cold Expert window；
4. GPU 执行并下载输出；
5. runtime 和 weights 都释放；
6. 后续 LLM 请求按 generation 原位恢复 Expert slots。

## 数值验收

Native Vulkan 与 llama.cpp oracle：

| Case | Cosine similarity | Max abs error |
|---|---:|---:|
| Chinese short | 0.999974766 | 0.000803143 |
| English short | 0.999965175 | 0.000919986 |
| Semantic pair | 0.999963835 | 0.001191165 |
| Batch 8 | 0.999966875 | 0.001312540 |
| Long input | 0.999955788 | 0.001098613 |

API 返回 768 维有限值向量，L2 norm 为 1。

## 统一显存验收

20 GiB total VRAM budget、Qwen 35B + Nomic：

| 初始项目 | Bytes |
|---|---:|
| Elastic arena | 18,790,293,504 |
| Expert slots | 18,789,572,608 |
| Embedding weights/runtime | 0 / 0 |
| Free/rounding tail | 720,896（0.00384%） |

一次 2-row 请求临时分配 273,530,880 bytes weights 和 1,572,864 bytes runtime。请求后
Embedding 两类占用均归零。随后 Chat 恢复 351 个借出 slots；最终 free tail 519,936
bytes（0.00277%）。

第一次请求包含 demand load，用时 2.88 s；weights 淘汰后重复相同请求复用 compiled plan，
重新加载并在 0.13 s 完成，输出 bit-for-bit 相同。

## 并发规则

同一 Vulkan queue/arena 上不能让 LLM 在引用 Expert slot 时被 Embedding write lease
失效：

- LLM in-flight execution 持 read lease；
- auxiliary graph recording 不持会导致升级死锁的 read lease；
- 真正需要挪用 Expert slot 的 auxiliary allocation 获取 write lease；
- write lease 等待在途 LLM 完成后才改变地址有效性。

这个规则修复了第一版“同线程从 read upgrade 到 write”的首请求死锁。Chat + Embedding
同时请求实测完成，无 deadlock/stale Expert。

## 对 Vision/Draft 的预留

统一 allocator 已定义 `Vision`、`Draft` 高地址 allocation class，但只有生命周期和分配
策略准备好，执行引擎尚未接入。未来无需让它们迁就 Expert slot 大小；可申请连续变长
range，并通过 cold window eviction 获得空间。

---

[模型索引](README.md) · [统一 VRAM](../architecture/unified-vram.md) ·
[Embedding API](../product/embedding-api.md)
