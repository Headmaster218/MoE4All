# Fork 演化时间线

[首页](../README.md) / 总览 / 时间线

## 2026-08-15：建立可测量的 Windows 基础

- `d9bd5a9`：Windows 原生运行兼容。
- `1038b5d`：读取 Windows 可用主机内存，自动规划 paging。
- `ebf5b79`：可选 Pager profiling，修正显式 submit cap。
- `8c49710`：LRU hit promotion 从扫描改为 O(1)。
- `898ff91`：修复 benchmark KV 格式标记。
- `16fbee2`：synthetic context depth，使 100K～250K 深上下文可以快速重复测试。

这一天的核心贡献不是最终速度，而是把后续所有结论变成可以复现的实验。

## 2026-08-17：Attention、DeltaNet 与 Q8 KV

- hd256 FlashAttention Prefill 落地，并回收消失的 score matrix 预留。
- DeltaNet Decode 增加 strided 路径。
- paged quant 专家 Decode 增加 subgroup kernel 与边界 mask。
- 对 Q8 KV 进行归因：节省 46.9% KV bytes，但 200K 时 Attention 比 F16 慢约 29%。
- 新增 hd256 Q8 Decode 专用路径。

这一阶段把“MoE 慢”拆成 Attention、循环层和 Pager 三类不同问题。

## 2026-08-18：深上下文 kernel 与流水线

- register-O FlashAttention、输出四 lane、深上下文 split。
- Q8 Decode 的 block clustering、workgroup/wave 扩大、packed fp16、PV dequant、chunk 1024。
- recorder 临时资源复用；这是 Decode 气泡收缩的最大单步之一。
- 多 slot Pager upload pipeline、victim scan 批处理和 expert staging batch。

Q8 Decode kernel 序列在 200K 累计约 +25.7%；但这一天也证明“更大 chunk”“更少
combine tile”并不单调变快。

## 2026-08-19：Prefill/Decode 分治与 Host Store

- Prefill 改为 layer-major Host Store、resident layers 和 A/B whole-layer lanes。
- CPU 直接 push 到 mapped ReBAR，去掉完整 GPU-visible Host mirror。
- Down 搬运与 Up/Gate 计算重叠，并跳过不必要的 Down submit。
- 深 Q8 Attention 继续减少 K 重读。
- 试验 size-aware cache 与 role 比例策略。

架构结论：Prefill 与 Decode 不应强行共用同一种缓存粒度。Prefill 顺序遍历整层，Decode
则由 router 决定稀疏 expert access。

## 2026-08-20：全局池、异步 Prefill ring 与总预算

- 将按 `(role,size)` 固定的六池先统一为 Q5/Q6 两个逻辑全局池。
- 每个逻辑池可跨多个 `<3 GiB` 物理 arena，全局分配/淘汰/复用。
- 删除旧六池兼容路径。
- 异步 refill Prefill layer ring，只要槽释放就补下一层。
- expert-cache 参数升级为总 VRAM budget，Prefill/Decode 共享预算并保留 Decode 热缓存。
- 完成 Q8/F16 × 0/32/64/128/250K × 4/6/8/10/12/14 GiB 大矩阵。
- 开始 GUI supervision 和浏览器控制面。

## 2026-08-21～22：Native Embedding 与统一 VRAM

- 受管 Embedding serving、parity harness、engine boundary。
- BERT WordPiece tokenizer 与 Nomic BERT 原生 CPU/Vulkan 执行。
- `/v1/embeddings` 接入原生引擎。
- 修正 persistent state、runtime reserve 与总 VRAM accounting。
- 统一逻辑 range allocator、shared physical shards、expert slot loan。
- LLM 与 Embedding 共用一个 Vulkan device/queue/arena。
- GUI 修复日志编码和实时速度显示。

8 月 22 日第一次验收仍把 Embedding weights 视作 persistent；两天后继续演化为按请求加载、
使用后释放。

## 2026-08-23：三级缓存、Ling 与 DeepSeek V4

- 增加 exclusive RAM/SSD expert tier，随后演化为均匀预加载和 inclusive shadow。
- Ling 3.0 Flash 的 KDA/MLA + grouped MoE 端到端跑通。
- DeepSeek V4 加入 FP8 KV、MXFP4 indexer、压缩缓存与相关图算子。
- Windows SSD piece read 并发、Host promotion batch、Gate/Up/Down 全角色 batch。
- MXFP4 complete-block Decode 与 F32 vec4 GEMV 优化。

这一阶段第一次真正面对“模型专家 payload 大于 VRAM 和 RAM 总和”。

## 2026-08-24：弹性统一池、trace、122B 调度

- 明确 full-RAM 与 bounded RAM/SSD 两条 host backing 路径。
- 统一 arena 完全弹性：Expert 从低地址、LLM/Embedding/Vision/Draft 从高地址生长。
- Embedding 权重按请求加载和淘汰；runtime reserve 不再物理独占。
- ordered Decode access trace 可完整 replay。
- 修复 Qwen recurrent state 在 empty cache 后未重置导致重复输出异常。
- resident/shared expert 与 miss promotion 重叠；shared expert 融入 paged MoE 计算。
- 针对 35B/122B 做 UGD、UG→D、RAM/SSD transfer 微基准，否决复杂 D 粒度分支。
- DeepSeek V4 campaign 收尾。

## 2026-08-25：Host RAM 原地 Vulkan DMA

- 使用 `VK_EXT_external_memory_host` 将普通 host cache 原位导入为 Vulkan transfer source。
- import 不成功的部分继续走 CPU→ReBAR fallback，不影响正确性。
- Windows 驱动实测 import ceiling 约 29 GiB。
- 从 first-come import 改为按三个 expert pool 比例分配有限 import 额度。
- Qwen 122B 在 13.97 GiB GPU expert arena + 45 GiB bounded RAM 下达到 23.2 tok/s。
- 刷新 fork README 并形成阶段性全貌。

---

[首页](../README.md) · [完整提交映射](../reference/commit-map.md) ·
[架构演化](architecture-evolution.md)
