---
kind: incident
status: diagnosed
date: 2026-09-26
scope: dsh-compaction-qwen38-session-cache
last_verified: 2026-09-26
verified_commit: 6f913991
---

# INC-20260926：DSH 裁剪暴露缺失的递归 checkpoint

## 摘要

长 Agent 会话接近上下文管理阈值时，DSH 会裁剪早期的大型工具结果。该行为合理且符合 harness 的职责：
旧工具输出已完成其任务，不应为了保住推理缓存而永久占用上下文。

2026-09-26 的真实会话中，裁剪后下一次请求却从 0 开始 Prefill 约 109K tokens。直接触发器是
DSH 的 `compaction/prune`，但根因位于引擎：旧状态只有“最后一条可编辑消息”checkpoint，缺少稳定的
Agent 系统提示词/工具前缀 checkpoint。裁剪改变早期历史后，唯一可用的 checkpoint 失效；Qwen3.8
递归层又不能退回任意公共 token 前缀，因此只能完整重算。

## 现场证据

会话 `消息中断后继续生成插件` 的 Turn 5 出现以下序列：

| 阶段 | cache read | 新 Prefill | 输出 |
| --- | ---: | ---: | ---: |
| Step 1 | 126,440 | 756 | 2,996 |
| DSH 裁剪 | 8 个 `compaction/prune`，原工具结果共约 27,071 tokens | - | - |
| Step 2 | 0 | 109,473 | 837 |
| Step 3 | 110,310 | 1,153 | 3,375 |

最早被裁剪的是 Turn 1 / Step 3 的大型工具结果，原缓存不再是新 prompt 的完整前缀。Step 2 从
12:27:06 运行到 12:31:19，用户界面在这段时间主要表现为长时间 `starting`。

裁剪前旧分支的 SSD KV 头与 Step 1 的 token 统计完全对应：

```text
cached=130192
agent_checkpoint=0
edit_checkpoint=127134
```

Step 2 完整 Prefill 后，Step 3 正常命中 110,310 tokens。最终分支写回 SSD 时已经具备两个 checkpoint：

```text
cached=114838
agent_checkpoint=8790
edit_checkpoint=110460
```

## 触发条件和根因

1. DSH 为控制上下文大小，裁剪位于旧历史中的大型工具结果。
2. 新 prompt 不再扩展旧 live state，也不再以旧 `edit checkpoint` 为前缀。
3. Qwen3.8 的 GDA/KDA/PLE 包含追加式递归状态，不能仅凭 attention KV 截断到任意公共前缀。
4. 该槽位的持久化状态没有 `agent checkpoint`，因此不存在可恢复的递归边界。
5. runner 将递归状态归零，并从 prompt 起点重新 Prefill。

checkpoint 目前是机会式建立的：只有实际 Prefill 穿过某个边界时才会捕获该边界。如果旧状态缺少
Agent checkpoint，而当前复用起点已经位于该边界之后，引擎不会主动回填它。该状态可以持续多个正常
增量请求，直到 DSH 裁剪、编辑旧消息或分支切换使末轮 checkpoint 失效。

## MTP 相关放大路径

本次完整重算在主模型层已经不可避免，MTP 不是 DSH 裁剪的原因。但当前 MTP 还有同类放大路径：

- MTP head 的 KV、`last_h` 和 turn checkpoints 绑定运行时 slot，没有随主 KV 一起写入 SSD。
- 冷恢复、跨槽迁移或进程重启后，主 KV 可能命中而 MTP 状态缺失。
- `prime_mtp_work` 当前会在 MTP 状态不可复用时直接 reset 主 KV。
- checkpoint 边界在该 reset 之前计算；reset 把实际起点改成 0 后，早于旧起点的 Agent 边界不会重新
  进入捕获列表，完整 Prefill 结束后仍可能只留下 edit checkpoint。

因此，DSH 的合理裁剪不应被禁用；需要修复的是主 KV、递归 checkpoint 与 MTP sidecar 的一致性。

## 被破坏的不变量

1. 每个可持久化的 Qwen3.8 Agent 会话都应保留稳定 Agent 前缀的递归 checkpoint。
2. MTP sidecar 缺失不能使一个已经验证可恢复的主模型 checkpoint 失效。
3. 任何 reset 若改变实际 Prefill 起点，都必须重新计算 checkpoint 捕获边界。
4. 上层 harness 可以自由裁剪旧工具结果；引擎应退化为“从最近稳定 checkpoint 重算”，而不是默认
   从 0 重算。

## 修复方向

1. 主 KV 命中但 MTP sidecar 缺失时，本次请求退回非 MTP，保留主模型命中状态。
2. 将 MTP head 状态作为主 KV 的 sidecar 一起保存、恢复和跨槽复制。
3. 在 reset 后重新 prepare prompt，或者显式按新起点重新建立 checkpoint 捕获计划。
4. 对缺少 Agent checkpoint 的旧状态增加自愈策略，避免它在增量复用期间永久缺失。

当前会话经过这次完整 Prefill 后已经补齐 `agent_checkpoint=8790`，但进程重启、跨槽恢复和缺少 MTP
sidecar 的路径仍未修复，因此本事故保持 `diagnosed`。

## 回归保护

后续修复至少覆盖以下场景：

- 130K 左右 Agent prompt 裁剪一个早期工具结果，只从 Agent checkpoint 继续 Prefill。
- SSD 冷恢复后执行同样裁剪，cache read 不得降为 0。
- 两槽之间迁移相同会话，主 KV 命中且 MTP sidecar 缺失时自动退回非 MTP。
- 服务重启后恢复带 Agent/Edit checkpoint 的 SSD KV，不得因 MTP prime 清空主 KV。
- 完整重建确实不可避免时，写回快照必须同时包含 Agent 和 Edit checkpoint。

## 相关代码

- `crates/infr-llama/src/seam/runner.rs`：递归 checkpoint 选择和 Prefill 起点。
- `crates/infr-llama/src/seam/weights.rs`：递归 checkpoint 捕获、恢复和失效。
- `crates/infr-llama/src/parallel.rs`：MTP prime、MTP head checkpoint 和主 KV reset。
- `crates/infr-llama/src/session_cache.rs`：主 KV 与两个 checkpoint 的 SSD 格式。
