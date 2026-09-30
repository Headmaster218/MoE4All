---
kind: architecture
status: current
scope: serving
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 服务与冷 KV 会话

## HTTP 边界

`infr-server` 提供：

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`：无状态 Responses API，支持 JSON 与 typed SSE。
- `POST /v1/embeddings`

HTTP 层负责 DTO、鉴权、参数校验、SSE、可选 request deadline 和错误映射。`serve.request_timeout_secs` 默认是 0（不设 deadline）；
配置非零值才启用 wall-clock deadline。SSE response 在 slot admission 前建立，排队和长运行期间发送 keep-alive；
客户端断开后，排队中的请求会取消等待，不会之后再占用 slot。模型执行通过 generator contract 进入 runtime；
server 不直接操作 Vulkan buffer。

Responses API 将无状态 input 映射到内部 Chat Completions 路径，并以 Responses 事件类型输出流；请求输入不会在服务端形成
持久对话事实源。流式 keep-alive 只维持连接，不代表模型在该时间窗内已产生 token。

## 请求生命周期

1. 解析模型、消息、reasoning、tools 和 image parts。
2. 进行 admission，等待可用 slot。
3. 渲染稳定 prompt，尝试复用 prefix/session state。
4. Prefill 和 Decode/MTP。
5. 将 reasoning/content/tool delta 编码为 SSE 或一次性响应。
6. 客户端断开时设置 cancel latch，runtime 在安全边界停止并释放 slot。

llama.cpp compatibility embedding runner 仅对进程 readiness probe 保留启动超时，不再对整个 Embedding 请求施加固定 600 秒 deadline；
不要将此行为与 `serve.request_timeout_secs` 混为一谈。

## 冷 KV 会话缓存

该功能只在显式配置目录时启用，并面向动态分段 Q8 KV：

- scheduler 选择空闲 resident slot；
- 后台将 checksummed state 写入 SSD；
- 释放其物理 KV segments；
- 后续 prefix match 可恢复到任意空闲 slot；
- 启动时扫描并验证 model fingerprint、geometry、format、容量和 TTL。

冷缓存扩大“可保留会话状态”的数量，不增加同时计算 slot 数。

此外，运行中的 session 可维护有限的 turn checkpoint，用于复用 Agent 后续轮次的稳定前缀，以及编辑上一轮时可复用的公共前缀。
并发 MTP 路径还会保存并复用对应的 head/recurrent checkpoint 与 reasoning 状态，避免恢复主 KV 后因辅助状态缺失而错误续跑；
无法匹配时可回退 ordinary Decode。它们属于当前进程内的计算状态，不是持久化聊天记录；重启、分支不匹配或校验失败时仍可退回
重新 Prefill。冷 KV SSD cache 则是另一层可选机制，用于 idle slot 释放设备 KV 后的恢复。

## 与上层会话的边界

调用方保存消息历史和工具结果，KV cache 只是可丢弃的计算加速状态。引擎重启或 cache 校验失败时，调用方用消息历史重新 Prefill；不能把 KV 文件当作唯一聊天记录。Agent/harness 的产品级状态归属见[后续提案](../../roadmap/agent-distribution.md)。
