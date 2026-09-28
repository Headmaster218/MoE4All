---
kind: guide
status: current
scope: serving-api
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 使用本地 API

先在 Windows 向导中选择“OpenAI 兼容 API”，或用 `infr serve` 启动模型。下面以本机默认地址为例；若启动时指定了 `--addr`，请替换地址。客户端的 Base URL 为 `http://127.0.0.1:8080/v1`。

## 查找模型与鉴权

`GET /health` 无需鉴权。设置了非空 `serve.api_key` 时，`/v1/models` 和生成、嵌入接口都要求 `Authorization: Bearer <key>`；空字符串等于未启用鉴权。

```powershell
$base = 'http://127.0.0.1:8080'
$headers = @{} # 已配置 API key 时改为 @{ Authorization = 'Bearer <key>' }
Invoke-RestMethod -Uri "$base/health"
(Invoke-RestMethod -Uri "$base/v1/models" -Headers $headers).data
```

从模型列表选取聊天模型 ID，填入下面的 `$chatModel`。如需调用嵌入接口，也应选取单独列出的嵌入模型 ID。

## 文本对话

```powershell
$chatModel = '<聊天模型 ID>'
$request = @{
    model = $chatModel
    messages = @(@{ role = 'user'; content = '你好，请用一句话介绍自己。' })
    max_tokens = 128
    stream = $false
} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Post -Uri "$base/v1/chat/completions" -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($request))
```

多轮对话时，客户端应在下一次请求的 `messages` 中重放所需历史；冷 KV cache 只是计算加速，不存储可替代消息历史的对话记录。工具调用后的下一轮也应带回 assistant 的 `tool_calls` 和相应工具消息。思考内容的保留方式见[思考模式控制](thinking-controls.md)。

把 `stream` 设为 `true` 可接收 SSE；客户端应处理增量、错误事件及最终 `[DONE]`。服务端的 keep-alive 仅维持连接，不表示已经生成 token。

## Responses API

0.9.0 提供无状态 `POST /v1/responses`，接受字符串或消息数组形式的 `input`，支持 JSON 响应和 typed SSE：

```powershell
$request = @{ model = $chatModel; input = '用一句话解释 Prefill。'; max_output_tokens = 128; stream = $false } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$base/v1/responses" -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($request))
```

本接口不保存服务端对话：`previous_response_id`、`conversation` 和 `store=true` 会被拒绝。需要连续对话时，由客户端传入完整的相关历史。当前也不支持远程图片 URL、`file_id`、`input_file`、自动截断和 hosted tools；不要把它当作完整的云端 Responses API。

## 图片与嵌入

启用 Vision 并加载兼容的 mmproj 后，Chat Completions 的 `content` 可交错包含文本和 `image_url`。图片须为 data URI 或 base64；远程 HTTP URL 不受支持。以下示例读取本地 PNG：

```powershell
$image = [Convert]::ToBase64String([IO.File]::ReadAllBytes('D:\Images\example.png'))
$request = @{
    model = $chatModel
    messages = @(@{ role = 'user'; content = @(
        @{ type = 'text'; text = '这张图里有什么？' },
        @{ type = 'image_url'; image_url = @{ url = "data:image/png;base64,$image" } }
    ) })
    max_tokens = 128
} | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "$base/v1/chat/completions" -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($request))
```

启用 Embedding 并加载嵌入模型后，可向 `/v1/embeddings` 发送单个字符串或字符串数组；返回 float vector 与 token usage。`encoding_format=base64` 不受支持。

```powershell
$embeddingModel = '<嵌入模型 ID>'
$request = @{ model = $embeddingModel; input = @('第一段文本', '第二段文本') } | ConvertTo-Json -Depth 4
$result = Invoke-RestMethod -Method Post -Uri "$base/v1/embeddings" -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($request))
$result.data | Select-Object index, @{Name='dimensions';Expression={$_.embedding.Count}}
```

具体可运行组合取决于[模型能力](../../reference/model-capabilities.md)和启动时的内存预算。接口行为以 `crates/infr-server/src/lib.rs` 和 `crates/infr-server/src/responses.rs` 在上述 `verified_commit` 的实现为准。
