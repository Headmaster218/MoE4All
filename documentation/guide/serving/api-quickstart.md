---
kind: guide
status: current
scope: serving-api
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 使用本地 API

先用启动向导选择“OpenAI 兼容 API”——Windows 是 `Start-INFR-Wizard.cmd`，Linux 是 [`Start-INFR-Wizard-Linux.sh`](../../../Start-INFR-Wizard-Linux.sh)——或直接用 `infr serve` 启动模型。下面以本机默认地址为例；若启动时指定了 `--addr`，请替换地址。客户端的 Base URL 为 `http://127.0.0.1:8080/v1`。

## 查找模型与鉴权

`GET /health` 无需鉴权。设置了非空 `serve.api_key` 时，`/v1/models`、生成、嵌入接口和 Responses 的状态端点（`GET`/`DELETE`/`POST .../cancel`）都要求 `Authorization: Bearer <key>`；空字符串等于未启用鉴权。

**本机使用 `127.0.0.1`；局域网访问可用 `0.0.0.0`，但应启用 API key。** 不带鉴权地把服务绑定到非回环地址（例如 `--addr 0.0.0.0:8080`），等于把本机的模型能力开放给同网段的所有人。

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

`POST /v1/responses` 接受字符串或消息数组形式的 `input`，支持 JSON 响应和 typed SSE：

```powershell
$request = @{ model = $chatModel; input = '用一句话解释 Prefill。'; max_output_tokens = 128; stream = $false } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$base/v1/responses" -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($request))
```

默认配置下这个接口不保存服务端对话：`previous_response_id`、`conversation` 和 `store=true` 会被拒绝，并指名你实际发送的那个字段。需要连续对话时，由客户端传入完整的相关历史。

启动时加上 `--set serve.response_store_entries=16`（或设 `INFR_RESPONSE_STORE_ENTRIES=16`）即开启服务端状态：`store` 的默认值随之变为 `true`，`previous_response_id` 与 `conversation` 可重放上一轮，`input` 里可以用 `{"type":"item_reference","id":"..."}` 引用此前输出的条目，并多出 `GET /v1/responses`、`GET|DELETE /v1/responses/{id}`、`GET /v1/responses/{id}/input_items` 与 `POST /v1/responses/{id}/cancel` 几个端点。历史链的用法是：

```powershell
$first = Invoke-RestMethod -Method Post -Uri "$base/v1/responses" -Headers $headers -ContentType 'application/json; charset=utf-8' `
  -Body ([Text.Encoding]::UTF8.GetBytes((@{ model = $chatModel; input = '用一句话解释 Prefill。' } | ConvertTo-Json)))
Invoke-RestMethod -Method Post -Uri "$base/v1/responses" -Headers $headers -ContentType 'application/json; charset=utf-8' `
  -Body ([Text.Encoding]::UTF8.GetBytes((@{ model = $chatModel; previous_response_id = $first.id; input = '那 Decode 呢？' } | ConvertTo-Json)))
Invoke-RestMethod -Uri "$base/v1/responses/$($first.id)/input_items" -Headers $headers
```

这份状态默认只活在进程内、LRU 有界、重启即失：被淘汰或随重启消失的 id，之后引用它会得到明确的 400，而不是悄悄从空上下文继续。要让链跨重启，再加 `--set serve.response_store_path=…`（或 `INFR_RESPONSE_STORE_PATH`）：它把同一份状态镜像成一个**明文** JSON 文件，每次改动后原子重写，启动时加载；文件无法解析会拒绝启动，而不会空库启动。`conversation` 配 `store=false` 同样被拒绝，因为未存储的轮次无法推进游标。`.../cancel` 总是返回 400——本服务器同步生成，响应在被赋予 id 时已经完成，要停止就断开这条流。请求里带 `include:["reasoning.encrypted_content"]` 时，reasoning 条目会多出一个 `encrypted_content` 句柄（`rehs_…`）：它是服务端存储的引用而不是密文，同样随 LRU、以及未配置路径时的重启而失效，把该句柄放进下一轮 `input` 的 `{"type":"reasoning","encrypted_content":"…"}` 即可重放这段推理内容；句柄无法解析时是明确的 400。这条路径要求 `store=true`（默认值随 `serve.response_store_entries` 开关而定）。当前仍不支持远程图片 URL、`file_id`、`input_file`、自动截断和 hosted tools；不要把它当作完整的云端 Responses API。

## 图片与嵌入

启用 Vision 并加载兼容的 mmproj 后，Chat Completions 的 `content` 可交错包含文本和 `image_url`。图片须为 data URI 或 base64；远程 HTTP URL 不受支持。以下示例读取本地 PNG：

支持 PNG、JPEG、WebP、GIF、BMP、TIFF、ICO；带 `image-codecs/` 的 Windows/Linux
发行包还支持 HEIC/HEIF（HEVC）和 AVIF。按文件内容识别，不依赖 MIME 或扩展名；
HEIF/AVIF 使用主图并应用文件内裁剪、旋转、镜像，再走同一视觉预处理。
不新增远程 URL 获取或动画播放。HEIF/AVIF 限制每边不超过 8192、面积不超过
64 Mi 像素、压缩数据不超过 256 MiB；损坏或不支持的编码会明确报错。

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
