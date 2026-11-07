---
kind: guide
status: current
scope: reasoning-controls
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 原生思考控制

MoE4All 将思考控制传递给 GGUF 内嵌的 `tokenizer.chat_template`。模型模板定义这些控制的含义。推理级别不是 token 预算、采样温度或 MoE 路由变化。

## CLI 和 Windows 向导

```powershell
.\infr.exe run --think --reasoning-effort medium 'D:\Models\model.gguf'
.\infr.exe serve --think --reasoning-effort medium 'D:\Models\model.gguf'
```

Windows 启动向导会在思考开关选择器之后提供推理强度选择器，并记住该选择。禁用思考时，它不会传递强度标志。此次更改未为 GUI 应用新增下拉框；请使用向导、CLI 或 API 控制这些选项。

等效配置为：

```toml
[sampling]
no_think = false
reasoning_effort = "medium"
# API 历史推理保留；不改变终端对话仅保留回答的历史。
# preserve_thinking = true
```

`--set sampling.reasoning_effort=medium` 也可用。省略该设置即可使用模板默认值。`--set sampling.reasoning_effort=none` 会清除继承的配置值，并不会禁用思考；请使用 `--no-think` 禁用思考。

未引入新的环境变量。现有 `INFR_NO_THINK` 行为不变。强度设置不会隐式覆盖 `--no-think`。

## Chat Completions API

向 `POST /v1/chat/completions` 发送：

```json
{
  "model": "your-served-model-id",
  "messages": [{"role": "user", "content": "Explain this problem."}],
  "reasoning_effort": "medium",
  "chat_template_kwargs": {
    "enable_thinking": true,
    "preserve_thinking": true
  },
  "max_completion_tokens": 4096,
  "stream": true
}
```

使用 Python OpenAI SDK 时，请将 `chat_template_kwargs` 放入 `extra_body`。在 HTTP 层，它是顶层字段；请勿发送字面量 `extra_body` 包装对象。

支持的 `chat_template_kwargs` 为 `enable_thinking`（布尔值）、`preserve_thinking`（布尔值）和 `reasoning_effort`（字符串）。其他键会被拒绝，以避免它们替换保留的提示词数据或被静默忽略。强度可置于顶层或嵌套位置；相同的重复值可接受，冲突的重复值返回 400。

缺失/null 字段会继承服务器配置。配置中缺失的强度和保留键会完全从 Jinja 中省略，从而保留各模板的默认值。请求值覆盖进程默认值，但不改变共享配置。同一组选项会渲染生成提示词和用于前缀缓存的稳定历史。

如需保留先前推理，请在后续请求中将返回的助手 `reasoning_content` 连同 `content` 一起重放。`reasoning` 也被接受为回退；两者同时提供时，以 `reasoning_content` 为准。模板决定保留哪些轮次。按 Qwen 模板规定，Qwen3.8 中的 `preserve_thinking=false` 仍会在当前工具调用轮保留推理。保留历史可能增加上下文长度和处理成本。

交互式 REPL 保留现有的仅回答历史。此次变更在那里添加原生强度控制，不会重新设计 REPL 历史格式。

## 模型差异

| 模型/模板 | 原生强度行为 |
| --- | --- |
| Qwen3.8-Flash-Next / Qwen3.8-27B | `low`、`medium`、`xhigh`；默认 `xhigh`。`high` 无效。 |
| 不含 `reasoning_effort` 的 Qwen3 风格模板 | 思考开关与此前相同；显式强度会失败，而不是被静默忽略。 |
| 于 2026-09-09 检查的 Gemma 4 31B 官方模板 | 具有思考开关和保留控制，但没有 `reasoning_effort` 变量。不要假定存在 medium/high 级别。 |
| DeepSeek-V4-Flash-0731 | 官方编码器定义 `low`、`high`、`max`（默认 `low`）。它随附 Python 编码代码而非标准 Jinja 模板。转换后的 GGUF 必须包含实际使用 `reasoning_effort` 的等效模板。此补丁不添加 Python 编码器，也不替换 GGUF 模板。 |
| 其他模板，包括 GPT-OSS 风格提示词格式 | 可透传原生变量。这并不表示 MoE4All 可以加载所有模型架构。 |

接受的协议/配置词汇是 `low`、`medium`、`high`、`xhigh`、`max`。这不是通用的五级量表。模型特定验证仍由内嵌模板完成；不会执行 `high` -> `xhigh` 翻译或通用系统消息注入。从不引用显式请求的强度/保留选项的模板会被拒绝。引用该变量是必要的能力检查，不证明第三方模板正确实现了每一个级别。旧 GGUF 转换可能需要更新模型模板。

格式错误的选项会产生 OpenAI 风格的 HTTP 400。模型模板的显式输入拒绝会为非流式请求产生 HTTP 400。对于流式请求，头部可能已是 200：响应随后包含一个 `invalid_request_error` SSE 事件，并紧接恰好一个 `[DONE]`。客户端必须检查 SSE 错误事件。损坏的模板和后端故障仍属于服务器错误。

## 验证

使用 Rust stable 以及仓库 CI 和 Windows 构建说明中的平台构建前置条件（包括适用于 Vulkan 的 `glslc`）。

```powershell
cargo fmt --all
cargo fmt --all -- --check
cargo test -p infr-core -p infr-chat -p infr-server -p infr-cli --locked
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

若要用 Python Jinja2 独立检查固件预期，请在 Python 环境中安装 `Jinja2` 并运行 `python scripts/check-thinking-fixtures.py`。这不会执行 Rust 实现，也不能替代 Rust 测试。

Qwen 固件矩阵覆盖 20 种情形：默认值、全部三个受支持级别、无效级别、关闭思考、系统消息、工具、推理保留和稳定历史。Rust 测试还覆盖经 JSON/SSE 的 DTO 传递、冲突、错误类型、保留字段、模板能力、并发渲染、配置继承和类型化输入错误。这些测试不需要模型权重。

请在 Windows 上使用实际 GGUF 单独执行 GPU 验收：比较 low/medium/xhigh 的渲染提示词，以相同采样运行短生成，验证历史/工具轮次，并在连续调用间切换级别。不同级别不必在每个问题上都产生单调递增的 token 数。此补丁不设置硬性的思考 token 限制。

## 来源和固件溯源信息

- [Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Qwen3.8-Flash-Next template](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/chat_template.jinja)
- [Qwen3.8-27B template](https://huggingface.co/Qwen/Qwen3.8-27B/blob/412f8b6/chat_template.jinja)
- [Gemma 4 31B template](https://huggingface.co/google/gemma-4-31B-it/blob/main/chat_template.jinja)
- [DeepSeek-V4-Flash-0731 encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/encoding/encoding_dsv4.py)

`crates/infr-chat/tests/fixtures/qwen38_chat_template.jinja` 是 Qwen/Qwen3.8-27B 的 Apache-2.0 模板经空白规范化后的文本提取，修订版为 `412f8b6`，读取于 2026-09-09。版权属于 Qwen 作者。其 170 行提取的源代码与 Flash-Next 模板（修订版 `34567a4`）比较后完全相同。它仅是回归测试固件；生产环境始终使用实际 GGUF 元数据。仓库的 Apache-2.0 许可证适用于该固件。未包含模型权重。
