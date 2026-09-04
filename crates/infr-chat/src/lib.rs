//! Shared chat text logic — turning conversations into model prompts and parsing model output back
//! out. **No GPU, no model, no inference.** Used by `infr-llama` (to feed the forward pass) and by
//! `infr-engine`/`infr-server` (the OpenAI-shaped API) so prompt rendering lives in ONE place instead
//! of being re-implemented next to each backend.
//!
//! - [`render_chat_jinja`] / [`render_chat_user`] — render a GGUF's embedded `tokenizer.chat_template`
//!   (jinja, via minijinja) into a prompt string. Return `None` when the GGUF has no template (or it
//!   fails to render); callers fail loud rather than fabricate a default — infr only supports
//!   models that ship a chat template.
//! - [`split_channels`] / [`parse_tool_calls`] — parse model output (reasoning vs answer,
//!   `<|tool_call>` blocks).

mod stream;
mod template;
mod tools;

pub use stream::{prompt_prefills_think, ChatStream, Delta};
pub use template::{
    render_chat_jinja, render_chat_oai, render_chat_user, render_template, TemplateError,
    IMAGE_PART_PLACEHOLDER, VISION_MARKER,
};
pub use tools::{
    parse_any_tool_calls, parse_hermes_tool_calls, parse_tool_calls, split_channels,
    split_reasoning, split_think, ToolCall,
};

/// One chat message (OpenAI-shaped; tool fields preserved for the agentic round-trip).
///
/// Deserializable so a host embedding this type can round-trip an OpenAI-shaped payload; every
/// field carries a serde default, so a payload without the newer fields (e.g. `images`) still
/// deserializes.
#[derive(Clone, Debug, Default, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
    /// The assistant's OUTGOING tool calls (OpenAI `message.tool_calls`), replayed into the prompt on
    /// the next turn so the model sees its own prior calls. Empty/None for non-assistant messages.
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// For a `tool`-role result message: which call it answers (OpenAI `tool_call_id`).
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    /// Vision input (stage V5): this message's `image_url` part payloads, IN PART ORDER —
    /// `data:` URIs or bare base64 (`infr_vision::decode_image_input` accepts both). Kept beside
    /// the flattened text content (which only joins `text` parts) so the chat layer can pair the
    /// rendered prompt's `<|image_pad|>` markers with their images in order. Empty on every
    /// text-only message, which is exactly the pre-V5 shape.
    #[serde(default)]
    pub images: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A payload from BEFORE the `images` field (and without the optional tool fields) must still
    /// deserialize — `#[serde(default)]` on every field keeps older/newer producers compatible.
    #[test]
    fn chat_message_without_images_deserializes_with_empty_images() {
        let m: ChatMessage =
            serde_json::from_str(r#"{"role": "user", "content": "hi"}"#).expect("deserialize");
        assert_eq!(m.role, "user");
        assert_eq!(m.content, "hi");
        assert!(m.images.is_empty(), "absent images field defaults to empty");
        assert!(m.tool_calls.is_none());
        assert!(m.tool_call_id.is_none());
        assert!(m.name.is_none());

        // …and a payload that DOES carry images keeps them, in order.
        let m: ChatMessage = serde_json::from_str(
            r#"{"role": "user", "content": "look", "images": ["data:image/png;base64,AAA", "BBB"]}"#,
        )
        .expect("deserialize");
        assert_eq!(
            m.images,
            vec!["data:image/png;base64,AAA".to_string(), "BBB".to_string()]
        );
    }
}
