//! Chat text logic re-exports shared by `infr run` and `infr serve` (prompt rendering,
//! channel/tool-call parsing, the reasoning/content delta splitter). The load-pipeline/session
//! orchestration this crate originally stubbed out (the `Engine` struct) is now the unified
//! transformer runner in `infr-llama` (`SeamModel` + the per-backend `ChatModel` impls, see
//! `docs/diffusion-gemma.md` and `docs/qwen35.md`) — every arch, including DiffusionGemma, is built
//! on it, so there is no separate "engine" seam left to implement here.
//!
//! Reference for chat/channel/tool behavior: `~/Projects/scratch/dgemma-openai-server.py`.

// Chat text logic (prompt rendering, channel/tool-call parsing, message types) lives in the shared
// `infr-chat` crate so backends and the server share ONE implementation. Re-exported here for the
// existing `infr_engine::{ChatMessage, …}` call sites.
pub use infr_chat::{
    parse_hermes_tool_calls, parse_tool_calls, split_channels, split_reasoning, split_think,
    ChatMessage, ToolCall, IMAGE_PART_PLACEHOLDER,
};

/// A streamed piece of a response (re-exported from the single splitter in `infr-chat`).
pub use infr_chat::{prompt_prefills_think, ChatStream, Delta};
