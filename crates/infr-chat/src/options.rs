use infr_core::config::{Config, ReasoningEffort};

/// Per-request template controls. No global state: an absent field inherits the process default.
/// These are prompt controls, not token budgets or sampling parameters.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatTemplateOptions {
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub preserve_thinking: Option<bool>,
}

impl ChatTemplateOptions {
    /// Thinking stays enabled by default, preserving MoE4All's existing no_think policy.
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            enable_thinking: Some(!cfg.sampling.no_think),
            reasoning_effort: cfg.sampling.reasoning_effort,
            preserve_thinking: cfg.sampling.preserve_thinking,
        }
    }

    /// Merge one request without mutating the server's shared configuration.
    pub fn resolve(&self, cfg: &Config) -> Self {
        let defaults = Self::from_config(cfg);
        Self {
            enable_thinking: self.enable_thinking.or(defaults.enable_thinking),
            reasoning_effort: self.reasoning_effort.or(defaults.reasoning_effort),
            preserve_thinking: self.preserve_thinking.or(defaults.preserve_thinking),
        }
    }
}
