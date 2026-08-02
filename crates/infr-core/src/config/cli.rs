//! The CLI layer: the bespoke clap flags plus `--set <config.path>=<value>`.
//!
//! `infr-core` cannot depend on clap (and must not), so the CLI fills a plain
//! [`ConfigOverrides`] and this module folds it into a [`PartialConfig`]. In S1 the CLI's
//! `DeviceOpts::resolve` / `SamplingOpts::resolve` stop writing `std::env::set_var` and write
//! `ConfigOverrides::flags` instead.
//!
//! **`--set` is ADDITIVE (§11 [DECIDE-3]).** Every flag that exists today keeps its name and
//! semantics; `--set` exists so the ~150 knobs with no dedicated flag are reachable without
//! inventing ~150 flags. Within the CLI layer:
//! - the bespoke flag WINS over a `--set` targeting the same field, and passing both prints a
//!   warning naming the field, so a user who typed `--ctx 32k --set device.ctx=8k` is told which
//!   one applied;
//! - two `--set`s for the same path are an ERROR, not a silent last-wins;
//! - an unknown `--set` path is a HARD error with a did-you-mean (§11 [DECIDE-5]/[DECIDE-6]) —
//!   unlike the file layer, which warns and ignores.
//!
//! **`--set` takes the CONFIG path, never the env name (§11 [DECIDE-6]).** They are not 1:1:
//! `INFR_NO_GEMM_WARP` maps to `gemm_warp=false`, `INFR_NO_GEMV_REG` and `INFR_GEMV_VARIANT` both
//! map to one `variant` field, and `INFR_MMV_MW` is tri-state. One grammar, copy-pasteable
//! between `--set` and the config file in both directions.

use std::path::PathBuf;

use super::{ConfigError, PartialConfig, SetPathError};

/// What the command line contributes to a `Config`.
///
/// `flags` is filled by the bespoke clap options; `sets` carries the raw `path=value` strings from
/// `--set`, in the order they were typed.
#[derive(Debug, Clone, Default)]
pub struct ConfigOverrides {
    /// `--config <PATH>`: skip the default lookup and use this file (an error if absent).
    pub config_path: Option<PathBuf>,
    /// `--set <config.path>=<value>`, unparsed.
    pub sets: Vec<String>,
    /// The bespoke flags (`--ctx`, `--temp`, …), already typed.
    pub flags: PartialConfig,
}

/// Fold [`ConfigOverrides`] into the CLI layer.
///
/// Warnings (a `--set` shadowed by a bespoke flag) are emitted through `tracing::warn!`;
/// [`parse_reporting`] is the pure half the tests use.
pub fn parse(overrides: &ConfigOverrides) -> Result<PartialConfig, ConfigError> {
    let (partial, warnings) = parse_reporting(overrides)?;
    for w in warnings {
        tracing::warn!("{w}");
    }
    Ok(partial)
}

/// [`parse`] without the logging side effect: returns the layer plus its warning lines.
pub fn parse_reporting(
    overrides: &ConfigOverrides,
) -> Result<(PartialConfig, Vec<String>), ConfigError> {
    let mut layer = PartialConfig::default();
    let mut seen: Vec<String> = Vec::new();
    let mut warnings = Vec::new();

    for entry in &overrides.sets {
        let (path, raw) = entry.split_once('=').ok_or_else(|| ConfigError::Value {
            path: entry.clone(),
            message: "expected `--set <config.path>=<value>`".to_string(),
        })?;
        let path = path.trim();
        if seen.iter().any(|p| p == path) {
            return Err(ConfigError::DuplicateSet {
                path: path.to_string(),
            });
        }
        match layer.set_path(path, raw) {
            Ok(()) => {}
            Err(SetPathError::Unknown) => {
                return Err(ConfigError::UnknownPath {
                    path: path.to_string(),
                    suggestion: super::suggest_path(path),
                })
            }
            Err(SetPathError::Value(message)) => {
                return Err(ConfigError::Value {
                    path: path.to_string(),
                    message,
                })
            }
        }
        if overrides.flags.is_path_set(path) {
            // Name the value the user typed — `--set x=` with an empty tail reads like a bug.
            warnings.push(format!(
                "[infr] config: `--set {path}={raw}` ignored — the dedicated flag for `{path}` wins"
            ));
        }
        seen.push(path.to_string());
    }

    // The bespoke flags go on LAST, so they win over any `--set` for the same field.
    layer.merge(overrides.flags.clone());
    Ok((layer, warnings))
}
