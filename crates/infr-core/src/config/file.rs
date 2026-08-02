//! The config-file layer: TOML → [`PartialConfig`], plus the file lookup.
//!
//! **Format (§4).** Section path == struct path, and the file speaks the POSITIVE field names —
//! `gemm_warp = false`, never `no_gemm_warp`. The `INFR_NO_*` spellings exist only in the
//! environment layer.
//!
//! ```toml
//! [device]
//! ctx = "32k"          # the shared size grammar: 8192 / 256k / 50%
//!
//! [kv]
//! type_k = "q8_0"
//!
//! [kernels.vulkan]
//! flash_splits = 2
//! gemm_warp = false
//! ```
//!
//! **Lookup (§4, first EXISTING file wins, no merging across files):**
//! 1. `--config <PATH>` — an error if the path does not exist,
//! 2. `./infr.toml`,
//! 3. `$XDG_CONFIG_HOME/infr/config.toml`, else `~/.config/infr/config.toml`.
//!
//! An absent file is a no-op, never an error. A MALFORMED file is an error.
//!
//! **Unknown keys warn and are ignored (§11 [DECIDE-5]).** That is what lets an older binary read
//! a file written for a newer one, and makes deleting a knob a non-breaking change. Typo
//! protection comes from the warning line, not from a hard failure. `--set` is the opposite: an
//! unknown path there is a hard error, because it was typed for this run. A value of the wrong
//! TYPE for a KNOWN key stays a hard error either way.
//!
//! The walk is over `toml::Value` rather than a serde-derived struct precisely because
//! `deny_unknown_fields` can only fail — it cannot warn-and-continue, and it cannot name the
//! surviving keys.

use std::path::{Path, PathBuf};

use super::{Config, ConfigError, PartialConfig, SetPathError};

/// Resolve and parse the config file. Returns the layer plus the file it came from; `None` means
/// no file was found anywhere in the lookup, which is a no-op.
///
/// Any unknown-key warnings are emitted through `tracing::warn!` here; use [`parse_str`] for the
/// pure half.
pub fn load(explicit: Option<&Path>) -> Result<(PartialConfig, Option<PathBuf>), ConfigError> {
    let Some(path) = discover(explicit)? else {
        return Ok((PartialConfig::default(), None));
    };
    let text = std::fs::read_to_string(&path).map_err(|source| ConfigError::Io {
        path: path.clone(),
        source,
    })?;
    let (partial, warnings) = parse_str(&text, &path)?;
    for w in warnings {
        tracing::warn!("{w}");
    }
    Ok((partial, Some(path)))
}

/// The 3-step lookup. `Ok(None)` = nothing found (not an error).
pub fn discover(explicit: Option<&Path>) -> Result<Option<PathBuf>, ConfigError> {
    if let Some(p) = explicit {
        // An EXPLICIT path that is not there is a user error — unlike the default locations.
        return if p.exists() {
            Ok(Some(p.to_path_buf()))
        } else {
            Err(ConfigError::Missing(p.to_path_buf()))
        };
    }
    let cwd = PathBuf::from("infr.toml");
    if cwd.exists() {
        return Ok(Some(cwd));
    }
    // Not `INFR_*` and not engine configuration — the standard XDG location of the file itself,
    // which by definition cannot come from the file (§5.3).
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")));
    let global = base.map(|b| b.join("infr").join("config.toml"));
    Ok(global.filter(|p| p.exists()))
}

/// Parse TOML text into a layer. Returns the layer plus one warning line per unknown key.
///
/// `label` only appears in messages — it does not have to exist on disk, which is what lets the
/// precedence tests drive this with a string literal.
pub fn parse_str(text: &str, label: &Path) -> Result<(PartialConfig, Vec<String>), ConfigError> {
    let doc: toml::Table = toml::from_str(text).map_err(|e| ConfigError::Toml {
        path: label.to_path_buf(),
        message: e.message().to_string(),
    })?;
    let known = Config::all_paths();
    let mut partial = PartialConfig::default();
    let mut warnings = Vec::new();
    walk(&doc, "", &known, &mut partial, &mut warnings)?;
    Ok((partial, warnings))
}

/// Recursive descent over the document, turning each scalar leaf into a `set_path` call.
fn walk(
    table: &toml::Table,
    prefix: &str,
    known: &[String],
    out: &mut PartialConfig,
    warnings: &mut Vec<String>,
) -> Result<(), ConfigError> {
    for (key, value) in table {
        let path = format!("{prefix}{key}");
        if let toml::Value::Table(sub) = value {
            // A table is only a SECTION if some known path lives under it; otherwise the whole
            // subtree is unknown and warning once beats warning per leaf.
            let section_prefix = format!("{path}.");
            if known.iter().any(|k| k.starts_with(&section_prefix)) {
                walk(sub, &section_prefix, known, out, warnings)?;
            } else {
                warnings.push(unknown_line(&path));
            }
            continue;
        }
        let raw = match scalar(value) {
            Some(s) => s,
            None => {
                return Err(ConfigError::Value {
                    path,
                    message: "expected a string, number, boolean or array".to_string(),
                })
            }
        };
        match out.set_path(&path, &raw) {
            Ok(()) => {}
            Err(SetPathError::Unknown) => warnings.push(unknown_line(&path)),
            Err(SetPathError::Value(message)) => return Err(ConfigError::Value { path, message }),
        }
    }
    Ok(())
}

/// The one line an unknown key prints. Shaped so a typo is obvious at a glance.
fn unknown_line(path: &str) -> String {
    match super::suggest_path(path) {
        Some(s) => format!("[infr] config: unknown key `{path}` (ignored) — did you mean `{s}`?"),
        None => format!("[infr] config: unknown key `{path}` (ignored)"),
    }
}

/// A TOML scalar (or a flat array, for the device lists) in the `--set` spelling. `None` for a
/// value shape no config field can hold.
fn scalar(value: &toml::Value) -> Option<String> {
    Some(match value {
        toml::Value::String(s) => s.clone(),
        toml::Value::Integer(i) => i.to_string(),
        toml::Value::Float(f) => f.to_string(),
        toml::Value::Boolean(b) => b.to_string(),
        toml::Value::Array(items) => {
            let parts: Option<Vec<String>> = items.iter().map(scalar).collect();
            parts?.join(",")
        }
        toml::Value::Datetime(_) | toml::Value::Table(_) => return None,
    })
}
