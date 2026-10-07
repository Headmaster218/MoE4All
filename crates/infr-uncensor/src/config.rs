//! The startup half of the control surface: which direction file this session projects, and over
//! which layers.
//!
//! This deliberately does NOT go through `infr_core::config` (the `infr.toml` / `INFR_*` / CLI
//! three-layer system with its manifest and its two drift-checking test lists). The knob is
//! session-scoped — the file is read once, at model load, and the choice cannot change without a
//! restart — so it enters as three environment variables that a launcher sets and the engine
//! reads. Adding it to the config subsystem would mean touching the manifest, the env table, the
//! example TOML, the reference doc, and both anti-drift test lists, for a knob nobody edits in a
//! TOML file while the server is running.

use crate::{DEFAULT_FIRST_LAYER, DEFAULT_LAST_LAYER};
use anyhow::{anyhow, Result};
use std::ffi::OsString;
use std::path::PathBuf;

/// Path of the control-vector file. Absent (or empty) means no projection at all.
pub const ENV_VECTOR: &str = "INFR_UNCENSOR_VECTOR";
/// Inclusive first layer to project, default [`DEFAULT_FIRST_LAYER`]. Named exactly as v1 named it,
/// so a launcher script works against either engine.
pub const ENV_FIRST: &str = "INFR_UNCENSOR_FIRST_LAYER";
/// Inclusive last layer to project, default [`DEFAULT_LAST_LAYER`] (clipped to the model's depth).
pub const ENV_LAST: &str = "INFR_UNCENSOR_LAST_LAYER";

/// What startup asked for. Read ONCE, at model load.
#[derive(Debug, Clone)]
pub struct Config {
    /// The direction file, or `None` for "no projection in this session".
    pub vector: Option<PathBuf>,
    /// Inclusive lower layer bound.
    pub first_layer: usize,
    /// Inclusive upper layer bound.
    pub last_layer: usize,
}

impl Default for Config {
    /// Off by default: naming a direction file IS the request for the projection, so the absence
    /// of one must leave a session bit-for-bit as it was before the feature existed.
    fn default() -> Self {
        Self {
            vector: None,
            first_layer: DEFAULT_FIRST_LAYER,
            last_layer: DEFAULT_LAST_LAYER,
        }
    }
}

impl Config {
    /// Read the three `INFR_UNCENSOR_*` variables.
    pub fn from_env() -> Result<Self> {
        Self::parse(|k| std::env::var_os(k))
    }

    /// [`Config::from_env`] over an injected lookup, so the parsing rules are testable without
    /// mutating the process environment (which every other test in the binary reads concurrently).
    fn parse(get: impl Fn(&str) -> Option<OsString>) -> Result<Self> {
        let vector = parse_vector(get(ENV_VECTOR));
        Ok(Self {
            first_layer: parse_layer(ENV_FIRST, get(ENV_FIRST), DEFAULT_FIRST_LAYER)?,
            last_layer: parse_layer(ENV_LAST, get(ENV_LAST), DEFAULT_LAST_LAYER)?,
            vector,
        })
    }

    /// Whether this session asks for a projection at all. Cheap, and the question the engine asks
    /// before it allocates anything for the feature.
    pub fn enabled(&self) -> bool {
        self.vector.is_some()
    }
}

/// An unset variable and a set-but-empty one mean the same thing: no file. A path is never an
/// empty string in practice, and `""` cannot be opened — treating it as "off" rather than as a
/// path keeps an exporter that writes `VAR=` for "disabled" from failing the launch.
fn parse_vector(raw: Option<OsString>) -> Option<PathBuf> {
    let raw = raw?;
    if raw.is_empty() {
        return None;
    }
    Some(PathBuf::from(raw))
}

/// A layer bound is a non-negative integer, or the session does not start. A typo'd layer range
/// silently falling back to the default would project layers the operator did not name.
fn parse_layer(name: &str, raw: Option<OsString>, default: usize) -> Result<usize> {
    let Some(raw) = raw else {
        return Ok(default);
    };
    let text = raw.to_string_lossy();
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(default);
    }
    trimmed.parse::<usize>().map_err(|_| {
        anyhow!(
            "{name} = `{text}` is not a layer index; expected a non-negative integer (or unset \
             for the default {default})"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn lookup(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
        let map: HashMap<String, OsString> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), OsString::from(*v)))
            .collect();
        move |k: &str| map.get(k).cloned()
    }

    #[test]
    fn unset_means_off_with_the_shipped_layer_range() {
        let c = Config::parse(|_| None).unwrap();
        assert_eq!(c.vector, None);
        assert!(!c.enabled());
        assert_eq!(
            (c.first_layer, c.last_layer),
            (DEFAULT_FIRST_LAYER, DEFAULT_LAST_LAYER)
        );
    }

    #[test]
    fn an_empty_vector_value_is_off_but_a_named_one_is_on() {
        let off = Config::parse(lookup(&[(ENV_VECTOR, "")])).unwrap();
        assert_eq!(
            off.vector, None,
            "a launcher that clears by writing `VAR=` must not fail to start"
        );

        let on = Config::parse(lookup(&[(ENV_VECTOR, "directions.gguf")])).unwrap();
        assert_eq!(on.vector, Some(PathBuf::from("directions.gguf")));
        assert!(on.enabled());
    }

    #[test]
    fn layer_bounds_are_read_and_trimmed() {
        let c = Config::parse(lookup(&[
            (ENV_VECTOR, "cvec.gguf"),
            (ENV_FIRST, " 2 "),
            (ENV_LAST, "9"),
        ]))
        .unwrap();
        assert_eq!((c.first_layer, c.last_layer), (2, 9));
    }

    #[test]
    fn an_unset_bound_keeps_the_default() {
        let c = Config::parse(lookup(&[(ENV_FIRST, "8")])).unwrap();
        assert_eq!(c.first_layer, 8);
        assert_eq!(c.last_layer, DEFAULT_LAST_LAYER);
    }

    #[test]
    fn a_layer_bound_that_is_not_a_number_refuses_to_start() {
        for bad in ["-1", "four", "4.0", "1 2"] {
            let e = Config::parse(lookup(&[(ENV_LAST, bad)]))
                .unwrap_err()
                .to_string();
            assert!(e.contains(ENV_LAST) && e.contains(bad), "{bad} => {e}");
        }
    }

    #[test]
    fn a_bound_is_still_checked_when_the_feature_is_off() {
        // `INFR_UNCENSOR_LAST=nonsense` without a vector names a range nobody uses, and it is still
        // a typo worth reporting rather than swallowing.
        let e = Config::parse(lookup(&[(ENV_FIRST, "nope")]))
            .unwrap_err()
            .to_string();
        assert!(e.contains(ENV_FIRST), "{e}");
    }
}
