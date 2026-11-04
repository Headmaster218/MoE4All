//! The startup half of the control surface: which direction file this session projects, and over
//! which layers.
//!
//! The three values are ordinary config knobs — `uncensor.vector`, `uncensor.first_layer` and
//! `uncensor.last_layer`, with the `INFR_UNCENSOR_*` spellings as their env aliases — so reading,
//! layering (TOML / env / `--set`) and the anti-drift accounting all happen in
//! [`infr_core::config`]. This crate takes the RESULT: [`Config`] is the engine-shaped view of that
//! table, and it is read ONCE, at model load, because the projection is allocated into the session
//! then or never.
//!
//! The per-request half of the control surface is the API field (`uncensor`), not a config key;
//! see [`crate::api_merge`] and [`crate::effective`].

use crate::{DEFAULT_FIRST_LAYER, DEFAULT_LAST_LAYER};
use std::path::PathBuf;

/// Env alias of [`infr_core::config::Config::uncensor`]`::vector`. Kept as a name because a refusal
/// to honour a named file must say which knob named it.
pub const ENV_VECTOR: &str = "INFR_UNCENSOR_VECTOR";
/// Env alias of `uncensor.first_layer`. Named exactly as v1 named it, so one launcher script drives
/// either engine.
pub const ENV_FIRST: &str = "INFR_UNCENSOR_FIRST_LAYER";
/// Env alias of `uncensor.last_layer`.
pub const ENV_LAST: &str = "INFR_UNCENSOR_LAST_LAYER";

/// What startup asked for, resolved once at model load.
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
    /// Off by default: naming a direction file IS the request for the projection, so the absence of
    /// one must leave a session bit-for-bit as it was before the feature existed.
    fn default() -> Self {
        Self {
            vector: None,
            first_layer: DEFAULT_FIRST_LAYER,
            last_layer: DEFAULT_LAST_LAYER,
        }
    }
}

impl Config {
    /// The engine-shaped view of the config table's `uncensor` section. An empty `vector` string is
    /// the config layer's `None`, so "explicitly off" and "never mentioned" arrive as one state.
    pub fn from_config(cfg: &infr_core::config::UncensorCfg) -> Self {
        Self {
            vector: cfg
                .vector
                .as_ref()
                .filter(|path| !path.as_os_str().is_empty())
                .cloned(),
            first_layer: cfg.first_layer,
            last_layer: cfg.last_layer,
        }
    }

    /// Whether a direction file was named — the only thing that makes the feature exist.
    pub fn enabled(&self) -> bool {
        self.vector.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table's defaults are this crate's defaults, and the feature stays off until a file is
    /// named — including when the layers alone are set, which is the "half a switch" state that must
    /// not allocate a projection nobody asked for.
    #[test]
    fn the_table_maps_to_the_engine_view_and_stays_off_until_a_file_is_named() {
        let table = infr_core::config::UncensorCfg::default();
        let cfg = Config::from_config(&table);
        assert_eq!(cfg.vector, None);
        assert!(!cfg.enabled());
        assert_eq!(
            (cfg.first_layer, cfg.last_layer),
            (DEFAULT_FIRST_LAYER, DEFAULT_LAST_LAYER)
        );

        let table = infr_core::config::UncensorCfg {
            vector: Some(PathBuf::from("directions.gguf")),
            first_layer: 2,
            last_layer: 9,
        };
        let cfg = Config::from_config(&table);
        assert_eq!(cfg.vector, Some(PathBuf::from("directions.gguf")));
        assert!(cfg.enabled());
        assert_eq!((cfg.first_layer, cfg.last_layer), (2, 9));

        // An empty path is "off", not "a file called nothing".
        let table = infr_core::config::UncensorCfg {
            vector: Some(PathBuf::new()),
            ..Default::default()
        };
        assert!(!Config::from_config(&table).enabled());
    }
}
