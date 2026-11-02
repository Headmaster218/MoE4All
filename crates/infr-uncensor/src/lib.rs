//! Uncensor: project a refusal direction out of the residual, and let the caller switch it off.
//!
//! # What the feature does
//!
//! A model that refuses has learned to move its own residual toward one direction at the layers
//! where it decides whether to answer at all. Projecting that direction out —
//! `h -= (h·v)·v`, once per layer, applied to every one of `qwen4exp`'s hyper-connection residual
//! streams — removes the tendency without touching the weights, so the model answers prompts it
//! would have turned down. Removing a direction the model wanted to move along costs a little of
//! what it had to say, which is why the layer range stops well before the head.
//!
//! The directions come from a llama.cpp-compatible `controlvector` GGUF (one `[n_embd]` f32 per
//! layer, named `direction.1` …), i.e. `--cvec-mode project --cvec-dir per-layer
//! --control-vector-layer-range {first} {last}`.
//!
//! # Why this is a crate of its own
//!
//! The engine has to do exactly three things for this feature: read one file at model load, emit
//! one projection op per covered layer, and honour one boolean per request. Everything that
//! DECIDES — is this file a control-vector file, which layers does it cover, is this request on or
//! off, what does an absent field mean, what value goes in the scale slot — lives here, so the
//! engine's diff stays a handful of lines and every rule about a downloaded file or an API field
//! is testable without a GPU or a model.
//!
//! # Control, in two layers
//!
//! 1. **Startup decides whether a projection EXISTS.** [`Config`] reads `INFR_UNCENSOR_VECTOR`
//!    (plus the two layer bounds) once, at model load. No file ⇒ the feature is absent: nothing is
//!    allocated, nothing is logged, and the model runs exactly as it did before. A file that
//!    cannot be honoured is a hard error, never a quiet disable — naming a direction file IS the
//!    request for the projection, and not doing it is the one outcome nobody can see from outside.
//! 2. **Each request decides whether it is ON.** [`api_merge`] turns the two spellings of the
//!    request-level field into one `Option<bool>`; [`effective`] turns that, together with what the
//!    session last ran, into the boolean the graph's scale slot gets.
//!
//! The switch is a device-side scalar read by the projection op, not a graph rebuild: with the
//! scale at `0.0` the op computes `h - 0·v`, which is the original model bit for bit, so a cached
//! or replayed plan stays valid across the switch. What the switch DOES cost is that a session
//! with a loaded file pays for the op on every covered layer even while it is off (measured at a
//! few tenths of a percent) — the alternative, baking the boolean into the graph, would throw the
//! compiled plan and the prompt cache away on every flip. Flipping is still not free in the sense
//! that matters: the KV cache holds activations that were computed under the other setting, so a
//! flip must invalidate it (see [`flipped`]), exactly as llama.cpp and Strata re-read the prompt
//! when `cvec` changes.
//!
//! Absent field means ON, matching Strata: if the operator installed a projection, they meant it,
//! and the per-request field is for turning it off for one call.

pub mod config;
mod vectors;

pub use config::Config;
pub use vectors::{load, ControlVectors};

/// `general.architecture` of the models this projection applies to: the wide multi-stream residual
/// of Qwen3.8. A single-stream model has one direction to project out of one vector, which is a
/// different (and much less useful) operation, so this feature does not claim to support it.
pub const QWEN4EXP: &str = "qwen4exp";

/// Default inclusive first projected layer. The layers below it are where the model works out what
/// the user actually asked for; projecting a refusal direction there damages comprehension.
pub const DEFAULT_FIRST_LAYER: usize = 4;
/// Default inclusive last projected layer, clipped to the model's depth at load. The layers above
/// it are where the model has already committed to an answer, and a direction removed too late
/// shows up as a model that rambles.
pub const DEFAULT_LAST_LAYER: usize = 44;

/// The request-level field, and the Strata spelling of the same field accepted as an alias.
pub const API_FIELD: &str = "uncensor";
/// See [`API_FIELD`]. Both spellings mean the same thing; giving both is only an error when they
/// disagree.
pub const API_ALIAS: &str = "experimental_speed_projection";

/// A control-vector file that has been checked against the model it is about to be projected onto,
/// plus the layer range it is projected over.
///
/// Carrying the range rather than re-deriving it is deliberate: cold init and the per-token graph
/// builder both go through [`resolve`], so the two can never be told different answers.
#[derive(Debug, Clone)]
pub struct Projection {
    dirs: ControlVectors,
    first: usize,
    last: usize,
}

impl Projection {
    /// Whether layer `l` (zero-based, as everywhere in the engine) is projected.
    pub fn covers(&self, l: usize) -> bool {
        self.first <= l && l <= self.last
    }

    /// The inclusive layer range this projection runs over.
    pub fn range(&self) -> (usize, usize) {
        (self.first, self.last)
    }

    /// Model width, i.e. the width of one direction.
    pub fn n_embd(&self) -> usize {
        self.dirs.n_embd()
    }

    /// Element offset of layer `l`'s direction in [`Projection::bytes`].
    pub fn dir_off(&self, l: usize) -> usize {
        self.dirs.dir_off(l)
    }

    /// The upload payload: every layer's direction as one contiguous f32 block.
    pub fn bytes(&self) -> &[u8] {
        self.dirs.bytes()
    }

    /// Size of that payload, in bytes.
    pub fn byte_len(&self) -> usize {
        self.dirs.byte_len()
    }

    /// Size of that payload in f32 elements — the tensor declaration a backend graph needs for the
    /// one Input the directions are bound to.
    pub fn n_floats(&self) -> usize {
        self.dirs.n_layer_dirs() * self.dirs.n_embd()
    }
}

/// The layer range a session projects, INCLUSIVE — `None` when no file is configured, which is the
/// default and the state every session that has not heard of this feature is in.
///
/// Only the CEILING is clipped, to the layers the model has. The floor is never moved down: a
/// request for "from layer 4 on" against a 3-layer model asks for nothing, and silently projecting
/// layer 2 would be a projection nobody asked for. [`resolve`] turns that `None` into the error it
/// is, and announces every clip; this function never logs, since the graph builder calls it.
///
/// Pure in `(config, n_layer)` and free of I/O, because the decode loop needs the SAME answer on a
/// warm call that cold init computed with it: the file is read once, the range is asked for once
/// per graph.
pub fn span(cfg: &Config, n_layer: usize) -> Option<(usize, usize)> {
    // The feature IS the file: without one there is no span, and nothing downstream is built.
    cfg.vector.as_ref()?;
    // Inclusive bounds against an inclusive ceiling.
    let last = cfg.last_layer.min(n_layer.saturating_sub(1));
    (cfg.first_layer <= last).then_some((cfg.first_layer, last))
}

/// Resolve the startup request against the model it is about to be pointed at: `None` when no file
/// is configured, a loaded and checked [`Projection`] otherwise.
///
/// The checks are ordered so that the cheapest and most likely mistake is reported first: a file
/// named for an architecture that has nothing to project, then a range that this model cannot
/// serve, then the file itself. Everything but the layer-ceiling clip is a hard error — the one
/// exception exists because [`DEFAULT_LAST_LAYER`] comes from one particular model, and a smaller
/// model must not fail to start over a default nobody chose.
pub fn resolve(
    cfg: &Config,
    qwen4exp: bool,
    n_embd: usize,
    n_layer: usize,
) -> anyhow::Result<Option<Projection>> {
    let Some(path) = cfg.vector.as_ref() else {
        return Ok(None);
    };
    if !qwen4exp {
        return Err(anyhow::anyhow!(
            "{} projects a direction onto the several residual streams of {QWEN4EXP}; this model \
             has one residual stream to project, so the feature does not apply to it",
            config::ENV_VECTOR
        ));
    }
    let Some((first, last)) = span(cfg, n_layer) else {
        return Err(anyhow::anyhow!(
            "{}-{} covers no layer of this {}-layer model",
            cfg.first_layer,
            cfg.last_layer,
            n_layer
        ));
    };
    let dirs = load(path, n_embd)?;
    if (first, last) != (cfg.first_layer, cfg.last_layer) {
        tracing::warn!(
            requested = %format!("{}-{}", cfg.first_layer, cfg.last_layer),
            used = %format!("{first}-{last}"),
            "uncensor: layer range clipped to the layers this model has"
        );
    }
    // The file is NOT clipped: a range the file cannot serve means the wrong file was named (or a
    // truncated one), and running the range the file DOES cover would quietly project layers the
    // operator did not ask for instead of the ones they did.
    if last > dirs.last_layer() {
        return Err(anyhow::anyhow!(
            "{} covers directions for layers 0..{}, but this session projects through layer {last} \
             — name a direction file for this model, or lower {}",
            path.display(),
            dirs.last_layer(),
            config::ENV_LAST
        ));
    }
    tracing::info!(
        vector = %path.display(),
        layers = %format!("{first}-{last}"),
        "uncensor: projecting one direction out of the residual after these layers"
    );
    Ok(Some(Projection { dirs, first, last }))
}

/// The two spellings of the per-request field were both given, and they disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conflict {
    /// Value of [`API_FIELD`].
    pub explicit: bool,
    /// Value of [`API_ALIAS`].
    pub alias: bool,
}

impl std::fmt::Display for Conflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "`{API_FIELD}` = {} contradicts `{API_ALIAS}` = {} — they are two names for one \
             switch, so send one value",
            self.explicit, self.alias
        )
    }
}

impl std::error::Error for Conflict {}

/// Merge the request's two spellings into the one three-valued question: `None` = "the caller has
/// no opinion", `Some(_)` = "run this way".
///
/// A field that is present must be a boolean, which the JSON layer already guarantees by the time
/// this is called; disagreeing spellings are refused rather than resolved by precedence, because a
/// client that sent both is confused about what it asked for and a silent winner hides it.
pub fn api_merge(
    explicit: Option<bool>,
    alias: Option<bool>,
) -> anyhow::Result<Option<bool>, Conflict> {
    match (explicit, alias) {
        (Some(a), Some(b)) if a != b => Err(Conflict {
            explicit: a,
            alias: b,
        }),
        (Some(a), _) => Ok(Some(a)),
        (_, Some(b)) => Ok(Some(b)),
        _ => Ok(None),
    }
}

/// The boolean to run this request with.
///
/// `requested` is the caller's answer ([`api_merge`]'s), `last_applied` is what THIS session last
/// really ran (`None` until its first forward pass, or after a restart). An internal pass that
/// carries no request of its own — an MTP draft, a verifier step, a speculative re-forward — must
/// inherit the session's current setting rather than fall back to the default: a verifier that ran
/// the opposite projection from the request it is verifying would propose tokens for a model that
/// is not the one being sampled from.
pub fn effective(requested: Option<bool>, last_applied: Option<bool>) -> bool {
    requested.or(last_applied).unwrap_or(true)
}

/// The scale slot's value for a decision. `0.0` makes the op compute `h - 0·v`, i.e. the untouched
/// residual, which is why the switch needs no graph change; `1.0` removes the whole component.
///
/// Only these two values are meaningful: a partial projection (`0.3`) is a knob nobody asked for,
/// and a control vector whose direction is unit-normalized is a boolean in disguise — anything
/// between 0 and 1 leaves part of the refusal in place, which reads as "it worked a bit".
pub fn scale(on: bool) -> f32 {
    if on {
        1.0
    } else {
        0.0
    }
}

/// Whether the projection state differs from what this session last ran, and therefore whether its
/// KV cache and any live/checkpointed session state computed from it must be thrown away.
///
/// The cached prefix is a set of activations produced under the OLD setting; keeping it while
/// running under the new one means the first tokens of the answer come from a residual that was
/// never projected the way the rest of the run projects it. That is the same reason llama.cpp and
/// Strata re-read the prompt when `cvec` flips, and it is the one thing here that a "just a
/// boolean" switch does not get for free.
pub fn flipped(last_applied: Option<bool>, now: bool) -> bool {
    last_applied != Some(now)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn cfg(vector: Option<&str>, first: usize, last: usize) -> Config {
        Config {
            vector: vector.map(PathBuf::from),
            first_layer: first,
            last_layer: last,
        }
    }

    /// `span` is asked once per graph, and must answer exactly what cold init announced: no file ⇒
    /// off, inclusive bounds, clipped to the layers that exist.
    #[test]
    fn span_is_off_without_a_file_and_clips_a_file_that_names_one() {
        assert_eq!(span(&Config::default(), 48), None);

        let uc = cfg(Some("cvec.gguf"), DEFAULT_FIRST_LAYER, DEFAULT_LAST_LAYER);
        assert_eq!(span(&uc, 48), Some((4, 44)));
        // A smaller model: the ceiling moves, the request does not become an empty range.
        assert_eq!(span(&uc, 20), Some((4, 19)));
        // A model with no layer at or above the floor: nothing to project, not a panic.
        assert_eq!(span(&uc, 3), None);
        // One layer of its own is still a range.
        assert_eq!(span(&cfg(Some("cvec.gguf"), 7, 7), 8), Some((7, 7)));
        // Reversed bounds ask for nothing.
        assert_eq!(span(&cfg(Some("cvec.gguf"), 9, 3), 48), None);
        // A zero-layer model must not underflow the ceiling.
        assert_eq!(span(&uc, 0), None);
    }

    #[test]
    fn resolve_is_off_without_a_file() {
        assert!(resolve(&Config::default(), true, 4, 48).unwrap().is_none());
    }

    #[test]
    fn resolve_refuses_a_model_it_cannot_project_before_reading_the_file() {
        // The path does not exist on purpose: the architecture check must come first, so the error
        // is "wrong model" rather than "no such file".
        let c = cfg(Some("definitely-not-here.gguf"), 4, 44);
        let e = resolve(&c, false, 4, 48).unwrap_err().to_string();
        assert!(
            e.contains(QWEN4EXP) && e.contains(config::ENV_VECTOR),
            "{e}"
        );
    }

    #[test]
    fn resolve_refuses_a_range_this_model_has_no_layer_for() {
        let c = cfg(Some("definitely-not-here.gguf"), 40, 44);
        let e = resolve(&c, true, 4, 12).unwrap_err().to_string();
        assert!(e.contains("covers no layer of this 12-layer model"), "{e}");
    }

    #[test]
    fn merge_takes_one_field_or_the_other_but_never_a_disagreement() {
        assert_eq!(api_merge(None, None).unwrap(), None);
        assert_eq!(api_merge(Some(false), None).unwrap(), Some(false));
        assert_eq!(api_merge(None, Some(false)).unwrap(), Some(false));
        assert_eq!(api_merge(Some(true), Some(true)).unwrap(), Some(true));
        assert_eq!(api_merge(Some(true), Some(false)), {
            Err(Conflict {
                explicit: true,
                alias: false,
            })
        });
        // The message names both spellings, since that is what the API client needs to see.
        let m = Conflict {
            explicit: false,
            alias: true,
        }
        .to_string();
        assert!(
            m.contains(API_FIELD) && m.contains(API_ALIAS) && m.contains("contradicts"),
            "{m}"
        );
    }

    #[test]
    fn a_request_with_no_opinion_follows_the_session_not_the_default() {
        // Nothing loaded: the feature is absent and the engine never calls this. Once it does
        // call it, the first pass turns "no opinion" into on, and every pass after that must keep
        // whatever the session is already running — including off.
        assert!(effective(None, None));
        assert!(effective(None, Some(true)));
        assert!(!effective(None, Some(false)));
        // An explicit answer always wins, in both directions.
        assert!(!effective(Some(false), Some(true)));
        assert!(effective(Some(true), Some(false)));
        assert!(!effective(Some(false), None));
    }

    #[test]
    fn scale_is_the_two_values_the_switch_has() {
        assert_eq!(scale(true), 1.0);
        assert_eq!(scale(false), 0.0);
    }

    #[test]
    fn flipping_is_measured_against_what_ran_last() {
        // The first pass of a session has run nothing, so there is no cache to invalidate yet —
        // and reporting it as a flip is harmless (the caller only resets state it has).
        assert!(flipped(None, true));
        assert!(flipped(None, false));
        assert!(!flipped(Some(true), true));
        assert!(!flipped(Some(false), false));
        assert!(flipped(Some(true), false));
        assert!(flipped(Some(false), true));
    }
}
