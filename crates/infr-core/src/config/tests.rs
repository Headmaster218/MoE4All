//! Acceptance criteria for the config scaffold — `docs/config-plan.md` §8.
//!
//! Every test here is PURE: the env layer is driven through an injected `HashMap` reader and the
//! file layer through a string literal, so nothing touches the process environment or the
//! filesystem. That is the property the whole campaign exists to buy (§8.8) — these tests run in
//! parallel with each other and with everything else in the binary, with no lock and no restore
//! guard.

use std::collections::HashMap;
use std::path::Path;

use super::manifest::{BadValue, Grammar, KEYS, NOT_KNOBS, NOT_MIGRATED};
use super::{cli, env, file, Config, ConfigError, ConfigOverrides, PartialConfig, SetPathError};

/// Drive the env layer with a synthetic environment. `recorder.rs`'s
/// `gemv_knobs_resolve_matches_env_reads` is the working precedent for this shape.
fn env_layer(pairs: &[(&str, &str)]) -> PartialConfig {
    try_env_layer(pairs).expect("env layer rejected a valid synthetic environment")
}

fn try_env_layer(pairs: &[(&str, &str)]) -> Result<PartialConfig, ConfigError> {
    let map: HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    env::parse(&|k| map.get(k).cloned())
}

fn file_layer(text: &str) -> PartialConfig {
    file::parse_str(text, Path::new("infr.toml"))
        .expect("file layer rejected a valid document")
        .0
}

fn cli_layer(sets: &[&str]) -> PartialConfig {
    cli::parse_reporting(&ConfigOverrides {
        sets: sets.iter().map(|s| (*s).to_string()).collect(),
        ..Default::default()
    })
    .expect("cli layer rejected a valid --set")
    .0
}

// ── §8.1 ─────────────────────────────────────────────────────────────────────

/// Every default is the shipped behaviour, and an empty stack of layers changes nothing.
///
/// The plan words this as "every field's `Default` equals the value in `manifest.rs`"; the
/// manifest records grammar and destination, not defaults (a default lives in exactly one place —
/// `impl Default` — precisely so it cannot disagree with a second copy). What is checkable, and
/// what this pins, is that (a) folding nothing is `Config::default()`, (b) folding three EMPTY
/// layers is `Config::default()`, and (c) the defaults §6/§10 call out by name are what they say.
#[test]
fn default_config_matches_documented_defaults() {
    assert_eq!(Config::default(), Config::load_from_layers(&[]));
    assert_eq!(
        Config::default(),
        Config::load_from_layers(&[file_layer(""), env_layer(&[]), cli_layer(&[])]),
        "an empty file + empty environment + no flags must resolve to the shipped defaults"
    );

    let d = Config::default();
    // §10.7: `Sampler::from_env`'s doc contract — unset ⇒ greedy, so goldens stay deterministic.
    assert_eq!(d.sampling.temp, 0.0);
    assert_eq!(d.sampling.top_k, 20);
    assert_eq!(d.sampling.top_p, 0.95);
    assert_eq!(d.sampling.seed, None);
    assert_eq!(d.sampling.max_new, 2048);
    // §6.1: `Option` means "the user pinned it"; the 1024 / iGPU-adaptive chain stays at its site.
    assert_eq!(d.device.ubatch, None);
    assert_eq!(d.device.ubatch_parallel, 256);
    assert_eq!(d.device.submit_dispatches, None);
    assert_eq!(d.device.subgroup_pref, None);
    // §6.3 / §6.4.
    assert_eq!(d.kv.slots, 4);
    assert!(d.kv.ring);
    assert!(!d.kv.force_q8);
    assert_eq!(d.paging.rocm_prefetch_slots, None);
    assert_eq!(d.paging.rocm_prefetch_max_bank_mb, None);
    // §6.5 / §10.2: the mmv tier is ON by default — `INFR_NO_MMV` is presence-INV.
    assert!(d.kernels.vulkan.mmv);
    assert!(!d.kernels.vulkan.mmv_decode);
    assert_eq!(d.kernels.vulkan.mmv_mw, None);
    assert_eq!(d.kernels.vulkan.flash_min_rows, 24);
    assert_eq!(d.kernels.vulkan.moe_small_m, 8);
    assert_eq!(d.kernels.vulkan.canvas_chunk_n, 3);
    assert!(
        d.kernels.vulkan.dn_chunk_scan,
        "positively spelled, read is_err()"
    );
    // §6.5b: `variant` is COMPUTED and defaults to Some("reg"), not None.
    assert_eq!(d.kernels.vulkan.gemv.variant.as_deref(), Some("reg"));
    assert_eq!(d.kernels.vulkan.gemv.rm, 2);
    assert_eq!(d.kernels.vulkan.gemv.rm_maxout, usize::MAX);
    assert_eq!(d.kernels.vulkan.gemv.rm_minout, 2048);
    assert_eq!(d.kernels.vulkan.gemv.sg_minout, 2048);
    assert_eq!(d.kernels.vulkan.gemv.sg_maxout, 8192);
    assert_eq!(d.kernels.vulkan.gemv.sg_nr, 2);
    // §6.7 / §6.9.
    assert_eq!(d.kernels.cpu.spin, 1 << 15);
    assert!(d.kernels.cpu.spinpool);
    assert_eq!(d.kernels.cpu.repack_mb, 4096);
    assert!(!d.kernels.cpu.reference);
    assert_eq!(d.serve.max_tokens_cap, 131_072);
    assert_eq!(d.serve.api_key, None);
    // §6.8: `INFR_MTP` is the exact string "1"; unset ⇒ off, and the three MTP hatches default on.
    assert!(!d.spec.mtp);
    assert!(d.spec.mtp_ckpt && d.spec.mtp_reprime && d.spec.mtp_draft_chain);
    assert_eq!(d.spec.k, 6);
    assert_eq!(d.spec.decode_chain, 8);
}

// ── §8.2 / §8.3 / §8.4 — precedence ──────────────────────────────────────────

/// The environment beats the config file.
#[test]
fn env_overrides_file() {
    let cfg = Config::load_from_layers(&[
        file_layer("[kernels.vulkan]\nflash_splits = 2\n"),
        env_layer(&[("INFR_FLASH_SPLITS", "4")]),
    ]);
    assert_eq!(cfg.kernels.vulkan.flash_splits, Some(4));
}

/// A CLI flag beats the environment.
#[test]
fn cli_overrides_env() {
    let cfg = Config::load_from_layers(&[
        env_layer(&[("INFR_FLASH_SPLITS", "4")]),
        cli_layer(&["kernels.vulkan.flash_splits=8"]),
    ]);
    assert_eq!(cfg.kernels.vulkan.flash_splits, Some(8));
}

/// A layer only overrides what it actually specifies — a `None` never clobbers a `Some`.
#[test]
fn absent_layer_does_not_clobber() {
    let cfg = Config::load_from_layers(&[
        env_layer(&[("INFR_FLASH_SPLITS", "4"), ("INFR_NO_GEMM_WARP", "1")]),
        // The file mentions ONLY [kv]; it must not reset the Vulkan section.
        file_layer("[kv]\nslots = 9\n"),
    ]);
    assert_eq!(cfg.kv.slots, 9);
    assert_eq!(cfg.kernels.vulkan.flash_splits, Some(4));
    assert!(!cfg.kernels.vulkan.gemm_warp);
    // …and everything neither layer mentioned is still the default.
    assert_eq!(cfg.kernels.vulkan.flash_min_rows, 24);
    assert!(cfg.kernels.vulkan.mmv);
}

/// Within the CLI layer, a bespoke flag WINS over a `--set` for the same field, and says so.
#[test]
fn bespoke_flag_wins_over_set_and_warns() {
    let mut flags = PartialConfig::default();
    flags.device.ctx = Some(Some(crate::SizeSpec::Bytes(32768)));
    let (layer, warnings) = cli::parse_reporting(&ConfigOverrides {
        sets: vec!["device.ctx=8192".to_string()],
        flags,
        ..Default::default()
    })
    .unwrap();
    let cfg = Config::load_from_layers(&[layer]);
    assert_eq!(cfg.device.ctx, Some(crate::SizeSpec::Bytes(32768)));
    assert_eq!(warnings.len(), 1, "the shadowed --set must be named");
    assert!(warnings[0].contains("device.ctx"), "{}", warnings[0]);
}

/// Two `--set`s for the same path are an error, not a silent last-wins (§11 [DECIDE-3]).
#[test]
fn duplicate_set_is_an_error() {
    let err = cli::parse_reporting(&ConfigOverrides {
        sets: vec!["kv.slots=2".to_string(), "kv.slots=3".to_string()],
        ..Default::default()
    })
    .unwrap_err();
    assert!(matches!(err, ConfigError::DuplicateSet { .. }), "{err:?}");
}

// ── §8.5 — unknown keys ──────────────────────────────────────────────────────

/// An unknown TOML key WARNS and is ignored (§11 [DECIDE-5]): an older binary must be able to
/// read a file written for a newer one, and deleting a knob must not break anyone's config.
#[test]
fn unknown_toml_key_warns_and_is_ignored() {
    let (partial, warnings) = file::parse_str(
        "[kernels.vulkan]\nflash_splt = 2\nflash_splits = 3\n",
        Path::new("infr.toml"),
    )
    .expect("an unknown key must NOT fail the load");
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("unknown key"), "{}", warnings[0]);
    assert!(
        warnings[0].contains("kernels.vulkan.flash_splt"),
        "{}",
        warnings[0]
    );
    assert!(
        warnings[0].contains("flash_splits"),
        "the warning is the typo protection, so it must suggest: {}",
        warnings[0]
    );
    // The rest of the document still applies.
    let cfg = Config::load_from_layers(&[partial]);
    assert_eq!(cfg.kernels.vulkan.flash_splits, Some(3));
}

/// A whole unknown SECTION warns once, not once per key inside it.
#[test]
fn unknown_toml_section_warns_once() {
    let (_, warnings) =
        file::parse_str("[kernels.opencl]\na = 1\nb = 2\n", Path::new("infr.toml")).unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("kernels.opencl"), "{}", warnings[0]);
}

/// Invalid TOML syntax is still a hard error (§11 [DECIDE-5]).
#[test]
fn malformed_toml_is_an_error() {
    let err = file::parse_str("[kv\nslots = 1\n", Path::new("infr.toml")).unwrap_err();
    assert!(matches!(err, ConfigError::Toml { .. }), "{err:?}");
}

// ── §8.6 — bad values ────────────────────────────────────────────────────────

/// A bad value fails where it fails TODAY, and is swallowed where it is swallowed today (R1).
///
/// Two classes, both taken from `manifest::KnobKey::bad_value`:
/// - [`BadValue::Error`] — `INFR_SG`, `INFR_SUBMIT_DISPATCHES` and the three multi-GPU device
///   lists reject garbage LOUDLY today. The env layer must keep erroring, not `unwrap_or`.
/// - [`BadValue::Ignored`] — everything else does `.and_then(parse).ok().unwrap_or(default)`, so
///   garbage falls back to the default and the layer reports "not specified".
///
/// The FILE layer is stricter for everyone (§11 [DECIDE-5]: a value of the wrong type for a known
/// key is a hard error), which the `ctx = "banana"` case at the end pins.
#[test]
fn bad_value_is_an_error_not_a_silent_default() {
    for key in KEYS {
        // Only the value-carrying grammars can HAVE a bad value; a presence knob accepts anything.
        if !matches!(
            key.grammar,
            Grammar::Int | Grammar::Float | Grammar::Size | Grammar::Mib | Grammar::DeviceList
        ) {
            continue;
        }
        let got = try_env_layer(&[(key.env, "banana")]);
        match key.bad_value {
            BadValue::Error => assert!(
                got.is_err(),
                "{} rejects a bad value today and must keep doing so",
                key.env
            ),
            BadValue::Ignored => {
                let layer = got.unwrap_or_else(|e| panic!("{}: {e}", key.env));
                let cfg = Config::load_from_layers(&[layer]);
                assert_eq!(
                    cfg.get_path(key.path),
                    Config::default().get_path(key.path),
                    "{} swallows a bad value today (R1) — it must still land on the default",
                    key.env
                );
            }
        }
    }

    // `INFR_SG` is the string-literal member of the Error class.
    assert!(try_env_layer(&[("INFR_SG", "8")]).is_err());
    assert!(try_env_layer(&[("INFR_SG", "16")]).is_ok());
    // …and the device lists enforce their (DIFFERENT) minimum device counts.
    assert!(try_env_layer(&[("INFR_PIPELINE", "0")]).is_err(), "min 2");
    assert!(
        try_env_layer(&[("INFR_TENSOR_PARALLEL", "0")]).is_err(),
        "min 2"
    );
    assert!(
        try_env_layer(&[("INFR_EXPERT_PARALLEL", "0")]).is_ok(),
        "min 1"
    );

    // The file layer: a value that does not parse into a KNOWN key's type is a hard error.
    let err = file::parse_str("[device]\nctx = \"banana\"\n", Path::new("infr.toml")).unwrap_err();
    match err {
        ConfigError::Value { path, .. } => assert_eq!(path, "device.ctx"),
        other => panic!("expected a value error, got {other:?}"),
    }
}

// ── §8.7 — polarity ──────────────────────────────────────────────────────────

/// The `presence-inv` truth table, for EVERY inverted knob in the manifest.
///
/// `INFR_NO_FOO` unset ⇒ the field is `true` (feature ON); set to `""`, `"0"` or `"1"` ⇒ the field
/// is `false` (feature OFF). The `"0"` row is the one that catches a wrong-grammar migration:
/// only PRESENCE matters, so `INFR_NO_GEMM_WARP=0` still turns warp GEMM off (§7.0).
#[test]
fn presence_inverted_knobs_have_the_right_polarity() {
    let defaults = Config::default();
    for key in KEYS {
        let (unset_repr, set_repr) = match key.grammar {
            Grammar::PresenceInv => ("true", "false"),
            Grammar::PresenceOptFalse => ("none", "false"),
            _ => continue,
        };
        assert_eq!(
            defaults.get_path(key.path).as_deref(),
            Some(unset_repr),
            "{} unset must leave `{}` at {unset_repr}",
            key.env,
            key.path
        );
        for value in ["", "0", "1"] {
            let cfg = Config::load_from_layers(&[env_layer(&[(key.env, value)])]);
            assert_eq!(
                cfg.get_path(key.path).as_deref(),
                Some(set_repr),
                "{}={value:?} must set `{}` to {set_repr} — presence, not value",
                key.env,
                key.path
            );
        }
    }
}

/// The three grammars that are NOT plain presence, spelled out because getting any of them wrong
/// is invisible to the goldens (§10.1, §10.5).
#[test]
fn non_presence_grammars_are_preserved() {
    // `budget::flag_from`: empty and "0" are OFF here, unlike every `is_ok()` knob.
    for (value, want) in [("", false), ("0", false), ("1", true), ("yes", true)] {
        let cfg = Config::load_from_layers(&[env_layer(&[("INFR_KV_OVERFLOW", value)])]);
        assert_eq!(cfg.kv.overflow, want, "INFR_KV_OVERFLOW={value:?}");
    }
    // Set AND != "0".
    for (value, want) in [("", true), ("0", false), ("1", true)] {
        let cfg = Config::load_from_layers(&[env_layer(&[("INFR_NO_THINK", value)])]);
        assert_eq!(cfg.sampling.no_think, want, "INFR_NO_THINK={value:?}");
    }
    // …and its inverted twin.
    for (value, want) in [("", false), ("0", true), ("1", false)] {
        let cfg = Config::load_from_layers(&[env_layer(&[("INFR_CPU_NO_SPINPOOL", value)])]);
        assert_eq!(
            cfg.kernels.cpu.spinpool, want,
            "INFR_CPU_NO_SPINPOOL={value:?}"
        );
    }
    // Tri-state (§10.3): unset = vendor default, "0" = force off, anything else = force on.
    assert_eq!(Config::default().kernels.vulkan.mmv_mw, None);
    for (value, want) in [("0", Some(false)), ("1", Some(true)), ("", Some(true))] {
        let cfg = Config::load_from_layers(&[env_layer(&[("INFR_MMV_MW", value)])]);
        assert_eq!(cfg.kernels.vulkan.mmv_mw, want, "INFR_MMV_MW={value:?}");
    }
    // Exact string literals (§10.4): `INFR_FLASH_BM` is compared to "32", not parsed.
    assert!(
        Config::load_from_layers(&[env_layer(&[("INFR_FLASH_BM", "32")])])
            .kernels
            .vulkan
            .flash_bm32
    );
    assert!(
        !Config::load_from_layers(&[env_layer(&[("INFR_FLASH_BM", "64")])])
            .kernels
            .vulkan
            .flash_bm32
    );
    // `INFR_MTP=true` does nothing today — only "1" enables it.
    assert!(
        Config::load_from_layers(&[env_layer(&[("INFR_MTP", "1")])])
            .spec
            .mtp
    );
    assert!(
        !Config::load_from_layers(&[env_layer(&[("INFR_MTP", "true")])])
            .spec
            .mtp
    );
    // `INFR_API_KEY=` (empty) means NO auth — the opposite of the presence grammar.
    assert_eq!(
        Config::load_from_layers(&[env_layer(&[("INFR_API_KEY", "")])])
            .serve
            .api_key,
        None
    );
    assert_eq!(
        Config::load_from_layers(&[env_layer(&[("INFR_API_KEY", "k")])])
            .serve
            .api_key
            .as_deref(),
        Some("k")
    );
}

/// The asymmetric mrows-attn pair: `INFR_NO_MROWS_ATTN` WINS when both are set (§6.5).
#[test]
fn mrows_attn_pair_is_asymmetric() {
    let both = Config::load_from_layers(&[env_layer(&[
        ("INFR_NO_MROWS_ATTN", "1"),
        ("INFR_MROWS_ATTN", "1"),
    ])]);
    assert_eq!(both.kernels.vulkan.mrows_attn, Some(false));
    let opt_in = Config::load_from_layers(&[env_layer(&[("INFR_MROWS_ATTN", "1")])]);
    assert_eq!(opt_in.kernels.vulkan.mrows_attn, Some(true));
    assert_eq!(Config::default().kernels.vulkan.mrows_attn, None);
}

/// `INFR_NO_GEMV_REG` silently wins over `INFR_GEMV_VARIANT` — R1-frozen (§10.11).
#[test]
fn gemv_variant_is_computed_from_two_keys() {
    let cfg = Config::load_from_layers(&[env_layer(&[
        ("INFR_NO_GEMV_REG", "1"),
        ("INFR_GEMV_VARIANT", "rm"),
    ])]);
    assert_eq!(cfg.kernels.vulkan.gemv.variant, None);
    let cfg = Config::load_from_layers(&[env_layer(&[("INFR_GEMV_VARIANT", "rm")])]);
    assert_eq!(cfg.kernels.vulkan.gemv.variant.as_deref(), Some("rm"));
}

/// §11 [DECIDE-8]: an unparseable KV dtype still counts as SPECIFIED (so it keeps suppressing
/// auto-q8) while yielding no dtype (so the runner still falls through to f16).
#[test]
fn kv_dtype_presence_survives_an_unparseable_name() {
    let cfg = Config::load_from_layers(&[env_layer(&[("INFR_KV_TYPE_K", "nonsense")])]);
    assert_eq!(
        cfg.kv.type_k, None,
        "no dtype — the runner falls through to f16"
    );
    assert!(cfg.kv.type_k_specified, "but the knob WAS supplied");
    let cfg = Config::load_from_layers(&[env_layer(&[("INFR_KV_TYPE_K", "q8_0")])]);
    assert_eq!(cfg.kv.type_k, Some(crate::DType::Q8_0));
    assert!(cfg.kv.type_k_specified);
    assert!(!Config::default().kv.type_k_specified);
}

// ── §8.8 — every key is read ─────────────────────────────────────────────────

/// Setting ANY manifest key through the injected reader must change what the layer specifies.
///
/// This is the test that stops a knob being silently dropped during a migration: add a key to the
/// manifest and forget the `env.rs` line, and this fails. Never touches the real environment.
#[test]
fn env_layer_reads_every_key() {
    let empty = env_layer(&[]);
    assert!(empty.is_empty(), "an empty environment specifies nothing");
    for key in KEYS {
        let layer = env_layer(&[(key.env, key.sample)]);
        assert!(
            !layer.is_empty(),
            "{}={:?} was not read by the env layer at all",
            key.env,
            key.sample
        );
        assert_ne!(
            layer, empty,
            "{}={:?} left the layer unchanged — is it missing from env.rs?",
            key.env, key.sample
        );
        assert!(
            layer.is_path_set(key.path),
            "{}={:?} did not land on `{}`",
            key.env,
            key.sample,
            key.path
        );
    }
}

/// Every manifest path is a REAL config path, so `--set`, the TOML schema and the manifest cannot
/// drift apart. (The reverse does not hold: `device.threads` and `kernels.cpu.reference` are
/// fields with no env key, by design.)
#[test]
fn manifest_paths_are_real_config_paths() {
    let known = Config::all_paths();
    for key in KEYS {
        assert!(
            known.iter().any(|p| p == key.path),
            "manifest path `{}` ({}) is not a field on Config",
            key.path,
            key.env
        );
    }
    let mut envs: Vec<&str> = KEYS.iter().map(|k| k.env).collect();
    envs.sort_unstable();
    let before = envs.len();
    envs.dedup();
    assert_eq!(
        before,
        envs.len(),
        "an env var is listed twice in the manifest"
    );
}

// ── §8.9 — the manifest matches the tree ─────────────────────────────────────

/// Every `INFR_*` literal in `crates/*/src` and `crates/*/build.rs` is accounted for.
///
/// Re-runs the §6.0 derivation in Rust (no shell, so it works under `cargo test` anywhere the
/// repo is checked out) and cross-checks it against `KEYS` ∪ `NOT_MIGRATED` ∪ `NOT_KNOBS`.
/// Without this, the next feature branch silently re-introduces an ungoverned knob.
#[test]
fn manifest_matches_the_tree() {
    let Some(crates) = repo_crates_dir() else {
        // Packaged/vendored build: `crates/` is not next to us. Nothing to check.
        return;
    };
    let mut found: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&crates).expect("read crates/") {
        let dir = entry.expect("crates/ entry").path();
        scan_for_keys(&dir.join("src"), &mut found);
        scan_file_for_keys(&dir.join("build.rs"), &mut found);
    }
    found.sort_unstable();
    found.dedup();

    let known: Vec<&str> = KEYS
        .iter()
        .map(|k| k.env)
        .chain(NOT_MIGRATED.iter().map(|(k, _)| *k))
        .chain(NOT_KNOBS.iter().copied())
        .collect();

    let missing: Vec<&String> = found
        .iter()
        .filter(|k| !known.contains(&k.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "these INFR_* keys exist in the tree but not in config::manifest — add them to KEYS (or \
         to NOT_MIGRATED with a reason): {missing:?}"
    );

    let stale: Vec<&str> = KEYS
        .iter()
        .map(|k| k.env)
        .filter(|k| !found.iter().any(|f| f == k))
        .collect();
    assert!(
        stale.is_empty(),
        "these manifest keys no longer appear anywhere in the tree — drop them: {stale:?}"
    );
}

/// `<repo>/crates`, or `None` when we are not building from the repo.
fn repo_crates_dir() -> Option<std::path::PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")) // …/crates/infr-core
        .parent()?
        .to_path_buf();
    dir.is_dir().then_some(dir)
}

fn scan_for_keys(dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_for_keys(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            scan_file_for_keys(&path, out);
        }
    }
}

/// The §6.0 grep, in Rust: `"INFR_[A-Z_0-9]*"` string LITERALS (so a doc comment mentioning a knob
/// does not count), minus the `_TEST_` fixtures.
fn scan_file_for_keys(path: &Path, out: &mut Vec<String>) {
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    let bytes = text.as_bytes();
    let mut i = 0;
    while let Some(rel) = text[i..].find("\"INFR_") {
        let start = i + rel + 1;
        let mut end = start;
        while end < bytes.len()
            && (bytes[end].is_ascii_uppercase()
                || bytes[end].is_ascii_digit()
                || bytes[end] == b'_')
        {
            end += 1;
        }
        // `end > start + "INFR_".len()` drops the bare `"INFR_"` prefix literal (this file's own
        // scanner needle, and any `starts_with` guard), which is not a key.
        if bytes.get(end) == Some(&b'"') && end > start + 5 {
            let key = &text[start..end];
            if !key.contains("_TEST_") {
                out.push(key.to_string());
            }
        }
        i = end;
    }
}

// ── §8.10 — `--set` path validation ──────────────────────────────────────────

/// `--set` with a typo'd path is a HARD error with a did-you-mean, never a silent no-op: it was
/// typed for THIS run, so ignoring it would give a wrong result with no second chance to notice
/// (§11 [DECIDE-5]/[DECIDE-6]).
#[test]
fn dotted_path_setter_rejects_unknown_paths() {
    let err = cli::parse_reporting(&ConfigOverrides {
        sets: vec!["kernels.vulkan.flash_splt=2".to_string()],
        ..Default::default()
    })
    .unwrap_err();
    match err {
        ConfigError::UnknownPath { path, suggestion } => {
            assert_eq!(path, "kernels.vulkan.flash_splt");
            assert_eq!(suggestion.as_deref(), Some("kernels.vulkan.flash_splits"));
        }
        other => panic!("expected an unknown-path error, got {other:?}"),
    }

    // A KNOWN path with a bad value is a value error, not an unknown-path error.
    let err = cli::parse_reporting(&ConfigOverrides {
        sets: vec!["kv.slots=banana".to_string()],
        ..Default::default()
    })
    .unwrap_err();
    assert!(matches!(err, ConfigError::Value { .. }), "{err:?}");

    // A section is not a settable path.
    let mut p = PartialConfig::default();
    assert_eq!(
        p.set_path("kernels.vulkan", "1"),
        Err(SetPathError::Unknown)
    );

    // …and env NAMES are deliberately not accepted (§11 [DECIDE-6]): they are not 1:1 with fields.
    assert_eq!(
        p.set_path("INFR_NO_GEMM_WARP", "1"),
        Err(SetPathError::Unknown)
    );
}

/// `--set` reaches the knobs with no dedicated flag, across every value grammar.
#[test]
fn set_path_covers_every_value_grammar() {
    let cfg = Config::load_from_layers(&[cli_layer(&[
        "kernels.vulkan.gemm_warp=false",   // bool
        "kv.slots=7",                       // usize
        "sampling.temp=0.6",                // f32
        "device.ctx=32k",                   // SizeSpec
        "kv.type_k=q8_0",                   // DType
        "spec.draft=/tmp/draft.gguf",       // PathBuf
        "multi.pipeline=Vulkan0,Vulkan1",   // Vec<usize>
        "kernels.vulkan.gemv.variant=none", // Option<String> → cleared
        "serve.api_key=hunter2",            // Option<String> → set
    ])]);
    assert!(!cfg.kernels.vulkan.gemm_warp);
    assert_eq!(cfg.kv.slots, 7);
    assert!((cfg.sampling.temp - 0.6).abs() < 1e-6);
    assert_eq!(cfg.device.ctx, Some(crate::SizeSpec::Bytes(32 * 1024)));
    assert_eq!(cfg.kv.type_k, Some(crate::DType::Q8_0));
    assert_eq!(
        cfg.spec.draft.as_deref(),
        Some(Path::new("/tmp/draft.gguf"))
    );
    assert_eq!(cfg.multi.pipeline.as_deref(), Some(&[0usize, 1][..]));
    assert_eq!(cfg.kernels.vulkan.gemv.variant, None);
    assert_eq!(cfg.serve.api_key.as_deref(), Some("hunter2"));
}

// ── §11 [DECIDE-7] — the file layer announces diagnostics ────────────────────

/// A `prof.*` / `debug.*` knob turned on by the FILE is announced at startup, naming the file and
/// the fields — otherwise "why is my server printing timings" is unanswerable.
#[test]
fn file_set_diagnostics_are_announced() {
    let layer = file_layer("[prof]\nprof2 = true\n\n[debug]\npoison_uninit = true\n");
    let line = super::announce_file_diagnostics(&layer, Path::new("/etc/infr/config.toml"))
        .expect("a file that enables diagnostics must announce it");
    assert!(line.contains("/etc/infr/config.toml"), "{line}");
    assert!(line.contains("prof.prof2"), "{line}");
    assert!(line.contains("debug.poison_uninit"), "{line}");

    // A file that touches nothing diagnostic says nothing.
    let quiet = file_layer("[kv]\nslots = 2\n");
    assert_eq!(
        super::announce_file_diagnostics(&quiet, Path::new("infr.toml")),
        None
    );
    // Neither does one that sets a diagnostic knob to its DEFAULT.
    let noop = file_layer("[prof]\nprof2 = false\n");
    assert_eq!(
        super::announce_file_diagnostics(&noop, Path::new("infr.toml")),
        None
    );
}

// ── the TOML schema itself ───────────────────────────────────────────────────

/// The file speaks the POSITIVE field names, and the sections nest exactly like the structs (§4).
#[test]
fn toml_sections_mirror_the_struct_paths() {
    let cfg = Config::load_from_layers(&[file_layer(
        "[device]\n\
         dev = \"vulkan1\"\n\
         ctx = \"32k\"\n\
         \n\
         [kv]\n\
         type_k = \"q8_0\"\n\
         \n\
         [kernels.vulkan]\n\
         flash_splits = 2\n\
         gemm_warp = false\n\
         \n\
         [kernels.vulkan.gemv]\n\
         rm = 4\n\
         \n\
         [multi]\n\
         pipeline = [0, 1]\n",
    )]);
    assert_eq!(cfg.device.dev.as_deref(), Some("vulkan1"));
    assert_eq!(cfg.device.ctx, Some(crate::SizeSpec::Bytes(32 * 1024)));
    assert_eq!(cfg.kv.type_k, Some(crate::DType::Q8_0));
    assert_eq!(cfg.kernels.vulkan.flash_splits, Some(2));
    assert!(!cfg.kernels.vulkan.gemm_warp);
    assert_eq!(cfg.kernels.vulkan.gemv.rm, 4);
    assert_eq!(cfg.multi.pipeline.as_deref(), Some(&[0usize, 1][..]));
}

/// The migrated set, pinned key by key. A slice flips its own entries AND updates this list, so
/// "which knobs have actually moved" is one diff away and a stray flip cannot ride along
/// unnoticed. Every key below must come back clean from the three R3 greps
/// (`docs/config-plan.md` §3) — its read site takes the value from a `Config`.
#[test]
fn migrated_keys_are_exactly_the_landed_slices() {
    /// S2 — `infr-core`'s own knobs: the two `tier::EnvRows` tables, the six `budget` spill knobs,
    /// the pager ring, and the three `fusion` escape hatches.
    const S2: &[&str] = &[
        "INFR_CANVAS_CHUNK_N",
        "INFR_KV_OVERFLOW",
        "INFR_KV_OVERFLOW_RESERVE_MB",
        "INFR_KV_OVERFLOW_VRAM_MB",
        "INFR_MOE_SMALL_M",
        "INFR_NO_FUSE_ADD",
        "INFR_PAGER_RING",
        "INFR_ROCM_NO_FUSE_ADD",
        "INFR_ROCM_NO_FUSE_NORM",
        "INFR_ROCM_WEIGHT_OVERFLOW",
        "INFR_ROCM_WEIGHT_OVERFLOW_RESERVE_MB",
        "INFR_ROCM_WEIGHT_VRAM_MB",
    ];

    /// S3 — `infr-cpu`: the three `kernels.cpu` knobs plus the three diagnostics its interpreter
    /// gates. (`kernels.cpu.reference` is on the same struct but has no env key — it was never a
    /// variable, so it is not in `KEYS`.)
    const S3: &[&str] = &[
        "INFR_CPU_NO_SPINPOOL",
        "INFR_CPU_REPACK_MB",
        "INFR_CPU_SPIN",
        "INFR_MOE_COUNTS_DEBUG",
        "INFR_MOE_COUNTS_DUMP",
        "INFR_PROF_OPS",
    ];

    let mut got: Vec<&str> = KEYS.iter().filter(|k| k.migrated).map(|k| k.env).collect();
    got.sort_unstable();
    let mut want: Vec<&str> = S2.iter().chain(S3).copied().collect();
    want.sort_unstable();
    assert_eq!(
        got, want,
        "the manifest's `migrated` set drifted from the slices that landed — flip an entry and \
         list it here in the SAME commit"
    );
}
