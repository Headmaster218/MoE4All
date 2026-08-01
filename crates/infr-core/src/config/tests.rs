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
use super::{
    cli, env, file, Config, ConfigError, ConfigOverrides, MetalDeviceTime, PartialConfig,
    SetPathError,
};

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
    // §10.7: `Sampler::from_cfg`'s doc contract — unset ⇒ greedy, so goldens stay deterministic.
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
    // RM: the Metal `MTLBinaryArchive` pipeline cache is ON by default, for the same reason — a
    // launch should not re-run the driver's AIR → ISA back end for every kernel. Env-key-less too.
    assert!(d.kernels.metal.pipeline_cache);
    assert_eq!(d.serve.max_tokens_cap, 131_072);
    assert_eq!(d.serve.api_key, None);
    // A per-request wall-clock deadline is OFF unless the operator asks for one: it truncates a
    // legitimate slow reply, so it cannot be a shipped default (§6.9).
    assert_eq!(d.serve.request_timeout_secs, 0);
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

/// The `Mib` and `Size` STRING spellings, at the layer that owns them.
///
/// These cases arrived here from `budget.rs`'s `mib_grammar` and `pager.rs`'s
/// `ring_bytes_clamp_boundaries_and_override_grammar`, which used to reach them through `&str`
/// façades (`budget::mib_from`, `pager::ring_bytes_from`) that existed only so those sweeps could
/// stay off the process environment. The façades are gone — the value-taking functions were always
/// environment-free — and the string half of the grammar is the env layer's job, so the spellings
/// are pinned where the parsing actually happens.
///
/// `banana` is not repeated here: [`bad_value_is_an_error_not_a_silent_default`] already sweeps it
/// across every value-carrying knob. What is here is the spellings that fail for a DIFFERENT reason
/// than "not a number at all" — a float and a negative for a `u64` MiB count, the empty value, and
/// a size suffix (`GiB`) that reads like the grammar but is not it. Each must leave the field
/// unspecified rather than half-parse.
#[test]
fn mib_and_size_string_spellings() {
    let d = Config::default();
    // Mib: trimmed, then `u64` — so a float, a negative and an empty value are all "not specified".
    for raw in ["", "1.5", "-1"] {
        let cfg = Config::load_from_layers(&[env_layer(&[("INFR_KV_OVERFLOW_VRAM_MB", raw)])]);
        assert_eq!(
            cfg.kv.overflow_vram_mb, d.kv.overflow_vram_mb,
            "INFR_KV_OVERFLOW_VRAM_MB={raw:?} must not specify a cap"
        );
    }
    // …and the accepted spelling, whitespace and all, reaches the field in MiB.
    let cfg = Config::load_from_layers(&[env_layer(&[("INFR_KV_OVERFLOW_VRAM_MB", "  512  ")])]);
    assert_eq!(cfg.kv.overflow_vram_mb, Some(512));

    // Size: `1GiB` is NOT the shared grammar's spelling (`1g` is) — it must read as unset, not as
    // a `1`-byte ring.
    for raw in ["", "1GiB"] {
        let cfg = Config::load_from_layers(&[env_layer(&[("INFR_PAGER_RING", raw)])]);
        assert_eq!(
            cfg.paging.ring, d.paging.ring,
            "INFR_PAGER_RING={raw:?} must not specify a ring size"
        );
    }
    let cfg = Config::load_from_layers(&[env_layer(&[("INFR_PAGER_RING", "1g")])]);
    assert_eq!(cfg.paging.ring, Some(super::SizeSpec::Bytes(1 << 30)));
}

/// The device-list grammar, inherited from `infr_llama::seam::parse_device_spec` when S4 deleted
/// that duplicate — these are its `seam_helper_tests` cases, moved to the surviving copy.
#[test]
fn device_spec_accepts_vulkan_and_bare_indices() {
    assert_eq!(
        super::parse_device_spec("Vulkan0,Vulkan1", 2).unwrap(),
        vec![0, 1]
    );
    // Bare indices and whitespace/empty segments are tolerated.
    assert_eq!(
        super::parse_device_spec(" 0 , 1 ,2, ", 1).unwrap(),
        vec![0, 1, 2]
    );
    // `min: 0` (the `--set`/TOML path) accepts an empty list — the count bound is the consumer's.
    assert_eq!(
        super::parse_device_spec("", 0).unwrap(),
        Vec::<usize>::new()
    );
}

#[test]
fn device_spec_rejects_garbage_and_too_few() {
    // Non-numeric / non-VulkanN part.
    assert!(super::parse_device_spec("Vulkan0,foo", 2).is_err());
    // Fewer than `min` devices.
    assert!(super::parse_device_spec("Vulkan0", 2).is_err());
    assert!(super::parse_device_spec("", 1).is_err());
    // Exactly `min` passes.
    assert!(super::parse_device_spec("Vulkan3", 1).is_ok());
}

/// The three device lists land on their fields, with the minimums the manifest records.
#[test]
fn device_lists_reach_their_fields() {
    let cfg = Config::load_from_layers(&[env_layer(&[
        ("INFR_PIPELINE", "Vulkan0,Vulkan1"),
        ("INFR_TENSOR_PARALLEL", "0,1"),
        ("INFR_EXPERT_PARALLEL", "2"),
    ])]);
    assert_eq!(cfg.multi.pipeline.as_deref(), Some(&[0usize, 1][..]));
    assert_eq!(cfg.multi.tensor_parallel.as_deref(), Some(&[0usize, 1][..]));
    assert_eq!(cfg.multi.expert_parallel.as_deref(), Some(&[2usize][..]));
    // Absent ⇒ single-device, exactly what the deleted `parse_device_list` returned as `None`.
    let none = Config::default();
    assert_eq!(none.multi.pipeline, None);
    assert!(none.multi.pipeline_p2p && none.multi.tp_p2p && none.multi.ep_p2p);
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
    // `INFR_REQUEST_TIMEOUT_SECS=0` is a VALUE ("no deadline"), not a bad one: it must survive the
    // layer so it can turn a file-configured deadline back off. Its sibling `INFR_MAX_TOKENS_CAP=0`
    // is the opposite — non-positive there is nonsense and is dropped back to the default.
    for (value, want) in [("0", 0u64), ("300", 300)] {
        let cfg = Config::load_from_layers(&[env_layer(&[("INFR_REQUEST_TIMEOUT_SECS", value)])]);
        assert_eq!(
            cfg.serve.request_timeout_secs, want,
            "INFR_REQUEST_TIMEOUT_SECS={value:?}"
        );
    }
    assert_eq!(
        Config::load_from_layers(&[
            file_layer("[serve]\nrequest_timeout_secs = 120\n"),
            env_layer(&[("INFR_REQUEST_TIMEOUT_SECS", "0")]),
        ])
        .serve
        .request_timeout_secs,
        0,
        "an explicit 0 in the environment must disarm a deadline set by the file"
    );
    assert_eq!(
        Config::load_from_layers(&[env_layer(&[("INFR_MAX_TOKENS_CAP", "0")])])
            .serve
            .max_tokens_cap,
        131_072
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

/// **The R3 exit criterion, as a test so it cannot rot** (`docs/config-plan.md` §7 S8).
///
/// After S8 the process environment is read for an `INFR_*` knob in exactly ONE place — the config
/// crate's env layer, through the injected reader `Config::load` passes it. Anything else is a
/// regression: a knob that has crept back onto ambient state, invisible to the config file, to
/// `--set`, and to any test that wants to drive it as a value.
///
/// This is the shell grep `env::var(_os)?("INFR_` over `crates/*/src` + `crates/*/build.rs`, in
/// Rust, minus comments. The allowed hits are exactly §6.10's permanent exclusions:
///
/// * `INFR_PROFILE` in the five `build.rs` — a BUILD-time input; a runtime `Config` cannot exist
///   when it is read (§5.3).
/// * `INFR_TEST_GGUF` / `INFR_TEST_MODEL` / `INFR_LLAMA_DIFFUSION_CLI` — test/dev fixtures that
///   point at files on disk, deliberately left on the environment.
///
/// `RAYON_NUM_THREADS` is not an `INFR_*` knob and is not in scope: `infr-cli` still PUBLISHES it,
/// because rayon's global pool has no other input.
#[test]
fn no_infr_env_reads_outside_the_config_layer() {
    /// Keys any file may still read directly (§6.10).
    const FIXTURE_KEYS: &[&str] = &[
        "INFR_TEST_GGUF",
        "INFR_TEST_MODEL",
        "INFR_LLAMA_DIFFUSION_CLI",
    ];

    let Some(crates) = repo_crates_dir() else {
        return; // packaged/vendored build — no `crates/` to scan
    };
    let mut offenders: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&crates).expect("read crates/") {
        let dir = entry.expect("crates/ entry").path();
        scan_for_env_reads(&dir.join("src"), &mut offenders);
        scan_file_for_env_reads(&dir.join("build.rs"), &mut offenders);
    }
    offenders.retain(|hit| {
        let (path, key) = hit.rsplit_once(' ').expect("hit is `path KEY`");
        if FIXTURE_KEYS.contains(&key) {
            return false;
        }
        // `INFR_PROFILE` is the build scripts' alone.
        !(key == "INFR_PROFILE" && path.ends_with("build.rs"))
    });
    assert!(
        offenders.is_empty(),
        "these sites read an INFR_* knob straight from the process environment — route it through \
         `config/env.rs` and take the value off the `Config` the caller already owns (R3): \
         {offenders:#?}"
    );
}

fn scan_for_env_reads(dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_for_env_reads(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            scan_file_for_env_reads(&path, out);
        }
    }
}

/// Every `std::env` var/var_os read of an `INFR_*` literal in `path` that is not inside a
/// comment, reported as `"<path> <KEY>"`. Comment lines are skipped so the campaign's own prose
/// (which names the reads it replaced) does not fail the check — a real read is never
/// comment-only.
fn scan_file_for_env_reads(path: &Path, out: &mut Vec<String>) {
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    for line in text.lines() {
        let code = line.trim_start();
        if code.starts_with("//") {
            continue;
        }
        for needle in ["env::var(\"INFR_", "env::var_os(\"INFR_"] {
            let mut from = 0;
            while let Some(rel) = code[from..].find(needle) {
                let start = from + rel + needle.len() - "INFR_".len();
                let end = code[start..]
                    .find('"')
                    .map(|n| start + n)
                    .unwrap_or(code.len());
                out.push(format!("{} {}", path.display(), &code[start..end]));
                from = end.max(from + 1);
            }
        }
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
    let layer = file_layer("[prof]\nops = true\n\n[debug]\npoison_uninit = true\n");
    let line = super::announce_file_diagnostics(&layer, Path::new("/etc/infr/config.toml"))
        .expect("a file that enables diagnostics must announce it");
    assert!(line.contains("/etc/infr/config.toml"), "{line}");
    assert!(line.contains("prof.ops"), "{line}");
    assert!(line.contains("debug.poison_uninit"), "{line}");

    // A file that touches nothing diagnostic says nothing.
    let quiet = file_layer("[kv]\nslots = 2\n");
    assert_eq!(
        super::announce_file_diagnostics(&quiet, Path::new("infr.toml")),
        None
    );
    // Neither does one that sets a diagnostic knob to its DEFAULT.
    let noop = file_layer("[prof]\nops = false\n");
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

/// `prof.ops` is THE per-op profiling switch, on every backend, and it is reachable three ways.
///
/// Each backend used to have its own: `INFR_PROF2` reached only vulkan, `INFR_PROF_OPS` only the
/// cpu backend, and `INFR_METAL_PROFILE` only metal — so whether
/// `INFR_PROF_OPS=1 infr bench …` profiled anything depended on `--dev`, silently. Any future
/// backend gates on this field (through `infr_core::prof::enabled`, which also ANDs warmup
/// suppression), never on a knob of its own.
#[test]
fn prof_ops_is_the_one_knob_reachable_by_env_set_and_file() {
    let d = Config::default();
    assert!(!d.prof.ops, "diagnostic — off by default");

    // All three layers reach the same field.
    assert!(
        Config::load_from_layers(&[env_layer(&[("INFR_PROF_OPS", "1")])])
            .prof
            .ops
    );
    assert!(
        Config::load_from_layers(&[cli_layer(&["prof.ops=true"])])
            .prof
            .ops
    );
    assert!(
        Config::load_from_layers(&[file_layer("[prof]\nops = true\n")])
            .prof
            .ops
    );
}

/// **Every** `prof.*` knob is reachable by `--set` (and therefore by the TOML file), with the
/// documented spelling. The profiling knobs are the ones people reach for from a shell mid-debug,
/// so a `prof.*` path that only the environment can set is a gap, not a detail.
#[test]
fn every_prof_knob_is_reachable_by_set() {
    let cfg = Config::load_from_layers(&[cli_layer(&[
        "prof.ops=true",
        "prof.op_shapes=true",
        "prof.stages=true",
        "prof.vram=true",
        "prof.diffusion_trace=true",
        "prof.out=/tmp/p.json",
        "prof.metal_device_time=counters",
        "prof.metal_debug=true",
    ])]);
    assert!(cfg.prof.ops);
    assert!(cfg.prof.op_shapes);
    assert!(cfg.prof.stages);
    assert!(cfg.prof.vram);
    assert!(cfg.prof.diffusion_trace);
    assert_eq!(cfg.prof.out.as_deref(), Some(Path::new("/tmp/p.json")));
    assert_eq!(cfg.prof.metal_device_time, MetalDeviceTime::Counters);
    assert!(cfg.prof.metal_debug);
}

/// Metal's device-timing mode is NAMED, not an integer level.
///
/// Its predecessor `INFR_METAL_PROFILE` compared against the exact string literals `"2"` and
/// `"3"`, so `=20` silently meant "profiling on, level unrecognized" rather than "≥ 2" — a footgun
/// the docs had to carry a paragraph about. Named modes remove the class: a value either parses to
/// a mode or does not.
#[test]
fn metal_device_time_is_named_modes_not_a_level() {
    assert_eq!(
        Config::default().prof.metal_device_time,
        MetalDeviceTime::Off
    );

    for (value, want) in [
        ("off", MetalDeviceTime::Off),
        ("flush", MetalDeviceTime::Flush),
        ("counters", MetalDeviceTime::Counters),
        ("COUNTERS", MetalDeviceTime::Counters),
    ] {
        let cfg = Config::load_from_layers(&[env_layer(&[("INFR_PROF_METAL_DEVICE_TIME", value)])]);
        assert_eq!(cfg.prof.metal_device_time, want, "{value:?}");
    }

    // The old grammar's trap: a numeric-looking value is not a level. It does not parse to a mode,
    // so it leaves the default rather than turning something half-on.
    for junk in ["2", "3", "20", "banana"] {
        let cfg = Config::load_from_layers(&[env_layer(&[("INFR_PROF_METAL_DEVICE_TIME", junk)])]);
        assert_eq!(
            cfg.prof.metal_device_time,
            MetalDeviceTime::Off,
            "{junk:?} must not select a mode"
        );
    }

    // `--set` is strict where the env layer is lenient (the file/`--set` rule for known keys).
    let mut p = PartialConfig::default();
    assert!(p.set_path("prof.metal_device_time", "2").is_err());
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
    ];

    /// S4 — `infr-llama`'s seam: the whole `kv`, `spec` and `multi` sections, `device.ubatch*`,
    /// `paging.cache` (read ONLY in the seam's placement
    /// binders), the two graph-shape gates, and the seam's own `prof.*` diagnostics.
    ///
    /// NOT here, deliberately: `sampling.*` — the SEAM reads it from `Config` now
    /// (`Sampler::from_cfg`), but `infr-cli`'s model-recommendation bridge
    /// (`set_default_sampling_env` + the DG bench arm) still reads `INFR_TEMP`/`INFR_TOP_K`/
    /// `INFR_TOP_P`/`INFR_SEED`/`INFR_IGNORE_EOS`/`INFR_MAX_NEW` from the environment until S8
    /// deletes it, and `INFR_NO_THINK` is `infr-chat`'s until S7. Likewise `prof.prof`
    /// (`infr-vulkan`'s recorder still reads it) and `kernels.vulkan.{delta_strided, no_replay,
    /// gpu_pos}` (§6.12's two-crate knobs — the llama half moved, S5 takes the Vulkan half).
    const S4: &[&str] = &[
        "INFR_CACHE",
        "INFR_DECODE_CHAIN",
        "INFR_PROF_STAGES",
        "INFR_PROF_DIFFUSION_TRACE",
        "INFR_EP_HOST",
        "INFR_EXPERT_PARALLEL",
        "INFR_KV_Q8",
        "INFR_KV_SLOTS",
        "INFR_KV_TYPE_K",
        "INFR_KV_TYPE_V",
        "INFR_MTP",
        "INFR_NO_GATED_RMSNORM",
        "INFR_NO_GPU_ARGMAX",
        "INFR_NO_GPU_DRAFT_PROB",
        "INFR_NO_GPU_EMBED",
        "INFR_NO_GPU_MTP_ACCEPT",
        "INFR_NO_GPU_SAMPLE",
        "INFR_NO_KV_RING",
        "INFR_NO_MTP_CKPT",
        "INFR_NO_MTP_DRAFT_CHAIN",
        "INFR_NO_MTP_REPRIME",
        "INFR_NO_QKV_FUSE",
        "INFR_PIPELINE",
        "INFR_PIPELINE_HOST",
        "INFR_SPEC_DEBUG",
        "INFR_SPEC_DRAFT",
        "INFR_SPEC_K",
        "INFR_TENSOR_PARALLEL",
        "INFR_TP_HOST",
        "INFR_UBATCH",
        "INFR_UBATCH_PARALLEL",
    ];

    /// S5a — `infr-vulkan`'s CONSTRUCTION-time tier: everything `VulkanBackend::new_with` resolves
    /// once, at device selection / capability probing / allocator setup. The six §5.2 capability
    /// maskers fold into the PROBE (they are NOT `Capabilities` fields and nothing downstream
    /// re-reads them), `device.subgroup_pref` + `device.submit_dispatches` are the last two of the
    /// five loud keys, and `device.dev` reaches `pick_default_device` as a parameter.
    ///
    /// NOT here, deliberately: `paging.stats` (`INFR_PAGER_STATS`) — the Vulkan pager reads it
    /// from `Config` now, and the rest of the pager plumbing landed in S6; and the
    /// whole recorder/adapter/gemm hot-path tier, which is S5b.
    const S5A: &[&str] = &[
        "INFR_CM_8X8",
        "INFR_DEBUG_COOPMAT",
        "INFR_DEV",
        "INFR_NO_COOPMAT",
        "INFR_NO_F16",
        "INFR_NO_I8DOT",
        "INFR_NO_PIPELINE_CACHE",
        "INFR_NO_PUSH_DESC",
        "INFR_NO_VRAM_GUARD",
        "INFR_POISON_UNINIT",
        "INFR_SG",
        "INFR_SUBMIT_DISPATCHES",
        "INFR_PROF_VRAM",
    ];

    /// S5b — `infr-vulkan`'s PER-OP / PER-DISPATCH tier: everything `recorder.rs`, `adapter.rs` and
    /// `gemm.rs` used to re-read inside a lowering or dispatch path. `Recorder` borrows
    /// `&VulkanCfg` off the backend that created it (R6), and the two MEMOIZED families are hoisted
    /// rather than de-memoized into per-call `getenv`s (§10.6): the eleven `kernels.vulkan.gemv.*`
    /// keys (was `OnceLock<GemvKnobs>`) come off the borrowed config, and
    /// `kernels.vulkan.bda_chunk_{elems,bytes}` (was two `AtomicU64` cells) become two `Recorder`
    /// fields resolved at construction.
    ///
    /// This also closes the three §6.12 two-crate knobs (`delta_strided`, `no_replay`, `gpu_pos` —
    /// the `infr-llama` halves moved in S4) and `prof.prof`, which S4 had to leave behind.
    ///
    /// NOT here, deliberately: `paging.stats` (`INFR_PAGER_STATS`) — the pager still
    /// reads the environment until S6; `prof.profile_out` (`infr-prof-rt`) and `debug.chat`
    /// (`infr-chat`) are S7.
    const S5B: &[&str] = &[
        "INFR_BDA_CHUNK_BYTES",
        "INFR_BDA_CHUNK_ELEMS",
        "INFR_BF16_COOPMAT",
        "INFR_DEBUG_BDA_CHUNK",
        "INFR_DEBUG_WIDE_DISPATCH",
        "INFR_DELTA_STRIDED",
        "INFR_DN_CHUNK_SCAN",
        "INFR_F8_COOPMAT",
        "INFR_F8_PREPACK",
        "INFR_FLASH_BM",
        "INFR_FLASH_DEQUANT",
        "INFR_FLASH_MIN_ROWS",
        "INFR_FLASH_SPLITS",
        "INFR_FLASH_STAGE",
        "INFR_FULLBARRIER",
        "INFR_GEMM_DIRECT_A",
        "INFR_GEMM_WIDE_TILE",
        "INFR_GEMV_RM",
        "INFR_GEMV_RM_MAXOUT",
        "INFR_GEMV_RM_MINOUT",
        "INFR_GEMV_SG_MAXOUT",
        "INFR_GEMV_SG_MINOUT",
        "INFR_GEMV_SG_NR",
        "INFR_GEMV_VARIANT",
        "INFR_I8_COOPMAT",
        "INFR_I8_ROW_SCALE",
        "INFR_KV_COOPMAT_BDA",
        "INFR_KV_INLINE",
        "INFR_MMV_DECODE",
        "INFR_MMV_FUSE_QUANT",
        "INFR_MMV_MW",
        "INFR_MMV_MW_WARPS",
        "INFR_MROWS_ATTN",
        "INFR_NOBARRIER",
        "INFR_NO_ATTN_HD",
        "INFR_NO_BM16",
        "INFR_NO_DN_CHUNK",
        "INFR_NO_DN_SPLIT",
        "INFR_NO_F32_MROW",
        "INFR_NO_F32_V4",
        "INFR_NO_FLASH_WARP",
        "INFR_NO_GEMM_WARP",
        "INFR_NO_GEMV_ID_SG",
        "INFR_NO_GEMV_REG",
        "INFR_NO_GEMV_RM",
        "INFR_NO_GEMV_SG",
        "INFR_NO_GPU_POS",
        "INFR_NO_MMQ",
        "INFR_NO_MMQ_FALLBACK",
        "INFR_NO_MMV",
        "INFR_NO_MMV_M4",
        "INFR_NO_MMV_O4",
        "INFR_NO_MOE_SM_POOL",
        "INFR_NO_MROW",
        "INFR_NO_MROW16",
        "INFR_NO_MROWS_ATTN",
        "INFR_NO_NC_FA",
        "INFR_NO_PV_WARP",
        "INFR_NO_QK_WARP",
        "INFR_NO_SMALL_BM",
        "INFR_PROF_OPS",
        "INFR_PROF_OP_SHAPES",
        "INFR_PV_SPLITS",
        "INFR_SEAM_NO_REPLAY",
    ];

    /// S6 — `infr-metal` (20). The backend is now `INFR_*`-free.
    ///
    /// Metal: `MetalBackend::new_with(cfg)`, and `exec.rs` reads `self.metal()` (a borrow of the
    /// backend's `Config`) at each selector. Fifteen of the twenty are `INFR_METAL_NO_*`
    /// `presence-inv` kill-switches; `INFR_METAL_NODELTA`/`INFR_METAL_NOMOE` are §6.12's
    /// read-both-ways pair collapsed onto ONE positive field each; `INFR_METAL_PROFILE` is the
    /// three-derived-boolean literal grammar (`is_ok()` / `== "2"` / `== "3"`), NOT an int level.
    ///
    /// `INFR_PAGER_STATS` is here at last: S5a moved the Vulkan pager.
    const S6: &[&str] = &[
        "INFR_METAL_LMHEAD_MRV",
        "INFR_METAL_NODELTA",
        "INFR_METAL_NOMOE",
        "INFR_METAL_NO_BF16_CMM",
        "INFR_METAL_NO_BF16_NATIVE",
        "INFR_METAL_NO_BF16_RT",
        "INFR_METAL_NO_CONV1D_PAR",
        "INFR_METAL_NO_DN_GATE_PREP",
        "INFR_METAL_NO_DN_NORM_PREP",
        "INFR_METAL_NO_F16_CMM",
        "INFR_METAL_NO_F16_NATIVE",
        "INFR_METAL_NO_F16_RT",
        "INFR_METAL_NO_F32_CMM",
        "INFR_METAL_NO_F32_NATIVE",
        "INFR_METAL_NO_F32_RT",
        "INFR_METAL_NO_KQUANT_NATIVE",
        "INFR_METAL_NO_Q5K_RT",
        "INFR_METAL_NO_RMSNORM_VEC4",
        "INFR_PROF_METAL_DEVICE_TIME",
        "INFR_PROF_METAL_DEBUG",
        "INFR_PAGER_STATS",
    ];

    /// S7 — the last four crates. `device.ctx` moved to the session-backed chats (`chat/mod.rs`'s
    /// `cfg_ctx`/`cfg_ctx_spec` + the diffusion chat's own copy); `sampling.no_think` and
    /// `debug.chat` to `infr-chat`'s renderer, which now takes the `&Config` its callers own;
    /// `prof.profile_out` is PUSHED into `infr-prof-rt` at startup (its report runs from a C
    /// `atexit` hook with nothing to borrow); `serve.*` moved onto `AppState`.
    ///
    /// NOT here: `INFR_DIFFUSION_VISUAL`, which §6.10 sends to a plain clap flag rather than to
    /// `Config` — it stays in `manifest::NOT_MIGRATED`.
    ///
    /// `INFR_REQUEST_TIMEOUT_SECS` was never migrated — it was BORN on `Config`, after the
    /// campaign, and its only read site is `infr-server`'s `AppState`. A knob with no legacy
    /// `std::env::var` site is still `migrated` (the flag means "the read site takes the value
    /// from a `Config`", which is trivially true), so it has to be listed in some slice or this
    /// test fails. It joins the slice that owns its section — `serve.*` is S7's — rather than
    /// growing a ninth, empty-by-construction one for post-campaign additions.
    const S7: &[&str] = &[
        "INFR_API_KEY",
        "INFR_CTX",
        "INFR_DEBUG_CHAT",
        "INFR_MAX_TOKENS_CAP",
        "INFR_NO_THINK",
        "INFR_PROF_OUT",
        "INFR_REQUEST_TIMEOUT_SECS",
    ];

    /// S8 — the CLI's own transitional bridge, and with it the LAST unmigrated knob. The six
    /// `sampling.*` keys had migrated READ sites since S4; what kept them `pending` was
    /// `infr-cli`, which re-published the resolved values into the process environment
    /// (`publish_transitional_env`) so that `apply_model_sampling_defaults` could probe
    /// `std::env::var(..).is_ok()` to decide whether a layer had specified them. S8 threads the
    /// `specified` `PartialConfig` down from `main()` instead, and deletes the publication.
    ///
    /// With this list, EVERY key in `KEYS` is migrated — which is what
    /// `no_infr_env_reads_outside_the_config_layer` enforces against the tree.
    const S8: &[&str] = &[
        "INFR_IGNORE_EOS",
        "INFR_MAX_NEW",
        "INFR_SEED",
        "INFR_TEMP",
        "INFR_TOP_K",
        "INFR_TOP_P",
    ];

    let mut got: Vec<&str> = KEYS.iter().filter(|k| k.migrated).map(|k| k.env).collect();
    got.sort_unstable();
    let mut want: Vec<&str> = S2
        .iter()
        .chain(S3)
        .chain(S4)
        .chain(S5A)
        .chain(S5B)
        .chain(S6)
        .chain(S7)
        .chain(S8)
        .copied()
        .collect();
    want.sort_unstable();
    assert_eq!(
        got, want,
        "the manifest's `migrated` set drifted from the slices that landed — flip an entry and \
         list it here in the SAME commit"
    );
}
