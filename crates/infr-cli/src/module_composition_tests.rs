//! Startup policy tests: no model files, backend initialization, network or GPU required.

use super::*;
use infr_core::{config::AutoProfile, SizeSpec};

fn serve_args(modules: u8) -> Vec<&'static str> {
    let mut args = vec!["infr", "serve"];
    if modules & 1 != 0 {
        args.push("chat.gguf");
    }
    if modules & 2 != 0 {
        args.extend(["--embedding-model", "embed.gguf"]);
    }
    if modules & 4 != 0 {
        args.extend(["--transcription-model", "audio.gguf"]);
    }
    args
}

fn command(args: &[&str]) -> Cmd {
    Cli::try_parse_from(args).unwrap().cmd.unwrap()
}

#[test]
fn all_seven_nonempty_module_combinations_parse_and_validate() {
    for modules in 1..8 {
        let cmd = command(&serve_args(modules));
        assert_eq!(uses_llm(&cmd), modules & 1 != 0);
        let Cmd::Serve {
            model,
            embedding_model,
            transcription,
            embedding_idle_timeout,
            mmproj,
            embedding_runner,
            ..
        } = cmd
        else {
            panic!("expected serve");
        };
        assert_eq!(model.is_some(), modules & 1 != 0);
        assert_eq!(embedding_model.is_some(), modules & 2 != 0);
        assert_eq!(
            transcription.transcription_model.is_some(),
            modules & 4 != 0
        );
        assert_eq!(embedding_idle_timeout, 300);
        assert_eq!(transcription.transcription_idle_timeout, 300);
        validate_serve_modules(
            model.as_deref(),
            embedding_model.as_deref(),
            &transcription,
            mmproj.as_deref(),
            embedding_runner.as_deref(),
        )
        .unwrap();
    }
}

#[test]
fn empty_combinations_and_unattached_projectors_or_runners_are_rejected() {
    for args in [
        vec!["infr", "serve"],
        vec!["infr", "serve", "--embedding-idle-timeout", "0"],
        vec!["infr", "serve", "--mmproj", "vision.gguf"],
        vec![
            "infr",
            "serve",
            "--embedding-model",
            "embed.gguf",
            "--mmproj",
            "vision.gguf",
        ],
        vec![
            "infr",
            "serve",
            "--transcription-model",
            "audio.gguf",
            "--mmproj",
            "vision.gguf",
        ],
        vec![
            "infr",
            "serve",
            "chat.gguf",
            "--embedding-runner",
            "llama-server",
        ],
    ] {
        assert!(Cli::try_parse_from(&args).is_err(), "accepted {args:?}");
    }
    // Programmatic callers have the same checks before model resolution or backend creation.
    let Cmd::Serve { transcription, .. } = command(&serve_args(2)) else {
        unreachable!();
    };
    assert!(validate_serve_modules(None, None, &transcription, None, None).is_err());
    assert!(validate_serve_modules(
        None,
        Some("embed.gguf"),
        &transcription,
        Some(Path::new("vision.gguf")),
        None
    )
    .is_err());
    assert!(validate_serve_modules(
        Some("chat.gguf"),
        None,
        &transcription,
        None,
        Some(Path::new("runner"))
    )
    .is_err());
    for modules in [1, 3, 5, 7] {
        let mut args = serve_args(modules);
        args.extend(["--mmproj", "vision.gguf"]);
        assert!(Cli::try_parse_from(args).is_ok());
    }
}

#[test]
fn auxiliary_commands_never_freeze_ram_or_rewrite_user_policy_and_caps() {
    let mut commands: Vec<Cmd> = [2, 4, 6]
        .into_iter()
        .map(|modules| command(&serve_args(modules)))
        .collect();
    commands.extend([
        command(&["infr", "serve-embedding", "embed.gguf"]),
        command(&["infr", "transcribe", "audio.gguf", "--file", "input.wav"]),
        command(&["infr", "devices"]),
        command(&["infr", "resources"]),
    ]);
    for cmd in commands {
        for profile in [AutoProfile::Conservative, AutoProfile::Aggressive] {
            for ram_budget in [
                None,
                Some(SizeSpec::Bytes(0)),
                Some(SizeSpec::Bytes(512 << 20)),
                Some(SizeSpec::Percent(0.25)),
            ] {
                let mut cfg = Config::default();
                cfg.device.auto_profile = profile;
                cfg.device.ram_budget = ram_budget;
                cfg.device.vram_budget = Some(SizeSpec::Bytes(256 << 20));
                cfg.device.vram_reserve = Some(SizeSpec::Percent(0.1));
                cfg.device.dev = Some("Vulkan1".into());
                let original = format!("{cfg:?}");
                assert!(!uses_llm(&cmd));
                assert!(freeze_llm_ram_budget(&cmd, &mut cfg).is_none());
                assert_eq!(
                    format!("{cfg:?}"),
                    original,
                    "aux startup changed user config"
                );
            }
        }
    }
}

#[test]
fn llm_paths_keep_budget_freezing_and_explicit_caps() {
    for modules in [1, 3, 5, 7] {
        let cmd = command(&serve_args(modules));
        assert!(uses_llm(&cmd));
        let mut cfg = Config::default();
        cfg.device.ram_budget = Some(SizeSpec::Bytes(512 << 20));
        cfg.device.vram_budget = Some(SizeSpec::Bytes(256 << 20));
        let original = format!("{cfg:?}");
        assert!(freeze_llm_ram_budget(&cmd, &mut cfg).is_none());
        assert_eq!(format!("{cfg:?}"), original);
    }
    assert!(uses_llm(&command(&["infr", "run", "chat.gguf"])));
    assert!(uses_llm(&command(&["infr", "bench", "chat.gguf"])));
    assert!(uses_llm(&command(&["infr", "multi", "chat.gguf@Vulkan1"])));
}

#[test]
fn chat_and_embedding_follow_one_device_flag_while_audio_stays_cpu() {
    for modules in 1..8 {
        for (dev, expected) in [
            ("cpu", Backend::Cpu),
            ("Vulkan1", Backend::Vulkan(Some("Vulkan1".into()))),
        ] {
            let mut args = serve_args(modules);
            args.extend(["--dev", dev]);
            let cmd = command(&args);
            let flags = cli_flag_layer(&cmd).unwrap();
            // An explicit device wins over the inherited/config device for both chat and embedding.
            let mut inherited = PartialConfig::default();
            inherited.device.dev = Some(Some("Vulkan0".into()));
            let cfg = Config::load_from_layers(&[inherited, flags]);
            assert_eq!(cfg.device.dev.as_deref(), Some(dev));
            assert_eq!(selected_backend(&cfg).unwrap(), expected);
            let Cmd::Serve { transcription, .. } = cmd else {
                unreachable!();
            };
            assert_eq!(transcription.transcription_dev, "cpu");
            assert_eq!(transcription.transcription_threads, 4);
        }
    }
}

#[test]
fn legacy_embedding_command_accepts_default_zero_and_custom_idle_deadlines() {
    for timeout in [None, Some("0"), Some("45")] {
        let mut args = vec!["infr", "serve-embedding", "embed.gguf", "--dev", "Vulkan1"];
        if let Some(timeout) = timeout {
            args.extend(["--embedding-idle-timeout", timeout]);
        }
        let cmd = command(&args);
        let cfg = Config::load_from_layers(&[cli_flag_layer(&cmd).unwrap()]);
        assert_eq!(
            selected_backend(&cfg).unwrap(),
            Backend::Vulkan(Some("Vulkan1".into()))
        );
        let Cmd::ServeEmbedding {
            embedding_idle_timeout,
            ..
        } = cmd
        else {
            unreachable!();
        };
        assert_eq!(
            embedding_idle_timeout,
            timeout.map(|s| s.parse::<u64>().unwrap()).unwrap_or(300)
        );
    }
    let mut args = serve_args(6);
    args.extend(["--embedding-idle-timeout", "0"]);
    assert!(matches!(
        command(&args),
        Cmd::Serve {
            embedding_idle_timeout: 0,
            ..
        }
    ));
}
