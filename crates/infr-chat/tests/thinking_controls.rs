//! Model-template regression tests, without weights or a GPU.
use infr_chat::{render_template, render_template_with_options, ChatTemplateOptions};
use infr_core::config::{Config, ReasoningEffort};
use serde_json::{json, Value};

const QWEN38: &str = include_str!("fixtures/qwen38_chat_template.jinja");

#[test]
fn official_qwen38_template_cases() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("fixtures/thinking_cases.json")).unwrap();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let options: ChatTemplateOptions = serde_json::from_value(case["options"].clone()).unwrap();
        let result = render_template_with_options(
            QWEN38,
            case["messages"].as_array().unwrap().clone(),
            case.get("tools").cloned().unwrap_or(Value::Null),
            "",
            "",
            case["generation"].as_bool().unwrap_or(true),
            &options,
        );
        if let Some(error) = case["error"].as_str() {
            assert!(result.unwrap_err().to_string().contains(error), "{name}");
            continue;
        }
        let out = result.unwrap_or_else(|e| panic!("{name}: {e}"));
        for marker in case["contains"].as_array().unwrap() {
            assert!(
                out.contains(marker.as_str().unwrap()),
                "{name}: missing {marker}: {out:?}"
            );
        }
        for marker in case["excludes"].as_array().unwrap() {
            assert!(
                !out.contains(marker.as_str().unwrap()),
                "{name}: unexpected {marker}: {out:?}"
            );
        }
        if let Some(suffix) = case["suffix"].as_str() {
            assert!(out.ends_with(suffix), "{name}: {out:?}");
        }
    }
}

#[test]
fn default_options_preserve_the_existing_prompt() {
    let msgs = vec![json!({"role":"user", "content":"hello"})];
    let old = render_template(QWEN38, msgs.clone(), Value::Null, "", "", true, true).unwrap();
    let new = render_template_with_options(
        QWEN38,
        msgs,
        Value::Null,
        "",
        "",
        true,
        &ChatTemplateOptions::default().resolve(&Config::default()),
    )
    .unwrap();
    assert_eq!(old, new);
}

#[test]
fn explicit_effort_is_rejected_by_non_effort_templates() {
    // A comment mentioning the option must not be mistaken for a supported input variable.
    let template = "{# reasoning_effort #}{{ messages[0].content }}";
    let result = render_template_with_options(
        template,
        vec![json!({"content":"hello"})],
        Value::Null,
        "",
        "",
        true,
        &ChatTemplateOptions {
            reasoning_effort: Some(ReasoningEffort::Medium),
            ..Default::default()
        },
    );
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("does not support reasoning_effort"));
    assert_eq!(
        render_template(
            template,
            vec![json!({"content":"hello"})],
            Value::Null,
            "",
            "",
            true,
            true,
        )
        .unwrap(),
        "hello"
    );
}

#[test]
fn unsupported_preserve_thinking_is_rejected() {
    let err = render_template_with_options(
        "hello",
        vec![],
        Value::Null,
        "",
        "",
        true,
        &ChatTemplateOptions {
            preserve_thinking: Some(false),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("does not support preserve_thinking"));
}

#[test]
fn model_defaults_and_macro_scoped_effort_are_preserved() {
    // A different template family with a medium default and a macro-local assignment (GPT-OSS
    // shape). This verifies the renderer, not model weight support.
    let template = "{% macro system() %}{% if reasoning_effort is not defined %}{% set reasoning_effort = 'medium' %}{% endif %}Reasoning: {{ reasoning_effort }}{% endmacro %}{{ system() }}";
    for (effort, expected) in [(None, "medium"), (Some(ReasoningEffort::High), "high")] {
        let out = render_template_with_options(
            template,
            vec![],
            Value::Null,
            "",
            "",
            true,
            &ChatTemplateOptions {
                reasoning_effort: effort,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(out, format!("Reasoning: {expected}"));
    }
}

#[test]
fn request_options_override_defaults_without_changing_shared_config() {
    let mut cfg = Config::default();
    cfg.sampling.no_think = true;
    cfg.sampling.reasoning_effort = Some(ReasoningEffort::Xhigh);
    cfg.sampling.preserve_thinking = Some(true);
    let request = ChatTemplateOptions {
        enable_thinking: Some(true),
        reasoning_effort: Some(ReasoningEffort::Low),
        preserve_thinking: Some(false),
    };
    assert_eq!(request.resolve(&cfg), request);
    let inherited = ChatTemplateOptions::default().resolve(&cfg);
    assert_eq!(inherited.enable_thinking, Some(false));
    assert_eq!(inherited.reasoning_effort, Some(ReasoningEffort::Xhigh));
    assert_eq!(inherited.preserve_thinking, Some(true));
}

#[test]
fn concurrent_cached_template_renders_keep_request_efforts_separate() {
    let threads: Vec<_> = [
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::Xhigh,
    ]
    .into_iter()
    .map(|effort| {
        std::thread::spawn(move || {
            let options = ChatTemplateOptions {
                reasoning_effort: Some(effort),
                ..Default::default()
            };
            for _ in 0..20 {
                let out = render_template_with_options(
                    QWEN38,
                    vec![json!({"role":"user","content":"hello"})],
                    Value::Null,
                    "",
                    "",
                    true,
                    &options,
                )
                .unwrap();
                assert_eq!(out.contains("set to low"), effort == ReasoningEffort::Low);
                assert_eq!(
                    out.contains("set to xhigh"),
                    effort == ReasoningEffort::Xhigh
                );
            }
        })
    })
    .collect();
    for thread in threads {
        thread.join().unwrap();
    }
}

#[test]
fn generation_and_stable_history_use_the_same_effort() {
    let mut history = Vec::new();
    for effort in [
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::Xhigh,
    ] {
        let options = ChatTemplateOptions {
            reasoning_effort: Some(effort),
            ..Default::default()
        };
        let render = |generation| {
            render_template_with_options(
                QWEN38,
                vec![json!({"role":"user","content":"hello"})],
                Value::Null,
                "",
                "",
                generation,
                &options,
            )
            .unwrap()
        };
        let stable = render(false);
        assert_eq!(
            render(true),
            format!("{stable}<|im_start|>assistant\n<think>\n")
        );
        history.push(stable);
    }
    assert_ne!(history[0], history[1]);
    assert_ne!(history[1], history[2]);
    assert_ne!(history[0], history[2]);
}

#[test]
fn qwen38_system_and_tools_render_as_a_branchable_prompt_prefix() {
    let system = json!({"role":"system", "content":"Follow the agent policy."});
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    }]);
    let options = ChatTemplateOptions::default();
    let render_probe = |content: &str| {
        render_template_with_options(
            QWEN38,
            vec![system.clone(), json!({"role":"user", "content":content})],
            tools.clone(),
            "",
            "",
            false,
            &options,
        )
        .unwrap()
    };
    let left = render_probe("A_INFR_REUSABLE_PREFIX_PROBE");
    let right = render_probe("Z_INFR_REUSABLE_PREFIX_PROBE");
    let prefix = left
        .chars()
        .zip(right.chars())
        .take_while(|(left, right)| left == right)
        .map(|(character, _)| character)
        .collect::<String>();
    let prompt = render_template_with_options(
        QWEN38,
        vec![system, json!({"role":"user", "content":"Do the task."})],
        tools,
        "",
        "",
        true,
        &options,
    )
    .unwrap();
    assert!(prompt.starts_with(&prefix));
    assert!(prefix.contains("# Tools"));
    assert!(prefix.ends_with("<|im_start|>user\n"));
}

#[test]
fn qwen38_latest_user_edit_keeps_the_complete_prior_history_prefix() {
    let options = ChatTemplateOptions::default();
    let history = vec![
        json!({"role":"system", "content":"Follow the policy."}),
        json!({"role":"user", "content":"First question"}),
        json!({"role":"assistant", "content":"First answer"}),
    ];
    let render_probe = |content: &str| {
        let mut messages = history.clone();
        messages.push(json!({"role":"user", "content":content}));
        render_template_with_options(QWEN38, messages, Value::Null, "", "", true, &options).unwrap()
    };
    let left = render_probe("A_INFR_EDIT_PREFIX_PROBE");
    let right = render_probe("Z_INFR_EDIT_PREFIX_PROBE");
    let prefix = left
        .chars()
        .zip(right.chars())
        .take_while(|(left, right)| left == right)
        .map(|(character, _)| character)
        .collect::<String>();
    let edited = render_probe("Completely different final question");

    assert!(edited.starts_with(&prefix));
    assert!(prefix.contains("First question"));
    assert!(prefix.contains("First answer"));
    assert!(prefix.ends_with("<|im_start|>user\n"));
}
