//! Native diagnostic for recurrent prefix reuse across Prefill shapes. Never runs by default.
use super::{DenseVulkanSession, SeamModel};
use crate::seam::{runner, weights::TurnRecurrentCkpt, PlacementScope, SeamKv, TurnCheckpoint};
use crate::EngineConfig;
use anyhow::anyhow;
use infr_core::backend::{Backend, Buffer};
use serde_json::json;

fn read(be: &dyn Backend, buffer: &dyn Buffer) -> Vec<u8> {
    let mut bytes = vec![0; buffer.len_bytes()];
    be.download(buffer, &mut bytes).unwrap();
    bytes
}

fn checkpoint(be: &dyn Backend, state: &SeamKv) -> Vec<Vec<u8>> {
    let ck = state.turn_recurrent_ckpts[0].as_ref().unwrap();
    assert!(ck.valid);
    ck.kbufs
        .iter()
        .chain(&ck.vbufs)
        .map(|b| read(be, b.as_ref()))
        .collect()
}

fn numeric_diff(left: &[Vec<u8>], right: &[Vec<u8>]) -> serde_json::Value {
    assert_eq!(left.len(), right.len());
    let mut total = 0usize;
    let mut changed = 0usize;
    let mut squared = 0f64;
    let mut max_abs = 0f32;
    let mut magnitude = 0f32;
    for (a, b) in left.iter().zip(right) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
            let a = f32::from_le_bytes(a.try_into().unwrap());
            let b = f32::from_le_bytes(b.try_into().unwrap());
            assert!(a.is_finite() && b.is_finite());
            total += 1;
            changed += usize::from(a.to_bits() != b.to_bits());
            let diff = (a - b).abs();
            max_abs = max_abs.max(diff);
            magnitude = magnitude.max(a.abs().max(b.abs()));
            squared += f64::from(diff).powi(2);
        }
    }
    json!({"floats": total, "changed": changed, "max_abs": max_abs,
        "rms": (squared / total.max(1) as f64).sqrt(), "max_magnitude": magnitude})
}

fn top(logits: &[f32]) -> [(usize, f32); 2] {
    assert!(logits.iter().all(|x| x.is_finite()));
    let mut values = [(usize::MAX, f32::NEG_INFINITY); 2];
    for (id, &value) in logits.iter().enumerate() {
        if value > values[0].1 {
            values[1] = values[0];
            values[0] = (id, value);
        } else if value > values[1].1 {
            values[1] = (id, value);
        }
    }
    values
}

fn forward(
    model: &SeamModel,
    session: &mut DenseVulkanSession,
    prompt: &[u32],
    max_new: usize,
    turn: Option<TurnCheckpoint>,
) -> Vec<f32> {
    let mut logits = Vec::new();
    let bind = crate::seam::BindWeight::new(|name: &str, _, _, _| {
        Err(anyhow!("diagnostic must not rebind {name}"))
    });
    runner::generate_dense_backend(
        &session.be,
        &bind,
        model.gguf(),
        model.config(),
        model.engine_cfg(),
        model.embd(),
        model.per_layer_embd(),
        prompt,
        max_new,
        |_| {},
        &mut session.pool.slots[0],
        session.max_ctx,
        None,
        None,
        None,
        Some(&mut logits),
        None,
        None,
        turn,
        None,
        None,
        None,
    )
    .unwrap();
    logits
}

#[test]
#[ignore = "requires MOE4ALL_PREFIX_MODEL and an explicitly selected idle GPU"]
fn recurrent_prefix_shape_and_restore_native() {
    let path = std::env::var_os("MOE4ALL_PREFIX_MODEL").expect("MOE4ALL_PREFIX_MODEL");
    let output = std::env::var_os("MOE4ALL_PREFIX_RESULT").expect("MOE4ALL_PREFIX_RESULT");
    let device: usize = std::env::var("MOE4ALL_PREFIX_DEVICE")
        .expect("explicit device")
        .parse()
        .unwrap();
    let ec = std::sync::Arc::new(EngineConfig::load_from_env());
    let model = SeamModel::load_with(std::path::Path::new(&path), None, ec).unwrap();
    assert!(
        model.config().qwen35,
        "this diagnostic isolates the reported 35B case"
    );
    let mut session = model.vulkan_session_on(Some(device), 48 * 1024).unwrap();
    if let Ok(expected) = std::env::var("MOE4ALL_PREFIX_EXPECT_DEVICE") {
        assert!(session.device_name().contains(&expected), "wrong test GPU");
    }
    let prompt = model
        .render_chat_messages(&[
            ("system", "You are a helpful assistant. Answer in English."),
            (
                "user",
                "Explain in three sentences why rain falls from clouds.",
            ),
        ])
        .unwrap();
    let tokens = model.encode(&prompt).unwrap();
    let boundary = 18;
    assert!(tokens.len() > boundary);
    println!("PREFIX_PROMPT tokens={} ids={tokens:?}", tokens.len());
    model
        .generate_vulkan_session(&mut session, &prompt, 1, None, |_| {})
        .unwrap();
    let _scope = PlacementScope::enter(session.pins().clone());
    session.pool.slots[0].as_mut().unwrap().reset();

    // Sequential single-row ingest, as in the short-Prefill scheduler. A sampled next token
    // is not materialized: each next call teacher-forces the actual next prompt token.
    for end in 1..=boundary {
        forward(&model, &mut session, &tokens[..end], 1, None);
    }
    {
        let kv = session.pool.slots[0].as_mut().unwrap();
        TurnRecurrentCkpt::begin(
            &mut kv.turn_recurrent_ckpts[0],
            &session.be,
            model.config(),
            &kv.kbufs,
            &kv.vbufs,
            kv.ple_state_buf.as_deref(),
            &tokens[..boundary],
        )
        .unwrap();
        kv.turn_recurrent_ckpts[0]
            .as_mut()
            .unwrap()
            .snapshot_all(
                &session.be,
                &kv.kbufs,
                &kv.vbufs,
                kv.ple_state_buf.as_deref(),
            )
            .unwrap();
    }
    let sequential = checkpoint(&session.be, session.pool.slots[0].as_ref().unwrap());
    let mut cold_logits = Vec::new();
    let mut cold_tokens = Vec::new();
    let mut feed = tokens[..boundary].to_vec();
    for &token in &tokens[boundary..] {
        feed.push(token);
        let logits = forward(&model, &mut session, &feed, 1, None);
        if feed.len() == tokens.len() {
            cold_logits.push(logits);
        }
    }
    for _ in 1..64 {
        let id = top(cold_logits.last().unwrap())[0].0 as u32;
        cold_tokens.push(id);
        feed.push(id);
        cold_logits.push(forward(&model, &mut session, &feed, 1, None));
    }
    cold_tokens.push(top(cold_logits.last().unwrap())[0].0 as u32);

    // Capture the same 18 tokens from the beginning of an unrelated long Prefill.
    session.pool.slots[0].as_mut().unwrap().reset();
    let mut long = tokens[..boundary].to_vec();
    long.extend(
        model
            .encode(&"A record contains a key and its associated value.\n".repeat(64))
            .unwrap(),
    );
    forward(
        &model,
        &mut session,
        &long,
        0,
        Some(TurnCheckpoint::new(Some(boundary), None)),
    );
    let batched = checkpoint(&session.be, session.pool.slots[0].as_ref().unwrap());
    let shape_diff = numeric_diff(&sequential, &batched);
    println!("PREFIX_SHAPE {shape_diff}");
    {
        let kv = session.pool.slots[0].as_mut().unwrap();
        assert_eq!(
            kv.restore_turn_recurrent(&session.be, &tokens).unwrap(),
            Some(boundary)
        );
        let ck = kv.turn_recurrent_ckpts[0].as_ref().unwrap();
        for (i, &layer) in ck.layers.iter().enumerate() {
            assert_eq!(read(&session.be, kv.kbufs[layer].as_ref()), batched[i]);
            assert_eq!(
                read(&session.be, kv.vbufs[layer].as_ref()),
                batched[ck.layers.len() + i]
            );
        }
        assert_eq!(kv.cached_tokens(), &tokens[..boundary]);
    }
    let mut feed = tokens[..boundary].to_vec();
    let mut warm = Vec::new();
    for &token in &tokens[boundary..] {
        feed.push(token);
        let logits = forward(&model, &mut session, &feed, 1, None);
        if feed.len() == tokens.len() {
            warm.push(logits);
        }
    }
    // Keep the same generated tokens in both arms, so a changed argmax cannot itself
    // change later inputs and disguise the location or scale of the original difference.
    for &token in &cold_tokens[..63] {
        feed.push(token);
        warm.push(forward(&model, &mut session, &feed, 1, None));
    }
    // Negative control: stop the identical 18-row batch immediately at its frontier,
    // without a long suffix and without restoration. Its continuation must equal the
    // restored one, ruling out unrelated suffix state or checkpoint-copy corruption.
    session.pool.slots[0].as_mut().unwrap().reset();
    forward(
        &model,
        &mut session,
        &tokens[..boundary + 1],
        0,
        Some(TurnCheckpoint::new(Some(boundary), None)),
    );
    assert_eq!(
        checkpoint(&session.be, session.pool.slots[0].as_ref().unwrap()),
        batched
    );
    let mut feed = tokens[..boundary].to_vec();
    for &token in &tokens[boundary..tokens.len() - 1] {
        feed.push(token);
        forward(&model, &mut session, &feed, 1, None);
    }
    for (step, token) in std::iter::once(&tokens[tokens.len() - 1])
        .chain(cold_tokens[..63].iter())
        .enumerate()
    {
        feed.push(*token);
        let direct = forward(&model, &mut session, &feed, 1, None);
        assert_eq!(
            direct, warm[step],
            "restored vs direct batch continuation at {step}"
        );
    }
    // Show the unconstrained warm continuation too, rather than inferring it from a
    // single argmax difference under teacher forcing.
    assert_eq!(
        session.pool.slots[0]
            .as_mut()
            .unwrap()
            .restore_turn_recurrent(&session.be, &tokens)
            .unwrap(),
        Some(boundary)
    );
    let mut feed = tokens[..boundary].to_vec();
    let mut logits = Vec::new();
    for &token in &tokens[boundary..] {
        feed.push(token);
        logits = forward(&model, &mut session, &feed, 1, None);
    }
    let mut warm_tokens = Vec::new();
    for _ in 0..64 {
        let id = top(&logits)[0].0 as u32;
        warm_tokens.push(id);
        if id == model.config().eos || model.config().eos_ids.contains(&id) {
            break;
        }
        feed.push(id);
        logits = forward(&model, &mut session, &feed, 1, None);
    }
    let layers = &session.pool.slots[0].as_ref().unwrap().turn_recurrent_ckpts[0]
        .as_ref()
        .unwrap()
        .layers;
    let state_diffs: Vec<_> = sequential.iter().zip(&batched).enumerate().map(|(i,(a,b))| {
        json!({"layer":layers[i % layers.len()],"kind":if i < layers.len() {"conv"} else {"state"},
            "diff":numeric_diff(std::slice::from_ref(a),std::slice::from_ref(b))})
    }).collect();
    let rows: Vec<_> = cold_logits
        .iter()
        .zip(&warm)
        .enumerate()
        .map(|(step, (a, b))| {
            let at = top(a);
            let bt = top(b);
            let max_abs = a
                .iter()
                .zip(b)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            json!({"step":step,"cold_top2":at,"warm_top2":bt,
            "cold_token":model.decode(&[at[0].0 as u32]).unwrap(),
            "warm_token":model.decode(&[bt[0].0 as u32]).unwrap(),
            "same_argmax":at[0].0==bt[0].0,"cold_margin":at[0].1-at[1].1,
            "warm_margin":bt[0].1-bt[1].1,"max_abs":max_abs})
        })
        .collect();
    let result = json!({"model":path.to_string_lossy(),"device":session.device_name(),
        "prompt_tokens":tokens.len(),"prefix_tokens":boundary,
        "restore_bit_exact":true,"restored_vs_direct_batch_logits_exact":true,
        "unrelated_suffix_does_not_change_checkpoint":true,
        "checkpoint_shape_diff":shape_diff,"state_diffs":state_diffs,
        "cold_text":model.decode(&cold_tokens).unwrap(),
        "warm_text":model.decode(&warm_tokens).unwrap(),"logits":rows});
    std::fs::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    println!("PREFIX_RESULT restored_vs_direct_batch_logits_exact=true");
}
