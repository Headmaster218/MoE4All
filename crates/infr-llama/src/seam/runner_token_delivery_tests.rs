use super::{deliver_parallel_token_step, parallel_decode_should_stop};
use crate::sampling::{ParallelSampler, Qwen4ReasoningEosGuard, RequestCtx, RequestSampling};
use crate::seam::token_delivery::{
    deliver_token_batch, ParallelTokenCallback, TokenAcknowledgements, TokenDelivery,
};
use crate::Config;
use std::sync::atomic::{AtomicBool, Ordering};

const LANES: usize = infr_core::backend::MAX_DECODE_ROWS;

#[test]
fn caller_callback_outlives_request_local_output_borrows() {
    fn local_outputs(on_token: ParallelTokenCallback<'_>) -> crate::seam::ParallelSampledOutput {
        let mut peer_outputs = Vec::new();
        let mut prompt_secs = Vec::new();
        let mut decode_secs = Vec::new();
        let mut hidden = Some(vec![vec![0.0]]);
        {
            let mut request = super::ParallelDecodeRequest {
                prompts: &[],
                prompt_ends: &[],
                checkpoint_boundaries: &[],
                mrope_plans: &[],
                peers: &mut [],
                peer_outputs: &mut peer_outputs,
                prompt_secs: &mut prompt_secs,
                decode_secs: &mut decode_secs,
                samplers: &mut [],
                reasoning_guards: &mut [],
                on_token,
                yield_requested: None,
                hidden: hidden.as_deref_mut(),
            };
            let accepted = match &mut request.on_token {
                ParallelTokenCallback::Legacy(callback) => callback(0, 42),
                ParallelTokenCallback::Batch(callback) => callback(&[(0, 42)])[0],
            };
            assert!(accepted);
            request.peer_outputs.push(vec![42]);
            request.prompt_secs.push(1.0);
            request.decode_secs.push(2.0);
            request.hidden.as_deref_mut().unwrap()[0].push(3.0);
        }
        crate::seam::ParallelSampledOutput {
            tokens: peer_outputs,
            prompt_secs,
            decode_secs,
            hidden: hidden.unwrap(),
        }
    }

    let mut legacy_tokens = Vec::new();
    let mut legacy = |lane, token| {
        legacy_tokens.push((lane, token));
        true
    };
    let legacy_output = local_outputs(ParallelTokenCallback::Legacy(&mut legacy));
    assert!(legacy(0, 43));
    assert_eq!(legacy_tokens, [(0, 42), (0, 43)]);

    let mut batch_tokens = Vec::new();
    let mut batch = |tokens: &[(usize, u32)]| {
        batch_tokens.extend_from_slice(tokens);
        [true; LANES]
    };
    let batch_output = local_outputs(ParallelTokenCallback::Batch(&mut batch));
    assert!(batch(&[(0, 43)])[0]);
    assert_eq!(batch_tokens, [(0, 42), (0, 43)]);

    for output in [legacy_output, batch_output] {
        assert_eq!(output.tokens, [vec![42]]);
        assert_eq!(output.prompt_secs, [1.0]);
        assert_eq!(output.decode_secs, [2.0]);
        assert_eq!(output.hidden, [vec![0.0, 3.0]]);
    }
}

fn config() -> Config {
    Config {
        eos: 900,
        eos_ids: vec![901],
        qwen4exp: true,
        qwen4_think_start: Some(10),
        qwen4_think_end: Some(11),
        ..Config::default()
    }
}

#[test]
fn batch_and_legacy_preserve_sampled_suffix_eos_accounting_and_cancelled_output() {
    let cfg = config();
    for lanes in 1..=LANES {
        for sample_from in 0..=lanes {
            for eos_mask in 0..(1usize << (lanes - sample_from)) {
                for ignore_eos in [false, true] {
                    let next = (sample_from..lanes)
                        .map(|lane| {
                            if eos_mask & (1 << (lane - sample_from)) != 0 {
                                if lane % 2 == 0 {
                                    cfg.eos
                                } else {
                                    cfg.eos_ids[0]
                                }
                            } else {
                                100 + lane as u32
                            }
                        })
                        .collect::<Vec<_>>();
                    for batch in [false, true] {
                        let mut generated = vec![Vec::new(); lanes];
                        let mut curs = vec![vec![7]; lanes];
                        let mut guards = vec![Qwen4ReasoningEosGuard::default(); lanes];
                        let mut emitted = Vec::new();
                        let mut expected_stop = false;
                        let mut expected_emitted = Vec::new();
                        for (row, &token) in next.iter().enumerate() {
                            let lane = sample_from + row;
                            let eos =
                                !ignore_eos && (token == cfg.eos || cfg.eos_ids.contains(&token));
                            if !eos {
                                expected_emitted.push((lane, token));
                            }
                            expected_stop |= eos || lane % 3 == 1;
                        }
                        let should_stop = if batch {
                            let mut callback = |tokens: &[(usize, u32)]| {
                                emitted.extend_from_slice(tokens);
                                let mut accepted = [false; LANES];
                                for &(lane, _) in tokens {
                                    accepted[lane] = lane % 3 != 1;
                                }
                                accepted
                            };
                            deliver_parallel_token_step(
                                &cfg,
                                ignore_eos,
                                sample_from,
                                &next,
                                &mut generated,
                                &mut curs,
                                &mut guards,
                                &mut ParallelTokenCallback::Batch(&mut callback),
                                None,
                                None,
                            )
                        } else {
                            let mut callback = |lane, token| {
                                emitted.push((lane, token));
                                lane % 3 != 1
                            };
                            deliver_parallel_token_step(
                                &cfg,
                                ignore_eos,
                                sample_from,
                                &next,
                                &mut generated,
                                &mut curs,
                                &mut guards,
                                &mut ParallelTokenCallback::Legacy(&mut callback),
                                None,
                                None,
                            )
                        };
                        assert_eq!(should_stop, expected_stop);
                        assert_eq!(emitted, expected_emitted);
                        for lane in 0..lanes {
                            if lane < sample_from {
                                assert!(generated[lane].is_empty());
                                assert_eq!(curs[lane], [7]);
                            } else {
                                let token = next[lane - sample_from];
                                assert_eq!(generated[lane], [token]);
                                let continued =
                                    expected_emitted.contains(&(lane, token)) && lane % 3 != 1;
                                assert_eq!(
                                    curs[lane],
                                    if continued { vec![7, token] } else { vec![7] }
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn eos_only_or_prefill_only_step_never_calls_delivery() {
    let cfg = config();
    for batch in [false, true] {
        for eos_only in [false, true] {
            let mut generated = vec![Vec::new(); LANES];
            let mut curs = vec![vec![7]; LANES];
            let mut guards = vec![Qwen4ReasoningEosGuard::default(); LANES];
            let next = if eos_only {
                vec![cfg.eos; LANES]
            } else {
                Vec::new()
            };
            let sample_from = if eos_only { 0 } else { LANES };
            let mut batch_callback = |_: &[(usize, u32)]| -> TokenAcknowledgements {
                panic!("no visible token can require delivery or ACK")
            };
            let mut legacy_callback =
                |_, _| -> bool { panic!("no visible token can require delivery or ACK") };
            let mut callback = if batch {
                ParallelTokenCallback::Batch(&mut batch_callback)
            } else {
                ParallelTokenCallback::Legacy(&mut legacy_callback)
            };
            let should_stop = deliver_parallel_token_step(
                &cfg,
                false,
                sample_from,
                &next,
                &mut generated,
                &mut curs,
                &mut guards,
                &mut callback,
                None,
                None,
            );
            assert_eq!(should_stop, eos_only);
            assert!(curs.iter().all(|tokens| tokens == &[7]));
            assert!(generated
                .iter()
                .all(|tokens| tokens.len() == usize::from(eos_only)));
        }
    }
}

struct CancelDuringAck<'a> {
    req: &'a RequestCtx,
    yield_requested: &'a AtomicBool,
    stop_ack: bool,
    abort: bool,
    should_yield: bool,
    lanes: usize,
    sent: Vec<usize>,
    acked: Vec<usize>,
}

impl TokenDelivery for CancelDuringAck<'_> {
    fn send_token(&mut self, lane: usize, _: u32) -> bool {
        assert!(
            self.acked.is_empty(),
            "all sends must precede ACK collection"
        );
        self.sent.push(lane);
        true
    }

    fn receive_ack(&mut self, lane: usize) -> bool {
        assert_eq!(self.sent.len(), self.lanes);
        self.acked.push(lane);
        if lane == 0 {
            if self.abort {
                self.req.abort();
            }
            if self.should_yield {
                self.yield_requested.store(true, Ordering::Release);
            }
        }
        !(self.stop_ack && lane == 0)
    }
}

#[test]
fn cancellation_or_yield_drains_every_ack_and_blocks_next_forward_and_ple() {
    let cfg = config();
    for lanes in 1..=LANES {
        for (stop_ack, abort, should_yield) in [
            (true, false, false),
            (false, true, false),
            (false, false, true),
        ] {
            let req = RequestCtx::new(Default::default());
            let yield_requested = AtomicBool::new(false);
            let mut delivery = CancelDuringAck {
                req: &req,
                yield_requested: &yield_requested,
                stop_ack,
                abort,
                should_yield,
                lanes,
                sent: Vec::with_capacity(lanes),
                acked: Vec::new(),
            };
            let mut generated = vec![Vec::new(); lanes];
            let mut curs = vec![vec![7]; lanes];
            let mut guards = vec![Qwen4ReasoningEosGuard::default(); lanes];
            let next = vec![100; lanes];
            let mut forwards = 0;
            let mut ple_tickets = 0;
            for _ in 0..2 {
                if parallel_decode_should_stop(Some(&req), false, Some(&yield_requested)) {
                    break;
                }
                forwards += 1;
                let mut callback =
                    |tokens: &[(usize, u32)]| deliver_token_batch::<LANES>(tokens, &mut delivery);
                let should_stop = deliver_parallel_token_step(
                    &cfg,
                    false,
                    0,
                    &next,
                    &mut generated,
                    &mut curs,
                    &mut guards,
                    &mut ParallelTokenCallback::Batch(&mut callback),
                    Some(&req),
                    Some(&yield_requested),
                );
                assert_eq!(delivery.acked, (0..lanes).collect::<Vec<_>>());
                if should_stop {
                    break;
                }
                ple_tickets += 1;
            }
            assert_eq!(forwards, 1);
            assert_eq!(ple_tickets, 0);
            assert!(generated.iter().all(|tokens| tokens == &[100]));
            assert_eq!(curs[0], if stop_ack { vec![7] } else { vec![7, 100] });
        }
    }
}

#[test]
fn legacy_lookup_style_yield_preserves_callback_order_and_stops_at_boundary() {
    let cfg = config();
    for lanes in 1..=LANES {
        let requested = AtomicBool::new(false);
        let mut generated = vec![Vec::new(); lanes];
        let mut curs = vec![vec![7]; lanes];
        let mut guards = vec![Qwen4ReasoningEosGuard::default(); lanes];
        let next = vec![100; lanes];
        let mut observed = Vec::new();
        let mut callback = |lane, token| {
            observed.push((lane, token));
            requested.store(true, Ordering::Release);
            true
        };
        assert!(deliver_parallel_token_step(
            &cfg,
            false,
            0,
            &next,
            &mut generated,
            &mut curs,
            &mut guards,
            &mut ParallelTokenCallback::Legacy(&mut callback),
            None,
            Some(&requested),
        ));
        assert_eq!(
            observed,
            (0..lanes).map(|lane| (lane, 100)).collect::<Vec<_>>()
        );
        assert!(curs.iter().all(|tokens| tokens == &[7, 100]));
    }
}

#[test]
fn reasoning_guard_tracks_each_sample_even_when_its_ack_cancels() {
    let cfg = config();
    for batch in [false, true] {
        let mut generated = vec![Vec::new(); 2];
        let mut curs = vec![vec![7]; 2];
        let mut guards = vec![Qwen4ReasoningEosGuard::from_prompt(&cfg, &[10]); 2];
        let mut batch_callback = |_: &[(usize, u32)]| [false; LANES];
        let mut legacy_callback = |_, _| false;
        let mut callback = if batch {
            ParallelTokenCallback::Batch(&mut batch_callback)
        } else {
            ParallelTokenCallback::Legacy(&mut legacy_callback)
        };
        assert!(deliver_parallel_token_step(
            &cfg,
            false,
            0,
            &[11, 10],
            &mut generated,
            &mut curs,
            &mut guards,
            &mut callback,
            None,
            None,
        ));
        assert!(!guards[0].blocks(&cfg, cfg.eos, false));
        assert!(guards[1].blocks(&cfg, cfg.eos, false));
        assert_eq!(generated, [vec![11], vec![10]]);
        assert_eq!(curs, [vec![7], vec![7]]);
    }
}

#[test]
fn delivery_keeps_per_lane_sample_ids_and_rng_progression() {
    let cfg = config();
    let sampling = infr_core::config::SamplingCfg::default();
    for lanes in 1..=LANES {
        let mut sampled = Vec::new();
        let mut controls = Vec::new();
        for lane in 0..lanes {
            let req = RequestCtx::new(RequestSampling {
                temp: Some(0.7),
                top_k: Some(4),
                top_p: Some(1.0),
                seed: Some(20 + lane as u64),
                ..Default::default()
            });
            sampled.push(ParallelSampler::new(&req, &sampling));
            controls.push(ParallelSampler::new(&req, &sampling));
        }
        let mut generated = vec![Vec::new(); lanes];
        let mut curs = vec![vec![7]; lanes];
        let mut guards = vec![Qwen4ReasoningEosGuard::default(); lanes];
        for _ in 0..4 {
            // A masked row must retain its already selected id through delivery.
            let next = sampled
                .iter_mut()
                .map(|sampler| sampler.sample(&mut [0.2, f32::NEG_INFINITY, 0.7, 0.9]))
                .collect::<Vec<_>>();
            let control = controls
                .iter_mut()
                .map(|sampler| sampler.sample(&mut [0.2, f32::NEG_INFINITY, 0.7, 0.9]))
                .collect::<Vec<_>>();
            assert_eq!(next, control);
            let mut callback = |tokens: &[(usize, u32)]| {
                let expected = next.iter().copied().enumerate().collect::<Vec<_>>();
                assert_eq!(tokens, expected.as_slice());
                [true; LANES]
            };
            assert!(!deliver_parallel_token_step(
                &cfg,
                false,
                0,
                &next,
                &mut generated,
                &mut curs,
                &mut guards,
                &mut ParallelTokenCallback::Batch(&mut callback),
                None,
                None,
            ));
            for lane in 0..lanes {
                assert_eq!(generated[lane].last(), Some(&next[lane]));
                assert_eq!(sampled[lane].next_uniform(), controls[lane].next_uniform());
            }
        }
    }
}
