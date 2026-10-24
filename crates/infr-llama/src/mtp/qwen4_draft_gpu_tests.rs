//! This is a synthetic-history, real-weight head module test, not target acceptance or E2E TPS.
#[cfg(test)]
mod fixtures {
    use super::super::*;
    use infr_core::WeightSource;
    use infr_vulkan::VulkanBackend;
    use sha2::{Digest, Sha256};
    use std::collections::BTreeSet;
    use std::path::PathBuf;
    use std::time::Instant;

    const WINDOW: usize = 32_768;
    const MAX_CTX: usize = 160_000;
    const COPY_CHUNK: usize = 4 * 1024 * 1024;
    const TRIALS: usize = 12;

    struct SharedTensor {
        buffer: Box<dyn Buffer>,
        dtype: DType,
        numel: usize,
    }

    impl SharedTensor {
        fn load(be: &VulkanBackend, gguf: &Gguf, cfg: &crate::Config, name: &str) -> Self {
            let info = gguf.tensors().iter().find(|t| t.name == name).unwrap();
            assert_eq!(info.shape, [cfg.n_embd, cfg.vocab], "{name}");
            assert!(
                info.dtype.is_quant(),
                "{name} must retain its source quantization"
            );
            let bytes = gguf.tensor_bytes_arc(name).unwrap();
            let numel = cfg.n_embd.checked_mul(cfg.vocab).unwrap();
            assert_eq!(bytes.len(), infr_gguf::nbytes(info.dtype, numel));
            let padded = infr_vulkan::linear::pad_to_u32_align(bytes.as_ref());
            let buffer = be.alloc(padded.len(), BufferUsage::Weights).unwrap();
            be.upload(buffer.as_ref(), &padded).unwrap();
            eprintln!(
                "head_weight name={name} dtype={:?} bytes={} sha256={:x}",
                info.dtype,
                bytes.len(),
                Sha256::digest(bytes.as_ref()),
            );
            Self {
                buffer,
                dtype: info.dtype,
                numel,
            }
        }

        fn shared(&self) -> SharedWeight<'_> {
            (self.buffer.as_ref(), self.dtype, self.numel)
        }
    }

    struct Fixture {
        heads: [Qwen4MtpSession; 2],
        embedding: SharedTensor,
        output: SharedTensor,
        zero: Box<dyn Buffer>,
        hidden: Vec<f32>,
        be: VulkanBackend,
    }

    #[derive(Clone, Copy, Default)]
    struct Timing {
        draft_us: f64,
        settle_us: f64,
    }

    fn required_path(key: &str) -> PathBuf {
        let path = PathBuf::from(std::env::var_os(key).unwrap_or_else(|| panic!("set {key}")));
        assert!(
            path.is_absolute() && path.is_file(),
            "{key} must name an absolute existing file"
        );
        path
    }

    fn assert_bytes(actual: &[u8], expected: &[u8], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length");
        if let Some(offset) = actual.iter().zip(expected).position(|(a, b)| a != b) {
            panic!(
                "{label}: byte {offset}: got {}, want {}",
                actual[offset], expected[offset]
            );
        }
    }

    impl Fixture {
        fn new() -> Self {
            let target_path = required_path("MOE4ALL_SPEC_TEST_MODEL");
            let draft_path = required_path("MOE4ALL_MTP_TEST_DRAFT");
            let device = std::env::var("MOE4ALL_TEST_DEVICE").expect("set MOE4ALL_TEST_DEVICE");
            let target = Gguf::open(&target_path).unwrap();
            let cfg = crate::Config::from_gguf(&target).unwrap();
            assert!(
                cfg.qwen4exp,
                "this fixture requires a supported Qwen3.8/4 target"
            );
            assert!(cfg.vocab > 1234);
            let mut engine = infr_core::config::Config::default();
            engine.device.dev = Some(device.clone());
            engine.spec.mtp = true;
            let be = VulkanBackend::new_with(Arc::new(engine)).unwrap();
            // Do not load the target trunk, experts or full target session state.
            let embedding = SharedTensor::load(&be, &target, &cfg, "token_embd.weight");
            let output_name = if target.tensors().iter().any(|t| t.name == "output.weight") {
                "output.weight"
            } else {
                "token_embd.weight"
            };
            let output = SharedTensor::load(&be, &target, &cfg, output_name);
            drop(target);
            let fixed = Qwen4MtpSession::load_fixed_vulkan(&be, &draft_path, &cfg).unwrap();
            let spec = qwen4_mtp_kv_spec(&cfg, WINDOW).unwrap();
            // Four physical K/V rings plus bounded module scratch; not a context-sized hidden prime.
            let arena_bytes = spec.segment_bytes.checked_mul(4).unwrap() + 256 * 1024 * 1024;
            be.init_unified_vram(arena_bytes).unwrap();
            let catch = Qwen4MtpSession::shared_catch_workspace(&be, &fixed, MAX_CTX, 4).unwrap();
            let heads = [
                Qwen4MtpSession::with_fixed_and_catch(
                    &be,
                    Arc::clone(&fixed),
                    MAX_CTX,
                    Arc::clone(&catch),
                    WINDOW,
                )
                .unwrap(),
                Qwen4MtpSession::with_fixed_and_catch(
                    &be,
                    Arc::clone(&fixed),
                    MAX_CTX,
                    Arc::clone(&catch),
                    WINDOW,
                )
                .unwrap(),
            ];
            assert!(Arc::ptr_eq(&heads[0].fixed, &heads[1].fixed));
            assert!(Arc::ptr_eq(&heads[0].catch, &heads[1].catch));
            assert_eq!(heads[0].attention_window, WINDOW);
            assert_eq!(heads[1].kv_capacity, WINDOW);
            let hidden = (0..heads[0].h_width())
                .map(|i| ((i * 17 + 23) % 101) as f32 / 256.0 - 0.2)
                .collect();
            let zero = be.alloc(COPY_CHUNK, BufferUsage::Staging).unwrap();
            be.upload(zero.as_ref(), &vec![0u8; COPY_CHUNK]).unwrap();
            eprintln!(
                "head_module target={} sidecar={} device={device} ctx={MAX_CTX} window={WINDOW} \
             n_embd={} n_head={} n_kv={} head_dim={} h_width={} arena_bytes={arena_bytes} \
             shared_fixed=true shared_catch=true synthetic_history=true target_trunk_loaded=false",
                target_path.display(),
                draft_path.display(),
                cfg.n_embd,
                cfg.n_head,
                cfg.n_kv,
                cfg.head_dim,
                heads[0].h_width(),
            );
            Self {
                heads,
                embedding,
                output,
                zero,
                hidden,
                be,
            }
        }

        fn shared(&self) -> SharedWeights<'_> {
            (self.embedding.shared(), self.output.shared())
        }

        fn row_bytes(&self) -> usize {
            self.heads[0].kv_spec.logical_bytes / WINDOW
        }

        fn reset_history(&self, start: usize, width: usize) {
            for head in &self.heads {
                head.settle_draft(&self.be, 0).unwrap();
            }
            let a = self.heads[0].kv_for_depth(&self.be, MAX_CTX).unwrap();
            let b = self.heads[1].kv_for_depth(&self.be, MAX_CTX).unwrap();
            assert!(
                a.segmented && b.segmented,
                "do not silently substitute flat KV"
            );
            let ring_bytes = self.heads[0].kv_spec.logical_bytes;
            let mut copies = Vec::new();
            for dst in [a.k.as_ref(), a.v.as_ref(), b.k.as_ref(), b.v.as_ref()] {
                for offset in (0..ring_bytes).step_by(COPY_CHUNK) {
                    copies.push((
                        self.zero.as_ref(),
                        0,
                        dst,
                        offset,
                        COPY_CHUNK.min(ring_bytes - offset),
                    ));
                }
            }
            // upload(segmented KV) is deliberately not used: the flat staging source is copied through
            // Backend's range API, which splits physical segmented destinations correctly.
            self.be.copy_buffer_ranges(&copies).unwrap();

            let mut slots = BTreeSet::new();
            slots.extend(0..DRAFT_TOKENS);
            slots.extend(WINDOW - DRAFT_TOKENS..WINDOW);
            slots.insert((start + WINDOW - 1) % WINDOW);
            slots.extend((0..width).map(|row| (start + row) % WINDOW));
            let slots = slots
                .into_iter()
                .filter(|&slot| start >= WINDOW || slot < start)
                .collect::<Vec<_>>();
            let row_bytes = self.row_bytes();
            let mut data = vec![0u8; 2 * slots.len() * row_bytes];
            for (index, &slot) in slots.iter().enumerate() {
                for side in 0..2 {
                    let offset = (side * slots.len() + index) * row_bytes;
                    for (d, bytes) in data[offset..offset + row_bytes]
                        .chunks_exact_mut(2)
                        .enumerate()
                    {
                        let value = (slot % 61 + 1) as f32 / 128.0 + (d % 31) as f32 / 4096.0;
                        let signed = if side == 0 { value } else { -value };
                        bytes.copy_from_slice(&half::f16::from_f32(signed).to_le_bytes());
                    }
                }
            }
            let marker = self.be.alloc(data.len(), BufferUsage::Staging).unwrap();
            self.be.upload(marker.as_ref(), &data).unwrap();
            let mut copies = Vec::new();
            for kv in [&a, &b] {
                for (side, dst) in [kv.k.as_ref(), kv.v.as_ref()].into_iter().enumerate() {
                    for (index, &slot) in slots.iter().enumerate() {
                        copies.push((
                            marker.as_ref(),
                            (side * slots.len() + index) * row_bytes,
                            dst,
                            slot * row_bytes,
                            row_bytes,
                        ));
                    }
                }
            }
            self.be.copy_buffer_ranges(&copies).unwrap();
        }

        fn rows(&self, slot: usize, start: usize, width: usize) -> Vec<[Vec<u8>; 2]> {
            let kv = self.heads[slot].kv.lock().unwrap();
            let row_bytes = self.row_bytes();
            (0..width)
                .map(|row| {
                    let mut bytes = [vec![0u8; row_bytes], vec![0u8; row_bytes]];
                    for (side, src) in [kv.k.as_ref(), kv.v.as_ref()].into_iter().enumerate() {
                        self.be
                            .download_range(
                                src,
                                (start + row) % WINDOW * row_bytes,
                                &mut bytes[side],
                            )
                            .unwrap();
                    }
                    bytes
                })
                .collect()
        }

        fn compare_visible(&self, depth: usize, label: &str) {
            let a = self.heads[0].kv.lock().unwrap();
            let b = self.heads[1].kv.lock().unwrap();
            let bytes = depth.min(WINDOW) * self.row_bytes();
            let mut left = vec![0u8; COPY_CHUNK.min(bytes)];
            let mut right = vec![0u8; left.len()];
            for (side, (a, b)) in [(a.k.as_ref(), b.k.as_ref()), (a.v.as_ref(), b.v.as_ref())]
                .into_iter()
                .enumerate()
            {
                for offset in (0..bytes).step_by(COPY_CHUNK) {
                    let count = COPY_CHUNK.min(bytes - offset);
                    self.be
                        .download_range(a, offset, &mut left[..count])
                        .unwrap();
                    self.be
                        .download_range(b, offset, &mut right[..count])
                        .unwrap();
                    assert_bytes(
                        &left[..count],
                        &right[..count],
                        &format!("{label}: side={side} offset={offset} depth={depth}"),
                    );
                }
            }
        }

        fn draft(&self, slot: usize, tail: bool, start: usize, width: usize) -> (Vec<u32>, f64) {
            self.be.sync().unwrap();
            let begin = Instant::now();
            let ids = self.heads[slot]
                .draft_with_tail_mode(
                    &self.be,
                    1234,
                    &self.hidden,
                    start,
                    i32::try_from(start).unwrap(),
                    width,
                    self.shared(),
                    tail,
                )
                .unwrap();
            let draft_us = begin.elapsed().as_secs_f64() * 1e6;
            assert_eq!(ids.len(), width - 1, "actual head must produce candidates");
            let kv = self.heads[slot].kv.lock().unwrap();
            let pending = kv
                .draft_journal
                .pending
                .expect("actual head transaction must exist");
            assert_eq!(pending.first_saved, width.min(WINDOW.saturating_sub(start)));
            (ids, draft_us)
        }

        fn settle(&self, slot: usize, retained: usize) -> f64 {
            let begin = Instant::now();
            self.heads[slot].settle_draft(&self.be, retained).unwrap();
            let settle_us = begin.elapsed().as_secs_f64() * 1e6;
            assert!(self.heads[slot]
                .kv
                .lock()
                .unwrap()
                .draft_journal
                .pending
                .is_none());
            settle_us
        }

        fn parity_transaction(&self, start: usize, width: usize, retained: usize, reverse: bool) {
            let old = self.rows(0, start, width);
            let other_old = self.rows(1, start, width);
            for row in 0..width {
                for side in 0..2 {
                    assert_bytes(
                        &old[row][side],
                        &other_old[row][side],
                        "initial overwritten row",
                    );
                }
            }
            let mut ids = [Vec::new(), Vec::new()];
            let order = if reverse { [1, 0] } else { [0, 1] };
            for slot in order {
                ids[slot] = self.draft(slot, slot == 1, start, width).0;
            }
            assert_eq!(
                ids[0], ids[1],
                "candidate parity start={start} width={width}"
            );
            self.compare_visible(start + width, "pending draft");
            let written = self.rows(0, start, width);
            assert!(
                written[width - 1].iter().flatten().any(|&byte| byte != 0),
                "the final real-weight step must write nonzero KV"
            );
            let settles = [self.settle(0, retained), self.settle(1, retained)];
            let first_saved = width.min(WINDOW.saturating_sub(start));
            for slot in 0..2 {
                let settled = self.rows(slot, start, width);
                for row in 0..width {
                    let expected = if row < retained {
                        &written[row]
                    } else if row >= first_saved {
                        &old[row]
                    } else {
                        continue; // Rejected new slots were never part of the previously-live prefix.
                    };
                    for side in 0..2 {
                        assert_bytes(&settled[row][side], &expected[side],
                        &format!("settlement slot={slot} start={start} width={width} retained={retained} row={row} side={side}"));
                    }
                }
                self.heads[slot].settle_draft(&self.be, 0).unwrap();
            }
            self.compare_visible(start + retained, "settled visible prefix");
            eprintln!(
                "head_parity start={start} width={width} retained={retained} \
            first_saved={first_saved} restored_rows={} candidates={:?} \
            settle_full_us={:.3} settle_kv_only_us={:.3}",
                width - retained.max(first_saved),
                ids[0],
                settles[0],
                settles[1]
            );
        }

        fn timing_pair(&self, trial: usize, warmup: bool) -> [Timing; 2] {
            const START: usize = 30_000;
            self.reset_history(START, 4);
            let order = if trial % 2 == 0 { [0, 1] } else { [1, 0] };
            let mut ids = [Vec::new(), Vec::new()];
            let mut samples = [Timing::default(); 2];
            for slot in order {
                (ids[slot], samples[slot].draft_us) = self.draft(slot, slot == 1, START, 4);
                samples[slot].settle_us = self.settle(slot, 0);
            }
            assert_eq!(ids[0], ids[1], "paired trial {trial}");
            eprintln!("head_pair warmup={warmup} trial={trial} order={order:?} start={START} width=4 \
            full_draft_us={:.3} kv_only_draft_us={:.3} full_settle_us={:.3} kv_only_settle_us={:.3} \
            restored_rows=0 candidates={:?}",
            samples[0].draft_us, samples[1].draft_us, samples[0].settle_us,
            samples[1].settle_us, ids[0]);
            samples
        }
    }

    fn median(mut samples: Vec<f64>) -> f64 {
        samples.sort_by(f64::total_cmp);
        let middle = samples.len() / 2;
        if samples.len() % 2 == 0 {
            (samples[middle - 1] + samples[middle]) / 2.0
        } else {
            samples[middle]
        }
    }

    #[test]
    #[ignore = "requires MOE4ALL_SPEC_TEST_MODEL, MOE4ALL_MTP_TEST_DRAFT and an idle MOE4ALL_TEST_DEVICE"]
    fn actual_head_tail_gpu_parity_and_paired_timing() {
        let fixture = Fixture::new();
        for width in [2, 3, 4] {
            for start in [30_000, WINDOW - 1, WINDOW, WINDOW + 1] {
                fixture.reset_history(start, width);
                fixture.parity_transaction(start, width, 1, (start + width) % 2 != 0);
                if width == 4 && start == WINDOW - 1 {
                    // A new draft immediately after partial acceptance must see the restored history.
                    fixture.parity_transaction(start + 1, width, 0, true);
                }
            }
        }
        for (start, retained) in [
            (WINDOW - 1, 0),
            (WINDOW - 1, 2),
            (WINDOW - 1, 4),
            (WINDOW + 1, 0),
            (WINDOW + 1, 4),
        ] {
            fixture.reset_history(start, 4);
            fixture.parity_transaction(start, 4, retained, false);
        }
        for trial in 0..2 {
            fixture.timing_pair(trial, true);
        }
        let pairs = (0..TRIALS)
            .map(|trial| fixture.timing_pair(trial, false))
            .collect::<Vec<_>>();
        let full = median(pairs.iter().map(|p| p[0].draft_us).collect());
        let tail = median(pairs.iter().map(|p| p[1].draft_us).collect());
        let paired_delta = median(
            pairs
                .iter()
                .map(|p| (p[1].draft_us / p[0].draft_us - 1.0) * 100.0)
                .collect(),
        );
        eprintln!(
            "head_module_summary trials={TRIALS} width=4 start=30000 full_median_us={full:.3} \
        kv_only_median_us={tail:.3} paired_delta_percent={paired_delta:.3} \
        full_settle_median_us={:.3} kv_only_settle_median_us={:.3} \
        draft_includes_graph_build_compile_execution_readback=true restore_timed_separately=true \
        target_acceptance_measured=false e2e_tps_measured=false",
            median(pairs.iter().map(|p| p[0].settle_us).collect()),
            median(pairs.iter().map(|p| p[1].settle_us).collect())
        );
    }
}
