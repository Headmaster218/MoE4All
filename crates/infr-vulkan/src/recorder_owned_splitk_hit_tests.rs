//! Child of recorder_owned_fused_tests; parent registers this module separately.
//! Test-only NFF-chunk UG+SiLU+Down partial, INCLUDING the final reduction cost.
use super::*;

#[path = "recorder_owned_hitfirst_matrix_tests.rs"]
mod hitfirst_v3;
#[path = "recorder_owned_same_family_dualqueue_tests.rs"]
mod same_family_dualqueue;

const CHUNKS: [usize; 3] = [32, 64, 128];
const LOCAL: usize = 128;
const FRAME_REPEATS: usize = 16;
const PARTIAL_NAN: u32 = 0x7fc0_51ab;
const MISS_ID: usize = 0;

fn split_backend() -> VulkanBackend {
    let requested = std::env::var("MOE4ALL_TEST_DEVICE").expect("explicit Vulkan1 device required");
    assert_eq!(
        requested, "Vulkan1",
        "parent serial wrapper device selection"
    );
    let be = fused_backend();
    let properties = unsafe {
        be.shared
            .instance
            .get_physical_device_properties(be.shared.physical_device)
    };
    let name =
        unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) }.to_string_lossy();
    eprintln!("SPLITK_V2_DEVICE requested={requested} physicalDeviceName={name}");
    assert!(
        name.contains("RX 7900 XTX"),
        "expected RX 7900 XTX, selected {name}"
    );
    assert!(
        properties.limits.max_compute_work_group_invocations >= 1024
            && properties.limits.max_compute_work_group_size[0] >= 1024,
        "fair builtin copy requires LOCAL1024 in both arms"
    );
    be
}

#[derive(Clone, Copy)]
struct SplitKernels {
    partial: ComputeKernel,
    reduce: ComputeKernel,
    chunk: usize,
}

fn external(be: &VulkanBackend, file: &str, shared: bool) -> ComputeKernel {
    let root = PathBuf::from(
        std::env::var_os("MOE4ALL_FUSION_V2_DIR")
            .expect("MOE4ALL_FUSION_V2_DIR must name external split-K probe SPVs"),
    );
    let path = root.join(file);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "SPIR-V words: {}", path.display());
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    assert_eq!(words.first(), Some(&0x0723_0203));
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3)
    });
    eprintln!(
        "SPLITK_SPV path={} bytes={} fnv1a64={hash:016x}",
        path.display(),
        bytes.len()
    );
    let name = Box::leak(format!("splitk_v2_{file}").into_boxed_str());
    be.kernel_sg(name, &words, 4, if shared { 64 } else { 40 }, 32)
}

impl SplitKernels {
    fn new(be: &VulkanBackend, p: Profile, chunk: usize, shared: bool) -> Self {
        assert!(CHUNKS.contains(&chunk));
        assert!(matches!(
            (p.gate_up, p.down),
            (DType::Q5K, DType::Q5K)
                | (DType::Iq2S, DType::Iq4Nl)
                | (DType::Iq3S, DType::Iq4Nl)
        ));
        let suffix = if shared { "-shared" } else { "" };
        Self {
            partial: external(
                be,
                &format!("splitk-{}-b{chunk}{suffix}.spv", format_name(p)),
                shared,
            ),
            reduce: external(be, &format!("splitk-reduce{suffix}.spv"), shared),
            chunk,
        }
    }
}

struct SplitFixture {
    ff: FusedFixture,
    partial: Box<dyn Buffer>,
    chunk: usize,
    partial_view: vk::DescriptorBufferInfo,
}

impl SplitFixture {
    fn new(
        be: &VulkanBackend,
        source: &Sources,
        p: Profile,
        rows: usize,
        shared: bool,
        chunk: usize,
    ) -> Self {
        let ff = FusedFixture::new(be, source, p, rows, shared);
        let body = rows * ff.physical_used() * p.nff.div_ceil(chunk) * p.ne * 4;
        let partial = mapped(be, GUARD + body + GUARD);
        let b = Recorder::vkb_exact(partial.as_ref());
        let alignment = unsafe {
            be.shared
                .instance
                .get_physical_device_properties(be.shared.physical_device)
        }
        .limits
        .min_storage_buffer_offset_alignment
        .max(1);
        let offset = b.offset + GUARD as u64;
        assert_eq!(
            offset % alignment,
            0,
            "actual guarded partial descriptor alignment"
        );
        assert_eq!(b.range as usize, GUARD + body + GUARD);
        let partial_view = vk::DescriptorBufferInfo::default()
            .buffer(b.buffer)
            .offset(offset)
            .range(body as u64);
        Self {
            ff,
            partial,
            chunk,
            partial_view,
        }
    }

    fn parts(&self) -> usize {
        self.ff.inner.p.nff.div_ceil(self.chunk)
    }

    fn partial_binding(&self) -> vk::DescriptorBufferInfo {
        // Runtime SSBO length is BODY only. Neither canary can satisfy capacity checks.
        self.partial_view
    }

    fn prepare(&mut self, source: &Sources, bank: usize, mixed: bool, resident: bool) {
        // Start from ALL resident weights; replace exactly one expert's three roles.
        // This is deliberately 7/8 or 9/10 routed hits, not the old sparse-H fixture.
        self.ff.prepare(source, bank, true);
        let f = &mut self.ff.inner;
        f.job_words.fill(0);
        f.job_count = 0;
        let all = (1u32 << f.p.used) - 1;
        f.hits = f
            .routed
            .chunks_exact(f.p.used)
            .map(|ids| {
                ids.iter().enumerate().fold(0, |mask, (slot, &id)| {
                    mask | (u32::from(!mixed || id as usize != MISS_ID) << slot)
                })
            })
            .collect();
        let mut weights = f.expected_weights.clone();
        if mixed {
            for role in 0..3 {
                let range = &source.ranges[role][MISS_ID];
                if !resident {
                    weights[range.clone()].fill(POISON);
                }
                let src = source.imported.device_addr().unwrap()
                    + (bank * source.span + range.start) as u64;
                let dst = f.weights.device_addr().unwrap() + range.start as u64;
                let at = (ORDINAL * CAPACITY + f.job_count) * 8;
                f.job_words[at..at + 8].copy_from_slice(&[
                    src as u32,
                    (src >> 32) as u32,
                    dst as u32,
                    (dst >> 32) as u32,
                    range.len() as u32,
                    (role * f.p.used + MISS_ID) as u32,
                    f.job_count as u32,
                    (role * f.p.used + MISS_ID) as u32,
                ]);
                f.job_count += 1;
            }
        }
        assert!(f
            .hits
            .iter()
            .all(|m| m.count_ones() as usize == f.p.used - usize::from(mixed)));
        assert!(f.hits.iter().all(|m| *m & !all == 0));
        write(f.weights.as_ref(), &weights);
        write(f.jobs.as_ref(), bytemuck::cast_slice(&f.job_words));
        f.set_status(if mixed { 4 } else { 1 }, f.job_count);
        f.write_masks();
        for b in [&f.gate, &f.up, &f.activation] {
            poison_float_body(b.as_ref(), 0);
        }
        self.ff.shared_masks();
        poison_float_body(self.partial.as_ref(), GUARD);
    }

    fn push(&self) -> Vec<u32> {
        let f = &self.ff.inner;
        let mut push = vec![
            f.p.ne as u32,
            f.p.nff as u32,
            f.rows as u32,
            f.p.used as u32,
            f.p.used as u32,
            self.ff.physical_used() as u32,
            0,
            f.p.used as u32,
            (2 * f.p.used) as u32,
            self.parts() as u32,
        ];
        if let Some(shared) = &self.ff.shared {
            for matrix in &shared.matrices {
                let address = matrix.device_addr().unwrap();
                assert_ne!(address, 0);
                push.extend([address as u32, (address >> 32) as u32]);
            }
        }
        push
    }

    fn partial(&self, rec: &Recorder<'_>, k: SplitKernels) {
        assert_eq!(self.chunk, k.chunk);
        let f = &self.ff.inner;
        rec.dispatch(
            k.partial,
            &[
                Recorder::vkb_exact(f.input.as_ref()),
                Recorder::vkb_exact(f.ids.as_ref()),
                Recorder::vkb_exact(f.tape.as_ref()),
                self.partial_binding(),
            ],
            1,
            bytemuck::cast_slice(&self.push()),
            (f.rows * self.ff.physical_used() * self.parts()) as u32,
        );
    }

    fn reduce(&self, rec: &Recorder<'_>, k: SplitKernels) {
        let f = &self.ff.inner;
        let mut output = Recorder::vkb_exact(f.output.as_ref());
        output.range -= GUARD as u64;
        rec.dispatch(
            k.reduce,
            &[
                self.partial_binding(),
                Recorder::vkb_exact(f.ids.as_ref()),
                Recorder::vkb_exact(f.tape.as_ref()),
                output,
            ],
            1,
            bytemuck::cast_slice(&self.push()),
            (f.rows * self.ff.physical_used() * f.p.ne).div_ceil(LOCAL) as u32,
        );
    }

    fn record(&self, rec: &Recorder<'_>, candidate: bool, k: SplitKernels) {
        let f = &self.ff.inner;
        if f.job_count != 0 {
            let scope = rec.independent_dispatch_scope();
            f.cu_copy(rec); // SAME builtin wide1024/groups8 in both arms.
            if candidate {
                self.partial(rec, k);
            } else {
                self.ff.projections(rec, 1, 0);
            }
            drop(scope); // Mandatory BDA/copy and partial/GU join, not bypassed.
        } else if candidate {
            self.partial(rec, k);
        } else {
            self.ff.projections(rec, 1, 0);
        }
        if candidate {
            // This real dispatch and its ordinary RAW barrier are inside the timed frame.
            self.reduce(rec, k);
        } else {
            self.ff.activation(rec);
            self.ff.down(rec, 1, 0);
        }
        if f.job_count != 0 {
            rec.global_barrier();
            self.ff.projections(rec, 2, 0);
            self.ff.activation(rec);
            self.ff.down(rec, 2, 0);
        }
    }

    fn readonly(&self) -> Vec<Vec<u8>> {
        let f = &self.ff.inner;
        let mut bytes: Vec<_> = [
            f.ids.as_ref(),
            f.tape.as_ref(),
            f.input.as_ref(),
            f.jobs.as_ref(),
            f.journal.as_ref(),
        ]
        .into_iter()
        .map(read)
        .collect();
        if let Some(shared) = &self.ff.shared {
            bytes.extend(shared.matrices.iter().map(|m| read(m.as_ref())));
        }
        bytes
    }

    fn verify_partial(&self) {
        let f = &self.ff.inner;
        let bytes = read(self.partial.as_ref());
        assert!(bytes[..GUARD]
            .iter()
            .chain(&bytes[bytes.len() - GUARD..])
            .all(|&v| v == POISON));
        let stride = self.parts() * f.p.ne * 4;
        for assignment in 0..f.rows * self.ff.physical_used() {
            let row = assignment / self.ff.physical_used();
            let slot = assignment % self.ff.physical_used();
            let active = slot == f.p.used || (f.hits[row] & (1 << slot)) != 0;
            for word in bytes[GUARD + assignment * stride..GUARD + (assignment + 1) * stride]
                .chunks_exact(4)
            {
                let bits = u32::from_le_bytes(word.try_into().unwrap());
                if active {
                    assert!(
                        f32::from_bits(bits).is_finite(),
                        "H partial must overwrite poison"
                    );
                } else {
                    assert_eq!(bits, PARTIAL_NAN, "M partial must stay untouched");
                }
            }
        }
    }

    fn verify(
        &self,
        expected: &[u8],
        source: &Sources,
        bank: usize,
        readonly: &[Vec<u8>],
    ) -> Errors {
        assert_eq!(
            self.readonly(),
            readonly,
            "exact read-only IDs/H-M masks/Tape/input/jobs/journal"
        );
        assert_eq!(
            source.bytes(bank),
            self.ff.inner.expected_weights,
            "exact imported source bytes"
        );
        self.verify_partial();
        let errors = self.ff.verify(expected); // Existing tolerance, no relaxation.
        let actual = read(self.ff.inner.output.as_ref());
        let mut near_zero = (0usize, 0.0f64);
        for (i, (a, b)) in actual[..actual.len() - GUARD]
            .chunks_exact(4)
            .zip(expected[..expected.len() - GUARD].chunks_exact(4))
            .enumerate()
        {
            let a = f32::from_le_bytes(a.try_into().unwrap()) as f64;
            let b = f32::from_le_bytes(b.try_into().unwrap()) as f64;
            if b.abs() <= ABS_TOL && (a - b).abs() > near_zero.1 {
                near_zero = (i, (a - b).abs());
            }
        }
        if near_zero.1 > ABS_TOL * 0.25 {
            eprintln!(
                "SPLITK_V2_NEAR_ZERO profile={} rows={} chunk={} index={} max_abs={} tolerance_unchanged=true",
                self.ff.inner.p.name, self.ff.inner.rows, self.chunk, near_zero.0, near_zero.1
            );
        }
        errors
    }
}

fn poison_float_body(buffer: &dyn Buffer, prefix: usize) {
    let mut bytes = vec![POISON; buffer.len_bytes()];
    let end = bytes.len() - GUARD;
    for word in bytes[prefix..end].chunks_exact_mut(4) {
        word.copy_from_slice(&PARTIAL_NAN.to_le_bytes());
    }
    write(buffer, &bytes);
}

fn small() -> [Profile; 2] {
    let mut p = native_profiles();
    p[0].ne = 256;
    p[0].nff = 256;
    p[1].ne = 512;
    p[1].nff = 288; // real IQ4NL32 tail:64/128 chunks leave32.
    p
}

fn sample(be: &VulkanBackend, f: &SplitFixture, k: SplitKernels) {
    let rec = be.recorder().unwrap();
    rec.seed_barrier();
    f.record(&rec, true, k);
    rec.publish_host_reads();
    rec.finish().unwrap();
}

#[test]
#[ignore = "parent serial GPU only; splitK64/128 Q5K/IQ2S, reduced rows1..8 shared-Q8 and replay"]
fn gpu_owned_splitk_hit_v2_correctness_rows_1_to_8() {
    let be = split_backend();
    for p in small() {
        let source = Sources::new(&be, p, 0);
        let mut errors = Errors::default();
        for shared in [false, true] {
            for chunk in CHUNKS {
                let k = SplitKernels::new(&be, p, chunk, shared);
                for rows in 1..=8 {
                    let mut f = SplitFixture::new(&be, &source, p, rows, shared, chunk);
                    let bank = rows % source.banks;
                    let expected = f.ff.reference(&be, &source, bank);
                    for mixed in [false, true] {
                        f.prepare(&source, bank, mixed, false);
                        let immutable = f.readonly();
                        sample(&be, &f, k);
                        errors.merge(f.verify(&expected, &source, bank, &immutable));
                        // Same metadata shape, different immutable source bank and fresh poison.
                        let next = (bank + 1) % source.banks;
                        let expected_next = f.ff.reference(&be, &source, next);
                        f.prepare(&source, next, mixed, false);
                        let immutable = f.readonly();
                        let rec = be.recorder_persistent().unwrap();
                        rec.seed_barrier();
                        f.record(&rec, true, k);
                        rec.publish_host_reads();
                        let recorded = rec.finish_record().unwrap();
                        recorded.replay().unwrap();
                        errors.merge(f.verify(&expected_next, &source, next, &immutable));
                        poison_float_body(f.partial.as_ref(), GUARD);
                        // Replay must overwrite H partials, not borrow the first replay's sums.
                        recorded.replay().unwrap();
                        errors.merge(f.verify(&expected_next, &source, next, &immutable));
                    }
                }
            }
        }
        println!(
            "SPLITK_V2_CORRECT profile={} reduced_ne={} reduced_nff={} rows=1..8 chunks=32,64,128 local=128 shared_Q8=true_false full_and_one_expert_miss=true maxerr={errors:?} atol={ABS_TOL} rtol={REL_TOL} weights_guards_exact=true",
            p.name, p.ne, p.nff
        );
    }
}

#[test]
#[ignore = "parent serial GPU only; H/M isolation, poisoned final partial, tail and outer SKIP"]
fn gpu_owned_splitk_hit_v2_masks_tail_and_skip() {
    let be = split_backend();
    assert!(be.shared.conditional_rendering.is_some());
    for p in small() {
        let source = Sources::new(&be, p, 0);
        for shared in [false, true] {
            for chunk in CHUNKS {
                let k = SplitKernels::new(&be, p, chunk, shared);
                let mut f = SplitFixture::new(&be, &source, p, 8, shared, chunk);
                let expected = f.ff.reference(&be, &source, 0);
                f.prepare(&source, 0, true, true);
                let immutable = f.readonly();
                let rec = be.recorder().unwrap();
                rec.seed_barrier();
                f.partial(&rec, k);
                f.reduce(&rec, k);
                rec.publish_host_reads();
                rec.finish().unwrap();
                assert_eq!(f.readonly(), immutable);
                f.verify_partial();
                let actual = read(f.ff.inner.output.as_ref());
                assert!(actual[actual.len() - GUARD..].iter().all(|&v| v == POISON));
                let mut errors = Errors::default();
                for a in 0..8 * f.ff.physical_used() {
                    let row = a / f.ff.physical_used();
                    let slot = a % f.ff.physical_used();
                    let active = slot == p.used || f.ff.inner.hits[row] & (1 << slot) != 0;
                    let range = a * p.ne * 4..(a + 1) * p.ne * 4;
                    if active {
                        errors.merge(compare(&actual[range.clone()], &expected[range]));
                    } else {
                        assert!(actual[range]
                            .chunks_exact(4)
                            .all(|w| w == 0.125f32.to_le_bytes()));
                    }
                }
                assert_eq!(
                    read(f.ff.inner.weights.as_ref()),
                    f.ff.inner.expected_weights
                );
                // Last partial is essential, including IQ2/288's32-element tail.
                f.prepare(&source, 0, false, true);
                let rec = be.recorder().unwrap();
                rec.seed_barrier();
                f.partial(&rec, k);
                rec.publish_host_reads();
                rec.finish().unwrap();
                let mut partial = read(f.partial.as_ref());
                let at = GUARD + (f.parts() - 1) * p.ne * 4;
                partial[at..at + 4].copy_from_slice(&PARTIAL_NAN.to_le_bytes());
                write(f.partial.as_ref(), &partial);
                let rec = be.recorder().unwrap();
                rec.seed_barrier();
                f.reduce(&rec, k);
                rec.publish_host_reads();
                rec.finish().unwrap();
                assert!(
                    f32::from_le_bytes(read(f.ff.inner.output.as_ref())[..4].try_into().unwrap())
                        .is_nan(),
                    "negative control: reducer must consume the poisoned final partial"
                );
                f.prepare(&source, 0, false, true);
                let before_partial = read(f.partial.as_ref());
                let before_output = read(f.ff.inner.output.as_ref());
                let control = mapped(&be, 16);
                write(control.as_ref(), &[0; 16]);
                let rec = be.recorder().unwrap();
                rec.seed_barrier();
                rec.begin_immutable_conditional_at(control.as_ref(), 0);
                f.partial(&rec, k);
                f.reduce(&rec, k);
                rec.end_conditional();
                rec.publish_host_reads();
                rec.finish().unwrap();
                assert_eq!(
                    read(f.partial.as_ref()),
                    before_partial,
                    "SKIP including shared slot"
                );
                assert_eq!(read(f.ff.inner.output.as_ref()), before_output);
                assert_eq!(read(control.as_ref()), vec![0; 16]);
                let immutable = f.readonly();
                sample(&be, &f, k);
                errors.merge(f.verify(&expected, &source, 0, &immutable));
                println!(
                    "SPLITK_V2_MASK profile={} rows=8 shared={shared} chunk={chunk} tail={} masks_exact=true last_partial_negative=true skip_exact=true maxerr={errors:?}",
                    p.name,
                    p.nff % chunk
                );
            }
        }
    }
}

fn frame(
    be: &VulkanBackend,
    fixtures: &[SplitFixture],
    k: SplitKernels,
    candidate: bool,
    query: &Timestamps<'_>,
    role_control: Option<Kernels>,
) -> RecordedCmd {
    let rec = be.recorder_persistent().unwrap();
    assert!(
        !rec.no_barrier
            && !rec.full_barrier
            && !rec.prof_ops
            && !rec.stages
            && rec.timeline_query.borrow().is_none()
    );
    rec.seed_barrier();
    query.begin(&rec);
    for _ in 0..FRAME_REPEATS {
        for f in fixtures {
            if let Some(control) = role_control.filter(|_| !candidate) {
                f.ff.record(&rec, Arm::PipelinePartialHitNoEntry, control);
            } else {
                f.record(&rec, candidate, k);
            }
            rec.global_barrier();
        }
    }
    query.end(&rec);
    rec.publish_host_reads();
    rec.finish_record().unwrap()
}

fn module(rows: usize) {
    let chunk = std::env::var("MOE4ALL_SPLITK_CHUNK")
        .map(|value| value.parse::<usize>().expect("explicit split-K chunk"))
        .unwrap_or(64);
    assert!(CHUNKS.contains(&chunk));
    let be = split_backend();
    let query = Timestamps::new(&be);
    let shared = std::env::var("MOE4ALL_SPLITK_SHARED").is_ok_and(|v| v == "1");
    let role_control = std::env::var("MOE4ALL_SPLITK_CONTROL").is_ok_and(|v| v == "role");
    for p in native_profiles() {
        if role_control && p.gate_up != DType::Q5K {
            continue;
        }
        let source = Sources::new(&be, p, 0); // THREE banks, not a fictitious cold full model.
        let k = SplitKernels::new(&be, p, chunk, shared);
        let control = role_control.then(|| Kernels::tree64_gu(&be, p, 256, 2));
        for mixed in [false, true] {
            let mut fixtures = Vec::new();
            let mut expected = Vec::new();
            for bank in 0..source.banks {
                let mut f = SplitFixture::new(&be, &source, p, rows, shared, chunk);
                expected.push(f.ff.reference(&be, &source, bank));
                f.prepare(&source, bank, mixed, false);
                let readonly = f.readonly();
                sample(&be, &f, k);
                f.verify(expected.last().unwrap(), &source, bank, &readonly);
                fixtures.push(f);
            }
            let baseline = frame(&be, &fixtures, k, false, &query, control);
            let candidate = frame(&be, &fixtures, k, true, &query, None);
            for recording in [&baseline, &candidate] {
                let start = std::time::Instant::now();
                while start.elapsed().as_millis() < 150 {
                    recording.replay().unwrap();
                }
            }
            let (base, new) = measure_pair(
                &be,
                &baseline,
                &candidate,
                &query,
                fixtures.len() * FRAME_REPEATS,
            );
            // No full BAR readback BETWEEN timed pairs. Fresh candidate-only replay
            // after timing prevents the baseline leaving useful Y/partials/weights.
            for (bank, f) in fixtures.iter_mut().enumerate() {
                f.prepare(&source, bank, mixed, false);
            }
            let readonly: Vec<_> = fixtures.iter().map(SplitFixture::readonly).collect();
            candidate.replay().unwrap();
            let mut errors = Errors::default();
            for (bank, f) in fixtures.iter().enumerate() {
                errors.merge(f.verify(&expected[bank], &source, bank, &readonly[bank]));
            }
            let f = &fixtures[0];
            let hits = rows * (p.used - usize::from(mixed));
            let payload: usize = if mixed {
                source.ranges.iter().map(|r| r[MISS_ID].len()).sum()
            } else {
                0
            };
            println!(
                "SPLITK_V2_MODULE profile={} ne={} nff={} rows={rows} chunk={chunk} local=128 mode={} hits={hits} routed={} miss_experts={} jobs={} copy_local=1024 copy_groups=8 payload_bytes_per_module={payload} partial_bytes_per_module={} parts={} active_partial_wgs={} full_UG_repeats=1 reduction_dispatch_in_timing=true M_FFN_in_timing=true role_control={role_control} shared={shared} candidate=splitK_full_H_plus_reduce banks={} frame_repeats={FRAME_REPEATS} source_bytes={} pairs={FRAME_PAIRS} warm_ms_per_arm=150 warmups=2 AB_BA=true baseline_us={base:?} candidate_us={new:?} baseline_median_us={} candidate_median_us={} maxerr={errors:?} atol={ABS_TOL} rtol={REL_TOL} exact_weights_guards=true working_set=three_banks_not_cold_model",
                p.name,
                p.ne,
                p.nff,
                if mixed {
                    "one_expert_three_role_copy"
                } else {
                    "copy_none"
                },
                rows * p.used,
                usize::from(mixed),
                f.ff.inner.job_count,
                f.partial.len_bytes() - 2 * GUARD,
                f.parts(),
                hits * f.parts(),
                fixtures.len(),
                source.span * source.banks,
                median(&base),
                median(&new)
            );
        }
    }
}

#[test]
#[ignore = "parent serial GPU only; two real shapes, copy-none/one-expert-copy, six full-module AB/BA pairs"]
fn gpu_owned_splitk_hit_v2_module_rows1() {
    module(1);
}
#[test]
#[ignore = "parent serial GPU only; two real shapes, full-module timing INCLUDING split reduction"]
fn gpu_owned_splitk_hit_v2_module_rows2() {
    module(2);
}
#[test]
#[ignore = "parent serial GPU only; two real shapes, four rows, no eight-row native timing"]
fn gpu_owned_splitk_hit_v2_module_rows4() {
    module(4);
}
