#![cfg(test)]

//! External-SPIR-V FFN probes. All GPU entry points require explicit parent serialization.
use super::*;

#[path = "recorder_owned_hit_event_tests.rs"]
pub(super) mod hit_event;
#[path = "recorder_owned_sparse_pipeline_tests.rs"]
mod sparse_pipeline;
#[path = "recorder_owned_splitk_hit_tests.rs"]
mod splitk_hit_v2;
#[path = "recorder_owned_tree64_local64_tests.rs"]
mod tree64_local64;
#[path = "recorder_owned_tree64_wide_nr_tests.rs"]
mod tree64_wide_nr;
use crate::ops::ComputeKernel;
use std::path::PathBuf;

const ABS_TOL: f64 = 2e-6;
const REL_TOL: f64 = 1e-3;
const FRAME_PAIRS: usize = 6;

fn fused_backend() -> VulkanBackend {
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev = Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("select an idle GPU"));
    cfg.kernels.vulkan.decode_gpu_resident_copy = Some(true);
    VulkanBackend::new_with(Arc::new(cfg)).unwrap()
}

fn format_name(p: Profile) -> &'static str {
    match (p.gate_up, p.down) {
        (DType::Q4K, DType::Q4K) => "q4k",
        (DType::Q5K, DType::Q5K) => "q5k",
        (DType::Q6K, DType::Q6K) => "q6k",
        (DType::Iq2S, DType::Iq4Nl) => "iq2s",
        (DType::Iq3S, DType::Iq4Nl) => "iq3s",
        _ => panic!("unsupported fused probe profile {p:?}"),
    }
}

fn load_kernel(
    be: &VulkanBackend,
    file: String,
    bindings: usize,
    push: u32,
    sg: bool,
) -> ComputeKernel {
    let root = PathBuf::from(
        std::env::var_os("MOE4ALL_FUSED_PROBE_DIR")
            .expect("MOE4ALL_FUSED_PROBE_DIR must name the externally compiled probe directory"),
    );
    let path = root.join(&file);
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(
        bytes.len() % 4,
        0,
        "SPIR-V word alignment: {}",
        path.display()
    );
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    assert_eq!(
        words.first(),
        Some(&0x0723_0203),
        "SPIR-V magic: {}",
        path.display()
    );
    // The kernel cache requires process-lifetime keys. Only the bounded external variant set leaks.
    let name = Box::leak(format!("fused_probe_{file}").into_boxed_str());
    if sg {
        be.kernel_sg(name, &words, bindings, push, 32)
    } else {
        be.kernel(name, &words, bindings, push)
    }
}

#[derive(Clone, Copy)]
struct Kernels {
    hit: ComputeKernel,
    gu: ComputeKernel,
    roles: ComputeKernel,
    local: usize,
    tiles: usize,
    gu_outputs: usize,
    grid_buffer: bool,
}

impl Kernels {
    fn new(be: &VulkanBackend, p: Profile, local: usize, tiles: usize, shared: bool) -> Self {
        assert!(matches!(local, 128 | 256));
        assert!(matches!(tiles, 1 | 2 | 4));
        assert!(p.nff <= 1024, "external builds use MAX_NFF=1024");
        let fmt = format_name(p);
        let kind = if shared { "shared" } else { "hit" };
        Self {
            hit: load_kernel(
                be,
                format!("owned-{fmt}-{local}-{kind}.spv"),
                4,
                if shared { 64 } else { 40 },
                true,
            ),
            gu: load_kernel(be, format!("owned-{fmt}-{local}-gu.spv"), 4, 40, true),
            roles: load_kernel(be, "owned-role-copy.spv".to_owned(), 2, 20, false),
            local,
            tiles,
            gu_outputs: local / 32,
            grid_buffer: false,
        }
    }

    fn tree64_gu(be: &VulkanBackend, p: Profile, local: usize, nr: usize) -> Self {
        Self::tree64_gu_with_grid(be, p, local, nr, false)
    }

    fn tree64_gu64_grid(be: &VulkanBackend, p: Profile, nr: usize) -> Self {
        Self::tree64_gu_with_grid(be, p, 64, nr, true)
    }

    fn tree64_gu_with_grid(
        be: &VulkanBackend,
        p: Profile,
        local: usize,
        nr: usize,
        grid_buffer: bool,
    ) -> Self {
        assert!(matches!(local, 64 | 128 | 256));
        assert!(matches!(nr, 1 | 2));
        assert!(p.nff <= 1024, "built-in Tree64 uses MAX_NFF=1024");
        let fmt = format_name(p);
        let suffix = if grid_buffer { "_grid_buffer" } else { "" };
        let name = Box::leak(
            format!("fused_probe_builtin_tree64_{fmt}_{local}_nr{nr}{suffix}").into_boxed_str(),
        );
        let spv = if grid_buffer {
            assert_eq!(local, 64);
            let grid = match p.gate_up {
                DType::Iq2S => &be.iq2s_grid,
                DType::Iq3S => &be.iq3s_grid,
                _ => panic!("buffer-grid Tree64 requires IQ2S or IQ3S"),
            };
            assert!(
                grid.is_some(),
                "existing native grid buffer must be available"
            );
            crate::gemm::gpu_owned_miss_gu_tree64_grid_buffer_spv(
                p.gate_up,
                local as u32,
                nr as u32,
            )
        } else {
            crate::gemm::gpu_owned_miss_gu_tree64_spv(p.gate_up, local as u32, nr as u32)
        }
        .expect("registered Tree64 GU format/local/NR/grid");
        let gu = be.kernel_sg(name, spv, 4 + usize::from(grid_buffer), 40, 32);
        let limits = unsafe {
            be.shared
                .instance
                .get_physical_device_properties(be.shared.physical_device)
        }
        .limits;
        let wide = limits.max_compute_work_group_invocations >= 1024
            && limits.max_compute_work_group_size[0] >= 1024;
        Self {
            hit: gu, // Unused placeholder: this factory must never JIT a FullHit pipeline.
            gu,
            roles: be.kernel(
                if wide {
                    "fused_probe_builtin_role_copy1024"
                } else {
                    "fused_probe_builtin_role_copy256"
                },
                crate::gemm::gpu_owned_role_copy_spv(wide),
                2,
                20,
            ),
            local,
            tiles: 1,
            gu_outputs: (local / 64) * nr,
            grid_buffer,
        }
    }
}

struct SharedWeights {
    matrices: [Box<dyn Buffer>; 3],
    expected: [Vec<u8>; 3],
}

impl SharedWeights {
    fn new(be: &VulkanBackend, p: Profile) -> Self {
        let expected: [Vec<u8>; 3] = std::array::from_fn(|role| {
            let mut bytes =
                infr_testkit::synth_weight(DType::Q8_0, p.ne * p.nff, 771 + role as u32);
            bytes.extend(std::iter::repeat_n(POISON, GUARD));
            bytes
        });
        let matrices = std::array::from_fn(|role| {
            let buffer = mapped(be, expected[role].len());
            write(buffer.as_ref(), &expected[role]);
            buffer
        });
        Self { matrices, expected }
    }
}

struct FusedFixture {
    inner: Fixture,
    shared: Option<SharedWeights>,
}

impl FusedFixture {
    fn new(be: &VulkanBackend, source: &Sources, p: Profile, rows: usize, shared: bool) -> Self {
        let mut inner = Fixture::new(be, source, p, rows);
        let shared = shared.then(|| SharedWeights::new(be, p));
        if shared.is_some() {
            let assignments = rows * (p.used + 1);
            inner.gate = mapped(be, assignments * p.nff * 4 + GUARD);
            inner.up = mapped(be, assignments * p.nff * 4 + GUARD);
            inner.activation = mapped(be, assignments * p.nff * 4 + GUARD);
            inner.output = mapped(be, assignments * p.ne * 4 + GUARD);
        }
        Self { inner, shared }
    }

    fn physical_used(&self) -> usize {
        self.inner.p.used + usize::from(self.shared.is_some())
    }

    fn shared_masks(&self) {
        if self.shared.is_none() {
            return;
        }
        let f = &self.inner;
        // Routed IDs remain packed by used; only H masks acquire the shared-slot bit.
        let mut ids = f.routed.clone();
        ids.extend(f.hits.iter().map(|mask| mask | (1 << f.p.used)));
        ids.extend(f.hits.iter().map(|mask| ((1 << f.p.used) - 1) ^ mask));
        write(f.ids.as_ref(), bytemuck::cast_slice(&ids));
    }

    fn prepare(&mut self, source: &Sources, bank: usize, all_hit: bool) {
        if all_hit {
            self.inner.prepare_all_hit(source, bank);
            assert_eq!(
                self.inner.job_count, 0,
                "all-hit fixture must publish zero jobs"
            );
            assert!(self
                .inner
                .hits
                .iter()
                .all(|&mask| mask == (1 << self.inner.p.used) - 1));
        } else {
            self.inner.prepare(source, bank, false);
        }
        self.shared_masks();
    }

    fn projections(&self, rec: &Recorder<'_>, bank: u32, mask: u32) {
        let f = &self.inner;
        if let Some(shared) = &self.shared {
            for (role, output) in [(0, f.gate.as_ref()), (1, f.up.as_ref())] {
                rec.linear_native_id_multi_paged_shared(
                    f.p.gate_up,
                    f.weights.device_addr().unwrap(),
                    0,
                    f.tape.as_ref(),
                    f.ids.as_ref(),
                    f.p.used,
                    role * f.p.used,
                    shared.matrices[role].as_ref(),
                    f.input.as_ref(),
                    false,
                    output,
                    f.p.ne,
                    f.p.nff,
                    f.rows,
                    mask,
                    bank,
                );
            }
        } else {
            f.projections(rec, bank, mask);
        }
    }

    fn activation(&self, rec: &Recorder<'_>) {
        let f = &self.inner;
        rec.silu_mul(
            f.gate.as_ref(),
            f.up.as_ref(),
            f.activation.as_ref(),
            f.rows * self.physical_used() * f.p.nff,
            None,
        );
    }

    fn down(&self, rec: &Recorder<'_>, bank: u32, mask: u32) {
        let f = &self.inner;
        if let Some(shared) = &self.shared {
            rec.linear_native_id_multi_paged_shared(
                f.p.down,
                f.weights.device_addr().unwrap(),
                0,
                f.tape.as_ref(),
                f.ids.as_ref(),
                f.p.used,
                2 * f.p.used,
                shared.matrices[2].as_ref(),
                f.activation.as_ref(),
                true,
                f.output.as_ref(),
                f.p.nff,
                f.p.ne,
                f.rows,
                mask,
                bank,
            );
        } else {
            rec.linear_native_id_multi_paged(
                f.p.down,
                f.weights.device_addr().unwrap(),
                0,
                f.tape.as_ref(),
                f.ids.as_ref(),
                f.p.used,
                2 * f.p.used,
                f.activation.as_ref(),
                true,
                f.output.as_ref(),
                f.p.nff,
                f.p.ne,
                f.rows,
                mask,
                bank,
            );
        }
    }

    fn reference(&mut self, be: &VulkanBackend, source: &Sources, bank: usize) -> Vec<u8> {
        self.inner.prepare(source, bank, true);
        self.shared_masks();
        let rec = be.recorder().unwrap();
        rec.seed_barrier();
        let mask = (1 << self.physical_used()) - 1;
        self.projections(&rec, 0, mask);
        self.activation(&rec);
        self.down(&rec, 0, mask);
        rec.publish_host_reads();
        rec.finish().unwrap();
        let result = read(self.inner.output.as_ref());
        assert!(
            result[..result.len() - GUARD]
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .any(|v| v.is_finite() && v.abs() > 1e-7),
            "nonzero native FFN oracle"
        );
        result
    }

    fn fused(&self, rec: &Recorder<'_>, kernels: Kernels, gu_only: bool) {
        let f = &self.inner;
        assert!(
            !kernels.grid_buffer || gu_only,
            "buffer-grid ABI is GU-only"
        );
        let mut push = vec![
            f.p.ne as u32,
            f.p.nff as u32,
            f.rows as u32,
            f.p.used as u32,
            f.p.used as u32,
            self.physical_used() as u32,
            0,
            f.p.used as u32,
            (2 * f.p.used) as u32,
            kernels.tiles as u32,
        ];
        if !gu_only {
            if let Some(shared) = &self.shared {
                for matrix in &shared.matrices {
                    let address = matrix.device_addr().unwrap();
                    assert_ne!(address, 0);
                    push.extend([address as u32, (address >> 32) as u32]);
                }
            }
        }
        let output = if gu_only { &f.activation } else { &f.output };
        let groups = if gu_only {
            f.rows * f.p.used * f.p.nff.div_ceil(kernels.gu_outputs)
        } else {
            f.rows * self.physical_used() * kernels.tiles
        };
        let mut buffers = Vec::with_capacity(4 + usize::from(kernels.grid_buffer));
        if kernels.grid_buffer {
            let grid = match f.p.gate_up {
                DType::Iq2S => &rec.be.iq2s_grid,
                DType::Iq3S => &rec.be.iq3s_grid,
                _ => panic!("buffer-grid Tree64 requires IQ2S or IQ3S"),
            };
            buffers.push(Recorder::vkb(
                grid.as_ref().expect("native grid buffer").as_ref(),
            ));
        }
        buffers.extend([
            Recorder::vkb(f.input.as_ref()),
            Recorder::vkb(f.ids.as_ref()),
            Recorder::vkb(f.tape.as_ref()),
            Recorder::vkb(output.as_ref()),
        ]);
        rec.dispatch(
            if gu_only { kernels.gu } else { kernels.hit },
            &buffers,
            1,
            bytemuck::cast_slice(&push),
            groups as u32,
        );
    }

    fn role_copy(&self, rec: &Recorder<'_>, kernel: ComputeKernel, phase: u32) {
        let f = &self.inner;
        let push = [ORDINAL as u32, phase, 0, f.p.used as u32, f.p.used as u32];
        rec.dispatch(
            kernel,
            &[
                Recorder::vkb(f.journal.as_ref()),
                Recorder::vkb(f.jobs.as_ref()),
            ],
            0,
            bytemuck::cast_slice(&push),
            8,
        );
    }

    fn record(&self, rec: &Recorder<'_>, arm: Arm, kernels: Kernels) {
        let f = &self.inner;
        match arm {
            Arm::Baseline => {
                if self.shared.is_none() {
                    f.record(rec, Variant::CuPartial, false);
                    return;
                }
                let independent = rec.independent_dispatch_scope();
                f.cu_copy(rec);
                self.projections(rec, 1, 0);
                drop(independent);
                self.activation(rec);
                self.down(rec, 1, 0);
                rec.global_barrier();
            }
            Arm::Full | Arm::Pipeline => {
                let independent = rec.independent_dispatch_scope();
                if matches!(arm, Arm::Full) {
                    f.cu_copy(rec);
                } else {
                    self.role_copy(rec, kernels.roles, 1);
                }
                self.fused(rec, kernels, false);
                drop(independent);
                if matches!(arm, Arm::Pipeline) && f.job_count != 0 {
                    let independent = rec.independent_dispatch_scope();
                    self.role_copy(rec, kernels.roles, 2);
                    self.fused(rec, kernels, true);
                    drop(independent);
                    self.down(rec, 2, 0);
                    return;
                }
            }
            Arm::PipelinePartialHit | Arm::PipelinePartialHitNoEntry => {
                let independent = rec.independent_dispatch_scope();
                self.role_copy(rec, kernels.roles, 1);
                self.projections(rec, 1, 0);
                drop(independent);
                let proof = crate::recorder::owned_role_pipeline::test_phase1_joined(rec);
                self.activation(rec);
                self.down(rec, 1, 0);
                if f.job_count == 0 {
                    return;
                }
                let no_entry = matches!(arm, Arm::PipelinePartialHitNoEntry);
                let selected = crate::recorder::owned_role_pipeline::test_phase2_scope(
                    rec,
                    proof,
                    no_entry,
                    |rec| {
                        self.role_copy(rec, kernels.roles, 2);
                        self.fused(rec, kernels, true);
                    },
                );
                assert_eq!(
                    selected, no_entry,
                    "requested phase2 entry must be selected"
                );
                self.down(rec, 2, 0);
                return;
            }
            Arm::PipelineNativeGu | Arm::PipelineNativeGuNoEntry => {
                let independent = rec.independent_dispatch_scope();
                self.role_copy(rec, kernels.roles, 1);
                self.projections(rec, 1, 0);
                drop(independent);
                let proof = crate::recorder::owned_role_pipeline::test_phase1_joined(rec);
                self.activation(rec);
                self.down(rec, 1, 0);
                if f.job_count == 0 {
                    return;
                }
                let no_entry = matches!(arm, Arm::PipelineNativeGuNoEntry);
                let selected = crate::recorder::owned_role_pipeline::test_phase2_scope(
                    rec,
                    proof,
                    no_entry,
                    |rec| {
                        self.role_copy(rec, kernels.roles, 2);
                        // Native M Gate/Up write G/U scratch, not the ACT consumed by H Down.
                        self.projections(rec, 2, 0);
                    },
                );
                assert_eq!(
                    selected, no_entry,
                    "requested phase2 entry must be selected"
                );
                self.activation(rec);
                self.down(rec, 2, 0);
                return;
            }
            Arm::OrdinaryHit => {
                self.projections(rec, 1, 0);
                self.activation(rec);
                self.down(rec, 1, 0);
                return;
            }
            Arm::FusedHit => {
                self.fused(rec, kernels, false);
                return;
            }
            Arm::OrdinaryGu => {
                self.projections(rec, 2, 0);
                self.activation(rec);
                return;
            }
            Arm::FusedGu => {
                self.fused(rec, kernels, true);
                return;
            }
        }
        if f.job_count != 0 {
            self.projections(rec, 2, 0);
            self.activation(rec);
            self.down(rec, 2, 0);
        }
    }

    fn verify(&self, reference: &[u8]) -> Errors {
        let f = &self.inner;
        let actual = read(f.output.as_ref());
        let errors = compare(
            &actual[..actual.len() - GUARD],
            &reference[..reference.len() - GUARD],
        );
        assert_eq!(
            read(f.weights.as_ref()),
            f.expected_weights,
            "exact payload and weight guards"
        );
        for buffer in [
            &f.gate,
            &f.up,
            &f.activation,
            &f.output,
            &f.commands,
            &f.commands_late,
            &f.dummy,
        ] {
            let bytes = read(buffer.as_ref());
            assert!(
                bytes[bytes.len() - GUARD..].iter().all(|&v| v == POISON),
                "exact scratch/sidecar guard"
            );
        }
        if let Some(shared) = &self.shared {
            for (buffer, expected) in shared.matrices.iter().zip(&shared.expected) {
                assert_eq!(
                    read(buffer.as_ref()),
                    *expected,
                    "exact read-only Q8 shared weights and guards"
                );
            }
        }
        errors
    }
}

#[derive(Clone, Copy, Debug)]
enum Arm {
    Baseline,
    Full,
    Pipeline,
    PipelinePartialHit,
    PipelinePartialHitNoEntry,
    PipelineNativeGu,
    PipelineNativeGuNoEntry,
    OrdinaryHit,
    FusedHit,
    OrdinaryGu,
    FusedGu,
}

#[derive(Clone, Copy, Debug, Default)]
struct Errors {
    max_abs: f64,
    max_rel: f64,
}

impl Errors {
    fn merge(&mut self, other: Self) {
        self.max_abs = self.max_abs.max(other.max_abs);
        self.max_rel = self.max_rel.max(other.max_rel);
    }
}

fn compare(actual: &[u8], expected: &[u8]) -> Errors {
    assert_eq!(actual.len(), expected.len());
    let mut errors = Errors::default();
    for (index, (actual, expected)) in actual
        .chunks_exact(4)
        .zip(expected.chunks_exact(4))
        .enumerate()
    {
        let actual = f32::from_le_bytes(actual.try_into().unwrap()) as f64;
        let expected = f32::from_le_bytes(expected.try_into().unwrap()) as f64;
        let delta = (actual - expected).abs();
        assert!(
            actual.is_finite()
                && expected.is_finite()
                && delta <= ABS_TOL + REL_TOL * expected.abs(),
            "FFN index={index} actual={actual} reference={expected} abs_error={delta} atol={ABS_TOL} rtol={REL_TOL}"
        );
        errors.max_abs = errors.max_abs.max(delta);
        errors.max_rel = errors.max_rel.max(delta / expected.abs().max(1e-8));
    }
    errors
}

fn small_profiles() -> [Profile; 5] {
    std::array::from_fn(|i| {
        let (name, gate_up, down, used) = match i {
            0 => ("q4k", DType::Q4K, DType::Q4K, 8),
            1 => ("q5k", DType::Q5K, DType::Q5K, 8),
            2 => ("q6k", DType::Q6K, DType::Q6K, 8),
            3 => ("iq2s_iq4nl", DType::Iq2S, DType::Iq4Nl, 10),
            _ => ("iq3s_iq4nl", DType::Iq3S, DType::Iq4Nl, 8),
        };
        Profile {
            name,
            ne: 256,
            nff: 256,
            used,
            gate_up,
            down,
        }
    })
}

fn native_profiles() -> [Profile; 2] {
    [
        Profile {
            name: "q5k_2048_512",
            ne: 2048,
            nff: 512,
            used: 8,
            gate_up: DType::Q5K,
            down: DType::Q5K,
        },
        Profile {
            name: "iq2s_iq4nl_2560_640",
            ne: 2560,
            nff: 640,
            used: 10,
            gate_up: DType::Iq2S,
            down: DType::Iq4Nl,
        },
    ]
}

fn sample_fused(be: &VulkanBackend, fixture: &FusedFixture, arm: Arm, kernels: Kernels) {
    let rec = be.recorder().unwrap();
    assert!(!rec.no_barrier && !rec.full_barrier && !rec.prof_ops);
    rec.seed_barrier();
    fixture.record(&rec, arm, kernels);
    rec.publish_host_reads();
    rec.finish().unwrap();
}

#[test]
#[ignore = "parent serial GPU; built-in SDK Tree64 GU only, exact ACT, no FullHit or Down pipeline"]
fn gpu_owned_tree64_gu_correctness_rows_1_to_8() {
    let be = fused_backend();
    for small in small_profiles() {
        let grid = matches!(small.gate_up, DType::Iq2S | DType::Iq3S);
        let native = Profile {
            ne: if grid { 2560 } else { 2048 },
            nff: if grid { 640 } else { 512 },
            ..small
        };
        for (p, max_rows, geometry) in [(small, 8, "small"), (native, 2, "native")] {
            // Correctness inputs only: the constructor's minimum is exactly three banks.
            let source = Sources::new(&be, p, 0);
            assert_eq!(source.banks, 3);
            let variants = [(128, 1), (128, 2), (256, 1), (256, 2)]
                .map(|(local, nr)| (Kernels::tree64_gu(&be, p, local, nr), nr));
            for rows in 1..=max_rows {
                let bank = (rows - 1) % source.banks;
                let mut fixture = FusedFixture::new(&be, &source, p, rows, false);
                fixture.inner.prepare(&source, bank, true);
                // Use the ordinary selected native GU path and its existing ACT arithmetic,
                // without invoking reference(), which would also construct a Down pipeline.
                let rec = be.recorder().unwrap();
                rec.seed_barrier();
                fixture.projections(&rec, 0, (1u32 << p.used) - 1);
                fixture.activation(&rec);
                rec.publish_host_reads();
                rec.finish().unwrap();
                let expected = read(fixture.inner.activation.as_ref());
                let reference_kernels: Vec<_> = be
                    .built_kernel_names()
                    .into_iter()
                    .filter(|name| name.starts_with("native_idm_") && name.contains(format_name(p)))
                    .collect();
                println!(
                    "OWNED_TREE64_GU_ORACLE profile={} geometry={geometry} ne={} nff={} rows={rows} used={} native_kernels={reference_kernels:?} reference=selected_native_GU_plus_silu exact_bytes_required=true",
                    p.name, p.ne, p.nff, p.used
                );
                assert!(
                    !reference_kernels.is_empty(),
                    "native GU oracle must record a kernel"
                );
                assert!(
                    reference_kernels.iter().all(|name| !name.contains("_sg")),
                    "selected native path is an SG rather than Tree64 reduction: profile={p:?} rows={rows} kernels={reference_kernels:?}; do not relax ACT equality or silently force a different oracle"
                );
                assert!(expected[expected.len() - GUARD..]
                    .iter()
                    .all(|&v| v == POISON));
                assert!(
                    expected[..expected.len() - GUARD]
                        .chunks_exact(4)
                        .all(|word| f32::from_le_bytes(word.try_into().unwrap()).is_finite()),
                    "finite native ACT reference"
                );
                for (kernels, nr) in variants {
                    // All weights are resident, but H/M and every scratch poison are reset.
                    fixture.inner.prepare(&source, bank, true);
                    let f = &fixture.inner;
                    let immutable: [(&str, &dyn Buffer); 11] = [
                        ("input", f.input.as_ref()),
                        ("ids_masks", f.ids.as_ref()),
                        ("tape", f.tape.as_ref()),
                        ("gate", f.gate.as_ref()),
                        ("up", f.up.as_ref()),
                        ("down_output", f.output.as_ref()),
                        ("journal", f.journal.as_ref()),
                        ("jobs", f.jobs.as_ref()),
                        ("commands", f.commands.as_ref()),
                        ("commands_late", f.commands_late.as_ref()),
                        ("dummy", f.dummy.as_ref()),
                    ];
                    let before: Vec<_> =
                        immutable.iter().map(|(_, buffer)| read(*buffer)).collect();
                    // FusedGu dispatches only GU; hit is a placeholder, never dispatched.
                    sample_fused(&be, &fixture, Arm::FusedGu, kernels);
                    let actual = read(f.activation.as_ref());
                    let width = p.nff * 4;
                    let mut hits = 0;
                    let mut misses = 0;
                    for assignment in 0..rows * p.used {
                        let hit = f.hits[assignment / p.used] & (1 << (assignment % p.used)) != 0;
                        let range = assignment * width..(assignment + 1) * width;
                        if hit {
                            hits += 1;
                            assert!(
                                actual[range]
                                    .chunks_exact(4)
                                    .all(|word| word == 0.125f32.to_le_bytes()),
                                "GU wrote H assignment={assignment} profile={p:?} rows={rows} local={} nr={nr}",
                                kernels.local
                            );
                        } else {
                            misses += 1;
                            if let Some(byte) = actual[range.clone()]
                                .iter()
                                .zip(&expected[range.clone()])
                                .position(|(a, b)| a != b)
                            {
                                let at = range.start + byte / 4 * 4;
                                let got =
                                    u32::from_le_bytes(actual[at..at + 4].try_into().unwrap());
                                let want =
                                    u32::from_le_bytes(expected[at..at + 4].try_into().unwrap());
                                panic!(
                                    "Tree64 M ACT not byte-exact profile={p:?} rows={rows} local={} nr={nr} bank={bank} assignment={assignment} element={} actual_bits={got:08x} reference_bits={want:08x} native_kernels={reference_kernels:?}; no tolerance permitted",
                                    kernels.local,
                                    byte / 4
                                );
                            }
                        }
                    }
                    assert!(hits > 0 && misses > 0, "exercise both mask banks' outcomes");
                    assert!(
                        actual[actual.len() - GUARD..].iter().all(|&v| v == POISON),
                        "ACT guard exact"
                    );
                    assert_eq!(
                        read(f.weights.as_ref()),
                        f.expected_weights,
                        "weights and every payload guard exact"
                    );
                    for ((name, buffer), seed) in immutable.iter().zip(&before) {
                        assert_eq!(read(*buffer), *seed, "GU-only must not change {name}");
                    }
                    println!(
                        "OWNED_TREE64_GU_EXACT profile={} geometry={geometry} ne={} nff={} rows={rows} used={} local={} nr={nr} gu_outputs={} bank={bank} source_banks=3 artifact=builtin_sdk_spv M_ACT_byteexact=true H_untouched_0_125=true guards_weights_exact=true full_hit_pipeline=false down_pipeline=false",
                        p.name, p.ne, p.nff, p.used, kernels.local, kernels.gu_outputs
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "parent serial GPU; external fused SPIR-V; reduced geometry, rows 1 through 8"]
fn gpu_owned_fused_correctness_rows_1_to_8() {
    let be = fused_backend();
    for p in small_profiles() {
        let source = Sources::new(&be, p, 0);
        let mut errors = Errors::default();
        for local in [128, 256] {
            let kernels = Kernels::new(&be, p, local, 1, false);
            for rows in 1..=8 {
                let mut fixture = FusedFixture::new(&be, &source, p, rows, false);
                let bank = rows % source.banks;
                let expected = fixture.reference(&be, &source, bank);
                for all_hit in [false, true] {
                    for arm in [Arm::Full, Arm::Pipeline, Arm::PipelinePartialHit] {
                        fixture.prepare(&source, bank, all_hit);
                        sample_fused(&be, &fixture, arm, kernels);
                        errors.merge(fixture.verify(&expected));
                    }
                }
            }
        }
        println!(
            "OWNED_FUSED_CORRECT profile={} rows=1..8 local=128,256 tiles=1 all_hit_and_mixed=true maxerr={errors:?} atol={ABS_TOL} rtol={REL_TOL} exact_weights_guards=true",
            p.name
        );
    }
}

#[test]
#[ignore = "parent serial GPU; real shared Q8 BDAs and physical used+1 stride, rows 1 through 8"]
fn gpu_owned_fused_shared_q8_correctness_rows_1_to_8() {
    let be = fused_backend();
    for p in [small_profiles()[1], small_profiles()[3]] {
        let source = Sources::new(&be, p, 0);
        let mut errors = Errors::default();
        for rows in 1..=8 {
            let mut fixture = FusedFixture::new(&be, &source, p, rows, true);
            let bank = rows % source.banks;
            let expected = fixture.reference(&be, &source, bank);
            for (local, tiles) in [(128, 1), (256, 4)] {
                let kernels = Kernels::new(&be, p, local, tiles, true);
                for all_hit in [false, true] {
                    for arm in [Arm::Full, Arm::Pipeline, Arm::PipelinePartialHit] {
                        fixture.prepare(&source, bank, all_hit);
                        sample_fused(&be, &fixture, arm, kernels);
                        errors.merge(fixture.verify(&expected));
                    }
                }
            }
        }
        println!(
            "OWNED_FUSED_SHARED_Q8 profile={} rows=1..8 routed_stride={} physical_stride={} maxerr={errors:?} exact_weights_guards=true",
            p.name,
            p.used,
            p.used + 1
        );
    }
}

#[test]
#[ignore = "parent serial GPU; isolated masked writes and untouched inactive output canaries"]
fn gpu_owned_fused_masked_hit_and_gu_outputs() {
    let be = fused_backend();
    for p in [small_profiles()[1], small_profiles()[3]] {
        let source = Sources::new(&be, p, 0);
        let mut fixture = FusedFixture::new(&be, &source, p, 8, false);
        let expected_output = fixture.reference(&be, &source, 0);
        let expected_activation = read(fixture.inner.activation.as_ref());
        for (local, tiles) in [(128, 1), (256, 4)] {
            let kernels = Kernels::new(&be, p, local, tiles, false);
            for arm in [Arm::FusedHit, Arm::FusedGu] {
                // Resident weights isolate masks and striding from the payload-copy variants.
                fixture.inner.prepare(&source, 0, true);
                sample_fused(&be, &fixture, arm, kernels);
                let f = &fixture.inner;
                let hit_only = matches!(arm, Arm::FusedHit);
                let (buffer, expected, width) = if hit_only {
                    (&f.output, &expected_output, p.ne * 4)
                } else {
                    (&f.activation, &expected_activation, p.nff * 4)
                };
                let actual = read(buffer.as_ref());
                let mut errors = Errors::default();
                for assignment in 0..f.rows * p.used {
                    let hit = f.hits[assignment / p.used] & (1 << (assignment % p.used)) != 0;
                    let range = assignment * width..(assignment + 1) * width;
                    if hit == hit_only {
                        errors.merge(compare(&actual[range.clone()], &expected[range]));
                    } else {
                        assert!(
                            actual[range]
                                .chunks_exact(4)
                                .all(|word| word == 0.125f32.to_le_bytes()),
                            "inactive assignment changed"
                        );
                    }
                }
                assert!(actual[actual.len() - GUARD..].iter().all(|&v| v == POISON));
                assert_eq!(read(f.weights.as_ref()), f.expected_weights);
                println!(
                    "OWNED_FUSED_MASKED profile={} arm={arm:?} rows=8 local={local} tiles={tiles} maxerr={errors:?} inactive_exact=true weights_exact=true",
                    p.name
                );
            }
        }
    }
}

#[test]
#[ignore = "parent serial GPU; exact phase-1 Gate/Up and phase-2 Down copy boundaries"]
fn gpu_owned_fused_role_copy_exact_phase_boundaries() {
    let be = fused_backend();
    let kernel = load_kernel(&be, "owned-role-copy.spv".to_owned(), 2, 20, false);
    for p in [small_profiles()[1], small_profiles()[3]] {
        let source = Sources::new(&be, p, 0);
        let mut fixture = FusedFixture::new(&be, &source, p, 1, false);
        fixture.prepare(&source, 0, false);
        let f = &fixture.inner;
        let before = read(f.weights.as_ref());
        for phase in [1, 2] {
            let rec = be.recorder().unwrap();
            rec.seed_barrier();
            fixture.role_copy(&rec, kernel, phase);
            rec.global_barrier();
            rec.publish_host_reads();
            rec.finish().unwrap();
            let mut expected = f.expected_weights.clone();
            if phase == 1 {
                for (expert, range) in source.ranges[2].iter().enumerate() {
                    if missing(2, expert) {
                        expected[range.clone()].copy_from_slice(&before[range.clone()]);
                    }
                }
            }
            assert_eq!(
                read(f.weights.as_ref()),
                expected,
                "exact phase={phase} profile={p:?}"
            );
            assert_eq!(
                read(f.jobs.as_ref()),
                bytemuck::cast_slice::<u32, u8>(&f.job_words)
            );
        }
    }
}

fn record_frame(
    be: &VulkanBackend,
    fixtures: &[FusedFixture],
    arm: Arm,
    kernels: Kernels,
    query: &Timestamps<'_>,
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
    for fixture in fixtures {
        fixture.record(&rec, arm, kernels);
        rec.global_barrier();
    }
    query.end(&rec);
    rec.publish_host_reads();
    rec.finish_record().unwrap()
}

fn measure_pair(
    be: &VulkanBackend,
    baseline: &RecordedCmd,
    candidate: &RecordedCmd,
    query: &Timestamps<'_>,
    layers: usize,
) -> (Vec<f64>, Vec<f64>) {
    for recording in [baseline, candidate] {
        recording.replay().unwrap();
        recording.replay().unwrap();
    }
    let mut base = Vec::new();
    let mut compared = Vec::new();
    for pair in 0..FRAME_PAIRS {
        let order = if pair % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for candidate_arm in order {
            (if candidate_arm { candidate } else { baseline })
                .replay()
                .unwrap();
            let us = query.elapsed(be) / layers as f64;
            if candidate_arm {
                compared.push(us);
            } else {
                base.push(us);
            }
        }
    }
    (base, compared)
}

fn continuous_frames(rows: usize) {
    let be = fused_backend();
    let query = Timestamps::new(&be);
    for p in native_profiles() {
        let source = Sources::new(&be, p, SOURCE_RING_BYTES);
        assert!(source.banks * source.miss_bytes > 128 * 1024 * 1024);
        for all_hit in [false, true] {
            let mut fixtures = Vec::new();
            let mut expected = Vec::new();
            for bank in 0..source.banks {
                let mut fixture = FusedFixture::new(&be, &source, p, rows, false);
                expected.push(fixture.reference(&be, &source, bank));
                fixture.prepare(&source, bank, all_hit);
                fixtures.push(fixture);
            }
            for local in [128, 256] {
                for tiles in [1, 2, 4] {
                    let kernels = Kernels::new(&be, p, local, tiles, false);
                    // Qualify native geometry with poisoned output, before any timed frame warms it.
                    for arm in [Arm::Full, Arm::Pipeline, Arm::PipelinePartialHit] {
                        // PartialHit uses GU_ONLY, whose workgroup grid does not depend on tiles.
                        if matches!(arm, Arm::PipelinePartialHit) && tiles != 1 {
                            continue;
                        }
                        fixtures[0].prepare(&source, 0, all_hit);
                        sample_fused(&be, &fixtures[0], arm, kernels);
                        fixtures[0].verify(&expected[0]);
                    }
                    let baseline = record_frame(&be, &fixtures, Arm::Baseline, kernels, &query);
                    for arm in [Arm::Full, Arm::Pipeline, Arm::PipelinePartialHit] {
                        if matches!(arm, Arm::PipelinePartialHit) && tiles != 1 {
                            continue;
                        }
                        let candidate = record_frame(&be, &fixtures, arm, kernels, &query);
                        let (base_us, candidate_us) =
                            measure_pair(&be, &baseline, &candidate, &query, fixtures.len());
                        // The last timed arm may be baseline. Verify an explicit candidate replay.
                        candidate.replay().unwrap();
                        let mut errors = Errors::default();
                        for (fixture, reference) in fixtures.iter().zip(&expected) {
                            errors.merge(fixture.verify(reference));
                        }
                        let active_hits: usize = fixtures[0]
                            .inner
                            .hits
                            .iter()
                            .map(|mask| mask.count_ones() as usize)
                            .sum();
                        let partial_hit = matches!(arm, Arm::PipelinePartialHit);
                        let hit_ffn = if partial_hit { "ordinary" } else { "fused" };
                        let active_fused_hit_wgs =
                            if partial_hit { 0 } else { active_hits * tiles };
                        let gu_repeats = if partial_hit { 1 } else { tiles };
                        println!(
                            "OWNED_FUSED_CONTINUOUS profile={} ne={} nff={} rows={rows} local={local} tiles={tiles} all_hit={all_hit} jobs={} hit_ffn={hit_ffn} active_fused_hit_wgs={active_fused_hit_wgs} gu_repeats={gu_repeats} baseline=CuPartial candidate={arm:?} layers={} source_bytes={} payload_bytes_per_frame={} pairs={FRAME_PAIRS} order=AB_BA baseline_us={base_us:?} candidate_us={candidate_us:?} baseline_median_us={} candidate_median_us={} maxerr={errors:?} atol={ABS_TOL} rtol={REL_TOL} exact_weights_guards=true",
                            p.name,
                            p.ne,
                            p.nff,
                            fixtures[0].inner.job_count,
                            fixtures.len(),
                            source.span * source.banks,
                            if all_hit {
                                0
                            } else {
                                source.banks * source.miss_bytes
                            },
                            median(&base_us),
                            median(&candidate_us)
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "parent serial GPU; Q5K 2048/512 and IQ2S/IQ4NL 2560/640; six AB/BA frame pairs"]
fn gpu_owned_fused_continuous_frames() {
    continuous_frames(1);
}

#[test]
#[ignore = "parent opt-in after single-row probe; identical bounded sweep with two rows"]
fn gpu_owned_fused_continuous_frames_rows2() {
    continuous_frames(2);
}

fn continuous_partial_hit_frames(rows: usize) {
    let be = fused_backend();
    let query = Timestamps::new(&be);
    let tiles = 1;
    let arm = Arm::PipelinePartialHit;
    for p in native_profiles() {
        let source = Sources::new(&be, p, SOURCE_RING_BYTES);
        assert!(source.banks * source.miss_bytes > 128 * 1024 * 1024);
        for all_hit in [false, true] {
            let mut fixtures = Vec::new();
            let mut expected = Vec::new();
            for bank in 0..source.banks {
                let mut fixture = FusedFixture::new(&be, &source, p, rows, false);
                expected.push(fixture.reference(&be, &source, bank));
                fixture.prepare(&source, bank, all_hit);
                fixtures.push(fixture);
            }
            for local in [128, 256] {
                let kernels = Kernels::new(&be, p, local, tiles, false);
                // Qualify both arms from poisoned output, without dispatching fused full-hit.
                for qualify in [Arm::Baseline, arm] {
                    fixtures[0].prepare(&source, 0, all_hit);
                    sample_fused(&be, &fixtures[0], qualify, kernels);
                    fixtures[0].verify(&expected[0]);
                }
                let baseline = record_frame(&be, &fixtures, Arm::Baseline, kernels, &query);
                let candidate = record_frame(&be, &fixtures, arm, kernels, &query);
                let (base_us, candidate_us) =
                    measure_pair(&be, &baseline, &candidate, &query, fixtures.len());
                // Read back only after all timed pairs, verifying each complete frame explicitly.
                baseline.replay().unwrap();
                let mut baseline_errors = Errors::default();
                for (fixture, reference) in fixtures.iter().zip(&expected) {
                    baseline_errors.merge(fixture.verify(reference));
                }
                candidate.replay().unwrap();
                let mut errors = Errors::default();
                for (fixture, reference) in fixtures.iter().zip(&expected) {
                    errors.merge(fixture.verify(reference));
                }
                println!(
                    "OWNED_FUSED_PARTIAL_HIT_CONTINUOUS profile={} ne={} nff={} rows={rows} local={local} tiles={tiles} all_hit={all_hit} jobs={} hit_ffn=ordinary active_fused_hit_wgs=0 gu_repeats=1 baseline=CuPartial candidate={arm:?} layers={} source_bytes={} payload_bytes_per_frame={} warm_replays_per_arm=2 pairs={FRAME_PAIRS} order=AB_BA baseline_us={base_us:?} candidate_us={candidate_us:?} baseline_median_us={} candidate_median_us={} baseline_maxerr={baseline_errors:?} maxerr={errors:?} atol={ABS_TOL} rtol={REL_TOL} exact_weights_guards=true",
                    p.name,
                    p.ne,
                    p.nff,
                    fixtures[0].inner.job_count,
                    fixtures.len(),
                    source.span * source.banks,
                    if all_hit {
                        0
                    } else {
                        source.banks * source.miss_bytes
                    },
                    median(&base_us),
                    median(&candidate_us)
                );
            }
        }
    }
}

#[test]
#[ignore = "parent serial GPU; only PipelinePartialHit vs CuPartial; tiles1; six warm AB/BA pairs"]
fn gpu_owned_fused_partial_hit_continuous_rows1() {
    continuous_partial_hit_frames(1);
}

#[test]
#[ignore = "parent serial GPU; same narrow PipelinePartialHit comparison with two rows"]
fn gpu_owned_fused_partial_hit_continuous_rows2() {
    continuous_partial_hit_frames(2);
}

#[test]
#[ignore = "parent opt-in attribution; isolated hit FFN and miss GU/ACT, no copy in either arm"]
fn gpu_owned_fused_module_diagnostics() {
    let be = fused_backend();
    let query = Timestamps::new(&be);
    for p in native_profiles() {
        let source = Sources::new(&be, p, SOURCE_RING_BYTES);
        let mut fixtures = Vec::new();
        let mut expected_outputs = Vec::new();
        let mut expected_activations = Vec::new();
        for bank in 0..source.banks {
            let mut fixture = FusedFixture::new(&be, &source, p, 1, false);
            expected_outputs.push(fixture.reference(&be, &source, bank));
            expected_activations.push(read(fixture.inner.activation.as_ref()));
            fixtures.push(fixture);
        }
        for (local, tiles) in [(128, 1), (256, 4)] {
            let kernels = Kernels::new(&be, p, local, tiles, false);
            for (ordinary, fused, label) in [
                (Arm::OrdinaryHit, Arm::FusedHit, "hit_only"),
                (Arm::OrdinaryGu, Arm::FusedGu, "gu_only"),
            ] {
                let baseline = record_frame(&be, &fixtures, ordinary, kernels, &query);
                let candidate = record_frame(&be, &fixtures, fused, kernels, &query);
                let (base_us, candidate_us) =
                    measure_pair(&be, &baseline, &candidate, &query, fixtures.len());
                candidate.replay().unwrap();
                let mut errors = Errors::default();
                for (index, fixture) in fixtures.iter().enumerate() {
                    let f = &fixture.inner;
                    let (buffer, expected, width) = if matches!(fused, Arm::FusedHit) {
                        (&f.output, &expected_outputs[index], p.ne * 4)
                    } else {
                        (&f.activation, &expected_activations[index], p.nff * 4)
                    };
                    let actual = read(buffer.as_ref());
                    for (assignment, &expert) in f.routed.iter().enumerate() {
                        let hit = (0..3).all(|role| !missing(role, expert as usize));
                        if hit == matches!(fused, Arm::FusedHit) {
                            let range = assignment * width..(assignment + 1) * width;
                            errors.merge(compare(&actual[range.clone()], &expected[range]));
                        }
                    }
                    assert!(actual[actual.len() - GUARD..].iter().all(|&v| v == POISON));
                    assert_eq!(read(f.weights.as_ref()), f.expected_weights);
                }
                let active_hits: usize = fixtures[0]
                    .inner
                    .hits
                    .iter()
                    .map(|mask| mask.count_ones() as usize)
                    .sum();
                let active_wgs = if matches!(fused, Arm::FusedHit) {
                    active_hits * tiles
                } else {
                    (p.used - active_hits) * p.nff.div_ceil(local / 32)
                };
                println!(
                    "OWNED_FUSED_MODULE profile={} module={label} local={local} tiles={tiles} active_wgs_per_layer={active_wgs} gu_repeats={} copy=none layers={} pairs={FRAME_PAIRS} baseline_us={base_us:?} candidate_us={candidate_us:?} baseline_median_us={} candidate_median_us={} maxerr={errors:?}",
                    p.name,
                    if matches!(fused, Arm::FusedHit) {
                        tiles
                    } else {
                        1
                    },
                    fixtures.len(),
                    median(&base_us),
                    median(&candidate_us)
                );
            }
        }
    }
}
