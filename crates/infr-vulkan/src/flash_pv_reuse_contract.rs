//! Recorder child: embedded production PV-reuse routes, without research artifacts or timing.
use super::*;
use infr_core::backend::BufferUsage;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

const HD: usize = 256;
const NKV: usize = 2;
const GUARD: usize = 64;
const INPUT_CANARY: u16 = 0x7b55;
const SENTINEL: f32 = -777.0;
const CW4: &str = "attn_flash_reg_hd256_br64_cw4";
const QK4: &str = "attn_flash_reg_hd256_br128_f16score_qk4";
const CW4_REUSE: &str = "attn_flash_reg_hd256_br64_cw4_pvreuse";
const QK4_REUSE: &str = "attn_flash_reg_hd256_br128_f16score_qk4_pvreuse";
const COMBINE: &str = "attn_flash_combine_hd256";

#[derive(Clone, Copy, Debug)]
struct Case {
    rows: usize,
    kv: usize,
    nh: usize,
}

impl Case {
    fn deep(self) -> bool {
        self.kv >= 65_536
    }

    fn padded(self) -> usize {
        self.rows.div_ceil(128) * 128
    }

    fn tile(self) -> usize {
        if self.deep() {
            128
        } else {
            64
        }
    }

    fn splits(self) -> usize {
        // The parent's original whole-ubatch fixture uses this same production helper.
        flash_split_count(
            None,
            (self.padded() / self.tile() * self.nh) as u32,
            self.kv,
            self.deep(),
        ) as usize
    }

    fn lengths(self) -> [usize; 4] {
        let scalar = self.splits() * self.padded() * self.nh;
        [scalar * HD, scalar, scalar, self.padded() * self.nh * HD]
    }

    fn baseline(self) -> (&'static str, &'static [u32]) {
        if self.deep() {
            (
                QK4,
                crate::gemm::attn_flash_reg_hd256_br128_f16score_qk4_spv(),
            )
        } else {
            (CW4, crate::gemm::attn_flash_reg_hd256_br64_cw4_spv())
        }
    }

    fn route(self, enabled: bool) -> (&'static str, &'static [u32]) {
        if enabled && self.rows >= 512 {
            crate::gemm::flash_pv_reuse_kernel(self.deep())
        } else {
            self.baseline()
        }
    }
}

const SHORT: [Case; 5] = [
    Case {
        rows: 128,
        kv: 2048,
        nh: 16,
    },
    Case {
        rows: 512,
        kv: 2048,
        nh: 16,
    },
    Case {
        rows: 512,
        kv: 2051,
        nh: 24,
    },
    Case {
        rows: 512,
        kv: 2048,
        nh: 32,
    },
    Case {
        rows: 129,
        kv: 2051,
        nh: 16,
    },
];
const LONG: [Case; 2] = [
    Case {
        rows: 128,
        kv: 70_019,
        nh: 16,
    },
    Case {
        rows: 512,
        kv: 70_019,
        nh: 24,
    },
];

fn backend() -> VulkanBackend {
    let mut cfg = infr_core::config::Config::default();
    cfg.kernels.vulkan.flash_pv_reuse = false;
    cfg.kernels.vulkan.flash_splits = None;
    cfg.kernels.vulkan.pipeline_cache_disk = false;
    let be = VulkanBackend::new_with(Arc::new(cfg))
        .expect("PV-reuse contract requires an available qualified Vulkan device");
    let caps = be.caps();
    // These are bounded correctness samples for the measured route, not a portable-device bench.
    assert!(
        cfg!(target_os = "windows")
            && caps.name.contains("7900 XTX")
            && !caps.integrated
            && caps.f16
            && caps.f16_coopmat()
            && caps.subgroup_min <= 32
            && caps.subgroup_max >= 32
            && be.prefers_hd256_prefill_cw4()
            && be.prefers_hd256_prefill_br128_f16score()
            && be.max_shared_memory_bytes() >= 29_440
            && be.max_shared_memory_bytes() < 128 * crate::FLASH_REG_SHARED_PER_ROW,
        "PV-reuse samples require Windows RX 7900 XTX, f16 16x16x16 coopmat, pinned wave32 and the BR64/CW4 shared-memory tier; got {caps:?}"
    );
    be
}

fn half_values(n: usize, padded: usize, seed: u32, scale: f32) -> Vec<u16> {
    assert!(padded >= n);
    // Same nonperiodic generator and F16 rounding as the original attention fixture.
    let mut values: Vec<u16> = (0..n)
        .map(|i| {
            let mut x = (i as u32).wrapping_add(seed.wrapping_mul(0x9e37_79b9));
            x ^= x >> 16;
            x = x.wrapping_mul(0x7feb_352d);
            x ^= x >> 15;
            half::f16::from_f32(((x >> 8) as f32 / 16_777_216.0 - 0.5) * scale).to_bits()
        })
        .collect();
    values.resize(padded, 0);
    values.resize(padded + GUARD, INPUT_CANARY);
    values
}

fn upload(be: &VulkanBackend, bytes: &[u8], usage: BufferUsage) -> Box<dyn Buffer> {
    let buffer = be
        .alloc(bytes.len(), usage)
        .expect("bounded fixture allocation");
    be.upload(buffer.as_ref(), bytes).expect("fixture upload");
    buffer
}

fn download(be: &VulkanBackend, buffer: &dyn Buffer) -> Vec<f32> {
    let mut words = vec![0.0; buffer.len_bytes() / 4];
    be.download(buffer, bytemuck::cast_slice_mut(&mut words))
        .expect("fixture readback");
    words
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

fn exact(label: &str, off: &[u8], on: &[u8]) {
    assert_eq!(off.len(), on.len(), "{label}: byte length");
    if let Some(i) = off.iter().zip(on).position(|(a, b)| a != b) {
        panic!(
            "{label}: firstdiff={i} off={} on={} off_fnv={:016x} on_fnv={:016x}",
            off[i],
            on[i],
            fnv(off),
            fnv(on)
        );
    }
}

fn kernel_names(be: &VulkanBackend) -> Vec<&'static str> {
    let mut names: Vec<_> = be.shared.kernels.lock().unwrap().keys().copied().collect();
    names.sort_unstable();
    names
}

fn assert_kernel(be: &VulkanBackend, name: &str, words: &[u32], buffers: usize, push: u32) {
    let kernels = be.shared.kernels.lock().unwrap();
    let kernel = kernels
        .get(name)
        .unwrap_or_else(|| panic!("Recorder did not build {name}"));
    let mut expected = std::collections::hash_map::DefaultHasher::new();
    words.hash(&mut expected);
    assert_eq!(
        (kernel.name, kernel.n_buf, kernel.push_size),
        (name, buffers, push)
    );
    assert_eq!(
        kernel.subgroup_contract,
        crate::ops::SubgroupContract::Fixed(32)
    );
    assert_eq!(
        kernel.spv_hash,
        expected.finish(),
        "{name}: embedded SPIR-V identity"
    );
    println!(
        "PV_REUSE actual_kernel={} spv_hash={:016x} bindings={} push={}",
        kernel.name, kernel.spv_hash, kernel.n_buf, kernel.push_size
    );
}

fn sampled_cpu_check(q: &[u16], k: &[u16], v: &[u16], out: &[f32], case: Case) {
    // The original fixture's dense baseline oracle and tolerance, not a replacement GPU algorithm.
    let f = |bits| half::f16::from_bits(bits).to_f32() as f64;
    for row in [0, case.rows - 1] {
        for head in [0, case.nh - 1] {
            let kv_head = head / (case.nh / NKV);
            let end = case.kv.min(case.kv - case.rows + row + 1);
            let qi = (row * case.nh + head) * HD;
            let scores: Vec<f64> = (0..end)
                .map(|key| {
                    let ki = (key * NKV + kv_head) * HD;
                    (0..HD).map(|d| f(q[qi + d]) * f(k[ki + d])).sum::<f64>() / (HD as f64).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = weights.iter().sum();
            for d in [0, HD / 2, HD - 1] {
                let reference = weights
                    .iter()
                    .enumerate()
                    .map(|(key, w)| w * f(v[(key * NKV + kv_head) * HD + d]))
                    .sum::<f64>()
                    / sum;
                let got = out[(row * case.nh + head) * HD + d] as f64;
                assert!((got - reference).abs() < 5e-3,
                    "baseline CPU sample {case:?} row={row} head={head} d={d}: got={got} want={reference}");
            }
        }
    }
}

fn run(case: Case) {
    // Fresh kernel inventory per case makes construction an actual Recorder route witness.
    let mut be = backend();
    let lengths = case.lengths();
    let limits = unsafe {
        be.shared
            .instance
            .get_physical_device_properties(be.shared.physical_device)
    }
    .limits;
    assert!(
        (case.padded() / case.tile() * case.nh) <= limits.max_compute_work_group_count[0] as usize
    );
    assert!(case.padded() * case.nh <= limits.max_compute_work_group_count[0] as usize);
    assert!(case.splits() <= limits.max_compute_work_group_count[1] as usize);
    assert!(
        limits.max_compute_work_group_invocations >= 256
            && limits.max_compute_work_group_size[0] >= 256
    );
    assert!(lengths
        .iter()
        .all(|&n| (n + GUARD) * 4 <= limits.max_storage_buffer_range as usize));
    let kvpad = case.kv.div_ceil(64) * 64;
    let inputs = [
        half_values(
            case.rows * case.nh * HD,
            case.padded() * case.nh * HD,
            101,
            0.6,
        ),
        half_values(case.kv * NKV * HD, kvpad * NKV * HD, 102, 0.6),
        half_values(case.kv * NKV * HD, kvpad * NKV * HD, 103, 0.7),
    ];
    let buffers: Vec<_> = inputs
        .iter()
        .map(|data| upload(&be, bytemuck::cast_slice(data), BufferUsage::Weights))
        .collect();
    let outputs: Vec<_> = lengths
        .iter()
        .map(|&n| {
            upload(
                &be,
                bytemuck::cast_slice(&vec![SENTINEL; n + GUARD]),
                BufferUsage::Activations,
            )
        })
        .collect();
    let mut reference: Option<Vec<Vec<f32>>> = None;
    for enabled in [false, true] {
        let mut cfg = be.cfg().clone();
        cfg.kernels.vulkan.flash_pv_reuse = enabled;
        be.cfg = Arc::new(cfg);
        assert_eq!(be.cfg().kernels.vulkan.flash_splits, None);
        assert!(!be.cfg().kernels.vulkan.pipeline_cache_disk);
        for (buffer, &n) in outputs.iter().zip(&lengths) {
            be.upload(
                buffer.as_ref(),
                bytemuck::cast_slice(&vec![SENTINEL; n + GUARD]),
            )
            .unwrap();
        }
        let before = kernel_names(&be);
        let rec = be.recorder().expect("production Recorder");
        assert_eq!(rec.vk().flash_pv_reuse, enabled);
        rec.attention_prefill_flash_reg(
            buffers[0].as_ref(),
            buffers[1].as_ref(),
            buffers[2].as_ref(),
            outputs[3].as_ref(),
            outputs[0].as_ref(),
            outputs[1].as_ref(),
            outputs[2].as_ref(),
            case.rows,
            case.kv,
            case.nh,
            NKV,
            HD,
            case.kv - case.rows,
            case.deep(),
            None,
        );
        rec.finish().expect("production partial + combine submit");
        let (name, words) = case.route(enabled);
        assert_kernel(&be, name, words, 6, 32);
        assert_kernel(
            &be,
            COMBINE,
            crate::gemm::attn_flash_combine_hd256_spv(),
            4,
            16,
        );
        let after = kernel_names(&be);
        let built: Vec<_> = after
            .iter()
            .copied()
            .filter(|name| !before.contains(name))
            .collect();
        let mut expected_built = if !enabled {
            vec![case.baseline().0, COMBINE]
        } else if case.rows >= 512 {
            vec![name]
        } else {
            Vec::new()
        };
        expected_built.sort_unstable();
        assert_eq!(
            built, expected_built,
            "{case:?}: actual Recorder construction delta, enabled={enabled}"
        );
        assert_eq!(
            after.contains(&CW4_REUSE) || after.contains(&QK4_REUSE),
            enabled && case.rows >= 512
        );
        println!("PV_REUSE case={case:?} enabled={enabled} built={built:?} splits={} ksplit={} scheduling=whole_ubatch", case.splits(), case.kv.div_ceil(case.splits()).div_ceil(64) * 64);
        let actual: Vec<_> = outputs.iter().map(|b| download(&be, b.as_ref())).collect();
        for (i, values) in actual.iter().enumerate() {
            assert!(
                values[..lengths[i]]
                    .iter()
                    .all(|x| x.is_finite() && *x != SENTINEL),
                "{case:?} enabled={enabled}: nonfinite or unwritten output {i}"
            );
            assert!(
                values[lengths[i]..]
                    .iter()
                    .all(|x| x.to_bits() == SENTINEL.to_bits()),
                "{case:?} enabled={enabled}: physical output guard {i}"
            );
        }
        assert!(
            actual[3][..case.rows * case.nh * HD]
                .iter()
                .any(|x| *x != 0.0),
            "vacuous combined output"
        );
        for (input, initial) in buffers.iter().zip(&inputs) {
            let mut got = vec![0u8; input.len_bytes()];
            be.download(input.as_ref(), &mut got).unwrap();
            exact(
                "immutable input, zero tail and guard",
                bytemuck::cast_slice(initial),
                &got,
            );
        }
        if let Some(off) = reference.as_ref() {
            for i in 0..4 {
                exact(
                    ["po", "pm", "pl", "combine"][i],
                    bytemuck::cast_slice(&off[i]),
                    bytemuck::cast_slice(&actual[i]),
                );
            }
        } else {
            // Match the original fixture's short-context CPU sampling policy (including a key tail).
            if case.kv <= 2051 {
                sampled_cpu_check(&inputs[0], &inputs[1], &inputs[2], &actual[3], case);
            }
            reference = Some(actual);
        }
    }
    println!(
        "PV_REUSE {case:?}: rawpartials/combine/padded_rows byteexact, input/output guards intact"
    );
}

#[test]
fn flash_pv_reuse_fixture_plan_contract() {
    for case in SHORT.into_iter().chain(LONG) {
        assert!((1..=512).contains(&case.rows) && case.rows <= case.kv);
        assert_eq!(case.nh % NKV, 0);
        assert!(case.kv <= 70_019);
        assert!(case.lengths().iter().sum::<usize>() * 4 < 128 * 1024 * 1024);
        assert_eq!(case.splits(), if case.deep() { 8 } else { 1 });
        assert_eq!(case.route(false).0, if case.deep() { QK4 } else { CW4 });
        assert_eq!(
            case.route(true).0,
            match (case.rows >= 512, case.deep()) {
                (false, false) => CW4,
                (false, true) => QK4,
                (true, false) => CW4_REUSE,
                (true, true) => QK4_REUSE,
            }
        );
        let (baseline, candidate) = (
            case.baseline().1,
            crate::gemm::flash_pv_reuse_kernel(case.deep()).1,
        );
        assert_eq!(baseline.first(), Some(&0x0723_0203));
        assert_eq!(candidate.first(), Some(&0x0723_0203));
        assert!(
            baseline != candidate,
            "embedded candidate must differ from baseline"
        );
    }
    assert!(SHORT.iter().any(|c| c.rows % 128 != 0 && c.kv % 64 != 0));
    assert_eq!(LONG[1].lengths()[0] * 4, 100_663_296);
}

#[test]
#[ignore = "parent-serialized qualified Windows RX 7900 XTX: embedded production CW4 cfg on/off exact, no timing"]
fn flash_pv_reuse_cw4_recorder_byte_contract() {
    for case in SHORT {
        run(case);
    }
}

#[test]
#[ignore = "parent-serialized qualified Windows RX 7900 XTX: bounded 70K production F16_SCORE/QK4 cfg on/off exact, no timing"]
fn flash_pv_reuse_qk4_recorder_byte_contract() {
    for case in LONG {
        run(case);
    }
}
