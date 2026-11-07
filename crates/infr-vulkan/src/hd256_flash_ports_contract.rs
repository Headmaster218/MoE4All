#![cfg(test)]

//! Recorder child for the PR146 hd256 ports: bounded correctness, no timing or probe SPIR-V.
//! The parent adds the module declaration. GPU execution is deliberately restricted to the
//! measured Windows AMD proprietary XTX route; the unrolled twin is inspected on the CPU only.
use super::*;
use infr_core::backend::BufferUsage;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

const NH: usize = 4;
const NKV: usize = 2;
const HD: usize = 256;
const GUARD: usize = 64;
const INPUT_CANARY: u16 = 0x7b55;
const SENTINEL: f32 = -777.25;
const NOUNROLL: &str = "attn_flash_warp_hd256_bm16_nounroll";
const UNROLL: &str = "attn_flash_warp_hd256_bm16";
const COMBINE: &str = "attn_flash_combine_hd256";

#[derive(Clone, Copy, Debug)]
struct Case {
    rows: usize,
    kv: usize,
    splits: usize,
}

impl Case {
    fn padded(self) -> usize {
        self.rows.div_ceil(64) * 64
    }

    fn input_lengths(self) -> [usize; 3] {
        // K/V have exactly kv rows, NOT ceil(kv/64) rows. The following 64 elements
        // are an immutable guard, much smaller than one KV row (NKV * HD elements).
        [
            self.padded() * NH * HD,
            self.kv * NKV * HD,
            self.kv * NKV * HD,
        ]
    }

    fn output_lengths(self) -> [usize; 4] {
        let scalar = self.splits * self.padded() * NH;
        [scalar * HD, scalar, scalar, self.padded() * NH * HD]
    }

    fn bytes(self) -> usize {
        self.input_lengths()
            .iter()
            .map(|n| (n + GUARD) * 2)
            .sum::<usize>()
            + self
                .output_lengths()
                .iter()
                .map(|n| (n + GUARD) * 4)
                .sum::<usize>()
    }
}

const CASES: [Case; 5] = [
    Case {
        rows: 65,
        kv: 65,
        splits: 1,
    },
    Case {
        rows: 129,
        kv: 513,
        splits: 1,
    },
    Case {
        rows: 64,
        kv: 512,
        splits: 1,
    },
    Case {
        rows: 257,
        kv: 4099,
        splits: 1,
    },
    Case {
        rows: 257,
        kv: 4099,
        splits: 3,
    },
];

#[test]
fn hd256_flash_ports_embedded_spv_and_fixture_contract() {
    // These are the two shipped builds of the CURRENT shader, not historical probe files.
    // Merely reading words does not ask a Vulkan driver to compile the unrolled variant.
    let unroll = crate::gemm::attn_flash_warp_hd256_bm16_spv();
    let nounroll = crate::gemm::attn_flash_warp_hd256_bm16_nounroll_spv();
    let combine = crate::gemm::attn_flash_combine_hd256_spv();
    for words in [unroll, nounroll, combine] {
        assert!(
            words.len() > 5,
            "embedded SPIR-V must contain more than its header"
        );
        assert_eq!(words.first(), Some(&0x0723_0203));
    }
    assert!(
        bytemuck::cast_slice::<u32, u8>(unroll) != bytemuck::cast_slice::<u32, u8>(nounroll),
        "unroll and NO_UNROLL must be distinct embedded byte sequences"
    );
    assert_eq!(NH / NKV, 2, "exercise both query heads of each GQA group");
    const {
        assert!(GUARD < NKV * HD);
    }
    for case in CASES {
        assert!((1..=257).contains(&case.rows) && case.rows <= case.kv && case.kv <= 4099);
        assert!(case.splits == 1 || case.splits == 3);
        assert!(
            case.bytes() < 32 * 1024 * 1024,
            "bounded GPU fixture: {case:?}"
        );
        assert_eq!(case.padded() % 64, 0);
        assert!(case.padded() - case.rows < 64);
        assert_eq!(case.input_lengths()[1], case.kv * NKV * HD);
        assert_eq!(case.input_lengths()[2], case.kv * NKV * HD);
        assert_eq!(
            flash_split_count(
                Some(case.splits),
                (case.padded() / 16 * NH) as u32,
                case.kv,
                false
            ),
            case.splits as u32
        );
    }
    assert!(CASES.iter().any(|c| c.rows % 64 != 0 && c.kv % 64 != 0));
    assert!(CASES.iter().any(|c| c.rows % 64 == 0 && c.kv % 64 == 0));
}

fn backend(splits: usize) -> VulkanBackend {
    let selected = std::env::var("MOE4ALL_TEST_DEVICE")
        .expect("parent must explicitly select the idle XTX with MOE4ALL_TEST_DEVICE=Vulkan1");
    assert_eq!(selected, "Vulkan1", "no implicit/default device selection");
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev = Some(selected);
    cfg.kernels.vulkan.flash_splits = Some(splits);
    cfg.kernels.vulkan.pipeline_cache_disk = false;
    // Pin the physical index as well: no fallback to another device if a selector is ignored.
    let be = VulkanBackend::new_on_with(1, Arc::new(cfg)).expect("explicit Vulkan1 backend");
    let caps = be.caps();
    assert!(
        cfg!(target_os = "windows")
            && caps.name.contains("7900 XTX")
            && !caps.integrated
            && caps.f16
            && caps.f16_coopmat()
            && caps.subgroup_min <= 32
            && caps.subgroup_max >= 32
            && be.max_shared_memory_bytes() >= 30_912,
        "requires the qualified Windows RX 7900 XTX, F16 coopmat and wave32; got {caps:?}"
    );
    // Check BEFORE recording: do not create the known-crashing unrolled pipeline on this host,
    // or silently run an unqualified RADV/other-driver path instead.
    assert!(
        be.is_amd_proprietary_driver(),
        "this test only executes production NO_UNROLL"
    );
    assert_eq!(be.cfg().device.dev.as_deref(), Some("Vulkan1"));
    assert_eq!(be.cfg().kernels.vulkan.flash_splits, Some(splits));
    assert!(!be.cfg().kernels.vulkan.pipeline_cache_disk);
    be
}

fn half_values(n: usize, padded: usize, seed: u32, scale: f32) -> Vec<u16> {
    assert!(n <= padded);
    let mut values: Vec<_> = (0..n)
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

fn read_f32(be: &VulkanBackend, buffer: &dyn Buffer) -> Vec<f32> {
    let mut values = vec![0.0; buffer.len_bytes() / 4];
    be.download(buffer, bytemuck::cast_slice_mut(&mut values))
        .expect("full readback with guard");
    values
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
        "actual embedded SPIR-V identity: {name}"
    );
    assert!(
        !kernels.contains_key(UNROLL),
        "unrolled pipeline must never be constructed here"
    );
    assert!(
        !kernels
            .keys()
            .any(|n| n.starts_with("attn_flash_reg_hd256")),
        "BM16 must not fall through to Register-O"
    );
}

fn sampled_f64_oracle(inputs: &[Vec<u16>; 3], output: &[f32], case: Case) {
    // Independent causal softmax over the actual quantized F16 input. No GPU baseline, no
    // production attention/oracle helper and no rounding to the shader's F16 probability tile.
    let f = |bits| half::f16::from_bits(bits).to_f64();
    let mut max_error = 0.0f64;
    for row in [0, case.rows - 1] {
        for head in 0..NH {
            let kv_head = head / (NH / NKV);
            let end = case.kv - case.rows + row + 1;
            let qi = (row * NH + head) * HD;
            let scores: Vec<f64> = (0..end)
                .map(|key| {
                    let ki = (key * NKV + kv_head) * HD;
                    (0..HD)
                        .map(|d| f(inputs[0][qi + d]) * f(inputs[1][ki + d]))
                        .sum::<f64>()
                        / (HD as f64).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = weights.iter().sum();
            assert!(sum.is_finite() && sum > 0.0);
            // Both sides of the 16/64-column boundaries, all four PV column-worker bands.
            for d in [0, 15, 16, 63, 64, 127, 128, 191, 192, 255] {
                let want = weights
                    .iter()
                    .enumerate()
                    .map(|(key, w)| w * f(inputs[2][(key * NKV + kv_head) * HD + d]))
                    .sum::<f64>()
                    / sum;
                let got = output[(row * NH + head) * HD + d] as f64;
                let error = (got - want).abs();
                assert!(
                    got.is_finite() && want.is_finite() && error < 3e-3,
                    "f64 oracle {case:?} row={row} head={head} d={d}: got={got} want={want} abs={error}"
                );
                max_error = max_error.max(error);
            }
        }
    }
    println!("HD256_PORTS {case:?}: f64 samples=80 max_abs={max_error:e}");
}

fn run(case: Case) {
    let be = backend(case.splits);
    let limits = unsafe {
        be.shared
            .instance
            .get_physical_device_properties(be.shared.physical_device)
    }
    .limits;
    assert!(case.padded() / 16 * NH <= limits.max_compute_work_group_count[0] as usize);
    assert!(case.padded() * NH <= limits.max_compute_work_group_count[0] as usize);
    assert!(case.splits <= limits.max_compute_work_group_count[1] as usize);
    assert!(
        limits.max_compute_work_group_invocations >= 128
            && limits.max_compute_work_group_size[0] >= 128
    );
    let ilen = case.input_lengths();
    let olen = case.output_lengths();
    assert!(ilen
        .iter()
        .all(|n| (n + GUARD) * 2 <= limits.max_storage_buffer_range as usize));
    assert!(olen
        .iter()
        .all(|n| (n + GUARD) * 4 <= limits.max_storage_buffer_range as usize));
    let mut inputs = [
        half_values(case.rows * NH * HD, ilen[0], 101, 0.8),
        half_values(ilen[1], ilen[1], 102, 0.8),
        half_values(ilen[2], ilen[2], 103, 0.7),
    ];
    // Make a GQA-group/column-band mix-up visible even after averaging thousands of keys.
    for (i, bits) in inputs[2][..ilen[2]].iter_mut().enumerate() {
        let bias = ((i / HD % NKV) as f32 - 0.5) * 0.5 + (i % HD / 64) as f32 * 0.03125;
        *bits = half::f16::from_f32(half::f16::from_bits(*bits).to_f32() + bias).to_bits();
    }
    let buffers: Vec<_> = inputs
        .iter()
        .map(|values| upload(&be, bytemuck::cast_slice(values), BufferUsage::Weights))
        .collect();
    let outputs: Vec<_> = olen
        .iter()
        .map(|n| {
            upload(
                &be,
                bytemuck::cast_slice(&vec![SENTINEL; n + GUARD]),
                BufferUsage::Activations,
            )
        })
        .collect();
    for (buffer, n) in buffers.iter().zip(ilen) {
        assert_eq!(
            buffer.len_bytes(),
            (n + GUARD) * 2,
            "no hidden KV tile padding"
        );
    }
    let rec = be.recorder().expect("production Recorder");
    assert_eq!(rec.vk().flash_splits, Some(case.splits));
    rec.attention_prefill_flash_hd256(
        buffers[0].as_ref(),
        buffers[1].as_ref(),
        buffers[2].as_ref(),
        outputs[3].as_ref(),
        outputs[0].as_ref(),
        outputs[1].as_ref(),
        outputs[2].as_ref(),
        case.rows,
        case.kv,
        NH,
        NKV,
        case.kv - case.rows,
        false, // no heuristic deep-context split override
        false, // register_o_tail_safe: force the production BM16 path
        None,  // real F16 KV buffers, no BDA/dequant substitute
    );
    rec.finish()
        .expect("one production partial + combine submission");
    assert_kernel(
        &be,
        NOUNROLL,
        crate::gemm::attn_flash_warp_hd256_bm16_nounroll_spv(),
        6,
        32,
    );
    assert_kernel(
        &be,
        COMBINE,
        crate::gemm::attn_flash_combine_hd256_spv(),
        4,
        16,
    );
    let actual: Vec<_> = outputs.iter().map(|b| read_f32(&be, b.as_ref())).collect();
    for (i, values) in actual.iter().enumerate() {
        let label = ["po", "pm", "pl", "output"][i];
        assert_eq!(values.len(), olen[i] + GUARD);
        assert!(
            values[..olen[i]]
                .iter()
                .all(|x| x.is_finite() && x.to_bits() != SENTINEL.to_bits()),
            "{case:?}: {label} active/padded region contains nonfinite or unwritten data"
        );
        assert!(
            values[olen[i]..]
                .iter()
                .all(|x| x.to_bits() == SENTINEL.to_bits()),
            "{case:?}: {label} physical guard was overwritten"
        );
    }
    assert!(
        actual[3][..case.rows * NH * HD]
            .iter()
            .any(|x| x.abs() > 1e-6),
        "vacuous output"
    );
    for (i, (buffer, initial)) in buffers.iter().zip(&inputs).enumerate() {
        let mut got = vec![0u8; buffer.len_bytes()];
        be.download(buffer.as_ref(), &mut got)
            .expect("read back complete input plus guard");
        let expected: &[u8] = bytemuck::cast_slice(initial);
        if let Some(byte) = expected.iter().zip(&got).position(|(a, b)| a != b) {
            panic!(
                "{case:?}: input {i} (including zero query padding and guard) changed at byte {byte}"
            );
        }
        assert_eq!(got.len(), expected.len());
    }
    sampled_f64_oracle(&inputs, &actual[3], case);
    println!(
        "HD256_PORTS {case:?}: NO_UNROLL production partial/combine, all padded writes and guards checked"
    );
}

#[test]
#[ignore = "parent-serialized GPU correctness only; MOE4ALL_TEST_DEVICE=Vulkan1, Windows AMD proprietary RX 7900 XTX; NO_UNROLL only"]
fn hd256_flash_ports_recorder_f64_and_guards() {
    for case in CASES {
        run(case);
    }
}
