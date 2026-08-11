//! Proves the Vulkan ops that stand UPSTREAM of a transformer layer's first Linear agree with the
//! host on the real weights of a real model: the in-shader weight dequant (`native_decode.glsl`'s
//! `dqblk`, shared by every GEMM/GEMV tier and by `Op::EmbedGather`) and `Op::RmsNorm`. A
//! percent-level disagreement in any of them is invisible to every existing parity test — those
//! compare one Vulkan build against another Vulkan build, or floor a cosine at 0.5 — and it
//! reaches every downstream op, so it is worth a guard of its own.
//!
//! Readout mechanism for the dequant: a ONE-HOT activation. `native_gemv.comp` computes
//! `y[o] = sum_k W[o*in_f + k] * x[k]` in f32; with `x = e_c` every term but `k == c` is an exact
//! `w * 0.0 == 0.0` and the tree reduction sums exact zeros, so `y[o]` is BIT-EQUAL to the GPU's
//! decoded `W[o][c]`. That turns the GEMV into a weight-decode probe with no arithmetic of its own
//! to blame, and one dispatch reads one whole column — `out_f` different rows at once.
//!
//! Run: `cargo test -p infr-vulkan --test weight_dequant_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_core::loader::{TensorInfo, WeightSource};
use infr_core::DType;
use infr_gguf::Gguf;
use infr_vulkan::VulkanBackend;
use std::path::PathBuf;

/// Columns per dispatch. The probe sweeps EVERY column of the tensor — one one-hot activation row
/// each — so the comparison covers every element of the weight, not a sample of the block
/// structure; this is the chunk the rows are batched into.
const COLS_PER_DISPATCH: usize = 128;

/// Locate a GGUF in the HF cache the way `infr-llama`'s model-gated tests do.
fn find_gguf(repo: &str, file: &str) -> Option<PathBuf> {
    let hub = std::env::var("HOME").ok()? + "/.cache/huggingface/hub";
    let base = format!("{hub}/models--{repo}/snapshots");
    std::fs::read_dir(&base).ok()?.find_map(|e| {
        let f = e.ok()?.path().join(file);
        f.exists().then_some(f)
    })
}

fn deepseek_v2_lite() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_DEEPSEEK2") {
        return Some(PathBuf::from(p));
    }
    find_gguf(
        "JenniSD--DeepSeek-V2-Lite-Chat-Q4_K_M-GGUF",
        "deepseek-v2-lite-chat-q4_k_m.gguf",
    )
}

/// Open the model, or `None` when the GGUF is not in the cache (every test here self-skips).
fn open_model() -> Option<Gguf> {
    let path = deepseek_v2_lite()?;
    Some(Gguf::open(&path).expect("open gguf"))
}

/// One named tensor's `TensorInfo`, panicking with the model's tensor count if it is absent — a
/// renamed tensor must fail loudly, not silently reduce the probe to nothing.
fn tensor<'g>(gguf: &'g Gguf, name: &str) -> &'g TensorInfo {
    gguf.tensors()
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| {
            panic!(
                "{name} absent from a model with {} tensors",
                gguf.tensors().len()
            )
        })
}

/// Read the GPU's decoded `W[o][c]` for every `o` and every `c` in `cols`, by running one
/// `native_gemv.comp` dispatch whose `rows` activation rows are the one-hot vectors `e_c`.
/// Returns `[col][out]`.
fn gpu_weight_columns(
    be: &VulkanBackend,
    dtype: DType,
    w_bytes: &[u8],
    in_f: usize,
    out_f: usize,
    cols: &[usize],
) -> Vec<Vec<f32>> {
    let rows = cols.len();
    let mut x = vec![0f32; rows * in_f];
    for (r, &c) in cols.iter().enumerate() {
        x[r * in_f + c] = 1.0;
    }
    let x_buf = be.alloc(x.len() * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let y_buf = be
        .alloc(rows * out_f * 4, BufferUsage::Activations)
        .unwrap();

    let (arena, addr) = be.alloc_arena_bda(w_bytes.len()).unwrap();
    be.upload(arena.as_ref(), w_bytes).unwrap();

    let rec = be.recorder().unwrap();
    rec.linear_native_at(
        dtype,
        addr,
        0,
        x_buf.as_ref(),
        y_buf.as_ref(),
        rows,
        in_f,
        out_f,
    );
    rec.finish().unwrap();

    let mut out = vec![0u8; rows * out_f * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let y: &[f32] = bytemuck::cast_slice(&out);
    (0..rows)
        .map(|r| y[r * out_f..(r + 1) * out_f].to_vec())
        .collect()
}

/// Running worst-case disagreement between the GPU's decoded weights and `dequant_block`'s, plus
/// the count of elements that were not bit-identical.
#[derive(Default)]
struct Diff {
    compared: usize,
    bit_mismatches: usize,
    max_abs: f32,
    max_rel: f32,
    nonzero_seen: bool,
}

impl Diff {
    /// Fold one dispatch's columns in, printing the first few offenders with their indices.
    fn accumulate(
        &mut self,
        label: &str,
        gpu: &[Vec<f32>],
        host: &[f32],
        in_f: usize,
        out_f: usize,
        cols: &[usize],
    ) {
        for (r, &c) in cols.iter().enumerate() {
            for o in 0..out_f {
                let (g, h) = (gpu[r][o], host[o * in_f + c]);
                let abs = (g - h).abs();
                let rel = if h == 0.0 { 0.0 } else { abs / h.abs() };
                self.compared += 1;
                self.nonzero_seen |= g != 0.0;
                self.max_abs = self.max_abs.max(abs);
                self.max_rel = self.max_rel.max(rel);
                if g.to_bits() != h.to_bits() {
                    if self.bit_mismatches < 8 {
                        println!(
                            "  {label} MISMATCH row {o} col {c}: gpu {g:.9} host {h:.9} \
                             (abs {abs:.3e}, rel {rel:.3e})"
                        );
                    }
                    self.bit_mismatches += 1;
                }
            }
        }
    }

    /// Print the verdict and assert bit-equality — the GPU and the host run the same decode
    /// arithmetic on the same bytes, so anything short of bitwise equality is a bug, not a
    /// tolerance question.
    fn assert_bit_identical(&self, label: &str) {
        println!(
            "{label}: {} elements compared, {} not bit-identical, max_abs {:.3e}, max_rel {:.3e}",
            self.compared, self.bit_mismatches, self.max_abs, self.max_rel
        );
        assert!(
            self.nonzero_seen,
            "{label}: every probed weight decoded to zero — the dispatch did not run"
        );
        assert_eq!(
            self.bit_mismatches, 0,
            "{label}: GPU in-shader weight dequant disagrees with dequant_block \
             (max_abs {:.3e}, max_rel {:.3e})",
            self.max_abs, self.max_rel
        );
    }
}

#[test]
#[ignore = "requires a Vulkan GPU and the DeepSeek-V2-Lite GGUF in the HF cache"]
fn q4k_weight_dequant_matches_host_on_deepseek_v2_lite() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let Some(gguf) = open_model() else {
        eprintln!("skip: DeepSeek-V2-Lite-Chat not in the HF cache");
        return;
    };
    const NAME: &str = "blk.0.attn_kv_a_mqa.weight";
    let info = tensor(&gguf, NAME);
    println!(
        "{NAME}: dtype {:?} shape {:?} nbytes {}",
        info.dtype, info.shape, info.nbytes
    );
    assert_eq!(info.dtype, DType::Q4K, "this probe is about Q4_K");
    let (in_f, out_f) = (info.shape[0], info.shape[1]);
    let dtype = info.dtype;
    let bytes = gguf.tensor_bytes(NAME).expect("tensor bytes").to_vec();

    let host = infr_gguf::dequant::dequant_block(dtype, &bytes).expect("host dequant");
    assert_eq!(host.len(), in_f * out_f);

    let mut diff = Diff::default();
    for chunk in (0..in_f).collect::<Vec<_>>().chunks(COLS_PER_DISPATCH) {
        let gpu = gpu_weight_columns(&be, dtype, &bytes, in_f, out_f, chunk);
        if chunk[0] == 0 {
            println!(
                "row 0 cols 0..3 host: {:.9} {:.9} {:.9}",
                host[0], host[1], host[2]
            );
            println!(
                "row 0 cols 0..3 gpu:  {:.9} {:.9} {:.9}",
                gpu[0][0], gpu[1][0], gpu[2][0]
            );
        }
        diff.accumulate("real Q4_K", &gpu, &host, in_f, out_f, chunk);
    }
    diff.assert_bit_identical("real Q4_K");
}

/// Token ids probed, spread across the vocabulary so different super-blocks and different row
/// alignments inside the embedding table are hit, not one corner of it.
fn probe_ids(vocab: usize, n: usize) -> Vec<i32> {
    (0..n).map(|i| (i * (vocab - 1) / (n - 1)) as i32).collect()
}

/// `Op::EmbedGather` reads the quantized `token_embd` table through the SAME `dqblk` the GEMV
/// family uses, while the CPU backend host-dequantizes that table — so the two must agree
/// bit-for-bit on every gathered row.
#[test]
#[ignore = "requires a Vulkan GPU and the DeepSeek-V2-Lite GGUF in the HF cache"]
fn embed_gather_matches_host_dequant_on_deepseek_v2_lite() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let Some(gguf) = open_model() else {
        eprintln!("skip: DeepSeek-V2-Lite-Chat not in the HF cache");
        return;
    };
    const NAME: &str = "token_embd.weight";
    let info = tensor(&gguf, NAME);
    println!(
        "{NAME}: dtype {:?} shape {:?} nbytes {}",
        info.dtype, info.shape, info.nbytes
    );
    let (ne, vocab) = (info.shape[0], info.shape[1]);
    let dtype = info.dtype;
    let (blk_elems, blk_bytes) = infr_gguf::block_layout(dtype);
    assert_eq!(
        ne % blk_elems,
        0,
        "a table row that is not a whole number of blocks would not be row-sliceable"
    );
    let row_bytes = ne / blk_elems * blk_bytes;
    let table = gguf.tensor_bytes(NAME).expect("tensor bytes");

    let ids = probe_ids(vocab, 64);
    let ids_buf = be.alloc(ids.len() * 4, BufferUsage::Activations).unwrap();
    be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&ids))
        .unwrap();
    let dst = be
        .alloc(ids.len() * ne * 4, BufferUsage::Activations)
        .unwrap();
    let (arena, addr) = be.alloc_arena_bda(table.len()).unwrap();
    be.upload(arena.as_ref(), table).unwrap();

    let rec = be.recorder().unwrap();
    // `scale = 1.0` and `row_bytes = 0` (the table fits u32): the dispatch is then a pure decode,
    // with no scaling multiply of its own to confound the comparison.
    rec.embed_gather_at(
        dtype,
        addr,
        ids_buf.as_ref(),
        dst.as_ref(),
        ids.len(),
        ne,
        1.0,
        0,
    );
    rec.finish().unwrap();
    let mut out = vec![0u8; ids.len() * ne * 4];
    be.download(dst.as_ref(), &mut out).unwrap();
    let gpu: &[f32] = bytemuck::cast_slice(&out);

    let (mut mismatches, mut max_abs, mut max_rel, mut nonzero) = (0usize, 0f32, 0f32, false);
    for (r, &id) in ids.iter().enumerate() {
        let off = id as usize * row_bytes;
        let host = infr_gguf::dequant::dequant_block(dtype, &table[off..off + row_bytes])
            .expect("host dequant of one embedding row");
        for i in 0..ne {
            let (g, h) = (gpu[r * ne + i], host[i]);
            let abs = (g - h).abs();
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(if h == 0.0 { 0.0 } else { abs / h.abs() });
            nonzero |= g != 0.0;
            if g.to_bits() != h.to_bits() {
                if mismatches < 8 {
                    println!("  embed MISMATCH id {id} elem {i}: gpu {g:.9} host {h:.9}");
                }
                mismatches += 1;
            }
        }
    }
    println!(
        "embed_gather: {} rows x {ne} elements, {mismatches} not bit-identical, \
         max_abs {max_abs:.3e}, max_rel {max_rel:.3e}",
        ids.len()
    );
    assert!(nonzero, "every gathered embedding was zero — nothing ran");
    assert_eq!(
        mismatches, 0,
        "Op::EmbedGather's in-shader dequant disagrees with dequant_block \
         (max_abs {max_abs:.3e}, max_rel {max_rel:.3e})"
    );
}

/// Host f64 RMSNorm reference: `y[i] = x[i] * 1/sqrt(mean(x²) + eps) * w[i]`, accumulated in f64 so
/// the comparison scores the GPU against the exact value rather than against one particular f32
/// summation order.
fn rmsnorm_f64(x: &[f32], w: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for (r, row) in x.chunks_exact(dim).enumerate() {
        let ss: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let scale = 1.0 / (ss / dim as f64 + eps as f64).sqrt();
        for i in 0..dim {
            y[r * dim + i] = (row[i] as f64 * scale * w[i] as f64) as f32;
        }
    }
    y
}

/// `Op::RmsNorm` on layer 0's real `attn_norm` weight, over real embedding rows — the exact input
/// the first Linear of the first block sees. Two dispatches, because the host gates `rows == 1`
/// onto the `-DWIDE` twin (decode) and everything else onto the 256-thread build (prefill), and a
/// reduction bug could live in either.
#[test]
#[ignore = "requires a Vulkan GPU and the DeepSeek-V2-Lite GGUF in the HF cache"]
fn rmsnorm_matches_f64_reference_on_deepseek_v2_lite_layer0() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let Some(gguf) = open_model() else {
        eprintln!("skip: DeepSeek-V2-Lite-Chat not in the HF cache");
        return;
    };
    let arch = gguf
        .metadata()
        .str("general.architecture")
        .expect("general.architecture")
        .to_string();
    let eps = gguf
        .metadata()
        .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .and_then(infr_core::loader::MetaValue::as_f64)
        .expect("rms eps") as f32;

    let embd = tensor(&gguf, "token_embd.weight");
    let (ne, vocab) = (embd.shape[0], embd.shape[1]);
    let (blk_elems, blk_bytes) = infr_gguf::block_layout(embd.dtype);
    let row_bytes = ne / blk_elems * blk_bytes;
    let table = gguf.tensor_bytes("token_embd.weight").expect("table bytes");

    let norm = tensor(&gguf, "blk.0.attn_norm.weight");
    println!(
        "blk.0.attn_norm.weight: dtype {:?} shape {:?}; eps {eps:e} (arch {arch})",
        norm.dtype, norm.shape
    );
    assert_eq!(
        norm.shape[0], ne,
        "attn_norm width must match the model dim"
    );
    let w = infr_gguf::dequant::dequant_block(
        norm.dtype,
        gguf.tensor_bytes("blk.0.attn_norm.weight").unwrap(),
    )
    .expect("host dequant of attn_norm");

    // Real activations: the embedding rows themselves, which is what layer 0 normalizes.
    let ids = probe_ids(vocab, 64);
    let mut x = Vec::with_capacity(ids.len() * ne);
    for &id in &ids {
        let off = id as usize * row_bytes;
        x.extend(
            infr_gguf::dequant::dequant_block(embd.dtype, &table[off..off + row_bytes])
                .expect("host dequant of one embedding row"),
        );
    }

    let w_buf = be.alloc(w.len() * 4, BufferUsage::Weights).unwrap();
    be.upload(w_buf.as_ref(), bytemuck::cast_slice(&w)).unwrap();

    for rows in [ids.len(), 1] {
        let n = rows * ne;
        let x_buf = be.alloc(n * 4, BufferUsage::Activations).unwrap();
        be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x[..n]))
            .unwrap();
        let y_buf = be.alloc(n * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.rmsnorm(
            x_buf.as_ref(),
            w_buf.as_ref(),
            y_buf.as_ref(),
            rows,
            ne,
            eps,
        );
        rec.finish().unwrap();
        let mut out = vec![0u8; n * 4];
        be.download(y_buf.as_ref(), &mut out).unwrap();
        let gpu: &[f32] = bytemuck::cast_slice(&out);
        let want = rmsnorm_f64(&x[..n], &w, ne, eps);

        let (mut max_abs, mut max_rel) = (0f32, 0f32);
        for i in 0..n {
            let abs = (gpu[i] - want[i]).abs();
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(if want[i] == 0.0 {
                0.0
            } else {
                abs / want[i].abs()
            });
        }
        let build = if rows == 1 { "WIDE (decode)" } else { "base" };
        println!("rmsnorm {build} rows={rows}: max_abs {max_abs:.3e}, max_rel {max_rel:.3e}");
        assert!(gpu.iter().any(|v| *v != 0.0), "rmsnorm wrote only zeros");
        // f32 rounding over a 2048-wide reduction; anything at the percent level this bug is about
        // is orders of magnitude above this.
        assert!(
            max_rel < 1e-5,
            "rmsnorm {build} diverges from the f64 reference: max_rel {max_rel:.3e}"
        );
    }
}

/// Relative L2 error of `got` against `want`, plus the worst element-wise relative error over the
/// elements that are not negligible next to the row's own scale (a near-zero output has an
/// unbounded relative error that says nothing about the kernel).
fn err_stats(got: &[f32], want: &[f32]) -> (f64, f64) {
    let scale = want.iter().fold(0f64, |m, &v| m.max(v.abs() as f64));
    let (mut num, mut den, mut worst) = (0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (g as f64, w as f64);
        num += (g - w) * (g - w);
        den += w * w;
        if w.abs() > 0.01 * scale {
            worst = worst.max(((g - w) / w).abs());
        }
    }
    ((num / den).sqrt(), worst)
}

/// Prices layer 0's FIRST Linear (`wkv_a_mqa`, Q4_K 2048→576) on the real weights and real
/// activations, against an f64 reference of the same chain — separating the two Vulkan routes the
/// adapter can take for it. Everything upstream is bit-exact (the two tests above), so whatever
/// this shows is the whole of Vulkan's layer-0 error.
///
/// `n = 576` is a multiple of 64 but not of 128, so production takes the dp4a `mmq` arm, whose
/// `quant_q8` prepass reduces the ACTIVATIONS to per-32-block int8 before the matmul. The
/// f32-activation GEMV decodes the same weights and keeps the activations in f32.
#[test]
#[ignore = "requires a Vulkan GPU and the DeepSeek-V2-Lite GGUF in the HF cache"]
fn layer0_kv_a_mqa_error_is_the_int8_activation_prepass() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let Some(gguf) = open_model() else {
        eprintln!("skip: DeepSeek-V2-Lite-Chat not in the HF cache");
        return;
    };
    let arch = gguf
        .metadata()
        .str("general.architecture")
        .expect("general.architecture")
        .to_string();
    let eps = gguf
        .metadata()
        .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .and_then(infr_core::loader::MetaValue::as_f64)
        .expect("rms eps") as f32;

    let embd = tensor(&gguf, "token_embd.weight");
    let (ne, vocab) = (embd.shape[0], embd.shape[1]);
    let (be_elems, be_bytes) = infr_gguf::block_layout(embd.dtype);
    let erow = ne / be_elems * be_bytes;
    let table = gguf.tensor_bytes("token_embd.weight").expect("table bytes");
    let nw = infr_gguf::dequant::dequant_block(
        tensor(&gguf, "blk.0.attn_norm.weight").dtype,
        gguf.tensor_bytes("blk.0.attn_norm.weight").unwrap(),
    )
    .expect("attn_norm");

    let lin = tensor(&gguf, "blk.0.attn_kv_a_mqa.weight");
    let (k, n) = (lin.shape[0], lin.shape[1]);
    assert_eq!(k, ne);
    assert_eq!(n % 64, 0, "mmq needs n % 64 == 0");
    let wq = gguf
        .tensor_bytes("blk.0.attn_kv_a_mqa.weight")
        .expect("wkv_a_mqa bytes");
    let wf = infr_gguf::dequant::dequant_block(lin.dtype, wq).expect("wkv_a_mqa dequant");

    // The reported repro is a 60-token prefill.
    let m = 60usize;
    let ids = probe_ids(vocab, m);
    let mut emb = Vec::with_capacity(m * ne);
    for &id in &ids {
        let off = id as usize * erow;
        emb.extend(
            infr_gguf::dequant::dequant_block(embd.dtype, &table[off..off + erow]).expect("row"),
        );
    }

    // f64 reference of the whole chain: normalize in f64, then dot in f64.
    let mut href = vec![0f32; m * n];
    for r in 0..m {
        let row = &emb[r * ne..(r + 1) * ne];
        let ss: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let s = 1.0 / (ss / ne as f64 + eps as f64).sqrt();
        let xn: Vec<f64> = (0..ne).map(|i| row[i] as f64 * s * nw[i] as f64).collect();
        for o in 0..n {
            let acc: f64 = (0..ne).map(|i| wf[o * ne + i] as f64 * xn[i]).sum();
            href[r * n + o] = acc as f32;
        }
    }

    // GPU: one rmsnorm feeding both routes, so they differ only in the matmul.
    let npad = m.div_ceil(64) * 64 + 64;
    let x_buf = be.alloc(m * ne * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&emb))
        .unwrap();
    let w_buf = be.alloc(nw.len() * 4, BufferUsage::Weights).unwrap();
    be.upload(w_buf.as_ref(), bytemuck::cast_slice(&nw))
        .unwrap();
    let hn = be.alloc(npad * ne * 4, BufferUsage::Activations).unwrap();
    let (arena, addr) = be.alloc_arena_bda(wq.len()).unwrap();
    be.upload(arena.as_ref(), wq).unwrap();

    let y_f32 = be.alloc(npad * n * 4, BufferUsage::Activations).unwrap();
    let y_mmq = be.alloc(npad * n * 4, BufferUsage::Activations).unwrap();
    let qa = be.alloc(npad * ne, BufferUsage::Activations).unwrap();
    let dact = be
        .alloc(npad * (ne / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let sact = be
        .alloc(npad * (ne / 32) * 2, BufferUsage::Activations)
        .unwrap();

    let rec = be.recorder().unwrap();
    rec.rmsnorm(x_buf.as_ref(), w_buf.as_ref(), hn.as_ref(), m, ne, eps);
    rec.linear_native_at(lin.dtype, addr, 0, hn.as_ref(), y_f32.as_ref(), m, ne, n);
    rec.quant_q8(
        hn.as_ref(),
        qa.as_ref(),
        dact.as_ref(),
        sact.as_ref(),
        m,
        ne,
    );
    rec.matmul_mmq_at(
        lin.dtype,
        qa.as_ref(),
        dact.as_ref(),
        sact.as_ref(),
        addr,
        0,
        y_mmq.as_ref(),
        m,
        ne,
        n,
    );
    rec.finish().unwrap();

    let read = |b: &dyn infr_core::backend::Buffer| -> Vec<f32> {
        let mut raw = vec![0u8; npad * n * 4];
        be.download(b, &mut raw).unwrap();
        bytemuck::cast_slice::<u8, f32>(&raw)[..m * n].to_vec()
    };
    let g32 = read(y_f32.as_ref());
    let gq8 = read(y_mmq.as_ref());

    println!(
        "row 0 first 3 — f64 ref: {:.5} {:.5} {:.5}",
        href[0], href[1], href[2]
    );
    println!(
        "row 0 first 3 — f32-act: {:.5} {:.5} {:.5}",
        g32[0], g32[1], g32[2]
    );
    println!(
        "row 0 first 3 — int8   : {:.5} {:.5} {:.5}",
        gq8[0], gq8[1], gq8[2]
    );
    let (l2_32, worst_32) = err_stats(&g32, &href);
    let (l2_q8, worst_q8) = err_stats(&gq8, &href);
    println!("f32-activation GEMV: rel L2 {l2_32:.3e}, worst element {worst_32:.3e}");
    println!("int8 mmq GEMM:       rel L2 {l2_q8:.3e}, worst element {worst_q8:.3e}");

    // The f32-activation route shares the (bit-exact) weight decode and only reorders an f32 sum,
    // so it must land at f32 rounding. This is the assertion that makes the int8 number above a
    // measurement of the PREPASS rather than of the kernel pair.
    assert!(
        l2_32 < 1e-5,
        "the f32-activation route itself diverges from the f64 reference: rel L2 {l2_32:.3e}"
    );
}
