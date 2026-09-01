//! The qwen3vl_merger ViT forward (stage V7) — Vulkan execution.
//!
//! Mirrors the CPU forward in `vit.rs` op-for-op (same math), but runs on INFR's Vulkan graph
//! backend with weights uploaded NATIVELY (f16 matrices as raw f16 bytes — never dequantized;
//! small vectors dequantized host-side to f32 for the f32-only `layernorm`/`add_bias` shaders).
//! The engine structure follows `infr-embedding`'s `NativeEmbeddingEngine`: one graph plan per
//! grid shape (cached), weights resident in GPU buffers, per-execute input/output bindings.
//!
//! dtype flow per block: all activations f32; the fused QKV projection runs as THREE `Op::Linear`s
//! on contiguous column slices of the `[in, 3d]` weight (GGUF ne0-first = output-major, so each
//! q/k/v output block is one contiguous byte range — no output-slicing copies); Q/K rope in f32;
//! then whole-buffer f32→f16 casts (`Op::Copy`, the adapter's `store_f16` path, src_off==0)
//! because `Op::Attention`'s Vulkan kernels read q/k/v as f16 (attn_partial's `f16vec4` loads).
//!
//! Activation tensors are declared 2-D `[rows, dim]`: the Vulkan scratch allocator 64-row-pads
//! Internal buffers using `shape[0]` as the row count and the prefill GEMM tiers write
//! `ceil(m/64)*64` rows into an Internal dst — a flat desc would break that invariant.
//!
//! The 2×2 spatial merge is a pure VIEW: token t's 4 merge-major patches ARE the contiguous
//! rows `4t..4t+3`, so `[n, d]` and `[n_tok, 4d]` share one memory layout (the CPU forward's
//! reshape copy is byte-identical bookkeeping) — post-LN feeds straight into mm.0.

use anyhow::{anyhow, bail, Context, Result};
use infr_core::{
    backend::{Backend, Bindings, Buffer, BufferUsage, Plan},
    graph::{AttnMask, Graph, Op},
    loader::TensorInfo,
    tensor::{DType, TensorDesc, TensorId},
    WeightSource,
};
use infr_gguf::Gguf;
use infr_vulkan::VulkanBackend;
use std::{collections::HashMap, path::Path, sync::Mutex};

use crate::vit::VIT_THETA;
use crate::{ClipConfig, PreparedImage, VisionWeights};

/// One uploaded weight: final device bytes + the graph tensor descriptor they fill.
struct WeightSpec {
    label: String,
    desc: TensorDesc,
    bytes: Vec<u8>,
}

/// Per-block weight-spec indices.
#[derive(Clone, Copy)]
struct BlockSpecs {
    ln1_w: usize,
    ln1_b: usize,
    q_w: usize,
    k_w: usize,
    v_w: usize,
    q_b: usize,
    k_b: usize,
    v_b: usize,
    out_w: usize,
    out_b: usize,
    ln2_w: usize,
    ln2_b: usize,
    up_w: usize,
    up_b: usize,
    down_w: usize,
    down_b: usize,
}

struct SpecLayout {
    patch_embd_w: usize,
    patch_embd_b: usize,
    blocks: Vec<BlockSpecs>,
    post_ln_w: usize,
    post_ln_b: usize,
    mm0_w: usize,
    mm0_b: usize,
    mm2_w: usize,
    mm2_b: usize,
}

struct VitPlan {
    plan: Box<dyn Plan>,
    patches: TensorId,
    pos: TensorId,
    pos_hw: TensorId,
    output: TensorId,
    weight_ids: Vec<TensorId>,
    patches_buf: Box<dyn Buffer>,
    pos_buf: Box<dyn Buffer>,
    pos_hw_buf: Box<dyn Buffer>,
    out_buf: Box<dyn Buffer>,
}

/// A Vulkan vision tower. Weights live in GPU buffers once; each distinct patch-grid shape gets
/// one compiled plan (cached in a HashMap keyed by `(n_patches, n_tokens)`).
pub struct VkVit {
    cfg: ClipConfig,
    pos_table: Vec<f32>,
    backend: Box<dyn Backend>,
    layout: SpecLayout,
    weight_descs: Vec<TensorDesc>,
    weights: Vec<Box<dyn Buffer>>,
    plans: Mutex<HashMap<(usize, usize), VitPlan>>,
}

/// Raw bytes of a GGUF matrix column slice (f16/f32 only — the dtypes the Vulkan Linear kernels
/// read without dequant). GGUF ne0-first = output-major: outputs `out_lo..out_hi` are one
/// contiguous element range `[in*out_lo, in*out_hi)`.
fn matrix_slice_spec(
    g: &Gguf,
    info: &TensorInfo,
    out_lo: usize,
    out_hi: usize,
) -> Result<WeightSpec> {
    let name = &info.name;
    if !matches!(info.dtype, DType::F16 | DType::F32) {
        bail!(
            "mmproj tensor {name} is {:?}; the Vulkan ViT uploads matrices natively and only \
             f16/f32 have native Linear reads",
            info.dtype
        );
    }
    let in_f: usize = info.shape[..info.shape.len() - 1].iter().product();
    let out_f = *info.shape.last().context("matrix tensor with no out dim")?;
    if out_lo >= out_hi || out_hi > out_f {
        bail!("mmproj tensor {name}: bad column slice {out_lo}..{out_hi} of {out_f}");
    }
    let eb = info.dtype.dense_bytes(1).context("dense dtype")?;
    let bytes = g
        .tensor_bytes(name)
        .map_err(|e| anyhow!("{e}"))
        .with_context(|| format!("read mmproj tensor {name}"))?;
    let lo = in_f * out_lo * eb;
    let hi = in_f * out_hi * eb;
    let desc = TensorDesc::new(vec![in_f, out_hi - out_lo], info.dtype);
    Ok(WeightSpec {
        label: format!("{name}[cols {out_lo}..{out_hi}]"),
        desc,
        bytes: bytes[lo..hi].to_vec(),
    })
}

/// Host-dequantized f32 vector slice (norm weights/biases and projection biases — the
/// `layernorm.comp`/`add_bias.comp` shaders read f32 only, and these vectors are tiny).
fn f32_vec_spec(g: &Gguf, info: &TensorInfo, off: usize, len: usize) -> Result<WeightSpec> {
    let name = &info.name;
    let raw = g.tensor_bytes(name).map_err(|e| anyhow!("{e}"))?;
    let full = infr_gguf::dequant::dequant_block(info.dtype, &raw)
        .with_context(|| format!("dequant {name}"))?;
    if off + len > full.len() {
        bail!(
            "mmproj tensor {name}: slice {off}..{} of {}",
            off + len,
            full.len()
        );
    }
    let mut bytes = Vec::with_capacity(len * 4);
    for v in &full[off..off + len] {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    Ok(WeightSpec {
        label: format!("{name}[{off}..{}]", off + len),
        desc: TensorDesc::new(vec![len], DType::F32),
        bytes,
    })
}

impl VkVit {
    /// Open the mmproj, create the Vulkan backend, build the weight catalog, and upload every
    /// weight natively (matrices raw f16/f32 bytes; vectors dequantized f32). Pinned to physical
    /// device `dev` (`Some(idx)` = `VulkanN`, via `VulkanBackend::new_on_with` — PR #21 review:
    /// the vision tower used to open its OWN backend on the DEFAULT device, so on a multi-GPU /
    /// iGPU+dGPU machine it could land on a different GPU than the model the caller selected).
    /// `None` = the default device.
    pub(crate) fn load_on(path: &Path, dev: Option<usize>) -> Result<Self> {
        if !path.is_file() {
            bail!("mmproj does not exist: {}", path.display());
        }
        let cfg = std::sync::Arc::new(infr_core::config::Config::default());
        let backend = match dev {
            Some(idx) => VulkanBackend::new_on_with(idx, cfg)
                .map_err(|e| anyhow!("initialize Vulkan vision backend (Vulkan{idx}): {e}"))?,
            None => VulkanBackend::new_with(cfg)
                .map_err(|e| anyhow!("initialize Vulkan vision backend: {e}"))?,
        };
        let g = Gguf::open(path)?;
        let w = VisionWeights::load(&g)?;
        let cfg = ClipConfig::from_gguf(&g)?;
        if cfg.is_deepstack_layers.iter().any(|&d| d) {
            bail!("mmproj ships deepstack tensors — deepstack injection is not implemented yet");
        }
        let d = cfg.embedding_length;
        let ff = cfg.feed_forward_length;
        let merge2 = cfg.spatial_merge_size * cfg.spatial_merge_size;

        let mut specs: Vec<WeightSpec> = Vec::new();
        fn push(specs: &mut Vec<WeightSpec>, spec: Result<WeightSpec>) -> Result<usize> {
            specs.push(spec?);
            Ok(specs.len() - 1)
        }
        let patch_embd_w = push(
            &mut specs,
            matrix_slice_spec(&g, &w.patch_embd_weight, 0, d),
        )?;
        let patch_embd_b = push(&mut specs, f32_vec_spec(&g, &w.patch_embd_bias, 0, d))?;
        let mut blocks = Vec::with_capacity(cfg.block_count);
        for b in &w.blocks {
            let q_w = push(&mut specs, matrix_slice_spec(&g, &b.attn_qkv_weight, 0, d))?;
            let k_w = push(
                &mut specs,
                matrix_slice_spec(&g, &b.attn_qkv_weight, d, 2 * d),
            )?;
            let v_w = push(
                &mut specs,
                matrix_slice_spec(&g, &b.attn_qkv_weight, 2 * d, 3 * d),
            )?;
            let q_b = push(&mut specs, f32_vec_spec(&g, &b.attn_qkv_bias, 0, d))?;
            let k_b = push(&mut specs, f32_vec_spec(&g, &b.attn_qkv_bias, d, d))?;
            let v_b = push(&mut specs, f32_vec_spec(&g, &b.attn_qkv_bias, 2 * d, d))?;
            blocks.push(BlockSpecs {
                ln1_w: push(&mut specs, f32_vec_spec(&g, &b.ln1_weight, 0, d))?,
                ln1_b: push(&mut specs, f32_vec_spec(&g, &b.ln1_bias, 0, d))?,
                q_w,
                k_w,
                v_w,
                q_b,
                k_b,
                v_b,
                out_w: push(&mut specs, matrix_slice_spec(&g, &b.attn_out_weight, 0, d))?,
                out_b: push(&mut specs, f32_vec_spec(&g, &b.attn_out_bias, 0, d))?,
                ln2_w: push(&mut specs, f32_vec_spec(&g, &b.ln2_weight, 0, d))?,
                ln2_b: push(&mut specs, f32_vec_spec(&g, &b.ln2_bias, 0, d))?,
                up_w: push(&mut specs, matrix_slice_spec(&g, &b.ffn_up_weight, 0, ff))?,
                up_b: push(&mut specs, f32_vec_spec(&g, &b.ffn_up_bias, 0, ff))?,
                down_w: push(&mut specs, matrix_slice_spec(&g, &b.ffn_down_weight, 0, d))?,
                down_b: push(&mut specs, f32_vec_spec(&g, &b.ffn_down_bias, 0, d))?,
            });
        }
        let post_ln_w = push(&mut specs, f32_vec_spec(&g, &w.post_ln_weight, 0, d))?;
        let post_ln_b = push(&mut specs, f32_vec_spec(&g, &w.post_ln_bias, 0, d))?;
        let mm0_w = push(
            &mut specs,
            matrix_slice_spec(&g, &w.mm0_weight, 0, d * merge2),
        )?;
        let mm0_b = push(&mut specs, f32_vec_spec(&g, &w.mm0_bias, 0, d * merge2))?;
        let mm2_w = push(
            &mut specs,
            matrix_slice_spec(&g, &w.mm2_weight, 0, cfg.projection_dim),
        )?;
        let mm2_b = push(
            &mut specs,
            f32_vec_spec(&g, &w.mm2_bias, 0, cfg.projection_dim),
        )?;
        let pos_raw = g
            .tensor_bytes(&w.position_embd_weight.name)
            .map_err(|e| anyhow!("{e}"))?;
        let pos_table = infr_gguf::dequant::dequant_block(w.position_embd_weight.dtype, &pos_raw)
            .context("dequant v.position_embd.weight")?;

        // Upload: weights → device buffers (native bytes verbatim), then drop host staging bytes.
        let mut weights = Vec::with_capacity(specs.len());
        let mut total = 0usize;
        for spec in &specs {
            let buf = backend
                .alloc_uninit(spec.bytes.len(), BufferUsage::Weights)
                .map_err(|e| anyhow!("allocate {}: {e}", spec.label))?;
            backend
                .upload(buf.as_ref(), &spec.bytes)
                .map_err(|e| anyhow!("upload {}: {e}", spec.label))?;
            total += spec.bytes.len();
            weights.push(buf);
        }
        let weight_descs = specs.iter().map(|s| s.desc.clone()).collect();
        tracing::info!(
            tensors = specs.len(),
            mib = total as f64 / 1048576.0,
            "vulkan vision tower weights resident"
        );
        Ok(Self {
            cfg,
            pos_table,
            backend: Box::new(backend),
            layout: SpecLayout {
                patch_embd_w,
                patch_embd_b,
                blocks,
                post_ln_w,
                post_ln_b,
                mm0_w,
                mm0_b,
                mm2_w,
                mm2_b,
            },
            weight_descs,
            weights,
            plans: Mutex::new(HashMap::new()),
        })
    }

    /// The dequantized base position-embed table, for `prepare_image_bytes`.
    pub(crate) fn pos_table(&self) -> &[f32] {
        &self.pos_table
    }

    pub(crate) fn config(&self) -> &ClipConfig {
        &self.cfg
    }

    fn build_plan(&self, n: usize, n_tok: usize) -> Result<VitPlan> {
        let cfg = &self.cfg;
        let d = cfg.embedding_length;
        let ff = cfg.feed_forward_length;
        let merge2 = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let nh = cfg.head_count;
        let hd = cfg.head_dim;
        let proj = cfg.projection_dim;
        let patch_in = 3 * cfg.patch_size * cfg.patch_size;
        let eps = cfg.layer_norm_epsilon;
        let scale = 1.0 / (hd as f32).sqrt();
        let sections = [(hd as u32) / 4; 4];

        let mut graph = Graph::new();
        // Activations are 2-D [rows, dim] (see the module docs for the 64-row-padding invariant).
        let f32d = |rows: usize, cols: usize| TensorDesc::new(vec![rows, cols], DType::F32);
        let f16d = |rows: usize, cols: usize| TensorDesc::new(vec![rows, cols], DType::F16);
        let patches = graph.input(f32d(n, patch_in));
        let pos = graph.input(f32d(n, d));
        let pos_hw = graph.input(TensorDesc::new(vec![n, 2], DType::I32));
        let output = graph.output(f32d(n_tok, proj));
        let weight_ids = self
            .weight_descs
            .iter()
            .map(|desc| graph.weight(desc.clone()))
            .collect::<Vec<_>>();
        let wid = |i: usize| weight_ids[i];
        let lay = &self.layout;

        // Activation scratch (f32 unless suffixed 16). x[0]/x[1] ping-pong the residual stream so
        // no Add ever rewrites a buffer another still-live op reads.
        let pe = graph.internal(f32d(n, d));
        let x = [graph.internal(f32d(n, d)), graph.internal(f32d(n, d))];
        let normed = graph.internal(f32d(n, d));
        let q32 = graph.internal(f32d(n, d));
        let k32 = graph.internal(f32d(n, d));
        let v32 = graph.internal(f32d(n, d));
        let qr = graph.internal(f32d(n, d));
        let kr = graph.internal(f32d(n, d));
        let q16 = graph.internal(f16d(n, d));
        let k16 = graph.internal(f16d(n, d));
        let v16 = graph.internal(f16d(n, d));
        let attn = graph.internal(f32d(n, d));
        let o32 = graph.internal(f32d(n, d));
        let up = graph.internal(f32d(n, ff));
        let act = graph.internal(f32d(n, ff));
        let down = graph.internal(f32d(n, d));
        let pl = graph.internal(f32d(n, d));
        let m0 = graph.internal(f32d(n_tok, d * merge2));
        let mg = graph.internal(f32d(n_tok, d * merge2));

        // ── patch embed + bias + position add ─────────────────────────────────────────────
        graph.push(Op::Linear {
            x: patches,
            weight: wid(lay.patch_embd_w),
            dst: pe,
            m: n as u32,
            in_f: patch_in as u32,
            out_f: d as u32,
            w_off: 0,
        });
        graph.push(Op::AddBias {
            x: pe,
            bias: wid(lay.patch_embd_b),
            dst: pe,
            rows: n as u32,
            n: d as u32,
        });
        graph.push(Op::Add {
            a: pe,
            b: pos,
            dst: x[0],
            n: (n * d) as u32,
        });

        // ── 27 transformer blocks ─────────────────────────────────────────────────────────
        // Residual threading: the block reads `xi` = x[cur], writes the attention residual into
        // `xo` = x[1-cur], adds the MLP output back IN PLACE into `xo`, and flips `cur` — the
        // next block then reads the freshly written stream.
        let mut cur = 0usize;
        for blk in &lay.blocks {
            let (xi, xo) = (x[cur], x[1 - cur]);
            graph.push(Op::LayerNorm {
                x: xi,
                weight: wid(blk.ln1_w),
                bias: wid(blk.ln1_b),
                dst: normed,
                rows: n as u32,
                dim: d as u32,
                eps,
            });
            for (dst, w) in [(q32, blk.q_w), (k32, blk.k_w), (v32, blk.v_w)] {
                graph.push(Op::Linear {
                    x: normed,
                    weight: wid(w),
                    dst,
                    m: n as u32,
                    in_f: d as u32,
                    out_f: d as u32,
                    w_off: 0,
                });
            }
            for (t, b) in [(q32, blk.q_b), (k32, blk.k_b), (v32, blk.v_b)] {
                graph.push(Op::AddBias {
                    x: t,
                    bias: wid(b),
                    dst: t,
                    rows: n as u32,
                    n: d as u32,
                });
            }
            graph.push(Op::Rope2D {
                q: q32,
                k: k32,
                pos_hw,
                dst_q: qr,
                dst_k: kr,
                n_head: nh as u32,
                head_dim: hd as u32,
                theta: VIT_THETA,
                sections,
            });
            // f32→f16 casts: Op::Attention's Vulkan kernels read q/k/v as f16 (whole-buffer
            // elementwise cast — the adapter's cross-dtype Copy path needs src_off == 0).
            for (src, dst) in [(qr, q16), (kr, k16), (v32, v16)] {
                graph.push(Op::Copy {
                    src,
                    src_off: 0,
                    dst,
                    dst_off: 0,
                    n: (n * d) as u32,
                });
            }
            graph.push(Op::Attention {
                q: q16,
                k_cache: k16,
                v_cache: v16,
                dst: attn,
                rows: n as u32,
                kv_len: n as u32,
                n_head: nh as u32,
                n_kv: nh as u32,
                head_dim: hd as u32,
                scale,
                mask: AttnMask::Canvas { lo: 0 },
                pos: 0,
                sinks: None,
            });
            graph.push(Op::Linear {
                x: attn,
                weight: wid(blk.out_w),
                dst: o32,
                m: n as u32,
                in_f: d as u32,
                out_f: d as u32,
                w_off: 0,
            });
            graph.push(Op::AddBias {
                x: o32,
                bias: wid(blk.out_b),
                dst: o32,
                rows: n as u32,
                n: d as u32,
            });
            graph.push(Op::Add {
                a: xi,
                b: o32,
                dst: xo,
                n: (n * d) as u32,
            });
            graph.push(Op::LayerNorm {
                x: xo,
                weight: wid(blk.ln2_w),
                bias: wid(blk.ln2_b),
                dst: normed,
                rows: n as u32,
                dim: d as u32,
                eps,
            });
            graph.push(Op::Linear {
                x: normed,
                weight: wid(blk.up_w),
                dst: up,
                m: n as u32,
                in_f: d as u32,
                out_f: ff as u32,
                w_off: 0,
            });
            graph.push(Op::AddBias {
                x: up,
                bias: wid(blk.up_b),
                dst: up,
                rows: n as u32,
                n: ff as u32,
            });
            graph.push(Op::Gelu {
                x: up,
                dst: act,
                rows: n as u32,
                cols: ff as u32,
            });
            graph.push(Op::Linear {
                x: act,
                weight: wid(blk.down_w),
                dst: down,
                m: n as u32,
                in_f: ff as u32,
                out_f: d as u32,
                w_off: 0,
            });
            graph.push(Op::AddBias {
                x: down,
                bias: wid(blk.down_b),
                dst: down,
                rows: n as u32,
                n: d as u32,
            });
            graph.push(Op::Add {
                a: xo,
                b: down,
                dst: xo,
                n: (n * d) as u32,
            });
            cur = 1 - cur;
        }
        // After the loop the residual stream is in x[cur].
        let last_x = x[cur];

        // ── post-LN + merger MLP ──────────────────────────────────────────────────────────
        graph.push(Op::LayerNorm {
            x: last_x,
            weight: wid(lay.post_ln_w),
            bias: wid(lay.post_ln_b),
            dst: pl,
            rows: n as u32,
            dim: d as u32,
            eps,
        });
        graph.push(Op::Linear {
            x: pl,
            weight: wid(lay.mm0_w),
            dst: m0,
            m: n_tok as u32,
            in_f: (d * merge2) as u32,
            out_f: (d * merge2) as u32,
            w_off: 0,
        });
        graph.push(Op::AddBias {
            x: m0,
            bias: wid(lay.mm0_b),
            dst: m0,
            rows: n_tok as u32,
            n: (d * merge2) as u32,
        });
        graph.push(Op::Gelu {
            x: m0,
            dst: mg,
            rows: n_tok as u32,
            cols: (d * merge2) as u32,
        });
        graph.push(Op::Linear {
            x: mg,
            weight: wid(lay.mm2_w),
            dst: output,
            m: n_tok as u32,
            in_f: (d * merge2) as u32,
            out_f: proj as u32,
            w_off: 0,
        });
        graph.push(Op::AddBias {
            x: output,
            bias: wid(lay.mm2_b),
            dst: output,
            rows: n_tok as u32,
            n: proj as u32,
        });

        let plan = self
            .backend
            .compile(&graph)
            .map_err(|e| anyhow!("compile vulkan ViT graph (n={n}): {e}"))?;
        let patches_buf = self
            .backend
            .alloc_uninit(n * patch_in * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("alloc patches staging: {e}"))?;
        let pos_buf = self
            .backend
            .alloc_uninit(n * d * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("alloc pos staging: {e}"))?;
        let pos_hw_buf = self
            .backend
            .alloc_uninit(n * 2 * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("alloc pos_hw staging: {e}"))?;
        let out_buf = self
            .backend
            .alloc_uninit(n_tok * proj * 4, BufferUsage::Readback)
            .map_err(|e| anyhow!("alloc output readback: {e}"))?;
        Ok(VitPlan {
            plan,
            patches,
            pos,
            pos_hw,
            output,
            weight_ids,
            patches_buf,
            pos_buf,
            pos_hw_buf,
            out_buf,
        })
    }

    fn execute_plan(&self, p: &mut VitPlan, img: &PreparedImage) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let n = img.n_patches();
        let n_tok = img.grid_nx * img.grid_ny;
        // 2D rope positions (y, x) per patch, merge-major — same as the CPU forward.
        let nx = img.grid_nx * cfg.spatial_merge_size;
        let mut pos_hw = vec![0i32; n * 2];
        for (i, slot) in pos_hw.chunks_mut(2).enumerate() {
            let (py, px) = crate::preprocess::merge_major_pos(i, nx, cfg.spatial_merge_size);
            slot[0] = py as i32;
            slot[1] = px as i32;
        }
        self.backend
            .upload(p.patches_buf.as_ref(), bytemuck::cast_slice(&img.patches))
            .map_err(|e| anyhow!("upload patches: {e}"))?;
        self.backend
            .upload(p.pos_buf.as_ref(), bytemuck::cast_slice(&img.pos_embed))
            .map_err(|e| anyhow!("upload pos embed: {e}"))?;
        self.backend
            .upload(p.pos_hw_buf.as_ref(), bytemuck::cast_slice(&pos_hw))
            .map_err(|e| anyhow!("upload pos_hw: {e}"))?;
        let mut bindings = Bindings::new();
        bindings
            .bind(p.patches, p.patches_buf.as_ref())
            .bind(p.pos, p.pos_buf.as_ref())
            .bind(p.pos_hw, p.pos_hw_buf.as_ref())
            .bind(p.output, p.out_buf.as_ref());
        for (id, buf) in p.weight_ids.iter().zip(&self.weights) {
            bindings.bind(*id, buf.as_ref());
        }
        self.backend
            .execute(p.plan.as_ref(), &bindings)
            .map_err(|e| anyhow!("execute vulkan ViT graph: {e}"))?;
        let mut bytes = vec![0u8; n_tok * cfg.projection_dim * 4];
        self.backend
            .download(p.out_buf.as_ref(), &mut bytes)
            .map_err(|e| anyhow!("download vulkan ViT output: {e}"))?;
        let out = bytemuck::cast_slice::<u8, f32>(&bytes).to_vec();
        if out.iter().any(|v| !v.is_finite()) {
            bail!("Vulkan ViT forward produced non-finite values");
        }
        Ok(out)
    }

    pub(crate) fn encode(&self, img: &PreparedImage) -> Result<Vec<f32>> {
        let n = img.n_patches();
        let n_tok = img.grid_nx * img.grid_ny;
        let mut plans = self.plans.lock().unwrap_or_else(|e| e.into_inner());
        if !plans.contains_key(&(n, n_tok)) {
            let plan = self.build_plan(n, n_tok)?;
            plans.insert((n, n_tok), plan);
        }
        let plan = plans.get_mut(&(n, n_tok)).expect("plan inserted above");
        self.execute_plan(plan, img)
    }
}
