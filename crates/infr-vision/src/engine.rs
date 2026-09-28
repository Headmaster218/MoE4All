//! Request-scoped Vulkan execution for the Qwen3-VL vision projector.

use crate::{merge_major_pos, prepare_image_bytes, ClipConfig, PreparedImage, VisionWeights};
use anyhow::{anyhow, bail, Context, Result};
use infr_core::{
    backend::{Backend, Bindings, Buffer, BufferUsage, Plan},
    graph::{AttnMask, Graph, Op},
    loader::TensorInfo,
    tensor::{DType, TensorDesc, TensorId},
    MemoryTier, ResourceKind, ResourceSnapshot, ResourceTracker, WeightSource,
};
use infr_gguf::Gguf;
use infr_vulkan::VulkanBackend;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, VecDeque},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

const VIT_THETA: f32 = 10_000.0;
const EMBEDDING_CACHE_MAX_BYTES: usize = 256 * 1024 * 1024;
const EMBEDDING_CACHE_MAX_IMAGES: usize = 16;

#[derive(Clone)]
enum WeightPayload {
    Raw {
        source_offset: usize,
        nbytes: usize,
    },
    F32Slice {
        element_offset: usize,
        elements: usize,
    },
}

#[derive(Clone)]
struct WeightSpec {
    label: String,
    source: String,
    desc: TensorDesc,
    payload: WeightPayload,
}

impl WeightSpec {
    fn nbytes(&self) -> usize {
        match self.payload {
            WeightPayload::Raw { nbytes, .. } => nbytes,
            WeightPayload::F32Slice { elements, .. } => elements * size_of::<f32>(),
        }
    }
}

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
    patch_embd_w_temporal: Option<usize>,
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

/// One image's merged visual tokens, in row-major `[tokens, projection_dim]` order.
#[derive(Clone)]
pub struct VisionEmbedding {
    pub values: Arc<Vec<f32>>,
    pub grid_nx: usize,
    pub grid_ny: usize,
    pub fingerprint: [u8; 32],
}

impl VisionEmbedding {
    pub fn n_tokens(&self) -> usize {
        self.grid_nx * self.grid_ny
    }
}

#[derive(Default)]
struct VisionEmbeddingCache {
    entries: VecDeque<([u8; 32], VisionEmbedding)>,
    bytes: usize,
}

impl VisionEmbeddingCache {
    fn get(&mut self, key: &[u8; 32]) -> Option<VisionEmbedding> {
        let index = self
            .entries
            .iter()
            .position(|(candidate, _)| candidate == key)?;
        let entry = self
            .entries
            .remove(index)
            .expect("cache index came from entries");
        let value = entry.1.clone();
        self.entries.push_back(entry);
        Some(value)
    }

    fn insert(&mut self, key: [u8; 32], value: VisionEmbedding) {
        let bytes = value.values.len().saturating_mul(size_of::<f32>());
        if bytes > EMBEDDING_CACHE_MAX_BYTES {
            return;
        }
        if let Some(index) = self
            .entries
            .iter()
            .position(|(candidate, _)| candidate == &key)
        {
            let (_, old) = self
                .entries
                .remove(index)
                .expect("cache index came from entries");
            self.bytes = self
                .bytes
                .saturating_sub(old.values.len().saturating_mul(size_of::<f32>()));
        }
        while !self.entries.is_empty()
            && (self.entries.len() >= EMBEDDING_CACHE_MAX_IMAGES
                || self.bytes.saturating_add(bytes) > EMBEDDING_CACHE_MAX_BYTES)
        {
            let (_, old) = self.entries.pop_front().expect("cache is not empty");
            self.bytes = self
                .bytes
                .saturating_sub(old.values.len().saturating_mul(size_of::<f32>()));
        }
        self.bytes = self.bytes.saturating_add(bytes);
        self.entries.push_back((key, value));
    }
}

fn image_fingerprint(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// A vision tower attached to the LLM's existing Vulkan device and unified VRAM arena.
///
/// Only metadata and the position table remain resident between calls. Each call admits all
/// projector weights directly from the mmproj file, encodes the complete image batch, then drops
/// both weights and graph runtime so the same arena ranges return to the expert cache.
pub struct NativeVisionEngine {
    model_path: PathBuf,
    cfg: ClipConfig,
    pos_table: Vec<f32>,
    backend: VulkanBackend,
    specs: Vec<WeightSpec>,
    layout: SpecLayout,
    weight_bytes: u64,
    execution: Mutex<()>,
    embedding_cache: Mutex<VisionEmbeddingCache>,
    resource: Arc<ResourceTracker>,
}

impl NativeVisionEngine {
    fn build_plan(&self, n: usize, n_tokens: usize) -> Result<VitPlan> {
        let cfg = &self.cfg;
        let d = cfg.embedding_length;
        let ff = cfg.feed_forward_length;
        let merge2 = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let patch_in = 3 * cfg.patch_size * cfg.patch_size;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let sections = [(cfg.head_dim as u32) / 4; 4];

        let mut graph = Graph::new();
        let f32d = |rows: usize, cols: usize| TensorDesc::new(vec![rows, cols], DType::F32);
        let f16d = |rows: usize, cols: usize| TensorDesc::new(vec![rows, cols], DType::F16);
        let patches = graph.input(f32d(n, patch_in));
        let pos = graph.input(f32d(n, d));
        let pos_hw = graph.input(TensorDesc::new(vec![n, 2], DType::I32));
        let output = graph.output(f32d(n_tokens, cfg.projection_dim));
        let weight_ids = self
            .specs
            .iter()
            .map(|spec| {
                let id = graph.weight(spec.desc.clone());
                graph.label(id, spec.label.clone())
            })
            .collect::<Vec<_>>();
        let wid = |index: usize| weight_ids[index];
        let layout = &self.layout;

        let patch_embed_first = graph.internal(f32d(n, d));
        let patch_embed_temporal = layout
            .patch_embd_w_temporal
            .map(|weight| (weight, graph.internal(f32d(n, d))));
        let patch_embed = if patch_embed_temporal.is_some() {
            graph.internal(f32d(n, d))
        } else {
            patch_embed_first
        };
        let state = [graph.internal(f32d(n, d)), graph.internal(f32d(n, d))];
        let normed = graph.internal(f32d(n, d));
        let q = graph.internal(f32d(n, d));
        let k = graph.internal(f32d(n, d));
        let v = graph.internal(f32d(n, d));
        let q_roped = graph.internal(f32d(n, d));
        let k_roped = graph.internal(f32d(n, d));
        let q16 = graph.internal(f16d(n, d));
        let k16 = graph.internal(f16d(n, d));
        let v16 = graph.internal(f16d(n, d));
        let attention = graph.internal(f32d(n, d));
        let projected = graph.internal(f32d(n, d));
        let up = graph.internal(f32d(n, ff));
        let activated = graph.internal(f32d(n, ff));
        let down = graph.internal(f32d(n, d));
        let post_ln = graph.internal(f32d(n, d));
        let merged = graph.internal(f32d(n_tokens, d * merge2));
        let merged_act = graph.internal(f32d(n_tokens, d * merge2));

        graph.push(Op::Linear {
            x: patches,
            weight: wid(layout.patch_embd_w),
            dst: patch_embed_first,
            m: n as u32,
            in_f: patch_in as u32,
            out_f: d as u32,
            w_off: 0,
        });
        if let Some((weight, temporal)) = patch_embed_temporal {
            // The source Conv3D has temporal depth two. GGUF stores its slices separately;
            // still images duplicate one frame, so both projections are summed before bias.
            graph.push(Op::Linear {
                x: patches,
                weight: wid(weight),
                dst: temporal,
                m: n as u32,
                in_f: patch_in as u32,
                out_f: d as u32,
                w_off: 0,
            });
            graph.push(Op::Add {
                a: patch_embed_first,
                b: temporal,
                dst: patch_embed,
                n: (n * d) as u32,
            });
        }
        graph.push(Op::AddBias {
            x: patch_embed,
            bias: wid(layout.patch_embd_b),
            dst: patch_embed,
            rows: n as u32,
            n: d as u32,
        });
        graph.push(Op::Add {
            a: patch_embed,
            b: pos,
            dst: state[0],
            n: (n * d) as u32,
        });

        let mut current = 0usize;
        for block in &layout.blocks {
            let (input, next) = (state[current], state[1 - current]);
            graph.push(Op::LayerNorm {
                x: input,
                weight: wid(block.ln1_w),
                bias: wid(block.ln1_b),
                dst: normed,
                rows: n as u32,
                dim: d as u32,
                eps: cfg.layer_norm_epsilon,
            });
            for (dst, weight) in [(q, block.q_w), (k, block.k_w), (v, block.v_w)] {
                graph.push(Op::Linear {
                    x: normed,
                    weight: wid(weight),
                    dst,
                    m: n as u32,
                    in_f: d as u32,
                    out_f: d as u32,
                    w_off: 0,
                });
            }
            for (tensor, bias) in [(q, block.q_b), (k, block.k_b), (v, block.v_b)] {
                graph.push(Op::AddBias {
                    x: tensor,
                    bias: wid(bias),
                    dst: tensor,
                    rows: n as u32,
                    n: d as u32,
                });
            }
            graph.push(Op::Rope2D {
                q,
                k,
                pos_hw,
                dst_q: q_roped,
                dst_k: k_roped,
                n_head: cfg.head_count as u32,
                head_dim: cfg.head_dim as u32,
                theta: VIT_THETA,
                sections,
            });
            for (src, dst) in [(q_roped, q16), (k_roped, k16), (v, v16)] {
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
                dst: attention,
                rows: n as u32,
                kv_len: n as u32,
                n_head: cfg.head_count as u32,
                n_kv: cfg.head_count as u32,
                head_dim: cfg.head_dim as u32,
                scale,
                mask: AttnMask::Canvas { lo: 0 },
                pos: 0,
                sinks: None,
            });
            graph.push(Op::Linear {
                x: attention,
                weight: wid(block.out_w),
                dst: projected,
                m: n as u32,
                in_f: d as u32,
                out_f: d as u32,
                w_off: 0,
            });
            graph.push(Op::AddBias {
                x: projected,
                bias: wid(block.out_b),
                dst: projected,
                rows: n as u32,
                n: d as u32,
            });
            graph.push(Op::Add {
                a: input,
                b: projected,
                dst: next,
                n: (n * d) as u32,
            });
            graph.push(Op::LayerNorm {
                x: next,
                weight: wid(block.ln2_w),
                bias: wid(block.ln2_b),
                dst: normed,
                rows: n as u32,
                dim: d as u32,
                eps: cfg.layer_norm_epsilon,
            });
            graph.push(Op::Linear {
                x: normed,
                weight: wid(block.up_w),
                dst: up,
                m: n as u32,
                in_f: d as u32,
                out_f: ff as u32,
                w_off: 0,
            });
            graph.push(Op::AddBias {
                x: up,
                bias: wid(block.up_b),
                dst: up,
                rows: n as u32,
                n: ff as u32,
            });
            graph.push(Op::Gelu {
                x: up,
                dst: activated,
                rows: n as u32,
                cols: ff as u32,
            });
            graph.push(Op::Linear {
                x: activated,
                weight: wid(block.down_w),
                dst: down,
                m: n as u32,
                in_f: ff as u32,
                out_f: d as u32,
                w_off: 0,
            });
            graph.push(Op::AddBias {
                x: down,
                bias: wid(block.down_b),
                dst: down,
                rows: n as u32,
                n: d as u32,
            });
            graph.push(Op::Add {
                a: next,
                b: down,
                dst: next,
                n: (n * d) as u32,
            });
            current = 1 - current;
        }

        graph.push(Op::LayerNorm {
            x: state[current],
            weight: wid(layout.post_ln_w),
            bias: wid(layout.post_ln_b),
            dst: post_ln,
            rows: n as u32,
            dim: d as u32,
            eps: cfg.layer_norm_epsilon,
        });
        graph.push(Op::Linear {
            x: post_ln,
            weight: wid(layout.mm0_w),
            dst: merged,
            m: n_tokens as u32,
            in_f: (d * merge2) as u32,
            out_f: (d * merge2) as u32,
            w_off: 0,
        });
        graph.push(Op::AddBias {
            x: merged,
            bias: wid(layout.mm0_b),
            dst: merged,
            rows: n_tokens as u32,
            n: (d * merge2) as u32,
        });
        graph.push(Op::Gelu {
            x: merged,
            dst: merged_act,
            rows: n_tokens as u32,
            cols: (d * merge2) as u32,
        });
        graph.push(Op::Linear {
            x: merged_act,
            weight: wid(layout.mm2_w),
            dst: output,
            m: n_tokens as u32,
            in_f: (d * merge2) as u32,
            out_f: cfg.projection_dim as u32,
            w_off: 0,
        });
        graph.push(Op::AddBias {
            x: output,
            bias: wid(layout.mm2_b),
            dst: output,
            rows: n_tokens as u32,
            n: cfg.projection_dim as u32,
        });

        let plan = self
            .backend
            .compile(&graph)
            .map_err(|error| anyhow!("compile vision graph for {n} patches: {error}"))?;
        let patches_buf = self.alloc(n * patch_in * size_of::<f32>(), BufferUsage::Staging)?;
        let pos_buf = self.alloc(n * d * size_of::<f32>(), BufferUsage::Staging)?;
        let pos_hw_buf = self.alloc(n * 2 * size_of::<i32>(), BufferUsage::Staging)?;
        let out_buf = self.alloc(
            n_tokens * cfg.projection_dim * size_of::<f32>(),
            BufferUsage::Readback,
        )?;
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

    fn alloc(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        self.backend
            .alloc_uninit(bytes, usage)
            .map_err(|error| anyhow!("allocate {bytes} vision bytes for {usage:?}: {error}"))
    }
}

impl NativeVisionEngine {
    pub fn load_vulkan_with_backend(path: &Path, backend: VulkanBackend) -> Result<Self> {
        if !path.is_file() {
            bail!("vision projector does not exist: {}", path.display());
        }
        let gguf = Gguf::open(path).map_err(|error| anyhow!(error.to_string()))?;
        let cfg = ClipConfig::from_gguf(&gguf)?;
        if cfg.is_deepstack_layers.iter().any(|&enabled| enabled) {
            bail!("vision projector uses deepstack, which is not implemented yet");
        }
        if !cfg.use_gelu {
            bail!("vision projector requires an unsupported non-GELU activation");
        }
        let weights = VisionWeights::load(&gguf)?;
        let (specs, layout, weight_bytes) = build_weight_catalog(&weights, &cfg)?;
        let pos_raw = gguf
            .tensor_bytes(&weights.position_embd_weight.name)
            .map_err(|error| anyhow!(error.to_string()))?;
        let pos_table =
            infr_gguf::dequant::dequant_block(weights.position_embd_weight.dtype, pos_raw)
                .context("dequantize vision position-embedding table")?;
        let expected_pos = cfg
            .embedding_length
            .checked_mul(cfg.base_grid)
            .and_then(|value| value.checked_mul(cfg.base_grid))
            .context("vision position-embedding size overflow")?;
        if pos_table.len() != expected_pos {
            bail!(
                "vision position table has {} elements; expected {expected_pos}",
                pos_table.len()
            );
        }
        let model_id = path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("qwen3vl-mmproj")
            .to_owned();
        tracing::info!(
            model = %model_id,
            tensors = specs.len(),
            weights_mib = weight_bytes as f64 / (1u64 << 20) as f64,
            blocks = cfg.block_count,
            projection_dim = cfg.projection_dim,
            "vision projector ready with request-scoped SSD residency"
        );
        Ok(Self {
            model_path: path.to_owned(),
            cfg,
            pos_table,
            backend,
            specs,
            layout,
            weight_bytes,
            execution: Mutex::new(()),
            embedding_cache: Mutex::new(VisionEmbeddingCache::default()),
            resource: Arc::new(ResourceTracker::new(
                format!("vision:{model_id}"),
                ResourceKind::VisionWeights,
                weight_bytes,
                0,
                MemoryTier::Ssd,
                weight_bytes,
            )),
        })
    }

    pub fn config(&self) -> &ClipConfig {
        &self.cfg
    }

    pub fn resource_snapshot(&self) -> ResourceSnapshot {
        self.resource.snapshot()
    }

    /// Encode every image in one request while sharing one temporary projector residency.
    pub fn encode_images(&self, images: &[Vec<u8>]) -> Result<Vec<VisionEmbedding>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let _serial = self
            .execution
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let fingerprints = images
            .iter()
            .map(|image| image_fingerprint(image))
            .collect::<Vec<_>>();
        let mut output = {
            let mut cache = self
                .embedding_cache
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            fingerprints
                .iter()
                .map(|fingerprint| cache.get(fingerprint))
                .collect::<Vec<_>>()
        };
        let misses = output
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.is_none().then_some(index))
            .collect::<Vec<_>>();
        if misses.is_empty() {
            tracing::info!(images = images.len(), "vision embedding cache hit");
            return Ok(output
                .into_iter()
                .map(|value| value.expect("all image embeddings were cached"))
                .collect());
        }
        let prepared = misses
            .iter()
            .map(|&index| prepare_image_bytes(&images[index], &self.cfg, &self.pos_table))
            .collect::<Result<Vec<_>>>()?;
        let _lease = self.resource.acquire();
        let gguf = Gguf::open(&self.model_path).map_err(|error| anyhow!(error.to_string()))?;
        let mut resident = Some(load_weight_buffers(&gguf, &self.specs, &self.backend)?);
        self.resource
            .set_residency(MemoryTier::Vram, self.weight_bytes);
        tracing::info!(
            weights_mib = self.weight_bytes as f64 / (1u64 << 20) as f64,
            images = misses.len(),
            cached_images = images.len() - misses.len(),
            "vision weights admitted to unified VRAM"
        );

        let mut plans = HashMap::new();
        let result = (|| {
            let weights = resident.as_ref().expect("vision weights loaded above");
            let mut encoded = Vec::with_capacity(prepared.len());
            for (&request_index, image) in misses.iter().zip(&prepared) {
                let key = (image.n_patches(), image.grid_nx * image.grid_ny);
                let plan = match plans.entry(key) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(self.build_plan(key.0, key.1)?)
                    }
                };
                encoded.push((
                    request_index,
                    VisionEmbedding {
                        values: Arc::new(self.execute_plan(plan, image, weights)?),
                        grid_nx: image.grid_nx,
                        grid_ny: image.grid_ny,
                        fingerprint: fingerprints[request_index],
                    },
                ));
            }
            Ok(encoded)
        })();

        let sync_result = self
            .backend
            .sync()
            .map_err(|error| anyhow!(error.to_string()));
        drop(plans);
        drop(resident.take());
        let release_result = self
            .backend
            .release_auxiliary_runtime()
            .map_err(|error| anyhow!(error.to_string()));
        self.resource.set_residency(MemoryTier::Ssd, 0);
        tracing::info!(
            weights_mib = self.weight_bytes as f64 / (1u64 << 20) as f64,
            "vision weights and runtime returned to SSD backing"
        );

        match result {
            Err(error) => {
                if let Err(cleanup) = sync_result.and(release_result) {
                    tracing::warn!(%cleanup, "vision cleanup also failed after request error");
                }
                Err(error)
            }
            Ok(encoded) => {
                sync_result.context("synchronize vision execution before releasing residency")?;
                release_result.context("release vision runtime from unified VRAM")?;
                let mut cache = self
                    .embedding_cache
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                for (index, value) in encoded {
                    cache.insert(value.fingerprint, value.clone());
                    output[index] = Some(value);
                }
                Ok(output
                    .into_iter()
                    .map(|value| value.expect("every requested image has an embedding"))
                    .collect())
            }
        }
    }

    fn execute_plan(
        &self,
        plan: &mut VitPlan,
        image: &PreparedImage,
        weights: &[Box<dyn Buffer>],
    ) -> Result<Vec<f32>> {
        let n = image.n_patches();
        let n_tokens = image.grid_nx * image.grid_ny;
        let patch_grid_x = image.grid_nx * self.cfg.spatial_merge_size;
        let mut pos_hw = vec![0i32; n * 2];
        for (index, position) in pos_hw.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let (y, x) = merge_major_pos(index, patch_grid_x, self.cfg.spatial_merge_size);
            position[0] = y as i32;
            position[1] = x as i32;
        }

        self.backend
            .upload(
                plan.patches_buf.as_ref(),
                bytemuck::cast_slice(&image.patches),
            )
            .map_err(|error| anyhow!("upload vision patches: {error}"))?;
        self.backend
            .upload(
                plan.pos_buf.as_ref(),
                bytemuck::cast_slice(&image.pos_embed),
            )
            .map_err(|error| anyhow!("upload vision position embeddings: {error}"))?;
        self.backend
            .upload(plan.pos_hw_buf.as_ref(), bytemuck::cast_slice(&pos_hw))
            .map_err(|error| anyhow!("upload vision 2D positions: {error}"))?;

        let mut bindings = Bindings::new();
        bindings
            .bind(plan.patches, plan.patches_buf.as_ref())
            .bind(plan.pos, plan.pos_buf.as_ref())
            .bind(plan.pos_hw, plan.pos_hw_buf.as_ref())
            .bind(plan.output, plan.out_buf.as_ref());
        for (id, buffer) in plan.weight_ids.iter().zip(weights) {
            bindings.bind(*id, buffer.as_ref());
        }
        self.backend
            .execute(plan.plan.as_ref(), &bindings)
            .map_err(|error| anyhow!("execute Vulkan vision graph: {error}"))?;

        let output_len = n_tokens
            .checked_mul(self.cfg.projection_dim)
            .context("vision output size overflow")?;
        let mut bytes = vec![0u8; output_len * size_of::<f32>()];
        self.backend
            .download(plan.out_buf.as_ref(), &mut bytes)
            .map_err(|error| anyhow!("download Vulkan vision output: {error}"))?;
        let output = bytemuck::cast_slice::<u8, f32>(&bytes).to_vec();
        if output.iter().any(|value| !value.is_finite()) {
            bail!("Vulkan vision graph produced non-finite values");
        }
        Ok(output)
    }
}

fn matrix_slice_spec(info: &TensorInfo, out_lo: usize, out_hi: usize) -> Result<WeightSpec> {
    if info.shape.len() < 2 {
        bail!("vision matrix {} has shape {:?}", info.name, info.shape);
    }
    if !matches!(info.dtype, DType::F16 | DType::F32 | DType::Bf16) && !info.dtype.is_quant() {
        bail!(
            "vision matrix {} uses unsupported dtype {:?}",
            info.name,
            info.dtype
        );
    }
    let in_features = info.shape[..info.shape.len() - 1]
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim))
        .context("vision matrix input size overflow")?;
    let out_features = *info
        .shape
        .last()
        .context("vision matrix has no output axis")?;
    if out_lo >= out_hi || out_hi > out_features {
        bail!(
            "vision matrix {} has invalid output slice {out_lo}..{out_hi} of {out_features}",
            info.name
        );
    }
    let (block_elements, block_bytes) = infr_gguf::block_layout(info.dtype);
    if !in_features.is_multiple_of(block_elements) {
        bail!(
            "vision matrix {} row width {in_features} is not aligned to {:?}'s {block_elements}-element blocks",
            info.name,
            info.dtype
        );
    }
    let row_bytes = in_features
        .checked_div(block_elements)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .context("vision matrix row-byte size overflow")?;
    let expected_bytes = row_bytes
        .checked_mul(out_features)
        .context("vision matrix byte size overflow")?;
    if expected_bytes != info.nbytes {
        bail!(
            "vision matrix {} occupies {} bytes; row layout implies {expected_bytes}",
            info.name,
            info.nbytes
        );
    }
    let source_offset = row_bytes
        .checked_mul(out_lo)
        .context("vision matrix slice offset overflow")?;
    let nbytes = row_bytes
        .checked_mul(out_hi - out_lo)
        .context("vision matrix slice size overflow")?;
    Ok(WeightSpec {
        label: format!("{}[out {out_lo}..{out_hi}]", info.name),
        source: info.name.clone(),
        desc: TensorDesc::new(vec![in_features, out_hi - out_lo], info.dtype),
        payload: WeightPayload::Raw {
            source_offset,
            nbytes,
        },
    })
}

fn f32_vector_spec(
    info: &TensorInfo,
    element_offset: usize,
    elements: usize,
) -> Result<WeightSpec> {
    let source_elements = info
        .shape
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim));
    let source_elements = source_elements.context("vision vector size overflow")?;
    let end = element_offset
        .checked_add(elements)
        .context("vision vector slice overflow")?;
    if elements == 0 || end > source_elements {
        bail!(
            "vision vector {} has invalid slice {element_offset}..{end} of {source_elements}",
            info.name
        );
    }
    Ok(WeightSpec {
        label: format!("{}[{element_offset}..{end}]", info.name),
        source: info.name.clone(),
        desc: TensorDesc::new(vec![elements], DType::F32),
        payload: WeightPayload::F32Slice {
            element_offset,
            elements,
        },
    })
}

fn push_spec(specs: &mut Vec<WeightSpec>, spec: Result<WeightSpec>) -> Result<usize> {
    let index = specs.len();
    specs.push(spec?);
    Ok(index)
}

fn build_weight_catalog(
    weights: &VisionWeights,
    cfg: &ClipConfig,
) -> Result<(Vec<WeightSpec>, SpecLayout, u64)> {
    let d = cfg.embedding_length;
    let ff = cfg.feed_forward_length;
    let merge2 = cfg.spatial_merge_size * cfg.spatial_merge_size;
    let mut specs = Vec::new();
    let patch_embd_w = push_spec(
        &mut specs,
        matrix_slice_spec(&weights.patch_embd_weight, 0, d),
    )?;
    let patch_embd_w_temporal = weights
        .patch_embd_weight_video
        .as_ref()
        .map(|weight| push_spec(&mut specs, matrix_slice_spec(weight, 0, d)))
        .transpose()?;
    let patch_embd_b = push_spec(&mut specs, f32_vector_spec(&weights.patch_embd_bias, 0, d))?;
    let mut blocks = Vec::with_capacity(weights.blocks.len());
    for block in &weights.blocks {
        let q_w = push_spec(&mut specs, matrix_slice_spec(&block.attn_qkv_weight, 0, d))?;
        let k_w = push_spec(
            &mut specs,
            matrix_slice_spec(&block.attn_qkv_weight, d, 2 * d),
        )?;
        let v_w = push_spec(
            &mut specs,
            matrix_slice_spec(&block.attn_qkv_weight, 2 * d, 3 * d),
        )?;
        let q_b = push_spec(&mut specs, f32_vector_spec(&block.attn_qkv_bias, 0, d))?;
        let k_b = push_spec(&mut specs, f32_vector_spec(&block.attn_qkv_bias, d, d))?;
        let v_b = push_spec(&mut specs, f32_vector_spec(&block.attn_qkv_bias, 2 * d, d))?;
        blocks.push(BlockSpecs {
            ln1_w: push_spec(&mut specs, f32_vector_spec(&block.ln1_weight, 0, d))?,
            ln1_b: push_spec(&mut specs, f32_vector_spec(&block.ln1_bias, 0, d))?,
            q_w,
            k_w,
            v_w,
            q_b,
            k_b,
            v_b,
            out_w: push_spec(&mut specs, matrix_slice_spec(&block.attn_out_weight, 0, d))?,
            out_b: push_spec(&mut specs, f32_vector_spec(&block.attn_out_bias, 0, d))?,
            ln2_w: push_spec(&mut specs, f32_vector_spec(&block.ln2_weight, 0, d))?,
            ln2_b: push_spec(&mut specs, f32_vector_spec(&block.ln2_bias, 0, d))?,
            up_w: push_spec(&mut specs, matrix_slice_spec(&block.ffn_up_weight, 0, ff))?,
            up_b: push_spec(&mut specs, f32_vector_spec(&block.ffn_up_bias, 0, ff))?,
            down_w: push_spec(&mut specs, matrix_slice_spec(&block.ffn_down_weight, 0, d))?,
            down_b: push_spec(&mut specs, f32_vector_spec(&block.ffn_down_bias, 0, d))?,
        });
    }
    let post_ln_w = push_spec(&mut specs, f32_vector_spec(&weights.post_ln_weight, 0, d))?;
    let post_ln_b = push_spec(&mut specs, f32_vector_spec(&weights.post_ln_bias, 0, d))?;
    let mm0_w = push_spec(
        &mut specs,
        matrix_slice_spec(&weights.mm0_weight, 0, d * merge2),
    )?;
    let mm0_b = push_spec(
        &mut specs,
        f32_vector_spec(&weights.mm0_bias, 0, d * merge2),
    )?;
    let mm2_w = push_spec(
        &mut specs,
        matrix_slice_spec(&weights.mm2_weight, 0, cfg.projection_dim),
    )?;
    let mm2_b = push_spec(
        &mut specs,
        f32_vector_spec(&weights.mm2_bias, 0, cfg.projection_dim),
    )?;
    let weight_bytes = specs.iter().try_fold(0u64, |sum, spec| {
        sum.checked_add(spec.nbytes() as u64)
            .context("vision projector byte count overflow")
    })?;
    Ok((
        specs,
        SpecLayout {
            patch_embd_w,
            patch_embd_w_temporal,
            patch_embd_b,
            blocks,
            post_ln_w,
            post_ln_b,
            mm0_w,
            mm0_b,
            mm2_w,
            mm2_b,
        },
        weight_bytes,
    ))
}

fn load_weight_buffers(
    gguf: &Gguf,
    specs: &[WeightSpec],
    backend: &VulkanBackend,
) -> Result<Vec<Box<dyn Buffer>>> {
    let sizes = specs.iter().map(WeightSpec::nbytes).collect::<Vec<_>>();
    let buffers = backend
        .alloc_uninit_batch(&sizes, BufferUsage::Weights)
        .map_err(|error| anyhow!("admit vision weights to unified VRAM: {error}"))?;
    for (spec, buffer) in specs.iter().zip(&buffers) {
        let source = gguf
            .tensor_bytes(&spec.source)
            .map_err(|error| anyhow!("read vision tensor {}: {error}", spec.source))?;
        match spec.payload {
            WeightPayload::Raw {
                source_offset,
                nbytes,
            } => {
                let end = source_offset
                    .checked_add(nbytes)
                    .context("vision matrix source range overflow")?;
                let bytes = source.get(source_offset..end).with_context(|| {
                    format!(
                        "vision weight {} source range {source_offset}..{end} exceeds {} bytes",
                        spec.label,
                        source.len()
                    )
                })?;
                backend
                    .upload(buffer.as_ref(), bytes)
                    .map_err(|error| anyhow!("upload vision weight {}: {error}", spec.label))?;
            }
            WeightPayload::F32Slice {
                element_offset,
                elements,
            } => {
                let values = infr_gguf::dequant::dequant_block(
                    gguf.tensors()
                        .iter()
                        .find(|tensor| tensor.name == spec.source)
                        .with_context(|| format!("missing vision tensor {}", spec.source))?
                        .dtype,
                    source,
                )
                .with_context(|| format!("dequantize vision vector {}", spec.source))?;
                let end = element_offset
                    .checked_add(elements)
                    .context("vision vector source range overflow")?;
                let values = values.get(element_offset..end).with_context(|| {
                    format!(
                        "vision vector {} source range {element_offset}..{end} exceeds {} elements",
                        spec.label,
                        values.len()
                    )
                })?;
                backend
                    .upload(buffer.as_ref(), bytemuck::cast_slice(values))
                    .map_err(|error| anyhow!("upload vision vector {}: {error}", spec.label))?;
            }
        }
    }
    Ok(buffers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
    use std::io::Cursor;

    #[test]
    fn q8_qkv_slices_are_disjoint_and_cover_the_tensor() {
        let in_features = 1152usize;
        let out_features = 3 * in_features;
        let nbytes = infr_gguf::nbytes(DType::Q8_0, in_features * out_features);
        let info = TensorInfo {
            name: "v.blk.0.attn_qkv.weight".to_owned(),
            shape: vec![in_features, out_features],
            dtype: DType::Q8_0,
            offset: 0,
            nbytes,
        };
        let specs = [
            matrix_slice_spec(&info, 0, in_features).unwrap(),
            matrix_slice_spec(&info, in_features, 2 * in_features).unwrap(),
            matrix_slice_spec(&info, 2 * in_features, 3 * in_features).unwrap(),
        ];
        let ranges = specs
            .iter()
            .map(|spec| match spec.payload {
                WeightPayload::Raw {
                    source_offset,
                    nbytes,
                } => source_offset..source_offset + nbytes,
                WeightPayload::F32Slice { .. } => unreachable!(),
            })
            .collect::<Vec<_>>();
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges[0].end, ranges[1].start);
        assert_eq!(ranges[1].end, ranges[2].start);
        assert_eq!(ranges[2].end, info.nbytes);
        assert_eq!(
            specs.iter().map(WeightSpec::nbytes).sum::<usize>(),
            info.nbytes
        );
    }

    #[test]
    #[ignore = "requires MMPROJ_TEST_PATH to name a local projector GGUF"]
    fn local_projector_catalog_matches_gguf_storage() {
        let path = std::env::var_os("MMPROJ_TEST_PATH")
            .expect("set MMPROJ_TEST_PATH to a local mmproj GGUF");
        let gguf = Gguf::open(Path::new(&path)).unwrap();
        let cfg = ClipConfig::from_gguf(&gguf).unwrap();
        let weights = VisionWeights::load(&gguf).unwrap();
        let (specs, layout, bytes) = build_weight_catalog(&weights, &cfg).unwrap();
        assert_eq!(layout.blocks.len(), cfg.block_count);
        assert!(
            layout.patch_embd_w_temporal.is_some(),
            "Qwen3-VL projector must retain both temporal Conv3D slices"
        );
        assert_eq!(bytes, specs.iter().map(|spec| spec.nbytes() as u64).sum());
        assert!(specs.iter().any(|spec| spec.desc.dtype == DType::Q8_0));
        for spec in specs {
            let source = gguf.tensor_bytes(&spec.source).unwrap();
            match spec.payload {
                WeightPayload::Raw {
                    source_offset,
                    nbytes,
                } => assert!(source_offset + nbytes <= source.len(), "{}", spec.label),
                WeightPayload::F32Slice {
                    element_offset,
                    elements,
                } => {
                    let dtype = gguf
                        .tensors()
                        .iter()
                        .find(|tensor| tensor.name == spec.source)
                        .unwrap()
                        .dtype;
                    let values = infr_gguf::dequant::dequant_block(dtype, source).unwrap();
                    assert!(element_offset + elements <= values.len(), "{}", spec.label);
                }
            }
        }
    }

    #[test]
    #[ignore = "requires MMPROJ_TEST_PATH and a Vulkan device"]
    fn local_projector_matches_streaming_cpu_reference() {
        let path = std::env::var_os("MMPROJ_TEST_PATH")
            .expect("set MMPROJ_TEST_PATH to a local mmproj GGUF");
        let path = Path::new(&path);
        let gguf = Gguf::open(path).unwrap();
        let cfg = ClipConfig::from_gguf(&gguf).unwrap();
        let weights = VisionWeights::load(&gguf).unwrap();
        let pos_raw = gguf
            .tensor_bytes(&weights.position_embd_weight.name)
            .unwrap();
        let pos_table =
            infr_gguf::dequant::dequant_block(weights.position_embd_weight.dtype, pos_raw).unwrap();

        // Two merged tokens exercise spatial ordering as well as every projector block. Larger
        // grids can be selected explicitly for local investigations without making CI run this
        // deliberately heavy ignored test.
        let width = std::env::var("VISION_PARITY_WIDTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(32);
        let height = std::env::var("VISION_PARITY_HEIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(64);
        let image = RgbImage::from_fn(width, height, |x, y| {
            Rgb([
                ((x * 7 + y * 3) & 0xff) as u8,
                ((x * 5 + 41) & 0xff) as u8,
                ((y * 9 + 17) & 0xff) as u8,
            ])
        });
        let mut png = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image)
            .write_to(&mut png, ImageFormat::Png)
            .unwrap();
        let png = png.into_inner();
        let prepared = prepare_image_bytes(&png, &cfg, &pos_table).unwrap();

        let started = std::time::Instant::now();
        let expected = crate::reference::encode(path, &prepared).unwrap();
        eprintln!("streaming CPU vision reference: {:?}", started.elapsed());

        let backend = VulkanBackend::new().unwrap();
        let engine = NativeVisionEngine::load_vulkan_with_backend(path, backend).unwrap();
        let resident = load_weight_buffers(&gguf, &engine.specs, &engine.backend).unwrap();
        let mut plan = engine
            .build_plan(prepared.n_patches(), prepared.grid_nx * prepared.grid_ny)
            .unwrap();
        let actual = engine
            .execute_plan(&mut plan, &prepared, &resident)
            .unwrap();
        assert_eq!(actual.len(), expected.len());

        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let mut dot = 0.0f64;
        let mut expected_sq = 0.0f64;
        let mut actual_sq = 0.0f64;
        for (&want, &got) in expected.iter().zip(&actual) {
            let error = (want - got).abs();
            max_abs = max_abs.max(error);
            sum_abs += error as f64;
            dot += want as f64 * got as f64;
            expected_sq += (want as f64).powi(2);
            actual_sq += (got as f64).powi(2);
        }
        let mean_abs = sum_abs / expected.len() as f64;
        let cosine = dot / (expected_sq.sqrt() * actual_sq.sqrt());
        eprintln!(
            "CPU/Vulkan vision parity: max_abs={max_abs:.6} mean_abs={mean_abs:.6} cosine={cosine:.9}"
        );
        if let Some(directory) = std::env::var_os("VISION_PARITY_DUMP") {
            let directory = Path::new(&directory);
            std::fs::create_dir_all(directory).unwrap();
            std::fs::write(
                directory.join("rust_cpu.f32"),
                bytemuck::cast_slice(&expected),
            )
            .unwrap();
            std::fs::write(
                directory.join("rust_vulkan.f32"),
                bytemuck::cast_slice(&actual),
            )
            .unwrap();
            std::fs::write(
                directory.join("rust_patches.f32"),
                bytemuck::cast_slice(&prepared.patches),
            )
            .unwrap();
            std::fs::write(
                directory.join("rust_pos_embed.f32"),
                bytemuck::cast_slice(&prepared.pos_embed),
            )
            .unwrap();
            std::fs::write(directory.join("input.png"), &png).unwrap();
        }
        assert!(
            max_abs < 0.1 && cosine > 0.9999,
            "CPU/Vulkan vision parity failed: max_abs={max_abs}, mean_abs={mean_abs}, cosine={cosine}"
        );
    }
}
