//! Native INFR execution for encoder-only Nomic-BERT embedding models.

use crate::{tokenizer::BertWordPiece, EmbeddingBatch, EmbeddingConfig, EmbeddingEngine};
use anyhow::{anyhow, bail, Context, Result};
use infr_core::{
    backend::{Backend, Bindings, Buffer, BufferUsage, Plan},
    graph::{Activation, AttnMask, Graph, Op},
    loader::{MetaValue, TensorInfo},
    tensor::{DType, TensorDesc, TensorId},
    MemoryTier, ResourceKind, ResourceSnapshot, ResourceTracker, WeightSource,
};
use infr_gguf::Gguf;
use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug)]
struct NomicConfig {
    public: EmbeddingConfig,
    layers: usize,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    ffn: usize,
    vocab: usize,
    eps: f32,
    rope_theta: f32,
}

impl NomicConfig {
    fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let public = EmbeddingConfig::from_gguf(gguf)?;
        if public.architecture != "nomic-bert" {
            bail!(
                "native embedding currently supports general.architecture=\"nomic-bert\"; got {:?}",
                public.architecture
            );
        }
        let md = gguf.metadata();
        let integer = |suffix: &str| -> Result<usize> {
            let key = format!("{}.{suffix}", public.architecture);
            usize::try_from(
                md.u64(&key)
                    .with_context(|| format!("GGUF missing {key}"))?,
            )
            .with_context(|| format!("GGUF {key} is too large"))
        };
        if matches!(
            md.get("nomic-bert.attention.causal"),
            Some(MetaValue::Bool(true))
        ) {
            bail!("native Nomic-BERT requires bidirectional attention");
        }
        let pooling = md.u64("nomic-bert.pooling_type").unwrap_or(1);
        if pooling != 1 {
            bail!("native Nomic-BERT requires mean pooling (pooling_type=1, got {pooling})");
        }
        let hidden = public.dimensions;
        let heads = integer("attention.head_count")?;
        if heads == 0 || hidden % heads != 0 {
            bail!("invalid Nomic-BERT head geometry: hidden={hidden}, heads={heads}");
        }
        let vocab = gguf
            .metadata()
            .get("tokenizer.ggml.tokens")
            .and_then(MetaValue::as_arr)
            .context("GGUF missing tokenizer.ggml.tokens")?
            .len();
        Ok(Self {
            layers: integer("block_count")?,
            head_dim: hidden / heads,
            ffn: integer("feed_forward_length")?,
            eps: md
                .get("nomic-bert.attention.layer_norm_epsilon")
                .and_then(MetaValue::as_f64)
                .unwrap_or(1e-12) as f32,
            rope_theta: md
                .get("nomic-bert.rope.freq_base")
                .and_then(MetaValue::as_f64)
                .unwrap_or(1000.0) as f32,
            public,
            hidden,
            heads,
            vocab,
        })
    }
}

#[derive(Clone)]
struct WeightSpec {
    label: String,
    desc: TensorDesc,
}

#[derive(Clone, Copy)]
struct LayerWeights {
    qkv: usize,
    attn_output: usize,
    attn_norm_weight: usize,
    attn_norm_bias: usize,
    ffn_gate: usize,
    ffn_up: usize,
    ffn_down: usize,
    output_norm_weight: usize,
    output_norm_bias: usize,
}

struct WeightLayout {
    token_embedding: usize,
    token_type_zero: usize,
    embedding_norm_weight: usize,
    embedding_norm_bias: usize,
    layers: Vec<LayerWeights>,
}

struct NativePlan {
    plan: Box<dyn Plan>,
    ids: TensorId,
    positions: TensorId,
    output: TensorId,
    weight_ids: Vec<TensorId>,
    ids_buffer: Box<dyn Buffer>,
    positions_buffer: Box<dyn Buffer>,
    output_buffer: Box<dyn Buffer>,
}

/// Nomic-BERT embedding inference executed directly by INFR's CPU or Vulkan graph backend.
pub struct NativeEmbeddingEngine {
    cfg: NomicConfig,
    tokenizer: BertWordPiece,
    backend: Box<dyn Backend>,
    specs: Vec<WeightSpec>,
    weights: Vec<Box<dyn Buffer>>,
    layout: WeightLayout,
    plans: Mutex<HashMap<usize, NativePlan>>,
    resource: Arc<ResourceTracker>,
}

impl NativeEmbeddingEngine {
    pub fn load_cpu(path: &Path, engine_cfg: Arc<infr_core::config::Config>) -> Result<Self> {
        Self::load(
            path,
            Box::new(infr_cpu::CpuBackend::new_with(engine_cfg)),
            MemoryTier::Ram,
        )
    }

    pub fn load_vulkan(path: &Path, engine_cfg: Arc<infr_core::config::Config>) -> Result<Self> {
        let backend = infr_vulkan::VulkanBackend::new_with(engine_cfg)
            .map_err(|error| anyhow!("initialize Vulkan embedding backend: {error}"))?;
        Self::load(path, Box::new(backend), MemoryTier::Vram)
    }

    /// Load on a Vulkan client derived from an already-warm LLM backend. The client shares the
    /// device and unified VRAM arena while keeping a separate Embedding execution graph.
    pub fn load_vulkan_with_backend(
        path: &Path,
        backend: infr_vulkan::VulkanBackend,
    ) -> Result<Self> {
        Self::load(path, Box::new(backend), MemoryTier::Vram)
    }

    fn load(path: &Path, backend: Box<dyn Backend>, tier: MemoryTier) -> Result<Self> {
        if !path.is_file() {
            bail!("embedding model does not exist: {}", path.display());
        }
        let gguf = Gguf::open(path).map_err(|error| anyhow!(error.to_string()))?;
        let cfg = NomicConfig::from_gguf(&gguf)?;
        let tokenizer = BertWordPiece::from_metadata(gguf.metadata(), cfg.public.max_context)?;
        let (specs, weights, layout, resident_bytes) = load_weights(&gguf, &cfg, backend.as_ref())?;
        let model_id = path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or(&cfg.public.name)
            .to_owned();
        tracing::info!(
            model = %model_id,
            architecture = %cfg.public.architecture,
            backend = backend.name(),
            tier = ?tier,
            weights_mib = resident_bytes as f64 / 1048576.0,
            "native embedding model ready"
        );
        Ok(Self {
            cfg,
            tokenizer,
            backend,
            specs,
            weights,
            layout,
            plans: Mutex::new(HashMap::new()),
            resource: Arc::new(ResourceTracker::new(
                format!("embedding:{model_id}"),
                ResourceKind::EmbeddingWeights,
                resident_bytes,
                resident_bytes,
                tier,
                resident_bytes,
            )),
        })
    }

    pub fn config(&self) -> &EmbeddingConfig {
        &self.cfg.public
    }

    pub fn resource_snapshot(&self) -> ResourceSnapshot {
        self.resource.snapshot()
    }

    pub fn embed(&self, inputs: &[String]) -> Result<EmbeddingBatch> {
        if inputs.is_empty() {
            bail!("input array must contain at least one string");
        }
        let encoded = inputs
            .iter()
            .map(|input| self.tokenizer.encode(input))
            .collect::<Result<Vec<_>>>()?;
        let prompt_tokens = encoded.iter().try_fold(0u32, |sum, ids| {
            sum.checked_add(ids.len() as u32)
                .context("embedding prompt-token count overflow")
        })?;
        let _lease = self.resource.acquire();
        let mut plans = self.plans.lock().unwrap_or_else(|error| error.into_inner());
        let mut embeddings = Vec::with_capacity(encoded.len());
        for ids in encoded {
            if !plans.contains_key(&ids.len()) {
                let plan = self.build_plan(ids.len())?;
                plans.insert(ids.len(), plan);
            }
            let plan = plans.get_mut(&ids.len()).expect("plan inserted above");
            embeddings.push(self.execute_plan(plan, &ids)?);
        }
        Ok(EmbeddingBatch {
            embeddings,
            prompt_tokens,
        })
    }

    fn build_plan(&self, rows: usize) -> Result<NativePlan> {
        let mut graph = Graph::new();
        let ids = graph.input(TensorDesc::new(vec![rows], DType::I32));
        let ids = graph.label(ids, "embedding.token_ids");
        let positions = graph.input(TensorDesc::new(vec![rows], DType::I32));
        let positions = graph.label(positions, "embedding.positions");
        let weight_ids = self
            .specs
            .iter()
            .map(|spec| {
                let id = graph.weight(spec.desc.clone());
                graph.label(id, spec.label.clone())
            })
            .collect::<Vec<_>>();
        let wid = |index: usize| weight_ids[index];
        let f32d = |n: usize| TensorDesc::new(vec![n], DType::F32);
        let f16d = |n: usize| TensorDesc::new(vec![n], DType::F16);
        let hidden_len = rows * self.cfg.hidden;
        let ffn_len = rows * self.cfg.ffn;
        let state = [
            graph.internal(f32d(hidden_len)),
            graph.internal(f32d(hidden_len)),
        ];
        let normed = graph.internal(f32d(hidden_len));
        let q = graph.internal(f32d(hidden_len));
        let k = graph.internal(f32d(hidden_len));
        let v = graph.internal(f32d(hidden_len));
        let q_roped = graph.internal(f32d(hidden_len));
        let k_roped = graph.internal(f32d(hidden_len));
        let q16 = graph.internal(f16d(hidden_len));
        let k16 = graph.internal(f16d(hidden_len));
        let v16 = graph.internal(f16d(hidden_len));
        let attention = graph.internal(f32d(hidden_len));
        let sublayer = graph.internal(f32d(hidden_len));
        let gate = graph.internal(f32d(ffn_len));
        let up = graph.internal(f32d(ffn_len));
        let activated = graph.internal(f32d(ffn_len));
        let output = graph.output(f32d(hidden_len));
        let output = graph.label(output, "embedding.last_hidden_state");

        graph.push(Op::EmbedGather {
            ids,
            table: wid(self.layout.token_embedding),
            dst: state[0],
            rows: rows as u32,
            ne: self.cfg.hidden as u32,
            scale: 1.0,
        });
        graph.push(Op::AddBias {
            x: state[0],
            bias: wid(self.layout.token_type_zero),
            dst: state[0],
            rows: rows as u32,
            n: self.cfg.hidden as u32,
        });
        graph.push(Op::LayerNorm {
            x: state[0],
            weight: wid(self.layout.embedding_norm_weight),
            bias: wid(self.layout.embedding_norm_bias),
            dst: state[1],
            rows: rows as u32,
            dim: self.cfg.hidden as u32,
            eps: self.cfg.eps,
        });

        let current = state[1];
        let residual = state[0];
        let matrix = self.cfg.hidden * self.cfg.hidden;
        for layer in &self.layout.layers {
            for (dst, offset) in [(q, 0usize), (k, matrix), (v, 2 * matrix)] {
                graph.push(Op::Linear {
                    x: current,
                    weight: wid(layer.qkv),
                    dst,
                    m: rows as u32,
                    in_f: self.cfg.hidden as u32,
                    out_f: self.cfg.hidden as u32,
                    w_off: offset as u32,
                });
            }
            for (src, dst) in [(q, q_roped), (k, k_roped)] {
                graph.push(Op::Rope {
                    x: src,
                    positions,
                    dst,
                    rows: rows as u32,
                    n_head: self.cfg.heads as u32,
                    head_dim: self.cfg.head_dim as u32,
                    rope_dim: self.cfg.head_dim as u32,
                    theta: self.cfg.rope_theta,
                    freq_factors: None,
                    x_stride: 0,
                    neox: true,
                    backward: false,
                });
            }
            for (src, dst) in [(q_roped, q16), (k_roped, k16), (v, v16)] {
                graph.push(Op::Copy {
                    src,
                    src_off: 0,
                    dst,
                    dst_off: 0,
                    n: hidden_len as u32,
                });
            }
            graph.push(Op::Attention {
                q: q16,
                k_cache: k16,
                v_cache: v16,
                dst: attention,
                rows: rows as u32,
                kv_len: rows as u32,
                n_head: self.cfg.heads as u32,
                n_kv: self.cfg.heads as u32,
                head_dim: self.cfg.head_dim as u32,
                scale: 1.0 / (self.cfg.head_dim as f32).sqrt(),
                mask: AttnMask::Canvas { lo: 0 },
                pos: 0,
                sinks: None,
            });
            graph.push(Op::Linear {
                x: attention,
                weight: wid(layer.attn_output),
                dst: sublayer,
                m: rows as u32,
                in_f: self.cfg.hidden as u32,
                out_f: self.cfg.hidden as u32,
                w_off: 0,
            });
            graph.push(Op::Add {
                a: current,
                b: sublayer,
                dst: residual,
                n: hidden_len as u32,
            });
            graph.push(Op::LayerNorm {
                x: residual,
                weight: wid(layer.attn_norm_weight),
                bias: wid(layer.attn_norm_bias),
                dst: normed,
                rows: rows as u32,
                dim: self.cfg.hidden as u32,
                eps: self.cfg.eps,
            });
            graph.push(Op::Linear {
                x: normed,
                weight: wid(layer.ffn_gate),
                dst: gate,
                m: rows as u32,
                in_f: self.cfg.hidden as u32,
                out_f: self.cfg.ffn as u32,
                w_off: 0,
            });
            graph.push(Op::Linear {
                x: normed,
                weight: wid(layer.ffn_up),
                dst: up,
                m: rows as u32,
                in_f: self.cfg.hidden as u32,
                out_f: self.cfg.ffn as u32,
                w_off: 0,
            });
            graph.push(Op::GatedAct {
                gate,
                up,
                dst: activated,
                rows: rows as u32,
                nff: self.cfg.ffn as u32,
                act: Activation::Silu,
                up_off: 0,
                up_stride: 0,
                gate_stride: 0,
                gate_block_width: 0,
                swiglu_clamp: None,
            });
            graph.push(Op::Linear {
                x: activated,
                weight: wid(layer.ffn_down),
                dst: sublayer,
                m: rows as u32,
                in_f: self.cfg.ffn as u32,
                out_f: self.cfg.hidden as u32,
                w_off: 0,
            });
            graph.push(Op::Add {
                a: normed,
                b: sublayer,
                dst: residual,
                n: hidden_len as u32,
            });
            graph.push(Op::LayerNorm {
                x: residual,
                weight: wid(layer.output_norm_weight),
                bias: wid(layer.output_norm_bias),
                dst: current,
                rows: rows as u32,
                dim: self.cfg.hidden as u32,
                eps: self.cfg.eps,
            });
        }
        graph.push(Op::Copy {
            src: current,
            src_off: 0,
            dst: output,
            dst_off: 0,
            n: hidden_len as u32,
        });

        let plan = self
            .backend
            .compile(&graph)
            .map_err(|error| anyhow!("compile native embedding graph: {error}"))?;
        let ids_buffer = self.alloc(rows * 4, BufferUsage::Staging)?;
        let positions_buffer = self.alloc(rows * 4, BufferUsage::Staging)?;
        let output_buffer = self.alloc(hidden_len * 4, BufferUsage::Readback)?;
        Ok(NativePlan {
            plan,
            ids,
            positions,
            output,
            weight_ids,
            ids_buffer,
            positions_buffer,
            output_buffer,
        })
    }

    fn execute_plan(&self, plan: &mut NativePlan, ids: &[u32]) -> Result<Vec<f32>> {
        self.backend
            .upload(plan.ids_buffer.as_ref(), bytemuck::cast_slice(ids))
            .map_err(|error| anyhow!("upload embedding token ids: {error}"))?;
        let positions = (0..ids.len() as u32).collect::<Vec<_>>();
        self.backend
            .upload(
                plan.positions_buffer.as_ref(),
                bytemuck::cast_slice(&positions),
            )
            .map_err(|error| anyhow!("upload embedding positions: {error}"))?;
        let mut bindings = Bindings::new();
        bindings
            .bind(plan.ids, plan.ids_buffer.as_ref())
            .bind(plan.positions, plan.positions_buffer.as_ref())
            .bind(plan.output, plan.output_buffer.as_ref());
        for ((id, buffer), spec) in plan.weight_ids.iter().zip(&self.weights).zip(&self.specs) {
            debug_assert_eq!(
                buffer.len_bytes(),
                spec.desc
                    .dtype
                    .dense_bytes(spec.desc.numel())
                    .unwrap_or(buffer.len_bytes())
            );
            bindings.bind(*id, buffer.as_ref());
        }
        self.backend
            .execute(plan.plan.as_ref(), &bindings)
            .map_err(|error| anyhow!("execute native embedding graph: {error}"))?;
        let mut bytes = vec![0u8; ids.len() * self.cfg.hidden * 4];
        self.backend
            .download(plan.output_buffer.as_ref(), &mut bytes)
            .map_err(|error| anyhow!("download native embedding output: {error}"))?;
        let hidden = bytemuck::cast_slice::<u8, f32>(&bytes);
        let mut pooled = vec![0.0f32; self.cfg.hidden];
        for row in hidden.chunks_exact(self.cfg.hidden) {
            for (dst, value) in pooled.iter_mut().zip(row) {
                *dst += *value;
            }
        }
        let inv_rows = 1.0 / ids.len() as f32;
        for value in &mut pooled {
            *value *= inv_rows;
        }
        let norm = pooled.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut pooled {
                *value /= norm;
            }
        }
        Ok(pooled)
    }

    fn alloc(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        self.backend
            .alloc_uninit(bytes, usage)
            .map_err(|error| anyhow!("allocate native embedding buffer: {error}"))
    }
}

impl EmbeddingEngine for NativeEmbeddingEngine {
    fn config(&self) -> &EmbeddingConfig {
        NativeEmbeddingEngine::config(self)
    }

    fn resource_snapshot(&self) -> ResourceSnapshot {
        NativeEmbeddingEngine::resource_snapshot(self)
    }

    fn embed(&self, inputs: &[String]) -> Result<EmbeddingBatch> {
        NativeEmbeddingEngine::embed(self, inputs)
    }
}

struct WeightLoader<'a> {
    gguf: &'a Gguf,
    backend: &'a dyn Backend,
    specs: Vec<WeightSpec>,
    buffers: Vec<Box<dyn Buffer>>,
    resident_bytes: u64,
}

impl<'a> WeightLoader<'a> {
    fn new(gguf: &'a Gguf, backend: &'a dyn Backend) -> Self {
        Self {
            gguf,
            backend,
            specs: Vec::new(),
            buffers: Vec::new(),
            resident_bytes: 0,
        }
    }

    fn tensor(&self, name: &str, shape: &[usize]) -> Result<&TensorInfo> {
        let info = self
            .gguf
            .tensors()
            .iter()
            .find(|tensor| tensor.name == name)
            .with_context(|| format!("GGUF missing tensor {name}"))?;
        if info.shape != shape {
            bail!(
                "GGUF tensor {name} has shape {:?}; expected {shape:?}",
                info.shape
            );
        }
        Ok(info)
    }

    fn push(&mut self, name: &str, shape: &[usize]) -> Result<usize> {
        let info = self.tensor(name, shape)?;
        let desc = TensorDesc::new(info.shape.clone(), info.dtype);
        let nbytes = info.nbytes;
        let bytes = self
            .gguf
            .tensor_bytes(name)
            .map_err(|error| anyhow!("read GGUF tensor {name}: {error}"))?;
        let buffer = self
            .backend
            .alloc_uninit(nbytes, BufferUsage::Weights)
            .map_err(|error| anyhow!("allocate embedding weight {name}: {error}"))?;
        self.backend
            .upload(buffer.as_ref(), bytes)
            .map_err(|error| anyhow!("upload embedding weight {name}: {error}"))?;
        let index = self.specs.len();
        self.specs.push(WeightSpec {
            label: name.to_owned(),
            desc,
        });
        self.buffers.push(buffer);
        self.resident_bytes += nbytes as u64;
        Ok(index)
    }
}

fn load_weights(
    gguf: &Gguf,
    cfg: &NomicConfig,
    backend: &dyn Backend,
) -> Result<(Vec<WeightSpec>, Vec<Box<dyn Buffer>>, WeightLayout, u64)> {
    let mut loader = WeightLoader::new(gguf, backend);
    let token_embedding = loader.push("token_embd.weight", &[cfg.hidden, cfg.vocab])?;
    let token_type_dtype = loader.tensor("token_types.weight", &[cfg.hidden, 2])?.dtype;
    if token_type_dtype != DType::F32 {
        bail!(
            "GGUF token_types.weight must be F32, got {:?}",
            token_type_dtype
        );
    }
    let token_type_bytes = gguf
        .tensor_bytes("token_types.weight")
        .map_err(|error| anyhow!("read GGUF tensor token_types.weight: {error}"))?;
    let row_bytes = cfg.hidden * 4;
    let token_type_buffer = loader
        .backend
        .alloc_uninit(row_bytes, BufferUsage::Weights)
        .map_err(|error| anyhow!("allocate token type row: {error}"))?;
    loader
        .backend
        .upload(token_type_buffer.as_ref(), &token_type_bytes[..row_bytes])
        .map_err(|error| anyhow!("upload token type row: {error}"))?;
    let token_type_zero = loader.specs.len();
    loader.specs.push(WeightSpec {
        label: "token_types.weight[row=0]".into(),
        desc: TensorDesc::new(vec![cfg.hidden], DType::F32),
    });
    loader.buffers.push(token_type_buffer);
    loader.resident_bytes += row_bytes as u64;
    let embedding_norm_weight = loader.push("token_embd_norm.weight", &[cfg.hidden])?;
    let embedding_norm_bias = loader.push("token_embd_norm.bias", &[cfg.hidden])?;

    let mut layers = Vec::with_capacity(cfg.layers);
    for layer in 0..cfg.layers {
        let name = |suffix: &str| format!("blk.{layer}.{suffix}");
        layers.push(LayerWeights {
            qkv: loader.push(&name("attn_qkv.weight"), &[cfg.hidden, 3 * cfg.hidden])?,
            attn_output: loader.push(&name("attn_output.weight"), &[cfg.hidden, cfg.hidden])?,
            attn_norm_weight: loader.push(&name("attn_output_norm.weight"), &[cfg.hidden])?,
            attn_norm_bias: loader.push(&name("attn_output_norm.bias"), &[cfg.hidden])?,
            ffn_gate: loader.push(&name("ffn_gate.weight"), &[cfg.hidden, cfg.ffn])?,
            ffn_up: loader.push(&name("ffn_up.weight"), &[cfg.hidden, cfg.ffn])?,
            ffn_down: loader.push(&name("ffn_down.weight"), &[cfg.ffn, cfg.hidden])?,
            output_norm_weight: loader.push(&name("layer_output_norm.weight"), &[cfg.hidden])?,
            output_norm_bias: loader.push(&name("layer_output_norm.bias"), &[cfg.hidden])?,
        });
    }
    let WeightLoader {
        specs,
        buffers,
        resident_bytes,
        ..
    } = loader;
    Ok((
        specs,
        buffers,
        WeightLayout {
            token_embedding,
            token_type_zero,
            embedding_norm_weight,
            embedding_norm_bias,
            layers,
        },
        resident_bytes,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Oracle {
        cases: Vec<OracleCase>,
    }

    #[derive(Deserialize)]
    struct OracleCase {
        name: String,
        input: Vec<String>,
        prompt_tokens: u32,
        dimensions: usize,
        embeddings: Vec<f32>,
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (x as f64, y as f64);
            dot += x * y;
            aa += x * x;
            bb += y * y;
        }
        dot / (aa.sqrt() * bb.sqrt())
    }

    fn run_oracle(engine: NativeEmbeddingEngine, oracle_path: &Path, all_cases: bool) {
        let bytes = std::fs::read(oracle_path).unwrap();
        let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
        let oracle: Oracle = serde_json::from_slice(bytes).unwrap();
        let cases = if all_cases {
            oracle.cases.as_slice()
        } else {
            &oracle.cases[..1]
        };
        for case in cases {
            let got = engine.embed(&case.input).unwrap();
            assert_eq!(
                got.prompt_tokens, case.prompt_tokens,
                "{} tokens",
                case.name
            );
            assert_eq!(got.embeddings.len(), case.input.len(), "{} rows", case.name);
            assert_eq!(
                case.embeddings.len(),
                case.input.len() * case.dimensions,
                "{} oracle shape",
                case.name
            );
            for (row, expected) in got
                .embeddings
                .iter()
                .zip(case.embeddings.chunks_exact(case.dimensions))
            {
                let cos = cosine(row, expected);
                let max_abs = row
                    .iter()
                    .zip(expected)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "native embedding {}: cosine={cos:.8}, max_abs={max_abs:.6}",
                    case.name
                );
                assert!(
                    cos > 0.999,
                    "{} cosine {cos} is below parity floor",
                    case.name
                );
                assert!(
                    max_abs < 0.005,
                    "{} max absolute error {max_abs} exceeds parity floor",
                    case.name
                );
            }
        }
    }

    #[test]
    fn real_nomic_cpu_matches_llama_cpp_when_available() {
        let (Ok(model), Ok(oracle)) = (
            std::env::var("INFR_EMBEDDING_TEST_MODEL"),
            std::env::var("INFR_EMBEDDING_TEST_ORACLE"),
        ) else {
            return;
        };
        let engine = NativeEmbeddingEngine::load_cpu(
            Path::new(&model),
            Arc::new(infr_core::config::Config::default()),
        )
        .unwrap();
        run_oracle(engine, Path::new(&oracle), false);
    }

    #[test]
    fn real_nomic_vulkan_matches_llama_cpp_when_requested() {
        let (Ok(model), Ok(oracle), Ok(_)) = (
            std::env::var("INFR_EMBEDDING_TEST_MODEL"),
            std::env::var("INFR_EMBEDDING_TEST_ORACLE"),
            std::env::var("INFR_EMBEDDING_TEST_VULKAN"),
        ) else {
            return;
        };
        let engine = NativeEmbeddingEngine::load_vulkan(
            Path::new(&model),
            Arc::new(infr_core::config::Config::default()),
        )
        .unwrap();
        run_oracle(engine, Path::new(&oracle), true);
    }
}
