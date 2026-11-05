//! Native, offline SenseVoiceSmall SANM encoder + greedy CTC.
//!
//! Numeric contract: https://github.com/FunAudioLLM/SenseVoice/blob/main/model.py
//! (Apache-2.0), not the historical sense-voice.cpp epsilon/position formula.
//! GGUF dimensions are fastest-first; Linear descriptors are row-major [out,in].
//! No matmul, attention or convolution is implemented outside Backend/Graph.
//!
//! A request executes one layer graph at a time. Each graph uses the same fixed
//! set of scratch handles, then is dropped before the next layer is compiled.
//! This also bounds the CPU interpreter, which retains all vals until execute
//! returns. No request/shape/trace cache is retained by the model.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::atomic::AtomicBool,
};

use anyhow::{bail, ensure, Context, Result};
use infr_core::{
    AttnMask, Backend, Bindings, Buffer, BufferUsage, DType, Graph, Op, TensorDesc, TensorId,
    TensorInfo, WeightSource,
};
use infr_gguf::{dequant::dequant_block, Gguf};

use crate::frontend::{Features, FrontendConfig};

#[cfg(test)]
#[path = "model_diagnostics.rs"]
mod diagnostics;

#[path = "sensevoice_gguf.rs"]
mod gguf_compat;
pub use gguf_compat::SenseVoiceConfig;

/// Service chunks are at most 28 seconds (~467 LFR rows). Direct model callers
/// must segment too: attention is noncausal within a chunk, not across chunks.
pub const MAX_FEATURE_ROWS: usize = 512;
const CTC_CHUNK_ROWS: usize = 32;

/// Invoked synchronously with row-major f32 tensors. Data is borrowed only for
/// the call; retaining it is the observer's responsibility. Errors abort infer.
/// Names include input.prefix, input.position, encoder.input,
/// encoder.encoders0.0.{norm1,q,k,v,attention,fsmn,attention_residual,norm2,relu,output},
/// encoder.encoders.N.*, encoder.after_norm, encoder.tp_encoders.N.*,
/// encoder.tp_norm and ctc.logits.START (including all four prefix rows).
type TraceCallback<'a> = dyn FnMut(&str, usize, usize, &[f32]) -> Result<()> + 'a;

struct Weight {
    desc: TensorDesc,
    buffer: Box<dyn Buffer>,
}

pub struct SenseVoiceModel {
    backend: Box<dyn Backend>,
    config: SenseVoiceConfig,
    frontend: FrontendConfig,
    weights: HashMap<String, Weight>,
    embedding: Vec<f32>,
    tokens: Vec<String>,
    weight_bytes: u64,
}

impl SenseVoiceModel {
    pub fn load(path: &Path, backend: Box<dyn Backend>) -> Result<Self> {
        Self::load_cancellable(path, backend, &AtomicBool::new(false))
    }

    /// Poll around GGUF parsing and between weight uploads; an individual backend
    /// allocation/upload or file read must finish before cancellation is observed.
    pub fn load_cancellable(
        path: &Path,
        backend: Box<dyn Backend>,
        cancel: &AtomicBool,
    ) -> Result<Self> {
        crate::check_cancel(cancel)?;
        let gguf =
            Gguf::open(path).with_context(|| format!("open SenseVoice GGUF {}", path.display()))?;
        crate::check_cancel(cancel)?;
        let parameters = gguf_compat::parse(gguf.metadata())?;
        crate::check_cancel(cancel)?;
        let specs = weight_specs(&parameters.config);
        let mut index = HashMap::new();
        for info in gguf.tensors() {
            crate::check_cancel(cancel)?;
            ensure!(
                index.insert(info.name.as_str(), info).is_none(),
                "duplicate tensor {}",
                info.name
            );
        }
        // Validate the complete tensor inventory before allocating or uploading.
        let expected: HashSet<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        for info in gguf.tensors() {
            crate::check_cancel(cancel)?;
            if info.name.starts_with("encoder.")
                || info.name.starts_with("ctc.")
                || info.name == "embed.weight"
            {
                ensure!(
                    expected.contains(info.name.as_str()),
                    "unexpected model tensor {}",
                    info.name
                );
            }
        }
        for spec in &specs {
            crate::check_cancel(cancel)?;
            validate_weight(
                index
                    .get(spec.name.as_str())
                    .copied()
                    .with_context(|| format!("missing tensor {}", spec.name))?,
                spec,
            )?;
        }
        let mut weights = HashMap::new();
        let mut embedding = Vec::new();
        let mut weight_bytes = 0u64;
        for spec in &specs {
            crate::check_cancel(cancel)?;
            let info = index[spec.name.as_str()];
            let raw = gguf.tensor_bytes(&spec.name)?;
            if spec.name == "embed.weight" {
                embedding = decode_small(info, raw)?;
                weight_bytes += (embedding.len() * 4) as u64;
                continue;
            }
            // Bias/norm/conv are small F32 weights on the graph seam. Conv's
            // original [kernel,1,channels] bytes already have channel-major
            // order, so removing the singleton requires no transpose/reversal.
            let (dtype, bytes) = if spec.small {
                (DType::F32, floats_to_bytes(&decode_small(info, raw)?))
            } else {
                (info.dtype, raw.to_vec())
            };
            crate::check_cancel(cancel)?;
            let buffer = backend
                .alloc_uninit(bytes.len(), BufferUsage::Weights)
                .with_context(|| format!("allocate {}", spec.name))?;
            crate::check_cancel(cancel)?;
            backend
                .upload(buffer.as_ref(), &bytes)
                .with_context(|| format!("upload {}", spec.name))?;
            crate::check_cancel(cancel)?;
            weight_bytes += buffer.len_bytes() as u64;
            weights.insert(
                spec.name.clone(),
                Weight {
                    desc: TensorDesc::new(spec.graph_shape.clone(), dtype),
                    buffer,
                },
            );
        }
        crate::check_cancel(cancel)?;
        backend.sync()?;
        crate::check_cancel(cancel)?;
        // GGUF mappings can now close. Backend buffers own their uploaded bytes;
        // the only host-resident model values are the small embedding + tokens.
        Ok(Self {
            backend,
            config: parameters.config,
            frontend: parameters.frontend,
            weights,
            embedding,
            tokens: parameters.tokens,
            weight_bytes,
        })
    }

    pub fn frontend_config(&self) -> &FrontendConfig {
        &self.frontend
    }

    /// Persistent uploaded weight buffers plus host f32 prompt embeddings.
    /// Excludes token strings, allocator overhead and backend-owned derived caches.
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    pub fn config(&self) -> &SenseVoiceConfig {
        &self.config
    }

    pub fn infer(&mut self, features: &Features, language: &str, use_itn: bool) -> Result<String> {
        self.infer_cancellable(features, language, use_itn, &AtomicBool::new(false))
    }

    /// Cancellation is checked between encoder layers, norms and CTC blocks.
    pub fn infer_cancellable(
        &mut self,
        features: &Features,
        language: &str,
        use_itn: bool,
        cancel: &AtomicBool,
    ) -> Result<String> {
        self.infer_observed(features, language, use_itn, None, cancel)
    }

    pub fn infer_with_trace(
        &mut self,
        features: &Features,
        language: &str,
        use_itn: bool,
        trace: &mut TraceCallback<'_>,
    ) -> Result<String> {
        self.infer_observed(
            features,
            language,
            use_itn,
            Some(trace),
            &AtomicBool::new(false),
        )
    }

    fn infer_observed(
        &self,
        features: &Features,
        language: &str,
        use_itn: bool,
        mut trace: Option<&mut TraceCallback<'_>>,
        cancel: &AtomicBool,
    ) -> Result<String> {
        let result = self.forward(features, language, use_itn, &mut trace, cancel);
        // Serial phase boundary, including error exits. The model retains no
        // plan or activation buffer, regardless of the request's shape.
        self.backend.release_transient_runtime();
        result
    }

    fn forward(
        &self,
        features: &Features,
        language: &str,
        use_itn: bool,
        trace: &mut Option<&mut TraceCallback<'_>>,
        cancel: &AtomicBool,
    ) -> Result<String> {
        crate::check_cancel(cancel)?;
        validate_features(features, self.config.input_size)?;
        let ids = prefix_ids(language, use_itn)?;
        let rows = features.rows + 4;
        let mut hidden = self.prepare_input(features, ids, trace)?;
        crate::check_cancel(cancel)?;
        hidden = self.run_layer(
            "encoder.encoders0.0",
            self.config.input_size,
            rows,
            hidden.as_ref(),
            trace,
        )?;
        for i in 0..self.config.encoder_layers - 1 {
            crate::check_cancel(cancel)?;
            hidden = self.run_layer(
                &format!("encoder.encoders.{i}"),
                self.config.hidden_size,
                rows,
                hidden.as_ref(),
                trace,
            )?;
        }
        crate::check_cancel(cancel)?;
        hidden = self.run_norm("encoder.after_norm", rows, hidden.as_ref(), trace)?;
        for i in 0..self.config.tp_layers {
            crate::check_cancel(cancel)?;
            hidden = self.run_layer(
                &format!("encoder.tp_encoders.{i}"),
                self.config.hidden_size,
                rows,
                hidden.as_ref(),
                trace,
            )?;
        }
        crate::check_cancel(cancel)?;
        hidden = self.run_norm("encoder.tp_norm", rows, hidden.as_ref(), trace)?;
        // Chunk only the independent CTC projection, never the bidirectional
        // encoder. Bound logits workspace to 32*vocab floats even for long chunks.
        #[cfg(test)]
        let ctc_started = std::time::Instant::now();
        let mut greedy = Vec::with_capacity(rows);
        for start in (0..rows).step_by(CTC_CHUNK_ROWS) {
            crate::check_cancel(cancel)?;
            let count = (rows - start).min(CTC_CHUNK_ROWS);
            let mut stage = Stage::new(self.backend.as_ref(), trace.is_some());
            let input = stage.borrow_input(hidden.as_ref(), rows, self.config.hidden_size);
            let slice = stage.scratch(count, self.config.hidden_size);
            stage.graph.push(Op::Copy {
                src: input,
                src_off: (start * self.config.hidden_size) as u32,
                dst: slice,
                dst_off: 0,
                n: (count * self.config.hidden_size) as u32,
            });
            let logits = stage.output(count, self.config.vocab_size);
            self.linear(&mut stage, "ctc.ctc_lo", slice, logits, count)?;
            let buffer = stage.run(logits, trace, &format!("ctc.logits.{start}"))?;
            crate::check_cancel(cancel)?;
            let values = download_f32(self.backend.as_ref(), buffer.as_ref())?;
            greedy.extend(greedy_ids(&values, count, self.config.vocab_size)?);
        }
        crate::check_cancel(cancel)?;
        // The first four time steps supervise language/emotion/event/ITN, not
        // speech CTC. Start a fresh collapse at the first acoustic frame.
        #[cfg(test)]
        {
            let decode_started = std::time::Instant::now();
            let text = decode_ctc(&greedy[4..], self.config.blank_id, &self.tokens);
            diagnostics::record(
                "ctc.decode",
                0.0,
                0.0,
                decode_started.elapsed().as_secs_f64(),
            );
            diagnostics::record(
                "ctc.pipeline",
                0.0,
                0.0,
                ctc_started.elapsed().as_secs_f64(),
            );
            text
        }
        #[cfg(not(test))]
        decode_ctc(&greedy[4..], self.config.blank_id, &self.tokens)
    }

    fn prepare_input(
        &self,
        features: &Features,
        ids: [usize; 4],
        trace: &mut Option<&mut TraceCallback<'_>>,
    ) -> Result<Box<dyn Buffer>> {
        let dim = self.config.input_size;
        let rows = features.rows + 4;
        let prefix: Vec<f32> = ids
            .iter()
            .flat_map(|id| self.embedding[id * dim..(id + 1) * dim].iter().copied())
            .collect();
        let position = sinusoidal_position(rows, dim)?;
        if let Some(cb) = trace.as_deref_mut() {
            cb("input.prefix", 4, dim, &prefix)?;
            cb("input.features", features.rows, dim, &features.data)?;
            cb("input.position", rows, dim, &position)?;
        }
        let mut stage = Stage::new(self.backend.as_ref(), trace.is_some());
        let prefix_id = stage.upload_input(&prefix, 4, dim)?;
        let features_id = stage.upload_input(&features.data, features.rows, dim)?;
        let position_id = stage.upload_input(&position, rows, dim)?;
        let combined = stage.scratch(rows, dim);
        stage.graph.push(Op::Copy {
            src: prefix_id,
            src_off: 0,
            dst: combined,
            dst_off: 0,
            n: (4 * dim) as u32,
        });
        stage.graph.push(Op::Copy {
            src: features_id,
            src_off: 0,
            dst: combined,
            dst_off: (4 * dim) as u32,
            n: (features.rows * dim) as u32,
        });
        stage.graph.push(Op::Scale {
            x: combined,
            dst: combined,
            s: (self.config.hidden_size as f32).sqrt(),
            n: (rows * dim) as u32,
        });
        let output = stage.output(rows, dim);
        stage.graph.push(Op::Add {
            a: combined,
            b: position_id,
            dst: output,
            n: (rows * dim) as u32,
        });
        stage.run(output, trace, "encoder.input")
    }

    fn run_layer(
        &self,
        name: &str,
        input_dim: usize,
        rows: usize,
        input: &dyn Buffer,
        trace: &mut Option<&mut TraceCallback<'_>>,
    ) -> Result<Box<dyn Buffer>> {
        let mut stage = self.layer_graph(name, input_dim, rows, input, trace.is_some())?;
        let output = stage.graph.outputs[0];
        stage.run(output, trace, &format!("{name}.output"))
    }

    fn layer_graph<'a>(
        &'a self,
        name: &str,
        input_dim: usize,
        rows: usize,
        input: &'a dyn Buffer,
        tracing: bool,
    ) -> Result<Stage<'a>> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n = (rows * h) as u32;
        let mut stage = Stage::new(self.backend.as_ref(), tracing);
        let input_id = stage.borrow_input(input, rows, input_dim);
        // Output is declared first, before optional trace outputs.
        let output = stage.output(rows, h);
        let normalized = stage.scratch(rows, input_dim);
        self.norm(
            &mut stage,
            &format!("{name}.norm1"),
            input_id,
            normalized,
            rows,
            input_dim,
        )?;
        stage.tap(&format!("{name}.norm1"), normalized);
        let q = stage.scratch(rows, h);
        let k = stage.scratch(rows, h);
        let v = stage.scratch(rows, h);
        for (which, dst) in [("q", q), ("k", k), ("v", v)] {
            self.linear(
                &mut stage,
                &format!("{name}.self_attn.linear_{which}"),
                normalized,
                dst,
                rows,
            )?;
            stage.tap(&format!("{name}.{which}"), dst);
        }
        let attention = stage.scratch(rows, h);
        stage.graph.push(Op::Attention {
            q,
            k_cache: k,
            v_cache: v,
            dst: attention,
            rows: rows as u32,
            kv_len: rows as u32,
            n_head: cfg.attention_heads as u32,
            n_kv: cfg.attention_heads as u32,
            head_dim: (h / cfg.attention_heads) as u32,
            scale: 1.0 / ((h / cfg.attention_heads) as f32).sqrt(),
            mask: AttnMask::Canvas { lo: 0 },
            pos: 0,
            sinks: None,
        });
        stage.tap(&format!("{name}.attention"), attention);
        // k is dead after attention; use its storage for noncausal FSMN(V)+V.
        let conv_weight =
            stage.weight(self.weight(&format!("{name}.self_attn.fsmn_block.weight"))?);
        stage.graph.push(Op::DepthwiseConv1d {
            x: v,
            weight: conv_weight,
            dst: k,
            rows: rows as u32,
            channels: h as u32,
            kernel: cfg.kernel_size as u32,
            pad_left: cfg.pad_left as u32,
            pad_right: cfg.pad_right as u32,
        });
        stage.graph.push(Op::Add {
            a: k,
            b: v,
            dst: k,
            n,
        });
        stage.tap(&format!("{name}.fsmn"), k);
        // q is dead after attention; reuse it for the projected attention.
        self.linear(
            &mut stage,
            &format!("{name}.self_attn.linear_out"),
            attention,
            q,
            rows,
        )?;
        stage.graph.push(Op::Add {
            a: q,
            b: k,
            dst: output,
            n,
        });
        if input_dim == h {
            stage.graph.push(Op::Add {
                a: output,
                b: input_id,
                dst: output,
                n,
            });
        }
        stage.tap(&format!("{name}.attention_residual"), output);
        // Reuse q again for norm2. All old q/k/v uses precede this write.
        self.norm(&mut stage, &format!("{name}.norm2"), output, q, rows, h)?;
        stage.tap(&format!("{name}.norm2"), q);
        let ff = stage.scratch(rows, cfg.linear_units);
        self.linear(&mut stage, &format!("{name}.feed_forward.w_1"), q, ff, rows)?;
        stage.graph.push(Op::Relu {
            x: ff,
            dst: ff,
            n: (rows * cfg.linear_units) as u32,
        });
        stage.tap(&format!("{name}.relu"), ff);
        self.linear(&mut stage, &format!("{name}.feed_forward.w_2"), ff, q, rows)?;
        stage.graph.push(Op::Add {
            a: output,
            b: q,
            dst: output,
            n,
        });
        Ok(stage)
    }

    fn run_norm(
        &self,
        name: &str,
        rows: usize,
        input: &dyn Buffer,
        trace: &mut Option<&mut TraceCallback<'_>>,
    ) -> Result<Box<dyn Buffer>> {
        let mut stage = Stage::new(self.backend.as_ref(), trace.is_some());
        let x = stage.borrow_input(input, rows, self.config.hidden_size);
        let out = stage.output(rows, self.config.hidden_size);
        self.norm(&mut stage, name, x, out, rows, self.config.hidden_size)?;
        stage.run(out, trace, name)
    }

    fn weight(&self, name: &str) -> Result<&Weight> {
        self.weights
            .get(name)
            .with_context(|| format!("missing loaded weight {name}"))
    }

    fn norm<'a>(
        &'a self,
        stage: &mut Stage<'a>,
        name: &str,
        x: TensorId,
        dst: TensorId,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        let w = stage.weight(self.weight(&format!("{name}.weight"))?);
        let bias = stage.weight(self.weight(&format!("{name}.bias"))?);
        stage.graph.push(Op::LayerNorm {
            x,
            weight: w,
            bias,
            dst,
            rows: rows as u32,
            dim: dim as u32,
            eps: self.config.norm_epsilon,
        });
        Ok(())
    }

    fn linear<'a>(
        &'a self,
        stage: &mut Stage<'a>,
        name: &str,
        x: TensorId,
        dst: TensorId,
        rows: usize,
    ) -> Result<()> {
        let w = self.weight(&format!("{name}.weight"))?;
        let out_f = w.desc.shape[0];
        let in_f = w.desc.shape[1];
        let weight = stage.weight(w);
        let bias = stage.weight(self.weight(&format!("{name}.bias"))?);
        stage.graph.push(Op::Linear {
            x,
            weight,
            dst,
            m: rows as u32,
            in_f: in_f as u32,
            out_f: out_f as u32,
            w_off: 0,
        });
        stage.graph.push(Op::AddBias {
            x: dst,
            bias,
            dst,
            rows: rows as u32,
            n: out_f as u32,
        });
        Ok(())
    }
}

struct Stage<'a> {
    backend: &'a dyn Backend,
    graph: Graph,
    borrowed: Vec<(TensorId, &'a dyn Buffer)>,
    owned: Vec<(TensorId, Box<dyn Buffer>)>,
    traces: Vec<(String, TensorId)>,
    tracing: bool,
}

impl<'a> Stage<'a> {
    fn new(backend: &'a dyn Backend, tracing: bool) -> Self {
        let mut graph = Graph::new();
        graph.no_decode_replay = true;
        Self {
            backend,
            graph,
            borrowed: Vec::new(),
            owned: Vec::new(),
            traces: Vec::new(),
            tracing,
        }
    }
    fn borrow_input(&mut self, buffer: &'a dyn Buffer, rows: usize, cols: usize) -> TensorId {
        let id = self
            .graph
            .input(TensorDesc::new(vec![rows, cols], DType::F32));
        self.borrowed.push((id, buffer));
        id
    }
    fn upload_input(&mut self, data: &[f32], rows: usize, cols: usize) -> Result<TensorId> {
        ensure!(data.len() == rows * cols, "stage input size mismatch");
        let buffer = self
            .backend
            .alloc(data.len() * 4, BufferUsage::Activations)?;
        self.backend
            .upload(buffer.as_ref(), &floats_to_bytes(data))?;
        let id = self
            .graph
            .input(TensorDesc::new(vec![rows, cols], DType::F32));
        self.owned.push((id, buffer));
        Ok(id)
    }
    fn weight(&mut self, weight: &'a Weight) -> TensorId {
        let id = self.graph.weight(weight.desc.clone());
        self.borrowed.push((id, weight.buffer.as_ref()));
        id
    }
    fn scratch(&mut self, rows: usize, cols: usize) -> TensorId {
        self.graph
            .internal(TensorDesc::new(vec![rows, cols], DType::F32))
    }
    fn output(&mut self, rows: usize, cols: usize) -> TensorId {
        self.graph
            .output(TensorDesc::new(vec![rows, cols], DType::F32))
    }
    fn tap(&mut self, name: &str, source: TensorId) {
        if !self.tracing {
            return;
        }
        let desc = self.graph.desc(source).clone();
        let n = desc.numel() as u32;
        let id = self.graph.output(desc);
        // Snapshot immediately: later imperative ops may overwrite source.
        self.graph.push(Op::Copy {
            src: source,
            src_off: 0,
            dst: id,
            dst_off: 0,
            n,
        });
        self.traces.push((name.to_owned(), id));
    }
    fn run(
        &mut self,
        output: TensorId,
        trace: &mut Option<&mut TraceCallback<'_>>,
        output_name: &str,
    ) -> Result<Box<dyn Buffer>> {
        #[cfg(test)]
        let stage_started = std::time::Instant::now();
        let mut outputs = self
            .graph
            .outputs
            .iter()
            .map(|&id| {
                let bytes = self.graph.desc(id).numel() * 4;
                Ok((id, self.backend.alloc(bytes, BufferUsage::Activations)?))
            })
            .collect::<Result<Vec<_>>>()?;
        {
            let mut bindings = Bindings::new();
            for &(id, buffer) in &self.borrowed {
                bindings.bind(id, buffer);
            }
            for (id, buffer) in self.owned.iter().chain(outputs.iter()) {
                bindings.bind(*id, buffer.as_ref());
            }
            #[cfg(test)]
            let compile_started = std::time::Instant::now();
            let plan = self
                .backend
                .compile(&self.graph)
                .with_context(|| format!("compile {output_name}"))?;
            #[cfg(test)]
            let compile_seconds = compile_started.elapsed().as_secs_f64();
            #[cfg(test)]
            let execute_started = std::time::Instant::now();
            self.backend
                .execute(plan.as_ref(), &bindings)
                .with_context(|| format!("execute {output_name}"))?;
            self.backend.sync()?;
            #[cfg(test)]
            diagnostics::record(
                output_name,
                compile_seconds,
                execute_started.elapsed().as_secs_f64(),
                stage_started.elapsed().as_secs_f64(),
            );
        }
        if let Some(cb) = trace.as_deref_mut() {
            for (name, id) in self
                .traces
                .iter()
                .map(|(n, i)| (n.as_str(), *i))
                .chain(std::iter::once((output_name, output)))
            {
                let buffer = outputs
                    .iter()
                    .find(|(i, _)| *i == id)
                    .context("trace output binding")?;
                let values = download_f32(self.backend, buffer.1.as_ref())?;
                ensure!(
                    values.iter().all(|x| x.is_finite()),
                    "nonfinite trace tensor {name}"
                );
                let shape = &self.graph.desc(id).shape;
                cb(name, shape[0], shape[1], &values)?;
            }
        }
        let index = outputs
            .iter()
            .position(|(id, _)| *id == output)
            .context("stage output binding")?;
        Ok(outputs.swap_remove(index).1)
    }
}

struct WeightSpec {
    name: String,
    gguf_shape: Vec<usize>,
    graph_shape: Vec<usize>,
    small: bool,
}

fn weight_specs(cfg: &SenseVoiceConfig) -> Vec<WeightSpec> {
    let mut specs = Vec::new();
    let h = cfg.hidden_size;
    let mut linear = |name: String, input: usize, output: usize| {
        specs.push(WeightSpec {
            name: format!("{name}.weight"),
            gguf_shape: vec![input, output],
            graph_shape: vec![output, input],
            small: false,
        });
        specs.push(WeightSpec {
            name: format!("{name}.bias"),
            gguf_shape: vec![output],
            graph_shape: vec![output],
            small: true,
        });
    };
    let layers = std::iter::once(("encoder.encoders0.0".to_string(), cfg.input_size))
        .chain((0..cfg.encoder_layers - 1).map(|i| (format!("encoder.encoders.{i}"), h)))
        .chain((0..cfg.tp_layers).map(|i| (format!("encoder.tp_encoders.{i}"), h)))
        .collect::<Vec<_>>();
    for (name, input) in &layers {
        for which in ["q", "k", "v"] {
            linear(format!("{name}.self_attn.linear_{which}"), *input, h);
        }
        linear(format!("{name}.self_attn.linear_out"), h, h);
        linear(format!("{name}.feed_forward.w_1"), h, cfg.linear_units);
        linear(format!("{name}.feed_forward.w_2"), cfg.linear_units, h);
    }
    linear("ctc.ctc_lo".into(), h, cfg.vocab_size);
    for (name, input) in &layers {
        for (norm, dim) in [("norm1", *input), ("norm2", h)] {
            for suffix in ["weight", "bias"] {
                specs.push(WeightSpec {
                    name: format!("{name}.{norm}.{suffix}"),
                    gguf_shape: vec![dim],
                    graph_shape: vec![dim],
                    small: true,
                });
            }
        }
        specs.push(WeightSpec {
            name: format!("{name}.self_attn.fsmn_block.weight"),
            gguf_shape: vec![cfg.kernel_size, 1, h],
            graph_shape: vec![h, cfg.kernel_size],
            small: true,
        });
    }
    for name in ["encoder.after_norm", "encoder.tp_norm"] {
        for suffix in ["weight", "bias"] {
            specs.push(WeightSpec {
                name: format!("{name}.{suffix}"),
                gguf_shape: vec![h],
                graph_shape: vec![h],
                small: true,
            });
        }
    }
    specs.push(WeightSpec {
        name: "embed.weight".into(),
        gguf_shape: vec![cfg.input_size, 16],
        graph_shape: vec![16, cfg.input_size],
        small: true,
    });
    specs
}

fn validate_weight(info: &TensorInfo, spec: &WeightSpec) -> Result<()> {
    ensure!(
        info.shape == spec.gguf_shape,
        "{} has GGUF shape {:?}; expected fastest-first {:?}",
        info.name,
        info.shape,
        spec.gguf_shape
    );
    ensure!(
        matches!(info.dtype, DType::F32 | DType::F16 | DType::Q8_0),
        "{}: unsupported v1 SenseVoice dtype {:?}; expected F32/F16/Q8_0",
        info.name,
        info.dtype
    );
    if info.dtype == DType::Q8_0 {
        ensure!(
            info.shape[0].is_multiple_of(32),
            "{}: Q8_0 row width {} is not block aligned",
            info.name,
            info.shape[0]
        );
    }
    let numel = info.shape.iter().product();
    ensure!(
        info.nbytes == infr_gguf::nbytes(info.dtype, numel),
        "{}: tensor byte size mismatch",
        info.name
    );
    Ok(())
}

fn decode_small(info: &TensorInfo, raw: &[u8]) -> Result<Vec<f32>> {
    let values =
        dequant_block(info.dtype, raw).with_context(|| format!("dequantize {}", info.name))?;
    ensure!(
        values.len() == info.shape.iter().product::<usize>(),
        "{}: decoded tensor length mismatch",
        info.name
    );
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "{}: nonfinite weight",
        info.name
    );
    Ok(values)
}

fn floats_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn download_f32(backend: &dyn Backend, buffer: &dyn Buffer) -> Result<Vec<f32>> {
    ensure!(
        buffer.len_bytes().is_multiple_of(4),
        "F32 buffer byte length must be divisible by four"
    );
    let mut bytes = vec![0u8; buffer.len_bytes()];
    backend.download(buffer, &mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}

fn validate_features(features: &Features, input_size: usize) -> Result<()> {
    ensure!(
        features.cols == input_size,
        "SenseVoice expects {input_size} CMVN/LFR columns; got {}",
        features.cols
    );
    ensure!(
        (1..=MAX_FEATURE_ROWS).contains(&features.rows),
        "SenseVoice requires 1..={MAX_FEATURE_ROWS} LFR rows; segment audio into <=30-second chunks"
    );
    ensure!(
        features.rows.checked_mul(features.cols) == Some(features.data.len()),
        "feature shape/data length mismatch"
    );
    ensure!(
        features.data.iter().all(|v| v.is_finite()),
        "features contain NaN or infinity"
    );
    Ok(())
}

fn prefix_ids(language: &str, use_itn: bool) -> Result<[usize; 4]> {
    let lid = match language {
        "auto" => 0,
        "zh" => 3,
        "en" => 4,
        "yue" => 7,
        "ja" => 11,
        "ko" => 12,
        "nospeech" => 13,
        _ => bail!("unsupported SenseVoice language {language:?}"),
    };
    Ok([lid, 1, 2, if use_itn { 14 } else { 15 }])
}

fn sinusoidal_position(rows: usize, dim: usize) -> Result<Vec<f32>> {
    ensure!(
        dim >= 4 && dim.is_multiple_of(2),
        "position depth must be even and at least four"
    );
    let half = dim / 2;
    let increment = 10000.0f32.ln() / (half - 1) as f32;
    let mut result = vec![0.0; rows * dim];
    for row in 0..rows {
        for i in 0..half {
            let phase = (row + 1) as f32 * (-(i as f32) * increment).exp();
            result[row * dim + i] = phase.sin();
            result[row * dim + half + i] = phase.cos();
        }
    }
    Ok(result)
}

fn greedy_ids(logits: &[f32], rows: usize, vocab: usize) -> Result<Vec<usize>> {
    ensure!(
        vocab > 0 && rows.checked_mul(vocab) == Some(logits.len()),
        "invalid CTC logit shape"
    );
    logits
        .chunks_exact(vocab)
        .map(|row| {
            ensure!(
                row.iter().all(|v| v.is_finite()),
                "CTC logits contain NaN or infinity"
            );
            let mut best = 0;
            for i in 1..vocab {
                if row[i] > row[best] {
                    best = i;
                }
            }
            Ok(best)
        })
        .collect()
}

fn decode_ctc(ids: &[usize], blank: usize, tokens: &[String]) -> Result<String> {
    ensure!(blank < tokens.len(), "CTC blank id out of range");
    let mut previous = None;
    let mut text = Vec::new();
    for &id in ids {
        let token = tokens.get(id).context("CTC token id out of range")?;
        let duplicate = previous == Some(id);
        // Blanks and special tokens still update the consecutive-id state.
        // A,blank,A -> AA; A,A -> A. Never remove blank before collapsing.
        previous = Some(id);
        if duplicate
            || id == blank
            || (token.starts_with("<|") && token.ends_with("|>"))
            || matches!(
                token.as_str(),
                "<s>" | "</s>" | "<unk>" | "<blank>" | "<pad>"
            )
        {
            continue;
        }
        // SentencePiece byte-fallback pieces must join before UTF-8 decoding.
        if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
            if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                text.push(byte);
                continue;
            }
        }
        // The lovemefan converter has already replaced U+2581 by ASCII spaces;
        // support the original pieces as well without tokenizing the transcript.
        text.extend_from_slice(token.replace('\u{2581}', " ").as_bytes());
    }
    Ok(String::from_utf8_lossy(&text).trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_cpu::CpuBackend;
    use std::sync::Arc;

    fn tokens(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn ctc_collapses_before_removing_blank_and_keeps_blank_a() {
        let vocab = tokens(&["<unk>", "A", "B"]);
        assert_eq!(decode_ctc(&[0, 1], 0, &vocab).unwrap(), "A");
        assert_eq!(decode_ctc(&[1, 1], 0, &vocab).unwrap(), "A");
        assert_eq!(decode_ctc(&[1, 0, 1], 0, &vocab).unwrap(), "AA");
        assert_eq!(
            decode_ctc(&[1, 0, 0, 1, 1, 2, 2], 0, &vocab).unwrap(),
            "AAB"
        );
        assert_eq!(decode_ctc(&[0, 0], 0, &vocab).unwrap(), "");
        assert!(decode_ctc(&[3], 0, &vocab).is_err());
        // Do not assume blank is always zero in the generic collapse helper.
        assert_eq!(
            decode_ctc(&[0, 1, 0], 1, &tokens(&["A", "<blank>"])).unwrap(),
            "AA"
        );
    }

    #[test]
    fn ctc_strips_controls_and_decodes_both_sentencepiece_layouts() {
        let vocab = tokens(&[
            "<unk>",
            "<s>",
            "</s>",
            "<|zh|>",
            "<|HAPPY|>",
            "▁hello",
            " world",
            "你",
            "好",
            "<|withitn|>",
            "<0xE4>",
            "<0xBD>",
            "<0xA0>",
        ]);
        assert_eq!(
            decode_ctc(&[3, 4, 5, 0, 6, 9, 2], 0, &vocab).unwrap(),
            "hello world"
        );
        assert_eq!(decode_ctc(&[7, 0, 7, 8], 0, &vocab).unwrap(), "你你好");
        assert_eq!(decode_ctc(&[10, 11, 12], 0, &vocab).unwrap(), "你");
    }

    #[test]
    fn greedy_uses_first_tie_and_rejects_nonfinite_or_bad_shape() {
        assert_eq!(
            greedy_ids(&[2.0, 2.0, -1.0, 0.0, 1.0, 3.0], 2, 3).unwrap(),
            [0, 2]
        );
        assert!(greedy_ids(&[f32::NAN, 1.0], 1, 2).is_err());
        assert!(greedy_ids(&[f32::INFINITY, 1.0], 1, 2).is_err());
        assert!(greedy_ids(&[0.0], 2, 2).is_err());
    }

    #[test]
    fn prompt_ids_and_position_match_official_endpoints() {
        for (language, lid) in [
            ("auto", 0),
            ("zh", 3),
            ("en", 4),
            ("yue", 7),
            ("ja", 11),
            ("ko", 12),
            ("nospeech", 13),
        ] {
            assert_eq!(prefix_ids(language, true).unwrap(), [lid, 1, 2, 14]);
            assert_eq!(prefix_ids(language, false).unwrap(), [lid, 1, 2, 15]);
        }
        assert!(prefix_ids("unknown", true).is_err());
        let pe = sinusoidal_position(3, 560).unwrap();
        for row in 0..3 {
            let pos = (row + 1) as f32;
            assert!((pe[row * 560] - pos.sin()).abs() < 1e-6);
            assert!((pe[row * 560 + 280] - pos.cos()).abs() < 1e-6);
            // Last frequency is 1/10000, not 10000^(-558/560).
            assert!((pe[row * 560 + 279] - (pos / 10000.0).sin()).abs() < 1e-8);
        }
    }

    #[test]
    fn features_and_tensor_layout_are_validated_before_execution() {
        let mut f = Features {
            data: vec![0.0; 560],
            rows: 1,
            cols: 560,
        };
        validate_features(&f, 560).unwrap();
        f.cols = 80;
        assert!(validate_features(&f, 560).is_err());
        f.cols = 560;
        f.data[3] = f32::NAN;
        assert!(validate_features(&f, 560).is_err());
        f.data[3] = 0.0;
        f.rows = MAX_FEATURE_ROWS + 1;
        assert!(validate_features(&f, 560).is_err());
        let spec = WeightSpec {
            name: "q.weight".into(),
            gguf_shape: vec![560, 512],
            graph_shape: vec![512, 560],
            small: false,
        };
        let mut info = TensorInfo {
            name: spec.name.clone(),
            shape: vec![560, 512],
            dtype: DType::F32,
            offset: 0,
            nbytes: 560 * 512 * 4,
        };
        validate_weight(&info, &spec).unwrap();
        info.shape.reverse();
        assert!(validate_weight(&info, &spec).is_err());
        info.shape.reverse();
        info.dtype = DType::Q8_0;
        info.nbytes = infr_gguf::nbytes(DType::Q8_0, 560 * 512);
        assert!(
            validate_weight(&info, &spec).is_err(),
            "560 is not a whole Q8 row"
        );
    }

    fn cpu() -> Box<dyn Backend> {
        Box::new(CpuBackend::new_with(Arc::new(
            infr_core::config::Config::default(),
        )))
    }

    // Exact small graph: Q=K=0 (uniform noncausal attention), V=[1,2],
    // identity attention output; FSMN channels have different asymmetric taps.
    // FFN has zero input weights and biases [-2,3], identity output weights.
    fn tiny_model(input_dim: usize) -> SenseVoiceModel {
        let backend = cpu();
        let cfg = SenseVoiceConfig {
            input_size: input_dim,
            hidden_size: 2,
            attention_heads: 1,
            linear_units: 2,
            encoder_layers: 1,
            tp_layers: 0,
            kernel_size: 3,
            pad_left: 1,
            pad_right: 1,
            norm_epsilon: 1e-5,
            blank_id: 0,
            vocab_size: 3,
            compatibility_defaults: Vec::new(),
        };
        let mut weights = HashMap::new();
        for spec in weight_specs(&cfg) {
            if spec.name == "embed.weight" {
                continue;
            }
            let mut values = vec![0.0; spec.graph_shape.iter().product()];
            match spec.name.as_str() {
                "encoder.encoders0.0.self_attn.linear_v.bias" => {
                    values.copy_from_slice(&[1.0, 2.0])
                }
                "encoder.encoders0.0.self_attn.fsmn_block.weight" => {
                    values.copy_from_slice(&[1.0, 0.0, 2.0, 2.0, 0.0, 1.0])
                }
                "encoder.encoders0.0.self_attn.linear_out.weight"
                | "encoder.encoders0.0.feed_forward.w_2.weight" => {
                    values.copy_from_slice(&[1.0, 0.0, 0.0, 1.0])
                }
                "encoder.encoders0.0.feed_forward.w_1.bias" => values.copy_from_slice(&[-2.0, 3.0]),
                n if n.ends_with("norm1.weight") || n.ends_with("norm2.weight") => values.fill(1.0),
                _ => {}
            }
            let bytes = floats_to_bytes(&values);
            let buffer = backend.alloc(bytes.len(), BufferUsage::Weights).unwrap();
            backend.upload(buffer.as_ref(), &bytes).unwrap();
            weights.insert(
                spec.name,
                Weight {
                    desc: TensorDesc::new(spec.graph_shape, DType::F32),
                    buffer,
                },
            );
        }
        SenseVoiceModel {
            backend,
            config: cfg,
            frontend: FrontendConfig::default(),
            weights,
            embedding: vec![0.0; 16 * input_dim],
            tokens: tokens(&["<unk>", "A", "B"]),
            weight_bytes: 0,
        }
    }

    #[test]
    fn cancellation_after_first_layer_stops_before_next_layer() {
        let mut model = tiny_model(4);
        // No second-layer weights exist: reaching that layer would fail differently.
        model.config.encoder_layers = 2;
        let features = Features {
            data: vec![0.0; 4],
            rows: 1,
            cols: 4,
        };
        let cancel = AtomicBool::new(false);
        let mut layers = Vec::new();
        let error = model
            .infer_observed(
                &features,
                "auto",
                true,
                Some(&mut |name, _, _, _| {
                    if name.ends_with(".output") {
                        layers.push(name.to_owned());
                        cancel.store(true, std::sync::atomic::Ordering::Release);
                    }
                    Ok(())
                }),
                &cancel,
            )
            .unwrap_err();
        assert!(error.downcast_ref::<crate::wav::AudioCancelled>().is_some());
        assert_eq!(layers, vec!["encoder.encoders0.0.output"]);
    }

    #[test]
    fn cancellation_at_ctc_block_stops_before_next_projection() {
        let model = tiny_model(4);
        let features = Features {
            data: vec![0.0; 40 * 4],
            rows: 40,
            cols: 4,
        };
        let cancel = AtomicBool::new(false);
        let mut blocks = Vec::new();
        let error = model
            .infer_observed(
                &features,
                "auto",
                true,
                Some(&mut |name, _, _, _| {
                    if name.starts_with("ctc.logits.") {
                        blocks.push(name.to_owned());
                        cancel.store(true, std::sync::atomic::Ordering::Release);
                    }
                    Ok(())
                }),
                &cancel,
            )
            .unwrap_err();
        assert!(error.downcast_ref::<crate::wav::AudioCancelled>().is_some());
        assert_eq!(blocks, vec!["ctc.logits.0"]);
    }

    #[test]
    fn cancelled_load_does_not_open_gguf() {
        let error = SenseVoiceModel::load_cancellable(
            Path::new("must-not-open-cancelled-audio-model.gguf"),
            cpu(),
            &AtomicBool::new(true),
        )
        .err()
        .expect("pre-cancelled load must fail");
        assert!(error.downcast_ref::<crate::wav::AudioCancelled>().is_some());
    }

    #[test]
    fn cpu_first_layer_skips_input_residual_fsmn_is_noncausal_and_ffn_is_relu() {
        let model = tiny_model(4);
        let input = model
            .backend
            .alloc(3 * 4 * 4, BufferUsage::Activations)
            .unwrap();
        model
            .backend
            .upload(input.as_ref(), &floats_to_bytes(&[10.0; 12]))
            .unwrap();
        let output = model
            .run_layer("encoder.encoders0.0", 4, 3, input.as_ref(), &mut None)
            .unwrap();
        let values = download_f32(model.backend.as_ref(), output.as_ref()).unwrap();
        assert_eq!(values, [4.0, 9.0, 5.0, 13.0, 3.0, 11.0]);
    }

    #[test]
    fn cpu_equal_dim_layer_keeps_input_residual_and_trace_snapshots_survive_reuse() {
        let model = tiny_model(2);
        let input = model
            .backend
            .alloc(3 * 2 * 4, BufferUsage::Activations)
            .unwrap();
        model
            .backend
            .upload(input.as_ref(), &floats_to_bytes(&[10.0; 6]))
            .unwrap();
        let mut seen = HashMap::new();
        let mut callback = |name: &str, rows: usize, cols: usize, values: &[f32]| -> Result<()> {
            assert_eq!(values.len(), rows * cols);
            seen.insert(name.to_owned(), values.to_vec());
            Ok(())
        };
        let output = model
            .run_layer(
                "encoder.encoders0.0",
                2,
                3,
                input.as_ref(),
                &mut Some(&mut callback),
            )
            .unwrap();
        assert_eq!(
            download_f32(model.backend.as_ref(), output.as_ref()).unwrap(),
            [14.0, 19.0, 15.0, 23.0, 13.0, 21.0]
        );
        assert_eq!(seen["encoder.encoders0.0.q"], [0.0; 6]);
        assert_eq!(
            seen["encoder.encoders0.0.fsmn"],
            [3.0, 4.0, 4.0, 8.0, 2.0, 6.0]
        );
        assert_eq!(
            seen["encoder.encoders0.0.relu"],
            [0.0, 3.0, 0.0, 3.0, 0.0, 3.0]
        );
    }

    #[test]
    fn cpu_attention_reads_future_values_and_repeated_layers_have_constant_graph_size() {
        let backend = cpu();
        let mut stage = Stage::new(backend.as_ref(), false);
        let q = stage.upload_input(&[0.0, 0.0, 0.0], 3, 1).unwrap();
        let k = stage.upload_input(&[0.0, 0.0, 0.0], 3, 1).unwrap();
        let v = stage.upload_input(&[1.0, 4.0, 10.0], 3, 1).unwrap();
        let out = stage.output(3, 1);
        stage.graph.push(Op::Attention {
            q,
            k_cache: k,
            v_cache: v,
            dst: out,
            rows: 3,
            kv_len: 3,
            n_head: 1,
            n_kv: 1,
            head_dim: 1,
            scale: 1.0,
            mask: AttnMask::Canvas { lo: 0 },
            pos: 0,
            sinks: None,
        });
        let output = stage.run(out, &mut None, "uniform").unwrap();
        assert_eq!(
            download_f32(backend.as_ref(), output.as_ref()).unwrap(),
            [5.0; 3]
        );
        let model = tiny_model(2);
        let input = model
            .backend
            .alloc(3 * 2 * 4, BufferUsage::Activations)
            .unwrap();
        let mut size = None;
        for _ in 0..8 {
            let graph = model
                .layer_graph("encoder.encoders0.0", 2, 3, input.as_ref(), false)
                .unwrap();
            assert!(graph.graph.tensors.len() < 32);
            let current = (graph.graph.tensors.len(), graph.graph.ops.len());
            if let Some(previous) = size {
                assert_eq!(current, previous);
            }
            size = Some(current);
        }
    }

    #[test]
    fn cpu_prefix_scaling_uses_hidden_size_not_560_and_prefix_precedes_features() {
        let mut model = tiny_model(4);
        for row in 0..16 {
            model.embedding[row * 4..row * 4 + 4].fill(row as f32);
        }
        let features = Features {
            data: vec![20.0; 4],
            rows: 1,
            cols: 4,
        };
        let out = model
            .prepare_input(&features, prefix_ids("zh", true).unwrap(), &mut None)
            .unwrap();
        let values = download_f32(model.backend.as_ref(), out.as_ref()).unwrap();
        let position = sinusoidal_position(5, 4).unwrap();
        for (r, value) in [3.0, 1.0, 2.0, 14.0, 20.0].into_iter().enumerate() {
            for c in 0..4 {
                assert!(
                    (values[r * 4 + c] - (value * 2.0f32.sqrt() + position[r * 4 + c])).abs()
                        < 1e-5
                );
            }
        }
    }

    // Opt-in, public-input-only harness. No downloads and no filesystem output.
    // Caller supplies MOE4ALL_AUDIO_TEST_MODEL and MOE4ALL_AUDIO_TEST_WAV. Optional:
    // MOE4ALL_AUDIO_TEST_EXPECT_TEXT (exact text), MOE4ALL_AUDIO_TEST_LANGUAGE (zh),
    // MOE4ALL_AUDIO_TEST_TRACE (comma-separated exact tensor names; stdout only).
    // CPU kernel settings still enter through Config::load_from_env, as usual.
    // The main validation session owns Cargo output dirs and stdout evidence.
    fn public_model(reference: bool) -> Result<()> {
        use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};
        let model_path = PathBuf::from(
            std::env::var_os("MOE4ALL_AUDIO_TEST_MODEL")
                .context("set MOE4ALL_AUDIO_TEST_MODEL to a local public GGUF")?,
        );
        let wav_path = PathBuf::from(
            std::env::var_os("MOE4ALL_AUDIO_TEST_WAV")
                .context("set MOE4ALL_AUDIO_TEST_WAV to a local public WAV")?,
        );
        let language = std::env::var("MOE4ALL_AUDIO_TEST_LANGUAGE").unwrap_or_else(|_| "zh".into());
        let wanted_trace = std::env::var("MOE4ALL_AUDIO_TEST_TRACE").unwrap_or_default();
        let cfg = Arc::new(infr_core::config::Config::load_from_env());
        let backend: Box<dyn Backend> = if reference {
            Box::new(CpuBackend::reference_with(cfg))
        } else {
            Box::new(CpuBackend::new_with(cfg))
        };
        let start = Instant::now();
        let mut model = SenseVoiceModel::load(&model_path, backend)?;
        println!(
            "reference={reference} load_seconds={} weight_bytes={} compatibility_defaults={:?}",
            start.elapsed().as_secs_f64(),
            model.weight_bytes(),
            model.config().compatibility_defaults
        );
        let mut transcript = String::new();
        crate::wav::for_each_wav_segment(
            &wav_path,
            28.0,
            &AtomicBool::new(false),
            &mut |segment| {
                let features =
                    crate::frontend::extract_features(&segment.samples, model.frontend_config())?;
                let start = Instant::now();
                let first = if wanted_trace.is_empty() {
                    model.infer(&features, &language, true)?
                } else {
                    model.infer_with_trace(
                        &features,
                        &language,
                        true,
                        &mut |name, rows, cols, values| {
                            if wanted_trace.split(',').any(|wanted| wanted == name) {
                                println!(
                                    "trace name={name} rows={rows} cols={cols} values={values:?}"
                                );
                            }
                            Ok(())
                        },
                    )?
                };
                println!(
                    "reference={reference} rows={} infer_seconds={} text={first:?}",
                    features.rows,
                    start.elapsed().as_secs_f64()
                );
                // The second call reuses the fixed weight buffers and no prior audio
                // state. Also catches stale CPU weight-cache identities/activations.
                let second = model.infer(&features, &language, true)?;
                ensure!(
                    first == second,
                    "repeated public inference changed transcript"
                );
                transcript.push_str(&first);
                Ok(())
            },
        )?;
        println!("transcript={transcript:?}");
        ensure!(
            !transcript.is_empty(),
            "public sample produced no transcript"
        );
        if let Ok(expected) = std::env::var("MOE4ALL_AUDIO_TEST_EXPECT_TEXT") {
            ensure!(
                transcript == expected,
                "public transcript mismatch: {transcript:?} != {expected:?}"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "explicit local public GGUF/WAV required; main session owns execution"]
    fn public_model_cpu() -> Result<()> {
        public_model(false)
    }

    #[test]
    #[ignore = "explicit local public GGUF/WAV required; main session owns execution"]
    fn public_model_cpu_reference() -> Result<()> {
        public_model(true)
    }
}
