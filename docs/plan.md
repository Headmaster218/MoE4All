# infr — original project plan (historical / broad roadmap)

> **Status:** this is the original master plan from the DiffusionGemma era. Most
> of it shipped — against autoregressive decoders (Llama / Qwen3 / Gemma), not
> DiffusionGemma. It's kept for the architecture overview, crate layout, the
> DiffusionGemma spec, and reference links. The backend-agnostic compute
> refactor it anticipated has since shipped: the op-list IR + `Backend` trait
> live in `crates/infr-core` (`graph.rs`, `backend.rs`) and run on CPU
> (`infr-cpu`), Vulkan (`infr-vulkan`), and Metal (`infr-metal`). Current perf
> state: [`docs/perf.md`](perf.md).

Pure-Rust LLM inference engine. Vulkan-first, designed to run on any mainstream
GPU. The only non-Rust surface is the GPU driver (called through thin Rust FFI)
and the compute shaders (SPIR-V).

---

## Vision

A from-the-metal inference server where **the server and model code never know
which GPU API is running underneath**. We ship a Vulkan backend first (covers
AMD/NVIDIA/Intel, and Apple via MoltenVK), then add native CUDA / Metal backends
later **without touching any layer above the backend**.

The whole architecture is organized around four pluggable seams so that "add a
GPU", "add a model", "add a format", or "add a decode style" each means
_implementing one trait_, never refactoring the stack.

---

## MVP scope

Deliberately narrow, to ship something real on the author's hardware first.

| Dimension    | MVP                                           | Later                                 |
| ------------ | --------------------------------------------- | ------------------------------------- |
| Format       | **GGUF**                                      | safetensors, MLX                      |
| Model source | **HuggingFace + Ollama pull**, or local path  | other registries, mirrors             |
| Model        | **DiffusionGemma** (diffusion decode)         | Llama / Qwen / Gemma (autoregressive) |
| GPU backend  | **Vulkan** (`ash` + SPIR-V) on **AMD (RADV)** | CUDA, native Metal                    |
| Decode       | **diffusion** (block denoise)                 | autoregressive (greedy / sampling)    |
| API          | **OpenAI-compatible HTTP** (streaming)        | embeddings, batching, multi-model     |

**MVP done = `curl`/opencode/Claude Code CLI can hold a streaming chat with
DiffusionGemma served from `infr` over Vulkan on the 7900 XTX.**

### Product surface (the `infr` CLI)

Three commands, all sharing the same engine + backend underneath (ollama-like
UX):

```bash
infr pull  <model-ref>        # download + cache a GGUF model (HF or Ollama)
infr run   <model-ref> [msg]  # interactive terminal chat (auto-pulls if missing)
infr serve <model-ref>        # start the OpenAI-compatible HTTP API
```

- **`infr pull`** — resolve a model reference and fetch the GGUF into the local
  cache. References:
  - `hf:org/repo[:file.gguf]` — HuggingFace (resolve repo, pick/verify the
    GGUF).
  - `ollama:`/`ol:` `name[:tag]` — standalone Ollama registry pull (HTTP).
  - a plain filesystem path to a `.gguf`.
  - (also: `infr bench` and `infr compare` for tok/s benchmarking vs llama.cpp.)
- **`infr run`** — load the model on the backend and drop into a simple REPL
  chat in the terminal (streams tokens, shows reasoning vs answer). Auto-pulls
  if not cached.
- **`infr serve`** — same load path, but exposes `/v1/chat/completions` +
  `/v1/models` (streaming) for opencode / Claude Code CLI. Auto-pulls if not
  cached.

`run` and `serve` are two front-ends over the **same** `infr-engine` (load →
backend → decode); `pull` is just the acquisition half on its own. Nothing in
`run`/`serve`/the engine knows which GPU backend is active.

### End-to-end flow

```
model-ref ──▶ infr-hub (resolve + download + cache)
          ──▶ infr-gguf (mmap + parse + upload tensors to backend buffers)
          ──▶ infr-model (build DiffusionGemma graph)  ──▶ Backend.compile()
          ──▶ infr-decode (diffusion loop)             ──▶ Backend.execute() per step
          ──▶ run: terminal REPL    |    serve: OpenAI HTTP/SSE
```

### Non-goals (MVP)

- No autoregressive models yet (diffusion only — but see note below).
- No training, no fine-tuning, no quantization _creation_ (we only _consume_
  GGUF quants).
- No multi-GPU, no tensor/pipeline parallelism.
- No non-Vulkan backends yet (the abstraction exists; impls come later).
- Perf is "correct and usable," not "beat llama.cpp." Tuning is a later phase.

> Note: "diffusion only" does **not** skip the transformer. DiffusionGemma's
> forward pass is a full transformer (30 layers, sliding-window attention on 25
> / full attention on 5, MoE with ~3.8B active params, RoPE, RMSNorm). The
> "diffusion" is the _decode strategy_ layered on top of that forward. So the
> MVP implements a complete transformer forward + a diffusion decode loop.

---

## Architecture

Bottom-up. Each named trait is the seam where future variants plug in.

```
┌────────────────────────────────────────────────────────────────────┐
│ server      axum + SSE  ->  OpenAI /v1/chat/completions, /v1/models  │  knows NOTHING about the GPU
├────────────────────────────────────────────────────────────────────┤
│ engine      session orchestration, chat templating, tool-call bridge │
├────────────────────────────────────────────────────────────────────┤
│ decode      trait DecodeStrategy   -> DiffusionDenoise  (AR later)   │
├────────────────────────────────────────────────────────────────────┤
│ model       trait Model            -> DiffusionGemma     (Llama …)   │  builds an abstract compute Graph
├────────────────────────────────────────────────────────────────────┤
│ runtime     Tensor handles, KV cache, Graph builder                  │
├────────────────────────────────────────────────────────────────────┤
│ loader      trait WeightSource     -> Gguf  (safetensors later)      │
├────────────────────────────────────────────────────────────────────┤
│ compute     trait Backend          -> Vulkan (CUDA / Metal …) │  the ONLY GPU-aware layer
├────────────────────────────────────────────────────────────────────┤
│ shaders     SPIR-V (reused / ported from ggml-vulkan)                │  not Rust
└────────────────────────────────────────────────────────────────────┘
```

Dependency rule: **everything above `compute` is generic over the backend.** The
`server` depends on `engine`, which holds a backend as `Box<dyn Backend>` (or a
generic `B: Backend`) and otherwise treats it as opaque.

---

## The backend abstraction (the core design goal)

> **Shipped.** The seam landed as `infr-core::{Graph, Op, Backend, Bindings}`
> with a validated CPU reference backend (`infr-cpu`), the Vulkan adapter
> (`infr-vulkan`), and a Metal backend (`infr-metal`).

The hard requirement still holds: _the server does not know what's running
behind it, and a new driver (CUDA / Metal / MLX) can be dropped in without
changing anything above `compute`._ The seam is drawn at the level of **semantic
tensor ops**: the model builds an ordered op-list (`Graph`) over typed tensor
handles; each backend compiles + executes it however it likes (Vulkan SPIR-V,
CPU loops, later cuBLAS / rocBLAS / Metal / MLX). The as-built trait, op set,
and the dtype-awareness decision live in the `infr-core` source (`graph.rs`,
`backend.rs`).

---

## Crate layout (Cargo workspace)

```
infr/
├── crates/
│   ├── infr-core       # Tensor, dtypes/quant, Graph, Op, Backend trait, errors
│   ├── infr-vulkan     # Backend impl: ash + gpu-allocator + SPIR-V dispatch
│   ├── infr-hub        # model resolve + download + cache (HuggingFace, Ollama)
│   ├── infr-gguf       # WeightSource impl: GGUF parse + metadata + tensor mapping
│   ├── infr-model      # Model trait + DiffusionGemma graph builder
│   ├── infr-decode     # DecodeStrategy + DiffusionDenoise (entropy-bound)
│   ├── infr-engine     # load pipeline, session orchestration, chat template, tool-call bridge
│   └── infr-server     # axum OpenAI-compatible HTTP + SSE
├── shaders/            # GLSL/comp sources + build step -> SPIR-V (reuse ggml-vulkan)
├── src/
│   └── main.rs         # the `infr` CLI: pull / run / serve  (clap subcommands)
├── bin/
│   └── smoke           # dev: f16 coop-matrix matmul on the GPU vs CPU reference
├── plan.md
├── README.md
└── LICENSE
```

---

## Component notes

### compute / Vulkan (`infr-vulkan`)

- `ash` for raw Vulkan; `gpu-allocator` for VRAM suballocation.
- Reuse **ggml's Vulkan `.comp` shaders → SPIR-V** for the tuned kernels (quant
  matmul, dequant, attention) instead of re-deriving quant math; compile at
  build time.
- Use `VK_KHR_cooperative_matrix` (verified on RADV/gfx1100) for fast f16
  matmul; fall back to a scalar/subgroup path where unavailable
  (capability-gated).
- Async compute queue + command-buffer batching for the
  compile-once/execute-many model.

### fetch / model acquisition (`infr-hub`)

- Resolve a model reference to a concrete local GGUF path, downloading + caching
  if needed.
- **infr has its OWN store** (we do NOT touch the system Ollama dirs — the
  root-owned `/var/lib/ollama` caused permission failures). The on-disk layout
  is still OCI/Ollama-style (content-addressed blobs dedup naturally), just
  under our own root.
  - Location: `$INFR_MODELS` if set, else `$XDG_CACHE_HOME/infr/models`
    (`~/.cache/infr/models`). Layout:
    `manifests/registry.ollama.ai/<ns>/<name>/<tag>` (ollama) +
    `manifests/huggingface.co/<org>/<repo>/<file>` (hf) +
    `blobs/sha256-<digest>`.
  - Resolve a tag: read its manifest, find the layer with mediaType
    `application/vnd.ollama.image.model` → that blob **is** the GGUF.
- **Ollama** (`ollama:`/`ol:` `name[:tag]`): standalone registry pull over plain
  HTTP (`registry.ollama.ai/v2/...` manifest + blobs) — no `ollama` CLI invoked.
  Writes blobs + manifest into our store; digest-verified.
- **HuggingFace** (`hf:`/`huggingface:` `org/repo[:file]`): hub HTTP API —
  resolve repo, list siblings, pick/verify the `.gguf`, download via
  `resolve/main/...` with resume, honor `HF_TOKEN`. Writes a synthesized
  manifest + the GGUF blob.
- A plain filesystem path to a `.gguf` is used as-is (no copy into the store).
- Both pulls: streaming with resume (HTTP Range), idempotent (skip if blob
  present), digest-verified, shared auto-width progress bar. No external CLI.

### loader / GGUF (`infr-gguf`)

- Parse GGUF: header, metadata KVs (arch, hyperparams, tokenizer, chat
  template), tensor directory (name, dtype, offset, shape).
- Memory-map the file; upload tensors to backend buffers (quantized blocks stay
  quantized).
- Expose embedded tokenizer + jinja chat template to the engine.

### model / DiffusionGemma (`infr-model`)

- Build the transformer graph from GGUF weights: token embed → 30 × (RMSNorm,
  attention, RMSNorm, MoE-FFN) → final norm → output projection.
- Attention: GQA; **sliding-window** for 25 layers, **full** for 5 (per the
  head_count_kv pattern); RoPE; per-layer KV.
- MoE: router + top-k expert gather (~3.8B active of 26B).
- Reference: the patched llama.cpp `diffusion-gemma` graph.

### decode / diffusion (`infr-decode`)

- Block (canvas) diffusion: denoise a fixed-size canvas over N steps;
  entropy-bound early stop; self-conditioning; commit per block.
- Channels: split `<|channel>thought` / `<channel|>` into reasoning vs final
  answer.

### engine + server (`infr-engine`, `infr-server`)

- OpenAI `/v1/chat/completions` (streaming + non-streaming) and `/v1/models`.
- Stream `reasoning_content` (thought) separately from `content` (answer).
- Parse the model's native `<|tool_call>call:NAME{…}<tool_call|>` into OpenAI
  `tool_calls`; pass tool definitions/results through so opencode/Claude Code
  agentic loops work.
- `engine` is generic over the backend; `server` only talks to `engine`.

---

## Target hardware (verified)

AMD RX 7900 XTX (RADV NAVI31, gfx1100), Mesa 26.1, Vulkan 1.4:

- `VK_KHR_cooperative_matrix` rev 2, `cooperativeMatrix = true` → fast matmul
  path available.
- `shaderFloat16 = true`, `VK_KHR_16bit_storage` → native f16.
- subgroup extended types / rotate → reductions, softmax, attention.
- `maxStorageBufferRange = 4 GB` → large weight tensors fit per binding.
- `maxComputeSharedMemorySize = 64 KB`.

---

## Roadmap / milestones

The MVP shipped against **autoregressive** decoders (Llama / Qwen2 / Qwen3 /
Qwen3-MoE / Gemma 3 / Gemma 4 incl. E2B) rather than DiffusionGemma — the
diffusion decode loop is still future work. Status as of 2026-06-29:

1. ✅ **Compute smoke test** — Vulkan enum + alloc + coop-matrix matmul vs CPU.
2. ✅ **Core trait + tensor/graph** — `infr-core` Tensor/dtypes/Graph/Op/Backend
   (the seam; the as-built shape lives in `graph.rs` / `backend.rs`).
3. ✅ **`infr pull`** — HF + Ollama resolve/download, shared HF hub cache.
4. ✅ **Vulkan backend** — matmul (coopmat + dp4a/mmq), broad dequant coverage,
   rmsnorm, rope, softmax, flash + non-FA attention, fused FFN.
5. ✅ **GGUF loader** — parse + upload; tokenizer + embedded chat template.
6. ✅ **Forward pass** — full stack, validated vs llama.cpp (agreement harness).
7. ✅ **Attention + MoE** — GQA + SWA + full-attn; qwen3moe routing/gather.
8. ✅ **Diffusion decode** — canvas denoise loop (DiffusionGemma). CPU + Vulkan
   shipped; Metal code-complete but hardware-unvalidated. See
   [`docs/diffusion-gemma.md`](diffusion-gemma.md).
9. ✅ **`infr run`** — terminal chat (streaming, reasoning split).
10. ✅ **`infr serve`** — axum OpenAI server (streaming, tool-call bridge).
11. 🔄 **Perf pass** — ongoing (coopmat/dp4a tuning, record-once decode, KV
    layout); see [`docs/perf.md`](perf.md).
12. ✅ **Backend-agnostic refactor + second backend** — DONE. The op-list seam
    (`crates/infr-core`) runs on CPU (`infr-cpu`), Vulkan (`infr-vulkan`), and
    Metal (`infr-metal`); CUDA / MLX remain future backends behind the same
    `Backend` trait.

---

## Candidate models (next)

New model families to support, ≤30B, ranked by ROI for infr's current
architecture. First step for any of them is the same: pull the GGUF and diff its
arch string + tensor names against the backbone infr already runs — that tells
us "config tweak" vs "new op". Currently supported: llama, qwen2/2.5, qwen3,
qwen3moe, qwen35 (DeltaNet-hybrid), qwen35moe, gemma3, gemma4/E2B,
DiffusionGemma; MTP for qwen35.

1. **Ornith-1.0** (DeepReinforce.AI, MIT — 9B/31B dense, 35B MoE). Built on the
   **Gemma 4 + Qwen 3.5 backbones infr already runs**, so likely just config +
   tensor-name + chat-template work, no new kernels. Agentic-coding model →
   on-mission for the coding-agent north star. **Lowest effort, do first** (also
   validates the "our backbone just eats a fine-tune" hypothesis).
2. **DeepSeek-V2-Lite / DeepSeek-Coder-V2-Lite** (16B, 2.4B active MoE). Adds
   **MLA (Multi-head Latent Attention)** — the missing attention family behind
   the entire DeepSeek V2/V3/V4 line. New attention op (medium effort); the
   op-list seam is built for exactly this. Has a llama.cpp GGUF oracle for
   validation. **Highest strategic value.**
3. **GLM-4.7-Flash** (~30B MoE) or **Ernie 4.5 21B-A3B** (MoE). Mostly standard
   MoE FFN (infr's batched-expert path covers it) + minor arch quirks (GLM
   post-norm / partial RoPE). Config + loader effort. Both hyped, both ≤30B;
   some GLM variants ship native MTP heads.
4. **Nemotron-Nano / Nemotron-H** (Mamba2-Transformer hybrid MoE). Adds **Mamba2
   SSM**, extending infr's existing `Conv1dSilu` + `DeltaNet` linear-attention
   machinery — a real differentiator (llama.cpp's Mamba-hybrid GGUF path is
   weak). Med-high effort + a weaker oracle → do after MLA.

MTP extension that unlocks several of the above (esp. Ornith's Gemma4 variants):
gemma4 mem-shared MTP mode (see [`mtp.md`](mtp.md) "Later").

---

## Vulkan / GPU features to investigate

Leads surfaced from the 2026-07 RDNA4 coopmat exploration (RX 9060 XT / Navi 44,
Mesa 26.1.4). Context on what's already been ruled out: `docs/perf.md` "Coopmat
operand tiers" — on coopmat **v1**, no operand swap (fp8/bf16/int8) beats f16
for these GEMMs. The items below are NOT yet tried.

1. **`VK_NV_cooperative_matrix2` (coopmat2) — TESTED, negative (2026-07-09).**
   Was the promising lead: its `coopMatPerElementNV` applies a per-block rank-1
   (row·col) scale **in-fragment**, which should remove the store-to-shared
   "rescale tax" that dragged int8 coopmat to 0.73× vs f16 (v1 has no
   per-element access). Toolchain confirmed (glslc compiles
   `GL_NV_cooperative_matrix2`; correct signature
   `void coopMatPerElementNV(out coopmat r, coopmat m, T fn)`), feature present
   on RDNA4 via `radv_cooperative_matrix2_nv=true`. But the probe
   `crates/infr-vulkan/examples/coopmat2_test.rs` (per-element vs the v1
   store-to-shared round trip) measured coopmat2 SLOWER at BOTH scales on RDNA4
   (RX 9060 XT, Mesa 26.1.4):

   | bench                                                    | coopmat2  | v1 (store-to-shared) | ratio (v1/cm2)         |
   | -------------------------------------------------------- | --------- | -------------------- | ---------------------- |
   | single-tile 16×16×16                                     | 2.16 µs   | 2.24 µs              | 1.04 (noise, ~tie)     |
   | **full-GEMM 512×2048×2048** (64 rescale blocks, 4096 wg) | 1239.8 µs | 1195.3 µs            | **0.964 (cm2 slower)** |

   Even at full-GEMM scale — 64 accumulated per-block barriers + shared-memory
   occupancy pressure across 4096 workgroups — coopmat2 loses. The rescale tax
   is CHEAP on this hardware (fast LDS + cheap workgroup barrier — same reason
   removing 18 decode barriers bought only ~0.4%), so per-element access doesn't
   help, and RADV's newer coopmat2 codegen carries its own overhead. **Sharper
   insight:** removing the tax doesn't help → the tax was never why int8 coopmat
   lost to f16; the loss is structural elsewhere (the int8 WMMA / staging), same
   as fp8/bf16. So no coopmat-operand swap and no coopmat2 per-element trick
   beats f16 on RDNA4.

   **FURTHER RESEARCH (when coopmat2 stabilizes on RADV):** the current numbers
   are on RADV's _experimental_, driconf-gated coopmat2 path (Mesa 26.1.4) — its
   per-element codegen is new and unoptimized, and the extension is NV (RADV
   "limited support"). If a future Mesa promotes coopmat2 out of the driconf
   gate / optimizes the `OpCooperativeMatrixPerElementOpNV` lowering, **re-run
   the kept probe** —
   `radv_cooperative_matrix2_nv=true cargo run --release -p infr-vulkan --example coopmat2_test`
   — to see whether the full-GEMM ratio flips above 1.0. Only if it does is the
   real int8-coopmat2 GEMM worth building. See [[kernel-capability-tiering]],
   [[int8-coopmat-status]].

2. **`VK_NV_cooperative_vector`** — matrix×**vector** (i.e. decode / GEMV,
   infr's current perf frontier). **NOT exposed by RADV yet** (checked both
   default and driconf modes on Mesa 26.1.4). Watch for it in a future Mesa — if
   it lands, it could route the decode GEMVs through the matrix unit. Nothing to
   do until RADV ships it.
3. **bf16-coopmat rate on a future Mesa** — RDNA4's `bfloat16_t` WMMA currently
   runs ~12-27% slower than `float16_t` on the same kernel (see `docs/perf.md`);
   likely RADV codegen immaturity for the newer path. Re-check on a future Mesa:
   if it reaches f16 rate, native bf16 (already built, opt-in
   `INFR_BF16_COOPMAT`) becomes a free-accuracy default for bf16 models.

---

## Risks / open questions

- **Reusing ggml SPIR-V**: licensing (MIT — compatible) and binding layout/ABI
  must be matched, or we port the GLSL to our own descriptor conventions.
- **Quant matmul correctness** is the classic footgun — validate each quant type
  against a CPU dequant reference early.
- **Graph abstraction vs perf**: the compile-once/execute-many design must not
  force per-op sync; batch aggressively in the Vulkan backend.
- **Diffusion decode** is novel in Rust — no reference to copy; port carefully
  from the llama.cpp implementation (see References below).

---

## Reference implementations (read these first)

All paths are on the author's machine; treat them as the source of truth.

- **`~/Projects/llama.cpp`** — checked out on branch **`diffusiongemma`** (PR
  `ggml-org/llama.cpp#24423`), already built with HIP. Key files:
  - `src/llama-model.cpp` + `src/llama-arch.cpp` — the `diffusion-gemma`
    architecture graph (embeddings → 30 layers w/ SWA-vs-full attention
    selection, MoE FFN, RoPE, RMSNorm → output). **The canonical forward to
    port.**
  - `examples/diffusion/diffusion.{cpp,h}` — diffusion algorithms incl.
    `diffusion_generate_entropy_bound` and the algorithm enum. **Port for
    `infr-decode`.**
  - `examples/diffusion-gemma-server/diffusion-gemma-visual-server.cpp` — the
    patched persistent server (tools passthrough, FA-aware sizing, fixed ubatch,
    `CMOE`/`NCMOE` expert offload). Reference for diffusion params, context
    sizing, channel/tool handling.
  - `conversion/diffusion_gemma.py` — GGUF conversion: **authoritative tensor
    names + metadata keys.**
  - `ggml/src/ggml-vulkan/vulkan-shaders/*.comp` + `ggml/src/ggml-vulkan/` — the
    Vulkan compute shaders to reuse (`mul_mm`, per-quant dequant, etc.) and how
    descriptor sets / specialization constants are wired.
- **`~/Projects/scratch/dgemma-openai-server.py`** — the working OpenAI shim we
  built. Port its logic into `infr-engine`/`infr-server`: channel split
  (`reasoning_content` vs `content`), `<|tool_call>` → OpenAI `tool_calls`
  parsing, tools passthrough, SSE streaming.
- **Test model:**
  `~/Projects/models/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf`
  (16 GB, Q4_K_M). Also `unsloth/diffusiongemma-26B-A4B-it-GGUF` on HF, or via
  the Ollama store.
- **Validation oracle:** `~/Projects/llama.cpp/build/bin/llama-diffusion-cli`
  produces reference outputs/logits to diff against.

---

## DiffusionGemma spec (verified from the test GGUF)

Read these from GGUF metadata at runtime — do not hardcode — but here they are
so the implementer knows the shape up front.

- **arch** `diffusion-gemma`; **vocab** 262144; **hidden** (`embedding_length`)
  2816; **layers** (`block_count`) 30; **train ctx** 262144.
- **Attention:** `head_count` 16; `head_count_kv` per layer = `[8,8,8,8,8,2]`
  repeating → layers **5, 11, 17, 23, 29 are full-attention** (kv_heads 2), the
  other 25 are **sliding-window** (kv_heads 8); `key_length`=`value_length`=512
  (full), `*_swa`=256. This SWA split is why 256K KV is only ~a few GB.
- **MoE:** ~3.8B active of 26B; expert tensors `blk.N.ffn_(gate|up|down)_exps`,
  router/gate, top-k routing.
- **Diffusion:** `canvas_length` 256 tokens/block; `mask_token` id **4**;
  block-autoregressive (commit blocks sequentially). Entropy-bound decoder
  defaults observed: algorithm 4 (CONFIDENCE_BASED), schedule 0, `max_steps≈48`,
  `temperature` 0.8, `eps` 0.001, `t∈[0.4,0.8]`, `entropy_bound` 0.1,
  `stability` 1, `confidence` 0.005, self-conditioning on.
- **Special tokens / wire format** (the GGUF embeds the jinja `chat_template`;
  drive it with `minijinja` rather than reimplementing):
  - turns: `<|turn>role\n … <turn|>`, leading `bos`.
  - channels: `<|channel>thought\n … <channel|> … final`. Reasoning = before
    `<channel|>`, answer = after.
  - tool decl:
    `<|tool>declaration:NAME{description:<|"|>…<|"|>,parameters:{…}}<tool|>`
  - tool call: `<|tool_call>call:NAME{key:value,…}<tool_call|>` (strings wrapped
    `<|"|>…<|"|>`).
  - tool response: `<|tool_response>response:NAME{…}<tool_response|>`
- **Quants available:** Q4_K_M (target), Q5_K_M, Q6_K, Q8_0, BF16.

---

## Dependencies & toolchain

- **Rust** 1.96+ (edition 2021; 2024 ok). Workspace with the crates listed
  above.
- **Crates (proposed):** `ash` + `gpu-allocator` (Vulkan); `half`, `bytemuck`
  (dtypes); `memmap2` (GGUF mmap); `serde`/`serde_json`; `reqwest` (rustls) +
  `sha2` + `indicatif` (hub download); `tokio` + `axum` + `tower-http` (server);
  `clap` (CLI); `minijinja` (chat templates); `tokenizers` (or implement the
  GGUF-embedded tokenizer); `thiserror`/`anyhow`; `tracing`.
- **Shaders:** GLSL `.comp` → SPIR-V via the `shaderc` crate in `build.rs`,
  **or** reuse precompiled `.spv` from the ggml build. Device features to
  request: `VK_KHR_cooperative_matrix`, `shaderFloat16`, `VK_KHR_16bit_storage`,
  `VK_KHR_shader_subgroup_extended_types`; one compute queue.

---

## Validation strategy

- **smoke:** f16 cooperative-matrix matmul vs a CPU naive matmul; assert
  relative error < 1e-2.
- **quant:** dequantize a known tensor (Rust/shader) and diff vs a CPU
  reference; validate each quant type (Q4_K_M, Q8_0) before trusting matmuls.
- **per-layer:** fixed prompt → compare hidden states / final logits (top-k
  token probabilities) against `llama-diffusion-cli` on the same model.
- **end-to-end:** same prompt + fixed seed → compare generated text/structure to
  the llama.cpp oracle.

---

## Milestone acceptance criteria

| #   | Milestone    | Done when                                                            |
| --- | ------------ | -------------------------------------------------------------------- |
| 1   | smoke        | f16 coop-matrix matmul on the GPU matches CPU within tolerance       |
| 2   | core         | `Tensor`/`Graph`/`Op`/`Backend` compile; a 2-op graph runs on Vulkan |
| 3   | `infr pull`  | `hf:`/`ollama:` standalone HTTP pull into our own store; resumable   |
| 4   | vk backend   | matmul/dequant/rmsnorm/rope/softmax each validated vs CPU            |
| 5   | gguf load    | DiffusionGemma weights upload; tokenizer + template exposed          |
| 6   | forward      | final logits match llama.cpp top-k on a fixed prompt                 |
| 7   | attn + moe   | full multi-layer forward matches reference (SWA + full + MoE)        |
| 8   | diffusion    | greedy/fixed-seed generation matches llama.cpp text                  |
| 9   | `infr run`   | interactive terminal chat streams a coherent answer                  |
| 10  | `infr serve` | opencode / Claude Code CLI complete an agentic tool turn             |
