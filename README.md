# INFR — AMD/Vulkan MoE Optimization Fork

> **Experimental fork of [kryptic-sh/infr](https://github.com/kryptic-sh/infr) focused on large-MoE inference, expert caching/paging, long-context performance, and heterogeneous memory execution on consumer AMD GPUs.**

This branch explores how far large Mixture-of-Experts models can be pushed on a relatively constrained consumer system:

* **GPU:** AMD Radeon RX 7900 XTX, 24 GB VRAM
* **Backend:** Vulkan / RDNA3
* **Host memory:** 64 GB DDR4
* **CPU:** Ryzen 5 5600X
* **Storage:** SSD-backed model/expert storage for models exceeding RAM capacity

The current snapshot is:

**`v0.1-moe-snapshot`**

This is a work in progress. The current code is being frozen and benchmarked before further architectural changes.

---

## Why this fork exists

Large MoE models have a very different memory behavior from conventional dense models: total model size can be far larger than available VRAM, while only a small subset of experts is active for each token.

The main focus of this fork is therefore not simply kernel throughput, but the complete expert-serving path:

```text
SSD
 │
 ▼
Host expert store / RAM cache
 │
 ▼
GPU expert residency / LRU cache
 │
 ▼
Vulkan execution
```

The work has focused on reducing the cost of moving, caching, selecting, and executing experts while overlapping as much data movement as possible with GPU computation.

A particular goal is to make large MoE inference practical on systems where **neither VRAM nor system RAM is large enough to hold the complete model comfortably**.

---

## Current results

These numbers are representative of the current `v0.1-moe-snapshot` on a single RX 7900 XTX.

They should be treated as **preliminary engineering results**, not yet as a fully standardized cross-runtime benchmark. A reproducible benchmark suite with exact model quantization, context length, cache size, command line, and repeated-run statistics is being prepared.

| Model                 |        Decode |                                                                      Prefill | Status                                                     |
| --------------------- | ------------: | ---------------------------------------------------------------------------: | ---------------------------------------------------------- |
| **Qwen3.6-35B-A3B**   | **~40 tok/s** | **~400+ tok/s at very deep context; up to ~3,000+ tok/s at shallow context** | Decode and prefill extensively optimized                   |
| **Qwen3.5-122B-A10B** | **~23 tok/s** |                                                                          WIP | Decode optimized; SSD-aware prefill pipeline pending       |
| **Ling 3.0 Flash**    | **~36 tok/s** |                                                                          WIP | Decode path working well                                   |
| **DeepSeek V4 Flash** |  **~4 tok/s** |                                                                          WIP | Early large-model result; substantial optimization remains |

For Qwen3.6-35B-A3B, one repeatable deep-context configuration reaches approximately:

* **~39 tok/s decode around 200K context**
* **~409–437 tok/s prefill around 250K synthetic KV depth**, depending on microbatch
* **~3,000–3,700 tok/s shallow-context prefill**, depending on batch/configuration

The purpose of the upcoming benchmark pass is to make these results directly reproducible instead of relying on isolated optimization measurements.

---

## Main optimization work

### 1. GPU expert residency and LRU caching

The decode path uses expert-level GPU residency rather than requiring complete MoE layers to remain permanently resident.

Work in this area includes:

* expert-level LRU caching
* O(1) hit promotion
* cache victim-selection optimization
* role / expert-size-aware cache pools
* reduced cache-management overhead
* profiling of hit, miss, upload, wait, and eviction behavior

The same GPU memory arena can be reused differently between prefill and decode rather than maintaining completely independent large allocations.

---

### 2. Host expert store

Expert weights are organized in a host-side representation designed around the access patterns of MoE inference.

This allows the runtime to separate:

* permanent GPU residency
* streamed experts/layers
* host-resident experts
* models that ultimately require SSD-backed storage

This becomes increasingly important for 100B+ and DeepSeek-class models where a 64 GB host cannot simply pin the complete expert bank.

---

### 3. Prefill layer streaming

For Qwen3.6-35B-A3B, prefill uses a different strategy from decode.

GPU memory is divided between:

* fixed resident layers
* **A/B whole-layer streaming lanes**

The next MoE layer can be transferred while the GPU is still executing the current layer.

On a representative 200K-context configuration:

* 8 / 40 MoE layers are resident
* 32 layers are streamed
* roughly 17.6 GiB is moved during the measured path
* transfers begin concurrently with live GPU work

This pipeline is one of the main reasons Qwen3.6-35B prefill performs substantially better than the current 100B+ model path.

---

### 4. Direct CPU → GPU-visible memory path

On Windows/RDNA3, this fork experiments with persistently mapped GPU-visible memory / ReBAR-backed pools.

For supported paths, expert or layer data can be written directly by the CPU into its final GPU-visible location instead of always using a conventional:

```text
CPU buffer
    ↓
staging buffer
    ↓
GPU copy
    ↓
final expert allocation
```

The goal is to reduce small-transfer and synchronization overhead in latency-sensitive expert paging.

Windows AMD allocation constraints require the logical cache to be divided into multiple physical mapped pools rather than using one very large mapped allocation.

---

### 5. Long-context attention and Q8 KV work

The fork also contains substantial long-context work independent of MoE paging itself, including:

* specialized FlashAttention prefill paths
* long-context attention tuning
* dedicated Q8 KV decode handling
* reduced transient Vulkan resource allocation
* command submission / synchronization optimization

At sufficiently long context lengths, attention increasingly dominates total runtime, so expert paging alone no longer determines throughput.

---

### 6. Large-model decode

The current snapshot can already run substantially larger models than the 35B-class configuration on the same 24 GB GPU / 64 GB host system.

Current experimental targets include:

* Qwen 122B-class MoE
* Ling 3.0 Flash
* DeepSeek V4 Flash

Decode has received significantly more optimization than prefill for these models.

**Large-model prefill is currently limited by SSD → RAM expert loading.**

A deeper SSD-aware prefetch / streaming pipeline is one of the next major pieces of work.

---

## Current limitations / open problems

This snapshot intentionally freezes several unresolved problems so that later improvements can be measured against a stable baseline.

### SSD-aware large-model prefill

The 35B model can keep enough of its working set in RAM for an efficient layer-streaming pipeline.

For significantly larger models this is no longer true.

The next step is a hierarchical prefetch system closer to:

```text
SSD
 ↓
RAM working-set cache
 ↓
GPU expert/layer cache
 ↓
Compute
```

with asynchronous prediction and transfer across all three storage tiers.

### Miss rate is not the whole story

Current work is also investigating whether aggregate expert-cache hit rate is sufficient to predict performance.

Two workloads with the same average miss rate may behave very differently if one produces:

* evenly distributed misses

while another produces:

* per-layer miss bursts
* consecutive-token miss clusters
* synchronized cache pressure

The effect of this miss topology on decode critical-path latency is still being investigated.

### Vulkan scheduling overhead

Without CUDA Graph-style execution capture, dynamic MoE residency and heterogeneous execution can expose host-side command submission and synchronization overhead.

Reducing this overhead further on Vulkan is another active area of work.

---

## Relation to FreeToken

This work was developed independently while investigating large-MoE inference on AMD RDNA3/Vulkan.

After the initial architecture and optimization work was already underway, I came across **[FreeToken](https://github.com/FlashML-org/FreeToken)**, whose authors independently explore several closely related system ideas for edge MoE serving, including:

* GPU expert caching
* CPU/GPU heterogeneous execution
* expert movement across the PCIe boundary
* prefill streaming
* memory-aware MoE scheduling

The two projects currently operate in substantially different environments.

**FreeToken:**

```text
NVIDIA
CUDA
large host-memory configurations
```

**This fork:**

```text
AMD RDNA3
Vulkan
24 GB VRAM
64 GB DDR4
SSD-backed large-model experiments
```

I am particularly interested in comparing how expert-cache behavior, miss handling, host-memory limitations, and dynamic scheduling differ between the CUDA/NVIDIA and Vulkan/AMD stacks.

This repository is **not a fork of FreeToken** and does not use its implementation.

---

## Development methodology

Development of this experimental branch is **heavily AI-assisted**.

My primary work on the project is:

* system architecture
* bottleneck identification
* profiling and measurement
* optimization strategy
* experimental design
* implementation direction
* correctness/performance validation
* iterative architectural decisions

Coding agents are used extensively for Rust/Vulkan implementation and code modification.

This distinction is stated explicitly because the purpose of this project is primarily to explore **system architecture and AI-assisted systems engineering**, rather than to present the repository as a hand-written kernel/runtime implementation by a single developer.

---

## Benchmarking roadmap

The next benchmark snapshot will standardize:

* exact model files and quantization
* context / synthetic KV depth
* KV-cache format
* expert-cache capacity
* RAM / VRAM utilization
* warm-up procedure
* repeated runs
* median throughput
* expert hit/miss statistics
* raw logs
* reproduction commands

Planned benchmark models:

1. Qwen3.6-35B-A3B
2. Qwen3.5-122B-A10B
3. Ling 3.0 Flash
4. DeepSeek V4 Flash

Where practical, workload-compatible measurements with FreeToken will also be reported, with hardware and precision differences made explicit.

---

## Upstream

This repository is based on **[kryptic-sh/infr](https://github.com/kryptic-sh/infr)**, a Pure-Rust, Vulkan-first LLM inference engine.

The original project provides the fundamental runtime, model support, Vulkan backend, and much of the infrastructure on which this experimental branch is built.

**The original INFR README and usage documentation continue below.**

---

# Original INFR README


# infr

[![CI](https://github.com/kryptic-sh/infr/actions/workflows/ci.yml/badge.svg)](https://github.com/kryptic-sh/infr/actions/workflows/ci.yml)

Pure-Rust LLM inference engine. Vulkan-first, built to run on any mainstream
GPU.

> Early WIP. The only non-Rust parts are the GPU driver calls (Vulkan via `ash`)
> and the compute shaders (SPIR-V).

## Goal

A from-the-metal inference server that works across AMD / NVIDIA / Intel
(Vulkan) and Apple (native Metal), plus a CPU reference — three backends behind
one `Backend` trait.

## Status

Runs **Llama / Qwen2 / Qwen3** (dense), **Gemma 3** (dense, sliding-window
attention + QK-norm + GeGLU), and **Gemma 4** (per-layer heterogeneous head
dims, proportional RoPE, V-norm, per-layer output scale — including the **E2B**
variant: per-layer input embeddings, per-layer FFN widths, KV-layer sharing) on
the Vulkan backend, competitive with llama.cpp at long context (`infr compare`).
**Qwen3.5 / Qwen3.6** (`qwen35` — hybrid gated-DeltaNet + attention, a sibling
of Qwen3-Next) run on the same unified runner, CPU + Vulkan (`docs/qwen35.md`).
**DiffusionGemma** (the original target — block text-diffusion MoE on a Gemma-4
backbone, entropy-bound denoise decode) runs end-to-end on CPU + Vulkan
(`docs/diffusion-gemma.md`).

```bash
infr pull   <model-ref>        # org/repo[:quant] (HuggingFace) | path to a .gguf
infr run    <model-ref> [msg]  # terminal chat (auto-pulls)
infr serve  <model-ref>        # OpenAI-compatible HTTP API
infr serve-embedding <gguf>    # OpenAI-compatible /v1/embeddings (llama.cpp worker)
infr bench / infr compare      # tok/s benchmarks vs llama.cpp
```

Model refs match llama.cpp's `-hf`: `org/repo[:quant]` (quant default `Q4_K_M`,
e.g. `infr run unsloth/Qwen3-14B-GGUF:Q4_K_M`). Models share the standard
**HuggingFace Hub cache** (`~/.cache/huggingface/hub`) with llama.cpp and
`huggingface_hub` — one download, used by both.

## Supported models

All run on the Vulkan GPU backend unless noted. The chat template (turn markers,
system prompt) is read from the GGUF's own `tokenizer.chat_template`.

| Family            | Arch (GGUF)       | Notes                                                   |
| ----------------- | ----------------- | ------------------------------------------------------- |
| Llama             | `llama`           | dense transformer                                       |
| Llama 4           | `llama4`          | sigmoid top-1 MoE + shared expert, iRoPE, paged experts |
| Qwen2 / Qwen2.5   | `qwen2`           | dense, QKV bias, NEOX rope                              |
| Qwen3             | `qwen3`           | dense, QK-norm                                          |
| Qwen3 MoE         | `qwen3moe`        | softmax router, top-_k_ experts, paged experts          |
| Gemma 3           | `gemma3`          | SWA + QK-norm + GeGLU, dual-RoPE                        |
| Gemma 4 (dense)   | `gemma4`          | per-layer head dims, proportional RoPE, V-norm          |
| Gemma 4 **E2B**   | `gemma4`          | + per-layer input embeddings / FFN, KV sharing          |
| Gemma 4 **MoE**   | `gemma4`          | 26B-A4B: dual FFN (dense GeGLU ∥ 8-of-128 routed), AR   |
| Qwen3.5 / Qwen3.6 | `qwen35`          | hybrid gated-DeltaNet + attention (NOT `qwen3next`)     |
| Qwen3.6 MoE       | `qwen35moe`       | `qwen35` skeleton + routed experts + shared expert      |
| DiffusionGemma    | `diffusion-gemma` | block text-diffusion MoE, entropy-bound denoise decode  |

Fine-tunes on any of these backbones run unchanged. **Ornith-1.0**
(DeepReinforce.AI agentic-coding) validated 2026-07-09 — the 9B rides `qwen35`
and the 35B rides `qwen35moe` with no code changes
(`infr run deepreinforce-ai/Ornith-1.0-9B-GGUF:Q4_K_M "..."`).
**Ternary-Bonsai** (Prism ML, weights trained to {-1, 0, +1}) validated
2026-07-12 — the 1.7B / 4B / 8B all ride `qwen3`, zero-code, both in the TQ2_0
repack (`superkaiii/Ternary-Bonsai-4B-GGUF`) and in llama.cpp's new **Q2_0**
weight dtype (2.25 bpw, GGML type 42 — native in-shader dequant + dp4a mmq, no
fork needed). infr is the **only engine that runs Q2_0 on a GPU** (llama.cpp
merged the dtype CPU-only) — numbers in
[`docs/perf/results.md`](docs/perf/results.md). Pull the `Q2_0_g64` files:
`infr run prism-ml/Ternary-Bonsai-8B-gguf:Q2_0_g64 "..."`.

```bash
# Qwen3 dense
infr run unsloth/Qwen3-1.7B-GGUF:Q4_K_M "What is the capital of France?"

# Qwen3 MoE (experts page through the VRAM LRU cache when they don't fit —
# see docs/config.md)
infr run unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M "Explain MoE routing."

# Llama 4 Scout (37 GB Q2_K) — paged expert cache runs it on a 24 GB card
infr run unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF:Q2_K "What is the capital of France?"

# Gemma 3
infr run unsloth/gemma-3-1b-it-GGUF:Q4_K_M "What is bash?"

# Gemma 4 — dense and the E2B variant
infr run unsloth/gemma-4-12b-it-GGUF:Q4_K_M  "What is the capital of France?"
infr run unsloth/gemma-4-E2B-it-GGUF:Q4_K_M  "What is bash?"

# DiffusionGemma — block text-diffusion decode (entropy-bound denoise)
infr run unsloth/diffusiongemma-26B-A4B-it-GGUF:Q4_K_M  "What is the capital of France?"

# Pick a specific quant with the `:quant` suffix (default is Q4_K_M)
infr run unsloth/Qwen3-8B-GGUF:Q6_K       "Summarize the plot of Hamlet."
infr run unsloth/Qwen3-0.6B-GGUF:IQ4_XS   "Write a haiku about Rust."

# MTP speculative decoding is currently DISABLED (rationale in docs/mtp.md).
# INFR_MTP=1 is ignored with a warning; MTP-head models run the ordinary decode
# path (their `nextn` tensors are simply unused) and are otherwise fully supported.
infr run unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_XL "Explain how a hash map works."

# Sampling defaults to the model's own recommended values; override per run:
infr run unsloth/Qwen3-1.7B-GGUF:Q4_K_M "Tell me a story." \
  --temp 0.7 --top-k 40 --top-p 0.95
```

## Configuration

Everything the engine can be told — device, context, sampling, KV format, paging
budgets, every kernel-tier switch — is one typed value resolved once at startup
from **four layers, later wins**:

```
defaults  <  config file (TOML)  <  INFR_* environment  <  CLI flags / --set
```

The config file is the **first existing** of `--config <PATH>` (an error if that
path does not exist), `./infr.toml`, then `$XDG_CONFIG_HOME/infr/config.toml`
(else `~/.config/infr/config.toml`). First match wins — there is no merging
across files, and finding no file is a no-op.

```toml
# ./infr.toml — see infr.example.toml for a commented starting point
[device]
ctx = "32k"

[kv]
type_k = "q8_0"

[kernels.vulkan]
flash_splits = 2
gemm_warp = false     # the file speaks the POSITIVE field names
```

**Every documented `INFR_*` variable still works** — nothing was renamed; the
variables now feed the same resolved value the file and the flags do. Knobs
without a dedicated flag are reachable with `--set <config.path>=<value>`, which
takes the same paths as the file:

```bash
infr bench "$M" -p 512 -n 0 --set kernels.vulkan.flash_splits=2
```

Where a bespoke flag and a `--set` name the same field, the flag wins and says
so (`--ctx 4096 --set device.ctx=8192` runs at 4096 and prints a warning).

Full reference — the per-section walkthrough, `--set` semantics, the unknown-key
behaviour, and the handful of `INFR_*` keys that are deliberately not
configuration — is in [`docs/config.md`](docs/config.md).

### Serving

```bash
# OpenAI-compatible HTTP API (streaming). Reuses a persistent KV cache across
# requests (common-prefix diff) for fast TTFT on shared-prefix chats.
infr serve unsloth/Qwen3-14B-GGUF:Q4_K_M          # default: 127.0.0.1:8080

curl -s localhost:8080/v1/chat/completions -d '{
  "model": "qwen3",
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "stream": true
}'
```

Embedding models use the mature llama.cpp implementation while INFR owns the public API,
authentication, admission control, process lifecycle, and resource accounting:

```bash
infr serve-embedding nomic-embed-text-v1.5.f16.gguf --dev Vulkan0
# Or host chat + embedding on one INFR endpoint:
infr serve chat-model.gguf --embedding-model nomic-embed-text-v1.5.f16.gguf
```

Works as a drop-in backend for OpenAI-API clients (opencode, the Claude Code
CLI, etc.). Tool calling renders the model's own `tokenizer.chat_template`
(Qwen, Llama-3.x, Gemma tool dialects supported).

`--temp` / `--top-k` / `--top-p` set the SERVER defaults (`--temp 0` = greedy);
a per-request OpenAI `temperature`/`top_p` still overrides them. See
[Configuration](#configuration).

On Windows, `Start-INFR-GUI.cmd` builds and opens the server-hosted browser control plane on port
8180. It manages model directories, downloads, profiles, memory estimates, and a supervised
`infr serve` worker. See [`crates/infr-gui/README.md`](crates/infr-gui/README.md).

## Performance

Measured against llama.cpp on an **AMD Radeon RX 7900 XTX** (RDNA3, Vulkan /
RADV), every validated model × quant, both engines on matched flags. Headline:

- **Decode** — the reproducible half — wins **29 of 35** rows at `tg128` and
  **24 of 35** at `tg64@d4096`.
- **`pp4@d4096`** (multi-turn ingest, the shape a coding agent actually runs) is
  the strongest column, roughly **1.5–2×** on the small and mid models.
- Losses concentrate on **Qwen3-14B and the larger MoEs**, mostly at depth.

**The full table, the per-row footnotes, and an honest account of where infr
loses are in [`docs/perf/results.md`](docs/perf/results.md).** Two caveats live
there and both matter: ratios move with _both_ engines, so snapshots taken
against different `llama-bench` builds are not comparable; and infr's
**prefill** columns vary up to ~30% run-to-run on an identical binary (a known
tier/chunk nondeterminism), so quote prefill to one significant figure and
decode as written.

To reproduce or extend the numbers — `infr bench` / `infr compare --sweep`
flag-for-flag against `llama-bench`, plus per-op GPU profiling — see
[`docs/perf/benchmarking.md`](docs/perf/benchmarking.md). The optimization
method and the recorded dead ends are in
[`docs/perf/playbook.md`](docs/perf/playbook.md); everything performance-related
is indexed at [`docs/perf/`](docs/perf/README.md).

> MTP self-speculative decode is currently **parked** — `INFR_MTP=1` is ignored
> with a warning and MTP-head GGUFs run the ordinary decode path. Rationale in
> [`docs/mtp.md`](docs/mtp.md).

## Scope

- **Format:** GGUF
- **Models:** Llama, Qwen2/2.5, Qwen3 (dense + MoE), Gemma 3, Gemma 4 (dense +
  E2B + 26B-A4B MoE), Qwen3.5/3.6 (dense + MoE) — all on GPU **and** the CPU
  reference; DiffusionGemma (block text-diffusion, CPU + GPU); Llama 4 (Scout —
  GPU by default via the paged expert cache, 37 GB Q2_K on a 24 GB card; pure
  CPU under `--dev cpu`)
- **GPU:** AMD / NVIDIA / Intel via Vulkan (cooperative-matrix matmul); Apple
  via a native **Metal backend** (`--dev metal`) covering every op the CPU
  reference does — dense, MoE (`qwen3moe`) and Qwen3.5 (`qwen35`). Dense is
  optimized (simdgroup-matrix GEMM + flash attention, raw-block quant decode;
  within ~1.3-1.5× of llama.cpp Metal on M3 Pro — architecture and numbers in
  [`docs/metal.md`](docs/metal.md))
- **Store:** the shared **HuggingFace Hub cache** — located via `$HF_HUB_CACHE`,
  else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`, in HF's own
  `models--<org>--<repo>/{blobs,snapshots,refs}` layout. A model pulled by
  `infr`, `llama.cpp`, or `huggingface_hub` is shared — downloaded once.
  `infr pull` fetches from `huggingface.co` over resumable HTTP Range with a
  progress bar; gated repos authenticate with `HF_TOKEN`.
- **API:** OpenAI-compatible HTTP (streaming) — works with opencode / Claude
  Code CLI

## Architecture

```
server   axum + SSE  ->  OpenAI /v1
chat     ChatModel        (autoregressive dense/MoE/qwen35; DiffusionGemma's block-diffusion loop)
runtime  SeamModel        tensors, KV cache, command/descriptor management (the unified runner)
loader   WeightSource     (Gguf; safetensors later)
compute  Backend          (Vulkan via ash + SPIR-V; native Metal via MSL; CPU reference)
```

## Documentation

Deeper design docs, backend architecture, and performance material live in
[`docs/`](docs/README.md) — start with that index. Highlights:
[`docs/perf/`](docs/perf/README.md) (all performance: results, benchmarking,
optimization playbook, kernel coverage), [`docs/config.md`](docs/config.md) (the
configuration reference), [`docs/metal.md`](docs/metal.md) and
[`docs/igpu.md`](docs/igpu.md) (backends).

## License

[MIT](LICENSE)
