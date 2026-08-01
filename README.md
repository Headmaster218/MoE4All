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

Works as a drop-in backend for OpenAI-API clients (opencode, the Claude Code
CLI, etc.). Tool calling renders the model's own `tokenizer.chat_template`
(Qwen, Llama-3.x, Gemma tool dialects supported).

`--temp` / `--top-k` / `--top-p` set the SERVER defaults (`--temp 0` = greedy);
a per-request OpenAI `temperature`/`top_p` still overrides them. See
[Configuration](#configuration).

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
