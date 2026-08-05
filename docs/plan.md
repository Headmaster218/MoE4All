# infr — project plan

> **How to read this.** Two halves. Everything under
> [What shipped](#what-shipped) down to
> [Product surface](#product-surface-the-infr-cli) is a **description of the
> tree as it stands on 2026-08-05** — kept here because no other doc carries the
> whole-system shape in one place. Everything from
> [Adding a model architecture](#adding-a-model-architecture-the-recipe) on is
> **forward-looking work**, written to be executable without re-deriving the
> codebase. The [historical record](#historical-the-original-milestones) of the
> original DiffusionGemma-era milestones is at the bottom.
>
> Current state of anything performance-related is NOT here — it is
> [`docs/perf/`](perf/README.md). The doc index is
> [`docs/README.md`](README.md).

Pure-Rust LLM inference engine. Vulkan-first, designed to run on any mainstream
GPU. The only non-Rust surface is the GPU driver (called through thin Rust FFI)
and the compute shaders (SPIR-V / MSL).

---

## Vision

A from-the-metal inference server where **the server and model code never know
which GPU API is running underneath**. Vulkan first (covers AMD/NVIDIA/Intel),
then native Metal, **without touching any layer above the backend**.

The architecture is organized around four pluggable seams so that "add a GPU",
"add a model", "add a format", or "add a decode style" each means _implementing
one trait_, never refactoring the stack.

---

## What shipped

The original MVP targeted DiffusionGemma over Vulkan. It shipped — but the
autoregressive families landed first and are now the primary path.

| Dimension    | Original MVP                       | As built (2026-08-05)                                                                                        |
| ------------ | ---------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Format       | GGUF                               | GGUF (safetensors still unbuilt)                                                                             |
| Model source | HF + Ollama pull, or a local path  | **HF only** — `org/repo[:quant]`, or a local path. The Ollama registry client was dropped                    |
| Store        | infr's own OCI-style store         | the **shared HF Hub cache** (`$HF_HUB_CACHE` > `$HF_HOME/hub` > `~/.cache/huggingface/hub`) — see `infr-hub` |
| Model        | DiffusionGemma                     | every arch string listed below, DiffusionGemma among them                                                    |
| GPU backend  | Vulkan on AMD (RADV)               | Vulkan (AMD/NVIDIA/Intel, incl. iGPU/UMA), native **Metal**, and a **CPU reference** — one `Backend` trait   |
| Decode       | diffusion (block denoise)          | autoregressive **and** block-diffusion; MTP speculative is built but **parked** ([`mtp.md`](mtp.md))         |
| API          | OpenAI-compatible HTTP (streaming) | same, plus a persistent KV prefix cache and multi-slot serve                                                 |
| Perf         | "correct and usable"               | benchmarked against llama.cpp per model × quant — [`perf/results.md`](perf/results.md)                       |

**Architectures accepted today** — what actually decides acceptance is the match
in `Config::from_gguf`; `arch::ALL` in `crates/infr-llama/src/arch.rs` lists the
same set, and the doc comment on each const says what makes that family
different:

`llama`, `llama4`, `qwen2`, `qwen3`, `qwen3moe`, `gemma3`, `gemma4` (dense / E2B
/ 26B-A4B MoE on the one string), `qwen35`, `qwen35moe`, `diffusion-gemma`,
`bitnet`, `bitnet-b1.58`.

Fine-tunes on those backbones run with no code change (Ornith-1.0 and
Ternary-Bonsai are the validated examples — see the root
[`README.md`](../README.md)).

---

## Architecture

Bottom-up. Each named trait is the seam where future variants plug in.

```
┌──────────────────────────────────────────────────────────────────────┐
│ infr-cli      pull / devices / run / serve / multi / bench / compare  │
├──────────────────────────────────────────────────────────────────────┤
│ infr-server   axum + SSE -> OpenAI /v1/chat/completions, /v1/models   │  knows NOTHING about the GPU
├──────────────────────────────────────────────────────────────────────┤
│ infr-chat     chat templating (the GGUF's own jinja), tool-call bridge│
├──────────────────────────────────────────────────────────────────────┤
│ infr-llama    Config + the unified seam runner: builds the Graph,     │
│               loads weights, steps KV; autoregressive AND diffusion   │
├──────────────────────────────────────────────────────────────────────┤
│ infr-gguf     trait WeightSource -> Gguf (safetensors later)          │
├──────────────────────────────────────────────────────────────────────┤
│ infr-core     Tensor, dtypes/quant, Graph, Op, trait Backend, pagers  │
├──────────────────────────────────────────────────────────────────────┤
│ backends      infr-cpu (reference) / infr-vulkan / infr-metal         │  the ONLY GPU-aware layer
├──────────────────────────────────────────────────────────────────────┤
│ shaders       GLSL -> SPIR-V (Vulkan), MSL (Metal)                    │  not Rust
└──────────────────────────────────────────────────────────────────────┘
```

Dependency rule: **everything above the backends is generic over `Backend`.**
The server depends on the engine, which holds a backend and otherwise treats it
as opaque.

### The backend seam

Drawn at the level of **semantic tensor ops**: the model builds an ordered
op-list (`Graph`) over typed tensor handles; each backend compiles + executes it
however it likes. The as-built trait, op set, and the dtype-awareness decision
live in `crates/infr-core/src/graph.rs` (`Op`) and
`crates/infr-core/src/backend.rs` (`Backend::compile` / `execute` /
`execute_chain`) — read those, not a paraphrase.

---

## Crate layout

The workspace members, from the root `Cargo.toml`:

```
infr/
├── crates/
│   ├── infr-core       # Tensor, dtypes/quant, Graph, Op, Backend trait, config, pagers, errors
│   ├── infr-cpu        # Backend impl: the reference backend (also the correctness oracle)
│   ├── infr-vulkan     # Backend impl: ash + gpu-allocator + SPIR-V dispatch, VRAM/expert pagers
│   ├── infr-metal      # Backend impl: native MSL (Apple GPUs)
│   ├── infr-gguf       # WeightSource impl: GGUF parse + metadata + tensor mapping (+ `examples/dump.rs`)
│   ├── infr-hub        # model resolve + download into the shared HuggingFace Hub cache
│   ├── infr-llama      # Config, arch table, the unified seam runner, tokenizer, sampling, diffusion
│   ├── infr-chat       # chat templating + the tool-call dialects
│   ├── infr-engine     # load pipeline + session orchestration
│   ├── infr-server     # axum OpenAI-compatible HTTP + SSE
│   ├── infr-cli        # the `infr` binary (clap subcommands)
│   ├── infr-prof       # per-op profiling front-end
│   ├── infr-prof-rt    # profiling runtime hooks
│   └── infr-testkit    # dev-dependency only: shared parity/case harness for backend tests
├── docs/               # this index: docs/README.md
├── ref/                # vendored reference C++ sources, read-only
└── scripts/            # perf sweep + profiling helpers
```

---

## Product surface (the `infr` CLI)

```bash
infr pull    <model-ref>        # download + cache (HF, or a path to a .gguf)
infr devices                    # list the Vulkan devices, marking the default
infr run     <model-ref> [msg]  # interactive terminal chat (auto-pulls)
infr serve   <model-ref>        # OpenAI-compatible HTTP API (streaming)
infr multi   <spec>…            # data-parallel: MODEL[@VulkanN] per GPU, Vulkan only
infr bench / infr compare       # tok/s benchmarks, and A/B against llama.cpp
```

**Model refs** match llama.cpp's `-hf`: `org/repo[:quant]`, quant
case-insensitive (`Q4_K_M` by default) or an explicit `*.gguf` filename; a bare
`hf:`/`huggingface:` prefix is accepted but optional; anything path-shaped is a
path. Parsing lives in `infr_hub::model_ref::ModelRef::parse`.

**Store:** the standard HuggingFace Hub cache, in HF's own
`models--<org>--<repo>/{blobs,snapshots,refs}` layout, so a model pulled by
`infr`, `llama.cpp` or `huggingface_hub` is downloaded once and shared. Located
by `infr_hub::store` (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else
`~/.cache/huggingface/hub`) — `hub/` specifically, not the
`~/.cache/huggingface` root, whose other subdirectories (`xet/`, HF's chunk
store) belong to `huggingface_hub` and are never read or written by infr.
Resumable HTTP Range with a progress bar; gated repos authenticate with
`HF_TOKEN`.

Configuration (four layers, later wins: defaults < TOML file < `INFR_*` env <
CLI flags/`--set`) is documented in [`config.md`](config.md).

---

## Adding a model architecture (the recipe)

This is the procedure every family since Qwen2 has followed. There is **no
plugin system**: an architecture is a set of fields on one `Config`, branched on
inside one weight-load loop and one graph builder. Work the steps in order — the
early ones are cheap and decide how much of the later ones you need.

**The rule that makes the rest cheap: make the CPU reference backend correct
first, and only then look at a GPU.** A wrong graph and a wrong shader look
identical from the top, and only one of them is cheap to debug.

### 0. Get the file and read what is actually in it

```bash
infr pull <org/repo>:Q4_K_M
cargo run -p infr-gguf --example dump -- ~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/*/<file>.gguf
```

That prints every metadata KV and the tensor directory (name, dtype, shape). Two
things come out of it: the exact `general.architecture` string, and the
tensor-name set.

### 1. Diff against a family infr already runs

Dump a GGUF of the closest supported family and diff the two tensor-name lists
and metadata keys. Then read the reference builder in llama.cpp
(`~/Projects/mxaddict/llama.cpp`): `src/llama-arch.cpp` for the arch's tensor
table and metadata keys, `src/models/<arch>.cpp` for its forward pass. **That
file is the specification** — every claim you make about the maths should come
out of it, and anything you did not read there should say so.

Classify the delta before writing code:

- **only metadata differs** → step 3 alone (a config branch).
- **tensor names/shapes differ, same maths** → steps 3–4.
- **the forward pass has a step no existing `Op` performs** → steps 3–5, and
  budget for the op landing on all three backends.

### 2. Decide whether it is worth it, and write that down

If the family has no model small enough to develop against on the local card,
say so in the plan and stage the work so the pieces are independently testable
before the untestable stage starts. [`deepseek.md`](deepseek.md) is the worked
example of this shape.

### 3. Declare the arch and parse it

- `crates/infr-llama/src/arch.rs` — add the `pub const`, with a doc comment
  saying what the family does differently (that is where the next reader looks).
  Add it to `TRANSFORMER` when it goes through the shared transformer path, and
  to `ALL`. Neither list gates loading — `TRANSFORMER` is read once to render
  the rejection message and `ALL` has no consumer — so keep them honest for the
  next reader, but do not expect either to make the arch work.
- `Config::from_gguf` in `crates/infr-llama/src/config.rs` — an unrecognised
  arch bails out of the `qk_norm` match at the top of that function, so add the
  arm there first, then the per-arch booleans below it (the existing `llama4` /
  `gemma4` / `qwen35_moe` flags are the pattern to copy). Metadata keys are
  namespaced by the arch string itself (`{arch}.block_count`), which the local
  `mk` closure already handles.

Never invent an arch string, and never rename metadata keys to be tidier: both
must match what the converter wrote into the file.

### 4. Load the weights

`crates/infr-llama/src/seam/weights.rs` holds the per-layer handle structs
(`LayerW`, `FfnW`, …); the `wload` closure in
`crates/infr-llama/src/seam/runner.rs` is what actually uploads each block. A
new tensor means a handle field plus a load site, gated on the config flag from
step 3 — not an unconditional load, which breaks every other family.

### 5. Build the graph (and, if needed, a new op)

The layer graph is assembled in `crates/infr-llama/src/seam/runner.rs`. Prefer
composing existing `Op`s. If the family genuinely needs a new one:

1. add the variant to `Op` in `crates/infr-core/src/graph.rs`, and make sure its
   `io()` names **every** buffer the body reads — the CPU backend panics on a
   read the `io()` did not declare, which is the check that catches this.
2. implement it in `infr-cpu` first;
3. add a parity case to `crates/infr-llama/tests/seam_op_parity.rs`;
4. implement it in `infr-vulkan` and `infr-metal`, each verified against the CPU
   result through that same parity test.

A backend left unimplemented must fail loudly, not silently skip: "not supported
here" arriving as "passed" is the failure mode this seam is most prone to.

### 6. Tokenizer and chat

Most families need nothing here — the tokenizer and the jinja chat template come
out of the GGUF. When they do:

- `crates/infr-llama/src/tokenizer.rs` — the `tokenizer.ggml.model` branch picks
  SPM vs BPE, and a `tokenizer.ggml.pre` value maps to a pre-tokenizer regex.
  Beware that these GGUF namespaces reuse arch-like literals with different
  meanings (`tokenizer.ggml.model == "llama"` means SentencePiece).
- chat-end markers are appended by `add_chat_eos`
  (`crates/infr-llama/src/util.rs`) — it appends a fixed list, so a family whose
  end-of-turn marker is not in it and not the GGUF's declared `eos_token_id`
  will ramble.
- recommended sampling defaults go in the `arch_sampling` family table in
  `crates/infr-cli/src/main.rs` (optional — it falls back to a generic triple).

### 7. Verify against the oracle

In this order:

1. **Logits vs llama.cpp** on a fixed prompt — build the reference
   (`~/Projects/mxaddict/llama.cpp`) and compare top-k on the CPU backend.
   Disagreement here is an arch bug; every later check is noise until it passes.
2. **CPU golden** — add a case to `crates/infr-llama/tests/cpu_backend.rs`. The
   goldens render a plain prompt through the model's own template, generate
   greedily, and lock an FNV-1a of the output. The CPU goldens are **not**
   `#[ignore]`d — each self-skips when its GGUF is not in the HF cache — so they
   run wherever the model exists. Capture with `INFR_BLESS=1`, then **read the
   generated text**: a blessed hash of garbage is a green light wired to
   nothing.

   ```bash
   INFR_TEMP=0 cargo test --release -p infr-llama --test cpu_backend -- --nocapture
   ```

3. **GPU seam** — the `gpu_seam_matches_cpu_*` tests in the same file compare
   the production Vulkan path against the CPU reference on the same model. These
   ones _are_ `#[ignore]`d behind a Vulkan device, so they need
   `--include-ignored` on a GPU box. Use the strict token-identical form for a
   dense model and the loose form (top-5 overlap + a cosine floor) for MoE,
   where near-tie routing legitimately flips between f32 CPU and f16 GPU.
4. **End to end** — `infr run <ref> "…"` produces coherent text, and
   `infr compare` puts a tok/s row next to llama.cpp.

### 8. Land it

`cargo clippy --all-targets -- -D warnings`, `cargo fmt --all`, `cargo test` on
the workspace. Then the paperwork that keeps the docs from drifting: a
`CHANGELOG.md` entry under `## [Unreleased]`, a row in the root `README.md`
supported-models table, and — if the family got its own design doc — a line in
[`docs/README.md`](README.md).

---

## Candidate models (next)

Ranked by ROI against the architecture above. Step 0 of the recipe is the same
for all of them and costs minutes; do that before believing any effort estimate
here.

1. **DeepSeek** (`deepseek` → `deepseek2` → `deepseek32` → `deepseek4`).
   **Highest strategic value**, and the only candidate with a written plan:
   [`deepseek.md`](deepseek.md). Adds **MLA (Multi-head Latent Attention)**, the
   missing attention family behind the whole V2/V3/V4 line. Staged around the
   constraint that only the first two stages have a model small enough to
   develop against, so stages 1–2 must leave MLA and MoE-routing pieces behind
   that are independently tested. Depends on the streaming work tracked as B36
   in [`backlog.md`](backlog.md).
2. **GLM-4.7-Flash** (~30B MoE) or **Ernie 4.5 21B-A3B** (MoE). Mostly standard
   MoE FFN — infr's batched-expert path covers it — plus minor arch quirks (GLM
   post-norm / partial RoPE). Expect recipe steps 3–4 only. Some GLM variants
   ship native MTP heads, which infr currently parks ([`mtp.md`](mtp.md)) —
   those tensors go unused rather than unsupported.
3. **Nemotron-Nano / Nemotron-H** (Mamba2-Transformer hybrid MoE). Adds **Mamba2
   SSM**, extending the existing `Conv1dSilu` + `DeltaNet` linear-attention
   machinery — a real differentiator, since llama.cpp's Mamba-hybrid GGUF path
   is weak. Med-high effort **and a weaker oracle**, which is the reason to do
   it after MLA rather than before.

---

## GPU / driver leads

The coopmat operand tiers (fp8/bf16/int8 vs f16), the coopmat2 per-element
probe, and the bf16-coopmat rate are recorded — with their measurements and the
re-check commands — under "Coopmat operand tiers" in
[`perf/playbook.md`](perf/playbook.md). They are **not** repeated here; that
file is the one that gets updated when a Mesa release moves them.

One lead lives only here because it has never been measurable:

- **`VK_NV_cooperative_vector`** — matrix×**vector**, i.e. decode / GEMV, which
  is infr's perf frontier. **Not exposed by RADV** as of Mesa 26.1.4 (checked in
  both default and driconf modes). If a future Mesa ships it, it could route the
  decode GEMVs through the matrix unit. Nothing to do until then — re-check with
  `vulkaninfo` before spending any time on it.

---

## Risks / open questions

- **Quant matmul correctness** — the classic footgun. Every quant type gets
  validated against the CPU dequant reference before it is trusted; the coverage
  table is in [`perf/kernels.md`](perf/kernels.md).
- **Graph abstraction vs perf** — the compile-once/execute-many design must not
  force per-op sync. Still the live tension in every perf slice.
- **Backend drift** — three backends behind one trait means an op can be right
  on CPU and wrong (or missing) on Vulkan or Metal. The parity tests are the
  only thing standing between that and a silent wrong answer.
- **safetensors** — still unbuilt. Every loader path assumes GGUF today.

---

## Reference implementations

Paths on the author's machine; treat them as the source of truth over any
summary in these docs.

- **`~/Projects/mxaddict/llama.cpp`** — mainline. `src/llama-arch.cpp` (arch
  tables, tensor names, metadata keys) and `src/models/<arch>.cpp` (the per-arch
  forward pass) are what a port is diffed against. `build/bin/` holds
  `llama-cli` / `llama-bench`, the oracles `infr compare` shells out to.
- **`~/Projects/mxaddict/llama.cpp-dg`** — the DiffusionGemma fork, which never
  merged upstream. Its `llama-diffusion-cli` is the oracle for
  `arch=diffusion-gemma`, resolved via `INFR_LLAMA_DIFFUSION_CLI` > `PATH` > the
  fork's build dirs — see `ModelBench::llama_diffusion_cli_path` and
  [`perf/benchmarking.md`](perf/benchmarking.md).
- **`~/Projects/scratch/dgemma-openai-server.py`** — the original OpenAI shim
  prototype (channel split, `<|tool_call>` parsing). Its logic now lives in
  `infr-chat` / `infr-server`; kept only as a historical reference.
- **`ref/`** in this repo — vendored reference C++ sources, read-only.

The DiffusionGemma model spec (canvas length, mask token, entropy-bound
defaults, the `<|turn>` / `<|channel>` / `<|tool_call>` wire format) is **not
duplicated here** — it is in [`diffusion-gemma.md`](diffusion-gemma.md),
alongside the denoise-graph design. Read those values from GGUF metadata at
runtime; do not hardcode them.

---

## Historical: the original milestones

The MVP shipped against **autoregressive** decoders (Llama / Qwen2 / Qwen3 /
Qwen3-MoE / Gemma 3 / Gemma 4) before DiffusionGemma. Kept for the record — the
acceptance criteria are what each milestone was actually held to.

| #   | Milestone        | Done when                                                            | Status |
| --- | ---------------- | -------------------------------------------------------------------- | ------ |
| 1   | compute smoke    | f16 coop-matrix matmul on the GPU matches CPU within tolerance       | ✅     |
| 2   | core trait       | `Tensor`/`Graph`/`Op`/`Backend` compile; a 2-op graph runs on Vulkan | ✅     |
| 3   | `infr pull`      | resolve + resumable download into the shared HF cache                | ✅     |
| 4   | Vulkan backend   | matmul/dequant/rmsnorm/rope/softmax each validated vs CPU            | ✅     |
| 5   | GGUF loader      | weights upload; tokenizer + embedded chat template exposed           | ✅     |
| 6   | forward pass     | final logits match llama.cpp top-k on a fixed prompt                 | ✅     |
| 7   | attention + MoE  | multi-layer forward matches reference (SWA + full + MoE)             | ✅     |
| 8   | diffusion decode | fixed-seed generation matches the llama.cpp fork's text              | ✅     |
| 9   | `infr run`       | interactive terminal chat streams a coherent answer                  | ✅     |
| 10  | `infr serve`     | opencode / Claude Code CLI complete an agentic tool turn             | ✅     |
| 11  | perf pass        | ongoing — [`perf/playbook.md`](perf/playbook.md)                     | 🔄     |
| 12  | second backend   | the op-list seam runs on CPU, Vulkan **and** Metal                   | ✅     |

The Metal backend and the CPU reference both arrived through milestone 12, which
is the load-bearing evidence that the seam holds: neither required a change
above `compute`.
