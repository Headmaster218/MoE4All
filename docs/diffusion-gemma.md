# DiffusionGemma (`diffusion-gemma`) — design for the unified seam

## Status

- **UNIFIED-seam DiffusionGemma, phases 1–4 landed (2026-07-05)**: design (phase
  0, `0785fcf`), config + weights + causal prefill (phase 1, `ca2575a`), canvas
  denoise + `AttnMask::Canvas` + self-conditioning (phase 2, `2452786`),
  entropy-bound block decode + `infr run` wiring (phase 3, `c9983c1`),
  serve/bench/compare + pre-seam scaffolding removal (phase 4, this doc's
  update). Built on the SAME transformer runner every other arch uses
  (`SeamModel` + the per-backend `ChatModel` impls) from day one — no bespoke
  "engine" seam.
- **Phase D: Metal denoise, code-complete but hardware-unvalidated.** Written
  BLIND on a Linux box (same precedent as the KV-quant Metal work — shipped
  blind, CI-green): a dedicated split-KV canvas kernel
  (`attention_canvas_f16kv`/`attention_canvas32_f16kv` + their f32 siblings in
  `crates/infr-metal/shaders/attention.metal`) gives Metal's attention the
  bidirectional `[lo, kv_len)`-for-every-row reach `AttnMask::Canvas` needs (see
  "Metal implementation notes" below); `DiffusionGemmaMetalSession`
  (`crates/infr-llama/src/seam_model.rs`) is the Vulkan session's twin;
  `infr run`/`serve`/`bench` now route `--dev metal` (`INFR_DEV=metal`) there
  instead of falling back to CPU. In-graph self-conditioning (`gpu_sc`, Phase B)
  widened to cover Metal too — its `Op::Softmax`/`Op::Linear` already handled
  the shape/dtype. Metal hardware validation is deferred to the M3 collaborator
  (see the checklist at the end of this section); CPU and Vulkan remain the
  validated backends until then. **Not done in Phase D**: a Metal batched-MoE
  pipeline — DG's fused MoE shape still runs Metal's per-token host-loop
  fallback (too large to build blind; same gap as Vulkan, see "Known gaps"
  below).
- **What works**: `infr run` (CPU + Vulkan, entropy-bound block-diffusion
  decode, multi-turn REPL — see the oracle-equivalent captured runs below);
  `infr serve` (OpenAI `/v1/chat/completions`, streaming + non-streaming,
  `reasoning_content` split via the shared channel splitter — the exact same
  `<|channel>thought…<channel|>` markers as the oracle); `infr bench` (DG-aware:
  reports the block-diffusion shape — pp tok/s, end-to-end generated tok/s, EB
  steps run, and the "in-step parallel" rate the oracle itself reports — since
  `llama-bench` has no diffusion mode to compare against); `infr compare` /
  `infr compare --sweep` (the mainline `llama-bench` route still has nothing to
  compare against, but the reference fork at `~/Projects/mxaddict/llama.cpp-dg`
  builds `llama-diffusion-cli` — a real, usable oracle for this arch — so the
  compare arm shells that out instead of bailing; see "Comparing against the
  fork oracle" below). CPU==Vulkan cross-backend parity validated per the
  validation ladder below.
- **Known gaps** (perf follow-ups, not correctness bugs):
  - Metal denoise is code-complete (Phase D, see the status bullet above) but
    **hardware-unvalidated** — nobody has run it on a real Metal device yet. CPU
    and Vulkan are the only backends whose DG output is confirmed correct.
  - Metal's canvas attention kernel is a NEW split-KV variant forced on for
    EVERY row regardless of row count (mirrors the Vulkan `attn_partial` canvas
    branch's own "perf pass deferred" — see attention.metal) — it never takes
    Metal's faster flash/flash2/vec tiers, so denoise throughput on Metal is a
    probably-conservative lower bound until it's profiled.
  - `sc_embT` (the in-graph self-conditioning soft-embedding weight,
    `[n_embd, n_vocab]`, `DType::F16`) has no native Metal GEMV — it flows
    through Metal's generic non-quant `Linear` path, which dequant-caches the
    WHOLE weight to f32 once per session (~2x the f16 footprint, one host
    round-trip dequant). Correct, just not the efficient path a quantized weight
    would get; see `weight_buf`'s VRAM-budget guard for the failure mode if it
    doesn't fit.
  - Host-side self-conditioning cost dominates Vulkan decode: the per-step
    self-conditioning MLP (`self_cond_pre_norm`/gate/up/down) runs on the CPU
    host, ~190 GFLOP/step, and end-to-end Vulkan decode lands around ~0.9 tok/s
    as a result.
  - The denoise graph is rebuilt (recompiled) on every `denoise()` call rather
    than cached/reused across steps.
  - The single-dispatch-per-stage batched-MoE Vulkan path doesn't cover DG's
    fused `ffn_gate_up_exps`/`ffn_down_exps` expert layout — DG's MoE FFN still
    runs the per-token expert loop on GPU (and, more severely, on CPU: a 256-row
    canvas × 30 layers × top-8-of-128 experts per-token loop is why the CPU
    reference denoise is impractically slow for this model size, not just
    "slower than Vulkan"). Metal shares this gap — Phase D did NOT build a Metal
    batched-MoE pipeline (too large to write blind); DG's MoE FFN on Metal also
    falls to the per-token host loop.
- **Removed in phase 4**: the pre-seam scaffolding (`infr-model`, `infr-decode`,
  the `infr-engine::Engine` stub) — dead code superseded by the unified runner
  since phase 1.

Block text-diffusion MoE on a **Gemma-4 backbone**. Reference implementation:
llama.cpp PR #24423 (checked out at `~/Projects/mxaddict/llama.cpp-dg`,
`src/models/diffusion-gemma.cpp` + `gemma4-common.h`); oracle binary
`build/bin/llama-diffusion-cli` (CPU build). Model:
`unsloth/diffusiongemma-26B-A4B-it-GGUF` (Q4_K_M ≈ 16 GB, fits the 7900 XTX).

This REPLACED the pre-seam scaffolding (`infr-model`/`infr-decode`/the
`infr-engine::Engine` stub, all dead code) — DiffusionGemma is built on the
unified transformer runner like every other arch; the scaffolding was deleted in
phase 4 (see Status above).

## Confirmed metadata (real GGUF dump, 2026-07-05)

- `general.architecture = diffusion-gemma`, 30 layers, n_embd=2816,
  vocab=262144, tied lm_head, `final_logit_softcapping=30`, rms_eps=1e-6,
  **attention scale = 1.0** (NO 1/√d — same as gemma4).
- SWA pattern 5:1 (`sliding_window_pattern` bool array; full-attn at layers
  5,11,17,23,29), window 1024.
- **Heterogeneous attn dims**: SWA layers hd=256, 16q/8kv, rope base 1e4, no
  freq_factors; FULL layers hd=512, 16q/2kv, rope base 1e6, `rope_freqs[256]`
  proportional-rope freq_factors. `attention.causal = False` in metadata (the
  forward builds its own region mask).
- **Global layers have NO `attn_v`** (missing at exactly 5,11,17,23,29): V = the
  raw k_proj output (pre-k-norm), reshaped, then the weightless V rms-norm.
- FFN per layer = **dense MLP ∥ 128-expert MoE, summed** (see wiring below):
  dense gate/up/down at n_ff=2112 (GELU-par), experts fused
  `ffn_gate_up_exps[2816,1408,128]` + `ffn_down_exps[704,2816,128]` with a
  per-expert scale `ffn_down_exps.scale[128]`, router `ffn_gate_inp[2816,128]` +
  elementwise input scale `ffn_gate_inp.scale[2816]`, 8 experts used, softmax
  gating.
- Per-layer scalars are `[1]` TENSORS: `blk.N.enc_layer_output_scale.weight`
  (encoder/prompt) and `blk.N.layer_output_scale.weight` (decoder/canvas).
- Top-level self-conditioning MLP: `self_cond_pre_norm[2816]`,
  `self_cond_gate/up [2816,2112]`, `self_cond_down [2112,2816]`.
- `diffusion.canvas_length = 256`, `tokenizer.ggml.mask_token_id = 4`.

## The architecture, per the PR source (verified)

- **Backbone identical to gemma4** (shared `gemma4-common.h`): SWA pattern,
  heterogeneous per-layer dims, sandwich norms, GeGLU/GELU, per-layer output
  scalars — machinery infr's gemma4 support already has, EXCEPT the dual FFN and
  the V=K reuse (new).
- **[prompt | canvas] regions**, split at `P = n_tokens − canvas_length`:
  1. input embeddings — prompt: `embed·√n_embd` (standard gemma); canvas:
     `rmsnorm_noscale(embed·√n_embd)` (weightless RMSNorm — infr has it as
     QkNorm-with-ones, the gemma4 V-norm trick);
  2. per-layer output scalar — prompt rows use the ENCODER scalar, canvas rows
     the DECODER scalar (two per-layer arrays in the GGUF);
  3. attention mask — prompt queries: causal over prompt only (SWA-clipped),
     NEVER see canvas; canvas queries: bidirectional over everything — global
     layers see all prompt + canvas, sliding layers the last `n_swa−1` prompt
     positions + all canvas.
- **Two-pass equivalence** (the key implementation fact, stated by the PR): a
  causal encoder prefill of the prompt + a bidirectional decoder denoise of the
  canvas (zero self-conditioning) reproduces the single no-cache forward. infr
  implements the two-pass form natively:
  - **Pass 1 (per block)**: standard causal prompt prefill through the existing
    gemma4 path, encoder scalars — KV cache rows `0..P` written once.
  - **Pass 2 (per denoise step)**: forward ONLY the C canvas rows with decoder
    scalars; `WriteKv` overwrites cache rows `P..P+C` each step;
    `Attention { rows: C, kv_len: P+C, mask: Canvas{..} }` — the one new seam
    piece.
  - **Self-conditioning is ON by default for canvas models**
    (`diffusion-cli.cpp:273 diff_params.self_conditioning = true`; the cli
    enables it before context creation). Step 0 gates it to zero (`sc_use=0`,
    reproducing the zero-SC exactness forward); every later step feeds the
    PREVIOUS step's raw canvas logits:
    `soft = softmax(logits·temp_inv) @ embdᵀ · √n_embd`, then `sc_pre_norm` →
    GELU-gated MLP (gate/up/down, n_ff=2112, `gelu(gate)·up`) →
    `canvas += sc_sig` → the weightless rms-norm. Oracle parity REQUIRES this
    block (Phase 2/3). llama.cpp pre-dequantizes/transposes `tok_embd` to f16
    `[n_vocab, n_embd]` once for the soft-embed matmul.
- **Block diffusion decode**: canvas of `diffusion.canvas_length` (256) mask
  tokens (`tokenizer.ggml.mask_token_id`); N denoise steps unmask progressively
  (entropy-bound sampler from the PR's `common/sampling`); a finished block
  commits (its rows become prompt: already in the KV cache — the next block's P
  grows by C) and the loop advances. `-n` derives the block count:
  `blocks = ceil(n_predict / canvas_length)`.

## FFN block wiring (exact, from `gemma4_build_ffn_moe`)

On the post-attention residual `attn_out` (already includes the attn residual
add):

```
dense = post_ffw_norm_1( gelu_par_ffn( ffn_norm(attn_out) ) )         # gate/up/down 2112, GELU
router_in  = rmsnorm_noscale(attn_out) · (1/√n_embd) · ffn_gate_inp_s  # UNNORMED residual!
logits     = ffn_gate_inp @ router_in                                  # [128]
moe_in     = pre_ffw_norm_2(attn_out)
moe = post_ffw_norm_2( moe_ffn(moe_in, softmax-top8(logits), GELU,
                               gate_up_exps fused, down_exps·down_exps_s) )
out = post_ffw_norm(dense + moe) + attn_out                            # final sandwich + residual
```

Notes: `build_moe_ffn(..., norm_w=true, ...)` — top-8 softmax weights are
renormalized to sum 1; the per-expert `ffn_down_exps.scale[128]` multiplies each
expert's down output. Both parallel branches read `attn_out`.

## Seam extensions required (small!)

1. **`AttnMask::Canvas { prompt_len }`** (name TBD) on `Op::Attention`: rows are
   all canvas queries; every row's valid range is `[lo, P+C)` where `lo = 0` on
   global layers and `max(0, P−(n_swa−1))` on SWA layers — i.e. NOT per-row
   causal; uniform bidirectional reach per layer. Backends: CPU interpreter
   arm + Vulkan scalar/split paths (per-row `qpos1` becomes `kv_len` for every
   row; `lo` from the mask instead of the causal window). Flash/nonfa can gate
   off initially (canvas C=256 rows at kv P+C — the split path handles it; perf
   pass later).
2. Per-phase per-layer scalars: the graph is built per phase, so the prefill
   graph bakes encoder scalars and the denoise graph decoder scalars — no op
   changes, just Config carrying both arrays.
3. Denoise-step graph variant in the runner: canvas embedding norm + decoder
   scalars + the Canvas mask + logits over ALL C rows (the spec-decode
   `logits_rows`-style multi-row LM head already exists on Metal; Vulkan's
   lm_head GEMV takes rows — confirm).

## Decode loop (engine level, backend-agnostic)

```
tokens = chat_prompt
for block in 0..blocks:
    P = len(tokens); canvas = [mask_id; C]
    prefill(tokens[cached..])            # causal, encoder scalars (suffix diff as usual)
    for step in 0..steps:
        logits = denoise_forward(canvas) # C rows, decoder scalars, Canvas mask
        unmask most-confident positions per the entropy-bound schedule
        if all unmasked: break
    tokens += canvas                     # block commits; KV rows P..P+C already valid?
                                         # (confirm: the PR re-prefills committed blocks
                                         #  causally or reuses the denoise KV — check)
    stop on EOS inside the block
```

Resolved from the oracle source (`diffusion-cli.cpp` run_turn, ~line 413+):

- **Block commit = causal re-processing.** Committed canvas tokens append to
  `prefix`, and the next block's pass treats them as prompt — in infr terms the
  committed block gets a causal suffix prefill (encoder scalars) through the
  normal session machinery before the next canvas denoise. The whole
  `[prefix | canvas]` must fit one ubatch (the oracle errors otherwise).
- **`trim_canvas`** cuts the block at an end token or a repetition loop; a
  partial cut (< canvas_length) ends the turn. Blocks derive from `-n`:
  `ceil(n_predict / canvas_length)`.
- **Sampler: entropy-bound is the DEFAULT for canvas models**
  (`use_eb = canvas_length > 0 && eb_mode != 2`), parameters metadata-driven
  with these fallbacks:
  `diffusion.eb_max_steps=48, eb_t_min=0.4, eb_t_max=0.8, eb_entropy_bound=0.1, eb_stability_threshold=1, eb_confidence_threshold=0.005`
  (CLI-overridable). Phase 3 ports `diffusion_generate_entropy_bound` from the
  PR.

## Live denoise view (`infr run`)

`infr run --diffusion-visual` turns on a live canvas view while it denoises a
block — a small opt-in port of the oracle's `--diffusion-visual` UX (see
`~/Projects/mxaddict/llama.cpp-dg/examples/diffusion/diffusion-cli.cpp`), not a
dependency on it. It is CLI presentation only, so it is a plain clap flag rather
than a `Config` field; the `INFR_DIFFUSION_VISUAL` spelling still works as the
flag's env fallback. Only active on a TTY stdout (piped/redirected output is
left alone); `--diffusion-visual=force` bypasses the TTY check for scripted
verification. Per step it decodes the block's current canvas fresh (a throwaway
tokenizer, ≤ `canvas_length` tokens, no GPU work), prints already-accepted runs
as normal text and still-renoising positions (this sampler has no literal mask
token — see `crate::diffusion`'s module doc) as a dim `·` placeholder, and
redraws a fixed-size scratch region in place (cursor-up + erase) so the terminal
never scrolls mid-block. The region is erased right before the finished reply
prints, so the ordinary transcript stays clean; the hook is a no-op with the
flag unset (byte-identical decode loop, zero GPU cost).

```bash
infr run <dg-gguf> --diffusion-visual
```

(The flag takes an optional value, so it must follow the model argument — put it
first and clap consumes the model as its value.)

## Validation ladder

1. Metadata parse matches the GGUF dump (Phase 1).
2. Prompt prefill: CPU==Vulkan, finite logits (Phase 1).
3. Denoise step: CPU==Vulkan; qualitative single-step unmask sanity (Phase 2).
4. End-to-end vs `llama-diffusion-cli` on the same GGUF, greedy/fixed-seed —
   token-identical or documented-equivalent (Phase 3).
5. Perf: each denoise step is a C=256-row prefill-shaped forward — the shape
   infr's batched machinery is strongest at (Phase 4 bench + compare rows;
   `infr compare`/`--sweep` measure this against the fork's
   `llama-diffusion-cli` oracle — see "Comparing against the fork oracle" below
   — since mainline `llama-bench` has no diffusion mode at all).

## Comparing against the fork oracle

`arch=diffusion-gemma` isn't merged into mainline llama.cpp, so `llama-bench`
can't run it — but the reference fork at `~/Projects/mxaddict/llama.cpp-dg`
builds `llama-diffusion-cli`, a real oracle for this arch. `infr compare` /
`infr compare --sweep` shell it out instead of bailing (the bail this doc used
to describe is gone).

**Binary resolution** (`ModelBench::llama_diffusion_cli_path`), in order:

1. `INFR_LLAMA_DIFFUSION_CLI` env var — explicit override.
2. `llama-diffusion-cli` on `PATH`.
3. The fork's own build directories: `build-vulkan/bin/llama-diffusion-cli`
   (GPU,
   `cmake -DGGML_VULKAN=ON -B build-vulkan && cmake --build build-vulkan --target llama-diffusion-cli`)
   or `build/bin/llama-diffusion-cli` (CPU-only) — picked by `-ngl 0` vs
   `-ngl >0`, matching every other arm.

**Caveat hit in practice**: mainline llama.cpp already ships its own generic
`llama-diffusion-cli` (LLaDA/Dream support), so tier 2 can resolve to a REAL
binary on `PATH` that still can't load `arch=diffusion-gemma` — it errors
`unknown model architecture: 'diffusion-gemma'` rather than failing to resolve.
`ModelBench::llama_diffusion` catches that specific failure signature on the
first rep and falls through to the fork build automatically (logging the
fallback), so a stock llama.cpp install on `PATH` doesn't silently break this
arm.

**Metric honesty**: DG is entropy-bound, so infr and the oracle run a DIFFERENT
NUMBER of denoise steps for the same `-n 256` request — end-to-end tok/s is NOT
apples-to-apples (whichever tool trims its canvas earlier looks faster for
free). The compare/sweep rows split accordingly:

- `dg-step` — per-step throughput (`canvas_length * steps / decode_secs`, infr's
  `DgBenchResult::parallel_ts` vs the oracle's own reported "in-step parallel N
  tok/s"). This is the real infr-vs-llama ratio and the only DG metric that
  feeds the sweep's ranked "BIGGEST GAPS" summary.
- `dg-e2e` — informational only: each side's own end-to-end tok/s with its own
  step count folded into the row text (e.g. `14.0t/s@48st` vs `46.8t/s@23st`),
  so the step-count mismatch is visible rather than hidden.

Example (Vulkan, `unsloth/diffusiongemma-26B-A4B-it-GGUF:Q4_K_M`):

```
diffusiongemma-26B-A4B-it-Q4_K_M.gguf dg-step    |       674 |      1077 |      0.63x
diffusiongemma-26B-A4B-it-Q4_K_M.gguf dg-e2e     |   14.0t/s@48st |   46.8t/s@23st | informational (steps differ)
```

## Metal implementation notes (Phase D)

Written BLIND on a Linux box: `infr-metal` compiles here
(`#![cfg(target_os = "macos")]` strips the whole crate off-macOS, so
`cargo check`/`clippy` only syntax-check it), but `.metal` shaders only actually
compile+run on a real Metal device, and macOS CI has no GPU-attached runner for
this arch's model files. Same situation as the KV-quant Metal work, which
shipped this way and landed CI-green.

- **Canvas attention** (`AttnMask::Canvas`,
  `crates/infr-metal/shaders/attention.metal`): every one of Metal's existing
  attention tiers (the plain kernel, the split-KV `ATTNSPLIT_KERNEL` family, the
  flash/flash2 kernels, the vector-flash `attnvec_*` kernels) computes its
  per-row bound from `pos + row_index` plus an optional sliding-window `lo` —
  none can express "every row shares the SAME `[lo, kv_len)` regardless of its
  own position". Repurposing an existing kernel's dead field with a sentinel
  (the way the Vulkan `attn_partial.comp` canvas branch reuses its `rows`
  push-constant slot) risked a genuine non-canvas dispatch and a canvas dispatch
  both landing on the same nonzero sentinel value in the SAME kernel, so canvas
  gets its own dedicated `ATTNSPLIT_CANVAS_KERNEL` macro (4 instantiations:
  `attention_canvas_f32`/ `attention_canvas_f16kv` at NSG=8/MAXHD=256,
  `attention_canvas32_f32`/ `attention_canvas32_f16kv` at NSG=32/MAXHD=128) that
  never serves a non-canvas caller, so its repurposed `p.pos`/`p.kv_len` fields
  (fixed `hi`/`lo` instead of their ordinary meaning) can't collide with
  anything. `exec.rs`'s `Op::Attention` arm forces every `AttnMask::Canvas` op
  straight to this kernel family (bypassing the flash/split/vec routing
  entirely), gated the same way the ordinary split kernels are (threadgroup-cap
  `fits()` checks, hd<=128 preferring the 32-wide variant). Q8_0 KV cache +
  canvas is explicitly rejected (`Error::Unsupported`) — Q8_0's attention family
  has no split-canvas sibling yet; every other KV dtype (native f16, or a
  block-quant that goes through the existing dequant-to-f16-scratch prepass) is
  covered.
- **Self-conditioning** (`gpu_sc`, Phase B): widened from Vulkan-only to
  `matches!(be.name(), "vulkan" | "metal")` — verified by reading, not running:
  `softmax_wide_f32` is a plain grid-stride loop over `(rows, dim)` with no
  shape restriction (handles the `[256, vocab]` shape unmodified), and
  `sc_embT`'s `DType::F16` weight already flows through Metal's existing
  non-quant `Linear` path (`weight_buf` + `linear_f32`) — functionally correct,
  though not a native f16 GEMV (see "Known gaps" above for the perf/memory
  caveat).
- **Batched MoE**: NOT built this phase — too large to write blind. DG's fused
  MoE shape already falls to Metal's per-token host loop (same as Vulkan's own
  gap for this shape), unchanged by Phase D.

**Hardware-validation checklist for the M3 collaborator** (each item currently
unverified):

- [ ] `cargo test --release -p infr-metal -- --include-ignored` on real hardware
      — confirms the canvas kernels compile+run at all (a syntax slip that
      Linux's empty-crate `cargo check` can't catch would surface here as an MSL
      compile error or a GPU validation failure).
- [ ] `attention_canvas_split32_matches_reference` /
      `attention_canvas_split8_hd256_matches_reference` (new in
      `tests/parity.rs`) pass within their `5e-3` relative-error tolerance
      against the CPU reference — the actual numeric proof the
      `[lo, kv_len)`-for-every-row math is right, not just that it dispatches.
- [ ] `cargo test -p infr-metal --test kernel_names -- --include-ignored` —
      confirms `attention_canvas*` resolve in the compiled MSL library (the
      tripwire test).
- [ ] A real DG denoise run (`infr run --dev metal <dg-gguf>`) produces
      coherent, non-garbage text — the end-to-end smoke test no unit test can
      substitute for.
- [ ] `infr bench --dev metal <dg-gguf> -p 128 -n 256` reports a sane
      (non-crashing, non-degenerate) pp/decode rate, ideally compared against
      the Vulkan number on the same machine if available.
- [ ] With self-conditioning on (the default after a step 0), confirm output
      stays coherent across multiple steps — this exercises `sc_embT`'s
      dequant-to-f32 Linear path AND the `softmax_wide_f32` kernel at the real
      `[C, vocab]` shape, neither of which any existing Metal test dispatches at
      this size.
- [ ] If any of the above fails on a memory-constrained device, check for the
      `weight_buf` "would exceed the GPU working set" error specifically
      (`sc_embT` dequant-caches to f32, ~2x its f16 size) before assuming a
      correctness bug.

## Oracle reference outputs (CPU, greedy `-s 42 --temp 0`, captured 2026-07-05)

```
$ llama-diffusion-cli -m <Q4_K_M> -p "What is the capital of France?" -n 64 -s 42 --temp 0
<|channel>thought
The user is asking for the capital of France.
    *   Country: France.
    *   Capital: Paris.
Provide the direct answer clearly.<channel|>The capital of France is Paris.
(10 EB steps, 1 block; 13.2 tok/s CPU, in-step parallel 132 tok/s)

$ llama-diffusion-cli -m <Q4_K_M> -p "Write a haiku about the moon." -n 64 -s 42 --temp 0
<|channel>thought
*   Topic: The moon.
    *   Format: Haiku (5-7-5 syllables).

    *   Night
(19 EB steps, 1 block; trim cut mid-thought at the 64-token budget)
```

Thinking channel markers: `<|channel>thought … <channel|>` (from the GGUF's own
chat template — the jinja renderer handles it like every other model).
