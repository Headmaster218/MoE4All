# DeepSeek support plan (V1 → V2/V3 → V3.2 → V4)

Status: **Stage 2 done.** CPU path works end-to-end on V2-Lite; Vulkan and Metal
MLA kernels are implemented, wired, and executed on their real devices (Vulkan
on the GPU box, Metal via the macOS CI job's parity suite); Vulkan MoE is
implemented; exp_probs_b loads from V3 GGUFs; the GPU seam test passes (cosine
0.9955 CPU-vs-Vulkan, matching greedy output vs llama.cpp c629da5); the YaRN
ramp is verified numerically against llama.cpp at short AND long context (see
the checklist).

Stage 1 (`deepseek`) was skipped — V2-Lite is the development model.

The reference implementation is llama.cpp at `b10218-1-gc629da5`, checked out
locally at `~/Projects/mxaddict/llama.cpp`. Every claim about DeepSeek's maths
in this document was read out of that tree, and every claim about what `infr`
already has was read out of this one. Where something was **not** verified it
says so — those lines are the ones to check first, not to trust.

Re-verified against both trees on 2026-08-05. That pass corrected the stage-1
rope-type mapping (it prescribed a permute that would have corrupted output),
renamed the router fields to where they actually live, and closed the Metal
shared-expert and LayerNorm questions. Everything else below survived the check
unchanged, including the llama.cpp line counts and every GGUF key name.

The generic procedure for adding any architecture — dump, diff, register, load,
graph, verify — is
[plan.md § Adding a model architecture](plan.md#adding-a-model-architecture-the-recipe).
This document is only what is DeepSeek-specific on top of it.

Disk paging is what stages 3–4 run on at all; it landed (`69b6de0`, `588653b`).
[backlog.md](backlog.md) B36 holds the paging optimizations that were measured
but **not** built, including the one §0.3 below depends on.

## Why this order

llama.cpp keeps **five** DeepSeek architectures, not one. They are separate
model classes with separate builders (`llama-model.cpp`, the
`LLM_ARCH_DEEPSEEK*` arms of `llama_model::create`):

| GGUF `general.architecture` | models                         | llama.cpp builder             |       size |
| --------------------------- | ------------------------------ | ----------------------------- | ---------: |
| `deepseek`                  | DeepSeek-LLM 7B/67B, MoE-16B   | `src/models/deepseek.cpp`     |  194 lines |
| `deepseek2`                 | V2, V2-Lite, V3, V3-0324, V3.1 | `src/models/deepseek2.cpp`    |  438 lines |
| `deepseek2-ocr`             | DeepSeek-OCR                   | `src/models/deepseek2ocr.cpp` |          — |
| `deepseek32`                | V3.2                           | `src/models/deepseek32.cpp`   |  506 lines |
| `deepseek4`                 | V4-Flash, V4-Pro               | `src/models/deepseek4.cpp`    | 1203 lines |

The staging follows from one hard constraint: **only the first two stages have a
model small enough to develop against.**

| stage | arch         | development model                                | fits a 24 GB card?   |
| ----- | ------------ | ------------------------------------------------ | -------------------- |
| 1     | `deepseek`   | `deepseek-llm-7b-chat`, `deepseek-moe-16b-chat`  | yes (~4 / ~10 GB Q4) |
| 2     | `deepseek2`  | `DeepSeek-V2-Lite-Chat` (16B total, 2.4B active) | yes (~10 GB Q4)      |
| 3     | `deepseek32` | **none — V3.2 is 671B**                          | no                   |
| 4     | `deepseek4`  | **none — V4-Flash smallest quant is 82.5 GB**    | no                   |

GGUFs for the stage-1/2 models exist (TheBloke, mradermacher, legraphista).
Stages 3 and 4 can only ever be exercised through the disk pager at 80+ GB, at a
few tokens/sec, with no CPU oracle that finishes in reasonable time. **So stages
1–2 must leave behind MLA and MoE-routing pieces that are independently tested,
because from stage 3 on there is no cheap way to find a bug.**

`deepseek2-ocr` is out of scope; it is a vision model that reuses the arch id.

## What `infr` already has

Verified against this tree. There is **no plugin system** — an architecture is a
set of fields on one `Config`, branched on inside one graph builder and one
weight-load loop.

Already present and directly reusable:

Note on naming: there is no `MoeConfig` type. Everything the router is
configured by lives as **fields on the `Op::MoeFfn` variant** in
`infr-core/src/graph.rs` (`gating`, `norm_w`, `scale`, `n_expert`, `n_used`,
`n_ff_exp`, `down_scale`, `fused_gate_up`, `weight_before`, `ep_band`) — grep
for the variant, not for a struct.

- **Sigmoid MoE gating** — `MoeGating::Sigmoid` (`infr-core/src/graph.rs`),
  CPU + Vulkan. This is V3's `scoring_func`.
- **Gate-weight normalisation on/off** (`Op::MoeFfn`'s `norm_w`) — DeepSeek's
  `norm_topk_prob`.
- **Routed scaling factor** (`Op::MoeFfn`'s `scale`), read from
  `{arch}.expert_weights_scale` — the same GGUF key DeepSeek uses.
- **Shared experts, plain-summed** — `FfnW::Moe { shexp }` with tensor names
  `ffn_{gate,up,down}_shexp`. DeepSeek's shared expert is plain-summed, so the
  llama4 path fits.
- **Expert count headroom** — the Vulkan `moe_topk.comp` supports up to 1024
  experts; V3/V4's 256 fit.
- **Expert paging** — `infr-core/src/pager.rs` + `hostpager.rs` +
  `infr-vulkan/src/pager.rs`, keyed `(layer, role, expert_id)`.
- **Partial RoPE** — `Op::Rope` rotates the first `rope_dim` of each head and
  passes the rest through, which is exactly DeepSeek's decoupled rope shape.
- **Per-layer heterogeneity** — `Config::layer_head_dim`, `layer_n_kv`,
  `layer_rope_theta`, `is_swa_layer`, `is_moe_layer` and friends already exist.
- **Low-rank projections** need no new op: `Linear → RmsNorm → Linear` covers
  `wq_a → q_a_norm → wq_b`.
- **All the weight quants** V4 ships — every IQ1/IQ2/IQ3/IQ4 and k-quant has a
  native Vulkan dense kernel.

Missing, and these are the real cost:

- **MLA attention.** `Op::Attention` carries a single `head_dim` shared by Q, K
  and V, and both the CPU and Vulkan implementations index caches as
  `rows × n_kv × head_dim`. MLA's absorbed form (K 576 wide, V 512 wide,
  `n_kv = 1`) is **not expressible**. Confirmed by grep: `mla`, `latent`,
  `kv_a`, `kv_b`, `lora_rank` appear nowhere under `crates/`.
- **Group-limited routing** (`n_group` / `topk_group`) — no field anywhere; the
  top-k shader is a flat global top-k.
- **Router bias correction** (`e_score_correction_bias` / `exp_probs_b`) — not
  loaded. This changes _which_ experts are selected, so ignoring it is wrong
  output, not a quality nudge. It also breaks an invariant the Vulkan
  `moe_topk.comp` documents: that both gating functions are monotone in the
  logit, so top-k-by-logit picks the same set. With a selection bias the pick
  must use the biased score and the weight must come from the unbiased one.
- **YaRN.** Zero occurrences of `yarn`, `rope_scaling`, `beta_fast`,
  `ext_factor` under `crates/`. (The hits in `ref/` are vendored llama.cpp
  sources, read-only, not compiled.)
- **Sparse attention / an indexer** — nothing analogous. `AttnMask` has only
  causal, sliding-window and the diffusion canvas.
- **`is_moe_layer` is periodic** (`(il+1) % step == 0`), but DeepSeek wants
  "first N dense, rest MoE" — a threshold. New branch.
- **Metal has no `MoeSharedExpertAdd` kernel** (`infr-metal/src/exec.rs` returns
  `Unsupported`), so any path using the _gated_ shared expert is Vulkan+CPU
  only. **Checked: this does not bite DeepSeek.** That op is only emitted for a
  per-token-sigmoid-gated shared expert (qwen35moe); an ungated one is summed in
  plain with `Op::Add`, which is the llama4 path and exists on all three
  backends — see `FfnW::Moe`'s `shexp` doc in `seam/weights.rs` and
  `Config::shexp_gated`.
- **Metal cannot run a DeepSeek MoE layer at all**, for a different reason than
  the one above. Its `Op::MoeFfn` arm implements softmax gating + top-k renorm +
  output-weighting and asserts on anything else, so V2-Lite
  (`norm_topk_prob = false`) already fails that assert, and V3 (sigmoid) fails
  it too. It also reads neither `exp_probs_b` nor the group-routing fields —
  softmax + renorm + `expert_group_count > 1` is a legal `deepseek2` config that
  used to pass the assert and then route with neither applied, silently; that
  combination now asserts as well. DeepSeek is CPU + Vulkan for the MoE layers
  until a Metal router gains those inputs. MLA attention itself IS implemented
  on Metal.

### Places a new arch string must be registered

1. `infr-llama/src/arch.rs` — the `pub const`, plus `arch::TRANSFORMER` and
   `arch::ALL`. Neither list gates anything: `TRANSFORMER` is read once to
   render the rejection message (`config.rs`), and `ALL` currently has **no
   consumer at all**. They are documentation that happens to compile — adding to
   them does not make the arch load.
2. `infr-llama/src/config.rs`, the `match arch.as_str()` inside
   `Config::from_gguf` — **this is the load gate**; an unknown arch fails here
   and nowhere else.
3. `infr-cli/src/main.rs`, `arch_sampling` — recommended sampling defaults
   (optional; falls back to `(0.6, 20, 0.95)`).
4. `infr-llama/src/tokenizer.rs`, the `match pre` — see below, this one is
   required for DeepSeek.

The two commits worth reading as templates are `5b44ef9` (BitNet — the floor for
a config-only arch, 5 files) and `e24399d` (llama4 Scout — the shape of an arch
that needs new op semantics, 15 files including `graph.rs` and all backends).

## Stage 0 — prerequisites, before any architecture

These are independent of which DeepSeek you start with, and two of them are
things you would otherwise discover as silent wrongness.

### 0.1 Dump a real GGUF first

`crates/infr-gguf/examples/dump.rs` exists for this. Run it on a real DeepSeek
GGUF before planning around anything below. Specifically confirm: the ggml type
ids used (anything outside `ggml_type_to_dtype` in `infr-gguf/src/lib.rs` fails
at `Gguf::open`), the exact metadata keys, and the head layouts.

### 0.2 Tokenizer — a real gap, not a formality

llama.cpp maps three DeepSeek pre-tokenizers (`src/llama-vocab.cpp`):

| `tokenizer.ggml.pre` | patterns | `clean_spaces` |
| -------------------- | -------: | -------------- |
| `deepseek-llm`       |        6 | false          |
| `deepseek-coder`     |        5 | false          |
| `deepseek-v3`        |        3 | false          |

V3 uses `deepseek-v3`; V4 reuses it (there is no `deepseek-v4` pre-type).

llama.cpp applies these as an **ordered list of successive splits**. `infr`
implements this in `build_multi_split_seq` (`infr-llama/src/tokenizer.rs`): N
`Split` pre-tokenizers in `Isolated` mode followed by `ByteLevel`, with the
regex lists in `infr-llama/src/util.rs` (`DEEPSEEK_LLM_PRE_RES`,
`DEEPSEEK_CODER_PRE_RES`, `DEEPSEEK_V3_PRE_RES`).

**Collapsing the list into one alternation is not equivalent** — and the
successive form is no longer an assumption either. It was checked against
`llama-tokenize` token ids; see open question 3, which is now resolved.

All fourteen regexes were diffed codepoint by codepoint against
`llama-vocab.cpp` (2026-08-09) and now match it exactly. Two did not, and the
symptom of each is instructive: a wrong character inside a class compiles, runs,
raises nothing, and merely moves a chunk boundary.

- `DEEPSEEK_LLM_PRE_RES[2]` opened its quote range with `'` (U+0027) where the
  reference has `‘` (U+2018), so the class ran U+0027–U+201F instead of the
  eight quote characters — most of the BMP, including the ASCII digits, Hebrew,
  Arabic and Devanagari.
- `DEEPSEEK_LLM_PRE_RES[1]` had `ℹ-ℿ` where the reference has `ℹℼ-ℿ`, adding
  U+213A and U+213B to the letter class.

`clean_spaces = false`, which llama.cpp sets for all three DeepSeek pre-types,
**needs no equivalent here**. It is read in exactly one place —
`llama_vocab::impl::detokenize` in `llama-vocab.cpp` — where it drops the space
before `?!.,`, strips a lone apostrophe between spaces, and closes up
`'s`/`'m`/`'re`/`'ve`. It is a detokenizer post-process (HF's
`clean_up_tokenization_spaces`) and never runs during encoding, so it cannot
affect token ids. llama.cpp defaults the flag to `false`, turns it ON for every
BPE vocab, and the DeepSeek arms turn it back off. `infr` detokenizes through
`tokenizers::Tokenizer::decode`, and that crate contains no such pass at all —
so `infr` is unconditionally in the `clean_spaces = false` state DeepSeek wants.
(It is therefore also unconditionally in that state for BPE pre-types where
llama.cpp leaves the flag ON. That is a display-only divergence, not an id one,
and it is out of scope here.)

The lists are guarded by
`deepseek_pre_split_boundaries_match_the_reference_lists` in `tokenizer.rs`,
which pins the chunk boundaries all three produce and needs no model file.

Getting this wrong degrades output with **no error at all**, which is why it is
stage 0.

### 0.3 Pager LRU

`docs/backlog.md` already records it: `Pager`'s `mark_mru`/`evict`/`take_slot`
are `O(n_slots)` per touch. At V4-Flash scale (256 experts × 43 layers ≈ 11k
blocks per role) that stops being acceptable. Not needed for stages 1–2; needed
before stage 4 is usable.

## Stage 1 — `deepseek` (V1)

**Smallest possible first step, and it buys a real model.** No MLA, no YaRN, no
indexer. Plain MHA plus DeepSeek-style MoE.

### Hyperparameters (`{arch}.` prefixed)

| key                                | maps to                    |
| ---------------------------------- | -------------------------- |
| `attention.layer_norm_rms_epsilon` | RMS eps                    |
| `leading_dense_block_count`        | `n_layer_dense_lead`       |
| `expert_feed_forward_length`       | `n_ff_exp`                 |
| `expert_shared_count`              | `n_expert_shared`          |
| `expert_weights_scale`             | routed scaling (default 0) |

V1 has **no** `expert_gating_func` and **no** `expert_weights_norm` — it
hardcodes softmax scoring and no normalisation.

### Attention

Vanilla MHA. `n_embd_head_v == n_embd_head_k == n_rot` (llama.cpp asserts it).
Full-dim rope on Q and K, rope type **NORM** (interleaved consecutive pairs).
`kq_scale = 1/sqrt(n_embd_head)`.

**Do NOT set `Config::permute_qk_neox`.** All five DeepSeek arches return
`LLAMA_ROPE_TYPE_NORM` from `llama_model_rope_type` (`src/llama-model.cpp`) —
the same arm as `LLM_ARCH_LLAMA`, not the NEOX arm that `LLM_ARCH_QWEN2` sits
in. The permute exists to make infr's interleaved (NORM) `Op::Rope` reproduce
**NEOX** for an arch whose GGUF stayed in HF rotate-half order (qwen2, bitnet —
see the field's own doc in `config.rs`). DeepSeek is already NORM with
converter-permuted rows, exactly like llama, so permuting would rotate the wrong
pairs and produce fluent nonsense. This applies to stage 2's `q_pe`/`k_pe` as
well; the one NEOX rope in the family is the V3.2 indexer (stage 3), which is
hardcoded NEOX against a NORM main rope.

Nothing new is required in the IR.

### MoE

`softmax(logits)` → top-k → gather → **no** normalisation → `× scale` → SwiGLU
experts → add shared expert. All of this exists.

### New work

- Arch registration (§ above).
- `is_moe_layer` needs a **first-N-dense threshold** mode alongside the existing
  periodic one.
- Shared-expert width: llama.cpp allocates `ffn_*_shexp` as
  `{n_embd, n_ff_exp * n_expert_shared}` — one fused branch of `n_expert_shared`
  experts' width. `infr` models exactly one branch of width `shexp_ff`, so
  setting `shexp_ff = n_ff_exp * n_expert_shared` should line up. **Verify
  against a real GGUF** — V2-Lite has `n_shared_experts = 2` and is the first
  case where this matters.

### Done when

- `cpu_deepseek_config` — opens the GGUF, asserts the arch string, asserts every
  gate boolean including that other arches' gates are false. (Pattern:
  `cpu_bitnet_config` in `infr-llama/tests/cpu_backend.rs`.)
- `cpu_deepseek_prefill_paris` — top-1 token after "The capital of France is".
  (Pattern: `cpu_bitnet_prefill_paris`.)
- `gpu_seam_matches_cpu_deepseek` — token-identical CPU vs Vulkan, `#[ignore]`d
  behind a GPU. Use the **strict** form for the 7B dense model and the **loose**
  form (top-5 overlap + `cosine > 0.5`) for MoE-16B, for the same reason
  `gpu_seam_matches_cpu_qwen35moe` does: routing near-ties legitimately flip.

**No CI golden.** The `cpu-goldens` job downloads two ~1B GGUFs; the smallest
DeepSeek is 7B. Commit `273f8d4` already removed qwen35 from that job because an
exact-token golden did not reproduce across machines.

## Stage 2 — `deepseek2` (V2, V2-Lite, V3, V3.1)

The big one, and the one with a 16 GB development model. Everything here is
reused by stage 3 almost verbatim.

### Hyperparameters

Adds to stage 1: `attention.q_lora_rank`, `attention.kv_lora_rank`,
`attention.key_length_mla`, `attention.value_length_mla`, `expert_weights_norm`,
`expert_gating_func` (1 softmax / 2 sigmoid / 3 softmax-on-weights / 4
sqrt-softplus), `rope.scaling.yarn_log_multiplier`, `expert_group_count`,
`expert_group_used_count`.

Derived: `key_length = kv_lora_rank + qk_rope_head_dim` (576 for V3),
`value_length = kv_lora_rank` (512), `head_count_kv = 1`,
`rope.dimension_count = qk_rope_head_dim` (64). **Read these from a GGUF rather
than trusting the numbers here** — they were derived from the conversion
script's formulas, not read out of a file.

Two loader details worth copying exactly:

- **`rope_yarn_log_mul /= 0.1`** on load. The convert script writes
  `0.1 * mscale_all_dim`; the loader divides it back out. Double-applying or
  double-cancelling this is a silent long-context quality bug.
- **"lite" detection.** llama.cpp has a heuristic on layer count and vocab size,
  but the graph actually decides on **tensor presence** (`wq` present ⇒ lite,
  else `wq_a`/`wq_b`). Port the tensor-presence test and drop the heuristic.

### Tensors

Per layer, beyond the stage-1 set:

| tensor            | GGUF name                      | shape                                       |
| ----------------- | ------------------------------ | ------------------------------------------- |
| `wq_a`            | `blk.%d.attn_q_a.weight`       | `{n_embd, q_lora_rank}`                     |
| `attn_q_a_norm`   | `blk.%d.attn_q_a_norm.weight`  | `{q_lora_rank}`                             |
| `wq_b`            | `blk.%d.attn_q_b.weight`       | `{q_lora_rank, n_head * head_k_mla}`        |
| `wq` (lite)       | `blk.%d.attn_q.weight`         | `{n_embd, n_head * head_k_mla}`             |
| `wkv_a_mqa`       | `blk.%d.attn_kv_a_mqa.weight`  | `{n_embd, kv_lora_rank + qk_rope}`          |
| `attn_kv_a_norm`  | `blk.%d.attn_kv_a_norm.weight` | `{kv_lora_rank}`                            |
| `wk_b`            | `blk.%d.attn_k_b.weight`       | `{qk_nope, kv_lora_rank, n_head}`           |
| `wv_b`            | `blk.%d.attn_v_b.weight`       | `{kv_lora_rank, v_head_dim, n_head}`        |
| `wo`              | `blk.%d.attn_output.weight`    | `{n_head * v_head_dim, n_embd}`             |
| `ffn_exp_probs_b` | `blk.%d.exp_probs_b.bias`      | `{n_expert}` — optional, V3's noaux_tc bias |

**`wk_b` is transposed relative to the HF weight and `wv_b` is not.** The
conversion script splits `kv_b` into `k_b`/`v_b` and calls `.transpose(1, 2)` on
`k_b` only. This is the classic MLA porting bug — getting it backwards produces
plausible-looking garbage.

### The attention, step by step

Head layout is **`[nope | rope]`, nope first**, on Q. The KV projection is
`[latent(512) | rope(64)]`.

```
q      = wq_b · RMSNorm_{q_a_norm}(wq_a · x)      # or wq · x when lite
q_nope = q[0 .. qk_nope]                          # per head
q_pe   = q[qk_nope .. qk_nope+qk_rope]

kv_cmpr_pe = wkv_a_mqa · x                        # {512+64, n_tokens}
kv_cmpr    = kv_cmpr_pe[0 .. 512]                 # the latent
k_pe       = kv_cmpr_pe[512 .. 576]               # ONE rope head, shared by all query heads

q_pe = rope(q_pe, n_rot=64, NORM)
k_pe = rope(k_pe, n_rot=64, NORM)
kv_cmpr = RMSNorm_{kv_a_norm}(kv_cmpr)            # AFTER the split, BEFORE absorption
```

`attn_kv_a_norm` applies **only** to the 512-wide latent, not to `k_pe`.
`attn_q_a_norm` sits **only** between `wq_a` and `wq_b`.

Then the absorbed form:

```
q_nope_absorbed = wk_b[h]ᵀ · q_nope[h]            # {128} -> {512}, per head
Q = concat(q_nope_absorbed, q_pe)                 # {576, n_head}
K = concat(kv_cmpr, k_pe)                         # {576, 1}
V = kv_cmpr                                       # {512, 1}  -- an ALIASED PREFIX VIEW of K
out = wv_b · attn(Q, K, V)                        # wv_b applied to the OUTPUT, {512}->{128} per head
```

Three things that will each silently corrupt output:

1. **Only one 576-wide row is cached per token per layer.** There is no separate
   V cache; V is the first 512 columns of K.
2. **`wv_b` is applied after the KQV product**, not to the cache.
3. **`kq_scale` divides by `head_k_mla` (192), not by 576** and not by the
   concatenated Q width.

There is also an "unabsorbed" legacy path for older GGUFs carrying `wkv_b`
instead of `wk_b`/`wv_b`: it up-projects to full MHA, broadcasts `k_pe` across
all heads, and uses ordinary K/V caches. **Skip it** unless a target GGUF needs
it — it is a much larger cache for identical output.

### YaRN

Two stages that must not double-apply:

```
attn_factor_org = attn_factor * (1 + 0.1·ln(1/freq_scale))
mscale          = attn_factor_org * (1 + 0.1·rope_yarn_log_mul·ln(1/freq_scale))
kq_scale        = mscale² / sqrt(head_k_mla)
```

The mscale² is folded into the **softmax scale**, not applied to Q, because
ggml's rope already applies `attn_factor` to the rotated slice and the first
line undoes it.

For `infr`, the frequency ramp itself should fold into the existing
`freq_factors` mechanism (`Op::Rope`'s optional per-pair divisor, already used
by gemma4's proportional rope): precompute the ramp on the host at load and bind
it like `rope_freqs.weight`. **This is an inference from the maths, not
something either codebase states — validate numerically against llama.cpp before
relying on it.**

### MoE

Order matters:

1. `logits = gate_inp · x`
2. score: softmax / sigmoid / sqrt-softplus per `expert_gating_func`
3. **`selection_probs = probs + exp_probs_b`** — bias affects _selection only_;
   the returned weights are read from the **unbiased** `probs`
4. group-limited routing when `n_expert_groups > 1`: per group take the **top
   2** scores and sum them for a group score, take the top `n_group_used`
   groups, mask the rest to `-inf`. The top-2-within-group is **hardcoded** in
   llama.cpp and matches V3, but is not the general `topk_group` formulation.
5. top-k over `selection_probs`
6. gather weights from `probs`
7. if `norm_w`: divide by `clamp(sum, 6.103515625e-5, inf)` — the clamp is the
   smallest normal f16
8. if `scale != 0 && != 1`: multiply
9. SwiGLU experts, weighted sum
10. add the shared expert; first `n_layer_dense_lead` layers instead run a plain
    dense SwiGLU with `n_ff` (not `n_ff_exp`) and no shared expert

### New work

- **MLA in the IR.** `Op::Attention` needs asymmetric K/V dims (or a new
  `Op::Mla`). This touches KV-cache sizing (`seam/mod.rs`, `seam/weights.rs`,
  `runner.rs` all compute `layer_n_kv * layer_head_dim`) and every attention
  kernel on CPU, Vulkan and Metal. **This is the single biggest item in the
  whole plan and it is worth prototyping on CPU alone first.**
- `MoeFfn` gains an `exp_probs_b` input and group-routing fields; the Vulkan
  `moe_topk.comp` must select on the biased score and weight from the unbiased.
- YaRN ramp precomputation at load.
- A new `MixerW::Mla(MlaW { .. })` variant plus a third branch in each of the
  three lockstep loops in `runner.rs` (`wload`, `wpush`, emit). The file's own
  comments say these MUST mirror; getting them out of step is silent corruption.

### Done when

- [x] Config + CPU-finite + CPU-top-token tests on V2-Lite
      (`cpu_deepseek2_config`, `cpu_deepseek2_prefill_finite`,
      `cpu_deepseek2_prefill_paris` — all added 2026-08-06, gated behind model
      file).
- [x] `gpu_seam_matches_cpu_deepseek2` — skeleton added 2026-08-06; passing on
      the GPU box 2026-08-07 (CPU-vs-Vulkan cosine 0.9955, matching top-5) after
      the YaRN ramp + wk_b/wv_b orientation fixes.
- [x] `cpu_deepseek2_golden` — hash-locked generation, blessed from the coherent
      post-fix output (2026-08-07).
- [x] **An op-level MLA parity test** in `infr-llama/tests/seam_op_parity.rs`
      against a hand-written CPU reference, following `deltanet_parity`. This is
      the one that matters: it is the only cheap check that survives into stages
      3–4.
- [x] **A numeric YaRN check against llama.cpp at a long context** — done
      2026-08-07 on the V2-Lite Q4_K GGUF vs llama.cpp `c629da5` (CPU
      reference):
  - 228-token prompt, infr CPU prefill vs llama.cpp last-row logits: cosine
    0.978, greedy token identical (185).
  - 4560-token prompt (positions past `n_ctx_orig`=4096, in the ramp region),
    infr Vulkan prefill vs llama.cpp: cosine 0.860, greedy token identical
    (549). The seam's ff divisors / mscale are context-independent (llama.cpp
    runs the full ramp at every context length), so the short-CPU and long-GPU
    runs exercise the same numbers; both greedy tokens match.
  - Both cosines sit in the established deepseek2 infr-vs-llama.cpp range
    (~0.79–0.91; MLA adds f16 cache + norm + rope stages per layer).
- [x] Metal MLA kernel — `mla_f16kv` in `attention.metal` + `exec.rs` dispatch
      (2026-08-06; ported from `mla.comp`, f16 KV cache), plus the YaRN
      `mla_f16kv_ff` twin. Executed for the first time by `mla_parity` /
      `mla_ff_parity` in the Metal parity suite on the macOS CI job (2026-08-07)
      — which also caught and fixed an ff/params buffer-index swap in the kernel
      declaration.
- [x] YaRN per-dimension frequency ramp in `Op::Rope` and MLA kernels — the
      `freq_factors` divisors (`ff[p] = 1/s(p)` from the corr_dims spectral
      ramp) + the constant `mla_scale = mscale²/√(qk_nope+qk_rope)` landed in
      `784704e` (2026-08-07); verified numerically above.

## Stage 3 — `deepseek32` (V3.2)

**~80% of this is stage 2 copied verbatim.** llama.cpp's `deepseek32.cpp` is
deepseek2's absorbed MLA path plus the lightning indexer. Non-MLA is rejected
outright. No small model exists; budget for slow iteration.

Adds: `attention.indexer.head_count`, `attention.indexer.key_length`,
`attention.indexer.top_k`, and `f_norm_eps` hardcoded to `1e-6`.
`expert_gating_func` is **mandatory** here (no fallback), and `q_lora_rank` is
mandatory (no lite variant).

### The lightning indexer

Per layer, unconditionally. It computes a scalar relevance score per (query
token, key token) and keeps the top-k keys for the real attention.

```
w[h, t]     = (indexer_proj · x)[h, t] / sqrt(index_head_dim · index_n_heads)
score[t, j] = Σ_h  w[h, t] · ReLU( q[h, t] · k[j] )  + causal_mask[t, j]
top_k       = argsort_top_k(score, min(n_kv, index_topk))
```

Note: **one key head shared by all indexer query heads** (MQA), the **ReLU is
inside the head-weighted sum**, and the `1/sqrt(d·H)` normaliser is pre-folded
into `w` to avoid scaling a huge score tensor.

New tensors: `indexer.k_norm.{weight,bias}`, `indexer.proj.weight`,
`indexer.attn_k.weight`, `indexer.attn_q_b.weight`.

Traps, each of which produces silent wrongness:

- **The indexer's rope type is NEOX, hardcoded**, while the main MLA rope is
  NORM. Same width, same frequencies, different pairing.
- **The indexer head layout is `[rope | nope]`** — the _opposite_ of the MLA
  head. Worse, llama.cpp writes the nope view's offset as `row_size(nope)`
  rather than `row_size(rope)`; these coincide only because both are 64 for
  V3.2. Port it as "offset = rope width" and assert it.
- `indexer_k_norm` is a real **LayerNorm with bias** (mean-centred), the only
  non-RMS norm anywhere in the family. **Confirmed: `infr` has no LayerNorm op**
  — `graph.rs` carries `RmsNorm` and `RmsNormAdd` and nothing mean-centred, so
  this is a new op on CPU, Vulkan and Metal, not a config flag.
- The indexer keeps a **second, independent KV cache**: one
  `index_head_dim`-wide row per token per layer, on top of the 576-wide MLA
  cache.
- A **Hadamard rotation** is applied to q and k. It is an orthogonal transform
  applied identically to both, so dot products are preserved: it exists for
  quantisation friendliness and **can be skipped entirely** in an unquantised
  port.

### How top-k feeds attention

llama.cpp does **not** gather or compact. It builds a `-inf` mask everywhere
except the selected indices, adds it to the ordinary causal mask, and runs dense
attention over the full `n_kv`. The FLOP saving is not realised — only the
numerics are faithful.

**This is the interesting decision for `infr`.** A port that wants the actual
speedup must gather, and the selected indices are per (query token, stream), not
per head. Doing the mask version first is the safe order: it is checkable
against llama.cpp token-for-token, and the gather can follow as a pure
optimisation.

## Stage 4 — `deepseek4` (V4-Flash / V4-Pro)

A genuinely different architecture, not an increment. Sharing with stage 2 is
limited to the MoE block, the FFN, norms, and generic rope/embedding plumbing.

**V4 is not MLA.** There is no `kv_lora_rank`, no `wk_b`/`wv_b`. Instead:

1. **Single-head MQA KV** — `wkv` is `{n_embd, n_embd_head}`, one KV head for
   all query heads. The Q path keeps its LoRA (`wq_a`/`q_a_norm`/`wq_b`) and
   adds an **unweighted per-head RMS-norm on Q** with no analogue in V2/V3.2.
2. **Low-rank grouped output projection** — `wo_a` + `wo_b` over
   `attention.output_group_count` groups.
3. **Attention sinks** — `attn_sinks {n_head}`.
4. **De-roping of the attention output** — the rope slice of the output is
   rotated _backwards_ by the query position before the output projection
   (`ggml_rope_ext_back`). Nothing else in the family does this.
5. **Hyper-connections** — `hc_mult` parallel residual streams with learned
   Sinkhorn-normalised mixing, replacing `x = x + f(x)` everywhere.
6. **Three-tier per-layer attention** keyed on
   `compress_ratios[il] ∈ {0, 4, 128}`.
7. **Compressor blocks** that softmax-pool blocks of tokens into single KV rows.
8. **Hash-routed MoE** on the first `hash_layer_count` layers.
9. **`sqrt(softplus)` gating**, mandatory.
10. **Per-layer SwiGLU clamping**, with V4 clamping the gate **pre-SiLU** where
    every other arch clamps post-SiLU.

No dense-lead layers, no NextN.

### `compress_ratio` is the master per-layer switch

`hparams.set_swa_pattern(0)` makes **every** layer sliding-window, so long-range
recall comes exclusively from the compressed caches.

| ratio | flavour                 | caches                                                |
| ----: | ----------------------- | ----------------------------------------------------- |
|     0 | pure sliding window     | raw SWA only                                          |
|     4 | CSA + lightning indexer | raw SWA + CSA(4:1) + LID(4:1) + two compressor states |
|   128 | HCA                     | raw SWA + HCA(128:1) + compressor state               |

Only `{0, 4, 128}` are accepted. Compressed layers use YaRN at
`compress_rope_theta`; ratio-0 layers use plain unscaled rope. `kq_scale` is
plain `1/sqrt(n_embd_head)` at all three call sites — none of stage 2's mscale²
games.

V4's indexer differs from V3.2's in one structural way: **there is no
`indexer_attn_k` and no `indexer_k_norm`.** The indexer keys come from the
compressor, so `index_topk` counts _compressed blocks_, not tokens.

### Sinkhorn hyper-connections

The residual stream is widened to `hc_mult` copies. Each sublayer is wrapped
`pre → sublayer → post`, where one matmul produces three chunks — `pre` (stream
collapse weights), `post` (per-stream output gates) and `comb` (an `hc × hc`
mixing matrix). `comb` is made approximately doubly-stochastic by Sinkhorn
iteration, so no stream's mass blows up or vanishes with depth.

```
comb = softmax(comb) + eps          # softmax over dst
norm_cols()                          # then n_iter column normalisations
for i in 1..n_iter: norm_rows(); norm_cols()
```

Then `out[i, dst] = x[i]·post[dst] + Σ_src residual[i, src]·comb[dst, src]`.

**Expect to get this wrong twice.** The index formula is
`logits[dst, src, t] = mixes[2·hc + dst + hc·src, t]·scale + base[...]`; the
loop is asymmetric (`n_iter` column normalisations, `n_iter − 1` row); eps is
added in three distinct places; and llama.cpp's own lambda names
`norm_rows`/`norm_cols` are **inverted** relative to its header's `dst`/`src`
vocabulary. Trust the index formula and the lambda bodies, not the names.

### The compressed-KV state machine

This is the **largest single porting risk in the family**. Seven cache
structures, three compressor states holding in-flight partial blocks, an
overlapping 2×ratio pooling window with `-inf` sentinel rows, absolute
position-in-block embeddings indexed by `pos % ratio`, and per-channel softmax
pooling. The index planning lives in `llama-kv-cache-dsv4.cpp` (1978 lines),
which **was not read in full** — the graph code is only meaningful given those
index plans, and the boundary conditions (partial block at the end of a prefill,
how visible length interacts with padded `n_kv`) are unspecified in what was
reviewed.

Budget stage 4 accordingly, and do not start it until stages 2–3 are solid.

## Open questions — check these before trusting the above

Ordered by how much damage a wrong assumption does.

1. **Head layouts and exact dims** — everything here about
   `192 / 576 / 512 / 64 / 128` came from conversion-script formulas, not from a
   GGUF. Dump a real file.
2. **ggml type ids in V4 GGUFs** — if any weight type falls outside
   `ggml_type_to_dtype`, the file fails at open and needs a new `DType`,
   `block_spec` and `dequant_block` arm. The i2_s commit `dbc8431` is the
   template.
3. **Whether N successive splits reproduce llama.cpp's tokenizer** (§0.2) — ✓
   RESOLVED (2026-08-09), for `deepseek-llm` only. `infr`'s ids were compared
   against `llama-tokenize --ids --no-bos --no-escape` on
   `deepseek-v2-lite-chat-q4_k_m.gguf`. Note that this GGUF is
   `tokenizer.ggml.pre == "deepseek-llm"`, **not** `deepseek-v3` — so what it
   exercises is the six-regex V1 list, the one that carried both transcription
   slips. They agreed **exactly on all 31 texts**, covering digits, decimals and
   grouped numbers, CJK, Hangul, Greek, Hebrew, Arabic, Devanagari, punctuation
   runs, code, emoji, CRLF, smart quotes and non-ASCII whitespace. So N
   successive `Isolated` splits do reproduce `unicode_regex_split` here.

   The comparison was shown to be capable of failing: re-introducing the U+0027
   slip made `infr` disagree with llama.cpp on 8 of 11 texts in the
   NBSP-before-punctuation battery (e.g. `"a \u{00A0}. b"` → llama.cpp
   `[64, 207, 1202, 13, 270]`, broken `infr` `[64, 30683, 13, 270]`).

   **`deepseek-coder` and `deepseek-v3` have no token-id coverage.** Both GGUFs
   in the local HF cache are `deepseek-llm`, so neither of those lists was
   exercised against real ids. They are structurally identical (same
   `build_multi_split_seq`) and byte-identical to the reference, and their chunk
   boundaries are pinned by the unit test — but that is not the same as having
   been checked against llama.cpp. Re-open this for either list if a matching
   GGUF appears.

4. **Shared-expert width when `n_shared_experts > 1`** — V2-Lite has 2.
5. **`rope_off`** — ✓ RESOLVED (2026-08-06). `Op::Rope` only rotates standalone
   k_pe slices (extracted via `CopyStrided`, no nope prefix). The q_pe rope is
   done inside the MLA kernel at offset `qk_nope_dim` — the offset lives in the
   kernel, not in `Op::Rope`. No `rope_off` field needed.
6. **YaRN** — RESOLVED (2026-08-07). The per-dimension frequency ramp IS
   implemented and the mscale² is a constant (both folded per `ggml_rope_yarn` +
   `deepseek2.cpp:162-172`). The earlier note claimed the ramp is "INERT for
   default deepseek2 GGUFs" because it assumed the convert script never writes
   `rope.scaling.factor`/`type` — **wrong**: the V2-Lite Q4_K GGUF declares
   `rope.scaling.type = yarn`, `factor = 40`, `original_context_length = 4096`,
   `yarn_log_multiplier = 0.0707`, which makes llama.cpp set
   `yarn_ext_factor = 1.0` (llama-context.cpp:189-191) and run the FULL ramp at
   every context length. Without it, infr's greedy output was
   `"Reply Collabor…"` garbage while llama.cpp produced coherent text. The ramp
   lives in `Op::Rope.freq_factors` (per-pair divisors, computed in the seam
   from the corr_dims spectral ramp) plus the MLA kernels' internal q_pe rope;
   the mscale² is folded into the MLA attention scale as a constant
   (`mscale = 1 + 0.1·log_mul·ln(factor)`, applied via
   `mla_scale = mscale²/√(qk_nope + qk_rope)` — note `qk_nope = head_k_mla` is
   128 for V2-Lite, so the denominator is √192, not √576). The rope vector
   mscale cancels to `rope_attn_factor` for deepseek2, so no vector scaling is
   needed in the kernels.
7. **DeepSeek's EOS** — `add_chat_eos` appends a fixed list that does not
   include `<｜end▁of▁sentence｜>`. It is normally the GGUF's declared
   `tokenizer.ggml.eos_token_id` and therefore already in `eos_ids`, but check
   whether the chat template ends turns on something else.
8. **`LLM_TENSOR_ATTN_KV_NORM` and `LLM_TENSOR_ATTN_KV_A_NORM` share the on-disk
   name** `blk.%d.attn_kv_a_norm`. Two enum values, one string — not
   distinguishable on disk.
9. **llama.cpp's V4 support is young.** Its model-type detection for V4 is a
   stub where both branches return `UNKNOWN`. Treat the reference as possibly
   buggy rather than authoritative.

## What was not covered

- `deepseek2-ocr` — out of scope.
- The **DSpark speculative module** (`dspark_block_size`,
  `dspark_target_layer_ids`, `dspark_markov_rank`). It is a separate head over
  V4's last three layers and does not appear in the graph builder at all. `infr`
  has MTP machinery (`docs/mtp.md`) that may host it; not investigated.
- V3.2's **NextN** tensors — loaded but skipped by llama.cpp.
- Performance. This plan is about correctness only. Nothing here is measured,
  and no throughput claim is made.
