# MTP (multi-token prediction) speculative decoding — qwen35 single-head

Issue #33. Reference: llama.cpp master (`--spec-type draft-mtp`, merged
2026-05-16 from PR #22673): `common/speculative.cpp`
`common_speculative_impl_draft_mtp` (the driver) and `src/models/qwen35.cpp`
`graph_mtp` (the head graph). Model: `unsloth/Qwen3.5-4B-MTP-GGUF:UD-Q4_K_XL` —
the `nextn.*` head tensors are baked into the MAIN GGUF (no sibling file for
qwen35; the `--mtp`/`mtp-*.gguf` sibling download flow is for other arch
families). llama.cpp reports ~1.5-2× generation speedup.

> **Status: LANDED.** The single-head MTP path shipped in `infr` — opt-in via
> `INFR_MTP=1`, wired for CPU + Vulkan (`crates/infr-llama/src/mtp/`,
> `chat/{cpu,vulkan}.rs`), with temp-aware stochastic spec-accept,
> token-identical to the target-only greedy path. Tests: `mtp_*` in
> `crates/infr-llama/tests/cpu_backend.rs`. The Phase 1–3 build plan below is
> kept as the design record (history, not a TODO).

## The head, exactly (qwen35: ONE MTP layer, `n_layer_nextn = 1`)

Tensors at `blk.{n_layer}.nextn.*` (the layer index AFTER the trunk), plus a
full standard qwen35 ATTENTION-layer tensor set at the same index:

- `nextn.eh_proj [2*ne, ne]`, `nextn.enorm [ne]`, `nextn.hnorm [ne]`
- `nextn.embed_tokens [ne, vocab]` (optional — falls back to the main tok_embd)
- `nextn.shared_head_head [ne, vocab]` + `nextn.shared_head_norm [ne]` (optional
  — fall back to the main output/output_norm)
- `attn_norm/attn_q (interleaved q+gate!)/attn_k/attn_v/attn_q_norm/attn_k_norm/ attn_output/attn_post_norm/ffn_gate/ffn_up/ffn_down`
  — the EXACT layer shape infr's unified runner already executes for qwen35
  full-attention layers (interleaved per-head q/gate split, sigmoid out-gate,
  qk-norm, m-rope sections ≡ NEOX at a single position, SwiGLU).

Forward for one draft row `(token t_{p+1}, target hidden h_p, position p+1)`:

```
e = rmsnorm(embed(t_{p+1}), enorm)        # embed from nextn.embed_tokens (or main)
h = rmsnorm(h_p, hnorm)                   # h_p = target's POST-output_norm hidden at p
x = eh_proj @ concat([e; h])              # [2ne] -> [ne]
x = qwen35_attention_layer(x, pos=p+1)    # own KV cache; standard causal attention
h_mtp = rmsnorm(x, shared_head_norm || output_norm)   # ALSO fed back when chaining drafts
logits = (shared_head_head || lm_head) @ h_mtp
```

**`h_p` is the target's hidden state AFTER `output_norm`** — exactly the lm_head
input the runner already materializes (`res->t_h_nextn` in the reference; the
`llama_get_embeddings_nextn` staging API reads it per batch row).

## The driver (single-head mode: `n_mtp_layers==1`, not mem-shared, not chained)

Two hooks around the ordinary target decode:

1. **`process(batch)` — catch-up, after EVERY target prefill/decode ubatch**:
   decode the same tokens through the MTP layer with the h input SHIFTED RIGHT
   by one (`embd[i] = h_tgt[i-1]`, `embd[first] = pending_h` carried from the
   previous batch), so the MTP layer's own KV always covers all committed
   positions. Afterwards stash `pending_h = h_tgt[last]`. (This is why MTP needs
   the target's h for EVERY prompt/verify row, not just sampled rows.)
2. **`draft(id_last, n_past)`**: feed `(id_last, pending_h)` at `pos = n_past` →
   greedy-sample the MTP logits. Loop: append ONLY the new token (the MTP KV
   grows), with `h` = the MTP head's OWN `h_mtp` from the previous draft row
   (the head self-chains during drafting). Stop when the top prob `< p_min`
   (only high-confidence drafts) or `n_max` tokens drafted (llama.cpp example:
   `--spec-draft-n-max 6`).

Verification is the ordinary spec verify (one batched target forward over the
draft, longest accepted prefix) — infr already has this (`spec_accept`, the
multi-row verify path). The next `process()` call over the verify batch
naturally re-syncs the MTP KV to whatever was accepted (draft-region KV rows are
simply overwritten at the same positions).

## infr build plan

- **Phase 1 — tap + weights**: parse `{arch}.nextn_predict_layers` + the
  `blk.{n_layer}.nextn.*`/extra-layer tensors into Config/weights (qwen35 only,
  `n_layer_nextn==1` enforced like the reference); add an opt-in per-call output
  of the post-`output_norm` hidden rows from `generate_dense_backend` (the
  `logits_out`-style hook, one op earlier — rows × ne f32). Validate: tensors
  load; `lm_head(h_row) == logits_row` consistency on the same forward.
- **Phase 2 — head forward + draft loop**: the MTP layer as a 1-layer graph
  through the unified runner machinery (own 1-layer KV bufs; reuse the qwen35
  attention emission verbatim), the catch-up + pending_h + draft loop
  (p_min/n_max) as an engine-level driver. CPU + Vulkan. Validate: drafted
  tokens/probs vs the llama.cpp oracle on fixed prompts (greedy), CPU==Vulkan.
- **Phase 3 — spec integration + surfaces**: wire into the existing verify
  machinery for run/serve (spec ≡ target-only greedy invariant, the bar the
  existing spec tests pin), INFR toggles, bench acceptance-rate + tok/s vs
  non-spec, docs + this file's status.

Later: gemma4 mem-shared mode (no separate KV, one graph), a qwen35moe MTP head
(the arch itself has since landed; the MTP head for it has not), chained-head
models (step35).

## Oracle commands (llama.cpp master, Vulkan build at

`~/Projects/mxaddict/llama.cpp/build/bin`)

```bash
llama-cli -m Qwen3.5-4B-MTP-UD-Q4_K_XL.gguf -ngl 99 --spec-type draft-mtp \
  --spec-draft-n-max 6 -p "..." -n 64 --temp 0   # MTP on
llama-cli -m ... -n 64 --temp 0                   # baseline, same binary
```

Captured 2026-07-05 (CPU build — no Vulkan headers on the box; the relative win
is what matters): "What is the capital of France?"
`-n 48 --temp 0 --single-turn`:

| mode                | generation |
| ------------------- | ---------- |
| baseline (no spec)  | 20.5 t/s   |
| `draft-mtp` n_max=6 | 41.0 t/s   |

**2.0× generation, byte-identical output** (diff of the two runs shows only
loading-spinner frames) — the spec ≡ target-greedy invariant holds in the
reference, and it's the bar infr's implementation must meet.

## Confirmed GGUF facts (Qwen3.5-4B-MTP UD-Q4_K_XL dump)

- `qwen35.block_count = 33` **INCLUDES the MTP layer** (32 trunk + 1 head at
  `blk.32`); `qwen35.nextn_predict_layers = 1`. **infr's Config today would
  treat blk.32 as a DeltaNet layer** ((32+1)%4 ≠ 0) and fail on missing ssm
  tensors — Phase 1 must set trunk
  `n_layer = block_count − nextn_predict_layers` and stash the head layer.
- blk.32 = full qwen35 attention layer (interleaved q+gate `attn_q [2560,8192]`,
  4 kv × hd 256, q/k norms, `attn_output [4096,2560]`, post_attention_norm,
  SwiGLU ffn 9216) + `nextn.eh_proj [5120,2560] Q8_0`,
  `nextn.enorm/hnorm/ shared_head_norm [2560] F32`. NO `nextn.embed_tokens` /
  `shared_head_head` → main tok_embd + tied lm_head fallbacks are the live path.
- 4B trunk: ne=2560, full-attn every 4th layer (3,7,…,31), DeltaNet elsewhere
  (ssm inner 4096, ts_rank 32 — larger than the 0.8B but same shape family).

---

## MTP is parked (2026-07-31)

**MTP self-speculative decode is DISABLED** (`infr_llama::mtp::mtp_enabled` is
the single kill-switch, and carries the full rationale). `INFR_MTP=1` is ignored
with a warning, and the MTP-head GGUFs (Qwen3.5-\*-MTP) run the **ordinary**
decode path — their `nextn` tensors are simply unused, which is harmless. Those
models are otherwise fully supported; only the speculative path is off.

Why: MTP's contract was that its output is **token-identical to non-speculative
greedy** — a pure speedup, not a quality trade. That no longer holds. The
int8-activation decode kernels every fast dtype now uses carry small per-token
rounding noise, and MTP's verify batch and the plain-decode chain it must match
are computed at **different sequence positions with different KV state**. The
same noise plain decode absorbs harmlessly is enough to flip a close-margin
greedy argmax between the two streams, so `mtp_spec_matches_target_only_greedy`
fails. Notably this is **not** a bit-identity bug (`mmv_row1_bit_identical`
passes — decode and verify share one kernel) and **not** an accuracy cliff (all
13 `gpu_seam_matches_cpu_*` pass; output stays coherent).

That guarantee was holding the rest of the engine hostage: it blocked Q6_K's
int8 decode tier (+10% decode, +34% prefill) on a speculative path that was
already our slowest row (0.59–0.78× vs llama.cpp). So MTP is parked and the
kernel wins ship. The identity test is `#[ignore]`d, **not weakened** — the
assertion is correct; re-enabling MTP means making it pass again, which needs an
accuracy mitigation (e.g. re-verify in f32 when the top-2 logit margin is
tight), not faster kernels.
