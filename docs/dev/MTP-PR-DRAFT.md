# Enabling MTP Speculative Decoding for qwen35/qwen35moe on MoE4All

## Summary

This PR enables MTP (Multi-Token Prediction) speculative decoding for `qwen35` and `qwen35moe` architectures on the Vulkan backend with paged-MoE expert streaming. On an RX 7700 XT 12GB running Ornith-1.5-35B-A3B (22GB, Q6_K/Q5_K/IQ4_XS mixed experts), greedy MTP achieves **93.6 tok/s vs 54.7 tok/s baseline (1.71×)** with α=1.0 using a properly trained MTP head.

## The Problem

MoE4All's MTP implementation (issue #33, `docs/mtp.md`) was complete for `qwen35` (dense) but had three blockers preventing it from working on `qwen35moe` (MoE) models:

1. **Head loading**: `load_mtp_head` only accepted `cfg.qwen35` with dense FFN; qwen35moe heads have MoE FFN (routed experts + shared expert)
2. **Verify path**: `generate_dense_backend`'s VERIFY branch bailed on `c.moe.is_some()`
3. **Naive weight upload**: the MTP driver's bind closure raw-uploaded ALL trunk weights to VRAM (OOM for models larger than VRAM)

Additionally, the shipped Ornith-1.5 MTP head was never trained (random init — see [shisa-ai's analysis](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY)), so a properly trained head (shisa-ai's KL-distilled graft) was used for validation.

## Changes

### Stage 1: qwen35moe MTP head support (`crates/infr-llama/src/mtp/mod.rs`)

- `MtpFfn` enum: `Dense { gate, up, down }` | `Moe { gate_inp, gate_exps, up_exps, down_exps, shexp }`
- `load_mtp_head`: accepts qwen35moe via `cfg.moe.is_some()`, shapes derived from Config
- `upload_mtp_head_bufs`: variable weight list per FFN variant
- `build_mtp_graph` + `build_mtp_draft_chain_graph`: emit `Op::MoeFfn` + shared expert + `Op::MoeSharedExpertAdd` for MoE variant

### Stage 2: Verify path + rollback

- `runner.rs` VERIFY gate relaxed: qwen35moe admitted when `moe_batched_ok` (all expert dtypes in `MOE_MMQ_DTYPES`)
- DeltaNet rollback filter verified correct for qwen35moe (`is_qwen35_attn_layer` covers the hybrid interval-4 structure)
- Unit test: `mtp_delta_filter_covers_qwen35moe_recurrent_layers`

### Stage 3: Paged-MoE verify integration

- **`mtp/backends.rs`**: naive raw-upload bind replaced with `vulkan_moe_binder` (the same placement planner + pager installer the normal path uses); new `generate_mtp_spec_vulkan_timed_on_state` with cold/warm binder split
- **`chat/vulkan.rs`**: MTP branch now routes through `ensure_session()` + `PlacementScope::enter()` on the shared session backend; `mtp_trunk: Option<SeamKv>` persists across cycles
- **`seam/mod.rs`**: dynamic MTP headroom reserve in the VRAM planner (computed from actual head tensor bytes + embed table size, replacing the hardcoded 2 GiB)

### Stage 4: Greedy fast path

- GPU argmax accept path (`Op::Argmax`, m×4 bytes readback) already existed; the full-logits D2H fallback (m×vocab×4B, 4-11 MB per verify at ~25 MB/s) was the dominant cost at temp>0
- `INFR_MTP_N_MAX` env var to tune draft length (default 6; 4 recommended for marginal heads)

## Known Limitations

1. **Greedy only in practice**: temp>0 uses `run_verify_full` which downloads full `m×vocab×4B` logits per cycle (D2H at ~25 MB/s over PCIe). A GPU-side stochastic accept or persistent staging buffer is needed for sampled MTP.
2. **Head session rebuilt per request**: the "no cross-turn KV reuse" design (backends.rs) rebuilds trunk+head sessions per `generate()` call. Persistent sessions across turns would eliminate ~300 ms of per-request setup but requires solving a self-referential borrow (documented in backends.rs).
3. **No fused KV write for QkNormMrope**: the vision mrope path emits `Op::QkNormMrope` + explicit `Op::WriteKv` (no peephole fusion). Decode replay is also unsupported for mrope graphs (static per-token rebuild).
4. **Acceptance rate is a model property**: the shipped Ornith-1.5 head was never trained (random init, confirmed by weight statistics and two independent engines). A properly trained head (e.g. shisa-ai's KL-distilled graft) achieves α=1.0 greedy / 60%+ MTP3 sampled.

## Performance

RX 7700 XT 12GB, Ornith-1.5-35B-A3B-APEX-MTP-I-Quality-MTPFIX (grafted shisa head), 8K context, greedy:

| Config | decode |
|--------|--------|
| Baseline (no MTP) | 54.7 tok/s |
| MTP (n_max=6, α=1.0) | **93.6 tok/s (1.71×)** |

Serve mode with the serialised engine (MTP-capable): prefill 446 tok/s, decode 69.5 tok/s at 12.5K-token prompts.

## Testing

- `cargo test -p infr-cpu` — 98+6 pass (incl. new QkNormMrope text-collapse + plane-selection tests)
- `cargo test -p infr-vision` — 15 pass
- `cargo check --workspace` — clean
- End-to-end: `INFR_MTP=1 infr run <qwen35moe-gguf> --temp 0` with MTP summary logging (α per cycle + aggregate)
