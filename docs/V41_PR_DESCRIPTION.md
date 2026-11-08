# DeepSeek-V4.1 support: run it, and fix the IQ1_M expert GEMM

Adds DeepSeek-V4.1 to the Vulkan backend and fixes a descriptor-binding defect that made
its dominant expert format produce exact zeros.

## The defect

`native_gemm_mmq_iq1_m.comp` declared a `layout(binding = 3) readonly buffer LUT` in the
`#elif defined(EXPERT_GRID)` arm, but `lut[]` is only read inside `#ifdef PAGED`. In the
non-paged build that stray declaration still claimed a slot, pushing `counts` / `offsets` /
`c` up by one: `c` landed on binding 6 while `matmul_mmq_experts` supplies only descriptors
0..5.

Every store was therefore discarded. The kernel did not fail, did not warn, and did not
trap — it wrote into a descriptor nobody bound, and the experts came back as exact zeros.

Removing the declaration puts `CNT` / `OFF` / `CB` back at 3 / 4 / 5, matching IQ2_S.

## What is verified, and what is not

**Verified:** the defect was real and the fix is pinned by tests. Before it, IQ1_M expert
output came back as exact zeros; after it, `grid_mmq_parity` matches the host. `ape: Option`
and the reuse-layer handling let the checkpoint load and emit tokens at all.

**Not verified, and previously claimed here:** a 2.3x prefill speedup. That figure was
measured with `INFR_DSV41_BATCHED_PREFILL=1`, and that path answers wrong. Same prompt,
greedy, both reproduced twice:

```
INFR_DSV41_BATCHED_PREFILL=1   "We need answer. Need answer. We"
default per-token              ". Wait, the answer is 100"
```

The batched path is numerically wrong even for a prompt short enough to fit one chunk, so
27.5 tok/s on it measures a different, incorrect computation rather than a faster model.
No speedup is claimed. Per-token throughput (the product path) is below 1.37 tok/s at
2K on this hardware, and that is the honest number.

## What else is in here

- `MOE_MMQ_PAGED_DTYPES` gains `DType::Iq1M`, so IQ1_M experts take the paged MMQ path
  instead of falling back to the one-output-per-workgroup GEMV.
- `ape: Option` on `Op::Dsv4Compress`. V4.1's `compress_ratios` is a role table {0,1,2},
  not a ratio table {0,4,128}: only layers {2,8,14,20} carry a compressor and the other 34
  reuse a source stream. The checkpoint contains zero `attn_compressor_ape` tensors.
- `grid_mmq_parity.rs` gains an IQ1_M case; the IQ2_S control runs first.
- Profiling: `main_merged` / `main_gaps` (how many disjoint runs the GPU intervals form),
  and `attributed` / `unattributed` (whether any host time is unaccounted for).

## Limitations worth knowing

Measured, not guessed — each of these came out of an A/B that failed to move the number:

- **Decode is 6.2 tok/s and nine tuning hypotheses did not change it.** Verified limits:
  bandwidth is not the cause (this machine's own probe reports 112.74 GB/s CpuToGpu
  against decode's ~6.6 GB/s), compute is not the cause (0.105 of ~10 TFLOPS), and host
  overhead is under 1% unattributed. The remaining candidates are structural — one output
  per workgroup, grid-decode cost at m=1, per-node dispatch overhead — and need kernel
  work rather than tuning.
- **The batched prefill path is not viable for V4.1 and is wrong, not merely slow or
  fragile.** With `INFR_DSV41_BATCHED_PREFILL=1`, pp16384 loses the device after ~20
  minutes (all 41 threads sleeping in `drm_syncobj_array_wait_timeout`) and pp100000 does
  the same; ubatch makes no difference. More importantly it computes wrong answers at
  every size tested, including sizes that fit a single chunk. `runner.rs:12370` says no
  batch>1 V4 graph has ever executed correctly because the `hc_mult`-width residual cannot
  cross a chunk boundary; the evidence here is that the failure is broader than the
  boundary. Without the switch, 16K does not hang but does not finish in 50 minutes on the
  per-token path.

  `runner.rs:12370` already documents the cause: the batched path builds one whole-model
  graph per chunk with `span = None`, so V4.1's `hc_mult`-width residual cannot cross a
  chunk boundary, and no batch>1 V4 graph has ever executed correctly. The per-token path
  is the product behaviour — correct, merely slow. The 4K/8K numbers measure the batched
  configuration, where it does run, and should not be read as default-path figures.

  The pager's `target_lanes=40 actual_lanes=5 resident_layers=0/40` is a VRAM consequence,
  not the cause.
- **MTP cannot be exercised with this checkpoint.** The code path is complete, but the
  GGUF carries no mtp/draft/nextn head (42 tensors, none of them).

## How to check whether it works

```
scripts/check-v41-answers.sh <control.gguf> <v41.gguf>
```

Runs a known-answer prompt through both checkpoints. The control is the point: without it a
V4.1 failure could equally be an engine fault, and "the output looks wrong" depends on the
reader already knowing the answer. Current result:

```
PASS  control  qwen2.5-1.5b-q4_k_m   "2+2=4"
FAIL  default  V4.1                  "We need answer. The question is"
FAIL  batched  V4.1                  "The user wants me to translate the analysis"
```

Until this passes, no throughput figure in this branch measures a working model.

## Testing

- `grid_mmq_parity`: `iq1m_expert_gemm_matches_host` and
  `grid_mmq_paged_expert_gemm_matches_host_under_eviction` pass; IQ2_S control unchanged.
- `infr-llama`: 411 passed / 0 failed.
- `synthetic_deepseek2`: 38 passed / 0 failed / 8 ignored.
- `infr-core`: 313 passed, 1 failed — `no_infr_env_reads_outside_the_config_layer`, which
  fails identically with these changes stashed (pre-existing: `runner.rs` and `engram.rs`
  read knobs straight from the environment instead of through `config/env.rs`).

## Method note

Several conclusions in `docs/V41_FIRST_TOKENS.md` were later retracted after A/B. The two
that mattered: "decode is at 70% of the machine's bandwidth" was withdrawn when the
ceiling turned out to be borrowed from another engine's PCIe probe on a different machine
(this one does 112 GB/s, not 9.5); and an arithmetic claim of 1047 ms of unattributed host
time was withdrawn after adding the counter and finding 26.4 ms. The log keeps both
corrections rather than rewriting history.
