# Benchmarking & profiling

`infr bench` matches `llama-bench`'s `-p`/`-n`/`-d`/`-r` flags, so the two are
directly comparable. Pipelines are compiled and GPU state is first-touched at
model load (`Llama::warmup`), so timing measures compute, not one-time setup.
**Run benchmarks one at a time** — concurrent GPU work skews results.

```bash
M='unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M'   # MoE perf target

# Prefill (pp = n_prompt/time) and decode (tg = n_gen/time):
infr bench "$M" -p 2048 -n 0 -r 3       # prefill 2048 tokens
infr bench "$M" -p 8000 -n 0 -r 2       # prefill at depth
infr bench "$M" -p 0 -n 64 -r 3         # decode 64 tokens
infr bench "$M" -p 0 -n 64 -d 2048      # decode at context depth 2048 (-d warms, untimed)
```

**Profile** per-op GPU time (timestamp queries) with `INFR_PROF_OPS=1`. Every
dispatch is timestamped and labeled **automatically with its kernel name** (plus
a few role overrides like `expert_gateup`/`expert_down`); no manual stamping. It
prints one block per submit and ONE aggregated `INFR_PROF_OPS GPU report` at
process exit (per-kernel totals, counts, avg, %GPU over all timed submits —
warmup runs unprofiled). Add `INFR_PROF_OP_SHAPES=1` for shape-itemized
GEMV/GEMM buckets (`mmvr:m4:1536x24576`). Decode's replay tape carries no
timestamps — profile decode with `INFR_SEAM_NO_REPLAY=1`. Details in
[`playbook.md`](playbook.md).

```bash
INFR_PROF_OPS=1 infr bench "$M" -p 2048 -n 0 -r 1 2>&1 | tail -30   # exit aggregate
```

**Validate Vulkan work** — any change touching `infr-vulkan` (kernels, recorder,
adapter, pager) must run its GPU tests and at least one end-to-end generation
under the Khronos validation layer, and fix every error AND warning it reports
before landing (validation silence is the bar, not "it produces the right
tokens" — robust-access reads, missing barriers, and binding-range overflows can
return plausible garbage instead of crashing):

```bash
VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation cargo test -p infr-vulkan -- --ignored
VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation infr run "$M" "smoke prompt"
```

The layer ships with the `vulkan-validation-layers` package. It slows GPU work
noticeably — use it for correctness passes, never inside timed benches.

**Compare to llama.cpp** — `infr compare` shells out to `infr bench` and the
system `llama-bench` with matching flags on coding-agent-shaped workloads
(prefill, decode-at-depth, whole turns). `--ctx` is comma-delimited:

```bash
infr compare "$M" --ctx 8000,16000 --gen 256 --turn 2048,256 --reps 2
```

**DiffusionGemma** has no upstream-merged `llama-bench` support, so
`infr compare`/`infr compare --sweep` route `arch=diffusion-gemma` models to a
different oracle: the reference fork's `llama-diffusion-cli`
(`~/Projects/mxaddict/llama.cpp-dg`, resolved via `INFR_LLAMA_DIFFUSION_CLI` >
`PATH` > the fork's `build-vulkan`/`build` directories — see
`ModelBench::llama_diffusion_cli_path` for the exact precedence and its PATH
fallback caveat). It prints two rows instead of the usual pp/tg matrix:
`dg-step` (in-step-parallel tok/s ratio — the apples-to-apples number, since
both implementations run entropy-bound and take a different number of denoise
steps) and `dg-e2e` (informational end-to-end tok/s, each side's own step count
folded into the row so the mismatch is visible). Details in
[`docs/diffusion-gemma.md`](../diffusion-gemma.md).

Useful knobs: `--temp` / `--top-k` / `--top-p` (sampling; `--temp 0` → greedy),
`--max-new`, `--ctx` — or the `sampling.*` / `device.*` config paths, or their
`INFR_*` twins. See [Configuration](#configuration).

**MoE expert placement**: resident when the expert banks fit VRAM (zero config,
zero change); otherwise every layer pages through a VRAM-resident LRU expert
cache (`infr_vulkan::pager`) sized to the remaining VRAM. `INFR_CACHE=<size>`
forces every layer through the pager with that budget regardless of fit (useful
for testing, or to free VRAM for a larger context). Every bank shape pages:
split gate/up (llama4/Qwen3-MoE/Qwen3.6-MoE), fused gate_up (DiffusionGemma,
Gemma-4 MoE — one double-width slot per expert), and mixed-dtype roles
(unsloth-dynamic quants bumping a subset of layers' banks to a wider K-quant —
one logical arena pool per expert byte size, shared across compatible roles). `INFR_PAGER_STATS=1` prints each pool's
hit/miss/eviction counts.

**Dense layer streaming**: DENSE models bigger than VRAM stream their per-layer
projection weights (attn q/k/v/o + FFN gate/up/down, as the same fused
qkv/gate_up groups the loader uploads) through the same paged VRAM machinery —
but schedule-driven, not LRU: a dense forward visits layers in one fixed order,
so residency uses an exact cyclic-sweep policy (Belady-parity — a stable
resident prefix plus one churn slot per pool) and there are NO readbacks
anywhere (every "miss" is known in advance; misses ride recorded ring→arena
copies on the same pipelined fenced-half staging ring the MoE path uses, so CPU
memcpys for later layers overlap GPU execution of earlier ones). Streamed
dispatches are the ordinary dense kernels reading the pool arena at a slot
element offset (the `w_off` convention) — no kernel variants, so streamed output
is token-identical to the resident run. Embeddings, lm_head, norms and biases
stay resident (lm_head is read at every token edge — streaming it adds its full
bytes to every token's PCIe bill with zero locality to exploit). Placement is
automatic (resident when everything fits — zero change); `INFR_CACHE=<size>`
forces streaming with that budget. Honest expectations: prefill amortizes
uploads across the whole batch (Qwen3-14B Q8_0, ~15.7 GB, at `INFR_CACHE=8g`:
pp512 987 t/s vs 1505 resident = 0.66×); decode has no locality to exploit, so
it is capped at PCIe_bw ÷ overflow_bytes per token — physics, not a bug (same
setup: ~7.0 GB re-uploaded per token ÷ ~22 GB/s ≈ 3.1 t/s ceiling, measured 3.1
t/s; the CPU backend does 4.4 t/s at that ~45% overflow, so streaming only beats
CPU when the overflow is smaller — measured crossover on this box is around a
quarter of the model overflowing). An MoE model whose DENSE part also doesn't
fit is out of scope and errors clearly.

**Size grammar** — `paging.cache` / `INFR_CACHE` and `device.ctx` / `INFR_CTX` /
`--ctx` share one value grammar (`infr_core::parse_size`): a plain number is the
base unit (bytes for `INFR_CACHE`, tokens for `INFR_CTX`), `k`/`m`/`g`/`t`
suffixes scale by 1024 (`INFR_CACHE=19g`, `INFR_CTX=256k`), and `%` resolves
against the device-appropriate base — available VRAM for the expert cache, the
free-VRAM KV capacity for the Vulkan context (`INFR_CACHE=80%`, `INFR_CTX=50%`;
on the CPU/Metal chat paths a ctx-`%` resolves against the model's trained
context).

**Resident-BDA weight arena** — always on; it is the only weight path. Routes
every weight allocation into one `bufferDeviceAddress` arena and has the kernels
read their weights by 64-bit device address instead of through per-tensor SSBO
descriptor bindings — dense projection weights and MoE expert banks read via
`-DSTREAMED` kernel twins, sub-tensors via sub-range descriptor binds, and the
paged expert cache composes on top unchanged. The addressing is
bitwise-identical to the retired u32-SSBO descriptor path across the whole model
zoo (dense, MoE, qwen35/DeltaNet, DiffusionGemma, and the paged Scout experts —
proven by the `gpu_seam` goldens and the streamed-parity suites), and runs
at-or-faster than that path on RDNA3 (7900 XTX) on the dense and qwen3-MoE
paths.
