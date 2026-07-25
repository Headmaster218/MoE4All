# backend-unification-plan.md — one seam for shared backend logic

infr has four compute backends — **CPU** (`infr-cpu`), **Vulkan**
(`infr-vulkan`), **Metal** (`infr-metal`), **ROCm/HIP** (`infr-rocm`) — that
already share a real seam (`Backend` trait, `Op`/`Graph` IR, quant tables, the
seam runner). But as the backends have grown, a lot of **device-agnostic host
logic** has been independently re-implemented in two, three, or four places —
and it drifts. This plan audits that duplication and lays out a staged
extraction so that **cross-backend logic lives once**, and adding a capability
(a fusion, a tier rule, a new arch op) is done in the seam, not copy-pasted per
backend.

Guiding principle: **extract the host logic, not the shaders.** A block-decode
kernel in GLSL/MSL/HIP can't become one Rust function — but the _decision_ to
run it, the _pattern-match_ that fuses it, the _threshold_ that selects it, and
the _spec_ it's checked against are pure Rust over the shared IR and belong in
`infr-core`/the seam. Backends supply the device-specific piece via a trait or a
predicate.

Scope note: this is a **refactor** plan — behavior must not change. Every
extraction lands with the existing per-backend tests still green and (where
applicable) the goldens unmoved. It composes with `docs/rocm-plan.md`'s parity
work: prefer landing a _shared_ implementation here over writing a fourth
per-backend copy there.

## Already shared (the seam we're extending — do not re-propose)

- **`Backend`/`Buffer`/`Plan` + `Capabilities`** — `infr-core/src/backend.rs`;
  all four backends implement them.
- **`Op`/`Graph` IR** — `infr-core/src/graph.rs`; the agnostic seam every
  backend walks.
- **`DType` / `MOE_MMQ_DTYPES` / `moe_mmq_ok`** — `infr-core/src/tensor.rs`.
- **`dequant_block` + quant tables** — `infr-gguf/src/dequant.rs`; host-side
  decode oracle used by cpu/rocm/metal.
- **`iquant_grids`** — `infr-core/src/iquant_grids.rs`; the single-source IQ
  codebooks (cpu/metal read directly, vulkan emits into shaders).
- **LRU `Pager`** — `infr-core/src/pager.rs`; the paging _policy_ (vulkan + rocm
  reuse it).
- **Agnostic seam runner `generate_dense_backend`** —
  `infr-llama/src/seam/ runner.rs`; owns the whole per-token loop, prefix-diff
  prefill, KV-slot reuse, and host sampling over `&dyn Backend`. The per-backend
  `generate_dense_{metal,rocm}_session` are thin wrappers. **This is the model
  to extend.**
- **Host sampling** — `infr-llama/src/sampling.rs` (temp/top-k/top-p/argmax) is
  already the single oracle; backends only implement the _device_ `Op::Sample`/
  `Op::Argmax` kernels tested against it.
- **Profiling** (`infr-prof`), **progress bars** (`infr_core::progress`) —
  shared.

## The duplication — extraction candidates

Ranked by value × cleanliness. Severity = how much drift-prone duplication;
extraction = how clean the refactor is.

### A. Peephole fusion → one shared Graph-rewrite pass — HIGH / CLEAN ⭐ ✅ LANDED (`e6d9c25`)

**Done (candidates A + G together).** Extracted to `infr-core/src/fusion.rs`:
`plan_fusions(graph, &FusionCfg) -> FusionPlan`, the union of all three passes
(`Linear→Add`, `RmsNorm→Linear`, `Rope/QkNormRope→WriteKv`), each gated by a
backend-supplied `weight_ok: &dyn Fn(DType)->bool` predicate + a `disable_env`
hatch. ROCm's live-range bound is applied to every backend. `DType::is_kquant`/
`is_legacy_round`/`is_iquant` added (candidate G). All three backends rewired to
consume the same `(fused, skip)`; net −170 lines; no golden moved (qwen3
`0xfd63781ea3bfa785` unmoved, Vulkan gpu_seam unmoved, ROCm 30/30). Original
audit notes below (historical).

`linear_add_peephole` is re-implemented near-identically in **three** backends —
vulkan `adapter.rs:827`, metal `exec.rs:719`, rocm `exec.rs:923` (folded into
`decode_fusion`, which also adds RmsNorm→Linear `exec.rs:898`). All three walk
`graph.ops`, match `Op::Linear{m:1, Internal dst}` + the following `Op::Add`,
resolve the residual operand, and build a `(fused: HashMap, skip: HashSet)`. The
**only** real difference is the dtype predicate (vulkan
`native_dense_supported||F16`; metal an explicit `Q4K|Q6K|Q8_0|…` list; rocm the
int8-decode set). `kv_write_peephole` (vulkan `adapter.rs:876`) and
RmsNorm→Linear are further backend-agnostic rewrites.

- **Extract:** an `infr-core` (or `infr-llama::seam`) Graph-rewrite module
  returning a `FusionPlan { fused, skip }`, parameterized by a
  `fn(DType) -> bool` capability predicate the backend supplies (folds in
  candidate G). Unify the env escape hatches (`INFR_NO_FUSE_ADD`,
  `INFR_ROCM_NO_FUSE_*`) into one policy. Pure host logic over the IR, no device
  types. Cleanest high-value win.

### B. Graph-executor skeleton (residency + op-walk) — HIGH / MED ⭐

Every backend runs the same skeleton: a per-`TensorId` residency array
(`dev: Vec<Option<DeviceBuf>>` + `vals: Vec<Option<Vec<f32>>>`), a walk over
`plan.graph.ops`, a `match *op` to `run_op`/`lower_op`, operand-residency
resolution, and dst-resident marking — rocm `ExecCtx` `exec.rs:474`, metal
`ExecState` `exec.rs:681` (its header literally says "identical to the CPU
interpreter"), cpu `lib.rs:465`, vulkan `lower_op` `adapter.rs:1067`. The
`match Op::` **arm set + order is parallel across all four**.

- **Extract:** a generic `GraphExecutor` skeleton owning the residency array,
  the op-walk, operand resolution, and the fusion-skip check, driven by a
  backend `OpDispatch` trait (one method per op, or `dispatch(op, ctx)`). The
  _structure_ is shared; the per-op _bodies_ stay per-backend (device dispatch).
  Biggest structural prize but touches every hot loop — **stage after A/F**
  prove the shared-IR-pass pattern.

### C. `be(msg) -> Error` helper — MED (trivial) / CLEAN

Identical error-wrapper duplicated in **7 files** (rocm ×5: `backend.rs:19`,
`exec.rs:21`, `kernels.rs:18`, `pager.rs:66`, `weight_pager.rs:63`; metal
`lib.rs:39`; vulkan `lib.rs:68`).

- **Extract:** one `infr_core::error::backend(msg)` constructor. Trivial, do it
  first alongside A.

### D. Chat `ChatModel` wrappers + `warmup`/`reset_kv` — MED / CLEAN

`chat/{vulkan,rocm,metal}.rs` are structurally identical:
`{model, session: Option<…Session>, …}`, `new`, `ensure_session` (lazy open +
`INFR_CTX` via `parse_size`), and a `ChatModel` impl whose `warmup` is
copy-paste `self.generate("Hi", 2, …)?; session.reset_cache()` (vulkan
`chat/vulkan.rs:143`, rocm `chat/rocm.rs:62`, metal `chat/metal.rs:87`);
`reset_kv` identical.

- **Extract:** a generic `SessionChat<S: SeamSession>` where `SeamSession`
  abstracts `reset_cache` + `generate`; the `ChatModel` impl (incl. warmup)
  becomes one blanket impl. A `feature_stub!` macro removes rocm's doubled
  `cfg(not(feature="rocm"))` placeholder too.

### E. `Dense{Vulkan,Metal,Rocm}Session` structs — MED / CLEAN

`seam/model.rs`: `DenseVulkanSession:87`, `DenseMetalSession:272`,
`DenseRocmSession:294` are the same shape (`{backend, pool: SlotPool, max_ctx}`;
vulkan adds `pins`) with identical `reset_cache` delegating to the
**already-shared** `SlotPool`.

- **Extract:** `DenseSession<B: Backend> { be: B, pool: SlotPool, max_ctx }`;
  collapses three structs + three `reset_cache` into one. Pairs with D.

### F. Capability-tiering / kernel-selection policy — MED / MED

The m-based tier thresholds are pure host arithmetic but live per-backend.
Vulkan is richest: `moe_small_m_threshold` (`adapter.rs:160`,
`INFR_MOE_SMALL_M`), the m=1 / m≤4 / m5..8 GEMV↔GEMM ladder (`adapter.rs:1340`),
the attention split-K count cap (`adapter.rs:941`). ROCm/metal have their own
smaller versions. The _shape/threshold_ math (when a tiny-m GEMV beats a tiled
GEMM, the MoE small-m cutoff, split-K chunking) is device-independent.

- **Extract:** a shared `tier` policy module — thresholds + `INFR_*` parsing +
  the m→path decision returning an enum the backend maps to its own kernels.
  Makes vulkan's existing cross-tier assertions (`adapter.rs:5935`) structural.

### G. Dtype-support predicates — MED / CLEAN (folds into A)

"Which quant formats does this path support" is scattered: vulkan
`native_dense_supported` (`linear.rs:257`), metal's re-typed inline list
(`exec.rs:737`), rocm's int8 set, plus dozens of raw `DType::Q4K` match sites
that drift independently.

- **Extract:** named predicates on `DType`/a `QuantClass` in `infr-core`
  (`is_kquant`, `is_legacy_round`, `native_dense_class`) that each backend
  intersects with its own kernel coverage — becomes candidate A's predicate
  parameter.

### H. Decode spec + shared parity harness — HIGH (logic) / HARD (device)

Block decode is re-implemented per shader language (GLSL/MSL/HIP) and **cannot**
be one Rust fn. But two things can be shared: (1) a **decode spec/constants**
module (block sizes, scale layout) that shaders are generated-from or
checked-against (the IQ grids already are, via `iquant_grids`); (2) a **shared
parity-test harness** built on the host `dequant_block` oracle that every
backend runs. Today vulkan has ~75 `*_matches_host` tests; **rocm/metal have
none under `src/`** — they lean on the oracle informally.

- **Extract:** a `backend-parity` test harness (drive a one-op `Graph` on the
  backend, compare to `dequant_block` + a reference GEMV) parameterized by the
  backend, so rocm/metal/vulkan/cpu all get the same coverage from one source.
  Closes the rocm/metal parity-test gap. Not a runtime dedup — a _spec + test_
  seam.

### I. KV/paging budget math — MED / MIXED

The paging _policy_ is shared (`Pager`); the _device wiring_ stays per-backend.
Shareable slices: the KV-format→bytes math (partly hoisted: `kv_bytes_per_elem`
`seam/model.rs:330`, `INFR_KV_TYPE_K/V` parsing) and the ring-size budget
arithmetic (`ring_bytes_policy` `pager.rs:606`). Also: vulkan's
`kv_overflow_report` is itself duplicated 4× internally (`lib.rs:3319`,
`tp.rs:783`, `ep.rs:498`, `pipeline.rs:644`).

- **Extract:** hoist the remaining KV bytes/budget math + `INFR_*` parsing to
  shared helpers; de-dup vulkan's internal `kv_overflow_report`. Leave device
  alloc/arena/staging per-backend.

### Already-fine / no action

- **J. KV ring/slot semantics** — prefix reuse, slot picking (`SlotPool::pick`)
  live in the shared seam; only the SWA ring-wrap check rides along with A.
- **K. Host sampling** — single oracle in `sampling.rs`; nothing to dedup (it's
  the invariant device kernels are tested against — ties into H).
- **L. Device enumeration** is vulkan-internal-only duplication (multi-device
  backends); low cross-backend value.

## Recommended extraction order

1. **A. Peephole fusion → shared Graph-rewrite pass** (+ **G** dtype predicates
   as its parameter) — cleanest high-value win; three near-identical copies
   collapse to one.
2. **C** `be()` helper + **E** `DenseSession<B>` + **D** `SessionChat<S>` —
   cheap, mechanical; removes ~7+3+3 copies and rocm's feature-stub doubling.
3. **F. tiering policy module** — moderate, high leverage; makes the existing
   cross-tier assertions structural.
4. **B. GraphExecutor skeleton** — the biggest structural prize, but it touches
   every backend's hot loop; do it only after A/F prove the shared-IR-pass
   pattern and with the full per-backend test suites as the guardrail.
5. **H** shared decode-spec + parity harness and **I** KV/budget math —
   spec/test and math hoists (not runtime dedup); H also closes the rocm/metal
   parity-test gap.

## Guardrails

- **Refactor, not rewrite:** each step is behavior-preserving. Land it with
  every backend's existing tests green; goldens must not move (the fusion/exec
  extractions are pure structure). A moved golden means a behavior change
  slipped in — stop.
- **Device stays per-backend:** shaders/kernels, command recording, buffer
  alloc, and the per-op dispatch bodies are NOT unified — only the host logic
  around them.
- **Backend-parameterized, not lowest-common-denominator:** the shared code
  takes a predicate/trait so each backend keeps its own capability set (e.g. its
  own quant coverage) — unifying the _logic_, not flattening the _capabilities_.
- **One PR per candidate**, smallest first, so a regression is bisectable to a
  single extraction.
