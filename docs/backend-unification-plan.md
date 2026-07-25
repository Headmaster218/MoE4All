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

### B. Graph-executor skeleton (residency + op-walk) — HIGH / MED ⭐ ⚠️ PARTLY LANDED (`ce45447`)

**Done — the residency CONTRACT and the WALK; not the container, not the per-op
bodies.** `infr-core/src/exec.rs` now hosts the device-agnostic half of the
skeleton, and every backend that can consume a given piece does:

- **`Provision` / `provisions(graph)`** — the pre-walk per-handle action (`Zero`
  an `Internal`/`Output`, `Load` a bound `Input`, `Skip` a `Weight` or an
  in-place KV `Input` per `Graph::in_place_inputs`). The cpu and Metal
  interpreters' setup loops were the same `match decl.kind` + `direct.contains`
  ladder; both now classify through it. Metal's one genuine extra arm (recording
  binds f32 `Input`s directly, so they need no host mirror) stays a guard inside
  its `Load` arm.
- **`writes_back` / `write_back_targets(graph)`** — the post-walk predicate
  `Output || (Input && F32 && !in_place)`, which was **byte-identical in three**
  executors (cpu, Metal, ROCm) and is the one that silently corrupts if it
  drifts. All three now select the same set; the copy itself (host memcpy /
  `hipMemcpyDtoD` / unified-memory store) stays per-backend.
- **`live_ops(ops, skip)`** — THE definition of the walk: graph order with the
  fusion-elided indices skipped. Four call sites: ROCm's executor (via
  `run_ops`), ROCm's **dense-weight prefetch schedule** (a separate pass whose
  cursor must track the op walk exactly or a staged bank goes to the wrong
  `Linear` — it now iterates the same function instead of re-spelling the
  filter), Metal's executor (via `run_ops`), and Vulkan's `execute_static`.
- **`OpDispatch` + `run_ops`** — the walk as a driver, generic
  (`impl OpDispatch`, never `&dyn`), adopted by the two backends whose per-op
  body is already one function: ROCm (`RocmDispatch`, which also does the per-op
  fusion payload lookup) and Metal (`MetalDispatch<const PROF: bool>`). Metal's
  two loops — profiled and not — collapse into one walk plus a
  **const**-parameter instantiation chosen once outside it, so the profiling
  half is still compiled out of the hot path rather than becoming a per-op
  branch.

**NOT done — the residency CONTAINER.** The four disagree for real reasons, not
drift: cpu keeps a host-only `Vec<Vec<f32>>`; ROCm keeps
`dev: Vec<Option<RocmBuffer>>` + a lazily-filled `vals: Vec<Option<Vec<f32>>>`;
Metal keeps both **plus** a `loc: Vec<Loc>` host/device tracker (43 sites) with
no ROCm analogue — because Metal's `ensure_device` uploads the host mirror and
ROCm's never does (it binds the bound buffer, or allocates zeroed pool scratch).
Vulkan has no per-op residency at all: its `Internal` scratch is allocated up
front by `alloc_scratch` and the walk only records. A common `Residency<D>`
would be either the LCD (drop `loc`) or the union (a dead `loc` on ROCm), and
either way all ~240 `vals`/`dev`/`loc` accesses **inside the per-op bodies**
would be rewritten — behavior-neutral in principle, a rewrite in fact.

**NOT done — a per-op trait method (`fn rms_norm(...)`, one per variant).** The
`match Op::` arm sets are parallel, but the arms need wildly different ambient
state: Vulkan's `lower_op` threads 15 extra parameters (recorder, scratch, pool,
dyn-attn contexts, mmv memo, streamed-weight substitution), ROCm's takes the two
per-op fusion payloads, Metal's the `Resident` + tape. A 27-method trait would
move ~10k lines of op bodies, save no lines (a `Op::X { a, b } =>` header
becomes an `fn x(&mut self, a: .., b: ..)` header), and still have to carry each
backend's ambient state in its `Self`. That is a rewrite, and the goldens are
the only thing that would catch a slip.

**NOT done — Vulkan on the shared skeleton.** It is a **recorder**, not an
interpreter: no host values, no residency transitions, and its loop carries five
device concerns between ops (submit splitter, shutdown poll, paged-MoE hand-off,
dense-streaming stage, and an E2B `Linear`+`GatedAct` peephole that **mutates**
the skip set mid-walk). `execute_static` takes `live_ops` and keeps its own
body; `record_decode_replay` can't take even that, because `live_ops` borrows
the skip set immutably. Hooks for all five would be LCD flattening, not
unification.

**NOT done — cpu on the walk.** Its ~2000 lines of op bodies are inline in the
`for op in &g.ops` loop, not a `run_op` function, so `OpDispatch` would mean
moving all of them into a trait method; and cpu runs **no** fusion pass, so a
shared `skip`-checking walk would add a per-op `HashSet::contains` the current
loop does not pay. cpu takes the residency contract only.

Behavior: same ops, same order, same host/device sync points, same error text,
same env vars; no golden moved. Original audit notes below (historical).

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

### C. `be(msg) -> Error` helper — MED (trivial) / CLEAN ✅ LANDED (`87bca71`)

**Done.** All 7 wrappers were verified byte-identical first (each already
delegated to `Error::backend`; only Vulkan's carried an extra doc comment +
`infr_prof::instrument`, and the shared constructor is itself instrumented, so
nothing was lost). `infr-core/src/error.rs` gained the free-function form
`infr_core::error::backend(msg)`; each of the 7 files now does
`use infr_core::error::backend as be;` instead of declaring a wrapper. All ~271
`be(...)` call sites and every error string are unchanged. Original audit below
(historical).

Identical error-wrapper duplicated in **7 files** (rocm ×5: `backend.rs:19`,
`exec.rs:21`, `kernels.rs:18`, `pager.rs:66`, `weight_pager.rs:63`; metal
`lib.rs:39`; vulkan `lib.rs:68`).

- **Extract:** one `infr_core::error::backend(msg)` constructor. Trivial, do it
  first alongside A.

### D. Chat `ChatModel` wrappers + `warmup`/`reset_kv` — MED / CLEAN ⚠️ PARTLY LANDED (`87bca71`)

**Done — the parts that really were mechanical.** Three shared pieces now live
in `chat/mod.rs` and every backend chat calls them:

- `ChatModel::warmup_session()` — a provided method holding the copy-pasted
  warmup body (`generate("Hi", 2, …)?` then `reset_kv()`). Deliberately NOT the
  `warmup` default (that stays a no-op so the stateless backends and the test
  mocks are untouched); Vulkan/Metal/ROCm opt in. Vulkan still wraps its call in
  `with_prof2_suppressed`, Metal still deliberately does not — the one real
  difference between the three, kept.
- `reset_session(&mut Option<DenseSession<B, X>>)` — the identical `reset_kv`
  body, generic over candidate E's session pairing; 5 call sites (three
  `reset_kv` impls plus `SpecMetalChat::warmup`'s two).
- `env_ctx_spec()` / `env_ctx(n_ctx_train)` — the `INFR_CTX` parse. Metal's and
  ROCm's `ensure_session` bodies were byte-identical; Vulkan takes the raw
  `SizeSpec` (it routes `Bytes`/`Percent` to different VRAM-fit constructors).
  `INFR_CTX` is now named in exactly one place.

**NOT done — `SessionChat<S>` + a blanket `ChatModel` impl.** The three are only
identical down to the struct header; past it they diverge for real reasons, and
forcing a generic would have meant either behavior changes or a generic with one
user. Concretely: Vulkan carries four extra fields (`mtp_head`, `mtp_checked`,
`mtp_vk`, `dev`) and a `new_on` constructor; Vulkan and Metal `generate` branch
into two DIFFERENT MTP drivers before touching the session, ROCm has none; ROCm
implements no `generate_constrained`; the three `ensure_session` bodies call
three differently-shaped session constructors (Vulkan's is a 3-way `SizeSpec`
match into `vulkan_session_on`/`_frac_on`/`_default_on`). A blanket impl would
have to push `generate` back into the trait parameter, at which point the
"shared" shell is the two fields + `reset_kv` that the three helpers above
already single-source. Re-scope this before re-attempting.

Also NOT done: the `feature_stub!` macro. There are only two
`cfg(not(feature = "rocm"))` placeholders (`RocmSeamChat` in `chat/rocm.rs`,
`DenseRocmSession` in `seam/model.rs`), they have different shapes, and each
stub method carries its own `unreachable!`/`bail!` text — a macro over two
dissimilar single-use stubs would obfuscate ~15 lines, not dedup them.

Original audit below (historical). `chat/{vulkan,rocm,metal}.rs` are
structurally identical: `{model, session: Option<…Session>, …}`, `new`,
`ensure_session` (lazy open + `INFR_CTX` via `parse_size`), and a `ChatModel`
impl whose `warmup` is copy-paste
`self.generate("Hi", 2, …)?; session.reset_cache()` (vulkan
`chat/vulkan.rs:143`, rocm `chat/rocm.rs:62`, metal `chat/metal.rs:87`);
`reset_kv` identical.

- **Extract:** a generic `SessionChat<S: SeamSession>` where `SeamSession`
  abstracts `reset_cache` + `generate`; the `ChatModel` impl (incl. warmup)
  becomes one blanket impl. A `feature_stub!` macro removes rocm's doubled
  `cfg(not(feature="rocm"))` placeholder too.

### E. `Dense{Vulkan,Metal,Rocm}Session` structs — MED / CLEAN ✅ LANDED (`87bca71`)

**Done.** One `DenseSession<B, X = ()> { be, pool, max_ctx, ext }` in
`seam/model.rs` with the single shared `reset_cache` (pure `SlotPool` policy, no
device involvement). `X` is per-backend extension state — `()` for Metal/ROCm,
`VulkanSessionPins` (an opaque wrapper so the crate-private `PlacementPins`
stays private) for Vulkan — i.e. backend-parameterized rather than flattened to
a lowest common denominator. `DenseVulkanSession`/`DenseMetalSession`/
`DenseRocmSession` remain as type aliases, so every public signature and call
site is unchanged; Vulkan's `device_name`/`vram`/`pins()` live in an
alias-specific `impl`. The non-`rocm` `DenseRocmSession` placeholder stays its
own struct (nothing ever constructs it). Original audit below (historical).

`seam/model.rs`: `DenseVulkanSession:87`, `DenseMetalSession:272`,
`DenseRocmSession:294` are the same shape (`{backend, pool: SlotPool, max_ctx}`;
vulkan adds `pins`) with identical `reset_cache` delegating to the
**already-shared** `SlotPool`.

- **Extract:** `DenseSession<B: Backend> { be: B, pool: SlotPool, max_ctx }`;
  collapses three structs + three `reset_cache` into one. Pairs with D.

### F. Capability-tiering / kernel-selection policy — MED / MED ⚠️ PARTLY LANDED (`6c5cbd2`)

**Done — the arithmetic that really was shared.** `infr-core/src/tier.rs` now
hosts the device-independent half of kernel selection; every measured number
stays with its backend and is passed IN as config:

- **`EnvRows { env, default, min, max }` + `get()`** — the `INFR_*` numeric
  row/count knob (parse → default → clamp). Vulkan's `INFR_MOE_SMALL_M` (default
  8, ceiling 64 — an unclamped override device-losts the GPU) and
  `INFR_CANVAS_CHUNK_N` (default 3, floor 1) now declare a config instead of
  re-spelling the parse chain.
- **`linear_tier(m, out_f, &[MultiRowBand]) -> LinearTier`** — the dense
  `Linear` m-ladder as an enum (`Decode` / `MultiRow { band }` / `Gemm`) the
  backend maps to its own kernels. `MultiRowBand` carries `min_m`/`max_m`/
  `narrow_max_m`/`wide_out_f_max`/`out_f_max`, which expresses all three shapes
  exactly: Vulkan's `MROW_BANDS` (m=2..=8 with the `out_f<=8192` wide-n cutoff
  above m=4, plus the m=9..=16 int8 band) and Metal's `MRV_BAND` (m=2..=8 up to
  the `out_f<65536` lm_head ceiling, which `INFR_METAL_LMHEAD_MRV` lifts). The
  capability half — kernel builds, dtype sets, `caps.i8_dot`, the env hatches —
  deliberately stays at each `Op::Linear` arm.
- **`adaptive_chunk` / `n_chunks` / `cap_chunk_count` / `baked_chunk`** — the
  flash-decoding split-K attention chunk policy (`AttnSplitCfg`: ~32
  chunks/head, 64..512 keys), shared by Vulkan's `attn_partial` and ROCm's
  `attention_split_partial`, plus the chunk-COUNT cap Vulkan needs because
  `attn_combine.comp` indexes `shared float wexp[1024]` (ROCm's combine reads
  from global memory and takes no cap — that stayed a Vulkan-only parameter).
  The ROCm copy's comment claimed it "mirrors the Vulkan policy"; it does not
  quite — Vulkan floors `span/32`, ROCm ceils. That one-key divergence is real
  and observable, so it is now explicit config (`ChunkRounding::{Down, Up}`)
  with a test pinning it, not silently unified.

Vulkan's cross-tier assertions are structural against the shared policy:
`split_k_chunk_count_cap_*` now also assert `adaptive_chunk(span, &ATTN_SPLIT)`
reproduces the old inline `(span/32).clamp(64,512)` for every span, and the new
`mrow_bands_match_inline_formula` / `mrv_band_matches_inline_formula` (Metal)
sweep the old inline row/width expressions verbatim. `infr-core` gained 7 unit
tests over boundaries + env parsing. No golden moved; no tier decision changes
for any `(m, out_f, span)`.

**NOT done — the prefill GEMM's narrow-n split-K count.** Vulkan's
`split_k_plan` and Metal's inline `ks_split` look like the same idea ("does the
tile grid fill the device? if not, split k") but encode different device facts:
different tile geometry (64×128 coopmat vs 32×64 threadgroup), different fill
targets (256 vs 160 workgroups), a `next_power_of_two` vs a min-K-steps tail
rule, and different applicability windows (`in_f>=1024` vs `m<16`). A shared
form would need one config field per line of either formula — LCD flattening,
not unification. Left per-backend; both keep their own test.

Also NOT moved: ROCm's dense `Linear` ladder is a bare `m > 1` (prefill WMMA vs
decode GEMV) with no multi-row band, so routing it through
`linear_tier(m, out_f, &[])` would add a call and remove nothing.
Per-dtype/per-vendor coverage sets (`mrow_int8_dtype_ok`, Metal's `prefer_*`)
are candidate G's territory, not arithmetic. Original audit below (historical).

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

### H. Decode spec + shared parity harness — HIGH (logic) / HARD (device) ✅ LANDED (`f817c25`)

**Done — both halves.**

**(1) The spec.** `infr-core/src/decode_spec.rs` is now the single named
description of every GGUF block format:
`block_spec(dtype) -> BlockSpec { dtype, block_elems, block_bytes, scales: &[ScaleSlot], name }`
plus `block_layout(dtype) -> (elems, bytes)` and the `WEIGHT_QUANTS` (24) /
`KV_ONLY_QUANTS` rosters. A `ScaleSlot` is `{ offset, enc, role }` — where a
block's scale field lives and how it is encoded (`F16`, MXFP4's `E8M0`, NVFP4's
four `Ue4m3` sub-scales, IQ1_M's `Iq1mSplitF16` `d` spread across the top
nibbles of its four scale words) — with `BlockSpec::write_scales` as the
primitive that writes them. Deliberately NOT here: the decode arithmetic;
`infr_gguf::dequant_block` stays the single host oracle and the kernels stay the
device implementations. A fourth Rust decoder would be a liability, not a spec.

Wired both ways, so a wrong entry fails a test rather than sitting unread:
`infr_gguf::block_layout` is now a thin delegation; `dequant_factored`'s second
10-entry `(qpb, bpb)` table is gone (it keeps only the "which formats are
factorable" decision); ROCm's `native_decode_fmt` / `native_i8_fmt` /
`moe_native_fmt` derive their block geometry from it instead of re-spelling
`(256, 144)` (pinned against the pre-hoist literals by a new `exec.rs` unit
test); and five Vulkan test-local `blk_bytes`/`block_geom`/`blk_shape` tables
(`decode_gemv_bw`, `mmv_mw_parity`, `mmv_mrow_legacy_formats`, `nc_gemm_parity`,
`mmv_row1_bit_identical` — 17 entries in the largest) collapse to one call.
infr-core gains 6 unit tests, incl. `block_bytes_match_the_ggml_formulas`, which
restates every `type_size` as its defining `QK_K` formula so a typo cannot pass.

**(2) The harness — `infr-testkit`.** A new dev-only crate (in the workspace,
`publish = false`, a `[dev-dependencies]` entry in cpu/rocm/metal/vulkan and
nothing else). It offers `synth_weight(dtype, n_elem, seed)` — ONE spec-driven
builder covering all 24 weight quants, replacing the per-suite
`synth_q`/`synth_q4k`/`synth_mxfp4`/`synth_nvfp4`/`synth_iq1m`/`lcg_bytes`
families that ROCm's and Metal's parity suites had independently reinvented —
plus `dequant_oracle`, an f64-accumulated `ref_linear`, a generic
`run_graph(&dyn Backend, &Graph, bound, out, n)`, and
`check_linear`/`sweep_linear{,_on}` returning scored `CaseReport`s (rel/abs/
`max|ref|`, with an all-zero oracle counted as VACUOUS, not passing).

**Why a crate, not a `cfg(test)` module in infr-core:** the oracle IS
`infr_gguf::dequant_block` and infr-gguf depends on infr-core, so an infr-core
module reaching for it is a dependency cycle — cargo feature or not. A separate
crate also keeps the harness out of every production dependency graph by
construction, which a `feature = "test-harness"` cannot promise under workspace
feature unification.

**New tests (11), all from that one source:**

- **cpu** — `infr-cpu/tests/decode_parity.rs`, 3 tests, NOT ignored (the
  harness's GPU-less instantiation, so a GPU-less box actually exercises it).
  All 24 weight quants at m=1 and m=2 plus F32/F16. Not a tautology: infr-cpu's
  quant `Linear` runs its own `vec_dot_*` decoders, an INDEPENDENT
  implementation from `dequant_block`. Measured worst case 7.6e-3 (Q2_K) against
  a 2e-2 bound — the gap is the CPU path's int8 ACTIVATION quant, not decode.
- **rocm** — `infr-rocm/tests/shared_decode_parity.rs`, 3 tests, `#[ignore]`d.
  All 24 at **m=1** (the `linear_i8_*` dp4a GEMV) and **m=16** (the WMMA
  matrix-core prefill GEMM) — two tiers the legacy
  `all_quant_linear_matches_cpu` sweep (one shape, m=2) never reached — against
  the HOST oracle rather than `CpuBackend`, which is itself lossy. **All 48
  cases pass**, worst case 4.7e-3 (Q6_K m=1) vs the 2e-2 bound; the f32-exact
  formats come in at ~3e-7.
- **metal** — `infr-metal/tests/shared_decode_parity.rs`, 3 tests,
  `#[ignore]`d + `#![cfg(target_os = "macos")]`. All 24 at m=1 and m=4 plus
  F32/F16. Compiles clean for `x86_64-apple-darwin`; **not executed — no Mac in
  this environment**, so its `2e-2` is the defensible ceiling, not a measured
  number (see below).
- **vulkan** — `infr-vulkan/tests/shared_decode_parity.rs`, 3 tests,
  `#[ignore]`d. All 24 at m=1 and m=4, which pins
  `linear::native_dense_dtypes`'s claimed roster (exactly `WEIGHT_QUANTS`)
  against what actually runs. All 48 pass at ~1.6e-7 worst case.

Plus **three existing Vulkan `*_matches_host` tests moved onto the harness** —
`rmsnorm_graph_matches_host`, `linear_graph_matches_host`,
`gated_act_silu_matches_host` in `adapter.rs` — same shapes, same references,
same tolerances, ~20 lines of alloc/upload/bind/execute/download each replaced
by `infr_testkit::run_graph`. Deliberately a demonstration, not a mass rewrite.

**Corrected audit note:** the original entry's "rocm/metal have none under
`src/`" was literally true and misleading. Both have large suites under `tests/`
(rocm `parity.rs` ~97 KB / 30 tests incl. a 24-format sweep; metal `parity.rs`
~140 KB with ~60 per-format quant-Linear tests). The real gap H closes is not
"no coverage" — it is **three independent copies of the block layouts and synth
builders**, and the m-tiers no single sweep reached.

**NOT done, and why:**

- **The Metal numbers are unmeasured.** Everything else was run on the RX 7900
  XTX; Metal was only compile-checked. Its `2e-2` matches ROCm's joint bound by
  argument (same two lossy stages), while `tests/parity.rs` already holds Metal
  to `1e-3` on the shapes it picks — so the sweep is coverage-shaped, not the
  tightest bound. **Re-measure on a Mac and tighten.**
- **Metal's `m >= 16` half-fragment coop-GEMM tier is not swept.** That kernel
  rounds BOTH weights and activations to f16 before the dot, so an f32 oracle is
  the wrong comparand (`tests/parity.rs` handles it by mirroring the rounding
  into its reference via `half_ops`). Teaching the harness a "device rounds to
  f16 first" mode is follow-up.
- **The other ~70 Vulkan `*_matches_host` tests stay put.** Several (the
  `moe_ffn_*` family, the pager/mmq integration tests) build multi-op graphs and
  read back several tensors, which needs a multi-output `run_graph` the harness
  does not have; others (`mmv_mrow_symmetric_q2k_q3k`, `small_m_bench`) drive
  the raw `Recorder`, below the `Backend` seam the harness takes, and are
  testing kernel-vs-kernel agreement rather than decode-vs-oracle.
- **`dequant_codebook`'s per-arm `let bpb = 18usize;` locals** (14 of them) were
  left alone. Each sits inside its own decoder next to that format's field
  offsets, and rewiring 14 arms is typo risk for no drift protection —
  `oracle_agrees_with_the_spec_geometry_for_every_weight_quant` already fails if
  any of them disagrees with the spec, which is the property that matters.
- **No shader GENERATION from the spec.** The plan offered "generated-from or
  checked-against"; this is checked-against. Emitting block strides into GLSL
  the way `iquant_grids` emits codebooks is a bigger, separate change.

Behavior: additive. No runtime path changed (the ROCm/gguf rewires are literal
value hoists), no golden moved — cpu 23/23 incl. qwen3 `0xfd63781ea3bfa785`,
gpu_seam 26/27 (`gpu_seam_matches_cpu_qwen3_q2k` fails identically on base —
pre-existing), ROCm 30/30 legacy + 3 new, Vulkan full suite green.

Original audit below (historical). Block decode is re-implemented per shader
language (GLSL/MSL/HIP) and **cannot** be one Rust fn. But two things can be
shared: (1) a **decode spec/constants** module (block sizes, scale layout) that
shaders are generated-from or checked-against (the IQ grids already are, via
`iquant_grids`); (2) a **shared parity-test harness** built on the host
`dequant_block` oracle that every backend runs. Today vulkan has ~75
`*_matches_host` tests; **rocm/metal have none under `src/`** — they lean on the
oracle informally.

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
   **C and E landed; D landed only in part** — see its entry for what a
   `SessionChat<S>` blanket impl would have had to break, and re-scope before
   re-attempting.
3. **F. tiering policy module** — moderate, high leverage; makes the existing
   cross-tier assertions structural. **Landed in part** — see its entry for the
   prefill-GEMM split-K formulas that stayed per-backend and why.
4. **B. GraphExecutor skeleton** — the biggest structural prize, but it touches
   every backend's hot loop; do it only after A/F prove the shared-IR-pass
   pattern and with the full per-backend test suites as the guardrail. **Landed
   in part** — the residency contract (`Provision`/`writes_back`) and the walk
   (`live_ops`/`OpDispatch`) are shared; see its entry for why the residency
   container, the per-op bodies, Vulkan's recorder and cpu's inline walk stayed
   where they are.
5. **H** shared decode-spec + parity harness and **I** KV/budget math —
   spec/test and math hoists (not runtime dedup). **H landed** —
   `infr_core::decode_spec` + the `infr-testkit` harness, with cpu/rocm/metal/
   vulkan parity sweeps built from it; see its entry for the Metal numbers that
   still need a Mac and the Vulkan tests that stayed put. **I** is open.

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
