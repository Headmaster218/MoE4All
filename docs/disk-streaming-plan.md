# Tiered weight paging — VRAM → DRAM → disk

Plan for running models that fit neither VRAM nor DRAM, by extending the
existing block pager into a **tiered** one whose bottom tier is the model file
itself, read by explicit positioned I/O rather than left to the OS page cache.

Status: **design, nothing implemented.** Claims about the tree name the file and
symbol they came from. Numbers this plan does not have are marked as phase-0
measurements rather than estimated here; the one worked example in §2 is
labelled as illustrative arithmetic, not a prediction.

## 1. What exists today

The residency machinery is already block-agnostic and already has two policies:

- `infr_core::pager::Pager` — pure host-side bookkeeping (`n_slots` slots, a
  `BlockId → slot` map, LRU order, batch epochs). No bytes, no device types.
  Three entry points: `touch` (recency/LRU, MoE decode), `touch_cold`
  (scan-resistant, prefill sweeps) and `schedule` (exact cyclic sweep,
  Belady-parity — dense layer streaming). `ring_bytes` prices the staging ring.
- `infr_vulkan::pager::GpuPager` — a `Pager` plus a VRAM slot arena (BDA), a
  device LUT, and uploads through a reused pinned staging ring.
  `MoePagerSession` drives it with `(layer, role, expert)` blocks;
  `DensePagerSession` drives it with per-layer weight-group blocks
  (`DenseSource`, `DensePoolSpec`).
- The **source of bytes** for both is zero-copy GGUF mmap views:
  `DenseSource::segments` is `Vec<Arc<dyn AsRef<[u8]> + Send + Sync>>`, built in
  the Vulkan binder (`infr-llama/src/seam/mod.rs`) by calling
  `Gguf::tensor_bytes_arc` once per component tensor.
- Placement is decided once per load in the seam: the MoE tier ladder, then the
  dense try-resident → smaller-ubatch → auto-q8 → stream ladder, priced against
  `Backend::device_alloc_room`.
- The CPU backend never copies weights: `CpuBuffer::Mapped(TensorBytes)` reads
  straight out of the mapping (`CpuBackend::map_weight`).
- The Metal backend has **no pager at all** and does not mmap: `alloc`/`upload`
  put every weight in a `StorageModeShared` `MTLBuffer`, i.e. wired anonymous
  memory. A model that exceeds RAM on Apple silicon cannot load at all today.

So the missing tier is exactly one: **below DRAM**. Above it, the policy layer
exists and is unit-tested.

### 1.1 What "DRAM tier" means today — and why it is not enough

Today the DRAM tier is the OS page cache, reached through the GGUF mmap
(`Gguf::open` maps the file and `madvise`s `WillNeed` + `HugePage`). That
already runs a model bigger than RAM — the kernel demand-pages it — so this
feature is **not** "make it possible". It is "make it not thrash":

1. **Wrong eviction policy.** The page cache is recency-based. A dense forward
   pass is a cyclic sweep over the whole weight set, which is LRU's pathological
   case: every page evicted exactly before its next use. Same pathology
   `Pager::touch_cold`/`schedule` exist to fix (that doc records the in-tree
   measurement: 768/768 blocks re-uploaded per rep on Scout pp512 under plain
   LRU). We can apply Belady-parity policy per tier; the kernel cannot, because
   it does not know the sweep order.
2. **No prefetch we control.** Kernel readahead is a sequential heuristic over
   file offsets. Our access order is known for the whole pass on dense, and one
   layer ahead for MoE on CPU (the router runs on the host), so reads can be
   issued before the stall instead of after the fault.
3. **No accounting.** Nothing can answer "what fraction of my weight working set
   is resident", so the placement ladder is blind below VRAM and the user gets
   no honest report of what the run will cost.

A fourth reason is about degree, not category: `DensePagerSession::stage`
memcpys mmap bytes into the pinned ring while holding the session mutex (see
`stage_dense_linear` in `infr-vulkan/src/adapter.rs`), and `schedule_staged`'s
copy is a rayon parallel section under that same `std::sync::Mutex`. A major
fault there **already** blocks on disk under the lock. An explicit read does not
invent that stall, it lengthens it — and, unlike the fault, it can be moved off
the critical path by prefetch. The fence wait that follows is outside the guard,
so the guard is not held across submission.

**Phase 0 measured the premise — DONE.** `scripts/paging-baseline.py` runs the
CPU backend under a cgroup-v2 `MemoryMax` with the model's page cache dropped
first, so the weights genuinely do not fit the memory the process may use. Full
table in `docs/perf/results.md`; on Llama-3.2-1B F16 (2.48 GB):

- **Decode collapses 23–33×** once the cap bites (22.5 t/s unlimited → 0.96 at 2
  GB → 0.67 at 1.5 GB), with 420–460 k major faults per 32-token run.
- **Prefill is flat** (46.9 → 46.6 t/s, −0.6%), reading 3.7 GB for the whole run
  — one weight sweep amortized over 512 tokens.
- At 1.5 GB decode moved **153 GB for 32 tokens = 1.9× the whole model per
  token**, where never caching anything at all would have read 1.0×. Recency
  eviction against a cyclic sweep, plus 4 KiB granularity and readahead, costs
  nearly double what a policy that simply gave up would.

That last line is the bar: a tier whose per-pass traffic is
`model − VRAM_home − DRAM_home` and whose I/O is issued in whole blocks starts
from under half the bytes, before any hit-rate benefit. The gaps in the baseline
— no GPU-side figure, no genuinely-over-RAM blob on this host — are recorded
with the table.

### 1.2 The constraint prior art already established

Backlog **B30** records a rejected experiment: copying the GGUF into an
anonymous mapping. Measured on a 16 GiB Qwen3.6-27B, warm cache: load 1.87 s →
10.5 s (5.6x), and 14 GiB of evictable page cache became 20.2 GiB of anonymous
RSS.

- **Anonymous, non-evictable host memory is the expensive kind.** Our DRAM arena
  is exactly that, so its budget is explicit, bounded, and off unless the
  placement plan says the model does not fit.
- **The mmap fast path must survive untouched.** When the model fits, we keep
  zero-copy mmap views and pay nothing — no arena, no copies, no worker threads.

## 2. Physics — what this can and cannot deliver

Per forward pass, with each block assigned one home tier (§3.5):

```
disk→host bytes = model − VRAM_home − DRAM_home
host→VRAM bytes = model − VRAM_home                    [discrete GPU only]
t_pass         ≈ max(disk_bytes/disk_bw, host_bytes/pcie_bw, t_compute)
```

The two boundaries carry different byte counts, so the pass is bounded by the
slower of the two transfers, not by one ratio.

For a dense model there is no locality to exploit — every weight is read exactly
once per token — so decode throughput is that bound divided by one token.
Illustrative arithmetic with round numbers (**not** a prediction; phase 0
supplies the real `disk_bw`, and `VRAM_home` is what is left after KV, the
activation reserve `dense_act_reserve_at` prices, and the staging ring, not the
raw card size): a 40 GB model with 20 GB of VRAM home and 8 GB of DRAM home
streams 12 GB/token from disk; at a few GB/s that is well under one token/s.
**That is the honest ceiling and it is the point** — it turns "cannot run" into
"runs", and the same bytes amortize across a whole prefill chunk, so prefill
stays usable while interactive decode does not. §7 turns this into a decision.

The phase-0 baseline is that same split measured from the other side: under a
hard memory cap, prefill lost 0.6% and decode lost 23–33×, because only decode
pays a whole weight sweep per token (§1.1).

MoE is the opposite case and the real prize: routing is skewed, so a small hot
set carries most tokens and the in-tree MoE pager already runs at a high
steady-state hit rate on Scout prefill (recorded in the pager campaign notes;
phase 0 re-measures it rather than carrying the figure forward). A DRAM tier
below it only has to cover the cold tail. The dense parts of a MoE model
(attention, norms, shared expert, router, embeddings, lm_head) stay fully
resident — small, and hit every token.

## 3. Architecture

### 3.1 Module layout

Two new modules in `infr-core`, and **no reorganisation of what exists**:

- `infr-core/src/blockio.rs` — `BlockExtent`, `BlockDesc`, the `BlockIo` trait,
  `FileBlockIo`, and the reader pool.
- `infr-core/src/hostpager.rs` — `HostPager`: the DRAM tier (arena, pins,
  prefetch queue), built on today's `infr_core::pager::Pager`.

`pager.rs` stays where it is. Folding all three into a `paging/` directory is
churn until there is a third reason to; do it then, with re-exports.

Both modules stay device-free: no `ash`, no `metal`, no GGUF types. That is what
lets three backends share them and what keeps the policy unit-testable without
hardware, as `pager.rs`'s own test module already is.

### 3.2 The block model

A **block** is one unit of paging: exactly what the seam's `wload` closure
(`infr-llama/src/seam/runner.rs`) uploads as a group — a fused qkv triple, a
fused gate+up pair, a single projection, one MoE expert's one role. That
definition already exists on both sides and keeping it is what stops the plan
and the loader drifting.

```rust
/// One contiguous byte range of the model file.
pub struct BlockExtent { pub offset: u64, pub len: usize }

/// A block's identity and where its bytes are, in upload order. A fused group
/// lists one extent per component tensor.
pub struct BlockDesc { pub id: BlockId, pub extents: Vec<BlockExtent>, pub nbytes: usize }
```

`TensorInfo` already carries `offset`/`nbytes` and `Gguf` knows
`data_region_start`, so the extents are derivable — but `Gguf` keeps only the
`Mmap`, not the path or the `File` (`Gguf::open` drops the descriptor). So
`infr-gguf` must **gain** a retained path (or `File`) plus an accessor returning
`(absolute_offset, len)` for a named tensor. Single file only: `Gguf::open` maps
one file, and sharded-GGUF support is deferred (§7).

**Blocks that cannot be re-read from the file.** The seam rewrites some tensors
at load — the qwen2 NEOX q/k row permute, the BitNet `I2S` → f16 dequant — and
the Vulkan dense streaming path already excludes them for that reason. It also
excludes dtypes whose kernels take no weight offset (`native_dense_supported`;
F16/F32 are out, so an f16 checkpoint has **no** streamable dense weights on
Vulkan today). Both exclusions are conditions on the tier planner, not
afterthoughts: the eligibility predicate moves out of the Vulkan-only
`dense_plan` block in `seam/mod.rs` and next to `fuse_gu_decision` /
`fuse_qkv_decision`, where the other shared enumeration rules live, so CPU and
Metal planning read the same predicate.

**The fused-group concat.** `wload` today materializes every multi-name group
into `WBytes::Owned(Vec<u8>)`, and the Vulkan streamed binder then ignores those
bytes and re-fetches per-component mmap views — the concat is built and dropped.
Paged blocks must not build it at all, so `wload` needs a "this group is paged"
arm that computes the group's byte total (for the existing drift guard, which
compares the plan's segment total against `tb.len()`) without concatenating.

### 3.3 Pins — the change the core `Pager` needs

Both new consumers hold a block's bytes across work the pager does not see: a
CPU kernel reads a whole weight for the duration of an op; a GPU staging copy
reads a host slot until the copy is recorded. A slot must be un-evictable while
borrowed, and the batch epoch cannot express it (per-batch, while a borrow is
per-op).

This is a bigger change to `Pager` than "add two methods":

- `pin(id)` / `unpin(id)` refcounts, and `take_slot` skipping pinned entries.
- `take_slot`'s all-slots-unavailable path currently **panics** (the
  within-batch assert). Pin exhaustion is runtime-reachable, so it must return
  an error — and `touch`/`touch_cold`/`schedule` return `Resolution`, not
  `Result`. Making exhaustion recoverable changes three public signatures and
  every caller in `infr-vulkan`. Budget for that, or keep exhaustion a panic and
  make the **sizing floor** a load-time check that cannot be violated at
  runtime.

**The sizing floor is not per-pass, it is per-pass × concurrency.**
`infr-server` admits `n_parallel` concurrent generations through one backend
(`slots: Arc<Semaphore>`), each at its own layer. The floor is therefore
`n_parallel × (max blocks pinned by one op)` — one layer's weight groups for
dense, top-k experts per layer for MoE. `TierPlan` prices that, and a budget
that cannot cover it is rejected at load with the knob named. Incremental
blocking acquisition of an unordered pin set across N requests is a deadlock, so
if the budget is tight the fallback is one permit for paged forwards
(serialize), stated in the banner — never a wait.

**The arena needs an aliasing argument, not a lifetime trick.**
`HostPager::pin(&self) -> Pin<'_>` hands out `&[u8]` into an arena other threads
mutate through the same `&self`. That is interior mutability with a real
soundness obligation: slot storage in `UnsafeCell`, the invariant "a pinned slot
is never written, and only the reader that owns the slot writes it before the
pin exists", and something that **enforces** it (the pin refcount, checked
inside the same lock that hands out slots) rather than a comment. Phase 1 runs
Miri over this module — the workspace already has a weekly miri job for
`SpinPool`, and this is exactly the kind of code it exists for.

```rust
pub struct Pin<'a> { /* Deref<Target = [u8]>; Drop → unpin */ }
impl HostPager {
    pub fn pin(&self, id: BlockId) -> Result<Pin<'_>>;      // blocking: may read disk
    pub fn try_pin(&self, id: BlockId) -> Option<Pin<'_>>;  // hit-only, never reads
    pub fn prefetch(&self, ids: &[BlockId]);                // queue; never blocks
}
```

### 3.4 Address-keyed caches — the silent-corruption trap

Three in-tree caches key on a **weight slice's address**, which is stable only
because weights are mmap'd or uploaded once. A paged slot reuses one address for
different blocks at identical length, so those keys collide and hand back
another block's data — plausible garbage, not an error:

- `infr-cpu`'s `repack_cache` / `repack6_cache`: `q4k_pack_for` / `q6k_pack_for`
  key on `(w.as_ptr() as usize, w.len())`. **Must** be re-keyed on `BlockId`
  before any CPU paging lands.
- `infr-cpu`'s `weight_cache` is safe as written — it keys on the `CpuBuffer`
  object address, not the slice — but it caches a dequantized f32 copy, so it is
  a budget consumer, not a correctness one.
- `infr-metal`'s `qui_cache` keys on `MTLBuffer::id()`, so a placeholder per
  block keeps it correct; but its factored arm **copies the transformed weight
  out** and retains it unboundedly. On a paged Metal model that is a second full
  copy of every touched block in host RAM, which defeats the whole budget. The
  Metal tier is therefore gated on the native-kernel arm (Q4_K/Q6_K read the
  bound buffer directly), with the factored path either bypassed or budgeted.

Every one of these is a cache whose key encodes an assumption paging breaks. The
audit for "what else keys on an address" is part of phase 1, not a later
cleanup.

### 3.5 The I/O engine

```rust
pub trait BlockIo: Send + Sync {
    /// Fill `dst[..desc.nbytes]` from the block's extents, in order.
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()>;
}
```

`FileBlockIo` uses positioned reads — `std::os::unix::fs::FileExt::read_at` /
`std::os::windows::fs::FileExt::seek_read` — so there is no shared cursor and no
seek/read race between workers. No new dependency. A fixed worker pool serves
the prefetch queue; thread count and prefetch depth are **hardcoded** until a
measurement demands a knob (§4).

A trivial in-memory `BlockIo` in `infr-testkit`, with fault injection (short
reads, errors, delays), is what makes every tier assertion able to go red.

**Deliberately not in v1**, recorded so it is not re-proposed: io_uring
(Linux-only, new dependency, and the win is queue depth threads already give);
`O_DIRECT`/`F_NOCACHE` (alignment constraints, and it loses whenever the page
cache is helping); mapping the file and memcpying from it (that is the page
cache again, with the fault-under-the-lock problem intact).

**Double caching is real.** A buffered `read_at` leaves a page-cache copy of
what we also hold in our arena, which on a model far larger than RAM halves
effective DRAM. `posix_fadvise(DONTNEED)` / `F_NOCACHE` is the phase-5 lever,
gated on the phase-0 counters.

**The file can change under us.** `Gguf::open` documents this as an unenforced
invariant and `infr_gguf::watch::WeightWatch` only detects it at CLI/request
checkpoints (backlog B30). Runtime re-reading makes it worse than it is with a
mapping: a truncate gives short reads and a rewrite gives silently different
bytes mid-generation. Minimum handling: `FileBlockIo` records
`(len, mtime, ino)` from its own fd at open and re-stats **once per forward
pass** (not per read), failing the generation loudly on a change. That is one
syscall per pass and it turns a silent-wrong-output class into an error.

### 3.6 Tier policy — two shapes, chosen by model class

**Dense: partition, do not cache.** For a cyclic sweep, caching the same block
in two tiers is waste — a block resident in VRAM never needs a DRAM copy. So
`TierPlan` assigns each block one home tier at load: VRAM home takes as many
blocks as its budget holds minus a streaming window, DRAM home takes as many of
the remainder as its budget holds, and everything left streams from disk each
pass into those windows under the existing `schedule` (cold-insert,
Belady-parity) policy. Per-pass disk traffic is then
`model − VRAM_home − DRAM_home`, which is the minimum any policy achieves on a
sweep.

**MoE: cache, and fill from the read path only.** Routing is skewed and
unpredictable, so both tiers cache with the existing policies (`touch` for
decode, `touch_cold` for prefill sweeps). The DRAM tier is filled **only** from
disk reads — never by copying a block back from VRAM, which spends a device→host
transfer on bytes that can be re-read from disk instead.

**Prefetch:**

- Dense, any backend: the order is known for the whole pass; issue block `l+k`'s
  read when layer `l` starts.
- MoE on CPU: the router runs on the host, so layer `l`'s ids are known before
  its FFN executes — prefetch at the router, execute after.
- MoE on GPU: ids arrive by device→host readback per layer (the paged `MoeFfn`
  arm in `adapter.rs`), so the next layer's ids do not exist yet and there is no
  exact prefetch to be had. v1 does nothing clever here; frequency-warmed
  promotion is a phase-5 lever, not part of the architecture.

### 3.7 Backend integration

`WBytes` (`seam/mod.rs`) gains a `Paged(BlockDesc)` variant — and **loses its
`Deref<Target = [u8]>` impl in the same change**. The `Deref` is infallible, so
leaving it forces a panic arm and silently keeps compiling at the sites that use
it (`pipeline_binder`'s `pad_to_u32_align(&tb)`, `tensor_parallel_binder`'s
`tp_slice_column(&tb, …)` / `tb.to_vec()`). Replacing it with
`fn bytes(&self) -> Result<&[u8]>` makes the compiler enumerate every site that
must now handle a paged block. A binder that receives `Paged` registers the
block with its backend's pager instead of allocating and uploading — what the
Vulkan dense binder already does via `DensePagerSession::register`.

**CPU (`infr-cpu`).** New `CpuBuffer::Paged` beside `Mapped`/`Owned`, and a
`CpuRead::Pinned(Pin<'_>)` variant so the existing `Deref<Target = [u8]>`
uniformity carries every kernel unchanged. `CpuBuffer::read` returns no
`Result`, so the disk error must surface **before** it: the interpreter pins
every weight an op names in a fallible pre-step, then executes with `read()`
infallible over already-pinned slots. That pre-step is also where prefetch for
the next op is issued. This is where tiering matters most — today the CPU
backend's only answer to "bigger than RAM" is the page cache.

**Vulkan (`infr-vulkan`).** `DenseSource`'s segments become a source enum: mmap
view (fast path, unchanged) or paged block. `stage` then has three cases: VRAM
hit (unchanged); VRAM miss with DRAM hit (memcpy DRAM → pinned ring); VRAM miss
with DRAM miss (read disk **straight into the pinned ring** — it is already host
memory — and admit to DRAM separately per policy). Never two copies of the same
bytes on the critical path. The `-DSTREAMED` shader twins are **not** a new
cost: `build.rs` already makes the streamed form the sole weight build.

**Metal (`infr-metal`).** Unified memory collapses the VRAM and DRAM tiers into
**one**: an arena slot is host memory the GPU reads directly, so a disk read
lands in the final destination with no staging ring and no second copy. Slot
offsets need no new kernels — `set_buffer` takes a per-binding byte offset and
`exec.rs` already binds weights at non-zero offsets; weights bind to `device`
address space, so the only alignment obligation is the 4-byte one the encoder
path already checks, and `slot_bytes % 4 == 0` (which the Vulkan arena already
requires). The real Metal work is the arena + `Pager` + `read_block` into
`contents()` at the slot offset, plus the `qui_cache` gate from §3.4. The same
single-tier collapse applies to **UMA Vulkan** (APU / iGPU / Strix Halo).

**Out of scope, stated so it is not assumed:** the multi-GPU wrappers
(`tensor_parallel_binder`, `expert_parallel_binder`, `pipeline_binder`) bypass
the pager entirely today — EP keeps experts resident per rank — and a host tier
under N devices needs its own budget story. MTP declares its own `BindWeight`
and loads a second weight set, which is a second unbudgeted claim on the tier.
Both keep the mmap path until a later slice.

### 3.8 Placement

`TierPlan` is pure arithmetic in `infr-core`, fed by the seam: the block list
with sizes, the VRAM ceiling (`Backend::device_alloc_room` — the measurement
that already outranks estimates), the host DRAM budget, the KV and activation
reserves the ladder already prices, and `n_parallel`. It returns each block's
home tier, the per-tier window sizes, and the numbers for the banner.

The existing ladder gains one rung at the bottom, and the banner states per-pass
disk bytes and the throughput ceiling they imply — so a user sees the number
before waiting for it.

**Host DRAM budget.** There is no host-memory probe in the tree. Explicit
`paging.dram` always wins; otherwise probe via the existing `libc` dependency
(`sysconf(_SC_PHYS_PAGES)`/`_SC_AVPHYS_PAGES`, `sysctl hw.memsize` + `vm_stat`).
Where no probe exists — Windows, which CI does not build (B30) — an explicit
budget is required and the absence is reported, never guessed. See §7.

## 4. Configuration

Two new keys in the `paging` section of `infr-core/src/config/manifest.rs`,
following the spellings already there (`paging.cache`, `paging.ring`,
`paging.stats`):

| env                | key           | type | meaning                                                         |
| ------------------ | ------------- | ---- | --------------------------------------------------------------- |
| `INFR_DRAM_CACHE`  | `paging.dram` | Size | DRAM tier budget; `0` disables the tier (mmap path unchanged)   |
| `INFR_DISK_STREAM` | `paging.disk` | Flag | Enable the disk tier; unset = auto, only when the plan needs it |

Reader threads, prefetch depth and page-cache dropping are **hardcoded** until a
measurement says otherwise — a knob with no known good value is a question
shipped to the user. `INFR_CACHE` (`paging.cache`) keeps its current meaning
(the VRAM paging budget); `INFR_PAGER_STATS` grows per-tier lines.

## 5. Phasing

**Phase 0 — measure the baseline (no code). DONE.** `scripts/paging-baseline.py`
and its table in `docs/perf/results.md`; headline in §1.1. The bar every later
phase is judged against, re-run with the same harness.

**Phase 1 — core, no backend wired. DONE.** `blockio.rs` (`BlockDesc`,
`FileBlockIo`, the file-replaced stamp), `hostpager.rs` (arena, `Pin`, the
`Loading`/`Ready` handshake), pins in `Pager`, `Gguf::tensor_file_range` /
`Gguf::path`, and the address-keyed cache fix from §3.4 (`CpuBuffer::uid`).
Verified by unit tests over a fake `BlockIo` — content correct after churn, a
pinned block never evicted, exhaustion surfacing rather than corrupting, failed
reads leaving nothing resident, extent order deciding the layout, the
file-change check firing — each shown to fail by breaking what it guards. The
arena's unsafe is clean under Miri (tree-borrows), now a weekly cron step.

Deferred out of this phase, deliberately: **`TierPlan`** (its only consumer is
the phase-2 placement decision; a plan type with no caller is machinery shaped
by a guess) and the **prefetch worker pool** (same reason — the pool's depth and
thread count are meaningless until something measures them, and `HostPager::pin`
reads synchronously in the meantime).

**Phase 2 — CPU backend on the DRAM tier. DONE (prefetch deferred).**
`CpuBuffer::Paged` + `CpuRead::Pinned`, the per-op pin pre-step driven by
`Op::io()`, `infr_cpu::paged` (one pool per weight-size class, planned up front
from the GGUF's tensor directory), the `paging.dram` key, and the
`INFR_PAGER_STATS` per-pool report. Measured against phase 0 in
`docs/perf/results.md`: decode 1.28x at a 2 GB cap and **2.06x at 1.5 GB**,
major faults 210-335x lower, and read volume flat as the cap tightens where
mmap's grows. Prefill costs 3-7.5%.

Prefetch is still not built: the synchronous read is what the numbers above
already beat, and the pool/depth knobs have no measured values yet.

The ORIGINAL phase-2 verification, for the record: greedy token identity against
the CPU reference path with the budget forced small enough to churn; hit rates
matching the policy's predicted `(n_slots − 1) / n_blocks` per sweep; a model
larger than the budget completing at all; and beating phase 0 on throughput
**and** major-fault count.

**Phase 3 — Vulkan third tier.** The source enum and the three-case stage path.
Verification: the pager parity tests currently cover
`ensure_resident`/`flush_lut` only, so this phase writes new ones for
`schedule_staged`, the ring, and `DensePagerSession::stage`, with budgets tuned
to force each of the three cases — asserting each case is actually taken, since
a case that never runs looks identical to one that works. Validation layer
clean; streamed vs resident logits identical.

**Phase 4 — Metal / UMA collapse.** Arena, pager, offset binding, the
`qui_cache` gate. Verification: Metal decode parity against CPU reference under
a forced-tiny budget, and the first over-RAM model to load on Apple silicon at
all.

**Phase 5 — levers, each gated on a measurement.** `fadvise`/`F_NOCACHE`;
frequency-warmed DRAM for MoE-on-GPU; io_uring if the pool proves queue-depth
bound; exclusive VRAM/DRAM placement for MoE; multi-GPU and MTP coverage.

## 6. Verification rules specific to this feature

- **Every tier transition is observable.** `INFR_PAGER_STATS` reports per tier:
  hits, misses, evictions, bytes read, and for disk how many reads were served
  from a completed prefetch versus blocked on the critical path. A prefetch that
  silently never fires is indistinguishable from one that works in every metric
  except throughput.
- **Correctness under churn is what breaks silently.** A wrong slot, a torn
  multi-extent read, an eviction under a live pin, or a stale address-keyed
  cache entry all produce plausible garbage rather than an error — hence
  content-checked tests at every tier, not residency-count tests.
- **Report the ceiling.** The banner states per-pass disk bytes and the implied
  throughput. A user waiting on sub-1 t/s should have been told.

## 7. Decisions

Made here, changeable by the user:

1. **The dense disk tier is opt-in, not an automatic ladder rung.** At the §2
   ceiling, silently choosing it for an interactive decode turns a clean "does
   not fit" into a session that looks hung. `paging.disk` (or an explicit
   `paging.dram`) engages it; the auto ladder stops where it does today unless
   asked. MoE is different — skewed routing makes it genuinely fast — so MoE may
   take the tier automatically.
2. **Sharded GGUF is out of scope for v1.** `Gguf::open` maps one file; the
   `-NNNNN-of-MMMMM` set is understood only by `infr-hub`'s downloader. A paged
   sharded model is rejected at load with that reason, and `BlockExtent` carries
   no file id until a second file exists to point at.

Needing the user's call:

3. **Windows host-memory probe.** `libc` covers Linux and macOS; Windows needs
   `GlobalMemoryStatusEx`, i.e. a new dependency (`sysinfo` or `windows-sys`).
   Recommendation: neither — Windows requires an explicit `paging.dram` and says
   so. CI builds only ubuntu and macos today.

## 8. Non-goals

KV cache on disk (`INFR_KV_OVERFLOW` already spills KV to host, and KV has
different access physics); compressed or re-quantized on-disk formats; network
or object-store block sources; training. Each changes the block model and
belongs to its own plan.
