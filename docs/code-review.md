# Code review — correctness, security, DRY, YAGNI

**Date:** 2026-08-01 · **Tree:** `main` @ `97a029d` · **Scope:** the 14
workspace crates (~135 k lines of Rust across `crates/*/src`, plus build scripts
and shader emitters).

## Method and honest scope

This is a **risk-prioritised** review, not a line-by-line audit of every file.
The reading order was: untrusted-input parsers (GGUF, tool-call bodies, chat
templates, HTTP DTOs) → network/filesystem code (`infr-hub`) → the
network-facing server → `unsafe` blocks (`infr-cpu/pool.rs`,
`infr-vulkan/lib.rs`, the SIMD kernels) → shared arithmetic (`infr-core`:
`budget`, `pager`, `decode_spec`, `config`) → sampling/tokenizer → spot checks
in the CLI and the big backend files.

**Not covered in depth:** `infr-vulkan/src/recorder.rs` (11.6 k lines),
`infr-vulkan/src/adapter.rs` (9.4 k), `infr-cpu/src/kernels.rs` beyond a
representative sample of the SIMD family, `infr-metal/src/exec.rs`, and
`infr-llama/src/seam/*`. Those are covered by the project's own parity/golden
suites; a targeted follow-up pass on them is worth scheduling separately.

Overall the codebase is in unusually good shape for its size: the parsers are
hardened against the obvious overflow/allocation attacks, the `unsafe` blocks
carry real safety arguments rather than boilerplate, and most modules pin their
behaviour with characterization tests against a named oracle. The findings below
are mostly edge cases, asymmetries between sibling code paths, and accumulated
dead scaffolding.

**Severity legend**

|       | meaning                                                       |
| ----- | ------------------------------------------------------------- |
| **H** | wrong results, memory-unsafety, or a remotely reachable crash |
| **M** | user-visible bug, missing hardening, or a real robustness gap |
| **L** | nit, latent risk, or cleanup                                  |

---

## 1. Correctness bugs

### C1 (H) — `INFR_SEED`/`--seed` collapses adjacent seeds; the documented fix only landed on the per-request path

`crates/infr-llama/src/sampling.rs:377-384`

```rust
pub(crate) fn seed_rng(cfg: &infr_core::config::SamplingCfg) -> u64 {
    cfg.seed.unwrap_or_else(|| { /* wall clock */ }) | 1
}
```

`resolve_seed` (line 287) was fixed to stop collapsing seeds — it remaps only
the degenerate `0` and passes every other value through:

```rust
Some(0) => 0x9E37_79B9_7F4A_7C15,
Some(s) => s,
None => seed_rng(cfg),
```

But the `None` arm falls through to `seed_rng`, which still applies `| 1`. So
the fix covers a **per-request** `seed` (OpenAI `"seed": 2`) and misses the
**process** seed. `--seed 2` and `--seed 3`, and `INFR_SEED=2` / `INFR_SEED=3`,
produce byte-identical output.

The regression test misses it because it pins an odd value:

```rust
// sampling.rs:627
assert_eq!(seed_rng(&pinned), 47);   // 47 | 1 == 47
```

**Fix:** move the `Some(0) => …` remap into `seed_rng` and drop the `| 1`;
extend `adjacent_seeds_produce_distinct_streams` to drive
`SamplingCfg { seed: Some(2/3) }` as well as `RequestSampling`.

### C2 (H) — `Backend::download` and `copy_buffer` have no bounds check (Vulkan)

`crates/infr-vulkan/src/lib.rs:3262`, `:3359`

`upload` validates:

```rust
if src.len() > vk_dst.size { return Err(be(format!("upload: {} bytes into a {}-byte buffer", …))); }
```

`download` does not. On the mapped-pointer path it issues

```rust
unsafe { std::ptr::copy_nonoverlapping(ptr as *const u8, dst.as_mut_ptr(), dst.len()) };
```

with `dst.len()` unchecked against `vk_src.size`. A caller passing an oversized
host slice reads past the end of the mapping — an out-of-bounds read of adjacent
VRAM/host memory (undefined behaviour, and an information leak into the caller's
buffer). On the staging path the same mistake becomes a `vkCmdCopyBuffer` whose
`size` exceeds the source buffer, which is a VUID violation the validation layer
will flag but the driver may not.

`copy_buffer(src, dst, bytes)` checks neither side's size against `bytes`, and
additionally ignores that a `Backing::BdaSub` sub-tensor's usable extent is
smaller than its block's `vk::Buffer`.

**Fix:** mirror `upload`'s guard in both, and include `sub_offset` in the extent
(`sub_offset + len <= mem_size`).

### C3 (M) — GGUF tensors whose element count is not a whole number of blocks are silently mis-sized

`crates/infr-gguf/src/lib.rs:268-280`

```rust
fn tensor_nbytes(dtype: DType, numel: usize) -> usize {
    if dtype == DType::I2S { return numel / 4 + 4; }
    let (be, bb) = block_layout(dtype);
    (numel / be) * bb
}
```

Integer division truncates. A crafted (or merely corrupt) GGUF declaring, say,
`Q4_K` with `numel = 100` gets `nbytes = 0`; `resolve` then happily returns a
zero-length slice that passes the file-size bounds check, and downstream
dequant/upload code indexes into it. The `I2S` arm has the same issue
(`numel = 3` → 4 bytes, all scale, no codes).

llama.cpp validates `ne[0] % blck_size == 0` at load. `infr` does not, and
`Gguf::open`'s otherwise-thorough hardening (overflow-checked `numel`, clamped
`with_capacity`, alignment validation, bounds-checked `resolve`) makes the
omission stand out.

**Fix:** in the `TensorInfo` conversion loop, reject `numel % block_elems != 0`
with an `Error::Loader`, and reject `numel % 4 != 0` for `I2S`.

### C4 (M) — a sharded/subdirectory GGUF on HuggingFace cannot be pulled, and re-pulls forever

`crates/infr-hub/src/pull.rs:94-119`, `crates/infr-hub/src/store.rs:53-73`

`repo_info` returns raw `rfilename`s from the HF model API. Those legitimately
contain subdirectories — unsloth's Dynamic-quant repos ship
`UD-Q4_K_XL/Model-UD-Q4_K_XL.gguf`, and the code already knows this (`is_mmproj`
does `name.to_lowercase().rsplit('/').next()`). Three things then go wrong:

1. `fetch_and_link` does `snap.join(filename)` and `symlink(…, &link)` without
   creating the intermediate directory → `ENOENT`, the pull fails.
2. Even with the directory created, the relative link target is wrong: the code
   always writes `../../blobs/{hex}`, which is only correct for a filename at
   the snapshot root. A one-level subdirectory needs `../../../blobs/{hex}`.
3. `Store::resolve_repo` enumerates the snapshot with a **non-recursive**
   `fs::read_dir`, so a subdirectory GGUF is never seen as cached — every
   `infr run` re-pulls multiple GB.

Related: `gguf_match`'s explicit-filename arm compares
`fname.eq_ignore_ascii_case(sel)`, so `org/repo:Model-UD-Q4_K_XL.gguf` can never
match the API's `UD-Q4_K_XL/Model-UD-Q4_K_XL.gguf`.

**Fix:** create `link.parent()` before `symlink`; compute the link target from
the filename's component depth; make `resolve_repo` walk the snapshot
recursively (or match on the path's final component).

### C5 (M) — `Pager::new(0)` panics on first use; the batch-overflow path is a hard panic

`crates/infr-core/src/pager.rs:226-258`

With `n_slots == 0`, `free` is empty and `lru` is empty, so `take_slot` runs
`self.lru.iter().position(…)` → `None` → the `unwrap_or_else` asserts
`cur_epoch == 0` and returns index `0` → `self.lru.remove(0)` → `None` →
`.expect("index from position()")` panics. Nothing guards `n_slots >= 1` in
`Pager::new`.

Separately, the documented "fail loudly" path is a bare `assert!`, i.e. an
engine-wide panic when the arena budget is too small to hold one dispatch batch.
The module's own docs say this is "a configuration error the caller should
surface" — surfacing it as a panic means an OOM-adjacent config produces a stack
trace rather than the actionable "increase `INFR_CACHE`" message the VRAM guard
produces for the analogous case.

**Fix:** `debug_assert!`/clamp `n_slots` to `>= 1` in `new`; make `take_slot`
return `Result` (or have callers pre-validate the batch size) so the message
reaches the user as an error.

### C6 (M) — the tool-call value parser scans for `:` across the whole remaining buffer

`crates/infr-chat/src/tools.rs:118-126`

```rust
let colon = match s[i..].iter().position(|&b| b == b':') { … };
let raw_key = String::from_utf8_lossy(&s[i..colon]).trim().trim_matches('"')…;
```

The key scan is not bounded by the object's closing brace or by string quoting.
For a body like `{foo}` (no colon at all in the object but one inside a later
value, e.g. a URL or a timestamp), the "key" swallows everything up to that
far-away colon, and the value parse resumes mid-token. Malformed model output
then produces a _plausible-looking_ tool call with a garbage argument name
rather than being rejected.

**Fix:** bound the colon search at the first unquoted `}`/`,` and treat a
missing colon as end-of-object (the `None` arm already returns; it just needs to
fire).

### C7 (L) — `sample_logits` panics on an empty logits slice

`crates/infr-llama/src/sampling.rs:517-535`

For `logits.len() == 0` and `temp > 0`, `truncated_softmax` takes the
`top_k == 0` branch, returns empty vectors, and `idx[idx.len() - 1]` underflows.
The greedy path returns token id `0` for an empty slice, which is silently wrong
rather than loud. Not reachable today (vocab is always non-empty), but it is a
one-line guard.

### C8 (L) — `Penalties::apply` indexes `logits` by raw token id

`crates/infr-llama/src/sampling.rs:343`, `:353`

`logits[t as usize]` panics if a generated id ever exceeds the logits row
length. Today the row is full-vocab so this holds, but a truncated/pruned logits
row (an lm-head slice, a draft head with a smaller vocab) would turn a repeat
penalty into a crash. Prefer `if let Some(l) = logits.get_mut(t as usize)`.

### C9 (L) — SPM merge reconstruction sorts f64 scores with `partial_cmp`

`crates/infr-llama/src/tokenizer.rs:177-181`

```rust
cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal).then(…));
```

A `NaN` score in `tokenizer.ggml.scores` (a corrupt GGUF) makes the comparator
non-transitive, which recent `sort_by` implementations detect and **panic** on
("user-provided comparison function does not correctly implement a total
order"). Use `f64::total_cmp`.

---

## 2. Security and robustness

### S1 (M) — unbounded recursion in two parsers reachable from untrusted input

Both are stack-overflow (SIGSEGV / abort, not a catchable panic) denials of
service:

- `crates/infr-gguf/src/lib.rs:196-207` — `read_meta_value`'s `ARRAY` arm
  recurses on `elem_type`. Nesting costs 12 bytes of file per level, so a ~1 MB
  crafted GGUF nests ~85 k deep. Reachable by anyone who can get a user to
  `infr run` a downloaded model.
- `crates/infr-chat/src/tools.rs:94` — `parse_value` recurses on `{`/`[` with no
  depth limit. Its input is **model output**, which in `infr serve` is steerable
  by the requesting client via prompt injection
  (`"reply with <|tool_call>call:x{a:{a:{a:…"`). One request can kill the server
  process.

**Fix:** thread a `depth: usize` through both, error past a fixed ceiling (64 is
generous — llama.cpp uses similar). The GGUF one is the easier and more
important of the two.

### S2 (M) — the API key is compared in non-constant time

`crates/infr-server/src/lib.rs:1197-1208`

```rust
.is_some_and(|tok| tok == key)
```

`str::eq` short-circuits on the first differing byte. With `serve.api_key` set
and the server reachable over a network, this is a byte-at-a-time timing oracle
on the token. The rest of the auth path is careful (checked before routing,
empty-key-means-disabled is explicit and tested); this is the one gap.

**Fix:** compare with a constant-time primitive (`subtle::ConstantTimeEq`, or a
hand-rolled
`a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0, |acc, (x, y)| acc | (x ^ y)) == 0`).

### S3 (M) — no HTTP timeouts on any hub request

`crates/infr-hub/src/pull.rs:128-161`

Both `http_client()` and the ad-hoc client in `head_lfs_sha` are built with no
`.timeout(…)`, no `.connect_timeout(…)`, and no read timeout.
`reqwest::blocking`'s default is _no timeout_: a server that accepts the
connection and then stalls hangs `infr pull` (and `infr run`'s auto-pull)
forever, with no way out but Ctrl-C. `stream_into` has the same property
mid-body.

Related, smaller: `head_lfs_sha` constructs a whole new `Client` per call rather
than reusing a pooled one — for a 40-shard model that is 40 fresh TLS
handshakes.

**Fix:** a `connect_timeout` (~10 s) plus a generous per-read timeout on both
clients; hoist the redirect-disabled client into a `OnceLock` alongside
`http_client`.

### S4 (M) — HF-supplied filenames are joined into local paths without traversal checks

`crates/infr-hub/src/pull.rs:114`, `:268`, `crates/infr-hub/src/store.rs:41`

`filename` comes from the remote model API's `siblings[].rfilename` and reaches
`snap.join(filename)` unvalidated. `Path::join` with a component containing `..`
escapes the snapshot directory, so a malicious or compromised API response can
plant a symlink at an arbitrary path the user can write. The blob side is safe
(content-addressed hex, and `sanitise()` scrubs the temp stem), and `repo_dir`
neutralises `/` in the repo name via `replace('/', "--")` — this is the one
unguarded join.

Same class, lower impact: `repo` is interpolated raw into
`https://huggingface.co/{repo}/…` with no percent-encoding.

**Fix:** reject any `rfilename` whose components include `..`, `.`, an absolute
root, or a Windows prefix, before it is used as a path — a five-line
`fn safe_relative(p: &str) -> bool`. This pairs naturally with the C4 fix, which
has to touch the same code.

### S5 (M) — chat templates render with no fuel/output limit

`crates/infr-chat/src/template.rs:28-89`

The Jinja template comes out of the downloaded GGUF's `tokenizer.chat_template`
and is compiled and rendered with a plain `minijinja::Environment` — no
`set_fuel`, no output size cap. A model shipping
`{% for i in range(100000000) %}x{% endfor %}` hangs (or OOMs) the process on
the first prompt, and in `serve` it does so inside a `spawn_blocking` while
holding a slot permit.

Also here: `ENV_CACHE` (line 23) is a process-global `HashMap<String, _>` keyed
by the full template source with no eviction. The doc argues it is bounded by
"one per model", which holds for the GGUF path — but `render_template` is `pub`,
so any future caller passing a per-request template turns it into an unbounded
leak. Worth a doc-level `#[doc(hidden)]`-ish warning or an LRU cap.

**Fix:** `env.set_fuel(Some(…))` on the built environment; cap the rendered
length.

### S6 (L) — `/health` and `/v1/models` are not behind the API key

`crates/infr-server/src/lib.rs:645-651`, `:720-737`

Only `POST /v1/chat/completions` calls `authorize`. With `serve.api_key` set, an
unauthenticated caller can still enumerate hosted model ids and probe liveness.
Minor, but if the key is configured the intent is clearly "this is exposed";
`/v1/models` should follow the same gate (`/health` arguably should stay open
for load balancers).

### S7 (L) — the non-streaming path cannot be cancelled by a disconnecting client

`crates/infr-server/src/lib.rs:854-856`

Documented in the code (`let cancel = AtomicBool::new(false);` with a comment),
but worth recording as a finding rather than a note: a client that disconnects
mid-generation on the non-streaming path burns a `--parallel` slot until
`max_tokens`. Axum can surface this via `tokio::select!` on the response future
being dropped; the `ChatOutcome`/`cancel` plumbing already exists on the
streaming side.

There is also no rate limiting and no per-request wall-clock ceiling anywhere —
with a 128 k default `max_tokens_cap`, N clients can pin all slots for a very
long time.

### S8 (L) — companion downloads are unverified

`crates/infr-hub/src/pull.rs:259-283`

`generation_config.json` is fetched with `expected_sha = None` and written to
the cache with no integrity check and no size cap. The code says so explicitly
("download best-effort, unverified") and the blast radius is small (it seeds
sampling defaults), but the file _does_ influence generation. A `HEAD` for the
LFS sha (already implemented as `head_lfs_sha`) or a size cap would close it.

---

## 3. `unsafe` / soundness

The `unsafe` in this tree is better-argued than average. Three things are still
worth changing.

### U1 (M) — `as_vk_buf` is a raw reinterpret where a checked downcast is available

`crates/infr-vulkan/src/lib.rs:111-119`

```rust
unsafe fn as_vk_buf(b: &dyn Buffer) -> &VkBuffer {
    &*(b as *const dyn Buffer as *const () as *const VkBuffer)
}
```

The safety contract ("must only be called with buffers returned by
`VulkanBackend::alloc`") is enforced by convention only, and every call site
invokes it inside `unsafe { … }` with a one-line
`// Safety: every buffer from this backend is a VkBuffer.` The `Buffer` trait
**already has `fn as_any(&self) -> &dyn std::any::Any`** (line 512), so a
checked downcast is available at essentially zero cost:

```rust
fn as_vk_buf(b: &dyn Buffer) -> Result<&VkBuffer> {
    b.as_any().downcast_ref::<VkBuffer>().ok_or_else(|| be("buffer is not a VkBuffer".into()))
}
```

This matters more now than it did: `infr multi` and the MTP draft path can put
more than one backend's buffers in flight in one process, and a mis-routed
`&dyn Buffer` here is undiagnosable memory corruption rather than an error.

### U2 (L) — `Vec<MaybeUninit<T>> → Vec<T>` via `transmute`

`crates/infr-cpu/src/pool.rs:314`

```rust
unsafe { std::mem::transmute::<Vec<std::mem::MaybeUninit<T>>, Vec<T>>(out) }
```

`Vec` is not `#[repr(C)]`; transmuting between two `Vec` instantiations is not a
guarantee the language makes, even when the element types have identical layout.
It works on every current rustc, but the sanctioned spelling is:

```rust
let (ptr, len, cap) = (out.as_mut_ptr(), out.len(), out.capacity());
std::mem::forget(out);
unsafe { Vec::from_raw_parts(ptr as *mut T, len, cap) }
```

Secondary: if `f` panics for one index, `run` re-panics _after_ the other tasks
wrote their slots, and `out` is dropped as `Vec<MaybeUninit<T>>` — no UB, but
every initialized `T` leaks its destructor. Worth a sentence in the doc comment,
since the current one says only that the panic "propagates out of `run` before
we get here".

### U3 (L) — the SIMD `SAFETY` comments state a justification that does not hold

`crates/infr-cpu/src/kernels.rs:22`, and the same phrasing throughout

> `// SAFETY: avx512bw detected at runtime; pointer bounds checked by slice indexing.`

The feature-detection half is correct. The bounds half is not: the loads are

```rust
_mm256_loadu_si256(qs[half * 32..].as_ptr() as *const __m256i)
_mm512_loadu_si512(q8b[k * 64..].as_ptr() as *const __m512i)
```

A `RangeFrom` slice validates only that the **start** index is in bounds — it
says nothing about the 32/64 bytes the intrinsic then reads. The loads happen to
be in bounds because the block geometry makes them so (`qs` is 128 B and
`half <= 3`; `q8b` is 256 B and `k <= 3`), which is the _real_ argument. Rewrite
the comment to state the geometric invariant, since a future block-layout change
would silently invalidate the current wording.

---

## 4. DRY

### D1 — the token-type registration block is duplicated verbatim

`crates/infr-llama/src/tokenizer.rs:96-118` and `:220-242`

~23 lines identical between `build_tokenizer` and `build_spm_tokenizer` (walk
`tokenizer.ggml.token_type`, split into `AddedToken::from(s, true)` /
`(s, false)`, add specials then added). The merges-parsing closure
(`splitn(2, ' ')` → `(String, String)`) is also duplicated at `:48-58` and
`:150-157`.

Both are pure functions of `(&Metadata, &[MetaValue])`; extracting
`fn register_token_types(md: &Metadata, toks: &[MetaValue], tok: &mut Tokenizer)`
and `fn parse_merges(arr: &[MetaValue]) -> Vec<(String, String)>` removes ~40
lines and one opportunity for the two tokenizer families to drift.

### D2 — the infr-vs-llama metric cell is written twice

`crates/infr-cli/src/main.rs:2734-2746` and `:2759-2771`

The `is` / `ls` / `ratio` formatting-and-`rows.push` block appears twice inside
`cmd_compare_sweep`, once for the four standard metrics and once for `mtp128`,
with only the error message differing. It is the same seven-line shape:

```rust
let is = iv.as_ref().map(|v| format!("{v:.0}")).unwrap_or_else(|e| { eprintln!(…); "ERR".into() });
let ls = lv.map(|v| format!("{v:.0}")).unwrap_or_else(|| "NA".into());
let ratio = match (iv.as_ref().ok(), lv) { (Some(&i), Some(l)) if l > 0.0 => { rows.push(…); … } _ => "-".into() };
println!("{short:<22} {metric_label:<10} | {is:>9} | {ls:>9} | {ratio:>10}");
```

Extract `fn print_cell(short, label, iv, lv, rows)`.

### D3 — two hand-rolled OpenAI error-envelope builders

`crates/infr-server/src/lib.rs:1106-1109`, `:1135-1138`, `:1142-1150`

`sse_error_event`, `json_error` and `param_error` each build
`{"error": {"message": …, "type": …}}` inline. One
`fn error_body(msg: &str, kind: &'static str, param: Option<&str>) -> serde_json::Value`
would give all three a single shape (and catch the fact that `param_error`
includes `"code": null` while the other two do not).

### D4 — the content-addressed short-circuit is written twice in `download_to_blob`

`crates/infr-hub/src/pull.rs:308-314` and `:329-335`

Deliberate (once before taking the `flock`, once after), and the comment
explains why — but the second copy silently dropped the `debug!` line the first
one has, so the two are already drifting. A local closure
`let hit = |sha: &str| { … };` keeps the intent and the logging in one place.

### D5 — `parse_hermes_tool_calls` and `parse_tool_calls` diverge on cleanup

`crates/infr-chat/src/tools.rs:244-292` vs `:306-326`

The pipe-marker parser strips a dangling opener through end-of-text and runs
`strip_markers` on the result (tested at `:810`). The Hermes parser does
neither: an unterminated `<tool_call>` and an unparseable body both leave raw
markup in `clean`, which is what the user sees. Same job, two behaviours. Worth
unifying on the stricter (pipe-marker) policy.

---

## 5. YAGNI / dead code

### Y1 — three `*_from` string-grammar wrappers are dead outside their own tests

- `crates/infr-core/src/budget.rs:133` — `mib_from`
- `crates/infr-core/src/budget.rs:154` — `reserve_from`
- `crates/infr-core/src/pager.rs:286` — `ring_bytes_from`

Each is a `&str` façade over the `Config`-field function next to it, kept "so
the grammar has one home". Grepping the tree, the only non-test references are
their own definitions and one doc-comment mention (`config/env.rs:73` names
`budget::mib_from` in prose). The config campaign moved the parsing into
`config::env` / `manifest.rs`, so these are now three public functions whose
sole purpose is to let their own tests avoid touching the environment — which
the value-taking functions already allow.

**Fix:** delete all three; rewrite `reserve_from_matches_the_inline_formula`,
`mib_grammar` and `ring_bytes_clamp_boundaries_and_override_grammar` against
`reserve_bytes`/`mib_bytes`/`ring_bytes` (the string→value step is already
covered by `config::env`'s own tests).

### Y2 — `normalize_messages` is `messages.to_vec()`

`crates/infr-chat/src/tools.rs:503-505`

Re-exported from both `infr-chat` and `infr-engine`, called from nowhere. Its
doc comment is candid: _"Currently a straight clone … Kept as a named seam so
any future normalisation … has one place to land."_ That is the definition of a
speculative seam — the actual normalisation that exists (`flatten_content`)
lives in `infr-server`, i.e. the future landing site already turned out to be
somewhere else. Delete it and the two re-exports.

### Y3 — `Backend::alloc_uninit`'s release-mode poison knob

`crates/infr-vulkan/src/lib.rs:3242-3255`

`debug.poison_uninit` / `INFR_POISON_UNINIT` forces a `0xFF` fill in release
builds. Debug builds already poison unconditionally. This is a debugging knob
for a class of bug (layout-sensitive read-before-write) that the `alloc` calloc
contract exists to prevent; it costs a config field, an env key, a manifest
entry, and a branch in the allocation hot path. Worth asking whether it has
earned its keep since the buffer-zero-init work landed — if it is still catching
bugs, keep it; if not, it is one fewer knob.

### Y4 — `SpillTally::admits` cap is described as diagnostic-only

`crates/infr-core/src/budget.rs:213-224`

The `cap` parameter exists to force whole-class host placement "so the
whole-host path stays reproducible" — a measurement affordance, not a product
behaviour, exposed as `INFR_KV_OVERFLOW_VRAM_MB`. Same question as Y3. (If it
stays: `self.vram_bytes.load() + bytes` can overflow `u64` and should be
`saturating_add`, and the load-then-record pair is a check-then-act race under
concurrent allocation.)

---

## 6. Minor / nits

| #   | Location                                    | Note                                                                                                                                                                                                                                                                                                                                                                                                                  |
| --- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| N1  | `infr-core/src/budget.rs:127`               | `mib_bytes` does `mb * 1024 * 1024` unchecked — a large-but-parseable `INFR_KV_OVERFLOW_VRAM_MB` overflows `u64` (panic in debug, wrap in release). Use `saturating_mul`.                                                                                                                                                                                                                                             |
| N2  | `infr-gguf/src/lib.rs:443-448`              | `Gguf::resolve` does a linear `find` over `tensors` on every lookup. At ~1–2 k tensors × ~1–2 k lookups this is a few million string compares at load. A `HashMap<String, usize>` built once in `open` removes it. Also: duplicate tensor names are silently accepted, first-wins.                                                                                                                                    |
| N3  | `infr-gguf/src/lib.rs:390`                  | `r.pos.div_ceil(alignment) * alignment` can overflow for a pathological (but power-of-two, so accepted) `general.alignment`. Cap the alignment at something sane, e.g. 4 MiB.                                                                                                                                                                                                                                         |
| N4  | `infr-gguf/src/lib.rs:375-387`              | `n_dims` is unbounded; GGUF caps it at 4. A bogus value just makes the parse fail on EOF, but rejecting it names the actual problem.                                                                                                                                                                                                                                                                                  |
| N5  | `infr-vulkan/src/recorder.rs:1014`, `:1024` | `Box::leak` runs **per op per record**, not per distinct shape — the comment says "one small string per op per record", which is accurate but means a long `INFR_PROF_OP_SHAPES` run leaks unboundedly even with a fixed shape set. Intern into a `RefCell<HashSet<&'static str>>`.                                                                                                                                   |
| N6  | `infr-cli/src/main.rs:791-795`              | `std::env::set_var("RAYON_NUM_THREADS", …)`. Correctly placed (in `main`, before any pool spins up) and correctly justified, but `set_var` becomes `unsafe` in edition 2024 — worth a `// TODO(edition-2024)` so the migration does not surprise anyone.                                                                                                                                                              |
| N7  | `infr-vulkan/src/lib.rs:2385-2445`          | `check_vram_budget` / `vram_budget_fits` are check-then-act with no lock: two threads allocating concurrently can both pass a budget only one fits in. Fine today (weight load is serial); worth a note now that `serve --parallel` exists.                                                                                                                                                                           |
| N8  | `infr-chat/src/template.rs:79-89`           | `cached_env` misses → both threads build → second `insert` overwrites. Harmless, but `entry().or_insert_with()` under one lock is shorter and does the right thing.                                                                                                                                                                                                                                                   |
| N9  | `infr-llama/src/sampling.rs:488-511`        | The `top_k == 0` path builds a `BinaryHeap` over the **whole** vocab every token (~150 k × 8 B ≈ 1.2 MB alloc + O(n) heapify per token) to then pop only the nucleus. `select_nth_unstable_by` against a nucleus-sized bound, or a bounded min-heap, avoids the per-token allocation. The comment claims it avoids "the full-vocab sort", which is true, but it does not avoid the full-vocab pass or the allocation. |
| N10 | `infr-core/src/config/env.rs:255`           | `v.dn_chunk_scan = presence_inv(get, "INFR_DN_CHUNK_SCAN")` — a positively-spelled key that _disables_ the feature. Documented ("Spelled positively, read with `.is_err()`"), and R1-frozen, but it is the one env key in the file whose name means the opposite of what it does. Worth a rename in the next breaking sweep.                                                                                          |
| N11 | `infr-hub/src/pull.rs:552-562`              | `sanitise` maps `/` → `_`, so `a/b.gguf` and `a_b.gguf` collide on the same `.dl-` temp and `.lock`. The `flock` serialises them, so no corruption — but a resume of one would splice onto the other's partial, caught only by the final sha check (and not at all for a non-LFS file). Hash the full name into the stem.                                                                                             |
| N12 | `infr-server/src/lib.rs:1235-1252`          | `flatten_content`'s `Some(other) => other.to_string()` JSON-serialises a numeric/boolean `content` into the prompt (e.g. `content: 42` becomes the literal `42`, `content: {"a":1}` becomes `{"a":1}`). Probably right; worth a test pinning the intent either way.                                                                                                                                                   |
| N13 | `infr-chat/src/tools.rs:398-411`            | `parse_any_tool_calls`'s bare-JSON arm returns `String::new()` for `clean`, discarding any text — correct for Llama-3.x (whole body is the call) but the two other arms preserve surrounding text. Asymmetry worth a comment.                                                                                                                                                                                         |

---

## Suggested order of work

1. **C1** (seed collapse) — one line, user-visible, has a test that needs
   widening.
2. **C2** (`download`/`copy_buffer` bounds) — one guard each, closes a UB path.
3. **S1** (recursion depth limits) — the GGUF half first; the tool-call half is
   remotely reachable in `serve`.
4. **C4 + S4** together (hub subdirectory support + path-traversal guard) — same
   code, and C4 is a real "this model won't download" bug against common repos.
5. **S2 + S3** (constant-time key compare, HTTP timeouts) — small and
   independent.
6. **C3** (GGUF block alignment), **C5** (`Pager` guards), **U1** (checked
   downcast).
7. **S5** (template fuel), then the DRY/YAGNI cleanups (**Y1**, **Y2**, **D1**,
   **D2**) as a single tidy-up commit.
