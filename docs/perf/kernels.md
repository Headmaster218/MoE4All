# kernels.md — cross-backend fast-kernel coverage

Single source of truth for **which weight quant formats have a native fast
linear kernel on each backend**. "Native" = a dedicated in-shader / int8-dot
decode path, as opposed to the `dequant → f16/f32` fallback (decode the whole
block to float, then run a generic float GEMV/GEMM). The fallback is correct but
streams more bytes and wastes the format's compression on the hot path.

`DType` is enumerated canonically in `crates/infr-vulkan/src/linear.rs`
(`ALL_DTYPES`, drift-guarded by an exhaustive match). Coverage tests: CPU in
`infr-cpu` (SIMD↔scalar bit-identity + tolerance-parity to `dequant_block`),
Vulkan op-parity suites, Metal `crates/infr-metal/tests/parity.rs` (numerically
validated on real Metal hardware in CI's `test-macos` job).

## Coverage — all 24 weight quant formats are native on all three backends

| Format    | CPU | Vulkan | Metal | Kind                       |
| --------- | :-: | :----: | :---: | -------------------------- |
| `Q4_0`    | ✅  |   ✅   |  ✅   | affine 4-bit               |
| `Q4_1`    | ✅  |   ✅   |  ✅   | affine 4-bit + min         |
| `Q5_0`    | ✅  |   ✅   |  ✅   | affine 5-bit               |
| `Q5_1`    | ✅  |   ✅   |  ✅   | affine 5-bit + min         |
| `Q8_0`    | ✅  |   ✅   |  ✅   | 8-bit, 32-block            |
| `Q2_K`    | ✅  |   ✅   |  ✅   | k-quant 2-bit              |
| `Q3_K`    | ✅  |   ✅   |  ✅   | k-quant 3-bit              |
| `Q4_K`    | ✅  |   ✅   |  ✅   | k-quant 4-bit              |
| `Q5_K`    | ✅  |   ✅   |  ✅   | k-quant 5-bit              |
| `Q6_K`    | ✅  |   ✅   |  ✅   | k-quant 6-bit              |
| `IQ4_NL`  | ✅  |   ✅   |  ✅   | non-linear codebook, flat  |
| `IQ4_XS`  | ✅  |   ✅   |  ✅   | non-linear, super-block    |
| `IQ2_XXS` | ✅  |   ✅   |  ✅   | grid, KSIGNS lookup        |
| `IQ2_XS`  | ✅  |   ✅   |  ✅   | grid, 9-bit index          |
| `IQ2_S`   | ✅  |   ✅   |  ✅   | grid-codebook              |
| `IQ3_XXS` | ✅  |   ✅   |  ✅   | grid                       |
| `IQ3_S`   | ✅  |   ✅   |  ✅   | grid, per-32 scale         |
| `IQ1_S`   | ✅  |   ✅   |  ✅   | 1-bit grid + delta         |
| `IQ1_M`   | ✅  |   ✅   |  ✅   | 1-bit, d-in-scales         |
| `TQ1_0`   | ✅  |   ✅   |  ✅   | ternary, base-3 packed     |
| `TQ2_0`   | ✅  |   ✅   |  ✅   | ternary, 2-bit             |
| `Q2_0`    | ✅  |   ✅   |  ✅   | Bonsai ternary, 64-block   |
| `MXFP4`   | ✅  |   ✅   |  ✅   | fp4 + E8M0 scale (gpt-oss) |
| `NVFP4`   | ✅  |   ✅   |  ✅   | fp4 + per-16 UE4M3 scale   |

**24/24 native on every backend. No weight quant falls back to dequant→float on
any backend.** Floats (`F16` / `F32` / `Bf16`) are native everywhere too.

## Not weight-linear kernels (correctly excluded)

- **`I2S`** (BitNet `i2_s`) — host-converted to `f16` in the runner's `wload`,
  so it never reaches a backend as `I2S`; no native kernel by design (all Vulkan
  `*_kernel_name`/spv gates return `None` for it).
- **`Turbo2` / `Turbo3` / `Turbo4`** — KV-cache quantization formats
  (TurboQuant), not weight formats; they do not participate in linear kernels.

## Per-backend decode strategy

- **CPU** (`infr-cpu`) — int8-quantized-activation dot: quantize the activation
  row to int8 once, then a per-block integer dot against the native weight codes
  (scalar → AVX2 → AVX-512BW → AVX-512-VNNI `dpbusd`), with up to 8-row
  cache-blocking tiles. Grid/codebook formats expand the grid row to signed i8
  once, then reuse the per-sub-block scale × int-dot. Ternary folds `(digit−1)`
  into signed i8 + a single-scale int dot. See `cpu.md` for the per-format
  landing history and measured speedups.
- **Vulkan** (`infr-vulkan`) — two families: `dqblk`-decode f16 coopmat GEMM for
  prefill (wide `m`), and dp4a `mmq` integer GEMV for decode (`m=1`). The int8
  `mmq` path (each thread owns its accumulator → scale-after for free) is the
  principled integer route; see `playbook.md` for why fp8/int8/bf16 _coopmat_
  operand swaps were measured and rejected in favour of the f16 coopmat GEMM.
- **Metal** (`infr-metal`) — `DEC16_<DT>` decode macros bake 16 consecutive
  weight elements into `wk[16]` per 16-element block index, instantiated across
  the GEMV / row-tile / coopmat-GEMM kernel families. Byte-wise decode
  (alignment-safe for odd block strides) rather than packed `ushort` loads. See
  `docs/metal.md`.

## Relationship to the perf docs

This doc tracks **coverage** (does a native kernel exist). For **throughput**
work — ratios vs llama.cpp, the optimization playbook, the bottleneck taxonomy —
see `playbook.md` (GPU / general) and `cpu.md` (CPU backend).
