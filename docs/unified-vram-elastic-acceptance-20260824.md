# Elastic unified VRAM acceptance — 2026-08-24

## Result

Paged MoE Experts, LLM activation scratch, and native Embedding weights/runtime now use the same
physical Vulkan arena. Fixed dense weights and KV/persistent state remain outside it. Expert slots
grow from low addresses; variable-size LLM, Embedding, Vision, and Draft allocations grow from high
addresses. When a variable allocation cannot fit, the pager evicts the coldest contiguous Expert
window and later restores every released slot in place.

The runtime reserve is no longer a second physical reservation. It is included in the arena and is
available to Expert slots whenever the corresponding runtime allocation is absent. Native
Embedding weights are loaded from their GGUF only for an active request and released after the GPU
execution and output download. Embedding activation allocations are transient; cached execution
plans retain only small host-visible input/readback buffers.

Post-load automatic context sizing keeps two independent budgets: KV/recurrent state must fit the
device's still-uncommitted room, while activation scratch may use the already-committed elastic
arena. This prevents a full Expert cache from making automatic context sizing report zero without
incorrectly allowing persistent KV to consume Expert slots.

Vision and Draft allocation classes follow the same high-address policy and are ready for their
engines to use. Dynamic KV allocation remains intentionally out of scope.

## Real-GPU acceptance

Hardware: AMD Radeon RX 7900 XTX. Configuration: 20 GiB total VRAM budget, 40 GiB RAM budget,
4096-token context.

Models:

- `Qwen3.6-35B-A3B-APEX-I-Balanced.gguf`
- `nomic-embed-text-v1.5.f16.gguf`

Initial arena accounting after both endpoints became ready:

| Category | Bytes | MiB |
|---|---:|---:|
| Elastic arena | 18,790,293,504 | 17,919.82 |
| Expert slots | 18,789,572,608 | 17,919.13 |
| Embedding weights | 0 | 0 |
| Embedding runtime | 0 | 0 |
| Free/slot-rounding tail | 720,896 | 0.69 |

The initial unusable tail is 0.00384% of the arena.

One two-row Embedding request temporarily admitted 273,530,880 bytes (260.86 MiB) of weights and a
1,572,864-byte runtime allocation. At the post-execution sample, transient runtime had already been
released; all Embedding classes returned to zero immediately after the request. Expert-slot
rounding/fragmentation left 22,568,960 bytes free while the weights were resident, 0.120% of the
arena. The following Chat request restored 351 loaned slots. After its final small LLM runtime
allocation, the complete arena was accounted as:

| Category | Bytes |
|---|---:|
| Expert slots | 18,789,433,344 |
| LLM runtime | 340,224 |
| Free tail | 519,936 |
| Total | 18,790,293,504 |

The final free tail is 0.00277% of the arena. Embedding weights and runtime were both zero.

## Functional acceptance

- `/v1/embeddings`: a 768-dimensional vector returned in 2.88 s including demand-loading the
  260.86 MiB model. Repeating the identical request after eviction reused the compiled plan,
  reloaded the weights, returned in 0.13 s, and matched bit-for-bit (`max_abs = 0`).
- `/v1/chat/completions` immediately afterward: returned `PASS`, 17 prompt tokens and 2 completion
  tokens in 1.10 s.
- A separate simultaneous Chat + Embedding pair also completed correctly (`SAFE`, 768-dimensional
  vectors), proving that the shared execution gate serializes conflicting arena use without a
  deadlock or stale Expert access.
- No error, panic, device-lost, or out-of-memory event appeared in the service log.

Raw final service log:
`target/perf/unified-vram-20260824-033627.stderr.log`. The final release-build smoke test after the
split-budget context fix is `target/perf/unified-vram-20260824-040332.stderr.log`; it returned a
768-dimensional normalized Embedding and then two correct Qwen Chat responses (`OK.` / `OK`). Its
final free arena tail was 519,936 bytes (0.00277%).

## Build and tests

- `cargo check -p infr-vulkan -p infr-embedding -p infr-llama -p infr-cli`
- `infr-vulkan` unified allocator tests: 7/7 passed.
- `infr-embedding` tests: 7/7 passed.
- `infr-llama` memory-plan budget test: passed.
- `infr-llama` post-load automatic-context split-budget tests: 2/2 passed.
- Native server `cargo build --release -p infr-cli`: passed with the real Vulkan SDK.
