# Long-context resource matrix

This Windows-only harness reproduces constrained machines on a larger host and drives real
multi-turn inference past three dynamic-KV boundaries. It is a correctness and capacity test, not
a throughput benchmark.

## Matrix

The API matrix contains eight cases:

| Synthetic machine | Configuration | Models |
|---|---|---|
| 16 GiB VRAM, 2 GiB occupied; 32 GiB RAM, 10 GiB occupied | automatic and manual (14 GiB VRAM / 22 GiB process RAM) | Qwen 35B and Qwen3.8 Q4 |
| 24 GiB VRAM, 2 GiB occupied; 64 GiB RAM, 10 GiB occupied | automatic and manual (22 GiB VRAM / 54 GiB process RAM) | Qwen 35B and Qwen3.8 Q4 |

One additional Qwen 35B CLI run performs three ordinary short chat turns under the 16/32 automatic
profile. API cases use one server slot, Q8 K/V, deterministic sampling, and three requests:

1. A prompt at 32K followed by 32 decode tokens. Decode must grow KV to 64K.
2. A prompt at 64K + 256. Incremental prefill must grow KV to 96K.
3. A prompt at 96K followed by 32 decode tokens. Decode must grow KV to 128K.

The final context is therefore above 96K. Requests carry the complete prior conversation and the
server must report a large `cached_tokens` prefix on turns two and three.

## Resource override

`--test-resource-profile PATH` is a hidden test-only CLI option. The JSON file supplies
`vram_total`, `vram_used`, `ram_total`, and `ram_used`.

- Host-memory probes are capped before automatic RAM policy runs.
- Vulkan planning sees the capped total and available VRAM.
- The Vulkan allocation guard subtracts backend allocations from the synthetic room, including on
  drivers without `VK_EXT_memory_budget`.
- A profile only reduces real capacity. It cannot invent RAM or VRAM.
- Startup always emits `TEST RESOURCE OVERRIDE ACTIVE`.

The external monitor samples process working set/private bytes, Windows per-process dedicated and
shared GPU memory, system available RAM, and page faults. It terminates only the verified test
process if working set or dedicated GPU memory exceeds the profile's available capacity.

## Setup and use

Copy `tests/context-resource/matrix.example.json` to
`tests/context-resource/matrix.local.json` and set the two local GGUF paths. The local manifest and
all result artifacts are gitignored.

```powershell
# Expand the nine cases without loading a model.
powershell.exe -NoLogo -NoProfile -File scripts/context-resource-matrix.ps1 -List

# Run all unfinished cases.
powershell.exe -NoLogo -NoProfile -File scripts/context-resource-matrix.ps1

# Retry one case, overwriting its previous files.
powershell.exe -NoLogo -NoProfile -File scripts/context-resource-matrix.ps1 `
  -CaseId 16vram-32ram-auto-qwen35 -Force
```

Successful cases are skipped on later invocations. Each case keeps requests, responses, server
logs, exact KV-growth events, 500 ms resource samples, and a result JSON under
`artifacts/context-resource-matrix/`. The aggregate table is `report.md` in that directory.

The prompt planner is an internal helper:

```text
infr __test-plan-prompt MODEL --messages template.json --target 32768 --output planned.json
```

It opens GGUF metadata and the model tokenizer only; it never constructs a GPU backend. The helper
renders the same chat template as `serve`, replaces exactly one `{{INFR_FILLER}}` marker, and finds
the closest prompt depth at or below the target.
