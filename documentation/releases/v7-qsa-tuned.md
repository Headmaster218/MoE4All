---
kind: release
status: stable
release: v7-qsa-tuned
engine_commit: 8e7206ee
run_date: 2026-10-19
evidence: measured
---

# v7-qsa-tuned Stable Release

## Release Position

`v7-qsa-tuned` is the final stable point for the Strata-alignment campaign. It keeps the
256K long-context engine and refuses the MTP two-window Phase 3 direction.

- Engine commit: `8e7206ee` (`perf: widen QSA decode kernels`)
- Previous rollback point: `v6-state-resident` / `89eab002`
- Model: Qwen3.8-Flash-Next-GSQ-RCO-abliterated IQ3_XXS
- Hardware: 2 x Radeon RX 7900 XTX, 62 GiB host RAM
- Placement: 26/22 layers, recurrent SSM/PLE state device-local, Q8 K/V
- Maximum context: 262144 tokens

## What Changed

- QSA Q8 gather processes four codes per thread and reuses one scale load per packed value.
- Single-row QSA score defaults to a 32-block / 1024-thread workgroup, reducing its dispatch
  grid by 4x.
- Both changes preserve the established QSA math. The complete test output is byte-identical
  to the previous decode-score tile.

## Measured Result

| Workload | Prefill | Decode |
| --- | ---: | ---: |
| 31K prompt, one request | 721.5 tok/s | **23.12 tok/s** |
| 31K prompt, two concurrent lanes | 401.9 tok/s per lane | **41.17 tok/s aggregate** |

The two-lane run used 262144-token slots, `-u256`, and completed both requests without failure or
cross-slot state contamination.

## Launch

Set the model path and a host DRAM budget suitable for the machine:

```bash
MODEL=/media/qiankun/新加卷1/models/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf
CFG=/media/qiankun/新加卷1/amdvlk/radv-test.toml
```

Common environment:

```bash
export INFR_AUTO_PROFILE=aggressive
export INFR_KV_OVERFLOW=1
export INFR_KV_OVERFLOW_VRAM_MB=0
export INFR_PIPELINE=Vulkan0,Vulkan2
export INFR_PIPELINE_OVERLAP=1
export INFR_PIPELINE_SPLIT=26,22
export INFR_PIPELINE_LAYER_RESERVE_MB=256
export INFR_VRAM_BUDGET=24g
export INFR_UBATCH_PARALLEL=256
```

Lowest-latency single request:

```bash
target/release/infr serve "$MODEL" \
  --config "$CFG" \
  --addr 127.0.0.1:8099 \
  --ctx 262144 \
  --parallel 1 \
  -u 256 \
  --set paging.dram=40GiB \
  --set paging.moe_layer_stream=false \
  --set kernels.vulkan.qsa_union=true
```

Two independent 256K slots:

```bash
target/release/infr serve "$MODEL" \
  --config "$CFG" \
  --addr 127.0.0.1:8099 \
  --ctx 262144 \
  --parallel 2 \
  -u 256 \
  --set paging.dram=40GiB \
  --set paging.moe_layer_stream=false \
  --set kernels.vulkan.qsa_union=true
```

The server is ready at `http://127.0.0.1:8099`. Send the two concurrent requests to the OpenAI
endpoint; concurrent arrival is what activates both lanes:

```bash
curl -s http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002","messages":[{"role":"user","content":"Explain cache pressure."}],"max_tokens":128,"stream":true}' \
  > lane-1.sse &

curl -s http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002","messages":[{"role":"user","content":"Explain memory bandwidth."}],"max_tokens":128,"stream":true}' \
  > lane-2.sse &

wait
```

Use sampling flags appropriate to the application. The release measurements used
`--temp 0 --seed 1` and fixed prompts only to make A/B results reproducible.

## Validation

- `cargo test -p infr-vulkan --test qsa_parity -- --nocapture`: 3 passed.
- 32K single and two-lane runs completed with clean KV state and valid streaming/timing output.
- Worktree was clean at the engine commit; release binary SHA-256 at validation time:
  `54a052dff4a96d66dd5cc7991852dcbd3e6cb28d65b806d9219e600a213e1f0c`.

## Explicitly Not Included

- MTP two-window Phase 3 is abandoned.
- F16 QSA block keys are rejected: they raised prefill to 921 tok/s but reduced decode to
  19.20 tok/s and changed complete output.
- Four-lane 32K production serving is not supported on this model/VRAM point. It needs about
  25.2 MiB more MoE activation scratch than the 24 GiB cards can safely provide.

The remaining 329 ms QSA-score cost over 31 decoded tokens is a future system-level item:
host-tier F32 block-key placement or lossless compression. It is not part of this release.
