# Qwen3.8 MTP and ordinary decode at 20K context

## Scope

Measured the current uncommitted `perf/qwen38-24g48g-ub3000` worktree on the
RX 7900 XTX with the 4.27 bpw M64 model and its Q4_K_M MTP sidecar. Every row
used a fresh process, aggressive automatic resource policy, Q8_0 K/V, a 32K
context capacity, greedy sampling, no thinking, and about 20K actual prompt
tokens. MTP used four VERIFY rows. Pager profiling supplied cycle/token
timestamps; the reported middle window is 25%-62.5% of generated tokens and
the late window is 62.5%-100%.

Both paths finalized `ubatch=4096` and four actual prefill lanes. MTP had a
14.22 GiB post-fixed arena and a 3.39 GiB planned expert cache; ordinary decode
had 16.32 GiB and 5.23 GiB. Both streamed all 48 expert layers during prefill,
with the same 4.44 GiB ring shape.

## Results

| Path and output | Prompt | Prefill | Decode, whole | Decode, middle | Decode, late | MTP alpha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP, count 0-200 | 19,988 | 936 tok/s | 60.2 tok/s | 61.4 tok/s | 61.3 tok/s | 0.994 |
| MTP, ordinary answer | 20,019 | 937 tok/s | 39.9 tok/s | 42.0 tok/s | 39.7 tok/s | 0.594 |
| Ordinary, count 0-200 | 19,988 | 1,035 tok/s | 37.7 tok/s | 38.6 tok/s | 37.8 tok/s | n/a |
| Ordinary, ordinary answer | 20,019 | 1,034 tok/s | 35.0 tok/s | 35.1 tok/s | 36.8 tok/s | n/a |

The count output is byte-identical between MTP and ordinary decode. At this
depth MTP is 62% faster on the high-acceptance count task and 14% faster on the
ordinary question. The ordinary-question result remains acceptance-sensitive:
its late MTP speed is only 8% above ordinary late decode.

## Prefill gap

The controlled 20K result does not reproduce a twofold prefill gap. MTP is
about 9.5% slower: 21.35-21.36 seconds versus 19.31-19.36 seconds.

For the count prompt, the five MTP target-prime chunks consume 20.818 seconds.
The detached-head catch-up and outer-loop work consume the remaining 0.534
seconds. Relative to ordinary prefill, roughly 1.509 seconds of the 2.043-second
gap is already inside the target's special prime path; only 0.534 seconds is
outside it. The ordinary-question split is effectively identical.

The special path differs from ordinary prefill in several concrete ways:

- It taps and downloads every four-stream hidden row for the detached head.
  At this prompt length and a 10,240-float row, that is about 781 MiB of hidden
  data, then another host-to-device feed into the head.
- It executes through the VERIFY-frontier driver in five separate calls and
  projects one frontier row per chunk. Ordinary prefill uses the headless
  batched-prefill path.
- Its prime timer also surrounds target binding/finalization and head catch-up,
  while ordinary prefill's steady path has less per-chunk wrapper work.

Measured target readback itself totals only about 208 ms. It is real work but
does not explain the full 1.509-second target-path difference; execution and
first-chunk setup are also slower. The detached head is therefore not the main
20K bottleneck, and waiting for experts is not a sufficient explanation either:
both paths retain four lanes and the same streamed-layer ring.

## Why the long-context gap grows

The earlier 200K MTP run used a 262,144-token fixed runtime. MTP fixed resources
reduced placement room enough to lower the requested 4,096-row chunk to 3,072,
and the prefill pager reported one actual lane instead of four. It then measured
455 tok/s. At 32K capacity the same branch keeps 4,096 rows and four lanes and
reaches 936-937 tok/s.

That is the main amplification mechanism behind the near-twofold observation:
MTP sidecar/runtime residency, full-window state, and the larger long-context
activation requirement push placement across both a ubatch rung and a lane
count boundary. The special prime path adds another approximately 10% at 20K,
but detached-head catch-up alone is too small to create the large gap. Deeper
attention/QSA work and reduced expert/runtime room then compound the placement
change at 200K.

## Optimization order

1. Drive MTP prompt prime through the ordinary batched-prefill scheduler, adding
   only a hidden-state tap and frontier result. This targets the measured
   1.5-second target-path excess without changing decode semantics.
2. Keep chunk hidden rows on the GPU and feed the detached head directly, or
   use a bounded device staging ring. This removes the approximately 781 MiB
   device-to-host-to-device round trip and the shifted host buffer.
3. Make the prefill planner account for MTP fixed residency while preserving a
   four-lane 4K chunk whenever the actual post-fixed arena permits it. The 200K
   result shows that crossing this placement boundary dominates small kernel
   savings.
4. Re-run 20K, 100K and 200K with placement, target-prime, head-catch-up and
   decode-cycle timings. A successful change should retain output identity,
   keep 4K/four-lane prefill deeper into the context range, and close the 20K
   prefill gap before claiming a long-context improvement.

