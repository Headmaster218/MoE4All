# Five-model kernel campaign

Status: completed at the agreed reduced validation scope. Selected five-model matrix, Unit13 defaults/targeted regressions and all13 independent cumulative prefix typechecks PASS. Qualified source changes are committed locally; O01 six-tuple/default-true and PLE=true are integrated, Q01 current gate retained. Historical pending statements below are superseded by the final closure. No push, release, production overwrite or further E0 expansion.
Base: `0bbc19b7911f98293b10c949bb66752c036a3e26` on
`release/1.0.0-maintenance`. This is an execution ledger, not a claim that every
catalogued upstream project has already been ported or benchmarked.

Current [gate/default disposition](#final-gate-and-default-disposition) supersedes
earlier nine-tuple, default-false and pending-BAAB statements below. Those dated
checkpoints remain historical evidence, not the current O01 eligibility set.

## Final Closure

The human reduced the remaining validation to necessary tests, expanding only on
anomalies. Four models completed full6K/30K/150K x1/x2, three Decode repetitions;
Ornith Q3 completed only6K x1/x2, one repetition. Do not infer its unmeasured long
contexts. Common settings are Windows/RX7900XTX, automatic aggressive with an
explicit768MiB VRAM reserve, Q8 K/V, MTP/vision/embedding off, two configured slots,
512 discarded warmup and1024 measured Decode tokens per lane.

[Complete selected results](D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/final-summary.md)
retain section/overall/lane rates, all repetition ranges and frozen Wave6/Wave12
identities. Timed Wave12 predates the final conservative O04/O06 observer guards
and O01 exclusion. Unit13 verifies those final selection changes, without another
long model matrix or a claim that its test binary is the timed native executable.

- Unit13: Core255, Vulkan275 (147 ignored), llama235 default tests pass.
- Necessary GPU regressions: O04 six plain/MRoPE observer cases; O06 six rewrite
  cases times two seeds; O01 six selected, three excluded and nine neighboring
  fallback shapes all pass exact output/guard/selection checks.
- All13 cumulative prefixes pass actual offline/locked release Cargo checks,
  including test targets and CLI/core/llama/Vulkan integration. These checks do
  not rerun benchmarks. Run: `r4-qualified-20261023-0712`, input hash
  `39b43b546dc81e29771baf86a059593cd59fc78deaef92dc2baa6562838124ff`.
- Committed code matches all44 normalized code files of the qualified Unit13
  snapshot. Negative prototypes and architectural deferrals remain recorded,
  rather than being enabled or counted as gains.

Fifteen source commits after `c973c209b` are local only:

| Item | Commit |
| --- | --- |
| D01 Q8 pair | `9de88d647` |
| R02 immutable scratch metadata | `ff3355f77` |
| G01 exact recurrent norm | `2e0ce24e2` |
| A01 segmented Q8 attention | `ca000b536` |
| A02 PV fragment reuse | `096dff4a3` |
| K02 packed k-quant MMQ | `b703b766d` |
| Q01 exact ordered Top-K | `dd99dee6d` |
| E01 SwiGLU/Q8 preparation | `1f05faa4a` |
| O06 HC up/mix and observer proof | `9eabf935e` |
| PLE dual-output fusion | `865e6ce7f` |
| O04 segmented K-write and observer proof | `307c03b2a` |
| O01 six qualified BM128 tuples | `69be8bbdb` |
| M01 local MTP buffers | `2284e0e3b` |
| CPU inline operand scan | `801f765a6` |
| K04 integrated-device fallback | `8033879fa` |

The earlier retained K04 commit `811b1bafa` and dual-output hazard fix
`c973c209b` remain in history. This ledger is committed separately. No production
executable or release package is replaced; no report is uploaded. The preexisting
user worklog remains outside these commits.

Evidence root: `D:\AIinfr\infr\local-work\evidence\infr\20261023-034328-kernel-campaign-c61e`.
Results root: `D:\AIinfr\infr\local-work\results\infr\20261023-034328-kernel-campaign-c61e`.
Shared build bucket: `local-work/build-cache/rust/rust-1.97.1-x86_64-pc-windows-msvc-static-slot2`.
Use portable x86-64 plus static MSVC CRT; never use target-cpu=native.

## Acceptance

- Preserve mathematical/state contracts and independent fallback paths.
- Require reproducible unit and real-module benefits; end-to-end may be neutral,
  but must not regress at its tested device/dtype/shape gate.
- Distinguish isolated device time, full module wall time and complete-cycle
  throughput. Keep actual regressions and numeric failures separate from masked wins.
- GPU runs and shared builds are serialized by the parent. Workers only edit
  disjoint shader/test files. Do not stop unrelated services or overwrite production binaries.
- Preserve preexisting worklog changes; commits stage only accepted, measured changes.
- Five inputs: 35B APEX Q5_K/Q6_K, Ornith122B Q2_K and Q3_K_S,
  FlashNext AD4.27 IQ2_S/IQ3_S/IQ4_NL and UDQ2 IQ2_XS/IQ3_XXS/IQ4_NL.

## Queue

"Retained" below means existing code or a narrowly qualified unit/module candidate
is kept for the parent's integration/non-regression gate, not that all five models
or a release default are accepted. Current checkout defaults are not deployment
approval. "Rejected" is scoped to the measured implementation/shape/route; larger
scheduler, ownership and publication changes are separately architecture-deferred.
Current G01 is sequential Q/K single-red v3: Unit11 strict24 and cold6 exact
contracts pass, and Wave12 G01-only ABBA completes all three representatives with
positive scoped Decode results. See the [Unit11 exact checkpoint](#unit11-exact-checkpoint)
and [Wave12 native checkpoint](#wave12-g01-native-checkpoint) for identities and
limits. The five-model final-v12 queue has started with separate prewarm; combined
E0 results and prefix r3/typechecks remain pending. No final acceptance is implied.
Earlier Unit5/6 fixture failures, Unit8 v2b numeric failures and Wave6 v1/nv48
Flash regression remain explicitly historical, not descriptions of current v3.
Ignored GPU tests are not default-suite passes; native exit0 is separate from
performance qualification. E01's completed isolated model groups remain
warm-neutral/mixed rather than evidence of a cold-off aggregate Prefill gain.

| ID | Work | State | Evidence / decision |
| --- | --- | --- | --- |
| G01 | GDN chain / recurrent+norm | v3 implemented; exact and three representative native gates passed; final combined E0 pending | Sequential Q/K shared reduction, PC20: Unit11 strict24 plus cold6/24 summaries/120 rounds pass persistent state/history/output bit-exact, without tolerance changes. Rotating GPU B/A0.693..0.751; Graph wall MIXED. Wave12 G01-only6K/x1 ABBA all12 runs pass:35B Decode+2.62649%, Flash AD+1.19775%, Ornith Q2+0.49754%; cold-first-off PP not attributed. Historical v1 Flash -2.63779%, v2b numeric failures, full-chain ratio3/4 negatives and extra QK-pair rejection remain preserved below; not current v3 failures or blanket all-model acceptance |
| K01 | IQ3 cooperative matrix | rejected measured full-projection module | All four BM variants exact in raw tests. Final PAGED BM64/BN16 trial: 512 distinct experts, top10, K2560/N640, balanced/skew, 16 exact cases but all80 paired rounds slower; median B/A 1.3805..2.0914. Actual baseline BM32 at prompt512/1024, BM64 at2048/4096. No production selection; no forced GEMV activation quantization |
| K02 | k-quant packed decode | retained narrow unit/module; E0 pending | Large primary24 plus Q6-wide8 cases exact, all160 paired rounds positive. Current BM32 gate: Q3_K K3072/N1024 or reverse, resident/PAGED BN64; Q6_K K2048/N512 or reverse, PAGED BN64 or resident BN128. Q2/Q5, BM64 and unmeasured tuples retain baseline |
| K03 | Tiny expert group reuse | rejected default-route candidate | 1440 exact cases,812 timed cohorts,38 hit/miss pairs. Same-expert cohorts can win, but all measured default four-row all-unique routes regress. No production grouping, CPU readback or guessed-overlap policy |
| K04 | IQ4 pair LUT | retained commit; final five-model E0 pending | `811b1bafa`; 192 raw cases and32 whole-layer450MiB cases exact, latency -30.7..52.6%. Eight AD/UD model A/B runs passed, warm full-model neutral/modestly positive. Windows/RDNA3 IQ4_NL K640/N2560 only |
| A01 | Q8 GQA8/16 reuse | retained measured extension; model evidence partly complete | Flat72 cases exact but G16 slower; segmented28 parity/48 timings positive. Unit4 added108 same-specialization flat/segmented comparisons and120 production Recorder/fallback comparisons. w4 A01 standard A/B completed on35B and Ornith Q2 at6K/30K; other models/150K remain pending. No F16-policy versus generic-F32 bit-exact claim |
| A02 | PV fragment reuse | retained measured unit/module and portable Recorder contract; E0 pending | 24 HD256 long-module cases positive; later16 whole-ubatch cases exact/positive, eight safety-gated cases skipped, not passed. Unit7 portable2 GPU tests pass seven bounded production Recorder cases: CW4/F16scoreQK4 raw partials/combine exact, tails/guards and actual embedded-kernel routes proved. Rows128/129 keep reuse gated off; rows512 reaches reuse. Full2048x150K coverage remains unqualified; no new timing claim |
| Q01 | Exact QSA Top-K | retained unit/module; E0 pending | Exact ties/NaNs/indices/guards and159 scan cases. Actual ratio4 blocks7680/38400 follow-up completed: both ordered/mask variants positive in all six matched rounds for every reported route. Unit4 proves actual row1 scan workspace/default and original fallbacks; no unmeasured crossover claim |
| Q02 | QSA prepare | rejected fused numerical variant; exact split fallback retained | Same projections show one F16 ULP difference at rows33/pos32765/MRoPE. BF16 projection-width parity and split-exact prepare pass; no measured fusion benefit or new default integration claimed |
| H01 | HC exact projection/reads | rejected new cross-layer matcher | Raw pair positive, but cold-Q8 whole HC latency +9.46% in5/5 rounds. Remove/reject matcher, not the existing within-layer fusion. Retain independent n_out=2 write-hazard correction and its regression test |
| E01 | SwiGLU -> Q8 consumer fusion | module/Graph retained; three native model groups execute successfully, warm-neutral/mixed | 1200 boundary modules and60 large-assignment cells exact; all300 large paired rounds positive, B/A0.4609..0.6480 with128..320MiB actual input reads. Unit5 Recorder96 and Unit6 Recorder96/rounding34 pass; Unit7 Graph18 valid plus6 rejection cases pass. w6-e01-flash/35b/ornith each completes all4 native runs. Warm Prefill neutral/mixed and Decode neutral; first-off cold Prefill is lower, so do not claim harmonic +5..10% gain. Broader/final E0 qualification pending; no scratch/lifetime shrink or unknown-driver enable |
| D01 | Dense fixed projections | retained unit/module and Graph; E0 pending | Real128MiB Q8 pair ~1.52->0.75us; whole176MiB GDN neutral/noisy. Unit4 actual Graph4/4 pass, outputs/inputs/weights/guards exact. Only Q8 m1/K2560/N48; independent capability and unchanged multirow/F32 fallbacks |
| M01 | MTP cost/strategy | local CPU reuse retained; policy architecture-deferred | Token-feed capacity/accepted-hidden reuse is implemented, extracted CPU/state/allocation controls pass. Changed-region30K -19.35%;150K mixed,5/7 faster, no full-cycle claim. Unit7 llama235 default tests pass; real MTP E0 remains pending. Adaptive width, vocabulary and cross-request VERIFY ownership remain separate architecture work |
| R01 | Batch-invariant regression coverage | retained regression contracts; Unit7 checkpoint passed | Keep ordinary Decode/VERIFY, batch/fallback, alias/guard and state contracts separate from performance claims. Unit7 Vulkan269 default tests pass with143 ignored, llama235 default tests pass, and all four selected GPU filter invocations pass. This does not qualify every ignored GPU test or final model/E0 gates |
| R02 | Submit/graph overlap | immutable metadata retained; architecture remainder deferred | OnceLock success-only scratch_layout/reset-index memoization; unchanged graph clone, buffers, phases, pager maps and submission order. CPU real-analysis tests include retry/two-plans/row/dtype/read-before-write and zero warm allocations. w4 quick35B A/B completed, Decode neutral; no GPU-work reduction claim |
| O01 | Historical Q8 BM128 | nine qualified BN128 tuples and Graph retained opt-in | Independent Unit4/5 strict intersection is9, not32 or all historical six. Unit7 actual Graph/Recorder all9 selected plus9 neighboring fallback cases pass; last neighbor retains split-K2 and its reducer. Unit6 neighbor failure is historical, repaired without extending selection. q8_prefill_bm128 remains false pending E0; no BN64 extension |
| O02 | 1024-key replay | broad/default replay restoration rejected | Historical35B paged30K/150K regression remains binding. q8_decode_replay_chunk1024=false; existing continuous q8_decode_chunk1024=true is a different retained path. No newly qualified broad replay policy |
| O03 | QSA Q8 scale sharing | rejected current global restoration | Exact, but default CM_QK/PV module regresses in5/6 context/tail groups. Four variants remain separately measured; old+0.2% does not override current module negatives |
| O04 | Direct Q8 K write | typed-F16 module and actual Graph retained; E0 pending | Plain/MRoPE code/scale bytes pass raw/module tests after typed F16 boundary repair. Unit7 all8 actual Graph cases pass, all physical K/V and gathered output bytes exact, with actual plain/MRoPE baseline and fused kernel witnesses. Unit5 name assertion and Unit6 config-OFF RTE-reference versus production RTZ-pack mismatch were fixture issues, now repaired and rerun; no tolerance relaxation |
| O05 | Retained local fusions | existing paths retained; PF4 module-negative | GatedRmsNorm, F32 LinearPair and Q8/F32 down-inject coverage logs pass. Current rotating UD F32 PF4 module +1.59% in5/5 rounds; AD Q8 is a separate fallback control. Existing policy retention is not a new PF4 win; parent model non-regression decides disposition |
| O06 | HC E14 | rebased unit/module retained; E0 pending | Legacy up+mix ~60% slower, rejected. Exact Wave32NR4 rebase -3.5..4.1% in5/5 up+mix rounds; whole HC mixed/no stable gain. Unit4/Unit6 Graph contracts pass; no full-model benefit inferred |
| O07 | Historical resident fusion variants | split disposition, not blanket retention | Indexed split-K direct read exact (including high logical addressing), but ~2x slower than gather+split/combine: reject. PLE LDS gate+norm exact/module-positive, actual Graph2/2 pass inUnit5 and completed Unit6 filter, cfg ple_gate_norm_lds=false pending E0. HC/GU/GDN belong to H01/E01/G01; old quantizing dp4a Decode stays rejected; successor-prefetch ownership changes remain deferred |
| O08 | Previously module-negative variants | rejected implementations closed | Early three HC up+mix, short-K Q8 RM2 and large grouped Q8 qkv+gate remain module-negative. Do not relabel tiny D01 pair or rebased O06 as those rejected variants; no new sweep |
| O09 | Wide expert / K2 / QSA tiles | old negatives closed; existing wide baseline retained | Old TN8/global widening, paged Q4_K K2 and QSA64 scorer not restored. Resident Q4/Q6 BN128/TN4 remains valid baseline; newly tested Q6 BM32/BN128 packed coverage belongs to K02. No broad IQ/PAGED widening |
| O10 | Transfer overlap / early publication | measured protocols rejected; redesign architecture-deferred | Same-batch CPU-push+DMA contention loses; early miss-ready publication changes output. Retain accepted DMA/compute overlap, promotion attachment and source/destination protection. New ownership/readiness protocol requires separate proof, not a local switch |

Every completed item must record actual tests, executable/source identity,
resources, measured timing and retained commit or rejection/defer reason.

## Historical Wave4 Model Comparisons

Historical checkpoint: the following results identify Wave4, not current G01 v3.
Current G01/native status is in Queue and the Wave12 checkpoint below.

All four w4 groups below completed off/on/on/off, four runs per group, exit0 and
no recorded benchmark failures. Values are the saved comparison JSON's aggregate
throughputs, not arithmetic averages of the two arm rates or new worker runs.
Decode x2 is aggregate throughput. Prefill, single-lane Decode and aggregate x2
must not be combined into one speedup. Suites are standard for A01 and quick for
G01/R02. All use Vulkan1, Q8 K/V, MTP off, aggressive profile and768MiB extra reserve.

The four e2e summaries all identify frozen engine-wave4 SHA256:
`AF6504F94A6BF18EE85E5F542E6179354B2A7CEC4E721981D7D2C781D379B52A`.
Its complete dirty/untracked source snapshot is `engine-wave4-source/`.

| Toggle / model | Context | Prefill off -> on tok/s (%) | Decode x1 off -> on (%) | Decode x2 off -> on (%) |
| --- | --- | --- | --- | --- |
| A01 / 35B | 6144 | 3025.99 ->3154.47 (+4.25%) | 119.31 ->133.78 (+12.13%) | 154.24 ->177.70 (+15.21%) |
| A01 / 35B | 30720 | 2873.23 ->2872.45 (-0.027%) | 82.98 ->113.53 (+36.82%) | 118.84 ->146.78 (+23.51%) |
| A01 / Ornith122B Q2_K | 6144 | 1259.55 ->1285.49 (+2.06%) | 32.43 ->34.89 (+7.61%) | 38.19 ->41.11 (+7.65%) |
| A01 / Ornith122B Q2_K | 30720 | 986.76 ->986.22 (-0.055%) | 27.38 ->30.47 (+11.28%) | 32.48 ->36.78 (+13.23%) |
| G01 / 35B | 6144 | 3280.97 ->3289.42 (+0.26%) | 119.27 ->123.25 (+3.33%) | 154.95 ->154.75 (-0.13%) |
| R02 / 35B | 6144 | 3340.39 ->3394.73 (+1.63%) | 120.58 ->120.40 (-0.14%) | 154.26 ->155.07 (+0.53%) |

Sources: [w4-a01][w4-a01], [w4-g01][w4-g01], [w4-r02][w4-r02],
[w4-a01-ornith-q2][w4-ornith]. Their matching `*-e2e-summary.json` files retain
complete commands, per-run ranges, binary hashes, status and report provenance.
Only each named switch is the A/B attribution; do not compare different groups'
absolute baselines as though they were one simultaneous experiment.

A01 Decode gains are demonstrated at these two models/contexts, not all five
inputs or150K. Both30K Prefill results are neutral. G01 x2 ranges overlap, so its
-0.13% is neutral/noisy rather than a demonstrated loss or gain; x1 is +3.33% in
this bounded screen. R02 Decode ranges likewise overlap and are neutral; its
observed Prefill +1.63% does not establish a GPU-kernel gain or broad E0 acceptance.
These historical35B G01 results do not qualify Flash/nv48 or cancel its later
native regression. The scoped Wave6 qualification checkpoint follows; no outcomes
for still-pending jobs are anticipated here.

### Historical Wave6 In-Progress Checkpoint

The following table and pending/proposal wording preserve the then-current Wave6
checkpoint, not today's queue. All19 groups later completed operationally, as
recorded in the Unit8 closure below; G01 localization, exact v3 repair and its
three representative native reruns subsequently completed in Unit11/Wave12.
The old v1 regression remains a negative result for that version only. Current
five-model final-v12 E0 is still pending; do not infer its outcome from either era.

The frozen native `infr-wave6.exe` SHA256 is
`592FAD2D7DB59334FEA9DD724FDD3CF31DEA77E4F610964F8EC7E0389708D59F`.
The parent's19-job isolated E0 queue is defined in [run-model-wave6.py][wave6-plan].
This native engine identity is separate from `unit-v7.exe` and the earlier
engine-wave4 comparisons above. The qualification checkpoint is now:

| Native group | Completed execution / observed behavior | Qualification |
| --- | --- | --- |
| w6-e01-flash / 35b / ornith | Each group completes4 off/on/on/off native runs, all exit0, no benchmark failures, same frozen engine | Warm Prefill neutral/mixed; Decode neutral. Lower first-off cold Prefill confounds the saved aggregate comparison; no demonstrated harmonic +5..10% Prefill gain and no blanket final acceptance |
| w6-g01-flash, v1/nv48 | All4 native runs exit0; Decode aggregate48.8037658->47.5164236 tok/s (-2.63779%). Off48.7224/48.8854 versus on47.7310/47.3037, disjoint ranges | Confirmed scoped regression: v1/nv48 NOT acceptable as-is. Correctness/module success and engine exit0 do not override this non-regression failure |
| w6-g01-ornith | Preliminary positive observations; qualification still in progress | Not final; no acceptance or cross-model extrapolation from queue/runner status |
| Other Wave6 E0 groups | Pending qualification | No success, gain or combined-model acceptance inferred |

E01 evidence: [Flash][w6-e01-flash], [35B][w6-e01-35b], [Ornith][w6-e01-ornith].
Their matching `*-e2e-summary.json` files record all4 engine passes per group.
The first-off Prefill samples are1191.61/2843.39/1236.54 tok/s, versus later warm-off
1359.50/3436.93/1283.59 for Flash/35B/Ornith respectively. Keep those cold/warm
conditions visible rather than promoting an aggregate percentage to a kernel win.
[G01 Flash comparison][w6-g01-flash] records the complete Decode arms above.

Carver's exact-norm v2 remains a proposal; Helmholtz's raw-baseline/cold audit is
pending. No v2 source change, rerun or repair benefit is claimed at this checkpoint.
The prior w4 35B +3.33% result remains historical and scoped, not general G01
retention. Unit/module correctness and cleanup byte identity are not substitutes
for these model-specific performance gates.

## Unit and Module Checkpoint

The table retains version-scoped Unit4/5/6/7 evidence; its G01 row is updated to
Unit11/Wave12. Older proposal/pending statements in explicitly historical sections
do not override this current row or Queue. Final combined E0 remains pending.

Unit-v4 completed all18 parent-serialized filter invocations with exit0. Unit7
completed all six invocations in [wave7-status.json][wave7-status]: Vulkan default,
llama default, A02 portable Recorder, O04 actual Graph, E01 Graph and O01 Graph.
Their results are individually scoped below; earlier Unit5/6 failures are not erased.
Previously completed Unit6 A01/Canvas, K02, GDN/guard, D01, HC up and PLE GPU filters
remain prior-snapshot evidence, not newly rerun Unit7 filters. Raw outputs remain
local; this ledger update performs no build, GPU run, release overwrite or upload.

Raw Unit7 logs: [Vulkan default][unit7-vulkan], [llama default][unit7-llama],
[A02][unit7-a02], [O04][unit7-o04], [E01][unit7-e01], [O01][unit7-o01].

Frozen Unit7 executable identities (SHA256):

- `unit-v7.exe`: `39F7EF0CAECF6227309E285E28977476979F07F4372D509E95A4259C9D3B4595`.
- `cpu-llama-v7.exe`: `78B2AB070018ABB7E6C5F8609A8F01404D96D609B059BAF17F9C552F1C19A034`.

| Item | Completed evidence | Remaining gate / failure classification |
| --- | --- | --- |
| Unit7 default suites | Vulkan269 passed,0 failed,143 ignored; llama235 passed,0 failed,0 ignored | Ignored GPU tests are not default passes. Four selected GPU invocations below also exit0; E0 remains open |
| A01/D01/Q01 | Unit4 attention108 same-specialization plus120 actual Recorder comparisons; D01 actual Graph4/4; Q01 actual workspace/scan default and ratio4 cases | Final combined model/non-regression gate; do not infer exact F32-policy equivalence from specialized F16 attention parity |
| A02 | Unit7 two portable GPU tests pass seven cases: rows128/512 plus query-tail129, context2048/2051/70019, heads16/24/32; exact raw po/pm/pl/combine, padding/guards, actual CW4 and F16scoreQK4 embedded-kernel names/SPV/ABI | Production cfg off/on, rows<512 gate preserved. No external SPVs, timing or filesystem requirement. Historical eight safety skips are not coverage; full2048x150K and E0 remain unqualified |
| K01 | Unit5 full module16/16 exact,464 parity checks,144 quant checks,32 finite checks; >=300ms device warm per side, five ABBA rounds | Correct but module-rejected: all80 paired rounds slower. [Full timing/shape report][k01-module] distinguishes true BM32/BM64 baselines and PAGED-only scope |
| E01 | Unit4 large60/60 exact,2696 comparisons zero differences,128..320MiB consumed inputs; Unit5 actual Recorder96/96; Unit6 portable Recorder96 and rounding34 pass; Unit7 Graph18 valid exact plus6 explicit rejection cases pass | Historical ne32/weight_before+Clamp fixture errors repaired and rerun with guards/repeats/bit checks intact. Native w6 Flash/35B/Ornith each4 engine runs pass; warm Prefill neutral/mixed, Decode neutral, no cold-off aggregate gain attribution. Broader/final E0 pending |
| G01 | Current v3: Unit11 strict24 and cold6 exact state/history/output contracts PASS,24 summary cells/120 rounds; PC20 unchanged. Wave12 G01-only ABBA all3 representatives/all12 runs PASS, Decode35B+2.62649%, Flash AD+1.19775%, Ornith Q2+0.49754% | Exact repair/localization is completed, not a pending v2 proposal. Rotating GPU improves but Graph wall MIXED; no cold-first-off PP gain attribution. Historical v1 Flash regression and v2b byte failures remain version-scoped. Final-v12 five-model combined E0 and broader/context gates remain pending; no overall acceptance |
| O01 | Nine BN128 tuples survive independent strict Unit4/5 intersection; Unit7 Graph/Recorder all9 selected plus9 neighboring fallbacks pass, actual kernels/guards/unchanged weights checked | Unit6 neighbor failure resolved by preserving the real split-K policy in the baseline reference; final neighbor uses split2 plus reducer, not a forced unsplit reference. cfg remains false pending E0. No BN64 or extra-tuple acceptance |
| O04 | Raw typed-F16 module exact; Unit7 all8 actual Graph plain/MRoPE cases pass at rows1/33/129, positions32767/32765/150003 and rope64/256; every physical K/V code/scale and gathered output byte exact | Historical Unit5 name mismatch resolved to actual plain qk_norm_rope_bda; Unit6 firstdiff7170 was CPU RTE versus observed production PackHalf2x16 RTZ, not an off/on failure. Strict RTZ reference and regression test repaired; no tolerance relaxation or production math change. E0 pending |
| O07 PLE | Postprojection module50 paired rounds positive; actual Graph2/2 pass in Unit5 and the completed Unit6 filter | cfg false, measured geometry/device/alias/MTP restrictions remain; E0 pending |
| H01/O06 | H01 dual-write structural regression kept; O06 rebased actual Graph tests pass | H01 matcher module-rejected (+9.46% cold-Q8 HC); O06 whole-HC result remains mixed despite positive up+mix pair |
| R02/M01 | Real scratch analysis retry/two-plan/row/dtype/read-before-write and warm-zero-allocation CPU controls; local MTP reuse controls; Unit7 default CPU suites pass | Host-analysis/allocation findings only. Preserve graph clone, phase/reset/pager ownership and fixed MTP semantics; complete-cycle/E0 gates remain parent-owned |

E01 current source uses compact safe2D grid and stack descriptors. Exact arithmetic
eligibility is Windows discrete RDNA3 plus reported AMD_PROPRIETARY driver8389003;
do not silently broaden the observed `(g*u)*reciprocal` order to other drivers.
The raw Recorder fixture may exercise scaling+Clamp directly, but MoeFfn Graph
validation forbids that combination; a rejected graph is not a numerical pass.

## Cleanup Identity Checkpoint

Parent cleanup removed unselected prototype source without altering the selected
production SPIR-V modules. Saved optimized proofs under `history-audit/final-cleanup/`
report byte identity for all29 A01 production tuples, both2 GDN retained norm
variants (SiLU/sigmoid), and all4 O03 baseline production variants (scalar, flash,
CM_QK, CM_QK/PV). Sources: [A01 proof][cleanup-a01], [GDN proof][cleanup-g01],
[O03 proof][cleanup-o03]. These later parent proofs supersede the earlier
artifact-only/preprocessing-only status in the preparation manifests; they are
not new GPU measurements or performance wins.

Keep archived before/after sources, tool/define/hash provenance and negative
research logs. In particular, GQA8/GQA16 research coverage is not a production
selector claim; GDN full-chain negatives and O03 global-restoration rejection
remain binding. Macro-off byte identity does not rehabilitate rejected macro-on
variants, enlarge any shape/device gate or establish E0 success.

## O01 Finite Gate

Only the following `(K,N,M)` tuples are independently repeated module-qualified,
all Q8_0, BN128, no split-K. `kernels.vulkan.q8_prefill_bm128=false` remains the
current default pending E0; the selected and neighboring Graph/fallback gates
have passed in Unit7:

```text
(3072,8192,512)    (8192,3072,1024)
(3072,12288,512)   (3072,12288,1024)
(2560,10240,1024)  (2560,12288,512)
(2560,12288,1024)  (2560,6144,512)
(6144,2560,512)
```

[Independent review][o01-review] finds32 Unit4 strict wins,10 Unit5 strict wins,
only9 in their intersection, all BN128 and zero BN64. One Unit5-only winner is
not added, and23 originally proposed extension tuples are excluded. Tests used
>=128MiB rotating valid synthetic Q8 weights and included cast/GEMM/staging;
inventory geometry is not proof of live-model route reachability or E0.

## Architectural Closure

- M01 adaptive drafting, vocabulary restriction and cross-request VERIFY batching
  change policy/state ownership and remain deferred; local Vec reuse is not that work.
- R02 metadata memoization is the finite retained implementation. Broad owned-graph
  promotion, phase/pager redesign and caller-class submit tuning are deferred, not
  claimed implemented by the cache.
- O10 measured same-batch contention and incorrect early-ready publication remain
  rejected. A new asynchronous readiness/ownership protocol needs separate review;
  accepted DMA/compute overlap is preserved.
- O07 successor prefetch and historical unresolved source identities are not new
  retained wins. Indexed-reader rejection, positive PLE pair, and H01/E01/G01
  outcomes stay separate; no replacement sweep is opened by this ledger.

The table records measured scopes rather than attributing every saved microsecond
to full-model throughput. Policy/device/ubatch gates must be recorded in each
retained local commit. Rejected prototype sources are archived before cleanup.

## Historical First Device Pass

- `g01-device-first.log`: RX7900XTX explicitly asserted. The original full-chain
  group-per-K-head design underfills the GPU on ratio3/4. It is not a universal win.
  Recurrent+norm preserves recurrent and convolution states bit-for-bit, but output
  differs by <=1.35e-7; this is not yet accepted as a production rewrite.
- `k04-device-first.log`: K32/64/640, N64/2560; BM32/64; 16 count/tail cases;
  STREAMED/PAGED; empty bucket and output canaries. All raw f32 output bytes match.
  This initial benchmark uses three weight banks fitting L2, so its 27-47% device
  reductions must not be presented as whole-model or cold-weight speedups.
- K04 executable SHA256 `99F8483259569AA29D574CE0DAE13ACB6F53D084FF3C1CAEF51CC63A5448CF36`.
  Same-binary A/B uses `kernels.vulkan.iq4_pair_lut=false/true`, only on
  Windows/RDNA3 IQ4_NL K640 N2560 expert MMQ. All other routes retain their fallback.
  Native quick benchmarks use true6K Prefill, 256 Prefill warmup, 512 Decode warmup,
  1024 scored tokens, x1 and x2; AD and UD alternate off/on/on/off.
- Windows residency workaround is explicit and constant: automatic aggressive
  plus 768MiB additional VRAM reserve. No memory-policy default was changed.

[w4-a01]: D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/w4-a01-comparison.json
[w4-g01]: D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/w4-g01-comparison.json
[w4-r02]: D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/w4-r02-comparison.json
[w4-ornith]: D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/w4-a01-ornith-q2-comparison.json
[k01-module]: D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/k01/layer-module-gates-unit5.md
[o01-review]: D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/o01/unit5-independent-review/analysis.json
[wave7-status]: D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/wave7-status.json
[wave6-plan]: D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/run-model-wave6.py
[cleanup-a01]: D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/history-audit/final-cleanup/offline-proof/summary.json
[cleanup-g01]: D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/history-audit/final-cleanup/g01-offline-proof/summary.json
[cleanup-o03]: D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/history-audit/final-cleanup/o03/offline-proof/results.json
[unit7-vulkan]: D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/__-unit-v7.log
[unit7-llama]: D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/__-cpu-llama-v7.log
[unit7-a02]: D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/flash_pv_reuse_contract-unit-v7.log
[unit7-o04]: D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/o04_segmented_write_actual_graph-unit-v7.log
[unit7-e01]: D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/swiglu_quant_moe_graph_-unit-v7.log
[unit7-o01]: D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/q8_prefill_bm128_graph_-unit-v7.log
[w6-e01-flash]: D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/w6-e01-flash-comparison.json
[w6-e01-35b]: D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/w6-e01-35b-comparison.json
[w6-e01-ornith]: D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/w6-e01-ornith-comparison.json
[w6-g01-flash]: D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/w6-g01-flash-comparison.json

## Historical Unit8 And Native Closure Checkpoint

Historical checkpoint: v2b failures and diagnostic-pending statements below are
preserved as recorded. They are superseded for current G01 by the Unit11 v3 exact
repair and Wave12 native checkpoint, without erasing any failed-version evidence.

All19 isolated Wave6 groups completed four native runs each. The separate K02
35B BAAB adds four, for80 operationally successful cases. This is not automatic
performance acceptance. The independent review retains every sample, phase and
section, including first-process negative Prefill and Decode observations:
[Wave6 review](D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/wave6-isolated-review.md).
The final combined long-context regressions remain pending.

Unit8 build passed with portable/static flags. Frozen Vulkan test SHA256 is
`5CA0EAE7FCB3D4F94C8FF5AF65C98B6E891AFBF336C469EDE4A0ECE70BBEE898`;
its exact dirty/untracked inputs are in `unit-v8-source/`. Default suites pass:
core255 (one ignored), Vulkan271 (144 ignored), llama235. The earlier Unit7
E01 Graph18 valid plus6 rejection cases, O04 Graph8, and O01 selected9 plus
neighbor9 remain repaired/passed, not unresolved historical fixture failures.

G01 v2b integrates the matched20-byte push ABI and baseline-shaped norm reduction.
It does NOT pass exact output: the strict test stops at nv32/SiLU/input1e-4,
output index512, bits857549721 versus857549723. Persistent state matches exactly.
The independent rotating fixture also fails its first nv32/SiLU bank, with1019
final-output words different; no timing cells are qualified. Neither tolerance
relaxation nor a causal claim about native regression is made. Raw-R and same-R
norm diagnostics are a separate pending test, not production eligibility.

The core liveness helper replaces operand `io()` lists with the existing
`io_inline()` API, retaining read-before-write and next-rewrite boundaries.
Its equivalence test passes. Six alternating ABBA/BAAB rounds on a generated
48-layer/945-op graph give aggregate new/old times0.1476,0.1534,0.1568 for
next-write, no-write and tail-read cases respectively. These are CPU-only
generated-graph timings, not actual native lowering or universal zero-allocation
claims. Final E0 will compare against frozen pre-helper Wave6, rather than hide
this always-on change inside both arms.

M01 standalone CLI old/new completed at6K and30K, four runs with the same prompt,
512 generated tokens, ubatch512 and all campaign switches off. Greedy displayed
text hashes, cycles, accepted/drafted counts and alpha agree exactly at each
length. Decode6K is30.8/30.8 tok/s;30K32.5/32.4, descriptively neutral rather than
a claimed full-cycle acceleration. Prefill is230/233 and240/241 tok/s in this
specific manual512 MTP route, not the normal aggressive Prefill route. Evidence:
[M01 completion](D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/m01/runs/20261023-103307-3817f977/completion.json),
[M01 results](D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/m01/runs/20261023-103307-3817f977/results.json).

Partial-shader cleanup now proves all14 before/after production SPVs byte-identical;
selected Reg-O reuse variants are untouched. Cumulative commit-staging-r2 repairs
module ownership/order and passed full child resolution, but remains pre-Unit8
and has NOT received real Cargo prefix typechecks or final qualification.

## Unit11 Exact Checkpoint

The Unit8 v2b failures and M01/native history above remain historical evidence,
not erased or retrospectively qualified. Unit11 repairs G01 with sequential Q/K
reductions reusing one shared reduction array (single-red v3), retaining the
20-byte push-constant ABI and strict byte contracts. No tolerance was relaxed.
The frozen `unit-v11.exe` SHA256, independently rehashed for this ledger update, is
`5EFE8E4336323BB3F67BC13E51EF111C6C1C9B8AB787181E823373D7986F2BF1`.
Exact source capture: `unit-v11-source/`; selected shader SHA256:
`298FF0506FA15C683B466D76586FD11B4F819F4DA3E0F652B9AEA43286A8BA98`.
Some fixture/log labels still say `G01v2b` / `v2b_pc20`; these passes identify the
Unit11 captured v3 bytes, not the older failed v2b shader.

| Unit11 contract / measurement | Observed result | Qualification boundary |
| --- | --- | --- |
| Strict complete Graph | All24 cases PASS: nv32/48/64, SiLU/Sigmoid, input scales0/1e-4/0.7/4; persistent state and final output bit-exact, actual split/fused routes verified | No approximate comparison or tolerance change |
| Cold/rotating fixture | All6 cases PASS;24 summary cells,120 paired ABBA/BAAB rounds; strict persistent state, Conv history and final output bit-exact | Includes strict four-step/rotating contracts; does not independently expose raw pre-norm R |
| Rotating Recorder GPU module | Six paired-median on/off (B/A) time ratios0.693195..0.751381, approximately0.693..0.751; all30 rotating rounds faster | Module time only, not native throughput or guaranteed cache misses |
| Graph wall | MIXED; nv32/Sigmoid rotating median ratio1.071650 (+7.165% time),3/5 rounds slower; other five rotating median cells improve | Neither all cells nor all rounds are positive; no full-model gain inferred |
| Default Vulkan / speculative guard |272 default tests PASS; MTP guard PASS | Ignored GPU tests are not default-suite passes; native qualification remains separate |

Sources: [strict24 log](D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/deltanet_norm_complete_graph_matches_split-unit-v11.log),
[cold6 raw log](D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/g01_cold_state_graph_recorder_exact_and_paired-unit-v11.log),
[rotating-state analysis](D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/g01/unit11-rotating-state-summary.md).
The analysis retains all pairs; its ratios are medians of paired time ratios,
not ratios of independently reported off/on medians.

Raw-R diagnostics localized the observed mismatch before norm; same-R norm replay
was exact. Runtime Q-only load/normalization v1 had no effect (446 raw-R words
still different), while query sequential single-red v2 repaired both known
synthetic/cold diagnostic samples. These controlled observations support the
phase-organization repair, not an unmeasured ISA/compiler causal claim or general
sample coverage. Production v3 Q/K correctness is established by the Unit11
contracts above, not by extrapolating those two diagnostic samples.

Temporary diagnostic child/mount removal followed exact-byte archival; failures,
external variants and [diagnostic archive](D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/g01-v2b-localization/archive-unit11/README.md)
remain preserved. Current completion/removal and binary identity are recorded in
the [campaign manifest](D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/manifest.json)
and [current handoff](D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/NEXT.md);
the archive README's earlier running/prepared-only status is historical.

At the earlier Unit11-only checkpoint, Wave12 had not yet run and G01 native
qualification was pending. That native status is historical and superseded by the
completed three-group Wave12 checkpoint below. Final five-model combined E0,
prefix r3 assembly and actual cumulative Cargo typechecks remain pending; r2's
CPU-only input/schema dry-run does not qualify r3 or any prefix build. No default-on,
cross-model, commit-prefix or future acceptance is claimed. This ledger-only
update runs no Cargo or GPU work.

## Wave12 G01 Native Checkpoint

This later checkpoint supersedes the Unit11-only paragraph's pending G01 native
status; earlier failed versions, negative native results and M01 history remain
preserved. G01 v3 completed all three representative ABBA groups (off/on/on/off),
12 native runs total, all status passed/exit0 with no recorded error lines or
benchmark failures. All runs identify `infr-wave12.exe` SHA256
`863CB86FFCE3D727077E41738F6CE74152A7F908C1690BBA3C28CD3CA4558BF1`.
These are quick-suite6144-context, x1 Decode, G01-only comparisons within the
same frozen engine; K04 stays enabled in both arms and the other campaign toggles
stay off. They are not the Wave12-versus-Wave6 combined-feature comparison.

| G01 v3 representative | Decode off -> on tok/s | Change | Observed arms / scope |
| --- | --- | --- | --- |
| 35B |121.0473718 ->124.2266632 |+2.62649% | Off120.9555/121.1393; on124.2228/124.2305 |
| Flash-Next AD |48.4594749 ->49.0398988 |+1.19775% | Off48.6999/48.2214; on48.9950/49.0849 |
| Ornith122B Q2_K |33.1710957 ->33.3361346 |+0.49754% | Off33.2571/33.0855; on33.3784/33.2940 |

The percentages use the saved comparison aggregates. Each group's first off
Prefill is cold; saved aggregate PP deltas are NOT attributed to G01 acceleration.
Positive Decode observations are bounded to these three ABBA samples at6K/x1,
not blanket all-model/context acceptance or proof of any ISA-level cause. They
do not erase the historical Flash v1/nv48 regression.

Sources: [35B comparison](D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/w12-g01-35b-comparison.json),
[Flash AD comparison](D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/w12-g01-flashnext-comparison.json),
[Ornith Q2 comparison](D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/w12-g01-ornith-q2-comparison.json).
Matching `*-e2e-summary.json` files retain all four runs, commands, executable
identity, status and raw-report paths for each group;
[native queue status](D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/g01-native-v12-status.json)
records all three groups passed.

Parent has now started the five-model `final-v12` full queue, comparing frozen
Wave12 against frozen Wave6 with a separate prewarm phase. Full6K/30K/150K,
x1/x2 and repeated Decode qualification remains PENDING; no final queue outcome,
combined-feature non-regression or future acceptance is inferred from its launch.
Prefix r3 and actual cumulative typechecks remain pending. This documentation-only
update starts no Cargo/GPU work and changes no source, index or commits.

## Selected Matrix Completion And Bounded Closure

Current checkpoint superseding the earlier queue-in-progress status above:
`final-summary.json` / `.md` are COMPLETE for the explicitly selected scope,
`complete=true`, `operational_success=true`, with zero missing/failed issues and
zero per-model issues. This is operational/coverage completion, NOT automatic
performance acceptance, all-positive results or approval of every feature default.
Off remains frozen Wave6 and on frozen Wave12, with the separately recorded
prewarm phase; their binary identities are the hashes already recorded above.

The selected scope is `reduced-ornith-q3`: 35B, Flash AD, Flash UD and Ornith Q2
each cover full6K/30K/150K, Decode x1/x2, three repetitions. Ornith Q3 covers ONLY
quick6K x1/x2, one repetition per arm; planning155392 is capacity planning, not a
150K measurement. Q3 has no30K/150K observations or long-context inference.
Stopping the oversized controller at the completed Q2 boundary did not discard
that model's saved passing native arms or turn the intentional scope change into
an engine failure. No further long model matrix is opened by this checkpoint.

Observed combined-feature changes from the saved report (percent; full-model
columns list6K/30K/150K in that order, x2 is aggregate throughput):

| Model / selected scope | Prefill | Decode x1 | Decode x2 |
| --- | --- | --- | --- |
| 35B, full3 |+0.45 /+3.73 /+2.62 |+15.05 /+39.82 /+65.22 |+14.52 /+23.80 /+61.49 |
| Flash AD, full3 |+1.94 /+2.43 /+3.92 |+1.64 /+1.08 /+2.74 |+1.70 /+0.78 /+2.07 |
| Flash UD, full3 |+0.84 /+3.19 /+4.86 |+0.67 /+1.11 /+1.76 |-0.14 /-0.21 /+1.73 |
| Ornith Q2, full3 |+0.96 /+3.44 /+2.42 |+6.70 /+11.59 /+37.59 |+8.19 /+13.36 /+36.01 |
| Ornith Q3,6K only, rep1 |+0.98 |+3.80 |+5.26 |

The small Flash UD x2 negative observations are retained, not rounded into wins;
the report preserves repetition ranges, lane/section rates and all saved arms.
These combined-engine comparisons do not isolate individual feature contributions
and cannot mask an isolated regression. Sources:
[final summary JSON](D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/final-summary.json),
[final summary report](D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/final-summary.md).

All four legacy route-witness cases are complete on frozen Wave12 / Flash AD;
these prove actual selected kernels, not merely config flags or process exit0:

| Route | Actual kernel / shape | Recorded dispatch count |
| --- | --- | --- |
| O01 pp512 |native_gemm_warp_q8_0_n128_ag_bm128, m512/K6144/N2560 |576 |
| O01 pp512 |same BM128, m512/K2560/N6144 |432 |
| O01 pp512 |same BM128, m512/K2560/N12288 |144 |
| Q01 pp512 |qsa_indexer_topk_ordered |96 |
| PLE pp512 |qwen_ple_gate_norm_lds2560 |12 |
| PLE tg1 |qwen_ple_gate_norm_lds2560 |1 |

[Route status and profile identities](D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/minimal-closure-v12/mini-v12-closure-r1/route/status.json)
record all four cases witnessed. Profile durations and legacy printed throughput
are NOT E0 performance data; legacy sole-sequence route reachability is not proof
that the API/parallel suite uses identical row shapes or dispatch counts.

Remaining runtime closure is limited to the TWO existing isolated Flash6K
downsides, Q01 and O01, via short same-Wave12 BAAB groups. At this checkpoint they
are in progress, not completed or accepted. O01/PLE default decisions and any
Q01 automatic-gate narrowing await parent judgment; no config/source change or
default-true eligibility extension is claimed here. Final commit assembly,
cumulative prefix typechecks and the subsequent unified build/regressions remain
pending. In particular, later O04/O06 observer fixes are unbuilt source changes,
not bytes measured by frozen Wave12. This append-only documentation update starts
no Cargo/GPU work and changes no other source, worklog, index or commits.

## Final Gate And Default Disposition

Both bounded warm same-Wave12 BAAB groups completed all four native runs with
status passed/exit0. They retain the frozen Wave12 identity above; this is not a
new build or remeasurement of subsequent source fixes. Final dispositions are:

| Item | Bounded closure observation | Parent disposition |
| --- | --- | --- |
| Q01 | Warm Flash6K PP -0.0249%, Decode -0.2646%; off/on ranges overlap for both | Retain current automatic gate and qsa_topk_ordered=true; no proposed >=7680-block narrowing, no claimed short-context gain |
| O01 | Warm Flash6K/ubatch512 PP246.91654->245.69618 tok/s (-0.49424%); off246.75113/247.08216 versus on246.06691/245.32656, disjoint ranges; repeats the prior scoped negative | Exclude all three witnessed Flash M512 tuples, retain ONLY six tuples, default-true only within that narrowed predicate. Helm owns predicate, Graph regression and config integration |
| PLE | Legacy route witnesses reached rows512 (12 dispatches) and rows1 (one dispatch), as recorded above | ple_gate_norm_lds=true; existing measured row/HC4/2560/device/alias/observer/MTP fallbacks remain mandatory, not blanket eligibility |

O01 excluded `(K,N,M)` tuples are `(6144,2560,512)`, `(2560,6144,512)` and
`(2560,12288,512)`. Current retained disposition is exactly:

```text
(3072,8192,512)    (8192,3072,1024)
(3072,12288,512)   (3072,12288,1024)
(2560,10240,1024)  (2560,12288,1024)
```

Historical nine-tuple module/Graph passes, actual Flash route counts and negative
timings remain preserved. Correctness and actual dispatch reachability did not
override the isolated PP regression. The six-tuple disposition supersedes the
earlier Queue/O01 finite-gate descriptions without erasing those earlier results.
O01 default-true must still require its narrow Q8/BN128/shape/device/route checks;
the three excluded tuples must use the original production fallback.

Sources: [Q01 warm BAAB](D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/mini-v12-closure-r1-q01-6k-baab-comparison.json),
[O01 warm BAAB](D:/AIinfr/infr/local-work/results/infr/20261023-034328-kernel-campaign-c61e/mini-v12-closure-r1-o01-6k-u512-baab-comparison.json).
Matching e2e summaries preserve all four on/off/off/on arms and executable hashes.

E0 scope is now closed at the selected matrix plus these necessary short anomaly
checks; do NOT expand the matrix. New CPU/Graph gate regressions, O04/O06 observer
regressions and the parent's unified GPU verification remain pending, along with
current cumulative prefix typechecks and final commit assembly. This ledger
records the final selection/default decisions, not a successful build or test of
Helm's integration or later observer fixes. No additional native/GPU/Cargo run,
other source/worklog edit, index operation or commit is performed by this update.

## Unit13 Final Qualification Checkpoint

This later checkpoint upgrades the preceding pending unit/default/observer status;
older pending statements and failed-version evidence remain historical. The frozen
`unit-v13.exe` SHA256, rehashed while reading the saved results, is
`BBCB94F6A55B8439450E6C07FDDC4ABFFC44701DCA0929263C386C7C6C558DEC`.

| Unit13 scope | Completed qualification |
| --- | --- |
| Default core |255 passed,0 failed,1 ignored |
| Default Vulkan |275 passed,0 failed,147 ignored; includes new O04/O06 pure-CPU observer controls and O01 six-tuple CPU gate test |
| Default llama |235 passed,0 failed,0 ignored |
| O04 actual Graph observer GPU |All6 plain/MRoPE partial Copy, sparse CopyStrided and full-Copy controls PASS; every physical K/V and output byte exact, real fused/fallback kernel witnesses, A01 segmented Decode disabled in both arms |
| O06 actual Graph rewrite GPU |All12 case/seed combinations PASS: prefix/suffix/sparse partial writes decline fusion; full Copy, packed CopyStrided and partial-then-full remain eligible; main/side-observer raw bits and guards exact |
| O01 actual Graph/Recorder GPU |All6 retained tuples,3 explicitly excluded Flash M512 tuples and9 neighboring fallbacks PASS, with actual route checks, guards and unchanged weights; split-K fallback/reducer behavior preserved |

Ignored GPU tests are not counted as default-suite passes; the three targeted GPU
filter invocations above were separately executed by the parent. Intentional
panic/error-path probes printed in default logs do not override their recorded
zero-failure test summaries. Evidence:
[core default](D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/__-cpu-core-v13.log),
[Vulkan default](D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/__-unit-v13.log),
[llama default](D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/__-cpu-llama-v13.log),
[O04 observer](D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/o04_overwrite_observer_actual_graph_byte_contract-unit-v13.log),
[O06 rewrite](D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/hc_up_mix_graph_gate_rewrite_contract-unit-v13.log),
[O01 selected/fallback](D:/AIinfr/infr/local-work/evidence/infr/20261023-034328-kernel-campaign-c61e/q8_prefill_bm128_graph_-unit-v13.log).

Final defaults are O01=true under ONLY the six-tuple predicate and PLE=true under
its retained narrow guards. Q01's warm BAAB is neutral; its current gate remains
unchanged. These current-source correctness/default passes do not rerun or
reattribute frozen Wave12 E0 timings, and do not restore excluded Flash tuples.
Independent CPU checks of all13 cumulative commit prefixes and actual final local
commits remain pending; a successful unified build is not a prefix-check pass.
E0 scope stays closed. This update reads existing logs only: no Cargo/GPU run,
source/worklog change, git/index operation or commit.
