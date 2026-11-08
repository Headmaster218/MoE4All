# DSv4.1 在 MoE4All 上首次出 token（2026-11-07 23:04）

## 结果

```
$ ./target/release/infr run DeepSeek-V4.1-Flash-...-00001-of-00004.gguf "Hello" \
      --max-new 8 --temp 0 --dev Vulkan1

[infr] host DMA import total: 123.00/123.00 GiB across 40/40 arena(s)
[infr] unified VRAM physical residency verified arena_bytes=12768860160 shards=1
Thinking. The user wants me to evaluate the
[prefill 32 tok @ 7 tok/s (4873 ms) | decode 8 tok @ 6.5 tok/s]
```

- 40/40 专家 arena 全驻留，123.00 GiB 导入完成
- 真正生成了 8 个 token，且内容连贯（"The user wants me to evaluate the"）
- 逐 token 采样 `--temp 0`

## 跨过的是什么

之前唯一的具名边界：

```
a V4 compressor always carries an APE table     (runner.rs:6900)
```

根因不是"缺一张表"，而是 **V4.1 的 `compress_ratios` 是角色表不是比例**：

| | V4 | **V4.1** |
|---|---|---|
| `compress_ratios` | 每层比例 `{0,4,128}` | **角色表** `{0,1,2}`，len 43 |
| 有 compressor 的层 | 所有非零层 | **只有 {2,8,14,20}** |
| 其余 role≠0 层 | 各持一个 compressor | **34 个复用层**，读源层发布的流 |
| `attn_compressor_ape` | 强制 | **全文件 0 个** |
| `attn_compressor_gate` | 每层 | **只在 {2,8,14}**，第 20 层没有 |

发射代码用 `ratio4 != 0` 判据 → 38 层全进 V4 的 per-layer compressor 分支 → 34 个复用层 panic。

## 三处修复（commit `68a5f1f`）

1. **compressor 分支改按归属判定** `lw.dsv4_compressed.is_some()`
   —— 正是 `wpush`/`wload` 用 `dsv41_is_kv_source(l)` 构造出来的判据。
   复用层不再发射 compressor，落到 source-plane gather。

2. **`Op::Dsv4Compress.ape` 改为 `Option`**
   —— V4.1 一个 APE 都没有。但**没有**把"缺失"穿透到 recorder/shader/四个后端，
   而是在 seam 里绑一张共享的全零 `@zero_ape`（分配/上传完全照 `hc_ones` 的做法），
   于是 score ring 加上 `+ 0.0` —— 与"全零表"的算术完全一致，而两个 arch 的发射语句一模一样。
   V4 仍传它自己的真表。

3. **ratio-1 源层没有 pooling gate**
   —— 第 20 层每行只池化 1 个 token，softmax 只有一个元素 = 1.0。
   `wload` 早就按 `ratio > 1` 条件读取；发射端现在同意，而不是去要一个被故意没加载的张量。

## 回归

```
synthetic_deepseek2:  38 passed, 0 failed, 8 ignored
```
修复前是 36 passed / 2 failed。V4 全部 36 个用例保持绿。

## 当前速度与 poorseek 的差距

| 项 | MoE4All(Vulkan) | poorseek(SYCL) |
|---|---:|---:|
| decode | **6.5 t/s** | ~27.7 t/s |
| prefill | **7 t/s** | 624–1063 t/s |

差 4.3x（decode）和 ~90x（prefill）。**但这是首次可运行，不是优化后的数字**，
而且只有 8 个 token、32 个 prompt token，冷启动未摊薄。

## 已知的下一个瓶颈（已定位，未修）

1. **`batched_prefill_ok` 里有 `!is_dsv4_family(c)`**（runner.rs:12253）
   —— 整个 DSv4 家族被排除在批量预填充之外，V4.1 现在是**一次一个 token 预填充**。
   这是 prefill 只有 7 t/s 的直接原因，也是与 poorseek 差距最大之处。
   代码注释给了理由：`hc_mult` 宽残差无法跨 span 携带，且从未有 batch>1 的 V4 图被执行过。

2. **`moe_batched_ok` 是全有或全无判定**（runner.rs:1349）
   —— 40 层里只有 1 层满足 MMQ，97.0% 的专家权重走 per-token id-GEMV。

3. V4.1 专家是 **IQ1_M(60) / IQ2_XXS(42) / IQ2_XS(18)** 三路混合，不是全 IQ1_M。
   IQ2_XXS/IQ2_XS 已有 MMQ shader，但 down bank 是 IQ1_M 的层会被整体拖慢。

---

## 新增测量（2026-11-07 23:24）

| 配置 | prefill | decode |
|---|---:|---:|
| 默认 ctx（1048576，clamp 到 688256） | 7 t/s | **6.6 t/s** |
| `--ctx 8192`（state 从 10.86 → 0.13 GiB） | 6 t/s | **6.4 t/s** |

**结论：把 VRAM 从专家缓存的角度多腾出 10 GiB，decode 反而略降。** 所以专家缓存命中率不是当前瓶颈。

VRAM 计划（`--ctx 8192`）：
```
total_room=27.65 GiB  fixed=6.06  state=0.13  runtime_elastic=11.05
expert_cache_target=10.16 GiB  elastic_pool=21.21 GiB (k=Q8_0, v=Q8_0)
```

## 带宽下限估算

- 每层专家总量 3149 MiB / 384 专家
- 每 token 选 6 专家 × 40 层 = 240 次专家访问 = **1.92 GiB**
- VRAM 缓存只装得下 123 GiB 的 ~8%，未命中约 **1.77 GiB/token** 要走 host RAM
- 若 RAM 带宽 60 GB/s → 30 ms/token → **33.9 t/s 上限**

实测 6.4–6.6 t/s，距带宽上限还有 ~5x，**说明瓶颈不在带宽，在调度开销**。

## 调度开销 —— 我最初的判断是错的，这里是修正后的

**先说我错在哪。** 我最初看 per-op 计数（`native_idm_iq2xxs_paged` 3004 次、
`hyper_mix_gates_decode` 3120 次……）时，把这些**累计**次数当成了每 token 的次数，
算出「每 token 约 1.8 万次 dispatch、每次 8.7us」，并据此断言瓶颈是 dispatch 数量。

`INFR_PROF_OPS` 的**单次 forward 行**才是可比的：

```
[prof:vulkan] per-op device time — 1.8 ms over 69 dispatches
[prof:vulkan] per-op device time — 4.3 ms over 69 dispatches
[prof:vulkan] per-op device time — 0.3 ms over  4 dispatches
```

**每次 forward 只有约 69 次 dispatch，设备时间 0.2–4.8 ms。** 1.8 万那个数不成立。

## 那么 156 ms/token 花在哪

`INFR_PROF_STAGES=1` 的 decode 汇总行给出了答案：

```
[cpu prof] prompt 32 tok in 5.91s (5.4 tok/s) | decode 32 tok in 5.63s (5.69 tok/s)
[dec prof] setup(build+compile+bind) 0.073ms/tok | exec(record+submit+gpu) 175.662ms/tok
```

- **图构建/编译/绑定：0.073 ms/token** —— 解码回放磁带确实在工作，构建不是开销
- **record+submit+gpu：175.7 ms/token** —— 全部时间在这里

但单个 forward 的 `submit+gpu` 只有 1.4–3.8 ms，40 层加起来也远不到 175 ms。
**差额是 GPU 在等：等专家数据的 DMA。** 123 GiB 专家里只有约 23 GiB（`arena_bytes=23262105600`）
能同时驻留，其余每次 forward 都要从 host RAM 经 PCIe 搬。

## 三个被我逐一排除的假设

| 假设 | 实验 | 结果 |
|---|---|---|
| 专家缓存太小 | `--ctx 8192` 多腾 10 GiB | 6.6 → 6.4 t/s，无改善 |
| 提交拆得太碎 | `submit_dispatches=512`（默认 64） | 6.4 → 6.6 t/s，噪声内 |
| dispatch 数量太多 | 见上：每 forward 仅 ~69 次 | **前提本身错了** |

所以瓶颈是**专家流式传输的带宽/延迟**，与 poorseek 那边「闸门占 40%」是不同的问题 ——
**我先前把它类比成同一个问题，那个类比是错的。**

## 下一步（按收益排序）

1. **降低专家流式传输的开销**：这是实测定位到的唯一大头（175.7 ms/token 里绝大部分）。
   方向包括：把 23 GiB 常驻集挑得更准（复用率高的专家优先）、增大 arena、
   或用 in-place 索引让 host 侧直接参与计算而不过 PCIe。
2. **批量预填充**：`batched_prefill_ok` 的 `!is_dsv4_family(c)` 让 V4.1 一次只预填充一个 token，
   这是 prefill 7 t/s 的直接原因，也是与 poorseek 1063 t/s 差距最大之处。
3. `moe_batched_ok` 的全有或全无判定（40 层只有 1 层满足 MMQ）。

## 排除的两个假设（2026-11-07 23:30）

| 假设 | 实验 | 结果 |
|---|---|---|
| 专家缓存太小 | `--ctx 8192` 多腾 10 GiB 给专家缓存 | 6.6 → **6.4 t/s**（无改善，略降） |
| 提交拆分太碎 | `--set device.submit_dispatches=512`（默认 64） | 6.4 → **6.6 t/s**（噪声内） |

**两个都排除。** 所以不是缓存命中率，也不是 submit 批次大小，
而是**每 token 约 1.8 万次 dispatch 的固有开销**（每次约 8.7 us，含 desciptor 绑定、
push constant 拷贝、barrier 判定等宿主侧成本）。

这与 poorseek 的结论同族 —— 那边是「每次 launch 都等上一次」的闸门占 40%，
这边是 dispatch 数量本身。**两条线都是调度级问题，不是算力。**

## 第四个排除项：显式扩大专家缓存也没用（2026-11-08 00:05）

```
--set paging.cache=22GiB  ->  expert_cache_target=10.23 GiB, arena 21.66 GiB
                           ->  decode 6.6 t/s   （与默认相同）
```

VRAM 总共只有 **27.65 GiB**（`total_room`），其中 fixed 6.06 + state 0.07 +
runtime_elastic 11.05 已经占掉，专家最多拿 ~21.7 GiB。
**123 GiB 专家里最多 18% 能同时驻留** —— 这是硬件上限，不是配置能调的。

## 结论：6.5 t/s 的成因

每 token 约 240 次专家访问（6 专家 × 40 层），其中约 82% 必须从 host RAM 经 PCIe 搬到 VRAM。
175.7 ms/token 里绝大部分是这次搬运的等待，不是计算，也不是 dispatch。

| 已排除 | 证据 |
|---|---|
| 专家缓存太小 | 多给 10 GiB → 无改善；显式 22GiB → 无改善 |
| 提交批次太碎 | 512 vs 64 → 噪声内 |
| dispatch 数量太多 | 每 forward 仅 ~69 次，设备时间 0.2–4.8 ms |
| 图构建开销 | setup 仅 0.073 ms/token |

---

## 批量预填充：实测证明**现在不能开**（2026-11-08 00:43）

上游 `batched_prefill_ok` 里有 `!is_dsv4_family(c)`。注释给的理由是「`hc_mult` 宽残差无法跨 span 携带，
且从未有 batch>1 的 V4 图被执行过」。我一度认为这个门是**保守的**（因为批量预填充是整图 `span=None`，
不触发那个 assert），于是用 `INFR_DSV41_BATCHED_PREFILL` 临时把两道门都打开实测。

### 结果：短 prompt 能跑，长 prompt **丢设备**

| prompt | 模式 | 结果 |
|---|---|---|
| 41 tok | 逐 token | 7 t/s ✅ |
| 41 tok | 批量 | **8 t/s** ✅（略快，但样本太小） |
| **2032 tok** | 逐 token | 6 t/s（344.8s）✅ |
| **2032 tok** | **批量** | ❌ **`The logical device has been lost`** |

批量路径跑了一个 3072 行的 chunk、约 5 分钟后设备丢失：

```
WARN infr_vulkan::adapter: Prefill producer cleanup after forward failure also failed
  cleanup=backend: async Prefill upload dependency failed: backend: wait segment fence:
  The logical device has been lost.
WARN infr_vulkan::pcache: [infr] the GPU device was lost during this run
  — discarding the pipeline cache rather than persisting a shader binary that may be what hung it.
```

### 驱动状态：**没有 wedge**

- 内核日志无 `xe ... wedged`、无 `guc_exec_queue_timedout`
- `infr devices` 仍然枚举到 Vulkan1 = Arc B70
- 逐 token 路径立刻恢复正常：`decode 24 tok @ 6.4 t/s`

所以这是**批量 V4.1 图本身的可复现挂起**，不是显卡问题，也不是把卡弄坏了。

### 结论：门关着是对的

上游那句「no batch > 1 V4 graph has ever been EXECUTED」不是保守措辞，是**陈述事实** ——
而第一次真去执行它，就挂了。启用开关的代码已**撤回**，只留下这条记录和 `runner.rs` 里
把实测结果写进注释的说明。

**这是本轮最有价值的否定结论**：省掉了后来人重踩一次的时间，也说明「少预填充一个 token」
并不是与 poorseek 差距 150x 的原因（41-token prompt 上批量只带来 7→8 t/s）。

---

## 专家 MMQ 路径：逐 tensor 化也救不了这个文件（2026-11-08 00:53）

perf-impl 建议把 `moe_batched_ok` 从「全有或全无」改成逐 tensor 判定，以便让已有 MMQ shader 的层跑起来。
**实测这个建议在真实 V4.1 文件上收益近乎为零**，原因是我自己核对 dtype 时发现的：

### 真实每层专家 dtype（40 层全扫）

| 位置 | IQ1_M | IQ2_XXS | IQ2_XS |
|---|---:|---:|---:|
| `ffn_gate_exps` / `ffn_up_exps` | 16 | **17** | 7 |
| `ffn_down_exps` | **28** | 8 | 4 |

### `MOE_MMQ_DTYPES`（`crates/infr-core/src/tensor.rs:172`）实际收录

`Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2K Q3K Q4K Q5K Q6K Iq4Nl Iq4Xs Iq2S Iq2Xs Iq3S Iq3Xxs Mxfp4 Nvfp4 Q2_0`

**`Iq1M` 不在里面 —— `Iq2Xxs` 也不在里面**（只有 `Iq2S` 和 `Iq2Xs`）。

### 于是

- `down` **40/40 层都不合格**（28 IQ1_M + 8 IQ2_XXS + 4 IQ2_XS，一个都没进名单）
- `gate/up` 只有 **7 层**的 IQ2_XS 合格
- shader 侧也印证：`shaders/` 里有 `native_gemm_mmq_iq2_s` 和 `native_gemm_mmq_iq_xs`，
  **没有 `iq2_xxs`、没有 `iq1_m`**，且 `native_gemm_mmq_iq_xs.comp` 头注释写明它处理的是
  `IQ2_XS [f16 d][u16 qs[32]][u8 scales[8]] = 74 bytes / 256 values`

**结论：把 gate 改成逐 tensor，最多让 7/40 层的 gate+up 走 MMQ，而每一层的 down 仍必须走 id-GEMV。
这类模型的速度上限卡在 IQ1_M/IQ2_XXS 的 down 投影上，不在判据形状上。**
要真正解决，需要写一个 **IQ1_M MMQ kernel**，而这需要对每个 8 元素子块维护**两路**激活和
（按符号位分开），不能复用 `MOE_MMQ_SACT_DTYPES` 的单标量方案 —— 这是一个独立的、有实质工作量的项目。

所以本轮**不动** `moe_batched_ok`：改判据形状换不来速度，只会把一处「保守但正确」变成
「复杂但不快」。这个结论已写进 `runner.rs` 的注释。

---

## Round 2：写了 IQ1_M 的 MMQ kernel，但它现在调用不到（2026-11-08 02:10）

### 做了什么

`crates/infr-vulkan/shaders/native_gemm_mmq_iq1_m.comp`（新，~290 行）+ 4 个变体接线。

**为什么需要它**：V4.1 的 `ffn_down_exps` 有 **28/40 层是 IQ1_M**，而 IQ1_M 被排除在 `MOE_MMQ_DTYPES`
之外，因为它的 min 项最难缠——**每个 8 元素子块都有自己的带符号 ±0.125 delta**，
`Σ(grid+delta)·x` 无法像 Q4_K/Q5_K 那样塌缩成 `scale · sact`。

**解法**（照 `native_gemm_mmq_q2_k` 的先例再细一档）：每个 32 元素激活块保留**两个** dp4a 累加器
（l=0,1 配 `dl1`，l=2,3 配 `dl2`，缩放不同所以不能先合并），外加**四个**按 l 自算的激活部分和：

```
Σ y_j x_j = dl1·(D_01 + d0·S_0 + d1·S_1) + dl2·(D_23 + d2·S_2 + d3·S_3)
```

`S_l` 直接从已 staged 的 `As` 码算出，**不需要 `sact` 缓冲，不额外读全局**。

### 验证到什么程度

| 项 | 状态 |
|---|---|
| shader 编译（4 变体 xp/xp32/xpg/xpg32） | ✅ 通过 |
| 索引推导 vs `dequant.rs` 的 CPU 参考 | ✅ 逐项一致（`qh0/qh1`、`qs[4*ib+l]`、`l=p>>1`、高低 32 位） |
| 缩放链路 vs `q2_k` 的先例 | ✅ 一致（`Bdl=d·sc`，`dval` 单独乘） |
| 真实模型输出 | ✅ 仍正确（"Answer: Paris"），无崩坏 |
| **GPU↔CPU parity 测试** | ❌ **未做** |
| **速度提升** | ❌ **没有**（6.7 vs 6.5 t/s，噪声内） |

### 为什么没有速度提升 —— 这是本轮最重要的发现

profile 显示跑的仍然是 `native_idm_iq1m_paged`，**新 kernel 一次都没被调用**。

原因：`moe_batched_ok` 是**全有或全无**判据（每一层的 gate/up/down 三项都必须合格）。
现在 IQ1_M 合格了，但 **`IQ2_XXS` 仍不合格**（没有 shader、不在 `MOE_MMQ_DTYPES` 里），
而 `ffn_gate_exps`/`ffn_up_exps` 里各**有 17 层是 IQ2_XXS**。所以整条判据依然 0/40 通过。

**更要命的是**：`moe_batched_ok` 唯一的下游是 `batched_prefill_ok`，而_V4.1 的批量预填充在 Round 1
已实测会被关掉（开长 prompt 会丢设备）_。所以**就算把判据改成逐 tensor，V4.1 还是走不到 MMQ 路径**。

这条 kernel 目前是**为未来准备好的资产**，不是本轮的提速。

### 下一步（按依赖顺序）

1. `IQ2_XXS` 的 MMQ kernel（17 层 gate/up 卡在这里；格式与 IQ2_XS 同族，有现成 shader 可参照）
2. 把 `moe_batched_ok` 改成逐 tensor 判定
3. **修好批量预填充的 device-lost**（`hc_mult` 宽残差跨 chunk 携带）—— 这是 1、2 之后收益能否兑现的前提
4. 给 IQ1_M kernel 补一个 GPU↔CPU parity 测试（照 `moe_mmq_fp4_parity.rs`）

---

## Round 3：又排除三个假设，仍未找到正解（2026-11-08 02:52）

### 目标：定位 179 ms/token 里那 ~140 ms 的等待

已知：一次 forward 只有 **69 次 dispatch、约 40 ms 设备时间**（40 层 × ~1 ms），
但 `exec(record+submit+gpu) = 179–205 ms/token`。设备空闲约 78%。

### 本轮排除的

| 假设 | 实验 | 结果 |
|---|---|---|
| **专家 cache 命中率低** | `--ctx 4096`（专家 VRAM 21.66 GiB） | hit 0.649 → **0.740**（+14%），decode **6.4 不变** ❌ |
| **decode 回放磁带开销** | `no_replay=true` vs `false` | 6.9 vs **7.0**，噪声内 ❌ |
| **冷启动/预热** | 8 tok vs 48 tok | 6.5 vs **6.7**，稳态一致，非冷启动 ❌ |

### 一个被我推翻的中间结论

我先从 pager 统计算出「每 token 要搬 3.556 GiB」，反推需要 19.86 GB/s——
**比 PCIe 实测 9.5 GB/s 还快**，物理上不成立。

查证后：专家 pool 是 **`DEVICE_LOCAL|HOST_VISIBLE` 的 ReBAR 映射内存**，
miss 只是换 slot 索引，**不是每次全量 PCIe 拷贝**（`native_weight_addr.glsl` 用设备地址直读）。
所以我那个"带宽瓶颈"的推论是错的，已撤回。

### 当前状态：未定位

已排除：专家命中率、回放磁带、冷启动、submit 批次大小（Round 1）、
dispatch 数量（Round 2 撤回）、图构建开销（0.073 ms）。

**剩下最可能的方向**（都未验证）：
- 每层之间的 fence / 同步等待（40 层 × 4.5 ms/层，而单次 submit 只要 3–10 µs，差 2 个数量级）
- `GPU-resident paged Decode controller` 的 16 层分片提交策略
- host 侧逐层准备激活（`quant_q8` 等）与 GPU 串行

**下一步该做的不是继续猜参数，而是给 decode 加逐层计时**——
在 `submit` 前后打点，直接看 40 层里哪一层/哪几层吃掉了那 140 ms。
这需要改代码（新增 prof 打点），但比继续盲试开关有价值得多。

---

## Round 3 后半：瓶颈找到了 —— `paging_sync` 占 80%（2026-11-08 03:14）

### 关键：诊断数据一直存在，只是被挡在门外

`profile_single = c.qwen4exp && pager_profile::active()`（runner.rs:12305）——
**只有 qwen4exp 家族能开这套逐层计时**，DSv4.1 永远进不去。
放开这个 `c.qwen4exp` 后，`--set prof.pager_profile=true` 立刻给出了完整分解。

（这个改动是安全的：`pager_profile::active()` 才是真正的开关，不设就零开销。
`profile_single` 下面读的全是 pager 自己的计数器与设备时间线，与架构无关。）

### 实测（6 个 decode token）

```
[single-decode-summary] tokens=6 wall=931.855ms setup=1.490ms exec=929.413ms
[single-decode-pager]   hits=3337 misses=983 evictions=983 dma=2697.715MiB dma_submits=177
  submits=599 submit_cpu=13.140ms record=138.970ms sync=747.792ms
  paging_calls=240 paging_sync=743.521ms
[single-decode-device]  main_busy=623.663ms dma_busy=0.000ms
```

| 项 | 时间 | 占 wall |
|---|---:|---:|
| **paging_sync** | **743.5 ms** | **79.8%** |
| record | 139.0 ms | 14.9% |
| submit_cpu | 13.1 ms | 1.4% |
| backend_setup | 5.5 ms | 0.6% |

逐层看（每层 6 次调用，40 层 → **240 次 paging 调用/token**）：
```
layer=1 calls=6 wall=28.825ms paging_sync=27.122ms record=3.453ms
layer=2 calls=6 wall=27.630ms paging_sync=25.616ms record=3.896ms
```
**每层 wall 的 92% 是 paging_sync，record 只占 12%。**

### 根因（已修正到准确的那一处）

我最初怀疑的是 `Pager::ensure_resident` 的 `copy_into_slot(...)` —— 那确实是 `vk.one_shot`，
**但它只是 parity 测试用的接口**（pager.rs:27 的模块注释写明）。

生产路径上真正记这笔账的是 **`crates/infr-vulkan/src/adapter.rs:17890`**，
在 `PagedStream` 的段收尾函数里，`record_paging_sync_wait(elapsed)` 包住的是**整段同步**：

```rust
pending.wait()?;                 // 等 GPU 完成
...
ps.drain()?;                     // 释放已 signal 的 fence 对象
ps.wait_cpu_promotion(be_)?;     // 等 CPU promotion
clear_frozen_moe_lut_slots(...)?;
ps.acquire_cpu_promotion(&fresh);
```

**每个 segment 都把整条队列排干**，然后才开下一个 recorder。40 层，每层多次。

### 已验证：`fence_wait` 不是那个开关

`decode_gpu_resident_fence_wait=false` 只跳过第一段 `pending.wait()`，
后面 `drain()` + `wait_cpu_promotion()` 仍然无条件执行。
实测：**6.4 t/s，与默认完全相同**，`paging_sync` 仍是每层 ~69 ms（wall 76.5 ms 的 91%）。

所以真正的解不是关开关，而是**让这些 segment 不必排干**：
- `drain()` 是为了尽快释放 transient 对象，可以延后
- `wait_cpu_promotion()` 理论上能与后续层的计算重叠
这是架构级改动，需要小心处理 fence 生命周期与 LUT 冻结窗口。

### 这解释了此前所有的"排除"

- 命中率无关：miss 数只影响**次数**，每次都同步等，所以提高命中率 14% 只是少了 14% 的等待——被噪声淹没（实测 6.4 不变）
- 带宽无关：不是传不动，是**每次都阻塞**
- dispatch 数无关：69 次 dispatch 是真的少
- 回放磁带无关：它省的是 record 时间（139ms，只占 15%）

### 下一步（明确、可验证）

把 miss 的拷贝从**逐个同步 one-shot** 改成**批量异步**：
1. 一次 collect 整批需要换入的 ids
2. 用已有的 `BackgroundTransferExecutor`（`pager.rs` 里就有 `execute_on_background`）一次提交
3. 在 batch 末尾统一等一次

这正是 poorseek 那边 `DS4_PROMPT_SYNC` 教训的反面——**它每次都等，你也要每次都等**；
把"240 次等待"变成"1 次等待"应该能拿到数量级收益。

---

## Round 4：sync 的结构性成因查清了，以及为什么调参都没用（2026-11-08 03:50）

### 复现（两份独立运行，数字稳定）

```
run A: paging_sync=743.5ms record=138.97ms submit_cpu=13.14ms  main_busy=623.7ms
run B: paging_sync=737.3ms record=141.07ms submit_cpu=12.79ms  main_busy=622.0ms
```

`main_busy` 621–623 ms 说明 **sync 期间 GPU 并不空闲**——它在处理前面那些层的计算。
所以这**不是"CPU 空等一个死 GPU"**，而是 **CPU 与 GPU 被同步点强制交替，无法流水**。

### 结构性成因（`adapter.rs` ~19676）

每个 MoE 层有两条路：

| 路径 | 条件 | 代价 |
|---|---|---|
| `inline_ok` | **该层 384 个专家全部常驻** | `touch_all_hits`，**零 sync** |
| 否则 | 任意一个专家缺失 | `sync_stream()` + 读回 ids + 逐个 stage |

而单层专家 **3.08 GiB**，VRAM 专家缓存只有 21.66 GiB ——
即使排满也只够 **7 层**，剩下 **33 层必然走 sync 分支**。

**这解释了为什么提高命中率没用**：命中率从 0.65 到 0.74 只是让"每层有 miss"变成"每层少几个 miss"，
而判据是 `all_resident` —— **只要不是全量常驻，就一次不少地 sync**。

### `submit_layers` 也不是杠杆（实测）

| submit_layers | decode |
|---|---|
| 0 (auto=16) | 6.4 t/s |
| 8 | 6.5 t/s |
| 16 | 6.4 t/s |
| **40**（整个模型一次提交） | **6.5 t/s** |

即使把整个 40 层合成一次提交，速度不变 —— 因为 sync 发生在**层内**，与提交批次无关。

### 收益上限（值得记住）

```
当前 155.3 ms/token = 6.4 t/s
paging_sync 123.9 ms/token
若完全消除 -> 31.4 ms/token = 31.8 t/s
```

**31.8 t/s 略高于 poorseek 的 27.7。** 靶子是对的：
把 paging_sync 真正消掉，就能追上甚至超过参照值。

### 还没做的（下一轮）

真正的解是让 **sync 不再是每层必需**。三条可能的路径，都未尝试：

1. **让 routed 专家集合在层内可预知**：若能在 GPU 上完成 routing 并直接 stage，
   就不需要"读回 ids → CPU 决定 → 再上传"这个往返，sync 可省。
   代码里已有 `hit_first` / GPU routing 的基础设施（日志 `hit_first=true`）。
2. **把 sync 的成本与下一层的计算重叠**：`main_busy` 已 622ms，GPU 是忙的；
   让 CPU 的 stage 工作排到 GPU 队列里，而不是阻塞等。
3. **降低单层专家占用使 `inline_ok` 可达**：需要 3.08 GiB/层全驻留，当前不可行，
   除非提高专家压缩率或改变驻留粒度 —— 属路线级决策。

这三条都不是调参能解决的，需要改 `adapter.rs` 的层内流水线。

---

## Round 5：账目终于平了，并且推翻了我上轮两个假设（2026-11-08 04:06）

### 修正 1：`main_busy` 不能当作"GPU 计算 floor"

上一轮我写「GPU 每 token 真算 103.7 ms，理论 floor 9.6 t/s」，**这个推论是错的**。
`main_busy` 是主队列设备区间的**并集**，它与 `paging_sync` 大幅重叠（622 vs 737 ms），
不是可以相加的独立项。

### 修正 2：`argmax_part` 的 85.9% 是**单次 forward 内**的占比，不是全程

全局累计才 124.2 ms / 40 次调用。拿它去比 `main_busy` 的 622 ms 得出"占 20%"，同样是跨口径比较。

### 账目现在平了（同一次运行、同一区间）

```
40 层 wall 合计        814.2 ms
整个 decode wall        931.9 ms
差额（非层开销）        117.7 ms   <- argmax / 采样 / 收尾
```

每 token：层内 **135.7 ms**，非层 **19.6 ms**。**argmax 只占约 13%，不是大头。**

### 本轮确认的两件事

1. **`argmax_part` 单次 3081 μs 扫 505 KiB = 0.17 GB/s**，比应有带宽慢约 3 个数量级，确实是烂 kernel；
   但修好它每 token 只省约 20 ms → 6.4 → 7.4 t/s。**值得修，但不解决问题。**
2. **大头仍在层内的 paging_sync**（123.9 ms/token），与 Round 3/4 的结论一致。

### 教训（写给下一轮的我）

这几轮我反复在**跨口径比较数字**上栽跟头：
- 把 `INFR_PROF_OPS` 的累计计数当单次 forward
- 把 `copy_into_slot`（测试接口）当生产路径
- 把 `main_busy` 当独立 floor
- 把单次 forward 的占比当全程占比

**规则：任何两个数字要相减/相比，必须先确认它们在同一个时间区间、同一个采样开关下产生。**
层内用 `single-decode-layer` 求和，全程用 `single-decode-summary`，不要混。

### 补：`argmax_part` 的 3 ms 也不是计算

同一次 forward 内的对比（`INFR_PROF_OPS`）：

```
native_idm_iq1m_paged   285.2 us   <- 真正的专家 GEMM（一次 256 专家的矩阵乘）
argmax_part            3436.5 us   <- 扫 129280 个 f32
```

**词表扫描比矩阵乘慢 12 倍**，不可能是计算量的问题。256 个 workgroup × 256 线程
扫 505 KiB 在 Arc B70 上应该是微秒级。

结论：这 3 ms 是**排队等待**——argmax 被排在一整层专家 GEMM 之后，
它的"设备时间戳"把等待前面的活也算了进去。和 `paging_sync` 是同一个现象的另一面。

因此**修 argmax shader 不会有收益**（它本来就不慢），这条线索到此为止。
真正的账仍然是：40 层 wall 814 ms，其中 paging_sync 约占 80%。

---

## Round 6：decode 的天花板与地板都量出来了，方向需要重新评估（2026-11-08 04:33）

### 本源代码结构（此前没看全的部分）

`adapter.rs:19064`：

```rust
let touch_all   = rows * n_used >= 3 * n_expert;      // 整层常驻
let layer_stream = touch_all && cfg.paging.moe_layer_stream;  // 整层一次性 staging
```

`paging.moe_layer_stream` **默认 true**（config/mod.rs:242），所以瓶颈不在这个开关，
而在 `touch_all`：decode 时 `rows=1`、`n_used≈8`、`n_expert=384`，
`8 >= 1152` 永远不成立 → **decode 天然用不上整层 streaming，只能逐专家 LRU + 每层 sync**。

这解释了为什么 Round 4 的 `submit_layers` 四档（0/8/16/40）完全无差别。

### 专家 GEMM 的真实速度（关键新数据）

单次 `native_idm_*` 的**真速度是 6–27 μs**（看 forward 尾部那些已经被喂饱的层）：

```
native_idm_iq2xs_paged   27.8 us   6.4 us   131.4 us
native_idm_iq1m_paged     6.7 us   6.5 us     6.4 us
```

汇总里 133 μs 的平均值**是被排队等待拉高的**，不是 kernel 慢。

按真速度折算：40 层 × 3 次 × 6–50 μs = **0.7–6.0 ms/token**，
而一个 decode token 实测 **155.3 ms**。**GEMM 连 5% 都不到。**

### 诚实的结论

到此为止，六轮测量把 decode 的时间**从头到尾翻了一遍**：

| 项 | 量级 | 是否瓶颈 |
|---|---|---|
| 专家 GEMM 本身 | 0.7–6 ms/token | ❌ |
| argmax / 采样 | ~20 ms/token | ❌ |
| 图构建 | 0.07 ms/token | ❌ |
| dispatch 数量 | 69 次/forward | ❌ |
| 专家命中率 | 0.65→0.74 无变化 | ❌ |
| 回放磁带 | 无变化 | ❌ |
| submit 批次 | 四档无变化 | ❌ |
| **paging_sync（每层一次）** | **124 ms/token** | ✅ **唯一剩下的** |

而这些 sync 的结构性成因（`touch_all` 不可达 + 每层必须有 miss）是**设计使然，不是 bug**。
在 27.6 GiB 显存跑 123 GiB 专家这个前提下，它可能是**固有的**。

### 下一轮该问的问题（而不是继续试参数）

1. **poorseek 是不是也在做同样的事？** 它跑的是 "V4.1-XXS"——**XXS 是更小的模型**，
   参数量可能小一个数量级，专家能**全部常驻**，于是完全不触发这条 paged 路径。
   如果是这样，**这个对比从一开始就不是同类比较**，需要先确认 poorseek 的模型规格与显存占用。
2. 若模型确实同级：需要在 `adapter.rs` 里做出**跨层 pipeline**
   （层 n 的 sync 与层 n+1 的计算重叠），这是真正的架构改动，不是开关。

---

## Round 7：27.7 t/s 的出处找到了，而且它不是同类比较（2026-11-08 04:51）

上一轮我提的问题（"poorseek 的 V4.1-XXS 是不是更小"）**答案是反的**，
但顺着它找到了真正的原因。原始文档：`poorseek/xolotl-sycl/docs/DEEPSEEK_V41.md:121`。

### 那个 27.7 的原始上下文

> **Results (RTX 4090 24 GB · Ryzen 7 5700X · 90 GB DDR4 · NVMe, 2026-10-12)**
>
> | Decode (256-token prompt, 256 tokens) | 3.4 tok/s | **25.5-26.3 tok/s**
> (27.7 with the opt-in ggml patch) |
>
> * **Decode routes with two selection-only biases** (`--route-bias 1.0`,
>   `--route-bias-file 0.6`, both defaults): a bonus for experts already in VRAM and a
>   penalty for experts that would be read from NVMe. ...
>   `--route-bias 0 --route-bias-file 0` = **exact routing at ~3.5 tok/s**.
> * Cost on 4096-token perplexity: code **2.38 → 2.87**.

**四点关键差异，逐条对照：**

| | poorseek 的 27.7 | 本机 |
|---|---|---|
| 硬件 | RTX 4090 24 GB + 90 GB DDR4 + NVMe | Arc Pro B70 31.9 GiB + 系统内存 |
| 模型 | GSQ 自研量化（PTQ1_0 三进制 1.75 bpw / Q2_0 2.25 bpw），**246 GiB 分片** | IQ2_XXS `-00001-of-00004`，**129 GiB** |
| **路由** | **被 route-bias 主动偏置**（偏好已在 VRAM 的专家、惩罚要读 NVMe 的专家）——**不是精确路由** | **精确路由** |
| 27.7 的附加条件 | 还要开 opt-in ggml patch；不开是 25.5-26.3 | — |

### 最重要的一条

**同一个文档写明：关掉 route-bias（精确路由）只有 ~3.5 tok/s。**

也就是说 25.5-27.7 这个数，是**用路由准确性换来的**，代价是 4096-token code ppl 从 2.38 劣化到 2.87。

**按精确路由口径比：本机 6.4-6.5 t/s vs poorseek 3.5 t/s —— 我们是它的 1.8 倍。**
之前"差 4 倍"的判断，建立在拿我们的精确路由去比人家的偏置路由上。

### 但也不能就此宣称"我们赢了"

它的 decode 分解（38 ms/token）是 **dense ~19 ms + expert tier ~16 ms**；
本机是 **层内 135 ms，其中 paging_sync 124 ms**。

**它的 expert tier 16 ms，我们的 expert 路径 124 ms —— 这一项我们慢了近 8 倍，是真实的差距。**
差别在于：它有 90 GB 系统内存 + NVMe 做专家后备，且用 route-bias 把命中率推到极高；
本机 21.66 GiB 的 ReBAR arena 在 123 GiB 专家面前命中率只有 0.65-0.74。

所以正确的结论是：

1. **"对标 27.7" 这个目标本身口径不对**——应该在"同为精确路由"下比（我们已领先），
   或者在"允许 route-bias"下比（那要先实现 route-bias，并且接受 ppl 劣化）。
2. **真正值得做的优化仍然是 expert 搬运**（124 ms vs 16 ms），
   但不是靠改 `paging_sync` 的同步结构——而是**提高命中率本身**，
   这需要一个显存感知的路由偏置（即 route-bias 的等价物），
   或者更大的 expert 后备带宽。
3. 本轮的 `Vulkan0`（核显 141 GiB UMA）路线**已实测不可行**：
   加载时 97.24 GiB 已被物理占用，只剩 43.09 GiB，装不下 129 GiB 模型，
   且这次尝试**触发了一次 device lost**。

---

## Round 8：预填充实测 5 tok/s —— 这才是最大的差距（2026-11-08 05:32）

### 首次实测 V4.1 的长上下文预填充

```
prefill 10160 tok @ 5 tok/s (1980142 ms = 33 分钟) | decode 2 tok @ 4.5 tok/s
VRAM plan: ctx=16384 expert_cache_target=10.03 GiB elastic_pool=21.08 GiB
```

对照 poorseek 文档（`DEEPSEEK_V41.md:118`）：

| | 本机（Arc B70 / IQ2_XXS 129 GiB / ctx 16K） | poorseek（RTX 4090 / GSQ 246 GiB） |
|---|---:|---:|
| **预填充 8192 tok** | **~5 tok/s** | **244-248 tok/s** |
| 差距 | — | **~49×** |
| decode | 6.4-6.5 tok/s | 25.5-26.3（偏置路由）/ ~3.5（精确路由） |

**预填充的差距（49×）比 decode 大一个数量级。**

### 这正好命中一个已知开关

Round 1 就查明：`batched_prefill_ok` 里有一项 `!is_dsv4_family(c)`（runner.rs:12253），
让 V4 家族**预填充一次只处理一个 token** —— 一整个 10160-token 的 prompt 被拆成
10160 次 forward。

而预填充本该走 `touch_all` 路径（`rows * n_used >= 3 * n_expert` 在大 rows 下成立），
一次性把整层专家搬进来，然后**批量**计算。现在这个优势完全没有被利用。

### Round 1 试过打开，但会丢设备

当时实测：短 prompt 能跑（41 tok 8 t/s vs 7），但 2032-token 会
`The logical device has been lost`（3072 行 chunk 跑约 5 分钟）。
注释里写的理由是 **`hc_mult` 宽残差无法跨 span 携带**。

### 这是下一轮唯一值得做的事

理由：
1. **收益最大** —— 49× 的差距，不是 decode 那种 8×
2. **方向明确** —— 不是猜，是已知的一个布尔条件
3. **不涉及路由/质量妥协** —— 纯粹的工程修复，不影响精度

需要解决的具体问题：让 **`hc_mult` 宽的残差能跨 chunk 携带**。
这是 Round 1 撞墙的那堵墙，也是 `batched_prefill_ok` 里那句
「no batch > 1 V4 graph has ever been executed」的含义。

---

## Round 9：批量预填充真正打通，预填充 6 → 12 t/s（2026-11-08 06:32）

### 根因：那个开关从来没被实现过

`batched_prefill_ok` 的注释从 Round 1 起就写着
「V4.1 is behind `INFR_DSV41_BATCHED_PREFILL` while that path is being brought up」。

**但那个字符串只存在于注释里。** 判据是无条件的 `... && !is_dsv4_family(c)`，
没有任何读环境变量的代码。所以此前每一次"设置这个变量去测批量路径"，
实际跑的都是 per-token 路径。

**破绽在数字上**：变量设上去后，ubatch 32/64/128/256 **全部报 6-7 t/s**。
批量预填充必须随 ubatch 变化；一个不随 ubatch 变化的速率，按定义就是 per-token。

### 接通后

```rust
let dsv41_batched_prefill = is_dsv4_family(c)
    && std::env::var_os("INFR_DSV41_BATCHED_PREFILL").is_some_and(|v| v != "0");
((c.moe.is_none() || moe_batched_ok) && !is_dsv4_family(c)) || dsv41_batched_prefill
```

实测（731-token prompt，ctx 4096，Vulkan1）：

| 配置 | t/s | 耗时 |
|---|---:|---:|
| per-token（默认） | 6 | 117 s |
| batched ub=64 | **10** | 73 s |
| batched ub=256 | **12** | 62 s |
| batched ub=512 | **12** | 63 s |
| batched ub=1024 | **12** | 63 s |

**3131-token prompt @ ub=512：12 t/s（270 s），无设备丢失。**

- **预填充 6 → 12 t/s，2.0×**
- ubatch ≥ 64 后饱和（chunk 内专家复用已达上限，再加行数没有额外收益）
- **Round 1 记录的 2032-token 丢设备没有再复现**（3131 token 稳定跑完）

### 仍然存在的差距

poorseek 8192 token 是 244-248 t/s，我们 12 t/s，**仍差约 20×**。
但那条差距的性质已经不同了：
- 它用 **int8 tensor core MMQ** 计算，我们用 **dp4a id-GEMV**
- 它的专家是 **PTQ1_0 三进制 / Q2_0**，我们是 IQ2_XXS

这不再是"路径没走通"，而是"走的路径本身算力利用不同"，属于另一个量级的工程。

### 下一步

1. 把 `ubatch` 的自动策略在 V4.1 上设到 ≥256（现在默认走到 32）——**一行改动，可能再拿 2×**
2. Round 2 写的 `native_gemm_mmq_iq1_m` kernel 现在**有意义了**：批量预填充一旦走通，
   `moe_batched_ok` 相关的 MMQ 路径才是真正的算力出口

---

## Round 10：批量预填充已经在跑 MMQ，zero_fill 是合理取舍（2026-11-08 07:27）

### 好消息：分组专家路径已经通了

`INFR_DSV41_BATCHED_PREFILL=1` 下的 kernel 名单里已经有：

```
native_gemm_mmq_q5k / mmq_q8_0 / mmq_q6k_streamed     <- MMQ 系列
moe_bucket_scatter / moe_bucket_scan / moe_bucket_count  <- 分组散布
expert_gateup / expert_down / moe_scatter_reduce
```

也就是说 Round 9 打通的批量路径**走的是按专家分组的 MMQ 管线**，
不是回落到 per-token 的 id-GEMV。这是 Round 2 写的 `native_gemm_mmq_iq1_m` 终于被用上的前提。

### `zero_fill` 占 51%，但它是合理的

聚合 profile 里最大的一项是：

```
   31.3s   51.43%   169 次   185.5 ms/次   zero_fill
```

单次 185 ms 看着惊人，但算一下量：

```
physical_slots = rows × (n_used + shared) = 512 × 9 = 4608 槽
gbuf = slots × gu_width × 4 ≈ 0.05–0.14 GiB
ubuf = slots × nff × 4      ≈ 0.03–0.07 GiB
单次合计 ≈ 0.08–0.21 GiB
```

**0.1–0.2 GiB 的清零在 B70 上不该花 185 ms**，所以这个数同样是**排队等待**
（和 Round 3–5 里 `argmax_part`、`main_busy` 的坑一模一样）。

### 这个清零能不能省？查过了，不能

`rec2.zero(gbuf, physical_slots * gu_width)` 存在的理由是：
`silu_mul` 会把 `gate` 和 `up` **每个槽**都算一遍，未路由的槽必须读成 0。
唯一的豁免是 `narrow_hit_activation`，它改用 `silu_mul_masked(..., mask: u32, ...)`
——**但那个 mask 是 `u32`，最多 32 个槽**，而 512 行的 `physical_slots` 是 4608。

可以逐行套 mask（每行 9 槽），但那要 **512 次小 dispatch**，
大概率比一次大 `zero_fill` 更慢，且要重写热点 kernel。

**结论：这是当前 kernel 结构下的合理取舍，不值得改。** 记录在此以免下次又翻到。

### 本轮的净产出

- 确认批量预填充走的是 MMQ 分组管线（不是回落）
- 关掉 `zero_fill` 这条线索，并留下"为什么不能关"的理由
- 没有改动任何代码——**没有找到比现状更好的方案时不动代码，也是一种结果**

---

## Round 11：预填充的时间账本 —— 72% 花在清零和激活上（2026-11-08 07:55）

### 完整 op 排序（3131-token prompt 里取一次运行的聚合）

```
   31.3s   51.58%   169 次   185.5 ms/次   zero_fill
   12.8s   21.03%   262 次    48.8 ms/次   silu_mul
  320.1ms   0.53%   240 次     1.3 ms/次   hyper_post
  308.4ms   0.51%   120 次     2.6 ms/次   moe_topk_sg
  226.4ms   0.37%   120 次     1.9 ms/次   rmsnorm_nw
  110.4ms   0.18%  1242 次    88.9 us/次   quant_q8
   60.0ms   0.10%   160 次   374.8 us/次   hyper_mix_gates
   37.4ms   0.06%     4 次     9.4 ms/次   expert_gateup      <- 专家 GEMM
   32.0ms   0.05%   906 次    35.4 us/次   native_gemm_mmq_q5k
```

**`zero_fill` + `silu_mul` = 44.1 s = 72.6%。**
**所有专家 GEMM 加起来不到 0.2 s。**

这和 Round 6 在 decode 上得到的结论是同一个形状：**矩阵乘不是瓶颈，路径上的簿记才是。**

### 这两个开销跟 chunk 大小无关（实测）

| ubatch | 3131-token 预填充 |
|---|---|
| 128 | 11 t/s (296 s) |
| 512 | 12 t/s (270 s) |

**只差 9%。** 因为 `zero_fill`/`silu_mul` 的成本是
`O(rows × physical_slots_per_row × nff)`，**总量只与总行数成正比**，
chunk 切多大都要把每一行的 9 个槽清一遍、激活一遍。

这也解释了 Round 9 的曲线为什么在 ub=64 之后就走平了：
chunk 变大只减少了**每 chunk 一次**的固定开销（topk、scatter 之类），
减不掉那 72% 的按行计费部分。

### 因此

- 预填充的 12 t/s 天花板**不在 ubatch**，已用 128/512 两点验证
- 也不在 GEMM（<0.2 s）
- 而在 `物理槽数 × 每行槽数` 的这个乘积上

`physical_slots = rows × (n_used + shared) = rows × 9`。这个 9 就是
`top_k + shared_expert`。**要降这个税，只有两条路**：
减小 chunk（无效，已证明）或**在更少的物理槽上做激活**，
即让激活与 `moe_accumulate` 融合、或按实际路由数而不是 9 个固定槽来遍历。
后者需要改 kernel 接口（现在的 `silu_mul` 是稠密的），属于实质性工程，
不是这轮能安全完成的。

**本轮不改代码，只把账本记清楚**：下一轮若有人想优化预填充，
应该直接看 `silu_mul` 能否改成按 mask/按实际路由遍历，
而不是再调 ubatch（已证明无收益）。

---

## Round 12：把量算准了 —— 40 GiB 的活花了 44 秒，效率 0.1%（2026-11-08 08:07）

### 先拿到真实模型参数（此前一直靠估）

直接读 GGUF 元数据（`deepseek41.*`），不再猜：

```
expert_count              = 384
expert_used_count         = 6      <- 我此前一直按 8 算
expert_shared_count       = 1
expert_feed_forward_length= 2304   <- 此前按 1536-4096 估
```

### 于是每笔开销可以精确算

```
physical_slots = rows × (n_used + shared) = 512 × 7 = 3584
gbuf = 3584 × 4608 × 4 = 0.062 GiB
ubuf = 3584 × 2304 × 4 = 0.031 GiB
单次 zero_fill = 0.092 GiB     -> 理论 0.22 ms @450GB/s   实测 185.5 ms
单次 silu_mul  = 0.092 GiB×3   -> 理论 0.22 ms            实测  48.8 ms
```

**实测效率 0.1%。**

全量：
```
169 次 zero_fill × 0.092 GiB = 15.6 GiB
262 次 silu_mul  × 0.276 GiB = 24.2 GiB   (读 g + 读 u + 写 y)
合计 ≈ 40 GiB
```
**在 450 GB/s 的卡上搬 40 GiB 只要 0.09 秒，实际花了 44 秒。**

### 所以这不是带宽问题，也不是量的问题

是**每一次操作都在等前面的工作完成**——GPU 没有流水起来。

这和本会话在 decode 上查到的成因是**同一个病**：
- decode：每层一次 `paging_sync`，743 ms / 931 ms
- prefill：169+262 次小操作，44 s / 61 s

两者的共同点不是"某个 kernel 慢"，而是**大量短操作之间没有重叠**。

### 本轮我否掉了自己两个中间假设

1. 先怀疑 `zero_fill` 是 `cmd_fill_buffer`（转移操作）会排干队列 —— **错**，
   `Recorder::sync` 只是发一个标准 pipeline barrier，不排干。
2. 再怀疑 `dst_transfer=true` 的 barrier 太重 —— **错**，那是正常的
   compute→transfer 屏障写法，而且是 spec 要求（注释里写了 ANV 会拒绝错误的 mask）。

两次都靠读代码否掉了，没有写进结论。

### 结论与下一步

**不该再盯着 zero_fill/silu_mul 的"量"做优化**——它们本来就只有 40 GiB。
该做的是让这些操作**不阻塞**：把 169 次 fill 和 262 次 silu 合并进更大的提交，
或者让它们与计算重叠。

但这一层已经是引擎的提交/屏障策略，改动面很大，且任何错误都会表现为
静默的错误结果而非崩溃。**建议下一轮先做一件事**：
确认这些操作是否各自触发了 submit（用 `INFR_PROF_STAGES` 或 submit 计数），
如果 169+262 次里有大量独立 submit，那就是明确的、可验证的优化点。

### 更正：prefill 不是 paging_sync 的锅

我上一条写「两者是同一个病（paging_sync）」——**算了一下，是错的，撤回。**

```
prefill 每层: wall 6.0 ms, paging_sync 5.755 ms (96%), record 0.579 ms, submits 2
40 层 × 2 chunk(731 tok @ 512) = 80 层次
  => paging_sync 总量 ≈ 0.46 s
实测 731-token 预填充 = 62 s
```

**0.46 / 62 = 1%。** 而 `zero_fill` + `silu_mul` 是 44 s / 72%。
两者**对不上**，所以"同一个根因"不成立。

单个 chunk 内的比例（96%）之所以误导人，是因为**一个 chunk 只有 40 层次**，
而 62 秒里绝大部分不在被 `single-decode-layer` 采样的那一段。

### 那么 prefill 的时间到底在哪

`zero_fill`(31.3s) + `silu_mul`(12.8s) = 44 s 是**真实的聚合值**，
来自整次运行的 `INFR_PROF_OPS` 汇总，不是采样区间。

但按真实参数量算，这些操作总共只有 **40 GiB**，理论 0.09 秒。
**44 秒里有 43.9 秒是空的。**

这 43.9 秒既不是 paging_sync（1%），也不在 record 里，那它只能是：
`INFR_PROF_OPS` 的设备时间戳**把提交之间的等待也计入了**，
而等待发生在**这些操作之外**。

### 结论：我不知道 prefill 的 62 秒花在哪，且现有工具分辨不出来

诚实地记下来总比编一个解释好。已知：
- 不是 GEMM（<0.2 s）
- 不是 paging_sync（~0.5 s）
- 不是 zero_fill/silu_mul 的真实搬运量（0.09 s）
- 不是 ubatch（128 vs 512 只差 9%）

**下一步需要的是 wall-clock 级的逐阶段计时**（像 decode 的
`setup/build` vs `exec/record+submit+gpu` 那样），而不是设备时间戳。
现在 prefill 路径没有这个拆分。

---

## Round 13：账对上了 —— 每行每层 2100 μs 的固定簿记（2026-11-08 08:36）

### 加了 prefill 的 wall-clock 拆分

给 chunk 组循环加了 `[pf prof]` 汇总（`ec.prof.stages` 门控，默认路径零开销）。
同时发现**每个 span 本来就有一行** `[pf prof]`，只是我之前从没看过：

```
[pf prof] m=218 l=0..40 build=0.5ms compile=0.7ms execute=18420.6ms
```

`build` 0.5 ms、`compile` 0.7 ms —— **execute 占 99.99%**。
所以不是图构建、不是编译计划，是执行。

### chunk 大小扫描把性质定死了

同一 prompt、同一模型，只改 chunk 行数：

| m（行） | execute | 每层 | μs / 行 / 层 |
|---:|---:|---:|---:|
| 26 | 2,989 ms | 75 ms | 2,874 |
| 64 | 6,436 ms | 161 ms | 2,514 |
| 218 | 18,404 ms | 460 ms | 2,111 |
| 256 | 21,494 ms | 537 ms | 2,099 |

**μs/行/层 基本恒定在 2100–2900**，也就是说 **execute 时间与总行数成正比**。

### 于是 62 秒终于能解释

```
731 token × 40 层 × 约 2100 μs ≈ 61.4 s
实测 62.2 s   ✓
```

**这才是我找了六轮的那个账。** 它既不是带宽（40 GiB 只要 0.09 s），
也不是 paging_sync（约 0.5 s），也不是 ubatch（改了只挪 9%），
而是**每一行、每一层都要付约 2.1 ms 的固定簿记**。

这一层的 GEMM 是 1 行 × 4608 × 2304 的专家乘，微秒级；
2.1 ms 里几乎全部不是计算。

### 这也解释了为什么 decode 看起来"更慢"却其实是同一件事

- decode：1 行 × 40 层 × 3.9 ms/层 = 155 ms/token
- prefill：1 行 × 40 层 × **2.1 ms/层** = 84 ms/token

**prefill 每行其实比 decode 还快**（2.1 vs 3.9 ms/层）。
decode 更慢是因为它额外背了 `paging_sync`；
prefill 已经避开那条路，但撞上另一个同样是"每行每层一次"的固定成本。

### 下一步（方向明确了，但要先定位这 2.1 ms 是什么）

现在需要一个**层内**的拆分：把 2.1 ms 分成
「上层的 prep/簿记」与「实际记录+提交的 GPU 工作」。
候选（按可能性）：
1. 每层每行一次的 `zero_fill`（已在 Round 10 确认是功能必需，但 2100 μs 远超其带宽成本）
2. 每层一次的 `moe_topk_sg` / bucket scatter 等路由簿记
3. 每层一次的 submit + fence 等待

如果 2.1 ms 里主要是 3，那它和 decode 的 `paging_sync` 是**同一类病**（只是数值不同），
修法也类似；如果是 1/2，那是另一个方向。**必须测，不能再猜。**

---

## Round 14：per-op 归因不可信，62 秒仍在层内（2026-11-08 08:58）

### 先排除了一件事：不是测量开销

关掉 GPU timeline 采样做对照：

| 配置 | 731-token 预填充 |
|---|---|
| 默认 | 62,473 ms |
| `INFR_TIMELINE=0` | 62,415 ms |

**差 0.1%，噪声。** 所以 62 秒是真实的，不是被采样拖慢的。

### 但 `INFR_PROF_OPS` 的 per-op 归因不可信

`ub=256` 下的 per-op 表：

```
copy_strided  650.7 μs × 3540 次 = 2.30 s   (3.8%)
zero_fill     147.1 μs ×  213 次 = 31.3 s  (51.7%)
```

`copy_strided` 每层约 29.5 次（40 层 × 3 chunk），单次 650 μs 对一次
`rows×nw/256` 个 workgroup 的行拷贝来说是离谱的——即使 `nw=4096` 也只搬 4 MiB。

**我试过两个解释，都不成立：**

1. 「时间戳含排队，所以所有 per-op 被统一抬高」
   → **错**。若是统一缩放，`copy_strided`(650 μs/次) 和
   `zero_fill`(147 μs/次) 的单位时间应该同比例，但它们总计分别是 2.30 s 和 31.3 s，
   **单位时间更小的那个反而总量更大**。不是统一缩放。

2. 「copy 本身慢」→ 与搬运量矛盾（4 MiB / 650 μs = 6 GB/s，慢 100 倍）。

### 诚实的现状

- **62 秒是真的**（关采样不变）
- **execute 与总行数成正比是真的**（Round 13 的四点扫描，μs/行/层 恒定）
- **但 62 秒在层内怎么分配，现有工具给不出可信答案**
- 每层每行 2.1 ms，而该层 GEMM 只要微秒级 —— 这个 2.1 ms 的构成仍未知

### 下一轮必须换测量手段

`INFR_PROF_OPS` 与 `[pf prof]` 都不足以定位。需要的是一段
**CPU 侧的逐层 wall-clock**：在 chunk 循环的 `for layer` 内打点，
记录每层 CPU 侧耗时，不依赖 GPU 时间戳。这样能回答一个不会自我矛盾的问题：
「这 2.1 ms 里，CPU 在等 GPU 还是在做别的事」。

在此之前不再对 per-op 数字做归因——这一轮已经因此推错了两次。

---

## Round 15：归因工具补齐了，账能闭合（2026-11-08 09:16）

### 新工具：`[single-decode-execute]`

`[parallel-token-profile]` 一直会算 `backend_execute - gpu_union`（设备时间线没归因到的部分）
和 `backend_execute - host_accounted`（host 记账之外的部分），但**单流路径从来不打印**。
现在补上了，算法与并行路径一致。

### 首次读数（decode，731-token prompt）

```
backend_execute = 227.732 ms
gpu_union       = 130.952 ms   (58%，设备真的在忙)
gpu_gap         =  96.780 ms   (42%，设备闲着)
host_accounted  = 226.645 ms
residual        =   1.087 ms
```

**`host_accounted` 与 `backend_execute` 相差 1.1 ms —— 账闭合了，没有藏在 residual 里的东西。**

host 侧分项（来自 `single-decode-pager`）：
```
sync           176.497 ms   <- 占 host 的 89%
record          31.443 ms
backend_setup   13.680 ms
submit_cpu       2.742 ms
合计           224.362 ms   ≈ host_accounted 226.645   ✓
```

### 由此确认的 decode 结论

- GPU 只在 **58%** 的时间里忙，**42% 在空等**
- host 的时间 **89% 花在 `sync`**
- 所以 decode 的瓶颈仍是「每次 sync 都在等，而 GPU 利用率只有 58%」
  这与 Round 3–5 的结论一致，但这次是**用闭环的账证明的**，而不是靠单点数字推断

### 本轮的实质产出

前两轮我在 per-op 数字上推错了两次。这轮换成**可闭合的归因**（进出相等、residual 近零），
这是能信的。工具已落在 `[single-decode-execute]`，提交 `aaeacf1`。

### 下一步

prefill 走的是 `execute_static`，而 `profile_single` 只在 prefill **之后**才开始采样，
所以上面这组数还没覆盖 prefill 的 62 秒。下一步是把同样的
`backend_execute / gpu_union / host_accounted / residual` 在 **prefill 的 chunk 循环内**累加输出，
直接得到那 62 秒的闭环归因。

---

## Round 17：syscall 层视角 —— futex 主导，`pread64` 是加载不是推理（2026-11-08 10:04）

### Round 16 留下的靶子

`execute_static_walk` 里，59.19 s 层 wall 中 sync=0、record=1.6 s，**57.5 s 未归因**。

### 换到操作系统视角

`/usr/bin/time -v`（731-token prompt）：

```
User time   (seconds):  17.53
System time (seconds): 158.26      <-- 9 倍于 user
Percent of CPU:         136%
Elapsed (wall):         129.24 s
```

**CPU 时间几乎全在内核态。** 这不是 ALU 计算，也不是空等 GPU。

### strace -c 的归因

```
% time   seconds  usecs/call   calls  errors  syscall
 64.65  1085.1     25634      42330    7012   futex
 34.12   572.7   1188196        482          pread64
  0.90    15.1        49     302221          sched_yield
  0.12     2.1      186       11017       5  ioctl
```

**`futex` 占 64.65%，`pread64` 占 34.12%。**

### 我两次误判 `pread64`，最后算清了

1. 先猜「推理期在从磁盘读专家」——查到 `pravd64` 持续到进程退出前一刻、最后 10 秒还有 96 次，看起来像。
2. 再一算总量：**482 × 327,352,320 B = 146.9 GiB**，而模型四个分片合计 **129.1 GiB**。
   **这就是模型加载**（还有元数据等余量）。它出现在末尾是因为加载与推理异步重叠。

**所以 `pread64` 不属于推理期成本，已排除。** 这一条如果没算总量就写进结论，会是本会话第六次错误归因。

### 站得住的结论

- **`futex` 42330 次、7012 次错误** —— 这是线程同步开销，占 cpu 时间的大头
- `sched_yield` 302221 次 —— 大量让出/重试，与 futex 竞争一致
- CPU 时间主要在内核态、由**锁竞争**产生，而不是在算东西

这给了一个新的、具体的方向：**prefill 的 57.5 s 大概率花在线程间同步等待上**，
这与 Round 16 看到的「层 wall 大、但 sync/record 小」是吻合的——
时间不在 GPU 命令上，也不在 paging_sync 上，而在 **host 侧的线程协调**。

### 下一步

定位那 42330 次 futex 属于哪个组件。候选：rayon 并行（`sched_yield` 30 万次很像）、
pager 的锁、PLE worker。查到后才知道能不能降。

**注意**：本轮结论比前几轮弱——它是「指向主机侧同步」的方向性证据，
还不是「这 57.5 s 精确分解」。不要再往下推成精确结论。
