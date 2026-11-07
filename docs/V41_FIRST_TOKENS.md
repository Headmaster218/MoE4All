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
