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
