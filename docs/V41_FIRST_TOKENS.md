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

## 调度开销（profile 证据）

解码 32 token 的 per-op 计数（`INFR_PROF_OPS=1`）：

| op | count | us/次 |
|---|---:|---:|
| native_idm_iq2xxs_paged | 3004 | 49.7 |
| hyper_mix_gates_decode | 3120 | 34.8 |
| rmsnorm | 3120 | 25.4 |
| hyper_pre | 3120 | 10.8 |
| linear_f32r_v4 | 3120 | 10.8 |
| rope + rope_back | 3120+1560 | ~6 |
| dsv4_gather | 1560 | 45.5 |
| moe_topk_sg | 1560 | 25.3 |

1560 = 39 token × 40 层，即**每 token 每层一次**。合计约 **1.8 万次 dispatch / token**，
156 ms/token ÷ 1.8 万 ≈ **8.7 us/dispatch** —— 正是 Vulkan submit 的开销量级。

**所以 V4.1 慢的根因是「每 token 约 1.8 万次小 dispatch」，不是任何一个 kernel 慢。**
这与 poorseek 侧「闸门占 40%」是同一族问题：**调度，不是算力**。

## 下一步（按收益排序）

1. **减少 dispatch 数**：40 层 × 约 30 个 op 是结构性的。decode 回放磁带（`no_replay=false`，即已启用）
   已经省掉了录制，但 dispatch 数本身没变。真正的解是融合（fuse）逐层的 rmsnorm/rope/hyper。
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
