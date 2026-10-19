---
kind: benchmark
status: historical
scope: two-lane decode cohort pipelining (Vulkan layer split)
run_date: 2026-10-19
commit: eb650d7b
evidence: measured
---

# 2026-10-19 双 lane 跨步流水（plan0/plan1 双缓冲，2 次调用调度）

**一句话结论**：32K 输入 / 256K 上下文 / `-u256` 的双 lane 并发解码，cohort ratio（`2×单请求
解码时间 ÷ 并发解码时间`）从 **1.26 提升到 1.78–1.80**，每 cohort 步 **145.7 → 103–106 ms**，
正好压在卡1（dev1）的理论下界（`50.8 + 51.1`）上。单请求解码不变。

## 溯源信息

- 运行 ID：`2026-10-19-pair-pipeline`
- 脏工作区状态：运行时的二进制由**未提交**工作区构建；其后该工作区作为 `eb650d7b` 提交，
  源码与运行二进制一致（HEAD 当时为 detached，tag 落在同一提交上）。
- 二进制 SHA-256：
  - 本记录（跨步流水）：`bin-infr-pipe` = `8c49d1927beadf439789f53bb0004033d7305e0f54443882a91fb42c748a1e4a`
  - 对照组（旧 3 次调用调度）：`bin-infr-single-dbuf` = `406d9a5fd14179ccdf612b54d861c57ffb9bffa0c0aee75613d9556e911fa62a`
  - 早期基线（第一波 async submit，无预编译）：`bin-infr-firstwave`
- 操作系统 / CPU / RAM / GPU / 驱动：Linux；Intel i7-10700K；62 GiB；
  2× AMD Radeon RX 7900 XTX（RADV GFX1100，24 GiB）；Mesa 23.2.1-1ubuntu3.1~22.04.4
- 模型 / 量化：`Qwen3.8-Flash-Next-GSQ-RCO-abliterated` IQ3_XXS，
  `…-00001-of-00002.gguf`（47.3 GiB）+ `…-00002-of-00002.gguf`（28.8 GiB）。
  **未计算整文件 SHA-256**（76 GiB，跨两分片）；身份以路径 + 大小 + mtime 为准。
- 生效配置：`INFR_PIPELINE=Vulkan0,Vulkan2`、`INFR_PIPELINE_OVERLAP=1`、
  `INFR_AUTO_PROFILE=aggressive`、`INFR_KV_OVERFLOW=1`、`INFR_KV_OVERFLOW_VRAM_MB=0`、
  `INFR_VRAM_BUDGET=24g`、`INFR_UBATCH_PARALLEL=256`、`INFR_PIPELINE_LAYER_RESERVE_MB=256`、
  `paging.dram=40GiB`、`kernels.vulkan.qsa_union=true`，CLI：
  `serve … --ctx 262144 --parallel 2 --temp 0 --seed 1 -u 256`。
  基准臂另加 `sampling.ignore_eos=true`（见「解读限制」）。

## 工作负载

- 实际提示词：32805 token（同一 prompt 并发两路；脚本先跑并发对，再跑一条单请求参考）
- 上下文容量 262144，深度 32768
- 输出 32 / 128 token，`temp 0`/`seed 1` 贪心
- 预热、重复与顺序：每组二进制 clean 臂各 2 次、交替执行；单请求臂紧随并发臂

## 结果

cohort ratio = `2 × 单请求 decoded_ms ÷ 并发 decoded_ms`；每步 = `并发 decoded_ms ÷ 输出 token 数`。

| 输出 | 旧调度 ratio | 跨步流水 ratio | 并发解码 | 每步 |
|---|---:|---:|---:|---:|
| 32 tok（clean ×2） | 1.2568 / 1.2560 | **1.7761 / 1.7841** | 4660.8 / 4680.9 → 3299.0 / 3301.8 ms | 145.7 → 103.1 ms |
| 32 tok（`INFR_PIPE_TIME=1`） | 1.2608 | **1.7207** | 4657.7 → 3412.4 ms | 145.6 → 106.6 ms |
| 128 tok | 1.1819 | **1.8021** | 20393.5 → 13378.1 ms | 159.3 → 104.5 ms |
| 短 prompt + 开 EOS | 1.266 | **1.605** | — | — |

流水版每步实测（`INFR_PIPE_TIME=1` 的 `[pipe overlap]` 标记，32 步均值）：

| 调用 | dev1 段 | dev0 段（尾部完成） | front+hook | joined |
|---|---:|---:|---:|---:|
| α（dev1 = A.tail(t)，dev0 = B.front(t)+B.tail(t)） | 50.8 ms | 48.3 ms | 11.8 ms | 54.4 ms |
| β（dev1 = B.tail(t)，dev0 = A.front(t+1)+A.tail(t+1)） | 51.1 ms | 36.3 ms | 2.2 ms | 51.3 ms |

## 底层推导

旧调度每步 3 次 `execute_pipelined`：`(None,[A.tail]) → (A.tail,[B.tail]) → (B.tail,[])`。
把 32 步按分量拆开（ms/步）：`A.dev0=37.5`、`A.dev1=53.5`、`B.dev0=33.1`、`B.dev1=51.0`。
第三步只有卡1 在干活，**卡0 全程空转 51 ms**；每步合计 145.7 ms，卡0 利用率 48%。

跨步流水的调度把「每步 3 次」变成「每步 2 次」，并让 lane A 比 lane B 先跑一步：

```
prologue: execute_pipelined(None, [A.front(0), A.tail(0)], hook: 上传 A 的 PLE)
每步 t:
  α: execute_pipelined(deferred=A.tail(t), stages=[B.front(t), B.tail(t)], hook: 上传 B 的 PLE)
  β: execute_pipelined(deferred=B.tail(t), stages=[A.front(t+1), A.tail(t+1)], hook: 上传 A 的 PLE)
```

于是每个 dev1 的一半都与**另一条 lane 的 dev0** 重叠，每步代价
`max(A.dev1, B.dev0+front) + max(B.dev1, A.dev0+front)`，实测 103–106 ms ≈ dev1 的两半之和。
要点：

- layer-0 的 `front` 从「同步 execute」改成 stage，PLE 上传改写进 `after_dev0` hook ——
  与既有 chunked prefill 流水同一种写法；两 lane 的 PLE 暂存与 recurrent 状态本来就独立
  （`ple_group[lane]`、`peers[lane-1].ple_state_buf`），hook 只写自己那条，无串扰。
- 单 lane 那一侧（plan0/plan1 预编译）实测**中性**：256 token 输出中位 94.83 ms/token（预编译）
  vs 94.92 ms/token（无预编译），逐字节一致。原因明确：宿主侧 build+compile+bind+prepare
  只有 ~0.3–0.4 ms/token，而单 lane 每 token 91–95 ms 全是 GPU 串行链。所以双缓冲是结构性
  准备，不是单 lane 的性能杠杆 —— 它只在两条 lane 并发时才有价值。
- **没有引入第二个 compute queue**：新增的全部是 async submit（同一队列上背靠背 submit +
  在依赖点 fence 等待）。

## 解读限制

- 基准臂开 `sampling.ignore_eos=true`，跑满固定 token 预算；EOS 开启的短 prompt 组单独验证。
- 在 EOS 组里，自然停止的那条 lane 与旧调度逐字节相同，另一条（先跑一步那条）**多 1 个
  token**：这是本实现有意的语义 —— 已经起跑的下一步会跑完并发出去，以保证 KV 行与 PLE
  状态自洽（两条请求本来独立）。cohort 整体仍是「任一 lane 停止 → 整队停」，**旧行为未改**。
- 「某 lane 预算用尽」护栏（游标越界 → 先补齐在飞半步再停）没有可复现场景，需要不等长
  prompt 预算才会命中。
- 只有 1 台机器、1 个模型、1 个量化、2 条 lane；layer split 仍是 26/22+head。
- dev1 现在是唯一上界（≈102 ms/步 = 2×51）。列表里 α 的 `front+hook=11.8 ms`（等 B 的 PLE
  ticket）目前被 dev1 完全吸收；dev1 变快后会露出来，届时把 B 的 PLE ticket 提前一个迭代提交。

## 产物

- 结果 JSON（含每条 lane 的逐 byte 输出文本）：`ab-32k-pair-dbuf/pair-*-clean*.json`、
  `pair-*-long128.json`、`pair-*-eos*.json`、`pair-base-pipetime.json`
- 服务器日志（含 `[pipe overlap]` 标记）：同目录 `server-*.log`
- 单 lane A/B：`ab-32k-single-u256-dbuf/run-*.json`
- 二进制存档：`bin-infr-{firstwave,single-dbuf,swap,pipe}`
- 测量脚本：`work/run_pair.mjs`、`work/run_single.mjs`
- 完整推导笔记：`ab-32k-pair-dbuf/NOTES-double-buffer-diagnosis.md`

以上位于 `/media/qiankun/新加卷1/MoE4All_work/`（仓库外的本机数据目录），不随 tag 发布。
