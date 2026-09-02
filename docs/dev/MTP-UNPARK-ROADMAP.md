# MTP 解 park 路线图 — f32 仲裁 / RequestCtx 管线 / scratch-arena 自适应

日期：2026-09-01
机器：AMD RX 7700 XT 12GB / i5-12600KF / 64GB DDR4 / Windows 11（ReBAR on）
模型：Ornith-1.5-35B-A3B-APEX-MTP-I-Quality-MTPFIX.gguf（shisa 蒸馏头嫁接版，见 MILESTONE-REPORT-0831.md §11）

## 一句话

三项工作（f32 仲裁、RequestCtx 管线、scratch/arena 自适应）全部完成后，MTP 解 park，
日常推理从 **42 t/s 提升到 75-93 t/s（1.8-2.2×）**——且是在 server API 上、任意上下文
长度、输出与普通 greedy 严格一致的前提下。2026-09-01 的本地实验版已在 131K 上跑通
75-81 t/s，属于概念验证，尚不可合入。

---

## 一、2026-09-01 实测数据（本次验证的证据基础）

### 1.1 基线与发布版

| 场景 | 速度 | 备注 |
|---|---:|---|
| v0.5.2 serve @131K（日常配置）| **41.5-42.7 t/s** | 与 PR 分支修复后持平 |
| v0.5.2 bench 默认（自动 expert cache）| pp 66 / tg 26 | 自动分配在 12GB 卡上选值很差 |
| v0.5.2 bench `paging.cache=4g` | pp ~250 / tg ~43-45 | 4g 为本机最优且最大可用（5g 起 VRAM guard 拒绝）|

### 1.2 PR 分支（margin 修复前后的回归与恢复）

| 场景 | 修复前 | 修复后 |
|---|---|---|
| serve/run @131K | ❌ "automatic MoE arena could not find a device-safe size" | ✅ 恢复，42 t/s |
| bench decode d4096（cache=3g）| 16.4 t/s（v0.5.2 同参 34.2）| 未重跑（margin 降级已恢复 131K 布局）|
| 修复 commit | — | `7e519769 fix(moe): degrade the runtime margin...` |

### 1.3 MTP 本地实验（实验版二进制，mtp_enabled 临时解 park，未提交）

| 配置 | decode | α（草稿接受率）| 备注 |
|---|---:|---|---|
| MTPFIX + **131K + q5_0 KV** + 贪心 MTP | **81.5 t/s** | ~0.95 | ⚠️ q5_0 KV 输出退化（见 §三.4），不可用 |
| MTPFIX + **131K + q4_0 KV** + 贪心 MTP | **75.3 t/s** | ~0.98 | 概念验证通过 |
| MTPFIX + 131K + q8_0 KV + 贪心 MTP | ❌ | — | prefill ring 下限 1566 MiB 不满足 |
| MTPFIX + 8K + q8_0 KV + 贪心 MTP | **93.6 t/s** | 1.000 | 8.31 已验证（MILESTONE-REPORT-0831 §11）|
| 无 MTP 基线 | ~42 t/s | — | |

结论：**q4/q5 KV 腾出 MTP 头会话空间的思路成立**（131K + MTP 可达 75-81 t/s），
但受 §二.3 的 VRAM 规划问题与 §三.4 的 KV 内核问题双重阻塞。

---

## 二、三项剩余工作

### 1. f32 / top-2 margin 仲裁 —— 输出一致性

**问题**：MTP 的契约是"输出与普通 greedy 严格一致，纯加速"。但 int8/f16 解码内核带
微小舍入噪声，MTP 的 verify 批量与普通解码链的批量形状/KV 状态不同——当 top-1 与
top-2 logit 差距极小时，噪声足以让两条链选出不同 token，其后整段回答完全分叉。
这就是 `mtp_spec_matches_target_only_greedy` 失败且被 `#[ignore]` 的原因。

**修法**：verify 时检测 top-2 margin 低于阈值的"险胜"位置，对该位置用 f32 高精度
重算（或服从普通解码链），使两条路径必然同 token。阈值需要扫参（精度 vs 命中率）。

**验收**：`mtp_spec_matches_target_only_greedy` 去掉 `#[ignore]` 后通过。

### 2. RequestCtx 管线 —— MTP 接入 server 的前提

**问题**：MTP turn 路径（`DenseSeamChat::generate_turn_impl` 的 MTP 分支）丢弃了
`RequestCtx` 与 `stable_prefix`，导致：

- 请求级 temperature/top_p/top_k/seed/penalty 在投机循环内**不生效**
- stop 序列、客户端断开**无法及时终止** GPU 计算
- GPU step gate 无法传递
- 前缀缓存不复用，**每轮全量重 prefill**（131K 下每轮多等数十秒）
- cached/completion token 统计不准

**修法**：MTP 分支把 `req`/`stable_prefix` 贯穿到 `generate_mtp_spec_*` 驱动：
abort 信号轮询、采样参数下沉、前缀复用与 MTP 回滚快照的兼容设计。

**验收**：`infr serve` + INFR_MTP=1 下，客户端断开即时释放 GPU 槽；同请求
MTP 与非 MTP 的 usage 统计一致；带 stop 序列的请求行为与非 MTP 路径相同。

### 3. scratch/arena 自适应 —— 任意上下文长度可启动

**问题**：MTP 头会话的 VRAM 预留使用固定 `scratch = 576 MiB`（历史手工调参：
256→512→768→576）。实测（2026-09-01，RX 7700 XT）：

- 131K：差 ~287 MiB → 头的 397.9 MiB `resident-bda` 块被 guard 拒绝
- scratch 提到 896 MiB → 131K 通过，但 **8K 反而失败**（静态值顾此失彼）
- 失败与 KV dtype/ctx 组合强相关，静态常数无法覆盖

**修法**：头装不下时走与 margin 降级（`7e519769`）相同的思路——**头-arena 联合
重试**：以头的实际分配反馈收缩 expert arena/elastic 窗口（前缀 ring 下限
1566 MiB 之上），允许头借用 arena 余量，而不是一次规划定生死。

**验收**：8K / 64K / 131K × q8/q4 KV 矩阵全部可启动，无需手工调 scratch。

---

## 三、附加发现（2026-09-01）

1. **q5_0 KV 内核疑似损坏**：无 MTP 的纯 trunk 在 131K + q5_0 KV 下输出退化为
   `"!!!"`（q8/q4 同条件正常）。GUI 的 KV 下拉本来只暴露 q8_0/f16——建议上游
   排查 q5_0/q5_1 的 KV 写读内核后再开放。
2. **runtime margin 降级修复**（`7e519769`）：物理 margin 分片在 131K 下曾使
   serve/run/bench 全部无法启动（回归 v0.5.2 可用的配置）；现在空间不足时按缺口
   收缩 margin、至零回退专家槽借用，仅在全失败时报错。
3. **v0.5.2 的自动 expert cache 在 12GB 卡上选值很差**：显式 `paging.cache=4g`
   使 prefill 66→250 t/s（3.7×）、decode 26→43 t/s。上游值得把自动分配策略修一修。
4. **MTPFIX 头 α≈0.98-1.0**（131K/8K 一致）：嫁接的 shisa 蒸馏头对本 trunk 的
   预测几乎完美，MTP 工程本身没有正确性问题。
5. **⚠️ MTPFIX.gguf 的 TRUNK 已损坏（09-01 晚补测确认）**：同一 CLI 路径、同一
   prompt 下，原始 Quality 模型正确回答（"391"），MTPFIX 输出退化为重复 `"!"`
   ——与 ctx（8K/131K）、KV dtype（q8/q4/q5）、MTP 开关**全部无关**，即损坏在
   文件本身（嫁接的 blk.40 张量替换或专家库 Q8_0 重量化损伤了 trunk 数据）。
   **推论：退化分布的 token 置信度极高，α≈1.0 是损坏的症状而非成功**——8.31
   深夜的 93.6 t/s 测速有效，但输出质量当时未验证。MTPFIX 需要用 graft 工具
   重新生成并先用原始 trunk 路径做输出质量回归，再重测 MTP。头-arena 联合重试
   修复（`ae55a6b1`）后，8K 与 131K 的 MTP 会话均已可正常启动（规划层面）。
6. **嫁接工具根因修复 + MTPFIX2 重嫁接（09-02）**：`graft_v2.py` 两个致命 bug——
   ①索引偏移做 32 字节对齐而数据写循环连续写（源文件本身连续），从第一个非
   32 倍数张量起索引与数据累积漂移，全文件读出垃圾（`output.weight` 全零 →
   logits 恒定 → "!!!"）；②`down_proj` 多做了 `transpose(0,2,1)`（gate/up 没有）。
   `graft_v3.py`（Temp）修复两者并带 753 张量字节级自检；MTPFIX2.gguf trunk
   输出质量回归通过（"391"），数值探针全部健康。
7. **⚠️ 终审（09-02）：健康文件上嫁接头 α = 0.000** —— 256 周期 0/1521 接受，
   MTP 纯负优化（~10 t/s vs trunk 54.7）。shisa 蒸馏头是对**其它主干**蒸馏的，
   对 Ornith-APEX 微调后的分布完全没有预测力；8.31 的 α=1.000 是损坏文件上
   退化分布的假象。**结论：这个模型要吃 MTP 红利，需要针对 Ornith-APEX 主干
   重新蒸馏/训练一个 MTP 头（模型侧工作），引擎侧管线已全部就绪**——
   头-arena 联合重试、host-embed 回退、token_embd 上传移除、台账诊断都已落地
   （`9da48058`/`5485336c`/`2aa199f5`），好头一到即可直接测。另：verify 前向
   在 131K 有 4-5× 于 decode 的效率问题（70-88ms vs 18.5ms，m=1-2），好头
   到位后也值得查。

---

## 四、完成后的收益

| 项 | 现状 | 三项全齐后 |
|---|---|---|
| 日常 decode | 42 t/s | **75-93 t/s**（1.8-2.2×）|
| 使用方式 | 终端 run（实验）| server API（OpenAI 兼容）|
| 上下文 | 8K（实验可靠）| 任意（131K 已实测跑通）|
| 输出 | 无一致性保证 | 与 greedy 严格一致（f32 仲裁）|
| 断开/停止 | 不生效 | 即时释放 GPU 槽 |

## 五、建议实施顺序

1. **scratch/arena 自适应**（先做：不依赖 park，直接提升现有可用性，131K+MTP 实验即可复现）
2. **f32/top-2 margin 仲裁**（解除 correctness gate，去 `#[ignore]`）
3. **RequestCtx 管线**（server 接入，最后做，纯工程量）
4. 全部完成后解 park → 更新 PR #21 → `mtp_spec_matches_target_only_greedy` 转正
