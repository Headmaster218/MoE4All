# MoE4All 项目状态总结（2026-09-02）

环境：AMD RX 7700 XT 12GB / i5-12600KF / 64GB DDR4 / Windows 11（ReBAR on）
分支：`split-pr21`（21 个提交，基于 main 3901476d，已推送，PR #21 已更新）
对照基线：官方 v0.5.2 发布版

---

## 一、功能可用性

### ✅ 完全可用（全部实测验证）

| 功能 | 实测状态 |
|---|---|
| serve @131K 日常配置 | 42 t/s（v0.5.2 同水平）；API/streaming/reasoning_content 正常 |
| run @131K q8 | 54.7 t/s（embed 表移内存 + VRAM 规划修复，比 v0.5.2 +30%）|
| run @8K | 50.4 t/s |
| Vision 图文理解（本 PR 新增）| 单图/多图/顺序端到端验证正确；data-URI/base64 输入 |
| multi 多模型同宿 / serve-embedding / pull | 正常 |
| bench 基准 | pp/tg/pg 全支持（llama-bench 同接口）|

### ⚠️ 可用但需调参

| 功能 | 要点 |
|---|---|
| expert cache | **必须显式 `paging.cache`**：12GB 卡自动分配选值差（prefill 66 vs 250 t/s）；本机最优 4g |
| q4_0 KV | 可用省显存（131K 下省 2.8GB），质量略降；q8 正常时的首选替代 |
| 长时服务 | ~1h 楔死已修（碎片整理重试）；serving 时退出 screenpipe 等占显存程序 |

### ❌ 不可用（已知问题）

| 功能 | 问题 | 状态 |
|---|---|---|
| MTP（`INFR_MTP=1`）| 引擎已就绪（Tiel-Coder 头验证 α=0.48），但 12GB 卡专家工作集装不下 + Ornith 缺有效头 → 负优化 | 等模型侧+硬件 |
| q5_0/q5_1 KV | 输出退化（内核疑似损坏，q8/q4 正常）| 待上游排查 |
| 嫁接 shisa 头的 MTP | 头内容与 trunk 不匹配（α=0.000，numpy 已证明）| 需重蒸馏 |
| http(s) 图片 URL | 未实现（data-URI/base64 正常）| — |
| 12GB 卡 Vulkan ViT | 显存不足自动回退 CPU（功能正常，速度慢）| 硬件限制 |

---

## 二、性能数据（全部实测）

| 场景 | v0.5.2 | 本分支 | 备注 |
|---|---:|---:|---|
| serve @131K | 41.5-42.7 | 42 | 持平 |
| run @131K q8 | — | **54.7** | +30%（embed 移内存 + 规划修复）|
| bench pp512（cache=4g）| ~250 | ~225-252 | 持平 |
| bench 默认（自动 cache）| pp 66 / tg 26 | — | **自动分配选值差，务必显式设** |
| Tiel-Coder trunk @8K | 25.5 | — | 25.6GB 模型 |
| Tiel-Coder + MTP @4096 | — | 2.0 | α=0.48 但分页流量主导，12GB 负优化 |

---

## 三、本周期修复清单（PR #21 第二轮响应）

### 评审意见（7 条全闭环）
1. Windows 编译（5 处错误）✅ 2. MTP 保持 park ✅ 3. RequestCtx 缺失（park 规避）✅
4. Vision 图文顺序（占位符方案 + 测试）✅ 5. Vision 支持范围文档 + 设备传递 ✅
6. D2H DMA 独立提交 + opt-in ✅ 7. 根目录清理 + 硬编码路径 ✅

### 实测挖出的深层 bug（评审未覆盖）
| bug | 修复 |
|---|---|
| 131K serve/run 启动回归（margin 物理分片改造引起）| margin 优雅降级（`7e519769`）|
| **MTP α 结构性钉死 0.000**（draft step-0 embed 喂错 token + accept 错位一行）| off-by-one 修复（`4a8e9116`），α 0.000 → 0.017 |
| 长服务 ~1h 楔死（unified-loan 碎片化）| 冷专家块逐出 + 碎片整理重试（`3dcac560`）|
| graft 工具两个致命 bug（索引漂移 + down 转置）→ MTPFIX 文件损坏 | graft_v3 + 字节级自检，MTPFIX2 重嫁接成功 |

### 诊断基建（新增，永久可用）
- VRAM 分配台账（per-label）+ 大分配实时 trace + 拒绝时 top-consumers dump
- 最近 40 次受保护分配环形缓冲（拒绝时定位重复分配模式）
- MTP stage 计时（draft/verify/catchup 分解）+ 头会话里程碑探针
- `gguf-diff` 示例工具（两 GGUF 张量目录对比 + 数值探针）

---

## 四、MTP 专项状态

| 项 | 结论 |
|---|---|
| 引擎管线 | ✅ 端到端打通（131K+MTP 可跑，off-by-one 修复后接受机制正常）|
| 正确性验证 | ✅ Tiel-Coder 头 α=0.480（72/150, 25 cycles）——与 vLLM 同类一致 |
| shisa 嫁接头 | ❌ 对 Ornith-APEX α=0.000（numpy 复现 + 幅度界证明头内容不匹配）|
| Ornith 原生头 | 随机初始化（未训练），shisa-ai 与本引擎交叉确认 |
| 12GB 卡 verify 性能 | 分页专家拉取流量主导（α=0.48 仍 2.0 t/s vs trunk 25.5）——工作集装不下 |
| f32 仲裁 / RequestCtx | 实现门槛已扫清，等 α>0.5 的头到位后落地 |

**结论：MTP 的剩余工作全在模型侧（针对目标 trunk 蒸馏有效头）。引擎侧管线、
正确性、诊断已完备，好头即插即用（改一行解 park）。**

---

## 五、日常使用建议（RX 7700 XT 12GB）

```powershell
# 日常 API（42-55 t/s）
infr serve <model.gguf> --ctx 131072 --set kv.type_k=q8_0 --set kv.type_v=q8_0

# 显存紧张时
+ --set paging.cache=4g        # 必显式设，自动分配选值差
+ KV 可降 q4_0（省 2.8GB，质量略降）

# 避免
- INFR_MTP=1                   # 头没好之前是负优化
- kv.type q5_0/q5_1            # 内核疑似损坏
- 不设 paging.cache            # 自动分配在 12GB 卡选值差
- serving 时开着 screenpipe    # 曾占 21GB 显存
```

---

## 六、后续待办

| 优先级 | 事项 | 侧 |
|---|---|---|
| 1 | Ornith-APEX 的 MTP 头重蒸馏（按 distill.py 实际 pairing 审计约定）| 模型 |
| 2 | paged-MoE 批量 verify 的分页流量优化（12GB 卡 MTP 提速的前提）| 引擎 |
| 3 | q5_0 KV 内核排查 | 引擎 |
| 4 | 自动 expert cache 分配策略修复 | 引擎 |
| 5 | f32/top-2 margin 仲裁 + RequestCtx 管线 → 解 park | 引擎 |
| 6 | D2H DMA 的 cfg(windows)/probe + Linux A/B | 引擎 |
