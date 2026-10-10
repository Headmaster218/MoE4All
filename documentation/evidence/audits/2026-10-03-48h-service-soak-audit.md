---
kind: audit
status: completed-with-limitations
scope: qwen38-48h-service-soak
audit_date: 2026-10-03
evidence: measured
---

# 2026-10-03 Qwen3.8 服务 48 小时运行审计

## 结论

同一 `infr.exe` 进程连续运行 48.26 小时后，未观察到随请求数或长上下文持续增长的进程内存、GPU 提交量、句柄、线程、TCP 连接或 SSD KV 容量。两次相隔 9.34 小时的快照中，工作集和 Private Bytes 均下降，GPU 三项计数完全不变，句柄、线程和连接数也下降；与此同时最高可见请求编号从约 1840 前进到 2242，并继续完成 100K 以上上下文的 KV 恢复、增量 Prefill、Decode 和后台 spill。

因此，本次观测范围内**没有资源泄漏证据，也不需要为释放资源而重启服务**。这是一份真实工作负载下的运行中审计，不是固定请求矩阵 benchmark，也不证明所有未覆盖功能路径都无泄漏。

## 运行工件与配置

- 版本输出：`infr 0.9.0`。
- 进程：PID 34424，启动于 2026-10-01 00:04:18（Asia/Shanghai）。
- 二进制：`target/release/infr.exe`，时间戳 2026-09-28 00:19:36，SHA-256 `64cc6e636bb811f85ac9eb274960bb13ea028a815b058a509e7fba582608b81c`。
- 二进制未嵌入可读取的 Git SHA；不能把当前工作区 HEAD 或相同版本号当作其精确源码提交。
- 硬件：AMD Radeon RX 7900 XTX，24 GiB；Windows Vulkan，设备 `Vulkan1`。
- 模型：Qwen3.8-Flash-Next-AD 4.27 bpw Q4_K_M-M64；Q8_0 K/V；每槽 160K 上下文。
- 核心配置：自动激进、`--parallel 2`、`spec.mtp=false`、SSD session cache 上限 10 GiB、闲置 90 秒后 spill。
- 服务同时配置了视觉 projector 和动态 embedding；观测窗口内确认了 embedding 载入/释放和 SSD KV spill/restore，没有观察到近期视觉请求。MTP 关闭，因此本报告不覆盖 MTP 生命周期。

服务的相关启动参数为：

```text
serve --dev Vulkan1 --ctx 160k --set device.auto_profile=aggressive
  --set kv.type_k=q8_0 --set kv.type_v=q8_0 --set spec.mtp=false
  --parallel 2 --set kv.session_idle_secs=90 --set kv.session_cache_max=10g
  --mmproj <Qwen3.8 mmproj> --embedding-model <Qwen3-Embedding-0.6B-Q8_0>
  --embedding-idle-timeout 60 <Qwen3.8 model>
```

## 跨时段资源快照

两次检查均为只读：没有停止服务、清空缓存、发起额外模型请求或改变配置。HTTP `/health` 在两次检查时均返回 200。

| 指标 | 38.92 h，2026-10-02 14:59:45 | 48.26 h，2026-10-03 00:20:04 | 9.34 h 变化 |
|---|---:|---:|---:|
| Working Set | 52.801 GiB | 51.862 GiB | -0.939 GiB |
| Private Bytes | 70.242 GiB | 70.173 GiB | -0.069 GiB |
| Peak Working Set | 55.476 GiB | 55.476 GiB | 0 |
| Peak Private Bytes | 70.296 GiB | 70.302 GiB | +0.006 GiB |
| Handles | 727 | 714 | -13 |
| Threads | 41 | 40 | -1 |
| Established TCP | 20 | 4 | -16 |
| GPU Dedicated | 22.186 GiB | 22.186 GiB | 0 |
| GPU Non-local | 31.094 GiB | 31.094 GiB | 0 |
| GPU Committed | 53.279 GiB | 53.279 GiB | 0 |
| SSD KV 文件 | 4 | 3 | -1 |
| SSD KV 总量 | 9.946 GiB | 8.539 GiB | -1.407 GiB |
| 系统可用物理内存 | 1.154 GiB | 1.467 GiB | +0.313 GiB |
| 系统 commit 余量 | 9.961 GiB | 8.978 GiB | -0.983 GiB |

系统 commit 余量下降而该进程 Private Bytes 同期下降，不能归因为服务私有分配增长。Peak Private Bytes 只增加约 6 MiB，也没有形成随时间单调增长的趋势。SSD KV 始终受 10 GiB 配额约束，文件会被替换和淘汰，不是只增不减。

第一次检查期间另做了 165 秒连续采样。当时一个约 54K 上下文请求处于 Prefill：Working Set 增加 832.5 MiB，但 Private Bytes 仅增加 79.9 MiB，GPU committed 增加 39.9 MiB，句柄减少 1，线程在 40--42 间变化。结合当前 Working Set 后续回落且未超过既有峰值，该变化符合模型文件页和缓存重新驻留，不像线性分配泄漏。

## 请求、KV 与资源回收活性

- 第一次快照最高可见请求编号约为 1840，面板为 `active=2`、`queued=18`；第二次已到 2242，面板为 `active=2`、`queued=2`。队列继续消化，没有形成卡死后只积压不前进的状态。
- 第二次快照中，一个槽位在 105,968 / 163,840 token 处 Decode，另一个在 62,315 / 163,840 token 处 Prefill。长上下文活动没有推动 Private Bytes 或 GPU committed 高于第一次快照。
- 请求 2239 命中 `cached_prefix_tokens=104062`，只 Prefill 474 个新 token，随后生成 680 token；这证明长 KV restore、增量续写和 Decode 在 48 小时后仍能继续工作。
- 同一时段完成了 104,062-token、2.487 GiB 的 restore（2.347 秒）和 110,522-token、3.026 GiB 的 spill（5.741 秒）；动态 KV 扩展到 131,072 token、4 个 segment。
- embedding 的 603.87 MiB 权重曾进入统一显存，闲置后权重和 runtime 计数均回到 0，并返回 SSD backing。释放后统一显存日志为：arena 15.88 GiB、expert 10.87 GiB、LLM runtime 35.5 MiB、free 673.6 MiB。
- Windows 事件日志未发现本进程相关的 Resource Exhaustion 2004、Display 4101、应用崩溃或挂起事件。

当前活动 TCP 客户端均能映射到本机代理进程，并与活动/排队请求数量同步回落，没有发现从服务启动时一直遗留的连接。面板中的 `failed=0` 只代表当时可见状态；由于服务没有持久化完整 48 小时请求日志或累计统计接口，本报告不把它解释成整个运行期零请求失败。

## I/O 与运行成本

进程累计 I/O 在 9.34 小时内增加约 287.96 GiB 读取和 321.75 GiB 写入，平均约 8.8 MiB/s 读、9.8 MiB/s 写。Windows 短采样中 page reads 活跃而 page writes 为 0，结合多次大 KV spill/restore，这更符合模型页读取和 session cache I/O，而不是持续匿名内存换出。

这不是容量泄漏，但属于需要长期关注的 SSD 写放大和寿命成本。后续若优化 session cache，应分别记录逻辑 KV 字节、实际磁盘写入、淘汰次数和 cache hit，不能只观察目录是否超过配额。

## 判定与限制

| 检查项 | 本次判定 | 依据或限制 |
|---|---|---|
| Host 内存泄漏 | 未观察到 | 9.34 小时后 Working Set 和 Private Bytes 均下降；Peak Private 近似不变 |
| VRAM / GPU commit 泄漏 | 未观察到 | Dedicated、Non-local、Committed 三项字节数完全不变 |
| 句柄 / 线程 / socket 泄漏 | 未观察到 | 三项计数均下降，线程只在工作负载范围内小幅波动 |
| 动态 KV 未释放 | 未观察到 | 超过 100K 的槽位仍可扩展、restore、spill；磁盘缓存受配额淘汰 |
| Embedding residency 未释放 | 未观察到 | idle eviction 后 weight/runtime 均归零 |
| SSD 写入成本 | 需要关注 | 9.34 小时写入约 321.75 GiB，不是泄漏但可能影响 SSD 寿命 |
| MTP 生命周期 | 未覆盖 | 运行时 `spec.mtp=false` |
| 视觉生命周期 | 未充分覆盖 | projector 已加载，但审计窗口未确认视觉请求 |
| 48 小时请求失败率 | 无法判定 | 缺少完整持久日志和累计统计，只能确认当前服务健康、队列前进 |

这份审计支持“当前文本双槽、动态 Q8 KV、SSD session cache 与动态 embedding 组合可稳定连续运行至少 48 小时”的有限结论。它不替代 MTP、视觉、多 GPU、异常恢复或固定压力模型的专项 soak test。

## 证据来源

证据来自运行中进程计数器、GPU Engine/Process Memory 计数器、TCP owner 映射、Windows 事件日志、KV 目录清单、HTTP 健康检查和当前控制台快照。检查过程没有保留完整原始控制台历史，也没有生成可独立重放的 trace；因此证据等级虽为实机 `measured`，仍按 `completed-with-limitations` 记录。二进制 SHA-256 是本页用于绑定运行工件的主要校验值。
