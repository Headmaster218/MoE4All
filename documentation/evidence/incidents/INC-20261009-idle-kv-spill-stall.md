---
kind: incident
status: historical
scope: idle-kv-spill
recorded: 2026-10-11
source_checkpoint: release-0.10.0
---

# 闲置 KV spill 阻塞 Decode

## 复现与影响

2026-10-09 在线对账中，一路 Decode 与另一个闲置 150K slot 写 SSD 重叠。12:47:53.392..56.564 的空窗约 3171.7 ms，下一次约 3223.85 ms；相关条目 150524 tokens、3249991712 bytes，spill 约3179 ms。短单测量因此跌到25.52..41.21 tok/s，spill 后重测恢复约42.81。

## 根因和修复

会话 cold-cache export 在共享 StepGate 内长时间执行 readback、hash/文件写入/同步，导致正常推理不能及时进入。不是“空闲 KV 本身计算了三秒”，也不证明 GPU kernel 变慢。

`b530332c7` 将 readback 分成约16MiB chunks，读取时取得 gate，hash、磁盘写入、fsync 和 GC 放到 gate 外，期间让推理继续。slot 所有权在 export 期间保持不变，索引 publish 仍同步；主 KV 和 MTP sidecar 持久化成功后才释放 live state。失败不能提前丢掉唯一可用会话。

## 验证界限

观察数据属于修复前在线服务；当前机制按 tag 源码核对。本轮未重跑 I/O overlap，也不把这项修复解释为消除了全部长三路稳态差距。回归需覆盖并发推理、失败写入、metadata publish、KV+MTP 同步恢复及安全 shutdown。详情见[在线对账](../benchmarks/2026-10-09-live-service-reconciliation.md)与[当前会话缓存](../../architecture/services/server-and-session-cache.md)。
