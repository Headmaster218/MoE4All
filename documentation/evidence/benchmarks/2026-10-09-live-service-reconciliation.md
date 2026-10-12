---
kind: benchmark
status: historical
scope: live-service
recorded: 2026-10-11
source_checkpoint: release-0.10.0
---

# 在线服务速度与离线结果对账

## 运行身份

2026-10-09 记录，Flash-Next AD 4.27bpw、7900 XTX、3 slots、ctx163840、aggressive/Q8、MTP/CPU miss off。二进制自报 0.9.0，SHA256 `E28B4E47FC59839E51DADE0EBEA11695FDF586A761E3B2118F1A2345986B638E`；没有准确源码 commit 证据，不能称为 stock 0.9.0 或 release-0.10.0 tag。四组 suite 共 192 requests、每请求 512 tokens、四轮每 cell、后三轮 warm。aux 服务未发请求，idle=90秒、SSDcache=10g、TTL=24小时。

指标以原报告为准：API rates 求和、SSE thirds 是 frame/s 近似而非真实 token 时间线，与[发布矩阵共同窗口](2026-10-09-release-0.10.0-matrix.md)不可直接合并。

## 不同请求与历史 matched 对照

| context / active | diverse fresh P | warm D | 历史 accepted D | matched 在线 D |
| --- | ---: | ---: | ---: | ---: |
| 30K / 1 | 1093.14 | 42.83 | 43.23 | 33.95，受 stall 污染 |
| 30K / 2 | 960.38 | 59.71 | 63.45 | 61.52 |
| 30K / 3 | 1033.31 | 70.16 | 71.03 | 69.52 |
| 150K / 1 | 751.83 | 39.53 | 40.57 | 39.60 |
| 150K / 2 | 957.82 | 48.10 | 49.92 | 48.56 |
| 150K / 3 | 1048.88 | 40.43 | 46.99 | 44.54 |

matched online 双路短 -3.04%、三路短 -2.12%、长单 -2.4%、长双 -2.73%、长三 -5.22%。短单污染轮范围 25.52..41.21；spill 完成后重测 42.81，不能拿 33.95 证明永久单路性能回退。长三路仍有稳态差距，原因没有完全归属。

相同 prompt 的 warm 并发共享工作更有利：30K 双/三约 69.90/84.88，对照生产不同 prompt 60.55/66.46；150K 59.77/64.45 对照 50.38/46.97。不能把前者当任意真实工作负载的并发能力保证。

## 状态历史与故障

闲置 150K KV spill 与约两次 3.2 秒的 Decode stall 对齐；readback/磁盘工作长持共享 gate 是可定位原因，见[事故记录](../incidents/INC-20261009-idle-kv-spill-stall.md)。一活跃 150K 初次 P=751.83、ring lane=1，idle KV 释放后 ring=4、P约1313..1316。可见 configured slots、resident idle KV 历史与 live claims 都重要，“active=1”不代表 arena 空。

RAM/commit headroom 很低是风险，不是已证明 hard fault/降频因果；没有 ETW hard-fault 或完整温度证据。当前源码的修复说明和这份修复前在线观测分开保存，不能把所有差距都归给一个 bug。

原始累积报告 `benchmark-data/live-api-speed-reconciliation-20261009.md` 不发布机器 IP、凭据或请求全文。
