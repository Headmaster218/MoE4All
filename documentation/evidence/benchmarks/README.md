---
kind: index
status: current
scope: benchmarks
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 基准测试记录

- [2026-09-27 Qwen3.8 整体性能与稳定性](2026-09-27-system-performance-stability.md)：20K/100K/150K，普通、MTP、视觉及双槽单活跃对照。注意测试提交不属于 `release-0.9.0` tag 血统。
- [2026-09-24 Qwen3.8 MTP 20K 对照](2026-09-24-qwen38-mtp-20k.md)
- [2026-09-24 Qwen3.8 MTP 150K 对照](2026-09-24-qwen38-mtp-150k.md)
- [2026-09-24 Qwen3.6 35B 20K/150K](2026-09-24-qwen36-35b-20k-150k.md)
- [社区性能反馈](community-reports.md)
- [2026-08-29 Windows 本地模型矩阵](2026-08-29-windows-local-model-matrix.md)
- [2026-08-20 Qwen3.6 APEX 矩阵](2026-08-20-qwen36-apex-matrix.md)
- [2026-08-24 Qwen3.5-122B 冷 2K 追踪](2026-08-24-qwen35-122b-cold-trace.md)
- [2026-08-25 MoE Pager 微基准](2026-08-25-moe-pager-microbench.md)
- [2026-08-03 已验证模型快照](2026-08-03-validated-models.md)

这些页面是历史运行记录，不自动代表当前 HEAD。架构和版本事实以 `release-0.9.0` tag 为准；每条实测仍以自身记录的 commit、硬件和口径为准。原始报告、逐轮结果、日志和 trace 存放于本机忽略的 `benchmark-data/`，不作为文档发布。新记录应使用 `_templates/benchmark.md` 并写明来源、证据限制和必要校验信息。
