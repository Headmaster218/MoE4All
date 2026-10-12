---
kind: index
status: current
scope: user-guide
last_verified: 2026-10-11
verified_commit: 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc
baseline_tag: release-0.10.0
---

# 用户指南

- [快速开始](getting-started.md)：Windows 发布包、模型和第一次启动。
- [在 Linux 上构建并运行](linux-build-and-run.md)：从源码构建后的启动向导、CLI 用法与平台注意。
- [API 使用](serving/api-quickstart.md)：Chat、Responses、图片、Embedding 与鉴权。
- [思考模式控制](serving/thinking-controls.md)：CLI、Wizard 和 Chat Completions API。
- [配置参考](../reference/configuration.md)：完整 TOML、环境变量和 `--set` 规则。
- [模型能力矩阵](../reference/model-capabilities.md)：哪些组合已经实现或验证。

根 [README](../../README.md) 是引擎下载入口；本目录承接 0.10.0 的完整操作说明。性能表的详细条件见[发布前实测矩阵](../evidence/benchmarks/2026-10-09-release-0.10.0-matrix.md)，不会把旧样本改写成新版本测量。

- [图片与 Embedding](serving/images-and-embeddings.md)：格式、解码库、payload 与辅助资源。
- [CPU miss](cpu-miss.md)：实验开关、适用条件和本机对照。
- [Agent 开发入口](agent-development.md)：干净初始化、既有 DSH 开发配置与进程边界。
