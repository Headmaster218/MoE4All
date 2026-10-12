# MoE4All Agent 源码入口

完整说明统一在 documentation：

- [当前产品架构](../documentation/architecture/product/agent-distribution.md)
- [构建与隔离运行](../documentation/guide/agent-development.md)
- [来源与本地改造迁移](../documentation/evidence/changes/product/dsh-source-and-local-migration.md)
- [发行剩余工作](../documentation/roadmap/agent-distribution.md)

请使用干净 Agent 初始化入口；旧 DevHarness 会复制本机设置与凭据，不适合制作公共发行物。
