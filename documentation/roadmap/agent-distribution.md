---
kind: proposal
status: proposed
scope: agent-product
updated: 2026-10-11
source_checkpoint: release-0.10.0
---

# Agent 发行剩余工作

## 已确定的方向

MoE4All Agent 是品牌化 DSH 加本项目维护的插件，复用原有 Agent、工具、审批、profile 和市场生态，不新建 Product Host、Engine Manager 或第二套控制协议。内部包名、CLI 和 `DSH_HOME` 等兼容标识可以保留，但进程必须指向独立的数据目录和端口。

0.10.0 已纳入品牌化源码 gitlink、八个插件 gitlink、干净初始化与隔离 Web 开发脚本。此前“品牌替换尚未完成”的判断已经过时。已实现机制集中在[当前产品架构](../architecture/product/agent-distribution.md)，不要把这份路线图当成第二份事实文档。

## 仍需交付和验收

1. 正式 Agent EXE 的构建、完整依赖打包、签名、首次启动与卸载路径。引擎 0.10.0 tag 本身不是完整 Agent EXE 发行；后续 preview 的实现和测试不回填成 tag 能力。
2. 内置插件随整包更新；可自动检查，但安装必须由用户确认。验证网络失败、回滚及用户数据保护。已有插件 updater 修改不等于完整应用更新器验收。
3. 首次引擎连接/可选受控启动，再进入手机连接引导；列网卡、可配置监听地址、鉴权和暴露风险。第三方服务必须主动开启，不能含私人地址、凭据或网络设置。
4. 可选的一键 DSH 迁移采用快照复制，绝不共享可写 profile、session、记忆数据库或设备配对状态；迁移前后两应用仍能独立同时运行。
5. 使用真正干净机器和最终发行包验证插件依赖 materialization、手机端审批/提问、定时任务、模型发现及外部引擎连接。只管理自己启动的引擎，不能停止已有生产服务。
6. 如采用 Desktop 外壳，再接入其源码与打包流程；当前 Web 开发入口不冒充桌面成品。

生产一致性指使用模式和功能契约，不指复用真实数据。模型发现失败、API 离线、市场不可达和端口冲突也必须可恢复；维护插件独立仓库的内部开发与引擎性能优化保持分线。迁移出处见[DSH 来源与本地改造记录](../evidence/changes/product/dsh-source-and-local-migration.md)。
