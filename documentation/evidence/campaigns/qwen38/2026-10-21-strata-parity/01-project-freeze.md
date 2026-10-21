# MoE4All 项目封存说明

日期：2026-10-21  
状态：冻结，暂停继续追赶 Strata 的引擎研发

## 当前工作区

仓库：

`/home/qiankun/Documents/Codex/2026-10-19/oe4all-github/work/MoE4All`

当前分支：

`perf/strata-parity`

当前提交：

`dca2cc4 perf(moe): expose Iq3S hit-first opt-in`

当前二进制：

- `target/release/infr`
- 版本：`infr 1.0.3`
- SHA-256：`b4ed21345dc1089d03bbc69dcc0e5eb6a8fb5d7776131b4b66d7ffa8383409fe`

## 冻结标签

### 正确性稳定版

- 标签：`freeze-iq3-correctness-g0-20261021`
- 提交：`dca2cc4`
- 二进制 SHA-256：`b4ed21345dc1089d03bbc69dcc0e5eb6a8fb5d7776131b4b66d7ffa8383409fe`
- 行为：含 Iq3S expert bank 的 resident graph 默认 `hit_first=false`
- 30K、128 token、两轮输出 hash：
  `49f384eef7391a151c5a3d9e9521cbcb8d5e85cf790db152bdf8d273b8e33eb2`
- 正确，但 decode 只有约 `0.34/0.60 tok/s`
- 结论：只能作为正确性 G0，不可作为日常性能版本

### Vulkan0 Prefill 候选版

- 标签：`freeze-iq3-vulkan0-prefill-candidate-20261021`
- 提交：`2d771ef`
- 该次测量的二进制 SHA-256：
  `8230a02316f6e830ff5039b57527e9299cc3f3d6a5f48020474cf63d87cf4ebc`
- 关键配置：Vulkan0 + `decode_gpu_resident=false`
- 30K、32 token、丢弃预热、3 轮 prefill：
  `401.34 / 403.79 / 402.99 tok/s`
- 输出 hash 三轮一致：`400972cd...`
- decode 约 `31.9 tok/s`，仅短跑参考
- 结论：目前 MoE4All 自己测到的最好 prefill 候选；不是完整 decode 正确性版本，不建议作为默认发布版

## “最好效果”的准确回答

### 日常可用

最好的是 Strata，不是 MoE4All。

Strata 同模型、本机历史实测约：

- 输入：`1129-2191 tok/s`
- 输出：`47-50 tok/s`

### MoE4All 正确性最好

`freeze-iq3-correctness-g0-20261021`：

- 30K、128 token 两轮结果完全一致
- 但 decode 只有约 `0.34/0.60 tok/s`

### MoE4All prefill 最好

`freeze-iq3-vulkan0-prefill-candidate-20261021`：

- 30K prefill 约 `403 tok/s`
- 三轮波动约 `0.61%`
- 但只是 prefill 候选，不是完整可用版本

### 曾经最快但不可用

hit-first decode 历史参考约 `9-16 tok/s`，但第 2 层开始 hidden 分叉，最终文本不稳定，不能发布。

## 已冻结原因

1. 正确且快的 decode 路径不存在。
2. hit-first 确定性修复风险高，预计还要数天，且无成功保证。
3. 即使修好 hit-first，距离 2500 输入 / 100 输出仍有大量架构工作。
4. 同样的 IQ3_XXS 模型在 Strata 上已经达到可用速度。
5. 继续投入的研究收益低于实际使用 Strata 的收益。

## 后续解冻条件

只有在以下任一条件成立时才建议重新启动：

- 明确要求继续做 MoE4All 研究，而不是追求日常可用速度。
- 获得新的硬件、更多显存或可验证的新算法。
- Strata 路径无法继续使用，必须恢复自有引擎。
- hit-first 主 journal/conditional 的顺序问题有明确代码级修复方案，而不是继续盲查。

