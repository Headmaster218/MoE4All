---
kind: campaign
status: historical
scope: prefill-mtp
recorded: 2026-10-11
source_checkpoint: release-0.10.0
---

# Prefill、MTP head 与并发收益策略

## 证据范围

2026-10-04..07，Flash-Next AD 4.27bpw、7900 XTX、aggressive/Q8，主要服务配置2slots/ctx160k/greedy/no-think。本页从 ignored optimization-20261004、optimization-20261005-decode、kv-prefill-loan、prefill-dma、prefill-dual-transport 的 result/log 和提交提炼。screen 常只有一轮，不冒充 final tag 回归；旧实验开关不自动属于当前 Config。

## MTP 改造不是一个固定倍率

原 baseline clean（EXE `8cbcec44448a6fb9df2b1820bd68db2d954e4f49100f5351dccc578ebdaec32d`、768-token 普通任务）：

| context | 单活跃 P | 单活跃 D | 双活跃 D 共同窗口 | 双活跃 D 前/中/后 |
| --- | ---: | ---: | ---: | --- |
| 30K | 571.221 | 38.023 | 47.369 | 46.842/48.494/46.770 |
| 90K | 556.107 | 34.103 | 42.282 | 42.594/42.515/41.738 |
| 150K | 530.770 | 29.487 | 36.148 | 36.056/36.470/35.919 |

非 clean baseline 明显更慢，不选其数字制造提升。ring-head 完整矩阵同顺序的单 D 为37.017/34.268/28.905，双 D45.500/42.477/36.648，并没有证明普通任务固定加速。

同 binary `ef958ddbeef13070e4793db2bd793e0255991637d6aec21e5f3caa076bc9e1ad`，90K/256tokens head horizon full/48K/32K 的 D33.272/33.640/34.043，P557.791/559.340/546.932，输出 hash 相同；150K full/32K D30.901/31.250。窗口限制首先解决驻留和追赶规模，不把一轮约2%的差别宣称稳定加速。`69d8d6203` 固定 head horizon，`1899765ad` 取消 MTP 隐含 ubatch2048 cap，`730b1354a` 收敛后续 runtime/fused paths。

强制并发 MTP 普通30K screen40.200，不启并发45.649；copy任务反而63.073。这说明接受率与负载重要，`5c85a1f54` 加入8-cycle/80%收益探测，默认不强制多路MTP。

## 失败也保留

adaptive 原型（EXE `75c524845948c3db0b1033516e55f513ef9e3ba552975048a417f04193578141`）在90K双路 Prefill 后 panic：`primed MTP lane has a frontier`，两请求 `parallel decode worker stopped before returning its KV`。原因是 ready/frontier 不变量不足，不应记录为完整矩阵通过。后续 ready 候选 `c18156de05b6e1c9e227d718fa5fb4907817fb01c8b721135bedce29a5151ef9` 的30K/90K短回归完成；alpha0.625/0.4583选择plain。

同候选768-token普通任务90K单35.283、双42.468；150K单29.558、双37.916。数数双路30K72.603、90K65.888、150K50.320，150K前/中/后54.379/49.837/46.743；fresh150K另跑62.855、67.096/63.189/58.281，历史状态明显影响结果。30K同候选 matched 并发关闭数数59.313，而MTP72.603；不能将高alpha负载收益推广到普通任务。

长 copy/no-EOS 或 matched disabled-count 的部分150K续跑失败，不能从“前几个case成功”补造150K结果。这些历史原型数据也不能覆盖当前49152 cutoff 的含义；当前策略见[MTP 架构](../../../architecture/runtime/qwen38-mtp.md)。

lookup draft 早期 copy screen on 明显回退；deferred on/off长screen48.319/49.000，未形成稳定收益。QSA sg8开关90K33.769/34.031，以及部分独立Q8/row-group screen 都不足以证明整体加速；后来的选择按新的精确A/B专项决定。

## Prefill 资源和 transport

`565578cdf` 允许借未提交KV范围；`70067ca83` 只为选中的Prefill slot增长KV；`9789f456f` 让imported host ranges走dedicated transfer，mapped non-imported tail保留CPU push，消费前等待timeline；`5324b8c7c` 跨chunk预取只上传不提前计算。

指定ubatch5120的三轮30K平均：原1328.209，DMA1400.027 tok/s（约+5.41%）；150K原1405.297 →1422.229（约+1.20%）。相同早期binary把prefill_upload_async关闭后30K约781.559，对照开启1328.209；这是该实验条件的transport对照，不是最终0.10.0任意配置加速70%。

后续CPU+DMA重叠三轮30K约1408.178，约比DMA1400.027高0.58%，接近噪声。更大ubatch不是无限收益：4096约1290、5120约1328、5632单screen1314.843；大batch需要同时看ring lanes和连续空间。256K普通run失败，另一dense run完成，不把dense结果当普通QSA同条件替代。

很多Prefill-only run仅生成极少token，result里的数百/数千“decodeTps”和零thirds无统计意义，本页不引用作Decode性能。仪表盘短chunk峰值也不代表长prompt均值。

## 最新 live planning

`c853ec6f0` 在活跃事务中联合选择ubatch/ring lanes，并按当前深度定QSA工作区，避免闲置/未来slot高水位过早压缩Prefill。优先合适的3+lane候选，2lane只有比3+方案多至少1024行时更有利，1lane兜底；不是把长Prefill拆成多个同时计算的请求。闲置KV仍保留真实内存，不能借已提交live state。

最终发布前4096配置与不同sampling/模型的数字见[完整矩阵](../../benchmarks/2026-10-09-release-0.10.0-matrix.md)，不能拿本页5120/一轮/768-token screen直比。共享专家Prefill输出修复 `e668c8827` 与混合采样行修复 `ef4d972b2` 是正确性改进，不为追求速度省略。
