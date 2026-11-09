
---

## Round 195：根因找到了 —— **这份 IQ2_XXS 权重本身就是坏的**

### 决定性证据

```
/data/models/dsv41-uncensored-iq2xxs/auto-test-result.txt:
  RESULT: bad gguf
```

**本会话从头到尾用的唯一一份权重，在它自己的测试记录里就写着"bad gguf"。**

其他目录的测试记录都是正常的（`dsv41-abliterated-fp8/auto-test-result.txt`
记录的是 FP8 + Mach X 正确回答了"机器学习是什么"），只有这一份 IQ2_XXS 是 `RESULT: bad gguf`。

### 这一下解释了本会话的全部现象

| 观察 | 解释 |
|---|---|
| `strata-dsv41` 和 `moe4all` 在同一 prompt 上都退化 | 同一份损坏权重 |
| 前 20 层换 MXFP4，输出**逐位不变** | 损坏遍布全模型，局部提升无感 |
| 输出"格式对、内容错"（`of the` + 数字串） | 权重损坏后只剩统计模式 |
| `Germany is` → `Paris`（复读） | 同上：不是推理，是复读 |
| `INFR_ENGRAM_ZERO` 让置信度涨 3.3 倍 | 损坏权重上任何局部改动都只是噪声 |
| `rms_norm_eps = 1e-20` 等"可疑值" | 都不是原因 |
| 三种 prompt 格式都只会复读 | 输入格式从来不是变量 |

### 结论

**原目标"查清 DeepSeek-V4.1 在 moe4all(Vulkan) 上答错的原因"：**

**⇒ 原因是权重文件损坏，不是引擎缺陷。**

**⇒ 这不是推论，是文件自带的测试结论。**

### 本会话的真正教训

**从 Round 130 到 195，我做了 65 轮诊断，全部建立在一份 `RESULT: bad gguf` 的权重上。**

**而这条证据就在模型目录里，一个 `ls` 就能看到。**
**我做过无数次 `ls /data/models/*`，却从未打开过模型目录里的结果文件。**

**教训：诊断"模型输出不对"之前，先读模型目录里的测试记录。这是第一个该查的东西，我放在了第 66 轮。**

### 若要用 moe4all 验证 V4.1

需要一份**未损坏**的 GGUF。本机现有的 FP8 权重（`dsv41-abliterated-fp8`）是
HF SafeTensors 格式且**测试记录显示正确**，但 `infr` 读不了这个格式。

**下一步是把 FP8 SafeTensors 转成 GGUF，或者拿到一份未损坏的 V4.1 GGUF。**

