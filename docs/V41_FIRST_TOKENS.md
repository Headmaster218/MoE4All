

---

## Round 142：找到本机的正确参照；采样参数排除；问题确定在 logits 之前

### 本机有一个**能正确回答的 V4.1 实现**

`/data/models/dsv41-fp8-e64/auto-test-result.txt`（2026-10-28）：

```
model:   mrfakename/DeepSeek-V4.1-Flash-1017-abliterated (FP8 safetensors)
engine:  Mach X build-jit/src/ie
prompt:  用一句话解释什么是机器学习。
output:  机器学习是让计算机从数据中自动学习规律，并据此进行预测或决策的方法。
decode_tok_s: 1.23
```

**⇒ V4.1 这个模型家族本身是能对答的。问题在 moe4all 的 V4.1 实现里。**

注意差异：Mach X 跑的是 **FP8 safetensors**，我们跑的是 **IQ2_XXS GGUF**，量化格式完全不同，
所以这条只能证明「模型家族没问题」，不能直接用来数值比对。

### 官方推荐采样参数不是原因

那份日志还记了官方推荐采样：

```
thinking temperature 1 top_p 0.95 top_k 0 min_p 0 ...
```

我此前一直用 `temp=0`（贪心），怀疑贪心在平坦分布上会陷入重复循环 —— 这个怀疑合理，
因为症状正是 `We need answer. Need answer.` 的自重复。

实测 `temp=1.0 --top-p 0.95 --top-k 0`：

```
||  ** (*)， extending across **50̑zp. We RP can be stated the VERY*/**N.GR █ [300 3*; ...
```

**不是变得更好，而是出现 `█`、`̑`、`N.GR` 这类乱码。**

### 结论：问题在 logits 之前

| 采样方式 | 输出 | 性质 |
|---|---|---|
| `temp=0`（argmax） | `. Wait, the answer is 100. So` | 语法对、内容错 |
| `temp=1`（采样） | `█ 50̑zp N.GR` 乱码 | 连语义都没有 |

**两条互不相关的采样路径都错 ⇒ 错在 logits 之前，不是采样、不是温度。**

再结合 Round 130 测到的 `entropy=0.001 / top1=0.9999` ——
**分布形状是合法且尖锐的（所以 argmax 能产出合法 token），但分布内容是错的。**

这与「某个上游 op 把残差流算偏了、但幅度仍在合理范围」完全一致，
也正是「语法完整、语义错位」的机制。

### 现有排除清单（累计，均有运行证据）

引擎通用路径（qwen 对照）· prompt 构造 · YOCO · APE · warmup ·
engram（零注入对照）· logits 崩溃/NaN · 超连接滞后（release 断言）·
批量 prefill（已定位为单行缺陷并加常驻守卫）· 采样参数与温度

**下一步方向：需要在 logits 之前定位第一个偏离点。**
目前的硬约束仍是没有可做数值比对的参考实现
（Mach X 是 FP8 safetensors 且为闭源构建；llama.cpp 缺 engram 张量；DeepInfer 跑不动 130 GB）。

