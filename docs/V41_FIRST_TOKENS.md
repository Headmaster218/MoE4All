

---

## Round 33：改了白名单，快了 2.16 倍，然后我误判了正确性 —— 以及为什么这一轮不算数

### 做了什么

按 Round 32 的分析，把 `DType::Iq1M` 加进 `MOE_MMQ_PAGED_DTYPES`（两个列表现在 20/20 一致）。

```
改前: [prefill 731 tok @ 12 tok/s (62763 ms)]
改后: [prefill 731 tok @ 25 tok/s (29079 ms)]
native_idm_iq1m_paged:  38.8s -> 10.6s   (276 次 -> 137 次)
```

**2.16 倍。** `synthetic_deepseek2` 38/0，`moe_mmq_drift` 5/5 全绿。

### 然后我误判了

我用一个自然语言 prompt 测输出：

```
prompt: "Explain in one short sentence why the sky is blue."
输出:   "The user wants me to translate the following English text into Chinese..."
```

**毫无关系。我判定 "MMQ 产生错误数值"，把改动 stash 回退。**

### 但基线也是同样的输出

回退后跑同一个 prompt：

```
基线(GEMV): "The user wants me to translate the following English text into Chinese. But the assistant's final message is..."
MMQ  版本 : "The user wants me to translate the following English text into Chinese. But be careful—the output should be the the translation..."
```

**开头逐字相同。**

那个病态输出是 **V4.1 在这个 prompt 上的固有问题**（chat template 或模型本身），
**与我的改动无关**。**我撤回"MMQ 产生错误数值"这个结论——它是错的。**

### 这一轮真正暴露的问题

1. **两次判错，方向相反。** 先说"成功"（只看速度），再说"失败"（看了一个两条路径都错的输出）。
2. **两次都缺少对照组。** 判"快"时没确认数值对；判"错"时没跑基线。
3. **`/tmp/p512.txt` 是随机词序列**（`gamma photon beta expert...`），用它测速度可以，
   但用它看"输出像不像话"毫无意义。我一度差点据此判断。

### 正确的验证方式

**不是看生成文本像不像样，而是做数值对比**：同一输入下两条路径的 hidden state / logits 是否在容差内一致。
这需要一个专门的对比工具，而不是靠人眼读输出。

### 状态

- **改动已 stash**（`stash@{0}: iq1m-paged-mmq-BROKEN-wrong-output`，这个名字本身也是误判的产物）
- **工作树回到 8d273de，基线二进制已重编**
- **2.16 倍这个数字是真的，但"是否正确"完全未验证**

### 下一步

**先建数值对比工具，再碰白名单。** 顺序不能反。
在这个工具存在之前，任何"快了 N 倍"的数字都不能提交。

### 教训

这一轮我把一个**未验证的加速**报成了成功，又用一个**没有对照组的观察**判成失败。
两次都是同一个毛病：**急着得出结论，而对照组还没跑。**

