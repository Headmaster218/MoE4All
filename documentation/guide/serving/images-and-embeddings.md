---
kind: guide
status: current
scope: auxiliary-services
last_verified: 2026-10-11
verified_commit: 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc
baseline_tag: release-0.10.0
---

# 图片与 Embedding 服务

启动时可给 `serve` 加 `--mmproj` 和 `--embedding-model`；两者共享 Vulkan unified arena，不是再开一个独立显存预算。CLI 参数和资源策略见[API 指南](api-quickstart.md)与[当前辅助服务架构](../../architecture/services/vision-and-embedding.md)。

## 图片请求

在 `/v1/chat/completions` 的 content parts 中发送 `image_url`，使用 data URI/base64；不要把任意外部 URL 当作已支持的下载入口。示例中的真实 base64 应由客户端生成。

```json
{"model":"your-model","messages":[{"role":"user","content":[{"type":"text","text":"描述这张图片"},{"type":"image_url","image_url":{"url":"data:image/png;base64,<base64>"}}]}]}
```

支持 PNG/JPEG/WebP/GIF/BMP/TIFF/ICO；HEIC/AVIF 需要发布包里的 native `image-codecs` 或绝对路径 `INFR_IMAGE_CODEC_DIR`。HEIF 读取 primary image，不表示支持视频或每帧序列。特定 Qwen3-VL projector 子集受支持，不能把任何 mmproj 都视为兼容；DeepStack 等未实现特性需看模型能力表。

HTTP 请求整体上限 64 MiB，base64 会膨胀；不是可以直接上传 64 MiB 图片。HEIF native 侧另有压缩输入 256 MiB、单边 8192、像素 64×1024² 的上限，HTTP 层可能先拒绝。smart resize/patch 限制不能替代解码前尺寸防护。

解码失败时先核对文件 magic、内容是否完整、声明格式与真实内容、native codec 文件是否齐全，不靠改 MIME 掩盖错误。

## Embedding

请求 `/v1/embeddings`，model 使用 `/v1/models` 中对应 Embedding 项。权重按需从 GGUF/SSD 载入，默认 idle timeout 60 秒后释放；0 表示常驻。没有额外的 RAM 权重 mirror。服务就绪时 resident=0 可以是正常 lazy 状态。

图片和 Embedding 会争用 GPU、runtime 空间及执行 gate，Decode 下降不必然是数值错误。[发布矩阵](../../evidence/benchmarks/2026-10-09-release-0.10.0-matrix.md)分开报告普通文本、Embedding overlap 与 Vision overlap。单次请求通过不足以证明并发录制安全。
