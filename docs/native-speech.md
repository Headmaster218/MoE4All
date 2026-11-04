# Native CPU speech (optional)

Build `infr-cli` with `--features tts` to include the native Melo implementation.
The default build does not include it. Configuring no speech model keeps normal
chat, embedding and transcription operation unchanged; no speech weights load.
This capability uses an independent CPU service, with no Python runtime,
subprocess, Vulkan model or interference with chat admission slots.

Model weights and frontend assets are external. `--speech-lexicon` / `--lexicon`
name an **asset directory** containing `lexicon.txt`, `tokens.txt`, `config.json`,
matching the converted GGUF. Use the conversion tool's model package together.

```text
infr serve --speech-model /models/melo/model.gguf --speech-lexicon /models/melo/assets --addr 127.0.0.1:8186
infr speak /models/melo/model.gguf --lexicon /models/melo/assets --text "你好，今天开始测试语音合成。" --output /controlled/clip.wav
```

`serve` can combine speech with any of chat, `--embedding-model` and
`--transcription-model`. Omit the chat positional model for auxiliary-only
operation. Speech is CPU-only even when chat uses Vulkan. Use a separate port
and a frozen executable when testing alongside an existing service.

Speech flags: `--speech-threads` (default 4), `--speech-idle-timeout` (300 seconds;
0 keeps weights loaded), `--speech-max-output-seconds` (30, range 1..120).
Weights load lazily and idle unload is owned by the native service. The lexicon
and model are required together, and assets are not downloaded automatically.

`speak` accepts `--format wav|pcm`, `--speaker-id`, `--speed`, `--threads`,
`--max-output-seconds` and `--seed`. Its output path must not already exist.
PCM is signed 16-bit mono little-endian at the model's native sample rate.

## HTTP

`POST /v1/audio/speech` uses existing bearer authentication. `/v1/models` lists
the configured speech model ID (the GGUF filename stem). Send JSON:

```json
{"model":"model","input":"你好，今天开始测试语音合成。","voice":"default","response_format":"wav","speed":1.0}
```

`input` is required (1..2048 characters, at most 8192 UTF-8 bytes). Other fields
are optional. `model` must match when supplied. `voice` is `default` or a numeric
speaker ID supported by the package. `response_format` is `wav` (default) or
`pcm`. `speed` is 0.25..4; optional `noise_scale` is 0..2 and `seed` is an unsigned
64-bit integer. Unknown fields/formats are rejected. This first API returns a
complete clip and does not support streaming.

Success returns `audio/wav` or `audio/pcm`, plus `x-audio-sample-rate` and
`x-audio-channels: 1`. WAV embeds its native rate; raw PCM clients must use the
header. The JSON body limit is 16 KiB. No temporary audio files are created.

One CPU synthesis executes at a time; at most four requests are admitted,
including queued requests. Further requests return 429. HTTP timeout covers body
reading, queueing and synthesis: `serve.request_timeout_secs` when positive,
capped at 600 seconds; otherwise speech uses 120 seconds. Client disconnect,
timeout or process shutdown requests cancellation. Admission and execution
permits remain owned by the worker until synchronous synthesis actually returns.
Output duration is additionally capped by the native service. No configured
speech service returns 503 (a build without the feature omits the endpoint).

No input text or generated audio is written to diagnostic logs by the adapter.
Unsupported frontend text or invalid speaker/model combinations can fail
synthesis; use the package's supported language and speaker definitions.
