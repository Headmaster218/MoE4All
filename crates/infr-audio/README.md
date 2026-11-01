# Native transcription

`infr-audio` runs SenseVoice Small through INFR's existing `Backend` and graph
operators. The initial implementation selects the CPU backend; model graphs,
audio decoding and HTTP transport do not depend on that selection. A future
device backend needs to implement the graph operations before it can be selected.
Unsupported devices fail explicitly.

No ONNX Runtime, sherpa, GGML, Python or FFmpeg is required at runtime. Download
the [SenseVoice Small Q8_0 GGUF](https://huggingface.co/lovemefan/sense-voice-gguf/blob/main/sense-voice-small-q8_0.gguf)
separately (about 278 MiB). Models and recordings are not included in the binary.

## Use

```sh
infr transcribe sense-voice-small-q8_0.gguf --file recording.wav --language auto
infr serve --transcription-model sense-voice-small-q8_0.gguf --addr 127.0.0.1:1234
# Or append the same transcription flags to an existing chat-model serve command.
```

The three launch wizards expose the same optional service under API settings.
CPU threads default to 4. Weights load on the first audio request and unload
after 300 idle seconds (`--transcription-idle-timeout`; 0 keeps them resident).
The audio worker and queue are independent of chat/vision/embedding scheduling.

```sh
curl http://127.0.0.1:1234/v1/audio/transcriptions \
  -F model=sense-voice-small-q8_0 -F file=@recording.wav -F language=auto
```

The model ID is the GGUF filename stem, also listed in `/v1/models`.
Supported multipart fields are `file`, `model`, `language`, `response_format`
(`json` or `text`) and `stream` (`true` or `false`). Languages: `auto`, `zh`,
`en`, `yue`, `ja`, `ko`. Server authentication applies to this endpoint too.
Uploads default to `<installation>/data/audio-upload`; set
`--transcription-upload-dir` to use another directory. Temporary uploads are
removed after the worker releases them, including cancellation/error paths.
At most four requests can upload, wait or execute at once; additional requests
receive HTTP 429 before creating a file. Configured request deadlines also cover
upload and queue time. One audio inference runs at a time.

## Audio and resource bounds

The initial decoder supports RIFF/WAVE integer PCM and IEEE float audio,
downmixes to mono and resamples to 16 kHz. Compressed audio, RF64, MP3, AAC and
browser WebM recordings require conversion before submission. Unsupported
containers produce an explicit error. HTTP multipart requests are limited to
128 MiB, including framing and parameters.

Long recordings are read incrementally, split at a nearby silence/zero crossing
when available, and processed in chunks of at most 28 seconds. No whole-file
waveform or whole-recording attention matrix is allocated. Streaming uses a
bounded response channel and does not accumulate the full transcript. JSON/text
responses necessarily retain their returned text. Acoustic boundaries can split
speech; this implementation has no separate VAD or cross-chunk decoder history.

`stream=true` returns SSE `transcription.text.delta` events containing each
completed chunk and its audio start/end, then `transcription.done`. Those are
chunk boundaries, not word alignments or a live microphone streaming protocol.
Keepalive comments continue while the worker is busy or queued. Cancellation is
checked between model layers and CTC blocks; an already running operator completes
before its resources are reclaimed. Transcript/audio content is never written to tracing.

The compatibility profile is explicitly restricted to official SenseVoice Small
dimensions/vocabulary. Missing CMVN/LayerNorm/blank metadata in this GGUF uses
the documented official Small defaults; conflicting metadata is rejected.
Frontend and model reference identities are recorded beside the implementation
in `src/frontend.rs` and `src/sensevoice_gguf.rs`.

## Extension points

- `AudioDecoder` supplies samples without coupling codecs to inference.
- `AcousticModel` owns the frontend and model graph; it receives a selected
  `Box<dyn Backend>` rather than allocating device resources itself.
- `BackendFactory` / `ModelFactory` provide lazy construction and device choice.
- `Transcriber` is the request-level interface, shared by CLI and HTTP.
- Resource leases cover each request; model eviction never interrupts inference.

Future GPU factories must obtain weights/runtime through that device's shared
resource manager and unified arena, rather than adding a second independent
VRAM budget. No GPU allocations are made by this CPU implementation.

CPU graph support adds generic stateless ReLU and depthwise convolution. Existing
LLM convolution, causal attention and persistent KV semantics remain unchanged.
Metal/Vulkan currently return explicit unsupported errors for the two new ops.
