---
kind: reference
status: current
scope: submission-protocol
last_verified: 2026-10-19
verified_commit: working-tree
---

# MoE4All Text Submission Protocol

## Scope

`MOE4ALL-SUBMISSION/1` is the transport and storage envelope for public benchmark submissions
(`.m4b`) and diagnostic error submissions (`.m4e`). Both are UTF-8 text, remain readable in a pull
request or text editor, survive copy/paste, and carry one lossless compact JSON payload.

The envelope version, payload schema and delivery protocol are independent:

- Envelope: `MOE4ALL-SUBMISSION/1`.
- Benchmark payload: `benchmark/1`, extension `.m4b`.
- Error payload: `error/1`, extension `.m4e`.
- Delivery: GitHub PR or the MoE4All submission API v1. Delivery does not change the file bytes.

The website integration defines the HTTP flow. The wizard creates a file-bound upload session,
opens a same-origin browser confirmation page, polls with a private session credential, uploads the
exact original bytes, and stores the returned private receipt separately. The default API base is
`https://api.moe4all.com/v1`; this address is reserved until the cloud service is deployed. An
explicit loopback `MOE4ALL_SUBMISSION_API_BASE=http://127.0.0.1:<port>/v1` is accepted for local
development, while other clear-text origins are rejected.

## Wire Format

Every file uses LF line endings when produced. Decoders must also accept CRLF and bare CR copied
from text applications. No UTF-8 BOM is written.

```text
MOE4ALL-SUBMISSION/1
kind: benchmark
schema: benchmark/1
id: 20261019T010203.004Z-00112233445566778899aabbccddeeff-000
created-at: 2026-10-19T01:02:03.004Z
engine: 1.0.3
encoding: gzip+base64
decoded-bytes: 6977
compressed-bytes: 2495
sha256: <64 lowercase hexadecimal characters>
summary-count: 4
summary-001: model: example.gguf | qwen35moe | 23.87 GiB
summary-002: device: Example GPU | 23.98 GiB
summary-003: profile: standard
summary-004: result 6K: prefill=3333.40 | x1=115.97 tok/s

-----BEGIN MOE4ALL PAYLOAD-----
<RFC 4648 standard Base64, wrapped at 76 ASCII characters>
-----END MOE4ALL PAYLOAD-----
```

Header names are lowercase ASCII letters, digits and hyphens. Duplicate headers, unknown kinds,
schema/kind mismatches, text after the end marker and malformed Base64 are errors. Header order is
canonical as shown; decoders need not require the order.

The only payload encoding in envelope version 1 is:

1. Serialize the payload as compact UTF-8 JSON with no insignificant whitespace.
2. Calculate SHA-256 over those exact decoded JSON bytes.
3. Compress those bytes as one gzip member using the highest ordinary compression level.
4. Encode the gzip bytes with RFC 4648 standard Base64 including normal `=` padding.
5. Wrap Base64 at 76 characters. The final payload line and end marker end with LF.

`decoded-bytes`, `compressed-bytes` and `sha256` are mandatory and must match. Encoders and decoders
limit both the complete text envelope and the decompressed JSON to 128 KiB, preventing decompression
bombs. JSON `NaN`, positive infinity and negative infinity are forbidden.

## Readable Summary

The header summary is a deterministic projection of the decoded payload, never a second source of
truth. Validators regenerate every `summary-NNN` line and reject a mismatch. Values are single-line,
control characters and repeated whitespace are normalized, and long description/error fields are
truncated only in the summary. The compressed payload remains lossless.

Benchmark summaries include model, architecture, model size, device, VRAM, profile and one compact
result line per measured context. Error summaries include failure kind, model/device when known,
the user description and final error. Rates use two digits after the decimal point; byte capacities
use GiB with two digits.

## Benchmark Payload

The decoded object retains benchmark JSON schema version 1. Its `id` equals the filename stem and
its `created_at_utc` selects `benchmark-database/v1/YYYY-MM/`. GitHub submissions add exactly one
new immutable `.m4b` file. A corrected measurement receives a new ID and file.

The launch wizard uses GitHub PR submission when the user has GitHub access. Without GitHub it uses
the website submission API. A failed or unavailable upload always preserves the local `.m4b` and
does not change a successful benchmark exit code.

Automatic benchmark suites emit this envelope on every supported platform. The Windows launch
wizard also converts its manual single-test result to the same envelope before offering upload.
Linux and macOS manual single-test upload is not exposed until their equivalent conversion and
review flow is implemented; they must not submit the older raw JSON as protocol data.

## Error Payload

Error reports never enter GitHub Issues or the benchmark database. Both fatal-process reports and
user-triggered runtime reports use `.m4e` and the website submission API, which forwards each
accepted report to a public PR under `error-database/v1/YYYY-MM/`.

The compact upload payload contains basic engine/model/hardware/launch information, the user-edited
description, startup log lines, the final error chain for fatal failures and the last 12 non-speed
diagnostic lines. It excludes live speed tables, prompts, generated responses, images, API keys,
hostnames, serial numbers and absolute local paths. Local rolling evidence remains separate and may be
larger; it is never embedded wholesale in `.m4e`.

## CI And Independent Decoding

The engine provides:

```text
infr submission verify <file.m4b-or-m4e>
infr submission decode <file> [--output decoded.json]
```

CI intentionally uses the independent, standard-library-only decoder instead of trusting the Rust
producer:

```text
python3 scripts/moe4all_submission.py verify <file>
python3 scripts/moe4all_submission.py decode <file> --output decoded.json
python3 scripts/validate-benchmark-submission.py <file.m4b>
```

Benchmark-only pull requests bypass compilation only when the diff adds exactly one path matching
`benchmark-database/v1/YYYY-MM/<id>.m4b`. CI verifies the armor, length fields, gzip, SHA-256,
payload JSON, generated summary, filename identity and benchmark schema.

## Website Decoder

The website must follow the same byte order and limits. In a browser it may:

1. Parse headers and concatenate the Base64 lines.
2. Decode with `atob` into bytes.
3. Decompress with `new DecompressionStream("gzip")`.
4. reject more than 128 KiB of decoded data;
5. verify `decoded-bytes` and SHA-256 with `crypto.subtle.digest("SHA-256", bytes)`;
6. decode UTF-8 with a fatal `TextDecoder`, parse strict JSON, and regenerate the summary.

The envelope permits exactly one gzip member and no trailing compressed bytes. Because the browser
`DecompressionStream` API does not expose how many input bytes it consumed, a browser using only
that API may label the result decoded but verification pending; exact member consumption is then
confirmed by Cloudflare's strict decoder. A browser decoder/library that exposes consumed input may
perform the same check locally.

The website may display the readable header immediately, but must not mark the submission verified
or store it until all payload and schema checks pass. Cloudflare must repeat server-side validation;
browser validation alone is not a trust boundary.
