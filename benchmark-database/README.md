# Benchmark Database

Automatic `infr bench --suite ...` runs and Windows launch-wizard manual benchmarks write one immutable,
Base64-armored UTF-8 text submission under:

```text
benchmark-database/v1/YYYY-MM/<unique-id>.m4b
```

Submit exactly one newly generated `.m4b` file per pull request. Do not rename an existing result or
edit it in place. Filenames are unique across the entire database, including different months. A
corrected measurement is a new run with a new ID.

The decoded JSON `id` is exactly the filename stem. New automatic results combine their UTC timestamp with
128 bits from the operating system's cryptographic random source; manual results use a full GUID.
All writers use atomic create-new semantics; the automatic writer retries its numbered local
collision suffix, while a manual GUID collision is refused. CI rejects an ID that differs from its
filename and rejects duplicate filenames anywhere in the database.

The automatic suites use real prompt contexts. One same-shape 6K Prefill warmup with different text
runs first and is not scored.
Quick measures 6K once, standard measures 6K and 30K once each, and full measures 6K, 30K and 150K
three times each. Every Decode is one continuous 1536-token request: the first 512 tokens warm the
concurrent scheduler and expert cache, then the remaining 1024 measured tokens are reported as
first, middle and last thirds. For a maximum concurrency of `N`, results include every case from
`x1` through `xN`. A failed case is recorded with its error while the remaining independent cases
continue.

Public results intentionally omit usernames, hostnames, local paths, device serial numbers and
other machine identifiers. Hardware fields that cannot be read portably remain `null` rather than
being guessed.

On Windows, the launch wizard enriches results with SMBIOS memory-module generation/capacity/speed,
the selected GPU's current and maximum PCIe generation/width, theoretical one-way PCIe bandwidth,
and the model volume's non-identifying disk model/media/interface. Its model-file read probe is
read-only and bounded to about one second total. Both application-visible throughput and physical
disk counter bytes are retained, so a Windows file-cache hit is labelled instead of being reported
as SSD throughput.

The text header provides a reviewable summary. The lossless compact JSON is gzip-compressed and
Base64-armored below it; CI independently verifies its SHA-256 and regenerates the summary. See
[`submission-format.md`](../documentation/reference/submission-format.md) for the byte-level format,
decoder commands, website requirements and limits.

Manual results are marked with `"profile": "manual"` and retain their explicit token counts and
raw average. They are useful for targeted comparisons but are not mixed into automatic-suite
aggregates.
