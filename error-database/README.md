# Error report database

This directory stores immutable public MoE4All error reports submitted as
`error-database/v1/YYYY-MM/<id>.m4e`.

Each pull request adds exactly one text submission. CI verifies the envelope,
checksums, decoded `error/1` payload, filename identity, privacy declarations and
directory month. Reports can contain incomplete or unverified user observations;
merging records the report and does not by itself confirm or fix the issue.
