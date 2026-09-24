# DSH source inventory

This inventory describes the local DSH installation used as the MoE4All agent baseline on
2026-09-23. Exact fetch metadata lives in `sources.lock.json`.

## Runtime and desktop

| Component | Installed release | Source commit | Local overlay |
|---|---:|---|---|
| DeepSeek Harness | `0.1.1-rc.2` | `b150a551b8d465e31e418e1b2eaf5e79bbb7d28e` | None found in the installed Harness package. |
| DSH Desktop | `0.6.3` | `e589bf688624871d953f8c58418f913c407dfc7e` | Stable port `1633`, with an explicit pre-launch availability check. |

The Desktop repository's root package metadata still says `0.1.1`; the application artifact and
release tag are `0.6.3`, and its bundled Harness dependencies are pinned to `0.1.1-rc.2`.

## Active profile plugins

| Package | Version | Source commit | Local overlay |
|---|---:|---|---|
| `dshmarket` | `1.45.1` | `1664caec99219b4902f1686e2e34614815d38346` | None found. |
| `@deads-inc/dsh-web-search-brave` | `1.0.0` | `10799a8c19ec6d672789257be609750515dbb513` | Selected as the profile search provider. |
| `@dsh-external/dsh-plugin-tts` | `0.3.1` | `ff26f361df77605599896fa2ba76eb875eddc1d2` | The installed lockfile commit is used instead of the similarly named tag. |
| `@linxin666/dsh-remote-web-ui` | `0.3.17` | `46f1616f26525c7b8a782968f02fa97fde2c5790` | DSH 0.1.1 event sockets and transport compatibility adapter. |
| `@modusensus/dsh-mneme` | `0.7.29` | `0b3e99f6ff4948ab4b823914a0a16f4411d25a4c` | `distill2`, `dream3`, config/index/settings wiring, and local embedding dependency stub. |
| `dsh-easyrewrite` | `2.4.1` | `8bfa03c946e81e6159d0c839ffb7a02c6c52c158` | None found. |
| `dsh-plugin-cron-scheduler` | `0.2.7` | `f7a57a64ac26e325ddacc59cd9368e6844657347` | None found. |

Mneme is published from the nested `dsh-mneme` package. The repository root still reports
`0.7.16`, while the nested package and installed artifact correctly report `0.7.29`.

## Reviewed candidate

| Package | Version | Source commit | Local overlay |
|---|---:|---|---|
| `dsh-web-file-uploader` | `0.4.0` | `63d474b2a3aea6b880a6c7f36aa7a1d0680a38a6` | Path confinement, filename hardening, same-origin fence, upload cap, and keep-forever default. |

The uploader is not enabled in the current web profile. It is retained because the local security
work is useful for the future distribution.

## Profile policy

The sanitized profile snapshot keeps the active plugin order, Brave selection, Mneme v2 policy,
title-generation route, and Hugging Face dependency stub. It intentionally excludes credentials,
memory/session state, attachments, logs, and `remote-web-ui.publicBaseUrl`.
