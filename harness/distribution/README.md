# Distribution profile snapshot

This directory is the sanitized, source-controlled form of the locally used DSH web profile.
It captures plugin composition and behavioral overrides without copying credentials, sessions,
memory databases, logs, attachments, public addresses, or machine-specific networking state.

- `profile/package.json` records the active plugin set.
- `profile/cordis.patch.yml` records the portable configuration overrides.
- `profile/stubs/hf-transformers-stub` preserves the current Mneme installation strategy while
  embeddings are supplied by the MoE4All OpenAI-compatible endpoint.

The source for every third-party package is under `../plugins` and pinned by
`../sources.lock.json`. This profile is an inventory baseline; workspace wiring and product
configuration belong to the later MoE4All distribution shell.
