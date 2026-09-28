# Distribution profile snapshot

This directory is the sanitized, source-controlled baseline for the MoE4All Agent profile. It
captures plugin composition and portable behavioral overrides without copying credentials,
sessions, memory databases, logs, attachments, public addresses, or machine-specific networking
state.

- `profile/package.json` records the maintained MoE4All plugin set. The release builder resolves
  these packages from tested submodule tarballs rather than updating them from the marketplace.
- `profile/cordis.patch.yml` records the portable configuration overrides.
- `profile/stubs/hf-transformers-stub` preserves the current Mneme installation strategy while
  embeddings are supplied by the MoE4All OpenAI-compatible endpoint.

The source for each bundled component is a Git submodule under `../plugins`; the root gitlinks and
each repository's `UPSTREAM.md` provide the exact source and attribution record. This profile is a
distribution baseline; machine-specific product settings remain outside source control.
