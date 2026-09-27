# MoE4All Harness development environment

This directory keeps the maintained DSH source and bundled plugins separate from the production DSH Desktop installation.

- `dsh/` is the `dsh-v0.1.1-rc.2` source checkpoint maintained in `Headmaster218/dsh-moe4all`.
- `plugins/` contains independent MoE4All plugin repositories.
- `dev-state/` is an ignored, isolated `DSH_HOME`. It never points at the production home.
- `runtime/` is an ignored Node and pnpm toolchain used for repeatable local runs.

Each maintained repository starts with an exact upstream source checkpoint. Upstream commit history is intentionally not imported; `UPSTREAM.md` records the original repository, release or commit, license, and attribution before MoE4All changes begin.

There are two user-facing entry points. Double-click the first one after cloning or after changing DSH or plugin source:

```text
harness\1-Build-or-Initialize.cmd
```

It initializes the isolated profile when needed, builds DSH, packs every local plugin, and installs the resulting packages into that profile. It does not start DSH.

Double-click the second one when you only want to start the already-built development harness:

```text
harness\2-Start.cmd
```

The equivalent PowerShell commands are:

```powershell
.\harness\scripts\Build-DevHarness.ps1
.\harness\scripts\Run-DevHarness.ps1
```

The build command preserves the existing isolated configuration. Use `Build-DevHarness.ps1 -RefreshConfig` only when you intentionally want to copy the current production settings into it again.

The initializer copies settings, credentials, the profile patch, and the workspace registry. It deliberately does not copy sessions, attachments, session projection caches, installed `node_modules`, or remote-device authorization state. Existing workspace records continue to point at their real directories; project files are not duplicated.
