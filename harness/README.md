# MoE4All Harness development environment

This directory keeps the maintained DSH source and bundled plugins separate from the production DSH Desktop installation.

- `dsh/` is the `dsh-v0.1.1-rc.2` source checkpoint maintained in `Headmaster218/dsh-moe4all`.
- `plugins/` contains independent MoE4All plugin repositories.
- `dev-state/` is an ignored, isolated `DSH_HOME`. It never points at the production home.
- `runtime/` is an ignored Node and pnpm toolchain used for repeatable local runs.

Each maintained repository starts with an exact upstream source checkpoint. Upstream commit history is intentionally not imported; `UPSTREAM.md` records the original repository, release or commit, license, and attribution before MoE4All changes begin.

Initialize the isolated profile from the current production configuration:

```powershell
.\harness\scripts\Initialize-DevHarness.ps1
```

Install and build the source workspace, then install the local plugin packages:

```powershell
.\harness\scripts\Install-DevHarness.ps1
```

Run the Web harness from the locally built source checkout:

```powershell
.\harness\scripts\Run-DevHarness.ps1
```

The initializer copies settings, credentials, the profile patch, and the workspace registry. It deliberately does not copy sessions, attachments, session projection caches, installed `node_modules`, or remote-device authorization state. Existing workspace records continue to point at their real directories; project files are not duplicated.
