# Local DSH changes to migrate

This is the source migration inventory for changes found in the locally used DSH installation and
the related engineering workspace. It deliberately records behavior, not user data or credentials.

| ID | Area | Local behavior | Target source tree | Migration status |
|---|---|---|---|---|
| DESKTOP-PORT | Desktop launcher | Uses fixed port 1633 so the remote entry point remains stable; startup still probes the port and fails clearly if occupied. | future maintained Desktop shell | Not migrated; the current isolated Web harness selects its own development port. |
| REMOTE-EVENTS | Remote Web UI | Adds gated forwarding for legacy `/api/events.mux` and `/api/events.host` WebSockets so approvals and questions reach a paired client. | `plugins/dsh-remote-web-ui-moe4all` | Migrated to the maintained submodule with contract and rewrite tests. |
| REMOTE-TRANSPORT | Remote Web UI | Adds a `createApiClient` fallback because DSH 0.1.1-rc.2 lacks the transport API expected by the plugin. | `plugins/dsh-remote-web-ui-moe4all` | Migrated as a compatibility adapter; remove after the harness and plugin share one transport contract. |
| MNEME-V2 | Memory plugin | Replaces whole-session redistillation with one-call per-turn `distill2`, adds cluster-driven `dream3`, and wires engine/reasoning settings. | `plugins/dsh-mneme-moe4all` | Migrated to the maintained submodule as single-call distill and persistent dream. |
| MNEME-EMBED | Memory/profile | Replaces the unused `@huggingface/transformers` dependency with a local stub because embeddings use the native MoE4All OpenAI-compatible endpoint. | `distribution/profile` | Migrated as an explicit profile override; replace with a native provider contract later. |
| PROFILE-POLICY | Profile | Selects Brave search, Mneme v2 policy, and the dedicated no-reasoning title route. | `distribution/profile/cordis.patch.yml` | Migrated without credentials or machine-specific public URLs. |
| UPLOAD-PATH | File uploader candidate | Sanitizes control characters, NTFS ADS separators and reserved device names; restricts remove/delete to a direct child of a known uploads root. | future `dsh-web-file-uploader-moe4all` repository | Preserved on the archived v1 branch but not yet migrated into a maintained submodule. |
| REMOTE-ENDPOINT | Profile/host | Uses a machine-specific public base URL and Windows port proxy for remote access. | product host configuration | Keep as user configuration/migration input; never ship the machine address as a default. |
| TITLE-DIAG | Diagnostics | Temporary proxy and probes captured title-generation requests and stream behavior. | test fixtures/developer diagnostics | Do not include logs or make the proxy part of production routing. |
| LOCAL-TOOLS | Skills/tools | Local ASR/media scripts, a guarded Bilibili comment workflow, and personal skills exist beside DSH. | optional MoE4All plugins/skills | Review and extract separately; never import secrets, queues, cookies, corpora, or personal workspaces. |

Portable profile settings for memory consolidation, title generation, and search are captured in
`distribution/profile/cordis.patch.yml`. Credentials, memory data, and the machine-specific remote
address remain outside source control.
