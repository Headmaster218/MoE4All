# INFR Browser Control Plane

`infr-gui` is a small server-hosted web control plane. It stays online while it starts, stops, or
switches a separate `infr serve` worker. A model switch therefore releases the old Vulkan process
and all of its allocations before the next model is loaded.

## Start on Windows

Double-click `Start-INFR-GUI.cmd` in the repository root, or run:

```powershell
& 'D:\AIinfr\infr\crates\infr-gui\Start-INFR-GUI.ps1'
```

The launcher:

1. incrementally builds `infr.exe` and `infr-gui.exe` in release mode;
2. creates `gui-data\admin.key` once and reuses it on later starts;
3. listens on `0.0.0.0:8180` so the page is reachable over ZeroTier;
4. prints the local/ZeroTier URL and management key;
5. runs in the foreground. `Ctrl+C` stops the GUI after gracefully draining its worker.

No scheduled task or auto-start entry is created. To use another listen address:

```powershell
& 'D:\AIinfr\infr\crates\infr-gui\Start-INFR-GUI.ps1' -ListenAddress '192.168.195.1:8180'
```

If Windows Firewall blocks the chosen port, allow that TCP port on the ZeroTier network profile.
The launcher does not modify firewall rules.

## Normal workflow

- Add one or more server directories in **模型库**. There is no automatic disk scan.
- Select a GGUF. Split GGUFs are grouped by their first shard; `mmproj` files are detected but not
  treated as language models.
- Save one or more profiles. Favorites sort first, followed by recently used models.
- Select **重新估算** before loading. The KV estimate uses INFR's actual model layout calculation;
  runtime and available expert-cache figures are conservative pre-load estimates.
- Select **启动 / 切换**. A switch requests graceful shutdown, waits up to 660 seconds for active GPU
  work to drain, and then starts the replacement worker.
- Use **强制停止** only when graceful draining is stuck. It terminates the worker process directly.

The GUI shows worker phase, PID, address, recent logs, and the latest reported Prefill/Decode rates.
Only one worker and one background download are managed at a time.

## Downloads

Use an INFR/HuggingFace model reference such as `org/repo:Q6_K`. The selector supports:

- `https://huggingface.co`
- `https://hf-mirror.com`
- another HuggingFace-compatible HTTP(S) origin

Downloads use the existing `infr pull` cache, resume, checksum, shard, and companion-file logic.
The resulting GGUF is added to the catalog automatically. `HF_TOKEN` can be set in the launcher's
environment for repositories that require credentials.

## Keys and exposure

The GUI management key and the worker's OpenAI-compatible API key are separate:

- `gui-data\admin.key` is passed through `--key-file` and protects every management API. The
  browser stores the entered value in local storage.
- **OpenAI API Key** is passed to the worker through `INFR_API_KEY`, not the process command line.

Traffic is plain HTTP, intended for the encrypted ZeroTier network. Do not expose port 8180 to the
public Internet. Profiles, including a configured worker API key, are stored in
`gui-data\state.json`; restrict that directory to the server account if other local users are not
trusted.

## Current and reserved capabilities

Chat/completion and Embedding GGUF workers are active now. The catalog and profile model keep task,
modality, projector, memory-tier and advanced-config concepts separate so the remaining engines can
be added without replacing the GUI:

- Embedding profiles start INFR's managed `llama.cpp` worker and expose it through INFR's own
  authenticated `/v1/embeddings` endpoint. The runner is auto-discovered, or can be selected per
  profile. CPU and Vulkan are supported; closing/switching the worker unloads its model.
- Rerank remains a reserved task and is rejected until its endpoint exists.
- Vision/mmproj files are discovered and shown, but are not yet passed into an inference worker.
- VRAM and RAM budgets are active. SSD is currently the existing model/mmap storage tier; explicit
  VRAM -> RAM -> SSD policy controls are reserved for the later three-level pager.

Persistent state lives only under `gui-data`, which is ignored by Git. Removing that directory
resets the GUI catalog, profiles, favorites, recents, and generated management key; it does not
delete model files.
