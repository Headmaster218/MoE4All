#!/usr/bin/env python3
"""Reproducible, CPU-only Melo Chinese input acquisition and native GGUF conversion.

Runtime has no Python dependency. Conversion needs an existing CPU-capable torch
and numpy installation; it never installs packages. Inputs, logs and caches use
--work-root; --output-dir selects the ready-to-use model directory. Example:
  python convert_melotts_chinese.py --work-root ./voice-setup --output-dir ./models/MeloTTS-Chinese
Use --download-only to acquire/verify original assets without importing torch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import uuid
from datetime import datetime, timezone

MODEL_REPO = "myshell-ai/MeloTTS-Chinese"
MODEL_REVISION = "af5d207a364ea4208c6f589c89f57f88414bdd16"
FRONTEND_REPO = "csukuangfj/vits-melo-tts-zh_en"
FRONTEND_REVISION = "a0d5c6a264c0ef92d70d8661d8cc502d79627cd6"
CONVERSION = "native-f32-zero-bert-dp-v1"
DEFAULT_ROOT = Path(os.environ.get("MOE4ALL_WORK_ROOT", str(Path(__file__).resolve().parent.parent / "local-work")))
EXPECTED_SHA256 = {
    "checkpoint.pth": "a74e9eadffff065c75eb6dfa040efa72cad23e72cfea70d39190bc174fb97093",
    "config.json": "d58b5acdab89ad2bbd65325affab309ae3cb964834b02f9a60587474e81c8bb9",
    "README.md": "40b5b860bacbb3947cbef838719e660d2fe4961eba5b484290b4bedc0253e208",
    "lexicon.txt": "7236884b02435ac5d10cf69b4be40a61b45aa676b5300f0e412f185748fee528",
    "tokens.txt": "d18664a7e12bd7ea1022ddaf951e534e136815016c5a809d6b64156bffb4369d",
    "LICENSE": "88a50e5a02bbc2a5c2f084dc19da751aa97b1690f5fda76cd8005c8634d1ca70",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(root, path):
    path = Path(path).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Output is outside local-work: {path}")
    return path


def download(url, destination):
    """TLS-verified curl, explicit final path, no HF/user-directory caches."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return
    part = destination.with_name(destination.name + ".part")
    curl = shutil.which("curl.exe" if os.name == "nt" else "curl")
    if not curl:
        raise RuntimeError("curl is required for downloading the original voice assets; no software is installed automatically")
    subprocess.run([curl, "--fail", "--location",
                    "--retry", "3", "--output", str(part), url], check=True)
    part.replace(destination)


def hf_url(repo, revision, filename):
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def acquire(inputs):
    assets = []
    for filename in ("checkpoint.pth", "config.json", "README.md"):
        assets.append((filename, hf_url(MODEL_REPO, MODEL_REVISION, filename)))
    for filename in ("lexicon.txt", "tokens.txt", "LICENSE"):
        assets.append((filename, hf_url(FRONTEND_REPO, FRONTEND_REVISION, filename)))
    for filename, url in assets:
        print(f"Input: {filename}", flush=True)
        download(url, inputs / filename)
        if sha256(inputs / filename) != EXPECTED_SHA256[filename]:
            raise RuntimeError(f"Input checksum mismatch: {filename}; preserving file for diagnosis")
    # A model's README and its converted voice package's LICENSE are preserved,
    # rather than assuming the inference runtime license covers its voice data.
    manifest = {"model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
                "frontend_repo": FRONTEND_REPO, "frontend_revision": FRONTEND_REVISION,
                "license": "MIT (upstream model card and converted voice LICENSE)",
                "assets": [{"path": name, "url": url, "bytes": (inputs / name).stat().st_size,
                            "sha256": sha256(inputs / name)} for name, url in assets]}
    previous = inputs / "sources.json"
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        if old != manifest:
            raise RuntimeError("Original input changed; refusing to overwrite recorded identity")
    previous.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def gguf_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, metadata, tensors):
    """GGUF v3 F32; row-major torch shape reversed in GGUF's dimension order."""
    entries = []
    for key, value in metadata.items():
        if isinstance(value, int):
            entry = struct.pack("<II", 4, value)  # GGUF_TYPE_UINT32
        else:
            entry = struct.pack("<I", 8) + gguf_string(str(value))
        entries.append(gguf_string(key) + entry)
    infos, offset = [], 0
    arrays = []
    for name, tensor in tensors.items():
        arr = tensor.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
        shape = arr.shape or (1,)
        infos.append(gguf_string(name) + struct.pack("<I", len(shape))
                     + struct.pack("<" + "Q" * len(shape), *reversed(shape))
                     + struct.pack("<IQ", 0, offset))  # GGML_TYPE_F32
        arrays.append(arr)
        offset += arr.nbytes
        offset = (offset + 31) // 32 * 32
    partial = path.with_suffix(".gguf.part")
    with partial.open("wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
        for entry in entries:
            f.write(entry)
        for info in infos:
            f.write(info)
        f.write(b"\0" * (-f.tell() % 32))
        for array in arrays:
            f.write(array.tobytes(order="C"))
            f.write(b"\0" * (-f.tell() % 32))
    partial.replace(path)


def convert(inputs, output, source_manifest):
    import torch
    import numpy as np
    torch.set_num_threads(1)
    checkpoint = torch.load(inputs / "checkpoint.pth", map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    config = json.loads((inputs / "config.json").read_text(encoding="utf-8"))
    tensors, removed, folded = {}, [], []
    for name, tensor in state.items():
        if name.startswith(("enc_q.", "sdp.", "dp_discriminator.", "discriminator.")):
            removed.append(name)
            continue
        if name in ("enc_p.bert_proj.weight", "enc_p.ja_bert_proj.weight"):
            removed.append(name)
            continue
        if name.endswith(".weight_g"):
            removed.append(name)
            continue
        if name.endswith(".weight_v"):
            g_name = name[:-1] + "g"
            if g_name not in state:
                raise RuntimeError(f"Missing weight_norm magnitude: {g_name}")
            v, g = tensor.float(), state[g_name].float()
            dims = tuple(range(1, v.ndim))
            norm = torch.linalg.vector_norm(v, dim=dims, keepdim=True)
            if torch.any(norm == 0):
                raise RuntimeError(f"Zero weight_norm norm in {name}")
            weight = v * (g / norm)
            oracle = torch._weight_norm(v, g, 0)
            delta = float((weight - oracle).abs().max())
            if not torch.allclose(weight, oracle, atol=1e-6, rtol=1e-5):
                raise RuntimeError(f"weight_norm mismatch: {name}: {delta}")
            target_name = name[:-2]
            folded.append({"name": target_name, "max_abs_error_vs_torch_weight_norm": delta})
            tensors[target_name] = weight
        elif isinstance(tensor, torch.Tensor):
            if not tensor.is_floating_point():
                raise RuntimeError(f"Unexpected non-floating inference tensor {name}: {tensor.dtype}")
            tensors[name] = tensor.float()
    for key in ("enc_p.bert_proj.bias", "enc_p.ja_bert_proj.bias", "emb_g.weight",
                "dp.proj.weight", "dec.conv_pre.weight"):
        if key not in tensors:
            raise RuntimeError(f"Missing expected inference tensor {key}")
    if any(not bool(torch.isfinite(t).all()) for t in tensors.values()):
        raise RuntimeError("Non-finite inference weights")
    metadata = {"general.architecture": "melo_tts", "general.name": "MeloTTS Chinese native CPU",
                "general.alignment": 32, "general.file_type": 0,
                "melo.config": json.dumps(config, separators=(",", ":")),
                "melo.sample_rate": config["data"]["sampling_rate"],
                "melo.speaker_id": config["data"]["spk2id"]["ZH"],
                "melo.lang_id": 3, "melo.tone_start": 0,
                "melo.bert_mode": "zero-with-projection-bias", "melo.sdp_ratio": "0",
                "melo.frontend": "melo-lexicon-longest-match-v1",
                "melo.source.repo": MODEL_REPO, "melo.source.revision": MODEL_REVISION,
                "melo.source.checkpoint_sha256": sha256(inputs / "checkpoint.pth"),
                "melo.source.frontend_revision": FRONTEND_REVISION,
                "melo.conversion": CONVERSION, "melo.license": "MIT"}
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "melotts-chinese-f32.gguf"
    write_gguf(destination, metadata, tensors)
    for name in ("config.json", "tokens.txt", "lexicon.txt", "LICENSE", "README.md", "sources.json"):
        shutil.copyfile(inputs / name, output / name)
    # Frontend vocabulary properties are explicit, independently versioned data.
    (output / "frontend.json").write_text(json.dumps({"language_id": 3, "tone_start": 0,
        "add_blank": config["data"]["add_blank"], "blank_id": 0,
        "num_tones": int(tensors["enc_p.tone_emb.weight"].shape[0]),
        "unknown_word_policy": "error", "sample_rate": config["data"]["sampling_rate"]}, indent=2) + "\n", encoding="utf-8")
    # Explicit, inspectable technical-term spellings. This is not a general
    # English G2P model: unknown ordinary words still fail in the native frontend.
    custom_path = output / "custom-lexicon.txt"
    if not custom_path.exists():
        entries = {}
        for line in (inputs / "lexicon.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 3:
                half = (len(parts) - 1) // 2
                entries.setdefault(parts[0].lower(), (parts[1:1+half], parts[1+half:]))
        custom = ["# Explicit letter-by-letter spellings for engineering terms.",
                  "# Format: word phone... tone...; editable without changing weights."]
        for term in ("INFR", "Jarvis", "Codex", "MeloTTS", "Vulkan", "CUDA", "GGUF", "ONNX", "ASR", "TTS", "CPU", "GPU", "VRAM", "PCB", "API", "VR", "XR"):
            if term.lower() in entries:
                continue
            try:
                pieces = [entries[c] for c in term.lower()]
            except KeyError as error:
                raise RuntimeError(f"Cannot create explicit technical spelling for {term}") from error
            phones = [p for entry in pieces for p in entry[0]]
            tones = [p for entry in pieces for p in entry[1]]
            custom.append(" ".join([term.lower(), *phones, *tones]))
        custom_path.write_text("\n".join(custom) + "\n", encoding="utf-8")
    report = {"conversion": CONVERSION, "time_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
              "source": source_manifest, "output": str(destination), "bytes": destination.stat().st_size,
              "sha256": sha256(destination), "tensor_count": len(tensors),
              "tensor_bytes": sum(t.numel() * 4 for t in tensors.values()),
              "tensor_layout": "torch contiguous row-major bytes; GGUF dimensions reversed",
              "folded_weight_norm": folded, "removed_tensors": removed,
              "tensors": [{"name": k, "torch_shape": list(v.shape), "bytes": v.numel() * 4}
                          for k, v in tensors.items()],
              "validation": "finite F32 weights; folded weights match torch._weight_norm; synthesis parity not yet tested"}
    (output / "conversion.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def resolve_paths(args):
    """Explicit output is portable; the legacy developer work layout remains supported."""
    task_id = args.task_id or (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-melo-setup-") + uuid.uuid4().hex[:8])
    if not task_id or any(c in task_id for c in "/\\:") or ".." in task_id:
        raise ValueError("task-id must be a single safe directory component")
    root = args.work_root.resolve()
    identity = "melotts-chinese-" + MODEL_REVISION[:12]
    output = args.output_dir.resolve() if args.output_dir else inside(root, root / "test-cache" / "models" / (identity + "-" + CONVERSION))
    return root, task_id, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--task-id", help="Optional unique name for setup logs and temporary files")
    parser.add_argument("--output-dir", type=Path, help="Save the native GGUF and all voice assets in this folder")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    try:
        root, task_id, output = resolve_paths(args)
    except ValueError as error:
        parser.error(str(error))
    if not args.download_only:
        missing = [name for name in ("torch", "numpy") if importlib.util.find_spec(name) is None]
        if missing:
            parser.error("Conversion requires an existing Python environment with " + ", ".join(missing)
                         + ". Follow the release-note setup steps; this script never installs dependencies.")
        if (output / "melotts-chinese-f32.gguf").exists():
            parser.error("Output model already exists; keep the existing model or choose a different --output-dir")
    temp = inside(root, root / "tmp" / "infr" / task_id)
    temp.mkdir(parents=True, exist_ok=True)
    for key in ("TEMP", "TMP", "TMPDIR"):
        os.environ[key] = str(temp)
    for key, tool in (("HF_HOME", "huggingface"), ("HF_HUB_CACHE", "huggingface/hub"),
                      ("TORCH_HOME", "torch"), ("PIP_CACHE_DIR", "pip"),
                      ("PYTHONPYCACHEPREFIX", "python-pycache")):
        os.environ[key] = str(root / "build-cache" / "dependencies" / tool)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    identity = "melotts-chinese-" + MODEL_REVISION[:12]
    inputs = inside(root, root / "inputs" / "models" / identity)
    evidence = inside(root, root / "evidence" / "infr" / task_id)
    evidence.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 1024 ** 3:
        raise RuntimeError("Need at least 1GiB free for original and derived model assets")
    print(json.dumps({"inputs": str(inputs), "output": str(output), "temp": str(temp)}, indent=2), flush=True)
    manifest = acquire(inputs)
    if args.download_only:
        report = manifest
    else:
        report = convert(inputs, output, manifest)
    (evidence / "melo-conversion.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("output", "bytes", "sha256", "tensor_count") if k in report}, indent=2))


if __name__ == "__main__":
    main()
