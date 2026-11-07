#!/usr/bin/env python3
"""Generate original PyTorch Melo phase goldens for INFR native CPU validation.

Uses real upstream inference modules at a pinned commit, with zero BERT,
deterministic duration (SDP ratio=0), and explicitly zero waveform noise. Only
the training-only monotonic alignment import is stubbed; no inference module is
replaced. Emits little-endian [time,channel] F32 tensors and a short PCM WAV.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import types
import wave

from convert_melotts_chinese import DEFAULT_ROOT, MODEL_REVISION, inside, download, sha256

UPSTREAM_REVISION = "209145371cff8fc3bd60d7be902ea69cbdb7965a"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-id", required=True)
    p.add_argument("--work-root", type=Path, default=DEFAULT_ROOT)
    args = p.parse_args()
    if any(c in args.task_id for c in "/\\:") or ".." in args.task_id:
        p.error("Invalid task-id")
    root = args.work_root.resolve()
    temp = inside(root, root / "tmp" / "infr" / args.task_id)
    for key in ("TEMP", "TMP", "TMPDIR"):
        os.environ[key] = str(temp)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["TORCH_HOME"] = str(root / "build-cache" / "dependencies" / "torch")
    source = inside(root, root / "inputs" / "models" / ("melotts-chinese-" + MODEL_REVISION[:12]))
    upstream = source / ("upstream-" + UPSTREAM_REVISION[:12])
    package = upstream / "melo"
    package.mkdir(parents=True, exist_ok=True)
    assets = []
    for name in ("__init__.py", "models.py", "modules.py", "attentions.py", "commons.py", "transforms.py"):
        url = f"https://raw.githubusercontent.com/myshell-ai/MeloTTS/{UPSTREAM_REVISION}/melo/{name}"
        download(url, package / name)
        assets.append({"path": name, "url": url, "sha256": sha256(package / name)})
    (upstream / "sources.json").write_text(json.dumps({"commit": UPSTREAM_REVISION, "files": assets}, indent=2), encoding="utf-8")
    sys.path.insert(0, str(upstream))
    import torch
    import numpy as np
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1729)
    stub = types.ModuleType("melo.monotonic_align")
    def training_only(*_a, **_kw):
        raise RuntimeError("Training alignment is intentionally unavailable in inference reference")
    stub.maximum_path = training_only
    sys.modules["melo.monotonic_align"] = stub
    models = importlib.import_module("melo.models")
    commons = importlib.import_module("melo.commons")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    state = torch.load(source / "checkpoint.pth", map_location="cpu", weights_only=True)["model"]
    model = models.SynthesizerTrn(len(config["symbols"]), config["data"]["filter_length"] // 2 + 1,
        32, n_speakers=config["data"]["n_speakers"], num_languages=state["enc_p.language_emb.weight"].shape[0],
        num_tones=state["enc_p.tone_emb.weight"].shape[0], **config["model"])
    model.load_state_dict(state, strict=True)
    model.eval()
    outdir = inside(root, root / "evidence" / "infr" / args.task_id / "melo-reference")
    outdir.mkdir(parents=True, exist_ok=True)
    records = []
    def save(name, tensor, layout="BCT"):
        a = tensor.detach().cpu().float().numpy()
        original_shape = list(a.shape)
        if a.ndim == 3:
            a = a[0].T if layout == "BCT" else a[0]
        a = np.ascontiguousarray(a, dtype="<f4")
        path = outdir / (name + ".f32")
        a.tofile(path)
        records.append({"name": name, "path": str(path), "file": path.name, "shape": list(a.shape),
                        "torch_shape": original_shape, "bytes": path.stat().st_size, "sha256": sha256(path)})
    token_ids = {s: int(n) for s, n in (line.split() for line in (source / "tokens.txt").read_text(encoding="utf-8").splitlines())}
    # The actual packaged lexicon entry: 你好 n i h ao 3 3 3 3.
    phones = [0, token_ids["n"], 0, token_ids["i"], 0, token_ids["h"], 0, token_ids["ao"], 0, token_ids["."], 0]
    tones = [0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 0]
    languages = [0 if i % 2 == 0 else 3 for i in range(len(phones))]
    x = torch.tensor([phones], dtype=torch.long)
    tone = torch.tensor([tones], dtype=torch.long)
    language = torch.tensor([languages], dtype=torch.long)
    length = torch.tensor([len(phones)], dtype=torch.long)
    bert = torch.zeros(1, 1024, len(phones))
    ja_bert = torch.zeros(1, 768, len(phones))
    handle = model.enc_p.encoder.register_forward_pre_hook(lambda _mod, inp: save("embedding", inp[0]))
    with torch.inference_mode():
        g = model.emb_g(torch.tensor([1])).unsqueeze(-1)
        encoded, mean, logs, mask = model.enc_p(x, length, tone, language, bert, ja_bert, g=g)
        handle.remove()
        save("encoder", encoded); save("stats", torch.cat([mean, logs],dim=1))
        save("prior_mean", mean); save("prior_logs", logs)
        logw = model.dp(encoded, mask, g=g)
        save("logw", logw)
        durations = torch.ceil(torch.exp(logw) * mask)
        save("durations", durations)
        ylength = torch.clamp_min(torch.sum(durations, [1, 2]), 1).long()
        ymask = commons.sequence_mask(ylength, None).unsqueeze(1).to(mask.dtype)
        path = commons.generate_path(durations, mask.unsqueeze(2) * ymask.unsqueeze(-1))
        expanded_mean = torch.matmul(path.squeeze(1), mean.transpose(1,2)).transpose(1,2)
        expanded_logs = torch.matmul(path.squeeze(1), logs.transpose(1,2)).transpose(1,2)
        save("expanded_mean", expanded_mean); save("expanded_logs", expanded_logs)
        z = expanded_mean  # z_p = mean + random * exp(logs) * noise_scale; noise_scale=0.
        save("prior", z)
        stage = 0
        for flow in reversed(model.flow.flows):
            z = flow(z, ymask, g=g, reverse=True)
            if not isinstance(flow, models.modules.Flip):
                save(f"flow.{3-stage}", z); stage += 1
        save("reverse_flow_output", z)
        actual = model.dec(z * ymask, g=g)
        decoder = model.dec
        y = decoder.conv_pre(z * ymask) + decoder.cond(g)
        save("dec.pre", y)
        for i in range(decoder.num_upsamples):
            y = decoder.ups[i](torch.nn.functional.leaky_relu(y, 0.1))
            save(f"decoder_upsample_{i}", y)
            y = sum(decoder.resblocks[i * decoder.num_kernels + j](y) for j in range(decoder.num_kernels)) / decoder.num_kernels
            save(f"dec.{i}", y)
        y = torch.tanh(decoder.conv_post(torch.nn.functional.leaky_relu(y)))
        torch.testing.assert_close(y, actual, atol=1e-7, rtol=1e-6)
        save("waveform", actual)
    audio = np.clip(actual[0,0].numpy() * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(outdir / "nihao-pytorch-reference.wav"), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(44100); wav.writeframes(audio.tobytes())
    report = {"upstream_revision": UPSTREAM_REVISION, "source": assets,
              "checkpoint_sha256": sha256(source / "checkpoint.pth"), "torch": torch.__version__,
              "num_threads": 2, "text": "你好。", "phones": phones, "tones": tones, "languages": languages,
              "noise_scale": 0, "sdp_ratio": 0, "speaker_id": 1, "sample_rate": 44100,
              "samples": len(audio), "tensors": records,
              "reference_scope": "Original PyTorch inference modules; training-only alignment import stubbed; no inference stubs"}
    (outdir / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    names = {"embedding","encoder","stats","logw","prior","flow.3","flow.2","flow.1","flow.0","dec.pre","dec.0","dec.1","dec.2","dec.3","dec.4","waveform"}
    fixture = {"phones": phones, "tones": tones, "languages": languages,
        "stages": [{"name": r["name"], "file": r["file"], "t": r["shape"][0], "c": r["shape"][1]}
                   for r in records if r["name"] in names]}
    (outdir / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n",encoding="utf-8")
    print(json.dumps({"path": str(outdir), "samples":len(audio), "tensors":len(records)}, indent=2))


if __name__ == "__main__":
    main()
