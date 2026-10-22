#!/usr/bin/env python3
"""Render a small, advisory benchmark review using trusted Git report blobs."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

from benchmark_index_sources import collect
from moe4all_submission import MAX_ARMOR_BYTES, decode_text
from submission_validation import validate_payload


MARKER = "<!-- moe4all-benchmark-review:v1 -->"
REPORT_PATH = re.compile(r"benchmark-database/v1/(\d{4}-(?:0[1-9]|1[0-2]))/([A-Za-z0-9][A-Za-z0-9._-]{0,159})\.m4b")
SHA = re.compile(r"[a-f0-9]{40}")
MAX_ROWS = 40
MAX_COMMENT_BYTES = 60000


def normal(value):
    return " ".join(value.casefold().split()) if isinstance(value, str) else ""


def cell(value, limit=100):
    text = " ".join(re.sub(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]", " ", str(value)).split())
    text = text[:limit] + ("..." if len(text) > limit else "")
    text = html.escape(text, quote=False)
    return re.sub(r"[\\|\[\]`*_!{}#~@]", lambda m: f"&#{ord(m[0])};", text)


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def capacity(value):
    # Physical heaps/RAM can exclude reserved bytes; compare nominal GiB capacities.
    return math.ceil(value / (1 << 30)) if number(value) and value >= (1 << 29) else None


def size(value):
    return f"{value / (1 << 30):.1f}G" if number(value) and value > 0 else "未知"


def rate(value):
    return f"{value:.1f}" if number(value) else "—"


def context_label(tokens):
    return f"{int(tokens) // 1024}K" if tokens and tokens % 1024 == 0 else str(int(tokens))


def kv(value):
    text = normal(value)
    match = re.fullmatch(r"some\((.+)\)", text)
    return match[1] if match else text


def model_name(payload):
    name = normal(payload["model"]["name"])
    name = re.sub(r"\.gguf$", "", name)
    return re.sub(r"-\d{5}-of-\d{5}$", "", name)


def same_model(a, b):
    if model_name(a) != model_name(b):
        return False
    left, right = a["model"], b["model"]
    for keys in (("architecture",), ("quantization", "quant")):
        x = next((normal(left.get(k)) for k in keys if left.get(k)), "")
        y = next((normal(right.get(k)) for k in keys if right.get(k)), "")
        if x and y and x != y:
            return False
    x, y = left.get("storage_bytes"), right.get("storage_bytes")
    return not (number(x) and number(y) and x > 0 and y > 0 and max(x, y) / min(x, y) > 1.01)


def gpu_family(gpu):
    name = normal(gpu.get("name"))
    mobile = bool(re.search(r"\blaptop\b|\bmobile\b|\b\d{4}m\b", name))
    for pattern, vendor in ((r"\brx\s+([679])\d{3}\b", "amd-rx"), (r"\b(?:geforce\s+)?rtx\s+([2345])\d{3}\b", "nvidia-rtx"), (r"\barc\s+([ab])\d{3}\b", "intel-arc")):
        match = re.search(pattern, name)
        if match:
            return vendor, match[1], mobile
    return None


def hardware(payload):
    h = payload["hardware"]
    return h.get("gpu") or {}, h.get("ram") or {}


def conditions(payload):
    c = payload["configuration"]
    gpu, _ = hardware(payload)
    backend = normal(c.get("backend"))
    if not backend:
        device = normal(c.get("device") or c.get("configured_device") or gpu.get("configured_device"))
        backend = next((b for b in ("vulkan", "metal", "cpu") if device.startswith(b)), "")
    depth = "real" if c.get("real_context") is True or c.get("depth_mode") == "real" else "synthetic" if c.get("synthetic_depth") is True or c.get("depth_mode") == "synthetic" else normal(c.get("depth_mode"))
    measurement = c.get("decode_measurement") or ("time" if c.get("decode_seconds_per_rep") else "tokens")
    return backend, kv(c.get("kv_type_k")), kv(c.get("kv_type_v")), c.get("mtp"), depth, measurement


def matching_level(current, candidate):
    if not same_model(current, candidate) or conditions(current) != conditions(candidate):
        return None
    a, ar = hardware(current)
    b, br = hardware(candidate)
    av, bv = capacity(a.get("vram_total_bytes")), capacity(b.get("vram_total_bytes"))
    same_vram = av is not None and av == bv
    if normal(a.get("name")) and normal(a.get("name")) == normal(b.get("name")) and same_vram:
        ram = capacity(ar.get("total_bytes"))
        return 1 if ram is not None and ram == capacity(br.get("total_bytes")) else 2
    if same_vram and gpu_family(a) is not None and gpu_family(a) == gpu_family(b):
        return 3
    return 4


def rows(payload):
    result = []
    config = payload["configuration"]
    for context in payload["contexts"]:
        tokens = context["requested_context_tokens"]
        failures = [f for f in context.get("failures", []) if isinstance(f, dict)]
        manual = context.get("manual_result") if payload["profile"] == "manual" else None
        if manual:
            kind = manual["kind"]
            slots = manual.get("parallel", config.get("parallel", 1)) if kind == "decode" else 1
            slots = slots if type(slots) in (int, float) and number(slots) and slots == int(slots) and 0 < slots <= 64 else None
            result.append({"key": (tokens, "manual_" + kind, slots), "kind": kind, "slots": slots, "rate": manual["average_tokens_per_second"], "sections": None, "partial": bool(failures), "note": "单项手动"})
            continue
        if context.get("prefill") is not None:
            partial = any("prefill" in str(f.get("stage", "")) for f in failures)
            result.append({"key": (tokens, "prefill", 1), "kind": "prefill", "slots": 1, "rate": context["prefill"]["tokens_per_second"], "sections": None, "partial": partial, "note": ""})
        elif failures:
            result.append({"key": (tokens, "prefill", 1), "kind": "prefill", "slots": 1, "rate": None, "sections": None, "partial": True, "note": "失败"})
        seen = set()
        for decode in context.get("decode", []):
            parallel = decode["parallel"]
            seen.add(parallel)
            failed = [f for f in failures if f.get("parallel") == parallel]
            requested = config.get("decode_repetitions")
            partial = bool(failed) or number(requested) and len(decode["repetitions"]) < requested
            result.append({"key": (tokens, "decode", parallel), "kind": "decode", "slots": parallel, "rate": decode["overall_tokens_per_second"], "sections": decode["section_mean_tokens_per_second"], "partial": partial, "note": ""})
        for parallel in sorted({f["parallel"] for f in failures if type(f.get("parallel")) is int and f["parallel"] > 0} - seen):
            result.append({"key": (tokens, "decode", parallel), "kind": "decode", "slots": parallel, "rate": None, "sections": None, "partial": True, "note": "失败"})
    return result


def historical(current, row, entries):
    groups = {level: [] for level in range(1, 5)}
    seen = {current["id"]}
    for entry in entries:
        payload = entry["payload"]
        if payload["id"] in seen:
            continue
        seen.add(payload["id"])
        level = matching_level(current, payload)
        if level is None:
            continue
        match = next((r for r in rows(payload) if r["key"] == row["key"] and not r["partial"] and number(r["rate"]) and r["rate"] > 0), None)
        if match:
            groups[level].append({**entry, "rate": match["rate"]})
    for level, matches in groups.items():
        if matches:
            if level >= 3:
                # Relaxing hardware must not turn different GPUs into one average.
                hardware_groups = {}
                for match in matches:
                    gpu, _ = hardware(match['payload'])
                    key = (normal(gpu.get('name')), capacity(gpu.get('vram_total_bytes')),
                           normal((match['payload']['hardware'].get('cpu') or {}).get('name')) if not gpu.get('name') else '')
                    hardware_groups.setdefault(key, []).append(match)
                gpu, ram = hardware(current)
                target_vram, target_ram = capacity(gpu.get('vram_total_bytes')), capacity(ram.get('total_bytes'))

                def closeness(item):
                    key, candidates = item
                    vram_gap = abs(key[1] - target_vram) if key[1] is not None and target_vram is not None else math.inf
                    ram_sizes = [capacity(hardware(e['payload'])[1].get('total_bytes')) for e in candidates]
                    ram_gap = min((abs(value - target_ram) for value in ram_sizes if value is not None and target_ram is not None), default=math.inf)
                    return vram_gap, ram_gap, -len(candidates), str(key)

                matches = min(hardware_groups.items(), key=closeness)[1]
            return level, sorted(matches, key=lambda e: e["payload"]["id"])
    return None, []


def differences(current, matches):
    notes = []
    c = current["configuration"]
    if any(p["payload"]["engine_version"] != current["engine_version"] for p in matches):
        notes.append("版本不同")
    for keys, label in ((("actual_ubatch", "configured_ubatch", "ubatch"), "ubatch"), (("auto_profile",), "策略"), (("configured_ram_budget", "ram_budget"), "RAM预算"), (("configured_vram_budget", "vram_budget"), "显存预算")):
        value = next((normal(str(c[k])) for k in keys if c.get(k) is not None), "")
        for entry in matches:
            other = entry["payload"]["configuration"]
            v = next((normal(str(other[k])) for k in keys if other.get(k) is not None), "")
            if value and v and value != v:
                notes.append(label + "不同")
                break
    return notes


def comparison(current, row, entries):
    sections = row.get("sections")
    fluctuation = ["分段波动较大"] if sections and max(sections) > min(sections) * 1.3 else []
    if row["partial"] or not number(row["rate"]):
        return {"matches": [], "level": None, "reference": None, "delta": None, "note": "失败或未完成，不比较"}
    level, matches = historical(current, row, entries)
    if not matches:
        return {"matches": [], "level": None, "reference": None, "delta": None, "note": "；".join(["暂无参考", *fluctuation])}
    median = statistics.median(e["rate"] for e in matches)
    delta = (row["rate"] / median - 1) * 100
    note = []
    if level >= 3:
        note.append("近似硬件参考")
    elif len(matches) < 3:
        note.append("样本较少")
    elif abs(delta) > 50:
        note.append("明显偏高" if delta > 0 else "明显偏低")
    elif abs(delta) > 30:
        note.append("偏高" if delta > 0 else "偏低")
    else:
        note.append("接近")
    note.extend(fluctuation)
    note.extend(differences(current, matches))
    return {"matches": matches, "level": level, "reference": median, "delta": delta, "note": "；".join(note)}


def row_label(row):
    return ("手动 " if row["key"][1].startswith("manual_") else "") + ("Prefill" if row["kind"] == "prefill" else f"Decode x{row['slots'] if row['slots'] is not None else '?'}")


def render_summary(payload, entries, repository, baseline, head, report_path, row_limit=MAX_ROWS):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository) or not all(SHA.fullmatch(s) for s in (baseline, head)) or not REPORT_PATH.fullmatch(report_path):
        raise ValueError("invalid summary provenance")
    c = payload["configuration"]
    gpu, ram = hardware(payload)
    model = payload["model"]
    quant = model.get("quantization") or model.get("quant")
    model_label = model["name"] + (f" / {quant}" if quant and normal(quant) not in normal(model["name"]) else "")
    profile = {"aggressive": "自动激进", "conservative": "自动保守", "manual": "手动"}.get(normal(c.get("auto_profile")), "未知")
    ubatch = c.get("actual_ubatch")
    if ubatch is None:
        configured = c.get("configured_ubatch", c.get("ubatch"))
        ubatch = f"未知（设置 {configured}）" if configured is not None else "未知"
    kv_label = f"{kv(c.get('kv_type_k')) or '未知'}/{kv(c.get('kv_type_v')) or '未知'}"
    source = f"https://github.com/{repository}/blob/{head}/{report_path}"
    lines = [MARKER, "### Benchmark 摘要", "", "| 模型 | GPU／显存 | RAM | 引擎 | 策略 | KV | 实际 ubatch |", "|---|---|---|---|---|---|---|",
             "| " + " | ".join(cell(v) for v in (model_label, f"{gpu.get('name') or '未知'} / {size(gpu.get('vram_total_bytes'))}", size(ram.get('total_bytes')), payload['engine_version'], profile, kv_label, ubatch)) + " |", "", "### 测试速度", "", "单位：tok/s，多路为聚合速度。", "", "| 上下文 | 项目 | 前段 | 中段 | 后段 | 平均 | 状态 |", "|---|---|---:|---:|---:|---:|---|"]
    all_rows = rows(payload)
    display = all_rows[:row_limit]
    for row in display:
        sections = row["sections"] or [None] * 3
        state = "失败" if not number(row["rate"]) else "部分完成" if row["partial"] else "完成"
        lines.append("| " + " | ".join([context_label(row['key'][0]), row_label(row), *[rate(s) for s in sections], rate(row['rate']), state]) + " |")
    lines.extend(["", "### 历史对比", "", "| 上下文／项目 | 本次 | 历史参考 | 差异 | 匹配条件 | 提示 |", "|---|---:|---:|---:|---|---|"])
    labels = {1: "同模型、GPU、RAM", 2: "同模型、GPU", 3: "同模型、GPU系列、显存", 4: "同模型、其他硬件"}
    for row in display:
        result = comparison(payload, row, entries)
        level, matches = result['level'], result['matches']
        links = []
        for i, entry in enumerate(matches[:3], 1):
            name = entry['payload']['id']
            month = entry['payload']['created_at_utc'][:7]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", name) or not re.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2])", month):
                raise ValueError("invalid historical identity")
            links.append(f"[{i}](https://github.com/{repository}/blob/{baseline}/benchmark-database/v1/{month}/{name}.m4b)")
        reference = rate(result['reference'])
        match_label = f"{labels[level]}，{len(matches)}份 {' '.join(links)}" if matches else "无匹配记录"
        if level in (3, 4):
            names = sorted({f"{hardware(e['payload'])[0].get('name') or '未知GPU'}/{size(hardware(e['payload'])[0].get('vram_total_bytes'))}/{size(hardware(e['payload'])[1].get('total_bytes'))} RAM" for e in matches})
            match_label += "；" + cell("、".join(names), 180)
        delta = f"{result['delta']:+.1f}%" if result['delta'] is not None else "—"
        lines.append(f"| {context_label(row['key'][0])} {row_label(row)} | {rate(row['rate'])} | {reference} | {delta} | {match_label} | {cell(result['note'], 160)} |")
    if len(all_rows) > row_limit:
        lines.extend(["", f"其余 {len(all_rows) - row_limit} 项见[报告原件]({source})。"])
    lines.extend(["", f"[报告原件]({source}) · 检查 `{head[:12]}` · 历史 `{baseline[:12]}`"])
    body = "\n".join(lines) + "\n"
    if len(body.encode('utf-8')) > MAX_COMMENT_BYTES:
        if row_limit > 1:
            return render_summary(payload, entries, repository, baseline, head, report_path, max(1, row_limit // 2))
        raise ValueError("summary exceeds the comment size limit")
    return body


def git(directory, *args):
    return subprocess.check_output(["git", *args], cwd=directory, timeout=60)


def build_review(directory, base, head, baseline, repository):
    if not all(SHA.fullmatch(s) for s in (base, head, baseline)):
        raise ValueError("expected immutable Git commits")
    if not git(directory, "diff", "--name-only", "-z", base + "..." + head, "--", "benchmark-database/v1/"):
        return {"status": "skipped", "head_sha": head}
    spec = importlib.util.spec_from_file_location("classify_submission_pr", Path(__file__).with_name("classify-submission-pr.py"))
    classifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classifier)
    try:
        classification = classifier.classify(base, head, cwd=directory)
    except ValueError:
        return {"status": "invalid", "head_sha": head, "body": MARKER + f"\n报告 PR 必须仅新增一份报告，本次未生成摘要。\n\n检查 `{head[:12]}`\n"}
    if classification['submission_only'] != 'true' or classification['submission_kind'] != 'benchmark':
        return {"status": "skipped", "head_sha": head}
    report_path = classification['submission_path']
    try:
        metadata = git(directory, "ls-tree", "-l", head, "--", report_path).split(b"\t", 1)[0].split()
        if len(metadata) != 4 or metadata[:2] != [b"100644", b"blob"] or int(metadata[3]) > MAX_ARMOR_BYTES:
            raise ValueError("invalid report blob")
        report = decode_text(git(directory, "cat-file", "blob", metadata[2].decode()).decode('utf-8'))
        validate_payload(report.kind, report.payload)
        match = REPORT_PATH.fullmatch(report_path)
        if report.kind != 'benchmark' or report.submission_id != match[2] or report.created_at[:7] != match[1]:
            raise ValueError("invalid report identity")
    except (ValueError, UnicodeError, TypeError, KeyError):
        return {"status": "invalid", "head_sha": head, "body": MARKER + f"\n报告校验未通过，本次未生成性能摘要。\n\n检查 `{head[:12]}`\n"}
    entries = collect(directory, baseline)['entries']
    return {"status": "ready", "head_sha": head, "baseline_sha": baseline, "report_path": report_path, "body": render_summary(report.payload, entries, repository, baseline, head, report_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'head', 'baseline', 'repository', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    result = build_review(Path.cwd(), args.base, args.head, args.baseline, args.repository)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'body'}))


if __name__ == '__main__':
    main()
