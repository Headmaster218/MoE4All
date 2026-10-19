#!/usr/bin/env python3
"""Dependency-free decoder and verifier for MoE4All ASCII-armored submissions."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ARMOR_VERSION = "MOE4ALL-SUBMISSION/1"
PAYLOAD_BEGIN = "-----BEGIN MOE4ALL PAYLOAD-----"
PAYLOAD_END = "-----END MOE4ALL PAYLOAD-----"
MAX_ARMOR_BYTES = 128 * 1024
MAX_DECODED_BYTES = 128 * 1024
HEADER_NAME = re.compile(r"^[a-z0-9-]+$")


@dataclass(frozen=True)
class Submission:
    kind: str
    schema: str
    submission_id: str
    created_at: str
    headers: dict[str, str]
    payload: dict[str, Any]


def _one_line(value: object, limit: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _nested(value: object, *path: str) -> object | None:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _gib(value: object, unknown: str) -> str:
    if not isinstance(value, int) or isinstance(value, bool):
        return unknown
    return f"{value / (1 << 30):.2f} GiB"


def generated_summary(kind: str, payload: dict[str, Any]) -> list[str]:
    if kind == "benchmark":
        model = _nested(payload, "model", "name") or "unknown model"
        architecture = _nested(payload, "model", "architecture") or "unknown"
        model_size = _gib(_nested(payload, "model", "storage_bytes"), "unknown size")
        gpu = _nested(payload, "hardware", "gpu", "name") or "unknown device"
        vram = _gib(_nested(payload, "hardware", "gpu", "vram_total_bytes"), "unknown VRAM")
        lines = [f"model: {model} | {architecture} | {model_size}", f"device: {gpu} | {vram}"]
        profile = payload.get("profile")
        if isinstance(profile, str):
            lines.append(f"profile: {profile}")
        contexts = payload.get("contexts")
        if isinstance(contexts, list):
            for context in contexts:
                if not isinstance(context, dict):
                    continue
                tokens = context.get("requested_context_tokens", 0)
                label = (
                    f"{tokens // 1024}K"
                    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens and tokens % 1024 == 0
                    else str(tokens)
                )
                result: list[str] = []
                prefill = _nested(context, "prefill", "tokens_per_second")
                if isinstance(prefill, (int, float)) and not isinstance(prefill, bool):
                    result.append(f"prefill={prefill:.2f}")
                decode = context.get("decode")
                if isinstance(decode, list):
                    for case in decode:
                        if not isinstance(case, dict):
                            continue
                        parallel = case.get("parallel", 0)
                        rate = case.get("overall_tokens_per_second")
                        if isinstance(rate, (int, float)) and not isinstance(rate, bool):
                            result.append(f"x{parallel}={rate:.2f}")
                manual = _nested(context, "manual_result", "average_tokens_per_second")
                if isinstance(manual, (int, float)) and not isinstance(manual, bool):
                    result.append(f"manual={manual:.2f}")
                if result:
                    lines.append(f"result {label}: {' | '.join(result)} tok/s")
        return lines

    lines = [f"failure: {payload.get('failure_kind', 'unknown')}"]
    model = _nested(payload, "model", "name")
    if isinstance(model, str):
        lines.append(f"model: {model}")
    device = _nested(payload, "hardware", "gpu", "name")
    if isinstance(device, str):
        lines.append(f"device: {device}")
    description = payload.get("description")
    if isinstance(description, str) and description.strip():
        lines.append(f"description: {_one_line(description, 240)}")
    message = _nested(payload, "error", "message")
    if isinstance(message, str):
        lines.append(f"error: {_one_line(message, 240)}")
    return lines


def decode_text(text: str) -> Submission:
    if len(text.encode("utf-8")) > MAX_ARMOR_BYTES:
        raise ValueError(f"submission exceeds {MAX_ARMOR_BYTES} bytes")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    if not lines or lines[0] != ARMOR_VERSION:
        raise ValueError(f"not a {ARMOR_VERSION} document")
    try:
        begin = lines.index(PAYLOAD_BEGIN)
        end = lines.index(PAYLOAD_END, begin + 1)
    except ValueError as error:
        raise ValueError("submission payload markers are missing") from error
    if any(line.strip() for line in lines[end + 1 :]):
        raise ValueError("unexpected text after payload")
    headers: dict[str, str] = {}
    for line in lines[1:begin]:
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"invalid header line {line!r}")
        name, value = line.split(":", 1)
        name = name.strip().lower()
        if not HEADER_NAME.fullmatch(name):
            raise ValueError(f"invalid header name {name!r}")
        if name in headers:
            raise ValueError(f"duplicate header {name!r}")
        headers[name] = value.strip()
    kind = headers.get("kind", "")
    expected_schema = {"benchmark": "benchmark/1", "error": "error/1"}.get(kind)
    if expected_schema is None:
        raise ValueError(f"unsupported submission kind {kind!r}")
    if headers.get("schema") != expected_schema:
        raise ValueError("submission schema does not match kind")
    if headers.get("encoding") != "gzip+base64":
        raise ValueError("submission encoding must be gzip+base64")
    compressed = base64.b64decode("".join(line.strip() for line in lines[begin + 1 : end]), validate=True)
    if len(compressed) != int(headers["compressed-bytes"]):
        raise ValueError("compressed payload length mismatch")
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    decoded = inflater.decompress(compressed, MAX_DECODED_BYTES + 1)
    if len(decoded) > MAX_DECODED_BYTES:
        raise ValueError(f"decoded payload exceeds {MAX_DECODED_BYTES} bytes")
    if not inflater.eof:
        raise ValueError("gzip payload is truncated or exceeds the decoded size limit")
    if inflater.unused_data or inflater.unconsumed_tail:
        raise ValueError("gzip payload must contain exactly one member and no trailing bytes")
    decoded += inflater.flush()
    if len(decoded) > MAX_DECODED_BYTES:
        raise ValueError(f"decoded payload exceeds {MAX_DECODED_BYTES} bytes")
    if len(decoded) != int(headers["decoded-bytes"]):
        raise ValueError("decoded payload length mismatch")
    if hashlib.sha256(decoded).hexdigest() != headers.get("sha256"):
        raise ValueError("submission SHA-256 mismatch")
    payload = json.loads(decoded.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(payload, dict):
        raise ValueError("decoded payload must be a JSON object")
    submission_id = payload.get("id")
    created_at = payload.get("created_at_utc")
    if not isinstance(submission_id, str) or headers.get("id") != submission_id:
        raise ValueError("header id does not match payload")
    if not isinstance(created_at, str) or headers.get("created-at") != created_at:
        raise ValueError("header timestamp does not match payload")
    engine = payload.get("engine_version") if isinstance(payload.get("engine_version"), str) else "unknown"
    if headers.get("engine") != _one_line(engine, 80):
        raise ValueError("header engine does not match payload")
    expected_summary = generated_summary(kind, payload)
    if int(headers.get("summary-count", "-1")) != len(expected_summary):
        raise ValueError("summary count does not match payload")
    for index, expected in enumerate(expected_summary, 1):
        if headers.get(f"summary-{index:03d}") != _one_line(expected, 500):
            raise ValueError(f"summary-{index:03d} does not match payload")
    summary_headers = [name for name in headers if name.startswith("summary-") and name != "summary-count"]
    if len(summary_headers) != len(expected_summary):
        raise ValueError("submission contains unexpected summary headers")
    return Submission(kind, expected_schema, submission_id, created_at, headers, payload)


def decode_file(path: Path) -> Submission:
    return decode_text(path.read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("path", type=Path)
    decode = subparsers.add_parser("decode")
    decode.add_argument("path", type=Path)
    decode.add_argument("--output", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        submission = decode_file(args.path)
        if args.command == "verify":
            print(f"valid {submission.kind} submission {submission.submission_id}")
        else:
            text = json.dumps(submission.payload, ensure_ascii=False, indent=2) + "\n"
            if args.output:
                args.output.write_text(text, encoding="utf-8")
            else:
                sys.stdout.write(text)
    except (OSError, UnicodeError, ValueError, KeyError) as error:
        print(f"{args.path}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
