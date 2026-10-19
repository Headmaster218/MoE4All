#!/usr/bin/env python3
"""Dependency-free decoder and verifier for MoE4All ASCII-armored submissions."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
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
    output: list[str] = []
    previous_space = False
    for char in str(value):
        code = ord(char)
        if code <= 31 or 127 <= code <= 159:
            char = " "
        # Rust char::is_whitespace uses Unicode White_Space, not Python's extra C0 separators.
        whitespace = char in "\t\n\v\f\r \u0085\u00a0\u1680\u2028\u2029\u202f\u205f\u3000" or "\u2000" <= char <= "\u200a"
        if whitespace:
            if previous_space:
                continue
            previous_space = True
            output.append(" ")
        else:
            previous_space = False
            output.append(char)
        if len(output) >= limit:
            output.append("...")
            break
    return "".join(output).strip()


def strict_json(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result or key in {"__proto__", "constructor", "prototype"} or any(0xD800 <= ord(c) <= 0xDFFF for c in key):
                raise ValueError("duplicate or reserved JSON field")
            result[key] = value
        return result

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 48:
            raise ValueError("JSON nesting exceeds 48 levels")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if (isinstance(value, int) and abs(value) > 9007199254740991) or (isinstance(value, float) and (not math.isfinite(value) or value == int(value) and abs(value) > 9007199254740991)):
                raise ValueError("JSON number exceeds the interoperable range")
        if isinstance(value, str) and any(0xD800 <= ord(c) <= 0xDFFF for c in value):
            raise ValueError("JSON text contains an unpaired surrogate")
        for child in value.values() if isinstance(value, dict) else value if isinstance(value, list) else []:
            visit(child, depth + 1)

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid JSON number")))
    except (RecursionError, OverflowError):
        raise ValueError("JSON nesting or numeric limits exceeded") from None
    visit(value)
    return value


def _nested(value: object, *path: str) -> object | None:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _gib(value: object, unknown: str) -> str:
    if type(value) not in (int, float) or not math.isfinite(value) or value != int(value) or not 0 <= value <= 9007199254740991:
        return unknown
    return f"{value / (1 << 30):.2f} GiB"


def generated_summary(kind: str, payload: dict[str, Any]) -> list[str]:
    if kind == "benchmark":
        model = _nested(payload, "model", "name")
        model = "unknown model" if model is None else model
        architecture = _nested(payload, "model", "architecture")
        architecture = "unknown" if architecture is None else architecture
        model_size = _gib(_nested(payload, "model", "storage_bytes"), "unknown size")
        gpu = _nested(payload, "hardware", "gpu", "name")
        gpu = "unknown device" if gpu is None else gpu
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
                if type(tokens) in (int, float) and math.isfinite(tokens) and tokens == int(tokens):
                    tokens = int(tokens)
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
                        if type(parallel) in (int, float) and math.isfinite(parallel) and parallel == int(parallel):
                            parallel = int(parallel)
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
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
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
    encoded = "".join(line.strip() for line in lines[begin + 1 : end])
    compressed = base64.b64decode(encoded, validate=True)
    if base64.b64encode(compressed).decode("ascii") != encoded:
        raise ValueError("noncanonical Base64 padding")
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
    payload = strict_json(decoded.decode("utf-8"))
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
    if path.stat().st_size > MAX_ARMOR_BYTES:
        raise ValueError(f"submission exceeds {MAX_ARMOR_BYTES} bytes")
    return decode_text(path.read_bytes().decode("utf-8"))


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
            from submission_validation import validate_payload
            validate_payload(submission.kind, submission.payload)
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
