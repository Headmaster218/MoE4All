#!/usr/bin/env python3
"""Validate immutable benchmark-database text submissions without project dependencies."""

from __future__ import annotations

import re
import sys
import json
from datetime import datetime
from pathlib import Path

from moe4all_submission import decode_file


RESULT_PATH = re.compile(r"^benchmark-database/v1/(\d{4}-\d{2})/([^/]+)\.m4b$")


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        submission = decode_file(path)
    except (OSError, UnicodeError, ValueError, KeyError) as error:
        return [f"{path}: invalid MoE4All benchmark submission: {error}"]
    if submission.kind != "benchmark":
        return [f"{path}: expected a benchmark submission"]
    value = submission.payload
    if value.get("schema_version") != 1:
        errors.append(f"{path}: schema_version must be 1")
    if submission.submission_id != path.stem:
        errors.append(f"{path}: id must equal the filename stem")
    created_at = value.get("created_at_utc")
    try:
        parsed_created_at = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        if parsed_created_at.tzinfo is None:
            raise ValueError("timestamp has no timezone")
        created_month = parsed_created_at.strftime("%Y-%m")
        if path.parent.name != created_month:
            errors.append(f"{path}: directory month must match created_at_utc")
    except ValueError:
        errors.append(f"{path}: created_at_utc must be RFC 3339")
    for field in ("created_at_utc", "engine_version", "profile", "model", "hardware", "contexts"):
        if field not in value:
            errors.append(f"{path}: missing required field {field!r}")
    profile = value.get("profile")
    if profile not in {"quick", "standard", "full", "manual"}:
        errors.append(f"{path}: profile must be quick, standard, full or manual")
    contexts = value.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        errors.append(f"{path}: contexts must be a non-empty array")
    else:
        for index, context in enumerate(contexts):
            if not isinstance(context, dict):
                errors.append(f"{path}: contexts[{index}] must be an object")
                continue
            requested_tokens = context.get("requested_context_tokens")
            if not isinstance(requested_tokens, int) or isinstance(requested_tokens, bool):
                errors.append(f"{path}: contexts[{index}] is missing requested_context_tokens")
            failures = context.get("failures")
            if not isinstance(failures, list):
                errors.append(f"{path}: contexts[{index}].failures must be an array")
            prefill = context.get("prefill")
            decode = context.get("decode")
            if prefill is None and profile != "manual" and not failures:
                errors.append(f"{path}: contexts[{index}] needs a Prefill result or failure")
            if profile == "manual" and not isinstance(context.get("manual_result"), dict):
                errors.append(f"{path}: manual contexts[{index}] needs manual_result")
            if not isinstance(decode, list):
                errors.append(f"{path}: contexts[{index}].decode must be an array")
    privacy_text = json.dumps(value, ensure_ascii=False).lower()
    if re.search(r"[a-z]:\\\\|/home/|/users/|\\\\users\\\\", privacy_text):
        errors.append(f"{path}: payload contains a local filesystem path")
    if '"hostname"' in privacy_text or '"serial"' in privacy_text:
        errors.append(f"{path}: payload contains a hostname or serial field")
    return errors


def main(argv: list[str]) -> int:
    root = Path.cwd()
    supplied = [Path(arg) for arg in argv[1:]]
    database_paths = sorted((root / "benchmark-database" / "v1").glob("*/*.m4b"))
    paths = supplied or database_paths
    if not paths:
        print("no benchmark submission files found", file=sys.stderr)
        return 1

    errors: list[str] = []
    basenames: dict[str, Path] = {}
    for path in database_paths:
        previous = basenames.get(path.name)
        if previous is not None and previous != path:
            errors.append(f"duplicate benchmark filename {path.name}: {previous} and {path}")
        basenames[path.name] = path
    for path in paths:
        relative = path.as_posix()
        if path.is_absolute():
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                pass
        if RESULT_PATH.fullmatch(relative) is None:
            errors.append(f"{path}: expected benchmark-database/v1/YYYY-MM/<id>.m4b")
            continue
        errors.extend(validate_file(path))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"validated {len(paths)} benchmark submission file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
