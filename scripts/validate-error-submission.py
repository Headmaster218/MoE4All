#!/usr/bin/env python3
"""Validate immutable error-database text submissions without project dependencies."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

from moe4all_submission import decode_file


RESULT_PATH = re.compile(r"^error-database/v1/(\d{4}-\d{2})/([^/]+)\.m4e$")
REQUIRED_OBJECTS = ("engine", "hardware", "model", "launch", "error", "logs", "privacy")


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        submission = decode_file(path)
    except (OSError, UnicodeError, ValueError, KeyError) as error:
        return [f"{path}: invalid MoE4All error submission: {error}"]
    if submission.kind != "error":
        return [f"{path}: expected an error submission"]
    value = submission.payload
    if value.get("schema_version") != 1:
        errors.append(f"{path}: schema_version must be 1")
    if submission.submission_id != path.stem:
        errors.append(f"{path}: id must equal the filename stem")
    created_at = value.get("created_at_utc")
    try:
        parsed = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timestamp has no timezone")
        if path.parent.name != parsed.strftime("%Y-%m"):
            errors.append(f"{path}: directory month must match created_at_utc")
    except ValueError:
        errors.append(f"{path}: created_at_utc must be RFC 3339")
    for field in ("created_at_utc", "engine_version", "failure_kind", "description", *REQUIRED_OBJECTS):
        if field not in value:
            errors.append(f"{path}: missing required field {field!r}")
    if not isinstance(value.get("failure_kind"), str) or not value.get("failure_kind"):
        errors.append(f"{path}: failure_kind must be a non-empty string")
    description = value.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append(f"{path}: description must be a non-empty string")
    elif len(description) > 8192:
        errors.append(f"{path}: description is unexpectedly large")
    for field in REQUIRED_OBJECTS:
        if not isinstance(value.get(field), dict):
            errors.append(f"{path}: {field} must be an object")
    logs = value.get("logs")
    if isinstance(logs, dict):
        if not isinstance(logs.get("startup"), list) or not isinstance(logs.get("tail"), list):
            errors.append(f"{path}: logs.startup and logs.tail must be arrays")
        if logs.get("speed_statistics_included") is not False:
            errors.append(f"{path}: speed statistics must not be included")
    privacy = value.get("privacy")
    if isinstance(privacy, dict):
        expected = {
            "prompt_or_response_included": False,
            "local_paths_redacted": True,
            "api_keys_included": False,
        }
        for field, required in expected.items():
            if privacy.get(field) is not required:
                errors.append(f"{path}: privacy.{field} must be {str(required).lower()}")
    privacy_text = json.dumps(value, ensure_ascii=False).lower()
    if re.search(r"[a-z]:\\\\|/home/|/users/|\\\\users\\\\", privacy_text):
        errors.append(f"{path}: payload contains a local filesystem path")
    if '"hostname"' in privacy_text or '"serial"' in privacy_text:
        errors.append(f"{path}: payload contains a hostname or serial field")
    return errors


def main(argv: list[str]) -> int:
    root = Path.cwd()
    supplied = [Path(arg) for arg in argv[1:]]
    database_paths = sorted((root / "error-database" / "v1").glob("*/*.m4e"))
    paths = supplied or database_paths
    if not paths:
        print("no error submission files found", file=sys.stderr)
        return 1

    errors: list[str] = []
    basenames: dict[str, Path] = {}
    for path in database_paths:
        previous = basenames.get(path.name)
        if previous is not None and previous != path:
            errors.append(f"duplicate error filename {path.name}: {previous} and {path}")
        basenames[path.name] = path
    for path in paths:
        relative = path.as_posix()
        if path.is_absolute():
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                pass
        if RESULT_PATH.fullmatch(relative) is None:
            errors.append(f"{path}: expected error-database/v1/YYYY-MM/<id>.m4e")
            continue
        errors.extend(validate_file(path))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"validated {len(paths)} error submission file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
