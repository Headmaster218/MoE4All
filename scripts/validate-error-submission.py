#!/usr/bin/env python3
"""Validate immutable error-database text submissions without project dependencies."""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

from moe4all_submission import decode_file
from submission_validation import validate_payload


RESULT_PATH = re.compile(r"^error-database/v1/(\d{4}-\d{2})/([^/]+)\.m4e$")


def validate_file(path: Path) -> list[str]:
    try:
        submission = decode_file(path)
        if submission.kind != "error":
            raise ValueError("expected an error submission")
        validate_payload(submission.kind, submission.payload)
        if submission.submission_id != path.stem:
            raise ValueError("id must equal the filename stem")
        created = datetime.fromisoformat(submission.created_at.replace("Z", "+00:00"))
        if path.parent.name != created.strftime("%Y-%m"):
            raise ValueError("directory month must match created_at_utc")
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, RecursionError):
        return [f"{path}: invalid error submission (format, schema, identity or privacy)"]
    return []

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
