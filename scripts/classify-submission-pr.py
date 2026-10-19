#!/usr/bin/env python3
"""Classify one immutable report addition using the PR's merge base."""
import re
import subprocess
import sys


def classify(base: str, head: str, cwd=None) -> dict:
    if not all(re.fullmatch(r"[a-f0-9]{40}", ref) for ref in (base, head)):
        raise ValueError("expected immutable Git commit IDs")
    raw = subprocess.check_output(["git", "diff", "--name-status", "-z", "--no-renames", base + "..." + head], cwd=cwd)
    fields = raw.split(b"\0")[:-1]
    if len(fields) % 2:
        raise ValueError("malformed Git diff")
    changes = list(zip(fields[::2], fields[1::2]))
    touched = any(path.startswith((b"benchmark-database/v1/", b"error-database/v1/")) for _, path in changes)
    result = {"submission_only": "false", "submission_kind": "", "submission_path": ""}
    if len(changes) == 1 and changes[0][0] == b"A":
        path = changes[0][1].decode("utf-8", errors="strict")
        match = re.fullmatch(r"(benchmark|error)-database/v1/[0-9]{4}-(?:0[1-9]|1[0-2])/[A-Za-z0-9][A-Za-z0-9._-]{0,159}\.(m4b|m4e)", path)
        if match and match[2] == {"benchmark": "m4b", "error": "m4e"}[match[1]]:
            return {"submission_only": "true", "submission_kind": match[1], "submission_path": path}
    if touched:
        raise ValueError("submission databases are immutable: add exactly one .m4b or .m4e report, without other changes")
    return result


if __name__ == "__main__":
    try:
        for key, value in classify(*sys.argv[1:]).items():
            print(f"{key}={value}")
    except (ValueError, UnicodeError, subprocess.CalledProcessError, TypeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
