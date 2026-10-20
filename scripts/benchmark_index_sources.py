#!/usr/bin/env python3
"""Read and revalidate benchmark originals from one immutable, trusted Git tree."""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from moe4all_submission import MAX_ARMOR_BYTES, decode_text
from submission_validation import validate_payload


def collect(directory, ref):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=directory)

    if not re.fullmatch(r"[A-Za-z0-9_./-]+", ref) or ref.startswith("-") or ".." in ref:
        raise ValueError("invalid source ref")
    head = git("rev-parse", "--verify", ref + "^{commit}").decode().strip()
    commit = git("log", "-1", "--format=%H", head, "--", "benchmark-database/v1").decode().strip()
    if not commit:
        commit = sorted(git("rev-list", "--max-parents=0", head).decode().splitlines())[0]
    created = git("show", "-s", "--format=%cI", commit).decode().strip()
    tree = git("ls-tree", "-r", "-l", "-z", commit, "--", "benchmark-database/v1")
    entries, ids, source_bytes = [], set(), 0
    for record in sorted(tree.split(b"\0")):
        if not record:
            continue
        meta, raw_path = record.split(b"\t", 1)
        path = raw_path.decode("utf-8")
        if path.endswith(".json"):
            raise ValueError("legacy benchmark JSON must be converted, not silently omitted")
        if not path.endswith(".m4b"):
            continue
        mode, kind, blob, size = meta.split()
        match = re.fullmatch(r"benchmark-database/v1/(\d{4}-(?:0[1-9]|1[0-2]))/([A-Za-z0-9][A-Za-z0-9._-]{0,159})\.m4b", path)
        if not match or mode != b"100644" or kind != b"blob" or int(size) > MAX_ARMOR_BYTES:
            raise ValueError("invalid benchmark source path, mode or size")
        if match[2] in ids or len(entries) >= 50000:
            raise ValueError("duplicate ID or too many benchmark reports")
        ids.add(match[2])
        report = decode_text(git("cat-file", "blob", blob.decode()).decode("utf-8"))
        validate_payload(report.kind, report.payload)
        source_bytes += len(json.dumps(report.payload, ensure_ascii=False).encode("utf-8"))
        if source_bytes > 256 * 1024 * 1024:
            raise ValueError("benchmark source payloads exceed the index build input limit")
        month = datetime.fromisoformat(report.created_at.replace("Z", "+00:00")).strftime("%Y-%m")
        if report.kind != "benchmark" or report.submission_id != match[2] or month != match[1]:
            raise ValueError("benchmark identity/month does not match its path")
        entries.append({"payload": report.payload})
    return {"commit": commit, "generated_at": created, "entries": entries}


if __name__ == "__main__":
    try:
        directory = Path(sys.argv[1]).resolve(strict=True)
        print(json.dumps(collect(directory, sys.argv[2]), ensure_ascii=False, separators=(",", ":")))
    except (OSError, ValueError, UnicodeError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
