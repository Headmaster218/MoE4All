#!/usr/bin/env python3
"""Phase-0 baseline for the tiered weight pager (docs/disk-streaming-plan.md §1.1).

Measures what the GGUF mmap + OS page cache actually deliver when the weights do
not fit the memory the process is allowed to use — the bar the bespoke DRAM/disk
tier has to beat, and the harness every later phase is re-measured against.

The squeeze is a cgroup-v2 `MemoryMax` (via `systemd-run --user --scope`), not a
bigger model: page cache charged to the cgroup is reclaimed under the limit, so a
2 GB model under a 1.5 GB cap streams from disk exactly as a 60 GB model does on
a 48 GB host, and the sweep runs in minutes instead of needing a >RAM blob. Each
run starts cold — `posix_fadvise(DONTNEED)` drops the model's page cache first,
which needs no root.

    scripts/paging-baseline.py MODEL.gguf --limits 8G,3G,2G,1.5G

Reports, per limit: prefill and decode throughput, MAJOR faults (the page cache
missing), and 512-byte blocks read from the device. Throughput alone cannot tell
a cold run from a thrashing one; the fault and I/O columns are what say which.
"""

import argparse
import json
import os
import re
import resource
import shutil
import subprocess
import sys

INFR = "./target/release/infr"


def parse_size(s):
    """`8G` / `1.5G` / `512M` / a byte count -> bytes."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kKmMgG]?)i?[bB]?", s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad size: {s!r}")
    return int(float(m.group(1)) * {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30}[m.group(2).lower()])


def drop_cache(path):
    """Evict this file's pages so the next run starts cold. Best effort: DONTNEED
    is advisory and only drops CLEAN, unmapped pages, which is exactly our case."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def run(cmd, limit_bytes):
    """Run `cmd` under a MemoryMax cgroup scope, returning (stdout, rusage delta)."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    wrapped = cmd
    if limit_bytes is not None:
        wrapped = [
            "systemd-run", "--user", "--scope", "--quiet",
            "-p", f"MemoryMax={limit_bytes}",
            "-p", "MemorySwapMax=0",  # swapping anonymous pages would hide the streaming cost
            "--",
        ] + cmd
    proc = subprocess.run(wrapped, capture_output=True, text=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"run failed ({proc.returncode}): {' '.join(wrapped)}")
    return proc.stdout, {
        "majflt": after.ru_majflt - before.ru_majflt,
        "inblock": after.ru_inblock - before.ru_inblock,
    }


def bench(model, dev, limit, flags, dram=None, cache=None):
    cmd = [INFR, "bench", model, "--dev", dev, "--json"] + flags
    if cache:
        # `paging.cache`: the VRAM paging budget. Forcing it small is what puts a GPU run on the
        # streaming path at all, and it must be IDENTICAL in both arms — otherwise the two legs
        # differ in where the weights live as well as in what backs the misses, and the comparison
        # says nothing about the host tier.
        cmd += ["--set", f"paging.cache={cache}"]
    if dram:
        # The host weight cache (`paging.dram`): weights read from the file into our own arena
        # under our own eviction policy, instead of mapped and left to the page cache.
        cmd += ["--set", f"paging.dram={dram}"]
    drop_cache(model)
    out, ru = run(cmd, limit)
    # `--json` mirrors llama-bench: [{"avg_ts": X, ...}]
    ts = json.loads(out.strip().splitlines()[-1])[0]["avg_ts"]
    return ts, ru


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dev", default="cpu", help="backend: cpu | Vulkan0 | metal")
    ap.add_argument("--limits", default="none", help="comma-separated MemoryMax values, or `none`")
    ap.add_argument("--prompt", type=int, default=512, help="prefill tokens (-p)")
    ap.add_argument("--gen", type=int, default=32, help="decode tokens (-n)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--dram", default=None,
                    help="also run each limit with this paging.dram budget (e.g. `1g`)")
    ap.add_argument("--cache", default=None,
                    help="paging.cache (VRAM paging budget), applied to BOTH arms — how a GPU run "
                         "is forced onto the streaming path")
    args = ap.parse_args()

    if not os.path.exists(INFR):
        raise SystemExit(f"{INFR} not built — run `cargo build --release`")
    if not shutil.which("systemd-run") and args.limits != "none":
        raise SystemExit("systemd-run not found; --limits needs cgroup v2 scopes")

    size = os.path.getsize(args.model)
    print(f"model: {args.model}")
    print(f"bytes: {size / 1e9:.2f} GB   dev: {args.dev}   reps: {args.reps}")
    print()
    print(f"{'limit':>8} {'mode':>6} {'pp t/s':>9} {'pp majflt':>10} {'pp read MB':>11} "
          f"{'tg t/s':>8} {'tg majflt':>10} {'tg read MB':>11}")

    arms = [("mmap", None)] + ([("dram", args.dram)] if args.dram else [])
    for spec in args.limits.split(","):
        limit = None if spec.strip() == "none" else parse_size(spec)
        for name, dram in arms:
            pp, pru = bench(args.model, args.dev, limit,
                            ["-p", str(args.prompt), "-n", "0", "-r", str(args.reps)],
                            dram, args.cache)
            tg, tru = bench(args.model, args.dev, limit,
                            ["-p", "0", "-n", str(args.gen), "-r", str(args.reps)],
                            dram, args.cache)
            print(f"{spec.strip():>8} {name:>6} {pp:>9.1f} {pru['majflt']:>10} "
                  f"{pru['inblock'] / 2048:>11.0f} {tg:>8.2f} {tru['majflt']:>10} "
                  f"{tru['inblock'] / 2048:>11.0f}")


if __name__ == "__main__":
    main()
