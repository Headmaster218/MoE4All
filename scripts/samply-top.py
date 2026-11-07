#!/usr/bin/env python3
"""Rank self-time per FUNCTION (default) or per source line (--lines) across all threads
of a samply profile, symbolicating native frames with addr2line (samply itself only
symbolicates in its UI).

Headless companion to `samply record` for CPU perf attribution — see docs/perf.md's
"CPU profiling (samply)". Zero dependencies beyond python3 + binutils addr2line.

    samply record --save-only -o prof.json.gz -- ./target/release/infr bench ...
    scripts/samply-top.py prof.json.gz [N] [--lines]
"""

import gzip
import json
import subprocess
import sys
from collections import Counter

args = [a for a in sys.argv[1:] if a != "--lines"]
per_line = "--lines" in sys.argv
path = args[0]
top_n = int(args[1]) if len(args) > 1 else 30
d = json.load(gzip.open(path))
interval = d["meta"].get("interval", 1.0)  # ms per sample
libs = d["libs"]

# Self time per (lib index, relative address) leaf frame. Thread-seconds, summed over
# every thread — a function at N% here is N% of all CPU work, not of wall time.
self_ms = Counter()
total = 0.0
for t in d["threads"]:
    st, ft, fu = t["stackTable"], t["frameTable"], t["funcTable"]
    rt, sa = t["resourceTable"], t["samples"]
    stacks = sa["stack"]
    weights = sa.get("weight") or [1] * len(stacks)
    st_frame, ft_addr, ft_func = st["frame"], ft["address"], ft["func"]
    fu_res, rt_lib = fu["resource"], rt["lib"]
    for s, w in zip(stacks, weights):
        if s is None:
            continue
        frame = st_frame[s]
        res = fu_res[ft_func[frame]]
        lib = rt_lib[res] if res >= 0 else -1
        self_ms[(lib, ft_addr[frame])] += w * interval
        total += w * interval

# Symbolicate EVERY distinct leaf address, one addr2line process per lib (addresses on
# stdin — argv would overflow on tens of thousands of frames).
by_lib = {}
for lib, addr in self_ms:
    by_lib.setdefault(lib, []).append(addr)
names = {}
for lib, addrs in by_lib.items():
    libpath = libs[lib]["path"] if lib >= 0 else None
    if libpath is None:
        continue
    try:
        out = subprocess.run(
            ["addr2line", "-f", "-C", "-e", libpath],
            input="\n".join(hex(a) for a in addrs),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        for i, addr in enumerate(addrs):
            fn, loc = out[2 * i], out[2 * i + 1]
            names[(lib, addr)] = (fn, loc.split("/")[-1])
    except Exception as e:  # noqa: BLE001 — missing addr2line/lib just degrades to raw addrs
        print(f"addr2line failed for {libpath}: {e}", file=sys.stderr)

# Aggregate by function name (or function+line with --lines) and rank.
agg = Counter()
for (lib, addr), ms in self_ms.items():
    libname = libs[lib]["name"] if lib >= 0 else "?"
    fn, loc = names.get((lib, addr), (hex(addr), "?"))
    key = f"[{libname}] {fn}  ({loc})" if per_line else f"[{libname}] {fn}"
    agg[key] += ms

print(f"total sampled: {total / 1000:.1f} thread-seconds ({len(d['threads'])} threads)")
for key, ms in agg.most_common(top_n):
    print(f"{ms / 1000:9.2f}s {ms / total * 100:5.1f}%  {key[:130]}")
