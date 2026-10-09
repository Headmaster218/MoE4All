#!/usr/bin/env python3
"""Linux launch workflow matching infr-wizard.ps1; standard library only."""
import argparse
import getpass
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tarfile

import linux_release

DEFAULTS = {
    "launch_mode": "run", "setup_mode": "conservative", "model": "", "device": "",
    "context": "", "ubatch": "", "threads": "", "config_path": "",
    "kv_preset": "q8", "kv_type_k": "q8_0", "kv_type_v": "q8_0",
    "configure_memory": False, "vram_budget": "", "vram_reserve": "",
    "expert_cache": "", "ram_budget": "", "host_dma": True,
    "pager_ring": "", "pager_ring_slots": "", "kv_overflow": False,
    "kv_overflow_vram_mb": "", "kv_overflow_reserve_mb": "",
    "submit_mode": "auto", "submit_cap": "64", "configure_diagnostics": False,
    "pager_stats": False, "pager_profile": False, "stage_profile": False, "vram_profile": False,
    "think_mode": "default", "configure_thinking": False, "reasoning_effort": "default",
    "max_new": "65536", "configure_sampling": False, "temperature": "",
    "top_k": "", "top_p": "", "seed": "", "server_addr": "127.0.0.1:8080",
    "server_parallel": "1", "server_auth": False, "server_session_cache": False,
    "session_cache_dir": "", "session_idle_secs": "120", "session_cache_max": "5GiB",
    "session_cache_ttl_hours": "24", "server_vision": False, "vision_projector": "",
    "server_embedding": False, "embedding_model": "", "embedding_idle_timeout": "300",
    "mtp_enabled": False, "mtp_model": "", "mtp_verify_tokens": "4",
    "cpu_miss_enabled": False, "cpu_miss_max": "1", "cpu_miss_cores": "",
    "bench_kind": "decode", "prompt_tokens": "1024", "gen_tokens": "128",
    "depth_mode": "none", "depth_tokens": "0", "reps": "1", "json_output": False,
    "custom_sets": "", "auto_overrides": "",
}
LEGACY = {
    "MODE": "launch_mode", "MODEL": "model", "PROFILE": "setup_mode", "CTX": "context",
    "UBATCH": "ubatch", "KV_K": "kv_type_k", "KV_V": "kv_type_v", "RAM": "ram_budget",
    "VRAM": "vram_budget", "MTP": "mtp_model", "MTP_K": "mtp_verify_tokens",
    "ADDR": "server_addr", "PARALLEL": "server_parallel", "MMPROJ": "vision_projector",
    "EMBEDDING": "embedding_model",
}
DOWNLOADS = {
    "llm": [
        ("Qwen3.6 35B APEX-I-Balanced", "https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/resolve/main/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf?download=true"),
        ("Qwen3.8 Flash-Next AD-4.27bpw (33 shards)", "https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64"),
    ],
    "mtp": [("Flash-Next shared Q4_K_M MTP", "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf?download=true")],
    "vision": [("Flash-Next F16 mmproj", "https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-Qwen3.8-Flash-Next-F16.gguf?download=true")],
    "embedding": [("Qwen3-Embedding 0.6B", "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/tree/main")],
}


def load_state(root, directory):
    state = dict(DEFAULTS, session_cache_dir=str(root / "kv-sessions"))
    path = directory / "wizard-state.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError("Invalid saved settings")
        for key, value in saved.items():
            if key in state:
                if not isinstance(value, type(DEFAULTS[key])):
                    raise ValueError("Invalid saved value: " + key)
                state[key] = value
        if "configure_thinking" not in saved:
            state["configure_thinking"] = state["think_mode"] != "default" or state["reasoning_effort"] != "default"
    elif (directory / "wizard.conf").exists():
        for line in (directory / "wizard.conf").read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key in LEGACY:
                state[LEGACY[key]] = value
        for enabled, model in [("mtp_enabled", "mtp_model"), ("server_vision", "vision_projector"), ("server_embedding", "embedding_model")]:
            state[enabled] = bool(state[model])
        state["configure_memory"] = bool(state["ram_budget"] or state["vram_budget"])
        state["auto_overrides"] = ",".join(k for k in ("ubatch", "ram_budget", "vram_budget") if state[k])
    state["launch_mode"] = {"chat": "run", "server": "serve", "benchmark": "bench"}.get(state["launch_mode"], state["launch_mode"])
    return state, path


def prompt(label, default="", required=False):
    while True:
        value = input(label + (" [" + str(default) + "]" if default != "" else "") + ": ").strip()
        value = "" if value == "-" else (value or default)
        if value != "" or not required:
            return str(value)
        print("A value is required.")


def yes(label, default=False):
    while True:
        value = input(label + (" [Y/n]: " if default else " [y/N]: ")).strip().lower()
        if not value:
            return default
        if value in ("y", "yes", "n", "no"):
            return value in ("y", "yes")
        print("Enter y or n.")


def integer(label, default, minimum=0, maximum=None, blank=False):
    while True:
        value = prompt(label, default)
        if not value and blank:
            return ""
        if value.isdigit() and int(value) >= minimum and (maximum is None or int(value) <= maximum):
            return value
        print("Integer outside the allowed range.")


def choice(label, default, options):
    print("\n" + label)
    for i, (value, text) in enumerate(options, 1):
        print(" {}[{}] {}".format("*" if value == default else " ", i, text))
    while True:
        value = input("Select (Enter = previous/default): ").strip()
        if not value and default in dict(options):
            return default
        if value.isdigit() and 1 <= int(value) <= len(options):
            return options[int(value) - 1][0]
        if value in dict(options):
            return value
        print("Choose one of the listed options.")


def show_recommendations(kind):
    for name, url in DOWNLOADS[kind]:
        print(name + "\n  " + url)


def model_candidates(directory, kind):
    candidates = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".gguf")
    if kind == "vision":
        return [p for p in candidates if p.name.lower().startswith("mmproj")]
    if kind in ("llm", "mtp"):
        candidates = [p for p in candidates if not p.name.lower().startswith("mmproj") and (
            not re.search(r"-\d{5}-of-\d{5}\.gguf$", p.name, re.I) or re.search(r"-00001-of-\d{5}\.gguf$", p.name, re.I))]
    if kind == "llm":
        candidates = [p for p in candidates if not p.name.lower().startswith("mtp")]
    return candidates


def model_path(kind, default, root, allow_hub=False):
    nearby = model_candidates(root, kind)
    if kind == "llm":
        remembered = [default]
        gui_state = root / "gui-data/state.json"
        if gui_state.is_file():
            try:
                gui = json.loads(gui_state.read_text(encoding="utf-8-sig"))
                for key in ("recent", "favorites"):
                    if isinstance(gui.get(key), list):
                        remembered.extend(gui[key])
                if isinstance(gui.get("profiles"), list):
                    remembered.extend(p.get("model_path", "") for p in gui["profiles"] if isinstance(p, dict))
            except (OSError, ValueError, AttributeError):
                print("Could not read GUI recent-model list; keeping wizard defaults.")
        for value in reversed(remembered):
            if isinstance(value, str) and value:
                path = Path(value).expanduser()
                path = (path if path.is_absolute() else root / path).resolve()
                if path.is_file() and path.suffix.lower() == ".gguf" and path not in nearby:
                    nearby.insert(0, path)
    while True:
        print("\n" + kind.upper() + " model; R = official recommended models")
        for i, path in enumerate(nearby, 1):
            print(" [{}] {}".format(i, path.name))
        if not default:
            show_recommendations(kind)
        selected_default = default or (str(nearby[0]) if kind == "llm" and nearby else "")
        value = prompt("GGUF file/directory, number, or R", selected_default, required=True).strip('"')
        if value.lower() in ("r", "recommended"):
            show_recommendations(kind)
            continue
        if value.isdigit() and 1 <= int(value) <= len(nearby):
            value = str(nearby[int(value) - 1])
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.is_file() and path.suffix.lower() == ".gguf":
            return str(path.resolve())
        if path.is_dir():
            candidates = model_candidates(path, kind)
            if len(candidates) == 1:
                return str(candidates[0].resolve())
            print("Directory has {} candidates; enter the exact GGUF file path.".format(len(candidates)))
            continue
        if allow_hub and not value.lower().endswith(".gguf") and not Path(value).is_absolute() and "/" in value:
            return value
        print("GGUF file not found.")


def reasoning_efforts(path):
    # Read bounded GGUF metadata, never tensor payloads or the entire model.
    with Path(path).open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size

        def take(n):
            if n < 0 or n > size - stream.tell():
                raise ValueError("Truncated GGUF")
            data = stream.read(n)
            if len(data) != n:
                raise ValueError("Truncated GGUF")
            return data

        def u32():
            return struct.unpack("<I", take(4))[0]

        def u64():
            return struct.unpack("<Q", take(8))[0]

        def text():
            n = u64()
            if n > 16 * 1024 * 1024:
                raise ValueError("Oversized GGUF string")
            return take(n).decode("utf-8")

        def skip(kind, depth=0):
            if depth > 8:
                raise ValueError("Nested GGUF array")
            if kind == 9:
                element, count = u32(), u64()
                if count > 1000000:
                    raise ValueError("Oversized GGUF array")
                for _ in range(count):
                    skip(element, depth + 1)
                return
            n = u64() if kind == 8 else {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}.get(kind)
            if n is None or n > size - stream.tell():
                raise ValueError("Invalid GGUF value")
            stream.seek(n, 1)

        if take(4) != b"GGUF" or u32() not in (2, 3):
            raise ValueError("Unsupported GGUF")
        u64()
        count = u64()
        if count > 1000000:
            raise ValueError("Oversized GGUF metadata")
        metadata = {}
        for _ in range(count):
            key, kind = text(), u32()
            if kind == 8 and key in ("general.architecture", "tokenizer.chat_template"):
                metadata[key] = text()
            else:
                skip(kind)
        if not re.search(r"\breasoning_effort\b", metadata.get("tokenizer.chat_template", "")):
            return []
        return ["low", "medium", "xhigh"] if metadata.get("general.architecture") == "qwen4exp" else ["low", "medium", "high", "max"]


def physical_cores():
    try:
        keys = set()
        for cpu in os.sched_getaffinity(0):
            base = Path("/sys/devices/system/cpu/cpu{}/topology".format(cpu))
            keys.add(((base / "physical_package_id").read_text().strip(), (base / "core_id").read_text().strip()))
        return max(1, len(keys))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def enumerate_devices(binary):
    result = subprocess.run([str(binary), "devices"], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError("Vulkan device enumeration failed; check the GPU driver.")
    options, default = [], ""
    for line in result.stdout.splitlines():
        match = re.match(r"\s*(Vulkan\d+):\s+(.+?)\s+\[([^\]]+)\](\s+<-\s+default)?\s*$", line)
        if match and not re.search(r"\bcpu\b", match[3], re.I):
            options.append((match[1], line.strip().split(" <-")[0]))
            if match[4]:
                default = match[1]
    if not options:
        raise ValueError("No Vulkan GPU found; check the GPU driver.")
    return options, default or options[0][0]


def configure(s, root, binary):
    s["launch_mode"] = choice("1. Purpose", s["launch_mode"], [("run", "Terminal chat (recommended)"), ("serve", "OpenAI-compatible API"), ("bench", "Benchmark")])
    s["model"] = model_path("llm", s["model"], root, allow_hub=True)
    if s["launch_mode"] != "bench":
        s["mtp_enabled"] = yes("2.2 Enable Qwen3.8 MTP?", s["mtp_enabled"])
        if s["mtp_enabled"]:
            s["mtp_model"] = model_path("mtp", s["mtp_model"], root)
            s["mtp_verify_tokens"] = choice("MTP verification width", s["mtp_verify_tokens"], [("4", "4 (recommended)"), ("3", "3"), ("2", "2")])
            print("Qwen3.8 Vulkan, greedy only. Non-greedy API requests use ordinary decode.")
    if s["launch_mode"] == "serve":
        if not s["vision_projector"]:
            directory = Path(s["model"]).parent
            matches = model_candidates(directory, "vision") if directory.is_dir() else []
            if len(matches) == 1:
                s["vision_projector"] = str(matches[0].resolve())
        s["server_vision"] = yes("2.3 Enable vision?", s["server_vision"])
        if s["server_vision"]:
            s["vision_projector"] = model_path("vision", s["vision_projector"], root)
        s["server_embedding"] = yes("2.4 Enable Embedding API?", s["server_embedding"])
        if s["server_embedding"]:
            s["embedding_model"] = model_path("embedding", s["embedding_model"], root)
            s["embedding_idle_timeout"] = integer("Embedding weight idle seconds (0 = resident)", s["embedding_idle_timeout"])
    devices, default = enumerate_devices(binary)
    s["device"] = choice("3. Runtime device", s["device"] if s["device"] in dict(devices) else default, devices)
    s["setup_mode"] = choice("4. Configuration", s["setup_mode"], [("conservative", "Automatic: conservative (recommended)"), ("aggressive", "Automatic: aggressive"), ("manual", "Manual")])
    if s["setup_mode"] == "manual":
        for key, label in [("ubatch", "Ubatch (blank = auto)"), ("threads", "CPU threads (blank = all)")]:
            s[key] = integer(label, s[key], 1, blank=True)
        while True:
            value = prompt("Config TOML (blank = default lookup)", s["config_path"])
            path = Path(value).expanduser()
            if not value or (path if path.is_absolute() else root / path).is_file():
                s["config_path"] = value
                break
            print("Config TOML file not found.")
        s["kv_preset"] = choice("KV cache", s["kv_preset"], [("auto", "Engine default"), ("q8", "Q8_0 K + V"), ("f16", "F16 K + V"), ("custom", "Custom K / V")])
        if s["kv_preset"] == "custom":
            for key in ("kv_type_k", "kv_type_v"):
                s[key] = prompt(key, s[key], required=True)
        elif s["kv_preset"] != "auto":
            s["kv_type_k"] = s["kv_type_v"] = "q8_0" if s["kv_preset"] == "q8" else "f16"
        s["configure_memory"] = yes("Configure memory and paging?", s["configure_memory"])
        if s["configure_memory"]:
            for key in ("vram_budget", "vram_reserve", "expert_cache", "ram_budget", "pager_ring"):
                s[key] = prompt(key + " (blank = auto)", s[key])
            s["host_dma"] = yes("Enable RAM-to-VRAM Host DMA?", s["host_dma"])
            s["pager_ring_slots"] = integer("Pager ring slots", s["pager_ring_slots"], 2, blank=True)
            s["kv_overflow"] = yes("Allow KV overflow to RAM?", s["kv_overflow"])
            if s["kv_overflow"]:
                for key in ("kv_overflow_vram_mb", "kv_overflow_reserve_mb"):
                    s[key] = integer(key, s[key], 1, blank=True)
        s["submit_mode"] = choice("Submit splitter", s["submit_mode"], [("auto", "Automatic feedback"), ("disabled", "Disabled / no-split"), ("fixed", "Fixed cap")])
        if s["submit_mode"] == "fixed":
            s["submit_cap"] = integer("Dispatch cap", s["submit_cap"], 1)
        s["configure_diagnostics"] = yes("Configure statistics / profilers?", s["configure_diagnostics"])
        if s["configure_diagnostics"]:
            for key in ("pager_stats", "pager_profile", "stage_profile", "vram_profile"):
                s[key] = yes(key, s[key])
        s["custom_sets"] = prompt("Extra --set (semicolon separated)", s["custom_sets"])
    else:
        s["kv_preset"], s["kv_type_k"], s["kv_type_v"] = "q8", "q8_0", "q8_0"
    s["auto_overrides"] = ""
    if s["launch_mode"] != "bench":
        s["configure_thinking"] = yes("Configure default reasoning? (API may override)", s["configure_thinking"])
        previous_mode, previous_effort = s["think_mode"], s["reasoning_effort"]
        s["think_mode"], s["reasoning_effort"] = "default", "default"
        if s["configure_thinking"]:
            s["think_mode"] = "think" if yes("Enable reasoning by default?", previous_mode != "no-think") else "no-think"
            if s["think_mode"] == "think":
                try:
                    efforts = reasoning_efforts(s["model"])
                except (OSError, ValueError, UnicodeError):
                    print("Could not read reasoning-effort support; keeping template default.")
                    efforts = []
                if efforts:
                    options = [("default", "Template default")] + [(v, v) for v in efforts]
                    s["reasoning_effort"] = choice("Reasoning effort", previous_effort if previous_effort in dict(options) else "default", options)
        s["max_new"] = integer("Max generated tokens (reasoning + answer)", s["max_new"], 1)
        s["configure_sampling"] = yes("Configure default sampling?", s["configure_sampling"])
        if s["configure_sampling"]:
            for key in ("temperature", "top_k", "top_p", "seed"):
                label = key + " (blank = model default)"
                s[key] = integer(label, s[key], 0, blank=True) if key in ("top_k", "seed") else prompt(label, s[key])
    s["cpu_miss_enabled"] = yes("5. Experimental CPU expert-miss computation?", s["cpu_miss_enabled"])
    if s["cpu_miss_enabled"]:
        count = physical_cores()
        print("Physical cores: {}. Requires AVX2 + FMA3 and full-RAM experts.".format(count))
        print("Linux uses OS scheduling; Windows hybrid-core pinning is not implemented here.")
        s["cpu_miss_max"] = integer("Maximum CPU misses (1-3)", s["cpu_miss_max"], 1, 3)
        previous = s["cpu_miss_cores"]
        if not previous.isdigit() or not 1 <= int(previous) <= count:
            previous = str(max(1, count - 2))
        s["cpu_miss_cores"] = integer("CPU compute cores", previous, 1, count)
    if s["launch_mode"] == "serve":
        if s["mtp_enabled"]:
            s["server_parallel"] = choice("6. Concurrent slots", s["server_parallel"] if s["server_parallel"] in ("1", "2") else "1", [("1", "Single-stream MTP"), ("2", "Two slots, opportunistic MTP")])
            if s["server_parallel"] == "2":
                s["mtp_verify_tokens"] = "4"
                print("One active decode uses MTP; two active decodes use ordinary batched decode.")
        else:
            s["server_parallel"] = integer("6. Concurrent slots", s["server_parallel"], 1)
    s["context"] = prompt("7. Context (blank = auto)", s["context"])
    if s["launch_mode"] == "serve":
        serial_mtp = s["mtp_enabled"] and s["server_parallel"] == "1" and not s["server_vision"] and not s["server_embedding"]
        s["server_session_cache"] = False if serial_mtp else yes("8. Cache idle KV sessions on SSD?", s["server_session_cache"])
        if serial_mtp:
            print("Single-stream text-only MTP disables SSD session caching.")
        if s["server_session_cache"] and yes("Customize SSD KV settings?", False):
            path = Path(prompt("Cache directory", s["session_cache_dir"], True)).expanduser()
            s["session_cache_dir"] = str((path if path.is_absolute() else root / path).resolve())
            s["session_idle_secs"] = integer("Spill after idle seconds", s["session_idle_secs"])
            s["session_cache_max"] = prompt("SSD cache limit", s["session_cache_max"], True)
            s["session_cache_ttl_hours"] = integer("Cache TTL hours", s["session_cache_ttl_hours"])
        while True:
            value = prompt("9. Listen address", s["server_addr"], True)
            match = re.fullmatch(r"(?:\[([^\]]+)\]|([^:]+)):(\d+)", value)
            try:
                if match and 1 <= int(match[3]) <= 65535:
                    ipaddress.ip_address(match[1] or match[2])
                    s["server_addr"] = value
                    break
            except ValueError:
                pass
            print("Enter a valid IP:port, such as 127.0.0.1:8080 or [::1]:8080.")
        s["server_auth"] = yes("Enable Bearer API-key authentication?", s["server_auth"])
    if s["launch_mode"] == "bench":
        s["bench_kind"] = choice("Benchmark type", s["bench_kind"], [("decode", "Decode"), ("prefill", "Prefill"), ("mixed", "Combined turn"), ("custom", "Custom -p / -n")])
        for key in ("prompt_tokens", "gen_tokens"):
            disabled = (key == "prompt_tokens" and s["bench_kind"] == "decode") or (key == "gen_tokens" and s["bench_kind"] == "prefill")
            s[key] = "0" if disabled else integer(key, s[key], 0 if s["bench_kind"] == "custom" else 1)
        s["depth_mode"] = choice("Context depth", s["depth_mode"], [("none", "None"), ("real", "Real warmup"), ("synthetic", "Synthetic")])
        s["depth_tokens"] = "0" if s["depth_mode"] == "none" else integer("Depth tokens", s["depth_tokens"], 1)
        s["reps"] = integer("Repetitions", s["reps"], 1)
        s["json_output"] = yes("Emit JSON?", s["json_output"])


def build_command(s, binary):
    if s["launch_mode"] not in ("run", "serve", "bench") or s["setup_mode"] not in ("conservative", "aggressive", "manual"):
        raise ValueError("Invalid saved mode or profile")
    if not s["model"]:
        raise ValueError("A model is required")
    if s["mtp_enabled"] and s["launch_mode"] != "bench":
        if not s["mtp_model"] or s["mtp_verify_tokens"] not in ("2", "3", "4"):
            raise ValueError("MTP requires a head and a width of 2, 3 or 4")
        if s["device"] and not s["device"].startswith("Vulkan"):
            raise ValueError("MTP requires a Vulkan GPU")
        if s["launch_mode"] == "serve" and s["server_parallel"] not in ("1", "2"):
            raise ValueError("MTP supports one or two configured slots")
        if s["launch_mode"] == "serve" and s["server_parallel"] == "2":
            s["mtp_verify_tokens"] = "4"
    args = [str(binary), s["launch_mode"]]

    def arg(flag, key):
        if s[key] != "":
            args.extend([flag, str(s[key])])

    def setting(name, value):
        args.extend(["--set", name + "=" + (str(value).lower() if isinstance(value, bool) else str(value))])

    arg("--dev", "device")
    if s["setup_mode"] == "manual":
        for flag, key in [("--config", "config_path"), ("--ubatch", "ubatch"), ("--threads", "threads")]:
            arg(flag, key)
    else:
        setting("device.auto_profile", s["setup_mode"])
        overrides = s["auto_overrides"].split(",")
        if "ubatch" in overrides:
            arg("--ubatch", "ubatch")
        for key, name in [("ram_budget", "device.ram_budget"), ("vram_budget", "device.vram_budget")]:
            if key in overrides and s[key]:
                setting(name, s[key])
    arg("--ctx", "context")
    if s["kv_preset"] != "auto":
        setting("kv.type_k", s["kv_type_k"])
        setting("kv.type_v", s["kv_type_v"])
    if s["setup_mode"] == "manual":
        if s["configure_memory"]:
            for name, key in [("device.vram_budget", "vram_budget"), ("device.vram_reserve", "vram_reserve"), ("paging.cache", "expert_cache"), ("device.ram_budget", "ram_budget"), ("paging.ring", "pager_ring"), ("paging.ring_slots", "pager_ring_slots")]:
                if s[key]:
                    setting(name, s[key])
            setting("paging.host_dma", s["host_dma"])
            setting("kv.overflow", s["kv_overflow"])
            if s["kv_overflow"]:
                for key in ("kv_overflow_vram_mb", "kv_overflow_reserve_mb"):
                    if s[key]:
                        setting("kv." + key[3:], s[key])
        if s["submit_mode"] != "auto":
            setting("device.submit_dispatches", "0" if s["submit_mode"] == "disabled" else s["submit_cap"])
        if s["configure_diagnostics"]:
            for name, key in [("paging.stats", "pager_stats"), ("prof.pager_profile", "pager_profile"), ("prof.stages", "stage_profile"), ("prof.vram", "vram_profile")]:
                setting(name, s[key])
    if s["cpu_miss_enabled"]:
        if not str(s["cpu_miss_cores"]).isdigit() or int(s["cpu_miss_cores"]) < 1 or s["cpu_miss_max"] not in ("1", "2", "3"):
            raise ValueError("Invalid CPU-miss core count or miss limit")
        setting("kernels.vulkan.cpu_miss_threads", s["cpu_miss_cores"])
        setting("kernels.vulkan.cpu_miss_max", s["cpu_miss_max"])
        for key in ("push", "host_result", "token_park"):
            setting("kernels.vulkan.cpu_miss_" + key, True)
        setting("kernels.vulkan.cpu_miss_spin", "262144")
    else:
        setting("kernels.vulkan.cpu_miss_threads", "0")
    if s["setup_mode"] == "manual":
        for entry in s["custom_sets"].split(";"):
            if not entry.strip():
                continue
            key, sep, value = entry.strip().partition("=")
            key = key.strip()
            if not sep or not key or key == "serve.api_key":
                raise ValueError("Invalid extra setting; use the dedicated API-key prompt")
            if key.startswith("kernels.vulkan.cpu_miss_") and any(a.startswith(key + "=") for a in args):
                continue
            setting(key, value.strip())
    if s["launch_mode"] == "bench":
        if s["bench_kind"] == "mixed":
            args.extend(["--pg", s["prompt_tokens"] + "," + s["gen_tokens"]])
        else:
            args.extend(["-p", s["prompt_tokens"], "-n", s["gen_tokens"]])
        if s["depth_mode"] != "none":
            args.extend(["-d" if s["depth_mode"] == "real" else "--synthetic-depth", s["depth_tokens"]])
        args.extend(["-r", s["reps"]])
        if s["json_output"]:
            args.append("--json")
    else:
        setting("spec.mtp", s["mtp_enabled"])
        if s["mtp_enabled"]:
            setting("spec.draft", s["mtp_model"])
            setting("spec.k", s["mtp_verify_tokens"])
        if s["think_mode"] in ("think", "no-think"):
            args.append("--" + s["think_mode"])
        if s["think_mode"] != "no-think" and s["reasoning_effort"] != "default":
            arg("--reasoning-effort", "reasoning_effort")
        arg("--max-new", "max_new")
        if s["mtp_enabled"]:
            args.extend(["--temp", "0"])
        elif s["configure_sampling"]:
            for flag, key in [("--temp", "temperature"), ("--top-k", "top_k"), ("--top-p", "top_p"), ("--seed", "seed")]:
                arg(flag, key)
    if s["launch_mode"] == "serve":
        if not s["server_auth"]:
            setting("serve.api_key", "")
        arg("--addr", "server_addr")
        arg("--parallel", "server_parallel")
        serial_mtp = s["mtp_enabled"] and s["server_parallel"] == "1" and not s["server_vision"] and not s["server_embedding"]
        if s["server_session_cache"] and not serial_mtp:
            for key in ("session_cache_dir", "session_idle_secs", "session_cache_max", "session_cache_ttl_hours"):
                setting("kv." + key, s[key])
        else:
            setting("kv.session_cache_dir", "")
        if s["server_vision"]:
            arg("--mmproj", "vision_projector")
        if s["server_embedding"]:
            arg("--embedding-model", "embedding_model")
            arg("--embedding-idle-timeout", "embedding_idle_timeout")
    args.append(s["model"])
    return args


def parser():
    p = argparse.ArgumentParser(description="MoE4All Linux wizard; Enter reuses defaults, '-' clears a value.")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", "-y", action="store_true", help="headless launch, including unauthenticated LAN serving")
    p.add_argument("--skip-update-check", action="store_true")
    updates = p.add_mutually_exclusive_group()
    updates.add_argument("--check-update", action="store_true", help="check releases only, then exit")
    updates.add_argument("--update", action="store_true", help="explicitly update a managed Linux package; confirmation required")
    for flag, key in [("mode", "launch_mode"), ("model", "model"), ("profile", "setup_mode"), ("dev", "device"), ("ctx", "context"), ("ubatch", "ubatch"), ("threads", "threads"), ("config", "config_path"), ("kv-k", "kv_type_k"), ("kv-v", "kv_type_v"), ("ram", "ram_budget"), ("vram", "vram_budget"), ("mtp", "mtp_model"), ("mtp-k", "mtp_verify_tokens"), ("addr", "server_addr"), ("parallel", "server_parallel"), ("mmproj", "vision_projector"), ("embedding", "embedding_model"), ("max-new", "max_new"), ("cpu-miss-cores", "cpu_miss_cores"), ("cpu-miss-max", "cpu_miss_max"), ("bench-p", "prompt_tokens"), ("bench-n", "gen_tokens"), ("bench-d", "depth_tokens")]:
        p.add_argument(*(["--" + flag] + (["-u"] if flag == "ubatch" else [])), dest=key, default=None)
    for flag in ("mtp", "mmproj", "embedding", "api-key"):
        p.add_argument("--no-" + flag, action="store_true")
    return p


def find_binary(root):
    for path in (root / "infr", root / "target/release/infr", Path(shutil.which("infr") or "/nonexistent")):
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    raise ValueError("infr not found beside the launcher or under target/release; build infr-cli first.")


def loopback(address):
    return bool(re.fullmatch(r"(?:127\.\d+\.\d+\.\d+|localhost|\[::1\]):\d+", address))


def check_updates(root, binary, interactive, only=True):
    try:
        current = subprocess.check_output([str(binary), "--version"], text=True, timeout=10).strip().split()[-1]
        release = linux_release.latest_release()
        if linux_release.version_key(release["version"]) <= linux_release.version_key(current):
            print("Update check: v{} is current".format(current))
            return False
        print("New engine release: v{}\n{}".format(release["version"], release["url"]))
        available = release.get("archive") and (root / "install-manifest.json").is_file() and binary == root / "infr"
        if not available:
            print("Source install or no compatible Linux package: check only; no files changed.")
        elif only:
            print("To update explicitly: bash ./Start-INFR-Wizard-Linux.sh --update")
        elif not interactive:
            print("Updating requires interactive confirmation; no files changed.")
        elif yes("Update this Linux package now?", False):
            linux_release.download_update(root, release)
            print("Update complete. Run the launcher again.")
            return True
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, tarfile.TarError) as exc:
        print("Update check unavailable; continuing offline: " + str(exc))
    return False


def main(argv=None):
    options = parser().parse_args(argv)
    root = options.root.resolve()
    binary = find_binary(root)
    interactive = sys.stdin.isatty() and not options.dry_run
    if options.check_update or options.update:
        check_updates(root, binary, interactive and not options.yes, only=not options.update)
        return 0
    if interactive and not options.dry_run and not options.skip_update_check and os.environ.get("MOE4ALL_NO_UPDATE_CHECK", "").lower() not in ("1", "true", "yes", "on"):
        check_updates(root, binary, not options.yes)
    directory = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "infr"
    s, state_path = load_state(root, directory)
    overrides = {key for key in DEFAULTS if getattr(options, key, None) is not None}
    for key in overrides:
        s[key] = getattr(options, key)
    s["auto_overrides"] = ",".join(sorted(set(s["auto_overrides"].split(",")) | (overrides & {"ubatch", "ram_budget", "vram_budget"})))
    for field, enabled in [("mtp_model", "mtp_enabled"), ("vision_projector", "server_vision"), ("embedding_model", "server_embedding")]:
        if field in overrides:
            s[enabled] = bool(s[field])
    for clear, field, enabled in [(options.no_mtp, "mtp_model", "mtp_enabled"), (options.no_mmproj, "vision_projector", "server_vision"), (options.no_embedding, "embedding_model", "server_embedding")]:
        if clear:
            s[field], s[enabled] = "", False
    if overrides & {"ram_budget", "vram_budget"}:
        s["configure_memory"] = True
    if overrides & {"kv_type_k", "kv_type_v"}:
        s["kv_preset"] = "custom"
    if "cpu_miss_cores" in overrides:
        s["cpu_miss_enabled"] = s["cpu_miss_cores"] not in ("", "0")
    if "depth_tokens" in overrides:
        s["depth_mode"] = "real"
    if overrides & {"prompt_tokens", "gen_tokens"}:
        s["bench_kind"] = "custom"
    if s["setup_mode"] not in ("conservative", "aggressive", "manual") or s["launch_mode"] not in ("run", "serve", "bench"):
        raise ValueError("Invalid saved mode or profile")
    inherited_key = os.environ.get("INFR_API_KEY", "")
    if options.no_api_key:
        s["server_auth"] = False
    elif inherited_key:
        s["server_auth"] = True
    reused = False
    if interactive and not options.yes:
        print("MoE4All Linux Launch Wizard\nEnter = previous/default; '-' = clear")
        if s["model"] and not overrides and not any((options.no_mtp, options.no_mmproj, options.no_embedding)) and Path(s["model"]).is_file():
            reused = yes("Start with previous settings?", True)
        if not reused:
            configure(s, root, binary)
        if options.no_api_key:
            s["server_auth"] = False
    api_key = ""
    if s["launch_mode"] == "serve" and s["server_auth"]:
        api_key = inherited_key
        if interactive and not options.yes:
            api_key = getpass.getpass("API key (hidden, Enter reuses inherited key): ") or api_key
        if not api_key:
            raise ValueError("API-key authentication requires INFR_API_KEY or hidden input")
    for key in ("model", "mtp_model", "vision_projector", "embedding_model", "config_path", "session_cache_dir"):
        value = s[key]
        if not value:
            continue
        if key == "model" and not value.lower().endswith(".gguf") and not Path(value).is_absolute() and "/" in value:
            continue  # Hugging Face reference, not an install-relative path.
        path = Path(value).expanduser()
        s[key] = str((path if path.is_absolute() else root / path).resolve())
    command = build_command(s, binary)
    print("\nCommand:\n  " + shlex.join(command))
    if s["launch_mode"] == "serve" and not loopback(s["server_addr"]) and not api_key:
        print("warning: network listen address without API-key authentication", file=sys.stderr)
        if not options.dry_run and not options.yes:
            if not interactive:
                raise ValueError("Unauthenticated network serving requires explicit --yes")
            if not yes("Continue without authentication?", False):
                return 0
    if options.dry_run:
        return 0
    if not options.yes and not reused:
        if not interactive:
            raise ValueError("Launch confirmation requires input or --yes")
        if not yes("10. Start now?", True):
            return 0
    directory.mkdir(parents=True, exist_ok=True)
    linux_release.write_json_atomic(state_path, s, mode=0o600)
    print("Settings saved: " + str(state_path), flush=True)
    environment = dict(os.environ)
    if api_key:
        environment["INFR_API_KEY"] = api_key
    else:
        environment.pop("INFR_API_KEY", None)
    os.chdir(root)
    os.execve(str(binary), command, environment)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EOFError, KeyboardInterrupt):
        print("Launch cancelled.", file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
