"""Upload validation shared by independent benchmark and error PR validators."""

from datetime import datetime, timedelta
import math
import re


def validate_payload(kind: str, p: object) -> None:
    def require(ok: bool, path: str) -> None:
        if not ok:
            raise ValueError(f"invalid submission field: {path}")

    def integer(value: object) -> bool:
        return type(value) in (int, float) and math.isfinite(value) and value == int(value) and 0 <= value <= 9007199254740991

    def number(value: object) -> bool:
        return type(value) in (int, float) and math.isfinite(value)

    def rate(value: object, path: str) -> None:
        require(number(value) and 0 <= value <= 1e8, path)

    def text(value: object, path: str) -> None:
        require(value is None or isinstance(value, str) and len(value) <= 1024, path)

    def capacity(value: object, path: str) -> None:
        require(value is None or integer(value), path)

    require(isinstance(p, dict) and type(p.get("schema_version")) in (int, float) and p["schema_version"] == 1, "/schema_version")
    require(isinstance(p.get("id"), str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", p["id"]) is not None, "/id")
    created = p.get("created_at_utc")
    require(isinstance(created, str) and created.endswith(("Z", "+00:00")), "/created_at_utc")
    try:
        require(datetime.fromisoformat(created.replace("Z", "+00:00")).utcoffset() == timedelta(0), "/created_at_utc")
    except ValueError:
        raise ValueError("invalid submission field: /created_at_utc") from None
    require(isinstance(p.get("engine_version"), str) and 0 < len(p["engine_version"]) <= 128, "/engine_version")
    if kind == "benchmark":
        require(p.get("profile") in ("quick", "standard", "full", "manual"), "/profile")
        for name in ("model", "hardware", "configuration"):
            require(isinstance(p.get(name), dict), "/" + name)
        model, hardware, configuration = p["model"], p["hardware"], p["configuration"]
        require(isinstance(model.get("name"), str) and 0 < len(model["name"]) <= 1024, "/model/name")
        for name in ("architecture", "quant", "quantization"):
            text(model.get(name), "/model/" + name)
        capacity(model.get("storage_bytes"), "/model/storage_bytes")
        text(hardware.get("os"), "/hardware/os")
        for name in ("gpu", "cpu", "ram"):
            require(hardware.get(name) is None or isinstance(hardware[name], dict), "/hardware/" + name)
        gpu, cpu, ram = (hardware.get(name) or {} for name in ("gpu", "cpu", "ram"))
        for value in (gpu.get("name"), gpu.get("configured_device"), cpu.get("name")):
            text(value, "/hardware")
        capacity(gpu.get("vram_total_bytes"), "/hardware/gpu/vram_total_bytes")
        capacity(ram.get("total_bytes"), "/hardware/ram/total_bytes")
        for name in ("backend", "device", "kv_type_k", "kv_type_v"):
            text(configuration.get(name), "/configuration/" + name)
        require(configuration.get("mtp") is None or type(configuration["mtp"]) is bool, "/configuration/mtp")
        for name in ("ubatch", "repetitions"):
            capacity(configuration.get(name), "/configuration/" + name)
        require(isinstance(p.get("contexts"), list) and 0 < len(p["contexts"]) <= 64, "/contexts")
        contexts = set()
        for c in p["contexts"]:
            require(isinstance(c, dict) and integer(c.get("requested_context_tokens")), "/contexts")
            tokens = c["requested_context_tokens"]
            require(tokens not in contexts, "/contexts/requested_context_tokens"); contexts.add(tokens)
            require(isinstance(c.get("failures"), list) and isinstance(c.get("decode"), list) and len(c["decode"]) <= 64, "/contexts/decode")
            if c.get("prefill") is not None:
                require(isinstance(c["prefill"], dict), "/contexts/prefill")
                rate(c["prefill"].get("tokens_per_second"), "/contexts/prefill/tokens_per_second")
            if p["profile"] == "manual":
                manual = c.get("manual_result")
                require(isinstance(manual, dict) and manual.get("kind") in ("prefill", "decode"), "/contexts/manual_result/kind")
                rate(manual.get("average_tokens_per_second"), "/contexts/manual_result/average_tokens_per_second")
            else:
                require(c.get("prefill") is not None or bool(c["failures"]), "/contexts/prefill")
            slots = set()
            for d in c["decode"]:
                require(isinstance(d, dict) and integer(d.get("parallel")) and 0 < d["parallel"] <= 64 and d["parallel"] not in slots, "/contexts/decode/parallel")
                slots.add(d["parallel"]); rate(d.get("overall_tokens_per_second"), "/contexts/decode/overall_tokens_per_second")
                require(isinstance(d.get("repetitions"), list) and 0 < len(d["repetitions"]) <= 100, "/contexts/decode/repetitions")
                means = d.get("section_mean_tokens_per_second")
                require(isinstance(means, list) and len(means) == 3, "/contexts/decode/section_mean_tokens_per_second")
                for value in means:
                    rate(value, "/contexts/decode/section_mean_tokens_per_second")
                for rep in d["repetitions"]:
                    require(isinstance(rep, dict) and isinstance(rep.get("overall"), dict) and isinstance(rep.get("sections"), list) and len(rep["sections"]) == 3, "/contexts/decode/repetitions")
                    for section in [rep["overall"], *rep["sections"]]:
                        require(isinstance(section, dict) and integer(section.get("tokens")) and number(section.get("seconds")) and section["seconds"] >= 0, "/contexts/decode/repetitions/sections")
                        rate(section.get("tokens_per_second"), "/contexts/decode/repetitions/sections")
                    lanes = rep.get("lane_tokens_per_second")
                    require(isinstance(lanes, list) and len(lanes) == 3 and all(isinstance(row, list) and len(row) == d["parallel"] for row in lanes), "/contexts/decode/repetitions/lane_tokens_per_second")
                    for row in lanes:
                        for value in row:
                            rate(value, "/contexts/decode/repetitions/lane_tokens_per_second")
    elif kind == "error":
        allowed = {"schema_version", "id", "created_at_utc", "engine_version", "engine_commit", "failure_kind", "description", "engine", "hardware", "model", "launch", "error", "logs", "privacy"}
        require(set(p) <= allowed, "/")
        require(isinstance(p.get("failure_kind"), str) and 0 < len(p["failure_kind"]) <= 80, "/failure_kind")
        require(isinstance(p.get("description"), str) and p["description"].strip() != "" and len(p["description"]) <= 4096, "/description")
        for name in ("engine", "hardware", "model", "launch", "error", "logs", "privacy"):
            require(isinstance(p.get(name), dict), "/" + name)
        error, logs, privacy = p["error"], p["logs"], p["privacy"]
        require(isinstance(error.get("message"), str) and isinstance(error.get("chain"), list), "/error")
        require(isinstance(logs.get("startup"), list) and len(logs["startup"]) <= 220 and isinstance(logs.get("tail"), list) and len(logs["tail"]) <= 12, "/logs")
        require(all(isinstance(line, str) for line in [*logs["startup"], *logs["tail"], *error["chain"]]), "/logs")
        require(len("".join([*logs["startup"], *logs["tail"]]).encode("utf-8")) <= 32768, "/logs")
        require(logs.get("speed_statistics_included") is False and privacy.get("prompt_or_response_included") is False and privacy.get("api_keys_included") is False and privacy.get("local_paths_redacted") is True, "/privacy")
    else:
        require(False, "/kind")
    validate_privacy(p)


def validate_privacy(value: object) -> None:
    forbidden = {"hostname", "serial_number", "serial", "username", "user_name", "prompt", "response", "messages", "api_key", "password", "private_redactions"}
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in forbidden:
                raise ValueError("submission contains a private field")
            validate_privacy(child)
    elif isinstance(value, list):
        for child in value:
            validate_privacy(child)
    elif isinstance(value, str):
        if re.search(r"[A-Za-z]:[\\/](?!\[)|/(?:home|Users)/(?!\[)[^\s/]+", value, re.I):
            raise ValueError("submission contains an unredacted local path")
        if re.search(r"\bBearer\s+(?!\[REDACTED\]|<redacted>)[A-Za-z0-9._~+/-]{8,}|\bsk-[A-Za-z0-9_-]{12,}|\b(?:api[_-]?key|password|secret)\s*[=:]\s*[\"']?(?!\[REDACTED\]|<redacted>|null\b|false\b|true\b)[A-Za-z0-9._~+/-]{8,}", value, re.I):
            raise ValueError("submission contains a suspected credential")
