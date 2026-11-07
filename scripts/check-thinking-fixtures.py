#!/usr/bin/env python3
"""Check the shared Qwen fixture cases with Python Jinja2 (requires Jinja2).

This validates the fixture expectations; it does NOT execute MoE4All's Rust
renderer, HTTP handlers, or model inference. Run the Rust tests separately.
"""

import json
from pathlib import Path

from jinja2 import Environment, TemplateError

FIXTURES = Path(__file__).resolve().parents[1] / "crates/infr-chat/tests/fixtures"


def raise_exception(message):
    raise TemplateError(message)


def main():
    env = Environment()
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
    template = env.from_string(
        (FIXTURES / "qwen38_chat_template.jinja").read_text(encoding="utf-8")
    )
    cases = json.loads((FIXTURES / "thinking_cases.json").read_text(encoding="utf-8"))
    failures = []
    for case in cases:
        context = {
            "messages": case["messages"],
            "tools": case.get("tools"),
            "bos_token": "",
            "eos_token": "",
            "add_generation_prompt": case.get("generation", True),
            **case["options"],
        }
        try:
            output = template.render(**context)
        except TemplateError as error:
            if case.get("error") and case["error"] in str(error):
                continue
            failures.append(f"{case['name']}: unexpected error: {error}")
            continue
        if "error" in case:
            failures.append(f"{case['name']}: expected rejection, rendered successfully")
        for marker in case.get("contains", []):
            if marker not in output:
                failures.append(f"{case['name']}: missing {marker!r}")
        for marker in case.get("excludes", []):
            if marker in output:
                failures.append(f"{case['name']}: unexpected {marker!r}")
        if "suffix" in case and not output.endswith(case["suffix"]):
            failures.append(f"{case['name']}: wrong generation suffix")
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        raise SystemExit(1)
    print(f"PASS: {len(cases)} Qwen3.8 fixture cases (Python Jinja2 only)")


if __name__ == "__main__":
    main()
