import base64
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from moe4all_submission import _one_line, decode_text, generated_summary, strict_json
from submission_validation import validate_payload

ROOT = Path(__file__).resolve().parent
VECTORS = json.loads((ROOT / "fixtures/submission-vectors.json").read_text(encoding="utf-8"))


def encode(kind, payload):
    decoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    compressed = gzip.compress(decoded, mtime=0)
    summary = generated_summary(kind, payload)
    headers = ["MOE4ALL-SUBMISSION/1", "kind: " + kind, "schema: " + kind + "/1", "id: " + payload["id"], "created-at: " + payload["created_at_utc"], "engine: " + _one_line(payload["engine_version"], 80), "encoding: gzip+base64", "sha256: " + hashlib.sha256(decoded).hexdigest(), "decoded-bytes: " + str(len(decoded)), "compressed-bytes: " + str(len(compressed)), "summary-count: " + str(len(summary))]
    headers += [f"summary-{i:03}: {_one_line(line, 500)}" for i, line in enumerate(summary, 1)]
    text = base64.b64encode(compressed).decode("ascii")
    return "\n".join([*headers, "-----BEGIN MOE4ALL PAYLOAD-----", *[text[i:i + 76] for i in range(0, len(text), 76)], "-----END MOE4ALL PAYLOAD-----", ""])


def module(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / (name + ".py"))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


class SubmissionTests(unittest.TestCase):
    def test_shared_one_line_vectors(self):
        for case in VECTORS["one_line"]:
            with self.subTest(case=case["name"]):
                text = case.get("text", case.get("unit", "") * case.get("count", 0))
                expected = case.get("expected", case.get("unit", "") * case.get("expected_count", 0) + case.get("suffix", ""))
                self.assertEqual(_one_line(text, case["limit"]), expected)

    def test_shared_positive_and_negative_schema_vectors(self):
        for kind in ("benchmark", "error"):
            validate_payload(kind, VECTORS[kind])
            for ending in ("\n", "\r\n", "\r"):
                self.assertEqual(decode_text(encode(kind, VECTORS[kind]).replace("\n", ending)).payload, VECTORS[kind])
        for case in VECTORS["invalid"]:
            payload = copy.deepcopy(VECTORS[case["kind"]]); target = payload
            parts = case["path"].strip("/").split("/")
            for part in parts[:-1]:
                target = target[int(part)] if isinstance(target, list) else target[part]
            target[int(parts[-1]) if isinstance(target, list) else parts[-1]] = case["value"]
            with self.subTest(case=case):
                with self.assertRaises(ValueError): validate_payload(case["kind"], payload)

    def test_description_and_logs_boundaries(self):
        payload = copy.deepcopy(VECTORS["error"])
        payload["description"] = "\U0001f600" * 4096
        validate_payload("error", payload); self.assertEqual(decode_text(encode("error", payload)).payload, payload)
        payload["description"] += "x"
        with self.assertRaises(ValueError): validate_payload("error", payload)
        for key, lines in (("startup", 221), ("tail", 13)):
            payload = copy.deepcopy(VECTORS["error"]); payload["logs"][key] = ["log"] * lines
            with self.assertRaises(ValueError): validate_payload("error", payload)

    def test_duplicate_reserved_and_unsafe_json(self):
        for text in ('{"x":1,"x":2}', '{"x":1,"\\u0078":2}', '{"constructor":{}}', '{"x":1e400}', '{"x":9007199254740992}', '{"x":"\\ud800"}'):
            with self.assertRaises(ValueError): strict_json(text)

    def test_ci_validators_share_the_upload_contract(self):
        for kind in ("benchmark", "error"):
            validator = module("validate-" + kind + "-submission")
            with tempfile.TemporaryDirectory(prefix="submission-ci-") as directory:
                parent = Path(directory) / "2026-10"; parent.mkdir()
                path = parent / (VECTORS[kind]["id"] + (".m4b" if kind == "benchmark" else ".m4e"))
                path.write_text(encode(kind, VECTORS[kind]), encoding="utf-8")
                self.assertEqual(validator.validate_file(path), [])
                invalid = copy.deepcopy(VECTORS[kind])
                if kind == "benchmark": invalid["hardware"]["gpu"]["vram_total_bytes"] = -1
                else: invalid["description"] = "x" * 4097
                path.write_text(encode(kind, invalid), encoding="utf-8")
                self.assertTrue(validator.validate_file(path))

    def test_numeric_equivalence_and_empty_optional_text_summary(self):
        payload = copy.deepcopy(VECTORS["benchmark"])
        payload["model"]["architecture"] = ""; payload["model"]["storage_bytes"] = 1024.0
        payload["contexts"][0]["requested_context_tokens"] = 2048.0
        payload["contexts"][0]["decode"][0]["parallel"] = 1.0
        validate_payload("benchmark", payload)
        text = encode("benchmark", payload)
        self.assertIn("result 2K: prefill=1000.00 | x1=50.00", text)
        self.assertEqual(decode_text(text).payload, payload)


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="submission-pr-")
        self.addCleanup(self.directory.cleanup); self.root = Path(self.directory.name)
        self.environment = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_TERMINAL_PROMPT": "0"}
        self.git("init", "--initial-branch=main"); self.write("code.txt", "base"); self.base = self.commit()
        self.classifier = module("classify-submission-pr")

    def git(self, *args):
        return subprocess.check_output(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", *args], cwd=self.root, env=self.environment, stderr=subprocess.DEVNULL).decode().strip()

    def write(self, path, text):
        target = self.root / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(text)

    def commit(self):
        self.git("add", "."); self.git("commit", "-m", "fixture"); return self.git("rev-parse", "HEAD")

    def test_both_kinds_use_lightweight_ci(self):
        for kind, extension in (("benchmark", "m4b"), ("error", "m4e")):
            self.git("checkout", "-B", "topic-" + kind, self.base)
            self.write(f"{kind}-database/v1/2026-10/fixture.{extension}", "fixture")
            result = self.classifier.classify(self.base, self.commit(), self.root)
            self.assertEqual(result["submission_only"], "true"); self.assertEqual(result["submission_kind"], kind)

    def test_stale_fork_uses_merge_base_not_two_point_diff(self):
        self.write("code.txt", "upstream changed"); advanced = self.commit()
        self.git("checkout", "-b", "behind", self.base)
        self.write("benchmark-database/v1/2026-10/fixture.m4b", "fixture")
        self.assertEqual(self.classifier.classify(advanced, self.commit(), self.root)["submission_only"], "true")

    def test_mixed_changes_and_mutation_cannot_use_report_only_ci(self):
        self.write("error-database/v1/2026-10/fixture.m4e", "fixture"); report_commit = self.commit()
        self.write("error-database/v1/2026-10/fixture.m4e", "changed")
        with self.assertRaises(ValueError): self.classifier.classify(report_commit, self.commit(), self.root)
        self.write("benchmark-database/v1/2026-10/new.m4b", "new"); self.write("code.txt", "changed")
        with self.assertRaises(ValueError): self.classifier.classify(report_commit, self.commit(), self.root)

    def test_plain_code_change_and_unsafe_report_name(self):
        self.write("code.txt", "changed"); code_commit = self.commit()
        self.assertEqual(self.classifier.classify(self.base, code_commit, self.root)["submission_only"], "false")
        self.write("error-database/v1/2026-10/bad name.m4e", "bad")
        with self.assertRaises(ValueError): self.classifier.classify(code_commit, self.commit(), self.root)

    def test_generated_index_cannot_be_submitted_or_bundled_with_a_report(self):
        self.write("benchmark-database/index/v1/manifest.json", "{}")
        generated = self.commit()
        with self.assertRaises(ValueError): self.classifier.classify(self.base, generated, self.root)
        self.write("benchmark-database/v1/2026-10/fixture.m4b", "fixture")
        with self.assertRaises(ValueError): self.classifier.classify(self.base, self.commit(), self.root)


if __name__ == "__main__": unittest.main()
