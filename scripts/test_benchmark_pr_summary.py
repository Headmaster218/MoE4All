import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from benchmark_pr_summary import (MAX_COMMENT_BYTES, MARKER, build_review, cell, comparison,
                                  conditions, historical, matching_level, render_summary, rows, same_model)
from test_submission import VECTORS, encode


GIB = 1 << 30
BASE = "a" * 40
HEAD = "b" * 40


def fixture(identity="review-fixture"):
    payload = copy.deepcopy(VECTORS["benchmark"])
    payload["id"] = identity
    payload["model"].update(name="Example-35B-Q4_K_M.gguf", quantization="Q4_K_M", storage_bytes=24 * GIB)
    payload["hardware"]["gpu"].update(name="AMD Radeon RX 7900 XTX", vram_total_bytes=24 * GIB)
    payload["configuration"].update(backend="vulkan", kv_type_k="q8_0", kv_type_v="q8_0", real_context=True, actual_ubatch=4096)
    return payload


def entry(payload):
    return {"payload": payload}


def measured_row(payload):
    return rows(payload)[1]


class MatchingTests(unittest.TestCase):
    def test_all_five_priority_levels(self):
        current = fixture()
        other = fixture("other")
        self.assertEqual(matching_level(current, other), 1)
        other["hardware"]["ram"]["total_bytes"] = 32 * GIB
        self.assertEqual(matching_level(current, other), 2)
        other["hardware"]["gpu"]["name"] = "AMD Radeon RX 7900 XT"
        self.assertEqual(matching_level(current, other), 3)
        other["hardware"]["gpu"].update(name="NVIDIA GeForce RTX 4090", vram_total_bytes=24 * GIB)
        self.assertEqual(matching_level(current, other), 4)
        other["model"]["name"] = "Different-35B-Q4_K_M.gguf"
        self.assertIsNone(matching_level(current, other))

    def test_unknown_capacity_is_not_an_exact_match(self):
        a, b = fixture(), fixture("other")
        for p in (a, b):
            p["hardware"]["gpu"]["vram_total_bytes"] = None
        self.assertEqual(matching_level(a, b), 4)
        a, b = fixture(), fixture("other")
        for p in (a, b):
            p["hardware"]["ram"]["total_bytes"] = None
        self.assertEqual(matching_level(a, b), 2)

    def test_nominal_physical_capacity_and_option_spellings(self):
        a, b = fixture(), fixture("other")
        b["hardware"]["gpu"]["vram_total_bytes"] = 25753026560
        b["hardware"]["ram"]["total_bytes"] = 68111454208
        b["configuration"].update(kv_type_k="Some(Q8_0)", kv_type_v="Some(Q8_0)")
        del b["configuration"]["backend"]
        b["hardware"]["gpu"]["configured_device"] = "Vulkan1"
        self.assertEqual(matching_level(a, b), 1)

    def test_model_shards_and_quantization_are_not_mixed(self):
        a, b = fixture(), fixture("other")
        b["model"]["name"] = "Example-35B-Q4_K_M-00001-of-00003.gguf"
        self.assertTrue(same_model(a, b))
        b["model"]["quantization"] = "Q8_0"
        self.assertFalse(same_model(a, b))
        b = fixture("other")
        b["model"]["storage_bytes"] *= 1.1
        self.assertFalse(same_model(a, b))

    def test_backend_kv_mtp_and_synthetic_are_never_relaxed(self):
        a = fixture()
        for field, value in (("backend", "cpu"), ("kv_type_k", "f16"), ("mtp", True), ("decode_measurement", "time")):
            b = fixture("other")
            b["configuration"][field] = value
            self.assertIsNone(matching_level(a, b), field)
        b = fixture("other")
        b["configuration"].update(real_context=False, synthetic_depth=True)
        self.assertNotEqual(conditions(a), conditions(b))

    def test_no_mixing_priorities_or_target_self(self):
        current = fixture()
        exact = fixture("exact")
        lower = fixture("lower")
        lower["hardware"]["ram"]["total_bytes"] = 32 * GIB
        lower["contexts"][0]["decode"][0]["overall_tokens_per_second"] = 999
        level, matches = historical(current, measured_row(current), [entry(current), entry(lower), entry(exact), entry(exact)])
        self.assertEqual(level, 1)
        self.assertEqual([m["payload"]["id"] for m in matches], ["exact"])
        self.assertEqual(matches[0]["rate"], 50)

    def test_each_workload_selects_its_own_bucket(self):
        current = fixture()
        exact = fixture("exact")
        exact["contexts"][0]["decode"] = []
        lower = fixture("lower")
        lower["hardware"]["ram"]["total_bytes"] = 32 * GIB
        refs = [entry(exact), entry(lower)]
        self.assertEqual(historical(current, rows(current)[0], refs)[0], 1)
        self.assertEqual(historical(current, measured_row(current), refs)[0], 2)

    def test_wrong_context_or_parallel_is_not_a_reference(self):
        current = fixture()
        wrong_context = fixture("context")
        wrong_context["contexts"][0]["requested_context_tokens"] *= 2
        wrong_parallel = fixture("parallel")
        wrong_parallel["contexts"][0]["decode"][0]["parallel"] = 2
        self.assertEqual(historical(current, measured_row(current), [entry(wrong_context), entry(wrong_parallel)]), (None, []))

    def test_hint_thresholds_are_advisory_and_sparse_is_explicit(self):
        current = fixture()
        refs = [entry(fixture(f"ref-{i}")) for i in range(3)]
        row = measured_row(current)
        for value, expected in ((50, "接近"), (70, "偏高"), (80, "明显偏高"), (20, "明显偏低")):
            row["rate"] = value
            self.assertIn(expected, comparison(current, row, refs)["note"])
        self.assertIn("样本较少", comparison(current, row, refs[:1])["note"])
        refs[0]["payload"]["hardware"]["gpu"]["name"] = "NVIDIA RTX 4090"
        self.assertNotIn("偏低", comparison(current, row, refs[:1])["note"])
        self.assertIn("近似硬件参考", comparison(current, row, refs[:1])["note"])

    def test_failure_partial_zero_and_fluctuation(self):
        current = fixture()
        other = fixture("other")
        other["contexts"][0]["failures"] = [{"parallel": 1, "stage": "decode"}]
        self.assertEqual(historical(current, measured_row(current), [entry(other)]), (None, []))
        other["contexts"][0]["failures"] = []
        other["contexts"][0]["decode"][0]["overall_tokens_per_second"] = 0
        self.assertEqual(historical(current, measured_row(current), [entry(other)]), (None, []))
        row = measured_row(current)
        row["sections"] = [50, 50, 0]
        self.assertIn("波动", comparison(current, row, [entry(fixture("ref"))])["note"])
        self.assertIn("波动", comparison(current, row, [])["note"])
        row["partial"] = True
        self.assertIn("不比较", comparison(current, row, [])['note'])


class RenderingTests(unittest.TestCase):
    def test_three_tables_no_mtp_column_source_pinned(self):
        current = fixture()
        body = render_summary(current, [entry(fixture("ref"))], "owner/repo", BASE, HEAD, "benchmark-database/v1/2026-10/review-fixture.m4b")
        self.assertTrue(body.startswith(MARKER))
        self.assertEqual(body.count("\n|---|"), 3)
        self.assertNotIn("MTP", body)
        self.assertIn("同模型、GPU、RAM", body)
        self.assertIn("/blob/" + BASE, body)
        self.assertIn("/blob/" + HEAD, body)

    def test_no_reference_and_other_hardware_are_visible(self):
        current = fixture()
        body = render_summary(current, [], "owner/repo", BASE, HEAD, "benchmark-database/v1/2026-10/review-fixture.m4b")
        self.assertIn("暂无参考", body)
        other = fixture("other")
        other["hardware"]["gpu"]["name"] = "NVIDIA RTX 4090"
        body = render_summary(current, [entry(other)], "owner/repo", BASE, HEAD, "benchmark-database/v1/2026-10/review-fixture.m4b")
        self.assertIn("NVIDIA RTX 4090", body)
        self.assertIn("64.0G RAM", body)

    def test_markdown_html_mentions_controls_cannot_escape_cells(self):
        escaped = cell('<script> @everyone | [click](https://bad)\n**bad**\u202e')
        for raw in ("<script>", "@everyone", "|", "[click]", "\n", "**", "\u202e"):
            self.assertNotIn(raw, escaped)
        current = fixture()
        current["engine_version"] = "@everyone | <img src=x>"
        body = render_summary(current, [], "owner/repo", BASE, HEAD, "benchmark-database/v1/2026-10/review-fixture.m4b")
        self.assertNotIn("@everyone", body)
        self.assertNotIn("<img", body)

    def test_manual_results_are_separate_from_suite(self):
        current = fixture()
        current["profile"] = "manual"
        current["contexts"][0]["manual_result"] = {"kind": "decode", "parallel": 1, "average_tokens_per_second": 50}
        self.assertEqual(len(rows(current)), 1)
        self.assertEqual(rows(current)[0]["key"], (2048, "manual_decode", 1))
        self.assertEqual(historical(current, rows(current)[0], [entry(fixture("ref"))]), (None, []))

    def test_large_report_is_bounded_and_links_to_original(self):
        current = fixture()
        original = current["contexts"][0]
        current["contexts"] = [{**copy.deepcopy(original), "requested_context_tokens": 2048 * (i + 1)} for i in range(64)]
        body = render_summary(current, [], "owner/repo", BASE, HEAD, "benchmark-database/v1/2026-10/review-fixture.m4b")
        self.assertLess(len(body.encode()), MAX_COMMENT_BYTES)
        self.assertIn("其余 88 项", body)


class GitReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-benchmark-review-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "user.name", "Fixture")
        (self.root / "placeholder").write_text("fixture", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-qm", "base")
        self.base = self.git("rev-parse", "HEAD").strip()

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, text=True)

    def report(self, payload=None, text=None):
        payload = payload or fixture()
        path = self.root / "benchmark-database/v1/2026-10" / (payload['id'] + ".m4b")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text if text is not None else encode("benchmark", payload), encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-qm", "report")
        return self.git("rev-parse", "HEAD").strip()

    def test_real_git_blob_review_without_executing_head(self):
        head = self.report()
        result = build_review(self.root, self.base, head, self.base, "owner/repo")
        self.assertEqual(result['status'], 'ready')
        self.assertIn("暂无参考", result['body'])

    def test_code_only_is_skipped_and_mixed_pr_is_invalid(self):
        (self.root / "code.py").write_text("raise RuntimeError('must never run')", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-qm", "code")
        head = self.git("rev-parse", "HEAD").strip()
        self.assertEqual(build_review(self.root, self.base, head, self.base, "owner/repo")['status'], 'skipped')
        head = self.report()
        result = build_review(self.root, self.base, head, self.base, "owner/repo")
        self.assertEqual(result['status'], 'invalid')

    def test_corrupt_oversized_and_unsafe_mode_reports_are_not_rendered(self):
        head = self.report(text="corrupt")
        self.assertEqual(build_review(self.root, self.base, head, self.base, "owner/repo")['status'], 'invalid')
        self.git("reset", "--soft", self.base)
        head = self.report(text="x" * (128 * 1024 + 1))
        self.assertEqual(build_review(self.root, self.base, head, self.base, "owner/repo")['status'], 'invalid')
        self.git("update-index", "--chmod=+x", "benchmark-database/v1/2026-10/review-fixture.m4b")
        self.git("commit", "-qm", "mode")
        head = self.git("rev-parse", "HEAD").strip()
        self.assertEqual(build_review(self.root, self.base, head, self.base, "owner/repo")['status'], 'invalid')

    def test_wrong_month_and_unsafe_refs(self):
        p = fixture()
        p['created_at_utc'] = '2026-09-19T00:00:00Z'
        head = self.report(p)
        self.assertEqual(build_review(self.root, self.base, head, self.base, "owner/repo")['status'], 'invalid')
        with self.assertRaises(ValueError):
            build_review(self.root, '--bad-ref', head, self.base, "owner/repo")

    def test_index_only_pr_does_not_receive_a_report_warning(self):
        path = self.root / 'benchmark-database/index/v1/manifest.json'
        path.parent.mkdir(parents=True)
        path.write_text('{}', encoding='utf-8')
        self.git('add', '.')
        self.git('commit', '-qm', 'index only')
        head = self.git('rev-parse', 'HEAD').strip()
        self.assertEqual(build_review(self.root, self.base, head, self.base, 'owner/repo')['status'], 'skipped')


if __name__ == '__main__':
    unittest.main()
