"""Portable setup CLI regressions; no model downloads or conversion dependencies."""
import argparse
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import convert_melotts_chinese as converter


class SetupTests(unittest.TestCase):
    def setUp(self):
        # CI uses its supplied TMPDIR; local runs explicitly supply the shared task temp.
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def args(self, **values):
        defaults = dict(work_root=self.root, task_id=None, output_dir=None)
        defaults.update(values)
        return argparse.Namespace(**defaults)

    def test_explicit_output_and_legacy_layout(self):
        target = self.root / "models with spaces" / "MeloTTS-Chinese"
        root, identity, output = converter.resolve_paths(self.args(output_dir=target))
        self.assertEqual(root, self.root.resolve())
        self.assertEqual(output, target.resolve())
        self.assertTrue(identity.startswith("20"))
        self.assertNotEqual(identity, converter.resolve_paths(self.args())[1])
        _, identity, output = converter.resolve_paths(self.args(task_id="test-task"))
        self.assertEqual(identity, "test-task")
        self.assertIn(self.root.resolve(), output.parents)
        self.assertEqual(output.parent.name, "models")

    def test_invalid_task_component(self):
        for value in ("..", "../outside", "drive:path", "a/b", "a\\b"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                converter.resolve_paths(self.args(task_id=value))

    def test_missing_dependencies_fail_before_download_or_writes(self):
        destination = self.root / "not-created"
        argv = ["converter", "--work-root", str(destination)]
        with mock.patch("sys.argv", argv), mock.patch.object(converter.importlib.util, "find_spec", return_value=None), mock.patch.object(converter, "acquire") as acquire, contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as raised:
                converter.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("never installs dependencies", errors.getvalue())
        acquire.assert_not_called()
        self.assertFalse(destination.exists())

    def test_existing_model_is_not_overwritten(self):
        destination = self.root / "voice"
        destination.mkdir()
        model = destination / "melotts-chinese-f32.gguf"
        model.write_bytes(b"keep-existing-model")
        argv = ["converter", "--work-root", str(self.root / "work"), "--output-dir", str(destination)]
        with mock.patch("sys.argv", argv), mock.patch.object(converter.importlib.util, "find_spec", return_value=object()), mock.patch.object(converter, "acquire") as acquire, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                converter.main()
        self.assertEqual(raised.exception.code, 2)
        acquire.assert_not_called()
        self.assertEqual(model.read_bytes(), b"keep-existing-model")

    def test_download_requires_existing_curl(self):
        with mock.patch.object(converter.shutil, "which", return_value=None), self.assertRaisesRegex(RuntimeError, "no software is installed"):
            converter.download("https://example.invalid/not-requested", self.root / "checkpoint.pth")
        self.assertFalse((self.root / "checkpoint.pth").exists())


if __name__ == "__main__":
    unittest.main()
