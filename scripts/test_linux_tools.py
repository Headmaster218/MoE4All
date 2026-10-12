#!/usr/bin/env python3
import contextlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import types
import unittest
from unittest import mock

import linux_release as release


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        for name in release.ROOT_FILES | release.SCRIPT_FILES:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="ascii")
        (self.source / "documentation").mkdir()
        (self.source / "documentation/README.md").write_text("guide\n")
        (self.source / "kv-sessions").mkdir()
        (self.source / "kv-sessions/private.infrkv").write_bytes(b"private")
        (self.source / "infr.toml").write_text("secret")
        self.output = self.root / "dist"

    def stage(self, version):
        archive = release.package(self.source, self.source / "infr", self.output, version)
        stage = self.root / ("stage-" + version)
        stage.mkdir()
        release.extract_package(archive, stage, version)
        return archive, stage

    def test_archive_checksum_manifest_permissions_and_private_exclusions(self):
        archive, stage = self.stage("0.10.0")
        manifest = release.validate_package(stage, "0.10.0")
        names = {entry["path"] for entry in manifest["files"]}
        self.assertNotIn("infr.toml", names)
        self.assertFalse(any("kv-sessions" in name for name in names))
        self.assertIn("documentation/README.md", names)
        self.assertFalse(any(name.endswith(".py") for name in names))
        self.assertEqual(manifest["stability"], "experimental")
        self.assertEqual(archive.with_name(archive.name + ".sha256").read_text().split()[0], release.sha256(archive))
        if os.name != "nt":
            self.assertTrue(os.access(stage / "infr", os.X_OK))

    def test_rejects_corrupt_manifest_unlisted_file_and_wrong_version(self):
        _, stage = self.stage("0.10.0")
        with self.assertRaises(ValueError):
            release.validate_package(stage, "0.11.0")
        (stage / "infr").write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            release.validate_package(stage, "0.10.0")
        (stage / "infr").write_text("fixture\n", encoding="ascii")
        (stage / "private.txt").write_text("unlisted")
        with self.assertRaises(ValueError):
            release.validate_package(stage, "0.10.0")

    def test_codec_package_is_managed_and_replaceable(self):
        codecs = self.root / "image-codecs"
        codecs.mkdir()
        for name in release.CODEC_FILES:
            (codecs / name).write_bytes(b"codec fixture")
        (self.root / "image-codec-sources.tar.gz").write_bytes(b"source fixture")
        archive = release.package(self.source, self.source / "infr", self.output, "0.10.0", codecs)
        stage = self.root / "codec-stage"
        stage.mkdir()
        manifest = release.extract_package(archive, stage, "0.10.0")
        names = {entry["path"] for entry in manifest["files"]}
        self.assertTrue({"image-codecs/" + n for n in release.CODEC_FILES}.issubset(names))
        self.assertFalse(any(name.endswith(".py") for name in names))
        self.assertTrue((self.output / "image-codec-sources.tar.gz").is_file())
        self.assertFalse(release.managed("image-codecs/evil.so"))
        (stage / "image-codecs/libheif.so.1").write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            release.validate_package(stage, "0.10.0")

    def test_rejects_path_traversal_links_duplicate_and_unmanaged_files(self):
        for name, kind in [("../outside", tarfile.REGTYPE), ("kv-sessions/private.infrkv", tarfile.REGTYPE), ("infr", tarfile.SYMTYPE), ("infr", tarfile.LNKTYPE)]:
            archive = self.root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                member = tarfile.TarInfo(release.package_name("0.10.0") + "/" + name)
                member.type = kind
                member.linkname = "outside"
                tar.addfile(member)
            target = self.root / "bad-stage"
            target.mkdir(exist_ok=True)
            with self.assertRaises(ValueError):
                release.extract_package(archive, target, "0.10.0")

    def test_rejects_duplicate_archive_members(self):
        archive = self.root / "duplicate.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for _ in range(2):
                tar.addfile(tarfile.TarInfo(release.package_name("0.10.0") + "/infr"))
        target = self.root / "duplicate-stage"
        target.mkdir()
        with self.assertRaises(ValueError):
            release.extract_package(archive, target, "0.10.0")

    @unittest.skipIf(os.name == "nt", "symlink creation requires Windows privileges")
    def test_update_rejects_symlinked_install_directory(self):
        _, installed = self.stage("0.10.0")
        _, stage = self.stage("0.11.0")
        outside = self.root / "outside"
        os.rename(installed / "documentation", outside)
        (installed / "documentation").symlink_to(outside, target_is_directory=True)
        with mock.patch.object(release, "engine_running", return_value=False):
            with self.assertRaises(ValueError):
                release.apply_update(installed, stage, "0.11.0")

    def test_fetch_rejects_untrusted_asset_urls(self):
        with self.assertRaises(ValueError):
            release.fetch("https://example.invalid/infr.tar.gz", self.root / "asset", 100)

    def test_download_checksum_failure_never_replaces_files(self):
        _, installed = self.stage("0.10.0")
        name = release.package_name("0.11.0") + ".tar.gz"
        chosen = {"version": "0.11.0", "name": name, "checksum": "checksum-url", "archive": "archive-url"}

        def fetch_fixture(url, output, maximum):
            output.write_bytes(("0" * 64 + "  " + name + "\n").encode() if url == "checksum-url" else b"corrupt archive")

        lock = types.SimpleNamespace(flock=lambda *args: None, LOCK_EX=1, LOCK_NB=2)
        with mock.patch.object(release.platform, "system", return_value="Linux"), mock.patch.object(release.platform, "machine", return_value="x86_64"), mock.patch.dict("sys.modules", {"fcntl": lock}), mock.patch.object(release, "fetch", side_effect=fetch_fixture), mock.patch.object(release, "apply_update") as apply:
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                release.download_update(installed, chosen)
            apply.assert_not_called()
        self.assertEqual(json.loads((installed / "install-manifest.json").read_text())["version"], "0.10.0")

    def test_update_preserves_user_data_and_checks_version(self):
        _, installed = self.stage("0.10.0")
        _, stage = self.stage("0.11.0")
        (installed / "infr.toml").write_text("private-config")
        (installed / "kv-sessions").mkdir()
        (installed / "kv-sessions/a.infrkv").write_bytes(b"cache")
        with mock.patch.object(release, "engine_running", return_value=False), mock.patch("subprocess.check_output", return_value="infr 0.11.0\n"):
            release.apply_update(installed, stage, "0.11.0")
        self.assertEqual(json.loads((installed / "install-manifest.json").read_text())["version"], "0.11.0")
        self.assertEqual((installed / "infr.toml").read_text(), "private-config")
        self.assertEqual((installed / "kv-sessions/a.infrkv").read_bytes(), b"cache")

    def test_update_rolls_back_if_executable_validation_fails(self):
        _, installed = self.stage("0.10.0")
        _, stage = self.stage("0.11.0")
        before = {p.relative_to(installed): p.read_bytes() for p in installed.rglob("*") if p.is_file()}
        with mock.patch.object(release, "engine_running", return_value=False), mock.patch("subprocess.check_output", side_effect=OSError("loader failed")):
            with self.assertRaises(OSError):
                release.apply_update(installed, stage, "0.11.0")
        after = {p.relative_to(installed): p.read_bytes() for p in installed.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_update_rolls_back_a_mid_replace_failure(self):
        _, installed = self.stage("0.10.0")
        _, stage = self.stage("0.11.0")
        before = {p.relative_to(installed): p.read_bytes() for p in installed.rglob("*") if p.is_file()}
        replace = os.replace
        calls = [0]

        def fail_once(source, target):
            calls[0] += 1
            if calls[0] == 3:
                raise OSError("injected replace failure")
            return replace(source, target)

        with mock.patch.object(release, "engine_running", return_value=False), mock.patch("os.replace", side_effect=fail_once):
            with self.assertRaises(OSError):
                release.apply_update(installed, stage, "0.11.0")
        self.assertEqual(before, {p.relative_to(installed): p.read_bytes() for p in installed.rglob("*") if p.is_file()})

    def test_running_engine_source_checkout_and_downgrade_are_rejected(self):
        _, installed = self.stage("0.10.0")
        _, stage = self.stage("0.11.0")
        with mock.patch.object(release, "engine_running", return_value=True):
            with self.assertRaises(ValueError):
                release.apply_update(installed, stage, "0.11.0")
        with self.assertRaises(ValueError):
            release.apply_update(installed, stage, "0.10.0")
        (installed / ".git").mkdir()
        with self.assertRaises(ValueError):
            release.apply_update(installed, stage, "0.11.0")

    def test_release_selection_checks_tags_prereleases_assets_and_numeric_order(self):
        releases = [{"tag_name": tag, "prerelease": pre, "assets": []} for tag, pre in [("release-0.9.0", False), ("release-0.10.0", False), ("agent-99.0.0", False), ("release-1.0.0", True)]]
        self.assertEqual(release.select_release(releases)["version"], "0.10.0")
        self.assertIsNone(release.select_release(releases)["archive"])


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        unittest.main(verbosity=2)
