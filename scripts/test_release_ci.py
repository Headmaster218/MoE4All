#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import tarfile
import unittest

import linux_release
import macos_release
import release_ci


COMMIT = "1" * 40
VERSION = "1.0.2"
REPOSITORY = "Headmaster218/MoE4All"


class ReleaseCiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.root = self.workspace / "artifacts"
        self.root.mkdir()
        self.output = self.workspace / "output"

    def tearDown(self):
        self.temporary.cleanup()

    def make_platform(self, platform_name, with_sources=False):
        directory = self.root / platform_name
        directory.mkdir()
        archive_name = release_ci.expected_archive(platform_name, VERSION)
        archive = directory / archive_name
        archive.write_bytes((platform_name + " archive").encode())
        checksum = directory / (archive_name + ".sha256")
        checksum.write_text(f"{release_ci.sha256(archive)}  {archive_name}\n", encoding="ascii")
        files = [archive, checksum]
        if with_sources:
            sources = directory / "image-codec-sources.tar.gz"
            sources.write_bytes(b"corresponding source")
            files.append(sources)
        manifest = directory / f"release-manifest-{platform_name}.json"
        release_ci.write_manifest(
            platform_name, VERSION, COMMIT, REPOSITORY, manifest, files,
            ["rustc=rustc 1.97.1", "runner=test"],
        )
        return directory, manifest

    def test_prepare_accepts_exact_three_platform_build(self):
        self.make_platform("windows-x86_64", with_sources=True)
        self.make_platform("linux-x86_64")
        self.make_platform("macos-arm64")
        output = self.output
        files = release_ci.prepare_promotion(self.root, output, COMMIT, VERSION, REPOSITORY)
        self.assertEqual(9, len(files))
        combined = json.loads((output / "release-build-manifest.json").read_text())
        self.assertEqual(COMMIT, combined["commit"])
        self.assertEqual(3, len(combined["platforms"]))
        self.assertNotIn(b"\r\n", (output / "SHA256SUMS").read_bytes())
        published_names = [entry["name"] for entry in combined["published_files"]]
        self.assertEqual(sorted(published_names), published_names)
        checksum_names = [
            line.split("  ", 1)[1]
            for line in (output / "SHA256SUMS").read_text(encoding="ascii").splitlines()
        ]
        self.assertEqual(sorted(checksum_names), checksum_names)

    def test_prepare_rejects_wrong_commit(self):
        self.make_platform("windows-x86_64", with_sources=True)
        self.make_platform("linux-x86_64")
        self.make_platform("macos-arm64")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            release_ci.prepare_promotion(
                self.root, self.output, "2" * 40, VERSION, REPOSITORY
            )

    def test_prepare_rejects_wrong_repository(self):
        self.make_platform("windows-x86_64", with_sources=True)
        self.make_platform("linux-x86_64")
        self.make_platform("macos-arm64")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            release_ci.prepare_promotion(
                self.root, self.output, COMMIT, VERSION, "someone-else/MoE4All"
            )

    def test_verify_rejects_tampered_archive(self):
        directory, manifest = self.make_platform("windows-x86_64", with_sources=True)
        archive = directory / release_ci.expected_archive("windows-x86_64", VERSION)
        archive.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            release_ci.verify_manifest(manifest, COMMIT, VERSION, REPOSITORY)

    def test_verify_rejects_tampered_checksum(self):
        directory, manifest = self.make_platform("windows-x86_64", with_sources=True)
        archive_name = release_ci.expected_archive("windows-x86_64", VERSION)
        (directory / (archive_name + ".sha256")).write_text(
            ("0" * 64) + "  " + archive_name + "\n", encoding="ascii"
        )
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            release_ci.verify_manifest(manifest, COMMIT, VERSION, REPOSITORY)

    def test_prepare_requires_all_platforms(self):
        self.make_platform("windows-x86_64", with_sources=True)
        self.make_platform("linux-x86_64")
        with self.assertRaisesRegex(ValueError, "Incomplete release artifacts"):
            release_ci.prepare_promotion(
                self.root, self.output, COMMIT, VERSION, REPOSITORY
            )

    def test_workspace_version(self):
        cargo = self.root / "Cargo.toml"
        cargo.write_text('[workspace.package]\nversion = "1.2.3"\n', encoding="utf-8")
        self.assertEqual("1.2.3", release_ci.workspace_version(cargo))

    def test_unix_packagers_preserve_prerelease_identity_and_reject_unsafe_versions(self):
        for module in (linux_release, macos_release):
            source = self.workspace / module.PLATFORM
            source.mkdir()
            (source / "documentation").mkdir()
            for name in module.ROOT_FILES | module.SCRIPT_FILES:
                (source / name).parent.mkdir(parents=True, exist_ok=True)
                (source / name).write_bytes(b"fixture\n")
            output = source / "output"
            for version in ("1.1.0", "1.1.0-beta.1", "1.1.0-rc.2", "1.1.0+build.7"):
                with self.subTest(platform=module.PLATFORM, version=version):
                    archive = module.package(source, source / "infr", output, version)
                    prefix = f"MoE4All-{module.PLATFORM}-v{version}"
                    self.assertEqual(archive.name, prefix + ".tar.gz")
                    with tarfile.open(archive, "r:gz") as bundle:
                        manifest = json.load(bundle.extractfile(prefix + "/install-manifest.json"))
                    self.assertEqual(manifest["version"], version)
                    self.assertEqual(manifest["tag"], "release-" + version)
                    self.assertEqual(archive.with_name(archive.name + ".sha256").read_text().split()[0], release_ci.sha256(archive))
            for invalid in ("../1.1.0", "1.1.0-beta/1", "1.1.0 beta", "release-1.1.0-beta.1"):
                with self.assertRaises(ValueError):
                    module.package_name(invalid)

    def test_prerelease_package_support_does_not_change_stable_update_selection(self):
        releases = [
            {"tag_name": "release-1.0.3", "prerelease": False},
            {"tag_name": "release-1.1.0-beta.1", "prerelease": True},
            {"tag_name": "release-1.1.0-rc.2", "prerelease": False},
            {"tag_name": "release-1.1.0", "draft": True},
        ]
        self.assertEqual(linux_release.select_release(releases)["version"], "1.0.3")


if __name__ == "__main__":
    unittest.main()
