#!/usr/bin/env python3
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import tarfile
import tempfile
import types
import unittest
from unittest import mock

import linux_release as release
import linux_wizard as wizard


def state(**values):
    result = dict(wizard.DEFAULTS, model="/models/model with spaces.gguf")
    result.update(values)
    return result


def settings(command):
    return dict(arg.split("=", 1) for index, arg in enumerate(command) if index and command[index - 1] == "--set")


class WizardTests(unittest.TestCase):
    def test_profiles_and_automatic_overrides(self):
        for profile in ("conservative", "aggressive", "manual"):
            with self.subTest(profile=profile):
                s = state(setup_mode=profile, context="150k", ubatch="3072", ram_budget="48g", auto_overrides="ubatch,ram_budget", configure_memory=True)
                command = wizard.build_command(s, "/pkg/infr")
                self.assertEqual(command[-1], s["model"])
                self.assertIn("150k", command)
                self.assertIn("3072", command)
                self.assertEqual(settings(command)["device.ram_budget"], "48g")
                self.assertEqual(settings(command)["kv.type_k"], "q8_0")
                self.assertEqual("device.auto_profile" in settings(command), profile != "manual")

    def test_inactive_manual_values_do_not_leak_to_auto(self):
        command = wizard.build_command(state(ubatch="512", ram_budget="8g", threads="2", custom_sets="bad"), "infr")
        self.assertNotIn("--ubatch", command)
        self.assertNotIn("--threads", command)
        self.assertNotIn("device.ram_budget", settings(command))

    def test_manual_memory_splitter_diagnostics_and_sampling(self):
        s = state(setup_mode="manual", configure_memory=True, vram_budget="24g", vram_reserve="1g", expert_cache="2g", pager_ring="1g", pager_ring_slots="3", host_dma=False, kv_overflow=True, kv_overflow_vram_mb="1024", kv_overflow_reserve_mb="256", submit_mode="fixed", submit_cap="256", configure_diagnostics=True, pager_stats=True, configure_sampling=True, temperature="0.7", top_k="20", top_p="0.9", seed="12", think_mode="think", reasoning_effort="medium", custom_sets="paging.expert_prefetch=false")
        command = wizard.build_command(s, "infr")
        values = settings(command)
        self.assertEqual(values["paging.host_dma"], "false")
        self.assertEqual(values["kv.overflow_vram_mb"], "1024")
        self.assertEqual(values["device.submit_dispatches"], "256")
        self.assertEqual(values["paging.stats"], "true")
        self.assertEqual(values["paging.expert_prefetch"], "false")
        for item in ("--think", "--reasoning-effort", "medium", "--temp", "0.7", "--seed", "12"):
            self.assertIn(item, command)

    def test_mtp_slots_vision_embedding_cache_matrix(self):
        for slots in ("1", "2"):
            for vision in (False, True):
                for embedding in (False, True):
                    with self.subTest(slots=slots, vision=vision, embedding=embedding):
                        s = state(launch_mode="serve", mtp_enabled=True, mtp_model="head.gguf", mtp_verify_tokens="2", server_parallel=slots, server_vision=vision, vision_projector="vision.gguf", server_embedding=embedding, embedding_model="embed.gguf", server_session_cache=True, session_cache_dir="/pkg/kv-sessions")
                        command = wizard.build_command(s, "infr")
                        values = settings(command)
                        self.assertEqual(values["spec.k"], "4" if slots == "2" else "2")
                        self.assertIn("0", command)
                        self.assertEqual("--mmproj" in command, vision)
                        self.assertEqual("--embedding-model" in command, embedding)
                        self.assertEqual(values["kv.session_cache_dir"] != "", slots == "2" or vision or embedding)

    def test_cpu_miss_1_2_3_and_unsupported_settings(self):
        for maximum in ("1", "2", "3"):
            command = wizard.build_command(state(cpu_miss_enabled=True, cpu_miss_cores="4", cpu_miss_max=maximum), "infr")
            self.assertEqual(settings(command)["kernels.vulkan.cpu_miss_max"], maximum)
        for maximum in ("0", "4"):
            with self.assertRaises(ValueError):
                wizard.build_command(state(cpu_miss_enabled=True, cpu_miss_cores="4", cpu_miss_max=maximum), "infr")

    def test_benchmark_modes_and_depth(self):
        for kind in ("decode", "prefill", "mixed", "custom"):
            for depth in ("none", "real", "synthetic"):
                command = wizard.build_command(state(launch_mode="bench", bench_kind=kind, depth_mode=depth, depth_tokens="30000", json_output=True), "infr")
                self.assertNotIn("--max-new", command)
                self.assertEqual("--pg" in command, kind == "mixed")
                self.assertEqual("--synthetic-depth" in command, depth == "synthetic")
                self.assertIn("--json", command)

    def test_auth_and_cache_disabled_explicitly(self):
        values = settings(wizard.build_command(state(launch_mode="serve"), "infr"))
        self.assertEqual(values["serve.api_key"], "")
        self.assertEqual(values["kv.session_cache_dir"], "")
        with self.assertRaises(ValueError):
            wizard.build_command(state(setup_mode="manual", custom_sets="serve.api_key=secret"), "infr")

    def test_legacy_import_and_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "wizard.conf"
            legacy.write_text("MODE=serve\nMODEL=/tmp/model.gguf\nPROFILE=aggressive\nUBATCH=3072\nRAM=48g\nMTP=/tmp/head.gguf\nMMPROJ=/tmp/mmproj.gguf\nEMBEDDING=/tmp/embed.gguf\n", encoding="utf-8")
            s, path = wizard.load_state(root, root)
            self.assertTrue(s["mtp_enabled"])
            self.assertTrue(s["server_vision"])
            self.assertTrue(s["server_embedding"])
            self.assertIn("3072", wizard.build_command(s, "infr"))
            release.write_json_atomic(path, s, 0o600)
            self.assertEqual(wizard.load_state(root, root)[0], s)

    def test_prompts_defaults_clear_and_eof(self):
        with mock.patch("builtins.input", side_effect=["", "-", "", "", "2"]):
            self.assertEqual(wizard.prompt("projector", "saved.gguf"), "saved.gguf")
            self.assertEqual(wizard.prompt("clear", "saved.gguf"), "")
            self.assertFalse(wizard.yes("update", False))
            self.assertTrue(wizard.yes("reuse", True))
            self.assertEqual(wizard.choice("mode", "run", [("run", "chat"), ("serve", "API")]), "serve")
        with mock.patch("builtins.input", side_effect=EOFError):
            with self.assertRaises(EOFError):
                wizard.yes("launch", True)

    def test_full_server_flow_order(self):
        s = state(launch_mode="serve")
        questions = []

        def input_default(label):
            questions.append(label)
            if "1. Purpose" in current[0]:
                return "2"
            return ""

        current = [""]
        original_choice = wizard.choice

        def choose(label, default, options):
            current[0] = label
            result = original_choice(label, default, options)
            current[0] = ""
            return result

        with mock.patch("builtins.input", side_effect=input_default), mock.patch.object(wizard, "choice", side_effect=choose), mock.patch.object(wizard, "model_path", return_value="/tmp/model.gguf"), mock.patch.object(wizard, "enumerate_devices", return_value=([("Vulkan0", "GPU")], "Vulkan0")):
            wizard.configure(s, Path("/tmp"), "stub")
        numbered = [next(i for i, q in enumerate(questions) if text in q) for text in (
            "2.2", "2.3", "2.4", "Configure default reasoning", "5.", "6.", "7.", "8.", "9.")]
        self.assertEqual(numbered, sorted(numbered))
        self.assertEqual(s["max_new"], "65536")

    def test_devices_exclude_cpu_and_keep_default(self):
        output = " Vulkan0: llvmpipe [cpu, 1 GiB device-local]\n Vulkan1: GPU A [discrete, 8 GiB device-local]\n Vulkan2: GPU B [discrete, 24 GiB device-local] <- default\n"
        with mock.patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, output, "")):
            devices, default = wizard.enumerate_devices("infr")
        self.assertEqual([v for v, _ in devices], ["Vulkan1", "Vulkan2"])
        self.assertEqual(default, "Vulkan2")

    def test_model_directories_match_windows_filtering_and_reject_ambiguity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "models"
            model.mkdir()
            for name in ("flash-00001-of-00002.gguf", "flash-00002-of-00002.gguf", "mmproj-flash.gguf", "mtp-flash.gguf"):
                (model / name).touch()
            self.assertEqual([p.name for p in wizard.model_candidates(model, "llm")], ["flash-00001-of-00002.gguf"])
            self.assertEqual([p.name for p in wizard.model_candidates(model, "vision")], ["mmproj-flash.gguf"])
            with mock.patch("builtins.input", return_value="models"):
                self.assertEqual(wizard.model_path("llm", "", root), str((model / "flash-00001-of-00002.gguf").resolve()))
            with mock.patch("builtins.input", side_effect=["models", "models/mtp-flash.gguf"]):
                self.assertEqual(wizard.model_path("mtp", "", root), str((model / "mtp-flash.gguf").resolve()))
            (model / "another.gguf").touch()
            with mock.patch("builtins.input", side_effect=["models", "models/another.gguf"]):
                self.assertEqual(wizard.model_path("llm", "", root), str((model / "another.gguf").resolve()))

    def test_old_reasoning_state_keeps_configure_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release.write_json_atomic(root / "wizard-state.json", {"think_mode": "think", "reasoning_effort": "medium"})
            self.assertTrue(wizard.load_state(root, root)[0]["configure_thinking"])

    def test_model_choice_reuses_saved_and_gui_recent_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "models").mkdir()
            model = root / "models/saved.gguf"
            other = root / "models/other.gguf"
            model.touch()
            other.touch()
            (root / "gui-data").mkdir()
            release.write_json_atomic(root / "gui-data/state.json", {"recent": [str(other)]})
            with mock.patch("builtins.input", return_value="2"):
                self.assertEqual(wizard.model_path("llm", str(model), root), str(other.resolve()))
            with mock.patch("builtins.input", return_value=""):
                self.assertEqual(wizard.model_path("llm", str(model), root), str(model.resolve()))

    def test_gguf_reasoning_capabilities_and_corrupt_files(self):
        def text(value):
            data = value.encode()
            return struct.pack("<Q", len(data)) + data

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.gguf"
            for arch, template, expected in [("qwen4exp", "reasoning_effort", ["low", "medium", "xhigh"]), ("qwen35moe", "reasoning_effort", ["low", "medium", "high", "max"]), ("qwen35moe", "enable_thinking", [])]:
                data = b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
                for key, value in [("general.architecture", arch), ("tokenizer.chat_template", template)]:
                    data += text(key) + struct.pack("<I", 8) + text(value)
                path.write_bytes(data)
                self.assertEqual(wizard.reasoning_efforts(path), expected)
            path.write_bytes(b"GGUF")
            with self.assertRaises(ValueError):
                wizard.reasoning_efforts(path)

    def test_main_headless_auth_and_no_launch_without_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(root), "INFR_API_KEY": "test-secret"}), mock.patch.object(wizard, "find_binary", return_value=root / "infr"), mock.patch.object(wizard.sys.stdin, "isatty", return_value=False), mock.patch("os.chdir"), mock.patch("os.execve") as execute:
                wizard.main(["--root", str(root), "--mode", "serve", "--model", "model.gguf", "--yes", "--no-api-key"])
                self.assertNotIn("INFR_API_KEY", execute.call_args.args[2])
                saved = (root / "infr/wizard-state.json").read_text()
                self.assertNotIn("test-secret", saved)
                execute.reset_mock()
                with self.assertRaises(ValueError):
                    wizard.main(["--root", str(root), "--mode", "serve", "--model", "model.gguf", "--addr", "0.0.0.0:8080", "--no-api-key"])
                execute.assert_not_called()

    def test_main_cli_overrides_saved_values_and_dry_run_is_side_effect_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "infr"
            directory.mkdir()
            saved = state(launch_mode="serve", kv_preset="auto", mtp_enabled=True, mtp_model="head.gguf", server_vision=True, vision_projector="vision.gguf", ubatch="2048")
            path = directory / "wizard-state.json"
            release.write_json_atomic(path, saved)
            before = path.read_bytes()
            with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(root), "INFR_API_KEY": "fixture-secret"}), mock.patch.object(wizard, "find_binary", return_value=root / "infr-bin"), mock.patch.object(wizard.sys.stdin, "isatty", return_value=True), mock.patch.object(wizard, "check_updates") as check, mock.patch("os.execve") as execute, contextlib.redirect_stdout(io.StringIO()) as output:
                wizard.main(["--root", str(root), "--dry-run", "--model", "relative.gguf", "--ubatch", "3072", "--kv-k", "q8_0", "--no-mtp", "--no-mmproj", "--no-api-key"])
            self.assertNotIn("fixture-secret", output.getvalue())
            self.assertIn("3072", output.getvalue())
            self.assertIn("kv.type_k=q8_0", output.getvalue())
            self.assertNotIn("--mmproj", output.getvalue())
            self.assertNotIn("spec.draft", output.getvalue())
            self.assertIn(str((root / "relative.gguf").resolve()), output.getvalue())
            self.assertEqual(before, path.read_bytes())
            execute.assert_not_called()
            check.assert_not_called()

    def test_main_inherited_key_reaches_child_but_not_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(root), "INFR_API_KEY": "fixture-secret"}), mock.patch.object(wizard, "find_binary", return_value=root / "infr"), mock.patch.object(wizard.sys.stdin, "isatty", return_value=False), mock.patch("os.chdir"), mock.patch("os.execve") as execute, contextlib.redirect_stdout(io.StringIO()) as output:
                wizard.main(["--root", str(root), "--mode", "serve", "--model", "model.gguf", "--yes"])
            self.assertEqual(execute.call_args.args[2]["INFR_API_KEY"], "fixture-secret")
            self.assertNotIn("fixture-secret", output.getvalue())
            self.assertNotIn("fixture-secret", (root / "infr/wizard-state.json").read_text())

    def test_loopback_hostnames_are_not_prefix_matched(self):
        for address in ("127.0.0.1:8080", "localhost:8080", "[::1]:8080"):
            self.assertTrue(wizard.loopback(address))
        for address in ("0.0.0.0:8080", "localhost.evil:8080", "192.168.1.1:8080"):
            self.assertFalse(wizard.loopback(address))

    def test_updates_check_only_never_applies_and_source_has_no_update_prompt(self):
        with mock.patch("subprocess.check_output", return_value="infr 0.10.0\n"), mock.patch.object(release, "latest_release", return_value={"version": "0.11.0", "url": "https://example.invalid", "archive": "asset"}), mock.patch.object(release, "download_update") as update, mock.patch.object(wizard, "yes") as confirm:
            self.assertFalse(wizard.check_updates(Path("/nonexistent"), Path("/nonexistent/infr"), True, only=True))
            confirm.assert_not_called()
            update.assert_not_called()

    def test_package_updates_require_explicit_option_and_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "install-manifest.json").write_text("{}")
            with mock.patch("subprocess.check_output", return_value="infr 0.10.0\n"), mock.patch.object(release, "latest_release", return_value={"version": "0.11.0", "url": "https://example.invalid", "archive": "asset"}), mock.patch.object(release, "download_update") as update, mock.patch.object(wizard, "yes", return_value=False) as confirm:
                self.assertFalse(wizard.check_updates(root, root / "infr", True))
                confirm.assert_not_called()
                self.assertFalse(wizard.check_updates(root, root / "infr", False, only=False))
                confirm.assert_not_called()
                self.assertFalse(wizard.check_updates(root, root / "infr", True, only=False))
                confirm.assert_called_once_with("Update this Linux package now?", False)
                update.assert_not_called()
                confirm.return_value = True
                self.assertTrue(wizard.check_updates(root, root / "infr", True, only=False))
                update.assert_called_once()


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
        os.rename(installed / "scripts", outside)
        (installed / "scripts").symlink_to(outside, target_is_directory=True)
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
