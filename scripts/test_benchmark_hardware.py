#!/usr/bin/env python3
"""Fixture-only collector and package tests. Runtime collectors need no Python.

Run on Linux (including WSL) with an explicit --fixture-root. If MOE4ALL_WORK_ROOT
is set, fixtures must be inside its tmp/moe4all directory. Darwin tests mock
native metadata and are not Mac hardware tests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tarfile
import tempfile
import time
import unittest

import linux_release
import macos_release


SCRIPT = Path(__file__).with_name("benchmark-hardware.sh")
SOURCE = SCRIPT.read_text(encoding="ascii")
LIBRARY = SOURCE.rsplit('\nmain "$@"', 1)[0]
FIXTURE_ROOT = None


class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="collector-", dir=FIXTURE_ROOT)
        self.root = Path(self.temp.name)
        self.model = self.root / 'sensitive-model [spaces] "quoted".bin'
        self.model.write_bytes(b"fixture\0" * 131072)
        self.proc = self.root / "proc"
        self.proc.mkdir()
        (self.proc / "cpuinfo").write_text('model name : Fixture "CPU"\\rev\nprocessor : 0\nprocessor : 1\n')
        (self.proc / "meminfo").write_text("MemTotal: 32768 kB\n")
        self.sys = self.root / "sys"
        self.pci = self.sys / "bus/pci/devices"
        self.pci.mkdir(parents=True)
        self.ids = self.root / "pci.ids"
        self.ids.write_text('1234  Fixture Vendor\n\t5678  Fixture GPU\n\t5679  Fixture GPU\n')
        topology = self.sys / "devices/pci0000:00"
        topology.mkdir(parents=True)
        self.physical_gpu = {}
        for bdf, bridge_bdf, device in (("0000:01:00.0", "0000:00:01.0", "0x5678"), ("0000:02:00.0", "0000:00:02.0", "0x5679")):
            endpoint_fields = {"class": "0x030000", "vendor": "0x1234", "device": device, "current_link_speed": "16.0 GT/s PCIe", "current_link_width": "8", "max_link_speed": "32.0 GT/s PCIe", "max_link_width": "16"}
            bus_directory = self.pci / bdf
            bus_directory.mkdir()
            for name, value in endpoint_fields.items():
                (bus_directory / name).write_text(value + "\n")
            bridge = topology / bridge_bdf
            bridge.mkdir()
            for name, value in {"class": "0x060400", "vendor": "0x1234", "device": "0x9000", "current_link_speed": "8.0 GT/s PCIe", "current_link_width": "4", "max_link_speed": "16.0 GT/s PCIe", "max_link_width": "4"}.items():
                (bridge / name).write_text(value + "\n")
            directory = bridge / bdf
            directory.mkdir()
            for name, value in endpoint_fields.items():
                (directory / name).write_text(value + "\n")
            self.physical_gpu[bdf] = directory
        raw = self.sys / "firmware/dmi/entries/17-0/raw"
        raw.parent.mkdir(parents=True)
        memory = bytearray(40)
        memory[0:2] = bytes([17, 40])
        struct.pack_into("<H", memory, 12, 16384)
        memory[18] = 34
        struct.pack_into("<H", memory, 21, 6000)
        memory[23:27] = bytes([1, 2, 0, 3])
        struct.pack_into("<H", memory, 32, 5600)
        raw.write_bytes(memory + b'Fixture Manufacturer\0SECRET_SERIAL\0Fixture Part\0\0')
        stat = self.model.stat()
        disk = self.sys / f"dev/block/{os.major(stat.st_dev)}:{os.minor(stat.st_dev)}"
        (disk / "device").mkdir(parents=True)
        (disk / "queue").mkdir()
        (disk / "device/model").write_text("Fixture SSD\n")
        (disk / "size").write_text("2048000\n")
        (disk / "queue/rotational").write_text("0\n")

    def tearDown(self):
        self.temp.cleanup()

    def run_script(self, source, args=()):
        before = time.monotonic()
        result = subprocess.run(["/bin/sh", "-s", "--", *args], input=source.encode("ascii"), capture_output=True, timeout=12)
        self.assertLess(time.monotonic() - before, 12)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertEqual(result.stderr, b"")
        self.assertLess(len(result.stdout), 65536)
        self.assertEqual(result.stdout.count(b"\n"), 1)
        value = json.loads(result.stdout)
        self.assertEqual(set(value["hardware"]), {"cpu", "ram", "gpu", "model_storage"})
        for forbidden in (b"SECRET_SERIAL", b"sensitive-model", b"private-user", b"machine-private"):
            self.assertNotIn(forbidden, result.stdout)
        return value

    def collect_fixture(self, gpu_name="Fixture GPU", bdf="", model=None, extra="", vendor="", device=""):
        bindings = {"os": "Linux", "backend": "vulkan", "proc_root": str(self.proc), "sys_root": str(self.sys), "pci_ids": str(self.ids), "model_path": str(self.model if model is None else model), "gpu_name": gpu_name, "pci_address": bdf, "gpu_vendor_id": vendor, "gpu_device_id": device, "vulkan_index": "0"}
        setup = "\n".join(f"{key}={shlex.quote(value)}" for key, value in bindings.items())
        return self.run_script(LIBRARY + '\ndiagnostics=; diagnostic_separator=\n' + setup + '\n' + extra + '\nlinux_memory\nlinux_gpu\nmodel_storage\nemit_result\n')

    def test_posix_stdin_and_missing_path_retains_cpu_memory(self):
        value = self.run_script(SOURCE, ["--model", str(self.root / "missing fixture.bin")])
        self.assertIsNotNone(value["hardware"]["cpu"])
        self.assertGreater(value["hardware"]["ram"]["total_bytes"], 0)
        for mode in ("sequential", "random_64k"):
            self.assertEqual(value["collection"]["probes"][f"model_storage.{mode}"]["status"], "failed")

    def test_bdf_precedes_name_and_reads_are_bounded(self):
        value = self.collect_fixture("wrong name ignored", "02:00.0")
        gpu = value["hardware"]["gpu"]
        self.assertEqual(gpu["pci_address"], "0000:02:00.0")
        self.assertEqual((gpu["pcie_generation"], gpu["pcie_width"]), (4, 8))
        self.assertEqual(gpu["pcie_theoretical_one_way_gb_per_second"], 15.752)
        self.assertEqual((gpu["pcie_endpoint_current_generation"], gpu["pcie_endpoint_current_width"]), (4, 8))
        self.assertEqual((gpu["pcie_endpoint_max_generation"], gpu["pcie_endpoint_max_width"]), (5, 16))
        self.assertEqual(gpu["pcie_max_scope"], "selected_gpu_endpoint_capability_not_slot")
        self.assertEqual((gpu["pcie_end_to_end_current_generation"], gpu["pcie_end_to_end_current_width"]), (3, 4))
        self.assertEqual(gpu["pcie_end_to_end_current_theoretical_one_way_gb_per_second"], 3.940)
        self.assertIsNone(gpu["pcie_end_to_end_unknown_reason"])
        self.assertEqual(len(gpu["pcie_path"]), 2)
        ram = value["hardware"]["ram"]
        self.assertEqual(ram["memory_generation"], "DDR5")
        self.assertEqual(ram["modules"][0]["capacity_bytes"], 17179869184)
        self.assertEqual(ram["modules"][0]["configured_speed_mt_s"], 5600)
        storage = value["hardware"]["model_storage"]
        self.assertEqual(storage["model"], "Fixture SSD")
        for mode in ("sequential", "random_64k"):
            read = storage["read_probe"][mode]
            self.assertGreater(read["bytes"], 0)
            self.assertLessEqual(read["bytes"], 268435456)
            self.assertGreaterEqual(read["seconds"], 0.01)
            self.assertLess(read["seconds"], 2)

    def test_duplicate_names_and_invalid_bdf_never_pick_first(self):
        for bdf, reason in (("", "ambiguous_selected_gpu"), ("bad-bdf", "invalid_pci_address"), ("03:00.0", "selected_gpu_not_found")):
            value = self.collect_fixture(bdf=bdf, model="")
            self.assertIsNone(value["hardware"]["gpu"])
            self.assertEqual(value["collection"]["probes"]["gpu.selection"]["reason"], reason)

    def test_pci_identity_disambiguates_duplicate_marketing_names(self):
        value = self.collect_fixture(vendor="1234", device="5679", model="")
        self.assertEqual(value["hardware"]["gpu"]["pci_address"], "0000:02:00.0")
        self.assertEqual(value["collection"]["probes"]["gpu.selection"]["status"], "ok")

    def test_exact_name_and_missing_link_property(self):
        value = self.collect_fixture("fixture gpu", model="")
        self.assertIsNone(value["hardware"]["gpu"])
        (self.pci / "0000:02:00.0/class").write_text("0x020000\n")
        (self.physical_gpu["0000:01:00.0"] / "current_link_speed").unlink()
        value = self.collect_fixture(model="")
        self.assertEqual(value["hardware"]["gpu"]["name"], "Fixture GPU")
        self.assertIsNone(value["hardware"]["gpu"]["pcie_generation"])
        self.assertEqual(value["collection"]["probes"]["gpu.pcie"]["status"], "unsupported")

    def test_unreadable_metadata_does_not_skip_reads(self):
        value = self.collect_fixture(extra='sys_root=/nonexistent-collector-sys')
        self.assertEqual(value["collection"]["probes"]["model_storage.metadata"]["status"], "unsupported")
        self.assertGreater(value["hardware"]["model_storage"]["read_probe"]["sequential"]["bytes"], 0)

    def test_missing_file_preserves_disk_metadata(self):
        # A real fixture path with a mocked dd failure keeps stat/disk metadata available.
        value = self.collect_fixture(extra='dd() { return 1; }\nmeasure_read() { read_json=null; probe "model_storage.$1" failed native_read_error native_dd; }')
        self.assertEqual(value["hardware"]["model_storage"]["model"], "Fixture SSD")
        self.assertEqual(value["collection"]["probes"]["model_storage.sequential"]["status"], "failed")

    def test_small_file_and_arguments(self):
        self.model.write_bytes(b"small")
        value = self.collect_fixture()
        self.assertEqual(value["collection"]["probes"]["model_storage.sequential"]["status"], "not_applicable")
        for args in (["--model"], ["--unknown"], ["--pci-address"], ["--gpu-vendor-id", "not-hex"]):
            value = self.run_script(SOURCE, args)
            self.assertEqual(value["collection"]["probes"]["arguments"]["status"], "failed")

    def test_mac_unified_memory_mock_only(self):
        # These shell functions emulate macOS native metadata on Linux. No Mac claim.
        mocks = '''
sysctl() { case $2 in machdep.cpu.brand_string) printf 'Apple Fixture\\n' ;; hw.logicalcpu) printf '8\\n' ;; hw.memsize) printf '17179869184\\n' ;; esac; }
uname() { case $1 in -s) printf 'Darwin\\n' ;; -m) printf 'arm64\\n' ;; esac; }
ioreg() { printf '  "model" = <"Apple Fixture GPU">\\n  "serial-number" = "SECRET_SERIAL"\\n'; }
'''
        value = self.run_script(LIBRARY + mocks + '\nmain --backend Metal\n')
        self.assertTrue(value["hardware"]["ram"]["unified_memory"])
        self.assertTrue(value["hardware"]["gpu"]["unified_memory"])
        self.assertNotIn("vram_total_bytes", value["hardware"]["gpu"])
        self.assertEqual(value["collection"]["probes"]["gpu.pcie"]["status"], "not_applicable")

    def test_unix_packages_include_exact_collector_and_manifest(self):
        for module in (linux_release, macos_release):
            root = self.root / module.PLATFORM
            root.mkdir()
            for name in module.ROOT_FILES | module.SCRIPT_FILES:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(SCRIPT.read_bytes() if name.endswith("benchmark-hardware.sh") else b"fixture\n")
            archive = module.package(root, root / "infr", root / "output", "1.0.0-test")
            with tarfile.open(archive) as bundle:
                prefix = module.package_name("1.0.0-test")
                members = {member.name.rstrip("/") for member in bundle.getmembers()}
                self.assertIn(prefix + "/scripts", members, module.PLATFORM)
                collector = bundle.extractfile(prefix + "/scripts/benchmark-hardware.sh").read()
                self.assertEqual(collector, SCRIPT.read_bytes())
                manifest = json.load(bundle.extractfile(prefix + "/install-manifest.json"))
                entry = next(entry for entry in manifest["files"] if entry["path"] == "scripts/benchmark-hardware.sh")
                self.assertEqual(entry["sha256"], hashlib.sha256(collector).hexdigest())
                stage_root = root / "stage"
                bundle.extractall(stage_root)
            stage = stage_root / prefix
            manifest_path = stage / "install-manifest.json"
            manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != "scripts/benchmark-hardware.sh"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (stage / "scripts/benchmark-hardware.sh").unlink()
            with self.assertRaisesRegex(ValueError, "missing runtime files", msg=module.PLATFORM):
                module.validate_package(stage, "1.0.0-test")
            self.assertTrue(module.managed("scripts/benchmark-hardware.sh"))
            self.assertFalse(module.managed("scripts/not-allowed.sh"))

    def test_launcher_allowlists_accept_collector_and_parent_directory(self):
        for name in ("Start-INFR-Wizard-Linux.sh", "Start-INFR-Wizard-macOS.sh"):
            source = (SCRIPT.parent.parent / name).read_text()
            managed = source.split("\nmanaged() {", 1)[1].split("\n}\n", 1)[0]
            # Execute only managed(), with no launcher startup/update side effects.
            # Both paths return from the explicit case before macOS-only zsh syntax.
            result = subprocess.run(["/bin/bash", "-s"], input=("managed() {" + managed + '\n}\nmanaged scripts/benchmark-hardware.sh || exit 1\nif managed scripts/not-allowed.sh; then exit 1; fi\n').encode(), capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, b"")
            extract = source.split("\nextract_package() {", 1)[1].split("\n}\n", 1)[0]
            self.assertIn("$relative == scripts/", extract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, required=True)
    options, remaining = parser.parse_known_args()
    FIXTURE_ROOT = options.fixture_root.resolve()
    if os.environ.get("MOE4ALL_WORK_ROOT"):
        allowed = (Path(os.environ["MOE4ALL_WORK_ROOT"]) / "tmp/moe4all").resolve()
        if allowed != FIXTURE_ROOT and allowed not in FIXTURE_ROOT.parents:
            raise SystemExit("Fixture root must be inside MOE4ALL_WORK_ROOT/tmp/moe4all")
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(FIXTURE_ROOT)
    unittest.main(argv=[__file__, *remaining])
