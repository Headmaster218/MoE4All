#!/usr/bin/env python3
"""Build and validate the experimental self-contained macOS arm64 package."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile


PLATFORM = "macOS-arm64"
ROOT_FILES = {
    "infr",
    "Start-INFR-Wizard-macOS.sh",
    "README.md",
    "README_EN.md",
    "CHANGELOG.md",
    "infr.example.toml",
    "LICENSE",
    "LICENSE-MIT",
    "NOTICE",
}
DOC_SUFFIXES = {".md", ".png", ".jpg", ".svg", ".webp"}


def version_key(version):
    match = re.fullmatch(r"(?:v|release-)?(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise ValueError("Expected a stable engine version: " + version)
    return tuple(map(int, match.groups()))


def package_name(version):
    version_key(version)
    return f"MoE4All-{PLATFORM}-v{version}"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def managed(name):
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or any(
        part in ("", ".", "..") for part in name.split("/")
    ):
        return False
    return name in ROOT_FILES | {"install-manifest.json"} or (
        name.startswith("documentation/") and path.suffix.lower() in DOC_SUFFIXES
    )


def validate_package(root, version):
    root = Path(root)
    manifest = json.loads((root / "install-manifest.json").read_text(encoding="utf-8"))
    identity = (
        manifest.get("schema_version"),
        manifest.get("updater_protocol"),
        manifest.get("product"),
        manifest.get("platform"),
        manifest.get("version"),
        manifest.get("tag"),
        manifest.get("stability"),
    )
    if identity != (1, 1, "moe4all-engine", PLATFORM, version, "release-" + version, "experimental"):
        raise ValueError("Incompatible package manifest")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 6000:
        raise ValueError("Invalid package file list")
    names = set()
    for entry in entries:
        name = entry.get("path")
        if not managed(name) or name == "install-manifest.json" or name in names:
            raise ValueError("Invalid or duplicate package path: " + str(name))
        names.add(name)
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Package entry is not a regular file: " + name)
        if path.stat().st_size != entry.get("size") or sha256(path) != entry.get("sha256"):
            raise ValueError("Package integrity check failed: " + name)
    if not {"infr", "Start-INFR-Wizard-macOS.sh"}.issubset(names):
        raise ValueError("Package is missing runtime files")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != names | {"install-manifest.json"}:
        raise ValueError("Unexpected unlisted package files")
    return manifest


def package(root, binary, output, version):
    root, binary, output = Path(root), Path(binary), Path(output)
    name = package_name(version)
    if not binary.is_file() or binary.is_symlink():
        raise ValueError("Release binary not found: " + str(binary))
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".macos-package-", dir=output) as temporary:
        stage = Path(temporary) / name
        stage.mkdir()
        for relative in sorted(ROOT_FILES - {"infr"}):
            source = root / relative
            if not source.is_file() or source.is_symlink():
                raise ValueError("Missing package source: " + relative)
            shutil.copyfile(source, stage / relative)
        shutil.copyfile(binary, stage / "infr")
        for source in sorted((root / "documentation").rglob("*")):
            relative = source.relative_to(root).as_posix()
            if source.is_file() and managed(relative):
                if source.is_symlink():
                    raise ValueError("Symlink in documentation: " + relative)
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        files = []
        for path in sorted(stage.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(stage).as_posix()
            os.chmod(path, 0o755 if relative in ("infr", "Start-INFR-Wizard-macOS.sh") else 0o644)
            files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256(path)})
        manifest = {
            "schema_version": 1,
            "updater_protocol": 1,
            "product": "moe4all-engine",
            "platform": PLATFORM,
            "version": version,
            "tag": "release-" + version,
            "stability": "experimental",
            "minimum_macos": "13.0",
            "architecture": "arm64",
            "files": files,
        }
        (stage / "install-manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        os.chmod(stage / "install-manifest.json", 0o644)
        validate_package(stage, version)
        archive = output / (name + ".tar.gz")
        partial = Path(temporary) / archive.name
        with tarfile.open(partial, "w:gz", format=tarfile.PAX_FORMAT) as bundle:
            bundle.add(stage, arcname=name, recursive=True)
        os.replace(partial, archive)
        checksum = archive.with_name(archive.name + ".sha256")
        checksum.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="ascii")
        return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--binary", type=Path, default=Path("target/release/infr"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    options = parser.parse_args()
    root = options.root.resolve()
    binary = options.binary if options.binary.is_absolute() else root / options.binary
    print(package(root, binary, options.output.resolve(), options.version))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        raise SystemExit(str(error))
