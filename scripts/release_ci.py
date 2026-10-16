#!/usr/bin/env python3
"""Write and verify immutable CI release manifests for artifact promotion."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


SCHEMA_VERSION = 1
PLATFORMS = {
    "windows-x86_64": "MoE4All-Windows-x86_64-v{version}.zip",
    "linux-x86_64": "MoE4All-Linux-x86_64-v{version}.tar.gz",
    "macos-arm64": "MoE4All-macOS-arm64-v{version}.tar.gz",
}
VERSION_RE = re.compile(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?")
REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_version(version):
    if not VERSION_RE.fullmatch(version):
        raise ValueError("Invalid release version: " + version)
    return version


def clean_name(path):
    value = os.fspath(path)
    name = Path(value).name
    if not name or value != name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("Release files must have a safe basename: " + value)
    return value


def validate_repository(repository):
    if not REPOSITORY_RE.fullmatch(repository):
        raise ValueError("Repository must be an owner/name slug: " + repository)
    return repository


def parse_metadata(items):
    metadata = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ValueError("Metadata must be lowercase key=value: " + item)
        if key in metadata:
            raise ValueError("Duplicate metadata key: " + key)
        metadata[key] = value
    return metadata


def expected_archive(platform_name, version):
    if platform_name not in PLATFORMS:
        raise ValueError("Unsupported release platform: " + platform_name)
    return PLATFORMS[platform_name].format(version=validate_version(version))


def write_manifest(platform_name, version, commit, repository, output, files, metadata):
    version = validate_version(version)
    repository = validate_repository(repository)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Commit must be a full lowercase Git SHA")
    archive = expected_archive(platform_name, version)
    entries = []
    names = set()
    for item in files:
        path = Path(item)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Release file is missing or not regular: " + str(path))
        path = path.resolve()
        name = clean_name(path.name)
        if name in names:
            raise ValueError("Duplicate release filename: " + name)
        names.add(name)
        entries.append({"name": name, "size": path.stat().st_size, "sha256": sha256(path)})
    if archive not in names or archive + ".sha256" not in names:
        raise ValueError("Manifest must include the platform archive and checksum")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "product": "moe4all-engine",
        "platform": platform_name,
        "version": version,
        "tag": "release-" + version,
        "commit": commit,
        "repository": repository,
        "archive": archive,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": parse_metadata(metadata),
        "files": sorted(entries, key=lambda entry: entry["name"]),
    }
    write_json_atomic(output, manifest)
    return manifest


def read_checksum(path, archive):
    pieces = Path(path).read_text(encoding="ascii").split()
    if len(pieces) != 2 or pieces[1].lstrip("*") != archive:
        raise ValueError("Malformed checksum file for " + archive)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", pieces[0]):
        raise ValueError("Malformed SHA256 for " + archive)
    return pieces[0].lower()


def verify_manifest(path, commit, version, repository):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Release manifest is missing or not regular: " + str(path))
    path = path.resolve()
    root = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    identity = (
        manifest.get("schema_version"),
        manifest.get("product"),
        manifest.get("version"),
        manifest.get("tag"),
        manifest.get("commit"),
        manifest.get("repository"),
    )
    expected = (
        SCHEMA_VERSION,
        "moe4all-engine",
        validate_version(version),
        "release-" + version,
        commit,
        validate_repository(repository),
    )
    if identity != expected:
        raise ValueError("Release manifest identity mismatch: " + str(path))
    platform_name = manifest.get("platform")
    archive = expected_archive(platform_name, version)
    if manifest.get("archive") != archive:
        raise ValueError("Unexpected archive name in " + str(path))
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Release manifest has no files: " + str(path))
    names = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid release manifest file entry")
        name = entry.get("name")
        if not isinstance(name, str) or clean_name(name) != name or name in names:
            raise ValueError("Invalid or duplicate manifest filename")
        size = entry.get("size")
        digest = entry.get("sha256")
        if not isinstance(size, int) or size < 0 or not isinstance(digest, str):
            raise ValueError("Invalid release manifest file metadata: " + name)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid release manifest SHA256: " + name)
        names.add(name)
        candidate = root / name
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("Manifest file is missing: " + str(candidate))
        if candidate.stat().st_size != size or sha256(candidate) != digest:
            raise ValueError("Manifest hash mismatch: " + str(candidate))
    checksum_name = archive + ".sha256"
    if archive not in names or checksum_name not in names:
        raise ValueError("Archive or checksum missing from manifest")
    if read_checksum(root / checksum_name, archive) != sha256(root / archive):
        raise ValueError("Published checksum does not match " + archive)
    return manifest


def copy_unique(source, output):
    target = output / source.name
    if target.exists():
        if sha256(target) != sha256(source):
            raise ValueError("Conflicting promoted asset: " + source.name)
        return target
    shutil.copyfile(source, target)
    return target


def prepare_promotion(root, output, commit, version, repository):
    validate_version(version)
    validate_repository(repository)
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root in output.parents:
        raise ValueError("Promotion output must not be inside the downloaded artifact tree")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Promotion output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    found = {}
    for path in root.rglob("release-manifest-*.json"):
        manifest = verify_manifest(path, commit, version, repository)
        platform_name = manifest["platform"]
        if platform_name in found:
            raise ValueError("Duplicate platform artifact: " + platform_name)
        found[platform_name] = (path.parent, manifest)
    if set(found) != set(PLATFORMS):
        missing = sorted(set(PLATFORMS) - set(found))
        extra = sorted(set(found) - set(PLATFORMS))
        raise ValueError("Incomplete release artifacts; missing={} extra={}".format(missing, extra))
    promoted = []
    platform_manifests = []
    for platform_name in sorted(found):
        directory, manifest = found[platform_name]
        archive = manifest["archive"]
        promoted.append(copy_unique(directory / archive, output))
        promoted.append(copy_unique(directory / (archive + ".sha256"), output))
        platform_manifests.append(manifest)
    windows_root, windows_manifest = found["windows-x86_64"]
    source_name = "image-codec-sources.tar.gz"
    if source_name not in {entry["name"] for entry in windows_manifest["files"]}:
        raise ValueError("Windows artifact is missing corresponding codec sources")
    promoted.append(copy_unique(windows_root / source_name, output))
    sums = output / "SHA256SUMS"
    with sums.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(
            "".join(
                "{}  {}\n".format(sha256(path), path.name)
                for path in sorted(promoted, key=lambda path: path.name)
            )
        )
    combined = {
        "schema_version": SCHEMA_VERSION,
        "product": "moe4all-engine",
        "version": version,
        "tag": "release-" + version,
        "commit": commit,
        "repository": repository,
        "platforms": platform_manifests,
        "published_files": [
            {"name": path.name, "size": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(promoted + [sums], key=lambda path: path.name)
        ],
    }
    combined_path = output / "release-build-manifest.json"
    write_json_atomic(combined_path, combined)
    return sorted(promoted + [sums, combined_path])


def workspace_version(cargo_toml):
    try:
        import tomllib
    except ImportError as error:
        raise ValueError("Python 3.11 or newer is required") from error
    with Path(cargo_toml).open("rb") as stream:
        data = tomllib.load(stream)
    return validate_version(data["workspace"]["package"]["version"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    version_parser = commands.add_parser("version")
    version_parser.add_argument("--cargo-toml", type=Path, default=Path("Cargo.toml"))
    write_parser = commands.add_parser("write")
    write_parser.add_argument("--platform", choices=sorted(PLATFORMS), required=True)
    write_parser.add_argument("--version", required=True)
    write_parser.add_argument("--commit", required=True)
    write_parser.add_argument("--repository", required=True)
    write_parser.add_argument("--output", type=Path, required=True)
    write_parser.add_argument("--metadata", action="append", default=[])
    write_parser.add_argument("files", nargs="+")
    promote_parser = commands.add_parser("prepare")
    promote_parser.add_argument("--root", type=Path, required=True)
    promote_parser.add_argument("--output", type=Path, required=True)
    promote_parser.add_argument("--version", required=True)
    promote_parser.add_argument("--commit", required=True)
    promote_parser.add_argument("--repository", required=True)
    options = parser.parse_args()
    if options.command == "version":
        print(workspace_version(options.cargo_toml))
    elif options.command == "write":
        write_manifest(options.platform, options.version, options.commit, options.repository,
                       options.output, options.files, options.metadata)
        print(options.output)
    else:
        for path in prepare_promotion(
                options.root, options.output, options.commit, options.version, options.repository):
            print(path)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error))
