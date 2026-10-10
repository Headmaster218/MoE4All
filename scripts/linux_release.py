#!/usr/bin/env python3
"""Developer-only Linux package/checksum fixtures; not shipped in release packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

API = "https://api.github.com/repos/Headmaster218/MoE4All/releases?per_page=100"
PLATFORM = "Linux-x86_64"
ROOT_FILES = {
    "infr", "Start-INFR-Wizard-Linux.sh", "README.md", "README_EN.md", "CHANGELOG.md",
    "infr.example.toml", "LICENSE", "LICENSE-MIT", "NOTICE",
}
SCRIPT_FILES = set()
CODEC_FILES = {"libheif.so.1", "libde265.so.0", "libdav1d.so.7", "libheif-LICENSE.txt",
               "libde265-LICENSE.txt", "dav1d-LICENSE.txt", "SOURCES.txt"}
MAX_BYTES = 1024 * 1024 * 1024


def version_key(version):
    match = re.fullmatch(r"(?:v|release-)?(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise ValueError("Expected a stable engine version: " + version)
    return tuple(map(int, match.groups()))


def package_name(version):
    version_key(version)
    return "MoE4All-{}-v{}".format(PLATFORM, version)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path, data, mode=0o644):
    path = Path(path)
    handle, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def managed(name):
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or any(p in (".", "..") for p in name.split("/")):
        return False
    return name in ROOT_FILES | SCRIPT_FILES | {"install-manifest.json"} or name in {"image-codecs/" + n for n in CODEC_FILES} or (
        name.startswith("documentation/") and path.suffix.lower() in {".md", ".png", ".jpg", ".svg", ".webp"})


def safe_target(root, name):
    root = root.resolve()
    if not managed(name):
        raise ValueError("Not a managed package file: " + name)
    target = root / name
    for component in [target] + list(target.parents):
        if component == root:
            break
        if component.is_symlink():
            raise ValueError("Refusing a symlink in install path: " + name)
    if root not in target.resolve().parents:
        raise ValueError("Install path escapes package root")
    return target


def validate_package(root, version):
    manifest = json.loads((root / "install-manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version"), manifest.get("updater_protocol"), manifest.get("product"),
        manifest.get("platform"), manifest.get("version"), manifest.get("tag")) != (
            1, 1, "moe4all-engine", PLATFORM, version, "release-" + version):
        raise ValueError("Incompatible package manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 5000:
        raise ValueError("Invalid package file list")
    names = set()
    for entry in files:
        name = entry["path"]
        if name in names or name == "install-manifest.json":
            raise ValueError("Duplicate or self-referential manifest entry")
        names.add(name)
        path = safe_target(root, name)
        if not path.is_file() or path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError("Package integrity check failed: " + name)
    if not ({"infr", "Start-INFR-Wizard-Linux.sh"} | SCRIPT_FILES).issubset(names):
        raise ValueError("Package is missing runtime files")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != names | {"install-manifest.json"}:
        raise ValueError("Unexpected unlisted package files")
    return manifest


def package(root, binary, output, version, codecs=None):
    name = package_name(version)
    if not binary.is_file():
        raise ValueError("Release binary not found: " + str(binary))
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".linux-package-", dir=output) as temporary:
        stage = Path(temporary) / name
        stage.mkdir()
        for relative in sorted((ROOT_FILES - {"infr"}) | SCRIPT_FILES):
            source = root / relative
            if not source.is_file() or source.is_symlink():
                raise ValueError("Missing package source: " + relative)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        shutil.copyfile(binary, stage / "infr")
        if codecs is not None:
            required = CODEC_FILES - {"libdav1d.so.7", "dav1d-LICENSE.txt"}
            if (codecs / "libdav1d.so.7").exists():
                required = CODEC_FILES
            for codec_name in sorted(required):
                source = codecs / codec_name
                if not source.is_file():
                    raise ValueError("Missing codec bundle file: " + codec_name)
                target = stage / "image-codecs" / codec_name
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, target)
            shutil.copyfile(codecs.parent / "image-codec-sources.tar.gz", output / "image-codec-sources.tar.gz")
        for source in sorted((root / "documentation").rglob("*")):
            relative = source.relative_to(root).as_posix()
            if source.is_file() and managed(relative):
                if source.is_symlink():
                    raise ValueError("Symlink in documentation")
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        files = []
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                relative = path.relative_to(stage).as_posix()
                os.chmod(path, 0o755 if relative in ("infr", "Start-INFR-Wizard-Linux.sh") else 0o644)
                files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256(path)})
        manifest = {"schema_version": 1, "updater_protocol": 1, "product": "moe4all-engine",
                    "platform": PLATFORM, "version": version, "tag": "release-" + version,
                    "glibc_min": platform.libc_ver()[1] or "2.35", "stability": "experimental", "files": files}
        write_json_atomic(stage / "install-manifest.json", manifest)
        validate_package(stage, version)
        archive = output / (name + ".tar.gz")
        partial = Path(temporary) / archive.name
        with tarfile.open(partial, "w:gz") as tar:
            tar.add(stage, arcname=name)
        os.replace(partial, archive)
        checksum = archive.with_name(archive.name + ".sha256")
        checksum.write_text(sha256(archive) + "  " + archive.name + "\n", encoding="ascii")
        return archive


def extract_package(archive, destination, version):
    prefix = package_name(version)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) > 6000 or sum(m.size for m in members) > MAX_BYTES:
            raise ValueError("Oversized archive")
        seen = set()
        for member in members:
            parts = member.name.split("/")
            if parts[0] != prefix or "\\" in member.name or any(p in ("", ".", "..") for p in parts):
                raise ValueError("Unsafe archive path: " + member.name)
            if member.name in seen or not (member.isdir() or member.isfile()):
                raise ValueError("Duplicate entry, link or special file in archive")
            seen.add(member.name)
            if member.isdir():
                continue
            relative = "/".join(parts[1:])
            target = safe_target(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("xb") as out:
                shutil.copyfileobj(source, out)
            os.chmod(target, 0o755 if relative in ("infr", "Start-INFR-Wizard-Linux.sh") else 0o644)
    return validate_package(destination, version)


def engine_running(binary):
    for path in Path("/proc").glob("[0-9]*/exe"):
        try:
            if os.path.samefile(path, binary):
                return True
        except (OSError, ValueError):
            continue
    return False


def apply_update(root, stage, version):
    if (root / ".git").exists() or not (root / "install-manifest.json").is_file():
        raise ValueError("Only an extracted managed Linux package can be updated")
    old = json.loads((root / "install-manifest.json").read_text(encoding="utf-8"))
    if old.get("platform") != PLATFORM or old.get("product") != "moe4all-engine" or old.get("updater_protocol") != 1:
        raise ValueError("Unrecognised installed package")
    if version_key(version) <= version_key(old["version"]):
        raise ValueError("Refusing a downgrade or same-version replacement")
    manifest = validate_package(stage, version)
    if engine_running(root / "infr"):
        raise ValueError("Stop the running engine before updating; no files changed")
    names = [entry["path"] for entry in manifest["files"]] + ["install-manifest.json"]
    # Validate every destination before replacing anything, including parent symlinks.
    for name in names:
        target = safe_target(root, name)
        if target.exists() and not target.is_file():
            raise ValueError("Install destination is not a regular file")
    installed = []
    backup = Path(tempfile.mkdtemp(prefix=".update-backup-", dir=root))
    keep_backup = False
    try:
        for name in names:
            target = root / name
            if target.exists():
                saved = backup / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
        try:
            for name in names:
                target = safe_target(root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / name, target)
                installed.append(name)
            actual = subprocess.check_output([str(root / "infr"), "--version"], text=True, timeout=10).strip().split()[-1]
            if version_key(actual) != version_key(version):
                raise ValueError("Updated executable has the wrong version")
        except BaseException as original:
            rollback_errors = []
            for name in reversed(installed):
                try:
                    target = root / name
                    if (backup / name).exists():
                        os.replace(backup / name, target)
                    elif target.exists():
                        target.unlink()
                except OSError as error:
                    rollback_errors.append(str(error))
            if rollback_errors:
                keep_backup = True
                raise ValueError("Rollback incomplete; preserve and recover backup at " + str(backup)) from original
            raise
    finally:
        if not keep_backup:
            shutil.rmtree(backup)


def select_release(releases):
    candidates = []
    for release in releases:
        if release.get("draft") or release.get("prerelease"):
            continue
        tag = release.get("tag_name", "")
        if not tag.startswith("release-"):
            continue
        version = tag[len("release-"):]
        try:
            version_key(version)
        except ValueError:
            continue
        name = package_name(version) + ".tar.gz"
        assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
        url = "https://github.com/Headmaster218/MoE4All/releases/tag/" + tag
        archive = assets.get(name) if name + ".sha256" in assets else None
        candidates.append({"version": version, "url": url, "archive": archive,
                           "checksum": assets.get(name + ".sha256"), "name": name})
    if not candidates:
        raise ValueError("No stable engine release found")
    return max(candidates, key=lambda r: version_key(r["version"]))


def latest_release():
    request = urllib.request.Request(API, headers={"Accept": "application/vnd.github+json", "User-Agent": "MoE4All-Linux-Wizard"})
    with urllib.request.urlopen(request, timeout=5) as response:
        data = response.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024:
        raise ValueError("Release response too large")
    return select_release(json.loads(data))


def fetch(url, output, maximum):
    if not url.startswith("https://github.com/Headmaster218/MoE4All/releases/download/release-"):
        raise ValueError("Unexpected release asset URL")
    with urllib.request.urlopen(url, timeout=30) as response, output.open("xb") as stream:
        if not response.geturl().startswith("https://"):
            raise ValueError("Insecure asset redirect")
        total = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > maximum:
                raise ValueError("Asset exceeds download limit")
            stream.write(block)


def download_update(root, release):
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise ValueError("This package updater supports Linux x86_64 only")
    # Serialize updater instances without leaving a stale lock after crashes.
    import fcntl
    with (root / ".moe4all-update.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another update is already in progress") from exc
        with tempfile.TemporaryDirectory(prefix=".update-stage-", dir=root) as temporary:
            temp = Path(temporary)
            archive, checksum = temp / release["name"], temp / "checksum"
            fetch(release["checksum"], checksum, 4096)
            pieces = checksum.read_text(encoding="ascii").split()
            if len(pieces) != 2 or pieces[1].lstrip("*") != release["name"] or not re.fullmatch(r"[0-9a-fA-F]{64}", pieces[0]):
                raise ValueError("Malformed release checksum")
            fetch(release["archive"], archive, MAX_BYTES)
            if sha256(archive) != pieces[0].lower():
                raise ValueError("Release checksum mismatch; no files changed")
            stage = temp / "files"
            stage.mkdir()
            extract_package(archive, stage, release["version"])
            apply_update(root, stage, release["version"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("package")
    pack.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    pack.add_argument("--binary", type=Path, default=Path("target/release/infr"))
    pack.add_argument("--output", type=Path, default=Path("dist"))
    pack.add_argument("--version", required=True)
    pack.add_argument("--codecs", type=Path)
    options = parser.parse_args()
    root = options.root.resolve()
    binary = options.binary if options.binary.is_absolute() else root / options.binary
    output = options.output if options.output.is_absolute() else root / options.output
    print(package(root, binary, output.resolve(), options.version,
                  options.codecs.resolve() if options.codecs else None))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, tarfile.TarError) as error:
        raise SystemExit(str(error))
