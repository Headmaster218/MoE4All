#!/usr/bin/env python3
"""Developer/CI build tool. Release users need neither Python nor system codecs."""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

PINS = {
    "libde265": ("https://github.com/strukturag/libde265", "ba62bf4cfb3242f3bf0a45617ff09e35236e4d82"),
    "dav1d": ("https://github.com/videolan/dav1d", "54706fc6bc0cdecab7e9593974a4039cc038fca7"),
    "libheif": ("https://github.com/strukturag/libheif", "f81f28ac014b1c28dee483ac45b72c6b05bc421a"),
}


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, stderr=subprocess.STDOUT)


def build(work, output, avif, jobs, source_root=None):
    windows = os.name == "nt"
    components = ["libde265"] + (["dav1d"] if avif else []) + ["libheif"]
    sources = {}
    for name in components:
        url, revision = PINS[name]
        source = (source_root or work / "sources") / name
        if not source.exists():
            source.mkdir(parents=True)
            run("git", "-C", source, "init")
            run("git", "-C", source, "fetch", "--depth=1", url, revision)
            run("git", "-C", source, "checkout", "--detach", "FETCH_HEAD")
        if (source / ".git").exists():
            actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            if actual != revision:
                raise ValueError("Unexpected source revision for " + name)
            if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
                raise ValueError("Modified codec source: " + str(source))
        else:
            # A corresponding-source archive has no Git metadata. It may be modified/rebuilt.
            marker = source.parent / "source-revisions.json"
            if not source_root or not marker.is_file() or json.loads(marker.read_text())[name] != revision:
                raise ValueError("Unrecognised source archive: " + str(source))
        sources[name] = source
    prefixes = {name: work / ("install-" + name) for name in components}
    common = ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
              "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON", "-DCMAKE_INSTALL_LIBDIR=lib"]
    if windows:
        common += ["-DCMAKE_C_COMPILER=cl.exe", "-DCMAKE_CXX_COMPILER=cl.exe",
                   "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW", "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"]
    else:
        common += ["-DCMAKE_INSTALL_RPATH=$ORIGIN", "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
                   "-DCMAKE_C_FLAGS=-march=x86-64", "-DCMAKE_CXX_FLAGS=-march=x86-64"]
    for name in components:
        directory = work / ("build-" + name)
        prefix = prefixes[name]
        if name == "dav1d":
            meson = [os.sys.executable, "-m", "mesonbuild.mesonmain"]
            args = meson + ["setup", directory, sources[name], "--buildtype=release", "--default-library=shared",
                    "--prefix=" + str(prefix), "--libdir=lib", "-Db_lto=true", "-Denable_tests=false",
                    "-Denable_tools=false", "-Denable_examples=false", "-Denable_docs=false"]
            if windows:
                args += ["-Db_vscrt=mt"]
            if not (directory / "build.ninja").exists():
                run(*args)
            run(*meson, "compile", "-C", directory, "-j", jobs)
            run(*meson, "install", "-C", directory)
            continue
        if name == "libde265":
            extra = ["-DENABLE_DECODER=OFF", "-DENABLE_ENCODER=OFF", "-DENABLE_SDL=OFF", "-DENABLE_AVX512=OFF"]
        else:
            extra = ["-DENABLE_PLUGIN_LOADING=OFF", "-DWITH_LIBDE265=ON", "-DWITH_LIBDE265_PLUGIN=OFF",
                     "-DWITH_X265=OFF", "-DWITH_X264=OFF", "-DWITH_OpenH264_DECODER=OFF",
                     "-DWITH_AOM_DECODER=OFF", "-DWITH_AOM_ENCODER=OFF", "-DWITH_DAV1D_PLUGIN=OFF",
                     "-DWITH_LIBSHARPYUV=OFF", "-DWITH_EXAMPLES=OFF", "-DWITH_GDK_PIXBUF=OFF",
                     "-DBUILD_TESTING=OFF", "-DBUILD_DOCUMENTATION=OFF", "-DWITH_HEADER_COMPRESSION=OFF",
                     "-DWITH_UNCOMPRESSED_CODEC=OFF", "-DWITH_DAV1D=" + ("ON" if avif else "OFF"),
                     "-DCMAKE_PREFIX_PATH=" + ";".join(str(prefixes[n]) for n in components if n != "libheif")]
        run("cmake", "-S", sources[name], "-B", directory, "-DCMAKE_INSTALL_PREFIX=" + str(prefix), *common, *extra)
        run("cmake", "--build", directory, "--parallel", jobs)
        run("cmake", "--install", directory)
    output.mkdir(parents=True, exist_ok=True)
    # Copy dereferenced SONAMEs: no symlinks in distributable archives or updater payloads.
    names = {"libde265": "libde265.dll" if windows else "libde265.so.0",
             "dav1d": "dav1d.dll" if windows else "libdav1d.so.7",
             "libheif": "heif.dll" if windows else "libheif.so.1"}
    for name in components:
        shutil.copyfile(prefixes[name] / ("bin" if windows else "lib") / names[name], output / names[name])
        license_path = sources[name] / "COPYING"
        if not license_path.exists():
            license_path = sources[name] / "COPYING.txt"
        shutil.copyfile(license_path, output / (name + "-LICENSE.txt"))
    instructions = ["MoE4All image codecs: decoder-only shared libraries; independently replaceable.",
                    "libheif/libde265: LGPL; dav1d: BSD. See the accompanying license files.",
                    "Corresponding source: image-codec-sources.tar.gz, distributed beside release archives.",
                    "Build: Python 3, Git, CMake, Ninja; AVIF also Meson and NASM; MSVC on Windows, GCC on Linux.",
                    "Extract the source archive, then: python3 build-image-codecs.py --source-root . --work ./build --output ./image-codecs",
                    "Add --heic-only to omit AVIF. Release users need no build tools."]
    instructions += [name + " " + PINS[name][0] + "/tree/" + PINS[name][1] for name in components]
    (output / "SOURCES.txt").write_text("\n".join(instructions) + "\n", encoding="utf-8")
    with tarfile.open(output.parent / "image-codec-sources.tar.gz", "w:gz") as archive:
        for name in components:
            archive.add(sources[name], arcname=name, filter=lambda info: None if ".git" in Path(info.name).parts else info)
        archive.add(__file__, arcname="build-image-codecs.py")
        data = json.dumps({name: PINS[name][1] for name in components}, indent=2).encode()
        marker = tarfile.TarInfo("source-revisions.json")
        marker.size = len(data)
        archive.addfile(marker, io.BytesIO(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("target/image-codec-build"))
    parser.add_argument("--output", type=Path, default=Path("target/image-codecs"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--heic-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    build(args.work.resolve(), args.output.resolve(), not args.heic_only, args.jobs,
          args.source_root.resolve() if args.source_root else None)
