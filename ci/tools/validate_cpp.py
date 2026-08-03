#!/usr/bin/env python3
"""Check C++ formatting and the optional native host build without GPU use."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {".git", ".local-artifacts", "reference", "target", "build", "ci"}
EXPLICIT_CPP_SOURCES = {"ci/tools/g0_native_observer.cpp"}


def cpp_files() -> list[Path]:
    # H0 validates the checkout/index.  A clean CI checkout has all project
    # sources tracked; local untracked worker files are handled by hygiene and
    # are not silently treated as a candidate tree.
    listed = subprocess.run(
        ["git", "ls-files", "--", "*.c", "*.cc", "*.cpp", "*.cxx", "*.h", "*.hh", "*.hpp", "*.hxx"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    names = set(listed.stdout.splitlines())
    for source_root in (ROOT / "native", ROOT / "include"):
        if source_root.exists():
            for pattern in ("*.c", "*.cc", "*.cpp", "*.cxx", "*.h", "*.hh", "*.hpp", "*.hxx"):
                names.update(path.relative_to(ROOT).as_posix() for path in source_root.rglob(pattern))
    files: list[Path] = []
    for name in names:
        path = (ROOT / name).resolve()
        relative = path.relative_to(ROOT)
        if path.is_file() and (
            relative.as_posix() in EXPLICIT_CPP_SOURCES
            or not any(part in EXCLUDED for part in relative.parts)
        ):
            files.append(path)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("format", "static"), required=True)
    args = parser.parse_args()
    files = cpp_files()
    if not files:
        print(f"{args.mode}: no C/C++ files were collected", file=sys.stderr)
        return 1
    if args.mode == "format":
        try:
            result = subprocess.run(
                ["clang-format", "--dry-run", "--Werror", *[str(path) for path in files]],
                cwd=ROOT,
                check=False,
                timeout=300,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            print(f"clang-format unavailable or timed out: {exc}", file=sys.stderr)
            return 1
    else:
        cmake = ROOT / "CMakeLists.txt"
        source_dir = ROOT
        if not cmake.exists():
            cmake = ROOT / "native/hip/CMakeLists.txt"
            source_dir = cmake.parent
        if not cmake.exists():
            print("static: C++ files exist but no CMakeLists.txt; no native host build target")
            return 0
        with tempfile.TemporaryDirectory(prefix="ullm-cmake-") as directory:
            build = Path(directory) / "build"
            configure = subprocess.run(
                ["cmake", "-S", str(source_dir), "-B", str(build), "-DULLM_ENABLE_HIP=OFF", "-DCMAKE_BUILD_TYPE=Debug"],
                cwd=ROOT,
                check=False,
                timeout=300,
            )
            result = configure if configure.returncode else subprocess.run(
                ["cmake", "--build", str(build), "--parallel", "1"],
                cwd=ROOT,
                check=False,
                timeout=300,
            )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
