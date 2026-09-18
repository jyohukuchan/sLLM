#!/usr/bin/env python3
"""Build the standalone Qwen3.8 llama.cpp logits consumer.

The llama.cpp checkout and build directory are supplied explicitly (or default
to the local reference snapshot used for the measurement).  This script only
compiles and links the consumer; it does not run a model or touch the
reference checkout.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    default_root = repo / ".local-artifacts" / "llama-mtp-20260914"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=default_root / "upstream")
    parser.add_argument("--build", type=Path, default=default_root / "build-gfx1030")
    parser.add_argument("--output", type=Path, default=repo / ".local-artifacts" / "qwen38-kld-llama-build" / "qwen38_kld_llama")
    parser.add_argument("--cxx", type=Path, default=None)
    parser.add_argument("--rocm-root", type=Path, default=Path(os.environ.get("ROCM_ROOT", "/opt/rocm/core-7.14")))
    args = parser.parse_args()

    cxx = args.cxx or Path(os.environ.get("CXX", str(args.rocm_root / "lib" / "llvm" / "bin" / "clang++")))
    source = repo / "ci" / "tools" / "qwen38_kld_llama.cpp"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(cxx),
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Wconversion",
        "-Wshadow",
        "-I",
        str(args.upstream / "include"),
        "-I",
        str(args.upstream / "ggml" / "include"),
        str(source),
        "-L",
        str(args.build / "bin"),
        # DT_RPATH is intentionally used here so the ROCm dependencies of
        # libggml-hip are found without requiring a second LD_LIBRARY_PATH
        # entry.  It does not alter the model or backend selected at runtime.
        "-Wl,--disable-new-dtags",
        "-Wl,-rpath," + str(args.build / "bin"),
        "-Wl,-rpath," + str(args.rocm_root / "lib"),
        "-Wl,-rpath,/opt/rocm/lib",
        "-lllama",
        "-lggml",
        "-lggml-base",
        "-o",
        str(args.output),
    ]
    print("+ " + shlex.join(command))
    subprocess.run(command, check=True)
    print(f"built {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
