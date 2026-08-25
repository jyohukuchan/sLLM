#!/usr/bin/env python3
"""Run the offline Rust host checks when a Rust workspace exists."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV_RUST_VERSION = "1.97.1"
MSRV_RUST_VERSION = "1.85.0"
MSRV_TARGET = "x86_64-unknown-linux-gnu"
RUSTUP_AUTO_INSTALL = "0"
B0_DISABLED_HIP_FLAGS = frozenset({
    "SLLM_ENABLE_HIP_COMPILE_PROBE",
    "SLLM_ENABLE_HIP_RUNTIME",
    "SLLM_ENABLE_PUBLIC_HIP_RUNTIME",
})
B0_ABSENT_ENVIRONMENT_VARIABLES = frozenset({
    "CARGO_BUILD_TARGET",
    "CARGO_BUILD_RUSTC",
    "CARGO_BUILD_RUSTC_WRAPPER",
    "CARGO_BUILD_RUSTFLAGS",
    "CARGO_BUILD_RUSTDOCFLAGS",
    "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS",
    "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTDOCFLAGS",
    "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER",
    "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER",
    "ROCM_PATH",
    "HIP_PATH",
    "CMAKE_HIP_ARCHITECTURES",
    "SLLM_HIP_CODEGEN_FEATURES",
    "SLLM_SEMANTIC_G1_AUTHORITY",
    "SLLM_HIP_COMPILER",
    "SLLM_HIP_COMPILER_LOGICAL",
    "SLLM_HIP_COMPILER_BROKER_SOCKET",
    "SLLM_HIP_COMPILER_BROKER_SESSION",
    "SLLM_HIP_COMPILER_BROKER_CLIENT",
    "SLLM_HIP_COMPILER_BROKER_CLIENT_SHA256",
    "SLLM_HIP_COMPILER_BROKER_CLIENT_FD",
    "SLLM_HIP_COMPILER_BROKER_TOKEN",
    "SLLM_SEMANTIC_G1_NATIVE_HIP_BUILD_DIR",
    "CXX",
    "CMAKE_HIP_COMPILER",
    "CMAKE_C_COMPILER",
    "CMAKE_CXX_COMPILER",
    "CMAKE_TOOLCHAIN_FILE",
    "CMAKE_PREFIX_PATH",
    "CMAKE_GENERATOR",
    "CMAKE_GENERATOR_PLATFORM",
    "CMAKE_GENERATOR_TOOLSET",
    "CMAKE_MAKE_PROGRAM",
    "CC",
    "HIPCC",
    "HIPCXX",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "OBJC_INCLUDE_PATH",
    "GCC_EXEC_PREFIX",
    "COMPILER_PATH",
    "LIBRARY_PATH",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "RUSTC",
    "RUSTC_BOOTSTRAP",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "RUSTFLAGS",
    "RUSTDOCFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_ENCODED_RUSTDOCFLAGS",
})
B0_SANITIZED_ENVIRONMENT_VARIABLES = B0_DISABLED_HIP_FLAGS | B0_ABSENT_ENVIRONMENT_VARIABLES


def b0_cargo_environment() -> dict[str, str]:
    """Return the sanitized offline environment shared by B0 Rust checks."""

    environment = os.environ.copy()
    for name in B0_ABSENT_ENVIRONMENT_VARIABLES:
        environment.pop(name, None)
    for name in B0_DISABLED_HIP_FLAGS:
        environment[name] = "0"
    environment["CARGO_NET_OFFLINE"] = "true"
    environment["RUSTUP_AUTO_INSTALL"] = RUSTUP_AUTO_INSTALL
    return environment


def msrv_check_command() -> list[str]:
    """Return the exact B0 MSRV cargo check command."""

    return [
        "cargo",
        f"+{MSRV_RUST_VERSION}",
        "check",
        "--jobs",
        "1",
        "--workspace",
        "--all-targets",
        "--locked",
        "--offline",
        "--target",
        MSRV_TARGET,
    ]


def command_for_mode(mode: str) -> list[str]:
    """Return the only accepted toolchain/subcommand pairing for each gate."""

    if mode == "format":
        return [
            "cargo",
            f"+{DEV_RUST_VERSION}",
            "fmt",
            "--all",
            "--",
            "--check",
        ]
    if mode == "clippy":
        return [
            "cargo",
            f"+{DEV_RUST_VERSION}",
            "clippy",
            "--jobs",
            "1",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--locked",
            "--offline",
            "--",
            "-D",
            "warnings",
        ]
    if mode == "msrv":
        return msrv_check_command()
    raise ValueError(f"unknown Rust validation mode: {mode}")


def validate_command_registration(mode: str, command: list[str]) -> None:
    """Keep the executable selection closed and the MSRV exception narrow."""

    expected = command_for_mode(mode)
    if command != expected:
        raise ValueError(
            f"{mode} must use the registered command {expected!r}, got {command!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("format", "clippy", "msrv"), required=True)
    args = parser.parse_args()
    if not (ROOT / "Cargo.toml").exists():
        print(f"{args.mode}: no Cargo.toml; Rust workspace not yet present")
        return 0
    command = command_for_mode(args.mode)
    validate_command_registration(args.mode, command)
    environment = b0_cargo_environment() if args.mode == "msrv" else os.environ.copy()
    if args.mode != "msrv":
        environment["RUSTUP_AUTO_INSTALL"] = RUSTUP_AUTO_INSTALL
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            timeout=300,
            env=environment,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"{args.mode}: {exc}", file=sys.stderr)
        return 1
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
