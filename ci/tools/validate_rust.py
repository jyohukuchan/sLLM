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
RUSTUP_AUTO_INSTALL = "0"


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
        return [
            "cargo",
            f"+{MSRV_RUST_VERSION}",
            "check",
            "--workspace",
            "--locked",
            "--offline",
        ]
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
    environment = os.environ.copy()
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
