#!/usr/bin/env python3
"""Compile and inspect one Phase 7 exact-target compatibility row."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import ContractError, ROOT, canonical_bytes, read_json  # type: ignore[no-redef]
else:
    from .common import ContractError, ROOT, canonical_bytes, read_json

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Phase 7 compatibility compile dependency missing: {exc}") from exc


SCHEMA_PATH = ROOT / "ci/schema/phase7-compatibility-compile-report-v1.schema.json"
SOURCE_PATH = ROOT / "native/hip/src/hip_compile_probe.hip.cpp"
TARGETS = (
    "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035",
    "gfx1036", "gfx1200", "gfx1201", "gfx942",
)
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
COMMAND_TIMEOUT_SECONDS = 300


def _fail(message: str) -> None:
    raise ContractError(message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, arguments: list[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments], capture_output=True, text=True,
        check=False, timeout=20,
    )
    if result.returncode != 0:
        _fail(f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def _command(identifier: str, argv: list[str], *, cwd: Path) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            argv, cwd=cwd, capture_output=True, check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(f"{identifier} failed: {exc}")
    if len(result.stdout) + len(result.stderr) > MAX_OUTPUT_BYTES:
        _fail(f"{identifier} output exceeded the limit")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", "replace")[-2000:]
        _fail(f"{identifier} exited {result.returncode}: {detail}")
    return {
        "id": identifier,
        "argv_sha256": hashlib.sha256(canonical_bytes(argv)).hexdigest(),
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "duration": time.monotonic() - started,
    }


def _validate_report(report: dict[str, Any]) -> None:
    schema = read_json(SCHEMA_PATH)
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(report),
        key=lambda error: list(error.path),
    )
    if errors:
        _fail("Phase 7 compatibility report schema failed: " + "; ".join(error.message for error in errors[:5]))
    target = report["target"]
    expected_device = f"hipv4-amdgcn-amd-amdhsa--{target}"
    if report["artifact"]["bundle_ids"] != [expected_device, "host-x86_64-unknown-linux-gnu-"]:
        _fail("Phase 7 compatibility bundle order or exact target drifted")
    if report["artifact"]["metadata_target"] != target:
        _fail("Phase 7 compatibility metadata target differs from its row")


def run(
    *, target: str, output_dir: Path, rocm_root: Path, repo: Path,
    strict_ci: bool, expected_sha: str | None, allow_dirty_local: bool,
) -> dict[str, Any]:
    if target not in TARGETS:
        _fail(f"unknown Phase 7 compatibility target: {target}")
    repo = repo.resolve()
    commit = _git(repo, ["rev-parse", "HEAD"])
    tree = _git(repo, ["rev-parse", "HEAD^{tree}"])
    dirty = bool(_git(repo, ["status", "--porcelain=v1", "--untracked-files=all"]))
    if strict_ci and (dirty or expected_sha != commit):
        _fail("strict Phase 7 compatibility compile requires a clean exact candidate")
    if dirty and not strict_ci and not allow_dirty_local:
        _fail("dirty local Phase 7 compatibility compile requires --allow-dirty-local")
    if output_dir.exists() or output_dir.is_symlink():
        _fail(f"refusing to overwrite Phase 7 compatibility output: {output_dir}")

    compiler = rocm_root / "bin/amdclang++"
    objcopy = rocm_root / "lib/llvm/bin/llvm-objcopy"
    bundler = rocm_root / "lib/llvm/bin/clang-offload-bundler"
    readobj = rocm_root / "lib/llvm/bin/llvm-readobj"
    runtime = rocm_root / "lib/libamdhip64.so"
    for path in (compiler, objcopy, bundler, readobj, runtime, SOURCE_PATH):
        if path.is_symlink() and path == SOURCE_PATH or not path.exists():
            _fail(f"Phase 7 compatibility input is missing: {path}")

    if strict_ci and os.environ.get("SLLM_H3_NETWORK_DISABLED") != "1":
        _fail("strict Phase 7 compatibility compile requires network-none execution")
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    commands: list[dict[str, Any]] = []
    temporary = Path(tempfile.mkdtemp(prefix=f"sllm-phase7-{target}-", dir="/tmp"))
    try:
        host_object = temporary / f"probe-{target}.o"
        host_binary = temporary / f"probe-{target}.elf"
        fatbin = temporary / f"probe-{target}.fatbin"
        device = temporary / f"probe-{target}.device.elf"
        compile_argv = [
            str(compiler), "-D__HIP_ROCclr__=1", "-O3", "-DNDEBUG", "-std=gnu++17",
            f"--offload-arch={target}", "-mcode-object-version=6", "-mno-wavefrontsize64",
            "-o", str(host_object), "-x", "hip", "-c", str(SOURCE_PATH),
        ]
        link_argv = [
            str(compiler), "-O3", "-DNDEBUG", f"--offload-arch={target}",
            "-mcode-object-version=6", "-mno-wavefrontsize64", "--hip-link",
            "--rtlib=compiler-rt", "-unwindlib=libgcc", str(host_object), "-o",
            str(host_binary), str(runtime),
        ]
        for identifier, argv in (("compile", compile_argv), ("link", link_argv)):
            commands.append(_command(identifier, argv, cwd=repo))
        commands.append(_command(
            "extract-fatbin", [str(objcopy), f"--dump-section=.hip_fatbin={fatbin}", str(host_object)], cwd=repo
        ))
        list_result = _command(
            "list-bundles", [str(bundler), "--list", "--type=o", f"--input={fatbin}"], cwd=repo
        )
        commands.append(list_result)
        bundles = [line.strip() for line in list_result["stdout"].decode("utf-8", "strict").splitlines() if line.strip()]
        expected_device = f"hipv4-amdgcn-amd-amdhsa--{target}"
        if bundles != [expected_device, "host-x86_64-unknown-linux-gnu-"]:
            _fail(f"Phase 7 compatibility bundle list is not exact: {bundles}")
        commands.append(_command(
            "extract-device",
            [str(bundler), "--unbundle", "--type=o", f"--targets={expected_device}", f"--input={fatbin}", f"--output={device}"],
            cwd=repo,
        ))
        notes = _command("inspect-device", [str(readobj), "--notes", str(device)], cwd=repo)
        metadata = notes["stdout"].decode("utf-8", "replace")
        match = re.search(r"amdhsa\.target:\s*amdgcn-amd-amdhsa--(gfx[0-9]+)", metadata)
        if not match or match.group(1) != target:
            _fail("Phase 7 device code metadata does not prove the exact target")
        version = _command("compiler-version", [str(compiler), "--version"], cwd=repo)["stdout"].decode("utf-8", "replace").splitlines()[0]
        if "23." not in version:
            _fail("Phase 7 compatibility compiler is not the pinned LLVM 23 family")
        # Only the five build/inspection commands form the report. Version checks are
        # preflight facts and do not change the exact row command count.
        command_records = [
            {key: command[key] for key in ("id", "argv_sha256", "exit_code")}
            for command in commands
        ]
        report = {
            "schema_version": "phase7-compatibility-compile-report-v1",
            "state": "PASS",
            "claim": {"compile_only": True, "runtime_verified": False, "numerics_verified": False, "performance_verified": False},
            "target": target,
            "candidate": {"commit": commit, "tree": tree, "immutable": not dirty},
            "toolchain": {
                "rocm_root": str(rocm_root), "rocm_release": "7.14.0",
                "compiler": str(compiler), "compiler_version": version,
                "code_object": "V6", "wave_size": 32,
            },
            "artifact": {
                "device_sha256": _sha_file(device), "device_bytes": device.stat().st_size,
                "bundle_ids": bundles, "metadata_target": match.group(1), "retained": False,
            },
            "execution": {
                "started_at": started_at.isoformat().replace("+00:00", "Z"),
                "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "duration_seconds": round(time.monotonic() - started, 6),
                "network_isolated": strict_ci,
                "commands": command_records,
                "cleanup": "PASS",
            },
        }
        _validate_report(report)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    output_dir.mkdir(parents=True)
    report_path = output_dir / "report.json"
    data = canonical_bytes(report)
    report_path.write_bytes(data)
    (output_dir / "report.json.sha256").write_text(hashlib.sha256(data).hexdigest() + "  report.json\n", encoding="ascii")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rocm-root", type=Path, default=Path("/opt/rocm/core-7.14"))
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--strict-ci", action="store_true")
    parser.add_argument("--expected-sha")
    parser.add_argument("--allow-dirty-local", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run(
            target=args.target, output_dir=args.output_dir, rocm_root=args.rocm_root,
            repo=args.repo, strict_ci=args.strict_ci, expected_sha=args.expected_sha,
            allow_dirty_local=args.allow_dirty_local,
        )
    except (ContractError, OSError, ValueError, UnicodeError) as exc:
        print(f"Phase 7 compatibility compile: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"Phase 7 compatibility compile: PASS target={report['target']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
