#!/usr/bin/env python3
"""Run the bounded Phase 7 short-odd GPU observation for a selected profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from common import ContractError, ROOT, canonical_bytes, read_json  # noqa: E402
from engine_performance_common import (  # noqa: E402
    metric_values,
    read_json as read_performance_json,
    resolved_row,
    summary_stats,
)
from phase7_lifecycle import validate_contracts  # noqa: E402

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Phase 7 GPU observation dependency missing: {exc}") from exc


SCHEMA_PATH = ROOT / "ci/schema/phase7-gpu-observation-v1.schema.json"
LOCK_PATH = ROOT / "docs/models/locks/qwen3.5-4b-bf16.json"
DEFAULT_CACHE = Path(
    "/home/homelab1/.cache/sllm/models/Qwen--Qwen3.5-4B/"
    "snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)
TUPLE_TARGETS = {
    "local-v620-gfx1030-rocm714-hwe617": "gfx1030",
    "local-r9700-gfx1201-rocm714-hwe617": "gfx1201",
}
EXECUTED_TIERS = ["tier_g0", "tier_g3", "tier_g4", "tier_p1"]
BUILD_CONFIG = {
    "cargo_command": "cargo +1.97.1 build --locked --offline --release -p sllm-cli",
    "cargo_profile": "release",
    "rust_toolchain": "1.97.1",
    "ROCM_PATH": "/opt/rocm",
    "HIP_PATH": "/opt/rocm",
    "SLLM_HIP_COMPILER": "/opt/rocm/bin/amdclang++",
    "SLLM_HIP_CODEGEN_FEATURES": "co_v6,wave32,xnack=unsupported,sramecc=unsupported,generic_processor_version=0",
    "SLLM_ENABLE_HIP_RUNTIME": "1",
    "SLLM_ENABLE_PUBLIC_HIP_RUNTIME": "1",
    "SLLM_ENABLE_HIP_COMPILE_PROBE": "0",
}


def _fail(message: str) -> None:
    raise ContractError(message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(arguments: list[str]) -> str:
    result = subprocess.run(["git", "-C", str(ROOT), *arguments], capture_output=True, text=True, check=False, timeout=20)
    if result.returncode != 0:
        _fail(f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def _run(argv: list[str], *, env: dict[str, str] | None = None, timeout: int = 1800) -> None:
    result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, check=False, timeout=timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout)[-4000:]
        _fail(f"Phase 7 command failed ({' '.join(argv[:3])}): {detail}")


def _selection(path: Path) -> dict[str, Any]:
    selection = read_json(path)
    if not isinstance(selection, dict) or selection.get("state") != "PASS":
        _fail("Phase 7 GPU observation requires a PASS profile selection")
    profile, _ = validate_contracts()
    expected = next(item for item in profile["profiles"] if item["name"] == selection.get("profile"))
    for key in ("host_rows", "compile_targets", "gpu_tuples", "gpu_tiers", "performance_lane", "retention_days", "timeout_minutes", "blocking"):
        if selection.get(key) != expected[key]:
            _fail(f"Phase 7 profile selection field drifted: {key}")
    if selection.get("claims") != profile["claims"]:
        _fail("Phase 7 profile selection claims drifted")
    return selection


def _build(target: str, root: Path, commit: str) -> tuple[Path, Path]:
    target_dir = root / f"build-{target}"
    env = dict(os.environ)
    env.update(BUILD_CONFIG)
    env["CMAKE_HIP_ARCHITECTURES"] = target
    env["CARGO_TARGET_DIR"] = str(target_dir)
    env["LD_LIBRARY_PATH"] = "/opt/rocm/core-7.14/lib"
    _run(["cargo", "+1.97.1", "build", "--locked", "--offline", "--release", "-p", "sllm-cli"], env=env, timeout=1800)
    binary = target_dir / "release/sllm"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        _fail(f"Phase 7 build did not produce the {target} CLI")
    manifest = root / f"build-identity-{target}.json"
    command = [
        sys.executable, "ci/tools/create_engine_build_identity.py",
        "--source-root", str(ROOT), "--output", str(manifest), "--binary", str(binary),
        "--target", target, "--source-base-revision", commit,
    ]
    configs = dict(BUILD_CONFIG)
    configs["CMAKE_HIP_ARCHITECTURES"] = target
    for key, value in configs.items():
        command.extend(["--build-config", f"{key}={value}"])
    _run(command, timeout=300)
    return binary, manifest


def _validate_summary(summary: dict[str, Any]) -> None:
    schema = read_json(SCHEMA_PATH)
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(summary),
        key=lambda error: list(error.path),
    )
    if errors:
        _fail("Phase 7 GPU summary schema failed: " + "; ".join(error.message for error in errors[:5]))
    targets = [row["target"] for row in summary["rows"]]
    expected = [TUPLE_TARGETS[row["tuple_id"]] for row in summary["rows"]]
    if targets != expected or len(set(targets)) != len(targets):
        _fail("Phase 7 GPU summary target/tuple identity drifted")
    if summary["executed_tiers"] != EXECUTED_TIERS:
        _fail("Phase 7 GPU summary claims tiers outside the executed direct full-model path")


def run(
    *, selection_path: Path, output_dir: Path, model_cache: Path,
    strict_ci: bool, expected_sha: str | None, allow_dirty_local: bool,
) -> dict[str, Any]:
    selection = _selection(selection_path)
    if selection["gpu_tiers"] != EXECUTED_TIERS:
        _fail("Phase 7 selection requests GPU tiers that this controller does not execute")
    commit = _git(["rev-parse", "HEAD"])
    tree = _git(["rev-parse", "HEAD^{tree}"])
    dirty = bool(_git(["status", "--porcelain=v1", "--untracked-files=all"]))
    if strict_ci and (dirty or expected_sha != commit):
        _fail("strict Phase 7 GPU observation requires a clean exact candidate")
    if dirty and not strict_ci and not allow_dirty_local:
        _fail("dirty local Phase 7 GPU observation requires --allow-dirty-local")
    if not model_cache.is_dir() or model_cache.is_symlink():
        _fail("Phase 7 model cache is missing or unsafe")
    if output_dir.exists() or output_dir.is_symlink():
        _fail(f"refusing to overwrite Phase 7 GPU output: {output_dir}")

    temporary = Path(tempfile.mkdtemp(prefix="sllm-phase7-gpu-", dir="/tmp"))
    rows: list[dict[str, Any]] = []
    try:
        for tuple_id in selection["gpu_tuples"]:
            target = TUPLE_TARGETS[tuple_id]
            binary, manifest = _build(target, temporary, commit)
            row_id = f"engine-performance-direct-4b-{target}-short-odd"
            row_output = temporary / row_id
            _run([
                sys.executable, "ci/tools/run_engine_performance.py",
                "--row", row_id,
                "--binary", str(binary),
                "--build-manifest", str(manifest),
                "--model-lock", str(LOCK_PATH),
                "--model-cache", str(model_cache),
                "--output-dir", str(row_output),
            ], timeout=5400)
            report_path = row_output / "report.json"
            raw_path = row_output / "raw-result.json"
            report, _report_raw, report_sha = read_performance_json(report_path, "Phase 7 performance report")
            raw, _raw_bytes, raw_sha = read_performance_json(raw_path, "Phase 7 raw performance result", 64 * 1024 * 1024)
            row = resolved_row(next(item for item in json.loads((ROOT / "ci/matrix/engine-performance-direct-v1.json").read_text())["rows"] if item["row_id"] == row_id))
            values = metric_values(raw, row)
            if report["state"] != "PASS" or report["claims"]["hard_gate"] or report["cleanup"]["process_group_gone"] is not True:
                _fail(f"Phase 7 performance row did not produce a clean observational PASS: {row_id}")
            rows.append({
                "tuple_id": tuple_id,
                "target": target,
                "row_id": row_id,
                "report_sha256": report_sha,
                "raw_sha256": raw_sha,
                "metrics": {name: summary_stats(samples) for name, samples in values.items()},
                "health": "PASS",
                "fallback": False,
                "cleanup": "PASS",
            })
        summary = {
            "schema_version": "phase7-gpu-observation-v1",
            "state": "PASS",
            "profile": selection["profile"],
            "performance_lane": selection["performance_lane"],
            "executed_tiers": EXECUTED_TIERS,
            "claims": {"performance_hard_gate": False, "optimized": False, "faster": False, "compatibility_lifecycle": "experimental"},
            "candidate": {"commit": commit, "tree": tree, "immutable": not dirty},
            "model": {"repo_id": "Qwen/Qwen3.5-4B", "resolved_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "lock_fingerprint": "sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae"},
            "rows": rows,
            "cleanup": "PASS",
        }
        _validate_summary(summary)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    output_dir.mkdir(parents=True)
    data = canonical_bytes(summary)
    (output_dir / "summary.json").write_bytes(data)
    (output_dir / "summary.json.sha256").write_text(hashlib.sha256(data).hexdigest() + "  summary.json\n", encoding="ascii")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--strict-ci", action="store_true")
    parser.add_argument("--expected-sha")
    parser.add_argument("--allow-dirty-local", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run(
            selection_path=args.selection, output_dir=args.output_dir,
            model_cache=args.model_cache, strict_ci=args.strict_ci,
            expected_sha=args.expected_sha, allow_dirty_local=args.allow_dirty_local,
        )
    except (ContractError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        print(f"Phase 7 GPU observation: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"Phase 7 GPU observation: PASS profile={summary['profile']} rows={len(summary['rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
