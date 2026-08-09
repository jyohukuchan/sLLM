#!/usr/bin/env python3
"""GPU-only G2 row runner.

The host contract can inspect this runner, but it cannot turn a CPU/stub run
into a numerical result.  Actual invocation requires an explicit GPU-runner
environment and the dedicated G2 executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ContractError, ROOT, read_json, sha256_file  # noqa: E402
from validate_rmsnorm_g2_contracts import (  # noqa: E402
    ABSOLUTE_RANGE, ATOL, BYTE_SIZE, CASE_IDS, CASE_ROWS, CASE_SEEDS, G2_BINARY,
    MODEL_LOCK_FINGERPRINT, MODEL_LOCK_PATH, RESOLVED_REVISION, ROWS, SCHEMAS, TOLERANCE_PATH,
    TOLERANCE_ID, candidate_sha256, extract_synthetic_slice_payload, validate_artifact,
    query_build_identity as _query_build_identity,
    validate_candidate, validate_matrix, validate_slice_record, validate_tolerance,
)

TIMEOUT_SECONDS = 600


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def query_build_identity(binary: Path, repo: Path = ROOT) -> dict[str, Any]:
    """Query the executable's control-plane identity before model/HIP work."""
    return _query_build_identity(binary, repo)


def _candidate(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "reviewed_sha": args.reviewed_sha,
        "tested_sha": args.tested_sha,
        "workflow_sha": args.workflow_sha,
        "git_tree_oid": args.tree_oid,
        "worktree_clean": True,
        "revision_input": "full-sha",
    }
    for name in ("reviewed_sha", "tested_sha", "workflow_sha", "git_tree_oid"):
        value = values[name]
        if not isinstance(value, str) or len(value) != 40 or value.lower() != value or any(c not in "0123456789abcdef" for c in value):
            raise ContractError(f"{name} must be a full lowercase SHA")
    return values


def _empty_case(order: int, reason: str) -> dict[str, Any]:
    return {
        "order": order, "id": CASE_IDS[order], "rows": CASE_ROWS[order], "n": 2560,
        "input_seed": CASE_SEEDS[order], "state": "FAIL", "request_sha256": "0" * 64,
        "output_sha256": "0" * 64, "reference_sha256": "0" * 64, "classification": "finite", "dispatch_count": 0,
        "fallback_used": False, "max_abs_error": 0.0, "max_rel_error": 0.0,
        "nan_count": 0, "inf_count": 0, "timeout": False, "crashed": False,
    }


def make_failure_report(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    slice_record: dict[str, Any],
    artifact: dict[str, Any],
    reason: str,
    *,
    exit_code: int = 1,
    timed_out: bool = False,
    crashed: bool = False,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> dict[str, Any]:
    started = _now()
    finished = _now()
    target = args.target
    device = next(item["device"] for item in validate_matrix(ROOT)["targets"] if item["target"] == target)
    lock_sha = sha256_file(ROOT / MODEL_LOCK_PATH)
    tolerance_sha = sha256_file(ROOT / TOLERANCE_PATH)
    report = {
        "schema_version": "rmsnorm-g2-report-v1",
        "report_id": f"rmsnorm-g2-{target}-{_sha((started + reason).encode())}",
        "row_id": f"rmsnorm-g2-{target}", "target": target, "state": "FAIL", "required": True,
        "candidate": candidate, "tree_oid": candidate["git_tree_oid"],
        "model": {"used": True, "full_model_used": False, "tokenizer_used": False, "generation_used": False, "lock_path": MODEL_LOCK_PATH, "lock_sha256": lock_sha, "fingerprint": MODEL_LOCK_FINGERPRINT, "resolved_revision": RESOLVED_REVISION, "slice": {"tensor_name": slice_record["tensor"]["name"], "source_shard": slice_record["tensor"]["source_shard"], "dtype": "BF16", "shape": [2560], "header_length_bytes": 79064, "data_offsets": [15360, 20480], "absolute_byte_range": [94432, 99552], "size_bytes": BYTE_SIZE, "sha256": slice_record["output"]["sha256"], "recipe_sha256": _sha(json.dumps(slice_record["recipe"], sort_keys=True, separators=(",", ":")).encode()), "raw_stored": False}},
        "tolerance": {"schema_path": TOLERANCE_PATH, "schema_sha256": tolerance_sha, "tolerance_id": TOLERANCE_ID, "atol": ATOL, "rtol": 0.015625},
        "artifact": {"artifact_schema_path": SCHEMAS["artifact"], "artifact_schema_sha256": sha256_file(ROOT / SCHEMAS["artifact"]), "artifact_id": artifact["artifact_id"], "binary_sha256": artifact["binary"]["sha256"], "binary_sidecar_sha256": artifact["binary"]["sidecar_sha256"], "binary_source_sha256": artifact["binary"]["source_sha256"], "binary_source_set_sha256": artifact["binary"]["build_source_set"]["source_set_sha256"], "binary_role": "dedicated-g2-runtime", "h3_or_g1_substitution": False},
        "scope": {"selected_backend": "hip", "model_used": True, "full_model_used": False, "tokenizer_used": False, "generation_used": False, "hip_only": True, "fallback_allowed": False, "fallback_used": False, "cpu_fallback_used": False, "dispatch_count": 0},
        "device": {**device, "target": target},
        "dispatch": {"backend": "hip", "kernel_id": 1, "kernel_symbol": "rmsnorm.baseline.wave32.v1", "device_symbol": "sllm_rmsnorm_baseline_wave32_v1", "dispatch_count": 0, "workgroup_size_x": 256, "fallback_allowed": False, "fallback_used": False},
        "prerequisites": [dict(item) for item in artifact["prerequisites"]],
        "cases": [_empty_case(order, reason) for order in range(6)],
        "health_pre": {"available": False, "reliable": False, "state": "UNAVAILABLE", "target": target, "ras_uncorrectable_count": 0},
        "health_post": {"available": False, "reliable": False, "state": "UNAVAILABLE", "target": target, "ras_uncorrectable_count": 0},
        "process_pre": {"state": "CLEAN", "residual_runner_children": [], "gpu_processes": []},
        "process_post": {"state": "CLEAN", "residual_runner_children": [], "gpu_processes": []},
        "execution": {"started_at": started, "finished_at": finished, "duration_ns": 0, "exit_code": exit_code, "timed_out": timed_out, "crashed": crashed, "binary_stdout_sha256": _sha(stdout), "binary_stderr_sha256": _sha(stderr), "failure_reason": reason},
        "collection": {"expected_cases": 6, "collected_cases": 6, "passed_cases": 0, "failed_cases": 6, "expected_rows": 1, "collected_rows": 1},
    }
    return report


def run_row(args: argparse.Namespace, repo: Path = ROOT, *, strict_git: bool = False) -> dict[str, Any]:
    if args.target not in ("gfx1030", "gfx1201"):
        raise ContractError("G2 runner target is not canonical")
    matrix = validate_matrix(repo)
    validate_tolerance(repo)
    candidate = _candidate(args)
    validate_candidate(candidate, repo, strict_git=strict_git)
    declared_slice_record = read_json(Path(args.slice_record))
    validate_slice_record(declared_slice_record, repo)
    artifact = validate_artifact(read_json(Path(args.artifact)), repo, binary_path=Path(args.binary))
    query_build_identity(Path(args.binary), repo)
    slice_record, payload = extract_synthetic_slice_payload(Path(args.slice_file), declared_slice_record, repo)
    if declared_slice_record["output"] != slice_record["output"]:
        raise ContractError("G2 runtime slice record SHA/size does not match the exact synthetic extractor output")
    if slice_record["recipe"]["commit"] != candidate["reviewed_sha"]:
        raise ContractError("G2 slice recipe is not bound to the reviewed candidate")
    if artifact["row_id"] != f"rmsnorm-g2-{args.target}" or artifact["candidate"] != candidate:
        raise ContractError("G2 artifact and candidate/target do not bind to the runner")
    if artifact["artifact_id"] != f"rmsnorm-g2-{args.target}-{artifact['binary']['sha256']}":
        raise ContractError("G2 runner artifact ID is stale")
    if Path(args.binary).name != G2_BINARY or artifact["binary"]["g2_binary_name"] != G2_BINARY:
        raise ContractError("G2 runner refuses G1/H3 binary substitution")
    if os.environ.get("SLLM_G2_GPU_EXECUTION") != "1":
        report = make_failure_report(args, candidate, slice_record, artifact, "GPU-only G2 execution was not explicitly enabled")
        _write_report(Path(args.output_dir), report)
        return report
    started_ns = time.monotonic_ns()
    raw_fd = -1
    try:
        if not hasattr(os, "memfd_create"):
            raise ContractError("G2 runner requires Linux memfd support to avoid persisting raw slices")
        raw_fd = os.memfd_create("sllm-g2-synthetic-slice", os.MFD_CLOEXEC)
        offset = 0
        while offset < len(payload):
            written = os.write(raw_fd, payload[offset:])
            if written <= 0:
                raise ContractError("G2 runner could not materialize the complete extractor payload in memory")
            offset += written
        os.lseek(raw_fd, 0, os.SEEK_SET)
        slice_arg = f"/proc/self/fd/{raw_fd}"
        completed = subprocess.run([str(args.binary), "--target", args.target, "--slice", slice_arg], capture_output=True, check=False, timeout=TIMEOUT_SECONDS, pass_fds=(raw_fd,))
        reason = "" if completed.returncode == 0 else "dedicated G2 binary failed; CPU/stub/unavailable is not PASS"
        report = make_failure_report(args, candidate, slice_record, artifact, reason or "binary result parsing is intentionally deferred to GPU evidence", exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        if completed.returncode == 0:
            report["state"] = "FAIL"
            report["execution"]["failure_reason"] = "binary success requires GPU evidence parser and oracle binding; host runner never fabricates PASS"
        report["execution"]["duration_ns"] = time.monotonic_ns() - started_ns
    except subprocess.TimeoutExpired as exc:
        report = make_failure_report(args, candidate, slice_record, artifact, "G2 binary timed out", timed_out=True, stdout=exc.stdout or b"", stderr=exc.stderr or b"")
    finally:
        if raw_fd >= 0:
            os.close(raw_fd)
    _write_report(Path(args.output_dir), report)
    return report


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "rmsnorm-g2-report.json"
    path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    (path.with_name(path.name + ".sha256")).write_text(sha256_file(path) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--target", required=True)
    parser.add_argument("--slice-record", required=True, type=Path)
    parser.add_argument("--slice-file", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reviewed-sha", required=True)
    parser.add_argument("--tested-sha", required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--tree-oid", required=True)
    args = parser.parse_args()
    try:
        report = run_row(args, args.repo.resolve(), strict_git=True)
    except (ContractError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"G2 runner: FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"schema_version": "rmsnorm-g2-runner-result-v1", "state": report["state"], "target": args.target, "collected_cases": report["collection"]["collected_cases"]}, sort_keys=True))
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
