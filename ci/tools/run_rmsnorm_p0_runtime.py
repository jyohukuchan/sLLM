#!/usr/bin/env python3
"""Fail-closed canonical RMSNorm P0 row runner.

Host use validates the complete contract and emits FAIL without starting the
producer.  Canonical execution additionally requires an explicit environment
gate and collects live pre/post health/process observations under the shared
host lock around the exact routed GPU execution.
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

from common import ContractError, ROOT, canonical_bytes, read_json, sha256_file  # noqa: E402
from validate_rmsnorm_p0_contracts import (  # noqa: E402
    DTYPE_CONTRACT, MATRIX_PATH, MEASUREMENT_ITERATIONS, MODEL_LOCK_FINGERPRINT,
    MODEL_LOCK_PATH, PRODUCER_STATUS, REVIEW_POLICY_PATH,
    P0_RUNTIME_LD_LIBRARY_PATH,
    RESOLVED_REVISION, TARGETS, TOTAL_DISPATCHES, WARMUP_ITERATIONS,
    artifact_summary, case_set_sha256, source_set, validate_artifact,
    validate_candidate, validate_matrix, validate_report,
    validate_review_policy, validate_runtime_result,
)
from run_g0_preflight import (  # noqa: E402
    amd_smi_list_json, nonblocking_host_lock, observe_health, observe_processes,
    require_available_observation,
)
from validate_g0_contracts import (  # noqa: E402
    AMD_SMI_EXECUTABLE, row_by_id, validate_g0_matrix, validate_health,
    validate_processes, validate_routing, validate_visibility_environment,
)

TIMEOUT_SECONDS = 900
OUTPUT_LIMIT_BYTES = 1 << 20


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _candidate(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "reviewed_sha": args.reviewed_sha,
        "tested_sha": args.tested_sha,
        "workflow_sha": args.workflow_sha,
        "git_tree_oid": args.tree_oid,
        "worktree_clean": True,
        "revision_input": "full-sha",
    }


def _unavailable_health(target: str) -> dict[str, Any]:
    return {"available": False, "reliable": False, "state": "UNAVAILABLE", "target": target, "ras_uncorrectable_count": 0}


def _clean_process() -> dict[str, Any]:
    return {"state": "CLEAN", "residual_runner_children": [], "gpu_processes": []}


def _observe_live(repo: Path, target: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    row = row_by_id(validate_g0_matrix(repo), f"g0-{target}")
    routing = amd_smi_list_json(row, executable=AMD_SMI_EXECUTABLE)
    visibility = validate_visibility_environment(
        {"HIP_VISIBLE_DEVICES": str(routing["hip_id"]), "CUDA_VISIBLE_DEVICES": None, "GPU_DEVICE_ORDINAL": None}
    )
    validate_routing(routing, visibility, row)
    observed_health = observe_health(
        row, routing, amd_smi=AMD_SMI_EXECUTABLE, sysfs_root=Path("/sys/bus/pci/devices")
    )
    require_available_observation(observed_health, "P0 live health")
    validate_health(observed_health, "P0 live", row)
    observed_processes = observe_processes(row, routing, amd_smi=AMD_SMI_EXECUTABLE)
    require_available_observation(observed_processes, "P0 live process")
    validate_processes(observed_processes, "P0 live", row)
    facts = observed_health["facts"]
    if facts["ras_uncorrectable_count"] != 0 or facts["sysfs_ras_uncorrectable_count"] != 0:
        raise ContractError("P0 live health reports an uncorrectable error")
    if observed_processes["gpu_processes"] or observed_processes["residual_runner_children"]:
        raise ContractError("P0 live process observation is not clean")
    health = {
        "available": True, "reliable": True, "state": "OK", "target": target,
        "ras_uncorrectable_count": 0,
    }
    return health, _clean_process(), routing


def _parse_runtime_stdout(stdout: bytes) -> dict[str, Any]:
    if not stdout or len(stdout) > OUTPUT_LIMIT_BYTES:
        raise ContractError("P0 producer stdout is empty or exceeds the bounded result size")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ContractError(f"duplicate P0 runtime key: {key}")
            value[key] = item
        return value

    try:
        document = json.loads(stdout.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"P0 producer output is not one JSON document: {exc}") from exc
    if not isinstance(document, dict) or canonical_bytes(document) != stdout:
        raise ContractError("P0 producer output is not one exact canonical JSON line")
    return document


def make_report(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    artifact: dict[str, Any],
    artifact_sha: str,
    reason: str,
    *,
    runtime_result: dict[str, Any] | None = None,
    exit_code: int = 1,
    timed_out: bool = False,
    crashed: bool = False,
    stdout: bytes = b"",
    stderr: bytes = b"",
    duration_ns: int = 0,
    health_pre: dict[str, Any] | None = None,
    health_post: dict[str, Any] | None = None,
    process_pre: dict[str, Any] | None = None,
    process_post: dict[str, Any] | None = None,
    repo: Path = ROOT,
) -> dict[str, Any]:
    target = args.target
    started = getattr(args, "_started_at", None) or _now()
    finished = _now()
    measurements = [] if runtime_result is None else runtime_result["cases"]
    measurement_sha = "0" * 64 if runtime_result is None else runtime_result["measurement_sha256"]
    runtime_sha = "0" * 64 if runtime_result is None else _sha(canonical_bytes(runtime_result))
    healthy = all(
        evidence is not None
        and evidence.get("state") == "OK"
        and evidence.get("available") is True
        and evidence.get("reliable") is True
        for evidence in (health_pre, health_post)
    )
    clean = all(
        evidence is not None
        and evidence.get("state") == "CLEAN"
        and not evidence.get("residual_runner_children")
        and not evidence.get("gpu_processes")
        for evidence in (process_pre, process_post)
    )
    state = "PASS" if runtime_result is not None and exit_code == 0 and not timed_out and not crashed and not stderr and healthy and clean else "FAIL"
    dispatch_count = TOTAL_DISPATCHES if runtime_result is not None else 0
    device = next(item["device"] for item in validate_matrix(repo)["targets"] if item["target"] == target)
    report = {
        "schema_version": "rmsnorm-p0-report-v1",
        "report_id": f"rmsnorm-p0-{target}-{_sha((started + reason).encode())}",
        "row_id": f"rmsnorm-p0-{target}", "target": target, "state": state, "required": True,
        "run": {"run_id": args.run_id, "run_attempt": args.run_attempt},
        "candidate": candidate, "tree_oid": candidate["git_tree_oid"],
        "matrix": {"path": MATRIX_PATH, "sha256": sha256_file(repo / MATRIX_PATH)},
        "review_policy": {"path": REVIEW_POLICY_PATH, "sha256": sha256_file(repo / REVIEW_POLICY_PATH)},
        "case_set_sha256": case_set_sha256(repo),
        "model_lock": {"path": MODEL_LOCK_PATH, "sha256": sha256_file(repo / MODEL_LOCK_PATH), "fingerprint": MODEL_LOCK_FINGERPRINT, "resolved_revision": RESOLVED_REVISION, "used": False},
        "source_set_sha256": source_set(repo)["sha256"],
        "artifact": {**artifact_summary(artifact, artifact_sha), "producer_status": PRODUCER_STATUS},
        "dtype": dict(DTYPE_CONTRACT),
        "scope": {"selected_backend": "hip", "gpu_execution": runtime_result is not None, "public_rmsnorm_path": True, "semantic_op_used": True, "model_used": False, "fallback_allowed": False, "fallback_used": False, "cpu_fallback_used": False, "optimized_claim": False, "other_engine_comparison": False, "performance_hard_gate": False},
        "device": {**device, "target": target},
        "dispatch": {"backend": "hip", "kernel_id": 1, "kernel_symbol": "rmsnorm.baseline.wave32.v1", "device_symbol": "sllm_rmsnorm_baseline_wave32_v1", "workgroup_size_x": 256, "dispatch_count": dispatch_count, "fallback_allowed": False, "fallback_used": False},
        "prerequisites": [dict(item) for item in artifact["prerequisites"]],
        "measurements": measurements, "measurement_sha256": measurement_sha, "runtime_result_sha256": runtime_sha,
        "health_pre": health_pre or _unavailable_health(target),
        "health_post": health_post or _unavailable_health(target),
        "process_pre": process_pre or _clean_process(),
        "process_post": process_post or _clean_process(),
        "execution": {"started_at": started, "finished_at": finished, "duration_ns": duration_ns, "exit_code": exit_code, "timed_out": timed_out, "crashed": crashed, "stdout_sha256": _sha(stdout), "stderr_sha256": _sha(stderr), "failure_reason": reason},
        "collection": {"expected_cases": 5, "collected_cases": len(measurements), "passed_cases": len(measurements), "failed_cases": 0 if measurements else 5, "warmup_dispatches": len(measurements) * WARMUP_ITERATIONS, "measurement_dispatches": len(measurements) * MEASUREMENT_ITERATIONS},
    }
    return validate_report(report, repo)


def run_row(
    args: argparse.Namespace, repo: Path = ROOT, *, strict_git: bool = False
) -> dict[str, Any]:
    if args.target not in TARGETS:
        raise ContractError("P0 runner target is not canonical")
    validate_matrix(repo)
    validate_review_policy(repo)
    candidate = validate_candidate(_candidate(args), repo, strict_git=strict_git)
    artifact_path = Path(args.artifact)
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise ContractError("P0 artifact input must be a regular non-symlink file")
    binary_path = Path(args.binary)
    if binary_path.parent.resolve() != artifact_path.parent.resolve():
        raise ContractError("P0 producer and artifact must share one canonical artifact directory")
    artifact_sha = sha256_file(artifact_path)
    artifact = validate_artifact(read_json(artifact_path), repo, binary_path=binary_path)
    if artifact["candidate"] != candidate or artifact["target"] != args.target:
        raise ContractError("P0 runner artifact is a stale/mixed candidate or target")
    args._started_at = _now()
    if os.environ.get("SLLM_P0_GPU_EXECUTION") != "1":
        report = make_report(
            args, candidate, artifact, artifact_sha,
            "GPU-only P0 execution was not explicitly enabled; host cannot produce numeric PASS",
            repo=repo,
        )
        _write_report(Path(args.output_dir), report)
        return report
    if any(getattr(args, name, None) is not None for name in ("health_pre", "health_post", "process_pre", "process_post")):
        raise ContractError("P0 canonical execution rejects precomputed pre/post observations")
    command = [
        str(args.binary), "--target", args.target,
        "--case-set-sha256", case_set_sha256(repo),
        "--warmup", str(WARMUP_ITERATIONS),
        "--iterations", str(MEASUREMENT_ITERATIONS),
        "--timing-contract", "rmsnorm-p0-timing-v1",
        "--reviewed-sha", candidate["reviewed_sha"],
        "--tested-sha", candidate["tested_sha"],
        "--workflow-sha", candidate["workflow_sha"],
        "--tree-oid", candidate["git_tree_oid"],
        "--artifact-id", artifact["artifact_id"],
        "--artifact-sha256", artifact_sha,
        "--binary-sha256", artifact["binary"]["sha256"],
        "--binary-sidecar-sha256", artifact["binary"]["sidecar_sha256"],
        "--source-set-sha256", artifact["source_set"]["sha256"],
        "--matrix-sha256", sha256_file(repo / MATRIX_PATH),
        "--model-lock-sha256", sha256_file(repo / MODEL_LOCK_PATH),
    ]
    health_pre = _unavailable_health(args.target)
    health_post = _unavailable_health(args.target)
    process_pre = _clean_process()
    process_post = _clean_process()
    started_ns = time.monotonic_ns()
    try:
        with nonblocking_host_lock(Path("/tmp/sllm-g0.lock")):
            health_pre, process_pre, routing_pre = _observe_live(repo, args.target)
            execution_environment = os.environ.copy()
            for selector in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "ROCR_VISIBLE_DEVICES"):
                execution_environment.pop(selector, None)
            execution_environment["HIP_VISIBLE_DEVICES"] = str(routing_pre["hip_id"])
            execution_environment["LD_LIBRARY_PATH"] = P0_RUNTIME_LD_LIBRARY_PATH
            command.extend([
                "--physical-hip-index", str(routing_pre["hip_id"]),
                "--device-bdf", str(routing_pre["bdf"]),
                "--device-uuid", str(routing_pre["uuid"]),
                "--device-product", str(routing_pre["product"]),
            ])
            execution_error: Exception | None = None
            try:
                completed = subprocess.run(
                    command, cwd=repo, env=execution_environment, capture_output=True, check=False,
                    timeout=TIMEOUT_SECONDS, start_new_session=True,
                )
            except Exception as exc:
                execution_error = exc
            finally:
                health_post, process_post, routing_post = _observe_live(repo, args.target)
            if routing_post != routing_pre:
                raise ContractError("P0 canonical device routing changed across execution")
            if execution_error is not None:
                raise execution_error
    except subprocess.TimeoutExpired as exc:
        report = make_report(
            args, candidate, artifact, artifact_sha, "P0 producer timed out",
            timed_out=True, stdout=exc.stdout or b"", stderr=exc.stderr or b"",
            duration_ns=time.monotonic_ns() - started_ns,
            health_pre=health_pre, health_post=health_post,
            process_pre=process_pre, process_post=process_post,
            repo=repo,
        )
        _write_report(Path(args.output_dir), report)
        return report
    duration_ns = time.monotonic_ns() - started_ns
    stdout = completed.stdout if isinstance(completed.stdout, bytes) else str(completed.stdout).encode()
    stderr = completed.stderr if isinstance(completed.stderr, bytes) else str(completed.stderr).encode()
    if len(stdout) + len(stderr) > OUTPUT_LIMIT_BYTES:
        raise ContractError("P0 producer output exceeds the bounded runner limit")
    if completed.returncode != 0 or stderr:
        report = make_report(
            args, candidate, artifact, artifact_sha,
            "P0 producer failed or wrote stderr; non-GPU/stub/fallback is not PASS",
            exit_code=completed.returncode, stdout=stdout, stderr=stderr,
            duration_ns=duration_ns, health_pre=health_pre, health_post=health_post,
            process_pre=process_pre, process_post=process_post,
            repo=repo,
        )
        _write_report(Path(args.output_dir), report)
        return report
    runtime_result = _parse_runtime_stdout(stdout)
    validate_runtime_result(runtime_result, artifact, artifact_sha, repo)
    report = make_report(
        args, candidate, artifact, artifact_sha,
        "complete dedicated producer measurements and public HIP event timing retained",
        runtime_result=runtime_result, exit_code=0, stdout=stdout, stderr=stderr,
        duration_ns=duration_ns, health_pre=health_pre, health_post=health_post,
        process_pre=process_pre, process_post=process_post,
        repo=repo,
    )
    _write_report(Path(args.output_dir), report)
    return report


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "rmsnorm-p0-report.json"
    path.write_bytes(canonical_bytes(report))
    path.with_name(path.name + ".sha256").write_text(sha256_file(path) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--target", required=True)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--health-pre", type=Path)
    parser.add_argument("--health-post", type=Path)
    parser.add_argument("--process-pre", type=Path)
    parser.add_argument("--process-post", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True, type=int)
    parser.add_argument("--reviewed-sha", required=True)
    parser.add_argument("--tested-sha", required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--tree-oid", required=True)
    args = parser.parse_args()
    try:
        report = run_row(args, args.repo.resolve(), strict_git=True)
    except (ContractError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"P0 runner: FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"schema_version": "rmsnorm-p0-runner-result-v1", "state": report["state"], "target": args.target, "collected_cases": report["collection"]["collected_cases"]}, sort_keys=True))
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
