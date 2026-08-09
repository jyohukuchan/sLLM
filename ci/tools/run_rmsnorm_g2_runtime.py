#!/usr/bin/env python3
"""GPU-only G2 row runner.

The host contract can inspect this runner, but it cannot turn a CPU/stub run
into a numerical result.  Actual invocation requires an explicit GPU-runner
environment and the dedicated G2 executable.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import selectors
import signal
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
    validate_candidate, validate_matrix, validate_slice_record, validate_tolerance, _schema_validate,
    extract_verified_slice_payload,
)

TIMEOUT_SECONDS = 600
PROTOCOL_SCHEMA = "rmsnorm-g2-runtime-result-v1"
MAX_PROTOCOL_BYTES = 16 * 1024 * 1024
MAX_PROTOCOL_FIELD_BYTES = 2 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unavailable_health(target: str) -> dict[str, Any]:
    return {"available": False, "reliable": False, "state": "UNAVAILABLE", "target": target, "ras_uncorrectable_count": 0}


def _clean_process() -> dict[str, Any]:
    return {"state": "CLEAN", "residual_runner_children": [], "gpu_processes": []}


def _load_observation(path: Path | None, target: str, *, kind: str) -> dict[str, Any]:
    if path is None:
        return _unavailable_health(target) if kind == "health" else _clean_process()
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"G2 {kind} observation must be a regular file")
    try:
        value = read_json(path)
    except (OSError, ValueError) as exc:
        raise ContractError(f"G2 {kind} observation is not valid JSON: {exc}") from exc
    if kind == "health":
        expected = {"available", "reliable", "state", "target", "ras_uncorrectable_count"}
        if set(value) != expected or value["target"] != target:
            raise ContractError("G2 health observation shape or target is invalid")
        if not isinstance(value["available"], bool) or not isinstance(value["reliable"], bool) or value["state"] not in {"OK", "UNAVAILABLE", "DEGRADED"} or not isinstance(value["ras_uncorrectable_count"], int) or value["ras_uncorrectable_count"] < 0:
            raise ContractError("G2 health observation values are invalid")
    else:
        if set(value) != {"state", "residual_runner_children", "gpu_processes"} or value != _clean_process():
            raise ContractError("G2 process observation is not clean and closed")
    return value


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            pass
        process.wait(timeout=2)


def _run_bounded_binary(argv: list[str], *, cwd: Path, pass_fds: tuple[int, ...]) -> subprocess.CompletedProcess[bytes]:
    """Run one G2 child with a hard protocol/output bound and group cleanup."""

    process = subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        pass_fds=pass_fds,
    )
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    chunks: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + TIMEOUT_SECONDS
    timed_out = False
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _kill_process_group(process)
                break
            events = selector.select(min(remaining, 0.25))
            if not events and process.poll() is not None:
                continue
            for stream, _mask in events:
                name = selector.get_key(stream).data
                data = os.read(stream.fileno(), 65536)
                if not data:
                    selector.unregister(stream)
                    stream.close()
                    continue
                chunks[name].extend(data)
                limit = MAX_PROTOCOL_BYTES if name == "stdout" else MAX_PROTOCOL_FIELD_BYTES
                if len(chunks[name]) > limit:
                    _kill_process_group(process)
                    raise ContractError(f"G2 binary {name} exceeded the bounded protocol output")
        if process.poll() is None:
            process.wait(timeout=2)
    except BaseException:
        if process.poll() is None:
            _kill_process_group(process)
        raise
    finally:
        selector.close()
    return subprocess.CompletedProcess(argv, 124 if timed_out else process.returncode, bytes(chunks["stdout"]), bytes(chunks["stderr"]))


def _decode_payload(value: Any, expected_size: int, label: str) -> bytes:
    if not isinstance(value, str) or not value or len(value) > MAX_PROTOCOL_FIELD_BYTES * 2:
        raise ContractError(f"G2 protocol {label} is missing or exceeds the field bound")
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ContractError(f"G2 protocol {label} is not canonical base64") from exc
    if base64.b64encode(decoded).decode("ascii") != value or len(decoded) != expected_size:
        raise ContractError(f"G2 protocol {label} is not the exact bounded BF16 payload")
    return decoded


def _parse_protocol(stdout: bytes, target: str) -> dict[str, Any]:
    if len(stdout) > MAX_PROTOCOL_BYTES or not stdout.endswith(b"\n") or stdout.count(b"\n") != 1:
        raise ContractError("G2 binary protocol is not one bounded JSON line")
    try:
        document = json.loads(stdout[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"G2 binary protocol is not UTF-8 JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ContractError("G2 binary protocol root is not an object")
    _schema_validate(ROOT, "runtime_result", document)
    expected_root = {"schema_version", "state", "target", "model_used", "full_model_used", "tokenizer_used", "generation_used", "selected_backend", "dispatch_count", "fallback_used", "cases"}
    if set(document) != expected_root or document["schema_version"] != PROTOCOL_SCHEMA or document["state"] != "PASS" or document["target"] != target or document["model_used"] is not True or document["full_model_used"] is not False or document["tokenizer_used"] is not False or document["generation_used"] is not False or document["selected_backend"] != "hip" or document["dispatch_count"] != 6 or document["fallback_used"] is not False:
        raise ContractError("G2 binary protocol root violates the exact no-fallback contract")
    if not isinstance(document["cases"], list) or len(document["cases"]) != 6:
        raise ContractError("G2 binary protocol does not contain exactly six cases")
    for order, case in enumerate(document["cases"]):
        expected_case = {"order", "id", "rows", "n", "input_seed", "request_b64", "output_b64", "dispatch"}
        if not isinstance(case, dict) or set(case) != expected_case or case["order"] != order or case["id"] != CASE_IDS[order] or case["rows"] != CASE_ROWS[order] or case["n"] != 2560 or case["input_seed"] != CASE_SEEDS[order]:
            raise ContractError("G2 binary protocol case identity/order is not canonical")
        _decode_payload(case["request_b64"], CASE_ROWS[order] * 2560 * 2, f"{case['id']} request")
        _decode_payload(case["output_b64"], CASE_ROWS[order] * 2560 * 2, f"{case['id']} output")
        dispatch = case["dispatch"]
        expected_dispatch = {"backend", "kernel_id", "kernel_symbol", "device_symbol", "dispatch_count", "workgroup_size_x", "fallback_allowed", "fallback_used"}
        if not isinstance(dispatch, dict) or set(dispatch) != expected_dispatch or dispatch != {"backend": "hip", "kernel_id": 1, "kernel_symbol": "rmsnorm.baseline.wave32.v1", "device_symbol": "sllm_rmsnorm_baseline_wave32_v1", "dispatch_count": 1, "workgroup_size_x": 256, "fallback_allowed": False, "fallback_used": False}:
            raise ContractError("G2 binary protocol dispatch evidence is not canonical")
    expected_json = json.dumps(document, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
    if stdout != expected_json:
        raise ContractError("G2 binary protocol is not canonical compact JSON")
    return document


def _oracle_cases(protocol: dict[str, Any], raw_scale: bytes) -> tuple[list[dict[str, Any]], bool, str]:
    import numpy as np
    if len(raw_scale) != 2560 * 2:
        raise ContractError("G2 raw scale payload is not BF16[2560]")
    sys.path.insert(0, str(ROOT))
    from tests.reference.oracles import bf16_decode, bf16_encode_rne
    from tests.reference.semantic_rmsnorm import classify_semantic_output, compare_semantic_outputs
    scale = np.frombuffer(raw_scale, dtype="<u2").copy()
    results: list[dict[str, Any]] = []
    all_passed = True
    for case in protocol["cases"]:
        request = _decode_payload(case["request_b64"], case["rows"] * 2560 * 2, f"{case['id']} request")
        actual_bytes = _decode_payload(case["output_b64"], case["rows"] * 2560 * 2, f"{case['id']} output")
        activation = np.frombuffer(request, dtype="<u2").reshape(case["rows"], 2560)
        actual = np.frombuffer(actual_bytes, dtype="<u2").reshape(case["rows"], 2560)
        activation_values = bf16_decode(activation).astype(np.float32, copy=False)
        scale_values = bf16_decode(scale).astype(np.float32, copy=False)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            squared = np.multiply(activation_values, activation_values, dtype=np.float32)
            sum_squares = np.sum(squared, axis=1, keepdims=True, dtype=np.float32)
            mean_square = np.divide(sum_squares, np.float32(2560), dtype=np.float32)
            denominator = np.add(mean_square, np.float32(1.0e-6), dtype=np.float32)
            inverse_rms = np.reciprocal(np.sqrt(denominator, dtype=np.float32), dtype=np.float32)
            effective_scale = np.add(np.float32(1.0), scale_values, dtype=np.float32)
            normalized = np.multiply(np.multiply(activation_values, inverse_rms, dtype=np.float32), effective_scale, dtype=np.float32)
        reference_bits = bf16_encode_rne(normalized).astype("<u2", copy=False)
        reference = bf16_decode(reference_bits).astype("<f4", copy=False)
        comparison = compare_semantic_outputs(actual, reference, atol=0.0078125, rtol=0.015625)
        actual_values = bf16_decode(actual).astype(np.float64, copy=False)
        reference_values = bf16_decode(reference_bits).astype(np.float64, copy=False)
        finite = np.isfinite(actual_values) & np.isfinite(reference_values)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.divide(np.abs(actual_values - reference_values), np.maximum(np.abs(reference_values), np.finfo(np.float64).tiny), where=finite, out=np.zeros_like(actual_values))
        actual_classes = classify_semantic_output(actual)
        nan_count = int(np.count_nonzero(actual_classes == "NaN"))
        inf_count = int(np.count_nonzero((actual_classes == "+Inf") | (actual_classes == "-Inf")))
        result = {
            "order": case["order"], "id": case["id"], "rows": case["rows"], "n": 2560, "input_seed": case["input_seed"],
            "state": "PASS" if comparison.passed else "FAIL", "request_sha256": _sha(request), "output_sha256": _sha(actual_bytes),
            "reference_sha256": _sha(reference_bits.tobytes()),
            "classification": "finite", "dispatch_count": 1, "fallback_used": False,
            "max_abs_error": float(comparison.max_abs_error or 0.0), "max_rel_error": float(np.max(relative[finite])) if np.any(finite) else 0.0, "nan_count": nan_count, "inf_count": inf_count, "timeout": False, "crashed": False,
        }
        if not comparison.passed:
            all_passed = False
        results.append(result)
    return results, all_passed, _sha(canonical_protocol_bytes(protocol))


def canonical_protocol_bytes(document: dict[str, Any]) -> bytes:
    return json.dumps(document, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"


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
    health_pre: dict[str, Any] | None = None,
    health_post: dict[str, Any] | None = None,
    process_pre: dict[str, Any] | None = None,
    process_post: dict[str, Any] | None = None,
    protocol_sha256: str = "0" * 64,
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
        "health_pre": health_pre or _unavailable_health(target),
        "health_post": health_post or _unavailable_health(target),
        "process_pre": process_pre or _clean_process(),
        "process_post": process_post or _clean_process(),
        "execution": {"started_at": started, "finished_at": finished, "duration_ns": 0, "exit_code": exit_code, "timed_out": timed_out, "crashed": crashed, "binary_stdout_sha256": _sha(stdout), "binary_stderr_sha256": _sha(stderr), "failure_reason": reason, "protocol_schema": PROTOCOL_SCHEMA, "protocol_sha256": protocol_sha256},
        "collection": {"expected_cases": 6, "collected_cases": 6, "passed_cases": 0, "failed_cases": 6, "expected_rows": 1, "collected_rows": 1},
    }
    return report


def run_row(args: argparse.Namespace, repo: Path = ROOT, *, strict_git: bool = False) -> dict[str, Any]:
    if args.target not in ("gfx1030", "gfx1201"):
        raise ContractError("G2 runner target is not canonical")
    validate_matrix(repo)
    validate_tolerance(repo)
    candidate = _candidate(args)
    validate_candidate(candidate, repo, strict_git=strict_git)
    declared_slice_record = read_json(Path(args.slice_record))
    validate_slice_record(declared_slice_record, repo)
    artifact = validate_artifact(read_json(Path(args.artifact)), repo, binary_path=Path(args.binary))
    query_build_identity(Path(args.binary), repo)
    if getattr(args, "slice_file", None) is not None:
        from validate_rmsnorm_g2_contracts import extract_synthetic_slice_payload
        slice_record, payload = extract_synthetic_slice_payload(Path(args.slice_file), declared_slice_record, repo)
    elif getattr(args, "cache_root", None) is not None:
        slice_record, payload = extract_verified_slice_payload(Path(args.cache_root), declared_slice_record, repo)
    else:
        raise ContractError("G2 runner requires either a temporary fixture or an explicit verified cache root")
    if declared_slice_record["output"] != slice_record["output"]:
        raise ContractError("G2 runtime slice record SHA/size does not match the exact same-FD extractor output")
    if slice_record["recipe"]["commit"] != candidate["reviewed_sha"]:
        raise ContractError("G2 slice recipe is not bound to the reviewed candidate")
    if artifact["row_id"] != f"rmsnorm-g2-{args.target}" or artifact["candidate"] != candidate:
        raise ContractError("G2 artifact and candidate/target do not bind to the runner")
    if artifact["artifact_id"] != f"rmsnorm-g2-{args.target}-{artifact['binary']['sha256']}":
        raise ContractError("G2 runner artifact ID is stale")
    if Path(args.binary).name != G2_BINARY or artifact["binary"]["g2_binary_name"] != G2_BINARY:
        raise ContractError("G2 runner refuses G1/H3 binary substitution")
    health_pre = _load_observation(getattr(args, "health_pre", None), args.target, kind="health")
    health_post = _load_observation(getattr(args, "health_post", None), args.target, kind="health")
    process_pre = _load_observation(getattr(args, "process_pre", None), args.target, kind="process")
    process_post = _load_observation(getattr(args, "process_post", None), args.target, kind="process")
    if os.environ.get("SLLM_G2_GPU_EXECUTION") != "1":
        report = make_failure_report(args, candidate, slice_record, artifact, "GPU-only G2 execution was not explicitly enabled", health_pre=health_pre, health_post=health_post, process_pre=process_pre, process_post=process_post)
        _write_report(Path(args.output_dir), report)
        return report
    if health_pre != _unavailable_health(args.target) and (health_pre["state"] != "OK" or not health_pre["available"] or not health_pre["reliable"]):
        raise ContractError("G2 canonical execution requires reliable healthy pre-execution evidence")
    if health_post != _unavailable_health(args.target) and (health_post["state"] != "OK" or not health_post["available"] or not health_post["reliable"]):
        raise ContractError("G2 canonical execution requires reliable healthy post-execution evidence")
    if health_pre == _unavailable_health(args.target) or health_post == _unavailable_health(args.target):
        raise ContractError("G2 canonical execution requires explicit pre/post health observations")
    if process_pre != _clean_process() or process_post != _clean_process():
        raise ContractError("G2 canonical execution requires clean pre/post process observations")
    started_ns = time.monotonic_ns()
    raw_fd = -1
    report: dict[str, Any]
    try:
        if not hasattr(os, "memfd_create"):
            raise ContractError("G2 runner requires Linux memfd support to avoid persisting raw slices")
        raw_fd = os.memfd_create("sllm-g2-verified-slice", os.MFD_CLOEXEC)
        offset = 0
        while offset < len(payload):
            written = os.write(raw_fd, payload[offset:])
            if written <= 0:
                raise ContractError("G2 runner could not materialize the complete extractor payload in memory")
            offset += written
        os.lseek(raw_fd, 0, os.SEEK_SET)
        completed = _run_bounded_binary([str(args.binary), "--target", args.target, "--slice-fd", str(raw_fd)], cwd=repo, pass_fds=(raw_fd,))
        protocol: dict[str, Any] | None = None
        oracle_cases: list[dict[str, Any]] | None = None
        protocol_sha = "0" * 64
        reason = ""
        if completed.returncode != 0:
            reason = "dedicated G2 binary failed; CPU/stub/unavailable is not PASS"
        elif completed.stderr:
            reason = "dedicated G2 binary wrote stderr; protocol evidence is not clean"
        else:
            try:
                protocol = _parse_protocol(completed.stdout, args.target)
                oracle_cases, oracle_passed, _ = _oracle_cases(protocol, payload)
                protocol_sha = _sha(completed.stdout)
                if not oracle_passed:
                    reason = "independent FP32 RMSNorm oracle rejected one or more canonical cases"
            except (ContractError, ValueError, OverflowError) as exc:
                reason = f"G2 protocol/oracle validation failed: {exc}"
        report = make_failure_report(args, candidate, slice_record, artifact, reason, exit_code=completed.returncode, timed_out=completed.returncode == 124, crashed=completed.returncode < 0, stdout=completed.stdout, stderr=completed.stderr, health_pre=health_pre, health_post=health_post, process_pre=process_pre, process_post=process_post, protocol_sha256=protocol_sha)
        if protocol is not None and oracle_cases is not None:
            report["cases"] = oracle_cases
            report["scope"]["dispatch_count"] = protocol["dispatch_count"]
            report["dispatch"]["dispatch_count"] = protocol["dispatch_count"]
            report["collection"] = {"expected_cases": 6, "collected_cases": 6, "passed_cases": sum(case["state"] == "PASS" for case in oracle_cases), "failed_cases": sum(case["state"] == "FAIL" for case in oracle_cases), "expected_rows": 1, "collected_rows": 1}
            report["state"] = "PASS" if not reason else "FAIL"
        report["execution"]["duration_ns"] = time.monotonic_ns() - started_ns
    except (ContractError, OSError, subprocess.SubprocessError) as exc:
        report = make_failure_report(args, candidate, slice_record, artifact, f"G2 binary execution failed closed: {exc}", health_pre=health_pre, health_post=health_post, process_pre=process_pre, process_post=process_post)
    finally:
        if raw_fd >= 0:
            os.close(raw_fd)
    from validate_rmsnorm_g2_contracts import validate_report
    validate_report(report, repo)
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--slice-file", type=Path)
    source.add_argument("--cache-root", type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reviewed-sha", required=True)
    parser.add_argument("--tested-sha", required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--tree-oid", required=True)
    parser.add_argument("--health-pre", type=Path)
    parser.add_argument("--health-post", type=Path)
    parser.add_argument("--process-pre", type=Path)
    parser.add_argument("--process-post", type=Path)
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
