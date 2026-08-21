#!/usr/bin/env python3
"""Fail-closed controller for the Phase 36 MI300X Session A matrix.

The controller intentionally owns only orchestration and bounded result
validation.  Numerical work is performed by the Rust/HIP evidence binaries;
their stdout is retained as raw evidence outside the tracked summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from run_rmsnorm_g1_runtime import (  # noqa: E402
    RunnerError,
    encode_request,
    independent_rmsnorm_oracle,
    parse_response,
)

SUMMARY_NAME = "phase36-mi300x-session-a-summary-v1.json"
SCHEMA_VERSION = "phase36-mi300x-session-a-summary-v1"
TARGET = "gfx942"
TIMEOUT_SECONDS = 900
PROCESS_GROUP_TERM_GRACE_SECONDS = 1.0
PROCESS_GROUP_KILL_GRACE_SECONDS = 2.0
MAX_STDOUT_BYTES = 4 * 1024 * 1024
TOTAL_CASES = 99
RMSNORM_WIDTHS = (1, 3, 255, 256, 257, 2560, 4096)
RMSNORM_EPSILON = 1.0e-5


class SessionAError(RuntimeError):
    """A malformed request or producer result; never converted to PASS."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _binary(bin_dir: Path, name: str) -> Path:
    # Do not resolve symlinks: execution of a substituted artifact is not a
    # valid Session A result, and the producer itself remains responsible for
    # its artifact identity checks.
    path = bin_dir / name
    if path.parent != bin_dir:
        raise SessionAError(f"binary escaped --bin-dir: {name}")
    return path


def matrix(bin_dir: Path, device_index: int, target: str) -> list[dict[str, Any]]:
    """Return the frozen 99-case logical matrix and exact producer commands."""
    if target != TARGET:
        raise SessionAError("Session A execution is restricted to exact target gfx942")
    def common(name: str) -> str:
        return str(_binary(bin_dir, name))
    flags = ["--device-index", str(device_index), "--target", target]
    return [
        {"family": "fp8-matmul", "binary": "sllm-fp8-matmul-evidence", "expected_cases": 2, "command": [common("sllm-fp8-matmul-evidence"), str(device_index), target], "cli": "positional"},
        {"family": "bf16-matmul", "binary": "sllm-matmul-g1-evidence", "expected_cases": 17, "command": [common("sllm-matmul-g1-evidence"), *flags, "--phase12-subset"], "cli": "flags+phase12-subset"},
        {"family": "elementwise", "binary": "sllm-elementwise-g1-evidence", "expected_cases": 21, "command": [common("sllm-elementwise-g1-evidence"), *flags], "cli": "flags"},
        {"family": "attention-preprocess", "binary": "sllm-attention-preprocess-g1-evidence", "expected_cases": 8, "command": [common("sllm-attention-preprocess-g1-evidence"), *flags], "cli": "flags"},
        {"family": "kv-state", "binary": "sllm-kv-state-g1-evidence", "expected_cases": 19, "command": [common("sllm-kv-state-g1-evidence"), *flags], "cli": "flags"},
        {"family": "full-attention", "binary": "sllm-full-attention-g1-evidence", "expected_cases": 16, "command": [common("sllm-full-attention-g1-evidence"), *flags, "--phase12-subset"], "cli": "flags+phase12-subset"},
        {"family": "output-gate", "binary": "sllm-output-gate-g1-evidence", "expected_cases": 6, "command": [common("sllm-output-gate-g1-evidence"), *flags], "cli": "flags"},
        {"family": "rmsnorm", "binary": "sllm-rmsnorm-g1-evidence", "expected_cases": len(RMSNORM_WIDTHS), "command": [common("sllm-rmsnorm-g1-evidence"), *flags], "cli": "flags+stdin-protocol"},
        {"family": "gdn", "binary": "sllm-linear-attention-g1-evidence", "expected_cases": 3, "command": [common("sllm-linear-attention-g1-evidence"), *flags, "--phase12-subset"], "cli": "flags+phase12-subset"},
    ]


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _parse_json(stdout: bytes, family: str) -> dict[str, Any]:
    if not stdout or len(stdout) > MAX_STDOUT_BYTES:
        raise SessionAError(f"{family}: empty or oversized JSON stdout")
    try:
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise SessionAError(f"{family}: duplicate JSON key {key}")
                result[key] = item
            return result

        def reject_constant(token: str) -> None:
            raise SessionAError(f"{family}: non-finite JSON constant {token}")

        value = json.loads(
            stdout.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SessionAError(f"{family}: producer stdout is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SessionAError(f"{family}: producer JSON must be an object")
    return value


def _result_count(document: dict[str, Any], family: str) -> int:
    cases = document.get("cases")
    if isinstance(cases, list):
        return len(cases)
    for key in ("operations", "selected_cases", "collected_cases"):
        value = document.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    raise SessionAError(f"{family}: no bounded case count in producer report")


def _dispatch_evidence(document: dict[str, Any], family: str) -> int:
    """Read only producer-owned dispatch fields; case count is never a proxy."""
    explicit: list[int] = []
    for key in ("dispatch_count", "kernel_dispatches"):
        value = document.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            explicit.append(value)
    cases = document.get("cases")
    if isinstance(cases, list):
        for index, case in enumerate(cases):
            if not isinstance(case, dict) or "dispatch_count" not in case:
                continue
            value = case["dispatch_count"]
            if not isinstance(value, int) or isinstance(value, bool):
                raise SessionAError(f"{family}: case {index} dispatch_count is not an integer")
            explicit.append(value)
    if not explicit:
        cases = document.get("cases")
        if family == "full-attention":
            if document.get("gpu_execution") is not True or not isinstance(cases, list) or not cases:
                raise SessionAError(f"{family}: producer native-dispatch marker is absent")
            if any(case.get("metadata_match") is not True or case.get("no_fallback") is not True for case in cases if isinstance(case, dict)):
                raise SessionAError(f"{family}: baseline metadata/fallback marker is not positive")
            if any(not isinstance(case, dict) for case in cases):
                raise SessionAError(f"{family}: malformed native-dispatch marker")
            return len(cases)
        if family == "kv-state":
            if document.get("gpu_execution") is not True or not isinstance(cases, list) or not cases:
                raise SessionAError(f"{family}: producer native-dispatch marker is absent")
            if any(case.get("no_fallback_observed") is not True for case in cases if isinstance(case, dict)):
                raise SessionAError(f"{family}: no-fallback marker is not positive")
            if any(not isinstance(case, dict) for case in cases):
                raise SessionAError(f"{family}: malformed native-dispatch marker")
            return len(cases)
        raise SessionAError(f"{family}: producer dispatch evidence is absent")
    if any(value <= 0 for value in explicit):
        raise SessionAError(f"{family}: producer dispatch evidence is zero")
    # A report-level total is authoritative when present.  Otherwise use the
    # sum of per-case counts, preserving the producer's positive evidence.
    if isinstance(document.get("dispatch_count"), int) and not isinstance(document["dispatch_count"], bool):
        return document["dispatch_count"]
    if isinstance(document.get("kernel_dispatches"), int) and not isinstance(document["kernel_dispatches"], bool):
        return document["kernel_dispatches"]
    case_counts = [case["dispatch_count"] for case in cases if isinstance(case, dict) and isinstance(case.get("dispatch_count"), int) and not isinstance(case.get("dispatch_count"), bool)] if isinstance(cases, list) else []
    if not case_counts:
        raise SessionAError(f"{family}: producer dispatch evidence is not attributable to cases")
    return sum(case_counts)


def _validate_common(document: dict[str, Any], family: str, expected: int, expected_device_index: int | None = None) -> int:
    if document.get("target") != TARGET:
        raise SessionAError(f"{family}: producer target is not gfx942")
    if document.get("state") != "PASS" or document.get("pass", True) is not True:
        raise SessionAError(f"{family}: producer did not report PASS")
    if expected_device_index is not None:
        observed_device = document.get("device_index")
        if isinstance(observed_device, bool) or not isinstance(observed_device, int) or observed_device != expected_device_index:
            raise SessionAError(f"{family}: producer device_index does not match the requested device")
    count = _result_count(document, family)
    if count != expected:
        raise SessionAError(f"{family}: selected {count} cases, expected {expected}")
    saw_cleanup = False
    for key, value in _walk(document):
        lowered = key.lower()
        if lowered.startswith("no_fallback"):
            if value is not True:
                raise SessionAError(f"{family}: positive no-fallback evidence is absent ({key})")
            continue
        if "fallback" in lowered and isinstance(value, bool) and value:
            raise SessionAError(f"{family}: fallback evidence is true ({key})")
        if "fallback" in lowered and isinstance(value, int) and not isinstance(value, bool) and value != 0:
            raise SessionAError(f"{family}: fallback count is nonzero ({key})")
        cleanup_key = "cleanup" in lowered or lowered in {"durable_quarantine", "retryable_cleanup", "cleanup_pending", "cleanup_durable", "cleanup_accounting_errors"}
        saw_cleanup = saw_cleanup or cleanup_key
        if cleanup_key and isinstance(value, int) and not isinstance(value, bool) and value != 0:
            raise SessionAError(f"{family}: cleanup is nonzero ({key})")
        if lowered in {"terminal_zero", "zero_after_shutdown"} and value is not True:
            raise SessionAError(f"{family}: cleanup did not return to zero")
    _dispatch_evidence(document, family)
    if not saw_cleanup:
        raise SessionAError(f"{family}: cleanup evidence is absent")
    if isinstance(document.get("selected_backend"), str) and document["selected_backend"] != "hip":
        raise SessionAError(f"{family}: selected backend is not HIP")
    _validate_family_contract(document, family)
    return count


def _validate_family_contract(document: dict[str, Any], family: str) -> None:
    cases = document.get("cases")
    if family == "fp8-matmul":
        if document.get("resident_dtype") != "e4m3fnuz" or document.get("provider") != "hipblaslt-native" or not isinstance(cases, list):
            raise SessionAError("fp8-matmul: gfx942 resident/provider contract is not FNUZ native")
        for index, case in enumerate(cases):
            if not isinstance(case, dict) or case.get("dispatch_count") != 2 or case.get("kernel_id") != 5 or case.get("kernel_symbol") != "matmul.fp8.outer.hipblaslt.v1" or case.get("device_symbol") != "hipblasLtMatmul":
                raise SessionAError(f"fp8-matmul: case {index} is not the gfx942 hipBLASLt provider")
        return
    if family == "bf16-matmul":
        if not isinstance(cases, list):
            raise SessionAError("bf16-matmul: case dispatch metadata is absent")
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                raise SessionAError(f"bf16-matmul: malformed case {index}")
            m = case.get("m")
            symbol, device = case.get("kernel_symbol"), case.get("device_symbol")
            if not isinstance(m, int) or not isinstance(symbol, str) or not isinstance(device, str):
                raise SessionAError(f"bf16-matmul: case {index} lacks kernel symbols")
            if m == 1 and (symbol != "matmul.bf16_fp32.decode.wave64.v1" or device != "sllm_matmul_bf16_fp32_decode_wave64_v1"):
                raise SessionAError(f"bf16-matmul: case {index} is not the gfx942 wave64 decode provider")
            if 1 < m <= 8 and (symbol != "matmul.bf16_fp32.decode.serial_rows.wave64.v1" or device != "sllm_matmul_bf16_fp32_decode_serial_rows_wave64_v1"):
                raise SessionAError(f"bf16-matmul: case {index} is not the gfx942 wave64 serial provider")
            if m > 8 and (symbol != "matmul.hipblas.gemm_ex.v2" or device != "hipblasGemmEx"):
                raise SessionAError(f"bf16-matmul: case {index} is not the gfx942 GEMM provider")
        return
    if family == "gdn":
        if not isinstance(cases, list):
            raise SessionAError("gdn: case dispatch metadata is absent")
        for index, case in enumerate(cases):
            if not isinstance(case, dict) or case.get("dispatch_count") != 2 or case.get("recurrent_kernel_id") != 2 or case.get("kernel_symbol") != "linear_attention.gdn.v1" or case.get("recurrent_device_symbol") != "sllm_linear_attention_recurrent_gated_norm_v1":
                raise SessionAError(f"gdn: case {index} is not the gfx942 baseline recurrent provider")


def _rmsnorm_payload(width: int) -> tuple[bytes, bytes, bytes]:
    if width not in RMSNORM_WIDTHS:
        raise SessionAError(f"unsupported RMSNorm width {width}")
    activation_words = tuple((0x3F80, 0xBF80, 0x3FC0, 0xC020, 0x4000, 0x3E80)[index % 6] for index in range(width))
    scale_words = tuple((0x3F00, 0xBF00, 0x3F80, 0x0000)[index % 4] for index in range(width))
    activation = b"".join(struct.pack("<H", value) for value in activation_words)
    raw_scale = b"".join(struct.pack("<H", value) for value in scale_words)
    try:
        request = encode_request((1, width), RMSNORM_EPSILON, activation, raw_scale)
    except RunnerError as exc:
        raise SessionAError(f"rmsnorm width {width}: request protocol generation failed") from exc
    oracle = independent_rmsnorm_oracle(activation, raw_scale, 1, width, RMSNORM_EPSILON)
    return request, oracle, activation


def _run_process(command: list[str], *, payload: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if payload is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise SessionAError(f"producer could not start: {exc}") from exc
    try:
        stdout, stderr = process.communicate(input=payload, timeout=TIMEOUT_SECONDS)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        cleanup_errors: list[str] = []
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as cleanup_error:
            cleanup_errors.append(f"SIGTERM failed: {cleanup_error}")
        try:
            process.communicate(timeout=PROCESS_GROUP_TERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as cleanup_error:
                cleanup_errors.append(f"SIGKILL failed: {cleanup_error}")
            try:
                process.communicate(timeout=PROCESS_GROUP_KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired as reap_error:
                cleanup_errors.append(f"reap timed out: {reap_error}")
            except OSError as reap_error:
                cleanup_errors.append(f"reap failed: {reap_error}")
        except OSError as reap_error:
            cleanup_errors.append(f"reap failed: {reap_error}")
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as cleanup_error:
                cleanup_errors.append(f"SIGKILL failed: {cleanup_error}")
            try:
                process.communicate(timeout=PROCESS_GROUP_KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired as final_reap_error:
                cleanup_errors.append(f"reap timed out: {final_reap_error}")
            except OSError as final_reap_error:
                cleanup_errors.append(f"reap failed: {final_reap_error}")
        if cleanup_errors:
            detail = "; ".join(cleanup_errors)
            raise SessionAError(
                f"timeout after {TIMEOUT_SECONDS}s; process-group cleanup failed: {detail}"
            ) from exc
        raise SessionAError(
            f"timeout after {TIMEOUT_SECONDS}s; process group terminated and reaped"
        ) from exc


def _command_device_index(command: list[str]) -> int:
    try:
        if "--device-index" in command:
            return int(command[command.index("--device-index") + 1])
        return int(command[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise SessionAError("producer command lacks a valid device index") from exc


def _run_family(row: dict[str, Any], output_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    family = row["family"]
    result: dict[str, Any] = {"family": family, "binary": row["binary"], "expected_cases": row["expected_cases"], "command": row["command"], "state": "DRY_RUN" if dry_run else "FAIL", "selected_cases": 0, "dispatch_count": 0, "fallback_used": False, "cleanup_retryable": 0, "cleanup_durable": 0}
    if dry_run:
        return result
    try:
        if family == "rmsnorm":
            raw_dir = output_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            cases: list[dict[str, Any]] = []
            try:
                device_index = _command_device_index(row["command"])
            except SessionAError as exc:
                raise SessionAError("rmsnorm: command lacks a valid device index") from exc
            for width in RMSNORM_WIDTHS:
                command = row["command"]
                request, oracle, _activation = _rmsnorm_payload(width)
                completed = _run_process(command, payload=request)
                raw = raw_dir / f"rmsnorm-width-{width}.bin"
                stdout = completed.stdout if isinstance(completed.stdout, bytes) else b""
                stderr = completed.stderr if isinstance(completed.stderr, bytes) else b"invalid stderr type"
                raw.write_bytes(stdout)
                if completed.returncode != 0 or stderr:
                    raise SessionAError(f"rmsnorm width {width}: exit={completed.returncode} or stderr present")
                try:
                    parsed = parse_response(stdout, expected_target=TARGET, expected_device_index=device_index, expected_shape=(1, width), expected_epsilon=RMSNORM_EPSILON)
                except (RunnerError, struct.error, ValueError) as exc:
                    raise SessionAError(f"rmsnorm width {width}: response metadata is invalid") from exc
                dispatch_count = parsed.get("dispatch_count")
                if not isinstance(dispatch_count, int) or isinstance(dispatch_count, bool) or dispatch_count != 1:
                    raise SessionAError(f"rmsnorm width {width}: dispatch provider contract is invalid")
                kernel_id = parsed.get("kernel_id")
                if not isinstance(kernel_id, int) or isinstance(kernel_id, bool) or kernel_id != 2:
                    raise SessionAError(f"rmsnorm width {width}: kernel provider contract is not gfx942 wave64")
                resource_counts = parsed.get("resource_counts")
                expected_resource_counts = {"allocation_count": 3, "copy_count": 3, "kernel_count": 1}
                if (
                    not isinstance(resource_counts, dict)
                    or any(
                        not isinstance(resource_counts.get(key), int)
                        or isinstance(resource_counts.get(key), bool)
                        or resource_counts.get(key) != value
                        for key, value in expected_resource_counts.items()
                    )
                ):
                    raise SessionAError(f"rmsnorm width {width}: provider resource contract is not 3/3/1")
                actual = parsed.get("output")
                if not isinstance(actual, bytes) or actual != oracle:
                    raise SessionAError(f"rmsnorm width {width}: output does not byte-match independent BF16 oracle")
                cases.append({"width": width, "dispatch_count": parsed["dispatch_count"], "kernel_id": parsed["kernel_id"], "allocation_count": resource_counts["allocation_count"], "copy_count": resource_counts["copy_count"], "kernel_count": resource_counts["kernel_count"], "oracle_match": True, "oracle_sha256": hashlib.sha256(oracle).hexdigest(), "output_sha256": hashlib.sha256(actual).hexdigest(), "protocol_version": 2})
            result.update({"state": "PASS", "selected_cases": len(cases), "dispatch_count": sum(item["dispatch_count"] for item in cases), "cases": cases})
            return result
        completed = _run_process(row["command"])
        raw = output_dir / "raw" / f"{family}.json"
        raw.parent.mkdir(parents=True, exist_ok=True)
        stdout = completed.stdout if isinstance(completed.stdout, bytes) else b""
        stderr = completed.stderr if isinstance(completed.stderr, bytes) else b"invalid stderr type"
        raw.write_bytes(stdout)
        if completed.returncode != 0 or stderr:
            raise SessionAError(f"exit={completed.returncode}, stderr={len(stderr)} bytes")
        document = _parse_json(stdout, family)
        count = _validate_common(document, family, row["expected_cases"], _command_device_index(row["command"]))
        dispatch = _dispatch_evidence(document, family)
        result.update({"state": "PASS", "selected_cases": count, "dispatch_count": dispatch})
    except SessionAError as exc:
        result["error"] = str(exc)
    return result


NOT_SCHEDULED_ERROR = "not scheduled after earlier family failure"


def _not_scheduled_result(row: dict[str, Any]) -> dict[str, Any]:
    """Retain the frozen matrix shape without launching work after a failure."""
    return {
        "family": row["family"],
        "binary": row["binary"],
        "expected_cases": row["expected_cases"],
        "command": row["command"],
        "state": "FAIL",
        "selected_cases": 0,
        "dispatch_count": 0,
        "fallback_used": False,
        "cleanup_retryable": 0,
        "cleanup_durable": 0,
        "error": NOT_SCHEDULED_ERROR,
    }


def validate_summary(summary: dict[str, Any]) -> None:
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("target") != TARGET:
        raise SessionAError("summary schema version or target is invalid")
    rows = summary.get("families")
    if not isinstance(rows, list) or len(rows) != 9:
        raise SessionAError("summary must contain the nine independent families")
    expected_rows = [("fp8-matmul", 2), ("bf16-matmul", 17), ("elementwise", 21), ("attention-preprocess", 8), ("kv-state", 19), ("full-attention", 16), ("output-gate", 6), ("rmsnorm", 7), ("gdn", 3)]
    if [(row.get("family"), row.get("expected_cases")) for row in rows] != expected_rows:
        raise SessionAError("summary family order or case counts drifted")
    for row in rows:
        selected_row = row.get("selected_cases")
        if not isinstance(selected_row, int) or isinstance(selected_row, bool) or not 0 <= selected_row <= row["expected_cases"]:
            raise SessionAError(f"summary selected count is invalid for {row.get('family')}")
        if row.get("state") == "PASS" and selected_row != row["expected_cases"]:
            raise SessionAError(f"summary PASS row is incomplete for {row.get('family')}")
    expected = sum(int(row["expected_cases"]) for row in rows)
    selected = sum(int(row.get("selected_cases", 0)) for row in rows)
    if expected != TOTAL_CASES:
        raise SessionAError(f"logical matrix total drifted: {expected}")
    if summary.get("expected_cases") != TOTAL_CASES or summary.get("selected_cases") != selected:
        raise SessionAError("summary case totals are inconsistent")
    if summary.get("state") == "PASS" and selected != TOTAL_CASES:
        raise SessionAError("PASS summary does not select all 99 cases")


def run_session_a(*, bin_dir: Path, device_index: int, target: str, output_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    if target != TARGET:
        raise SessionAError("--target must be exactly gfx942 for Session A")
    if device_index < 0:
        raise SessionAError("--device-index must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = matrix(bin_dir, device_index, target)
    results: list[dict[str, Any]] = []
    stopped = False
    for row in rows:
        if stopped:
            results.append(_not_scheduled_result(row))
            continue
        result = _run_family(row, output_dir, dry_run=dry_run)
        results.append(result)
        if result["state"] == "FAIL":
            stopped = True
    selected = sum(int(row["selected_cases"]) for row in results)
    state = "DRY_RUN" if dry_run else ("PASS" if selected == TOTAL_CASES and all(row["state"] == "PASS" for row in results) else "FAIL")
    failure_count = sum(
        row["state"] == "FAIL" and row.get("error") != NOT_SCHEDULED_ERROR
        for row in results
    )
    summary: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "state": state, "started_at": _utc_now(), "target": target, "device_index": device_index, "dry_run": dry_run, "expected_cases": TOTAL_CASES, "selected_cases": selected, "families": results, "raw_outputs": "raw/", "failure_count": failure_count}
    validate_summary(summary)
    (output_dir / SUMMARY_NAME).write_bytes(_json_bytes(summary))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        summary = run_session_a(bin_dir=args.bin_dir, device_index=args.device_index, target=args.target, output_dir=args.output_dir, dry_run=args.dry_run)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if summary["state"] in {"PASS", "DRY_RUN"} else 1
    except SessionAError as exc:
        print(f"phase36 Session A: FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
