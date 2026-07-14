#!/usr/bin/env python3
"""Capture and split one marked AQ4 resident diagnostic rocprof session."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


PRODUCER = load_tool("aq4_p3_producer_for_diagnostic_capture", ROOT / "tools/build-aq4-p3-selection-raw.py")
PROFILER = load_tool("aq4_p2_profiler_for_diagnostic_capture", ROOT / "tools/profile-aq4-p2-family-exclusive.py")

SCHEMA = "ullm.aq4_p3_diagnostic_rocprof_capture.v1"
MARKER_PREFIX = "ullm.aq4_p2.run.v1"
MARKER_CLOCK = "rocprofv3_monotonic_ns"
MAX_ROWS = 500_000
MARKER_KEYS = {
    "run_id", "session_id", "case_id", "case_sha256", "run_index", "run_kind"
}
MEMORY_COPY_KINDS = {
    "d2h", "h2d", "d2d", "h2h", "peer", "peertopeer", "hosttodevice",
    "devicetohost", "devicetodevice", "hosttohost",
}


class CaptureError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def self_hash(value: dict[str, Any], field: str) -> str:
    clone = json.loads(json.dumps(value, allow_nan=False))
    clone[field] = None
    return hashlib.sha256(canonical(clone)).hexdigest()


def ref(snapshot: Any) -> dict[str, str]:
    return {"path": str(snapshot.path), "sha256": snapshot.sha256}


def profiler_command(
    profiler: Any, output_directory: Path, output_name: str, runner_command: list[str]
) -> list[str]:
    if not output_directory.is_absolute():
        raise CaptureError("profile output directory must be absolute")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", output_name):
        raise CaptureError("profile output name is unsafe")
    if not runner_command or any(not isinstance(item, str) or not item for item in runner_command):
        raise CaptureError("runner command is empty or invalid")
    if not Path(runner_command[0]).is_absolute():
        raise CaptureError("runner executable path must be absolute")
    return [
        str(profiler.path),
        "--kernel-trace",
        "--hip-runtime-trace",
        "--memory-copy-trace",
        "--marker-trace",
        "--output-format",
        "csv",
        "--output-directory",
        str(output_directory),
        "--output-file",
        output_name,
        "--",
        *runner_command,
    ]


def run_profile(command: list[str], output_directory: Path, timeout: float) -> None:
    if timeout <= 0.0:
        raise CaptureError("profile timeout must be positive")
    if not output_directory.is_absolute():
        raise CaptureError("profile output directory must be absolute")
    PROFILER.canonical_path(output_directory.parent, "profile output parent")
    if output_directory.exists() or output_directory.is_symlink():
        raise CaptureError("profile output directory already exists")
    output_directory.mkdir(mode=0o700)
    stdout_path = output_directory / "rocprof.stdout"
    stderr_path = output_directory / "rocprof.stderr"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            process_group = process.pid

            def group_alive() -> bool:
                process.poll()
                try:
                    os.killpg(process_group, 0)
                except ProcessLookupError:
                    return False
                return True

            def signal_group(value: signal.Signals, wait_seconds: float) -> None:
                try:
                    os.killpg(process_group, value)
                except ProcessLookupError:
                    return
                deadline = time.monotonic() + wait_seconds
                while group_alive() and time.monotonic() < deadline:
                    time.sleep(0.02)

            signal_group(signal.SIGINT, 0.5)
            if group_alive():
                signal_group(signal.SIGTERM, 0.5)
            if group_alive():
                signal_group(signal.SIGKILL, 5.0)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if group_alive():
                raise CaptureError(
                    "rocprof diagnostic capture timed out and process group cleanup failed"
                ) from error
            raise CaptureError("rocprof diagnostic capture timed out") from error
    if return_code != 0:
        suffix = " (possible OOM/SIGKILL)" if return_code in {-9, 9, 137} else ""
        raise CaptureError(f"rocprof diagnostic capture failed with exit {return_code}{suffix}")


def discover(output_directory: Path) -> dict[str, Path]:
    patterns = {
        "kernel": "*_kernel_trace.csv",
        "hip_api": "*_hip_api_trace.csv",
        "memory_copy": "*_memory_copy_trace.csv",
        "marker": "*_marker_api_trace.csv",
    }
    result: dict[str, Path] = {}
    for kind, pattern in patterns.items():
        matches = sorted(output_directory.rglob(pattern))
        if len(matches) != 1:
            raise CaptureError(f"expected exactly one {kind} trace, got {len(matches)}")
        result[kind] = matches[0]
    return result


def csv_rows(snapshot: Any, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = snapshot.data.decode("utf-8-sig")
    except UnicodeError as error:
        raise CaptureError(f"{label} is not UTF-8") from error
    reader = csv.DictReader(text.splitlines())
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)):
        raise CaptureError(f"{label} header is missing or duplicated")
    rows: list[dict[str, str]] = []
    for line, row in enumerate(reader, 2):
        if len(rows) >= MAX_ROWS or None in row or any(value is None for value in row.values()):
            raise CaptureError(f"{label} row {line} is invalid")
        rows.append({key: value for key, value in row.items()})
    return fields, rows


def one_column(fields: list[str], aliases: tuple[str, ...], label: str) -> str:
    matches = [field for field in aliases if field in fields]
    if len(matches) != 1:
        raise CaptureError(f"trace must have exactly one {label} column")
    return matches[0]


def interval_columns(fields: list[str]) -> tuple[str, str]:
    return (
        one_column(fields, ("Start_Timestamp", "BeginNs", "start_ns"), "start"),
        one_column(fields, ("End_Timestamp", "EndNs", "end_ns"), "end"),
    )


def parse_marker_name(name: str) -> dict[str, str]:
    parts = name.split("/")
    if not parts or parts[0] != MARKER_PREFIX:
        raise CaptureError(f"unknown marker name: {name}")
    values: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise CaptureError(f"invalid marker field: {part}")
        key, value = part.split("=", 1)
        if key in values or not value:
            raise CaptureError(f"duplicate or empty marker field: {key}")
        values[key] = value
    if set(values) != MARKER_KEYS:
        raise CaptureError("marker fields differ")
    return values


def markers(snapshot: Any, raw: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    fields, rows = csv_rows(snapshot, "marker trace")
    name_column = one_column(fields, ("Name", "Marker_Name", "name"), "marker name")
    start_column, end_column = interval_columns(fields)
    expected_session = raw["resident"]["session_id"]
    result: list[dict[str, Any]] = []
    previous_end = -1
    for row in rows:
        values = parse_marker_name(row[name_column].strip())
        try:
            index = int(values["run_index"])
            start = int(row[start_column])
            end = int(row[end_column])
        except ValueError as error:
            raise CaptureError("marker integer field is invalid") from error
        expected_kind = "warmup" if index < 2 else "measured"
        if (
            index != len(result)
            or index > 11
            or values["run_kind"] != expected_kind
            or values["run_id"] != run_id
            or values["session_id"] != expected_session
            or values["case_id"] != raw["case_id"]
            or values["case_sha256"] != raw["case_sha256"]
            or start < 0
            or end <= start
            or start < previous_end
        ):
            raise CaptureError("marker order/kind/identity/interval differs")
        result.append({**values, "run_index": index, "start_ns": start, "end_ns": end})
        previous_end = end
    if len(result) != 12:
        raise CaptureError("marker trace must contain exactly 12 balanced run ranges")
    return result


def rows_by_marker(
    fields: list[str], rows: list[dict[str, str]], ranges: list[dict[str, Any]], label: str
) -> dict[int, list[dict[str, str]]]:
    start_column, end_column = interval_columns(fields)
    result = {index: [] for index in range(12)}
    for line, row in enumerate(rows, 2):
        try:
            start, end = int(row[start_column]), int(row[end_column])
        except ValueError as error:
            raise CaptureError(f"{label} row {line} clock is invalid") from error
        if start < 0 or end <= start:
            raise CaptureError(f"{label} row {line} interval is invalid")
        containing = [item for item in ranges if item["start_ns"] <= start and end <= item["end_ns"]]
        crossing = [item for item in ranges if start < item["end_ns"] and end > item["start_ns"]]
        if len(containing) > 1 or (crossing and len(containing) != 1):
            raise CaptureError(f"{label} row {line} crosses a run marker")
        if containing:
            result[containing[0]["run_index"]].append(row)
    return result


def validate_memory_copy_rows(fields: list[str], rows: list[dict[str, str]]) -> None:
    name_column = one_column(
        fields, ("Name", "Kind", "Direction", "Operation", "name"), "memory copy kind"
    )
    seen: set[str] = set()
    correlation_column = one_column(
        fields, ("Correlation_Id", "Correlation_ID", "Index", "correlation_id"),
        "memory correlation id",
    )
    for line, row in enumerate(rows, 2):
        correlation = row[correlation_column].strip()
        kind = re.sub(r"[^a-z0-9]", "", row[name_column].lower())
        if not correlation or correlation in seen:
            raise CaptureError(f"memory copy row {line} correlation differs")
        seen.add(correlation)
        if kind not in MEMORY_COPY_KINDS:
            raise CaptureError(f"unknown memory copy operation: {row[name_column]}")


def validate_all_kernel_names(fields: list[str], rows: list[dict[str, str]]) -> None:
    name_column = one_column(
        fields, ("Kernel_Name", "KernelName", "Name", "kernel_name"), "kernel name"
    )
    for row in rows:
        name = row[name_column].strip()
        try:
            family = PROFILER.classify_kernel(name)
        except PROFILER.ProfileError as error:
            raise CaptureError(f"kernel family classification failed: {error}") from error
        if family is None:
            raise CaptureError(f"unknown kernel family in source trace: {name}")


def validate_all_hip_api_names(fields: list[str], rows: list[dict[str, str]]) -> None:
    name_column = one_column(
        fields, ("Function", "Api_Name", "API_Name", "Name", "function"), "HIP API name"
    )
    known = (
        PRODUCER.D2H_APIS
        | PRODUCER.SYNC_APIS
        | PRODUCER.KNOWN_OTHER_MEMCPY_APIS
        | PRODUCER.KNOWN_OTHER_SYNC_APIS
    )
    for row in rows:
        raw_name = row[name_column].strip()
        name = PRODUCER.normalized_api_name(raw_name)
        if name in known:
            continue
        if "memcpy" in name or "synchron" in name:
            raise CaptureError(f"unknown transfer/synchronization HIP API: {raw_name}")


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(
                json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("ascii")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> Any:
    if path.exists() or path.is_symlink():
        raise CaptureError(f"refusing to overwrite split trace: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return PRODUCER.capture(path.resolve(), "split trace")


def capability(profiler_value: dict[str, Any]) -> dict[str, Any]:
    value = {
        "schema_version": PRODUCER.CAPABILITY_SCHEMA,
        "status": "complete",
        "measurement_eligible": False,
        "capability_sha256": None,
        "tool": {"name": "rocprofv3", "version": profiler_value["version"]},
        "domains": {
            "kernel_dispatch": True,
            "hip_api": True,
            "memory_copy": True,
            "d2h_memcpy": True,
            "stream_synchronize": True,
            "device_synchronize": True,
        },
        "rocprof_config": {
            "kernel_trace": True,
            "hip_api_trace": True,
            "memory_copy_trace": True,
            "marker_trace": True,
            "api_filter": "all_functions",
        },
    }
    value["capability_sha256"] = PRODUCER.self_hash(value, "capability_sha256")
    return value


def validate_resident_evidence(
    identity_path: Path, summary_path: Path, raw_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[Any], str]:
    snapshots: list[Any] = []
    identity_snapshot = PRODUCER.capture(identity_path.resolve(), "identity")
    snapshots.append(identity_snapshot)
    identity_value = PRODUCER.parse_json(identity_snapshot, "identity")
    identity = PRODUCER.validate_identity(identity_value, identity_snapshot)
    summary_snapshot = PRODUCER.capture(summary_path.resolve(), "resident summary")
    snapshots.append(summary_snapshot)
    summary_value = PRODUCER.parse_json(summary_snapshot, "resident summary")
    run_id = PRODUCER.validate_summary(
        summary_value, summary_snapshot, identity, "diagnostic"
    )
    raw_snapshot = PRODUCER.capture(raw_path.resolve(), "resident raw")
    snapshots.append(raw_snapshot)
    raw_value = PRODUCER.parse_json(raw_snapshot, "resident raw")
    raw_run_id, _runs = PRODUCER.validate_raw(
        raw_value, identity, {run_id: summary_snapshot}, "diagnostic"
    )
    if raw_run_id != run_id:
        raise CaptureError("resident raw/summary run_id differs")
    return identity, summary_value, raw_value, snapshots, run_id


def _assemble(
    *,
    traces: dict[str, Path],
    identity_path: Path,
    summary_path: Path,
    raw_path: Path,
    profiler_value: dict[str, Any],
    command: list[str],
    output_directory: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    if artifact_path.exists() or artifact_path.is_symlink():
        raise CaptureError("capture artifact already exists")
    split_directory = output_directory / "measured-runs"
    if split_directory.exists() or split_directory.is_symlink():
        raise CaptureError("measured split directory already exists")
    identity, _summary, raw, evidence_snapshots, run_id = validate_resident_evidence(
        identity_path, summary_path, raw_path
    )
    trace_snapshots = {
        kind: PRODUCER.capture(path.resolve(), f"{kind} trace") for kind, path in traces.items()
    }
    if len({snapshot.sha256 for snapshot in trace_snapshots.values()}) != len(trace_snapshots):
        raise CaptureError("source trace bytes were reused across domains")
    ranges = markers(trace_snapshots["marker"], raw, run_id)
    parsed: dict[str, tuple[list[str], dict[int, list[dict[str, str]]]]] = {}
    for kind in ("kernel", "hip_api", "memory_copy"):
        fields, rows = csv_rows(trace_snapshots[kind], f"{kind} trace")
        if kind == "kernel":
            validate_all_kernel_names(fields, rows)
        elif kind == "hip_api":
            validate_all_hip_api_names(fields, rows)
        else:
            validate_memory_copy_rows(fields, rows)
        parsed[kind] = (fields, rows_by_marker(fields, rows, ranges, f"{kind} trace"))
    split_directory.mkdir(mode=0o700)
    capability_value = capability(profiler_value)
    capability_path = output_directory / "capture-capabilities.json"
    if capability_path.exists() or capability_path.is_symlink():
        raise CaptureError("capture capability output already exists")
    write_json_atomic(capability_path, capability_value)
    capability_snapshot = PRODUCER.capture(capability_path.resolve(), "capture capabilities")
    PRODUCER.validate_capture_capabilities(capability_value, "diagnostic")
    profile_runs: list[dict[str, Any]] = []
    split_snapshots: list[Any] = []
    used_kernel_traces: set[str] = set()
    used_api_traces: set[str] = set()
    for index in range(2, 12):
        kernel_fields, kernel_runs = parsed["kernel"]
        if not kernel_runs[index]:
            raise CaptureError(f"measured run {index} kernel trace is empty")
        kernel_output_fields = list(kernel_fields)
        kernel_rows = [dict(row) for row in kernel_runs[index]]
        if "Phase" not in kernel_output_fields:
            kernel_output_fields.append("Phase")
            for row in kernel_rows:
                row["Phase"] = "prefill"
        api_fields, api_runs = parsed["hip_api"]
        if not api_runs[index]:
            raise CaptureError(f"measured run {index} HIP API trace is empty")
        memory_fields, memory_runs = parsed["memory_copy"]
        kernel_snapshot = write_csv(
            split_directory / f"run-{index:02d}_kernel_trace.csv",
            kernel_output_fields,
            kernel_rows,
        )
        api_snapshot = write_csv(
            split_directory / f"run-{index:02d}_hip_api_trace.csv",
            api_fields,
            api_runs[index],
        )
        memory_snapshot = write_csv(
            split_directory / f"run-{index:02d}_memory_copy_trace.csv",
            memory_fields,
            memory_runs[index],
        )
        split_snapshots.extend((kernel_snapshot, api_snapshot, memory_snapshot))
        if kernel_snapshot.sha256 in used_kernel_traces or api_snapshot.sha256 in used_api_traces:
            raise CaptureError("measured kernel or HIP API trace bytes were reused")
        used_kernel_traces.add(kernel_snapshot.sha256)
        used_api_traces.add(api_snapshot.sha256)
        PRODUCER.parse_kernel_trace(kernel_snapshot, "paged-kv-table-validation-v1")
        PRODUCER.parse_hip_api_trace(api_snapshot, capability_value)
        profile_runs.append(
            {
                "schema_version": PRODUCER.PROFILE_BINDING_SCHEMA,
                "case_id": raw["case_id"],
                "case_sha256": raw["case_sha256"],
                "identity_sha256": identity["identity_sha256"],
                "resident_run_index": index,
                "measurement_eligible": False,
                "clock_domain": MARKER_CLOCK,
                "kernel_trace_complete": True,
                "hip_api_trace_complete": True,
                "capture_capabilities": ref(capability_snapshot),
                "kernel_trace": ref(kernel_snapshot),
                "hip_api_trace": ref(api_snapshot),
            }
        )
    artifact = {
        "schema_version": SCHEMA,
        "status": "complete_diagnostic",
        "measurement_eligible": False,
        "promotion_eligible": False,
        "artifact_sha256": None,
        "binding": {
            "run_id": run_id,
            "resident_session_id": raw["resident"]["session_id"],
            "case_id": raw["case_id"],
            "case_sha256": raw["case_sha256"],
            "identity_sha256": identity["identity_sha256"],
            "device": identity["_resident_driver_identity"]["runtime_device"],
            "identity": ref(evidence_snapshots[0]),
            "resident_summary": ref(evidence_snapshots[1]),
            "resident_raw": ref(evidence_snapshots[2]),
        },
        "profiler": {
            **profiler_value,
            "command": command,
            "command_sha256": hashlib.sha256(canonical(command)).hexdigest(),
            "subprocess_profile_runs": 1,
        },
        "source_traces": {kind: ref(snapshot) for kind, snapshot in trace_snapshots.items()},
        "capture_capabilities": ref(capability_snapshot),
        "marker_contract": {
            "schema_version": MARKER_PREFIX,
            "clock_domain": MARKER_CLOCK,
            "range_count": 12,
            "warmup_indices": [0, 1],
            "measured_indices": list(range(2, 12)),
            "warmup_excluded": True,
        },
        "producer_profile_runs": profile_runs,
        "memory_copy_traces": [
            ref(snapshot) for snapshot in split_snapshots if "memory_copy" in snapshot.path.name
        ],
        "eligibility_blockers": [
            "rocprof instrumentation overhead forbids performance promotion",
            "one-case diagnostic evidence does not satisfy seven-prompt promotion coverage",
        ],
    }
    artifact["artifact_sha256"] = self_hash(artifact, "artifact_sha256")
    for snapshot in [*evidence_snapshots, *trace_snapshots.values(), *split_snapshots, capability_snapshot]:
        snapshot.verify()
    write_json_atomic(artifact_path, artifact)
    return artifact


def assemble(**kwargs: Any) -> dict[str, Any]:
    output_directory = kwargs["output_directory"]
    artifact_path = kwargs["artifact_path"]
    split_directory = output_directory / "measured-runs"
    capability_path = output_directory / "capture-capabilities.json"
    split_existed = split_directory.exists() or split_directory.is_symlink()
    capability_existed = capability_path.exists() or capability_path.is_symlink()
    artifact_existed = artifact_path.exists() or artifact_path.is_symlink()
    try:
        return _assemble(**kwargs)
    except Exception:
        if not split_existed and split_directory.exists() and not split_directory.is_symlink():
            shutil.rmtree(split_directory)
        if not capability_existed and capability_path.exists() and not capability_path.is_symlink():
            capability_path.unlink()
        if not artifact_existed and artifact_path.exists() and not artifact_path.is_symlink():
            artifact_path.unlink()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "assemble"))
    parser.add_argument("--profiler", type=Path, required=True)
    parser.add_argument("--profile-output-directory", type=Path, required=True)
    parser.add_argument("--profile-output-name", default="aq4-p3-diagnostic")
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--resident-summary", type=Path, required=True)
    parser.add_argument("--resident-raw", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--runner-command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        profiler_snapshot = PROFILER.capture(
            args.profiler, "rocprofv3 executable", require_executable=True
        )
        profiler_value = PROFILER.profiler_version(profiler_snapshot)
        runner_command = args.runner_command or []
        if not runner_command:
            raise CaptureError("runner command is required")
        runner_snapshot = PROFILER.capture(
            Path(runner_command[0]), "runner executable", require_executable=True
        )
        profiler_value["runner_executable"] = ref(runner_snapshot)
        command = profiler_command(
            profiler_snapshot,
            args.profile_output_directory,
            args.profile_output_name,
            runner_command,
        )
        if args.command == "capture":
            run_profile(command, args.profile_output_directory, args.timeout)
        elif not args.profile_output_directory.is_dir():
            raise CaptureError("assemble profile output directory is missing")
        traces = discover(args.profile_output_directory)
        artifact = assemble(
            traces=traces,
            identity_path=args.identity,
            summary_path=args.resident_summary,
            raw_path=args.resident_raw,
            profiler_value=profiler_value,
            command=command,
            output_directory=args.profile_output_directory,
            artifact_path=args.artifact,
        )
        profiler_snapshot.verify()
        runner_snapshot.verify()
        print(json.dumps({"status": artifact["status"], "promotion_eligible": False}, sort_keys=True))
        return 0
    except (CaptureError, PRODUCER.ProducerError, PROFILER.ProfileError, OSError, subprocess.SubprocessError) as error:
        print(f"AQ4 P3 diagnostic rocprof capture failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
