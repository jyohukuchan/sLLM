#!/usr/bin/env python3
"""Profile one AQ4 P2 resident case and attribute non-overlapping GPU time by family.

The live ``profile`` command launches rocprofv3 exactly once around the supplied
resident one-case command.  The ``parse`` command is offline and is used for
synthetic validation.  Neither command treats this diagnostic artifact as P2
performance evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


ARTIFACT_SCHEMA = "ullm.aq4_p2_family_exclusive_profile.v1"
MAPPING_SCHEMA = "ullm.aq4_p2_kernel_family_mapping.v1"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_TRACE_BYTES = 128 * 1024 * 1024
MAX_TRACE_ROWS = 500_000
CHUNK_BYTES = 1024 * 1024
EXPECTED_DEVICE = {
    "runtime_device_index": 1,
    "device_id": "r9700-rdna4",
    "backend": "hip",
    "name": "AMD Radeon Graphics",
    "architecture": "gfx1201",
}

# Patterns are deliberately conservative.  An unknown production kernel is
# unclassified and fails the default zero threshold instead of being guessed.
FAMILY_PATTERNS: dict[str, tuple[str, ...]] = {
    "paged_validation": (
        r"paged[_:]?kv[_:]?write",
        r"qwen35[_:]?qk[_:]?norm[_:]?rope",
        r"qwen35[_:]?q[_:]?split",
        r"paged[_:]?(?:cache|block)[_:]?valid",
    ),
    "aq4_projection": (
        r"aq4.*(?:matvec|gemm|projection|register[_:]?bm8)",
        r"(?:matvec|gemm|projection|register[_:]?bm8).*aq4",
    ),
    "attention": (
        r"paged[_:]?decode[_:]?(?:attn|attention)",
        r"paged[_:]?causal[_:]?gqa",
        r"attention[_:]?(?:read|split)",
    ),
    "recurrent": (
        r"linear[_:]?attn",
        r"gated[_:]?delta",
        r"recurrent",
        r"qkv[_:]?prepare",
    ),
    "normalization": (
        r"rmsnorm",
        r"rms[_:]?norm",
        r"silu[_:]?mul",
        r"sigmoid[_:]?mul",
        r"(?:^|[_:])rope(?:[_:](?:f32|bf16))?[_:]kernel(?:$|[_:])",
        r"(?:^|[_:])add[_:]?kernel",
    ),
    "head": (
        r"(?:^|[_:])top1(?:[_:]|$)",
        r"lm[_:]?head",
        r"argmax",
    ),
}
COMPILED_PATTERNS = {
    family: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
    for family, patterns in FAMILY_PATTERNS.items()
}

_TEST_HOOK: Callable[[], None] | None = None


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class Snapshot:
    path: Path
    identity: tuple[int, ...]
    sha256: str
    data: bytes | None

    def verify(self) -> None:
        try:
            current = self.path.lstat()
        except OSError as error:
            raise ProfileError(f"snapshot path disappeared: {self.path}: {error}") from error
        if _identity(current) != self.identity:
            raise ProfileError(f"snapshot identity changed: {self.path}")


@dataclass(frozen=True)
class KernelInterval:
    dispatch_id: str
    kernel_name: str
    start_ns: int
    end_ns: int
    family: str | None
    phase: str | None


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def capture(path: Path, label: str, maximum: int | None = None) -> Snapshot:
    try:
        before = path.lstat()
    except OSError as error:
        raise ProfileError(f"cannot stat {label}: {error}") from error
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ProfileError(f"{label} must be a single-link regular file")
    if maximum is not None and before.st_size > maximum:
        raise ProfileError(f"{label} exceeds {maximum} bytes")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProfileError(f"cannot open {label}: {error}") from error
    digest = hashlib.sha256()
    parts: list[bytes] | None = [] if maximum is not None else None
    try:
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(before):
            raise ProfileError(f"{label} identity changed while opening")
        while chunk := os.read(descriptor, CHUNK_BYTES):
            digest.update(chunk)
            if parts is not None:
                parts.append(chunk)
        after_fd = os.fstat(descriptor)
        after_path = path.lstat()
        if _identity(after_fd) != _identity(before) or _identity(after_path) != _identity(before):
            raise ProfileError(f"{label} identity changed while reading")
    finally:
        os.close(descriptor)
    return Snapshot(path, _identity(before), digest.hexdigest(), b"".join(parts) if parts is not None else None)


def parse_json(snapshot: Snapshot, label: str) -> dict[str, Any]:
    if snapshot.data is None:
        raise ProfileError(f"{label} has no bounded JSON bytes")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ProfileError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            snapshot.data,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProfileError(f"non-finite JSON in {label}: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProfileError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ProfileError(f"{label} root must be an object")
    return value


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def case_sha256(case: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(case))
    clone["case_sha256"] = None
    return canonical_sha256(clone)


def self_sha256(value: dict[str, Any], field: str) -> str:
    clone = json.loads(json.dumps(value))
    if field not in clone:
        raise ProfileError(f"self-hash field is missing: {field}")
    clone[field] = None
    return canonical_sha256(clone)


def mapping_sha256() -> str:
    return canonical_sha256(
        {"schema_version": MAPPING_SCHEMA, "families": FAMILY_PATTERNS}
    )


def classify_kernel(name: str) -> str | None:
    matches = [
        family
        for family, patterns in COMPILED_PATTERNS.items()
        if any(pattern.search(name) for pattern in patterns)
    ]
    if len(matches) > 1:
        raise ProfileError(f"kernel matches multiple families: {name}: {matches}")
    return matches[0] if matches else None


def _column(fieldnames: list[str], aliases: tuple[str, ...], label: str, required: bool) -> str | None:
    matches = [name for name in aliases if name in fieldnames]
    if len(matches) > 1:
        raise ProfileError(f"trace has duplicate aliases for {label}: {matches}")
    if not matches:
        if required:
            raise ProfileError(f"trace is missing {label} column")
        return None
    return matches[0]


def parse_trace(snapshot: Snapshot) -> tuple[list[KernelInterval], dict[str, Any]]:
    if snapshot.data is None:
        raise ProfileError("trace bytes were not captured")
    try:
        text = snapshot.data.decode("utf-8-sig")
    except UnicodeError as error:
        raise ProfileError(f"trace is not UTF-8: {error}") from error
    reader = csv.DictReader(text.splitlines())
    fieldnames = reader.fieldnames
    if not fieldnames or len(fieldnames) != len(set(fieldnames)):
        raise ProfileError("trace header is absent or contains duplicate columns")
    dispatch_column = _column(
        fieldnames, ("Dispatch_Id", "Dispatch_ID", "Index", "dispatch_id"), "dispatch id", True
    )
    kernel_column = _column(
        fieldnames, ("Kernel_Name", "KernelName", "Name", "kernel_name"), "kernel name", True
    )
    start_column = _column(
        fieldnames, ("Start_Timestamp", "BeginNs", "start_ns"), "start timestamp", True
    )
    end_column = _column(
        fieldnames, ("End_Timestamp", "EndNs", "end_ns"), "end timestamp", True
    )
    phase_column = _column(fieldnames, ("Phase", "phase"), "phase", False)
    assert dispatch_column and kernel_column and start_column and end_column
    intervals: list[KernelInterval] = []
    dispatches: set[str] = set()
    previous_start = -1
    for line_number, row in enumerate(reader, 2):
        if len(intervals) >= MAX_TRACE_ROWS:
            raise ProfileError(f"trace exceeds {MAX_TRACE_ROWS} rows")
        if None in row:
            raise ProfileError(f"trace row {line_number} has extra fields")
        dispatch_id = (row.get(dispatch_column) or "").strip()
        kernel_name = (row.get(kernel_column) or "").strip()
        start_raw = (row.get(start_column) or "").strip()
        end_raw = (row.get(end_column) or "").strip()
        if not dispatch_id or not kernel_name or not start_raw or not end_raw:
            raise ProfileError(f"trace row {line_number} is partial")
        if dispatch_id in dispatches:
            raise ProfileError(f"duplicate dispatch id: {dispatch_id}")
        dispatches.add(dispatch_id)
        try:
            start_ns = int(start_raw)
            end_ns = int(end_raw)
        except ValueError as error:
            raise ProfileError(f"trace row {line_number} has a non-integer clock") from error
        if start_ns < 0 or end_ns <= start_ns:
            raise ProfileError(f"trace row {line_number} has an invalid clock interval")
        if start_ns < previous_start:
            raise ProfileError(f"trace row {line_number} is out of timestamp order")
        previous_start = start_ns
        phase_raw = (row.get(phase_column) or "").strip().lower() if phase_column else ""
        if phase_raw and phase_raw not in {"prefill", "decode"}:
            raise ProfileError(f"trace row {line_number} has an unknown phase: {phase_raw}")
        intervals.append(
            KernelInterval(
                dispatch_id=dispatch_id,
                kernel_name=kernel_name,
                start_ns=start_ns,
                end_ns=end_ns,
                family=classify_kernel(kernel_name),
                phase=phase_raw or None,
            )
        )
    if not intervals:
        raise ProfileError("trace contains no kernel intervals")
    schema = {
        "columns": fieldnames,
        "dispatch_id": dispatch_column,
        "kernel_name": kernel_column,
        "start_timestamp": start_column,
        "end_timestamp": end_column,
        "phase": phase_column,
        "clock_unit": "nanoseconds",
    }
    return intervals, schema


def _union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    left, right = ordered[0]
    for start, end in ordered[1:]:
        if start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + right - left


def aggregate(intervals: list[KernelInterval]) -> dict[str, Any]:
    if not intervals:
        return {
            "kernel_count": 0,
            "inclusive_sum_ns": 0,
            "gpu_total_union_ns": 0,
            "inclusive_overcount_ns": 0,
            "overlap_union_ns": 0,
            "cross_family_overlap_ns": 0,
            "unclassified_ns": 0,
            "families": {
                family: {"exclusive_ns": 0, "non_overlap_ns": 0, "active_union_ns": 0}
                for family in FAMILY_PATTERNS
            },
        }
    events: dict[int, list[tuple[int, int]]] = {}
    for index, interval in enumerate(intervals):
        events.setdefault(interval.start_ns, []).append((1, index))
        events.setdefault(interval.end_ns, []).append((-1, index))
    active: set[int] = set()
    families = {
        family: {"exclusive_ns": 0, "non_overlap_ns": 0, "active_union_ns": 0}
        for family in FAMILY_PATTERNS
    }
    overlap_ns = 0
    cross_family_ns = 0
    unclassified_ns = 0
    total_union_ns = 0
    previous: int | None = None
    for timestamp in sorted(events):
        if previous is not None and timestamp > previous and active:
            duration = timestamp - previous
            total_union_ns += duration
            active_families = {intervals[index].family for index in active}
            unknown = None in active_families
            known = {family for family in active_families if family is not None}
            if len(active) > 1:
                overlap_ns += duration
            if unknown:
                unclassified_ns += duration
            elif len(known) == 1:
                family = next(iter(known))
                families[family]["exclusive_ns"] += duration
                if len(active) == 1:
                    families[family]["non_overlap_ns"] += duration
            else:
                cross_family_ns += duration
        # End events are applied before start events at the same timestamp.
        for direction, index in sorted(events[timestamp]):
            if direction < 0:
                if index not in active:
                    raise ProfileError("trace interval event order is inconsistent")
                active.remove(index)
        for direction, index in sorted(events[timestamp]):
            if direction > 0:
                active.add(index)
        previous = timestamp
    if active:
        raise ProfileError("trace interval sweep ended with active kernels")
    for family in families:
        families[family]["active_union_ns"] = _union_ns(
            (interval.start_ns, interval.end_ns)
            for interval in intervals
            if interval.family == family
        )
    inclusive = sum(interval.end_ns - interval.start_ns for interval in intervals)
    partition = sum(value["exclusive_ns"] for value in families.values())
    if partition + cross_family_ns + unclassified_ns != total_union_ns:
        raise ProfileError("exclusive timing partition does not conserve GPU union time")
    return {
        "kernel_count": len(intervals),
        "inclusive_sum_ns": inclusive,
        "gpu_total_union_ns": total_union_ns,
        "inclusive_overcount_ns": inclusive - total_union_ns,
        "overlap_union_ns": overlap_ns,
        "cross_family_overlap_ns": cross_family_ns,
        "unclassified_ns": unclassified_ns,
        "families": families,
    }


def _milliseconds(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key.endswith("_ns") and isinstance(item, int):
                result[f"{key[:-3]}_ms"] = item / 1_000_000.0
            else:
                result[key] = _milliseconds(item)
        return result
    return value


def binding(
    case_snapshot: Snapshot,
    identity_snapshot: Snapshot,
    binary_snapshot: Snapshot,
    package_snapshot: Snapshot,
    policy_snapshot: Snapshot,
) -> dict[str, Any]:
    cases = parse_json(case_snapshot, "case binding")
    identity = parse_json(identity_snapshot, "identity")
    policy = parse_json(policy_snapshot, "policy")
    rows = cases.get("cases")
    if (
        cases.get("schema_version") != "ullm.aq4_production_p2_expanded.v2"
        or cases.get("status") != "bound_one_case_smoke"
        or cases.get("case_count") != 1
        or not isinstance(rows, list)
        or len(rows) != 1
    ):
        raise ProfileError("case binding must contain exactly one P2 case")
    case = rows[0]
    if not isinstance(case, dict) or case.get("case_sha256") != case_sha256(case):
        raise ProfileError("case self-hash differs")
    if cases.get("canonical_case_sha256") != canonical_sha256(rows):
        raise ProfileError("case binding canonical hash differs")
    if case.get("prefill_requested_m") != 128 or case.get("resolved_m") != 128:
        raise ProfileError("resident profile case must bind requested/resolved M=128")
    if case.get("device") != EXPECTED_DEVICE:
        raise ProfileError("case is not bound to the exact R9700/gfx1201 device")
    if identity.get("schema_version") != "ullm.aq4_production_p2_identity.v2" or identity.get("status") != "bound":
        raise ProfileError("identity is not bound v2")
    if identity.get("identity_sha256") != self_sha256(identity, "identity_sha256"):
        raise ProfileError("identity self-hash differs")
    resident = identity.get("resident_driver_identity")
    if not isinstance(resident, dict) or resident.get("runtime_device") != EXPECTED_DEVICE:
        raise ProfileError("resident identity device differs")
    if resident.get("binary_sha256") != binary_snapshot.sha256:
        raise ProfileError("resident binary hash differs from identity")
    if resident.get("package_manifest_sha256") != package_snapshot.sha256:
        raise ProfileError("package manifest hash differs from identity")
    hash_binding = identity.get("hash_binding")
    if (
        identity.get("expanded_manifest_sha256") != case_snapshot.sha256
        or not isinstance(hash_binding, dict)
        or hash_binding.get("bound_case_manifest_sha256") != case_snapshot.sha256
    ):
        raise ProfileError("identity case binding differs")
    for field in (
        "worker_binary_sha256",
        "package_manifest_sha256",
        "package_content_sha256",
        "served_model_manifest_sha256",
    ):
        if hash_binding.get(field) != resident.get(field):
            raise ProfileError(f"identity {field} binding differs")
    if policy != {"schema_version": "ullm.aq4_production_p2_threshold_policy.v1", "status": "bound"}:
        raise ProfileError("threshold policy differs")
    return {
        "case": {
            "case_id": case.get("case_id"),
            "case_sha256": case["case_sha256"],
            "case_binding_sha256": case_snapshot.sha256,
            "prefill_requested_m": 128,
            "resolved_m": 128,
        },
        "identity": {
            "identity_file_sha256": identity_snapshot.sha256,
            "identity_sha256": identity.get("identity_sha256"),
            "model_id": resident.get("model_id"),
            "model_revision": resident.get("model_revision"),
            "worker_binary_sha256": resident.get("worker_binary_sha256"),
            "served_model_manifest_sha256": resident.get("served_model_manifest_sha256"),
            "guard_set_sha256": resident.get("guard_set_sha256"),
        },
        "device": EXPECTED_DEVICE,
        "resident_binary_sha256": binary_snapshot.sha256,
        "package_manifest_sha256": package_snapshot.sha256,
        "package_content_sha256": resident.get("package_content_sha256"),
        "policy_sha256": policy_snapshot.sha256,
    }


def profiler_version(profiler: Path) -> dict[str, Any]:
    result = subprocess.run(
        [os.fspath(profiler), "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
    )
    if result.returncode != 0 or len(result.stdout) > 64 * 1024:
        raise ProfileError("rocprofv3 version query failed")
    text = result.stdout.decode("utf-8", errors="strict").strip()
    match = re.search(r"version:\s*([^\s]+)", text)
    rocm = re.search(r"rocm_version:\s*([^\s]+)", text)
    if match is None:
        raise ProfileError("rocprofv3 version output schema differs")
    return {
        "tool": "rocprofv3",
        "path": os.fspath(profiler.resolve(strict=True)),
        "version": match.group(1),
        "rocm_version": rocm.group(1) if rocm else None,
        "version_output_sha256": hashlib.sha256(result.stdout).hexdigest(),
    }


def profiler_command(
    profiler: Path, output_directory: Path, output_name: str, resident_command: list[str]
) -> list[str]:
    if not resident_command or any(not item for item in resident_command):
        raise ProfileError("resident command must not be empty")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", output_name):
        raise ProfileError("profiler output name is unsafe")
    return [
        os.fspath(profiler),
        "--kernel-trace",
        "--output-format",
        "csv",
        "--output-directory",
        os.fspath(output_directory),
        "--output-file",
        output_name,
        "--",
        *resident_command,
    ]


def run_profile(command: list[str], output_directory: Path, timeout: float) -> Path:
    if output_directory.exists() or output_directory.is_symlink():
        raise ProfileError(f"profile output directory already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    stdout_path = output_directory / "profiler.stdout"
    stderr_path = output_directory / "profiler.stderr"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        result = subprocess.run(command, check=False, stdout=stdout, stderr=stderr, timeout=timeout)
    if result.returncode != 0:
        raise ProfileError(f"rocprofv3 profile command failed with exit {result.returncode}")
    candidates = sorted(output_directory.rglob("*_kernel_trace.csv"))
    if len(candidates) != 1:
        raise ProfileError(f"expected exactly one rocprofv3 kernel trace CSV, got {len(candidates)}")
    return candidates[0]


def build_artifact(
    *,
    trace_snapshot: Snapshot,
    trace_schema: dict[str, Any],
    intervals: list[KernelInterval],
    binding_value: dict[str, Any],
    profiler_value: dict[str, Any],
    command: list[str],
    maximum_unclassified_fraction: float,
) -> dict[str, Any]:
    if not 0.0 <= maximum_unclassified_fraction <= 1.0:
        raise ProfileError("unclassified threshold must be in [0, 1]")
    total = aggregate(intervals)
    phases = {
        phase: aggregate([interval for interval in intervals if interval.phase == phase])
        for phase in ("prefill", "decode")
    }
    phase_unknown = aggregate([interval for interval in intervals if interval.phase is None])
    union = total["gpu_total_union_ns"]
    unknown_fraction = total["unclassified_ns"] / union if union else 0.0
    unknown_names = sorted({interval.kernel_name for interval in intervals if interval.family is None})
    mapping_complete = unknown_fraction <= maximum_unclassified_fraction
    phase_complete = phase_unknown["kernel_count"] == 0
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "status": "profiled_diagnostic",
        "measurement_eligible": False,
        "promotion": False,
        "binding": binding_value,
        "profiler": {**profiler_value, "command": command, "subprocess_profile_runs": 1},
        "trace": {
            "sha256": trace_snapshot.sha256,
            "bytes": trace_snapshot.identity[4],
            "schema": trace_schema,
            "kernel_count": len(intervals),
        },
        "mapping": {
            "schema_version": MAPPING_SCHEMA,
            "sha256": mapping_sha256(),
            "maximum_unclassified_fraction": maximum_unclassified_fraction,
            "observed_unclassified_fraction": unknown_fraction,
            "unknown_kernel_names": unknown_names,
            "complete": mapping_complete,
        },
        "timing_ns": {
            "total": total,
            "prefill": phases["prefill"],
            "decode": phases["decode"],
            "unclassified_phase": phase_unknown,
        },
        "timing_ms": _milliseconds(
            {
                "total": total,
                "prefill": phases["prefill"],
                "decode": phases["decode"],
                "unclassified_phase": phase_unknown,
            }
        ),
        "eligibility_blockers": [
            "family-exclusive profiling is diagnostic and separate from 2 warmup + 10 measured performance evidence",
            *([] if mapping_complete else ["unclassified GPU time exceeds the configured threshold"]),
            *([] if phase_complete else ["trace lacks exact prefill/decode attribution"]),
        ],
        "schedule_separation": {
            "warmup_runs": 2,
            "measured_runs": 10,
            "profile_aggregation_used_for_performance": False,
            "inclusive_kernel_sum_used_as_gpu_total": False,
        },
    }


def write_artifact(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ProfileError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("parse", "profile"))
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--profile-output-directory", type=Path)
    parser.add_argument("--profile-output-name", default="aq4-p2-family-exclusive")
    parser.add_argument("--profiler", type=Path, default=Path("/opt/rocm/bin/rocprofv3"))
    parser.add_argument("--case-binding", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--resident-binary", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--maximum-unclassified-fraction", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--resident-command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        snapshots = [
            capture(args.case_binding, "case binding", MAX_JSON_BYTES),
            capture(args.identity, "identity", MAX_JSON_BYTES),
            capture(args.resident_binary, "resident binary"),
            capture(args.package_manifest, "package manifest"),
            capture(args.policy, "policy", MAX_JSON_BYTES),
        ]
        binding_value = binding(*snapshots)
        version = profiler_version(args.profiler)
        command = profiler_command(
            args.profiler,
            args.profile_output_directory or Path("profile-output"),
            args.profile_output_name,
            args.resident_command or [],
        )
        if args.command == "profile":
            if args.trace is not None or args.profile_output_directory is None:
                raise ProfileError("profile requires --profile-output-directory and forbids --trace")
            trace_path = run_profile(command, args.profile_output_directory, args.timeout)
        else:
            if args.trace is None:
                raise ProfileError("parse requires --trace")
            trace_path = args.trace
        trace_snapshot = capture(trace_path, "kernel trace", MAX_TRACE_BYTES)
        intervals, trace_schema = parse_trace(trace_snapshot)
        artifact = build_artifact(
            trace_snapshot=trace_snapshot,
            trace_schema=trace_schema,
            intervals=intervals,
            binding_value=binding_value,
            profiler_value=version,
            command=command,
            maximum_unclassified_fraction=args.maximum_unclassified_fraction,
        )
        if not artifact["mapping"]["complete"]:
            raise ProfileError("unclassified GPU time exceeds configured threshold")
        if _TEST_HOOK is not None:
            _TEST_HOOK()
        for snapshot in [*snapshots, trace_snapshot]:
            snapshot.verify()
        write_artifact(args.artifact, artifact)
        print(json.dumps({"status": artifact["status"], "measurement_eligible": False}, sort_keys=True))
        return 0
    except (OSError, ProfileError, subprocess.SubprocessError, ValueError) as error:
        print(f"AQ4 P2 family-exclusive profile failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
