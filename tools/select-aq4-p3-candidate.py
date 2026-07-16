#!/usr/bin/env python3
"""Select an AQ4 P3 optimization candidate from hash-bound P2 evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUALIFICATION = load_tool("aq4_p3_upstream_qualification_for_selector", ROOT / "tools/aq4_p3_upstream_qualification.py")


RAW_SCHEMA = "ullm.aq4_p2_candidate_selection_raw.v1"
PROFILE_SCHEMA = "ullm.aq4_p2_family_exclusive_profile.v1"
OUTPUT_SCHEMA = "ullm.aq4_p3_candidate_selection.v1"
POLICY_VERSION = "ullm.aq4_p3_candidate_selection_policy.v1"
MAX_EVIDENCE_BYTES = 32 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REPRESENTATIVE_PROMPTS = 7
MIN_ABOVE_NOISE = 4

CANDIDATES: dict[str, dict[str, Any]] = {
    "sequence-output-direct-v1": {
        "family": "attention_recurrent",
        "requires_d2h_count": False,
        "requires_stream_sync_count": False,
        "candidate_a": True,
    },
    "paged-kv-table-validation-v1": {
        "family": "paged_validation",
        "requires_d2h_count": True,
        "requires_stream_sync_count": True,
    },
    "aq4-register-bm8-v1": {
        "family": "aq4_projection",
        "requires_d2h_count": False,
        "requires_stream_sync_count": False,
    },
    "chunk-execution-v1": {
        "family": "attention_recurrent",
        "requires_d2h_count": False,
        "requires_stream_sync_count": False,
    },
    "projection-norm-activation-fusion-v1": {
        "family": "normalization",
        "requires_d2h_count": False,
        "requires_stream_sync_count": False,
    },
}

RAW_ROOT_FIELDS = {
    "schema_version",
    "status",
    "measurement_eligible",
    "smoke_only",
    "promotion_eligible",
    "promotion_ineligibility_reason",
    "evidence_sha256",
    "upstream_qualification",
    "identity",
    "capabilities",
    "representative_prompt_count",
    "measurements",
    "full_model_pairs",
}
IDENTITY_FIELDS = {
    "identity_sha256",
    "case_manifest_sha256",
    "binary_sha256",
    "package_content_sha256",
}
CAPABILITY_FIELDS = {
    "family_exclusive_timing",
    "d2h_count",
    "stream_sync_count",
}
CANDIDATE_A_CAPABILITY_FIELDS = {
    "direct_sequence_output",
    "d2d_bytes",
    "d2d_copy_count",
    "launch_count",
    "component_latency",
    "full_model_latency",
    "workspace_peak_vram",
    "fallback_reasons",
    "alias_size_admission_safety",
    "direct_copy_fidelity",
}
QUALIFICATION_REF_FIELDS = {
    "path", "sha256", "qualification_sha256", "status", "promotion_eligible", "reason"
}
MEASUREMENT_FIELDS = {
    "candidate_id",
    "family",
    "prompt_id",
    "case_sha256",
    "identity_sha256",
    "resolved_m",
    "baseline_p50_ms",
    "baseline_cv",
    "ci95_halfwidth_ms",
    "recoverable_family_exclusive_ms",
    "d2h_count",
    "d2h_time_ms",
    "stream_sync_count",
    "stream_sync_time_ms",
}
PAIR_FIELDS = {
    "candidate_id",
    "pair_id",
    "case_sha256",
    "identity_sha256",
    "baseline_ms",
    "candidate_ms",
}
CANDIDATE_A_MEASUREMENT_FIELDS = MEASUREMENT_FIELDS | {
    "baseline_d2d_bytes",
    "candidate_d2d_bytes",
    "baseline_d2d_copy_count",
    "candidate_d2d_copy_count",
    "baseline_launch_count",
    "candidate_launch_count",
    "baseline_component_p50_ms",
    "baseline_component_p95_ms",
    "candidate_component_p50_ms",
    "candidate_component_p95_ms",
    "baseline_full_model_p95_ms",
    "candidate_full_model_p50_ms",
    "candidate_full_model_p95_ms",
    "baseline_workspace_bytes",
    "candidate_workspace_bytes",
    "baseline_peak_vram_bytes",
    "candidate_peak_vram_bytes",
    "baseline_fallback_count",
    "candidate_fallback_count",
    "baseline_fallback_reasons",
    "candidate_fallback_reasons",
    "direct_alias_safe",
    "direct_size_safe",
    "direct_admission_safe",
    "direct_fidelity_binding_sha256",
    "copy_fidelity_binding_sha256",
}
CANDIDATE_A_PAIR_FIELDS = PAIR_FIELDS | {
    "baseline_d2d_bytes",
    "candidate_d2d_bytes",
    "baseline_d2d_copy_count",
    "candidate_d2d_copy_count",
    "baseline_launch_count",
    "candidate_launch_count",
    "baseline_workspace_bytes",
    "candidate_workspace_bytes",
    "baseline_peak_vram_bytes",
    "candidate_peak_vram_bytes",
    "baseline_fallback_count",
    "candidate_fallback_count",
    "baseline_fallback_reasons",
    "candidate_fallback_reasons",
    "direct_alias_safe",
    "direct_size_safe",
    "direct_admission_safe",
    "direct_fidelity_binding_sha256",
    "copy_fidelity_binding_sha256",
}
PROFILE_ROOT_FIELDS = {
    "schema_version",
    "status",
    "measurement_eligible",
    "promotion",
    "binding",
    "profiler",
    "trace",
    "mapping",
    "timing_ns",
    "timing_ms",
    "eligibility_blockers",
    "schedule_separation",
}
PROFILE_BINDING_FIELDS = {
    "case",
    "identity",
    "device",
    "resident_binary_sha256",
    "package_manifest_sha256",
    "package_content_sha256",
    "policy_sha256",
    "served_model_manifest_sha256",
    "worker_binary_sha256",
}
PROFILE_CASE_FIELDS = {
    "case_id",
    "case_sha256",
    "case_binding_sha256",
    "prefill_requested_m",
    "resolved_m",
}
PROFILE_IDENTITY_FIELDS = {
    "identity_file_sha256",
    "identity_sha256",
    "model_id",
    "model_revision",
    "worker_binary_sha256",
    "served_model_manifest_sha256",
    "guard_set_sha256",
    "build_git_commit",
    "protocol",
}
PROFILE_DEVICE_FIELDS = {
    "runtime_device_index",
    "device_id",
    "backend",
    "name",
    "architecture",
}
PROFILE_PROFILER_FIELDS = {
    "tool",
    "path",
    "executable_sha256",
    "version",
    "rocm_version",
    "version_output_sha256",
    "command",
    "subprocess_profile_runs",
}
PROFILE_TRACE_FIELDS = {"sha256", "bytes", "schema", "kernel_count"}
PROFILE_TRACE_SCHEMA_FIELDS = {
    "columns",
    "dispatch_id",
    "kernel_name",
    "start_timestamp",
    "end_timestamp",
    "phase",
    "clock_unit",
}
PROFILE_MAPPING_FIELDS = {
    "schema_version",
    "sha256",
    "maximum_unclassified_fraction",
    "observed_unclassified_fraction",
    "unknown_kernel_names",
    "complete",
}
PROFILE_TIMING_ROOT_FIELDS = {"total", "prefill", "decode", "unclassified_phase"}
PROFILE_TIMING_NS_FIELDS = {
    "kernel_count",
    "inclusive_sum_ns",
    "gpu_total_union_ns",
    "inclusive_overcount_ns",
    "overlap_union_ns",
    "cross_family_overlap_ns",
    "unclassified_ns",
    "families",
}
PROFILE_TIMING_MS_FIELDS = {
    "kernel_count",
    "inclusive_sum_ms",
    "gpu_total_union_ms",
    "inclusive_overcount_ms",
    "overlap_union_ms",
    "cross_family_overlap_ms",
    "unclassified_ms",
    "families",
}
PROFILE_FAMILY_NS_FIELDS = {"exclusive_ns", "non_overlap_ns", "active_union_ns"}
PROFILE_FAMILY_MS_FIELDS = {"exclusive_ms", "non_overlap_ms", "active_union_ms"}
PROFILE_SCHEDULE_FIELDS = {
    "warmup_runs",
    "measured_runs",
    "profile_aggregation_used_for_performance",
    "inclusive_kernel_sum_used_as_gpu_total",
}

# Two-sided Student t, 97.5th percentile. P2 uses at most 30 paired runs.
T_CRITICAL_975 = (
    0.0,
    12.7062047364,
    4.30265272975,
    3.18244630528,
    2.7764451052,
    2.57058183564,
    2.44691184879,
    2.36462425101,
    2.30600413503,
    2.26215716285,
    2.22813885196,
    2.20098516008,
    2.17881282966,
    2.16036865646,
    2.14478668792,
    2.13144954556,
    2.11990529922,
    2.10981557783,
    2.10092204024,
    2.09302405441,
    2.08596344727,
    2.07961384473,
    2.0738730679,
    2.06865761042,
    2.06389856163,
    2.05953855275,
    2.05552943864,
    2.05183051648,
    2.0484071418,
    2.04522964213,
)


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class Snapshot:
    path: Path
    identity: tuple[int, ...]
    sha256: str
    data: bytes

    def verify(self) -> None:
        try:
            current = self.path.lstat()
        except OSError as error:
            raise SelectionError(f"evidence disappeared: {self.path}: {error}") from error
        if file_identity(current) != self.identity:
            raise SelectionError(f"evidence identity changed: {self.path}")


@dataclass(frozen=True)
class RawSource:
    semantic_sha256: str
    identity: dict[str, str]
    capabilities: dict[str, bool]
    measurements: tuple[dict[str, Any], ...]
    pairs: tuple[dict[str, Any], ...]
    promotion_eligible: bool
    upstream_qualification: dict[str, Any]


def file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def capture(path: Path) -> Snapshot:
    path = path.absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except OSError as error:
            raise SelectionError(f"cannot inspect evidence path: {error}") from error
        if stat.S_ISLNK(info.st_mode):
            raise SelectionError(f"evidence path contains a symlink: {current}")
    path = path.resolve(strict=True)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise SelectionError(f"evidence must be a regular file: {path}")
    if before.st_size > MAX_EVIDENCE_BYTES:
        raise SelectionError(f"evidence exceeds {MAX_EVIDENCE_BYTES} bytes: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if file_identity(opened) != file_identity(before):
            raise SelectionError(f"evidence identity changed while opening: {path}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > MAX_EVIDENCE_BYTES:
                raise SelectionError(f"evidence exceeds {MAX_EVIDENCE_BYTES} bytes: {path}")
            chunks.append(chunk)
            digest.update(chunk)
        after_fd = os.fstat(descriptor)
        after_path = path.lstat()
        if (
            file_identity(after_fd) != file_identity(before)
            or file_identity(after_path) != file_identity(before)
        ):
            raise SelectionError(f"evidence identity changed while reading: {path}")
    finally:
        os.close(descriptor)
    return Snapshot(path, file_identity(before), digest.hexdigest(), b"".join(chunks))


def parse_json(snapshot: Snapshot) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SelectionError(f"duplicate JSON key in {snapshot.path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            snapshot.data,
            object_pairs_hook=object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SelectionError(f"non-finite JSON number in {snapshot.path}: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SelectionError(f"invalid JSON evidence {snapshot.path}: {error}") from error
    if not isinstance(value, dict):
        raise SelectionError(f"evidence root must be an object: {snapshot.path}")
    ensure_finite_tree(value, f"evidence {snapshot.path}")
    return value


def ensure_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SelectionError(f"{label} contains a non-finite number")
    elif isinstance(value, dict):
        for key, child in value.items():
            ensure_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            ensure_finite_tree(child, f"{label}[{index}]")


def exact_fields(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise SelectionError(f"{label} fields differ: missing={missing}, unknown={unknown}")


def require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SelectionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_bool(value: Any, expected: bool, label: str) -> None:
    if value is not expected:
        raise SelectionError(f"{label} must be {str(expected).lower()}")


def require_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise SelectionError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise SelectionError(f"{label} must be a finite number")
    if positive and result <= 0.0:
        raise SelectionError(f"{label} must be positive")
    if not positive and result < 0.0:
        raise SelectionError(f"{label} must be non-negative")
    return result


def require_count(value: Any, label: str, *, allow_none: bool = False) -> int | None:
    if allow_none and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SelectionError(f"{label} must be a non-negative integer")
    return value


def require_integer_median(value: Any, label: str) -> int | float:
    """Accept only an exact median attainable from an even integer sample.

    Ten integer observations can have an integer or half-integer median.  Keeping
    the half preserves the direction of one-byte/resource differences instead of
    silently truncating them before the non-regression gates run.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionError(
            f"{label} must be a non-negative integer or half-integer median"
        )
    if isinstance(value, int):
        if value < 0:
            raise SelectionError(
                f"{label} must be a non-negative integer or half-integer median"
            )
        return value
    number = value
    if (
        not math.isfinite(number)
        or number < 0.0
        or number > float(1 << 52)
        or not (number * 2.0).is_integer()
    ):
        raise SelectionError(
            f"{label} must be a non-negative integer or half-integer median"
        )
    return value


def require_reason_list(value: Any, label: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        raise SelectionError(f"{label} must be a string array")
    if len(value) != len(set(value)):
        raise SelectionError(f"{label} contains duplicate reasons")
    return sorted(value)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def normalized_raw(value: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(value, allow_nan=False))
    clone["evidence_sha256"] = None
    clone["measurements"] = sorted(
        clone["measurements"],
        key=lambda row: (
            row.get("candidate_id", ""),
            row.get("prompt_id", ""),
            row.get("case_sha256", ""),
            row.get("resolved_m", -1),
        ),
    )
    clone["full_model_pairs"] = sorted(
        clone["full_model_pairs"],
        key=lambda row: (
            row.get("candidate_id", ""),
            row.get("pair_id", ""),
            row.get("case_sha256", ""),
        ),
    )
    return clone


def semantic_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(normalized_raw(value))).hexdigest()


def _validate_candidate_a_measurement(
    row: dict[str, Any], label: str, parsed: dict[str, Any]
) -> None:
    positive_fields = {
        "baseline_component_p50_ms",
        "baseline_component_p95_ms",
        "candidate_component_p50_ms",
        "candidate_component_p95_ms",
        "baseline_full_model_p95_ms",
        "candidate_full_model_p50_ms",
        "candidate_full_model_p95_ms",
    }
    count_fields = {
        "baseline_d2d_copy_count",
        "candidate_d2d_copy_count",
        "baseline_launch_count",
        "candidate_launch_count",
        "baseline_fallback_count",
        "candidate_fallback_count",
    }
    integer_median_fields = {
        "baseline_d2d_bytes",
        "candidate_d2d_bytes",
        "baseline_workspace_bytes",
        "candidate_workspace_bytes",
        "baseline_peak_vram_bytes",
        "candidate_peak_vram_bytes",
    }
    for field in sorted(positive_fields):
        parsed[field] = require_number(row[field], f"{label}.{field}", positive=True)
    for field in sorted(count_fields):
        parsed[field] = require_count(row[field], f"{label}.{field}")
    for field in sorted(integer_median_fields):
        parsed[field] = require_integer_median(row[field], f"{label}.{field}")
    for field in ("baseline_fallback_reasons", "candidate_fallback_reasons"):
        parsed[field] = require_reason_list(row[field], f"{label}.{field}")
    for field in ("direct_alias_safe", "direct_size_safe", "direct_admission_safe"):
        if not isinstance(row[field], bool):
            raise SelectionError(f"{label}.{field} must be boolean")
        parsed[field] = row[field]
    for field in ("direct_fidelity_binding_sha256", "copy_fidelity_binding_sha256"):
        parsed[field] = require_digest(row[field], f"{label}.{field}")
    if parsed["baseline_full_model_p95_ms"] < parsed["baseline_p50_ms"]:
        raise SelectionError(f"{label}.baseline_full_model_p95_ms is below baseline p50")
    if parsed["candidate_full_model_p95_ms"] < parsed["candidate_full_model_p50_ms"]:
        raise SelectionError(f"{label}.candidate_full_model_p95_ms is below candidate p50")
    if parsed["baseline_component_p95_ms"] < parsed["baseline_component_p50_ms"]:
        raise SelectionError(f"{label}.baseline_component_p95_ms is below baseline component p50")
    if parsed["candidate_component_p95_ms"] < parsed["candidate_component_p50_ms"]:
        raise SelectionError(f"{label}.candidate_component_p95_ms is below candidate component p50")


def _validate_candidate_a_pair(
    row: dict[str, Any], label: str, parsed: dict[str, Any]
) -> None:
    count_fields = {
        "baseline_d2d_copy_count",
        "candidate_d2d_copy_count",
        "baseline_launch_count",
        "candidate_launch_count",
        "baseline_fallback_count",
        "candidate_fallback_count",
    }
    byte_fields = {
        "baseline_d2d_bytes",
        "candidate_d2d_bytes",
        "baseline_workspace_bytes",
        "candidate_workspace_bytes",
        "baseline_peak_vram_bytes",
        "candidate_peak_vram_bytes",
    }
    for field in sorted(count_fields):
        parsed[field] = require_count(row[field], f"{label}.{field}")
    for field in sorted(byte_fields):
        parsed[field] = require_count(row[field], f"{label}.{field}")
    for field in ("baseline_fallback_reasons", "candidate_fallback_reasons"):
        parsed[field] = require_reason_list(row[field], f"{label}.{field}")
    for field in ("direct_alias_safe", "direct_size_safe", "direct_admission_safe"):
        if not isinstance(row[field], bool):
            raise SelectionError(f"{label}.{field} must be boolean")
        parsed[field] = row[field]
    for field in ("direct_fidelity_binding_sha256", "copy_fidelity_binding_sha256"):
        parsed[field] = require_digest(row[field], f"{label}.{field}")


def validate_raw(value: dict[str, Any]) -> RawSource:
    exact_fields(value, RAW_ROOT_FIELDS, "raw evidence")
    if value["schema_version"] != RAW_SCHEMA or value["status"] not in {"complete", "one_case_diagnostic"}:
        raise SelectionError("raw evidence schema/status differs")
    promotion = value["status"] == "complete"
    require_bool(value["measurement_eligible"], promotion, "raw measurement_eligible")
    require_bool(value["smoke_only"], not promotion, "raw smoke_only")
    require_bool(value["promotion_eligible"], promotion, "raw promotion_eligible")
    reason = value["promotion_ineligibility_reason"]
    if promotion and reason is not None:
        raise SelectionError("promotion raw must not contain an ineligibility reason")
    if not promotion and reason != QUALIFICATION.REASON:
        raise SelectionError("diagnostic raw ineligibility reason differs")
    qualification_ref = value["upstream_qualification"]
    if not isinstance(qualification_ref, dict):
        raise SelectionError("raw upstream qualification must be an object")
    exact_fields(qualification_ref, QUALIFICATION_REF_FIELDS, "raw upstream qualification")
    qualification_path = Path(qualification_ref["path"])
    qualification_snapshot = capture(qualification_path)
    if qualification_snapshot.sha256 != require_digest(qualification_ref["sha256"], "raw upstream qualification file SHA-256"):
        raise SelectionError("raw upstream qualification file SHA-256 differs")
    try:
        qualification_value = parse_json(qualification_snapshot)
        qualification_result = QUALIFICATION.validate(qualification_value)
    except QUALIFICATION.QualificationError as error:
        raise SelectionError(f"raw upstream qualification differs: {error}") from error
    expected_qualification_status = "valid_qualified_go" if promotion else "valid_rejected_no_go"
    if qualification_result["status"] != expected_qualification_status:
        raise SelectionError("raw upstream qualification variant differs")
    expected_projection = {
        "path": str(qualification_snapshot.path), "sha256": qualification_snapshot.sha256,
        "qualification_sha256": qualification_result["qualification_sha256"],
        "status": qualification_value["status"],
        "promotion_eligible": qualification_result["promotion_eligible"],
        "reason": qualification_result["reason"],
    }
    if qualification_ref != expected_projection:
        raise SelectionError("raw upstream qualification projection differs")
    if (
        type(value["representative_prompt_count"]) is not int
        or value["representative_prompt_count"] != (REPRESENTATIVE_PROMPTS if promotion else 1)
    ):
        raise SelectionError(
            f"raw representative_prompt_count must be {REPRESENTATIVE_PROMPTS if promotion else 1}"
        )

    identity = value["identity"]
    capabilities = value["capabilities"]
    if not isinstance(identity, dict) or not isinstance(capabilities, dict):
        raise SelectionError("raw identity/capabilities must be objects")
    exact_fields(identity, IDENTITY_FIELDS, "raw identity")
    # Fixed candidate fixtures retain the v1 capability contract.  Candidate A is
    # deliberately conditional so old E/N evidence cannot silently claim direct
    # sequence-output safety metrics it never captured.
    capability_fields = set(CAPABILITY_FIELDS)
    measurement_values = value.get("measurements", [])
    pair_values = value.get("full_model_pairs", [])
    if not isinstance(measurement_values, list):
        measurement_values = []
    if not isinstance(pair_values, list):
        pair_values = []
    has_candidate_a = any(
        isinstance(row, dict) and row.get("candidate_id") == "sequence-output-direct-v1"
        for row in measurement_values
    ) or any(
        isinstance(row, dict) and row.get("candidate_id") == "sequence-output-direct-v1"
        for row in pair_values
    )
    if has_candidate_a:
        capability_fields |= CANDIDATE_A_CAPABILITY_FIELDS
    exact_fields(capabilities, capability_fields, "raw capabilities")
    identity_value = {
        field: require_digest(identity[field], f"raw identity.{field}")
        for field in sorted(IDENTITY_FIELDS)
    }
    capability_value: dict[str, bool] = {}
    for field in sorted(capability_fields):
        if not isinstance(capabilities[field], bool):
            raise SelectionError(f"raw capabilities.{field} must be boolean")
        capability_value[field] = capabilities[field]

    measurements = value["measurements"]
    pairs = value["full_model_pairs"]
    if not isinstance(measurements, list) or not isinstance(pairs, list):
        raise SelectionError("raw measurements/full_model_pairs must be arrays")
    parsed_measurements: list[dict[str, Any]] = []
    seen_measurements: set[tuple[str, str]] = set()
    for index, row in enumerate(measurements):
        label = f"raw measurements[{index}]"
        if not isinstance(row, dict):
            raise SelectionError(f"{label} must be an object")
        candidate_id = row.get("candidate_id")
        row_fields = CANDIDATE_A_MEASUREMENT_FIELDS if candidate_id == "sequence-output-direct-v1" else MEASUREMENT_FIELDS
        exact_fields(row, row_fields, label)
        if candidate_id not in CANDIDATES:
            raise SelectionError(f"{label}.candidate_id is unknown: {candidate_id}")
        if row["family"] != CANDIDATES[candidate_id]["family"]:
            raise SelectionError(f"{label}.family differs from candidate policy")
        prompt_id = row["prompt_id"]
        if not isinstance(prompt_id, str) or not prompt_id:
            raise SelectionError(f"{label}.prompt_id must be a non-empty string")
        key = (candidate_id, prompt_id)
        if key in seen_measurements:
            raise SelectionError(f"duplicate candidate/prompt measurement: {key}")
        seen_measurements.add(key)
        if row["identity_sha256"] != identity_value["identity_sha256"]:
            raise SelectionError(f"{label}.identity_sha256 differs from raw identity")
        case_sha = require_digest(row["case_sha256"], f"{label}.case_sha256")
        resolved_m = require_count(row["resolved_m"], f"{label}.resolved_m")
        assert resolved_m is not None
        if resolved_m <= 0:
            raise SelectionError(f"{label}.resolved_m must be positive")
        parsed_row = {
                "candidate_id": candidate_id,
                "family": row["family"],
                "prompt_id": prompt_id,
                "case_sha256": case_sha,
                "identity_sha256": row["identity_sha256"],
                "resolved_m": resolved_m,
                "baseline_p50_ms": require_number(
                    row["baseline_p50_ms"], f"{label}.baseline_p50_ms", positive=True
                ),
                "baseline_cv": require_number(row["baseline_cv"], f"{label}.baseline_cv"),
                "ci95_halfwidth_ms": require_number(
                    row["ci95_halfwidth_ms"], f"{label}.ci95_halfwidth_ms"
                ),
                "recoverable_family_exclusive_ms": require_number(
                    row["recoverable_family_exclusive_ms"],
                    f"{label}.recoverable_family_exclusive_ms",
                ),
                "d2h_count": require_count(
                    row["d2h_count"], f"{label}.d2h_count", allow_none=True
                ),
                "d2h_time_ms": None
                if row["d2h_time_ms"] is None
                else require_number(row["d2h_time_ms"], f"{label}.d2h_time_ms"),
                "stream_sync_count": require_count(
                    row["stream_sync_count"],
                    f"{label}.stream_sync_count",
                    allow_none=True,
                ),
                "stream_sync_time_ms": None
                if row["stream_sync_time_ms"] is None
                else require_number(
                    row["stream_sync_time_ms"], f"{label}.stream_sync_time_ms"
                ),
            }
        if CANDIDATES[candidate_id].get("candidate_a"):
            _validate_candidate_a_measurement(row, label, parsed_row)
        parsed_measurements.append(parsed_row)

    parsed_pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(pairs):
        label = f"raw full_model_pairs[{index}]"
        if not isinstance(row, dict):
            raise SelectionError(f"{label} must be an object")
        candidate_id = row.get("candidate_id")
        row_fields = CANDIDATE_A_PAIR_FIELDS if candidate_id == "sequence-output-direct-v1" else PAIR_FIELDS
        exact_fields(row, row_fields, label)
        if candidate_id not in CANDIDATES:
            raise SelectionError(f"{label}.candidate_id is unknown: {candidate_id}")
        pair_id = row["pair_id"]
        if not isinstance(pair_id, str) or not pair_id:
            raise SelectionError(f"{label}.pair_id must be a non-empty string")
        key = (candidate_id, pair_id)
        if key in seen_pairs:
            raise SelectionError(f"duplicate candidate/pair measurement: {key}")
        seen_pairs.add(key)
        if row["identity_sha256"] != identity_value["identity_sha256"]:
            raise SelectionError(f"{label}.identity_sha256 differs from raw identity")
        parsed_pair = {
                "candidate_id": candidate_id,
                "pair_id": pair_id,
                "case_sha256": require_digest(row["case_sha256"], f"{label}.case_sha256"),
                "identity_sha256": row["identity_sha256"],
                "baseline_ms": require_number(
                    row["baseline_ms"], f"{label}.baseline_ms", positive=True
                ),
                "candidate_ms": require_number(
                    row["candidate_ms"], f"{label}.candidate_ms", positive=True
                ),
            }
        if CANDIDATES[candidate_id].get("candidate_a"):
            _validate_candidate_a_pair(row, label, parsed_pair)
        parsed_pairs.append(parsed_pair)

    declared_sha = require_digest(value["evidence_sha256"], "raw evidence_sha256")
    calculated_sha = semantic_sha256(value)
    if declared_sha != calculated_sha:
        raise SelectionError("raw evidence semantic SHA-256 differs")
    return RawSource(
        semantic_sha256=calculated_sha,
        identity=identity_value,
        capabilities=capability_value,
        measurements=tuple(parsed_measurements) if promotion else (),
        pairs=tuple(parsed_pairs) if promotion else (),
        promotion_eligible=promotion,
        upstream_qualification=expected_projection,
    )


def _profile_nonnegative_number(value: Any, label: str, *, integer: bool = False) -> None:
    if integer:
        if type(value) is not int or value < 0:
            raise SelectionError(f"{label} must be a non-negative integer")
        return
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise SelectionError(f"{label} must be a non-negative finite float")


def _profile_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise SelectionError(f"{label} must be a {'string' if allow_empty else 'non-empty string'}")
    return value


def _validate_profile_timing(value: Any, label: str, *, milliseconds: bool) -> None:
    if not isinstance(value, dict):
        raise SelectionError(f"{label} must be an object")
    exact_fields(value, PROFILE_TIMING_ROOT_FIELDS, label)
    aggregate_fields = PROFILE_TIMING_MS_FIELDS if milliseconds else PROFILE_TIMING_NS_FIELDS
    family_fields = PROFILE_FAMILY_MS_FIELDS if milliseconds else PROFILE_FAMILY_NS_FIELDS
    suffix = "ms" if milliseconds else "ns"
    for phase in sorted(PROFILE_TIMING_ROOT_FIELDS):
        aggregate = value[phase]
        if not isinstance(aggregate, dict):
            raise SelectionError(f"{label}.{phase} must be an object")
        exact_fields(aggregate, aggregate_fields, f"{label}.{phase}")
        _profile_nonnegative_number(
            aggregate["kernel_count"], f"{label}.{phase}.kernel_count", integer=True
        )
        for field in sorted(aggregate_fields - {"kernel_count", "families"}):
            _profile_nonnegative_number(
                aggregate[field],
                f"{label}.{phase}.{field}",
                integer=not milliseconds,
            )
        families = aggregate["families"]
        if not isinstance(families, dict):
            raise SelectionError(f"{label}.{phase}.families must be an object")
        if set(families) != set(CANDIDATES[item]["family"] for item in CANDIDATES) - {"attention_recurrent"} | {"attention", "recurrent", "head"}:
            # The canonical profiler owns six concrete kernel families.
            expected = {
                "paged_validation",
                "aq4_projection",
                "attention",
                "recurrent",
                "normalization",
                "head",
            }
            if set(families) != expected:
                raise SelectionError(
                    f"{label}.{phase}.families fields differ: "
                    f"missing={sorted(expected - set(families))}, "
                    f"unknown={sorted(set(families) - expected)}"
                )
        for family, metrics in families.items():
            if not isinstance(metrics, dict):
                raise SelectionError(f"{label}.{phase}.families.{family} must be an object")
            exact_fields(metrics, family_fields, f"{label}.{phase}.families.{family}")
            for field in sorted(family_fields):
                _profile_nonnegative_number(
                    metrics[field],
                    f"{label}.{phase}.families.{family}.{field}",
                    integer=not milliseconds,
                )
        if suffix == "ns":
            partition = (
                sum(metrics["exclusive_ns"] for metrics in families.values())
                + aggregate["cross_family_overlap_ns"]
                + aggregate["unclassified_ns"]
            )
            if partition != aggregate["gpu_total_union_ns"]:
                raise SelectionError(f"{label}.{phase} partition does not conserve GPU time")


def validate_diagnostic_profile(value: dict[str, Any], snapshot: Snapshot) -> dict[str, str]:
    if value.get("schema_version") != PROFILE_SCHEMA:
        raise SelectionError(f"unsupported evidence schema: {value.get('schema_version')!r}")
    exact_fields(value, PROFILE_ROOT_FIELDS, "diagnostic profile")
    if value.get("status") != "profiled_diagnostic":
        raise SelectionError("diagnostic profile status differs")
    require_bool(value.get("measurement_eligible"), False, "profile measurement_eligible")
    require_bool(value.get("promotion"), False, "profile promotion")
    binding = value.get("binding")
    if not isinstance(binding, dict):
        raise SelectionError("diagnostic profile binding must be an object")
    exact_fields(binding, PROFILE_BINDING_FIELDS, "diagnostic profile binding")
    case = binding["case"]
    identity = binding["identity"]
    device = binding["device"]
    if not all(isinstance(item, dict) for item in (case, identity, device)):
        raise SelectionError("diagnostic profile binding objects are incomplete")
    exact_fields(case, PROFILE_CASE_FIELDS, "diagnostic profile binding.case")
    exact_fields(identity, PROFILE_IDENTITY_FIELDS, "diagnostic profile binding.identity")
    exact_fields(device, PROFILE_DEVICE_FIELDS, "diagnostic profile binding.device")
    for field in (
        "case_sha256",
        "case_binding_sha256",
        "identity_file_sha256",
        "identity_sha256",
        "worker_binary_sha256",
        "served_model_manifest_sha256",
        "guard_set_sha256",
        "resident_binary_sha256",
        "package_manifest_sha256",
        "package_content_sha256",
        "policy_sha256",
        "served_model_manifest_sha256",
        "worker_binary_sha256",
    ):
        source = case if field in case else identity if field in identity else binding
        require_digest(source[field], f"diagnostic profile {field}")
    for field in ("prefill_requested_m", "resolved_m"):
        if type(case[field]) is not int or case[field] <= 0:
            raise SelectionError(f"diagnostic profile binding.case.{field} must be positive integer")
    if type(device["runtime_device_index"]) is not int or device["runtime_device_index"] < 0:
        raise SelectionError("diagnostic profile device index is invalid")
    for field in ("case_id",):
        _profile_string(case[field], f"diagnostic profile binding.case.{field}")
    for field in ("model_id", "model_revision", "protocol"):
        _profile_string(identity[field], f"diagnostic profile binding.identity.{field}")
    build_commit = _profile_string(
        identity["build_git_commit"], "diagnostic profile binding.identity.build_git_commit"
    )
    if re.fullmatch(r"[0-9a-f]{40}", build_commit) is None:
        raise SelectionError("diagnostic profile build_git_commit must be lowercase 40-hex")
    for field in ("device_id", "backend", "name", "architecture"):
        _profile_string(device[field], f"diagnostic profile binding.device.{field}")

    profiler = value["profiler"]
    trace = value["trace"]
    mapping = value["mapping"]
    schedule = value["schedule_separation"]
    if not all(isinstance(item, dict) for item in (profiler, trace, mapping, schedule)):
        raise SelectionError("diagnostic profile nested objects are incomplete")
    exact_fields(profiler, PROFILE_PROFILER_FIELDS, "diagnostic profile profiler")
    exact_fields(trace, PROFILE_TRACE_FIELDS, "diagnostic profile trace")
    exact_fields(mapping, PROFILE_MAPPING_FIELDS, "diagnostic profile mapping")
    exact_fields(schedule, PROFILE_SCHEDULE_FIELDS, "diagnostic profile schedule")
    trace_schema = trace["schema"]
    if not isinstance(trace_schema, dict):
        raise SelectionError("diagnostic profile trace.schema must be an object")
    exact_fields(trace_schema, PROFILE_TRACE_SCHEMA_FIELDS, "diagnostic profile trace.schema")
    for field in ("tool", "path", "version"):
        _profile_string(profiler[field], f"diagnostic profile profiler.{field}")
    if profiler["rocm_version"] is not None:
        _profile_string(profiler["rocm_version"], "diagnostic profile profiler.rocm_version")
    require_digest(profiler["executable_sha256"], "diagnostic profile profiler.executable_sha256")
    require_digest(
        profiler["version_output_sha256"],
        "diagnostic profile profiler.version_output_sha256",
    )
    if (
        type(profiler["subprocess_profile_runs"]) is not int
        or profiler["subprocess_profile_runs"] != 1
    ):
        raise SelectionError("diagnostic profile profiler.subprocess_profile_runs must be integer one")
    command = profiler["command"]
    if type(command) is not list or not command or any(type(item) is not str or not item for item in command):
        raise SelectionError("diagnostic profile profiler.command must be non-empty string array")
    columns = trace_schema["columns"]
    if (
        type(columns) is not list
        or not columns
        or any(type(item) is not str or not item for item in columns)
        or len(columns) != len(set(columns))
    ):
        raise SelectionError("diagnostic profile trace.schema.columns differs")
    for field in ("dispatch_id", "kernel_name", "start_timestamp", "end_timestamp", "clock_unit"):
        _profile_string(trace_schema[field], f"diagnostic profile trace.schema.{field}")
    if trace_schema["phase"] is not None:
        _profile_string(trace_schema["phase"], "diagnostic profile trace.schema.phase")
    if trace_schema["clock_unit"] != "nanoseconds":
        raise SelectionError("diagnostic profile trace clock unit differs")
    require_digest(trace["sha256"], "diagnostic profile trace.sha256")
    require_digest(mapping["sha256"], "diagnostic profile mapping.sha256")
    _profile_nonnegative_number(trace["bytes"], "diagnostic profile trace.bytes", integer=True)
    _profile_nonnegative_number(
        trace["kernel_count"], "diagnostic profile trace.kernel_count", integer=True
    )
    _profile_string(mapping["schema_version"], "diagnostic profile mapping.schema_version")
    for field in ("maximum_unclassified_fraction", "observed_unclassified_fraction"):
        _profile_nonnegative_number(mapping[field], f"diagnostic profile mapping.{field}")
        number = mapping[field]
        if number > 1.0:
            raise SelectionError(f"diagnostic profile mapping.{field} exceeds one")
    if type(mapping["complete"]) is not bool or type(mapping["unknown_kernel_names"]) is not list:
        raise SelectionError("diagnostic profile mapping types differ")
    if any(type(item) is not str or not item for item in mapping["unknown_kernel_names"]):
        raise SelectionError("diagnostic profile mapping unknown kernel names differ")
    if type(value["eligibility_blockers"]) is not list or any(
        type(item) is not str or not item for item in value["eligibility_blockers"]
    ):
        raise SelectionError("diagnostic profile eligibility_blockers differ")
    if (
        type(schedule["warmup_runs"]) is not int
        or schedule["warmup_runs"] != 2
        or type(schedule["measured_runs"]) is not int
        or schedule["measured_runs"] != 10
        or type(schedule["profile_aggregation_used_for_performance"]) is not bool
        or schedule["profile_aggregation_used_for_performance"] is not False
        or type(schedule["inclusive_kernel_sum_used_as_gpu_total"]) is not bool
        or schedule["inclusive_kernel_sum_used_as_gpu_total"] is not False
    ):
        raise SelectionError("diagnostic profile schedule differs")
    _validate_profile_timing(value["timing_ns"], "diagnostic profile timing_ns", milliseconds=False)
    _validate_profile_timing(value["timing_ms"], "diagnostic profile timing_ms", milliseconds=True)
    return {"sha256": snapshot.sha256, "identity_sha256": identity["identity_sha256"]}


def stable_float(value: float) -> float:
    if not math.isfinite(value):
        raise SelectionError("derived value is non-finite")
    if value == 0.0:
        return 0.0
    result = float(f"{value:.15g}")
    if not math.isfinite(result):
        raise SelectionError("rounded derived value is non-finite")
    return result


def finite_derived(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise SelectionError(f"{label} is non-finite")
    return value


def safe_fsum(values: Iterable[float], label: str) -> float:
    try:
        result = math.fsum(sorted(values))
    except (OverflowError, ValueError) as error:
        raise SelectionError(f"{label} overflowed") from error
    return finite_derived(result, label)


def safe_squared_difference(value: float, center: float, label: str) -> float:
    try:
        result = (value - center) ** 2
    except OverflowError as error:
        raise SelectionError(f"{label} overflowed") from error
    return finite_derived(result, label)


def stable_mean(values: Iterable[float]) -> float:
    ordered = list(values)
    return finite_derived(safe_fsum(ordered, "mean sum") / len(ordered), "mean")


def median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise SelectionError("median of empty values")
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return finite_derived(
        safe_fsum((ordered[middle - 1], ordered[middle]), "median sum") / 2.0,
        "median",
    )


def above_strict(value: float, threshold: float) -> bool:
    return value > threshold and not math.isclose(
        value, threshold, rel_tol=1e-12, abs_tol=1e-15
    )


def paired_ci(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(pairs)
    if count < 2:
        return {
            "pair_count": count,
            "mean_improvement_ms": None,
            "ci95_halfwidth_ms": None,
            "ci95_lower_ms": None,
            "ci95_upper_ms": None,
        }
    if count > 30:
        raise SelectionError("full-model paired sample count exceeds 30")
    improvements = sorted(
        finite_derived(
            row["baseline_ms"] - row["candidate_ms"], "paired improvement"
        )
        for row in pairs
    )
    mean = stable_mean(improvements)
    squared = sorted(
        safe_squared_difference(value, mean, "paired squared deviation")
        for value in improvements
    )
    sample_variance = finite_derived(
        safe_fsum(squared, "paired variance sum") / (count - 1),
        "paired sample variance",
    )
    standard_error = finite_derived(
        math.sqrt(sample_variance / count), "paired standard error"
    )
    halfwidth = finite_derived(
        T_CRITICAL_975[count - 1] * standard_error, "paired CI halfwidth"
    )
    return {
        "pair_count": count,
        "mean_improvement_ms": stable_float(mean),
        "ci95_halfwidth_ms": stable_float(halfwidth),
        "ci95_lower_ms": stable_float(mean - halfwidth),
        "ci95_upper_ms": stable_float(mean + halfwidth),
    }


def _candidate_a_reasons(
    rows: list[dict[str, Any]], pairs: list[dict[str, Any]], reasons: list[str]
) -> None:
    if not rows or not pairs:
        return
    if any(row["candidate_d2d_bytes"] > row["baseline_d2d_bytes"] for row in rows):
        reasons.append("candidate_a_d2d_bytes_regressed")
    if not any(row["candidate_d2d_bytes"] < row["baseline_d2d_bytes"] for row in rows):
        reasons.append("candidate_a_d2d_bytes_not_reduced")
    if any(row["candidate_d2d_copy_count"] > row["baseline_d2d_copy_count"] for row in rows):
        reasons.append("candidate_a_d2d_copy_count_regressed")
    if not any(row["candidate_d2d_copy_count"] < row["baseline_d2d_copy_count"] for row in rows):
        reasons.append("candidate_a_d2d_copy_count_not_reduced")
    if any(row["candidate_launch_count"] > row["baseline_launch_count"] for row in rows):
        reasons.append("candidate_a_launch_count_regressed")
    if any(
        row["candidate_workspace_bytes"] > row["baseline_workspace_bytes"]
        for row in rows
    ):
        reasons.append("candidate_a_workspace_regressed")
    if any(
        row["candidate_peak_vram_bytes"] > row["baseline_peak_vram_bytes"]
        for row in rows
    ):
        reasons.append("candidate_a_peak_vram_regressed")
    if any(row["candidate_fallback_count"] > row["baseline_fallback_count"] for row in rows):
        reasons.append("candidate_a_fallback_regressed")
    if any(
        row["candidate_fallback_count"] == 0 and row["candidate_fallback_reasons"]
        or not set(row["candidate_fallback_reasons"]).issubset(row["baseline_fallback_reasons"])
        for row in rows
    ):
        reasons.append("candidate_a_fallback_reasons_inconsistent")
    if any(
        not row["direct_alias_safe"] or not row["direct_size_safe"] or not row["direct_admission_safe"]
        for row in rows
    ) or any(
        not row["direct_alias_safe"] or not row["direct_size_safe"] or not row["direct_admission_safe"]
        for row in pairs
    ):
        reasons.append("candidate_a_safety_regression")
    if any(
        row["direct_fidelity_binding_sha256"] != row["copy_fidelity_binding_sha256"]
        for row in rows
    ) or any(
        row["direct_fidelity_binding_sha256"] != row["copy_fidelity_binding_sha256"]
        for row in pairs
    ):
        reasons.append("candidate_a_fidelity_binding_mismatch")
    if any(
        row["candidate_d2d_bytes"] > row["baseline_d2d_bytes"]
        or row["candidate_d2d_copy_count"] > row["baseline_d2d_copy_count"]
        or row["candidate_launch_count"] > row["baseline_launch_count"]
        or row["candidate_workspace_bytes"] > row["baseline_workspace_bytes"]
        or row["candidate_peak_vram_bytes"] > row["baseline_peak_vram_bytes"]
        or row["candidate_fallback_count"] > row["baseline_fallback_count"]
        for row in pairs
    ):
        reasons.append("candidate_a_full_model_safety_regression")
    if any(
        not set(row["candidate_fallback_reasons"]).issubset(row["baseline_fallback_reasons"])
        for row in pairs
    ):
        reasons.append("candidate_a_full_model_fallback_regression")


def evaluate_candidate(
    candidate_id: str,
    measurements: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    capabilities: dict[str, bool],
    raw_present: bool,
) -> dict[str, Any]:
    policy = CANDIDATES[candidate_id]
    rows = sorted(measurements, key=lambda row: (row["prompt_id"], row["case_sha256"]))
    prompt_results: list[dict[str, Any]] = []
    for row in rows:
        baseline = row["baseline_p50_ms"]
        recoverable_share = finite_derived(
            row["recoverable_family_exclusive_ms"] / baseline,
            "recoverable share E",
        )
        cv_term = finite_derived(3.0 * row["baseline_cv"], "noise floor CV term")
        ci_numerator = finite_derived(
            2.0 * row["ci95_halfwidth_ms"], "noise floor CI numerator"
        )
        ci_term = finite_derived(ci_numerator / baseline, "noise floor CI term")
        noise_floor = finite_derived(max(0.05, cv_term, ci_term), "noise floor N")
        prompt_results.append(
            {
                "prompt_id": row["prompt_id"],
                "case_sha256": row["case_sha256"],
                "resolved_m": row["resolved_m"],
                "recoverable_share_e": stable_float(recoverable_share),
                "noise_floor_n": stable_float(noise_floor),
                "e_above_n": above_strict(recoverable_share, noise_floor),
                "d2h_count": row["d2h_count"],
                "d2h_time_ms": row["d2h_time_ms"],
                "stream_sync_count": row["stream_sync_count"],
                "stream_sync_time_ms": row["stream_sync_time_ms"],
            }
        )
        if policy.get("candidate_a"):
            prompt_results[-1].update(
                {
                    field: row[field]
                    for field in CANDIDATE_A_MEASUREMENT_FIELDS - MEASUREMENT_FIELDS
                }
            )
    recoverable_e = median(item["recoverable_share_e"] for item in prompt_results) if prompt_results else None
    noise_n = median(item["noise_floor_n"] for item in prompt_results) if prompt_results else None
    above = [item for item in prompt_results if item["e_above_n"]]
    paired = paired_ci(pairs)
    reasons: list[str] = []
    if not raw_present:
        reasons.append("eligible_raw_evidence_missing")
    if not capabilities.get("family_exclusive_timing", False):
        reasons.append("family_exclusive_timing_missing")
    if len(prompt_results) != REPRESENTATIVE_PROMPTS:
        reasons.append("representative_prompt_count_not_7")
    if recoverable_e is None or noise_n is None or not above_strict(recoverable_e, noise_n):
        reasons.append("aggregate_e_not_above_n")
    if len(above) < MIN_ABOVE_NOISE:
        reasons.append("representative_above_noise_lt_4")
    if not any(item["resolved_m"] == 128 for item in above):
        reasons.append("m128_above_noise_missing")
    if not any(item["resolved_m"] != 128 for item in above):
        reasons.append("non_m128_above_noise_missing")
    if paired["pair_count"] < 2:
        reasons.append("paired_full_model_sample_lt_2")
    lower = paired["ci95_lower_ms"]
    if lower is None or not above_strict(lower, 0.0):
        reasons.append("paired_full_model_ci95_not_positive")

    if policy["requires_d2h_count"]:
        if not capabilities.get("d2h_count", False) or any(
            item["d2h_count"] is None or item["d2h_time_ms"] is None
            for item in prompt_results
        ):
            reasons.append("paged_kv_d2h_count_missing")
        if not capabilities.get("stream_sync_count", False) or any(
            item["stream_sync_count"] is None or item["stream_sync_time_ms"] is None
            for item in prompt_results
        ):
            reasons.append("paged_kv_stream_sync_count_missing")
        observed = any(
            (item["d2h_count"] or 0) > 0 or (item["stream_sync_count"] or 0) > 0
            for item in prompt_results
        )
        if not observed:
            reasons.append("paged_kv_transfer_or_sync_not_observed")

    if policy.get("candidate_a"):
        for field in sorted(CANDIDATE_A_CAPABILITY_FIELDS):
            if not capabilities.get(field, False):
                reasons.append(f"candidate_a_{field}_missing")
        _candidate_a_reasons(prompt_results, pairs, reasons)

    result = {
        "candidate_id": candidate_id,
        "family": policy["family"],
        "eligible": not reasons,
        "reason_codes": sorted(set(reasons)),
        "recoverable_share_e": stable_float(recoverable_e) if recoverable_e is not None else None,
        "noise_floor_n": stable_float(noise_n) if noise_n is not None else None,
        "e_minus_n": stable_float(
            finite_derived(recoverable_e - noise_n, "candidate E minus N")
        )
        if recoverable_e is not None and noise_n is not None
        else None,
        "representative": {
            "required_prompt_count": REPRESENTATIVE_PROMPTS,
            "observed_prompt_count": len(prompt_results),
            "minimum_above_noise": MIN_ABOVE_NOISE,
            "above_noise_count": len(above),
            "m128_above_noise": any(item["resolved_m"] == 128 for item in above),
            "non_m128_above_noise": any(item["resolved_m"] != 128 for item in above),
            "prompts": prompt_results,
        },
        "paired_full_model_95ci": paired,
        "required_evidence": {
            "family_exclusive_timing": capabilities.get("family_exclusive_timing", False),
            "d2h_count": capabilities.get("d2h_count", False),
            "stream_sync_count": capabilities.get("stream_sync_count", False),
        },
    }
    if policy.get("candidate_a"):
        result["required_evidence"].update(
            {
                field: capabilities.get(field, False)
                for field in sorted(CANDIDATE_A_CAPABILITY_FIELDS)
            }
        )
    return result


def select(values: list[tuple[Snapshot, dict[str, Any]]]) -> dict[str, Any]:
    raw_sources: list[RawSource] = []
    profiles: list[dict[str, str]] = []
    for snapshot, value in values:
        schema = value.get("schema_version")
        if schema == RAW_SCHEMA:
            raw_sources.append(validate_raw(value))
        elif schema == PROFILE_SCHEMA:
            profiles.append(validate_diagnostic_profile(value, snapshot))
        else:
            raise SelectionError(f"unsupported evidence schema: {schema!r}")
    if not values:
        raise SelectionError("at least one evidence file is required")

    identities = {source.identity["identity_sha256"] for source in raw_sources}
    if len(identities) > 1:
        raise SelectionError("raw evidence identity SHA-256 values differ")
    qualification_ids = {
        source.upstream_qualification["qualification_sha256"] for source in raw_sources
    }
    if len(qualification_ids) > 1:
        raise SelectionError("raw evidence upstream P2 qualifications differ")
    qualification_statuses = {
        source.upstream_qualification["status"] for source in raw_sources
    }
    if len(qualification_statuses) > 1:
        raise SelectionError("qualified and rejected raw evidence cannot be mixed")
    measurements: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for source in raw_sources:
        measurements.extend(source.measurements)
        pairs.extend(source.pairs)
    measurement_keys = [(row["candidate_id"], row["prompt_id"]) for row in measurements]
    pair_keys = [(row["candidate_id"], row["pair_id"]) for row in pairs]
    if len(measurement_keys) != len(set(measurement_keys)):
        raise SelectionError("duplicate candidate/prompt measurement across evidence files")
    if len(pair_keys) != len(set(pair_keys)):
        raise SelectionError("duplicate candidate/pair measurement across evidence files")

    candidates = []
    for candidate_id in sorted(CANDIDATES):
        candidate_measurements = [
            row for row in measurements if row["candidate_id"] == candidate_id
        ]
        candidate_pairs = [row for row in pairs if row["candidate_id"] == candidate_id]
        candidate_sources = [
            source
            for source in raw_sources
            if any(row["candidate_id"] == candidate_id for row in source.measurements)
            or any(row["candidate_id"] == candidate_id for row in source.pairs)
        ]
        candidate_capabilities = {
            field: bool(candidate_sources)
            and all(source.capabilities[field] for source in candidate_sources)
            for field in (
                CAPABILITY_FIELDS
                | (CANDIDATE_A_CAPABILITY_FIELDS if CANDIDATES[candidate_id].get("candidate_a") else set())
            )
        }
        candidates.append(
            evaluate_candidate(
                candidate_id,
                candidate_measurements,
                candidate_pairs,
                candidate_capabilities,
                bool(candidate_measurements or candidate_pairs),
            )
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda candidate: (
            -candidate["e_minus_n"],
            -candidate["representative"]["above_noise_count"],
            -candidate["paired_full_model_95ci"]["ci95_lower_ms"],
            candidate["candidate_id"],
        ),
    )
    result = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "selected" if ranked else "no_eligible_candidate",
        "selected_candidate_id": ranked[0]["candidate_id"] if ranked else None,
        "eligible_candidate_ids": [candidate["candidate_id"] for candidate in ranked],
        "policy": {
            "schema_version": POLICY_VERSION,
            "noise_floor_formula": "max(0.05,3*baseline_cv,2*ci95_halfwidth_ms/baseline_p50_ms)",
            "representative_prompt_count": REPRESENTATIVE_PROMPTS,
            "minimum_prompts_above_noise": MIN_ABOVE_NOISE,
            "requires_m128_and_non_m128": True,
            "paired_full_model_ci95_lower_must_exceed_zero": True,
            "selection_order": "e_minus_n_desc,above_noise_count_desc,paired_ci95_lower_desc,candidate_id_asc",
        },
        "input_binding": {
            "identity_sha256": next(iter(identities)) if identities else None,
            "raw_evidence_semantic_sha256": sorted(
                source.semantic_sha256 for source in raw_sources
            ),
            "diagnostic_profile_file_sha256": sorted(profile["sha256"] for profile in profiles),
            "diagnostic_profiles_measurement_eligible": False,
            "upstream_qualification_sha256": next(iter(qualification_ids)) if qualification_ids else None,
            "upstream_qualification_status": next(iter(qualification_statuses)) if qualification_statuses else None,
            "upstream_p2_promotion_eligible": all(source.promotion_eligible for source in raw_sources) if raw_sources else False,
            "upstream_p2_reason": raw_sources[0].upstream_qualification["reason"] if raw_sources else None,
        },
        "input_warnings": (
            [
                "diagnostic family-exclusive profiles are not measurement eligible and do not provide D2H/stream-sync counts"
            ]
            if profiles
            else []
        ),
        "candidates": candidates,
    }
    ensure_finite_tree(result, "selection output")
    return result


def write_output(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SelectionError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_finite_tree(value, "selection output")
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False
    ).encode("ascii") + b"\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise SelectionError(f"refusing to overwrite output: {path}") from error
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        snapshots = [capture(path) for path in args.evidence]
        values = [(snapshot, parse_json(snapshot)) for snapshot in snapshots]
        result = select(values)
        for snapshot in snapshots:
            snapshot.verify()
        write_output(args.output, result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "selected_candidate_id": result["selected_candidate_id"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, SelectionError) as error:
        print(f"select-aq4-p3-candidate: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
