#!/usr/bin/env python3
"""Select an AQ4 P3 optimization candidate from hash-bound P2 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


RAW_SCHEMA = "ullm.aq4_p2_candidate_selection_raw.v1"
PROFILE_SCHEMA = "ullm.aq4_p2_family_exclusive_profile.v1"
OUTPUT_SCHEMA = "ullm.aq4_p3_candidate_selection.v1"
POLICY_VERSION = "ullm.aq4_p3_candidate_selection_policy.v1"
MAX_EVIDENCE_BYTES = 32 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REPRESENTATIVE_PROMPTS = 7
MIN_ABOVE_NOISE = 4

CANDIDATES: dict[str, dict[str, Any]] = {
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
    "evidence_sha256",
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
    "stream_sync_count",
}
PAIR_FIELDS = {
    "candidate_id",
    "pair_id",
    "case_sha256",
    "identity_sha256",
    "baseline_ms",
    "candidate_ms",
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
    return value


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
    result = float(value)
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


def validate_raw(value: dict[str, Any]) -> RawSource:
    exact_fields(value, RAW_ROOT_FIELDS, "raw evidence")
    if value["schema_version"] != RAW_SCHEMA or value["status"] != "complete":
        raise SelectionError("raw evidence schema/status differs")
    require_bool(value["measurement_eligible"], True, "raw measurement_eligible")
    require_bool(value["smoke_only"], False, "raw smoke_only")
    require_bool(value["promotion_eligible"], True, "raw promotion_eligible")
    if value["representative_prompt_count"] != REPRESENTATIVE_PROMPTS:
        raise SelectionError(
            f"raw representative_prompt_count must be {REPRESENTATIVE_PROMPTS}"
        )

    identity = value["identity"]
    capabilities = value["capabilities"]
    if not isinstance(identity, dict) or not isinstance(capabilities, dict):
        raise SelectionError("raw identity/capabilities must be objects")
    exact_fields(identity, IDENTITY_FIELDS, "raw identity")
    exact_fields(capabilities, CAPABILITY_FIELDS, "raw capabilities")
    identity_value = {
        field: require_digest(identity[field], f"raw identity.{field}")
        for field in sorted(IDENTITY_FIELDS)
    }
    capability_value: dict[str, bool] = {}
    for field in sorted(CAPABILITY_FIELDS):
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
        exact_fields(row, MEASUREMENT_FIELDS, label)
        candidate_id = row["candidate_id"]
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
        parsed_measurements.append(
            {
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
                "stream_sync_count": require_count(
                    row["stream_sync_count"],
                    f"{label}.stream_sync_count",
                    allow_none=True,
                ),
            }
        )

    parsed_pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(pairs):
        label = f"raw full_model_pairs[{index}]"
        if not isinstance(row, dict):
            raise SelectionError(f"{label} must be an object")
        exact_fields(row, PAIR_FIELDS, label)
        candidate_id = row["candidate_id"]
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
        parsed_pairs.append(
            {
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
        )

    declared_sha = require_digest(value["evidence_sha256"], "raw evidence_sha256")
    calculated_sha = semantic_sha256(value)
    if declared_sha != calculated_sha:
        raise SelectionError("raw evidence semantic SHA-256 differs")
    return RawSource(
        semantic_sha256=calculated_sha,
        identity=identity_value,
        capabilities=capability_value,
        measurements=tuple(parsed_measurements),
        pairs=tuple(parsed_pairs),
    )


def validate_diagnostic_profile(value: dict[str, Any], snapshot: Snapshot) -> dict[str, str]:
    if value.get("schema_version") != PROFILE_SCHEMA:
        raise SelectionError(f"unsupported evidence schema: {value.get('schema_version')!r}")
    exact_fields(value, PROFILE_ROOT_FIELDS, "diagnostic profile")
    if value.get("status") != "profiled_diagnostic":
        raise SelectionError("diagnostic profile status differs")
    require_bool(value.get("measurement_eligible"), False, "profile measurement_eligible")
    require_bool(value.get("promotion"), False, "profile promotion")
    timing = value.get("timing_ns")
    if not isinstance(timing, dict) or not isinstance(timing.get("prefill"), dict):
        raise SelectionError("diagnostic profile family timing is missing")
    families = timing["prefill"].get("families")
    if not isinstance(families, dict):
        raise SelectionError("diagnostic profile prefill families are missing")
    binding = value.get("binding")
    identity_sha = None
    if isinstance(binding, dict) and isinstance(binding.get("identity"), dict):
        identity_sha = binding["identity"].get("identity_sha256")
    if identity_sha is not None:
        require_digest(identity_sha, "diagnostic profile identity_sha256")
    return {"sha256": snapshot.sha256, "identity_sha256": identity_sha or ""}


def stable_float(value: float) -> float:
    if value == 0.0:
        return 0.0
    return float(f"{value:.15g}")


def stable_mean(values: Iterable[float]) -> float:
    ordered = sorted(values)
    return math.fsum(ordered) / len(ordered)


def median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise SelectionError("median of empty values")
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return math.fsum((ordered[middle - 1], ordered[middle])) / 2.0


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
    improvements = sorted(row["baseline_ms"] - row["candidate_ms"] for row in pairs)
    mean = stable_mean(improvements)
    squared = sorted((value - mean) ** 2 for value in improvements)
    sample_variance = math.fsum(squared) / (count - 1)
    standard_error = math.sqrt(sample_variance / count)
    halfwidth = T_CRITICAL_975[count - 1] * standard_error
    return {
        "pair_count": count,
        "mean_improvement_ms": stable_float(mean),
        "ci95_halfwidth_ms": stable_float(halfwidth),
        "ci95_lower_ms": stable_float(mean - halfwidth),
        "ci95_upper_ms": stable_float(mean + halfwidth),
    }


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
        recoverable_share = row["recoverable_family_exclusive_ms"] / baseline
        noise_floor = max(
            0.05,
            3.0 * row["baseline_cv"],
            2.0 * row["ci95_halfwidth_ms"] / baseline,
        )
        prompt_results.append(
            {
                "prompt_id": row["prompt_id"],
                "case_sha256": row["case_sha256"],
                "resolved_m": row["resolved_m"],
                "recoverable_share_e": stable_float(recoverable_share),
                "noise_floor_n": stable_float(noise_floor),
                "e_above_n": above_strict(recoverable_share, noise_floor),
                "d2h_count": row["d2h_count"],
                "stream_sync_count": row["stream_sync_count"],
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
            item["d2h_count"] is None for item in prompt_results
        ):
            reasons.append("paged_kv_d2h_count_missing")
        if not capabilities.get("stream_sync_count", False) or any(
            item["stream_sync_count"] is None for item in prompt_results
        ):
            reasons.append("paged_kv_stream_sync_count_missing")
        observed = any(
            (item["d2h_count"] or 0) > 0 or (item["stream_sync_count"] or 0) > 0
            for item in prompt_results
        )
        if not observed:
            reasons.append("paged_kv_transfer_or_sync_not_observed")

    return {
        "candidate_id": candidate_id,
        "family": policy["family"],
        "eligible": not reasons,
        "reason_codes": sorted(set(reasons)),
        "recoverable_share_e": stable_float(recoverable_e) if recoverable_e is not None else None,
        "noise_floor_n": stable_float(noise_n) if noise_n is not None else None,
        "e_minus_n": stable_float(recoverable_e - noise_n)
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
        measurement_sources = [
            source
            for source in raw_sources
            if any(row["candidate_id"] == candidate_id for row in source.measurements)
        ]
        candidate_capabilities = {
            field: bool(measurement_sources)
            and all(source.capabilities[field] for source in measurement_sources)
            for field in CAPABILITY_FIELDS
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
    return {
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


def write_output(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SelectionError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False
    ).encode("ascii") + b"\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
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
