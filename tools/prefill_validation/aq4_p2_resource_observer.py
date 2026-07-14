#!/usr/bin/env python3
"""Validate a bounded AQ4 P2 resource-observer sidecar.

The observer is intentionally separate from the benchmark process.  A production adapter can
capture rocm-smi (or a CPU RSS equivalent) into this schema while a case is running, then pass the
sidecar here before publishing measurement evidence.  The validator never infers a peak from a
producer summary and never stores prompt or generated content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 32 * 1024 * 1024
ROOT_FIELDS = {"schema_version", "case_id", "case_sha256", "device_id", "observer", "samples", "peak"}
OBSERVER_FIELDS = {"argv_sha256", "shell", "tool", "sample_period_ms", "target_process_name", "status"}
SAMPLE_FIELDS = {"monotonic_ms", "vram_used_bytes", "workspace_bytes", "power_watts", "temperature_c", "process_snapshot"}
PEAK_FIELDS = {"vram_used_bytes", "workspace_bytes", "power_watts", "temperature_c", "sample_index"}


class ObserverError(ValueError):
    pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ObserverError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ObserverError(f"{label} must be a bounded regular file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(ObserverError(f"non-finite JSON: {item}")),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ObserverError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ObserverError(f"{label} root must be an object")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def finite_number(value: Any, label: str, *, integer: bool = False) -> int | float:
    if integer:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ObserverError(f"{label} must be a non-negative integer")
        return value
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ObserverError(f"{label} must be finite and non-negative")
    return value


def validate(value: dict[str, Any], *, expected_case_id: str | None = None, expected_case_sha256: str | None = None) -> dict[str, Any]:
    if set(value) != ROOT_FIELDS or value.get("schema_version") != "ullm.aq4_p2_resource_observation.v1":
        raise ObserverError("resource observation root differs")
    if not isinstance(value.get("case_id"), str) or not value["case_id"]:
        raise ObserverError("resource observation case_id is invalid")
    if expected_case_id is not None and value["case_id"] != expected_case_id:
        raise ObserverError("resource observation case differs")
    if not isinstance(value.get("case_sha256"), str) or len(value["case_sha256"]) != 64:
        raise ObserverError("resource observation case hash is invalid")
    if expected_case_sha256 is not None and value["case_sha256"] != expected_case_sha256:
        raise ObserverError("resource observation case hash differs")
    if not isinstance(value.get("device_id"), str) or not value["device_id"]:
        raise ObserverError("resource observation device is invalid")
    observer = value["observer"]
    if not isinstance(observer, dict) or set(observer) != OBSERVER_FIELDS:
        raise ObserverError("resource observer contract differs")
    if not isinstance(observer["argv_sha256"], str) or len(observer["argv_sha256"]) != 64 or observer["shell"] is not False or not isinstance(observer["tool"], str) or not observer["tool"] or finite_number(observer["sample_period_ms"], "observer.sample_period_ms") <= 0 or not isinstance(observer["target_process_name"], str) or not observer["target_process_name"] or observer["status"] != "complete":
        raise ObserverError("resource observer contract is invalid")
    samples = value["samples"]
    if not isinstance(samples, list) or not samples:
        raise ObserverError("resource observation has no samples")
    normalized = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or set(sample) != SAMPLE_FIELDS:
            raise ObserverError(f"resource sample {index} differs")
        item = {
            "monotonic_ms": finite_number(sample["monotonic_ms"], f"sample {index}.monotonic_ms"),
            "vram_used_bytes": finite_number(sample["vram_used_bytes"], f"sample {index}.vram_used_bytes", integer=True),
            "workspace_bytes": finite_number(sample["workspace_bytes"], f"sample {index}.workspace_bytes", integer=True),
            "power_watts": finite_number(sample["power_watts"], f"sample {index}.power_watts"),
            "temperature_c": finite_number(sample["temperature_c"], f"sample {index}.temperature_c"),
            "process_snapshot": sample["process_snapshot"],
        }
        if not isinstance(item["process_snapshot"], list):
            raise ObserverError(f"resource sample {index}.process_snapshot is invalid")
        normalized.append(item)
    peak = value["peak"]
    if not isinstance(peak, dict) or set(peak) != PEAK_FIELDS:
        raise ObserverError("resource peak differs")
    peak_index = peak["sample_index"]
    if not isinstance(peak_index, int) or isinstance(peak_index, bool) or not 0 <= peak_index < len(normalized):
        raise ObserverError("resource peak sample index is invalid")
    expected_peak = {
        "vram_used_bytes": max(item["vram_used_bytes"] for item in normalized),
        "workspace_bytes": max(item["workspace_bytes"] for item in normalized),
        "power_watts": max(item["power_watts"] for item in normalized),
        "temperature_c": max(item["temperature_c"] for item in normalized),
    }
    for field, expected in expected_peak.items():
        finite_number(peak[field], f"peak.{field}", integer=field.endswith("bytes"))
        if peak[field] != expected:
            raise ObserverError(f"resource peak.{field} is not the observed maximum")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--case-sha256")
    args = parser.parse_args(argv)
    try:
        value = load(args.input, "resource observation")
        validate(value, expected_case_id=args.case_id, expected_case_sha256=args.case_sha256)
        report = {
            "schema_version": "ullm.aq4_p2_resource_observer_validation.v1",
            "status": "valid",
            "case_id": value["case_id"],
            "case_sha256": value["case_sha256"],
            "observation_sha256": sha_bytes(canonical(value)),
            "peak": value["peak"],
            "sample_count": len(value["samples"]),
        }
        if args.output.exists() or args.output.is_symlink():
            raise ObserverError("refusing to overwrite output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.incomplete")
        with temporary.open("x", encoding="utf-8") as target:
            json.dump(report, target, ensure_ascii=True, sort_keys=True, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(args.output)
        print(json.dumps(report, sort_keys=True))
        return 0
    except (ObserverError, OSError, ValueError) as error:
        print(f"P2 resource observation validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
