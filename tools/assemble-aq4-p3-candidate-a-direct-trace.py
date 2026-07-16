#!/usr/bin/env python3
"""Assemble candidate-A direct-route evidence from runtime counters and a profiler record.

The runtime record is emitted by ``Qwen35Aq4ModelRuntime`` only when
``ULLM_AQ4_P3_DIRECT_TRACE_DIAGNOSTIC=1``.  It contains route-apply counters; it does
not estimate transfer volume or timing.  Peak VRAM, latency, and fidelity are accepted
only from the separately hash-bound profiler record.  This tool deliberately refuses
to synthesize missing observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any


RUNTIME_SCHEMA = "ullm.aq4_p3_candidate_a_direct_runtime_observation.v1"
PROFILER_SCHEMA = "ullm.aq4_p3_candidate_a_direct_profiler_observation.v1"
TRACE_SCHEMA = "ullm.aq4_p3_candidate_a_direct_sequence_output_trace.v1"
CANDIDATE_ID = "sequence-output-direct-v1"
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_REASON_COUNT = 16
MAX_TEXT = 128
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")

RUNTIME_FIELDS = {
    "schema_version", "status", "record_sha256", "side", "binding_kind", "binding_id",
    "request_id", "implementation_id", "source_id", "source_sha256", "candidate_id",
    "case_id", "case_sha256", "identity_sha256", "diagnostic_gate",
    "direct_sequence_output_enabled", "measurement_eligible", "counters",
}
PROFILER_FIELDS = {
    "schema_version", "status", "record_sha256", "side", "binding_kind", "binding_id",
    "request_id", "implementation_id", "source_id", "source_sha256", "candidate_id",
    "case_id", "case_sha256", "identity_sha256", "timing_lane", "measurement_eligible",
    "component_ms", "full_model_ms", "peak_vram_bytes", "fidelity_binding_sha256",
}
COUNTER_FIELDS = {
    "invocation_count", "d2d_bytes", "d2d_copy_count", "launch_count", "workspace_bytes",
    "fallback_count", "fallback_reasons", "direct_alias_safe", "direct_size_safe",
    "direct_admission_safe", "failed_invocation_count", "failure_reasons",
}
TRACE_FIELDS = {
    "schema_version", "status", "trace_sha256", "binding_kind", "binding_id",
    "candidate_id", "case_id", "case_sha256", "identity_sha256", "implementation_id",
    "source_id", "source_sha256", "request_id", "events",
}
EVENT_FIELDS = {"event_id", "event_sha256", "side", "metric", "value"}
RUN_METRICS = {
    "d2d_bytes", "d2d_copy_count", "launch_count", "component_ms", "full_model_ms",
    "workspace_bytes", "peak_vram_bytes", "fallback_count", "fallback_reasons",
    "alias_safe", "size_safe", "admission_safe", "fidelity_binding_sha256",
}
PAIR_METRICS = RUN_METRICS - {"component_ms", "full_model_ms"}


class AssemblerError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def self_hash(value: dict[str, Any], field: str) -> str:
    clone = json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))
    clone[field] = None
    return hashlib.sha256(canonical(clone)).hexdigest()


def file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


class Snapshot:
    def __init__(self, path: Path, identity: tuple[int, ...], sha256: str, data: bytes):
        self.path, self.identity, self.sha256, self.data = path, identity, sha256, data

    def verify(self) -> None:
        try:
            current = capture(self.path, "input verification")
        except (AssemblerError, OSError) as error:
            raise AssemblerError(f"input verification failed: {self.path}: {error}") from error
        if current.identity != self.identity or current.sha256 != self.sha256:
            raise AssemblerError(f"input identity or SHA-256 changed: {self.path}")


def capture(path: Path, label: str) -> Snapshot:
    if not path.is_absolute():
        raise AssemblerError(f"{label} path must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise AssemblerError(f"{label} path contains a symlink: {current}")
        except OSError as error:
            raise AssemblerError(f"cannot inspect {label} path: {error}") from error
    try:
        path = path.resolve(strict=True)
        before = path.lstat()
    except OSError as error:
        raise AssemblerError(f"cannot open {label}: {error}") from error
    if not stat.S_ISREG(before.st_mode):
        raise AssemblerError(f"{label} must be a regular file")
    if before.st_size > MAX_INPUT_BYTES:
        raise AssemblerError(f"{label} exceeds {MAX_INPUT_BYTES} bytes")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        opened = os.fstat(descriptor)
        if file_identity(opened) != file_identity(before):
            raise AssemblerError(f"{label} identity changed while opening")
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_INPUT_BYTES:
                raise AssemblerError(f"{label} exceeds {MAX_INPUT_BYTES} bytes")
            chunks.append(chunk)
            digest.update(chunk)
        if file_identity(os.fstat(descriptor)) != file_identity(before) or file_identity(path.lstat()) != file_identity(before):
            raise AssemblerError(f"{label} identity changed while reading")
    finally:
        os.close(descriptor)
    return Snapshot(path, file_identity(before), digest.hexdigest(), b"".join(chunks))


def parse_json(snapshot: Snapshot, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise AssemblerError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(snapshot.data, object_pairs_hook=pairs,
                          parse_constant=lambda token: (_ for _ in ()).throw(
                              AssemblerError(f"non-finite JSON in {label}: {token}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AssemblerError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise AssemblerError(f"{label} root must be an object")
    ensure_finite(value, label)
    return value


def ensure_finite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AssemblerError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            ensure_finite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            ensure_finite(child, f"{label}[{index}]")


def exact(value: dict[str, Any], fields: set[str], label: str) -> None:
    missing, unknown = sorted(fields - set(value)), sorted(set(value) - fields)
    if missing or unknown:
        raise AssemblerError(f"{label} fields differ: missing={missing}, unknown={unknown}")


def text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_TEXT or not ID_RE.fullmatch(value):
        raise AssemblerError(f"{label} must be a bounded identifier")
    return value


def sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise AssemblerError(f"{label} must be a lowercase SHA-256 digest")
    return value


def boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise AssemblerError(f"{label} must be boolean")
    return value


def integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AssemblerError(f"{label} must be a non-negative integer")
    return value


def positive_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise AssemblerError(f"{label} must be a positive finite float")
    return value


def nullable_float(value: Any, label: str, *, required: bool) -> float | None:
    if value is None and not required:
        return None
    return positive_float(value, label)


def validate_common(value: dict[str, Any], *, fields: set[str], label: str, side: str,
                    binding_kind: str, binding_id: str, common: dict[str, str]) -> None:
    exact(value, fields, label)
    if value["status"] != "complete" or value["side"] != side or value["binding_kind"] != binding_kind or value["binding_id"] != binding_id:
        raise AssemblerError(f"{label} status/side/binding differs")
    if value["record_sha256"] != self_hash(value, "record_sha256"):
        raise AssemblerError(f"{label} self-hash differs")
    for key in ("request_id", "implementation_id", "source_id"):
        if text(value[key], f"{label}.{key}") != common[key]:
            raise AssemblerError(f"{label}.{key} differs")
    if sha(value["source_sha256"], f"{label}.source_sha256") != common["source_sha256"]:
        raise AssemblerError(f"{label}.source_sha256 differs")
    if value["candidate_id"] != CANDIDATE_ID:
        raise AssemblerError(f"{label}.candidate_id differs")
    if text(value["case_id"], f"{label}.case_id") != common["case_id"] or sha(value["case_sha256"], f"{label}.case_sha256") != common["case_sha256"] or sha(value["identity_sha256"], f"{label}.identity_sha256") != common["identity_sha256"]:
        raise AssemblerError(f"{label} case/identity differs")


def validate_runtime(snapshot: Snapshot, *, side: str, binding_kind: str, binding_id: str,
                     common: dict[str, str] | None) -> tuple[dict[str, str], dict[str, Any]]:
    value = parse_json(snapshot, "runtime observation")
    base = {key: value.get(key) for key in ("request_id", "implementation_id", "source_id", "source_sha256", "case_id", "case_sha256", "identity_sha256")}
    for key in ("request_id", "implementation_id", "source_id", "case_id"):
        text(base[key], f"runtime observation.{key}")
    for key in ("source_sha256", "case_sha256", "identity_sha256"):
        sha(base[key], f"runtime observation.{key}")
    if common is None:
        common = {key: str(base[key]) for key in base}
    validate_common(value, fields=RUNTIME_FIELDS, label="runtime observation", side=side,
                    binding_kind=binding_kind, binding_id=binding_id, common=common)
    if not boolean(value["diagnostic_gate"], "runtime observation.diagnostic_gate") or not value["diagnostic_gate"]:
        raise AssemblerError("runtime diagnostic gate was not explicitly enabled")
    if boolean(value["measurement_eligible"], "runtime observation.measurement_eligible"):
        raise AssemblerError("instrumented runtime observation cannot be measurement eligible")
    expected_direct = side == "candidate"
    if boolean(value["direct_sequence_output_enabled"], "runtime observation.direct_sequence_output_enabled") != expected_direct:
        raise AssemblerError("runtime direct-route gate does not match side")
    counters = value["counters"]
    if not isinstance(counters, dict):
        raise AssemblerError("runtime observation.counters must be an object")
    exact(counters, COUNTER_FIELDS, "runtime observation.counters")
    for key in ("invocation_count", "d2d_bytes", "d2d_copy_count", "launch_count", "workspace_bytes", "fallback_count", "failed_invocation_count"):
        integer(counters[key], f"runtime observation.counters.{key}")
    for key in ("direct_alias_safe", "direct_size_safe", "direct_admission_safe"):
        boolean(counters[key], f"runtime observation.counters.{key}")
    if counters["invocation_count"] == 0 or counters["launch_count"] == 0:
        raise AssemblerError("runtime counters contain no completed dispatch")
    if counters["d2d_copy_count"] > counters["invocation_count"] or counters["fallback_count"] > counters["invocation_count"]:
        raise AssemblerError("runtime counters exceed invocation count")
    if counters["failed_invocation_count"] or counters["failure_reasons"]:
        raise AssemblerError("runtime observation contains a failed invocation")
    reasons = counters["fallback_reasons"]
    if not isinstance(reasons, list) or len(reasons) > MAX_REASON_COUNT or any(type(item) is not str or not item or len(item) > MAX_TEXT or not item.isascii() or ID_RE.fullmatch(item) is None for item in reasons) or len(reasons) != len(set(reasons)):
        raise AssemblerError("runtime fallback reasons are invalid")
    if counters["fallback_count"] == 0 and reasons:
        raise AssemblerError("runtime fallback reasons are inconsistent")
    return common, counters


def validate_profiler(snapshot: Snapshot, *, side: str, binding_kind: str, binding_id: str,
                      common: dict[str, str]) -> dict[str, Any]:
    value = parse_json(snapshot, "profiler observation")
    validate_common(value, fields=PROFILER_FIELDS, label="profiler observation", side=side,
                    binding_kind=binding_kind, binding_id=binding_id, common=common)
    lane = value["timing_lane"]
    if lane not in {"profiler_off", "instrumented"}:
        raise AssemblerError("profiler timing_lane is unknown")
    if boolean(value["measurement_eligible"], "profiler observation.measurement_eligible") != (lane == "profiler_off"):
        raise AssemblerError("profiler measurement eligibility does not match timing lane")
    required_latency = binding_kind == "run"
    component = nullable_float(value["component_ms"], "profiler observation.component_ms", required=required_latency)
    full_model = nullable_float(value["full_model_ms"], "profiler observation.full_model_ms", required=required_latency)
    integer(value["peak_vram_bytes"], "profiler observation.peak_vram_bytes")
    fidelity = sha(value["fidelity_binding_sha256"], "profiler observation.fidelity_binding_sha256")
    return {"timing_lane": lane, "component_ms": component, "full_model_ms": full_model,
            "peak_vram_bytes": value["peak_vram_bytes"], "fidelity_binding_sha256": fidelity}


def event(side: str, metric: str, value: Any) -> dict[str, Any]:
    result = {"event_id": f"{side}-{metric}", "event_sha256": None, "side": side, "metric": metric, "value": value}
    result["event_sha256"] = self_hash(result, "event_sha256")
    return result


def assemble(paths: dict[str, Path], output: Path, binding_kind: str, binding_id: str) -> dict[str, Any]:
    if binding_kind not in {"run", "pair"}:
        raise AssemblerError("binding kind must be run or pair")
    snapshots = {key: capture(path.absolute(), f"{key} record") for key, path in paths.items()}
    common: dict[str, str] | None = None
    counters: dict[str, dict[str, Any]] = {}
    profiles: dict[str, dict[str, Any]] = {}
    for side in ("baseline", "candidate"):
        common, counters[side] = validate_runtime(snapshots[f"{side}_runtime"], side=side,
                                                   binding_kind=binding_kind, binding_id=binding_id, common=common)
        profiles[side] = validate_profiler(snapshots[f"{side}_profiler"], side=side,
                                           binding_kind=binding_kind, binding_id=binding_id, common=common)
    if profiles["baseline"]["fidelity_binding_sha256"] != profiles["candidate"]["fidelity_binding_sha256"]:
        raise AssemblerError("direct/copy fidelity binding differs")
    assert common is not None
    metrics = RUN_METRICS if binding_kind == "run" else PAIR_METRICS
    values: dict[str, dict[str, Any]] = {"baseline": {}, "candidate": {}}
    for side in ("baseline", "candidate"):
        c, p = counters[side], profiles[side]
        values[side] = {
            "d2d_bytes": c["d2d_bytes"], "d2d_copy_count": c["d2d_copy_count"],
            "launch_count": c["launch_count"], "workspace_bytes": c["workspace_bytes"],
            "peak_vram_bytes": p["peak_vram_bytes"], "fallback_count": c["fallback_count"],
            "fallback_reasons": sorted(c["fallback_reasons"]), "alias_safe": c["direct_alias_safe"],
            "size_safe": c["direct_size_safe"], "admission_safe": c["direct_admission_safe"],
            "fidelity_binding_sha256": p["fidelity_binding_sha256"],
        }
        if binding_kind == "run":
            values[side]["component_ms"] = p["component_ms"]
            values[side]["full_model_ms"] = p["full_model_ms"]
    events = [event(side, metric, values[side][metric]) for side in ("baseline", "candidate") for metric in sorted(metrics)]
    result: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA, "status": "complete", "trace_sha256": None,
        "binding_kind": binding_kind, "binding_id": binding_id, "candidate_id": CANDIDATE_ID,
        "case_id": common["case_id"], "case_sha256": common["case_sha256"],
        "identity_sha256": common["identity_sha256"], "implementation_id": common["implementation_id"],
        "source_id": common["source_id"], "source_sha256": common["source_sha256"],
        "request_id": common["request_id"], "events": events,
    }
    result["trace_sha256"] = self_hash(result, "trace_sha256")
    for snapshot in snapshots.values():
        snapshot.verify()
    if output.exists() or output.is_symlink():
        raise AssemblerError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode("ascii") + b"\n"
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-kind", choices=("run", "pair"), required=True)
    parser.add_argument("--binding-id", required=True)
    parser.add_argument("--baseline-runtime", type=Path, required=True)
    parser.add_argument("--baseline-profiler", type=Path, required=True)
    parser.add_argument("--candidate-runtime", type=Path, required=True)
    parser.add_argument("--candidate-profiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = assemble({
            "baseline_runtime": args.baseline_runtime,
            "baseline_profiler": args.baseline_profiler,
            "candidate_runtime": args.candidate_runtime,
            "candidate_profiler": args.candidate_profiler,
        }, args.output, args.binding_kind, text(args.binding_id, "binding_id"))
    except (AssemblerError, OSError, ValueError) as error:
        print(f"error: {error}")
        return 2
    eligible = all(
        parse_json(capture(path.absolute(), "profiler record"), "profiler record")["timing_lane"] == "profiler_off"
        for path in (args.baseline_profiler, args.candidate_profiler)
    )
    print(json.dumps({"status": result["status"], "measurement_eligible": eligible,
                      "trace_sha256": result["trace_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
