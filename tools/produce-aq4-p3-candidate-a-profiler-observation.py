#!/usr/bin/env python3
"""Build one candidate-A profiler observation from a bound raw profiler capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RAW_SCHEMA = "ullm.aq4_p3_candidate_a_direct_profiler_raw.v1"
OBSERVATION_SCHEMA = "ullm.aq4_p3_candidate_a_direct_profiler_observation.v1"
LANE = "profiler_off_measurement"
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_SAMPLES = 64
MAX_TEXT = 128
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")
COMMON_FIELDS = {
    "side", "binding_kind", "binding_id", "request_id", "implementation_id",
    "source_id", "source_sha256", "candidate_id", "case_id", "case_sha256",
    "identity_sha256",
}
RAW_FIELDS = {
    "schema_version", "status", "record_sha256", *COMMON_FIELDS,
    "timing_lane", "measurement_eligible", "command", "exit_code",
    "started_unix_ns", "completed_unix_ns", "samples",
}
SAMPLE_FIELDS = {
    "component_ms", "full_model_ms", "peak_vram_bytes", "fidelity_binding_sha256"
}
REF_FIELDS = {"path", "sha256", "device", "inode", "nlink"}
OBSERVATION_FIELDS = {
    "schema_version", "status", "record_sha256", *COMMON_FIELDS,
    "timing_lane", "measurement_eligible", "profiler", "raw_capture", "parser",
    "profiler_version", "command", "exit_code", "started_unix_ns",
    "completed_unix_ns", "sample_count", "component_ms", "full_model_ms",
    "peak_vram_bytes", "fidelity_binding_sha256",
}


class ProfilerEvidenceError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def self_hash(value: dict[str, Any], field: str) -> str:
    clone = json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))
    clone[field] = None
    return hashlib.sha256(canonical(clone)).hexdigest()


def exact(value: dict[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ProfilerEvidenceError(
            f"{label} fields differ: missing={missing}, unknown={unknown}"
        )


def identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or ID_RE.fullmatch(value) is None:
        raise ProfilerEvidenceError(f"{label} must be a bounded identifier")
    return value


def digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ProfilerEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def count(value: Any, label: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum or value > (1 << 53) - 1:
        raise ProfilerEvidenceError(f"{label} must be a safe non-negative integer")
    return value


def positive_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0:
        raise ProfilerEvidenceError(f"{label} must be a positive finite float")
    return value


def identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns,
    )


@dataclass(frozen=True)
class Snapshot:
    path: Path
    identity: tuple[int, ...]
    sha256: str
    data: bytes

    def reference(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "device": self.identity[0],
            "inode": self.identity[1],
            "nlink": self.identity[3],
        }

    def verify(self, label: str, *, executable: bool = False) -> None:
        current = capture(self.path, label, executable=executable)
        if current.identity != self.identity or current.sha256 != self.sha256:
            raise ProfilerEvidenceError(f"{label} identity or SHA-256 changed")


def capture(path: Path, label: str, *, executable: bool = False) -> Snapshot:
    path = path.absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if stat.S_ISLNK(current.lstat().st_mode):
            raise ProfilerEvidenceError(f"{label} path contains a symlink")
    path = path.resolve(strict=True)
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > MAX_INPUT_BYTES
        or (executable and before.st_mode & 0o111 == 0)
    ):
        raise ProfilerEvidenceError(f"{label} file identity is invalid")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    chunks: list[bytes] = []
    observed = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if identity(opened) != identity(before):
            raise ProfilerEvidenceError(f"{label} identity changed while opening")
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > MAX_INPUT_BYTES:
                raise ProfilerEvidenceError(f"{label} exceeds input bound")
            chunks.append(chunk)
            observed.update(chunk)
        if identity(os.fstat(descriptor)) != identity(before) or identity(path.lstat()) != identity(before):
            raise ProfilerEvidenceError(f"{label} identity changed while reading")
    finally:
        os.close(descriptor)
    return Snapshot(path, identity(before), observed.hexdigest(), b"".join(chunks))


def parse_json(snapshot: Snapshot, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ProfilerEvidenceError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            snapshot.data,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProfilerEvidenceError(f"non-finite JSON in {label}: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProfilerEvidenceError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ProfilerEvidenceError(f"{label} must be an object")
    return value


def validate_common(value: dict[str, Any], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in (
        "side", "binding_kind", "binding_id", "request_id", "implementation_id",
        "source_id", "candidate_id", "case_id",
    ):
        result[field] = identifier(value[field], f"{label}.{field}")
    for field in ("source_sha256", "case_sha256", "identity_sha256"):
        result[field] = digest(value[field], f"{label}.{field}")
    if result["side"] not in {"baseline", "candidate"}:
        raise ProfilerEvidenceError(f"{label}.side differs")
    if result["binding_kind"] not in {"run", "pair"}:
        raise ProfilerEvidenceError(f"{label}.binding_kind differs")
    if result["candidate_id"] != "sequence-output-direct-v1":
        raise ProfilerEvidenceError(f"{label}.candidate_id differs")
    return result


def derive_raw(value: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    exact(value, RAW_FIELDS, "raw profiler capture")
    if (
        value["schema_version"] != RAW_SCHEMA
        or value["status"] != "complete"
        or value["record_sha256"] != self_hash(value, "record_sha256")
        or value["timing_lane"] != LANE
        or value["measurement_eligible"] is not True
    ):
        raise ProfilerEvidenceError("raw profiler capture status/hash/lane differs")
    common = validate_common(value, "raw profiler capture")
    command = value["command"]
    if (
        not isinstance(command, list)
        or not command
        or len(command) > 64
        or any(not isinstance(item, str) or not item or len(item) > 4096 for item in command)
    ):
        raise ProfilerEvidenceError("raw profiler command is invalid")
    forbidden = ("prompt", "token", "completion", "generated", "secret")
    if any(any(fragment in item.lower() for fragment in forbidden) for item in command):
        raise ProfilerEvidenceError("raw profiler command may contain token or prompt material")
    if count(value["exit_code"], "raw profiler exit_code") != 0:
        raise ProfilerEvidenceError("raw profiler exit code differs")
    started = count(value["started_unix_ns"], "raw profiler started", positive=True)
    completed = count(value["completed_unix_ns"], "raw profiler completed", positive=True)
    if completed <= started:
        raise ProfilerEvidenceError("raw profiler timestamp order differs")
    samples = value["samples"]
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_SAMPLES:
        raise ProfilerEvidenceError("raw profiler sample count differs")
    components: list[float] = []
    full_models: list[float] = []
    peaks: list[int] = []
    fidelities: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ProfilerEvidenceError("raw profiler sample must be an object")
        exact(sample, SAMPLE_FIELDS, f"raw profiler samples[{index}]")
        if common["binding_kind"] == "run":
            components.append(positive_float(sample["component_ms"], "component_ms"))
            full_models.append(positive_float(sample["full_model_ms"], "full_model_ms"))
        elif sample["component_ms"] is not None or sample["full_model_ms"] is not None:
            raise ProfilerEvidenceError("pair profiler sample contains latency")
        peaks.append(count(sample["peak_vram_bytes"], "peak_vram_bytes", positive=True))
        fidelities.add(digest(sample["fidelity_binding_sha256"], "fidelity_binding_sha256"))
    if len(fidelities) != 1:
        raise ProfilerEvidenceError("raw profiler fidelity binding differs across samples")
    derived = {
        "command": list(command),
        "exit_code": 0,
        "started_unix_ns": started,
        "completed_unix_ns": completed,
        "sample_count": len(samples),
        "component_ms": math.fsum(components) / len(components) if components else None,
        "full_model_ms": math.fsum(full_models) / len(full_models) if full_models else None,
        "peak_vram_bytes": max(peaks),
        "fidelity_binding_sha256": next(iter(fidelities)),
    }
    return common, derived


def _validate_ref(value: Any, label: str, *, executable: bool = False) -> Snapshot:
    if not isinstance(value, dict):
        raise ProfilerEvidenceError(f"{label} reference must be an object")
    exact(value, REF_FIELDS, f"{label} reference")
    path_value = value["path"]
    if not isinstance(path_value, str) or not path_value or len(path_value) > 4096:
        raise ProfilerEvidenceError(f"{label}.path must be a bounded absolute path")
    path = Path(path_value)
    if not path.is_absolute():
        raise ProfilerEvidenceError(f"{label}.path must be absolute")
    snapshot = capture(path, label, executable=executable)
    if (
        str(snapshot.path) != path_value
        or snapshot.sha256 != digest(value["sha256"], f"{label}.sha256")
        or snapshot.identity[0] != count(value["device"], f"{label}.device")
        or snapshot.identity[1] != count(value["inode"], f"{label}.inode")
        or snapshot.identity[3] != 1
        or count(value["nlink"], f"{label}.nlink", positive=True) != 1
    ):
        raise ProfilerEvidenceError(f"{label} identity differs")
    return snapshot


def validate_observation(value: dict[str, Any]) -> dict[str, Any]:
    exact(value, OBSERVATION_FIELDS, "profiler observation")
    if (
        value["schema_version"] != OBSERVATION_SCHEMA
        or value["status"] != "complete"
        or value["record_sha256"] != self_hash(value, "record_sha256")
        or value["timing_lane"] != LANE
        or value["measurement_eligible"] is not True
    ):
        raise ProfilerEvidenceError("profiler observation status/hash/lane differs")
    common = validate_common(value, "profiler observation")
    profiler = _validate_ref(value["profiler"], "profiler executable", executable=True)
    raw = _validate_ref(value["raw_capture"], "raw profiler capture")
    parser = _validate_ref(value["parser"], "profiler parser")
    if parser.path != Path(__file__).resolve():
        raise ProfilerEvidenceError("profiler parser path differs")
    raw_common, derived = derive_raw(parse_json(raw, "raw profiler capture"))
    if common != raw_common:
        raise ProfilerEvidenceError("profiler observation raw binding differs")
    if value["command"] != derived["command"]:
        raise ProfilerEvidenceError("profiler observation derived command differs")
    if derived["command"][0] != str(profiler.path):
        raise ProfilerEvidenceError("profiler command executable differs")
    for field, expected in derived.items():
        if value[field] != expected:
            raise ProfilerEvidenceError(f"profiler observation derived {field} differs")
    if not isinstance(value["profiler_version"], str) or not value["profiler_version"] or len(value["profiler_version"]) > 4096:
        raise ProfilerEvidenceError("profiler version is invalid")
    profiler.verify("profiler executable", executable=True)
    raw.verify("raw profiler capture")
    parser.verify("profiler parser")
    return {**common, **derived, "timing_lane": LANE, "measurement_eligible": True}


def build(raw_path: Path, profiler_path: Path) -> tuple[dict[str, Any], list[Snapshot]]:
    raw = capture(raw_path, "raw profiler capture")
    profiler = capture(profiler_path, "profiler executable", executable=True)
    parser = capture(Path(__file__), "profiler parser")
    raw_value = parse_json(raw, "raw profiler capture")
    common, derived = derive_raw(raw_value)
    if derived["command"][0] != str(profiler.path):
        raise ProfilerEvidenceError("raw profiler command executable differs")
    version_result = subprocess.run(
        [str(profiler.path), "--version"], check=False, capture_output=True,
        text=True, timeout=30,
    )
    version = (version_result.stdout + version_result.stderr).strip()
    if version_result.returncode != 0 or not version or len(version) > 4096:
        raise ProfilerEvidenceError("profiler version probe failed")
    result: dict[str, Any] = {
        "schema_version": OBSERVATION_SCHEMA,
        "status": "complete",
        "record_sha256": None,
        **common,
        "timing_lane": LANE,
        "measurement_eligible": True,
        "profiler": profiler.reference(),
        "raw_capture": raw.reference(),
        "parser": parser.reference(),
        "profiler_version": version,
        **derived,
    }
    result["record_sha256"] = self_hash(result, "record_sha256")
    validate_observation(result)
    return result, [raw, profiler, parser]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-capture", required=True, type=Path)
    parser.add_argument("--profiler", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result, snapshots = build(args.raw_capture, args.profiler)
        for snapshot in snapshots:
            snapshot.verify("profiler evidence input", executable=snapshot.path == args.profiler.resolve())
        if args.output.exists() or args.output.is_symlink():
            raise ProfilerEvidenceError("refusing to overwrite profiler observation")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode("ascii") + b"\n"
        temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, args.output, follow_symlinks=False)
        except FileExistsError as error:
            raise ProfilerEvidenceError("refusing to overwrite profiler observation") from error
        temporary.unlink()
        return 0
    except (OSError, subprocess.SubprocessError, ProfilerEvidenceError) as error:
        print(f"profiler evidence error: {error}", file=sys.stderr)
        return 2
    finally:
        if "temporary" in locals() and temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
