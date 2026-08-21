#!/usr/bin/env python3
"""Aggregate the Phase 36 Session D performance/profile evidence.

The direct runners retain raw output outside the repository.  This controller
reads those reports, verifies their immutable identities and protocol, and
emits only a bounded, tracked-size summary.  It never turns a missing raw
artifact or an observer-lane mismatch into an ``unavailable`` performance
value: those are fail-closed errors.  ``unavailable`` is reserved for a
metric that the producer explicitly cannot provide (for example, a llama
monitor peak when only the wrapper's device-memory snapshot exists).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, canonical_bytes  # noqa: E402
import phase36_session_d_profile as profile  # noqa: E402


SCHEMA_VERSION = "phase36-mi300x-session-d-summary-v1"
TARGET = "gfx942"
GPU_UUID = "GPU-1228c84fe776f2f4"
MODEL_SIZE = "4B"
WARMUPS = 3
MEASURED = 10
CASES = ("short-odd", "32-32", "prefill-long", "decode-long", "long-10001")
LLAMA_CASES = ("short-odd", "32x32", "prefill-long", "decode-long", "long-10001")
LLAMA_COMMIT = "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70"
LLAMA_TAG = "b10453"
SOURCE_BASE_COMMIT = "faf39339d42c837c1ff899f90b03632ac5fe57af"
SOURCE_BASE_TREE = "274a07514f1338b715acc70057a56a3485431142"
ROCM_ROOT = "/opt/rocm"
ROCM_VERSION = "7.14.0"
SHORT_ODD = [1, 3, 17, 37, 73, 255, 256, 257, 2, 5, 11, 19, 23, 29, 31, 41, 43]
CASE_COUNTS = {
    "short-odd": (17, 17),
    "32-32": (32, 32),
    "32x32": (32, 32),
    "prefill-long": (1024, 128),
    "decode-long": (32, 256),
    "long-10001": (10_001, 2),
}
WEIGHTS = ("bf16", "fp8")
SLLM_ROWS = 10
LLAMA_ROWS = 5
METRICS = (
    "ttft_ns",
    "prefill_ns",
    "prefill_tokens_per_second",
    "tpot_ns",
    "decode_tokens_per_second",
    "e2e_ns",
)
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_RAW_BYTES = 128 * 1024 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE12 = ROOT / "ci/matrix/phase12-mi300x-summary-v1.json"
DEFAULT_PHASE35 = ROOT / "ci/matrix/phase35-attention-gdn-summary-v1.json"


class SessionDError(ContractError):
    """Malformed, incomplete, or identity-inconsistent Session D evidence."""


def _fail(message: str) -> None:
    raise SessionDError(message)


def _regular(path: Path, label: str, *, allow_empty: bool = False, max_bytes: int = MAX_RAW_BYTES) -> Path:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be a regular non-symlink file: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        _fail(f"cannot stat {label}: {exc}")
    if not allow_empty and size == 0:
        _fail(f"{label} is empty: {path}")
    if size > max_bytes:
        _fail(f"{label} exceeds its bounded size: {path}")
    return path


def sha256_file(path: Path, label: str = "raw file", *, allow_empty: bool = False, max_bytes: int = MAX_RAW_BYTES) -> str:
    path = _regular(path, label, allow_empty=allow_empty, max_bytes=max_bytes)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _fail(f"cannot read {label}: {exc}")
    return digest.hexdigest()


def _json_load(path: Path, label: str) -> Any:
    path = _regular(path, label)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label}: duplicate JSON key {key}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        _fail(f"{label}: non-finite JSON value {token}")

    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            _fail(f"{label}: JSON is oversized")
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
    except SessionDError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"{label}: malformed JSON: {exc}")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _fail(f"{label} is not finite numeric evidence")
    result = float(value)
    if positive and result <= 0:
        _fail(f"{label} must be positive")
    if result < 0:
        _fail(f"{label} must be nonnegative")
    return result


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} is not an integer")
    if positive and value <= 0:
        _fail(f"{label} must be positive")
    if value < 0:
        _fail(f"{label} must be nonnegative")
    return value


def _distribution(values: Sequence[float], label: str) -> dict[str, float | int]:
    if len(values) != MEASURED:
        _fail(f"{label} must have exactly {MEASURED} measured values")
    ordered = sorted(_number(value, f"{label} sample", positive=True) for value in values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    median = statistics.median(ordered)
    mad = statistics.median([abs(item - median) for item in ordered])
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": percentile(0.10),
        "median": float(median),
        "p90": percentile(0.90),
        "max": ordered[-1],
        "mad": float(mad),
    }


def _artifact(path: Path, label: str) -> dict[str, Any]:
    path = _regular(path, label, max_bytes=MAX_ARTIFACT_BYTES)
    return {"name": path.name, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path, label, max_bytes=MAX_ARTIFACT_BYTES)}


def _identity_name(value: Mapping[str, Any]) -> str | None:
    if isinstance(value.get("name"), str):
        return value["name"]
    if isinstance(value.get("path"), str):
        return Path(value["path"]).name
    return None


def _identity_path(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.exists():
        _fail(f"{label} does not exist as a non-symlink path: {path}")
    if path.is_file():
        return _artifact(path, label)
    if not path.is_dir():
        _fail(f"{label} is neither file nor directory: {path}")
    entries: list[dict[str, Any]] = []
    total_size = 0
    try:
        files = sorted(item for item in path.rglob("*") if item.is_file() and not item.is_symlink())
    except OSError as exc:
        _fail(f"cannot enumerate {label}: {exc}")
    if not files:
        _fail(f"{label} directory is empty: {path}")
    for item in files:
        identity = _artifact(item, f"{label} member")
        relative = str(item.relative_to(path))
        entries.append({"path": relative, "size_bytes": identity["size_bytes"], "sha256": identity["sha256"]})
        total_size += identity["size_bytes"]
    manifest_sha = _sha_json(entries)
    return {"name": path.name, "path": str(path.resolve()), "size_bytes": total_size, "sha256": manifest_sha, "member_count": len(entries)}


def _raw_digest(path: Path, expected: str, label: str, *, allow_empty: bool = False) -> dict[str, Any]:
    actual = sha256_file(path, label, allow_empty=allow_empty)
    if actual != expected:
        _fail(f"{label}: raw SHA-256 changed")
    return {"name": path.name, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": actual}


def _validate_raw_item(item: Any, label: str, *, allow_empty: bool = False) -> dict[str, Any]:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
        _fail(f"{label}: raw manifest item is malformed")
    path = _regular(Path(item["path"]), label, allow_empty=allow_empty)
    expected = item["sha256"]
    if not isinstance(expected, str) or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        _fail(f"{label}: raw digest is malformed")
    return _raw_digest(path, expected, label, allow_empty=allow_empty)


def _row_report_artifact(raw: Mapping[str, Any], label: str) -> dict[str, Any] | None:
    """Retain row.json when a producer emits it; raw manifest remains authoritative."""
    stdout = raw.get("stdout")
    if not isinstance(stdout, Mapping) or not isinstance(stdout.get("path"), str):
        _fail(f"{label}: stdout path is absent while locating row.json")
    row_path = Path(stdout["path"]).parent / "row.json"
    if not row_path.exists() or row_path.is_symlink():
        return None
    return _artifact(row_path, f"{label} row.json")


def _check_cleanup(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        _fail(f"{label}: cleanup evidence is absent")
    retryable = value.get("retryable_cleanup", value.get("cleanup_failures"))
    durable = value.get("durable_quarantine", 0)
    if retryable != 0 or durable != 0:
        _fail(f"{label}: cleanup is nonzero")
    if "terminal_zero" in value and value["terminal_zero"] is not True:
        _fail(f"{label}: terminal_zero is not true")
    if "all_requests_dropped" in value and value["all_requests_dropped"] is not True:
        _fail(f"{label}: all_requests_dropped is not true")


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _tokens(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        _fail(f"{label}: token IDs are malformed")
    return list(value)


def _expected_input_ids(case: str) -> list[int]:
    input_count, _output_count = CASE_COUNTS[case]
    if case == "long-10001":
        return [profile.EXPECTED_INPUT_ID] * input_count
    result = list(SHORT_ODD)
    result.extend(((index * 7919 + 41) % 248000) for index in range(len(result), input_count))
    return result


def _sample_derived(sample: Mapping[str, Any], label: str, input_count: int, output_count: int) -> dict[str, float]:
    derived = sample.get("derived")
    if not isinstance(derived, dict):
        _fail(f"{label}: derived timing is absent")
    ttft = _number(derived.get("ttft_ns"), f"{label} ttft_ns", positive=True)
    prefill = _number(derived.get("prefill_ns"), f"{label} prefill_ns", positive=True)
    prefill_tps = _number(derived.get("prefill_tokens_per_second"), f"{label} prefill_tokens_per_second", positive=True)
    e2e = _number(derived.get("e2e_ns"), f"{label} e2e_ns", positive=True)
    decode_tps = _number(derived.get("decode_tokens_per_second"), f"{label} decode_tokens_per_second", positive=True)
    decode_tokens = _integer(derived.get("decode_tokens"), f"{label} decode_tokens", positive=True)
    if decode_tokens != output_count - 1:
        _fail(f"{label}: decode token count is not output_tokens-1")
    tpot = derived.get("tpot_ns")
    if not isinstance(tpot, list) or not tpot:
        _fail(f"{label}: tpot_ns array is absent")
    scalars = [_number(item, f"{label} tpot_ns", positive=True) for item in tpot]
    if len(scalars) < output_count - 1:
        _fail(f"{label}: tpot_ns array is shorter than generated decode steps")
    return {
        "ttft_ns": ttft,
        "prefill_ns": prefill,
        "prefill_tokens_per_second": prefill_tps,
        "tpot_ns": float(statistics.median(scalars)),
        "decode_tokens_per_second": decode_tps,
        "e2e_ns": e2e,
    }


def _metric_block(samples: Sequence[Mapping[str, Any]], label: str, input_count: int, output_count: int) -> dict[str, Any]:
    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for index, sample in enumerate(samples):
        derived = _sample_derived(sample, f"{label} sample {index}", input_count, output_count)
        for metric in METRICS:
            values[metric].append(derived[metric])
    return {metric: _distribution(series, f"{label} {metric}") for metric, series in values.items()}


def _memory_metric(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {"state": "unavailable", "reason": label + " is absent"}
    return {"state": "available", "value": _integer(value, label)}


def _monitor_peak(item: Mapping[str, Any], label: str) -> tuple[int, int, int, list[dict[str, Any]]]:
    monitor = item.get("monitor")
    if not isinstance(monitor, Mapping):
        _fail(f"{label}: external monitor metadata is absent")
    if monitor.get("errors") != [] or monitor.get("cadence_ms") != 100:
        _fail(f"{label}: external monitor errors/cadence are invalid")
    raw = item.get("raw")
    path_value = raw.get("monitor_tsv", {}).get("path") if isinstance(raw, dict) else None
    if not isinstance(path_value, str) and isinstance(monitor, dict):
        path_value = monitor.get("tsv")
    if not isinstance(path_value, str):
        _fail(f"{label}: external monitor TSV path is absent")
    path = Path(path_value)
    if isinstance(raw, dict) and isinstance(raw.get("monitor_tsv"), dict):
        raw_item = raw["monitor_tsv"]
        if not isinstance(raw_item.get("sha256"), str):
            _fail(f"{label}: monitor raw digest is absent")
        retained = _validate_raw_item(raw_item, f"{label} monitor TSV")
    else:
        retained = _artifact(path, f"{label} monitor TSV")
    try:
        with path.open("r", newline="", encoding="ascii") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
    except (OSError, UnicodeError, csv.Error) as exc:
        _fail(f"{label}: cannot read monitor TSV: {exc}")
    if not rows:
        _fail(f"{label}: monitor TSV has no samples")
    if monitor.get("samples") != len(rows):
        _fail(f"{label}: external monitor sample count does not match TSV")
    memory = item.get("memory")
    baseline = memory.get("baseline") if isinstance(memory, Mapping) else None
    settled = memory.get("settled") if isinstance(memory, Mapping) else None
    if not isinstance(baseline, Mapping) or not isinstance(settled, Mapping):
        _fail(f"{label}: external baseline/settled memory evidence is absent")
    baseline_hbm = _integer(baseline.get("hbm_bytes"), f"{label} baseline HBM")
    baseline_gtt = _integer(baseline.get("gtt_bytes"), f"{label} baseline GTT")
    settled_hbm = _integer(settled.get("hbm_bytes"), f"{label} settled HBM")
    settled_gtt = _integer(settled.get("gtt_bytes"), f"{label} settled GTT")
    if settled.get("settled") is not True or settled_hbm != baseline_hbm or settled_gtt != baseline_gtt:
        _fail(f"{label}: post-process HBM/GTT did not return to baseline")
    peaks: list[int] = []
    samples: list[tuple[int, int, int]] = []
    previous_timestamp = -1
    for index, row in enumerate(rows):
        try:
            timestamp = _integer(int(row["timestamp_ns"]), f"{label} monitor timestamp {index}", positive=True)
            hbm = _integer(int(row["hbm_bytes"]), f"{label} monitor sample {index}")
            gtt = _integer(int(row["gtt_bytes"]), f"{label} monitor GTT sample {index}")
            if timestamp <= previous_timestamp:
                _fail(f"{label}: monitor timestamps are not strictly monotonic")
            previous_timestamp = timestamp
            peaks.append(hbm)
            samples.append((timestamp, hbm, gtt))
        except (KeyError, TypeError, ValueError) as exc:
            _fail(f"{label}: malformed monitor sample: {exc}")
    return max(peaks), baseline_hbm, settled_hbm, [retained]


def _validate_sllm_row(row: Any, expected_weight: str, expected_case: str, expected_models: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    label = f"sLLM {expected_weight}/{expected_case}"
    if not isinstance(row, dict) or row.get("state") != "PASS" or row.get("gpu_uuid") != GPU_UUID or row.get("device_index") != 0:
        _fail(f"{label}: row is not PASS")
    identity = row.get("row")
    input_count, output_count = CASE_COUNTS[expected_case]
    expected_input_ids = _expected_input_ids(expected_case)
    if not isinstance(identity, dict) or identity.get("weight") != expected_weight or identity.get("case_id") != expected_case or identity.get("model_size") != MODEL_SIZE or identity.get("target") != TARGET or identity.get("device_index") != 0 or identity.get("input_token_count") != input_count or identity.get("requested_output_tokens") != output_count:
        _fail(f"{label}: row identity mismatch")
    input_ids = _tokens(identity.get("input_token_ids"), f"{label} input")
    if input_ids != expected_input_ids:
        _fail(f"{label}: fixed input IDs drift")
    result = row.get("result")
    if not isinstance(result, dict) or result.get("benchmark_schema_version") != "engine-performance-direct-v1" or result.get("state") != "PASS" or result.get("lane") != "direct":
        _fail(f"{label}: direct result is not PASS")
    config = result.get("config")
    if not isinstance(config, dict) or config.get("input_token_ids") != input_ids or config.get("input_token_count") != input_count or config.get("max_new_tokens") != output_count or config.get("warmups") != WARMUPS or config.get("measured") != MEASURED or config.get("greedy") is not True or config.get("kv_cache_encoding") != "fp16" or config.get("lane") != "direct" or config.get("render") is not False or config.get("tokenizer") is not False:
        _fail(f"{label}: protocol/input configuration drift")
    audit = result.get("audit")
    if not isinstance(audit, dict) or audit.get("target") != TARGET or audit.get("selected_backend") != "hip" or audit.get("all_dispatches_hip") is not True or audit.get("fallback_used") is not False:
        _fail(f"{label}: HIP/fallback evidence is invalid")
    _check_cleanup(result.get("cleanup"), label)
    samples = result.get("measured", {}).get("samples") if isinstance(result.get("measured"), dict) else None
    if not isinstance(samples, list) or len(samples) != MEASURED or result.get("measured", {}).get("count") != MEASURED:
        _fail(f"{label}: measured sample count is not 10")
    generated: list[int] | None = None
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            _fail(f"{label}: sample {index} is malformed")
        tokens = sample.get("tokens")
        if not isinstance(tokens, dict) or _tokens(tokens.get("input_token_ids"), f"{label} sample input") != input_ids:
            _fail(f"{label}: sample input token drift")
        current = _tokens(tokens.get("generated_token_ids"), f"{label} sample output")
        visible = _tokens(tokens.get("visible_token_ids"), f"{label} sample visible output")
        if len(current) != output_count or current != visible:
            _fail(f"{label}: sample output shape mismatch")
        if generated is None:
            generated = current
        elif current != generated:
            _fail(f"{label}: measured output drift")
        sample_audit = sample.get("audit")
        if not isinstance(sample_audit, dict) or sample_audit.get("selected_backend") != "hip" or sample_audit.get("target") != TARGET or sample_audit.get("all_dispatches_hip") is not True or sample_audit.get("fallback_used") is not False:
            _fail(f"{label}: sample HIP/fallback evidence invalid")
        _check_cleanup(sample.get("cleanup"), f"{label} sample {index}")
    if generated is None:
        _fail(f"{label}: no generated token sequence")
    if expected_case == "long-10001" and expected_weight == "bf16" and generated != profile.EXPECTED_OUTPUT_IDS:
        _fail(f"{label}: 10001/2 output token drift")
    raw_items: list[dict[str, Any]] = []
    raw = row.get("raw")
    if not isinstance(raw, dict):
        _fail(f"{label}: raw manifest is absent")
    for name in ("stdout", "stderr"):
        raw_items.append(_validate_raw_item(raw.get(name), f"{label} raw {name}", allow_empty=name == "stderr"))
    row_artifact = _row_report_artifact(raw, label)
    if row_artifact is not None:
        raw_items.append(row_artifact)
    model_item = row.get("model")
    lock_item = row.get("lock")
    if not isinstance(model_item, dict) or model_item.get("sha256") != expected_models[expected_weight]["model"]["sha256"]:
        _fail(f"{label}: model identity drift")
    if not isinstance(lock_item, dict) or lock_item.get("sha256") != expected_models[expected_weight]["lock"]["sha256"]:
        _fail(f"{label}: lock identity drift")
    memory = result.get("memory")
    internal_resident = memory.get("resident_vram_bytes") if isinstance(memory, dict) else None
    internal_peak = memory.get("peak_vram_bytes") if isinstance(memory, dict) else None
    monitor_peak, monitor_baseline, monitor_settled, monitor_items = _monitor_peak(row, label)
    raw_items.extend(monitor_items)
    metrics = _metric_block(samples, label, input_count, output_count)
    return {
        "row_id": identity.get("row_id"),
        "weight": expected_weight,
        "case_id": expected_case,
        "input_tokens": len(input_ids),
        "output_tokens": output_count,
        "input_ids_sha256": _sha_json(input_ids),
        "output_ids": generated,
        "metrics": metrics,
        "memory": {
            "internal_resident_hbm_bytes": _memory_metric(internal_resident, label + " internal resident HBM"),
            "internal_peak_hbm_bytes": _memory_metric(internal_peak, label + " internal peak HBM"),
            "monitor_external_peak_hbm_bytes": {"state": "available", "value": monitor_peak},
            "monitor_external_baseline_hbm_bytes": {"state": "available", "value": monitor_baseline},
            "monitor_external_settled_hbm_bytes": {"state": "available", "value": monitor_settled},
        },
        "raw_artifacts": raw_items,
    }


def _llama_samples(document: Mapping[str, Any], label: str) -> tuple[list[int], list[Mapping[str, Any]]]:
    input_ids = _tokens(document.get("input_token_ids"), label + " input")
    measured = document.get("measured")
    if not isinstance(measured, dict) or measured.get("count") != MEASURED or not isinstance(measured.get("samples"), list) or len(measured["samples"]) != MEASURED:
        _fail(f"{label}: measured sample count is not 10")
    return input_ids, measured["samples"]


def _validate_llama_row(report: Mapping[str, Any], expected_model: dict[str, Any], expected_case: str, gpu_uuid: str) -> dict[str, Any]:
    label = f"llama {expected_case}"
    if report.get("state") != "PASS" or report.get("target") != TARGET or report.get("gpu_uuid") != gpu_uuid:
        _fail(f"{label}: row identity is not exact gfx942/GPU UUID")
    document = report.get("result")
    if not isinstance(document, dict):
        _fail(f"{label}: row direct result is absent")
    if not isinstance(document, dict) or document.get("schema_version") != "llama-phase36-session-d-v1" or document.get("state") != "PASS" or document.get("record_kind") != "result":
        _fail(f"{label}: wrapper JSON is not PASS")
    llama = document.get("llama")
    if not isinstance(llama, dict) or llama.get("commit") != LLAMA_COMMIT or llama.get("tag") != LLAMA_TAG:
        _fail(f"{label}: fixed llama commit/tag identity drift")
    for key in ("comparison_class", "e1_classification"):
        if key in document and document[key] != "E1_SYSTEM_EQUIVALENT":
            _fail(f"{label}: E1 artifact comparison is mislabeled")
    target = document.get("target")
    if not isinstance(target, dict) or target.get("exact") != TARGET or target.get("gpu_uuid") != gpu_uuid:
        _fail(f"{label}: exact target/GPU UUID mismatch")
    if document.get("case_id") != expected_case:
        _fail(f"{label}: case identity mismatch")
    input_count, output_count = CASE_COUNTS[expected_case]
    expected_input_ids = _expected_input_ids(expected_case)
    row_identity = report.get("row")
    if not isinstance(row_identity, dict) or row_identity.get("case_id") != expected_case or row_identity.get("input_token_count") != input_count or row_identity.get("requested_output_tokens") != output_count or row_identity.get("input_token_ids") != expected_input_ids:
        _fail(f"{label}: producer row shape/input identity drift")
    model = document.get("model")
    if not isinstance(model, dict) or model.get("sha256") != expected_model["model"]["sha256"] or model.get("weights") != "BF16" or model.get("kv") != "F16":
        _fail(f"{label}: model or KV identity mismatch")
    protocol = document.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("warmup_requests") != WARMUPS or protocol.get("measured_requests") != MEASURED or protocol.get("max_new_tokens") != output_count or protocol.get("n_ctx") != input_count + output_count or protocol.get("greedy") is not True or protocol.get("n_batch") != 10_001 or protocol.get("n_ubatch") != 512 or protocol.get("offload_kqv") is not True or protocol.get("op_offload") is not True:
        _fail(f"{label}: llama protocol drift")
    cleanup = document.get("cleanup")
    _check_cleanup(cleanup, label)
    audit = document.get("audit")
    offload = document.get("offload_evidence")
    if not isinstance(audit, dict) or audit.get("full_gpu_offload") is not True or audit.get("errors") != [] or not isinstance(offload, dict) or offload.get("visible_gpu_device_count") != 1:
        _fail(f"{label}: llama GPU offload evidence invalid")
    for key, value in _walk(document):
        if key.lower() in {"fallback_used", "cpu_fallback_used", "partial_offload"} and value is not False:
            _fail(f"{label}: llama fallback marker is not false")
    input_ids, samples = _llama_samples(document, label)
    if input_ids != expected_input_ids:
        _fail(f"{label}: fixed input IDs drift")
    generated: list[int] | None = None
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            _fail(f"{label}: sample {index} malformed")
        tokens = sample.get("tokens")
        if not isinstance(tokens, dict) or _tokens(tokens.get("input_token_ids"), f"{label} sample input") != input_ids:
            _fail(f"{label}: sample input drift")
        current = _tokens(tokens.get("generated_token_ids"), f"{label} sample output")
        visible = _tokens(tokens.get("visible_token_ids"), f"{label} sample visible output")
        if len(current) != output_count or visible != current:
            _fail(f"{label}: sample output shape/visibility drift")
        if generated is None:
            generated = current
        elif current != generated:
            _fail(f"{label}: output drift")
        derived = sample.get("derived")
        if not isinstance(derived, dict):
            _fail(f"{label}: sample derived timing absent")
    if generated is None:
        _fail(f"{label}: no output sequence")
    if len(generated) != output_count:
        _fail(f"{label}: output token count drift")
    observed = offload.get("observed")
    device_memory = observed.get("device_memory") if isinstance(observed, dict) else None
    resident = device_memory.get("observed_decrease_bytes") if isinstance(device_memory, dict) else None
    metrics = _metric_block(samples, label, len(input_ids), len(generated))
    monitor_peak, monitor_baseline, monitor_settled, monitor_items = _monitor_peak(report, label)
    raw_items = []
    raw = report.get("raw")
    if not isinstance(raw, dict):
        _fail(f"{label}: raw manifest is absent")
    for name in ("stdout", "stderr"):
        raw_items.append(_validate_raw_item(raw.get(name), f"{label} raw {name}", allow_empty=name == "stderr"))
    row_artifact = _row_report_artifact(raw, label)
    if row_artifact is not None:
        raw_items.append(row_artifact)
    return {
        "row_id": document.get("row_id"),
        "weight": "bf16",
        "case_id": expected_case,
        "input_tokens": len(input_ids),
        "output_tokens": len(generated),
        "input_ids_sha256": _sha_json(input_ids),
        "output_ids": generated,
        "metrics": metrics,
        "memory": {
            "internal_resident_hbm_bytes": _memory_metric(resident, label + " internal resident HBM"),
            "internal_peak_hbm_bytes": {"state": "unavailable", "reason": "llama wrapper exposes no allocator peak field"},
            "monitor_external_peak_hbm_bytes": {"state": "available", "value": monitor_peak},
            "monitor_external_baseline_hbm_bytes": {"state": "available", "value": monitor_baseline},
            "monitor_external_settled_hbm_bytes": {"state": "available", "value": monitor_settled},
        },
        "raw_artifacts": raw_items,
    }


def _load_llama_rows(llama_summary: Path, expected_model: dict[str, Any], expected_binary: dict[str, Any], gpu_uuid: str, llama_model_manifest: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = _json_load(llama_summary, "llama summary")
    if not isinstance(summary, dict) or summary.get("schema_version") != "phase36-session-d-llama-v1" or summary.get("state") != "PASS" or summary.get("target") != TARGET or summary.get("gpu_uuid") != gpu_uuid:
        _fail("llama summary schema/target/UUID is invalid")
    protocol = summary.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("warmups") != WARMUPS or protocol.get("measured") != MEASURED or protocol.get("n_batch") != 10_001 or protocol.get("n_ubatch") != 512 or protocol.get("weights") != "BF16" or protocol.get("kv") != "F16":
        _fail("llama summary protocol drifted")
    llama_identity = summary.get("llama")
    if not isinstance(llama_identity, dict) or llama_identity.get("commit") != LLAMA_COMMIT or llama_identity.get("tag") != LLAMA_TAG:
        _fail("llama summary fixed commit/tag identity drift")
    if summary.get("matrix", {}).get("row_count") != LLAMA_ROWS or summary.get("matrix", {}).get("cases") != list(LLAMA_CASES):
        _fail("llama summary case matrix is incomplete")
    binary = summary.get("binary")
    model = summary.get("model")
    if not isinstance(binary, dict) or binary.get("sha256") != expected_binary["sha256"] or binary.get("size_bytes") != expected_binary["size_bytes"] or _identity_name(binary) != expected_binary["name"]:
        _fail("llama wrapper binary identity drifted")
    if not isinstance(model, dict) or model.get("sha256") != expected_model["model"]["sha256"] or model.get("size_bytes") != expected_model["model"]["size_bytes"] or _identity_name(model) != expected_model["model"]["name"]:
        _fail("llama BF16 model identity drifted")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_artifacts: list[dict[str, Any]] = [{"name": llama_summary.name, "path": str(llama_summary.resolve()), "size_bytes": llama_summary.stat().st_size, "sha256": sha256_file(llama_summary, "llama summary")}]
    if llama_model_manifest is not None:
        raw_artifacts.append(llama_model_manifest)
    raw_rows = summary.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != LLAMA_ROWS:
        _fail("llama summary rows are missing or duplicated")
    for report in raw_rows:
        if not isinstance(report, dict) or not isinstance(report.get("row"), dict):
            _fail("llama summary row is malformed")
        producer_case = report["row"].get("case_id")
        case = "32-32" if producer_case == "32x32" else producer_case
        if producer_case not in LLAMA_CASES or case in seen:
            _fail(f"llama wrapper case is missing or duplicated: {producer_case}")
        seen.add(case)
        validated = _validate_llama_row(report, expected_model, producer_case, gpu_uuid)
        validated["case_id"] = case
        rows.append(validated)
        raw_artifacts.extend(validated["raw_artifacts"])
    if seen != set(CASES):
        _fail("llama wrapper case matrix is incomplete")
    return [next(row for row in rows if row["case_id"] == case) for case in CASES], raw_artifacts


def _load_sllm_rows(path: Path, expected_models: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    document = _json_load(path, "sLLM performance summary")
    if not isinstance(document, dict) or document.get("state") != "PASS" or document.get("schema_version") != "phase36-session-d-performance-v1":
        _fail("sLLM performance summary is not the expected PASS schema")
    protocol = document.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("warmups") != WARMUPS or protocol.get("measured") != MEASURED or protocol.get("greedy") is not True or protocol.get("kv_cache_encoding") != "fp16":
        _fail("sLLM performance protocol drifted")
    matrix = document.get("matrix")
    if document.get("target") != TARGET or document.get("gpu_uuid") != GPU_UUID or document.get("device_index") != 0 or document.get("model_size") != MODEL_SIZE or not isinstance(matrix, dict) or matrix.get("row_count") != SLLM_ROWS or matrix.get("weights") != list(WEIGHTS) or matrix.get("cases") != list(CASES):
        _fail("sLLM performance target or matrix count is invalid")
    raw_rows = document.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != SLLM_ROWS:
        _fail("sLLM performance rows are missing or duplicated")
    by_key: dict[tuple[str, str], Any] = {}
    for item in raw_rows:
        if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
            _fail("sLLM row is malformed")
        row = item["row"]
        key = (str(row.get("weight")), str(row.get("case_id")))
        if key in by_key:
            _fail(f"duplicate sLLM row: {key}")
        by_key[key] = item
    expected = {(weight, case) for weight in WEIGHTS for case in CASES}
    if set(by_key) != expected:
        _fail("sLLM matrix does not contain BF16/FP8 x five cases exactly")
    return [_validate_sllm_row(by_key[(weight, case)], weight, case, expected_models) for weight in WEIGHTS for case in CASES]


def _validate_source_identity(path: Path, binary: Mapping[str, Any]) -> dict[str, Any]:
    document = _json_load(path, "source identity")
    if not isinstance(document, dict) or document.get("schema_version") != "phase36-session-d-source-identity-v1":
        _fail("source identity schema is invalid")
    if document.get("base_commit") != SOURCE_BASE_COMMIT or document.get("base_tree") != SOURCE_BASE_TREE:
        _fail("source identity base commit/tree drifted")
    if document.get("sllm_binary_sha256") != binary["sha256"]:
        _fail("source identity is not bound to the sLLM binary")
    overrides = document.get("session_d_cli_overrides")
    expected_override_paths = {
        "crates/sllm-cli/src/benchmark.rs",
        "crates/sllm-cli/src/main.rs",
        "crates/sllm-cli/src/model.rs",
    }
    if not isinstance(overrides, dict) or set(overrides) != expected_override_paths or any(not isinstance(value, str) or len(value) != 64 for value in overrides.values()):
        _fail("source identity Session D override set is invalid")
    build = document.get("build")
    if not isinstance(build, dict) or build.get("rocm_root") != ROCM_ROOT or build.get("rocm_version") != ROCM_VERSION or build.get("hip_compiler") != ROCM_ROOT + "/bin/amdclang++" or build.get("logical_target") != TARGET:
        _fail("source identity build/toolchain tuple is invalid")
    return document


def _validate_llama_model_manifest(path: Path, llama_model: Mapping[str, Any], bf16_lock_path: Path) -> dict[str, str]:
    document = _json_load(path, "llama model manifest")
    if not isinstance(document, dict) or document.get("schema_version") != "phase5-p3-llama-cpp-artifacts-v1":
        _fail("llama model manifest schema is invalid")
    model = document.get("model")
    conversion = document.get("conversion")
    run = conversion.get("run") if isinstance(conversion, dict) else None
    args = run.get("args") if isinstance(run, dict) else None
    if not isinstance(model, dict) or model.get("repo_id") != "Qwen/Qwen3.5-4B" or model.get("resolved_revision") != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a":
        _fail("llama model manifest upstream identity is invalid")
    if not isinstance(args, list) or any(not isinstance(item, str) for item in args) or "--no-mtp" not in args or "--outtype" not in args or "bf16" not in args or run.get("result") != "PASS":
        _fail("llama model manifest conversion is not BF16 no-MTP PASS")
    if run.get("output_sha256") != llama_model["sha256"] or run.get("output_size_bytes") != llama_model["size_bytes"]:
        _fail("llama model manifest output identity drifted")
    bf16_lock = _json_load(bf16_lock_path, "BF16 derived lock")
    fingerprints = bf16_lock.get("source_lock_fingerprints") if isinstance(bf16_lock, dict) else None
    if not isinstance(fingerprints, list) or model.get("lock_fingerprint") not in fingerprints:
        _fail("llama and sLLM BF16 models do not share the locked upstream source")
    return {"repo_id": model["repo_id"], "revision": model["resolved_revision"]}


def _validate_profile_summary(path: Path, profile_raw_dir: Path) -> dict[str, Any]:
    document = _json_load(path, "Session D profile summary")
    if not isinstance(document, dict) or document.get("schema_version") != profile.SCHEMA_VERSION or document.get("state") != "PASS" or document.get("target") != TARGET:
        _fail("profile summary schema/target/state is invalid")
    kernel = document.get("kernel")
    if not isinstance(kernel, dict):
        _fail("profile kernel totals are empty")
    total_duration = _integer(kernel.get("total_duration_ns"), "profile total kernel duration", positive=True)
    total_calls = _integer(kernel.get("calls"), "profile total kernel calls", positive=True)
    if _integer(kernel.get("trace_dispatches"), "profile trace dispatches", positive=True) != total_calls:
        _fail("profile stats calls do not close against trace dispatches")
    categories = kernel.get("categories")
    if not isinstance(categories, list) or len(categories) != len(profile.CATEGORIES) or {item.get("category") for item in categories if isinstance(item, dict)} != set(profile.CATEGORIES):
        _fail("profile categories are incomplete or duplicated")
    duration = 0
    calls = 0
    names: set[str] = set()
    seen_categories: set[str] = set()
    for item in categories:
        if not isinstance(item, dict):
            _fail("profile category row is malformed")
        value = _integer(item.get("total_duration_ns"), "profile category duration")
        count = _integer(item.get("calls"), "profile category calls")
        category = item.get("category")
        if category not in profile.CATEGORIES or category in seen_categories:
            _fail("profile categories are missing, duplicated, or unknown")
        seen_categories.add(category)
        if value <= 0 or count <= 0:
            _fail("profile category duration/calls are not positive")
        duration += value
        calls += count
        kernel_names = item.get("kernel_names")
        if not isinstance(kernel_names, list) or any(not isinstance(name, str) for name in kernel_names):
            _fail("profile category names are malformed")
        if names.intersection(kernel_names):
            _fail("profile kernel name appears in more than one category")
        names.update(kernel_names)
        share = _number(item.get("device_time_share"), "profile category device share")
        expected_share = value / total_duration
        if abs(share - expected_share) > 1.0e-12:
            _fail("profile category device share does not match duration total")
    if seen_categories != set(profile.CATEGORIES):
        _fail("profile category set is incomplete")
    if duration != total_duration or calls != total_calls:
        _fail("profile category totals do not close")
    share_sum = _number(kernel.get("category_share_sum"), "profile category share sum")
    if abs(share_sum - 1.0) > 1.0e-9:
        _fail("profile category shares do not close")
    external = document.get("kernel_external")
    if not isinstance(external, dict) or external.get("state") not in {"available", "unavailable"}:
        _fail("profile kernel_external state is invalid")
    interval = _integer(external.get("kernel_interval_union_ns"), "profile kernel interval union")
    if external.get("state") == "available":
        host = _integer(external.get("host_wall_ns"), "profile host wall")
        external_ns = _integer(external.get("external_ns"), "profile external wall")
        if external_ns + interval != host or external_ns < 0:
            _fail("profile kernel_external closure is invalid")
    elif not isinstance(external.get("reason"), str) or not external["reason"]:
        _fail("profile unavailable reason is absent")
    raw = document.get("raw_sha256")
    if not isinstance(raw, dict) or set(raw) != {"kernel_stats", "kernel_trace", "hip_api_stats", "memory_copy_stats", "execution_json"}:
        _fail("profile raw digest manifest is incomplete")
    for name, item in raw.items():
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            _fail(f"profile raw digest is malformed: {name}")
        raw_path = profile_raw_dir / Path(item["path"]).name
        if not raw_path.is_file() or raw_path.is_symlink() or sha256_file(raw_path, f"profile raw {name}") != item["sha256"]:
            _fail(f"profile raw digest changed or missing: {name}")
    expected_manifest = _sha_json(raw)
    if document.get("raw_manifest_sha256") != expected_manifest:
        _fail("profile raw manifest digest does not close")
    return {
        "schema_version": document["schema_version"],
        "raw_manifest_sha256": expected_manifest,
        "categories": categories,
        "kernel_external": external,
        "raw_artifacts": [{"name": name, "path": str((profile_raw_dir / Path(item["path"]).name).resolve()), "size_bytes": (profile_raw_dir / Path(item["path"]).name).stat().st_size, "sha256": item["sha256"]} for name, item in raw.items()],
    }


def _reference(path: Path | None, label: str) -> dict[str, Any]:
    if path is None:
        return {"state": "unavailable", "reason": f"{label} summary path was not supplied"}
    if not path.is_file() or path.is_symlink():
        return {"state": "unavailable", "reason": f"{label} summary is not present"}
    document = _json_load(path, label + " summary")
    digest = sha256_file(path, label + " summary")
    if label == "Phase 12":
        if not isinstance(document, dict) or document.get("state") != "PASS":
            return {"state": "unavailable", "reason": "Phase 12 summary is not PASS", "summary_sha256": digest}
        rows = document.get("performance", {}).get("rows") if isinstance(document, dict) else None
        if not isinstance(rows, list):
            return {"state": "unavailable", "reason": "Phase 12 performance rows are absent", "summary_sha256": digest}
        has_long = any(isinstance(row, dict) and row.get("case") == "long-10001" for row in rows)
        reason = "tracked Phase 12 summary loaded"
        if not has_long:
            reason += "; no 10001/2 row is not synthesized"
        return {"state": "available", "reason": reason, "summary_sha256": digest, "rows": rows}
    if not isinstance(document, dict) or document.get("state") != "COMPLETE":
        return {"state": "unavailable", "reason": "Phase 35 summary is not COMPLETE", "summary_sha256": digest}
    performance = document.get("performance", {}).get("combined_final_source")
    if not isinstance(performance, dict):
        return {"state": "unavailable", "reason": "Phase 35 current-target performance rows are absent", "summary_sha256": digest}
    return {"state": "available", "reason": "tracked Phase 35 current V620/R9700 values loaded", "summary_sha256": digest, "rows": performance}


def _phase12_changes(sllm_rows: list[dict[str, Any]], phase12: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compare current sLLM medians with the tracked Phase 12 rows.

    Phase 12 predates the 10001/2 case, so that row is emitted explicitly as
    unavailable.  Latencies are converted from milliseconds to nanoseconds;
    throughput and memory remain in their producer units.  Memory comparisons
    intentionally use the external monitor values while the row retains the
    separate internal-vs-external memory evidence.
    """
    entries: list[dict[str, Any]] = []
    phase_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    if phase12.get("state") == "available":
        raw_rows = phase12.get("rows")
        if isinstance(raw_rows, list):
            for row in raw_rows:
                if not isinstance(row, Mapping):
                    continue
                dtype = str(row.get("dtype", "")).upper()
                weight = {"BF16": "bf16", "FNUZ FP8": "fp8", "FP8": "fp8"}.get(dtype)
                case = row.get("case")
                if weight is not None and isinstance(case, str):
                    phase_rows[(weight, "32-32" if case == "32x32" else case)] = row
    by_key = {(row["weight"], row["case_id"]): row for row in sllm_rows}
    metric_specs = (
        ("ttft_ns", "ttft_ms_median", "lower_is_better", 1_000_000.0, "metrics"),
        ("e2e_ns", "e2e_ms_median", "lower_is_better", 1_000_000.0, "metrics"),
        ("prefill_tokens_per_second", "prefill_tps_median", "higher_is_better", 1.0, "metrics"),
        ("decode_tokens_per_second", "decode_tps_median", "higher_is_better", 1.0, "metrics"),
        ("resident_hbm_bytes", "resident_bytes", "lower_is_better", 1.0, "memory"),
        ("peak_hbm_bytes", "peak_bytes", "lower_is_better", 1.0, "memory"),
    )
    for weight in WEIGHTS:
        for case in CASES:
            baseline = phase_rows.get((weight, case))
            if baseline is None:
                entries.append({
                    "weight": weight,
                    "case_id": case,
                    "state": "unavailable",
                    "reason": ("Phase 12 tracked summary has no 10001/2 row" if case == "long-10001" else str(phase12.get("reason", "Phase 12 baseline is unavailable"))),
                })
                continue
            current = by_key[(weight, case)]
            metric_rows: list[dict[str, Any]] = []
            for metric, baseline_key, direction, scale, source_kind in metric_specs:
                if source_kind == "metrics":
                    current_value = current["metrics"][metric]["median"]
                else:
                    memory_key = "monitor_external_settled_hbm_bytes" if metric == "resident_hbm_bytes" else "monitor_external_peak_hbm_bytes"
                    current_metric = current["memory"][memory_key]
                    if current_metric.get("state") != "available":
                        _fail(f"Phase 12 comparison {weight}/{case} {metric}: current monitor value is unavailable")
                    current_value = current_metric["value"]
                baseline_value = _number(baseline.get(baseline_key), f"Phase 12 {weight}/{case} {baseline_key}", positive=True) * scale
                current_value = _number(current_value, f"current {weight}/{case} {metric}", positive=True)
                metric_rows.append({
                    "metric": metric,
                    "direction": direction,
                    "current_value": current_value,
                    "phase12_value": baseline_value,
                    "ratio_current_over_phase12": current_value / baseline_value,
                    "percent_change": (current_value / baseline_value - 1.0) * 100.0,
                })
            entries.append({"weight": weight, "case_id": case, "state": "available", "metrics": metric_rows})
    return entries


def _phase35_changes(sllm_rows: list[dict[str, Any]], phase35: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compare the current MI300X BF16 10001/2 median with Phase 35 peers."""
    current = next(
        (row for row in sllm_rows if row["weight"] == "bf16" and row["case_id"] == "long-10001"),
        None,
    )
    if current is None:
        _fail("current BF16 long-10001 row is absent")
    current_ns = _number(current["metrics"]["e2e_ns"]["median"], "current BF16 long-10001 e2e", positive=True)
    rows = phase35.get("rows") if phase35.get("state") == "available" else None
    changes: list[dict[str, Any]] = []
    for target in ("gfx1030", "gfx1201"):
        peer = rows.get(target) if isinstance(rows, Mapping) else None
        if not isinstance(peer, Mapping) or "candidate_ns" not in peer:
            changes.append({
                "target": target,
                "state": "unavailable",
                "reason": str(phase35.get("reason", "Phase 35 candidate is unavailable")),
            })
            continue
        phase35_ns = _number(peer["candidate_ns"], f"Phase 35 {target} candidate_ns", positive=True)
        changes.append({
            "target": target,
            "state": "available",
            "metric": "e2e_ns",
            "direction": "lower_is_better",
            "current_mi300x_value": current_ns,
            "phase35_value": phase35_ns,
            "ratio_current_over_phase35": current_ns / phase35_ns,
            "percent_change": (current_ns / phase35_ns - 1.0) * 100.0,
        })
    return changes


def _ratios(sllm_rows: list[dict[str, Any]], llama_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    llama_by_case = {row["case_id"]: row for row in llama_rows}
    output: list[dict[str, Any]] = []
    for row in sllm_rows:
        if row["weight"] != "bf16":
            continue
        peer = llama_by_case[row["case_id"]]
        metrics: list[dict[str, Any]] = []
        for metric in METRICS:
            direction = "higher_is_better" if metric in {"prefill_tokens_per_second", "decode_tokens_per_second"} else "lower_is_better"
            left = row["metrics"][metric]["median"]
            right = peer["metrics"][metric]["median"]
            if right <= 0:
                _fail(f"llama {row['case_id']} {metric} median is nonpositive")
            metrics.append({"metric": metric, "direction": direction, "sllm_over_llama_ratio": left / right})
        output.append({"case_id": row["case_id"], "metrics": metrics})
    return output


def _fp8_ratios(sllm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case = {(row["weight"], row["case_id"]): row for row in sllm_rows}
    output: list[dict[str, Any]] = []
    for case in CASES:
        bf16 = by_case[("bf16", case)]
        fp8 = by_case[("fp8", case)]
        metrics: list[dict[str, Any]] = []
        for metric in METRICS:
            direction = "higher_is_better" if metric in {"prefill_tokens_per_second", "decode_tokens_per_second"} else "lower_is_better"
            denominator = bf16["metrics"][metric]["median"]
            if denominator <= 0:
                _fail(f"BF16 {case} {metric} median is nonpositive")
            metrics.append({"metric": metric, "direction": direction, "fp8_over_bf16_ratio": fp8["metrics"][metric]["median"] / denominator})
        output.append({"case_id": case, "metrics": metrics})
    return output


def aggregate_session_d(
    *,
    sllm_summary: Path,
    llama_summary: Path,
    profile_summary: Path,
    profile_raw_dir: Path,
    binary: Path,
    llama_binary: Path,
    bf16_model: Path,
    bf16_lock: Path,
    fp8_model: Path,
    fp8_lock: Path,
    llama_model: Path,
    source: Path,
    rocm_root: str,
    rocm_version: str,
    gpu_uuid: str,
    llama_model_manifest: Path | None = None,
    phase12_summary: Path | None = DEFAULT_PHASE12,
    phase35_summary: Path | None = DEFAULT_PHASE35,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if TARGET != "gfx942" or gpu_uuid != GPU_UUID:
        _fail(f"exact gfx942 and GPU UUID {GPU_UUID} are required")
    if rocm_root != ROCM_ROOT or rocm_version != ROCM_VERSION:
        _fail(f"exact ROCm tuple {ROCM_ROOT} / {ROCM_VERSION} is required")
    if llama_model_manifest is None:
        _fail("llama model manifest is required for E1 model lineage")
    expected_models = {
        "bf16": {"model": _artifact(bf16_model, "BF16 model"), "lock": _artifact(bf16_lock, "BF16 lock")},
        "fp8": {"model": _artifact(fp8_model, "FP8 model"), "lock": _artifact(fp8_lock, "FP8 lock")},
    }
    llama_binary_identity = _artifact(llama_binary, "llama wrapper binary")
    llama_model_identity = _artifact(llama_model, "llama BF16 model")
    llama_manifest_identity = _artifact(llama_model_manifest, "llama model manifest")
    upstream_identity = _validate_llama_model_manifest(llama_model_manifest, llama_model_identity, bf16_lock)
    binary_identity = _artifact(binary, "sLLM binary")
    _validate_source_identity(source, binary_identity)
    identity = {
        "binary": binary_identity,
        "llama_binary": llama_binary_identity,
        "bf16_model": expected_models["bf16"]["model"],
        "bf16_lock": expected_models["bf16"]["lock"],
        "fp8_model": expected_models["fp8"]["model"],
        "fp8_lock": expected_models["fp8"]["lock"],
        "llama_bf16_model": llama_model_identity,
        "source": _identity_path(source, "source identity"),
        "rocm": {"root": rocm_root, "version": rocm_version},
        "gpu": {"target": TARGET, "uuid": gpu_uuid},
    }
    identity["llama_model_manifest"] = llama_manifest_identity
    sllm_rows = _load_sllm_rows(sllm_summary, expected_models)
    llama_rows, llama_artifacts = _load_llama_rows(llama_summary, {"model": llama_model_identity}, llama_binary_identity, gpu_uuid, llama_manifest_identity)
    profile_report = _validate_profile_summary(profile_summary, profile_raw_dir)
    phase12 = _reference(phase12_summary, "Phase 12")
    phase35 = _reference(phase35_summary, "Phase 35")
    sllm_artifacts = [item for row in sllm_rows for item in row["raw_artifacts"]]
    profile_artifacts = profile_report["raw_artifacts"]
    raw_artifacts = sllm_artifacts + llama_artifacts + profile_artifacts + [{"name": sllm_summary.name, "path": str(sllm_summary.resolve()), "size_bytes": sllm_summary.stat().st_size, "sha256": sha256_file(sllm_summary, "sLLM summary")}, {"name": profile_summary.name, "path": str(profile_summary.resolve()), "size_bytes": profile_summary.stat().st_size, "sha256": sha256_file(profile_summary, "profile summary")}]
    raw_manifest_sha = _sha_json(raw_artifacts)
    artifact_differences = {
        "engine": "sLLM HIP direct runtime versus llama.cpp wrapper",
        "sllm_binary": identity["binary"]["sha256"],
        "llama_wrapper": identity["llama_binary"]["sha256"],
        "sllm_bf16_model": {"name": identity["bf16_model"]["name"], "size_bytes": identity["bf16_model"]["size_bytes"], "sha256": identity["bf16_model"]["sha256"]},
        "llama_bf16_model": {"name": identity["llama_bf16_model"]["name"], "size_bytes": identity["llama_bf16_model"]["size_bytes"], "sha256": identity["llama_bf16_model"]["sha256"]},
        "same_upstream_repo_id_revision": upstream_identity,
        "different_gguf_tensor_set_converter": "distinct sLLM and llama.cpp GGUF SHA-256/size/converter identities; the validated llama manifest records BF16 --no-mtp conversion",
        "strict_identical": False,
    }
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "state": "PASS",
        "recorded_at": _utc_now(),
        "target": TARGET,
        "identity": identity,
        "protocol": {"warmups": WARMUPS, "measured": MEASURED, "cases": list(CASES), "sllm_rows": SLLM_ROWS, "llama_rows": LLAMA_ROWS, "input_output": "case-specific, long-10001 is 10001/2", "kv_cache_encoding": "fp16", "greedy": True},
        "sllm": {"rows": sllm_rows, "row_count": len(sllm_rows)},
        "llama": {"rows": llama_rows, "row_count": len(llama_rows)},
        "profile": profile_report,
        "comparisons": {"e1_bf16": {"classification": "E1_SYSTEM_EQUIVALENT", "strict_identical": False, "artifact_differences": artifact_differences, "rows": _ratios(sllm_rows, llama_rows)}, "fp8_vs_bf16": {"llama_comparison": "not_applicable", "rows": _fp8_ratios(sllm_rows)}},
        "historical": {
            "phase12": phase12,
            "phase35_current_v620_r9700": phase35,
            "phase12_changes": _phase12_changes(sllm_rows, phase12),
            "phase35_changes": _phase35_changes(sllm_rows, phase35),
        },
        "raw_artifacts": raw_artifacts,
        "raw_manifest_sha256": raw_manifest_sha,
    }
    if output_path is not None:
        if output_path.exists() or output_path.is_symlink():
            _fail(f"refusing to overwrite summary output: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(canonical_bytes(summary))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sllm-summary", type=Path, required=True)
    parser.add_argument("--llama-summary", type=Path, required=True)
    parser.add_argument("--profile-summary", type=Path, required=True)
    parser.add_argument("--profile-raw-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--llama-binary", type=Path, required=True)
    parser.add_argument("--bf16-model", type=Path, required=True)
    parser.add_argument("--bf16-lock", type=Path, required=True)
    parser.add_argument("--fp8-model", type=Path, required=True)
    parser.add_argument("--fp8-lock", type=Path, required=True)
    parser.add_argument("--llama-model", type=Path, required=True)
    parser.add_argument("--llama-model-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rocm-root", required=True)
    parser.add_argument("--rocm-version", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--phase12-summary", type=Path, default=DEFAULT_PHASE12)
    parser.add_argument("--phase35-summary", type=Path, default=DEFAULT_PHASE35)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = aggregate_session_d(
            sllm_summary=args.sllm_summary, llama_summary=args.llama_summary, profile_summary=args.profile_summary, profile_raw_dir=args.profile_raw_dir,
            binary=args.binary, llama_binary=args.llama_binary, bf16_model=args.bf16_model, bf16_lock=args.bf16_lock, fp8_model=args.fp8_model, fp8_lock=args.fp8_lock,
            llama_model=args.llama_model, llama_model_manifest=args.llama_model_manifest,
            source=args.source, rocm_root=args.rocm_root, rocm_version=args.rocm_version, gpu_uuid=args.gpu_uuid,
            phase12_summary=args.phase12_summary, phase35_summary=args.phase35_summary, output_path=args.output,
        )
    except SessionDError as exc:
        print(f"phase36 Session D summary: FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
