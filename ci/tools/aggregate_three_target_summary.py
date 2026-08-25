#!/usr/bin/env python3
"""Build the bounded, tracked comparison for the three canonical GPU targets.

The Phase 49/50/51 aggregators retain the detailed producer evidence outside
the repository.  This tool is the small publication boundary for that
evidence: it validates the exact target and seven-row matrix, copies only
bounded row metrics/digests, and records correctness/resource disposition and
known constraints.  It never copies token arrays, event timelines, or raw
monitor traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from statistics import median
from typing import Any, Mapping, NoReturn


CASE_SPECS: tuple[tuple[str, int, int], ...] = (
    ("short-odd", 17, 17),
    ("32-32", 32, 32),
    ("prefill-long", 1024, 128),
    ("decode-long", 32, 256),
    ("long-10001", 10_001, 2),
    ("long-100000", 100_000, 2),
    ("decode-20000", 32, 20_000),
)
CASE_IDS = tuple(case_id for case_id, _, _ in CASE_SPECS)
METRICS = ("e2e_ns", "ttft_ns", "tpot_ns")
TARGETS: dict[str, dict[str, Any]] = {
    "gfx1030": {
        "family": "RDNA2",
        "phase": "phase49",
        "schema_version": "phase49-v620-summary-v1",
    },
    "gfx1201": {
        "family": "RDNA4",
        "phase": "phase50",
        "schema_version": "phase50-r9700-summary-v1",
    },
    "gfx942": {
        "family": "CDNA3",
        "phase": "phase51",
        "schema_version": "phase51-mi300x-summary-v1",
    },
}
TARGET_ORDER = tuple(TARGETS)
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_REASON_CHARS = 2048
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class ThreeTargetError(RuntimeError):
    """Malformed, stale, or unsafe target summary."""


def _fail(message: str) -> NoReturn:
    raise ThreeTargetError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _reject_constant(token: str) -> NoReturn:
    _fail(f"non-finite JSON constant {token}")


def load_json(path: Path) -> dict[str, Any]:
    """Load a regular JSON object while rejecting duplicate/non-finite data."""
    if path.is_symlink() or not path.is_file():
        _fail(f"summary is not a regular non-symlink file: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        _fail(f"cannot read summary {path}: {exc}")
    if not data or len(data) > MAX_JSON_BYTES:
        _fail(f"summary is empty or oversized: {path}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{path}: duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates, parse_constant=_reject_constant)
    except ThreeTargetError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"{path}: malformed JSON: {exc}")
    if not isinstance(value, dict):
        _fail(f"{path}: summary is not an object")
    return value


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _text(value: Any, label: str, *, max_chars: int = MAX_REASON_CHARS) -> str:
    if not isinstance(value, str) or not value or len(value) > max_chars:
        _fail(f"{label}: expected a non-empty bounded string")
    return value


def _bool_or_none(value: Any, label: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        _fail(f"{label}: expected boolean or null")
    return value


def _finite_number(value: Any, label: str, *, positive: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label}: expected a number")
    converted = float(value)
    if not math.isfinite(converted) or (positive and converted <= 0):
        _fail(f"{label}: expected a finite {'positive ' if positive else ''}number")
    return value


def _digest_or_none(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        _fail(f"{label}: expected a SHA-256 digest or null")
    return value


def _source_failure_records(target: str, source: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Return bounded failure/constraint records without copying raw evidence."""
    records: list[dict[str, str]] = []
    source_failures = source.get("failures")
    if source_failures is not None:
        if not isinstance(source_failures, list):
            _fail(f"{target}: failures must be an array")
        for index, failure in enumerate(source_failures):
            if not isinstance(failure, dict):
                _fail(f"{target}: failures[{index}] is not an object")
            case_id = failure.get("case_id", "summary")
            engine = failure.get("engine", "summary")
            kind = failure.get("kind", "failure")
            reason = failure.get("reason", "unspecified failure")
            if not isinstance(case_id, str) or not case_id or len(case_id) > 128:
                _fail(f"{target}: failures[{index}] case_id is invalid")
            if not isinstance(engine, str) or not engine or len(engine) > 64:
                _fail(f"{target}: failures[{index}] engine is invalid")
            if not isinstance(kind, str) or not kind or len(kind) > 64:
                _fail(f"{target}: failures[{index}] kind is invalid")
            records.append({"id": f"failure-{target}-{case_id}-{engine}", "target": target, "scope": case_id, "reason": _text(f"{kind}: {reason}", f"{target}: failures[{index}] reason")})
    for row in rows:
        row_failures = row.get("failures")
        if row_failures is None:
            continue
        if not isinstance(row_failures, dict):
            _fail(f"{target}/{row.get('case_id')}: failures must be an object")
        for engine, failure in row_failures.items():
            if failure is None:
                continue
            if not isinstance(failure, dict):
                _fail(f"{target}/{row.get('case_id')}/{engine}: failure is not an object")
            kind = failure.get("kind", "failure")
            reason = failure.get("reason", "unspecified failure")
            if not isinstance(kind, str) or not kind or len(kind) > 64:
                _fail(f"{target}/{row.get('case_id')}/{engine}: kind is invalid")
            records.append({"id": f"failure-{target}-{row.get('case_id')}-{engine}", "target": target, "scope": str(row.get("case_id")), "reason": _text(f"{kind}: {reason}", f"{target}/{row.get('case_id')}/{engine} reason")})
    return records


def _custom_constraints(target: str, source: Mapping[str, Any]) -> list[dict[str, str]]:
    values = source.get("known_constraints", [])
    if values is None:
        return []
    if not isinstance(values, list):
        _fail(f"{target}: known_constraints must be an array")
    result: list[dict[str, str]] = []
    for index, value in enumerate(values):
        if isinstance(value, str):
            result.append({"id": f"constraint-{target}-{index + 1}", "target": target, "scope": "summary", "reason": _text(value, f"{target}: known_constraints[{index}]")})
            continue
        if not isinstance(value, dict):
            _fail(f"{target}: known_constraints[{index}] is not an object or string")
        identifier = value.get("id", f"constraint-{target}-{index + 1}")
        scope = value.get("scope", "summary")
        reason = value.get("reason")
        if not isinstance(identifier, str) or not identifier or len(identifier) > 128:
            _fail(f"{target}: known_constraints[{index}] id is invalid")
        if not isinstance(scope, str) or not scope or len(scope) > 128:
            _fail(f"{target}: known_constraints[{index}] scope is invalid")
        result.append({"id": identifier, "target": target, "scope": scope, "reason": _text(reason, f"{target}: known_constraints[{index}] reason")})
    return result


def _normalise_stats(value: Any, label: str) -> dict[str, int | float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        _fail(f"{label}: metrics must be an object or null")
    result: dict[str, int | float] = {}
    for field in ("median", "mad", "count", "min", "max"):
        if field not in value:
            _fail(f"{label}: missing {field}")
        result[field] = _finite_number(value[field], f"{label}.{field}", positive=field in {"median", "min", "max"})
    if isinstance(result["count"], float) and not result["count"].is_integer():
        _fail(f"{label}.count: expected an integer")
    if int(result["count"]) < 1 or int(result["count"]) > 10:
        _fail(f"{label}.count: outside bounded range")
    if float(result["mad"]) < 0:
        _fail(f"{label}.mad: expected non-negative")
    return result


def _normalise_row(target: str, row: Mapping[str, Any], expected: tuple[str, int, int], index: int) -> dict[str, Any]:
    case_id, input_count, output_count = expected
    if row.get("case_id") != case_id or row.get("input_token_count") != input_count or row.get("requested_output_tokens") != output_count:
        _fail(f"{target}: row {index} does not match frozen case {case_id}")
    protocol = row.get("protocol")
    if not isinstance(protocol, dict):
        _fail(f"{target}/{case_id}: protocol is absent")
    protocol_fields = {field: protocol.get(field) for field in ("warmups", "measured", "context_length", "ignore_eos")}
    if not isinstance(protocol_fields["warmups"], int) or not isinstance(protocol_fields["measured"], int) or not isinstance(protocol_fields["context_length"], int) or not isinstance(protocol_fields["ignore_eos"], bool):
        _fail(f"{target}/{case_id}: protocol is malformed")
    row_ids = row.get("row_ids")
    if not isinstance(row_ids, dict) or not all(isinstance(row_ids.get(engine), str) and row_ids[engine] for engine in ("sllm", "llama")):
        _fail(f"{target}/{case_id}: row_ids are malformed")
    sample_count = row.get("measured_sample_count")
    if not isinstance(sample_count, dict) or not all(isinstance(sample_count.get(engine), int) and 0 <= sample_count[engine] <= 10 for engine in ("sllm", "llama")):
        _fail(f"{target}/{case_id}: measured_sample_count is malformed")
    tokens = row.get("tokens")
    if not isinstance(tokens, dict):
        _fail(f"{target}/{case_id}: tokens are absent")
    token_result = {
        "input_sha256": _digest_or_none(tokens.get("input_sha256"), f"{target}/{case_id}.tokens.input_sha256"),
        "generated_equal": _bool_or_none(tokens.get("generated_equal"), f"{target}/{case_id}.tokens.generated_equal"),
        "visible_equal": _bool_or_none(tokens.get("visible_equal"), f"{target}/{case_id}.tokens.visible_equal"),
        "stop_equal": _bool_or_none(tokens.get("stop_equal"), f"{target}/{case_id}.tokens.stop_equal"),
    }
    for engine in ("sllm", "llama"):
        for name in ("generated_sha256", "visible_sha256", "stop_sha256"):
            value = tokens.get(name)
            if not isinstance(value, dict) or engine not in value:
                _fail(f"{target}/{case_id}.tokens.{name}: missing {engine}")
            token_result.setdefault(name, {})[engine] = _digest_or_none(value[engine], f"{target}/{case_id}.tokens.{name}.{engine}")
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        _fail(f"{target}/{case_id}: metrics are absent")
    normal_metrics: dict[str, Any] = {}
    for engine in ("sllm", "llama"):
        engine_metrics = metrics.get(engine)
        if engine_metrics is None:
            normal_metrics[engine] = None
            continue
        if not isinstance(engine_metrics, dict):
            _fail(f"{target}/{case_id}.metrics.{engine}: expected object or null")
        normal_metrics[engine] = {metric: _normalise_stats(engine_metrics.get(metric), f"{target}/{case_id}.metrics.{engine}.{metric}") for metric in METRICS}
    gates = row.get("gates")
    if not isinstance(gates, dict):
        _fail(f"{target}/{case_id}: gates are absent")
    normal_gates: dict[str, Any] = {}
    for metric in METRICS:
        gate = gates.get(metric)
        if gate is None:
            normal_gates[metric] = None
            continue
        if not isinstance(gate, dict):
            _fail(f"{target}/{case_id}.gates.{metric}: expected object or null")
        normal_gates[metric] = {
            "sllm_median": _finite_number(gate.get("sllm_median"), f"{target}/{case_id}.gates.{metric}.sllm_median", positive=True),
            "sllm_mad": _finite_number(gate.get("sllm_mad"), f"{target}/{case_id}.gates.{metric}.sllm_mad"),
            "llama_median": _finite_number(gate.get("llama_median"), f"{target}/{case_id}.gates.{metric}.llama_median", positive=True),
            "llama_mad": _finite_number(gate.get("llama_mad"), f"{target}/{case_id}.gates.{metric}.llama_mad"),
            "limit": _finite_number(gate.get("limit"), f"{target}/{case_id}.gates.{metric}.limit", positive=True),
            "pass": _bool_or_none(gate.get("pass"), f"{target}/{case_id}.gates.{metric}.pass"),
        }
    failures = row.get("failures")
    normal_failures: dict[str, Any] | None = None
    if failures is not None:
        if not isinstance(failures, dict):
            _fail(f"{target}/{case_id}: failures must be an object")
        normal_failures = {}
        for engine, failure in failures.items():
            if engine not in {"sllm", "llama"}:
                _fail(f"{target}/{case_id}: unknown failure engine {engine!r}")
            if failure is None:
                normal_failures[engine] = None
                continue
            if not isinstance(failure, dict):
                _fail(f"{target}/{case_id}/{engine}: failure is not an object")
            kind = failure.get("kind", "failure")
            reason = failure.get("reason", "unspecified failure")
            if not isinstance(kind, str) or not kind or len(kind) > 64:
                _fail(f"{target}/{case_id}/{engine}: kind is invalid")
            normal_failures[engine] = {"kind": kind, "reason": _text(reason, f"{target}/{case_id}/{engine} reason")}
    has_failure = isinstance(normal_failures, dict) and any(value is not None for value in normal_failures.values())
    if any(normal_metrics[engine] is None for engine in ("sllm", "llama")) and not has_failure:
        _fail(f"{target}/{case_id}: missing metrics require an explicit failure record")
    return {
        "case_id": case_id,
        "input_token_count": input_count,
        "requested_output_tokens": output_count,
        "protocol": protocol_fields,
        "row_ids": {"sllm": row_ids["sllm"], "llama": row_ids["llama"]},
        "measured_sample_count": {"sllm": sample_count["sllm"], "llama": sample_count["llama"]},
        "tokens": token_result,
        "metrics": normal_metrics,
        "gates": normal_gates,
        "row_state": "FAIL" if has_failure else "PASS",
        "failures": normal_failures if has_failure else None,
    }


def _file_sha256(path: Path) -> str:
    """Hash an evidence file without copying it into the tracked summary."""
    if path.is_symlink() or not path.is_file():
        _fail(f"evidence is not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        _fail(f"cannot hash evidence {path}: {exc}")
    return digest.hexdigest()


def _raw_metric_stats(values: list[Any], label: str) -> dict[str, int | float]:
    if not values:
        _fail(f"{label}: empty measured distribution")
    numbers: list[float] = []
    for index, value in enumerate(values):
        number = _finite_number(value, f"{label}[{index}]", positive=True)
        numbers.append(float(number))
    middle = float(median(numbers))
    return {
        "median": middle,
        "mad": float(median([abs(value - middle) for value in numbers])),
        "count": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def _raw_stop_key(value: Any, label: str) -> tuple[Any, ...]:
    if not isinstance(value, dict):
        _fail(f"{label}: stop is absent")
    kind = value.get("kind")
    token_id = value.get("token_id")
    version = value.get("version")
    reason_version = value.get("reason_version")
    if kind not in {"max_new_tokens", "stop_token"} or isinstance(version, bool) or not isinstance(version, int) or version != 1:
        _fail(f"{label}: stop identity is malformed")
    if token_id is not None and (isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0):
        _fail(f"{label}: stop token is malformed")
    if reason_version is not None and (isinstance(reason_version, bool) or not isinstance(reason_version, int) or reason_version != 1):
        _fail(f"{label}: stop reason version is stale")
    return kind, token_id, version, reason_version


def _raw_result_metrics(result: Mapping[str, Any], target: str, engine: str, expected: tuple[str, int, int]) -> dict[str, Any]:
    case_id, input_count, output_count = expected
    row = result.get("row")
    # The llama Phase 49 result keeps row identity in the envelope and only
    # repeats a subset of it under result.row.  Validate counts when present;
    # _raw_engine_row validates the complete producer row above.
    if row is not None and (not isinstance(row, dict) or ("input_token_count" in row and row.get("input_token_count") != input_count) or ("requested_output_tokens" in row and row.get("requested_output_tokens") != output_count)):
        _fail(f"{target}/{engine}/{case_id}: result row shape is stale")
    samples_group = result.get("measured")
    if not isinstance(samples_group, dict) or not isinstance(samples_group.get("samples"), list):
        _fail(f"{target}/{engine}/{case_id}: measured samples are absent")
    samples = samples_group["samples"]
    expected_count = 3 if case_id in {"long-100000", "decode-20000"} else 10
    if len(samples) != expected_count or samples_group.get("count") != expected_count:
        _fail(f"{target}/{engine}/{case_id}: measured sample count is stale")
    metric_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    token_records: list[tuple[list[int], list[int], list[int], tuple[Any, ...]]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            _fail(f"{target}/{engine}/{case_id}: sample {index} is malformed")
        tokens = sample.get("tokens")
        if not isinstance(tokens, dict):
            _fail(f"{target}/{engine}/{case_id}: sample {index} tokens are absent")
        input_ids = tokens.get("input_token_ids")
        generated = tokens.get("generated_token_ids")
        visible = tokens.get("visible_token_ids")
        if not all(isinstance(value, list) and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value) for value in (input_ids, generated, visible)):
            _fail(f"{target}/{engine}/{case_id}: sample {index} token sequence is malformed")
        if len(input_ids) != input_count or len(generated) != output_count or visible != generated:
            _fail(f"{target}/{engine}/{case_id}: sample {index} token shape differs")
        derived = sample.get("derived")
        if not isinstance(derived, dict):
            _fail(f"{target}/{engine}/{case_id}: sample {index} derived metrics are absent")
        e2e = _finite_number(derived.get("e2e_ns"), f"{target}/{engine}/{case_id}/e2e_ns", positive=True)
        ttft = _finite_number(derived.get("ttft_ns"), f"{target}/{engine}/{case_id}/ttft_ns", positive=True)
        metric_values["e2e_ns"].append(float(e2e))
        metric_values["ttft_ns"].append(float(ttft))
        raw_tpot = derived.get("tpot_ns")
        if not isinstance(raw_tpot, list) or len(raw_tpot) != output_count - 1 or not raw_tpot:
            _fail(f"{target}/{engine}/{case_id}: sample {index} tpot_ns is empty")
        tpot = [float(_finite_number(value, f"{target}/{engine}/{case_id}/tpot_ns", positive=True)) for value in raw_tpot]
        events = sample.get("events")
        if not isinstance(events, dict):
            _fail(f"{target}/{engine}/{case_id}: sample {index} timing events are absent")
        first_ns = events.get("first_token_ns")
        request_ns = events.get("request_start_ns")
        cleanup_ns = events.get("cleanup_ns", events.get("cleanup_complete_ns"))
        later = events.get("later_token_publications_ns", events.get("token_publications_ns"))
        if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (first_ns, request_ns, cleanup_ns)) or not isinstance(later, list) or len(later) != output_count - 1 or not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in later):
            _fail(f"{target}/{engine}/{case_id}: sample {index} timing events are malformed")
        if float(ttft) != first_ns - request_ns or float(e2e) != cleanup_ns - request_ns:
            _fail(f"{target}/{engine}/{case_id}: sample {index} derived timing disagrees with events")
        previous = first_ns
        for measured_tpot, publication in zip(tpot, later):
            if measured_tpot != publication - previous:
                _fail(f"{target}/{engine}/{case_id}: sample {index} TPOT disagrees with events")
            previous = publication
        metric_values["tpot_ns"].append(float(median(tpot)))
        token_records.append((input_ids, generated, visible, _raw_stop_key(sample.get("stop"), f"{target}/{engine}/{case_id}/sample-{index}")))
    first = token_records[0]
    if any(record != first for record in token_records[1:]):
        _fail(f"{target}/{engine}/{case_id}: measured token or stop output is not deterministic")
    row_id = (row or {}).get("row_id") or result.get("row_id")
    if not isinstance(row_id, str) or not row_id or len(row_id) > 256:
        _fail(f"{target}/{engine}/{case_id}: producer row_id is malformed")
    return {
        "row_id": row_id,
        "input_ids": first[0],
        "generated": first[1],
        "visible": first[2],
        "stop": first[3],
        "stats": {metric: _raw_metric_stats(values, f"{target}/{engine}/{case_id}/{metric}") for metric, values in metric_values.items()},
        "sample_count": expected_count,
    }


def _raw_engine_row(target: str, engine: str, envelope: Mapping[str, Any], expected: tuple[str, int, int]) -> dict[str, Any]:
    case_id, input_count, output_count = expected
    if envelope.get("state") != "PASS":
        _fail(f"{target}/{engine}/{case_id}: producer state is not PASS")
    result = envelope.get("result") if engine == "llama" or "result" in envelope else envelope
    if not isinstance(result, dict) or result.get("state") != "PASS":
        _fail(f"{target}/{engine}/{case_id}: result state is not PASS")
    row = envelope.get("row")
    if not isinstance(row, dict):
        row = result.get("row")
    if not isinstance(row, dict) or row.get("case_id") != case_id:
        _fail(f"{target}/{engine}/{case_id}: producer row identity is stale")
    protocol = {field: row.get(field) for field in ("warmups", "measured", "context_length", "ignore_eos")}
    expected_warmups = 1 if case_id in {"long-100000", "decode-20000"} else 3
    expected_measured = 3 if case_id in {"long-100000", "decode-20000"} else 10
    expected_protocol = {"warmups": expected_warmups, "measured": expected_measured, "context_length": 131072 if case_id in {"long-100000", "decode-20000"} else input_count + output_count, "ignore_eos": case_id == "decode-20000"}
    if (not isinstance(protocol["warmups"], int) or isinstance(protocol["warmups"], bool) or not isinstance(protocol["measured"], int) or isinstance(protocol["measured"], bool) or not isinstance(protocol["context_length"], int) or isinstance(protocol["context_length"], bool) or not isinstance(protocol["ignore_eos"], bool) or protocol != expected_protocol):
        _fail(f"{target}/{engine}/{case_id}: producer protocol is stale")
    metrics = _raw_result_metrics(result, target, engine, expected)
    return {"row": row, "result": result, **metrics}


def _phase49_row_sources(normal_path: Path, long_path: Path, decode_path: Path, llama_path: Path, artifact_normal: str | None, artifact_long: str | None, llama_artifact: str | None) -> list[dict[str, Any]]:
    normal_sha = _file_sha256(normal_path)
    long_sha = _file_sha256(long_path)
    decode_sha = _file_sha256(decode_path)
    llama_sha = _file_sha256(llama_path)
    result: list[dict[str, Any]] = []
    for case_id in CASE_IDS:
        if case_id in {"short-odd", "32-32", "prefill-long", "decode-long", "long-10001"}:
            result.append({"case_id": case_id, "sllm": {"path": str(normal_path.resolve()), "sha256": normal_sha}, "llama": {"path": str(llama_path.resolve()), "sha256": llama_sha}, "artifact_sha256": artifact_normal, "llama_artifact_sha256": llama_artifact, "identity_scope": "final-adopted", "comparability": "comparable", "notes": "sLLM row from the adopted Phase 49 final-normal5 producer; llama.cpp row from the seven-row control."})
        elif case_id == "long-100000":
            result.append({"case_id": case_id, "sllm": {"path": str(long_path.resolve()), "sha256": long_sha}, "llama": {"path": str(llama_path.resolve()), "sha256": llama_sha}, "artifact_sha256": artifact_long, "llama_artifact_sha256": llama_artifact, "identity_scope": "non-adopted-v2", "comparability": "non-comparable", "notes": "sLLM long-prefill-v2 evidence is retained as a diagnostic; the candidate was rejected and uses a different binary from final-normal5."})
        else:
            result.append({"case_id": case_id, "sllm": {"path": str(decode_path.resolve()), "sha256": decode_sha}, "llama": {"path": str(llama_path.resolve()), "sha256": llama_sha}, "artifact_sha256": None, "llama_artifact_sha256": llama_artifact, "identity_scope": "binary-sha-unavailable", "comparability": "non-comparable", "notes": "sLLM direct decode-20000 evidence has no recorded binary SHA; no binary identity is inferred."})
    return result


def _producer_artifact_sha(summary: Mapping[str, Any], label: str) -> str | None:
    binary = summary.get("binary")
    if binary is None:
        return None
    if not isinstance(binary, Mapping):
        _fail(f"{label}: binary metadata is malformed")
    return _digest_or_none(binary.get("sha256"), f"{label} binary SHA")


def build_phase49_split_summary(normal_path: Path, long_path: Path, decode_path: Path, llama_path: Path) -> dict[str, Any]:
    """Build an explicit per-row Phase 49 source without inventing one binary."""
    normal = load_json(normal_path)
    long = load_json(long_path)
    decode = load_json(decode_path)
    llama = load_json(llama_path)
    if normal.get("schema_version") != "phase49-v620-sllm-v1" or normal.get("target") != "gfx1030" or normal.get("state") != "PASS":
        _fail("Phase 49 normal5 producer identity is stale")
    if long.get("schema_version") != "phase49-v620-sllm-v1" or long.get("target") != "gfx1030" or long.get("state") != "PASS":
        _fail("Phase 49 long100000 producer identity is stale")
    if llama.get("schema_version") != "phase49-v620-llama-v1" or llama.get("target") != "gfx1030" or llama.get("state") != "PASS":
        _fail("Phase 49 llama producer identity is stale")
    if not isinstance(normal.get("gpu_uuid"), str) or normal.get("gpu_uuid") != long.get("gpu_uuid") or normal.get("gpu_uuid") != llama.get("gpu_uuid"):
        _fail("Phase 49 split evidence GPU identity is inconsistent")
    if decode.get("benchmark_schema_version") != "engine-performance-direct-v2" or decode.get("state") != "PASS":
        _fail("Phase 49 decode20k direct evidence identity is stale")
    decode_ids = decode.get("identities")
    if not isinstance(decode_ids, dict) or decode_ids.get("engine") != "sllm" or decode_ids.get("target") != "gfx1030" or decode_ids.get("backend") != "hip":
        _fail("Phase 49 decode20k direct evidence target is stale")
    normal_rows = normal.get("rows")
    long_rows = long.get("rows")
    llama_rows = llama.get("rows")
    if not isinstance(normal_rows, list) or len(normal_rows) != 5 or not isinstance(long_rows, list) or len(long_rows) != 1 or not isinstance(llama_rows, list) or len(llama_rows) != 7:
        _fail("Phase 49 split evidence does not contain 5+1 sLLM and 7 llama rows")
    normal_by_case = {item.get("row", {}).get("case_id"): item for item in normal_rows if isinstance(item, dict) and isinstance(item.get("row"), dict)}
    long_by_case = {item.get("row", {}).get("case_id"): item for item in long_rows if isinstance(item, dict) and isinstance(item.get("row"), dict)}
    llama_by_case = {item.get("row", {}).get("case_id"): item for item in llama_rows if isinstance(item, dict) and isinstance(item.get("row"), dict)}
    if set(normal_by_case) != set(CASE_IDS[:5]) or set(long_by_case) != {"long-100000"} or set(llama_by_case) != set(CASE_IDS):
        _fail("Phase 49 split evidence case set is stale")
    decode_row = decode.get("row")
    if not isinstance(decode_row, dict) or decode_row.get("input_token_count") != 32 or decode_row.get("requested_output_tokens") != 20000:
        _fail("Phase 49 decode20k direct row shape is stale")
    # The direct report predates the final wrapper row name; map only the
    # frozen case label while retaining the original evidence path and row id.
    decode_envelope = {"state": decode.get("state"), "row": {"row_id": decode_row.get("row_id"), "case_id": "decode-20000", "input_token_count": 32, "requested_output_tokens": 20000, "warmups": decode.get("config", {}).get("warmups"), "measured": decode.get("config", {}).get("measured"), "context_length": decode.get("config", {}).get("context_length"), "ignore_eos": decode.get("config", {}).get("ignore_eos")}, "result": dict(decode)}
    decode_envelope["result"]["row"] = dict(decode_row)
    decode_envelope["result"]["row"]["case_id"] = "decode-20000"
    sllm_rows: dict[str, dict[str, Any]] = {}
    for case_id, expected in zip(CASE_IDS[:5], CASE_SPECS[:5]):
        sllm_rows[case_id] = _raw_engine_row("gfx1030", "sllm", normal_by_case[case_id], expected)
    sllm_rows["long-100000"] = _raw_engine_row("gfx1030", "sllm", long_by_case["long-100000"], CASE_SPECS[5])
    sllm_rows["decode-20000"] = _raw_engine_row("gfx1030", "sllm", decode_envelope, CASE_SPECS[6])
    llama_result_rows: dict[str, dict[str, Any]] = {}
    for case_id, expected in zip(CASE_IDS, CASE_SPECS):
        llama_result_rows[case_id] = _raw_engine_row("gfx1030", "llama", llama_by_case[case_id], expected)
    rows: list[dict[str, Any]] = []
    for case_id, input_count, output_count in CASE_SPECS:
        left = sllm_rows[case_id]
        right = llama_result_rows[case_id]
        if left["input_ids"] != right["input_ids"]:
            _fail(f"gfx1030/{case_id}: sLLM/llama input token sequence differs")
        stop_equal = left["stop"] == right["stop"]
        if left["stop"][3] != right["stop"][3]:
            # llama Phase 49 producer omits reason_version; retain the
            # semantic stop comparison but make the protocol mismatch visible.
            stop_equal = None
        gates: dict[str, Any] = {}
        for metric in METRICS:
            lstat, rstat = left["stats"][metric], right["stats"][metric]
            limit = float(rstat["median"]) + max(float(lstat["mad"]), float(rstat["mad"]))
            gates[metric] = {"sllm_median": lstat["median"], "sllm_mad": lstat["mad"], "llama_median": rstat["median"], "llama_mad": rstat["mad"], "limit": limit, "pass": None if metric == "tpot_ns" and output_count < 17 else float(lstat["median"]) <= limit}
        sllm_row = left["row"]
        llama_row = right["row"]
        rows.append({
            "case_id": case_id,
            "input_token_count": input_count,
            "requested_output_tokens": output_count,
            "protocol": {"warmups": sllm_row["warmups"], "measured": sllm_row["measured"], "context_length": sllm_row["context_length"], "ignore_eos": sllm_row["ignore_eos"]},
            "row_ids": {"sllm": left["row_id"], "llama": right["row_id"]},
            "measured_sample_count": {"sllm": left["sample_count"], "llama": right["sample_count"]},
            "tokens": {"input_sha256": _sha(left["input_ids"]), "generated_sha256": {"sllm": _sha(left["generated"]), "llama": _sha(right["generated"])}, "visible_sha256": {"sllm": _sha(left["visible"]), "llama": _sha(right["visible"])}, "stop_sha256": {"sllm": _sha(left["stop"]), "llama": _sha(right["stop"])}, "generated_equal": left["generated"] == right["generated"], "visible_equal": left["visible"] == right["visible"], "stop_equal": stop_equal},
            "metrics": {"sllm": left["stats"], "llama": right["stats"]},
            "gates": gates,
            "failures": None,
        })
    artifact_normal = _producer_artifact_sha(normal, "Phase 49 normal5")
    artifact_long = _producer_artifact_sha(long, "Phase 49 long100000")
    llama_artifact = _producer_artifact_sha(llama, "Phase 49 llama")
    return {
        "schema_version": "phase49-v620-summary-v1",
        "state": "PASS",
        "target": "gfx1030",
        "gpu_uuid": normal.get("gpu_uuid"),
        "gpu_bdf": "0000:03:00.0",
        "matrix": {"cases": list(CASE_IDS), "row_count": 7},
        "rows": rows,
        "_identity_scope": "per-row-mixed",
        "_row_sources": _phase49_row_sources(normal_path, long_path, decode_path, llama_path, artifact_normal, artifact_long, llama_artifact),
        "known_constraints": [
            {"id": "phase49-mixed-sllm-identity", "target": "gfx1030", "scope": "identity", "reason": f"Phase 49 seven-row sLLM evidence is per-row mixed: first five use final adopted binary {artifact_normal or 'unavailable'}, long-100000 uses a different long-prefill-v2 binary {artifact_long or 'unavailable'}, and decode-20000 has no recorded binary SHA."},
            {"id": "phase49-long100000-non-adopted-v2", "target": "gfx1030", "scope": "long-100000", "reason": "long-prefill-v2 evidence is retained as a non-adopted diagnostic and is not comparable as the final adopted Phase 49 binary."},
            {"id": "phase49-decode20000-binary-unavailable", "target": "gfx1030", "scope": "decode-20000", "reason": "The direct sLLM decode-20000 report has no binary SHA; no binary identity is inferred."},
            {"id": "phase49-llama-stop-reason-version-legacy", "target": "gfx1030", "scope": "all-rows", "reason": "The Phase 49 llama.cpp producer records stop version=1 without reason_version; the protocol mismatch is retained and not patched in evidence."},
        ],
    }


def _target_status(target: str, source: Mapping[str, Any], rows: list[Mapping[str, Any]], constraints: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    correctness_bad_kinds = {"correctness", "token_mismatch", "oracle_mismatch", "schema", "crash", "timeout", "fallback"}
    resource_bad_kinds = {"oom", "resource", "memory", "cleanup", "crash", "timeout", "fallback"}
    correctness = "PASS"
    resources = "PASS"
    for constraint in constraints:
        kind = constraint["reason"].split(":", 1)[0].lower()
        if kind in correctness_bad_kinds:
            correctness = "FAIL"
        if kind in resource_bad_kinds:
            resources = "FAIL"
    return (
        {"status": correctness, "evidence": "bounded-seven-row-summary", "constraint_count": sum(1 for item in constraints if item["target"] in {target, "all"} and item["reason"].split(":", 1)[0].lower() in correctness_bad_kinds)},
        {"status": resources, "evidence": "bounded-seven-row-summary", "constraint_count": sum(1 for item in constraints if item["target"] in {target, "all"} and item["reason"].split(":", 1)[0].lower() in resource_bad_kinds)},
    )


def _normalise_evidence_ref(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        _fail(f"{label}: evidence reference is not an object")
    path = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path, str) or not path or len(path) > 4096:
        _fail(f"{label}.path: expected a bounded non-empty string")
    if not isinstance(digest, str) or SHA_RE.fullmatch(digest) is None:
        _fail(f"{label}.sha256: expected a SHA-256 digest")
    return {"path": path, "sha256": digest}


def _normalise_row_sources(target: str, source: Mapping[str, Any], source_path: str) -> tuple[str, list[dict[str, Any]]]:
    identity_scope = source.get("_identity_scope", "single-summary")
    if identity_scope not in {"single-summary", "per-row-mixed"}:
        _fail(f"{target}: identity scope is invalid")
    raw_sources = source.get("_row_sources")
    if raw_sources is None:
        if identity_scope == "per-row-mixed":
            _fail(f"{target}: per-row-mixed identity requires explicit row_sources")
        # A conventional aggregate producer has one source object for both
        # engines.  Keep the per-row shape in the publication schema so that
        # consumers never need a second interpretation of row provenance.
        digest = _sha(source)
        evidence = {"path": source_path, "sha256": digest}
        raw_sources = [
            {
                "case_id": case_id,
                "sllm": evidence,
                "llama": evidence,
                "artifact_sha256": None,
                "llama_artifact_sha256": None,
                "identity_scope": "aggregate-summary",
                "comparability": "comparable",
                "notes": "Both engine rows were supplied by one validated aggregate producer summary.",
            }
            for case_id, _, _ in CASE_SPECS
        ]
    if not isinstance(raw_sources, list) or len(raw_sources) != len(CASE_SPECS):
        _fail(f"{target}: row_sources must contain exactly seven entries")
    normalised: list[dict[str, Any]] = []
    for index, (expected_case, _, _) in enumerate(CASE_SPECS):
        value = raw_sources[index]
        if not isinstance(value, Mapping) or value.get("case_id") != expected_case:
            _fail(f"{target}: row_sources[{index}] does not match frozen case {expected_case}")
        artifact = _digest_or_none(value.get("artifact_sha256"), f"{target}/{expected_case}.artifact_sha256")
        llama_artifact = _digest_or_none(value.get("llama_artifact_sha256"), f"{target}/{expected_case}.llama_artifact_sha256")
        row_scope = value.get("identity_scope")
        if row_scope not in {"final-adopted", "non-adopted-v2", "binary-sha-unavailable", "llama-control", "aggregate-summary"}:
            _fail(f"{target}/{expected_case}: row identity scope is invalid")
        if identity_scope == "per-row-mixed" and row_scope == "aggregate-summary":
            _fail(f"{target}/{expected_case}: mixed identity cannot use aggregate-summary row scope")
        if identity_scope == "single-summary" and row_scope != "aggregate-summary":
            _fail(f"{target}/{expected_case}: single-summary identity cannot use per-row scope")
        comparability = value.get("comparability")
        if comparability not in {"comparable", "non-comparable"}:
            _fail(f"{target}/{expected_case}: row comparability is invalid")
        normalised.append(
            {
                "case_id": expected_case,
                "sllm": _normalise_evidence_ref(value.get("sllm"), f"{target}/{expected_case}.sllm"),
                "llama": _normalise_evidence_ref(value.get("llama"), f"{target}/{expected_case}.llama"),
                "artifact_sha256": artifact,
                "llama_artifact_sha256": llama_artifact,
                "identity_scope": row_scope,
                "comparability": comparability,
                "notes": _text(value.get("notes"), f"{target}/{expected_case}.notes"),
            }
        )
    return identity_scope, normalised


def aggregate_target_summaries(summaries: Mapping[str, Mapping[str, Any]], *, sources: Mapping[str, str] | None = None, known_constraints: list[Mapping[str, str]] | None = None) -> dict[str, Any]:
    """Validate and aggregate one final seven-row summary per canonical target."""
    if set(summaries) != set(TARGETS):
        _fail(f"expected exactly targets {', '.join(TARGET_ORDER)}")
    sources = sources or {target: f"<memory:{target}>" for target in TARGET_ORDER}
    extra_constraints = known_constraints or []
    all_targets: list[dict[str, Any]] = []
    correctness_by_target: list[dict[str, Any]] = []
    resources_by_target: list[dict[str, Any]] = []
    all_constraints: list[dict[str, str]] = []
    normalised_extra_constraints: list[dict[str, str]] = []
    for item in extra_constraints:
        if not isinstance(item, Mapping):
            _fail("known constraint is not an object")
        target = item.get("target", "all")
        scope = item.get("scope", "summary")
        identifier = item.get("id")
        reason = item.get("reason")
        if not isinstance(target, str) or target not in TARGETS and target != "all":
            _fail(f"known constraint target is invalid: {target}")
        if not isinstance(identifier, str) or not identifier or len(identifier) > 128:
            _fail("known constraint id is invalid")
        if not isinstance(scope, str) or not scope or len(scope) > 128:
            _fail("known constraint scope is invalid")
        constraint = {"id": identifier, "target": target, "scope": scope, "reason": _text(reason, "known constraint reason")}
        normalised_extra_constraints.append(constraint)
        all_constraints.append(constraint)
    for target in TARGET_ORDER:
        source = summaries[target]
        if not isinstance(source, Mapping):
            _fail(f"{target}: summary is not an object")
        expected = TARGETS[target]
        if source.get("schema_version") != expected["schema_version"]:
            _fail(f"{target}: unexpected schema_version {source.get('schema_version')!r}")
        if source.get("target") != target:
            _fail(f"{target}: exact target identity is missing")
        if source.get("state") not in {"PASS", "FAIL"}:
            _fail(f"{target}: source state is invalid")
        for identity_field in ("gpu_uuid", "gpu_bdf"):
            if not isinstance(source.get(identity_field), str) or not source[identity_field]:
                _fail(f"{target}: source {identity_field} is missing")
        matrix = source.get("matrix")
        if not isinstance(matrix, dict) or matrix.get("cases") != list(CASE_IDS) or matrix.get("row_count") != 7:
            _fail(f"{target}: summary matrix is not the frozen seven-row set")
        raw_rows = source.get("rows")
        if not isinstance(raw_rows, list) or len(raw_rows) != 7:
            _fail(f"{target}: expected exactly seven rows")
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for index, expected_case in enumerate(CASE_SPECS):
            row = raw_rows[index]
            if not isinstance(row, Mapping):
                _fail(f"{target}: row {index} is not an object")
            normal = _normalise_row(target, row, expected_case, index)
            if normal["case_id"] in seen:
                _fail(f"{target}: duplicate case row {normal['case_id']}")
            seen.add(normal["case_id"])
            rows.append(normal)
        constraints = []
        local_constraint_ids: set[str] = set()
        for constraint in _custom_constraints(target, source) + _source_failure_records(target, source, raw_rows):
            if constraint["id"] in local_constraint_ids:
                continue
            local_constraint_ids.add(constraint["id"])
            constraints.append(constraint)
        all_constraints.extend(constraints)
        applicable_extra_constraints = [
            constraint for constraint in normalised_extra_constraints if constraint["target"] in {target, "all"}
        ]
        correctness, resources = _target_status(target, source, raw_rows, constraints + applicable_extra_constraints)
        correctness_by_target.append({"target": target, **correctness})
        resources_by_target.append({"target": target, **resources})
        identity = {key: source[key] for key in ("target", "gpu_uuid", "gpu_bdf") if key in source}
        for optional in ("actual_arch", "wavefront_size", "rocm_root", "rocm_source_root", "rocm_version"):
            if optional in source:
                identity[optional] = source[optional]
        identity_scope, row_sources = _normalise_row_sources(target, source, str(sources[target]))
        source_kind = "per-row-inputs" if identity_scope == "per-row-mixed" else "aggregate-summary"
        all_targets.append({
            "target": target,
            "family": expected["family"],
            "phase": expected["phase"],
            "source": {"path": str(sources[target]), "sha256": _sha(source), "schema_version": source["schema_version"], "kind": source_kind},
            "state": source.get("state") if source.get("state") in {"PASS", "FAIL"} else "FAIL",
            "identity_scope": identity_scope,
            "identity": identity,
            "matrix": {"cases": list(CASE_IDS), "row_count": 7},
            "row_sources": row_sources,
            "rows": rows,
        })
    # Keep IDs deterministic while retaining the first record when a producer
    # and command-line constraint use the same identifier.
    deduped_constraints: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for constraint in all_constraints:
        if constraint["id"] in seen_ids:
            continue
        seen_ids.add(constraint["id"])
        deduped_constraints.append(constraint)
    family_breakdown = [
        {
            "family": family,
            "targets": [target for target in TARGET_ORDER if TARGETS[target]["family"] == family],
            "row_count": 7,
            "source_states": [item["state"] for item in all_targets if item["family"] == family],
            "correctness": [item["status"] for item in correctness_by_target if item["target"] in [target for target in TARGET_ORDER if TARGETS[target]["family"] == family]][0],
            "resources": [item["status"] for item in resources_by_target if item["target"] in [target for target in TARGET_ORDER if TARGETS[target]["family"] == family]][0],
        }
        for family in ("RDNA2", "RDNA4", "CDNA3")
    ]
    return {
        "schema_version": "three-target-gpu-summary-v1",
        "state": "PASS",
        "matrix": {"cases": list(CASE_IDS), "row_count": 7, "target_count": 3},
        "targets": all_targets,
        "gpu_family_breakdown": family_breakdown,
        "target_selector": {
            "kind": "exact-gfx-target",
            "fallback_allowed": False,
            "default_target": None,
            "targets": [{"target": target, "family": TARGETS[target]["family"], "phase": TARGETS[target]["phase"]} for target in TARGET_ORDER],
        },
        "correctness": {"by_target": correctness_by_target, "all_pass": all(item["status"] == "PASS" for item in correctness_by_target)},
        "resources": {"by_target": resources_by_target, "all_pass": all(item["status"] == "PASS" for item in resources_by_target)},
        "known_constraints": deduped_constraints,
    }


def _parse_constraint(value: str, index: int) -> dict[str, str]:
    parts = value.split(":", 2)
    if len(parts) == 3 and parts[0] in (set(TARGETS) | {"all"}):
        target, scope, reason = parts
    else:
        target, scope, reason = "all", "summary", value
    return {"id": f"cli-constraint-{index}", "target": target, "scope": scope, "reason": reason}


def _write_output(path: Path, document: Mapping[str, Any]) -> None:
    payload = canonical_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != payload:
            _fail(f"refusing to overwrite existing aggregate: {path}")
        return
    path.write_bytes(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    gfx1030_group = parser.add_mutually_exclusive_group()
    gfx1030_group.add_argument("--gfx1030", dest="gfx1030", type=Path, help="one already-aggregated Phase 49 seven-row summary")
    gfx1030_group.add_argument("--gfx1030-normal5", dest="gfx1030_normal5", type=Path, help="Phase 49 final-normal5 five-row raw evidence")
    parser.add_argument("--gfx1030-long100000", type=Path, help="Phase 49 long-prefill-v2 long-100000 raw evidence (required with --gfx1030-normal5)")
    parser.add_argument("--gfx1030-decode20000", type=Path, help="Phase 49 direct decode-20000 raw evidence (required with --gfx1030-normal5)")
    parser.add_argument("--gfx1030-llama", type=Path, help="Phase 49 seven-row llama.cpp control evidence (required with --gfx1030-normal5)")
    parser.add_argument("--gfx1201", type=Path, required=True)
    parser.add_argument("--gfx942", type=Path, required=True)
    parser.add_argument("--known-constraint", action="append", default=[], help="target:scope:reason (repeatable); omit prefix for an all-target constraint")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        split_paths = (args.gfx1030_long100000, args.gfx1030_decode20000, args.gfx1030_llama)
        if args.gfx1030 is None and args.gfx1030_normal5 is None:
            parser.error("one of --gfx1030 or --gfx1030-normal5 is required")
        if args.gfx1030_normal5 is not None and any(path is None for path in split_paths):
            parser.error("--gfx1030-normal5 requires --gfx1030-long100000, --gfx1030-decode20000, and --gfx1030-llama")
        if args.gfx1030 is not None and any(path is not None for path in split_paths):
            parser.error("split Phase 49 inputs cannot be combined with --gfx1030")
        if args.gfx1030 is not None:
            gfx1030_summary = load_json(args.gfx1030)
            gfx1030_source = str(args.gfx1030.resolve())
        else:
            assert args.gfx1030_normal5 is not None
            assert args.gfx1030_long100000 is not None
            assert args.gfx1030_decode20000 is not None
            assert args.gfx1030_llama is not None
            gfx1030_summary = build_phase49_split_summary(args.gfx1030_normal5, args.gfx1030_long100000, args.gfx1030_decode20000, args.gfx1030_llama)
            gfx1030_source = "phase49-split-inputs"
        summaries = {"gfx1030": gfx1030_summary, "gfx1201": load_json(args.gfx1201), "gfx942": load_json(args.gfx942)}
        sources = {"gfx1030": gfx1030_source, "gfx1201": str(args.gfx1201.resolve()), "gfx942": str(args.gfx942.resolve())}
        document = aggregate_target_summaries(summaries, sources=sources, known_constraints=[_parse_constraint(item, index) for index, item in enumerate(args.known_constraint, 1)])
        if args.output is not None:
            _write_output(args.output, document)
        print(json.dumps(document, ensure_ascii=False, sort_keys=True))
        return 0
    except ThreeTargetError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
