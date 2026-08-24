#!/usr/bin/env python3
"""Aggregate the two bounded Phase 49 V620 producer summaries.

The producer scripts retain the raw request evidence.  This tool is the
bounded publication boundary: it re-opens only the producer summaries,
checks their PASS identity and the small set of fields needed for a
cross-engine comparison, and publishes medians/MADs plus token/stop digests.
It deliberately never copies generated token arrays, event timelines, or
other unbounded producer evidence into the tracked aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from statistics import median
from typing import Any, Mapping, NoReturn, Sequence


TARGET = "gfx1030"
GPU_UUID = "GPU-76a08c022586fed6"
SLLM_SCHEMA = "phase49-v620-sllm-v1"
LLAMA_SCHEMA = "phase49-v620-llama-v1"
SLLM_ROW_SCHEMA = "phase49-v620-sllm-row-v1"
LLAMA_ROW_SCHEMA = "phase49-v620-llama-row-v1"
SLLM_DIRECT_SCHEMA = "engine-performance-direct-v2"
LLAMA_WRAPPER_SCHEMA = "llama-phase49-v620-v1"
LLAMA_COMMIT = "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70"
LLAMA_TAG = "b10453"
SLLM_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
STOP_IDS = (248046, 248044)
SHORT_ODD = [1, 3, 17, 37, 73, 255, 256, 257, 2, 5, 11, 19, 23, 29, 31, 41, 43]
CASE_SPECS: tuple[tuple[str, int, int], ...] = (
    ("short-odd", 17, 17),
    ("32-32", 32, 32),
    ("prefill-long", 1024, 128),
    ("decode-long", 32, 256),
    ("long-10001", 10_001, 2),
    ("long-100000", 100_000, 2),
    ("decode-20000", 32, 20_000),
)
CASE_IDS = tuple(item[0] for item in CASE_SPECS)
EXTENDED_CASES = {"long-100000", "decode-20000"}
METRICS = ("e2e_ns", "ttft_ns", "tpot_ns")
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_TOKEN_ID = 248319
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

CONTROL_TOKEN_FIELDS = (
    "input_token_ids",
    "generated_token_ids",
    "visible_token_ids",
    "decode_input_token_ids",
)
CONTROL_STOP_FIELDS = ("version", "reason_version", "kind", "token_id")
CONTROL_AUDIT_FIELDS = (
    "selected_backend",
    "target",
    "device_index",
    "model_fingerprint",
    "plan_digest",
    "fallback_used",
    "all_dispatches_hip",
    "submission_count",
    "kernel_dispatch_count",
    "segment_count",
    "boundary_count",
)
CONTROL_COMPARISON = {
    "mode": "exact",
    "scope": "first_warmup_reference_against_every_remaining_warmup_and_measured_sample",
    "reference_source": "warmups.samples[0]",
    "token_fields": list(CONTROL_TOKEN_FIELDS),
    "stop_fields": list(CONTROL_STOP_FIELDS),
    "dispatch_fields": list(CONTROL_AUDIT_FIELDS),
    "dispatch_count_rule": "exact_when_token_and_stop_fields_match",
}


def input_ids_for(case_id: str) -> list[int]:
    """Reconstruct the frozen matrix input instead of trusting both producers."""
    if case_id == "short-odd":
        return list(SHORT_ODD)
    if case_id in {"32-32", "prefill-long"}:
        count = 32 if case_id == "32-32" else 1024
        result = list(SHORT_ODD)
        result.extend(((index * 7919 + 41) % 248000) for index in range(len(result), count))
        return result
    if case_id in {"decode-long", "decode-20000"}:
        result = list(SHORT_ODD)
        result.extend(((index * 7919 + 41) % 248000) for index in range(len(result), 32))
        return result
    if case_id == "long-10001":
        return [23066] * 10_001
    if case_id == "long-100000":
        return [23066] * 100_000
    _fail(f"unknown fixed Phase 49 case: {case_id}")


class Phase49Error(RuntimeError):
    """Malformed, stale, or unsafe Phase 49 evidence."""


def _fail(message: str) -> NoReturn:
    raise Phase49Error(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _reject_constant(token: str) -> NoReturn:
    _fail(f"non-finite JSON constant {token}")


def load_json(path: Path) -> dict[str, Any]:
    """Load a producer summary with duplicate/non-finite rejection."""
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
    except Phase49Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"{path}: malformed JSON: {exc}")
    if not isinstance(value, dict):
        _fail(f"{path}: summary is not an object")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label}: value is not numeric")
    converted = float(value)
    if not math.isfinite(converted) or (positive and converted <= 0):
        _fail(f"{label}: value is non-finite or non-positive")
    return converted


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def summary_stats(values: Sequence[float], label: str) -> dict[str, float | int]:
    if not values:
        _fail(f"{label}: empty measured distribution")
    finite = [_finite(item, f"{label}[{index}]", positive=True) for index, item in enumerate(values)]
    middle = float(median(finite))
    deviations = [abs(item - middle) for item in finite]
    return {
        "median": middle,
        "mad": float(median(deviations)),
        "count": len(finite),
        "min": min(finite),
        "max": max(finite),
    }


def _semantic_stop(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label}: stop is not an object")
    kind = value.get("kind")
    token_id = value.get("token_id")
    if not isinstance(kind, str) or not kind or token_id is not None and (isinstance(token_id, bool) or not isinstance(token_id, int)):
        _fail(f"{label}: stop identity is malformed")
    return {"kind": kind, "token_id": token_id}


def _tokens(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 or item > MAX_TOKEN_ID for item in value):
        _fail(f"{label}: token sequence is malformed")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
        _fail(f"{label}: digest is malformed")
    return value


def _validate_stop(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label}: stop is not an object")
    if value.get("version") != 1 or value.get("reason_version") != 1:
        _fail(f"{label}: stop protocol version is stale")
    kind = value.get("kind")
    token_id = value.get("token_id")
    if kind not in {"max_new_tokens", "stop_token"}:
        _fail(f"{label}: stop reason is invalid")
    if token_id is not None and (isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0):
        _fail(f"{label}: stop token ID is malformed")
    if kind == "max_new_tokens" and token_id is not None:
        _fail(f"{label}: max-new-token stop must not carry a token ID")
    if kind == "stop_token" and token_id is None:
        _fail(f"{label}: stop-token stop must carry a token ID")
    return {key: value.get(key) for key in CONTROL_STOP_FIELDS}


def _validate_dispatch_audit(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label}: dispatch audit is absent")
    if value.get("selected_backend") != "hip" or value.get("target") != TARGET or value.get("device_index") != 0:
        _fail(f"{label}: exact HIP target identity is invalid")
    if not isinstance(value.get("model_fingerprint"), str) or not value["model_fingerprint"]:
        _fail(f"{label}: model fingerprint is absent")
    if not isinstance(value.get("plan_digest"), str) or not value["plan_digest"]:
        _fail(f"{label}: plan digest is absent")
    if value.get("fallback_used") is not False or value.get("all_dispatches_hip") is not True:
        _fail(f"{label}: HIP/no-fallback evidence is invalid")
    for key in ("submission_count", "kernel_dispatch_count", "segment_count", "boundary_count"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            _fail(f"{label}: {key} is not a positive integer")
    return {key: value.get(key) for key in CONTROL_AUDIT_FIELDS}


def _validate_sample_cleanup(value: Any, label: str, expected_index: int | None = None) -> None:
    if not isinstance(value, dict):
        _fail(f"{label}: cleanup is absent")
    if value.get("request_dropped") is not True or value.get("allocator_cleanup_validated") is not True:
        _fail(f"{label}: request cleanup was not validated")
    if value.get("retryable_cleanup") != 0 or value.get("durable_quarantine") != 0:
        _fail(f"{label}: request cleanup is non-empty")
    if expected_index is not None and value.get("sample_index") != expected_index:
        _fail(f"{label}: sample index is not contiguous")


def _validate_memory_cleanup(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        _fail(f"{label}: memory accounting is absent")
    start = value.get("request_start")
    after = value.get("after_cleanup")
    if not isinstance(start, dict) or not isinstance(after, dict):
        _fail(f"{label}: request memory snapshots are absent")
    for snapshot_name, snapshot in (("request_start", start), ("after_cleanup", after)):
        if snapshot.get("poisoned") is not False:
            _fail(f"{label}.{snapshot_name}: allocation accounting is poisoned")
        for category in ("model_resident", "request_state", "workspace"):
            section = snapshot.get(category)
            if not isinstance(section, dict) or isinstance(section.get("current_bytes"), bool) or not isinstance(section.get("current_bytes"), int) or section.get("current_bytes") < 0:
                _fail(f"{label}.{snapshot_name}.{category}: current allocation accounting is malformed")
    if after["request_state"]["current_bytes"] != 0 or after["workspace"]["current_bytes"] != 0:
        _fail(f"{label}: request cleanup left non-zero allocation bytes")
    if after["model_resident"]["current_bytes"] != start["model_resident"]["current_bytes"]:
        _fail(f"{label}: request cleanup changed resident model allocation")


def _validate_sllm_control(control: Any, label: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(control, dict):
        _fail(f"{label}: correctness reference is absent")
    expected_metadata = {
        "label": "correctness-reference",
        "execution_path": "first-warmup-sample",
        "timing_instrumentation": "on",
        "included_in_performance_statistics": False,
        "source": {"kind": "warmup-sample", "sample_index": 0, "request_count": 0},
        "comparison": CONTROL_COMPARISON,
    }
    for key, expected in expected_metadata.items():
        if control.get(key) != expected:
            _fail(f"{label}: correctness-reference metadata is stale in {key}")
    tokens = control.get("tokens")
    if not isinstance(tokens, dict):
        _fail(f"{label}: correctness-reference tokens are absent")
    token_values = {key: _tokens(tokens.get(key), f"{label} tokens.{key}") for key in CONTROL_TOKEN_FIELDS}
    if not token_values["generated_token_ids"] or token_values["visible_token_ids"] != token_values["generated_token_ids"] or token_values["decode_input_token_ids"] != token_values["generated_token_ids"][:-1]:
        _fail(f"{label}: correctness-reference token shape is invalid")
    stop = _validate_stop(control.get("stop"), f"{label} stop")
    audit = _validate_dispatch_audit(control.get("audit"), f"{label} audit")
    memory = control.get("memory")
    _validate_memory_cleanup(memory, f"{label} memory")
    cleanup = control.get("cleanup")
    if not isinstance(cleanup, dict) or cleanup.get("reference_sample") is not True:
        _fail(f"{label}: correctness-reference cleanup marker is absent")
    _validate_sample_cleanup(cleanup, f"{label} cleanup")
    return token_values, stop, audit


def _validate_timing_sample(sample: Mapping[str, Any], label: str, output_count: int, engine: str) -> None:
    if engine == "sllm":
        if sample.get("execution_path") != "timed-production" or sample.get("timing_instrumentation") != "on":
            _fail(f"{label}: timed-production instrumentation identity is invalid")
        events = sample.get("events")
        later_key, cleanup_key = "later_token_publications_ns", "cleanup_ns"
    else:
        events = sample.get("events")
        later_key, cleanup_key = "token_publications_ns", "cleanup_complete_ns"
    derived = sample.get("derived")
    if not isinstance(events, dict) or not isinstance(derived, dict):
        _fail(f"{label}: events/derived are absent")
    ordered = [events.get("request_start_ns"), events.get("prefill_submit_ns"), events.get("prefill_complete_ns"), events.get("first_token_ns")]
    later = events.get(later_key)
    if not isinstance(later, list) or len(later) != output_count - 1 or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in later):
        _fail(f"{label}: token publication count is invalid")
    ordered.extend(later)
    ordered.extend([events.get("stop_ns"), events.get(cleanup_key)])
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in ordered) or any(right <= left for left, right in zip(ordered, ordered[1:])):
        _fail(f"{label}: event timestamps are not strictly ordered")
    for key in ("ttft_ns", "prefill_ns", "e2e_ns"):
        if isinstance(derived.get(key), bool) or not isinstance(derived.get(key), int) or derived[key] <= 0:
            _fail(f"{label}: derived {key} is invalid")
    if derived["ttft_ns"] != events["first_token_ns"] - events["request_start_ns"] or derived["prefill_ns"] != events["prefill_complete_ns"] - events["prefill_submit_ns"] or derived["e2e_ns"] != events[cleanup_key] - events["request_start_ns"]:
        _fail(f"{label}: event/derived timing mismatch")
    tpot = derived.get("tpot_ns")
    if not isinstance(tpot, list) or len(tpot) != output_count - 1 or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in tpot):
        _fail(f"{label}: TPOT count/value is invalid")
    previous = events["first_token_ns"]
    for item, publication in zip(tpot, later):
        if item != publication - previous:
            _fail(f"{label}: TPOT/event timing mismatch")
        previous = publication
    if derived.get("decode_tokens") != output_count - 1:
        _fail(f"{label}: decode token count is invalid")


def _sample_metrics(sample: Mapping[str, Any], label: str, input_ids: list[int], output_count: int, engine: str) -> tuple[dict[str, Any], dict[str, Any]]:
    tokens = sample.get("tokens")
    if not isinstance(tokens, dict):
        _fail(f"{label}: tokens are absent")
    sample_input = _tokens(tokens.get("input_token_ids"), f"{label} input")
    if sample_input != input_ids:
        _fail(f"{label}: input token sequence differs from row matrix")
    generated = _tokens(tokens.get("generated_token_ids"), f"{label} generated")
    visible = _tokens(tokens.get("visible_token_ids"), f"{label} visible")
    if len(generated) != output_count or visible != generated:
        _fail(f"{label}: generated/visible token contract is invalid")
    stop = _validate_stop(sample.get("stop"), f"{label} stop")
    derived = sample.get("derived")
    if not isinstance(derived, dict):
        _fail(f"{label}: derived metrics are absent")
    e2e = _finite(derived.get("e2e_ns"), f"{label} e2e_ns", positive=True)
    ttft = _finite(derived.get("ttft_ns"), f"{label} ttft_ns", positive=True)
    raw_tpot = derived.get("tpot_ns")
    if not isinstance(raw_tpot, list) or len(raw_tpot) != output_count - 1:
        _fail(f"{label}: tpot_ns is malformed")
    if not raw_tpot:
        _fail(f"{label}: tpot_ns is empty")
    tpot = [
        _finite(item, f"{label} tpot_ns[{index}]", positive=True)
        for index, item in enumerate(raw_tpot)
    ]
    return (
        {"e2e_ns": e2e, "ttft_ns": ttft, "tpot_ns": float(median(tpot))},
        {"generated": generated, "visible": visible, "stop": stop},
    )


def _validate_summary_header(summary: Mapping[str, Any], engine: str) -> None:
    expected_schema = SLLM_SCHEMA if engine == "sllm" else LLAMA_SCHEMA
    if summary.get("schema_version") != expected_schema or summary.get("state") != "PASS":
        _fail(f"{engine}: producer summary is not PASS/current schema")
    if summary.get("target") != TARGET or summary.get("gpu_uuid") != GPU_UUID:
        _fail(f"{engine}: exact V620 target identity is invalid")
    if engine == "llama":
        llama = summary.get("llama")
        if not isinstance(llama, dict) or llama.get("commit") != LLAMA_COMMIT or llama.get("tag") != LLAMA_TAG:
            _fail("llama: top-level source identity is invalid")
    matrix = summary.get("matrix")
    if not isinstance(matrix, dict) or matrix.get("row_count") != len(CASE_SPECS) or matrix.get("cases") != list(CASE_IDS):
        _fail(f"{engine}: fixed seven-row matrix identity is invalid")
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != len(CASE_SPECS):
        _fail(f"{engine}: summary row count is not seven")


def _optional_sha(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        _fail(f"{label}: identity is not an object")
    digest = value.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        _fail(f"{label}: SHA-256 identity is malformed")
    return digest


def _validate_report_identity(report: Mapping[str, Any], label: str, expected_binary_sha: str | None, expected_model_sha: str | None, expected_lock_sha: str | None) -> None:
    if expected_binary_sha is not None:
        actual = _optional_sha(report.get("binary"), f"{label} binary")
        if actual != expected_binary_sha:
            _fail(f"{label}: binary identity differs from producer summary")
    elif "binary" in report:
        _optional_sha(report.get("binary"), f"{label} binary")
    if expected_model_sha is not None:
        actual = _optional_sha(report.get("model"), f"{label} model")
        if actual != expected_model_sha:
            _fail(f"{label}: model identity differs from producer summary")
    elif "model" in report:
        _optional_sha(report.get("model"), f"{label} model")
    if expected_lock_sha is not None:
        actual = _optional_sha(report.get("lock"), f"{label} lock")
        if actual != expected_lock_sha:
            _fail(f"{label}: lock identity differs from producer summary")
    elif "lock" in report:
        _optional_sha(report.get("lock"), f"{label} lock")


def _validate_direct_v2(result: Mapping[str, Any], row: Mapping[str, Any], label: str) -> None:
    if result.get("lane_definition") != "pretokenized direct engine: request start excludes render/tokenize":
        _fail(f"{label}: direct lane definition is invalid")
    model_load = result.get("model_load")
    if not isinstance(model_load, dict) or model_load.get("event") != "model_load" or model_load.get("load_count") != 1:
        _fail(f"{label}: model-load evidence is absent")
    for key in ("start_ns", "model_ready_ns", "duration_ns"):
        if isinstance(model_load.get(key), bool) or not isinstance(model_load.get(key), int) or model_load[key] < 0:
            _fail(f"{label}: model-load timing is invalid")
    if model_load["duration_ns"] <= 0 or model_load["model_ready_ns"] - model_load["start_ns"] != model_load["duration_ns"]:
        _fail(f"{label}: model-load timing is inconsistent")
    session_cleanup = result.get("session_cleanup")
    if not isinstance(session_cleanup, dict) or session_cleanup.get("retryable_cleanup") != 0 or session_cleanup.get("durable_quarantine") != 0:
        _fail(f"{label}: session cleanup is not empty")
    memory = result.get("memory")
    if not isinstance(memory, dict):
        _fail(f"{label}: top-level memory evidence is absent")
    for key in ("placement_total_memory_bytes", "placement_available_memory_bytes", "placement_required_bytes", "placement_model_resident_bytes", "placement_request_state_bytes", "placement_safety_reserve_bytes", "workspace_separate_allocation_bytes", "workspace_arena_bytes", "model_resident_high_water_bytes", "resident_vram_bytes", "peak_vram_bytes"):
        if isinstance(memory.get(key), bool) or not isinstance(memory.get(key), int) or memory[key] < 0:
            _fail(f"{label}: top-level memory field {key} is invalid")
    for snapshot_name in ("model_ready", "after_model_drop"):
        snapshot = memory.get(snapshot_name)
        if not isinstance(snapshot, dict) or snapshot.get("poisoned") is not False:
            _fail(f"{label}: {snapshot_name} allocation snapshot is invalid")
        for key in ("current_bytes", "high_water_bytes"):
            if isinstance(snapshot.get(key), bool) or not isinstance(snapshot.get(key), int) or snapshot[key] < 0:
                _fail(f"{label}: {snapshot_name}.{key} allocation is invalid")
        for category in ("model_resident", "request_state", "workspace"):
            section = snapshot.get(category)
            if not isinstance(section, dict) or any(isinstance(section.get(key), bool) or not isinstance(section.get(key), int) or section[key] < 0 for key in ("current_bytes", "high_water_bytes")):
                _fail(f"{label}: {snapshot_name}.{category} allocation is invalid")
    if memory.get("resident_vram_source") != "model_resident_allocator_high_water" or memory.get("peak_source") != "runtime_allocator":
        _fail(f"{label}: top-level memory source identity is invalid")
    identities = result.get("identities")
    if not isinstance(identities, dict) or identities.get("engine") != "sllm" or identities.get("backend") != "hip" or isinstance(identities.get("session_id"), bool) or not isinstance(identities.get("session_id"), int) or identities.get("session_id", 0) <= 0 or identities.get("device_index") != 0 or identities.get("target") != TARGET:
        _fail(f"{label}: direct identity binding is invalid")
    model_identity = identities.get("model")
    binding = identities.get("binding")
    if not isinstance(model_identity, dict) or model_identity.get("model_size") != "4B" or model_identity.get("repo_id") != "Qwen/Qwen3.5-4B" or model_identity.get("resolved_revision") != SLLM_MODEL_REVISION:
        _fail(f"{label}: direct model identity is invalid")
    lock_fingerprint = _digest(model_identity.get("lock_fingerprint"), f"{label}.model.lock_fingerprint")
    if not isinstance(binding, dict) or _digest(binding.get("model_fingerprint"), f"{label}.binding.model_fingerprint") != lock_fingerprint:
        _fail(f"{label}: model fingerprint binding is invalid")
    plan_digest = _digest(binding.get("plan_digest"), f"{label}.binding.plan_digest")
    audit = result.get("audit")
    expected_requests = row["warmups"] + row["measured"]
    if not isinstance(audit, dict) or audit.get("device_index") != 0 or audit.get("model_load_count") != 1 or audit.get("request_model_load_count") != 0 or audit.get("model_reused") is not True or audit.get("sample_count") != expected_requests or audit.get("total_request_count") != expected_requests or audit.get("correctness_control_request_count") != 0 or audit.get("correctness_control_source") != "first-warmup-sample" or audit.get("correctness_control_reference_sample_index") != 0 or audit.get("model_fingerprint") != lock_fingerprint or _digest(audit.get("plan_digest"), f"{label}.audit.plan_digest") != plan_digest:
        _fail(f"{label}: aggregate audit identity is invalid")
    for key in ("submission_count", "kernel_dispatch_count", "segment_count", "boundary_count"):
        if isinstance(audit.get(key), bool) or not isinstance(audit.get(key), int) or audit[key] <= 0:
            _fail(f"{label}: aggregate audit count {key} is invalid")
    config = result.get("config")
    if not isinstance(config, dict) or config.get("tokenizer") is not False or config.get("render") is not False or config.get("lane") != "direct" or config.get("kv_cache_encoding") != "fp16" or isinstance(config.get("effective_context_length"), bool) or not isinstance(config.get("effective_context_length"), int) or config.get("effective_context_length") <= 0 or isinstance(config.get("completion_timeout_seconds"), bool) or not isinstance(config.get("completion_timeout_seconds"), int) or config.get("completion_timeout_seconds") <= 0:
        _fail(f"{label}: direct lane/config identity is invalid")
    stop_policy = config.get("stop_policy")
    if not isinstance(stop_policy, dict) or stop_policy.get("visible_stop_tokens") is not False or stop_policy.get("ignore_eos") is not row["ignore_eos"] or stop_policy.get("stop_token_ids") != ([] if row["ignore_eos"] else list(STOP_IDS)):
        _fail(f"{label}: stop policy is invalid")


def _validate_report_resources(report: Mapping[str, Any], label: str) -> None:
    process = report.get("process")
    if not isinstance(process, dict) or not isinstance(process.get("capture"), dict) or process["capture"].get("process_group_gone") is not True:
        _fail(f"{label}: producer process group cleanup is absent")
    memory = report.get("memory")
    if not isinstance(memory, dict) or not isinstance(memory.get("baseline"), dict) or not isinstance(memory.get("settled"), dict):
        _fail(f"{label}: producer HBM/GTT resource evidence is absent")
    baseline = memory["baseline"]
    settled = memory["settled"]
    if settled.get("settled") is not True or settled.get("hbm_bytes") != baseline.get("hbm_bytes") or settled.get("gtt_bytes") != baseline.get("gtt_bytes"):
        _fail(f"{label}: producer HBM/GTT resources did not return to baseline")
    monitor = report.get("monitor")
    if not isinstance(monitor, dict) or monitor.get("errors") not in ([], None) or not isinstance(monitor.get("samples"), int) or monitor.get("samples") <= 0:
        _fail(f"{label}: producer resource monitor evidence is invalid")


def _validate_engine_specific(result: Mapping[str, Any], report: Mapping[str, Any], engine: str, label: str, row: Mapping[str, Any]) -> None:
    if engine == "sllm":
        if result.get("benchmark_schema_version") != SLLM_DIRECT_SCHEMA or result.get("state") != "PASS" or result.get("lane") != "direct":
            _fail(f"{label}: sLLM direct result is not PASS/current schema")
        identities = result.get("identities")
        audit = result.get("audit")
        if not isinstance(identities, dict) or identities.get("engine") != "sllm" or identities.get("backend") != "hip" or identities.get("target") != TARGET:
            _fail(f"{label}: sLLM HIP identity is invalid")
        model_identity = identities.get("model")
        if isinstance(model_identity, dict) and "resolved_revision" in model_identity and model_identity.get("resolved_revision") != SLLM_MODEL_REVISION:
            _fail(f"{label}: sLLM model revision is not the fixed Phase 49 revision")
        if not isinstance(audit, dict) or audit.get("selected_backend") != "hip" or audit.get("target") != TARGET or audit.get("all_dispatches_hip") is not True or audit.get("fallback_used") is not False or audit.get("weight_encoding") != "bf16":
            _fail(f"{label}: sLLM HIP/no-fallback evidence is invalid")
        cleanup = result.get("cleanup")
        if not isinstance(cleanup, dict) or cleanup.get("all_requests_dropped") is not True or cleanup.get("correctness_control_request_count") != 0 or cleanup.get("correctness_control_source") != "first-warmup-sample" or cleanup.get("correctness_control_reference_sample_index") != 0 or cleanup.get("retryable_cleanup") != 0 or cleanup.get("durable_quarantine") != 0:
            _fail(f"{label}: sLLM cleanup evidence is invalid")
        _validate_direct_v2(result, row, label)
        _validate_sllm_control(result.get("correctness_control"), f"{label} correctness_control")
    else:
        if result.get("schema_version") != LLAMA_WRAPPER_SCHEMA or result.get("state") != "PASS":
            _fail(f"{label}: llama wrapper result is not PASS/current schema")
        llama = result.get("llama")
        target = result.get("target")
        if not isinstance(llama, dict) or llama.get("commit") != LLAMA_COMMIT or llama.get("tag") != LLAMA_TAG:
            _fail(f"{label}: llama source identity is invalid")
        if not isinstance(target, dict) or target.get("exact") != TARGET or target.get("gpu_uuid") != GPU_UUID or target.get("logical_device_index") != 0:
            _fail(f"{label}: llama target identity is invalid")
        model = result.get("model")
        if not isinstance(model, dict) or model.get("format") != "GGUF" or model.get("weights") != "BF16" or model.get("kv") != "F16":
            _fail(f"{label}: llama model identity is invalid")
        offload = result.get("offload_evidence")
        if not isinstance(offload, dict) or offload.get("gpu_offload_supported") is not True or offload.get("visible_gpu_device_count") != 1 or offload.get("selected_device", {}).get("type") != "GPU" or offload.get("requested", {}).get("n_gpu_layers") != -1 or offload.get("requested", {}).get("split_mode") != "none" or offload.get("requested", {}).get("main_gpu") != 0 or offload.get("requested", {}).get("offload_kqv") is not True or offload.get("requested", {}).get("op_offload") is not True or offload.get("observed", {}).get("offloaded_layers") != offload.get("observed", {}).get("offloadable_layers"):
            _fail(f"{label}: llama full-offload evidence is invalid")
        cleanup = result.get("cleanup")
        if not isinstance(cleanup, dict) or cleanup.get("backend_release_completed") is not True or cleanup.get("cleanup_failures") != 0:
            _fail(f"{label}: llama cleanup evidence is invalid")
    _validate_report_resources(report, label)


def _row_data(
    report: Any,
    engine: str,
    case_id: str,
    input_count: int,
    output_count: int,
    *,
    expected_binary_sha: str | None = None,
    expected_model_sha: str | None = None,
    expected_lock_sha: str | None = None,
) -> dict[str, Any]:
    label = f"{engine}/{case_id}"
    if not isinstance(report, dict) or report.get("state") != "PASS":
        _fail(f"{label}: row report is not PASS")
    _validate_report_identity(report, label, expected_binary_sha, expected_model_sha, expected_lock_sha)
    expected_row_schema = SLLM_ROW_SCHEMA if engine == "sllm" else LLAMA_ROW_SCHEMA
    if report.get("schema_version") != expected_row_schema:
        _fail(f"{label}: row schema is stale")
    if report.get("target") != TARGET or report.get("gpu_uuid") != GPU_UUID:
        _fail(f"{label}: row GPU identity is invalid")
    if engine == "sllm" and report.get("weight") != "bf16":
        _fail(f"{label}: sLLM weight identity is invalid")
    row = report.get("row")
    expected_row_id = f"phase49-v620-{engine}-{case_id}"
    if not isinstance(row, dict) or row.get("row_id") != expected_row_id or row.get("case_id") != case_id or row.get("input_token_count") != input_count or row.get("requested_output_tokens") != output_count or (engine == "sllm" and row.get("model_size") != "4B"):
        _fail(f"{label}: row matrix identity is invalid")
    input_ids = _tokens(row.get("input_token_ids"), f"{label} row input")
    expected_input = input_ids_for(case_id)
    if input_ids != expected_input or len(input_ids) != input_count:
        _fail(f"{label}: row input token count is invalid")
    if row.get("target") != TARGET:
        _fail(f"{label}: row target is invalid")
    result = report.get("result")
    if not isinstance(result, dict):
        _fail(f"{label}: producer result is absent")
    _validate_engine_specific(result, report, engine, label, row)
    expected_warmups = 1 if case_id in EXTENDED_CASES else 3
    expected_measured = 3 if case_id in EXTENDED_CASES else 10
    expected_context = 131_072 if case_id in EXTENDED_CASES else input_count + output_count
    expected_ignore_eos = case_id == "decode-20000"
    if row.get("warmups") != expected_warmups or row.get("measured") != expected_measured or row.get("context_length") != expected_context or row.get("ignore_eos") is not expected_ignore_eos:
        _fail(f"{label}: row protocol identity is invalid")
    if engine == "sllm":
        config = result.get("config")
        if not isinstance(config, dict) or config.get("input_token_ids") != input_ids or config.get("input_token_count") != input_count or config.get("max_new_tokens") != output_count or config.get("greedy") is not True or config.get("warmups") != expected_warmups or config.get("measured") != expected_measured or config.get("context_length") != expected_context or config.get("ignore_eos") is not expected_ignore_eos or config.get("prefill_chunk_tokens") is not None:
            _fail(f"{label}: sLLM direct protocol is stale")
    else:
        protocol = result.get("protocol")
        expected_protocol = {
            "batch_size": 1,
            "sequences": 1,
            "warmup_requests": expected_warmups,
            "measured_requests": expected_measured,
            "max_new_tokens": output_count,
            "n_ctx": expected_context,
            "n_batch": 2048,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "split_mode": "none",
            "main_gpu": 0,
            "offload_kqv": True,
            "op_offload": True,
            "greedy": True,
            "ignore_eos": expected_ignore_eos,
            "stop_token_ids": [] if expected_ignore_eos else list(STOP_IDS),
            "bos_inserted": False,
        }
        if not isinstance(protocol, dict) or any(protocol.get(key) != value for key, value in expected_protocol.items()):
            _fail(f"{label}: llama protocol is stale")
    if engine == "llama" and not isinstance(result.get("row"), dict):
        # The llama wrapper publishes row_id/case_id/input_token_ids at its
        # result top level; the sLLM direct benchmark uses a nested row.
        result_row = result
    else:
        result_row = result.get("row")
    if not isinstance(result_row, dict) or result_row.get("case_id") != case_id or result_row.get("input_token_ids") != input_ids:
        _fail(f"{label}: result row identity differs")
    if engine == "sllm" and (result_row.get("input_token_count") != input_count or result_row.get("requested_output_tokens") != output_count):
        _fail(f"{label}: sLLM result row shape differs")
    if result_row.get("row_id") not in (None, row.get("row_id")):
        _fail(f"{label}: result row ID differs")
    warmups = result.get("warmups")
    if not isinstance(warmups, dict) or warmups.get("count") != expected_warmups or not isinstance(warmups.get("samples"), list) or len(warmups["samples"]) != expected_warmups:
        _fail(f"{label}: warmup sample count is invalid")
    measured = result.get("measured")
    if not isinstance(measured, dict) or measured.get("count") != expected_measured or not isinstance(measured.get("samples"), list) or len(measured["samples"]) != expected_measured:
        _fail(f"{label}: measured sample count is invalid")
    if engine == "sllm":
        control_tokens, control_stop, control_audit = _validate_sllm_control(result.get("correctness_control"), f"{label} correctness_control")
        if control_tokens["input_token_ids"] != input_ids or len(control_tokens["generated_token_ids"]) != output_count:
            _fail(f"{label}: correctness-reference token identity is invalid")
        model_identity = result.get("identities", {}).get("model") if isinstance(result.get("identities"), dict) else None
        if isinstance(model_identity, dict) and isinstance(model_identity.get("lock_fingerprint"), str) and control_audit["model_fingerprint"] != model_identity["lock_fingerprint"]:
            _fail(f"{label}: correctness-reference model fingerprint differs from lock identity")
        dispatch_totals = {key: 0 for key in ("submission_count", "kernel_dispatch_count", "segment_count", "boundary_count")}
        for group_name in ("warmups", "measured"):
            for index, sample in enumerate(result[group_name]["samples"]):
                sample_label = f"{label} {group_name} sample {index}"
                if not isinstance(sample, dict):
                    _fail(f"{sample_label}: sample is malformed")
                _validate_timing_sample(sample, sample_label, output_count, engine)
                tokens = sample.get("tokens")
                if not isinstance(tokens, dict):
                    _fail(f"{sample_label}: tokens are absent")
                sample_tokens = {key: _tokens(tokens.get(key), f"{sample_label} tokens.{key}") for key in CONTROL_TOKEN_FIELDS}
                if sample_tokens != control_tokens:
                    _fail(f"{sample_label}: token identity differs from correctness reference")
                sample_stop = _validate_stop(sample.get("stop"), f"{sample_label} stop")
                if sample_stop != control_stop:
                    _fail(f"{sample_label}: stop identity differs from correctness reference")
                sample_audit = _validate_dispatch_audit(sample.get("audit"), f"{sample_label} audit")
                if sample_audit != control_audit:
                    _fail(f"{sample_label}: dispatch identity/counts differ from correctness reference")
                for key in dispatch_totals:
                    dispatch_totals[key] += sample_audit[key]
                _validate_memory_cleanup(sample.get("memory"), f"{sample_label} memory")
                _validate_sample_cleanup(sample.get("cleanup"), f"{sample_label} cleanup", index)
        aggregate_audit = result.get("audit")
        if not isinstance(aggregate_audit, dict) or any(aggregate_audit.get(key) != value for key, value in dispatch_totals.items()):
            _fail(f"{label}: aggregate dispatch counts do not match per-sample evidence")
    else:
        for group_name in ("warmups", "measured"):
            for index, sample in enumerate(result[group_name]["samples"]):
                if not isinstance(sample, dict):
                    _fail(f"{label} {group_name} sample {index}: sample is malformed")
                _validate_timing_sample(sample, f"{label} {group_name} sample {index}", output_count, engine)
    metrics: dict[str, list[float]] = {name: [] for name in METRICS}
    token_records: list[dict[str, Any]] = []
    for index, sample in enumerate(measured["samples"]):
        sample_label = f"{label}/sample-{index}"
        _validate_timing_sample(sample, sample_label, output_count, engine)
        values, tokens = _sample_metrics(sample, sample_label, input_ids, output_count, engine)
        token_records.append(tokens)
        for metric in METRICS:
            metrics[metric].append(values[metric])
    first = token_records[0]
    if any(record != first for record in token_records[1:]):
        _fail(f"{label}: producer measured token/stop output is not deterministic")
    return {
        "row_id": expected_row_id,
        "input_ids": input_ids,
        "output_tokens": first["generated"],
        "visible_tokens": first["visible"],
        "stop": first["stop"],
        "stats": {metric: summary_stats(values, f"{label}/{metric}") for metric, values in metrics.items()},
        "sample_count": expected_measured,
    }


def _gate(sllm: Mapping[str, Any], llama: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in METRICS:
        left = sllm["stats"][metric]
        right = llama["stats"][metric]
        limit = float(right["median"]) + max(float(left["mad"]), float(right["mad"]))
        result[metric] = {
            "sllm_median": left["median"],
            "sllm_mad": left["mad"],
            "llama_median": right["median"],
            "llama_mad": right["mad"],
            "limit": limit,
            "pass": float(left["median"]) <= limit,
        }
    return result


def aggregate_summaries(sllm: Mapping[str, Any], llama: Mapping[str, Any], *, sllm_source: str = "<memory>", llama_source: str = "<memory>") -> dict[str, Any]:
    """Validate and aggregate two already-loaded producer summaries."""
    _validate_summary_header(sllm, "sllm")
    _validate_summary_header(llama, "llama")
    sllm_binary_sha = _optional_sha(sllm.get("binary"), "sllm summary binary") if "binary" in sllm else None
    llama_binary_sha = _optional_sha(llama.get("binary"), "llama summary binary") if "binary" in llama else None
    sllm_models = sllm.get("models")
    sllm_model_sha = _optional_sha(sllm_models.get("bf16"), "sllm summary BF16 model") if isinstance(sllm_models, dict) and "bf16" in sllm_models else None
    sllm_lock_sha = _optional_sha(sllm_models.get("bf16_lock"), "sllm summary BF16 lock") if isinstance(sllm_models, dict) and "bf16_lock" in sllm_models else None
    llama_model = llama.get("model")
    llama_model_sha = _optional_sha(llama_model, "llama summary model") if "model" in llama else None
    sllm_rows = {item.get("row", {}).get("case_id"): item for item in sllm["rows"] if isinstance(item, dict) and isinstance(item.get("row"), dict)}
    llama_rows = {item.get("row", {}).get("case_id"): item for item in llama["rows"] if isinstance(item, dict) and isinstance(item.get("row"), dict)}
    if set(sllm_rows) != set(CASE_IDS) or set(llama_rows) != set(CASE_IDS):
        _fail("producer summaries have missing or duplicate case rows")
    sllm_lock_fingerprints: list[str | None] = []
    for case_id in CASE_IDS:
        result = sllm_rows[case_id].get("result")
        model_identity = result.get("identities", {}).get("model") if isinstance(result, dict) and isinstance(result.get("identities"), dict) else None
        if isinstance(model_identity, dict) and "lock_fingerprint" in model_identity:
            fingerprint = model_identity.get("lock_fingerprint")
            if not isinstance(fingerprint, str) or not fingerprint:
                _fail(f"sllm/{case_id}: lock fingerprint is malformed")
            sllm_lock_fingerprints.append(fingerprint)
        else:
            sllm_lock_fingerprints.append(None)
    if any(value is not None for value in sllm_lock_fingerprints):
        if any(value is None for value in sllm_lock_fingerprints) or len(set(sllm_lock_fingerprints)) != 1:
            _fail("sLLM rows use inconsistent model lock fingerprints")
    rows: list[dict[str, Any]] = []
    totals = {"e2e_ns": True, "ttft_ns": True, "tpot_ns": True}
    for case_id, input_count, output_count in CASE_SPECS:
        left = _row_data(
            sllm_rows[case_id],
            "sllm",
            case_id,
            input_count,
            output_count,
            expected_binary_sha=sllm_binary_sha,
            expected_model_sha=sllm_model_sha,
            expected_lock_sha=sllm_lock_sha,
        )
        right = _row_data(
            llama_rows[case_id],
            "llama",
            case_id,
            input_count,
            output_count,
            expected_binary_sha=llama_binary_sha,
            expected_model_sha=llama_model_sha,
        )
        if left["input_ids"] != right["input_ids"]:
            _fail(f"{case_id}: input token sequence differs between engines")
        generated_equal = left["output_tokens"] == right["output_tokens"]
        visible_equal = left["visible_tokens"] == right["visible_tokens"]
        stop_equal = left["stop"] == right["stop"]
        # The two engines are E1 system-equivalent while their GGUF tensor
        # sets/converters differ.  Keep cross-engine output identity as a
        # bounded diagnostic; per-engine token/stop determinism and all
        # protocol/shape/cleanup checks above remain hard requirements.
        gates = _gate(left, right)
        for metric in METRICS:
            if metric == "tpot_ns" and output_count < 17:
                gates[metric]["pass"] = None
            elif not gates[metric]["pass"]:
                totals[metric] = False
        rows.append({
            "case_id": case_id,
            "input_token_count": input_count,
            "requested_output_tokens": output_count,
            "protocol": {
                "warmups": 1 if case_id in EXTENDED_CASES else 3,
                "measured": 3 if case_id in EXTENDED_CASES else 10,
                "context_length": 131_072 if case_id in EXTENDED_CASES else input_count + output_count,
                "ignore_eos": case_id == "decode-20000",
            },
            "row_ids": {"sllm": left["row_id"], "llama": right["row_id"]},
            "measured_sample_count": {"sllm": left["sample_count"], "llama": right["sample_count"]},
            "tokens": {
                "input_sha256": _sha(left["input_ids"]),
                "generated_sha256": {"sllm": _sha(left["output_tokens"]), "llama": _sha(right["output_tokens"])},
                "visible_sha256": {"sllm": _sha(left["visible_tokens"]), "llama": _sha(right["visible_tokens"])},
                "stop_sha256": {"sllm": _sha(left["stop"]), "llama": _sha(right["stop"])},
                "generated_equal": generated_equal,
                "visible_equal": visible_equal,
                "stop_equal": stop_equal,
            },
            "metrics": {"sllm": left["stats"], "llama": right["stats"]},
            "gates": gates,
        })
    gate = {
        "formula": "sLLM median <= llama.cpp median + max(sLLM MAD, llama.cpp MAD)",
        "e2e": totals["e2e_ns"],
        "ttft": totals["ttft_ns"],
        "tpot": totals["tpot_ns"],
        "all_pass": all(totals.values()),
    }
    return {
        "schema_version": "phase49-v620-summary-v1",
        "state": "PASS" if gate["all_pass"] else "FAIL",
        "target": TARGET,
        "gpu_uuid": GPU_UUID,
        "inputs": {
            "sllm": {"path": sllm_source, "sha256": _sha(sllm)},
            "llama": {"path": llama_source, "sha256": _sha(llama)},
        },
        "identities": {
            "sllm": {"schema_version": SLLM_SCHEMA, "engine": "sllm", "backend": "hip", "target": TARGET, "gpu_uuid": GPU_UUID},
            "llama": {"schema_version": LLAMA_SCHEMA, "engine": "llama.cpp", "commit": LLAMA_COMMIT, "tag": LLAMA_TAG, "target": TARGET, "gpu_uuid": GPU_UUID},
        },
        "matrix": {"cases": list(CASE_IDS), "row_count": len(rows)},
        "rows": rows,
        "gate": gate,
    }


def aggregate(sllm_path: Path, llama_path: Path) -> dict[str, Any]:
    """File-oriented convenience wrapper used by the CLI and host tests."""
    return aggregate_summaries(load_json(sllm_path), load_json(llama_path), sllm_source=str(sllm_path.resolve()), llama_source=str(llama_path.resolve()))


def _write_output(path: Path, document: Mapping[str, Any]) -> None:
    payload = canonical_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        try:
            if path.is_symlink() or path.read_bytes() != payload:
                _fail(f"refusing to overwrite existing aggregate: {path}")
        except OSError as exc:
            _fail(f"cannot inspect existing aggregate {path}: {exc}")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except OSError as exc:
        _fail(f"cannot publish aggregate {path}: {exc}")
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sllm-summary", "--sllm", dest="sllm_summary", required=True, type=Path)
    parser.add_argument("--llama-summary", "--llama", dest="llama_summary", required=True, type=Path)
    parser.add_argument("--output", "--output-path", dest="output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output is not None and args.output_dir is not None:
        print("FAIL: --output and --output-dir are mutually exclusive", file=sys.stderr)
        return 2
    output = args.output
    if args.output_dir is not None:
        output = args.output_dir / "phase49-v620-summary-v1.json"
    try:
        document = aggregate(args.sllm_summary, args.llama_summary)
        if output is not None:
            _write_output(output, document)
        print(json.dumps(document, ensure_ascii=False, sort_keys=True))
        return 0 if document["state"] == "PASS" else 1
    except Phase49Error as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
