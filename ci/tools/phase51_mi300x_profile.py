#!/usr/bin/env python3
"""Strict aggregation for the Phase 51 MI300X direct-profile lane.

Device time comes only from rocprofv3 ``kernel_stats``.  The kernel trace is
used for dispatch closure and interval-union evidence, never as a second copy
of device time.  Phase 51 direct benchmark reports use
``engine-performance-direct-v2`` and do not emit an MTP-width field; this tool
therefore reports MTP evidence as unavailable and makes no MTP-validation
claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, NoReturn, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, canonical_bytes  # noqa: E402


SCHEMA_VERSION = "phase51-mi300x-profile-v1"
DIRECT_SCHEMA_VERSION = "engine-performance-direct-v2"
TARGET = "gfx942"
MODEL_FINGERPRINT = "sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae"
INPUT_TOKENS = 10_001
OUTPUT_TOKENS = 2
CONTEXT_LENGTH = INPUT_TOKENS + OUTPUT_TOKENS
WARMUPS = 1
MEASURED = 3
EXPECTED_INPUT_ID = 23_066
EXPECTED_OUTPUT_IDS = [EXPECTED_INPUT_ID, EXPECTED_INPUT_ID]
MAX_RAW_BYTES = 128 * 1024 * 1024
CATEGORIES = ("projection", "full_attention", "gdn", "mtp_or_other")


class Phase51MI300XProfileError(ContractError):
    """Malformed or incomplete Phase 51 profile evidence."""


def _fail(message: str) -> NoReturn:
    raise Phase51MI300XProfileError(message)


def sha256_file(path: Path) -> str:
    """Hash one retained raw artifact, rejecting links and special files."""

    if path.is_symlink() or not path.is_file():
        _fail(f"required regular raw file is missing: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        _fail(f"cannot stat raw file {path}: {exc}")
    if size > MAX_RAW_BYTES:
        _fail(f"raw file is too large: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _fail(f"cannot read raw file {path}: {exc}")
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    sha256_file(path)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda token: _fail(f"non-finite JSON value: {token}"),
            )
    except Phase51MI300XProfileError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot read JSON {path}: {exc}")


def _read_csv(path: Path, required: Sequence[str], label: str) -> list[dict[str, str]]:
    """Read a bounded CSV and reject duplicate headers and ragged rows."""

    sha256_file(path)
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            try:
                header = [item.strip() for item in next(reader)]
            except StopIteration:
                _fail(f"{label} CSV is empty")
            if not header or any(not item for item in header):
                _fail(f"{label} CSV has an empty header")
            if len(set(header)) != len(header):
                _fail(f"{label} CSV has duplicate headers")
            missing = [item for item in required if item not in header]
            if missing:
                _fail(f"{label} CSV is missing columns: {', '.join(missing)}")
            rows: list[dict[str, str]] = []
            for line_number, values in enumerate(reader, start=2):
                if not values or all(not value.strip() for value in values):
                    _fail(f"{label} CSV has a blank row at line {line_number}")
                if len(values) != len(header):
                    _fail(f"{label} CSV row {line_number} has {len(values)} fields, expected {len(header)}")
                rows.append({key: value.strip() for key, value in zip(header, values)})
    except Phase51MI300XProfileError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        _fail(f"cannot read {label} CSV {path}: {exc}")
    if not rows:
        _fail(f"{label} CSV has no data rows")
    return rows


def _first_column(row: dict[str, str], names: Sequence[str], label: str) -> str:
    present = [(name, row[name]) for name in names if name in row]
    if not present:
        _fail(f"{label} row has no {('/'.join(names))} column")
    values = {value for _, value in present}
    if len(values) != 1:
        _fail(f"{label} row has conflicting aliases for {('/'.join(names))}")
    value = next(iter(values)).strip()
    if not value:
        _fail(f"{label} row has an empty {('/'.join(names))}")
    return value


def _positive_int(value: str, label: str) -> int:
    try:
        parsed = int(value, 10)
    except (TypeError, ValueError):
        _fail(f"{label} is not an integer: {value!r}")
    if parsed <= 0:
        _fail(f"{label} must be positive: {parsed}")
    return parsed


def _nonnegative_int(value: str, label: str) -> int:
    try:
        parsed = int(value, 10)
    except (TypeError, ValueError):
        _fail(f"{label} is not an integer: {value!r}")
    if parsed < 0:
        _fail(f"{label} must be nonnegative: {parsed}")
    return parsed


def _kernel_name(row: dict[str, str], label: str) -> str:
    return _first_column(row, ("Name", "Kernel_Name"), label)


def _matches_projection(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.startswith("cijk_")
        or "hipblasgemm" in lowered
        or "hipblasltmatmul" in lowered
        or "sllm_matmul_" in lowered
        or "matmul_" in lowered
    )


def _matches_full_attention(name: str) -> bool:
    return "causal_attention" in name.lower()


def _matches_gdn(name: str) -> bool:
    return "linear_attention" in name.lower()


def classify_kernel(name: str) -> str:
    """Assign exactly one Phase 36-compatible semantic category."""

    if not isinstance(name, str) or not name.strip():
        _fail("kernel name must be a non-empty string")
    name = name.strip()
    matches = [
        category
        for category, predicate in (
            ("projection", _matches_projection),
            ("full_attention", _matches_full_attention),
            ("gdn", _matches_gdn),
        )
        if predicate(name)
    ]
    if len(matches) > 1:
        _fail(f"kernel name is semantically ambiguous: {name}")
    return matches[0] if matches else "mtp_or_other"


def _trace_rows(rows: Iterable[dict[str, str]]) -> tuple[list[tuple[int, int, str, str]], int]:
    parsed: list[tuple[int, int, str, str]] = []
    dispatch_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        label = f"kernel trace row {index}"
        name = _kernel_name(row, label)
        start = _nonnegative_int(
            _first_column(row, ("Start_Timestamp", "StartNs", "Start"), label),
            f"{label} start",
        )
        end = _nonnegative_int(
            _first_column(row, ("End_Timestamp", "EndNs", "End"), label),
            f"{label} end",
        )
        if end <= start:
            _fail(f"{label} duration must be positive: start={start}, end={end}")
        dispatch = row.get("Dispatch_Id")
        if dispatch is not None and dispatch.strip():
            normalized = dispatch.strip()
            if normalized in dispatch_ids:
                _fail(f"duplicate Dispatch_Id in kernel trace: {normalized}")
            dispatch_ids.add(normalized)
        parsed.append((start, end, name, dispatch.strip() if dispatch else ""))
    if not parsed:
        _fail("kernel trace has no dispatches")
    intervals = sorted((start, end) for start, end, _name, _dispatch in parsed)
    interval_union = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            interval_union += current_end - current_start
            current_start, current_end = start, end
    interval_union += current_end - current_start
    return parsed, interval_union


def _stats_rows(rows: Iterable[dict[str, str]]) -> tuple[list[dict[str, Any]], int, int]:
    parsed: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, row in enumerate(rows, start=1):
        label = f"kernel stats row {index}"
        name = _kernel_name(row, label)
        if name in seen_names:
            _fail(f"duplicate kernel Name in stats: {name}")
        seen_names.add(name)
        calls = _positive_int(_first_column(row, ("Calls",), label), f"{label} Calls")
        duration = _positive_int(
            _first_column(row, ("TotalDurationNs",), label),
            f"{label} TotalDurationNs",
        )
        parsed.append(
            {
                "name": name,
                "calls": calls,
                "total_duration_ns": duration,
                "category": classify_kernel(name),
            }
        )
    if not parsed:
        _fail("kernel stats has no rows")
    total_calls = sum(item["calls"] for item in parsed)
    total_duration = sum(item["total_duration_ns"] for item in parsed)
    if total_calls <= 0 or total_duration <= 0:
        _fail("kernel stats has no positive calls or device duration")
    return parsed, total_calls, total_duration


def _auxiliary_rows(rows: Iterable[dict[str, str]], label: str) -> dict[str, int]:
    calls = 0
    duration = 0
    for index, row in enumerate(rows, start=1):
        row_label = f"{label} row {index}"
        calls += _nonnegative_int(_first_column(row, ("Calls",), row_label), f"{row_label} Calls")
        duration += _nonnegative_int(
            _first_column(row, ("TotalDurationNs",), row_label),
            f"{row_label} TotalDurationNs",
        )
    return {"calls": calls, "total_duration_ns": duration}


def _one_profile_file(profile_dir: Path, suffix: str) -> Path:
    if profile_dir.is_symlink() or not profile_dir.is_dir():
        _fail(f"profile directory is not a regular directory: {profile_dir}")
    matches = sorted(
        path
        for path in profile_dir.glob(f"*{suffix}")
        if not path.is_symlink() and path.is_file()
    )
    if len(matches) != 1:
        _fail(f"profile expected exactly one *{suffix}, found {len(matches)}")
    return matches[0]


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _values(document: Any, key: str) -> list[Any]:
    return [value for found, value in _walk(document) if found == key]


def _json_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _integer_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        _fail(f"{label} is not an integer list")
    return value


def _direct_result(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        _fail("execution JSON is not an object")
    if isinstance(document.get("result"), dict):
        if document.get("state") != "PASS":
            _fail("execution JSON outer state is not PASS")
        return document["result"]
    return document


def _validate_sample(sample: Any, label: str) -> None:
    if not isinstance(sample, dict):
        _fail(f"{label} is not an object")
    tokens = sample.get("tokens")
    if not isinstance(tokens, dict):
        _fail(f"{label} tokens object is absent")
    input_ids = _integer_list(tokens.get("input_token_ids"), f"{label} input token IDs")
    if len(input_ids) != INPUT_TOKENS or any(item != EXPECTED_INPUT_ID for item in input_ids):
        _fail(f"{label} input IDs are not exactly 10001 copies of token 23066")
    generated = _integer_list(tokens.get("generated_token_ids"), f"{label} generated output IDs")
    visible = _integer_list(tokens.get("visible_token_ids"), f"{label} visible output IDs")
    decode_input = _integer_list(tokens.get("decode_input_token_ids"), f"{label} decode input IDs")
    if generated != EXPECTED_OUTPUT_IDS or visible != EXPECTED_OUTPUT_IDS:
        _fail(f"{label} output count/IDs are not exactly {EXPECTED_OUTPUT_IDS}")
    if decode_input != EXPECTED_OUTPUT_IDS[:-1]:
        _fail(f"{label} decode input IDs do not match the two-token generation contract")
    stop = sample.get("stop")
    if not isinstance(stop, dict):
        _fail(f"{label} stop evidence is absent")
    if (
        stop.get("version") != 1
        or stop.get("reason_version") != 1
        or stop.get("kind") != "max_new_tokens"
        or stop.get("token_id") is not None
    ):
        _fail(f"{label} terminal stop is not the exact max_new_tokens contract")


def validate_execution_report(document: Any) -> dict[str, Any]:
    """Validate the exact current Phase 51 direct 10001/2 execution contract."""

    result = _direct_result(document)
    if (
        result.get("benchmark_schema_version") != DIRECT_SCHEMA_VERSION
        or result.get("state") != "PASS"
        or result.get("lane") != "direct"
    ):
        _fail("execution JSON is not a PASS engine-performance-direct-v2 direct report")

    identities = result.get("identities")
    if not isinstance(identities, dict):
        _fail("execution direct identities object is absent")
    if identities.get("engine") != "sllm" or identities.get("backend") != "hip":
        _fail("execution direct engine/backend identity is not sllm/HIP")
    if identities.get("target") != TARGET:
        _fail(f"execution direct target is not exact {TARGET}")
    model = identities.get("model")
    binding = identities.get("binding")
    if not isinstance(model, dict) or model.get("lock_fingerprint") != MODEL_FINGERPRINT:
        _fail("execution model lock fingerprint is absent or incorrect")
    if not isinstance(binding, dict) or binding.get("model_fingerprint") != MODEL_FINGERPRINT:
        _fail("execution binding model fingerprint is absent or incorrect")

    audit = result.get("audit")
    if not isinstance(audit, dict):
        _fail("execution aggregate audit is absent")
    if audit.get("selected_backend") != "hip":
        _fail("execution did not select HIP")
    if audit.get("target") != TARGET:
        _fail(f"execution audit target is not exact {TARGET}")
    if audit.get("all_dispatches_hip") is not True:
        _fail("execution lacks all-dispatches-HIP evidence")
    if audit.get("fallback_used") is not False:
        _fail("execution used fallback")
    if audit.get("model_fingerprint") != MODEL_FINGERPRINT:
        _fail("execution audit model fingerprint is absent or incorrect")

    targets = _values(result, "target")
    if not targets or any(value != TARGET for value in targets):
        _fail(f"execution contains a missing or non-{TARGET} target")
    selected_backends = _values(result, "selected_backend")
    if not selected_backends or any(value != "hip" for value in selected_backends):
        _fail("execution contains missing or non-HIP selected_backend evidence")
    dispatch_markers = _values(result, "all_dispatches_hip")
    if not dispatch_markers or any(value is not True for value in dispatch_markers):
        _fail("execution contains incomplete all-dispatches-HIP evidence")
    fallback_values = _values(result, "fallback_used")
    if not fallback_values or any(value is not False for value in fallback_values):
        _fail("execution fallback evidence is missing or non-false")
    for key in ("cpu_fallback_used", "partial_offload"):
        if any(value is not False for value in _values(result, key)):
            _fail(f"execution used fallback or partial offload: {key}")
    for key in ("model_fingerprint", "lock_fingerprint"):
        values = _values(result, key)
        if not values or any(value != MODEL_FINGERPRINT for value in values):
            _fail(f"execution {key} evidence is missing or conflicts")

    config = result.get("config")
    row = result.get("row")
    if not isinstance(config, dict) or not isinstance(row, dict):
        _fail("execution direct config/row identity is absent")
    if row.get("case_id") != "long-10001":
        _fail("execution row case is not long-10001")
    if config.get("input_token_count") != INPUT_TOKENS or row.get("input_token_count") != INPUT_TOKENS:
        _fail(f"execution input count is not {INPUT_TOKENS}")
    if config.get("max_new_tokens") != OUTPUT_TOKENS or row.get("requested_output_tokens") != OUTPUT_TOKENS:
        _fail(f"execution requested output count is not {OUTPUT_TOKENS}")
    if config.get("context_length") != CONTEXT_LENGTH:
        _fail(f"execution context length is not {CONTEXT_LENGTH}")
    if config.get("warmups") != WARMUPS or config.get("measured") != MEASURED:
        _fail(f"execution profiling repetitions are not {WARMUPS} warmup/{MEASURED} measured")

    warmups = result.get("warmups")
    measured = result.get("measured")
    if (
        not isinstance(warmups, dict)
        or warmups.get("count") != WARMUPS
        or not isinstance(warmups.get("samples"), list)
        or len(warmups["samples"]) != WARMUPS
    ):
        _fail("execution warmup count/sample list is not exactly 1")
    if (
        not isinstance(measured, dict)
        or measured.get("count") != MEASURED
        or not isinstance(measured.get("samples"), list)
        or len(measured["samples"]) != MEASURED
    ):
        _fail("execution measured count/sample list is not exactly 3")
    for section, samples in (("warmup", warmups["samples"]), ("measured", measured["samples"])):
        for index, sample in enumerate(samples):
            _validate_sample(sample, f"execution {section} sample {index}")

    control = result.get("correctness_control")
    _validate_sample(control, "execution correctness control")

    input_values = _values(result, "input_token_ids")
    if not input_values:
        _fail("execution input token IDs are absent")
    for value in input_values:
        input_ids = _integer_list(value, "execution input token IDs")
        if len(input_ids) != INPUT_TOKENS or any(item != EXPECTED_INPUT_ID for item in input_ids):
            _fail("execution input IDs are not exactly 10001 copies of token 23066")
    input_counts = _values(result, "input_token_count")
    if not input_counts or any(value != INPUT_TOKENS for value in input_counts):
        _fail(f"execution input count is missing or not {INPUT_TOKENS}")

    generated_values = _values(result, "generated_token_ids")
    if not generated_values:
        _fail("execution generated output IDs are absent")
    for value in generated_values:
        if _integer_list(value, "execution generated output IDs") != EXPECTED_OUTPUT_IDS:
            _fail(f"execution output IDs are not {EXPECTED_OUTPUT_IDS}")
    for value in _values(result, "visible_token_ids"):
        if _integer_list(value, "execution visible output IDs") != EXPECTED_OUTPUT_IDS:
            _fail(f"execution visible output IDs are not {EXPECTED_OUTPUT_IDS}")

    cleanup = result.get("cleanup")
    session_cleanup = result.get("session_cleanup")
    if not isinstance(cleanup, dict) or cleanup.get("all_requests_dropped") is not True:
        _fail("execution terminal cleanup/all-requests-dropped evidence is absent")
    if not isinstance(session_cleanup, dict):
        _fail("execution session cleanup evidence is absent")
    for key in ("retryable_cleanup", "durable_quarantine"):
        values = _values(result, key)
        if not values:
            _fail(f"execution cleanup field {key} is absent")
        if any(isinstance(value, bool) or not isinstance(value, int) or value != 0 for value in values):
            _fail(f"execution cleanup field {key} is not terminal-zero")

    return {
        "benchmark_schema_version": DIRECT_SCHEMA_VERSION,
        "target": TARGET,
        "selected_backend": "hip",
        "all_dispatches_hip": True,
        "fallback_used": False,
        "model_fingerprint": MODEL_FINGERPRINT,
        "input_tokens": INPUT_TOKENS,
        "input_ids_sha256": _json_sha([EXPECTED_INPUT_ID] * INPUT_TOKENS),
        "input_ids_mode": "all_equal_23066",
        "output_tokens": OUTPUT_TOKENS,
        "output_ids": EXPECTED_OUTPUT_IDS,
        "output_ids_sha256": _json_sha(EXPECTED_OUTPUT_IDS),
        "case_id": "long-10001",
        "context_length": CONTEXT_LENGTH,
        "warmups": WARMUPS,
        "measured": MEASURED,
        "cleanup_terminal_zero": True,
        "mtp": {
            "evidence_state": "unavailable",
            "source": "not-emitted",
            "validation_claimed": False,
            "reason": "engine-performance-direct-v2 does not emit an MTP-width field",
        },
    }


def aggregate_profile(
    profile_dir: Path,
    execution_json: Path,
    *,
    host_wall_ns: int | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Aggregate one strict rocprofv3 profile and optionally write a summary."""

    kernel_stats_path = _one_profile_file(profile_dir, "_kernel_stats.csv")
    hip_api_path = _one_profile_file(profile_dir, "_hip_api_stats.csv")
    memory_copy_path = _one_profile_file(profile_dir, "_memory_copy_stats.csv")
    kernel_trace_path = _one_profile_file(profile_dir, "_kernel_trace.csv")
    execution = _read_json(execution_json)
    execution_summary = validate_execution_report(execution)

    stats_rows = _read_csv(kernel_stats_path, ("Calls", "TotalDurationNs"), "kernel stats")
    trace_rows = _read_csv(kernel_trace_path, (), "kernel trace")
    hip_rows = _read_csv(hip_api_path, ("Calls", "TotalDurationNs"), "HIP API stats")
    copy_rows = _read_csv(memory_copy_path, ("Calls", "TotalDurationNs"), "memory copy stats")
    parsed_stats, total_calls, total_duration = _stats_rows(stats_rows)
    parsed_trace, interval_union = _trace_rows(trace_rows)
    stats_calls = {item["name"]: item["calls"] for item in parsed_stats}
    trace_calls = Counter(item[2] for item in parsed_trace)
    if total_calls != len(parsed_trace) or stats_calls != dict(trace_calls):
        _fail("kernel stats Calls do not close against kernel trace dispatches")

    buckets: dict[str, dict[str, Any]] = {
        category: {
            "category": category,
            "calls": 0,
            "total_duration_ns": 0,
            "device_time_share": 0.0,
            "kernel_names": [],
        }
        for category in CATEGORIES
    }
    unknown_names: list[str] = []
    for item in parsed_stats:
        bucket = buckets[item["category"]]
        bucket["calls"] += item["calls"]
        bucket["total_duration_ns"] += item["total_duration_ns"]
        bucket["kernel_names"].append(item["name"])
        if item["category"] == "mtp_or_other":
            unknown_names.append(item["name"])
    for bucket in buckets.values():
        bucket["kernel_names"] = sorted(bucket["kernel_names"])
        bucket["device_time_share"] = bucket["total_duration_ns"] / total_duration
    share_sum = sum(bucket["device_time_share"] for bucket in buckets.values())
    if abs(share_sum - 1.0) > 1.0e-12:
        _fail(f"kernel category shares do not close: {share_sum}")

    if host_wall_ns is not None:
        if isinstance(host_wall_ns, bool) or not isinstance(host_wall_ns, int) or host_wall_ns < 0:
            _fail("host wall duration must be a nonnegative integer")
        if host_wall_ns < interval_union:
            _fail(f"host wall duration {host_wall_ns} is shorter than kernel interval union {interval_union}")
        external: dict[str, Any] = {
            "state": "available",
            "host_wall_ns": host_wall_ns,
            "kernel_interval_union_ns": interval_union,
            "external_ns": host_wall_ns - interval_union,
            "external_share_of_host_wall": (
                (host_wall_ns - interval_union) / host_wall_ns if host_wall_ns else 0.0
            ),
        }
    else:
        external = {
            "state": "unavailable",
            "reason": "host wall duration was not supplied by the profiling invocation",
            "kernel_interval_union_ns": interval_union,
        }

    raw_paths = {
        "kernel_stats": kernel_stats_path,
        "kernel_trace": kernel_trace_path,
        "hip_api_stats": hip_api_path,
        "memory_copy_stats": memory_copy_path,
        "execution_json": execution_json,
    }
    raw_sha256 = {
        name: {"path": path.name, "sha256": sha256_file(path)}
        for name, path in raw_paths.items()
    }
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "state": "PASS",
        "target": TARGET,
        "observer_effect": "rocprofv3 runtime trace; profiled wall is diagnostic-only",
        "execution": execution_summary,
        "kernel": {
            "calls": total_calls,
            "total_duration_ns": total_duration,
            "categories": [buckets[category] for category in CATEGORIES],
            "category_share_sum": share_sum,
            "unknown_kernel_names": sorted(unknown_names),
            "trace_dispatches": len(parsed_trace),
            "trace_interval_union_ns": interval_union,
        },
        "hip_api": _auxiliary_rows(hip_rows, "HIP API stats"),
        "memory_copy": _auxiliary_rows(copy_rows, "memory copy stats"),
        "kernel_external": external,
        "raw_sha256": raw_sha256,
        "raw_manifest_sha256": _json_sha(raw_sha256),
    }
    if output_path is not None:
        if output_path.exists() and output_path.is_symlink():
            _fail(f"summary output must not be a symlink: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(canonical_bytes(summary))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--execution-json", type=Path, required=True)
    parser.add_argument("--host-wall-ns", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = aggregate_profile(
            args.profile_dir,
            args.execution_json,
            host_wall_ns=args.host_wall_ns,
            output_path=args.output,
        )
    except Phase51MI300XProfileError as exc:
        print(f"phase51 MI300X profile: FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
