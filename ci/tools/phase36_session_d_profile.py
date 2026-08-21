#!/usr/bin/env python3
"""Strict, bounded aggregation for the Phase 36 Session D profile lane.

The profiler is an observer lane.  Device time is taken from rocprofv3's
``kernel_stats`` report, while the trace is used only for the kernel interval
union and dispatch/resource evidence.  The host wall clock is never mixed into
the device total; it is reported as a separate ``kernel_external`` value.
Raw reports remain outside the repository.  The returned summary binds every
input report to a SHA-256 digest so a tracked summary cannot silently drift
from its retained raw evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, canonical_bytes  # noqa: E402


SCHEMA_VERSION = "phase36-session-d-profile-v1"
TARGET = "gfx942"
INPUT_TOKENS = 10_001
OUTPUT_TOKENS = 2
EXPECTED_INPUT_ID = 23_066
EXPECTED_OUTPUT_IDS = [EXPECTED_INPUT_ID, EXPECTED_INPUT_ID]
MAX_RAW_BYTES = 128 * 1024 * 1024
CATEGORIES = ("projection", "full_attention", "gdn", "mtp_or_other")


class SessionDProfileError(ContractError):
    """Raised when profile or execution evidence cannot pass closed."""


def _fail(message: str) -> None:
    raise SessionDProfileError(message)


def sha256_file(path: Path) -> str:
    """Hash one retained raw artifact, rejecting symlinks and special files."""

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
            return json.load(stream, object_pairs_hook=reject_duplicates, parse_constant=lambda token: _fail(f"non-finite JSON value: {token}"))
    except SessionDProfileError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot read JSON {path}: {exc}")


def _read_csv(path: Path, required: Sequence[str], label: str) -> list[dict[str, str]]:
    """Read a bounded CSV and reject duplicate headers or ragged rows."""

    sha256_file(path)
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            try:
                header = next(reader)
            except StopIteration:
                _fail(f"{label} CSV is empty")
            header = [item.strip() for item in header]
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
    except SessionDProfileError:
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
    except (TypeError, ValueError) as exc:
        _fail(f"{label} is not an integer: {value!r}")
    if parsed <= 0:
        _fail(f"{label} must be positive: {parsed}")
    return parsed


def _nonnegative_int(value: str, label: str) -> int:
    try:
        parsed = int(value, 10)
    except (TypeError, ValueError) as exc:
        _fail(f"{label} is not an integer: {value!r}")
    if parsed < 0:
        _fail(f"{label} must be nonnegative: {parsed}")
    return parsed


def _kernel_name(row: dict[str, str], label: str) -> str:
    return _first_column(row, ("Name", "Kernel_Name"), label)


def _duration_from_trace(row: dict[str, str], label: str) -> tuple[int, int, int]:
    start = _nonnegative_int(_first_column(row, ("Start_Timestamp", "StartNs", "Start"), label), f"{label} start")
    end = _nonnegative_int(_first_column(row, ("End_Timestamp", "EndNs", "End"), label), f"{label} end")
    if end <= start:
        _fail(f"{label} duration must be positive: start={start}, end={end}")
    return start, end, end - start


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
    """Return exactly one semantic bucket for a rocprof kernel name.

    ``mtp_or_other`` deliberately contains unknown names.  MTP has no unique
    device symbol in the current runtime, so a kernel name alone cannot claim
    that it belongs to the MTP graph.  Ambiguous names are rejected instead of
    being assigned by an accidental marker precedence.
    """

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
        start, end, _duration = _duration_from_trace(row, label)
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
    union = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            union += current_end - current_start
            current_start, current_end = start, end
    union += current_end - current_start
    return parsed, union


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
        duration = _positive_int(_first_column(row, ("TotalDurationNs",), label), f"{label} TotalDurationNs")
        parsed.append({"name": name, "calls": calls, "total_duration_ns": duration, "category": classify_kernel(name)})
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
        duration += _nonnegative_int(_first_column(row, ("TotalDurationNs",), row_label), f"{row_label} TotalDurationNs")
    return {"calls": calls, "total_duration_ns": duration}


def _one_profile_file(profile_dir: Path, suffix: str) -> Path:
    if profile_dir.is_symlink() or not profile_dir.is_dir():
        _fail(f"profile directory is not a regular directory: {profile_dir}")
    matches = sorted(path for path in profile_dir.glob(f"*{suffix}") if not path.is_symlink() and path.is_file())
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


def _validate_zero_fields(document: Any, key: str) -> None:
    values = _values(document, key)
    if not values:
        return
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            _fail(f"execution cleanup field {key} is not terminal-zero")


def validate_execution_report(document: Any) -> dict[str, Any]:
    """Validate the exact MI300X target-only 10001/2 execution contract."""

    if not isinstance(document, dict) or document.get("state") != "PASS":
        _fail("execution JSON state is not PASS")
    state_pass = _values(document, "pass")
    if state_pass and any(value is not True for value in state_pass):
        _fail("execution JSON pass marker is not true")

    targets = _values(document, "target")
    if not targets or any(value != TARGET for value in targets):
        _fail(f"execution target is not exact {TARGET}")
    backends = _values(document, "selected_backend")
    if not backends or any(value != "hip" for value in backends):
        _fail("execution did not select HIP")
    dispatch_markers = _values(document, "all_dispatches_hip")
    if not dispatch_markers or any(value is not True for value in dispatch_markers):
        _fail("execution lacks all-dispatches-HIP evidence")
    fallback_values = _values(document, "fallback_used")
    if not fallback_values:
        _fail("execution fallback_used marker is absent")
    for key in ("fallback_used", "cpu_fallback_used", "partial_offload"):
        values = _values(document, key)
        if any(value is not False for value in values):
            _fail(f"execution used fallback or partial offload: {key}")

    cleanups = [value for key, value in _walk(document) if key == "cleanup" and isinstance(value, dict)]
    if not cleanups:
        _fail("execution cleanup object is absent")
    cleanup_marker_count = 0
    for cleanup in cleanups:
        for key in ("retryable_cleanup", "durable_quarantine", "retryable", "durable", "final_current_bytes", "final_request_state_bytes", "final_workspace_bytes"):
            if key in cleanup:
                cleanup_marker_count += 1
                value = cleanup[key]
                if isinstance(value, bool) or not isinstance(value, int) or value != 0:
                    _fail(f"execution cleanup field {key} is not terminal-zero")
        for key in ("terminal_zero", "zero_after_shutdown"):
            if key in cleanup and cleanup[key] is not True:
                _fail(f"execution cleanup marker {key} is not true")
    if cleanup_marker_count == 0:
        _fail("execution cleanup has no terminal-zero fields")
    for key in ("retryable_cleanup", "durable_quarantine"):
        if not _values(document, key):
            _fail(f"execution cleanup field {key} is absent")
    for key in ("retryable_cleanup", "durable_quarantine"):
        _validate_zero_fields(document, key)

    input_ids_values = _values(document, "input_token_ids") + _values(document, "input_ids")
    input_ids: list[int] | None = None
    if input_ids_values:
        candidate = input_ids_values[0]
        if not isinstance(candidate, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in candidate):
            _fail("execution input IDs are not an integer list")
        if any(value != candidate for value in input_ids_values[1:]):
            _fail("execution input ID lists conflict")
        input_ids = candidate
        if len(input_ids) != INPUT_TOKENS:
            _fail(f"execution input token count is not {INPUT_TOKENS}")
        if any(value != EXPECTED_INPUT_ID for value in input_ids):
            _fail("execution input IDs are not all the expected 23066 token")

    input_counts = _values(document, "input_token_count") + _values(document, "input_tokens") + _values(document, "prompt_tokens")
    if input_ids is None:
        if not input_counts or any(value != INPUT_TOKENS for value in input_counts):
            _fail(f"execution input count is not {INPUT_TOKENS}")
        digest_values = _values(document, "input_ids_sha256") + _values(document, "input_token_ids_sha256") + _values(document, "input_ids_digest")
        expected_digest = _json_sha([EXPECTED_INPUT_ID] * INPUT_TOKENS)
        if not digest_values or any(value != expected_digest for value in digest_values):
            _fail("execution input IDs digest is absent or does not match the expected sequence")
    elif input_counts and any(value != INPUT_TOKENS for value in input_counts):
        _fail(f"execution input count is not {INPUT_TOKENS}")

    output_values = _values(document, "generated_token_ids") + _values(document, "output_ids") + _values(document, "visible_token_ids")
    if not output_values:
        _fail("execution output IDs are absent")
    output_ids = output_values[0]
    if any(value != output_ids for value in output_values[1:]) or output_ids != EXPECTED_OUTPUT_IDS:
        _fail(f"execution output IDs are not {EXPECTED_OUTPUT_IDS}")
    output_counts = _values(document, "output_token_count") + _values(document, "output_tokens") + _values(document, "completion_tokens")
    if output_counts and any(value != OUTPUT_TOKENS for value in output_counts):
        _fail(f"execution output count is not {OUTPUT_TOKENS}")

    mtp_values = []
    for key in ("mtp_draft_width", "mtp_draft_width_requested", "mtp_width"):
        mtp_values.extend(_values(document, key))
    if not mtp_values or any(isinstance(value, bool) or not isinstance(value, int) or value != 0 for value in mtp_values):
        _fail("execution is not an explicit target-only MTP width-0 run")

    return {
        "target": TARGET,
        "selected_backend": "hip",
        "all_dispatches_hip": True,
        "fallback_used": False,
        "input_tokens": INPUT_TOKENS,
        "input_ids_sha256": _json_sha(input_ids if input_ids is not None else [EXPECTED_INPUT_ID] * INPUT_TOKENS),
        "input_ids_mode": "all_equal_23066" if input_ids is not None else "digest",
        "output_tokens": OUTPUT_TOKENS,
        "output_ids": EXPECTED_OUTPUT_IDS,
        "output_ids_sha256": _json_sha(EXPECTED_OUTPUT_IDS),
        "mtp_draft_width": 0,
        "cleanup_terminal_zero": True,
    }


def _host_wall_from_execution(document: Any) -> int | None:
    values: list[Any] = []
    for key in ("host_wall_ns", "host_wall_duration_ns", "wall_time_ns", "e2e_ns"):
        values.extend(_values(document, key))
    if not values:
        return None
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        _fail("execution host wall duration is invalid")
    if any(value != values[0] for value in values[1:]):
        _fail("execution host wall durations conflict")
    return values[0]


def aggregate_profile(
    profile_dir: Path,
    execution_json: Path,
    *,
    host_wall_ns: int | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Aggregate one strict rocprofv3 profile and optionally write its summary."""

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
    hip_summary = _auxiliary_rows(hip_rows, "HIP API stats")
    copy_summary = _auxiliary_rows(copy_rows, "memory copy stats")

    buckets: dict[str, dict[str, Any]] = {
        category: {"category": category, "calls": 0, "total_duration_ns": 0, "device_time_share": 0.0, "kernel_names": []}
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

    wall = host_wall_ns if host_wall_ns is not None else _host_wall_from_execution(execution)
    if wall is not None:
        if isinstance(wall, bool) or not isinstance(wall, int) or wall < 0:
            _fail("host wall duration must be a nonnegative integer")
        if wall < interval_union:
            _fail(f"host wall duration {wall} is shorter than kernel interval union {interval_union}")
        external: dict[str, Any] = {
            "state": "available",
            "host_wall_ns": wall,
            "kernel_interval_union_ns": interval_union,
            "external_ns": wall - interval_union,
            "external_share_of_host_wall": (wall - interval_union) / wall if wall else 0.0,
        }
    else:
        external = {
            "state": "unavailable",
            "reason": "host wall duration was not supplied by the execution report or CLI",
            "kernel_interval_union_ns": interval_union,
        }

    raw_paths = {
        "kernel_stats": kernel_stats_path,
        "kernel_trace": kernel_trace_path,
        "hip_api_stats": hip_api_path,
        "memory_copy_stats": memory_copy_path,
        "execution_json": execution_json,
    }
    raw_sha256 = {name: {"path": path.name, "sha256": sha256_file(path)} for name, path in raw_paths.items()}
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
            "trace_dispatches": len(trace_rows),
            "trace_interval_union_ns": interval_union,
        },
        "hip_api": hip_summary,
        "memory_copy": copy_summary,
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
        summary = aggregate_profile(args.profile_dir, args.execution_json, host_wall_ns=args.host_wall_ns, output_path=args.output)
    except SessionDProfileError as exc:
        print(f"phase36 Session D profile: FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
