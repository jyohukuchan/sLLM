#!/usr/bin/env python3
"""Aggregate a Phase 87 one-token decode rocprof trace.

This is a diagnostic reader for a completed ``rocprofv3`` run.  It keeps
kernel symbols unchanged and reports two byte domains separately:

* ``logical_read_bytes_estimate`` is a shape/encoding estimate supplied by
  the caller.  It is useful for a roofline table, but is not a DRAM counter.
* ``measured_dram_read_bytes`` is read from an optional counter CSV.  ROCm's
  GL2C/EA request counters are labelled as interface requests and are never
  silently presented as physical DRAM traffic.

The default decode segment is the dispatch interval between the last two
``sllm_token_selector_fixed_topk_final_v1`` markers.  That corresponds to one
committed target decode transition in the reviewed Qwen benchmark.  If a
trace does not contain that marker, the caller must provide an explicit
dispatch range; the tool does not guess from timestamps or kernel counts.

Raw profiler output is expected to remain under ``.local-artifacts``.  The
JSON summary binds every input file to a SHA-256 digest so that a tracked
summary cannot drift from the retained trace.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "phase87-decode-profile-v1"
FINAL_SELECTOR = "sllm_token_selector_fixed_topk_final_v1"
MAX_RAW_BYTES = 512 * 1024 * 1024
MAX_ROWS = 2_000_000
TARGETS = {"gfx1030", "gfx1201"}


class Phase87ProfileError(RuntimeError):
    """Raised when a profile cannot be interpreted fail-closed."""


def _fail(message: str) -> None:
    raise Phase87ProfileError(message)


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is not a regular file: {path}")
    try:
        if path.stat().st_size > MAX_RAW_BYTES:
            _fail(f"{label} is too large: {path}")
    except OSError as exc:
        _fail(f"cannot stat {label} {path}: {exc}")
    return path


def _sha256(path: Path) -> str:
    _regular(path, "raw input")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        _fail(f"cannot read {path}: {exc}")
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Any:
    _regular(path, label)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: _fail(
                f"non-finite JSON value in {label}: {value}"
            ),
        )
    except Phase87ProfileError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot parse {label}: {exc}")


def _find_one(profile_dir: Path, suffix: str) -> Path:
    if profile_dir.is_symlink() or not profile_dir.is_dir():
        _fail(f"profile directory is not a directory: {profile_dir}")
    matches = sorted(
        path
        for path in profile_dir.rglob(f"*{suffix}")
        if path.is_file() and not path.is_symlink()
    )
    if len(matches) != 1:
        _fail(f"expected exactly one *{suffix}, found {len(matches)}")
    return matches[0]


def _find_optional(profile_dir: Path, suffix: str) -> Path | None:
    if profile_dir.is_symlink() or not profile_dir.is_dir():
        _fail(f"profile directory is not a directory: {profile_dir}")
    matches = sorted(
        path
        for path in profile_dir.rglob(f"*{suffix}")
        if path.is_file() and not path.is_symlink()
    )
    if len(matches) > 1:
        _fail(f"expected at most one *{suffix}, found {len(matches)}")
    return matches[0] if matches else None


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    _regular(path, label)
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if not header:
                _fail(f"{label} is empty")
            header = [item.strip() for item in header]
            if any(not item for item in header) or len(set(header)) != len(header):
                _fail(f"{label} has an invalid or duplicate header")
            rows: list[dict[str, str]] = []
            for line_number, values in enumerate(reader, start=2):
                if not values or all(not value.strip() for value in values):
                    continue
                if len(rows) >= MAX_ROWS:
                    _fail(f"{label} exceeds bounded row count")
                if len(values) != len(header):
                    _fail(
                        f"{label} row {line_number} has {len(values)} fields; "
                        f"expected {len(header)}"
                    )
                rows.append({key: value.strip() for key, value in zip(header, values)})
    except Phase87ProfileError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        _fail(f"cannot read {label}: {exc}")
    if not rows:
        _fail(f"{label} has no data rows")
    return rows


def _column(row: Mapping[str, str], names: Sequence[str], label: str) -> str:
    found = [(name, row[name]) for name in names if name in row]
    if not found:
        _fail(f"{label} has no {'/'.join(names)} column")
    values = {value.strip() for _name, value in found}
    if len(values) != 1 or not next(iter(values)):
        _fail(f"{label} has conflicting or empty {'/'.join(names)} columns")
    return next(iter(values))


def _int(value: str, label: str, *, positive: bool = False) -> int:
    try:
        parsed = int(value.strip(), 10)
    except (AttributeError, TypeError, ValueError) as exc:
        _fail(f"{label} is not an integer: {value!r}")
    if parsed < 0 or (positive and parsed == 0):
        _fail(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return parsed


def _trace_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    seen_dispatch: set[int] = set()
    for index, row in enumerate(rows, start=1):
        label = f"kernel trace row {index}"
        name = _column(row, ("Kernel_Name", "KernelName", "Name", "kernel_name"), label)
        start = _int(_column(row, ("Start_Timestamp", "StartNs", "Start"), label), f"{label} start")
        end = _int(_column(row, ("End_Timestamp", "EndNs", "End"), label), f"{label} end")
        if end <= start:
            _fail(f"{label} has non-positive duration")
        dispatch_value = row.get("Dispatch_Id", "").strip()
        dispatch = _int(dispatch_value, f"{label} Dispatch_Id", positive=True) if dispatch_value else None
        if dispatch is not None:
            if dispatch in seen_dispatch:
                _fail(f"duplicate Dispatch_Id in kernel trace: {dispatch}")
            seen_dispatch.add(dispatch)
        parsed.append(
            {
                "name": name,
                "start_ns": start,
                "end_ns": end,
                "duration_ns": end - start,
                "dispatch_id": dispatch,
                "grid_x": _int(row.get("Grid_Size_X", "0"), f"{label} Grid_Size_X"),
            }
        )
    if not parsed:
        _fail("kernel trace has no dispatches")
    if any(item["dispatch_id"] is None for item in parsed):
        _fail("kernel trace must contain Dispatch_Id for deterministic decode segmentation")
    return sorted(parsed, key=lambda item: (item["dispatch_id"], item["start_ns"]))


def _union(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _marker_range(path: Path, marker_name: str, marker_index: int) -> tuple[int, int, dict[str, Any]]:
    rows = _read_csv(path, "marker trace")
    matches: list[tuple[int, int, str]] = []
    for index, row in enumerate(rows, start=1):
        label = f"marker trace row {index}"
        name = _column(
            row,
            ("Name", "Marker_Name", "MarkerName", "Message", "Range_Name", "Function"),
            label,
        )
        if marker_name not in name:
            continue
        start = _int(
            _column(row, ("Start_Timestamp", "StartNs", "Start", "Begin"), label),
            f"{label} start",
        )
        if any(key in row for key in ("End_Timestamp", "EndNs", "End")):
            end = _int(
                _column(row, ("End_Timestamp", "EndNs", "End"), label),
                f"{label} end",
            )
        else:
            duration = _int(
                _column(row, ("Duration", "DurationNs"), label),
                f"{label} duration",
            )
            end = start + duration
        if end <= start:
            _fail(f"{label} marker range is not positive")
        matches.append((start, end, name))
    if not matches:
        _fail(f"marker trace has no range containing {marker_name!r}")
    if marker_index < 0:
        marker_index += len(matches)
    if marker_index < 0 or marker_index >= len(matches):
        _fail(f"marker index {marker_index} is outside {len(matches)} matching ranges")
    start, end, name = matches[marker_index]
    return start, end, {
        "marker_name": name,
        "marker_index": marker_index,
        "matching_marker_count": len(matches),
        "start_timestamp_ns": start,
        "end_timestamp_ns": end,
    }


def _segment_from_marker(
    rows: list[dict[str, Any]], marker_path: Path, marker_name: str, marker_index: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start, end, marker = _marker_range(marker_path, marker_name, marker_index)
    if any(item["start_ns"] < end and item["end_ns"] > start
           and not (item["start_ns"] >= start and item["end_ns"] <= end)
           for item in rows):
        _fail("a kernel crosses the decode marker boundary; coverage is incomplete")
    selected = [item for item in rows if item["start_ns"] >= start and item["end_ns"] <= end]
    if not selected:
        _fail(f"marker {marker_name!r} contains no complete kernel dispatch")
    selected = sorted(selected, key=lambda item: item["dispatch_id"])
    first_id = _dispatch_id(selected[0])
    last_id = _dispatch_id(selected[-1])
    expected = list(range(first_id, last_id + 1))
    actual = [_dispatch_id(item) for item in selected]
    if actual != expected:
        _fail(
            f"marker {marker_name!r} maps to a non-contiguous dispatch range "
            f"{first_id}..{last_id}"
        )
    busy = _union((item["start_ns"], item["end_ns"]) for item in selected)
    return selected, {
        "boundary_source": "roctx_marker_range",
        "marker": marker,
        "start_dispatch_id": first_id,
        "end_dispatch_id": last_id,
        "dispatch_count": len(selected),
        "start_timestamp_ns": start,
        "end_timestamp_ns": end,
        "span_ns": end - start,
        "gpu_busy_union_ns": busy,
        "timestamp_gap_ns": end - start - busy,
        "fixed_topk_final_markers": sum(
            FINAL_SELECTOR in str(item["name"]) for item in selected
        ),
    }


def _dispatch_id(item: Mapping[str, Any]) -> int:
    value = item.get("dispatch_id")
    if not isinstance(value, int):
        _fail("internal trace row has no dispatch id")
    return value


def _decode_segment(
    rows: list[dict[str, Any]],
    *,
    explicit_start: int | None,
    explicit_end: int | None,
    transitions_from_end: int,
    full_marker_interval: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dispatches = [_dispatch_id(item) for item in rows]
    if explicit_start is not None or explicit_end is not None:
        if explicit_start is None or explicit_end is None:
            _fail("--decode-dispatch-start and --decode-dispatch-end must be paired")
        if explicit_start <= 0 or explicit_end < explicit_start:
            _fail("decode dispatch range is invalid")
        start_id, end_id = explicit_start, explicit_end
        boundary_source = "explicit_dispatch_range"
    else:
        marker_ids = [
            _dispatch_id(item)
            for item in rows
            if FINAL_SELECTOR in str(item["name"])
        ]
        if full_marker_interval:
            if len(marker_ids) < 2:
                _fail("trace has fewer than two fixed-top-k final markers")
            start_id = marker_ids[0] + 1
            end_id = marker_ids[-1]
            boundary_source = "fixed_topk_final_marker_full_interval"
        else:
            if transitions_from_end <= 0:
                _fail("--decode-transitions-from-end must be positive")
            if len(marker_ids) < transitions_from_end + 1:
                _fail(
                    f"trace has {len(marker_ids)} fixed-top-k final markers; "
                    f"need at least {transitions_from_end + 1} to isolate a decode transition"
                )
            # Marker i closes one target token.  The interval after the previous
            # marker and through the selected marker is one decode transition.
            end_id = marker_ids[-transitions_from_end]
            previous_id = marker_ids[-transitions_from_end - 1]
            start_id = previous_id + 1
            boundary_source = "fixed_topk_final_marker_interval"
    selected = [item for item in rows if start_id <= _dispatch_id(item) <= end_id]
    if not selected:
        _fail(f"decode dispatch range {start_id}..{end_id} selected no trace rows")
    if _dispatch_id(selected[0]) != start_id or _dispatch_id(selected[-1]) != end_id:
        _fail(
            f"decode dispatch range {start_id}..{end_id} is not contiguous in trace "
            f"(available {dispatches[0]}..{dispatches[-1]})"
        )
    start_ns = min(item["start_ns"] for item in selected)
    end_ns = max(item["end_ns"] for item in selected)
    busy_ns = _union((item["start_ns"], item["end_ns"]) for item in selected)
    return selected, {
        "boundary_source": boundary_source,
        "start_dispatch_id": start_id,
        "end_dispatch_id": end_id,
        "dispatch_count": len(selected),
        "start_timestamp_ns": start_ns,
        "end_timestamp_ns": end_ns,
        "span_ns": end_ns - start_ns,
        "gpu_busy_union_ns": busy_ns,
        "timestamp_gap_ns": end_ns - start_ns - busy_ns,
        "fixed_topk_final_markers": sum(
            FINAL_SELECTOR in str(item["name"]) for item in selected
        ),
    }


def _family(name: str) -> str:
    lowered = name.lower()
    if "causal_attention" in lowered or "flash_attn" in lowered:
        return "full_attention"
    if "kv_" in lowered or "kvstate" in lowered:
        return "kv_cache"
    if "linear_attention" in lowered:
        return "gdn"
    if any(s in lowered for s in ("to_nvfp4", "to_fp8", "to_mxfp", "quantize")):
        return "activation_quantize"
    if "nvfp4" in lowered or "w4a4" in lowered:
        return "nvfp4_matmul"
    if "fp8" in lowered or "w8a8" in lowered or "_f8bs_" in lowered:
        return "fp8_matmul"
    if "matmul_bf16" in lowered or "_bbs_" in lowered:
        return "bf16_matmul"
    if "gdn" in lowered:
        return "gdn"
    if "rmsnorm" in lowered or "norm" in lowered:
        return "rmsnorm_or_elementwise"
    if "selector" in lowered or "argmax" in lowered or "sample" in lowered:
        return "sampling"
    if "copy" in lowered or "memcpy" in lowered:
        return "runtime_copy"
    if "elementwise" in lowered:
        return "elementwise"
    return "other"


_SHAPE_RE = re.compile(r"(?:^|[_ ])k(\d+)[_ ]?n(\d+)(?:[_ ]|$)", re.IGNORECASE)


def _parse_shape_spec(value: str) -> tuple[str, int | None, int, int, int, str]:
    # Pattern: glob[@grid=GRID]=MxKxN:format.  The optional grid qualifier
    # separates two production shapes that share one generic kernel symbol.
    try:
        assignment, encoding = value.rsplit(":", 1)
        pattern, geometry = assignment.rsplit("=", 1)
        grid: int | None = None
        if "@grid=" in pattern:
            pattern, grid_text = pattern.rsplit("@grid=", 1)
            grid = int(grid_text)
        m_text, k_text, n_text = geometry.lower().split("x")
        m, k, n = (int(item) for item in (m_text, k_text, n_text))
    except (TypeError, ValueError) as exc:
        _fail(f"invalid --shape {value!r}; expected GLOB=MxKxN:FORMAT")
    if not pattern or (grid is not None and grid <= 0) or min(m, k, n) <= 0 or encoding not in {"nvfp4", "fp8", "bf16", "mxfp8", "mxfp6"}:
        _fail(f"invalid --shape {value!r}")
    return pattern, grid, m, k, n, encoding


def _shape_for(name: str, grid_x: int | None, specs: list[tuple[str, int | None, int, int, int, str]]) -> tuple[int, int, int, str] | None:
    matches = {
        (m, k, n, encoding)
        for pattern, expected_grid, m, k, n, encoding in specs
        if fnmatch.fnmatchcase(name, pattern)
        and (expected_grid is None or expected_grid == grid_x)
    }
    if len(matches) > 1:
        _fail(f"conflicting shape mappings for {name!r} grid={grid_x}")
    return next(iter(matches), None)


def _estimate_bytes(encoding: str, m: int, k: int, n: int) -> int:
    if encoding == "nvfp4":
        # E2M1 values are packed two per byte; block-16 E4M3 scales are one
        # byte each. Each row is packed separately. Weight and activation
        # tensor scales are two FP32 scalars, independent of M and N.
        return n * math.ceil(k / 2) + math.ceil(k / 16) * n + m * math.ceil(k / 2) + m * math.ceil(k / 16) + 8
    if encoding == "fp8":
        return k * n + 4 * n + m * k + 4 * m
    if encoding == "bf16":
        return 2 * (k * n + m * k)
    if encoding == "mxfp8":
        return k * n + math.ceil(k / 32) * n + m * k + math.ceil(k / 32) * m
    if encoding == "mxfp6":
        return math.ceil(k * n * 6 / 8) + math.ceil(k / 32) * n + math.ceil(m * k * 6 / 8) + math.ceil(k / 32) * m
    _fail(f"unsupported byte estimate encoding: {encoding}")


def _segment_kernel_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        name = str(row["name"])
        grid_x = int(row.get("grid_x", 0))
        item = grouped.setdefault(
            (name, grid_x),
            {
                "kernel_name": name,
                "grid_x": int(row.get("grid_x", 0)),
                "family": _family(name),
                "dispatch_count": 0,
                "duration_ns": 0,
            },
        )
        item["dispatch_count"] += 1
        item["duration_ns"] += int(row["duration_ns"])
    return sorted(grouped.values(), key=lambda item: (-item["duration_ns"], item["kernel_name"]))


def _attach_estimates(items: list[dict[str, Any]], specs: list[tuple[str, int | None, int, int, int, str]]) -> None:
    # A generic symbol can serve multiple production shapes.  A broad glob
    # without a grid qualifier is therefore rejected when it matches more
    # than one (symbol, grid) group; silently charging one shape to another
    # would corrupt the roofline table.
    for index, (pattern, expected_grid, _m, _k, _n, _encoding) in enumerate(specs):
        matches = {
            (item["kernel_name"], item.get("grid_x"))
            for item in items
            if fnmatch.fnmatchcase(item["kernel_name"], pattern)
            and (expected_grid is None or expected_grid == item.get("grid_x"))
        }
        if expected_grid is None and len(matches) > 1:
            _fail(
                f"shape glob {pattern!r} matches multiple kernel/grid groups; "
                "add an exact symbol or @grid qualifier"
            )
    for item in items:
        shape = _shape_for(item["kernel_name"], item.get("grid_x"), specs)
        item["logical_bytes_source"] = "unavailable_without_explicit_shape"
        item["logical_read_bytes_estimate"] = None
        item["logical_read_bytes_per_dispatch_estimate"] = None
        item["effective_bandwidth_gib_per_s_estimate"] = None
        if shape is None:
            continue
        m, k, n, encoding = shape
        per_dispatch = _estimate_bytes(encoding, m, k, n)
        bytes_estimate = per_dispatch * item["dispatch_count"]
        item["shape"] = {"m": m, "k": k, "n": n, "encoding": encoding}
        item["logical_read_bytes_estimate"] = bytes_estimate
        item["logical_read_bytes_per_dispatch_estimate"] = per_dispatch
        item["logical_bytes_source"] = "per_dispatch_shape_formula_times_dispatch_count"
        seconds = item["duration_ns"] / 1_000_000_000.0
        if seconds > 0:
            item["effective_bandwidth_gib_per_s_estimate"] = bytes_estimate / seconds / (1024**3)


def _stats_summary(rows: Iterable[Mapping[str, str]]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        label = f"kernel stats row {index}"
        name = _column(row, ("Name", "Kernel_Name", "KernelName"), label)
        calls = _int(_column(row, ("Calls",), label), f"{label} Calls", positive=True)
        duration = _int(_column(row, ("TotalDurationNs",), label), f"{label} TotalDurationNs", positive=True)
        values.append({"kernel_name": name, "family": _family(name), "calls": calls, "duration_ns": duration})
    if not values:
        _fail("kernel stats has no rows")
    return sorted(values, key=lambda item: (-item["duration_ns"], item["kernel_name"]))


def _trace_stats(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row["name"])
        item = grouped.setdefault(
            name,
            {"kernel_name": name, "family": _family(name), "calls": 0, "duration_ns": 0},
        )
        item["calls"] += 1
        item["duration_ns"] += int(row["duration_ns"])
    return sorted(grouped.values(), key=lambda item: (-item["duration_ns"], item["kernel_name"]))


def _counter_bytes(path: Path, start_dispatch_id: int, end_dispatch_id: int) -> dict[str, Any]:
    rows = _read_csv(path, "counter CSV")
    # rocprofv3's counter collection is long form: one row per counter per
    # launch instance.  The instance identity must include the correlation and
    # agent fields; grouping only by Dispatch_Id can merge launches from two
    # agents or two queues.  Aggregate ``GL2C_EA_RDREQ`` rows are ignored.
    required = ("Dispatch_Id", "Agent_Id", "Kernel_Name", "Counter_Name", "Counter_Value")
    missing = [name for name in required if name not in rows[0]]
    if missing:
        return {
            "state": "unavailable",
            "reason": "counter CSV is not rocprofv3 long form",
            "missing_columns": missing,
        }
    aliases: dict[str, set[str]] = {
        "32B": {"GL2C_EA_RDREQ_32B", "SLLM_GL2C_EA_RDREQ_32B"},
        "64B": {"GL2C_EA_RDREQ_64B", "SLLM_GL2C_EA_RDREQ_64B"},
        "96B": {"GL2C_EA_RDREQ_96B", "SLLM_GL2C_EA_RDREQ_96B"},
        "128B": {"GL2C_EA_RDREQ_128B", "SLLM_GL2C_EA_RDREQ_128B"},
        "256B": {"GL2C_EA_RDREQ_256B", "SLLM_GL2C_EA_RDREQ_256B"},
    }
    bin_bytes = {name: int(name[:-1]) for name in aliases}
    identity_fields = (
        "Correlation_Id",
        "Dispatch_Id",
        "Agent_Id",
        "Queue_Id",
        "Process_Id",
        "Thread_Id",
        "Kernel_Id",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "Grid_Size",
    )
    groups: dict[tuple[str, ...], dict[str, dict[str, list[float]]]] = {}
    ignored_aggregate: dict[str, int] = {}
    malformed_rows: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            dispatch = int(row["Dispatch_Id"])
        except (KeyError, TypeError, ValueError):
            malformed_rows.append(f"row {row_number}: invalid Dispatch_Id")
            continue
        if dispatch < start_dispatch_id or dispatch > end_dispatch_id:
            continue
        counter_name = row.get("Counter_Name", "").strip()
        normalized = next(
            (name for name, names in aliases.items() if counter_name in names), None
        )
        if normalized is None:
            if counter_name in {"GL2C_EA_RDREQ", "SLLM_GL2C_EA_RDREQ"}:
                ignored_aggregate[counter_name] = ignored_aggregate.get(counter_name, 0) + 1
            continue
        try:
            value = float(row["Counter_Value"])
        except (KeyError, TypeError, ValueError):
            malformed_rows.append(f"row {row_number}: invalid Counter_Value")
            continue
        if not math.isfinite(value) or value < 0.0:
            malformed_rows.append(f"row {row_number}: nonfinite or negative Counter_Value")
            continue
        key = tuple(row.get(field, "") for field in identity_fields)
        groups.setdefault(key, {}).setdefault(normalized, {}).setdefault(counter_name, []).append(value)

    # Infer the requested bin set from counters actually present.  gfx1201
    # normally has 32/64/128/256B; gfx1030 has 32/64/96/128B.  A partial CSV
    # remains visible as partial coverage instead of being silently treated as
    # complete evidence.
    present_bins = {
        name for group in groups.values() for name in group if name in aliases
    }
    if "256B" in present_bins:
        expected_bins = ("32B", "64B", "128B", "256B")
    elif "96B" in present_bins:
        expected_bins = ("32B", "64B", "96B", "128B")
    else:
        expected_bins = tuple(name for name in ("32B", "64B", "96B", "128B", "256B") if name in present_bins)
    if not expected_bins:
        return {
            "state": "unavailable",
            "reason": "decode dispatch rows have no explicit sized GL2C/EA RDREQ columns",
            "dispatch_range": [start_dispatch_id, end_dispatch_id],
            "ignored_aggregate_counters": ignored_aggregate,
            "malformed_rows": malformed_rows,
        }

    total_bytes = 0.0
    complete_instances = 0
    incomplete_instances = 0
    missing_by_dispatch: dict[str, int] = {}
    duplicate_aliases: list[dict[str, Any]] = []
    per_kernel: dict[str, dict[str, Any]] = {}
    per_kernel_grid: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in groups.items():
        values: dict[str, float] = {}
        alias_conflict = False
        for bin_name, by_alias in group.items():
            flattened = [value for values_list in by_alias.values() for value in values_list]
            if len(by_alias) > 1:
                duplicate_aliases.append(
                    {"dispatch_id": key[1], "kernel_name": key[7], "bin": bin_name, "aliases": sorted(by_alias)}
                )
            if not flattened:
                continue
            if max(flattened) != min(flattened):
                alias_conflict = True
                continue
            # One alias value is authoritative for a bin.  Do not add GL2C
            # and SLLM-prefixed aliases together.
            values[bin_name] = flattened[0]
        missing_bins = [name for name in expected_bins if name not in values]
        if alias_conflict or missing_bins:
            incomplete_instances += 1
            missing_by_dispatch[str(key[1])] = len(missing_bins) + int(alias_conflict)
            continue
        complete_instances += 1
        instance_bytes = sum(values[name] * bin_bytes[name] for name in expected_bins)
        total_bytes += instance_bytes
        kernel_name = key[7]
        entry = per_kernel.setdefault(
            kernel_name,
            {"complete_instance_count": 0, "read_request_bytes": 0.0, "bin_request_counts": {name: 0.0 for name in expected_bins}},
        )
        entry["complete_instance_count"] += 1
        entry["read_request_bytes"] += instance_bytes
        grid_entry = per_kernel_grid.setdefault(
            (kernel_name, key[10]),
            {"kernel_name": kernel_name, "grid_size": key[10],
             "complete_instance_count": 0, "read_request_bytes": 0.0},
        )
        grid_entry["complete_instance_count"] += 1
        grid_entry["read_request_bytes"] += instance_bytes
        for name in expected_bins:
            entry["bin_request_counts"][name] += values[name]

    state = "available_interface_request_estimate" if complete_instances else "unavailable"
    if incomplete_instances and complete_instances:
        state = "partial_interface_request_estimate"
    return {
        "state": state,
        "bytes": total_bytes if complete_instances else 0.0,
        "source": "rocprof_gl2c_ea_request_count_times_request_size",
        "caveat": "interface request bytes; do not call physical DRAM bus bytes",
        "dispatch_range": [start_dispatch_id, end_dispatch_id],
        "expected_bins": list(expected_bins),
        "present_bins": sorted(present_bins),
        "candidate_instance_count": len(groups),
        "complete_instance_count": complete_instances,
        "incomplete_instance_count": incomplete_instances,
        "missing_bins_by_dispatch": missing_by_dispatch,
        "duplicate_aliases": duplicate_aliases,
        "ignored_aggregate_counters": ignored_aggregate,
        "malformed_rows": malformed_rows,
        "per_kernel": dict(sorted(per_kernel.items())),
        "per_kernel_grid": list(per_kernel_grid.values()),
    }


def _validate_benchmark(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        _fail("benchmark JSON must be an object")
    if document.get("state") != "PASS":
        _fail("benchmark state is not PASS")
    target = document.get("target")
    if target is not None and target not in TARGETS:
        _fail(f"benchmark target is unsupported: {target}")
    protocol = document.get("protocol")
    if isinstance(protocol, dict) and protocol.get("active_requests") not in (None, 1):
        _fail("benchmark is not a single-request run")
    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        _fail("benchmark has no rows")
    first = rows[0]
    if not isinstance(first, dict):
        _fail("benchmark row is not an object")
    runs = first.get("runs")
    if not isinstance(runs, list) or not runs:
        _fail("benchmark row has no runs")
    run = runs[-1]
    audit = run.get("audit") if isinstance(run, dict) else None
    if not isinstance(audit, dict) or audit.get("all_dispatches_hip") is not True or audit.get("fallback_used") is not False:
        _fail("benchmark audit is missing or not HIP-only/no-fallback")
    return {
        "schema_version": document.get("schema_version"),
        "target": target,
        "benchmark_mode": document.get("benchmark_mode"),
        "protocol": protocol if isinstance(protocol, dict) else None,
        "mtp": document.get("mtp"),
        "row": {
            "prompt_tokens": first.get("prompt_tokens"),
            "output_tokens": first.get("output_tokens"),
            "decode_transition_count": run.get("decode_transition_count") if isinstance(run, dict) else None,
            "audit_kernel_dispatch_count": audit.get("kernel_dispatch_count") if isinstance(audit, dict) else None,
        },
    }


def aggregate(
    profile_dir: Path,
    *,
    benchmark_json: Path | None,
    counter_csv: Path | None,
    marker_csv: Path | None,
    marker_name: str,
    marker_index: int,
    shape_values: Sequence[str],
    decode_dispatch_start: int | None,
    decode_dispatch_end: int | None,
    decode_transitions_from_end: int,
    full_marker_interval: bool,
    output: Path | None,
) -> dict[str, Any]:
    trace_path = _find_one(profile_dir, "_kernel_trace.csv")
    stats_path = _find_optional(profile_dir, "_kernel_stats.csv")
    trace = _trace_rows(_read_csv(trace_path, "kernel trace"))
    stats = (
        _stats_summary(_read_csv(stats_path, "kernel stats"))
        if stats_path is not None
        else _trace_stats(trace)
    )
    benchmark = _validate_benchmark(_read_json(benchmark_json, "benchmark JSON")) if benchmark_json else {"state": "unavailable"}
    mtp_requested = bool(
        isinstance(benchmark.get("mtp"), dict)
        and benchmark["mtp"].get("requested") is True
    )
    if (
        (mtp_requested or benchmark_json is None)
        and marker_csv is None
        and decode_dispatch_start is None
        and decode_dispatch_end is None
    ):
        _fail(
            "MTP or unidentified decode boundaries require a ROCTX marker "
            "or explicit --decode-dispatch-start/--decode-dispatch-end"
        )
    if marker_csv is not None:
        selected, segment = _segment_from_marker(
            trace, marker_csv, marker_name, marker_index
        )
    else:
        selected, segment = _decode_segment(
            trace,
            explicit_start=decode_dispatch_start,
            explicit_end=decode_dispatch_end,
            transitions_from_end=decode_transitions_from_end,
            full_marker_interval=full_marker_interval,
        )
    specs = [_parse_shape_spec(value) for value in shape_values]
    kernels = _segment_kernel_rows(selected)
    _attach_estimates(kernels, specs)
    counters = (
        _counter_bytes(
            counter_csv,
            int(segment["start_dispatch_id"]),
            int(segment["end_dispatch_id"]),
        )
        if counter_csv
        else {"state": "unavailable", "reason": "no counter CSV supplied"}
    )
    raw_paths: dict[str, Path] = {"kernel_trace": trace_path}
    if stats_path is not None:
        raw_paths["kernel_stats"] = stats_path
    if benchmark_json:
        raw_paths["benchmark_json"] = benchmark_json
    if counter_csv:
        raw_paths["counter_csv"] = counter_csv
    if marker_csv:
        raw_paths["marker_csv"] = marker_csv
    raw_sha256 = {name: {"path": str(path), "sha256": _sha256(path)} for name, path in raw_paths.items()}
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "state": "PASS",
        "observer_effect": "rocprofv3 trace; profiled timing is diagnostic-only",
        "benchmark": benchmark,
        "trace": {"path": str(trace_path), "dispatch_count": len(trace), "dispatch_id_range": [trace[0]["dispatch_id"], trace[-1]["dispatch_id"]]},
        "decode_segment": segment,
        "decode_kernels": kernels,
        "whole_run_kernel_stats": {
            "source": "rocprof_kernel_stats_csv" if stats_path is not None else "kernel_trace_derived",
            "rows": stats,
        },
        "measured_dram_read": counters,
        "byte_domains": {
            "logical_read_bytes_estimate": "shape formula only; cache reuse and provider partitioning are not measured",
            "measured_dram_read_bytes": "counter domain only; absent unless counter CSV has byte/request columns",
        },
        "raw_sha256": raw_sha256,
        "raw_manifest_sha256": hashlib.sha256(json.dumps(raw_sha256, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    if output:
        if output.is_symlink():
            _fail(f"output must not be a symlink: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--benchmark-json", type=Path)
    parser.add_argument("--counter-csv", type=Path)
    parser.add_argument("--marker-csv", type=Path)
    parser.add_argument("--marker-name", default="sllm_phase87_decode_")
    parser.add_argument("--marker-index", type=int, default=-1)
    parser.add_argument("--shape", action="append", default=[], help="GLOB=MxKxN:FORMAT; repeat for exact kernels")
    parser.add_argument("--decode-dispatch-start", type=int)
    parser.add_argument("--decode-dispatch-end", type=int)
    parser.add_argument("--decode-transitions-from-end", type=int, default=1)
    parser.add_argument(
        "--decode-full-marker-interval",
        action="store_true",
        help="target-only: interval from first to last selector; MTP requires ROCTX or explicit dispatch bounds",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = aggregate(
            args.profile_dir,
            benchmark_json=args.benchmark_json,
            counter_csv=args.counter_csv,
            marker_csv=args.marker_csv,
            marker_name=args.marker_name,
            marker_index=args.marker_index,
            shape_values=args.shape,
            decode_dispatch_start=args.decode_dispatch_start,
            decode_dispatch_end=args.decode_dispatch_end,
            decode_transitions_from_end=args.decode_transitions_from_end,
            full_marker_interval=args.decode_full_marker_interval,
            output=args.output,
        )
    except Phase87ProfileError as exc:
        print(f"phase87 decode profile: FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
