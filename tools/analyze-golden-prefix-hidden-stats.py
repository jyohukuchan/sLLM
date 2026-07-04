#!/usr/bin/env python3
"""Summarize hidden-distribution metrics from golden-prefix JSONL results.

The current generator for distribution metrics is still evolving, so this script
parses legacy and newer schemas defensively. Missing fields are tolerated and
skipped from aggregation rather than failing the run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "golden-prefix-hidden-stats-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize golden-prefix hidden-distribution rows across JSONL files."
    )
    parser.add_argument(
        "jsonl_paths",
        nargs="+",
        type=Path,
        help="One or more JSONL files to analyze.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=False,
        help="Write summary JSON to this path.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        required=False,
        help="Write Markdown table to this path.",
    )
    return parser.parse_args()


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def parse_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def normalize_run_mode(raw: Any) -> str:
    value = parse_str(raw)
    if not value:
        return "actual_prefix"
    return value


def safe_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"JSONL path is not a file: {path}")


@dataclass(frozen=True)
class LayerLocation:
    token: int | None = None
    hidden: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {"token": self.token, "hidden": self.hidden}


@dataclass(frozen=True)
class GroupKey:
    device_index: int | None
    backend: str
    run_mode: str
    layer_start: int | None
    layer_end_exclusive: int | None
    layer_index: int | None


@dataclass
class DistributionMetric:
    mean: float | None = None
    rms: float | None = None
    l2_norm: float | None = None
    max_abs_diff: float | None = None
    max_abs_diff_location: LayerLocation | None = None
    per_token_max_mse: float | None = None
    per_token_max_mse_location: LayerLocation | None = None
    top_abs_diff_locations: list[LayerLocation] | None = None


@dataclass
class ParsedRow:
    key: GroupKey
    source: str
    output: DistributionMetric
    input: DistributionMetric
    output_failure_class: str | None
    input_failure_class: str | None


def _safe_rms_from_mse(mse: float | None) -> float | None:
    if mse is None:
        return None
    if mse < 0:
        return None
    return math.sqrt(mse)


def _first_non_none(values: list[Any]) -> Any:
    for value in values:
        parsed = parse_float(value)
        if parsed is not None:
            return parsed
    return None


def _parse_location(raw: Any) -> LayerLocation | None:
    if raw is None:
        return None
    if isinstance(raw, LayerLocation):
        return raw
    if isinstance(raw, int):
        return LayerLocation(token=parse_int(raw), hidden=None)
    if isinstance(raw, float):
        return LayerLocation(token=parse_int(raw), hidden=None)
    if isinstance(raw, (list, tuple)) and len(raw) >= 1:
        token = parse_int(raw[0])
        hidden = parse_int(raw[1]) if len(raw) >= 2 else None
        return LayerLocation(token=token, hidden=hidden)
    if isinstance(raw, dict):
        token = _first_non_none(
            [
                raw.get("token"),
                raw.get("token_idx"),
                raw.get("token_index"),
                raw.get("position"),
            ]
        )
        hidden = _first_non_none(
            [
                raw.get("hidden"),
                raw.get("hidden_idx"),
                raw.get("hidden_index"),
                raw.get("hidden_position"),
            ]
        )
        if token is not None or hidden is not None:
            return LayerLocation(token=parse_int(token), hidden=parse_int(hidden))
    return None


def _parse_max_abs_diff_location(payload: Any) -> LayerLocation | None:
    if not isinstance(payload, dict):
        return _parse_location(payload)
    location_candidates = [
        payload.get("max_abs_diff_location"),
        payload.get("max_abs_diff_pos"),
        payload.get("max_abs_diff_index"),
        payload.get("location"),
        payload.get("index"),
    ]
    for candidate in location_candidates:
        location = _parse_location(candidate)
        if location is not None:
            return location
    return None


def _parse_top_abs_diff_locations(payload: Any) -> list[LayerLocation]:
    if not isinstance(payload, dict):
        return []
    raw_locations = payload.get("top_abs_diff_locations")
    if not isinstance(raw_locations, list):
        return []
    locations: list[LayerLocation] = []
    for raw in raw_locations:
        location = _parse_location(raw)
        if location is not None:
            locations.append(location)
    return locations


def _extract_max_token_metric(payload: Any) -> tuple[float | None, LayerLocation | None]:
    if payload is None:
        return None, None

    explicit_candidates = [
        parse_float(payload.get("max_mse")) if isinstance(payload, dict) else None,
        parse_float(payload.get("mse_max")) if isinstance(payload, dict) else None,
        parse_float(payload.get("max_abs_mse")) if isinstance(payload, dict) else None,
    ]
    explicit_value = _first_non_none([v for v in explicit_candidates if v is not None])
    if explicit_value is not None:
        if isinstance(payload, dict):
            return (
                explicit_value,
                _parse_location(
                    _first_non_none(
                        [
                            payload.get("max_mse_location"),
                            payload.get("max_mse_index"),
                            payload.get("argmax"),
                            payload.get("index"),
                        ]
                    )
                ),
            )
        return explicit_value, None

    if isinstance(payload, list):
        dict_values: list[tuple[float, LayerLocation | None]] = []
        scalar_values: list[tuple[float, int]] = []
        for idx, item in enumerate(payload):
            if isinstance(item, dict):
                value = parse_float(item.get("mse"))
                if value is None:
                    value = parse_float(item.get("per_token_mse"))
                if value is None:
                    value = parse_float(item.get("max_mse"))
                if value is None:
                    continue
                location = _parse_location(item.get("diff_max_abs_location"))
                if location is None:
                    token_index = parse_int(item.get("token_index"))
                    location = LayerLocation(
                        token=token_index if token_index is not None else idx,
                        hidden=None,
                    )
                dict_values.append((value, location))
            else:
                value = parse_float(item)
                if value is not None:
                    scalar_values.append((value, idx))

        if dict_values:
            max_value, location = max(dict_values, key=lambda item: item[0])
            return max_value, location

        values = scalar_values
        values = [(v, idx) for v, idx in values if v is not None]
        if not values:
            return None, None
        max_value, max_idx = max(values, key=lambda item: item[0])
        return max_value, LayerLocation(token=max_idx, hidden=None)

    if not isinstance(payload, dict):
        return None, None

    for key in ("mse", "max_mse_by_token", "mse_by_token", "values"):
        values_obj = payload.get(key)
        if isinstance(values_obj, list):
            values = [(parse_float(value), idx) for idx, value in enumerate(values_obj)]
            values = [(value, idx) for value, idx in values if value is not None]
            if not values:
                continue
            max_value, max_idx = max(values, key=lambda item: item[0])
            return max_value, LayerLocation(token=max_idx, hidden=None)
    for key, values_obj in payload.items():
        if isinstance(values_obj, list):
            values = [(parse_float(value), idx) for idx, value in enumerate(values_obj)]
            values = [(value, idx) for value, idx in values if value is not None]
            if not values:
                continue
            max_value, max_idx = max(values, key=lambda item: item[0])
            return max_value, LayerLocation(token=max_idx, hidden=None)
    return None, None


def _parse_distribution_metrics(payload: Any, legacy: dict[str, Any]) -> DistributionMetric:
    if not isinstance(payload, dict):
        return _legacy_distribution_metrics(legacy)

    # Prefer explicit diff stats, then older diff payloads, then top-level.
    diff_payload = payload.get("diff_stats")
    if not isinstance(diff_payload, dict):
        diff_payload = payload.get("diff")
    source = diff_payload if isinstance(diff_payload, dict) else payload

    mean = _first_non_none(
        [
            source.get("mean"),
            source.get("mean_abs_diff"),
            source.get("avg"),
            source.get("avg_abs_diff"),
        ]
    )
    rms = _first_non_none(
        [
            source.get("rms"),
            source.get("root_mean_square"),
            source.get("rmse"),
            source.get("input_rmse"),
            source.get("output_rmse"),
        ]
    )
    l2_norm = parse_float(source.get("l2_norm")) or parse_float(source.get("l2"))
    max_abs_diff = _first_non_none(
        [
            source.get("max_abs_diff"),
            source.get("max_abs_error"),
            source.get("max_abs"),
            source.get("max_error"),
        ]
    )
    max_abs_location = _parse_max_abs_diff_location(source)
    if max_abs_location is None:
        max_abs_location = _parse_max_abs_diff_location(payload)
    top_abs_locations = _parse_top_abs_diff_locations(payload)

    per_token_payload = None
    for candidate in (
        payload.get("per_token"),
        source.get("per_token"),
        payload.get("per_token_metrics"),
        source.get("per_token_metrics"),
    ):
        if candidate is not None:
            per_token_payload = candidate
            break
    per_token_max_mse, per_token_max_mse_location = _extract_max_token_metric(per_token_payload)
    # location from top-level hints if per-token payload did not include one.
    if per_token_max_mse is not None and per_token_max_mse_location is None:
        if isinstance(source, dict):
            per_token_max_mse_location = _parse_location(source.get("per_token_max_mse_location"))
        if per_token_max_mse_location is None and isinstance(payload, dict):
            per_token_max_mse_location = _parse_location(payload.get("per_token_max_mse_location"))

    return DistributionMetric(
        mean=parse_float(mean),
        rms=parse_float(rms),
        l2_norm=parse_float(l2_norm),
        max_abs_diff=parse_float(max_abs_diff),
        max_abs_diff_location=max_abs_location,
        per_token_max_mse=per_token_max_mse,
        per_token_max_mse_location=per_token_max_mse_location,
        top_abs_diff_locations=top_abs_locations,
    )


def _legacy_distribution_metrics(legacy: dict[str, Any]) -> DistributionMetric:
    mean = parse_float(legacy.get("mean_abs_diff"))
    mse = parse_float(legacy.get("mse"))
    rms = _safe_rms_from_mse(parse_float(legacy.get("output_mse")))
    return DistributionMetric(
        mean=mean,
        rms=rms,
        l2_norm=None,
        max_abs_diff=parse_float(legacy.get("max_abs_diff")),
        max_abs_diff_location=None,
        per_token_max_mse=parse_float(legacy.get("per_token_max_mse")),
        per_token_max_mse_location=_parse_location(
            legacy.get("per_token_max_mse_location")
            or legacy.get("max_abs_diff_location")
            or legacy.get("max_abs_location")
        ),
        top_abs_diff_locations=[],
    )


def _parse_row(item: dict[str, Any], source: str) -> ParsedRow:
    if not isinstance(item, dict):
        raise ValueError("JSONL row is not an object")

    key = GroupKey(
        device_index=parse_int(item.get("device_index")),
        backend=parse_str(item.get("backend")) or "unknown",
        run_mode=normalize_run_mode(item.get("run_mode")),
        layer_start=parse_int(item.get("layer_start")),
        layer_end_exclusive=parse_int(item.get("layer_end_exclusive")),
        layer_index=parse_int(item.get("layer_index")),
    )

    output = _parse_distribution_metrics(item.get("output_distribution"), item)
    if output.rms is None:
        output.rms = _safe_rms_from_mse(parse_float(item.get("mse")))
    if output.mean is None:
        output.mean = parse_float(item.get("mean_abs_diff"))

    # Per-token max MSE can be recorded explicitly at top-level in old rows.
    legacy_output_per_token = parse_float(item.get("per_token_max_mse"))
    if legacy_output_per_token is not None and output.per_token_max_mse is None:
        output.per_token_max_mse = legacy_output_per_token

    input_payload = item.get("input_distribution")
    input_metrics = _parse_distribution_metrics(input_payload, item)
    input_legacy_mean = parse_float(item.get("input_mean_abs_diff"))
    input_legacy_rms = _safe_rms_from_mse(parse_float(item.get("input_mse")))
    input_legacy_max_abs = parse_float(item.get("input_max_abs_diff"))
    if input_payload is None:
        input_metrics.mean = input_legacy_mean
        input_metrics.rms = input_legacy_rms
        input_metrics.max_abs_diff = input_legacy_max_abs
    else:
        # Fill any missing input values from explicit legacy input fields.
        if input_legacy_mean is not None:
            input_metrics.mean = input_legacy_mean
        if input_legacy_rms is not None:
            input_metrics.rms = input_legacy_rms
        if input_legacy_max_abs is not None and input_metrics.max_abs_diff is None:
            input_metrics.max_abs_diff = input_legacy_max_abs
        # Backward-compatibility: keep legacy per-token summary if parser missed it.
        if input_metrics.rms is None:
            input_metrics.rms = input_legacy_rms
    if input_payload is None and parse_float(item.get("input_per_token_max_mse")) is not None:
        input_metrics.per_token_max_mse = parse_float(item.get("input_per_token_max_mse"))

    return ParsedRow(
        key=key,
        source=source,
        output=output,
        input=input_metrics,
        output_failure_class=parse_str(item.get("failure_class")),
        input_failure_class=parse_str(item.get("input_failure_class")),
    )


def read_jsonl(path: Path) -> list[ParsedRow]:
    rows: list[ParsedRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number} row is not an object")
            rows.append(_parse_row(item, source=str(path)))
    return rows


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


@dataclass
class LayerSummary:
    key: GroupKey
    count: int
    input_mean: float | None
    input_rms: float | None
    input_l2_norm: float | None
    output_mean: float | None
    output_rms: float | None
    output_l2_norm: float | None
    output_max_abs_diff: float | None
    output_max_abs_diff_location: LayerLocation | None
    output_per_token_max_mse: float | None
    output_per_token_max_mse_location: LayerLocation | None
    output_top_hidden_counts: dict[str, int]
    failure_class: str | None
    samples: list[ParsedRow]


def _pick_failure_class(rows: list[ParsedRow]) -> str | None:
    candidates: list[str | None] = []
    for row in rows:
        if row.output_failure_class:
            candidates.append(row.output_failure_class)
        if row.input_failure_class:
            candidates.append(row.input_failure_class)

    if not candidates:
        return None

    def score(value: str) -> tuple[int, int, str]:
        lowered = value.lower()
        if any(x in lowered for x in ["drift", "error", "nan", "inf", "overflow"]):
            return (3, len(lowered), value)
        if "possible_quantization_error" in lowered or "quantization" in lowered:
            return (2, len(lowered), value)
        if "possible" in lowered:
            return (1, len(lowered), value)
        return (0, len(lowered), value)

    return sorted(candidates, key=lambda v: score(v), reverse=True)[0]


def _aggregate(rows: list[ParsedRow]) -> LayerSummary:
    output_rms: list[float] = []
    output_mean: list[float] = []
    output_l2: list[float] = []
    input_rms: list[float] = []
    input_mean: list[float] = []
    input_l2: list[float] = []
    output_per_token: list[tuple[float, LayerLocation | None]] = []
    top_hidden_counts: Counter[str] = Counter()

    output_max_abs = None
    output_max_abs_location = None
    for row in rows:
        out = row.output
        inp = row.input

        if out.rms is not None:
            output_rms.append(out.rms)
        if out.mean is not None:
            output_mean.append(out.mean)
        if out.l2_norm is not None:
            output_l2.append(out.l2_norm)
        if inp.rms is not None:
            input_rms.append(inp.rms)
        if inp.mean is not None:
            input_mean.append(inp.mean)
        if inp.l2_norm is not None:
            input_l2.append(inp.l2_norm)
        if out.per_token_max_mse is not None:
            output_per_token.append((out.per_token_max_mse, out.per_token_max_mse_location))
        for location in out.top_abs_diff_locations or []:
            if location.hidden is not None:
                top_hidden_counts[str(location.hidden)] += 1

        if out.max_abs_diff is not None and (
            output_max_abs is None or abs(out.max_abs_diff) > abs(output_max_abs)
        ):
            output_max_abs = out.max_abs_diff
            output_max_abs_location = out.max_abs_diff_location

    return LayerSummary(
        key=rows[0].key,
        count=len(rows),
        input_mean=mean(input_mean),
        input_rms=mean(input_rms),
        input_l2_norm=mean(input_l2),
        output_mean=mean(output_mean),
        output_rms=mean(output_rms),
        output_l2_norm=mean(output_l2),
        output_max_abs_diff=output_max_abs,
        output_max_abs_diff_location=output_max_abs_location,
        output_per_token_max_mse=(
            max((value for value, _ in output_per_token), default=None)
        ),
        output_per_token_max_mse_location=(
            sorted(output_per_token, key=lambda item: item[0], reverse=True)[0][1]
            if output_per_token
            else None
        ),
        output_top_hidden_counts=dict(top_hidden_counts.most_common(12)),
        failure_class=_pick_failure_class(rows),
        samples=rows,
    )


def range_label(start: int | None, end: int | None) -> str:
    if start is None and end is None:
        return "?"
    if start is None:
        return f"?..{end}"
    if end is None:
        return f"{start}..?"
    return f"{start}..{end}"


def device_backend_label(device_index: int | None, backend: str) -> str:
    if device_index is None:
        return backend
    return f"{device_index}/{backend}"


def fmt_num(value: float | None, width: int = 6) -> str:
    if value is None:
        return "-"
    return f"{value:.{width}f}"


def fmt_loc(value: int | None) -> str:
    return "-" if value is None else str(value)


def summarize_layers(rows: list[ParsedRow]) -> list[LayerSummary]:
    buckets: dict[GroupKey, list[ParsedRow]] = defaultdict(list)
    for row in rows:
        buckets[row.key].append(row)
    return [ _aggregate(bucket_rows) for bucket_rows in buckets.values() ]


def make_table_rows(layer_rows: list[LayerSummary]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in sorted(
        layer_rows,
        key=lambda item: (
            item.key.layer_index if item.key.layer_index is not None else 1 << 30,
            item.key.layer_start if item.key.layer_start is not None else -1,
            item.key.backend,
            item.key.run_mode,
            item.key.layer_end_exclusive if item.key.layer_end_exclusive is not None else -1,
            item.key.device_index if item.key.device_index is not None else 1 << 30,
        ),
    ):
        location = row.output_max_abs_diff_location
        records.append(
            {
                "layer": row.key.layer_index,
                "device_backend": device_backend_label(row.key.device_index, row.key.backend),
                "run_mode": row.key.run_mode,
                "range": range_label(row.key.layer_start, row.key.layer_end_exclusive),
                "input_diff_rms": row.input_rms,
                "output_diff_rms": row.output_rms,
                "output_diff_max_abs": row.output_max_abs_diff,
                "max_abs_token": location.token if location else None,
                "max_abs_hidden": location.hidden if location else None,
                "failure_class": row.failure_class or "-",
                "summary": {
                    "input_distribution": {
                        "mean": row.input_mean,
                        "rms": row.input_rms,
                        "l2_norm": row.input_l2_norm,
                        "max_abs_diff": row.samples[0].input.max_abs_diff,
                        "max_abs_diff_location": (
                            row.samples[0].input.max_abs_diff_location.to_dict()
                            if row.samples[0].input.max_abs_diff_location
                            else None
                        ),
                        "per_token_max_mse": row.samples[0].input.per_token_max_mse,
                        "per_token_max_mse_location": (
                            row.samples[0].input.per_token_max_mse_location.to_dict()
                            if row.samples[0].input.per_token_max_mse_location
                            else None
                        ),
                    },
                    "output_distribution": {
                        "mean": row.output_mean,
                        "rms": row.output_rms,
                        "l2_norm": row.output_l2_norm,
                        "max_abs_diff": row.output_max_abs_diff,
                        "max_abs_diff_location": (
                            location.to_dict() if location else None
                        ),
                        "per_token_max_mse": row.output_per_token_max_mse,
                        "per_token_max_mse_location": (
                            row.output_per_token_max_mse_location.to_dict()
                            if row.output_per_token_max_mse_location
                            else None
                        ),
                        "top_hidden_counts": row.output_top_hidden_counts,
                    },
                },
            }
        )
    return records


def build_markdown_table(rows: list[LayerSummary]) -> str:
    lines = [
        "| layer | device/backend | run_mode | range | input_diff_rms | output_diff_rms | output_diff_max_abs | max_abs_token | max_abs_hidden | failure_class |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        location = row.output_max_abs_diff_location
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.key.layer_index) if row.key.layer_index is not None else "-",
                    device_backend_label(row.key.device_index, row.key.backend),
                    row.key.run_mode,
                    range_label(row.key.layer_start, row.key.layer_end_exclusive),
                    fmt_num(row.input_rms),
                    fmt_num(row.output_rms),
                    fmt_num(row.output_max_abs_diff),
                    fmt_loc(location.token if location else None),
                    fmt_loc(location.hidden if location else None),
                    row.failure_class or "-",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _score_badness(rows: list[LayerSummary]) -> float:
    """
    Badness is a practical heuristic:
    prefer larger output diff RMS, and also prefer sudden increases between adjacent layers.
    """
    latest_rms = rows[-1].output_rms
    if latest_rms is None:
        return 0.0
    score = latest_rms
    if len(rows) >= 2:
        prev_rms = rows[-2].output_rms
        if prev_rms is not None and prev_rms > 0:
            score *= (latest_rms / max(prev_rms, 1e-12))
    if rows[-1].output_per_token_max_mse is not None:
        score = max(score, rows[-1].output_per_token_max_mse)
    return score


def first_distribution_bad_layer(layer_rows: list[LayerSummary]) -> dict[str, Any] | None:
    if not layer_rows:
        return None

    grouped: dict[tuple[str, int | None, str, str | None], list[LayerSummary]] = defaultdict(
        list
    )
    for row in layer_rows:
        grouped[
            (
                row.key.run_mode,
                row.key.device_index,
                row.key.backend,
                range_label(row.key.layer_start, row.key.layer_end_exclusive),
            )
        ].append(row)

    best: dict[str, Any] | None = None
    best_score = -1.0

    for key, rows in grouped.items():
        run_mode, device_index, backend, layer_range = key
        ordered = sorted(
            rows,
            key=lambda item: item.key.layer_index if item.key.layer_index is not None else -1,
        )
        for idx, row in enumerate(ordered):
            score = _score_badness(ordered[: idx + 1])
            if score > best_score:
                best_score = score
                bad_layer = row.key.layer_index
                if bad_layer is None:
                    continue
                best = {
                    "layer_index": bad_layer,
                    "device_index": row.key.device_index,
                    "backend": row.key.backend,
                    "run_mode": run_mode,
                    "range": layer_range,
                    "reason": "largest_output_diff_rms_or_sudden_jump",
                    "output_diff_rms": row.output_rms,
                    "output_diff_max_abs": row.output_max_abs_diff,
                    "output_per_token_max_mse": row.output_per_token_max_mse,
                    "score": score,
                    "samples": row.count,
                }

    return best


def build_summary(
    row_groups: list[LayerSummary], sources: list[Path], output_path: Path | None
) -> dict[str, Any]:
    rows = sorted(
        row_groups,
        key=lambda item: (
            item.key.layer_index if item.key.layer_index is not None else 1 << 30,
            item.key.layer_start if item.key.layer_start is not None else -1,
            item.key.backend,
            item.key.run_mode,
            item.key.device_index if item.key.device_index is not None else -1,
        ),
    )
    table_rows = make_table_rows(rows)
    table_markdown = build_markdown_table(rows)
    hot_hidden_counts: Counter[str] = Counter()
    for row in rows:
        hot_hidden_counts.update(row.output_top_hidden_counts)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources": [str(path) for path in sources],
        "summary": {
            "first_distribution_bad_layer": first_distribution_bad_layer(rows),
            "layer_count": len(rows),
            "hot_hidden_counts": dict(hot_hidden_counts.most_common(24)),
            "rows": [
                {
                    "layer": row.key.layer_index,
                    "device_index": row.key.device_index,
                    "backend": row.key.backend,
                    "run_mode": row.key.run_mode,
                    "layer_start": row.key.layer_start,
                    "layer_end_exclusive": row.key.layer_end_exclusive,
                    "count": row.count,
                    "input": {
                        "mean": row.input_mean,
                        "rms": row.input_rms,
                        "l2_norm": row.input_l2_norm,
                        "per_token_max_mse": row.samples[0].input.per_token_max_mse
                        if row.samples
                        else None,
                        "per_token_max_mse_location": (
                            row.samples[0].input.per_token_max_mse_location.to_dict()
                            if row.samples and row.samples[0].input.per_token_max_mse_location
                            else None
                        ),
                    },
                    "output": {
                        "mean": row.output_mean,
                        "rms": row.output_rms,
                        "l2_norm": row.output_l2_norm,
                        "max_abs_diff": row.output_max_abs_diff,
                        "max_abs_diff_location": (
                            row.output_max_abs_diff_location.to_dict()
                            if row.output_max_abs_diff_location
                            else None
                        ),
                        "per_token_max_mse": row.output_per_token_max_mse,
                        "per_token_max_mse_location": (
                            row.output_per_token_max_mse_location.to_dict()
                            if row.output_per_token_max_mse_location
                            else None
                        ),
                        "top_hidden_counts": row.output_top_hidden_counts,
                    },
                    "failure_class": row.failure_class or "-",
                }
                for row in rows
            ],
        },
        "table_rows": table_rows,
        "markdown": table_markdown,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    for path in args.jsonl_paths:
        safe_path(path)

    all_rows: list[ParsedRow] = []
    for path in args.jsonl_paths:
        all_rows.extend(read_jsonl(path))
    layer_rows = summarize_layers(all_rows)
    summary = build_summary(layer_rows, args.jsonl_paths, args.summary_json)

    markdown = summary["markdown"]
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
