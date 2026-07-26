#!/usr/bin/env python3
"""Produce marker-attributed AQ4_0 decode wall-time accounting from rocprofv3 CSVs.

The profiler's marker duration is deliberately kept separate from the wall-clock
measurement emitted by ``ullm-aq4-decode-step-profile``.  GPU dispatches are
attributed by the start of their correlated HIP API call, rather than by their
GPU timestamp, because launches are asynchronous.

The result distinguishes:

* module-launched compute kernels (the AQ4_0 kernel-time denominator),
* other GPU dispatches such as DMA copy kernels,
* the union and gaps of GPU intervals, and
* host HIP API timing.  Host API durations overlap GPU execution and are
  evidence, not additive components of wall time.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


STEP_PREFIX = "ullm.aq4.decode.step.v1/"
STEP_PATTERN = re.compile(
    r"^ullm\.aq4\.decode\.step\.v1/step_index=(?P<index>[0-9]+)/cache_start=(?P<cache>[0-9]+)$"
)
ATTENTION_KERNELS = frozenset(
    {
        "ullm_paged_decode_attn_f32_kernel",
        "ullm_paged_decode_attn_split_partial_f32_kernel",
        "ullm_paged_decode_attn_split_merge_f32_kernel",
    }
)


class TraceError(RuntimeError):
    """Raised when an input trace cannot support trustworthy accounting."""


@dataclass(frozen=True)
class StepRange:
    index: int
    cache_start: int
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class HipApi:
    function: str
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class Dispatch:
    name: str
    api_function: str
    correlation_id: int
    api_start_ns: int
    api_end_ns: int
    queue_id: str
    stream_id: str
    workgroup_size: tuple[int, int, int]
    grid_size: tuple[int, int, int]
    start_ns: int
    end_ns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-trace", type=Path, required=True)
    parser.add_argument("--hip-api-trace", type=Path, required=True)
    parser.add_argument("--marker-trace", type=Path, required=True)
    parser.add_argument(
        "--profile-stdout",
        type=Path,
        help="JSON-lines stdout from ullm-aq4-decode-step-profile; supplies matching wall time",
    )
    parser.add_argument(
        "--wall-ms",
        type=float,
        help="explicit per-token wall time when no matching profile stdout is available",
    )
    parser.add_argument("--expected-cache-start", type=int)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path, required: set[str], label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise TraceError(f"{label} is not a regular file: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise TraceError(f"{label} has an incompatible CSV header")
            return list(reader)
    except UnicodeDecodeError as error:
        raise TraceError(f"{label} is not UTF-8 CSV") from error
    except csv.Error as error:
        raise TraceError(f"{label} is malformed CSV") from error


def integer(row: dict[str, str], key: str, label: str) -> int:
    try:
        value = int(row[key], 10)
    except (KeyError, ValueError) as error:
        raise TraceError(f"{label}.{key} is not an integer") from error
    if value < 0:
        raise TraceError(f"{label}.{key} is negative")
    return value


def positive_interval(start_ns: int, end_ns: int, label: str) -> None:
    if end_ns < start_ns:
        raise TraceError(f"{label} has a negative duration")


def read_steps(path: Path, expected_cache_start: int | None) -> list[StepRange]:
    rows = read_rows(
        path,
        {"Domain", "Function", "Start_Timestamp", "End_Timestamp"},
        "marker trace",
    )
    steps: list[StepRange] = []
    for row in rows:
        name = row["Function"]
        if not name.startswith(STEP_PREFIX):
            continue
        match = STEP_PATTERN.fullmatch(name)
        if match is None:
            raise TraceError(f"unexpected AQ4_0 decode marker name: {name!r}")
        start_ns = integer(row, "Start_Timestamp", "marker")
        end_ns = integer(row, "End_Timestamp", "marker")
        if end_ns <= start_ns:
            raise TraceError("decode marker has non-positive duration")
        steps.append(
            StepRange(
                index=int(match.group("index"), 10),
                cache_start=int(match.group("cache"), 10),
                start_ns=start_ns,
                end_ns=end_ns,
            )
        )
    if not steps:
        raise TraceError("marker trace contains no AQ4_0 decode-step ranges")
    steps.sort(key=lambda step: step.index)
    for expected_index, step in enumerate(steps):
        if step.index != expected_index:
            raise TraceError("decode-step marker indices are not contiguous from zero")
        if step.cache_start != steps[0].cache_start + expected_index:
            raise TraceError("decode-step marker cache lengths are not contiguous")
        if expected_index and step.start_ns < steps[expected_index - 1].end_ns:
            raise TraceError("decode-step marker ranges overlap")
    if expected_cache_start is not None and steps[0].cache_start != expected_cache_start:
        raise TraceError(
            "first decode-step marker cache length does not match --expected-cache-start"
        )
    return steps


def read_apis(path: Path) -> dict[int, HipApi]:
    rows = read_rows(
        path,
        {"Function", "Correlation_Id", "Start_Timestamp", "End_Timestamp"},
        "HIP API trace",
    )
    result: dict[int, HipApi] = {}
    for row in rows:
        correlation_id = integer(row, "Correlation_Id", "HIP API")
        start_ns = integer(row, "Start_Timestamp", "HIP API")
        end_ns = integer(row, "End_Timestamp", "HIP API")
        positive_interval(start_ns, end_ns, "HIP API")
        if correlation_id in result:
            raise TraceError("HIP API correlation ID is duplicated")
        result[correlation_id] = HipApi(row["Function"], start_ns, end_ns)
    if not result:
        raise TraceError("HIP API trace is empty")
    return result


def classify_kernel(name: str) -> str:
    if name in ATTENTION_KERNELS:
        return "attention"
    if name.startswith("ullm_linear_attn_"):
        return "linear_attention"
    if name.startswith("ullm_aq4_"):
        return "aq4_projection"
    if "rmsnorm" in name or "norm" in name:
        return "normalization"
    if name.startswith("ullm_qwen35_") or "paged_kv" in name:
        return "attention_support"
    if name.startswith("ullm_"):
        return "other_ullm"
    return "runtime_support"


def step_index_for_timestamp(starts: Sequence[int], steps: Sequence[StepRange], timestamp: int) -> int | None:
    index = bisect.bisect_right(starts, timestamp) - 1
    if index < 0 or timestamp >= steps[index].end_ns:
        return None
    return index


def read_dispatches(
    path: Path,
    apis: dict[int, HipApi],
    steps: Sequence[StepRange],
) -> tuple[list[list[Dispatch]], dict[str, int]]:
    rows = read_rows(
        path,
        {
            "Kind",
            "Kernel_Name",
            "Correlation_Id",
            "Start_Timestamp",
            "End_Timestamp",
            "Workgroup_Size_X",
            "Workgroup_Size_Y",
            "Workgroup_Size_Z",
            "Grid_Size_X",
            "Grid_Size_Y",
            "Grid_Size_Z",
        },
        "kernel trace",
    )
    starts = [step.start_ns for step in steps]
    per_step: list[list[Dispatch]] = [[] for _ in steps]
    ignored = Counter()
    for row in rows:
        if row["Kind"] != "KERNEL_DISPATCH":
            continue
        correlation_id = integer(row, "Correlation_Id", "kernel dispatch")
        api = apis.get(correlation_id)
        if api is None:
            ignored["without_correlated_hip_api"] += 1
            continue
        step_index = step_index_for_timestamp(starts, steps, api.start_ns)
        if step_index is None:
            ignored["outside_decode_marker"] += 1
            continue
        start_ns = integer(row, "Start_Timestamp", "kernel dispatch")
        end_ns = integer(row, "End_Timestamp", "kernel dispatch")
        positive_interval(start_ns, end_ns, "kernel dispatch")
        per_step[step_index].append(
            Dispatch(
                name=row["Kernel_Name"],
                api_function=api.function,
                correlation_id=correlation_id,
                api_start_ns=api.start_ns,
                api_end_ns=api.end_ns,
                queue_id=row.get("Queue_Id", ""),
                stream_id=row.get("Stream_Id", ""),
                workgroup_size=(
                    integer(row, "Workgroup_Size_X", "kernel dispatch"),
                    integer(row, "Workgroup_Size_Y", "kernel dispatch"),
                    integer(row, "Workgroup_Size_Z", "kernel dispatch"),
                ),
                grid_size=(
                    integer(row, "Grid_Size_X", "kernel dispatch"),
                    integer(row, "Grid_Size_Y", "kernel dispatch"),
                    integer(row, "Grid_Size_Z", "kernel dispatch"),
                ),
                start_ns=start_ns,
                end_ns=end_ns,
            )
        )
    return per_step, dict(sorted(ignored.items()))


def apis_in_steps(
    apis: dict[int, HipApi], steps: Sequence[StepRange]
) -> list[list[HipApi]]:
    starts = [step.start_ns for step in steps]
    per_step: list[list[HipApi]] = [[] for _ in steps]
    for api in apis.values():
        step_index = step_index_for_timestamp(starts, steps, api.start_ns)
        if step_index is not None:
            per_step[step_index].append(api)
    return per_step


def integer_stats(values: Iterable[int]) -> dict[str, int | float]:
    collected = list(values)
    if not collected:
        return {"count": 0, "total_ns": 0, "mean_ns": 0.0, "min_ns": 0, "max_ns": 0}
    return {
        "count": len(collected),
        "total_ns": sum(collected),
        "mean_ns": statistics.fmean(collected),
        "min_ns": min(collected),
        "max_ns": max(collected),
    }


def count_stats(values: Iterable[int]) -> dict[str, int | float]:
    collected = list(values)
    if not collected:
        return {"sample_count": 0, "total_count": 0, "mean_count": 0.0, "min_count": 0, "max_count": 0}
    return {
        "sample_count": len(collected),
        "total_count": sum(collected),
        "mean_count": statistics.fmean(collected),
        "min_count": min(collected),
        "max_count": max(collected),
    }


def duration(dispatch: Dispatch) -> int:
    return dispatch.end_ns - dispatch.start_ns


def summarize_timeline(
    steps: Sequence[StepRange],
    events_by_step: Sequence[Sequence[Dispatch]],
    predicate: Callable[[Dispatch], bool],
) -> dict[str, object]:
    per_step: list[dict[str, int]] = []
    all_event_counts: list[int] = []
    durations: list[int] = []
    spans: list[int] = []
    unions: list[int] = []
    gaps: list[int] = []
    overlaps: list[int] = []
    leading: list[int] = []
    trailing: list[int] = []
    marker_durations: list[int] = []
    for step, events in zip(steps, events_by_step, strict=True):
        selected = sorted((event for event in events if predicate(event)), key=lambda event: (event.start_ns, event.end_ns))
        marker_duration = step.end_ns - step.start_ns
        marker_durations.append(marker_duration)
        if not selected:
            per_step.append(
                {
                    "index": step.index,
                    "cache_start": step.cache_start,
                    "event_count": 0,
                    "inclusive_duration_ns": 0,
                    "activity_span_ns": 0,
                    "interval_union_ns": 0,
                    "inter_event_gap_ns": 0,
                    "overlap_ns": 0,
                    "marker_leading_no_event_ns": marker_duration,
                    "marker_trailing_no_event_ns": marker_duration,
                    "marker_duration_ns": marker_duration,
                }
            )
            all_event_counts.append(0)
            continue
        inclusive = sum(duration(event) for event in selected)
        first_start = selected[0].start_ns
        cursor = selected[0].end_ns
        union = cursor - first_start
        gap = 0
        overlap = 0
        for event in selected[1:]:
            if event.start_ns > cursor:
                gap += event.start_ns - cursor
                union += duration(event)
                cursor = event.end_ns
            elif event.end_ns > cursor:
                overlap += cursor - event.start_ns
                union += event.end_ns - cursor
                cursor = event.end_ns
            else:
                overlap += duration(event)
        activity_span = cursor - first_start
        lead = first_start - step.start_ns
        trail = step.end_ns - cursor
        per_step.append(
            {
                "index": step.index,
                "cache_start": step.cache_start,
                "event_count": len(selected),
                "inclusive_duration_ns": inclusive,
                "activity_span_ns": activity_span,
                "interval_union_ns": union,
                "inter_event_gap_ns": gap,
                "overlap_ns": overlap,
                "marker_leading_no_event_ns": lead,
                "marker_trailing_no_event_ns": trail,
                "marker_duration_ns": marker_duration,
            }
        )
        all_event_counts.append(len(selected))
        durations.append(inclusive)
        spans.append(activity_span)
        unions.append(union)
        gaps.append(gap)
        overlaps.append(overlap)
        leading.append(lead)
        trailing.append(trail)
    return {
        "per_step": per_step,
        "event_count_per_step": count_stats(all_event_counts),
        "inclusive_duration": integer_stats(durations),
        "activity_span": integer_stats(spans),
        "interval_union": integer_stats(unions),
        "inter_event_gap": integer_stats(gaps),
        "overlap": integer_stats(overlaps),
        "marker_leading_no_event": integer_stats(leading),
        "marker_trailing_no_event": integer_stats(trailing),
        "marker_duration": integer_stats(marker_durations),
    }


def summarize_gap_next_dispatch_api_relation(
    events_by_step: Sequence[Sequence[Dispatch]],
) -> dict[str, object]:
    """Classify each directly observed GPU gap by the next API's timing.

    API and GPU timestamps are in the same rocprof trace timebase.  A
    relationship is intentionally not called a cause: the profiler callback
    itself can change when an API returns and a queued dispatch may await other
    runtime/device work.
    """

    gap_durations: dict[str, list[int]] = defaultdict(list)
    api_functions: dict[str, Counter[str]] = defaultdict(Counter)
    next_dispatch_groups: dict[str, list[int]] = defaultdict(list)
    per_step: list[dict[str, object]] = []
    for events in events_by_step:
        selected = sorted(events, key=lambda event: (event.start_ns, event.end_ns))
        per_step_counts: Counter[str] = Counter()
        per_step_durations: Counter[str] = Counter()
        if selected:
            cursor = selected[0].end_ns
            for event in selected[1:]:
                if event.start_ns > cursor:
                    gap_ns = event.start_ns - cursor
                    if event.api_start_ns >= cursor:
                        relation = "next_api_started_after_prior_gpu_end"
                    elif event.api_end_ns > cursor:
                        relation = "next_api_in_progress_at_prior_gpu_end"
                    else:
                        relation = "next_api_completed_before_prior_gpu_end"
                    gap_durations[relation].append(gap_ns)
                    api_functions[relation][event.api_function] += 1
                    if event.api_function == "hipModuleLaunchKernel":
                        group = classify_kernel(event.name)
                    else:
                        group = f"nonmodule:{event.api_function}"
                    next_dispatch_groups[group].append(gap_ns)
                    per_step_counts[relation] += 1
                    per_step_durations[relation] += gap_ns
                    cursor = event.end_ns
                elif event.end_ns > cursor:
                    cursor = event.end_ns
        per_step.append(
            {
                "gap_count_by_relation": dict(sorted(per_step_counts.items())),
                "gap_ns_by_relation": dict(sorted(per_step_durations.items())),
            }
        )
    categories: dict[str, object] = {}
    for relation in sorted(gap_durations):
        stats = integer_stats(gap_durations[relation])
        categories[relation] = {
            "gap_count": stats.pop("count"),
            "gap_duration": stats,
            "next_api_functions": dict(sorted(api_functions[relation].items())),
        }
    groups: dict[str, object] = {}
    for group in sorted(next_dispatch_groups):
        stats = integer_stats(next_dispatch_groups[group])
        groups[group] = {
            "gap_count": stats.pop("count"),
            "gap_duration": stats,
        }
    return {
        "interpretation": (
            "Each all-dispatch GPU gap is partitioned by the temporal relation of its next "
            "dispatch's correlated HIP API to the preceding GPU interval. This is evidence "
            "about queue readiness, not a causal launch-overhead decomposition; rocprof API "
            "callbacks can perturb the relation."
        ),
        "categories": categories,
        "next_dispatch_groups": groups,
        "per_step": per_step,
    }


def summarize_kernel_geometry(events_by_step: Sequence[Sequence[Dispatch]]) -> dict[str, object]:
    """Report observed launch geometry variants without inferring graph support."""

    variants: dict[str, Counter[tuple[tuple[int, int, int], tuple[int, int, int]]]] = defaultdict(
        Counter
    )
    for events in events_by_step:
        for event in events:
            variants[event.name][(event.workgroup_size, event.grid_size)] += 1
    kernels: list[dict[str, object]] = []
    for name in sorted(variants):
        shape_counts = variants[name]
        shapes = [
            {
                "workgroup_size": list(workgroup),
                "grid_size": list(grid),
                "dispatch_count": count,
            }
            for (workgroup, grid), count in sorted(shape_counts.items())
        ]
        dispatch_count = sum(shape_counts.values())
        kernels.append(
            {
                "name": name,
                "dispatch_count": dispatch_count,
                "shape_variant_count": len(shapes),
                "single_observed_launch_geometry": len(shapes) == 1,
                "shapes": shapes,
            }
        )
    return {
        "interpretation": (
            "A single observed grid/workgroup shape across the marked range is structural "
            "evidence for graph-capture investigation only. Pointer and scalar updates, module "
            "capture support, and host synchronization remain separate requirements."
        ),
        "all_named_dispatches_single_observed_geometry": all(
            bool(entry["single_observed_launch_geometry"]) for entry in kernels
        ),
        "kernels": kernels,
    }


def summarize_families(events_by_step: Sequence[Sequence[Dispatch]]) -> dict[str, object]:
    by_family_time: dict[str, int] = defaultdict(int)
    by_family_count: dict[str, int] = defaultdict(int)
    by_kernel_time: dict[str, int] = defaultdict(int)
    by_kernel_count: dict[str, int] = defaultdict(int)
    for events in events_by_step:
        for event in events:
            if event.api_function != "hipModuleLaunchKernel":
                continue
            event_duration = duration(event)
            family = classify_kernel(event.name)
            by_family_time[family] += event_duration
            by_family_count[family] += 1
            by_kernel_time[event.name] += event_duration
            by_kernel_count[event.name] += 1
    total = sum(by_kernel_time.values())
    if not total:
        raise TraceError("no module-launched dispatch belongs to a decode marker")
    families = {
        name: {
            "inclusive_ns": by_family_time[name],
            "kernel_count": by_family_count[name],
            "fraction_of_module_kernel_time": by_family_time[name] / total,
        }
        for name in sorted(by_family_time)
    }
    kernels = [
        {
            "name": name,
            "inclusive_ns": by_kernel_time[name],
            "kernel_count": by_kernel_count[name],
            "fraction_of_module_kernel_time": by_kernel_time[name] / total,
        }
        for name in by_kernel_time
    ]
    kernels.sort(key=lambda entry: (-int(entry["inclusive_ns"]), str(entry["name"])))
    return {"inclusive_ns": total, "families": families, "kernels": kernels}


def summarize_host_apis(apis_by_step: Sequence[Sequence[HipApi]]) -> dict[str, object]:
    by_name: dict[str, list[int]] = defaultdict(list)
    counts_per_step: dict[str, list[int]] = defaultdict(list)
    names = sorted({api.function for apis in apis_by_step for api in apis})
    for name in names:
        for apis in apis_by_step:
            matching = [api for api in apis if api.function == name]
            counts_per_step[name].append(len(matching))
            by_name[name].extend(api.end_ns - api.start_ns for api in matching)
    functions = {}
    for name in names:
        duration_stats = integer_stats(by_name[name])
        functions[name] = {
            "call_count": duration_stats.pop("count"),
            "api_duration": duration_stats,
            "calls_per_step": count_stats(counts_per_step[name]),
        }
    return {
        "non_additive_note": (
            "HIP API durations can overlap asynchronous GPU execution; they must not be "
            "summed with GPU intervals to form wall time."
        ),
        "functions": functions,
    }


def parse_profile_stdout(path: Path, steps: Sequence[StepRange]) -> dict[str, object]:
    if not path.is_file():
        raise TraceError(f"profile stdout is not a regular file: {path}")
    samples: list[dict[str, object]] = []
    summary: dict[str, object] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "measured_decode_step":
            samples.append(event)
        elif event.get("event") == "summary":
            summary = event
    if len(samples) != len(steps):
        raise TraceError(
            f"profile stdout has {len(samples)} measured steps, marker trace has {len(steps)}"
        )
    samples.sort(key=lambda sample: int(sample["step_index"]))
    durations: list[int] = []
    for expected, (sample, marker_step) in enumerate(zip(samples, steps, strict=True)):
        if int(sample["step_index"]) != expected:
            raise TraceError("profile stdout step indices are not contiguous from zero")
        if int(sample["cache_len_start"]) != marker_step.cache_start:
            raise TraceError("profile stdout cache start differs from marker trace")
        elapsed_seconds = float(sample["elapsed_seconds"])
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise TraceError("profile stdout has invalid elapsed_seconds")
        durations.append(round(elapsed_seconds * 1_000_000_000))
    result: dict[str, object] = {
        "source": str(path),
        "source_sha256": sha256(path),
        "samples": integer_stats(durations),
        "per_step_ns": durations,
    }
    if summary is not None:
        result["reported_summary"] = summary
    return result


def wall_from_args(args: argparse.Namespace, steps: Sequence[StepRange]) -> dict[str, object] | None:
    if args.profile_stdout is not None:
        if args.wall_ms is not None:
            raise TraceError("use either --profile-stdout or --wall-ms, not both")
        return parse_profile_stdout(args.profile_stdout, steps)
    if args.wall_ms is None:
        return None
    if not math.isfinite(args.wall_ms) or args.wall_ms <= 0:
        raise TraceError("--wall-ms must be finite and positive")
    per_step_ns = round(args.wall_ms * 1_000_000)
    return {
        "source": "--wall-ms",
        "samples": integer_stats([per_step_ns] * len(steps)),
        "per_step_ns": [per_step_ns] * len(steps),
    }


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        json.dump(value, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    steps = read_steps(args.marker_trace, args.expected_cache_start)
    if args.expected_steps is not None and len(steps) != args.expected_steps:
        raise TraceError(
            f"marker trace has {len(steps)} steps, expected {args.expected_steps}"
        )
    apis = read_apis(args.hip_api_trace)
    dispatches_by_step, ignored_dispatches = read_dispatches(args.kernel_trace, apis, steps)
    host_apis_by_step = apis_in_steps(apis, steps)
    module_timeline = summarize_timeline(
        steps, dispatches_by_step, lambda event: event.api_function == "hipModuleLaunchKernel"
    )
    all_dispatch_timeline = summarize_timeline(steps, dispatches_by_step, lambda event: True)
    all_dispatch_timeline["gap_next_dispatch_api_relation"] = (
        summarize_gap_next_dispatch_api_relation(dispatches_by_step)
    )
    dma_timeline = summarize_timeline(
        steps, dispatches_by_step, lambda event: event.api_function.startswith("hipMemcpy")
    )
    kernel = summarize_families(dispatches_by_step)
    kernel_geometry = summarize_kernel_geometry(dispatches_by_step)
    host_apis = summarize_host_apis(host_apis_by_step)
    wall = wall_from_args(args, steps)
    module_kernel_ns = int(kernel["inclusive_ns"])
    all_gpu_ns = int(all_dispatch_timeline["inclusive_duration"]["total_ns"])
    nonmodule_gpu_ns = all_gpu_ns - module_kernel_ns
    time_accounting: dict[str, object] = {
        "module_kernel": {
            "inclusive_ns": module_kernel_ns,
            "per_token_ns": module_kernel_ns / len(steps),
        },
        "all_gpu_dispatch": {
            "inclusive_ns": all_gpu_ns,
            "per_token_ns": all_gpu_ns / len(steps),
        },
        "nonmodule_gpu_dispatch": {
            "inclusive_ns": nonmodule_gpu_ns,
            "per_token_ns": nonmodule_gpu_ns / len(steps),
        },
    }
    if wall is not None:
        wall_total_ns = int(wall["samples"]["total_ns"])
        wall_per_token_ns = float(wall["samples"]["mean_ns"])
        time_accounting["wall"] = {
            "total_ns": wall_total_ns,
            "per_token_ns": wall_per_token_ns,
        }
        time_accounting["module_kernel"]["fraction_of_wall"] = module_kernel_ns / wall_total_ns
        time_accounting["all_gpu_dispatch"]["fraction_of_wall"] = all_gpu_ns / wall_total_ns
        time_accounting["kernel_outside_wall"] = {
            "inclusive_ns": wall_total_ns - module_kernel_ns,
            "per_token_ns": (wall_total_ns - module_kernel_ns) / len(steps),
            "fraction_of_wall": (wall_total_ns - module_kernel_ns) / wall_total_ns,
            "definition": "matching per-step wall time minus module-launched GPU kernel inclusive time",
        }
    queue_ids = sorted(
        {event.queue_id for events in dispatches_by_step for event in events if event.queue_id}
    )
    stream_ids = sorted(
        {event.stream_id for events in dispatches_by_step for event in events if event.stream_id}
    )
    return {
        "schema_version": "ullm.aq4_decode_walltime_accounting.v1",
        "method": {
            "dispatch_attribution": (
                "The start timestamp of each correlated HIP API call must lie inside an outer "
                "AQ4_0 decode-step marker. GPU timestamp-only attribution is not used."
            ),
            "wall_clock": (
                "Wall time comes only from the profiler driver's Instant measurements, never "
                "from a rocprof marker/range duration."
            ),
            "gpu_idle": (
                "inter_event_gap is the direct sum of gaps between unioned KERNEL_DISPATCH GPU "
                "intervals inside each attributed step; it excludes marker lead/trail."
            ),
        },
        "inputs": {
            "kernel_trace": {"path": str(args.kernel_trace), "sha256": sha256(args.kernel_trace)},
            "hip_api_trace": {"path": str(args.hip_api_trace), "sha256": sha256(args.hip_api_trace)},
            "marker_trace": {"path": str(args.marker_trace), "sha256": sha256(args.marker_trace)},
        },
        "measurement": {
            "decode_step_count": len(steps),
            "cache_start": steps[0].cache_start,
            "cache_end_exclusive": steps[-1].cache_start + 1,
            "steps": [
                {
                    "index": step.index,
                    "cache_start": step.cache_start,
                    "marker_start_ns": step.start_ns,
                    "marker_end_ns": step.end_ns,
                }
                for step in steps
            ],
            "queue_ids": queue_ids,
            "stream_ids": stream_ids,
            "ignored_kernel_dispatches": ignored_dispatches,
        },
        "wall_clock": wall,
        "time_accounting": time_accounting,
        "module_kernel": kernel,
        "kernel_geometry": kernel_geometry,
        "gpu_timeline": {
            "all_dispatches": all_dispatch_timeline,
            "module_launch_dispatches": module_timeline,
            "dma_copy_dispatches": dma_timeline,
        },
        "host_hip_api": host_apis,
    }


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
        write_json(args.output, result)
    except TraceError as error:
        print(f"analyze-aq4-decode-walltime-accounting: {error}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
