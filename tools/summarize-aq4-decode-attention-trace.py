#!/usr/bin/env python3
"""Summarize AQ4_0 decode GPU time from a rocprof trace.

Each ``KERNEL_DISPATCH`` is attributed to the enclosing outer decode marker by
the start time of its correlated ``hipModuleLaunchKernel`` API call.  This is
deliberate: HIP launches are asynchronous, so assigning a dispatch from its
GPU timestamps alone can leak work across a marker boundary.  The result is
inclusive GPU time for module-launched kernels, not wall-clock throughput.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


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
    """Raised for an untrustworthy or incompatible profiler trace."""


@dataclass(frozen=True)
class StepRange:
    index: int
    cache_start: int
    start_ns: int
    end_ns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-trace", type=Path, required=True)
    parser.add_argument("--hip-api-trace", type=Path, required=True)
    parser.add_argument("--marker-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-cache-start",
        type=int,
        default=None,
        help="require the first marked decode step to have this cache length",
    )
    return parser.parse_args()


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


def read_steps(marker_trace: Path, expected_cache_start: int | None) -> list[StepRange]:
    rows = read_rows(
        marker_trace,
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
    if (
        expected_cache_start is not None
        and steps[0].cache_start != expected_cache_start
    ):
        raise TraceError(
            "first decode-step marker cache length does not match --expected-cache-start"
        )
    return steps


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_launch_times(hip_api_trace: Path) -> dict[int, int]:
    rows = read_rows(
        hip_api_trace,
        {"Function", "Correlation_Id", "Start_Timestamp", "End_Timestamp"},
        "HIP API trace",
    )
    launches: dict[int, int] = {}
    for row in rows:
        if row["Function"] != "hipModuleLaunchKernel":
            continue
        correlation = integer(row, "Correlation_Id", "HIP API launch")
        start_ns = integer(row, "Start_Timestamp", "HIP API launch")
        end_ns = integer(row, "End_Timestamp", "HIP API launch")
        if end_ns < start_ns:
            raise TraceError("HIP API launch has negative duration")
        if correlation in launches:
            raise TraceError("HIP API launch correlation ID is duplicated")
        launches[correlation] = start_ns
    if not launches:
        raise TraceError("HIP API trace contains no hipModuleLaunchKernel calls")
    return launches


def summarize(
    kernel_trace: Path, hip_api_trace: Path, steps: list[StepRange]
) -> dict[str, object]:
    rows = read_rows(
        kernel_trace,
        {"Kind", "Kernel_Name", "Start_Timestamp", "End_Timestamp"},
        "kernel trace",
    )
    by_kernel_ns: dict[str, int] = defaultdict(int)
    by_kernel_count: dict[str, int] = defaultdict(int)
    by_family_ns: dict[str, int] = defaultdict(int)
    by_family_count: dict[str, int] = defaultdict(int)
    matched_dispatches = 0
    ignored_dispatches_outside_markers = 0
    ignored_dispatches_without_module_launch = 0
    marker_starts = [step.start_ns for step in steps]
    launches = read_launch_times(hip_api_trace)
    for row in rows:
        if row["Kind"] != "KERNEL_DISPATCH":
            continue
        correlation = integer(row, "Correlation_Id", "kernel")
        # rocprof also represents DMA operations as KERNEL_DISPATCH records,
        # but their HIP correlation is e.g. hipMemcpyDtoHAsync rather than a
        # compute-kernel launch.  They are intentionally out of this
        # module-launch compute-kernel denominator.
        if correlation not in launches:
            ignored_dispatches_without_module_launch += 1
            continue
        launch_start_ns = launches[correlation]
        start_ns = integer(row, "Start_Timestamp", "kernel")
        end_ns = integer(row, "End_Timestamp", "kernel")
        if end_ns < start_ns:
            raise TraceError("kernel dispatch has negative duration")
        step_index = bisect.bisect_right(marker_starts, launch_start_ns) - 1
        if step_index < 0 or launch_start_ns >= steps[step_index].end_ns:
            ignored_dispatches_outside_markers += 1
            continue
        name = row["Kernel_Name"]
        duration = end_ns - start_ns
        family = classify_kernel(name)
        matched_dispatches += 1
        by_kernel_ns[name] += duration
        by_kernel_count[name] += 1
        by_family_ns[family] += duration
        by_family_count[family] += 1
    total_ns = sum(by_kernel_ns.values())
    if total_ns == 0:
        raise TraceError("no module-launched kernel dispatch belongs to decode markers")
    attention_ns = sum(by_kernel_ns[name] for name in ATTENTION_KERNELS)
    return {
        "schema_version": "ullm.aq4_decode_attention_trace_summary.v1",
        "measurement": {
            "kind": "marker_attributed_module_launch_inclusive_kernel_time",
            "attribution": (
                "hipModuleLaunchKernel correlation start inside an outer "
                "AQ4_0 decode-step marker"
            ),
            "throughput_derived_from_profiler": False,
            "decode_step_count": len(steps),
            "cache_start": steps[0].cache_start,
            "cache_end_exclusive": steps[-1].cache_start + 1,
            "matched_kernel_dispatches": matched_dispatches,
            "ignored_kernel_dispatches_outside_decode_markers": (
                ignored_dispatches_outside_markers
            ),
            "ignored_kernel_dispatches_without_hip_module_launch": (
                ignored_dispatches_without_module_launch
            ),
        },
        "kernel": {
            "inclusive_ns": total_ns,
            "attention_inclusive_ns": attention_ns,
            "attention_fraction_of_inclusive_kernel_time": attention_ns / total_ns,
            "families": {
                name: {
                    "inclusive_ns": by_family_ns[name],
                    "kernel_count": by_family_count[name],
                    "fraction_of_inclusive_kernel_time": by_family_ns[name] / total_ns,
                }
                for name in sorted(by_family_ns)
            },
            "attention_kernels": {
                name: {
                    "inclusive_ns": by_kernel_ns[name],
                    "kernel_count": by_kernel_count[name],
                    "fraction_of_inclusive_kernel_time": by_kernel_ns[name] / total_ns,
                }
                for name in sorted(ATTENTION_KERNELS)
            },
            "top_kernels": [
                {
                    "name": name,
                    "inclusive_ns": duration,
                    "kernel_count": by_kernel_count[name],
                    "fraction_of_inclusive_kernel_time": duration / total_ns,
                }
                for name, duration in sorted(
                    by_kernel_ns.items(), key=lambda item: (-item[1], item[0])
                )[:20]
            ],
        },
        "trace": {
            "kernel_trace": str(kernel_trace.resolve()),
            "kernel_trace_sha256": sha256(kernel_trace),
            "hip_api_trace": str(hip_api_trace.resolve()),
            "hip_api_trace_sha256": sha256(hip_api_trace),
        },
    }


def atomic_write(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    try:
        steps = read_steps(args.marker_trace, args.expected_cache_start)
        document = summarize(args.kernel_trace, args.hip_api_trace, steps)
        document["trace"]["marker_trace"] = str(args.marker_trace.resolve())
        document["trace"]["marker_trace_sha256"] = sha256(args.marker_trace)
        atomic_write(args.output, document)
    except TraceError as error:
        print(f"summarize-aq4-decode-attention-trace: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
