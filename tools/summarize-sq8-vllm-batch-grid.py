#!/usr/bin/env python3
"""Build a compact Markdown batch-grid summary table from benchmark JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_str(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def format_gib(bytes_value: Any) -> str | None:
    value = as_int(bytes_value)
    if value is None:
        return None
    return f"{value / 1024 ** 3:.2f}"


def format_float(value: Any, digits: int = 2) -> str:
    parsed = as_float(value)
    if parsed is None:
        return "-"
    return f"{parsed:.{digits}f}"


def workload_prefix(workload: dict[str, Any]) -> str:
    prompt = as_int(workload.get("prompt_tokens"))
    generated = as_int(workload.get("generated_tokens"))
    if prompt is None or generated is None:
        return ""
    return f"pp{prompt}-tg{generated}"


def harness_class(row: dict[str, Any]) -> str:
    harness = as_dict(row.get("harness"))
    explicit = harness.get("class")
    if explicit:
        return str(explicit)
    engine = as_dict(row.get("engine")).get("name")
    case_id = as_str(row.get("case_id"))
    if engine == "vLLM":
        return "serving_throughput_benchmark"
    if "mixed-real-batch-no-final" in case_id:
        return "cli_model_loop_diagnostic"
    return "-"


def should_keep(
    row: dict[str, Any],
    workload_prefix_filter: str,
    case_substring: str,
    requests_filter: set[int],
) -> bool:
    case_id = as_str(row.get("case_id"))
    workload = as_dict(row.get("workload"))
    case_match = True
    workload_match = True
    requests_match = True
    requests = (
        as_int(workload.get("concurrent_requests"))
        or as_int(workload.get("batch_size"))
        or 0
    )

    if case_substring:
        case_match = case_substring in case_id

    if workload_prefix_filter:
        prefix = workload_prefix_filter
        workload_label = workload_prefix(workload)
        workload_match = (
            prefix in case_id or (workload_label and workload_label.startswith(prefix))
        )

    if requests_filter:
        requests_match = requests in requests_filter

    status = row.get("status")
    status_match = status == "ok" if status is not None else True

    return case_match and workload_match and requests_match and status_match


def iter_rows(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from None


def parse_requests_filter(value: str) -> set[int]:
    if not value:
        return set()
    requests: set[int] = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            raise ValueError("--requests contains an empty item")
        parsed = as_int(item)
        if parsed is None or parsed < 1:
            raise ValueError(f"--requests item must be a positive integer: {item}")
        requests.add(parsed)
    return requests


def iter_markdown_rows(
    paths: Iterable[Path],
    workload_prefix_filter: str,
    case_substring: str,
    requests_filter: set[int],
) -> Iterator[list[str]]:
    for row in iter_rows(paths):
        if not should_keep(row, workload_prefix_filter, case_substring, requests_filter):
            continue
        workload = as_dict(row.get("workload"))
        metrics = as_dict(row.get("metrics"))
        memory = as_dict(row.get("memory"))
        engine = as_dict(row.get("engine"))
        requested = (
            as_int(workload.get("concurrent_requests"))
            or as_int(workload.get("batch_size"))
            or 0
        )
        yield [
            as_str(engine.get("name")),
            as_str(row.get("case_id")),
            harness_class(row),
            f"{requested}",
            format_float(workload.get("prompt_tokens"), digits=0),
            format_float(workload.get("generated_tokens"), digits=0),
            format_float(metrics.get("prefill_tokens_per_second")),
            format_float(metrics.get("decode_tokens_per_second")),
            format_float(metrics.get("total_tokens_per_second")),
            format_gib(memory.get("vram_consumed_bytes"))
            or format_gib(memory.get("consumed_total_bytes"))
            or "-",
            format_float(metrics.get("decode_tokens_per_second_times_vram_consumed_gib")),
        ]


def markdown_lines(
    paths: Iterable[Path],
    workload_prefix_filter: str,
    case_substring: str,
    requests_filter: set[int],
) -> Iterator[str]:
    header = [
        "Engine",
        "Case",
        "Harness",
        "Requests",
        "Prompt tokens",
        "Generated tokens",
        "Prefill tok/s",
        "Decode tok/s",
        "Total tok/s",
        "Consumed GiB",
        "Decode x GiB",
    ]
    yield "| " + " | ".join(header) + " |"
    yield "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    for row in iter_markdown_rows(
        paths, workload_prefix_filter, case_substring, requests_filter
    ):
        yield (
            "| "
            + " | ".join(
                [
                    as_str(row[0]),
                    as_str(row[1]),
                    as_str(row[2]),
                    as_str(row[3]),
                    as_str(row[4]),
                    as_str(row[5]),
                    as_str(row[6]),
                    as_str(row[7]),
                    as_str(row[8]),
                    as_str(row[9]),
                    as_str(row[10]),
                ]
            )
            + " |"
        )


def markdown_table(
    paths: list[Path],
    workload_prefix_filter: str,
    case_substring: str,
    requests_filter: set[int] | None = None,
) -> str:
    return "\n".join(
        markdown_lines(paths, workload_prefix_filter, case_substring, requests_filter or set())
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument(
        "--workload-prefix",
        default="",
        help="filter by workload prefix like pp16-tg8",
    )
    parser.add_argument("--case-substring", default="")
    parser.add_argument(
        "--requests",
        default="",
        help="comma-separated concurrent request counts to keep, for example 2,4,8",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in args.jsonl:
        if not path.exists():
            print(f"input file does not exist: {path}", file=sys.stderr)
            return 1
    try:
        requests_filter = parse_requests_filter(args.requests)
        for line in markdown_lines(
            args.jsonl,
            args.workload_prefix,
            args.case_substring,
            requests_filter,
        ):
            print(line)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
