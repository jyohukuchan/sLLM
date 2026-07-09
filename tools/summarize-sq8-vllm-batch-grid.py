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


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    return None


def as_str(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_lower_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    return value.strip().lower()


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


def requested_concurrency(workload: dict[str, Any]) -> int | None:
    return (
        as_int(workload.get("concurrent_requests"))
        or as_int(workload.get("batch_size"))
    )


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


def serving_parity_candidate(row: dict[str, Any]) -> bool:
    harness = as_dict(row.get("harness"))
    explicit_candidate = as_bool(harness.get("serving_parity_candidate"))
    if explicit_candidate is not None:
        return explicit_candidate
    return harness_class(row) == "serving_throughput_benchmark"


def harness_summary(row: dict[str, Any]) -> str:
    requests = requested_concurrency(as_dict(row.get("workload"))) or "-"
    return (
        f"case_id={as_str(row.get('case_id'))} "
        f"requests={requests} "
        f"harness_class={harness_class(row)} "
        f"serving_parity_candidate={serving_parity_candidate(row)}"
    )


def iter_selected_rows(
    paths: Iterable[Path],
    workload_prefix_filter: str,
    case_substring: str,
    requests_filter: set[int],
    harness_class_filter: str = "",
) -> Iterator[dict[str, Any]]:
    for row in iter_rows(paths):
        if should_keep(
            row,
            workload_prefix_filter,
            case_substring,
            requests_filter,
            harness_class_filter,
        ):
            yield row


def selected_rows(
    paths: Iterable[Path],
    workload_prefix_filter: str,
    case_substring: str,
    requests_filter: set[int],
    harness_class_filter: str = "",
) -> list[dict[str, Any]]:
    return list(
        iter_selected_rows(
            paths,
            workload_prefix_filter,
            case_substring,
            requests_filter,
            harness_class_filter,
        )
    )


def serving_parity_gate_failures(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["no selected rows"]

    failure_reasons: list[str] = []

    false_candidates = [row for row in rows if not serving_parity_candidate(row)]
    if false_candidates:
        failure_reasons.append("selected rows include serving_parity_candidate=false")
        failure_reasons.extend(f"  - {harness_summary(row)}" for row in false_candidates)

    classes = {harness_class(row) for row in rows}
    if len(classes) > 1:
        joined_classes = ", ".join(sorted(classes))
        failure_reasons.append(
            f"selected rows include mixed harness.class values: {joined_classes}"
        )
        for row in rows:
            failure_reasons.append(f"  - {harness_summary(row)}")

    return failure_reasons


def should_keep(
    row: dict[str, Any],
    workload_prefix_filter: str,
    case_substring: str,
    requests_filter: set[int],
    harness_class_filter: str = "",
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

    if harness_class_filter:
        harness_match = harness_class(row) == harness_class_filter
    else:
        harness_match = True

    return (
        case_match
        and workload_match
        and requests_match
        and status_match
        and harness_match
    )


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


def parse_required_engines_filter(value: str) -> set[str]:
    if not value:
        return set()
    required: set[str] = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            raise ValueError("--require-engines contains an empty item")
        required.add(item)
    return required


def iter_markdown_rows(rows: Iterable[dict[str, Any]]) -> Iterator[list[str]]:
    for row in rows:
        workload = as_dict(row.get("workload"))
        metrics = as_dict(row.get("metrics"))
        memory = as_dict(row.get("memory"))
        engine = as_dict(row.get("engine"))
        requested = requested_concurrency(workload) or 0
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


def markdown_lines(rows: Iterable[dict[str, Any]]) -> Iterator[str]:
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
    for row in iter_markdown_rows(rows):
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
    harness_class_filter: str = "",
) -> str:
    rows = selected_rows(
        paths,
        workload_prefix_filter,
        case_substring,
        requests_filter or set(),
        harness_class_filter,
    )
    return "\n".join(markdown_lines(rows))


def required_engines_gate_failures(
    rows: list[dict[str, Any]],
    required_engines: set[str],
) -> list[str]:
    if not required_engines:
        return []
    available = {as_str(as_dict(row.get("engine")).get("name")) for row in rows}
    missing = sorted(required_engines - available)
    if not missing:
        return []
    return [f"missing required engine(s): {', '.join(missing)}"]


def ullm_sq_kernel_families_gate_failures(
    rows: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    for row in rows:
        engine_name = as_str(as_dict(row.get("engine")).get("name"))
        workload = as_dict(row.get("workload"))
        if engine_name != "uLLM":
            continue
        if as_str(workload.get("format_id")) != "SQ8_0":
            continue

        kernel_families = workload.get("sq_projection_kernel_families")
        if not isinstance(kernel_families, str) or (
            not kernel_families.strip() or kernel_families.strip().lower() == "none"
        ):
            failures.append(
                f"selected uLLM SQ8_0 row missing valid workload.sq_projection_kernel_families: {harness_summary(row)}"
            )
    return failures


def _sq_projection_boundary_has_batch(boundary: Any) -> bool:
    if not isinstance(boundary, str):
        return False
    if not boundary.strip():
        return False
    parts = [part.strip().lower() for part in boundary.split("+")]
    return "batch" in parts


def _request_case_ids(rows: list[dict[str, Any]]) -> str:
    case_ids = {
        as_str(row.get("case_id")) for row in rows if row.get("case_id") is not None
    }
    if not case_ids:
        return ""
    return f" (case_ids: {', '.join(sorted(case_ids))})"


def _normalized_row_has_real_batch(row: dict[str, Any]) -> bool:
    workload = as_dict(row.get("workload"))
    batching = as_dict(row.get("batching"))
    return as_lower_str(batching.get("mode")) == "real" or (
        as_lower_str(workload.get("batching_mode")) == "real"
    )


def _normalized_field_true(row: dict[str, Any], field: str) -> bool:
    workload = as_dict(row.get("workload"))
    batching = as_dict(row.get("batching"))
    return as_bool(workload.get(field)) is True or as_bool(batching.get(field)) is True


def _normalized_field_false(row: dict[str, Any], field: str) -> bool:
    workload = as_dict(row.get("workload"))
    batching = as_dict(row.get("batching"))
    return as_bool(workload.get(field)) is False or as_bool(batching.get(field)) is False


def normalized_throughput_comparison_gate_failures(
    rows: list[dict[str, Any]],
    requests_filter: set[int],
) -> list[str]:
    if not rows:
        return ["no selected rows"]

    if requests_filter:
        request_counts = sorted(requests_filter)
    else:
        request_counts = sorted(
            {
                requested
                for row in rows
                if (requested := requested_concurrency(as_dict(row.get("workload"))))
                is not None
            }
        )
    if not request_counts:
        return ["no selected rows with request count"]

    failures: list[str] = []
    for request_count in request_counts:
        request_rows = [
            row
            for row in rows
            if requested_concurrency(as_dict(row.get("workload"))) == request_count
        ]
        ullm_rows = [
            row
            for row in request_rows
            if as_str(as_dict(row.get("engine")).get("name")) == "uLLM"
        ]
        vllm_rows = [
            row
            for row in request_rows
            if as_str(as_dict(row.get("engine")).get("name")) == "vLLM"
        ]
        if not ullm_rows:
            failures.append(
                f"request {request_count} missing required uLLM row for normalized throughput comparison"
                f"{_request_case_ids(vllm_rows)}"
            )
        if not vllm_rows:
            failures.append(
                f"request {request_count} missing required vLLM row for normalized throughput comparison"
                f"{_request_case_ids(ullm_rows)}"
            )

        for row in request_rows:
            row_engine = as_str(as_dict(row.get("engine")).get("name"))
            case_id = as_str(row.get("case_id"))
            prefix = f"request {request_count} case_id={case_id}"
            if row_engine == "vLLM":
                if harness_class(row) != "serving_throughput_benchmark":
                    failures.append(
                        f"{prefix}: vLLM row must be harness.class=serving_throughput_benchmark "
                        f"(got {harness_class(row)})"
                    )
                continue
            if row_engine != "uLLM":
                continue

            workload = as_dict(row.get("workload"))
            if harness_class(row) != "cli_model_loop_diagnostic":
                failures.append(
                    f"{prefix}: uLLM row must be harness.class=cli_model_loop_diagnostic "
                    f"(got {harness_class(row)})"
                )
            if as_str(workload.get("format_id")) != "SQ8_0":
                failures.append(
                    f"{prefix}: uLLM row format_id must be SQ8_0 "
                    f"(got {as_str(workload.get('format_id'))})"
                )
            if not _normalized_row_has_real_batch(row):
                failures.append(
                    f"{prefix}: uLLM row requires batching mode real "
                    "(from batching.mode or workload.batching_mode)"
                )
            if not _normalized_field_true(row, "prefill_real_batch"):
                failures.append(
                    f"{prefix}: uLLM row requires prefill_real_batch=true "
                    "(from workload.prefill_real_batch or batching.prefill_real_batch)"
                )
            if not _normalized_field_true(row, "decode_real_batch"):
                failures.append(
                    f"{prefix}: uLLM row requires decode_real_batch=true "
                    "(from workload.decode_real_batch or batching.decode_real_batch)"
                )
            if not _normalized_field_false(row, "final_logits_in_total"):
                failures.append(
                    f"{prefix}: uLLM row requires final_logits_in_total=false "
                    "(from workload.final_logits_in_total or batching.final_logits_in_total)"
                )
            if not _sq_projection_boundary_has_batch(
                workload.get("sq_projection_boundary")
            ):
                failures.append(
                    f"{prefix}: uLLM row requires sq_projection_boundary containing 'batch' "
                    f"(got {as_str(workload.get('sq_projection_boundary'))})"
                )
            batch_count = as_int(workload.get("sq_fp8_batch_matvec_count"))
            expected_batch_count = as_int(
                workload.get("sq_fp8_expected_all_batch_matvec_count")
            )
            if batch_count is None or batch_count <= 0:
                failures.append(
                    f"{prefix}: uLLM row requires positive sq_fp8_batch_matvec_count "
                    f"(got {as_str(workload.get('sq_fp8_batch_matvec_count'))})"
                )
            if expected_batch_count is None or expected_batch_count <= 0:
                failures.append(
                    f"{prefix}: uLLM row requires positive "
                    f"sq_fp8_expected_all_batch_matvec_count "
                    f"(got {as_str(workload.get('sq_fp8_expected_all_batch_matvec_count'))})"
                )
            if (
                batch_count is not None
                and expected_batch_count is not None
                and batch_count < expected_batch_count
            ):
                failures.append(
                    f"{prefix}: uLLM row requires sq_fp8_batch_matvec_count >= sq_fp8_expected_all_batch_matvec_count "
                    f"(got {batch_count}/{expected_batch_count})"
                )

    return failures


def ullm_sq_batch_coverage_gate_failures(
    rows: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    for row in rows:
        engine_name = as_str(as_dict(row.get("engine")).get("name"))
        workload = as_dict(row.get("workload"))
        if engine_name != "uLLM":
            continue
        if as_str(workload.get("format_id")) != "SQ8_0":
            continue

        boundary = workload.get("sq_projection_boundary")
        if not _sq_projection_boundary_has_batch(boundary):
            failures.append(
                "selected uLLM SQ8_0 row missing batch projection boundary: "
                f"{harness_summary(row)} sq_projection_boundary={as_str(boundary)}"
            )
            continue

        batch_count = as_int(workload.get("sq_fp8_batch_matvec_count"))
        expected_batch_count = as_int(
            workload.get("sq_fp8_expected_all_batch_matvec_count")
        )
        if batch_count is None or batch_count <= 0:
            failures.append(
                "selected uLLM SQ8_0 row missing or non-positive sq_fp8_batch_matvec_count: "
                f"{harness_summary(row)} sq_fp8_batch_matvec_count={as_str(workload.get('sq_fp8_batch_matvec_count'))}"
            )
            continue
        if expected_batch_count is None or expected_batch_count <= 0:
            failures.append(
                "selected uLLM SQ8_0 row missing or non-positive sq_fp8_expected_all_batch_matvec_count: "
                f"{harness_summary(row)} sq_fp8_expected_all_batch_matvec_count={as_str(workload.get('sq_fp8_expected_all_batch_matvec_count'))}"
            )
            continue
        if batch_count < expected_batch_count:
            failures.append(
                "selected uLLM SQ8_0 row lacks full batch projection coverage: "
                f"{harness_summary(row)} sq_fp8_batch_matvec_count={batch_count} "
                f"sq_fp8_expected_all_batch_matvec_count={expected_batch_count}"
            )
    return failures


def required_engines_grid_gate_failures(
    rows: list[dict[str, Any]],
    required_engines: set[str],
    requests_filter: set[int],
) -> list[str]:
    if not required_engines:
        return []

    if requests_filter:
        request_counts = requests_filter
    else:
        request_counts = {
            requested_concurrency(as_dict(row.get("workload")))
            for row in rows
            if requested_concurrency(as_dict(row.get("workload"))) is not None
        }
    if not request_counts:
        return required_engines_gate_failures(rows, required_engines)

    failure_reasons: list[str] = []
    for request_count in sorted(request_counts):
        request_rows = [
            row
            for row in rows
            if requested_concurrency(as_dict(row.get("workload"))) == request_count
        ]
        available = {
            as_str(as_dict(row.get("engine")).get("name")) for row in request_rows
        }
        missing = sorted(required_engines - available)
        if missing:
            failure_reasons.append(
                f"request {request_count} missing required engine(s): {', '.join(missing)}"
            )

    return failure_reasons


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
    parser.add_argument(
        "--require-serving-parity",
        action="store_true",
        help="fail when selected rows are not serving parity candidates",
    )
    parser.add_argument(
        "--harness-class",
        default="",
        help="filter by harness class, for example serving_throughput_benchmark",
    )
    parser.add_argument(
        "--require-engines",
        default="",
        help="comma-separated required engine names, for example uLLM,vLLM",
    )
    parser.add_argument(
        "--require-engine-grid",
        action="store_true",
        help="require required engines per request count instead of global coverage",
    )
    parser.add_argument(
        "--require-ullm-sq-kernel-families",
        action="store_true",
        help="fail when uLLM SQ8_0 rows miss a non-none workload.sq_projection_kernel_families",
    )
    parser.add_argument(
        "--require-ullm-sq-batch-coverage",
        action="store_true",
        help=(
            "fail when uLLM SQ8_0 rows lack batch projection coverage with "
            "valid sq_projection_boundary and matvec counters"
        ),
    )
    parser.add_argument(
        "--require-normalized-throughput-comparison",
        action="store_true",
        help=(
            "require a same-shape normalized throughput comparison shape: "
            "uLLM cli_model_loop_diagnostic and vLLM serving rows per request count"
        ),
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
        required_engines = parse_required_engines_filter(args.require_engines)
        if args.require_engine_grid and not required_engines:
            raise ValueError(
                "--require-engine-grid requires --require-engines to be specified"
            )

        harness_class_filter = args.harness_class
        requires_gate = (
            args.require_serving_parity
            or bool(required_engines)
            or args.require_ullm_sq_kernel_families
            or args.require_ullm_sq_batch_coverage
            or args.require_normalized_throughput_comparison
        )
        if requires_gate:
            rows = selected_rows(
                args.jsonl,
                args.workload_prefix,
                args.case_substring,
                requests_filter,
                harness_class_filter,
            )
            selected_count = len(rows)
            if args.require_serving_parity:
                serving_parity_failures = serving_parity_gate_failures(rows)
            else:
                serving_parity_failures = []
            if args.require_engine_grid:
                required_engine_failures = required_engines_grid_gate_failures(
                    rows, required_engines, requests_filter
                )
            else:
                required_engine_failures = required_engines_gate_failures(
                    rows, required_engines
                )
            if args.require_ullm_sq_kernel_families:
                ullm_sq_kernel_families_failures = (
                    ullm_sq_kernel_families_gate_failures(rows)
                )
            else:
                ullm_sq_kernel_families_failures = []
            if args.require_ullm_sq_batch_coverage:
                ullm_sq_batch_coverage_failures = (
                    ullm_sq_batch_coverage_gate_failures(rows)
                )
            else:
                ullm_sq_batch_coverage_failures = []
            if args.require_normalized_throughput_comparison:
                normalized_throughput_comparison_failures = (
                    normalized_throughput_comparison_gate_failures(
                        rows, requests_filter
                    )
                )
            else:
                normalized_throughput_comparison_failures = []
        else:
            rows = iter_selected_rows(
                args.jsonl,
                args.workload_prefix,
                args.case_substring,
                requests_filter,
                harness_class_filter,
            )
            selected_count = None
            serving_parity_failures = []
            required_engine_failures = []
            ullm_sq_kernel_families_failures = []
            ullm_sq_batch_coverage_failures = []
            normalized_throughput_comparison_failures = []
        for line in markdown_lines(rows):
            print(line)
        if (
            serving_parity_failures
            or required_engine_failures
            or ullm_sq_kernel_families_failures
            or ullm_sq_batch_coverage_failures
            or normalized_throughput_comparison_failures
        ):
            print(
                f"serving parity gate failed: {selected_count} selected row(s)",
                file=sys.stderr,
            )
            for line in (
                serving_parity_failures
                + required_engine_failures
                + ullm_sq_kernel_families_failures
                + ullm_sq_batch_coverage_failures
                + normalized_throughput_comparison_failures
            ):
                print(line, file=sys.stderr)
            return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
