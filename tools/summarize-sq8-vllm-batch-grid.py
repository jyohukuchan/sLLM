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


def format_mib(bytes_value: Any) -> str | None:
    value = as_int(bytes_value)
    if value is None:
        return None
    return f"{value / 1024 ** 2:.2f}"


def format_slash_value(
    left: Any,
    right: Any,
    formatter,
) -> str:
    left_present = left is not None
    right_present = right is not None
    if not left_present and not right_present:
        return "-"

    left_text = formatter(left)
    right_text = formatter(right)
    if left_text is None:
        left_text = "?"
    if right_text is None:
        right_text = "?"
    return f"{left_text}/{right_text}"


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


def iter_markdown_rows(
    rows: Iterable[dict[str, Any]], show_sq_details: bool = False
) -> Iterator[list[str]]:
    for row in rows:
        workload = as_dict(row.get("workload"))
        metrics = as_dict(row.get("metrics"))
        memory = as_dict(row.get("memory"))
        engine = as_dict(row.get("engine"))
        requested = requested_concurrency(workload) or 0
        output = [
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
        if show_sq_details:
            batch_matvec = as_int(workload.get("sq_fp8_batch_matvec_count"))
            expected_batch_matvec = as_int(
                workload.get("sq_fp8_expected_all_batch_matvec_count")
            )
            host_staging_read_count = workload.get(
                "sq_diagnostic_host_staging_read_count"
            )
            host_staging_write_count = workload.get(
                "sq_diagnostic_host_staging_write_count"
            )
            host_staging_read_bytes = workload.get(
                "sq_diagnostic_host_staging_read_bytes"
            )
            host_staging_write_bytes = workload.get(
                "sq_diagnostic_host_staging_write_bytes"
            )
            output.extend(
                [
                    as_str(workload.get("sq_projection_boundary")),
                    as_str(workload.get("sq_projection_kernel_families")),
                    f"{batch_matvec}/{expected_batch_matvec}"
                    if batch_matvec is not None and expected_batch_matvec is not None
                    else "-",
                    format_slash_value(
                        host_staging_read_count,
                        host_staging_write_count,
                        as_int,
                    ),
                    format_slash_value(
                        host_staging_read_bytes,
                        host_staging_write_bytes,
                        format_mib,
                    ),
                ]
            )
        yield output


def markdown_lines(
    rows: Iterable[dict[str, Any]], show_sq_details: bool = False
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
    if show_sq_details:
        header.extend(
            [
                "SQ boundary",
                "SQ family",
                "SQ batch",
                "SQ staging ops",
                "SQ staging MiB",
            ]
        )
    yield "| " + " | ".join(header) + " |"
    separator = [
        "---",
        "---",
        "---",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
    ]
    if show_sq_details:
        separator.extend(["---", "---", "---", "---:", "---:"])
    yield "| " + " | ".join(separator) + " |"
    for row in iter_markdown_rows(rows, show_sq_details=show_sq_details):
        yield (
            "| "
            + " | ".join(as_str(cell) for cell in row)
            + " |"
        )


def markdown_table(
    paths: list[Path],
    workload_prefix_filter: str,
    case_substring: str,
    requests_filter: set[int] | None = None,
    harness_class_filter: str = "",
    show_sq_details: bool = False,
) -> str:
    rows = selected_rows(
        paths,
        workload_prefix_filter,
        case_substring,
        requests_filter or set(),
        harness_class_filter,
    )
    return "\n".join(markdown_lines(rows, show_sq_details=show_sq_details))


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

    def _entry_is_valid(entry: str) -> bool:
        if "=" not in entry:
            return False
        boundary, family = (part.strip() for part in entry.split("=", 1))
        if not boundary or not family:
            return False
        return family.lower() != "none"

    for row in rows:
        engine_name = as_str(as_dict(row.get("engine")).get("name"))
        workload = as_dict(row.get("workload"))
        if engine_name != "uLLM":
            continue
        if as_str(workload.get("format_id")) != "SQ8_0":
            continue

        kernel_families = workload.get("sq_projection_kernel_families")
        if not isinstance(kernel_families, str):
            failures.append(
                f"selected uLLM SQ8_0 row missing valid workload.sq_projection_kernel_families: {harness_summary(row)}"
            )
            continue

        entries = [part.strip() for part in kernel_families.split(",")]
        if (
            not kernel_families.strip()
            or kernel_families.strip().lower() == "none"
            or not entries
            or any(not _entry_is_valid(entry) for entry in entries)
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


def _all_equal_integer_from_list(values: Any) -> int | None:
    if not isinstance(values, list):
        return None
    if not values:
        return None

    first: int | None = None
    for raw in values:
        current = as_int(raw)
        if current is None:
            return None
        if first is None:
            first = current
        elif current != first:
            return None
    return first


def _per_request_shape_for_row(
    workload: dict[str, Any],
    request_count: int,
) -> tuple[int, int] | None:
    prompt_per_request = workload.get("prompt_tokens_per_request")
    generated_per_request = workload.get("generated_tokens_per_request")
    has_explicit_shape = (
        "prompt_tokens_per_request" in workload
        or "generated_tokens_per_request" in workload
    )

    if has_explicit_shape:
        prompt_value = _all_equal_integer_from_list(prompt_per_request)
        generated_value = _all_equal_integer_from_list(generated_per_request)
        if prompt_value is None or generated_value is None:
            return None
        return (prompt_value, generated_value)

    if request_count <= 0:
        return None

    prompt_tokens = as_int(workload.get("prompt_tokens"))
    generated_tokens = as_int(workload.get("generated_tokens"))
    if (
        prompt_tokens is None
        or generated_tokens is None
        or prompt_tokens % request_count != 0
        or generated_tokens % request_count != 0
    ):
        return None

    return (prompt_tokens // request_count, generated_tokens // request_count)


def _per_request_shapes_for_rows(
    rows: list[dict[str, Any]],
    request_count: int,
) -> set[tuple[int, int]]:
    return {
        shape
        for row in rows
        for shape in (
            _per_request_shape_for_row(as_dict(row.get("workload")), request_count),
        )
        if shape is not None
    }


def _shape_set_to_text(shapes: set[tuple[int, int]]) -> str:
    return ", ".join(f"{shape}" for shape in sorted(shapes)) or "-"


def _model_name_set_to_text(names: set[str]) -> str:
    return ", ".join(sorted(names)) or "-"


def _model_name_set(rows: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for row in rows:
        model = as_dict(row.get("model"))
        raw_name = model.get("name")
        if raw_name is None:
            continue
        if not isinstance(raw_name, str):
            raw_name = str(raw_name)
        name = raw_name.strip()
        if name:
            names.add(name)
    return names


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
            if not _sq_projection_boundary_has_batch(workload.get("sq_projection_boundary")):
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

        if ullm_rows and vllm_rows:
            ullm_shapes = _per_request_shapes_for_rows(ullm_rows, request_count)
            vllm_shapes = _per_request_shapes_for_rows(vllm_rows, request_count)
            ullm_model_names = _model_name_set(ullm_rows)
            vllm_model_names = _model_name_set(vllm_rows)

            if not ullm_shapes:
                failures.append(
                    f"request {request_count}: missing per-request prompt/generated shape for uLLM "
                    f"(uLLM shapes={_shape_set_to_text(ullm_shapes)} "
                    f"vLLM shapes={_shape_set_to_text(vllm_shapes)})"
                )
            if not vllm_shapes:
                failures.append(
                    f"request {request_count}: missing per-request prompt/generated shape for vLLM "
                    f"(uLLM shapes={_shape_set_to_text(ullm_shapes)} "
                    f"vLLM shapes={_shape_set_to_text(vllm_shapes)})"
                )
            if not ullm_model_names or not vllm_model_names:
                failures.append(
                    f"request {request_count}: missing model.name for normalized model comparison "
                    f"(uLLM={_model_name_set_to_text(ullm_model_names)} "
                    f"vLLM={_model_name_set_to_text(vllm_model_names)})"
                )

            shape_intersection = ullm_shapes.intersection(vllm_shapes)
            if ullm_shapes and vllm_shapes and not shape_intersection:
                failures.append(
                    "request "
                    f"{request_count} per-request prompt/generated shape mismatch: "
                    f"uLLM={sorted(ullm_shapes)} vLLM={sorted(vllm_shapes)}"
                )
            model_name_intersection = ullm_model_names.intersection(vllm_model_names)
            if ullm_model_names and vllm_model_names and not model_name_intersection:
                failures.append(
                    f"request {request_count}: model.name mismatch "
                    f"uLLM={_model_name_set_to_text(ullm_model_names)} "
                    f"vLLM={_model_name_set_to_text(vllm_model_names)}"
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


def ullm_sq_no_host_staging_gate_failures(
    rows: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    host_staging_fields = (
        "sq_diagnostic_host_staging_read_count",
        "sq_diagnostic_host_staging_write_count",
        "sq_diagnostic_host_staging_read_bytes",
        "sq_diagnostic_host_staging_write_bytes",
    )
    for row in rows:
        engine_name = as_str(as_dict(row.get("engine")).get("name"))
        workload = as_dict(row.get("workload"))
        if engine_name != "uLLM":
            continue
        if as_str(workload.get("format_id")) != "SQ8_0":
            continue

        for field in host_staging_fields:
            raw_value = workload.get(field)
            if raw_value is None:
                failures.append(
                    f"selected uLLM SQ8_0 row missing host staging metric: "
                    f"{harness_summary(row)} {field} is missing"
                )
                continue
            value = as_int(raw_value)
            if value is None:
                failures.append(
                    f"selected uLLM SQ8_0 row has malformed host staging metric: "
                    f"{harness_summary(row)} {field}={as_str(raw_value)}"
                )
                break
            if value != 0:
                failures.append(
                    f"selected uLLM SQ8_0 row has non-zero host staging metric: "
                    f"{harness_summary(row)} {field}={as_str(raw_value)}"
                )
                break
    return failures


def ullm_sq_host_staging_write_count_gate_failures(
    rows: list[dict[str, Any]],
    max_write_count: int,
) -> list[str]:
    failures: list[str] = []
    for row in rows:
        engine_name = as_str(as_dict(row.get("engine")).get("name"))
        workload = as_dict(row.get("workload"))
        if engine_name != "uLLM":
            continue
        if as_str(workload.get("format_id")) != "SQ8_0":
            continue
        raw_value = workload.get("sq_diagnostic_host_staging_write_count")
        if raw_value is None:
            continue
        write_count = as_int(raw_value)
        if write_count is None:
            failures.append(
                f"selected uLLM SQ8_0 row has malformed host staging write count: "
                f"{harness_summary(row)} sq_diagnostic_host_staging_write_count={as_str(raw_value)}"
            )
            continue
        if write_count > max_write_count:
            failures.append(
                "selected uLLM SQ8_0 row exceeds max sq_diagnostic_host_staging_write_count: "
                f"{harness_summary(row)} sq_diagnostic_host_staging_write_count={write_count} "
                f"max={max_write_count}"
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
        "--require-ullm-sq-no-host-staging",
        action="store_true",
        help=(
            "fail when selected uLLM SQ8_0 rows are missing "
            "workload.sq_diagnostic_host_staging_* values or any value is non-zero"
        ),
    )
    parser.add_argument(
        "--max-ullm-sq-host-staging-write-count",
        type=int,
        default=None,
        help=(
            "fail when selected uLLM SQ8_0 rows have workload.sq_diagnostic_host_staging_write_count "
            "greater than this limit"
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
    parser.add_argument(
        "--show-sq-details",
        action="store_true",
        help=(
            "show SQ8_0 implementation details columns: "
            "SQ boundary, SQ family, SQ batch, SQ staging ops, SQ staging MiB"
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
        if (
            args.max_ullm_sq_host_staging_write_count is not None
            and args.max_ullm_sq_host_staging_write_count < 0
        ):
            raise ValueError(
                "--max-ullm-sq-host-staging-write-count must be a non-negative integer"
            )
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
            or args.require_ullm_sq_no_host_staging
            or args.max_ullm_sq_host_staging_write_count is not None
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
            if args.require_ullm_sq_no_host_staging:
                ullm_sq_no_host_staging_failures = (
                    ullm_sq_no_host_staging_gate_failures(rows)
                )
            else:
                ullm_sq_no_host_staging_failures = []
            if args.max_ullm_sq_host_staging_write_count is not None:
                max_ullm_sq_host_staging_write_count = (
                    args.max_ullm_sq_host_staging_write_count
                )
                ullm_sq_host_staging_write_count_failures = (
                    ullm_sq_host_staging_write_count_gate_failures(
                        rows, max_ullm_sq_host_staging_write_count
                    )
                )
            else:
                ullm_sq_host_staging_write_count_failures = []
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
            ullm_sq_no_host_staging_failures = []
            ullm_sq_host_staging_write_count_failures = []
            normalized_throughput_comparison_failures = []
        for line in markdown_lines(rows, show_sq_details=args.show_sq_details):
            print(line)
        if (
            serving_parity_failures
            or required_engine_failures
            or ullm_sq_kernel_families_failures
            or ullm_sq_batch_coverage_failures
            or ullm_sq_no_host_staging_failures
            or ullm_sq_host_staging_write_count_failures
            or normalized_throughput_comparison_failures
        ):
            print(
                f"batch-grid gate failed: {selected_count} selected row(s)",
                file=sys.stderr,
            )
            for line in (
                serving_parity_failures
                + required_engine_failures
                + ullm_sq_kernel_families_failures
                + ullm_sq_batch_coverage_failures
                + ullm_sq_no_host_staging_failures
                + ullm_sq_host_staging_write_count_failures
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
