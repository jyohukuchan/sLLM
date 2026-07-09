from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch
from typing import Any
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_tool(filename: str):
    path = REPO_ROOT / "tools" / filename
    module_name = filename.replace("-", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = load_tool("summarize-sq8-vllm-batch-grid.py")


def make_row(
    *,
    case_id: str,
    engine_name: str,
    prompt_tokens: int,
    generated_tokens: int,
    batch_size: int,
    prefill_tps: float = 12.3,
    decode_tps: float = 23.4,
    total_tps: float = 35.7,
    consumed_bytes: int = 10 * 1024**3,
    decode_x_gib: float = 100.0,
    harness: dict | None = None,
    format_id: str | None = None,
    sq_projection_kernel_families: str | None = None,
    sq_projection_boundary: str | None = None,
    sq_fp8_batch_matvec_count: int | None = None,
    sq_fp8_expected_all_batch_matvec_count: int | None = None,
    prompt_tokens_per_request: list[int] | None = None,
    generated_tokens_per_request: list[int] | None = None,
    batching: dict[str, Any] | None = None,
    batching_mode: str | None = None,
    prefill_real_batch: bool | None = None,
    decode_real_batch: bool | None = None,
    final_logits_in_total: bool | None = None,
    sq_diagnostic_host_staging_read_count: int | None = None,
    sq_diagnostic_host_staging_write_count: int | None = None,
    sq_diagnostic_host_staging_read_bytes: int | None = None,
    sq_diagnostic_host_staging_write_bytes: int | None = None,
    model_name: str | None = None,
    model_format: str | None = None,
    model_quantization: str | None = None,
    context_length: int | None = None,
    result_validity: dict[str, Any] | None = None,
) -> dict:
    row = {
        "case_id": case_id,
        "status": "ok",
        "engine": {"name": engine_name},
        "workload": {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "concurrent_requests": batch_size,
        },
        "metrics": {
            "prefill_tokens_per_second": prefill_tps,
            "decode_tokens_per_second": decode_tps,
            "total_tokens_per_second": total_tps,
            "decode_tokens_per_second_times_vram_consumed_gib": decode_x_gib,
        },
        "memory": {"vram_consumed_bytes": consumed_bytes},
    }
    if harness is not None:
        row["harness"] = harness
    if format_id is not None:
        row["workload"]["format_id"] = format_id
    if sq_projection_kernel_families is not None:
        row["workload"]["sq_projection_kernel_families"] = sq_projection_kernel_families
    if sq_projection_boundary is not None:
        row["workload"]["sq_projection_boundary"] = sq_projection_boundary
    if sq_fp8_batch_matvec_count is not None:
        row["workload"]["sq_fp8_batch_matvec_count"] = sq_fp8_batch_matvec_count
    if sq_fp8_expected_all_batch_matvec_count is not None:
        row["workload"]["sq_fp8_expected_all_batch_matvec_count"] = (
            sq_fp8_expected_all_batch_matvec_count
        )
    if prompt_tokens_per_request is not None:
        row["workload"]["prompt_tokens_per_request"] = prompt_tokens_per_request
    if generated_tokens_per_request is not None:
        row["workload"]["generated_tokens_per_request"] = generated_tokens_per_request
    if batching is not None:
        row["batching"] = batching
    if batching_mode is not None:
        row["workload"]["batching_mode"] = batching_mode
    if prefill_real_batch is not None:
        row["workload"]["prefill_real_batch"] = prefill_real_batch
    if decode_real_batch is not None:
        row["workload"]["decode_real_batch"] = decode_real_batch
    if final_logits_in_total is not None:
        row["workload"]["final_logits_in_total"] = final_logits_in_total
    if sq_diagnostic_host_staging_read_count is not None:
        row["workload"][
            "sq_diagnostic_host_staging_read_count"
        ] = sq_diagnostic_host_staging_read_count
    if sq_diagnostic_host_staging_write_count is not None:
        row["workload"][
            "sq_diagnostic_host_staging_write_count"
        ] = sq_diagnostic_host_staging_write_count
    if sq_diagnostic_host_staging_read_bytes is not None:
        row["workload"][
            "sq_diagnostic_host_staging_read_bytes"
        ] = sq_diagnostic_host_staging_read_bytes
    if sq_diagnostic_host_staging_write_bytes is not None:
        row["workload"][
            "sq_diagnostic_host_staging_write_bytes"
        ] = sq_diagnostic_host_staging_write_bytes
    if model_name is not None:
        row["model"] = {"name": model_name}
    if model_format is not None:
        row.setdefault("model", {})["format"] = model_format
    if model_quantization is not None:
        row.setdefault("model", {})["quantization"] = model_quantization
    if context_length is not None:
        row["workload"]["context_length"] = context_length
    if result_validity is not None:
        row["result_validity"] = result_validity
    return row


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


class SummarizeSq8VllmBatchGridTests(unittest.TestCase):
    def test_table_filter_and_columns_for_pp16_tg8(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "results.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-pp16-tg8-b2",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                            )
                        ),
                        json.dumps(make_row(case_id="vllm-pp16-tg8-b1", engine_name="vLLM", prompt_tokens=16, generated_tokens=8, batch_size=1)),
                        json.dumps(make_row(case_id="sq8-pp16-tg16-b2", engine_name="uLLM", prompt_tokens=16, generated_tokens=16, batch_size=2)),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            table = TOOL.markdown_table([path], "pp16-tg8", "", {2})
            lines = table.splitlines()

            self.assertTrue(lines[0].startswith("| Engine | Case | Harness | Requests |"))
            self.assertIn(
                "uLLM | sq8-mixed-real-batch-no-final-pp16-tg8-b2 | cli_model_loop_diagnostic",
                table,
            )
            self.assertIn("vLLM | vllm-pp16-tg8-b2 | serving_throughput_benchmark", table)
            self.assertIn("vLLM | vllm-pp16-tg8-b2", table)
            self.assertNotIn("vllm-pp16-tg8-b1", table)
            self.assertNotIn("sq8-pp16-tg16-b2", table)
            self.assertIn("16", table)
            self.assertIn("8", table)
            self.assertIn("10.00", table)
            self.assertIn("12.30", table)

    def test_markdown_table_hides_sq_details_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "results.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                format_id="SQ8_0",
                                harness={"class": "cli_model_loop_diagnostic"},
                                sq_projection_boundary="batch",
                                sq_projection_kernel_families="batch=direct",
                                sq_fp8_batch_matvec_count=6720,
                                sq_fp8_expected_all_batch_matvec_count=6720,
                                sq_diagnostic_host_staging_read_count=0,
                                sq_diagnostic_host_staging_write_count=72,
                                sq_diagnostic_host_staging_read_bytes=0,
                                sq_diagnostic_host_staging_write_bytes=1572864,
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            table = TOOL.markdown_table([path], "pp16-tg8", "", {2})
            lines = table.splitlines()
            self.assertTrue(lines[0].startswith("| Engine | Case | Harness | Requests |"))
            self.assertNotIn("SQ boundary", lines[0])
            self.assertNotIn("batch=direct", table)
            self.assertNotIn("6720", table)

    def test_markdown_table_with_sq_details_enabled_shows_sq_columns_and_values(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "results.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                format_id="SQ8_0",
                                harness={"class": "cli_model_loop_diagnostic"},
                                sq_projection_boundary="batch",
                                sq_projection_kernel_families="batch=direct",
                                sq_fp8_batch_matvec_count=6720,
                                sq_fp8_expected_all_batch_matvec_count=6720,
                                sq_diagnostic_host_staging_read_count=0,
                                sq_diagnostic_host_staging_write_count=72,
                                sq_diagnostic_host_staging_read_bytes=0,
                                sq_diagnostic_host_staging_write_bytes=1572864,
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b1",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=1,
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            table = TOOL.markdown_table(
                [path], "pp16-tg8", "", {1, 2}, show_sq_details=True
            )
            lines = table.splitlines()
            self.assertIn(
                "SQ boundary | SQ family | SQ batch | SQ staging ops | SQ staging MiB",
                lines[0],
            )
            self.assertEqual(len(lines), 1 + 1 + 3)

            sq8_line = next(
                line for line in lines if "sq8-mixed-real-batch-no-final-pp16-tg8-b2" in line
            )
            parsed_sq8 = split_markdown_row(sq8_line)
            self.assertEqual(parsed_sq8[-5], "batch")
            self.assertEqual(parsed_sq8[-4], "batch=direct")
            self.assertEqual(parsed_sq8[-3], "6720/6720")
            self.assertEqual(parsed_sq8[-2], "0/72")
            self.assertEqual(parsed_sq8[-1], "0.00/1.50")

            vllm_line = next(
                line for line in lines if "vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2" in line
            )
            parsed_vllm = split_markdown_row(vllm_line)
            self.assertEqual(parsed_vllm[-5], "-")
            self.assertEqual(parsed_vllm[-4], "-")
            self.assertEqual(parsed_vllm[-3], "-")
            self.assertEqual(parsed_vllm[-2], "-")
            self.assertEqual(parsed_vllm[-1], "-")

            missing_sq8_line = next(
                line for line in lines if "sq8-mixed-real-batch-no-final-pp16-tg8-b1" in line
            )
            parsed_missing_sq8 = split_markdown_row(missing_sq8_line)
            self.assertEqual(parsed_missing_sq8[-5], "-")
            self.assertEqual(parsed_missing_sq8[-4], "-")
            self.assertEqual(parsed_missing_sq8[-3], "-")
            self.assertEqual(parsed_missing_sq8[-2], "-")
            self.assertEqual(parsed_missing_sq8[-1], "-")

    def test_cli_show_sq_details_adds_sq_columns(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "results.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                format_id="SQ8_0",
                                sq_projection_boundary="batch",
                                sq_projection_kernel_families="batch=direct",
                                sq_fp8_batch_matvec_count=6720,
                                sq_fp8_expected_all_batch_matvec_count=6720,
                                sq_diagnostic_host_staging_read_count=1,
                                sq_diagnostic_host_staging_write_count=2,
                                sq_diagnostic_host_staging_read_bytes=1024,
                                sq_diagnostic_host_staging_write_bytes=1048576,
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--show-sq-details",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 0)
            self.assertEqual(stderr.getvalue(), "")
            output = stdout.getvalue()
            self.assertIn(
                "SQ boundary | SQ family | SQ batch | SQ staging ops | SQ staging MiB",
                output,
            )
            self.assertIn("batch=direct", output)
            self.assertIn("6720/6720", output)
            self.assertIn("1/2", output)
            self.assertIn("0.00/1.00", output)

    def test_markdown_table_with_sq_details_shows_missing_host_staging_as_question_mark(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "results.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                format_id="SQ8_0",
                                sq_projection_boundary="batch",
                                sq_projection_kernel_families="batch=direct",
                                sq_fp8_batch_matvec_count=6720,
                                sq_fp8_expected_all_batch_matvec_count=6720,
                                sq_diagnostic_host_staging_read_count=12,
                                sq_diagnostic_host_staging_write_bytes=1048576,
                            )
                        )
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            table = TOOL.markdown_table(
                [path], "pp16-tg8", "", {2}, show_sq_details=True
            )
            lines = table.splitlines()

            self.assertIn(
                "SQ boundary | SQ family | SQ batch | SQ staging ops | SQ staging MiB",
                lines[0],
            )
            sq8_line = next(
                line for line in lines if "sq8-pp16-tg8-b2" in line
            )
            parsed_sq8 = split_markdown_row(sq8_line)
            self.assertEqual(parsed_sq8[-5], "batch")
            self.assertEqual(parsed_sq8[-4], "batch=direct")
            self.assertEqual(parsed_sq8[-3], "6720/6720")
            self.assertEqual(parsed_sq8[-2], "12/?")
            self.assertEqual(parsed_sq8[-1], "?/1.00")

    def test_parse_requests_filter_rejects_bad_items(self) -> None:
        self.assertEqual(TOOL.parse_requests_filter("2, 4,8"), {2, 4, 8})
        with self.assertRaises(ValueError):
            TOOL.parse_requests_filter("2,,8")
        with self.assertRaises(ValueError):
            TOOL.parse_requests_filter("0")

    def test_parse_required_engines_filter(self) -> None:
        self.assertEqual(
            TOOL.parse_required_engines_filter("uLLM,vLLM"), {"uLLM", "vLLM"}
        )
        with self.assertRaises(ValueError):
            TOOL.parse_required_engines_filter("uLLM,,vLLM")
        with self.assertRaises(ValueError):
            TOOL.parse_required_engines_filter("uLLM,")

    def test_invalid_json_reports_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "results.jsonl"
            path.write_text('{"case_id":"ok"}\n{bad-json}\n', encoding="utf-8")
            with self.assertRaises(ValueError) as cm:
                TOOL.markdown_table([path], "", "")
        self.assertIn("results.jsonl:2", str(cm.exception))

    def test_require_serving_parity_fails_for_mixed_harness_rows(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "mixed.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="qwen3-14b-sq8-full-mixed-real-batch-no-final-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                # legacy row: no harness object, infer via case_id
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2-tp1-rocr",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = TOOL.selected_rows([path], "pp16-tg8", "", {2, 4, 8})
            self.assertTrue(TOOL.serving_parity_gate_failures(rows))
            self.assertIn(
                "mixed harness.class values",
                "\n".join(TOOL.serving_parity_gate_failures(rows)),
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2,4,8",
                    "--require-serving-parity",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)

    def test_require_serving_parity_fails_for_ullm_serving_candidate_parity_blockers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "parity_blockers.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="qwen3-14b-sq8-full-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={
                                    "class": "serving_throughput_benchmark",
                                    "serving_parity_candidate": True,
                                    "ullm_serving_candidate": {
                                        "parity_blockers": ["runner_known_gap"]
                                    },
                                },
                            )
                        )
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = TOOL.selected_rows([path], "pp16-tg8", "", {2, 4, 8})
            failure_lines = TOOL.serving_parity_gate_failures(rows)
            self.assertTrue(failure_lines)
            self.assertIn(
                "selected rows include uLLM serving candidate parity blockers",
                "\n".join(failure_lines),
            )
            self.assertIn("runner_known_gap", "\n".join(failure_lines))

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2,4,8",
                    "--require-serving-parity",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("runner_known_gap", stderr.getvalue())

    def test_require_serving_parity_and_engines_fails_when_required_engine_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "serving_filter.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2-tp1-rocr",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2,4,8",
                    "--harness-class",
                    "serving_throughput_benchmark",
                    "--require-serving-parity",
                    "--require-engines",
                    "uLLM,vLLM",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("missing required engine(s): uLLM", stderr.getvalue())

    def test_require_engines_only_fails_when_required_engine_missing(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "vllm_only.jsonl"
            path.write_text(
                json.dumps(
                    make_row(
                        case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2-tp1-rocr",
                        engine_name="vLLM",
                        prompt_tokens=16,
                        generated_tokens=8,
                        batch_size=2,
                        harness={"class": "serving_throughput_benchmark"},
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-engines",
                    "uLLM,vLLM",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("vLLM | vllm-r9700", stdout.getvalue())
            self.assertIn("missing required engine(s): uLLM", stderr.getvalue())

    def test_require_engines_grid_fails_when_any_request_count_missing_required_engine(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "grid_missing.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2-tp1-rocr",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b4-tp1-rocr",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=4,
                                harness={"class": "serving_throughput_benchmark"},
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2,4,8",
                    "--require-engines",
                    "uLLM,vLLM",
                    "--require-engine-grid",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn(
                "request 4 missing required engine(s): uLLM",
                stderr.getvalue(),
            )
            self.assertIn(
                "request 8 missing required engine(s): uLLM, vLLM",
                stderr.getvalue(),
            )

    def test_require_normalized_throughput_comparison_passes_for_matching_request_shape(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "normalized_match.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "cli_model_loop_diagnostic"},
                                format_id="SQ8_0",
                                batching={
                                    "mode": "real",
                                    "prefill_real_batch": True,
                                    "decode_real_batch": True,
                                    "final_logits_in_total": False,
                                },
                                sq_projection_boundary="batch",
                                sq_fp8_batch_matvec_count=14,
                                sq_fp8_expected_all_batch_matvec_count=14,
                                context_length=1048576,
                                model_name="Qwen3.5-9B",
                                result_validity={
                                    "state": "valid",
                                    "classification": "optimized_component",
                                    "implementation_valid": True,
                                    "quality_comparison_valid": True,
                                    "performance_comparison_valid": True,
                                    "artifact_manifest_sha256": "a" * 64,
                                    "reason_codes": [],
                                },
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                                context_length=256,
                                model_name="Qwen3.5-9B",
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-normalized-throughput-comparison",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 0)
            self.assertEqual(stderr.getvalue(), "")

    def test_normalized_comparison_rejects_quarantined_ullm_row(self) -> None:
        ullm_row = make_row(
            case_id="sq8-pp16-tg8-b2",
            engine_name="uLLM",
            prompt_tokens=16,
            generated_tokens=8,
            batch_size=2,
            harness={"class": "cli_model_loop_diagnostic"},
            format_id="SQ8_0",
            batching={
                "mode": "real",
                "prefill_real_batch": True,
                "decode_real_batch": True,
                "final_logits_in_total": False,
            },
            sq_projection_boundary="batch",
            sq_fp8_batch_matvec_count=14,
            sq_fp8_expected_all_batch_matvec_count=14,
            context_length=1024,
            model_name="Qwen3-14B-FP8",
            result_validity={
                "state": "quarantined",
                "classification": "connection_diagnostic",
                "implementation_valid": False,
                "quality_comparison_valid": False,
                "performance_comparison_valid": False,
                "reason_codes": ["source_fp8_weight_scale_inv_not_applied"],
            },
        )
        vllm_row = make_row(
            case_id="vllm-pp16-tg8-b2",
            engine_name="vLLM",
            prompt_tokens=16,
            generated_tokens=8,
            batch_size=2,
            harness={"class": "serving_throughput_benchmark"},
            context_length=1024,
            model_name="Qwen3-14B-FP8",
        )

        failures = TOOL.normalized_throughput_comparison_gate_failures(
            [ullm_row, vllm_row],
            {2},
        )

        self.assertTrue(
            any("not implementation-valid" in failure for failure in failures),
            failures,
        )
        self.assertTrue(
            any("not performance-comparison-valid" in failure for failure in failures),
            failures,
        )
        self.assertTrue(
            any("source_fp8_weight_scale_inv_not_applied" in failure for failure in failures),
            failures,
        )

    def test_implementation_validity_gate_requires_exact_true_and_manifest_hash(self) -> None:
        row = make_row(
            case_id="sq8-component",
            engine_name="uLLM",
            prompt_tokens=1,
            generated_tokens=0,
            batch_size=1,
            format_id="SQ8_0",
            result_validity={
                "state": "valid",
                "classification": "source_correct_reference",
                "implementation_valid": True,
                "quality_comparison_valid": True,
                "performance_comparison_valid": False,
                "artifact_manifest_sha256": "b" * 64,
                "reason_codes": [],
            },
        )

        self.assertEqual(
            TOOL.ullm_sq_result_validity_gate_failures(
                [row],
                require_performance_comparison=False,
            ),
            [],
        )
        performance_failures = TOOL.ullm_sq_result_validity_gate_failures(
            [row],
            require_performance_comparison=True,
        )
        self.assertTrue(
            any("not performance-comparison-valid" in failure for failure in performance_failures)
        )

        row["result_validity"]["implementation_valid"] = "true"
        row["result_validity"]["artifact_manifest_sha256"] = "not-a-hash"
        implementation_failures = TOOL.ullm_sq_result_validity_gate_failures(
            [row],
            require_performance_comparison=False,
        )
        self.assertTrue(
            any("not implementation-valid" in failure for failure in implementation_failures)
        )

    def test_require_normalized_throughput_comparison_fails_when_context_length_is_too_small(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "normalized_context_length_too_small.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "cli_model_loop_diagnostic"},
                                format_id="SQ8_0",
                                batching={
                                    "mode": "real",
                                    "prefill_real_batch": True,
                                    "decode_real_batch": True,
                                    "final_logits_in_total": False,
                                },
                                sq_projection_boundary="batch",
                                sq_fp8_batch_matvec_count=14,
                                sq_fp8_expected_all_batch_matvec_count=14,
                                context_length=1024,
                                model_name="Qwen3.5-9B",
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                                context_length=10,
                                model_name="Qwen3.5-9B",
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-normalized-throughput-comparison",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn(
                "request 2 case_id=vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2: "
                "workload.context_length=10 is smaller than prompt+generated=12",
                stderr.getvalue(),
            )
            self.assertIn("prompt+generated=12", stderr.getvalue())

    def test_require_normalized_throughput_comparison_fails_when_context_length_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "normalized_context_length_missing.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "cli_model_loop_diagnostic"},
                                format_id="SQ8_0",
                                batching={
                                    "mode": "real",
                                    "prefill_real_batch": True,
                                    "decode_real_batch": True,
                                    "final_logits_in_total": False,
                                },
                                sq_projection_boundary="batch",
                                sq_fp8_batch_matvec_count=14,
                                sq_fp8_expected_all_batch_matvec_count=14,
                                model_name="Qwen3.5-9B",
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                                context_length=256,
                                model_name="Qwen3.5-9B",
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-normalized-throughput-comparison",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn(
                "request 2 case_id=sq8-pp16-tg8-b2: "
                "workload.context_length is missing or malformed",
                stderr.getvalue(),
            )

    def test_require_normalized_throughput_comparison_fails_when_per_request_prompt_generated_shape_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "normalized_shape_mismatch.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-pp24-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=24,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "cli_model_loop_diagnostic"},
                                format_id="SQ8_0",
                                batching={
                                    "mode": "real",
                                    "prefill_real_batch": True,
                                    "decode_real_batch": True,
                                    "final_logits_in_total": False,
                                },
                                sq_projection_boundary="batch",
                                sq_fp8_batch_matvec_count=14,
                                sq_fp8_expected_all_batch_matvec_count=14,
                                context_length=1048576,
                                model_name="Qwen3.5-9B",
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp24-tg8-b2",
                                engine_name="vLLM",
                                prompt_tokens=24,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                                prompt_tokens_per_request=[9, 9],
                                generated_tokens_per_request=[4, 4],
                                context_length=256,
                                model_name="Qwen3.5-9B",
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp24-tg8",
                    "--requests",
                    "2",
                    "--require-normalized-throughput-comparison",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn(
                "request 2 per-request prompt/generated shape mismatch",
                stderr.getvalue(),
            )

    def test_require_normalized_throughput_comparison_fails_when_model_name_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "normalized_model_name_mismatch.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "cli_model_loop_diagnostic"},
                                format_id="SQ8_0",
                                batching={
                                    "mode": "real",
                                    "prefill_real_batch": True,
                                    "decode_real_batch": True,
                                    "final_logits_in_total": False,
                                },
                                sq_projection_boundary="batch",
                                sq_fp8_batch_matvec_count=14,
                                sq_fp8_expected_all_batch_matvec_count=14,
                                context_length=1048576,
                                model_name="Qwen3.5-9B",
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                                context_length=256,
                                model_name="Qwen3-14B-FP8",
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-normalized-throughput-comparison",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("model.name mismatch", stderr.getvalue())
            self.assertIn("uLLM=Qwen3.5-9B", stderr.getvalue())

    def test_require_normalized_throughput_comparison_fails_when_model_name_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "normalized_model_name_missing.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "cli_model_loop_diagnostic"},
                                format_id="SQ8_0",
                                batching={
                                    "mode": "real",
                                    "prefill_real_batch": True,
                                    "decode_real_batch": True,
                                    "final_logits_in_total": False,
                                },
                                sq_projection_boundary="batch",
                                sq_fp8_batch_matvec_count=14,
                                sq_fp8_expected_all_batch_matvec_count=14,
                                context_length=1048576,
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                                context_length=256,
                                model_name="Qwen3-14B-FP8",
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-normalized-throughput-comparison",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("missing model.name", stderr.getvalue())
            self.assertIn("uLLM=-", stderr.getvalue())

    def test_require_normalized_throughput_comparison_fails_when_missing_ullm_or_vllm_row(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            missing_ullm_path = Path(workdir) / "missing_ullm.jsonl"
            missing_ullm_path.write_text(
                json.dumps(
                    make_row(
                        case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2",
                        engine_name="vLLM",
                        prompt_tokens=16,
                        generated_tokens=8,
                        batch_size=2,
                        harness={"class": "serving_throughput_benchmark"},
                        context_length=256,
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(missing_ullm_path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-normalized-throughput-comparison",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("request 2 missing required uLLM", stderr.getvalue())

            missing_vllm_path = Path(workdir) / "missing_vllm.jsonl"
            missing_vllm_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "cli_model_loop_diagnostic"},
                                format_id="SQ8_0",
                                batching={
                                    "mode": "real",
                                    "prefill_real_batch": True,
                                    "decode_real_batch": True,
                                    "final_logits_in_total": False,
                                },
                                sq_projection_boundary="batch",
                                sq_fp8_batch_matvec_count=14,
                                sq_fp8_expected_all_batch_matvec_count=14,
                                context_length=1048576,
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(missing_vllm_path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-normalized-throughput-comparison",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("request 2 missing required vLLM", stderr.getvalue())

    def test_require_normalized_throughput_comparison_fails_when_final_logits_in_total_true(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "invalid_final_logits.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "cli_model_loop_diagnostic"},
                                format_id="SQ8_0",
                                batching={
                                    "mode": "real",
                                    "prefill_real_batch": True,
                                    "decode_real_batch": True,
                                    "final_logits_in_total": True,
                                },
                                sq_projection_boundary="batch",
                                sq_fp8_batch_matvec_count=14,
                                sq_fp8_expected_all_batch_matvec_count=14,
                                context_length=1048576,
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                                context_length=256,
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-normalized-throughput-comparison",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("case_id=sq8-pp16-tg8-b2", stderr.getvalue())
            self.assertIn("final_logits_in_total=false", stderr.getvalue())

    def test_require_normalized_throughput_comparison_fails_when_harness_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            mismatch_rows_path = Path(workdir) / "harness_mismatch.jsonl"
            mismatch_rows_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "logical"},
                                format_id="SQ8_0",
                                batching={
                                    "mode": "real",
                                    "prefill_real_batch": True,
                                    "decode_real_batch": True,
                                    "final_logits_in_total": False,
                                },
                                sq_projection_boundary="batch",
                                sq_fp8_batch_matvec_count=14,
                                sq_fp8_expected_all_batch_matvec_count=14,
                                context_length=1048576,
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                                context_length=256,
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(mismatch_rows_path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-normalized-throughput-comparison",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("case_id=sq8-pp16-tg8-b2", stderr.getvalue())
            self.assertIn("cli_model_loop_diagnostic", stderr.getvalue())

            mismatch_rows_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "cli_model_loop_diagnostic"},
                                format_id="SQ8_0",
                                batching={
                                    "mode": "real",
                                    "prefill_real_batch": True,
                                    "decode_real_batch": True,
                                    "final_logits_in_total": False,
                                },
                                sq_projection_boundary="batch",
                                sq_fp8_batch_matvec_count=14,
                                sq_fp8_expected_all_batch_matvec_count=14,
                                context_length=1048576,
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "cli_model_loop_diagnostic"},
                                context_length=256,
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(mismatch_rows_path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-normalized-throughput-comparison",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("case_id=vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2", stderr.getvalue())
            self.assertIn("serving_throughput_benchmark", stderr.getvalue())

    def test_require_engines_grid_passes_when_all_request_counts_cover_required_engines(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "grid_pass.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2-tp1-rocr",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b4-tp1-rocr",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=4,
                                harness={"class": "serving_throughput_benchmark"},
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b4",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=4,
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2,4",
                    "--require-engines",
                    "uLLM,vLLM",
                    "--require-engine-grid",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 0)
            self.assertEqual(stderr.getvalue(), "")

    def test_require_engine_grid_without_require_engines_fails(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "grid_without_engines.jsonl"
            path.write_text(
                json.dumps(
                    make_row(
                        case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2-tp1-rocr",
                        engine_name="vLLM",
                        prompt_tokens=16,
                        generated_tokens=8,
                        batch_size=2,
                        harness={"class": "serving_throughput_benchmark"},
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--require-engine-grid",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 1)
            self.assertIn(
                "--require-engine-grid requires --require-engines to be specified",
                stderr.getvalue(),
            )

    def test_require_serving_parity_and_engines_passes_when_all_required_present(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "serving_both.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2-tp1-rocr",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="sq8-qwen3-14b-sq8-full-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2,4,8",
                    "--harness-class",
                    "serving_throughput_benchmark",
                    "--require-serving-parity",
                    "--require-engines",
                    "uLLM,vLLM",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 0)
            self.assertEqual(stderr.getvalue(), "")

    def test_require_serving_parity_passes_for_serving_only_rows(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "serving.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2-tp1-rocr",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b4-tp1-rocr",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=4,
                                harness={"class": "serving_throughput_benchmark"},
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = TOOL.selected_rows([path], "pp16-tg8", "", {2, 4, 8})
            self.assertEqual(TOOL.serving_parity_gate_failures(rows), [])
            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2,4,8",
                    "--require-serving-parity",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 0)

    def test_harness_class_filter_prefers_vllm_serving_rows_and_passes_parity_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "mixed.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-pp16-tg8-b2-tp1-rocr",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "serving_throughput_benchmark"},
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-pp16-tg8-b2-legacy",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                harness={"class": "cli_model_loop_diagnostic"},
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = TOOL.selected_rows(
                [path],
                "pp16-tg8",
                "",
                {2, 4, 8},
                "serving_throughput_benchmark",
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["case_id"],
                "vllm-r9700-qwen3-14b-fp8-pp16-tg8-b2-tp1-rocr",
            )

            streamed_rows = list(
                TOOL.iter_selected_rows(
                    [path],
                    "pp16-tg8",
                    "",
                    {2, 4, 8},
                    "serving_throughput_benchmark",
                )
            )
            self.assertEqual(len(streamed_rows), 1)

            table = TOOL.markdown_table(
                [path],
                "pp16-tg8",
                "",
                {2, 4, 8},
                "serving_throughput_benchmark",
            )
            self.assertEqual(len(table.splitlines()), 3)
            self.assertIn(
                "vllm-r9700-qwen3-14b-fp8-pp16-tg8-b2-tp1-rocr",
                table,
            )
            self.assertNotIn("mixed-real-batch-no-final", table)
            self.assertNotIn("legacy", table)

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2,4,8",
                    "--harness-class",
                    "serving_throughput_benchmark",
                    "--require-serving-parity",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 0)
            self.assertEqual(stderr.getvalue(), "")
            out = stdout.getvalue()
            self.assertIn(
                "vLLM | vllm-r9700-qwen3-14b-fp8-pp16-tg8-b2-tp1-rocr",
                out,
            )
            self.assertNotIn("mixed-real-batch-no-final", out)
            self.assertNotIn("legacy", out)

    def test_require_serving_parity_fails_when_no_rows_selected(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "nomatch.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b1",
                                engine_name="vLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=1,
                            )
                        )
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = TOOL.selected_rows([path], "pp16-tg8", "", {2, 4, 8})
            self.assertEqual(
                TOOL.serving_parity_gate_failures(rows), ["no selected rows"]
            )
            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2,4,8",
                    "--require-serving-parity",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)

    def test_require_ullm_sq_kernel_families_fails_when_field_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "missing_kernel_families.jsonl"
            path.write_text(
                json.dumps(
                    make_row(
                        case_id="sq8-qwen3-14b-sq8-smoke-pp16-tg8-b2",
                        engine_name="uLLM",
                        prompt_tokens=16,
                        generated_tokens=8,
                        batch_size=2,
                        format_id="SQ8_0",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-ullm-sq-kernel-families",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("sq_projection_kernel_families", stderr.getvalue())
            self.assertIn(
                "case_id=sq8-qwen3-14b-sq8-smoke-pp16-tg8-b2",
                stderr.getvalue(),
            )

    def test_require_ullm_sq_kernel_families_fails_when_value_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "none_kernel_families.jsonl"
            path.write_text(
                json.dumps(
                    make_row(
                        case_id="sq8-qwen3-14b-sq8-smoke-pp16-tg8-b2",
                        engine_name="uLLM",
                        prompt_tokens=16,
                        generated_tokens=8,
                        batch_size=2,
                        format_id="SQ8_0",
                        sq_projection_kernel_families="none",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-ullm-sq-kernel-families",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("sq_projection_kernel_families", stderr.getvalue())
            self.assertIn(
                "case_id=sq8-qwen3-14b-sq8-smoke-pp16-tg8-b2",
                stderr.getvalue(),
            )

    def test_require_ullm_sq_kernel_families_fails_when_value_is_batch_none(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "batch_none_kernel_families.jsonl"
            path.write_text(
                json.dumps(
                    make_row(
                        case_id="sq8-qwen3-14b-sq8-smoke-pp16-tg8-b2",
                        engine_name="uLLM",
                        prompt_tokens=16,
                        generated_tokens=8,
                        batch_size=2,
                        format_id="SQ8_0",
                        sq_projection_kernel_families="batch=none",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-ullm-sq-kernel-families",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("sq_projection_kernel_families", stderr.getvalue())
            self.assertIn(
                "case_id=sq8-qwen3-14b-sq8-smoke-pp16-tg8-b2",
                stderr.getvalue(),
            )

    def test_require_ullm_sq_kernel_families_fails_when_value_is_missing_family(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "missing_family_kernel_families.jsonl"
            path.write_text(
                json.dumps(
                    make_row(
                        case_id="sq8-qwen3-14b-sq8-smoke-pp16-tg8-b2",
                        engine_name="uLLM",
                        prompt_tokens=16,
                        generated_tokens=8,
                        batch_size=2,
                        format_id="SQ8_0",
                        sq_projection_kernel_families="batch=",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-ullm-sq-kernel-families",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("sq_projection_kernel_families", stderr.getvalue())
            self.assertIn(
                "case_id=sq8-qwen3-14b-sq8-smoke-pp16-tg8-b2",
                stderr.getvalue(),
            )

    def test_require_ullm_sq_kernel_families_fails_when_value_is_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "malformed_kernel_families.jsonl"
            path.write_text(
                json.dumps(
                    make_row(
                        case_id="sq8-qwen3-14b-sq8-smoke-pp16-tg8-b2",
                        engine_name="uLLM",
                        prompt_tokens=16,
                        generated_tokens=8,
                        batch_size=2,
                        format_id="SQ8_0",
                        sq_projection_kernel_families="batch",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-ullm-sq-kernel-families",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("sq_projection_kernel_families", stderr.getvalue())
            self.assertIn(
                "case_id=sq8-qwen3-14b-sq8-smoke-pp16-tg8-b2",
                stderr.getvalue(),
            )

    def test_require_ullm_sq_kernel_families_passes_when_valid(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "valid_kernel_families.jsonl"
            path.write_text(
                json.dumps(
                    make_row(
                        case_id="sq8-qwen3-14b-sq8-smoke-pp16-tg8-b2",
                        engine_name="uLLM",
                        prompt_tokens=16,
                        generated_tokens=8,
                        batch_size=2,
                        format_id="SQ8_0",
                        sq_projection_kernel_families="batch=direct",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-ullm-sq-kernel-families",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 0)
            self.assertEqual(stderr.getvalue(), "")

    def test_require_ullm_sq_kernel_families_passes_with_future_family_names(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "future_family_kernel_families.jsonl"
            path.write_text(
                json.dumps(
                    make_row(
                        case_id="sq8-qwen3-14b-sq8-smoke-pp16-tg8-b2",
                        engine_name="uLLM",
                        prompt_tokens=16,
                        generated_tokens=8,
                        batch_size=2,
                        format_id="SQ8_0",
                        sq_projection_kernel_families="single=direct,batch=fused_v0",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-ullm-sq-kernel-families",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 0)
            self.assertEqual(stderr.getvalue(), "")


    def test_require_ullm_sq_batch_coverage_fails_when_boundary_missing_or_none(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "missing_boundary.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                format_id="SQ8_0",
                                sq_fp8_batch_matvec_count=14,
                                sq_fp8_expected_all_batch_matvec_count=14,
                            )
                        ),
                        json.dumps(
                            make_row(
                                case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2-none",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                format_id="SQ8_0",
                                sq_projection_boundary="none",
                                sq_fp8_batch_matvec_count=14,
                                sq_fp8_expected_all_batch_matvec_count=14,
                            )
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-ullm-sq-batch-coverage",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("sq_projection_boundary", stderr.getvalue())
            self.assertIn(
                "sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                stderr.getvalue(),
            )

    def test_require_ullm_sq_batch_coverage_fails_when_counts_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "insufficient_batch_counts.jsonl"
            path.write_text(
                json.dumps(
                    make_row(
                        case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                        engine_name="uLLM",
                        prompt_tokens=16,
                        generated_tokens=8,
                        batch_size=2,
                        format_id="SQ8_0",
                        sq_projection_boundary="batch",
                        sq_fp8_batch_matvec_count=9,
                        sq_fp8_expected_all_batch_matvec_count=14,
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-ullm-sq-batch-coverage",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn(
                "lacks full batch projection coverage",
                stderr.getvalue(),
            )
            self.assertIn(
                "sq_fp8_batch_matvec_count=9 sq_fp8_expected_all_batch_matvec_count=14",
                stderr.getvalue(),
            )

    def test_require_ullm_sq_batch_coverage_passes_when_batch_boundary_and_counts_ok(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "valid_batch_coverage.jsonl"
            path.write_text(
                json.dumps(
                    make_row(
                        case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                        engine_name="uLLM",
                        prompt_tokens=16,
                        generated_tokens=8,
                        batch_size=2,
                        format_id="SQ8_0",
                        sq_projection_boundary="single+batch",
                        sq_fp8_batch_matvec_count=21,
                        sq_fp8_expected_all_batch_matvec_count=14,
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-ullm-sq-batch-coverage",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 0)
            self.assertEqual(stderr.getvalue(), "")


    def test_require_ullm_sq_no_host_staging_fails_when_non_zero_read_count(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "non_zero_host_staging.jsonl"
            path.write_text(
                json.dumps(
                    make_row(
                        case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                        engine_name="uLLM",
                        prompt_tokens=16,
                        generated_tokens=8,
                        batch_size=2,
                        format_id="SQ8_0",
                        sq_diagnostic_host_staging_read_count=1,
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-ullm-sq-no-host-staging",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("sq_diagnostic_host_staging_read_count", stderr.getvalue())

    def test_require_ullm_sq_host_staging_write_count_passes_within_limit(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "write_count_within_limit.jsonl"
            path.write_text(
                json.dumps(
                    make_row(
                        case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                        engine_name="uLLM",
                        prompt_tokens=16,
                        generated_tokens=8,
                        batch_size=2,
                        format_id="SQ8_0",
                        sq_diagnostic_host_staging_write_count=24,
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--max-ullm-sq-host-staging-write-count",
                    "24",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 0)
            self.assertEqual(stderr.getvalue(), "")

    def test_require_ullm_sq_host_staging_write_count_fails_when_over_limit(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "write_count_over_limit.jsonl"
            path.write_text(
                json.dumps(
                    make_row(
                        case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                        engine_name="uLLM",
                        prompt_tokens=16,
                        generated_tokens=8,
                        batch_size=2,
                        format_id="SQ8_0",
                        sq_diagnostic_host_staging_write_count=25,
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--max-ullm-sq-host-staging-write-count",
                    "24",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("sq_diagnostic_host_staging_write_count=25", stderr.getvalue())
            self.assertIn("max=24", stderr.getvalue())

    def test_require_ullm_sq_no_host_staging_passes_when_all_zero(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "zero_host_staging.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_row(
                                case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                                engine_name="uLLM",
                                prompt_tokens=16,
                                generated_tokens=8,
                                batch_size=2,
                                format_id="SQ8_0",
                                sq_diagnostic_host_staging_read_count=0,
                                sq_diagnostic_host_staging_write_count=0,
                                sq_diagnostic_host_staging_read_bytes=0,
                                sq_diagnostic_host_staging_write_bytes=0,
                            )
                        )
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-ullm-sq-no-host-staging",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 0)
            self.assertEqual(stderr.getvalue(), "")

    def test_require_ullm_sq_no_host_staging_fails_when_missing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "missing_host_staging.jsonl"
            path.write_text(
                json.dumps(
                    make_row(
                        case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                        engine_name="uLLM",
                        prompt_tokens=16,
                        generated_tokens=8,
                        batch_size=2,
                        format_id="SQ8_0",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-ullm-sq-no-host-staging",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("missing host staging metric", stderr.getvalue())
            self.assertIn("sq_diagnostic_host_staging_read_count", stderr.getvalue())
            self.assertIn("sq_diagnostic_host_staging_write_count", stderr.getvalue())
            self.assertIn("sq_diagnostic_host_staging_read_bytes", stderr.getvalue())
            self.assertIn("sq_diagnostic_host_staging_write_bytes", stderr.getvalue())

    def test_require_ullm_sq_no_host_staging_fails_when_metric_is_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "malformed_host_staging.jsonl"
            row = make_row(
                case_id="sq8-mixed-real-batch-no-final-pp16-tg8-b2",
                engine_name="uLLM",
                prompt_tokens=16,
                generated_tokens=8,
                batch_size=2,
                format_id="SQ8_0",
            )
            row["workload"]["sq_diagnostic_host_staging_write_bytes"] = "unknown"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with patch.object(
                sys,
                "argv",
                [
                    "summarize.py",
                    str(path),
                    "--workload-prefix",
                    "pp16-tg8",
                    "--requests",
                    "2",
                    "--require-ullm-sq-no-host-staging",
                ],
            ):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = TOOL.main()
            self.assertEqual(status, 2)
            self.assertIn("malformed host staging metric", stderr.getvalue())
            self.assertIn("sq_diagnostic_host_staging_write_bytes=unknown", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
