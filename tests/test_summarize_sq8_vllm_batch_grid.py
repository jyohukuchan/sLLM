from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch
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
    return row


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


if __name__ == "__main__":
    unittest.main()
