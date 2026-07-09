from __future__ import annotations

import importlib.util
import json
import tempfile
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
) -> dict:
    return {
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


class SummarizeSq8VllmBatchGridTests(unittest.TestCase):
    def test_table_filter_and_columns_for_pp16_tg8(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "results.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(make_row(case_id="sq8-pp16-tg8-b2", engine_name="uLLM", prompt_tokens=16, generated_tokens=8, batch_size=2)),
                        json.dumps(make_row(case_id="vllm-pp16-tg8-b2", engine_name="vLLM", prompt_tokens=16, generated_tokens=8, batch_size=2)),
                        json.dumps(make_row(case_id="vllm-pp16-tg8-b1", engine_name="vLLM", prompt_tokens=16, generated_tokens=8, batch_size=1)),
                        json.dumps(make_row(case_id="sq8-pp16-tg16-b2", engine_name="uLLM", prompt_tokens=16, generated_tokens=16, batch_size=2)),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            table = TOOL.markdown_table([path], "pp16-tg8", "", {2})
            lines = table.splitlines()

            self.assertTrue(lines[0].startswith("| Engine | Case | Requests |"))
            self.assertIn("uLLM | sq8-pp16-tg8-b2", table)
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

    def test_invalid_json_reports_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "results.jsonl"
            path.write_text('{"case_id":"ok"}\n{bad-json}\n', encoding="utf-8")
            with self.assertRaises(ValueError) as cm:
                TOOL.markdown_table([path], "", "")
            self.assertIn("results.jsonl:2", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
