from __future__ import annotations

import importlib.util
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


TOOL = load_tool("summarize-benchmark-results.py")


def row(case_id: str, *, status: str = "ok", workload: dict | None = None) -> dict:
    return {
        "case_id": case_id,
        "status": status,
        "engine": {"name": "uLLM"},
        "model": {"name": "Qwen3.5-9B", "quantization": "SQ8_0"},
        "hardware": {"gpus": [{"name": "AMD Radeon Graphics"}]},
        "workload": {
            "prompt_tokens": 2,
            "generated_tokens": 1,
            "batch_size": 1,
            **(workload or {}),
        },
        "batching": {"mode": "real"},
        "metrics": {"end_to_end_total_tokens_per_second": 10.0},
        "_source_file": f"{case_id}.jsonl",
    }


class SummarizeBenchmarkResultsTests(unittest.TestCase):
    def test_default_table_excludes_unmarked_materialized_sq_fallback(self) -> None:
        direct = row(
            "direct",
            workload={"sq_execution_mode": "direct_fp8_dequant_matvec"},
        )
        unmarked_fallback = row(
            "unmarked-fallback",
            workload={"sq_execution_mode": "materialized_f32_fallback"},
        )
        allowed_fallback = row(
            "allowed-fallback",
            workload={
                "sq_execution_mode": "materialized_f32_fallback",
                "fallback_allowed": True,
            },
        )

        table = TOOL.markdown_table(
            [unmarked_fallback, direct, allowed_fallback],
            include_failed=False,
        )

        self.assertIn("SQ mode", table)
        self.assertIn("direct.jsonl", table)
        self.assertIn("allowed-fallback.jsonl", table)
        self.assertNotIn("unmarked-fallback.jsonl", table)

    def test_include_failed_table_keeps_unmarked_fallback_for_audit(self) -> None:
        unmarked_fallback = row(
            "unmarked-fallback",
            workload={"sq_execution_mode": "materialized_f32_fallback"},
        )

        table = TOOL.markdown_table([unmarked_fallback], include_failed=True)

        self.assertIn("unmarked-fallback.jsonl", table)
        self.assertIn("materialized_f32_fallback", table)


if __name__ == "__main__":
    unittest.main()
