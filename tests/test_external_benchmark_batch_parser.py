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


TOOL = load_tool("run-external-benchmark.py")


class ExternalBenchmarkBatchParserTests(unittest.TestCase):
    def test_preserves_batch_throughput_accounting_executor_and_kv_memory(self) -> None:
        report = {
            "top_k": 8,
            "workload": {
                "batch_size": 2,
                "concurrent_requests": 2,
                "prefill_mode": "cold",
                "prompt_tokens_per_request": [4, 4],
                "cached_prefix_tokens_per_request": [0, 0],
                "new_prefill_tokens_per_request": [4, 4],
                "total_context_tokens_after_prefill_per_request": [4, 4],
                "generated_tokens_per_request": [2, 2],
                "fixed_decode_steps": True,
            },
            "batching": {
                "mode": "logical",
                "prefill_executor": "cached_prefix_rdna4_fp8_auto",
                "resolved_prefill_executor": "cached_prefix_flash2_fp8q",
                "prefill_real_batch": False,
                "decode_executor": "sequential_package_token_ids_generate",
            },
            "metrics": {
                "prefill_total_input_tokens": 8,
                "cached_prefix_total_tokens": 0,
                "total_context_tokens_after_prefill": 8,
                "estimated_prefill_attention_work_tokens": 20,
                "decode_total_generated_tokens": 2,
                "generated_tokens_total": 4,
                "end_to_end_total_tokens": 12,
                "prefill_wall_ms_sum": 80.0,
                "decode_wall_ms_sum": 20.0,
                "batch_wall_ms": 150.0,
                "prefill_total_input_tps": 100.0,
                "decode_total_generated_tps": 100.0,
                "end_to_end_total_tps": 80.0,
                "per_request_decode_tps_mean": 90.0,
                "time_to_first_token_ms_p50": 40.0,
                "time_to_first_token_ms_p95": 70.0,
                "request_latency_ms_p50": 75.0,
                "request_latency_ms_p95": 100.0,
                "time_per_output_token_ms_p50": 10.0,
                "time_per_output_token_ms_p95": 12.0,
            },
            "memory": {
                "vram_baseline_bytes": None,
                "vram_peak_bytes": None,
                "vram_consumed_bytes": None,
                "kv_cache_bytes_total": 98304,
            },
            "correctness": {
                "verified_all": True,
            },
            "verified": True,
        }
        memory = {
            "baseline_total_bytes": 1000,
            "peak_total_bytes": 2000,
            "consumed_total_bytes": 1000,
        }

        metrics = TOOL.parse_ullm_batch_throughput_metrics(report, memory)
        row = {
            "workload": {
                "batch_size": 1,
                "concurrent_requests": 1,
                "kv_cache_dtype": "f32",
                "prefill_executor": None,
                "resolved_prefill_executor": None,
            },
            "metrics": metrics,
            "memory": memory.copy(),
        }
        TOOL.enrich_ullm_batch_workload(row, report)
        TOOL.enrich_ullm_batch_memory(row, report)
        correctness = TOOL.parse_ullm_batch_throughput_correctness(report)

        self.assertEqual(metrics["prefill_total_input_tokens"], 8)
        self.assertEqual(metrics["decode_total_generated_tokens"], 2)
        self.assertEqual(metrics["end_to_end_total_tokens"], 12)
        self.assertEqual(metrics["prefill_total_input_tokens_per_second"], 100.0)
        self.assertEqual(metrics["decode_total_generated_tokens_per_second"], 100.0)
        self.assertEqual(metrics["end_to_end_total_tokens_per_second"], 80.0)
        self.assertEqual(metrics["prefill_wall_time_seconds"], 0.08)
        self.assertEqual(metrics["decode_wall_time_seconds"], 0.02)
        self.assertEqual(metrics["total_wall_time_seconds"], 0.15)

        self.assertEqual(row["workload"]["batch_size"], 2)
        self.assertEqual(row["workload"]["concurrent_requests"], 2)
        self.assertEqual(row["workload"]["prefill_mode"], "cold")
        self.assertEqual(row["workload"]["prompt_tokens_per_request"], [4, 4])
        self.assertEqual(row["workload"]["cached_prefix_tokens_per_request"], [0, 0])
        self.assertEqual(row["workload"]["new_prefill_tokens_per_request"], [4, 4])
        self.assertEqual(row["workload"]["total_context_tokens_after_prefill_per_request"], [4, 4])
        self.assertEqual(row["workload"]["generated_tokens_per_request"], [2, 2])
        self.assertEqual(row["workload"]["cached_prefix_total_tokens"], 0)
        self.assertEqual(row["workload"]["total_context_tokens_after_prefill"], 8)
        self.assertEqual(row["workload"]["estimated_prefill_attention_work_tokens"], 20)
        self.assertEqual(row["workload"]["prefill_executor"], "cached_prefix_rdna4_fp8_auto")
        self.assertEqual(
            row["workload"]["resolved_prefill_executor"],
            "cached_prefix_flash2_fp8q",
        )
        self.assertEqual(row["memory"]["kv_cache_bytes_total"], 98304)
        self.assertIsNotNone(correctness)
        self.assertTrue(correctness["verified_all"])


if __name__ == "__main__":
    unittest.main()
