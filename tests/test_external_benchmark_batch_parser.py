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

    def test_parses_component_prefill_real_batch_key_value_output(self) -> None:
        stdout = (
            'runtime-causal-attn-batch-smoke backend=hip device_index=2 '
            'name="AMD Radeon Graphics" prefill_mode=cold '
            'executor=causal_attn_batch_f32_flash2 batching_mode=real '
            'batch_count=2 concurrent_requests=2 prompt_tokens_per_request=32 '
            'prefill_total_input_tokens=64 q_heads=4 kv_heads=1 head_dim=16 value_dim=16 '
            'estimated_prefill_attention_work_tokens=1056 measured_repeats=1 '
            'wall_ms_mean=0.076391 wall_ms_min=0.076391 wall_ms_max=0.076391 '
            'prefill_total_input_tps=837795.028210 attention_pair_tps_mean=13823617.965467 '
            'request_parallelism=2 token_parallelism=32 verification=sampled sample_count=30 '
            'sampled_max_abs_diff=0.000000008 verified=true'
        )
        memory = {
            "baseline_total_bytes": 1000,
            "peak_total_bytes": 2000,
            "consumed_total_bytes": 1000,
        }

        report = TOOL.parse_key_value_stdout(stdout)
        metrics = TOOL.parse_ullm_component_prefill_metrics(report, memory)
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
        TOOL.enrich_ullm_component_prefill_row(row, report)

        self.assertEqual(report["command"], "runtime-causal-attn-batch-smoke")
        self.assertEqual(report["name"], "AMD Radeon Graphics")
        self.assertTrue(report["verified"])
        self.assertEqual(metrics["prefill_total_input_tokens"], 64)
        self.assertEqual(metrics["prefill_total_input_tokens_per_second"], 837795.028210)
        self.assertEqual(metrics["attention_pair_tps_mean"], 13823617.965467)
        self.assertEqual(metrics["decode_total_generated_tokens"], 0)
        self.assertEqual(row["workload"]["batch_size"], 2)
        self.assertEqual(row["workload"]["concurrent_requests"], 2)
        self.assertEqual(row["workload"]["prefill_mode"], "cold")
        self.assertEqual(row["workload"]["prompt_tokens_per_request"], [32, 32])
        self.assertEqual(row["workload"]["new_prefill_tokens_per_request"], [32, 32])
        self.assertEqual(row["workload"]["estimated_prefill_attention_work_tokens"], 1056)
        self.assertEqual(row["batching"]["mode"], "real")
        self.assertTrue(row["batching"]["prefill_real_batch"])
        self.assertEqual(row["batching"]["prefill_executor_request_parallelism"], 2)
        self.assertEqual(row["batching"]["prefill_executor_token_parallelism"], 32)

    def test_parses_package_prefill_component_real_batch_key_value_output(self) -> None:
        stdout = (
            "package-prefill-aq4-matvec-batch-smoke "
            "package=/tmp/model.ullm.d "
            'tensor="model.language_model.layers.3.self_attn.k_proj.weight" '
            "prompt_tokens=2 hidden=4096 rows=1024 cols=4096 "
            "input_elements=8192 output_elements=2048 "
            "executor=aq4_matvec_batch_f32 real_batch=true token_parallelism=2 "
            "request_parallelism=1 backend=hip device_index=2 "
            'name="AMD Radeon Graphics" warmup_runs=1 measured_repeats=1 '
            "wall_ms_mean=0.104112 wall_ms_min=0.104112 wall_ms_max=0.104112 "
            "token_tps_mean=19210.081451 element_tps_mean=19671123.405563 "
            "max_abs_diff=0.000000101 verified=true"
        )
        memory = {
            "baseline_total_bytes": 1000,
            "peak_total_bytes": 2000,
            "consumed_total_bytes": 1000,
        }

        report = TOOL.parse_key_value_stdout(stdout)
        metrics = TOOL.parse_ullm_component_prefill_metrics(report, memory)
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
        TOOL.enrich_ullm_component_prefill_row(row, report)

        self.assertEqual(report["command"], "package-prefill-aq4-matvec-batch-smoke")
        self.assertEqual(metrics["prefill_total_input_tokens"], 2)
        self.assertEqual(metrics["prefill_total_input_tokens_per_second"], 19210.081451)
        self.assertEqual(metrics["prefill_wall_time_seconds"], 0.000104112)
        self.assertEqual(row["workload"]["prefill_mode"], "cold")
        self.assertEqual(row["workload"]["prompt_tokens_per_request"], [2])
        self.assertEqual(row["batching"]["mode"], "real")
        self.assertTrue(row["batching"]["prefill_real_batch"])
        self.assertEqual(row["batching"]["prefill_executor"], "aq4_matvec_batch_f32")
        self.assertEqual(row["batching"]["prefill_executor_request_parallelism"], 1)
        self.assertEqual(row["batching"]["prefill_executor_token_parallelism"], 2)
        self.assertEqual(row["batching"]["component_package"], "/tmp/model.ullm.d")

    def test_preserves_requested_batch_for_flattened_package_component(self) -> None:
        stdout = (
            "package-prefill-aq4-matvec-batch-smoke "
            "package=/tmp/model.ullm.d "
            'tensor="model.language_model.layers.3.self_attn.k_proj.weight" '
            "prompt_tokens=8 hidden=4096 rows=1024 cols=4096 "
            "executor=aq4_matvec_batch_f32 real_batch=true token_parallelism=8 "
            "request_parallelism=1 wall_ms_mean=0.200000 token_tps_mean=40000.000000 "
            "max_abs_diff=0.000000100 verified=true"
        )
        memory = {
            "baseline_total_bytes": 1000,
            "peak_total_bytes": 2000,
            "consumed_total_bytes": 1000,
        }

        report = TOOL.parse_key_value_stdout(stdout)
        metrics = TOOL.parse_ullm_component_prefill_metrics(report, memory)
        row = {
            "workload": {
                "batch_size": 4,
                "concurrent_requests": 4,
                "prompt_tokens": 2,
                "generated_tokens": 0,
                "kv_cache_dtype": "f32",
                "prefill_executor": None,
                "resolved_prefill_executor": None,
            },
            "metrics": metrics,
            "memory": memory.copy(),
        }
        TOOL.enrich_ullm_component_prefill_row(row, report)

        self.assertEqual(metrics["prefill_total_input_tokens"], 8)
        self.assertEqual(row["workload"]["batch_size"], 4)
        self.assertEqual(row["workload"]["concurrent_requests"], 4)
        self.assertEqual(row["workload"]["prompt_tokens_per_request"], [2, 2, 2, 2])
        self.assertEqual(row["workload"]["total_context_tokens_after_prefill"], 8)
        self.assertEqual(row["workload"]["component_total_input_tokens"], 8)
        self.assertEqual(row["batching"]["mode"], "real")
        self.assertTrue(row["batching"]["prefill_real_batch"])
        self.assertEqual(row["batching"]["prefill_executor_request_parallelism"], 1)
        self.assertEqual(row["batching"]["prefill_executor_token_parallelism"], 8)

    def test_parses_model_loop_hybrid_throughput_key_value_output(self) -> None:
        stdout = (
            "package-self-attn-mlp-block-model-loop-smoke "
            "package=/tmp/model.ullm.d layers=[3, 7] layers_csv=3,7 "
            "input_source=embedding_token_ids prefill_mode=token_id_layer_stack "
            "sq_overlay=true sq_candidate=sq-fp8-w8a16-r9700-v0 "
            "sq_artifact=/tmp/sq-artifact sq_schema_version=sq-fp8-artifact-v0.1 "
            "sq_fp8_tensor_count=22 sq_passthrough_tensor_count=753 sq_row_chunk=256 "
            "request_batch_executor=true fused_request_batch=false throughput_row=false "
            "batching_mode=real "
            "prefill_executor=stack_prefill_request_batch_step decode_executor=stack_ready_batch "
            "prefill_real_batch=true decode_real_batch=true "
            "prefill_executor_request_parallelism=3 decode_executor_request_parallelism=2 "
            "final_lm_head_guard=true final_top1_tokens_csv=42,43,44 "
            "final_topk_tokens_csv=42,7,5;43,8,6;44,9,1 "
            "final_topk_logits_csv=3.250000,2.000000,1.500000;4.500000,4.000000,3.500000;5.750000,5.000000,4.250000 "
            "sequence_len=3 request_count=3 concurrent_requests=3 "
            "prompt_tokens=[1, 2, 1] prompt_tokens_csv=1,2,1 "
            "max_new_tokens=[2, 1, 0] max_new_tokens_csv=2,1,0 "
            "total_tokens=[3, 3, 1] total_tokens_csv=3,3,1 "
            "prefill_total_input_tokens=4 decode_total_generated_tokens=3 "
            "end_to_end_total_tokens=7 prefill_wall_ms=47.124272 "
            "decode_wall_ms=35.822329 total_wall_ms=82.946601 "
            "prefill_total_input_tps=84.881948 "
            "decode_total_generated_tps=83.746649 "
            "end_to_end_total_tps=84.391644 "
            "prefill_batch_request_counts=[3, 1] prefill_batch_request_counts_csv=3,1 "
            "decode_batch_ready_counts=[2, 1] decode_batch_ready_counts_csv=2,1 "
            "generated_tokens=[2, 1, 0] generated_tokens_csv=2,1,0 "
            "layer_max_abs_diff=0.000000000 block_max_abs_diff=0.000000000 "
            "backend=hip device_index=2 name=\"AMD Radeon Graphics\" verified=true"
        )
        memory = {
            "baseline_total_bytes": 1000,
            "peak_total_bytes": 2000,
            "consumed_total_bytes": 1000,
        }

        report = TOOL.parse_key_value_stdout(stdout)
        metrics = TOOL.parse_ullm_model_loop_metrics(report, memory)
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
        TOOL.enrich_ullm_model_loop_row(row, report)

        self.assertEqual(report["command"], "package-self-attn-mlp-block-model-loop-smoke")
        self.assertEqual(metrics["prefill_total_input_tokens"], 4)
        self.assertEqual(metrics["decode_total_generated_tokens"], 3)
        self.assertEqual(metrics["end_to_end_total_tokens"], 7)
        self.assertEqual(metrics["prefill_total_input_tokens_per_second"], 84.881948)
        self.assertEqual(metrics["decode_total_generated_tokens_per_second"], 83.746649)
        self.assertEqual(metrics["end_to_end_total_tokens_per_second"], 84.391644)
        self.assertEqual(row["workload"]["batch_size"], 3)
        self.assertEqual(row["workload"]["concurrent_requests"], 3)
        self.assertEqual(row["workload"]["prompt_tokens_per_request"], [1, 2, 1])
        self.assertEqual(row["workload"]["generated_tokens_per_request"], [2, 1, 0])
        self.assertEqual(
            row["workload"]["total_context_tokens_after_prefill_per_request"],
            [3, 3, 1],
        )
        self.assertEqual(row["workload"]["prefill_mode"], "token_id_layer_stack")
        self.assertEqual(row["workload"]["input_source"], "embedding_token_ids")
        self.assertTrue(row["workload"]["sq_overlay"])
        self.assertEqual(row["workload"]["sq_candidate"], "sq-fp8-w8a16-r9700-v0")
        self.assertEqual(row["workload"]["sq_artifact"], "/tmp/sq-artifact")
        self.assertEqual(row["workload"]["sq_schema_version"], "sq-fp8-artifact-v0.1")
        self.assertEqual(row["workload"]["sq_fp8_tensor_count"], 22)
        self.assertEqual(row["workload"]["sq_passthrough_tensor_count"], 753)
        self.assertEqual(row["workload"]["sq_row_chunk"], 256)
        self.assertEqual(row["workload"]["final_top1_tokens"], [42, 43, 44])
        self.assertEqual(
            row["workload"]["final_topk_tokens"],
            [[42, 7, 5], [43, 8, 6], [44, 9, 1]],
        )
        self.assertEqual(
            row["workload"]["final_topk_logits"],
            [[3.25, 2.0, 1.5], [4.5, 4.0, 3.5], [5.75, 5.0, 4.25]],
        )
        self.assertEqual(row["workload"]["layers_csv"], "3,7")
        self.assertEqual(row["batching"]["mode"], "real")
        self.assertTrue(row["batching"]["prefill_real_batch"])
        self.assertEqual(row["batching"]["prefill_executor_request_parallelism"], 3)
        self.assertEqual(row["batching"]["prefill_batch_request_counts"], [3, 1])
        self.assertTrue(row["batching"]["decode_real_batch"])
        self.assertEqual(row["batching"]["decode_executor_request_parallelism"], 2)
        self.assertTrue(row["batching"]["request_batch_executor"])
        self.assertFalse(row["batching"]["fused_request_batch"])
        self.assertFalse(row["batching"]["throughput_row"])
        self.assertEqual(row["batching"]["component_package"], "/tmp/model.ullm.d")


if __name__ == "__main__":
    unittest.main()
