from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
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


TOOL = load_tool("run-external-benchmark.py")


class ExternalBenchmarkBatchParserTests(unittest.TestCase):
    def test_parses_topk_matrix_rows_with_comma_or_colon_items(self) -> None:
        self.assertEqual(
            TOOL.parse_int_matrix_csv("42,7,5;43,8,6"),
            [[42, 7, 5], [43, 8, 6]],
        )
        self.assertEqual(
            TOOL.parse_int_matrix_csv("42:7:5;43:8:6"),
            [[42, 7, 5], [43, 8, 6]],
        )
        self.assertEqual(
            TOOL.parse_float_matrix_csv("3.25,2.0;4.5,4.0"),
            [[3.25, 2.0], [4.5, 4.0]],
        )
        self.assertEqual(
            TOOL.parse_float_matrix_csv("3.25:2.0;4.5:4.0"),
            [[3.25, 2.0], [4.5, 4.0]],
        )

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

    def test_parses_cached_prefix_prefill_dispatch_selected_info(self) -> None:
        stdout = (
            "runtime-cached-prefix-attn-smoke backend=hip device_index=2 "
            'name="AMD Radeon Graphics" prefill_mode=cached_prefix '
            "executor=cached_prefix_rdna4_fp8_auto resolved_executor=cached_prefix_flash2_fp8q "
            "executor_selection=backend_dispatch "
            "selected_implementation_id=cached_prefix_rdna4_fp8_auto "
            "dispatch_operation=cached_prefix_attention "
            "dispatch_phase=prefill dispatch_format_id=SQ8_0 dispatch_gpu_arch=RDNA4 "
            "kv_cache_dtype=fp8_e4m3 cached_prefix_tokens=4096 new_prefill_tokens=16 "
            "total_context_tokens_after_prefill=4112 "
            "q_heads=16 kv_heads=1 head_dim=256 value_dim=256 "
            "estimated_prefill_attention_work_tokens=65536 measured_repeats=1 "
            "wall_ms_mean=0.050000 wall_ms_min=0.050000 wall_ms_max=0.050000 "
            "prefill_total_input_tps=320000.000000 "
            "attention_pair_tps_mean=1310720000.000000 "
            "verification=sampled sample_count=10 sampled_max_abs_diff=0.000000001 "
            "verified=true"
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

        self.assertEqual(report["command"], "runtime-cached-prefix-attn-smoke")
        self.assertEqual(row["workload"]["prefill_executor"], "cached_prefix_rdna4_fp8_auto")
        self.assertEqual(
            row["workload"]["resolved_prefill_executor"],
            "cached_prefix_flash2_fp8q",
        )
        self.assertEqual(row["workload"]["prompt_tokens_per_request"], [16])
        self.assertEqual(row["workload"]["cached_prefix_tokens_per_request"], [4096])
        self.assertEqual(row["workload"]["new_prefill_tokens_per_request"], [16])
        self.assertEqual(
            row["workload"]["total_context_tokens_after_prefill_per_request"],
            [4112],
        )
        self.assertEqual(row["workload"]["cached_prefix_total_tokens"], 4096)
        self.assertEqual(row["workload"]["total_context_tokens_after_prefill"], 4112)
        self.assertEqual(row["workload"]["component_total_input_tokens"], 16)
        self.assertEqual(row["workload"]["estimated_prefill_attention_work_tokens"], 65536)
        self.assertEqual(metrics["prefill_total_input_tokens"], 16)
        self.assertEqual(row["batching"]["prefill_executor"], "cached_prefix_rdna4_fp8_auto")
        self.assertEqual(
            row["batching"]["resolved_prefill_executor"],
            "cached_prefix_flash2_fp8q",
        )
        self.assertEqual(
            row["workload"]["selected_implementation_id"],
            "cached_prefix_rdna4_fp8_auto",
        )
        self.assertEqual(
            row["workload"]["dispatch_selected_implementation_id"],
            "cached_prefix_rdna4_fp8_auto",
        )
        self.assertEqual(row["workload"]["executor_selection"], "backend_dispatch")
        self.assertEqual(row["workload"]["dispatch_operation"], "cached_prefix_attention")
        self.assertEqual(row["workload"]["dispatch_phase"], "prefill")
        self.assertEqual(row["workload"]["dispatch_format_id"], "SQ8_0")
        self.assertEqual(row["workload"]["dispatch_gpu_arch"], "RDNA4")
        self.assertEqual(row["batching"]["executor_selection"], "backend_dispatch")
        self.assertEqual(row["batching"]["dispatch_operation"], "cached_prefix_attention")
        self.assertEqual(row["batching"]["dispatch_phase"], "prefill")
        self.assertEqual(row["batching"]["dispatch_format_id"], "SQ8_0")
        self.assertEqual(row["batching"]["dispatch_gpu_arch"], "RDNA4")

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
            "format_id=SQ8_0 sq_overlay=true sq_candidate=SQ8_0 "
            "sq_candidate_legacy=sq-fp8-w8a16-r9700-v0 "
            "sq_format_id=SQ8_0 sq_implementation_id=sq-fp8-w8a16-r9700-v0 "
            "sq_artifact=/tmp/sq-artifact sq_schema_version=sq-fp8-artifact-v0.1 "
            "sq_fp8_tensor_count=22 sq_passthrough_tensor_count=753 sq_row_chunk=256 "
            "sq_execution_mode=direct_fp8_dequant_matvec "
            "sq_projection_boundary=pair "
            "sq_projection_implementation_ids=pair=sq8_0_matvec_pair_rdna4_direct "
            "sq_projection_kernel_families=pair=direct "
            "sq_fp8_single_matvec_count=0 "
            "sq_fp8_batch_matvec_count=0 sq_fp8_expected_all_batch_matvec_count=14 "
            "sq_fp8_pair_matvec_count=8 "
            "sq_fp8_triple_matvec_count=0 "
            "request_batch_executor=true fused_request_batch=false throughput_row=true "
            "load_excluded_from_total=true final_logits_in_total=true "
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
            "decode_wall_ms=35.822329 final_logits_wall_ms=12.000000 "
            "layer_load_ms=34.000000 total_wall_ms=94.946601 outer_wall_ms=128.946601 "
            "artifact_materialization_ms=7.500000 "
            "prefill_total_input_tps=84.881948 "
            "decode_total_generated_tps=83.746649 "
            "end_to_end_total_tps=73.725646 "
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
        self.assertEqual(metrics["end_to_end_total_tokens_per_second"], 73.725646)
        self.assertEqual(metrics["final_logits_wall_time_seconds"], 0.012)
        self.assertEqual(metrics["layer_load_wall_time_seconds"], 0.034)
        self.assertEqual(metrics["artifact_load_wall_time_seconds"], 0.034)
        self.assertEqual(metrics["artifact_materialization_wall_time_seconds"], 0.0075)
        self.assertEqual(metrics["load_excluded_total_wall_time_seconds"], 0.094946601)
        self.assertEqual(metrics["load_included_total_wall_time_seconds"], 0.128946601)
        self.assertEqual(metrics["total_wall_time_seconds"], 0.094946601)
        self.assertEqual(metrics["outer_wall_time_seconds"], 0.128946601)
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
        self.assertEqual(row["workload"]["format_id"], "SQ8_0")
        self.assertTrue(row["workload"]["sq_overlay"])
        self.assertEqual(row["workload"]["sq_candidate"], "SQ8_0")
        self.assertEqual(row["workload"]["sq_candidate_legacy"], "sq-fp8-w8a16-r9700-v0")
        self.assertEqual(row["workload"]["sq_format_id"], "SQ8_0")
        self.assertEqual(
            row["workload"]["sq_implementation_id"],
            "sq-fp8-w8a16-r9700-v0",
        )
        self.assertEqual(row["workload"]["sq_artifact"], "/tmp/sq-artifact")
        self.assertEqual(row["workload"]["sq_schema_version"], "sq-fp8-artifact-v0.1")
        self.assertEqual(row["workload"]["sq_fp8_tensor_count"], 22)
        self.assertEqual(row["workload"]["sq_passthrough_tensor_count"], 753)
        self.assertEqual(row["workload"]["sq_row_chunk"], 256)
        self.assertEqual(
            row["workload"]["sq_execution_mode"],
            "direct_fp8_dequant_matvec",
        )
        self.assertEqual(row["workload"]["sq_projection_boundary"], "pair")
        self.assertEqual(
            row["workload"]["sq_projection_implementation_ids"],
            "pair=sq8_0_matvec_pair_rdna4_direct",
        )
        self.assertEqual(
            row["workload"]["sq_projection_kernel_families"],
            "pair=direct",
        )
        self.assertEqual(row["workload"]["sq_fp8_single_matvec_count"], 0)
        self.assertEqual(row["workload"]["sq_fp8_batch_matvec_count"], 0)
        self.assertEqual(row["workload"]["sq_fp8_expected_all_batch_matvec_count"], 14)
        self.assertEqual(row["workload"]["sq_fp8_pair_matvec_count"], 8)
        self.assertEqual(row["workload"]["sq_fp8_triple_matvec_count"], 0)
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
        self.assertTrue(row["batching"]["throughput_row"])
        self.assertTrue(row["batching"]["load_excluded_from_total"])
        self.assertTrue(row["batching"]["final_logits_in_total"])
        self.assertEqual(row["batching"]["component_package"], "/tmp/model.ullm.d")

    def test_accepts_materialized_fallback_for_selected_layer_diagnostic(self) -> None:
        stdout = (
            "sq-fp8-token-ids-model-loop-smoke "
            "package=/tmp/model.ullm.d layers=[3, 7] layers_csv=3,7 "
            "sq_execution_mode=materialized_f32_fallback "
            "throughput_row=true "
            "prefill_wall_ms=40.124272 decode_wall_ms=35.822329 "
            "final_logits_wall_ms=12.000000 layer_load_ms=34.000000 total_wall_ms=94.946601 "
            "outer_wall_ms=128.946601 prefill_total_input_tokens=4 "
            "decode_total_generated_tokens=3 end_to_end_total_tokens=7 "
            "prefill_total_input_tps=84.881948 decode_total_generated_tps=83.746649 "
            "end_to_end_total_tps=73.725646 generated_tokens_csv=2,1 request_count=2 "
            "prompt_tokens_csv=1,2 backend=hip device_index=2 name=\"AMD Radeon Graphics\" "
            "verified=true"
        )
        report = TOOL.parse_key_value_stdout(stdout)

        status, error = TOOL.classify_sq_execution_fallback(
            "ok", None, "ullm-model-loop-throughput", report, False
        )
        self.assertEqual(status, "ok")
        self.assertIsNone(error)

    def test_rejects_materialized_fallback_for_non_diagnostic_full_model_loop(self) -> None:
        stdout = (
            "package-self-attn-mlp-block-model-loop-smoke "
            "package=/tmp/model.ullm.d layers=[3, 7] layers_csv=3,7 "
            "sq_execution_mode=materialized_f32_fallback "
            "throughput_row=true "
            "prefill_wall_ms=47.124272 decode_wall_ms=35.822329 final_logits_wall_ms=12.000000 "
            "layer_load_ms=34.000000 total_wall_ms=94.946601 outer_wall_ms=128.946601 "
            "prefill_total_input_tokens=4 decode_total_generated_tokens=3 end_to_end_total_tokens=7 "
            "prefill_total_input_tps=84.881948 decode_total_generated_tps=83.746649 "
            "end_to_end_total_tps=73.725646 generated_tokens_csv=2,1 request_count=2 "
            "prompt_tokens_csv=1,2 backend=hip device_index=2 name=\"AMD Radeon Graphics\" "
            "verified=true"
        )
        report = TOOL.parse_key_value_stdout(stdout)

        status, error = TOOL.classify_sq_execution_fallback(
            "ok", None, "ullm-model-loop-throughput", report, False
        )
        self.assertEqual(status, "failed")
        self.assertIsNotNone(error)
        self.assertEqual(error["type"], "invalid_fallback")

    def test_accepts_materialized_fallback_when_cli_marked(self) -> None:
        stdout = (
            "package-self-attn-mlp-block-model-loop-smoke "
            "sq_execution_mode=materialized_f32_fallback "
            "throughput_row=true "
            "backend=hip device_index=2 name=\"AMD Radeon Graphics\" "
            "verified=true"
        )
        report = TOOL.parse_key_value_stdout(stdout)

        status, error = TOOL.classify_sq_execution_fallback(
            "ok", None, "ullm-model-loop-throughput", report, True
        )
        self.assertEqual(status, "ok")
        self.assertIsNone(error)

    def test_accepts_materialized_fallback_when_report_marks_allowed(self) -> None:
        stdout = (
            "package-self-attn-mlp-block-model-loop-smoke "
            "fallback_allowed=true sq_execution_mode=materialized_f32_fallback "
            "throughput_row=true "
            "backend=hip device_index=2 name=\"AMD Radeon Graphics\" "
            "verified=true"
        )
        report = TOOL.parse_key_value_stdout(stdout)

        status, error = TOOL.classify_sq_execution_fallback(
            "ok", None, "ullm-model-loop-throughput", report, False
        )
        self.assertEqual(status, "ok")
        self.assertIsNone(error)

    def test_accepts_materialized_fallback_when_report_marks_diagnostic(self) -> None:
        stdout = (
            "package-self-attn-mlp-block-model-loop-smoke "
            "diagnostic=true sq_execution_mode=materialized_f32_fallback "
            "throughput_row=true "
            "backend=hip device_index=2 name=\"AMD Radeon Graphics\" "
            "verified=true"
        )
        report = TOOL.parse_key_value_stdout(stdout)

        status, error = TOOL.classify_sq_execution_fallback(
            "ok", None, "ullm-model-loop-throughput", report, False
        )
        self.assertEqual(status, "ok")
        self.assertIsNone(error)

    def test_classifies_benchmark_harness_for_vllm_and_ullm_model_loop(self) -> None:
        vllm = TOOL.classify_benchmark_harness("vllm-throughput")
        model_loop = TOOL.classify_benchmark_harness("ullm-model-loop-throughput")

        self.assertNotEqual(vllm["class"], model_loop["class"])
        self.assertEqual(vllm["class"], "serving_throughput_benchmark")
        self.assertTrue(vllm["serving_parity_candidate"])
        self.assertFalse(vllm["includes_http_server"])
        self.assertEqual(vllm["harness_type"], "vllm_bench_throughput_cli")
        self.assertEqual(model_loop["class"], "cli_model_loop_diagnostic")
        self.assertFalse(model_loop["serving_parity_candidate"])
        self.assertFalse(model_loop["includes_http_server"])
        self.assertEqual(model_loop["harness_type"], "ullm_cli_model_loop")

    def test_classifies_benchmark_harness_for_ullm_serving_throughput(self) -> None:
        serving = TOOL.classify_benchmark_harness("ullm-serving-throughput")
        self.assertEqual(serving["class"], "ullm_serving_throughput_candidate")
        self.assertFalse(serving["serving_parity_candidate"])
        self.assertFalse(serving["includes_http_server"])
        self.assertEqual(
            serving["notes"],
            ["source=ullm_offline_serving_throughput_candidate"],
        )
        self.assertEqual(
            serving["harness_type"], "ullm_offline_serving_throughput_cli"
        )

    def test_runs_main_with_ullm_serving_throughput_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_jsonl = root / "results.jsonl"
            stdout_log = root / "stdout.log"
            stderr_log = root / "stderr.log"
            memory_log = root / "memory.jsonl"
            result_json = root / "raw.json"
            report = {
                "workload": {
                    "batch_size": 4,
                    "concurrent_requests": 3,
                },
                "batching": {
                    "mode": "real",
                    "prefill_executor": "cached_prefix_rdna4_fp8_auto",
                    "resolved_prefill_executor": "cached_prefix_flash2_fp8q",
                    "prefill_real_batch": True,
                    "decode_real_batch": True,
                },
                "metrics": {
                    "prefill_total_input_tps": 100.0,
                    "decode_total_generated_tps": 200.0,
                    "end_to_end_total_tps": 80.0,
                    "prefill_total_input_tokens": 400,
                    "decode_total_generated_tokens": 240,
                    "generated_tokens_total": 240,
                    "end_to_end_total_tokens": 640,
                    "prefill_wall_ms_sum": 80.0,
                    "decode_wall_ms_sum": 20.0,
                    "batch_wall_ms": 150.0,
                    "request_latency_ms_p50": 75.0,
                    "request_latency_ms_p95": 100.0,
                    "time_to_first_token_ms_p50": 40.0,
                    "time_to_first_token_ms_p95": 70.0,
                    "time_per_output_token_ms_p50": 10.0,
                    "time_per_output_token_ms_p95": 12.0,
                    "per_request_decode_tps_mean": 90.0,
                },
                "serving": {
                    "candidate_contract_version": "v2026-1",
                    "serving_loop_kind": "offline_token_batching",
                    "scheduler_policy": "real_request_batch",
                    "request_source": "synthetic_workload",
                    "request_arrival_pattern": "steady",
                    "tokenizer_included": True,
                    "http_server_included": False,
                    "runtime_reused_across_requests": True,
                    "weights_reloaded_per_request": False,
                    "load_excluded_from_total": True,
                    "final_logits_in_total": False,
                },
                "correctness": {"verified_all": True},
                "verified": True,
                "memory": {"kv_cache_bytes_total": 98304},
            }
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            command = [
                sys.executable,
                str(REPO_ROOT / "tools" / "run-external-benchmark.py"),
                "--run-id",
                "serving-throughput-smoke",
                "--case-id",
                "serving-throughput-case",
                "--output-jsonl",
                str(output_jsonl),
                "--stdout-log",
                str(stdout_log),
                "--stderr-log",
                str(stderr_log),
                "--memory-log",
                str(memory_log),
                "--engine-name",
                "uLLM",
                "--model-name",
                "Qwen3",
                "--model-format",
                "ullm-package",
                "--model-quantization",
                "AQ4",
                "--context-length",
                "128",
                "--prompt-tokens",
                "16",
                "--generated-tokens",
                "4",
                "--batch-size",
                "1",
                "--concurrent-requests",
                "1",
                "--kv-cache-dtype",
                "f32",
                "--parse",
                "ullm-serving-throughput",
                "--result-json",
                str(result_json),
                "--",
                sys.executable,
                "-c",
                "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))",
                str(report_path),
            ]
            process = subprocess.run(
                command,
                check=False,
                text=True,
                capture_output=True,
                cwd=REPO_ROOT,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            row = json.loads(
                output_jsonl.read_text(encoding="utf-8").strip().splitlines()[-1]
            )

            self.assertEqual(row["harness"]["class"], "ullm_serving_throughput_candidate")
            self.assertFalse(row["harness"]["serving_parity_candidate"])
            self.assertEqual(row["workload"]["batch_size"], 4)
            self.assertEqual(row["workload"]["concurrent_requests"], 3)
            self.assertEqual(row["batching"]["mode"], "real")
            self.assertEqual(
                row["metrics"]["prefill_total_input_tokens_per_second"], 100.0
            )
            self.assertEqual(
                row["metrics"]["decode_total_generated_tokens_per_second"], 200.0
            )
            self.assertEqual(row["memory"]["kv_cache_bytes_total"], 98304)
            candidate = row["harness"]["ullm_serving_candidate"]
            self.assertEqual(candidate["candidate_contract_version"], "v2026-1")
            self.assertEqual(candidate["serving_loop_kind"], "offline_token_batching")
            self.assertEqual(candidate["scheduler_policy"], "real_request_batch")
            self.assertEqual(candidate["request_source"], "synthetic_workload")
            self.assertEqual(candidate["request_arrival_pattern"], "steady")
            self.assertEqual(candidate["tokenizer_included"], True)
            self.assertEqual(candidate["http_server_included"], False)
            self.assertEqual(candidate["runtime_reused_across_requests"], True)
            self.assertEqual(candidate["weights_reloaded_per_request"], False)
            self.assertEqual(candidate["load_excluded_from_total"], True)
            self.assertEqual(candidate["final_logits_in_total"], False)
            self.assertEqual(candidate["parity_blockers"], [])

    def test_runs_main_with_ullm_serving_throughput_detects_contract_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_jsonl = root / "results.jsonl"
            stdout_log = root / "stdout.log"
            stderr_log = root / "stderr.log"
            memory_log = root / "memory.jsonl"
            result_json = root / "raw.json"
            report = {
                "workload": {
                    "batch_size": 4,
                    "concurrent_requests": 2,
                },
                "batching": {
                    "mode": "logical",
                    "prefill_executor": "cached_prefix_rdna4_fp8_auto",
                    "resolved_prefill_executor": "cached_prefix_flash2_fp8q",
                    "prefill_real_batch": False,
                    "decode_real_batch": False,
                },
                "serving": {
                    "candidate_contract_version": "v2026-2",
                    "serving_loop_kind": "offline_token_batching",
                    "scheduler_policy": "request_batch",
                    "request_source": "synthetic_workload",
                    "request_arrival_pattern": "constant",
                    "tokenizer_included": True,
                    "http_server_included": False,
                    "runtime_reused_across_requests": False,
                    "weights_reloaded_per_request": True,
                    "load_excluded_from_total": False,
                    "final_logits_in_total": True,
                    "parity_blockers": [
                        "runner_known_gap",
                        "batching_mode_not_real",
                    ],
                },
                "metrics": {
                    "prefill_total_input_tps": 90.0,
                    "decode_total_generated_tps": 180.0,
                    "end_to_end_total_tps": 70.0,
                    "prefill_total_input_tokens": 200,
                    "decode_total_generated_tokens": 160,
                    "generated_tokens_total": 160,
                    "end_to_end_total_tokens": 360,
                    "prefill_wall_ms_sum": 80.0,
                    "decode_wall_ms_sum": 20.0,
                    "batch_wall_ms": 150.0,
                },
                "correctness": {"verified_all": True},
                "verified": True,
            }
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            command = [
                sys.executable,
                str(REPO_ROOT / "tools" / "run-external-benchmark.py"),
                "--run-id",
                "serving-throughput-blockers",
                "--case-id",
                "serving-throughput-blockers-case",
                "--output-jsonl",
                str(output_jsonl),
                "--stdout-log",
                str(stdout_log),
                "--stderr-log",
                str(stderr_log),
                "--memory-log",
                str(memory_log),
                "--engine-name",
                "uLLM",
                "--model-name",
                "Qwen3",
                "--model-format",
                "ullm-package",
                "--model-quantization",
                "AQ4",
                "--context-length",
                "128",
                "--prompt-tokens",
                "16",
                "--generated-tokens",
                "4",
                "--batch-size",
                "1",
                "--concurrent-requests",
                "1",
                "--kv-cache-dtype",
                "f32",
                "--parse",
                "ullm-serving-throughput",
                "--result-json",
                str(result_json),
                "--",
                sys.executable,
                "-c",
                "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))",
                str(report_path),
            ]
            process = subprocess.run(
                command,
                check=False,
                text=True,
                capture_output=True,
                cwd=REPO_ROOT,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            row = json.loads(
                output_jsonl.read_text(encoding="utf-8").strip().splitlines()[-1]
            )

            blockers = row["harness"]["ullm_serving_candidate"]["parity_blockers"]
            self.assertEqual(
                blockers,
                [
                    "runner_known_gap",
                    "batching_mode_not_real",
                    "prefill_real_batch_not_true",
                    "decode_real_batch_not_true",
                    "runtime_reused_across_requests_not_true",
                    "weights_reloaded_per_request_not_false",
                    "load_excluded_from_total_not_true",
                    "final_logits_in_total_not_false",
                ],
            )

    def test_prompt_guard_bundle_fields_attached_when_token_logits_check_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "prompt_guard_bundle.json"
            bundle_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "checks": [
                            {
                                "name": "prompt_suite_token_logits",
                                "passed": True,
                                "metrics": {
                                    "acceptance_mode": "strict",
                                    "strict_passed": True,
                                    "behavioral_passed": True,
                                    "compared_case_count": 12,
                                    "generated_token_match_count": 10,
                                    "generated_text_match_count": 9,
                                    "generated_without_stop_text_match_count": 8,
                                    "top_logits_match_count": 7,
                                    "max_prefill_top_logit_abs_diff": 0.004,
                                    "max_decode_last_top_logit_abs_diff": 0.005,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            bundle = TOOL.load_prompt_guard_bundle(bundle_path)
            row = {"artifacts": {}}

            TOOL.attach_prompt_guard_bundle_fields(row, bundle_path, bundle)

            self.assertEqual(
                row["quality"]["prompt_suite_regression_status"], "passed"
            )
            self.assertEqual(row["guards"]["prompt_guard_bundle"]["status"], "ok")
            self.assertEqual(
                row["guards"]["prompt_guard_bundle"]["artifact"], str(bundle_path)
            )
            self.assertEqual(row["guards"]["prompt_guard_bundle"]["acceptance_mode"], "strict")
            self.assertEqual(
                row["guards"]["prompt_guard_bundle"]["compared_case_count"], 12
            )
            self.assertEqual(
                row["guards"]["prompt_guard_bundle"][
                    "max_decode_last_top_logit_abs_diff"
                ],
                0.005,
            )
            self.assertTrue(row["guards"]["prompt_guard_bundle"]["passed"])
            self.assertEqual(
                row["artifacts"]["prompt_guard_bundle_json"], str(bundle_path)
            )

    def test_prompt_guard_bundle_fields_attached_when_token_logits_check_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "prompt_guard_bundle.json"
            bundle_path.write_text(
                json.dumps({"passed": False, "checks": [{"name": "other_check"}]}),
                encoding="utf-8",
            )
            bundle = TOOL.load_prompt_guard_bundle(bundle_path)
            row = {"artifacts": {}}

            TOOL.attach_prompt_guard_bundle_fields(row, bundle_path, bundle)

            self.assertEqual(
                row["quality"]["prompt_suite_regression_status"], "not_attached"
            )
            self.assertFalse(row["guards"]["prompt_guard_bundle"]["passed"])
            self.assertIsNone(
                row["guards"]["prompt_guard_bundle"]["acceptance_mode"]
            )
            self.assertIsNone(row["guards"]["prompt_guard_bundle"]["strict_passed"])
            self.assertIsNone(
                row["guards"]["prompt_guard_bundle"]["max_prefill_top_logit_abs_diff"]
            )

    def test_parses_sq_fp8_selected_layer_model_loop_projection_telemetry(self) -> None:
        stdout = (
            "sq-fp8-token-ids-model-loop-smoke "
            "package=/tmp/model.ullm.d layers=[3, 7] layers_csv=3,7 "
            "input_source=embedding_token_ids prefill_mode=token_id_layer_stack "
            "format_id=SQ8_0 sq_overlay=true sq_candidate=SQ8_0 "
            "sq_candidate_legacy=sq-fp8-w8a16-r9700-v0 "
            "sq_format_id=SQ8_0 sq_implementation_id=sq-fp8-w8a16-r9700-v0 "
            "sq_artifact=/tmp/sq-artifact sq_schema_version=sq-fp8-artifact-v0.1 "
            "sq_fp8_tensor_count=22 sq_passthrough_tensor_count=753 sq_row_chunk=256 "
            "sq_execution_mode=materialized_f32_fallback "
            "sq_projection_boundary=none "
            "sq_projection_implementation_ids=none "
            "sq_projection_kernel_families=none "
            "sq_fp8_single_matvec_count=0 "
            "sq_fp8_batch_matvec_count=0 sq_fp8_pair_matvec_count=0 "
            "sq_fp8_triple_matvec_count=0 "
            "request_batch_executor=true fused_request_batch=false throughput_row=true "
            "load_excluded_from_total=true final_logits_in_total=true "
            "batching_mode=real "
            "prefill_executor=stack_prefill_request_batch_step decode_executor=stack_ready_batch "
            "prefill_real_batch=true decode_real_batch=true "
            "prefill_executor_request_parallelism=3 decode_executor_request_parallelism=2 "
            "prefill_batch_request_counts=[3, 1] decode_batch_ready_counts=[2, 1] "
            "prefill_wall_ms=40.124272 decode_wall_ms=35.822329 final_logits_wall_ms=12.000000 "
            "layer_load_ms=34.000000 total_wall_ms=94.946601 outer_wall_ms=128.946601 "
            "prefill_total_input_tokens=4 decode_total_generated_tokens=3 "
            "end_to_end_total_tokens=7 prefill_total_input_tps=84.881948 "
            "decode_total_generated_tps=83.746649 "
            "end_to_end_total_tps=73.725646 "
            "generated_tokens_csv=2,1 "
            "cached_tokens=[3, 3] "
            "final_top1_tokens_csv=42,43 "
            "final_topk_tokens_csv=42,7,5;43,8,6 "
            "final_topk_logits_csv=3.250000,2.000000,1.500000;4.500000,4.000000,3.500000 "
            "sequence_len=3 request_count=2 concurrent_requests=2 "
            "prompt_tokens_csv=1,2 max_new_tokens_csv=2,1 total_tokens_csv=3,3 "
            "layer_max_abs_diff=0.000000000 block_max_abs_diff=0.000000000 "
            "prefill_batch_request_counts_csv=3,1 decode_batch_ready_counts_csv=2,1 "
            "backend=hip device_index=2 name=\"AMD Radeon Graphics\" verified=true"
        )
        memory = {
            "baseline_total_bytes": 1000,
            "peak_total_bytes": 2000,
            "consumed_total_bytes": 1000,
        }

        report = TOOL.parse_key_value_stdout(stdout)
        row = {
            "workload": {
                "batch_size": 1,
                "concurrent_requests": 1,
                "kv_cache_dtype": "f32",
                "prefill_executor": None,
                "resolved_prefill_executor": None,
            },
            "batching": {
                "mode": "real",
                "prefill_executor": None,
                "resolved_prefill_executor": None,
            },
            "metrics": {},
            "memory": memory.copy(),
        }

        TOOL.enrich_ullm_model_loop_row(row, report)

        self.assertEqual(report["command"], "sq-fp8-token-ids-model-loop-smoke")
        self.assertEqual(row["workload"]["sq_execution_mode"], "materialized_f32_fallback")
        self.assertEqual(row["workload"]["sq_projection_boundary"], "none")
        self.assertEqual(row["workload"]["sq_fp8_single_matvec_count"], 0)
        self.assertEqual(row["workload"]["sq_fp8_batch_matvec_count"], 0)
        self.assertEqual(row["workload"]["sq_fp8_pair_matvec_count"], 0)
        self.assertEqual(row["workload"]["sq_fp8_triple_matvec_count"], 0)
        self.assertEqual(row["workload"]["sq_projection_implementation_ids"], "none")
        self.assertEqual(row["workload"]["sq_projection_kernel_families"], "none")
        self.assertEqual(row["workload"]["batch_size"], 2)
        self.assertEqual(row["workload"]["concurrent_requests"], 2)
        self.assertEqual(row["workload"]["prefill_mode"], "token_id_layer_stack")
        self.assertEqual(row["workload"]["format_id"], "SQ8_0")
        self.assertTrue(row["workload"]["sq_overlay"])

    def test_parses_sq_fp8_projection_rows_without_kernel_families(self) -> None:
        stdout = (
            "sq-fp8-token-ids-model-loop-smoke "
            "package=/tmp/model.ullm.d layers=[0, 1] layers_csv=0,1 "
            "input_source=embedding_token_ids prefill_mode=token_id_layer_stack "
            "format_id=SQ8_0 sq_overlay=true sq_candidate=SQ8_0 "
            "sq_candidate_legacy=sq-fp8-w8a16-r9700-v0 "
            "sq_format_id=SQ8_0 sq_implementation_id=sq-fp8-w8a16-r9700-v0 "
            "sq_artifact=/tmp/sq-artifact sq_schema_version=sq-fp8-artifact-v0.1 "
            "sq_fp8_tensor_count=22 sq_passthrough_tensor_count=753 sq_row_chunk=256 "
            "sq_execution_mode=direct_fp8_dequant_matvec "
            "sq_projection_boundary=single "
            "sq_projection_implementation_ids=single=sq8_0_matvec_r9700_direct "
            "sq_fp8_single_matvec_count=24 sq_fp8_batch_matvec_count=0 "
            "sq_fp8_pair_matvec_count=0 sq_fp8_triple_matvec_count=0 "
            "request_batch_executor=true fused_request_batch=false throughput_row=true "
            "load_excluded_from_total=true final_logits_in_total=true "
            "batching_mode=real "
            "prefill_executor=stack_prefill_request_batch_step decode_executor=stack_ready_batch "
            "prefill_real_batch=true decode_real_batch=true "
            "prefill_executor_request_parallelism=3 decode_executor_request_parallelism=2 "
            "prefill_batch_request_counts=[3, 1] decode_batch_ready_counts=[2, 1] "
            "prefill_wall_ms=40.124272 decode_wall_ms=35.822329 final_logits_wall_ms=12.000000 "
            "layer_load_ms=34.000000 total_wall_ms=94.946601 outer_wall_ms=128.946601 "
            "prefill_total_input_tokens=4 decode_total_generated_tokens=3 "
            "end_to_end_total_tokens=7 prefill_total_input_tps=84.881948 "
            "decode_total_generated_tps=83.746649 "
            "end_to_end_total_tps=73.725646 "
            "generated_tokens_csv=2,1 "
            "cached_tokens=[3, 3] "
            "final_top1_tokens_csv=42,43 "
            "final_topk_tokens_csv=42,7,5;43,8,6 "
            "final_topk_logits_csv=3.250000,2.000000,1.500000;4.500000,4.000000,3.500000 "
            "sequence_len=3 request_count=2 concurrent_requests=2 "
            "prompt_tokens_csv=1,2 max_new_tokens_csv=2,1 total_tokens_csv=3,3 "
            "layer_max_abs_diff=0.000000000 block_max_abs_diff=0.000000000 "
            "prefill_batch_request_counts_csv=3,1 decode_batch_ready_counts_csv=2,1 "
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
            "batching": {
                "mode": "real",
                "prefill_executor": None,
                "resolved_prefill_executor": None,
            },
            "metrics": metrics,
            "memory": memory.copy(),
        }

        TOOL.enrich_ullm_model_loop_row(row, report)

        self.assertEqual(report["command"], "sq-fp8-token-ids-model-loop-smoke")
        self.assertEqual(row["workload"]["sq_projection_boundary"], "single")
        self.assertEqual(
            row["workload"]["sq_projection_implementation_ids"],
            "single=sq8_0_matvec_r9700_direct",
        )
        self.assertNotIn("sq_projection_kernel_families", row["workload"])

    def test_parses_sq_fp8_package_self_attn_layer_batch_smoke_key_value_output(self) -> None:
        stdout = (
            "sq-fp8-package-self-attn-layer-batch-smoke "
            "package=/tmp/model.ullm.d layer=3 format_id=SQ8_0 "
            "sq_overlay=true sq_candidate=SQ8_0 "
            "sq_candidate_legacy=sq-fp8-w8a16-r9700-v0 "
            "sq_format_id=SQ8_0 sq_implementation_id=sq-fp8-w8a16-r9700-v0 "
            "sq_artifact=/tmp/sq-artifact sq_schema_version=sq-fp8-artifact-v0.1 "
            "sq_fp8_tensor_count=22 sq_passthrough_tensor_count=753 sq_row_chunk=256 "
            "sq_execution_mode=direct_fp8_dequant_matvec_batch "
            "sq_projection_boundary=batch "
            "sq_projection_implementation_ids=batch=sq8_0_matvec_batch_r9700_direct "
            "sq_projection_kernel_families=batch=direct "
            "sq_fp8_single_matvec_count=0 sq_fp8_batch_matvec_count=14 "
            "sq_fp8_expected_all_batch_matvec_count=14 sq_fp8_pair_matvec_count=0 "
            "sq_fp8_triple_matvec_count=0 real_batch=true token_parallelism=32 "
            "request_parallelism=2 "
            "executor=segmented_rmsnorm_f32+sq8_0_matvec_batch_r9700_direct+"
            "qwen35_qk_norm_rope_batch_f32+causal_attn_f32+sigmoid_mul_f32+"
            "sq8_0_matvec_batch_r9700_direct+add_f32 "
            "prefill_total_input_tokens=14 decode_total_generated_tokens=0 "
            "end_to_end_total_tokens=14 prefill_total_input_tps=120.5 "
            "decode_total_generated_tps=0.0 end_to_end_total_tps=120.5 "
            "prefill_wall_ms=12.345678 decode_wall_ms=0.0 final_logits_wall_ms=0.0 "
            "layer_load_ms=100.0 total_wall_ms=112.345678 outer_wall_ms=115.345678 "
            "backend=hip device_index=2 name=\"AMD Radeon Graphics\" "
            "verified=true"
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

        self.assertEqual(report["command"], "sq-fp8-package-self-attn-layer-batch-smoke")
        self.assertEqual(metrics["prefill_total_input_tokens"], 14)
        self.assertEqual(metrics["prefill_total_input_tokens_per_second"], 120.5)
        self.assertTrue(row["workload"]["sq_overlay"])
        self.assertEqual(row["workload"]["format_id"], "SQ8_0")
        self.assertEqual(row["workload"]["sq_projection_boundary"], "batch")
        self.assertEqual(
            row["workload"]["sq_projection_kernel_families"],
            "batch=direct",
        )
        self.assertEqual(
            row["workload"]["sq_projection_implementation_ids"],
            "batch=sq8_0_matvec_batch_r9700_direct",
        )
        self.assertEqual(row["workload"]["sq_fp8_batch_matvec_count"], 14)
        self.assertEqual(row["workload"]["sq_fp8_expected_all_batch_matvec_count"], 14)
        self.assertEqual(
            row["workload"]["prefill_executor"],
            "segmented_rmsnorm_f32+sq8_0_matvec_batch_r9700_direct+qwen35_qk_norm_rope_batch_f32+causal_attn_f32+sigmoid_mul_f32+sq8_0_matvec_batch_r9700_direct+add_f32",
        )
        self.assertEqual(row["batching"]["mode"], "real")
        self.assertEqual(
            row["batching"]["component_command"],
            "sq-fp8-package-self-attn-layer-batch-smoke",
        )
        self.assertEqual(
            row["batching"]["prefill_executor"],
            row["workload"]["prefill_executor"],
        )
        self.assertEqual(
            row["batching"]["resolved_prefill_executor"],
            row["workload"]["prefill_executor"],
        )
        self.assertTrue(row["batching"]["prefill_real_batch"])
        self.assertEqual(row["batching"]["prefill_executor_token_parallelism"], 32)
        self.assertEqual(row["batching"]["prefill_executor_request_parallelism"], 2)

    def test_parses_vllm_throughput_metrics_with_json_and_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "vllm_output.json"
            output_path.write_text(
                (
                    "{"
                    '"elapsed_time": 5.5, '
                    '"requests_per_second": 18.5, '
                    '"tokens_per_second": 2400.0, '
                    '"total_num_tokens": 13200'
                    "}"
                ),
                encoding="utf-8",
            )

            stdout = (
                "Some benchmark startup logs...\n"
                "Throughput: 18.5 requests/s, 2400.0 total tokens/s, 900.0 output tokens/s\n"
                "Total num prompt tokens: 11000\n"
            )
            memory = {
                "baseline_total_bytes": 1_000,
                "peak_total_bytes": 3_000,
                "consumed_total_bytes": 2 * 1024**3,
            }

            metrics = TOOL.parse_vllm_metrics(stdout, output_path, memory)
            self.assertAlmostEqual(metrics["prefill_tokens_per_second"], 2000.0, places=7)
            self.assertAlmostEqual(metrics["decode_tokens_per_second"], 900.0, places=7)
            self.assertAlmostEqual(metrics["total_tokens_per_second"], 2400.0, places=7)
            self.assertAlmostEqual(metrics["requests_per_second"], 18.5, places=7)
            self.assertEqual(metrics["vram_consumed_bytes"], 2 * 1024**3)
            self.assertEqual(metrics["vram_baseline_bytes"], 1_000)
            self.assertEqual(metrics["vram_peak_bytes"], 3_000)
            self.assertAlmostEqual(
                metrics["decode_tokens_per_second_times_vram_consumed_gib"], 1800.0, places=7
            )

    def test_classifies_rocm_fp8_unsupported_runtime_patterns(self) -> None:
        cases = (
            "not supported",
            "unsupported",
            "no kernel image is available",
            "hipErrorNoBinaryForGpu",
            "invalid device function",
            "not compiled for this GPU",
            "gfx1201 is not supported",
        )
        for text in cases:
            status, error = TOOL.classify_failure(1, False, text)
            self.assertEqual(status, "unsupported", text)
            self.assertEqual(error["type"], "unsupported_runtime", text)
            self.assertEqual(
                error["message"],
                "Benchmark command reported unsupported runtime or model path.",
                text,
            )

    def test_classify_failure_preserves_oom_and_timeout(self) -> None:
        status, error = TOOL.classify_failure(1, False, "RuntimeError: CUDA out of memory")
        self.assertEqual(status, "oom")
        self.assertEqual(error["type"], "oom")

        status, error = TOOL.classify_failure(1, True, "hipErrorNoBinaryForGpu")
        self.assertEqual(status, "failed")
        self.assertEqual(error["type"], "timeout")

    def test_parses_model_loop_single_request_csv_fields(self) -> None:
        stdout = (
            "package-token-ids-mixed-request-state-smoke "
            "package=/tmp/model.ullm.d layers_csv=0,1,2 input_source=embedding_token_ids "
            "first_layer_input_source=device_embedding "
            "prefill_mode=token_id_full_mixed_request_state "
            "request_batch_executor=true fused_request_batch=false throughput_row=true "
            "load_excluded_from_total=true final_logits_in_total=true "
            "batching_mode=single "
            "prefill_executor=mixed_request_state_layer_batch_step "
            "decode_executor=mixed_request_state_layer_batch_step "
            "prefill_real_batch=false decode_real_batch=false "
            "mixed_request_state_real_batch_projection_used=false "
            "prefill_sq_fp8_batch_matvec_count=0 decode_sq_fp8_batch_matvec_count=0 "
            "prefill_executor_request_parallelism=1 decode_executor_request_parallelism=1 "
            "final_top1_tokens_csv=151353 final_topk_tokens_csv=151353 "
            "final_topk_logits_csv=6.250000 "
            "sequence_len=3 request_count=1 concurrent_requests=1 "
            "prompt_tokens_csv=2 max_new_tokens_csv=1 total_tokens_csv=3 "
            "prefill_total_input_tokens=2 decode_total_generated_tokens=1 "
            "end_to_end_total_tokens=3 prefill_wall_ms=88.0 decode_wall_ms=12.0 "
            "final_logits_wall_ms=200.0 layer_load_ms=9000.0 total_wall_ms=300.0 "
            "outer_wall_ms=9300.0 prefill_total_input_tps=22.727273 "
            "decode_total_generated_tps=83.333333 end_to_end_total_tps=10.000000 "
            "prefill_batch_request_counts_csv=1 decode_batch_request_counts_csv=1 "
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

        self.assertEqual(metrics["artifact_load_wall_time_seconds"], 9.0)
        self.assertIsNone(metrics["artifact_materialization_wall_time_seconds"])
        self.assertEqual(metrics["load_excluded_total_wall_time_seconds"], 0.3)
        self.assertEqual(metrics["load_included_total_wall_time_seconds"], 9.3)
        self.assertEqual(row["workload"]["prompt_tokens_per_request"], [2])
        self.assertEqual(row["workload"]["first_layer_input_source"], "device_embedding")
        self.assertEqual(row["workload"]["generated_tokens_per_request"], [1])
        self.assertEqual(row["workload"]["total_context_tokens_after_prefill_per_request"], [3])
        self.assertEqual(row["workload"]["final_top1_tokens"], [151353])
        self.assertEqual(row["workload"]["final_topk_tokens"], [[151353]])
        self.assertEqual(row["workload"]["final_topk_logits"], [[6.25]])
        self.assertEqual(row["batching"]["prefill_batch_request_counts"], [1])
        self.assertTrue(row["batching"]["throughput_row"])
        self.assertFalse(row["batching"]["prefill_real_batch"])
        self.assertFalse(row["batching"]["decode_real_batch"])
        self.assertFalse(row["batching"]["prefill_request_grouped"])
        self.assertFalse(row["batching"]["decode_request_grouped"])
        self.assertFalse(row["batching"]["mixed_request_state_real_batch_projection_used"])
        self.assertEqual(row["batching"]["prefill_sq_fp8_batch_matvec_count"], 0)
        self.assertEqual(row["batching"]["decode_sq_fp8_batch_matvec_count"], 0)
        self.assertEqual(row["workload"]["prefill_sq_fp8_batch_matvec_count"], 0)
        self.assertEqual(row["workload"]["decode_sq_fp8_batch_matvec_count"], 0)

    def test_parses_model_loop_single_request_with_top_k_zero(self) -> None:
        stdout = (
            "package-token-ids-mixed-request-state-smoke "
            "package=/tmp/model.ullm.d layers=[0,1,2] layers_csv=0,1,2 input_source=embedding_token_ids "
            "prefill_mode=token_id_full_mixed_request_state "
            "request_batch_executor=true fused_request_batch=false throughput_row=true "
            "load_excluded_from_total=true final_logits_in_total=false "
            "batching_mode=single "
            "prefill_executor=mixed_request_state_layer_batch_step "
            "decode_executor=mixed_request_state_layer_batch_step "
            "prefill_real_batch=false decode_real_batch=false "
            "mixed_request_state_real_batch_projection_used=false "
            "prefill_sq_fp8_batch_matvec_count=0 decode_sq_fp8_batch_matvec_count=0 "
            "prefill_executor_request_parallelism=1 decode_executor_request_parallelism=1 "
            "final_lm_head_guard=false lm_head_top_k=0 "
            "final_top1_tokens_csv= final_topk_tokens_csv= final_topk_logits_csv= "
            "sequence_len=3 request_count=1 concurrent_requests=1 "
            "prompt_tokens_csv=2 max_new_tokens_csv=1 total_tokens_csv=3 "
            "prefill_total_input_tokens=2 decode_total_generated_tokens=1 "
            "end_to_end_total_tokens=3 prefill_wall_ms=88.0 decode_wall_ms=12.0 "
            "final_logits_wall_ms=0.0 layer_load_ms=9000.0 total_wall_ms=100.0 "
            "outer_wall_ms=9300.0 prefill_total_input_tps=22.727273 "
            "decode_total_generated_tps=83.333333 end_to_end_total_tps=10.000000 "
            "prefill_batch_request_counts_csv=1 decode_batch_request_counts_csv=1 "
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

        self.assertIsNone(metrics["artifact_materialization_wall_time_seconds"])
        self.assertEqual(metrics["final_logits_wall_time_seconds"], 0.0)
        self.assertEqual(metrics["total_wall_time_seconds"], 0.1)
        self.assertFalse(row["batching"]["final_logits_in_total"])
        self.assertEqual(row["batching"]["prefill_batch_request_counts"], [1])
        self.assertNotIn("final_top1_tokens", row["workload"])
        self.assertNotIn("final_topk_tokens", row["workload"])
        self.assertNotIn("final_topk_logits", row["workload"])

    def test_parses_model_loop_grouped_request_state_csv_fields(self) -> None:
        stdout = (
            "package-token-ids-mixed-request-state-smoke "
            "package=/tmp/model.ullm.d layers_csv=0,1,2 input_source=embedding_token_ids "
            "prefill_mode=token_id_full_mixed_request_state "
            "request_batch_executor=true fused_request_batch=false throughput_row=true "
            "load_excluded_from_total=true final_logits_in_total=true "
            "batching_mode=grouped "
            "prefill_executor=mixed_request_state_layer_batch_step "
            "decode_executor=mixed_request_state_layer_batch_step "
            "prefill_real_batch=false decode_real_batch=false "
            "mixed_request_state_real_batch_projection_used=false "
            "prefill_request_grouped=true decode_request_grouped=true "
            "prefill_grouped_request_parallelism=2 decode_grouped_request_parallelism=3 "
            "prefill_sq_fp8_batch_matvec_count=0 decode_sq_fp8_batch_matvec_count=0 "
            "prefill_executor_request_parallelism=2 decode_executor_request_parallelism=3 "
            "final_top1_tokens_csv=151353,151354 final_topk_tokens_csv=151353,151354 "
            "final_topk_logits_csv=6.250000,5.500000 "
            "sequence_len=3 request_count=2 concurrent_requests=2 "
            "prompt_tokens_csv=2,3 max_new_tokens_csv=1,2 total_tokens_csv=3,5 "
            "prefill_total_input_tokens=5 decode_total_generated_tokens=3 end_to_end_total_tokens=8 "
            "prefill_wall_ms=88.0 decode_wall_ms=12.0 final_logits_wall_ms=200.0 "
            "layer_load_ms=9000.0 total_wall_ms=300.0 outer_wall_ms=9300.0 "
            "prefill_total_input_tps=22.727273 decode_total_generated_tps=83.333333 "
            "end_to_end_total_tps=10.000000 "
            "prefill_batch_request_counts_csv=2,1 decode_batch_request_counts_csv=3,1 "
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

        self.assertEqual(row["batching"]["mode"], "grouped")
        self.assertFalse(row["batching"]["prefill_real_batch"])
        self.assertFalse(row["batching"]["decode_real_batch"])
        self.assertTrue(row["batching"]["prefill_request_grouped"])
        self.assertTrue(row["batching"]["decode_request_grouped"])
        self.assertFalse(row["batching"]["mixed_request_state_real_batch_projection_used"])
        self.assertEqual(row["batching"]["prefill_grouped_request_parallelism"], 2)
        self.assertEqual(row["batching"]["decode_grouped_request_parallelism"], 3)
        self.assertEqual(row["batching"]["prefill_sq_fp8_batch_matvec_count"], 0)
        self.assertEqual(row["batching"]["decode_sq_fp8_batch_matvec_count"], 0)
        self.assertEqual(row["workload"]["prefill_sq_fp8_batch_matvec_count"], 0)
        self.assertEqual(row["workload"]["decode_sq_fp8_batch_matvec_count"], 0)
        self.assertEqual(row["batching"]["prefill_batch_request_counts"], [2, 1])
        self.assertEqual(row["workload"]["prompt_tokens_per_request"], [2, 3])

    def test_parses_sq_stack_batch_resident_diagnostic_as_grouped(self) -> None:
        stdout = (
            "sq-fp8-package-self-attn-stack-batch-smoke "
            "package=/tmp/model.ullm.d layers=[3] layers_csv=3 "
            'layer_kinds=["self_attention"] input_source=embedding_token_ids '
            "prefill_mode=token_id_full_mixed_request_state format_id=SQ8_0 "
            "full_mixed_request_state=true request_state_dispatch=true "
            "request_batch_executor=true fused_request_batch=false throughput_row=true "
            "load_excluded_from_total=true final_logits_in_total=true "
            "sq_overlay=true sq_candidate=SQ8_0 sq_candidate_legacy=none "
            "sq_format_id=SQ8_0 sq_implementation_id=sq-fp8-w8a16-r9700-v0-layer3-full-projections "
            "sq_artifact=/tmp/sq8-layer3 sq_schema_version=sq-fp8-artifact-v0.1 "
            "sq_fp8_tensor_count=7 sq_passthrough_tensor_count=768 sq_row_chunk=256 "
            "sq_execution_mode=direct_fp8_dequant_matvec "
            "sq_projection_boundary=single+triple "
            "sq_projection_implementation_ids=single=sq8_0_matvec_r9700_direct,triple=sq8_0_matvec_triple_r9700_direct "
            "sq_projection_kernel_families=single=direct,triple=direct "
            "sq_fp8_single_matvec_count=24 sq_fp8_batch_matvec_count=0 "
            "sq_fp8_expected_all_batch_matvec_count=21 sq_fp8_pair_matvec_count=0 "
            "sq_fp8_triple_matvec_count=6 prefill_sq_fp8_batch_matvec_count=0 "
            "decode_sq_fp8_batch_matvec_count=0 batching_mode=grouped "
            "prefill_executor=mixed_request_state_layer_batch_step "
            "decode_executor=mixed_request_state_layer_batch_step "
            "prefill_real_batch=false decode_real_batch=false "
            "mixed_request_state_real_batch_projection_used=false "
            "prefill_request_grouped=true decode_request_grouped=true "
            "prefill_grouped_request_parallelism=2 decode_grouped_request_parallelism=2 "
            "prefill_executor_request_parallelism=2 decode_executor_request_parallelism=2 "
            "final_top1_tokens_csv=83620,71285 final_topk_tokens_csv=83620;71285 "
            "final_topk_logits_csv=5.254159451;4.457036018 "
            "sequence_len=3 request_count=2 concurrent_requests=2 "
            "prompt_tokens_csv=2,2 max_new_tokens_csv=1,1 total_tokens_csv=3,3 "
            "prefill_total_input_tokens=4 decode_total_generated_tokens=2 end_to_end_total_tokens=6 "
            "prefill_wall_ms=63.302781 decode_wall_ms=13.334493 "
            "final_logits_wall_ms=256.640297 layer_load_ms=1912.050748 "
            "total_wall_ms=333.277571 outer_wall_ms=2515.554313 "
            "prefill_total_input_tps=63.188377 decode_total_generated_tps=149.986955 "
            "end_to_end_total_tps=18.003012 "
            "prefill_batch_request_counts_csv=2,2 decode_batch_request_counts_csv=2 "
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

        self.assertEqual(report["command"], "sq-fp8-package-self-attn-stack-batch-smoke")
        self.assertEqual(row["workload"]["format_id"], "SQ8_0")
        self.assertEqual(
            row["workload"]["sq_execution_mode"],
            "direct_fp8_dequant_matvec",
        )
        self.assertEqual(row["workload"]["sq_projection_boundary"], "single+triple")
        self.assertEqual(
            row["workload"]["sq_projection_kernel_families"],
            "single=direct,triple=direct",
        )
        self.assertEqual(row["workload"]["sq_fp8_batch_matvec_count"], 0)
        self.assertEqual(row["workload"]["sq_fp8_expected_all_batch_matvec_count"], 21)
        self.assertEqual(row["workload"]["sq_fp8_single_matvec_count"], 24)
        self.assertEqual(row["workload"]["sq_fp8_triple_matvec_count"], 6)
        self.assertEqual(row["batching"]["mode"], "grouped")
        self.assertFalse(row["batching"]["prefill_real_batch"])
        self.assertFalse(row["batching"]["decode_real_batch"])
        self.assertFalse(row["batching"]["mixed_request_state_real_batch_projection_used"])
        self.assertTrue(row["batching"]["prefill_request_grouped"])
        self.assertTrue(row["batching"]["decode_request_grouped"])

    def test_parses_sq_stack_batch_resident_qkv_batch_as_real(self) -> None:
        stdout = (
            "sq-fp8-package-self-attn-stack-batch-smoke "
            "package=/tmp/model.ullm.d layers=[3] layers_csv=3 "
            'layer_kinds=["self_attention"] input_source=embedding_token_ids '
            "prefill_mode=token_id_full_mixed_request_state format_id=SQ8_0 "
            "full_mixed_request_state=true request_state_dispatch=true "
            "request_batch_executor=true fused_request_batch=false throughput_row=true "
            "load_excluded_from_total=true final_logits_in_total=true "
            "sq_overlay=true sq_candidate=SQ8_0 sq_candidate_legacy=none "
            "sq_format_id=SQ8_0 sq_implementation_id=sq-fp8-w8a16-r9700-v0-layer3-full-projections "
            "sq_artifact=/tmp/sq8-layer3 sq_schema_version=sq-fp8-artifact-v0.1 "
            "sq_fp8_tensor_count=7 sq_passthrough_tensor_count=768 sq_row_chunk=256 "
            "sq_execution_mode=direct_fp8_dequant_matvec "
            "sq_projection_boundary=single+batch "
            "sq_projection_implementation_ids=single=sq8_0_matvec_r9700_direct,batch=sq8_0_matvec_batch_r9700_direct "
            "sq_projection_kernel_families=single=direct,batch=direct "
            "sq_fp8_single_matvec_count=24 sq_fp8_batch_matvec_count=9 "
            "sq_fp8_expected_all_batch_matvec_count=21 sq_fp8_pair_matvec_count=0 "
            "sq_fp8_triple_matvec_count=0 prefill_sq_fp8_batch_matvec_count=6 "
            "decode_sq_fp8_batch_matvec_count=3 batching_mode=real "
            "prefill_executor=mixed_request_state_layer_batch_step "
            "decode_executor=mixed_request_state_layer_batch_step "
            "prefill_real_batch=true decode_real_batch=true "
            "mixed_request_state_real_batch_projection_used=true "
            "prefill_request_grouped=true decode_request_grouped=true "
            "prefill_grouped_request_parallelism=2 decode_grouped_request_parallelism=2 "
            "prefill_executor_request_parallelism=2 decode_executor_request_parallelism=2 "
            "final_top1_tokens_csv=83620,71285 final_topk_tokens_csv=83620;71285 "
            "final_topk_logits_csv=5.254159451;4.457036018 "
            "sequence_len=3 request_count=2 concurrent_requests=2 "
            "prompt_tokens_csv=2,2 max_new_tokens_csv=1,1 total_tokens_csv=3,3 "
            "prefill_total_input_tokens=4 decode_total_generated_tokens=2 end_to_end_total_tokens=6 "
            "prefill_wall_ms=90.886509 decode_wall_ms=17.402586 "
            "final_logits_wall_ms=251.744748 layer_load_ms=1920.270551 "
            "total_wall_ms=360.033843 outer_wall_ms=2544.641603 "
            "prefill_total_input_tps=44.010932 decode_total_generated_tps=114.925448 "
            "end_to_end_total_tps=16.665100 "
            "prefill_batch_request_counts_csv=2,2 decode_batch_request_counts_csv=2 "
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

        self.assertEqual(row["workload"]["sq_projection_boundary"], "single+batch")
        self.assertEqual(
            row["workload"]["sq_projection_kernel_families"],
            "single=direct,batch=direct",
        )
        self.assertEqual(row["workload"]["sq_fp8_batch_matvec_count"], 9)
        self.assertEqual(row["workload"]["sq_fp8_expected_all_batch_matvec_count"], 21)
        self.assertEqual(row["workload"]["sq_fp8_single_matvec_count"], 24)
        self.assertEqual(row["workload"]["sq_fp8_triple_matvec_count"], 0)
        self.assertEqual(row["batching"]["mode"], "real")
        self.assertTrue(row["batching"]["prefill_real_batch"])
        self.assertTrue(row["batching"]["decode_real_batch"])
        self.assertTrue(row["batching"]["mixed_request_state_real_batch_projection_used"])
        self.assertEqual(row["batching"]["prefill_sq_fp8_batch_matvec_count"], 6)
        self.assertEqual(row["batching"]["decode_sq_fp8_batch_matvec_count"], 3)

    def test_parses_sq_stack_batch_resident_all_batch_as_real(self) -> None:
        stdout = (
            "sq-fp8-package-self-attn-stack-batch-smoke "
            "package=/tmp/model.ullm.d layers=[3] layers_csv=3 "
            'layer_kinds=["self_attention"] input_source=embedding_token_ids '
            "prefill_mode=token_id_full_mixed_request_state format_id=SQ8_0 "
            "full_mixed_request_state=true request_state_dispatch=true "
            "request_batch_executor=true fused_request_batch=false throughput_row=true "
            "load_excluded_from_total=true final_logits_in_total=true "
            "sq_overlay=true sq_candidate=SQ8_0 sq_candidate_legacy=none "
            "sq_format_id=SQ8_0 sq_implementation_id=sq-fp8-w8a16-r9700-v0-layer3-full-projections "
            "sq_artifact=/tmp/sq8-layer3 sq_schema_version=sq-fp8-artifact-v0.1 "
            "sq_fp8_tensor_count=7 sq_passthrough_tensor_count=768 sq_row_chunk=256 "
            "sq_execution_mode=direct_fp8_dequant_matvec "
            "sq_projection_boundary=batch "
            "sq_projection_implementation_ids=batch=sq8_0_matvec_batch_r9700_direct "
            "sq_projection_kernel_families=batch=direct "
            "sq_fp8_single_matvec_count=0 sq_fp8_batch_matvec_count=21 "
            "sq_fp8_expected_all_batch_matvec_count=21 sq_fp8_pair_matvec_count=0 "
            "sq_fp8_triple_matvec_count=0 prefill_sq_fp8_batch_matvec_count=14 "
            "decode_sq_fp8_batch_matvec_count=7 batching_mode=real "
            "prefill_executor=mixed_request_state_layer_batch_step "
            "decode_executor=mixed_request_state_layer_batch_step "
            "prefill_real_batch=true decode_real_batch=true "
            "mixed_request_state_real_batch_projection_used=true "
            "prefill_request_grouped=true decode_request_grouped=true "
            "prefill_grouped_request_parallelism=2 decode_grouped_request_parallelism=2 "
            "prefill_executor_request_parallelism=2 decode_executor_request_parallelism=2 "
            "final_top1_tokens_csv=83620,71285 final_topk_tokens_csv=83620;71285 "
            "final_topk_logits_csv=5.254159451;4.457036018 "
            "sequence_len=3 request_count=2 concurrent_requests=2 "
            "prompt_tokens_csv=2,2 max_new_tokens_csv=1,1 total_tokens_csv=3,3 "
            "prefill_total_input_tokens=4 decode_total_generated_tokens=2 end_to_end_total_tokens=6 "
            "prefill_wall_ms=93.810924 decode_wall_ms=17.065428 "
            "final_logits_wall_ms=243.864610 layer_load_ms=1971.033787 "
            "total_wall_ms=354.740962 outer_wall_ms=2587.416005 "
            "prefill_total_input_tps=42.638957 decode_total_generated_tps=117.196006 "
            "end_to_end_total_tps=16.913750 "
            "prefill_batch_request_counts_csv=2,2 decode_batch_request_counts_csv=2 "
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

        self.assertEqual(row["workload"]["sq_projection_boundary"], "batch")
        self.assertEqual(
            row["workload"]["sq_projection_kernel_families"],
            "batch=direct",
        )
        self.assertEqual(row["workload"]["sq_fp8_batch_matvec_count"], 21)
        self.assertEqual(row["workload"]["sq_fp8_expected_all_batch_matvec_count"], 21)
        self.assertEqual(row["workload"]["sq_fp8_single_matvec_count"], 0)
        self.assertEqual(row["workload"]["sq_fp8_triple_matvec_count"], 0)
        self.assertEqual(row["batching"]["mode"], "real")
        self.assertTrue(row["batching"]["prefill_real_batch"])
        self.assertTrue(row["batching"]["decode_real_batch"])
        self.assertTrue(row["batching"]["mixed_request_state_real_batch_projection_used"])
        self.assertEqual(row["batching"]["prefill_sq_fp8_batch_matvec_count"], 14)
        self.assertEqual(row["batching"]["decode_sq_fp8_batch_matvec_count"], 7)

    def test_parses_sq_full_mixed_request_state_real_batch(self) -> None:
        layers_csv = ",".join(str(index) for index in range(40))
        stdout = (
            "sq-fp8-token-ids-mixed-request-state-smoke "
            "package=/tmp/qwen3.ullm.d "
            f"layers=[0, 1, 2] layers_csv={layers_csv} "
            'layer_kinds=["self_attention"] input_source=embedding_token_ids '
            "prefill_mode=token_id_full_mixed_request_state format_id=SQ8_0 "
            "full_mixed_request_state=true request_state_dispatch=true "
            "request_batch_executor=true fused_request_batch=false throughput_row=true "
            "load_excluded_from_total=true final_logits_in_total=true "
            "sq_overlay=true sq_candidate=SQ8_0 sq_candidate_legacy=none "
            "sq_format_id=SQ8_0 sq_artifact=/tmp/full-sq8 "
            "sq_schema_version=sq-fp8-artifact-v0.1 "
            "sq_fp8_tensor_count=281 sq_passthrough_tensor_count=442 sq_row_chunk=256 "
            "sq_execution_mode=direct_fp8_dequant_matvec "
            "sq_projection_boundary=batch "
            "sq_projection_implementation_ids=batch=sq8_0_matvec_batch_r9700_direct "
            "sq_projection_kernel_families=batch=direct "
            "sq_fp8_single_matvec_count=0 sq_fp8_batch_matvec_count=560 "
            "sq_fp8_expected_all_batch_matvec_count=560 sq_fp8_pair_matvec_count=0 "
            "sq_fp8_triple_matvec_count=0 prefill_sq_fp8_batch_matvec_count=280 "
            "sq_diagnostic_host_staging_read_count=960 "
            "sq_diagnostic_host_staging_write_count=880 "
            "sq_diagnostic_host_staging_read_bytes=125829120 "
            "sq_diagnostic_host_staging_write_bytes=104857600 "
            "decode_sq_fp8_batch_matvec_count=280 batching_mode=real "
            "prefill_executor=mixed_request_state_layer_batch_step "
            "decode_executor=mixed_request_state_layer_batch_step "
            "prefill_real_batch=true decode_real_batch=true "
            "mixed_request_state_real_batch_projection_used=true "
            "prefill_request_grouped=true decode_request_grouped=true "
            "prefill_grouped_request_parallelism=2 decode_grouped_request_parallelism=2 "
            "prefill_executor_request_parallelism=2 decode_executor_request_parallelism=2 "
            "final_top1_tokens_csv=220,102001 final_topk_tokens_csv=220;102001 "
            "final_topk_logits_csv=17.403524399;17.794414520 "
            "sequence_len=2 request_count=2 concurrent_requests=2 "
            "prompt_tokens_csv=1,1 max_new_tokens_csv=1,1 total_tokens_csv=2,2 "
            "prefill_total_input_tokens=2 decode_total_generated_tokens=2 end_to_end_total_tokens=4 "
            "prefill_wall_ms=1015.488141 decode_wall_ms=882.083457 "
            "final_logits_wall_ms=129259.937368 layer_load_ms=9841.676059 "
            "total_wall_ms=131157.508966 outer_wall_ms=141176.971089 "
            "prefill_total_input_tps=1.969496 decode_total_generated_tps=2.267359 "
            "end_to_end_total_tps=0.030498 "
            "prefill_batch_request_counts_csv=2 decode_batch_request_counts_csv=2 "
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
                "batch_size": 2,
                "concurrent_requests": 2,
                "kv_cache_dtype": "f32",
                "prefill_executor": None,
                "resolved_prefill_executor": None,
            },
            "metrics": metrics,
            "memory": memory.copy(),
        }
        TOOL.enrich_ullm_model_loop_row(row, report)

        self.assertEqual(row["workload"]["layers_csv"], layers_csv)
        self.assertEqual(row["workload"]["sq_fp8_tensor_count"], 281)
        self.assertEqual(row["workload"]["sq_projection_boundary"], "batch")
        self.assertEqual(
            row["workload"]["sq_projection_kernel_families"],
            "batch=direct",
        )
        self.assertEqual(row["workload"]["sq_fp8_batch_matvec_count"], 560)
        self.assertEqual(row["workload"]["sq_fp8_expected_all_batch_matvec_count"], 560)
        self.assertEqual(row["workload"]["sq_fp8_single_matvec_count"], 0)
        self.assertEqual(row["workload"]["sq_diagnostic_host_staging_read_count"], 960)
        self.assertEqual(row["workload"]["sq_diagnostic_host_staging_write_count"], 880)
        self.assertEqual(row["workload"]["sq_diagnostic_host_staging_read_bytes"], 125829120)
        self.assertEqual(row["workload"]["sq_diagnostic_host_staging_write_bytes"], 104857600)
        self.assertEqual(row["batching"]["mode"], "real")
        self.assertTrue(row["batching"]["prefill_real_batch"])
        self.assertTrue(row["batching"]["decode_real_batch"])
        self.assertEqual(row["batching"]["prefill_sq_fp8_batch_matvec_count"], 280)
        self.assertEqual(row["batching"]["decode_sq_fp8_batch_matvec_count"], 280)
        self.assertEqual(row["workload"]["final_top1_tokens"], [220, 102001])


if __name__ == "__main__":
    unittest.main()
