import importlib.util
import json
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "ci/tools/aggregate_phase50_r9700.py"
SCHEMA_PATH = ROOT / "ci/schema/phase50-r9700-summary-v1.schema.json"
spec = importlib.util.spec_from_file_location("aggregate_phase50_r9700", TOOL_PATH)
assert spec is not None and spec.loader is not None
aggregate_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate_tool)


def _sample(input_ids, output_count, base, index):
    generated = [output_count] * output_count
    start = 1_000_000 + base * 100 + index * 100
    prefill_submit = start + 1
    prefill_complete = start + 2
    first_token = start + 3
    delta = max(1, base // 4 + index)
    later = [first_token + delta * offset for offset in range(1, output_count)]
    stop = later[-1] + 1
    cleanup = stop + 1
    snapshot = lambda request, workspace: {
        "model_resident": {"current_bytes": 10, "high_water_bytes": 10},
        "request_state": {"current_bytes": request, "high_water_bytes": request},
        "workspace": {"current_bytes": workspace, "high_water_bytes": workspace},
        "current_bytes": 10 + request + workspace,
        "high_water_bytes": 10 + request + workspace,
        "poisoned": False,
    }
    return {
        "tokens": {
            "input_token_ids": input_ids,
            "generated_token_ids": generated,
            "visible_token_ids": list(generated),
            "decode_input_token_ids": generated[:-1],
        },
        "stop": {"version": 1, "reason_version": 1, "kind": "max_new_tokens", "token_id": None},
        "execution_path": "timed-production",
        "timing_instrumentation": "on",
        "audit": {
            "selected_backend": "hip",
            "target": aggregate_tool.TARGET,
            "device_index": 0,
            "model_fingerprint": "sha256:" + "f" * 64,
            "plan_digest": "sha256:" + "e" * 64,
            "fallback_used": False,
            "all_dispatches_hip": True,
            "submission_count": 1,
            "kernel_dispatch_count": 1,
            "segment_count": 1,
            "boundary_count": 1,
        },
        "memory": {
            "request_start": snapshot(1, 1),
            "after_cleanup": snapshot(0, 0),
        },
        "cleanup": {
            "sample_index": index,
            "request_dropped": True,
            "allocator_cleanup_validated": True,
            "retryable_cleanup": 0,
            "durable_quarantine": 0,
        },
        "derived": {
            "e2e_ns": cleanup - start,
            "ttft_ns": first_token - start,
            "prefill_ns": prefill_complete - prefill_submit,
            "prefill_tokens_per_second": len(input_ids) * 1_000_000_000,
            "tpot_ns": [delta] * (output_count - 1),
            "decode_tokens": output_count - 1,
            "decode_tokens_per_second": (output_count - 1) * 1_000_000_000 / max(1, later[-1] - first_token),
        },
        "events": {
            "request_start_ns": start,
            "prefill_submit_ns": prefill_submit,
            "prefill_complete_ns": prefill_complete,
            "first_token_ns": first_token,
            "later_token_publications_ns": later,
            "token_publications_ns": later,
            "stop_ns": stop,
            "cleanup_ns": cleanup,
            "cleanup_complete_ns": cleanup,
        },
    }


def _producer(engine, *, e2e_base=100, llama_e2e_base=None):
    rows = []
    for row_index, (case_id, input_count, output_count) in enumerate(aggregate_tool.CASE_SPECS):
        input_ids = aggregate_tool.input_ids_for(case_id)
        measured = 3 if case_id in aggregate_tool.EXTENDED_CASES else 10
        base = e2e_base if llama_e2e_base is None else llama_e2e_base
        samples = [_sample(input_ids, output_count, base + row_index, index) for index in range(measured)]
        row_id = f"phase50-r9700-{engine}-{case_id}"
        row = {
            "row_id": row_id,
            "model_size": "4B",
            "case_id": case_id,
            "input_token_ids": input_ids,
            "input_token_count": input_count,
            "requested_output_tokens": output_count,
            "target": aggregate_tool.TARGET,
            "gpu_uuid": aggregate_tool.GPU_UUID,
            "gpu_bdf": aggregate_tool.GPU_BDF,
            "device_index": 0,
            "warmups": 1 if case_id in aggregate_tool.EXTENDED_CASES else 3,
            "measured": measured,
            "context_length": 131_072 if case_id in aggregate_tool.EXTENDED_CASES else input_count + output_count,
            "ignore_eos": case_id == "decode-20000",
        }
        binary_sha = "a" * 64
        model_sha = "b" * 64
        if engine == "sllm":
            control_sample = samples[0]
            control = {
                "label": "correctness-reference",
                "execution_path": "first-warmup-sample",
                "timing_instrumentation": "on",
                "included_in_performance_statistics": False,
                "source": {"kind": "warmup-sample", "sample_index": 0, "request_count": 0},
                "tokens": dict(control_sample["tokens"]),
                "stop": dict(control_sample["stop"]),
                "audit": dict(control_sample["audit"]),
                "memory": dict(control_sample["memory"]),
                "cleanup": {"reference_sample": True, "request_dropped": True, "allocator_cleanup_validated": True, "retryable_cleanup": 0, "durable_quarantine": 0},
                "comparison": aggregate_tool.CONTROL_COMPARISON,
            }
            total_requests = row["warmups"] + row["measured"]
            lock_fingerprint = "sha256:" + "f" * 64
            plan_digest = "sha256:" + "e" * 64
            top_snapshot = {
                "model_resident": {"current_bytes": 10, "high_water_bytes": 10},
                "request_state": {"current_bytes": 0, "high_water_bytes": 1},
                "workspace": {"current_bytes": 0, "high_water_bytes": 1},
                "current_bytes": 10,
                "high_water_bytes": 11,
                "poisoned": False,
            }
            result = {
                "benchmark_schema_version": aggregate_tool.SLLM_DIRECT_SCHEMA,
                "state": "PASS",
                "lane": "direct",
                "lane_definition": "pretokenized direct engine: request start excludes render/tokenize",
                "row": dict(row),
                "identities": {"engine": "sllm", "backend": "hip", "session_id": 1, "device_index": 0, "target": aggregate_tool.TARGET, "model": {"model_size": "4B", "repo_id": "Qwen/Qwen3.5-4B", "resolved_revision": aggregate_tool.SLLM_MODEL_REVISION, "lock_fingerprint": lock_fingerprint}, "binding": {"model_fingerprint": lock_fingerprint, "plan_digest": plan_digest}},
                "model_load": {"event": "model_load", "start_ns": 0, "model_ready_ns": 1, "duration_ns": 1, "load_count": 1},
                "memory": {"placement_total_memory_bytes": 32 * 1024**3, "placement_available_memory_bytes": 30 * 1024**3, "placement_required_bytes": 2 * 1024**3, "placement_model_resident_bytes": 1 * 1024**3, "placement_request_state_bytes": 512 * 1024**2, "placement_safety_reserve_bytes": 256 * 1024**2, "workspace_separate_allocation_bytes": 0, "workspace_arena_bytes": 128 * 1024**2, "model_ready": top_snapshot, "after_model_drop": top_snapshot, "model_resident_high_water_bytes": 10, "resident_vram_bytes": 10, "resident_vram_source": "model_resident_allocator_high_water", "peak_vram_bytes": 11, "peak_source": "runtime_allocator"},
                "audit": {"selected_backend": "hip", "target": aggregate_tool.TARGET, "device_index": 0, "all_dispatches_hip": True, "fallback_used": False, "weight_encoding": "bf16", "model_load_count": 1, "request_model_load_count": 0, "model_reused": True, "model_fingerprint": lock_fingerprint, "plan_digest": plan_digest, "submission_count": total_requests, "kernel_dispatch_count": total_requests, "segment_count": total_requests, "boundary_count": total_requests, "sample_count": total_requests, "correctness_control_request_count": 0, "correctness_control_source": "first-warmup-sample", "correctness_control_reference_sample_index": 0, "total_request_count": total_requests},
                "cleanup": {
                    "all_requests_dropped": True,
                    "correctness_control_request_count": 0,
                    "correctness_control_source": "first-warmup-sample",
                    "correctness_control_reference_sample_index": 0,
                    "retryable_cleanup": 0,
                    "durable_quarantine": 0,
                    "warmup_request_count": row["warmups"],
                    "measured_request_count": row["measured"],
                    "request_cleanup_count": total_requests,
                    "performance_sample_count": total_requests,
                    "all_requests_dropped": True,
                },
                "config": {
                    "input_token_ids": input_ids,
                    "input_token_count": input_count,
                    "max_new_tokens": output_count,
                    "greedy": True,
                    "warmups": row["warmups"],
                    "measured": row["measured"],
                    "context_length": row["context_length"],
                    "ignore_eos": row["ignore_eos"],
                    "prefill_chunk_tokens": None,
                    "effective_prefill_chunk_tokens": min(input_count, 512),
                    "effective_context_length": row["context_length"],
                    "completion_timeout_seconds": 3600,
                    "tokenizer": False,
                    "render": False,
                    "lane": "direct",
                    "kv_cache_encoding": "fp16",
                    "stop_policy": {"stop_token_ids": [] if row["ignore_eos"] else list(aggregate_tool.STOP_IDS), "ignore_eos": row["ignore_eos"], "visible_stop_tokens": False},
                },
                "session_cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0},
                "correctness_control": control,
                "warmups": {"count": row["warmups"], "samples": samples[:row["warmups"]]},
                "measured": {"count": measured, "samples": samples},
            }
            row_schema = aggregate_tool.SLLM_ROW_SCHEMA
        else:
            result = {
                "schema_version": aggregate_tool.LLAMA_WRAPPER_SCHEMA,
                "state": "PASS",
                "llama": {"commit": aggregate_tool.LLAMA_COMMIT, "tag": aggregate_tool.LLAMA_TAG},
                "target": {"exact": aggregate_tool.TARGET, "gpu_uuid": aggregate_tool.GPU_UUID, "logical_device_index": 0},
                "model": {"format": "GGUF", "weights": "BF16", "kv": "F16"},
                "protocol": {
                    "batch_size": 1, "sequences": 1, "warmup_requests": row["warmups"], "measured_requests": row["measured"],
                    "max_new_tokens": output_count, "n_ctx": row["context_length"], "n_batch": 2048, "n_ubatch": 512,
                    "n_gpu_layers": -1, "split_mode": "none", "main_gpu": 0, "offload_kqv": True, "op_offload": True,
                    "greedy": True, "ignore_eos": row["ignore_eos"], "stop_token_ids": [] if row["ignore_eos"] else list(aggregate_tool.STOP_IDS), "bos_inserted": False,
                },
                "row_id": row_id,
                "case_id": case_id,
                "input_token_ids": input_ids,
                "row": dict(row),
                "offload_evidence": {
                    "gpu_offload_supported": True,
                    "visible_gpu_device_count": 1,
                    "selected_device": {"type": "GPU"},
                    "requested": {"n_gpu_layers": -1, "split_mode": "none", "main_gpu": 0, "offload_kqv": True, "op_offload": True},
                    "observed": {"offloaded_layers": 41, "offloadable_layers": 41},
                },
                "cleanup": {"backend_release_completed": True, "cleanup_failures": 0},
                "warmups": {"count": row["warmups"], "samples": samples[:row["warmups"]]},
                "measured": {"count": measured, "samples": samples},
            }
            row_schema = aggregate_tool.LLAMA_ROW_SCHEMA
        rows.append({"schema_version": row_schema, "state": "PASS", "target": aggregate_tool.TARGET, "gpu_uuid": aggregate_tool.GPU_UUID, "gpu_bdf": aggregate_tool.GPU_BDF, "weight": "bf16" if engine == "sllm" else None, "binary": {"sha256": binary_sha}, "model": {"sha256": model_sha}, "row": row, "process": {"capture": {"process_group_gone": True}}, "memory": {"baseline": {"hbm_bytes": 10, "gtt_bytes": 20}, "settled": {"hbm_bytes": 10, "gtt_bytes": 20, "settled": True}}, "monitor": {"samples": 1, "errors": []}, "result": result})
    schema = aggregate_tool.SLLM_SCHEMA if engine == "sllm" else aggregate_tool.LLAMA_SCHEMA
    return {
        "schema_version": schema,
        "state": "PASS",
        "target": aggregate_tool.TARGET,
        "gpu_uuid": aggregate_tool.GPU_UUID,
        "gpu_bdf": aggregate_tool.GPU_BDF,
        "llama": {"commit": aggregate_tool.LLAMA_COMMIT, "tag": aggregate_tool.LLAMA_TAG} if engine == "llama" else None,
        "binary": {"sha256": binary_sha},
        "models": {"bf16": {"sha256": model_sha}} if engine == "sllm" else None,
        "model": {"sha256": model_sha} if engine == "llama" else None,
        "matrix": {"cases": list(aggregate_tool.CASE_IDS), "row_count": 7},
        "rows": rows,
    }


def _mark_failed(producer, engine, case_index=5, kind="oom"):
    report = producer["rows"][case_index]
    report["state"] = "FAIL"
    report.pop("result", None)
    report["command"] = ["/opt/sllm/benchmark", "--case-id", report["row"]["case_id"]]
    report["failure"] = {"kind": kind, "reason": "grow virtual KV physical commitment: out of memory"}
    report["raw"] = {
        "stdout": {"path": "/evidence/stdout.json", "sha256": "c" * 64},
        "stderr": {"path": "/evidence/stderr.log", "sha256": "d" * 64},
        "monitor_tsv": {"path": "/evidence/hbm-gtt.tsv", "sha256": "e" * 64},
    }
    producer["state"] = "FAIL"
    producer["failure_count"] = 1
    producer["failures"] = [{"case_id": report["row"]["case_id"], "row_id": report["row"]["row_id"], "kind": kind, "reason": report["failure"]["reason"]}]
    return producer


class Phase50R9700SummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text())

    def test_pass_and_bounded_schema(self):
        document = aggregate_tool.aggregate_summaries(_producer("sllm"), _producer("llama"))
        self.assertEqual(document["state"], "PASS")
        self.assertTrue(document["gate"]["all_pass"])
        self.assertEqual(len(document["rows"]), 7)
        jsonschema.Draft202012Validator(self.schema).validate(document)
        self.assertNotIn("input_token_ids", json.dumps(document))
        self.assertNotIn("generated_token_ids", json.dumps(document))

    def test_summary_schema_binds_case_protocol_and_state_gate(self):
        document = aggregate_tool.aggregate_summaries(_producer("sllm"), _producer("llama"))
        invalid = json.loads(json.dumps(document))
        invalid["rows"][0]["protocol"]["context_length"] = 131072
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(self.schema).validate(invalid)
        invalid = json.loads(json.dumps(document))
        invalid["state"] = "FAIL"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(self.schema).validate(invalid)

    def test_performance_gate_failure_is_reported(self):
        document = aggregate_tool.aggregate_summaries(_producer("sllm", e2e_base=1_000), _producer("llama", e2e_base=100))
        self.assertEqual(document["state"], "FAIL")
        self.assertFalse(document["gate"]["e2e"])
        self.assertFalse(document["gate"]["all_pass"])
        jsonschema.Draft202012Validator(self.schema).validate(document)

    def test_failed_row_is_aggregated_and_peer_metrics_are_unavailable(self):
        sllm = _mark_failed(_producer("sllm"), "sllm")
        document = aggregate_tool.aggregate_summaries(sllm, _producer("llama"))
        self.assertEqual(document["state"], "FAIL")
        self.assertFalse(document["gate"]["all_pass"])
        failed = document["rows"][5]
        self.assertIsNone(failed["metrics"]["sllm"])
        self.assertIsNotNone(failed["metrics"]["llama"])
        self.assertIsNone(failed["gates"]["e2e_ns"])
        self.assertEqual(failed["failures"]["sllm"]["kind"], "oom")
        self.assertEqual(document["failure_count"], 1)
        jsonschema.Draft202012Validator(self.schema).validate(document)

    def test_engine_internal_token_mismatch_still_fails_closed(self):
        sllm = _producer("sllm")
        sllm["rows"][0]["result"]["measured"]["samples"][0]["tokens"]["generated_token_ids"][0] += 1
        with self.assertRaises(aggregate_tool.Phase50Error):
            aggregate_tool.aggregate_summaries(sllm, _producer("llama"))

    def test_cross_engine_token_mismatch_is_observed_not_gated(self):
        llama = _producer("llama")
        # The fixture aliases the first warmups to measured samples, so one
        # measured-only pass updates every aliased sample exactly once.
        for sample in llama["rows"][0]["result"]["measured"]["samples"]:
            generated = sample["tokens"]["generated_token_ids"]
            generated[0] += 1
            sample["tokens"]["visible_token_ids"] = list(generated)
            sample["tokens"]["decode_input_token_ids"] = list(generated[:-1])
        document = aggregate_tool.aggregate_summaries(_producer("sllm"), llama)
        row = document["rows"][0]
        self.assertEqual(document["state"], "PASS")
        self.assertTrue(document["gate"]["all_pass"])
        self.assertFalse(row["tokens"]["generated_equal"])
        self.assertFalse(row["tokens"]["visible_equal"])
        self.assertTrue(row["tokens"]["stop_equal"])
        jsonschema.Draft202012Validator(self.schema).validate(document)

    def test_cross_engine_stop_mismatch_is_observed_not_gated(self):
        llama = _producer("llama")
        for group in ("warmups", "measured"):
            for sample in llama["rows"][0]["result"][group]["samples"]:
                sample["stop"] = {"version": 1, "reason_version": 1, "kind": "stop_token", "token_id": 248046}
        document = aggregate_tool.aggregate_summaries(_producer("sllm"), llama)
        row = document["rows"][0]
        self.assertEqual(document["state"], "PASS")
        self.assertTrue(row["tokens"]["generated_equal"])
        self.assertTrue(row["tokens"]["visible_equal"])
        self.assertFalse(row["tokens"]["stop_equal"])
        jsonschema.Draft202012Validator(self.schema).validate(document)

    def test_missing_row_fails_closed(self):
        llama = _producer("llama")
        llama["rows"].pop()
        with self.assertRaises(aggregate_tool.Phase50Error):
            aggregate_tool.aggregate_summaries(_producer("sllm"), llama)

    def test_nonfinite_metric_fails_closed(self):
        sllm = _producer("sllm")
        sllm["rows"][0]["result"]["measured"]["samples"][0]["derived"]["e2e_ns"] = float("nan")
        with self.assertRaises(aggregate_tool.Phase50Error):
            aggregate_tool.aggregate_summaries(sllm, _producer("llama"))

    def test_stale_separate_control_fails_closed(self):
        sllm = _producer("sllm")
        sllm["rows"][0]["result"]["correctness_control"]["label"] = "correctness-only"
        with self.assertRaises(aggregate_tool.Phase50Error):
            aggregate_tool.aggregate_summaries(sllm, _producer("llama"))

    def test_wrong_warmup_count_fails_closed(self):
        sllm = _producer("sllm")
        sllm["rows"][0]["row"]["warmups"] = 1
        with self.assertRaises(aggregate_tool.Phase50Error):
            aggregate_tool.aggregate_summaries(sllm, _producer("llama"))

    def test_wrong_protocol_metadata_fails_closed(self):
        llama = _producer("llama")
        llama["rows"][0]["result"]["protocol"]["warmup_requests"] = 1
        with self.assertRaises(aggregate_tool.Phase50Error):
            aggregate_tool.aggregate_summaries(_producer("sllm"), llama)

    def test_inconsistent_lock_fingerprint_fails_closed(self):
        sllm = _producer("sllm")
        sllm["rows"][1]["result"]["identities"]["model"]["lock_fingerprint"] = "different-lock"
        with self.assertRaises(aggregate_tool.Phase50Error):
            aggregate_tool.aggregate_summaries(sllm, _producer("llama"))

    def test_missing_direct_timing_contract_fails_closed(self):
        sllm = _producer("sllm")
        sllm["rows"][0]["result"]["measured"]["samples"][0].pop("events")
        with self.assertRaises(aggregate_tool.Phase50Error):
            aggregate_tool.aggregate_summaries(sllm, _producer("llama"))

    def test_short_tpot_distribution_fails_closed(self):
        sllm = _producer("sllm")
        sllm["rows"][0]["result"]["measured"]["samples"][0]["derived"]["tpot_ns"] = [1]
        with self.assertRaises(aggregate_tool.Phase50Error):
            aggregate_tool.aggregate_summaries(sllm, _producer("llama"))

    def test_hbm_gtt_cleanup_mismatch_fails_closed(self):
        sllm = _producer("sllm")
        sllm["rows"][0]["memory"]["settled"]["hbm_bytes"] += 1
        with self.assertRaises(aggregate_tool.Phase50Error):
            aggregate_tool.aggregate_summaries(sllm, _producer("llama"))

    def test_token_id_upper_bound_fails_closed(self):
        sllm = _producer("sllm")
        sllm["rows"][0]["result"]["measured"]["samples"][0]["tokens"]["generated_token_ids"][0] = 248320
        with self.assertRaises(aggregate_tool.Phase50Error):
            aggregate_tool.aggregate_summaries(sllm, _producer("llama"))

    def test_llama_stop_reason_version_is_required(self):
        llama = _producer("llama")
        llama["rows"][0]["result"]["measured"]["samples"][0]["stop"].pop("reason_version")
        with self.assertRaises(aggregate_tool.Phase50Error):
            aggregate_tool.aggregate_summaries(_producer("sllm"), llama)


if __name__ == "__main__":
    unittest.main()
