from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci" / "tools"))

import phase36_session_d_profile as profile  # noqa: E402
import run_phase36_session_d as runner  # noqa: E402


def _write(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _derived(input_count: int, output_count: int, index: int) -> dict[str, object]:
    return {
        "ttft_ns": 100 + index,
        "prefill_ns": 200 + index,
        "prefill_tokens_per_second": 1_000_000.0 + index,
        "tpot_ns": [300 + index] * max(1, output_count - 1),
        "decode_tokens": output_count - 1,
        "decode_tokens_per_second": 500_000.0 + index,
        "e2e_ns": 600 + index,
    }


def _sample(input_ids: list[int], output_ids: list[int], output_count: int, index: int) -> dict[str, object]:
    return {
        "tokens": {"input_token_ids": input_ids, "generated_token_ids": output_ids, "visible_token_ids": output_ids},
        "derived": _derived(len(input_ids), output_count, index),
        "audit": {"selected_backend": "hip", "target": "gfx942", "all_dispatches_hip": True, "fallback_used": False},
        "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0, "all_requests_dropped": True},
    }


class SessionDSummaryTests(unittest.TestCase):
    def _fixture(self) -> tuple[tempfile.TemporaryDirectory[str], dict[str, object]]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        binary = root / "sllm"; binary_sha = _write(binary, b"sllm-binary")
        llama_binary = root / "llama"; llama_binary_sha = _write(llama_binary, b"llama-wrapper")
        bf16_model = root / "bf16.gguf"; bf16_sha = _write(bf16_model, b"bf16-model")
        fp8_model = root / "fp8.gguf"; fp8_sha = _write(fp8_model, b"fp8-model")
        llama_model = root / "llama-bf16.gguf"; llama_model_sha = _write(llama_model, b"llama-bf16-model")
        source_fingerprint = "sha256:" + "f" * 64
        bf16_lock = root / "bf16.lock"; bf16_lock_sha = _write(bf16_lock, json.dumps({"source_lock_fingerprints": [source_fingerprint]}).encode())
        fp8_lock = root / "fp8.lock"; fp8_lock_sha = _write(fp8_lock, b"fp8-lock")
        source = root / "source.json"
        source.write_text(json.dumps({"schema_version": "phase36-session-d-source-identity-v1", "base_commit": runner.SOURCE_BASE_COMMIT, "base_tree": runner.SOURCE_BASE_TREE, "session_d_cli_overrides": {"crates/sllm-cli/src/benchmark.rs": "a" * 64, "crates/sllm-cli/src/main.rs": "b" * 64, "crates/sllm-cli/src/model.rs": "c" * 64}, "sllm_binary_sha256": binary_sha, "build": {"rocm_root": runner.ROCM_ROOT, "rocm_version": runner.ROCM_VERSION, "hip_compiler": runner.ROCM_ROOT + "/bin/amdclang++", "logical_target": "gfx942"}}), encoding="utf-8")
        llama_manifest = root / "llama-manifest.json"
        llama_manifest.write_text(json.dumps({"schema_version": "phase5-p3-llama-cpp-artifacts-v1", "model": {"repo_id": "Qwen/Qwen3.5-4B", "resolved_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "lock_fingerprint": source_fingerprint}, "conversion": {"run": {"args": ["convert_hf_to_gguf.py", "--outtype", "bf16", "--no-mtp"], "result": "PASS", "output_sha256": llama_model_sha, "output_size_bytes": llama_model.stat().st_size}}}), encoding="utf-8")
        raw = root / "raw"; raw.mkdir()

        def sllm_row(weight: str, case: str, input_count: int, output_count: int, index: int) -> dict[str, object]:
            ids = runner._expected_input_ids(case)
            outputs = [23066, 23066] if case == "long-10001" else [7] * output_count
            row_id = f"phase36-d-{weight}-{case}"
            row_dir = raw / row_id; row_dir.mkdir()
            stdout = row_dir / "stdout.json"; stderr = row_dir / "stderr.log"; monitor = row_dir / "hbm-gtt.tsv"
            stdout_sha = _write(stdout, b"{}")
            stderr_sha = _write(stderr, b"")
            monitor_sha = _write(monitor, b"timestamp_ns\thbm_bytes\tgtt_bytes\n1\t1000\t10\n2\t1200\t10\n")
            samples = [_sample(ids, outputs, output_count, i) for i in range(10)]
            result = {
                "benchmark_schema_version": "engine-performance-direct-v1", "state": "PASS", "lane": "direct",
                "config": {"input_token_ids": ids, "input_token_count": input_count, "max_new_tokens": output_count, "warmups": 3, "measured": 10, "greedy": True, "kv_cache_encoding": "fp16", "lane": "direct", "render": False, "tokenizer": False},
                "audit": {"selected_backend": "hip", "target": "gfx942", "all_dispatches_hip": True, "fallback_used": False},
                "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0, "all_requests_dropped": True},
                "measured": {"count": 10, "samples": samples},
                "memory": {"resident_vram_bytes": 8000, "peak_vram_bytes": 9000},
            }
            report = {"schema_version": "phase36-session-d-performance-row-v1", "state": "PASS", "gpu_uuid": runner.GPU_UUID, "device_index": 0, "row": {"row_id": row_id, "weight": weight, "model_size": "4B", "case_id": case, "input_token_ids": ids, "input_token_count": input_count, "requested_output_tokens": output_count, "target": "gfx942", "device_index": 0}, "model": {"sha256": bf16_sha if weight == "bf16" else fp8_sha}, "lock": {"sha256": bf16_lock_sha if weight == "bf16" else fp8_lock_sha}, "result": result, "memory": {"baseline": {"hbm_bytes": 1000, "gtt_bytes": 10}, "settled": {"hbm_bytes": 1000, "gtt_bytes": 10, "settled": True}}, "raw": {"stdout": {"path": str(stdout), "sha256": stdout_sha}, "stderr": {"path": str(stderr), "sha256": stderr_sha}, "monitor_tsv": {"path": str(monitor), "sha256": monitor_sha}}, "monitor": {"cadence_ms": 100, "samples": 2, "errors": [], "tsv": str(monitor)}}
            (row_dir / "row.json").write_text(json.dumps(report), encoding="utf-8")
            return report

        specs = {"short-odd": (17, 17), "32-32": (32, 32), "prefill-long": (1024, 128), "decode-long": (32, 256), "long-10001": (10001, 2)}
        rows = [sllm_row(weight, case, *specs[case], index) for index, (weight, case) in enumerate((w, c) for w in ("bf16", "fp8") for c in runner.CASES)]
        sllm = {"schema_version": "phase36-session-d-performance-v1", "state": "PASS", "target": "gfx942", "gpu_uuid": runner.GPU_UUID, "device_index": 0, "model_size": "4B", "protocol": {"warmups": 3, "measured": 10, "greedy": True, "kv_cache_encoding": "fp16"}, "matrix": {"weights": list(runner.WEIGHTS), "cases": list(runner.CASES), "row_count": 10}, "rows": rows}
        sllm_path = root / "sllm.json"; sllm_path.write_text(json.dumps(sllm), encoding="utf-8")

        llama_dir = root / "llama-output"; llama_dir.mkdir()
        llama_raw = llama_dir / "raw"; llama_raw.mkdir()
        llama_model_identity = {"path": str(llama_model), "size_bytes": llama_model.stat().st_size, "sha256": llama_model_sha}
        llama_reports = []
        for case, (input_count, output_count) in specs.items():
            producer_case = "32x32" if case == "32-32" else case
            ids = runner._expected_input_ids(producer_case)
            outputs = [23066, 23066] if case == "long-10001" else [7] * output_count
            samples = []
            for index in range(10):
                sample = _sample(ids, outputs, output_count, index)
                samples.append(sample)
            doc = {"schema_version": "llama-phase36-session-d-v1", "record_kind": "result", "state": "PASS", "llama": {"commit": runner.LLAMA_COMMIT, "tag": runner.LLAMA_TAG}, "target": {"exact": "gfx942", "gpu_uuid": runner.GPU_UUID, "logical_device_index": 0}, "row_id": f"llama-{producer_case}", "case_id": producer_case, "input_token_ids": ids, "model": {"sha256": llama_model_sha, "format": "GGUF", "weights": "BF16", "kv": "F16"}, "protocol": {"warmup_requests": 3, "measured_requests": 10, "max_new_tokens": output_count, "n_ctx": input_count + output_count, "greedy": True, "n_batch": 10001, "n_ubatch": 512, "offload_kqv": True, "op_offload": True}, "warmups": {"count": 3, "samples": []}, "measured": {"count": 10, "samples": samples}, "offload_evidence": {"visible_gpu_device_count": 1, "observed": {"device_memory": {"observed_decrease_bytes": 7000}}}, "cleanup": {"backend_release_completed": True, "cleanup_failures": 0}, "audit": {"full_gpu_offload": True, "errors": []}}
            row_dir = llama_raw / case; row_dir.mkdir()
            stdout = row_dir / "stdout.json"; stderr = row_dir / "stderr.log"; monitor = row_dir / "hbm-gtt.tsv"
            stdout_sha = _write(stdout, json.dumps(doc).encode())
            stderr_sha = _write(stderr, b"")
            monitor_sha = _write(monitor, b"timestamp_ns\thbm_bytes\tgtt_bytes\n1\t1100\t10\n2\t1400\t10\n")
            report = {"schema_version": "phase36-session-d-llama-row-v1", "state": "PASS", "row": {"row_id": f"llama-{producer_case}", "case_id": producer_case, "input_token_ids": ids, "input_token_count": input_count, "requested_output_tokens": output_count, "model_sha256": llama_model_sha}, "model": llama_model_identity, "target": "gfx942", "gpu_uuid": runner.GPU_UUID, "memory": {"baseline": {"hbm_bytes": 1100, "gtt_bytes": 10}, "settled": {"hbm_bytes": 1100, "gtt_bytes": 10, "settled": True}}, "monitor": {"cadence_ms": 100, "samples": 2, "errors": []}, "raw": {"stdout": {"path": str(stdout), "sha256": stdout_sha}, "stderr": {"path": str(stderr), "sha256": stderr_sha}, "monitor_tsv": {"path": str(monitor), "sha256": monitor_sha}}, "result": doc}
            (row_dir / "row.json").write_text(json.dumps(report), encoding="utf-8")
            # Keep the producer summary as the aggregator's single llama input.
            llama_reports.append(report)
        llama_summary = {"schema_version": "phase36-session-d-llama-v1", "state": "PASS", "target": "gfx942", "gpu_uuid": runner.GPU_UUID, "llama": {"commit": runner.LLAMA_COMMIT, "tag": runner.LLAMA_TAG}, "protocol": {"warmups": 3, "measured": 10, "batch_size": 1, "n_batch": 10001, "n_ubatch": 512, "weights": "BF16", "kv": "F16"}, "matrix": {"cases": list(runner.LLAMA_CASES), "row_count": 5}, "binary": {"path": str(llama_binary), "size_bytes": llama_binary.stat().st_size, "sha256": llama_binary_sha}, "model": llama_model_identity, "rows": llama_reports}
        llama_summary_path = llama_dir / "phase36-session-d-llama-v1.json"
        llama_summary_path.write_text(json.dumps(llama_summary), encoding="utf-8")

        profile_raw = root / "profile-raw"; profile_raw.mkdir()
        raw_map = {}
        for name in ("kernel_stats", "kernel_trace", "hip_api_stats", "memory_copy_stats", "execution_json"):
            path = profile_raw / f"case_{name}.csv"; digest = _write(path, (name + "\n").encode()); raw_map[name] = {"path": path.name, "sha256": digest}
        categories = []
        for category, duration in zip(profile.CATEGORIES, (400, 300, 200, 100)):
            categories.append({"category": category, "calls": 1, "total_duration_ns": duration, "device_time_share": duration / 1000.0, "kernel_names": [category + "_kernel"]})
        profile_doc = {"schema_version": profile.SCHEMA_VERSION, "state": "PASS", "target": "gfx942", "kernel": {"calls": 4, "trace_dispatches": 4, "total_duration_ns": 1000, "categories": categories, "category_share_sum": 1.0}, "kernel_external": {"state": "available", "host_wall_ns": 2000, "kernel_interval_union_ns": 1000, "external_ns": 1000}, "raw_sha256": raw_map, "raw_manifest_sha256": runner._sha_json(raw_map)}
        profile_path = root / "profile.json"; profile_path.write_text(json.dumps(profile_doc), encoding="utf-8")
        phase12 = root / "phase12.json"; phase12.write_text(json.dumps({"performance": {"rows": []}}), encoding="utf-8")
        phase35 = root / "phase35.json"; phase35.write_text(json.dumps({"state": "COMPLETE", "performance": {"combined_final_source": {"gfx1030": {"candidate_ns": 1}, "gfx1201": {"candidate_ns": 2}}}}), encoding="utf-8")
        args = {"sllm_summary": sllm_path, "llama_summary": llama_summary_path, "profile_summary": profile_path, "profile_raw_dir": profile_raw, "binary": binary, "llama_binary": llama_binary, "bf16_model": bf16_model, "bf16_lock": bf16_lock, "fp8_model": fp8_model, "fp8_lock": fp8_lock, "llama_model": llama_model, "llama_model_manifest": llama_manifest, "source": source, "rocm_root": runner.ROCM_ROOT, "rocm_version": runner.ROCM_VERSION, "gpu_uuid": runner.GPU_UUID, "phase12_summary": phase12, "phase35_summary": phase35}
        return temporary, args

    def test_pass_and_schema(self) -> None:
        temporary, args = self._fixture(); self.addCleanup(temporary.cleanup)
        summary = runner.aggregate_session_d(**args)
        schema = json.loads((ROOT / "ci/schema/phase36-mi300x-session-d-summary-v1.schema.json").read_text())
        errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(summary))
        self.assertEqual(errors, [])
        self.assertEqual(len(summary["sllm"]["rows"]), 10)
        self.assertEqual(len(summary["llama"]["rows"]), 5)
        self.assertEqual(summary["comparisons"]["e1_bf16"]["classification"], "E1_SYSTEM_EQUIVALENT")
        self.assertEqual([row["target"] for row in summary["historical"]["phase35_changes"]], ["gfx1030", "gfx1201"])
        self.assertEqual(summary["historical"]["phase35_changes"][0]["phase35_value"], 1.0)
        self.assertGreater(summary["historical"]["phase35_changes"][0]["ratio_current_over_phase35"], 1.0)

    def test_missing_duplicate_wrong_target_sample_fallback_cleanup_rejected(self) -> None:
        temporary, args = self._fixture(); self.addCleanup(temporary.cleanup)
        missing = copy.deepcopy(args); missing["llama_summary"] = Path(temporary.name) / "missing"
        with self.assertRaisesRegex(runner.SessionDError, "sLLM performance|llama summary"):
            runner.aggregate_session_d(**missing)
        duplicate = copy.deepcopy(args)
        llama_doc = json.loads(duplicate["llama_summary"].read_text())
        llama_doc["rows"].append(copy.deepcopy(llama_doc["rows"][0]))
        duplicate["llama_summary"].write_text(json.dumps(llama_doc), encoding="utf-8")
        with self.assertRaisesRegex(runner.SessionDError, "rows are missing or duplicated"):
            runner.aggregate_session_d(**duplicate)
        wrong_target = copy.deepcopy(args); p = wrong_target["sllm_summary"]; d = json.loads(p.read_text()); d["target"] = "gfx1201"; p.write_text(json.dumps(d))
        with self.assertRaisesRegex(runner.SessionDError, "target"):
            runner.aggregate_session_d(**wrong_target)

    def test_sample_drift_fallback_cleanup_and_raw_digest_rejected(self) -> None:
        for mutation, message in (("sample", "sample input"), ("fallback", "fallback"), ("cleanup", "cleanup"), ("digest", "raw SHA")):
            temporary, args = self._fixture(); self.addCleanup(temporary.cleanup)
            data = json.loads(args["sllm_summary"].read_text())
            row = data["rows"][0]
            if mutation == "sample": row["result"]["measured"]["samples"][0]["tokens"]["input_token_ids"][0] = 999
            if mutation == "fallback": row["result"]["audit"]["fallback_used"] = True
            if mutation == "cleanup": row["result"]["cleanup"]["retryable_cleanup"] = 1
            if mutation == "digest": row["raw"]["stdout"]["sha256"] = "0" * 64
            args["sllm_summary"].write_text(json.dumps(data), encoding="utf-8")
            with self.subTest(message=message), self.assertRaisesRegex(runner.SessionDError, message):
                runner.aggregate_session_d(**args)

    def test_e1_strict_identical_and_profile_closure_rejected(self) -> None:
        temporary, args = self._fixture(); self.addCleanup(temporary.cleanup)
        data = json.loads(args["profile_summary"].read_text()); data["kernel_external"]["external_ns"] = 2; args["profile_summary"].write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(runner.SessionDError, "profile kernel_external"):
            runner.aggregate_session_d(**args)

    def test_profile_digest_and_e1_classification_are_not_synthesized(self) -> None:
        temporary, args = self._fixture(); self.addCleanup(temporary.cleanup)
        data = json.loads(args["profile_summary"].read_text()); data["raw_manifest_sha256"] = "0" * 64; args["profile_summary"].write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(runner.SessionDError, "manifest digest"):
            runner.aggregate_session_d(**args)

    def test_e1_strict_identical_label_is_rejected(self) -> None:
        temporary, args = self._fixture(); self.addCleanup(temporary.cleanup)
        llama_summary = json.loads(args["llama_summary"].read_text())
        llama_summary["rows"][0]["result"]["comparison_class"] = "STRICT_IDENTICAL"
        args["llama_summary"].write_text(json.dumps(llama_summary), encoding="utf-8")
        with self.assertRaisesRegex(runner.SessionDError, "mislabeled"):
            runner.aggregate_session_d(**args)

    def test_exact_uuid_rocm_source_and_manifest_are_required(self) -> None:
        for mutation, message in (("uuid", "exact gfx942"), ("rocm", "exact ROCm tuple"), ("source", "not bound"), ("manifest", "BF16 no-MTP")):
            temporary, args = self._fixture(); self.addCleanup(temporary.cleanup)
            if mutation == "uuid":
                args["gpu_uuid"] = "GPU-wrong"
            elif mutation == "rocm":
                args["rocm_root"] = "/opt/rocm/core-7.14"
            elif mutation == "source":
                document = json.loads(args["source"].read_text()); document["sllm_binary_sha256"] = "0" * 64; args["source"].write_text(json.dumps(document))
            else:
                document = json.loads(args["llama_model_manifest"].read_text()); document["conversion"]["run"]["args"].remove("--no-mtp"); args["llama_model_manifest"].write_text(json.dumps(document))
            with self.subTest(mutation=mutation), self.assertRaisesRegex(runner.SessionDError, message):
                runner.aggregate_session_d(**args)

    def test_case_shape_monitor_and_profile_share_drift_are_rejected(self) -> None:
        for mutation, message in (("shape", "row identity"), ("monitor", "monitor errors/cadence"), ("share", "device share")):
            temporary, args = self._fixture(); self.addCleanup(temporary.cleanup)
            if mutation in {"shape", "monitor"}:
                document = json.loads(args["sllm_summary"].read_text())
                if mutation == "shape": document["rows"][0]["row"]["requested_output_tokens"] = 18
                else: document["rows"][0]["monitor"]["errors"] = ["observer failed"]
                args["sllm_summary"].write_text(json.dumps(document))
            else:
                document = json.loads(args["profile_summary"].read_text()); document["kernel"]["categories"][0]["device_time_share"] = 0.5; args["profile_summary"].write_text(json.dumps(document))
            with self.subTest(mutation=mutation), self.assertRaisesRegex(runner.SessionDError, message):
                runner.aggregate_session_d(**args)


if __name__ == "__main__":
    unittest.main()
