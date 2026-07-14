from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/workloads/aq4-production-opt-p2-case-manifest-v0.1.json"
POLICY = ROOT / "benchmarks/workloads/aq4-production-opt-p2-threshold-policy-template-v0.1.json"
PLAN_VALIDATE = ROOT / "benchmarks/workloads/validate-aq4-production-opt-p2-manifest.py"
EXPAND = ROOT / "tools/expand-aq4-production-p2.py"
BIND = ROOT / "tools/bind-aq4-production-p2-identity.py"
RUN = ROOT / "tools/run-aq4-production-p2.py"
BUILD = ROOT / "tools/build-aq4-prefill-validation-result.py"
VALIDATE = ROOT / "tools/validate-aq4-production-p2-evidence.py"


def invoke(tool: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["python3", str(tool), *args], cwd=ROOT, capture_output=True, text=True)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


class Aq4ProductionP2EvidenceTests(unittest.TestCase):
    def expand(self, root: Path, manifest: Path = MANIFEST) -> tuple[Path, dict]:
        output = root / "expanded.json"
        completed = invoke(EXPAND, "--manifest", str(manifest), "--output", str(output))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return output, json.loads(output.read_text())

    def bind(self, root: Path, *, worker_body: str = "#!/bin/sh\nprintf ok", large_package: bool = False) -> dict[str, Path | dict]:
        expanded_path, expanded = self.expand(root)
        package = root / "package"; package.mkdir()
        weights = package / "weights.bin"
        if large_package:
            with weights.open("wb") as target:
                target.seek(65 * 1024 * 1024); target.write(b"x")
        else: weights.write_bytes(b"synthetic-package")
        package_manifest = package / "package-manifest.json"; write_json(package_manifest, {"schema": "synthetic"})
        worker = root / "worker"; worker.write_text(worker_body, encoding="utf-8"); worker.chmod(0o755)
        tokenizer = root / "tokenizer.json"; write_json(tokenizer, {"tokenizer": "synthetic"})
        served = root / "served-model.json"; write_json(served, {"model": "Qwen3.5-9B"})
        model = root / "model-identity.json"; write_json(model, {"id": "synthetic-qwen35", "revision": "fixture", "format_id": "AQ4_0", "implementation_id": "synthetic"})
        graph = root / "graph.json"; write_json(graph, {"source": "fixture", "nodes": 1})
        state_schema = root / "state-schema.json"; write_json(state_schema, {"schema": "fixture-state-v1"})
        source = root / "source-oracle.json"; write_json(source, {"schema_version": "ullm.qwen35_aq4_source_oracle.v1", "oracle_kind": "independent_source", "status": "fixture", "evidence_class": "synthetic_fixture", "promotion_eligible": False})
        source_validation = root / "source-oracle-validation.json"; write_json(source_validation, {"schema_version": "ullm.qwen35_aq4_p2_oracle_validator.v1", "status": "valid", "oracle_kind": "independent_source", "manifest_sha256": sha(source), "production_eligible": False, "blockers": ["synthetic fixture"]})
        power = root / "power.json"; write_json(power, {"policy_binding": {"expected_power_limit_watts": 300, "allowed_power_tolerance_watts": 5, "maximum_temperature_c": 95, "minimum_vram_headroom_bytes": 1}})
        correctness = root / "correctness.json"; write_json(correctness, {"max_hidden_relative_l2": 1.0, "max_hidden_max_abs": 1.0, "max_logits_relative_l2": 1.0, "max_logits_max_abs": 1.0, "minimum_top_k_overlap": 1})
        baseline = root / "baseline.json"; write_json(baseline, {"prefill_tokens_per_second_p50": 100.0, "prefill_tokens_per_second_p95": 100.0, "oom": False})
        identity = root / "identity.json"; bound_policy = root / "bound-policy.json"
        completed = invoke(BIND, "--manifest", str(MANIFEST), "--expanded", str(expanded_path), "--policy", str(POLICY), "--worker", str(worker), "--package-root", str(package), "--package-manifest", str(package_manifest), "--tokenizer", str(tokenizer), "--served-model-manifest", str(served), "--model-identity", str(model), "--graph", str(graph), "--state", str(state_schema), "--source-oracle", str(source), "--power-capture", str(power), "--correctness-thresholds", str(correctness), "--baseline-result", str(baseline), "--effective-at", "2026-07-14T12:00:00Z", "--git-commit", "a" * 40, "--output", str(identity), "--bound-policy", str(bound_policy))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return {"expanded_path": expanded_path, "expanded": expanded, "package": package, "package_manifest": package_manifest, "worker": worker, "source": source, "source_validation": source_validation, "identity": identity, "policy": bound_policy, "baseline": baseline}

    def case(self, bound: dict[str, Path | dict], root: Path, predicate) -> tuple[Path, dict]:
        expanded = bound["expanded"]
        assert isinstance(expanded, dict)
        value = next(item for item in expanded["cases"] if predicate(item))
        path = root / f"{value['case_id']}.case.json"; write_json(path, value)
        return path, value

    def raw(self, bound: dict[str, Path | dict], root: Path, case_path: Path, case: dict, *, preflight: dict | None = None, lock: Path | None = None, mode: str = "cpu_synthetic", trace: Path | None = None, limit: int = 262144) -> tuple[Path, subprocess.CompletedProcess[str]]:
        measurement = root / f"{case['case_id']}.measurement.json"
        row = {"prefill_ms": 10.0, "ttft_ms": 11.0, "decode_ms": 5.0, "inter_token_latency_ms": 1.0, "end_to_end_ms": 20.0, "vram_peak_bytes": 100, "workspace_peak_bytes": 10, "actual_token_batch_width": max(1, case["resolved_m"]), "actual_request_batch_width": 1}
        write_json(measurement, {"schema_version": "ullm.aq4_p2_measurements.v1", "case_id": case["case_id"], "warmup_runs": [row, row], "measured_runs": [row for _ in range(10)]})
        state = root / f"{case['case_id']}.state.json"
        write_json(state, {"schema_version": "ullm.aq4_p2_state_evidence.v1", "case_id": case["case_id"], "status": "valid", "checks": {field: True for field in ("finite_outputs", "shape_contract_passed", "kv_state_cache_passed", "scheduler_progress_passed", "chunk_equivalence_passed", "cancel_reset_passed", "publish_failure_reset_passed")}, "reset": {"attempted": True, "complete": True, "failed": False}, "fallback": {"unexpected_count": 0, "fail_closed_count": 0, "unsupported_count": 0, "reasons": []}, "memory": {"oom": None, "headroom_bytes": 100, "observed_peak_bytes": 100, "workspace_peak_bytes": 10}})
        if preflight is None: preflight = {"weights_bytes": 1, "persistent_state_bytes": 1, "kv_cache_bytes": 1, "workspace_bytes": 1, "temporary_bytes": 1, "vram_headroom_bytes": 100, "gpu_process_snapshot": []}
        preflight_path = root / f"{case['case_id']}.preflight.json"; write_json(preflight_path, preflight)
        output = root / f"{case['case_id']}.raw.json"
        command = ["--run-root", str(root), "--case", str(case_path), "--expanded", str(bound["expanded_path"]), "--identity", str(bound["identity"]), "--policy", str(bound["policy"]), "--preflight", str(preflight_path), "--measurement", str(measurement), "--state", str(state), "--executable", str(bound["worker"]), "--package-root", str(bound["package"]), "--mode", mode, "--max-output-bytes", str(limit), "--output", str(output)]
        if lock: command += ["--lock", str(lock)]
        if trace: command += ["--trace", str(trace)]
        completed = invoke(RUN, *command)
        return output, completed

    def result(self, bound: dict[str, Path | dict], root: Path, case_path: Path, case: dict, raw: Path, *, path_result: Path | None = None, trace: Path | None = None, trace_bundle: dict[str, Path] | None = None) -> tuple[Path, subprocess.CompletedProcess[str]]:
        independent = root / f"{case['case_id']}.independent.json"
        correctness = {field: True for field in ("finite", "shape_contract_passed", "source_oracle_passed", "greedy_tokens_exact", "kv_state_cache_passed", "scheduler_progress_passed", "chunk_equivalence_passed", "cancel_reset_passed", "publish_failure_reset_passed")}
        correctness["path_oracle_passed"] = case["mode"] != "all_m1"
        correctness["final_hidden"] = {"relative_l2": 0.0, "max_abs": 0.0}; correctness["logits"] = {"relative_l2": 0.0, "max_abs": 0.0, "top_k_overlap": 1}
        write_json(independent, {"schema_version": "ullm.aq4_p2_independent_validation.v1", "status": "valid", "validator_independent": True, "case_id": case["case_id"], "case_sha256": case["case_sha256"], "raw_sha256": sha(raw), "source_oracle_sha256": sha(bound["source"]), "path_oracle_result_sha256": sha(path_result) if path_result else None, "trace_sha256": sha(trace) if trace else None, "correctness": correctness})
        output = root / f"{case['case_id']}.result.json"
        command = ["--run-root", str(root), "--case", str(case_path), "--expanded", str(bound["expanded_path"]), "--raw", str(raw), "--identity", str(bound["identity"]), "--policy", str(bound["policy"]), "--source-oracle", str(bound["source"]), "--source-oracle-validation", str(bound["source_validation"]), "--independent-validation", str(independent), "--output", str(output)]
        if path_result: command += ["--path-oracle-result", str(path_result)]
        if trace:
            command += ["--trace", str(trace)]
            if trace_bundle:
                command += ["--trace-manifest", str(trace_bundle["manifest"]), "--trace-executor-record", str(trace_bundle["executor_record"]), "--trace-binding", str(trace_bundle["binding"]), "--trace-report", str(trace_bundle["report"])]
                for source in trace_bundle.get("sources", []): command += ["--trace-source", str(source)]
        return output, invoke(BUILD, *command)

    def fabricated_trace_bundle(self, root: Path, report_sha256: str) -> tuple[Path, dict[str, Path]]:
        source = ROOT / "tests/fixtures/production-execution-trace-p1/schema-r1"
        destination = root / "trace-bundle"; shutil.copytree(source, destination)
        trace = destination / "verified.json"; value = json.loads(trace.read_text())
        value["verification"]["independent_validation"]["report_sha256"] = report_sha256; write_json(trace, value)
        binding = destination / "verified-binding.json"; binding_value = json.loads(binding.read_text()); binding_value["trace_sha256"] = sha(trace); write_json(binding, binding_value)
        return trace, {"manifest": destination / "manifest.json", "executor_record": destination / "executor-record.json", "binding": binding, "report": destination / "report.json"}

    def test_cached_prefix_axis_same_state_oracle_context_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, expanded = self.expand(Path(directory))
            self.assertEqual(expanded["case_count"], 6214)
            self.assertEqual(expanded["stage_case_count"], {"smoke": 84, "representative": 2245, "full": 3885})
            by_id = {case["case_id"]: case for case in expanded["cases"]}
            cached = [case for case in expanded["cases"] if case["mode"] == "cached_prefix_chunked"]
            self.assertTrue(cached)
            for case in cached:
                oracle = by_id[case["path_oracle_case_id"]]
                self.assertGreater(case["cached_prefix_tokens"], 0)
                self.assertLessEqual(case["cached_prefix_tokens"] + case["prompt_tokens"], 4096)
                self.assertEqual((oracle["phase"], oracle["cached_prefix_tokens"], oracle["prompt_tokens"], oracle["prefill_requested_m"], oracle["resolved_m"]), (case["phase"], case["cached_prefix_tokens"], case["prompt_tokens"], case["prefill_requested_m"], 1))
            self.assertFalse(any(case["mode"] == "cached_prefix_chunked" and case["prompt_tokens"] == 4096 for case in expanded["cases"]))

    def test_expansion_rejects_empty_cached_axis_context_edge_and_count_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, mutation in enumerate(("empty", "edge", "count")):
                value = json.loads(MANIFEST.read_text())
                full = next(stage for stage in value["stages"] if stage["stage_id"] == "full")
                if mutation == "empty": full["prefill"]["cached_prefix_tokens"] = []
                elif mutation == "edge": full["prefill"]["cached_prefix_prompt_tokens"].append(4096)
                else: full["expected_case_count"]["total"] += 1
                manifest = root / f"bad-{index}.json"; write_json(manifest, value)
                completed = invoke(EXPAND, "--manifest", str(manifest), "--output", str(root / f"out-{index}.json"))
                self.assertNotEqual(completed.returncode, 0, mutation)

    def test_bound_policy_is_fully_bound_self_hashed_and_large_file_streams(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bound = self.bind(root, large_package=True)
            policy = json.loads(Path(bound["policy"]).read_text())
            self.assertEqual(policy["status"], "bound"); self.assertEqual(policy["effective_at"], "2026-07-14T12:00:00Z")
            self.assertTrue(all(policy["hash_binding"][field] for field in policy["binding_contract"]["required_hash_fields"]))
            completed = invoke(PLAN_VALIDATE, str(MANIFEST), "--policy", str(POLICY), "--bound-policy", str(bound["policy"]))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(Path(bound["package"]).joinpath("weights.bin").stat().st_size, 65 * 1024 * 1024 + 1)
            policy["power_condition"]["maximum_temperature_c"] += 1
            write_json(Path(bound["policy"]), policy)
            self.assertNotEqual(invoke(PLAN_VALIDATE, str(MANIFEST), "--policy", str(POLICY), "--bound-policy", str(bound["policy"])).returncode, 0)

    def test_binder_rejects_unbound_threshold_and_expanded_case_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bound = self.bind(root)
            bad = json.loads(Path(bound["expanded_path"]).read_text()); bad["cases"][0]["prompt_tokens"] += 1
            tampered = root / "tampered-expanded.json"; write_json(tampered, bad)
            # Reuse all bound artifact paths, but a changed expanded case must fail before output.
            identity = json.loads(Path(bound["identity"]).read_text()); artifacts = identity["artifacts"]
            command = ["--manifest", str(MANIFEST), "--expanded", str(tampered), "--policy", str(POLICY), "--worker", artifacts["worker"], "--package-root", artifacts["package_root"], "--package-manifest", artifacts["package_manifest"], "--tokenizer", artifacts["tokenizer"], "--served-model-manifest", artifacts["served_model_manifest"], "--model-identity", str(root / "model-identity.json"), "--graph", artifacts["graph"], "--state", artifacts["state"], "--source-oracle", artifacts["source_oracle"], "--power-capture", artifacts["power_capture"], "--correctness-thresholds", str(root / "correctness.json"), "--baseline-result", artifacts["baseline_result"], "--effective-at", "2026-07-14T12:00:00Z", "--git-commit", "a" * 40, "--output", str(root / "bad-id.json"), "--bound-policy", str(root / "bad-policy.json")]
            self.assertNotEqual(invoke(BIND, *command).returncode, 0)

    def test_runner_rejects_executable_bypass_and_does_not_store_argv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bound = self.bind(root)
            case_path, case = self.case(bound, root, lambda item: item["stage_id"] == "smoke" and item["mode"] == "all_m1")
            raw, completed = self.raw(bound, root, case_path, case)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            declared = json.loads(raw.read_text())["declared_execution"]
            self.assertFalse(declared["argv_values_recorded"]); self.assertNotIn("command_argv", json.loads(raw.read_text()))
            raw.unlink()
            other = root / "other-worker"; other.write_text("#!/bin/sh\nexit 0\n"); other.chmod(0o755)
            original = bound["worker"]; bound["worker"] = other
            bypass, bypassed = self.raw(bound, root, case_path, case)
            self.assertNotEqual(bypassed.returncode, 0); self.assertFalse(bypass.exists())
            bound["worker"] = original
            self.assertNotEqual(invoke(RUN, "--command", "sh").returncode, 0)

    def test_runner_rejects_bound_policy_and_package_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bound = self.bind(root)
            case_path, case = self.case(bound, root, lambda item: item["stage_id"] == "smoke" and item["mode"] == "all_m1")
            Path(bound["package"]).joinpath("weights.bin").write_bytes(b"changed")
            raw, completed = self.raw(bound, root, case_path, case)
            self.assertNotEqual(completed.returncode, 0); self.assertFalse(raw.exists())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bound = self.bind(root)
            case_path, case = self.case(bound, root, lambda item: item["stage_id"] == "smoke" and item["mode"] == "all_m1")
            policy = json.loads(Path(bound["policy"]).read_text()); policy["effective_at"] = "2026-07-14T12:00:01Z"; write_json(Path(bound["policy"]), policy)
            raw, completed = self.raw(bound, root, case_path, case)
            self.assertNotEqual(completed.returncode, 0); self.assertFalse(raw.exists())

    def test_runner_lock_gpu_snapshot_and_bounded_output_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bound = self.bind(root)
            case_path, case = self.case(bound, root, lambda item: item["stage_id"] == "representative" and item["device"]["device_id"] == "r9700-rdna4" and item["mode"] == "all_m1")
            lock = root / "ullm-r9700-p2-exclusive.lock"; handle = lock.open("a+"); fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                raw, completed = self.raw(bound, root, case_path, case, lock=lock)
                self.assertNotEqual(completed.returncode, 0); self.assertEqual(json.loads(raw.read_text())["failure_reason"], "r9700_queue_busy")
            finally: fcntl.flock(handle.fileno(), fcntl.LOCK_UN); handle.close()
            raw.unlink()
            foreign = {"weights_bytes": 1, "persistent_state_bytes": 1, "kv_cache_bytes": 1, "workspace_bytes": 1, "temporary_bytes": 1, "vram_headroom_bytes": 100, "gpu_process_snapshot": [{"pid": 123, "process_name": "foreign-worker", "vram_bytes": 1}]}
            foreign_raw, foreign_run = self.raw(bound, root, case_path, case, preflight=foreign, lock=lock)
            self.assertNotEqual(foreign_run.returncode, 0); self.assertEqual(json.loads(foreign_raw.read_text())["failure_reason"], "foreign_gpu_process")
        for size, expected in ((32, "ok"), (33, "failed")):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); bound = self.bind(root, worker_body=f"#!/bin/sh\nhead -c {size} /dev/zero\n")
                case_path, case = self.case(bound, root, lambda item: item["stage_id"] == "smoke" and item["mode"] == "all_m1")
                raw, _ = self.raw(bound, root, case_path, case, limit=32)
                value = json.loads(raw.read_text()); self.assertEqual(value["status"], expected); self.assertEqual(value["execution"]["output_overflow"], size > 32)

    def test_builder_normative_schema_raw_status_and_partial_matrix_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bound = self.bind(root)
            case_path, case = self.case(bound, root, lambda item: item["stage_id"] == "smoke" and item["mode"] == "all_m1")
            raw, completed = self.raw(bound, root, case_path, case); self.assertEqual(completed.returncode, 0, completed.stderr)
            result, built = self.result(bound, root, case_path, case, raw); self.assertEqual(built.returncode, 0, built.stderr)
            value = json.loads(result.read_text()); self.assertEqual(value["schema_version"], "ullm.prefill_validation.v1"); self.assertFalse(value["promotion"]["eligible"]); self.assertEqual(value["performance"]["warmup_runs"], 2); self.assertEqual(value["performance"]["measured_runs"], 10)
            report = root / "report.json"
            checked = invoke(VALIDATE, "--run-root", str(root), "--expanded", str(bound["expanded_path"]), "--identity", str(bound["identity"]), "--policy", str(bound["policy"]), "--source-oracle", str(bound["source"]), "--result", str(result), "--output", str(report))
            self.assertNotEqual(checked.returncode, 0)
            self.assertTrue(any(code.startswith("partial_matrix:") for code in json.loads(report.read_text())["failure_codes"]))
            self.assertNotEqual(invoke(BUILD, "--status", "failed").returncode, 0)

    def test_builder_rejects_raw_status_override_dummy_trace_and_path_state_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bound = self.bind(root)
            case_path, case = self.case(bound, root, lambda item: item["stage_id"] == "smoke" and item["mode"] == "all_m1")
            raw, _ = self.raw(bound, root, case_path, case); raw_value = json.loads(raw.read_text()); raw_value["status"] = "failed"; write_json(raw, raw_value)
            result, completed = self.result(bound, root, case_path, case, raw)
            self.assertNotEqual(completed.returncode, 0); self.assertFalse(result.exists())
            self.assertNotEqual(invoke(BUILD, "--independent-valid").returncode, 0)
            # A hash-linked JSON object is not a trace unless the normative
            # production trace contract itself passes.
            raw_value["status"] = "ok"; write_json(raw, raw_value)
            dummy_trace = root / "dummy-trace.json"; write_json(dummy_trace, {"schema_version": "dummy", "status": "ok", "scope": "component"})
            _, dummy = self.result(bound, root, case_path, case, raw, trace=dummy_trace)
            self.assertNotEqual(dummy.returncode, 0)

    def test_builder_rejects_cached_path_oracle_from_different_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bound = self.bind(root)
            expanded = bound["expanded"]; assert isinstance(expanded, dict)
            candidate = next(item for item in expanded["cases"] if item["stage_id"] == "representative" and item["device"]["device_id"] == "cpu-reference" and item["scope"] == "component" and item["mode"] == "cached_prefix_chunked")
            oracle = next(item for item in expanded["cases"] if item["case_id"] == candidate["path_oracle_case_id"])
            oracle_path = root / f"{oracle['case_id']}.case.json"; write_json(oracle_path, oracle)
            oracle_raw, oracle_run = self.raw(bound, root, oracle_path, oracle); self.assertEqual(oracle_run.returncode, 0, oracle_run.stderr)
            oracle_result, oracle_build = self.result(bound, root, oracle_path, oracle, oracle_raw); self.assertEqual(oracle_build.returncode, 0, oracle_build.stderr)
            tampered = json.loads(oracle_result.read_text()); tampered["workload"]["cached_prefix_tokens"] = 0; write_json(oracle_result, tampered)
            candidate_path = root / f"{candidate['case_id']}.case.json"; write_json(candidate_path, candidate)
            candidate_raw, candidate_run = self.raw(bound, root, candidate_path, candidate); self.assertEqual(candidate_run.returncode, 0, candidate_run.stderr)
            output, built = self.result(bound, root, candidate_path, candidate, candidate_raw, path_result=oracle_result)
            self.assertNotEqual(built.returncode, 0); self.assertFalse(output.exists())

    def test_builder_rejects_fabricated_detached_trace_report_hashes(self) -> None:
        for forged in ("x", "0" * 64):
            with self.subTest(forged=forged[:8]), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); bound = self.bind(root)
                case_path, case = self.case(bound, root, lambda item: item["stage_id"] == "smoke" and item["scope"] == "full_model" and item["mode"] == "all_m1")
                trace, bundle = self.fabricated_trace_bundle(root, forged)
                raw, ran = self.raw(bound, root, case_path, case, trace=trace); self.assertEqual(ran.returncode, 0, ran.stderr)
                output, built = self.result(bound, root, case_path, case, raw, trace=trace, trace_bundle=bundle)
                self.assertNotEqual(built.returncode, 0); self.assertFalse(output.exists())

    def test_validator_duplicate_extra_identity_escape_and_control_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bound = self.bind(root)
            case_path, case = self.case(bound, root, lambda item: item["stage_id"] == "smoke" and item["mode"] == "all_m1" and item["control_id"] == "reference_source_oracle")
            raw, _ = self.raw(bound, root, case_path, case); result, built = self.result(bound, root, case_path, case, raw); self.assertEqual(built.returncode, 0, built.stderr)
            value = json.loads(result.read_text()); value["workload"]["prompt_tokens"] += 1; value["promotion"]["eligible"] = True; write_json(result, value)
            extra = root / "extra.json"; extra_value = copy.deepcopy(value); extra_value["case_id"] = "extra-case"; write_json(extra, extra_value)
            report = root / "report.json"
            completed = invoke(VALIDATE, "--run-root", str(root), "--expanded", str(bound["expanded_path"]), "--identity", str(bound["identity"]), "--policy", str(bound["policy"]), "--source-oracle", str(bound["source"]), "--result", str(result), "--result", str(result), "--result", str(extra), "--output", str(report))
            self.assertNotEqual(completed.returncode, 0)
            codes = json.loads(report.read_text())["failure_codes"]
            self.assertIn("duplicate_result_path", codes); self.assertTrue(any(code.startswith("extra_result_case:") for code in codes)); self.assertTrue(any(code.startswith("result_workload_prompt_tokens:") for code in codes)); self.assertTrue(any(code.startswith("producer_promotion_claim:") for code in codes))
            self.assertTrue(any(code.startswith("control_promotion:") for code in codes))
            escaped = root.parent / "escaped-report.json"
            self.assertNotEqual(invoke(VALIDATE, "--run-root", str(root), "--expanded", str(bound["expanded_path"]), "--identity", str(bound["identity"]), "--policy", str(bound["policy"]), "--source-oracle", str(bound["source"]), "--result", str(result), "--output", str(escaped)).returncode, 0)
            self.assertFalse(escaped.exists())

    def test_final_validator_rejects_fabricated_detached_trace_report_hashes(self) -> None:
        for forged in ("x", "0" * 64):
            with self.subTest(forged=forged[:8]), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); bound = self.bind(root)
                case_path, case = self.case(bound, root, lambda item: item["stage_id"] == "smoke" and item["scope"] == "full_model" and item["mode"] == "all_m1")
                raw, ran = self.raw(bound, root, case_path, case); self.assertEqual(ran.returncode, 0, ran.stderr)
                result, built = self.result(bound, root, case_path, case, raw); self.assertEqual(built.returncode, 0, built.stderr)
                trace, bundle = self.fabricated_trace_bundle(root, forged)
                result_value = json.loads(result.read_text()); detached = json.loads(bundle["report"].read_text())
                result_value["evidence"]["execution_trace"] = {
                    "schema_version": "ullm.production_execution_trace.v1", "trace_id": json.loads(trace.read_text())["trace_id"],
                    "path": str(trace), "sha256": sha(trace), "scope": "full_model",
                    "validation": {
                        "manifest": {"path": str(bundle["manifest"]), "sha256": sha(bundle["manifest"])},
                        "executor_record": {"path": str(bundle["executor_record"]), "sha256": sha(bundle["executor_record"])},
                        "binding": {"path": str(bundle["binding"]), "sha256": sha(bundle["binding"])},
                        "detached_report": {"path": str(bundle["report"]), "sha256": sha(bundle["report"]), "report": detached},
                        "source_traces": [], "strict_validation": detached,
                    },
                }
                independent_path = Path(result_value["evidence"]["independent_validation"]["path"]); independent = json.loads(independent_path.read_text()); independent["trace_sha256"] = sha(trace); write_json(independent_path, independent)
                result_value["evidence"]["independent_validation"]["sha256"] = sha(independent_path); write_json(result, result_value)
                report = root / "forged-report-validation.json"
                checked = invoke(VALIDATE, "--run-root", str(root), "--expanded", str(bound["expanded_path"]), "--identity", str(bound["identity"]), "--policy", str(bound["policy"]), "--source-oracle", str(bound["source"]), "--result", str(result), "--output", str(report))
                self.assertNotEqual(checked.returncode, 0)
                self.assertTrue(any(code.startswith("trace_strict_validation:") for code in json.loads(report.read_text())["failure_codes"]))

    def test_trace_negative_scope_independent_reset_fallback_memory(self) -> None:
        spec = importlib.util.spec_from_file_location("p2_validate", VALIDATE); self.assertIsNotNone(spec and spec.loader)
        module = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
        trace = json.loads((ROOT / "tests/fixtures/production-execution-trace-p1/schema-r1/verified.json").read_text())
        case = {"scope": "full_model", "phase": "cold_prefill", "resolved_m": 4}
        self.assertEqual(module.trace_failures(trace, case, "case"), [])
        mutations = (
            ("scope", lambda value: value.__setitem__("scope", "component")),
            ("independent", lambda value: value["verification"]["independent_validation"].__setitem__("status", "not_run")),
            ("reset", lambda value: value["state_commit"]["reset"].__setitem__("complete", False)),
            ("fallback", lambda value: value["fallback"].__setitem__("unexpected_fallback_count", 1)),
            ("memory", lambda value: value["memory"].__setitem__("oom", {"observed": True})),
        )
        for name, mutate in mutations:
            changed = copy.deepcopy(trace); mutate(changed)
            self.assertTrue(any(name in failure for failure in module.trace_failures(changed, case, "case")), name)

    def test_full_trace_bundle_exact_case_association_and_decode_prefill_swaps(self) -> None:
        spec = importlib.util.spec_from_file_location("p2_build_association", BUILD); self.assertIsNotNone(spec and spec.loader)
        module = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
        trace_path = ROOT / "tests/fixtures/production-execution-trace-p1/schema-r1/verified.json"
        trace = json.loads(trace_path.read_text()); trace_sha = sha(trace_path)
        case = {
            "case_id": "cpu-production-trace-r1", "fixture_id": "cpu-production-trace-r1", "scope": "full_model",
            "phase": "cold_prefill", "mode": "cold_batched", "prompt_tokens": 4, "cached_prefix_tokens": 0,
            "context_tokens": 4, "decode_start_tokens": 4, "generated_tokens": 1, "prefill_requested_m": 4,
            "resolved_m": 4, "decode_request_count": 1, "sampling": None, "control_id": "aq4_0_target",
            "format_id": "AQ4_0", "implementation_id": "cpu_fixture_v1",
            "device": {"backend": "cpu", "name": "CPU fixture", "architecture": "x86_64", "runtime_device_index": 0},
        }
        fields = ("fixture_id", "scope", "phase", "mode", "prompt_tokens", "cached_prefix_tokens", "context_tokens", "decode_start_tokens", "generated_tokens", "prefill_requested_m", "resolved_m", "decode_request_count", "sampling", "control_id", "format_id", "implementation_id", "device")
        raw = {"case_contract": {key: case.get(key) for key in fields}, "links": {"trace": {"path": str(trace_path.resolve()), "sha256": trace_sha, "trace_id": trace["trace_id"]}}}
        identity = {"model_identity": trace["identity"]["model"], "artifacts": {"served_model_manifest": str(trace_path.with_name("manifest.json"))}, "hash_binding": {
            "served_model_manifest_sha256": trace["identity"]["served_model_manifest_sha256"],
            "worker_binary_sha256": trace["identity"]["worker"]["binary_sha256"],
            "package_manifest_sha256": trace["identity"]["package"]["manifest_sha256"],
        }}
        measurement = {"measured_runs": [{} for _ in range(10)], "trace_aggregation": {
            "schema_version": "ullm.aq4_p2_trace_aggregation.v1", "case_id": case["case_id"], "trace_id": trace["trace_id"],
            "trace_sha256": trace_sha, "sample_count": 10,
            "phase_wall_time_ms": {item["phase_id"]: item["wall_time_ms"] for item in trace["phases"]},
            "terminal_audit_sha256": module.trace_terminal_sha(trace),
        }}
        module.validate_trace_association(trace, case, raw, identity, measurement, trace_path, trace_sha)
        other_decode = copy.deepcopy(case); other_decode.update({"case_id": "other-decode-case", "fixture_id": "other-decode-case", "phase": "decode", "prompt_tokens": 4, "resolved_m": 1, "prefill_requested_m": 1})
        with self.assertRaises(module.ResultError): module.validate_trace_association(trace, other_decode, raw, identity, measurement, trace_path, trace_sha)
        other_prefill_raw = copy.deepcopy(raw); other_prefill_raw["links"]["trace"]["path"] = str(trace_path.with_name("verified-copy.json"))
        with self.assertRaises(module.ResultError): module.validate_trace_association(trace, case, other_prefill_raw, identity, measurement, trace_path, trace_sha)


if __name__ == "__main__": unittest.main()
