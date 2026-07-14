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
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/workloads/aq4-production-opt-p2-case-manifest-v0.1.json"
POLICY = ROOT / "benchmarks/workloads/aq4-production-opt-p2-threshold-policy-template-v0.1.json"
PLAN_VALIDATE = ROOT / "benchmarks/workloads/validate-aq4-production-opt-p2-manifest.py"
EXPAND = ROOT / "tools/expand-aq4-production-p2.py"
BIND = ROOT / "tools/bind-aq4-production-p2-identity.py"
RUN = ROOT / "tools/run-aq4-production-p2.py"
BUILD = ROOT / "tools/build-aq4-prefill-validation-result.py"
VALIDATE = ROOT / "tools/validate-aq4-production-p2-evidence.py"
RAW_V2_ADAPTER = ROOT / "tools/prefill_validation/aq4_p2_raw_v2_adapter.py"


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
    def driver_fixture(self, root: Path) -> Path:
        driver = root / "fixture-driver.py"
        driver.write_text('''#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
def arg(name): return Path(sys.argv[sys.argv.index(name)+1])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
case_path=arg("--case"); identity_path=arg("--identity"); preflight_path=arg("--preflight"); output=arg("--output"); fixture=json.loads(arg("--fixture").read_text()); case=json.loads(case_path.read_text()); identity=json.loads(identity_path.read_text()); preflight=json.loads(preflight_path.read_text()); index=int(output.stem.split("-")[1].split(".")[0])
if fixture.get("mode")=="partial" and index==2: raise SystemExit(2)
worker=Path(identity["artifacts"]["worker"]); driver=Path(__file__).resolve(); role_swap=fixture.get("mode")=="role_swap"; effective_index=0 if fixture.get("mode")=="reuse" else index
timing={"request_elapsed_ms":20.0+effective_index,"generation":{"cache_n":case["cached_prefix_tokens"],"prompt_n":case["prompt_tokens"],"prompt_ms":10.0+effective_index,"prompt_per_token_ms":1.0,"prompt_per_second":100.0,"predicted_n":case["generated_tokens"],"predicted_ms":5.0,"predicted_per_token_ms":1.0,"predicted_per_second":1.0},"generated_tokens":case["generated_tokens"]}
timing_sha=hashlib.sha256(json.dumps(timing,separators=(",",":"),ensure_ascii=False).encode()).hexdigest(); reset={"attempted":1,"complete":1,"failed":0}; mode=fixture.get("mode","success")
if mode=="reset": reset={"attempted":1,"complete":0,"failed":1}
lifecycle={"prepare":2,"commit":2,"discard":0,"error":0,"cancel":0,"prefill":{"prepare":1,"commit":1,"discard":0},"publication":{"prepare":1,"commit":1,"discard":0},"reset":reset}
unexpected=1 if mode=="fallback" else 0; fallback={"count":unexpected,"unexpected_count":unexpected,"reasons":[]}
failed=mode=="preflight_fail"; audit=None if failed else {"deterministic_digest_sha256":"a"*64,"outcome":"length","coverage_complete":True,"physical_operation_invocations":1,"total_records":1}
driver_identity=None if failed else {"served_model_manifest_sha256":identity["hash_binding"]["served_model_manifest_sha256"],"model_id":identity["model_identity"]["id"],"model_revision":identity["model_identity"]["revision"],"format_id":case["format_id"],"implementation_id":case["implementation_id"],"manifest_worker_binary_path":str(worker.resolve()),"manifest_worker_binary_sha256":identity["hash_binding"]["worker_binary_sha256"],"benchmark_binary_path":str((worker if role_swap else driver).resolve()),"benchmark_binary_sha256":sha(worker if role_swap else driver),"benchmark_worker_roles_distinct":not role_swap,"package_root":identity["artifacts"]["package_root"],"package_content_sha256":identity["hash_binding"]["package_content_sha256"],"package_manifest_sha256":identity["hash_binding"]["package_manifest_sha256"],"package_file_count":1,"package_bytes":1,"manifest_device_architecture":case["device"]["architecture"],"runtime_device":{"requested_device_index":case["device"]["runtime_device_index"],"observed_device_id":0,"observed_backend":case["device"]["backend"],"observed_name":case["device"]["name"],"observed_architecture":case["device"]["architecture"]},"execution_profile":"fixture"}
def link(path): return {"path":str(path.resolve()),"sha256":sha(path)}
value={"schema_version":"ullm.qwen35_aq4_p2.full_model_driver.v2","raw_target_schema_version":"ullm.aq4_production_p2_raw_result.v2","scope":case["scope"],"status":"failed" if failed else "ok","immutable_status":failed,"case_id":case["case_id"],"case_sha256":case["case_sha256"],"identity":driver_identity,"requested_m":case["prefill_requested_m"],"resolved_m":None if failed else case["resolved_m"],"actual_token_batch_width":None if failed else case["resolved_m"],"actual_request_batch_width":None if failed else case["request_count"],"timing":timing,"audit":audit,"lifecycle":None if failed else lifecycle,"reset":None if failed else reset,"outcome":None if failed else "length","oom":None,"fallback":fallback,"preflight":{"input":preflight,"required_environment_count":0,"required_environment_verified":not failed,"binary_roles_verified":not failed,"package_tree_verified":not failed},"failure":{"stage":"environment_preflight","reason_code":"fixture_rejected"} if failed else None,"links":{"case":link(case_path),"identity":link(identity_path),"preflight":link(preflight_path),"timing":{"json_pointer":"/timing","sha256":timing_sha},"audit":{"json_pointer":"/audit","sha256":None if failed else "a"*64}},"adapter":{"target_schema_version":"ullm.aq4_production_p2_raw_result.v2","mapping_version":"ullm.aq4_p2_full_model_to_raw.v1","exact_root_fields":True,"benchmark_binary_role":"executed_benchmark_driver","manifest_worker_role":"served_identity_reference","raw_v2_requires_role_aware_adapter":True}}
if mode=="case_swap": value["case_id"]="another-case"
if mode=="unknown": value["unknown_field"]=True
if mode=="link_hash": value["links"]["case"]["sha256"]="0"*64
text=json.dumps(value,separators=(",",":"))
if mode=="duplicate": text=text.replace('{"schema_version":','{"schema_version":"duplicate","schema_version":',1)
output.write_text(text); raise SystemExit(1 if failed else 0)
''', encoding="utf-8")
        driver.chmod(0o755); return driver

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
        model_contract = {"id": "ullm-qwen3.5-9b-aq4", "revision": "aq4-reasoning-v0.1-candidate", "format_id": "AQ4_0", "implementation_id": "qwen35_aq4_rdna4_v1"}
        served = root / "served-model.json"; write_json(served, {"public": {"id": model_contract["id"], "revision": model_contract["revision"]}, "format": {"format_id": model_contract["format_id"], "implementation_id": model_contract["implementation_id"]}})
        model = root / "model-identity.json"; write_json(model, model_contract)
        graph = root / "graph.json"; write_json(graph, {"source": "fixture", "nodes": 1})
        state_schema = root / "state-schema.json"; write_json(state_schema, {"schema": "fixture-state-v1"})
        source_identity = {"model_id": model_contract["id"], "model_revision": model_contract["revision"], "source_checkpoint": {"aggregate_sha256": "d" * 64}, "tokenizer": {"aggregate_sha256": "e" * 64}}
        source = root / "source-oracle.json"; write_json(source, {"schema_version": "ullm.qwen35_aq4_source_oracle.v1", "oracle_kind": "independent_source", "status": "fixture", "evidence_class": "synthetic_fixture", "promotion_eligible": False, "identity": source_identity})
        source_validation = root / "source-oracle-validation.json"; write_json(source_validation, {"schema_version": "ullm.qwen35_aq4_p2_oracle_validator.v1", "status": "valid", "oracle_kind": "independent_source", "manifest_sha256": sha(source), "production_eligible": False, "blockers": ["synthetic fixture"]})
        power = root / "power.json"; write_json(power, {"policy_binding": {"expected_power_limit_watts": 300, "allowed_power_tolerance_watts": 5, "maximum_temperature_c": 95, "minimum_vram_headroom_bytes": 1}})
        correctness = root / "correctness.json"; write_json(correctness, {"max_hidden_relative_l2": 1.0, "max_hidden_max_abs": 1.0, "max_logits_relative_l2": 1.0, "max_logits_max_abs": 1.0, "minimum_top_k_overlap": 1})
        baseline = root / "baseline.json"; write_json(baseline, {"prefill_tokens_per_second_p50": 100.0, "prefill_tokens_per_second_p95": 100.0, "oom": False})
        identity = root / "identity.json"; bound_policy = root / "bound-policy.json"
        completed = invoke(BIND, "--manifest", str(MANIFEST), "--expanded", str(expanded_path), "--policy", str(POLICY), "--worker", str(worker), "--package-root", str(package), "--package-manifest", str(package_manifest), "--tokenizer", str(tokenizer), "--served-model-manifest", str(served), "--model-identity", str(model), "--graph", str(graph), "--state", str(state_schema), "--source-oracle", str(source), "--power-capture", str(power), "--correctness-thresholds", str(correctness), "--baseline-result", str(baseline), "--effective-at", "2026-07-14T12:00:00Z", "--git-commit", "a" * 40, "--output", str(identity), "--bound-policy", str(bound_policy))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return {"expanded_path": expanded_path, "expanded": expanded, "package": package, "package_manifest": package_manifest, "worker": worker, "source": source, "source_validation": source_validation, "identity": identity, "policy": bound_policy, "baseline": baseline, "model": model, "served": served}

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

    def adapted_raw(self, bound: dict[str, Path | dict], root: Path, case_path: Path, case: dict, mode: str = "success") -> tuple[Path, subprocess.CompletedProcess[str]]:
        preflight = root / "adapter-preflight.json"; write_json(preflight, {"weights_bytes": 1, "persistent_state_bytes": 1, "kv_cache_bytes": 1, "workspace_bytes": 1, "temporary_bytes": 1, "vram_headroom_bytes": 100, "gpu_process_snapshot": []})
        fixture = root / "adapter-fixture.json"; write_json(fixture, {"mode": mode})
        driver = self.driver_fixture(root); raw = root / "adapted.raw.json"
        command = ["--run-root", str(root), "--case", str(case_path), "--identity", str(bound["identity"]), "--policy", str(bound["policy"]), "--preflight", str(preflight), "--driver", str(driver), "--served-model-manifest", str(bound["served"]), "--fixture", str(fixture), "--raw-dir", str(root / "driver-runs"), "--output", str(raw), "--measurement", str(root / "adapted.measurement.json"), "--state", str(root / "adapted.state.json"), "--driver-lifecycle-input", str(root / "adapted.lifecycle.json"), "--cpu-fixture"]
        return raw, invoke(RAW_V2_ADAPTER, *command)

    def calibration_evidence(self, bound: dict[str, Path | dict], root: Path, case: dict, compare_kind: str, path_result: Path | None = None) -> Path:
        identity = json.loads(Path(bound["identity"]).read_text()); policy = json.loads(Path(bound["policy"]).read_text()); expanded = bound["expanded"]; assert isinstance(expanded, dict)
        step_count = case["generated_tokens"] if case["phase"] == "decode" else 1
        source_value = json.loads(Path(bound["source"]).read_text()); source_root = root / "source-full-calibration"; source_root.mkdir(exist_ok=True)
        source_manifest = source_root / "manifest.json"
        write_json(source_manifest, {"schema_version": "ullm.qwen35_aq4_source_calibration.v1", "oracle_kind": "independent_source_full", "status": "available", "identity": {**source_value["identity"], "hidden_size": 4096, "vocab_size": 248320}, "parent_sampled_oracle": {"path": str(Path(bound["source"]).resolve()), "manifest_sha256": sha(Path(bound["source"])), "schema_version": source_value["schema_version"]}})
        hashes = identity["hash_binding"]

        def target_root(target_case: dict, oracle_kind: str) -> Path:
            destination = root / f"{target_case['case_id']}.{oracle_kind}.target"; destination.mkdir(exist_ok=True)
            write_json(destination / "manifest.json", {
                "schema_version": "ullm.qwen35_aq4_target_calibration.v1", "oracle_kind": oracle_kind, "status": "available", "capture_complete": True, "promotion_eligible": False,
                "identity": {"model_id": source_value["identity"]["model_id"], "model_revision": source_value["identity"]["model_revision"], "format_id": identity["model_identity"]["format_id"], "implementation_id": identity["model_identity"]["implementation_id"], "package_content_sha256": hashes["package_content_sha256"], "package_manifest_sha256": hashes["package_manifest_sha256"], "worker_binary_sha256": hashes["worker_binary_sha256"]},
                "binding": {"case_id": target_case["case_id"], "case_sha256": target_case["case_sha256"], "requested_m": target_case["prefill_requested_m"], "resolved_m": target_case["resolved_m"], "device": {"requested_index": target_case["device"]["runtime_device_index"], "device_id": target_case["device"]["device_id"], "backend": target_case["device"]["backend"], "name": target_case["device"]["name"], "architecture": target_case["device"]["architecture"]}, "source": {"manifest": {"path": str(source_manifest), "sha256": sha(source_manifest)}}},
            })
            return destination

        path_oracle_fields = {}
        if compare_kind == "source_gate":
            reference_root = source_root; candidate_root = target_root(case, "aq4_target")
        else:
            assert path_result is not None
            oracle_result = json.loads(path_result.read_text()); oracle_source_comparison = json.loads(Path(oracle_result["calibration"]["source_gate"]["comparison"]["path"]).read_text())
            reference_root = Path(oracle_source_comparison["candidate"]["path"]); candidate_root = target_root(case, "aq4_optimized")
            path_oracle_fields = {"path_oracle_case_id": oracle_result["case_id"], "path_oracle_result_sha256": sha(path_result), "path_oracle_calibration_manifest_sha256": sha(reference_root / "manifest.json")}
        comparison = root / f"{case['case_id']}.{compare_kind}.comparison.json"
        roles = {"source_gate": ("independent_source_full", "aq4_target"), "path_gate": ("aq4_target", "aq4_optimized")}[compare_kind]
        comparison_value = {
            "schema_version": "ullm.qwen35_aq4_calibration_comparison.v1", "status": "valid", "promotion_eligible": False,
            "created_utc": "2026-07-14T12:00:00Z", "compare_kind": compare_kind,
            "reference": {"path": str(reference_root), "manifest_sha256": sha(reference_root / "manifest.json"), "schema_version": "fixture.v1", "oracle_kind": roles[0]},
            "candidate": {"path": str(candidate_root), "manifest_sha256": sha(candidate_root / "manifest.json"), "schema_version": "fixture.v1", "oracle_kind": roles[1]},
            "vector_contract": {"hidden_shape": [4096], "logits_shape": [248320], "dtype": "f32", "endianness": "little", "metric_denominator": "max(reference_l2,1e-30)", "top_k": 10},
            "rows": {"file": "rows.jsonl", "record_count": step_count, "sha256": hashlib.sha256(f"rows:{compare_kind}:{case['case_id']}".encode()).hexdigest()},
            "summary": {"row_count": step_count, "nonfinite_rows": 0, "greedy_mismatch_rows": 0, "max_hidden_relative_l2": 0.0, "max_hidden_max_abs": 0.0, "max_logits_relative_l2": 0.0, "max_logits_max_abs": 0.0, "minimum_top_k_overlap": 10},
            "observed_values_only": True,
        }
        write_json(comparison, comparison_value)
        evidence = root / f"{case['case_id']}.{compare_kind}.calibration.json"
        write_json(evidence, {
            "schema_version": "ullm.aq4_p2_calibration_evidence.v1", "status": "valid", "compare_kind": compare_kind,
            "case": case, "canonical_case_sha256": expanded["canonical_case_sha256"], "step_count": step_count,
            "identity": {"model": identity["model_identity"], "source_oracle_sha256": sha(Path(bound["source"])), "package_content_sha256": hashes["package_content_sha256"], "package_manifest_sha256": hashes["package_manifest_sha256"], "worker_binary_sha256": hashes["worker_binary_sha256"], "policy_sha256": policy["hash_binding"]["policy_sha256"]},
            "comparison": {"path": str(comparison), "sha256": sha(comparison)},
            **path_oracle_fields,
        })
        return evidence

    def result(self, bound: dict[str, Path | dict], root: Path, case_path: Path, case: dict, raw: Path, *, path_result: Path | None = None, trace: Path | None = None, trace_bundle: dict[str, Path] | None = None, source_calibration: Path | bool | None = None, path_calibration: Path | bool | None = None) -> tuple[Path, subprocess.CompletedProcess[str]]:
        independent = root / f"{case['case_id']}.independent.json"
        correctness = {field: True for field in ("finite", "shape_contract_passed", "source_oracle_passed", "greedy_tokens_exact", "kv_state_cache_passed", "scheduler_progress_passed", "chunk_equivalence_passed", "cancel_reset_passed", "publish_failure_reset_passed")}
        correctness["path_oracle_passed"] = case["mode"] != "all_m1"
        correctness["final_hidden"] = {"relative_l2": 0.0, "max_abs": 0.0}; correctness["logits"] = {"relative_l2": 0.0, "max_abs": 0.0, "top_k_overlap": 1}
        write_json(independent, {"schema_version": "ullm.aq4_p2_independent_validation.v1", "status": "valid", "validator_independent": True, "case_id": case["case_id"], "case_sha256": case["case_sha256"], "raw_sha256": sha(raw), "source_oracle_sha256": sha(bound["source"]), "path_oracle_result_sha256": sha(path_result) if path_result else None, "trace_sha256": sha(trace) if trace else None, "correctness": correctness})
        output = root / f"{case['case_id']}.result.json"
        if source_calibration is None: source_calibration = self.calibration_evidence(bound, root, case, "source_gate")
        if case["mode"] in {"cold_batched", "cached_prefix_chunked"} and path_calibration is None and path_result is not None: path_calibration = self.calibration_evidence(bound, root, case, "path_gate", path_result)
        command = ["--run-root", str(root), "--case", str(case_path), "--expanded", str(bound["expanded_path"]), "--raw", str(raw), "--identity", str(bound["identity"]), "--policy", str(bound["policy"]), "--source-oracle", str(bound["source"]), "--source-oracle-validation", str(bound["source_validation"]), "--independent-validation", str(independent), "--output", str(output)]
        if source_calibration is not False: command += ["--source-calibration-evidence", str(source_calibration)]
        if path_result: command += ["--path-oracle-result", str(path_result)]
        if path_calibration is not False and path_calibration is not None: command += ["--path-calibration-evidence", str(path_calibration)]
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
            production = [case for case in expanded["cases"] if case["scope"] == "production_server"]
            self.assertTrue(production)
            for case in production:
                self.assertIsNotNone(case["sampling"]); self.assertIsNotNone(case["control"]); self.assertTrue(case["implementation_id"])
                self.assertTrue(all(case["device"].get(field) is not None for field in ("backend", "name", "architecture", "runtime_device_index")))
                self.assertEqual(case["request_count"], 1)
                self.assertEqual(case["decode_request_count"], 1 if case["phase"] == "decode" else 0)

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
            expanded_contract = json.loads(Path(bound["expanded_path"]).read_text())
            target_cases = [case for case in expanded_contract["cases"] if case["control_id"] == "aq4_0_target"]
            self.assertTrue(target_cases); self.assertEqual({case["implementation_id"] for case in target_cases}, {"qwen35_aq4_rdna4_v1"})
            self.assertEqual(json.loads(Path(bound["identity"]).read_text())["model_identity"]["implementation_id"], "qwen35_aq4_rdna4_v1")
            bad = json.loads(Path(bound["expanded_path"]).read_text()); bad["cases"][0]["prompt_tokens"] += 1
            tampered = root / "tampered-expanded.json"; write_json(tampered, bad)
            # Reuse all bound artifact paths, but a changed expanded case must fail before output.
            identity = json.loads(Path(bound["identity"]).read_text()); artifacts = identity["artifacts"]
            command = ["--manifest", str(MANIFEST), "--expanded", str(tampered), "--policy", str(POLICY), "--worker", artifacts["worker"], "--package-root", artifacts["package_root"], "--package-manifest", artifacts["package_manifest"], "--tokenizer", artifacts["tokenizer"], "--served-model-manifest", artifacts["served_model_manifest"], "--model-identity", str(root / "model-identity.json"), "--graph", artifacts["graph"], "--state", artifacts["state"], "--source-oracle", artifacts["source_oracle"], "--power-capture", artifacts["power_capture"], "--correctness-thresholds", str(root / "correctness.json"), "--baseline-result", artifacts["baseline_result"], "--effective-at", "2026-07-14T12:00:00Z", "--git-commit", "a" * 40, "--output", str(root / "bad-id.json"), "--bound-policy", str(root / "bad-policy.json")]
            self.assertNotEqual(invoke(BIND, *command).returncode, 0)
            mismatch_model = root / "mismatch-model.json"; model = json.loads(Path(bound["model"]).read_text()); model["implementation_id"] = "qwen35_aq4_wrong_v1"; write_json(mismatch_model, model)
            mismatch = list(command); mismatch[mismatch.index(str(tampered))] = str(bound["expanded_path"]); mismatch[mismatch.index(str(root / "model-identity.json"))] = str(mismatch_model)
            rejected = invoke(BIND, *mismatch); self.assertNotEqual(rejected.returncode, 0); self.assertIn("implementation differs", rejected.stderr)
            partial = copy.deepcopy(expanded_contract); removed = partial["cases"].pop(); partial["case_count"] -= 1; partial["expected_case_count"]["total"] -= 1; partial["stage_case_count"][removed["stage_id"]] -= 1
            partial["canonical_case_sha256"] = hashlib.sha256(json.dumps(partial["cases"], ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            partial_path = root / "partial-rehashed-expanded.json"; write_json(partial_path, partial)
            partial_command = list(command); partial_command[partial_command.index(str(tampered))] = str(partial_path)
            partial_command[partial_command.index(str(root / "bad-id.json"))] = str(root / "partial-id.json"); partial_command[partial_command.index(str(root / "bad-policy.json"))] = str(root / "partial-policy.json")
            partial_rejected = invoke(BIND, *partial_command); self.assertNotEqual(partial_rejected.returncode, 0); self.assertIn("complete planning expansion", partial_rejected.stderr)
            validator_spec = importlib.util.spec_from_file_location("p2_complete_validator", VALIDATE); assert validator_spec and validator_spec.loader
            validator_module = importlib.util.module_from_spec(validator_spec); validator_spec.loader.exec_module(validator_module)
            self.assertEqual(validator_module.complete_expansion_failure(partial, identity, root), "expanded_not_complete_planning_set")

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

    def test_role_aware_raw_v2_adapter_end_to_end_and_fail_closed_mutations(self) -> None:
        select = lambda item: item["scope"] == "full_model" and item["phase"] == "cold_prefill" and item["mode"] == "all_m1" and item["control_id"] == "aq4_0_target"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bound = self.bind(root); case_path, case = self.case(bound, root, select)
            raw, adapted = self.adapted_raw(bound, root, case_path, case); self.assertEqual(adapted.returncode, 0, adapted.stderr)
            value = json.loads(raw.read_text()); self.assertEqual(len(value["links"]["driver_runs"]), 12)
            self.assertNotEqual(value["executed_benchmark_driver"]["sha256"], value["served_identity_reference"]["worker_sha256"])
            self.assertIsNone(value["links"]["trace"]); self.assertIn("driver_lifecycle_input", value["links"])
            self.assertTrue(json.loads((root / "adapted.lifecycle.json").read_text())["not_a_production_execution_trace"])
            result, built = self.result(bound, root, case_path, case, raw); self.assertEqual(built.returncode, 0, built.stderr)
            result_value = json.loads(result.read_text()); self.assertIsNone(result_value["evidence"]["execution_trace"]); self.assertFalse(result_value["promotion"]["eligible"])
            report = root / "partial-validation.json"; checked = invoke(VALIDATE, "--run-root", str(root), "--expanded", str(bound["expanded_path"]), "--identity", str(bound["identity"]), "--policy", str(bound["policy"]), "--source-oracle", str(bound["source"]), "--result", str(result), "--output", str(report))
            self.assertNotEqual(checked.returncode, 0); self.assertTrue(any(code.startswith("partial_matrix:") for code in json.loads(report.read_text())["failure_codes"]))
        for mode in ("role_swap", "reset", "fallback", "partial", "reuse", "case_swap", "unknown", "duplicate", "link_hash"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); bound = self.bind(root); case_path, case = self.case(bound, root, select)
                raw, rejected = self.adapted_raw(bound, root, case_path, case, mode); self.assertNotEqual(rejected.returncode, 0, mode); self.assertTrue((root / "driver-runs").is_dir()); self.assertFalse(raw.exists())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bound = self.bind(root); case_path, case = self.case(bound, root, select)
            raw, rejected = self.adapted_raw(bound, root, case_path, case, "preflight_fail"); self.assertNotEqual(rejected.returncode, 0); self.assertTrue(raw.is_file()); self.assertTrue(json.loads(raw.read_text())["immutable_status"]); self.assertIsNone(json.loads(raw.read_text())["links"]["trace"])
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
            self.assertEqual(value["calibration"]["source_gate"]["metrics"]["minimum_top_k_overlap"], 10.0); self.assertIsNone(value["calibration"]["path_gate"])
            self.assertFalse(any("calibration" in key for key in value["performance"]))
            report = root / "report.json"
            checked = invoke(VALIDATE, "--run-root", str(root), "--expanded", str(bound["expanded_path"]), "--identity", str(bound["identity"]), "--policy", str(bound["policy"]), "--source-oracle", str(bound["source"]), "--result", str(result), "--output", str(report))
            self.assertNotEqual(checked.returncode, 0)
            self.assertTrue(any(code.startswith("partial_matrix:") for code in json.loads(report.read_text())["failure_codes"]))
            self.assertNotEqual(invoke(BUILD, "--status", "failed").returncode, 0)

    def test_builder_calibration_binding_swap_hash_identity_threshold_nonfinite_unknown_and_missing(self) -> None:
        mutations = ("case_swap", "comparison_swap", "hash", "identity", "threshold", "nonfinite", "unknown", "symlink", "hardlink", "missing")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); bound = self.bind(root)
                case_path, case = self.case(bound, root, lambda item: item["stage_id"] == "smoke" and item["mode"] == "all_m1")
                raw, ran = self.raw(bound, root, case_path, case); self.assertEqual(ran.returncode, 0, ran.stderr)
                evidence = self.calibration_evidence(bound, root, case, "source_gate")
                if mutation == "case_swap":
                    expanded = bound["expanded"]; assert isinstance(expanded, dict)
                    other = next(item for item in expanded["cases"] if item["mode"] == "all_m1" and item["case_id"] != case["case_id"])
                    evidence = self.calibration_evidence(bound, root, other, "source_gate")
                elif mutation == "comparison_swap":
                    expanded = bound["expanded"]; assert isinstance(expanded, dict)
                    other = next(item for item in expanded["cases"] if item["mode"] == "all_m1" and item["case_id"] != case["case_id"])
                    other_evidence = self.calibration_evidence(bound, root, other, "source_gate")
                    value = json.loads(evidence.read_text()); value["comparison"] = json.loads(other_evidence.read_text())["comparison"]; write_json(evidence, value)
                elif mutation in {"hash", "identity", "unknown"}:
                    value = json.loads(evidence.read_text())
                    if mutation == "hash": value["comparison"]["sha256"] = "0" * 64
                    elif mutation == "identity": value["identity"]["worker_binary_sha256"] = "0" * 64
                    else: value["unknown"] = True
                    write_json(evidence, value)
                elif mutation in {"threshold", "nonfinite"}:
                    value = json.loads(evidence.read_text()); comparison = Path(value["comparison"]["path"]); compared = json.loads(comparison.read_text())
                    compared["summary"]["max_hidden_relative_l2"] = 2.0 if mutation == "threshold" else float("nan")
                    write_json(comparison, compared); value["comparison"]["sha256"] = sha(comparison); write_json(evidence, value)
                elif mutation in {"symlink", "hardlink"}:
                    outside = root / "hardlink-peer.json"; shutil.copyfile(evidence, outside); evidence.unlink(); os.link(outside, evidence)
                    if mutation == "symlink": evidence.unlink(); evidence.symlink_to(outside.name)
                result, built = self.result(bound, root, case_path, case, raw, source_calibration=False if mutation == "missing" else evidence)
                self.assertNotEqual(built.returncode, 0, built.stderr); self.assertFalse(result.exists())

    def test_builder_rejects_bound_source_reference_swap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bound = self.bind(root)
            case_path, case = self.case(bound, root, lambda item: item["stage_id"] == "smoke" and item["mode"] == "all_m1")
            raw, ran = self.raw(bound, root, case_path, case); self.assertEqual(ran.returncode, 0, ran.stderr)
            evidence = self.calibration_evidence(bound, root, case, "source_gate"); value = json.loads(evidence.read_text())
            alternate_sampled = root / "alternate-sampled-source.json"; shutil.copyfile(Path(bound["source"]), alternate_sampled)
            alternate_root = root / "alternate-source-full"; alternate_root.mkdir()
            original_manifest = json.loads((root / "source-full-calibration" / "manifest.json").read_text())
            original_manifest["parent_sampled_oracle"]["path"] = str(alternate_sampled)
            write_json(alternate_root / "manifest.json", original_manifest)
            comparison = Path(value["comparison"]["path"]); compared = json.loads(comparison.read_text())
            compared["reference"]["path"] = str(alternate_root); compared["reference"]["manifest_sha256"] = sha(alternate_root / "manifest.json")
            write_json(comparison, compared); value["comparison"]["sha256"] = sha(comparison); write_json(evidence, value)
            result, built = self.result(bound, root, case_path, case, raw, source_calibration=evidence)
            self.assertNotEqual(built.returncode, 0, built.stderr); self.assertFalse(result.exists())

    def test_optimized_requires_separate_path_calibration_and_all_m1_rejects_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bound = self.bind(root); expanded = bound["expanded"]; assert isinstance(expanded, dict)
            candidate = next(item for item in expanded["cases"] if item["stage_id"] == "smoke" and item["mode"] == "cold_batched")
            oracle = next(item for item in expanded["cases"] if item["case_id"] == candidate["path_oracle_case_id"])
            oracle_path = root / f"{oracle['case_id']}.case.json"; write_json(oracle_path, oracle)
            oracle_raw, ran = self.raw(bound, root, oracle_path, oracle); self.assertEqual(ran.returncode, 0, ran.stderr)
            oracle_result, built = self.result(bound, root, oracle_path, oracle, oracle_raw); self.assertEqual(built.returncode, 0, built.stderr)
            candidate_path = root / f"{candidate['case_id']}.case.json"; write_json(candidate_path, candidate)
            candidate_raw, ran = self.raw(bound, root, candidate_path, candidate); self.assertEqual(ran.returncode, 0, ran.stderr)
            candidate_result, built = self.result(bound, root, candidate_path, candidate, candidate_raw, path_result=oracle_result); self.assertEqual(built.returncode, 0, built.stderr)
            calibration = json.loads(candidate_result.read_text())["calibration"]
            self.assertNotEqual(calibration["source_gate"]["comparison"]["sha256"], calibration["path_gate"]["comparison"]["sha256"])
            unexpected_path = Path(calibration["path_gate"]["path"])
            path_comparison = Path(calibration["path_gate"]["comparison"]["path"])
            evidence_value = json.loads(unexpected_path.read_text()); comparison_value = json.loads(path_comparison.read_text())
            candidate_result.unlink()

            # Each of the externally supplied all-M1 bindings is rebuilt from
            # the result and its source calibration instead of being trusted.
            evidence_value["path_oracle_result_sha256"] = "0" * 64; write_json(unexpected_path, evidence_value)
            rejected_output, rejected = self.result(bound, root, candidate_path, candidate, candidate_raw, path_result=oracle_result, path_calibration=unexpected_path)
            self.assertNotEqual(rejected.returncode, 0); self.assertFalse(rejected_output.exists())
            evidence_value["path_oracle_result_sha256"] = sha(oracle_result)
            evidence_value["path_oracle_calibration_manifest_sha256"] = "0" * 64; write_json(unexpected_path, evidence_value)
            rejected_output, rejected = self.result(bound, root, candidate_path, candidate, candidate_raw, path_result=oracle_result, path_calibration=unexpected_path)
            self.assertNotEqual(rejected.returncode, 0); self.assertFalse(rejected_output.exists())
            evidence_value["path_oracle_calibration_manifest_sha256"] = comparison_value["reference"]["manifest_sha256"]

            swapped_reference = root / "swapped-all-m1-target"
            shutil.copytree(Path(comparison_value["reference"]["path"]), swapped_reference)
            swapped_comparison = copy.deepcopy(comparison_value); swapped_comparison["reference"]["path"] = str(swapped_reference)
            write_json(path_comparison, swapped_comparison); evidence_value["comparison"]["sha256"] = sha(path_comparison); write_json(unexpected_path, evidence_value)
            rejected_output, rejected = self.result(bound, root, candidate_path, candidate, candidate_raw, path_result=oracle_result, path_calibration=unexpected_path)
            self.assertNotEqual(rejected.returncode, 0); self.assertFalse(rejected_output.exists())
            write_json(path_comparison, comparison_value); evidence_value["comparison"]["sha256"] = sha(path_comparison); write_json(unexpected_path, evidence_value)

            other_oracle = next(item for item in expanded["cases"] if item["stage_id"] == "smoke" and item["mode"] == "all_m1" and item["case_id"] != oracle["case_id"])
            other_path = root / f"{other_oracle['case_id']}.case.json"; write_json(other_path, other_oracle)
            other_raw, ran = self.raw(bound, root, other_path, other_oracle); self.assertEqual(ran.returncode, 0, ran.stderr)
            other_result, built = self.result(bound, root, other_path, other_oracle, other_raw); self.assertEqual(built.returncode, 0, built.stderr)
            rejected_output, rejected = self.result(bound, root, candidate_path, candidate, candidate_raw, path_result=other_result, path_calibration=unexpected_path)
            self.assertNotEqual(rejected.returncode, 0); self.assertFalse(rejected_output.exists())

            missing, rejected = self.result(bound, root, candidate_path, candidate, candidate_raw, path_result=oracle_result, path_calibration=False)
            self.assertNotEqual(rejected.returncode, 0); self.assertFalse(missing.exists())

            candidate_result, built = self.result(bound, root, candidate_path, candidate, candidate_raw, path_result=oracle_result, path_calibration=unexpected_path); self.assertEqual(built.returncode, 0, built.stderr)
            tampered_result = json.loads(candidate_result.read_text())
            tampered_result["calibration"]["path_gate"]["path_oracle"]["path_oracle_calibration_manifest_sha256"] = "0" * 64
            write_json(candidate_result, tampered_result)
            report = root / "path-chain-report.json"
            checked = invoke(VALIDATE, "--run-root", str(root), "--expanded", str(bound["expanded_path"]), "--identity", str(bound["identity"]), "--policy", str(bound["policy"]), "--source-oracle", str(bound["source"]), "--result", str(oracle_result), "--result", str(candidate_result), "--output", str(report))
            self.assertNotEqual(checked.returncode, 0)
            self.assertTrue(any(code.startswith("calibration_path_gate:") for code in json.loads(report.read_text())["failure_codes"]))

            oracle_result.unlink()
            rejected_output, rejected = self.result(bound, root, oracle_path, oracle, oracle_raw, path_calibration=unexpected_path)
            self.assertNotEqual(rejected.returncode, 0); self.assertFalse(rejected_output.exists())

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
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bundle = ROOT / "tests/fixtures/production-execution-trace-p1/schema-r1"
            manifest = json.loads(MANIFEST.read_text()); manifest["identity_binding"]["sampling"] = None; manifest["identity_binding"]["implementation_id"] = "cpu_fixture_v1"
            manifest["axes"]["devices"] = [{"device_id": "cpu-fixture", "backend": "cpu", "name": "CPU fixture", "architecture": "x86_64", "runtime_device_index": 0, "required": True}]
            manifest["axes"]["controls"] = [{"control_id": "aq4_0_target", "role": "target", "format_id": "AQ4_0", "implementation_id": "cpu_fixture_v1", "trace_control": None, "promotion_eligible": False}]
            manifest["stages"] = [{"stage_id": "trace", "order": 1, "fixture_id": "cpu-production-trace-r1", "devices": ["cpu-fixture"], "controls": ["aq4_0_target"], "sampling": None, "prefill": {"prompt_tokens": [4], "requested_m": [4], "modes": ["all_m1", "cold_batched"], "scopes": ["full_model"], "cached_prefix_tokens": [], "cached_prefix_prompt_tokens": [], "generated_tokens": 1}, "decode": {"start_context_tokens": [4], "request_count": 1, "generated_tokens": 1, "scopes": []}, "expected_case_count": {"prefill": 2, "decode": 0, "total": 2}}]
            manifest_path = root / "manifest.json"; write_json(manifest_path, manifest)
            expanded_path = root / "expanded.json"; expanded_run = invoke(EXPAND, "--manifest", str(manifest_path), "--output", str(expanded_path)); self.assertEqual(expanded_run.returncode, 0, expanded_run.stderr)
            expanded = json.loads(expanded_path.read_text()); case = next(item for item in expanded["cases"] if item["mode"] == "cold_batched")
            model_path = root / "model.json"; write_json(model_path, json.loads((bundle / "verified.json").read_text())["identity"]["model"])
            tokenizer = root / "tokenizer.json"; write_json(tokenizer, {"fixture": True}); graph = root / "graph.json"; write_json(graph, {"fixture": True}); state = root / "state.json"; write_json(state, {"fixture": True})
            source = root / "source.json"; write_json(source, {"schema_version": "ullm.qwen35_aq4_source_oracle.v1", "oracle_kind": "independent_source", "status": "fixture"})
            power = root / "power.json"; write_json(power, {"policy_binding": {"expected_power_limit_watts": 1, "allowed_power_tolerance_watts": 0, "maximum_temperature_c": 100, "minimum_vram_headroom_bytes": 1}})
            correctness = root / "correctness.json"; write_json(correctness, {"max_hidden_relative_l2": 1, "max_hidden_max_abs": 1, "max_logits_relative_l2": 1, "max_logits_max_abs": 1, "minimum_top_k_overlap": 1})
            baseline = root / "baseline.json"; write_json(baseline, {"prefill_tokens_per_second_p50": 1, "prefill_tokens_per_second_p95": 1, "oom": False})
            identity_path = root / "identity.json"; policy_path = root / "policy.json"
            bind_args = ["--manifest", str(manifest_path), "--expanded", str(expanded_path), "--policy", str(POLICY), "--worker", str(bundle / "worker.bin"), "--package-root", str(bundle), "--package-manifest", str(bundle / "package.json"), "--tokenizer", str(tokenizer), "--served-model-manifest", str(bundle / "manifest.json"), "--model-identity", str(model_path), "--graph", str(graph), "--state", str(state), "--source-oracle", str(source), "--power-capture", str(power), "--correctness-thresholds", str(correctness), "--baseline-result", str(baseline), "--effective-at", "2026-07-14T12:00:00Z", "--git-commit", "a" * 40, "--output", str(identity_path), "--bound-policy", str(policy_path)]
            production_rejected = invoke(BIND, *bind_args); self.assertNotEqual(production_rejected.returncode, 0); self.assertIn("official production case-set", production_rejected.stderr)
            bound = invoke(BIND, *bind_args, "--fixture-only"); self.assertEqual(bound.returncode, 0, bound.stderr)
            self.assertEqual(json.loads(identity_path.read_text())["evidence_class"], "fixture_only"); self.assertFalse(json.loads(identity_path.read_text())["promotion_eligible"])
            identity = json.loads(identity_path.read_text()); trace_path = bundle / "verified.json"; trace = json.loads(trace_path.read_text()); trace_sha = sha(trace_path)
            fields = ("fixture_id", "scope", "phase", "mode", "prompt_tokens", "cached_prefix_tokens", "context_tokens", "decode_start_tokens", "generated_tokens", "prefill_requested_m", "resolved_m", "request_count", "decode_request_count", "sampling", "control", "control_id", "format_id", "implementation_id", "device")
            raw = {"case_contract": {key: case.get(key) for key in fields}, "links": {"trace": {"path": str(trace_path), "sha256": trace_sha, "trace_id": trace["trace_id"]}}}
            spec = importlib.util.spec_from_file_location("p2_build_association", BUILD); assert spec and spec.loader
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            args = SimpleNamespace(trace=trace_path, trace_manifest=bundle / "manifest.json", trace_executor_record=bundle / "executor-record.json", trace_binding=bundle / "verified-binding.json", trace_report=bundle / "report.json", trace_source=[])
            strict_trace, _ = module.validate_trace_bundle(args, ROOT, case)
            measurement = {"measured_runs": [{} for _ in range(10)], "trace_aggregation": {"schema_version": "ullm.aq4_p2_trace_aggregation.v1", "case_id": case["case_id"], "trace_id": trace["trace_id"], "trace_sha256": trace_sha, "sample_count": 10, "phase_wall_time_ms": {item["phase_id"]: item["wall_time_ms"] for item in trace["phases"]}, "terminal_audit_sha256": module.trace_terminal_sha(trace)}}
            module.validate_trace_association(strict_trace, case, raw, identity, measurement, trace_path, trace_sha)
            other_decode = next((item for item in expanded["cases"] if item["phase"] == "decode"), copy.deepcopy(case)); other_decode.update({"case_id": "other-decode-case", "fixture_id": "other-decode-case", "phase": "decode"})
            with self.assertRaises(module.ResultError): module.validate_trace_association(trace, other_decode, raw, identity, measurement, trace_path, trace_sha)
            other_prefill_raw = copy.deepcopy(raw); other_prefill_raw["case_contract"] = {**raw["case_contract"], "fixture_id": "other-prefill"}
            with self.assertRaises(module.ResultError): module.validate_trace_association(trace, case, other_prefill_raw, identity, measurement, trace_path, trace_sha)


if __name__ == "__main__": unittest.main()
