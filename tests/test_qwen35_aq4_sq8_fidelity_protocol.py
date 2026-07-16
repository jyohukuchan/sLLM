from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "qwen35_aq4_sq8_fidelity_protocol.py"
spec = importlib.util.spec_from_file_location("sq8_protocol", SCRIPT)
assert spec and spec.loader
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Sq8ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.split = self.root / "split"
        self.split.mkdir()
        rows = []
        for index in range(48):
            rows.append({
                "case_id": f"sq8-case-{index:02d}",
                "case_sha256": f"{index + 1:064x}",
                "fixture_sha256": f"{index + 101:064x}",
                "prompt_token_ids_sha256": f"{index + 201:064x}",
                "context_token_ids_sha256": f"{index + 301:064x}",
                "prompt_tokens": 1011 + index,
                "cached_prefix_tokens": 0,
                "context_tokens": 1011 + index,
                "generated_tokens": 0,
                "baseline_mode": "all_m1",
                "prefill_requested_m": 1,
                "resolved_m": 1,
                "step": 0,
                "row_count": 1,
                "subset": "calibration" if index < 24 else "holdout",
            })
        for name, subset in (("calibration-cases.jsonl", rows[:24]), ("holdout-cases.jsonl", rows[24:])):
            (self.split / name).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in subset))
        policy = protocol.policy()
        (self.split / "policy.json").write_text(json.dumps(policy, sort_keys=True) + "\n")
        manifest = {
            "schema_version": "ullm.qwen35_aq4_sq8_fidelity_split.v1",
            "status": "ready_for_calibration",
            "selected_case_count": 48,
            "calibration_case_count": 24,
            "holdout_case_count": 24,
            "calibration_sha256": digest(self.split / "calibration-cases.jsonl"),
            "holdout_sha256": digest(self.split / "holdout-cases.jsonl"),
            "policy_sha256": digest(self.split / "policy.json"),
        }
        (self.split / "split-manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
        self.source_v32 = self.root / "source-v32.json"
        self.source_v32.write_text('{"source":"fixture-v32"}\n')
        self.receipt_dir = self.root / "receipt"
        self.receipt_dir.mkdir()
        self.product_root = self.root / "product"
        (self.product_root / "artifacts").mkdir(parents=True)
        (self.product_root / "package").mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _receipt(self, *, actual: bool = True) -> Path:
        request = "sq8-promotion-" + "a" * 64
        worker_path = self.receipt_dir / f"worker-{len(list(self.receipt_dir.glob('worker-*'))):03d}"
        worker_path.write_bytes(b"worker")
        worker_path.chmod(0o555)
        binding_path = self.product_root / "artifacts" / "binding.json"
        binding_value = {"schema_version": "ullm.qwen35_aq4_sq8_qkv_z_overlay.v2", "format_id": "AQ4_0", "overlay_format_id": "SQ8_0", "implementation_id": "qwen35_aq4_sq8_linear_qkv_z_overlay_v1", "content_sha256": "f" * 64, "tensor_set_sha256": "1" * 64, "tensor_names": [f"tensor-{index:02d}" for index in range(48)], "package": {"manifest_sha256": ""}}
        binding_path.write_text(json.dumps(binding_value, sort_keys=True) + "\n")
        package_path = self.product_root / "package" / "manifest.json"
        package_path.write_text('{"package":"sq8"}\n')
        package = {"manifest_path": str(package_path.resolve()), "manifest_sha256": digest(package_path)}
        binding_value["package"]["manifest_sha256"] = package["manifest_sha256"]
        binding_path.write_text(json.dumps(binding_value, sort_keys=True) + "\n")
        overlay = {"binding_manifest_path": str(binding_path.resolve()), "binding_manifest_sha256": digest(binding_path), "content_sha256": "f" * 64, "tensor_set_sha256": "1" * 64, "tensor_count": 48, "artifact_inventory": {"root": str((self.product_root / "artifacts").resolve()), "uid": 0, "gid": 0, "directory_count": 1, "directory_mode": "0555", "regular_file_count": 1, "regular_file_bytes": binding_path.stat().st_size, "regular_file_mode": "0444", "regular_file_nlink": 1, "symlink_count": 0, "special_count": 0, "entries": [{"path": ".", "kind": "directory", "mode": "0555", "uid": 0, "gid": 0, "nlink": 1, "bytes": 0}, {"path": "binding.json", "kind": "regular", "mode": "0444", "uid": 0, "gid": 0, "nlink": 1, "bytes": binding_path.stat().st_size}]}}
        source = {"tree_sha256": "3" * 40, "archive_sha256": "4" * 64}
        prepared_path = self.receipt_dir / ("prepared.json" if actual else "prepared-only.json")
        readiness = {"schema": "ullm.bridge_container_readiness.v1", "container": {"name": "open-webui", "id": "1" * 64, "image_id": "sha256:" + "2" * 64, "config_image": "ullm/open-webui:test"}, "network": {"name": "open-webui-network", "id": "3" * 64, "driver": "bridge", "bridge_interface": "br-" + "3" * 12}, "endpoint": {"url": "http://172.20.0.1:8000/readyz", "path": "/readyz", "expected_status": 200, "expected_body": '{"status":"ready"}', "expected_body_sha256": hashlib.sha256(b'{"status":"ready"}').hexdigest(), "timeout_seconds": 5}}
        audit_path = self.receipt_dir / "authorization-audit.json"
        audit_path.write_text(json.dumps({"schema_version": "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1", "verdict": "implementation_ready", "actual": "not_executed", "fixed_request_id": request}) + "\n")
        profile_path = self.receipt_dir / "profile.json"
        profile_value = {"schema_version": "ullm.served_model.profile.v1", "format": {"implementation_id": "qwen35_aq4_sq8_linear_qkv_z_overlay_v1"}, "worker": {"binary": str(worker_path.resolve()), "identity": {"execution_profile": "rdna4_aq4_resident_sq8_linear_qkv_z_overlay"}}, "product": {"root": str(self.product_root.resolve()), "artifact": {"manifest_path": "artifacts/binding.json", "content_sha256_from_receipt": ["overlay", "content_sha256"]}, "package": {"manifest_path": "package/manifest.json"}}, "promotion": {"receipt": str(prepared_path.resolve()), "source_commit_from_receipt": ["source_commit"], "required_schema_version": protocol.SQ8_RECEIPT_SCHEMA, "overlay_from_receipt": ["overlay"], "release_from_receipt": ["release"], "package_from_receipt": ["package"], "actual_evidence_from_receipt": ["actual"], "request_id_from_receipt": ["request_id"], "authorization_audit_from_receipt": ["authorization_audit"], "readiness_from_receipt": ["readiness"], "readiness": readiness, "release_source_commit": "5" * 40}}
        profile_path.write_text(json.dumps(profile_value, sort_keys=True) + "\n")
        served_path = self.receipt_dir / "served-model.json"
        served_value = {"schema_version": "ullm.served_model.v2", "promotion": {"receipt_sha256": "0" * 64, "implementation_id": "qwen35_aq4_sq8_linear_qkv_z_overlay_v1"}}
        served_path.write_text(json.dumps(served_value, sort_keys=True) + "\n")
        served = {"path": str(served_path.resolve()), "semantic_sha256": protocol._served_semantic_sha256(served_path)}
        worker = {"path": str(worker_path.resolve()), "sha256": digest(worker_path), "bytes": worker_path.stat().st_size, "mode": "0555", "nlink": 1}
        profile = {"path": str(profile_path.resolve()), "sha256": digest(profile_path)}
        release = {"worker": worker, "profile": profile, "served_model": served}
        auth = {"path": str(audit_path.resolve()), "sha256": digest(audit_path)}
        prepared = {"schema_version": protocol.SQ8_RECEIPT_SCHEMA, "status": "prepared_not_executed", "request_id": request, "source_commit": "5" * 40, "source_provenance": source, "release": release, "overlay": overlay, "package": package, "authorization_audit": auth, "readiness": readiness, "actual": {"status": "pending", "required": True}}
        prepared_path.write_text(json.dumps(prepared, sort_keys=True) + "\n")
        maintenance = self.receipt_dir / "maintenance.json"; maintenance.write_text(json.dumps({"schema_version": "ullm.qwen35_aq4.sq8_overlay_gpu_promotion_maintenance.v1", "status": "passed", "actual_run_count": 1, "failure": None, "promotion_request_id": request, "candidate_pre": {}, "candidate_post": {}, "stopped_observations": [{"service": {"active": False, "running": False, "main_pid": 0, "worker_pid": 0, "lock_owned": False}, "owners": {"worker_pids": [], "amd_pids": [], "kfd_pids": []}}] * 2, "vram_headroom_bytes": 1, "lock": {"held": True, "released": True, "path": "/run/ullm/device-1.lock"}, "restore": {"attempted": True, "passed": True}}) + "\n")
        telemetry = {"schema_version": "ullm.qwen35_aq4.sq8_promotion_telemetry.v1", "projection": {"single_matvec_count": 0, "batch_matvec_count": 1, "pair_matvec_count": 1, "triple_matvec_count": 0, "fallback_count": 0}, "diagnostic_host_staging": {"read_count": 0, "write_count": 0, "read_bytes": 0, "write_bytes": 0}}
        binding = {"schema_version": "ullm.qwen35_aq4.sq8_promotion_telemetry_binding.v1", "request_id": request, "hash_encoding": "canonical_json_ascii_sort_keys_compact_v1", "telemetry_sha256": protocol.sha_bytes(protocol.canonical(telemetry))}
        manifest_identity = {"implementation_id": "qwen35_aq4_sq8_linear_qkv_z_overlay_v1", "execution_profile": "rdna4_aq4_resident_sq8_linear_qkv_z_overlay", "artifact_content_sha256": overlay["content_sha256"], "artifact_manifest_sha256": overlay["binding_manifest_sha256"], "package_manifest_sha256": package["manifest_sha256"]}
        output_identity = {"token_count": 2, "token_ids_sha256": "6" * 64, "token_ids_recorded": False}
        executor = self.receipt_dir / "executor.json"; executor.write_text(json.dumps({"schema_version": "ullm.production_executor_record.v1", "status": "ok", "sq8_promotion_evidence": {"schema_version": "ullm.qwen35_aq4.sq8_promotion_executor.v1", "request_id": request, "manifest_identity": manifest_identity, "telemetry": telemetry, "telemetry_binding": binding, "output_identity": output_identity}}, sort_keys=True) + "\n")
        actual_value = {"status": "actual_verified", "required": True, "prepared_receipt": {"path": str(prepared_path.resolve()), "sha256": digest(prepared_path)}, "maintenance_evidence": {"path": maintenance.name, "sha256": digest(maintenance)}, "executor_record": {"path": executor.name, "sha256": digest(executor)}, "gpu_exclusive_preflight": {"mode": "maintenance_stable2", "stable_observation_count": 2, "worker_pids": [], "amd_smi_owners": [], "kfd_owners": [], "lock": {"path": "/run/ullm/device-1.lock", "free": True}, "vram_headroom_bytes": 1}, "telemetry": telemetry, "telemetry_binding": binding, "manifest_identity": manifest_identity, "output_identity": output_identity}
        receipt = {"schema_version": protocol.SQ8_RECEIPT_SCHEMA, "status": "actual_verified" if actual else "prepared_not_executed", "request_id": request, "source_commit": "5" * 40, "source_provenance": source, "release": release, "overlay": overlay, "package": package, "authorization_audit": auth, "readiness": readiness, "actual": actual_value if actual else {"status": "pending", "required": True}}
        path = self.receipt_dir / ("actual.json" if actual else "prepared-only.json")
        path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        return path

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, text=True, capture_output=True)

    def _plan(self, actual: bool = True, name: str | None = None) -> Path:
        receipt = self._receipt(actual=actual)
        output = self.root / (name or ("plan.json" if actual else "preflight-plan.json"))
        result = self._run("plan", "--split-root", str(self.split), "--actual-receipt", str(receipt), "--source-v32", str(self.source_v32), "--output", str(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        return output

    def _metrics(self, plan: Path, subset: str) -> Path:
        value = json.loads(plan.read_text())
        rows = json.loads("[" + ",".join((self.split / ("calibration-cases.jsonl" if subset == "calibration" else "holdout-cases.jsonl")).read_text().splitlines()) + "]")
        for row in rows:
            row["metrics"] = {name: (1.0 if name in protocol.BINARY_METRICS else (2.0 if name == "hidden_max_abs" else 0.8)) for name in protocol.METRIC_POLICY}
        path = self.root / f"{subset}-metrics.json"
        path.write_text(json.dumps({"schema_version": protocol.METRICS_SCHEMA, "identity": value["identity"], "subset": subset, "rows": rows}, sort_keys=True) + "\n")
        return path

    def test_actual_receipt_binds_plan_and_freeze_recomputes(self) -> None:
        plan = self._plan()
        metrics = self._metrics(plan, "calibration")
        freeze = self.root / "freeze.json"
        result = self._run("freeze", "--plan", str(plan), "--metrics", str(metrics), "--output", str(freeze))
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(freeze.read_text())
        self.assertEqual(value["calibration_case_count"], 24)
        self.assertEqual(value["holdout_evaluations_remaining"], 1)
        self.assertAlmostEqual(value["derived_bounds"]["topk_overlap_rate_k10"]["bound"], 0.79)
        checked = self._run("validate-freeze", "--plan", str(plan), "--metrics", str(metrics), "--freeze", str(freeze))
        self.assertEqual(checked.returncode, 0, checked.stderr)
        standalone = subprocess.run([sys.executable, str(ROOT / "tools/validate-qwen35-aq4-sq8-fidelity.py"), "--plan", str(plan), "--metrics", str(metrics), "--freeze", str(freeze)], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(standalone.returncode, 0, standalone.stderr)
        value["derived_bounds"]["topk_overlap_rate_k10"]["bound"] = 0.99
        tampered = self.root / "tampered-freeze.json"; tampered.write_text(json.dumps(value) + "\n")
        rejected = self._run("validate-freeze", "--plan", str(plan), "--metrics", str(metrics), "--freeze", str(tampered))
        self.assertNotEqual(rejected.returncode, 0)

    def test_prepared_only_is_preflight_and_freeze_rejected(self) -> None:
        plan = self._plan(actual=False)
        self.assertEqual(json.loads(plan.read_text())["status"], "preflight_only")
        metrics = self._metrics(plan, "calibration")
        result = self._run("freeze", "--plan", str(plan), "--metrics", str(metrics), "--output", str(self.root / "bad-freeze.json"))
        self.assertNotEqual(result.returncode, 0)

    def test_relative_l2_row_rejected(self) -> None:
        plan = self._plan()
        metrics = self._metrics(plan, "calibration")
        value = json.loads(metrics.read_text()); value["rows"][0]["metrics"]["logits_relative_l2"] = 1.01
        metrics.write_text(json.dumps(value) + "\n")
        result = self._run("freeze", "--plan", str(plan), "--metrics", str(metrics), "--output", str(self.root / "bad.json"))
        self.assertNotEqual(result.returncode, 0)

    def test_actual_receipt_identity_and_unknown_field_tamper_is_rejected(self) -> None:
        mutations = (
            ("request_id", lambda value: value.__setitem__("request_id", "sq8-promotion-" + "b" * 64)),
            ("source", lambda value: value["source_provenance"].__setitem__("archive_sha256", "9" * 64)),
            ("overlay-content", lambda value: value["overlay"].__setitem__("content_sha256", "9" * 64)),
            ("overlay-tensor-set", lambda value: value["overlay"].__setitem__("tensor_set_sha256", "9" * 64)),
            ("package", lambda value: value["package"].__setitem__("manifest_sha256", "9" * 64)),
            ("token", lambda value: value["actual"]["output_identity"].__setitem__("token_ids_sha256", "9" * 64)),
            ("telemetry", lambda value: value["actual"]["telemetry_binding"].__setitem__("telemetry_sha256", "9" * 64)),
            ("maintenance", lambda value: value["actual"]["maintenance_evidence"].__setitem__("sha256", "9" * 64)),
            ("inventory-unknown", lambda value: value["overlay"]["artifact_inventory"].__setitem__("unexpected", True)),
            ("inventory-root", lambda value: value["overlay"]["artifact_inventory"].__setitem__("root", "/tmp/not-the-binding-root")),
            ("inventory-nlink", lambda value: value["overlay"]["artifact_inventory"]["entries"][1].__setitem__("nlink", 2)),
            ("unknown", lambda value: value.__setitem__("unexpected", True)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                receipt = self._receipt()
                value = json.loads(receipt.read_text()); mutate(value); receipt.write_text(json.dumps(value) + "\n")
                result = self._run("plan", "--split-root", str(self.split), "--actual-receipt", str(receipt), "--source-v32", str(self.source_v32), "--output", str(self.root / f"bad-{name}.json"))
                self.assertNotEqual(result.returncode, 0)

    def test_metrics_are_strict_24_rows_and_reject_unknown_missing_duplicate_nonfinite(self) -> None:
        plan = self._plan()
        metrics = self._metrics(plan, "calibration")
        base = json.loads(metrics.read_text())
        cases = []
        missing = json.loads(json.dumps(base)); missing["rows"][0]["metrics"].pop("topk_overlap_rate_k10"); cases.append(missing)
        unknown = json.loads(json.dumps(base)); unknown["rows"][0]["metrics"]["unexpected"] = 0.5; cases.append(unknown)
        duplicate = json.dumps(base).replace('"rows":', '"rows":', 1)
        nonfinite = json.loads(json.dumps(base)); nonfinite["rows"][0]["metrics"]["logits_cosine"] = float("nan"); cases.append(nonfinite)
        for index, value in enumerate(cases):
            path = self.root / f"strict-{index}.json"; path.write_text(json.dumps(value, allow_nan=True) + "\n")
            result = self._run("freeze", "--plan", str(plan), "--metrics", str(path), "--output", str(self.root / f"strict-out-{index}.json"))
            self.assertNotEqual(result.returncode, 0)
        duplicate_path = self.root / "duplicate.json"
        duplicate_path.write_text('{"schema_version":"' + protocol.METRICS_SCHEMA + '","schema_version":"' + protocol.METRICS_SCHEMA + '"}\n')
        result = self._run("freeze", "--plan", str(plan), "--metrics", str(duplicate_path), "--output", str(self.root / "duplicate-out.json"))
        self.assertNotEqual(result.returncode, 0)

    def test_plan_resource_limits_and_vram_headroom_are_bound(self) -> None:
        plan = self._plan()
        value = json.loads(plan.read_text())
        self.assertEqual(value["resource_contract"]["jobs"], 1)
        self.assertEqual(value["resource_contract"]["chunk_elements"], 65_536)
        for field, bad in (("jobs", 2), ("case_concurrency", 2), ("chunk_elements", 131_072), ("vram_headroom_bytes_min", 0)):
            tampered = json.loads(json.dumps(value)); tampered["resource_contract"][field] = bad
            path = self.root / f"plan-{field}.json"; path.write_text(json.dumps(tampered) + "\n")
            metrics = self._metrics(path, "calibration")
            result = self._run("freeze", "--plan", str(path), "--metrics", str(metrics), "--output", str(self.root / f"bad-resource-{field}.json"))
            self.assertNotEqual(result.returncode, 0)

    def test_source_v32_tamper_is_rejected_after_plan_creation(self) -> None:
        plan = self._plan()
        self.source_v32.write_text('{"source":"tampered"}\n')
        metrics = self._metrics(plan, "calibration")
        result = self._run("freeze", "--plan", str(plan), "--metrics", str(metrics), "--output", str(self.root / "bad-source-v32.json"))
        self.assertNotEqual(result.returncode, 0)

    def test_bound_actual_receipt_and_executor_files_are_rechecked(self) -> None:
        plan = self._plan(); receipt = Path(json.loads(plan.read_text())["identity"]["sq8_receipt_path"]); metrics = self._metrics(plan, "calibration")
        maintenance = self.receipt_dir / "maintenance.json"; maintenance.write_text("{}\n")
        result = self._run("freeze", "--plan", str(plan), "--metrics", str(metrics), "--output", str(self.root / "bad-maintenance.json"))
        self.assertNotEqual(result.returncode, 0)
        maintenance.write_text(json.dumps({"promotion_request_id": "sq8-promotion-" + "a" * 64}) + "\n")
        value = json.loads(receipt.read_text()); value["source_commit"] = "8" * 40; receipt.write_text(json.dumps(value) + "\n")
        result = self._run("freeze", "--plan", str(plan), "--metrics", str(metrics), "--output", str(self.root / "bad-receipt.json"))
        self.assertNotEqual(result.returncode, 0)

    def test_crash_consumes_attempt_and_retry_is_refused(self) -> None:
        plan = self._plan(); metrics = self._metrics(plan, "calibration"); freeze = self.root / "freeze.json"
        self.assertEqual(self._run("freeze", "--plan", str(plan), "--metrics", str(metrics), "--output", str(freeze)).returncode, 0)
        preflight = self.root / "preflight.json"
        self.assertEqual(self._run("preflight-holdout", "--plan", str(plan), "--freeze", str(freeze), "--output", str(preflight)).returncode, 0)
        holdout = self._metrics(plan, "holdout"); ledger = self.root / "ledger.json"; result = self._run("execute-holdout", "--preflight", str(preflight), "--metrics", str(holdout), "--ledger", str(ledger), "--output", str(self.root / "result.json"), "--crash-after-sentinel")
        self.assertNotEqual(result.returncode, 0); self.assertTrue(ledger.exists())
        retry = self._run("execute-holdout", "--preflight", str(preflight), "--metrics", str(holdout), "--ledger", str(ledger), "--output", str(self.root / "retry.json"))
        self.assertNotEqual(retry.returncode, 0)

    def test_holdout_success_is_single_consumed_result(self) -> None:
        plan = self._plan(); calibration = self._metrics(plan, "calibration"); freeze = self.root / "freeze.json"
        self.assertEqual(self._run("freeze", "--plan", str(plan), "--metrics", str(calibration), "--output", str(freeze)).returncode, 0)
        preflight = self.root / "preflight.json"
        self.assertEqual(self._run("preflight-holdout", "--plan", str(plan), "--freeze", str(freeze), "--output", str(preflight)).returncode, 0)
        output = self.root / "holdout-result.json"; ledger = self.root / "success-ledger.json"; holdout = self._metrics(plan, "holdout")
        result = self._run("execute-holdout", "--preflight", str(preflight), "--metrics", str(holdout), "--ledger", str(ledger), "--output", str(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text())["status"], "passed")
        self.assertEqual(json.loads(ledger.read_text())["remaining_after"], 0)

    def test_holdout_gate_failure_still_consumes_remaining_zero(self) -> None:
        plan = self._plan(); calibration = self._metrics(plan, "calibration"); freeze = self.root / "freeze-failure.json"
        self.assertEqual(self._run("freeze", "--plan", str(plan), "--metrics", str(calibration), "--output", str(freeze)).returncode, 0)
        preflight = self.root / "preflight-failure.json"
        self.assertEqual(self._run("preflight-holdout", "--plan", str(plan), "--freeze", str(freeze), "--output", str(preflight)).returncode, 0)
        holdout = self._metrics(plan, "holdout"); value = json.loads(holdout.read_text())
        for row in value["rows"]:
            for name in ("topk_overlap_rate_k10", "logits_cosine", "hidden_cosine"):
                row["metrics"][name] = 0.1
            for name in ("logits_relative_l2", "hidden_relative_l2"):
                row["metrics"][name] = 0.9
        holdout.write_text(json.dumps(value) + "\n")
        output = self.root / "failed-result.json"; ledger = self.root / "failure-ledger.json"
        result = self._run("execute-holdout", "--preflight", str(preflight), "--metrics", str(holdout), "--ledger", str(ledger), "--output", str(output))
        self.assertNotEqual(result.returncode, 0)
        result_value = json.loads(output.read_text())
        self.assertEqual(result_value["status"], "failed")
        self.assertEqual(result_value["evaluations_remaining"], 0)

    def test_preexisting_ledger_or_output_is_fail_closed_without_staging_leak(self) -> None:
        plan = self._plan(); calibration = self._metrics(plan, "calibration"); freeze = self.root / "freeze-boundary.json"
        self.assertEqual(self._run("freeze", "--plan", str(plan), "--metrics", str(calibration), "--output", str(freeze)).returncode, 0)
        preflight = self.root / "preflight-boundary.json"
        self.assertEqual(self._run("preflight-holdout", "--plan", str(plan), "--freeze", str(freeze), "--output", str(preflight)).returncode, 0)
        holdout = self._metrics(plan, "holdout")
        existing_ledger = self.root / "existing-ledger.json"; existing_ledger.write_text("{}\n")
        rejected = self._run("execute-holdout", "--preflight", str(preflight), "--metrics", str(holdout), "--ledger", str(existing_ledger), "--output", str(self.root / "unused-result.json"))
        self.assertNotEqual(rejected.returncode, 0)
        output = self.root / "existing-result.json"; output.write_text("{}\n"); ledger = self.root / "output-ledger.json"
        rejected = self._run("execute-holdout", "--preflight", str(preflight), "--metrics", str(holdout), "--ledger", str(ledger), "--output", str(output))
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(list(self.root.glob(".*.incomplete")), [])

    def test_six_independent_mutations_are_rejected(self) -> None:
        prepared = self._plan(actual=False)
        prepared_value = json.loads(prepared.read_text())
        prepared_value["status"] = "ready_for_calibration"
        prepared_value["preflight_only"] = False
        escalated = self.root / "escalated-plan.json"; escalated.write_text(json.dumps(prepared_value) + "\n")
        metrics = self._metrics(escalated, "calibration")
        self.assertNotEqual(self._run("freeze", "--plan", str(escalated), "--metrics", str(metrics), "--output", str(self.root / "escalated-freeze.json")).returncode, 0)

        plan = self._plan(name="plan-stale.json"); calibration = self._metrics(plan, "calibration")
        freeze = self.root / "stale-freeze.json"
        self.assertEqual(self._run("freeze", "--plan", str(plan), "--metrics", str(calibration), "--output", str(freeze)).returncode, 0)
        forged = json.loads(freeze.read_text()); forged["derived_bounds"]["logits_cosine"]["bound"] = 0.0; stale_freeze = self.root / "forged-freeze.json"; stale_freeze.write_text(json.dumps(forged) + "\n")
        self.assertNotEqual(self._run("preflight-holdout", "--plan", str(plan), "--freeze", str(stale_freeze), "--output", str(self.root / "stale-preflight.json")).returncode, 0)

        plan = self._plan(name="plan-case.json"); calibration = self._metrics(plan, "calibration")
        freeze = self.root / "case-freeze.json"
        self.assertEqual(self._run("freeze", "--plan", str(plan), "--metrics", str(calibration), "--output", str(freeze)).returncode, 0)
        with (self.split / "holdout-cases.jsonl").open("a") as stream: stream.write("{}\n")
        self.assertNotEqual(self._run("preflight-holdout", "--plan", str(plan), "--freeze", str(freeze), "--output", str(self.root / "stale-case-preflight.json")).returncode, 0)
        self.split.joinpath("holdout-cases.jsonl").write_text("\n".join(self.split.joinpath("holdout-cases.jsonl").read_text().splitlines()[:24]) + "\n")

        plan = self._plan(name="plan-resource.json"); value = json.loads(plan.read_text()); value["resource_contract"]["unexpected"] = True; unknown_plan = self.root / "unknown-resource-plan.json"; unknown_plan.write_text(json.dumps(value) + "\n")
        calibration = self._metrics(unknown_plan, "calibration")
        self.assertNotEqual(self._run("freeze", "--plan", str(unknown_plan), "--metrics", str(calibration), "--output", str(self.root / "unknown-resource-freeze.json")).returncode, 0)

        receipt = self._receipt(); value = json.loads(receipt.read_text()); value["actual"]["manifest_identity"]["execution_profile"] = "wrong"; wrong_receipt = self.root / "wrong-profile-receipt.json"; wrong_receipt.write_text(json.dumps(value) + "\n")
        self.assertNotEqual(self._run("plan", "--split-root", str(self.split), "--actual-receipt", str(wrong_receipt), "--source-v32", str(self.source_v32), "--output", str(self.root / "wrong-profile-plan.json")).returncode, 0)

        receipt = self._receipt(); value = json.loads(receipt.read_text()); value["readiness"]["network"]["id"] = "4" * 64; wrong_readiness = self.root / "wrong-readiness-receipt.json"; wrong_readiness.write_text(json.dumps(value) + "\n")
        self.assertNotEqual(self._run("plan", "--split-root", str(self.split), "--actual-receipt", str(wrong_readiness), "--source-v32", str(self.source_v32), "--output", str(self.root / "wrong-readiness-plan.json")).returncode, 0)

    def test_integer_alias_matrix_and_frozen_contracts_fail_closed(self) -> None:
        for alias in (False, 1.0, -1, protocol.SAFE_INT_MAX + 1):
            with self.subTest(metric_row_alias=repr(alias)):
                plan = self._plan(name=f"plan-row-{len(str(alias))}.json")
                metrics = self._metrics(plan, "calibration")
                value = json.loads(metrics.read_text())
                value["rows"][0]["row_count"] = alias
                metrics.write_text(json.dumps(value, allow_nan=True) + "\n")
                result = self._run("freeze", "--plan", str(plan), "--metrics", str(metrics), "--output", str(self.root / f"row-alias-{len(str(alias))}.json"))
                self.assertNotEqual(result.returncode, 0)

        plan = self._plan(name="plan-resource-alias.json")
        plan_value = json.loads(plan.read_text())
        for alias in (True, 1.0, -1, protocol.SAFE_INT_MAX + 1):
            with self.subTest(resource_jobs_alias=repr(alias)):
                tampered = json.loads(json.dumps(plan_value))
                tampered["resource_contract"]["jobs"] = alias
                path = self.root / f"resource-alias-{len(str(alias))}.json"
                path.write_text(json.dumps(tampered, allow_nan=True) + "\n")
                metrics = self._metrics(path, "calibration")
                result = self._run("freeze", "--plan", str(path), "--metrics", str(metrics), "--output", str(self.root / f"resource-alias-out-{len(str(alias))}.json"))
                self.assertNotEqual(result.returncode, 0)
        for field, alias in (("case_concurrency", 1.0), ("chunk_elements", False), ("max_rows", 24.0), ("max_case_file_bytes", True), ("vram_headroom_bytes_min", 1.0), ("vram_observed_headroom_bytes", True), ("vram_headroom_required", 1)):
            with self.subTest(resource_field_alias=field):
                tampered = json.loads(json.dumps(plan_value))
                tampered["resource_contract"][field] = alias
                path = self.root / f"resource-field-alias-{field}.json"
                path.write_text(json.dumps(tampered, allow_nan=True) + "\n")
                metrics = self._metrics(path, "calibration")
                result = self._run("freeze", "--plan", str(path), "--metrics", str(metrics), "--output", str(self.root / f"resource-field-alias-out-{field}.json"))
                self.assertNotEqual(result.returncode, 0)

        receipt_aliases = (
            ("overlay", "tensor_count", 48.0),
            ("worker", "bytes", 6.0),
            ("worker", "nlink", True),
            ("gpu", "stable_observation_count", 2.0),
            ("output", "token_count", 2.0),
        )
        for component, field, alias in receipt_aliases:
            with self.subTest(receipt_integer_alias=f"{component}.{field}"):
                receipt = self._receipt()
                value = json.loads(receipt.read_text())
                if component == "overlay":
                    value["overlay"][field] = alias
                elif component == "worker":
                    value["release"]["worker"][field] = alias
                elif component == "gpu":
                    value["actual"]["gpu_exclusive_preflight"][field] = alias
                else:
                    value["actual"]["output_identity"][field] = alias
                receipt.write_text(json.dumps(value) + "\n")
                result = self._run("plan", "--split-root", str(self.split), "--actual-receipt", str(receipt), "--source-v32", str(self.source_v32), "--output", str(self.root / f"receipt-alias-{component}-{field}.json"))
                self.assertNotEqual(result.returncode, 0)

        # The receipt-alias mutations above rewrite the shared fixture receipt;
        # rebuild a clean plan before validating the freeze/preflight contracts.
        plan = self._plan(name="plan-contracts.json")
        calibration = self._metrics(plan, "calibration")
        metrics_identity_alias = json.loads(calibration.read_text())
        metrics_identity_alias["identity"]["worker"]["bytes"] = 6.0
        metrics_identity_path = self.root / "metrics-identity-alias.json"
        metrics_identity_path.write_text(json.dumps(metrics_identity_alias) + "\n")
        self.assertNotEqual(self._run("freeze", "--plan", str(plan), "--metrics", str(metrics_identity_path), "--output", str(self.root / "metrics-identity-alias-freeze.json")).returncode, 0)
        freeze = self.root / "contract-freeze.json"
        self.assertEqual(self._run("freeze", "--plan", str(plan), "--metrics", str(calibration), "--output", str(freeze)).returncode, 0)
        frozen = json.loads(freeze.read_text())
        for field, value in (("calibration_case_count", 0), ("relative_l2_rejection_ceiling", 0.0), ("attempt_boundary", {"remaining_before": 0, "remaining_after": 1, "failure_consumes_attempt": False}), ("holdout_evaluations_remaining", 1.0)):
            with self.subTest(freeze_contract=field):
                tampered = json.loads(json.dumps(frozen))
                tampered[field] = value
                path = self.root / f"freeze-contract-{field}.json"
                path.write_text(json.dumps(tampered) + "\n")
                result = self._run("preflight-holdout", "--plan", str(plan), "--freeze", str(path), "--output", str(self.root / f"preflight-contract-{field}.json"))
                self.assertNotEqual(result.returncode, 0)

        preflight = self.root / "contract-preflight.json"
        self.assertEqual(self._run("preflight-holdout", "--plan", str(plan), "--freeze", str(freeze), "--output", str(preflight)).returncode, 0)
        preflight_value = json.loads(preflight.read_text())
        holdout = self._metrics(plan, "holdout")
        preflight_identity_alias = json.loads(json.dumps(preflight_value))
        preflight_identity_alias["identity"]["worker"]["nlink"] = True
        preflight_identity_path = self.root / "preflight-identity-alias.json"
        preflight_identity_path.write_text(json.dumps(preflight_identity_alias) + "\n")
        self.assertNotEqual(self._run("execute-holdout", "--preflight", str(preflight_identity_path), "--metrics", str(holdout), "--ledger", str(self.root / "identity-alias-ledger.json"), "--output", str(self.root / "identity-alias-result.json")).returncode, 0)
        for field, value in (("holdout_case_count", 0), ("evaluations_remaining", 1.0), ("attempt_boundary", {"remaining_before": 0, "remaining_after": 1, "failure_consumes_attempt": False})):
            with self.subTest(preflight_contract=field):
                tampered = json.loads(json.dumps(preflight_value))
                tampered[field] = value
                path = self.root / f"preflight-contract-{field}.json"
                path.write_text(json.dumps(tampered) + "\n")
                result = self._run("execute-holdout", "--preflight", str(path), "--metrics", str(holdout), "--ledger", str(self.root / f"contract-ledger-{field}.json"), "--output", str(self.root / f"contract-result-{field}.json"))
                self.assertNotEqual(result.returncode, 0)

    def test_ledger_publication_is_create_new_under_two_process_race(self) -> None:
        plan = self._plan(); calibration = self._metrics(plan, "calibration"); freeze = self.root / "race-freeze.json"
        self.assertEqual(self._run("freeze", "--plan", str(plan), "--metrics", str(calibration), "--output", str(freeze)).returncode, 0)
        preflight = self.root / "race-preflight.json"
        self.assertEqual(self._run("preflight-holdout", "--plan", str(plan), "--freeze", str(freeze), "--output", str(preflight)).returncode, 0)
        holdout = self._metrics(plan, "holdout"); ledger = self.root / "race-ledger.json"; output = self.root / "race-result.json"
        command = [sys.executable, str(SCRIPT), "execute-holdout", "--preflight", str(preflight), "--metrics", str(holdout), "--ledger", str(ledger), "--output", str(output)]
        first = subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        second = subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        first_result = first.communicate(timeout=20); second_result = second.communicate(timeout=20)
        self.assertEqual(sum(process.returncode == 0 for process in (first, second)), 1, (first_result, second_result))
        self.assertTrue(ledger.is_file())


if __name__ == "__main__":
    unittest.main()
