from __future__ import annotations

import os
import hashlib
import json
import struct
import subprocess
import tempfile
import unittest
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / (
    "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/"
    "p2/aq4-layer0-qkv-fused-gpu-probe-v0.1/input/run-fused-gpu-probe-gate.sh"
)
PROBE_ROOT = ROOT / (
    "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/"
    "p2/aq4-layer0-qkv-fused-gpu-probe-v0.1/input/probe-binary-v0.1"
)


class Aq4Layer0QkvFusedGpuProbeGateTest(unittest.TestCase):
    def test_shell_syntax_and_fused_contract(self) -> None:
        subprocess.run(["bash", "-n", str(GATE)], check=True)
        text = GATE.read_text(encoding="utf-8")
        for token in (
            'EXPECTED_BUILD_COMMIT="6082df4966190ae4977b699460a5ecb93fee8e34"',
            'EXPECTED_PROBE_BINARY_SHA256="42752e7a29614f59f72f90bed6797c3e925b032bffb1a4196c462c8476386840"',
            'EXPECTED_BUILD_RECEIPT_SHA256="90e9ef6d383f7ef25e9526659f035e40291ba1a5efa7f8ba36340c8b245d9504"',
            "stable2_stopped", "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1",
            "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL=1",
            "ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB=4",
            '"qkv-standalone.f32le"', '"ullm.aq4_layer0_qkv_z_gate_beta_runtime_probe.v2"',
            "promotion_eligible=false", "observer_sample_once", "start_observer", "stop_observer",
            "trap - EXIT INT TERM HUP", "cleanup_rc", "start_rc", '"${SYSTEMCTL[@]}" start "$SERVICE"',
            '"architecture_default:gfx1201"', '"diagnostic_standalone_reference"',
            '"concatenated_little_endian_f32_rows"', '"end_row_exclusive": 8192',
            'mkdir -p -- "$BASE/attempts"', "attempt directory create-new failed", "prepare_runtime_probe",
            'install -m 0555 -- "$PROBE" "$RUNTIME_PROBE"', "runtime-probe-stat.json",
            'MOCK_ARCHIVE_SETUP', 'runtime_probe_mode=0555', 'MOCK_OBSERVER_LOOP',
            'ROCM_SMI_METRIC_ARGS=(metric -g "$EXPECTED_PHYSICAL_CARD" -m -u -p -t --json)',
            'ROCM_SMI_STATIC_ARGS=(static -g "$EXPECTED_PHYSICAL_CARD" -a --json)',
            'target_graphics_version', 'mem_usage',
            'POST_START_TIMEOUT_SECONDS="${POST_START_TIMEOUT_SECONDS:-120}"',
            'post_start_readiness.v1', 'post_start_record_attempt',
            'post_start_check_mock', 'deadline_timeout', 'health_200',
            'additional_gpu_probe_executed', 'systemctl_operations',
        ):
            self.assertIn(token, text)
        self.assertNotIn("--showmeminfo", text)
        self.assertNotIn("--showuse", text)
        self.assertNotIn("--showpower", text)

    def test_mock_preflight_is_read_only(self) -> None:
        self.assertTrue(PROBE_ROOT.is_dir())
        with tempfile.TemporaryDirectory(prefix="ullm-aq4-fused-gate-") as directory:
            env = os.environ.copy()
            env.update(
                MOCK_PREFLIGHT="1",
                MOCK_BASE=directory,
                MOCK_PROBE_ROOT=str(PROBE_ROOT),
                MOCK_PROBE=str(PROBE_ROOT / "ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe"),
                MOCK_RECEIPT=str(PROBE_ROOT / "build-receipt.json"),
                MOCK_SUMS=str(PROBE_ROOT / "SHA256SUMS"),
                PYTHON="/usr/bin/python3",
            )
            result = subprocess.run(["bash", str(GATE)], env=env, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("mock_preflight=1 service_stop=0 gpu_run=0", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertNotIn("systemctl", result.stdout + result.stderr)

            archive_base = Path(directory) / "fresh-base"
            probe_copy = Path(directory) / "probe-checkout-copy"
            shutil.copytree(PROBE_ROOT, probe_copy)
            (probe_copy / "ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe").chmod(0o775)
            archive_env = env.copy()
            archive_env.update(
                MOCK_BASE=str(archive_base), MOCK_ARCHIVE_SETUP="1",
                MOCK_PROBE_ROOT=str(probe_copy),
                MOCK_PROBE=str(probe_copy / "ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe"),
                MOCK_RECEIPT=str(probe_copy / "build-receipt.json"),
                MOCK_SUMS=str(probe_copy / "SHA256SUMS"),
            )
            archive_result = subprocess.run(
                ["bash", str(GATE)], env=archive_env, text=True, capture_output=True, check=False,
            )
            self.assertEqual(archive_result.returncode, 0, archive_result.stderr)
            self.assertIn("mock_archive_setup=1 runtime_probe_mode=0555 runtime_probe_nlink=1", archive_result.stdout)
            attempt = archive_base / "attempts" / "attempt1"
            runtime_probe = attempt / "ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe"
            runtime_meta = attempt / "runtime-probe-stat.json"
            self.assertTrue(runtime_probe.is_file() and not runtime_probe.is_symlink())
            self.assertEqual(runtime_probe.stat().st_nlink, 1)
            self.assertEqual(runtime_probe.stat().st_mode & 0o7777, 0o555)
            self.assertEqual(hashlib.sha256(runtime_probe.read_bytes()).hexdigest(),
                             "42752e7a29614f59f72f90bed6797c3e925b032bffb1a4196c462c8476386840")
            metadata = json.loads(runtime_meta.read_text(encoding="utf-8"))
            self.assertEqual(metadata["runtime"]["sha256"],
                             "42752e7a29614f59f72f90bed6797c3e925b032bffb1a4196c462c8476386840")
            self.assertEqual(metadata["runtime"]["mode"], "0555")
            self.assertEqual(metadata["runtime"]["nlink"], 1)
            self.assertNotIn("systemctl", archive_result.stdout + archive_result.stderr)
            rerun = subprocess.run(
                ["bash", str(GATE)], env=archive_env, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn("refusing to overwrite existing path", rerun.stderr)

    def test_mock_observer_samples_pinned_card_without_service_or_gpu(self) -> None:
        self.assertTrue(PROBE_ROOT.is_dir())
        with tempfile.TemporaryDirectory(prefix="ullm-aq4-fused-observer-") as directory:
            base = Path(directory)
            rocm_smi = base / "fake-amd-smi"
            rocm_smi.write_text(
                "#!/bin/sh\n"
                "case \"$1\" in\n"
                "  metric) printf '%s\\n' '{\"gpu_data\":[{\"gpu\":2,\"usage\":{\"gfx_activity\":{\"value\":1,\"unit\":\"%\"}},\"power\":{\"socket_power\":{\"value\":14,\"unit\":\"W\"}},\"mem_usage\":{\"used_vram\":{\"value\":1,\"unit\":\"MB\"}}}]}' ;;\n"
                "  static) printf '%s\\n' '{\"gpu_data\":[{\"gpu\":2,\"asic\":{\"target_graphics_version\":\"gfx1201\"}}]}' ;;\n"
                "  *) exit 2 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            rocm_smi.chmod(0o755)
            env = os.environ.copy()
            env.update(
                MOCK_PREFLIGHT="1",
                MOCK_OBSERVER="1",
                MOCK_BASE=directory,
                MOCK_PROBE_ROOT=str(PROBE_ROOT),
                MOCK_PROBE=str(PROBE_ROOT / "ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe"),
                MOCK_RECEIPT=str(PROBE_ROOT / "build-receipt.json"),
                MOCK_SUMS=str(PROBE_ROOT / "SHA256SUMS"),
                ROCM_SMI=str(rocm_smi),
                PYTHON="/usr/bin/python3",
            )
            result = subprocess.run(["bash", str(GATE)], env=env, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("mock_observer=1 service_stop=0 gpu_run=0", result.stdout)
            self.assertIn('"physical_card":2', (base / "monitor.log").read_text(encoding="utf-8"))
            self.assertTrue((base / "observer-sample.marker").is_file())
            self.assertFalse((base / "observer-failed.marker").exists())
            self.assertNotIn("systemctl", result.stdout + result.stderr)

            loop_base = base / "observer-loop"
            loop_base.mkdir()
            loop_env = env.copy()
            loop_env.update(MOCK_BASE=str(loop_base), MOCK_OBSERVER_LOOP="1")
            loop_result = subprocess.run(
                ["bash", str(GATE)], env=loop_env, text=True, capture_output=True, check=False,
            )
            self.assertEqual(loop_result.returncode, 0, loop_result.stderr)
            self.assertIn("mock_observer_loop=1 observer_stopped=1", loop_result.stdout)
            self.assertTrue((loop_base / "observer-sample.marker").is_file())
            self.assertFalse((loop_base / "observer-failed.marker").exists())

            failing_smi = base / "fake-amd-smi-failing"
            failing_smi.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
            failing_smi.chmod(0o755)
            failure_base = base / "observer-failure"
            failure_base.mkdir()
            failure_env = env.copy()
            failure_env.update(MOCK_BASE=str(failure_base), ROCM_SMI=str(failing_smi), MOCK_OBSERVER_LOOP="0")
            failure_result = subprocess.run(
                ["bash", str(GATE)], env=failure_env, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(failure_result.returncode, 0)
            self.assertTrue((failure_base / "observer-failed.marker").is_file())
            self.assertFalse((failure_base / "observer-sample.marker").exists())

    def _mock_post_start_env(self, base: Path, **overrides: str) -> dict[str, str]:
        probe = PROBE_ROOT / "ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe"
        env = os.environ.copy()
        env.update(
            {
                "MOCK_PREFLIGHT": "1",
                "MOCK_POST_START_READINESS": "1",
                "MOCK_BASE": str(base),
                "MOCK_PROBE_ROOT": str(PROBE_ROOT),
                "MOCK_PROBE": str(probe),
                "MOCK_RECEIPT": str(PROBE_ROOT / "build-receipt.json"),
                "MOCK_SUMS": str(PROBE_ROOT / "SHA256SUMS"),
                "PYTHON": "/usr/bin/python3",
                "POST_START_TIMEOUT_SECONDS": "1",
                "POST_START_POLL_SECONDS": "0.1",
            }
        )
        env.update(overrides)
        return env

    def test_mock_post_start_retries_owner_and_health_then_records_success(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ullm-aq4-post-start-success-") as directory:
            base = Path(directory)
            env = self._mock_post_start_env(
                base,
                MOCK_POST_START_OWNER_FAILURES="1",
                MOCK_POST_START_HEALTH_FAILURES="1",
            )
            result = subprocess.run(
                ["bash", str(GATE)], env=env, text=True, capture_output=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("mock_post_start_readiness=1 status=passed attempts=2", result.stdout)
            artifact = json.loads(
                (base / "attempts/attempt1/post-start-readiness.json").read_text(encoding="utf-8")
            )
            self.assertEqual(artifact["status"], "passed")
            self.assertEqual(artifact["attempt_count"], 2)
            self.assertGreaterEqual(artifact["elapsed_ns"], 0)
            self.assertEqual(
                artifact["predicate_order"],
                [
                    "service_active", "service_running", "new_main_pid", "nrestarts",
                    "active_hashes", "lock_identity", "flock_held", "owner", "health_200",
                ],
            )
            self.assertIn("owner", artifact["attempts"][0]["failure_reasons"])
            self.assertIn("health_200", artifact["attempts"][0]["failure_reasons"])
            self.assertEqual(artifact["attempts"][-1]["failure_reasons"], {})
            final_conditions = artifact["attempts"][-1]["conditions"]
            for key in ("new_main_pid", "nrestarts", "active_hashes", "lock_identity", "flock_held"):
                self.assertEqual(final_conditions[key], "true")
            self.assertEqual(
                artifact["safety"],
                {
                    "additional_gpu_probe_executed": False,
                    "additional_probe_executed": False,
                    "systemctl_operations": 0,
                },
            )

    def test_mock_post_start_timeout_is_rc1_with_final_reasons_and_no_probe(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ullm-aq4-post-start-timeout-") as directory:
            base = Path(directory)
            env = self._mock_post_start_env(base, MOCK_POST_START_FORCE_TIMEOUT="1")
            result = subprocess.run(
                ["bash", str(GATE)], env=env, text=True, capture_output=True, check=False
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("status=deadline_timeout", result.stderr)
            self.assertIn("gpu_probe=0", result.stderr)
            self.assertIn("systemctl_operations=0", result.stderr)
            artifact_path = base / "attempts/attempt1/post-start-readiness.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["status"], "deadline_timeout")
            self.assertGreaterEqual(artifact["attempt_count"], 1)
            self.assertIn("owner", artifact["last_failure"])
            self.assertIn("health_200", artifact["last_failure"])
            self.assertFalse(artifact["safety"]["additional_gpu_probe_executed"])
            self.assertFalse(artifact["safety"]["additional_probe_executed"])
            self.assertEqual(artifact["safety"]["systemctl_operations"], 0)
            self.assertFalse((base / "attempts/attempt1/output").exists())

    def test_restore_attempts_service_start_after_cleanup_failure(self) -> None:
        text = GATE.read_text(encoding="utf-8")
        restore = text[text.index("restore_after_failure()"):text.index("execute_probe()", text.index("restore_after_failure()"))]
        self.assertIn("cleanup_rc=$?", restore)
        self.assertIn("start_rc=$?", restore)
        self.assertIn('"${SYSTEMCTL[@]}" start "$SERVICE"', restore)
        self.assertLess(restore.index("cleanup_lock_substrate"), restore.index('"${SYSTEMCTL[@]}" start "$SERVICE"'))
        self.assertIn('[[ "$observer_rc" = 0 && "$cleanup_rc" = 0 && "$start_rc" = 0 ]] || code=1', restore)

    @staticmethod
    def _validator_script() -> str:
        text = GATE.read_text(encoding="utf-8")
        start = text.index("validate_output_contract()")
        start = text.index("<<'PY'\n", start) + len("<<'PY'\n")
        end = text.index("\nPY\n}", start)
        return text[start:end]

    @staticmethod
    def _write_valid_output(root: Path) -> tuple[str, str, str]:
        input_sha = "a" * 64
        package_sha = "b" * 64
        input_path = root / "runtime-input.jsonl"
        input_path.write_text("{}\n", encoding="utf-8")
        shapes = {"qkv": 8192, "qkv_standalone": 8192, "z": 4096, "gate": 32, "beta": 32}
        outputs: dict[str, dict[str, object]] = {}
        for key, shape in shapes.items():
            name = "qkv-standalone.f32le" if key == "qkv_standalone" else f"{key}.f32le"
            raw = struct.pack("<f", 1.0) * shape
            (root / name).write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            outputs[key] = {
                "path": f"/immutable/{name}",
                "row_shape": [shape],
                "bytes": len(raw),
                "sha256": digest,
                "cases": [{
                    "output_offset_bytes": 0,
                    "output_elements": shape,
                    "output_sha256": digest,
                    "finite": True,
                }],
            }
        report = {
            "schema_version": "ullm.aq4_layer0_qkv_z_gate_beta_runtime_probe.v2",
            "status": "valid", "classification": "unclassified", "promotion_eligible": False, "fused": True,
            "operation": "aq4_matvec_qkv_z_gate_beta_f32",
            "device": {"backend": "hip", "device_index": 1, "device_id": 0, "gcn_arch_name": "gfx1201"},
            "visibility": {"hip_visible_devices": "1", "ullm_hip_visible_devices": "1"},
            "guard": {
                "hip_aq4_matvec_kernel_required": True, "fused_kernel_required": True, "fallback_allowed": False,
                "fused_rpb_effective": 4,
                "relevant_environment": {
                    "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL": "1",
                    "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL": "1",
                    "ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB": "4",
                    "ULLM_AQ4_MATVEC_RPB": None,
                },
            },
            "input": {
                "rows": 1, "sidecar_sha256": input_sha,
                "identity": {"pre_stat": "1", "post_stat": "1", "consumed_sha256": input_sha},
            },
            "package": {"manifest_sha256": package_sha},
            "qkv_component_reference": {
                "reference_backend": "hip", "reference_kind": "diagnostic_standalone_reference",
                "operation": "standalone_aq4_matvec_f32", "standalone_rpb_raw": None,
                "standalone_rpb_effective": 32, "standalone_rpb_source": "architecture_default:gfx1201",
                "standalone_output_key": "qkv_standalone", "max_abs": 0.0, "relative_l2": 0.0,
            },
            "qkv_row_segments": [
                {"name": "Q", "start_row": 0, "end_row_exclusive": 2048},
                {"name": "K", "start_row": 2048, "end_row_exclusive": 4096},
                {"name": "V", "start_row": 4096, "end_row_exclusive": 8192},
            ],
            "output_layout": {
                "format": "concatenated_little_endian_f32_rows", "dtype": "f32", "row_order": "input_jsonl_order",
                "qkv_shape": [8192], "qkv_standalone_shape": [8192], "z_shape": [4096],
                "gate_shape": [32], "beta_shape": [32],
            },
            "outputs": outputs,
        }
        (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return str(input_path), input_sha, package_sha

    def test_output_validator_rejects_wrong_reference_and_layout(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ullm-aq4-fused-validator-") as directory:
            root = Path(directory)
            input_path, input_sha, package_sha = self._write_valid_output(root)
            validator = root / "validate.py"
            validator.write_text(self._validator_script(), encoding="utf-8")

            def run_validator() -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["/usr/bin/python3", str(validator), directory, input_path, input_sha, package_sha],
                    text=True, capture_output=True, check=False,
                )

            valid = run_validator()
            self.assertEqual(valid.returncode, 0, valid.stderr)
            report_path = root / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["qkv_component_reference"]["reference_backend"] = "cpu"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertNotEqual(run_validator().returncode, 0)
            report["qkv_component_reference"]["reference_backend"] = "hip"
            report["output_layout"]["qkv_shape"] = [1]
            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertNotEqual(run_validator().returncode, 0)

    def test_preflight_modes_are_mutually_exclusive(self) -> None:
        env = os.environ.copy()
        env.update(MOCK_PREFLIGHT="1", PREFLIGHT_ONLY="1", PREFLIGHT_LOCKED_ONLY="1")
        result = subprocess.run(["bash", str(GATE)], env=env, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 64)
        self.assertIn("mutually exclusive", result.stderr)


if __name__ == "__main__":
    unittest.main()
