from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
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
            "promotion_eligible=false",
        ):
            self.assertIn(token, text)

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

    def test_preflight_modes_are_mutually_exclusive(self) -> None:
        env = os.environ.copy()
        env.update(MOCK_PREFLIGHT="1", PREFLIGHT_ONLY="1", PREFLIGHT_LOCKED_ONLY="1")
        result = subprocess.run(["bash", str(GATE)], env=env, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 64)
        self.assertIn("mutually exclusive", result.stderr)


if __name__ == "__main__":
    unittest.main()
