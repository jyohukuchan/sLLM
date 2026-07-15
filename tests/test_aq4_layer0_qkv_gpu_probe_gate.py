from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / (
    "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/"
    "p2/aq4-layer0-qkv-gpu-probe-v0.1/input/run-gpu-probe-gate.sh"
)
PROBE_ROOT = ROOT / (
    "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/"
    "p2/aq4-layer0-qkv-gpu-probe-v0.1/input/probe-binary-v0.1"
)


class Aq4Layer0QkvGpuProbeGateTest(unittest.TestCase):
    def test_shell_syntax_and_fixed_contract(self) -> None:
        subprocess.run(["bash", "-n", str(GATE)], check=True)
        text = GATE.read_text(encoding="utf-8")
        for token in (
            "EXPECTED_INPUT_SHA256=\"c009a9bded30b1b9a7c704c622bd3106b3d17989c438f91eb20bb16817348e17\"",
            "EXPECTED_PACKAGE_SHA256=\"a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad\"",
            "EXPECTED_BUILD_COMMIT=\"2bcef0d897d43ea1ff397dc558f7e0e179d8a904\"",
            "EXPECTED_LOGICAL_DEVICE_INDEX=\"1\"",
            "EXPECTED_FILTERED_HIP_ORDINAL=\"0\"",
            "EXPECTED_DEVICE_ARCHITECTURE=\"gfx1201\"",
            "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1",
            '"fused"] is False',
            '"promotion_eligible"] is False',
            '"classification"] == "unclassified"',
            "RUNTIME_DIR_INSTALL=(sudo -n -- install",
            "RUNTIME_DIR_REMOVE=(sudo -n -- rmdir)",
            "(trap - EXIT INT TERM; run_observer) & observer_pid=$!",
        ):
            self.assertIn(token, text)

    def test_mock_preflight_is_read_only_and_never_invokes_runtime(self) -> None:
        self.assertTrue(PROBE_ROOT.is_dir(), "release artifact must be built before this test")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            env = os.environ.copy()
            env.update(
                {
                    "MOCK_PREFLIGHT": "1",
                    "PREFLIGHT_ONLY": "1",
                    "MOCK_BASE": str(base),
                    "MOCK_PROBE_ROOT": str(PROBE_ROOT),
                    "MOCK_PROBE": str(PROBE_ROOT / "ullm-aq4-layer0-qkv-runtime-probe"),
                    "MOCK_RECEIPT": str(PROBE_ROOT / "build-receipt.json"),
                    "MOCK_SUMS": str(PROBE_ROOT / "SHA256SUMS"),
                    "PYTHON": "/usr/bin/python3",
                }
            )
            result = subprocess.run(
                ["bash", str(GATE)], env=env, text=True, capture_output=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("mock_preflight=1 service_stop=0 gpu_run=0", result.stdout)
            self.assertFalse(any(base.iterdir()))

    def test_preflight_modes_are_mutually_exclusive(self) -> None:
        env = os.environ.copy()
        env.update({"MOCK_PREFLIGHT": "1", "PREFLIGHT_ONLY": "1", "PREFLIGHT_LOCKED_ONLY": "1"})
        result = subprocess.run(
            ["bash", str(GATE)], env=env, text=True, capture_output=True, check=False
        )
        self.assertEqual(result.returncode, 64)
        self.assertIn("mutually exclusive", result.stderr)

    def test_tampered_input_is_rejected_before_execution(self) -> None:
        source = ROOT / (
            "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/"
            "p2/aq4-layer0-matvec-oracle-integration-v0.1/runtime-input.jsonl"
        )
        with tempfile.TemporaryDirectory() as temporary:
            tampered = Path(temporary) / "runtime-input.jsonl"
            tampered.write_bytes(source.read_bytes() + b"\n")
            os.chmod(tampered, 0o444)
            env = os.environ.copy()
            env.update(
                {
                    "MOCK_PREFLIGHT": "1",
                    "PREFLIGHT_ONLY": "1",
                    "MOCK_INPUT": str(tampered),
                    "MOCK_PROBE_ROOT": str(PROBE_ROOT),
                    "MOCK_PROBE": str(PROBE_ROOT / "ullm-aq4-layer0-qkv-runtime-probe"),
                    "MOCK_RECEIPT": str(PROBE_ROOT / "build-receipt.json"),
                    "MOCK_SUMS": str(PROBE_ROOT / "SHA256SUMS"),
                    "PYTHON": "/usr/bin/python3",
                }
            )
            result = subprocess.run(
                ["bash", str(GATE)], env=env, text=True, capture_output=True, check=False
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("input sidecar SHA differs", result.stderr)


if __name__ == "__main__":
    unittest.main()
