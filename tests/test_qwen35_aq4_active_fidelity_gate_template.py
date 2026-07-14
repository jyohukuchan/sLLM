from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-calibration-active-v0.1/input/run-active-fidelity-capture-gate.sh"


class Qwen35Aq4ActiveFidelityGateTemplateTests(unittest.TestCase):
    def test_template_is_syntax_checked_and_fails_on_unresolved_source_identity(self) -> None:
        syntax = subprocess.run(["bash", "-n", str(GATE)], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        with tempfile.TemporaryDirectory(prefix="ullm-aq4-active-gate-") as directory:
            env = os.environ.copy()
            env.update(MOCK_PREFLIGHT="1", PREFLIGHT_ONLY="1")
            completed = subprocess.run([str(GATE)], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("unresolved placeholder", completed.stderr)
            self.assertNotIn("systemctl", completed.stdout + completed.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_preflight_modes_are_mutually_exclusive_without_service_action(self) -> None:
        env = os.environ.copy()
        env.update(MOCK_PREFLIGHT="1", PREFLIGHT_ONLY="1", PREFLIGHT_LOCKED_ONLY="1")
        completed = subprocess.run([str(GATE)], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 64)
        self.assertNotIn("systemctl stop", completed.stdout + completed.stderr)

    def test_paths_package_args_and_identity_pins_are_explicit(self) -> None:
        text = GATE.read_text(encoding="utf-8")
        for needle in (
            'OUTPUT="$BASE/output"',
            'METRICS="$BASE/metrics.json"',
            'GATE_LOG="$BASE/gate.log"',
            'MONITOR_LOG="$BASE/monitor.log"',
            'RUN_LOG="$BASE/run.log"',
            'SOURCE_ARTIFACT="$BASE/SOURCE_ARTIFACT_ROOT_PLACEHOLDER"',
            'EXPECTED_SOURCE_ARTIFACT_SHA256="__SOURCE_ARTIFACT_SHA256__"',
            'EXPECTED_SOURCE_MANIFEST_SHA256="__SOURCE_MANIFEST_SHA256__"',
            'EXPECTED_CAPTURE_BINARY_SHA256="__CAPTURE_BINARY_SHA256__"',
            'PACKAGE_MANIFEST="$PACKAGE_ROOT/package/manifest.json"',
            'EXPECTED_PLAN_SHA256="03e921b050d64ae75206ff561b19ba563c1fac69ccf40dfab15176dee2b63854"',
            'EXPECTED_CASES_SHA256="53f256bc8f5ed4036cfb1a9a98c0c9d9197bb980e1ef91d7ff01cf73001369a8"',
            'EXPECTED_SPLIT_SHA256="966878f3d9eb13f5b485825208f8072521724f308f5ee3d8a003b0b051198887"',
            'EXPECTED_POLICY_SHA256="302c3219af286a970ddf39ed090021ef102b51b2d188c0ff337f6b9dd04d1a03"',
            'EXPECTED_CALIBRATION_SHA256="20c09f22bb1ca4dfac907de09febddb01ed0228c3f4a17c01efd646491e0983f"',
            'REQUIRED_BUILD_COMMIT="b1755da2a8ed188e3afac52dc1303ebaec3d09f5"',
            '--served-model-manifest "$ACTIVE"',
            '--split-root "$SPLIT_ROOT"',
            '--source "$SOURCE_ARTIFACT"',
            '--cases "$CASES"',
            '--output "$OUTPUT"',
            '--expected-guard-sha256 "$GUARD_SHA256"',
        ):
            self.assertIn(needle, text, needle)
        self.assertNotIn("CANDIDATE_BIN", text)
        self.assertIn("one_model_load=1 nonfinite_rows=0", text)
        self.assertIn("for guard_name in \"${REQUIRED_GUARDS[@]}\"; do guard_env+=(\"$guard_name=1\"); done", text)
        self.assertIn("tools/capture-qwen35-aq4-fidelity.py", text)
        self.assertIn("tools/validate-qwen35-aq4-fidelity-capture.py", text)

    def test_output_sha_verifier_accepts_valid_files_and_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ullm-aq4-active-output-") as directory:
            output = Path(directory)
            files = {"manifest.json": b'{"row_count":24}\n', "rows.jsonl": b'{"case_id":"row-0"}\n', "vectors.f32le": b"\x00\x01\x02\x03"}
            for name, payload in files.items():
                (output / name).write_bytes(payload)
            sums = "".join(f"{hashlib.sha256(payload).hexdigest()}  {name}\n" for name, payload in files.items())
            (output / "SHA256SUMS").write_text(sums, encoding="ascii")
            command = '(cd "$OUTPUT" && sha256sum -c SHA256SUMS)'
            valid = subprocess.run(["bash", "-c", f'OUTPUT="$1"; {command}', "verify", str(output)], cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            (output / "rows.jsonl").write_bytes(b'{"case_id":"tampered"}\n')
            tampered = subprocess.run(["bash", "-c", f'OUTPUT="$1"; {command}', "verify", str(output)], cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("rows.jsonl: FAILED", tampered.stdout + tampered.stderr)


if __name__ == "__main__":
    unittest.main()
