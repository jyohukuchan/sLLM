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
    def test_gate_is_syntax_checked_and_mock_preflight_is_read_only(self) -> None:
        syntax = subprocess.run(["bash", "-n", str(GATE)], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        with tempfile.TemporaryDirectory(prefix="ullm-aq4-active-gate-") as directory:
            env = os.environ.copy()
            env.update(MOCK_PREFLIGHT="1", PREFLIGHT_ONLY="1")
            completed = subprocess.run([str(GATE)], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
            worker = ROOT / "target/reasoning-v2/release/ullm-aq4-worker"
            if worker.stat().st_nlink == 1:
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("mock_preflight=1 service_stop=0 gpu_run=0", completed.stdout)
                self.assertIn("device_mapping=rocm_smi_card2->hip_visible_token1->filtered_hip_ordinal0->global_logical_device1 expected_architecture=gfx1201", completed.stdout)
                self.assertIn("device_index_boundary=global0_cpu global1_filtered_hip_ordinal0", completed.stdout)
            else:
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("worker identity differs", completed.stderr)
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
            'INPUT_DIR="$REPO/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-calibration-capture-v0.1/input-v32"',
            'CASES="$INPUT_DIR/cases.json"',
            'SOURCE_ARTIFACT="$REPO/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-calibration-capture-v0.1/attempts/source-attempt-v32-20260714T180609Z/source-full"',
            'EXPECTED_SOURCE_ARTIFACT_SHA256="6d27caef27dabf02dcc56b0b298290f9811355ba36c34e6c9d23939baf50edde"',
            'EXPECTED_SOURCE_MANIFEST_SHA256="78a6de7d2cae4c2ff31952cfe345fefbce55dfd67db7a4904ba10f4e5f7438bc"',
            'CAPTURE_BINARY_ROOT="$BASE/input/capture-binary-v0.1"',
            'EXPECTED_CAPTURE_BINARY_SHA256="82c878a4974cdbc442458c6b3366b0eae20d355896d8b18d5d76fe311c0b083e"',
            'EXPECTED_BUILD_RECEIPT_SHA256="3d09df92aa2bef098c8c64ef7bcd63ed0b23dd2160a44dfa3799421477440ede"',
            'EXPECTED_BUILD_COMMIT="05a8ab661b8e56559353f5a530ec8abac08b9a68"',
            'EXPECTED_BUILD_TREE_SHA256="12e6d777f37d648ede369263296cd5606676a441"',
            'EXPECTED_CARGO_LOCK_SHA256="10df8371ae3a33ed792dc4e8c15dd6196a8a7e176e377ef275e75b3219aa157b"',
            'EXPECTED_ROCM_SMI_PHYSICAL_CARD="2"',
            'EXPECTED_HIP_VISIBLE_TOKEN="1"',
            'EXPECTED_FILTERED_HIP_ORDINAL=0',
            'EXPECTED_CPU_GLOBAL_DEVICE_INDEX=0',
            'EXPECTED_LOGICAL_DEVICE_INDEX=1',
            '[[ "$EXPECTED_ROCM_SMI_PHYSICAL_CARD" = "2" && "$EXPECTED_HIP_VISIBLE_TOKEN" = "1" && "$EXPECTED_FILTERED_HIP_ORDINAL" = 0 && "$EXPECTED_CPU_GLOBAL_DEVICE_INDEX" = 0 && "$EXPECTED_LOGICAL_DEVICE_INDEX" = 1 && "$DEVICE_ARCHITECTURE" = "gfx1201" ]] || fail "device mapping differs"',
            'PACKAGE_MANIFEST="$PACKAGE_ROOT/package/manifest.json"',
            'SYSTEMCTL=(sudo -n -- systemctl)',
            'RUNTIME_DIR_INSTALL=(sudo -n -- install -d -o homelab1 -g homelab1 -m 0750)',
            'RUNTIME_DIR_REMOVE=(sudo -n -- rmdir)',
            '"${RUNTIME_DIR_INSTALL[@]}" "$RUNTIME_DIR" || fail "runtime directory create failed"',
            '"${RUNTIME_DIR_REMOVE[@]}" "$RUNTIME_DIR"',
            'EXPECTED_PLAN_SHA256="1b4f8c244e922ab73c0bb026216d8333a9cfe57c23e6695c4141554d117693c0"',
            'EXPECTED_CASES_SHA256="53f256bc8f5ed4036cfb1a9a98c0c9d9197bb980e1ef91d7ff01cf73001369a8"',
            'EXPECTED_SPLIT_SHA256="966878f3d9eb13f5b485825208f8072521724f308f5ee3d8a003b0b051198887"',
            'EXPECTED_POLICY_SHA256="302c3219af286a970ddf39ed090021ef102b51b2d188c0ff337f6b9dd04d1a03"',
            'EXPECTED_CALIBRATION_SHA256="20c09f22bb1ca4dfac907de09febddb01ed0228c3f4a17c01efd646491e0983f"',
            'assert plan["execution_contract"]["source_torch_threads"] == 32',
            'assert receipt["schema_version"] == "ullm.aq4_fidelity_capture_build_receipt.v1"',
            '[[ "$(stat -Lc \'%F:%h\' "$FIDELITY_BIN")" = "regular file:1" ]]',
            '[[ "$(stat -Lc \'%F:%h\' "$WORKER")" = "regular file:1" ]] || fail "worker identity differs"',
            '--served-model-manifest "$ACTIVE"',
            '--split-root "$SPLIT_ROOT"',
            '--source "$SOURCE_ARTIFACT"',
            '--cases "$CASES"',
            '--output "$OUTPUT"',
            '--expected-guard-sha256 "$GUARD_SHA256"',
            '"ULLM_HIP_VISIBLE_DEVICES=$EXPECTED_HIP_VISIBLE_TOKEN"',
            '"HIP_VISIBLE_DEVICES=$EXPECTED_HIP_VISIBLE_TOKEN"',
            '--device-index "$EXPECTED_LOGICAL_DEVICE_INDEX"',
        ):
            self.assertIn(needle, text, needle)
        self.assertNotIn("CANDIDATE_BIN", text)
        self.assertNotIn('chown homelab1:homelab1 "$RUNTIME_DIR"', text)
        self.assertNotIn('chmod 750 "$RUNTIME_DIR"', text)
        self.assertNotIn('chown homelab1:homelab1 "$LOCK"', text)
        self.assertNotIn('chmod 600 "$LOCK"', text)
        self.assertNotIn("runuser -u homelab1", text)
        self.assertIn('env "${guard_env[@]}" \\\n  timeout --signal=TERM --kill-after=30s 1200s "$FIDELITY_BIN"', text)
        self.assertIn('device_mapping=rocm_smi_card${EXPECTED_ROCM_SMI_PHYSICAL_CARD}->hip_visible_token${EXPECTED_HIP_VISIBLE_TOKEN}->filtered_hip_ordinal${EXPECTED_FILTERED_HIP_ORDINAL}->global_logical_device${EXPECTED_LOGICAL_DEVICE_INDEX}', text)
        self.assertIn('device_index_boundary=global${EXPECTED_CPU_GLOBAL_DEVICE_INDEX}_cpu global${EXPECTED_LOGICAL_DEVICE_INDEX}_filtered_hip_ordinal${EXPECTED_FILTERED_HIP_ORDINAL}', text)
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
