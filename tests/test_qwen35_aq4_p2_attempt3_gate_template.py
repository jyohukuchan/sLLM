from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/differential-trace-gpu-v1-input"
ATTEMPT2 = INPUT / "run-gpu-gate-attempt2.sh"
ATTEMPT3 = INPUT / "run-gpu-gate-attempt3.sh"


def shell_verifier(output: Path) -> subprocess.CompletedProcess[str]:
    command = '(cd "$OUTPUT" && sha256sum -c SHA256SUMS)'
    return subprocess.run(
        ["bash", "-c", f'OUTPUT="$1"; {command}', "attempt3-test", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class Qwen35Aq4P2Attempt3GateTemplateTests(unittest.TestCase):
    def test_attempt3_diff_only_renames_paths_and_fixes_verifier_cwd(self) -> None:
        attempt2 = ATTEMPT2.read_text(encoding="utf-8")
        expected = attempt2.replace("attempt2", "attempt3").replace(
            'sha256sum -c "$OUTPUT/SHA256SUMS"',
            '(cd "$OUTPUT" && sha256sum -c SHA256SUMS)',
        )
        self.assertEqual(ATTEMPT3.read_text(encoding="utf-8"), expected)

    def test_attempt3_has_exact_terminal_verifier(self) -> None:
        text = ATTEMPT3.read_text(encoding="utf-8")
        self.assertEqual(text.count('(cd "$OUTPUT" && sha256sum -c SHA256SUMS)'), 1)
        self.assertNotIn('sha256sum -c "$OUTPUT/SHA256SUMS"', text)
        self.assertEqual(text.splitlines()[-3], '(cd "$OUTPUT" && sha256sum -c SHA256SUMS)')

    def test_cwd_verifier_accepts_three_files_and_rejects_payload_tamper(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ullm-aq4-attempt3-test-") as directory:
            output = Path(directory)
            for name, content in (
                ("manifest.json", b'{"rows":3}\n'),
                ("payload.jsonl", b'{"case_id":"fixture-prompt-0"}\n'),
                ("runtime.json", b'{"rows":3}\n'),
            ):
                (output / name).write_bytes(content)
            sums = "".join(
                f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n"
                for name in ("manifest.json", "payload.jsonl", "runtime.json")
            )
            (output / "SHA256SUMS").write_text(sums, encoding="ascii")

            verified = shell_verifier(output)
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertIn("manifest.json: OK", verified.stdout)
            self.assertIn("payload.jsonl: OK", verified.stdout)
            self.assertIn("runtime.json: OK", verified.stdout)

            (output / "payload.jsonl").write_bytes(b'{"case_id":"tampered"}\n')
            tampered = shell_verifier(output)
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("payload.jsonl: FAILED", tampered.stdout + tampered.stderr)

    def test_attempt3_paths_are_disjoint_from_attempt1_and_attempt2(self) -> None:
        text = ATTEMPT3.read_text(encoding="utf-8")
        self.assertNotIn("attempt2", text)

        expected_attempt3 = {
            "OUTPUT": "$BASE/differential-trace-gpu-v1-attempt3",
            "GATE_LOG": "$BASE/differential-trace-gpu-v1-attempt3-gate.log",
            "MONITOR_LOG": "$BASE/differential-trace-gpu-v1-attempt3-monitor.log",
            "RUN_LOG": "$BASE/differential-trace-gpu-v1-attempt3-run.log",
            "STOP_MARKER": "$BASE/differential-trace-gpu-v1-attempt3-service-stopped.marker",
            "OBSERVER_FAIL_MARKER": "$BASE/differential-trace-gpu-v1-attempt3-observer-failed.marker",
            "OBSERVER_SAMPLE_MARKER": "$BASE/differential-trace-gpu-v1-attempt3-observer-sample.marker",
            "RUN_STARTED_MARKER": "$BASE/differential-trace-gpu-v1-attempt3-run-started.marker",
            "CANDIDATE_BIN": "$INPUT_DIR/ullm-aq4-differential-trace-detached-attempt3",
        }
        for variable, value in expected_attempt3.items():
            self.assertIn(f'{variable}="{value}"', text.splitlines(), msg=variable)

        attempt3_paths = set(expected_attempt3.values())
        attempt1_paths = {value.replace("-attempt3", "") for value in attempt3_paths}
        attempt2_paths = {value.replace("-attempt3", "-attempt2") for value in attempt3_paths}
        self.assertTrue(attempt3_paths.isdisjoint(attempt1_paths))
        self.assertTrue(attempt3_paths.isdisjoint(attempt2_paths))


if __name__ == "__main__":
    unittest.main()
