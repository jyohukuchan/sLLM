from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/production-execution-trace-p1/schema-r1"
PRODUCER = ROOT / "tools/produce-production-execution-trace.py"
VALIDATOR = ROOT / "tools/validate-production-execution-trace.py"


class ProductionExecutionTraceTests(unittest.TestCase):
    def test_cpu_fixture_has_detached_valid_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "schema-r1"
            shutil.copytree(FIXTURE, work)
            for name in ("trace.json", "binding.json", "report.json", "verified.json", "verified-binding.json"):
                (work / name).unlink(missing_ok=True)
            produced = subprocess.run(
                ["python3", str(PRODUCER), "--manifest", str(work / "manifest.json"), "--executor-record", str(work / "executor-record.json"), "--output", str(work / "trace.json"), "--binding-output", str(work / "binding.json")],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(produced.returncode, 0, produced.stderr)
            report = work / "report.json"
            verified = work / "verified.json"
            verified_binding = work / "verified-binding.json"
            completed = subprocess.run(
                [
                    "python3",
                    str(VALIDATOR),
                    "--trace",
                    str(work / "trace.json"),
                    "--manifest",
                    str(work / "manifest.json"),
                    "--executor-record",
                    str(work / "executor-record.json"),
                    "--binding",
                    str(work / "binding.json"),
                    "--output",
                    str(report),
                    "--verified-trace",
                    str(verified),
                    "--verified-binding",
                    str(verified_binding),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                json.loads(verified.read_text())["verification"]["independent_validation"]["report_sha256"],
                hashlib.sha256(report.read_bytes()).hexdigest(),
            )

    def test_validator_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"schema_version": "x", "schema_version": "y"}\n')
            completed = subprocess.run(
                ["python3", str(VALIDATOR), "--trace", str(path), "--manifest", str(path), "--executor-record", str(path), "--binding", str(path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)

    def test_validator_rejects_boolean_oom_and_direct_server_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "schema-r1"
            shutil.copytree(FIXTURE, work)
            trace = json.loads((work / "trace.json").read_text())
            trace["memory"]["oom"] = False
            (work / "trace.json").write_text(json.dumps(trace))
            completed = subprocess.run(
                ["python3", str(VALIDATOR), "--trace", str(work / "trace.json"), "--manifest", str(work / "manifest.json"), "--executor-record", str(work / "executor-record.json"), "--binding", str(work / "binding.json")],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
