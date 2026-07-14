from __future__ import annotations

import hashlib
import importlib.util
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
    def test_capture_graph_uses_normative_sources_and_sampling(self) -> None:
        spec = importlib.util.spec_from_file_location("capture_aq4", ROOT / "tools/capture-aq4-resident-executor-record.py")
        self.assertIsNotNone(spec and spec.loader)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        graph = module.layer_graph({"passthrough_tensors": [{"name": "model.language_model.embed_tokens.weight", "shape": [8, 4]}], "tensors": []}, {"public": {"id": "fixture", "context_length": 16}, "format": {"format_id": "AQ4_0"}})
        self.assertEqual(graph["model_graph"]["source"], "adapter_derived")
        self.assertEqual(graph["state_schema"]["source"], "adapter_derived")
        self.assertIn("sampling", graph["model_graph"]["canonical"]["terminal_components"])

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

    def test_verified_trace_rejects_report_attesting_final_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "schema-r1"
            shutil.copytree(FIXTURE, work)
            report = json.loads((work / "report.json").read_text())
            report["trace_sha256"] = hashlib.sha256((work / "verified.json").read_bytes()).hexdigest()
            (work / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            completed = subprocess.run(
                ["python3", str(VALIDATOR), "--trace", str(work / "verified.json"), "--manifest", str(work / "manifest.json"), "--executor-record", str(work / "executor-record.json"), "--binding", str(work / "verified-binding.json"), "--report", str(work / "report.json")],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)

    def test_validator_rejects_phase_context_tamper_after_rebinding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "schema-r1"
            shutil.copytree(FIXTURE, work)
            trace = json.loads((work / "trace.json").read_text())
            trace["phases"][1]["context_tokens_after"] += 7
            trace_raw = (json.dumps(trace, indent=2, sort_keys=True) + "\n").encode()
            (work / "trace.json").write_bytes(trace_raw)
            binding = json.loads((work / "binding.json").read_text())
            binding["trace_sha256"] = hashlib.sha256(trace_raw).hexdigest()
            (work / "binding.json").write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
            completed = subprocess.run(
                ["python3", str(VALIDATOR), "--trace", str(work / "trace.json"), "--manifest", str(work / "manifest.json"), "--executor-record", str(work / "executor-record.json"), "--binding", str(work / "binding.json")],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)

    def test_validator_rejects_unreconstructed_129_internal_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "schema-r1"
            shutil.copytree(FIXTURE, work)
            record = json.loads((work / "executor-record.json").read_text())
            record["state_commit"]["prepared_batch_count"] = 129
            record["state_commit"]["committed_batch_count"] = 129
            (work / "executor-record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            for name in ("trace.json", "binding.json"):
                (work / name).unlink()
            produced = subprocess.run(["python3", str(PRODUCER), "--manifest", str(work / "manifest.json"), "--executor-record", str(work / "executor-record.json"), "--output", str(work / "trace.json"), "--binding-output", str(work / "binding.json")], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(produced.returncode, 0, produced.stderr)
            completed = subprocess.run(["python3", str(VALIDATOR), "--trace", str(work / "trace.json"), "--manifest", str(work / "manifest.json"), "--executor-record", str(work / "executor-record.json"), "--binding", str(work / "binding.json")], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
