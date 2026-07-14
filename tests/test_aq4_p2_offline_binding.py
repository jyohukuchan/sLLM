from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/generate-aq4-p2-offline-binding.py"
RUN_ROOT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1"
P0 = RUN_ROOT / "p0/p0-snapshot.json"
P1_RECORD = RUN_ROOT / "p1/production-executor-record-live-v4.json"
P1_TRACE = RUN_ROOT / "p1/production-trace-live-v4.json"
SOURCE = RUN_ROOT / "p2/source-oracle-v2/manifest.json"
ACTIVE = Path("/etc/ullm/served-models/active.json")


class Aq4P2OfflineBindingTests(unittest.TestCase):
    def run_tool(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", str(TOOL), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def generate(self, directory: Path, **overrides: Path) -> None:
        inputs = {
            "--active-manifest": ACTIVE,
            "--p0-snapshot": P0,
            "--p1-executor-record": overrides.get("record", P1_RECORD),
            "--p1-trace": overrides.get("trace", P1_TRACE),
            "--source-oracle": SOURCE,
        }
        command = ["--output-dir", str(directory)]
        for flag, path in inputs.items():
            command.extend((flag, str(path)))
        completed = self.run_tool(*command)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_generation_extracts_lossless_graph_state_and_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "p2"
            self.generate(output)
            binding = json.loads((output / "binding-inputs.json").read_text())
            model = json.loads((output / "model_identity.json").read_text())
            record = json.loads(P1_RECORD.read_text())
            self.assertEqual(set(model), {"id", "revision", "format_id", "implementation_id"})
            self.assertNotIn("upstream_id", model)
            self.assertNotIn("upstream_revision", model)
            self.assertEqual(model, binding["model_identity"])
            self.assertEqual(json.loads((output / "graph.json").read_text()), record["graph"]["model_graph"]["canonical"])
            self.assertEqual(json.loads((output / "state.json").read_text()), record["graph"]["state_schema"]["canonical"])
            self.assertFalse(binding["promotion_eligible"])
            self.assertEqual(binding["path_oracle"]["status"], "not_run")
            self.assertFalse(binding["path_oracle"]["source_oracle_substitute"])
            self.assertFalse(binding["path_oracle"]["p1_trace_substitute"])
            self.assertFalse(binding["path_oracle"]["fixture_substitute"])
            self.assertEqual(json.loads((output / "correctness-threshold-audit.json").read_text())["decision"], "BLOCKED")
            checked = self.run_tool("--validate", "--output-dir", str(output))
            self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_validator_rejects_unknown_binding_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "p2"
            self.generate(output)
            binding = json.loads((output / "binding-inputs.json").read_text())
            binding["unexpected"] = True
            (output / "binding-inputs.json").write_text(json.dumps(binding) + "\n")
            checked = self.run_tool("--validate", "--output-dir", str(output))
            self.assertNotEqual(checked.returncode, 0)

    def test_validator_rejects_duplicate_binding_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "p2"
            self.generate(output)
            raw = (output / "binding-inputs.json").read_text()
            raw = raw.replace('"status": "blocked"', '"status": "blocked",\n  "status": "blocked"', 1)
            (output / "binding-inputs.json").write_text(raw)
            checked = self.run_tool("--validate", "--output-dir", str(output))
            self.assertNotEqual(checked.returncode, 0)

    def test_validator_rejects_graph_value_tamper_even_before_rebinding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "p2"
            self.generate(output)
            graph = json.loads((output / "graph.json").read_text())
            graph["context_length"] += 1
            (output / "graph.json").write_text(json.dumps(graph) + "\n")
            checked = self.run_tool("--validate", "--output-dir", str(output))
            self.assertNotEqual(checked.returncode, 0)

    def test_validator_rejects_path_oracle_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "p2"
            self.generate(output)
            binding = json.loads((output / "binding-inputs.json").read_text())
            binding["path_oracle"]["source_oracle_substitute"] = True
            (output / "binding-inputs.json").write_text(json.dumps(binding) + "\n")
            checked = self.run_tool("--validate", "--output-dir", str(output))
            self.assertNotEqual(checked.returncode, 0)

    def test_generator_rejects_duplicate_unknown_and_digest_mismatch_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duplicate = root / "duplicate-record.json"
            raw = P1_RECORD.read_text()
            duplicate.write_text(raw.rsplit("}", 1)[0] + ', "status": "ok"}\n')
            output = root / "duplicate-output"
            completed = self.run_tool(
                "--output-dir", str(output), "--p0-snapshot", str(P0), "--p1-executor-record", str(duplicate),
                "--p1-trace", str(P1_TRACE), "--source-oracle", str(SOURCE),
            )
            self.assertNotEqual(completed.returncode, 0)
            unknown = root / "unknown-record.json"
            record = json.loads(P1_RECORD.read_text())
            record["graph"]["unknown"] = True
            unknown.write_text(json.dumps(record) + "\n")
            output = root / "unknown-output"
            completed = self.run_tool(
                "--output-dir", str(output), "--p0-snapshot", str(P0), "--p1-executor-record", str(unknown),
                "--p1-trace", str(P1_TRACE), "--source-oracle", str(SOURCE),
            )
            self.assertNotEqual(completed.returncode, 0)
            mismatch = root / "mismatch-trace.json"
            trace = json.loads(P1_TRACE.read_text())
            trace["graph"]["model_graph"]["sha256"] = "0" * 64
            mismatch.write_text(json.dumps(trace) + "\n")
            output = root / "mismatch-output"
            completed = self.run_tool(
                "--output-dir", str(output), "--p0-snapshot", str(P0), "--p1-executor-record", str(P1_RECORD),
                "--p1-trace", str(mismatch), "--source-oracle", str(SOURCE),
            )
            self.assertNotEqual(completed.returncode, 0)

    def test_validator_rejects_hash_manifest_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "p2"
            self.generate(output)
            sums = output / "SHA256SUMS"
            lines = sums.read_text().splitlines()
            lines[0] = "0" * 64 + lines[0][64:]
            sums.write_text("\n".join(lines) + "\n")
            checked = self.run_tool("--validate", "--output-dir", str(output))
            self.assertNotEqual(checked.returncode, 0)


if __name__ == "__main__":
    unittest.main()
