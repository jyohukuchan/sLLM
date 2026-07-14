from __future__ import annotations

import json
import hashlib
import importlib.util
import os
import shutil
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
SPEC = importlib.util.spec_from_file_location("aq4_p2_offline_binding", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


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

    @staticmethod
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def rehash_bundle(self, output: Path) -> None:
        binding_path = output / "binding-inputs.json"
        binding = json.loads(binding_path.read_text())
        binding["artifacts"]["validation_report"]["sha256"] = self.sha(output / "validation-report.json")
        binding_path.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
        manifest_path = output / "hash-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for artifact in manifest["artifacts"]:
            path = output / artifact["path"]
            artifact["bytes"] = path.stat().st_size
            artifact["sha256"] = self.sha(path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        sums = "".join(
            f"{self.sha(output / name)}  {name}\n"
            for name in sorted(MODULE.EXPECTED_BUNDLE_FILES - {"SHA256SUMS"})
        )
        (output / "SHA256SUMS").write_text(sums)

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

    def test_validator_rejects_validation_report_unknown_field_after_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "p2"
            self.generate(output)
            report = json.loads((output / "validation-report.json").read_text())
            report["unexpected"] = True
            (output / "validation-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            self.rehash_bundle(output)
            checked = self.run_tool("--validate", "--output-dir", str(output))
            self.assertNotEqual(checked.returncode, 0)

    def test_validator_rejects_validation_report_semantic_tamper_after_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "p2"
            self.generate(output)
            report = json.loads((output / "validation-report.json").read_text())
            report["path_oracle_status"] = "passed"
            (output / "validation-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            self.rehash_bundle(output)
            checked = self.run_tool("--validate", "--output-dir", str(output))
            self.assertNotEqual(checked.returncode, 0)

    def test_validator_rejects_validation_report_link_type_after_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "p2"
            self.generate(output)
            binding = json.loads((output / "binding-inputs.json").read_text())
            binding["artifacts"]["validation_report"]["type"] = "model_graph"
            (output / "binding-inputs.json").write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
            # Update only the outer manifests so this reaches link semantics.
            manifest = json.loads((output / "hash-manifest.json").read_text())
            for artifact in manifest["artifacts"]:
                if artifact["path"] == "binding-inputs.json":
                    artifact["bytes"] = (output / "binding-inputs.json").stat().st_size
                    artifact["sha256"] = self.sha(output / "binding-inputs.json")
            (output / "hash-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            sums = "".join(
                f"{self.sha(output / name)}  {name}\n"
                for name in sorted(MODULE.EXPECTED_BUNDLE_FILES - {"SHA256SUMS"})
            )
            (output / "SHA256SUMS").write_text(sums)
            checked = self.run_tool("--validate", "--output-dir", str(output))
            self.assertNotEqual(checked.returncode, 0)

    def test_validator_rejects_unexpected_and_missing_bundle_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unexpected = root / "unexpected"
            self.generate(unexpected)
            (unexpected / "extra.json").write_text("{}\n")
            checked = self.run_tool("--validate", "--output-dir", str(unexpected))
            self.assertNotEqual(checked.returncode, 0)
            missing = root / "missing"
            self.generate(missing)
            (missing / "state.json").unlink()
            checked = self.run_tool("--validate", "--output-dir", str(missing))
            self.assertNotEqual(checked.returncode, 0)

    def test_validator_rejects_sha256sums_symlink_and_output_directory_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "p2"
            self.generate(output)
            external = root / "sums.txt"
            shutil.copy2(output / "SHA256SUMS", external)
            (output / "SHA256SUMS").unlink()
            (output / "SHA256SUMS").symlink_to(external)
            checked = self.run_tool("--validate", "--output-dir", str(output))
            self.assertNotEqual(checked.returncode, 0)
            real_output = root / "real-p2"
            self.generate(real_output)
            alias = root / "p2-alias"
            alias.symlink_to(real_output, target_is_directory=True)
            checked = self.run_tool("--validate", "--output-dir", str(alias))
            self.assertNotEqual(checked.returncode, 0)

    def test_validator_rejects_sha256sums_external_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "p2"
            self.generate(output)
            os.link(output / "SHA256SUMS", root / "external-sums.txt")
            checked = self.run_tool("--validate", "--output-dir", str(output))
            self.assertNotEqual(checked.returncode, 0)

    def test_validator_rejects_artifact_json_external_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "p2"
            self.generate(output)
            os.link(output / "graph.json", root / "external-graph.json")
            checked = self.run_tool("--validate", "--output-dir", str(output))
            self.assertNotEqual(checked.returncode, 0)

    def test_generator_rejects_hardlinked_active_and_p1_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active.json"
            shutil.copy2(ACTIVE, active)
            os.link(active, root / "active-hardlink.json")
            completed = self.run_tool(
                "--output-dir", str(root / "active-output"), "--active-manifest", str(active),
                "--p0-snapshot", str(P0), "--p1-executor-record", str(P1_RECORD),
                "--p1-trace", str(P1_TRACE), "--source-oracle", str(SOURCE),
            )
            self.assertNotEqual(completed.returncode, 0)
            record = root / "record.json"
            shutil.copy2(P1_RECORD, record)
            os.link(record, root / "record-hardlink.json")
            completed = self.run_tool(
                "--output-dir", str(root / "p1-output"), "--p0-snapshot", str(P0),
                "--p1-executor-record", str(record), "--p1-trace", str(P1_TRACE),
                "--source-oracle", str(SOURCE),
            )
            self.assertNotEqual(completed.returncode, 0)

    def test_generator_rejects_input_parent_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_parent = root / "real"
            real_parent.mkdir()
            record = real_parent / "record.json"
            shutil.copy2(P1_RECORD, record)
            alias = root / "alias"
            alias.symlink_to(real_parent, target_is_directory=True)
            completed = self.run_tool(
                "--output-dir", str(root / "output"), "--p0-snapshot", str(P0),
                "--p1-executor-record", str(alias / "record.json"), "--p1-trace", str(P1_TRACE),
                "--source-oracle", str(SOURCE),
            )
            self.assertNotEqual(completed.returncode, 0)

    def test_stable_read_rejects_deterministic_rename_toctou(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_text('{"value":1}\n')
            moved = Path(tmp) / "moved.json"

            def hook(target: Path, phase: str) -> None:
                if target == path and phase == "after_open":
                    MODULE._READ_TEST_HOOK = None
                    target.rename(moved)
                    target.write_text('{"value":1}\n')

            MODULE._READ_TEST_HOOK = hook
            try:
                with self.assertRaises(MODULE.BindingError):
                    MODULE.load_json(path, "rename fixture")
            finally:
                MODULE._READ_TEST_HOOK = None

    def test_stable_read_rejects_deterministic_append_toctou(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_text('{"value":1}\n')

            def hook(target: Path, phase: str) -> None:
                if target == path and phase == "after_read":
                    MODULE._READ_TEST_HOOK = None
                    with target.open("ab") as stream:
                        stream.write(b" ")

            MODULE._READ_TEST_HOOK = hook
            try:
                with self.assertRaises(MODULE.BindingError):
                    MODULE.load_json(path, "append fixture")
            finally:
                MODULE._READ_TEST_HOOK = None

    def test_stable_read_rejects_deterministic_same_size_rewrite_toctou(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            original = b'{"value":1}\n'
            path.write_bytes(original)

            def hook(target: Path, phase: str) -> None:
                if target == path and phase == "after_read":
                    MODULE._READ_TEST_HOOK = None
                    descriptor = os.open(target, os.O_WRONLY)
                    try:
                        os.pwrite(descriptor, b'{"value":2}\n', 0)
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)

            MODULE._READ_TEST_HOOK = hook
            try:
                with self.assertRaises(MODULE.BindingError):
                    MODULE.load_json(path, "rewrite fixture")
            finally:
                MODULE._READ_TEST_HOOK = None


if __name__ == "__main__":
    unittest.main()
