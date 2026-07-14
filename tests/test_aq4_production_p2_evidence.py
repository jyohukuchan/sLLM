from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/workloads/aq4-production-opt-p2-case-manifest-v0.1.json"
POLICY = ROOT / "benchmarks/workloads/aq4-production-opt-p2-threshold-policy-template-v0.1.json"
EXPAND = ROOT / "tools/expand-aq4-production-p2.py"
BIND = ROOT / "tools/bind-aq4-production-p2-identity.py"
RUN = ROOT / "tools/run-aq4-production-p2.py"
BUILD = ROOT / "tools/build-aq4-prefill-validation-result.py"
VALIDATE = ROOT / "tools/validate-aq4-production-p2-evidence.py"


def invoke(tool: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["python3", str(tool), *args], cwd=ROOT, capture_output=True, text=True)


class Aq4ProductionP2EvidenceTests(unittest.TestCase):
    def test_expansion_matches_normative_case_count_and_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "expanded.json"
            completed = invoke(EXPAND, "--manifest", str(MANIFEST), "--output", str(output))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            expanded = json.loads(output.read_text())
            self.assertEqual(expanded["case_count"], 4864)
            self.assertEqual(expanded["expected_case_count"]["total"], 4864)
            self.assertTrue(all("prompt content" not in json.dumps(case).lower() for case in expanded["cases"]))
            linked = [case for case in expanded["cases"] if case.get("path_oracle_case_id")]
            self.assertTrue(linked)
            self.assertTrue(all(case["path_oracle_case_id"].startswith("p2-") for case in linked))

    def _prepare(self, root: Path) -> tuple[Path, Path, dict]:
        expanded = root / "expanded.json"
        self.assertEqual(invoke(EXPAND, "--manifest", str(MANIFEST), "--output", str(expanded)).returncode, 0)
        package = root / "package"
        package.mkdir()
        (package / "weights.bin").write_bytes(b"synthetic-package")
        package_manifest = root / "package-manifest.json"; package_manifest.write_text('{"schema":"synthetic"}\n')
        worker = root / "worker.bin"; worker.write_bytes(b"synthetic-worker")
        tokenizer = root / "tokenizer.json"; tokenizer.write_text('{"tokenizer":"synthetic"}\n')
        served = root / "served-model.json"; served.write_text('{"model":"Qwen3.5-9B"}\n')
        model = root / "model-identity.json"; model.write_text('{"id":"synthetic-qwen35","revision":"fixture","format_id":"AQ4_0","implementation_id":"synthetic"}\n')
        graph = root / "graph.json"; graph.write_text('{"source":"fixture","nodes":1}\n')
        state = root / "state.json"; state.write_text('{"schema":"fixture-state-v1"}\n')
        oracle = root / "source-oracle.json"; oracle.write_text('{"oracle":"independent","version":1}\n')
        power = root / "power.json"; power.write_text('{"device":"cpu-reference","watts":0}\n')
        baseline = root / "baseline.json"; baseline.write_text('{"status":"synthetic"}\n')
        identity = root / "identity.json"; bound_policy = root / "bound-policy.json"
        completed = invoke(BIND, "--manifest", str(MANIFEST), "--policy", str(POLICY), "--expanded", str(expanded), "--worker", str(worker), "--package-root", str(package), "--package-manifest", str(package_manifest), "--tokenizer", str(tokenizer), "--served-model-manifest", str(served), "--model-identity", str(model), "--graph", str(graph), "--state", str(state), "--source-oracle", str(oracle), "--power-capture", str(power), "--baseline-result", str(baseline), "--git-commit", "a" * 40, "--output", str(identity), "--bound-policy", str(bound_policy))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        case = next(case for case in json.loads(expanded.read_text())["cases"] if case["stage_id"] == "smoke" and case["mode"] == "all_m1")
        case_path = root / "case.json"; case_path.write_text(json.dumps(case, sort_keys=True) + "\n")
        preflight = root / "preflight.json"; preflight.write_text(json.dumps({"weights_bytes": 1, "persistent_state_bytes": 1, "kv_cache_bytes": 1, "workspace_bytes": 1, "temporary_bytes": 1, "vram_headroom_bytes": 1, "gpu_process_snapshot": []}) + "\n")
        raw = root / "raw.json"
        completed = invoke(RUN, "--case", str(case_path), "--identity", str(identity), "--preflight", str(preflight), "--output", str(raw), "--command", "python3", "-c", "print('synthetic')")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = root / "result.json"
        completed = invoke(BUILD, "--case", str(case_path), "--expanded", str(expanded), "--raw", str(raw), "--identity", str(identity), "--policy", str(bound_policy), "--source-oracle", str(oracle), "--output", str(result))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = root / "report.json"
        completed = invoke(VALIDATE, "--expanded", str(expanded), "--identity", str(identity), "--policy", str(bound_policy), "--source-oracle", str(oracle), "--result", str(result), "--output", str(report))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(json.loads(report.read_text())["promotion_eligible"])
        return raw, report, case

    def test_cpu_synthetic_bridge_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw, report, _ = self._prepare(Path(directory))
            self.assertEqual(json.loads(report.read_text())["status"], "valid")
            raw.write_text(raw.read_text().replace('"status": "ok"', '"status": "failed"', 1))
            tampered = Path(directory) / "tampered-report.json"
            completed = invoke(VALIDATE, "--expanded", str(Path(directory) / "expanded.json"), "--identity", str(Path(directory) / "identity.json"), "--policy", str(Path(directory) / "bound-policy.json"), "--source-oracle", str(Path(directory) / "source-oracle.json"), "--result", str(Path(directory) / "result.json"), "--output", str(tampered))
            self.assertNotEqual(completed.returncode, 0)

    def test_production_missing_real_artifacts_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root / "case.json"; case.write_text('{"case_id":"p2-production-missing","device":{"device_id":"cpu-reference"},"scope":"production_server"}\n')
            preflight = root / "preflight.json"; preflight.write_text(json.dumps({"weights_bytes": 1, "persistent_state_bytes": 1, "kv_cache_bytes": 1, "workspace_bytes": 1, "temporary_bytes": 1, "vram_headroom_bytes": 1, "gpu_process_snapshot": []}) + "\n")
            output = root / "raw.json"
            completed = invoke(RUN, "--case", str(case), "--preflight", str(preflight), "--mode", "production", "--output", str(output), "--command", "python3", "-c", "print('must-not-run')")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(json.loads(output.read_text())["status"], {"failed", "skipped"})


if __name__ == "__main__":
    unittest.main()
