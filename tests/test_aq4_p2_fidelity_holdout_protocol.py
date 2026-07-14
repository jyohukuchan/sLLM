from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GEN_PATH = ROOT / "tools" / "generate-aq4-p2-fidelity-holdout.py"
spec = importlib.util.spec_from_file_location("fidelity_generator", GEN_PATH)
assert spec and spec.loader
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class FidelityProtocolTests(unittest.TestCase):
    def make_inputs(self, root: Path) -> tuple[Path, Path]:
        fixtures = root / "fixtures"
        fixtures.mkdir()
        cases = []
        entries = []
        prompts = (1011, 1024, 1339, 2048)
        modes = ("all_m1", "cold_batched")
        m_values = (1, 8, 16, 32, 64, 128)
        for prompt in prompts:
            for mode in modes:
                for requested_m in m_values:
                    case = {
                        "case_id": f"synthetic-{prompt}-{mode}-m{requested_m}", "fixture_id": "synthetic", "case_sha256": None,
                        "stage_id": "representative", "stage_order": 2, "scope": "production_server", "phase": "cold_prefill", "mode": mode, "baseline_mode": mode,
                        "prompt_tokens": prompt, "cached_prefix_tokens": 0, "context_tokens": prompt, "decode_start_tokens": 0, "prefill_requested_m": requested_m,
                        "resolved_m": 1 if mode == "all_m1" else requested_m, "request_count": 1, "decode_request_count": 0, "generated_tokens": 0,
                        "device": {"device_id": "r9700-rdna4", "backend": "hip", "name": "R9700", "architecture": "RDNA4", "runtime_device_index": 0},
                        "control_id": "aq4_0_target", "control": {}, "sampling": {"mode": "greedy"}, "format_id": "AQ4_0", "implementation_id": "qwen35_aq4_rdna4_v1",
                        "path_oracle_case_id": None, "path_oracle_result_sha256": None,
                    }
                    case["case_sha256"] = generator.case_hash(case)
                    cases.append(case)
                    ids = list(range(prompt))
                    fixture = {"schema_version": generator.FIXTURE_SCHEMA, "cases": [{"case_id": case["case_id"], "prompt_token_ids": ids, "step_count": 0}]}
                    fixture_path = fixtures / f"{case['case_id']}.json"
                    fixture_path.write_bytes(json.dumps(fixture, sort_keys=True, indent=2).encode() + b"\n")
                    entries.append({"case_id": case["case_id"], "case_sha256": case["case_sha256"], "fixture_path": str(fixture_path), "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(), "prompt_tokens": prompt, "context_tokens": prompt, "generated_tokens": 0, "prompt_token_ids_sha256": hashlib.sha256(canonical(ids)).hexdigest()})
        expanded = {"schema_version": generator.EXPANDED_SCHEMA, "case_count": len(cases), "stage_case_count": {"representative": len(cases)}, "cases": cases}
        expanded_path = root / "expanded.json"
        expanded_path.write_bytes(json.dumps(expanded, sort_keys=True, indent=2).encode() + b"\n")
        index = {"schema_version": generator.INDEX_SCHEMA, "case_count": len(entries), "cases": entries}
        index_path = root / "fixture-index.json"
        index_path.write_bytes(json.dumps(index, sort_keys=True, indent=2).encode() + b"\n")
        return expanded_path, index_path

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(GEN_PATH), *args], cwd=ROOT, text=True, capture_output=True)

    def test_deterministic_split_and_validator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expanded, index = self.make_inputs(root)
            first, second = root / "first", root / "second"
            for output in (first, second):
                result = self.run_cli("split", "--expanded", str(expanded), "--fixture-index", str(index), "--output", str(output))
                self.assertEqual(result.returncode, 0, result.stderr)
                check = subprocess.run([sys.executable, str(ROOT / "tools" / "validate-aq4-p2-fidelity-holdout.py"), "--split-root", str(output)], cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(check.returncode, 0, check.stderr)
            self.assertEqual((first / "calibration-cases.jsonl").read_bytes(), (second / "calibration-cases.jsonl").read_bytes())
            self.assertEqual((first / "holdout-cases.jsonl").read_bytes(), (second / "holdout-cases.jsonl").read_bytes())
            calibration = (first / "calibration-cases.jsonl").read_text().splitlines()
            holdout = (first / "holdout-cases.jsonl").read_text().splitlines()
            self.assertEqual(len(calibration), 24)
            self.assertEqual(len(holdout), 24)
            self.assertTrue(set(calibration).isdisjoint(set(holdout)))

    def test_calibration_freeze_and_reject_missing_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expanded, index = self.make_inputs(root)
            split = root / "split"
            result = self.run_cli("split", "--expanded", str(expanded), "--fixture-index", str(index), "--output", str(split))
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((split / "split-manifest.json").read_bytes())
            metric_rows = []
            for line in (split / "calibration-cases.jsonl").read_text().splitlines():
                row = json.loads(line)
                item = {field: row[field] for field in ("case_id", "case_sha256", "fixture_sha256", "prompt_token_ids_sha256", "context_token_ids_sha256", "prompt_tokens", "context_tokens", "baseline_mode", "prefill_requested_m", "resolved_m", "step", "row_count")}
                item["metrics"] = {name: (1.0 if name in generator.BINARY_RATE_METRICS else 0.8) for name in generator.METRICS}
                metric_rows.append(item)
            metrics = {"schema_version": generator.METRICS_SCHEMA, "split_manifest_sha256": hashlib.sha256((split / "split-manifest.json").read_bytes()).hexdigest(), "subset": "calibration", "rows": metric_rows}
            metrics_path = root / "metrics.json"
            metrics_path.write_text(json.dumps(metrics, sort_keys=True, indent=2) + "\n")
            receipt = root / "freeze-receipt.json"
            frozen = self.run_cli("freeze", "--split-root", str(split), "--metrics", str(metrics_path), "--output", str(receipt))
            self.assertEqual(frozen.returncode, 0, frozen.stderr)
            check = subprocess.run([sys.executable, str(ROOT / "tools" / "validate-aq4-p2-fidelity-holdout.py"), "--split-root", str(split), "--receipt", str(receipt)], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(check.returncode, 0, check.stderr)
            receipt_value = json.loads(receipt.read_text())
            self.assertAlmostEqual(receipt_value["derived_bounds"]["token_agreement_rate"]["bound"], generator.wilson_lower_one_sided(24, 24), places=12)
            self.assertIsNone(receipt_value["derived_bounds"]["hidden_max_abs"]["bound"])
            metrics["rows"][0]["metrics"].pop("logits_cosine")
            bad = root / "bad-metrics.json"
            bad.write_text(json.dumps(metrics, sort_keys=True) + "\n")
            rejected = self.run_cli("freeze", "--split-root", str(split), "--metrics", str(bad), "--output", str(root / "bad-receipt.json"))
            self.assertNotEqual(rejected.returncode, 0)

    def test_relative_l2_over_one_is_pathological_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expanded, index = self.make_inputs(root)
            split = root / "split"
            result = self.run_cli("split", "--expanded", str(expanded), "--fixture-index", str(index), "--output", str(split))
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = []
            for line in (split / "calibration-cases.jsonl").read_text().splitlines():
                row = json.loads(line)
                item = {field: row[field] for field in ("case_id", "case_sha256", "fixture_sha256", "prompt_token_ids_sha256", "context_token_ids_sha256", "prompt_tokens", "context_tokens", "baseline_mode", "prefill_requested_m", "resolved_m", "step", "row_count")}
                item["metrics"] = {name: (1.0 if name in generator.BINARY_RATE_METRICS else 0.8) for name in generator.METRICS}
                item["metrics"]["hidden_max_abs"] = 100.0
                item["metrics"]["logits_relative_l2"] = 1.01
                rows.append(item)
            metrics = {"schema_version": generator.METRICS_SCHEMA, "split_manifest_sha256": hashlib.sha256((split / "split-manifest.json").read_bytes()).hexdigest(), "subset": "calibration", "rows": rows}
            metrics_path = root / "metrics.json"
            metrics_path.write_text(json.dumps(metrics) + "\n")
            rejected = self.run_cli("freeze", "--split-root", str(split), "--metrics", str(metrics_path), "--output", str(root / "receipt.json"))
            self.assertNotEqual(rejected.returncode, 0)

    def test_attempt2_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expanded, index = self.make_inputs(root)
            split = root / "split"
            result = self.run_cli("split", "--expanded", str(expanded), "--fixture-index", str(index), "--output", str(split))
            self.assertEqual(result.returncode, 0, result.stderr)
            policy = json.loads((split / "policy.json").read_text())
            policy["attempt2_threshold_source_forbidden"] = False
            (split / "policy.json").write_text(json.dumps(policy) + "\n")
            check = subprocess.run([sys.executable, str(ROOT / "tools" / "validate-aq4-p2-fidelity-holdout.py"), "--split-root", str(split)], cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(check.returncode, 0)

    def test_identity_and_attempt2_metrics_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expanded, index = self.make_inputs(root)
            split = root / "split"
            result = self.run_cli("split", "--expanded", str(expanded), "--fixture-index", str(index), "--output", str(split))
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = []
            for line in (split / "calibration-cases.jsonl").read_text().splitlines():
                row = json.loads(line)
                item = {field: row[field] for field in ("case_id", "fixture_sha256", "prompt_token_ids_sha256", "context_token_ids_sha256", "prompt_tokens", "context_tokens", "baseline_mode", "prefill_requested_m", "resolved_m", "step", "row_count")}
                item["case_sha256"] = "0" * 64
                item["metrics"] = {name: (1.0 if name in generator.BINARY_RATE_METRICS else 0.8) for name in generator.METRICS}
                rows.append(item)
            metrics = {"schema_version": generator.METRICS_SCHEMA, "split_manifest_sha256": hashlib.sha256((split / "split-manifest.json").read_bytes()).hexdigest(), "subset": "calibration", "rows": rows, "source": "attempt2 observed"}
            metrics_path = root / "bad-metrics.json"
            metrics_path.write_text(json.dumps(metrics) + "\n")
            rejected = self.run_cli("freeze", "--split-root", str(split), "--metrics", str(metrics_path), "--output", str(root / "receipt.json"))
            self.assertNotEqual(rejected.returncode, 0)


if __name__ == "__main__":
    unittest.main()
