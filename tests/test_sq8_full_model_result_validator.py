import copy
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "tools" / "validate-sq8-full-model-result.py"
FIXTURE = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "2026-07-10"
    / "sq8-full-model-v0.1"
    / "full-model-m8-final.json"
)


class Sq8FullModelResultValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not FIXTURE.is_file():
            raise RuntimeError(f"fixed full-model fixture is absent: {FIXTURE}")
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        spec = importlib.util.spec_from_file_location("sq8_full_model_validator", VALIDATOR)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load validator: {VALIDATOR}")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def run_validator(
        self, path: Path, *, contract_only: bool = False
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(VALIDATOR)]
        if contract_only:
            command.append("--contract-only")
        command.append(str(path))
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def write_modified(self, root: Path, mutate) -> Path:
        value = copy.deepcopy(self.fixture)
        mutate(value)
        path = root / "modified.json"
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path

    def assert_contract_rejects(self, mutate, expected_error: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_modified(Path(temporary), mutate)
            result = self.run_validator(path, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(expected_error, result.stderr)

    def test_final_fixture_passes_promotion(self) -> None:
        result = self.run_validator(FIXTURE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("passed=true mode=promotion trusted=true", result.stdout)
        self.assertIn("layers=40 resident_payloads=162 payload_records=163", result.stdout)
        self.assertIn("projections=280 activation_quantizations=160", result.stdout)
        self.assertIn("hash_stability_checks=13 top1=353", result.stdout)

    def test_final_fixture_passes_contract_only_as_untrusted(self) -> None:
        result = self.run_validator(FIXTURE, contract_only=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode=contract-only trusted=false", result.stdout)

    def test_self_consistent_timing_tamper_is_contract_only(self) -> None:
        def mutate(value) -> None:
            value["timing"]["stack"]["samples_ms"][0] += 0.01
            value["timing"]["full_stack_and_head"]["samples_ms"][0] += 0.01
            for name in ("stack", "full_stack_and_head"):
                samples = value["timing"][name]["samples_ms"]
                value["timing"][name]["p50_ms"] = self.module.percentile(samples, 0.50)
                value["timing"][name]["p95_ms"] = self.module.percentile(samples, 0.95)

        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_modified(Path(temporary), mutate)
            promoted = self.run_validator(path)
            self.assertNotEqual(promoted.returncode, 0)
            self.assertIn("promotion trust anchor", promoted.stderr)

            contract_only = self.run_validator(path, contract_only=True)
            self.assertEqual(contract_only.returncode, 0, contract_only.stderr)
            self.assertIn("mode=contract-only trusted=false", contract_only.stdout)

    def test_gate_threshold_boundaries_are_inclusive(self) -> None:
        metrics = {
            "nonfinite_count": 0,
            "relative_l2": 0.10,
            "cosine_similarity": 0.995,
        }
        self.assertTrue(self.module.gate_verdict(metrics, 0.10, 0.995))

        above = dict(metrics)
        above["relative_l2"] = math.nextafter(0.10, math.inf)
        self.assertFalse(self.module.gate_verdict(above, 0.10, 0.995))

        below = dict(metrics)
        below["cosine_similarity"] = math.nextafter(0.995, -math.inf)
        self.assertFalse(self.module.gate_verdict(below, 0.10, 0.995))

    def test_reported_passed_does_not_hide_failed_layer_gate(self) -> None:
        def mutate(value) -> None:
            gate = value["layer_boundaries"][0]["optimized_vs_cpu_sq8"]
            gate["metrics"]["relative_l2"] = math.nextafter(0.10, math.inf)
            gate["passed"] = True
            value["passed"] = True

        self.assert_contract_rejects(mutate, "passed does not match its metrics")

    def test_counter_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["execution"].__setitem__("projections", 279),
            "execution.projections differs",
        )

    def test_dispatch_count_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["execution"]["dispatch_implementation_counts"].__setitem__(
                "mem_v1_default_tile_16x128x128", 159
            ),
            "dispatch counts do not cover exactly 280 projections",
        )

    def test_hash_stability_counter_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["execution"].__setitem__(
                "timed_output_hash_stability_checks", 12
            ),
            "execution.timed_output_hash_stability_checks differs",
        )

    def test_topk_overlap_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["final_head"].__setitem__(
                "device_vllm_top_10_overlap", 8
            ),
            "device_vllm_top_10_overlap is inconsistent",
        )

    def test_timing_percentile_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["timing"]["stack"].__setitem__(
                "p50_ms", value["timing"]["stack"]["p50_ms"] + 1.0
            ),
            "timing.stack.p50_ms does not match its samples",
        )

    def test_source_hash_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["source"].__setitem__(
                "artifact_content_sha256", "0" * 64
            ),
            "source.artifact_content_sha256 differs",
        )

    def test_payload_count_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["payloads"]["layer_norms"].pop(),
            "payloads.layer_norms must contain exactly 160 entries",
        )

    def test_layer_order_corruption_is_rejected(self) -> None:
        def mutate(value) -> None:
            value["layer_boundaries"][0], value["layer_boundaries"][1] = (
                value["layer_boundaries"][1],
                value["layer_boundaries"][0],
            )

        self.assert_contract_rejects(mutate, "layer_boundaries[0] is out of order")

    def test_layer_health_nonfinite_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["layer_boundaries"][0]["optimized_health"].__setitem__(
                "nonfinite", 1
            ),
            "layer_boundaries[0].optimized_health reports non-finite values",
        )

    def test_vram_sum_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["vram"].__setitem__(
                "minimum_accounted_resident_bytes",
                value["vram"]["minimum_accounted_resident_bytes"] + 1,
            ),
            "minimum_accounted_resident_bytes is not the component sum",
        )

    def test_nonfinite_value_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["cpu_oracle"].__setitem__("elapsed_ms", math.inf),
            "cpu_oracle.elapsed_ms must be finite",
        )

    def test_extra_schema_key_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value.__setitem__("untrusted_note", "not in schema"),
            "result keys differ",
        )


if __name__ == "__main__":
    unittest.main()
