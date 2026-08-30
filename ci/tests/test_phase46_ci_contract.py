#!/usr/bin/env python3
"""Host-only schema and policy contracts for Phase 46 tool outputs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "ci/schema"
POLICY = ROOT / "ci/policy/kv-cache-default-v1.json"
DATASET = ROOT / "ci/fixtures/phase46-kv-quality-baseline-v1.json"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value: str = "a") -> str:
    return value * 64


def manifest() -> dict[str, object]:
    return {
        "schema_version": "sllm-phase46-tool-run-v1",
        "struct_size": 13,
        "canonicalization": "sllm-sorted-json-v1",
        "operation": "fixture",
        "state": "PASS",
        "selected_count": 1,
        "tool": {
            "repository": "https://github.com/89chin/sLLM",
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "package": "sllm-tools",
            "version": "0.1.0",
            "executable_sha256": digest("b"),
            "arguments": ["fixture"],
            "environment": {"offline": "true"},
        },
        "recipe": {"id": "fixture", "version": "v1", "config_sha256": digest("c")},
        "sources": [{"role": "source", "logical_name": "source.bin", "size_bytes": 1, "sha256": digest("d")}],
        "outputs": [{"role": "output", "logical_name": "result.bin", "size_bytes": 1, "sha256": digest("e")}],
        "raw_evidence": [],
        "identities": {"model": "fixture"},
        "metrics": {"selected": 1},
        "extensions": {},
    }


class Phase46CiContractTests(unittest.TestCase):
    def assert_valid(self, schema_name: str, value: dict[str, object]) -> None:
        schema = load(SCHEMA_DIR / schema_name)
        errors = sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(value),
            key=lambda error: list(error.path),
        )
        self.assertEqual(errors, [], "schema errors: " + "; ".join(error.message for error in errors))

    def assert_invalid(self, schema_name: str, value: dict[str, object]) -> None:
        schema = load(SCHEMA_DIR / schema_name)
        self.assertTrue(list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(value)))

    def test_all_phase46_schemas_are_valid_draft202012(self) -> None:
        for path in sorted(SCHEMA_DIR.glob("phase46-*.schema.json")):
            Draft202012Validator.check_schema(load(path))
        Draft202012Validator.check_schema(load(SCHEMA_DIR / "kv-cache-default-v1.schema.json"))

    def test_kv_quality_dataset_has_frozen_identity_and_boundaries(self) -> None:
        raw = DATASET.read_bytes()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "a2252d882ffd7e1fbb546d86b2b573bd2410467382c7da874f4fbd3dc8adc77d",
        )
        value = json.loads(raw)
        self.assert_valid("phase46-kv-quality-dataset-v1.schema.json", value)
        self.assertEqual(
            [case["length"] for case in value["cases"]],
            [1, 15, 16, 17, 255, 256, 257, 511, 512, 513],
        )

    def test_common_manifest_identity_and_additive_contract(self) -> None:
        value = manifest()
        self.assert_valid("phase46-tool-run-v1.schema.json", value)
        unknown = copy.deepcopy(value)
        unknown["unexpected_required_field"] = True
        self.assert_invalid("phase46-tool-run-v1.schema.json", unknown)
        zero = copy.deepcopy(value)
        zero["selected_count"] = 0
        self.assert_invalid("phase46-tool-run-v1.schema.json", zero)
        bad_digest = copy.deepcopy(value)
        bad_digest["tool"]["executable_sha256"] = "sha256:" + digest("f")
        self.assert_invalid("phase46-tool-run-v1.schema.json", bad_digest)

    def test_quality_result_rejects_zero_and_nonfinite_metrics(self) -> None:
        value = {
            "$schema": "https://sllm.dev/schema/phase46-quality-result-v1.schema.json",
            "schema_version": "sllm-phase46-quality-result-v1",
            "struct_size": 7,
            "state": "PASS",
            "metric": "perplexity",
            "manifest": manifest(),
            "result": {"loss_sum": 1.0, "token_count": 2, "mean_nll": 0.5, "perplexity": 1.6487212707},
            "extensions": {},
        }
        self.assert_valid("phase46-quality-result-v1.schema.json", value)
        zero = copy.deepcopy(value)
        zero["result"]["token_count"] = 0
        self.assert_invalid("phase46-quality-result-v1.schema.json", zero)
        nonfinite = copy.deepcopy(value)
        nonfinite["result"]["perplexity"] = math.nan
        # Python's jsonschema treats NaN as a ``number`` although it is not a
        # JSON number.  Keep the explicit finite-number guard in the contract
        # test so a permissive parser cannot turn NaN into a quality result.
        self.assertFalse(math.isfinite(nonfinite["result"]["perplexity"]))

    def test_benchmark_result_requires_nonzero_measurements_and_distinguishes_resource_state(self) -> None:
        sample = {
            "iteration": 1,
            "state": "PASS",
            "reason": None,
            "wall_ns": 10,
            "gpu_ns": 9,
            "model_load_ns": 2,
            "e2e_ns": 8,
            "ttft_ns": 3,
            "tpot_ns": 1,
            "prefill_ns": 4,
            "decode_ns": 4,
        }
        resource = {"status": "measured", "bytes": 0}
        resources = {name: resource for name in ("hbm_before", "hbm_peak", "hbm_settled", "gtt_before", "gtt_peak", "gtt_settled", "model_resident", "kv_logical", "kv_physical", "workspace")}
        value = {
            "$schema": "https://sllm.dev/schema/phase46-benchmark-result-v1.schema.json",
            "schema_version": "sllm-phase46-benchmark-result-v1",
            "struct_size": 7,
            "state": "PASS",
            "manifest": manifest(),
            "payload": {
                "configuration": {"request_count": 1, "parallelism": 1, "context_tokens": 17, "sampling": "greedy", "kv_encoding": "fp16", "gpu_identity": "fixture", "provider": "hip", "fallback": False, "cleanup": True},
                "warmups": [sample], "measured": [sample], "rejected": [],
                "timing": {name: {"count": 1, "min": 1, "p10": 1, "median": 1, "p90": 1, "max": 1, "mad": 0} for name in ("wall_ns", "model_load_ns", "e2e_ns", "ttft_ns", "tpot_ns", "prefill_ns", "decode_ns")} | {"gpu_ns": None},
                "resources": resources,
                "decisions": {name: "PASS" for name in ("correctness", "quality", "performance", "memory", "fallback", "cleanup")},
            },
            "extensions": {},
        }
        self.assert_valid("phase46-benchmark-result-v1.schema.json", value)
        zero = copy.deepcopy(value)
        zero["payload"]["measured"] = []
        self.assert_invalid("phase46-benchmark-result-v1.schema.json", zero)
        unsupported = copy.deepcopy(value)
        unsupported["payload"]["resources"]["hbm_peak"] = {"status": "unsupported", "reason": "sampler unavailable"}
        self.assert_valid("phase46-benchmark-result-v1.schema.json", unsupported)

    def test_debug_dump_schema_is_opt_in_bounded_and_closed(self) -> None:
        value = {
            "$schema": "https://sllm.dev/schema/phase46-debug-dump-v1.schema.json",
            "schema_version": "sllm-phase46-debug-dump-v1",
            "struct_size": 8,
            "manifest": manifest(),
            "metadata": {"target": "fixture", "token_count": 1},
            "tokens": [1],
            "tensors": [{"name": "hidden", "dtype": "BF16", "shape": [1], "layout": "row-major", "endianness": "little", "quantization": None, "scale_plane": None, "values": [0.0]}],
            "logits": [{"layer": 0, "position": 0, "top_k": [{"token_index": 1, "logit": 0.0}]}],
            "extensions": {},
        }
        self.assert_valid("phase46-debug-dump-v1.schema.json", value)
        forbidden = copy.deepcopy(value)
        forbidden["metadata"]["prompt"] = "must not be persisted"
        self.assert_invalid("phase46-debug-dump-v1.schema.json", forbidden)
        over_limit = copy.deepcopy(value)
        over_limit["logits"][0]["top_k"] = [{"token_index": i, "logit": 0.0} for i in range(65)]
        self.assert_invalid("phase46-debug-dump-v1.schema.json", over_limit)

    def test_kv_policy_is_baseline_frozen_and_target_independent(self) -> None:
        value = load(POLICY)
        self.assert_valid("kv-cache-default-v1.schema.json", value)
        self.assertNotIn("policy_digest", value)
        self.assertEqual(value["scope"]["model"]["repo_id"], "Qwen/Qwen3.5-4B")
        self.assertEqual(value["scope"]["model"]["dtype"], "BF16")
        lock_binding = value["scope"]["model"]["model_lock"]
        lock_path = ROOT / lock_binding["path"]
        self.assertTrue(lock_path.is_file())
        locked_model = load(lock_path)
        self.assertEqual(locked_model["fingerprint"], lock_binding["fingerprint"])
        self.assertTrue(value["freeze"]["candidate_excluded"])
        self.assertEqual(value["freeze"]["basis"], "fp16-only-baseline")
        self.assertEqual(value["freeze"]["dataset"]["sample_order"], "listed")
        self.assertEqual({target["decision"] for target in value["targets"]}, {"insufficient-evidence"})
        self.assertTrue(all(target["candidate"] is None for target in value["targets"]))
        self.assertEqual(value["failure_rules"]["missing"], "fail")
        self.assertEqual(value["failure_rules"]["nonfinite"], "fail")
        self.assertEqual(value["failure_rules"]["zero_selected"], "fail")
        self.assertEqual(value["failure_rules"]["zero_metric_samples"], "fail")
        self.assertEqual(value["failure_rules"]["reruns"]["same_identity"], True)
        for threshold in value["thresholds"].values():
            self.assertIn(threshold["comparison"], {"inclusive", "exclusive"})
        targets = {target["target"]: target for target in value["targets"]}
        baseline_schema = load(SCHEMA_DIR / "phase46-qwen35-quality-baseline-v1.schema.json")
        self.assertIn(
            "gfx942:sramecc+:xnack-",
            baseline_schema["properties"]["target"]["enum"],
        )
        self.assertTrue(
            all(target["baseline"]["dataset"]["sample_order"] == "listed" for target in value["targets"])
        )
        self.assertEqual(targets["gfx942:sramecc+:xnack-"]["physical_variant"], "E4M3-FNUZ")
        self.assertEqual(targets["gfx1201"]["physical_variant"], "E4M3-OCP")
        self.assertEqual(targets["gfx1030"]["physical_variant"], "E5M2-software")
        self.assertEqual(targets["gfx1030"]["baseline"]["status"], "frozen")
        self.assertTrue(targets["gfx1030"]["baseline"]["evidence"]["cleanup_empty"])
        self.assertEqual(targets["gfx1030"]["baseline"]["evidence"]["top1_agreement"], 1.0)
        self.assertEqual(targets["gfx1030"]["baseline"]["evidence"]["sample_count"], 20)
        self.assertEqual(targets["gfx1201"]["baseline"]["status"], "required-before-candidate")
        self.assertIsNone(targets["gfx1201"]["baseline"]["evidence"])
        self.assertEqual(targets["gfx942:sramecc+:xnack-"]["baseline"]["status"], "required-before-candidate")


if __name__ == "__main__":
    unittest.main(verbosity=2)
