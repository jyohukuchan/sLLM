#!/usr/bin/env python3
"""Fail-closed host contract for Phase 54 direct KV research evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "ci/schema/phase54-kv-fp8-block16-research-evidence-v1.schema.json"
REPORT_DIR = Path("/home/homelab1/.local/share/sllm-evidence/phase54/gfx1030")


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest_spec(spec: dict[str, object]) -> str:
    encoded = json.dumps(spec, ensure_ascii=False, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def direct_reports() -> list[Path]:
    return sorted(REPORT_DIR.glob("direct-*.json"))


def control_report() -> dict[str, object]:
    """Construct a self-contained valid PASS report for CI without local GPU evidence."""
    spec: dict[str, object] = {
        "schema_version": "sllm-phase54-kv-candidate-spec-v1",
        "candidate_id": "production-control-v2",
        "scale_selector": "independent-k-v-closed-enum-v1",
        "rounding": "nearest-even",
        "k_recipe": "floor",
        "v_recipe": "floor",
        "transform": "none",
        "calibration_digest": None,
        "descriptor_compatibility": "exact-production-v2",
    }
    case_specs = [
        ("append-head-dim-15", 15, 1),
        ("append-head-dim-16", 16, 1),
        ("append-head-dim-17", 17, 1),
        ("append-head-dim-255", 255, 1),
        ("attention-kv2", 256, 2),
        ("append-head-dim-257", 257, 1),
        ("signed-zero-only", 16, 1),
        ("recipe-distinguishing", 16, 1),
    ]
    cases = []
    for case_id, head_dim, token_count in case_specs:
        attention = case_id == "attention-kv2"
        cases.append({
            "id": case_id,
            "head_dim": head_dim,
            "token_count": token_count,
            "signed_zero_only": case_id == "signed-zero-only",
            "recipe_distinguishing": case_id == "recipe-distinguishing",
            "key_values_exact": True,
            "value_values_exact": True,
            "key_scales_exact": True,
            "value_scales_exact": True,
            "tail_padding_zero": True,
            "append_direct": True,
            "attention_direct": attention,
            "attention_numerical_match": True,
            "attention_key_contributes": attention,
            "finite": True,
        })
    return {
        "$schema": "https://sllm.dev/schema/phase54-kv-fp8-block16-research-evidence-v1.schema.json",
        "schema_version": "sllm-phase54-kv-fp8-block16-research-evidence-v1",
        "state": "PASS",
        "target": "gfx1030",
        "device_index": 0,
        "encoding": "kv-fp8-e5-block16",
        "physical_variant": "E5M2-software",
        "production_descriptor_id": "kv-fp8-e5-block16-v2",
        "descriptor_compatibility": "exact-production-v2",
        "candidate_spec": spec,
        "candidate_spec_sha256": digest_spec(spec),
        "binary_sha256": "sha256:" + "0" * 64,
        "host": {
            "pass": True,
            "head_dimensions": [15, 16, 17, 255, 256, 257],
            "recipes": ["floor", "floor"],
            "signed_zero_only": True,
            "recipe_distinguishing": False,
            "tail_padding_zero": True,
            "finite": True,
        },
        "cases": cases,
        "execution": {
            "selected_backend": "hip",
            "gpu_execution": True,
            "fallback_allowed": False,
            "fallback_used": False,
            "append_dispatches": 8,
            "attention_dispatches": 1,
            "sequential_residents": True,
        },
        "cleanup": {"retryable": 0, "durable": 0, "terminal_zero": True},
        "error": None,
    }


def nonfloor_report() -> dict[str, object]:
    value = control_report()
    spec = value["candidate_spec"]
    spec.update({
        "candidate_id": "phase54-k-floor-v-ceil-v1",
        "v_recipe": "ceil",
        "descriptor_compatibility": "research-build-semantic-override-not-v2-compatible",
    })
    value["descriptor_compatibility"] = "research-build-semantic-override-not-v2-compatible"
    value["host"]["recipes"] = ["floor", "ceil"]
    value["candidate_spec_sha256"] = digest_spec(spec)
    del value["production_descriptor_id"]
    return value


class Phase54DirectEvidenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load(SCHEMA_PATH)
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema, format_checker=FormatChecker())

    def assert_valid(self, value: dict[str, object]) -> None:
        errors = sorted(self.validator.iter_errors(value), key=lambda error: list(error.path))
        self.assertEqual(errors, [], "; ".join(error.message for error in errors))

    def assert_invalid(self, value: dict[str, object]) -> None:
        self.assertTrue(list(self.validator.iter_errors(value)))

    def test_self_contained_control_and_nonfloor_fixtures_validate(self) -> None:
        self.assert_valid(control_report())
        self.assert_valid(nonfloor_report())

    def report(self, candidate_id: str | None = None) -> dict[str, object]:
        paths = direct_reports()
        if not paths:
            if candidate_id in (None, "production-control-v2"):
                return control_report()
            if candidate_id == "phase54-k-floor-v-ceil-v1":
                return nonfloor_report()
            self.fail(f"self-contained candidate report not found: {candidate_id}")
        values = [load(path) for path in paths]
        if candidate_id is not None:
            for value in values:
                if value["candidate_spec"]["candidate_id"] == candidate_id:
                    return value
            self.fail(f"candidate report not found: {candidate_id}")
        return values[0]

    def test_checked_in_schema_and_external_direct_reports_validate(self) -> None:
        paths = direct_reports()
        if not paths:
            self.skipTest(f"direct Phase 54 reports are not available under {REPORT_DIR}")
        expected = {
            "production-control-v2",
            "phase54-k-floor-v-ceil-v1",
            "phase54-k-floor-v-nearest-even-v1",
            "phase54-k-ceil-v-floor-v1",
            "phase54-k-nearest-even-v-floor-v1",
        }
        observed = set()
        for path in paths:
            value = load(path)
            self.assert_valid(value)
            observed.add(value["candidate_spec"]["candidate_id"])
            self.assertEqual(value["candidate_spec_sha256"], digest_spec(value["candidate_spec"]))
        self.assertTrue(expected.issubset(observed), (expected, observed))

    def test_candidate_recipe_id_compatibility_and_descriptor_presence_are_paired(self) -> None:
        for path in direct_reports():
            value = load(path)
            spec = value["candidate_spec"]
            production = spec["k_recipe"] == "floor" and spec["v_recipe"] == "floor"
            self.assertEqual("production-control-v2" if production else f"phase54-k-{spec['k_recipe']}-v-{spec['v_recipe']}-v1", spec["candidate_id"])
            self.assertEqual("exact-production-v2" if production else "research-build-semantic-override-not-v2-compatible", value["descriptor_compatibility"])
            if production:
                self.assertIn("production_descriptor_id", value)
            else:
                self.assertNotIn("production_descriptor_id", value)

        value = self.report("production-control-v2")
        invalid = copy.deepcopy(value)
        invalid["candidate_spec"]["candidate_id"] = "phase54-k-floor-v-ceil-v1"
        self.assert_invalid(invalid)
        invalid = copy.deepcopy(value)
        invalid["host"]["recipes"] = ["floor", "ceil"]
        self.assert_invalid(invalid)
        invalid = copy.deepcopy(value)
        invalid["descriptor_compatibility"] = "research-build-semantic-override-not-v2-compatible"
        self.assert_invalid(invalid)
        nonfloor = next(
            (load(path) for path in direct_reports()
             if load(path)["candidate_spec"]["candidate_id"] != "production-control-v2"),
            nonfloor_report(),
        )
        invalid = copy.deepcopy(value)
        del invalid["production_descriptor_id"]
        self.assert_invalid(invalid)
        invalid = copy.deepcopy(nonfloor)
        invalid["production_descriptor_id"] = "kv-fp8-e5-block16-v2"
        self.assert_invalid(invalid)

    def test_exact_gfx_target_encoding_variant_and_production_descriptor_mapping(self) -> None:
        value = self.report("production-control-v2")
        self.assertEqual(value["target"], "gfx1030")
        self.assertEqual(value["encoding"], "kv-fp8-e5-block16")
        self.assertEqual(value["physical_variant"], "E5M2-software")

        gfx1201 = copy.deepcopy(value)
        gfx1201.update({
            "target": "gfx1201",
            "encoding": "kv-fp8-e4-block16",
            "physical_variant": "E4M3-OCP",
            "production_descriptor_id": "kv-fp8-e4-block16-v2",
        })
        self.assert_valid(gfx1201)
        for field, wrong in (("encoding", "kv-fp8-e5-block16"), ("physical_variant", "E5M2-software"), ("production_descriptor_id", "kv-fp8-e5-block16-v2")):
            invalid = copy.deepcopy(gfx1201)
            invalid[field] = wrong
            self.assert_invalid(invalid)

    def test_pass_requires_dimensions_special_cases_attention_and_direct_execution(self) -> None:
        value = self.report()
        self.assertEqual([case["id"] for case in value["cases"]], [
            "append-head-dim-15", "append-head-dim-16", "append-head-dim-17", "append-head-dim-255",
            "attention-kv2", "append-head-dim-257", "signed-zero-only", "recipe-distinguishing",
        ])
        attention = value["cases"][4]
        self.assertEqual(attention["token_count"], 2)
        self.assertTrue(attention["attention_direct"])
        self.assertTrue(attention["attention_numerical_match"])
        self.assertTrue(attention["attention_key_contributes"])
        for field, replacement in (("signed_zero_only", False), ("recipe_distinguishing", False)):
            invalid = copy.deepcopy(value)
            invalid["cases"][6 if field == "signed_zero_only" else 7][field] = replacement
            self.assert_invalid(invalid)
        for field, replacement in (("token_count", 1), ("attention_key_contributes", False), ("attention_numerical_match", False)):
            invalid = copy.deepcopy(value)
            invalid["cases"][4][field] = replacement
            self.assert_invalid(invalid)
        invalid = copy.deepcopy(value)
        invalid["cases"][0], invalid["cases"][1] = invalid["cases"][1], invalid["cases"][0]
        self.assert_invalid(invalid)

        for field, replacement in (("selected_backend", "cpu"), ("gpu_execution", False), ("fallback_allowed", True), ("fallback_used", True), ("sequential_residents", False), ("append_dispatches", 0), ("attention_dispatches", 0)):
            invalid = copy.deepcopy(value)
            invalid["execution"][field] = replacement
            self.assert_invalid(invalid)
        for field, replacement in (("retryable", 1), ("durable", 1), ("terminal_zero", False)):
            invalid = copy.deepcopy(value)
            invalid["cleanup"][field] = replacement
            self.assert_invalid(invalid)

    def test_pass_requires_hashes_and_finite_counts(self) -> None:
        value = self.report()
        for owner, field in ((value, "candidate_spec_sha256"), (value, "binary_sha256")):
            invalid = copy.deepcopy(owner)
            invalid[field] = "sha256:not-a-digest"
            self.assert_invalid(invalid)
        invalid = copy.deepcopy(value)
        invalid["host"]["finite"] = False
        self.assert_invalid(invalid)
        invalid = copy.deepcopy(value)
        invalid["cases"][0]["finite"] = False
        self.assert_invalid(invalid)
        invalid = copy.deepcopy(value)
        invalid["execution"]["append_dispatches"] = -1
        self.assert_invalid(invalid)
        invalid = copy.deepcopy(value)
        invalid["cleanup"]["retryable"] = -1
        self.assert_invalid(invalid)

    def test_schema_is_registered_with_manifest_validation(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT / "ci/tools"))
        import validate_json_manifests  # noqa: PLC0415

        self.assertIn(SCHEMA_PATH.relative_to(ROOT).as_posix(), validate_json_manifests.PHASE54_SCHEMA_FILES)


if __name__ == "__main__":
    unittest.main()
