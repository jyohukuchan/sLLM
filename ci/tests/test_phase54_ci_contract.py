#!/usr/bin/env python3
"""Closed host contracts for the Phase 54 KV-quality research report."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "ci/schema/phase54-qwen35-kv-quality-research-v1.schema.json"
sys.path.insert(0, str(ROOT / "ci/tools"))

import validate_json_manifests  # noqa: E402


def digest(character: str) -> str:
    return "sha256:" + character * 64


def candidate_spec() -> dict[str, object]:
    return {
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


def spec_digest(spec: dict[str, object]) -> str:
    # CandidateSpec field insertion order mirrors the Rust struct's serde
    # compact serialization, which is the runner's canonical digest input.
    encoded = json.dumps(spec, ensure_ascii=False, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def measured_row() -> dict[str, object]:
    return {
        "reference_top1": 42,
        "observed_top1": 42,
        "top1_match": True,
        "kld": 0.001,
        "max_abs_logit_delta": 0.125,
        "max_abs_logit_index": 17,
        "reference_finite": True,
        "observed_finite": True,
    }


def comparison(encoding: str, descriptor_id: str) -> dict[str, object]:
    return {
        "encoding": encoding,
        "descriptor_id": descriptor_id,
        "aggregate": {
            "selected_count": 2,
            "metric_sample_counts": {
                "perplexity": 1,
                "kld": 2,
                "top1": 2,
                "task": 1,
                "long-context": 1,
            },
            "perplexity_relative_delta": -0.002,
            "kld_p99": 0.001,
            "top1_agreement": 1.0,
            "task_score_delta": 0.0,
            "long_context_score_delta": 0.0,
            "first_top1_divergence": None,
            "maximum_logit_delta": {
                "case_id": "b255",
                "row": "decode",
                "measured_row_index": 1,
                "logit_index": 17,
                "max_abs_logit_delta": 0.125,
            },
            "finite": True,
            "hip_dispatches": 2,
            "fallback_used": False,
            "all_dispatches_hip": True,
        },
        "cases": [
            {
                "case_id": "b255",
                "length": 255,
                "long": True,
                "prefill_nll": {"reference": 2.0, "observed": 1.998, "delta": -0.002},
                "prefill": measured_row(),
                "decode": measured_row(),
            }
        ],
    }


def repeat(index: int, block: str, descriptor: str, mxfp8: str) -> dict[str, object]:
    return {
        "repeat": index,
        "completely_sequential_order": [
            "fp16",
            "production-control-block16",
            "candidate-block16",
            "mxfp8",
        ],
        "fp16_released_before_production_control": True,
        "production_control_released_before_candidate": True,
        "candidate_released_before_mxfp8": True,
        "mxfp8_released_after_repeat": True,
        "production_control": comparison(block, descriptor),
        "candidate": comparison(block, descriptor),
        "mxfp8": comparison(mxfp8, f"{mxfp8}-v1"),
    }


def report(repeat_count: int = 1, target: str = "gfx1030") -> dict[str, object]:
    spec = candidate_spec()
    block = "kv-fp8-e5-block16" if target == "gfx1030" else "kv-fp8-e4-block16"
    descriptor = f"{block}-v2"
    mxfp8 = "kv-mxfp8-e5" if target == "gfx1030" else "kv-mxfp8-e4"
    return {
        "$schema": "https://sllm.dev/schema/phase54-qwen35-kv-quality-research-v1.schema.json",
        "schema_version": "sllm-phase54-qwen35-kv-quality-research-v1",
        "state": "PASS",
        "identity": {
            "policy_sha256": digest("a"),
            "dataset_sha256": digest("b"),
            "model_lock_fingerprint": "sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae",
            "model_lock_sha256": digest("d"),
            "derived_lock_fingerprint": digest("e"),
            "derived_lock_sha256": digest("f"),
            "binary_sha256": digest("1"),
            "candidate_spec_sha256": spec_digest(spec),
            "production_descriptor_id": descriptor,
            "candidate_descriptor_id": descriptor,
            "descriptor_compatibility": "exact-production-v2",
        },
        "target": target,
        "device_index": 0,
        "encoding": block,
        "candidate_spec": spec,
        "repeat_count": repeat_count,
        "reference_encoding": "fp16",
        "session_scope": "single-process-single-hip-execution-session",
        "sequential_residents": True,
        "repeats": [repeat(index, block, descriptor, mxfp8) for index in range(1, repeat_count + 1)],
        "cleanup": {"retryable": 0, "durable": 0, "poisoned": False, "terminal_zero": True},
    }


def select_research_candidate(value: dict[str, object]) -> None:
    candidate_id = "phase54-k-ceil-v-nearest-even-v1"
    spec = value["candidate_spec"]
    spec.update({
        "candidate_id": candidate_id,
        "k_recipe": "ceil",
        "v_recipe": "nearest-even",
        "descriptor_compatibility": "research-build-semantic-override-not-v2-compatible",
    })
    descriptor = f"{value['identity']['production_descriptor_id']}-{candidate_id}"
    value["identity"].update({
        "candidate_spec_sha256": spec_digest(spec),
        "candidate_descriptor_id": descriptor,
        "descriptor_compatibility": "research-build-semantic-override-not-v2-compatible",
    })
    for item in value["repeats"]:
        item["candidate"]["descriptor_id"] = descriptor


def select_transform_candidate(
    value: dict[str, object], candidate_id: str, transform: str
) -> None:
    spec = value["candidate_spec"]
    spec.update({
        "candidate_id": candidate_id,
        "transform": transform,
        "descriptor_compatibility": "research-build-semantic-override-not-v2-compatible",
    })
    descriptor = f"{value['identity']['production_descriptor_id']}-{candidate_id}"
    value["identity"].update({
        "candidate_spec_sha256": spec_digest(spec),
        "candidate_descriptor_id": descriptor,
        "descriptor_compatibility": "research-build-semantic-override-not-v2-compatible",
    })
    for item in value["repeats"]:
        item["candidate"]["descriptor_id"] = descriptor


class Phase54CiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema, format_checker=FormatChecker())

    def assert_valid(self, value: dict[str, object]) -> None:
        errors = sorted(self.validator.iter_errors(value), key=lambda error: list(error.path))
        self.assertEqual(errors, [], "; ".join(error.message for error in errors))

    def assert_invalid(self, value: dict[str, object]) -> None:
        self.assertTrue(list(self.validator.iter_errors(value)))

    def test_one_and_three_repeat_reports_validate_for_both_targets(self) -> None:
        for target in ("gfx1030", "gfx1201"):
            for repeat_count in (1, 3):
                with self.subTest(target=target, repeat_count=repeat_count):
                    self.assert_valid(report(repeat_count, target))
        research = report()
        select_research_candidate(research)
        self.assert_valid(research)

    def test_kq_and_vo_transform_candidate_identities_are_closed(self) -> None:
        candidates = (
            (
                "phase54-kq-transpose16x16-all-full-v1",
                "transpose16x16-all-full",
            ),
            (
                "phase54-vo-transpose16x16-layer19-v1",
                "transpose16x16-v-layer19-output-inverse",
            ),
            (
                "phase54-vo-transpose16x16-layers19-31-v1",
                "transpose16x16-v-layers19-31-output-inverse",
            ),
        )
        for candidate_id, transform in candidates:
            with self.subTest(candidate_id=candidate_id):
                value = report()
                select_transform_candidate(value, candidate_id, transform)
                self.assert_valid(value)
                invalid = copy.deepcopy(value)
                invalid["candidate_spec"]["transform"] = "none"
                self.assert_invalid(invalid)
                invalid = copy.deepcopy(value)
                invalid["candidate_spec"]["v_recipe"] = "ceil"
                self.assert_invalid(invalid)

    def test_two_repeats_mismatched_lengths_and_numbering_are_rejected(self) -> None:
        invalid = report()
        invalid["repeat_count"] = 2
        invalid["repeats"] = [copy.deepcopy(invalid["repeats"][0]) for _ in range(2)]
        self.assert_invalid(invalid)

        for declared, actual in ((1, 3), (3, 1)):
            with self.subTest(declared=declared, actual=actual):
                mismatch = report(declared)
                prototype = mismatch["repeats"][0]
                mismatch["repeats"] = [copy.deepcopy(prototype) for _ in range(actual)]
                for index, item in enumerate(mismatch["repeats"], 1):
                    item["repeat"] = index
                self.assert_invalid(mismatch)

        numbering = report(3)
        numbering["repeats"][1]["repeat"] = 1
        self.assert_invalid(numbering)

    def test_false_release_cleanup_and_execution_claims_are_rejected(self) -> None:
        for field in (
            "fp16_released_before_production_control",
            "production_control_released_before_candidate",
            "candidate_released_before_mxfp8",
            "mxfp8_released_after_repeat",
        ):
            invalid = report()
            invalid["repeats"][0][field] = False
            self.assert_invalid(invalid)

        for field, value in (("retryable", 1), ("durable", 1), ("poisoned", True), ("terminal_zero", False)):
            invalid = report()
            invalid["cleanup"][field] = value
            self.assert_invalid(invalid)

        for field, value in (("fallback_used", True), ("all_dispatches_hip", False), ("finite", False)):
            invalid = report()
            invalid["repeats"][0]["candidate"]["aggregate"][field] = value
            self.assert_invalid(invalid)

    def test_bad_digests_nonfinite_numbers_and_unknown_fields_are_rejected(self) -> None:
        for field in ("candidate_spec_sha256", "binary_sha256", "dataset_sha256"):
            invalid = report()
            invalid["identity"][field] = "sha256:not-a-digest"
            self.assert_invalid(invalid)

        for value in (math.nan, math.inf, -math.inf):
            invalid = report()
            invalid["repeats"][0]["candidate"]["aggregate"]["kld_p99"] = value
            self.assert_invalid(invalid)

        for owner in (
            lambda value: value,
            lambda value: value["candidate_spec"],
            lambda value: value["repeats"][0]["candidate"]["aggregate"],
            lambda value: value["repeats"][0]["candidate"]["cases"][0]["prefill"],
        ):
            invalid = report()
            owner(invalid)["unknown"] = True
            self.assert_invalid(invalid)

    def test_false_v2_identity_and_invalid_candidate_specs_are_rejected(self) -> None:
        false_v2 = report()
        select_research_candidate(false_v2)
        false_v2["identity"]["candidate_descriptor_id"] = "kv-fp8-e5-block16-v2"
        false_v2["repeats"][0]["candidate"]["descriptor_id"] = "kv-fp8-e5-block16-v2"
        self.assert_invalid(false_v2)

        mutations = (
            lambda spec: spec.__setitem__("candidate_id", "phase54-k-floor-v-ceil-v1"),
            lambda spec: spec.__setitem__("scale_selector", "standard-mx-floor-power-v1"),
            lambda spec: spec.__setitem__("rounding", "stochastic"),
            lambda spec: spec.__setitem__("calibration_digest", digest("9")),
            lambda spec: spec.pop("v_recipe"),
        )
        for mutate in mutations:
            invalid = report()
            mutate(invalid["candidate_spec"])
            self.assert_invalid(invalid)

    def test_target_order_session_and_locator_contracts_are_exact(self) -> None:
        divergence = report()
        divergence["repeats"][0]["candidate"]["aggregate"]["first_top1_divergence"] = {
            "case_id": "b255",
            "row": "prefill",
            "measured_row_index": 0,
            "reference_top1": 42,
            "observed_top1": 43,
        }
        self.assert_valid(divergence)

        invalid = report(target="gfx1030")
        invalid["encoding"] = "kv-fp8-e4-block16"
        self.assert_invalid(invalid)

        invalid = report()
        invalid["repeats"][0]["completely_sequential_order"] = [
            "fp16", "candidate-block16", "production-control-block16", "mxfp8"
        ]
        self.assert_invalid(invalid)

        invalid = report()
        invalid["session_scope"] = "multiple-processes"
        self.assert_invalid(invalid)

        invalid = report()
        del invalid["repeats"][0]["candidate"]["aggregate"]["maximum_logit_delta"]["measured_row_index"]
        self.assert_invalid(invalid)

    def test_schema_is_registered_with_manifest_validation(self) -> None:
        self.assertIn(SCHEMA_PATH.relative_to(ROOT).as_posix(), validate_json_manifests.PHASE54_SCHEMA_FILES)


if __name__ == "__main__":
    unittest.main()
