#!/usr/bin/env python3
"""Fail-closed host contracts for Phase 54 Qwen KV attribution reports."""

from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "ci/schema/phase54-qwen35-kv-attribution-research-v1.schema.json"
REPORT_DIR = Path(
    "/home/homelab1/.local/share/sllm-evidence/phase54/gfx1030/attribution-matrix-v1"
)
DATASET_SHA256 = "sha256:a2252d882ffd7e1fbb546d86b2b573bd2410467382c7da874f4fbd3dc8adc77d"
MODEL_FINGERPRINT = "sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae"
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
CASES = (
    ("b001", 1, False),
    ("b015", 15, False),
    ("b016", 16, False),
    ("b017", 17, False),
    ("b255", 255, True),
    ("b256", 256, True),
    ("b257", 257, True),
    ("b511", 511, True),
    ("b512", 512, True),
    ("b513", 513, True),
)


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(character: str) -> str:
    return "sha256:" + character * 64


def row() -> dict[str, object]:
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


def comparison(mode: str) -> dict[str, object]:
    return {
        "reference_encoding": "fp16-state/off",
        "observed_encoding": f"fp16-state/{mode}",
        "aggregate": {
            "selected_count": 20,
            "metric_sample_counts": {
                "kld": 20,
                "top1": 20,
                "nll": 10,
                "task": 10,
                "long-context": 12,
            },
            "kld_p99": 0.001,
            "top1_agreement": 1.0,
            "reference_nll": 2.0,
            "observed_nll": 2.001,
            "nll_delta": 0.001,
            "task_score_delta": 0.0,
            "long_context_score_delta": 0.0,
            "first_top1_divergence": None,
            "maximum_logit_delta": {
                "case_id": "b001",
                "row": "prefill",
                "measured_row_index": 0,
                "logit_index": 17,
                "max_abs_logit_delta": 0.125,
            },
            "finite": True,
            "hip_dispatches": 8786,
            "fallback_used": False,
            "all_dispatches_hip": True,
        },
        "cases": [
            {
                "case_id": case_id,
                "length": length,
                "long": long,
                "prefill_nll": {"reference": 2.0, "observed": 2.001, "delta": 0.001},
                "prefill": row(),
                "decode": row(),
            }
            for case_id, length, long in CASES
        ],
    }


def report(mode: str = "key-only", layer: int = 3, repeat_count: int = 1) -> dict[str, object]:
    return {
        "$schema": "https://sllm.dev/schema/phase54-qwen35-kv-attribution-research-v1.schema.json",
        "schema_version": "sllm-phase54-qwen35-kv-attribution-research-v1",
        "state": "PASS",
        "research_only": True,
        "identity": {
            "dataset_sha256": DATASET_SHA256,
            "model_lock_fingerprint": MODEL_FINGERPRINT,
            "model_lock_sha256": digest("a"),
            "derived_lock_fingerprint": digest("b"),
            "derived_lock_sha256": digest("c"),
            "binary_sha256": digest("d"),
        },
        "target": "gfx1030",
        "device_index": 0,
        "layer": layer,
        "semantics": "fp16-state/block16-roundtrip",
        "audit_semantics_verified": True,
        "reference_mode": "off",
        "intervention_mode": mode,
        "kv_state": "fp16-state",
        "session_scope": "single-process-single-hip-execution-session",
        "sequential_residents": True,
        "repeats": [
            {
                "repeat": index,
                "order": ["fp16-state/off", "fp16-state/intervention"],
                "reference_released_before_intervention": True,
                "intervention_released_after_repeat": True,
                "comparison": comparison(mode),
            }
            for index in range(1, repeat_count + 1)
        ],
        "cleanup": {
            "retryable": 0,
            "durable": 0,
            "poisoned": False,
            "terminal_zero": True,
        },
    }


class Phase54AttributionContractTests(unittest.TestCase):
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

    def test_inline_fixture_covers_all_modes_layers_and_repeat_counts(self) -> None:
        for mode in ("key-only", "value-only", "key-and-value"):
            for layer in LAYERS:
                for repeat_count in (1, 3):
                    with self.subTest(mode=mode, layer=layer, repeat_count=repeat_count):
                        self.assert_valid(report(mode, layer, repeat_count))

    def test_external_attribution_reports_are_optional_additional_validation(self) -> None:
        paths = sorted(REPORT_DIR.glob("*.json"))
        if not paths:
            return
        self.assertEqual(len(paths), 24)
        observed_modes = set()
        observed_layers = set()
        for path in paths:
            value = load(path)
            self.assert_valid(value)
            observed_modes.add(value["intervention_mode"])
            observed_layers.add(value["layer"])
        self.assertEqual(observed_modes, {"key-only", "value-only", "key-and-value"})
        self.assertEqual(observed_layers, set(LAYERS))

    def test_exact_scope_order_release_execution_and_cleanup_are_required(self) -> None:
        for field, replacement in (
            ("target", "gfx1201"),
            ("layer", 4),
            ("semantics", "other"),
            ("audit_semantics_verified", False),
            ("reference_mode", "key-only"),
            ("kv_state", "kv-fp8"),
            ("session_scope", "multi-process"),
            ("sequential_residents", False),
        ):
            invalid = report()
            invalid[field] = replacement
            self.assert_invalid(invalid)

        invalid = report()
        invalid["repeats"][0]["order"] = ["fp16-state/intervention", "fp16-state/off"]
        self.assert_invalid(invalid)
        invalid = report()
        invalid["repeats"][0]["reference_released_before_intervention"] = False
        self.assert_invalid(invalid)
        invalid = report()
        invalid["repeats"][0]["intervention_released_after_repeat"] = False
        self.assert_invalid(invalid)

        for field, replacement in (
            ("fallback_used", True),
            ("all_dispatches_hip", False),
            ("finite", False),
            ("hip_dispatches", 0),
        ):
            invalid = report()
            invalid["repeats"][0]["comparison"]["aggregate"][field] = replacement
            self.assert_invalid(invalid)
        for field, replacement in (
            ("retryable", 1),
            ("durable", 1),
            ("poisoned", True),
            ("terminal_zero", False),
        ):
            invalid = report()
            invalid["cleanup"][field] = replacement
            self.assert_invalid(invalid)

    def test_modes_metrics_hashes_and_shape_are_fail_closed(self) -> None:
        invalid = report("key-only")
        invalid["repeats"][0]["comparison"]["observed_encoding"] = "fp16-state/value-only"
        self.assert_invalid(invalid)
        invalid = report("key-only")
        invalid["intervention_mode"] = "off"
        self.assert_invalid(invalid)

        for field in ("model_lock_sha256", "derived_lock_fingerprint", "binary_sha256"):
            invalid = report()
            invalid["identity"][field] = "sha256:not-a-digest"
            self.assert_invalid(invalid)
        invalid = report()
        invalid["identity"]["dataset_sha256"] = digest("e")
        self.assert_invalid(invalid)
        invalid = report()
        invalid["identity"]["model_lock_fingerprint"] = digest("f")
        self.assert_invalid(invalid)

        for value in (math.nan, math.inf, -math.inf):
            invalid = report()
            invalid["repeats"][0]["comparison"]["aggregate"]["kld_p99"] = value
            self.assert_invalid(invalid)
        invalid = report()
        invalid["repeats"][0]["comparison"]["cases"].pop()
        self.assert_invalid(invalid)
        invalid = report()
        invalid["repeats"][0]["comparison"]["cases"][1]["length"] = 16
        self.assert_invalid(invalid)
        invalid = report()
        invalid["repeats"][0]["comparison"]["cases"][0], invalid["repeats"][0]["comparison"]["cases"][1] = (
            invalid["repeats"][0]["comparison"]["cases"][1],
            invalid["repeats"][0]["comparison"]["cases"][0],
        )
        self.assert_invalid(invalid)
        invalid = report()
        invalid["unknown"] = True
        self.assert_invalid(invalid)

        for repeat_count, actual in ((1, 3), (3, 1)):
            invalid = report(repeat_count)
            prototype = invalid["repeats"][0]
            invalid["repeats"] = [copy.deepcopy(prototype) for _ in range(actual)]
            for index, item in enumerate(invalid["repeats"], 1):
                item["repeat"] = index
            self.assert_invalid(invalid)
        invalid = report(repeat_count=3)
        invalid["repeats"][1]["repeat"] = 1
        self.assert_invalid(invalid)

    def test_schema_is_registered_with_manifest_validation(self) -> None:
        sys.path.insert(0, str(ROOT / "ci/tools"))
        import validate_json_manifests  # noqa: PLC0415

        self.assertIn(
            SCHEMA_PATH.relative_to(ROOT).as_posix(),
            validate_json_manifests.PHASE54_SCHEMA_FILES,
        )


if __name__ == "__main__":
    unittest.main()
