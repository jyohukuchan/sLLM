#!/usr/bin/env python3
"""Focused unit tests for the frozen SQ8 v0.2 consumer evaluator."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = ROOT / "tools" / "evaluate-sq8-gate-v0.2.py"
PREPARER_PATH = ROOT / "tools" / "prepare-sq8-gate-v0.2-capture.py"
GATE_PATH = ROOT / "docs" / "plans" / "sq8-numerical-gate-v0.2-relative-to-fp32-reference.json"

SPEC = importlib.util.spec_from_file_location("sq8_gate_v02_evaluator", EVALUATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVALUATOR
SPEC.loader.exec_module(EVALUATOR)

PREPARER_SPEC = importlib.util.spec_from_file_location("sq8_gate_v02_preparer", PREPARER_PATH)
assert PREPARER_SPEC is not None and PREPARER_SPEC.loader is not None
PREPARER = importlib.util.module_from_spec(PREPARER_SPEC)
sys.modules[PREPARER_SPEC.name] = PREPARER
PREPARER_SPEC.loader.exec_module(PREPARER)


class Sq8GateV02EvaluatorTests(unittest.TestCase):
    def test_frozen_gate_and_probe_selection(self) -> None:
        gate, digest = EVALUATOR.load_frozen_gate(GATE_PATH)
        self.assertEqual(digest, EVALUATOR.EXPECTED_GATE_SHA256)
        selected = EVALUATOR.hidden_probe_ids(gate)
        self.assertEqual(len(selected), 512)
        for stream in gate["corpus"]["primary_decode_streams"]:
            case_id = stream["id"]
            self.assertIn((case_id, 0), selected)
            self.assertIn((case_id, stream["forced_decode_tokens"] - 1), selected)

    def test_frozen_coverage_requires_every_m128_checkpoint(self) -> None:
        gate, _ = EVALUATOR.load_frozen_gate(GATE_PATH)
        expected = EVALUATOR.expected_position_sets(gate)
        self.assertEqual(len(expected["primary"]), 4096)
        self.assertEqual(len(expected["boundaries"]), 17)
        self.assertEqual(len(expected["prefill"]), 97)
        self.assertEqual(len(expected["layer_required"]), 626)
        self.assertIn(
            "m128_chunks_with_declared_tail:raw-p4095-g1:decode:00000",
            expected["prefill"],
        )
        self.assertIn(
            "m128_chunks_with_declared_tail:chat-p2048-g512:decode:00511",
            expected["prefill"],
        )
        capture_ids = PREPARER.frozen_capture_position_ids(gate)
        self.assertEqual(len(capture_ids), 4210)
        self.assertEqual(
            PREPARER.reference_index_qualification(gate, {"positions": []})["missing_positions"],
            4210,
        )
        self.assertEqual(len(EVALUATOR.expected_reference_case_keys(gate)), 17)
        reference_qualification = EVALUATOR.index_reference_qualification(gate, {})
        self.assertFalse(reference_qualification["complete"])
        self.assertIn("does not record", reference_qualification["reason"])

    def test_preparer_blocks_partial_reference_before_a_gpu_plan(self) -> None:
        gate, digest = EVALUATOR.load_frozen_gate(GATE_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "partial-index.json"
            output = root / "blocked-plan.json"
            index.write_text(
                json.dumps(
                    {
                        "schema_version": PREPARER.REFERENCE_INDEX_SCHEMA,
                        "frozen_gate": {"sha256": digest},
                        "positions": [],
                    }
                ),
                encoding="utf-8",
            )
            result = PREPARER.build_plan(
                SimpleNamespace(
                    gate=GATE_PATH,
                    reference=index,
                    candidate="flash2-staged-wave32",
                    role="candidate",
                    diagnostic_only=False,
                    artifact=root / "not-read-before-blocking",
                    package=root / "not-read-before-blocking",
                    output=output,
                )
            )
            self.assertEqual(result, 0)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "blocked_reference_or_capture")
            self.assertEqual(receipt["coverage"]["missing_positions"], 4210)

    def test_preparer_blocks_a_complete_index_with_incomplete_reference_receipts(self) -> None:
        gate, digest = EVALUATOR.load_frozen_gate(GATE_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "index.json"
            output = root / "blocked-plan.json"
            index.write_text(
                json.dumps(
                    {
                        "schema_version": PREPARER.REFERENCE_INDEX_SCHEMA,
                        "frozen_gate": {"sha256": digest},
                        "positions": [
                            {"id": position_id}
                            for position_id in sorted(PREPARER.frozen_capture_position_ids(gate))
                        ],
                        "reference_qualification": {"complete": False, "missing_cases": ["sequential_m1:x"]},
                    }
                ),
                encoding="utf-8",
            )
            PREPARER.build_plan(
                SimpleNamespace(
                    gate=GATE_PATH,
                    reference=index,
                    candidate="flash2-staged-wave32",
                    role="candidate",
                    diagnostic_only=False,
                    artifact=root / "not-read-before-blocking",
                    package=root / "not-read-before-blocking",
                    output=output,
                )
            )
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "blocked_reference_or_capture")
            self.assertIn("strict-F32", receipt["reason"])

    def test_capture_runtime_configuration_must_match_except_selector(self) -> None:
        identity = {
            "executable_sha256": "a" * 64,
            "device_identity": {"device_id": 0},
            "runtime_compiler_versions": {"feature": "rocm-ck-gfx1201"},
            "hip_guard_environment": {"ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL": "1"},
        }
        matching = EVALUATOR.check_matched_capture_configuration(
            [("control-0", {"identity": identity}), ("candidate-0", {"identity": dict(identity)})]
        )
        self.assertEqual(matching, [])
        different = dict(identity)
        different["hip_guard_environment"] = {"ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL": None}
        errors = EVALUATOR.check_matched_capture_configuration(
            [("control-0", {"identity": identity}), ("candidate-0", {"identity": different})]
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("hip_guard_environment", errors[0])

    def test_frozen_gate_hash_rejection(self) -> None:
        value = json.loads(GATE_PATH.read_text(encoding="utf-8"))
        value["status"] = "tampered-for-test"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gate.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(EVALUATOR.GateError):
                EVALUATOR.load_frozen_gate(path)

    def test_repeat_envelopes_follow_frozen_median_wording(self) -> None:
        upper = EVALUATOR.as_upper_gate([1.0, 2.0, 9.0], [3.0, 4.0], 1.05, 0.25)
        self.assertEqual(upper["control_median"], 2.0)
        self.assertEqual(upper["repeat_envelope"], 7.0)
        self.assertEqual(upper["threshold"], 9.35)
        self.assertTrue(upper["passed"])
        lower = EVALUATOR.as_lower_gate([0.2, 0.8, 0.9], [0.1, 0.7], 0.001)
        self.assertEqual(lower["control_median"], 0.8)
        self.assertEqual(lower["repeat_envelope"], 0.6000000000000001)
        self.assertAlmostEqual(lower["threshold"], 0.199)
        self.assertFalse(lower["passed"])

    def test_topk_uses_lower_token_id_as_tie_break(self) -> None:
        values = np.array([1.0, 4.0, 4.0, 3.0], dtype=np.float32)
        self.assertEqual(EVALUATOR.top_ids(values, 3).tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
