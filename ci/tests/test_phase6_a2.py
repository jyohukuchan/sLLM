#!/usr/bin/env python3
"""Focused negative tests for the Phase 6 A2 contract."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from common import ContractError, read_json  # noqa: E402
from validate_phase6_a2 import CONTRACT_PATH, POLICY_PATH, validate_contract  # noqa: E402


class Phase6A2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = read_json(ROOT / CONTRACT_PATH)
        cls.policy = read_json(ROOT / POLICY_PATH)

    def reject(self, mutate) -> None:
        contract = copy.deepcopy(self.contract)
        mutate(contract)
        with self.assertRaises(ContractError):
            validate_contract(contract, self.policy, repo=ROOT)

    def test_checked_in_contract_passes(self) -> None:
        validate_contract(copy.deepcopy(self.contract), self.policy, repo=ROOT)

    def test_normative_pin_cannot_follow_current(self) -> None:
        self.reject(lambda value: value["openai_drift"]["normative"].update(openapi_commit=value["openai_drift"]["current_observation"]["openapi_commit"]))

    def test_wire_drift_cannot_be_silently_ignored(self) -> None:
        self.reject(lambda value: value["openai_drift"]["stream_changes"].append("changed"))

    def test_facts_only_unit_cannot_gain_destination(self) -> None:
        def mutate(value):
            next(unit for unit in value["llama_reuse"]["units"] if unit["reuse_mode"] == "facts-only")["planned_local"] = "src/copied.rs"
        self.reject(mutate)

    def test_engine_revision_drift_is_rejected(self) -> None:
        self.reject(lambda value: value["facts_only_readers"][0].update(commit="0" * 40))

    def test_dependency_count_drift_is_rejected(self) -> None:
        self.reject(lambda value: value["dependency_policy"]["counts"].update(edges=0))


if __name__ == "__main__":
    unittest.main()
