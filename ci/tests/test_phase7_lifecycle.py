#!/usr/bin/env python3
"""Host-only tests for the Phase 7 lifecycle profile contract."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from ci.tools import phase7_lifecycle as lifecycle


class Phase7LifecycleTests(unittest.TestCase):
    def test_checked_in_contracts_and_profile_resolution(self) -> None:
        profile, compatibility = lifecycle.validate_contracts()
        self.assertEqual(len(compatibility["tuples"]), 2)
        daily = lifecycle.resolve_profile(
            profile, event="schedule", schedule=profile["workflow"]["daily_cron"]
        )
        weekly = lifecycle.resolve_profile(
            profile, event="schedule", schedule=profile["workflow"]["weekly_cron"]
        )
        release = lifecycle.resolve_profile(
            profile, event="release", release_action="published"
        )
        self.assertEqual(daily["gpu_tuples"], lifecycle.EXPECTED_TUPLES)
        self.assertEqual(daily["compile_targets"], ["gfx1030", "gfx1201"])
        self.assertEqual(weekly["compile_targets"], lifecycle.EXPECTED_TARGETS)
        self.assertFalse(weekly["blocking"])
        self.assertTrue(release["blocking"])
        self.assertEqual(release["retention_days"], 90)

    def test_unknown_or_overridden_triggers_fail_closed(self) -> None:
        profile, _ = lifecycle.validate_contracts()
        with self.assertRaises(Exception):
            lifecycle.resolve_profile(profile, event="schedule", schedule="0 0 1 1 *")
        with self.assertRaises(Exception):
            lifecycle.resolve_profile(
                profile,
                event="schedule",
                schedule=profile["workflow"]["daily_cron"],
                requested_profile="release",
            )
        with self.assertRaises(Exception):
            lifecycle.resolve_profile(profile, event="release", release_action="created")
        with self.assertRaises(Exception):
            lifecycle.resolve_profile(profile, event="workflow_dispatch", requested_profile="unknown")

    def test_mutated_profile_schema_and_semantics_are_rejected(self) -> None:
        profile, _ = lifecycle.validate_contracts()
        schema = lifecycle._load(lifecycle.PROFILE_SCHEMA_PATH, "schema")
        changed = copy.deepcopy(profile)
        changed["claims"]["performance_hard_gate"] = True
        with self.assertRaises(Exception):
            lifecycle._validate_schema(changed, schema, "changed profile")

        changed = copy.deepcopy(profile)
        changed["profiles"][0]["gpu_tuples"] = [lifecycle.EXPECTED_TUPLES[1]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in (
                lifecycle.PROFILE_SCHEMA_PATH,
                lifecycle.COMPATIBILITY_PATH,
                lifecycle.COMPATIBILITY_SCHEMA_PATH,
                lifecycle.TUPLE_SCHEMA_PATH,
            ):
                destination = root / source.relative_to(lifecycle.ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
            destination = root / lifecycle.PROFILE_PATH.relative_to(lifecycle.ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(Exception):
                lifecycle.validate_contracts(root)


if __name__ == "__main__":
    unittest.main()
