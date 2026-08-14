#!/usr/bin/env python3
"""Host-only tests for the Phase 7 device observation summary contract."""

from __future__ import annotations

import copy
import unittest

from ci.tools import run_phase7_gpu_observation as observation


def stats(value: float, count: int = 3) -> dict:
    return {"median": value, "p10": value, "p90": value, "mad": 0, "min": value, "max": value, "count": count}


class Phase7ObservationTests(unittest.TestCase):
    def _summary(self) -> dict:
        metrics = {name: stats(1.0, 1 if "vram" in name else 10) for name in ("ttft_ns", "prefill_ns", "tpot_ns", "decode_token_per_s", "prefill_token_per_s", "e2e_ns", "resident_vram_bytes", "peak_vram_bytes")}
        return {
            "schema_version": "phase7-gpu-observation-v1",
            "state": "PASS",
            "profile": "daily",
            "performance_lane": "p0-observation",
            "executed_tiers": ["tier_g0", "tier_g3", "tier_g4", "tier_p1"],
            "claims": {"performance_hard_gate": False, "optimized": False, "faster": False, "compatibility_lifecycle": "experimental"},
            "candidate": {"commit": "1" * 40, "tree": "2" * 40, "immutable": True},
            "model": {"repo_id": "Qwen/Qwen3.5-4B", "resolved_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "lock_fingerprint": "sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae"},
            "rows": [{"tuple_id": "local-r9700-gfx1201-rocm714-hwe617", "target": "gfx1201", "row_id": "engine-performance-direct-4b-gfx1201-short-odd", "report_sha256": "3" * 64, "raw_sha256": "4" * 64, "metrics": metrics, "health": "PASS", "fallback": False, "cleanup": "PASS"}],
            "cleanup": "PASS",
        }

    def test_valid_observation_summary(self) -> None:
        observation._validate_summary(self._summary())

    def test_hard_gate_fallback_and_tuple_target_mismatch_fail(self) -> None:
        changed = copy.deepcopy(self._summary())
        changed["claims"]["performance_hard_gate"] = True
        with self.assertRaises(Exception):
            observation._validate_summary(changed)
        changed = copy.deepcopy(self._summary())
        changed["rows"][0]["fallback"] = True
        with self.assertRaises(Exception):
            observation._validate_summary(changed)
        changed = copy.deepcopy(self._summary())
        changed["rows"][0]["target"] = "gfx1030"
        with self.assertRaises(Exception):
            observation._validate_summary(changed)


if __name__ == "__main__":
    unittest.main()
