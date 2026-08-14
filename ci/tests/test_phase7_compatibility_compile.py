#!/usr/bin/env python3
"""Contract tests for Phase 7 compatibility compile reports."""

from __future__ import annotations

import copy
import unittest

from ci.tools import run_phase7_compatibility_compile as compile_row


class Phase7CompatibilityCompileTests(unittest.TestCase):
    def _report(self) -> dict:
        target = "gfx1201"
        return {
            "schema_version": "phase7-compatibility-compile-report-v1",
            "state": "PASS",
            "claim": {"compile_only": True, "runtime_verified": False, "numerics_verified": False, "performance_verified": False},
            "target": target,
            "candidate": {"commit": "1" * 40, "tree": "2" * 40, "immutable": True},
            "toolchain": {"rocm_root": "/opt/rocm", "rocm_release": "7.14.0", "compiler": "/opt/rocm/bin/amdclang++", "compiler_version": "AMD clang 23.0.0git", "code_object": "V6", "wave_size": 32},
            "artifact": {"device_sha256": "3" * 64, "device_bytes": 17, "bundle_ids": [f"hipv4-amdgcn-amd-amdhsa--{target}", "host-x86_64-unknown-linux-gnu-"], "metadata_target": target, "retained": False},
            "execution": {"started_at": "2026-08-14T00:00:00Z", "finished_at": "2026-08-14T00:00:01Z", "duration_seconds": 1.0, "network_isolated": True, "commands": [{"id": name, "argv_sha256": str(index + 4) * 64, "exit_code": 0} for index, name in enumerate(("compile", "link", "extract-fatbin", "list-bundles", "extract-device"))], "cleanup": "PASS"},
        }

    def test_valid_report_contract(self) -> None:
        compile_row._validate_report(self._report())

    def test_runtime_claim_and_target_mismatch_fail(self) -> None:
        changed = copy.deepcopy(self._report())
        changed["claim"]["runtime_verified"] = True
        with self.assertRaises(Exception):
            compile_row._validate_report(changed)
        changed = copy.deepcopy(self._report())
        changed["artifact"]["metadata_target"] = "gfx1030"
        with self.assertRaises(Exception):
            compile_row._validate_report(changed)


if __name__ == "__main__":
    unittest.main()
