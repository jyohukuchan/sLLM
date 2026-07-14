from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/prefill_validation/aq4_p2_resource_observer.py"
SPEC = importlib.util.spec_from_file_location("aq4_p2_resource_observer", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def observation() -> dict:
    samples = [
        {"monotonic_ms": 1.0, "vram_used_bytes": 10, "workspace_bytes": 3, "power_watts": 5.0, "temperature_c": 40.0, "process_snapshot": []},
        {"monotonic_ms": 2.0, "vram_used_bytes": 20, "workspace_bytes": 4, "power_watts": 6.0, "temperature_c": 41.0, "process_snapshot": []},
    ]
    return {
        "schema_version": "ullm.aq4_p2_resource_observation.v1",
        "case_id": "case-1",
        "case_sha256": "a" * 64,
        "device_id": "cpu-reference",
        "observer": {
            "argv_sha256": "b" * 64,
            "shell": False,
            "tool": "cpu-rss-observer",
            "sample_period_ms": 10.0,
            "target_process_name": "driver",
            "status": "complete",
        },
        "samples": samples,
        "peak": {"vram_used_bytes": 20, "workspace_bytes": 4, "power_watts": 6.0, "temperature_c": 41.0, "sample_index": 1},
    }


class Aq4P2ResourceObserverTests(unittest.TestCase):
    def test_peak_is_recomputed_from_samples(self) -> None:
        value = observation()
        MODULE.validate(value, expected_case_id="case-1", expected_case_sha256="a" * 64)
        tampered = copy.deepcopy(value)
        tampered["peak"]["workspace_bytes"] = 3
        with self.assertRaises(MODULE.ObserverError):
            MODULE.validate(tampered)

    def test_missing_samples_and_shell_observer_are_rejected(self) -> None:
        missing = observation()
        missing["samples"] = []
        with self.assertRaises(MODULE.ObserverError):
            MODULE.validate(missing)
        shell = observation()
        shell["observer"]["shell"] = True
        with self.assertRaises(MODULE.ObserverError):
            MODULE.validate(shell)


if __name__ == "__main__":
    unittest.main()
