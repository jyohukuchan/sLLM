from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/prefill_validation/aq4_p2_raw_v2_adapter.py"
SPEC = importlib.util.spec_from_file_location("aq4_p2_raw_v2_adapter", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Aq4P2RawV2AdapterBindingTests(unittest.TestCase):
    def bound_pair(self) -> tuple[dict, dict]:
        policy = {
            "status": "bound",
            "hash_binding": {"policy_sha256": None},
            "power_condition": {
                "expected_power_limit_watts": 300,
                "allowed_power_tolerance_watts": 5,
                "maximum_temperature_c": 95,
                "minimum_vram_headroom_bytes": 1,
            },
        }
        policy["hash_binding"]["policy_sha256"] = MODULE.policy_self_hash(policy)
        identity = {
            "schema_version": "ullm.aq4_production_p2_identity.v2",
            "status": "bound",
            "identity_sha256": None,
            "policy_sha256": policy["hash_binding"]["policy_sha256"],
        }
        identity["identity_sha256"] = MODULE.identity_self_hash(identity)
        return policy, identity

    def test_policy_and_identity_are_rechecked(self) -> None:
        policy, identity = self.bound_pair()
        MODULE.validate_policy_binding(policy, identity)
        tampered = copy.deepcopy(policy)
        tampered["power_condition"]["maximum_temperature_c"] += 1
        with self.assertRaises(MODULE.AdapterError):
            MODULE.validate_policy_binding(tampered, identity)

    def test_preflight_rejects_unknown_or_negative_fields(self) -> None:
        preflight = {
            "weights_bytes": 1,
            "persistent_state_bytes": 1,
            "kv_cache_bytes": 1,
            "workspace_bytes": 1,
            "temporary_bytes": 1,
            "vram_headroom_bytes": 1,
            "gpu_process_snapshot": [],
        }
        MODULE.validate_preflight(preflight)
        bad = dict(preflight)
        bad["unexpected"] = 0
        with self.assertRaises(MODULE.AdapterError):
            MODULE.validate_preflight(bad)
        bad = dict(preflight)
        bad["vram_headroom_bytes"] = -1
        with self.assertRaises(MODULE.AdapterError):
            MODULE.validate_preflight(bad)

    def test_command_uses_bound_case_device_index_and_no_shell_string(self) -> None:
        case = {
            "case_id": "p2-smoke-full_model-cold_prefill-all_m1-n128-m1-cpu-reference-aq4_0_target",
            "prefill_requested_m": 1,
            "device": {"runtime_device_index": 0},
        }
        argv = MODULE.build_driver_command(
            Path("driver"), Path("served.json"), Path("fixture.json"), Path("case.json"),
            Path("identity.json"), Path("preflight.json"), Path("result.json"), case,
        )
        self.assertIn("--device-index", argv)
        self.assertEqual(argv[argv.index("--device-index") + 1], "0")
        self.assertIn("--case-id", argv)
        self.assertEqual(argv[argv.index("--case-id") + 1], case["case_id"])
        self.assertNotIn("shell", argv)


if __name__ == "__main__":
    unittest.main()
