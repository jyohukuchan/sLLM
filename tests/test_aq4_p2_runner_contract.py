from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/run-aq4-production-p2.py"
SPEC = importlib.util.spec_from_file_location("run_aq4_production_p2", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Aq4P2RunnerContractTests(unittest.TestCase):
    def test_driver_argv_binds_case_device_and_request_width(self) -> None:
        case = {
            "case_id": "p2-smoke-full_model-cold_prefill-all_m1-n128-m1-cpu-reference-aq4_0_target",
            "prefill_requested_m": 1,
            "device": {"runtime_device_index": 0},
        }
        argv = MODULE.build_full_model_driver_argv(
            Path("driver"), Path("served.json"), Path("fixture.json"), Path("case.json"),
            Path("identity.json"), Path("preflight.json"), Path("result.json"), case,
        )
        self.assertEqual(argv[argv.index("--device-index") + 1], "0")
        self.assertEqual(argv[argv.index("--m") + 1], "1")
        self.assertEqual(argv[argv.index("--case-id") + 1], case["case_id"])

    def test_cpu_case_is_unsupported_for_hip_resident_worker(self) -> None:
        case = {"device": {"backend": "cpu"}}
        self.assertTrue(MODULE.cpu_resident_unsupported(case, {"worker": {"identity": {"device": "gfx1201"}}}))
        self.assertFalse(MODULE.cpu_resident_unsupported(case, {"worker": {"identity": {"device": "host"}}}))
        self.assertFalse(MODULE.cpu_resident_unsupported({"device": {"backend": "hip"}}, {"worker": {"identity": {"device": "gfx1201"}}}))

    def test_cpu_static_gate_does_not_require_runtime_probe(self) -> None:
        # A CPU preparation run only needs the served manifest contract.  Missing runtime fields
        # cannot accidentally turn a HIP worker into a CPU-success row.
        case = {"device": {"backend": "cpu"}}
        served = {"worker": {"identity": {"device": "gfx1201", "execution_profile": "rdna4_aq4_resident"}}}
        self.assertTrue(MODULE.cpu_resident_unsupported(case, served))


if __name__ == "__main__":
    unittest.main()
