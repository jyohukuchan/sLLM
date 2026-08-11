import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from ci.tools.validate_execution_transfer_g1_contracts import ROOT, REPORT_SCHEMA, validate


def report() -> dict[str, object]:
    sizes = [1, 3, 17, 255, 256, 257]
    return {
        "schema_version": "execution-transfer-g1-report-v1", "state": "PASS",
        "target": "gfx1030", "device_index": 0, "max_transfer_bytes": 1_073_741_824,
        "scope": {"selected_backend": "hip", "fallback_allowed": False, "fallback_used": False,
                  "cpu_fallback_used": False, "gpu_execution": True, "model_used": False,
                  "semantic_op_used": False, "kernel_dispatch_count": 0},
        "counts": {"cases": 6, "allocations": 6, "h2d_transfers": 6, "d2h_transfers": 6},
        "cases": [{"id": f"bytes-{size}", "order": order, "offset_bytes": order * 3 + 1,
                   "size_bytes": size, "h2d_state": "success", "d2h_state": "success",
                   "exact_match": True} for order, size in enumerate(sizes)],
        "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0},
    }


class ExecutionTransferG1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(Path(REPORT_SCHEMA).read_text(encoding="utf-8"))
        cls.validator = Draft202012Validator(cls.schema)

    def test_static_contract_and_positive_report(self) -> None:
        validate()
        self.assertEqual(list(self.validator.iter_errors(report())), [])

    def test_report_rejects_fallback_wrong_order_cleanup_and_unknown_fields(self) -> None:
        mutations = []
        fallback = copy.deepcopy(report())
        fallback["scope"]["fallback_used"] = True
        mutations.append(fallback)
        order = copy.deepcopy(report())
        order["cases"][2]["size_bytes"] = 16
        mutations.append(order)
        cleanup = copy.deepcopy(report())
        cleanup["cleanup"]["retryable_cleanup"] = 1
        mutations.append(cleanup)
        extra = copy.deepcopy(report())
        extra["unknown"] = True
        mutations.append(extra)
        for mutation in mutations:
            self.assertNotEqual(list(self.validator.iter_errors(mutation)), [])


if __name__ == "__main__":
    unittest.main()
