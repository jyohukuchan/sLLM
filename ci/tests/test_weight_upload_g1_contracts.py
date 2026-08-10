from __future__ import annotations

import copy
import unittest

from jsonschema import ValidationError

from ci.tools import validate_weight_upload_g1_contracts as contracts


def report() -> dict[str, object]:
    return {
        "schema_version": "weight-upload-g1-report-v1",
        "state": "PASS",
        "target": "gfx1030",
        "device_index": 0,
        "lock_fingerprint": "sha256:32265444b7cdd2a00e4e4e3e6aa8375a05acf6cddfcb9ffc348f54f67a7cd935",
        "plan_digest": "sha256:0820227fdc4129e5ff100e0aa87db7663d75703c9ba723bc4adc950a3af6ab66",
        "tensor_name": "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "tensor_dtype": "BF16",
        "tensor_size_bytes": 20 * 1024 * 1024,
        "source_file": "model.safetensors-00002-of-00002.safetensors",
        "source_file_sha256": "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
        "source_range": [42435872, 63407392],
        "destination_offset": 7,
        "peak_host_staging_bytes": 16 * 1024 * 1024,
        "scope": {
            "selected_backend": "hip",
            "fallback_allowed": False,
            "fallback_used": False,
            "cpu_fallback_used": False,
            "gpu_execution": True,
            "model_cache_used": True,
            "weight_payload_used": True,
            "model_execution": False,
            "semantic_op_used": False,
            "kernel_dispatch_count": 0,
            "network_used": False,
        },
        "counts": {"allocations": 1, "chunks": 2, "h2d_transfers": 2, "d2h_transfers": 2},
        "chunks": [
            {"order": 0, "tensor_offset": 0, "source_offset": 42435872, "destination_offset": 7, "size_bytes": 16 * 1024 * 1024, "h2d_state": "success", "d2h_state": "success", "exact_match": True},
            {"order": 1, "tensor_offset": 16 * 1024 * 1024, "source_offset": 59213088, "destination_offset": 7 + 16 * 1024 * 1024, "size_bytes": 4 * 1024 * 1024, "h2d_state": "success", "d2h_state": "success", "exact_match": True},
        ],
        "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0},
    }


class WeightUploadG1Contracts(unittest.TestCase):
    def test_static_matrix_schema_and_bridge_contracts(self) -> None:
        contracts.validate()
        contracts.validate_report(report())

    def test_report_rejects_fallback_and_open_objects(self) -> None:
        fallback = report()
        fallback["scope"]["fallback_used"] = True  # type: ignore[index]
        with self.assertRaises(ValidationError):
            contracts.validate_report(fallback)
        extra = report()
        extra["unexpected"] = True
        with self.assertRaises(ValidationError):
            contracts.validate_report(extra)

    def test_report_rejects_noncontiguous_source_and_destination(self) -> None:
        wrong_source = copy.deepcopy(report())
        wrong_source["chunks"][1]["source_offset"] += 1  # type: ignore[index]
        with self.assertRaises(ValueError):
            contracts.validate_report(wrong_source)
        wrong_destination = copy.deepcopy(report())
        wrong_destination["chunks"][1]["destination_offset"] += 1  # type: ignore[index]
        with self.assertRaises((ValidationError, ValueError)):
            contracts.validate_report(wrong_destination)


if __name__ == "__main__":
    unittest.main()
