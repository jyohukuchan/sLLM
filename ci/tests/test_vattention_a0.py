from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "ci/tools/run_vattention_a0.py"
SPEC = importlib.util.spec_from_file_location("run_vattention_a0", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def valid_probe(target: str = "gfx1030") -> dict[str, object]:
    device = MODULE.CANONICAL[target]
    page = 2 * 1024 * 1024
    logical = 16 * 4096 * 2048
    free_before_qwen = 30_000_000_000
    free_after_reserve = free_before_qwen
    observed = 16 * page
    free_after_create = free_after_reserve - observed
    return {
        "protocol": MODULE.PROTOCOL,
        "state": "PASS",
        "device": {
            "logical_index": 0,
            "product": device["product"],
            "target": target,
            "bdf": device["bdf"],
            "vmm_supported": True,
        },
        "granularity": {
            "minimum_bytes": 4096,
            "recommended_bytes": page,
            "selected_physical_page_bytes": page,
        },
        "primitive": {
            "reserved_bytes": 3 * page,
            "mapped_pages": 3,
            "contiguous_kernel_oracle": True,
            "remap_oracle": True,
            "event_synchronized_before_unmap": True,
            "nonaligned_byte_offset": 37,
        },
        "qwen_shape": {
            "model": "Qwen/Qwen3.5-4B",
            "full_attention_layers": 8,
            "regions": 16,
            "kv_heads": 4,
            "head_dim": 256,
            "element_bytes": 2,
            "bytes_per_token_per_region": 2048,
            "logical_token_capacity": 4096,
            "tokens_per_physical_page": 1024,
            "logical_reserved_bytes": logical,
            "requested_physical_bytes": observed,
            "observed_physical_commit_bytes": observed,
            "virtual_reserve_physical_delta_bytes": 0,
            "activated_pages_per_step": 16,
            "boundary_tokens": [1023, 1024, 1025, 37],
        },
        "latency_us": {
            "warmup_iterations": 5,
            "measured_iterations": 101,
            "activate_p50": 500.0,
            "activate_p95": 600.0,
            "create_p50": 100.0,
            "create_p95": 150.0,
            "map_p50": 200.0,
            "map_p95": 250.0,
            "set_access_p50": 200.0,
            "set_access_p95": 220.0,
            "deactivate_p50": 700.0,
            "deactivate_p95": 900.0,
            "unmap_p50": 300.0,
            "unmap_p95": 400.0,
            "release_p50": 400.0,
            "release_p95": 500.0,
        },
        "memory_info": {
            "total_bytes": 32_000_000_000,
            "free_before_bytes": 31_000_000_000,
            "free_after_primitive_reserve_bytes": 31_000_000_000,
            "free_before_qwen_reserve_bytes": free_before_qwen,
            "free_after_qwen_reserve_bytes": free_after_reserve,
            "free_after_first_create_bytes": free_after_create,
            "free_after_first_map_bytes": free_after_create,
            "free_after_cleanup_bytes": free_before_qwen,
        },
        "fallback_used": False,
        "cleanup_complete": True,
    }


class VAttentionA0ContractTests(unittest.TestCase):
    def test_accepts_canonical_probe(self) -> None:
        self.assertEqual(MODULE.validate_probe(valid_probe(), "gfx1030")["state"], "PASS")

    def test_rejects_fallback(self) -> None:
        probe = valid_probe()
        probe["fallback_used"] = True
        with self.assertRaises(MODULE.A0Error):
            MODULE.validate_probe(probe, "gfx1030")

    def test_rejects_missing_boundary(self) -> None:
        probe = valid_probe()
        probe["qwen_shape"]["boundary_tokens"] = [1023, 1024, 37]
        with self.assertRaises(MODULE.A0Error):
            MODULE.validate_probe(probe, "gfx1030")

    def test_rejects_non_sparse_commit(self) -> None:
        probe = valid_probe()
        probe["qwen_shape"]["observed_physical_commit_bytes"] = probe["qwen_shape"]["logical_reserved_bytes"]
        with self.assertRaises(MODULE.A0Error):
            MODULE.validate_probe(probe, "gfx1030")

    def test_rejects_cleanup_shortfall(self) -> None:
        probe = valid_probe()
        probe["memory_info"]["free_after_cleanup_bytes"] -= 4 * 1024 * 1024
        with self.assertRaises(MODULE.A0Error):
            MODULE.validate_probe(probe, "gfx1030")

    def test_rejects_target_substitution(self) -> None:
        probe = valid_probe()
        probe["device"]["target"] = "gfx1201"
        with self.assertRaises(MODULE.A0Error):
            MODULE.validate_probe(probe, "gfx1030")

    def test_amd_smi_mapping(self) -> None:
        text = """GPU: 0
    BDF: 0000:03:00.0
    HIP_ID: 1
    HIP_UUID: GPU-76a08c022586fed6

GPU: 1
    BDF: 0000:07:00.0
    HIP_ID: 2
    HIP_UUID: GPU-a8e9ddefa2d60f55
"""
        mapping = MODULE.validate_canonical_mapping(text, "gfx1201")
        self.assertEqual(mapping["physical_hip_index"], 2)

    def test_amd_smi_mapping_rejects_bdf_drift(self) -> None:
        text = """GPU: 0
    BDF: 0000:43:00.0
    HIP_ID: 1
    HIP_UUID: GPU-76a08c022586fed6
"""
        with self.assertRaises(MODULE.A0Error):
            MODULE.validate_canonical_mapping(text, "gfx1030")

    def test_health_accepts_zero_ecc(self) -> None:
        metric = {
            "gpu_data": [{
                "gpu": 0,
                "temperature": {
                    "edge": {"value": 31, "unit": "C"},
                    "hotspot": {"value": 34, "unit": "C"},
                    "mem": {"value": 33, "unit": "C"},
                },
                "ecc": {
                    "total_uncorrectable_count": 0,
                    "total_deferred_count": 0,
                    "cache_uncorrectable_count": 0,
                },
                "power": {
                    "socket_power": {"value": 12, "unit": "W"},
                    "throttle_status": "UNTHROTTLED",
                },
                "usage": {"gfx_activity": {"value": 0, "unit": "%"}},
            }],
        }
        self.assertEqual(MODULE.validate_health(metric, 0)["ecc"]["total_uncorrectable_count"], 0)

    def test_health_rejects_uncorrectable_ecc(self) -> None:
        metric = {
            "gpu_data": [{
                "gpu": 0,
                "temperature": {
                    "edge": {"value": 31, "unit": "C"},
                    "hotspot": {"value": 34, "unit": "C"},
                    "mem": {"value": 33, "unit": "C"},
                },
                "ecc": {
                    "total_uncorrectable_count": 1,
                    "total_deferred_count": 0,
                    "cache_uncorrectable_count": 0,
                },
                "power": {
                    "socket_power": {"value": 12, "unit": "W"},
                    "throttle_status": "UNTHROTTLED",
                },
                "usage": {"gfx_activity": {"value": 0, "unit": "%"}},
            }],
        }
        with self.assertRaises(MODULE.A0Error):
            MODULE.validate_health(metric, 0)


if __name__ == "__main__":
    unittest.main()
