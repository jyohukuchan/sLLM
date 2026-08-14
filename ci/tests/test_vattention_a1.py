from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "ci/tools"
sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "run_vattention_a1.py"
SPEC = importlib.util.spec_from_file_location("run_vattention_a1", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def valid_probe(directory: Path, target: str = "gfx1030") -> dict[str, object]:
    page = 2 * 1024 * 1024
    device = MODULE.a0.CANONICAL[target]
    rows: list[dict[str, object]] = []
    for mode_id, mode in enumerate(MODULE.MODES):
        for q in MODULE.QUERY_LENGTHS:
            for k in MODULE.KV_LENGTHS:
                output = MODULE._float_to_bf16(MODULE._reference(q, k)).astype("<u2")
                filename = f"{mode}-q{q}-k{k}.bf16"
                (directory / filename).write_bytes(output.tobytes())
                blocks = (k + 255) // 256
                used_plane = k * 4 * 256 * 2
                if mode == "contiguous":
                    committed = 2 * 4096 * 4 * 256 * 2
                elif mode == "vattention":
                    committed = 2 * ((used_plane + page - 1) // page) * page
                else:
                    committed = 2 * blocks * 256 * 4 * 256 * 2
                rows.append({
                    "mode": mode,
                    "mode_id": mode_id,
                    "query_length": q,
                    "kv_length": k,
                    "setup_us": 100.0,
                    "grow_us": 40.0 if mode == "vattention" else 0.0,
                    "kernel_p50_us": 200.0,
                    "kernel_p95_us": 220.0,
                    "logical_bytes": 2 * 4096 * 4 * 256 * 2,
                    "committed_bytes": committed,
                    "metadata_bytes": blocks * 4 if mode == "paged" else 0,
                    "observed_vram_delta_bytes": committed,
                    "output_file": filename,
                    "nonidentity_block_table": mode == "paged" and blocks > 1,
                })
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
        "shape": {
            "q_heads": 16,
            "kv_heads": 4,
            "head_dim": 256,
            "logical_capacity": 4096,
            "paged_block_tokens": 256,
            "query_lengths": MODULE.QUERY_LENGTHS,
            "kv_lengths": MODULE.KV_LENGTHS,
        },
        "algorithm": {
            "class": "FA2-style tiled online-softmax proxy",
            "kernel_symbol": MODULE.KERNEL_SYMBOL,
            "contiguous_and_vattention_same_kernel": True,
            "kv_layout": "token-major",
            "causal_alignment": "bottom-right",
        },
        "vmm": {
            "minimum_page_bytes": 4096,
            "recommended_page_bytes": page,
            "selected_page_bytes": page,
        },
        "warmup_iterations": 3,
        "measured_iterations": 9,
        "results": rows,
        "fallback_used": False,
        "cleanup_complete": True,
    }


class VAttentionA1ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_accepts_complete_matrix_and_numpy_oracle(self) -> None:
        validated = MODULE.validate_probe(valid_probe(self.directory), "gfx1030", self.directory)
        self.assertTrue(validated["oracle"]["all_modes_numerically_equivalent"])
        self.assertEqual(len(validated["results"]), 36)

    def test_rejects_fallback(self) -> None:
        probe = valid_probe(self.directory)
        probe["fallback_used"] = True
        with self.assertRaises(MODULE.A1Error):
            MODULE.validate_probe(probe, "gfx1030", self.directory)

    def test_rejects_target_substitution(self) -> None:
        probe = valid_probe(self.directory)
        probe["device"]["target"] = "gfx1201"
        with self.assertRaises(MODULE.A1Error):
            MODULE.validate_probe(probe, "gfx1030", self.directory)

    def test_rejects_missing_boundary_case(self) -> None:
        probe = valid_probe(self.directory)
        probe["results"].pop()
        with self.assertRaises(MODULE.A1Error):
            MODULE.validate_probe(probe, "gfx1030", self.directory)

    def test_rejects_distinct_vattention_kernel_symbol(self) -> None:
        probe = valid_probe(self.directory)
        probe["algorithm"]["kernel_symbol"] = "vattention_special_kernel"
        with self.assertRaises(MODULE.A1Error):
            MODULE.validate_probe(probe, "gfx1030", self.directory)

    def test_rejects_paged_identity_table_for_multiblock_case(self) -> None:
        probe = valid_probe(self.directory)
        row = next(
            row for row in probe["results"]
            if row["mode"] == "paged" and row["kv_length"] == 257
        )
        row["nonidentity_block_table"] = False
        with self.assertRaises(MODULE.A1Error):
            MODULE.validate_probe(probe, "gfx1030", self.directory)

    def test_rejects_numerical_oracle_failure(self) -> None:
        probe = valid_probe(self.directory)
        path = self.directory / "paged-q37-k1025.bf16"
        raw = np.frombuffer(path.read_bytes(), dtype="<u2").copy()
        raw[0] = np.uint16(0x7F80)
        path.write_bytes(raw.tobytes())
        with self.assertRaises(MODULE.A1Error):
            MODULE.validate_probe(probe, "gfx1030", self.directory)

    def test_rejects_missing_raw_output(self) -> None:
        probe = valid_probe(self.directory)
        (self.directory / "contiguous-q1-k255.bf16").unlink()
        with self.assertRaises(MODULE.A1Error):
            MODULE.validate_probe(probe, "gfx1030", self.directory)

    def test_production_probe_requires_unmapped_rejection(self) -> None:
        document = {
            "protocol": "sllm-vattention-a1-production-v1",
            "state": "PASS",
            "target": "gfx1030",
            "layout": "token-major",
            "memory_kind": "virtual-contiguous",
            "boundary_tokens": [1023, 1024, 1025],
            "committed_bytes_per_plane": [2097152, 2097152, 4194304],
            "unmapped_readback_rejected": False,
            "numerical_oracle": True,
            "fallback_used": False,
            "cleanup_complete": True,
        }
        with self.assertRaises(MODULE.A1Error):
            MODULE.validate_production_probe(document, "gfx1030")


if __name__ == "__main__":
    unittest.main()
