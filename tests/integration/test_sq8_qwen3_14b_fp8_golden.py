from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "sq8_canonical_artifact.py"
FIXTURE_PATH = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "sq8"
    / "qwen3-14b-fp8-layer0-q-proj-v0.1.json"
)


def load_module():
    spec = importlib.util.spec_from_file_location("sq8_canonical_artifact", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SQ8 = load_module()


class Qwen3_14bFp8CanonicalGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        raw_model_dir = os.environ.get("ULLM_QWEN3_14B_FP8_DIR")
        required = os.environ.get("ULLM_REQUIRE_REAL_SQ8_GOLDEN") == "1"
        if not raw_model_dir:
            if required:
                raise AssertionError(
                    "ULLM_QWEN3_14B_FP8_DIR is required when "
                    "ULLM_REQUIRE_REAL_SQ8_GOLDEN=1"
                )
            raise unittest.SkipTest("ULLM_QWEN3_14B_FP8_DIR is not set")
        cls.model_dir = Path(raw_model_dir)
        if not cls.model_dir.is_dir():
            if required:
                raise AssertionError(f"model directory does not exist: {cls.model_dir}")
            raise unittest.SkipTest(f"model directory does not exist: {cls.model_dir}")

    def test_checkpoint_identity_matches_frozen_fixture(self) -> None:
        self.assertEqual(
            SQ8.sha256_file(self.model_dir / "config.json"),
            self.fixture["config_sha256"],
        )
        self.assertEqual(
            SQ8.sha256_file(self.model_dir / "model.safetensors.index.json"),
            self.fixture["index_sha256"],
        )

    def test_checkpoint_inventory_has_complete_pairs(self) -> None:
        contract = SQ8.load_source_contract(self.model_dir)
        inventory, _index = SQ8.collect_tensor_inventory(self.model_dir)
        pairs, passthrough = SQ8.pair_fp8_weights(
            inventory,
            contract.weight_block_shape,
        )

        self.assertEqual(len(inventory), 723)
        self.assertEqual(len(pairs), 280)
        self.assertEqual(len(passthrough), 163)
        self.assertEqual(
            sum(pair.weight.length for pair in pairs),
            13212057600,
        )
        self.assertEqual(
            sum(pair.scale.length for pair in pairs),
            1612800,
        )

    def test_layer0_q_projection_round_trip_matches_source_golden(self) -> None:
        weight_fixture = self.fixture["weight"]
        scale_fixture = self.fixture["scale"]
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifact"
            manifest = SQ8.build_canonical_artifact(
                self.model_dir,
                artifact_dir,
                tensor_names=[weight_fixture["name"]],
                copy_chunk_bytes=1024 * 1024,
            )
            verification = SQ8.verify_canonical_artifact(artifact_dir)
            entry = manifest["quantized_tensors"][0]
            weight_path = artifact_dir / entry["weight"]["file"]
            scale_path = artifact_dir / entry["scale"]["file"]

            self.assertTrue(verification["verified"])
            self.assertEqual(entry["weight"]["sha256"], weight_fixture["sha256"])
            self.assertEqual(entry["scale"]["sha256"], scale_fixture["sha256"])
            self.assertEqual(weight_path.stat().st_size, weight_fixture["bytes"])
            self.assertEqual(scale_path.stat().st_size, scale_fixture["bytes"])

            weight_bytes = bytearray(weight_path.read_bytes())
            scale_bytes = bytearray(scale_path.read_bytes())
            weight = torch.frombuffer(weight_bytes, dtype=torch.uint8).view(
                torch.float8_e4m3fn
            ).reshape(weight_fixture["shape"])
            scale = torch.frombuffer(scale_bytes, dtype=torch.bfloat16).reshape(
                scale_fixture["shape"]
            )
            digest = hashlib.sha256()
            block_rows, block_cols = scale_fixture["block_shape"]
            for block_row, block_col in self.fixture["reconstruction_blocks"]:
                row_start = block_row * block_rows
                col_start = block_col * block_cols
                block = (
                    weight[
                        row_start : row_start + block_rows,
                        col_start : col_start + block_cols,
                    ].float()
                    * scale[block_row, block_col].float()
                )
                digest.update(block.contiguous().numpy().astype("<f4").tobytes())

            self.assertEqual(
                digest.hexdigest(),
                self.fixture["reconstruction_f32_le_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
