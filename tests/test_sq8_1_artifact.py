from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
SQ8_0_MODULE_PATH = REPO_ROOT / "tools" / "sq8_canonical_artifact.py"
SQ8_1_MODULE_PATH = REPO_ROOT / "tools" / "sq8_1_artifact.py"
WEIGHT_NAME = "model.layers.0.self_attn.q_proj.weight"
SCALE_NAME = f"{WEIGHT_NAME}_scale_inv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SQ8_0 = load_module("sq8_canonical_artifact", SQ8_0_MODULE_PATH)
SQ8_1 = load_module("sq8_1_artifact", SQ8_1_MODULE_PATH)


def write_sq8_0_source(root: Path) -> Path:
    model_dir = root / "source-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "quantization_config": {
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "activation_scheme": "dynamic",
                    "weight_block_size": [128, 128],
                },
            }
        ),
        encoding="utf-8",
    )
    # Deliberately cross K=16 and K=32 boundaries so the independent SQ8_1
    # payload/scale planes exercise all physical tail cases.
    weights = (
        (torch.arange(3 * 33, dtype=torch.float32).reshape(3, 33) % 17 - 8) / 4
    ).to(torch.float8_e4m3fn)
    scales = torch.tensor([[1.0]], dtype=torch.bfloat16)
    save_file(
        {
            WEIGHT_NAME: weights,
            SCALE_NAME: scales,
            "model.embed_tokens.weight": torch.arange(4, dtype=torch.bfloat16).reshape(2, 2),
        },
        model_dir / "model.safetensors",
    )
    return model_dir


def rewrite_sq8_1_manifest(artifact: Path, mutate) -> None:
    path = artifact / SQ8_1.MANIFEST_FILE
    manifest = SQ8_1.read_json(path)
    mutate(manifest)
    manifest.pop("integrity", None)
    manifest["integrity"] = {"content_sha256": SQ8_1.artifact_content_sha256(manifest)}
    SQ8_1.write_json(path, manifest)


class Sq8_1ArtifactTests(unittest.TestCase):
    def test_pack_layout_zero_saturation_and_tail_shapes(self) -> None:
        for cols, expected_stride in ((1, 16), (15, 16), (16, 16), (17, 32), (31, 32), (32, 32), (33, 48)):
            values = [0.0] * cols + [float(index - 16) for index in range(cols)]
            tensor = SQ8_1.pack_tensor_from_values("tail", 2, cols, values)
            self.assertEqual(tensor.payload_row_stride, expected_stride)
            self.assertEqual(len(tensor.payload), 2 * expected_stride)
            self.assertEqual(len(tensor.scales_f16_le), 2 * SQ8_1.groups_per_row(cols) * 2)
            self.assertEqual(tensor.scale(0, 0), 1.0)
            self.assertEqual([tensor.code(0, col) for col in range(cols)], [0] * cols)
            self.assertEqual(tensor.code(1, 0), -127)
            self.assertNotIn(0x80, tensor.payload[:cols])
            self.assertEqual(tensor.payload[cols:expected_stride], b"\0" * (expected_stride - cols))
            SQ8_1.validate_tensor(tensor)

    def test_f16_ceiling_and_f32_contract(self) -> None:
        raw = SQ8_1.f32(1.0001)
        bits = SQ8_1.ceil_fp16(raw)
        self.assertGreaterEqual(SQ8_1.f16_bits_to_f32(bits), raw)
        self.assertEqual(bits, 0x3C01)
        self.assertEqual(SQ8_1.ceil_fp16(math.ldexp(1.0, -25)), 0x0001)
        with self.assertRaisesRegex(SQ8_1.ArtifactError, "overflow"):
            SQ8_1.ceil_fp16(65520.0)
        with self.assertRaisesRegex(SQ8_1.ArtifactError, "finite"):
            SQ8_1.pack_tensor_from_values("nonfinite", 1, 1, [math.inf])

    def test_w8a16_and_explicit_w8a8_references_match_manual_block_math(self) -> None:
        weights = [
            *[float((index % 13) - 6) / 3.0 for index in range(33)],
            *[float((index % 11) - 5) / 2.0 for index in range(33)],
        ]
        tensor = SQ8_1.pack_tensor_from_values("matvec", 2, 33, weights)
        activation = [float((index % 9) - 4) / 5.0 for index in range(33)]

        expected_w8a16 = []
        for row in range(tensor.rows):
            total = 0.0
            for block in range(tensor.groups_per_row):
                start = block * 32
                stop = min(start + 32, tensor.cols)
                partial = 0.0
                for col in range(start, stop):
                    partial += float(tensor.code(row, col)) * activation[col]
                total += partial * tensor.scale(row, block)
            expected_w8a16.append(total)
        actual_w8a16 = SQ8_1.matvec_w8a16(tensor, activation)
        self.assertEqual(actual_w8a16, expected_w8a16)

        activation_codes, activation_scales = SQ8_1.quantize_activation(activation)
        expected_w8a8 = []
        for row in range(tensor.rows):
            total = 0.0
            for block in range(tensor.groups_per_row):
                start = block * 32
                stop = min(start + 32, tensor.cols)
                dot = sum(
                    tensor.code(row, col)
                    * (activation_codes[col] - 256 if activation_codes[col] >= 128 else activation_codes[col])
                    for col in range(start, stop)
                )
                activation_scale = SQ8_1.f16_bits_to_f32(
                    int.from_bytes(activation_scales[2 * block : 2 * block + 2], "little")
                )
                total += dot * tensor.scale(row, block) * activation_scale
            expected_w8a8.append(total)
        self.assertEqual(SQ8_1.matvec_w8a8_explicit(tensor, activation), expected_w8a8)

    def test_build_read_verify_and_sq8_0_noninterference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_model = write_sq8_0_source(root)
            canonical = root / "canonical"
            SQ8_0.build_canonical_artifact(source_model, canonical, tensor_names=[WEIGHT_NAME])
            source_before = {
                path.relative_to(canonical): path.read_bytes()
                for path in canonical.rglob("*")
                if path.is_file()
            }
            output = root / "sq8_1"
            SQ8_1.build_sq8_1_artifact(canonical, output)
            verification = SQ8_1.verify_sq8_1_artifact(output)
            tensor = SQ8_1.read_sq8_1_tensor(output, WEIGHT_NAME)

            self.assertEqual(verification["format_id"], "SQ8_1")
            self.assertEqual(tensor.rows, 3)
            self.assertEqual(tensor.cols, 33)
            self.assertEqual(tensor.payload_row_stride, 48)
            self.assertEqual(
                source_before,
                {
                    path.relative_to(canonical): path.read_bytes()
                    for path in canonical.rglob("*")
                    if path.is_file()
                },
            )
            with self.assertRaisesRegex(SQ8_0.ArtifactError, "sq_manifest.json"):
                SQ8_0.verify_canonical_artifact(output)

    def test_reader_rejects_checksum_bypass_and_nonzero_tail_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_model = write_sq8_0_source(root)
            canonical = root / "canonical"
            SQ8_0.build_canonical_artifact(source_model, canonical, tensor_names=[WEIGHT_NAME])
            clean = root / "clean"
            SQ8_1.build_sq8_1_artifact(canonical, clean)
            damaged = root / "damaged"
            shutil.copytree(clean, damaged)
            manifest = SQ8_1.read_json(damaged / SQ8_1.MANIFEST_FILE)
            payload_path = damaged / manifest["tensors"][0]["payload"]["file"]
            payload = bytearray(payload_path.read_bytes())
            payload[33] = 1
            payload_path.write_bytes(payload)
            entry = manifest["tensors"][0]
            entry["payload"]["sha256"] = SQ8_1.sha256_file(payload_path)
            rewrite_sq8_1_manifest(damaged, lambda replacement: replacement["tensors"][0].update(entry))
            with self.assertRaisesRegex(SQ8_1.ArtifactError, "tail padding"):
                SQ8_1.verify_sq8_1_artifact(damaged)


if __name__ == "__main__":
    unittest.main()
