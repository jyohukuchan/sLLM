from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.audit_aq4_weight_provenance import run_audit, scale_values


def bf16_bytes(values: list[float]) -> bytes:
    result = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        result.extend(struct.pack("<H", bits >> 16))
    return bytes(result)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header: dict[str, object] = {}
    offset = 0
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(payload)]}
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for _, (_, _, payload) in tensors.items():
            handle.write(payload)


class Aq4WeightProvenanceTest(unittest.TestCase):
    def test_streaming_passthrough_and_quantized_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            package_dir = root / "package"
            source_dir.mkdir()
            (package_dir / "tensors").mkdir(parents=True)
            (package_dir / "codebooks").mkdir()
            source_path = source_dir / "model.safetensors"

            codebook = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
            scales = scale_values("e4m3")
            unit_index = int(np.flatnonzero(scales == 1.0)[0])
            source_quant = [float(codebook[index] * 2.0) for index in (0, 1, 2, 3, 4, 5, 6, 7)]
            source_passthrough = [0.5, 1.0, 1.5, 2.0]
            write_safetensors(
                source_path,
                {
                    "quant.weight": ("BF16", [2, 4], bf16_bytes(source_quant)),
                    "norm.weight": ("BF16", [4], bf16_bytes(source_passthrough)),
                },
            )
            (package_dir / "codebooks" / "test.f32").write_bytes(codebook.tobytes())
            (package_dir / "tensors" / "quant.idx4").write_bytes(bytes((0x10 | 0x00, 0x32, 0x54, 0x76)))
            (package_dir / "tensors" / "quant.scale_u8").write_bytes(bytes((unit_index, unit_index, unit_index, unit_index)))
            passthrough_payload = source_path.read_bytes()
            # Locate the second tensor payload without loading any package-sized data.
            reader_header_len = struct.unpack("<Q", passthrough_payload[:8])[0]
            header = json.loads(passthrough_payload[8 : 8 + reader_header_len])
            start, end = header["norm.weight"]["data_offsets"]
            data_start = 8 + reader_header_len
            norm_raw = passthrough_payload[data_start + start : data_start + end]
            (package_dir / "passthrough").mkdir()
            (package_dir / "passthrough" / "norm.raw").write_bytes(norm_raw)
            manifest = {
                "schema_version": "ullm-prototype-manifest-v0.1",
                "source_model_dir": str(source_dir),
                "codebooks": [],
                "tensors": [
                    {
                        "name": "quant.weight",
                        "source_file": str(source_path),
                        "dtype": "BF16",
                        "shape": [2, 4],
                        "family": "test",
                        "candidate_id": "test",
                        "scale_format": "e4m3",
                        "group_size": 2,
                        "tensor_scale": 2.0,
                        "scale_window": 0,
                        "elements": 8,
                        "groups": 4,
                        "index_file": "tensors/quant.idx4",
                        "index_encoding": "idx4_low_nibble_first",
                        "scale_file": "tensors/quant.scale_u8",
                        "scale_encoding": "u8_scale_table_index",
                        "codebook_file": "codebooks/test.f32",
                        "metrics": {},
                    }
                ],
                "passthrough_tensors": [
                    {
                        "name": "norm.weight",
                        "source_file": str(source_path),
                        "dtype": "BF16",
                        "shape": [4],
                        "family": "other",
                        "elements": 4,
                        "payload_file": "passthrough/norm.raw",
                        "payload_encoding": "raw_safetensors_payload",
                        "payload_bytes": len(norm_raw),
                        "payload_sha256": __import__("hashlib").sha256(norm_raw).hexdigest(),
                    }
                ],
            }
            (package_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            report = run_audit(package_dir, source_dir, ("quant.weight", "norm.weight"), chunk_groups=1, served_manifest=None)
            quant, passthrough = report["results"]
            self.assertEqual(quant["status"], "ok")
            self.assertTrue(quant["shape_exact"])
            self.assertTrue(quant["payload_lengths_exact"])
            self.assertLess(quant["measured_relative_mse"], 1e-3)
            self.assertEqual(passthrough["status"], "ok")
            self.assertTrue(passthrough["source_payload_hash_exact"])
            self.assertTrue(report["source_root_exact"])

    def test_shape_mismatch_is_reported_as_transpose_suspect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            package_dir = root / "package"
            source_dir.mkdir()
            package_dir.mkdir()
            source_path = source_dir / "model.safetensors"
            write_safetensors(source_path, {"quant.weight": ("BF16", [2, 4], bf16_bytes([1.0] * 8))})
            manifest = {
                "schema_version": "ullm-prototype-manifest-v0.1",
                "source_model_dir": str(source_dir),
                "tensors": [
                    {
                        "name": "quant.weight",
                        "source_file": str(source_path),
                        "dtype": "BF16",
                        "shape": [4, 2],
                        "family": "test",
                        "candidate_id": "test",
                        "scale_format": "e4m3",
                        "group_size": 2,
                        "tensor_scale": 1.0,
                        "elements": 8,
                        "groups": 4,
                        "index_file": "missing.idx4",
                        "index_encoding": "idx4_low_nibble_first",
                        "scale_file": "missing.scale_u8",
                        "scale_encoding": "u8_scale_table_index",
                        "codebook_file": "missing.f32",
                        "metrics": {},
                    }
                ],
            }
            (package_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            result = run_audit(package_dir, source_dir, ("quant.weight",), served_manifest=None)["results"][0]
            self.assertEqual(result["status"], "ok")
            self.assertFalse(result["shape_exact"])
            self.assertTrue(result["transpose_suspected"])


if __name__ == "__main__":
    unittest.main()
