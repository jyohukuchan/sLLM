"""Regression tests for the SQ8_0 logical KV-prefix diagnostic."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "tools" / "evaluate-sq8_0-paged-decode-kv-diagnostic.py"


def load_module():
    specification = importlib.util.spec_from_file_location("sq8_kv_diagnostic", EVALUATOR)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"failed to load {EVALUATOR}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def write_f32le(path: Path, values: list[float]) -> str:
    payload = b"".join(struct.pack("<f", value) for value in values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def write_capture(root: Path, name: str, *, layer_one_v: list[float]) -> Path:
    capture_dir = root / name / "capture"
    layers = []
    for layer_index, v_values in enumerate(([1.0, 2.0], layer_one_v)):
        k_path = capture_dir / f"layer{layer_index}-k.f32le"
        v_path = capture_dir / f"layer{layer_index}-v.f32le"
        k_values = [float(layer_index + 1), float(layer_index + 2)]
        k_hash = write_f32le(k_path, k_values)
        v_hash = write_f32le(v_path, v_values)
        layers.append(
            {
                "layer_index": layer_index,
                "k_elements": len(k_values),
                "k_file": str(k_path.relative_to((root / name))),
                "k_f32_le_sha256": k_hash,
                "v_elements": len(v_values),
                "v_file": str(v_path.relative_to((root / name))),
                "v_f32_le_sha256": v_hash,
            }
        )
    result = {
        "requests": [
            {
                "request_id": "request-a",
                "prompt_token_ids": [11, 12],
                "kv_cache_prefix_capture": {
                    "schema_version": "ullm.sq8_0.paged_decode_kv_prefix_capture.v1",
                    "generated_index": 1,
                    "cache_len": 3,
                    "layer_count": len(layers),
                    "layers": layers,
                },
            }
        ]
    }
    result_path = root / name / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path


class EvaluateSq8PagedDecodeKvDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def evaluate(self, direct: Path, candidate: Path, output: Path) -> dict:
        previous_argv = sys.argv
        try:
            sys.argv = [
                str(EVALUATOR),
                "--direct-result",
                str(direct),
                "--candidate-result",
                str(candidate),
                "--output",
                str(output),
            ]
            with redirect_stdout(io.StringIO()):
                status = self.module.main()
        finally:
            sys.argv = previous_argv
        self.assertEqual(status, 0)
        return json.loads(output.read_text(encoding="utf-8"))

    def test_records_bitwise_equal_cache_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.evaluate(
                write_capture(root, "direct", layer_one_v=[3.0, 4.0]),
                write_capture(root, "tile", layer_one_v=[3.0, 4.0]),
                root / "summary.json",
            )
            self.assertTrue(summary["all_values_finite"])
            self.assertTrue(summary["all_bitwise_equal"])
            self.assertIsNone(summary["first_difference"])

    def test_records_first_layer_component_difference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.evaluate(
                write_capture(root, "direct", layer_one_v=[3.0, 4.0]),
                write_capture(root, "tile", layer_one_v=[3.0, 4.5]),
                root / "summary.json",
            )
            self.assertFalse(summary["all_bitwise_equal"])
            self.assertEqual(summary["first_difference"]["layer_index"], 1)
            self.assertEqual(summary["first_difference"]["component"], "v")


if __name__ == "__main__":
    unittest.main()
