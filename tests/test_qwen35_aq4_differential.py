from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qwen35_aq4_differential", ROOT / "tools/trace-qwen35-aq4-differential.py")
assert SPEC and SPEC.loader
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)


def write_trace(root: Path, *, mismatch_layer: int | None = None) -> None:
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema_version": TRACE.SCHEMA}) + "\n", encoding="utf-8")
    stages = [{"stage": "embedding", "sample": {"coordinates": [0], "elements": 1, "values": [1.0]}}]
    for index in range(32):
        stages.append({"stage": "decoder_layer", "layer_index": index, "sample": {"coordinates": [0], "elements": 1, "values": [2.0 if index == mismatch_layer else 1.0]}})
    stages.extend([
        {"stage": "final_norm", "sample": {"coordinates": [0], "elements": 1, "values": [1.0]}},
        {"stage": "lm_head", "sample": {"coordinates": [0], "elements": 1, "values": [1.0]}},
    ])
    row = {"case_id": "c", "step": 0, "context_length": 2, "context_token_ids_sha256": "a" * 64, "stages": stages, "greedy_token_id": 1}
    (root / "payload.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


class Qwen35Aq4DifferentialTests(unittest.TestCase):
    def test_full_context_hash_contract(self) -> None:
        self.assertEqual(
            TRACE._context_hash([11, 12, 13]),
            "42ea52c728680a54afafd1c1e1e45f13300c3ceb962f320f3900196a0c46215c",
        )
        self.assertEqual(
            TRACE._context_hash([11, 12, 13, 220]),
            "6af1601b9bf35d095b24c5bac3a95a01bf77d047b576441d0a5f9510eec66249",
        )
        self.assertEqual(
            TRACE._context_hash([21, 22]),
            "3bca9e21e3b6f741ed412f91d7696146c254ff68bd9be9ca41b1d172eb3549e6",
        )

    def test_first_decoder_layer_mismatch_is_localized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = parent / "source"
            path = parent / "path"
            output = parent / "analysis"
            write_trace(source)
            write_trace(path, mismatch_layer=7)
            report = TRACE.analyze(source, path, output)
            self.assertEqual(report["reports"][0]["diagnosis"], "decoder_layer_7")

    def test_missing_intermediate_stages_are_not_overclassified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = parent / "source"
            path = parent / "path"
            output = parent / "analysis"
            write_trace(source)
            path.mkdir()
            (path / "manifest.json").write_text(json.dumps({"schema_version": TRACE.SCHEMA}) + "\n", encoding="utf-8")
            row = {"case_id": "c", "step": 0, "context_length": 2, "context_token_ids_sha256": "a" * 64, "stages": [{"stage": "final_norm", "sample": {"coordinates": [0], "elements": 1, "values": [2.0]}}], "greedy_token_id": 9}
            (path / "payload.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = TRACE.analyze(source, path, output)
            self.assertEqual(report["reports"][0]["diagnosis"], "inconclusive_missing_intermediate_aq4_trace")


if __name__ == "__main__":
    unittest.main()
