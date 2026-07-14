from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "qwen35-aq4-p2-oracle"
REAL_SOURCE = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/source-oracle-v2"


def load_tool(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAPTURE = load_tool("capture_qwen35_aq4_p2_oracle", "capture-qwen35-aq4-p2-oracle.py")
VALIDATE = load_tool("validate_qwen35_aq4_p2_oracle", "validate-qwen35-aq4-p2-oracle.py")
EXPORTER = load_tool("export_qwen35_aq4_source_oracle", "export-qwen35-aq4-source-oracle.py")
ORACLE = CAPTURE.oracle


class Qwen35Aq4P2OracleTests(unittest.TestCase):
    def _capture_pair(self, root: Path) -> tuple[Path, Path, Path]:
        source = root / "source"
        path = root / "path"
        link = root / "link"
        cases = FIXTURE / "cases.json"
        payload = FIXTURE / "payload.jsonl"
        CAPTURE.capture(
            type(
                "Args",
                (),
                {
                    "output": source,
                    "cases": cases,
                    "payload": payload,
                    "kind": "source",
                    "source_root": FIXTURE / "source-model",
                    "evidence_class": "synthetic_fixture",
                },
            )()
        )
        CAPTURE.capture(
            type(
                "Args",
                (),
                {
                    "output": path,
                    "cases": cases,
                    "payload": payload,
                    "kind": "path",
                    "tokenizer_root": FIXTURE / "source-model",
                    "tokenizer_file": list(ORACLE.TOKENIZER_FILES),
                    "artifact_manifest": FIXTURE / "path-artifact.json",
                    "package_manifest": FIXTURE / "package-manifest.json",
                    "model_id": "Qwen/Qwen3.5-9B",
                    "model_revision": None,
                    "evidence_class": "synthetic_fixture",
                },
            )()
        )
        CAPTURE.link(type("Args", (), {"source_oracle": source, "path_oracle": path, "output": link})())
        return source, path, link

    def test_fixture_pair_is_streaming_valid_but_not_promotable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, path, link = self._capture_pair(Path(temporary))
            source_report = VALIDATE.validate_oracle(source, "source")
            path_report = VALIDATE.validate_oracle(path, "path")
            link_report = VALIDATE.validate_link(link, source, path)
            self.assertEqual(source_report["status"], "valid")
            self.assertEqual(path_report["status"], "valid")
            self.assertFalse(source_report["promotion_eligible"])
            self.assertFalse(path_report["promotion_eligible"])
            self.assertFalse(link_report["promotion_eligible"])
            self.assertFalse(source_report["usable_as_source_evidence"])
            self.assertFalse(path_report["usable_as_path_evidence"])
            self.assertFalse(link_report["usable_as_p2_oracle_link"])
            self.assertTrue(link_report["agreement"]["greedy_token_exact"])
            self.assertTrue(link_report["agreement"]["topk_exact"])
            self.assertEqual(link_report["agreement"]["record_count"], 3)

    def test_payload_tamper_is_rejected_without_trusting_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, _, _ = self._capture_pair(Path(temporary))
            payload = source / "payload.jsonl"
            payload.write_text(payload.read_text(encoding="utf-8").replace('"greedy_token_id":7', '"greedy_token_id":6', 1), encoding="utf-8")
            with self.assertRaises(ORACLE.OracleError):
                ORACLE.validate_manifest(source, expected_kind="source")

    def test_checkpoint_aggregate_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, _, _ = self._capture_pair(Path(temporary))
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["identity"]["source_checkpoint"]["aggregate_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ORACLE.OracleError):
                ORACLE.validate_manifest(source, expected_kind="source")

    def test_duplicate_json_key_and_sample_bound_fail_closed(self) -> None:
        raw = '{"case_id":"x","case_id":"y","greedy_token_id":1}'
        with self.assertRaises(ORACLE.OracleError):
            json.loads(raw, object_pairs_hook=ORACLE.reject_duplicate_keys)
        sample = {"dtype": "f32", "indices": list(range(257)), "shape": [257], "values": [0.0] * 257}
        with self.assertRaises(ORACLE.OracleError):
            ORACLE.validate_payload_record({"case_id": "x", "greedy_token_id": 1, "hidden_sample": sample, "logit_sample": sample, "step": 0, "topk": [{"logit": 1.0, "token_id": 1}]}, "record")

    def test_source_probe_reports_missing_independent_forward_artifact(self) -> None:
        report = VALIDATE.probe_source(FIXTURE / "source-model", None)
        self.assertTrue(report["source_model"]["status"], "available")
        self.assertEqual(report["independent_forward_artifact"]["status"], "blocked")
        self.assertIn("checkpoint metadata alone is not an oracle", report["independent_forward_artifact"]["blocker"])

    def test_global_topk_ties_use_smallest_token_id(self) -> None:
        import torch

        topk = EXPORTER._topk(torch.tensor([1.0, 3.0, 3.0, 2.0]), count=3)
        self.assertEqual([entry["token_id"] for entry in topk], [1, 2, 3])
        self.assertEqual(topk[0]["token_id"], 1)

    def test_reversed_topk_tie_is_rejected(self) -> None:
        sample = {"dtype": "f32", "indices": [0], "shape": [4], "values": [0.0]}
        record = {"case_id": "tie", "greedy_token_id": 2, "hidden_sample": sample, "logit_sample": sample, "step": 0, "topk": [{"logit": 3.0, "token_id": 2}, {"logit": 3.0, "token_id": 1}]}
        with self.assertRaises(ORACLE.OracleError):
            ORACLE.validate_payload_record(record, "tie record")

    def test_runtime_and_checksums_are_exactly_cross_checked(self) -> None:
        manifest = ORACLE.load_json(REAL_SOURCE / "manifest.json")
        VALIDATE._validate_runtime(REAL_SOURCE, manifest)
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "source"
            shutil.copytree(REAL_SOURCE, copied)
            runtime_path = copied / "runtime.json"
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime["run"]["row_count"] += 1
            runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
            with self.assertRaises(ORACLE.OracleError):
                VALIDATE._validate_runtime(copied, manifest)

    def test_link_detects_path_payload_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, path, link = self._capture_pair(Path(temporary))
            payload = path / "payload.jsonl"
            payload.write_text(payload.read_text(encoding="utf-8").replace('"values":[0.125,-0.5]', '"values":[0.125,-0.25]', 1), encoding="utf-8")
            with self.assertRaises(ORACLE.OracleError):
                VALIDATE.validate_link(link, source, path)


if __name__ == "__main__":
    unittest.main()
