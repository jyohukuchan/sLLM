from __future__ import annotations

import importlib.util
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2"


def load_tool(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ATTEST = load_tool("validate_qwen35_aq4_p2_path_attestation", "validate-qwen35-aq4-p2-path-attestation.py")


class Qwen35Aq4P2PathAttestationTests(unittest.TestCase):
    def test_production_attestation_binds_detached_path_and_worker_copies(self) -> None:
        report = ATTEST.validate(
            Namespace(
                attestation=P2 / "path-oracle-gpu-attestation-v1.json",
                raw_root=P2 / "path-oracle-gpu-run-v1",
                base_path=P2 / "path-oracle-v1",
                path=P2 / "path-oracle-v2",
                source_path=P2 / "source-oracle-v2",
                cases=ROOT / "tests/fixtures/qwen35-aq4-p2-oracle/cases.json",
            )
        )
        self.assertEqual(report["status"], "valid_with_blockers")
        self.assertEqual(report["corrected_path"]["status"], "valid")
        self.assertEqual(report["execution"]["binary"]["evidence_copy_nlink"], 1)
        self.assertEqual(report["execution"]["binary"]["worker_evidence_copy_nlink"], 1)
        self.assertFalse(report["path_regression"]["exact_greedy"])
        self.assertFalse(report["path_regression"]["exact_topk"])


if __name__ == "__main__":
    unittest.main()
