from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


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
    def test_dynamic_loader_is_temporary_and_fail_closed(self) -> None:
        self.assertNotIn("qwen35_aq4_p2_oracle_attestation", sys.modules)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools = root / "tools"
            tools.mkdir()
            good = tools / "good.py"
            good.write_text(
                "from dataclasses import dataclass\n"
                "@dataclass(frozen=True)\n"
                "class Record:\n"
                "    value: int\n",
                encoding="utf-8",
            )
            broken = tools / "broken.py"
            broken.write_text("partial = True\nraise RuntimeError('broken import')\n", encoding="utf-8")
            replaced = tools / "replaced.py"
            replaced.write_text(
                "import sys\nsys.modules[__name__] = object()\n",
                encoding="utf-8",
            )
            with mock.patch.object(ATTEST, "ROOT", root):
                success_name = "aq4_attestation_loader_success"
                loaded = ATTEST._load(success_name, good.name)
                self.assertEqual(loaded.Record(7).value, 7)
                self.assertNotIn(success_name, sys.modules)

                collision_name = "aq4_attestation_loader_collision"
                sentinel = object()
                sys.modules[collision_name] = sentinel
                try:
                    with self.assertRaisesRegex(RuntimeError, "already registered"):
                        ATTEST._load(collision_name, good.name)
                    self.assertIs(sys.modules[collision_name], sentinel)
                finally:
                    sys.modules.pop(collision_name, None)

                failure_name = "aq4_attestation_loader_failure"
                with self.assertRaisesRegex(RuntimeError, "broken import"):
                    ATTEST._load(failure_name, broken.name)
                self.assertNotIn(failure_name, sys.modules)

                replacement_name = "aq4_attestation_loader_replacement"
                with self.assertRaisesRegex(RuntimeError, "registration changed"):
                    ATTEST._load(replacement_name, replaced.name)
                self.assertNotIn(replacement_name, sys.modules)

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
