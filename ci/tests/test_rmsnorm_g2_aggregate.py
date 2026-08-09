from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

import aggregate_rmsnorm_g2_results as aggregator  # noqa: E402
import run_rmsnorm_g2_runtime as runner  # noqa: E402
import validate_rmsnorm_g2_contracts as contracts  # noqa: E402
from common import ContractError  # noqa: E402
from ci.tests.test_rmsnorm_g2_runner import artifact, candidate, fresh_g2_binary  # noqa: E402
from ci.tests.test_rmsnorm_g2_slice import slice_record  # noqa: E402


def failed_report(target: str, artifact_document: dict[str, object]) -> dict[str, object]:
    args = Namespace(target=target, artifact="unused", output_dir="unused", reviewed_sha="a" * 40, tested_sha="a" * 40, workflow_sha="a" * 40, tree_oid="b" * 40)
    record = slice_record()
    record["output"] = {"size_bytes": contracts.BYTE_SIZE, "sha256": "f" * 64}
    report = runner.make_failure_report(args, candidate(), record, artifact_document, "")
    return report


class G2AggregateTests(unittest.TestCase):
    def _files(self, root: Path) -> tuple[list[Path], list[Path]]:
        reports: list[Path] = []
        artifacts: list[Path] = []
        for target in contracts.TARGETS:
            document = artifact(target, candidate())
            artifact_path = root / f"{target}-artifact.json"
            artifact_path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
            artifacts.append(artifact_path)
            report_path = root / f"{target}-report.json"
            actual = fresh_g2_binary().read_bytes()
            binary_path = root / "sllm-rmsnorm-g2-evidence"
            binary_path.write_bytes(actual)
            binary_path.chmod(0o755)
            (root / "sllm-rmsnorm-g2-evidence.sha256").write_bytes((document["binary"]["sha256"] + "  sllm-rmsnorm-g2-evidence\n").encode())
            report_path.write_text(json.dumps(failed_report(target, document), sort_keys=True), encoding="utf-8")
            reports.append(report_path)
        return reports, artifacts

    def test_exact_two_rows_aggregate_remains_fail_until_a5_parser_exists(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g2-aggregate-") as directory:
            reports, artifacts = self._files(Path(directory))
            result = aggregator.aggregate_reports(reports, artifacts, candidate=candidate())
            self.assertEqual(result["state"], "FAIL")
            self.assertEqual(result["counts"]["collected_cases"], 12)
            self.assertEqual([row["target"] for row in result["rows"]], ["gfx1030", "gfx1201"])

    def test_missing_duplicate_stale_mixed_and_zero_collection_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g2-aggregate-") as directory:
            root = Path(directory)
            reports, artifacts = self._files(root)
            for bad_reports, bad_artifacts, label in ((reports[:1], artifacts, "missing"), (reports + [reports[0]], artifacts, "duplicate")):
                with self.subTest(label=label), self.assertRaises(ContractError):
                    aggregator.aggregate_reports(bad_reports, bad_artifacts, candidate=candidate())
            stale = json.loads(reports[1].read_text(encoding="utf-8"))
            stale["candidate"]["reviewed_sha"] = "f" * 40
            reports[1].write_text(json.dumps(stale), encoding="utf-8")
            with self.assertRaises(ContractError):
                aggregator.aggregate_reports(reports, artifacts, candidate=candidate())

            reports, artifacts = self._files(root)
            zero = json.loads(reports[0].read_text(encoding="utf-8"))
            zero["state"] = "FAIL"
            zero["scope"]["dispatch_count"] = 0
            zero["dispatch"]["dispatch_count"] = 0
            zero["health_pre"]["available"] = False
            zero["health_pre"]["reliable"] = False
            zero["health_pre"]["state"] = "UNAVAILABLE"
            zero["health_post"] = copy.deepcopy(zero["health_pre"])
            for case in zero["cases"]:
                case["state"] = "FAIL"
                case["dispatch_count"] = 0
            zero["collection"] = {"expected_cases": 6, "collected_cases": 6, "passed_cases": 0, "failed_cases": 6, "expected_rows": 1, "collected_rows": 1}
            reports[0].write_text(json.dumps(zero), encoding="utf-8")
            result = aggregator.aggregate_reports(reports, artifacts, candidate=candidate())
            self.assertEqual(result["state"], "FAIL")
            self.assertNotEqual(result["counts"]["passed_rows"], 2)

    def test_handwritten_pass_report_is_rejected_before_aggregate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g2-aggregate-") as directory:
            root = Path(directory)
            reports, artifacts = self._files(root)
            forged = json.loads(reports[0].read_text(encoding="utf-8"))
            forged["state"] = "PASS"
            reports[0].write_text(json.dumps(forged), encoding="utf-8")
            with self.assertRaises(ContractError):
                aggregator.aggregate_reports(reports, artifacts, candidate=candidate())

    def test_aggregate_identity_and_count_hashes_are_not_declarative(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g2-aggregate-") as directory:
            root = Path(directory)
            reports, artifacts = self._files(root)
            aggregate = aggregator.aggregate_reports(reports, artifacts, candidate=candidate())
            for section in ("matrix", "model_lock", "tolerance"):
                changed = copy.deepcopy(aggregate)
                changed[section]["sha256"] = "0" * 64
                with self.subTest(section=section), self.assertRaises(ContractError):
                    contracts.validate_aggregate(changed)
            changed = copy.deepcopy(aggregate)
            changed["counts"]["collected_cases"] = 0
            with self.assertRaises(ContractError):
                contracts.validate_aggregate(changed)

    def test_aggregate_validates_preserved_artifacts_without_single_live_builder_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g2-aggregate-") as directory:
            reports, artifacts = self._files(Path(directory))
            original = aggregator.validate_artifact
            ownership_modes: list[bool] = []

            def observe(*args: object, **kwargs: object) -> dict[str, object]:
                ownership_modes.append(bool(kwargs.get("require_builder_owned_output", True)))
                return original(*args, **kwargs)

            with patch.object(aggregator, "validate_artifact", side_effect=observe):
                aggregator.aggregate_reports(reports, artifacts, candidate=candidate())
            self.assertEqual(ownership_modes, [False, False])


if __name__ == "__main__":
    unittest.main()
