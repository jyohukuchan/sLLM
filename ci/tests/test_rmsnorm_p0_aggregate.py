from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from common import ContractError, canonical_bytes, sha256_file  # noqa: E402
import aggregate_rmsnorm_p0_results as aggregator  # noqa: E402
import run_rmsnorm_p0_runtime as runner  # noqa: E402
import validate_rmsnorm_p0_contracts as contracts  # noqa: E402
from ci.tests.test_rmsnorm_p0_runner import (  # noqa: E402
    candidate, clean_process, ok_health, runtime_result, write_artifact,
)


def write_complete_failure_report(
    root: Path, target: str
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    row_root = root / target
    artifact_path, _, artifact = write_artifact(row_root, target)
    result = runtime_result(target, artifact, artifact_path)
    args = Namespace(
        target=target, run_id="p0-two-row-test", run_attempt=1,
        _started_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z"),
    )
    report = runner.make_report(
        args, candidate(), artifact, sha256_file(artifact_path),
        "complete canonical values retained; A5 numeric PASS remains locked",
        runtime_result=result, exit_code=0, duration_ns=1_000_000,
        stdout=canonical_bytes(result), health_pre=ok_health(target),
        health_post=ok_health(target), process_pre=clean_process(),
        process_post=clean_process(),
    )
    report_path = row_root / "report.json"
    report_path.write_bytes(canonical_bytes(report))
    return report_path, artifact_path, report, artifact


def write_disposition(
    root: Path,
    report_paths: list[Path],
    artifact_paths: list[Path],
    reports: list[dict[str, object]],
) -> tuple[Path, dict[str, object]]:
    rows = []
    for order, (report_path, artifact_path, report) in enumerate(
        zip(report_paths, artifact_paths, reports, strict=True)
    ):
        rows.append({
            "order": order,
            "row_id": contracts.ROWS[order],
            "target": contracts.TARGETS[order],
            "report_sha256": sha256_file(report_path),
            "artifact_sha256": sha256_file(artifact_path),
            "measurement_sha256": report["measurement_sha256"],
            "complete_measurements": True,
            "cases": contracts._case_summaries(report),
        })
    value = candidate()
    document = {
        "schema_version": "rmsnorm-p0-review-disposition-v1",
        "disposition_id": f"rmsnorm-p0-review-{value['reviewed_sha']}",
        "performance_sanity_disposition": "review_required",
        "threshold": contracts.expected_review_policy()["threshold"],
        "candidate": value,
        "tree_oid": value["git_tree_oid"],
        "matrix": {
            "path": contracts.MATRIX_PATH,
            "sha256": sha256_file(ROOT / contracts.MATRIX_PATH),
        },
        "review_policy": {
            "path": contracts.REVIEW_POLICY_PATH,
            "sha256": sha256_file(ROOT / contracts.REVIEW_POLICY_PATH),
        },
        "case_set_sha256": contracts.case_set_sha256(ROOT),
        "model_lock": {
            "path": contracts.MODEL_LOCK_PATH,
            "sha256": sha256_file(ROOT / contracts.MODEL_LOCK_PATH),
            "fingerprint": contracts.MODEL_LOCK_FINGERPRINT,
            "resolved_revision": contracts.RESOLVED_REVISION,
        },
        "source_set_sha256": contracts.source_set(ROOT)["sha256"],
        "review": {
            "decision": "accept_observation_without_threshold",
            "reviewer": "host-contract-reviewer",
            "reason": "Reviewed both complete canonical rows without an approved threshold.",
            "reviewed_at": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z"),
        },
        "canonical_rows": rows,
        "claims": contracts.expected_review_policy()["claims"],
    }
    path = root / "review-disposition.json"
    path.write_bytes(canonical_bytes(document))
    return path, document


def complete_inputs(
    root: Path,
) -> tuple[list[Path], list[Path], list[dict[str, object]], Path, dict[str, object]]:
    reports_and_artifacts = [
        write_complete_failure_report(root, target) for target in contracts.TARGETS
    ]
    report_paths = [item[0] for item in reports_and_artifacts]
    artifact_paths = [item[1] for item in reports_and_artifacts]
    reports = [item[2] for item in reports_and_artifacts]
    disposition_path, disposition = write_disposition(
        root, report_paths, artifact_paths, reports
    )
    return report_paths, artifact_paths, reports, disposition_path, disposition


class P0AggregateTests(unittest.TestCase):
    def test_review_disposition_generation_requires_explicit_human_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-review-generation-") as directory:
            root = Path(directory)
            report_paths, artifact_paths, reports, _, _ = complete_inputs(root)
            disposition = aggregator.generate_review_disposition(
                report_paths,
                artifact_paths,
                reports,
                candidate=candidate(),
                reviewer="named-reviewer",
                reason="Reviewed both complete rows without a performance threshold.",
                reviewed_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                repo=ROOT,
            )
            self.assertEqual(disposition["performance_sanity_disposition"], "review_required")
            with self.assertRaises(ContractError):
                aggregator.generate_review_disposition(
                    report_paths,
                    artifact_paths,
                    reports,
                    candidate=candidate(),
                    reviewer="named-reviewer",
                    reason="none",
                    reviewed_at=disposition["review"]["reviewed_at"],
                    repo=ROOT,
                )

    def test_complete_two_row_review_produces_pass_with_review_required_disposition(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-aggregate-") as directory:
            root = Path(directory)
            report_paths, artifact_paths, _, disposition_path, _ = complete_inputs(root)
            aggregate = aggregator.aggregate_reports(
                report_paths, artifact_paths, disposition_path,
                candidate=candidate(), repo=ROOT,
            )
            self.assertEqual(aggregate["state"], "PASS")
            self.assertEqual(aggregate["counts"]["collected_rows"], 2)
            self.assertEqual(aggregate["counts"]["collected_cases"], 10)
            self.assertEqual(aggregate["counts"]["passed_rows"], 2)
            contracts.validate_aggregate(aggregate)

    def test_aggregate_rejects_missing_duplicate_reordered_and_stale_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-aggregate-") as directory:
            root = Path(directory)
            report_paths, artifact_paths, _, disposition_path, _ = complete_inputs(root)
            attempts = (
                (report_paths[:1], artifact_paths, disposition_path, None),
                ([report_paths[0], report_paths[0]], artifact_paths, disposition_path, None),
                (list(reversed(report_paths)), artifact_paths, disposition_path, None),
                (report_paths, list(reversed(artifact_paths)), disposition_path, None),
                (
                    report_paths, artifact_paths, disposition_path,
                    datetime.now(timezone.utc) + timedelta(hours=25),
                ),
            )
            for reports, artifacts, disposition, now in attempts:
                with self.subTest(reports=reports, artifacts=artifacts, now=now):
                    with self.assertRaises(ContractError):
                        aggregator.aggregate_reports(
                            reports, artifacts, disposition,
                            candidate=candidate(), repo=ROOT, now=now,
                        )

    def test_review_rejects_missing_metadata_and_incomplete_or_mixed_values(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-review-") as directory:
            root = Path(directory)
            report_paths, artifact_paths, reports, _, disposition = complete_inputs(root)
            mutations = (
                lambda value: value["review"].__setitem__("reviewer", "TBD"),
                lambda value: value["review"].__setitem__("reason", "none"),
                lambda value: value["review"].__setitem__("reviewed_at", "not-a-date"),
                lambda value: value["review"].__setitem__(
                    "reviewed_at", "2099-01-01T00:00:00Z"
                ),
                lambda value: value["canonical_rows"].pop(),
                lambda value: value["canonical_rows"].reverse(),
                lambda value: value["canonical_rows"][0].__setitem__(
                    "measurement_sha256", "f" * 64
                ),
                lambda value: value["canonical_rows"][0]["cases"].pop(),
                lambda value: value["threshold"].__setitem__("approved", True),
                lambda value: value["claims"].__setitem__("optimized", True),
            )
            for mutation in mutations:
                changed = copy.deepcopy(disposition)
                mutation(changed)
                with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                    contracts.validate_disposition(
                        changed, ROOT, reports=reports,
                        report_sha256s=[sha256_file(path) for path in report_paths],
                        artifact_sha256s=[sha256_file(path) for path in artifact_paths],
                    )

    def test_report_artifact_and_candidate_identity_forgery_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-identity-") as directory:
            root = Path(directory)
            report_paths, artifact_paths, _, disposition_path, _ = complete_inputs(root)
            stale_candidate = candidate()
            for field in ("reviewed_sha", "tested_sha", "workflow_sha"):
                stale_candidate[field] = "c" * 40
            stale_candidate["git_tree_oid"] = "d" * 40
            with self.assertRaises(ContractError):
                aggregator.aggregate_reports(
                    report_paths, artifact_paths, disposition_path,
                    candidate=stale_candidate, repo=ROOT,
                )
            report = copy.deepcopy(contracts.read_json(report_paths[0]))
            report["artifact"]["artifact_sha256"] = "f" * 64
            report_paths[0].write_bytes(canonical_bytes(report))
            with self.assertRaises(ContractError):
                aggregator.aggregate_reports(
                    report_paths, artifact_paths, disposition_path,
                    candidate=candidate(), repo=ROOT,
                )

    def test_complete_measurements_reject_health_process_and_scope_deterioration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-health-") as directory:
            root = Path(directory)
            report_paths, _, _, _, _ = complete_inputs(root)
            base = contracts.read_json(report_paths[0])
            mutations = (
                lambda value: value["health_post"].__setitem__(
                    "ras_uncorrectable_count",
                    value["health_pre"]["ras_uncorrectable_count"] + 1,
                ),
                lambda value: value["health_post"].__setitem__("state", "DEGRADED"),
                lambda value: value["process_post"].__setitem__(
                    "residual_runner_children", [123]
                ),
                lambda value: value["scope"].__setitem__("gpu_execution", False),
                lambda value: value["dtype"].__setitem__("weight", "F32"),
                lambda value: value["execution"].__setitem__("duration_ns", 0),
                lambda value: value["execution"].__setitem__(
                    "started_at", "2099-01-01T00:00:00Z"
                ),
            )
            for mutation in mutations:
                changed = copy.deepcopy(base)
                mutation(changed)
                with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                    contracts.validate_report(changed)

    def test_handwritten_aggregate_pass_with_incomplete_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-pass-") as directory:
            root = Path(directory)
            report_paths, artifact_paths, _, disposition_path, _ = complete_inputs(root)
            aggregate = aggregator.aggregate_reports(
                report_paths, artifact_paths, disposition_path,
                candidate=candidate(), repo=ROOT,
            )
            forged = copy.deepcopy(aggregate)
            forged["state"] = "PASS"
            forged["rows"][0]["dispatch_count"] = 0
            forged["counts"]["failed_rows"] = 0
            with self.assertRaises(ContractError):
                contracts.validate_aggregate(forged)


if __name__ == "__main__":
    unittest.main()
