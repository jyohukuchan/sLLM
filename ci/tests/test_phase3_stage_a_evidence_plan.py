#!/usr/bin/env python3
"""Focused host-only negative tests for the Phase 3 Stage A evidence plan."""

from __future__ import annotations

import copy
import inspect
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jsonschema import Draft202012Validator

CI_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(CI_TOOLS) not in sys.path:
    sys.path.insert(0, str(CI_TOOLS))

from common import ROOT, canonical_bytes  # noqa: E402
import plan_phase3_stage_a_evidence as planner  # noqa: E402
import validate_rmsnorm_g1_contracts as g1_contracts  # noqa: E402


SHA = "a" * 40
TREE = "b" * 40


class Phase3StageAEvidencePlanTests(unittest.TestCase):
    def identity(self, _repo: Path, expected: dict[str, object]) -> dict[str, object]:
        return planner.api_only_identity_verifier(_repo, expected)

    def make_plan(self, root: Path, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "repo": ROOT,
            "run_root": root,
            "run_id": "123456",
            "run_attempt": "1",
            "reviewed_sha": SHA,
            "tested_sha": SHA,
            "workflow_sha": SHA,
            "tree_oid": TREE,
            "identity_verifier": self.identity,
        }
        values.update(overrides)
        return planner.build_plan(**values)  # type: ignore[arg-type]

    def test_canonical_plan_is_deterministic_and_has_one_newline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p3-plan-") as directory:
            parent = Path(directory)
            first = self.make_plan(parent / "stage-a-1")
            second = self.make_plan(parent / "stage-a-1")
            first_bytes = canonical_bytes(first)
            second_bytes = canonical_bytes(second)
            self.assertEqual(first_bytes, second_bytes)
            self.assertTrue(first_bytes.endswith(b"\n"))
            self.assertFalse(first_bytes.endswith(b"\n\n"))
            self.assertEqual(first_bytes.count(b"\n"), 1)
            self.assertEqual(json.loads(first_bytes), second)

    def test_plan_records_non_gpu_state_and_contract_derived_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p3-plan-") as directory:
            plan = self.make_plan(Path(directory) / "stage-a-1")
        self.assertEqual(plan["evidence_state"], "NOT_EXECUTED")
        self.assertFalse(plan["evidence_claim"]["gpu_evidence"])  # type: ignore[index]
        self.assertEqual(plan["target_order"], ["gfx1030", "gfx1201"])
        self.assertEqual(plan["h3"]["workspace"], {  # type: ignore[index]
            "mount_destination": "/workspace",
            "mount_read_only": True,
            "workdir": "/workspace",
        })
        g1_rows = plan["g1"]["rows"]  # type: ignore[index]
        self.assertIn("cargo-target-gfx1030", g1_rows[0]["cargo_target_dir"])
        self.assertIn("native-hip-build-gfx1201", g1_rows[1]["native_hip_build_dir"])
        self.assertLess(len(g1_rows[1]["socket_path_projection"].encode()), 108)
        self.assertTrue(plan["g2"]["builder_root"].endswith("/target"))  # type: ignore[index]
        self.assertTrue(plan["g2"]["rows"][0]["builder_output_path"].endswith("target/release/sllm-rmsnorm-g2-evidence"))  # type: ignore[index]
        self.assertEqual(plan["p0"]["rows"][0]["builder_owned_output"]["binary_name"], "sllm-rmsnorm-p0-evidence")  # type: ignore[index]
        self.assertTrue(plan["authority_files"])
        self.assertEqual(plan["authority_files_sha256"], planner.sha256_json(plan["authority_files"]))  # type: ignore[arg-type]

    def test_invalid_run_identity_is_rejected(self) -> None:
        cases = (("0", "1"), ("01", "1"), ("not-numeric", "1"), ("1", "0"), ("1", "-1"), ("1", "1.0"))
        with tempfile.TemporaryDirectory(prefix="p3-plan-") as directory:
            for run_id, attempt in cases:
                with self.subTest(run_id=run_id, attempt=attempt):
                    with self.assertRaises(planner.PlanError):
                        self.make_plan(Path(directory) / f"stage-{len(run_id)}-{attempt}", run_id=run_id, run_attempt=attempt)

    def test_invalid_repo_and_run_root_shapes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p3-plan-") as directory:
            parent = Path(directory)
            existing = parent / "existing"
            existing.mkdir()
            linked = parent / "linked"
            linked.symlink_to(parent / "missing", target_is_directory=True)
            cases = [
                {"repo": Path("relative/repo")},
                {"run_root": existing},
                {"run_root": linked},
                {"run_root": parent / ("x" * 40)},
            ]
            for override in cases:
                with self.subTest(override=override):
                    with self.assertRaises(planner.PlanError):
                        self.make_plan(parent / "stage-valid", **override)

    def test_identity_mismatch_and_dirty_candidate_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p3-plan-") as directory:
            parent = Path(directory)
            with self.assertRaises(planner.PlanError):
                self.make_plan(parent / "mismatch", tested_sha="c" * 40)

            def dirty(_repo: Path, expected: dict[str, object]) -> dict[str, object]:
                result = dict(expected)
                result["worktree_clean"] = False
                return result

            with self.assertRaises(planner.PlanError):
                self.make_plan(parent / "dirty", identity_verifier=dirty)

            def wrong_tree(_repo: Path, expected: dict[str, object]) -> dict[str, object]:
                result = dict(expected)
                result["git_tree_oid"] = "d" * 40
                return result

            with self.assertRaises(planner.PlanError):
                self.make_plan(parent / "wrong-tree", identity_verifier=wrong_tree)

    def test_matrix_target_order_and_duplication_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p3-plan-") as directory:
            cases = (
                (planner.g1_contracts, "rows"),
                (planner.g2_contracts, "targets"),
                (planner.p0_contracts, "targets"),
            )
            for module, field in cases:
                original = module.validate_matrix
                for mutation in ("swapped", "duplicate"):
                    with self.subTest(module=module.__name__, mutation=mutation):
                        matrix = copy.deepcopy(original(ROOT))
                        if mutation == "swapped":
                            matrix[field][0], matrix[field][1] = matrix[field][1], matrix[field][0]
                        else:
                            matrix[field][1] = copy.deepcopy(matrix[field][0])
                        with mock.patch.object(module, "validate_matrix", return_value=matrix):
                            with self.assertRaises(planner.PlanError):
                                self.make_plan(Path(directory) / f"{module.__name__}-{mutation}")

    def test_schema_closes_g1_g2_and_p0_row_order(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p3-plan-") as directory:
            plan = self.make_plan(Path(directory) / "schema-order")
        schema = json.loads((ROOT / planner.PLAN_SCHEMA).read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        for section in ("g1", "g2", "p0"):
            with self.subTest(section=section):
                swapped = copy.deepcopy(plan)
                swapped[section]["rows"][0], swapped[section]["rows"][1] = (
                    swapped[section]["rows"][1], swapped[section]["rows"][0]
                )
                duplicate = copy.deepcopy(plan)
                duplicate[section]["rows"][1] = copy.deepcopy(duplicate[section]["rows"][0])
                self.assertTrue(list(validator.iter_errors(swapped)))
                self.assertTrue(list(validator.iter_errors(duplicate)))

    def test_authority_rejects_symlink_components_and_repo_escape(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p3-authority-") as directory:
            root = Path(directory) / "repo"
            outside = Path(directory) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "safe").mkdir()
            (root / "safe" / "file").write_text("safe\n", encoding="utf-8")
            (outside / "file").write_text("outside\n", encoding="utf-8")
            original = planner.AUTHORITY_FILES
            try:
                planner.AUTHORITY_FILES = ("safe/file",)
                self.assertEqual(planner._authority_records(root)[0]["path"], "safe/file")

                (root / "safe-link").symlink_to(outside, target_is_directory=True)
                planner.AUTHORITY_FILES = ("safe-link/file",)
                with self.assertRaises(planner.PlanError):
                    planner._authority_records(root)

                (root / "safe" / "file-link").symlink_to(outside / "file")
                planner.AUTHORITY_FILES = ("safe/file-link",)
                with self.assertRaises(planner.PlanError):
                    planner._authority_records(root)
            finally:
                planner.AUTHORITY_FILES = original

    def test_workflow_validator_failure_propagates_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p3-plan-") as directory:
            with mock.patch.object(
                planner.workflow_contracts,
                "validate_rmsnorm_h3_workflow",
                side_effect=planner.ContractError("patched workflow drift"),
            ):
                with self.assertRaisesRegex(planner.PlanError, "patched workflow drift"):
                    self.make_plan(Path(directory) / "x")

    def test_plan_construction_has_no_filesystem_or_execution_side_effects(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p3-plan-") as directory:
            parent = Path(directory)
            run_root = parent / "stage-a-1"
            before = sorted(parent.iterdir())
            with mock.patch.object(planner.subprocess, "run", side_effect=AssertionError("subprocess forbidden")), \
                mock.patch.object(planner.subprocess, "Popen", side_effect=AssertionError("subprocess forbidden")), \
                mock.patch.object(planner.g1_builder, "build_runtime_artifact", side_effect=AssertionError("builder forbidden")), \
                mock.patch.object(planner.g1_builder, "CompilerBroker", side_effect=AssertionError("socket broker forbidden")):
                plan = self.make_plan(run_root)
            self.assertIsNotNone(plan)
            self.assertEqual(before, sorted(parent.iterdir()))
            self.assertFalse(run_root.exists())
            self.assertFalse((parent / "stage-a-1" / "artifacts").exists())

    def test_api_only_verifier_is_not_a_cli_bypass(self) -> None:
        self.assertIn("API-only", planner.api_only_identity_verifier.__doc__ or "")
        parser = planner._parser()
        self.assertNotIn("identity-verifier", [action.dest for action in parser._actions])
        self.assertNotIn("identity_verifier", inspect.signature(planner.main).parameters)

        cli_args = [
            "--repo", str(ROOT), "--run-root", "/tmp/p3-stage-a",
            "--run-id", "123456", "--run-attempt", "1",
            "--reviewed-sha", SHA, "--tested-sha", SHA,
            "--workflow-sha", SHA, "--tree-oid", TREE,
        ]
        stdout = io.BytesIO()
        with mock.patch.object(planner, "build_plan", return_value={}) as build_plan, \
             mock.patch.object(planner.sys, "stdout", mock.Mock(buffer=stdout)):
            self.assertEqual(planner.main(cli_args), 0)
        self.assertNotIn("identity_verifier", build_plan.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
