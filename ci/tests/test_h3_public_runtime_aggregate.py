#!/usr/bin/env python3
"""Fail-closed aggregation tests for exactly two H3 public-runtime rows."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT / "ci/tools", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from aggregate_h3_public_runtime_results import (  # noqa: E402
    EXPECTED_ROWS,
    ContractError,
    aggregate,
    write_summary,
)
import aggregate_h3_public_runtime_results as aggregate_module  # noqa: E402
from validate_h3_public_runtime_contracts import validate_against_schema, read_json  # noqa: E402
from ci.tests.test_h3_public_runtime_contracts import ArtifactFixture  # noqa: E402


class H3PublicRuntimeAggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix=".h3-public-aggregate-", dir=ROOT)
        self.root = Path(self.tempdir.name)
        self.artifact_dir = self.root / "rows"
        self.artifact_dir.mkdir()
        self.fixtures = [ArtifactFixture("gfx1030"), ArtifactFixture("gfx1201")]
        for fixture in self.fixtures:
            shutil.copytree(fixture.row_dir, self.artifact_dir / fixture.row_id)
        self.needs = self.root / "needs.json"
        self.needs.write_text(json.dumps({"state": "PASS", "rows": list(EXPECTED_ROWS)}) + "\n", encoding="utf-8")
        self.output_dir = self.root / "aggregate"
        self.identity = {"commit": "a" * 40, "tree": "b" * 40, "clean": True}

    def tearDown(self) -> None:
        for fixture in self.fixtures:
            fixture.close()
        self.tempdir.cleanup()

    def _args(self) -> Namespace:
        return Namespace(
            repo=ROOT,
            artifact_dir=self.artifact_dir,
            output_dir=self.output_dir,
            run_id="unit-h3-public-runtime",
            run_attempt=1,
            reviewed_sha="a" * 40,
            tested_sha="a" * 40,
            workflow_sha="a" * 40,
            tree_oid="b" * 40,
            needs_json=self.needs,
        )

    def test_exact_two_pass_rows_aggregate_and_validate_against_schema(self) -> None:
        with patch("aggregate_h3_public_runtime_results.git_identity", return_value=("a" * 40, "b" * 40, True)):
            summary = aggregate(self._args())
        self.assertEqual(summary["state"], "PASS")
        self.assertEqual(summary["expected_rows"], list(EXPECTED_ROWS))
        self.assertEqual([row["state"] for row in summary["rows"]], ["PASS", "PASS"])
        hashes = write_summary(self.output_dir, summary)
        self.assertEqual(len(hashes["sha256"]), 64)
        self.assertEqual(json.loads((self.output_dir / "aggregate.json").read_text()), summary)
        self.assertEqual(
            (self.output_dir / "aggregate.json.sha256").read_text(),
            f"{hashes['sha256']}  aggregate.json\n",
        )
        validate_against_schema(summary, read_json(ROOT / "ci/schema/hip-runtime-aggregate-v1.schema.json"), "aggregate")
        self.assertTrue((self.output_dir / "aggregate.json.sha256").is_file())

    def test_hard_exit_after_first_payload_link_cannot_leave_orphan_sidecar(self) -> None:
        script = f"""
import os
import sys
sys.path.insert(0, {str(ROOT / 'ci/tools')!r})
import aggregate_h3_public_runtime_results as aggregate
from pathlib import Path
real_link = aggregate._link_fd_no_replace
def hard_exit(source_fd, output_fd, name):
    real_link(source_fd, output_fd, name)
    if name == 'aggregate.json':
        os._exit(73)
aggregate._link_fd_no_replace = hard_exit
aggregate.write_summary(Path({str(self.output_dir)!r}), {{'state': 'PASS'}})
"""
        result = subprocess.run([sys.executable, "-c", script], check=False)
        self.assertEqual(result.returncode, 73)
        self.assertTrue((self.output_dir / "aggregate.json").is_file())
        self.assertFalse((self.output_dir / "aggregate.json.sha256").exists())
        self.assertEqual(json.loads((self.output_dir / "aggregate.json").read_text()), {"state": "PASS"})

    def test_post_link_exception_before_bookkeeping_cleans_owned_payload(self) -> None:
        real_link = aggregate_module._link_fd_no_replace

        def raise_after_link(source_fd: int, output_fd: int, name: str) -> None:
            real_link(source_fd, output_fd, name)
            if name == "aggregate.json":
                raise RuntimeError("forced exception after successful link")

        with patch("aggregate_h3_public_runtime_results._link_fd_no_replace", side_effect=raise_after_link):
            with self.assertRaises(RuntimeError):
                write_summary(self.output_dir, {"state": "PASS"})
        self.assertEqual(list(self.output_dir.iterdir()), [])

    def test_exact_path_iterdir_swap_is_rejected_without_external_write(self) -> None:
        outside = self.root / "outside-iterdir"
        outside.mkdir()
        moved = self.root / "aggregate-original"
        original_iterdir = Path.iterdir

        def swap_after_validation(path: Path):
            iterator = original_iterdir(path)
            if path == self.output_dir:
                path.rename(moved)
                path.symlink_to(outside, target_is_directory=True)
            return iterator

        try:
            with patch.object(Path, "iterdir", swap_after_validation):
                with self.assertRaises(ContractError):
                    write_summary(self.output_dir, {"state": "PASS"}, workspace_root=self.root)
            self.assertFalse((outside / "aggregate.json").exists())
            self.assertFalse((outside / "aggregate.json.sha256").exists())
        finally:
            if self.output_dir.is_symlink():
                self.output_dir.unlink()
            if moved.exists():
                moved.rename(self.output_dir)

    def test_concurrent_output_leaf_replacement_is_rejected_without_external_write(self) -> None:
        outside = self.root / "outside-concurrent"
        outside.mkdir()
        moved = self.root / "aggregate-concurrent-original"
        listed = threading.Event()
        replaced = threading.Event()
        replacement_errors: list[BaseException] = []
        original_listdir = os.listdir

        def replace_leaf() -> None:
            try:
                if not listed.wait(timeout=5):
                    raise AssertionError("aggregate publication did not reach the FD emptiness check")
                self.output_dir.rename(moved)
                self.output_dir.symlink_to(outside, target_is_directory=True)
                replaced.set()
            except BaseException as exc:  # report thread failures in the test process
                replacement_errors.append(exc)

        def pause_after_fd_listing(path):
            result = original_listdir(path)
            if isinstance(path, int) and not listed.is_set():
                listed.set()
                if not replaced.wait(timeout=5):
                    raise AssertionError("concurrent output replacement did not complete")
            return result

        replacer = threading.Thread(target=replace_leaf, name="h3-output-replacer")
        replacer.start()
        try:
            with patch("aggregate_h3_public_runtime_results.os.listdir", side_effect=pause_after_fd_listing):
                with self.assertRaises(ContractError):
                    write_summary(self.output_dir, {"state": "PASS"}, workspace_root=self.root)
            replacer.join(timeout=5)
            self.assertFalse(replacer.is_alive())
            self.assertEqual(replacement_errors, [])
            self.assertFalse((outside / "aggregate.json").exists())
            self.assertFalse((outside / "aggregate.json.sha256").exists())
        finally:
            if replacer.is_alive():
                replacer.join(timeout=5)
            if self.output_dir.is_symlink():
                self.output_dir.unlink()
            if moved.exists():
                moved.rename(self.output_dir)

    def test_output_leaf_and_ancestor_races_and_material_are_rejected(self) -> None:
        outside = self.root / "outside-leaf"
        outside.mkdir()
        self.output_dir.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ContractError):
            write_summary(self.output_dir, {"state": "PASS"}, workspace_root=self.root)
        self.output_dir.unlink()

        with self.assertRaises(ContractError):
            write_summary(self.root / "missing-parent" / "aggregate", {"state": "PASS"}, workspace_root=self.root)

        self.output_dir.mkdir()
        unexpected = self.output_dir / "unexpected"
        unexpected.write_bytes(b"material")
        with self.assertRaises(ContractError):
            write_summary(self.output_dir, {"state": "PASS"}, workspace_root=self.root)
        self.assertEqual(unexpected.read_bytes(), b"material")

        unexpected.unlink()
        aggregate = self.output_dir / "aggregate.json"
        aggregate.write_bytes(b"pre-existing")
        with self.assertRaises(ContractError):
            write_summary(self.output_dir, {"state": "PASS"}, workspace_root=self.root)
        self.assertEqual(aggregate.read_bytes(), b"pre-existing")

    def test_parent_symlink_attacks_are_rejected_for_all_external_paths(self) -> None:
        parent_alias = self.root / "workspace-alias"
        parent_alias.symlink_to(self.root, target_is_directory=True)
        mutations = (
            ("artifact_dir", parent_alias / "rows", "artifact directory parent symlink"),
            ("output_dir", parent_alias / "aggregate", "output directory parent symlink"),
            ("needs_json", parent_alias / "needs.json", "needs JSON parent symlink"),
        )
        for attribute, value, label in mutations:
            with self.subTest(label=label):
                args = self._args()
                setattr(args, attribute, value)
                with patch("aggregate_h3_public_runtime_results.git_identity", return_value=("a" * 40, "b" * 40, True)), self.assertRaises(ContractError):
                    aggregate(args)

    def test_workspace_escape_is_rejected_for_all_external_paths(self) -> None:
        escaped = ROOT.parent / f"{self.root.name}-outside"
        mutations = (
            ("artifact_dir", escaped / "rows", "artifact directory escape"),
            ("output_dir", escaped / "aggregate", "output directory escape"),
            ("needs_json", escaped / "needs.json", "needs JSON escape"),
        )
        for attribute, value, label in mutations:
            with self.subTest(label=label):
                args = self._args()
                setattr(args, attribute, value)
                with patch("aggregate_h3_public_runtime_results.git_identity", return_value=("a" * 40, "b" * 40, True)), self.assertRaises(ContractError):
                    aggregate(args)

    def test_missing_duplicate_unknown_and_extra_artifacts_fail_closed(self) -> None:
        mutations = (
            (lambda: shutil.rmtree(self.artifact_dir / "h3-public-gfx1030"), "missing row"),
            (lambda: shutil.copytree(self.artifact_dir / "h3-public-gfx1030", self.artifact_dir / "h3-public-gfx1030-copy"), "duplicate row"),
            (lambda: (self.artifact_dir / "unknown-row").mkdir(), "unknown row"),
            (lambda: (self.artifact_dir / "h3-public-gfx1030" / "extra.bin").write_bytes(b"extra"), "extra output"),
            (lambda: (self.artifact_dir / "h3-public-gfx1030" / "extra-directory").mkdir(), "extra row-root directory"),
            (lambda: (self.artifact_dir / "h3-public-gfx1030" / "extra-symlink").symlink_to(self.artifact_dir / "h3-public-gfx1030" / "report.json"), "extra row-root symlink"),
            (lambda: (self.artifact_dir / "h3-public-gfx1030" / "build" / "extra-directory").mkdir(), "extra build directory"),
            (lambda: (self.artifact_dir / "h3-public-gfx1030" / "build" / "extra-symlink").symlink_to(self.artifact_dir / "h3-public-gfx1030" / "report.json"), "extra build symlink"),
        )
        for mutation, label in mutations:
            with self.subTest(label=label):
                mutation()
                with patch("aggregate_h3_public_runtime_results.git_identity", return_value=("a" * 40, "b" * 40, True)), self.assertRaises(ContractError):
                    aggregate(self._args())
                self.tearDown()
                self.setUp()

    def test_needs_json_is_bounded_local_regular_and_exact(self) -> None:
        mutations = (
            ({"state": "success", "rows": list(EXPECTED_ROWS)}, "success is not PASS"),
            ({"state": "PASS", "rows": list(reversed(EXPECTED_ROWS))}, "rows are not canonical order"),
            ({"state": "PASS", "rows": None}, "rows are missing"),
            ({"state": "PASS", "rows": list(EXPECTED_ROWS), "extra": True}, "extra needs field"),
        )
        for value, label in mutations:
            with self.subTest(label=label):
                self.needs.write_text(json.dumps(value) + "\n", encoding="utf-8")
                with patch("aggregate_h3_public_runtime_results.git_identity", return_value=("a" * 40, "b" * 40, True)), self.assertRaises(ContractError):
                    aggregate(self._args())

        self.needs.write_bytes(b"x" * (16 * 1024 + 1))
        with patch("aggregate_h3_public_runtime_results.git_identity", return_value=("a" * 40, "b" * 40, True)), self.assertRaises(ContractError):
            aggregate(self._args())

        self.needs.unlink()
        self.needs.mkdir()
        with patch("aggregate_h3_public_runtime_results.git_identity", return_value=("a" * 40, "b" * 40, True)), self.assertRaises(ContractError):
            aggregate(self._args())
        self.needs.rmdir()
        needs_target = self.root / "needs-target.json"
        needs_target.write_text(json.dumps({"state": "PASS", "rows": list(EXPECTED_ROWS)}) + "\n", encoding="utf-8")
        self.needs.symlink_to(needs_target)
        with patch("aggregate_h3_public_runtime_results.git_identity", return_value=("a" * 40, "b" * 40, True)), self.assertRaises(ContractError):
            aggregate(self._args())

    def test_identity_needs_and_scope_tampering_fail_closed(self) -> None:
        for argument, value, label in (
            ("reviewed_sha", "c" * 40, "reviewed identity"),
            ("tree_oid", "c" * 40, "tree identity"),
        ):
            with self.subTest(label=label):
                args = self._args()
                setattr(args, argument, value)
                with patch("aggregate_h3_public_runtime_results.git_identity", return_value=("a" * 40, "b" * 40, True)), self.assertRaises(ContractError):
                    aggregate(args)

        self.needs.write_text(json.dumps({"state": "FAIL", "rows": list(EXPECTED_ROWS)}) + "\n", encoding="utf-8")
        with patch("aggregate_h3_public_runtime_results.git_identity", return_value=("a" * 40, "b" * 40, True)), self.assertRaises(ContractError):
            aggregate(self._args())

        metadata_path = self.artifact_dir / "h3-public-gfx1030" / "hip-runtime-artifact.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["scope"]["gpu_execution"] = True
        metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
        with patch("aggregate_h3_public_runtime_results.git_identity", return_value=("a" * 40, "b" * 40, True)), self.assertRaises(ContractError):
            aggregate(self._args())

    def test_dirty_checkout_is_rejected_before_aggregation(self) -> None:
        with patch("aggregate_h3_public_runtime_results.git_identity", return_value=("a" * 40, "b" * 40, False)), self.assertRaises(ContractError):
            aggregate(self._args())


if __name__ == "__main__":
    unittest.main()
