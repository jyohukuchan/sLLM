from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "quarantine-invalid-sq8-source-scale-results.py"


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "quarantine_invalid_sq8_source_scale_results",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def invalid_row(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "status": "ok",
        "engine": {"name": TOOL.TARGET_ENGINE},
        "model": {"name": TOOL.TARGET_MODEL},
        "workload": {"sq_artifact": TOOL.TARGET_ARTIFACT},
    }


class QuarantineInvalidSq8SourceScaleResultsTests(unittest.TestCase):
    def test_quarantine_row_marks_only_matching_ullm_row(self) -> None:
        target = invalid_row("target")
        vllm = invalid_row("vllm")
        vllm["engine"]["name"] = "vLLM"

        self.assertTrue(TOOL.quarantine_row(target))
        self.assertFalse(TOOL.quarantine_row(vllm))
        self.assertEqual(target["status"], "ok")
        self.assertEqual(target["result_validity"], TOOL.QUARANTINE_VALIDITY)
        self.assertNotIn("result_validity", vllm)

    def test_quarantine_row_is_idempotent_and_rejects_conflict(self) -> None:
        target = invalid_row("target")
        self.assertTrue(TOOL.quarantine_row(target))
        self.assertFalse(TOOL.quarantine_row(target))
        target["result_validity"] = {"state": "valid"}
        with self.assertRaisesRegex(ValueError, "conflicting result_validity"):
            TOOL.quarantine_row(target)

    def test_migrate_checks_inventory_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "2026-07-09" / "case" / "results.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(invalid_row("target")) + "\n", encoding="utf-8")

            with patch.object(TOOL, "EXPECTED_MATCHES", 1), patch.object(
                TOOL, "EXPECTED_FILES", 1
            ):
                dry_run = TOOL.migrate(root, apply=False)
                self.assertEqual(dry_run["modified_rows"], 1)
                self.assertNotIn(
                    "result_validity",
                    json.loads(path.read_text(encoding="utf-8")),
                )
                applied = TOOL.migrate(root, apply=True)

            self.assertEqual(applied["matched_rows"], 1)
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(row["result_validity"], TOOL.QUARANTINE_VALIDITY)

    def test_repository_inventory_has_no_unmarked_matching_rows(self) -> None:
        matched = []
        for path in TOOL.result_paths(REPO_ROOT / "benchmarks" / "results"):
            for row in TOOL.load_jsonl(path):
                if TOOL.matches_invalid_source_scale_row(row):
                    matched.append((path, row))

        self.assertEqual(len(matched), TOOL.EXPECTED_MATCHES)
        self.assertEqual(len({path for path, _row in matched}), TOOL.EXPECTED_FILES)
        self.assertEqual(
            [
                (str(path), row.get("case_id"))
                for path, row in matched
                if row.get("result_validity") != TOOL.QUARANTINE_VALIDITY
            ],
            [],
        )


if __name__ == "__main__":
    unittest.main()
