"""Host contracts for the tracked Stage9 corpus reconstruction recipe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "qwen38_draft_vocab_sources", ROOT / "ci/tools/qwen38_draft_vocab_sources.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DraftVocabSourceTests(unittest.TestCase):
    def test_lock_has_fixed_order_targets_and_expected_digests(self) -> None:
        lock = json.loads((ROOT / "ci/matrix/qwen38-draft-vocab-source-lock-v1.json").read_text())
        self.assertEqual(lock["schema_version"], MODULE.SCHEMA_VERSION)
        self.assertEqual(
            [item["name"] for item in lock["repositories"]],
            list(MODULE.REPOSITORY_ORDER),
        )
        self.assertEqual(
            [lock["code_categories"][name]["target_bytes"] for name in ("code_py", "code_cpp", "code_other")],
            [40_000_000, 50_000_000, 15_000_000],
        )
        self.assertEqual(lock["manifest_sha256"], "3c6fc65c10eee31d86ccd7a2e54e58eaa9282dfea68d7861fe171bad28329ab4")
        self.assertEqual(lock["candidate_vocab_sha256"], "24bff6b41785a7729bff183dfea7997e6446173e0df7254cc5761a7519fdebd0")

    def test_git_candidates_are_utf8_bounded_nonblank_and_hash_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            (repo / "good.py").write_text("print('ok')\n", encoding="utf-8")
            (repo / "blank.py").write_text(" \n\t\n", encoding="utf-8")
            (repo / "large.py").write_text("x" * 32, encoding="utf-8")
            (repo / "bad.py").write_bytes(b"\xff\xfe")
            (repo / "ignored.txt").write_text("ignored", encoding="utf-8")
            subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
                cwd=repo,
                check=True,
            )
            category = {
                "suffixes": [".py"],
                "max_source_bytes": 16,
                "max_rows": 100,
                "repository_revision": "revision-1",
            }
            candidates = MODULE._candidate_paths("fixture", repo, category)
            self.assertEqual([item["file"] for item in candidates], ["good.py"])
            self.assertEqual(candidates[0]["max_bytes"], len(b"print('ok')\n"))

    def test_selection_hash_includes_order_and_source_identity(self) -> None:
        first = {
            "dataset": "repo",
            "revision": "r1",
            "file": "a.py",
            "sha256": "a" * 64,
            "max_bytes": 1,
        }
        second = {**first, "file": "b.py", "sha256": "b" * 64}
        self.assertNotEqual(MODULE._selection_hash([first, second]), MODULE._selection_hash([second, first]))
        self.assertEqual(MODULE._selection_hash([first, second]), MODULE._selection_hash([first, second]))


if __name__ == "__main__":
    unittest.main()
