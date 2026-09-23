"""Focused host tests for the Stage 9 compact draft-vocabulary evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "mtp_expected_acceptance",
    Path(__file__).resolve().parents[1] / "tools/mtp_expected_acceptance.py",
)
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MtpExpectedAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_vocab = MODULE.VOCAB
        MODULE.VOCAB = 8
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        MODULE.VOCAB = self.original_vocab
        self.temp.cleanup()

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _map(self, ids: list[int]) -> tuple[Path, np.ndarray]:
        path = self.root / "mtp-draft-vocab-98304.u32"
        np.asarray(ids, dtype="<u4").tofile(path)
        metadata = {
            "schema_version": MODULE.VOCAB_MAP_SCHEMA,
            "N": len(ids),
            "vocab_count": len(ids),
            "vocab_sha256": self._sha(path),
            "manifest_sha256": "1" * 64,
            "tokenizer_sha256": "2" * 64,
            "special_token_ids": [ids[0]],
            "domains": [],
        }
        path.with_suffix(".metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return path, np.asarray(ids, dtype=np.uint32)

    def _entry(
        self,
        draft: np.ndarray,
        target: np.ndarray,
        *,
        draft_keys: list[tuple[int, int]] | None = None,
        target_keys: list[tuple[int, int]] | None = None,
    ) -> dict:
        draft_path = self.root / f"draft-{draft.shape[0]}-{draft.shape[1]}.f32"
        target_path = self.root / f"target-{target.shape[0]}-{target.shape[1]}.f32"
        draft.astype("<f4").tofile(draft_path)
        target.astype("<f4").tofile(target_path)
        draft_keys = draft_keys or [(index, 0) for index in range(draft.shape[0])]
        target_keys = target_keys or [(index, 0) for index in range(target.shape[0])]
        return {
            "case_id": "synthetic",
            "draft_logits_file": str(draft_path),
            "draft_logits_sha256": f"sha256:{self._sha(draft_path)}",
            "target_logits_file": str(target_path),
            "target_logits_sha256": f"sha256:{self._sha(target_path)}",
            "draft_logit_rows": [
                {"sequence_index": sequence, "block_row": block}
                for sequence, block in draft_keys
            ],
            "target_logit_rows": [
                {"sequence_index": sequence, "block_row": block}
                for sequence, block in target_keys
            ],
        }

    def test_compact_support_remaps_global_ids_and_preserves_tie_order(self) -> None:
        row = np.asarray([4.0, 4.0, 4.0], dtype=np.float32)
        support = MODULE.support(row, top_k=2, top_p=1.0, token_ids=np.asarray([1, 5, 7]))
        self.assertEqual(list(support), [1, 5])

        # The target uses global rows 1 and 5. Comparing compact local IDs
        # (0 and 1) would incorrectly make this a disjoint distribution.
        target = np.full((1, 8), -20.0, dtype=np.float32)
        target[0, [1, 5, 7]] = [4.0, 4.0, 4.0]
        candidate = row.reshape(1, 3)
        entry = self._entry(candidate, target)
        _, vocab_map = self._map([1, 5, 7])
        rows = MODULE.run_rows(entry, top_k=2, top_p=1.0, candidate_draft_vocab_map=vocab_map)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0][1], 1.0, places=12)

    def test_identity_map_matches_full_vocab_baseline(self) -> None:
        target = np.full((2, 8), -20.0, dtype=np.float32)
        target[0, [0, 3, 6]] = [5.0, 4.0, 3.0]
        target[1, [1, 2, 7]] = [5.0, 4.0, 3.0]
        entry = self._entry(target.copy(), target.copy())
        baseline = MODULE.run_rows(entry, top_k=3, top_p=0.95)
        map_path, vocab_map = self._map(list(range(8)))
        self.assertEqual(MODULE.load_candidate_draft_vocab_map(map_path).tolist(), list(range(8)))
        mapped = MODULE.run_rows(entry, top_k=3, top_p=0.95, candidate_draft_vocab_map=vocab_map)
        self.assertEqual([row[0] for row in baseline], [row[0] for row in mapped])
        for left, right in zip(baseline, mapped):
            self.assertAlmostEqual(left[1], right[1], places=12)

    def test_prompt_rows_can_differ_without_cross_prompt_pairing(self) -> None:
        target = np.full((2, 8), -20.0, dtype=np.float32)
        target[:, 0] = 5.0
        draft = np.full((1, 3), -20.0, dtype=np.float32)
        draft[0, 0] = 5.0
        entry = self._entry(
            draft,
            target,
            draft_keys=[(4, 0)],
            target_keys=[(4, 0), (4, 1)],
        )
        _, vocab_map = self._map([0, 2, 5])
        rows = MODULE.run_rows(entry, top_k=1, top_p=1.0, candidate_draft_vocab_map=vocab_map)
        self.assertEqual(rows, [(0, 1.0)])

    def test_map_metadata_and_range_validation_fail_closed(self) -> None:
        path, _ = self._map([0, 2, 5])
        metadata = json.loads(path.with_suffix(".metadata.json").read_text())
        metadata["N"] = 2
        path.with_suffix(".metadata.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "N must equal"):
            MODULE.load_candidate_draft_vocab_map(path)

        path.write_bytes(np.asarray([0, 0, 5], dtype="<u4").tobytes())
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            MODULE.load_candidate_draft_vocab_map(path)

        path.write_bytes(np.asarray([0, 2, 8], dtype="<u4").tobytes())
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            MODULE.load_candidate_draft_vocab_map(path)

    def test_report_shape_and_hash_are_checked(self) -> None:
        draft = np.zeros((1, 3), dtype=np.float32)
        target = np.zeros((1, 8), dtype=np.float32)
        entry = self._entry(draft, target)
        _, vocab_map = self._map([0, 2, 5])
        Path(entry["draft_logits_file"]).write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            MODULE.run_rows(entry, top_k=1, top_p=1.0, candidate_draft_vocab_map=vocab_map)

        self.assertNotEqual(vocab_map.size, 0)


if __name__ == "__main__":
    unittest.main()
