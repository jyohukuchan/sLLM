"""Host contracts for the deterministic Stage9 vocabulary generator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "qwen38_draft_vocab", Path(__file__).resolve().parents[1] / "tools/qwen38_draft_vocab.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DraftVocabTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers
        except ImportError as exc:  # pragma: no cover - CI installs the host tool set.
            self.skipTest(f"tokenizers unavailable: {exc}")
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        tokenizer = Tokenizer(models.WordLevel(
            {"[UNK]": 0, "alpha": 1, "beta": 2, "gamma": 3, "<special>": 4},
            unk_token="[UNK]",
        ))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.add_special_tokens(["<special>"])
        self.tokenizer_path = self.root / "tokenizer.json"
        tokenizer.save(str(self.tokenizer_path))

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _manifest(self, domains: list[dict], *, vocab_size: int = 2) -> Path:
        document = {
            "schema_version": MODULE.SCHEMA_VERSION,
            "tokenizer": {"path": str(self.tokenizer_path), "sha256": self._sha(self.tokenizer_path)},
            "vocab_size": vocab_size,
            "required_special_tokens": ["<special>"],
            "overlap_exclusions": [{
                "dataset": "mtp-bench-v1", "revision": "r1", "file": "inputs.jsonl",
                "sha256": "0" * 64, "reason": "benchmark inputs are excluded",
            }],
            "domains": domains,
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        return path

    def _source(self, name: str, text: str, *, dataset: str = "dataset", **kwargs: object) -> dict:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        result = {
            "dataset": dataset,
            "revision": "revision-1",
            "file": name,
            "path": str(path),
            "sha256": self._sha(path),
            "format": "text",
            "max_bytes": 10000,
            "max_rows": 100,
        }
        result.update(kwargs)
        return result

    @staticmethod
    def _read_ids(path: Path) -> list[int]:
        raw = path.read_bytes()
        return [value[0] for value in struct.iter_unpack("<I", raw)]

    def test_deterministic_ranking_and_special_ids_are_sorted(self) -> None:
        # alpha, beta, and gamma have equal aggregate score; lower ID wins.
        domains = [
            {"name": "a", "files": [self._source("a.txt", "alpha beta")]},
            {"name": "b", "files": [self._source("b.txt", "gamma beta")]},
        ]
        manifest = self._manifest(domains, vocab_size=3)
        first = self.root / "first.u32"
        second = self.root / "second.u32"
        first_meta = MODULE.generate_vocabulary(manifest, first)
        second_meta = MODULE.generate_vocabulary(manifest, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(first_meta, second_meta)
        self.assertEqual(self._read_ids(first), [1, 2, 4])
        self.assertEqual(first_meta["vocab_count"], 3)
        self.assertEqual(first_meta["special_token_ids"], [4])
        self.assertEqual(first_meta["vocab_sha256"], self._sha(first))

    def test_domains_are_normalized_before_equal_weighting(self) -> None:
        # Pooled counts would rank beta (9/11) above alpha. Equal domain
        # weighting ranks alpha (1.0 in its domain) above beta (0.9).
        scores = MODULE.score_domains([{1: 1}, {2: 9, 3: 1}])
        self.assertAlmostEqual(scores[1], 0.5)
        self.assertAlmostEqual(scores[2], 0.45)
        self.assertEqual(sorted(scores, key=lambda token: (-scores[token], token)), [1, 2, 3])

    def test_missing_or_wrong_hash_fails_closed(self) -> None:
        source = self._source("source.txt", "alpha")
        source["sha256"] = "1" * 64
        manifest = self._manifest([{"name": "a", "files": [source]}])
        with self.assertRaisesRegex(MODULE.ManifestError, "SHA-256 mismatch"):
            MODULE.generate_vocabulary(manifest, self.root / "out.u32")

        malformed = json.loads(manifest.read_text(encoding="utf-8"))
        del malformed["overlap_exclusions"]
        manifest.write_text(json.dumps(malformed), encoding="utf-8")
        with self.assertRaisesRegex(MODULE.ManifestError, "overlap_exclusions"):
            MODULE.load_manifest(manifest)

    def test_overlap_identity_and_row_exclusion_are_enforced(self) -> None:
        source = self._source("excluded.txt", "alpha\nbeta\n")
        source["exclude_text_sha256"] = [hashlib.sha256(b"alpha").hexdigest()]
        manifest = self._manifest([{"name": "a", "files": [source]}])
        metadata = MODULE.generate_vocabulary(manifest, self.root / "out.u32")
        self.assertEqual(metadata["domains"][0]["files"][0]["excluded_rows"], 1)

        overlapping = dict(source)
        overlapping["dataset"] = "mtp-bench-v1"
        overlapping["revision"] = "r1"
        overlapping["file"] = "inputs.jsonl"
        overlapping["sha256"] = "0" * 64
        overlap_manifest = self._manifest([{"name": "a", "files": [overlapping]}])
        with self.assertRaisesRegex(MODULE.ManifestError, "overlap exclusion"):
            MODULE.load_manifest(overlap_manifest)

    def test_parquet_uses_selected_columns_filters_and_row_bound(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as parquet
        except ImportError as exc:  # pragma: no cover - CI installs parquet support for the tool.
            self.skipTest(f"pyarrow unavailable: {exc}")
        parquet_path = self.root / "rows.parquet"
        parquet.write_table(pa.table({
            "content": ["alpha", "beta", "gamma"],
            "role": ["assistant", "user", "assistant"],
            "repo_id": ["external", "external", "excluded"],
            "unused": ["must not be selected"] * 3,
        }), parquet_path)
        source = {
            "dataset": "dataset", "revision": "revision-1", "file": "rows.parquet",
            "path": str(parquet_path), "sha256": self._sha(parquet_path), "format": "parquet",
            "columns": ["content"],
            "filters": [
                {"column": "role", "op": "eq", "value": "assistant"},
                {"column": "repo_id", "op": "not_in", "value": ["excluded"]},
            ],
            "max_bytes": 10000, "max_rows": 2,
        }
        manifest = self._manifest([{"name": "chat", "files": [source]}], vocab_size=1)
        metadata = MODULE.generate_vocabulary(manifest, self.root / "chat.u32")
        file_stats = metadata["domains"][0]["files"][0]
        self.assertEqual(file_stats["rows"], 1)
        self.assertEqual(file_stats["scanned_rows"], 1)
        self.assertEqual(self._read_ids(self.root / "chat.u32"), [4])


if __name__ == "__main__":
    unittest.main()
