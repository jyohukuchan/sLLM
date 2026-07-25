"""Regression tests for the SQ8_0 paged-decode full-model gate evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "tools" / "evaluate-sq8_0-paged-decode-tile-gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sq8_paged_decode_tile_gate", EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f32le(path: Path, values: list[float]) -> str:
    payload = b"".join(struct.pack("<f", value) for value in values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class EvaluateSq8PagedDecodeTileGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        # The production evaluator is deliberately fixed to the real model
        # shapes.  Tiny synthetic vectors keep these structural tests quick.
        self.module.FINAL_HIDDEN_ELEMENTS = 2
        self.module.LOGITS_ELEMENTS = 4

    @staticmethod
    def criteria() -> dict:
        return {
            "schema_version": "ullm.sq8_0.paged_decode_tile_gate_criteria.v1",
            "frozen_before_measurement": True,
            "reference_route": "direct",
            "candidate_routes": ["tile128", "tile256"],
            "generation": {"max_new_tokens": 4, "captured_decode_indices": [1, 2, 3]},
            "thresholds": {
                "max_abs": 2.0e-5,
                "relative_l2": 1.0e-5,
                "cosine_similarity": 0.999999,
                "all_values_must_be_finite": True,
                "token_exact_match": True,
            },
            "case_groups": [
                {
                    "name": "boundary",
                    "requests": [
                        {
                            "request_id": "request-a",
                            "prompt_tokens": 127,
                            "decode_cache_lengths": [128, 129, 130],
                        },
                        {
                            "request_id": "request-b",
                            "prompt_tokens": 128,
                            "decode_cache_lengths": [129, 130, 131],
                        },
                    ],
                }
            ],
        }

    def write_route(self, root: Path, route: str, *, token_delta: int = 0) -> None:
        requests = []
        for request_index, expected in enumerate(self.criteria()["case_groups"][0]["requests"]):
            oracle = root / "cases" / route / "boundary" / "oracle"
            generated = [0, 1, 2, 1]
            if token_delta:
                generated[2] += token_delta
            captures = []
            for generated_index, cache_len in enumerate(
                expected["decode_cache_lengths"], start=1
            ):
                prefix = f"{expected['request_id']}-g{generated_index}"
                hidden = oracle / f"{prefix}-hidden.f32le"
                logits = oracle / f"{prefix}-logits.f32le"
                hidden_hash = f32le(hidden, [1.0 + generated_index, float(request_index)])
                logits_values = [0.0, 0.0, 0.0, 0.0]
                logits_values[generated[generated_index]] = 10.0
                logits_hash = f32le(logits, logits_values)
                captures.append(
                    {
                        "generated_index": generated_index,
                        "cache_len": cache_len,
                        "position": cache_len - 1,
                        "top1_token_id": generated[generated_index],
                        "final_hidden_file": str(hidden.relative_to(oracle.parent)),
                        "final_hidden_f32_le_sha256": hidden_hash,
                        "logits_file": str(logits.relative_to(oracle.parent)),
                        "logits_f32_le_sha256": logits_hash,
                    }
                )
            prompt_tokens = list(range(1, expected["prompt_tokens"] + 1))
            requests.append(
                {
                    "request_id": expected["request_id"],
                    "prompt_token_ids": prompt_tokens,
                    "max_new_tokens": 4,
                    "generated_token_ids": generated,
                    "generated_steps": [
                        {
                            "generated_index": index,
                            "token_id": token,
                            "cache_len": len(prompt_tokens) + index,
                        }
                        for index, token in enumerate(generated)
                    ],
                    "decode_oracle_captures": captures,
                }
            )
        document = {
            "passed": True,
            "prefill_mode": "m128-chunk128",
            "prefill_chunk_tokens": 128,
            "paged_decode_split_source_tile": None if route == "direct" else int(route[4:]),
            "device": {"gcn_arch_name": "gfx1201"},
            "cancelled_request": None,
            "requests": requests,
        }
        result = root / "cases" / route / "boundary" / "result.json"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps(document), encoding="utf-8")

    def evaluate(self, root: Path) -> tuple[int, dict]:
        previous_argv = sys.argv
        try:
            sys.argv = [str(EVALUATOR), "--result-dir", str(root)]
            with redirect_stdout(io.StringIO()):
                status = self.module.main()
        finally:
            sys.argv = previous_argv
        return status, json.loads((root / "summary.json").read_text(encoding="utf-8"))

    def prepare_root(self, root: Path, *, tile128_delta: int = 0) -> None:
        (root / "gate-criteria.json").write_text(
            json.dumps(self.criteria()), encoding="utf-8"
        )
        self.write_route(root, "direct")
        self.write_route(root, "tile128", token_delta=tile128_delta)
        self.write_route(root, "tile256")

    def test_accepts_exact_full_vector_and_token_matches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare_root(root)
            status, summary = self.evaluate(root)
            self.assertEqual(status, 0)
            self.assertTrue(summary["passed"])
            self.assertTrue(summary["routes"]["tile128"]["token_exact_match"])
            self.assertEqual(summary["routes"]["tile256"]["vector_comparison_count"], 12)

    def test_rejects_a_greedy_token_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare_root(root, tile128_delta=1)
            status, summary = self.evaluate(root)
            self.assertEqual(status, 2)
            self.assertFalse(summary["passed"])
            self.assertFalse(summary["routes"]["tile128"]["token_exact_match"])

    def test_rejects_a_relaxed_pre_window_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare_root(root)
            criteria = self.criteria()
            criteria["thresholds"]["max_abs"] = 1.0e-4
            (root / "gate-criteria.json").write_text(json.dumps(criteria), encoding="utf-8")
            status, summary = self.evaluate(root)
            self.assertEqual(status, 2)
            self.assertFalse(summary["passed"])
            self.assertIn("weaker", summary["failures"][0])


if __name__ == "__main__":
    unittest.main()
