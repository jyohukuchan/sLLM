from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_aq4_p2_input_controls", ROOT / "tools/audit_aq4_p2_input_controls.py"
)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def write_oracle(root: Path, greedy: int) -> None:
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "fixture-oracle",
                "ranking": {
                    "greedy": "maximum_logit_then_smallest_token_id",
                    "topk": "logit_descending_then_token_id_ascending",
                },
                "cases": [
                    {
                        "case_id": "c",
                        "prompt_token_count": 2,
                        "prompt_token_ids_sha256": AUDIT.token_hash([11, 12]),
                        "step_count": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "payload.jsonl").write_text(
        json.dumps(
            {
                "case_id": "c",
                "step": 0,
                "greedy_token_id": greedy,
                "topk": [{"token_id": greedy, "logit": 3.0}],
            }
        )
        + "\n",
        encoding="utf-8",
    )


class Aq4P2InputControlTests(unittest.TestCase):
    def test_compact_oracle_without_context_fields_is_not_misclassified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            path = root / "path"
            write_oracle(source, 7)
            write_oracle(path, 8)
            report = AUDIT.audit_oracles(source, path)
            self.assertEqual(report["context_mismatch_rows"], 0)
            self.assertEqual(report["greedy_mismatch_rows"], 1)
            self.assertFalse(report["rows"][0]["context_length_observed"])
            self.assertFalse(report["rows"][0]["context_token_ids_hash_observed"])

    def test_real_control_fixtures_match_calibration_tokens_and_causal_layout(self) -> None:
        report = AUDIT.audit_calibration_and_fixtures(
            ROOT / "benchmarks/workloads/qwen35-aq4-p2-source-calibration-cases-v0.1.json",
            ROOT / "tests/fixtures/aq4-p2-input-controls/pure-prefill.json",
            ROOT / "tests/fixtures/aq4-p2-input-controls/gateway-request.json",
        )
        self.assertTrue(report["token_ids_equal_across_controls"])
        self.assertTrue(report["pure_prefill"]["position_ids_exact"])
        self.assertTrue(report["pure_prefill"]["causal_mask_exact"])
        self.assertTrue(report["gateway_fixture"]["generated_step_count_exact"])
        self.assertTrue(report["gateway_fixture"]["decode_positions_exact"])
        self.assertTrue(report["gateway_fixture"]["context_length_exact"])
        self.assertTrue(report["vocab_slicing_exact"])

    def test_cpu_matvec_reference_matches_runtime_fixture(self) -> None:
        report = AUDIT.audit_matvec()
        self.assertTrue(report["exact"])
        self.assertTrue(report["batch_reference_exact"])
        self.assertEqual(report["actual"], [112.5, 30.0])


if __name__ == "__main__":
    unittest.main()
