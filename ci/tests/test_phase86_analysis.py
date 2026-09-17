"""Small numerical contracts for the Phase86 prompt-level analysis."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "phase86_analysis", Path(__file__).resolve().parents[1] / "tools/analyze_phase86_mtp_catch_up.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PromptClusterTests(unittest.TestCase):
    def test_identical_pairs_have_zero_interval_and_unit_p(self):
        result = MODULE.paired_cluster([0.0] * 26, samples=511)
        self.assertEqual(result["paired_bootstrap_95pct"], [0.0, 0.0])
        self.assertEqual(result["sign_flip_two_sided_p"], 1.0)
        self.assertEqual(result["prompt_clusters"], 26)

    def test_constant_positive_pairs_and_label_swap(self):
        positive = MODULE.paired_cluster([0.125] * 26, samples=1023)
        negative = MODULE.paired_cluster([-0.125] * 26, samples=1023)
        self.assertEqual(positive["paired_bootstrap_95pct"], [0.125, 0.125])
        self.assertEqual(negative["paired_bootstrap_95pct"], [-0.125, -0.125])
        self.assertEqual(positive["sign_flip_two_sided_p"], negative["sign_flip_two_sided_p"])
        self.assertLess(positive["sign_flip_two_sided_p"], .01)

    def test_prompt_means_are_equally_weighted(self):
        result = MODULE.paired_cluster([.25, -.125, 0], samples=511)
        self.assertAlmostEqual(result["mean_difference"], 1 / 24)
        self.assertEqual(result["prompt_clusters"], 3)

    def test_single_prompt_cannot_claim_cluster_inference(self):
        with self.assertRaises(ValueError):
            MODULE.paired_cluster([.2])

    def test_second_proposal_sequence_index_is_already_absolute(self):
        # A second proposal at absolute input 5 must pair with T[5], not T[6].
        trows = [{"sequence_index": i, "block_row": 0, "top1_token": 100 + i,
                  "margin": 1.0} for i in range(7)]
        prows = [{"sequence_index": 4, "block_row": 0, "top1_token": 104, "margin": .5},
                 {"sequence_index": 5, "block_row": 1, "top1_token": 105, "margin": .25}]
        reference = {"full_prefix_sha256": "same", "draft_logit_rows": trows,
                     "target_hidden_sha256": "t"}
        production = {"full_prefix_sha256": "same", "draft_logit_rows": prows,
                      "target_logit_rows": prows, "target_hidden_sha256": "p"}
        with patch.object(MODULE, "forced_report", side_effect=[
                ({"target": "gfx1030", "phase86": {"mode": "T"}}, {"prompt": reference}),
                ({"target": "gfx1030", "phase86": {"mode": "P"}}, {"prompt": production})]):
            result = MODULE.target_conditioned_comparison(Path("T"), Path("P"))
        self.assertEqual([s["top1_equal_to_T"] for s in result["prompts"][0]["steps"]], [1, 1])


if __name__ == "__main__":
    unittest.main()
