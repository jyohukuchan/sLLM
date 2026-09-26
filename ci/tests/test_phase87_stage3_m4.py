"""Boundary checks for the Stage 3 M4 expected-throughput calculation."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import phase87_stage3_m4 as m4  # noqa: E402


class Stage3M4Tests(unittest.TestCase):
    def test_second_proposal_requires_first_acceptance(self):
        self.assertEqual(m4.expected_tokens_per_block(0, 1), 1)
        self.assertEqual(m4.expected_tokens_per_block(1, 0), 2)
        self.assertEqual(m4.expected_tokens_per_block(1, 1), 3)
        self.assertAlmostEqual(m4.expected_tokens_per_block(.8, .7), 2.36)
        for values in ((-0.01, .5), (.5, 1.01)):
            with self.assertRaises(ValueError):
                m4.expected_tokens_per_block(*values)

    def test_cluster_interval_uses_all_26_prompts(self):
        with self.assertRaises(ValueError):
            m4.paired_interval([.02] * 25)
        result = m4.paired_interval([.02] * 26, samples=1000)
        self.assertEqual(result["clusters"], 26)
        self.assertEqual(result["ci95"], [.02, .02])


if __name__ == "__main__":
    unittest.main()
