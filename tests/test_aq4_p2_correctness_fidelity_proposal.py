from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "benchmarks/workloads/aq4-production-opt-p2-correctness-fidelity-proposal-v0.1.json"
DOCUMENT = ROOT / "docs/proposals/aq4-p2-correctness-fidelity-amendment-v0.1.md"
THRESHOLD_TEMPLATE = ROOT / "benchmarks/workloads/aq4-production-opt-p2-threshold-policy-template-v0.1.json"
THRESHOLD_AUDIT = ROOT / "benchmarks/aq4-production-prefill-decode/p2/correctness-threshold-audit.json"


class Aq4P2CorrectnessFidelityProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))

    def test_proposal_is_explicitly_non_binding_and_threshold_free(self) -> None:
        self.assertEqual(self.proposal["schema_version"], "ullm.aq4_p2_correctness_fidelity_proposal.v1")
        self.assertEqual(self.proposal["status"], "proposed")
        self.assertEqual(self.proposal["normative_status"], "non_binding_review")
        self.assertFalse(self.proposal["promotion_eligible"])
        thresholds = self.proposal["threshold_derivation"]
        self.assertTrue(thresholds["frozen_before_candidate_execution"])
        self.assertTrue(thresholds["observed_attempt2_must_not_change_policy"])
        self.assertTrue(thresholds["no_threshold_values_in_proposal"])
        self.assertIn("attempt2 observed rows or VRAM/power samples", thresholds["forbidden_bases"])

    def test_source_contract_records_current_exact_gate_and_proposed_holdout_split(self) -> None:
        source = self.proposal["contract_split"]["bf16_source_vs_lossy_aq4"]
        current = source["current_contract"]
        self.assertTrue(current["exact_greedy_token_sequence_required"])
        self.assertTrue(current["exact_top_k_ranking_or_policy_overlap_required"])
        holdout = source["proposed_contract"]["independent_holdout"]
        self.assertFalse(holdout["exact_row_greedy_required"])
        self.assertFalse(holdout["exact_row_top_k_required"])
        self.assertTrue(holdout["thresholds_must_be_frozen_before_candidate_run"])
        self.assertTrue(holdout["holdout_must_be_disjoint_from_tuning_cases"])
        self.assertGreaterEqual(
            set(holdout["fidelity_metrics_required"]),
            {
                "token_agreement_rate",
                "top_k_overlap",
                "logits_cosine_similarity",
                "logits_relative_l2",
                "hidden_relative_l2",
                "quality_task_score",
            },
        )

    def test_active_path_requires_exact_behavior_and_state(self) -> None:
        path = self.proposal["contract_split"]["candidate_vs_active_path"]
        self.assertEqual(path["oracle_kind"], "same_artifact_all_m1")
        self.assertTrue(path["exact_behavior_required"])
        self.assertGreaterEqual(
            set(path["exact_fields"]),
            {
                "greedy_token_id",
                "top_k_token_ids_and_order",
                "context_token_ids_sha256",
                "kv_cache_lengths",
                "absolute_positions",
                "scheduler_request_ownership",
                "terminal_outcome_and_reset",
            },
        )

    def test_minimum_amendment_contains_evidence_validator_and_rejection_contracts(self) -> None:
        amendment = self.proposal["minimum_p2_spec_amendment"]
        required = set(amendment["required_evidence"])
        self.assertTrue(any("disjoint" in value for value in required))
        self.assertTrue(any("independent BF16" in value for value in required))
        self.assertTrue(any("frozen policy" in value for value in required))
        checks = set(amendment["validator_tests"])
        self.assertTrue(any("attempt2" in value for value in checks))
        self.assertTrue(any("holdout overlap" in value for value in checks))
        rejection = set(amendment["rejection_criteria"])
        self.assertTrue(any("behavioral mismatch" in value for value in rejection))
        self.assertTrue(any("post-hoc" in value for value in rejection))

    def test_existing_unbound_policy_and_blocked_audit_remain_unchanged_in_scope(self) -> None:
        template = json.loads(THRESHOLD_TEMPLATE.read_text(encoding="utf-8"))
        audit = json.loads(THRESHOLD_AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(template["status"], "unbound_template")
        self.assertTrue(template["binding_contract"]["unbound_template_is_planning_only"])
        self.assertEqual(audit["status"], "blocked")
        self.assertIsNone(audit["values"])
        self.assertFalse(audit["promotion_eligible"])

    def test_document_uses_required_proposal_sections_and_mentions_attempt2_boundary(self) -> None:
        document = DOCUMENT.read_text(encoding="utf-8")
        for heading in ("## 前回の要点", "## 今回の変更点", "## 次の行動"):
            self.assertIn(heading, document)
        for phrase in ("attempt2", "active AQ4", "independent holdout", "exact behavioral"):
            self.assertIn(phrase, document)


if __name__ == "__main__":
    unittest.main()
