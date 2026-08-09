"""Adversarial checks for the shared semantic G1 workflow validator."""

from __future__ import annotations

import copy
import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

import validate_json_manifests as manifests  # noqa: E402
import validate_matrix  # noqa: E402


class SemanticG1ManifestValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = ROOT / manifests.SEMANTIC_G1_WORKFLOW_PATH
        documents = dict(manifests.workflow_documents())
        self.document = documents[self.workflow]

    def test_current_reviewed_workflow_matches_hash_and_exact_topology(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.workflow.read_bytes()).hexdigest(),
            manifests.SEMANTIC_G1_WORKFLOW_SHA256,
        )
        manifests.validate_semantic_g1_workflow(self.workflow, copy.deepcopy(self.document))

        steps = self.document["jobs"]["semantic-rmsnorm-g1"]["steps"]
        self.assertNotIn("TREE_OID", steps[2]["run"])
        self.assertNotIn("--tree-oid", steps[3]["run"])
        self.assertEqual(steps[4]["with"]["path"].splitlines(), list(manifests.SEMANTIC_G1_UPLOAD_PATHS))
        self.assertNotIn("**", steps[4]["with"]["path"])

    def test_tree_oid_argv_append_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        steps = mutated["jobs"]["semantic-rmsnorm-g1"]["steps"]
        steps[3]["run"] += '  --tree-oid "$TREE_OID"\n'

        with self.assertRaisesRegex(manifests.ContractError, "explicit safe allowlist|topology/order/actions/env/argv/upload/cleanup"):
            manifests.validate_semantic_g1_workflow(self.workflow, mutated)

    def test_upload_scope_append_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        steps = mutated["jobs"]["semantic-rmsnorm-g1"]["steps"]
        steps[4]["with"]["path"] += "${{ env.RUN_ROOT }}/forged-output/**\n"

        with self.assertRaisesRegex(manifests.ContractError, "explicit safe allowlist|topology/order/actions/env/argv/upload/cleanup"):
            manifests.validate_semantic_g1_workflow(self.workflow, mutated)

    def test_upload_runtime_glob_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        steps = mutated["jobs"]["semantic-rmsnorm-g1"]["steps"]
        steps[4]["with"]["path"] = "\n".join([
            *manifests.SEMANTIC_G1_UPLOAD_PATHS[:-1],
            "${{ env.RUN_ROOT }}/artifacts/rmsnorm-semantic-g1-gfx1201/**",
        ]) + "\n"

        with self.assertRaisesRegex(manifests.ContractError, "explicit safe allowlist|unsafe path"):
            manifests.validate_semantic_g1_workflow(self.workflow, mutated)

    def test_upload_omission_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        steps = mutated["jobs"]["semantic-rmsnorm-g1"]["steps"]
        steps[4]["with"]["path"] = "\n".join(manifests.SEMANTIC_G1_UPLOAD_PATHS[:-1]) + "\n"

        with self.assertRaisesRegex(manifests.ContractError, "explicit safe allowlist"):
            manifests.validate_semantic_g1_workflow(self.workflow, mutated)

    def test_reviewed_abi_and_dispatch_inputs_are_explicitly_owned(self) -> None:
        paths = manifests.read_json(ROOT / "ci/matrix/path-to-suite-v1.json")
        validate_matrix.validate_semantic_g1_path_ownership(paths)
        rules = {rule["pattern"]: set(rule["suite_ids"]) for rule in paths["rules"]}
        self.assertIn("h3-rmsnorm-compile-only", rules["include/sllm/hip.h"])
        self.assertIn("h3-rmsnorm-compile-only", rules["crates/sllm-core/src/op.rs"])

    def test_removing_semantic_abi_or_dispatch_ownership_is_rejected(self) -> None:
        original = manifests.read_json(ROOT / "ci/matrix/path-to-suite-v1.json")
        for target in ("include/sllm/hip.h", "crates/sllm-core/src/op.rs"):
            with self.subTest(target=target):
                mutated = copy.deepcopy(original)
                for rule in mutated["rules"]:
                    if rule["pattern"] == target:
                        rule["suite_ids"].remove("h0-rmsnorm-semantic-g1-contract")
                        break
                with self.assertRaisesRegex(manifests.ContractError, "not explicitly owned"):
                    validate_matrix.validate_semantic_g1_path_ownership(mutated)


if __name__ == "__main__":
    unittest.main()
