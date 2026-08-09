from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from common import ContractError  # noqa: E402
import validate_rmsnorm_g2_contracts as contracts  # noqa: E402
import validate_rmsnorm_g1_contracts as sealed_contracts  # noqa: E402


class G2SchemaTests(unittest.TestCase):
    def test_matrix_tolerance_and_all_schemas_are_closed(self) -> None:
        contracts.validate_contracts(ROOT)
        self.assertEqual(contracts.validate_matrix(ROOT)["targets"][0]["target"], "gfx1030")
        self.assertEqual(contracts.validate_matrix(ROOT)["targets"][1]["target"], "gfx1201")
        for name in contracts.SCHEMAS:
            schema = json.loads((ROOT / contracts.SCHEMAS[name]).read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            self.assertTrue(list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors({})), name)

    def test_unknown_keys_and_case_order_are_rejected(self) -> None:
        matrix = contracts.expected_matrix()
        matrix["unexpected"] = True
        with self.assertRaises(ContractError):
            contracts._schema_validate(ROOT, "matrix", matrix)

    def test_matrix_rejects_missing_duplicate_stale_and_mixed_rows(self) -> None:
        matrix = contracts.expected_matrix()
        for mutation in (
            lambda value: value["targets"].pop(),
            lambda value: value["targets"].append(copy.deepcopy(value["targets"][0])),
            lambda value: value["targets"][0].__setitem__("target", "gfx1201"),
            lambda value: value["cases"].reverse(),
        ):
            changed = copy.deepcopy(matrix)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                # Exercise the same closed validator used by the CLI without
                # altering a checked-in manifest.
                contracts._schema_validate(ROOT, "matrix", changed)
                if changed != contracts.expected_matrix():
                    raise ContractError("intentional matrix drift")

    def test_canonical_build_source_set_covers_cargo_build_and_cmake_inputs(self) -> None:
        manifest = contracts.read_json(ROOT / contracts.G2_BUILD_INPUTS_PATH)
        paths = contracts._build_inputs_manifest(ROOT, manifest)
        required_omissions = {
            "crates/sllm-core/src/backend.rs", "crates/sllm-core/src/execution.rs",
            "crates/sllm-core/src/fake.rs", "crates/sllm-core/src/handles.rs",
            "crates/sllm-core/src/model.rs", "crates/sllm-core/src/registry.rs",
            "crates/sllm-hip-sys/Cargo.toml", "crates/sllm-hip-sys/src/evidence_bindings.rs",
            "native/hip/src/abi_layout_probe.cpp", "native/hip/src/header_c_compile.c",
            "native/hip/src/header_cpp_compile.cpp",
        }
        self.assertTrue(required_omissions.issubset(paths))
        source_set = contracts._source_set(ROOT)
        self.assertEqual(tuple(source_set["canonical_order"]), paths)
        self.assertEqual(source_set["source_set_sha256"], contracts.sha256_json(source_set["files"]))

    def test_build_source_manifest_rejects_omission_reorder_mutation_and_digest_mismatch(self) -> None:
        manifest = contracts.read_json(ROOT / contracts.G2_BUILD_INPUTS_PATH)
        mutations = []
        omitted = copy.deepcopy(manifest)
        omitted["source_paths"].pop(5)
        mutations.append(omitted)
        reordered = copy.deepcopy(manifest)
        reordered["source_paths"][1], reordered["source_paths"][2] = reordered["source_paths"][2], reordered["source_paths"][1]
        mutations.append(reordered)
        mutated = copy.deepcopy(manifest)
        mutated["source_paths"][5]["path"] = "crates/sllm-core/src/lib.rs"
        mutations.append(mutated)
        digest = copy.deepcopy(manifest)
        digest["source_order_sha256"] = "f" * 64
        mutations.append(digest)
        for changed in mutations:
            with self.subTest(changed=changed), self.assertRaises(ContractError):
                contracts._build_inputs_manifest(ROOT, changed)
        identity = contracts.expected_build_identity(ROOT)["identity"]
        changed_identity = {**identity, "source_set_sha256": "f" * 64}
        with self.assertRaises(ContractError):
            contracts._validate_embedded_build_identity(
                contracts.G2_IDENTITY_MARKER + contracts.canonical_bytes(changed_identity),
                ROOT,
            )

    def test_tolerance_nullable_sha_has_draft_and_stdlib_parity(self) -> None:
        schema = contracts._schema(ROOT, "tolerance")
        tolerance = contracts.read_json(ROOT / contracts.TOLERANCE_PATH)
        probes = (
            (None, True),
            ("a" * 64, True),
            (7, False),
            ([], False),
            (True, False),
            ("not-a-sha", False),
        )
        for value, expected in probes:
            document = copy.deepcopy(tolerance)
            document["approval"]["calibration_candidate_sha256"] = value
            draft_valid = not list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document))
            try:
                sealed_contracts.validate_schema(document, schema, "G2 tolerance differential probe")
            except sealed_contracts.EvidenceError:
                stdlib_valid = False
            else:
                stdlib_valid = True
            with self.subTest(value=value):
                self.assertEqual(draft_valid, expected)
                self.assertEqual(stdlib_valid, expected)


if __name__ == "__main__":
    unittest.main()
