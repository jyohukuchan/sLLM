from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker, RefResolver

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from common import ContractError, load_manifests, sha256_file, sha256_json  # noqa: E402
import validate_matrix as registration  # noqa: E402
import validate_rmsnorm_g1_contracts as sealed_contracts  # noqa: E402
import validate_rmsnorm_p0_contracts as contracts  # noqa: E402


def schema_example(node: dict[str, object], root: dict[str, object]) -> object:
    if "$ref" in node:
        reference = str(node["$ref"])
        if not reference.startswith("#/$defs/"):
            raise AssertionError(f"unsupported P0 schema reference: {reference}")
        return schema_example(root["$defs"][reference[8:]], root)  # type: ignore[index]
    if "const" in node:
        return copy.deepcopy(node["const"])
    if "enum" in node:
        return copy.deepcopy(node["enum"][0])  # type: ignore[index]
    schema_type = node.get("type")
    if schema_type == "object":
        properties = node.get("properties", {})
        return {
            name: schema_example(properties[name], root)  # type: ignore[index]
            for name in node.get("required", [])  # type: ignore[union-attr]
        }
    if schema_type == "array":
        prefix = [
            schema_example(child, root)
            for child in node.get("prefixItems", [])  # type: ignore[arg-type]
        ]
        items = node.get("items")
        minimum = int(node.get("minItems", 0))
        if isinstance(items, dict):
            prefix.extend(
                schema_example(items, root)
                for _ in range(max(0, minimum - len(prefix)))
            )
        return prefix
    if schema_type == "string":
        if node.get("format") == "date-time":
            return "2026-08-08T12:34:56Z"
        pattern = str(node.get("pattern", ""))
        if "64" in pattern and "[0-9a-f]" in pattern:
            return "a" * 64
        if "40" in pattern and "[0-9a-f]" in pattern:
            return "a" * 40
        return "x" * max(1, int(node.get("minLength", 1)))
    if schema_type in ("integer", "number"):
        return max(1, int(node.get("minimum", 0)))
    if schema_type == "boolean":
        return True
    if schema_type == "null":
        return None
    raise AssertionError(f"cannot build P0 schema example for {node!r}")


def draft_valid(
    document: object, fragment: dict[str, object], root: dict[str, object]
) -> bool:
    resolver = RefResolver.from_schema(root)
    return not list(
        Draft202012Validator(
            fragment, resolver=resolver, format_checker=FormatChecker()
        ).iter_errors(document)
    )


def stdlib_valid(
    document: object, fragment: dict[str, object], root: dict[str, object]
) -> bool:
    return not sealed_contracts._closed_schema_errors(
        document, fragment, root, "<p0-fragment>"
    )


class P0SchemaTests(unittest.TestCase):
    def test_contracts_matrix_policy_and_all_schemas_validate(self) -> None:
        contracts.validate_contracts(ROOT)
        matrix = contracts.validate_matrix(ROOT)
        policy = contracts.validate_review_policy(ROOT)
        self.assertEqual([target["target"] for target in matrix["targets"]], list(contracts.TARGETS))
        self.assertEqual([case["id"] for case in matrix["cases"]], list(contracts.CASE_IDS))
        self.assertEqual([case["n"] for case in matrix["cases"][-3:]], [255, 256, 257])
        self.assertEqual(matrix["cases"][1]["n"], 2560)
        self.assertEqual(matrix["dtype"], contracts.DTYPE_CONTRACT)
        self.assertEqual(matrix["timing"]["warmup_iterations"], 5)
        self.assertEqual(matrix["timing"]["measurement_iterations"], 21)
        self.assertFalse(policy["threshold"]["approved"])
        self.assertEqual(policy["performance_sanity_disposition"], "review_required")

    def test_dispatch_block_size_is_bound_to_public_runtime_constant(self) -> None:
        header = (ROOT / "include/sllm/hip.h").read_text(encoding="utf-8")
        runtime = (ROOT / "native/hip/src/public_runtime.hip.cpp").read_text(encoding="utf-8")
        self.assertIn(
            "#define SLLM_HIP_RMSNORM_WORKGROUP_SIZE UINT32_C(256)", header
        )
        self.assertIn("SLLM_HIP_RMSNORM_WORKGROUP_SIZE", runtime)
        self.assertEqual(contracts.DISPATCH_BLOCK_SIZE, 256)

    def test_every_object_boundary_is_closed_with_draft_stdlib_parity(self) -> None:
        checked = 0
        for name, relative in contracts.SCHEMAS.items():
            schema = json.loads((ROOT / relative).read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)

            def walk(node: object, schema_path: str) -> None:
                nonlocal checked
                if isinstance(node, dict):
                    if node.get("type") == "object":
                        self.assertIs(
                            node.get("additionalProperties"), False,
                            f"{name}:{schema_path}",
                        )
                        example = schema_example(node, schema)
                        self.assertTrue(draft_valid(example, node, schema))
                        self.assertTrue(stdlib_valid(example, node, schema))
                        forged = copy.deepcopy(example)
                        self.assertIsInstance(forged, dict)
                        forged["__forged_p0_key__"] = True  # type: ignore[index]
                        self.assertFalse(draft_valid(forged, node, schema))
                        self.assertFalse(stdlib_valid(forged, node, schema))
                        checked += 1
                    for key, child in node.items():
                        walk(child, f"{schema_path}/{key}")
                elif isinstance(node, list):
                    for index, child in enumerate(node):
                        walk(child, f"{schema_path}/{index}")

            walk(schema, "")
        self.assertGreater(checked, 35)

    def test_matrix_rejects_missing_duplicate_stale_mixed_and_reordered_values(self) -> None:
        matrix = contracts.expected_matrix()
        mutations = (
            lambda value: value["targets"].pop(),
            lambda value: value["targets"].append(copy.deepcopy(value["targets"][0])),
            lambda value: value["targets"][0].__setitem__("target", "gfx1201"),
            lambda value: value["targets"].reverse(),
            lambda value: value["cases"].reverse(),
            lambda value: value["cases"][2].__setitem__("n", 254),
            lambda value: value["timing"].__setitem__("measurement_iterations", 20),
        )
        for mutation in mutations:
            changed = copy.deepcopy(matrix)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                contracts._schema_validate(ROOT, "matrix", changed)
                if changed != contracts.expected_matrix():
                    raise ContractError("intentional exact P0 matrix drift")

    def test_policy_rejects_threshold_or_claim_promotion(self) -> None:
        policy = contracts.expected_review_policy()
        for mutation in (
            lambda value: value["threshold"].__setitem__("approved", True),
            lambda value: value["threshold"].__setitem__("threshold_id", "invented"),
            lambda value: value["claims"].__setitem__("optimized", True),
            lambda value: value["claims"].__setitem__("faster_than_other_engine", True),
            lambda value: value["claims"].__setitem__("performance_hard_gate_established", True),
        ):
            changed = copy.deepcopy(policy)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                contracts._schema_validate(ROOT, "review_policy", changed)

    def test_public_path_manifest_closes_host_contract_and_existing_runtime_inputs(self) -> None:
        manifest = contracts.read_json(ROOT / contracts.P0_PUBLIC_PATH_INPUTS_PATH)
        paths = contracts.public_path_source_paths(ROOT, manifest)
        required = {
            "crates/sllm-core/src/op.rs",
            "crates/sllm-core/src/backend.rs",
            "crates/sllm-hip/src/bridge.rs",
            "crates/sllm-hip/src/runtime.rs",
            "crates/sllm-hip/src/rmsnorm.rs",
            "crates/sllm-hip-sys/src/bindings.rs",
            "crates/sllm-hip-sys/build.rs",
            "Cargo.lock",
            "Cargo.toml",
            "native/hip/CMakeLists.txt",
            "include/sllm/hip.h",
            "native/hip/src/public_runtime.hip.cpp",
            "native/hip/src/rmsnorm_kernel.hip.cpp",
        }
        self.assertEqual(paths, contracts.EXPECTED_SOURCE_PATHS)
        self.assertEqual(len(paths), 82)
        self.assertTrue(required.issubset(paths))
        self.assertTrue(manifest["dedicated_producer_included"])
        self.assertEqual(
            manifest["a5_enablement_requires"],
            list(contracts.A5_ENABLEMENT_REQUIREMENTS),
        )
        self.assertIn("ci/tools/build_rmsnorm_p0_runtime.py", paths)
        self.assertIn("crates/sllm-hip/src/bin/sllm-rmsnorm-p0-evidence.rs", paths)
        source_set = contracts.source_set(ROOT)
        self.assertEqual(source_set["identity"], contracts.P0_SOURCE_SET_IDENTITY)
        self.assertEqual(
            [item["path"] for item in source_set["files"]], list(paths)
        )
        self.assertEqual(source_set["sha256"], sha256_json(source_set["files"]))
        for source in source_set["files"]:
            self.assertEqual(source["sha256"], sha256_file(ROOT / source["path"]))

    def test_public_path_manifest_rejects_omission_reorder_path_digest_and_a5_drift(self) -> None:
        manifest = contracts.read_json(ROOT / contracts.P0_PUBLIC_PATH_INPUTS_PATH)

        def refresh(value: dict[str, object]) -> None:
            entries = value["source_paths"]
            self.assertIsInstance(entries, list)
            for order, entry in enumerate(entries):
                entry["order"] = order
            value["source_order_sha256"] = sha256_json(
                [entry["path"] for entry in entries]
            )

        mutations: list[dict[str, object]] = []
        omitted = copy.deepcopy(manifest)
        omitted["source_paths"].pop(7)
        refresh(omitted)
        mutations.append(omitted)
        reordered = copy.deepcopy(manifest)
        reordered["source_paths"][7]["path"], reordered["source_paths"][8]["path"] = (
            reordered["source_paths"][8]["path"],
            reordered["source_paths"][7]["path"],
        )
        refresh(reordered)
        mutations.append(reordered)
        path_mutated = copy.deepcopy(manifest)
        path_mutated["source_paths"][14]["path"] = "crates/sllm-core/src/not-op.rs"
        refresh(path_mutated)
        mutations.append(path_mutated)
        digest_mutated = copy.deepcopy(manifest)
        digest_mutated["source_order_sha256"] = "f" * 64
        mutations.append(digest_mutated)
        producer_forged = copy.deepcopy(manifest)
        producer_forged["dedicated_producer_included"] = False
        mutations.append(producer_forged)
        a5_weakened = copy.deepcopy(manifest)
        a5_weakened["a5_enablement_requires"] = ["unreviewed-boundary"]
        mutations.append(a5_weakened)
        for changed in mutations:
            with self.subTest(changed=changed), self.assertRaises(ContractError):
                contracts.public_path_source_paths(ROOT, changed)

    def test_source_set_changes_for_representative_real_file_bytes(self) -> None:
        representatives = (
            "crates/sllm-core/src/op.rs",
            "crates/sllm-core/src/backend.rs",
            "crates/sllm-hip/src/bridge.rs",
            "crates/sllm-hip/src/runtime.rs",
            "crates/sllm-hip/src/rmsnorm.rs",
            "crates/sllm-hip-sys/src/bindings.rs",
            "crates/sllm-hip-sys/build.rs",
            "Cargo.toml",
            "native/hip/CMakeLists.txt",
            "include/sllm/hip.h",
            "native/hip/src/public_runtime.hip.cpp",
            "native/hip/src/rmsnorm_kernel.hip.cpp",
        )
        with tempfile.TemporaryDirectory(prefix="sllm-p0-source-set-") as directory:
            copy_root = Path(directory)
            for relative in contracts.public_path_source_paths(ROOT):
                target = copy_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
            baseline = contracts.source_set(copy_root)["sha256"]
            for relative in representatives:
                path = copy_root / relative
                original = path.read_bytes()
                path.write_bytes(original + b"\nP0 source-set byte mutation\n")
                with self.subTest(relative=relative):
                    self.assertNotEqual(contracts.source_set(copy_root)["sha256"], baseline)
                path.write_bytes(original)
                self.assertEqual(contracts.source_set(copy_root)["sha256"], baseline)

    def test_suite_host_and_path_registration_are_fail_closed(self) -> None:
        suites, host, paths = load_manifests(ROOT)
        registration.validate_p0_suite_registration(suites)
        registration.validate_p0_path_ownership(paths)
        h0 = next(row for row in host["rows"] if row["row_id"] == "h0")
        self.assertIn(registration.P0_SUITE_ID, h0["suite_ids"])

        changed_suites = copy.deepcopy(suites)
        suite = next(
            item for item in changed_suites["suites"]
            if item["suite_id"] == registration.P0_SUITE_ID
        )
        suite["test_ids"].pop()
        with self.assertRaises(ContractError):
            registration.validate_p0_suite_registration(changed_suites)

        changed_paths = copy.deepcopy(paths)
        rule = next(
            item for item in changed_paths["rules"]
            if item["pattern"] == "native/hip/src/public_runtime.hip.cpp"
        )
        rule["suite_ids"].remove(registration.P0_SUITE_ID)
        with self.assertRaises(ContractError):
            registration.validate_p0_path_ownership(changed_paths)

        for relative in (
            contracts.P0_PUBLIC_PATH_INPUTS_PATH,
            "crates/sllm-core/src/op.rs",
            "crates/sllm-hip/src/bridge.rs",
            "crates/sllm-hip-sys/src/bindings.rs",
            "crates/sllm-hip-sys/build.rs",
            "native/hip/CMakeLists.txt",
        ):
            changed_paths = copy.deepcopy(paths)
            rule = next(
                item for item in changed_paths["rules"] if item["pattern"] == relative
            )
            rule["suite_ids"].remove(registration.P0_SUITE_ID)
            with self.subTest(relative=relative), self.assertRaises(ContractError):
                registration.validate_p0_path_ownership(changed_paths)


if __name__ == "__main__":
    unittest.main()
