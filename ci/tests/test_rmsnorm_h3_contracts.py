from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

import common
import run_rmsnorm_h3_compile as runner
import validate_json_manifests as manifests
import validate_matrix as matrix_registry
import validate_rmsnorm_h3_contracts as validator


class RmsNormH3ContractTests(unittest.TestCase):
    def _workflow(self) -> tuple[Path, dict[str, object]]:
        path = ROOT / ".github/workflows/rmsnorm-h3-compile.yml"
        return path, yaml.safe_load(path.read_text(encoding="utf-8"))

    def _assert_workflow_rejected(self, mutation, label: str) -> None:
        path, workflow = self._workflow()
        mutation(copy.deepcopy(workflow))
        with self.subTest(label=label):
            with self.assertRaises(common.ContractError):
                manifests.validate_rmsnorm_h3_workflow(path, workflow)

    def _replace_run(self, workflow: dict[str, object], step_index: int, old: str, new: str) -> None:
        steps = workflow["jobs"]["h3-rmsnorm"]["steps"]
        assert isinstance(steps, list)
        step = steps[step_index]
        assert isinstance(step, dict)
        step["run"] = step["run"].replace(old, new)

    def _manifest_copy(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="sllm-rmsnorm-h3-registry-"))
        for relative in ("ci/matrix/suites-v1.json", "ci/matrix/host-v1.json", "ci/matrix/path-to-suite-v1.json"):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        return root

    def test_static_contract_and_source_symbol_map_pass(self) -> None:
        toolchain, matrix, rows = validator.validate_static(ROOT)
        self.assertEqual(toolchain["toolchain_id"], "rocm-7.14.0")
        self.assertEqual(set(rows), {"h3-rmsnorm-gfx1030", "h3-rmsnorm-gfx1201"})
        self.assertEqual(matrix["device_symbol"], runner.DEVICE_SYMBOL)
        self.assertEqual(matrix["source_symbol_map"], runner.SOURCE_SYMBOL_MAP)

    def test_matrix_rejects_wrong_target_and_unknown_row(self) -> None:
        original = runner.read_json(ROOT / "ci/matrix/rmsnorm-h3-compile-v1.json")
        changed = copy.deepcopy(original)
        changed["rows"][0]["codegen"]["target"] = "gfx1201"

        def read(path: Path):
            if path == ROOT / "ci/matrix/rmsnorm-h3-compile-v1.json":
                return changed
            return runner.read_json(path)

        with patch.object(runner, "read_json", side_effect=read):
            with self.assertRaises(runner.ContractError):
                runner.validate_matrix(ROOT)

    def test_schema_rejects_unknown_top_level_field(self) -> None:
        schema = json.loads((ROOT / validator.SCHEMAS["compile"]).read_text(encoding="utf-8"))
        document = json.loads((ROOT / "ci/matrix/rmsnorm-h3-compile-v1.json").read_text(encoding="utf-8"))
        document["unexpected"] = True
        errors = list(validator.Draft202012Validator(schema).iter_errors(document))
        self.assertTrue(errors)

    def test_source_inventory_excludes_probe_and_fake_runtime(self) -> None:
        matrix = runner.read_json(ROOT / "ci/matrix/rmsnorm-h3-compile-v1.json")
        all_paths = [item["path"] for inventory in matrix["source_sets"].values() for item in inventory["files"]]
        self.assertNotIn("native/hip/src/hip_compile_probe.hip.cpp", all_paths)
        self.assertNotIn("native/hip/src/public_runtime_stub.cpp", all_paths)
        self.assertNotIn("native/hip/src/hip_stub.cpp", all_paths)

    def test_global_dispatch_accepts_dedicated_workflow_and_host_fallback_rejects_it(self) -> None:
        path, workflow = self._workflow()
        self.assertEqual(manifests.validate_workflow(path, workflow), [])
        with self.assertRaises(common.ContractError):
            manifests.validate_host_workflow(path, workflow)

    def test_global_workflow_security_and_closed_rows_fail_closed(self) -> None:
        mutations = (
            (lambda w: w["jobs"]["h3-rmsnorm"]["steps"][0].__setitem__("uses", "actions/checkout@" + "f" * 40), "wrong action SHA"),
            (lambda w: w["env"].__setitem__("RMSNORM_H3_IMAGE_REFERENCE", "docker.io/rocm/dev-ubuntu-24.04:latest"), "mutable image tag"),
            (lambda w: w["env"].pop("SLLM_H3_NETWORK_DISABLED"), "missing network-disabled environment"),
            (lambda w: w["env"].__setitem__("SLLM_H3_NETWORK_DISABLED", "0"), "wrong network-disabled environment"),
            (lambda w: self._replace_run(w, 3, "src=$GITHUB_WORKSPACE,dst=/workspace,readonly", "src=$GITHUB_WORKSPACE,dst=/workspace,rw"), "writable source checkout"),
            (lambda w: self._replace_run(w, 3, "src=$RUN_ROOT,dst=/tmp", "src=/tmp,dst=/tmp"), "broad host /tmp mount"),
            (lambda w: self._replace_run(w, 3, "--network none", "--network bridge"), "enabled network"),
            (lambda w: self._replace_run(w, 3, '--mount "type=bind,src=$RUN_ROOT,dst=/tmp"', '--device /dev/kfd\n            --mount "type=bind,src=$RUN_ROOT,dst=/tmp"'), "GPU/device access"),
            (lambda w: self._replace_run(w, 2, 'test "$(git rev-parse HEAD^{tree})" = "$TREE_OID"\n', ""), "missing tree identity check"),
            (lambda w: self._replace_run(w, 5, 'mkdir -p -m 700 "$GITHUB_WORKSPACE/.local-artifacts/rmsnorm-h3-aggregate"', 'mkdir -m 700 "$GITHUB_WORKSPACE/.local-artifacts/rmsnorm-h3-aggregate"'), "missing aggregate parent creation"),
            (lambda w: self._replace_run(w, 5, "$RUN_ROOT/sllm-rmsnorm-h3-aggregate-", "/tmp/sllm-rmsnorm-h3-aggregate-"), "aggregate host/container namespace drift"),
            (lambda w: w["jobs"]["h3-rmsnorm"]["steps"][4].__setitem__("run", "set -eu\n"), "missing gfx1201 row"),
            (lambda w: w["jobs"]["h3-rmsnorm"]["steps"][6]["with"].__setitem__("path", ".local-artifacts/rmsnorm-h3-aggregate/host-bundle-gfx1030.elf\n"), "binary upload"),
        )
        for mutation, label in mutations:
            path, workflow = self._workflow()
            changed = copy.deepcopy(workflow)
            mutation(changed)
            with self.subTest(label=label):
                with self.assertRaises(common.ContractError):
                    manifests.validate_rmsnorm_h3_workflow(path, changed)

    def test_registry_tier_marker_and_every_dedicated_path_are_closed(self) -> None:
        suites, _, paths = common.load_manifests(ROOT)
        dedicated = next(item for item in suites["suites"] if item["suite_id"] == matrix_registry.H3_RMSNORM_SUITE_ID)
        self.assertEqual(dedicated["tier"], "tier_h3_rmsnorm")
        self.assertEqual(dedicated["marker"], "tier_h3_rmsnorm")
        self.assertEqual(dedicated["attributes"], {key: False for key in common.ALLOWED_ATTRIBUTES})
        self.assertEqual(dedicated["test_ids"], matrix_registry.EXPECTED_H3_RMSNORM_TEST_IDS)
        self.assertEqual(dedicated["commands"], [{"command_id": "h3-rmsnorm-contracts", "argv": ["{python}", "-m", "unittest", "ci.tests.test_rmsnorm_h3_contracts", "ci.tests.test_rmsnorm_h3_runner", "ci.tests.test_rmsnorm_h3_aggregate"]}])
        mutated_suites = copy.deepcopy(suites)
        mutated_dedicated = next(item for item in mutated_suites["suites"] if item["suite_id"] == matrix_registry.H3_RMSNORM_SUITE_ID)
        mutated_dedicated["test_ids"].reverse()
        with self.assertRaises(common.ContractError):
            matrix_registry.validate_rmsnorm_suite_registration(mutated_suites)
        matrix_registry.validate_rmsnorm_path_ownership(paths)
        for relative in ("ci/tools/common.py", "ci/matrix/suites-v1.json", "ci/matrix/path-to-suite-v1.json", "ci/toolchains/rocm-7.14.0.json"):
            mutated = copy.deepcopy(paths)
            rule = next(item for item in mutated["rules"] if item["pattern"] == relative)
            rule["suite_ids"].remove(matrix_registry.H3_RMSNORM_SUITE_ID)
            with self.subTest(relative=relative):
                with self.assertRaises(common.ContractError):
                    matrix_registry.validate_rmsnorm_path_ownership(mutated)

        for mutation, label in (
            (lambda item: item.__setitem__("tier", "tier_h3"), "wrong dedicated tier"),
            (lambda item: item.__setitem__("marker", "tier_h3"), "wrong dedicated marker"),
        ):
            root = self._manifest_copy()
            try:
                suites_document = common.read_json(root / "ci/matrix/suites-v1.json")
                item = next(entry for entry in suites_document["suites"] if entry["suite_id"] == matrix_registry.H3_RMSNORM_SUITE_ID)
                mutation(item)
                (root / "ci/matrix/suites-v1.json").write_text(json.dumps(suites_document) + "\n", encoding="utf-8")
                with self.subTest(label=label):
                    with self.assertRaises(common.ContractError):
                        common.load_manifests(root)
            finally:
                shutil.rmtree(root)

    def test_generic_h3_and_public_runtime_profiles_remain_unchanged(self) -> None:
        suites, _, paths = common.load_manifests(ROOT)
        by_id = {item["suite_id"]: item for item in suites["suites"]}
        expected_false = {key: False for key in common.ALLOWED_ATTRIBUTES}
        self.assertEqual(by_id["h3-compile-only-contract"]["tier"], "tier_h3")
        self.assertEqual(by_id["h3-compile-only-contract"]["marker"], "tier_h3")
        self.assertEqual(by_id["h3-compile-only-contract"]["attributes"], expected_false)
        self.assertEqual(by_id["h3-public-runtime-compile-only"]["tier"], "tier_h3")
        self.assertEqual(by_id["h3-public-runtime-compile-only"]["marker"], "tier_h3")
        self.assertEqual(by_id["h3-public-runtime-compile-only"]["attributes"], expected_false)
        self.assertIn("h3-compile-only-contract", next(item for item in paths["rules"] if item["pattern"] == ".github/workflows/h3-compile.yml")["suite_ids"])
        self.assertIn("h3-public-runtime-compile-only", next(item for item in paths["rules"] if item["pattern"] == ".github/workflows/h3-public-runtime-compile.yml")["suite_ids"])


if __name__ == "__main__":
    unittest.main()
