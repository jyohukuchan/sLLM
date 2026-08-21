#!/usr/bin/env python3
"""Pure, focused tests for the B0 normalized Rust dependency policy."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from common import ContractError, read_json  # noqa: E402
from validate_rust_dependencies import (  # noqa: E402
    B0_ABSENT_ENVIRONMENT_VARIABLES,
    B0_DISABLED_HIP_FLAGS,
    B0_SANITIZED_ENVIRONMENT_VARIABLES,
    POLICY_PATH,
    SCHEMA_PATH,
    SECTION_NAMES,
    MSRV_AUTHORITY,
    MSRV_TARGET,
    _cargo_metadata,
    _find_declared_dependency,
    run_cargo_check,
    validate_manifest_against_observed,
)

HOSTILE_CARGO_ENVIRONMENT = {
    name: "hostile" for name in B0_SANITIZED_ENVIRONMENT_VARIABLES
}
HOSTILE_CARGO_ENVIRONMENT.update({
    "CARGO_BUILD_TARGET": "wasm32-unknown-unknown",
    "PATH": "/hostile/path",
    "CARGO_HOME": "/hostile/cargo-home",
    "RUSTUP_HOME": "/hostile/rustup-home",
    "RUSTUP_AUTO_INSTALL": "hostile",
})


class RustDependencyPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = read_json(ROOT / POLICY_PATH)
        cls.schema = read_json(ROOT / SCHEMA_PATH)
        cls.observed = {section: copy.deepcopy(cls.policy[section]) for section in SECTION_NAMES}

    def assert_policy_rejected(self, mutate, *, schema: bool = True) -> None:
        document = copy.deepcopy(self.policy)
        mutate(document)
        with self.assertRaises(ContractError):
            validate_manifest_against_observed(
                document,
                self.observed,
                schema=self.schema if schema else None,
            )

    def test_checked_in_policy_is_happy_path_without_cargo(self) -> None:
        validate_manifest_against_observed(
            copy.deepcopy(self.policy), self.observed, schema=self.schema
        )

    def test_checksum_drift_is_rejected(self) -> None:
        def mutate(document):
            next(item for item in document["packages"] if item["identity"]["name"] == "ahash")["checksum"] = "0" * 64

        self.assert_policy_rejected(mutate)

    def test_license_drift_is_rejected(self) -> None:
        def mutate(document):
            next(item for item in document["packages"] if item["identity"]["name"] == "ahash")["license"] = "MIT"

        self.assert_policy_rejected(mutate)

    def test_unknown_source_is_rejected(self) -> None:
        def mutate(document):
            next(item for item in document["packages"] if item["identity"]["name"] == "ahash")["identity"]["source"] = "git+https://example.invalid/repo"

        self.assert_policy_rejected(mutate)

    def test_package_missing_and_duplicate_identities_are_rejected(self) -> None:
        self.assert_policy_rejected(lambda document: document["packages"].pop(), schema=False)
        self.assert_policy_rejected(
            lambda document: document["packages"].append(copy.deepcopy(document["packages"][0])),
            schema=False,
        )

    def test_edge_kind_and_target_drift_are_rejected(self) -> None:
        self.assert_policy_rejected(lambda document: document["edges"][0].update(kind="build"))
        self.assert_policy_rejected(lambda document: document["edges"][0].update(target="cfg(any())"))

    def test_resolved_feature_drift_is_rejected(self) -> None:
        def mutate(document):
            package = next(item for item in document["packages"] if item["identity"]["name"] == "tokenizers")
            package["features"] = ["onig", "default"]

        self.assert_policy_rejected(mutate)

    def test_tokenizers_forbidden_feature_assertion_is_rejected(self) -> None:
        def mutate(document):
            document["feature_assertions"]["tokenizers"]["forbidden"] = ["default"]

        self.assert_policy_rejected(mutate)

    def test_minijinja_loader_or_default_feature_drift_is_rejected(self) -> None:
        def mutate(document):
            document["feature_assertions"]["minijinja"]["resolved"].append("loader")

        self.assert_policy_rejected(mutate)

    def test_minijinja_requested_feature_drift_is_rejected(self) -> None:
        def mutate(document):
            document["feature_assertions"]["minijinja"]["requested"].remove("fuel")

        self.assert_policy_rejected(mutate)

    def test_server_runtime_requested_feature_drift_is_rejected(self) -> None:
        def mutate(document):
            document["feature_assertions"]["server_runtime"]["dependencies"][0]["requested"] = ["json"]

        self.assert_policy_rejected(mutate)

    def test_server_runtime_default_feature_drift_is_rejected(self) -> None:
        def mutate(document):
            document["feature_assertions"]["server_runtime"]["dependencies"][0]["uses_default_features"] = True

        self.assert_policy_rejected(mutate)

    def test_esaxx_presence_and_feature_closure_is_explicit(self) -> None:
        def mutate(document):
            package = next(item for item in document["packages"] if item["identity"]["name"] == "esaxx-rs")
            package["features"] = ["cpp"]

        self.assert_policy_rejected(mutate)

    def test_wasip2_missing_rust_version_is_rejected(self) -> None:
        def mutate(document):
            package = next(item for item in document["packages"] if item["identity"]["name"] == "wasip2")
            package["rust_version"] = None

        self.assert_policy_rejected(mutate)

    def test_wasip2_different_rust_version_is_rejected(self) -> None:
        def mutate(document):
            package = next(item for item in document["packages"] if item["identity"]["name"] == "wasip2")
            package["rust_version"] = "1.88.0"

        self.assert_policy_rejected(mutate)

    def test_absolute_path_is_rejected(self) -> None:
        self.assert_policy_rejected(
            lambda document: document["workspace_members"][0].update(manifest="/tmp/Cargo.toml")
        )

    def test_schema_required_field_mutation_is_rejected(self) -> None:
        self.assert_policy_rejected(lambda document: document.pop("counts"))

    def test_renamed_active_dependency_matches_alias_and_preserves_manifest_name(self) -> None:
        package_dependencies = [
            {
                "name": "serde",
                "rename": "serde_lib",
                "kind": None,
                "target": None,
            }
        ]
        resolve_dependency = {"name": "serde_lib", "pkg": "registry+serde@1.0.0"}

        declared = _find_declared_dependency(
            package_dependencies,
            resolve_dependency["name"],
            kind="normal",
            target=None,
        )

        self.assertIsNotNone(declared)
        self.assertEqual(declared["name"], "serde")
        self.assertEqual(declared["rename"], "serde_lib")

    def test_unrenamed_dependency_does_not_match_a_resolve_alias(self) -> None:
        declared = _find_declared_dependency(
            [{"name": "serde", "rename": None, "kind": None, "target": None}],
            "serde_lib",
            kind="normal",
            target=None,
        )

        self.assertIsNone(declared)

    def test_cargo_metadata_command_is_offline_and_target_independent(self) -> None:
        observed = {}

        def successful_runner(command, **kwargs):
            observed["command"] = command
            observed["kwargs"] = kwargs
            return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

        with patch.dict(
            os.environ,
            HOSTILE_CARGO_ENVIRONMENT,
            clear=False,
        ):
            self.assertEqual(_cargo_metadata(ROOT, runner=successful_runner), {})

        self.assertEqual(
            observed["command"],
            ["cargo", f"+{MSRV_AUTHORITY}", "metadata", "--locked", "--offline", "--format-version", "1"],
        )
        self.assertNotIn("--jobs", observed["command"])
        self.assertEqual(observed["kwargs"]["env"]["CARGO_NET_OFFLINE"], "true")
        self.assertEqual(observed["kwargs"]["env"]["RUSTUP_AUTO_INSTALL"], "0")
        for name in ("PATH", "CARGO_HOME", "RUSTUP_HOME"):
            self.assertEqual(observed["kwargs"]["env"][name], HOSTILE_CARGO_ENVIRONMENT[name])
        for name in B0_DISABLED_HIP_FLAGS:
            self.assertEqual(observed["kwargs"]["env"][name], "0")
        for name in B0_ABSENT_ENVIRONMENT_VARIABLES:
            self.assertNotIn(name, observed["kwargs"]["env"])
        self.assertNotIn("CARGO_BUILD_TARGET", observed["kwargs"]["env"])

    def test_cargo_check_command_pins_recorded_target_and_sanitizes_environment(self) -> None:
        observed = {}

        def successful_runner(command, **kwargs):
            observed["command"] = command
            observed["kwargs"] = kwargs
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with patch.dict(
            os.environ,
            HOSTILE_CARGO_ENVIRONMENT,
            clear=False,
        ):
            run_cargo_check(ROOT, runner=successful_runner)

        self.assertEqual(
            observed["command"],
            [
                "cargo", f"+{MSRV_AUTHORITY}", "check", "--jobs", "1", "--workspace", "--all-targets", "--locked", "--offline",
                "--target", MSRV_TARGET,
            ],
        )
        self.assertEqual(
            list(zip(observed["command"], observed["command"][1:])).count(("--jobs", "1")),
            1,
        )
        self.assertEqual(observed["kwargs"]["env"]["CARGO_NET_OFFLINE"], "true")
        self.assertEqual(observed["kwargs"]["env"]["RUSTUP_AUTO_INSTALL"], "0")
        for name in ("PATH", "CARGO_HOME", "RUSTUP_HOME"):
            self.assertEqual(observed["kwargs"]["env"][name], HOSTILE_CARGO_ENVIRONMENT[name])
        for name in B0_DISABLED_HIP_FLAGS:
            self.assertEqual(observed["kwargs"]["env"][name], "0")
        for name in B0_ABSENT_ENVIRONMENT_VARIABLES:
            self.assertNotIn(name, observed["kwargs"]["env"])
        self.assertNotIn("CARGO_BUILD_TARGET", observed["kwargs"]["env"])

    def test_cargo_check_command_failure_is_not_a_pass(self) -> None:
        def failed_runner(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], 17, stdout="", stderr="synthetic cargo failure")

        with self.assertRaises(ContractError):
            run_cargo_check(ROOT, runner=failed_runner)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if os.environ.get("SLLM_EMIT_TEST_COUNTS") == "1":
        selected = result.testsRun
        failed = len(result.failures) + len(result.errors)
        skipped = len(result.skipped)
        print(
            "SLLM_UNITTEST_COUNTS="
            + json.dumps(
                {
                    "collected": selected,
                    "selected": selected,
                    "passed": selected - failed - skipped,
                    "failed": failed,
                    "skipped": skipped,
                    "deselected": 0,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
