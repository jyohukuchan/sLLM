#!/usr/bin/env python3
"""Small host-contract checks for the Phase 1 CI control plane."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from common import (  # noqa: E402
    ALLOWED_ATTRIBUTES,
    ALLOWED_TIERS,
    DEV_RUST_VERSION,
    MSRV_RUST_VERSION,
    load_manifests,
    sha256_json,
)
from validate_matrix import main as validate_matrix_main  # noqa: E402
from validate_rust import RUSTUP_AUTO_INSTALL, command_for_mode  # noqa: E402


class HostContractTests(unittest.TestCase):
    def test_host_matrix_is_exactly_three_rows(self) -> None:
        _, host, _ = load_manifests(ROOT)
        self.assertEqual([row["row_id"] for row in host["rows"]], ["h0", "h1", "h2"])
        self.assertTrue(all(row["required"] for row in host["rows"]))

    def test_registry_markers_and_attributes_are_closed(self) -> None:
        suites, _, _ = load_manifests(ROOT)
        self.assertEqual(suites["allowed_tiers"], list(ALLOWED_TIERS))
        self.assertEqual(suites["allowed_attributes"], list(ALLOWED_ATTRIBUTES))
        for suite in suites["suites"]:
            self.assertEqual(set(suite["attributes"]), set(ALLOWED_ATTRIBUTES))
            self.assertEqual(suite["marker"], suite["tier"])
            self.assertEqual(suite["attributes"]["requires_gpu"], suite["tier"].startswith(("tier_g", "tier_p")))

    def test_seed_and_manifest_digest_are_deterministic(self) -> None:
        first = json.loads((ROOT / "ci/matrix/host-v1.json").read_text(encoding="utf-8"))
        second = json.loads((ROOT / "ci/matrix/host-v1.json").read_text(encoding="utf-8"))
        self.assertEqual(sha256_json(first), sha256_json(second))
        self.assertEqual([row["seed"] for row in first["rows"]], [1729, 2718, 314159])

    def test_commands_are_local_and_network_free(self) -> None:
        suites, _, _ = load_manifests(ROOT)
        for suite in suites["suites"]:
            for command in suite["commands"]:
                self.assertTrue("{python}" in command["argv"] or command["argv"][0] == "cargo")
                self.assertFalse(any("://" in arg for arg in command["argv"]))

    def test_rust_toolchain_registration_is_exact(self) -> None:
        suites, _, _ = load_manifests(ROOT)
        cargo_commands = [
            (suite["suite_id"], command["command_id"], command["argv"])
            for suite in suites["suites"]
            for command in suite["commands"]
            if command["argv"][0] == "cargo"
        ]
        self.assertEqual(
            cargo_commands,
            [("h1-host-contract", "cargo-test-workspace", [
                "cargo", f"+{DEV_RUST_VERSION}", "test", "--workspace", "--locked", "--offline",
            ])],
        )
        self.assertNotIn(f"+{MSRV_RUST_VERSION}", json.dumps(suites, sort_keys=True))
        self.assertEqual(
            command_for_mode("format"),
            ["cargo", f"+{DEV_RUST_VERSION}", "fmt", "--all", "--", "--check"],
        )
        clippy_command = command_for_mode("clippy")
        self.assertEqual(
            clippy_command,
            [
                "cargo", f"+{DEV_RUST_VERSION}", "clippy", "--jobs", "1",
                "--workspace", "--all-targets", "--all-features", "--locked",
                "--offline", "--", "-D", "warnings",
            ],
        )
        self.assertEqual(list(zip(clippy_command, clippy_command[1:])).count(("--jobs", "1")), 1)
        msrv_command = command_for_mode("msrv")
        self.assertEqual(
            msrv_command,
            [
                "cargo", f"+{MSRV_RUST_VERSION}", "check", "--jobs", "1",
                "--workspace", "--locked", "--offline",
            ],
        )
        self.assertEqual(list(zip(msrv_command, msrv_command[1:])).count(("--jobs", "1")), 1)
        self.assertEqual(RUSTUP_AUTO_INSTALL, "0")

    def test_fixture_paths_are_explicitly_owned_by_h0_and_their_consumer_tier(self) -> None:
        _, _, paths = load_manifests(ROOT)
        rules = {rule["pattern"]: set(rule["suite_ids"]) for rule in paths["rules"]}
        self.assertEqual(rules["tests/fixtures/api_cases.json"], {"h0-python", "h1-host-contract"})
        for name in ("boundary_cases.json", "kv_layout.json", "sampling_cases.json"):
            self.assertEqual(rules[f"tests/fixtures/{name}"], {"h0-python", "h2-tiny-oracle"})

    def test_matrix_validator_proves_markers_and_fixture_consumers(self) -> None:
        self.assertEqual(validate_matrix_main(), 0)


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
