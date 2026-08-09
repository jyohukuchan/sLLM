"""Adversarial, host-only checks for parent-issued exact actions."""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

import exact_actions  # noqa: E402


class ExactActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="sllm-g1-exact-action-")
        self.root = Path(self._temporary.name)
        self.cwd = self.root / "cwd"
        self.cwd.mkdir()
        self.output_parent = self.root / "outputs"
        self.output_parent.mkdir()
        self.tool = self.root / "sealed-tool"
        self.tool.write_bytes(b"sealed executable bytes\n")
        self.input = self.root / "input.hpp"
        self.input.write_bytes(b"reviewed input bytes\n")

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _manifest(self, *, marker: str = "retained") -> dict[str, object]:
        executable = exact_actions.file_record(self.tool, role="executable", label="test executable")
        executable.pop("role")
        executable["seals"] = 15
        return exact_actions.make_manifest(
            executable=executable,
            argv0="/proc/self/fd/198",
            argv=["--compile", str(self.input), "-o", str(self.output_parent / "result.o")],
            cwd=self.cwd,
            environment={"PATH": "/usr/bin:/bin", "LANG": "C", "UNRELATED": marker},
            inputs=[
                exact_actions.file_record(self.input, role="source", label="test input"),
                exact_actions.file_record(self.tool, role="launcher", label="test launcher"),
                exact_actions.file_record(self.tool, role="linker", label="test linker"),
            ],
            implicit=[exact_actions.implicit_record(role="configuration", value=b"target=gfx1030\n")],
            response_files=[exact_actions.implicit_record(role="response-file", value=b"-O3\n")],
            outputs=[exact_actions.output_record(self.output_parent / "result.o", label="test output")],
            target="gfx1030",
        )

    @staticmethod
    def _resign(manifest: dict[str, object]) -> dict[str, object]:
        unsigned = dict(manifest)
        unsigned.pop("manifest_digest")
        manifest["manifest_digest"] = exact_actions.sha256(unsigned)
        return manifest

    def test_digest_rejects_mutation_of_every_bound_field(self) -> None:
        manifest = self._manifest()
        mutations = {
            "action_id": lambda value: "0" * 64,
            "executable": lambda value: {**value, "sha256": "1" * 64},
            "argv0": lambda value: "/tmp/attacker-tool",
            "argv": lambda value: [*value, "-DATTACK=1"],
            "cwd": lambda value: {**value, "inode": int(value["inode"]) + 1},
            "environment": lambda value: [["ATTACK", "1"], *value],
            "inputs": lambda value: [{**value[0], "sha256": "2" * 64}],
            "implicit": lambda value: [{**value[0], "bytes_hex": "00"}],
            "response_files": lambda value: [{**value[0], "bytes_hex": "00"}],
            "outputs": lambda value: [{**value[0], "path": str(self.output_parent / "attacker.o")}],
            "target": lambda value: "gfx9999",
            "occurrence_index": lambda value: 1,
            "occurrence_limit": lambda value: 2,
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field):
                forged = copy.deepcopy(manifest)
                forged[field] = mutate(forged[field])
                with self.assertRaises(exact_actions.ExactActionError):
                    exact_actions.validate_manifest(forged)
        nested_mutations = {
            ("schema_version",): "forged-version",
            ("executable", "path"): "/tmp/attacker-tool",
            ("executable", "resolved_path"): "/tmp/attacker-tool",
            ("executable", "size_bytes"): 0,
            ("executable", "device"): 0,
            ("executable", "inode"): 0,
            ("executable", "seals"): 0,
            ("argv", 0): "--attacker-compile",
            ("argv", 1): "attacker.hpp",
            ("argv", 2): "--not-output",
            ("argv", 3): "/tmp/attacker.o",
            ("cwd", "path"): "/tmp",
            ("cwd", "resolved_path"): "/tmp",
            ("cwd", "device"): 0,
            ("cwd", "inode"): 0,
            ("environment", 0, 0): "ATTACK_ENV",
            ("environment", 0, 1): "attacker-value",
            ("inputs", 0, "role"): "attacker-input",
            ("inputs", 0, "path"): "/tmp/attacker-input",
            ("inputs", 0, "resolved_path"): "/tmp/attacker-input",
            ("inputs", 0, "size_bytes"): 0,
            ("inputs", 0, "device"): 0,
            ("inputs", 0, "inode"): 0,
            ("inputs", 1, "role"): "attacker-launcher",
            ("inputs", 1, "sha256"): "6" * 64,
            ("inputs", 2, "role"): "attacker-linker",
            ("inputs", 2, "sha256"): "7" * 64,
            ("implicit", 0, "role"): "attacker-config",
            ("implicit", 0, "size_bytes"): 0,
            ("implicit", 0, "sha256"): "4" * 64,
            ("response_files", 0, "role"): "attacker-response",
            ("response_files", 0, "size_bytes"): 0,
            ("response_files", 0, "sha256"): "5" * 64,
            ("outputs", 0, "parent", "path"): "/tmp",
            ("outputs", 0, "parent", "resolved_path"): "/tmp",
            ("outputs", 0, "parent", "device"): 0,
            ("outputs", 0, "parent", "inode"): 0,
        }
        for path, value in nested_mutations.items():
            with self.subTest(path=path):
                forged = copy.deepcopy(manifest)
                destination = forged
                for element in path[:-1]:
                    destination = destination[element]
                destination[path[-1]] = value
                with self.assertRaises(exact_actions.ExactActionError):
                    exact_actions.validate_manifest(forged)
        forged_digest = copy.deepcopy(manifest)
        forged_digest["manifest_digest"] = "3" * 64
        with self.assertRaises(exact_actions.ExactActionError):
            exact_actions.validate_manifest(forged_digest)

    def test_cross_mixed_valid_actions_do_not_authorize_execution(self) -> None:
        broker = exact_actions.OneShotBroker()
        first, first_issued = broker.issue("first", self._manifest())
        second, second_issued = broker.issue("second", self._manifest(marker="other"))
        self.assertTrue(first_issued)
        self.assertTrue(second_issued)
        mixed = copy.deepcopy(first)
        mixed["environment"] = second["environment"]
        self._resign(mixed)
        with self.assertRaises(exact_actions.ExactActionError):
            broker.consume(mixed)
        self.assertEqual(broker.consume(first)["action_id"], first["action_id"])
        self.assertEqual(broker.consume(second)["action_id"], second["action_id"])

    def test_atomic_consumption_allows_exactly_one_concurrent_consumer_and_no_replay(self) -> None:
        broker = exact_actions.OneShotBroker()
        issued, created = broker.issue("one", self._manifest())
        self.assertTrue(created)
        barrier = threading.Barrier(8)
        outcomes: list[str] = []
        lock = threading.Lock()

        def consume() -> None:
            barrier.wait()
            try:
                broker.consume(copy.deepcopy(issued))
                outcome = "consumed"
            except exact_actions.ExactActionError:
                outcome = "rejected"
            with lock:
                outcomes.append(outcome)

        threads = [threading.Thread(target=consume) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(outcomes.count("consumed"), 1)
        self.assertEqual(outcomes.count("rejected"), 7)
        with self.assertRaises(exact_actions.ExactActionError):
            broker.consume(issued)

    def test_crashed_client_never_receives_a_reissued_action(self) -> None:
        broker = exact_actions.OneShotBroker()
        first, created = broker.issue("fixed-parent-recipe", self._manifest())
        self.assertTrue(created)
        # A second observation after the first client dies receives the same
        # issued identity, never a fresh authorization.
        replacement, created = broker.issue("fixed-parent-recipe", self._manifest())
        self.assertFalse(created)
        self.assertEqual(replacement, first)
        broker.terminal()
        with self.assertRaises(exact_actions.ExactActionError):
            broker.issue("new", self._manifest())

    def test_live_validation_detects_input_cwd_and_output_races(self) -> None:
        input_manifest = self._manifest()
        self.input.write_bytes(b"attacker replacement\n")
        with self.assertRaises(exact_actions.ExactActionError):
            exact_actions.validate_live_manifest(input_manifest)

        self.input.write_bytes(b"reviewed input bytes\n")
        output_manifest = self._manifest()
        (self.output_parent / "result.o").write_bytes(b"attacker output\n")
        with self.assertRaises(exact_actions.ExactActionError):
            exact_actions.validate_live_manifest(output_manifest)

        (self.output_parent / "result.o").unlink()
        cwd_manifest = self._manifest()
        replacement = self.root / "replacement-cwd"
        self.cwd.rename(replacement)
        self.cwd.mkdir()
        with self.assertRaises(exact_actions.ExactActionError):
            exact_actions.validate_live_manifest(cwd_manifest)

    def test_post_validation_mutation_cannot_be_consumed_by_sealed_view(self) -> None:
        manifest = self._manifest()
        checked = exact_actions.validate_live_manifest(manifest)
        view = exact_actions.seal_input_view(checked)
        try:
            self.input.write_bytes(b"attacker bytes after validation\n")
            source = next(item for item in view.transcript()["inputs"] if item["path"] == str(self.input))
            retained = os.pread(view._descriptors[0], int(source["size_bytes"]), 0)
            self.assertEqual(retained, b"reviewed input bytes\n")
            self.assertNotEqual(retained, self.input.read_bytes())
            self.assertTrue(all(value.startswith("/proc/self/fd/") or value.startswith("--") or value.startswith("-o") or value == str(self.output_parent / "result.o") for value in view.argv))
        finally:
            view.close()

    def test_mutation_between_live_validation_and_snapshot_is_rejected(self) -> None:
        manifest = self._manifest()
        checked = exact_actions.validate_live_manifest(manifest)
        self.input.write_bytes(b"attacker bytes in the TOCTOU gap\n")
        with self.assertRaises(exact_actions.ExactActionError):
            exact_actions.seal_input_view(checked)

    def test_numeric_strings_are_not_action_authority(self) -> None:
        broker = exact_actions.OneShotBroker()
        issued, _created = broker.issue("reviewed", self._manifest())
        for value in ("1", "2", "5", "30"):
            with self.subTest(value=value):
                forged = copy.deepcopy(issued)
                forged["action_id"] = value
                with self.assertRaises(exact_actions.ExactActionError):
                    broker.consume(forged)


if __name__ == "__main__":
    unittest.main()
