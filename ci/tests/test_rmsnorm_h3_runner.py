from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

import run_rmsnorm_h3_compile as runner


class RmsNormH3RunnerTests(unittest.TestCase):
    def test_output_root_is_exclusive_direct_tmp_child(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rmsnorm-runner-test-") as directory:
            existing = Path(directory) / "sllm-rmsnorm-h3-existing"
            existing.mkdir()
            with self.assertRaises(runner.ContractError):
                runner.output_root(existing)
        with self.assertRaises(runner.ContractError):
            runner.output_root(Path("/tmp/not-the-dedicated-prefix"))

    def test_rendered_commands_have_one_placeholder_target(self) -> None:
        _, _, rows = runner.validate_matrix(ROOT)
        commands = runner.render_commands(rows["h3-rmsnorm-gfx1030"], ROOT, Path("/tmp/sllm-rmsnorm-h3-build-test"))
        self.assertEqual(len(commands), 4)
        for command in commands:
            self.assertEqual(sum(token == "--offload-arch=gfx1030" for token in command), 1)
            self.assertNotIn("gfx1201", command)
            self.assertTrue(all("{" not in token and "}" not in token for token in command))
        self.assertIn("--hip-link", commands[-1])
        self.assertIn("-rtlib=compiler-rt", " ".join(commands[-1]).replace("--rtlib=", "-rtlib="))

    def test_process_cleanup_and_nonzero_fail_closed(self) -> None:
        with patch.object(runner, "COMPILER", sys.executable):
            step = runner.run_process([sys.executable, "-c", "raise SystemExit(0)"], cwd=ROOT, timeout=5, output_limit=4096)
            self.assertEqual(step["exit_code"], 0)
            with self.assertRaises(runner.ContractError):
                runner.run_process([sys.executable, "-c", "raise SystemExit(9)"], cwd=ROOT, timeout=5, output_limit=4096)

    def test_shell_tokens_are_rejected_before_spawn(self) -> None:
        with patch.object(runner, "COMPILER", sys.executable):
            with self.assertRaises(runner.ContractError):
                runner.run_process([sys.executable, "-c", "pass;echo unsafe"], cwd=ROOT, timeout=5, output_limit=4096)


if __name__ == "__main__":
    unittest.main()
