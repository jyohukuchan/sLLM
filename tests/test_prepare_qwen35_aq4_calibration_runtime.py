from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "prepare-qwen35-aq4-calibration-runtime.py"
spec = importlib.util.spec_from_file_location("calibration_runtime", SCRIPT)
assert spec and spec.loader
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class CalibrationRuntimePreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_stage_binary_breaks_cargo_hardlink_and_revalidates_exact_identity(self) -> None:
        source = self.root / "source"
        source.write_bytes(b"capture-binary")
        os.link(source, self.root / "cargo-hardlink")
        output = self.root / "staged"
        receipt = self.root / "receipt.json"
        value = runtime.stage_binary(source, output, receipt)
        expected = hashlib.sha256(b"capture-binary").hexdigest()
        self.assertEqual(source.stat().st_nlink, 2)
        self.assertEqual(output.stat().st_nlink, 1)
        self.assertEqual(output.stat().st_mode & 0o7777, 0o555)
        self.assertEqual(value["staged"]["sha256"], expected)
        self.assertEqual(runtime.validate_binary(output, expected, len(b"capture-binary"))["nlink"], 1)
        self.assertEqual(json.loads(receipt.read_text())["execution_contract"]["child_self_validation_required"], True)

    def test_official_restore_poller_is_loaded_from_the_pinned_tool(self) -> None:
        promotion = runtime._promotion_module()
        self.assertEqual(promotion.poll_restored.__name__, "poll_restored")
        self.assertEqual(promotion.default_service_snapshot.__name__, "default_service_snapshot")
        self.assertEqual(promotion.default_owner_snapshot.__name__, "default_owner_snapshot")

    def test_stage_and_validate_reject_existing_symlink_hardlink_and_content_change(self) -> None:
        source = self.root / "source"
        source.write_bytes(b"capture")
        output = self.root / "staged"
        receipt = self.root / "receipt.json"
        runtime.stage_binary(source, output, receipt)
        with self.assertRaises(runtime.RuntimePreparationError):
            runtime.stage_binary(source, output, self.root / "second-receipt.json")
        hardlink = self.root / "staged-hardlink"
        os.link(output, hardlink)
        with self.assertRaises(runtime.RuntimePreparationError):
            runtime.validate_binary(output, hashlib.sha256(b"capture").hexdigest(), 7)
        hardlink.unlink()
        symlink = self.root / "staged-symlink"
        symlink.symlink_to(output)
        with self.assertRaises(runtime.RuntimePreparationError):
            runtime.binary_identity(symlink, "symlink")
        output.chmod(0o755)
        output.write_bytes(b"changed")
        output.chmod(0o555)
        with self.assertRaises(runtime.RuntimePreparationError):
            runtime.validate_binary(output, hashlib.sha256(b"capture").hexdigest(), 7)

    def test_ready_snapshot_requires_exact_main_worker_lock_amd_and_kfd_identity(self) -> None:
        service = {
            "active": True, "running": True, "healthy": True, "main_pid": 101,
            "worker_pid": 202, "nrestarts": 0, "lock_owned": True, "lock_holders": [101],
        }
        owners = {"worker_pids": [202], "amd_pids": [202], "kfd_pids": [202]}
        runtime._validate_ready(service, owners, expected_nrestarts=0)
        mutations = (
            ("main", {**service, "main_pid": 0}, owners),
            ("worker", {**service, "worker_pid": 0}, owners),
            ("lock", {**service, "lock_holders": [202]}, owners),
            ("restarts", {**service, "nrestarts": 1}, owners),
            ("worker_owner", service, {**owners, "worker_pids": []}),
            ("amd_owner", service, {**owners, "amd_pids": [303]}),
            ("kfd_owner", service, {**owners, "kfd_pids": []}),
        )
        for name, changed_service, changed_owners in mutations:
            with self.subTest(name=name), self.assertRaises(runtime.RuntimePreparationError):
                runtime._validate_ready(changed_service, changed_owners, expected_nrestarts=0)


if __name__ == "__main__":
    unittest.main()
