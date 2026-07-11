from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence, cast
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "sq8_full_campaign_production.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PRODUCTION = load_module("test_sq8_full_campaign_production_module", MODULE_PATH)


VALIDATOR_SOURCE = r"""#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import sq8_canonical_artifact as canonical


root = Path(sys.argv[1])
mode_path = root / "mode"
if not mode_path.is_file() or canonical.VALUE != "head-canonical":
    raise SystemExit(23)
mode = mode_path.read_text(encoding="ascii").strip()
if mode == "timeout":
    time.sleep(10)
if mode == "stderr":
    sys.stderr.write("unexpected diagnostic\n")
if mode == "nonzero":
    raise SystemExit(19)
if mode == "oversize":
    sys.stdout.buffer.write(b"x" * 4096)
    raise SystemExit(0)
if mode == "duplicate":
    sys.stdout.write('{"schema_version":"first","schema_version":"second"}')
    raise SystemExit(0)
if mode == "nonfinite":
    sys.stdout.write('{"schema_version":NaN}')
    raise SystemExit(0)
if mode == "overflow-float":
    sys.stdout.write('{"schema_version":1e999}')
    raise SystemExit(0)

receipt = {
    "schema_version": "ullm.sq8_product_promotion.v1",
    "product_root": str(root),
    "created_at": "2026-07-12T00:00:00Z",
    "model_revision": "9a283b4a5efbc09ce247e0ae5b02b744739e525a",
    "artifact": {"payloads_hashed": True},
    "package": {"payloads_hashed": True},
    "read_only": True,
    "full_payloads": True,
    "verified": True,
}
if mode == "payload-false":
    receipt["full_payloads"] = False
if mode == "read-only-false":
    receipt["read_only"] = False
if mode == "verified-false":
    receipt["verified"] = False
if mode == "artifact-payload-false":
    receipt["artifact"]["payloads_hashed"] = False
print(json.dumps(receipt, sort_keys=True))
"""

CANONICAL_SOURCE = 'VALUE = "head-canonical"\n'


class ProductionFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.product = self.root / "product"
        self.runtime = self.root / "runtime"
        self.repo.mkdir(mode=0o700)
        self.product.mkdir(mode=0o700)
        self.runtime.mkdir(mode=0o700)
        (self.repo / "tools").mkdir()
        self.validator_source = (
            self.repo / "tools" / "validate-sq8-product-promotion.py"
        )
        self.canonical_source = self.repo / "tools" / "sq8_canonical_artifact.py"
        self.validator_source.write_text(VALIDATOR_SOURCE, encoding="ascii")
        self.canonical_source.write_text(CANONICAL_SOURCE, encoding="ascii")
        self.set_mode("normal")
        self._git("init", "-q")
        self._git("config", "user.email", "production-test@example.invalid")
        self._git("config", "user.name", "Production Test")
        self._git("add", "tools")
        self._git("commit", "-qm", "add pinned promotion tools")
        self.commit = self._git("rev-parse", "HEAD").stdout.decode("ascii").strip()
        self.settings = PRODUCTION.ProductionPreflightSettings(
            repo_root=self.repo,
            product_root=self.product,
            python_executable=Path(sys.executable).resolve(),
            private_runtime_parent=self.runtime,
        )

    def __enter__(self) -> ProductionFixture:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.temporary.cleanup()

    def _git(self, *args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ("/usr/bin/git", "-C", os.fspath(self.repo), *args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    def set_mode(self, mode: str) -> None:
        (self.product / "mode").write_text(mode, encoding="ascii")

    def anchor(self) -> Any:
        return PRODUCTION.GitAnchor.capture(self.settings, expected_commit=self.commit)


class RecordingRunner:
    def __init__(self) -> None:
        self.argvs: list[tuple[str, ...]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        stdout_limit: int,
        stderr_limit: int,
    ) -> Any:
        self.argvs.append(tuple(argv))
        return PRODUCTION.SYSTEM_COMMAND_RUNNER.run(
            argv,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
        )


class CallbackRunner(RecordingRunner):
    def __init__(self, callback: Any) -> None:
        super().__init__()
        self.callback = callback

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        stdout_limit: int,
        stderr_limit: int,
    ) -> Any:
        result = super().run(
            argv,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
        )
        if len(argv) == 4 and argv[1] == "-B":
            self.callback()
        return result


class KeyboardInterruptRunner(RecordingRunner):
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        stdout_limit: int,
        stderr_limit: int,
    ) -> Any:
        if len(argv) == 4 and argv[1] == "-B":
            raise KeyboardInterrupt
        return super().run(
            argv,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
        )


class ProductionSettingsTests(unittest.TestCase):
    def test_settings_are_frozen_and_lock_path_has_no_override(self) -> None:
        settings = PRODUCTION.production_preflight_settings()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            settings.repo_root = Path("/tmp/replacement")
        self.assertEqual(
            len(inspect.signature(PRODUCTION.canonical_campaign_lock_path).parameters),
            0,
        )
        self.assertEqual(
            PRODUCTION.canonical_campaign_lock_path(),
            Path("/run/user")
            / str(os.geteuid())
            / "ullm-sq8-full-openwebui-campaign.lock",
        )

    def test_settings_reject_relative_and_noncanonical_paths(self) -> None:
        with self.assertRaisesRegex(
            PRODUCTION.ProductionPreflightError, "absolute Path"
        ):
            PRODUCTION.ProductionPreflightSettings(
                Path("repo"), Path("/product"), Path("/python"), Path("/runtime")
            )
        with self.assertRaisesRegex(
            PRODUCTION.ProductionPreflightError, "lexically canonical"
        ):
            PRODUCTION.ProductionPreflightSettings(
                Path("/repo/../other"),
                Path("/product"),
                Path("/python"),
                Path("/runtime"),
            )


class BoundedCommandRunnerTests(unittest.TestCase):
    def test_runner_times_out_and_bounds_stdout_and_stderr_during_reads(self) -> None:
        runner = PRODUCTION.BoundedCommandRunner()
        python = os.fspath(Path(sys.executable).resolve())
        cwd = Path(tempfile.gettempdir())
        with self.assertRaisesRegex(PRODUCTION.ProductionPreflightError, "timed out"):
            runner.run(
                (python, "-c", "import time; time.sleep(10)"),
                cwd=cwd,
                timeout_seconds=0.05,
                stdout_limit=64,
                stderr_limit=64,
            )
        for stream, program in (
            ("stdout", "import sys; sys.stdout.buffer.write(b'x'*4096)"),
            ("stderr", "import sys; sys.stderr.buffer.write(b'x'*4096)"),
        ):
            with (
                self.subTest(stream=stream),
                self.assertRaisesRegex(
                    PRODUCTION.ProductionPreflightError,
                    rf"{stream} exceeded its byte limit",
                ),
            ):
                runner.run(
                    (python, "-c", program),
                    cwd=cwd,
                    timeout_seconds=2.0,
                    stdout_limit=64,
                    stderr_limit=64,
                )

    def test_runner_wraps_process_start_failure(self) -> None:
        with self.assertRaisesRegex(
            PRODUCTION.ProductionPreflightError,
            "failed to execute a bounded command",
        ):
            PRODUCTION.BoundedCommandRunner().run(
                ("/definitely/not/a/command",),
                cwd=Path(tempfile.gettempdir()),
                timeout_seconds=1.0,
                stdout_limit=64,
                stderr_limit=64,
            )


class GitAnchorTests(unittest.TestCase):
    def test_capture_preserves_exact_untracked_status_bytes(self) -> None:
        with ProductionFixture() as fixture:
            marker = fixture.repo / ".rocprofv3" / "marker"
            marker.parent.mkdir()
            marker.write_text("leave this alone\n", encoding="ascii")
            anchor = fixture.anchor()
            self.assertEqual(anchor.status_raw, b"?? .rocprofv3/marker\x00")
            self.assertTrue(marker.is_file())
            anchor.revalidate()

    def test_capture_requires_explicit_matching_full_commit(self) -> None:
        with ProductionFixture() as fixture:
            with self.assertRaisesRegex(
                PRODUCTION.ProductionPreflightError, "exactly 40"
            ):
                PRODUCTION.GitAnchor.capture(
                    fixture.settings, expected_commit=fixture.commit[:8]
                )
            with self.assertRaisesRegex(
                PRODUCTION.ProductionPreflightError, "explicit expected commit"
            ):
                PRODUCTION.GitAnchor.capture(fixture.settings, expected_commit="0" * 40)

    def test_revalidate_rejects_status_and_head_drift(self) -> None:
        with ProductionFixture() as fixture:
            anchor = fixture.anchor()
            (fixture.repo / "untracked").write_text("drift\n", encoding="ascii")
            with self.assertRaisesRegex(
                PRODUCTION.ProductionPreflightError, "Git anchor drifted"
            ):
                anchor.revalidate()

        with ProductionFixture() as fixture:
            anchor = fixture.anchor()
            (fixture.repo / "committed").write_text("next\n", encoding="ascii")
            fixture._git("add", "committed")
            fixture._git("commit", "-qm", "move HEAD")
            with self.assertRaisesRegex(
                PRODUCTION.ProductionPreflightError,
                "explicit expected commit",
            ):
                anchor.revalidate()


class HeadToolSnapshotTests(unittest.TestCase):
    def test_snapshot_ignores_dirty_worktree_content_and_uses_head(self) -> None:
        with ProductionFixture() as fixture:
            fixture.validator_source.write_text(
                "raise SystemExit('worktree source must not execute')\n",
                encoding="ascii",
            )
            anchor = fixture.anchor()
            with PRODUCTION.HeadPromotionToolSnapshotOwner.create(
                fixture.settings, anchor
            ) as owner:
                self.assertEqual(
                    owner.validator_path.read_text(encoding="ascii"), VALIDATOR_SOURCE
                )

    def test_snapshot_uses_exact_head_blobs_and_cleans_up(self) -> None:
        with ProductionFixture() as fixture:
            anchor = fixture.anchor()
            owner = PRODUCTION.HeadPromotionToolSnapshotOwner.create(
                fixture.settings, anchor
            )
            root = owner.root
            try:
                self.assertEqual(
                    owner.validator_path.read_text(encoding="ascii"), VALIDATOR_SOURCE
                )
                self.assertEqual(
                    owner.canonical_path.read_text(encoding="ascii"), CANONICAL_SOURCE
                )
                self.assertEqual(root.stat().st_mode & 0o777, 0o700)
                self.assertEqual((root / "tools").stat().st_mode & 0o777, 0o700)
                self.assertEqual(owner.validator_path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(owner.canonical_path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(set(root.iterdir()), {root / "tools"})
                owner.revalidate()
            finally:
                owner.close()
            self.assertFalse(root.exists())

    def test_snapshot_allows_an_unrelated_private_sibling(self) -> None:
        with ProductionFixture() as fixture:
            anchor = fixture.anchor()
            owner = PRODUCTION.HeadPromotionToolSnapshotOwner.create(
                fixture.settings, anchor
            )
            sibling = fixture.runtime / "other-owned-runtime-state"
            sibling.mkdir(mode=0o700)
            try:
                owner.revalidate()
                owner.close()
            finally:
                sibling.rmdir()
            self.assertFalse(owner.root.exists())

    def test_snapshot_rejects_private_copy_and_worktree_source_changes(self) -> None:
        with ProductionFixture() as fixture:
            anchor = fixture.anchor()
            owner = PRODUCTION.HeadPromotionToolSnapshotOwner.create(
                fixture.settings, anchor
            )
            root = owner.root
            owner.canonical_path.write_text("VALUE = 'changed'\n", encoding="ascii")
            with self.assertRaisesRegex(
                PRODUCTION.ProductionPreflightError, "snapshot changed"
            ):
                owner.revalidate()
            with self.assertRaises(PRODUCTION.ProductionPreflightError):
                owner.close()
            self.assertFalse(root.exists())

        with ProductionFixture() as fixture:
            anchor = fixture.anchor()
            owner = PRODUCTION.HeadPromotionToolSnapshotOwner.create(
                fixture.settings, anchor
            )
            root = owner.root
            fixture.canonical_source.write_text(
                "VALUE = 'worktree-drift'\n", encoding="ascii"
            )
            with self.assertRaisesRegex(
                PRODUCTION.ProductionPreflightError, "Git anchor drifted"
            ):
                owner.revalidate()
            with self.assertRaises(PRODUCTION.ProductionPreflightError):
                owner.close()
            self.assertFalse(root.exists())

    def test_snapshot_rejects_symlink_substitution_and_still_cleans(self) -> None:
        with ProductionFixture() as fixture:
            anchor = fixture.anchor()
            owner = PRODUCTION.HeadPromotionToolSnapshotOwner.create(
                fixture.settings, anchor
            )
            root = owner.root
            owner.validator_path.unlink()
            owner.validator_path.symlink_to(fixture.validator_source)
            with self.assertRaisesRegex(
                PRODUCTION.ProductionPreflightError, "snapshot changed"
            ):
                owner.revalidate()
            with self.assertRaises(PRODUCTION.ProductionPreflightError):
                owner.close()
            self.assertFalse(root.exists())

    def test_creation_cleans_partial_snapshot_on_keyboard_interrupt(self) -> None:
        with ProductionFixture() as fixture:
            anchor = fixture.anchor()
            original = PRODUCTION._read_head_blob
            calls = 0

            def interrupt_second(*args: Any, **kwargs: Any) -> bytes:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt
                return cast(bytes, original(*args, **kwargs))

            with (
                mock.patch.object(
                    PRODUCTION, "_read_head_blob", side_effect=interrupt_second
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                PRODUCTION.HeadPromotionToolSnapshotOwner.create(
                    fixture.settings, anchor
                )
            self.assertEqual(list(fixture.runtime.iterdir()), [])


class PromotionValidationTests(unittest.TestCase):
    def run_validation(
        self,
        fixture: ProductionFixture,
        *,
        runner: Any = PRODUCTION.SYSTEM_COMMAND_RUNNER,
    ) -> dict[str, Any]:
        anchor = PRODUCTION.GitAnchor.capture(
            fixture.settings,
            expected_commit=fixture.commit,
            runner=runner,
        )
        with PRODUCTION.HeadPromotionToolSnapshotOwner.create(
            fixture.settings, anchor, runner=runner
        ) as owner:
            return cast(
                dict[str, Any],
                PRODUCTION.run_pinned_full_promotion_validation(
                    fixture.settings, anchor, owner, runner=runner
                ),
            )

    def test_full_validation_is_delegated_without_metadata_only(self) -> None:
        with ProductionFixture() as fixture:
            runner = RecordingRunner()
            receipt = self.run_validation(fixture, runner=runner)
            self.assertIs(receipt["full_payloads"], True)
            validation_argv = [
                argv for argv in runner.argvs if len(argv) == 4 and argv[1] == "-B"
            ]
            self.assertEqual(len(validation_argv), 1)
            self.assertNotIn("--metadata-only", validation_argv[0])
            self.assertEqual(validation_argv[0][-1], os.fspath(fixture.product))

    def test_validation_rejects_stderr_nonzero_and_strict_json_failures(self) -> None:
        cases = {
            "stderr": "wrote to stderr",
            "nonzero": "validation failed",
            "duplicate": "duplicate JSON key",
            "nonfinite": "non-finite JSON number",
            "overflow-float": "non-finite JSON number",
            "payload-false": "full_payloads flag is not true",
            "read-only-false": "read_only flag is not true",
            "verified-false": "verified flag is not true",
            "artifact-payload-false": "artifact payload hashing flag is not true",
        }
        for mode, message in cases.items():
            with self.subTest(mode=mode), ProductionFixture() as fixture:
                fixture.set_mode(mode)
                with self.assertRaisesRegex(
                    PRODUCTION.ProductionPreflightError, message
                ):
                    self.run_validation(fixture)

    def test_validation_rejects_timeout_and_oversize_during_capture(self) -> None:
        for mode, attribute, value, message in (
            ("timeout", "PROMOTION_TIMEOUT_SECONDS", 0.05, "timed out"),
            (
                "oversize",
                "PROMOTION_STDOUT_MAX_BYTES",
                128,
                "stdout exceeded its byte limit",
            ),
        ):
            with self.subTest(mode=mode), ProductionFixture() as fixture:
                fixture.set_mode(mode)
                with (
                    mock.patch.object(PRODUCTION, attribute, value),
                    self.assertRaisesRegex(
                        PRODUCTION.ProductionPreflightError, message
                    ),
                ):
                    self.run_validation(fixture)

    def test_validation_rejects_product_root_symlink_before_delegation(self) -> None:
        with ProductionFixture() as fixture:
            link = fixture.root / "product-link"
            link.symlink_to(fixture.product, target_is_directory=True)
            settings = dataclasses.replace(fixture.settings, product_root=link)
            anchor = PRODUCTION.GitAnchor.capture(
                settings, expected_commit=fixture.commit
            )
            with PRODUCTION.HeadPromotionToolSnapshotOwner.create(
                settings, anchor
            ) as owner:
                with self.assertRaisesRegex(
                    PRODUCTION.ProductionPreflightError, "symbolic link"
                ):
                    PRODUCTION.run_pinned_full_promotion_validation(
                        settings, anchor, owner
                    )

    def test_validation_detects_source_change_after_subprocess(self) -> None:
        with ProductionFixture() as fixture:
            anchor = fixture.anchor()
            owner = PRODUCTION.HeadPromotionToolSnapshotOwner.create(
                fixture.settings, anchor
            )
            root = owner.root
            runner = CallbackRunner(
                lambda: owner.canonical_path.write_text(
                    "VALUE = 'changed-after-run'\n", encoding="ascii"
                )
            )
            with self.assertRaisesRegex(
                PRODUCTION.ProductionPreflightError, "snapshot changed"
            ):
                PRODUCTION.run_pinned_full_promotion_validation(
                    fixture.settings, anchor, owner, runner=runner
                )
            with self.assertRaises(PRODUCTION.ProductionPreflightError):
                owner.close()
            self.assertFalse(root.exists())

    def test_keyboard_interrupt_is_preserved_and_context_cleans_owner(self) -> None:
        with ProductionFixture() as fixture:
            runner = KeyboardInterruptRunner()
            anchor = PRODUCTION.GitAnchor.capture(
                fixture.settings,
                expected_commit=fixture.commit,
                runner=runner,
            )
            owner_root: Path | None = None
            with self.assertRaises(KeyboardInterrupt):
                with PRODUCTION.HeadPromotionToolSnapshotOwner.create(
                    fixture.settings, anchor, runner=runner
                ) as owner:
                    owner_root = owner.root
                    PRODUCTION.run_pinned_full_promotion_validation(
                        fixture.settings, anchor, owner, runner=runner
                    )
            assert owner_root is not None
            self.assertFalse(owner_root.exists())


if __name__ == "__main__":
    unittest.main()
