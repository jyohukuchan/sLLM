from __future__ import annotations

import hashlib
import importlib.util
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "sq8_full_campaign_bundle.py"
VALIDATOR_PATH = ROOT / "tools" / "validate-sq8-openwebui-release.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUNDLE = load_module("test_sq8_full_campaign_bundle_tool", TOOL_PATH)
VALIDATOR = load_module("test_sq8_full_campaign_bundle_validator", VALIDATOR_PATH)


def permit(_raw: bytes, _label: str) -> None:
    return None


class BundleFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.parent = Path(self.temporary.name)
        self.final = self.parent / "sq8-openwebui-v0.1"
        self.bundle = BUNDLE.AtomicCampaignDirectory(
            self.final, uid=os.getuid(), gid=os.getgid()
        )

    def close(self) -> None:
        if not self.bundle.closed and not self.bundle.published:
            self.bundle.abort()
        self.temporary.cleanup()

    def __enter__(self) -> BundleFixture:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def populate(self) -> None:
        for relative in BUNDLE.expected_prevalidation_paths():
            self.bundle.write_bytes(relative, b"evidence\n", scan=permit)

    def add_validation(self) -> None:
        path = self.bundle.stage_path / BUNDLE.VALIDATION_FILE
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        try:
            os.write(descriptor, b'{"validated":true}\n')
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class FullCampaignBundleTests(unittest.TestCase):
    def test_frozen_paths_equal_the_independent_validator_contract(self) -> None:
        self.assertEqual(
            set(BUNDLE.expected_prevalidation_paths()), VALIDATOR.BUNDLE_FILES
        )

    def test_exact_prevalidation_layout_then_one_time_publication(self) -> None:
        with BundleFixture() as fixture:
            component = fixture.bundle.component_directory("api-contract")
            (component / "temporary").write_bytes(b"not final evidence")
            fixture.populate()
            fixture.bundle.validate_before_independent_validator()
            fixture.bundle.clear_component_work()
            fixture.add_validation()
            published = fixture.bundle.publish()

            self.assertEqual(published, fixture.final)
            self.assertTrue(fixture.bundle.published)
            self.assertTrue(fixture.bundle.closed)
            self.assertFalse(fixture.bundle.stage_path.exists())
            self.assertFalse(fixture.bundle.work_path.exists())
            self.assertEqual(
                set(path.name for path in published.iterdir()),
                set(BUNDLE.PREVALIDATION_ROOT_FILES)
                | {"browser", BUNDLE.VALIDATION_FILE},
            )
            self.assertEqual(
                set(path.name for path in (published / "browser").iterdir()),
                set(BUNDLE.BROWSER_FILES),
            )
            for path in published.rglob("*"):
                mode = stat.S_IMODE(path.lstat().st_mode)
                self.assertEqual(mode, 0o700 if path.is_dir() else 0o600)

    def test_publish_requires_independent_validation_file(self) -> None:
        with BundleFixture() as fixture:
            fixture.populate()
            fixture.bundle.validate_before_independent_validator()
            with self.assertRaises(BUNDLE.CampaignBundleError):
                fixture.bundle.publish()
            self.assertFalse(fixture.final.exists())

    def test_publish_requires_empty_external_component_work_root(self) -> None:
        with BundleFixture() as fixture:
            fixture.populate()
            fixture.bundle.component_directory("latency")
            fixture.add_validation()
            with self.assertRaisesRegex(
                BUNDLE.CampaignBundleError, "work root is not empty"
            ):
                fixture.bundle.publish()
            self.assertFalse(fixture.final.exists())

    def test_layout_rejects_missing_extra_wrong_mode_and_symlink(self) -> None:
        defects = ("missing", "extra", "mode", "symlink")
        for defect in defects:
            with self.subTest(defect=defect), BundleFixture() as fixture:
                fixture.populate()
                target = fixture.bundle.stage_path / "environment.json"
                if defect == "missing":
                    target.unlink()
                elif defect == "extra":
                    extra = fixture.bundle.stage_path / "producer-verdict.json"
                    extra.write_bytes(b"extra\n")
                    os.chmod(extra, 0o600)
                elif defect == "mode":
                    os.chmod(target, 0o644)
                else:
                    target.unlink()
                    target.symlink_to(fixture.bundle.stage_path / "model-identity.json")
                with self.assertRaises(BUNDLE.CampaignBundleError):
                    fixture.bundle.validate_before_independent_validator()

    def test_existing_destination_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            final = parent / "existing"
            final.mkdir()
            marker = final / "marker"
            marker.write_bytes(b"keep")
            with self.assertRaises(BUNDLE.CampaignBundleError):
                BUNDLE.AtomicCampaignDirectory(final, uid=os.getuid(), gid=os.getgid())
            self.assertEqual(marker.read_bytes(), b"keep")

    def test_write_bytes_is_exclusive_atomic_and_secret_scanned(self) -> None:
        secret = b"private-release-token"

        def reject(raw: bytes, _label: str) -> None:
            if secret in raw:
                raise BUNDLE.CampaignBundleError("secret found")

        with BundleFixture() as fixture:
            with self.assertRaisesRegex(BUNDLE.CampaignBundleError, "secret found"):
                fixture.bundle.write_bytes(
                    "environment.json", b"prefix" + secret, scan=reject
                )
            self.assertFalse(fixture.bundle.artifact_path("environment.json").exists())
            self.assertFalse(
                any(
                    "incomplete" in name for name in os.listdir(fixture.bundle.stage_fd)
                )
            )
            evidence = fixture.bundle.write_bytes(
                "environment.json", b"safe\n", scan=reject
            )
            self.assertEqual(evidence.bytes, 5)
            self.assertEqual(evidence.sha256, hashlib.sha256(b"safe\n").hexdigest())
            with self.assertRaises(BUNDLE.CampaignBundleError):
                fixture.bundle.write_bytes("environment.json", b"second\n", scan=reject)

    def test_write_bytes_rejects_an_unbounded_inline_artifact(self) -> None:
        with BundleFixture() as fixture:
            with self.assertRaises(BUNDLE.CampaignBundleError):
                fixture.bundle.write_bytes(
                    "environment.json",
                    b"x" * (BUNDLE.MAX_INLINE_BYTES + 1),
                    scan=permit,
                )
            self.assertFalse(fixture.bundle.artifact_path("environment.json").exists())

    def test_copy_file_streams_exact_bound_source(self) -> None:
        with BundleFixture() as fixture:
            source = fixture.parent / "source.png"
            raw = b"\x89PNG\r\n\x1a\n" + b"x" * (BUNDLE.COPY_CHUNK_BYTES + 17)
            source.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            evidence = fixture.bundle.copy_file(
                source,
                "browser/openwebui-stop-before.png",
                expected_bytes=len(raw),
                expected_sha256=digest,
                maximum_bytes=len(raw),
                scan=permit,
            )
            self.assertEqual(evidence, BUNDLE.FileEvidence(len(raw), digest))
            self.assertEqual(
                fixture.bundle.artifact_path(
                    "browser/openwebui-stop-before.png"
                ).read_bytes(),
                raw,
            )

    def test_copy_file_rejects_hash_symlink_and_hardlink_sources(self) -> None:
        for defect in ("hash", "symlink", "hardlink"):
            with self.subTest(defect=defect), BundleFixture() as fixture:
                source = fixture.parent / "source"
                raw = b"source bytes\n"
                source.write_bytes(raw)
                expected = hashlib.sha256(raw).hexdigest()
                if defect == "hash":
                    expected = "0" * 64
                elif defect == "symlink":
                    real = fixture.parent / "real"
                    source.rename(real)
                    source.symlink_to(real)
                else:
                    os.link(source, fixture.parent / "second-link")
                with self.assertRaises(BUNDLE.CampaignBundleError):
                    fixture.bundle.copy_file(
                        source,
                        "browser/post-header-failure.png",
                        expected_bytes=len(raw),
                        expected_sha256=expected,
                        maximum_bytes=1024,
                        scan=permit,
                    )
                self.assertFalse(
                    fixture.bundle.artifact_path(
                        "browser/post-header-failure.png"
                    ).exists()
                )

    def test_component_labels_are_bounded_unique_and_cleaned_on_abort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "campaign"
            bundle = BUNDLE.AtomicCampaignDirectory(
                final, uid=os.getuid(), gid=os.getgid()
            )
            stage = bundle.stage_path
            work = bundle.work_path
            component = bundle.component_directory("direct-cancel")
            (component / "nested").mkdir()
            (component / "nested" / "raw").write_bytes(b"raw")
            with self.assertRaises(BUNDLE.CampaignBundleError):
                bundle.component_directory("direct-cancel")
            for invalid in ("", "../escape", "UPPER", "a" * 65):
                with self.subTest(label=invalid):
                    with self.assertRaises(BUNDLE.CampaignBundleError):
                        bundle.component_directory(invalid)
            bundle.abort()
            self.assertFalse(stage.exists())
            self.assertFalse(work.exists())
            self.assertFalse(final.exists())

    def test_validation_file_must_be_regular_private_and_nonempty(self) -> None:
        defects = ("empty", "mode", "symlink")
        for defect in defects:
            with self.subTest(defect=defect), BundleFixture() as fixture:
                fixture.populate()
                validation = fixture.bundle.stage_path / BUNDLE.VALIDATION_FILE
                if defect == "symlink":
                    validation.symlink_to(
                        fixture.bundle.stage_path / "environment.json"
                    )
                else:
                    validation.write_bytes(b"" if defect == "empty" else b"{}\n")
                    os.chmod(validation, 0o600 if defect == "empty" else 0o644)
                with self.assertRaises(BUNDLE.CampaignBundleError):
                    fixture.bundle.publish()


if __name__ == "__main__":
    unittest.main()
