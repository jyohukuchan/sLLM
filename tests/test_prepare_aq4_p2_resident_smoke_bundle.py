from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1"
SPEC = importlib.util.spec_from_file_location(
    "aq4_p2_resident_smoke_bundle",
    ROOT / "tools/prepare-aq4-p2-resident-smoke-bundle.py",
)
assert SPEC and SPEC.loader
BUNDLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUNDLE)


def copy_bundle(tmp_path: Path) -> Path:
    destination = tmp_path / "bundle"
    shutil.copytree(ARTIFACT, destination)
    return destination


def rewrite_json(path: Path, value: dict) -> None:
    path.chmod(0o644)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o444)


def test_checked_in_bundle_passes_offline_validation() -> None:
    value = BUNDLE.validate(ARTIFACT)
    assert value["status"] == "prepared_not_executed"
    assert value["promotion"] is False
    assert value["offline_evidence"] == {
        "schema_hash_path_link_toctou_validation": "passed",
        "runner_dry_run": "passed",
        "synthetic_fake_ready_validation": "passed",
        "model_load_executed": False,
        "gpu_command_executed": False,
        "service_touched": False,
    }
    assert value["actual_live_observations"]["runtime_identity"] is None
    assert value["actual_live_observations"]["power"] is None
    assert value["actual_live_observations"]["vram"] is None


def test_rejects_unknown_bundle_schema(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    path = root / "bundle.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["schema_version"] = "ullm.aq4_p2_resident_smoke_binding_bundle.v999"
    rewrite_json(path, value)
    with pytest.raises(BUNDLE.BundleError, match="schema/status/promotion"):
        BUNDLE.validate(root)


def test_rejects_payload_hash_mutation(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    path = root / "fixture.json"
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    path.chmod(0o444)
    with pytest.raises(BUNDLE.BundleError, match="declared hash"):
        BUNDLE.validate(root)


def test_rejects_unsafe_member_path(tmp_path: Path) -> None:
    with pytest.raises(BUNDLE.BundleError, match="unsafe bundle member path"):
        BUNDLE._safe_member(tmp_path, "../fixture.json")
    with pytest.raises(BUNDLE.BundleError, match="unsafe bundle member path"):
        BUNDLE._safe_member(tmp_path, "/tmp/fixture.json")


def test_rejects_symlink_member(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    fixture = root / "fixture.json"
    target = tmp_path / "external-fixture.json"
    shutil.copyfile(fixture, target)
    fixture.unlink()
    fixture.symlink_to(target)
    with pytest.raises(BUNDLE.BundleError, match="type/link/mode"):
        BUNDLE.validate(root)


def test_rejects_hardlink_member(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    fixture = root / "fixture.json"
    target = tmp_path / "external-fixture.json"
    shutil.copyfile(fixture, target)
    target.chmod(0o444)
    fixture.unlink()
    os.link(target, fixture)
    with pytest.raises(BUNDLE.BundleError, match="type/link/mode"):
        BUNDLE.validate(root)


def test_final_pass_detects_toctou_mutation(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)

    def mutate(bundle_root: Path) -> None:
        path = bundle_root / "policy.json"
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b" ")
        path.chmod(0o444)

    BUNDLE._VALIDATION_HOOK = mutate
    try:
        with pytest.raises(BUNDLE.BundleError, match="TOCTOU mutation detected"):
            BUNDLE.validate(root)
    finally:
        BUNDLE._VALIDATION_HOOK = None


def test_package_tree_hash_matches_driver_algorithm_and_rejects_symlink(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "a.bin").write_bytes(b"a")
    nested = package / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"b")
    aggregate = __import__("hashlib").sha256()
    for relative in ("a.bin", "nested/b.bin"):
        digest = __import__("hashlib").sha256((package / relative).read_bytes()).digest()
        aggregate.update(relative.encode() + b"\0" + digest + b"\n")
    actual, count = BUNDLE.package_tree_sha256(package)
    assert actual == aggregate.hexdigest()
    assert count == 2
    (package / "link").symlink_to(package / "a.bin")
    with pytest.raises(BUNDLE.BundleError, match="symlink rejected"):
        BUNDLE.package_tree_sha256(package)
