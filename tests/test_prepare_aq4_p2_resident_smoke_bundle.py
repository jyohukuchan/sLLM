from __future__ import annotations

import importlib.util
import hashlib
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


@pytest.fixture(scope="module")
def trusted_reconstruction():
    return BUNDLE.reconstruct()


def rebind_transport(root: Path) -> None:
    bundle_path = root / "bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    for name in BUNDLE.REQUIRED_FILES:
        bundle["files"][name]["sha256"] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    rewrite_json(bundle_path, bundle)
    lines = []
    for name in sorted([*BUNDLE.REQUIRED_FILES, "bundle.json"]):
        lines.append(f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n")
    sums = root / "SHA256SUMS"
    sums.chmod(0o644)
    sums.write_text("".join(lines), encoding="ascii")
    sums.chmod(0o444)


def test_checked_in_bundle_passes_offline_validation(trusted_reconstruction) -> None:
    value = BUNDLE.validate(ARTIFACT, trusted_reconstruction)
    assert value["status"] == "prepared_not_executed"
    assert value["promotion"] is False
    assert value["offline_evidence"] == {
        "trust_root_reconstruction": "passed",
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


def test_rejects_unknown_bundle_schema(tmp_path: Path, trusted_reconstruction) -> None:
    root = copy_bundle(tmp_path)
    path = root / "bundle.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["schema_version"] = "ullm.aq4_p2_resident_smoke_binding_bundle.v999"
    rewrite_json(path, value)
    rebind_transport(root)
    with pytest.raises(BUNDLE.BundleError, match="semantic reconstruction differs: bundle"):
        BUNDLE.validate(root, trusted_reconstruction)


def test_rejects_payload_hash_mutation(tmp_path: Path, trusted_reconstruction) -> None:
    root = copy_bundle(tmp_path)
    path = root / "fixture.json"
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    path.chmod(0o444)
    rebind_transport(root)
    with pytest.raises(BUNDLE.BundleError, match="semantic reconstruction differs: fixture"):
        BUNDLE.validate(root, trusted_reconstruction)


def test_rejects_unsafe_member_path(tmp_path: Path) -> None:
    with pytest.raises(BUNDLE.BundleError, match="unsafe bundle member path"):
        BUNDLE._safe_member(tmp_path, "../fixture.json")
    with pytest.raises(BUNDLE.BundleError, match="unsafe bundle member path"):
        BUNDLE._safe_member(tmp_path, "/tmp/fixture.json")


def test_rejects_symlink_member(tmp_path: Path, trusted_reconstruction) -> None:
    root = copy_bundle(tmp_path)
    fixture = root / "fixture.json"
    target = tmp_path / "external-fixture.json"
    shutil.copyfile(fixture, target)
    fixture.unlink()
    fixture.symlink_to(target)
    with pytest.raises(BUNDLE.BundleError, match="type/link/mode"):
        BUNDLE.validate(root, trusted_reconstruction)


def test_rejects_hardlink_member(tmp_path: Path, trusted_reconstruction) -> None:
    root = copy_bundle(tmp_path)
    fixture = root / "fixture.json"
    target = tmp_path / "external-fixture.json"
    shutil.copyfile(fixture, target)
    target.chmod(0o444)
    fixture.unlink()
    os.link(target, fixture)
    with pytest.raises(BUNDLE.BundleError, match="type/link/mode"):
        BUNDLE.validate(root, trusted_reconstruction)


def test_final_pass_detects_toctou_mutation(tmp_path: Path, trusted_reconstruction) -> None:
    root = copy_bundle(tmp_path)

    def mutate(bundle_root: Path) -> None:
        path = bundle_root / "policy.json"
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b" ")
        path.chmod(0o444)

    BUNDLE._VALIDATION_HOOK = mutate
    try:
        with pytest.raises(BUNDLE.BundleError, match="TOCTOU mutation detected"):
            BUNDLE.validate(root, trusted_reconstruction)
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


@pytest.mark.parametrize(
    "variant",
    ("official_case", "runtime_case", "fixture", "identity", "preflight", "policy", "served_model", "trust_roots"),
)
def test_semantic_rebind_is_rejected_from_independent_trust_roots(tmp_path: Path, trusted_reconstruction, variant: str) -> None:
    root = copy_bundle(tmp_path)
    if variant == "official_case":
        path = root / "official-case.json"; value = json.loads(path.read_text()); value["case"]["prompt_tokens"] = 129
    elif variant == "runtime_case":
        path = root / "case-binding.json"; value = json.loads(path.read_text()); value["cases"][0]["device"]["runtime_device_index"] = 0; value["runtime_binding"]["bound_device"]["runtime_device_index"] = 0
    elif variant == "fixture":
        path = root / "fixture.json"; value = json.loads(path.read_text()); value["cases"][0]["prompt_token_ids"][0] += 1
    elif variant == "identity":
        path = root / "identity.json"; value = json.loads(path.read_text()); value["resident_driver_identity"]["model_revision"] = "semantic-rebind"
    elif variant == "preflight":
        path = root / "preflight.json"; value = json.loads(path.read_text()); value["workspace_bytes"] = 1
    elif variant == "policy":
        path = root / "policy.json"; value = json.loads(path.read_text()); value["status"] = "semantic-rebind"
    elif variant == "served_model":
        path = root / "served-model.json"; value = json.loads(path.read_text()); value["public"]["revision"] = "semantic-rebind"
    else:
        path = root / "trust-roots.json"; value = json.loads(path.read_text()); value["source"]["tree"] = "0" * 40
    rewrite_json(path, value)
    rebind_transport(root)
    with pytest.raises(BUNDLE.BundleError, match="independent semantic reconstruction differs"):
        BUNDLE.validate(root, trusted_reconstruction)


def test_nested_unknown_and_duplicate_fields_are_rejected(tmp_path: Path, trusted_reconstruction) -> None:
    unknown_root = copy_bundle(tmp_path / "unknown")
    policy = unknown_root / "policy.json"
    value = json.loads(policy.read_text()); value["nested_unknown"] = {"accepted": False}
    rewrite_json(policy, value); rebind_transport(unknown_root)
    with pytest.raises(BUNDLE.BundleError, match="semantic reconstruction differs"):
        BUNDLE.validate(unknown_root, trusted_reconstruction)

    duplicate_root = copy_bundle(tmp_path / "duplicate")
    identity = duplicate_root / "identity.json"
    raw = identity.read_text(encoding="utf-8").replace('  "status": "bound"', '  "status": "bound",\n  "status": "bound"', 1)
    identity.chmod(0o644); identity.write_text(raw, encoding="utf-8"); identity.chmod(0o444)
    rebind_transport(duplicate_root)
    with pytest.raises(BUNDLE.BundleError, match="duplicate JSON key"):
        BUNDLE.validate(duplicate_root, trusted_reconstruction)
