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
BINDING = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-binding-v4"
VALIDATOR_COMMIT = "82635456825503c535ce0b662e72a7a233d18c40"
VALIDATOR_SHA = "02367c9369e2e25f2a768d4cd81812b82dc58973ad0882d938c283ddf7793ab7"
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
        if name in bundle["files"]:
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
        "trusted_runner_subprocess_required": True,
        "runner_dry_run": "passed",
        "synthetic_fake_ready_validation": "passed",
        "model_load_executed": False,
        "gpu_command_executed": False,
        "service_touched": False,
    }
    assert value["actual_live_observations"]["runtime_identity"] is None
    assert value["actual_live_observations"]["power"] is None
    assert value["actual_live_observations"]["vram"] is None
    assert value["resident_driver"]["source_commit"] == BUNDLE.DRIVER_COMMIT
    assert value["runner"]["source_commit"] == BUNDLE.SOURCE_COMMIT
    assert value["historical_predecessor"] == {
        "source_commit": "0fd7993843d0d7f1096d89079ce06922871d9f1a",
        "status": "superseded_historical_prepared",
        "execution_eligible": False,
    }
    superseded = json.loads((ARTIFACT / "SUPERSEDED-0fd7993.json").read_text())
    assert superseded["execution_eligible"] is False
    assert superseded["promotion"] is False


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


@pytest.mark.parametrize("variant", ("late_unknown", "late_missing", "late_replace"))
def test_final_directory_reenumeration_rejects_late_mutation(tmp_path: Path, trusted_reconstruction, variant: str) -> None:
    root = copy_bundle(tmp_path)
    replacement = tmp_path / "replacement-policy.json"
    shutil.copyfile(root / "policy.json", replacement)
    replacement.chmod(0o444)

    def mutate(bundle_root: Path) -> None:
        if variant == "late_unknown":
            (bundle_root / "late-unknown.json").write_text("{}\n", encoding="utf-8")
        elif variant == "late_missing":
            (bundle_root / "policy.json").unlink()
        else:
            (bundle_root / "policy.json").unlink()
            shutil.copyfile(replacement, bundle_root / "policy.json")
            (bundle_root / "policy.json").chmod(0o444)

    BUNDLE._VALIDATION_HOOK = mutate
    try:
        with pytest.raises(BUNDLE.BundleError, match="late bundle directory mutation"):
            BUNDLE.validate(root, trusted_reconstruction)
    finally:
        BUNDLE._VALIDATION_HOOK = None


@pytest.mark.parametrize("variant", ("relative_served", "parent_driver", "served_sha", "extra_arg"))
def test_launch_command_rejects_nonexact_following_arguments_and_bindings(tmp_path: Path, variant: str) -> None:
    value = json.loads((ARTIFACT / "launch-command.json").read_text())
    if variant == "relative_served":
        value["resident_driver_argv"][2] = "active.json"
    elif variant == "parent_driver":
        value["resident_driver_argv"][0] = str(BUNDLE.CANONICAL_ROOT / "subdir/../resident-driver")
    elif variant == "served_sha":
        value["bindings"]["served_model_manifest"]["sha256"] = "0" * 64
    else:
        value["resident_driver_argv"].append("--unexpected")
    with pytest.raises(BUNDLE.BundleError, match="launch command"):
        BUNDLE.validate_launch_command(value)


def test_launch_path_rejects_parent_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "driver").write_bytes(b"driver")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(BUNDLE.BundleError, match="symlink component"):
        BUNDLE.reject_symlink_components(alias / "driver", "launch driver")


def test_runner_generated_plan_and_subprocess_evidence_are_bound() -> None:
    plan_raw = (ARTIFACT / "dry-run.json").read_bytes()
    plan = json.loads(plan_raw)
    evidence = json.loads((ARTIFACT / "runner-dry-run-evidence.json").read_text())
    assert (plan["case_count"], plan["transaction_count"], plan["warmup_runs"], plan["measured_runs"]) == (1, 12, 2, 10)
    assert plan["execution_mode"] == "one_case_smoke"
    assert plan["smoke_only"] is True
    assert plan["promotion_eligible"] is False
    assert plan["validation"]["mode"] == "validate_only"
    assert plan["validation"]["driver_fake_handshake"] == "passed"
    assert evidence["runner_subprocess_count"] == 1
    assert evidence["exit_code"] == 0
    assert evidence["stdout"] == {"sha256": hashlib.sha256(b"").hexdigest(), "utf8": ""}
    assert evidence["stderr"] == {"sha256": hashlib.sha256(b"").hexdigest(), "utf8": ""}
    assert evidence["plan"]["sha256"] == hashlib.sha256(plan_raw).hexdigest()
    assert evidence["normal_profile"] == {"case_count": 84, "separate": True}


@pytest.mark.parametrize("variant", ("smoke_only", "transaction_count", "normal_profile"))
def test_rejects_rebound_runner_plan_or_normal_profile(tmp_path: Path, trusted_reconstruction, variant: str) -> None:
    root = copy_bundle(tmp_path)
    if variant == "normal_profile":
        path = root / "runner-dry-run-evidence.json"
        value = json.loads(path.read_text())
        value["normal_profile"]["case_count"] = 1
    else:
        path = root / "dry-run.json"
        value = json.loads(path.read_text())
        value[variant] = False if variant == "smoke_only" else 84
    rewrite_json(path, value)
    rebind_transport(root)
    with pytest.raises(BUNDLE.BundleError, match="trusted runner"):
        BUNDLE.validate(root, trusted_reconstruction)


def test_checked_in_v4_binding_sidecar_passes_and_pins_final_runner_validator() -> None:
    value = BUNDLE.validate_binding(VALIDATOR_COMMIT, VALIDATOR_SHA, BINDING)
    assert value["status"] == "prepared_not_executed"
    assert value["promotion"] is False
    assert value["launch_eligible"] is False
    assert value["requires_immutable_launcher"] is True
    assert value["predecessor"] == {"commit": "791a20c", "status": "SUPERSEDED", "execution_eligible": False}
    roots = value["trust_roots"]
    assert roots["source_commit"] == "e93a2c162eb059cb2db883953d331f7a158d3a16"
    assert roots["source_tree"] == "65ac07cac2f025a6cde6eae153435e6ff86e679d"
    assert roots["runner"] == {"git_blob": "a4b0ea2bbae805441715a40b9c4ded406ede4414", "sha256": "0d68f7141ea531e2200251597d601f9060b21b723faae2c8f96ae586c8cbeccc"}
    assert roots["validator"]["source_commit"] == VALIDATOR_COMMIT
    assert roots["validator"]["sha256"] == VALIDATOR_SHA
    assert roots["resident_driver"]["blob_unchanged"] is True
    assert roots["resident_driver"]["binary_sha256"] == BUNDLE.EXPECTED_DRIVER_SHA


def test_v4_binding_records_actual_runner_and_mandatory_validator_subprocesses() -> None:
    plan_raw = (BINDING / "runner-plan.json").read_bytes()
    plan = json.loads(plan_raw)
    evidence = json.loads((BINDING / "runner-subprocess-evidence.json").read_text())
    report_raw = (BINDING / "validator-report.json").read_bytes()
    validator = plan["validation"]["trusted_bundle_validator"]
    assert (plan["case_count"], plan["transaction_count"], plan["warmup_runs"], plan["measured_runs"]) == (1, 12, 2, 10)
    assert plan["smoke_only"] is True and plan["promotion_eligible"] is False
    assert plan["validation"]["root_contract"] == "ullm.aq4_p2_resident_smoke_bundle_root.v4"
    assert set(plan["validation"]["members"]) == set(BUNDLE.REQUIRED_FILES) | {"bundle.json", "SHA256SUMS"}
    assert plan["validation"]["fake_driver_subprocess_count"] == 1
    assert plan["validation"]["resident_driver_argv"] == BUNDLE.resident_driver_argv()
    assert validator["subprocess_count"] == 1
    assert validator["source"] == {"path": str(BUNDLE.BINDING_VALIDATOR_EXEC), "sha256": VALIDATOR_SHA}
    assert validator["report_sha256"] == hashlib.sha256(BUNDLE.canonical(validator["report"])).hexdigest()
    assert evidence["runner_subprocess_count"] == 1
    assert evidence["exit_code"] == 0
    assert evidence["stdout"] == {"sha256": hashlib.sha256(b"").hexdigest(), "utf8": ""}
    assert evidence["stderr"] == {"sha256": hashlib.sha256(b"").hexdigest(), "utf8": ""}
    assert evidence["plan"]["sha256"] == hashlib.sha256(plan_raw).hexdigest()
    assert evidence["trusted_validator"]["report_file_sha256"] == hashlib.sha256(report_raw).hexdigest()


def test_v4_binding_keeps_generic_runner_outputs_outside_immutable_input_root() -> None:
    manifest = json.loads((BINDING / "binding-manifest.json").read_text())
    assert manifest["cycle_control"] == {
        "input_root_unchanged_after_runner": True,
        "generated_outputs_outside_input_root": True,
        "input_root_dry_run_not_replaced": True,
        "generic_runner_schema_not_embedded_back_into_input_root": True,
    }
    assert manifest["input_root"]["members"]["dry-run.json"]["sha256"] == hashlib.sha256((ARTIFACT / "dry-run.json").read_bytes()).hexdigest()
    assert manifest["outputs"]["runner_plan_sha256"] == hashlib.sha256((BINDING / "runner-plan.json").read_bytes()).hexdigest()
    assert manifest["next_stage"]["name"] == "L immutable launcher"
    assert manifest["next_stage"]["required"] is True


def test_v4_binding_rejects_report_replacement(tmp_path: Path) -> None:
    root = tmp_path / "binding"
    shutil.copytree(BINDING, root)
    report = root / "validator-report.json"
    report.chmod(0o644)
    value = json.loads(report.read_text())
    value["promotion"] = True
    rewrite_json(report, value)
    with pytest.raises(BUNDLE.BundleError, match="validator report differs"):
        BUNDLE.validate_binding(VALIDATOR_COMMIT, VALIDATOR_SHA, root)


@pytest.mark.parametrize("variant", ("late_unknown", "late_missing", "late_replace"))
def test_v4_binding_final_reenumeration_rejects_late_mutation(tmp_path: Path, variant: str) -> None:
    root = tmp_path / "binding"
    shutil.copytree(BINDING, root)
    replacement = tmp_path / "replacement-report.json"
    shutil.copyfile(root / "validator-report.json", replacement)
    replacement.chmod(0o444)

    def mutate(binding_root: Path) -> None:
        if variant == "late_unknown":
            (binding_root / "late-unknown.json").write_text("{}\n", encoding="utf-8")
        elif variant == "late_missing":
            (binding_root / "validator-report.json").unlink()
        else:
            (binding_root / "validator-report.json").unlink()
            shutil.copyfile(replacement, binding_root / "validator-report.json")
            (binding_root / "validator-report.json").chmod(0o444)

    BUNDLE._BINDING_VALIDATION_HOOK = mutate
    try:
        with pytest.raises(BUNDLE.BundleError, match="late binding sidecar"):
            BUNDLE.validate_binding(VALIDATOR_COMMIT, VALIDATOR_SHA, root)
    finally:
        BUNDLE._BINDING_VALIDATION_HOOK = None
