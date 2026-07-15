from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/prepare-qwen35-aq4-sq8-overlay-gpu-promotion.py"
SPEC = importlib.util.spec_from_file_location("prepare_sq8_gpu_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readiness() -> dict[str, object]:
    body = '{"status":"ready"}'
    return {
        "schema": "ullm.bridge_container_readiness.v1",
        "container": {
            "name": "open-webui",
            "id": "4" * 64,
            "image_id": "sha256:" + "5" * 64,
            "config_image": "ullm/open-webui:test",
        },
        "network": {
            "name": "open-webui-network",
            "id": "6" * 64,
            "driver": "bridge",
            "bridge_interface": "br-" + "6" * 12,
        },
        "endpoint": {
            "url": "http://172.20.0.1:8000/readyz",
            "path": "/readyz",
            "expected_status": 200,
            "expected_body": body,
            "expected_body_sha256": hashlib.sha256(body.encode("ascii")).hexdigest(),
            "timeout_seconds": 5,
        },
    }


def fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    product = tmp_path / "product"
    artifact = product / "artifact"
    package = product / "package"
    artifact.mkdir(parents=True)
    package.mkdir()
    package_manifest = package / "manifest.json"
    package_manifest.write_text('{"schema_version":"fixture"}\n', encoding="ascii")
    names = [
        f"model.language_model.layers.{index}.linear_attn.{suffix}.weight"
        for index in range(24)
        for suffix in ("in_proj_qkv", "in_proj_z")
    ]
    binding = {
        "schema_version": "ullm.qwen35_aq4_sq8_qkv_z_overlay.v2",
        "format_id": "AQ4_0",
        "overlay_format_id": "SQ8_0",
        "implementation_id": TOOL.IMPLEMENTATION_ID,
        "content_sha256": "b" * 64,
        "tensor_set_sha256": "c" * 64,
        "tensor_names": names,
        "package": {"manifest_sha256": sha(package_manifest)},
    }
    (artifact / "binding.json").write_text(json.dumps(binding) + "\n", encoding="ascii")
    profile = {
        "schema_version": "ullm.served_model.profile.v1",
        "public": {"id": "fixture-overlay", "revision": "fixture-v1"},
        "format": {"format_id": "AQ4_0", "implementation_id": TOOL.IMPLEMENTATION_ID},
        "worker": {
            "binary": "unused",
            "required_environment": list(TOOL.REQUIRED_OVERLAY_ENV),
            "identity": {
                "device": "gfx1201",
                "execution_profile": TOOL.EXECUTION_PROFILE,
            },
        },
        "product": {
            "root": str(product),
            "artifact": {"manifest_path": "artifact/binding.json"},
            "package": {"manifest_path": "package/manifest.json"},
        },
        "promotion": {
            "receipt": "unused",
            "required_schema_version": "unused",
            "evidence_from_receipt": ["evidence"],
            "evidence_sha256_from_receipt": ["sha"],
        },
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile) + "\n", encoding="ascii")
    worker = tmp_path / "worker"
    worker.write_bytes(b"immutable worker fixture\n")
    worker.chmod(0o755)
    return profile_path, worker, sha(artifact / "binding.json")


class FakeGenerator:
    @staticmethod
    def generate(profile_path: Path, output_path: Path) -> None:
        profile = json.loads(profile_path.read_text())
        product = profile["product"]
        root = Path(product["root"])
        artifact = root / product["artifact"]["manifest_path"]
        package = root / product["package"]["manifest_path"]
        binding = json.loads(artifact.read_text())
        value = {
            "public": profile["public"],
            "format": profile["format"],
            "worker": {
                "binary": profile["worker"]["binary"],
                "identity": profile["worker"]["identity"],
            },
            "product": {
                "root": str(root),
                "artifact": {
                    "manifest_sha256": sha(artifact),
                    "content_sha256": binding["content_sha256"],
                },
                "package": {"manifest_sha256": sha(package)},
            },
        }
        TOOL.write_json_exclusive(output_path, value)

    generate_prepared_candidate = generate


class FakeReceiptWriter:
    @staticmethod
    def write_receipt(**kwargs: object) -> dict[str, object]:
        output = Path(str(kwargs["output_path"]))
        value: dict[str, object] = {
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
            "status": "prepared_not_executed",
            "actual": {"status": "pending", "required": True},
        }
        TOOL.write_json_exclusive(output, value)
        return value


def test_builder_materializes_create_new_immutable_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile, worker, binding_sha = fixture(tmp_path)
    commit = "a" * 40
    monkeypatch.setattr(
        TOOL,
        "git_value",
        lambda *args: (
            commit
            if args[0] == "rev-parse" and args[1].endswith("^{commit}")
            else "d" * 40
        ),
    )
    monkeypatch.setattr(TOOL, "source_archive_sha256", lambda _: "e" * 64)
    monkeypatch.setattr(TOOL, "readiness_identity", readiness)
    monkeypatch.setattr(TOOL, "command_text", lambda argv, **_: "fixture-version")
    monkeypatch.setattr(
        TOOL,
        "load_module",
        lambda _name, path: (
            FakeReceiptWriter if path == TOOL.RECEIPT_WRITER else FakeGenerator
        ),
    )
    output = tmp_path / "gate-output"
    args = argparse.Namespace(
        release_source_commit=commit,
        output=output,
        profile=profile,
        worker_binary=worker,
    )

    result = TOOL.materialize(args)

    assert result["actual_run_allowed"] is False
    assert result["gate_sha256"] == sha(output / "gate.json")
    assert (output.stat().st_mode & 0o777) == 0o555
    copied = output / "ullm-aq4-worker"
    assert copied.read_bytes() == worker.read_bytes()
    assert copied.stat().st_nlink == 1
    assert (copied.stat().st_mode & 0o777) == 0o555
    gate = json.loads((output / "gate.json").read_text())
    build = json.loads((output / "build-receipt.json").read_text())
    expected_worker = str(copied.resolve())
    assert build["worker"]["source_path"] == expected_worker
    assert build["worker"]["immutable_path"] == expected_worker
    assert build["worker"]["source_sha256"] == build["worker"]["immutable_sha256"]
    assert build["worker"]["source_mode"] == build["worker"]["immutable_mode"] == "0555"
    assert build["worker"]["source_nlink"] == build["worker"]["immutable_nlink"] == 1
    profile_value = json.loads((output / "profile.json").read_text())
    assert profile_value["promotion"] == {
        "receipt": str(output / "promotion-receipt.json"),
        "source_commit_from_receipt": ["source_commit"],
        "required_schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
        "overlay_from_receipt": ["overlay"],
        "release_from_receipt": ["release"],
        "package_from_receipt": ["package"],
        "actual_evidence_from_receipt": ["actual"],
        "request_id_from_receipt": ["request_id"],
        "authorization_audit_from_receipt": ["authorization_audit"],
        "authorization_lineage_from_receipt": ["authorization_lineage"],
        "readiness_from_receipt": ["readiness"],
        "authorization_lineage": None,
        "readiness": readiness(),
        "release_source_commit": commit,
    }
    request_id = gate["request"]["actual"]["request_id"]
    assert request_id.startswith("sq8-promotion-") and len(request_id) == 78
    assert gate["request"]["actual"]["telemetry_environment"] == {
        "ULLM_SQ8_PROMOTION_EVIDENCE_REQUEST_ID": request_id
    }
    assert gate["release_source_commit"] == commit
    assert gate["profile_identity"]["artifact_binding_sha256"] == binding_sha
    assert gate["actual_evidence_requirements"]["projection_counts"] == {
        "batch_matvec_count": ">0",
        "pair_matvec_count": ">0",
        "single_matvec_count": 0,
        "triple_matvec_count": 0,
        "fallback_count": 0,
    }
    assert gate["classification"] == {
        "promotion": "unclassified",
        "fidelity": "unclassified",
        "holdout_used": False,
        "policy_relaxed": False,
    }
    for line in (output / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert sha(output / name) == expected

    with pytest.raises(TOOL.GateError, match="refusing to reuse"):
        TOOL.materialize(args)


def test_profile_and_binding_preflight_fail_closed(tmp_path: Path) -> None:
    profile_path, _, _ = fixture(tmp_path)
    profile = json.loads(profile_path.read_text())
    profile["worker"]["required_environment"].remove(
        "ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL"
    )
    with pytest.raises(TOOL.GateError, match="required environment"):
        TOOL.validate_profile(profile)

    binding_path = Path(profile["product"]["root"]) / "artifact/binding.json"
    binding = json.loads(binding_path.read_text())
    binding["tensor_names"].pop()
    package = Path(profile["product"]["root"]) / "package/manifest.json"
    with pytest.raises(TOOL.GateError, match="exactly 48"):
        TOOL.validate_binding(binding, package)


def test_authorization_requires_paired_flags_and_exact_audited_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile, worker, binding_sha = fixture(tmp_path)
    unauthorized_runtime = tmp_path / "unauthorized-runtime"
    unauthorized_runtime.mkdir()
    audited_worker = unauthorized_runtime / "ullm-aq4-worker"
    worker.rename(audited_worker)
    worker = audited_worker
    commit = "a" * 40
    tree = "d" * 40
    archive = "e" * 64
    monkeypatch.setattr(
        TOOL,
        "git_value",
        lambda *args: (
            commit if args[0] == "rev-parse" and args[1].endswith("^{commit}") else tree
        ),
    )
    monkeypatch.setattr(TOOL, "source_archive_sha256", lambda _: archive)
    monkeypatch.setattr(TOOL, "readiness_identity", readiness)
    monkeypatch.setattr(TOOL, "command_text", lambda argv, **_: "fixture-version")
    monkeypatch.setattr(
        TOOL,
        "load_module",
        lambda _name, path: (
            FakeReceiptWriter if path == TOOL.RECEIPT_WRITER else FakeGenerator
        ),
    )
    audit_sha = "9" * 64
    audit_path = tmp_path / "audit-receipt.json"
    audit_path.write_text("{}\n", encoding="ascii")
    audit_path.chmod(0o444)
    lineage_path = (tmp_path / "lineage-input-manifest.json").resolve()
    lineage_path.write_text("{}\n", encoding="ascii")
    lineage_path.chmod(0o444)
    lineage_raw = lineage_path.read_bytes()
    lineage_validated = {
        "path": str(lineage_path),
        "sha256": hashlib.sha256(lineage_raw).hexdigest(),
        "entries_sha256": "8" * 64,
        "raw": lineage_raw,
    }
    monkeypatch.setattr(
        TOOL.lineage_tool,
        "validate_manifest",
        lambda *args, **kwargs: lineage_validated,
    )
    monkeypatch.setattr(
        TOOL.lineage_tool, "validate_reference", lambda *args, **kwargs: args[0]
    )
    expected_output = Path(
        f"/tmp/ullm-sq8-overlay-gpu-promotion-gate-authorized-{audit_sha[:16]}"
    )
    if expected_output.exists():
        import shutil

        expected_output.chmod(0o755)
        shutil.rmtree(expected_output)
    worker_sha = sha(worker)
    package_sha = sha(tmp_path / "product/package/manifest.json")
    expected_request = TOOL.fixed_promotion_request_id(
        commit=commit,
        tree=tree,
        archive_sha256=archive,
        worker_sha256=worker_sha,
        binding_sha256=binding_sha,
        content_sha256="b" * 64,
        tensor_set_sha256="c" * 64,
        package_sha256=package_sha,
        readiness=readiness(),
        authorization_lineage=None,
        authorization_lineage_manifest={
            "schema_version": TOOL.lineage_tool.REFERENCE_SCHEMA,
            "input_path": str(lineage_path),
            "sha256": lineage_validated["sha256"],
            "entries_sha256": lineage_validated["entries_sha256"],
        },
    )
    audit = {
        "path": str(audit_path.resolve()),
        "sha256": audit_sha,
        "request_id": expected_request,
        "worker_sha256": worker_sha,
        "binding_sha256": binding_sha,
        "package_sha256": package_sha,
        "runtime": str(unauthorized_runtime),
    }
    monkeypatch.setattr(
        TOOL, "validate_independent_audit", lambda *args, **kwargs: audit
    )

    common = dict(
        release_source_commit=commit,
        output=expected_output,
        profile=profile,
        worker_binary=worker,
        authorization_lineage_manifest=lineage_path,
    )
    with pytest.raises(TOOL.GateError, match="required together"):
        TOOL.materialize(
            argparse.Namespace(
                **common, authorize_actual_run=True, independent_audit_receipt=None
            )
        )
    with pytest.raises(TOOL.GateError, match="required together"):
        TOOL.materialize(
            argparse.Namespace(
                **common,
                authorize_actual_run=False,
                independent_audit_receipt=audit_path,
            )
        )
    legacy = dict(common)
    legacy.pop("authorization_lineage_manifest")
    with pytest.raises(TOOL.GateError, match="lineage manifest"):
        TOOL.materialize(
            argparse.Namespace(
                **legacy,
                authorize_actual_run=True,
                independent_audit_receipt=audit_path,
            )
        )

    bad_audit = dict(audit)
    bad_audit["worker_sha256"] = "0" * 64
    monkeypatch.setattr(
        TOOL, "validate_independent_audit", lambda *args, **kwargs: bad_audit
    )
    with pytest.raises(TOOL.GateError, match="differs from independently audited"):
        TOOL.materialize(
            argparse.Namespace(
                **common,
                authorize_actual_run=True,
                independent_audit_receipt=audit_path,
            )
        )
    monkeypatch.setattr(
        TOOL, "validate_independent_audit", lambda *args, **kwargs: audit
    )

    result = TOOL.materialize(
        argparse.Namespace(
            **common, authorize_actual_run=True, independent_audit_receipt=audit_path
        )
    )
    gate = json.loads((expected_output / "gate.json").read_text())
    build = json.loads((expected_output / "build-receipt.json").read_text())
    expected_worker = str((expected_output / "ullm-aq4-worker").resolve())
    assert result["actual_run_allowed"] is True
    assert gate["status"] == "authorized_pending_execution"
    assert gate["actual_run_allowed"] is True
    assert gate["authorization"]["max_attempts"] == 1
    assert gate["authorization"]["independent_audit_receipt"] == {
        "path": str(audit_path.resolve()),
        "sha256": audit_sha,
    }
    assert build["worker"]["source_path"] == expected_worker
    assert build["worker"]["immutable_path"] == expected_worker
    assert build["worker"]["source_sha256"] == build["worker"]["immutable_sha256"]
    assert build["worker"]["source_mode"] == build["worker"]["immutable_mode"] == "0555"
    assert build["worker"]["source_nlink"] == build["worker"]["immutable_nlink"] == 1
    assert build["inputs"]["independent_audit_receipt"] == {
        "path": str(audit_path.resolve()),
        "sha256": audit_sha,
    }
    forbidden = str(unauthorized_runtime.resolve()).encode("utf-8")
    assert {entry.name for entry in expected_output.iterdir()} == TOOL.RUNTIME_MEMBERS
    assert all(
        forbidden not in entry.read_bytes() for entry in expected_output.iterdir()
    )
    with pytest.raises(TOOL.GateError, match="refusing to reuse"):
        TOOL.materialize(
            argparse.Namespace(
                **common,
                authorize_actual_run=True,
                independent_audit_receipt=audit_path,
            )
        )
    import shutil

    expected_output.chmod(0o755)
    shutil.rmtree(expected_output)


@pytest.mark.parametrize(
    "injected",
    [
        lambda old: str(old / "ullm-aq4-worker"),
        lambda old: {"nested": [{"source_path": str(old / "ullm-aq4-worker")}]},
        lambda old: str(
            old.parent / "path-alias" / ".." / old.name / "ullm-aq4-worker"
        ),
        lambda old: "file://" + str(old / "ullm-aq4-worker"),
    ],
)
def test_authorized_runtime_recursive_scan_rejects_old_path_and_aliases(
    tmp_path: Path, injected: object
) -> None:
    runtime = tmp_path / "authorized"
    old = tmp_path / "unauthorized"
    runtime.mkdir()
    old.mkdir()
    for name in TOOL.RUNTIME_MEMBERS:
        path = runtime / name
        if name.endswith(".json"):
            path.write_text("{}\n", encoding="ascii")
        else:
            path.write_bytes(b"fixture\n")
    value = {"self": str(runtime / "ullm-aq4-worker"), "injected": injected(old)}
    (runtime / "gate.json").write_text(json.dumps(value) + "\n", encoding="ascii")
    with pytest.raises(TOOL.GateError, match="audited runtime path"):
        TOOL.reject_runtime_references(runtime, old)
    (runtime / "gate.json").write_text(
        json.dumps({"self": str(runtime / "ullm-aq4-worker")}) + "\n",
        encoding="ascii",
    )
    TOOL.reject_runtime_references(runtime, old)


def test_authorized_output_path_is_independent_of_request_derivation() -> None:
    audit_sha = "7" * 64
    first = TOOL.authorized_output_path(audit_sha)
    request_a = TOOL.fixed_promotion_request_id(
        commit="a" * 40,
        tree="b" * 40,
        archive_sha256="c" * 64,
        worker_sha256="d" * 64,
        binding_sha256="e" * 64,
        content_sha256="f" * 64,
        tensor_set_sha256="1" * 64,
        package_sha256="2" * 64,
        readiness={"version": 1},
        authorization_lineage=None,
    )
    request_b = TOOL.fixed_promotion_request_id(
        commit="a" * 40,
        tree="b" * 40,
        archive_sha256="c" * 64,
        worker_sha256="d" * 64,
        binding_sha256="e" * 64,
        content_sha256="f" * 64,
        tensor_set_sha256="1" * 64,
        package_sha256="2" * 64,
        readiness={"version": 2},
        authorization_lineage=None,
    )
    assert request_a != request_b
    assert TOOL.authorized_output_path(audit_sha) == first


def test_worker_copy_rejects_hardlink_and_detects_live_identity(tmp_path: Path) -> None:
    source = tmp_path / "worker"
    source.write_bytes(b"worker\n")
    source.chmod(0o755)
    os.link(source, tmp_path / "worker-link")
    with pytest.raises(TOOL.GateError, match="single-link"):
        TOOL.copy_binary_exclusive(source, tmp_path / "copied")


def test_audit_receipt_rejects_writable_and_symlink(tmp_path: Path) -> None:
    writable = tmp_path / "audit.json"
    writable.write_text("{}\n", encoding="ascii")
    writable.chmod(0o644)
    with pytest.raises(TOOL.GateError, match="immutable 0444"):
        TOOL.validate_independent_audit(
            writable,
            commit="a" * 40,
            tree="b" * 40,
            archive_sha256="c" * 64,
            authorization_lineage_manifest={},
        )
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="ascii")
    target.chmod(0o444)
    link = tmp_path / "audit-link.json"
    link.symlink_to(target)
    with pytest.raises(TOOL.GateError, match="immutable 0444"):
        TOOL.validate_independent_audit(
            link,
            commit="a" * 40,
            tree="b" * 40,
            archive_sha256="c" * 64,
            authorization_lineage_manifest={},
        )


def test_prior_failure_lineage_binds_consumed_receipt_and_rejects_weak_files(
    tmp_path: Path,
) -> None:
    request_id = "sq8-promotion-" + "7" * 64
    path = tmp_path / "promotion-failure-receipt.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
                "status": "actual_failed",
                "request_id": request_id,
                "actual": {"status": "failed", "request_id": request_id},
            }
        )
        + "\n",
        encoding="ascii",
    )
    path.chmod(0o444)
    no_go_path = tmp_path / "no-go-audit.json"
    no_go_path.write_text(
        json.dumps(
            {
                "schema_version": TOOL.AUDIT_SCHEMA,
                "verdict": "implementation_no_go",
                "actual": "not_executed",
                "reason_code": "restore_retry_terminal_identity_not_fail_closed",
                "audited_source": {"commit": "8" * 40},
                "runtime": {"gate": {"sha256": "9" * 64}},
            }
        )
        + "\n",
        encoding="ascii",
    )
    no_go_path.chmod(0o444)

    lineage = TOOL.prior_failure_lineage(path, no_go_path)

    assert lineage == {
        "schema": TOOL.AUTHORIZATION_LINEAGE_SCHEMA,
        "disposition": "consumed_failed_not_reusable",
        "prior_request_id": request_id,
        "prior_failure_receipt": {"path": str(path.resolve()), "sha256": sha(path)},
        "prior_no_go_audit": {
            "path": str(no_go_path.resolve()),
            "sha256": sha(no_go_path),
            "verdict": "implementation_no_go",
            "reason_code": "restore_retry_terminal_identity_not_fail_closed",
            "audited_source_commit": "8" * 40,
            "audited_gate_sha256": "9" * 64,
        },
    }
    path.chmod(0o644)
    with pytest.raises(TOOL.GateError, match="immutable"):
        TOOL.prior_failure_lineage(path)
    path.chmod(0o444)
    link = tmp_path / "failure-link.json"
    link.symlink_to(path)
    with pytest.raises(TOOL.GateError, match="immutable"):
        TOOL.prior_failure_lineage(link)
