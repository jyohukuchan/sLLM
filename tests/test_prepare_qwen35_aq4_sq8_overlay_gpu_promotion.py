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
            "identity": {"device": "gfx1201", "execution_profile": TOOL.EXECUTION_PROFILE},
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


def test_builder_materializes_create_new_immutable_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    profile, worker, binding_sha = fixture(tmp_path)
    commit = "a" * 40
    monkeypatch.setattr(
        TOOL,
        "git_value",
        lambda *args: commit if args[0] == "rev-parse" and args[1].endswith("^{commit}") else "d" * 40,
    )
    monkeypatch.setattr(TOOL, "source_archive_sha256", lambda _: "e" * 64)
    monkeypatch.setattr(TOOL, "command_text", lambda argv, **_: "fixture-version")
    monkeypatch.setattr(
        TOOL,
        "load_module",
        lambda _name, path: FakeReceiptWriter if path == TOOL.RECEIPT_WRITER else FakeGenerator,
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
