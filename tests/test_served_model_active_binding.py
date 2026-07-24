from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/served_model_active_binding.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BINDING = load_module("test_served_model_active_binding_tool", TOOL)


def manifest_bytes(*, whitespace: bool = False) -> bytes:
    value = {
        "schema_version": "ullm.served_model.v2",
        "public": {"id": "ullm-qwen3-14b-sq8"},
        "worker": {"protocol": "ullm.worker.v2"},
    }
    if whitespace:
        return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    return (
        json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")


def make_pair(tmp_path: Path) -> tuple[Path, Path, bytes]:
    raw = manifest_bytes()
    candidate = tmp_path / "candidate.json"
    active = tmp_path / "active.json"
    candidate.write_bytes(raw)
    active.write_bytes(raw)
    return candidate, active, raw


def test_observer_binds_exact_candidate_and_ordered_active_bytes(
    tmp_path: Path,
) -> None:
    candidate, active, raw = make_pair(tmp_path)
    wall = iter((10, 20))
    monotonic = iter((100, 200))
    observer = BINDING.ActiveManifestBinding(
        candidate_path=candidate,
        active_path=active,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_stages=("preflight", "final"),
        wall_clock_ns=lambda: next(wall),
        monotonic_clock_ns=lambda: next(monotonic),
    )

    first = observer.observe("preflight")
    second = observer.observe("final")
    artifacts = observer.artifacts()

    assert first["sequence"] == 0
    assert second["sequence"] == 1
    assert first["bytes_equal"] is True
    assert artifacts.candidate_manifest == raw
    rows = [
        json.loads(line)
        for line in artifacts.observations_jsonl.decode("ascii").splitlines()
    ]
    assert [row["stage"] for row in rows] == ["preflight", "final"]
    assert [row["observed_unix_ns"] for row in rows] == [10, 20]
    summary = json.loads(artifacts.binding_json)
    assert summary["status"] == "complete"
    assert summary["candidate"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert summary["observations"]["sha256"] == hashlib.sha256(
        artifacts.observations_jsonl
    ).hexdigest()
    assert set(artifacts.by_name()) == {
        BINDING.CANDIDATE_ARTIFACT_NAME,
        BINDING.OBSERVATIONS_ARTIFACT_NAME,
        BINDING.BINDING_ARTIFACT_NAME,
    }


def test_semantically_equal_but_byte_different_active_manifest_is_rejected(
    tmp_path: Path,
) -> None:
    candidate, active, raw = make_pair(tmp_path)
    active.write_bytes(manifest_bytes(whitespace=True))
    observer = BINDING.ActiveManifestBinding(
        candidate_path=candidate,
        active_path=active,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_stages=("preflight",),
    )

    with pytest.raises(
        BINDING.ActiveBindingError,
        match="bytes differ from candidate",
    ):
        observer.observe("preflight")


def test_candidate_change_after_pin_is_rejected(tmp_path: Path) -> None:
    candidate, active, raw = make_pair(tmp_path)
    observer = BINDING.ActiveManifestBinding(
        candidate_path=candidate,
        active_path=active,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_stages=("preflight",),
    )
    candidate.write_bytes(manifest_bytes(whitespace=True))

    with pytest.raises(BINDING.ActiveBindingError, match="changed after it was pinned"):
        observer.observe("preflight")


def test_path_entry_replacement_during_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    original = b"a" * (BINDING.READ_CHUNK_BYTES + 1)
    replacement = tmp_path / "replacement.json"
    path.write_bytes(original)
    replacement.write_bytes(original)
    real_read = BINDING.os.read
    replaced = False

    def read_then_replace(descriptor: int, amount: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, amount)
        if chunk and not replaced:
            os.replace(replacement, path)
            replaced = True
        return chunk

    monkeypatch.setattr(BINDING.os, "read", read_then_replace)
    with pytest.raises(BINDING.ActiveBindingError, match="changed while it was read"):
        BINDING.stable_read_regular(
            path,
            "manifest",
            maximum=len(original) + 1,
        )


def test_symlink_world_writable_and_noncanonical_paths_are_rejected(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}\n")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(BINDING.ActiveBindingError):
        BINDING.stable_read_regular(link, "manifest", maximum=1024)

    target.chmod(0o666)
    with pytest.raises(BINDING.ActiveBindingError, match="unsafe"):
        BINDING.stable_read_regular(target, "manifest", maximum=1024)

    noncanonical = tmp_path / "child" / ".." / "target.json"
    with pytest.raises(BINDING.ActiveBindingError, match="lexically canonical"):
        BINDING.stable_read_regular(noncanonical, "manifest", maximum=1024)


def test_claim_reference_requires_authorized_campaign_and_candidate(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
    candidate_value = {
        "schema_version": "ullm.served_model.v2",
        "public": {"id": "ullm-qwen3-14b-sq8"},
        "format": {"format_id": "SQ8_0"},
        "worker": {
            "protocol": "ullm.worker.v2",
            "binary_sha256": "4" * 64,
        },
        "promotion": {
            "source_commit": "a" * 40,
            "receipt_sha256": "5" * 64,
        },
    }
    raw = (
        json.dumps(candidate_value, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
    )
    candidate = tmp_path / "candidate.json"
    active = tmp_path / "active.json"
    candidate.write_bytes(raw)
    active.write_bytes(raw)
    outputs = tmp_path / "outputs"
    claims = tmp_path / "claims"
    outcomes = tmp_path / "outcomes"
    outputs.mkdir(mode=0o700)
    claims.mkdir(mode=0o700)
    outcomes.mkdir(mode=0o700)
    aq4_source = tmp_path / "aq4-source"
    aq4_source.mkdir(mode=0o700)
    aq4_worker = tmp_path / "aq4-worker"
    aq4_worker.write_bytes(b"fixture worker\n")
    bundle_root = outputs / "aq4-bundle"
    bundle_root.mkdir(mode=0o700)
    promotion_evidence = tmp_path / "promotion-evidence.json"
    promotion_receipt = tmp_path / "promotion-receipt.json"
    promotion_evidence.write_bytes(b'{"fixture":"evidence"}\n')
    promotion_receipt.write_bytes(b'{"fixture":"receipt"}\n')
    policy = BINDING.campaign_authorization.RegistryPolicy(
        claim_registry=claims,
        outcome_registry=outcomes,
        required_uid=os.geteuid(),
    )
    final = outputs / "sq8-full"
    authorization = tmp_path / "authorization.json"
    authorization_document = {
        "schema_version": BINDING.campaign_authorization.AUTHORIZATION_SCHEMA,
        "authorization_id": "sq8-v2-window-20260724-001",
        "issued_at": BINDING.campaign_authorization.utc_timestamp(
            now - timedelta(minutes=1)
        ),
        "expires_at": BINDING.campaign_authorization.utc_timestamp(
            now + timedelta(hours=1)
        ),
        "max_attempts": 1,
        "authorization_note": "Reviewed candidate-only evidence window.",
        "purpose": "temporary_candidate_active_evidence_collection_only",
        "required_final_route": "restore_exact_aq4_then_bundle_v2_activation",
        "source": {"commit": "a" * 40, "tree": "b" * 40},
        "before": {
            "model_id": "ullm-qwen3.5-9b-aq4",
            "format_id": "AQ4_0",
            "manifest_sha256": "1" * 64,
            "worker_protocol": "ullm.worker.v2",
            "worker_binary_path": str(aq4_worker),
            "worker_binary_sha256": "2" * 64,
            "promotion_source_commit": "c" * 40,
            "promotion_receipt_path": str(promotion_receipt),
            "promotion_receipt_sha256": "8" * 64,
        },
        "aq4_release": {
            "source": {
                "root": str(aq4_source),
                "commit": "c" * 40,
                "tree": "d" * 40,
            },
            "openwebui_image": (
                BINDING.campaign_authorization.FIXED_OPENWEBUI_IMAGE
            ),
            "promotion_evidence": {
                "source_path": str(promotion_evidence),
                "path": str(bundle_root / promotion_evidence.name),
                "sha256": "a" * 64,
            },
            "promotion_receipt": {
                "source_path": str(promotion_receipt),
                "path": str(bundle_root / promotion_receipt.name),
                "sha256": "8" * 64,
            },
            "release_evidence_path": str(bundle_root / "release-evidence.json"),
            "release_validator_path": str(bundle_root / "release-validator.json"),
            "browser_validator_path": str(bundle_root / "browser-validator.json"),
        },
        "candidate": {
            "model_id": "ullm-qwen3-14b-sq8",
            "format_id": "SQ8_0",
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "worker_protocol": "ullm.worker.v2",
            "worker_binary_sha256": "4" * 64,
            "promotion_source_commit": "a" * 40,
            "promotion_receipt_sha256": "5" * 64,
        },
        "campaigns": {
            "aq4_reasoning_release": {
                "run_id": "aq4-reasoning-run",
                "final_path": str(outputs / "aq4-reasoning"),
            },
            "aq4_reasoning_browser": {
                "run_id": "aq4-browser-run",
                "final_path": str(bundle_root / "browser-evidence.json"),
            },
            "aq4_bundle": {
                "run_id": "aq4-bundle-run",
                "final_path": str(bundle_root / "bundle.json"),
            },
            "sq8_full": {"run_id": "sq8-run", "final_path": str(final)},
            "reasoning_release": {
                "run_id": "reasoning-run",
                "final_path": str(outputs / "reasoning"),
            },
            "reasoning_browser": {
                "run_id": "browser-run",
                "final_path": str(outputs / "browser.json"),
            },
        },
        "rollback": {
            "backup_path": str(outputs / "aq4-backup.json"),
            "systemd_unit_sha256": "6" * 64,
            "environment_sha256": "7" * 64,
        },
        "prior_outcome": None,
    }
    BINDING.campaign_authorization.issue_authorization(
        authorization_document,
        authorization,
        now=now,
        policy=policy,
    )
    claim = BINDING.campaign_authorization.claim_authorization(
        authorization,
        now=now,
        policy=policy,
    )
    observer = BINDING.ActiveManifestBinding(
        candidate_path=candidate,
        active_path=active,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_stages=("preflight",),
        authorization_path=authorization,
        campaign_name="sq8_full",
        run_id="sq8-run",
        final_path=final,
        expected_source_commit="a" * 40,
        authorization_policy=policy,
        authorization_now=lambda: now,
    )
    observer.observe("preflight")
    summary = json.loads(observer.artifacts().binding_json)
    assert summary["claim"]["sha256"] == claim.snapshot.sha256
    assert summary["claim"]["authorization_sha256"] == (
        claim.authorization.snapshot.sha256
    )
    assert summary["campaign"] == {
        "name": "sq8_full",
        "run_id": "sq8-run",
        "final_path": str(final),
    }

    with pytest.raises(BINDING.ActiveBindingError, match="run/output"):
        BINDING.ActiveManifestBinding(
            candidate_path=candidate,
            active_path=active,
            expected_sha256=hashlib.sha256(raw).hexdigest(),
            expected_stages=("preflight",),
            authorization_path=authorization,
            campaign_name="sq8_full",
            run_id="wrong-run",
            final_path=final,
            authorization_policy=policy,
            authorization_now=lambda: now,
        )
    assert stat.S_IMODE(claim.snapshot.path.stat().st_mode) == 0o444


@pytest.mark.parametrize(
    ("mode", "candidate", "active", "sha"),
    [
        ("legacy", "candidate", None, None),
        ("v2", None, "active", "a" * 64),
        ("v2", "candidate", None, "a" * 64),
        ("v2", "candidate", "active", None),
    ],
)
def test_explicit_dispatch_rejects_mixed_or_partial_inputs(
    tmp_path: Path,
    mode: str,
    candidate: str | None,
    active: str | None,
    sha: str | None,
) -> None:
    with pytest.raises(BINDING.ActiveBindingError):
        BINDING.optional_v2_binding(
            mode=mode,
            candidate_path=None if candidate is None else tmp_path / candidate,
            active_path=None if active is None else tmp_path / active,
            expected_sha256=sha,
            expected_stages=("preflight",),
        )


def test_legacy_dispatch_has_no_file_side_effect(tmp_path: Path) -> None:
    assert (
        BINDING.optional_v2_binding(
            mode="legacy",
            candidate_path=None,
            active_path=None,
            expected_sha256=None,
            expected_stages=("preflight",),
        )
        is None
    )
    assert list(tmp_path.iterdir()) == []
