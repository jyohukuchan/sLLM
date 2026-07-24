from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_module(
    "test_sq8_v2_active_manifest_validator_tool",
    TOOLS / "validate-sq8-openwebui-release.py",
)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("ascii")
        + b"\n"
    )


def fixture(tmp_path: Path) -> tuple[object, list[dict[str, object]]]:
    candidate = {
        "schema_version": "ullm.served_model.v2",
        "public": {
            "id": "ullm-qwen3-14b-sq8",
            "revision": VALIDATOR.MODEL_REVISION,
        },
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
    candidate_raw = canonical(candidate)
    candidate_sha = hashlib.sha256(candidate_raw).hexdigest()
    (tmp_path / "candidate-served-model.json").write_bytes(candidate_raw)
    source_path = str(tmp_path / "source-candidate.json")
    active_path = VALIDATOR.FIXED_ACTIVE_MANIFEST_PATH
    claim = {
        "path": str(tmp_path / "claim.json"),
        "sha256": "6" * 64,
        "bytes": 123,
        "authorization_path": str(tmp_path / "authorization.json"),
        "authorization_sha256": "7" * 64,
    }
    file_identity = {
        "device": 1,
        "inode": 2,
        "mode": 0o444,
        "links": 1,
        "uid": 1000,
        "gid": 1000,
        "bytes": len(candidate_raw),
        "mtime_ns": 10,
        "ctime_ns": 10,
    }
    rows: list[dict[str, object]] = []
    for sequence, stage in enumerate(VALIDATOR.ACTIVE_BINDING_PHASE_ORDER):
        rows.append(
            {
                "schema_version": (
                    "ullm.served_model.active_manifest_observation.v1"
                ),
                "sequence": sequence,
                "stage": stage,
                "observed_unix_ns": 100 + sequence,
                "observed_monotonic_ns": 200 + sequence,
                "candidate": {
                    "path": source_path,
                    "sha256": candidate_sha,
                    "identity": file_identity,
                },
                "active": {
                    "path": active_path,
                    "sha256": candidate_sha,
                    "identity": {**file_identity, "inode": 3},
                },
                "bytes_equal": True,
                "claim": claim,
            }
        )
    (tmp_path / "active-manifest-observations.jsonl").write_bytes(
        b"".join(canonical(row) for row in rows)
    )
    identity = types.SimpleNamespace(
        model_identity={
            "schema_version": VALIDATOR.MODEL_IDENTITY_SCHEMA_V2,
            "served_model_manifest": {
                "artifact": "candidate-served-model.json",
                "source_path": source_path,
                "bytes": len(candidate_raw),
                "sha256": candidate_sha,
                "schema_version": "ullm.served_model.v2",
                "model_id": "ullm-qwen3-14b-sq8",
                "model_revision": VALIDATOR.MODEL_REVISION,
                "format_id": "SQ8_0",
                "worker_protocol": "ullm.worker.v2",
                "worker_binary_sha256": "4" * 64,
                "promotion_source_commit": "a" * 40,
                "promotion_receipt_sha256": "5" * 64,
            },
            "campaign_authorization_claim": claim,
        }
    )
    return identity, rows


def test_v2_validator_recomputes_candidate_and_ordered_observation_binding(
    tmp_path: Path,
) -> None:
    identity, _rows = fixture(tmp_path)

    result = VALIDATOR.validate_v2_active_manifest_evidence(
        tmp_path, identity
    )

    assert result["observation_count"] == len(
        VALIDATOR.ACTIVE_BINDING_PHASE_ORDER
    )
    assert result["stages"] == list(VALIDATOR.ACTIVE_BINDING_PHASE_ORDER)


def test_v2_validator_rejects_stage_or_exact_candidate_byte_drift(
    tmp_path: Path,
) -> None:
    identity, rows = fixture(tmp_path)
    rows[2]["stage"] = "wrong"
    (tmp_path / "active-manifest-observations.jsonl").write_bytes(
        b"".join(canonical(row) for row in rows)
    )
    with pytest.raises(VALIDATOR.ValidationError):
        VALIDATOR.validate_v2_active_manifest_evidence(tmp_path, identity)

    _identity, _rows = fixture(tmp_path)
    (tmp_path / "candidate-served-model.json").write_bytes(
        (tmp_path / "candidate-served-model.json").read_bytes() + b" "
    )
    with pytest.raises(
        VALIDATOR.ValidationError,
        match="bytes differ",
    ):
        VALIDATOR.validate_v2_active_manifest_evidence(tmp_path, identity)


def test_v2_validator_rejects_consistent_nonproduction_active_path(
    tmp_path: Path,
) -> None:
    identity, rows = fixture(tmp_path)
    for row in rows:
        row["active"]["path"] = str(tmp_path / "candidate-copy-active.json")
    (tmp_path / "active-manifest-observations.jsonl").write_bytes(
        b"".join(canonical(row) for row in rows)
    )

    with pytest.raises(
        VALIDATOR.ValidationError,
        match="fixed production path",
    ):
        VALIDATOR.validate_v2_active_manifest_evidence(tmp_path, identity)
