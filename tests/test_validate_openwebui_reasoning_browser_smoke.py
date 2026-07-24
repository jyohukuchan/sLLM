from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/validate-openwebui-reasoning-browser-smoke.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reasoning_browser_validator", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def evidence() -> dict:
    request = {
        "sha256": "a" * 64,
        "utf8_bytes": 128,
        "model_id_sha256": "b" * 64,
        "has_reasoning_content_key": True,
        "assistant_has_reasoning_content": False,
    }
    return {
        "schema_version": TOOL.SCHEMA_VERSION,
        "model_id_sha256": "b" * 64,
        "first_answer": {"utf8_bytes": 20, "sha256": "c" * 64},
        "expanded_view": {"utf8_bytes": 40, "sha256": "f" * 64},
        "second_answer": {"utf8_bytes": 21, "sha256": "d" * 64},
        "provider_switch_performed": True,
        "provider_switch_model_id_sha256": "2" * 64,
        "provider_switch_answer": {"utf8_bytes": 22, "sha256": "3" * 64},
        "provider_return_performed": True,
        "provider_return_model_id_sha256": "b" * 64,
        "provider_return_answer": {"utf8_bytes": 23, "sha256": "6" * 64},
        "reasoning_details_expanded": True,
        "provider_request_count": 4,
        "provider_requests": [
            request,
            {**request, "sha256": "e" * 64},
            {**request, "sha256": "4" * 64, "model_id_sha256": "2" * 64},
            {**request, "sha256": "5" * 64, "model_id_sha256": "b" * 64},
        ],
        "hidden_reasoning_reinserted": False,
        "page_error_count": 0,
        "page_error_digests": [],
    }


def no_switch_evidence() -> dict:
    value = evidence()
    for field in TOOL.SWITCH_EVIDENCE_FIELDS:
        value.pop(field)
    value["provider_request_count"] = 2
    value["provider_requests"] = value["provider_requests"][:2]
    return value


def v3_evidence() -> dict:
    value = evidence()
    value["schema_version"] = TOOL.SCHEMA_VERSION_V3
    value["source_commit"] = "1" * 40
    value["identity"] = {
        "manifest_sha256": "7" * 64,
        "worker_binary_sha256": "8" * 64,
        "tokenizer_sha256": "9" * 64,
        "openwebui_image": "registry.example/open-webui@sha256:" + "a" * 64,
    }
    return value


def write_v4_evidence(
    tmp_path: Path,
    *,
    schema_version: str = TOOL.SCHEMA_VERSION_V4,
    active_path: str = TOOL.FIXED_ACTIVE_MANIFEST_PATH,
) -> Path:
    if schema_version not in {TOOL.SCHEMA_VERSION_V4, TOOL.SCHEMA_VERSION_V5}:
        raise AssertionError("unsupported lineage fixture schema")
    root = tmp_path / (
        "browser-output-v5"
        if schema_version == TOOL.SCHEMA_VERSION_V5
        else "browser-output-v4"
    )
    root.mkdir()
    candidate_raw = b'{"candidate":true}\n'
    candidate_sha256 = hashlib.sha256(candidate_raw).hexdigest()
    authorization = tmp_path / "authorization.json"
    authorization.write_bytes(b'{"authorization":true}\n')
    authorization.chmod(0o444)
    claim_path = tmp_path / "authorization.claimed.json"
    claim_path.write_bytes(b'{"claim":true}\n')
    claim_path.chmod(0o444)
    claim = {
        "path": str(claim_path),
        "sha256": hashlib.sha256(claim_path.read_bytes()).hexdigest(),
        "bytes": len(claim_path.read_bytes()),
        "authorization_path": str(authorization),
        "authorization_sha256": hashlib.sha256(
            authorization.read_bytes()
        ).hexdigest(),
    }
    file_identity = {
        "device": 1,
        "inode": 2,
        "mode": 0o444,
        "links": 1,
        "uid": 1000,
        "gid": 1000,
        "bytes": len(candidate_raw),
        "mtime_ns": 3,
        "ctime_ns": 4,
    }
    rows = [
        {
            "schema_version": TOOL.ACTIVE_OBSERVATION_SCHEMA,
            "sequence": sequence,
            "stage": stage,
            "observed_unix_ns": sequence,
            "observed_monotonic_ns": sequence,
            "candidate": {
                "path": str(tmp_path / "candidate-source.json"),
                "sha256": candidate_sha256,
                "identity": file_identity,
            },
            "active": {
                "path": active_path,
                "sha256": candidate_sha256,
                "identity": file_identity,
            },
            "bytes_equal": True,
            "claim": claim,
        }
        for sequence, stage in enumerate(TOOL.ACTIVE_BINDING_STAGES)
    ]
    observations_raw = b"".join(
        (
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("ascii")
        for row in rows
    )
    binding = {
        "schema_version": TOOL.ACTIVE_BINDING_SCHEMA,
        "status": "complete",
        "candidate": {
            "artifact": "candidate-served-model.json",
            "source_path": str(tmp_path / "candidate-source.json"),
            "sha256": candidate_sha256,
            "bytes": len(candidate_raw),
        },
        "actual_active_path": active_path,
        "expected_stages": list(TOOL.ACTIVE_BINDING_STAGES),
        "observation_count": len(rows),
        "observations": {
            "artifact": "active-manifest-observations.jsonl",
            "sha256": hashlib.sha256(observations_raw).hexdigest(),
            "bytes": len(observations_raw),
        },
        "claim": claim,
        "campaign": {
            "name": "reasoning_browser",
            "run_id": "reasoning-browser-run",
            "final_path": str(root),
        },
    }
    binding_raw = (
        json.dumps(binding, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")
    artifact_raws = {
        "candidate-served-model.json": candidate_raw,
        "active-manifest-observations.jsonl": observations_raw,
        "active-manifest-binding.json": binding_raw,
    }
    artifacts = {
        name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in artifact_raws.items()
    }
    lineage = {
        "schema_version": TOOL.CAMPAIGN_LINEAGE_SCHEMA_V2,
        "campaign": {
            "name": "reasoning_browser",
            "run_id": "reasoning-browser-run",
            "final_path": str(root),
            "final_kind": "directory",
            "files": sorted(TOOL.BROWSER_OUTPUT_FILES_V2),
        },
        "claim": claim,
        "artifacts": artifacts,
        "artifact_inventory_sha256": TOOL._lineage_inventory(artifacts),
        "observations": {
            "count": len(rows),
            "stages": [
                {
                    "sequence": sequence,
                    "stage": stage,
                    "sha256": hashlib.sha256(
                        observations_raw.splitlines(keepends=True)[sequence]
                    ).hexdigest(),
                }
                for sequence, stage in enumerate(TOOL.ACTIVE_BINDING_STAGES)
            ],
        },
    }
    value = v3_evidence()
    value["schema_version"] = schema_version
    value["identity"]["manifest_sha256"] = candidate_sha256
    value["campaign_lineage"] = lineage
    if schema_version == TOOL.SCHEMA_VERSION_V5:
        value["identity"]["openwebui_image"] = (
            TOOL.authorization.FIXED_OPENWEBUI_IMAGE
        )
        value["browser_image"] = TOOL.authorization.FIXED_BROWSER_IMAGE
        server = {
            "container_id": "1" * 64,
            "image_id": TOOL.authorization.FIXED_OPENWEBUI_IMAGE.rsplit(
                "@",
                1,
            )[1],
            "config_image": TOOL.authorization.FIXED_OPENWEBUI_CONFIG_IMAGE,
            "name": f"/{TOOL.authorization.FIXED_OPENWEBUI_CONTAINER_NAME}",
            "running": True,
            "pid": 1234,
            "started_at": "2026-07-24T00:00:00.000000000Z",
        }
        value["openwebui_server"] = {
            "before": server,
            "after": dict(server),
        }
    evidence_path = root / TOOL.BROWSER_EVIDENCE_FILE
    for name, raw in {
        **artifact_raws,
        TOOL.BROWSER_EVIDENCE_FILE: json.dumps(
            value, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
        + b"\n",
    }.items():
        path = root / name
        path.write_bytes(raw)
        path.chmod(0o444)
    root.chmod(0o555)
    return evidence_path


def test_validator_accepts_hash_only_browser_gate(tmp_path: Path) -> None:
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(evidence()), encoding="ascii")

    report = TOOL.validate(path)

    assert report["structurally_valid"] is True
    assert report["gate_eligible"] is True


def test_validator_accepts_v2_browser_gate_without_a_switch_cycle(tmp_path: Path) -> None:
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(no_switch_evidence()), encoding="ascii")

    report = TOOL.validate(path)

    assert report["input_schema_version"] == TOOL.SCHEMA_VERSION
    assert report["structurally_valid"] is True
    assert report["gate_eligible"] is True
    assert report["provider_request_count"] == 2


def test_validator_accepts_strict_identity_bearing_v3_browser_gate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "browser-v3.json"
    path.write_text(json.dumps(v3_evidence()), encoding="ascii")

    report = TOOL.validate(path)

    assert report["input_schema_version"] == TOOL.SCHEMA_VERSION_V3
    assert report["structurally_valid"] is True
    assert report["gate_eligible"] is True


def test_validator_accepts_v4_and_recomputes_full_campaign_lineage(
    tmp_path: Path,
) -> None:
    path = write_v4_evidence(tmp_path)

    report = TOOL.validate(path)

    assert report["input_schema_version"] == TOOL.SCHEMA_VERSION_V4
    assert report["schema_version"] == TOOL.VALIDATOR_SCHEMA_VERSION_V2
    assert report["campaign_lineage"]["run_id"] == "reasoning-browser-run"
    assert report["campaign_lineage"]["observation_count"] == 5
    assert report["gate_eligible"] is True


def test_validator_rejects_v4_nonproduction_active_path(
    tmp_path: Path,
) -> None:
    path = write_v4_evidence(
        tmp_path,
        active_path=str(tmp_path / "candidate-copy-active.json"),
    )

    with pytest.raises(TOOL.ValidationError, match="active binding differs"):
        TOOL.validate(path)


def test_validator_accepts_v5_with_distinct_browser_and_server_images(
    tmp_path: Path,
) -> None:
    path = write_v4_evidence(
        tmp_path,
        schema_version=TOOL.SCHEMA_VERSION_V5,
    )

    report = TOOL.validate(path)
    document = json.loads(path.read_text(encoding="ascii"))

    assert report["input_schema_version"] == TOOL.SCHEMA_VERSION_V5
    assert report["schema_version"] == TOOL.VALIDATOR_SCHEMA_VERSION_V3
    assert (
        document["browser_image"]
        == TOOL.authorization.FIXED_BROWSER_IMAGE
    )
    assert document["openwebui_server"]["before"] == document[
        "openwebui_server"
    ]["after"]
    assert (
        document["identity"]["openwebui_image"]
        == TOOL.authorization.FIXED_OPENWEBUI_IMAGE
    )
    assert report["gate_eligible"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.pop("browser_image"),
        lambda value: value.__setitem__("browser_image", "browser:latest"),
    ),
)
def test_validator_rejects_v5_browser_image_shape_mutations(
    tmp_path: Path,
    mutation,
) -> None:
    path = write_v4_evidence(
        tmp_path,
        schema_version=TOOL.SCHEMA_VERSION_V5,
    )
    value = json.loads(path.read_text(encoding="ascii"))
    mutation(value)
    path.chmod(0o644)
    path.write_text(json.dumps(value), encoding="ascii")
    path.chmod(0o444)

    with pytest.raises(TOOL.ValidationError):
        TOOL.validate(path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("container_id", "2" * 64),
        ("image_id", "sha256:" + "0" * 64),
        ("pid", 4321),
        ("started_at", "2026-07-24T00:01:00.000000000Z"),
    ),
)
def test_validator_rejects_v5_openwebui_server_change_during_gate(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    path = write_v4_evidence(
        tmp_path,
        schema_version=TOOL.SCHEMA_VERSION_V5,
    )
    value = json.loads(path.read_text(encoding="ascii"))
    value["openwebui_server"]["after"][field] = replacement
    path.chmod(0o644)
    path.write_text(json.dumps(value), encoding="ascii")
    path.chmod(0o444)

    with pytest.raises(TOOL.ValidationError, match="changed during browser"):
        TOOL.validate(path)


def test_validator_rejects_v5_consistent_but_unfixed_openwebui_server(
    tmp_path: Path,
) -> None:
    path = write_v4_evidence(
        tmp_path,
        schema_version=TOOL.SCHEMA_VERSION_V5,
    )
    value = json.loads(path.read_text(encoding="ascii"))
    alternate = {
        "container_id": "1" * 64,
        "image_id": "sha256:" + "0" * 64,
        "config_image": "attacker/open-webui:fixed-looking",
        "name": "/open-webui",
        "running": True,
        "pid": 1234,
        "started_at": "2026-07-24T00:00:00.000000000Z",
    }
    value["openwebui_server"] = {
        "before": alternate,
        "after": dict(alternate),
    }
    path.chmod(0o644)
    path.write_text(json.dumps(value), encoding="ascii")
    path.chmod(0o444)

    with pytest.raises(TOOL.ValidationError, match="fixed identity"):
        TOOL.validate(path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("container_id", "1" * 63),
        ("pid", True),
        ("pid", 0),
        ("started_at", ""),
        ("started_at", "\N{SNOWMAN}"),
    ),
)
def test_validator_rejects_malformed_dynamic_openwebui_server_identity(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    path = write_v4_evidence(
        tmp_path,
        schema_version=TOOL.SCHEMA_VERSION_V5,
    )
    value = json.loads(path.read_text(encoding="ascii"))
    value["openwebui_server"]["before"][field] = replacement
    value["openwebui_server"]["after"][field] = replacement
    path.chmod(0o644)
    path.write_text(json.dumps(value), encoding="ascii")
    path.chmod(0o444)

    with pytest.raises(TOOL.ValidationError, match="openwebui_server"):
        TOOL.validate(path)


def test_validator_rejects_v4_observation_replay(
    tmp_path: Path,
) -> None:
    path = write_v4_evidence(tmp_path)
    root = path.parent
    observations = root / "active-manifest-observations.jsonl"
    observations.chmod(0o644)
    lines = observations.read_bytes().splitlines(keepends=True)
    observations.write_bytes(b"".join([lines[0], *lines[:-1]]))
    observations.chmod(0o444)

    with pytest.raises(TOOL.ValidationError):
        TOOL.validate(path)


def test_validator_rejects_v4_symlinked_lineage_artifact(
    tmp_path: Path,
) -> None:
    path = write_v4_evidence(tmp_path)
    root = path.parent
    root.chmod(0o755)
    candidate = root / "candidate-served-model.json"
    candidate.chmod(0o644)
    raw = candidate.read_bytes()
    candidate.unlink()
    target = tmp_path / "candidate-target.json"
    target.write_bytes(raw)
    candidate.symlink_to(target)
    root.chmod(0o555)

    with pytest.raises(TOOL.ValidationError, match="regular non-symlink"):
        TOOL.validate(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("source_commit", "1" * 39),
        lambda value: value["identity"].__setitem__("manifest_sha256", "A" * 64),
        lambda value: value["identity"].__setitem__(
            "openwebui_image", "sha256:" + "a" * 64
        ),
        lambda value: value["identity"].__setitem__("extra", "a" * 64),
        lambda value: value.pop("identity"),
    ],
)
def test_validator_rejects_v3_identity_mutations(
    tmp_path: Path, mutation
) -> None:
    value = v3_evidence()
    mutation(value)
    path = tmp_path / "browser-v3.json"
    path.write_text(json.dumps(value), encoding="ascii")

    with pytest.raises(TOOL.ValidationError):
        TOOL.validate(path)


def test_validator_does_not_reinterpret_v2_with_v3_identity(
    tmp_path: Path,
) -> None:
    value = v3_evidence()
    value["schema_version"] = TOOL.SCHEMA_VERSION_V2
    path = tmp_path / "mixed-browser.json"
    path.write_text(json.dumps(value), encoding="ascii")

    with pytest.raises(TOOL.ValidationError, match="root fields differ"):
        TOOL.validate(path)


def test_validator_does_not_reinterpret_v3_with_v4_lineage(
    tmp_path: Path,
) -> None:
    v4_path = write_v4_evidence(tmp_path)
    value = json.loads(v4_path.read_text(encoding="ascii"))
    value["schema_version"] = TOOL.SCHEMA_VERSION_V3
    v4_path.chmod(0o644)
    v4_path.write_text(json.dumps(value), encoding="ascii")
    v4_path.chmod(0o444)

    with pytest.raises(TOOL.ValidationError, match="root fields differ"):
        TOOL.validate(v4_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("reasoning_details_expanded", False),
        lambda value: value.__setitem__("response", "secret"),
        lambda value: value.__setitem__("page_error_count", 1),
    ],
)
def test_validator_rejects_unsafe_or_failed_browser_records(tmp_path: Path, mutation) -> None:
    value = evidence()
    mutation(value)
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(value), encoding="ascii")

    with pytest.raises(TOOL.ValidationError):
        TOOL.validate(path)


def test_validator_reports_reinserted_reasoning_as_gate_failure(tmp_path: Path) -> None:
    value = evidence()
    value["provider_requests"][-1]["assistant_has_reasoning_content"] = True
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(value), encoding="ascii")

    report = TOOL.validate(path)

    assert report["structurally_valid"] is True
    assert report["gate_eligible"] is False
    assert report["reasons"] == [
        "last provider request contains assistant reasoning_content"
    ]


def test_validator_require_pass_uses_distinct_exit_code(tmp_path: Path) -> None:
    value = evidence()
    value["provider_requests"][-1]["assistant_has_reasoning_content"] = True
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(value), encoding="ascii")

    assert TOOL.main([str(path)]) == 0
    assert TOOL.main([str(path), "--require-pass"]) == 2


def test_validator_rejects_provider_switch_model_mismatch(tmp_path: Path) -> None:
    value = evidence()
    value["provider_requests"][-2]["model_id_sha256"] = "4" * 64
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(value), encoding="ascii")

    with pytest.raises(TOOL.ValidationError, match="provider switch request model"):
        TOOL.validate(path)


def test_validator_rejects_initial_model_mismatch(tmp_path: Path) -> None:
    value = evidence()
    value["provider_requests"][0]["model_id_sha256"] = "4" * 64
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(value), encoding="ascii")

    with pytest.raises(TOOL.ValidationError, match="initial provider request model"):
        TOOL.validate(path)


def test_validator_rejects_non_switching_provider(tmp_path: Path) -> None:
    value = evidence()
    value["provider_switch_model_id_sha256"] = value["model_id_sha256"]
    value["provider_requests"][-2]["model_id_sha256"] = value["model_id_sha256"]
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(value), encoding="ascii")

    with pytest.raises(TOOL.ValidationError, match="provider switch model is not distinct"):
        TOOL.validate(path)


def test_validator_reads_v1_hash_only_record(tmp_path: Path) -> None:
    value = evidence()
    value["schema_version"] = TOOL.SCHEMA_VERSION_V1
    value.pop("provider_switch_performed")
    value.pop("provider_switch_model_id_sha256")
    value.pop("provider_switch_answer")
    value.pop("provider_return_performed")
    value.pop("provider_return_model_id_sha256")
    value.pop("provider_return_answer")
    for request in value["provider_requests"]:
        request.pop("model_id_sha256")
    value["provider_request_count"] = 2
    value["provider_requests"] = value["provider_requests"][:2]
    path = tmp_path / "browser-v1.json"
    path.write_text(json.dumps(value), encoding="ascii")

    report = TOOL.validate(path)

    assert report["input_schema_version"] == TOOL.SCHEMA_VERSION_V1
    assert report["gate_eligible"] is True


def test_validator_rejects_file_identity_change_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(evidence()), encoding="ascii")
    original_fstat = os.fstat
    calls = 0

    def racing_fstat(descriptor: int):
        nonlocal calls
        calls += 1
        if calls == 2:
            metadata = path.stat()
            os.utime(
                path,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1),
            )
        return original_fstat(descriptor)

    monkeypatch.setattr(TOOL.os, "fstat", racing_fstat)

    with pytest.raises(TOOL.ValidationError, match="changed while being read"):
        TOOL.validate(path)
