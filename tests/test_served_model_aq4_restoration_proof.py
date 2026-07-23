from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools import served_model_aq4_restoration_proof as PROOF


AUTH_HASH = "1" * 64
CLAIM_HASH = "2" * 64
WORKER_HASH = "3" * 64
MANIFEST_RAW = b'{"schema_version":"ullm.served_model.v2"}\n'
MANIFEST_HASH = hashlib.sha256(MANIFEST_RAW).hexdigest()
ACTIVE = Path("/etc/ullm/served-models/active.json")
SERVICE = "ullm-openai.service"


def valid_proof() -> dict[str, object]:
    return {
        "schema_version": PROOF.SCHEMA_VERSION,
        "authorization_sha256": AUTH_HASH,
        "claim_sha256": CLAIM_HASH,
        "captured_at": "2026-07-24T12:00:00Z",
        "active_manifest": {
            "path": str(ACTIVE),
            "expected_sha256": MANIFEST_HASH,
            "observed_sha256": MANIFEST_HASH,
            "bytes_equal": True,
        },
        "service": {
            "unit": SERVICE,
            "active_state": "active",
            "sub_state": "running",
            "boot_id": "11111111-2222-3333-4444-555555555555",
            "n_restarts": 0,
        },
        "gateway": {
            "pid": 100,
            "ppid": 1,
            "starttime_ticks": 10,
            "executable_sha256": "4" * 64,
        },
        "worker": {
            "pid": 101,
            "ppid": 100,
            "starttime_ticks": 11,
            "executable_sha256": WORKER_HASH,
        },
        "endpoints": {
            "gateway_healthz": {"status": 200},
            "gateway_readyz": {"status": 200},
            "gateway_models": {
                "status": 200,
                "model_ids": ["ullm-qwen3.5-9b-aq4"],
            },
            "openwebui_health": {"status": 200},
            "openwebui_models": {
                "status": 200,
                "model_ids": ["ullm-qwen3.5-9b-aq4"],
            },
        },
        "epoch_stable": True,
        "passed": True,
    }


def validate(value: dict[str, object]) -> None:
    PROOF.validate_proof(
        value,
        authorization_sha256=AUTH_HASH,
        claim_sha256=CLAIM_HASH,
        active_manifest_path=ACTIVE,
        expected_manifest_sha256=MANIFEST_HASH,
        expected_worker_sha256=WORKER_HASH,
        service_unit=SERVICE,
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["active_manifest"].update(path="/tmp/other.json"),
        lambda value: value["service"].update(unit="other.service"),
        lambda value: value["worker"].update(executable_sha256="9" * 64),
        lambda value: value["worker"].update(ppid=999),
        lambda value: value["endpoints"]["gateway_models"].update(
            model_ids=["ullm-qwen3-14b-sq8"]
        ),
        lambda value: value.update(epoch_stable=False),
    ),
)
def test_structured_live_proof_rejects_identity_mutations(mutate: object) -> None:
    value = valid_proof()
    mutate(value)
    with pytest.raises(PROOF.RestorationProofError):
        validate(value)


def test_collect_live_proof_is_epoch_stable_and_secret_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = tmp_path / "api-key"
    session = tmp_path / "session.jwt"
    api_key.write_text("private-api-key\n", encoding="ascii")
    session.write_text("private-session-token\n", encoding="ascii")
    api_key.chmod(0o600)
    session.chmod(0o600)
    service = valid_proof()["service"]
    gateway = valid_proof()["gateway"]
    worker = valid_proof()["worker"]

    def service_reader(_unit: str) -> tuple[dict, dict, dict]:
        return (
            copy.deepcopy(service),
            copy.deepcopy(gateway),
            copy.deepcopy(worker),
        )

    def http_json(
        *,
        port: int,
        target: str,
        authorization: bytes | None = None,
    ) -> tuple[int, object]:
        if target.endswith("models"):
            return 200, {"data": [{"id": "ullm-qwen3.5-9b-aq4"}]}
        return 200, {"status": "ok"}

    monkeypatch.setattr(PROOF, "_http_json", http_json)
    result = PROOF.collect_live_proof(
        authorization_sha256=AUTH_HASH,
        claim_sha256=CLAIM_HASH,
        active_manifest_path=ACTIVE,
        expected_manifest_sha256=MANIFEST_HASH,
        expected_worker_sha256=WORKER_HASH,
        service_unit=SERVICE,
        api_key_file=api_key,
        openwebui_session_token_file=session,
        manifest_reader=lambda _path: MANIFEST_RAW,
        now=lambda: datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
        service_reader=service_reader,
    )

    validate(result)
    serialized = PROOF.canonical_json_bytes(result)
    assert b"private-api-key" not in serialized
    assert b"private-session-token" not in serialized
