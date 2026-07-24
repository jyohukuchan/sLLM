from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/run-openwebui-reasoning-browser-smoke.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reasoning_browser_runner", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()

SESSION_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjQwMDAwMDAwMDB9.signature"


def test_v2_dispatch_binds_reasoning_browser_run_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.json"
    observed: dict[str, object] = {}

    def construct(**arguments: object) -> object:
        observed.update(arguments)
        return type(
            "Binding",
            (),
            {"candidate": type("Candidate", (), {"path": candidate})()},
        )()

    monkeypatch.setattr(TOOL, "optional_v2_binding", construct)
    manifest, binding = TOOL._select_manifest_and_binding(
        active_binding_mode="v2",
        manifest=None,
        candidate_served_model_manifest=candidate,
        active_served_model_manifest=tmp_path / "active.json",
        expected_served_model_manifest_sha256="a" * 64,
        campaign_authorization=tmp_path / "authorization.json",
        run_id="reasoning-browser-run",
        output=tmp_path / "browser-output.json",
    )

    assert binding is not None
    assert manifest == candidate
    assert observed["campaign_name"] == "reasoning_browser"
    assert observed["run_id"] == "reasoning-browser-run"
    assert observed["final_path"] == tmp_path / "browser-output.json"


def _transaction_environment(
    *,
    stage: str,
    staging: Path,
    authorization: Path,
    claim: Path,
    authorization_sha256: str = "a" * 64,
    claim_sha256: str = "b" * 64,
) -> dict[str, str]:
    return {
        TOOL.TRANSACTION_STAGING_OUTPUT_ENV: str(staging),
        TOOL.TRANSACTION_STAGE_ENV: stage,
        TOOL.TRANSACTION_AUTHORIZATION_ENV: str(authorization),
        TOOL.TRANSACTION_CLAIM_ENV: str(claim),
        TOOL.TRANSACTION_AUTHORIZATION_SHA256_ENV: authorization_sha256,
        TOOL.TRANSACTION_CLAIM_SHA256_ENV: claim_sha256,
    }


def _v2_transaction_binding(
    *,
    final: Path,
    authorization: Path,
    claim: Path,
    run_id: str = "browser-run",
) -> SimpleNamespace:
    return SimpleNamespace(
        campaign_name="reasoning_browser",
        run_id=run_id,
        final_path=final,
        claim=SimpleNamespace(
            authorization_path=authorization,
            authorization_sha256="a" * 64,
            path=claim,
            sha256="b" * 64,
        ),
    )


def test_transaction_staging_is_opt_in_and_preserves_browser_default() -> None:
    output = Path("relative-legacy-browser.json")

    selected, transaction_run_id = TOOL._transaction_publication_output(
        authorized_output=output,
        active_binding_mode="legacy",
        campaign_authorization_path=None,
        run_id=None,
        active_binding=None,
        environment={
            TOOL.TRANSACTION_STAGE_ENV: "forged-but-not-opted-in",
            TOOL.TRANSACTION_CLAIM_SHA256_ENV: "not-a-hash",
        },
    )

    assert selected == output
    assert transaction_run_id is None


def test_v2_browser_transaction_staging_is_mandatory(tmp_path: Path) -> None:
    with pytest.raises(TOOL.SmokeError, match="requires locked transaction"):
        TOOL._transaction_publication_output(
            authorized_output=tmp_path / "authorized-browser",
            active_binding_mode="v2",
            campaign_authorization_path=tmp_path / "authorization.json",
            run_id="browser-run",
            active_binding=SimpleNamespace(),
            environment={},
        )


def test_sq8_browser_staging_binds_claim_stage_run_and_final_without_leak(
    tmp_path: Path,
) -> None:
    final = tmp_path / "authorized-browser"
    staging = tmp_path / "private-browser-stage"
    authorization = tmp_path / "authorization.json"
    claim = tmp_path / "claim.json"
    binding = _v2_transaction_binding(
        final=final,
        authorization=authorization,
        claim=claim,
    )

    selected, transaction_run_id = TOOL._transaction_publication_output(
        authorized_output=final,
        active_binding_mode="v2",
        campaign_authorization_path=authorization,
        run_id="browser-run",
        active_binding=binding,
        environment=_transaction_environment(
            stage="reasoning_browser",
            staging=staging,
            authorization=authorization,
            claim=claim,
        ),
    )

    assert selected == staging
    assert transaction_run_id == "browser-run"
    public_result = {
        "output": str(final),
        "evidence": str(final / TOOL.BROWSER_EVIDENCE_FILE),
        "run_id": transaction_run_id,
    }
    assert str(staging) not in json.dumps(public_result, sort_keys=True)
    assert binding.final_path == final


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong-stage",
        "wrong-authorization",
        "wrong-authorization-hash",
        "wrong-final",
        "double-slash",
        "equal",
        "overlap",
        "existing",
    ],
)
def test_sq8_browser_staging_rejects_forged_environment_and_paths(
    tmp_path: Path,
    mutation: str,
) -> None:
    final = tmp_path / "authorized-browser"
    staging = tmp_path / "private-browser-stage"
    authorization = tmp_path / "authorization.json"
    claim = tmp_path / "claim.json"
    binding = _v2_transaction_binding(
        final=final,
        authorization=authorization,
        claim=claim,
    )
    environment = _transaction_environment(
        stage="reasoning_browser",
        staging=staging,
        authorization=authorization,
        claim=claim,
    )
    if mutation == "wrong-stage":
        environment[TOOL.TRANSACTION_STAGE_ENV] = "reasoning_release"
    elif mutation == "wrong-authorization":
        environment[TOOL.TRANSACTION_AUTHORIZATION_ENV] = str(
            tmp_path / "forged-authorization.json"
        )
    elif mutation == "wrong-authorization-hash":
        environment[TOOL.TRANSACTION_AUTHORIZATION_SHA256_ENV] = "c" * 64
    elif mutation == "wrong-final":
        binding.final_path = tmp_path / "different-final"
    elif mutation == "double-slash":
        environment[TOOL.TRANSACTION_STAGING_OUTPUT_ENV] = (
            f"{tmp_path}//private-browser-stage"
        )
    elif mutation == "equal":
        environment[TOOL.TRANSACTION_STAGING_OUTPUT_ENV] = str(final)
    elif mutation == "overlap":
        environment[TOOL.TRANSACTION_STAGING_OUTPUT_ENV] = str(
            final / "nested-stage"
        )
    elif mutation == "existing":
        staging.mkdir()

    with pytest.raises(TOOL.SmokeError, match="transaction"):
        TOOL._transaction_publication_output(
            authorized_output=final,
            active_binding_mode="v2",
            campaign_authorization_path=authorization,
            run_id="browser-run",
            active_binding=binding,
            environment=environment,
        )


def test_fresh_aq4_browser_staging_loads_claim_and_derives_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "authorized-aq4-browser.json"
    staging = tmp_path / "private-aq4-browser.json"
    authorization = tmp_path / "authorization.json"
    claim = tmp_path / "claim.json"
    record = SimpleNamespace(
        snapshot=SimpleNamespace(path=claim, sha256="b" * 64),
        authorization=SimpleNamespace(
            snapshot=SimpleNamespace(
                path=authorization,
                sha256="a" * 64,
            ),
            document={
                "campaigns": {
                    "aq4_reasoning_browser": {
                        "run_id": "aq4-browser-run",
                        "final_path": str(final),
                    }
                }
            },
        ),
    )
    monkeypatch.setattr(
        TOOL.campaign_authorization,
        "load_live_claim",
        lambda path, **_kwargs: record
        if path == authorization
        else pytest.fail("unexpected authorization path"),
    )
    environment = _transaction_environment(
        stage="aq4_reasoning_browser",
        staging=staging,
        authorization=authorization,
        claim=claim,
    )

    selected, transaction_run_id = TOOL._transaction_publication_output(
        authorized_output=final,
        active_binding_mode="legacy",
        campaign_authorization_path=None,
        run_id=None,
        active_binding=None,
        environment=environment,
    )

    assert selected == staging
    assert transaction_run_id == "aq4-browser-run"
    with pytest.raises(TOOL.SmokeError, match="authorization claim or output"):
        TOOL._transaction_publication_output(
            authorized_output=tmp_path / "forged-final.json",
            active_binding_mode="legacy",
            campaign_authorization_path=None,
            run_id=None,
            active_binding=None,
            environment=environment,
        )


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evidence(model_id: str, switch_model_id: str | None = None) -> dict:
    primary = digest(model_id)

    def request(model_hash: str, suffix: str) -> dict:
        return {
            "sha256": digest(f"request-{suffix}"),
            "utf8_bytes": 128,
            "model_id_sha256": model_hash,
            "has_reasoning_content_key": True,
            "assistant_has_reasoning_content": False,
        }

    result = {
        "schema_version": "ullm.openwebui.reasoning_browser_smoke.v2",
        "model_id_sha256": primary,
        "first_answer": {"utf8_bytes": 20, "sha256": "c" * 64},
        "expanded_view": {"utf8_bytes": 40, "sha256": "f" * 64},
        "second_answer": {"utf8_bytes": 21, "sha256": "d" * 64},
        "reasoning_details_expanded": True,
        "provider_request_count": 2,
        "provider_requests": [
            request(primary, "one"),
            request(primary, "two"),
        ],
        "hidden_reasoning_reinserted": False,
        "page_error_count": 0,
        "page_error_digests": [],
    }
    if switch_model_id is not None:
        switched = digest(switch_model_id)
        result.update(
            {
                "provider_switch_performed": True,
                "provider_switch_model_id_sha256": switched,
                "provider_switch_answer": {"utf8_bytes": 22, "sha256": "3" * 64},
                "provider_return_performed": True,
                "provider_return_model_id_sha256": primary,
                "provider_return_answer": {"utf8_bytes": 23, "sha256": "6" * 64},
                "provider_request_count": 4,
                "provider_requests": [
                    request(primary, "one"),
                    request(primary, "two"),
                    request(switched, "three"),
                    request(primary, "four"),
                ],
            }
        )
    return result


class FakeProcess:
    def __init__(self, output, payload: bytes) -> None:
        self.output = output
        self.payload = payload

    def wait(self, *, timeout: float) -> int:
        del timeout
        self.output.write(self.payload)
        self.output.flush()
        return 0


def test_runner_publishes_valid_hash_only_evidence_and_binds_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text(SESSION_JWT + "\n", encoding="ascii")
    script = tmp_path / "smoke.cjs"
    script.write_text("console.log('{}')\n", encoding="ascii")
    output = tmp_path / "browser.json"
    model_id = "ullm-qwen3.5-9b-aq4"
    switch_model_id = "llama-qwen3.5-9b-ud-q4"
    payload = (json.dumps(evidence(model_id, switch_model_id)) + "\n").encode("ascii")
    commands: list[list[str]] = []

    def fake_popen(command, *, stdout, **kwargs):
        del kwargs
        commands.append(command)
        return FakeProcess(stdout, payload)

    monkeypatch.setattr(TOOL.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        TOOL,
        "_validate_manifest_identity",
        lambda _path, _model: {
            "manifest_sha256": "m" * 64,
            "format_id": "AQ4_0",
            "worker": {"binary": "/opt/ullm/bin/ullm-aq4-worker"},
        },
    )
    result = TOOL.execute(
        output=output,
        manifest=tmp_path / "manifest.json",
        openwebui_session_token_file=token,
        browser_image="sha256:" + "a" * 64,
        openwebui_url="http://127.0.0.1:3000/",
        model_id=model_id,
        model_name="uLLM Qwen3.5 9B AQ4",
        switch_model_id=switch_model_id,
        switch_model_name="llama.cpp Qwen3.5 9B UD-Q4_K_XL",
        browser_script=script,
    )

    assert result["provider_request_count"] == 4
    assert output.read_bytes() == payload
    assert json.loads(output.read_text(encoding="ascii"))["model_id_sha256"] == digest(
        model_id
    )
    assert commands
    assert "--network=host" in commands[0]
    assert "NODE_PATH=/usr/src/app/node_modules" in commands[0]
    assert f"ULLM_MODEL_ID={model_id}" in commands[0]
    assert f"OPENWEBUI_SWITCH_MODEL_ID={switch_model_id}" in commands[0]


def test_runner_publishes_gate_eligible_evidence_without_a_provider_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text(SESSION_JWT + "\n", encoding="ascii")
    script = tmp_path / "smoke.cjs"
    script.write_text("console.log('{}')\n", encoding="ascii")
    output = tmp_path / "browser.json"
    model_id = "ullm-qwen3.5-9b-aq4"
    payload = (json.dumps(evidence(model_id)) + "\n").encode("ascii")
    commands: list[list[str]] = []

    def fake_popen(command, *, stdout, **kwargs):
        del kwargs
        commands.append(command)
        return FakeProcess(stdout, payload)

    monkeypatch.setattr(TOOL.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        TOOL,
        "_validate_manifest_identity",
        lambda _path, _model: {
            "manifest_sha256": "m" * 64,
            "format_id": "AQ4_0",
            "worker": {"binary": "/opt/ullm/bin/ullm-aq4-worker"},
        },
    )
    result = TOOL.execute(
        output=output,
        manifest=tmp_path / "manifest.json",
        openwebui_session_token_file=token,
        browser_image="sha256:" + "a" * 64,
        openwebui_url="http://127.0.0.1:3000/",
        model_id=model_id,
        model_name="uLLM Qwen3.5 9B AQ4",
        browser_script=script,
    )

    document = json.loads(output.read_text(encoding="ascii"))
    assert result["provider_request_count"] == 2
    assert TOOL._load_validator().validate(output)["gate_eligible"] is True
    assert not (set(document) & TOOL.SWITCH_EVIDENCE_FIELDS)
    assert all("OPENWEBUI_SWITCH_MODEL_" not in part for part in commands[0])


def test_fresh_aq4_runner_publishes_only_to_transaction_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = tmp_path / "token"
    token.write_text(SESSION_JWT + "\n", encoding="ascii")
    script = tmp_path / "smoke.cjs"
    script.write_text("console.log('{}')\n", encoding="ascii")
    final = tmp_path / "authorized-browser.json"
    staging = tmp_path / "private-stage" / "browser.json"
    authorization = tmp_path / "authorization.json"
    claim = tmp_path / "claim.json"
    model_id = "ullm-qwen3.5-9b-aq4"
    payload = (json.dumps(evidence(model_id)) + "\n").encode("ascii")

    def fake_popen(_command, *, stdout, **_kwargs):
        return FakeProcess(stdout, payload)

    monkeypatch.setattr(TOOL.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        TOOL,
        "_validate_manifest_identity",
        lambda _path, _model: {
            "manifest_sha256": "m" * 64,
            "format_id": "AQ4_0",
            "worker": {"binary": "/opt/ullm/bin/ullm-aq4-worker"},
        },
    )
    record = SimpleNamespace(
        snapshot=SimpleNamespace(path=claim, sha256="b" * 64),
        authorization=SimpleNamespace(
            snapshot=SimpleNamespace(
                path=authorization,
                sha256="a" * 64,
            ),
            document={
                "campaigns": {
                    "aq4_reasoning_browser": {
                        "run_id": "fresh-aq4-browser",
                        "final_path": str(final),
                    }
                }
            },
        ),
    )
    monkeypatch.setattr(
        TOOL.campaign_authorization,
        "load_live_claim",
        lambda _path, **_kwargs: record,
    )
    for name, value in _transaction_environment(
        stage="aq4_reasoning_browser",
        staging=staging,
        authorization=authorization,
        claim=claim,
    ).items():
        monkeypatch.setenv(name, value)

    result = TOOL.execute(
        output=final,
        manifest=tmp_path / "manifest.json",
        openwebui_session_token_file=token,
        browser_image="sha256:" + "a" * 64,
        openwebui_url="http://127.0.0.1:3000/",
        model_id=model_id,
        model_name="uLLM Qwen3.5 9B AQ4",
        browser_script=script,
    )

    assert not final.exists()
    assert staging.read_bytes() == payload
    assert result["output"] == str(final)
    assert result["run_id"] == "fresh-aq4-browser"
    assert str(staging) not in json.dumps(result, sort_keys=True)


def test_legacy_browser_publication_never_replaces_a_raced_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "browser.json"
    output.write_bytes(b"racer-owned\n")

    with pytest.raises(TOOL.SmokeError, match="already exists"):
        TOOL._atomic_publish(output, b'{"fixture":true}\n')

    assert output.read_bytes() == b"racer-owned\n"


def test_v2_browser_directory_publication_never_replaces_a_raced_output(
    tmp_path: Path,
) -> None:
    stage = tmp_path / ".browser.incomplete"
    stage.mkdir()
    target = tmp_path / "browser"
    target.mkdir()
    marker = target / "belongs-to-racer"
    marker.write_bytes(b"preserve")
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(TOOL.SmokeError, match="already exists"):
            TOOL._rename_directory_noreplace(
                descriptor,
                stage.name,
                target.name,
            )
    finally:
        os.close(descriptor)

    assert marker.read_bytes() == b"preserve"
    assert stage.is_dir()


def test_v2_staging_publication_validates_with_actual_lineage_root_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "private-browser-stage"
    authorized_final = tmp_path / "authorized-browser-final"
    artifacts = TOOL.ActiveBindingArtifacts(
        b'{"candidate":true}\n',
        b'{"observation":true}\n',
        b'{"binding":true}\n',
    )
    calls: list[tuple[Path, Path | None]] = []

    class Validator:
        @staticmethod
        def validate(
            path: Path,
            *,
            lineage_root_override: Path | None = None,
        ) -> dict[str, object]:
            calls.append((path, lineage_root_override))
            return {"gate_eligible": True}

    monkeypatch.setattr(TOOL, "_load_validator", lambda: Validator)

    published = TOOL._publish_v2_output_directory(
        staging,
        artifacts,
        b'{"evidence":true}\n',
        lineage_final_output=authorized_final,
    )

    assert published == staging
    assert staging.is_dir()
    assert not authorized_final.exists()
    assert len(calls) == 2
    assert calls[0][1] == calls[0][0].parent
    assert calls[1] == (
        staging / TOOL.BROWSER_EVIDENCE_FILE,
        staging,
    )
    assert {
        entry.name for entry in staging.iterdir()
    } == TOOL.BROWSER_OUTPUT_FILES_V2


def test_active_binding_path_publishes_lineage_bearing_v4_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text(SESSION_JWT + "\n", encoding="ascii")
    script = tmp_path / "smoke.cjs"
    script.write_text("console.log('{}')\n", encoding="ascii")
    output = tmp_path / "browser-v4"
    staging = tmp_path / "private-browser-v4"
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}\n", encoding="ascii")
    candidate_raw = candidate.read_bytes()
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
    model_id = "ullm-qwen3-14b-sq8"
    payload = (json.dumps(evidence(model_id)) + "\n").encode("ascii")
    stages: list[str] = []
    claim_reference = claim

    class Binding:
        campaign_name = "reasoning_browser"
        run_id = "browser-run"
        final_path = output
        claim = SimpleNamespace(
            authorization_path=authorization,
            authorization_sha256=claim_reference["authorization_sha256"],
            path=claim_path,
            sha256=claim_reference["sha256"],
        )

        def observe(self, stage: str) -> None:
            stages.append(stage)

        def artifacts(self) -> object:
            identity = {
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
                    "schema_version": "ullm.served_model.active_manifest_observation.v1",
                    "sequence": sequence,
                    "stage": stage,
                    "observed_unix_ns": sequence,
                    "observed_monotonic_ns": sequence,
                    "candidate": {
                        "path": str(candidate),
                        "sha256": candidate_sha256,
                        "identity": identity,
                    },
                    "active": {
                        "path": "/etc/ullm/served-models/active.json",
                        "sha256": candidate_sha256,
                        "identity": identity,
                    },
                    "bytes_equal": True,
                    "claim": claim,
                }
                for sequence, stage in enumerate(stages)
            ]
            observations = b"".join(
                (
                    json.dumps(
                        row, separators=(",", ":"), sort_keys=True
                    )
                    + "\n"
                ).encode("ascii")
                for row in rows
            )
            binding_document = {
                "schema_version": "ullm.served_model.active_binding.v1",
                "status": "complete",
                "candidate": {
                    "artifact": "candidate-served-model.json",
                    "source_path": str(candidate),
                    "sha256": candidate_sha256,
                    "bytes": len(candidate_raw),
                },
                "actual_active_path": (
                    "/etc/ullm/served-models/active.json"
                ),
                "expected_stages": list(TOOL.ACTIVE_BINDING_STAGES),
                "observation_count": len(rows),
                "observations": {
                    "artifact": "active-manifest-observations.jsonl",
                    "sha256": hashlib.sha256(observations).hexdigest(),
                    "bytes": len(observations),
                },
                "claim": claim,
                "campaign": {
                    "name": "reasoning_browser",
                    "run_id": "browser-run",
                    "final_path": str(output),
                },
            }
            binding_raw = (
                json.dumps(
                    binding_document, separators=(",", ":"), sort_keys=True
                )
                + "\n"
            ).encode("ascii")
            return TOOL.ActiveBindingArtifacts(
                candidate_raw,
                observations,
                binding_raw,
            )

    binding = Binding()
    monkeypatch.setattr(
        TOOL,
        "_select_manifest_and_binding",
        lambda **_kwargs: (candidate, binding),
    )
    monkeypatch.setattr(
        TOOL,
        "_validate_manifest_identity",
        lambda _path, _model: {
            "manifest_sha256": candidate_sha256,
            "format_id": "SQ8_0",
            "worker": {
                "binary": "/opt/ullm/bin/ullm-sq8-worker",
                "binary_sha256": "2" * 64,
            },
        },
    )
    image_policy = TOOL._load_openwebui_image_verifier().authorization
    v3_identity = {
        "manifest_sha256": candidate_sha256,
        "worker_binary_sha256": "2" * 64,
        "tokenizer_sha256": "3" * 64,
        "openwebui_image": image_policy.FIXED_OPENWEBUI_IMAGE,
    }
    observed_server_images: list[str] = []

    def release_identity(
        _manifest: Path,
        _summary: dict[str, object],
        server_image: str,
    ) -> tuple[str, dict[str, str]]:
        observed_server_images.append(server_image)
        return "5" * 40, v3_identity

    monkeypatch.setattr(
        TOOL,
        "_v3_release_identity",
        release_identity,
    )

    def fake_popen(_command, *, stdout, **_kwargs):
        return FakeProcess(stdout, payload)

    monkeypatch.setattr(TOOL.subprocess, "Popen", fake_popen)
    server_checks: list[tuple[str, str]] = []
    server_observation = {
        "container_id": "1" * 64,
        "image_id": image_policy.FIXED_OPENWEBUI_IMAGE.rsplit("@", 1)[1],
        "config_image": image_policy.FIXED_OPENWEBUI_CONFIG_IMAGE,
        "name": f"/{image_policy.FIXED_OPENWEBUI_CONTAINER_NAME}",
        "running": True,
        "pid": 1234,
        "started_at": "2026-07-24T00:00:00.000000000Z",
    }

    def verify_server(*, docker: str, expected_image: str) -> dict[str, object]:
        server_checks.append((docker, expected_image))
        return dict(server_observation)

    monkeypatch.setattr(TOOL, "_verify_openwebui_server", verify_server)
    for name, value in _transaction_environment(
        stage="reasoning_browser",
        staging=staging,
        authorization=authorization,
        claim=claim_path,
        authorization_sha256=claim["authorization_sha256"],
        claim_sha256=claim["sha256"],
    ).items():
        monkeypatch.setenv(name, value)
    result = TOOL.execute(
        output=output,
        manifest=None,
        active_binding_mode="v2",
        candidate_served_model_manifest=candidate,
        active_served_model_manifest=Path(
            "/etc/ullm/served-models/active.json"
        ),
        expected_served_model_manifest_sha256=candidate_sha256,
        campaign_authorization=tmp_path / "authorization.json",
        run_id="browser-run",
        openwebui_session_token_file=token,
        browser_image="sha256:" + "a" * 64,
        openwebui_image=v3_identity["openwebui_image"],
        openwebui_url="http://127.0.0.1:3000/",
        model_id=model_id,
        model_name="uLLM Qwen3 14B SQ8",
        browser_script=script,
    )

    evidence_path = staging / TOOL.BROWSER_EVIDENCE_FILE
    document = json.loads(evidence_path.read_text(encoding="ascii"))
    assert result["schema_version"] == TOOL.BROWSER_EVIDENCE_SCHEMA_V5
    assert document["source_commit"] == "5" * 40
    assert document["identity"] == v3_identity
    assert document["browser_image"] == "sha256:" + "a" * 64
    assert document["openwebui_server"] == {
        "before": server_observation,
        "after": server_observation,
    }
    assert observed_server_images == [v3_identity["openwebui_image"]]
    assert server_checks == [
        ("docker", v3_identity["openwebui_image"]),
        ("docker", v3_identity["openwebui_image"]),
    ]
    assert (
        TOOL._load_validator().validate(
            evidence_path,
            lineage_root_override=staging,
        )["gate_eligible"]
        is True
    )
    assert stages == list(TOOL.ACTIVE_BINDING_STAGES)
    assert not output.exists()
    assert {entry.name for entry in staging.iterdir()} == TOOL.BROWSER_OUTPUT_FILES_V2
    assert all(entry.stat().st_mode & 0o777 == 0o444 for entry in staging.iterdir())


def test_runner_cli_allows_switch_arguments_to_be_omitted(tmp_path: Path) -> None:
    args = TOOL.parse_args(
        [
            "--output",
            str(tmp_path / "browser.json"),
            "--manifest",
            str(tmp_path / "active.json"),
            "--openwebui-session-token-file",
            str(tmp_path / "token"),
            "--browser-image",
            "sha256:" + "a" * 64,
            "--openwebui-url",
            "http://127.0.0.1:3000/",
            "--model-id",
            "ullm-qwen3.5-9b-aq4",
            "--model-name",
            "uLLM Qwen3.5 9B AQ4",
        ]
    )

    assert args.switch_model_id is None
    assert args.switch_model_name is None
    assert args.openwebui_image is None


def test_runner_rejects_external_model_binding_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text(SESSION_JWT, encoding="ascii")
    script = tmp_path / "smoke.cjs"
    script.write_text("console.log('{}')\n", encoding="ascii")
    output = tmp_path / "browser.json"
    payload = (json.dumps(evidence("candidate", "switch")) + "\n").encode("ascii")

    def fake_popen(command, *, stdout, **kwargs):
        del command, kwargs
        return FakeProcess(stdout, payload)

    monkeypatch.setattr(TOOL.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        TOOL,
        "_validate_manifest_identity",
        lambda _path, _model: {
            "manifest_sha256": "m" * 64,
            "format_id": "AQ4_0",
            "worker": {"binary": "/opt/ullm/bin/ullm-aq4-worker"},
        },
    )
    with pytest.raises(TOOL.SmokeError, match="primary model identity"):
        TOOL.execute(
            output=output,
            manifest=tmp_path / "manifest.json",
            openwebui_session_token_file=token,
            browser_image="sha256:" + "a" * 64,
            openwebui_url="http://127.0.0.1:3000",
            model_id="different",
            model_name="candidate",
            switch_model_id="switch",
            switch_model_name="switch",
            browser_script=script,
        )
    assert not output.exists()


def test_runner_rejects_a_v1_active_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "active-v1.json"
    manifest.write_text(
        json.dumps({"schema_version": "ullm.served_model.v1"}), encoding="ascii"
    )
    with pytest.raises(TOOL.SmokeError, match="not v2"):
        TOOL._validate_manifest_identity(manifest, "ullm-qwen3.5-9b-aq4")


@pytest.mark.parametrize(
    ("format_id", "binary", "expected"),
    [
        ("AQ4_0", "/opt/ullm/bin/ullm-aq4-worker", "ullm-aq4-worker"),
        ("SQ8_0", "/opt/ullm/bin/ullm-sq8-worker", "ullm-sq8-worker"),
    ],
)
def test_browser_worker_process_name_is_manifest_derived(
    format_id: str, binary: str, expected: str
) -> None:
    assert (
        TOOL._worker_process_basename(
            {"format_id": format_id, "worker": {"binary": binary}}
        )
        == expected
    )


@pytest.mark.parametrize(
    ("format_id", "binary"),
    [
        ("SQ8_0", "/opt/ullm/bin/ullm-worker"),
        ("SQ8_0", "ullm-sq8-worker"),
        ("SQ8_0", "/opt/ullm/bin/llama-server"),
        ("SQ8_0", "/opt/ullm/bin/ullm-aq4-worker"),
        ("AQ4_0", "/opt/ullm/bin/ullm-sq8-worker"),
        ("UNKNOWN", "/opt/ullm/bin/ullm-sq8-worker"),
    ],
)
def test_browser_worker_process_name_rejects_unbound_values(
    format_id: str, binary: str
) -> None:
    with pytest.raises(TOOL.SmokeError, match="worker executable"):
        TOOL._worker_process_basename(
            {"format_id": format_id, "worker": {"binary": binary}}
        )


def test_v3_release_identity_rehashes_live_tokenizer_files(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    members = {
        "tokenizer.json": b'{"fixture":true}\n',
        "tokenizer_config.json": b'{"chat_template":"fixture"}\n',
    }
    for name, raw in members.items():
        (tokenizer / name).write_bytes(raw)
    manifest = {
        "promotion": {"source_commit": "1" * 40},
        "tokenizer": {
            "root": str(tokenizer),
            "files": {
                name: hashlib.sha256(raw).hexdigest()
                for name, raw in members.items()
            },
        },
    }
    manifest_path = tmp_path / "candidate.json"
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="ascii",
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    summary = {
        "manifest_sha256": manifest_sha256,
        "worker": {"binary_sha256": "2" * 64},
    }

    source_commit, identity = TOOL._v3_release_identity(
        manifest_path,
        summary,
        "registry.example/open-webui@sha256:" + "3" * 64,
    )

    aggregate = hashlib.sha256()
    for name in sorted(members):
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(hashlib.sha256(members[name]).digest())
    assert source_commit == "1" * 40
    assert identity == {
        "manifest_sha256": manifest_sha256,
        "worker_binary_sha256": "2" * 64,
        "tokenizer_sha256": aggregate.hexdigest(),
        "openwebui_image": "registry.example/open-webui@sha256:" + "3" * 64,
    }

    with pytest.raises(TOOL.SmokeError, match="changed after validation"):
        TOOL._v3_release_identity(
            manifest_path,
            {**summary, "manifest_sha256": "0" * 64},
            "registry.example/open-webui@sha256:" + "3" * 64,
        )
    with pytest.raises(TOOL.SmokeError, match="content-addressed"):
        TOOL._v3_release_identity(
            manifest_path,
            summary,
            "sha256:" + "3" * 64,
        )

    (tokenizer / "tokenizer.json").write_bytes(b"changed\n")
    with pytest.raises(TOOL.SmokeError, match="tokenizer file hash differs"):
        TOOL._v3_release_identity(
            manifest_path,
            summary,
            "registry.example/open-webui@sha256:" + "3" * 64,
        )


def test_v3_upgrade_adds_exact_release_identity_without_mutating_v2() -> None:
    original = evidence("ullm-qwen3-14b-sq8")
    identity = {
        "manifest_sha256": "1" * 64,
        "worker_binary_sha256": "2" * 64,
        "tokenizer_sha256": "3" * 64,
        "openwebui_image": "registry.example/open-webui@sha256:" + "4" * 64,
    }

    upgraded = TOOL._upgrade_browser_evidence_v3(
        original,
        source_commit="5" * 40,
        identity=identity,
    )

    assert original["schema_version"] == TOOL.BROWSER_EVIDENCE_SCHEMA_V2
    assert "identity" not in original
    assert upgraded["schema_version"] == TOOL.BROWSER_EVIDENCE_SCHEMA_V3
    assert upgraded["source_commit"] == "5" * 40
    assert upgraded["identity"] == identity


@pytest.mark.parametrize("value", ["browser:latest", "sha256:ABC" + "a" * 61])
def test_runner_requires_immutable_browser_image(value: str) -> None:
    with pytest.raises(TOOL.SmokeError, match="immutable Docker"):
        TOOL._validate_image(value)


def test_runner_validates_explicit_browser_container_user() -> None:
    assert TOOL._validate_container_user("1000:1000") == (1000, 1000)
    with pytest.raises(TOOL.SmokeError, match="UID:GID"):
        TOOL._validate_container_user("root")


def test_runner_rejects_gateway_api_key_as_openwebui_session() -> None:
    with pytest.raises(TOOL.SmokeError, match="not a JWT"):
        TOOL._validate_openwebui_session_token(
            b"gateway-api-key", minimum_validity_seconds=30, now_seconds=1
        )
    TOOL._validate_openwebui_session_token(
        SESSION_JWT.encode("ascii"), minimum_validity_seconds=30, now_seconds=1
    )


@pytest.mark.parametrize(
    "worker_process_basename",
    ["ullm-aq4-worker", "ullm-sq8-worker"],
)
def test_alternating_r9700_coordinator_serializes_provider_ownership(
    worker_process_basename: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions: list[tuple[str, str]] = []

    def service(_systemctl: str, action: str, name: str) -> None:
        actions.append((action, name))

    def wait(_rocm: str, expected: set[str], timeout_seconds: float = 60.0) -> None:
        del timeout_seconds
        actions.append(("gpu", ",".join(sorted(expected))))

    monkeypatch.setattr(TOOL, "_service_command", service)
    monkeypatch.setattr(TOOL, "_wait_for_gpu_owner", wait)
    monkeypatch.setattr(TOOL, "_wait_for_tcp_port", lambda *_args, **_kwargs: None)
    coordinator = TOOL._AlternatingServiceCoordinator(
        "systemctl",
        "rocm-smi",
        "ullm-openai.service",
        "llama-qwen35-udq4.service",
        worker_process_basename,
    )

    coordinator.transition("before-switch")
    assert coordinator.owner == "llama"
    coordinator.transition("before-return")
    assert coordinator.owner == "ullm"
    assert actions == [
        ("stop", "ullm-openai.service"),
        ("gpu", ""),
        ("start", "llama-qwen35-udq4.service"),
        ("gpu", "llama-server"),
        ("stop", "llama-qwen35-udq4.service"),
        ("gpu", ""),
        ("start", "ullm-openai.service"),
        ("gpu", worker_process_basename),
    ]


def test_gpu_owner_probe_accepts_rocm_no_process_output(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[list[str]] = []

    def run(args, **_kwargs):
        observed.append(args)
        return TOOL.subprocess.CompletedProcess(
            args, 0, stdout="", stderr="WARNING: No JSON data to report.\n"
        )

    monkeypatch.setattr(
        TOOL.subprocess,
        "run",
        run,
    )

    assert TOOL._target_gpu_processes(TOOL.CANONICAL_ROCM_SMI) == []
    assert observed == [
        [
            TOOL.CANONICAL_PYTHON,
            *TOOL.ROCM_PYTHON_ARGUMENTS,
            TOOL.CANONICAL_ROCM_SMI,
            "--showpids",
            "--json",
        ]
    ]
