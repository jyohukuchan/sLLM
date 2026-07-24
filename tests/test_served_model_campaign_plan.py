from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import served_model_campaign_plan as PLAN


def test_fixed_plan_paths_equal_authorization_policy_defaults() -> None:
    policy = PLAN.authorization.RegistryPolicy()
    assert PLAN.ACTIVE_MANIFEST == policy.active_manifest_path
    assert PLAN.SYSTEMD_UNIT == policy.systemd_unit_path
    assert PLAN.ENVIRONMENT_FILE == policy.environment_file_path
    assert PLAN.SERVICE_UNIT == policy.service_unit
    assert PLAN.OPENWEBUI_SESSION_TOKEN_FILE == Path(
        "/run/ullm-campaign-secrets/openwebui-session.jwt"
    )


def load_script(name: str, module_name: str) -> object:
    spec = importlib.util.spec_from_file_location(module_name, TOOLS / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_script(
    "run-served-model-v2-cross-model-campaign.py",
    "test_cross_model_campaign_cli",
)
CLAIM = load_script(
    "claim-served-model-v2-cross-model-campaign-authorization.py",
    "test_cross_model_claim_cli",
)
RECOVERY = load_script(
    "recover-served-model-v2-cross-model-campaign.py",
    "test_cross_model_recovery_cli",
)
ISSUE = load_script(
    "issue-served-model-v2-cross-model-campaign-authorization.py",
    "test_cross_model_issue_cli",
)


def authorization_document(tmp_path: Path) -> dict[str, object]:
    outputs = tmp_path / "outputs"
    aq4_source = tmp_path / "aq4-source"
    return {
        "source": {"commit": "a" * 40, "tree": "b" * 40},
        "before": {
            "worker_binary_path": str(tmp_path / "aq4-worker"),
            "promotion_source_commit": "e" * 40,
        },
        "aq4_release": {
            "source": {
                "root": str(aq4_source),
                "commit": "e" * 40,
                "tree": "f" * 40,
            },
            "openwebui_image": PLAN.authorization.FIXED_OPENWEBUI_IMAGE,
            "promotion_evidence": {
                "source_path": str(tmp_path / "promotion-evidence.json"),
                "path": str(outputs / "promotion-evidence.json"),
                "sha256": "2" * 64,
            },
            "promotion_receipt": {
                "source_path": str(tmp_path / "promotion-receipt.json"),
                "path": str(outputs / "promotion-receipt.json"),
                "sha256": "3" * 64,
            },
            "release_evidence_path": str(outputs / "release-evidence.json"),
            "release_validator_path": str(outputs / "release-validator.json"),
            "browser_validator_path": str(outputs / "browser-validator.json"),
        },
        "candidate": {
            "manifest_sha256": "c" * 64,
            "worker_binary_sha256": "d" * 64,
        },
        "campaigns": {
            name: {
                "run_id": f"{name}-run",
                "final_path": str(outputs / name),
            }
            for name in (
                "aq4_reasoning_release",
                "aq4_reasoning_browser",
                "aq4_bundle",
                "sq8_full",
                "reasoning_release",
                "reasoning_browser",
            )
        },
        "rollback": {"backup_path": str(outputs / "aq4-backup.json")},
    }


def test_fixed_plan_derives_every_command_without_caller_vectors(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    authorization = tmp_path / "authorization.json"
    candidate = tmp_path / "candidate.json"
    commands = PLAN.derive_commands(
        source_root=source,
        authorization_path=authorization,
        candidate_manifest=candidate,
        authorization_document=authorization_document(tmp_path),
    )

    flattened = [
        *commands.candidate_reconciliation,
        *commands.candidate_checks,
        commands.sq8_full,
        commands.reasoning_release,
        commands.reasoning_browser,
        *commands.reverse_reconciliation,
        *commands.aq4_reasoning_release,
        *commands.aq4_reasoning_browser,
        *commands.aq4_bundle,
        *commands.final_checks,
    ]
    assert commands.candidate_reconciliation == commands.reverse_reconciliation
    assert commands.candidate_checks == commands.final_checks
    verifier = (
        *PLAN.PYTHON_PREFIX,
        str(source / "tools/verify-openwebui-container-image.py"),
        "--docker",
        PLAN.DOCKER,
    )
    assert commands.candidate_reconciliation[-1] == verifier
    assert commands.candidate_checks[-1] == verifier
    assert commands.reverse_reconciliation[-1] == verifier
    assert commands.final_checks[-1] == verifier
    assert all(command and all(argument for argument in command) for command in flattened)
    for command in flattened:
        if command[0] == PLAN.PYTHON:
            assert command[:4] == PLAN.PYTHON_PREFIX
    for command in (
        commands.sq8_full,
        commands.reasoning_release,
        commands.reasoning_browser,
    ):
        assert "--active-binding-mode" in command
        assert command[command.index("--active-binding-mode") + 1] == "v2"
        assert "--campaign-authorization" in command
        assert "--candidate-served-model-manifest" in command
        assert "--active-served-model-manifest" in command
        assert "--manifest" not in command
    assert commands.sq8_full[4] == str(
        source / "tools/run-sq8-full-openwebui-campaign.py"
    )
    assert commands.reasoning_release[4] == str(
        source / "tools/run-generic-reasoning-release-campaign.py"
    )
    assert commands.reasoning_browser[4] == str(
        source / "tools/run-openwebui-reasoning-browser-smoke.py"
    )
    assert str(PLAN.ACTIVE_MANIFEST) in commands.sq8_full
    assert PLAN.SERVICE_UNIT in commands.reasoning_release
    assert PLAN.SERVICE_UNIT in commands.reasoning_browser
    for command in (
        commands.aq4_reasoning_release[0],
        commands.aq4_reasoning_browser[0],
    ):
        assert "--active-binding-mode" not in command
        assert "--run-id" not in command
        assert "--manifest" in command
        assert "--campaign-authorization" not in command
        assert command[4].startswith(str(tmp_path / "aq4-source"))
    assert "--token-file" in commands.aq4_reasoning_release[0]
    assert "--token-file" in commands.aq4_reasoning_browser[0]
    assert (
        "--openwebui-session-token-file"
        not in commands.aq4_reasoning_browser[0]
    )
    assert commands.aq4_reasoning_browser[0][
        commands.aq4_reasoning_browser[0].index("--model-name") + 1
    ] == "uLLM Qwen3.5 9B AQ4 reasoning"
    assert commands.aq4_reasoning_release[2][4] == str(
        source / "tools/publish-generic-reasoning-validator-report.py"
    )
    evidence_command = commands.aq4_reasoning_release[1]
    assert evidence_command[
        evidence_command.index("--openwebui-image") + 1
    ] == PLAN.authorization.FIXED_OPENWEBUI_IMAGE
    assert commands.aq4_reasoning_browser[1][4] == str(
        source / "tools/publish-generic-reasoning-validator-report.py"
    )
    assert "--bundle-version" in commands.aq4_bundle[0]
    assert commands.aq4_bundle[0][
        commands.aq4_bundle[0].index("--bundle-version") + 1
    ] == "v1"


def test_fixed_plan_rejects_authorized_openwebui_image_mismatch(
    tmp_path: Path,
) -> None:
    value = authorization_document(tmp_path)
    value["aq4_release"]["openwebui_image"] = (
        "fixture/openwebui@sha256:" + "1" * 64
    )
    with pytest.raises(PLAN.PlanError, match="OpenWebUI image differs"):
        PLAN.derive_commands(
            source_root=tmp_path / "source",
            authorization_path=tmp_path / "authorization.json",
            candidate_manifest=tmp_path / "candidate.json",
            authorization_document=value,
        )


@pytest.mark.parametrize(
    "flag",
    (
        "--active-manifest",
        "--systemd-unit",
        "--environment-file",
        "--candidate-reconcile-command-json",
        "--sq8-full-command-json",
        "--require-inactive-service",
    ),
)
def test_production_cli_rejects_arbitrary_path_and_command_flags(
    flag: str,
) -> None:
    with pytest.raises(SystemExit):
        RUNNER.parse_args(
            [
                "--preflight-only",
                "--authorization",
                "/tmp/auth.json",
                "--source-root",
                "/tmp/source",
                "--candidate-manifest",
                "/tmp/candidate.json",
                flag,
                "attacker-controlled",
            ]
        )


def test_campaign_and_recovery_runners_reject_another_source_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}\n", encoding="ascii")
    args = SimpleNamespace(
        source_root=source,
        candidate_manifest=candidate,
        command_timeout_seconds=10.0,
    )
    record = SimpleNamespace()

    with pytest.raises(
        RUNNER.TransactionError,
        match="sealed source root",
    ):
        RUNNER._request(args, record)
    with pytest.raises(
        RECOVERY.recovery.RecoveryError,
        match="sealed source root",
    ):
        RECOVERY._request(args, record)


def test_campaign_preflight_report_exposes_both_source_seals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path = tmp_path / "authorization.json"
    record = SimpleNamespace(
        snapshot=SimpleNamespace(
            path=authorization_path,
            sha256="a" * 64,
        ),
        document={},
    )
    request = SimpleNamespace(inactive_services=("ullm-openai.service",))
    result = SimpleNamespace(
        authorization=record,
        source_commit="b" * 40,
        source_tree="c" * 40,
        source_seal=SimpleNamespace(fingerprint_sha256="d" * 64),
        aq4_source_seal=SimpleNamespace(fingerprint_sha256="e" * 64),
        active=SimpleNamespace(sha256="f" * 64),
        candidate=SimpleNamespace(sha256="1" * 64),
        candidate_summary={"worker": {"binary_sha256": "2" * 64}},
    )
    monkeypatch.setattr(
        RUNNER,
        "parse_args",
        lambda _argv=None: SimpleNamespace(
            authorization=authorization_path,
            source_root=tmp_path,
            candidate_manifest=tmp_path / "candidate.json",
            command_timeout_seconds=10.0,
            preflight_only=True,
            execute=False,
            confirm_authorization_sha256=None,
        ),
    )
    monkeypatch.setattr(
        RUNNER.authorization,
        "load_authorization",
        lambda *_args, **_kwargs: record,
    )
    monkeypatch.setattr(RUNNER, "_request", lambda *_args: request)
    monkeypatch.setattr(RUNNER, "preflight", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(RUNNER, "default_inactive_checker", lambda _value: None)
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(
        RUNNER,
        "_emit",
        lambda value, **_kwargs: emitted.append(value),
    )

    assert RUNNER.main([]) == 0
    assert len(emitted) == 1
    report = emitted[0]
    assert report["source_seal_sha256"] == "d" * 64
    assert report["aq4_source_seal_sha256"] == "e" * 64


def test_campaign_execute_cli_requires_root_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RUNNER,
        "parse_args",
        lambda _argv=None: SimpleNamespace(execute=True),
    )
    monkeypatch.setattr(RUNNER.os, "geteuid", lambda: 1000)
    assert RUNNER.main([]) == 1


def test_recovery_preflight_report_exposes_sq8_source_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path = tmp_path / "authorization.json"
    record = SimpleNamespace(
        snapshot=SimpleNamespace(
            path=authorization_path,
            sha256="a" * 64,
        )
    )
    claim = SimpleNamespace(
        authorization=record,
        snapshot=SimpleNamespace(sha256="b" * 64),
    )
    request = SimpleNamespace()
    pinned = SimpleNamespace(
        claim=claim,
        transaction_preflight=SimpleNamespace(
            source_seal=SimpleNamespace(fingerprint_sha256="c" * 64)
        ),
        active_state="sq8",
        active_before=SimpleNamespace(sha256="d" * 64),
        backup=SimpleNamespace(sha256="e" * 64),
        backup_requires_publication=False,
    )
    monkeypatch.setattr(
        RECOVERY,
        "parse_args",
        lambda _argv=None: SimpleNamespace(
            authorization=authorization_path,
            source_root=tmp_path,
            candidate_manifest=tmp_path / "candidate.json",
            command_timeout_seconds=10.0,
            preflight_only=True,
            execute_recovery=False,
            confirm_authorization_sha256=None,
        ),
    )
    monkeypatch.setattr(
        RECOVERY.authorization,
        "load_claim",
        lambda *_args, **_kwargs: claim,
    )
    monkeypatch.setattr(RECOVERY, "_request", lambda *_args: request)
    monkeypatch.setattr(
        RECOVERY.recovery,
        "preflight_recovery",
        lambda *_args, **_kwargs: pinned,
    )
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(
        RECOVERY,
        "_emit",
        lambda value, **_kwargs: emitted.append(value),
    )

    assert RECOVERY.main([]) == 0
    assert len(emitted) == 1
    report = emitted[0]
    assert report["source_seal_sha256"] == "c" * 64


def test_standalone_claim_cli_is_read_only_disabled(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    authorization = tmp_path / "authorization.json"
    before = set(tmp_path.iterdir())
    assert CLAIM.main(["--authorization", str(authorization)]) == 2
    assert set(tmp_path.iterdir()) == before
    assert "standalone authorization mutation is disabled" in capsys.readouterr().err


def test_issue_cli_requires_and_forwards_source_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document = tmp_path / "reviewed.json"
    output = tmp_path / "authorization.json"
    source = tmp_path / "source"
    source.mkdir()
    document.write_text("{}\n", encoding="ascii")
    with pytest.raises(SystemExit):
        ISSUE.parse_args(
            ["--document", str(document), "--output", str(output)]
        )

    captured: dict[str, object] = {}

    def fake_issue(
        value: dict[str, object],
        destination: Path,
        **kwargs: object,
    ) -> object:
        captured.update(
            {
                "document": value,
                "destination": destination,
                **kwargs,
            }
        )
        return SimpleNamespace(
            document={
                "schema_version": "fixture.authorization.v1",
                "authorization_id": "fixture-001",
            },
            snapshot=SimpleNamespace(
                sha256="a" * 64,
                path=destination,
            ),
        )

    monkeypatch.setattr(ISSUE, "issue_authorization", fake_issue)
    assert (
        ISSUE.main(
            [
                "--document",
                str(document),
                "--output",
                str(output),
                "--source-root",
                str(source),
            ]
        )
        == 0
    )
    assert captured["document"] == {}
    assert captured["destination"] == output
    assert captured["source_root"] == source
    report = json.loads(capsys.readouterr().out)
    assert report["authorization_sha256"] == "a" * 64
    assert report["output"] == str(output)


def test_recovery_cli_has_no_arbitrary_operational_paths_or_commands() -> None:
    parsed = RECOVERY.parse_args(
        [
            "--preflight-only",
            "--authorization",
            "/tmp/auth.json",
            "--source-root",
            "/tmp/source",
            "--candidate-manifest",
            "/tmp/candidate.json",
        ]
    )
    assert parsed.preflight_only is True
    with pytest.raises(SystemExit):
        RECOVERY.parse_args(
            [
                "--preflight-only",
                "--authorization",
                "/tmp/auth.json",
                "--source-root",
                "/tmp/source",
                "--candidate-manifest",
                "/tmp/candidate.json",
                "--active-manifest",
                "/tmp/other.json",
            ]
        )
