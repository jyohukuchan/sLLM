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
    return {
        "source": {"commit": "a" * 40, "tree": "b" * 40},
        "candidate": {
            "manifest_sha256": "c" * 64,
            "worker_binary_sha256": "d" * 64,
        },
        "campaigns": {
            name: {
                "run_id": f"{name}-run",
                "final_path": str(outputs / name),
            }
            for name in ("sq8_full", "reasoning_release", "reasoning_browser")
        },
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
        *commands.final_checks,
    ]
    assert commands.candidate_reconciliation == commands.reverse_reconciliation
    assert commands.candidate_checks == commands.final_checks
    assert all(command and all(argument for argument in command) for command in flattened)
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
    assert commands.sq8_full[1] == str(
        source / "tools/run-sq8-full-openwebui-campaign.py"
    )
    assert commands.reasoning_release[1] == str(
        source / "tools/run-generic-reasoning-release-campaign.py"
    )
    assert commands.reasoning_browser[1] == str(
        source / "tools/run-openwebui-reasoning-browser-smoke.py"
    )
    assert str(PLAN.ACTIVE_MANIFEST) in commands.sq8_full
    assert PLAN.SERVICE_UNIT in commands.reasoning_release
    assert PLAN.SERVICE_UNIT in commands.reasoning_browser


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
