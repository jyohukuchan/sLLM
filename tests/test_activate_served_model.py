from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/activate-served-model.py"
FIXTURE = ROOT / "services/openai-gateway/tests/fixtures/served-model/aq4"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ACTIVATOR = load_module("test_activate_served_model_tool", TOOL_PATH)
BUNDLE_FIXTURES = load_module(
    "test_activate_served_model_bundle_fixtures",
    ROOT / "tests/test_validate_generic_reasoning_release_bundle.py",
)


def activation_directory(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "slot"
    shutil.copytree(FIXTURE, root)
    source = root / "served-model.json"
    candidate = root / "candidate.json"
    source.rename(candidate)
    return root, candidate


def v2_activation_directory(tmp_path: Path) -> tuple[Path, Path]:
    root, candidate = activation_directory(tmp_path)
    document = json.loads(candidate.read_text(encoding="ascii"))
    document["schema_version"] = "ullm.served_model.v2"
    document["worker"]["protocol"] = "ullm.worker.v2"
    document["promotion"]["source_commit"] = "1" * 40
    document["reasoning"] = {
        "enabled_by_default": False,
        "dialect_id": "synthetic.multi-token.v1",
        "start_token_ids": [10, 11],
        "end_token_ids": [20, 21],
        "forced_end_token_ids": [20, 21],
        "initial_phase": "reasoning",
        "eos_policy": "close",
        "effort_budgets": {"low": 2, "medium": 4, "high": 8},
        "max_budget_tokens": 8,
        "reserved_answer_tokens": 1,
        "history_reasoning_policy": "omit",
    }
    candidate.write_text(json.dumps(document), encoding="ascii")
    return root, candidate


def command(script: str, *arguments: Path | str) -> tuple[str, ...]:
    return (sys.executable, "-c", script, *(os.fspath(value) for value in arguments))


def test_activation_is_atomic_and_runs_commands_in_stage_order(tmp_path: Path) -> None:
    root, candidate = activation_directory(tmp_path)
    active = root / "active.json"
    active.write_bytes(b"old-active-manifest")
    events = root / "events"
    append = "from pathlib import Path; import os,sys; Path(sys.argv[1]).open('a').write(os.environ['ULLM_ACTIVATION_STAGE']+'\\n')"

    result = ACTIVATOR.activate(
        candidate,
        active,
        check_commands=[command(append, events)],
        reconcile_commands=[command(append, events)],
        final_check_commands=[command(append, events)],
    )

    assert active.read_bytes() == candidate.read_bytes()
    assert result.manifest_sha256 == hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert result.model_id == "ullm-qwen3.5-9b-aq4"
    assert events.read_text(encoding="utf-8").splitlines() == [
        "check",
        "reconcile",
        "final-check",
    ]
    assert not list(root.glob(".served-model.candidate.*"))
    assert not list(root.glob(".served-model.rollback.*"))


def test_v2_activation_requires_a_release_bundle(tmp_path: Path) -> None:
    root, candidate = v2_activation_directory(tmp_path)
    active = root / "active.json"
    active.write_bytes(b"known-old-active")

    with pytest.raises(ACTIVATOR.ActivationError, match="release bundle"):
        ACTIVATOR.activate(candidate, active)

    assert active.read_bytes() == b"known-old-active"


def test_v2_bootstrap_requires_explicit_inactive_services_and_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, candidate = v2_activation_directory(tmp_path)
    active = root / "active.json"
    active.write_bytes((FIXTURE / "served-model.json").read_bytes())
    unit = root / "ullm-openai.service"
    environment = root / "ullm-openai.env"
    unit.write_bytes(b"[Service]\nExecStart=/usr/bin/ullm\n")
    environment.write_bytes(b"ULLM_TEST=1\n")
    backup = root / "previous-active.json"

    with pytest.raises(ACTIVATOR.ActivationError, match="inactive services"):
        ACTIVATOR.activate(
            candidate,
            active,
            bootstrap_v2=True,
            bootstrap_backup=backup,
            systemd_unit=unit,
            environment_file=environment,
        )
    assert active.read_bytes() == (FIXTURE / "served-model.json").read_bytes()
    assert not backup.exists()

    monkeypatch.setattr(ACTIVATOR, "_require_inactive_services", lambda _services: None)
    result = ACTIVATOR.activate(
        candidate,
        active,
        bootstrap_v2=True,
        bootstrap_backup=backup,
        systemd_unit=unit,
        environment_file=environment,
        require_inactive_services=("ullm-openai.service", "llama-qwen35-udq4.service"),
    )

    assert result.manifest_sha256 == hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert active.read_bytes() == candidate.read_bytes()
    assert backup.read_bytes() == (FIXTURE / "served-model.json").read_bytes()


def test_v2_activation_binds_bundle_and_rollback_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, candidate = v2_activation_directory(tmp_path)
    active = root / "active.json"
    old = b"known-old-active"
    active.write_bytes(old)
    unit = root / "ullm-openai.service"
    environment = root / "ullm-openai.env"
    unit.write_bytes(b"[Service]\nExecStart=/usr/bin/ullm\n")
    environment.write_bytes(b"ULLM_TEST=1\n")
    bundle = root / "release-bundle.json"
    bundle.write_text("{}", encoding="ascii")

    summary = ACTIVATOR.load_validator().validation_summary(candidate)
    monkeypatch.setattr(
        ACTIVATOR,
        "load_bundle_validator",
        lambda: type(
            "BundleValidator",
            (),
            {"validate": lambda _self, _path: {"gate_eligible": True}},
        )(),
    )
    bundle.write_text(
        json.dumps(
            {
                "source_commit": "1" * 40,
                "identity": {
                    "manifest_sha256": summary["manifest_sha256"],
                    "worker_binary_sha256": summary["worker"]["binary_sha256"],
                },
                "rollback_target": {
                    "manifest_sha256": hashlib.sha256(old).hexdigest(),
                    "systemd_unit_sha256": hashlib.sha256(unit.read_bytes()).hexdigest(),
                    "environment_sha256": hashlib.sha256(environment.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="ascii",
    )

    result = ACTIVATOR.activate(
        candidate,
        active,
        release_bundle=bundle,
        systemd_unit=unit,
        environment_file=environment,
    )

    assert result.manifest_sha256 == hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert active.read_bytes() == candidate.read_bytes()


def test_v2_activation_rejects_rollback_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, candidate = v2_activation_directory(tmp_path)
    active = root / "active.json"
    active.write_bytes(b"known-old-active")
    unit = root / "ullm-openai.service"
    environment = root / "ullm-openai.env"
    unit.write_bytes(b"unit")
    environment.write_bytes(b"environment")
    bundle = root / "release-bundle.json"
    summary = ACTIVATOR.load_validator().validation_summary(candidate)
    monkeypatch.setattr(
        ACTIVATOR,
        "load_bundle_validator",
        lambda: type(
            "BundleValidator",
            (),
            {"validate": lambda _self, _path: {"gate_eligible": True}},
        )(),
    )
    bundle.write_text(
        json.dumps(
            {
                "source_commit": "1" * 40,
                "identity": {
                    "manifest_sha256": summary["manifest_sha256"],
                    "worker_binary_sha256": summary["worker"]["binary_sha256"],
                },
                "rollback_target": {
                    "manifest_sha256": "f" * 64,
                    "systemd_unit_sha256": hashlib.sha256(unit.read_bytes()).hexdigest(),
                    "environment_sha256": hashlib.sha256(environment.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="ascii",
    )

    with pytest.raises(ACTIVATOR.ActivationError, match="previous active manifest"):
        ACTIVATOR.activate(
            candidate,
            active,
            release_bundle=bundle,
            systemd_unit=unit,
            environment_file=environment,
        )

    assert active.read_bytes() == b"known-old-active"


def test_v2_activation_accepts_real_complete_bundle_binding(tmp_path: Path) -> None:
    root, candidate = v2_activation_directory(tmp_path)
    active = root / "active.json"
    old = b"known-old-active"
    active.write_bytes(old)
    unit = root / "ullm-openai.service"
    environment = root / "ullm-openai.env"
    unit.write_bytes(b"[Service]\nExecStart=/usr/bin/ullm\n")
    environment.write_bytes(b"ULLM_TEST=1\n")

    bundle = BUNDLE_FIXTURES.make_bundle(root)
    summary = ACTIVATOR.load_validator().validation_summary(candidate)
    release_path = root / "release.json"
    release = json.loads(release_path.read_text(encoding="ascii"))
    release["identity"]["manifest_sha256"] = summary["manifest_sha256"]
    release["identity"]["worker_binary_sha256"] = summary["worker"]["binary_sha256"]
    release["source_commit"] = "1" * 40
    release["active_promotion_source_commit"] = "1" * 40
    release["source_commit_aligned"] = True
    BUNDLE_FIXTURES.write_json(release_path, release)
    release_report = BUNDLE_FIXTURES.RELEASE_FIXTURE.TOOL.validate(release_path)
    release_report_path = root / "release-validator.json"
    BUNDLE_FIXTURES.write_json(release_report_path, release_report)

    promotion_path = root / "promotion-evidence.json"
    promotion = json.loads(promotion_path.read_text(encoding="ascii"))
    promotion["source_commit"] = "1" * 40
    promotion["worker_binary_sha256"] = summary["worker"]["binary_sha256"]
    promotion["ephemeral_bundle"]["manifest_sha256"] = summary["manifest_sha256"]
    BUNDLE_FIXTURES.write_json(promotion_path, promotion)
    receipt_path = root / "promotion-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    receipt["source_commit"] = "1" * 40
    receipt["evidence"]["sha256"] = BUNDLE_FIXTURES.digest(promotion_path)
    BUNDLE_FIXTURES.write_json(receipt_path, receipt)

    bundle_document = json.loads(bundle.read_text(encoding="ascii"))
    bundle_document["source_commit"] = "1" * 40
    bundle_document["active_promotion_source_commit"] = "1" * 40
    bundle_document["identity"] = release["identity"]
    bundle_document["rollback_target"] = {
        "manifest_sha256": hashlib.sha256(old).hexdigest(),
        "systemd_unit_sha256": hashlib.sha256(unit.read_bytes()).hexdigest(),
        "environment_sha256": hashlib.sha256(environment.read_bytes()).hexdigest(),
    }
    for name, path in (
        ("release_evidence", release_path),
        ("release_validator", release_report_path),
        ("promotion_evidence", promotion_path),
        ("promotion_receipt", receipt_path),
    ):
        bundle_document["artifacts"][name]["sha256"] = BUNDLE_FIXTURES.digest(path)
    BUNDLE_FIXTURES.write_json(bundle, bundle_document)

    result = ACTIVATOR.activate(
        candidate,
        active,
        release_bundle=bundle,
        systemd_unit=unit,
        environment_file=environment,
    )

    assert result.model_id == "ullm-qwen3.5-9b-aq4"
    assert active.read_bytes() == candidate.read_bytes()


@pytest.mark.parametrize("failed_stage", ["check", "reconcile", "final-check"])
def test_command_failure_restores_old_active_manifest(
    tmp_path: Path, failed_stage: str
) -> None:
    root, candidate = activation_directory(tmp_path)
    active = root / "active.json"
    old = b"known-old-active"
    active.write_bytes(old)
    succeed = command("raise SystemExit(0)")
    fail = command("raise SystemExit(19)")

    with pytest.raises(ACTIVATOR.ActivationError):
        ACTIVATOR.activate(
            candidate,
            active,
            check_commands=[fail if failed_stage == "check" else succeed],
            reconcile_commands=[fail if failed_stage == "reconcile" else succeed],
            final_check_commands=[
                fail if failed_stage == "final-check" else succeed
            ],
        )

    assert active.read_bytes() == old
    assert not list(root.glob(".served-model.candidate.*"))
    assert not list(root.glob(".served-model.rollback.*"))


def test_failure_during_first_activation_removes_new_active(tmp_path: Path) -> None:
    root, candidate = activation_directory(tmp_path)
    active = root / "active.json"

    with pytest.raises(ACTIVATOR.ActivationError):
        ACTIVATOR.activate(
            candidate,
            active,
            check_commands=[command("raise SystemExit(1)")],
        )

    assert not active.exists()


def test_failure_after_reconcile_runs_rollback_hook_after_manifest_restore(
    tmp_path: Path,
) -> None:
    root, candidate = activation_directory(tmp_path)
    active = root / "active.json"
    old = b"known-old-active"
    active.write_bytes(old)
    database_state = root / "database-state"
    database_state.write_text("old", encoding="utf-8")
    set_state = (
        "from pathlib import Path; import sys; "
        "Path(sys.argv[1]).write_text(sys.argv[2], encoding='utf-8')"
    )
    verify_restored = (
        "from pathlib import Path; import sys; "
        "assert Path(sys.argv[1]).read_bytes() == b'known-old-active'; "
        "Path(sys.argv[2]).write_text('old', encoding='utf-8')"
    )

    with pytest.raises(ACTIVATOR.ActivationError):
        ACTIVATOR.activate(
            candidate,
            active,
            reconcile_commands=[command(set_state, database_state, "new")],
            final_check_commands=[command("raise SystemExit(1)")],
            rollback_commands=[command(verify_restored, active, database_state)],
        )

    assert active.read_bytes() == old
    assert database_state.read_text(encoding="utf-8") == "old"


def test_preflight_failure_does_not_switch_or_run_commands(tmp_path: Path) -> None:
    root, candidate = activation_directory(tmp_path)
    active = root / "active.json"
    old = b"known-old-active"
    active.write_bytes(old)
    marker = root / "must-not-exist"
    document = json.loads(candidate.read_text(encoding="utf-8"))
    document["unexpected"] = "tampered"
    candidate.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ACTIVATOR.ActivationError, match="preflight"):
        ACTIVATOR.activate(
            candidate,
            active,
            check_commands=[command("from pathlib import Path; import sys; Path(sys.argv[1]).touch()", marker)],
        )

    assert active.read_bytes() == old
    assert not marker.exists()


def test_atomic_replace_failure_keeps_old_active_and_cleans_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, candidate = activation_directory(tmp_path)
    active = root / "active.json"
    old = b"known-old-active"
    active.write_bytes(old)

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(ACTIVATOR.os, "replace", fail_replace)
    with pytest.raises(ACTIVATOR.ActivationError):
        ACTIVATOR.activate(candidate, active)

    assert active.read_bytes() == old
    assert not list(root.glob(".served-model.candidate.*"))
    assert not list(root.glob(".served-model.rollback.*"))


def test_symlink_active_and_world_writable_candidate_are_rejected(
    tmp_path: Path,
) -> None:
    root, candidate = activation_directory(tmp_path)
    target = root / "target.json"
    target.write_bytes(b"old")
    active = root / "active.json"
    active.symlink_to(target)

    with pytest.raises(ACTIVATOR.ActivationError, match="symlink"):
        ACTIVATOR.activate(candidate, active)

    active.unlink()
    candidate.chmod(candidate.stat().st_mode | stat.S_IWOTH)
    with pytest.raises(ACTIVATOR.ActivationError, match="unsafe"):
        ACTIVATOR.activate(candidate, active)


def test_cli_discards_command_output_and_reports_fixed_failure(tmp_path: Path) -> None:
    root, candidate = activation_directory(tmp_path)
    active = root / "active.json"
    old = b"known-old-active"
    active.write_bytes(old)
    secret = "command-output-must-not-leak"
    failing_command = json.dumps(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1]); print(sys.argv[1], file=sys.stderr); raise SystemExit(1)",
            secret,
        ]
    )

    result = subprocess.run(
        [
            sys.executable,
            TOOL_PATH,
            "--candidate",
            candidate,
            "--active-manifest",
            active,
            "--check-command-json",
            failing_command,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "served-model activation failed\n"
    assert secret not in result.stdout + result.stderr
    assert active.read_bytes() == old
