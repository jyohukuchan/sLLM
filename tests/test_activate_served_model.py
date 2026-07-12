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


def activation_directory(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "slot"
    shutil.copytree(FIXTURE, root)
    source = root / "served-model.json"
    candidate = root / "candidate.json"
    source.rename(candidate)
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
