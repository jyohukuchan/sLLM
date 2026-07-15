from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/prepare-aq4-p3-profile-operator.py"
SPEC = importlib.util.spec_from_file_location("aq4_p3_profile_operator", SCRIPT)
assert SPEC and SPEC.loader
OPERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = OPERATOR
SPEC.loader.exec_module(OPERATOR)


def sample(timestamp: int, identity: str = "a", *, clean: bool = True) -> dict:
    return {
        "captured_monotonic_ns": timestamp,
        "blocking_identity_sha256": identity,
        "clean": clean,
        "relevant": {"all_required_absent": True},
    }


def sealed(root: Path, name: str, value: dict) -> None:
    OPERATOR.write_sealed(root, name, value)


def test_monitor_requires_one_unchanged_clean_streak_and_confirmation() -> None:
    values = iter([sample(1_000_000_000), sample(2_000_000_000), sample(3_000_000_000)])
    result = OPERATOR.monitor({}, lambda _ready: next(values), lambda _seconds: None, interval=0.0, maximum=1.0, minimum_span=1.0, required=2)
    assert result["passed"] is True
    assert result["resets"] == []
    assert result["span_seconds"] == 1.0
    assert result["confirmation"]["blocking_identity_sha256"] == "a"


def test_monitor_records_identity_reset_even_if_later_streak_passes() -> None:
    values = iter([sample(1_000_000_000, "a"), sample(2_000_000_000, "b"), sample(3_000_000_000, "b"), sample(4_000_000_000, "b")])
    result = OPERATOR.monitor({}, lambda _ready: next(values), lambda _seconds: None, interval=0.0, maximum=1.0, minimum_span=1.0, required=2)
    assert result["passed"] is True
    assert result["resets"] == [{"sample_index": 1, "reason": "blocking_identity_changed"}]


def test_sealed_inventory_rejects_member_tampering(tmp_path: Path) -> None:
    root = tmp_path / "sealed"
    sealed(root, "record.json", {"status": "go"})
    assert OPERATOR.verify_sums(root)["mode"] == "0555"
    os.chmod(root / "record.json", 0o644)
    (root / "record.json").write_text("{}\n", encoding="ascii")
    with pytest.raises(OPERATOR.OperatorError, match="sealed member differs"):
        OPERATOR.verify_sums(root)


def test_seal_existing_preserves_members_and_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    (root / "trace.csv").write_bytes(b"a,b\n1,2\n")
    first = OPERATOR.seal_existing(root)
    second = OPERATOR.seal_existing(root)
    assert first == second
    assert set(first["members"]) == {"trace.csv"}
    assert root.stat().st_mode & 0o777 == 0o555
    assert (root / "trace.csv").stat().st_mode & 0o777 == 0o444


def test_actual_command_is_exactly_one_non_shell_profile_execution() -> None:
    argv = OPERATOR.actual_argv()
    assert argv == [
        "/usr/bin/python3.12",
        str(ROOT / "tools/run-aq4-p2-resident-smoke-maintenance.py"),
        "--mode",
        "execute",
        "--profile-diagnostic",
        "--ready-artifact",
        str(OPERATOR.PROFILE_READY),
        "--evidence-output",
        str(OPERATOR.MAINTENANCE_EVIDENCE),
        "--confirm-one-case",
    ]
    assert len(argv) == 10
    assert argv.count("--confirm-one-case") == 1


def test_prepare_and_validate_operator_self_hash_and_restore_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    quiet_root = tmp_path / "quiet"
    previous_root = tmp_path / "previous"
    output_root = tmp_path / "operator"
    fresh = [tmp_path / f"fresh-{index}" for index in range(9)]
    ready = {"maintenance": {"restore_poll": {"timeout_seconds": 120.0}}}
    quiet = {
        "schema_version": OPERATOR.QUIET_SCHEMA,
        "status": "go",
        "decision": "GO",
        "resets": [],
        "policy": {"required_consecutive_clean_samples": 2, "minimum_sample_span_seconds": 1.0},
        "summary": {"final_streak_samples": 2, "final_streak_span_seconds": 1.0, "confirmation_passed": True, "fresh_outputs_absent": True},
        "read_only": True,
        "actual_executed": False,
        "gpu_command_executed": False,
        "service_touched": False,
        "secret_material_recorded": False,
    }
    sealed(quiet_root, "quiet-window.json", quiet)
    sealed(previous_root, "command-manifest.json", {"schema_version": "historical.v6"})
    monkeypatch.setattr(OPERATOR, "QUIET_ROOT", quiet_root)
    monkeypatch.setattr(OPERATOR, "PREVIOUS_OPERATOR_ROOT", previous_root)
    monkeypatch.setattr(OPERATOR, "ready_authority", lambda: (ready, {"root": "ready-v6"}))
    monkeypatch.setattr(OPERATOR, "fresh_paths", lambda _ready: fresh)

    value = OPERATOR.prepare_operator(output_root)
    validated = OPERATOR.validate_operator(output_root)["value"]
    assert validated == value
    assert value["failure_contract"]["retry_forbidden"] is True
    assert value["failure_contract"]["outer_restore_in_finally"] is True
    assert value["failure_contract"]["restore_timeout_seconds"] == 120.0
    assert value["failure_contract"]["children_remaining_must_be_empty"] is True
    clone = json.loads(json.dumps(value))
    declared = clone["manifest_sha256"]
    clone["manifest_sha256"] = None
    assert declared == OPERATOR.sha_bytes(OPERATOR.canonical(clone))


def test_validate_quiet_rejects_any_reset(tmp_path: Path) -> None:
    root = tmp_path / "quiet"
    value = {
        "schema_version": OPERATOR.QUIET_SCHEMA,
        "status": "go",
        "decision": "GO",
        "resets": [{"reason": "blocking_identity_changed"}],
        "policy": {"required_consecutive_clean_samples": 2, "minimum_sample_span_seconds": 1.0},
        "summary": {"final_streak_samples": 2, "final_streak_span_seconds": 1.0, "confirmation_passed": True, "fresh_outputs_absent": True},
        "read_only": True,
        "actual_executed": False,
        "gpu_command_executed": False,
        "service_touched": False,
        "secret_material_recorded": False,
    }
    sealed(root, "quiet-window.json", value)
    with pytest.raises(OPERATOR.OperatorError, match="decision/safety"):
        OPERATOR.validate_quiet(root)
