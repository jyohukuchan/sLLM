from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aq4_p2_holdout_runner", ROOT / "tools" / "run-aq4-p2-fidelity-holdout.py"
)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_actual_verified_receipt_requires_exact_status(tmp_path: Path) -> None:
    receipt = tmp_path / "actual.json"
    receipt.write_text('{"status":"prepared_not_executed"}\n', encoding="ascii")
    with pytest.raises(RUNNER.HoldoutError, match="status"):
        RUNNER._actual_verified(receipt)


def test_failure_receipt_is_create_new_and_immutable(tmp_path: Path) -> None:
    plan = {
        "split_manifest_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "calibration_cases_sha256": "c" * 64,
        "holdout_cases_sha256": "d" * 64,
        "freeze_receipt_sha256": "e" * 64,
        "actual_verified_receipt": {"path": str(tmp_path / "actual.json"), "sha256": "f" * 64},
        "identity": {"device_architecture": "gfx1201"},
    }
    output = tmp_path / "failure.json"
    value = RUNNER._failure(output, plan, "1" * 64, "nonzero", "fixture failure", 1)
    assert value["immutable"] is True
    assert value["failure_kind"] == "nonzero"
    with pytest.raises(RUNNER.HoldoutError, match="overwrite"):
        RUNNER._failure(output, plan, "1" * 64, "timeout", "retry")


def test_execute_refuses_existing_attempt_marker(tmp_path: Path) -> None:
    preflight = tmp_path / "preflight.json"
    marker = tmp_path / "attempt.json"
    marker.write_text('{"status":"started"}\n', encoding="ascii")
    plan = {
        "schema_version": RUNNER.PREFLIGHT_SCHEMA,
        "status": "ready_for_execute",
        "subset": "holdout",
        "row_count": 24,
        "paths": {"attempt_marker": str(marker), "active_output": str(tmp_path / "active"), "split_root": str(tmp_path), "source_artifact": str(tmp_path)},
        "execution_contract": {"command": ["false"], "command_sha256": "0" * 64, "timeout_seconds": 1},
        "split_manifest_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "calibration_cases_sha256": "c" * 64,
        "holdout_cases_sha256": "d" * 64,
        "freeze_receipt_sha256": "e" * 64,
        "actual_verified_receipt": {"path": str(tmp_path / "actual.json"), "sha256": "f" * 64},
        "identity": {},
        "freeze_receipt_path": str(tmp_path / "freeze.json"),
    }
    preflight.write_text(json.dumps(plan) + "\n", encoding="ascii")
    with pytest.raises(RUNNER.HoldoutError, match="retry"):
        RUNNER._execute(type("Args", (), {"preflight": preflight, "receipt_output": tmp_path / "result.json", "device_index": 0})())


def test_decision_only_consumes_frozen_bounds() -> None:
    receipt = {"derived_bounds": {name: {"bound": 0.5, "direction": spec["direction"]} for name, spec in RUNNER.PROTOCOL.METRICS.items()}}
    means = {name: (0.75 if spec["direction"] == "higher" else 0.25) for name, spec in RUNNER.PROTOCOL.METRICS.items()}
    decision, checks = RUNNER._decision(means, receipt)
    assert decision == "go"
    assert all(item["pass"] for item in checks.values())


def test_atomic_receipts_are_single_link_read_only_files(tmp_path: Path) -> None:
    output = tmp_path / "sealed.json"
    RUNNER._atomic_json(output, {"status": "sealed"}, "receipt")
    info = output.stat()
    assert info.st_nlink == 1
    assert info.st_mode & 0o777 == 0o444
    link = tmp_path / "link.json"
    link.symlink_to(output)
    with pytest.raises(RUNNER.HoldoutError, match="overwrite"):
        RUNNER._atomic_json(link, {"status": "tampered"}, "receipt")


def test_runtime_identity_requires_frozen_single_gpu_contract() -> None:
    expected = {
        "served_model_manifest_sha256": "a" * 64,
        "package_manifest_sha256": "b" * 64,
        "worker_binary_sha256": "c" * 64,
        "selected_cases_sha256": "d" * 64,
        "split_manifest_sha256": "e" * 64,
        "policy_sha256": "f" * 64,
        "holdout_cases_sha256": "d" * 64,
        "quantized_artifact_revision": "rev",
        "device_architecture": "gfx1201",
        "build_sha256": "1" * 64,
        "source_identity": {"model_revision": "src", "tokenizer": {"aggregate_sha256": "2" * 64}, "source_checkpoint": {"aggregate_sha256": "3" * 64}},
    }
    runtime_identity = {key: value for key, value in expected.items() if key in {"served_model_manifest_sha256", "package_manifest_sha256", "worker_binary_sha256", "selected_cases_sha256", "split_manifest_sha256", "policy_sha256", "holdout_cases_sha256", "quantized_artifact_revision", "build_sha256"}}
    runtime_identity.update({"upstream_model_revision": "src", "tokenizer_aggregate_sha256": "2" * 64, "source_checkpoint_aggregate_sha256": "3" * 64, "device": {"architecture": "gfx1201"}, "one_process": True, "one_model_load": True, "gpu_parallelism": 2})
    active = {"manifest": {"runtime": {"runtime": runtime_identity, "model_loads": 1}}}
    with pytest.raises(RUNNER.HoldoutError, match="GPU-parallelism"):
        RUNNER._runtime_identity(active, expected)
