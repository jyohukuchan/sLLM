#!/usr/bin/env python3
"""Deterministic negative tests for every required host fail-closed gate."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_host_results import main as aggregate_main  # noqa: E402
from common import (  # noqa: E402
    ContractError,
    COUNT_KEYS,
    DEV_RUST_VERSION,
    HOST_PYTHON_VERSION,
    MSRV_RUST_VERSION,
    ROOT,
    command_content_hash,
    command_hash,
    identity,
    iso_z,
    load_manifests,
    manifest_bundle_hash,
    matrix_manifest_hash,
    registered_row_commands,
    result_report_bytes,
    sha256_bytes,
    sha256_json,
    tuple_digest,
    utc_now,
    validate_result_payload,
)
from network_guard import NetworkIsolationError  # noqa: E402
from run_host_suite import (  # noqa: E402
    actual_counts,
    main as run_host_suite_main,
    run_bounded_process,
    run_command,
    validate_execution_identity,
)
from tracked_tree import main as tracked_tree_main  # noqa: E402


def _test_toolchain() -> dict[str, object]:
    schema = json.loads((ROOT / "ci/schema/test-result-v1.schema.json").read_text(encoding="utf-8"))
    package_schema = schema["properties"]["toolchain"]["properties"]["host_packages"]
    package_names = package_schema.get("required") or ["pytest"]
    return {
        "python": HOST_PYTHON_VERSION,
        "platform": "self-test",
        "system": "Linux",
        "machine": "x86_64",
        "git": "git self-test",
        "rustc_dev": f"rustc {DEV_RUST_VERSION} (self-test)",
        "cargo_dev": f"cargo {DEV_RUST_VERSION} (self-test)",
        "rustc_msrv": f"rustc {MSRV_RUST_VERSION} (self-test)",
        "cargo_msrv": f"cargo {MSRV_RUST_VERSION} (self-test)",
        "clang_format": "clang-format self-test",
        "cmake": "cmake self-test",
        "host_packages": {name: "self-test" for name in package_names},
    }


def result_for(
    row_id: str,
    *,
    state: str = "PASS",
    created_offset: timedelta = timedelta(seconds=-1),
    evidence_mode: str = "local-development",
) -> dict[str, object]:
    suites, host, _ = load_manifests(ROOT)
    row = next(item for item in host["rows"] if item["row_id"] == row_id)
    command_records = registered_row_commands(suites, row, ROOT)
    now = utc_now()
    started = now + created_offset
    finished = started + timedelta(seconds=1)
    address_space_limit = 4 * 1024 * 1024 * 1024 if row_id == "h2" else None
    per_step_counts = {
        "collected": 1,
        "selected": 1,
        "passed": 1 if state == "PASS" else 0,
        "failed": 1 if state == "FAIL" else 0,
        "skipped": 0,
        "deselected": 0,
    }
    steps: list[dict[str, object]] = []
    for command_id, _ in command_records:
        step_resource = {
            "wall_time_limit_seconds": 30.0,
            "timed_out": False,
            "max_rss_bytes": 1,
            "max_rss_limit_bytes": 1024,
            "rss_breach": False,
            "cpu_user_seconds": 0.0,
            "cpu_system_seconds": 0.0,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
            "output_bytes": 0,
            "stdout_captured_bytes": 0,
            "stderr_captured_bytes": 0,
            "captured_output_bytes": 0,
            "output_limit_bytes": 1024,
            "output_breach": False,
            "network_isolated": True,
            "network_guard_strategy": "self-test",
            "address_space_limit_bytes": address_space_limit,
            "address_space_limit_enforced": address_space_limit is not None,
        }
        steps.append({
            "step_id": command_id,
            "state": state,
            "started_at": iso_z(started),
            "finished_at": iso_z(finished),
            "duration_seconds": (finished - started).total_seconds(),
            "exit_code": 0 if state == "PASS" else 1,
            "stdout_sha256": sha256_bytes(b""),
            "stderr_sha256": sha256_bytes(b""),
            "diagnostic": "self-test",
            "selection_required": True,
            "count_source": "validator-command",
            "counts": dict(per_step_counts),
            "resource": step_resource,
        })
    cases = [
        {
            "case_id": step["step_id"],
            **{key: value for key, value in step.items() if key != "step_id"},
        }
        for step in steps
    ]
    selected_counts = {
        key: sum(int(step["counts"][key]) for step in steps)
        for key in COUNT_KEYS
    }
    commit = identity(ROOT)["commit"]
    row_resource = {
        "wall_time_limit_seconds": 30.0,
        "wall_time_breach": False,
        "max_rss_bytes": 1,
        "max_rss_limit_bytes": 1024,
        "rss_breach": False,
        "runner_max_rss_bytes": 1,
        "fixture_size_bytes": 0,
        "fixture_size_limit_bytes": 1024,
        "fixture_size_breach": False,
        "output_bytes": 0,
        "captured_output_bytes": 0,
        "row_output_limit_bytes": 4096,
        "output_breach": False,
        "address_space_limit_bytes": address_space_limit,
        "commands_expected": len(command_records),
        "commands_executed": len(command_records),
        "commands_complete": True,
        "network_isolated": True,
        "network_guard_strategies": ["self-test"],
    }
    command = [argv for _, argv in command_records]
    toolchain = _test_toolchain()
    local_development = evidence_mode == "local-development"
    return {
        "schema_version": "test-result-v1",
        "result_id": f"selftest.{row_id}",
        "suite_id": f"host-{row_id}",
        "tier": row["tier"],
        "state": state,
        "required": True,
        "evidence_mode": evidence_mode,
        "run_id": "selftest",
        "run_attempt": 1,
        "reviewed_sha": commit,
        "tested_sha": commit,
        "workflow_sha": commit,
        "git_tree_oid": identity(ROOT)["tree"],
        "worktree_clean": not local_development,
        "matrix_manifest_sha256": matrix_manifest_hash(ROOT),
        "matrix_row_id": row_id,
        "tuple_digest": tuple_digest(row),
        "command": command,
        "command_sha256": command_hash(command),
        "toolchain": toolchain,
        "toolchain_sha256": sha256_json(toolchain),
        "artifact": {
            "content_sha256": command_content_hash(steps),
            "manifest_sha256": manifest_bundle_hash(ROOT),
        },
        "created_at": iso_z(started),
        "started_at": iso_z(started),
        "finished_at": iso_z(finished),
        "duration_seconds": max(0.0, (finished - started).total_seconds()),
        "seed": row["seed"],
        "counts": selected_counts,
        "resource": row_resource,
        "cases": cases,
        "steps": steps,
        "diagnostic": {
            "message": "self-test",
            "errors": [] if state == "PASS" else [f"self-test state={state}"],
            "warnings": (
                ["LOCAL DEVELOPMENT ONLY: this report is not immutable evidence"]
                if local_development else []
            ),
            "network_disabled": True,
            "model_disabled": True,
            "gpu_fallback_disabled": True,
            "network_guard_self_test": True,
        },
    }


def write_report(directory: Path, payload: dict[str, object], *, bad_hash: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "report.json"
    raw = result_report_bytes(payload)
    path.write_bytes(raw)
    digest = "0" * 64 if bad_hash else sha256_bytes(raw)
    (directory / "report.json.sha256").write_text(f"{digest}  report.json\n", encoding="utf-8")


def expect_contract_failure(payload: dict[str, object], label: str) -> None:
    try:
        validate_result_payload(payload)
    except ContractError:
        return
    raise AssertionError(f"{label} was accepted")


def expect_call_failure(function, label: str) -> None:
    try:
        function()
    except (ContractError, NetworkIsolationError):
        return
    raise AssertionError(f"{label} was accepted")


def needs_file(directory: Path, value: object = "success") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "needs.json"
    path.write_text(json.dumps({row: {"result": value} for row in ("h0", "h1", "h2")}) + "\n", encoding="utf-8")
    return path


def aggregate(directory: Path, needs: Path, output: Path, *extra: str) -> int:
    return aggregate_main([
        "--needs-json", str(needs),
        "--artifact-dir", str(directory),
        "--output-dir", str(output),
        "--run-id", "selftest",
        "--allow-local-development",
        *extra,
    ])


def strict_aggregate(directory: Path, needs: Path, output: Path) -> int:
    commit = identity(ROOT)["commit"]
    return aggregate_main([
        "--needs-json", str(needs),
        "--artifact-dir", str(directory),
        "--output-dir", str(output),
        "--run-id", "selftest",
        "--strict-ci",
        "--reviewed-sha", commit,
        "--tested-sha", commit,
        "--workflow-sha", commit,
    ])


def assert_zero_actual_selection() -> None:
    zero = {key: 0 for key in COUNT_KEYS}
    encoded = json.dumps(zero, sort_keys=True, separators=(",", ":"))
    cases = (
        ([sys.executable, "-m", "pytest", "-m", "tier_h1", "tests"], f"ULLM_PYTEST_COUNTS={encoded}"),
        (["cargo", f"+{DEV_RUST_VERSION}", "test", "--workspace"], "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;"),
        ([sys.executable, "ci/tests/test_h2_oracle.py"], f"ULLM_UNITTEST_COUNTS={encoded}"),
    )
    for command, output in cases:
        counts, warning, source = actual_counts(command, output, 0)
        if counts["selected"] != 0 or not warning or source not in {"pytest-machine", "cargo-harness", "unittest-machine"}:
            raise AssertionError(f"zero actual selection was accepted for {source}: {counts} {warning}")


def assert_identity_and_network_failures(root: Path) -> None:
    head = "a" * 40
    expect_call_failure(
        lambda: validate_execution_identity(
            strict_ci=True,
            allow_dirty_local=False,
            worktree_clean=False,
            head_sha=head,
            reviewed_sha=head,
            tested_sha=head,
            workflow_sha=head,
        ),
        "strict dirty identity",
    )
    expect_call_failure(
        lambda: validate_execution_identity(
            strict_ci=True,
            allow_dirty_local=False,
            worktree_clean=True,
            head_sha=head,
            reviewed_sha=head,
            tested_sha="b" * 40,
            workflow_sha=head,
        ),
        "mismatched execution SHA",
    )
    expect_call_failure(
        lambda: validate_execution_identity(
            strict_ci=False,
            allow_dirty_local=False,
            worktree_clean=False,
            head_sha=head,
            reviewed_sha=head,
            tested_sha=head,
            workflow_sha=head,
        ),
        "unapproved dirty local identity",
    )
    mode = validate_execution_identity(
        strict_ci=False,
        allow_dirty_local=True,
        worktree_clean=False,
        head_sha=head,
        reviewed_sha=head,
        tested_sha=head,
        workflow_sha=head,
    )
    if mode != "local-development":
        raise AssertionError(f"dirty local opt-out selected the wrong evidence mode: {mode}")

    with patch("run_host_suite.prepare_isolation", side_effect=NetworkIsolationError("forced self-test failure")):
        step, _ = run_command(
            "network-failure",
            [sys.executable, "-c", "print('must not run')"],
            timeout_seconds=2,
            repo=root,
            output_dir=root / "network-failure",
            max_rss_bytes=256 * 1024 * 1024,
            output_limit_bytes=1024,
            address_space_limit_bytes=None,
        )
    if step["state"] != "INFRA_ERROR" or step["counts"]["selected"] != 0 or step["resource"]["network_isolated"]:
        raise AssertionError(f"network guard failure was accepted: {step}")

    result = run_bounded_process(
        [sys.executable, "-c", "print('x' * 4096)"],
        repo=root,
        timeout_seconds=5,
        max_rss_bytes=256 * 1024 * 1024,
        output_limit_bytes=32,
    )
    if not result[5]:
        raise AssertionError("command output breach was not observed")


def assert_post_execution_mutation_failure(root: Path) -> None:
    started = utc_now()
    finished = started + timedelta(milliseconds=1)
    step = {
        "step_id": "selftest.mutator",
        "state": "PASS",
        "started_at": iso_z(started),
        "finished_at": iso_z(finished),
        "duration_seconds": 0.001,
        "exit_code": 0,
        "stdout_sha256": sha256_bytes(b""),
        "stderr_sha256": sha256_bytes(b""),
        "diagnostic": "self-test",
        "selection_required": True,
        "count_source": "validator-command",
        "counts": {
            "collected": 1,
            "selected": 1,
            "passed": 1,
            "failed": 0,
            "skipped": 0,
            "deselected": 0,
        },
        "resource": {
            "wall_time_limit_seconds": 30.0,
            "timed_out": False,
            "max_rss_bytes": 1,
            "max_rss_limit_bytes": 1024,
            "rss_breach": False,
            "cpu_user_seconds": 0.0,
            "cpu_system_seconds": 0.0,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
            "output_bytes": 0,
            "stdout_captured_bytes": 0,
            "stderr_captured_bytes": 0,
            "captured_output_bytes": 0,
            "output_limit_bytes": 1024,
            "output_breach": False,
            "network_isolated": True,
            "network_guard_strategy": "self-test",
            "address_space_limit_bytes": None,
            "address_space_limit_enforced": False,
        },
    }
    commit = identity(ROOT)["commit"]
    clean = {"tracked": [], "untracked": []}
    mutations = {
        "tracked": {
            "tracked": [" M ci/matrix/suites-v1.json"],
            "untracked": [],
        },
        "untracked": {
            "tracked": [],
            "untracked": ["ci-mutated.tmp"],
        },
    }
    for label, dirty in mutations.items():
        output_dir = root / f"runner-{label}-mutation"
        with (
            patch("run_host_suite.worktree_status", side_effect=[clean, dirty]),
            patch(
                "run_host_suite.registered_row_commands",
                return_value=[("selftest.mutator", [sys.executable, "self-test"])],
            ),
            patch("run_host_suite.run_command", return_value=(step, "")),
            patch("run_host_suite.fixture_size_bytes", return_value=0),
            patch("run_host_suite.runner_max_rss_bytes", return_value=1),
            patch("run_host_suite.toolchain_snapshot", return_value=_test_toolchain()),
            patch("run_host_suite.validate_required_toolchain"),
        ):
            exit_code = run_host_suite_main([
                "--row", "h0",
                "--output-dir", str(output_dir),
                "--strict-ci",
                "--reviewed-sha", commit,
                "--tested-sha", commit,
                "--workflow-sha", commit,
            ])
        if exit_code != 1:
            raise AssertionError(
                f"post-execution strict {label} mutation returned "
                f"{exit_code}, expected 1"
            )
        payload = json.loads(
            (output_dir / "report.json").read_text(encoding="utf-8")
        )
        validate_result_payload(payload)
        if payload["state"] != "FAIL" or payload["worktree_clean"] is not False:
            raise AssertionError(
                f"post-execution strict {label} mutation emitted immutable PASS evidence"
            )
        if not any(
            "became dirty during command execution" in error
            for error in payload["diagnostic"]["errors"]
        ):
            raise AssertionError(
                f"post-execution {label} mutation diagnostic is missing"
            )


def run() -> None:
    valid = result_for("h0")
    validate_result_payload(valid)
    invalid_schema = copy.deepcopy(valid)
    invalid_schema.pop("diagnostic")
    expect_contract_failure(invalid_schema, "missing schema member")
    invalid_case_key = copy.deepcopy(valid)
    invalid_case_key["cases"] = [{"step_id": "selftest.h0", "state": "PASS"}]
    expect_contract_failure(invalid_case_key, "case step_id mismatch")
    invalid_state = copy.deepcopy(valid)
    invalid_state["state"] = "UNKNOWN"
    expect_contract_failure(invalid_state, "unknown state")
    zero = copy.deepcopy(valid)
    zero["counts"] = {key: 0 for key in COUNT_KEYS}
    zero["cases"] = []
    zero["steps"] = []
    expect_contract_failure(zero, "zero collection")

    duplicate = copy.deepcopy(valid)
    duplicate["steps"].append(copy.deepcopy(duplicate["steps"][0]))
    duplicate["cases"].append(copy.deepcopy(duplicate["cases"][0]))
    duplicate["counts"] = {"collected": 2, "selected": 2, "passed": 2, "failed": 0, "skipped": 0, "deselected": 0}
    duplicate["resource"]["commands_expected"] = 2
    duplicate["resource"]["commands_executed"] = 2
    duplicate["artifact"]["content_sha256"] = command_content_hash(duplicate["steps"])
    expect_contract_failure(duplicate, "duplicate case")
    bad_command_hash = copy.deepcopy(valid)
    bad_command_hash["command_sha256"] = "0" * 64
    expect_contract_failure(bad_command_hash, "command hash mismatch")
    bad_toolchain_hash = copy.deepcopy(valid)
    bad_toolchain_hash["toolchain_sha256"] = "0" * 64
    expect_contract_failure(bad_toolchain_hash, "toolchain hash mismatch")
    bad_content_hash = copy.deepcopy(valid)
    bad_content_hash["artifact"]["content_sha256"] = "0" * 64
    expect_contract_failure(bad_content_hash, "artifact content hash mismatch")
    bad_output = copy.deepcopy(valid)
    bad_output["steps"][0]["resource"]["stdout_bytes"] = 33
    bad_output["steps"][0]["resource"]["output_bytes"] = 33
    expect_contract_failure(bad_output, "unreported output breach")
    bad_rss = copy.deepcopy(valid)
    bad_rss["steps"][0]["resource"]["max_rss_bytes"] = 2048
    expect_contract_failure(bad_rss, "unreported RSS breach")
    bad_fixture = copy.deepcopy(valid)
    bad_fixture["resource"]["fixture_size_bytes"] = 2048
    expect_contract_failure(bad_fixture, "unreported fixture breach")
    bad_row_output = copy.deepcopy(valid)
    bad_row_output["resource"]["output_breach"] = True
    expect_contract_failure(bad_row_output, "row output breach")
    bad_network = copy.deepcopy(valid)
    bad_network["steps"][0]["resource"]["network_isolated"] = False
    expect_contract_failure(bad_network, "network isolation breach")

    assert_zero_actual_selection()

    with tempfile.TemporaryDirectory(prefix="ullm-ci-self-test-") as raw_dir:
        root = Path(raw_dir)
        assert_identity_and_network_failures(root)
        assert_post_execution_mutation_failure(root)
        bad_rust = root / "bad.rs"
        bad_rust.write_text("fn main(){println!(\"format mutation\");}\n", encoding="utf-8")
        if subprocess.run(["rustfmt", "--check", str(bad_rust)], capture_output=True, check=False, text=True).returncode == 0:
            raise AssertionError("intentional formatting mutation was accepted")
        if subprocess.run([sys.executable, "-c", "raise AssertionError('intentional test mutation')"], capture_output=True, check=False, text=True).returncode == 0:
            raise AssertionError("intentional test failure was accepted")

        needs = needs_file(root)
        valid_dir = root / "valid"
        for row_id in ("h0", "h1", "h2"):
            write_report(valid_dir / row_id, result_for(row_id))
        if aggregate(valid_dir, needs, root / "aggregate") != 0:
            raise AssertionError("valid aggregation did not pass")
        for row_id in ("h0", "h1", "h2"):
            if not (root / "aggregate" / "report.json").exists():
                raise AssertionError(f"aggregate report missing after {row_id}")

        if strict_aggregate(valid_dir, needs, root / "strict-local-output") != 3:
            raise AssertionError(
                "local-development PASS reports satisfied strict aggregation"
            )

        strict_valid = root / "strict-valid"
        for row_id in ("h0", "h1", "h2"):
            write_report(
                strict_valid / row_id,
                result_for(row_id, evidence_mode="required-ci"),
            )
        if strict_aggregate(strict_valid, needs, root / "strict-output") != 0:
            raise AssertionError("valid required-ci aggregation did not pass")
        if aggregate(
            strict_valid, needs, root / "local-required-output"
        ) != 3:
            raise AssertionError(
                "required-ci reports satisfied local-development aggregation"
            )

        forged_command = root / "forged-command"
        for row_id in ("h0", "h1", "h2"):
            payload = result_for(row_id)
            if row_id == "h1":
                payload["command"][0].append("--forged")
                payload["command_sha256"] = command_hash(payload["command"])
            write_report(forged_command / row_id, payload)
        if aggregate(
            forged_command, needs, root / "forged-command-output"
        ) != 3:
            raise AssertionError(
                "self-consistent but unregistered command content was accepted"
            )

        forged_step = root / "forged-step"
        for row_id in ("h0", "h1", "h2"):
            payload = result_for(row_id)
            if row_id == "h2":
                payload["steps"][0]["step_id"] = "forged.command-identity"
                payload["cases"][0]["case_id"] = "forged.command-identity"
                payload["artifact"]["content_sha256"] = command_content_hash(
                    payload["steps"]
                )
            write_report(forged_step / row_id, payload)
        if aggregate(forged_step, needs, root / "forged-step-output") != 3:
            raise AssertionError(
                "self-consistent but unregistered command identity was accepted"
            )

        missing = root / "missing"
        write_report(missing / "h0", result_for("h0"))
        write_report(missing / "h1", result_for("h1"))
        if aggregate(missing, needs, root / "missing-output") != 3:
            raise AssertionError("missing row was accepted")

        duplicate = root / "duplicate"
        write_report(duplicate / "h0", result_for("h0"))
        write_report(duplicate / "copy", result_for("h0"))
        write_report(duplicate / "h1", result_for("h1"))
        write_report(duplicate / "h2", result_for("h2"))
        if aggregate(duplicate, needs, root / "duplicate-output") != 3:
            raise AssertionError("duplicate row was accepted")

        unknown = root / "unknown"
        for row_id in ("h0", "h1", "h2"):
            payload = result_for(row_id)
            if row_id == "h2":
                payload["matrix_row_id"] = "h9"
            write_report(unknown / row_id, payload)
        if aggregate(unknown, needs, root / "unknown-output") != 3:
            raise AssertionError("unknown row was accepted")

        stale = root / "stale"
        for row_id in ("h0", "h1", "h2"):
            write_report(stale / row_id, result_for(row_id, created_offset=timedelta(days=-2)))
        if aggregate(stale, needs, root / "stale-output") != 3:
            raise AssertionError("stale result was accepted")

        bad_schema = root / "bad-schema"
        payload = result_for("h0")
        payload.pop("diagnostic")
        write_report(bad_schema / "h0", payload)
        if aggregate(bad_schema, needs, root / "bad-schema-output") != 3:
            raise AssertionError("invalid schema was accepted")

        mismatch = root / "hash-mismatch"
        for row_id in ("h0", "h1", "h2"):
            write_report(mismatch / row_id, result_for(row_id), bad_hash=row_id == "h1")
        if aggregate(mismatch, needs, root / "hash-output") != 3:
            raise AssertionError("hash mismatch was accepted")

        missing_sidecar = root / "missing-sidecar"
        for row_id in ("h0", "h1", "h2"):
            write_report(missing_sidecar / row_id, result_for(row_id))
        (missing_sidecar / "h1/report.json.sha256").unlink()
        if aggregate(missing_sidecar, needs, root / "missing-sidecar-output") != 3:
            raise AssertionError("missing sidecar was accepted")

        wrong_attempt = root / "wrong-attempt"
        for row_id in ("h0", "h1", "h2"):
            payload = result_for(row_id)
            if row_id == "h0":
                payload["run_attempt"] = 2
            write_report(wrong_attempt / row_id, payload)
        if aggregate(wrong_attempt, needs, root / "attempt-output") != 3:
            raise AssertionError("wrong run attempt was accepted")

        mismatched_sha = root / "mismatched-sha"
        for row_id in ("h0", "h1", "h2"):
            payload = result_for(row_id)
            if row_id == "h0":
                payload["tested_sha"] = "0" * 40
            write_report(mismatched_sha / row_id, payload)
        if aggregate(mismatched_sha, needs, root / "mismatched-sha-output") != 3:
            raise AssertionError("mismatched SHA was accepted")

        manifest_mismatch = root / "manifest-mismatch"
        for row_id in ("h0", "h1", "h2"):
            payload = result_for(row_id)
            if row_id == "h0":
                payload["artifact"]["manifest_sha256"] = "0" * 64
            write_report(manifest_mismatch / row_id, payload)
        if aggregate(manifest_mismatch, needs, root / "manifest-mismatch-output") != 3:
            raise AssertionError("manifest hash mismatch was accepted")

        nonpass = root / "nonpass"
        write_report(nonpass / "h0", result_for("h0", state="FAIL"))
        write_report(nonpass / "h1", result_for("h1"))
        write_report(nonpass / "h2", result_for("h2"))
        if aggregate(nonpass, needs, root / "nonpass-output") != 1:
            raise AssertionError("non-PASS result was accepted")

        non_success_needs = needs_file(root / "non-success", "failure")
        if aggregate(valid_dir, non_success_needs, root / "needs-output") != 1:
            raise AssertionError("non-success needs was accepted")
        missing_needs = root / "missing-needs.json"
        missing_needs.write_text(json.dumps({"h0": {"result": "success"}}), encoding="utf-8")
        if aggregate(valid_dir, missing_needs, root / "missing-needs-output") != 3:
            raise AssertionError("missing needs row was accepted")

        git_root = root / "git"
        (git_root / "ci/policy").mkdir(parents=True)
        (git_root / "ci/policy/hygiene-allowlist-v1.json").write_text('{"schema_version":"hygiene-allowlist-v1","entries":[]}\n', encoding="utf-8")
        (git_root / "ci/schema").mkdir(parents=True)
        (git_root / "ci/schema/hygiene-allowlist-v1.schema.json").write_text((ROOT / "ci/schema/hygiene-allowlist-v1.schema.json").read_text(encoding="utf-8"), encoding="utf-8")
        (git_root / ".local-artifacts").mkdir()
        (git_root / ".local-artifacts/raw.bin").write_bytes(b"prohibited")
        subprocess.run(["git", "init", "-q"], cwd=git_root, check=True)
        subprocess.run(["git", "config", "user.email", "ci-self-test@example.invalid"], cwd=git_root, check=True)
        subprocess.run(["git", "config", "user.name", "CI self-test"], cwd=git_root, check=True)
        subprocess.run(["git", "add", "."], cwd=git_root, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=git_root, check=True)
        if tracked_tree_main(["--repo", str(git_root)]) == 0:
            raise AssertionError("prohibited tracked path was accepted")


def main() -> int:
    argparse.ArgumentParser().parse_args()
    try:
        run()
    except (AssertionError, ContractError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"fail-closed self-test: FAIL: {exc}", file=sys.stderr)
        return 1
    print("fail-closed self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
