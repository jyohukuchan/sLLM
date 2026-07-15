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


def unsealed_json(root: Path, name: str, value: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(OPERATOR.pretty(value))


def finalizer_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int,
    pre_stop_noop: bool = False,
    pre_stop_case: str | None = None,
) -> dict:
    paths = {
        "ROOT": tmp_path,
        "PROFILE_READY_ROOT": tmp_path / "profile-ready-v16",
        "PROFILE_READY": tmp_path / "profile-ready-v16/ready-binding.json",
        "QUIET_ROOT": tmp_path / "quiet-v18",
        "OPERATOR_ROOT": tmp_path / "operator-command-v13",
        "MAINTENANCE_EVIDENCE": tmp_path / "maintenance-v11",
        "OPERATOR_RESULT": tmp_path / "operator-result-v13",
        "ACTUAL_AUDIT": tmp_path / "actual-audit-v13",
        "PROFILE_RUNTIME": tmp_path / "runtime-v10",
        "PROFILE_EXECUTE_EVIDENCE": tmp_path / "execute-evidence-v10",
        "PROFILE_CAPTURE": tmp_path / "capture-v10",
    }
    for name, path in paths.items():
        monkeypatch.setattr(OPERATOR, name, path)
    succeeded = returncode == 0
    status = "passed" if succeeded else "failed"
    manifest = {
        "inputs": {"profile_ready": {"artifact_commit": "a" * 40}},
        "manifest_sha256": "b" * 64,
        "command_sha256": "c" * 64,
        "argv": ["/usr/bin/python3.12", "maintenance.py"],
        "fresh_outputs": [{"path": f"fresh-{index}", "absent": True} for index in range(9)],
    }
    unsealed_json(paths["OPERATOR_ROOT"], "command-manifest.json", manifest)
    unsealed_json(paths["PROFILE_READY_ROOT"], "ready-binding.json", {})
    if pre_stop_noop:
        assert not succeeded
        maintenance = {
            "status": "failed",
            "mode": "execute",
            "failure": {
                "stage": "pre-stop-snapshot",
                "reason": "restored worker does not uniquely own target GPU",
                "launcher_started": False,
            },
            "package_integrity": {"full_hash_count": 1, "full_content": {"passed": True}, "integrity_identity": {"passed": True}},
            "restore": {"attempted": False, "error": None, "passed": True, "post_start": None},
            "lock_substrate_cleanup": None,
            "capture": None,
            "launcher": None,
            "pre_stop": None,
            "stopped_gates": None,
            "stopped_gate_poll": None,
            "lock_substrate": None,
            "sequence": ["sudo-prevalidate"],
            "process_counts": {
                "sudo": 1,
                "sudo_keepalive": 0,
                "systemctl_stop": 0,
                "launcher": 0,
                "systemctl_start": 0,
                "capture_tool": 0,
                "rocprof": 0,
                "docker": 0,
                "docker_exec": 0,
                "container_curl": 0,
                "container_curl_total": 0,
                "container_curl_version": 0,
                "container_curl_endpoint": 0,
                "stopped_gate_polls": 0,
                "stopped_gate_probe_commands": 0,
            },
            "safety": {
                "service_touched": False,
                "service_stopped": False,
                "gpu_command_executed": False,
                "model_load_executed": False,
            },
            "secret_material_recorded": False,
        }
        if pre_stop_case == "restore_false":
            maintenance["restore"]["passed"] = False
        elif pre_stop_case == "restore_unknown":
            del maintenance["restore"]["passed"]
        elif pre_stop_case == "later_stage":
            maintenance["failure"]["stage"] = "service-stop"
        elif pre_stop_case == "process_started":
            maintenance["process_counts"]["launcher"] = 1
        elif pre_stop_case == "service_stop_started":
            maintenance["process_counts"]["systemctl_stop"] = 1
        elif pre_stop_case == "capture_started":
            maintenance["process_counts"]["capture_tool"] = 1
        elif pre_stop_case == "rocprof_started":
            maintenance["process_counts"]["rocprof"] = 1
        elif pre_stop_case == "service_touched":
            maintenance["safety"]["service_touched"] = True
        elif pre_stop_case == "cleanup_present":
            maintenance["lock_substrate_cleanup"] = {"passed": True, "runner_children": [], "holder_pids": []}
    else:
        maintenance = {
            "status": status,
            "mode": "execute",
            "failure": None if succeeded else {"stage": "profile-capture", "reason": "capture failed", "launcher_started": True},
            "package_integrity": {"full_hash_count": 1, "full_content": {"passed": True}, "integrity_identity": {"passed": True}},
            "restore": {"attempted": True, "passed": True, "duration_ns": 1, "final_metadata_recheck": {"within_absolute_deadline": True}},
            "lock_substrate_cleanup": {"passed": True, "runner_children": [], "holder_pids": []},
            "capture": {"exit_code": returncode, "capture_tool_invocations": 1, "rocprof_invocations": 1},
            "process_counts": {"capture_tool": 1, "launcher": 1, "rocprof": 1},
            "secret_material_recorded": False,
        }
    sealed(paths["MAINTENANCE_EVIDENCE"], "launcher-evidence.json", maintenance)
    launcher = {
        "status": status,
        "runner": {"exit_code": returncode, "stdout": {"file": "runner.stdout.bin"}, "stderr": {"file": "runner.stderr.bin"}},
        "validator": {"exit_code": 0, "stdout": {"file": "validator.stdout.bin"}, "stderr": {"file": "validator.stderr.bin"}},
        "failure": None if succeeded else {"stage": "runner", "reason": "capture failed", "children_remaining": [], "cleanup_passed": True},
    }
    if not pre_stop_noop:
        unsealed_json(paths["PROFILE_EXECUTE_EVIDENCE"], "launcher-evidence.json", launcher)
        for name in ("runner.stdout.bin", "runner.stderr.bin", "validator.stdout.bin", "validator.stderr.bin"):
            (paths["PROFILE_EXECUTE_EVIDENCE"] / name).write_bytes(b"")
        OPERATOR.seal_existing(paths["PROFILE_EXECUTE_EVIDENCE"])
        unsealed_json(paths["PROFILE_RUNTIME"], "resident-batch.summary.json", {"status": "complete", "resident_model_loads": 1})
        unsealed_json(paths["PROFILE_RUNTIME"], "resident-batch.driver-process.json", {"cleanup": {"passed": True}})
        if succeeded:
            unsealed_json(paths["PROFILE_CAPTURE"], "capture-artifact.json", {"status": "complete_diagnostic", "measurement_eligible": False, "promotion_eligible": False})
            nested = paths["PROFILE_CAPTURE"] / "measured-runs"
            nested.mkdir()
            (nested / "run-00_kernel_trace.csv").write_bytes(b"Kind,Name\nKERNEL,fixture\n")
        else:
            unsealed_json(paths["PROFILE_CAPTURE"], "capture-failure.json", {"schema_version": "ullm.aq4_p3_diagnostic_rocprof_failure.v2", "status": "failed", "reason": "capture failed", "children_remaining": [], "process_group_cleanup_complete": True, "ready_candidate_audit": {"reason_code": "marker_missing"}, "streams": {"rocprof.stdout": {"bytes": 0, "sha256": OPERATOR.sha_bytes(b"")}, "rocprof.stderr": {"bytes": len(b"capture failed\n"), "sha256": OPERATOR.sha_bytes(b"capture failed\n")}}})
        (paths["PROFILE_CAPTURE"] / "rocprof.stdout").write_bytes(b"")
        (paths["PROFILE_CAPTURE"] / "rocprof.stderr").write_bytes(b"" if succeeded else b"capture failed\n")
    elif pre_stop_case == "partial_runtime":
        paths["PROFILE_RUNTIME"].mkdir()
    paths["OPERATOR_RESULT"].mkdir()
    (paths["OPERATOR_RESULT"] / "operator.stdout.bin").write_bytes(OPERATOR.pretty({"status": status, "mode": "execute", "evidence": str(paths["MAINTENANCE_EVIDENCE"] / "launcher-evidence.json")}))
    (paths["OPERATOR_RESULT"] / "operator.stderr.bin").write_bytes(b"" if succeeded else b"maintenance failed\n")
    pre = {"service": {"active_state": "active", "sub_state": "running", "nrestarts": 0, "main_pid": 10}, "worker": {"path": "/worker", "pid": 20, "sha256": "f" * 64}, "gpu": {"device": "fixture"}, "owners": {"amd_smi": [20], "kfd": [20]}, "lock": {"path": "/run/lock", "busy": True, "identity": [1, 2, 3, 1, 0]}, "hashes": {"fixture": "same"}, "formal_health_sha256": "d" * 64}
    if pre_stop_noop:
        post = {**pre, "phase": "post_actual_evidence_recovery", "source": "fresh_read_only_phase_aware_probe", "previous_authorization_source": "sealed_operator_manifest_no_live_absence_recheck", "actual_outputs_permitted": True, "targeted_processes": [], "read_only": True, "service_touched": False, "gpu_workload_executed": False}
        if pre_stop_case == "epoch_changed":
            post = json.loads(json.dumps(post))
            post["service"]["main_pid"] = 11
        elif pre_stop_case == "owner_changed":
            post = json.loads(json.dumps(post))
            post["owners"]["amd_smi"] = [20, 30]
        elif pre_stop_case == "lock_changed":
            post = json.loads(json.dumps(post))
            post["lock"]["identity"][1] = 3
        elif pre_stop_case == "hash_changed":
            post = json.loads(json.dumps(post))
            post["hashes"]["fixture"] = "changed"
        elif pre_stop_case == "health_changed":
            post = json.loads(json.dumps(post))
            post["formal_health_sha256"] = "c" * 64
    else:
        post = {"phase": "post_actual_evidence_recovery", "source": "fresh_read_only_phase_aware_probe", "previous_authorization_source": "sealed_operator_manifest_no_live_absence_recheck", "actual_outputs_permitted": True, "service": {"active_state": "active", "sub_state": "running", "nrestarts": 0, "main_pid": 11}, "worker": {"path": "/worker", "pid": 21, "sha256": "f" * 64}, "gpu": {"device": "fixture"}, "owners": {"amd_smi": [21], "kfd": [21]}, "lock": {"path": "/run/lock", "busy": True, "identity": [1, 2, 3, 1, 0]}, "hashes": pre["hashes"], "formal_health_sha256": pre["formal_health_sha256"], "targeted_processes": [], "read_only": True, "service_touched": False, "gpu_workload_executed": False}
    monkeypatch.setattr(OPERATOR, "validate_operator", lambda _root=paths["OPERATOR_ROOT"]: {"value": manifest})
    monkeypatch.setattr(OPERATOR, "validate_quiet", lambda _root=paths["QUIET_ROOT"]: {"value": {"confirmation": pre}})
    monkeypatch.setattr(OPERATOR, "capture_recovery_snapshot", lambda _ready: post)
    monkeypatch.setattr(
        OPERATOR,
        "finalizer_source_authority",
        lambda: {
            "role": "existing_evidence_recovery_only_not_execution_authority",
            "path": str(SCRIPT),
            "commit": "e" * 40,
            "git_blob": "f" * 40,
            "sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(OPERATOR, "validate_finalizer_source_authority", lambda _value: None)
    monkeypatch.setattr(OPERATOR, "git", lambda *_args: "e" * 40)
    return paths


def test_v13_namespaces_bind_fresh_v16_ready_and_v10_profile_outputs() -> None:
    assert OPERATOR.PROFILE_READY_ROOT.name == "resident-one-case-smoke-profile-ready-v16"
    assert OPERATOR.PROFILE_READY_DRY_RUN_ROOT.name == "resident-one-case-smoke-profile-ready-dry-run-v16"
    assert OPERATOR.HISTORICAL_READY_V15_ROOT.name == "resident-one-case-smoke-profile-ready-v15"
    assert OPERATOR.OFFLINE_CAPTURE_ROOT.name == "aq4-p3-diagnostic-rocprof-capture-offline-reassembly-v11"
    assert OPERATOR.OFFLINE_EVIDENCE_ROOT.name == "resident-one-case-smoke-profile-maintenance-offline-reassembly-evidence-v11"
    assert OPERATOR.QUIET_ROOT.name == "resident-one-case-smoke-profile-quiet-window-v18"
    assert OPERATOR.OPERATOR_ROOT.name == "resident-one-case-smoke-profile-operator-command-v13"
    assert OPERATOR.MAINTENANCE_EVIDENCE.name == "resident-one-case-smoke-profile-maintenance-evidence-v11"
    assert OPERATOR.PROFILE_RUNTIME.name == "resident-one-case-smoke-profile-execute-v10"
    assert OPERATOR.PROFILE_EXECUTE_EVIDENCE.name == "resident-one-case-smoke-profile-execute-evidence-v10"
    assert OPERATOR.PROFILE_CAPTURE.name == "aq4-p3-diagnostic-rocprof-capture-v10"
    assert OPERATOR.OPERATOR_RESULT.name == "resident-one-case-smoke-profile-operator-result-v13"
    assert OPERATOR.ACTUAL_AUDIT.name == "resident-one-case-smoke-profile-actual-audit-v13"
    assert OPERATOR.EXECUTE_RUNTIME.name == "resident-one-case-smoke-execute-v10"
    assert OPERATOR.EXECUTE_EVIDENCE.name == "resident-one-case-smoke-execute-evidence-v10"
    assert OPERATOR.PREVIOUS_OPERATOR_ROOT.name == "resident-one-case-smoke-profile-operator-command-v12"
    assert OPERATOR.PREVIOUS_OPERATOR_V11_ROOT.name == "resident-one-case-smoke-profile-operator-command-v11"
    assert OPERATOR.PREVIOUS_OPERATOR_V10_ROOT.name == "resident-one-case-smoke-profile-operator-command-v10"
    assert OPERATOR.PREVIOUS_OPERATOR_RESULT_V10.name == "resident-one-case-smoke-profile-operator-result-v10"
    assert OPERATOR.PREVIOUS_ACTUAL_AUDIT_V10.name == "resident-one-case-smoke-profile-actual-audit-v10"
    assert OPERATOR.EXECUTE_BINDING_ROOT.name == "resident-one-case-smoke-execute-binding-v10"


def test_execute_binding_v10_and_launcher_authorities_are_exact() -> None:
    inventory = OPERATOR.verify_sums(OPERATOR.EXECUTE_BINDING_ROOT)
    assert (
        inventory["sha256sums_sha256"]
        == OPERATOR.EXECUTE_BINDING_SHA256SUMS_SHA256
    )
    assert (
        OPERATOR.git(
            "rev-parse",
            f"{OPERATOR.EXECUTE_BINDING_ARTIFACT_COMMIT}^{{tree}}",
        )
        == OPERATOR.EXECUTE_BINDING_ARTIFACT_TREE
    )
    assert (
        OPERATOR.git(
            "rev-parse",
            f"{OPERATOR.EXECUTE_BINDING_ARTIFACT_COMMIT}:"
            f"{OPERATOR.EXECUTE_BINDING_ROOT.relative_to(ROOT)}",
        )
        == OPERATOR.EXECUTE_BINDING_ROOT_TREE
    )
    OPERATOR.verify_inventory_commit(
        OPERATOR.EXECUTE_BINDING_ROOT,
        inventory,
        OPERATOR.EXECUTE_BINDING_ARTIFACT_COMMIT,
    )
    binding_path = OPERATOR.EXECUTE_BINDING_ROOT / "execute-binding.json"
    launcher_path = OPERATOR.EXECUTE_BINDING_ROOT / "launcher-trust.json"
    assert OPERATOR.sha_file(binding_path) == OPERATOR.EXECUTE_BINDING_MANIFEST_SHA256
    assert OPERATOR.sha_file(launcher_path) == OPERATOR.EXECUTE_LAUNCHER_TRUST_SHA256
    binding = OPERATOR.load(binding_path, "execute binding")
    launcher = OPERATOR.load(launcher_path, "launcher trust")
    assert binding["run_id"] == "p2-r9700-resident-one-case-smoke-execute-v10"
    assert binding["actual_eligible"] is False
    assert binding["runner_output"] == str(OPERATOR.EXECUTE_RUNTIME)
    assert binding["evidence_output"] == str(OPERATOR.EXECUTE_EVIDENCE)
    assert launcher["commit"] == OPERATOR.EXECUTE_LAUNCHER_COMMIT
    assert launcher["tree"] == OPERATOR.EXECUTE_LAUNCHER_TREE
    assert launcher["git_blob"] == OPERATOR.EXECUTE_LAUNCHER_BLOB
    assert launcher["sha256"] == OPERATOR.EXECUTE_LAUNCHER_SHA256
    assert launcher["actual_eligible"] is False


def test_v16_ready_execute_binding_and_fresh_output_authorities_are_exact() -> None:
    ready, inventory = OPERATOR.ready_authority()
    assert inventory["sha256sums_sha256"] == OPERATOR.READY_SHA256SUMS_SHA256
    dry_inventory = OPERATOR.verify_sums(OPERATOR.PROFILE_READY_DRY_RUN_ROOT)
    assert (
        dry_inventory["sha256sums_sha256"]
        == OPERATOR.READY_DRY_RUN_SHA256SUMS_SHA256
    )
    assert ready["actual_eligible"] is True
    assert ready["authorization"]["run_id"] == "p2-r9700-resident-one-case-smoke-profile-diagnostic-v10"
    assert OPERATOR.QUIET_SCHEMA.endswith(".v18")
    assert OPERATOR.OPERATOR_SCHEMA.endswith(".v13")
    assert OPERATOR.OPERATOR_RESULT_SCHEMA.endswith(".v13")
    assert OPERATOR.ACTUAL_AUDIT_SCHEMA.endswith(".v13")
    roots = OPERATOR.root_set()
    assert OPERATOR.EXECUTE_BINDING_ROOT in roots
    assert OPERATOR.PROFILE_READY_DRY_RUN_ROOT in roots
    assert OPERATOR.HISTORICAL_READY_V15_ROOT in roots
    assert OPERATOR.OFFLINE_CAPTURE_ROOT in roots
    assert OPERATOR.OFFLINE_EVIDENCE_ROOT in roots
    execute_inventory = OPERATOR.verify_sums(OPERATOR.EXECUTE_BINDING_ROOT)
    OPERATOR.verify_inventory_commit(
        OPERATOR.EXECUTE_BINDING_ROOT,
        execute_inventory,
        OPERATOR.EXECUTE_BINDING_ARTIFACT_COMMIT,
    )
    fresh = OPERATOR.fresh_paths(ready)
    assert len(fresh) == len({str(path) for path in fresh}) == 9
    previous = OPERATOR.previous_operator_v12_state()
    assert previous["state"] == "authorized_sealed"
    assert previous["authorization_commit"] == OPERATOR.PREVIOUS_OPERATOR_V12_COMMIT
    assert previous["authorization_tree"] == OPERATOR.PREVIOUS_OPERATOR_V12_TREE
    assert previous["maximum_invocations"] == 1
    assert previous["historical_actual_v11"]["state"] == "pre_stop_failed_sealed"
    actual = OPERATOR.previous_actual_v12_state()
    assert actual["state"] == "executed_sealed"
    assert actual["artifact_commit"] == OPERATOR.PREVIOUS_ACTUAL_V12_COMMIT
    assert actual["artifact_tree"] == OPERATOR.PREVIOUS_ACTUAL_V12_TREE
    assert actual["file_count"] == OPERATOR.PREVIOUS_ACTUAL_V12_FILE_COUNT
    assert actual["returncode"] == 1
    assert actual["invocation_count"] == actual["maximum_invocations"] == 1
    assert actual["retry_performed"] is False


def test_historical_v15_and_current_offline_v11_authorities_are_separate() -> None:
    historical = OPERATOR.historical_ready_v15_authority()
    offline = OPERATOR.offline_reassembly_authority()
    assert historical["status"] == "ready_for_one_case"
    assert historical["actual_eligible"] is True
    assert offline["artifact_commit"] == OPERATOR.OFFLINE_ARTIFACT_COMMIT
    assert offline["artifact_tree"] == OPERATOR.OFFLINE_ARTIFACT_TREE
    assert offline["file_count"] == 42
    assert offline["value"]["status"] == "offline_reassembled_sealed"
    assert OPERATOR.OFFLINE_CAPTURE_ROOT != OPERATOR.PROFILE_CAPTURE
    assert OPERATOR.OFFLINE_EVIDENCE_ROOT != OPERATOR.MAINTENANCE_EVIDENCE
    assert not OPERATOR.PROFILE_CAPTURE.exists()
    assert not OPERATOR.MAINTENANCE_EVIDENCE.exists()


def test_previous_v12_seal_is_independent_of_current_v13_poststate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_command = OPERATOR.previous_operator_v12_state()
    before_actual = OPERATOR.previous_actual_v12_state()
    assert before_command["state"] == "authorized_sealed"
    assert before_actual["state"] == "executed_sealed"
    assert before_actual["file_count"] == 35

    current = {
        "PROFILE_READY_ROOT": tmp_path / "profile-ready-v16",
        "PROFILE_READY": tmp_path / "profile-ready-v16/ready-binding.json",
        "QUIET_ROOT": tmp_path / "quiet-v18",
        "OPERATOR_ROOT": tmp_path / "operator-command-v13",
        "MAINTENANCE_EVIDENCE": tmp_path / "maintenance-v11",
        "PROFILE_RUNTIME": tmp_path / "execute-v10",
        "PROFILE_EXECUTE_EVIDENCE": tmp_path / "execute-evidence-v10",
        "PROFILE_CAPTURE": tmp_path / "capture-v10",
        "OPERATOR_RESULT": tmp_path / "operator-result-v13",
        "ACTUAL_AUDIT": tmp_path / "actual-audit-v13",
    }
    for name, path in current.items():
        monkeypatch.setattr(OPERATOR, name, path)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="ascii")
        else:
            path.mkdir(parents=True, exist_ok=True)
    dry_run = tmp_path / "profile-ready-dry-run-v16"
    dry_run.mkdir()
    (dry_run / "dry-run.json").write_text("{}\n", encoding="ascii")

    after_command = OPERATOR.previous_operator_v12_state()
    after_actual = OPERATOR.previous_actual_v12_state()
    assert after_command == before_command
    assert after_actual == before_actual
    historical_paths = set(OPERATOR.previous_actual_v12_fresh_paths())
    assert historical_paths.isdisjoint(current.values())


def test_actual_v11_rejects_partial_or_mixed_final_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maintenance = tmp_path / "maintenance-v9"
    result = tmp_path / "operator-result-v11"
    audit = tmp_path / "actual-audit-v11"
    runtime = tmp_path / "runtime-v9"
    execute = tmp_path / "execute-v9"
    capture = tmp_path / "capture-v9"
    monkeypatch.setattr(OPERATOR, "ACTUAL_V11_MAINTENANCE_EVIDENCE", maintenance)
    monkeypatch.setattr(OPERATOR, "ACTUAL_V11_OPERATOR_RESULT", result)
    monkeypatch.setattr(OPERATOR, "ACTUAL_V11_AUDIT", audit)
    monkeypatch.setattr(OPERATOR, "ACTUAL_V11_PROFILE_RUNTIME", runtime)
    monkeypatch.setattr(OPERATOR, "ACTUAL_V11_PROFILE_EXECUTE_EVIDENCE", execute)
    monkeypatch.setattr(OPERATOR, "ACTUAL_V11_PROFILE_CAPTURE", capture)
    maintenance.mkdir()
    with pytest.raises(OPERATOR.OperatorError, match="partial or mixed"):
        OPERATOR.actual_v11_state()
    result.mkdir()
    audit.mkdir()
    runtime.mkdir()
    with pytest.raises(OPERATOR.OperatorError, match="partial or mixed"):
        OPERATOR.actual_v11_state()


def patch_historical_actual_paths(
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, Path],
) -> None:
    for name, key in (
        ("HISTORICAL_MAINTENANCE_EVIDENCE_V8", "MAINTENANCE_EVIDENCE"),
        ("HISTORICAL_PROFILE_RUNTIME_V8", "PROFILE_RUNTIME"),
        ("HISTORICAL_PROFILE_EXECUTE_EVIDENCE_V8", "PROFILE_EXECUTE_EVIDENCE"),
        ("HISTORICAL_PROFILE_CAPTURE_V8", "PROFILE_CAPTURE"),
        ("HISTORICAL_OPERATOR_RESULT_V9", "OPERATOR_RESULT"),
        ("HISTORICAL_ACTUAL_AUDIT_V9", "ACTUAL_AUDIT"),
    ):
        monkeypatch.setattr(OPERATOR, name, paths[key])


def previous_v10_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> list[Path]:
    monkeypatch.setattr(OPERATOR, "ROOT", tmp_path)
    monkeypatch.setattr(OPERATOR, "PREVIOUS_OPERATOR_V10_ROOT", tmp_path / "operator-command-v10")
    monkeypatch.setattr(OPERATOR, "PROFILE_RUNTIME", tmp_path / "runtime-v9")
    monkeypatch.setattr(OPERATOR, "PROFILE_EXECUTE_EVIDENCE", tmp_path / "execute-evidence-v9")
    monkeypatch.setattr(OPERATOR, "MAINTENANCE_EVIDENCE", tmp_path / "maintenance-v9")
    monkeypatch.setattr(OPERATOR, "PROFILE_CAPTURE", tmp_path / "capture-v9")
    monkeypatch.setattr(OPERATOR, "PREVIOUS_OPERATOR_RESULT_V10", tmp_path / "operator-result-v10")
    monkeypatch.setattr(OPERATOR, "PREVIOUS_ACTUAL_AUDIT_V10", tmp_path / "actual-audit-v10")
    paths = OPERATOR.previous_authorization_v10_fresh_paths()
    argv = OPERATOR.actual_argv()
    manifest = {
        "schema_version": OPERATOR.PREVIOUS_OPERATOR_V10_SCHEMA,
        "argv": argv,
        "command_sha256": OPERATOR.sha_bytes(OPERATOR.canonical(argv)),
        "authorization": {
            "maximum_invocations": 1,
            "explicit_confirmation_flag_count": 1,
            "profile_diagnostic_flag_count": 1,
            "ready_artifact_flag_count": 1,
            "evidence_output_flag_count": 1,
        },
        "execution": {
            "maximum_invocations": 1,
            "shell": False,
            "requires_fresh_output_recheck_immediately_before_execution": True,
        },
        "fresh_outputs": [{"path": str(path), "absent": True} for path in paths],
        "actual_executed": False,
        "gpu_command_executed": False,
        "service_touched": False,
        "secret_material_embedded": False,
        "manifest_sha256": None,
    }
    manifest["manifest_sha256"] = OPERATOR.sha_bytes(OPERATOR.canonical(manifest))
    sealed(OPERATOR.PREVIOUS_OPERATOR_V10_ROOT, "command-manifest.json", manifest)
    inventory = OPERATOR.verify_sums(OPERATOR.PREVIOUS_OPERATOR_V10_ROOT)
    commit = "c" * 40
    tree = "d" * 40
    monkeypatch.setattr(OPERATOR, "PREVIOUS_OPERATOR_V10_COMMIT", commit)
    monkeypatch.setattr(OPERATOR, "PREVIOUS_OPERATOR_V10_TREE", tree)
    monkeypatch.setattr(
        OPERATOR,
        "PREVIOUS_OPERATOR_V10_MANIFEST_SHA256",
        OPERATOR.sha_file(OPERATOR.PREVIOUS_OPERATOR_V10_ROOT / "command-manifest.json"),
    )
    monkeypatch.setattr(
        OPERATOR,
        "PREVIOUS_OPERATOR_V10_SUMS_SHA256",
        inventory["sha256sums_sha256"],
    )
    monkeypatch.setattr(OPERATOR, "verify_inventory_commit", lambda *_args: None)

    expected = "\n".join(
        (
            "operator-command-v10/SHA256SUMS",
            "operator-command-v10/command-manifest.json",
        )
    )

    def previous_git(*args: str) -> str:
        if args[:1] == ("rev-parse",):
            return tree
        if args[:3] == ("ls-tree", "-r", "--name-only"):
            return expected
        raise AssertionError(f"unexpected Git query: {args}")

    monkeypatch.setattr(OPERATOR, "git", previous_git)
    return paths


def test_previous_operator_v10_accepts_authorized_not_invoked_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = previous_v10_fixture(tmp_path, monkeypatch)
    state = OPERATOR.previous_authorization_v10_state()
    assert state["state"] == "authorized_not_invoked_preflight_blocked"
    assert state["invocation_count"] == 0
    assert state["maximum_invocations"] == 1
    assert state["result_present"] is state["audit_present"] is False
    assert state["actual_executed"] is False
    assert [item["path"] for item in state["fresh_outputs"]] == [
        str(path) for path in paths
    ]
    assert all(item["present"] is False for item in state["fresh_outputs"])


def test_previous_operator_v10_rejects_any_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = previous_v10_fixture(tmp_path, monkeypatch)
    paths[0].mkdir()
    with pytest.raises(OPERATOR.OperatorError, match="partial outputs"):
        OPERATOR.previous_authorization_v10_state()


def test_historical_actual_v9_accepts_fully_absent_pre_execution_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        "MAINTENANCE_EVIDENCE": tmp_path / "maintenance-v8",
        "PROFILE_RUNTIME": tmp_path / "runtime-v8",
        "PROFILE_EXECUTE_EVIDENCE": tmp_path / "execute-evidence-v8",
        "PROFILE_CAPTURE": tmp_path / "capture-v8",
        "OPERATOR_RESULT": tmp_path / "operator-result-v9",
        "ACTUAL_AUDIT": tmp_path / "actual-audit-v9",
    }
    patch_historical_actual_paths(monkeypatch, paths)
    state = OPERATOR.historical_actual_v9_state()
    assert state["state"] == "not_executed"
    assert state["actual_executed"] is False
    assert len(state["fresh_outputs"]) == 9
    assert all(item["present"] is False for item in state["fresh_outputs"])


def test_historical_actual_v9_rejects_partial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        "MAINTENANCE_EVIDENCE": tmp_path / "maintenance-v8",
        "PROFILE_RUNTIME": tmp_path / "runtime-v8",
        "PROFILE_EXECUTE_EVIDENCE": tmp_path / "execute-evidence-v8",
        "PROFILE_CAPTURE": tmp_path / "capture-v8",
        "OPERATOR_RESULT": tmp_path / "operator-result-v9",
        "ACTUAL_AUDIT": tmp_path / "actual-audit-v9",
    }
    patch_historical_actual_paths(monkeypatch, paths)
    paths["OPERATOR_RESULT"].mkdir()
    with pytest.raises(OPERATOR.OperatorError, match="partial or mixed"):
        OPERATOR.historical_actual_v9_state()


def test_historical_actual_v9_accepts_only_sealed_commit_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = finalizer_fixture(tmp_path, monkeypatch, returncode=1)
    OPERATOR.finalize_actual(returncode=1, start_unix_ns=100, end_unix_ns=200)
    patch_historical_actual_paths(monkeypatch, paths)
    commit = "f" * 40
    tree = "a" * 40
    monkeypatch.setattr(OPERATOR, "HISTORICAL_ACTUAL_V9_COMMIT", commit)
    monkeypatch.setattr(OPERATOR, "HISTORICAL_ACTUAL_V9_TREE", tree)
    monkeypatch.setattr(OPERATOR, "HISTORICAL_OPERATOR_MANIFEST_V9_COMMIT", "e" * 40)
    monkeypatch.setattr(OPERATOR, "HISTORICAL_OPERATOR_RESULT_V9_SCHEMA", OPERATOR.OPERATOR_RESULT_SCHEMA)
    monkeypatch.setattr(OPERATOR, "HISTORICAL_ACTUAL_AUDIT_V9_SCHEMA", OPERATOR.ACTUAL_AUDIT_SCHEMA)
    roots = [
        paths["MAINTENANCE_EVIDENCE"],
        paths["PROFILE_EXECUTE_EVIDENCE"],
        paths["PROFILE_RUNTIME"],
        paths["PROFILE_CAPTURE"],
        paths["OPERATOR_RESULT"],
        paths["ACTUAL_AUDIT"],
    ]
    inventories = [OPERATOR.verify_sums(root) for root in roots]
    expected = sorted(
        {
            str(path.relative_to(tmp_path))
            for inventory in inventories
            for path in [
                Path(inventory["root"]) / "SHA256SUMS",
                *(Path(item["path"]) for item in inventory["members"].values()),
            ]
        }
    )
    monkeypatch.setattr(OPERATOR, "HISTORICAL_ACTUAL_V9_FILE_COUNT", len(expected))
    monkeypatch.setattr(OPERATOR, "verify_inventory_commit", lambda *_args: None)

    def historical_git(*args: str) -> str:
        if args[:1] == ("rev-parse",):
            return tree
        if args[:3] == ("ls-tree", "-r", "--name-only"):
            return "\n".join(expected)
        raise AssertionError(f"unexpected Git query: {args}")

    monkeypatch.setattr(OPERATOR, "git", historical_git)
    state = OPERATOR.historical_actual_v9_state()
    assert state["state"] == "executed_sealed"
    assert state["outcome"] == "failed"
    assert state["returncode"] == 1
    assert state["invocation_count"] == state["maximum_invocations"] == 1
    assert state["retry_performed"] is False
    assert state["file_count"] == len(expected)

    monkeypatch.setattr(OPERATOR, "HISTORICAL_ACTUAL_V9_TREE", "b" * 40)
    with pytest.raises(OPERATOR.OperatorError, match="Git tree differs"):
        OPERATOR.historical_actual_v9_state()


@pytest.mark.parametrize("returncode", (0, 17))
def test_finalizer_immutably_seals_success_and_nonzero_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
) -> None:
    paths = finalizer_fixture(tmp_path, monkeypatch, returncode=returncode)
    audit = OPERATOR.finalize_actual(returncode=returncode, start_unix_ns=100, end_unix_ns=200)
    succeeded = returncode == 0
    assert audit["status"] == ("passed_immutable_evidence_preserved_restore_passed" if succeeded else "failed_immutable_evidence_preserved_restore_passed")
    assert audit["failure"] is None if succeeded else audit["failure"]["returncode"] == returncode
    assert audit["execution"]["maximum_invocations"] == 1
    assert audit["execution"]["invocation_count"] == 1
    assert audit["execution"]["shell"] is False
    assert audit["execution"]["retry_performed"] is False
    assert audit["restore"]["passed"] is True
    assert audit["restore_classification"] == "outer_finally_restored_new_epoch"
    assert audit["pre_stop_failure_snapshot"] is None
    assert audit["cleanup"]["trusted_lock_substrate_cleanup_required"] is True
    assert audit["cleanup"]["retry_forbidden_and_not_performed"] is True
    assert audit["profile_artifacts"]["status"] == ("complete_diagnostic" if succeeded else "failure_evidence_only")
    validated = OPERATOR.validate_actual()
    assert validated["result"]["status"] == ("passed" if succeeded else "failed")
    assert validated["result"]["maximum_invocations"] == 1
    assert validated["result"]["invocation_count"] == 1
    assert validated["result"]["shell"] is False
    assert validated["result"]["retry_performed"] is False
    for root in (paths["OPERATOR_RESULT"], paths["ACTUAL_AUDIT"], paths["PROFILE_RUNTIME"], paths["PROFILE_CAPTURE"]):
        assert OPERATOR.verify_sums(root)["mode"] == "0555"
    if succeeded:
        assert (paths["PROFILE_CAPTURE"] / "measured-runs").stat().st_mode & 0o777 == 0o555


def test_finalizer_seals_strict_pre_stop_noop_failure_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = finalizer_fixture(
        tmp_path,
        monkeypatch,
        returncode=1,
        pre_stop_noop=True,
    )
    audit = OPERATOR.finalize_actual(
        returncode=1,
        start_unix_ns=100,
        end_unix_ns=18_000_000_100,
    )
    assert audit["status"] == "failed_immutable_evidence_preserved_restore_passed"
    assert audit["execution"]["invocation_count"] == 1
    assert audit["execution"]["maximum_invocations"] == 1
    assert audit["execution"]["retry_performed"] is False
    assert audit["execution"]["elapsed_ns"] == 18_000_000_000
    assert audit["restore"] == {
        "attempted": False,
        "error": None,
        "passed": True,
        "post_start": None,
    }
    assert audit["restore_classification"] == "pre_stop_untouched_same_epoch"
    assert audit["pre_stop_failure_snapshot"]["owner_identity_evidence"] == "unavailable_not_recorded_by_pre_stop_probe"
    assert audit["pre_stop_failure_snapshot"]["normative_external_owner_pids"] is None
    assert audit["recovery_snapshot"]["previous_authorization_source"] == "sealed_operator_manifest_no_live_absence_recheck"
    assert audit["cleanup"]["trusted_lock_substrate_cleanup_required"] is False
    assert audit["evidence"]["execute"] is None
    assert audit["evidence"]["runtime"] is None
    assert audit["evidence"]["capture"] is None
    assert OPERATOR.validate_actual()["audit"] == audit
    for root in (paths["PROFILE_EXECUTE_EVIDENCE"], paths["PROFILE_RUNTIME"], paths["PROFILE_CAPTURE"]):
        assert not root.exists()


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("restore_false", "pre-stop no-op restore evidence"),
        ("restore_unknown", "pre-stop no-op restore evidence"),
        ("later_stage", "pre-stop no-op restore evidence"),
        ("process_started", "pre-stop no-op restore evidence"),
        ("service_stop_started", "pre-stop no-op restore evidence"),
        ("capture_started", "pre-stop no-op restore evidence"),
        ("rocprof_started", "pre-stop no-op restore evidence"),
        ("service_touched", "pre-stop no-op restore evidence"),
        ("cleanup_present", "pre-stop no-op restore evidence"),
        ("partial_runtime", "downstream artifacts"),
        ("epoch_changed", "recovery epoch"),
        ("owner_changed", "owner/residual state"),
        ("lock_changed", "recovery epoch"),
        ("hash_changed", "recovery epoch"),
        ("health_changed", "recovery epoch"),
    ),
)
def test_finalizer_rejects_unsafe_pre_stop_noop_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    finalizer_fixture(
        tmp_path,
        monkeypatch,
        returncode=1,
        pre_stop_noop=True,
        pre_stop_case=case,
    )
    with pytest.raises(OPERATOR.OperatorError, match=message):
        OPERATOR.finalize_actual(
            returncode=1,
            start_unix_ns=100,
            end_unix_ns=18_000_000_100,
        )


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
    sealed(previous_root, "command-manifest.json", {"schema_version": "historical.v8"})
    monkeypatch.setattr(OPERATOR, "QUIET_ROOT", quiet_root)
    monkeypatch.setattr(OPERATOR, "PREVIOUS_OPERATOR_ROOT", previous_root)
    monkeypatch.setattr(OPERATOR, "ready_authority", lambda: (ready, {"root": "ready-v16"}))
    monkeypatch.setattr(OPERATOR, "fresh_paths", lambda _ready: fresh)
    monkeypatch.setattr(
        OPERATOR,
        "previous_operator_v12_state",
        lambda: {
            "state": "authorized_sealed",
            "authorization_commit": OPERATOR.PREVIOUS_OPERATOR_V12_COMMIT,
            "authorization_tree": OPERATOR.PREVIOUS_OPERATOR_V12_TREE,
            "manifest_file_sha256": OPERATOR.PREVIOUS_OPERATOR_V12_MANIFEST_SHA256,
            "maximum_invocations": 1,
            "inventory": {"root": str(previous_root)},
        },
    )
    previous_actual_v12 = {
        "state": "executed_sealed",
        "artifact_commit": OPERATOR.PREVIOUS_ACTUAL_V12_COMMIT,
        "artifact_tree": OPERATOR.PREVIOUS_ACTUAL_V12_TREE,
        "file_count": OPERATOR.PREVIOUS_ACTUAL_V12_FILE_COUNT,
        "invocation_count": 1,
        "maximum_invocations": 1,
        "retry_performed": False,
        "previous_operator_v12": {
            "authorization_commit": OPERATOR.PREVIOUS_OPERATOR_V12_COMMIT,
        },
    }
    monkeypatch.setattr(
        OPERATOR,
        "previous_actual_v12_state",
        lambda: previous_actual_v12,
    )
    monkeypatch.setattr(
        OPERATOR,
        "historical_ready_v15_authority",
        lambda: {"status": "ready_for_one_case", "actual_eligible": True},
    )
    monkeypatch.setattr(
        OPERATOR,
        "offline_reassembly_authority",
        lambda: {
            "value": {"status": "offline_reassembled_sealed"},
            "artifact_commit": OPERATOR.OFFLINE_ARTIFACT_COMMIT,
            "artifact_tree": OPERATOR.OFFLINE_ARTIFACT_TREE,
            "file_count": 42,
        },
    )

    value = OPERATOR.prepare_operator(output_root)
    validated = OPERATOR.validate_operator(output_root)["value"]
    assert validated == value
    assert value["failure_contract"]["retry_forbidden"] is True
    assert value["failure_contract"]["outer_restore_in_finally"] is True
    assert value["failure_contract"]["restore_timeout_seconds"] == 120.0
    assert value["failure_contract"]["children_remaining_must_be_empty"] is True
    assert value["inputs"]["previous_operator_v12"]["state"] == "authorized_sealed"
    assert value["inputs"]["previous_actual_v12"]["state"] == "executed_sealed"
    assert value["inputs"]["historical_ready_v15"]["artifact_commit"] == OPERATOR.HISTORICAL_READY_V15_COMMIT
    assert value["inputs"]["offline_reassembly_v11"]["artifact_commit"] == OPERATOR.OFFLINE_ARTIFACT_COMMIT
    assert value["pre_execution_audit"]["previous_operator_v12"] == "authorized_sealed"
    assert value["pre_execution_audit"]["previous_actual_v12"] == "executed_sealed"
    clone = json.loads(json.dumps(value))
    declared = clone["manifest_sha256"]
    clone["manifest_sha256"] = None
    assert declared == OPERATOR.sha_bytes(OPERATOR.canonical(clone))

    tampered = json.loads(json.dumps(value))
    tampered["inputs"]["previous_actual_v12"]["retry_performed"] = True
    tampered["manifest_sha256"] = None
    tampered["manifest_sha256"] = OPERATOR.sha_bytes(OPERATOR.canonical(tampered))
    tampered_root = tmp_path / "operator-v13-tampered"
    sealed(tampered_root, "command-manifest.json", tampered)
    with pytest.raises(OPERATOR.OperatorError, match="previous/final-state binding"):
        OPERATOR.validate_operator(tampered_root)


def test_prepare_operator_requires_previous_actual_v12_final_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(OPERATOR, "ready_authority", lambda: ({}, {}))
    monkeypatch.setattr(OPERATOR, "validate_quiet", lambda _root: {"value": {}})
    monkeypatch.setattr(
        OPERATOR,
        "previous_operator_v12_state",
        lambda: {"state": "authorized_sealed"},
    )
    monkeypatch.setattr(
        OPERATOR,
        "previous_actual_v12_state",
        lambda: {"state": "not_executed"},
    )
    with pytest.raises(OPERATOR.OperatorError, match="previous actual-v12 final state"):
        OPERATOR.prepare_operator(tmp_path / "operator-v13")


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
