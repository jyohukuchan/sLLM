from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import shutil
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
        "QUIET_ROOT": tmp_path / "quiet-v19",
        "OPERATOR_ROOT": tmp_path / "operator-command-v14",
        "MAINTENANCE_EVIDENCE": tmp_path / "maintenance-v11",
        "OPERATOR_RESULT": tmp_path / "operator-result-v14",
        "ACTUAL_AUDIT": tmp_path / "actual-audit-v14",
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


def test_v15_paths_and_current_authority_are_bound() -> None:
    assert OPERATOR.PROFILE_READY_ROOT.name == "resident-one-case-smoke-profile-ready-v17"
    assert OPERATOR.PROFILE_READY_DRY_RUN_ROOT.name == "resident-one-case-smoke-profile-ready-dry-run-v17"
    assert OPERATOR.HISTORICAL_READY_V15_ROOT.name == "resident-one-case-smoke-profile-ready-v15"
    assert OPERATOR.OFFLINE_CAPTURE_ROOT.name == "aq4-p3-diagnostic-rocprof-capture-offline-reassembly-v12"
    assert OPERATOR.OFFLINE_EVIDENCE_ROOT.name == "resident-one-case-smoke-profile-maintenance-offline-reassembly-evidence-v12"
    assert OPERATOR.QUIET_ROOT.name == "resident-one-case-smoke-profile-quiet-window-v20"
    assert OPERATOR.OPERATOR_ROOT.name == "resident-one-case-smoke-profile-operator-command-v15"
    assert OPERATOR.MAINTENANCE_EVIDENCE.name == "resident-one-case-smoke-profile-maintenance-evidence-v12"
    assert OPERATOR.PROFILE_RUNTIME.name == "resident-one-case-smoke-profile-execute-v11"
    assert OPERATOR.PROFILE_EXECUTE_EVIDENCE.name == "resident-one-case-smoke-profile-execute-evidence-v11"
    assert OPERATOR.PROFILE_CAPTURE.name == "aq4-p3-diagnostic-rocprof-capture-v11"
    assert OPERATOR.OPERATOR_RESULT.name == "resident-one-case-smoke-profile-operator-result-v15"
    assert OPERATOR.ACTUAL_AUDIT.name == "resident-one-case-smoke-profile-actual-audit-v15"
    assert OPERATOR.PREVIOUS_ACTUAL_V14_MAINTENANCE_EVIDENCE.name == "resident-one-case-smoke-profile-maintenance-evidence-v11"
    assert OPERATOR.PREVIOUS_ACTUAL_V14_PROFILE_RUNTIME.name == "resident-one-case-smoke-profile-execute-v10"
    assert OPERATOR.PREVIOUS_ACTUAL_V14_PROFILE_EXECUTE_EVIDENCE.name == "resident-one-case-smoke-profile-execute-evidence-v10"
    assert OPERATOR.PREVIOUS_ACTUAL_V14_PROFILE_CAPTURE.name == "aq4-p3-diagnostic-rocprof-capture-v10"
    assert OPERATOR.EXECUTE_RUNTIME.name == "resident-one-case-smoke-execute-v10"
    assert OPERATOR.EXECUTE_EVIDENCE.name == "resident-one-case-smoke-execute-evidence-v10"
    assert OPERATOR.PREVIOUS_OPERATOR_ROOT.name == "resident-one-case-smoke-profile-operator-command-v13"
    assert OPERATOR.PREVIOUS_OPERATOR_V12_ROOT.name == "resident-one-case-smoke-profile-operator-command-v12"
    assert OPERATOR.PREVIOUS_OPERATOR_V11_ROOT.name == "resident-one-case-smoke-profile-operator-command-v11"
    assert OPERATOR.PREVIOUS_OPERATOR_V10_ROOT.name == "resident-one-case-smoke-profile-operator-command-v10"
    assert OPERATOR.PREVIOUS_OPERATOR_RESULT_V10.name == "resident-one-case-smoke-profile-operator-result-v10"
    assert OPERATOR.PREVIOUS_ACTUAL_AUDIT_V10.name == "resident-one-case-smoke-profile-actual-audit-v10"
    assert OPERATOR.EXECUTE_BINDING_ROOT.name == "resident-one-case-smoke-execute-binding-v10"
    assert OPERATOR.CURRENT_V15_AUTHORITY_BOUND is True


def test_v15_fresh9_is_exact_and_absent_after_authority_binding() -> None:
    capture = OPERATOR.PROFILE_CAPTURE
    assert isinstance(capture, Path)
    expected = [
        OPERATOR.PROFILE_RUNTIME,
        OPERATOR.PROFILE_EXECUTE_EVIDENCE,
        OPERATOR.MAINTENANCE_EVIDENCE,
        capture,
        capture / "capture-artifact.json",
        capture / "rocprof.stdout",
        capture / "rocprof.stderr",
        OPERATOR.OPERATOR_RESULT,
        OPERATOR.ACTUAL_AUDIT,
    ]
    fresh = OPERATOR.current_fresh_paths()
    assert fresh == expected
    assert len(fresh) == len({str(path) for path in fresh}) == 9
    assert all(path.is_absolute() and ".." not in path.parts for path in fresh)
    assert all(not path.exists() and not path.is_symlink() for path in fresh)
    OPERATOR.require_current_v15_authority()


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


def test_execute_binding_v11_is_path_namespace_authority_only() -> None:
    authority = OPERATOR.execute_binding_v11_namespace_authority()
    binding = authority["binding"]
    assert authority["artifact_commit"] == OPERATOR.EXECUTE_BINDING_V11_ARTIFACT_COMMIT
    assert authority["root_tree"] == OPERATOR.EXECUTE_BINDING_V11_ROOT_TREE
    assert authority["profile_namespace_authority"] is False
    assert authority["execution_authority"] is False
    assert binding["runner_output"] == str(OPERATOR.EXECUTE_RUNTIME_V11)
    assert binding["evidence_output"] == str(OPERATOR.EXECUTE_EVIDENCE_V11)
    assert binding["actual_eligible"] is False
    assert authority["launcher_trust"]["commit"] == OPERATOR.EXECUTE_LAUNCHER_V11_COMMIT
    assert authority["launcher_trust"]["tree"] == OPERATOR.EXECUTE_LAUNCHER_V11_TREE
    assert authority["launcher_trust"]["git_blob"] == OPERATOR.EXECUTE_LAUNCHER_V11_BLOB
    assert authority["launcher_trust"]["sha256"] == OPERATOR.EXECUTE_LAUNCHER_V11_SHA256
    assert OPERATOR.EXECUTE_RUNTIME_V11 != OPERATOR.PROFILE_RUNTIME
    assert OPERATOR.EXECUTE_EVIDENCE_V11 != OPERATOR.PROFILE_EXECUTE_EVIDENCE


def test_v15_schema_boundary_and_previous_v14_authority_are_exact() -> None:
    assert OPERATOR.QUIET_SCHEMA.endswith(".v20")
    assert OPERATOR.OPERATOR_SCHEMA.endswith(".v15")
    assert OPERATOR.OPERATOR_RESULT_SCHEMA.endswith(".v15")
    assert OPERATOR.ACTUAL_AUDIT_SCHEMA.endswith(".v15")
    roots = OPERATOR.current_v15_actual_roots()
    assert len(roots) == len(set(roots.values())) == 6
    OPERATOR.require_current_v15_authority()
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
    actual = OPERATOR.previous_actual_v14_state()
    assert actual["state"] == "executed_sealed_failure_restore_passed"
    assert actual["artifact_commit"] == OPERATOR.PREVIOUS_ACTUAL_V14_COMMIT
    assert actual["artifact_tree"] == OPERATOR.PREVIOUS_ACTUAL_V14_TREE
    assert actual["journal_commit"] == OPERATOR.PREVIOUS_ACTUAL_V14_JOURNAL_COMMIT
    assert actual["file_count"] == OPERATOR.PREVIOUS_ACTUAL_V14_FILE_COUNT
    assert actual["returncode"] == 1
    assert actual["invocation_count"] == actual["maximum_invocations"] == 1
    assert actual["retry_performed"] is False
    assert actual["restore_passed"] is True


def test_current_v15_authority_formal_readback_is_exact_and_read_only() -> None:
    authority = OPERATOR.current_v15_authority()
    assert authority["authority_bound"] is True
    assert authority["ready_artifact_commit"] == OPERATOR.READY_ARTIFACT_COMMIT
    assert authority["ready_actual_eligible"] is True
    assert authority["offline_artifact_commit"] == OPERATOR.OFFLINE_ARTIFACT_COMMIT
    assert authority["offline_file_count"] == 42
    assert authority["actual_root_count"] == 6
    assert authority["fresh_output_count"] == 9
    assert authority["fresh_outputs_absent"] is True
    assert authority["previous_operator_v13_state"] == "authorized_not_invoked_preflight_blocked"
    assert authority["previous_actual_v14_state"] == "executed_sealed_failure_restore_passed"
    assert authority["read_only"] is True
    assert authority["actual_executed"] is False
    assert authority["gpu_command_executed"] is False
    assert authority["service_touched"] is False


def test_historical_ready_v15_is_self_contained_and_poststate_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(OPERATOR, "PROFILE_READY_ROOT", tmp_path / "current-v99")
    monkeypatch.setattr(OPERATOR, "PROFILE_READY", tmp_path / "current-v99/ready.json")
    monkeypatch.setattr(
        OPERATOR,
        "PROFILE_READY_DRY_RUN_ROOT",
        tmp_path / "current-dry-v99",
    )
    monkeypatch.setattr(OPERATOR, "MAINTENANCE", tmp_path / "current-maintenance.py")
    monkeypatch.setattr(
        OPERATOR,
        "load_maintenance",
        lambda: (_ for _ in ()).throw(AssertionError("current loader forbidden")),
    )
    value = OPERATOR.historical_ready_v15_authority()
    assert value["status"] == "ready_for_one_case"
    assert value["actual_eligible"] is True
    names = {
        node.id
        for node in ast.walk(
            ast.parse(inspect.getsource(OPERATOR.historical_ready_v15_authority))
        )
        if isinstance(node, ast.Name)
    }
    assert names.isdisjoint(
        {
            "PROFILE_READY_ROOT",
            "PROFILE_READY",
            "PROFILE_READY_DRY_RUN_ROOT",
            "PREVIOUS_V13_PROFILE_READY",
            "MAINTENANCE",
            "load_maintenance",
        }
    )


def test_historical_ready_v15_rejects_root_inventory_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify = OPERATOR.verify_sums

    def mismatched(root: Path) -> dict:
        inventory = verify(root)
        if root == OPERATOR.HISTORICAL_READY_V15_ROOT:
            inventory = json.loads(json.dumps(inventory))
            inventory["members"].pop("qa-attestation.json")
        return inventory

    monkeypatch.setattr(OPERATOR, "verify_sums", mismatched)
    with pytest.raises(OPERATOR.OperatorError, match="authority differs"):
        OPERATOR.historical_ready_v15_authority()


def test_historical_ready_v15_rejects_source_git_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_bytes = OPERATOR.git_bytes

    def mismatched(*args: str) -> bytes:
        if args[:1] == ("show",) and args[1].startswith(
            OPERATOR.HISTORICAL_READY_V15_MAINTENANCE_COMMIT
        ):
            return b"tampered historical source"
        return git_bytes(*args)

    monkeypatch.setattr(OPERATOR, "git_bytes", mismatched)
    with pytest.raises(OPERATOR.OperatorError, match="source authority differs"):
        OPERATOR.historical_ready_v15_authority()


@pytest.mark.parametrize("field", ("qa", "trust_path"))
def test_historical_ready_v15_rejects_embedded_qa_or_wrong_source_path(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    load = OPERATOR.load

    def mismatched(path: Path, label: str) -> dict:
        value = load(path, label)
        if field == "qa" and path.name == "qa-attestation.json":
            value["automated_tests"]["aggregate"]["passed"] -= 1
        if field == "trust_path" and path.name == "harness-trust.json":
            value["path"] = "/wrong/historical-maintenance.py"
        return value

    monkeypatch.setattr(OPERATOR, "load", mismatched)
    match = "QA aggregate differs" if field == "qa" else "source authority differs"
    with pytest.raises(OPERATOR.OperatorError, match=match):
        OPERATOR.historical_ready_v15_authority()


def test_audit_current_real_authority_integration_is_clean() -> None:
    audit = OPERATOR.audit_current()
    assert audit["status"] == "clean"
    assert audit["fresh_outputs_absent"] is True
    assert audit["targeted_processes"] == []
    assert audit["owners"]["amd_smi"] == [audit["worker"]["pid"]]
    assert audit["owners"]["kfd"] == [audit["worker"]["pid"]]
    assert audit["actual_executed"] is False
    assert audit["service_touched"] is False


def test_previous_v13_authority_is_immutable_uninvoked_and_poststate_independent() -> None:
    state = OPERATOR.previous_authorization_v13_state()
    assert state["state"] == "authorized_not_invoked_preflight_blocked"
    assert state["reason"] == "external_owner_after_seal_before_invocation"
    assert state["authorization_commit"] == OPERATOR.PREVIOUS_OPERATOR_V13_COMMIT
    assert state["authorization_tree"] == OPERATOR.PREVIOUS_OPERATOR_V13_TREE
    assert state["authorization_root_tree"] == OPERATOR.PREVIOUS_OPERATOR_V13_ROOT_TREE
    assert state["manifest_file_sha256"] == OPERATOR.PREVIOUS_OPERATOR_V13_MANIFEST_SHA256
    assert state["manifest_semantic_sha256"] == OPERATOR.PREVIOUS_OPERATOR_V13_SEMANTIC_SHA256
    assert state["command_sha256"] == OPERATOR.PREVIOUS_OPERATOR_V13_COMMAND_SHA256
    assert state["invocation_count"] == 0
    assert state["maximum_invocations"] == 1
    assert state["result_present"] is state["audit_present"] is False
    assert state["actual_executed"] is False
    assert state["gpu_command_executed"] is False
    assert state["service_touched"] is False
    assert len(state["fresh_outputs"]) == 9
    assert all(item["present"] is False for item in state["fresh_outputs"])
    assert OPERATOR.git(
        "rev-parse", f"{OPERATOR.PREVIOUS_OPERATOR_V13_COMMIT}^{{tree}}"
    ) == OPERATOR.PREVIOUS_OPERATOR_V13_TREE
    assert OPERATOR.git(
        "rev-parse",
        f"{OPERATOR.PREVIOUS_OPERATOR_V13_COMMIT}:"
        f"{OPERATOR.PREVIOUS_OPERATOR_V13_ROOT.relative_to(ROOT)}",
    ) == OPERATOR.PREVIOUS_OPERATOR_V13_ROOT_TREE
    assert OPERATOR.PREVIOUS_OPERATOR_V13_COMMIT != OPERATOR.git("rev-parse", "HEAD")


def test_previous_v13_validator_does_not_reference_current_v14_outputs() -> None:
    source = inspect.getsource(OPERATOR.previous_operator_v13_fresh_paths)
    for forbidden in (
        "PROFILE_RUNTIME",
        "PROFILE_EXECUTE_EVIDENCE",
        "MAINTENANCE_EVIDENCE",
        "PROFILE_CAPTURE",
        "OPERATOR_RESULT",
        "ACTUAL_AUDIT",
    ):
        assert f" {forbidden}," not in source
        assert f" {forbidden} /" not in source


def test_previous_quiet_v18_is_immutable_but_rejected_as_current_v19() -> None:
    authority = OPERATOR.previous_quiet_v18_authority()
    assert authority["artifact_commit"] == OPERATOR.PREVIOUS_QUIET_V18_COMMIT
    assert authority["root_tree"] == OPERATOR.PREVIOUS_QUIET_V18_ROOT_TREE
    assert authority["json_sha256"] == OPERATOR.PREVIOUS_QUIET_V18_JSON_SHA256
    with pytest.raises(OPERATOR.OperatorError, match="decision/safety"):
        OPERATOR.validate_quiet(OPERATOR.PREVIOUS_QUIET_V18_ROOT)


def test_audit_current_integrates_real_prepared_and_binding_mode_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = OPERATOR.verify_sums(OPERATOR.PREPARED_ROOT)
    binding = OPERATOR.verify_sums(OPERATOR.BINDING_ROOT)
    assert prepared["members"]["resident-driver"]["mode"] == "0555"
    assert all(
        member["mode"] == "0444"
        for name, member in prepared["members"].items()
        if name != "resident-driver"
    )
    assert all(member["mode"] == "0444" for member in binding["members"].values())

    monkeypatch.setattr(OPERATOR, "CURRENT_V15_AUTHORITY_BOUND", False)
    touched = False

    def forbidden_ready() -> tuple[dict, dict]:
        nonlocal touched
        touched = True
        raise AssertionError("unbound audit must not inspect ready authority")

    monkeypatch.setattr(OPERATOR, "ready_authority", forbidden_ready)
    with pytest.raises(OPERATOR.OperatorError, match="authority is unbound"):
        OPERATOR.audit_current()
    assert touched is False


@pytest.mark.parametrize("driver_mode", (0o444, 0o644))
def test_prepared_driver_rejects_non_executable_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    driver_mode: int,
) -> None:
    prepared = tmp_path / "prepared-v2"
    shutil.copytree(OPERATOR.PREPARED_ROOT, prepared, copy_function=shutil.copy2)
    monkeypatch.setattr(OPERATOR, "PREPARED_ROOT", prepared)
    os.chmod(prepared / "resident-driver", driver_mode)
    with pytest.raises(OPERATOR.OperatorError, match="sealed member differs"):
        OPERATOR.verify_sums(prepared)


def test_prepared_json_rejects_executable_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = tmp_path / "prepared-v2"
    shutil.copytree(OPERATOR.PREPARED_ROOT, prepared, copy_function=shutil.copy2)
    monkeypatch.setattr(OPERATOR, "PREPARED_ROOT", prepared)
    os.chmod(prepared / "identity.json", 0o555)
    with pytest.raises(OPERATOR.OperatorError, match="sealed member differs"):
        OPERATOR.verify_sums(prepared)


def test_previous_v14_actual_is_separate_from_bound_v15_paths() -> None:
    actual = OPERATOR.previous_actual_v14_state()
    assert actual["state"] == "executed_sealed_failure_restore_passed"
    historical = {
        OPERATOR.PREVIOUS_ACTUAL_V14_MAINTENANCE_EVIDENCE,
        OPERATOR.PREVIOUS_ACTUAL_V14_PROFILE_RUNTIME,
        OPERATOR.PREVIOUS_ACTUAL_V14_PROFILE_EXECUTE_EVIDENCE,
        OPERATOR.PREVIOUS_ACTUAL_V14_PROFILE_CAPTURE,
        OPERATOR.PREVIOUS_ACTUAL_V14_OPERATOR_RESULT,
        OPERATOR.PREVIOUS_ACTUAL_V14_AUDIT,
    }
    assert OPERATOR.OPERATOR_RESULT not in historical
    assert OPERATOR.ACTUAL_AUDIT not in historical
    assert set(OPERATOR.current_v15_actual_roots().values()).isdisjoint(historical)


def test_previous_v14_seal_is_independent_of_current_v15_poststate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_v14 = OPERATOR.previous_actual_v14_state()
    before_v13 = OPERATOR.previous_authorization_v13_state()
    current = {
        "MAINTENANCE_EVIDENCE": tmp_path / "maintenance-v12",
        "PROFILE_RUNTIME": tmp_path / "runtime-v11",
        "PROFILE_EXECUTE_EVIDENCE": tmp_path / "execute-evidence-v11",
        "PROFILE_CAPTURE": tmp_path / "capture-v11",
        "OPERATOR_RESULT": tmp_path / "operator-result-v15",
        "ACTUAL_AUDIT": tmp_path / "actual-audit-v15",
    }
    for name, path in current.items():
        monkeypatch.setattr(OPERATOR, name, path)
        path.mkdir(parents=True)
        (path / "later-version-evidence").write_bytes(b"poststate")
    assert OPERATOR.previous_actual_v14_state() == before_v14
    assert OPERATOR.previous_authorization_v13_state() == before_v13
    assert set(OPERATOR.current_v15_actual_roots().values()) == set(current.values())


def test_operator_source_authority_uses_path_last_change() -> None:
    record = OPERATOR.trusted_operator_source_record()
    assert record["source_commit"] == record["artifact_commit"]
    assert record["source_commit"] == OPERATOR.git(
        "log",
        "-1",
        "--format=%H",
        "--",
        str(SCRIPT.relative_to(ROOT)),
    )


@pytest.mark.parametrize("tamper", ("source_bytes", "blob_authority"))
def test_operator_source_authority_rejects_tamper_and_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    commit = "1" * 40
    tree = "2" * 40
    blob = OPERATOR.git("hash-object", str(SCRIPT))
    raw = SCRIPT.read_bytes()

    def fake_git(*args: str) -> str:
        if args[:4] == ("log", "-1", "--format=%H", "--"):
            return commit
        if args == ("rev-parse", f"{commit}^{{tree}}"):
            return tree
        if args[:2] == ("rev-parse", f"{commit}:{SCRIPT.relative_to(ROOT)}"):
            return blob
        if args[:1] == ("hash-object",):
            return "0" * 40 if tamper == "blob_authority" else blob
        raise AssertionError(f"unexpected Git call: {args}")

    monkeypatch.setattr(OPERATOR, "git", fake_git)
    monkeypatch.setattr(
        OPERATOR,
        "git_bytes",
        lambda *_args: b"tampered-source" if tamper == "source_bytes" else raw,
    )
    with pytest.raises(OPERATOR.OperatorError, match="last-change authority"):
        OPERATOR.trusted_operator_source_record()


def test_previous_v13_seal_is_independent_of_current_v14_poststate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_v13 = OPERATOR.previous_authorization_v13_state()
    before_command = OPERATOR.previous_operator_v12_state()
    before_actual = OPERATOR.previous_actual_v12_state()
    assert before_command["state"] == "authorized_sealed"
    assert before_actual["state"] == "executed_sealed"
    assert before_actual["file_count"] == 35

    current = {
        "PROFILE_READY_ROOT": tmp_path / "profile-ready-v16",
        "PROFILE_READY": tmp_path / "profile-ready-v16/ready-binding.json",
        "QUIET_ROOT": tmp_path / "quiet-v19",
        "OPERATOR_ROOT": tmp_path / "operator-command-v14",
        "MAINTENANCE_EVIDENCE": tmp_path / "maintenance-v11",
        "PROFILE_RUNTIME": tmp_path / "execute-v10",
        "PROFILE_EXECUTE_EVIDENCE": tmp_path / "execute-evidence-v10",
        "PROFILE_CAPTURE": tmp_path / "capture-v10",
        "OPERATOR_RESULT": tmp_path / "operator-result-v14",
        "ACTUAL_AUDIT": tmp_path / "actual-audit-v14",
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

    after_v13 = OPERATOR.previous_authorization_v13_state()
    after_command = OPERATOR.previous_operator_v12_state()
    after_actual = OPERATOR.previous_actual_v12_state()
    assert after_v13 == before_v13
    assert after_command == before_command
    assert after_actual == before_actual
    historical_paths = set(OPERATOR.previous_actual_v12_fresh_paths())
    assert historical_paths.isdisjoint(current.values())
    previous_v13_paths = set(OPERATOR.previous_operator_v13_fresh_paths())
    assert previous_v13_paths.isdisjoint(current.values())


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


def previous_v13_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> list[Path]:
    root = tmp_path / "operator-command-v13"
    monkeypatch.setattr(OPERATOR, "ROOT", tmp_path)
    monkeypatch.setattr(OPERATOR, "PREVIOUS_OPERATOR_V13_ROOT", root)
    monkeypatch.setattr(OPERATOR, "PREVIOUS_QUIET_V18_ROOT", tmp_path / "quiet-v18")
    monkeypatch.setattr(OPERATOR, "PREVIOUS_V13_PYTHON", Path("/usr/bin/python3.12"))
    monkeypatch.setattr(OPERATOR, "PREVIOUS_V13_MAINTENANCE", tmp_path / "maintenance.py")
    monkeypatch.setattr(OPERATOR, "PREVIOUS_V13_PROFILE_READY", tmp_path / "ready-v16/ready-binding.json")
    monkeypatch.setattr(OPERATOR, "PREVIOUS_V13_PROFILE_RUNTIME", tmp_path / "runtime-v10")
    monkeypatch.setattr(OPERATOR, "PREVIOUS_V13_PROFILE_EXECUTE_EVIDENCE", tmp_path / "execute-evidence-v10")
    monkeypatch.setattr(OPERATOR, "PREVIOUS_V13_MAINTENANCE_EVIDENCE", tmp_path / "maintenance-v11")
    monkeypatch.setattr(OPERATOR, "PREVIOUS_V13_PROFILE_CAPTURE", tmp_path / "capture-v10")
    monkeypatch.setattr(OPERATOR, "PREVIOUS_V13_OPERATOR_RESULT", tmp_path / "operator-result-v13")
    monkeypatch.setattr(OPERATOR, "PREVIOUS_V13_ACTUAL_AUDIT", tmp_path / "actual-audit-v13")
    quiet_summary = {
        "sample_count": 27,
        "final_streak_samples": 27,
        "final_streak_span_seconds": 130.0,
        "reset_count": 0,
        "confirmation_passed": True,
        "fresh_outputs_absent": True,
    }
    quiet_sha = "9" * 64
    monkeypatch.setattr(OPERATOR, "PREVIOUS_QUIET_V18_JSON_SHA256", quiet_sha)
    monkeypatch.setattr(
        OPERATOR,
        "previous_quiet_v18_authority",
        lambda: {
            "artifact_commit": OPERATOR.PREVIOUS_QUIET_V18_COMMIT,
            "artifact_tree": OPERATOR.PREVIOUS_QUIET_V18_TREE,
            "root_tree": OPERATOR.PREVIOUS_QUIET_V18_ROOT_TREE,
            "json_sha256": quiet_sha,
            "status": "go",
            "decision": "GO",
            "summary": quiet_summary,
            "inventory": {},
        },
    )
    paths = OPERATOR.previous_operator_v13_fresh_paths()
    argv = OPERATOR.previous_operator_v13_argv()
    manifest = {
        "schema_version": OPERATOR.PREVIOUS_OPERATOR_V13_SCHEMA,
        "status": "audited_ready_for_single_explicit_profile_diagnostic",
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
        "inputs": {
            "quiet_window": {
                "path": str(OPERATOR.PREVIOUS_QUIET_V18_ROOT / "quiet-window.json"),
                "sha256": quiet_sha,
                "decision": "GO",
                "status": "go",
            },
            "previous_operator_v12": {
                "state": "authorized_sealed",
                "authorization_commit": OPERATOR.PREVIOUS_OPERATOR_V12_COMMIT,
            },
            "previous_actual_v12": {
                "state": "executed_sealed",
                "artifact_commit": OPERATOR.PREVIOUS_ACTUAL_V12_COMMIT,
                "artifact_tree": OPERATOR.PREVIOUS_ACTUAL_V12_TREE,
                "file_count": OPERATOR.PREVIOUS_ACTUAL_V12_FILE_COUNT,
                "invocation_count": 1,
                "maximum_invocations": 1,
                "retry_performed": False,
            },
        },
        "fresh_outputs": [{"path": str(path), "absent": True} for path in paths],
        "quiet_final_streak": quiet_summary,
        "actual_executed": False,
        "gpu_command_executed": False,
        "service_touched": False,
        "secret_material_embedded": False,
        "manifest_sha256": None,
    }
    manifest["manifest_sha256"] = OPERATOR.sha_bytes(OPERATOR.canonical(manifest))
    sealed(root, "command-manifest.json", manifest)
    inventory = OPERATOR.verify_sums(root)
    commit = "c" * 40
    tree = "d" * 40
    root_tree = "e" * 40
    monkeypatch.setattr(OPERATOR, "PREVIOUS_OPERATOR_V13_COMMIT", commit)
    monkeypatch.setattr(OPERATOR, "PREVIOUS_OPERATOR_V13_TREE", tree)
    monkeypatch.setattr(OPERATOR, "PREVIOUS_OPERATOR_V13_ROOT_TREE", root_tree)
    monkeypatch.setattr(OPERATOR, "PREVIOUS_OPERATOR_V13_MANIFEST_SHA256", OPERATOR.sha_file(root / "command-manifest.json"))
    monkeypatch.setattr(OPERATOR, "PREVIOUS_OPERATOR_V13_SEMANTIC_SHA256", manifest["manifest_sha256"])
    monkeypatch.setattr(OPERATOR, "PREVIOUS_OPERATOR_V13_SUMS_SHA256", inventory["sha256sums_sha256"])
    monkeypatch.setattr(OPERATOR, "PREVIOUS_OPERATOR_V13_COMMAND_SHA256", manifest["command_sha256"])
    monkeypatch.setattr(OPERATOR, "verify_inventory_commit", lambda *_args: None)
    expected = "\n".join(
        (
            "operator-command-v13/SHA256SUMS",
            "operator-command-v13/command-manifest.json",
        )
    )

    def previous_git(*args: str) -> str:
        if args == ("rev-parse", f"{commit}^{{tree}}"):
            return tree
        if args == ("rev-parse", f"{commit}:operator-command-v13"):
            return root_tree
        if args[:3] == ("ls-tree", "-r", "--name-only"):
            return expected if args[-1] == "operator-command-v13" else ""
        raise AssertionError(f"unexpected Git query: {args}")

    monkeypatch.setattr(OPERATOR, "git", previous_git)
    return paths


@pytest.mark.parametrize("path_index", range(9))
def test_previous_operator_v13_ignores_later_worktree_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_index: int,
) -> None:
    paths = previous_v13_fixture(tmp_path, monkeypatch)
    path = paths[path_index]
    if path_index in (4, 5, 6):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"partial")
    else:
        path.mkdir(parents=True, exist_ok=True)
    state = OPERATOR.previous_authorization_v13_state()
    assert state["state"] == "authorized_not_invoked_preflight_blocked"
    assert state["invocation_count"] == 0
    assert all(item["present"] is False for item in state["fresh_outputs"])


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
    manifest = OPERATOR.load(
        paths["OPERATOR_ROOT"] / "command-manifest.json",
        "operator manifest",
    )
    assert validated["result"]["schema_version"] == OPERATOR.OPERATOR_RESULT_SCHEMA
    assert validated["audit"]["schema_version"] == OPERATOR.ACTUAL_AUDIT_SCHEMA
    assert validated["result"]["status"] == ("passed" if succeeded else "failed")
    assert validated["result"]["maximum_invocations"] == 1
    assert validated["result"]["invocation_count"] == 1
    assert validated["result"]["shell"] is False
    assert validated["result"]["retry_performed"] is False
    assert validated["result"]["manifest_semantic_sha256"] == manifest["manifest_sha256"]
    assert validated["result"]["command_sha256"] == manifest["command_sha256"]
    assert validated["audit"]["authority_commit"] == validated["result"]["operator_manifest_commit"]
    assert validated["audit"]["manifest_file_sha256"] == validated["result"]["manifest_file_sha256"]
    assert validated["audit"]["finalizer_authority"] == validated["result"]["finalizer_authority"]
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


def test_actual_command_is_exactly_one_non_shell_profile_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(OPERATOR, "MAINTENANCE_EVIDENCE", tmp_path / "maintenance-v12")
    monkeypatch.setattr(OPERATOR, "PROFILE_RUNTIME", tmp_path / "runtime-v11")
    monkeypatch.setattr(OPERATOR, "PROFILE_EXECUTE_EVIDENCE", tmp_path / "execute-evidence-v11")
    monkeypatch.setattr(OPERATOR, "PROFILE_CAPTURE", tmp_path / "capture-v11")
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
        str(tmp_path / "maintenance-v12"),
        "--confirm-one-case",
    ]
    assert len(argv) == 10
    assert argv.count("--confirm-one-case") == 1


def test_v15_actual_namespace_rejects_previous_v14_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        OPERATOR,
        "MAINTENANCE_EVIDENCE",
        OPERATOR.PREVIOUS_ACTUAL_V14_MAINTENANCE_EVIDENCE,
    )
    monkeypatch.setattr(OPERATOR, "PROFILE_RUNTIME", tmp_path / "runtime-v11")
    monkeypatch.setattr(OPERATOR, "PROFILE_EXECUTE_EVIDENCE", tmp_path / "execute-evidence-v11")
    monkeypatch.setattr(OPERATOR, "PROFILE_CAPTURE", tmp_path / "capture-v11")
    with pytest.raises(OPERATOR.OperatorError, match="overlaps historical v14"):
        OPERATOR.current_v15_actual_roots()


def test_v15_actual_namespace_rejects_any_unbound_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(OPERATOR, "PROFILE_CAPTURE", None)
    with pytest.raises(OPERATOR.OperatorError, match="namespace is unbound"):
        OPERATOR.current_v15_actual_roots()


def test_prepare_and_validate_operator_self_hash_and_restore_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    quiet_root = tmp_path / "quiet"
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
    monkeypatch.setattr(OPERATOR, "CURRENT_V15_AUTHORITY_BOUND", True)
    monkeypatch.setattr(OPERATOR, "QUIET_ROOT", quiet_root)
    monkeypatch.setattr(OPERATOR, "MAINTENANCE_EVIDENCE", tmp_path / "maintenance-v12")
    monkeypatch.setattr(OPERATOR, "PROFILE_RUNTIME", tmp_path / "runtime-v11")
    monkeypatch.setattr(OPERATOR, "PROFILE_EXECUTE_EVIDENCE", tmp_path / "execute-evidence-v11")
    monkeypatch.setattr(OPERATOR, "PROFILE_CAPTURE", tmp_path / "capture-v11")
    monkeypatch.setattr(OPERATOR, "OPERATOR_RESULT", tmp_path / "operator-result-v15")
    monkeypatch.setattr(OPERATOR, "ACTUAL_AUDIT", tmp_path / "actual-audit-v15")
    monkeypatch.setattr(OPERATOR, "ready_authority", lambda: (ready, {"root": "ready-v16"}))
    monkeypatch.setattr(OPERATOR, "fresh_paths", lambda _ready: fresh)
    monkeypatch.setattr(OPERATOR, "current_fresh_paths", lambda: fresh)
    previous_v13 = {
        "state": "authorized_not_invoked_preflight_blocked",
        "reason": "external_owner_after_seal_before_invocation",
        "authorization_commit": OPERATOR.PREVIOUS_OPERATOR_V13_COMMIT,
        "authorization_tree": OPERATOR.PREVIOUS_OPERATOR_V13_TREE,
        "authorization_root_tree": OPERATOR.PREVIOUS_OPERATOR_V13_ROOT_TREE,
        "manifest_file_sha256": OPERATOR.PREVIOUS_OPERATOR_V13_MANIFEST_SHA256,
        "manifest_semantic_sha256": OPERATOR.PREVIOUS_OPERATOR_V13_SEMANTIC_SHA256,
        "command_sha256": OPERATOR.PREVIOUS_OPERATOR_V13_COMMAND_SHA256,
        "maximum_invocations": 1,
        "invocation_count": 0,
        "result_present": False,
        "audit_present": False,
        "actual_executed": False,
        "gpu_command_executed": False,
        "service_touched": False,
        "fresh_outputs": [
            {"path": f"previous-fresh-{index}", "present": False}
            for index in range(9)
        ],
        "quiet_v18": {
            "artifact_commit": OPERATOR.PREVIOUS_QUIET_V18_COMMIT,
            "root_tree": OPERATOR.PREVIOUS_QUIET_V18_ROOT_TREE,
            "json_sha256": OPERATOR.PREVIOUS_QUIET_V18_JSON_SHA256,
        },
    }
    monkeypatch.setattr(
        OPERATOR,
        "previous_authorization_v13_state",
        lambda: previous_v13,
    )
    previous_actual_v14 = {
        "state": "executed_sealed_failure_restore_passed",
        "artifact_commit": OPERATOR.PREVIOUS_ACTUAL_V14_COMMIT,
        "artifact_tree": OPERATOR.PREVIOUS_ACTUAL_V14_TREE,
        "journal_commit": OPERATOR.PREVIOUS_ACTUAL_V14_JOURNAL_COMMIT,
        "journal_tree": OPERATOR.PREVIOUS_ACTUAL_V14_JOURNAL_TREE,
        "file_count": OPERATOR.PREVIOUS_ACTUAL_V14_FILE_COUNT,
        "returncode": 1,
        "invocation_count": 1,
        "maximum_invocations": 1,
        "retry_performed": False,
        "restore_passed": True,
        "previous_operator_v14": {
            "authorization_commit": OPERATOR.PREVIOUS_OPERATOR_V14_COMMIT,
        },
    }
    monkeypatch.setattr(
        OPERATOR,
        "previous_actual_v14_state",
        lambda: previous_actual_v14,
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
    assert value["inputs"]["previous_operator_v13_historical"]["state"] == "authorized_not_invoked_preflight_blocked"
    assert value["inputs"]["previous_operator_v13_historical"]["reason"] == "external_owner_after_seal_before_invocation"
    assert value["inputs"]["previous_actual_v14"]["state"] == "executed_sealed_failure_restore_passed"
    assert value["inputs"]["historical_ready_v15"]["artifact_commit"] == OPERATOR.HISTORICAL_READY_V15_COMMIT
    assert value["inputs"]["offline_reassembly_v12"]["artifact_commit"] == OPERATOR.OFFLINE_ARTIFACT_COMMIT
    assert value["pre_execution_audit"]["previous_operator_v13_historical"] == "authorized_not_invoked_preflight_blocked"
    assert value["pre_execution_audit"]["previous_operator_v13_reason"] == "external_owner_after_seal_before_invocation"
    assert value["pre_execution_audit"]["previous_actual_v14"] == "executed_sealed_failure_restore_passed"
    clone = json.loads(json.dumps(value))
    declared = clone["manifest_sha256"]
    clone["manifest_sha256"] = None
    assert declared == OPERATOR.sha_bytes(OPERATOR.canonical(clone))

    tampered = json.loads(json.dumps(value))
    tampered["inputs"]["previous_operator_v13_historical"]["reason"] = "different"
    tampered["manifest_sha256"] = None
    tampered["manifest_sha256"] = OPERATOR.sha_bytes(OPERATOR.canonical(tampered))
    tampered_root = tmp_path / "operator-v15-tampered"
    sealed(tampered_root, "command-manifest.json", tampered)
    with pytest.raises(OPERATOR.OperatorError, match="previous/final-state binding"):
        OPERATOR.validate_operator(tampered_root)

    tampered_fresh = json.loads(json.dumps(value))
    tampered_fresh["fresh_outputs"][0]["path"] = str(tmp_path / "substitute")
    tampered_fresh["manifest_sha256"] = None
    tampered_fresh["manifest_sha256"] = OPERATOR.sha_bytes(
        OPERATOR.canonical(tampered_fresh)
    )
    tampered_fresh_root = tmp_path / "operator-v15-fresh-tampered"
    sealed(tampered_fresh_root, "command-manifest.json", tampered_fresh)
    with pytest.raises(OPERATOR.OperatorError, match="authorization/safety"):
        OPERATOR.validate_operator(tampered_fresh_root)

    fresh[0].mkdir()
    assert OPERATOR.validate_operator(output_root)["value"] == value


def test_prepare_operator_requires_historical_v13_final_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(OPERATOR, "CURRENT_V15_AUTHORITY_BOUND", True)
    monkeypatch.setattr(OPERATOR, "MAINTENANCE_EVIDENCE", tmp_path / "maintenance-v12")
    monkeypatch.setattr(OPERATOR, "PROFILE_RUNTIME", tmp_path / "runtime-v11")
    monkeypatch.setattr(OPERATOR, "PROFILE_EXECUTE_EVIDENCE", tmp_path / "execute-evidence-v11")
    monkeypatch.setattr(OPERATOR, "PROFILE_CAPTURE", tmp_path / "capture-v11")
    monkeypatch.setattr(OPERATOR, "ready_authority", lambda: ({}, {}))
    monkeypatch.setattr(OPERATOR, "validate_quiet", lambda _root: {"value": {}})
    monkeypatch.setattr(
        OPERATOR,
        "previous_authorization_v13_state",
        lambda: {"state": "authorized_sealed"},
    )
    with pytest.raises(OPERATOR.OperatorError, match="historical operator-v13 final state"):
        OPERATOR.prepare_operator(tmp_path / "operator-v15")


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
