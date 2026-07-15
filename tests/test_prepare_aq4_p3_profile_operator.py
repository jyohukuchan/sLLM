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
) -> dict:
    paths = {
        "ROOT": tmp_path,
        "PROFILE_READY_ROOT": tmp_path / "profile-ready-v11",
        "PROFILE_READY": tmp_path / "profile-ready-v11/ready-binding.json",
        "QUIET_ROOT": tmp_path / "quiet-v14",
        "OPERATOR_ROOT": tmp_path / "operator-command-v9",
        "MAINTENANCE_EVIDENCE": tmp_path / "maintenance-v8",
        "OPERATOR_RESULT": tmp_path / "operator-result-v9",
        "ACTUAL_AUDIT": tmp_path / "actual-audit-v9",
        "PROFILE_RUNTIME": tmp_path / "runtime-v8",
        "PROFILE_EXECUTE_EVIDENCE": tmp_path / "execute-evidence-v8",
        "PROFILE_CAPTURE": tmp_path / "capture-v8",
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
    maintenance = {
        "status": status,
        "mode": "execute",
        "failure": None if succeeded else {"stage": "profile-capture", "reason": "capture failed", "launcher_started": True},
        "package_integrity": {"full_hash_count": 1, "full_content": {"passed": True}, "integrity_identity": {"passed": True}},
        "restore": {"passed": True, "duration_ns": 1, "final_metadata_recheck": {"within_absolute_deadline": True}},
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
    paths["OPERATOR_RESULT"].mkdir()
    (paths["OPERATOR_RESULT"] / "operator.stdout.bin").write_bytes(OPERATOR.pretty({"status": status, "mode": "execute", "evidence": str(paths["MAINTENANCE_EVIDENCE"] / "launcher-evidence.json")}))
    (paths["OPERATOR_RESULT"] / "operator.stderr.bin").write_bytes(b"" if succeeded else b"maintenance failed\n")
    pre = {"service": {"main_pid": 10}, "worker": {"pid": 20}, "owners": {"amd_smi": [20], "kfd": [20]}, "hashes": {"fixture": "same"}, "formal_health_sha256": "d" * 64}
    post = {"service": {"active_state": "active", "sub_state": "running", "nrestarts": 0, "main_pid": 11}, "worker": {"pid": 21}, "gpu": {"device": "fixture"}, "owners": {"amd_smi": [21], "kfd": [21]}, "lock": {"busy": True}, "hashes": pre["hashes"], "formal_health_sha256": pre["formal_health_sha256"], "targeted_processes": []}
    monkeypatch.setattr(OPERATOR, "validate_operator", lambda _root=paths["OPERATOR_ROOT"]: {"value": manifest})
    monkeypatch.setattr(OPERATOR, "validate_quiet", lambda _root=paths["QUIET_ROOT"]: {"value": {"confirmation": pre}})
    monkeypatch.setattr(OPERATOR, "capture_snapshot", lambda _ready: post)
    monkeypatch.setattr(OPERATOR, "git", lambda *_args: "e" * 40)
    return paths


def test_v9_namespaces_bind_fresh_v11_ready_and_v8_profile_outputs() -> None:
    assert OPERATOR.PROFILE_READY_ROOT.name == "resident-one-case-smoke-profile-ready-v11"
    assert OPERATOR.QUIET_ROOT.name == "resident-one-case-smoke-profile-quiet-window-v14"
    assert OPERATOR.OPERATOR_ROOT.name == "resident-one-case-smoke-profile-operator-command-v9"
    assert OPERATOR.MAINTENANCE_EVIDENCE.name == "resident-one-case-smoke-profile-maintenance-evidence-v8"
    assert OPERATOR.PROFILE_RUNTIME.name == "resident-one-case-smoke-profile-execute-v8"
    assert OPERATOR.PROFILE_EXECUTE_EVIDENCE.name == "resident-one-case-smoke-profile-execute-evidence-v8"
    assert OPERATOR.PROFILE_CAPTURE.name == "aq4-p3-diagnostic-rocprof-capture-v8"
    assert OPERATOR.OPERATOR_RESULT.name == "resident-one-case-smoke-profile-operator-result-v9"
    assert OPERATOR.ACTUAL_AUDIT.name == "resident-one-case-smoke-profile-actual-audit-v9"
    assert OPERATOR.PREVIOUS_OPERATOR_ROOT.name == "resident-one-case-smoke-profile-operator-command-v8"
    assert OPERATOR.EXECUTE_BINDING_ROOT.name == "resident-one-case-smoke-execute-binding-v8"


def test_v11_ready_execute_binding_and_fresh_output_authorities_are_exact() -> None:
    ready, inventory = OPERATOR.ready_authority()
    assert inventory["sha256sums_sha256"] == OPERATOR.READY_SHA256SUMS_SHA256
    assert ready["authorization"]["run_id"] == "p2-r9700-resident-one-case-smoke-profile-diagnostic-v8"
    assert OPERATOR.QUIET_SCHEMA.endswith(".v14")
    assert OPERATOR.OPERATOR_SCHEMA.endswith(".v9")
    assert OPERATOR.OPERATOR_RESULT_SCHEMA.endswith(".v9")
    assert OPERATOR.ACTUAL_AUDIT_SCHEMA.endswith(".v9")
    roots = OPERATOR.root_set()
    assert OPERATOR.EXECUTE_BINDING_ROOT in roots
    assert any(root.name == "resident-one-case-smoke-profile-ready-dry-run-v11" for root in roots)
    execute_inventory = OPERATOR.verify_sums(OPERATOR.EXECUTE_BINDING_ROOT)
    OPERATOR.verify_inventory_commit(
        OPERATOR.EXECUTE_BINDING_ROOT,
        execute_inventory,
        OPERATOR.EXECUTE_BINDING_ARTIFACT_COMMIT,
    )
    fresh = OPERATOR.fresh_paths(ready)
    assert len(fresh) == len({str(path) for path in fresh}) == 9
    historical = OPERATOR.historical_actual_v9_state()
    assert historical["state"] in {"not_executed", "executed_sealed"}
    if historical["state"] == "not_executed":
        assert all(not item["present"] for item in historical["fresh_outputs"])
    else:
        assert historical["artifact_commit"] == OPERATOR.HISTORICAL_ACTUAL_V9_COMMIT
        assert historical["artifact_tree"] == OPERATOR.HISTORICAL_ACTUAL_V9_TREE
        assert historical["file_count"] == OPERATOR.HISTORICAL_ACTUAL_V9_FILE_COUNT
        assert historical["returncode"] == 1
        assert historical["invocation_count"] == historical["maximum_invocations"] == 1
        assert historical["retry_performed"] is False


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
    monkeypatch.setattr(OPERATOR, "ready_authority", lambda: (ready, {"root": "ready-v11"}))
    monkeypatch.setattr(OPERATOR, "fresh_paths", lambda _ready: fresh)

    value = OPERATOR.prepare_operator(output_root)
    validated = OPERATOR.validate_operator(output_root)["value"]
    assert validated == value
    assert value["failure_contract"]["retry_forbidden"] is True
    assert value["failure_contract"]["outer_restore_in_finally"] is True
    assert value["failure_contract"]["restore_timeout_seconds"] == 120.0
    assert value["failure_contract"]["children_remaining_must_be_empty"] is True
    assert value["inputs"]["historical_operator_v8"]["root"] == str(previous_root)
    assert value["pre_execution_audit"]["historical_operator_v8"] == "immutable_readback"
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
