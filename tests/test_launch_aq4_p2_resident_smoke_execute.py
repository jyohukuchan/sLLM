from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/launch-aq4-p2-resident-smoke.py"
SPEC = importlib.util.spec_from_file_location("aq4_p2_execute_launcher", SCRIPT)
assert SPEC and SPEC.loader
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)
TRUSTED_LAUNCHER_SHA = LAUNCHER.sha_bytes(SCRIPT.read_bytes())


def _validator_success() -> bytes:
    return b'{"promotion": false, "run_id": "p2-r9700-resident-one-case-smoke-binding-v6", "status": "prepared_not_executed"}\n'


def _ready_binding(tmp_path: Path) -> tuple[dict, Path, Path, str]:
    evidence = tmp_path / "execute-evidence"
    result = tmp_path / "execute-result"
    run_id = "execute-test-run"
    value = json.loads(json.dumps(LAUNCHER.execute_binding_document()))
    value.update(status="ready_for_explicit_execute", actual_eligible=True, blocked_reasons=[], evidence_output=str(evidence), runner_output=str(result), run_id=run_id)
    value["live_preflight"] = {"required": True, "path": str(evidence / "live-preflight.json"), "sha256": None, "replaces_synthetic_preflight": True}
    return value, evidence, result, run_id


def _profile_binding(tmp_path: Path) -> tuple[dict, Path, Path, str]:
    evidence = tmp_path / "profile-execute-evidence"
    result = tmp_path / "profile-execute-result"
    run_id = "profile-execute-test-run"
    value = json.loads(json.dumps(LAUNCHER.profile_execute_binding_document()))
    value.update(status="ready_for_explicit_execute", actual_eligible=True, blocked_reasons=[], evidence_output=str(evidence), runner_output=str(result), run_id=run_id)
    capture_output = tmp_path / "profile-capture"
    value["profile_diagnostic"]["output"] = {"directory": str(capture_output), "artifact": str(capture_output / "capture-artifact.json")}
    value["live_preflight"] = {"required": True, "path": str(evidence / "live-preflight.json"), "sha256": None, "replaces_synthetic_preflight": True}
    return value, evidence, result, run_id


def _write_profile_result(root: Path, binding: dict) -> None:
    session_id = "profile-session-test"
    root.mkdir()
    raw = {
        "case_id": LAUNCHER.CASE_ID,
        "case_sha256": LAUNCHER.CASE_SHA,
        "baseline_identity": {"run_id": binding["run_id"]},
        "resident": {"session_id": session_id},
        "execution_mode": "one_case_smoke",
        "promotion_eligible": False,
    }
    (root / f"{LAUNCHER.CASE_ID}.raw.json").write_text(json.dumps(raw) + "\n")
    (root / "resident-batch.summary.json").write_text("{}\n")
    ranges = []
    for index in range(12):
        kind = "warmup" if index < 2 else "measured"
        name = (
            f"ullm.aq4_p2.run.v1/run_id={binding['run_id']}/session_id={session_id}/"
            f"case_id={LAUNCHER.CASE_ID}/case_sha256={LAUNCHER.CASE_SHA}/"
            f"run_index={index}/run_kind={kind}"
        )
        ranges.append({"name": name, "run_index": index, "run_kind": kind, "push_result": 0, "pop_result": 0})
    sidecar = {
        "schema_version": "ullm.aq4_p2_resident_roctx_ranges.v1",
        "status": "complete_diagnostic",
        "measurement_eligible": False,
        "promotion_eligible": False,
        "audit_sha256": None,
        "pid": 123,
        "thread_id": 456,
        "library": {**binding["profile_diagnostic"]["roctx_library"], "components": []},
        "ranges": ranges,
    }
    sidecar["audit_sha256"] = LAUNCHER.sha_bytes(LAUNCHER.canonical(sidecar))
    (root / "resident-batch.roctx-ranges.json").write_text(json.dumps(sidecar, sort_keys=True) + "\n")


def _gates() -> dict:
    commands = LAUNCHER.expected_live_probe_contracts()
    return {
        "passed": True,
        "environment": LAUNCHER.EXECUTE_ENV,
        "services": [
            {"unit": "ullm-openai.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0},
            {"unit": "llama-qwen35-udq4.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0},
        ],
        "old_worker_pids": [],
        "runtime_mapping": {"runtime_device_index": 1, "visible_token": "1", "amd_smi_index": 2, "bdf": LAUNCHER.GPU_BDF, "uuid": LAUNCHER.GPU_UUID, "kfd_id": LAUNCHER.KFD_ID, "node_id": 2},
        "amd_smi_owners": [], "kfd_owners": [],
        "lock": {"path": str(LAUNCHER.LOCK_PATH), "free": True, "device": 66306, "inode": 123},
        "vram": {"total_bytes": 32_624_000_000, "used_bytes": 0, "free_bytes": 32_624_000_000, "headroom_bytes": 32_624_000_000},
        "probes": [
            {"label": label, "argv": argv, "exit_code": exit_code, "stdout_sha256": "0" * 64, "stderr_sha256": "0" * 64, "captured_unix_ns": index}
            for index, (label, (argv, exit_code)) in enumerate(commands.items())
        ],
    }


def test_execute_bound_generates_live_sidecar_and_exact_runner_argv(tmp_path: Path) -> None:
    binding, evidence_path, result_path, run_id = _ready_binding(tmp_path)
    calls: list[list[str]] = []
    restores: list[bool] = []

    def validator(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, _validator_success(), b"")

    def runner(command: list[str], environment: dict[str, str], on_started):
        assert environment == LAUNCHER.EXECUTE_ENV
        assert command == LAUNCHER.execute_runner_argv(binding)
        assert command[-8:] == [
            "--driver-command", str(LAUNCHER.RESIDENT_DRIVER), "--served-model-manifest", str(LAUNCHER.SERVED_MANIFEST),
            "--device-index", "1", "--build-git-commit", LAUNCHER.RESIDENT_COMMIT,
        ]
        result_path.mkdir()
        (result_path / "case.raw.json").write_text("{}\n")
        (result_path / "resident-batch.summary.json").write_text("{}\n")
        on_started()
        return {"completed": subprocess.CompletedProcess(command, 0, b"", b""), "keepalives": [{"label": "sudo-keepalive-1", "argv": [str(LAUNCHER.SUDO), "-n", "-v"], "exit_code": 0}], "keepalive_failed": False, "gpu_command_executed": True, "model_load_executed": True}

    def restore() -> dict:
        restores.append(True)
        return {"required": False, "service_stop_performed": False, "state_preserved": True}

    code, evidence = LAUNCHER.execute_bound(binding, evidence_path, result_path, run_id, trusted_launcher_sha=TRUSTED_LAUNCHER_SHA, run=validator, gate_provider=_gates, restore_provider=restore, runner_executor=runner)
    assert code == 0
    assert len(calls) == 1 and restores == [True]
    assert evidence["status"] == "passed"
    assert evidence["sequence"] == ["validator", "pre-exec-gates", "runner"]
    assert evidence["process_counts"]["runner"] == 1
    assert evidence["safety"]["gpu_command_executed"] is True
    assert evidence["safety"]["model_load_executed"] is True
    assert evidence["sudo_keepalive"]["failed"] is False
    live = evidence_path / "live-preflight.json"
    assert live.stat().st_mode & 0o777 == 0o444
    value = json.loads(live.read_text())
    assert value["prepared_preflight"]["role"] == "synthetic_bundle_contract_only"
    assert value["runtime_mapping"]["amd_smi_index"] == 2
    assert value["compute_owners"] == {"amd_smi": [], "kfd": []}
    assert value["environment"] == LAUNCHER.EXECUTE_ENV
    assert evidence["result"]["files"] == {"case.raw.json": LAUNCHER.sha_bytes(b"{}\n"), "resident-batch.summary.json": LAUNCHER.sha_bytes(b"{}\n")}


def test_profile_diagnostic_runner_argv_is_exact_and_normal_argv_is_unchanged(tmp_path: Path) -> None:
    normal, _, _, _ = _ready_binding(tmp_path / "normal")
    profile, _, _, _ = _profile_binding(tmp_path / "profile")
    normal_argv = LAUNCHER.execute_runner_argv(normal)
    profile_argv = LAUNCHER.execute_runner_argv(profile)
    assert "--profile-roctx-ranges" not in normal_argv
    assert "--roctx-library" not in normal_argv
    driver_index = profile_argv.index("--driver-command")
    assert profile_argv[driver_index - 5:driver_index] == [
        "--profile-roctx-ranges",
        "--roctx-library",
        str(LAUNCHER.ROCTX_LIBRARY),
        "--roctx-library-sha256",
        LAUNCHER.ROCTX_LIBRARY_SHA,
    ]
    stripped = profile_argv[:driver_index - 5] + profile_argv[driver_index:]
    expected = list(normal_argv)
    expected[expected.index(normal["runner_output"])] = profile["runner_output"]
    expected[expected.index(normal["run_id"])] = profile["run_id"]
    expected[expected.index(str(Path(normal["evidence_output"]) / "live-preflight.json"))] = str(Path(profile["evidence_output"]) / "live-preflight.json")
    assert stripped == expected


def test_profile_roctx_sdk_authority_and_generic_runner_cli_are_exact() -> None:
    library = LAUNCHER.ROCTX_LIBRARY
    metadata = library.lstat()
    assert library == Path("/opt/rocm-7.2.1/lib/librocprofiler-sdk-roctx.so.1.1.0")
    assert library.resolve(strict=True) == LAUNCHER.ROCTX_LIBRARY_RESOLVED == library
    assert stat.S_ISREG(metadata.st_mode) and not library.is_symlink()
    assert stat.S_IMODE(metadata.st_mode) == LAUNCHER.ROCTX_LIBRARY_MODE == 0o644
    assert metadata.st_nlink == 1
    assert metadata.st_size == LAUNCHER.ROCTX_LIBRARY_BYTES == 456232
    assert LAUNCHER.sha_bytes(library.read_bytes()) == LAUNCHER.ROCTX_LIBRARY_SHA == "1a5831a3817eac29f63d1442dc348ba31b417202b7ce15f3aed9c09a8f4773c9"
    dynamic = subprocess.run(["/usr/bin/readelf", "-d", str(library)], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert dynamic.returncode == 0 and dynamic.stderr == b""
    assert f"Library soname: [{LAUNCHER.ROCTX_LIBRARY_SONAME}]".encode() in dynamic.stdout

    normal = LAUNCHER.execute_runner_argv(LAUNCHER.execute_binding_document())
    profile = LAUNCHER.execute_runner_argv(LAUNCHER.profile_execute_binding_document())
    assert "--profile-roctx-ranges" not in normal and "--roctx-library" not in normal
    assert profile.count("--profile-roctx-ranges") == profile.count("--roctx-library") == profile.count("--roctx-library-sha256") == 1
    index = profile.index("--profile-roctx-ranges")
    assert profile[index:index + 5] == ["--profile-roctx-ranges", "--roctx-library", str(library), "--roctx-library-sha256", LAUNCHER.ROCTX_LIBRARY_SHA]


def test_profile_execute_cli_builds_only_the_exact_ready_profile_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[tuple[dict, Path, Path, str, str]] = []

    def fake_execute(binding, evidence, result, run_id, *, trusted_launcher_sha, **kwargs):
        observed.append((binding, evidence, result, run_id, trusted_launcher_sha))
        return 0, {"status": "passed", "mode": "execute"}

    monkeypatch.setattr(LAUNCHER, "execute_bound", fake_execute)
    code = LAUNCHER.main([
        "--mode", "profile-execute",
        "--evidence-output", str(LAUNCHER.PROFILE_EVIDENCE_OUTPUT),
        "--runner-output", str(LAUNCHER.PROFILE_RUN_OUTPUT),
        "--run-id", LAUNCHER.PROFILE_RUN_ID,
        "--trusted-launcher-sha", TRUSTED_LAUNCHER_SHA,
    ])
    assert code == 0 and len(observed) == 1
    binding, evidence, result, run_id, trusted_sha = observed[0]
    assert binding == LAUNCHER.ready_profile_execute_binding()
    assert evidence == LAUNCHER.PROFILE_EVIDENCE_OUTPUT
    assert result == LAUNCHER.PROFILE_RUN_OUTPUT
    assert run_id == LAUNCHER.PROFILE_RUN_ID and trusted_sha == TRUSTED_LAUNCHER_SHA


def test_profile_execute_cli_rejects_noncanonical_target_before_execute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(LAUNCHER, "execute_bound", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not execute")))
    code = LAUNCHER.main([
        "--mode", "profile-execute",
        "--evidence-output", str(tmp_path / "wrong"),
        "--runner-output", str(LAUNCHER.PROFILE_RUN_OUTPUT),
        "--run-id", LAUNCHER.PROFILE_RUN_ID,
        "--trusted-launcher-sha", TRUSTED_LAUNCHER_SHA,
    ])
    assert code == 1


def test_execute_bound_profile_diagnostic_validates_exact_roctx_evidence(tmp_path: Path) -> None:
    binding, evidence_path, result_path, run_id = _profile_binding(tmp_path)

    def validator(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, _validator_success(), b"")

    def runner(command: list[str], environment: dict[str, str], on_started, target):
        assert command == LAUNCHER.execute_runner_argv(binding)
        assert target["path"].endswith(LAUNCHER.PROFILE_RUNNER_TARGET_MANIFEST_NAME)
        _write_profile_result(result_path, binding)
        on_started()
        artifact_path = Path(binding["profile_diagnostic"]["output"]["artifact"])
        artifact_path.parent.mkdir()
        artifact_raw = b"{}\n"
        artifact_path.write_bytes(artifact_raw)
        artifact_path.chmod(0o444)
        return {
            "completed": subprocess.CompletedProcess(command, 0, b"", b""),
            "keepalives": [],
            "keepalive_failed": False,
            "gpu_command_executed": True,
                "model_load_executed": True,
                "profile_diagnostics": {
                    "schema_version": "ullm.aq4_p3_profile_executor_diagnostics.v1",
                    "runner_finished": True,
                    "capture_artifact": {"path": str(artifact_path), "sha256": LAUNCHER.sha_bytes(artifact_raw), "mode": 0o444},
                    "failure_evidence": None,
                    "validation_error": None,
                    "executor_exception": None,
                },
                "profile_capture": {
                "status": "complete_diagnostic",
                "runner_profiled": True,
                "validator_profiled": False,
                "gates_profiled": False,
                    "capture_tool_invocations": 1,
                    "rocprof_invocations": 1,
                    "rocprof_started": True,
                    "runner_started": True,
                    "runner_start_known": True,
                    "runner_completed": True,
                    "target_manifest_sha256": target["sha256"],
                    "target_manifest_semantic_sha256": target["manifest_sha256"],
                    "target_argv_sha256": LAUNCHER.sha_bytes(LAUNCHER.canonical(command)),
                    "environment_sha256": LAUNCHER.sha_bytes(LAUNCHER.canonical(environment)),
                    "capture_stdout_sha256": "0" * 64,
                    "capture_stderr_sha256": "0" * 64,
                    "timed_out": False,
                    "cleanup_passed": True,
                    "children_state_known": True,
                    "children_remaining": [],
                },
        }

    code, evidence = LAUNCHER.execute_bound(binding, evidence_path, result_path, run_id, trusted_launcher_sha=TRUSTED_LAUNCHER_SHA, run=validator, gate_provider=_gates, profile_runner_executor=runner)
    assert code == 0
    assert evidence["status"] == "passed"
    assert evidence["profile_diagnostic"]["run_id"] == run_id
    assert evidence["profile_diagnostic"]["resident_session_id"] == "profile-session-test"
    assert evidence["profile_diagnostic"]["ranges"]["count"] == 12
    assert evidence["profile_diagnostic"]["promotion_eligible"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown-field", "runner-without-known-start", "cleanup-contradiction",
        "complete-timeout", "unknown-children-with-cleanup", "unknown-children-list",
    ),
)
def test_profile_capture_summary_rejects_unknown_fields_and_lifecycle_contradictions(
    tmp_path: Path, mutation: str
) -> None:
    binding, evidence_path, result_path, run_id = _profile_binding(tmp_path)

    def validator(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, _validator_success(), b"")

    def runner(command: list[str], environment: dict[str, str], on_rocprof_started, target):
        capture = {
            "status": "complete_diagnostic",
            "runner_profiled": True,
            "validator_profiled": False,
            "gates_profiled": False,
            "capture_tool_invocations": 1,
            "rocprof_invocations": 1,
            "rocprof_started": True,
            "runner_started": True,
            "runner_start_known": True,
            "runner_completed": True,
            "target_manifest_sha256": target["sha256"],
            "target_manifest_semantic_sha256": target["manifest_sha256"],
            "target_argv_sha256": LAUNCHER.sha_bytes(LAUNCHER.canonical(command)),
            "environment_sha256": LAUNCHER.sha_bytes(LAUNCHER.canonical(environment)),
            "capture_stdout_sha256": "0" * 64,
            "capture_stderr_sha256": "0" * 64,
            "timed_out": False,
            "cleanup_passed": True,
            "children_state_known": True,
            "children_remaining": [],
        }
        if mutation == "unknown-field":
            capture["unknown"] = False
        elif mutation == "runner-without-known-start":
            capture["runner_start_known"] = False
        elif mutation == "cleanup-contradiction":
            capture["cleanup_passed"] = False
        elif mutation == "complete-timeout":
            capture["timed_out"] = True
        elif mutation == "unknown-children-with-cleanup":
            capture["children_state_known"] = False
        elif mutation == "unknown-children-list":
            capture["children_state_known"] = False
            capture["cleanup_passed"] = False
            capture["children_remaining"] = [123]
        on_rocprof_started()
        return {
            "completed": subprocess.CompletedProcess(command, 1, b"", b""),
            "keepalives": [],
            "keepalive_failed": False,
            "gpu_command_executed": "unknown",
            "model_load_executed": "unknown",
            "profile_capture": capture,
            "profile_diagnostics": {
                "schema_version": "ullm.aq4_p3_profile_executor_diagnostics.v1",
                "runner_finished": True,
                "capture_artifact": {"path": str(tmp_path / "capture-artifact.json"), "sha256": "0" * 64, "mode": 0o444},
                "failure_evidence": None,
                "validation_error": None,
                "executor_exception": None,
            },
        }

    code, evidence = LAUNCHER.execute_bound(
        binding,
        evidence_path,
        result_path,
        run_id,
        trusted_launcher_sha=TRUSTED_LAUNCHER_SHA,
        run=validator,
        gate_provider=_gates,
        profile_runner_executor=runner,
    )
    assert code == 1
    assert evidence["failure"]["reason"] == "profile capture outcome contract differs"
    assert evidence["failure"]["runner_started"] is False


def test_profile_executor_exception_after_rocprof_start_preserves_unknown_runner_and_children(
    tmp_path: Path,
) -> None:
    binding, evidence_path, result_path, run_id = _profile_binding(tmp_path)

    def executor(_command, _environment, on_rocprof_started, _target):
        on_rocprof_started()
        raise OSError("synthetic executor failure")

    code, evidence = LAUNCHER.execute_bound(
        binding,
        evidence_path,
        result_path,
        run_id,
        trusted_launcher_sha=TRUSTED_LAUNCHER_SHA,
        run=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, _validator_success(), b""),
        gate_provider=_gates,
        profile_runner_executor=executor,
    )
    assert code == 1
    assert evidence["profile_diagnostics"] is None
    assert evidence["failure"] == {
        "stage": "runner",
        "reason": "synthetic executor failure",
        "runner_started": False,
        "rocprof_started": True,
        "runner_start_known": False,
        "runner_completed": False,
        "cleanup_passed": False,
        "children_state_known": False,
        "children_remaining": [],
    }


def test_valid_failure_diagnostics_are_saved_and_copied_to_launcher_failure(tmp_path: Path) -> None:
    binding, evidence_path, result_path, run_id = _profile_binding(tmp_path)
    failure_path = Path(binding["profile_diagnostic"]["output"]["directory"]) / "capture-failure.json"
    failure_path.parent.mkdir()
    failure_raw = b'{"status":"failed"}\n'
    failure_path.write_bytes(failure_raw)
    failure_path.chmod(0o444)

    def executor(command, environment, on_rocprof_started, target):
        on_rocprof_started()
        return {
            "completed": subprocess.CompletedProcess(command, 1, b"", b""),
            "keepalives": [],
            "keepalive_failed": False,
            "gpu_command_executed": "unknown",
            "model_load_executed": "unknown",
            "profile_capture": {
                "status": "failed", "runner_profiled": True, "validator_profiled": False,
                "gates_profiled": False, "capture_tool_invocations": 1, "rocprof_invocations": 1,
                "rocprof_started": True, "runner_start_known": False, "runner_started": False,
                "runner_completed": False, "target_manifest_sha256": target["sha256"],
                "target_manifest_semantic_sha256": target["manifest_sha256"],
                "target_argv_sha256": LAUNCHER.sha_bytes(LAUNCHER.canonical(command)),
                "environment_sha256": LAUNCHER.sha_bytes(LAUNCHER.canonical(environment)),
                "capture_stdout_sha256": "0" * 64, "capture_stderr_sha256": "0" * 64,
                "timed_out": True, "cleanup_passed": False, "children_state_known": False,
                "children_remaining": [],
            },
            "profile_diagnostics": {
                "schema_version": "ullm.aq4_p3_profile_executor_diagnostics.v1",
                "runner_finished": False,
                "capture_artifact": None,
                "failure_evidence": {"path": str(failure_path), "sha256": LAUNCHER.sha_bytes(failure_raw), "mode": 0o444, "reason": "synthetic timeout"},
                "validation_error": "capture failed",
                "executor_exception": None,
            },
        }

    code, evidence = LAUNCHER.execute_bound(
        binding,
        evidence_path,
        result_path,
        run_id,
        trusted_launcher_sha=TRUSTED_LAUNCHER_SHA,
        run=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, _validator_success(), b""),
        gate_provider=_gates,
        profile_runner_executor=executor,
    )
    assert code == 1
    assert evidence["profile_diagnostics"]["failure_evidence"]["path"] == str(failure_path)
    assert evidence["failure"]["rocprof_started"] is True
    assert evidence["failure"]["runner_start_known"] is False
    assert evidence["failure"]["children_state_known"] is False
    assert evidence["failure"]["cleanup_passed"] is False


@pytest.mark.parametrize("evidence_kind", ("capture_artifact", "failure_evidence"))
def test_profile_diagnostics_reject_other_absolute_0444_evidence_paths(
    tmp_path: Path, evidence_kind: str
) -> None:
    binding, evidence_path, result_path, run_id = _profile_binding(tmp_path)
    other = tmp_path / f"other-{evidence_kind}.json"
    other_raw = b"{}\n"
    other.write_bytes(other_raw)
    other.chmod(0o444)

    def executor(command, environment, on_rocprof_started, target):
        complete = evidence_kind == "capture_artifact"
        on_rocprof_started()
        capture = {
            "status": "complete_diagnostic" if complete else "failed",
            "runner_profiled": True, "validator_profiled": False, "gates_profiled": False,
            "capture_tool_invocations": 1, "rocprof_invocations": 1, "rocprof_started": True,
            "runner_start_known": complete, "runner_started": complete, "runner_completed": complete,
            "target_manifest_sha256": target["sha256"], "target_manifest_semantic_sha256": target["manifest_sha256"],
            "target_argv_sha256": LAUNCHER.sha_bytes(LAUNCHER.canonical(command)),
            "environment_sha256": LAUNCHER.sha_bytes(LAUNCHER.canonical(environment)),
            "capture_stdout_sha256": "0" * 64, "capture_stderr_sha256": "0" * 64,
            "timed_out": False, "cleanup_passed": True, "children_state_known": True,
            "children_remaining": [],
        }
        reference = {"path": str(other), "sha256": LAUNCHER.sha_bytes(other_raw), "mode": 0o444}
        diagnostics = {
            "schema_version": "ullm.aq4_p3_profile_executor_diagnostics.v1",
            "runner_finished": complete,
            "capture_artifact": reference if complete else None,
            "failure_evidence": None if complete else {**reference, "reason": "synthetic"},
            "validation_error": None if complete else "synthetic",
            "executor_exception": None,
        }
        return {
            "completed": subprocess.CompletedProcess(command, 0 if complete else 1, b"", b""),
            "keepalives": [], "keepalive_failed": False,
            "gpu_command_executed": True if complete else "unknown",
            "model_load_executed": True if complete else "unknown",
            "profile_capture": capture, "profile_diagnostics": diagnostics,
        }

    code, evidence = LAUNCHER.execute_bound(
        binding, evidence_path, result_path, run_id,
        trusted_launcher_sha=TRUSTED_LAUNCHER_SHA,
        run=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, _validator_success(), b""),
        gate_provider=_gates, profile_runner_executor=executor,
    )
    assert code == 1
    assert evidence["failure"]["reason"] == "profile capture diagnostics evidence differs"
    assert evidence["profile_diagnostics"] is None


def test_profile_execute_rejects_generic_runner_executor_before_evidence(tmp_path: Path) -> None:
    binding, evidence_path, result_path, run_id = _profile_binding(tmp_path)
    with pytest.raises(LAUNCHER.LauncherError, match="generic runner executor is forbidden"):
        LAUNCHER.execute_bound(
            binding,
            evidence_path,
            result_path,
            run_id,
            trusted_launcher_sha=TRUSTED_LAUNCHER_SHA,
            runner_executor=lambda *_args: {},
        )
    assert not evidence_path.exists() and not result_path.exists()


@pytest.mark.parametrize("mutation", ("missing", "audit", "run-id", "library", "range"))
def test_profile_diagnostic_roctx_evidence_fail_closed(tmp_path: Path, mutation: str) -> None:
    binding, _, result_path, _ = _profile_binding(tmp_path)
    _write_profile_result(result_path, binding)
    sidecar_path = result_path / "resident-batch.roctx-ranges.json"
    if mutation == "missing":
        sidecar_path.unlink()
    elif mutation == "run-id":
        raw_path = result_path / f"{LAUNCHER.CASE_ID}.raw.json"
        raw = json.loads(raw_path.read_text()); raw["baseline_identity"]["run_id"] = "wrong"; raw_path.write_text(json.dumps(raw))
    else:
        sidecar = json.loads(sidecar_path.read_text())
        if mutation == "audit":
            sidecar["audit_sha256"] = "0" * 64
        elif mutation == "library":
            sidecar["library"]["sha256"] = "0" * 64
        else:
            sidecar["ranges"][2]["run_kind"] = "warmup"
        sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises((LAUNCHER.LauncherError, FileNotFoundError)):
        LAUNCHER.validate_profile_result(result_path, binding)


def test_keepalive_failure_interrupts_runner_and_finally_restores(tmp_path: Path) -> None:
    binding, evidence_path, result_path, run_id = _ready_binding(tmp_path)
    restored = False

    def validator(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, _validator_success(), b"")

    def failed_runner(command: list[str], environment: dict[str, str], on_started):
        on_started()
        return {"completed": subprocess.CompletedProcess(command, -2, b"partial", b""), "keepalives": [{"label": "sudo-keepalive-1", "argv": [str(LAUNCHER.SUDO), "-n", "-v"], "exit_code": 1}], "keepalive_failed": True, "gpu_command_executed": "unknown", "model_load_executed": "unknown"}

    def restore() -> dict:
        nonlocal restored
        restored = True
        return {"required": True, "service_stop_performed": False, "state_preserved": True, "priority": "restore_before_reporting"}

    code, evidence = LAUNCHER.execute_bound(binding, evidence_path, result_path, run_id, trusted_launcher_sha=TRUSTED_LAUNCHER_SHA, run=validator, gate_provider=_gates, restore_provider=restore, runner_executor=failed_runner)
    assert code == 1 and restored is True
    assert evidence["failure"]["stage"] == "runner"
    assert evidence["failure"]["runner_started"] is True
    assert "keepalive failed" in evidence["failure"]["reason"]
    assert evidence["restore"]["state_preserved"] is True
    assert evidence["safety"]["gpu_command_executed"] == "unknown"
    assert evidence["safety"]["model_load_executed"] == "unknown"


def test_fake_runner_start_failure_keeps_gpu_and_model_flags_false(tmp_path: Path) -> None:
    binding, evidence_path, result_path, run_id = _ready_binding(tmp_path)

    def validator(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, _validator_success(), b"")

    def start_failure(command: list[str], environment: dict[str, str], on_started):
        raise OSError("synthetic spawn failure")

    code, evidence = LAUNCHER.execute_bound(binding, evidence_path, result_path, run_id, trusted_launcher_sha=TRUSTED_LAUNCHER_SHA, run=validator, gate_provider=_gates, runner_executor=start_failure)
    assert code == 1
    assert evidence["process_counts"]["runner"] == 0
    assert evidence["failure"]["runner_started"] is False
    assert evidence["safety"]["gpu_command_executed"] is False
    assert evidence["safety"]["model_load_executed"] is False
    assert evidence["safety"]["execution_state_source"] == "runner_not_started"


def test_fake_runner_midway_failure_records_proven_gpu_and_model_activity(tmp_path: Path) -> None:
    binding, evidence_path, result_path, run_id = _ready_binding(tmp_path)

    def validator(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, _validator_success(), b"")

    def midway_failure(command: list[str], environment: dict[str, str], on_started):
        on_started()
        return {"completed": subprocess.CompletedProcess(command, 9, b"partial", b""), "keepalives": [], "keepalive_failed": False, "gpu_command_executed": True, "model_load_executed": True}

    code, evidence = LAUNCHER.execute_bound(binding, evidence_path, result_path, run_id, trusted_launcher_sha=TRUSTED_LAUNCHER_SHA, run=validator, gate_provider=_gates, runner_executor=midway_failure)
    assert code == 1
    assert evidence["failure"]["runner_started"] is True
    assert evidence["safety"]["gpu_command_executed"] is True
    assert evidence["safety"]["model_load_executed"] is True
    assert "runner-after" in evidence["trust_verifications"]


@pytest.mark.parametrize(
    ("swap_point", "runner_started", "activity"),
    [("validator-before", False, False), ("runner-before", False, False), ("runner-after", True, True), ("finalize-before", True, True)],
)
def test_execute_snapshot_rejects_stage_specific_toctou_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, swap_point: str, runner_started: bool, activity: bool) -> None:
    binding, evidence_path, result_path, run_id = _ready_binding(tmp_path)
    watched = tmp_path / "watched-tool"
    watched.write_bytes(b"trusted")
    replacement = tmp_path / "replacement-tool"
    replacement.write_bytes(b"trusted")
    original_validate = LAUNCHER.validate_execute_constants

    def validate_with_watched(snapshot, self_sha):
        value = original_validate(snapshot, self_sha)
        snapshot.file(watched, LAUNCHER.sha_bytes(b"trusted"), "TOCTOU watched tool")
        return value

    swapped = False

    def hook(point: str):
        nonlocal swapped
        if point == swap_point and not swapped:
            swapped = True
            os.replace(replacement, watched)

    def validator(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, _validator_success(), b"")

    def successful_runner(command: list[str], environment: dict[str, str], on_started):
        on_started()
        result_path.mkdir()
        (result_path / "resident-batch.summary.json").write_text("{}\n")
        return {"completed": subprocess.CompletedProcess(command, 0, b"", b""), "keepalives": [], "keepalive_failed": False, "gpu_command_executed": True, "model_load_executed": True}

    monkeypatch.setattr(LAUNCHER, "validate_execute_constants", validate_with_watched)
    code, evidence = LAUNCHER.execute_bound(binding, evidence_path, result_path, run_id, trusted_launcher_sha=TRUSTED_LAUNCHER_SHA, run=validator, gate_provider=_gates, runner_executor=successful_runner, verification_hook=hook)
    assert code == 1 and swapped is True
    assert evidence["process_counts"]["runner"] == int(runner_started)
    assert evidence["safety"]["gpu_command_executed"] is activity
    assert evidence["safety"]["model_load_executed"] is activity
    assert "replacement" in evidence["failure"]["reason"]


def test_real_runner_wrapper_uses_fake_sudo_keepalive_and_interrupts_on_failure() -> None:
    sudo_calls = 0

    def fake_sudo(argv, **kwargs):
        nonlocal sudo_calls
        sudo_calls += 1
        return subprocess.CompletedProcess(argv, 1, b"", b"")

    started = time.monotonic()
    outcome = LAUNCHER.run_runner_with_sudo_keepalive(
        [sys.executable, "-c", "import time; time.sleep(10)"], dict(os.environ), sudo_run=fake_sudo, interval=0.02,
    )
    completed, records, failed = outcome["completed"], outcome["keepalives"], outcome["keepalive_failed"]
    assert time.monotonic() - started < 2
    assert sudo_calls == 1 and len(records) == 1 and failed is True
    assert completed.returncode != 0
    assert records[0]["argv"] == [str(LAUNCHER.SUDO), "-n", "-v"]
    assert outcome["gpu_command_executed"] == "unknown"
    assert outcome["model_load_executed"] == "unknown"


def test_real_runner_wrapper_keeps_running_with_fake_valid_sudo() -> None:
    def fake_sudo(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    outcome = LAUNCHER.run_runner_with_sudo_keepalive(
        [sys.executable, "-c", "import time; time.sleep(.08)"], dict(os.environ), sudo_run=fake_sudo, interval=0.02,
    )
    completed, records, failed = outcome["completed"], outcome["keepalives"], outcome["keepalive_failed"]
    assert completed.returncode == 0 and failed is False
    assert len(records) >= 1
    assert all(record["argv"] == [str(LAUNCHER.SUDO), "-n", "-v"] for record in records)
    assert outcome["gpu_command_executed"] is True
    assert outcome["model_load_executed"] is True


def test_execute_rejects_output_reuse_before_starting_processes(tmp_path: Path) -> None:
    binding, evidence_path, result_path, run_id = _ready_binding(tmp_path)
    evidence_path.mkdir()
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("no process may start")

    with pytest.raises(LAUNCHER.LauncherError, match="already exists"):
        LAUNCHER.execute_bound(binding, evidence_path, result_path, run_id, trusted_launcher_sha=TRUSTED_LAUNCHER_SHA, run=forbidden, gate_provider=_gates)
    assert calls == 0


def test_execute_binding_remains_ineligible_until_live_sidecar_and_qa() -> None:
    value = LAUNCHER.execute_binding_document()
    assert value["actual_eligible"] is False
    assert value["live_preflight"]["sha256"] is None
    assert value["tools"]["sudo"]["prevalidate_argv"] == [str(LAUNCHER.SUDO), "-n", "-v"]
    assert value["blocked_reasons"] == ["live preflight sidecar is absent", "independent execute-launcher QA is pending"]


def test_canonical_execute_binding_fails_closed_until_sdk_launcher_is_recascaded() -> None:
    with pytest.raises(LAUNCHER.LauncherError, match="launcher self differs"):
        LAUNCHER.load_execute_binding(LAUNCHER.EXECUTE_BINDING_PATH)
    binding = json.loads(LAUNCHER.EXECUTE_BINDING_PATH.read_text())
    trust = json.loads(LAUNCHER.EXECUTE_LAUNCHER_TRUST_PATH.read_text())
    assert binding["actual_eligible"] is False
    assert trust["actual_eligible"] is False and trust["status"] == "qa_pending"
    committed = subprocess.run(
        ["git", "show", f'{trust["commit"]}:tools/launch-aq4-p2-resident-smoke.py'],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    assert LAUNCHER.sha_bytes(committed.stdout) == trust["sha256"]
    assert committed.stdout != SCRIPT.read_bytes()


def test_execute_rejects_untrusted_launcher_self_before_output_creation(tmp_path: Path) -> None:
    binding, evidence_path, result_path, run_id = _ready_binding(tmp_path)
    with pytest.raises(LAUNCHER.LauncherError, match="self differs"):
        LAUNCHER.execute_bound(binding, evidence_path, result_path, run_id, trusted_launcher_sha="0" * 64, gate_provider=_gates)
    assert not evidence_path.exists() and not result_path.exists()


def test_execute_binding_parent_chain_creation_rejects_symlink(tmp_path: Path) -> None:
    nested = tmp_path / "new" / "deep" / "parent"
    LAUNCHER.ensure_directory_chain(nested, "test parent")
    assert nested.is_dir()
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(LAUNCHER.LauncherError, match="symlink component"):
        LAUNCHER.ensure_directory_chain(alias / "child", "test parent")


def _gate_router(*, duplicate_bdf: bool = False, active_service: bool = False):
    target = {"gpu": 2, "bdf": LAUNCHER.GPU_BDF, "uuid": LAUNCHER.GPU_UUID, "kfd_id": LAUNCHER.KFD_ID, "node_id": 2, "partition_id": 0}
    other = {"gpu": 0, "bdf": "0000:01:00.0", "uuid": "other", "kfd_id": 1, "node_id": 0, "partition_id": 0}
    if duplicate_bdf:
        other["bdf"] = LAUNCHER.GPU_BDF

    def run(argv, **kwargs):
        if argv == [str(LAUNCHER.SUDO), "-n", "-v"]:
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        if argv[:2] == [str(LAUNCHER.SYSTEMCTL), "show"]:
            stdout = b"ActiveState=active\nSubState=running\nMainPID=99\n" if active_service else b"ActiveState=inactive\nSubState=dead\nMainPID=0\n"
            return subprocess.CompletedProcess(argv, 0, stdout, b"")
        if argv[0] == str(LAUNCHER.PGREP):
            return subprocess.CompletedProcess(argv, 1, b"", b"")
        if argv[0] == str(LAUNCHER.AMD_SMI) and argv[1] == "list":
            return subprocess.CompletedProcess(argv, 0, json.dumps([other, target]).encode(), b"")
        if argv[0] == str(LAUNCHER.ROCMINFO):
            stdout = b"Name:                    gfx1201\nUuid:                    GPU-a8e9ddefa2d60f55\nMarketing Name:          AMD Radeon Graphics\n"
            return subprocess.CompletedProcess(argv, 0, stdout, b"")
        if argv[0] == str(LAUNCHER.AMD_SMI) and argv[1] == "process":
            return subprocess.CompletedProcess(argv, 0, b'[{"gpu":2,"process_list":[{"process_info":"No running processes detected"}]}]', b"")
        if argv[0] == str(LAUNCHER.AMD_SMI) and argv[1] == "static":
            return subprocess.CompletedProcess(argv, 0, b'{"gpu_data": [{"gpu": 2, "vram": {"size": {"value": 32624, "unit": "MB"}}}]}', b"")
        raise AssertionError(argv)

    return run


def test_collect_execute_gates_uses_order_independent_unique_gpu_mapping_and_no_owners() -> None:
    lock = {"path": str(LAUNCHER.LOCK_PATH), "free": True, "device": 66306, "inode": 123}
    gates = LAUNCHER.collect_execute_gates(run=_gate_router(), environment=dict(LAUNCHER.EXECUTE_ENV), kfd_owner_provider=lambda: [], lock_provider=lambda: lock)
    assert gates["passed"] is True
    assert gates["runtime_mapping"] == {"runtime_device_index": 1, "visible_token": "1", "amd_smi_index": 2, "bdf": LAUNCHER.GPU_BDF, "uuid": LAUNCHER.GPU_UUID, "kfd_id": LAUNCHER.KFD_ID, "node_id": 2}
    assert gates["amd_smi_owners"] == gates["kfd_owners"] == []
    assert gates["amd_smi_process"]["reason_code"] == "accepted_zero_sentinel"
    assert len(gates["amd_smi_process"]["raw_sha256"]) == 64
    assert gates["probes"][0]["argv"] == [str(LAUNCHER.SUDO), "-n", "-v"]


def _active_amd_process(pid: int = 4101820) -> bytes:
    return json.dumps([{
        "gpu": 2,
        "process_list": [{"process_info": {
            "name": "/home/homelab1/coding-local/ultimateLLM/uLLM-project/target/reasoning-v2/release/ullm-aq4-worker",
            "pid": pid,
            "mem_usage": {"value": 7_351_832_576, "unit": "B"},
            "cu_occupancy": "N/A",
            "evicted_time": {"value": 682, "unit": "ms"},
        }}],
    }]).encode()


def test_strict_amd_process_parser_accepts_live_active_and_exact_zero_sentinel() -> None:
    active = LAUNCHER.parse_amd_process_owners(_active_amd_process())
    assert active["owners"] == [4101820]
    assert active["diagnostic"]["reason_code"] == "accepted_owner_records"
    sentinel = b'[{"gpu":2,"process_list":[{"process_info":"No running processes detected"}]}]'
    zero = LAUNCHER.parse_amd_process_owners(sentinel)
    assert zero["owners"] == []
    assert zero["diagnostic"] == {
        "schema_version": "ullm.aq4_p2_amd_process_parse_diagnostic.v1",
        "status": "accepted",
        "reason_code": "accepted_zero_sentinel",
        "raw_sha256": LAUNCHER.sha_bytes(sentinel),
        "raw_bytes": len(sentinel),
        "top_level_type": "list",
        "top_level_length": 1,
        "root_keys": ["gpu", "process_list"],
        "process_list_type": "list",
        "process_list_length": 1,
        "entry_key_sets": [["process_info"]],
        "process_info_types": ["string"],
    }


@pytest.mark.parametrize(
    ("value", "reason_code"),
    [
        ([{"gpu": 2, "process_list": []}], "process_list_not_nonempty_list"),
        ([{"gpu": 2, "process_list": [{"process_info": "N/A"}]}], "sentinel_mixed_or_unknown"),
        ([{"gpu": 2, "process_list": [{"process_info": "No running processes detected"}, {"process_info": {}}]}], "sentinel_mixed_or_unknown"),
        ([{"gpu": 1, "process_list": [{"process_info": "No running processes detected"}]}], "gpu_index_differs"),
        ([{"gpu": 2, "process_list": [{"process_info": "No running processes detected"}], "extra": 1}], "gpu_root_keys_differ"),
    ],
)
def test_strict_amd_process_parser_rejects_malformed_zero_variants(value: object, reason_code: str) -> None:
    raw = json.dumps(value).encode()
    with pytest.raises(LAUNCHER.AmdProcessSchemaError) as caught:
        LAUNCHER.parse_amd_process_owners(raw)
    assert caught.value.diagnostic["reason_code"] == reason_code
    assert caught.value.diagnostic["raw_sha256"] == LAUNCHER.sha_bytes(raw)
    assert "ullm-aq4-worker" not in LAUNCHER.canonical(caught.value.diagnostic).decode()


@pytest.mark.parametrize(
    ("mutate", "reason_code"),
    [
        (lambda info: info.update(pid=True), "process_pid_differs"),
        (lambda info: info.update(extra=1), "process_info_keys_differ"),
        (lambda info: info["mem_usage"].update(unit="MiB"), "process_mem_usage_differs"),
    ],
)
def test_strict_amd_process_parser_rejects_malformed_active_records(mutate, reason_code: str) -> None:
    value = json.loads(_active_amd_process())
    mutate(value[0]["process_list"][0]["process_info"])
    with pytest.raises(LAUNCHER.AmdProcessSchemaError) as caught:
        LAUNCHER.parse_amd_process_owners(json.dumps(value).encode())
    assert caught.value.diagnostic["reason_code"] == reason_code


def test_collect_execute_gates_rejects_active_service_duplicate_mapping_and_kfd_owner() -> None:
    with pytest.raises(LAUNCHER.LauncherError, match="service is not inactive"):
        LAUNCHER.collect_execute_gates(run=_gate_router(active_service=True), environment=dict(LAUNCHER.EXECUTE_ENV), kfd_owner_provider=lambda: [], lock_provider=lambda: {})
    with pytest.raises(LAUNCHER.LauncherError, match="unique identity"):
        LAUNCHER.collect_execute_gates(run=_gate_router(duplicate_bdf=True), environment=dict(LAUNCHER.EXECUTE_ENV), kfd_owner_provider=lambda: [], lock_provider=lambda: {})
    with pytest.raises(LAUNCHER.LauncherError, match="KFD compute owners"):
        LAUNCHER.collect_execute_gates(run=_gate_router(), environment=dict(LAUNCHER.EXECUTE_ENV), kfd_owner_provider=lambda: [123], lock_provider=lambda: {})


def _kfd_fixture(root: Path, pid: int = 123, gpuid: bytes = b"51545") -> Path:
    queue = root / str(pid) / "queues" / "0"
    queue.mkdir(parents=True)
    source = queue / "gpuid"
    source.write_bytes(gpuid)
    return source


def test_kfd_owner_snapshot_records_stable_raw_sources(tmp_path: Path) -> None:
    raw = b"51545"
    _kfd_fixture(tmp_path, gpuid=raw)
    snapshot = LAUNCHER._kfd_owner_snapshot(tmp_path, allowed_owners={123})
    assert snapshot["classification"] == "stable"
    assert snapshot["attempt_count"] == 1
    assert snapshot["owners"] == [123]
    assert snapshot["sources"] == [{"pid": 123, "queue": 0, "raw_sha256": LAUNCHER.sha_bytes(raw), "raw_bytes": len(raw), "line_ending": "none", "parsed_gpuid": LAUNCHER.KFD_ID}]
    assert snapshot["secret_material_recorded"] is False


def test_kfd_owner_snapshot_rescans_followup_enoent_as_disappearance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _kfd_fixture(tmp_path)
    original_open = LAUNCHER.os.open
    injected = False

    def disappearing_open(path, flags, *args, **kwargs):
        nonlocal injected
        if Path(path) == source and not injected:
            injected = True
            source.unlink()
            source.parent.rmdir()
            source.parent.parent.rmdir()
            source.parent.parent.parent.rmdir()
            raise FileNotFoundError(2, "synthetic KFD disappearance")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(LAUNCHER.os, "open", disappearing_open)
    snapshot = LAUNCHER._kfd_owner_snapshot(tmp_path, allowed_owners=set())
    assert injected is True
    assert snapshot["classification"] == "stable_after_disappearance"
    assert snapshot["attempt_count"] == 2
    assert snapshot["attempts"][0] == {"attempt": 0, "classification": "entry_disappeared", "stage": "gpuid_open", "pid": 123}
    assert snapshot["owners"] == []


def test_kfd_owner_snapshot_accepts_single_lf_compatibility_form(tmp_path: Path) -> None:
    raw = b"51545\n"
    _kfd_fixture(tmp_path, gpuid=raw)
    snapshot = LAUNCHER._kfd_owner_snapshot(tmp_path, allowed_owners={123})
    assert snapshot["owners"] == [123]
    assert snapshot["sources"][0]["line_ending"] == "lf"
    assert snapshot["sources"][0]["raw_sha256"] == LAUNCHER.sha_bytes(raw)


@pytest.mark.parametrize(
    ("gpuid", "reason_code"),
    (
        (b"not-a-gpuid", "gpuid_schema_differs"),
        (b"0", "gpuid_schema_differs"),
        (b"051545", "gpuid_schema_differs"),
        (b" 51545", "gpuid_schema_differs"),
        (b"+51545", "gpuid_schema_differs"),
        (b"-51545", "gpuid_schema_differs"),
        (b"51545\r\n", "gpuid_schema_differs"),
        (b"51545\n\n", "gpuid_schema_differs"),
        (b"", "gpuid_schema_differs"),
    ),
)
def test_kfd_owner_snapshot_rejects_malformed_gpuid(tmp_path: Path, gpuid: bytes, reason_code: str) -> None:
    _kfd_fixture(tmp_path, gpuid=gpuid)
    with pytest.raises(LAUNCHER.KfdOwnerScanError) as caught:
        LAUNCHER._kfd_owner_snapshot(tmp_path)
    assert caught.value.diagnostic["reason_code"] == reason_code


def test_kfd_owner_snapshot_rejects_symlink_eacces_and_foreign_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _kfd_fixture(tmp_path)
    source.unlink()
    source.symlink_to(source.parent / "missing-gpuid-target")
    with pytest.raises(LAUNCHER.KfdOwnerScanError) as symlink:
        LAUNCHER._kfd_owner_snapshot(tmp_path)
    assert symlink.value.diagnostic["reason_code"] == "source_type_or_symlink"
    source.unlink()
    source.write_bytes(b"51545")
    original_open = LAUNCHER.os.open

    def denied_open(path, flags, *args, **kwargs):
        if Path(path) == source:
            raise PermissionError(13, "synthetic KFD denial")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(LAUNCHER.os, "open", denied_open)
    with pytest.raises(LAUNCHER.KfdOwnerScanError) as denied:
        LAUNCHER._kfd_owner_snapshot(tmp_path)
    assert denied.value.diagnostic["reason_code"] == "source_os_error"
    assert denied.value.diagnostic["errno_name"] == "EACCES"
    monkeypatch.setattr(LAUNCHER.os, "open", original_open)
    with pytest.raises(LAUNCHER.KfdOwnerScanError) as foreign:
        LAUNCHER._kfd_owner_snapshot(tmp_path, allowed_owners=set())
    assert foreign.value.diagnostic["reason_code"] == "foreign_owner"
    assert foreign.value.diagnostic["pid"] == 123


@pytest.mark.parametrize(
    "mutate",
    [
        lambda gates: gates["runtime_mapping"].update(unknown=1),
        lambda gates: gates["runtime_mapping"].update(node_id=3),
        lambda gates: gates["lock"].update(unknown=1),
        lambda gates: gates["lock"].update(inode=-1),
        lambda gates: gates["vram"].update(total_bytes=1, free_bytes=1, headroom_bytes=1),
        lambda gates: gates["vram"].update(used_bytes=1),
        lambda gates: gates["probes"][0].update(exit_code=1),
        lambda gates: gates["probes"][0].update(stdout_sha256="A" * 64),
        lambda gates: gates["probes"][1].update(label="sudo-n", argv=[str(LAUNCHER.SUDO), "-n", "-v"]),
        lambda gates: gates["probes"][0].update(unknown=1),
    ],
)
def test_launcher_rejects_qa_nested_schema_negatives_before_writing_sidecar(tmp_path: Path, mutate) -> None:
    binding, evidence_path, _, _ = _ready_binding(tmp_path)
    evidence_path.mkdir()
    gates = _gates()
    mutate(gates)
    with pytest.raises(LAUNCHER.LauncherError):
        LAUNCHER.make_live_preflight(binding, gates, evidence_path)
    assert not (evidence_path / "live-preflight.json").exists()
