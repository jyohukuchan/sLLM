from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


CAPTURE = load(
    "capture_aq4_p3_diagnostic_profile",
    ROOT / "tools/capture-aq4-p3-diagnostic-profile.py",
)
LAUNCHER = load(
    "launch_aq4_p2_resident_smoke_for_profile_boundary",
    ROOT / "tools/launch-aq4-p2-resident-smoke.py",
)
FIXTURES = load(
    "aq4_p3_producer_test_fixtures",
    ROOT / "tests/test_build_aq4_p3_selection_raw.py",
)


def write_marker_trace(path: Path, run_id: str, case_id: str, case_sha: str) -> None:
    rows = ["Name,Start_Timestamp,End_Timestamp"]
    for index in range(12):
        kind = "warmup" if index < 2 else "measured"
        name = (
            f"{CAPTURE.MARKER_PREFIX}/run_id={run_id}/session_id=fixture-session/"
            f"case_id={case_id}/case_sha256={case_sha}/run_index={index}/run_kind={kind}"
        )
        rows.append(f"{name},{index * 1000 + 100},{index * 1000 + 900}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_source_traces(root: Path, run_id: str, case_id: str, case_sha: str) -> dict[str, Path]:
    marker = root / "diag_marker_api_trace.csv"
    kernel = root / "diag_kernel_trace.csv"
    api = root / "diag_hip_api_trace.csv"
    memory = root / "diag_memory_copy_trace.csv"
    write_marker_trace(marker, run_id, case_id, case_sha)
    kernel_rows = ["Dispatch_Id,Kernel_Name,Start_Timestamp,End_Timestamp"]
    api_rows = ["Correlation_Id,Function,Start_Timestamp,End_Timestamp"]
    memory_rows = ["Correlation_Id,Name,Start_Timestamp,End_Timestamp"]
    for index in range(12):
        base = index * 1000
        kernel_rows.append(f"{index},hip_paged_kv_write_kernel,{base + 200},{base + 300}")
        api_rows.append(f"{index},hipMemcpyDtoHAsync,{base + 310},{base + 350}")
        memory_rows.append(f"{index},D2H,{base + 310},{base + 350}")
    kernel.write_text("\n".join(kernel_rows) + "\n", encoding="utf-8")
    api.write_text("\n".join(api_rows) + "\n", encoding="utf-8")
    memory.write_text("\n".join(memory_rows) + "\n", encoding="utf-8")
    return {"kernel": kernel, "hip_api": api, "memory_copy": memory, "marker": marker}


def resident_evidence(tmp_path: Path):
    identity_path, identity = FIXTURES.identity_fixture(tmp_path)
    summary_path = FIXTURES.summary_fixture(
        tmp_path / "summary.json", identity_path, "diag-run", diagnostic=True
    )
    case_id, case_sha = "diag-case", "8" * 64
    raw_path = FIXTURES.raw_fixture(
        tmp_path / "raw.json",
        identity_path,
        identity,
        "diag-run",
        case_id,
        case_sha,
        128,
        100.0,
        diagnostic=True,
    )
    return identity_path, summary_path, raw_path, case_id, case_sha


def write_target_manifest(
    path: Path,
    argv: list[str],
    *,
    input_indices: tuple[int, ...] = (0,),
    output_indices: tuple[int, ...] = (),
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": CAPTURE.TARGET_SCHEMA,
        "status": "bound",
        "manifest_sha256": None,
        "argv": argv,
        "environment": {"ULLM_TEST_PROFILE_TARGET": "1"},
        "input_files": [
            {
                "argument_index": index,
                "path": argv[index],
                "sha256": hashlib.sha256(Path(argv[index]).read_bytes()).hexdigest(),
                "executable": index == 0,
            }
            for index in input_indices
        ],
        "runtime_paths": [],
        "output_paths": [
            {"argument_index": index, "path": argv[index]} for index in output_indices
        ],
        "authorization": {
            "maximum_invocations": 1,
            "target_role": "profile_runner_only",
            "promotion_eligible": False,
        },
    }
    value["manifest_sha256"] = CAPTURE.self_hash(value, "manifest_sha256")
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return value


def test_profiler_command_enables_all_required_domains(tmp_path: Path) -> None:
    profiler = tmp_path / "rocprofv3"
    profiler.write_bytes(b"fake")
    profiler.chmod(0o555)
    snapshot = CAPTURE.PROFILER.capture(profiler, "profiler", require_executable=True)
    command = CAPTURE.profiler_command(
        snapshot, (tmp_path / "out").resolve(), "diag", ["/bin/true"]
    )
    assert command.count("--kernel-trace") == 1
    assert command.count("--hip-runtime-trace") == 1
    assert command.count("--memory-copy-trace") == 1
    assert command.count("--marker-trace") == 1
    assert command[1:3] == ["--log-level", "error"]
    assert command[-2:] == ["--", "/bin/true"]


def test_fake_rocprof_runs_once_and_discovers_exact_trace_set(tmp_path: Path) -> None:
    fake = tmp_path / "rocprofv3"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "args=sys.argv[1:]\n"
        "out=pathlib.Path(args[args.index('--output-directory')+1])\n"
        "name=args[args.index('--output-file')+1]\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "for suffix in ('kernel_trace','hip_api_trace','memory_copy_trace','marker_api_trace'):\n"
        "    (out/f'{name}_{suffix}.csv').write_text('Name,Start_Timestamp,End_Timestamp\\n')\n",
        encoding="utf-8",
    )
    fake.chmod(0o555)
    snapshot = CAPTURE.PROFILER.capture(fake, "fake rocprof", require_executable=True)
    output = (tmp_path / "capture").resolve()
    command = CAPTURE.profiler_command(snapshot, output, "one", ["/bin/true"])
    CAPTURE.run_profile(command, output, 10.0)
    assert set(CAPTURE.discover(output)) == {"kernel", "hip_api", "memory_copy", "marker"}
    assert (output / "rocprof.stdout").exists()
    with pytest.raises(CAPTURE.CaptureError, match="already exists"):
        CAPTURE.run_profile(command, output, 10.0)
    assert not (output / "capture-failure.json").exists()


def test_run_profile_passes_only_bound_base_environment_and_signals_start_once(
    tmp_path: Path,
) -> None:
    observed = tmp_path / "environment.json"
    profiler = tmp_path / "environment-profiler"
    profiler.write_text(
        "#!/usr/bin/python3\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(os.environ), sort_keys=True))\n",
        encoding="utf-8",
    )
    profiler.chmod(0o555)
    starts: list[str] = []
    output = (tmp_path / "environment-capture").resolve()
    CAPTURE.run_profile(
        [str(profiler), str(observed)],
        output,
        10.0,
        environment={"ULLM_EXACT_BASE": "bound"},
        on_started=lambda: starts.append("runner-wrapper"),
    )
    environment = json.loads(observed.read_text(encoding="utf-8"))
    assert environment["ULLM_EXACT_BASE"] == "bound"
    assert "HOME" not in environment and "LD_PRELOAD" not in environment
    assert starts == ["runner-wrapper"]


def test_pinned_profiler_uses_verified_fd_and_rejects_sha_or_symlink_swap(
    tmp_path: Path,
) -> None:
    profiler = tmp_path / "rocprofv3-real"
    profiler.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "if '--version' in sys.argv:\n"
        "    print('rocprofv3 version: 1.1.0 rocm_version: 7.0.0')\n"
        "    raise SystemExit(0)\n"
        "args=sys.argv[1:]\n"
        "out=pathlib.Path(args[args.index('--output-directory')+1])\n"
        "name=args[args.index('--output-file')+1]\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "for suffix in ('kernel_trace','hip_api_trace','memory_copy_trace','marker_api_trace'):\n"
        "    (out/f'{name}_{suffix}.csv').write_text('Name,Start_Timestamp,End_Timestamp\\n')\n",
        encoding="utf-8",
    )
    profiler.chmod(0o555)
    invocation = tmp_path / "rocprofv3"
    invocation.symlink_to(profiler.name)
    digest = hashlib.sha256(profiler.read_bytes()).hexdigest()

    with pytest.raises(CAPTURE.CaptureError, match="SHA-256 differs"):
        CAPTURE.PinnedProfiler.open(invocation.resolve(strict=False), "0" * 64)

    pinned = CAPTURE.PinnedProfiler.open(invocation, digest)
    try:
        version = CAPTURE.pinned_profiler_version(pinned)
        assert version["version"] == "1.1.0"
        output = (tmp_path / "pinned-output").resolve()
        command = CAPTURE.profiler_command(pinned, output, "pinned", ["/bin/true"])
        assert command[0] == pinned.fd_path
        CAPTURE.run_profile(
            command,
            output,
            10.0,
            pass_fds=(pinned.descriptor,),
            verifier=pinned.verify,
        )
        assert set(CAPTURE.discover(output)) == {
            "kernel",
            "hip_api",
            "memory_copy",
            "marker",
        }

        replacement = tmp_path / "rocprofv3-replacement"
        replacement.write_bytes(profiler.read_bytes())
        replacement.chmod(0o555)
        invocation.unlink()
        invocation.symlink_to(replacement.name)
        with pytest.raises(CAPTURE.CaptureError, match="symlink changed"):
            pinned.verify()
    finally:
        pinned.close()


def test_target_command_manifest_binds_exact_argv_and_input_hashes(tmp_path: Path) -> None:
    executable = tmp_path / "launcher"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o555)
    output = (tmp_path / "runner-output.json").resolve()
    manifest = (tmp_path / "target.json").resolve()
    argv = [str(executable.resolve()), "--output", str(output)]
    value = write_target_manifest(manifest, argv, output_indices=(2,))
    expected_file_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    loaded, snapshots = CAPTURE.load_target_command_manifest(
        manifest, expected_file_sha
    )
    try:
        assert loaded == value
        assert loaded["argv"] == argv
        assert len(snapshots) == 2

        executable.chmod(0o755)
        with pytest.raises(CAPTURE.PROFILER.ProfileError, match="identity changed"):
            snapshots[1].verify()
    finally:
        snapshots[0].close()

    executable.chmod(0o555)
    reordered = write_target_manifest(
        manifest,
        [str(executable.resolve()), str(output), "--output"],
        output_indices=(1,),
    )
    assert reordered["manifest_sha256"] == CAPTURE.self_hash(
        reordered, "manifest_sha256"
    )
    with pytest.raises(CAPTURE.CaptureError, match="file SHA-256 differs"):
        CAPTURE.load_target_command_manifest(manifest, expected_file_sha)

    unbound = (tmp_path / "unbound.json").resolve()
    write_target_manifest(unbound, [str(executable.resolve()), str(output)])
    with pytest.raises(CAPTURE.CaptureError, match="coverage differs"):
        CAPTURE.load_target_command_manifest(
            unbound, hashlib.sha256(unbound.read_bytes()).hexdigest()
        )

    changed_environment = (tmp_path / "changed-environment.json").resolve()
    changed = write_target_manifest(changed_environment, [str(executable.resolve())])
    changed["environment"] = {"ULLM_TEST_PROFILE_TARGET": "changed"}
    changed_environment.write_text(json.dumps(changed, sort_keys=True) + "\n")
    with pytest.raises(CAPTURE.CaptureError, match="self-hash differs"):
        CAPTURE.load_target_command_manifest(
            changed_environment,
            hashlib.sha256(changed_environment.read_bytes()).hexdigest(),
        )


def test_runtime_path_identity_is_pinned_by_target_manifest(tmp_path: Path) -> None:
    executable = tmp_path / "runner"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o555)
    runtime_directory = tmp_path / "bundle"
    runtime_directory.mkdir()
    manifest = (tmp_path / "runtime-target.json").resolve()
    value = write_target_manifest(
        manifest, [str(executable.resolve()), str(runtime_directory.resolve())]
    )
    metadata = runtime_directory.lstat()
    value["runtime_paths"] = [
        {
            "argument_index": 1,
            "path": str(runtime_directory.resolve()),
            "kind": "directory",
            "identity": list(CAPTURE.PROFILER._identity(metadata)),
        }
    ]
    value["manifest_sha256"] = CAPTURE.self_hash(value, "manifest_sha256")
    manifest.write_text(json.dumps(value, sort_keys=True) + "\n")
    loaded, snapshots = CAPTURE.load_target_command_manifest(
        manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    try:
        assert loaded["runtime_paths"] == value["runtime_paths"]
        replacement = tmp_path / "replacement-bundle"
        replacement.mkdir()
        runtime_directory.rmdir()
        replacement.rename(runtime_directory)
        with pytest.raises(CAPTURE.CaptureError, match="identity changed"):
            snapshots[-1].verify()
    finally:
        snapshots[0].close()


def test_post_spawn_manifest_path_swap_emits_failure_and_blocks_success(
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "launcher"
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o555)
    manifest = (tmp_path / "target.json").resolve()
    write_target_manifest(manifest, [str(launcher.resolve())])
    expected_file_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    _value, snapshots = CAPTURE.load_target_command_manifest(
        manifest, expected_file_sha
    )
    pinned_manifest = snapshots[0]
    replacement = tmp_path / "target-replacement.json"
    replacement.write_bytes(manifest.read_bytes())
    swapper = tmp_path / "swap-manifest"
    swapper.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "os.replace(sys.argv[1], sys.argv[2])\n",
        encoding="utf-8",
    )
    swapper.chmod(0o555)
    output_directory = (tmp_path / "profile-output").resolve()
    artifact = tmp_path / "success-artifact.json"
    try:
        with pytest.raises(CAPTURE.CaptureError, match="post-spawn.*identity changed"):
            CAPTURE.run_profile(
                [str(swapper), str(replacement), str(manifest)],
                output_directory,
                10.0,
                verifier=pinned_manifest.verify,
                failure_context={"target_command_manifest": CAPTURE.ref(pinned_manifest)},
            )
        failure_path = output_directory / "capture-failure.json"
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        assert failure_path.stat().st_mode & 0o777 == 0o444
        assert failure["failure_sha256"] == CAPTURE.self_hash(
            failure, "failure_sha256"
        )
        assert failure["promotion_eligible"] is False
        assert not artifact.exists()
        with pytest.raises(CAPTURE.CaptureError, match="failure evidence exists"):
            CAPTURE.assemble(
                output_directory=output_directory,
                artifact_path=artifact,
            )
        assert not artifact.exists()
    finally:
        pinned_manifest.close()


def test_profile_timeout_and_oom_exit_fail_closed(tmp_path: Path) -> None:
    sleeper = tmp_path / "sleep-profiler"
    sleeper.write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n", encoding="utf-8"
    )
    sleeper.chmod(0o555)
    with pytest.raises(CAPTURE.CaptureError, match="timed out"):
        CAPTURE.run_profile([str(sleeper)], (tmp_path / "timeout").resolve(), 0.05)

    oom = tmp_path / "oom-profiler"
    oom.write_text("#!/usr/bin/env python3\nraise SystemExit(137)\n", encoding="utf-8")
    oom.chmod(0o555)
    with pytest.raises(CAPTURE.CaptureError, match="possible OOM"):
        CAPTURE.run_profile([str(oom)], (tmp_path / "oom").resolve(), 10.0)


def test_timeout_kills_sigint_ignoring_child_process_group(tmp_path: Path) -> None:
    sentinel_ready = tmp_path / "outer.ready"
    restore_request = tmp_path / "outer.restore-request"
    restored = tmp_path / "outer.restored"
    outer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import pathlib,time;"
                f"ready=pathlib.Path({str(sentinel_ready)!r});"
                f"request=pathlib.Path({str(restore_request)!r});"
                f"restored=pathlib.Path({str(restored)!r});"
                "ready.write_text('alive');"
                "\nwhile not request.exists(): time.sleep(0.01)\n"
                "restored.write_text('complete')"
            ),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 5.0
    while not sentinel_ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sentinel_ready.exists()
    child_pid = tmp_path / "child.pid"
    profiler = tmp_path / "stubborn-profiler"
    profiler.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, signal, subprocess, sys, time\n"
        "child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(30)'])\n"
        f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))\n"
        "signal.signal(signal.SIGINT, lambda *_: None)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    profiler.chmod(0o555)
    output = (tmp_path / "stubborn").resolve()
    try:
        with pytest.raises(CAPTURE.CaptureError, match="timed out"):
            CAPTURE.run_profile(
                [str(profiler)],
                output,
                0.1,
                failure_context={"outer_harness": "fake-sentinel"},
            )
        pid = int(child_pid.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert outer.poll() is None
        failure = json.loads((output / "capture-failure.json").read_text())
        assert failure["status"] == "failed"
        assert failure["promotion_eligible"] is False
        assert failure["outer_harness_signalled"] is False
        assert failure["process_group_cleanup_complete"] is True
        assert failure["failure_sha256"] == CAPTURE.self_hash(failure, "failure_sha256")
        restore_request.write_text("restore", encoding="utf-8")
        assert outer.wait(timeout=5.0) == 0
        assert restored.read_text() == "complete"
    finally:
        if outer.poll() is None:
            outer.terminate()
            outer.wait(timeout=5.0)


def test_marker_contract_binds_exact_12_runs_and_rejects_missing(tmp_path: Path) -> None:
    case_sha = "8" * 64
    marker = tmp_path / "marker.csv"
    write_marker_trace(marker, "diag-run", "diag-case", case_sha)
    snapshot = CAPTURE.PRODUCER.capture(marker.resolve(), "marker")
    raw = {
        "case_id": "diag-case",
        "case_sha256": case_sha,
        "resident": {"session_id": "fixture-session"},
    }
    ranges = CAPTURE.markers(snapshot, raw, "diag-run")
    assert [item["run_index"] for item in ranges] == list(range(12))
    marker.write_text("\n".join(marker.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(CAPTURE.CaptureError, match="exactly 12"):
        CAPTURE.markers(CAPTURE.PRODUCER.capture(marker.resolve(), "short"), raw, "diag-run")


def test_assemble_splits_measured_runs_and_emits_diagnostic_producer_bindings(
    tmp_path: Path,
) -> None:
    identity_path, summary_path, raw_path, case_id, case_sha = resident_evidence(tmp_path)
    output = tmp_path / "profile"
    output.mkdir()
    traces = write_source_traces(output, "diag-run", case_id, case_sha)
    artifact_path = tmp_path / "capture.json"
    artifact = CAPTURE.assemble(
        traces=traces,
        identity_path=identity_path,
        summary_path=summary_path,
        raw_path=raw_path,
        profiler_value={"tool": "rocprofv3", "version": "1.1.0"},
        command=["rocprofv3", "--", "runner"],
        output_directory=output,
        artifact_path=artifact_path,
    )
    assert artifact["measurement_eligible"] is False
    assert artifact["marker_contract"]["warmup_excluded"] is True
    assert [item["resident_run_index"] for item in artifact["producer_profile_runs"]] == list(
        range(2, 12)
    )
    assert len(artifact["memory_copy_traces"]) == 10
    assert artifact["artifact_sha256"] == CAPTURE.self_hash(artifact, "artifact_sha256")
    assert artifact_path.exists()
    producer_manifest = {
        "schema_version": CAPTURE.PRODUCER.INPUT_SCHEMA,
        "status": "one_case_diagnostic",
        "measurement_eligible": False,
        "smoke_only": True,
        "promotion_eligible": False,
        "manifest_sha256": None,
        "candidate": {
            "candidate_id": "paged-kv-table-validation-v1",
            "family": "paged_validation",
        },
        "identity": FIXTURES.ref(identity_path),
        "resident_summaries": [FIXTURES.ref(summary_path)],
        "representative_cases": [
            {
                "prompt_id": "one-case-profile",
                "case_id": case_id,
                "case_sha256": case_sha,
                "resolved_m": 128,
                "resident_raw": FIXTURES.ref(raw_path),
                "profile_runs": artifact["producer_profile_runs"],
            }
        ],
        "full_model_pairs": [],
    }
    producer_manifest["manifest_sha256"] = CAPTURE.PRODUCER.manifest_sha256(
        producer_manifest
    )
    producer_path = tmp_path / "producer-manifest.json"
    FIXTURES.write_json(producer_path, producer_manifest)
    diagnostic = FIXTURES.build_manifest(producer_path)
    assert diagnostic["status"] == "one_case_diagnostic"
    assert diagnostic["measurement_eligible"] is False
    with pytest.raises(CAPTURE.CaptureError, match="already exists"):
        CAPTURE.assemble(
            traces=traces,
            identity_path=identity_path,
            summary_path=summary_path,
            raw_path=raw_path,
            profiler_value={"tool": "rocprofv3", "version": "1.1.0"},
            command=["rocprofv3", "--", "runner"],
            output_directory=output,
            artifact_path=artifact_path,
        )


@pytest.mark.parametrize(
    ("trace_kind", "old", "new", "message"),
    [
        ("kernel", "hip_paged_kv_write_kernel", "unknown_warmup_kernel", "unknown kernel"),
        ("hip_api", "hipMemcpyDtoHAsync", "hipMemcpyAsync", "unknown transfer"),
    ],
)
def test_warmup_unknown_source_rows_fail_and_partial_outputs_are_cleaned(
    tmp_path: Path, trace_kind: str, old: str, new: str, message: str
) -> None:
    identity_path, summary_path, raw_path, case_id, case_sha = resident_evidence(tmp_path)
    output = tmp_path / "profile"
    output.mkdir()
    traces = write_source_traces(output, "diag-run", case_id, case_sha)
    path = traces[trace_kind]
    rows = path.read_text().splitlines()
    rows[1] = rows[1].replace(old, new)
    path.write_text("\n".join(rows) + "\n")
    artifact_path = tmp_path / "capture.json"
    with pytest.raises((CAPTURE.CaptureError, CAPTURE.PRODUCER.ProducerError), match=message):
        CAPTURE.assemble(
            traces=traces,
            identity_path=identity_path,
            summary_path=summary_path,
            raw_path=raw_path,
            profiler_value={"tool": "rocprofv3", "version": "1.1.0"},
            command=["rocprofv3", "--", "runner"],
            output_directory=output,
            artifact_path=artifact_path,
        )
    assert not (output / "measured-runs").exists()
    assert not (output / "capture-capabilities.json").exists()
    assert not artifact_path.exists()


def launcher_profile_binding(tmp_path: Path) -> tuple[dict, Path, Path, str]:
    evidence = tmp_path / "profile-launcher-evidence"
    result = tmp_path / "profile-runner-output"
    run_id = "profile-boundary-test-run"
    value = json.loads(json.dumps(LAUNCHER.profile_execute_binding_document()))
    value.update(
        status="ready_for_explicit_execute",
        actual_eligible=True,
        blocked_reasons=[],
        evidence_output=str(evidence),
        runner_output=str(result),
        run_id=run_id,
    )
    value["live_preflight"] = {
        "required": True,
        "path": str(evidence / "live-preflight.json"),
        "sha256": None,
        "replaces_synthetic_preflight": True,
    }
    return value, evidence, result, run_id


def launcher_gates() -> dict:
    commands = LAUNCHER.expected_live_probe_contracts()
    return {
        "passed": True,
        "environment": LAUNCHER.EXECUTE_ENV,
        "services": [
            {"unit": unit, "active_state": "inactive", "sub_state": "dead", "main_pid": 0}
            for unit in LAUNCHER.SERVICE_UNITS
        ],
        "old_worker_pids": [],
        "runtime_mapping": {
            "runtime_device_index": 1,
            "visible_token": "1",
            "amd_smi_index": 2,
            "bdf": LAUNCHER.GPU_BDF,
            "uuid": LAUNCHER.GPU_UUID,
            "kfd_id": LAUNCHER.KFD_ID,
            "node_id": 2,
        },
        "amd_smi_owners": [],
        "kfd_owners": [],
        "lock": {"path": str(LAUNCHER.LOCK_PATH), "free": True, "device": 1, "inode": 2},
        "vram": {
            "total_bytes": 32_000_000_000,
            "used_bytes": 0,
            "free_bytes": 32_000_000_000,
            "headroom_bytes": 32_000_000_000,
        },
        "probes": [
            {
                "label": label,
                "argv": argv,
                "exit_code": exit_code,
                "stdout_sha256": "0" * 64,
                "stderr_sha256": "0" * 64,
                "captured_unix_ns": index + 1,
            }
            for index, (label, (argv, exit_code)) in enumerate(commands.items())
        ],
    }


def write_launcher_profile_result(root: Path, binding: dict) -> None:
    session_id = "profile-boundary-session"
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
        ranges.append(
            {
                "name": (
                    f"ullm.aq4_p2.run.v1/run_id={binding['run_id']}/session_id={session_id}/"
                    f"case_id={LAUNCHER.CASE_ID}/case_sha256={LAUNCHER.CASE_SHA}/"
                    f"run_index={index}/run_kind={kind}"
                ),
                "run_index": index,
                "run_kind": kind,
                "push_result": 0,
                "pop_result": 0,
            }
        )
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
    (root / "resident-batch.roctx-ranges.json").write_text(
        json.dumps(sidecar, sort_keys=True) + "\n"
    )


def test_launcher_runs_validator_and_gates_before_profile_runner_only(tmp_path: Path) -> None:
    binding, evidence_path, result_path, run_id = launcher_profile_binding(tmp_path)
    events: list[str] = []

    def validator(argv, **kwargs):
        events.append("validator")
        assert "env" not in kwargs
        return subprocess.CompletedProcess(
            argv,
            0,
            b'{"promotion": false, "run_id": "p2-r9700-resident-one-case-smoke-binding-v4", "status": "prepared_not_executed"}\n',
            b"",
        )

    def gates() -> dict:
        events.append("gates")
        return launcher_gates()

    def profile_executor(command, environment, on_started, target):
        events.append("capture")
        assert environment == LAUNCHER.EXECUTE_ENV
        assert events == ["validator", "gates", "capture"]
        loaded, snapshots = CAPTURE.load_target_command_manifest(
            Path(target["path"]), target["sha256"]
        )
        try:
            assert loaded["argv"] == command
            assert loaded["environment"] == LAUNCHER.EXECUTE_ENV
            assert loaded["authorization"]["target_role"] == "profile_runner_only"
            assert loaded["output_paths"] == [{"argument_index": 19, "path": str(result_path)}]
        finally:
            snapshots[0].close()
        on_started()
        write_launcher_profile_result(result_path, binding)
        return {
            "completed": subprocess.CompletedProcess(command, 0, b"", b""),
            "keepalives": [],
            "keepalive_failed": False,
            "gpu_command_executed": True,
            "model_load_executed": True,
            "profile_capture": {
                "status": "complete_diagnostic",
                "runner_profiled": True,
                "validator_profiled": False,
                "gates_profiled": False,
                "capture_tool_invocations": 1,
                "rocprof_invocations": 1,
                "target_manifest_sha256": target["sha256"],
            },
        }

    trusted = LAUNCHER.sha_bytes((ROOT / "tools/launch-aq4-p2-resident-smoke.py").read_bytes())
    code, evidence = LAUNCHER.execute_bound(
        binding,
        evidence_path,
        result_path,
        run_id,
        trusted_launcher_sha=trusted,
        run=validator,
        gate_provider=gates,
        profile_runner_executor=profile_executor,
    )
    assert code == 0 and evidence["status"] == "passed"
    assert events == ["validator", "gates", "capture"]
    assert evidence["profile_capture"]["runner_profiled"] is True
    assert evidence["profile_capture"]["validator_profiled"] is False
    assert evidence["profile_capture"]["gates_profiled"] is False
    assert evidence["profile_runner_target"]["path"].endswith(
        LAUNCHER.PROFILE_RUNNER_TARGET_MANIFEST_NAME
    )


def test_captured_validator_warning_remains_fail_closed_before_profile_runner(
    tmp_path: Path,
) -> None:
    binding, evidence_path, result_path, run_id = launcher_profile_binding(tmp_path)
    calls: list[str] = []
    # Regression fixture from the failed evidence committed in 4c89c602.
    warning = (
        b"W20260715 13:07:09.605996 139283158812032 simple_timer.cpp:55] "
        b"[rocprofv3] tool initialization ::     0.004369 sec\n"
    )

    def validator(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv,
            0,
            b'{"promotion": false, "run_id": "p2-r9700-resident-one-case-smoke-binding-v4", "status": "prepared_not_executed"}\n',
            warning,
        )

    def forbidden_executor(*_args):
        calls.append("profile-runner")
        raise AssertionError("profile runner must not start")

    trusted = LAUNCHER.sha_bytes((ROOT / "tools/launch-aq4-p2-resident-smoke.py").read_bytes())
    code, evidence = LAUNCHER.execute_bound(
        binding,
        evidence_path,
        result_path,
        run_id,
        trusted_launcher_sha=trusted,
        run=validator,
        gate_provider=launcher_gates,
        profile_runner_executor=forbidden_executor,
    )
    assert code == 1 and calls == []
    assert evidence["failure"] == {
        "stage": "validator",
        "reason": "trusted validator subprocess rejected root/B",
        "runner_started": False,
    }
    assert evidence["safety"]["gpu_command_executed"] is False
    assert evidence["safety"]["model_load_executed"] is False


def test_profile_launcher_without_capture_executor_never_starts_runner(tmp_path: Path) -> None:
    binding, evidence_path, result_path, run_id = launcher_profile_binding(tmp_path)

    def validator(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv,
            0,
            b'{"promotion": false, "run_id": "p2-r9700-resident-one-case-smoke-binding-v4", "status": "prepared_not_executed"}\n',
            b"",
        )

    trusted = LAUNCHER.sha_bytes((ROOT / "tools/launch-aq4-p2-resident-smoke.py").read_bytes())
    code, evidence = LAUNCHER.execute_bound(
        binding,
        evidence_path,
        result_path,
        run_id,
        trusted_launcher_sha=trusted,
        run=validator,
        gate_provider=launcher_gates,
    )
    assert code == 1
    assert evidence["failure"] == {
        "stage": "runner",
        "reason": "profile runner executor is required",
        "runner_started": False,
    }
    assert evidence["process_counts"]["runner"] == 0
    assert evidence["safety"]["execution_state_source"] == "runner_not_started"
    assert not result_path.exists()
