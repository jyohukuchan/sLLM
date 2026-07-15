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
        "capture_helpers": CAPTURE.capture_helper_contract(),
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
        on_rocprof_started=lambda: starts.append("rocprof"),
    )
    environment = json.loads(observed.read_text(encoding="utf-8"))
    assert environment["ULLM_EXACT_BASE"] == "bound"
    assert "HOME" not in environment and "LD_PRELOAD" not in environment
    assert starts == ["rocprof"]


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
        with pytest.raises(CAPTURE.CaptureError, match="identity changed"):
            snapshots[1].verify()
    finally:
        CAPTURE.close_target_snapshots(snapshots)

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
        CAPTURE.close_target_snapshots(snapshots)


def test_target_swap_between_verify_and_spawn_executes_only_pinned_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted = (tmp_path / "trusted-runner.py").resolve()
    replacement = (tmp_path / "replacement-runner.py").resolve()
    observed = (tmp_path / "observed.txt").resolve()
    trusted.write_text(
        "import pathlib,sys\npathlib.Path(sys.argv[1]).write_text('trusted')\n",
        encoding="utf-8",
    )
    replacement.write_text(
        "import pathlib,sys\npathlib.Path(sys.argv[1]).write_text('replacement')\n",
        encoding="utf-8",
    )
    interpreter = Path(sys.executable).resolve()
    manifest = (tmp_path / "fd-target.json").resolve()
    value = write_target_manifest(
        manifest,
        [str(interpreter), str(trusted), str(observed)],
        input_indices=(0, 1),
        output_indices=(2,),
    )
    loaded, snapshots = CAPTURE.load_target_command_manifest(
        manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    effective, target_fds = CAPTURE.pinned_target_argv(loaded, snapshots)
    assert loaded["argv"] == value["argv"]
    assert effective[:2] != loaded["argv"][:2]
    fake_profiler = (tmp_path / "rocprofv3").resolve()
    fake_profiler.write_text(
        "#!/usr/bin/python3\n"
        "import subprocess,sys\n"
        "target=sys.argv[sys.argv.index('--')+1:]\n"
        "fds=tuple(int(item.rsplit('/',1)[1]) for item in target if item.startswith('/proc/self/fd/'))\n"
        "raise SystemExit(subprocess.run(target,pass_fds=fds).returncode)\n",
        encoding="utf-8",
    )
    fake_profiler.chmod(0o555)
    backup = tmp_path / "trusted-backup.py"
    real_popen = CAPTURE.subprocess.Popen

    def swapping_popen(*args, **kwargs):
        trusted.rename(backup)
        replacement.rename(trusted)
        try:
            process = real_popen(*args, **kwargs)
            deadline = time.monotonic() + 5.0
            while not observed.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert observed.read_text(encoding="utf-8") == "trusted"
            return process
        finally:
            trusted.rename(replacement)
            backup.rename(trusted)

    monkeypatch.setattr(CAPTURE.subprocess, "Popen", swapping_popen)
    output = (tmp_path / "fd-profile-output").resolve()
    logical = [str(fake_profiler), "--", *loaded["argv"]]
    effective_command = [str(fake_profiler), "--", *effective]
    try:
        with pytest.raises(CAPTURE.CaptureError, match="post-spawn.*identity changed"):
            CAPTURE.run_profile(
                effective_command,
                output,
                10.0,
                pass_fds=target_fds,
                verifier=lambda: [snapshot.verify() for snapshot in snapshots],
                logical_command=logical,
            )
        assert observed.read_text(encoding="utf-8") == "trusted"
        failure = json.loads((output / "capture-failure.json").read_text())
        assert failure["command_sha256"] == hashlib.sha256(CAPTURE.canonical(logical)).hexdigest()
        assert failure["effective_command_sha256"] == hashlib.sha256(CAPTURE.canonical(effective_command)).hexdigest()
        assert failure["command_sha256"] != failure["effective_command_sha256"]
    finally:
        CAPTURE.close_target_snapshots(snapshots)


def test_runtime_directory_and_lock_swap_execute_through_pinned_path_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = (tmp_path / "bundle").resolve()
    bundle.mkdir()
    (bundle / "payload.txt").write_text("trusted-bundle", encoding="utf-8")
    replacement_bundle = tmp_path / "replacement-bundle"
    replacement_bundle.mkdir()
    (replacement_bundle / "payload.txt").write_text("replacement-bundle", encoding="utf-8")
    lock = (tmp_path / "runner.lock").resolve()
    lock.write_text("trusted-lock", encoding="utf-8")
    replacement_lock = tmp_path / "replacement.lock"
    replacement_lock.write_text("replacement-lock", encoding="utf-8")
    observed = (tmp_path / "runtime-observed.json").resolve()
    runner = (tmp_path / "runtime-runner.py").resolve()
    runner.write_text(
        "import json,os,pathlib,sys\n"
        "bundle=pathlib.Path(sys.argv[1]); lock=pathlib.Path(sys.argv[2]); observed=pathlib.Path(sys.argv[3])\n"
        "fd=os.open(lock,os.O_RDWR); os.lseek(fd,0,os.SEEK_SET); os.write(fd,b'pinned-lock '); os.close(fd)\n"
        "observed.write_text(json.dumps({'payload':(bundle/'payload.txt').read_text()}))\n",
        encoding="utf-8",
    )
    interpreter = Path(sys.executable).resolve()
    manifest = (tmp_path / "runtime-fd-target.json").resolve()
    value = write_target_manifest(
        manifest,
        [str(interpreter), str(runner), str(bundle), str(lock), str(observed)],
        input_indices=(0, 1),
        output_indices=(4,),
    )
    value["runtime_paths"] = [
        {"argument_index": 2, "path": str(bundle), "kind": "directory", "identity": list(CAPTURE.PROFILER._identity(bundle.lstat()))},
        {"argument_index": 3, "path": str(lock), "kind": "regular_file", "identity": list(CAPTURE.PROFILER._identity(lock.lstat()))},
    ]
    value["manifest_sha256"] = CAPTURE.self_hash(value, "manifest_sha256")
    manifest.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    loaded, snapshots = CAPTURE.load_target_command_manifest(
        manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    effective, target_fds = CAPTURE.pinned_target_argv(loaded, snapshots)
    assert all(effective[index].startswith("/proc/self/fd/") for index in (0, 1, 2, 3))
    fake_profiler = (tmp_path / "runtime-rocprofv3").resolve()
    fake_profiler.write_text(
        "#!/usr/bin/python3\n"
        "import subprocess,sys\n"
        "target=sys.argv[sys.argv.index('--')+1:]\n"
        "fds=tuple(int(item.rsplit('/',1)[1]) for item in target if item.startswith('/proc/self/fd/'))\n"
        "raise SystemExit(subprocess.run(target,pass_fds=fds).returncode)\n",
        encoding="utf-8",
    )
    fake_profiler.chmod(0o555)
    bundle_backup = tmp_path / "bundle-backup"
    lock_backup = tmp_path / "lock-backup"
    real_popen = CAPTURE.subprocess.Popen

    def swapping_popen(*args, **kwargs):
        bundle.rename(bundle_backup)
        replacement_bundle.rename(bundle)
        lock.rename(lock_backup)
        replacement_lock.rename(lock)
        try:
            process = real_popen(*args, **kwargs)
            deadline = time.monotonic() + 5.0
            while not observed.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert json.loads(observed.read_text())["payload"] == "trusted-bundle"
            assert lock_backup.read_text().startswith("pinned-lock")
            assert lock.read_text() == "replacement-lock"
            return process
        finally:
            bundle.rename(replacement_bundle)
            bundle_backup.rename(bundle)
            lock.rename(replacement_lock)
            lock_backup.rename(lock)

    monkeypatch.setattr(CAPTURE.subprocess, "Popen", swapping_popen)
    try:
        with pytest.raises(CAPTURE.CaptureError, match="post-spawn.*identity changed"):
            CAPTURE.run_profile(
                [str(fake_profiler), "--", *effective],
                (tmp_path / "runtime-fd-output").resolve(),
                10.0,
                pass_fds=target_fds,
                verifier=lambda: [snapshot.verify() for snapshot in snapshots],
            )
        assert json.loads(observed.read_text())["payload"] == "trusted-bundle"
        assert lock.read_text().startswith("pinned-lock")
    finally:
        CAPTURE.close_target_snapshots(snapshots)


def test_helper_swap_executes_verified_bytes_and_preserves_execution_hash(tmp_path: Path) -> None:
    helper = (tmp_path / "helper.py").resolve()
    replacement = (tmp_path / "replacement-helper.py").resolve()
    helper.write_text("VALUE = 'trusted'\n", encoding="utf-8")
    replacement.write_text("VALUE = 'replacement'\n", encoding="utf-8")
    expected_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
    pinned = CAPTURE.PinnedPythonHelper.open(helper, expected_sha)
    backup = tmp_path / "helper-backup.py"
    try:
        helper.rename(backup)
        replacement.rename(helper)
        module = pinned.load("verified_helper_swap_test")
        assert module.VALUE == "trusted"
        helper.rename(replacement)
        backup.rename(helper)
        with pytest.raises(CAPTURE.CaptureError, match="helper identity changed"):
            pinned.verify()
        assert pinned.evidence("test")["sha256"] == expected_sha
    finally:
        os.close(pinned.descriptor)


@pytest.mark.parametrize(
    ("source", "expected_sha", "module_attribute", "expected_value", "inject"),
    [
        (CAPTURE.SELECTOR_PATH, CAPTURE.SELECTOR_SHA256, "RAW_SCHEMA", CAPTURE.SELECTOR.RAW_SCHEMA, False),
        (CAPTURE.PROFILE_HELPER_PATH, CAPTURE.PROFILE_HELPER_SHA256, "ARTIFACT_SCHEMA", CAPTURE.PROFILER.ARTIFACT_SCHEMA, False),
        (CAPTURE.PRODUCER_PATH, CAPTURE.PRODUCER_SHA256, "RAW_SCHEMA", CAPTURE.PRODUCER.RAW_SCHEMA, True),
    ],
)
def test_transitive_helper_swap_never_executes_replacement_bytes(
    tmp_path: Path,
    source: Path,
    expected_sha: str,
    module_attribute: str,
    expected_value: str,
    inject: bool,
) -> None:
    helper = (tmp_path / source.name).resolve()
    replacement = (tmp_path / f"replacement-{source.name}").resolve()
    helper.write_bytes(source.read_bytes())
    replacement.write_text("raise RuntimeError('replacement helper executed')\n", encoding="utf-8")
    pinned = CAPTURE.PinnedPythonHelper.open(helper, expected_sha)
    backup = tmp_path / f"backup-{source.name}"
    try:
        helper.rename(backup)
        replacement.rename(helper)
        injected = {"selector": CAPTURE.SELECTOR, "profiler": CAPTURE.PROFILER} if inject else None
        module = pinned.load(f"verified_{source.stem}_swap_test", injected_modules=injected)
        assert getattr(module, module_attribute) == expected_value
        if inject:
            assert module.SELECTOR is CAPTURE.SELECTOR
            assert module.PROFILER is CAPTURE.PROFILER
        helper.rename(replacement)
        backup.rename(helper)
        with pytest.raises(CAPTURE.CaptureError, match="helper identity changed"):
            pinned.verify()
    finally:
        os.close(pinned.descriptor)


def test_capture_helper_closure_is_exact_and_reuses_verified_modules() -> None:
    assert [item["role"] for item in CAPTURE.capture_helper_contract()] == [
        "selection_raw_producer",
        "candidate_selector",
        "profile_family_classifier",
    ]
    assert CAPTURE.PRODUCER.SELECTOR is CAPTURE.SELECTOR
    assert CAPTURE.PRODUCER.PROFILER is CAPTURE.PROFILER
    with pytest.raises(CAPTURE.CaptureError, match="module injection differs"):
        CAPTURE.PRODUCER_HELPER.load(
            "invalid_dependency_closure",
            injected_modules={"selector": CAPTURE.SELECTOR},
        )


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
        CAPTURE.close_target_snapshots(snapshots)


def test_profile_timeout_and_oom_exit_fail_closed(tmp_path: Path) -> None:
    sleeper = tmp_path / "sleep-profiler"
    sleeper.write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n", encoding="utf-8"
    )
    sleeper.chmod(0o555)
    timeout_output = (tmp_path / "timeout").resolve()
    timeout_logical = ["/trusted/rocprofv3", "--", "/trusted/runner"]
    with pytest.raises(CAPTURE.CaptureError, match="timed out"):
        CAPTURE.run_profile([str(sleeper)], timeout_output, 0.05, logical_command=timeout_logical)
    timeout_failure = json.loads((timeout_output / "capture-failure.json").read_text())
    assert timeout_failure["command_sha256"] == hashlib.sha256(CAPTURE.canonical(timeout_logical)).hexdigest()

    oom = tmp_path / "oom-profiler"
    oom.write_text("#!/usr/bin/env python3\nraise SystemExit(137)\n", encoding="utf-8")
    oom.chmod(0o555)
    oom_output = (tmp_path / "oom").resolve()
    oom_logical = ["/trusted/rocprofv3", "--", "/trusted/runner", "--one-case"]
    with pytest.raises(CAPTURE.CaptureError, match="possible OOM"):
        CAPTURE.run_profile([str(oom)], oom_output, 10.0, logical_command=oom_logical)
    oom_failure = json.loads((oom_output / "capture-failure.json").read_text())
    assert oom_failure["command_sha256"] == hashlib.sha256(CAPTURE.canonical(oom_logical)).hexdigest()


def test_post_capture_assembly_failure_evidence_keeps_logical_command_binding(tmp_path: Path) -> None:
    output = tmp_path / "post-capture-assembly-failure"
    output.mkdir()
    logical = ["/trusted/rocprofv3", "--", "/trusted/python", "/trusted/runner.py"]
    effective = ["/proc/self/fd/11", "--", "/proc/self/fd/12", "/proc/self/fd/13"]
    CAPTURE.write_failure_evidence(
        output,
        "synthetic assemble failure",
        logical,
        {"stage": "assemble"},
        effective_command=effective,
    )
    failure = json.loads((output / "capture-failure.json").read_text())
    assert failure["command_sha256"] == hashlib.sha256(CAPTURE.canonical(logical)).hexdigest()
    assert failure["effective_command_sha256"] == hashlib.sha256(CAPTURE.canonical(effective)).hexdigest()
    assert failure["context"] == {"stage": "assemble"}
    assert (output / "capture-failure.json").stat().st_mode & 0o777 == 0o444
    unknown = tmp_path / "cleanup-unknown"
    unknown.mkdir()
    CAPTURE.write_failure_evidence(unknown, "process group cleanup failed", logical, None)
    unknown_failure = json.loads((unknown / "capture-failure.json").read_text())
    assert unknown_failure["process_group_cleanup_complete"] is False
    assert unknown_failure["children_state_known"] is False
    assert unknown_failure["children_remaining"] == []


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
        assert failure["children_state_known"] is True
        assert failure["children_remaining"] == []
        assert failure["failure_sha256"] == CAPTURE.self_hash(failure, "failure_sha256")
        restore_request.write_text("restore", encoding="utf-8")
        assert outer.wait(timeout=5.0) == 0
        assert restored.read_text() == "complete"
    finally:
        if outer.poll() is None:
            outer.terminate()
            outer.wait(timeout=5.0)


def test_success_exit_with_live_descendant_is_cleaned_and_fails_closed(tmp_path: Path) -> None:
    child_pid = tmp_path / "escaped-child.pid"
    profiler = tmp_path / "early-exit-profiler"
    profiler.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib,subprocess,sys\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
        f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))\n",
        encoding="utf-8",
    )
    profiler.chmod(0o555)
    output = (tmp_path / "early-exit-output").resolve()
    with pytest.raises(CAPTURE.CaptureError, match="descendants were terminated"):
        CAPTURE.run_profile([str(profiler)], output, 10.0)
    pid = int(child_pid.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    failure = json.loads((output / "capture-failure.json").read_text())
    assert failure["children_state_known"] is True
    assert failure["children_remaining"] == []
    assert failure["process_group_cleanup_complete"] is True


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
    assert artifact_path.stat().st_mode & 0o777 == 0o444
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
    capture_output = tmp_path / "profile-capture"
    value["profile_diagnostic"]["output"] = {"directory": str(capture_output), "artifact": str(capture_output / "capture-artifact.json")}
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
            CAPTURE.close_target_snapshots(snapshots)
        on_started()
        write_launcher_profile_result(result_path, binding)
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
    assert evidence["profile_diagnostics"]["capture_artifact"]["mode"] == 0o444
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
        "rocprof_started": False,
        "runner_start_known": True,
        "runner_completed": False,
        "cleanup_passed": True,
        "children_state_known": True,
        "children_remaining": [],
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
        "rocprof_started": False,
        "runner_start_known": True,
        "runner_completed": False,
        "cleanup_passed": True,
        "children_state_known": True,
        "children_remaining": [],
    }
    assert evidence["process_counts"]["runner"] == 0
    assert evidence["safety"]["execution_state_source"] == "runner_not_started"
    assert not result_path.exists()
