from __future__ import annotations

import importlib.util
import json
import os
import sys
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
    with pytest.raises(CAPTURE.CaptureError, match="timed out"):
        CAPTURE.run_profile([str(profiler)], (tmp_path / "stubborn").resolve(), 0.1)
    pid = int(child_pid.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


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
