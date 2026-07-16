from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


ASSEMBLER = load_tool(
    "aq4_p3_direct_trace_assembler",
    ROOT / "tools/assemble-aq4-p3-candidate-a-direct-trace.py",
)
PRODUCER = load_tool(
    "aq4_p3_direct_trace_parser",
    ROOT / "tools/build-aq4-p3-selection-raw.py",
)


COMMON = {
    "request_id": "request-1",
    "implementation_id": "qwen35-aq4-direct-v1",
    "source_id": "runtime-route-apply",
    "source_sha256": "a" * 64,
    "candidate_id": "sequence-output-direct-v1",
    "case_id": "case-1",
    "case_sha256": "b" * 64,
    "identity_sha256": "c" * 64,
}


def seal(value: dict[str, object], field: str) -> dict[str, object]:
    value[field] = ASSEMBLER.self_hash(value, field)
    return value


def write_json(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")
    return path


def runtime(side: str, *, binding_kind: str = "run", binding_id: str = "run-1") -> dict[str, object]:
    candidate = side == "candidate"
    counters = {
        "invocation_count": 2,
        "d2d_bytes": 0 if candidate else 1024,
        "d2d_copy_count": 0 if candidate else 2,
        "launch_count": 4,
        "workspace_bytes": 4096,
        "fallback_count": 0,
        "fallback_reasons": [],
        "direct_alias_safe": True,
        "direct_size_safe": True,
        "direct_admission_safe": True,
        "failed_invocation_count": 0,
        "failure_reasons": [],
    }
    return seal({
        "schema_version": ASSEMBLER.RUNTIME_SCHEMA,
        "status": "complete",
        "record_sha256": None,
        "side": side,
        "binding_kind": binding_kind,
        "binding_id": binding_id,
        **COMMON,
        "diagnostic_gate": True,
        "direct_sequence_output_enabled": candidate,
        "evidence_lane": "instrumented_diagnostic",
        "measurement_eligible": False,
        "terminal_status": "completed",
        "counters": counters,
    }, "record_sha256")


def profiler(
    side: str,
    root: Path,
    *,
    binding_kind: str = "run",
    binding_id: str = "run-1",
) -> dict[str, object]:
    candidate = side == "candidate"
    executable = root / "rocprofv3-fixture"
    if not executable.exists():
        executable.write_text("#!/bin/sh\necho rocprofv3-fixture-1.0\n", encoding="ascii")
        executable.chmod(0o755)
    raw = {
        "schema_version": ASSEMBLER.PROFILER_EVIDENCE.RAW_SCHEMA,
        "status": "complete",
        "record_sha256": None,
        "side": side,
        "binding_kind": binding_kind,
        "binding_id": binding_id,
        **COMMON,
        "timing_lane": "profiler_off_measurement",
        "measurement_eligible": True,
        "command": [str(executable.resolve()), "--fixture-capture", binding_id, side],
        "exit_code": 0,
        "started_unix_ns": 100,
        "completed_unix_ns": 200,
        "samples": [
            {
                "component_ms": None if binding_kind == "pair" else (1.25 if candidate else 1.5),
                "full_model_ms": None if binding_kind == "pair" else (3.25 if candidate else 3.5),
                "peak_vram_bytes": 8192 if candidate else 9216,
                "fidelity_binding_sha256": "d" * 64,
            }
        ],
    }
    raw["record_sha256"] = ASSEMBLER.PROFILER_EVIDENCE.self_hash(raw, "record_sha256")
    raw_path = write_json(root / f"{side}-{binding_kind}-{binding_id}-raw.json", raw)
    value, _snapshots = ASSEMBLER.PROFILER_EVIDENCE.build(raw_path, executable)
    return value


def files(tmp_path: Path, *, binding_kind: str = "run", binding_id: str = "run-1") -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    values = {
        "baseline_runtime": runtime("baseline", binding_kind=binding_kind, binding_id=binding_id),
        "baseline_profiler": profiler("baseline", tmp_path, binding_kind=binding_kind, binding_id=binding_id),
        "candidate_runtime": runtime("candidate", binding_kind=binding_kind, binding_id=binding_id),
        "candidate_profiler": profiler("candidate", tmp_path, binding_kind=binding_kind, binding_id=binding_id),
    }
    return {key: write_json(tmp_path / f"{key}.json", value) for key, value in values.items()}


def assemble(tmp_path: Path, *, binding_kind: str = "run", binding_id: str = "run-1") -> tuple[dict[str, object], Path]:
    paths = files(tmp_path, binding_kind=binding_kind, binding_id=binding_id)
    output = tmp_path / "trace.json"
    value = ASSEMBLER.assemble(paths, output, binding_kind, binding_id)
    return value, output


def test_assembles_run_trace_from_runtime_and_profiler_records(tmp_path: Path) -> None:
    value, output = assemble(tmp_path)
    assert value["implementation_id"] == COMMON["implementation_id"]
    assert value["source_id"] == COMMON["source_id"]
    assert value["request_id"] == COMMON["request_id"]
    assert value["trace_sha256"] == ASSEMBLER.self_hash(value, "trace_sha256")
    parsed = PRODUCER.parse_direct_sequence_output_trace(
        PRODUCER.capture(output.resolve(), "direct trace"),
        case_id=COMMON["case_id"], case_sha256=COMMON["case_sha256"],
        identity_sha256=COMMON["identity_sha256"], candidate_id=COMMON["candidate_id"],
        binding_kind="run", binding_id="run-1",
        implementation_id=COMMON["implementation_id"], source_id=COMMON["source_id"],
        source_sha256=COMMON["source_sha256"], request_id=COMMON["request_id"],
    )
    assert parsed["baseline"]["d2d_bytes"] == 1024
    assert parsed["candidate"]["d2d_copy_count"] == 0
    profiler_metrics = {
        "component_ms", "full_model_ms", "peak_vram_bytes",
        "fidelity_binding_sha256",
    }
    assert all(
        event["measurement_eligible"] is (event["metric"] in profiler_metrics)
        for event in value["events"]
    )
    assert all(
        event["evidence_lane"]
        == (
            "profiler_off_measurement"
            if event["metric"] in profiler_metrics
            else "instrumented_diagnostic"
        )
        for event in value["events"]
    )
    assert output.exists()


def test_rust_serialized_runtime_observation_matches_assembler_schema(
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    environment["CARGO_BUILD_JOBS"] = "1"
    completed = subprocess.run(
        [
            "cargo", "run", "--quiet", "-p", "ullm-engine", "--example",
            "aq4_p3_direct_observation_fixture",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
    )
    path = tmp_path / "rust-runtime.json"
    path.write_bytes(completed.stdout)
    snapshot = ASSEMBLER.capture(path.resolve(), "Rust runtime observation")
    common, counters = ASSEMBLER.validate_runtime(
        snapshot,
        side="candidate",
        binding_kind="run",
        binding_id="run-1",
        common=None,
    )
    assert common == {key: COMMON[key] for key in common}
    assert counters["invocation_count"] == 1
    assert counters["d2d_copy_count"] == 0


def test_assembles_pair_without_synthesizing_latency(tmp_path: Path) -> None:
    value, _ = assemble(tmp_path, binding_kind="pair", binding_id="pair-1")
    assert len(value["events"]) == 2 * len(ASSEMBLER.PAIR_METRICS)
    assert all(event["metric"] not in {"component_ms", "full_model_ms"} for event in value["events"])


def test_rejects_runtime_tamper_unknown_and_profiler_nonfinite(tmp_path: Path) -> None:
    paths = files(tmp_path)
    tampered = json.loads(paths["candidate_runtime"].read_text(encoding="utf-8"))
    tampered["counters"]["d2d_bytes"] = 7
    paths["candidate_runtime"].write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ASSEMBLER.AssemblerError, match="self-hash"):
        ASSEMBLER.assemble(paths, tmp_path / "tampered.json", "run", "run-1")

    paths = files(tmp_path / "unknown")
    value = json.loads(paths["baseline_runtime"].read_text(encoding="utf-8"))
    value["unknown"] = 1
    write_json(paths["baseline_runtime"], value)
    with pytest.raises(ASSEMBLER.AssemblerError, match="self-hash|fields differ"):
        ASSEMBLER.assemble(paths, tmp_path / "unknown-trace.json", "run", "run-1")

    paths = files(tmp_path / "nonfinite")
    value = json.loads(paths["candidate_profiler"].read_text(encoding="utf-8"))
    value["component_ms"] = float("nan")
    paths["candidate_profiler"].write_text(json.dumps(value, allow_nan=True), encoding="utf-8")
    with pytest.raises(ASSEMBLER.AssemblerError, match="non-finite"):
        ASSEMBLER.assemble(paths, tmp_path / "nonfinite-trace.json", "run", "run-1")


def test_rejects_diagnostic_gate_off_and_fidelity_mismatch(tmp_path: Path) -> None:
    paths = files(tmp_path)
    value = json.loads(paths["candidate_runtime"].read_text(encoding="utf-8"))
    value["diagnostic_gate"] = False
    seal(value, "record_sha256")
    write_json(paths["candidate_runtime"], value)
    with pytest.raises(ASSEMBLER.AssemblerError, match="diagnostic gate"):
        ASSEMBLER.assemble(paths, tmp_path / "gate.json", "run", "run-1")

    paths = files(tmp_path / "fidelity")
    value = json.loads(paths["candidate_profiler"].read_text(encoding="utf-8"))
    value["fidelity_binding_sha256"] = "e" * 64
    seal(value, "record_sha256")
    write_json(paths["candidate_profiler"], value)
    with pytest.raises(ASSEMBLER.AssemblerError, match="fidelity"):
        ASSEMBLER.assemble(paths, tmp_path / "fidelity-trace.json", "run", "run-1")


def test_profiler_requires_raw_provenance_and_rejects_raw_tamper(
    tmp_path: Path,
) -> None:
    paths = files(tmp_path)
    value = json.loads(paths["candidate_profiler"].read_text(encoding="utf-8"))
    del value["raw_capture"]
    seal(value, "record_sha256")
    write_json(paths["candidate_profiler"], value)
    with pytest.raises(ASSEMBLER.AssemblerError, match="fields differ"):
        ASSEMBLER.assemble(paths, tmp_path / "missing-raw.json", "run", "run-1")

    paths = files(tmp_path / "tamper")
    value = json.loads(paths["candidate_profiler"].read_text(encoding="utf-8"))
    raw_path = Path(value["raw_capture"]["path"])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["samples"][0]["peak_vram_bytes"] += 1
    raw["record_sha256"] = ASSEMBLER.PROFILER_EVIDENCE.self_hash(
        raw, "record_sha256"
    )
    write_json(raw_path, raw)
    with pytest.raises(ASSEMBLER.AssemblerError, match="profiler evidence"):
        ASSEMBLER.assemble(paths, tmp_path / "raw-tamper.json", "run", "run-1")


def test_profiler_producer_rechecks_inode_sha_and_bounds(tmp_path: Path) -> None:
    value = profiler("candidate", tmp_path)
    raw_path = Path(value["raw_capture"]["path"])
    executable = Path(value["profiler"]["path"])
    _result, snapshots = ASSEMBLER.PROFILER_EVIDENCE.build(raw_path, executable)
    original = raw_path.read_bytes()
    replacement = raw_path.with_name("replacement.json")
    replacement.write_bytes(original)
    os.replace(replacement, raw_path)
    with pytest.raises(
        ASSEMBLER.PROFILER_EVIDENCE.ProfilerEvidenceError,
        match="identity or SHA-256 changed",
    ):
        snapshots[0].verify("raw profiler capture")

    oversized = tmp_path / "oversized.raw"
    with oversized.open("wb") as handle:
        handle.truncate(ASSEMBLER.PROFILER_EVIDENCE.MAX_INPUT_BYTES + 1)
    with pytest.raises(
        ASSEMBLER.PROFILER_EVIDENCE.ProfilerEvidenceError,
        match="file identity is invalid",
    ):
        ASSEMBLER.PROFILER_EVIDENCE.capture(oversized, "oversized raw")


def test_profiler_raw_hardlink_and_secret_command_are_rejected(
    tmp_path: Path,
) -> None:
    value = profiler("candidate", tmp_path)
    raw_path = Path(value["raw_capture"]["path"])
    hardlink = tmp_path / "raw-hardlink.json"
    hardlink.hardlink_to(raw_path)
    with pytest.raises(
        ASSEMBLER.PROFILER_EVIDENCE.ProfilerEvidenceError,
        match="file identity is invalid",
    ):
        ASSEMBLER.PROFILER_EVIDENCE.capture(raw_path, "linked raw")
    hardlink.unlink()

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["command"].append("--prompt=secret-token")
    raw["record_sha256"] = ASSEMBLER.PROFILER_EVIDENCE.self_hash(
        raw, "record_sha256"
    )
    with pytest.raises(
        ASSEMBLER.PROFILER_EVIDENCE.ProfilerEvidenceError,
        match="token or prompt",
    ):
        ASSEMBLER.PROFILER_EVIDENCE.derive_raw(raw)


def test_instrumented_profiler_timing_cannot_be_laundered(
    tmp_path: Path,
) -> None:
    paths = files(tmp_path)
    value = json.loads(paths["candidate_profiler"].read_text(encoding="utf-8"))
    value["timing_lane"] = "instrumented_diagnostic"
    value["measurement_eligible"] = False
    seal(value, "record_sha256")
    write_json(paths["candidate_profiler"], value)
    with pytest.raises(ASSEMBLER.AssemblerError, match="profiler evidence|timing lane"):
        ASSEMBLER.assemble(paths, tmp_path / "laundered.json", "run", "run-1")


def test_failed_runtime_terminal_cannot_enter_complete_trace(tmp_path: Path) -> None:
    paths = files(tmp_path)
    value = json.loads(paths["candidate_runtime"].read_text(encoding="utf-8"))
    value["status"] = "failed"
    value["terminal_status"] = "error"
    value["counters"]["failed_invocation_count"] = 1
    value["counters"]["failure_reasons"] = ["prefill_dispatch_failed"]
    seal(value, "record_sha256")
    write_json(paths["candidate_runtime"], value)
    with pytest.raises(ASSEMBLER.AssemblerError, match="status|terminal"):
        ASSEMBLER.assemble(paths, tmp_path / "failed-runtime.json", "run", "run-1")


def test_profiler_resealed_version_and_probe_tamper_are_rejected(tmp_path: Path) -> None:
    for field in ("version", "probe"):
        paths = files(tmp_path / field)
        value = json.loads(paths["candidate_profiler"].read_text(encoding="utf-8"))
        if field == "version":
            value["profiler_version"] = "rocprofv3-fixture-9.9"
        else:
            value["profiler_version_probe"]["stdout"]["sha256"] = "0" * 64
        seal(value, "record_sha256")
        write_json(paths["candidate_profiler"], value)
        with pytest.raises(ASSEMBLER.AssemblerError, match="profiler evidence"):
            ASSEMBLER.assemble(
                paths, tmp_path / f"{field}-tamper.json", "run", "run-1"
            )


def test_profiler_replaced_executable_is_rejected_during_assembly(tmp_path: Path) -> None:
    paths = files(tmp_path)
    value = json.loads(paths["candidate_profiler"].read_text(encoding="utf-8"))
    executable = Path(value["profiler"]["path"])
    replacement = executable.with_name("rocprofv3-replacement")
    replacement.write_text("#!/bin/sh\necho rocprofv3-fixture-2.0\n", encoding="ascii")
    replacement.chmod(0o755)
    os.replace(replacement, executable)
    with pytest.raises(ASSEMBLER.AssemblerError, match="profiler evidence"):
        ASSEMBLER.assemble(paths, tmp_path / "replaced.json", "run", "run-1")


def test_profiler_version_probe_has_bounded_timeout_and_empty_stderr_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "slow-profiler"
    executable.write_text("#!/bin/sh\nsleep 2\necho too-late\n", encoding="ascii")
    executable.chmod(0o755)
    snapshot = ASSEMBLER.PROFILER_EVIDENCE.capture(
        executable, "slow profiler", executable=True
    )
    monkeypatch.setattr(
        ASSEMBLER.PROFILER_EVIDENCE, "VERSION_PROBE_TIMEOUT_SECONDS", 0.05
    )
    monkeypatch.setattr(
        ASSEMBLER.PROFILER_EVIDENCE, "VERSION_PROBE_REAP_TIMEOUT_SECONDS", 0.1
    )
    with pytest.raises(
        ASSEMBLER.PROFILER_EVIDENCE.ProfilerEvidenceError, match="timed out"
    ):
        ASSEMBLER.PROFILER_EVIDENCE.probe_profiler_version(snapshot)

    noisy = tmp_path / "noisy-profiler"
    noisy.write_text(
        "#!/bin/sh\necho rocprofv3-fixture-1.0\necho diagnostic >&2\n",
        encoding="ascii",
    )
    noisy.chmod(0o755)
    noisy_snapshot = ASSEMBLER.PROFILER_EVIDENCE.capture(
        noisy, "noisy profiler", executable=True
    )
    monkeypatch.setattr(
        ASSEMBLER.PROFILER_EVIDENCE, "VERSION_PROBE_TIMEOUT_SECONDS", 30
    )
    with pytest.raises(
        ASSEMBLER.PROFILER_EVIDENCE.ProfilerEvidenceError, match="stderr must be empty"
    ):
        ASSEMBLER.PROFILER_EVIDENCE.probe_profiler_version(noisy_snapshot)
