from __future__ import annotations

import importlib.util
import json
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
        "measurement_eligible": False,
        "counters": counters,
    }, "record_sha256")


def profiler(side: str, *, binding_kind: str = "run", binding_id: str = "run-1", lane: str = "profiler_off") -> dict[str, object]:
    candidate = side == "candidate"
    return seal({
        "schema_version": ASSEMBLER.PROFILER_SCHEMA,
        "status": "complete",
        "record_sha256": None,
        "side": side,
        "binding_kind": binding_kind,
        "binding_id": binding_id,
        **COMMON,
        "timing_lane": lane,
        "measurement_eligible": lane == "profiler_off",
        "component_ms": None if binding_kind == "pair" else (1.25 if candidate else 1.5),
        "full_model_ms": None if binding_kind == "pair" else (3.25 if candidate else 3.5),
        "peak_vram_bytes": 8192 if candidate else 9216,
        "fidelity_binding_sha256": "d" * 64,
    }, "record_sha256")


def files(tmp_path: Path, *, binding_kind: str = "run", binding_id: str = "run-1") -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    values = {
        "baseline_runtime": runtime("baseline", binding_kind=binding_kind, binding_id=binding_id),
        "baseline_profiler": profiler("baseline", binding_kind=binding_kind, binding_id=binding_id),
        "candidate_runtime": runtime("candidate", binding_kind=binding_kind, binding_id=binding_id),
        "candidate_profiler": profiler("candidate", binding_kind=binding_kind, binding_id=binding_id),
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
    assert output.exists()


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
