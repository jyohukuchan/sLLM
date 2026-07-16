from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COMPARE = load_tool(
    "compare_aq4_layer0_cpu_gpu_stage_stream",
    ROOT / "tools/compare-aq4-layer0-cpu-gpu-stage-stream.py",
)
VERIFY = load_tool(
    "verify_aq4_layer0_package_embedding_fixture",
    ROOT / "tools/verify-aq4-layer0-package-embedding-fixture.py",
)


def write_frame_stream(path: Path, values_by_key: dict[tuple[str, int, str, int, int, str], list[float]]) -> None:
    with path.open("wb") as output:
        for case_id, step, context_hash, context_length, timestep, stage in sorted(values_by_key):
            values = values_by_key[(case_id, step, context_hash, context_length, timestep, stage)]
            payload = struct.pack(f"<{len(values)}f", *values)
            header = {
                "kind": "stage",
                "case_id": case_id,
                "step": step,
                "context_token_ids_sha256": context_hash,
                "context_length": context_length,
                "timestep": timestep,
                "stage": stage,
                "dtype": "f32le",
                "shape": [len(values)],
                "bytes": len(payload),
            }
            output.write(json.dumps(header, separators=(",", ":")).encode("utf-8") + b"\n")
            output.write(payload)
        output.write(b'{"kind":"end"}\n')


def test_cpu_gpu_stage_comparator_consumes_hash_bound_final_frames(tmp_path: Path) -> None:
    keys = COMPARE.expected_keys()
    cpu_values = {key: [1.0, 2.0, 3.0, 4.0] for key in keys}
    gpu_values = {key: list(values) for key, values in cpu_values.items()}
    changed = next(key for key in keys if key[-1] == "post_norm")
    gpu_values[changed][1] = 2.5
    cpu_stream = tmp_path / "cpu.framed"
    gpu_stream = tmp_path / "gpu.framed"
    output = tmp_path / "comparison.json"
    write_frame_stream(cpu_stream, cpu_values)
    write_frame_stream(gpu_stream, gpu_values)

    result = COMPARE.compare(cpu_stream, gpu_stream, output)

    assert result["status"] == "valid"
    assert set(result["stages"]) == set(COMPARE.STAGES)
    assert result["stages"]["post_norm"]["max_abs"] == 0.5
    assert result["stages"]["layer_output"]["max_abs"] == 0.0
    assert (tmp_path / "SHA256SUMS").is_file()


def test_package_embedding_fixture_verifier_requires_bit_exact_f32_rows(tmp_path: Path) -> None:
    package = tmp_path / "package"
    payload_path = package / "passthrough" / "embedding.raw"
    payload_path.parent.mkdir(parents=True)
    package_row = struct.pack("<H", 0x3F80) * VERIFY.HIDDEN
    payload = package_row * 2
    payload_path.write_bytes(payload)
    manifest = {
        "passthrough_tensors": [
            {
                "name": VERIFY.EMBEDDING_TENSOR,
                "dtype": "BF16",
                "shape": [2, VERIFY.HIDDEN],
                "elements": 2 * VERIFY.HIDDEN,
                "payload_file": "passthrough/embedding.raw",
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ]
    }
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    fixture_root = tmp_path / "fixture"
    sidecar = fixture_root / "input" / "row.f32le"
    sidecar.parent.mkdir(parents=True)
    f32_row = struct.pack("<f", 1.0) * VERIFY.HIDDEN
    sidecar.write_bytes(f32_row)
    token_ids = [1]
    case = {
        "kind": "case",
        "case_id": "fixture",
        "step": 0,
        "context_token_ids": token_ids,
        "context_token_ids_sha256": VERIFY.canonical_token_ids_hash(token_ids),
        "context_length": 1,
        "residual_path": "input/row.f32le",
        "residual_sha256": hashlib.sha256(f32_row).hexdigest(),
        "residual_shape": [1, VERIFY.HIDDEN],
        "residual_dtype": "f32le",
    }
    header = {
        "kind": "header",
        "schema_version": VERIFY.HYBRID_SCHEMA,
        "tensor_name": VERIFY.EMBEDDING_TENSOR,
        "dtype": "f32",
        "shape": [VERIFY.HIDDEN],
        "residual_encoding": "f32le_row_major",
        "source_model_index_sha256": "a" * 64,
    }
    hybrid_input = fixture_root / "hybrid-input.jsonl"
    hybrid_input.write_text(
        json.dumps(header, sort_keys=True) + "\n" + json.dumps(case, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = VERIFY.verify(package, hybrid_input, tmp_path / "receipt.json")

    assert result["status"] == "valid"
    assert result["cases"][0]["case_id"] == "fixture"
