#!/usr/bin/env python3
"""Compare layer-0 QKV/Z BF16, SQ8 overlay, and production AQ4 matvecs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


SCHEMA = "ullm.qwen35_aq4_sq8_overlay_cpu_oracle.v1"
LAYER_INDEX = 0
ROWS = 3
HIDDEN = 4096
TENSORS = {
    "qkv": ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", 8192),
    "z": ("model.language_model.layers.0.linear_attn.in_proj_z.weight", 4096),
}


class OracleError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--source-model-dir", required=True, type=Path)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--depth-artifact-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--row-chunk", type=int, default=128)
    return parser.parse_args()


def _load_builder() -> ModuleType:
    path = Path(__file__).resolve().with_name("build-qwen35-aq4-sq8-overlay.py")
    spec = importlib.util.spec_from_file_location("qwen35_aq4_sq8_overlay_builder", path)
    if spec is None or spec.loader is None:
        raise OracleError("overlay builder helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise OracleError(f"expected regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_f32(path: Path, elements: int) -> torch.Tensor:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != elements * 4:
        raise OracleError(f"f32 payload geometry differs: {path}")
    values = np.fromfile(path, dtype="<f4", count=elements)
    if values.size != elements:
        raise OracleError(f"f32 payload is truncated: {path}")
    return torch.from_numpy(values.copy()).reshape(-1)


def tensor_sha(values: torch.Tensor) -> str:
    array = values.detach().to(torch.float32).contiguous().cpu().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    candidate = candidate.to(torch.float64).reshape(-1)
    reference = reference.to(torch.float64).reshape(-1)
    if candidate.shape != reference.shape:
        raise OracleError("comparison shapes differ")
    finite = bool(torch.isfinite(candidate).all() and torch.isfinite(reference).all())
    if not finite:
        return {"finite": False}
    delta = candidate - reference
    absolute = delta.abs()
    reference_norm = float(torch.linalg.vector_norm(reference))
    candidate_norm = float(torch.linalg.vector_norm(candidate))
    denominator = candidate_norm * reference_norm
    return {
        "finite": True,
        "elements": int(candidate.numel()),
        "max_abs": float(absolute.max()) if candidate.numel() else 0.0,
        "mean_abs": float(absolute.mean()) if candidate.numel() else 0.0,
        "rmse": float(torch.sqrt(torch.mean(delta * delta))) if candidate.numel() else 0.0,
        "relative_l2": float(torch.linalg.vector_norm(delta)) / max(reference_norm, 1.0e-30),
        "cosine": float(torch.dot(candidate, reference) / denominator) if denominator else 1.0,
    }


def metrics_with_rows(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise OracleError("row comparison shapes differ")
    return {
        "aggregate": metrics(candidate, reference),
        "rows": [metrics(candidate[row], reference[row]) for row in range(candidate.shape[0])],
    }


def _source_file(source_model_dir: Path, tensor_name: str) -> Path:
    index = json.loads((source_model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not isinstance(weight_map.get(tensor_name), str):
        raise OracleError(f"source tensor is absent from index: {tensor_name}")
    path = source_model_dir / weight_map[tensor_name]
    if path.is_symlink() or not path.is_file():
        raise OracleError(f"source safetensor differs: {path}")
    return path


def bf16_matvec(
    source_model_dir: Path,
    tensor_name: str,
    inputs: torch.Tensor,
    output_rows: int,
    row_chunk: int,
) -> torch.Tensor:
    source_file = _source_file(source_model_dir, tensor_name)
    outputs = torch.empty((inputs.shape[0], output_rows), dtype=torch.float32)
    with safe_open(source_file, framework="pt", device="cpu") as handle:
        view = handle.get_slice(tensor_name)
        if str(view.get_dtype()) != "BF16" or list(view.get_shape()) != [output_rows, HIDDEN]:
            raise OracleError(f"source tensor dtype/shape differs: {tensor_name}")
        for start in range(0, output_rows, row_chunk):
            end = min(start + row_chunk, output_rows)
            weight = view[start:end].to(torch.float32)
            outputs[:, start:end] = torch.matmul(inputs, weight.transpose(0, 1))
            del weight
    return outputs


def sq8_matvec(
    artifact_dir: Path,
    entry: dict[str, Any],
    inputs: torch.Tensor,
    output_rows: int,
    row_chunk: int,
) -> torch.Tensor:
    if entry.get("shape") != [output_rows, HIDDEN] or entry.get("scale_block_cols") != 256:
        raise OracleError("SQ8 tensor geometry differs")
    payload_path = artifact_dir / str(entry["payload_file"])
    scale_path = artifact_dir / str(entry["scale_file"])
    blocks = HIDDEN // 256
    if payload_path.stat().st_size != output_rows * HIDDEN:
        raise OracleError("SQ8 payload size differs")
    if scale_path.stat().st_size != output_rows * blocks * 4:
        raise OracleError("SQ8 scale size differs")
    outputs = torch.empty((inputs.shape[0], output_rows), dtype=torch.float32)
    with payload_path.open("rb") as payload, scale_path.open("rb") as scales:
        for start in range(0, output_rows, row_chunk):
            count = min(row_chunk, output_rows - start)
            encoded_raw = payload.read(count * HIDDEN)
            scale_raw = scales.read(count * blocks * 4)
            if len(encoded_raw) != count * HIDDEN or len(scale_raw) != count * blocks * 4:
                raise OracleError("SQ8 payload is truncated")
            encoded_u8 = torch.from_numpy(np.frombuffer(encoded_raw, dtype=np.uint8).copy())
            encoded = encoded_u8.view(torch.float8_e4m3fn).to(torch.float32).reshape(count, HIDDEN)
            scale = torch.from_numpy(np.frombuffer(scale_raw, dtype="<f4").copy()).reshape(count, blocks)
            weight = encoded.reshape(count, blocks, 256) * scale[:, :, None]
            outputs[:, start : start + count] = torch.matmul(inputs, weight.reshape(count, HIDDEN).transpose(0, 1))
            del encoded_u8, encoded, scale, weight
    return outputs


def validate_binding(
    builder: ModuleType,
    artifact_dir: Path,
    source_model_dir: Path,
    package_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    binding_path = artifact_dir / "binding.json"
    binding = builder.read_json(binding_path)
    names = builder.exact_tensor_names(builder.read_json(source_model_dir / "config.json"))
    manifest, content_sha = builder.validate_sq_manifest(artifact_dir, names)
    builder.validate_bound_source_provenance(
        binding, source_model_dir, artifact_dir, manifest, names
    )
    inventory = builder.artifact_inventory(artifact_dir, binding)
    policy = binding.get("artifact_policy")
    if not isinstance(policy, dict) or any(
        policy.get(name) != inventory.get(name)
        for name in ("uid", "gid", "directory_mode", "file_mode", "regular_file_nlink")
    ):
        raise OracleError("overlay artifact immutability policy differs")
    if (
        binding.get("schema_version") != builder.BINDING_SCHEMA
        or binding.get("tensor_names") != sorted(names)
        or binding.get("tensor_set_sha256") != builder.tensor_set_sha256(names)
        or binding.get("content_sha256") != content_sha
        or binding.get("source", {}).get("model_dir") != str(source_model_dir.resolve())
        or binding.get("package", {}).get("root") != str(package_dir.resolve())
        or binding.get("package", {}).get("manifest_sha256") != sha256_file(package_dir / "manifest.json")
    ):
        raise OracleError("overlay binding identity differs")
    sq_bound = binding.get("sq_manifest")
    if not isinstance(sq_bound, dict) or sq_bound != {
        "path": "sq_manifest.json",
        "sha256": sha256_file(artifact_dir / "sq_manifest.json"),
    }:
        raise OracleError("overlay SQ manifest binding differs")
    return binding, manifest, names


def main() -> int:
    args = parse_args()
    if args.row_chunk <= 0:
        raise OracleError("row-chunk must be positive")
    if args.output_json.exists():
        raise OracleError(f"refusing to overwrite report: {args.output_json}")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    builder = _load_builder()
    binding, manifest, _ = validate_binding(
        builder, args.artifact_dir, args.source_model_dir, args.package_dir
    )
    entries = {item["name"]: item for item in manifest["fp8_tensors"]}
    step_root = args.depth_artifact_root / "steps/layer-0/baseline"
    source_root = args.depth_artifact_root / "source/layer-0/shared"
    input_path = step_root / "input-normed.f32le"
    input_values = read_f32(input_path, ROWS * HIDDEN).reshape(ROWS, HIDDEN)
    results: dict[str, Any] = {}
    for short_name, (tensor_name, output_rows) in TENSORS.items():
        bf16 = bf16_matvec(
            args.source_model_dir, tensor_name, input_values, output_rows, args.row_chunk
        )
        sq8 = sq8_matvec(
            args.artifact_dir, entries[tensor_name], input_values, output_rows, args.row_chunk
        )
        aq4_path = step_root / f"{short_name}.f32le"
        source_capture_path = source_root / f"source-{short_name}.f32le"
        aq4 = read_f32(aq4_path, ROWS * output_rows).reshape(ROWS, output_rows)
        source_capture = read_f32(source_capture_path, ROWS * output_rows).reshape(ROWS, output_rows)
        results[short_name] = {
            "tensor_name": tensor_name,
            "shape": [output_rows, HIDDEN],
            "outputs": {
                "bf16_direct_sha256": tensor_sha(bf16),
                "sq8_sha256": tensor_sha(sq8),
                "aq4_capture": {"path": str(aq4_path), "sha256": sha256_file(aq4_path)},
                "bf16_existing_capture": {
                    "path": str(source_capture_path),
                    "sha256": sha256_file(source_capture_path),
                },
            },
            "sq8_vs_bf16": metrics_with_rows(sq8, bf16),
            "aq4_vs_bf16": metrics_with_rows(aq4, bf16),
            "existing_bf16_capture_vs_direct": metrics_with_rows(source_capture, bf16),
        }
        del bf16, sq8, aq4, source_capture
    report = {
        "schema_version": SCHEMA,
        "status": "valid",
        "scope": "layer0_linear_attention_qkv_z_cpu_matvec",
        "promotion": False,
        "identity": {
            "tool_sha256": sha256_file(Path(__file__)),
            "binding_manifest_sha256": sha256_file(args.artifact_dir / "binding.json"),
            "content_sha256": binding["content_sha256"],
            "tensor_set_sha256": binding["tensor_set_sha256"],
            "sq_manifest_sha256": sha256_file(args.artifact_dir / "sq_manifest.json"),
            "package_manifest_sha256": sha256_file(args.package_dir / "manifest.json"),
            "source_config_sha256": sha256_file(args.source_model_dir / "config.json"),
            "source_index_sha256": sha256_file(args.source_model_dir / "model.safetensors.index.json"),
            "input_normed": {"path": str(input_path), "sha256": sha256_file(input_path)},
        },
        "execution": {
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "row_chunk": args.row_chunk,
            "rows": ROWS,
            "hidden": HIDDEN,
            "ru_maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "results": results,
        "interpretation": "implementation diagnostic only; no promotion threshold is applied",
    }
    if not all(
        result[comparison]["aggregate"].get("finite") is True
        for result in results.values()
        for comparison in ("sq8_vs_bf16", "aq4_vs_bf16", "existing_bf16_capture_vs_direct")
    ):
        raise OracleError("oracle comparison contains non-finite values")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output_json.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise OracleError(f"refusing to overwrite report: {args.output_json}") from error
    print(json.dumps({"schema_version": SCHEMA, "status": "valid", "output": str(args.output_json)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        raise SystemExit(str(error)) from error
