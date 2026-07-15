#!/usr/bin/env python3
"""Run a bounded CPU-only QKV/Z hybrid accumulation diagnostic through linear layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import struct
import subprocess
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


SCHEMA = "ullm.aq4_layer_depth_accumulation.cpu.v1"
HIDDEN = 4096
QKV_ROWS = 8192
ROWS = 3
ROW_CHUNK = 256
TRACKS = ("baseline", "qkv_only", "z_only", "combined")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_f32(path: Path, expected: int | None = None) -> list[float]:
    raw = path.read_bytes()
    if len(raw) % 4:
        raise ValueError(f"unaligned f32 sidecar: {path}")
    values = list(struct.unpack(f"<{len(raw) // 4}f", raw))
    if expected is not None and len(values) != expected:
        raise ValueError(f"f32 shape differs for {path}: got {len(values)} expected {expected}")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"non-finite f32 sidecar: {path}")
    return values


def write_f32(path: Path, values: list[float]) -> str:
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"refusing non-finite f32 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = struct.pack(f"<{len(values)}f", *values)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def metric(actual: list[float], reference: list[float]) -> dict[str, Any]:
    if len(actual) != len(reference) or len(actual) % ROWS:
        raise ValueError("metric geometry differs")
    width = len(actual) // ROWS

    def one(lhs: list[float], rhs: list[float], rows: int) -> dict[str, Any]:
        diff_sq = sum((a - b) ** 2 for a, b in zip(lhs, rhs))
        ref_sq = sum(b * b for b in rhs)
        return {
            "rows": rows,
            "elements_per_row": len(lhs) // rows,
            "max_abs": max((abs(a - b) for a, b in zip(lhs, rhs)), default=0.0),
            "relative_l2": math.sqrt(diff_sq / ref_sq) if ref_sq else (0.0 if not diff_sq else math.inf),
            "nonfinite": False,
        }

    return {
        "aggregate": one(actual, reference, ROWS),
        "per_step": [one(actual[row * width : (row + 1) * width], reference[row * width : (row + 1) * width], 1) for row in range(ROWS)],
    }


def source_tensor_path(source_model: Path, tensor_name: str) -> tuple[Path, Path]:
    index_path = source_model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return index_path, source_model / index["weight_map"][tensor_name]


def build_source_projection(
    source_model: Path,
    input_normed: Path,
    layer_index: int,
    family: str,
    output: Path,
) -> dict[str, Any]:
    rows = QKV_ROWS if family == "qkv" else HIDDEN
    tensor_name = f"model.language_model.layers.{layer_index}.linear_attn.in_proj_{family}.weight"
    index_path, shard_path = source_tensor_path(source_model, tensor_name)
    inputs = read_f32(input_normed, ROWS * HIDDEN)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_digest = hashlib.sha256()
    payload_digest = hashlib.sha256()
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(tensor_name)
        if list(weight.shape) != [rows, HIDDEN] or weight.dtype != torch.bfloat16:
            raise ValueError(f"source {family} tensor contract differs: {tensor_name}")
        for start in range(0, rows, ROW_CHUNK):
            chunk = weight[start : min(start + ROW_CHUNK, rows)].detach().contiguous()
            payload_digest.update(chunk.view(torch.uint16).numpy().tobytes())
        with output.open("xb") as target:
            for row in range(ROWS):
                vector = torch.tensor(inputs[row * HIDDEN : (row + 1) * HIDDEN], dtype=torch.float32)
                for start in range(0, rows, ROW_CHUNK):
                    chunk = weight[start : min(start + ROW_CHUNK, rows)].to(torch.float32)
                    result = torch.matmul(vector, chunk.transpose(0, 1)).contiguous()
                    if not bool(torch.isfinite(result).all()):
                        raise ValueError(f"non-finite source {family} projection")
                    raw = result.numpy().tobytes()
                    target.write(raw)
                    output_digest.update(raw)
    report = {
        "tensor_name": tensor_name,
        "shape": [ROWS, rows],
        "dtype": "f32",
        "operation": "live_input_f32_matmul_source_bf16_weight_cast_f32_accumulate_f32",
        "input_normed_path": str(input_normed),
        "input_normed_sha256": sha256_file(input_normed),
        "output_path": str(output),
        "output_sha256": output_digest.hexdigest(),
        "source": {
            "index_path": str(index_path),
            "index_sha256": sha256_file(index_path),
            "shard_path": str(shard_path),
            "tensor_payload_sha256": payload_digest.hexdigest(),
            "tensor_dtype": "BF16",
            "tensor_shape": [rows, HIDDEN],
        },
        "memory": {"row_chunk": ROW_CHUNK, "full_f32_weight_materialized": False},
        "promotion": False,
        "holdout": "not_run",
        "policy_evaluation": "policy_not_evaluated",
        "thresholds": None,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_step(
    engine: Path,
    package: Path,
    residual: str,
    source_qkv: str,
    source_z: str,
    output: Path,
    layer_index: int,
    chunk_bytes: int,
) -> dict[str, Any]:
    command = [
        str(engine), "package-linear-attn-depth-step-diagnostic", str(package), residual,
        source_qkv, source_z, str(output), "0", str(chunk_bytes), str(layer_index), str(ROWS),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"depth step failed: {' '.join(command)}\n{completed.stdout}\n{completed.stderr}")
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    report["stdout"] = completed.stdout.strip()
    return report


def slice_golden_input(golden: Path, output: Path) -> dict[str, Any]:
    metadata_path = golden / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("hidden_size") != HIDDEN or metadata.get("layers", [{}])[0].get("before_shape") != [1, 16, HIDDEN]:
        raise ValueError("golden layer-0 input contract differs")
    source = golden / metadata["layers"][0]["before_file"]
    values = read_f32(source, 16 * HIDDEN)[: ROWS * HIDDEN]
    digest = write_f32(output, values)
    return {
        "fixture_root": str(golden), "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path), "source_path": str(source),
        "source_sha256": sha256_file(source), "slice": {"token_ids": [1, 2, 3], "rows": ROWS, "hidden": HIDDEN},
        "output_path": str(output), "output_sha256": digest,
    }


def interaction(output: Path, depth: int, values: dict[str, list[float]]) -> dict[str, Any]:
    baseline = values["baseline"]
    prediction = [(q - b) + (z - b) for b, q, z in zip(baseline, values["qkv_only"], values["z_only"])]
    observed = [c - b for b, c in zip(baseline, values["combined"])]
    residual = [actual - predicted for actual, predicted in zip(observed, prediction)]
    prediction_path = output / f"depth-{depth}-additive-prediction-delta.f32le"
    observed_path = output / f"depth-{depth}-combined-delta.f32le"
    residual_path = output / f"depth-{depth}-interaction-residual.f32le"
    prediction_norm = math.sqrt(sum(value * value for value in prediction))
    observed_norm = math.sqrt(sum(value * value for value in observed))
    return {
        "definition": "(combined-baseline)-((qkv_only-baseline)+(z_only-baseline))",
        "observation": "amplification" if observed_norm > prediction_norm else "cancellation" if observed_norm < prediction_norm else "additive",
        "combined_delta_vs_additive_prediction": metric(observed, prediction),
        "additive_prediction_delta_l2": prediction_norm,
        "combined_delta_l2": observed_norm,
        "interaction_residual_l2": math.sqrt(sum(value * value for value in residual)),
        "additive_prediction_delta_sha256": write_f32(prediction_path, prediction),
        "combined_delta_sha256": write_f32(observed_path, observed),
        "interaction_residual_sha256": write_f32(residual_path, residual),
        "finite": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError(f"refusing to overwrite output: {args.output}")
        args.output.mkdir(parents=True)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        input_binding = slice_golden_input(args.golden, args.output / "input" / "residual-depth0.f32le")
        residuals = {track: str(args.output / "input" / "residual-depth0.f32le") for track in TRACKS}
        checkpoints: dict[str, Any] = {}
        all_steps: dict[str, Any] = {}
        max_child_hwm = 0

        for layer in range(3):
            layer_steps: dict[str, Any] = {}
            shared_sources: dict[str, Path] = {}
            if layer == 0:
                baseline_dir = args.output / "steps" / f"layer-{layer}" / "baseline"
                layer_steps["baseline"] = run_step(args.engine, args.package, residuals["baseline"], "-", "-", baseline_dir, layer, args.chunk_bytes)
                live_input = baseline_dir / "input-normed.f32le"
                for family in ("qkv", "z"):
                    source_path = args.output / "source" / f"layer-{layer}" / "shared" / f"source-{family}.f32le"
                    build_source_projection(args.source_model, live_input, layer, family, source_path)
                    shared_sources[family] = source_path

            for track in TRACKS:
                if layer == 0 and track == "baseline":
                    report = layer_steps[track]
                elif track == "baseline":
                    step_dir = args.output / "steps" / f"layer-{layer}" / track
                    report = run_step(args.engine, args.package, residuals[track], "-", "-", step_dir, layer, args.chunk_bytes)
                else:
                    if layer == 0:
                        qkv_path = str(shared_sources["qkv"]) if track in ("qkv_only", "combined") else "-"
                        z_path = str(shared_sources["z"]) if track in ("z_only", "combined") else "-"
                    else:
                        prep_dir = args.output / "preflight" / f"layer-{layer}" / track
                        run_step(args.engine, args.package, residuals[track], "-", "-", prep_dir, layer, args.chunk_bytes)
                        qkv_path = "-"
                        z_path = "-"
                        if track in ("qkv_only", "combined"):
                            source = args.output / "source" / f"layer-{layer}" / track / "source-qkv.f32le"
                            build_source_projection(args.source_model, prep_dir / "input-normed.f32le", layer, "qkv", source)
                            qkv_path = str(source)
                        if track in ("z_only", "combined"):
                            source = args.output / "source" / f"layer-{layer}" / track / "source-z.f32le"
                            build_source_projection(args.source_model, prep_dir / "input-normed.f32le", layer, "z", source)
                            z_path = str(source)
                    step_dir = args.output / "steps" / f"layer-{layer}" / track
                    report = run_step(args.engine, args.package, residuals[track], qkv_path, z_path, step_dir, layer, args.chunk_bytes)
                layer_steps[track] = report
                residuals[track] = str(args.output / "steps" / f"layer-{layer}" / track / "layer-output.f32le")
                max_child_hwm = max(max_child_hwm, int(report.get("memory", {}).get("vm_hwm_kib") or 0))

            depth = layer + 1
            outputs = {track: read_f32(Path(residuals[track]), ROWS * HIDDEN) for track in TRACKS}
            checkpoint_metrics = {f"{track}_vs_baseline": metric(outputs[track], outputs["baseline"]) for track in TRACKS[1:]}
            checkpoints[str(depth)] = {
                "depth": depth, "last_layer_index": layer,
                "hidden": {track: {"path": residuals[track], "sha256": sha256_file(Path(residuals[track])), "shape": [ROWS, HIDDEN]} for track in TRACKS},
                "linear_recurrent_states": {track: {"path": str(args.output / "steps" / f"layer-{layer}" / track / "recurrent.f32le"), "sha256": layer_steps[track]["outputs"]["recurrent_sha256"], "state_digests": layer_steps[track]["recurrent_state_digests"]} for track in TRACKS},
                "metrics": checkpoint_metrics,
                "interaction": interaction(args.output / "interactions", depth, outputs),
            }
            all_steps[str(layer)] = layer_steps

        depth_one = checkpoints["1"]["metrics"]["combined_vs_baseline"]["aggregate"]["relative_l2"]
        for value in checkpoints.values():
            current = value["metrics"]["combined_vs_baseline"]["aggregate"]["relative_l2"]
            value["combined_amplification_vs_depth1"] = current / depth_one if depth_one else None
        config_path = args.source_model / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
        report = {
            "schema_version": SCHEMA, "status": "partial_valid_blocked_at_depth4",
            "classification": "unclassified", "promotion": False, "holdout": "not_run",
            "policy_evaluation": "policy_not_evaluated", "thresholds": None,
            "device": {"backend": "cpu", "requested_index": 0},
            "package": {"root": str(args.package), "manifest_sha256": sha256_file(args.package / "manifest.json")},
            "engine": {"path": str(args.engine), "sha256": sha256_file(args.engine)},
            "source_model": {"root": str(args.source_model), "config_sha256": sha256_file(config_path)},
            "input": input_binding,
            "layer_topology": {"layer_types_0_3": config["layer_types"][:4], "full_attention_interval": config["full_attention_interval"]},
            "tracks": list(TRACKS), "state_reset": "each_track_independent_zero_state_per_linear_layer_sequence",
            "checkpoints": checkpoints, "steps": all_steps,
            "maximum_faithful_depth": 3,
            "requested_depth4": {
                "status": "blocked_not_run", "blocking_layer_index": 3, "layer_type": "full_attention",
                "exact_blocker": "the mixed golden-prefix CPU loop is monolithic and does not expose a per-layer interface that accepts each track live external residual, preserves and returns track-local causal KV state, and composes with the existing linear-layer QKV/Z diagnostic hooks",
                "required_hook": "one mixed-decoder diagnostic step interface accepting external residual and track-local state, routing existing QKV/Z overrides on linear layers, and returning layer output plus updated linear or causal-KV state",
            },
            "memory": {"track_parallelism": 1, "layer_parallelism": 1, "source_weight_row_chunk": ROW_CHUNK, "torch_threads": 1, "python_ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "max_child_vm_hwm_kib": max_child_hwm},
            "decision": "measure mixed-layer depth accumulation after adding the missing external-residual self-attention sequence hook; do not infer depth4 from depth3",
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"report": str(args.output / "report.json"), "maximum_faithful_depth": 3, "depth4": "blocked_not_run"}, sort_keys=True))
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"depth accumulation failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
