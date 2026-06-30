#!/usr/bin/env python3
"""Chunked Python reference for one aq tensor dry-run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def load_sampler_module():
    path = Path(__file__).resolve().parent / "run-aq-tensor-sample.py"
    spec = importlib.util.spec_from_file_location("aq_tensor_sampler", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load sampler module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_codebook(path: Path, family: str, candidate_id: str) -> torch.Tensor:
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data.get("codebooks", []):
        if entry.get("family") == family and entry.get("candidate_id") == candidate_id:
            values = entry.get("values_f32", [])
            if len(values) != 16:
                raise ValueError(
                    f"codebook for family={family}, candidate={candidate_id} "
                    f"has {len(values)} entries, expected 16"
                )
            return torch.tensor(values, dtype=torch.float32)
    raise KeyError(f"codebook not found for family={family}, candidate={candidate_id}")


def find_tensor_path(sampler, model_dir: Path, tensor_name: str) -> Path:
    for path in sampler.iter_safetensor_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as handle:
            if tensor_name in handle.keys():
                return path
    raise KeyError(f"tensor {tensor_name} not found under {model_dir}")


def lower_median(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        raise ValueError("cannot take median of empty tensor")
    return values.sort().values[(values.numel() - 1) // 2]


def choose_tensor_scale_chunked(
    groups: torch.Tensor,
    scales: torch.Tensor,
    codebook: torch.Tensor,
    group_chunk: int,
) -> float:
    max_code = codebook.abs().max().clamp_min(1e-12)
    chunks: list[torch.Tensor] = []
    for start in range(0, groups.shape[0], group_chunk):
        end = min(start + group_chunk, groups.shape[0])
        target = groups[start:end].abs().amax(dim=1) / max_code
        target = target[target > 0]
        if target.numel() > 0:
            chunks.append(target.cpu())
    if not chunks:
        return 1.0
    target_median = float(lower_median(torch.cat(chunks)))
    scale_median = float(lower_median(scales.cpu()))
    if not torch.isfinite(torch.tensor(target_median)) or target_median <= 0:
        return 1.0
    if not torch.isfinite(torch.tensor(scale_median)) or scale_median <= 0:
        return 1.0
    tensor_scale = target_median / scale_median
    return tensor_scale if tensor_scale > 0 else 1.0


def verify_tensor(args: argparse.Namespace) -> dict:
    sampler = load_sampler_module()
    candidate = next(
        item for item in sampler.ROUND1_CANDIDATES if item.candidate_id == args.candidate
    )
    codebook = load_codebook(args.codebook_json, args.family, args.candidate)
    scales = sampler.scale_values(candidate.scale_format)
    tensor_path = find_tensor_path(sampler, args.model_dir, args.tensor)

    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(args.tensor)
    tensor_dtype = str(tensor.dtype)
    tensor_shape = [int(dim) for dim in tensor.shape]
    flat = tensor.to(torch.float32).flatten()
    usable = (flat.numel() // candidate.group_size) * candidate.group_size
    if usable == 0:
        raise ValueError("tensor has no full quantization group")
    groups = flat[:usable].view(-1, candidate.group_size)

    tensor_scale = (
        choose_tensor_scale_chunked(groups, scales, codebook, args.group_chunk)
        if candidate.tensor_scale != "none"
        else 1.0
    )
    max_code = codebook.abs().max().clamp_min(1e-12)
    index_counts = torch.zeros(codebook.numel(), dtype=torch.long)
    sse = 0.0
    ref_sse = 0.0
    max_abs_error = 0.0
    scale_index_min = scales.numel()
    scale_index_max = 0
    improved_groups = 0

    with torch.no_grad():
        for start in range(0, groups.shape[0], args.group_chunk):
            end = min(start + args.group_chunk, groups.shape[0])
            chunk = groups[start:end]
            scaled = chunk / tensor_scale
            target_scale = scaled.abs().amax(dim=1) / max_code
            center = sampler.nearest_scale_indices(target_scale, scales)
            best_error = torch.full((chunk.shape[0],), torch.inf, dtype=torch.float32)
            best_recon = torch.zeros_like(chunk)
            best_nearest = torch.zeros_like(chunk, dtype=torch.long)
            best_scale_index = torch.zeros((chunk.shape[0],), dtype=torch.long)

            for offset in range(-args.scale_window, args.scale_window + 1):
                idx = (center + offset).clamp(0, scales.numel() - 1)
                group_scale = scales.index_select(0, idx)
                normalized = scaled / group_scale[:, None]
                nearest = (normalized[:, :, None] - codebook[None, None, :]).abs().argmin(dim=2)
                quantized = codebook.index_select(0, nearest.flatten()).view_as(chunk)
                recon = quantized * group_scale[:, None] * tensor_scale
                square_error = (chunk - recon).square()
                error = square_error.sum(dim=1)
                mask = error < best_error
                best_error = torch.where(mask, error, best_error)
                best_recon = torch.where(mask[:, None], recon, best_recon)
                best_nearest = torch.where(mask[:, None], nearest, best_nearest)
                best_scale_index = torch.where(mask, idx, best_scale_index)

            diff = chunk - best_recon
            sse += float(diff.square().sum())
            ref_sse += float(chunk.square().sum())
            max_abs_error = max(max_abs_error, float(diff.abs().max()))
            index_counts += torch.bincount(best_nearest.flatten(), minlength=codebook.numel())
            improved_groups += int((best_scale_index != center).sum())
            scale_index_min = min(scale_index_min, int(best_scale_index.min()))
            scale_index_max = max(scale_index_max, int(best_scale_index.max()))

    elements = int(groups.numel())
    return {
        "schema_version": "aq-one-tensor-python-verify-v0.1",
        "model_dir": str(args.model_dir),
        "tensor": args.tensor,
        "tensor_path": str(tensor_path),
        "tensor_dtype": tensor_dtype,
        "tensor_shape": tensor_shape,
        "candidate": args.candidate,
        "family": args.family,
        "scale_window": args.scale_window,
        "group_chunk": args.group_chunk,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "tensor_scale": tensor_scale,
        "elements": elements,
        "groups": int(groups.shape[0]),
        "mse": sse / elements,
        "relative_mse": sse / ref_sse if ref_sse > 0 else 0.0,
        "max_abs_error": max_abs_error,
        "scale_index_min": scale_index_min,
        "scale_index_max": scale_index_max,
        "scale_window_improved_groups": improved_groups,
        "index_counts": [int(value) for value in index_counts.tolist()],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--codebook-json", type=Path, required=True)
    parser.add_argument("--scale-window", type=int, default=4)
    parser.add_argument("--group-chunk", type=int, default=32768)
    parser.add_argument("--torch-threads", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    result = verify_tensor(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
