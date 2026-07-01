#!/usr/bin/env python3
"""Run AQ codebook/local-scale/global-scale alternating optimization experiments."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open


SCHEMA_VERSION = "aq-codebook-opt-experiment-v0.1"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_sampler_module():
    module_path = Path(__file__).with_name("run-aq-tensor-sample.py")
    spec = importlib.util.spec_from_file_location("run_aq_tensor_sample", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_threads = max(1, min(os.cpu_count() or 1, 64))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", action="append", help="Candidate ID to run; can be repeated.")
    parser.add_argument("--tensor-pattern", default=r"\.weight$")
    parser.add_argument("--family", action="append", help="Family filter; can be repeated.")
    parser.add_argument("--max-tensors", type=int, default=8)
    parser.add_argument("--max-tensors-per-family", type=int, default=None)
    parser.add_argument("--max-elements-per-tensor", type=int, default=262144)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--scale-window", type=int, default=4)
    parser.add_argument(
        "--global-scale-dtype",
        choices=("fp16", "fp32"),
        default="fp32",
        help="Quantization dtype for the updated global-scale.",
    )
    parser.add_argument("--torch-threads", type=int, default=default_threads)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def quantize_global_scale(value: float, dtype: str) -> float:
    if dtype == "fp16":
        return float(torch.tensor(value, dtype=torch.float16).to(torch.float32))
    return float(torch.tensor(value, dtype=torch.float32))


def chunk_slices(count: int, chunk: int) -> Iterable[tuple[int, int]]:
    start = 0
    while start < count:
        end = min(count, start + chunk)
        yield start, end
        start = end


def assign_codebook_and_local_scale(
    groups: torch.Tensor,
    global_scale: float,
    scales: torch.Tensor,
    codebook: torch.Tensor,
    scale_window: int,
    assignment_chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign local-scale index and codebook-index for each block."""
    scales = scales.to(groups.device)
    codebook = codebook.to(groups.device)
    n_blocks, block_size = groups.shape
    if block_size <= 0:
        raise ValueError("block-size must be positive")

    gs = float(global_scale)
    if not math.isfinite(gs) or gs <= 0:
        gs = 1.0
    max_code = float(codebook.abs().max().clamp_min(1e-12))
    local_indices = torch.empty((n_blocks,), dtype=torch.long, device=groups.device)
    code_indices = torch.empty((n_blocks, block_size), dtype=torch.long, device=groups.device)

    for start, end in chunk_slices(n_blocks, assignment_chunk):
        block = groups[start:end].to(torch.float32)
        absmax = block.abs().amax(dim=1)
        target_scale = (absmax / (max_code * gs)).clamp(scales[0].item(), scales[-1].item())
        center = torch.searchsorted(scales, target_scale)
        center = center.clamp(0, scales.numel() - 1)
        best_error = torch.full((block.shape[0],), float("inf"), dtype=torch.float32, device=groups.device)
        best_local_idx = torch.zeros((block.shape[0],), dtype=torch.long, device=groups.device)
        best_code_idx = torch.zeros((block.shape[0], block.shape[1]), dtype=torch.long, device=groups.device)

        for offset in range(-scale_window, scale_window + 1):
            local_idx = (center + offset).clamp(0, scales.numel() - 1)
            local_scale = scales.index_select(0, local_idx).to(torch.float32)
            normalized = block / (gs * local_scale[:, None])
            nearest = (normalized[:, :, None] - codebook[None, None, :]).abs().argmin(dim=2)
            recon = torch.gather(codebook, 0, nearest.reshape(-1)).view_as(block) * local_scale[:, None] * gs
            error = (block - recon).square().sum(dim=1)
            replace = error < best_error
            if bool(replace.any()):
                best_error[replace] = error[replace]
                best_local_idx[replace] = local_idx[replace]
                best_code_idx[replace] = nearest[replace]

        local_indices[start:end] = best_local_idx
        code_indices[start:end] = best_code_idx

    return local_indices, code_indices


def update_codebook_chunked(
    groups: torch.Tensor,
    global_scale: float,
    old_codebook: torch.Tensor,
    local_indices: torch.Tensor,
    code_indices: torch.Tensor,
    scales: torch.Tensor,
    codebook_size: int = 16,
    assignment_chunk: int = 1024,
    fixed_zero: bool = False,
) -> torch.Tensor:
    """Least-squares codebook update by chunked accumulation."""
    gs = float(global_scale)
    if not math.isfinite(gs) or gs <= 0:
        gs = 1.0
    scales = scales.to(groups.device)
    old_codebook = old_codebook.to(groups.device).to(torch.float32)
    n_blocks = groups.shape[0]
    numer = torch.zeros((codebook_size,), dtype=torch.float64, device=groups.device)
    denom = torch.zeros((codebook_size,), dtype=torch.float64, device=groups.device)
    for start, end in chunk_slices(n_blocks, assignment_chunk):
        block = groups[start:end].to(torch.float32)
        local_scale = scales.index_select(0, local_indices[start:end]).to(torch.float32)
        code_index = code_indices[start:end]
        factor = (gs * local_scale[:, None]).expand_as(block).reshape(-1).to(torch.float64)
        flat_values = block.reshape(-1).to(torch.float64)
        flat_codes = code_index.reshape(-1)
        for code_idx in range(codebook_size):
            mask = flat_codes == code_idx
            if bool(mask.any()):
                selected_values = flat_values[mask]
                selected_factors = factor[mask]
                numer[code_idx] += (selected_values * selected_factors).sum()
                denom[code_idx] += (selected_factors * selected_factors).sum()

    new_codebook = old_codebook.to(torch.float32).clone()
    has_samples = denom > 0
    new_codebook[has_samples] = (numer[has_samples] / denom[has_samples]).to(torch.float32)
    if fixed_zero:
        zero_index = int(old_codebook.abs().argmin())
        new_codebook[zero_index] = 0.0
    return new_codebook.to(torch.float16).to(torch.float32)


def update_global_scale_chunked(
    groups: torch.Tensor,
    local_indices: torch.Tensor,
    code_indices: torch.Tensor,
    scales: torch.Tensor,
    codebook: torch.Tensor,
    fallback: float,
    assignment_chunk: int = 1024,
) -> float:
    """Closed-form least-squares scalar for current assignments."""
    n_blocks = groups.shape[0]
    scales = scales.to(groups.device)
    num = torch.tensor(0.0, dtype=torch.float64)
    den = torch.tensor(0.0, dtype=torch.float64)
    for start, end in chunk_slices(n_blocks, assignment_chunk):
        block = groups[start:end].to(torch.float64)
        local_scale = scales.index_select(0, local_indices[start:end]).to(torch.float64)
        code = codebook.index_select(0, code_indices[start:end].reshape(-1)).to(torch.float64).view_as(block.to(torch.float64))
        factor = (local_scale[:, None] * code).reshape(-1)
        values = block.reshape(-1)
        num += (values * factor).sum()
        den += (factor * factor).sum()

    den_value = float(den)
    if den_value <= 0 or not math.isfinite(den_value):
        return fallback
    value = float(num / den)
    if not math.isfinite(value) or value <= 0:
        return fallback
    return value


def metrics_from_assignments(
    groups: torch.Tensor,
    global_scale: float,
    local_indices: torch.Tensor,
    code_indices: torch.Tensor,
    scales: torch.Tensor,
    codebook: torch.Tensor,
    assignment_chunk: int = 1024,
) -> dict[str, float | int]:
    block_count = groups.shape[0]
    scales = scales.to(groups.device)
    gs = float(global_scale)
    if not math.isfinite(gs) or gs <= 0:
        gs = 1.0

    total_sse = torch.tensor(0.0, dtype=torch.float64)
    raw_sse = torch.tensor(0.0, dtype=torch.float64)
    block_sse_sum = torch.tensor(0.0, dtype=torch.float64)
    dot = torch.tensor(0.0, dtype=torch.float64)
    raw_norm_sq = torch.tensor(0.0, dtype=torch.float64)
    recon_norm_sq = torch.tensor(0.0, dtype=torch.float64)
    max_abs = torch.tensor(0.0, dtype=torch.float64)
    zero_mask = groups == 0
    zero_total = torch.tensor(0, dtype=torch.long)
    zero_recon = torch.tensor(0, dtype=torch.long)

    group_sse_values: list[float] = []
    for start, end in chunk_slices(block_count, assignment_chunk):
        block = groups[start:end].to(torch.float64)
        local_scale = scales.index_select(0, local_indices[start:end]).to(torch.float64)
        code = codebook.index_select(0, code_indices[start:end].reshape(-1)).to(torch.float64).view_as(block)
        recon = local_scale[:, None] * code * gs
        diff = block - recon
        sq = diff.square()
        block_sse = sq.sum(dim=1)
        total_sse += block_sse.sum()
        raw_sse += block.square().sum()
        block_sse_sum += block_sse.sum()
        dot += (block * recon).sum()
        raw_norm_sq += block.square().sum()
        recon_norm_sq += recon.square().sum()
        max_abs = torch.maximum(max_abs, sq.max().sqrt())

        chunk_zero = zero_mask[start:end]
        if bool(chunk_zero.any()):
            zero_total += chunk_zero.sum().to(torch.long)
            zero_recon += (recon[chunk_zero] == 0).to(torch.long).sum()
        group_sse_values.extend(block_sse.tolist())

    group_sse_tensor = torch.tensor(group_sse_values, dtype=torch.float64)
    mean_sse_per_block = float(block_sse_sum / block_count) if block_count > 0 else 0.0
    p95_group_error = float(group_sse_tensor.quantile(0.95).item()) if group_sse_values else 0.0
    rel_mse = float(total_sse / raw_sse) if float(raw_sse) > 0 else 0.0
    norm = float(torch.sqrt(raw_norm_sq * recon_norm_sq))
    metrics: dict[str, float | int] = {
        "mse": float(total_sse / groups.numel()) if groups.numel() > 0 else 0.0,
        "relative_mse": rel_mse,
        "max_abs_error": float(max_abs),
        "cosine_similarity": float(dot / norm) if norm > 0 else 1.0,
        "mean_group_error": mean_sse_per_block,
        "p95_group_error": p95_group_error,
        "sampled_blocks": int(block_count),
        "sampled_elements": int(groups.numel()),
        "global_scale": gs,
    }
    if int(zero_total) > 0:
        metrics["zero_preservation_rate"] = float(int(zero_recon) / int(zero_total))
    return metrics


def estimate_operations(
    blocks: int,
    block_size: int,
    scale_candidates: int,
    codebook_size: int,
    scale_window: int,
    iterations: int,
) -> dict[str, int | float]:
    # Rough counts for comparison, not cycle-accurate.
    baseline_ops = blocks * (2 * scale_window + 1) * block_size * codebook_size
    assign_iter_ops = baseline_ops  # one nearest + code search pass
    codebook_update_ops = blocks * block_size  # accumulate least-squares numerator/denominator
    global_update_ops = blocks * block_size  # num/den update
    iter_ops = assign_iter_ops + codebook_update_ops + global_update_ops
    final_assignment_ops = baseline_ops
    exhaustive_ops = blocks * scale_candidates * block_size * codebook_size
    alternating_ops = iter_ops * iterations + final_assignment_ops
    return {
        "scale_candidates": int(scale_candidates),
        "baseline_ops": int(baseline_ops),
        "alternating_per_iteration_ops": int(iter_ops),
        "final_assignment_ops": int(final_assignment_ops),
        "alternating_ops": int(alternating_ops),
        "exhaustive_local_assignment_ops": int(exhaustive_ops),
        "alternating_to_baseline_ratio": float(alternating_ops / max(1, baseline_ops)),
        "exhaustive_to_baseline_ratio": float(exhaustive_ops / max(1, baseline_ops)),
        "exhaustive_to_alternating_ratio": float(exhaustive_ops / max(1, alternating_ops)),
    }


def run_exhaustive_local_assignment(
    groups: torch.Tensor,
    global_scale: float,
    scales: torch.Tensor,
    codebook: torch.Tensor,
    assignment_chunk: int = 128,
) -> None:
    gs = float(global_scale)
    if not math.isfinite(gs) or gs <= 0:
        gs = 1.0
    scales = scales.to(groups.device)
    codebook = codebook.to(groups.device)
    n_scales = scales.shape[0]
    n_blocks = groups.shape[0]
    for start, end in chunk_slices(n_blocks, assignment_chunk):
        block = groups[start:end].to(torch.float32)
        recon = gs * scales.view(1, 1, n_scales, 1) * codebook.view(1, 1, 1, -1)
        errors = (block[:, :, None, None] - recon).square().sum(dim=1)
        _ = errors.reshape(block.shape[0], -1).argmin(dim=1)


def run_candidate(
    sampler,
    args: argparse.Namespace,
    tensor_name: str,
    tensor: torch.Tensor,
    candidate,
    seed: int,
    assignment_chunk: int = 1024,
) -> dict[str, Any]:
    import time

    if tensor.ndim < 2 or not tensor.is_floating_point():
        raise ValueError(f"{tensor_name} is not a float 2D+ tensor")

    scales = sampler.scale_values(candidate.scale_format)
    run_seed = seed
    timing: dict[str, Any] = {}
    total_start = time.perf_counter()

    sample_start = time.perf_counter()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(run_seed)
    groups, _ = sampler.sample_groups_with_columns(
        tensor,
        candidate.group_size,
        args.max_elements_per_tensor,
        generator,
    )
    timing["sampling_sec"] = time.perf_counter() - sample_start

    baseline_start = time.perf_counter()
    baseline = sampler.evaluate_candidate(groups, candidate, args.scale_window)
    timing["baseline_sec"] = time.perf_counter() - baseline_start

    alternating_start = time.perf_counter()
    codebook = sampler.codebook_from_groups(groups, candidate.codebook_mode).to(torch.float16).to(torch.float32)
    global_scale = float(sampler.choose_tensor_scale(groups, candidate, scales, codebook))
    global_scale = quantize_global_scale(global_scale, args.global_scale_dtype)
    fixed_zero = candidate.codebook_mode.startswith("zero_")

    iter_timings = []
    local_indices = torch.empty((groups.shape[0],), dtype=torch.long, device=groups.device)
    code_indices = torch.empty((groups.shape[0],), dtype=torch.long, device=groups.device)
    for iteration in range(args.iterations):
        it_start = time.perf_counter()
        local_indices, code_indices = assign_codebook_and_local_scale(
            groups,
            global_scale,
            scales,
            codebook,
            args.scale_window,
            assignment_chunk=assignment_chunk,
        )
        codebook = update_codebook_chunked(
            groups,
            global_scale,
            codebook,
            local_indices,
            code_indices,
            scales,
            assignment_chunk=assignment_chunk,
            fixed_zero=fixed_zero,
        )
        next_scale = update_global_scale_chunked(
            groups,
            local_indices,
            code_indices,
            scales,
            codebook,
            fallback=global_scale,
            assignment_chunk=assignment_chunk,
        )
        next_scale = quantize_global_scale(next_scale, args.global_scale_dtype)
        if next_scale > 0 and math.isfinite(next_scale):
            global_scale = float(next_scale)
        iter_timings.append({"iteration": iteration, "elapsed_sec": time.perf_counter() - it_start})

    final_assign_start = time.perf_counter()
    local_indices, code_indices = assign_codebook_and_local_scale(
        groups,
        global_scale,
        scales,
        codebook,
        args.scale_window,
        assignment_chunk=assignment_chunk,
    )
    timing["final_assignment_sec"] = time.perf_counter() - final_assign_start
    timing["alternating_sec"] = time.perf_counter() - alternating_start
    timing["iterations"] = iter_timings

    alt = metrics_from_assignments(
        groups,
        global_scale,
        local_indices,
        code_indices,
        scales,
        codebook,
        assignment_chunk=assignment_chunk,
    )
    alt["codebook"] = [float(v) for v in codebook.to(torch.float32).tolist()]
    alt["iterations"] = args.iterations
    alt["local_scale_format"] = candidate.scale_format
    alt["codebook_mode"] = candidate.codebook_mode
    alt["final_global_scale"] = float(global_scale)
    alt["effective_bpp"] = 4.0 + 8.0 / candidate.group_size

    exhaustive = None
    if groups.shape[0] <= 2048:
        exp_start = time.perf_counter()
        run_exhaustive_local_assignment(groups, global_scale, scales, codebook)
        exhaustive = time.perf_counter() - exp_start
    timing["exhaustive_local_assignment_sec"] = exhaustive
    timing["total_sec"] = time.perf_counter() - total_start

    return {
        "candidate_id": candidate.candidate_id,
        "status": "ok",
        "timing_sec": timing,
        "baseline": baseline,
        "alternating": alt,
        "operation_counts": estimate_operations(
            blocks=groups.shape[0],
            block_size=candidate.group_size,
            scale_candidates=scales.numel(),
            codebook_size=codebook.numel(),
            scale_window=args.scale_window,
            iterations=args.iterations,
        ),
        "seed": run_seed,
        "sampled_blocks": int(groups.shape[0]),
        "sampled_elements": int(groups.numel()),
    }


def main() -> int:
    args = parse_args()
    if args.max_tensors_per_family is not None and args.max_tensors_per_family < 1:
        raise SystemExit("--max-tensors-per-family must be >= 1")
    if args.max_tensors < 1:
        raise SystemExit("--max-tensors must be >= 1")
    if args.max_elements_per_tensor < 1:
        raise SystemExit("--max-elements-per-tensor must be >= 1")
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if args.scale_window < 0:
        raise SystemExit("--scale-window must be >= 0")
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be >= 1")
    if args.torch_interop_threads < 1:
        raise SystemExit("--torch-interop-threads must be >= 1")

    sampler = load_sampler_module()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)

    candidates = list(sampler.ROUND1_CANDIDATES)
    if args.candidate:
        selected = set(args.candidate)
        candidates = [candidate for candidate in candidates if candidate.candidate_id in selected]
        missing = selected - {candidate.candidate_id for candidate in candidates}
        if missing:
            raise SystemExit(f"unknown candidate IDs: {', '.join(sorted(missing))}")

    tensor_pattern = re.compile(args.tensor_pattern)
    tensors = sampler.discover_tensors(args.model_dir, tensor_pattern)
    if args.family:
        allowed = set(args.family)
        tensors = [(name, path) for name, path in tensors if sampler.family_for_tensor(name) in allowed]
    tensors = sampler.limit_tensors_by_family(tensors, args.max_tensors, args.max_tensors_per_family)
    if not tensors:
        raise SystemExit("no tensors matched")

    assignment_chunk = 1024
    tensor_rows = []
    for tensor_name, path in tensors:
        family = sampler.family_for_tensor(tensor_name)
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(tensor_name)
        if tensor.ndim < 2 or not tensor.is_floating_point():
            continue
        tensor_shape = tuple(int(dim) for dim in tensor.shape)
        tensor_dtype = str(tensor.dtype).replace("torch.", "")

        candidate_results = []
        for idx, candidate in enumerate(candidates):
            try:
                candidate_row = run_candidate(
                    sampler,
                    args,
                    tensor_name,
                    tensor,
                    candidate,
                    seed=args.seed + idx * 9973,
                    assignment_chunk=assignment_chunk,
                )
            except Exception as exc:  # noqa: BLE001
                candidate_row = {
                    "candidate_id": candidate.candidate_id,
                    "status": "failed",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            candidate_results.append(candidate_row)

        tensor_rows.append(
            {
                "tensor_name": tensor_name,
                "family": family,
                "tensor_shape": list(tensor_shape),
                "tensor_dtype": tensor_dtype,
                "candidate_results": candidate_results,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "settings": {
            "model_dir": str(args.model_dir),
            "tensor_pattern": args.tensor_pattern,
            "families": args.family,
            "candidate": args.candidate,
            "max_tensors": args.max_tensors,
            "max_tensors_per_family": args.max_tensors_per_family,
            "max_elements_per_tensor": args.max_elements_per_tensor,
            "iterations": args.iterations,
            "scale_window": args.scale_window,
            "global_scale_dtype": args.global_scale_dtype,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
            "seed": args.seed,
        },
        "results": tensor_rows,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
