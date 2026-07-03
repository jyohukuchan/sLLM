#!/usr/bin/env python3
"""Try same-bpp AQ compensation variants on IQ4 replacement rows."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aq_scale_formats import scale_values  # noqa: E402


SCHEMA_VERSION = "aq-iq4-compensation-experiment-v0.1"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_threads = max(1, min(os.cpu_count() or 1, 64))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ggml-type", action="append", default=["IQ4_XS"])
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--scale-format", default="ue3m5")
    parser.add_argument(
        "--floor-scale-format",
        action="append",
        default=["e3m4"],
        help="Run these lower/alternate scale formats first and pass them as monotonic floor states.",
    )
    parser.add_argument("--codebook-token", default="flloyd16")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--scale-window", type=int, default=4)
    parser.add_argument("--global-scale-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--max-elements-per-tensor", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=2501)
    parser.add_argument("--assignment-chunk", type=int, default=1024)
    parser.add_argument("--torch-threads", type=int, default=default_threads)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    return parser.parse_args()


def load_source_rows(path: Path, ggml_types: set[str] | None, max_rows: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            quant = row.get("quantization", {})
            if ggml_types and quant.get("ggml_type") not in ggml_types:
                continue
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def tensor_name_from_source(row: dict[str, Any]) -> str:
    names = row.get("scope", {}).get("tensor_names", [])
    if not names:
        raise ValueError("source row does not contain scope.tensor_names")
    return str(names[0])


def tensor_path_map(sampler, model_dir: Path) -> dict[str, Path]:
    import re

    tensors = sampler.discover_tensors(model_dir, re.compile(r"\.weight$"))
    return {name: path for name, path in tensors}


def chunk_slices(count: int, chunk: int) -> Iterable[tuple[int, int]]:
    start = 0
    while start < count:
        end = min(count, start + chunk)
        yield start, end
        start = end


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def weighted_mean(values: list[tuple[float, float]]) -> float:
    weight = sum(item[1] for item in values)
    if weight <= 0:
        return float("nan")
    return float(sum(value * item_weight for value, item_weight in values) / weight)


def quantize_global_scale(value: float, dtype: str) -> float:
    if dtype == "fp16":
        return float(torch.tensor(value, dtype=torch.float16).to(torch.float32))
    return float(torch.tensor(value, dtype=torch.float32))


def make_scale_subset(scales: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 8:
        return scales.to(torch.float32)
    count = 1 << bits
    if count >= scales.numel():
        return scales.to(torch.float32)
    positions = torch.linspace(0, scales.numel() - 1, count)
    indices = positions.round().to(torch.long).unique(sorted=True)
    cursor = 0
    while indices.numel() < count and cursor < scales.numel():
        extra = torch.tensor([cursor], dtype=torch.long)
        indices = torch.cat([indices, extra]).unique(sorted=True)
        cursor += 1
    return scales.index_select(0, indices[:count]).to(torch.float32)


def positive_codebook(values: torch.Tensor, count: int, iterations: int = 6) -> torch.Tensor:
    values = values.to(torch.float32)
    values = values[torch.isfinite(values) & (values > 0)]
    if values.numel() == 0:
        return torch.ones(count, dtype=torch.float32)
    if count >= values.numel():
        codebook = values.sort().values
        if codebook.numel() < count:
            pad = codebook[-1].repeat(count - codebook.numel())
            codebook = torch.cat([codebook, pad])
        return codebook[:count].to(torch.float32)
    q = torch.linspace(0.0, 1.0, count + 2, dtype=torch.float32)[1:-1]
    codebook = torch.quantile(values, q).to(torch.float32).clamp_min(1e-12).sort().values
    for _ in range(iterations):
        nearest = (values[:, None] - codebook[None, :]).abs().argmin(dim=1)
        updated = codebook.clone()
        for idx in range(count):
            mask = nearest == idx
            if bool(mask.any()):
                updated[idx] = values[mask].mean().clamp_min(1e-12)
        codebook = updated.sort().values
    return codebook.to(torch.float32)


def dequant_from_indices(
    groups: torch.Tensor,
    global_scale: float,
    scales: torch.Tensor,
    codebook: torch.Tensor,
    local_indices: torch.Tensor,
    code_indices: torch.Tensor,
    block_multipliers: torch.Tensor | None = None,
) -> torch.Tensor:
    local = scales.index_select(0, local_indices.to(torch.long)).to(torch.float32)
    code = codebook.index_select(0, code_indices.reshape(-1).to(torch.long)).view_as(groups).to(torch.float32)
    mult = 1.0 if block_multipliers is None else block_multipliers.to(torch.float32)
    if isinstance(mult, torch.Tensor):
        return code * local[:, None] * float(global_scale) * mult[:, None]
    return code * local[:, None] * float(global_scale)


def metrics_from_recon(groups: torch.Tensor, recon: torch.Tensor) -> dict[str, float | int]:
    diff = (groups - recon).to(torch.float64)
    raw = groups.to(torch.float64)
    sse = float(diff.square().sum())
    ref = float(raw.square().sum())
    block_sse = diff.square().sum(dim=1)
    dot = float((raw * recon.to(torch.float64)).sum())
    norm = float(raw.square().sum().sqrt() * recon.to(torch.float64).square().sum().sqrt())
    return {
        "mse": float(sse / groups.numel()) if groups.numel() else 0.0,
        "relative_mse": float(sse / ref) if ref > 0 else 0.0,
        "max_abs_error": float(diff.abs().max()) if groups.numel() else 0.0,
        "cosine_similarity": float(dot / norm) if norm > 0 else 1.0,
        "mean_group_error": float(block_sse.mean()) if block_sse.numel() else 0.0,
        "p95_group_error": float(torch.quantile(block_sse, 0.95)) if block_sse.numel() else 0.0,
        "sampled_blocks": int(groups.shape[0]),
        "sampled_elements": int(groups.numel()),
    }


def assign_single_codebook(
    groups: torch.Tensor,
    global_scale: float,
    scales: torch.Tensor,
    codebook: torch.Tensor,
    scale_window: int,
    block_multipliers: torch.Tensor | None = None,
    target_mode: str = "absmax",
    assignment_chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_blocks, block_size = groups.shape
    scales = scales.to(torch.float32)
    codebook = codebook.to(torch.float32)
    local_indices = torch.empty((n_blocks,), dtype=torch.long)
    code_indices = torch.empty((n_blocks, block_size), dtype=torch.long)
    block_errors = torch.empty((n_blocks,), dtype=torch.float32)
    recon_out = torch.empty_like(groups, dtype=torch.float32)
    mult = torch.ones((n_blocks,), dtype=torch.float32) if block_multipliers is None else block_multipliers.to(torch.float32)
    gs = max(float(global_scale), 1e-30)
    max_code = float(codebook.abs().max().clamp_min(1e-12))

    for start, end in chunk_slices(n_blocks, assignment_chunk):
        block = groups[start:end].to(torch.float32)
        block_mult = mult[start:end]
        if target_mode == "p95":
            target_abs = torch.quantile(block.abs(), 0.95, dim=1)
        elif target_mode == "top2":
            sorted_abs = block.abs().sort(dim=1).values
            target_abs = sorted_abs[:, -2] if block.shape[1] >= 2 else sorted_abs[:, -1]
        else:
            target_abs = block.abs().amax(dim=1)
        target_scale = (target_abs / (max_code * gs * block_mult).clamp_min(1e-30)).clamp(
            float(scales[0]),
            float(scales[-1]),
        )
        center = torch.searchsorted(scales, target_scale).clamp(0, scales.numel() - 1)
        best_error = torch.full((block.shape[0],), float("inf"), dtype=torch.float32)
        best_local = torch.zeros((block.shape[0],), dtype=torch.long)
        best_code = torch.zeros((block.shape[0], block_size), dtype=torch.long)
        best_recon = torch.zeros_like(block)
        for offset in range(-scale_window, scale_window + 1):
            local_idx = (center + offset).clamp(0, scales.numel() - 1)
            local = scales.index_select(0, local_idx).to(torch.float32)
            denom = (gs * block_mult * local).clamp_min(1e-30)
            normalized = block / denom[:, None]
            nearest = (normalized[:, :, None] - codebook[None, None, :]).abs().argmin(dim=2)
            code = codebook.index_select(0, nearest.reshape(-1)).view_as(block)
            recon = code * denom[:, None]
            error = (block - recon).square().sum(dim=1)
            mask = error < best_error
            best_error = torch.where(mask, error, best_error)
            best_local = torch.where(mask, local_idx, best_local)
            best_code = torch.where(mask[:, None], nearest, best_code)
            best_recon = torch.where(mask[:, None], recon, best_recon)
        local_indices[start:end] = best_local
        code_indices[start:end] = best_code
        block_errors[start:end] = best_error
        recon_out[start:end] = best_recon

    return local_indices, code_indices, recon_out, block_errors


def update_single_codebook(
    groups: torch.Tensor,
    global_scale: float,
    scales: torch.Tensor,
    local_indices: torch.Tensor,
    code_indices: torch.Tensor,
    codebook_size: int,
    block_multipliers: torch.Tensor | None = None,
) -> torch.Tensor:
    mult = torch.ones((groups.shape[0],), dtype=torch.float32) if block_multipliers is None else block_multipliers.to(torch.float32)
    local = scales.index_select(0, local_indices.to(torch.long)).to(torch.float64)
    factor = (float(global_scale) * mult.to(torch.float64)[:, None] * local[:, None]).expand_as(groups).reshape(-1)
    values = groups.to(torch.float64).reshape(-1)
    codes = code_indices.reshape(-1)
    out = torch.zeros((codebook_size,), dtype=torch.float64)
    for idx in range(codebook_size):
        mask = codes == idx
        if bool(mask.any()):
            f = factor[mask]
            denom = (f * f).sum()
            if float(denom) > 0:
                out[idx] = (values[mask] * f).sum() / denom
    return out.to(torch.float16).to(torch.float32).sort().values


def update_global_scale(
    groups: torch.Tensor,
    scales: torch.Tensor,
    codebook: torch.Tensor,
    local_indices: torch.Tensor,
    code_indices: torch.Tensor,
    fallback: float,
    dtype: str,
    block_multipliers: torch.Tensor | None = None,
) -> float:
    mult = torch.ones((groups.shape[0],), dtype=torch.float32) if block_multipliers is None else block_multipliers.to(torch.float32)
    local = scales.index_select(0, local_indices.to(torch.long)).to(torch.float64)
    code = codebook.index_select(0, code_indices.reshape(-1).to(torch.long)).to(torch.float64).view_as(groups)
    factor = (mult.to(torch.float64)[:, None] * local[:, None] * code).reshape(-1)
    values = groups.to(torch.float64).reshape(-1)
    denom = float((factor * factor).sum())
    if denom <= 0 or not math.isfinite(denom):
        return fallback
    value = float((values * factor).sum() / denom)
    if value <= 0 or not math.isfinite(value):
        return fallback
    return quantize_global_scale(value, dtype)


def update_global_scale_dual(
    groups: torch.Tensor,
    scales: torch.Tensor,
    codebook0: torch.Tensor,
    codebook1: torch.Tensor,
    local_indices: torch.Tensor,
    code_indices: torch.Tensor,
    selector: torch.Tensor,
    fallback: float,
    dtype: str,
) -> float:
    local = scales.index_select(0, local_indices.to(torch.long)).to(torch.float64)
    code = torch.empty_like(groups, dtype=torch.float64)
    for flag, codebook in [(False, codebook0), (True, codebook1)]:
        mask = selector == flag
        if bool(mask.any()):
            selected_codes = codebook.index_select(0, code_indices[mask].reshape(-1).to(torch.long))
            code[mask] = selected_codes.to(torch.float64).view_as(groups[mask])
    factor = (local[:, None] * code).reshape(-1)
    values = groups.to(torch.float64).reshape(-1)
    denom = float((factor * factor).sum())
    if denom <= 0 or not math.isfinite(denom):
        return fallback
    value = float((values * factor).sum() / denom)
    if value <= 0 or not math.isfinite(value):
        return fallback
    return quantize_global_scale(value, dtype)


def optimize_single_variant(
    groups: torch.Tensor,
    initial_global_scale: float,
    scales: torch.Tensor,
    initial_codebook: torch.Tensor,
    iterations: int,
    scale_window: int,
    global_scale_dtype: str,
    target_mode: str = "absmax",
    assignment_chunk: int = 1024,
) -> dict[str, Any]:
    codebook = initial_codebook.to(torch.float32).clone()
    global_scale = float(initial_global_scale)
    for _ in range(iterations):
        local_idx, code_idx, _, _ = assign_single_codebook(
            groups,
            global_scale,
            scales,
            codebook,
            scale_window,
            target_mode=target_mode,
            assignment_chunk=assignment_chunk,
        )
        codebook = update_single_codebook(groups, global_scale, scales, local_idx, code_idx, codebook.numel())
        global_scale = update_global_scale(
            groups,
            scales,
            codebook,
            local_idx,
            code_idx,
            global_scale,
            global_scale_dtype,
        )
    local_idx, code_idx, recon, _ = assign_single_codebook(
        groups,
        global_scale,
        scales,
        codebook,
        scale_window,
        target_mode=target_mode,
        assignment_chunk=assignment_chunk,
    )
    metrics = metrics_from_recon(groups, recon)
    return {
        "metrics": metrics,
        "global_scale": global_scale,
        "codebook": [float(v) for v in codebook.tolist()],
        "_local_indices": local_idx,
        "_code_indices": code_idx,
        "_recon": recon,
    }


def superblock_ids(blocks: int, blocks_per_superblock: int) -> torch.Tensor:
    return torch.arange(blocks, dtype=torch.long) // blocks_per_superblock


def expand_superblock_values(values: torch.Tensor, blocks: int, blocks_per_superblock: int) -> torch.Tensor:
    ids = superblock_ids(blocks, blocks_per_superblock)
    return values.index_select(0, ids)


def estimate_superblock_multipliers(
    groups: torch.Tensor,
    recon_without_multiplier: torch.Tensor,
    blocks_per_superblock: int,
) -> torch.Tensor:
    ids = superblock_ids(groups.shape[0], blocks_per_superblock)
    count = int(ids.max()) + 1 if ids.numel() else 0
    out = torch.ones((count,), dtype=torch.float32)
    for idx in range(count):
        mask = ids == idx
        raw = groups[mask].to(torch.float64).reshape(-1)
        base = recon_without_multiplier[mask].to(torch.float64).reshape(-1)
        denom = float((base * base).sum())
        if denom > 0:
            value = float((raw * base).sum() / denom)
            if value > 0 and math.isfinite(value):
                out[idx] = value
    return out.clamp_min(1e-6)


def optimize_superblock_scale_variant(
    groups: torch.Tensor,
    initial_global_scale: float,
    local_scales: torch.Tensor,
    initial_codebook: torch.Tensor,
    superblock_bits: int,
    blocks_per_superblock: int,
    iterations: int,
    scale_window: int,
    global_scale_dtype: str,
    assignment_chunk: int,
) -> dict[str, Any]:
    blocks = groups.shape[0]
    super_count = int(math.ceil(blocks / blocks_per_superblock))
    sb_values = torch.ones((super_count,), dtype=torch.float32)
    sb_codebook = torch.ones((1 << superblock_bits,), dtype=torch.float32)
    codebook = initial_codebook.to(torch.float32).clone()
    global_scale = float(initial_global_scale)

    for _ in range(iterations):
        block_mult = expand_superblock_values(sb_values, blocks, blocks_per_superblock)
        local_idx, code_idx, recon, _ = assign_single_codebook(
            groups,
            global_scale,
            local_scales,
            codebook,
            scale_window,
            block_multipliers=block_mult,
            assignment_chunk=assignment_chunk,
        )
        codebook = update_single_codebook(
            groups,
            global_scale,
            local_scales,
            local_idx,
            code_idx,
            codebook.numel(),
            block_multipliers=block_mult,
        )
        base_recon = dequant_from_indices(groups, global_scale, local_scales, codebook, local_idx, code_idx)
        targets = estimate_superblock_multipliers(groups, base_recon, blocks_per_superblock)
        sb_codebook = positive_codebook(targets, 1 << superblock_bits)
        nearest = (targets[:, None] - sb_codebook[None, :]).abs().argmin(dim=1)
        sb_values = sb_codebook.index_select(0, nearest)
        block_mult = expand_superblock_values(sb_values, blocks, blocks_per_superblock)
        global_scale = update_global_scale(
            groups,
            local_scales,
            codebook,
            local_idx,
            code_idx,
            global_scale,
            global_scale_dtype,
            block_multipliers=block_mult,
        )

    block_mult = expand_superblock_values(sb_values, blocks, blocks_per_superblock)
    local_idx, code_idx, recon, _ = assign_single_codebook(
        groups,
        global_scale,
        local_scales,
        codebook,
        scale_window,
        block_multipliers=block_mult,
        assignment_chunk=assignment_chunk,
    )
    metrics = metrics_from_recon(groups, recon)
    return {
        "metrics": metrics,
        "global_scale": global_scale,
        "superblock_bits": superblock_bits,
        "blocks_per_superblock": blocks_per_superblock,
        "superblock_count": super_count,
        "superblock_scale_min": float(sb_values.min()),
        "superblock_scale_max": float(sb_values.max()),
        "superblock_scale_codebook": [float(v) for v in sb_codebook.tolist()],
        "_recon": recon,
    }


def evaluate_scale_index_bias(
    groups: torch.Tensor,
    full_scales: torch.Tensor,
    state,
    blocks_per_superblock: int,
) -> dict[str, Any]:
    local_indices = state.local_indices.to(torch.long)
    code_indices = state.code_indices.to(torch.long)
    adjusted = local_indices.clone()
    ids = superblock_ids(groups.shape[0], blocks_per_superblock)
    over_range = 0
    exact = 0
    for idx in range(int(ids.max()) + 1 if ids.numel() else 0):
        mask = ids == idx
        values = local_indices[mask]
        low = int(values.min())
        high = int(values.max())
        if high - low > 127:
            over_range += 1
            center = int(torch.median(values.to(torch.float32)).item())
            low = max(0, min(full_scales.numel() - 128, center - 63))
            high = low + 127
            adjusted[mask] = values.clamp(low, high)
        else:
            exact += 1
    recon = dequant_from_indices(
        groups,
        state.global_scale,
        full_scales,
        state.codebook.to(torch.float32),
        adjusted,
        code_indices,
    )
    metrics = metrics_from_recon(groups, recon)
    return {
        "metrics": metrics,
        "blocks_per_superblock": blocks_per_superblock,
        "exact_superblocks": exact,
        "over_range_superblocks": over_range,
        "_recon": recon,
    }


def block_shape_scores(groups: torch.Tensor) -> torch.Tensor:
    abs_groups = groups.abs()
    top = abs_groups.amax(dim=1)
    mean_abs = abs_groups.mean(dim=1).clamp_min(1e-12)
    return top / mean_abs


def initial_dual_codebooks(sampler, groups: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = block_shape_scores(groups)
    threshold = torch.median(scores)
    selector = scores > threshold
    if not bool(selector.any()) or bool(selector.all()):
        selector = torch.arange(groups.shape[0]) % 2 == 0
    codebooks = []
    for flag in [False, True]:
        subset = groups[selector == flag]
        if subset.numel() == 0:
            subset = groups
        codebooks.append(sampler.codebook_from_groups(subset, mode).to(torch.float16).to(torch.float32))
    return codebooks[0], codebooks[1], selector.to(torch.long)


def optimize_dual_codebook_variant(
    sampler,
    groups: torch.Tensor,
    initial_global_scale: float,
    scales: torch.Tensor,
    codebook_mode: str,
    iterations: int,
    scale_window: int,
    global_scale_dtype: str,
    assignment_chunk: int,
) -> dict[str, Any]:
    cb0, cb1, selector = initial_dual_codebooks(sampler, groups, codebook_mode)
    global_scale = float(initial_global_scale)
    chosen_local = torch.zeros((groups.shape[0],), dtype=torch.long)
    chosen_codes = torch.zeros((groups.shape[0], groups.shape[1]), dtype=torch.long)
    chosen_recon = torch.zeros_like(groups)

    for _ in range(iterations):
        assignments = []
        for codebook in [cb0, cb1]:
            local_idx, code_idx, recon, errors = assign_single_codebook(
                groups,
                global_scale,
                scales,
                codebook,
                scale_window,
                assignment_chunk=assignment_chunk,
            )
            assignments.append((local_idx, code_idx, recon, errors))
        selector = assignments[1][3] < assignments[0][3]
        for idx, codebook in enumerate([cb0, cb1]):
            mask = selector == bool(idx)
            if bool(mask.any()):
                local_idx, code_idx, _, _ = assignments[idx]
                updated = update_single_codebook(
                    groups[mask],
                    global_scale,
                    scales,
                    local_idx[mask],
                    code_idx[mask],
                    codebook.numel(),
                )
                if idx == 0:
                    cb0 = updated
                else:
                    cb1 = updated
        chosen_local = torch.where(selector, assignments[1][0], assignments[0][0])
        chosen_codes = torch.where(selector[:, None], assignments[1][1], assignments[0][1])
        chosen_recon = torch.where(selector[:, None], assignments[1][2], assignments[0][2])
        global_scale = update_global_scale_dual(
            groups,
            scales,
            cb0,
            cb1,
            chosen_local,
            chosen_codes,
            selector,
            global_scale,
            global_scale_dtype,
        )

    assignments = []
    for codebook in [cb0, cb1]:
        assignments.append(
            assign_single_codebook(
                groups,
                global_scale,
                scales,
                codebook,
                scale_window,
                assignment_chunk=assignment_chunk,
            )
        )
    selector = assignments[1][3] < assignments[0][3]
    chosen_recon = torch.where(selector[:, None], assignments[1][2], assignments[0][2])
    metrics = metrics_from_recon(groups, chosen_recon)
    return {
        "metrics": metrics,
        "global_scale": global_scale,
        "selector_true_blocks": int(selector.sum()),
        "selector_false_blocks": int((~selector).sum()),
        "codebook0": [float(v) for v in cb0.tolist()],
        "codebook1": [float(v) for v in cb1.tolist()],
        "_recon": chosen_recon,
    }


def apply_outlier_correction(
    groups: torch.Tensor,
    recon: torch.Tensor,
    blocks_per_superblock: int,
    correction_bits: int,
) -> dict[str, Any]:
    corrected = recon.clone()
    residual = (groups - recon).to(torch.float32)
    flat_res = residual.reshape(-1)
    groups_per_super = blocks_per_superblock * groups.shape[1]
    chosen_positions: list[int] = []
    chosen_values: list[float] = []
    for start in range(0, flat_res.numel(), groups_per_super):
        end = min(flat_res.numel(), start + groups_per_super)
        chunk = flat_res[start:end]
        if chunk.numel() == 0:
            continue
        local_pos = int(chunk.abs().argmax())
        chosen_positions.append(start + local_pos)
        chosen_values.append(float(chunk[local_pos]))
    if chosen_values:
        values = torch.tensor(chosen_values, dtype=torch.float32)
        levels = (1 << (correction_bits - 1)) - 1
        correction_scale = float(values.abs().max().clamp_min(1e-12) / levels)
        q = torch.round(values / correction_scale).clamp(-levels, levels)
        deq = q * correction_scale
        flat_corrected = corrected.reshape(-1)
        for pos, value in zip(chosen_positions, deq.tolist()):
            flat_corrected[pos] += value
    else:
        correction_scale = 1.0
    metrics = metrics_from_recon(groups, corrected)
    return {
        "metrics": metrics,
        "outlier_count": len(chosen_positions),
        "blocks_per_superblock": blocks_per_superblock,
        "correction_bits": correction_bits,
        "correction_scale": correction_scale,
        "_recon": corrected,
    }


def add_bpp(item: dict[str, Any], *, index_bits: float = 4.0, local_bits: float, group_size: int, extra_bits_per_group: float = 0.0) -> dict[str, Any]:
    effective_bpp = index_bits + local_bits / group_size + extra_bits_per_group / group_size
    item["effective_bpp"] = effective_bpp
    item["bpp_terms"] = {
        "index_bits_per_raw_value": index_bits,
        "local_scale_bits_per_block": local_bits,
        "group_size": group_size,
        "extra_bits_per_block": extra_bits_per_group,
    }
    return item


def strip_private(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: strip_private(v) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, list):
        return [strip_private(v) for v in value]
    return value


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    variants = sorted({key for row in results for key in row["variants"]})
    out: dict[str, Any] = {"variants": {}, "by_family": {}, "total_rows": len(results)}
    for variant in variants:
        rows = [row for row in results if variant in row["variants"]]
        rels = [float(row["variants"][variant]["metrics"]["relative_mse"]) for row in rows]
        weighted = [(float(row["variants"][variant]["metrics"]["relative_mse"]), float(row["n_elements"])) for row in rows]
        wins = sum(1 for row in rows if float(row["variants"][variant]["metrics"]["relative_mse"]) < float(row["ud_relative_mse"]))
        out["variants"][variant] = {
            "mean_relative_mse": mean(rels),
            "element_weighted_relative_mse": weighted_mean(weighted),
            "wins_vs_ud_rows": wins,
            "effective_bpp": rows[0]["variants"][variant].get("effective_bpp"),
            "ratio_vs_ud_mean": mean(rels) / mean([float(row["ud_relative_mse"]) for row in rows]),
        }
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        buckets[row["family"]].append(row)
    for family, rows in sorted(buckets.items()):
        out["by_family"][family] = {
            "rows": len(rows),
            "ud_mean_relative_mse": mean([float(row["ud_relative_mse"]) for row in rows]),
            "variants": {
                variant: mean(
                    [float(row["variants"][variant]["metrics"]["relative_mse"]) for row in rows if variant in row["variants"]]
                )
                for variant in variants
                if any(variant in row["variants"] for row in rows)
            },
        }
    return out


def main() -> int:
    args = parse_args()
    if args.group_size != 32:
        raise SystemExit("this experiment currently expects --group-size 32 for same-bpp IQ4 comparisons")
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if args.scale_window < 0:
        raise SystemExit("--scale-window must be >= 0")
    if args.max_elements_per_tensor < args.group_size:
        raise SystemExit("--max-elements-per-tensor must be >= --group-size")
    if args.torch_threads < 1 or args.torch_interop_threads < 1:
        raise SystemExit("torch thread counts must be >= 1")
    if args.assignment_chunk < 1:
        raise SystemExit("--assignment-chunk must be >= 1")

    sampler = load_module(Path(__file__).with_name("run-aq-tensor-sample.py"), "run_aq_tensor_sample")
    optimizer = load_module(Path(__file__).with_name("run-aq-codebook-opt-experiment.py"), "run_aq_codebook_opt")
    args.model_dir = args.model_dir.expanduser().resolve()
    args.source_rows = args.source_rows.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)

    ggml_types = set(args.ggml_type) if args.ggml_type else None
    source_rows = load_source_rows(args.source_rows, ggml_types, args.max_rows)
    if not source_rows:
        raise SystemExit("no source rows selected")
    tensor_paths = tensor_path_map(sampler, args.model_dir)

    candidate_id = f"aq4_{args.scale_format}_g{args.group_size}_ts_{args.codebook_token}"
    candidate = sampler.candidate_from_id(candidate_id)
    if candidate is None:
        raise SystemExit(f"unsupported candidate id: {candidate_id}")
    floor_candidates = []
    for fmt in args.floor_scale_format:
        if fmt == args.scale_format:
            continue
        floor_candidate_id = f"aq4_{fmt}_g{args.group_size}_ts_{args.codebook_token}"
        floor_candidate = sampler.candidate_from_id(floor_candidate_id)
        if floor_candidate is None:
            raise SystemExit(f"unsupported floor candidate id: {floor_candidate_id}")
        floor_candidates.append(floor_candidate)

    opt_args = argparse.Namespace(
        max_elements_per_tensor=args.max_elements_per_tensor,
        iterations=args.iterations,
        scale_window=args.scale_window,
        global_scale_dtype=args.global_scale_dtype,
    )

    full_scales = scale_values(args.scale_format).to(torch.float32)
    scale7 = make_scale_subset(full_scales, 7)
    scale6 = make_scale_subset(full_scales, 6)
    start_time = time.perf_counter()
    results: list[dict[str, Any]] = []

    for row_index, source in enumerate(source_rows):
        row_seed = args.seed + row_index
        tensor_name = tensor_name_from_source(source)
        path = tensor_paths.get(tensor_name)
        if path is None:
            raise RuntimeError(f"tensor not found in model: {tensor_name}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(tensor_name)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(row_seed)
        groups, _ = sampler.sample_groups_with_columns(
            tensor,
            args.group_size,
            args.max_elements_per_tensor,
            generator,
        )
        groups = groups.to(torch.float32)

        floor_states = []
        floor_results = {}
        for floor_candidate in floor_candidates:
            floor_result = optimizer.run_candidate(
                sampler,
                opt_args,
                tensor_name,
                tensor,
                floor_candidate,
                row_seed,
                floor_states=floor_states,
                assignment_chunk=args.assignment_chunk,
            )
            floor_state = floor_result.pop("_optimized_state")
            floor_states.append(floor_state)
            floor_results[floor_candidate.scale_format] = strip_private(floor_result)

        baseline_result = optimizer.run_candidate(
            sampler,
            opt_args,
            tensor_name,
            tensor,
            candidate,
            row_seed,
            floor_states=floor_states,
            assignment_chunk=args.assignment_chunk,
        )
        state = baseline_result.pop("_optimized_state")
        baseline_recon = dequant_from_indices(
            groups,
            state.global_scale,
            state.scales.to(torch.float32),
            state.codebook.to(torch.float32),
            state.local_indices.to(torch.long),
            state.code_indices.to(torch.long),
        )
        baseline_metrics = metrics_from_recon(groups, baseline_recon)
        baseline = {
            "metrics": baseline_metrics,
            "global_scale": float(state.global_scale),
            "codebook": [float(v) for v in state.codebook.to(torch.float32).tolist()],
            "local_scale_count": int(state.scales.numel()),
        }
        add_bpp(baseline, local_bits=8, group_size=args.group_size)

        variants: dict[str, dict[str, Any]] = {
            "baseline_ue3m5_g32": baseline,
        }

        scale_bias = evaluate_scale_index_bias(groups, full_scales, state, blocks_per_superblock=8)
        add_bpp(scale_bias, local_bits=7, group_size=args.group_size, extra_bits_per_group=8 / 8)
        variants["sb_index_bias_l7_s8"] = scale_bias

        clipped = optimize_single_variant(
            groups,
            state.global_scale,
            full_scales,
            state.codebook.to(torch.float32),
            args.iterations,
            args.scale_window,
            args.global_scale_dtype,
            target_mode="p95",
            assignment_chunk=args.assignment_chunk,
        )
        add_bpp(clipped, local_bits=8, group_size=args.group_size)
        variants["clipped_p95_l8"] = clipped

        local7 = optimize_single_variant(
            groups,
            state.global_scale,
            scale7,
            state.codebook.to(torch.float32),
            args.iterations,
            args.scale_window,
            args.global_scale_dtype,
            assignment_chunk=args.assignment_chunk,
        )
        add_bpp(local7, local_bits=7, group_size=args.group_size)
        variants["local7_no_extra"] = local7

        sb8 = optimize_superblock_scale_variant(
            groups,
            state.global_scale,
            scale7,
            state.codebook.to(torch.float32),
            superblock_bits=8,
            blocks_per_superblock=8,
            iterations=args.iterations,
            scale_window=args.scale_window,
            global_scale_dtype=args.global_scale_dtype,
            assignment_chunk=args.assignment_chunk,
        )
        add_bpp(sb8, local_bits=7, group_size=args.group_size, extra_bits_per_group=8 / 8)
        variants["sb_scale8_l7_s8"] = sb8

        sb4 = optimize_superblock_scale_variant(
            groups,
            state.global_scale,
            scale7,
            state.codebook.to(torch.float32),
            superblock_bits=4,
            blocks_per_superblock=4,
            iterations=args.iterations,
            scale_window=args.scale_window,
            global_scale_dtype=args.global_scale_dtype,
            assignment_chunk=args.assignment_chunk,
        )
        add_bpp(sb4, local_bits=7, group_size=args.group_size, extra_bits_per_group=4 / 4)
        variants["sb_scale4_l7_s4"] = sb4

        dual = optimize_dual_codebook_variant(
            sampler,
            groups,
            state.global_scale,
            scale7,
            candidate.codebook_mode,
            args.iterations,
            args.scale_window,
            args.global_scale_dtype,
            args.assignment_chunk,
        )
        add_bpp(dual, local_bits=7, group_size=args.group_size, extra_bits_per_group=1)
        variants["dual_codebook_l7_sel1"] = dual

        outlier512_base = local7
        out512 = apply_outlier_correction(
            groups,
            outlier512_base["_recon"],
            blocks_per_superblock=16,
            correction_bits=7,
        )
        add_bpp(out512, local_bits=7, group_size=args.group_size, extra_bits_per_group=16 / 16)
        variants["outlier1_s16_l7_i7"] = out512

        local6 = optimize_single_variant(
            groups,
            state.global_scale,
            scale6,
            state.codebook.to(torch.float32),
            args.iterations,
            args.scale_window,
            args.global_scale_dtype,
            assignment_chunk=args.assignment_chunk,
        )
        add_bpp(local6, local_bits=6, group_size=args.group_size)
        variants["local6_no_extra"] = local6

        out256 = apply_outlier_correction(
            groups,
            local6["_recon"],
            blocks_per_superblock=8,
            correction_bits=8,
        )
        add_bpp(out256, local_bits=6, group_size=args.group_size, extra_bits_per_group=16 / 8)
        variants["outlier1_s8_l6_i8"] = out256

        q = source.get("quantization", {})
        metrics = source.get("metrics", {})
        results.append(
            {
                "tensor_name": tensor_name,
                "family": sampler.family_for_tensor(tensor_name),
                "n_elements": int(q.get("n_elements", tensor.numel())),
                "ud_ggml_type": q.get("ggml_type"),
                "ud_bpp": q.get("effective_bpp"),
                "ud_relative_mse": metrics.get("relative_mse"),
                "sampled_blocks": int(groups.shape[0]),
                "sampled_elements": int(groups.numel()),
                "floor_results": floor_results,
                "variants": strip_private(variants),
            }
        )
        del tensor, groups

    output = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "description": "Same-bpp superblock/outlier compensation checks for IQ4 replacement.",
        "settings": {
            "model_dir": str(args.model_dir),
            "source_rows": str(args.source_rows),
            "ggml_type_filter": sorted(ggml_types) if ggml_types else None,
            "candidate_id": candidate_id,
            "group_size": args.group_size,
            "scale_format": args.scale_format,
            "floor_scale_formats": args.floor_scale_format,
            "iterations": args.iterations,
            "scale_window": args.scale_window,
            "global_scale_dtype": args.global_scale_dtype,
            "max_elements_per_tensor": args.max_elements_per_tensor,
            "seed": args.seed,
            "assignment_chunk": args.assignment_chunk,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
        },
        "summary": summarize(results),
        "elapsed_sec": time.perf_counter() - start_time,
        "results": results,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
