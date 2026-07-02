#!/usr/bin/env python3
"""Run first-pass aq tensor sampling experiments.

This tool is intentionally small and conservative: it loads one tensor at a
time, samples groups, evaluates candidate 4-bit codebook + 8-bit scale formats,
and writes JSONL rows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aq_scale_formats import (  # noqa: E402
    scale_complexity,
    scale_format_dominates,
    scale_subset_index_map,
    scale_values as shared_scale_values,
)


SCHEMA_VERSION = "aq-experiment-result-v0.1"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    scale_format: str
    group_size: int
    tensor_scale: str
    family_scale: str
    codebook_mode: str
    codebook_granularity: str = "per_tensor_sample"


@dataclass
class EvaluationState:
    candidate_id: str
    scale_format: str
    group_size: int
    codebook_mode: str
    tensor_scale: float
    scales: torch.Tensor
    codebook: torch.Tensor
    local_indices: torch.Tensor
    code_indices: torch.Tensor
    metrics: dict[str, float | int | str | None]


ROUND1_CANDIDATES = [
    Candidate("aq4_e8m0_g32_zf15", "e8m0", 32, "none", "none", "zero_free15"),
    Candidate("aq4_e8m0_g16_zf15", "e8m0", 16, "none", "none", "zero_free15"),
    Candidate("aq4_e4m3_g16_ts_zf15", "e4m3", 16, "bf16", "none", "zero_free15"),
    Candidate("aq4_e4m3_g32_ts_zf15", "e4m3", 32, "bf16", "none", "zero_free15"),
    Candidate("aq4_e5m2_g32_ts_zf15", "e5m2", 32, "bf16", "none", "zero_free15"),
    Candidate("aq4_ue5m3_g32_ts_zf15", "ue5m3", 32, "bf16", "none", "zero_free15"),
    Candidate("aq4_e8m0_g64_sym7", "e8m0", 64, "none", "none", "symmetric7"),
    Candidate("aq4_e4m3_g8_ts_flloyd16", "e4m3", 8, "bf16", "none", "free_lloyd16"),
    Candidate("aq4_e4m3_g16_ts_free16", "e4m3", 16, "bf16", "none", "free16"),
    Candidate("aq4_e4m3_g16_ts_zlloyd15", "e4m3", 16, "bf16", "none", "zero_lloyd15"),
    Candidate("aq4_e4m3_g16_ts_flloyd16", "e4m3", 16, "bf16", "none", "free_lloyd16"),
    Candidate("aq4_e4m3_g32_ts_zlloyd15", "e4m3", 32, "bf16", "none", "zero_lloyd15"),
    Candidate("aq4_e4m3_g32_ts_flloyd16", "e4m3", 32, "bf16", "none", "free_lloyd16"),
    Candidate("aq4_e4m3_g64_ts_flloyd16", "e4m3", 64, "bf16", "none", "free_lloyd16"),
    Candidate("aq4_e5m2_g16_ts_zf15", "e5m2", 16, "bf16", "none", "zero_free15"),
    Candidate("aq4_e5m2_g16_ts_zlloyd15", "e5m2", 16, "bf16", "none", "zero_lloyd15"),
    Candidate("aq4_ue5m3_g16_ts_zf15", "ue5m3", 16, "bf16", "none", "zero_free15"),
    Candidate("aq4_ue5m3_g16_ts_zlloyd15", "ue5m3", 16, "bf16", "none", "zero_lloyd15"),
    Candidate("aq4_e8m0_g16_zlloyd15", "e8m0", 16, "none", "none", "zero_lloyd15"),
]


def candidate_from_id(candidate_id: str) -> Candidate | None:
    match = re.fullmatch(
        r"aq4_(?P<scale>e8m0|u?e\d+m\d+)_g(?P<group>\d+)_(?:(?P<tensor_scale>ts)_)?(?P<codebook>zf15|zlloyd15|flloyd16|free16|sym7)",
        candidate_id,
    )
    if match is None:
        return None
    codebook_mode_by_token = {
        "zf15": "zero_free15",
        "zlloyd15": "zero_lloyd15",
        "flloyd16": "free_lloyd16",
        "free16": "free16",
        "sym7": "symmetric7",
    }
    return Candidate(
        candidate_id=candidate_id,
        scale_format=match.group("scale"),
        group_size=int(match.group("group")),
        tensor_scale="bf16" if match.group("tensor_scale") else "none",
        family_scale="none",
        codebook_mode=codebook_mode_by_token[match.group("codebook")],
    )


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def iter_safetensor_files(model_dir: Path) -> Iterable[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        seen = set()
        for filename in weight_map.values():
            path = model_dir / filename
            if path not in seen:
                seen.add(path)
                yield path
        return
    yield from sorted(model_dir.glob("*.safetensors"))


def discover_tensors(model_dir: Path, tensor_pattern: re.Pattern[str]) -> list[tuple[str, Path]]:
    tensors: list[tuple[str, Path]] = []
    for path in iter_safetensor_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if tensor_pattern.search(key):
                    tensors.append((key, path))
    return tensors


def limit_tensors_by_family(
    tensors: list[tuple[str, Path]],
    max_tensors: int,
    max_tensors_per_family: int | None,
) -> list[tuple[str, Path]]:
    if max_tensors_per_family is None:
        return tensors[:max_tensors]
    counts: dict[str, int] = defaultdict(int)
    selected: list[tuple[str, Path]] = []
    for name, path in tensors:
        family = family_for_tensor(name)
        if counts[family] >= max_tensors_per_family:
            continue
        selected.append((name, path))
        counts[family] += 1
        if len(selected) >= max_tensors:
            break
    return selected


def family_for_tensor(name: str) -> str:
    if "self_attn.q_proj" in name:
        return "attn_q"
    if "self_attn.k_proj" in name:
        return "attn_k"
    if "self_attn.v_proj" in name:
        return "attn_v"
    if "self_attn.o_proj" in name:
        return "attn_o"
    if "linear_attn.in_proj_qkv" in name:
        return "linear_attn_qkv"
    if "linear_attn.in_proj_a" in name:
        return "linear_attn_a"
    if "linear_attn.in_proj_b" in name:
        return "linear_attn_b"
    if "linear_attn.in_proj_z" in name:
        return "linear_attn_z"
    if "linear_attn.out_proj" in name:
        return "linear_attn_out"
    if "mlp.gate_proj" in name:
        return "mlp_gate"
    if "mlp.up_proj" in name:
        return "mlp_up"
    if "mlp.down_proj" in name:
        return "mlp_down"
    if "embed_tokens" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    if "router" in name:
        return "moe_router"
    if "experts" in name:
        return "moe_expert"
    return "other"


def decode_e8m0() -> torch.Tensor:
    codes = torch.arange(0, 255, dtype=torch.float32)
    return torch.pow(torch.tensor(2.0, dtype=torch.float32), codes - 127.0)


def decode_ieee_like_float(exp_bits: int, mant_bits: int, bias: int) -> torch.Tensor:
    values = []
    max_exp = (1 << exp_bits) - 1
    for exp in range(max_exp):
        for mant in range(1 << mant_bits):
            if exp == 0:
                if mant == 0:
                    continue
                value = (mant / float(1 << mant_bits)) * (2.0 ** (1 - bias))
            else:
                value = (1.0 + mant / float(1 << mant_bits)) * (2.0 ** (exp - bias))
            values.append(value)
    return torch.tensor(sorted(set(values)), dtype=torch.float32)


def decode_ue5m3() -> torch.Tensor:
    values = []
    bias = 15
    for exp in range(32):
        for mant in range(8):
            if exp == 0:
                if mant == 0:
                    continue
                value = (mant / 8.0) * (2.0 ** (1 - bias))
            else:
                value = (1.0 + mant / 8.0) * (2.0 ** (exp - bias))
            values.append(value)
    return torch.tensor(sorted(set(values)), dtype=torch.float32)


def scale_values(scale_format: str) -> torch.Tensor:
    return shared_scale_values(scale_format)


def sample_groups(
    tensor: torch.Tensor,
    group_size: int,
    max_elements: int,
    generator: torch.Generator,
) -> torch.Tensor:
    groups, _ = sample_groups_with_columns(tensor, group_size, max_elements, generator)
    return groups


def sample_groups_with_columns(
    tensor: torch.Tensor,
    group_size: int,
    max_elements: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    flat = tensor.detach().flatten().to(torch.float32)
    usable = (flat.numel() // group_size) * group_size
    if usable == 0:
        raise ValueError("tensor is smaller than one group")
    grouped = flat[:usable].view(-1, group_size)
    max_groups = max(1, max_elements // group_size)
    if grouped.shape[0] <= max_groups:
        ids = torch.arange(grouped.shape[0], dtype=torch.long)
        selected = grouped.contiguous()
    else:
        ids = torch.randint(grouped.shape[0], (max_groups,), generator=generator)
        selected = grouped.index_select(0, ids).contiguous()

    columns = None
    if tensor.ndim == 2:
        cols = int(tensor.shape[1])
        offsets = torch.arange(group_size, dtype=torch.long)
        columns = (ids[:, None] * group_size + offsets[None, :]) % cols
    return selected, columns


def normalized_values(groups: torch.Tensor) -> torch.Tensor:
    values, _ = normalized_values_and_weights(groups, None)
    return values


def normalized_values_and_weights(
    groups: torch.Tensor,
    group_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    amax = groups.abs().amax(dim=1)
    mask = amax > 0
    if not bool(mask.any()):
        values = torch.zeros(1, dtype=torch.float32)
        weights = torch.ones(1, dtype=torch.float32) if group_weights is not None else None
        return values, weights
    values = (groups[mask] / amax[mask, None]).flatten()
    weights = group_weights[mask].to(torch.float32).flatten() if group_weights is not None else None
    return values, weights


def codebook_from_normalized_values(
    norm: torch.Tensor,
    mode: str,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode in {"zero_free15", "zero_lloyd15"}:
        nonzero = norm[norm.abs() > 0]
        if nonzero.numel() == 0:
            values = torch.zeros(15, dtype=torch.float32)
        else:
            q = torch.linspace(0.03, 0.97, 15)
            values = torch.quantile(nonzero, q)
        codebook = torch.cat([torch.zeros(1), values]).sort().values
        if mode == "zero_lloyd15":
            codebook = lloyd_refine_codebook(norm, codebook, fixed_zero=True, weights=weights)
    elif mode == "symmetric7":
        abs_values = norm.abs()
        abs_values = abs_values[abs_values > 0]
        if abs_values.numel() == 0:
            pos = torch.zeros(7, dtype=torch.float32)
        else:
            q = torch.linspace(0.15, 0.98, 7)
            pos = torch.quantile(abs_values, q).clamp_min(0)
        codebook = torch.cat([-pos.flip(0), torch.zeros(1), pos, torch.tensor([1.0])])
        codebook = codebook[:16].sort().values
    elif mode in {"free16", "free_lloyd16"}:
        q = torch.linspace(0.02, 0.98, 16)
        codebook = torch.quantile(norm, q).sort().values
        if mode == "free_lloyd16":
            codebook = lloyd_refine_codebook(norm, codebook, fixed_zero=False, weights=weights)
    else:
        raise ValueError(f"unknown codebook mode: {mode}")
    return codebook.to(torch.float32)


def codebook_from_groups(
    groups: torch.Tensor,
    mode: str,
    group_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    values, weights = normalized_values_and_weights(groups, group_weights)
    return codebook_from_normalized_values(values, mode, weights)


def lloyd_refine_codebook(
    values: torch.Tensor,
    initial: torch.Tensor,
    fixed_zero: bool,
    weights: torch.Tensor | None = None,
    iterations: int = 8,
) -> torch.Tensor:
    codebook = initial.to(torch.float32).clone()
    weights = weights.to(torch.float32).clamp_min(0) if weights is not None else None
    zero_index = None
    if fixed_zero:
        zero_index = int(codebook.abs().argmin())
        codebook[zero_index] = 0.0
    for _ in range(iterations):
        nearest = (values[:, None] - codebook[None, :]).square().argmin(dim=1)
        updated = codebook.clone()
        for idx in range(codebook.numel()):
            if fixed_zero and idx == zero_index:
                updated[idx] = 0.0
                continue
            mask = nearest == idx
            if bool(mask.any()):
                if weights is None:
                    updated[idx] = values[mask].mean()
                else:
                    selected_weights = weights[mask]
                    denom = selected_weights.sum()
                    if float(denom) > 0:
                        updated[idx] = (values[mask] * selected_weights).sum() / denom
        codebook = updated.sort().values
        if fixed_zero:
            zero_index = int(codebook.abs().argmin())
            codebook[zero_index] = 0.0
    return codebook.sort().values


def build_family_codebooks(
    args: argparse.Namespace,
    tensors: list[tuple[str, Path]],
    candidates: list[Candidate],
    activation_stats: dict[str, torch.Tensor],
) -> dict[tuple[str, str], torch.Tensor]:
    result: dict[tuple[str, str], torch.Tensor] = {}
    for candidate_index, candidate in enumerate(candidates):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + 100_000 + candidate_index)
        family_values: dict[str, list[torch.Tensor]] = defaultdict(list)
        family_weights: dict[str, list[torch.Tensor]] = defaultdict(list)
        for tensor_name, path in tensors:
            family = family_for_tensor(tensor_name)
            with safe_open(path, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(tensor_name)
            if tensor.ndim < 2 or not tensor.is_floating_point():
                continue
            tensor_shape = tuple(int(dim) for dim in tensor.shape)
            groups, columns = sample_groups_with_columns(
                tensor,
                candidate.group_size,
                args.max_elements_per_tensor,
                generator,
            )
            group_weights = None
            if args.weighted_codebook:
                if columns is None:
                    raise ValueError(f"cannot apply activation stats to non-2D tensor {tensor_name}")
                activation_second_moment = activation_stats_for_tensor(tensor_name, tensor_shape, activation_stats)
                group_weights = activation_second_moment.index_select(0, columns.flatten()).view_as(groups)
            values, weights = normalized_values_and_weights(groups, group_weights)
            family_values[family].append(values)
            if weights is not None:
                family_weights[family].append(weights)
            del tensor, groups
        for family, chunks in family_values.items():
            values = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
            weights = None
            if args.weighted_codebook:
                weight_chunks = family_weights[family]
                weights = torch.cat(weight_chunks) if len(weight_chunks) > 1 else weight_chunks[0]
            result[(family, candidate.candidate_id)] = codebook_from_normalized_values(
                values,
                candidate.codebook_mode,
                weights,
            )
    return result


def choose_tensor_scale(groups: torch.Tensor, candidate: Candidate, scales: torch.Tensor, codebook: torch.Tensor) -> float:
    if candidate.tensor_scale == "none":
        return 1.0
    max_code = float(codebook.abs().max().clamp_min(1e-12))
    target = groups.abs().amax(dim=1) / max_code
    target = target[target > 0]
    if target.numel() == 0:
        return 1.0
    scale_median = float(scales.median())
    if not math.isfinite(scale_median) or scale_median <= 0:
        return 1.0
    tensor_scale = float(target.median()) / scale_median
    if not math.isfinite(tensor_scale) or tensor_scale <= 0:
        return 1.0
    return tensor_scale


def nearest_scale_indices(target: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    target = target.clamp(float(scales[0]), float(scales[-1]))
    idx = torch.searchsorted(scales, target)
    idx = idx.clamp(0, scales.numel() - 1)
    prev = (idx - 1).clamp(0, scales.numel() - 1)
    choose_prev = (target - scales[prev]).abs() < (target - scales[idx]).abs()
    return torch.where(choose_prev, prev, idx)


def objective_error(
    square_error: torch.Tensor,
    group_weights: torch.Tensor | None,
    weighted_scale_search: bool,
) -> torch.Tensor:
    if weighted_scale_search and group_weights is not None:
        return (square_error * group_weights.to(torch.float32).clamp_min(0)).sum(dim=1)
    return square_error.sum(dim=1)


def metrics_from_recon(
    groups: torch.Tensor,
    recon: torch.Tensor,
    best_error: torch.Tensor,
    candidate: Candidate,
    group_weights: torch.Tensor | None,
    scales: torch.Tensor,
    best_scale: torch.Tensor,
) -> dict[str, float | int | str | None]:
    diff = groups - recon
    mse = float(diff.square().mean())
    denom = float(groups.square().mean().clamp_min(1e-30))
    weighted_mse = None
    weighted_relative_mse = None
    if group_weights is not None:
        weights = group_weights.to(torch.float32).clamp_min(0)
        mean_weight = weights.mean().clamp_min(1e-30)
        weights = weights / mean_weight
        weighted_sse = (diff.square() * weights).sum()
        weighted_denom = (groups.square() * weights).sum().clamp_min(1e-30)
        weighted_mse = float(weighted_sse / weights.sum().clamp_min(1e-30))
        weighted_relative_mse = float(weighted_sse / weighted_denom)
    dot = float((groups * recon).sum())
    norm = float(groups.square().sum().sqrt() * recon.square().sum().sqrt())
    zero_mask = groups == 0
    zero_preservation = None
    if bool(zero_mask.any()):
        zero_preservation = float((recon[zero_mask] == 0).to(torch.float32).mean())
    max_scale = float(scales[-1])
    min_scale = float(scales[0])
    saturation = ((best_scale == max_scale) | (best_scale == min_scale)).to(torch.float32).mean()
    effective_bpp = 4.0 + 8.0 / candidate.group_size

    return {
        "effective_bpp": effective_bpp,
        "mse": mse,
        "relative_mse": mse / denom,
        "weighted_mse": weighted_mse,
        "weighted_relative_mse": weighted_relative_mse,
        "max_abs_error": float(diff.abs().max()),
        "cosine_similarity": dot / norm if norm > 0 else 1.0,
        "saturation_rate": float(saturation),
        "zero_preservation_rate": zero_preservation,
        "mean_group_error": float(best_error.mean()),
        "p95_group_error": float(torch.quantile(best_error, 0.95)),
        "sampled_groups": int(groups.shape[0]),
        "sampled_elements": int(groups.numel()),
    }


def evaluate_candidate(
    groups: torch.Tensor,
    candidate: Candidate,
    scale_window: int,
    codebook_override: torch.Tensor | None = None,
    group_weights: torch.Tensor | None = None,
    weighted_scale_search: bool = False,
    weighted_codebook: bool = False,
    floor_states: list[EvaluationState] | None = None,
) -> dict[str, float | int | str | None]:
    scales = scale_values(candidate.scale_format)
    if codebook_override is not None:
        codebook = codebook_override
    else:
        codebook_weights = group_weights if weighted_codebook else None
        codebook = codebook_from_groups(groups, candidate.codebook_mode, codebook_weights)
    tensor_scale = choose_tensor_scale(groups, candidate, scales, codebook)
    scaled_groups = groups / tensor_scale
    max_code = codebook.abs().max().clamp_min(1e-12)
    target_scale = scaled_groups.abs().amax(dim=1) / max_code
    center = nearest_scale_indices(target_scale, scales)

    best_error = torch.full((groups.shape[0],), torch.inf, dtype=torch.float32)
    best_scale = torch.zeros((groups.shape[0],), dtype=torch.float32)
    best_scale_idx = torch.zeros((groups.shape[0],), dtype=torch.long)
    best_code_idx = torch.zeros((groups.shape[0], groups.shape[1]), dtype=torch.long)
    best_recon = torch.zeros_like(groups)

    offsets = range(-scale_window, scale_window + 1)
    for offset in offsets:
        idx = (center + offset).clamp(0, scales.numel() - 1)
        group_scale = scales.index_select(0, idx)
        normalized = scaled_groups / group_scale[:, None]
        distances = (normalized[:, :, None] - codebook[None, None, :]).abs()
        nearest = distances.argmin(dim=2)
        quantized = codebook.index_select(0, nearest.flatten()).view_as(groups)
        recon = quantized * group_scale[:, None] * tensor_scale
        square_error = (groups - recon).square()
        error = objective_error(square_error, group_weights, weighted_scale_search)
        mask = error < best_error
        best_error = torch.where(mask, error, best_error)
        best_scale = torch.where(mask, group_scale, best_scale)
        best_scale_idx = torch.where(mask, idx, best_scale_idx)
        best_code_idx = torch.where(mask[:, None], nearest, best_code_idx)
        best_recon = torch.where(mask[:, None], recon, best_recon)

    metrics = metrics_from_recon(groups, best_recon, best_error, candidate, group_weights, scales, best_scale)
    metrics["tensor_scale_value"] = tensor_scale
    best_state = EvaluationState(
        candidate_id=candidate.candidate_id,
        scale_format=candidate.scale_format,
        group_size=candidate.group_size,
        codebook_mode=candidate.codebook_mode,
        tensor_scale=tensor_scale,
        scales=scales.detach().to(torch.float32).cpu().clone(),
        codebook=codebook.detach().to(torch.float32).cpu().clone(),
        local_indices=best_scale_idx.detach().cpu().clone(),
        code_indices=best_code_idx.detach().cpu().clone(),
        metrics={key: value for key, value in metrics.items() if not key.startswith("_")},
    )
    objective_key = "weighted_relative_mse" if weighted_scale_search and group_weights is not None else "relative_mse"
    best_objective = float(metrics[objective_key] or metrics["relative_mse"])
    lifted_floor_count = 0

    for floor_state in floor_states or []:
        if floor_state.group_size != candidate.group_size:
            continue
        if floor_state.codebook_mode != candidate.codebook_mode:
            continue
        if not scale_format_dominates(candidate.scale_format, floor_state.scale_format):
            continue
        mapping = scale_subset_index_map(floor_state.scales, scales)
        lifted_scale_idx = mapping.index_select(0, floor_state.local_indices)
        group_scale = scales.index_select(0, lifted_scale_idx)
        quantized = floor_state.codebook.index_select(0, floor_state.code_indices.flatten()).view_as(groups)
        recon = quantized * group_scale[:, None] * float(floor_state.tensor_scale)
        square_error = (groups - recon).square()
        error = objective_error(square_error, group_weights, weighted_scale_search)
        floor_metrics = metrics_from_recon(groups, recon, error, candidate, group_weights, scales, group_scale)
        floor_metrics["tensor_scale_value"] = float(floor_state.tensor_scale)
        floor_metrics["lifted_from_candidate_id"] = floor_state.candidate_id
        floor_metrics["lifted_from_scale_format"] = floor_state.scale_format
        lifted_floor_count += 1
        floor_objective = float(floor_metrics[objective_key] or floor_metrics["relative_mse"])
        if floor_objective < best_objective - 1e-15:
            best_objective = floor_objective
            metrics = floor_metrics
            best_state = EvaluationState(
                candidate_id=candidate.candidate_id,
                scale_format=candidate.scale_format,
                group_size=candidate.group_size,
                codebook_mode=candidate.codebook_mode,
                tensor_scale=float(floor_state.tensor_scale),
                scales=scales.detach().to(torch.float32).cpu().clone(),
                codebook=floor_state.codebook.detach().to(torch.float32).cpu().clone(),
                local_indices=lifted_scale_idx.detach().cpu().clone(),
                code_indices=floor_state.code_indices.detach().cpu().clone(),
                metrics={key: value for key, value in floor_metrics.items() if not key.startswith("_")},
            )

    metrics["lifted_floor_count"] = lifted_floor_count
    metrics["_evaluation_state"] = best_state
    return metrics


def row_for_result(
    args: argparse.Namespace,
    tensor_name: str,
    tensor_shape: tuple[int, ...],
    tensor_dtype: str,
    family: str,
    candidate: Candidate,
    metrics: dict[str, float | int | str | None],
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "timestamp_utc": utc_now(),
        "status": "ok",
        "model": {
            "name": args.model_name,
            "source": "local",
            "path": str(args.model_dir),
            "dtype_reference": args.reference_dtype,
            "revision": None,
        },
        "scope": {
            "type": "tensor_sample",
            "tensor_names": [tensor_name],
            "families": [family],
            "sample_elements_per_tensor": args.max_elements_per_tensor,
            "seed": args.seed,
            "tensor_shape": list(tensor_shape),
            "tensor_dtype": tensor_dtype,
        },
        "candidate": {
            "candidate_id": candidate.candidate_id,
            "index_bits": 4,
            "codebook": {
                "mode": candidate.codebook_mode,
                "storage_dtype": "bf16",
                "granularity": args.codebook_granularity,
                "entry_count": 16,
            },
            "scale": {
                "format": candidate.scale_format,
                "bits": 8,
                "group_size": candidate.group_size,
                "granularity": "per_group",
                "tensor_scale": candidate.tensor_scale,
                "family_scale": candidate.family_scale,
            },
            "group_layout": {"axis": "contiguous", "tile_shape": None},
            "optimizer": {
                "objective": "activation_weighted_mse" if args.weighted_scale_search else "mse",
                "weighted_metrics": bool(args.activation_stats),
                "weighted_scale_search": args.weighted_scale_search,
                "weighted_codebook": args.weighted_codebook,
                "scale_search": f"nearest_plus_minus_{args.scale_window}",
                "codebook_update": (
                    "weighted_lloyd"
                    if args.weighted_codebook and "lloyd" in candidate.codebook_mode
                    else "lloyd"
                    if "lloyd" in candidate.codebook_mode
                    else "quantile_init_only"
                ),
            },
        },
        "inputs": {
            "tensor_pattern": args.tensor_pattern,
            "family_filter": args.family,
            "max_tensors_per_family": args.max_tensors_per_family,
            "codebook_granularity": args.codebook_granularity,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
            "activation_stats": str(args.activation_stats) if args.activation_stats else None,
            "sampling_policy": "shared_by_block_size",
            "candidate_order": "scale_dominance_first",
            "monotonic_floor": "enabled_for_dominated_unsigned_em_scale_formats",
        },
        "metrics": metrics,
        "artifacts": {},
        "notes": args.note,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_threads = max(1, min(os.cpu_count() or 1, 64))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--reference-dtype", default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="aq-round1")
    parser.add_argument("--tensor-pattern", default=r"\.weight$")
    parser.add_argument("--family", action="append", help="Family to include; can be repeated.")
    parser.add_argument("--max-tensors", type=int, default=8)
    parser.add_argument("--max-tensors-per-family", type=int, default=None)
    parser.add_argument("--max-elements-per-tensor", type=int, default=262144)
    parser.add_argument(
        "--codebook-granularity",
        choices=("per_tensor_sample", "per_family_sample"),
        default="per_tensor_sample",
    )
    parser.add_argument("--scale-window", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=default_threads)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--candidate", action="append", help="Candidate ID to run; default is round1.")
    parser.add_argument(
        "--activation-stats",
        type=Path,
        default=None,
        help="Optional activation second-moment stats as a safetensors file or directory.",
    )
    parser.add_argument(
        "--weighted-scale-search",
        action="store_true",
        help="Use activation weights, when provided, to choose the best group scale.",
    )
    parser.add_argument(
        "--weighted-codebook",
        action="store_true",
        help="Use activation weights, when provided, during Lloyd codebook refinement.",
    )
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def load_activation_stats(path: Path | None) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "activation_second_moments.safetensors"
    if not path.exists():
        raise SystemExit(f"activation stats not found: {path}")
    if path.suffix != ".safetensors":
        raise SystemExit("--activation-stats currently expects a safetensors file or directory")

    stats: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            stats[key] = handle.get_tensor(key).to(torch.float32).flatten().contiguous()
    return stats


def activation_stats_for_tensor(
    tensor_name: str,
    tensor_shape: tuple[int, ...],
    stats: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    if not stats:
        return None
    key_stem = tensor_name.removesuffix(".weight")
    module_stem = key_stem.removeprefix("model.")
    candidates = (
        tensor_name,
        key_stem,
        module_stem,
        f"{tensor_name}.input_second_moment",
        f"{key_stem}.input_second_moment",
        f"{module_stem}.input_second_moment",
    )
    for key in candidates:
        values = stats.get(key)
        if values is None:
            continue
        if len(tensor_shape) != 2:
            raise ValueError(f"activation stats require a 2D tensor, got shape {tensor_shape}")
        in_features = int(tensor_shape[1])
        if values.numel() != in_features:
            raise ValueError(
                f"activation stats for {tensor_name} have {values.numel()} values, expected {in_features}"
            )
        return values
    raise ValueError(f"activation stats are missing for tensor {tensor_name}")


def main() -> int:
    args = parse_args()
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be >= 1")
    if args.torch_interop_threads < 1:
        raise SystemExit("--torch-interop-threads must be >= 1")
    if args.max_tensors_per_family is not None and args.max_tensors_per_family < 1:
        raise SystemExit("--max-tensors-per-family must be >= 1")
    if args.weighted_scale_search and args.activation_stats is None:
        raise SystemExit("--weighted-scale-search requires --activation-stats")
    if args.weighted_codebook and args.activation_stats is None:
        raise SystemExit("--weighted-codebook requires --activation-stats")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    args.model_dir = args.model_dir.expanduser().resolve()
    args.activation_stats = args.activation_stats.expanduser().resolve() if args.activation_stats else None
    args.model_name = args.model_name or args.model_dir.name
    activation_stats = load_activation_stats(args.activation_stats)
    tensor_pattern = re.compile(args.tensor_pattern)
    candidates = list(ROUND1_CANDIDATES)
    if args.candidate:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        candidates = []
        missing = []
        for candidate_id in args.candidate:
            candidate = by_id.get(candidate_id) or candidate_from_id(candidate_id)
            if candidate is None:
                missing.append(candidate_id)
            else:
                candidates.append(candidate)
        if missing:
            raise SystemExit(f"unknown candidate IDs: {', '.join(sorted(missing))}")
    candidates = [
        candidate
        for _, candidate in sorted(
            enumerate(candidates),
            key=lambda item: (
                item[1].group_size,
                item[1].codebook_mode,
                item[1].tensor_scale,
                item[1].family_scale,
                scale_complexity(item[1].scale_format),
                item[0],
            ),
        )
    ]

    tensors = discover_tensors(args.model_dir, tensor_pattern)
    if args.family:
        allowed = set(args.family)
        tensors = [(name, path) for name, path in tensors if family_for_tensor(name) in allowed]
    tensors = limit_tensors_by_family(tensors, args.max_tensors, args.max_tensors_per_family)
    if not tensors:
        raise SystemExit("no tensors matched")

    family_codebooks: dict[tuple[str, str], torch.Tensor] = {}
    if args.codebook_granularity == "per_family_sample":
        family_codebooks = build_family_codebooks(args, tensors, candidates, activation_stats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    with args.output.open("a", encoding="utf-8") as output:
        for tensor_name, path in tensors:
            family = family_for_tensor(tensor_name)
            with safe_open(path, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(tensor_name)
            tensor_shape = tuple(int(dim) for dim in tensor.shape)
            tensor_dtype = str(tensor.dtype).replace("torch.", "")
            if tensor.ndim < 2 or not tensor.is_floating_point():
                continue
            sample_cache: dict[int, tuple[torch.Tensor, torch.Tensor | None]] = {}
            evaluation_states: list[EvaluationState] = []
            activation_second_moment = None
            activation_stats_error: Exception | None = None
            if activation_stats:
                try:
                    activation_second_moment = activation_stats_for_tensor(tensor_name, tensor_shape, activation_stats)
                except Exception as exc:  # noqa: BLE001 - record one failure row per candidate below.
                    activation_stats_error = exc
            for candidate in candidates:
                try:
                    if activation_stats_error is not None:
                        raise activation_stats_error
                    cached = sample_cache.get(candidate.group_size)
                    if cached is None:
                        cached = sample_groups_with_columns(
                            tensor,
                            candidate.group_size,
                            args.max_elements_per_tensor,
                            generator,
                        )
                        sample_cache[candidate.group_size] = cached
                    groups, columns = cached
                    group_weights = None
                    if activation_second_moment is not None:
                        if columns is None:
                            raise ValueError(f"cannot apply activation stats to non-2D tensor {tensor_name}")
                        group_weights = activation_second_moment.index_select(0, columns.flatten()).view_as(groups)
                    codebook_override = family_codebooks.get((family, candidate.candidate_id))
                    metrics = evaluate_candidate(
                        groups,
                        candidate,
                        args.scale_window,
                        codebook_override,
                        group_weights=group_weights,
                        weighted_scale_search=args.weighted_scale_search,
                        weighted_codebook=args.weighted_codebook,
                        floor_states=evaluation_states,
                    )
                    evaluation_state = metrics.pop("_evaluation_state", None)
                    if isinstance(evaluation_state, EvaluationState):
                        evaluation_states.append(evaluation_state)
                    row = row_for_result(
                        args,
                        tensor_name,
                        tensor_shape,
                        tensor_dtype,
                        family,
                        candidate,
                        metrics,
                    )
                except Exception as exc:  # noqa: BLE001 - result rows should capture failures.
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": args.run_id,
                        "timestamp_utc": utc_now(),
                        "status": "failed",
                        "model": {
                            "name": args.model_name,
                            "source": "local",
                            "path": str(args.model_dir),
                            "dtype_reference": args.reference_dtype,
                            "revision": None,
                        },
                        "scope": {
                            "type": "tensor_sample",
                            "tensor_names": [tensor_name],
                            "families": [family],
                            "sample_elements_per_tensor": args.max_elements_per_tensor,
                            "seed": args.seed,
                            "tensor_shape": list(tensor_shape),
                            "tensor_dtype": tensor_dtype,
                        },
                        "candidate": {"candidate_id": candidate.candidate_id},
                        "inputs": {
                            "tensor_pattern": args.tensor_pattern,
                            "family_filter": args.family,
                            "max_tensors_per_family": args.max_tensors_per_family,
                            "codebook_granularity": args.codebook_granularity,
                            "torch_threads": args.torch_threads,
                            "torch_interop_threads": args.torch_interop_threads,
                            "activation_stats": str(args.activation_stats) if args.activation_stats else None,
                        },
                        "metrics": {},
                        "artifacts": {},
                        "notes": args.note,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                output.write("\n")
            del tensor
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
