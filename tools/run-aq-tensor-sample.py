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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open


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
    if scale_format == "e8m0":
        return decode_e8m0()
    if scale_format == "e5m2":
        return decode_ieee_like_float(exp_bits=5, mant_bits=2, bias=15)
    if scale_format == "e4m3":
        return decode_ieee_like_float(exp_bits=4, mant_bits=3, bias=7)
    if scale_format == "ue5m3":
        return decode_ue5m3()
    raise ValueError(f"unknown scale format: {scale_format}")


def sample_groups(
    tensor: torch.Tensor,
    group_size: int,
    max_elements: int,
    generator: torch.Generator,
) -> torch.Tensor:
    flat = tensor.detach().flatten().to(torch.float32)
    usable = (flat.numel() // group_size) * group_size
    if usable == 0:
        raise ValueError("tensor is smaller than one group")
    grouped = flat[:usable].view(-1, group_size)
    max_groups = max(1, max_elements // group_size)
    if grouped.shape[0] <= max_groups:
        return grouped.contiguous()
    ids = torch.randint(grouped.shape[0], (max_groups,), generator=generator)
    return grouped.index_select(0, ids).contiguous()


def normalized_values(groups: torch.Tensor) -> torch.Tensor:
    amax = groups.abs().amax(dim=1)
    mask = amax > 0
    if not bool(mask.any()):
        return torch.zeros(1, dtype=torch.float32)
    return (groups[mask] / amax[mask, None]).flatten()


def codebook_from_normalized_values(norm: torch.Tensor, mode: str) -> torch.Tensor:
    if mode in {"zero_free15", "zero_lloyd15"}:
        nonzero = norm[norm.abs() > 0]
        if nonzero.numel() == 0:
            values = torch.zeros(15, dtype=torch.float32)
        else:
            q = torch.linspace(0.03, 0.97, 15)
            values = torch.quantile(nonzero, q)
        codebook = torch.cat([torch.zeros(1), values]).sort().values
        if mode == "zero_lloyd15":
            codebook = lloyd_refine_codebook(norm, codebook, fixed_zero=True)
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
            codebook = lloyd_refine_codebook(norm, codebook, fixed_zero=False)
    else:
        raise ValueError(f"unknown codebook mode: {mode}")
    return codebook.to(torch.float32)


def codebook_from_groups(groups: torch.Tensor, mode: str) -> torch.Tensor:
    return codebook_from_normalized_values(normalized_values(groups), mode)


def lloyd_refine_codebook(
    values: torch.Tensor,
    initial: torch.Tensor,
    fixed_zero: bool,
    iterations: int = 8,
) -> torch.Tensor:
    codebook = initial.to(torch.float32).clone()
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
                updated[idx] = values[mask].mean()
        codebook = updated.sort().values
        if fixed_zero:
            zero_index = int(codebook.abs().argmin())
            codebook[zero_index] = 0.0
    return codebook.sort().values


def build_family_codebooks(
    args: argparse.Namespace,
    tensors: list[tuple[str, Path]],
    candidates: list[Candidate],
) -> dict[tuple[str, str], torch.Tensor]:
    result: dict[tuple[str, str], torch.Tensor] = {}
    for candidate_index, candidate in enumerate(candidates):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + 100_000 + candidate_index)
        family_values: dict[str, list[torch.Tensor]] = defaultdict(list)
        for tensor_name, path in tensors:
            family = family_for_tensor(tensor_name)
            with safe_open(path, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(tensor_name)
            if tensor.ndim < 2 or not tensor.is_floating_point():
                continue
            groups = sample_groups(
                tensor,
                candidate.group_size,
                args.max_elements_per_tensor,
                generator,
            )
            family_values[family].append(normalized_values(groups))
            del tensor, groups
        for family, chunks in family_values.items():
            values = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
            result[(family, candidate.candidate_id)] = codebook_from_normalized_values(values, candidate.codebook_mode)
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


def evaluate_candidate(
    groups: torch.Tensor,
    candidate: Candidate,
    scale_window: int,
    codebook_override: torch.Tensor | None = None,
) -> dict[str, float | int | str]:
    scales = scale_values(candidate.scale_format)
    codebook = codebook_override if codebook_override is not None else codebook_from_groups(groups, candidate.codebook_mode)
    tensor_scale = choose_tensor_scale(groups, candidate, scales, codebook)
    scaled_groups = groups / tensor_scale
    max_code = codebook.abs().max().clamp_min(1e-12)
    target_scale = scaled_groups.abs().amax(dim=1) / max_code
    center = nearest_scale_indices(target_scale, scales)

    best_error = torch.full((groups.shape[0],), torch.inf, dtype=torch.float32)
    best_scale = torch.zeros((groups.shape[0],), dtype=torch.float32)
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
        error = (groups - recon).square().sum(dim=1)
        mask = error < best_error
        best_error = torch.where(mask, error, best_error)
        best_scale = torch.where(mask, group_scale, best_scale)
        best_recon = torch.where(mask[:, None], recon, best_recon)

    diff = groups - best_recon
    mse = float(diff.square().mean())
    denom = float(groups.square().mean().clamp_min(1e-30))
    dot = float((groups * best_recon).sum())
    norm = float(groups.square().sum().sqrt() * best_recon.square().sum().sqrt())
    zero_mask = groups == 0
    zero_preservation = None
    if bool(zero_mask.any()):
        zero_preservation = float((best_recon[zero_mask] == 0).to(torch.float32).mean())
    max_scale = float(scales[-1])
    min_scale = float(scales[0])
    saturation = ((best_scale == max_scale) | (best_scale == min_scale)).to(torch.float32).mean()
    effective_bpp = 4.0 + 8.0 / candidate.group_size

    return {
        "effective_bpp": effective_bpp,
        "mse": mse,
        "relative_mse": mse / denom,
        "weighted_mse": None,
        "max_abs_error": float(diff.abs().max()),
        "cosine_similarity": dot / norm if norm > 0 else 1.0,
        "saturation_rate": float(saturation),
        "zero_preservation_rate": zero_preservation,
        "mean_group_error": float(best_error.mean()),
        "p95_group_error": float(torch.quantile(best_error, 0.95)),
        "sampled_groups": int(groups.shape[0]),
        "sampled_elements": int(groups.numel()),
        "tensor_scale_value": tensor_scale,
    }


def row_for_result(
    args: argparse.Namespace,
    tensor_name: str,
    tensor_shape: tuple[int, ...],
    tensor_dtype: str,
    family: str,
    candidate: Candidate,
    metrics: dict[str, float | int | str],
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
                "objective": "mse",
                "weighted": False,
                "scale_search": f"nearest_plus_minus_{args.scale_window}",
                "codebook_update": "quantile_init_only",
            },
        },
        "inputs": {
            "tensor_pattern": args.tensor_pattern,
            "family_filter": args.family,
            "max_tensors_per_family": args.max_tensors_per_family,
            "codebook_granularity": args.codebook_granularity,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
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
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be >= 1")
    if args.torch_interop_threads < 1:
        raise SystemExit("--torch-interop-threads must be >= 1")
    if args.max_tensors_per_family is not None and args.max_tensors_per_family < 1:
        raise SystemExit("--max-tensors-per-family must be >= 1")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    args.model_dir = args.model_dir.expanduser().resolve()
    args.model_name = args.model_name or args.model_dir.name
    tensor_pattern = re.compile(args.tensor_pattern)
    candidates = ROUND1_CANDIDATES
    if args.candidate:
        selected = set(args.candidate)
        candidates = [candidate for candidate in ROUND1_CANDIDATES if candidate.candidate_id in selected]
        missing = selected - {candidate.candidate_id for candidate in candidates}
        if missing:
            raise SystemExit(f"unknown candidate IDs: {', '.join(sorted(missing))}")

    tensors = discover_tensors(args.model_dir, tensor_pattern)
    if args.family:
        allowed = set(args.family)
        tensors = [(name, path) for name, path in tensors if family_for_tensor(name) in allowed]
    tensors = limit_tensors_by_family(tensors, args.max_tensors, args.max_tensors_per_family)
    if not tensors:
        raise SystemExit("no tensors matched")

    family_codebooks: dict[tuple[str, str], torch.Tensor] = {}
    if args.codebook_granularity == "per_family_sample":
        family_codebooks = build_family_codebooks(args, tensors, candidates)

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
            for candidate in candidates:
                try:
                    groups = sample_groups(
                        tensor,
                        candidate.group_size,
                        args.max_elements_per_tensor,
                        generator,
                    )
                    codebook_override = family_codebooks.get((family, candidate.candidate_id))
                    metrics = evaluate_candidate(groups, candidate, args.scale_window, codebook_override)
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
