#!/usr/bin/env python3
"""Compare quantized checkpoint weights against a floating-point reference.

The first supported external format is ModelOpt-style NVFP4 exported as
safetensors: packed E2M1 values in uint8, one E4M3 scale per 16 weights, and
one float32 tensor scale. The tool samples aligned 16-value groups instead of
materializing full dequantized tensors.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open


SCHEMA_VERSION = "quantized-weight-error-v0.1"
NVFP4_GROUP_SIZE = 16
NVFP4_BYTES_PER_GROUP = NVFP4_GROUP_SIZE // 2
E2M1_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def iter_safetensor_files(model_dir: Path) -> Iterable[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        seen: set[Path] = set()
        for filename in index.get("weight_map", {}).values():
            path = model_dir / filename
            if path not in seen:
                seen.add(path)
                yield path
        return
    yield from sorted(model_dir.glob("*.safetensors"))


def build_tensor_file_map(model_dir: Path) -> dict[str, Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return {name: model_dir / filename for name, filename in index.get("weight_map", {}).items()}

    mapping: dict[str, Path] = {}
    for path in iter_safetensor_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                mapping[key] = path
    return mapping


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


def discover_modelopt_nvfp4_tensors(
    quant_path: Path,
    tensor_pattern: re.Pattern[str],
    allowed_families: set[str] | None,
) -> list[str]:
    tensors: list[str] = []
    with safe_open(quant_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in handle.keys():
            if not tensor_pattern.search(key):
                continue
            if allowed_families is not None and family_for_tensor(key) not in allowed_families:
                continue
            scale_key = key.removesuffix(".weight") + ".weight_scale"
            scale2_key = key.removesuffix(".weight") + ".weight_scale_2"
            if scale_key not in keys or scale2_key not in keys:
                continue
            tensor_slice = handle.get_slice(key)
            if tensor_slice.get_dtype() != "U8":
                continue
            tensors.append(key)
    return tensors


def limit_tensors_by_family(
    tensors: list[str],
    max_tensors: int,
    max_tensors_per_family: int | None,
) -> list[str]:
    if max_tensors_per_family is None:
        return tensors[:max_tensors]
    counts: dict[str, int] = {}
    selected: list[str] = []
    for name in tensors:
        family = family_for_tensor(name)
        count = counts.get(family, 0)
        if count >= max_tensors_per_family:
            continue
        selected.append(name)
        counts[family] = count + 1
        if len(selected) >= max_tensors:
            break
    return selected


def unpack_e2m1_low_high(packed_bytes: torch.Tensor) -> torch.Tensor:
    low = packed_bytes & 0x0F
    high = (packed_bytes >> 4) & 0x0F
    nibbles = torch.stack([low, high], dim=2).reshape(-1, NVFP4_GROUP_SIZE)
    magnitudes = E2M1_MAGNITUDES.index_select(0, (nibbles & 0x07).flatten().long()).view_as(nibbles)
    signs = (nibbles & 0x08) != 0
    return torch.where(signs, -magnitudes, magnitudes)


def sample_group_ids(
    rows: int,
    cols: int,
    max_groups: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups_per_row = cols // NVFP4_GROUP_SIZE
    total_groups = rows * groups_per_row
    sampled = min(max_groups, total_groups)
    if sampled == total_groups:
        ids = torch.arange(total_groups, dtype=torch.long)
    else:
        ids = torch.randint(total_groups, (sampled,), generator=generator, dtype=torch.long)
    return ids // groups_per_row, ids % groups_per_row


def metric_row(
    args: argparse.Namespace,
    tensor_name: str,
    base_shape: tuple[int, ...],
    sampled_groups: int,
    metrics: dict[str, float | int | None],
) -> dict:
    rows, cols = base_shape
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "timestamp_utc": utc_now(),
        "status": "ok",
        "reference_model": {
            "name": args.base_model_name,
            "source": "local",
            "path": str(args.base_model_dir),
            "dtype": args.reference_dtype,
        },
        "quantized_model": {
            "name": args.quant_model_name,
            "source": "local",
            "path": str(args.quant_model_path),
            "format": "modelopt_nvfp4",
        },
        "scope": {
            "type": "tensor_group_sample",
            "tensor_names": [tensor_name],
            "families": [family_for_tensor(tensor_name)],
            "tensor_shape": list(base_shape),
            "group_size": NVFP4_GROUP_SIZE,
            "sampled_groups": sampled_groups,
            "sampled_elements": sampled_groups * NVFP4_GROUP_SIZE,
            "seed": args.seed,
        },
        "quantization": {
            "value_format": "e2m1",
            "index_bits": 4,
            "packing": "uint8_low_nibble_first",
            "scale": {
                "format": "e4m3",
                "bits": 8,
                "group_size": NVFP4_GROUP_SIZE,
                "granularity": "per_group",
            },
            "tensor_scale": {"format": "float32", "granularity": "per_tensor"},
            "effective_bpp": 4.0 + (8.0 / NVFP4_GROUP_SIZE) + (32.0 / float(rows * cols)),
        },
        "inputs": {
            "tensor_pattern": args.tensor_pattern,
            "family_filter": args.family,
            "max_tensors_per_family": args.max_tensors_per_family,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
            "activation_stats": str(args.activation_stats) if args.activation_stats else None,
        },
        "metrics": metrics,
        "artifacts": {},
        "notes": args.note,
    }


def failure_row(args: argparse.Namespace, tensor_name: str, error_type: str, message: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "timestamp_utc": utc_now(),
        "status": "failed",
        "reference_model": {"name": args.base_model_name, "path": str(args.base_model_dir)},
        "quantized_model": {
            "name": args.quant_model_name,
            "path": str(args.quant_model_path),
            "format": "modelopt_nvfp4",
        },
        "scope": {"type": "tensor_group_sample", "tensor_names": [tensor_name], "families": [family_for_tensor(tensor_name)]},
        "inputs": {
            "tensor_pattern": args.tensor_pattern,
            "family_filter": args.family,
            "max_tensors_per_family": args.max_tensors_per_family,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
            "activation_stats": str(args.activation_stats) if args.activation_stats else None,
        },
        "metrics": {},
        "artifacts": {},
        "notes": args.note,
        "error": {"type": error_type, "message": message},
    }


def compare_tensor(
    base_path: Path,
    quant_path: Path,
    tensor_name: str,
    max_groups: int,
    generator: torch.Generator,
    activation_stats: dict[str, torch.Tensor],
) -> tuple[tuple[int, ...], int, dict[str, float | int | None]]:
    scale_key = tensor_name.removesuffix(".weight") + ".weight_scale"
    scale2_key = tensor_name.removesuffix(".weight") + ".weight_scale_2"

    with safe_open(base_path, framework="pt", device="cpu") as base_handle:
        base = base_handle.get_tensor(tensor_name).to(torch.float32)

    with safe_open(quant_path, framework="pt", device="cpu") as quant_handle:
        packed = quant_handle.get_tensor(tensor_name)
        scales = quant_handle.get_tensor(scale_key).to(torch.float32)
        tensor_scale = quant_handle.get_tensor(scale2_key).to(torch.float32)

    if base.ndim != 2:
        raise ValueError(f"reference tensor must be 2D, got {tuple(base.shape)}")
    rows, cols = (int(base.shape[0]), int(base.shape[1]))
    if cols % NVFP4_GROUP_SIZE != 0:
        raise ValueError(f"reference column count must be divisible by {NVFP4_GROUP_SIZE}, got {cols}")
    expected_packed = (rows, cols // 2)
    expected_scales = (rows, cols // NVFP4_GROUP_SIZE)
    if tuple(packed.shape) != expected_packed:
        raise ValueError(f"packed tensor shape {tuple(packed.shape)} does not match expected {expected_packed}")
    if tuple(scales.shape) != expected_scales:
        raise ValueError(f"scale tensor shape {tuple(scales.shape)} does not match expected {expected_scales}")
    if tensor_scale.numel() != 1:
        raise ValueError(f"tensor scale must be scalar, got shape {tuple(tensor_scale.shape)}")

    row_ids, group_ids = sample_group_ids(rows, cols, max_groups, generator)
    sampled_groups = int(row_ids.numel())

    base_cols = group_ids[:, None] * NVFP4_GROUP_SIZE + torch.arange(NVFP4_GROUP_SIZE)
    byte_cols = group_ids[:, None] * NVFP4_BYTES_PER_GROUP + torch.arange(NVFP4_BYTES_PER_GROUP)

    base_groups = base[row_ids[:, None], base_cols]
    packed_groups = packed[row_ids[:, None], byte_cols]
    decoded = unpack_e2m1_low_high(packed_groups)
    group_scales = scales[row_ids, group_ids].to(torch.float32)
    recon = decoded * group_scales[:, None] * tensor_scale.reshape(()).to(torch.float32)

    finite_recon = torch.isfinite(recon)
    finite_scales = torch.isfinite(group_scales)
    diff = base_groups - recon
    mse = float(diff.square().mean())
    denom = float(base_groups.square().mean().clamp_min(1e-30))
    weighted_mse = None
    weighted_relative_mse = None
    activation_second_moment = activation_stats_for_tensor(tensor_name, (rows, cols), activation_stats)
    if activation_second_moment is not None:
        group_weights = activation_second_moment.index_select(0, base_cols.flatten()).view_as(base_groups)
        group_weights = group_weights.to(torch.float32).clamp_min(0)
        group_weights = group_weights / group_weights.mean().clamp_min(1e-30)
        weighted_sse = (diff.square() * group_weights).sum()
        weighted_denom = (base_groups.square() * group_weights).sum().clamp_min(1e-30)
        weighted_mse = float(weighted_sse / group_weights.sum().clamp_min(1e-30))
        weighted_relative_mse = float(weighted_sse / weighted_denom)
    dot = float((base_groups * recon).sum())
    norm = float(base_groups.square().sum().sqrt() * recon.square().sum().sqrt())
    abs_error = diff.abs().flatten()
    zero_mask = base_groups == 0
    zero_preservation = None
    if bool(zero_mask.any()):
        zero_preservation = float((recon[zero_mask] == 0).to(torch.float32).mean())

    metrics: dict[str, float | int | None] = {
        "mse": mse,
        "relative_mse": mse / denom,
        "weighted_mse": weighted_mse,
        "weighted_relative_mse": weighted_relative_mse,
        "mean_abs_error": float(abs_error.mean()),
        "p95_abs_error": float(torch.quantile(abs_error, 0.95)),
        "max_abs_error": float(abs_error.max()),
        "cosine_similarity": dot / norm if norm > 0 else 1.0,
        "zero_preservation_rate": zero_preservation,
        "scale_nonfinite_rate": float((~finite_scales).to(torch.float32).mean()),
        "recon_nonfinite_rate": float((~finite_recon).to(torch.float32).mean()),
        "tensor_scale_value": float(tensor_scale.reshape(())),
    }
    return (rows, cols), sampled_groups, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_threads = max(1, min(os.cpu_count() or 1, 64))
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--base-model-name", default=None)
    parser.add_argument("--reference-dtype", default="bf16")
    parser.add_argument("--quant-model-path", type=Path, required=True)
    parser.add_argument("--quant-model-name", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="quant-error")
    parser.add_argument("--tensor-pattern", default=r"\.weight$")
    parser.add_argument("--family", action="append", help="Family to include; can be repeated.")
    parser.add_argument("--max-tensors", type=int, default=8)
    parser.add_argument("--max-tensors-per-family", type=int, default=None)
    parser.add_argument("--max-groups-per-tensor", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=default_threads)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument(
        "--activation-stats",
        type=Path,
        default=None,
        help="Optional activation second-moment stats as a safetensors file or directory.",
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
    if args.max_tensors < 1:
        raise SystemExit("--max-tensors must be >= 1")
    if args.max_tensors_per_family is not None and args.max_tensors_per_family < 1:
        raise SystemExit("--max-tensors-per-family must be >= 1")
    if args.max_groups_per_tensor < 1:
        raise SystemExit("--max-groups-per-tensor must be >= 1")

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)

    args.base_model_dir = args.base_model_dir.expanduser().resolve()
    args.quant_model_path = args.quant_model_path.expanduser().resolve()
    args.activation_stats = args.activation_stats.expanduser().resolve() if args.activation_stats else None
    args.base_model_name = args.base_model_name or args.base_model_dir.name
    args.quant_model_name = args.quant_model_name or args.quant_model_path.parent.name
    activation_stats = load_activation_stats(args.activation_stats)

    tensor_pattern = re.compile(args.tensor_pattern)
    allowed_families = set(args.family) if args.family else None
    base_map = build_tensor_file_map(args.base_model_dir)
    tensors = discover_modelopt_nvfp4_tensors(args.quant_model_path, tensor_pattern, allowed_families)
    tensors = limit_tensors_by_family(tensors, args.max_tensors, args.max_tensors_per_family)
    if not tensors:
        raise SystemExit("no quantized tensors matched")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    with args.output.open("a", encoding="utf-8") as output:
        for tensor_name in tensors:
            try:
                base_path = base_map[tensor_name]
                shape, sampled_groups, metrics = compare_tensor(
                    base_path,
                    args.quant_model_path,
                    tensor_name,
                    args.max_groups_per_tensor,
                    generator,
                    activation_stats,
                )
                row = metric_row(args, tensor_name, shape, sampled_groups, metrics)
            except Exception as exc:  # noqa: BLE001 - failed rows are useful in benchmark logs.
                row = failure_row(args, tensor_name, type(exc).__name__, str(exc))
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
