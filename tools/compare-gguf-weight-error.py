#!/usr/bin/env python3
"""Compare GGUF checkpoint tensors against a safetensors reference model.

This is an analysis tool for external GGUF baselines such as Unsloth Dynamic.
It imports llama.cpp's gguf-py package from a reference-source checkout and
dequantizes one tensor at a time to avoid retaining multiple large arrays.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open


SCHEMA_VERSION = "quantized-weight-error-v0.1"


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
    return "other"


def gguf_to_hf_name(gguf_name: str) -> str | None:
    if gguf_name == "output.weight":
        return "lm_head.weight"
    if gguf_name == "token_embd.weight":
        return "model.language_model.embed_tokens.weight"

    match = re.fullmatch(r"blk\.(\d+)\.(.+)", gguf_name)
    if not match:
        return None
    layer = match.group(1)
    suffix = match.group(2)
    suffix_map = {
        "ffn_down.weight": "mlp.down_proj.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight": "linear_attn.in_proj_z.weight",
        "ssm_out.weight": "linear_attn.out_proj.weight",
        "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
        "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    }
    mapped = suffix_map.get(suffix)
    if mapped is None:
        return None
    return f"model.language_model.layers.{layer}.{mapped}"


def is_float_qtype(qtype_name: str) -> bool:
    return qtype_name in {"F32", "F16", "BF16", "F64"}


def discover_tensors(reader, args: argparse.Namespace, tensor_pattern: re.Pattern[str]) -> list:
    allowed_families = set(args.family) if args.family else None
    selected = []
    for tensor in reader.tensors:
        hf_name = gguf_to_hf_name(tensor.name)
        if hf_name is None:
            continue
        family = family_for_tensor(hf_name)
        if family in {"embed", "lm_head"} and not args.include_embedding:
            continue
        if allowed_families is not None and family not in allowed_families:
            continue
        if not tensor_pattern.search(hf_name) and not tensor_pattern.search(tensor.name):
            continue
        qtype_name = tensor.tensor_type.name
        if is_float_qtype(qtype_name) and not args.include_float_tensors:
            continue
        selected.append(tensor)
    return selected[: args.max_tensors]


def dequantize_gguf_tensor(tensor, dequantize_fn) -> torch.Tensor:
    if tensor.tensor_type.name == "F32":
        return torch.from_numpy(tensor.data.view("float32").copy()).to(torch.float32)
    if tensor.tensor_type.name == "F16":
        return torch.from_numpy(tensor.data.view("float16").astype("float32", copy=False).copy())
    arr = dequantize_fn(tensor.data, tensor.tensor_type)
    return torch.from_numpy(arr.copy()).to(torch.float32)


def sample_values(
    base: torch.Tensor,
    quantized: torch.Tensor,
    max_elements: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_base = base.flatten()
    flat_quant = quantized.flatten()
    if flat_base.numel() != flat_quant.numel():
        raise ValueError(f"element count mismatch: reference={flat_base.numel()} gguf={flat_quant.numel()}")
    sampled = min(max_elements, flat_base.numel())
    if sampled == flat_base.numel():
        ids = torch.arange(sampled, dtype=torch.long)
    else:
        ids = torch.randint(flat_base.numel(), (sampled,), generator=generator, dtype=torch.long)
    return flat_base.index_select(0, ids), flat_quant.index_select(0, ids)


def compute_metrics(base_sample: torch.Tensor, quant_sample: torch.Tensor, dequant_seconds: float) -> dict[str, float | None]:
    diff = base_sample - quant_sample
    mse = float(diff.square().mean())
    denom = float(base_sample.square().mean().clamp_min(1e-30))
    dot = float((base_sample * quant_sample).sum())
    norm = float(base_sample.square().sum().sqrt() * quant_sample.square().sum().sqrt())
    abs_error = diff.abs()
    zero_mask = base_sample == 0
    zero_preservation = None
    if bool(zero_mask.any()):
        zero_preservation = float((quant_sample[zero_mask] == 0).to(torch.float32).mean())
    return {
        "mse": mse,
        "relative_mse": mse / denom,
        "mean_abs_error": float(abs_error.mean()),
        "p95_abs_error": float(torch.quantile(abs_error, 0.95)),
        "max_abs_error": float(abs_error.max()),
        "cosine_similarity": dot / norm if norm > 0 else 1.0,
        "zero_preservation_rate": zero_preservation,
        "dequant_seconds": dequant_seconds,
    }


def metric_row(
    args: argparse.Namespace,
    tensor,
    hf_name: str,
    shape: tuple[int, ...],
    sampled_elements: int,
    metrics: dict[str, float | None],
) -> dict:
    n_elements = int(tensor.n_elements)
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
            "path": str(args.gguf_path),
            "format": "gguf",
        },
        "scope": {
            "type": "tensor_sample",
            "tensor_names": [hf_name],
            "gguf_tensor_names": [tensor.name],
            "families": [family_for_tensor(hf_name)],
            "tensor_shape": list(shape),
            "sampled_elements": sampled_elements,
            "seed": args.seed,
        },
        "quantization": {
            "ggml_type": tensor.tensor_type.name,
            "ggml_type_id": int(tensor.tensor_type),
            "n_bytes": int(tensor.n_bytes),
            "n_elements": n_elements,
            "effective_bpp": (8.0 * float(tensor.n_bytes) / float(n_elements)) if n_elements else None,
        },
        "inputs": {
            "tensor_pattern": args.tensor_pattern,
            "family_filter": args.family,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
            "gguf_python_path": str(args.gguf_python_path),
        },
        "metrics": metrics,
        "artifacts": {},
        "notes": args.note,
    }


def failure_row(args: argparse.Namespace, tensor_name: str, hf_name: str | None, error_type: str, message: str) -> dict:
    family = family_for_tensor(hf_name) if hf_name is not None else "unknown"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "timestamp_utc": utc_now(),
        "status": "failed",
        "reference_model": {"name": args.base_model_name, "path": str(args.base_model_dir)},
        "quantized_model": {"name": args.quant_model_name, "path": str(args.gguf_path), "format": "gguf"},
        "scope": {"type": "tensor_sample", "tensor_names": [hf_name], "gguf_tensor_names": [tensor_name], "families": [family]},
        "inputs": {
            "tensor_pattern": args.tensor_pattern,
            "family_filter": args.family,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
            "gguf_python_path": str(args.gguf_python_path),
        },
        "metrics": {},
        "artifacts": {},
        "notes": args.note,
        "error": {"type": error_type, "message": message},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_threads = max(1, min(os.cpu_count() or 1, 64))
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--base-model-name", default=None)
    parser.add_argument("--reference-dtype", default="bf16")
    parser.add_argument("--gguf-path", type=Path, required=True)
    parser.add_argument("--quant-model-name", default=None)
    parser.add_argument("--gguf-python-path", type=Path, default=Path("reference-src/llama.cpp/gguf-py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="gguf-error")
    parser.add_argument("--tensor-pattern", default=r"\.weight$")
    parser.add_argument("--family", action="append", help="Family to include; can be repeated.")
    parser.add_argument("--include-float-tensors", action="store_true")
    parser.add_argument("--include-embedding", action="store_true")
    parser.add_argument("--max-tensors", type=int, default=8)
    parser.add_argument("--max-elements-per-tensor", type=int, default=262144)
    parser.add_argument("--max-dequant-elements", type=int, default=200_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=default_threads)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be >= 1")
    if args.torch_interop_threads < 1:
        raise SystemExit("--torch-interop-threads must be >= 1")
    if args.max_tensors < 1:
        raise SystemExit("--max-tensors must be >= 1")
    if args.max_elements_per_tensor < 1:
        raise SystemExit("--max-elements-per-tensor must be >= 1")

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)

    args.base_model_dir = args.base_model_dir.expanduser().resolve()
    args.gguf_path = args.gguf_path.expanduser().resolve()
    args.gguf_python_path = args.gguf_python_path.expanduser().resolve()
    args.base_model_name = args.base_model_name or args.base_model_dir.name
    args.quant_model_name = args.quant_model_name or args.gguf_path.stem

    sys.path.insert(0, str(args.gguf_python_path))
    from gguf import GGUFReader  # noqa: PLC0415
    from gguf.quants import dequantize  # noqa: PLC0415

    reader = GGUFReader(args.gguf_path, "r")
    base_map = build_tensor_file_map(args.base_model_dir)
    tensor_pattern = re.compile(args.tensor_pattern)
    tensors = discover_tensors(reader, args, tensor_pattern)
    if not tensors:
        raise SystemExit("no GGUF tensors matched")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    with args.output.open("a", encoding="utf-8") as output:
        for tensor in tensors:
            hf_name = gguf_to_hf_name(tensor.name)
            try:
                if int(tensor.n_elements) > args.max_dequant_elements:
                    raise ValueError(
                        f"tensor has {int(tensor.n_elements)} elements, above --max-dequant-elements={args.max_dequant_elements}"
                    )
                if hf_name is None:
                    raise ValueError("no HF tensor mapping")
                base_path = base_map[hf_name]
                with safe_open(base_path, framework="pt", device="cpu") as base_handle:
                    base = base_handle.get_tensor(hf_name).to(torch.float32)
                start = time.perf_counter()
                quantized = dequantize_gguf_tensor(tensor, dequantize)
                dequant_seconds = time.perf_counter() - start
                if tuple(base.shape) != tuple(quantized.shape):
                    raise ValueError(f"shape mismatch: reference={tuple(base.shape)} gguf={tuple(quantized.shape)}")
                base_sample, quant_sample = sample_values(base, quantized, args.max_elements_per_tensor, generator)
                metrics = compute_metrics(base_sample, quant_sample, dequant_seconds)
                row = metric_row(args, tensor, hf_name, tuple(int(dim) for dim in base.shape), int(base_sample.numel()), metrics)
            except Exception as exc:  # noqa: BLE001 - failed rows are useful in benchmark logs.
                row = failure_row(args, tensor.name, hf_name, type(exc).__name__, str(exc))
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
