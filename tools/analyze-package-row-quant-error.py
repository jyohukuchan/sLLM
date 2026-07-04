#!/usr/bin/env python3
"""Inspect package AQ4 quantization error for selected tensor rows.

This is intentionally row-scoped: a 9B package contains several large matrices,
and the golden-prefix drift diagnostics currently point at one hot hidden
channel. Loading just that output row keeps the check cheap and avoids holding
whole tensors in memory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import struct
from pathlib import Path
from typing import Any

import torch
from aq_scale_formats import scale_values
from safetensors import safe_open


SCHEMA_VERSION = "package-row-quant-error-v0.1"
DEFAULT_TENSOR_SUFFIXES = ("linear_attn.out_proj.weight", "mlp.down_proj.weight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare selected package AQ4 tensor rows against source safetensors rows."
    )
    parser.add_argument("package_dir", type=Path, help="uLLM package directory containing manifest.json.")
    parser.add_argument(
        "--hidden-index",
        type=int,
        default=3994,
        help="Output hidden row index to inspect.",
    )
    parser.add_argument(
        "--layers",
        default="0,4,5,6",
        help="Comma-separated language-model layer indices to inspect.",
    )
    parser.add_argument(
        "--tensor-suffix",
        action="append",
        default=[],
        help=(
            "Tensor suffix to inspect, relative to model.language_model.layers.N. "
            "May be repeated. Defaults to linear_attn.out_proj.weight and mlp.down_proj.weight."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of largest element errors to include per row.",
    )
    parser.add_argument("--summary-json", type=Path, help="Write summary JSON to this path.")
    parser.add_argument("--markdown", type=Path, help="Write summary Markdown to this path.")
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    layers: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"invalid descending layer range: {token}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(token))
    if not layers:
        raise ValueError("--layers did not select any layer")
    return sorted(set(layers))


def read_manifest(package_dir: Path) -> dict[str, Any]:
    manifest_path = package_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tensor_by_name(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("name")): item for item in manifest.get("tensors", [])}


def read_f32_file(path: Path) -> torch.Tensor:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} length is not divisible by 4")
    count = len(data) // 4
    values = struct.unpack(f"<{count}f", data)
    return torch.tensor(values, dtype=torch.float32)


def read_file_window(path: Path, offset: int, length: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(length)
    if len(data) != length:
        raise ValueError(f"{path} returned {len(data)} bytes at offset {offset}, expected {length}")
    return data


def decode_idx4_low_nibble_first(data: bytes, elements: int) -> torch.Tensor:
    packed = torch.tensor(list(data), dtype=torch.uint8)
    indices = torch.empty(packed.numel() * 2, dtype=torch.long)
    indices[0::2] = (packed & 0x0F).to(torch.long)
    indices[1::2] = ((packed >> 4) & 0x0F).to(torch.long)
    return indices[:elements]


def load_source_row(source_file: Path, tensor_name: str, row_index: int) -> torch.Tensor:
    with safe_open(str(source_file), framework="pt", device="cpu") as handle:
        source_slice = handle.get_slice(tensor_name)
        shape = source_slice.get_shape()
        if len(shape) != 2:
            raise ValueError(f"{tensor_name} is not 2D in source safetensors: {shape}")
        if row_index < 0 or row_index >= shape[0]:
            raise ValueError(f"row {row_index} is outside source shape {shape} for {tensor_name}")
        return source_slice[row_index].to(torch.float32).contiguous()


def dequantize_package_row(package_dir: Path, tensor: dict[str, Any], row_index: int) -> torch.Tensor:
    shape = tensor.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"{tensor.get('name')} shape is not 2D: {shape}")
    rows = int(shape[0])
    cols = int(shape[1])
    if row_index < 0 or row_index >= rows:
        raise ValueError(f"row {row_index} is outside package shape {shape} for {tensor.get('name')}")

    group_size = int(tensor["group_size"])
    if cols % group_size != 0:
        raise ValueError(f"{tensor.get('name')} row cols {cols} is not divisible by group size {group_size}")

    row_start = row_index * cols
    if row_start % 2 != 0:
        raise ValueError(f"{tensor.get('name')} row start {row_start} is not idx4 byte-aligned")
    index_offset = row_start // 2
    index_len = math.ceil(cols / 2)

    group_start = row_start // group_size
    groups_per_row = cols // group_size

    index_data = read_file_window(package_dir / str(tensor["index_file"]), index_offset, index_len)
    scale_data = read_file_window(package_dir / str(tensor["scale_file"]), group_start, groups_per_row)
    codebook = read_f32_file(package_dir / str(tensor["codebook_file"]))
    if codebook.numel() != 16:
        raise ValueError(f"{tensor.get('codebook_file')} has {codebook.numel()} entries, expected 16")

    indices = decode_idx4_low_nibble_first(index_data, cols)
    scale_indices = torch.tensor(list(scale_data), dtype=torch.long)
    scales = scale_values(str(tensor["scale_format"])).to(torch.float32)
    if torch.any(scale_indices >= scales.numel()):
        bad = int(torch.max(scale_indices).item())
        raise ValueError(f"scale index {bad} is outside scale table with {scales.numel()} values")

    combined_scales = scales[scale_indices] * float(tensor["tensor_scale"])
    expanded_scales = combined_scales.repeat_interleave(group_size)[:cols]
    return codebook[indices] * expanded_scales


def metric_row(
    package_dir: Path,
    tensor: dict[str, Any],
    row_index: int,
    top_k: int,
) -> dict[str, Any]:
    tensor_name = str(tensor["name"])
    source = load_source_row(Path(str(tensor["source_file"])), tensor_name, row_index)
    recon = dequantize_package_row(package_dir, tensor, row_index)
    if source.numel() != recon.numel():
        raise ValueError(f"{tensor_name} row length mismatch: source={source.numel()} recon={recon.numel()}")

    error = source - recon
    abs_error = torch.abs(error)
    sse = float(torch.sum(error * error).item())
    ref_sse = float(torch.sum(source * source).item())
    elements = int(source.numel())
    mse = sse / elements if elements else 0.0
    relative_mse = sse / ref_sse if ref_sse > 0.0 else 0.0
    rms = math.sqrt(mse)
    max_abs_error, max_abs_col_tensor = torch.max(abs_error, dim=0)
    max_abs_col = int(max_abs_col_tensor.item())
    manifest_metrics = {
        "mse": tensor.get("metrics", {}).get("mse"),
        "relative_mse": tensor.get("metrics", {}).get("relative_mse"),
        "max_abs_error": tensor.get("metrics", {}).get("max_abs_error"),
    }
    manifest_max_abs = manifest_metrics["max_abs_error"]
    row_max_matches_manifest = (
        isinstance(manifest_max_abs, (int, float))
        and abs(float(max_abs_error.item()) - float(manifest_max_abs)) <= 1e-6
    )

    group_size = int(tensor["group_size"])
    group_errors = error.reshape(-1, group_size)
    group_mse = torch.mean(group_errors * group_errors, dim=1)
    worst_group_mse, worst_group_tensor = torch.max(group_mse, dim=0)
    worst_group = int(worst_group_tensor.item())

    top_count = max(0, min(top_k, elements))
    top_entries: list[dict[str, Any]] = []
    if top_count:
        values, indices = torch.topk(abs_error, k=top_count)
        for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            col = int(index)
            top_entries.append(
                {
                    "rank": rank,
                    "column_index": col,
                    "group_in_row": col // group_size,
                    "abs_error": float(value),
                    "source": float(source[col].item()),
                    "recon": float(recon[col].item()),
                    "error": float(error[col].item()),
                }
            )

    return {
        "tensor_name": tensor_name,
        "source_file": str(tensor["source_file"]),
        "row_index": row_index,
        "shape": tensor["shape"],
        "group_size": group_size,
        "tensor_scale": tensor.get("tensor_scale"),
        "scale_format": tensor.get("scale_format"),
        "elements": elements,
        "mse": mse,
        "rms": rms,
        "relative_mse": relative_mse,
        "max_abs_error": float(max_abs_error.item()),
        "max_abs_column": max_abs_col,
        "max_abs_source": float(source[max_abs_col].item()),
        "max_abs_recon": float(recon[max_abs_col].item()),
        "max_abs_signed_error": float(error[max_abs_col].item()),
        "row_max_matches_manifest_max_abs": row_max_matches_manifest,
        "source_rms": math.sqrt(ref_sse / elements) if elements else 0.0,
        "recon_rms": math.sqrt(float(torch.sum(recon * recon).item()) / elements) if elements else 0.0,
        "worst_group_in_row": worst_group,
        "worst_group_global": (row_index * elements // group_size) + worst_group,
        "worst_group_mse": float(worst_group_mse.item()),
        "manifest_metrics": manifest_metrics,
        "top_abs_error_columns": top_entries,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| layer | tensor | row | row_rms | row_rel_mse | row_max_abs | manifest_max_abs | max_match | max_col | worst_group |",
        "|---:|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        parts = row["tensor_name"].split(".")
        layer = parts[3] if len(parts) > 3 else "-"
        tensor = row["tensor_name"].split(f"layers.{layer}.", 1)[-1]
        manifest = row.get("manifest_metrics", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    layer,
                    tensor,
                    str(row["row_index"]),
                    fmt_float(row["rms"]),
                    fmt_float(row["relative_mse"]),
                    fmt_float(row["max_abs_error"]),
                    fmt_float(manifest.get("max_abs_error")),
                    "yes" if row.get("row_max_matches_manifest_max_abs") else "no",
                    str(row["max_abs_column"]),
                    str(row["worst_group_in_row"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir
    manifest = read_manifest(package_dir)
    by_name = tensor_by_name(manifest)
    layers = parse_layers(args.layers)
    suffixes = tuple(args.tensor_suffix or DEFAULT_TENSOR_SUFFIXES)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for layer in layers:
        for suffix in suffixes:
            name = f"model.language_model.layers.{layer}.{suffix}"
            tensor = by_name.get(name)
            if tensor is None:
                missing.append(name)
                continue
            rows.append(metric_row(package_dir, tensor, args.hidden_index, args.top_k))

    rows.sort(key=lambda item: (tensor_sort_key(item["tensor_name"]), item["row_index"]))
    worst_by_rms = max(rows, key=lambda item: item["rms"], default=None)
    worst_by_max_abs = max(rows, key=lambda item: item["max_abs_error"], default=None)
    manifest_max_match_count = sum(
        1 for row in rows if row.get("row_max_matches_manifest_max_abs")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "package_dir": str(package_dir),
        "hidden_index": args.hidden_index,
        "layers": layers,
        "tensor_suffixes": list(suffixes),
        "row_count": len(rows),
        "missing_tensors": missing,
        "manifest_max_match_count": manifest_max_match_count,
        "manifest_max_match_ratio": manifest_max_match_count / len(rows) if rows else None,
        "worst_by_rms": worst_by_rms,
        "worst_by_max_abs": worst_by_max_abs,
        "rows": rows,
    }


def tensor_sort_key(tensor_name: str) -> tuple[int, int, str]:
    parts = tensor_name.split(".")
    layer = 1_000_000
    if len(parts) > 3:
        try:
            layer = int(parts[3])
        except ValueError:
            pass
    suffix_priority = 0 if "linear_attn.out_proj.weight" in tensor_name else 1
    return layer, suffix_priority, tensor_name


def main() -> int:
    args = parse_args()
    summary = build_summary(args)
    if args.summary_json:
        write_json(args.summary_json, summary)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_table(summary["rows"]), encoding="utf-8")
    print(
        "package-row-quant-error "
        f"rows={summary['row_count']} hidden_index={summary['hidden_index']} "
        f"worst_rms={fmt_float(summary['worst_by_rms']['rms'] if summary['worst_by_rms'] else None)} "
        f"worst_max_abs={fmt_float(summary['worst_by_max_abs']['max_abs_error'] if summary['worst_by_max_abs'] else None)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
