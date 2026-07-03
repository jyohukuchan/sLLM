#!/usr/bin/env python3
"""Compare AQ local-scale formats on an explicit JSONL tensor row set."""

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
from typing import Any

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aq_scale_formats import scale_values  # noqa: E402


SCHEMA_VERSION = "aq-local-scale-format-comparison-v0.2"


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
    parser.add_argument("--format", action="append", required=True, help="local-scale format, e.g. e4m3 or ue3m5")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--codebook-token", default="flloyd16")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--scale-window", type=int, default=4)
    parser.add_argument("--global-scale-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--max-elements-per-tensor", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=2501)
    parser.add_argument("--torch-threads", type=int, default=default_threads)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--ggml-type", action="append", help="Only include source rows with this GGML type.")
    parser.add_argument("--max-rows", type=int, default=None)
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
            q = row.get("quantization", {})
            if ggml_types and q.get("ggml_type") not in ggml_types:
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


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def weighted_mean(pairs: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return float("nan")
    return float(sum(value * weight for value, weight in pairs) / total_weight)


def summarize_rows(results: list[dict[str, Any]], formats: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_rows": len(results),
        "formats": {},
        "by_ggml_type": {},
        "by_family": {},
    }
    for fmt in formats:
        values: list[float] = []
        weights: list[tuple[float, float]] = []
        low_clamp = 0
        high_clamp = 0
        wins_vs_ud = 0
        for row in results:
            item = row["formats"][fmt]
            rel = float(item["alternating"]["relative_mse"])
            values.append(rel)
            weights.append((rel, float(row["n_elements"])))
            low_clamp += int(item["alternating"]["local_scale_clamped_low"])
            high_clamp += int(item["alternating"]["local_scale_clamped_high"])
            if rel < float(row["ud_relative_mse"]):
                wins_vs_ud += 1
        summary["formats"][fmt] = {
            "mean_relative_mse": mean(values),
            "element_weighted_relative_mse": weighted_mean(weights),
            "min_relative_mse": min(values) if values else None,
            "max_relative_mse": max(values) if values else None,
            "wins_vs_ud_rows": wins_vs_ud,
            "local_scale_clamped_low": low_clamp,
            "local_scale_clamped_high": high_clamp,
        }

    for key_name in ("ud_ggml_type", "family"):
        out_key = "by_ggml_type" if key_name == "ud_ggml_type" else "by_family"
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in results:
            buckets[str(row[key_name])].append(row)
        for bucket, rows in sorted(buckets.items()):
            bucket_summary: dict[str, Any] = {
                "rows": len(rows),
                "ud_mean_relative_mse": mean([float(row["ud_relative_mse"]) for row in rows]),
                "ud_element_weighted_relative_mse": weighted_mean(
                    [(float(row["ud_relative_mse"]), float(row["n_elements"])) for row in rows]
                ),
                "formats": {},
            }
            for fmt in formats:
                vals = [float(row["formats"][fmt]["alternating"]["relative_mse"]) for row in rows]
                weighted = [
                    (float(row["formats"][fmt]["alternating"]["relative_mse"]), float(row["n_elements"]))
                    for row in rows
                ]
                bucket_summary["formats"][fmt] = {
                    "mean_relative_mse": mean(vals),
                    "element_weighted_relative_mse": weighted_mean(weighted),
                    "wins_vs_ud_rows": sum(
                        1
                        for row in rows
                        if float(row["formats"][fmt]["alternating"]["relative_mse"]) < float(row["ud_relative_mse"])
                    ),
                }
            summary[out_key][bucket] = bucket_summary

    if "e4m3" in summary["formats"]:
        e4m3 = summary["formats"]["e4m3"]["mean_relative_mse"]
        for fmt, item in summary["formats"].items():
            item["ratio_vs_e4m3_mean"] = float(item["mean_relative_mse"] / e4m3) if e4m3 else None
    return summary


def extract_local_scale_stats(candidate_result: dict[str, Any]):
    state = candidate_result.pop("_optimized_state", None)
    if state is None:
        return None
    local_indices = state.local_indices.to(torch.long)
    scales = state.scales.to(torch.float32)
    if local_indices.numel() == 0:
        return
    idx_min = int(local_indices.min())
    idx_max = int(local_indices.max())
    alt = candidate_result.setdefault("alternating", {})
    alt["local_scale_index_min"] = idx_min
    alt["local_scale_index_max"] = idx_max
    alt["local_scale_value_min"] = float(scales[idx_min])
    alt["local_scale_value_max"] = float(scales[idx_max])
    alt["local_scale_clamped_low"] = int((local_indices == 0).sum())
    alt["local_scale_clamped_high"] = int((local_indices == scales.numel() - 1).sum())
    alt["local_scale_count"] = int(scales.numel())
    return state


def main() -> int:
    args = parse_args()
    if args.group_size < 1:
        raise SystemExit("--group-size must be >= 1")
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if args.scale_window < 0:
        raise SystemExit("--scale-window must be >= 0")
    if args.max_elements_per_tensor < args.group_size:
        raise SystemExit("--max-elements-per-tensor must be >= --group-size")
    if args.torch_threads < 1 or args.torch_interop_threads < 1:
        raise SystemExit("torch thread counts must be >= 1")

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

    formats = [fmt.lower() for fmt in args.format]
    for fmt in formats:
        _ = scale_values(fmt)
    candidates = []
    for fmt in formats:
        candidate_id = f"aq4_{fmt}_g{args.group_size}_ts_{args.codebook_token}"
        candidate = sampler.candidate_from_id(candidate_id)
        if candidate is None:
            raise SystemExit(f"unsupported candidate id: {candidate_id}")
        candidates.append(candidate)

    opt_args = argparse.Namespace(
        max_elements_per_tensor=args.max_elements_per_tensor,
        iterations=args.iterations,
        scale_window=args.scale_window,
        global_scale_dtype=args.global_scale_dtype,
    )

    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    for row_index, source in enumerate(source_rows):
        tensor_name = tensor_name_from_source(source)
        path = tensor_paths.get(tensor_name)
        if path is None:
            raise RuntimeError(f"tensor not found in model: {tensor_name}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(tensor_name)

        family = sampler.family_for_tensor(tensor_name)
        q = source.get("quantization", {})
        metrics = source.get("metrics", {})
        result_row: dict[str, Any] = {
            "tensor_name": tensor_name,
            "family": family,
            "n_elements": int(q.get("n_elements", tensor.numel())),
            "ud_ggml_type": q.get("ggml_type"),
            "ud_bpp": q.get("effective_bpp"),
            "ud_relative_mse": metrics.get("relative_mse"),
            "formats": {},
        }
        floor_states = []
        row_seed = args.seed + row_index
        for candidate in candidates:
            candidate_result = optimizer.run_candidate(
                sampler,
                opt_args,
                tensor_name,
                tensor,
                candidate,
                row_seed,
                floor_states=floor_states,
            )
            state = extract_local_scale_stats(candidate_result)
            if state is not None:
                floor_states.append(state)
            result_row["formats"][candidate.scale_format] = candidate_result
        results.append(result_row)
        del tensor

    summary = summarize_rows(results, formats)
    summary["elapsed_sec"] = time.perf_counter() - start
    output = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "description": "Compare local-scale formats on explicit UD comparison rows using aq4 g16 free-Lloyd16 alternating optimization.",
        "settings": {
            "model_dir": str(args.model_dir),
            "source_rows": str(args.source_rows),
            "formats": formats,
            "group_size": args.group_size,
            "codebook_token": args.codebook_token,
            "iterations": args.iterations,
            "scale_window": args.scale_window,
            "global_scale_dtype": args.global_scale_dtype,
            "max_elements_per_tensor": args.max_elements_per_tensor,
            "seed": args.seed,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
            "ggml_type_filter": sorted(ggml_types) if ggml_types else None,
            "max_rows": args.max_rows,
        },
        "summary": summary,
        "results": results,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
