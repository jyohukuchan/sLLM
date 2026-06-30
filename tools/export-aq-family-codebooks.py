#!/usr/bin/env python3
"""Export sampled aq family codebooks for converter dry-runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open


DEFAULT_CANDIDATES = [
    "aq4_e4m3_g16_ts_flloyd16",
    "aq4_e4m3_g8_ts_flloyd16",
]


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
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--activation-stats", type=Path, default=None)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--tensor-pattern", default=r"\.weight$")
    parser.add_argument("--max-tensors", type=int, default=64)
    parser.add_argument("--max-tensors-per-family", type=int, default=4)
    parser.add_argument("--max-elements-per-tensor", type=int, default=262144)
    parser.add_argument("--weighted-codebook", action="store_true")
    parser.add_argument(
        "--missing-activation-stats",
        choices=("error", "unweighted"),
        default="error",
        help="Behavior for --weighted-codebook tensors without activation stats.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=64)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def build_family_codebooks(
    sampler,
    args: argparse.Namespace,
    tensors: list[tuple[str, Path]],
    candidates,
    activation_stats: dict[str, torch.Tensor],
):
    codebooks: dict[tuple[str, str], torch.Tensor] = {}
    weighting: dict[tuple[str, str], str] = {}
    fallbacks: list[dict[str, str]] = []

    for candidate_index, candidate in enumerate(candidates):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + 100_000 + candidate_index)
        family_values: dict[str, list[torch.Tensor]] = defaultdict(list)
        family_weights: dict[str, list[torch.Tensor]] = defaultdict(list)
        family_missing_stats: set[str] = set()
        for tensor_name, path in tensors:
            family = sampler.family_for_tensor(tensor_name)
            with safe_open(path, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(tensor_name)
            if tensor.ndim < 2 or not tensor.is_floating_point():
                continue
            tensor_shape = tuple(int(dim) for dim in tensor.shape)
            groups, columns = sampler.sample_groups_with_columns(
                tensor,
                candidate.group_size,
                args.max_elements_per_tensor,
                generator,
            )
            group_weights = None
            if args.weighted_codebook:
                if columns is None:
                    raise ValueError(f"cannot apply activation stats to non-2D tensor {tensor_name}")
                try:
                    activation_second_moment = sampler.activation_stats_for_tensor(
                        tensor_name,
                        tensor_shape,
                        activation_stats,
                    )
                except ValueError as exc:
                    if args.missing_activation_stats != "unweighted":
                        raise
                    family_missing_stats.add(family)
                    fallbacks.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "family": family,
                            "tensor": tensor_name,
                            "reason": str(exc),
                        }
                    )
                else:
                    group_weights = activation_second_moment.index_select(0, columns.flatten()).view_as(groups)
            values, weights = sampler.normalized_values_and_weights(groups, group_weights)
            family_values[family].append(values)
            if weights is not None:
                family_weights[family].append(weights)
            del tensor, groups

        for family, chunks in family_values.items():
            values = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
            weights = None
            weight_mode = "unweighted"
            if args.weighted_codebook and family not in family_missing_stats:
                weight_chunks = family_weights[family]
                weights = torch.cat(weight_chunks) if len(weight_chunks) > 1 else weight_chunks[0]
                weight_mode = "weighted"
            elif args.weighted_codebook:
                weight_mode = "unweighted_missing_activation_stats"
            key = (family, candidate.candidate_id)
            codebooks[key] = sampler.codebook_from_normalized_values(
                values,
                candidate.codebook_mode,
                weights,
            )
            weighting[key] = weight_mode

    return codebooks, weighting, fallbacks


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    args.model_dir = args.model_dir.expanduser().resolve()
    args.activation_stats = args.activation_stats.expanduser().resolve() if args.activation_stats else None
    sampler = load_sampler_module()

    selected_candidate_ids = set(args.candidate or DEFAULT_CANDIDATES)
    candidates = [item for item in sampler.ROUND1_CANDIDATES if item.candidate_id in selected_candidate_ids]
    missing = selected_candidate_ids - {item.candidate_id for item in candidates}
    if missing:
        raise SystemExit(f"unknown candidates: {', '.join(sorted(missing))}")

    tensors = sampler.discover_tensors(args.model_dir, re.compile(args.tensor_pattern))
    if args.family:
        allowed = set(args.family)
        tensors = [(name, path) for name, path in tensors if sampler.family_for_tensor(name) in allowed]
    tensors = sampler.limit_tensors_by_family(tensors, args.max_tensors, args.max_tensors_per_family)
    activation_stats = sampler.load_activation_stats(args.activation_stats)
    if args.weighted_codebook and not activation_stats:
        raise SystemExit("--weighted-codebook requires --activation-stats")

    build_args = SimpleNamespace(
        seed=args.seed,
        max_elements_per_tensor=args.max_elements_per_tensor,
        weighted_codebook=args.weighted_codebook,
        missing_activation_stats=args.missing_activation_stats,
    )
    codebooks, weighting, fallbacks = build_family_codebooks(
        sampler,
        build_args,
        tensors,
        candidates,
        activation_stats,
    )

    rows = []
    for (family, candidate_id), codebook in sorted(codebooks.items()):
        values = [float(value) for value in codebook.to(torch.float32).tolist()]
        rows.append(
            {
                "family": family,
                "candidate_id": candidate_id,
                "entry_count": len(values),
                "weighting": weighting[(family, candidate_id)],
                "values_f32": values,
                "min": min(values),
                "max": max(values),
            }
        )

    result = {
        "schema_version": "aq-family-codebook-export-v0.1",
        "timestamp_utc": utc_now(),
        "model_dir": str(args.model_dir),
        "activation_stats": str(args.activation_stats) if args.activation_stats else None,
        "weighted_codebook": args.weighted_codebook,
        "missing_activation_stats": args.missing_activation_stats,
        "activation_weighting_fallbacks": fallbacks,
        "seed": args.seed,
        "max_elements_per_tensor": args.max_elements_per_tensor,
        "max_tensors": args.max_tensors,
        "max_tensors_per_family": args.max_tensors_per_family,
        "family_filter": args.family,
        "candidate_filter": sorted(selected_candidate_ids),
        "tensor_names": [name for name, _ in tensors],
        "notes": args.note,
        "codebooks": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
