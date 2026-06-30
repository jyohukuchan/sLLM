#!/usr/bin/env python3
"""Export sampled aq family codebooks for converter dry-runs."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=64)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


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
    )
    codebooks = sampler.build_family_codebooks(build_args, tensors, candidates, activation_stats)

    rows = []
    for (family, candidate_id), codebook in sorted(codebooks.items()):
        values = [float(value) for value in codebook.to(torch.float32).tolist()]
        rows.append(
            {
                "family": family,
                "candidate_id": candidate_id,
                "entry_count": len(values),
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
