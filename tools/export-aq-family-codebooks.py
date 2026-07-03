#!/usr/bin/env python3
"""Export sampled aq family codebooks for converter dry-runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TensorRef:
    name: str
    path: Path
    family: str
    codebook_scope: str
    candidate_id: str | None
    n_elements: int | None


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


def resolve_candidates(sampler, candidate_ids: list[str]):
    by_id = {candidate.candidate_id: candidate for candidate in sampler.ROUND1_CANDIDATES}
    candidates = []
    missing = []
    for candidate_id in candidate_ids:
        candidate = by_id.get(candidate_id)
        if candidate is None and hasattr(sampler, "candidate_from_id"):
            candidate = sampler.candidate_from_id(candidate_id)
        if candidate is None:
            missing.append(candidate_id)
        else:
            candidates.append(candidate)
    if missing:
        raise SystemExit(f"unknown candidates: {', '.join(sorted(missing))}")
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, default=None)
    parser.add_argument("--activation-stats", type=Path, default=None)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--tensor-pattern", default=r"\.weight$")
    parser.add_argument("--max-tensors", type=int, default=None)
    parser.add_argument("--max-tensors-per-family", type=int, default=None)
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


def load_plan_tensor_refs(
    path: Path,
    model_dir: Path,
    tensor_pattern: re.Pattern[str],
    allowed_families: set[str] | None,
) -> list[TensorRef]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    refs: list[TensorRef] = []
    for row in plan.get("tensors", []):
        if row.get("action") != "quantize":
            continue
        name = str(row["name"])
        if not tensor_pattern.search(name):
            continue
        family = str(row["family"])
        if allowed_families is not None and family not in allowed_families:
            continue
        source_file = Path(str(row["source_file"]))
        if not source_file.is_absolute():
            source_file = model_dir / source_file
        candidate_id = row.get("quant_format")
        refs.append(
            TensorRef(
                name=name,
                path=source_file,
                family=family,
                codebook_scope=str(row.get("codebook_scope") or family),
                candidate_id=str(candidate_id) if candidate_id is not None else None,
                n_elements=int(row["n_elements"]) if row.get("n_elements") is not None else None,
            )
        )
    return refs


def discover_tensor_refs(
    sampler,
    model_dir: Path,
    tensor_pattern: re.Pattern[str],
    allowed_families: set[str] | None,
) -> list[TensorRef]:
    refs = []
    for name, path in sampler.discover_tensors(model_dir, tensor_pattern):
        family = sampler.family_for_tensor(name)
        if allowed_families is not None and family not in allowed_families:
            continue
        refs.append(
            TensorRef(
                name=name,
                path=path,
                family=family,
                codebook_scope=family,
                candidate_id=None,
                n_elements=None,
            )
        )
    return refs


def limit_tensor_refs_by_family(
    tensors: list[TensorRef],
    max_tensors: int | None,
    max_tensors_per_family: int | None,
) -> list[TensorRef]:
    selected: list[TensorRef] = []
    family_counts: dict[str, int] = defaultdict(int)
    for tensor in tensors:
        if max_tensors is not None and len(selected) >= max_tensors:
            break
        if max_tensors_per_family is not None and family_counts[tensor.family] >= max_tensors_per_family:
            continue
        selected.append(tensor)
        family_counts[tensor.family] += 1
    return selected


def build_family_codebooks(
    sampler,
    args: argparse.Namespace,
    tensors: list[TensorRef],
    candidates,
    activation_stats: dict[str, torch.Tensor],
):
    codebooks: dict[tuple[str, str], torch.Tensor] = {}
    weighting: dict[tuple[str, str], str] = {}
    families: dict[tuple[str, str], str] = {}
    source_tensors: dict[tuple[str, str], list[str]] = defaultdict(list)
    source_elements: dict[tuple[str, str], int] = defaultdict(int)
    fallbacks: list[dict[str, str]] = []

    for candidate_index, candidate in enumerate(candidates):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + 100_000 + candidate_index)
        scope_values: dict[str, list[torch.Tensor]] = defaultdict(list)
        scope_weights: dict[str, list[torch.Tensor]] = defaultdict(list)
        scope_missing_stats: set[str] = set()
        scope_families: dict[str, str] = {}
        for tensor_ref in tensors:
            if tensor_ref.candidate_id is not None and tensor_ref.candidate_id != candidate.candidate_id:
                continue
            family = tensor_ref.family
            scope = tensor_ref.codebook_scope
            scope_families.setdefault(scope, family)
            with safe_open(tensor_ref.path, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(tensor_ref.name)
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
                        tensor_ref.name,
                        tensor_shape,
                        activation_stats,
                    )
                except ValueError as exc:
                    if args.missing_activation_stats != "unweighted":
                        raise
                    scope_missing_stats.add(scope)
                    fallbacks.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "family": family,
                            "codebook_scope": scope,
                            "tensor": tensor_ref.name,
                            "reason": str(exc),
                        }
                    )
                else:
                    group_weights = activation_second_moment.index_select(0, columns.flatten()).view_as(groups)
            values, weights = sampler.normalized_values_and_weights(groups, group_weights)
            scope_values[scope].append(values)
            if weights is not None:
                scope_weights[scope].append(weights)
            key = (scope, candidate.candidate_id)
            families[key] = family
            source_tensors[key].append(tensor_ref.name)
            if tensor_ref.n_elements is not None:
                source_elements[key] += tensor_ref.n_elements
            del tensor, groups

        for scope, chunks in scope_values.items():
            values = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
            weights = None
            weight_mode = "unweighted"
            if args.weighted_codebook and scope not in scope_missing_stats:
                weight_chunks = scope_weights[scope]
                weights = torch.cat(weight_chunks) if len(weight_chunks) > 1 else weight_chunks[0]
                weight_mode = "weighted"
            elif args.weighted_codebook:
                weight_mode = "unweighted_missing_activation_stats"
            key = (scope, candidate.candidate_id)
            codebooks[key] = sampler.codebook_from_normalized_values(
                values,
                candidate.codebook_mode,
                weights,
            )
            weighting[key] = weight_mode
            families[key] = scope_families[scope]

    return codebooks, weighting, families, source_tensors, source_elements, fallbacks


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    args.model_dir = args.model_dir.expanduser().resolve()
    args.plan_json = args.plan_json.expanduser().resolve() if args.plan_json else None
    args.activation_stats = args.activation_stats.expanduser().resolve() if args.activation_stats else None
    sampler = load_sampler_module()

    tensor_pattern = re.compile(args.tensor_pattern)
    allowed = set(args.family) if args.family else None
    if args.plan_json is not None:
        tensors = load_plan_tensor_refs(args.plan_json, args.model_dir, tensor_pattern, allowed)
        candidate_ids = args.candidate or sorted(
            {tensor.candidate_id for tensor in tensors if tensor.candidate_id is not None}
        )
        max_tensors = args.max_tensors
        max_tensors_per_family = args.max_tensors_per_family
    else:
        tensors = discover_tensor_refs(sampler, args.model_dir, tensor_pattern, allowed)
        candidate_ids = args.candidate or DEFAULT_CANDIDATES
        max_tensors = args.max_tensors if args.max_tensors is not None else 64
        max_tensors_per_family = (
            args.max_tensors_per_family if args.max_tensors_per_family is not None else 4
        )
    candidates = resolve_candidates(sampler, list(candidate_ids))
    tensors = limit_tensor_refs_by_family(tensors, max_tensors, max_tensors_per_family)
    activation_stats = sampler.load_activation_stats(args.activation_stats)
    if args.weighted_codebook and not activation_stats:
        raise SystemExit("--weighted-codebook requires --activation-stats")

    build_args = SimpleNamespace(
        seed=args.seed,
        max_elements_per_tensor=args.max_elements_per_tensor,
        weighted_codebook=args.weighted_codebook,
        missing_activation_stats=args.missing_activation_stats,
    )
    codebooks, weighting, families, source_tensors, source_elements, fallbacks = build_family_codebooks(
        sampler,
        build_args,
        tensors,
        candidates,
        activation_stats,
    )

    rows = []
    for (scope, candidate_id), codebook in sorted(codebooks.items()):
        family = families[(scope, candidate_id)]
        values = [float(value) for value in codebook.to(torch.float32).tolist()]
        row = {
            "family": family,
            "candidate_id": candidate_id,
            "entry_count": len(values),
            "weighting": weighting[(scope, candidate_id)],
            "values_f32": values,
            "min": min(values),
            "max": max(values),
            "source_tensor_count": len(source_tensors[(scope, candidate_id)]),
            "source_tensors": source_tensors[(scope, candidate_id)],
        }
        if scope != family:
            row["codebook_scope"] = scope
        if source_elements[(scope, candidate_id)]:
            row["source_elements"] = source_elements[(scope, candidate_id)]
        rows.append(row)

    result = {
        "schema_version": "aq-family-codebook-export-v0.2",
        "timestamp_utc": utc_now(),
        "model_dir": str(args.model_dir),
        "plan_json": str(args.plan_json) if args.plan_json else None,
        "activation_stats": str(args.activation_stats) if args.activation_stats else None,
        "weighted_codebook": args.weighted_codebook,
        "missing_activation_stats": args.missing_activation_stats,
        "activation_weighting_fallbacks": fallbacks,
        "seed": args.seed,
        "max_elements_per_tensor": args.max_elements_per_tensor,
        "max_tensors": max_tensors,
        "max_tensors_per_family": max_tensors_per_family,
        "family_filter": args.family,
        "candidate_filter": [candidate.candidate_id for candidate in candidates],
        "tensor_names": [tensor.name for tensor in tensors],
        "notes": args.note,
        "codebooks": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
