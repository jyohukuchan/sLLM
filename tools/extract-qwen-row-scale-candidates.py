#!/usr/bin/env python3
"""Extract row-scale override candidates from Qwen layer module traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


CANDIDATE_SCHEMA_VERSION = "qwen-row-scale-candidates-v0.1"
MANIFEST_SCHEMA_VERSION = "row-scale-overrides-v0.1"
SMOKE_SCHEMA_VERSION = "package-row-scale-overrides-v0.1"

ROW_DOT_TENSOR_SUFFIX = {
    "attention_out_proj": "linear_attn.out_proj.weight",
    "mlp_down_proj": "mlp.down_proj.weight",
    "self_attention_o_proj": "self_attn.o_proj.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Named qwen-layer-module-trace JSONL file. May be repeated.",
    )
    parser.add_argument("--candidates-json", type=Path)
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument("--smoke-json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--source-prefix", default="qwen-layer-module-trace-scale-fit")
    parser.add_argument("--min-rmse-improvement", type=float, default=0.0)
    parser.add_argument("--min-original-rmse", type=float, default=0.0)
    parser.add_argument(
        "--duplicate-policy",
        choices=["best", "fail"],
        default="best",
        help="How to handle repeated layer/tensor/row candidates across traces.",
    )
    return parser.parse_args()


def parse_named_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--trace must be LABEL=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"--trace label must not be empty: {spec!r}")
    return label, Path(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"failed to parse {path}:{line_number}: {err}") from err
            if not isinstance(value, dict):
                raise ValueError(f"trace row must be a JSON object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"trace has no rows: {path}")
    return rows


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def row_scale_observations(
    label: str,
    path: Path,
    row: dict[str, Any],
    source_prefix: str,
) -> list[dict[str, Any]]:
    layer_index = row.get("layer_index")
    hidden_index = row.get("hidden_index")
    if not isinstance(layer_index, int) or not isinstance(hidden_index, int):
        return []
    row_dot = row.get("row_dot")
    if not isinstance(row_dot, dict):
        return []

    observations = []
    for projection_name, tensor_suffix in ROW_DOT_TENSOR_SUFFIX.items():
        projection = row_dot.get(projection_name)
        if not isinstance(projection, dict):
            continue
        scale_fit = projection.get("scale_fit")
        if not isinstance(scale_fit, dict):
            continue
        scale = finite_number(scale_fit.get("optimal_scale"))
        if scale is None or scale <= 0.0:
            continue
        tensor_name = f"model.language_model.layers.{layer_index}.{tensor_suffix}"
        source = f"{source_prefix}:{label}:{projection_name}"
        observations.append(
            {
                "label": label,
                "path": str(path),
                "fixture": row.get("fixture"),
                "schema_version": row.get("schema_version"),
                "layer_index": layer_index,
                "layer_type": row.get("layer_type"),
                "hidden_index": hidden_index,
                "row_index": hidden_index,
                "projection": projection_name,
                "tensor_suffix": tensor_suffix,
                "tensor_name": tensor_name,
                "scale": scale,
                "source": source,
                "original_rmse": finite_number(scale_fit.get("original_rmse")),
                "scaled_rmse": finite_number(scale_fit.get("scaled_rmse")),
                "rmse_improvement_ratio": finite_number(scale_fit.get("rmse_improvement_ratio")),
                "original_max_abs_error": finite_number(scale_fit.get("original_max_abs_error")),
                "scaled_max_abs_error": finite_number(scale_fit.get("scaled_max_abs_error")),
                "worst_token_index": scale_fit.get("worst_token_index"),
                "worst_scaled_token_index": scale_fit.get("worst_scaled_token_index"),
            }
        )
    return observations


def passes_filters(
    observation: dict[str, Any],
    min_rmse_improvement: float,
    min_original_rmse: float,
) -> bool:
    original_rmse = observation.get("original_rmse")
    if isinstance(original_rmse, (int, float)) and float(original_rmse) < min_original_rmse:
        return False
    improvement = observation.get("rmse_improvement_ratio")
    if improvement is None:
        return min_rmse_improvement <= 0.0
    return float(improvement) >= min_rmse_improvement


def candidate_key(observation: dict[str, Any]) -> tuple[int, str, int]:
    return (
        int(observation["layer_index"]),
        str(observation["tensor_suffix"]),
        int(observation["row_index"]),
    )


def selection_score(observation: dict[str, Any]) -> tuple[float, float, float, str]:
    improvement = observation.get("rmse_improvement_ratio")
    scaled_rmse = observation.get("scaled_rmse")
    original_rmse = observation.get("original_rmse")
    return (
        float(improvement) if isinstance(improvement, (int, float)) else float("-inf"),
        -(float(scaled_rmse) if isinstance(scaled_rmse, (int, float)) else float("inf")),
        float(original_rmse) if isinstance(original_rmse, (int, float)) else 0.0,
        str(observation.get("label", "")),
    )


def build_candidates(
    observations: list[dict[str, Any]],
    duplicate_policy: str,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    for observation in observations:
        by_key.setdefault(candidate_key(observation), []).append(observation)

    duplicate_keys = [key for key, values in by_key.items() if len(values) > 1]
    if duplicate_policy == "fail" and duplicate_keys:
        formatted = ", ".join(f"layer={key[0]} suffix={key[1]} row={key[2]}" for key in duplicate_keys)
        raise ValueError(f"duplicate row-scale candidates: {formatted}")

    candidates = []
    for key, key_observations in sorted(by_key.items()):
        selected = max(key_observations, key=selection_score)
        candidates.append(
            {
                "layer_index": key[0],
                "tensor_suffix": key[1],
                "tensor_name": selected["tensor_name"],
                "row_index": key[2],
                "scale": selected["scale"],
                "source": selected["source"],
                "selected_label": selected["label"],
                "projection": selected["projection"],
                "observation_count": len(key_observations),
                "observations": sorted(
                    key_observations,
                    key=lambda item: (
                        str(item.get("label", "")),
                        str(item.get("projection", "")),
                        str(item.get("path", "")),
                    ),
                ),
            }
        )
    return candidates


def manifest_json(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "entries": [
            {
                "tensor_name": candidate["tensor_name"],
                "row_index": candidate["row_index"],
                "scale": candidate["scale"],
                "source": candidate["source"],
            }
            for candidate in candidates
        ],
    }


def smoke_json(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "overrides": [
            {
                "layer_index": candidate["layer_index"],
                "tensor_suffix": candidate["tensor_suffix"],
                "row_index": candidate["row_index"],
                "scale": candidate["scale"],
            }
            for candidate in candidates
        ],
    }


def fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.9g}"
    if value is None:
        return "-"
    return str(value)


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen Row-Scale Candidates",
        "",
        f"- schema: `{summary['schema_version']}`",
        f"- trace count: `{len(summary['traces'])}`",
        f"- observation count: `{len(summary['observations'])}`",
        f"- candidate count: `{len(summary['candidates'])}`",
        "",
        "| layer | tensor_suffix | row | scale | selected | obs | rmse | scaled_rmse | improvement |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for candidate in summary["candidates"]:
        selected = next(
            item
            for item in candidate["observations"]
            if item["label"] == candidate["selected_label"]
            and item["projection"] == candidate["projection"]
            and item["scale"] == candidate["scale"]
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(candidate["layer_index"]),
                    str(candidate["tensor_suffix"]),
                    fmt(candidate["row_index"]),
                    fmt(candidate["scale"]),
                    str(candidate["selected_label"]),
                    fmt(candidate["observation_count"]),
                    fmt(selected.get("original_rmse")),
                    fmt(selected.get("scaled_rmse")),
                    fmt(selected.get("rmse_improvement_ratio")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.trace:
        raise SystemExit("at least one --trace is required")
    if not any([args.candidates_json, args.manifest_json, args.smoke_json, args.markdown]):
        raise SystemExit("at least one output path is required")

    traces = [parse_named_path(spec) for spec in args.trace]
    observations = []
    for label, path in traces:
        for row in read_jsonl(path):
            observations.extend(row_scale_observations(label, path, row, args.source_prefix))
    filtered_observations = [
        observation
        for observation in observations
        if passes_filters(observation, args.min_rmse_improvement, args.min_original_rmse)
    ]
    candidates = build_candidates(filtered_observations, args.duplicate_policy)
    summary = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "filters": {
            "min_rmse_improvement": args.min_rmse_improvement,
            "min_original_rmse": args.min_original_rmse,
            "duplicate_policy": args.duplicate_policy,
        },
        "traces": [{"label": label, "path": str(path)} for label, path in traces],
        "observations": filtered_observations,
        "candidates": candidates,
    }

    if args.candidates_json:
        write_json(args.candidates_json, summary)
    if args.manifest_json:
        write_json(args.manifest_json, manifest_json(candidates))
    if args.smoke_json:
        write_json(args.smoke_json, smoke_json(candidates))
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(summary), encoding="utf-8")
    print(f"qwen-row-scale-candidates candidates={len(candidates)} observations={len(filtered_observations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
