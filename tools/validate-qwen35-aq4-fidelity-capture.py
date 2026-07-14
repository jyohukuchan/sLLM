#!/usr/bin/env python3
"""Validate the hash-bound 24-row fidelity metrics input."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ullm.aq4_p2_fidelity_calibration_metrics.v1"
MAX_ROWS = 24
METRICS = {"token_agreement_rate", "topk_overlap_rate_k10", "logits_cosine", "logits_relative_l2", "hidden_cosine", "hidden_relative_l2", "hidden_max_abs", "bf16_top1_retained_in_aq4_top10_rate"}
SHA_FIELDS = ("split_manifest_sha256", "policy_sha256", "calibration_cases_sha256", "source_manifest_sha256", "active_manifest_sha256")


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SPLIT = _load("aq4_fidelity_split_validator", "validate-aq4-p2-fidelity-holdout.py")


class ValidationError(ValueError):
    pass


def sha(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"{label} must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValidationError(f"{label} must be finite")


def validate(metrics_path: Path, split_root: Path) -> dict[str, Any]:
    try:
        SPLIT.validate(split_root)
    except Exception as error:
        raise ValidationError(f"split validation failed: {error}") from error
    raw = metrics_path.read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        raise ValidationError("metrics output exceeds bounded size")
    value = json.loads(raw, object_pairs_hook=lambda pairs: _pairs(pairs), parse_constant=lambda token: (_ for _ in ()).throw(ValidationError(f"non-finite JSON value: {token}")))
    required = {"schema_version", "status", "subset", "row_count", *SHA_FIELDS, "identity", "rows"}
    if not isinstance(value, dict) or set(value) != required or value.get("schema_version") != SCHEMA or value.get("status") != "ready_for_freeze" or value.get("subset") != "calibration" or value.get("row_count") != MAX_ROWS:
        raise ValidationError("metrics root schema/status differs")
    for field in SHA_FIELDS:
        if not isinstance(value[field], str) or len(value[field]) != 64 or any(char not in "0123456789abcdef" for char in value[field]):
            raise ValidationError(f"{field} is not a SHA-256 digest")
    expected_rows = {}
    for line in (split_root / "calibration-cases.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        expected_rows[row["case_id"]] = row
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != MAX_ROWS:
        raise ValidationError("metrics rows must contain exactly 24 entries")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("case_id") in seen:
            raise ValidationError("metrics rows contain duplicate case_id")
        case_id = row.get("case_id")
        seen.add(case_id)
        if case_id not in expected_rows:
            raise ValidationError(f"metrics row is extra: {case_id}")
        expected = expected_rows[case_id]
        for field in ("case_sha256", "fixture_sha256", "prompt_token_ids_sha256", "context_token_ids_sha256", "prompt_tokens", "context_tokens", "baseline_mode", "prefill_requested_m", "resolved_m", "step", "row_count"):
            if row.get(field) != expected.get(field):
                raise ValidationError(f"metrics row {case_id} identity differs: {field}")
        if row.get("step") != 0 or row.get("row_count") != 1:
            raise ValidationError(f"metrics row {case_id} is not step zero")
        greedy = row.get("greedy")
        ordered = row.get("ordered_top10")
        if not isinstance(greedy, dict) or set(greedy) != {"source", "active", "exact"} or not isinstance(greedy["source"], int) or not isinstance(greedy["active"], int) or greedy["exact"] != (greedy["source"] == greedy["active"]):
            raise ValidationError(f"metrics greedy contract differs: {case_id}")
        if not isinstance(ordered, dict) or set(ordered) != {"source", "active", "exact", "overlap"} or not isinstance(ordered["source"], list) or not isinstance(ordered["active"], list) or len(ordered["source"]) != 10 or len(ordered["active"]) != 10 or ordered["exact"] != (ordered["source"] == ordered["active"]):
            raise ValidationError(f"metrics ordered top10 contract differs: {case_id}")
        finite(ordered["overlap"], f"{case_id}.overlap")
        if not 0.0 <= float(ordered["overlap"]) <= 1.0:
            raise ValidationError(f"{case_id}.overlap is outside [0,1]")
        metrics_row = row.get("metrics")
        if not isinstance(metrics_row, dict) or set(metrics_row) != METRICS:
            raise ValidationError(f"metrics set differs: {case_id}")
        for name, metric in metrics_row.items():
            finite(metric, f"{case_id}.{name}")
            if float(metric) < 0 or name in {"token_agreement_rate", "topk_overlap_rate_k10", "logits_cosine", "hidden_cosine", "bf16_top1_retained_in_aq4_top10_rate"} and float(metric) > 1:
                raise ValidationError(f"{case_id}.{name} is outside domain")
            if name in {"logits_relative_l2", "hidden_relative_l2"} and float(metric) > 1:
                raise ValidationError(f"{case_id}.{name} exceeds pathological ceiling")
        raw_row = row.get("raw")
        if not isinstance(raw_row, dict) or set(raw_row) != {"hidden", "logits", "source_top1_retained_in_active_top10"} or raw_row["source_top1_retained_in_active_top10"] != (greedy["source"] in ordered["active"]):
            raise ValidationError(f"raw sufficient-statistics contract differs: {case_id}")
        for name, elements in (("hidden", 4096), ("logits", 248320)):
            stats = raw_row[name]
            if not isinstance(stats, dict) or stats.get("elements") != elements:
                raise ValidationError(f"raw {name} shape differs: {case_id}")
            for field in ("reference_norm_sq", "candidate_norm_sq", "dot", "delta_norm_sq", "relative_l2", "cosine", "max_abs"):
                finite(stats.get(field), f"{case_id}.raw.{name}.{field}")
    if seen != set(expected_rows):
        raise ValidationError("metrics rows are missing calibration cases")
    return {"status": "ok", "row_count": MAX_ROWS, "metrics_sha256": sha(metrics_path, "metrics")}


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = child
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(validate(args.metrics, args.split_root), sort_keys=True))
        return 0
    except (ValidationError, OSError, ValueError) as error:
        print(f"Qwen3.5 AQ4 fidelity metrics validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
