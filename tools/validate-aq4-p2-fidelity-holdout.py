#!/usr/bin/env python3
"""Validate a prepared AQ4 P2 fidelity split and (optionally) its freeze receipt."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("aq4_fidelity_protocol", HERE / "generate-aq4-p2-fidelity-holdout.py")
if spec is None or spec.loader is None:  # pragma: no cover
    raise RuntimeError("cannot load protocol generator")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


class ValidationError(ValueError):
    pass


def finite(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValidationError(f"{label} is not finite")


def validate(root: Path, receipt_path: Path | None = None) -> dict[str, Any]:
    manifest, manifest_raw = protocol.load(root / "split-manifest.json", "split manifest")
    policy, policy_raw = protocol.load(root / "policy.json", "policy")
    if manifest.get("schema_version") != protocol.SPLIT_SCHEMA or manifest.get("status") != "ready_for_calibration":
        raise ValidationError("split manifest schema/status differs")
    if policy.get("schema_version") != protocol.POLICY_SCHEMA or policy.get("status") != "formula_frozen_unbound":
        raise ValidationError("policy schema/status differs")
    expected_names = {"calibration-cases.jsonl", "holdout-cases.jsonl", "policy.json", "split-manifest.json"}
    sums_path = root / "SHA256SUMS"
    protocol.reject_path(sums_path, "SHA256SUMS")
    sums: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or parts[1] in sums or not protocol.SHA_RE.fullmatch(parts[0]):
            raise ValidationError("invalid SHA256SUMS")
        sums[parts[1]] = parts[0]
    if set(sums) != expected_names:
        raise ValidationError("SHA256SUMS file set differs")
    for name in expected_names:
        if protocol.sha_file(root / name, name) != sums[name]:
            raise ValidationError(f"checksum differs: {name}")
    if manifest.get("calibration_sha256") != sums["calibration-cases.jsonl"] or manifest.get("holdout_sha256") != sums["holdout-cases.jsonl"] or manifest.get("policy_sha256") != sums["policy.json"]:
        raise ValidationError("manifest does not bind split files")
    calibration = protocol.read_jsonl(root / "calibration-cases.jsonl", "calibration cases")
    holdout = protocol.read_jsonl(root / "holdout-cases.jsonl", "holdout cases")
    if len(calibration) != 24 or len(holdout) != 24:
        raise ValidationError("split must contain 24 calibration and 24 holdout rows")
    all_rows = calibration + holdout
    ids = [row.get("case_id") for row in all_rows]
    hashes = [row.get("case_sha256") for row in all_rows]
    if any(not isinstance(value, str) or not protocol.SHA_RE.fullmatch(value) for value in hashes):
        raise ValidationError("case hashes are invalid")
    if len(set(ids)) != 48 or len(set(hashes)) != 48:
        raise ValidationError("split is not disjoint")
    if set(ids) & protocol.ATTEMPT2_CASE_IDS or set(row.get("context_token_ids_sha256") for row in all_rows) & protocol.ATTEMPT2_CONTEXT_HASHES:
        raise ValidationError("attempt2 artifact entered split")
    if any(row.get("subset") != ("calibration" if row in calibration else "holdout") for row in all_rows):
        raise ValidationError("row subset binding differs")
    for row in all_rows:
        for field in ("fixture_sha256", "prompt_token_ids_sha256", "context_token_ids_sha256"):
            if not isinstance(row.get(field), str) or not protocol.SHA_RE.fullmatch(row[field]):
                raise ValidationError(f"invalid row identity: {field}")
        if row.get("cached_prefix_tokens") != 0 or row.get("prompt_tokens") != row.get("context_tokens") or row.get("generated_tokens") != 0 or row.get("step") != 0 or row.get("row_count") != 1:
            raise ValidationError(f"row full-context/step contract differs: {row.get('case_id')}")
        if row.get("baseline_mode") not in ("all_m1", "cold_batched") or row.get("prompt_tokens") not in (1011, 1024, 1339, 2048):
            raise ValidationError(f"row stratum differs: {row.get('case_id')}")
    strata: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in all_rows:
        strata.setdefault((row["prompt_tokens"], row["baseline_mode"]), []).append(row)
    if len(strata) != 8 or any(len(rows) != 6 for rows in strata.values()):
        raise ValidationError("strata must be eight groups of six")
    if any(sum(row["subset"] == subset for row in rows) != 3 for rows in strata.values() for subset in ("calibration", "holdout")):
        raise ValidationError("each stratum must have 3 calibration and 3 holdout rows")
    for key, rows_in_stratum in strata.items():
        ordered = sorted(rows_in_stratum, key=lambda item: protocol.digest_sort_key(item["case_sha256"]))
        expected_calibration = {item["case_id"] for item in ordered[:3]}
        expected_holdout = {item["case_id"] for item in ordered[3:]}
        actual_calibration = {item["case_id"] for item in rows_in_stratum if item["subset"] == "calibration"}
        actual_holdout = {item["case_id"] for item in rows_in_stratum if item["subset"] == "holdout"}
        if actual_calibration != expected_calibration or actual_holdout != expected_holdout:
            raise ValidationError(f"deterministic assignment differs: {key}")
    if manifest.get("selected_case_count") != 48 or manifest.get("calibration_case_count") != 24 or manifest.get("holdout_case_count") != 24:
        raise ValidationError("manifest counts differ")
    if manifest.get("split_domain") != "ullm.aq4_p2_fidelity_split.v1\\0 + case_sha256" or manifest.get("algorithm") != "sha256(domain || case_sha256), lexicographic within each prompt_tokens/baseline_mode stratum; first 3 calibration, last 3 holdout":
        raise ValidationError("split algorithm is not frozen")
    exclusions = manifest.get("attempt2_exclusions")
    if not isinstance(exclusions, dict) or exclusions.get("case_ids") != sorted(protocol.ATTEMPT2_CASE_IDS) or exclusions.get("context_token_ids_sha256") != sorted(protocol.ATTEMPT2_CONTEXT_HASHES):
        raise ValidationError("attempt2 exclusion contract differs")
    manifest_strata = manifest.get("strata")
    if not isinstance(manifest_strata, list) or len(manifest_strata) != 8:
        raise ValidationError("manifest strata are missing")
    by_key = {(item.get("prompt_tokens"), item.get("baseline_mode")): item for item in manifest_strata if isinstance(item, dict)}
    if len(by_key) != 8:
        raise ValidationError("manifest strata keys differ")
    for key, rows_in_stratum in strata.items():
        item = by_key.get(key)
        if not isinstance(item, dict) or item.get("case_count") != 6 or item.get("calibration_case_ids") != [x["case_id"] for x in sorted(rows_in_stratum, key=lambda x: protocol.digest_sort_key(x["case_sha256"]))[:3]] or item.get("holdout_case_ids") != [x["case_id"] for x in sorted(rows_in_stratum, key=lambda x: protocol.digest_sort_key(x["case_sha256"]))[3:]]:
            raise ValidationError(f"manifest stratum binding differs: {key}")
    if policy.get("attempt2_threshold_source_forbidden") is not True or policy.get("observed_attempt2_values_forbidden") is not True or policy.get("calibration_subset_only_for_active_bf16_envelope") is not True or policy.get("holdout_evaluation_allowed_once") is not True:
        raise ValidationError("policy safety flags are not frozen")
    if not isinstance(policy.get("metrics"), dict) or set(policy["metrics"]) != set(protocol.METRICS):
        raise ValidationError("policy metric set differs")
    for name, expected in protocol.METRICS.items():
        actual = policy["metrics"].get(name)
        if not isinstance(actual, dict) or any(actual.get(field) != expected.get(field) for field in ("direction", "aggregation", "margin", "absolute_floor", "absolute_ceiling", "sample_minimum")):
            raise ValidationError(f"policy metric contract differs: {name}")
    result = {"status": "ok", "calibration": 24, "holdout": 24, "split_manifest_sha256": protocol.sha_bytes(manifest_raw), "policy_sha256": protocol.sha_bytes(policy_raw)}
    if receipt_path is not None:
        receipt, receipt_raw = protocol.load(receipt_path, "freeze receipt")
        if receipt.get("schema_version") != protocol.RECEIPT_SCHEMA or receipt.get("status") != "frozen_calibration_envelope":
            raise ValidationError("freeze receipt schema/status differs")
        if receipt.get("split_manifest_sha256") != result["split_manifest_sha256"] or receipt.get("policy_sha256") != result["policy_sha256"] or receipt.get("calibration_case_count") != 24 or receipt.get("holdout_status") != "not_started" or receipt.get("holdout_evaluations_remaining") != 1:
            raise ValidationError("freeze receipt binding/state differs")
        bounds = receipt.get("derived_bounds")
        if not isinstance(bounds, dict) or set(bounds) != set(protocol.METRICS):
            raise ValidationError("freeze bounds metric set differs")
        for name, spec in protocol.METRICS.items():
            item = bounds[name]
            if not isinstance(item, dict) or item.get("direction") != spec["direction"] or item.get("sample_count") != 24:
                raise ValidationError(f"freeze bound identity differs: {name}")
            for field in ("calibration_mean", "margin", "bound"):
                finite(item.get(field), f"freeze {name}.{field}")
            bound = float(item["bound"])
            if bound < spec["absolute_floor"] if spec["absolute_floor"] is not None else False:
                raise ValidationError(f"freeze bound below floor: {name}")
            if bound > spec["absolute_ceiling"]:
                raise ValidationError(f"freeze bound above ceiling: {name}")
        result["receipt_sha256"] = protocol.sha_bytes(receipt_raw)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(validate(args.split_root, args.receipt), sort_keys=True))
        return 0
    except (ValidationError, protocol.ProtocolError, OSError, ValueError) as error:
        print(f"AQ4 P2 fidelity split validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
