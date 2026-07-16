#!/usr/bin/env python3
"""Build hash-bound SQ8 P2 metrics from one strict 24-row GPU capture."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROTOCOL = _load("qwen35_aq4_sq8_fidelity_protocol_builder", "qwen35_aq4_sq8_fidelity_protocol.py")
VALIDATOR = _load("qwen35_aq4_full_calibration_validator_builder", "validate-qwen35-aq4-p2-full-calibration.py")
COMPARATOR = _load("qwen35_aq4_full_calibration_comparator_builder", "compare-qwen35-aq4-p2-calibration.py")


class MetricsBuildError(ValueError):
    pass


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    value, _ = PROTOCOL.load_json(path, label)
    if not isinstance(value, dict):
        raise MetricsBuildError(f"{label} must be an object")
    return value


def _guard_sha(names: Any) -> str:
    if not isinstance(names, list) or len(names) != 35 or len(set(names)) != 35 or not all(isinstance(name, str) and name for name in names):
        raise MetricsBuildError("served required-environment guard set differs")
    digest = hashlib.sha256(b"ullm-aq4-p2-resident-guards-v1\0")
    for name in sorted(names):
        digest.update(f"{name}=1\n".encode())
    return digest.hexdigest()


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise MetricsBuildError(f"{label} has unknown or missing fields")
    return value


def _binding_tensor_names(binding: dict[str, Any]) -> list[str]:
    names = binding.get("tensor_names")
    canonical = PROTOCOL.SQ8_RUNTIME_TENSOR_NAMES
    if (
        not isinstance(names, list)
        or any(type(name) is not str for name in names)
        or len(names) != len(canonical)
        or len(set(names)) != len(canonical)
        or set(names) != set(canonical)
    ):
        raise MetricsBuildError("SQ8 binding tensor-name set differs")
    return names


def _target_artifact(value: Any, expected_scalars: dict[str, str], binding_names: list[str]) -> dict[str, Any]:
    artifact = _exact_keys(value, set(expected_scalars) | {"tensor_names"}, "SQ8 target artifact identity")
    for field, expected in expected_scalars.items():
        if type(artifact[field]) is not str or artifact[field] != expected:
            raise MetricsBuildError(f"SQ8 target artifact identity differs: {field}")
    names = artifact["tensor_names"]
    canonical = list(PROTOCOL.SQ8_RUNTIME_TENSOR_NAMES)
    if not isinstance(names, list) or any(type(name) is not str for name in names) or names != canonical:
        raise MetricsBuildError("SQ8 target runtime tensor-name order differs")
    if set(names) != set(binding_names):
        raise MetricsBuildError("SQ8 target/binding tensor-name authority differs")
    return artifact


def _bind_target(plan: dict[str, Any], source: dict[str, Any], target: dict[str, Any]) -> None:
    identity = plan["identity"]
    if source["manifest"].get("schema_version") != "ullm.qwen35_aq4_source_calibration.v1" or source["manifest"].get("oracle_kind") != "independent_source_full":
        raise MetricsBuildError("SQ8 source artifact kind differs")
    if target["manifest"].get("schema_version") != "ullm.qwen35_aq4_target_calibration.v1" or target["manifest"].get("oracle_kind") != "aq4_sq8_target":
        raise MetricsBuildError("SQ8 target artifact kind differs")
    target_identity = target["manifest"]["identity"]
    runtime = target["manifest"]["runtime"]["runtime"]
    receipt = _strict_json(Path(identity["sq8_receipt_path"]), "SQ8 actual receipt")
    binding_path = Path(receipt["overlay"]["binding_manifest_path"])
    binding = _strict_json(binding_path, "SQ8 binding manifest")
    if VALIDATOR.sha256_file(binding_path, "SQ8 binding manifest") != receipt["overlay"]["binding_manifest_sha256"]:
        raise MetricsBuildError("SQ8 binding manifest SHA differs from actual receipt")
    binding_names = _binding_tensor_names(binding)
    served_path = Path(identity["served_model"]["path"])
    served = _strict_json(served_path, "SQ8 served model")
    profile_path = Path(receipt["release"]["profile"]["path"])
    profile = _strict_json(profile_path, "SQ8 profile")
    expected_artifact_scalars = {
        "package_manifest_sha256": identity["package"]["manifest_sha256"],
        "artifact_manifest_sha256": receipt["overlay"]["binding_manifest_sha256"],
        "content_sha256": identity["overlay_content_sha256"],
        "tensor_set_sha256": identity["overlay_tensor_set_sha256"],
    }
    _target_artifact(target_identity.get("artifact"), expected_artifact_scalars, binding_names)
    if target_identity.get("format_id") != "AQ4_0" or target_identity.get("implementation_id") != "qwen35_aq4_sq8_linear_qkv_z_overlay_v1":
        raise MetricsBuildError("SQ8 target format/implementation differs")
    if target_identity.get("package_manifest_sha256") != identity["package"]["manifest_sha256"] or target_identity.get("worker_binary_sha256") != identity["worker"]["sha256"]:
        raise MetricsBuildError("SQ8 target package/worker differs")
    source_identity = source["manifest"]["identity"]
    for field in ("model_id", "model_revision", "source_checkpoint", "tokenizer", "hidden_size", "vocab_size"):
        if target_identity.get(field) != source_identity.get(field):
            raise MetricsBuildError(f"SQ8 target source-model identity differs: {field}")
    if runtime.get("name") != "ullm-aq4-sq8-fidelity-capture" or runtime.get("one_model_load") is not True:
        raise MetricsBuildError("SQ8 target capture runtime differs")
    served_sha = VALIDATOR.sha256_file(served_path, "SQ8 served model")
    expected_runtime = {
        "split_manifest_sha256": identity["split_manifest_sha256"],
        "policy_sha256": identity["policy_sha256"],
        "calibration_cases_sha256": identity["calibration_cases_sha256"],
        "served_model_manifest_sha256": served_sha,
        "package_manifest_sha256": identity["package"]["manifest_sha256"],
        "worker_binary_sha256": identity["worker"]["sha256"],
        "guard_sha256": _guard_sha(served["worker"]["required_environment"]),
        "upstream_model_revision": source_identity["model_revision"],
        "quantized_artifact_revision": served["public"]["revision"],
        "source_checkpoint_aggregate_sha256": source_identity["source_checkpoint"]["aggregate_sha256"],
        "tokenizer_aggregate_sha256": source_identity["tokenizer"]["aggregate_sha256"],
    }
    for field, expected in expected_runtime.items():
        if runtime.get(field) != expected:
            raise MetricsBuildError(f"SQ8 target runtime binding differs: {field}")
    device = runtime.get("device")
    if not isinstance(device, dict) or device.get("architecture") != served["worker"]["identity"]["device"] or device.get("architecture") != profile["worker"]["identity"]["device"]:
        raise MetricsBuildError("SQ8 target GPU/profile identity differs")
    if target["manifest"]["cases"].get("path") != source["manifest"]["cases"].get("path") or target["manifest"]["cases"].get("sha256") != source["manifest"]["cases"].get("sha256"):
        raise MetricsBuildError("SQ8 target source cases differ")


def build(plan_path: Path, source_root: Path, target_root: Path, comparison_root: Path, output: Path) -> dict[str, Any]:
    plan, _ = PROTOCOL._check_plan(plan_path)
    if plan["status"] != "ready_for_calibration" or plan["preflight_only"] is not False or plan["identity"]["receipt_state"] != "actual_verified":
        raise MetricsBuildError("actual_verified SQ8 plan is required")
    source_report = VALIDATOR.validate(source_root)
    target_report = VALIDATOR.validate(target_root)
    if source_report["status"] != "valid" or source_report["row_count"] != PROTOCOL.MAX_ROWS or target_report["status"] != "valid" or target_report["row_count"] != PROTOCOL.MAX_ROWS:
        raise MetricsBuildError("SQ8 source/target artifact must contain exactly 24 finite rows")
    source = COMPARATOR.load_artifact(source_root)
    target = COMPARATOR.load_artifact(target_root)
    _bind_target(plan, source, target)
    COMPARATOR.compare(source, target, "sq8_source_gate", comparison_root)
    comparison_manifest = comparison_root / "manifest.json"
    comparison = _strict_json(comparison_manifest, "SQ8 comparison manifest")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in (comparison_root / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(raw)
        rows[(item["case_id"], item["step"])] = item
    case_rows = PROTOCOL.read_jsonl(Path(plan["calibration"]["path"]), "SQ8 calibration cases")
    output_rows = []
    allowed = {"case_id", "case_sha256", "fixture_sha256", "fixture_path", "prompt_token_ids_sha256", "context_token_ids_sha256", "prompt_tokens", "cached_prefix_tokens", "context_tokens", "generated_tokens", "baseline_mode", "prefill_requested_m", "resolved_m", "step", "row_count", "subset"}
    for case in case_rows:
        compared = rows.get((case["case_id"], case["step"]))
        if compared is None:
            raise MetricsBuildError(f"SQ8 comparison row is missing: {case['case_id']}")
        source_row = source["rows"].get((case["case_id"], case["step"]))
        target_row = target["rows"].get((case["case_id"], case["step"]))
        if source_row is None or target_row is None or source_row["input_token_ids_sha256"] != case["context_token_ids_sha256"] or target_row["input_token_ids_sha256"] != case["context_token_ids_sha256"]:
            raise MetricsBuildError(f"SQ8 case token identity differs: {case['case_id']}")
        greedy = compared["greedy"]
        ordered = compared["ordered_top10"]
        metrics = {
            "token_agreement_rate": float(greedy["source"] == greedy["target"]),
            "topk_overlap_rate_k10": float(compared["top_k_overlap"]) / 10.0,
            "logits_cosine": float(compared["logits"]["cosine"]),
            "logits_relative_l2": float(compared["logits"]["relative_l2"]),
            "hidden_cosine": float(compared["hidden"]["cosine"]),
            "hidden_relative_l2": float(compared["hidden"]["relative_l2"]),
            "hidden_max_abs": float(compared["hidden"]["max_abs"]),
            "bf16_top1_retained_in_aq4_top10_rate": float(greedy["source"] in ordered["target"]),
        }
        output_rows.append({**{key: case[key] for key in allowed}, "metrics": metrics})
    metrics_value = {
        "schema_version": PROTOCOL.METRICS_SCHEMA,
        "identity": plan["identity"],
        "subset": "calibration",
        "evidence": {
            "source_artifact": {"root": str(source_root.resolve()), "manifest_sha256": source["manifest_sha256"]},
            "target_artifact": {"root": str(target_root.resolve()), "manifest_sha256": target["manifest_sha256"]},
            "comparison": {"root": str(comparison_root.resolve()), "manifest_sha256": VALIDATOR.sha256_file(comparison_manifest, "SQ8 comparison manifest")},
        },
        "rows": output_rows,
    }
    PROTOCOL._rows(metrics_value, case_rows, plan["identity"])
    PROTOCOL.atomic_json(output, metrics_value)
    return {"status": "ready_for_freeze", "row_count": len(output_rows), "metrics_sha256": PROTOCOL.sha_file(output, "SQ8 metrics"), "comparison_manifest_sha256": VALIDATOR.sha256_file(comparison_manifest, "SQ8 comparison manifest")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(args.plan, args.source, args.target, args.comparison, args.output), sort_keys=True))
        return 0
    except (MetricsBuildError, PROTOCOL.ProtocolError, VALIDATOR.ValidationError, COMPARATOR.ComparisonError, OSError, ValueError, KeyError) as error:
        print(f"SQ8 fidelity metrics build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
