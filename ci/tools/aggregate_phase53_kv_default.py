#!/usr/bin/env python3
"""Fail-closed Phase 53 target decision and runtime-mapping candidate builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError


class ContractError(ValueError):
    """An input cannot be used as Phase 53 evidence."""


TARGETS = ("gfx942:sramecc+:xnack-", "gfx1201", "gfx1030")
ENCODINGS = {
    "gfx942:sramecc+:xnack-": "kv-fp8-e4-block16",
    "gfx1201": "kv-fp8-e4-block16",
    "gfx1030": "kv-fp8-e5-block16",
}
DESCRIPTORS = {
    "gfx942:sramecc+:xnack-": "kv-fp8-e4-block16-v2",
    "gfx1201": "kv-fp8-e4-block16-v2",
    "gfx1030": "kv-fp8-e5-block16-v2",
}
SCALE_RECIPE = "standard-mx-floor-power-v1"
MODEL_FINGERPRINT = "sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae"
METRICS = (
    "perplexity_relative_delta",
    "kld_p99",
    "top1_agreement",
    "task_score_delta",
    "long_context_score_delta",
)
SAMPLE_KEYS = ("perplexity", "kld", "top1", "task", "long-context")
PERFORMANCE_CASES = (
    ("short-odd", "normal", 17, 17, {"warmups": 3, "measured": 10, "context_length": 34, "ignore_eos": False}),
    ("32-32", "normal", 32, 32, {"warmups": 3, "measured": 10, "context_length": 64, "ignore_eos": False}),
    ("prefill-long", "normal", 1024, 128, {"warmups": 3, "measured": 10, "context_length": 1152, "ignore_eos": False}),
    ("decode-long", "normal", 32, 256, {"warmups": 3, "measured": 10, "context_length": 288, "ignore_eos": False}),
    ("long-10001", "normal", 10_001, 2, {"warmups": 3, "measured": 10, "context_length": 10_003, "ignore_eos": False}),
    ("long-100000", "long-running", 100_000, 2, {"warmups": 1, "measured": 3, "context_length": 131_072, "ignore_eos": False}),
    ("decode-20000", "long-running", 32, 20_000, {"warmups": 1, "measured": 3, "context_length": 131_072, "ignore_eos": True}),
)
ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "ci/schema"
CONTRACTS = {
    "kv-cache-default-v1": {
        "policy": "kv-cache-default-v1.schema.json",
        "correctness": "phase53-kv-fp8-block16-evidence-v1.schema.json",
        "quality": "phase53-qwen35-kv-quality-candidate-v1.schema.json",
        "performance/resource": "phase53-performance-resource-evidence-v1.schema.json",
        "correctness_version": "sllm-phase53-kv-fp8-block16-evidence-v1",
        "quality_version": "sllm-phase53-qwen35-kv-quality-candidate-v1",
        "performance_version": "sllm-phase53-performance-resource-evidence-v1",
        "output_version": "v1",
    },
    "kv-cache-default-v2": {
        "policy": "kv-cache-default-v2.schema.json",
        "correctness": "phase53-kv-fp8-block16-evidence-v2.schema.json",
        "quality": "phase53-qwen35-kv-quality-candidate-v2.schema.json",
        "performance/resource": "phase53-performance-resource-evidence-v2.schema.json",
        "correctness_version": "sllm-phase53-kv-fp8-block16-evidence-v2",
        "quality_version": "sllm-phase53-qwen35-kv-quality-candidate-v2",
        "performance_version": "sllm-phase53-performance-resource-evidence-v2",
        "output_version": "v2",
    },
}
_VALIDATORS: dict[str, Draft202012Validator] = {}


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path} is not a JSON object")
    return value, sha256_bytes(raw)


def contract_for_policy(policy: dict[str, Any]) -> dict[str, str]:
    version = policy.get("schema_version")
    require(version in CONTRACTS, f"unsupported policy schema version {version!r}")
    return CONTRACTS[version]


def schema_validator(kind: str, contract: dict[str, str]) -> Draft202012Validator:
    schema_name = contract[kind]
    cached = _VALIDATORS.get(schema_name)
    if cached is not None:
        return cached
    schema_path = SCHEMA_DIR / schema_name
    schema, _ = read_json(schema_path)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise ContractError(f"invalid repository schema {schema_path}: {error.message}") from error
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    _VALIDATORS[schema_name] = validator
    return validator


def validate_input(value: dict[str, Any], kind: str, path: Path, contract: dict[str, str]) -> None:
    errors = sorted(
        schema_validator(kind, contract).iter_errors(value),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if not errors:
        return
    error = errors[0]
    location = "/".join(str(part) for part in error.absolute_path) or "<root>"
    raise ContractError(
        f"{kind} report {path} does not match {contract[kind]} at {location}: {error.message}"
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def finite_number(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is missing or not numeric")
    number = float(value)
    require(math.isfinite(number), f"{label} is non-finite")
    return number


def index_reports(
    paths: Iterable[Path], kind: str, policy_digest: str, contract: dict[str, str]
) -> dict[str, tuple[dict[str, Any], str]]:
    indexed: dict[str, tuple[dict[str, Any], str]] = {}
    for path in paths:
        value, digest = read_json(path)
        validate_input(value, kind, path, contract)
        target = value.get("target")
        require(target in TARGETS, f"{kind} report has unknown target {target!r}")
        require(target not in indexed, f"duplicate {kind} report for {target}")
        require(value.get("encoding") == ENCODINGS[target], f"{kind} encoding mismatch for {target}")
        reported_policy = value.get("policy_sha256")
        if kind == "quality":
            identity = value.get("identity")
            require(isinstance(identity, dict), "quality identity is missing")
            reported_policy = identity.get("policy_sha256")
        if contract["output_version"] == "v2":
            identity = value.get("identity") if kind == "quality" else value
            require(isinstance(identity, dict), f"{kind} revised-recipe identity is missing")
            require(identity.get("descriptor_id") == DESCRIPTORS[target], f"{kind} descriptor mismatch for {target}")
            require(identity.get("scale_recipe") == SCALE_RECIPE, f"{kind} scale recipe mismatch for {target}")
        require(reported_policy == policy_digest, f"{kind} policy digest mismatch for {target}")
        indexed[target] = value, digest
    return indexed


def threshold_pass(metric: str, value: float, threshold: dict[str, Any]) -> bool:
    limit = finite_number(threshold.get("value"), f"threshold {metric}")
    comparison = threshold.get("comparison")
    require(comparison in ("inclusive", "exclusive"), f"threshold {metric} comparison is invalid")
    if metric == "top1_agreement":
        return value >= limit if comparison == "inclusive" else value > limit
    return value <= limit if comparison == "inclusive" else value < limit


def quality_result(
    report: dict[str, Any], thresholds: dict[str, Any], contract: dict[str, str]
) -> tuple[bool, list[str]]:
    require(report.get("schema_version") == contract["quality_version"], "quality schema version mismatch")
    require(report.get("state") in ("PASS", "FAIL"), "quality report state is invalid")
    require(report.get("sequential_residents") is True, "quality residents were not sequential")
    target = report.get("target")
    comparison = report.get("mxfp8_comparison")
    require(isinstance(comparison, dict) and comparison.get("reference_only") is True, "MXFP8 comparison identity differs")
    compare_mxfp8 = target != "gfx942:sramecc+:xnack-"
    if compare_mxfp8:
        expected_mxfp8 = "kv-mxfp8-e5" if target == "gfx1030" else "kv-mxfp8-e4"
        require(comparison == {"status": "complete", "encoding": expected_mxfp8, "reference_only": True}, "MXFP8 comparison identity differs")
        require(report.get("completely_sequential_order") == ["fp16", "block16", "mxfp8"], "quality resident order differs")
    else:
        require(comparison == {
            "status": "unsupported",
            "encoding": None,
            "reference_only": True,
            "reason": "gfx942 OCP MXFP8 is intentionally unsupported because CDNA3 FNUZ element bytes differ",
        }, "gfx942 MXFP8 unsupported disposition differs")
        require(report.get("completely_sequential_order") == ["fp16", "block16"], "quality resident order differs")
    require(isinstance(report.get("selected_count"), int) and report["selected_count"] > 0, "quality selected_count is zero or missing")
    identity = report.get("identity")
    require(isinstance(identity, dict), "quality identity is missing")
    require(identity.get("model_lock_fingerprint") == MODEL_FINGERPRINT, "quality model lock mismatch")
    for field in ("dataset_sha256", "model_lock_sha256", "derived_lock_fingerprint", "derived_lock_sha256", "binary_sha256"):
        require(isinstance(identity.get(field), str) and identity[field].startswith("sha256:") and len(identity[field]) == 71, f"quality {field} is invalid")
    repeats = report.get("repeats")
    require(isinstance(repeats, list) and len(repeats) == 3, "quality requires exactly three repeats")
    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for expected_repeat, repeat in enumerate(repeats, 1):
        require(isinstance(repeat, dict) and repeat.get("repeat") == expected_repeat, "quality repeats are not ordered 1..3")
        require(repeat.get("fp16_released_before_block16") is True, "quality repeat retained FP16 before block16")
        if compare_mxfp8:
            require(repeat.get("block16_released_before_mxfp8") is True and repeat.get("mxfp8_released_after_repeat") is True, "quality repeat did not release triple sequential residents")
            require("block16_released_after_repeat" not in repeat, "three-way repeat used the gfx942 release shape")
        else:
            require(repeat.get("block16_released_after_repeat") is True, "quality repeat retained block16 after the repeat")
            require("block16_released_before_mxfp8" not in repeat and "mxfp8_released_after_repeat" not in repeat and "mxfp8" not in repeat, "gfx942 repeat claimed unsupported MXFP8 execution")
        for candidate_name in (("block16", "mxfp8") if compare_mxfp8 else ("block16",)):
            candidate = repeat.get(candidate_name)
            require(isinstance(candidate, dict), f"quality {candidate_name} metrics missing")
            require(isinstance(candidate.get("selected_count"), int) and candidate["selected_count"] > 0, f"quality {candidate_name} selected_count is zero")
            counts = candidate.get("metric_sample_counts")
            require(isinstance(counts, dict), f"quality {candidate_name} metric sample counts are missing")
            for key in SAMPLE_KEYS:
                require(isinstance(counts.get(key), int) and counts[key] > 0, f"quality {candidate_name} {key} sample count is zero")
            require(candidate.get("fallback_used") is False and candidate.get("all_dispatches_hip") is True, f"quality {candidate_name} used fallback or non-HIP dispatch")
            require(isinstance(candidate.get("hip_dispatches"), int) and candidate["hip_dispatches"] > 0, f"quality {candidate_name} dispatch count is zero")
            for metric in METRICS:
                observed = finite_number(candidate.get(metric), f"quality {candidate_name} {metric}")
                if candidate_name == "block16":
                    values[metric].append(observed)
    reasons: list[str] = []
    policy_names = {
        "perplexity_relative_delta": "perplexity_relative_delta",
        "kld_p99": "kld_p99",
        "top1_agreement": "top1_agreement",
        "task_score_delta": "task_score_delta",
        "long_context_score_delta": "long_context_score_delta",
    }
    for metric, observations in values.items():
        aggregate = statistics.median(observations)
        if not threshold_pass(metric, aggregate, thresholds[policy_names[metric]]):
            reasons.append(f"{metric} threshold failed: median={aggregate}")
    if report.get("state") == "FAIL":
        reasons.append("quality report is FAIL")
    return not reasons, reasons


def correctness_result(
    report: dict[str, Any], contract: dict[str, str]
) -> tuple[bool, bool, bool, list[str]]:
    require(report.get("schema_version") == contract["correctness_version"], "correctness schema version mismatch")
    execution = report.get("execution")
    cleanup = report.get("cleanup")
    require(isinstance(execution, dict) and isinstance(cleanup, dict), "correctness execution/cleanup is missing")
    fallback = execution.get("fallback_used") is False and execution.get("fallback_allowed") is False
    clean = cleanup.get("retryable") == 0 and cleanup.get("durable") == 0 and cleanup.get("terminal_zero") is True
    passed = report.get("state") == "PASS" and execution.get("gpu_execution") is True
    reasons: list[str] = []
    if not passed:
        reasons.append("correctness GPU oracle did not PASS")
    if not fallback:
        reasons.append("fallback contract failed")
    if not clean:
        reasons.append("cleanup contract failed")
    return passed, fallback, clean, reasons


def performance_resource_result(
    report: dict[str, Any], contract: dict[str, str] | None = None
) -> tuple[bool, bool, bool, list[str]]:
    expected_version = (contract or CONTRACTS["kv-cache-default-v1"])["performance_version"]
    require(report.get("schema_version") == expected_version, "performance/resource schema version mismatch")
    require(report.get("producer") == "ci/tools/build_phase53_performance_resource.py", "performance/resource producer identity differs")
    for field in ("binary_sha256", "hbm_observation_sha256"):
        value = report.get(field)
        require(isinstance(value, str) and len(value) == 71 and value.startswith("sha256:"), f"performance/resource {field} is invalid")
    require(report.get("model_lock_fingerprint") == MODEL_FINGERPRINT, "performance/resource model lock differs")
    require(report.get("selected_count") == 7, "performance/resource requires exactly seven rows")
    rows = report.get("rows")
    require(isinstance(rows, list) and len(rows) == 7, "performance/resource row count differs")
    for row_number, (row, expected) in enumerate(zip(rows, PERFORMANCE_CASES), 1):
        case_id, case_class, input_count, output_count, protocol = expected
        require(
            row.get("row") == row_number and row.get("case_id") == case_id and row.get("class") == case_class
            and row.get("input_token_count") == input_count and row.get("requested_output_tokens") == output_count
            and row.get("protocol") == protocol and row.get("execution_order") == ["fp16", "block16"],
            f"performance/resource frozen row {row_number} identity or protocol differs",
        )
        for run_name in ("fp16", "candidate"):
            run = row.get(run_name)
            require(isinstance(run, dict), f"performance/resource {case_id}/{run_name} is missing")
            require(run.get("generated") is True and run.get("hip_only") is True and run.get("fallback_used") is False and run.get("cleanup_empty") is True and run.get("hbm_gtt_settled") is True, f"performance/resource {case_id}/{run_name} is not HIP-only generated clean settled evidence")
            digest = run.get("direct_report_sha256")
            require(isinstance(digest, str) and len(digest) == 71 and digest.startswith("sha256:"), f"performance/resource {case_id}/{run_name} source digest is invalid")
            for field in ("median_e2e_ns", "tokens_per_second"):
                require(finite_number(run.get(field), f"performance {case_id}/{run_name}/{field}") > 0, f"performance {case_id}/{run_name}/{field} is zero")
            for field in ("logical_kv_bytes", "physical_kv_bytes"):
                require(isinstance(run.get(field), int) and run[field] > 0, f"resource {case_id}/{run_name}/{field} is zero")
            require(isinstance(run.get("hbm_peak_delta_bytes"), int) and run["hbm_peak_delta_bytes"] >= 0, f"resource {case_id}/{run_name}/hbm_peak_delta_bytes is invalid")
        ratio = finite_number(row.get("candidate_to_fp16_throughput_ratio"), f"performance {case_id} ratio")
        observed_ratio = row["candidate"]["tokens_per_second"] / row["fp16"]["tokens_per_second"]
        require(math.isclose(ratio, observed_ratio, rel_tol=1e-12, abs_tol=0.0), f"performance {case_id} ratio differs from raw medians")
    memory = report.get("memory")
    decisions = report.get("decisions")
    cleanup = report.get("cleanup")
    require(isinstance(memory, dict) and isinstance(decisions, dict) and isinstance(cleanup, dict), "performance/resource summary missing")
    require(memory.get("fp16_bytes_per_token_head_plane") == 512 and memory.get("candidate_bytes_per_token_head_plane") == 272, "memory logical layout identity differs")
    require(memory.get("logical_reduction_fraction") == 0.46875 and memory.get("physical_measured") is True, "memory reduction was not measured")
    require(isinstance(memory.get("candidate_physical_kv_bytes_max"), int) and memory["candidate_physical_kv_bytes_max"] == max(row["candidate"]["physical_kv_bytes"] for row in rows), "memory physical maximum differs from rows")
    performance = report.get("state") == "PASS" and decisions.get("performance") == "pass"
    resource = report.get("state") == "PASS" and decisions.get("resource") == "pass"
    memory_ok = report.get("state") == "PASS" and decisions.get("memory") == "pass"
    reasons = []
    if not performance:
        reasons.append("performance decision failed")
    if not resource:
        reasons.append("resource decision failed")
    if not memory_ok:
        reasons.append("memory decision failed")
    return performance, resource, memory_ok, reasons


def aggregate_documents(
    policy: dict[str, Any],
    policy_digest: str,
    correctness: dict[str, tuple[dict[str, Any], str]],
    quality: dict[str, tuple[dict[str, Any], str]],
    performance_resource: dict[str, tuple[dict[str, Any], str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = contract_for_policy(policy)
    is_v2 = contract["output_version"] == "v2"
    freeze = policy.get("freeze")
    thresholds = policy.get("thresholds")
    require(isinstance(freeze, dict) and isinstance(thresholds, dict), "policy freeze/thresholds missing")
    dataset = freeze.get("dataset")
    require(isinstance(dataset, dict), "policy dataset missing")
    dataset_digest = dataset.get("digest")
    require(isinstance(dataset_digest, str), "policy dataset digest missing")
    targets = policy.get("targets")
    require(isinstance(targets, list) and [entry.get("target") for entry in targets] == list(TARGETS), "policy target order/scope differs")
    rows: list[dict[str, Any]] = []
    mappings: list[dict[str, str]] = []
    for target in TARGETS:
        row_identity: dict[str, Any] = {
            "target": target,
            "candidate_encoding": ENCODINGS[target],
        }
        if is_v2:
            row_identity.update({"descriptor_id": DESCRIPTORS[target], "scale_recipe": SCALE_RECIPE})
        correctness_item = correctness.get(target)
        quality_item = quality.get(target)
        performance_item = performance_resource.get(target)
        if correctness_item is None or quality_item is None:
            missing = []
            if correctness_item is None:
                missing.append("correctness report missing")
            if quality_item is None:
                missing.append("quality report missing")
            if performance_item is None:
                missing.append("performance/resource report missing")
            rows.append({
                **row_identity,
                "correctness": "insufficient-evidence", "quality": "insufficient-evidence",
                "performance": "insufficient-evidence", "resource": "insufficient-evidence", "memory": "insufficient-evidence",
                "fallback": "insufficient-evidence", "cleanup": "insufficient-evidence",
                "decision": "insufficient-evidence", "reasons": missing,
                "quality_report_sha256": quality_item[1] if quality_item else None,
                "correctness_report_sha256": correctness_item[1] if correctness_item else None,
                "performance_resource_report_sha256": performance_item[1] if performance_item else None,
            })
            continue
        correct_ok, fallback_ok, cleanup_ok, reasons = correctness_result(correctness_item[0], contract)
        quality_identity = quality_item[0]["identity"]
        require(quality_identity.get("dataset_sha256") == dataset_digest, f"quality dataset digest mismatch for {target}")
        quality_ok, quality_reasons = quality_result(quality_item[0], thresholds, contract)
        reasons.extend(quality_reasons)
        if performance_item is None:
            if correct_ok and quality_ok and fallback_ok and cleanup_ok:
                decision = "insufficient-evidence"
                reasons.append("performance/resource report missing")
            else:
                decision = "retain-fp16"
                reasons.append("performance/resource measurement skipped after correctness or quality failure")
            rows.append({
                **row_identity,
                "correctness": "pass" if correct_ok else "fail", "quality": "pass" if quality_ok else "fail",
                "performance": "insufficient-evidence", "resource": "insufficient-evidence", "memory": "insufficient-evidence",
                "fallback": "pass" if fallback_ok else "fail", "cleanup": "pass" if cleanup_ok else "fail",
                "decision": decision, "reasons": reasons,
                "quality_report_sha256": quality_item[1], "correctness_report_sha256": correctness_item[1],
                "performance_resource_report_sha256": None,
            })
            continue
        performance_ok, resource_ok, memory_ok, performance_reasons = performance_resource_result(performance_item[0], contract)
        reasons.extend(performance_reasons)
        decision = "adopt" if correct_ok and quality_ok and performance_ok and resource_ok and memory_ok and fallback_ok and cleanup_ok else "retain-fp16"
        rows.append({
            **row_identity,
            "correctness": "pass" if correct_ok else "fail", "quality": "pass" if quality_ok else "fail",
            "performance": "pass" if performance_ok else "fail", "resource": "pass" if resource_ok else "fail", "memory": "pass" if memory_ok else "fail",
            "fallback": "pass" if fallback_ok else "fail", "cleanup": "pass" if cleanup_ok else "fail",
            "decision": decision, "reasons": reasons,
            "quality_report_sha256": quality_item[1], "correctness_report_sha256": correctness_item[1],
            "performance_resource_report_sha256": performance_item[1],
        })
        if decision == "adopt":
            mapping = {"target": target, "encoding": ENCODINGS[target], "model_lock_fingerprint": MODEL_FINGERPRINT}
            if is_v2:
                mapping.update({"descriptor_id": DESCRIPTORS[target], "scale_recipe": SCALE_RECIPE})
            mappings.append(mapping)
    output_version = contract["output_version"]
    summary = {
        "$schema": f"https://sllm.dev/schema/phase53-kv-default-summary-{output_version}.schema.json",
        "schema_version": f"sllm-phase53-kv-default-summary-{output_version}", "state": "PASS",
        "policy_sha256": policy_digest, "dataset_sha256": dataset_digest, "targets": rows,
    }
    mapping = {
        "$schema": f"https://sllm.dev/schema/phase53-runtime-mapping-candidate-{output_version}.schema.json",
        "schema_version": f"sllm-phase53-runtime-mapping-candidate-{output_version}", "status": "candidate-not-runtime-policy",
        "policy_sha256": policy_digest, "safety_default": "fp16",
        "promotion_scope": "first-promotion-qwen35-4b-bf16-dense-text-full-attention-single-gpu", "mappings": mappings,
    }
    return summary, mapping


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def aggregate(policy_path: Path, correctness_paths: Iterable[Path], quality_paths: Iterable[Path], performance_resource_paths: Iterable[Path] = ()) -> tuple[dict[str, Any], dict[str, Any]]:
    policy, policy_digest = read_json(policy_path)
    contract = contract_for_policy(policy)
    validate_input(policy, "policy", policy_path, contract)
    correctness = index_reports(correctness_paths, "correctness", policy_digest, contract)
    quality = index_reports(quality_paths, "quality", policy_digest, contract)
    performance_resource = index_reports(performance_resource_paths, "performance/resource", policy_digest, contract)
    return aggregate_documents(policy, policy_digest, correctness, quality, performance_resource)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, action="append", default=[])
    parser.add_argument("--quality", type=Path, action="append", default=[])
    parser.add_argument("--performance-resource", type=Path, action="append", default=[])
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--runtime-mapping", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary, mapping = aggregate(args.policy, args.correctness, args.quality, args.performance_resource)
        atomic_write_json(args.summary, summary)
        atomic_write_json(args.runtime_mapping, mapping)
    except (ContractError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
