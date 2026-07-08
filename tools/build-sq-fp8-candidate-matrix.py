#!/usr/bin/env python3
"""Build the R9700 SQ FP8 format candidate matrix from the current policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "sq-fp8-format-candidate-matrix-v0.1"
POLICY_SCHEMA_VERSION = "sq-fp8-policy-v0.1"
ARTIFACT_RESULT_SCHEMA_VERSION = "sq-fp8-policy-artifact-result-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-json", required=True, type=Path)
    parser.add_argument("--artifact-result-json", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--date", required=True)
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_policy(policy: dict[str, Any], path: Path) -> None:
    schema_version = policy.get("schema_version")
    if schema_version != POLICY_SCHEMA_VERSION:
        raise SystemExit(
            f"{path}: policy schema_version must be {POLICY_SCHEMA_VERSION!r}, got {schema_version!r}"
        )
    if not isinstance(policy.get("fp8_selection"), dict):
        raise SystemExit(f"{path}: policy fp8_selection must be an object")
    if not isinstance(policy.get("scale"), dict):
        raise SystemExit(f"{path}: policy scale must be an object")


def validate_artifact_result(artifact: dict[str, Any], path: Path) -> None:
    schema_version = artifact.get("schema_version")
    if schema_version != ARTIFACT_RESULT_SCHEMA_VERSION:
        raise SystemExit(
            f"{path}: artifact result schema_version must be "
            f"{ARTIFACT_RESULT_SCHEMA_VERSION!r}, got {schema_version!r}"
        )
    if not isinstance(artifact.get("storage"), dict):
        raise SystemExit(f"{path}: artifact result storage must be an object")


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def storage_value(storage: dict[str, Any], key: str) -> int | None:
    return int_or_none(storage.get(key))


def scale_estimate_from_artifact(
    artifact: dict[str, Any] | None,
    *,
    scale_dtype: str,
    divisor: int,
) -> dict[str, Any]:
    if artifact is None:
        return {
            "source": "not_measured",
            "scale_dtype": scale_dtype,
            "estimated_fp8_scale_bytes": None,
            "estimated_compact_resident_bytes": None,
            "note": "artifact result not provided",
        }
    storage = artifact["storage"]
    current_scale_bytes = storage_value(storage, "fp8_scale_bytes")
    current_compact_bytes = storage_value(storage, "compact_resident_bytes_estimate")
    if current_scale_bytes is None or current_compact_bytes is None:
        return {
            "source": "not_measured",
            "scale_dtype": scale_dtype,
            "estimated_fp8_scale_bytes": None,
            "estimated_compact_resident_bytes": None,
            "note": "artifact result is missing scale or compact byte estimates",
        }
    estimated_scale_bytes = current_scale_bytes // divisor
    estimated_compact_bytes = current_compact_bytes - current_scale_bytes + estimated_scale_bytes
    return {
        "source": "estimated_from_current_policy_artifact",
        "scale_dtype": scale_dtype,
        "estimated_fp8_scale_bytes": estimated_scale_bytes,
        "estimated_compact_resident_bytes": estimated_compact_bytes,
        "delta_compact_resident_bytes_vs_current": estimated_compact_bytes - current_compact_bytes,
        "note": "estimate changes only scale bytes for the current partial policy tensor set",
    }


def current_storage(artifact: dict[str, Any] | None) -> dict[str, Any]:
    if artifact is None:
        return {
            "source": "not_measured",
            "note": "run build-sq-fp8-w8a16-artifact.py and materialize smoke first",
        }
    storage = artifact["storage"]
    keys = [
        "artifact_disk_usage_bytes_approx",
        "artifact_file_count",
        "fp8_tensor_count",
        "passthrough_tensor_count",
        "fp8_payload_bytes",
        "fp8_scale_bytes",
        "passthrough_source_bytes_estimate",
        "compact_resident_bytes_estimate",
        "materialized_working_set_bytes_estimate",
    ]
    return {
        "source": "measured_policy_artifact_result",
        **{key: storage.get(key) for key in keys},
        "artifact_dir": artifact.get("artifact_dir"),
        "artifact_manifest": artifact.get("artifact_manifest"),
    }


def common_quality_gate(policy: dict[str, Any]) -> dict[str, Any]:
    acceptance = policy.get("acceptance_rule")
    prompt_bundle = policy.get("prompt_bundle_result")
    return {
        "promotion_rule": "strict_top1",
        "diagnostic_only": [
            "topk_common",
            "baseline_top1_rank_in_sq_topk",
            "sq_top1_minus_baseline_top1_logit",
            "short_text_generation_health",
        ],
        "source_policy_acceptance_rule": acceptance if isinstance(acceptance, dict) else None,
        "source_prompt_bundle_result": prompt_bundle if isinstance(prompt_bundle, dict) else None,
    }


def common_throughput_gate() -> dict[str, Any]:
    return {
        "required_metrics": [
            "prefill_total_input_tps",
            "decode_total_generated_tps",
            "end_to_end_total_tps",
            "vram_peak_bytes",
            "compact_resident_bytes",
            "materialized_working_set_bytes",
        ],
        "comparison_baseline": "AQ4 latest R9700 baseline in the same workload grid",
        "overlay_load_timing_is_speed_result": False,
        "allowed_speed_paths": [
            "native_fp8_runtime",
            "materialization_aware_runtime_path",
            "explicit_materialized_working_set_path",
        ],
    }


def candidate(
    *,
    candidate_id: str,
    status: str,
    priority: int,
    purpose: str,
    weight_payload_dtype: str,
    activation_dtype: str,
    scale: dict[str, Any],
    fp8_selection: dict[str, Any],
    fallback_policy: Any,
    storage: dict[str, Any],
    implementation: dict[str, Any],
    risks: list[str],
    next_action: str,
    quality_gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "status": status,
        "priority": priority,
        "purpose": purpose,
        "weight_payload_dtype": weight_payload_dtype,
        "activation_dtype": activation_dtype,
        "scale": scale,
        "fp8_selection": fp8_selection,
        "fallback_policy": fallback_policy,
        "quality_gate": quality_gate,
        "throughput_gate": common_throughput_gate(),
        "storage": storage,
        "implementation": implementation,
        "risks": risks,
        "next_action": next_action,
    }


def build_matrix(
    *,
    policy: dict[str, Any],
    policy_json: Path,
    artifact: dict[str, Any] | None,
    artifact_result_json: Path | None,
    date: str,
) -> dict[str, Any]:
    fp8_selection = policy["fp8_selection"]
    fallback_policy = policy.get("fallback_policy", [])
    policy_scale = policy["scale"]
    quality_gate = common_quality_gate(policy)
    scale_block_cols = policy_scale.get("block_cols")
    scale_granularity = policy_scale.get("granularity")

    current_scale = {
        "granularity": scale_granularity,
        "block_cols": scale_block_cols,
        "dtype": policy_scale.get("dtype", "f32"),
    }
    scale16 = {**current_scale, "dtype": "fp16_or_bf16"}
    scale8 = {**current_scale, "dtype": "fp8_e4m3_or_e5m2"}

    candidates = [
        candidate(
            candidate_id="sq-fp8-w8a16-r9700-v0",
            status="current_regression_subset_not_full_policy",
            priority=1,
            purpose="baseline FP8 weight candidate with F32 row-block scale",
            weight_payload_dtype="fp8_e4m3",
            activation_dtype="bf16_or_f32",
            scale=current_scale,
            fp8_selection=fp8_selection,
            fallback_policy=fallback_policy,
            storage=current_storage(artifact),
            implementation={
                "artifact_builder_support": "supported",
                "runtime_support": "materialize_smoke_supported_overlay_quality_partial",
                "native_fp8_compute": "not_yet_connected_for_package_throughput",
            },
            risks=[
                "current policy covers only a six-layer regression subset",
                "case_a top8 overlap is low even though strict top1 passes",
            ],
            next_action="broaden quality coverage or connect this policy to selected-layer throughput guard",
            quality_gate=quality_gate,
        ),
        candidate(
            candidate_id="sq-fp8-w8a16-r9700-v1-scale16",
            status="planned_experiment",
            priority=2,
            purpose="reduce F32 scale bytes while keeping W8A16 compute shape",
            weight_payload_dtype="fp8_e4m3",
            activation_dtype="bf16_or_f32",
            scale=scale16,
            fp8_selection=fp8_selection,
            fallback_policy=fallback_policy,
            storage=scale_estimate_from_artifact(artifact, scale_dtype="fp16_or_bf16", divisor=2),
            implementation={
                "artifact_builder_support": "missing_scale_dtype_option",
                "runtime_support": "missing_scale16_materialization",
                "native_fp8_compute": "not_yet_connected",
            },
            risks=[
                "scale quantization may move top1 for the current tight guard",
                "scale conversion overhead must not dominate decode",
            ],
            next_action="add scale dtype option to artifact builder and rerun strict top1 prompt bundle",
            quality_gate=quality_gate,
        ),
        candidate(
            candidate_id="sq-fp8-w8a16-r9700-v1-scale8",
            status="planned_risk_probe",
            priority=3,
            purpose="test whether FP8 scale is viable enough to reduce scale bandwidth",
            weight_payload_dtype="fp8_e4m3",
            activation_dtype="bf16_or_f32",
            scale=scale8,
            fp8_selection=fp8_selection,
            fallback_policy=fallback_policy,
            storage=scale_estimate_from_artifact(artifact, scale_dtype="fp8_e4m3_or_e5m2", divisor=4),
            implementation={
                "artifact_builder_support": "missing_scale_dtype_option",
                "runtime_support": "missing_scale8_materialization",
                "native_fp8_compute": "not_yet_connected",
            },
            risks=[
                "scale quantization may be too coarse for strict top1",
                "FP8 value plus FP8 scale may need separate raw-value and scale math",
            ],
            next_action="only run after scale16 clarifies quality and runtime overhead",
            quality_gate=quality_gate,
        ),
        candidate(
            candidate_id="sq-fp8-w8a8-r9700-v0",
            status="planned_after_w8a16_baseline",
            priority=4,
            purpose="measure whether R9700 native FP8 activation path improves total throughput",
            weight_payload_dtype="fp8_e4m3",
            activation_dtype="fp8_e4m3",
            scale=current_scale,
            fp8_selection=fp8_selection,
            fallback_policy=fallback_policy,
            storage={
                "source": "weight_storage_estimated_from_current_policy_artifact",
                "weight_storage": current_storage(artifact),
                "activation_storage": "not_measured",
                "note": "W8A8 requires runtime activation quantization metadata beyond the weight artifact",
            },
            implementation={
                "artifact_builder_support": "partial_weight_payload_only",
                "runtime_support": "missing_activation_fp8_path",
                "native_fp8_compute": "planned_rdna4_only",
            },
            risks=[
                "activation FP8 can break quality even if weight-only guard passes",
                "full FP8 compute path may expose unsupported tensor shapes",
            ],
            next_action="defer until W8A16 quality guard and selected-layer throughput path are stable",
            quality_gate=quality_gate,
        ),
        candidate(
            candidate_id="sq-fp8-hybrid-r9700-v0",
            status="planned_conservative_policy_family",
            priority=5,
            purpose="increase FP8 coverage only where strict top1 remains stable and fallback risky families",
            weight_payload_dtype="fp8_e4m3_plus_fallback",
            activation_dtype="bf16_or_f32",
            scale=current_scale,
            fp8_selection=fp8_selection,
            fallback_policy=fallback_policy,
            storage={
                "source": "policy_dependent",
                "current_policy_storage": current_storage(artifact),
                "note": "storage must be regenerated for each expanded hybrid policy",
            },
            implementation={
                "artifact_builder_support": "supported_for_policy_json",
                "runtime_support": "materialize_smoke_supported_overlay_quality_partial",
                "native_fp8_compute": "not_yet_connected_for_package_throughput",
            },
            risks=[
                "too much fallback can erase memory and throughput benefit",
                "per-layer policy can overfit short guards",
            ],
            next_action="use as fallback direction if broader W8A16 strict top1 fails",
            quality_gate=quality_gate,
        ),
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "date": date,
        "target": {
            "gpu": "R9700/RDNA4",
            "runtime_device_index": 2,
            "scope": "single_gpu_sq_candidate_evaluation",
        },
        "source_policy": {
            "path": str(policy_json),
            "policy_id": policy.get("policy_id"),
            "candidate_id": policy.get("candidate_id"),
            "status": policy.get("status"),
        },
        "source_artifact_result": {
            "path": str(artifact_result_json) if artifact_result_json is not None else None,
            "schema_version": artifact.get("schema_version") if artifact else None,
            "artifact_dir": artifact.get("artifact_dir") if artifact else None,
        },
        "decision_rules": {
            "quality_promotion_rule": "strict_top1",
            "diagnostic_fields_do_not_override_strict_top1": True,
            "overlay_load_timing_is_speed_result": False,
            "full_package_real_batch_required_for_final_comparison": True,
            "full_package_real_batch_blocks_candidate_exploration": False,
        },
        "measurement_contract": {
            "required_throughput_metrics": [
                "prefill_total_input_tps",
                "decode_total_generated_tps",
                "end_to_end_total_tps",
            ],
            "required_memory_metrics": [
                "vram_peak_bytes",
                "compact_resident_bytes",
                "materialized_working_set_bytes",
            ],
            "allowed_intermediate_paths": [
                "component",
                "selected_layer_stack",
                "materialization_aware_runtime_path",
            ],
        },
        "candidates": candidates,
        "next_steps": [
            "Generate and track candidate artifacts from this matrix.",
            "Run strict top1 prompt bundle before treating any candidate as promoted.",
            "Connect selected-layer stack to token-id embedding plus final norm/lm_head quality guard.",
            "Continue T1a full-package real batch runner for final AQ4/SQ comparison rows.",
        ],
    }


def bytes_to_gib(value: Any) -> str:
    raw = int_or_none(value)
    if raw is None:
        return "n/a"
    return f"{raw / (1024 ** 3):.3f}"


def write_markdown(path: Path, matrix: dict[str, Any]) -> None:
    rows = []
    for item in matrix["candidates"]:
        storage = item["storage"]
        compact = storage.get("compact_resident_bytes_estimate")
        if compact is None:
            compact = storage.get("estimated_compact_resident_bytes")
        if compact is None and isinstance(storage.get("current_policy_storage"), dict):
            compact = storage["current_policy_storage"].get("compact_resident_bytes_estimate")
        if compact is None and isinstance(storage.get("weight_storage"), dict):
            compact = storage["weight_storage"].get("compact_resident_bytes_estimate")
        rows.append(
            "| {candidate_id} | {status} | {scale_dtype} | {activation_dtype} | {compact_gib} | {next_action} |".format(
                candidate_id=item["candidate_id"],
                status=item["status"],
                scale_dtype=item["scale"].get("dtype"),
                activation_dtype=item["activation_dtype"],
                compact_gib=bytes_to_gib(compact),
                next_action=item["next_action"],
            )
        )

    content = "\n".join(
        [
            "# SQ FP8 Format Candidate Matrix v0.1",
            "",
            "## 前回の要点",
            "",
            "- `kup6_gate5_down5` は6層strict-top1 regression subsetであり、full SQ policyではない。",
            "- 現在の実FP8 payload artifactはmaterialize smokeまで確認済みだが、throughput結果ではない。",
            "",
            "## 今回の変更点",
            "",
            "- SQ format候補を機械可読matrixとして固定した。",
            "- full-package real batch runnerは最終比較には必要だが、候補探索の開始blockerにはしない。",
            "- overlay host materialize/load timingを速度結果として使わない方針をmatrixにも入れた。",
            "",
            "## Candidate Matrix",
            "",
            "| candidate | status | scale dtype | activation | compact resident GiB | next action |",
            "| --- | --- | --- | --- | ---: | --- |",
            *rows,
            "",
            "## 次の行動",
            "",
            "1. このmatrixから候補artifactを生成し、strict top1 prompt bundleを通す。",
            "2. selected-layer stackへtoken-id embedding、final norm/lm_head、quality guardを接続する。",
            "3. T1aのfull-package real batch runnerを継続し、最終AQ4/SQ比較行を作る。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    policy = load_json_object(args.policy_json)
    validate_policy(policy, args.policy_json)

    artifact = None
    if args.artifact_result_json is not None:
        artifact = load_json_object(args.artifact_result_json)
        validate_artifact_result(artifact, args.artifact_result_json)

    matrix = build_matrix(
        policy=policy,
        policy_json=args.policy_json,
        artifact=artifact,
        artifact_result_json=args.artifact_result_json,
        date=args.date,
    )
    write_json(args.output_json, matrix)
    if args.output_md is not None:
        write_markdown(args.output_md, matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
