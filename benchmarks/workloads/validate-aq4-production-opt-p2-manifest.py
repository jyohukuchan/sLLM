#!/usr/bin/env python3
"""CPU-only structural validator for the AQ4 P2 planning manifest and policy.

This validator expands no cases, starts no worker, and does not access a GPU or
network.  It intentionally validates only the planning contract; execution and
promotion remain gated by the independent P1 validators.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any


EXPECTED_PROMPTS = [1, 8, 32, 128, 512, 1011, 1024, 1339, 2048, 3584, 4096]
EXPECTED_M = [1, 8, 16, 32, 64, 128]
EXPECTED_MODES = ["all_m1", "cold_batched", "cached_prefix_chunked"]
EXPECTED_DECODE_CONTEXTS = [16, 128, 512, 1024, 1339, 2048, 3584]
EXPECTED_SCOPES = ["component", "full_model", "production_server"]
EXPECTED_STAGES = ["smoke", "representative", "full"]
REQUIRED_CONTROLS = {
    "aq4_0_target",
    "sq8_0_cross_format",
    "reference_source_oracle",
}


class ValidationError(Exception):
    pass


def reject_constant(value: str) -> Any:
    raise ValidationError(f"non-finite JSON constant: {value}")


def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"not a regular file: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=no_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ValidationError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"root must be an object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def ensure_unique_ints(values: Any, name: str) -> list[int]:
    require(isinstance(values, list), f"{name} must be an array")
    require(all(isinstance(value, int) and not isinstance(value, bool) for value in values), f"{name} must contain integers")
    require(len(values) == len(set(values)), f"{name} contains duplicate values")
    return values


def count_prefill(stage: dict[str, Any], device_id: str, production: bool) -> int:
    selection = stage["prefill"]
    prompts = selection.get("production_server_prompt_tokens", []) if production else selection["prompt_tokens"]
    scopes = ["production_server"] if production else ["component", "full_model"]
    controls = stage.get("controls_by_device", {}).get(device_id, stage.get("controls", []))
    return len(prompts) * len(selection["requested_m"]) * len(selection["modes"]) * len(scopes) * len(controls)


def count_decode(stage: dict[str, Any], device_id: str, production: bool) -> int:
    selection = stage["decode"]
    contexts = selection.get("production_server_start_context_tokens", []) if production else selection["start_context_tokens"]
    scopes = ["production_server"] if production else ["component", "full_model"]
    controls = stage.get("controls_by_device", {}).get(device_id, stage.get("controls", []))
    return len(contexts) * len(scopes) * len(controls)


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    require(manifest.get("schema_version") == "ullm.aq4_production_p2_case_manifest.v1", "unexpected manifest schema")
    require(manifest.get("status") == "planning_only", "manifest must remain planning_only")
    require(manifest.get("identity_binding", {}).get("context_limit_tokens") == 4096, "context limit must be 4096")

    safety = manifest.get("execution_safety", {})
    require(safety.get("gpu_processes_at_once") == 1, "GPU process limit must be one")
    require(safety.get("r9700_exclusive") is True, "R9700 must be exclusive")
    require(safety.get("preflight_required") is True, "preflight must be required")
    require(safety.get("stream_tensors_and_logits") is True, "tensor/logit streaming must be enabled")
    require(safety.get("retain_full_logit_matrix") is False, "full logits retention is unsafe")
    require(safety.get("retain_full_attention_matrix") is False, "full attention retention is unsafe")
    require(safety.get("live_requests_allowed_during_preparation") is False, "live requests must remain disabled")

    axes = manifest.get("axes", {})
    require(ensure_unique_ints(axes.get("prefill_prompt_tokens"), "prefill_prompt_tokens") == EXPECTED_PROMPTS, "prefill prompt axis mismatch")
    require(ensure_unique_ints(axes.get("prefill_requested_m"), "prefill_requested_m") == EXPECTED_M, "prefill M axis mismatch")
    require(axes.get("prefill_modes") == EXPECTED_MODES, "prefill mode axis mismatch")
    require(ensure_unique_ints(axes.get("decode_start_context_tokens"), "decode_start_context_tokens") == EXPECTED_DECODE_CONTEXTS, "decode context axis mismatch")
    require(axes.get("decode_requested_m") == 1, "decode M must be one")
    require(axes.get("decode_generated_tokens") == 64, "decode generated token count must be 64")
    require(axes.get("scopes") == EXPECTED_SCOPES, "scope axis mismatch")
    require({device.get("device_id") for device in axes.get("devices", [])} == {"cpu-reference", "r9700-rdna4"}, "CPU and R9700 devices are required")

    controls = axes.get("controls", [])
    control_ids = {control.get("control_id") for control in controls}
    require(control_ids == REQUIRED_CONTROLS, "AQ4, SQ8, and reference controls are required")
    control_by_id = {control["control_id"]: control for control in controls}
    require(control_by_id["aq4_0_target"].get("promotion_eligible") is True, "AQ4 target must be promotion eligible")
    require(control_by_id["sq8_0_cross_format"].get("promotion_eligible") is False, "SQ8 control must not promote")
    require(control_by_id["reference_source_oracle"].get("promotion_eligible") is False, "reference control must not promote")
    dependencies = manifest.get("control_dependencies", {})
    require(dependencies.get("target_control_id") == "aq4_0_target", "target dependency is not AQ4")
    require(set(dependencies.get("required_control_ids", [])) == {"sq8_0_cross_format", "reference_source_oracle"}, "AQ4 control dependencies are incomplete")
    require(dependencies.get("missing_control_action") == "ineligible", "missing controls must fail closed")

    stages = manifest.get("stages", [])
    require([stage.get("stage_id") for stage in stages] == EXPECTED_STAGES, "stage order must be smoke, representative, full")
    totals: dict[str, int] = {}
    for stage in stages:
        require(stage.get("execution_allowed_now") is False, f"stage {stage.get('stage_id')} cannot execute during preparation")
        require(stage.get("devices"), f"stage {stage.get('stage_id')} has no devices")
        require(set(stage.get("controls", [])) >= REQUIRED_CONTROLS or stage.get("controls_by_device"), f"stage {stage.get('stage_id')} lacks controls")
        prefill = stage["prefill"]
        decode = stage["decode"]
        ensure_unique_ints(prefill["prompt_tokens"], f"{stage['stage_id']}.prefill.prompt_tokens")
        ensure_unique_ints(prefill["requested_m"], f"{stage['stage_id']}.prefill.requested_m")
        require(set(prefill["requested_m"]) <= set(EXPECTED_M), f"{stage['stage_id']} uses unsupported M")
        require(set(prefill["modes"]) <= set(EXPECTED_MODES), f"{stage['stage_id']} uses unsupported mode")
        require(set(prefill.get("scopes", [])) <= set(EXPECTED_SCOPES), f"{stage['stage_id']} uses unsupported scope")
        require(decode["requested_m"] == 1 and decode["generated_tokens"] == 64, f"{stage['stage_id']} decode contract mismatch")
        require(set(decode["start_context_tokens"]) <= set(EXPECTED_DECODE_CONTEXTS), f"{stage['stage_id']} uses unsupported decode context")
        for production_prompts in [prefill.get("production_server_prompt_tokens", [])]:
            require(set(production_prompts) <= set(prefill["prompt_tokens"]), f"{stage['stage_id']} production prompt is outside stage axis")
        for production_contexts in [decode.get("production_server_start_context_tokens", [])]:
            require(set(production_contexts) <= set(decode["start_context_tokens"]), f"{stage['stage_id']} production context is outside stage axis")
        for device_id in stage["devices"]:
            controls_for_device = stage.get("controls_by_device", {}).get(device_id, stage.get("controls", []))
            require(set(controls_for_device) <= REQUIRED_CONTROLS, f"{stage['stage_id']} has unknown control for {device_id}")
            require(controls_for_device, f"{stage['stage_id']} has no controls for {device_id}")
            totals[f"{stage['stage_id']}.{device_id}.prefill_non_server"] = count_prefill(stage, device_id, False)
            totals[f"{stage['stage_id']}.{device_id}.prefill_production_server"] = count_prefill(stage, device_id, True)
            totals[f"{stage['stage_id']}.{device_id}.decode_non_server"] = count_decode(stage, device_id, False)
            totals[f"{stage['stage_id']}.{device_id}.decode_production_server"] = count_decode(stage, device_id, True)
        expected = stage.get("expected_case_count", {})
        if stage["stage_id"] == "smoke":
            require(expected.get("prefill") == totals["smoke.cpu-reference.prefill_non_server"], "smoke prefill count mismatch")
            require(expected.get("decode") == totals["smoke.cpu-reference.decode_non_server"], "smoke decode count mismatch")
            require(expected.get("total") == expected["prefill"] + expected["decode"], "smoke total count mismatch")
        elif stage["stage_id"] == "representative":
            expected_keys = {
                "prefill_cpu_non_server": totals["representative.cpu-reference.prefill_non_server"],
                "prefill_cpu_production_server": totals["representative.cpu-reference.prefill_production_server"],
                "prefill_r9700_non_server": totals["representative.r9700-rdna4.prefill_non_server"],
                "prefill_r9700_production_server": totals["representative.r9700-rdna4.prefill_production_server"],
                "decode_cpu_non_server": totals["representative.cpu-reference.decode_non_server"],
                "decode_cpu_production_server": totals["representative.cpu-reference.decode_production_server"],
                "decode_r9700_non_server": totals["representative.r9700-rdna4.decode_non_server"],
                "decode_r9700_production_server": totals["representative.r9700-rdna4.decode_production_server"],
            }
            for key, actual in expected_keys.items():
                require(expected.get(key) == actual, f"representative {key} count mismatch")
            require(expected.get("total") == sum(expected_keys.values()), "representative total count mismatch")
        else:
            expected_keys = {
                "prefill_cpu": totals["full.cpu-reference.prefill_non_server"] + totals["full.cpu-reference.prefill_production_server"],
                "prefill_r9700": totals["full.r9700-rdna4.prefill_non_server"] + totals["full.r9700-rdna4.prefill_production_server"],
                "decode_cpu": totals["full.cpu-reference.decode_non_server"] + totals["full.cpu-reference.decode_production_server"],
                "decode_r9700": totals["full.r9700-rdna4.decode_non_server"] + totals["full.r9700-rdna4.decode_production_server"],
            }
            for key, actual in expected_keys.items():
                require(expected.get(key) == actual, f"full {key} count mismatch")
            require(expected.get("total") == sum(expected_keys.values()), "full total count mismatch")
        require(4096 in EXPECTED_PROMPTS, "context edge prompt 4096 is required")
        require(all(context + 64 <= 4096 for context in EXPECTED_DECODE_CONTEXTS), "decode context plus 64 exceeds context limit")

    queue = manifest.get("r9700_queue", {})
    require(queue.get("max_concurrent_cases") == 1, "R9700 queue must be serial")
    require(queue.get("order") == ["preflight", "smoke", "representative", "full"], "R9700 queue order mismatch")
    require(queue.get("lock_release_on_failure") is True, "R9700 lock must release on failure")
    require(queue.get("other_positive_vram_process_action") == "skip_and_record_busy", "busy R9700 action must be recorded")

    return {"valid": True, "stage_counts": totals}


def validate_policy(policy: dict[str, Any]) -> None:
    require(policy.get("schema_version") == "ullm.aq4_production_p2_threshold_policy.v1", "unexpected policy schema")
    require(policy.get("status") == "unbound_template", "policy must remain an unbound template")
    require(policy.get("hash_binding", {}).get("binding_required_before_measurement") is True, "policy hash binding is not required")
    require(policy.get("applies_to", {}).get("target_format_id") == "AQ4_0", "policy target format must be AQ4_0")
    require(policy.get("applies_to", {}).get("target_gpu_name") == "Radeon AI PRO R9700", "policy target GPU must be R9700")
    require(policy.get("power_condition", {}).get("max_concurrent_gpu_cases") == 1, "policy power queue must be serial")
    require(policy.get("power_condition", {}).get("binding_required") is True, "power condition binding is not required")
    prefill = policy.get("performance_thresholds", {}).get("prefill", {})
    require(prefill.get("prompt_1011_min_tokens_per_second") == 318.19, "prompt 1011 floor mismatch")
    require(prefill.get("prompt_2048_tokenwise_baseline_ratio_min") == 5.0, "prompt 2048 ratio mismatch")
    require(prefill.get("prompt_1024_target_tokens_per_second") == 1000.0, "prompt 1024 target mismatch")
    decode = policy.get("performance_thresholds", {}).get("decode", {})
    require(decode.get("generated_tokens_per_case") == 64, "decode generation mismatch")
    require(decode.get("requested_m") == 1, "decode M mismatch")
    require(decode.get("context_1339_min_tokens_per_second") == 53.3, "context 1339 floor mismatch")
    require(decode.get("short_context_p50_regression_fraction_max") == 0.05, "short-context regression mismatch")
    dependency = policy.get("control_dependency", {})
    require(dependency.get("target_control_id") == "aq4_0_target", "policy target dependency mismatch")
    require({item.get("control_id") for item in dependency.get("required_before_target_comparison", [])} == {"sq8_0_cross_format", "reference_source_oracle"}, "policy control dependencies incomplete")
    require(policy.get("promotion_rules", {}).get("component_scope_can_promote") is False, "component promotion must be forbidden")
    require(policy.get("promotion_rules", {}).get("full_model_scope_can_promote") is False, "full-model promotion must be forbidden")
    require(policy.get("promotion_rules", {}).get("live_requests_before_parent_p1_gate") is False, "live requests must remain disabled")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--policy", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        manifest = load_json(args.manifest)
        policy = load_json(args.policy)
        summary = validate_manifest(manifest)
        validate_policy(policy)
    except ValidationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
