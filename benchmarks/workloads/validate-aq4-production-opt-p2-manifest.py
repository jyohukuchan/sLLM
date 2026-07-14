#!/usr/bin/env python3
"""CPU-only structural validator for the AQ4 P2 planning manifest and policy.

This validator expands no cases, starts no worker, and does not access a GPU or
network.  It intentionally validates only the planning contract; execution and
promotion remain gated by the independent P1 validators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys
from typing import Any


EXPECTED_PROMPTS = [1, 8, 32, 128, 512, 1011, 1024, 1339, 2048, 3584, 4096]
EXPECTED_M = [1, 8, 16, 32, 64, 128]
EXPECTED_MODES = ["all_m1", "cold_batched", "cached_prefix_chunked"]
EXPECTED_DECODE_CONTEXTS = [16, 128, 512, 1024, 1339, 2048, 3584]
EXPECTED_SCOPES = ["component", "full_model", "production_server"]
EXPECTED_STAGES = ["smoke", "representative", "full"]
EXPECTED_DEVICES = {"cpu-reference", "r9700-rdna4", "v620-rdna2"}
EXPECTED_DEVICE_DEFINITIONS = {
    "cpu-reference": {"backend": "cpu", "gpu_architecture": None, "gpu_name": None, "required": True},
    "r9700-rdna4": {"backend": "hip", "gpu_architecture": "RDNA4", "gpu_name": "Radeon AI PRO R9700", "required": True},
    "v620-rdna2": {"backend": "hip", "gpu_architecture": "RDNA2", "gpu_name": "Radeon PRO V620", "required": False, "capability_decision_required": True},
}
EXPECTED_STAGE_DEVICES = {
    "smoke": ["cpu-reference"],
    "representative": ["cpu-reference", "r9700-rdna4"],
    "full": ["cpu-reference", "r9700-rdna4"],
}
EXPECTED_PARENT_GATES = {
    "smoke": "P1_gate_not_yet_confirmed",
    "representative": "P1_gate_confirmed_and_identity_frozen",
    "full": "representative_cpu_and_r9700_baseline_accepted",
}
EXPECTED_STAGE_AXES = {
    "smoke": {
        "prefill_prompt_tokens": [128, 1011],
        "prefill_requested_m": [1, 16, 128],
        "prefill_modes": ["all_m1", "cold_batched"],
        "prefill_scopes": ["component", "full_model"],
        "production_server_prompt_tokens": [],
        "cached_prefix_prompt_tokens": [],
        "decode_contexts": [16, 1339],
        "decode_scopes": ["component", "full_model"],
        "production_server_contexts": [],
    },
    "representative": {
        "prefill_prompt_tokens": [128, 512, 1011, 1024, 1339, 2048, 3584],
        "prefill_requested_m": [1, 8, 16, 32, 64, 128],
        "prefill_modes": EXPECTED_MODES,
        "prefill_scopes": EXPECTED_SCOPES,
        "production_server_prompt_tokens": [1011, 1024, 1339, 2048],
        "cached_prefix_prompt_tokens": [128, 512, 1011, 1024, 1339, 2048, 3584],
        "decode_contexts": EXPECTED_DECODE_CONTEXTS,
        "decode_scopes": EXPECTED_SCOPES,
        "production_server_contexts": [128, 1339, 2048],
    },
    "full": {
        "prefill_prompt_tokens": EXPECTED_PROMPTS,
        "prefill_requested_m": EXPECTED_M,
        "prefill_modes": EXPECTED_MODES,
        "prefill_scopes": EXPECTED_SCOPES,
        "production_server_prompt_tokens": EXPECTED_PROMPTS,
        "cached_prefix_prompt_tokens": [1, 8, 32, 128, 512, 1011, 1024, 1339, 2048, 3584],
        "decode_contexts": EXPECTED_DECODE_CONTEXTS,
        "decode_scopes": EXPECTED_SCOPES,
        "production_server_contexts": EXPECTED_DECODE_CONTEXTS,
    },
}
REQUIRED_CONTROLS = {
    "aq4_0_target",
    "sq8_0_cross_format",
    "reference_source_oracle",
}
CONTROL_DEFINITIONS = {
    "aq4_0_target": {"role": "target", "format_id": "AQ4_0", "promotion_eligible": True},
    "sq8_0_cross_format": {"role": "cross_format_control", "format_id": "SQ8_0", "promotion_eligible": False},
    "reference_source_oracle": {"role": "independent_reference", "format_id": "REFERENCE", "promotion_eligible": False},
}
REQUIRED_PREFLIGHT_FIELDS = [
    "weights_bytes",
    "persistent_state_bytes",
    "kv_cache_bytes",
    "workspace_bytes",
    "temporary_bytes",
    "vram_headroom_bytes",
    "gpu_process_snapshot",
]
REQUIRED_HASH_FIELDS = [
    "model_identity_sha256",
    "tokenizer_sha256",
    "served_model_manifest_sha256",
    "worker_binary_sha256",
    "package_manifest_sha256",
    "package_content_sha256",
    "graph_identity_sha256",
    "state_schema_sha256",
    "source_oracle_sha256",
    "path_oracle_identity_sha256",
    "baseline_result_sha256",
    "power_capture_sha256",
    "policy_sha256",
    "bound_case_manifest_sha256",
]
REQUIRED_ADAPTER_FLAGS = [
    "adapter_must_be_real",
    "adapter_must_execute_declared_binary",
    "adapter_must_execute_declared_package",
    "adapter_must_record_requested_and_resolved_m",
    "adapter_must_record_actual_token_and_request_batch_width",
    "adapter_must_record_fallback_and_reason",
    "adapter_must_record_preflight_and_peak_memory",
    "adapter_must_record_prepare_commit_discard_reset",
    "adapter_must_publish_atomic_incomplete_suffix",
    "adapter_must_not_claim_component_as_full_model",
    "adapter_must_not_modify_wire_protocol",
    "adapter_must_not_reuse_historical_identity",
    "adapter_must_fail_closed_when_live_request_or_gpu_queue_is_busy",
    "adapter_must_keep_user_content_out_of_evidence",
    "adapter_must_reject_unbound_policy",
    "adapter_must_require_manifest_expansion",
    "adapter_must_bind_manifest_sha256",
    "adapter_must_emit_structured_executor_record",
    "adapter_must_preserve_oom_unsupported_skipped_status",
]
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


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
    reject_nonfinite(value, str(path))
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def reject_nonfinite(value: Any, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"non-finite number: {name}")
    if isinstance(value, dict):
        for key, child in value.items():
            reject_nonfinite(child, f"{name}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_nonfinite(child, f"{name}[{index}]")


def ensure_unique_ints(values: Any, name: str, *, nonempty: bool = True) -> list[int]:
    require(isinstance(values, list), f"{name} must be an array")
    require(all(isinstance(value, int) and not isinstance(value, bool) for value in values), f"{name} must contain integers")
    require(not nonempty or bool(values), f"{name} must not be empty")
    require(len(values) == len(set(values)), f"{name} contains duplicate values")
    return values


def ensure_unique_strings(values: Any, name: str, *, nonempty: bool = True) -> list[str]:
    require(isinstance(values, list), f"{name} must be an array")
    require(all(isinstance(value, str) and value for value in values), f"{name} must contain nonempty strings")
    require(not nonempty or bool(values), f"{name} must not be empty")
    require(len(values) == len(set(values)), f"{name} contains duplicate values")
    return values


def require_hash_or_null(value: Any, name: str) -> None:
    require(value is None or (isinstance(value, str) and HASH_RE.fullmatch(value) is not None), f"{name} must be lowercase SHA-256 or null")


def count_prefill(stage: dict[str, Any], device_id: str, production: bool) -> int:
    selection = stage["prefill"]
    prompts = selection.get("production_server_prompt_tokens", []) if production else selection["prompt_tokens"]
    cached_prompts = [value for value in selection.get("cached_prefix_prompt_tokens", []) if value in prompts]
    scopes = ["production_server"] if production else ["component", "full_model"]
    controls = stage.get("controls_by_device", {}).get(device_id, stage.get("controls", []))
    cold_modes = int("all_m1" in selection["modes"]) + int("cold_batched" in selection["modes"])
    cached_modes = 2 if "cached_prefix_chunked" in selection["modes"] else 0
    phase_width = len(prompts) * cold_modes + len(cached_prompts) * len(selection.get("cached_prefix_tokens", [])) * cached_modes
    return phase_width * len(selection["requested_m"]) * len(scopes) * len(controls)


def count_decode(stage: dict[str, Any], device_id: str, production: bool) -> int:
    selection = stage["decode"]
    contexts = selection.get("production_server_start_context_tokens", []) if production else selection["start_context_tokens"]
    scopes = ["production_server"] if production else ["component", "full_model"]
    controls = stage.get("controls_by_device", {}).get(device_id, stage.get("controls", []))
    return len(contexts) * len(scopes) * len(controls)


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    require(manifest.get("schema_version") == "ullm.aq4_production_p2_case_manifest.v1", "unexpected manifest schema")
    require(manifest.get("status") == "planning_only", "manifest must remain planning_only")
    identity = manifest.get("identity_binding", {})
    require(identity.get("model_family") == "Qwen3.5" and identity.get("model_name") == "Qwen3.5-9B", "target model identity differs")
    require(identity.get("format_id") == "AQ4_0", "target format must be AQ4_0")
    require(identity.get("path_oracle_identity") == "same-artifact-all-m1", "path oracle identity differs")
    require(identity.get("power_condition_identity") == "r9700-exclusive-power-template-v0.1", "power condition identity differs")
    require(identity.get("context_limit_tokens") == 4096, "context limit must be 4096")
    for field in [
        "model_identity_sha256",
        "tokenizer_sha256",
        "served_model_manifest_sha256",
        "worker_binary_sha256",
        "package_manifest_sha256",
        "package_content_sha256",
        "graph_identity_sha256",
        "state_schema_sha256",
        "source_oracle_sha256",
    ]:
        require(field in identity, f"identity_binding.{field} is missing")
        require_hash_or_null(identity.get(field), f"identity_binding.{field}")
    require(identity.get("build_git_commit") is None or (isinstance(identity.get("build_git_commit"), str) and re.fullmatch(r"[0-9a-f]{40}", identity["build_git_commit"])), "build_git_commit must be a lowercase Git SHA or null")

    safety = manifest.get("execution_safety", {})
    require(safety.get("gpu_processes_at_once") == 1, "GPU process limit must be one")
    require(safety.get("r9700_exclusive") is True, "R9700 must be exclusive")
    require(safety.get("preflight_required") is True, "preflight must be required")
    require(safety.get("stream_tensors_and_logits") is True, "tensor/logit streaming must be enabled")
    require(safety.get("retain_full_logit_matrix") is False, "full logits retention is unsafe")
    require(safety.get("retain_full_attention_matrix") is False, "full attention retention is unsafe")
    require(safety.get("case_cleanup_required") is True, "case cleanup must be required")
    require(safety.get("oom_is_immutable_result") is True, "OOM must remain immutable evidence")
    require(safety.get("live_requests_allowed_during_preparation") is False, "live requests must remain disabled")
    require(safety.get("preflight_fields") == REQUIRED_PREFLIGHT_FIELDS, "preflight fields differ")

    axes = manifest.get("axes", {})
    require(ensure_unique_ints(axes.get("prefill_prompt_tokens"), "prefill_prompt_tokens") == EXPECTED_PROMPTS, "prefill prompt axis mismatch")
    require(ensure_unique_ints(axes.get("prefill_requested_m"), "prefill_requested_m") == EXPECTED_M, "prefill M axis mismatch")
    require(axes.get("prefill_modes") == EXPECTED_MODES, "prefill mode axis mismatch")
    require(ensure_unique_ints(axes.get("decode_start_context_tokens"), "decode_start_context_tokens") == EXPECTED_DECODE_CONTEXTS, "decode context axis mismatch")
    require(axes.get("decode_request_count") == 1, "decode request count must be one")
    require(axes.get("decode_generated_tokens") == 64, "decode generated token count must be 64")
    require(axes.get("scopes") == EXPECTED_SCOPES, "scope axis mismatch")
    devices = axes.get("devices", [])
    require(isinstance(devices, list) and len(devices) == len(EXPECTED_DEVICES), "device axis must contain CPU, R9700, and V620 exactly once")
    device_ids = [device.get("device_id") for device in devices]
    require(set(device_ids) == EXPECTED_DEVICES and len(device_ids) == len(set(device_ids)), "device axis differs")
    device_by_id = {device["device_id"]: device for device in devices}
    for device_id, expected_device in EXPECTED_DEVICE_DEFINITIONS.items():
        for field, expected in expected_device.items():
            require(device_by_id[device_id].get(field) == expected, f"{device_id}.{field} differs")

    controls = axes.get("controls", [])
    control_ids = {control.get("control_id") for control in controls}
    require(control_ids == REQUIRED_CONTROLS, "AQ4, SQ8, and reference controls are required")
    control_by_id = {control["control_id"]: control for control in controls}
    for control_id, definition in CONTROL_DEFINITIONS.items():
        control = control_by_id[control_id]
        for field, expected in definition.items():
            require(control.get(field) == expected, f"{control_id}.{field} differs")
    require(control_by_id["sq8_0_cross_format"].get("required_for_aq4_comparison") is True, "SQ8 control dependency must be explicit")
    require(control_by_id["reference_source_oracle"].get("required_for_aq4_comparison") is True, "source oracle dependency must be explicit")
    dependencies = manifest.get("control_dependencies", {})
    require(dependencies.get("target_control_id") == "aq4_0_target", "target dependency is not AQ4")
    require(set(dependencies.get("required_control_ids", [])) == {"sq8_0_cross_format", "reference_source_oracle"}, "AQ4 control dependencies are incomplete")
    require(dependencies.get("missing_control_action") == "ineligible", "missing controls must fail closed")
    require(dependencies.get("control_hashes_must_be_bound_before_r9700_promotion") is True, "control hashes must bind before R9700 promotion")
    require(dependencies.get("sq8_0_policy") == "reference_control_only_until_sq8_oracle_and_builder_gate", "SQ8 policy differs")
    require(dependencies.get("source_oracle_independent_capture_required") is True, "source oracle must be independently captured")
    for field in ["sq8_oracle_sha256", "sq8_builder_gate_sha256", "reference_source_oracle_sha256"]:
        require_hash_or_null(dependencies.get(field), f"control_dependencies.{field}")

    path_oracle = manifest.get("path_oracle_contract", {})
    require(path_oracle.get("mode") == "all_m1", "path oracle mode must be all_m1")
    require(path_oracle.get("required_for_modes") == ["cold_batched", "cached_prefix_chunked"], "path oracle modes differ")
    require(path_oracle.get("case_link_field") == "path_oracle_case_id", "path oracle case link field differs")
    require(path_oracle.get("result_hash_field") == "path_oracle_result_sha256", "path oracle result hash field differs")
    require(path_oracle.get("same_artifact_required") is True, "path oracle must use the same artifact")
    require(path_oracle.get("same_cached_prefix_state_required") is True, "cached path oracle must preserve prefix state")
    require(path_oracle.get("all_m1_fallback_must_be_recorded") is True, "all-M=1 fallback must be recorded")

    validation_dependencies = manifest.get("validation_dependencies", {})
    topology_controls = validation_dependencies.get("required_topology_controls", [])
    require(len(topology_controls) == 1, "Qwen3 dense topology control dependency is required")
    topology_control = topology_controls[0]
    require(topology_control == {
        "control_id": "qwen3_dense_full_model_control",
        "topology": "Qwen3",
        "scope": "full_model",
        "required_devices": ["cpu-reference", "r9700-rdna4"],
        "promotion_eligible": False,
        "status": "external_dependency",
        "artifact_sha256": None,
    }, "Qwen3 dense topology dependency differs")
    v620_dependency = validation_dependencies.get("v620_capability_decision", {})
    require(v620_dependency == {
        "device_id": "v620-rdna2",
        "status": "unbound",
        "allowed_statuses": ["supported", "unsupported", "skipped"],
        "reason_required_for_non_supported": True,
        "artifact_sha256": None,
    }, "V620 capability dependency differs")

    adapter = manifest.get("command_adapter_requirements", {})
    require(adapter.get("p1_runner_ownership") == "tools/run-aq4-production-performance-matrix.py", "P1 runner ownership differs")
    for field in REQUIRED_ADAPTER_FLAGS:
        require(adapter.get(field) is True, f"adapter requirement {field} must be true")

    stages = manifest.get("stages", [])
    require([stage.get("stage_id") for stage in stages] == EXPECTED_STAGES, "stage order must be smoke, representative, full")
    totals: dict[str, int] = {}
    for stage in stages:
        stage_id = stage.get("stage_id")
        require(stage_id in EXPECTED_STAGE_AXES, f"unknown stage {stage_id}")
        expected_axes = EXPECTED_STAGE_AXES[stage_id]
        require(stage.get("order") == EXPECTED_STAGES.index(stage_id) + 1, f"{stage_id} order differs")
        require(stage.get("execution_allowed_now") is False, f"stage {stage_id} cannot execute during preparation")
        require(stage.get("devices") == EXPECTED_STAGE_DEVICES[stage_id], f"{stage_id} devices differ")
        expected_controls = (
            {"cpu-reference": ["aq4_0_target", "sq8_0_cross_format", "reference_source_oracle"]}
            if stage_id == "smoke" else
            {"cpu-reference": ["aq4_0_target", "sq8_0_cross_format", "reference_source_oracle"], "r9700-rdna4": ["aq4_0_target", "reference_source_oracle"]}
        )
        if stage_id == "smoke":
            require(stage.get("controls") == expected_controls["cpu-reference"], "smoke controls differ")
            require("controls_by_device" not in stage, "smoke must use its single controls list")
        else:
            require(stage.get("controls_by_device") == expected_controls, f"{stage_id} controls differ")
            require("controls" not in stage, f"{stage_id} must use controls_by_device")
        prefill = stage["prefill"]
        decode = stage["decode"]
        require(ensure_unique_ints(prefill["prompt_tokens"], f"{stage_id}.prefill.prompt_tokens") == expected_axes["prefill_prompt_tokens"], f"{stage_id} prefill prompt axis differs")
        require(ensure_unique_ints(prefill["requested_m"], f"{stage_id}.prefill.requested_m") == expected_axes["prefill_requested_m"], f"{stage_id} prefill M axis differs")
        require(ensure_unique_strings(prefill["modes"], f"{stage_id}.prefill.modes") == expected_axes["prefill_modes"], f"{stage_id} prefill mode axis differs")
        require(ensure_unique_strings(prefill["scopes"], f"{stage_id}.prefill.scopes") == expected_axes["prefill_scopes"], f"{stage_id} prefill scope axis differs")
        require(ensure_unique_ints(prefill.get("production_server_prompt_tokens", []), f"{stage_id}.prefill.production_server_prompt_tokens", nonempty=False) == expected_axes["production_server_prompt_tokens"], f"{stage_id} production prompt axis differs")
        require(ensure_unique_ints(prefill.get("cached_prefix_tokens"), f"{stage_id}.prefill.cached_prefix_tokens") == [128], f"{stage_id} cached prefix width differs")
        require(ensure_unique_ints(prefill.get("cached_prefix_prompt_tokens", []), f"{stage_id}.prefill.cached_prefix_prompt_tokens", nonempty=False) == expected_axes["cached_prefix_prompt_tokens"], f"{stage_id} cached prefix prompt axis differs")
        require(ensure_unique_ints(decode["start_context_tokens"], f"{stage_id}.decode.start_context_tokens") == expected_axes["decode_contexts"], f"{stage_id} decode context axis differs")
        require(decode["request_count"] == 1 and decode["generated_tokens"] == 64, f"{stage_id} decode contract mismatch")
        require(ensure_unique_strings(decode["scopes"], f"{stage_id}.decode.scopes") == expected_axes["decode_scopes"], f"{stage_id} decode scope axis differs")
        require(ensure_unique_ints(decode.get("production_server_start_context_tokens", []), f"{stage_id}.decode.production_server_start_context_tokens", nonempty=False) == expected_axes["production_server_contexts"], f"{stage_id} production decode axis differs")
        require(all(value <= 4096 for value in prefill["prompt_tokens"]), f"{stage_id} prompt exceeds context limit")
        require(all(value + 64 <= 4096 for value in decode["start_context_tokens"]), f"{stage_id} decode context plus generated tokens exceeds context limit")
        require(all(value + prefix <= 4096 for value in prefill["cached_prefix_prompt_tokens"] for prefix in prefill["cached_prefix_tokens"]), f"{stage_id} cached prefix exceeds context limit")
        require(("cached_prefix_chunked" in prefill["modes"]) == bool(prefill["cached_prefix_prompt_tokens"]), f"{stage_id} cached mode/prefix axis mismatch")
        require(stage.get("requires_parent_gate") == EXPECTED_PARENT_GATES[stage_id], f"{stage_id} parent gate differs")
        if stage_id == "full":
            require(stage.get("context_edge_rule") == "prefill prompt 4096 is prefill-only; decode starts plus 64 generated tokens must remain <= 4096", "full context edge rule differs")
        else:
            require("context_edge_rule" not in stage, f"{stage_id} must not define a full-stage context edge rule")
        for device_id in stage["devices"]:
            controls_for_device = stage.get("controls_by_device", {}).get(device_id, stage.get("controls", []))
            require(ensure_unique_strings(controls_for_device, f"{stage_id} controls for {device_id}") == expected_controls[device_id], f"{stage_id} controls differ for {device_id}")
            totals[f"{stage_id}.{device_id}.prefill_non_server"] = count_prefill(stage, device_id, False)
            totals[f"{stage_id}.{device_id}.prefill_production_server"] = count_prefill(stage, device_id, True)
            totals[f"{stage_id}.{device_id}.decode_non_server"] = count_decode(stage, device_id, False)
            totals[f"{stage_id}.{device_id}.decode_production_server"] = count_decode(stage, device_id, True)
        expected = stage.get("expected_case_count", {})
        if stage_id == "smoke":
            require(expected.get("prefill") == totals["smoke.cpu-reference.prefill_non_server"], "smoke prefill count mismatch")
            require(expected.get("decode") == totals["smoke.cpu-reference.decode_non_server"], "smoke decode count mismatch")
            require(expected.get("total") == expected["prefill"] + expected["decode"], "smoke total count mismatch")
        elif stage_id == "representative":
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

    queue = manifest.get("r9700_queue", {})
    require(queue.get("queue_id") == "r9700-exclusive-p2-v0.1", "R9700 queue id differs")
    require(queue.get("device_id") == "r9700-rdna4", "R9700 queue device differs")
    require(queue.get("max_concurrent_cases") == 1, "R9700 queue must be serial")
    require(queue.get("order") == ["preflight", "smoke", "representative", "full"], "R9700 queue order mismatch")
    require(queue.get("within_stage_order") == ["control_reference", "aq4_target", "production_server"], "R9700 within-stage order differs")
    require(queue.get("between_cases") == ["drain_streams", "reset_executor_state", "capture_power_and_vram", "release_case_buffers"], "R9700 cleanup order differs")
    require(queue.get("positive_vram_processes_allowed") == ["ullm-aq4-worker"], "R9700 allowed process list differs")
    require(queue.get("lock_release_on_failure") is True, "R9700 lock must release on failure")
    require(queue.get("other_positive_vram_process_action") == "skip_and_record_busy", "busy R9700 action must be recorded")

    require(manifest.get("case_id_width_label_rule") == "prefill uses m{prefill_requested_m}; decode uses requests{decode_request_count}; never interpret decode request count as prefill M", "case id width rule differs")
    publication = manifest.get("publication", {})
    require(publication == {
        "result_schema": "inference-benchmark-result-v0.1",
        "validation_schema": "ullm.prefill_validation.v1",
        "trace_schema": "ullm.production_execution_trace.v1",
        "temporary_suffix": ".incomplete",
        "atomic_publish": True,
        "never_overwrite_existing_evidence": True,
        "failure_rows_are_immutable": True,
    }, "publication contract differs")

    return {"valid": True, "stage_counts": totals}


def validate_policy(policy: dict[str, Any]) -> None:
    require(policy.get("schema_version") == "ullm.aq4_production_p2_threshold_policy.v1", "unexpected policy schema")
    require(policy.get("status") == "unbound_template", "policy must remain an unbound template")
    require(policy.get("effective_at") is None and policy.get("scope") == "planning_only", "policy template scope differs")
    binding_contract = policy.get("binding_contract", {})
    require(binding_contract.get("bound_status") == "bound", "bound policy status is not declared")
    require(binding_contract.get("required_before_case_execution") is True, "bound policy is not required before execution")
    require(binding_contract.get("policy_sha256_rule") == "sha256-canonical-json-with-hash_binding.policy_sha256-null", "policy hash rule differs")
    require(binding_contract.get("required_hash_fields") == REQUIRED_HASH_FIELDS, "required policy hash fields differ")
    require(binding_contract.get("required_power_fields") == ["expected_power_limit_watts", "allowed_power_tolerance_watts", "maximum_temperature_c", "minimum_vram_headroom_bytes"], "required power fields differ")
    require(binding_contract.get("required_correctness_thresholds") == ["max_hidden_relative_l2", "max_hidden_max_abs", "max_logits_relative_l2", "max_logits_max_abs", "minimum_top_k_overlap"], "required correctness thresholds differ")
    require(binding_contract.get("unbound_template_is_planning_only") is True, "unbound template planning guard is missing")
    require(policy.get("hash_binding", {}).get("binding_required_before_measurement") is True, "policy hash binding is not required")
    hash_binding = policy.get("hash_binding", {})
    for field in REQUIRED_HASH_FIELDS:
        require(field in hash_binding, f"policy hash field is missing: {field}")
        require(hash_binding[field] is None, f"planning policy hash field must remain null: {field}")
    require(policy.get("applies_to", {}).get("target_format_id") == "AQ4_0", "policy target format must be AQ4_0")
    require(policy.get("applies_to", {}).get("target_gpu_name") == "Radeon AI PRO R9700", "policy target GPU must be R9700")
    require(policy.get("power_condition", {}).get("max_concurrent_gpu_cases") == 1, "policy power queue must be serial")
    require(policy.get("power_condition", {}).get("binding_required") is True, "power condition binding is not required")
    power = policy["power_condition"]
    require(power.get("capture_required") == [
        "rocm_smi_device_identity",
        "rocm_smi_temperature_c",
        "rocm_smi_power_watts",
        "rocm_smi_power_limit_watts",
        "rocm_smi_vram_used_bytes",
        "rocm_smi_process_snapshot",
        "driver_version",
        "runtime_version",
        "host_identity",
    ], "power capture fields differ")
    for field in binding_contract["required_power_fields"]:
        require(power.get(field) is None, f"planning power field must remain null: {field}")
    prefill = policy.get("performance_thresholds", {}).get("prefill", {})
    require(prefill.get("prompt_1011_min_tokens_per_second") == 318.19, "prompt 1011 floor mismatch")
    require(prefill.get("prompt_2048_tokenwise_baseline_ratio_min") == 5.0, "prompt 2048 ratio mismatch")
    require(prefill.get("prompt_1024_target_tokens_per_second") == 1000.0, "prompt 1024 target mismatch")
    require(prefill.get("actual_token_batch_width_must_be_recorded") is True, "actual prefill width must be recorded")
    require(prefill.get("unexpected_fallback_is_failure") is True and prefill.get("oom_is_failure") is True, "prefill failure policy differs")
    decode = policy.get("performance_thresholds", {}).get("decode", {})
    require(decode.get("generated_tokens_per_case") == 64, "decode generation mismatch")
    require(decode.get("request_count") == 1 and decode.get("actual_token_batch_width_required") == 1, "decode request/token width contract differs")
    require(decode.get("context_1339_min_tokens_per_second") == 53.3, "context 1339 floor mismatch")
    require(decode.get("short_context_p50_regression_fraction_max") == 0.05, "short-context regression mismatch")
    dependency = policy.get("control_dependency", {})
    require(dependency.get("target_control_id") == "aq4_0_target", "policy target dependency mismatch")
    require({item.get("control_id") for item in dependency.get("required_before_target_comparison", [])} == {"sq8_0_cross_format", "reference_source_oracle"}, "policy control dependencies incomplete")
    required_controls = {item["control_id"]: item for item in dependency["required_before_target_comparison"]}
    require(required_controls["reference_source_oracle"].get("format_id") == "REFERENCE" and required_controls["reference_source_oracle"].get("role") == "independent source oracle" and required_controls["reference_source_oracle"].get("promotion_eligible") is False, "source oracle control definition differs")
    require(required_controls["sq8_0_cross_format"].get("format_id") == "SQ8_0" and required_controls["sq8_0_cross_format"].get("role") == "cross-format implementation control" and required_controls["sq8_0_cross_format"].get("promotion_eligible") is False, "SQ8 control definition differs")
    require(required_controls["reference_source_oracle"].get("required_scopes") == EXPECTED_SCOPES, "source oracle scopes differ")
    require(required_controls["reference_source_oracle"].get("required_devices") == ["cpu-reference"], "source oracle device differs")
    require(required_controls["sq8_0_cross_format"].get("required_scopes") == ["component", "full_model"], "SQ8 scopes differ")
    require(required_controls["sq8_0_cross_format"].get("required_devices") == ["cpu-reference"], "SQ8 device differs")
    require(dependency.get("matching_identity_fields") == [
        "model_identity_sha256",
        "tokenizer_sha256",
        "package_manifest_sha256",
        "package_content_sha256",
        "format_id",
        "implementation_id",
        "resolved_m",
        "backend_identity",
        "device_identity",
        "prompt_tokens",
        "cached_prefix_tokens",
        "context_tokens",
        "decode_start_tokens",
        "prefill_requested_m",
        "decode_request_count",
        "phase",
        "baseline_mode",
        "source_oracle_identity",
        "sampling",
        "power_capture_sha256",
    ], "policy identity matching fields differ")
    require(policy.get("promotion_rules", {}).get("component_scope_can_promote") is False, "component promotion must be forbidden")
    require(policy.get("promotion_rules", {}).get("full_model_scope_can_promote") is False, "full-model promotion must be forbidden")
    require(policy.get("promotion_rules", {}).get("live_requests_before_parent_p1_gate") is False, "live requests must remain disabled")


def canonical_policy_sha256(policy: dict[str, Any]) -> str:
    value = json.loads(json.dumps(policy, ensure_ascii=False))
    value.setdefault("hash_binding", {})["policy_sha256"] = None
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_bound_policy(policy: dict[str, Any]) -> None:
    """Validate the artifact that an execution adapter must consume.

    The committed policy is deliberately an unbound planning template. A
    runner must create a separate bound copy and satisfy this check before any
    case is launched.
    """
    require(policy.get("schema_version") == "ullm.aq4_production_p2_threshold_policy.v1", "unexpected bound policy schema")
    require(policy.get("status") == "bound", "execution requires a bound policy")
    require(isinstance(policy.get("effective_at"), str) and policy["effective_at"], "bound policy effective_at is missing")
    contract = policy.get("binding_contract", {})
    require(contract.get("required_before_case_execution") is True, "bound policy execution guard is missing")
    require(contract.get("required_hash_fields") == REQUIRED_HASH_FIELDS, "bound policy hash field contract differs")
    require(contract.get("required_power_fields") == ["expected_power_limit_watts", "allowed_power_tolerance_watts", "maximum_temperature_c", "minimum_vram_headroom_bytes"], "bound policy power field contract differs")
    require(contract.get("required_correctness_thresholds") == ["max_hidden_relative_l2", "max_hidden_max_abs", "max_logits_relative_l2", "max_logits_max_abs", "minimum_top_k_overlap"], "bound policy correctness field contract differs")
    hashes = policy.get("hash_binding", {})
    for field in REQUIRED_HASH_FIELDS:
        value = hashes.get(field)
        require(isinstance(value, str) and HASH_RE.fullmatch(value) is not None, f"bound policy hash is invalid: {field}")
    require(hashes["policy_sha256"] == canonical_policy_sha256(policy), "policy self-hash differs")
    power = policy.get("power_condition", {})
    for field in contract.get("required_power_fields", []):
        value = power.get(field)
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0, f"bound power field is invalid: {field}")
    require(power.get("expected_power_limit_watts", 0) > 0 and power.get("maximum_temperature_c", 0) > 0 and power.get("minimum_vram_headroom_bytes", 0) > 0, "bound power limits/headroom must be positive")
    correctness = policy.get("correctness_thresholds", {})
    for field in contract.get("required_correctness_thresholds", []):
        value = correctness.get(field)
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0, f"bound correctness threshold is invalid: {field}")
    overlap = correctness.get("minimum_top_k_overlap")
    require(isinstance(overlap, int) and not isinstance(overlap, bool) and 0 <= overlap <= correctness.get("top_k", 0), "bound top-k overlap is invalid")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--policy", type=pathlib.Path, required=True)
    parser.add_argument("--bound-policy", type=pathlib.Path, help="optional bound policy artifact required by an execution adapter")
    args = parser.parse_args()
    try:
        manifest = load_json(args.manifest)
        policy = load_json(args.policy)
        summary = validate_manifest(manifest)
        validate_policy(policy)
        if args.bound_policy is not None:
            validate_bound_policy(load_json(args.bound_policy))
            summary["bound_policy_valid"] = True
    except (ValidationError, KeyError, TypeError, IndexError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
