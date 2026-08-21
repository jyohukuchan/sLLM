#!/usr/bin/env python3
"""Dependency-free validator for the Phase 45 adapter/lifecycle contract.

The JSON Schema describes the interchange shape.  This validator pins the
semantic limits and the fail-closed boundaries so a permissive schema consumer
cannot widen adapter identity, model lifecycle, or offline loading policy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/phase45_adapter_lifecycle_v1.json"
SCHEMA = ROOT / "ci/schema/phase45-adapter-lifecycle-v1.schema.json"
LLAMA_COMMIT = "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70"
LLAMA_RELEASE = "b10453"
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")

POSITIVE_IDS = {
    "lora-qwen-bf16-single-scale-one",
    "lora-qwen-bf16-ordered-set",
    "control-qwen-bf16-half-open-range",
    "disabled-adapter-preserves-none-identity",
    "identity-binds-base-plan-adapter-control-target",
    "registry-lazy-load-ready",
    "registry-preload-within-quota",
    "registry-load-coalesced",
    "registry-drain-last-owner-shutdown",
    "registry-lru-idle-eviction",
    "router-alias-only-admin-action",
    "request-ordered-adapter-control-extension",
    "offline-derived-artifact-manifest",
    "synthetic-slice-oracle",
    "qwen-bf16-gpu-smoke-rdna2",
    "qwen-bf16-gpu-smoke-rdna4",
}

REJECTION_IDS = {
    "lora-wrong-base-lock",
    "lora-missing-target-tensor",
    "lora-shape-mismatch",
    "lora-dtype-mismatch",
    "lora-rank-zero",
    "lora-rank-over-limit",
    "lora-scale-nonfinite",
    "lora-scale-below-min",
    "lora-scale-above-max",
    "lora-duplicate-adapter",
    "lora-order-not-canonical",
    "control-wrong-base-lock",
    "control-missing-layer",
    "control-range-overlap",
    "control-range-not-half-open",
    "control-dtype-mismatch",
    "control-scale-out-of-range",
    "unsupported-model-capability",
    "unsupported-low-bit-dtype",
    "manifest-path-not-regular",
    "manifest-network-source",
    "manifest-alias-conflict",
    "registry-alias-count-over-limit",
    "registry-resident-quota-over-limit",
    "registry-active-lru-eviction",
    "registry-unload-inflight-owner",
    "registry-illegal-transition",
    "request-adapter-count-over-limit",
    "request-control-count-over-limit",
    "request-unknown-alias",
    "request-loading-or-draining",
    "admin-action-with-path",
    "mi300x-runtime-not-claimed",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load(path: Path) -> Any:
    """Load JSON without accepting duplicate names, NaN, or Infinity."""

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


def _expect(value: Any, expected: Any, message: str) -> None:
    require(value == expected, message)


def _expect_dict(value: Any, message: str) -> dict[str, Any]:
    require(isinstance(value, dict), message)
    return value


def _cases(document: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cases = _expect_dict(document.get("cases"), "cases missing")
    positive = cases.get("positive")
    rejection = cases.get("rejection")
    require(
        isinstance(positive, list) and len(positive) == len(POSITIVE_IDS),
        "positive case count changed",
    )
    require(
        isinstance(rejection, list) and len(rejection) == len(REJECTION_IDS),
        "rejection case count changed",
    )

    def rows(values: list[Any], label: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for row in values:
            item = _expect_dict(row, f"{label} case must be an object")
            case_id = item.get("id")
            require(isinstance(case_id, str) and case_id, f"{label} case id missing")
            require(case_id not in result, f"duplicate case id: {case_id}")
            require(isinstance(item.get("surface"), str), f"{case_id}: surface missing")
            require(isinstance(item.get("input"), dict), f"{case_id}: input must be an object")
            require(isinstance(item.get("expected"), dict), f"{case_id}: expected must be an object")
            result[case_id] = item
        return result

    positive_rows = rows(positive, "positive")
    rejection_rows = rows(rejection, "rejection")
    require(set(positive_rows) == POSITIVE_IDS, "positive case identity set changed")
    require(set(rejection_rows) == REJECTION_IDS, "rejection case identity set changed")
    for case_id, row in positive_rows.items():
        require(row["expected"].get("result") == "accepted", f"{case_id}: positive result changed")
    for case_id, row in rejection_rows.items():
        expected = row["expected"]
        require(
            expected.get("admission") == "before_gpu_admission",
            f"{case_id}: validation is not pre-GPU admission",
        )
        require(
            expected.get("status") in {400, 404, 409, 413, 429, 503},
            f"{case_id}: rejection status is not in the pinned error map",
        )
        require(isinstance(expected.get("code"), str) and expected["code"], f"{case_id}: rejection code missing")
    return positive_rows, rejection_rows


def validate(document: Any, schema: Any) -> None:
    root = _expect_dict(document, "fixture root must be an object")
    schema_root = _expect_dict(schema, "schema root must be an object")
    _expect(schema_root.get("$schema"), "https://json-schema.org/draft/2020-12/schema", "schema draft changed")
    _expect(schema_root.get("$id"), "https://sllm.dev/schema/phase45-adapter-lifecycle-v1.schema.json", "schema id changed")
    _expect(root.get("$schema"), schema_root["$id"], "fixture schema reference changed")
    _expect(root.get("schema_version"), "sllm-phase45-adapter-lifecycle-v1", "fixture schema version changed")
    _expect(root.get("profile_version"), "sllm-phase45-adapter-lifecycle-v1", "profile version changed")
    _expect(root.get("recorded_at"), "2026-08-22", "recorded date changed")

    pins = _expect_dict(root.get("spec_pins"), "spec_pins missing")
    llama = _expect_dict(pins.get("llama_cpp"), "llama.cpp pin missing")
    _expect(llama.get("release"), LLAMA_RELEASE, "llama.cpp release changed")
    _expect(llama.get("commit"), LLAMA_COMMIT, "llama.cpp commit changed")
    require(re.fullmatch(r"[0-9a-f]{40}", llama["commit"]) is not None, "llama.cpp commit is not a full lowercase SHA-1")
    _expect(llama.get("use"), "behavior-reference-only", "llama.cpp reuse boundary changed")
    _expect(pins.get("artifact_source"), "verified-offline-derived-artifacts-only", "artifact source policy changed")
    _expect(pins.get("execution_capability"), "reviewed-dense-bf16-qwen-v1", "execution capability pin changed")

    scope = _expect_dict(root.get("scope"), "scope missing")
    _expect(scope, {
        "execution": "reviewed-dense-bf16-qwen-only",
        "adapter_formats": ["preloaded-lora-v1", "control-vector-v1"],
        "base_artifact": "verified-gguf-derived-lock",
        "offline_only": True,
        "fallback": False,
        "gpu_provider_changes": "additive-broadcast-add-existing-elementwise-family",
    }, "scope changed")
    _expect(root.get("non_goals"), [
        "phase46-conversion-quantization-benchmark-quality-tools",
        "phase47-tool-mcp-execution",
        "phase48-webui",
        "mi300x-real-execution",
        "model-architecture-additions",
        "new-hardware-backends",
        "parallel-multi-gpu",
    ], "non-goals changed")

    lora = _expect_dict(root.get("lora"), "lora profile missing")
    _expect(lora, {
        "id": "preloaded-lora-v1",
        "execution": "reviewed-dense-bf16-qwen-v1",
        "source": "verified-offline-derived-artifact",
        "target_dtype": "bf16",
        "adapter_dtype": "bf16",
        "max_preloaded_per_model": 8,
        "max_request_adapters": 4,
        "rank": {"min": 1, "max": 256, "non_aligned_examples": [1, 3, 17, 255, 256]},
        "scale": {"type": "finite-f32", "min": -16.0, "max": 16.0, "default": 1.0},
        "order": "canonical-sorted-unique",
        "disabled_identity": "adapter:none-v1",
        "admission": "verify-lock-artifact-target-shapes-before-gpu",
        "unsupported": "reject",
    }, "LoRA policy changed")

    control = _expect_dict(root.get("control_vectors"), "control vector profile missing")
    _expect(control, {
        "id": "control-vector-v1",
        "execution": "reviewed-dense-bf16-qwen-v1",
        "source": "verified-offline-derived-artifact",
        "dtype": "bf16",
        "max_request_vectors": 4,
        "layer_range": "half-open",
        "overlap": "reject",
        "scale": {"type": "finite-f32", "min": -16.0, "max": 16.0, "default": 1.0},
        "order": "canonical-sorted-unique",
        "admission": "verify-lock-artifact-layer-range-shape-before-gpu",
        "unsupported": "reject",
    }, "control-vector policy changed")

    identity = _expect_dict(root.get("identity"), "identity policy missing")
    _expect(identity, {
        "canonical_encoding": "sllm-phase45-identity-v1",
        "fields": [
            "base_model_lock_fingerprint",
            "derived_plan_digest",
            "ordered_adapter_artifact_ids",
            "ordered_adapter_scales",
            "ordered_control_vector_artifact_ids",
            "ordered_control_vector_scales",
            "target_semantics",
            "renderer_identity",
            "tokenizer_identity",
        ],
        "alias_is_identity": False,
        "path_is_identity": False,
        "disabled_adapter_identity": "adapter:none-v1",
        "prefix_checkpoint_binds": True,
        "silent_cross_identity_reuse": False,
    }, "identity policy changed")

    registry = _expect_dict(root.get("registry"), "registry profile missing")
    _expect(registry, {
        "states": ["unloaded", "loading", "ready", "draining", "failed", "quarantined"],
        "max_loaded_models": 16,
        "max_configured_aliases": 64,
        "lease": "linearizable",
        "load": "coalesced-by-immutable-identity",
        "drain": "reject-new-then-shutdown-after-last-owner",
        "lru": "idle-only-active-never-evict",
        "resident_quota": "bounded-by-configured-bytes",
        "failure": "publish-nothing-and-quarantine-until-explicit-clear",
        "offline": True,
        "shared_assets": "tokenizer-template-shared-by-base-owner",
    }, "registry policy changed")

    router = _expect_dict(root.get("router"), "router profile missing")
    _expect(router, {
        "manifest": "alias-to-verified-model-and-artifact-identities",
        "admin_action": "alias-only",
        "request_extension": "sllm-ordered-adapter-control-selection-v1",
        "loading_status": 503,
        "draining_status": 503,
        "unknown_alias_status": 404,
        "queue_full_status": 429,
        "auth": "admin-for-lifecycle-user-for-inference",
        "paths_or_credentials_in_request": False,
        "openai_profiles_unchanged": True,
    }, "router policy changed")

    manifest = _expect_dict(root.get("manifest"), "manifest policy missing")
    _expect(manifest, {
        "id": "sllm-model-lifecycle-manifest-v1",
        "paths": "manifest-defined-regular-files",
        "verified_before_publish": True,
        "network": False,
        "symlink": "reject",
        "special_file": "reject",
        "path_race": "reject",
        "model_lock": "exact-fingerprint-and-file-digest",
        "derived_artifact": "exact-source-lock-tool-recipe-output-digest",
    }, "manifest policy changed")

    cli_server = _expect_dict(root.get("cli_server"), "CLI/server boundary missing")
    _expect(cli_server, {
        "manifest_option": "--models",
        "management_surface": "cli-and-admin-server",
        "admin_actions": ["load", "preload", "unload", "clear-quarantine", "evict-idle"],
        "admin_arguments": "alias-only",
        "request_fields": ["sllm.adapters", "sllm.control_vectors"],
        "request_order": "preserve-then-canonical-validate",
        "gpu_admission": "after-preflight-and-registry-lease",
    }, "CLI/server boundary changed")

    verification = _expect_dict(root.get("verification"), "verification matrix missing")
    _expect(verification, {
        "synthetic_oracle": ["slice-logit-delta", "disabled-bit-and-token-identity", "identity-digest"],
        "gpu_models": ["qwen3.5-4b-bf16"],
        "gpu_targets": ["gfx1030", "gfx1201"],
        "gpu_requirements": ["exact-target", "hip-only", "fallback-false", "cleanup-zero", "resident-baseline-restored"],
        "gfx942": "compile-only-or-deferred-no-runtime-pass",
        "full_model_smoke": True,
        "mi300x_real": "deferred",
    }, "verification matrix changed")

    _cases(root)


def main() -> int:
    try:
        document = load(FIXTURE)
        schema = load(SCHEMA)
        validate(document, schema)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"phase45 profile: FAIL: {error}")
        return 1
    print("phase45 profile: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
