#!/usr/bin/env python3
"""Dependency-free validator for the Phase 44 machine contract.

The checked-in Draft 2020-12 schema describes the interchange shape.  This
validator pins the security boundary and all semantic limits so a permissive
schema consumer cannot silently widen template execution, reasoning control,
CLI source selection, or checkpoint identity.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/phase44_template_reasoning_cli_v1.json"
SCHEMA = ROOT / "ci/schema/phase44-template-reasoning-cli-v1.schema.json"
LLAMA_COMMIT = "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70"
LLAMA_RELEASE = "b10453"
MINIJINJA_VERSION = "2.24.0"
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")

POSITIVE_IDS = {
    "template-canonical-roles-special-unicode",
    "template-tool-reasoning-block",
    "template-add-generation-prompt",
    "template-source-limit-exact",
    "template-kwargs-depth-exact",
    "reviewed-qwen-default-identity",
    "reasoning-disabled",
    "reasoning-enabled-budget-one",
    "reasoning-template-default-budget-4096",
    "reasoning-nonaligned-budget",
    "reasoning-early-close",
    "responses-effort-mapping",
    "cli-prompt-file-source",
    "cli-interactive-stdin-source",
    "cli-reverse-prompt-boundary",
    "cli-typed-transcript-boundary",
    "checkpoint-fresh-resume-same-identity",
    "checkpoint-opaque-state-owner",
}

REJECTION_IDS = {
    "template-source-over-limit",
    "template-output-over-limit",
    "template-message-count-over-limit",
    "template-kwargs-key-over-limit",
    "template-kwargs-bytes-over-limit",
    "template-kwargs-depth-over-limit",
    "template-recursion-over-limit",
    "template-fuel-over-limit",
    "template-unknown-variable",
    "template-unknown-filter-function",
    "template-forbidden-include-import",
    "template-digest-missing-or-wrong",
    "template-file-symlink-special-race",
    "reasoning-budget-zero",
    "reasoning-budget-over-limit",
    "reasoning-disabled-with-budget",
    "reasoning-closing-sequence-max-output-insufficient",
    "reasoning-grammar-all-candidates-forbidden",
    "reasoning-unsupported-gemma-raw-text",
    "reasoning-anthropic-thinking-unsupported",
    "cli-prompt-source-conflict",
    "cli-prompt-file-stdin-conflict",
    "cli-prompt-file-invalid",
    "cli-prompt-file-over-limit",
    "cli-reverse-prompt-count-over-limit",
    "cli-reverse-prompt-bytes-over-limit",
    "cli-reverse-prompt-not-stop",
    "cli-transcript-message-over-limit",
    "cli-transcript-bytes-over-limit",
    "cli-transcript-unknown-role",
    "checkpoint-identity-mismatch",
    "checkpoint-corrupt-truncated",
    "checkpoint-quota-over-limit",
    "forbidden-template-process-network",
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
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {value}")),
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
    require(isinstance(positive, list) and len(positive) == len(POSITIVE_IDS), "positive case count changed")
    require(isinstance(rejection, list) and len(rejection) == len(REJECTION_IDS), "rejection case count changed")

    def rows(values: list[Any], label: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for row in values:
            item = _expect_dict(row, f"{label} case must be an object")
            case_id = item.get("id")
            require(isinstance(case_id, str) and case_id, f"{label} case id missing")
            require(case_id not in result, f"duplicate case id: {case_id}")
            require(isinstance(item.get("input"), dict), f"{case_id}: input must be an object")
            require(isinstance(item.get("expected"), dict), f"{case_id}: expected must be an object")
            result[case_id] = item
        return result

    positive_rows = rows(positive, "positive")
    rejection_rows = rows(rejection, "rejection")
    require(set(positive_rows) == POSITIVE_IDS, "positive case identity set changed")
    require(set(rejection_rows) == REJECTION_IDS, "rejection case identity set changed")
    for case_id, row in positive_rows.items():
        expected = row["expected"]
        require(expected.get("result") == "accepted", f"{case_id}: positive result changed")
    for case_id, row in rejection_rows.items():
        expected = row["expected"]
        require(expected.get("admission") == "before_gpu_admission", f"{case_id}: validation is not pre-GPU admission")
        require(expected.get("status") in {400, 413}, f"{case_id}: rejection status is not 4xx")
        require(isinstance(expected.get("code"), str) and expected["code"], f"{case_id}: rejection code missing")
    return positive_rows, rejection_rows


def validate(document: Any, schema: Any) -> None:
    root = _expect_dict(document, "fixture root must be an object")
    schema_root = _expect_dict(schema, "schema root must be an object")
    _expect(schema_root.get("$schema"), "https://json-schema.org/draft/2020-12/schema", "schema draft changed")
    _expect(schema_root.get("$id"), "https://sllm.dev/schema/phase44-template-reasoning-cli-v1.schema.json", "schema id changed")
    _expect(root.get("$schema"), schema_root["$id"], "fixture schema reference changed")
    _expect(root.get("schema_version"), "sllm-phase44-template-reasoning-cli-v1", "fixture schema version changed")
    _expect(root.get("profile_version"), "sllm-phase44-template-reasoning-cli-v1", "profile version changed")
    _expect(root.get("recorded_at"), "2026-08-22", "recorded date changed")

    pins = _expect_dict(root.get("spec_pins"), "spec_pins missing")
    llama = _expect_dict(pins.get("llama_cpp"), "llama.cpp pin missing")
    _expect(llama.get("release"), LLAMA_RELEASE, "llama.cpp release changed")
    _expect(llama.get("commit"), LLAMA_COMMIT, "llama.cpp commit changed")
    require(re.fullmatch(r"[0-9a-f]{40}", llama["commit"]) is not None, "llama.cpp commit is not a full lowercase SHA-1")
    _expect(llama.get("use"), "behavior-reference-only", "llama.cpp reuse boundary changed")
    minijinja = _expect_dict(pins.get("minijinja"), "MiniJinja pin missing")
    _expect(minijinja.get("version"), MINIJINJA_VERSION, "MiniJinja version changed")
    _expect(minijinja.get("default_features"), False, "MiniJinja default features must remain disabled")
    _expect(minijinja.get("features"), ["builtins", "fuel", "json", "macros", "multi_template", "serde"], "MiniJinja feature allowlist changed")
    _expect(minijinja.get("api_guarantees"), ["per-render-fuel", "runtime-recursion-limit", "strict-undefined"], "MiniJinja safety APIs changed")
    _expect(minijinja.get("disabled_integrations"), ["dynamic-loader", "stack-growth", "custom-syntax", "url", "process", "filesystem"], "MiniJinja disabled integrations changed")

    template = _expect_dict(root.get("template_profile"), "template profile missing")
    for key, value, message in [
        ("id", "sllm-generic-jinja-v1", "template profile id changed"),
        ("provider", "minijinja-2.24.0", "template provider changed"),
        ("reviewed_default", "qwen-reviewed-v1", "reviewed default changed"),
        ("custom_opt_in", True, "custom template opt-in changed"),
        ("admission", "compile-render-tokenize-before-scheduler-gpu", "template admission boundary changed"),
    ]:
        _expect(template.get(key), value, message)
    digest = _expect_dict(template.get("custom_template_digest"), "template digest policy missing")
    _expect(digest, {"required": True, "format": "sha256:<64 lowercase hex>", "algorithm": "sha256", "source_encoding": "utf-8"}, "template digest policy changed")
    require(SHA256.fullmatch("sha256:" + "0" * 64) is not None, "internal sha256 format check failed")
    limits = _expect_dict(template.get("source_limits"), "template limits missing")
    _expect(limits, {"template_source_bytes": 65536, "rendered_output_bytes": 16777216, "messages": 1024, "kwargs_keys": 64, "kwargs_total_bytes": 1048576, "kwargs_depth": 32, "recursion": 32, "fuel_instructions": 1000000}, "template limits changed")
    _expect(template.get("context_fields"), ["messages", "tools", "special_tokens", "add_generation_prompt", "enable_thinking", "reasoning_effort", "custom_kwargs"], "template context changed")
    _expect(template.get("allowed_constructs"), ["interpolation", "if", "elif", "else", "for", "set", "macro"], "template construct allowlist changed")
    _expect(template.get("allowed_filters"), ["default", "first", "join", "last", "length", "list", "lower", "replace", "sort", "trim", "unique", "upper", "tojson"], "template filter allowlist changed")
    _expect(template.get("allowed_tests"), ["defined", "undefined", "none", "boolean", "integer", "float", "string", "sequence", "mapping", "iterable", "number"], "template test allowlist changed")
    _expect(template.get("forbidden_constructs"), ["include", "import", "extends", "dynamic-loader", "custom-syntax", "unrestricted-method", "unrestricted-attribute", "dunder-access", "host-callback"], "forbidden template constructs changed")
    _expect(template.get("forbidden_capabilities"), ["filesystem", "environment", "network", "process", "secret", "credential", "path", "clock", "host-object", "method-callback", "url-integration", "stack-growth", "dynamic-loader", "tool-execution", "mcp-request"], "forbidden template capabilities changed")
    identity = _expect_dict(template.get("identity"), "template identity missing")
    _expect(identity, {"required_fields": ["profile_version", "template_digest", "source_bytes", "kwargs_digest", "rendered_bytes_digest"], "checkpoint_binds": ["template_digest", "profile_version"], "reviewed_default_state_isolation": True}, "template identity changed")
    _expect(_expect_dict(template.get("file_boundary"), "template file boundary missing"), {"reader": "cli-only", "regular_file_only": True, "read_once": True, "reject_symlink": True, "reject_special_file": True, "reject_size_race": True, "reject_invalid_utf8": True, "reject_nul": True}, "template file boundary changed")

    reasoning = _expect_dict(root.get("reasoning_control"), "reasoning control missing")
    _expect(reasoning.get("id"), "sllm-reasoning-control-v1", "reasoning profile id changed")
    _expect(reasoning.get("modes"), ["disabled", "enabled", "template-default"], "reasoning modes changed")
    _expect(reasoning.get("mode_mapping"), {"disabled": "ThinkingModeV1::Disabled", "enabled": "ThinkingModeV1::Enabled", "template-default": "ThinkingModeV1::TemplateDefault"}, "reasoning mode mapping changed")
    _expect(reasoning.get("budget"), {"optional": True, "unit": "generated_reasoning_tokens", "min": 1, "max": 4096, "non_aligned_examples": [3, 127, 2049], "disabled_with_budget": "reject"}, "reasoning budget changed")
    _expect(reasoning.get("control"), {"owner": "frontend-token-selector", "in_generation": ["cancel", "force-close"], "forced_close_uses_selector_mask": True, "forced_close_counts_as_generated": True, "host_token_fallback": False, "post_decode_token_rewrite": False, "separate_decode_loop": False}, "reasoning control ownership changed")
    _expect(reasoning.get("transition"), {"assistant_generation_prefix": "starts-reasoning-active", "early_close": "normal-generation-after-closing-marker", "budget_exhausted": "force-bounded-closing-sequence", "closing_marker": "</think>", "max_output_includes_closing_sequence": True, "all_candidates_forbidden": "reject"}, "reasoning transition changed")
    _expect(reasoning.get("admission"), {"before_gpu_admission": True, "requires_closing_sequence_fit": True, "intersects": ["grammar", "stop", "device-selector", "sampling", "cancellation"], "raw_prompt": "reject", "gemma_raw_text": "reject", "missing_reasoning_marker": "reject", "anthropic_thinking": "reject"}, "reasoning admission changed")
    _expect(reasoning.get("wire_mapping"), {"responses_reasoning_effort": {"low": 1024, "medium": 2048, "high": 4096}, "chat_thinking": "shared-frontend-controller", "anthropic_thinking": "unsupported", "stream_splitter": "existing-reasoning-splitter", "non_stream_and_stream": "same-generation-result"}, "reasoning wire mapping changed")

    cli = _expect_dict(root.get("interactive_cli"), "interactive CLI profile missing")
    _expect(cli.get("command"), "chat", "interactive command changed")
    _expect(cli.get("existing_generate_unchanged"), True, "generate compatibility boundary changed")
    _expect(cli.get("input_mode"), "line-oriented-utf8-no-tty-raw-mode", "interactive input mode changed")
    _expect(cli.get("output_mode"), "versioned-json-lines-v1", "interactive output mode changed")
    sources = ["--prompt", "--message", "--prompt-file", "interactive-stdin"]
    _expect(cli.get("prompt_sources"), sources, "prompt sources changed")
    expected_conflicts = {("--prompt", "--message"), ("--prompt", "--prompt-file"), ("--prompt", "interactive-stdin"), ("--message", "--prompt-file"), ("--message", "interactive-stdin"), ("--prompt-file", "interactive-stdin")}
    actual_conflicts = {tuple(pair) for pair in cli.get("prompt_source_conflicts", [])}
    _expect(actual_conflicts, expected_conflicts, "prompt-source conflict matrix changed")
    _expect(cli.get("stdin_rule"), "interactive-stdin-only-when-no-explicit-prompt-source", "stdin source rule changed")
    _expect(_expect_dict(cli.get("prompt_file"), "prompt-file policy missing"), {"max_bytes": 16777216, "read_once": True, "regular_file_only": True, "reject_symlink": True, "reject_special_file": True, "reject_size_race": True, "reject_invalid_utf8": True, "reject_nul": True}, "prompt-file policy changed")
    _expect(_expect_dict(cli.get("turn_limits"), "turn limits missing"), {"messages": 1024, "message_bytes": 16777216, "transcript_bytes": 16777216, "max_stop_sequences": 4}, "turn limits changed")
    _expect(_expect_dict(cli.get("reverse_prompt"), "reverse prompt policy missing"), {"max_count": 4, "total_bytes": 1048576, "match_area": "visible-output", "boundary": "turn-return", "matched_text_in_next_input": False, "distinct_from_stop": True, "stop_semantics": "generation-finish"}, "reverse-prompt policy changed")
    _expect(_expect_dict(cli.get("typed_transcript"), "typed transcript policy missing"), {"encoding": "canonical-json-utf8", "roles": ["system", "user", "assistant"], "max_messages": 1024, "max_bytes": 16777216, "unknown_roles": "reject", "successful_turn_only": True, "diagnostic_payload_logging": False}, "typed transcript policy changed")
    _expect(_expect_dict(cli.get("checkpoint"), "checkpoint policy missing"), {"owner": "sllm-core-SessionCheckpoint-CheckpointStore", "schema_id": "sllm-session-checkpoint-v1", "atomic_write": True, "directory_mode": "0700", "file_mode": "0600", "checksum": True, "quota": True, "implicit_global_session": False, "mid_generation_resume": False, "identity_fields": ["model_lock_fingerprint", "derived_artifact_identity", "adapter_identity", "renderer_identity", "tokenizer_identity", "target_semantics", "plan_digest", "token_sequence_digest", "kv_encoding", "kv_descriptor_digest", "context_policy_digest"], "conversation_and_opaque_state": "owner-defined", "restore_mismatch": "reject"}, "checkpoint policy changed")
    _expect(_expect_dict(cli.get("checkpoint_limits"), "checkpoint limits missing"), {"checkpoint_bytes": 68719476736, "header_bytes": 4096, "sections": 4096, "identity_field_bytes": 1024, "token_history": 1048576, "conversation_bytes": 16777216}, "checkpoint limits changed")
    _expect(_expect_dict(cli.get("resume_identity"), "resume identity missing"), {"fresh_and_resume_token_sequence": "exact-transcript-template-options", "wrong_model": "reject", "wrong_renderer": "reject", "wrong_tokenizer": "reject", "wrong_target": "reject", "wrong_plan": "reject", "wrong_kv": "reject", "corrupt_or_truncated": "reject", "quota_exceeded": "reject", "cancelled_turn_published": False}, "resume identity policy changed")
    _expect(_expect_dict(cli.get("security"), "CLI security policy missing"), {"prompt_token_secret_metrics": False, "prompt_token_secret_normal_logs": False, "tool_execution": False, "mcp_execution": False}, "CLI security policy changed")

    positive, rejection = _cases(root)
    for case_id, key, expected in [
        ("template-source-limit-exact", "template_source_bytes", 65536),
        ("template-kwargs-depth-exact", "kwargs_depth", 32),
        ("reasoning-enabled-budget-one", "budget", 1),
        ("reasoning-template-default-budget-4096", "budget", 4096),
        ("reasoning-nonaligned-budget", "budget", 2049),
        ("cli-reverse-prompt-boundary", "reverse_prompt_count", 4),
        ("cli-typed-transcript-boundary", "messages", 1024),
    ]:
        _expect(positive[case_id]["input"].get(key), expected, f"{case_id}: boundary changed")
    _expect(positive["cli-reverse-prompt-boundary"]["input"].get("reverse_prompt_bytes"), 1048576, "reverse prompt exact byte boundary changed")
    _expect(positive["cli-typed-transcript-boundary"]["input"].get("transcript_bytes"), 16777216, "transcript exact byte boundary changed")
    for case_id, key, expected in [
        ("template-source-over-limit", "template_source_bytes", 65537),
        ("template-output-over-limit", "rendered_output_bytes", 16777217),
        ("template-message-count-over-limit", "messages", 1025),
        ("template-kwargs-key-over-limit", "kwargs_keys", 65),
        ("template-kwargs-bytes-over-limit", "kwargs_total_bytes", 1048577),
        ("template-kwargs-depth-over-limit", "kwargs_depth", 33),
        ("template-recursion-over-limit", "recursion", 33),
        ("template-fuel-over-limit", "fuel_instructions", 1000001),
        ("reasoning-budget-zero", "budget", 0),
        ("reasoning-budget-over-limit", "budget", 4097),
        ("cli-reverse-prompt-count-over-limit", "reverse_prompt_count", 5),
        ("cli-reverse-prompt-bytes-over-limit", "reverse_prompt_bytes", 1048577),
        ("cli-transcript-message-over-limit", "messages", 1025),
        ("cli-transcript-bytes-over-limit", "transcript_bytes", 16777217),
        ("checkpoint-quota-over-limit", "checkpoint_bytes", 68719476737),
    ]:
        _expect(rejection[case_id]["input"].get(key), expected, f"{case_id}: rejection boundary changed")
    _expect(rejection["cli-prompt-source-conflict"]["input"].get("sources"), ["--prompt", "--prompt-file"], "prompt conflict case changed")
    _expect(rejection["cli-prompt-file-stdin-conflict"]["input"].get("sources"), ["--prompt-file", "interactive-stdin"], "stdin conflict case changed")
    for case_id in ("checkpoint-identity-mismatch", "checkpoint-corrupt-truncated"):
        _expect(rejection[case_id]["expected"].get("admission"), "before_gpu_admission", f"{case_id}: checkpoint admitted too early")


def main() -> int:
    fixture_path = Path(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else FIXTURE
    schema_path = Path(__import__("sys").argv[2]) if len(__import__("sys").argv) > 2 else SCHEMA
    validate(load(fixture_path), load(schema_path))
    print("Phase44 template/reasoning/interactive CLI fixture/schema: PASS")
    print("llama.cpp b10453, MiniJinja 2.24.0, sandbox limits, reasoning control, source conflicts, and checkpoint identity are fixed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
