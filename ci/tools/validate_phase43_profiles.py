#!/usr/bin/env python3
"""Dependency-free validator for the Phase 43 protocol profile contract.

The JSON Schema is the interchange description.  This validator intentionally
also pins the semantic identity, limits, closed event graphs, and rejection
matrix so a permissive schema consumer cannot silently widen the profile.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/phase43_protocol_profiles_v1.json"
SCHEMA = ROOT / "ci/schema/phase43-protocol-profile-v1.schema.json"
OPENAI_COMMIT = "010421dcbd0475277ea8c3e6c1e1cbca4659c4bd"
LLAMA_COMMIT = "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70"
PROFILE_IDS = {"openai-responses-v1", "anthropic-messages-v1"}
SHA40 = re.compile(r"^[0-9a-f]{40}$")


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
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {value}")),
    )


def _profile_rows(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = document.get("profiles")
    require(isinstance(profiles, list) and len(profiles) == 2, "profile count must be exactly two")
    rows: dict[str, dict[str, Any]] = {}
    for row in profiles:
        require(isinstance(row, dict), "profile row must be an object")
        profile_id = row.get("id")
        require(profile_id in PROFILE_IDS, f"unknown profile id: {profile_id!r}")
        require(profile_id not in rows, f"duplicate profile id: {profile_id}")
        rows[profile_id] = row
    require(set(rows) == PROFILE_IDS, "profile identity set changed")
    return rows


def _validate_events(profile_id: str, events: dict[str, Any]) -> None:
    require(isinstance(events, dict), f"{profile_id}: events missing")
    initial = events.get("initial")
    terminal = events.get("terminal")
    transitions = events.get("transitions")
    require(isinstance(initial, str) and isinstance(terminal, list) and terminal, f"{profile_id}: event identity missing")
    require(isinstance(transitions, list) and transitions, f"{profile_id}: event transitions missing")
    states = {initial, *terminal}
    seen: set[tuple[str, str]] = set()
    for transition in transitions:
        require(isinstance(transition, dict), f"{profile_id}: malformed transition")
        source, event, target = (transition.get(key) for key in ("from", "event", "to"))
        require(isinstance(source, str) and isinstance(event, str) and isinstance(target, str), f"{profile_id}: incomplete transition")
        require(source in states or source == "in_progress" or source == "message_open" or source == "block_open" or source == "item_open", f"{profile_id}: unknown transition source {source}")
        states.add(target)
        key = (source, event)
        require(key not in seen, f"{profile_id}: duplicate transition {key}")
        seen.add(key)
    require("terminal_after_terminal" in events.get("forbidden", []), f"{profile_id}: terminal closure is not pinned")
    require("success_after_error" in events.get("forbidden", []), f"{profile_id}: error closure is not pinned")
    require("DONE_sentinel" in events.get("forbidden", []), f"{profile_id}: [DONE] sentinel must remain forbidden")


def validate(document: Any, schema: Any) -> None:
    require(isinstance(document, dict), "fixture root must be an object")
    require(isinstance(schema, dict), "schema root must be an object")
    require(schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema", "schema draft changed")
    require(schema.get("$id") == "https://sllm.dev/schema/phase43-protocol-profile-v1.schema.json", "schema id changed")
    require(document.get("$schema") == schema["$id"], "fixture schema reference changed")
    require(document.get("schema_version") == "phase43-protocol-profile-v1", "fixture schema version changed")
    require(document.get("profile_version") == "sllm-phase43-protocol-profiles-v1", "profile version changed")

    pins = document.get("spec_pins")
    require(isinstance(pins, dict), "spec_pins missing")
    openai = pins.get("openai")
    require(isinstance(openai, dict), "OpenAI pin missing")
    require(openai.get("openapi_version") == "2.3.0", "OpenAI OpenAPI version changed")
    require(openai.get("commit") == OPENAI_COMMIT and SHA40.fullmatch(OPENAI_COMMIT), "OpenAI pin changed")
    require(openai.get("operation") == "POST /v1/responses", "OpenAI operation changed")
    anthropic = pins.get("anthropic")
    require(isinstance(anthropic, dict) and anthropic.get("api_version") == "2023-06-01", "Anthropic API version changed")
    llama = pins.get("llama_cpp")
    require(isinstance(llama, dict) and llama.get("release") == "b10453", "llama.cpp release changed")
    require(llama.get("commit") == LLAMA_COMMIT and SHA40.fullmatch(LLAMA_COMMIT), "llama.cpp pin changed")

    limits = document.get("limits")
    require(isinstance(limits, dict), "limits missing")
    expected_limits = {
        "request_body_bytes": 100663296,
        "tool_description_bytes": 16384,
        "tool_schema_bytes": 1048576,
        "call_id_bytes": 256,
        "arguments_bytes": 16777216,
        "result_bytes": 16777216,
        "input_items": 2048,
        "messages": 1024,
        "content_blocks_per_message": 256,
        "text_bytes": 16777216,
        "completion_tokens": 4096,
        "resumable_completion_tokens": 40,
        "stream_delta_bytes": 16384,
        "replay_event_bytes": 65536,
        "replay_session_bytes": 262144,
    }
    for key, value in expected_limits.items():
        require(limits.get(key) == value, f"limit changed: {key}")
    require(limits.get("tool_definitions") == {"min": 1, "max": 128}, "tool definition limit changed")
    require(limits.get("parallel_calls") == {"min": 1, "max": 16}, "parallel call limit changed")
    require(limits.get("stop_sequences") == {"min": 1, "max": 4}, "stop sequence limit changed")
    tool_name = limits.get("tool_name")
    require(isinstance(tool_name, dict) and tool_name.get("max_bytes") == 64 and tool_name.get("pattern") == "^[A-Za-z0-9_-]{1,64}$", "tool name limit changed")

    common = document.get("common_tool_protocol")
    require(isinstance(common, dict), "common tool protocol missing")
    require(common.get("tool_choice") == ["auto", "none", "required", "specific"], "common tool choice changed")
    require(common.get("execution") == "client-owned-result-roundtrip-only", "tool execution boundary changed")
    policy = common.get("parallel_policy")
    require(policy == {"default_max_calls": 16, "false_max_calls": 1, "disable_parallel_tool_use_max_calls": 1}, "parallel policy changed")
    require(common.get("canonical_envelope", {}).get("arguments_are_grammar_constrained") is True, "grammar constraint is not required")

    boundary = document.get("no_execution_boundary")
    require(isinstance(boundary, dict) and boundary.get("status") == "protocol-only", "no-execution boundary changed")
    require(boundary.get("phase47_approval_required") is True, "Phase 47 approval boundary changed")
    forbidden = set(boundary.get("forbidden", []))
    require({"process_spawn", "network_io", "filesystem_io", "credential_resolution", "mcp_request"} <= forbidden, "external execution boundary widened")

    rows = _profile_rows(document)
    responses = rows["openai-responses-v1"]
    require(responses.get("endpoint") == "/v1/responses", "Responses endpoint changed")
    require(responses.get("spec_pin") == f"openai-openapi-{OPENAI_COMMIT}", "Responses normative pin changed")
    require(responses.get("required_headers") == ["content-type"], "Responses required headers changed")
    require(responses.get("optional_headers") == ["authorization"], "Responses optional headers changed")
    require(responses.get("request", {}).get("store", {}).get("accepted") == [False], "Responses store:false requirement changed")
    require("previous_response_id" not in responses.get("request", {}).get("optional", []), "stateful Responses store was enabled")
    require(responses.get("response", {}).get("object") == "response", "Responses object changed")
    require({"id", "created_at", "model", "status", "output", "usage"} <= set(responses.get("response", {}).get("required_fields", [])), "Responses response fields are incomplete")
    require("error" in responses.get("response", {}).get("terminal_events", []), "Responses error terminal changed")
    _validate_events("openai-responses-v1", responses.get("events", {}))

    messages = rows["anthropic-messages-v1"]
    require(messages.get("endpoint") == "/v1/messages", "Anthropic endpoint changed")
    require(messages.get("spec_pin") == "anthropic-2023-06-01", "Anthropic normative pin changed")
    require(set(messages.get("required_headers", [])) == {"content-type", "anthropic-version"}, "Anthropic required headers changed")
    require(messages.get("optional_headers") == ["authorization"], "Anthropic optional headers changed")
    require(messages.get("request", {}).get("anthropic_version") == "2023-06-01", "Anthropic header version changed")
    require(set(messages.get("request", {}).get("message_roles", [])) == {"user", "assistant"}, "Anthropic roles changed")
    require(messages.get("response", {}).get("object") == "message", "Anthropic object changed")
    require({"id", "model", "content", "stop_reason", "stop_sequence", "usage"} <= set(messages.get("response", {}).get("required_fields", [])), "Anthropic response fields are incomplete")
    _validate_events("anthropic-messages-v1", messages.get("events", {}))

    cases = document.get("cases")
    require(isinstance(cases, dict), "cases missing")
    positive = cases.get("positive")
    rejection = cases.get("rejection")
    require(isinstance(positive, list) and len(positive) >= 8, "positive matrix is incomplete")
    require(isinstance(rejection, list) and len(rejection) >= 12, "rejection matrix is incomplete")
    all_cases = positive + rejection
    ids = [case.get("id") for case in all_cases if isinstance(case, dict)]
    require(len(ids) == len(set(ids)) and all(isinstance(case_id, str) and case_id for case_id in ids), "case IDs must be unique")
    require({case["profile"] for case in positive} == PROFILE_IDS, "positive profile coverage is incomplete")
    require({case["profile"] for case in rejection} == PROFILE_IDS, "rejection profile coverage is incomplete")
    for case in positive:
        request = case.get("request", {})
        required = rows[case["profile"]]["request"]["required"]
        require(all(field in request for field in required), f"{case.get('id')}: positive request is incomplete")
    for case in rejection:
        require(case.get("expected", {}).get("admission") == "before_gpu_admission", f"{case.get('id')}: validation is not pre-admission")
        require(case["expected"].get("status") in {400, 413}, f"{case.get('id')}: status is not 4xx")
    expected_rejections = {
        "responses-unknown-field", "responses-duplicate-field", "responses-nonfinite", "responses-unsupported-state", "responses-unknown-input-item",
        "anthropic-version-missing", "anthropic-version-mismatch", "anthropic-tool-result-order", "anthropic-duplicate-tool-result",
        "parallel-call-limit", "request-body-limit", "empty-tools",
    }
    require({case["id"] for case in rejection} == expected_rejections, "rejection case identity set changed")
    body_limit = next(case for case in rejection if case["id"] == "request-body-limit")
    require(body_limit.get("body_bytes") == limits["request_body_bytes"] + 1 and body_limit["expected"]["status"] == 413, "body limit boundary changed")
    require(next(case for case in rejection if case["id"] == "anthropic-version-missing")["expected"].get("param") == "anthropic-version", "Anthropic header rejection changed")
    execution_case = next(case for case in positive if case["id"] == "execution-payload-is-data")
    require(execution_case["expected"].get("execution") == "not-performed", "execution no-op case changed")


def main() -> int:
    fixture_path = Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE
    schema_path = Path(sys.argv[2]) if len(sys.argv) > 2 else SCHEMA
    validate(load(fixture_path), load(schema_path))
    print("Phase43 protocol profiles fixture/schema: PASS")
    print("Responses 2.3.0 and Anthropic 2023-06-01 pins, limits, event closure, and no-execution boundary are fixed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
