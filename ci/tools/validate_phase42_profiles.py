#!/usr/bin/env python3
"""Dependency-free validator for the Phase 42 profile fixture.

This is intentionally narrower than a general JSON-Schema implementation.  It
checks the closed identity and boundary matrix that must not drift, while the
checked-in Draft 2020-12 schema remains the machine-readable description for
consumers that provide a full schema validator.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/phase42_profiles_v1.json"
SCHEMA = ROOT / "ci/schema/phase42-profile-v1.schema.json"
OPENAI_PIN = "117ce5680e4269f6656a4fd70d28f9755630d938"
LLAMA_PIN = "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70"
PROFILE_IDS = {
    "openai-completions-v1",
    "openai-embeddings-v1",
    "sllm-rerank-v1",
    "sllm-token-utilities-v1",
    "sllm-infill-v1",
}
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(document: object, schema: object) -> None:
    require(isinstance(document, dict), "fixture root must be an object")
    require(isinstance(schema, dict), "schema root must be an object")
    require(schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema", "schema draft changed")
    require(schema.get("$id") == "https://sllm.dev/schema/phase42-profile-v1.schema.json", "schema id changed")
    require(document.get("schema_version") == "sllm-phase42-profile-fixture-v1", "fixture schema version changed")
    require(document.get("profile_version") == "sllm-inference-endpoints-v1", "profile version changed")

    pins = document.get("spec_pins")
    require(isinstance(pins, dict), "spec_pins is missing")
    require(pins.get("openai_openapi_version") == "2.3.0", "OpenAI OpenAPI version changed")
    require(pins.get("openai_openapi_commit") == OPENAI_PIN, "OpenAI OpenAPI pin changed")
    require(pins.get("llama_cpp_commit") == LLAMA_PIN, "llama.cpp pin changed")
    require(pins.get("llama_cpp_release") == "b10453", "llama.cpp release changed")
    require(SHA40.fullmatch(OPENAI_PIN) is not None and SHA40.fullmatch(LLAMA_PIN) is not None, "pins are not full SHA-1 values")

    identity = document.get("semantic_identity")
    require(isinstance(identity, dict), "semantic_identity is missing")
    tokenizer = identity.get("tokenizer")
    require(isinstance(tokenizer, dict), "tokenizer identity is missing")
    require(tokenizer.get("utility_version") == "tokenizer-utility-v1", "tokenizer utility version changed")
    require(tokenizer.get("special_token_policy") == "model-default-no-client-override", "special-token policy changed")
    require(tokenizer.get("byte_fallback") == "lossless-byte-array", "byte fallback semantics changed")
    require(tokenizer.get("empty_input") == "accepted", "empty tokenizer input semantics changed")
    require(tokenizer.get("max_input_bytes") == 16 * 1024 * 1024, "tokenizer byte bound changed")
    require(tokenizer.get("max_tokens") == 1_048_576, "tokenizer token bound changed")

    template = identity.get("template")
    require(isinstance(template, dict) and template.get("template_digest_required") is True, "template digest is not required")
    require(template.get("arbitrary_jinja") is False and template.get("custom_kwargs") is False, "arbitrary template execution enabled")

    embedding = identity.get("embedding")
    require(isinstance(embedding, dict), "embedding identity is missing")
    require(embedding.get("pooling") == "arithmetic-mean-over-final-hidden-rows", "embedding pooling changed")
    require(embedding.get("normalization") == "l2", "embedding normalization changed")
    require(embedding.get("output_dtype") == "float32", "embedding dtype changed")
    require(embedding.get("multimodal") is False, "multimodal embedding scope changed")

    rerank = identity.get("rerank")
    require(isinstance(rerank, dict), "rerank identity is missing")
    require(rerank.get("score") == "l2-normalized-query-document-dot-product", "rerank score semantics changed")
    require(rerank.get("higher_is_better") is True, "rerank ordering changed")
    require(rerank.get("tie_break") == "original_document_index", "rerank tie semantics changed")
    require(rerank.get("top_n") == "1..=document_count-no-clamp", "rerank top_n semantics changed")

    infill = identity.get("infill")
    require(isinstance(infill, dict), "infill identity is missing")
    require(infill.get("production_status") == "unsupported-until-verified-template", "production infill capability changed")
    require(infill.get("fallback") == "none", "infill fallback semantics changed")
    require(infill.get("mi300x") == "deferred-until-exact-gfx942-runtime", "MI300X scope changed")

    profiles = document.get("profiles")
    require(isinstance(profiles, list) and len(profiles) == len(PROFILE_IDS), "profile count changed")
    by_id: dict[str, dict[str, object]] = {}
    for profile in profiles:
        require(isinstance(profile, dict), "profile row must be an object")
        profile_id = profile.get("id")
        require(isinstance(profile_id, str) and profile_id in PROFILE_IDS, f"unknown profile id: {profile_id!r}")
        require(profile_id not in by_id, f"duplicate profile id: {profile_id}")
        require(profile.get("method") == "POST", f"{profile_id}: method changed")
        require(isinstance(profile.get("request"), dict), f"{profile_id}: request matrix missing")
        require(isinstance(profile.get("response"), dict), f"{profile_id}: response matrix missing")
        require(isinstance(profile.get("limits"), dict), f"{profile_id}: limits missing")
        require(isinstance(profile.get("rejections"), list) and profile["rejections"], f"{profile_id}: rejection matrix missing")
        by_id[profile_id] = profile
    require(set(by_id) == PROFILE_IDS, "profile id set changed")

    completion = by_id["openai-completions-v1"]
    require(completion["endpoint"] == "/v1/completions", "completion endpoint changed")
    require(completion["normative_pin"] == f"openai-openapi-{OPENAI_PIN}", "completion normative pin changed")
    require(set(completion["request"]["prompt_shapes"]) == {"string", "string_array", "token_array", "token_array_array"}, "completion prompt shapes changed")
    require(completion["request"]["max_tokens"] == {"min": 1, "max": 4096, "default": 256}, "completion max_tokens bounds changed")
    require(completion["request"]["n"] == {"min": 1, "max": 8, "default": 1}, "completion n bounds changed")
    require(completion["request"]["logprobs"] == {"min": 0, "max": 5}, "completion logprobs bounds changed")

    embeddings = by_id["openai-embeddings-v1"]
    require(embeddings["endpoint"] == "/v1/embeddings", "embedding endpoint changed")
    require(embeddings["normative_pin"] == f"openai-openapi-{OPENAI_PIN}", "embedding normative pin changed")
    require(set(embeddings["request"]["input_shapes"]) == {"string", "string_array", "token_array", "token_array_array"}, "embedding input shapes changed")
    require(embeddings["request"]["pooling"] == "mean" and embeddings["request"]["normalization"] == "l2", "embedding hidden pooling changed")
    require(embeddings["request"]["dimensions"] == {"model_lock_dimension": True, "match_required": True}, "embedding dimension semantics changed")
    require(embeddings["limits"]["input_items"] == 256, "embedding input item bound changed")

    rerank_profile = by_id["sllm-rerank-v1"]
    require(rerank_profile["endpoint"] == "/v1/rerank", "rerank endpoint changed")
    require(rerank_profile["compatibility"] == "sllm-native-not-openai", "rerank compatibility claim changed")
    require(rerank_profile["request"]["documents"]["max_items"] == 256, "rerank document bound changed")
    require(rerank_profile["request"]["top_n"]["clamp"] is False, "rerank top_n clamp enabled")

    utility = by_id["sllm-token-utilities-v1"]
    require(utility["request"]["gpu_execution"] is False, "token utility GPU scope changed")
    require(utility["request"]["template"] == "verified-template-only", "template scope changed")
    require(utility["endpoint"] == "/v1/tokenize|/v1/detokenize|/v1/apply-template|/v1/input-tokens", "token utility endpoint set changed")

    infill_profile = by_id["sllm-infill-v1"]
    require(infill_profile["endpoint"] == "/v1/infill", "infill endpoint changed")
    require(infill_profile["compatibility"] == "sllm-native-not-openai", "infill compatibility claim changed")
    require("unsupported_model_capability" in infill_profile["rejections"], "infill capability rejection missing")

    cases = document.get("cases")
    require(isinstance(cases, dict), "cases are missing")
    positive = cases.get("positive")
    negative = cases.get("negative")
    require(isinstance(positive, list) and len(positive) >= 6, "positive boundary matrix is incomplete")
    require(isinstance(negative, list) and len(negative) >= 7, "negative boundary matrix is incomplete")
    ids = [case.get("id") for case in positive + negative if isinstance(case, dict)]
    require(len(ids) == len(set(ids)), "case IDs must be unique")
    require(all(isinstance(case, dict) and isinstance(case.get("request"), dict) for case in positive + negative), "case requests must be objects")
    for case in negative:
        require(isinstance(case.get("error"), dict), f"negative case {case.get('id')!r} has no error")
        require(case["error"].get("status") in {400, 413}, f"negative case {case.get('id')!r} status is not 4xx")
        require(case["error"].get("code") in {"invalid_json", "invalid_value", "unsupported_parameter", "request_too_large"}, f"negative case {case.get('id')!r} error code is not pinned")

    required_negative = {"completion-unknown-field", "completion-nonfinite", "embedding-mixed-input", "rerank-empty-documents", "template-unverified-model", "infill-production-unsupported", "mi300x-deferred"}
    require({case["id"] for case in negative} == required_negative, "negative case identity set changed")
    unknown_case = next(case for case in negative if case["id"] == "completion-unknown-field")
    require(unknown_case["error"] == {"status": 400, "code": "invalid_value", "param": "mystery"}, "unknown-field error mapping changed")
    embedding_case = next(case for case in positive if case["id"] == "embedding-base64-dimension")
    require(embedding_case.get("oracle", {}).get("pooling") == "arithmetic-mean-over-final-hidden-rows", "embedding oracle pooling missing")
    rerank_case = next(case for case in positive if case["id"] == "rerank-stable-contract")
    require(rerank_case.get("oracle", {}).get("order") == [0, 1], "rerank stable tie oracle missing")


def main() -> int:
    validate(load(FIXTURE), load(SCHEMA))
    print("Phase42 profile fixture/schema: PASS")
    print("Implementation/runtime and RDNA exact-GPU acceptance are tracked separately; MI300X runtime is deferred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
