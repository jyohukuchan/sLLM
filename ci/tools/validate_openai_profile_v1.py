#!/usr/bin/env python3
"""Fail-closed static validator for the pinned Chat Completions fixture."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/openai_chat_profile_v1.json"
PIN = "117ce5680e4269f6656a4fd70d28f9755630d938"


def validate(document: object) -> None:
    if not isinstance(document, dict):
        raise ValueError("fixture must be an object")
    if document.get("schema_version") != "openai-chat-profile-fixture-v1":
        raise ValueError("fixture schema version differs")
    official = document.get("official_openapi")
    if not isinstance(official, dict) or official.get("commit") != PIN:
        raise ValueError("official OpenAPI pin differs")
    positive = document.get("positive")
    if not isinstance(positive, dict):
        raise ValueError("positive fixture is absent")
    request = positive.get("request")
    if not isinstance(request, dict) or request.get("temperature") != 0.0 or request.get("n") != 1:
        raise ValueError("positive request does not retain fixed profile boundaries")
    stream = positive.get("stream")
    if not isinstance(stream, dict) or stream.get("terminal") != "[DONE]":
        raise ValueError("SSE terminal fixture differs")
    negative = document.get("negative")
    if not isinstance(negative, list) or len(negative) < 6:
        raise ValueError("negative matrix is incomplete")
    ids = [case.get("case_id") for case in negative if isinstance(case, dict)]
    if len(ids) != len(negative) or len(set(ids)) != len(ids):
        raise ValueError("negative case identities are invalid")
    required = {
        "malformed-json",
        "unsupported-tools",
        "invalid-n",
        "top-logprobs-without-logprobs",
        "unsupported-json-schema-keyword",
        "tool-message",
        "multipart-non-user-content",
        "unknown-model",
    }
    if set(ids) != required:
        raise ValueError("negative matrix differs from the fixed profile subset")


def main() -> int:
    validate(json.loads(FIXTURE.read_text(encoding="utf-8")))
    print("OpenAI Chat Completions profile-v1 fixture: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
