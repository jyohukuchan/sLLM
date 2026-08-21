#!/usr/bin/env python3
"""Fail-closed semantic validator for the compact Phase 45 GPU summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "ci/matrix/phase45-adapter-lifecycle-gpu-summary-v1.json"
SCHEMA = ROOT / "ci/schema/phase45-adapter-lifecycle-gpu-summary-v1.schema.json"


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {value}")),
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(document: Any, schema: Any) -> None:
    require(isinstance(document, dict), "summary root must be an object")
    require(isinstance(schema, dict), "schema root must be an object")
    require(document.get("$schema") == schema.get("$id"), "summary schema reference changed")
    require(schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema", "schema draft changed")
    require(document.get("schema_version") == "sllm-phase45-adapter-lifecycle-gpu-summary-v1", "summary version changed")
    require(document.get("recorded_at") == "2026-08-22", "recorded date changed")
    require(document.get("profile") == "sllm-phase45-adapter-lifecycle-v1", "profile binding changed")
    require(document.get("identity_prefixes") == {
        "model_lock": "f143d7", "gguf": "c571c5", "derived_plan": "a3b79a", "prompt": "8c0b75"
    }, "identity prefixes changed")
    require(document.get("cases") == {
        "disabled": {"logits_prefix": "d76c95", "dispatches": 492},
        "lora": {"logits_prefix": "ad5b9f", "dispatches": 497},
        "control": {"logits_prefix": "a06a90", "dispatches": 495},
        "combined": {"logits_prefix": "5c777f", "dispatches": 500},
    }, "case evidence changed")
    require(document.get("case_target_coverage") == "disabled-lora-control-combined-each-target", "case target coverage changed")
    require(document.get("repeatability") == "each-case-bitwise-identical-two-runs", "repeatability changed")
    targets = document.get("targets")
    require(isinstance(targets, list) and len(targets) == 2, "target count changed")
    expected = {
        "gfx1030": ("V620", 16588),
        "gfx1201": ("R9700", 18001),
    }
    seen: set[str] = set()
    for target in targets:
        require(isinstance(target, dict), "target row must be an object")
        name = target.get("target")
        require(name in expected and name not in seen, "target identity changed")
        seen.add(name)
        device, elapsed = expected[name]
        require(target.get("device") == device and target.get("elapsed_ms") == elapsed, f"{name}: timing changed")
        require(target.get("release_build") is True, f"{name}: release-build claim changed")
        require(target.get("hip_only") is True, f"{name}: HIP-only claim changed")
        require(target.get("fallback") is False, f"{name}: fallback claim changed")
        require(target.get("resident_bytes") == 8411592192, f"{name}: resident bytes changed")
        require(target.get("request_workspace_baseline_restored") is True, f"{name}: baseline claim changed")
        require(target.get("pre_allocations") == 0 and target.get("final_allocations") == 0, f"{name}: allocation cleanup changed")
        require(target.get("retryable") == 0 and target.get("quarantine") == 0, f"{name}: lifecycle cleanup changed")
        require(target.get("broadcast_add") == {"m_values": [1, 3], "h": 17, "mismatch": 0, "cleanup": "pass"}, f"{name}: BroadcastAdd evidence changed")
    require(seen == set(expected), "target coverage changed")
    require(document.get("gfx942") == "compile-only-or-deferred-no-runtime-pass", "gfx942 boundary changed")
    require(document.get("raw_artifacts_tracked") is False, "raw artifact tracking changed")


def main() -> int:
    try:
        validate(load(SUMMARY), load(SCHEMA))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"phase45 GPU summary: FAIL: {error}")
        return 1
    print("phase45 GPU summary: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
