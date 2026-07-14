#!/usr/bin/env python3
"""Deterministically expand the AQ4 P2 manifest into hash-bound cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
MAX_JSON_BYTES = 4 * 1024 * 1024


class ExpansionError(ValueError):
    pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        if key in value:
            raise ExpansionError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def load(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ExpansionError("manifest must be a bounded regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(ExpansionError(f"non-finite JSON number: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ExpansionError(f"invalid manifest JSON: {error}") from error
    if not isinstance(value, dict):
        raise ExpansionError("manifest root must be an object")
    return value, raw


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or ID_RE.fullmatch(value) is None:
        raise ExpansionError(f"invalid {label}")
    return value


def nonempty_unique_ints(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value) or len(value) != len(set(value)):
        raise ExpansionError(f"{label} must contain unique positive integers")
    return value


def controls(stage: dict[str, Any], device_id: str) -> list[str]:
    value = stage.get("controls_by_device", {}).get(device_id, stage.get("controls"))
    if not isinstance(value, list) or not value or len(value) != len(set(value)):
        raise ExpansionError(f"controls are invalid for {device_id}")
    return [safe_id(item, "control_id") for item in value]


def case_hash(case: dict[str, Any]) -> str:
    value = json.loads(json.dumps(case))
    value["case_sha256"] = None
    return digest(canonical(value))


def append_prefill(
    target: list[dict[str, Any]], stage: dict[str, Any], device: dict[str, Any],
    control_by_id: dict[str, dict[str, Any]], sampling: Any,
    scope: str, phase: str, mode: str, prompts: list[int], prefixes: list[int],
) -> None:
    prefill = stage["prefill"]
    device_id = device["device_id"]
    for prefix in prefixes:
        for prompt in prompts:
            if prefix + prompt > 4096:
                raise ExpansionError(f"cached-prefix context exceeds 4096: {stage['stage_id']} prefix={prefix} prompt={prompt}")
            for requested_m in prefill["requested_m"]:
                for control_id in controls(stage, device_id):
                    control = control_by_id[control_id]
                    prefix_label = f"-prefix{prefix}" if phase == "cached_prefix_prefill" else ""
                    case_id = f"p2-{stage['stage_id']}-{scope}-{phase}-{mode}{prefix_label}-n{prompt}-m{requested_m}-{device_id}-{control_id}"
                    value = {
                        "case_id": safe_id(case_id, "case_id"), "fixture_id": safe_id(stage.get("fixture_id", case_id), "fixture_id"), "case_sha256": None,
                        "stage_id": stage["stage_id"], "stage_order": stage["order"], "scope": scope,
                        "phase": phase, "mode": mode, "baseline_mode": mode,
                        "prompt_tokens": prompt, "cached_prefix_tokens": prefix,
                        "context_tokens": prefix + prompt, "decode_start_tokens": prefix + prompt if prefill.get("generated_tokens", 0) else 0,
                        "prefill_requested_m": requested_m,
                        "resolved_m": 1 if mode == "all_m1" else requested_m,
                        "request_count": 1, "decode_request_count": 0, "generated_tokens": prefill.get("generated_tokens", 0),
                        "device": {key: device.get(key) for key in ("device_id", "backend", "name", "architecture", "runtime_device_index")},
                        "control_id": control_id,
                        "control": control.get("trace_control", control), "sampling": stage.get("sampling", sampling), "format_id": control["format_id"], "implementation_id": control["implementation_id"],
                        "path_oracle_case_id": None, "path_oracle_result_sha256": None,
                    }
                    value["case_sha256"] = case_hash(value)
                    target.append(value)


def expand_stage(stage: dict[str, Any], device: dict[str, Any], control_by_id: dict[str, dict[str, Any]], sampling: Any) -> list[dict[str, Any]]:
    stage_id = safe_id(stage.get("stage_id"), "stage_id")
    device_id = safe_id(device.get("device_id"), "device_id")
    stage["stage_id"] = stage_id
    device["device_id"] = device_id
    prefill = stage.get("prefill")
    decode = stage.get("decode")
    if not isinstance(prefill, dict) or not isinstance(decode, dict):
        raise ExpansionError(f"{stage_id} phase axes are missing")
    prompts = nonempty_unique_ints(prefill.get("prompt_tokens"), f"{stage_id}.prompt_tokens")
    nonempty_unique_ints(prefill.get("requested_m"), f"{stage_id}.requested_m")
    modes = prefill.get("modes")
    if not isinstance(modes, list) or "all_m1" not in modes or "cold_batched" not in modes:
        raise ExpansionError(f"{stage_id} must include all_m1 and cold_batched")
    cached_enabled = "cached_prefix_chunked" in modes
    cached_prefixes = prefill.get("cached_prefix_tokens", [])
    cached_prompts = prefill.get("cached_prefix_prompt_tokens", [])
    if cached_enabled:
        cached_prefixes = nonempty_unique_ints(cached_prefixes, f"{stage_id}.cached_prefix_tokens")
        cached_prompts = nonempty_unique_ints(cached_prompts, f"{stage_id}.cached_prefix_prompt_tokens")
    elif cached_prompts:
        raise ExpansionError(f"{stage_id} has a cached prefix axis without cached mode")
    result: list[dict[str, Any]] = []
    for scope in prefill.get("scopes", []):
        scope_prompts = prefill.get("production_server_prompt_tokens", []) if scope == "production_server" else prompts
        if not scope_prompts:
            continue
        append_prefill(result, stage, device, control_by_id, sampling, scope, "cold_prefill", "all_m1", scope_prompts, [0])
        append_prefill(result, stage, device, control_by_id, sampling, scope, "cold_prefill", "cold_batched", scope_prompts, [0])
        if cached_enabled:
            scope_cached = [item for item in cached_prompts if item in scope_prompts]
            if not scope_cached:
                raise ExpansionError(f"{stage_id}.{scope} cached-prefix axis is empty")
            append_prefill(result, stage, device, control_by_id, sampling, scope, "cached_prefix_prefill", "all_m1", scope_cached, cached_prefixes)
            append_prefill(result, stage, device, control_by_id, sampling, scope, "cached_prefix_prefill", "cached_prefix_chunked", scope_cached, cached_prefixes)
    contexts = nonempty_unique_ints(decode.get("start_context_tokens"), f"{stage_id}.decode contexts")
    for scope in decode.get("scopes", []):
        scope_contexts = decode.get("production_server_start_context_tokens", []) if scope == "production_server" else contexts
        for context in scope_contexts:
            if context + decode.get("generated_tokens", 0) > 4096:
                raise ExpansionError(f"decode context exceeds 4096: {stage_id} context={context}")
            for control_id in controls(stage, device_id):
                control = control_by_id[control_id]
                case_id = f"p2-{stage_id}-{scope}-decode-n{context}-requests{decode.get('request_count')}-{device_id}-{control_id}"
                value = {
                    "case_id": safe_id(case_id, "case_id"), "fixture_id": safe_id(stage.get("fixture_id", case_id), "fixture_id"), "case_sha256": None,
                    "stage_id": stage_id, "stage_order": stage["order"], "scope": scope,
                    "phase": "decode", "mode": "decode", "baseline_mode": "decode",
                    "prompt_tokens": context, "cached_prefix_tokens": 0, "context_tokens": context,
                    "decode_start_tokens": context, "prefill_requested_m": 0, "resolved_m": 1,
                    "request_count": decode.get("request_count"), "decode_request_count": decode.get("request_count"), "generated_tokens": decode.get("generated_tokens"),
                    "device": {key: device.get(key) for key in ("device_id", "backend", "name", "architecture", "runtime_device_index")},
                    "control_id": control_id,
                    "control": control.get("trace_control", control), "sampling": stage.get("sampling", sampling), "format_id": control["format_id"], "implementation_id": control["implementation_id"],
                    "path_oracle_case_id": None, "path_oracle_result_sha256": None,
                }
                value["case_sha256"] = case_hash(value)
                result.append(value)
    return result


def expand(manifest: dict[str, Any], manifest_sha256: str) -> dict[str, Any]:
    if manifest.get("schema_version") != "ullm.aq4_production_p2_case_manifest.v1" or manifest.get("status") != "planning_only":
        raise ExpansionError("unexpected or executable manifest")
    stages = manifest.get("stages")
    devices = manifest.get("axes", {}).get("devices")
    if not isinstance(stages, list) or not isinstance(devices, list):
        raise ExpansionError("manifest stages/devices are missing")
    device_by_id = {safe_id(item.get("device_id"), "device_id"): item for item in devices}
    controls_axis = manifest.get("axes", {}).get("controls")
    if not isinstance(controls_axis, list): raise ExpansionError("control axis is missing")
    control_by_id = {safe_id(item.get("control_id"), "control_id"): item for item in controls_axis}
    target_contracts = {(item.get("format_id"), item.get("implementation_id")) for item in controls_axis if item.get("role") == "target"}
    binding = manifest.get("identity_binding", {})
    if target_contracts != {(binding.get("format_id"), binding.get("implementation_id"))}:
        raise ExpansionError("target control differs from model implementation contract")
    sampling = manifest.get("identity_binding", {}).get("sampling")
    cases: list[dict[str, Any]] = []
    stage_counts: dict[str, int] = {}
    for stage in sorted(stages, key=lambda item: item.get("order", 0)):
        before = len(cases)
        for device_id in stage.get("devices", []):
            if device_id not in device_by_id:
                raise ExpansionError(f"unknown device: {device_id}")
            cases.extend(expand_stage(stage, device_by_id[device_id], control_by_id, sampling))
        actual = len(cases) - before
        expected = stage.get("expected_case_count", {}).get("total")
        if actual != expected:
            raise ExpansionError(f"{stage['stage_id']} case count {actual} differs from expected {expected}")
        stage_counts[stage["stage_id"]] = actual
    cases.sort(key=lambda item: item["case_id"])
    if len({item["case_id"] for item in cases}) != len(cases):
        raise ExpansionError("duplicate expanded case id")
    for item in cases:
        device = item.get("device", {})
        required = (item.get("implementation_id"), item.get("request_count"), device.get("backend"), device.get("name"), device.get("architecture"), device.get("runtime_device_index"))
        if any(value is None for value in required): raise ExpansionError(f"case execution identity is incomplete: {item.get('case_id')}")
        if item.get("scope") == "production_server" and (item.get("sampling") is None or item.get("control") is None): raise ExpansionError(f"production case sampling/control is incomplete: {item.get('case_id')}")
    oracle_index = {
        (item["stage_id"], item["scope"], item["phase"], item["cached_prefix_tokens"], item["prompt_tokens"], item["prefill_requested_m"], item["device"]["device_id"], item["control_id"]): item
        for item in cases if item["mode"] == "all_m1" and item["phase"] != "decode"
    }
    for item in cases:
        if item["mode"] not in {"cold_batched", "cached_prefix_chunked"}:
            continue
        key = (item["stage_id"], item["scope"], item["phase"], item["cached_prefix_tokens"], item["prompt_tokens"], item["prefill_requested_m"], item["device"]["device_id"], item["control_id"])
        oracle = oracle_index.get(key)
        if oracle is None or oracle["resolved_m"] != 1:
            raise ExpansionError(f"same-state all-M1 oracle is missing for {item['case_id']}")
        item["path_oracle_case_id"] = oracle["case_id"]
        item["case_sha256"] = case_hash(item)
    expected_total = sum(stage_counts.values())
    if len(cases) != expected_total:
        raise ExpansionError("complete case count mismatch")
    return {
        "schema_version": "ullm.aq4_production_p2_expanded.v2", "manifest_id": manifest.get("manifest_id"),
        "manifest_sha256": manifest_sha256, "case_count": len(cases), "stage_case_count": stage_counts,
        "expected_case_count": {"total": expected_total}, "path_oracle_contract": manifest.get("path_oracle_contract"),
        "canonical_case_sha256": digest(canonical(cases)), "cases": cases,
    }


def atomic_write(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise ExpansionError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("xb") as target:
        target.write((json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode())
        target.flush(); os.fsync(target.fileno())
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest, raw = load(args.manifest); value = expand(manifest, digest(raw)); atomic_write(args.output, value)
        print(json.dumps({"status": "ok", "case_count": value["case_count"]}, sort_keys=True)); return 0
    except (ExpansionError, OSError, ValueError) as error:
        print(f"P2 manifest expansion failed: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
