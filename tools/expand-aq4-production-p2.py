#!/usr/bin/env python3
"""Deterministically expand the AQ4 P2 planning manifest into case records.

This tool is planning/evidence preparation only.  It never starts a worker and
does not infer missing identity or oracle hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

HASH_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
MAX_BYTES = 4 * 1024 * 1024


class ExpansionError(ValueError):
    pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ExpansionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ExpansionError(f"{label} must be a bounded regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(ExpansionError(f"non-finite number: {value}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExpansionError(f"cannot parse {label}: {error}") from error


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ExpansionError(f"identity file is unavailable: {path}")
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or ID_RE.fullmatch(value) is None:
        raise ExpansionError(f"{label} is invalid")
    return value


def controls(stage: dict[str, Any], device_id: str) -> list[str]:
    values = stage.get("controls_by_device", {}).get(device_id, stage.get("controls", []))
    if not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values):
        raise ExpansionError(f"controls are missing for {device_id}")
    return values


def expand_stage(stage: dict[str, Any], device: dict[str, Any]) -> list[dict[str, Any]]:
    stage_id = safe_id(stage.get("stage_id"), "stage_id")
    device_id = safe_id(device.get("device_id"), "device_id")
    device_identity = {key: device.get(key) for key in ("device_id", "backend", "gpu_architecture", "gpu_name")}
    result: list[dict[str, Any]] = []
    prefill = stage.get("prefill", {})
    decode = stage.get("decode", {})
    controls_for_device = controls(stage, device_id)
    production_prompts = prefill.get("production_server_prompt_tokens", [])
    production_contexts = decode.get("production_server_start_context_tokens", [])
    for scope in ("component", "full_model", "production_server"):
        prompts = production_prompts if scope == "production_server" else prefill.get("prompt_tokens", [])
        contexts = production_contexts if scope == "production_server" else decode.get("start_context_tokens", [])
        if scope == "production_server" and not prompts and not contexts:
            continue
        for mode in prefill.get("modes", []):
            phase = "cached_prefix_prefill" if mode == "cached_prefix_chunked" else "cold_prefill"
            # ``cached_prefix_prompt_tokens`` describes the prefixes that the
            # runner may materialize; it does not remove prompt lengths from
            # the normative matrix.  Every mode therefore keeps the same
            # prompt axis so the manifest's expected counts remain exact.
            phase_prompts = prompts
            for prompt_tokens in phase_prompts:
                for requested_m in prefill.get("requested_m", []):
                        for control_id in controls_for_device:
                            case_id = f"p2-{stage_id}-{scope}-{phase}-{mode}-n{prompt_tokens}-m{requested_m}-{device_id}-{control_id}"
                            result.append({
                                "case_id": safe_id(case_id, "case_id"), "stage_id": stage_id, "stage_order": stage.get("order"), "scope": scope, "phase": phase, "mode": mode,
                                "prompt_tokens": prompt_tokens, "cached_prefix_tokens": 0 if phase == "cold_prefill" else (prefill.get("cached_prefix_tokens", [0])[0]), "context_tokens": 0, "decode_start_tokens": 0,
                                "prefill_requested_m": requested_m, "decode_request_count": 0, "device": device_identity, "control_id": control_id, "format_id": "AQ4_0" if control_id == "aq4_0_target" else "SQ8_0" if control_id == "sq8_0_cross_format" else "REFERENCE", "path_oracle_case_id": None, "path_oracle_result_sha256": None,
                            })
        for context_tokens in contexts:
            for control_id in controls_for_device:
                case_id = f"p2-{stage_id}-{scope}-decode-n{context_tokens}-requests{decode.get('request_count', 1)}-{device_id}-{control_id}"
                result.append({
                    "case_id": safe_id(case_id, "case_id"), "stage_id": stage_id, "stage_order": stage.get("order"), "scope": scope, "phase": "decode", "mode": "decode",
                    "prompt_tokens": 0, "cached_prefix_tokens": 0, "context_tokens": context_tokens, "decode_start_tokens": context_tokens, "prefill_requested_m": 0, "decode_request_count": decode.get("request_count", 1), "generated_tokens": decode.get("generated_tokens", 64),
                    "device": device_identity, "control_id": control_id, "format_id": "AQ4_0" if control_id == "aq4_0_target" else "SQ8_0" if control_id == "sq8_0_cross_format" else "REFERENCE", "path_oracle_case_id": None, "path_oracle_result_sha256": None,
                })
    return result


def expand(manifest: dict[str, Any], manifest_sha256: str) -> dict[str, Any]:
    if manifest.get("schema_version") != "ullm.aq4_production_p2_case_manifest.v1": raise ExpansionError("unexpected case manifest schema")
    if manifest.get("status") != "planning_only": raise ExpansionError("case manifest must remain planning_only")
    stages = manifest.get("stages")
    devices = manifest.get("axes", {}).get("devices")
    if not isinstance(stages, list) or not isinstance(devices, list): raise ExpansionError("manifest stages/devices are missing")
    by_id = {safe_id(device.get("device_id"), "device_id"): device for device in devices}
    cases: list[dict[str, Any]] = []
    for stage in sorted(stages, key=lambda value: value.get("order", 0)):
        for device_id in stage.get("devices", []):
            if device_id not in by_id: raise ExpansionError(f"stage references unknown device: {device_id}")
            cases.extend(expand_stage(stage, by_id[device_id]))
    cases.sort(key=lambda value: value["case_id"])
    seen = set()
    for case in cases:
        if case["case_id"] in seen: raise ExpansionError("duplicate expanded case id")
        seen.add(case["case_id"])
    # Link non-all-M1 prefill cases to the same artifact/control/device/path case.
    index = {(c["stage_id"], c["scope"], c["phase"], c["prompt_tokens"], c["device"]["device_id"], c["control_id"]): c for c in cases if c["phase"] == "cold_prefill" and c["mode"] == "all_m1" and c["prefill_requested_m"] == 1}
    for case in cases:
        if case["phase"] in {"cold_prefill", "cached_prefix_prefill"} and case["mode"] != "all_m1":
            linked = index.get((case["stage_id"], case["scope"], "cold_prefill", case["prompt_tokens"], case["device"]["device_id"], case["control_id"]))
            if linked is None: raise ExpansionError(f"missing all-M1 path oracle case for {case['case_id']}")
            case["path_oracle_case_id"] = linked["case_id"]
    expected_by_stage = {stage["stage_id"]: stage.get("expected_case_count", {}) for stage in stages}
    expected_total = sum(int(values.get("total", 0)) for values in expected_by_stage.values())
    if len(cases) != expected_total:
        raise ExpansionError(f"expanded case count {len(cases)} differs from manifest expected total {expected_total}")
    payload = {"schema_version": "ullm.aq4_production_p2_expanded.v1", "manifest_sha256": manifest_sha256, "manifest_id": manifest.get("manifest_id"), "case_count": len(cases), "cases": cases, "path_oracle_contract": manifest.get("path_oracle_contract"), "expected_case_count": {"by_stage": expected_by_stage, "total": expected_total}, "canonical_case_sha256": sha_bytes(canonical(cases))}
    return payload


def atomic_write(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink(): raise ExpansionError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    raw = (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as target:
        target.write(raw); target.flush(); os.fsync(target.fileno())
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        raw = args.manifest.read_bytes(); manifest = load(args.manifest, "case manifest"); atomic_write(args.output, expand(manifest, sha_bytes(raw))); print(json.dumps({"status": "ok", "case_count": len(json.loads(args.output.read_text())["cases"])})); return 0
    except (ExpansionError, OSError, ValueError) as error:
        print(f"P2 manifest expansion failed: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
