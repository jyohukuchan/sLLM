#!/usr/bin/env python3
"""Validate detached GPU metadata correction for a bounded AQ4 path oracle.

The original v1 output is never changed.  This validator reconstructs the raw
run hashes, device mapping, resource samples, service recovery identity, and
source-token replay binding before accepting a corrected v2 attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORACLE = _load("qwen35_aq4_p2_oracle_attestation", "qwen35_aq4_p2_oracle.py")
VALIDATE = _load("validate_qwen35_aq4_p2_oracle_attestation", "validate-qwen35-aq4-p2-oracle.py")


def _sha(path: Path) -> str:
    return ORACLE.sha256_file(path)


def _rust_replay_sha(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"ullm.qwen35_aq4.calibration_replay.v1\0")
    digest.update(len(token_ids).to_bytes(8, "little"))
    for token_id in token_ids:
        digest.update(token_id.to_bytes(8, "little"))
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = ORACLE.load_json(path)
    if not isinstance(value, dict):
        raise ORACLE.OracleError(f"{path} must contain an object")
    return value


def _validate_raw_root(raw_root: Path, expected: dict[str, str]) -> None:
    sums = raw_root / "SHA256SUMS"
    if _sha(sums) != expected.get("SHA256SUMS"):
        raise ORACLE.OracleError("raw evidence SHA256SUMS digest differs")
    for name, digest in expected.items():
        if name == "SHA256SUMS":
            continue
        path = raw_root / name
        if not path.is_file() or path.is_symlink() or _sha(path) != digest:
            raise ORACLE.OracleError(f"raw evidence hash differs: {name}")


def _validate_device(raw_root: Path, command: dict[str, Any]) -> dict[str, Any]:
    if command.get("device_index") != 1 or command.get("physical_gpu_index") != 2 or command.get("gfx_architecture", command.get("rocm_architecture")) != "gfx1201":
        raise ORACLE.OracleError("declared GPU mapping is not runtime-index 1 -> physical GPU 2 gfx1201")
    if command.get("environment") != {"HIP_VISIBLE_DEVICES": "1", "ULLM_HIP_VISIBLE_DEVICES": "1"}:
        raise ORACLE.OracleError("HIP visible-device environment is not exact")
    devices = (raw_root / "devices.txt").read_text(encoding="utf-8")
    if devices.count("gfx1201") != 1 or "GPU[2]" not in devices:
        raise ORACLE.OracleError("ROCm device evidence does not identify one GPU[2] gfx1201")
    monitor = (raw_root / "monitor.log").read_text(encoding="utf-8")
    if "--device-index 1" not in monitor or "total=34208743424" not in monitor:
        raise ORACLE.OracleError("monitor evidence does not bind runtime device index and VRAM total")
    samples = [(int(used), float(power)) for used, power in re.findall(r"used=(\d+) power=([0-9.]+)", monitor)]
    if not samples:
        raise ORACLE.OracleError("monitor evidence has no resource sample")
    used, power = max(samples, key=lambda item: item[0])
    if used != 7343022080 or power != 21.0:
        raise ORACLE.OracleError("monitor peak resource sample differs")
    return {"physical_gpu_index": 2, "runtime_device_index": 1, "gfx_architecture": "gfx1201", "vram_total_bytes": 34208743424, "vram_baseline_bytes": 87384064, "vram_peak_bytes": used, "power_peak_watts": power}


def _validate_replay(source_root: Path, path_root: Path, cases_path: Path) -> dict[str, Any]:
    source = ORACLE.validate_manifest(source_root, expected_kind="independent_source")
    path = ORACLE.validate_manifest(path_root, expected_kind="same_artifact_all_m1")
    cases_doc = _load_json(cases_path)
    cases = cases_doc.get("cases")
    if not isinstance(cases, list):
        raise ORACLE.OracleError("cases JSON has no cases list")
    source_rows: dict[str, list[int]] = {}
    for row in ORACLE.payload_records(source_root, source):
        source_rows.setdefault(row["case_id"], []).append(row["greedy_token_id"])
    runtime = _load_json(path_root / "runtime.json")
    binding = runtime.get("source_replay", {}).get("cases")
    if not isinstance(binding, list):
        raise ORACLE.OracleError("path runtime has no per-case source replay binding")
    observed = {item.get("case_id"): item for item in binding if isinstance(item, dict)}
    case_results = []
    for case in source["cases"]:
        case_id = case["case_id"]
        case_input = next((item for item in cases if item.get("case_id") == case_id), None)
        if case_input is None or ORACLE.canonical_token_ids_hash(case_input.get("prompt_token_ids", [])) != case["prompt_token_ids_sha256"]:
            raise ORACLE.OracleError(f"case prompt token hash differs for {case_id}")
        tokens = source_rows.get(case_id, [])
        item = observed.get(case_id)
        expected_sha = _rust_replay_sha(tokens)
        expected_contexts = [
            {"step": step, "length": len(case_input["prompt_token_ids"]) + step, "token_ids_sha256": ORACLE.canonical_token_ids_hash(case_input["prompt_token_ids"] + tokens[:step])}
            for step in range(case["step_count"])
        ]
        ok = bool(item and item.get("length") == len(tokens) == case["step_count"] and item.get("source_sequence_sha256") == expected_sha and item.get("contexts") == expected_contexts)
        case_results.append({"case_id": case_id, "length": len(tokens), "source_sequence_sha256": expected_sha, "contexts": expected_contexts, "position_semantics": "step_i_uses_prompt_plus_greedy_tokens_before_i", "exact": ok})
    return {"status": "valid" if all(item["exact"] for item in case_results) else "blocked", "cases": case_results}


def validate(args: argparse.Namespace) -> dict[str, Any]:
    attestation = _load_json(args.attestation)
    if attestation.get("schema_version") != "ullm.qwen35_aq4_path_oracle_attestation.v1":
        raise ORACLE.OracleError("attestation schema is invalid")
    raw = attestation.get("raw_evidence")
    if not isinstance(raw, dict) or raw.get("root") != args.raw_root.name:
        raise ORACLE.OracleError("attestation raw evidence root differs")
    _validate_raw_root(args.raw_root, raw["files"])
    device = _validate_device(args.raw_root, attestation["execution"])
    try:
        base_report = VALIDATE.validate_oracle(args.base_path, "path")
    except ORACLE.OracleError as error:
        # v1 is intentionally retained as immutable raw evidence. Its old
        # runtime sidecar cannot satisfy the corrected production runtime schema.
        base_manifest = ORACLE.validate_manifest(args.base_path, expected_kind="path")
        base_report = {
            "schema_version": "ullm.qwen35_aq4_p2_oracle_validator.v1",
            "status": "valid",
            "oracle_kind": base_manifest["oracle_kind"],
            "manifest_sha256": _sha(args.base_path / "manifest.json"),
            "payload_sha256": base_manifest["payload"]["sha256"],
            "record_count": base_manifest["payload"]["record_count"],
            "usable_as_path_evidence": False,
            "promotion_eligible": False,
            "blockers": [f"metadata_invalid: {error}"],
        }
    path_report = VALIDATE.validate_oracle(args.path, "path")
    if base_report["manifest_sha256"] != attestation["base_path"]["manifest_sha256"] or base_report["payload_sha256"] != attestation["base_path"]["payload_sha256"] or _sha(args.base_path / "runtime.json") != attestation["base_path"]["runtime_sha256"]:
        raise ORACLE.OracleError("base path oracle hash binding differs")
    if path_report["manifest_sha256"] != attestation["corrected_path"]["manifest_sha256"] or path_report["payload_sha256"] != attestation["corrected_path"]["payload_sha256"] or _sha(args.path / "runtime.json") != attestation["corrected_path"]["runtime_sha256"]:
        raise ORACLE.OracleError("corrected path oracle hash binding differs")
    runtime = _load_json(args.path / "runtime.json")
    base_runtime = _load_json(args.base_path / "runtime.json")
    base_metadata_invalid = base_runtime.get("device") == "cpu"
    if runtime.get("device_kind") != "gpu" or runtime.get("device_index") != 1 or runtime.get("visible_devices") != "1" or runtime.get("evidence_scope") != "production_gpu":
        raise ORACLE.OracleError("corrected runtime device metadata differs")
    replay = _validate_replay(args.source_path, args.path, args.cases)
    comparison = ORACLE.compare_payloads(args.source_path, ORACLE.validate_manifest(args.source_path, expected_kind="source"), args.path, ORACLE.validate_manifest(args.path, expected_kind="path"))
    policy = attestation.get("policy_audit")
    if not isinstance(policy, dict) or policy.get("status") != "blocked_unbound" or policy.get("values") is not None:
        raise ORACLE.OracleError("AQ4 P2 threshold policy is unexpectedly bound or malformed")
    for key in ("template_path", "prefill_validation_spec_path", "threshold_audit_path"):
        policy_path = ROOT / policy[key]
        hash_key = {"template_path": "template_sha256", "prefill_validation_spec_path": "prefill_validation_spec_sha256", "threshold_audit_path": "threshold_audit_sha256"}[key]
        if _sha(policy_path) != policy[hash_key]:
            raise ORACLE.OracleError(f"policy audit source hash differs: {policy_path}")
    blockers = []
    if base_metadata_invalid:
        blockers.append("base_v1_metadata_invalid: runtime.json declared cpu for the GPU execution")
    if replay["status"] != "valid":
        blockers.append("replay_step_alignment_invalid")
    if not comparison["greedy_token_exact"] or not comparison["topk_exact"] or not comparison["hidden_sample_within_atol"] or not comparison["logit_sample_within_atol"]:
        blockers.append("source_comparison_bounded_agreement_failed")
    if not comparison["hidden_sample_shape_exact"] or not comparison["logit_sample_shape_exact"]:
        blockers.append("source_comparison_bounded_shape_failed")
    blockers.append("policy_missing: no hash-bound AQ4 P2 bounded relative-L2/cosine/top-k policy")
    blockers.append("path_regression_requires_exact_same_artifact_all_m1_contract")
    return {"schema_version": "ullm.qwen35_aq4_path_oracle_attestation_validator.v1", "status": "valid_with_blockers", "metadata_valid": not base_metadata_invalid, "base_path": base_report, "corrected_path": path_report, "execution": {"device": device, "service_recovery": attestation["execution"]["service_recovery"]}, "step_alignment": replay, "source_comparison": comparison, "path_regression": {"status": "diagnostic_only", "all_m1": True, "exact_greedy": comparison["greedy_token_exact"], "exact_topk": comparison["topk_exact"]}, "policy_audit": policy, "blockers": blockers}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(validate(args), ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, ORACLE.OracleError) as error:
        print(f"Qwen3.5 AQ4 P2 path attestation validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
