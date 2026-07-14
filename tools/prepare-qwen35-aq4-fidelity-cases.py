#!/usr/bin/env python3
"""Materialize the 24-row split as source-calibration cases and an execution plan.

The existing BF16 exporter accepts its own case schema, while the pre-registered
measurement lane owns ``calibration-cases.jsonl``.  This bridge performs only the
hash-bound translation; it does not load a model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAX_ROWS = 24
CASES_SCHEMA = "ullm.qwen35_aq4_source_calibration_cases.v1"
PLAN_SCHEMA = "ullm.aq4_p2_fidelity_capture_plan.v1"
EXPECTED_SPLIT_MANIFEST_SHA256 = "966878f3d9eb13f5b485825208f8072521724f308f5ee3d8a003b0b051198887"
EXPECTED_POLICY_SHA256 = "302c3219af286a970ddf39ed090021ef102b51b2d188c0ff337f6b9dd04d1a03"
EXPECTED_CALIBRATION_CASES_SHA256 = "20c09f22bb1ca4dfac907de09febddb01ed0228c3f4a17c01efd646491e0983f"
EXPECTED_SERVED_MODEL_MANIFEST_SHA256 = "feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44"
EXPECTED_PACKAGE_MANIFEST_SHA256 = "a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad"
EXPECTED_WORKER_BINARY_SHA256 = "177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d"
EXPECTED_GUARD_SHA256 = "4eafd9bc149792b9c9849fed07a70830a42cf8227b85431130eec8f41708abc0"
EXPECTED_DEVICE_ARCHITECTURE = "gfx1201"
EXPECTED_QUANTIZED_ARTIFACT_REVISION = "aq4-reasoning-v0.1-candidate"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATE = _load("aq4_fidelity_split_validator_for_cases", "validate-aq4-p2-fidelity-holdout.py")
PROTOCOL = _load("aq4_fidelity_protocol_for_cases", "generate-aq4-p2-fidelity-holdout.py")


class PrepareError(ValueError):
    pass


def validate_plan(plan: dict[str, Any], *, expected_split_sha: str, expected_policy_sha: str, expected_calibration_sha: str, expected_served_sha: str, expected_package_sha: str, expected_worker_sha: str, expected_guard_sha: str, expected_device_architecture: str, expected_quantized_revision: str) -> None:
    required = {"schema_version", "status", "promotion_eligible", "row_count", "full_context_step", "one_source_model_load", "one_active_model_load", "split_manifest_sha256", "policy_sha256", "calibration_cases_sha256", "source_cases_sha256", "source", "active", "execution_contract", "expected_active_identity", "bounds"}
    if set(plan) != required or plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "ready_for_source_and_active_capture" or plan.get("promotion_eligible") is not False or plan.get("row_count") != MAX_ROWS or plan.get("full_context_step") != 0 or plan.get("one_source_model_load") is not True or plan.get("one_active_model_load") is not True:
        raise PrepareError("fidelity plan schema/status differs")
    if plan.get("split_manifest_sha256") != expected_split_sha or plan.get("policy_sha256") != expected_policy_sha or plan.get("calibration_cases_sha256") != expected_calibration_sha:
        raise PrepareError("fidelity plan split binding differs")
    expected_identity = {"served_model_manifest_sha256": expected_served_sha, "package_manifest_sha256": expected_package_sha, "worker_binary_sha256": expected_worker_sha, "guard_sha256": expected_guard_sha, "device_architecture": expected_device_architecture, "quantized_artifact_revision": expected_quantized_revision}
    if plan.get("expected_active_identity") != expected_identity:
        raise PrepareError("fidelity plan active identity differs")
    if not isinstance(plan.get("source", {}).get("command_template"), str) or any(flag not in plan["source"]["command_template"] for flag in ("--split-root SPLIT_ROOT", "--expected-split-manifest-sha256 EXPECTED_SPLIT_MANIFEST_SHA256", "--expected-policy-sha256 EXPECTED_POLICY_SHA256", "--expected-calibration-cases-sha256 EXPECTED_CALIBRATION_CASES_SHA256", "--expected-cases-sha256 EXPECTED_SOURCE_CASES_SHA256")) or not isinstance(plan.get("active", {}).get("command_template"), str) or any(flag not in plan["active"]["command_template"] for flag in ("--expected-split-manifest-sha256", "--expected-policy-sha256", "--expected-calibration-cases-sha256", "--expected-served-model-manifest-sha256", "--expected-package-manifest-sha256", "--expected-worker-binary-sha256", "--expected-guard-sha256", "--expected-device-architecture", "--expected-quantized-artifact-revision")):
        raise PrepareError("fidelity plan command templates omit required bindings")


def sha(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise PrepareError(f"{label} must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, value: Any) -> str:
    if os.path.lexists(path):
        raise PrepareError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def fixture_tokens(root: Path, row: dict[str, Any]) -> list[int]:
    fixture_path = Path(row["fixture_path"])
    if not fixture_path.is_absolute():
        fixture_path = root / fixture_path
    if sha(fixture_path, f"fixture {row['case_id']}") != row["fixture_sha256"]:
        raise PrepareError(f"fixture hash differs: {row['case_id']}")
    value = json.loads(fixture_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != PROTOCOL.FIXTURE_SCHEMA or not isinstance(value.get("cases"), list) or len(value["cases"]) != 1:
        raise PrepareError(f"fixture schema differs: {row['case_id']}")
    case = value["cases"][0]
    tokens = case.get("prompt_token_ids")
    if case.get("case_id") != row["case_id"] or not isinstance(tokens, list) or len(tokens) != row["prompt_tokens"]:
        raise PrepareError(f"fixture token contract differs: {row['case_id']}")
    if hashlib.sha256(json.dumps(tokens, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest() != row["prompt_token_ids_sha256"]:
        raise PrepareError(f"fixture prompt hash differs: {row['case_id']}")
    if PROTOCOL.context_hash(tokens) != row["context_token_ids_sha256"]:
        raise PrepareError(f"fixture context hash differs: {row['case_id']}")
    return [int(token) for token in tokens]


def prepare(split_root: Path, output: Path, plan_output: Path, *, expected_split_sha: str, expected_policy_sha: str, expected_calibration_sha: str, expected_served_sha: str, expected_package_sha: str, expected_worker_sha: str, expected_guard_sha: str, expected_device_architecture: str, expected_quantized_revision: str) -> dict[str, Any]:
    try:
        VALIDATE.validate(split_root)
    except Exception as error:
        raise PrepareError(f"split validation failed: {error}") from error
    manifest_path = split_root / "split-manifest.json"
    policy_path = split_root / "policy.json"
    cases_path = split_root / "calibration-cases.jsonl"
    split_sha = sha(manifest_path, "split manifest")
    policy_sha = sha(policy_path, "policy")
    calibration_sha = sha(cases_path, "calibration cases")
    expected = {"split_manifest_sha256": expected_split_sha, "policy_sha256": expected_policy_sha, "calibration_cases_sha256": expected_calibration_sha}
    actual = {"split_manifest_sha256": split_sha, "policy_sha256": policy_sha, "calibration_cases_sha256": calibration_sha}
    if actual != expected:
        raise PrepareError("split/policy/calibration SHA does not match the pinned execution contract")
    for label, value in (("served manifest", expected_served_sha), ("package manifest", expected_package_sha), ("worker", expected_worker_sha), ("guard", expected_guard_sha)):
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise PrepareError(f"expected {label} SHA is invalid")
    if not expected_device_architecture or not expected_quantized_revision:
        raise PrepareError("expected active device/revision binding is empty")
    rows = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != MAX_ROWS:
        raise PrepareError(f"calibration split must contain exactly {MAX_ROWS} rows")
    cases = []
    seen: set[str] = set()
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in seen:
            raise PrepareError("calibration split contains duplicate case IDs")
        seen.add(case_id)
        tokens = fixture_tokens(split_root, row)
        cases.append({"case_id": case_id, "prompt_token_ids": tokens, "step_count": 1, "semantic_input_id": case_id, "observation": "fidelity_full_context_step0"})
    cases_sha = atomic(output, {"schema_version": CASES_SCHEMA, "cases": cases})
    plan = {
        "schema_version": PLAN_SCHEMA, "status": "ready_for_source_and_active_capture", "promotion_eligible": False,
        "row_count": MAX_ROWS, "full_context_step": 0, "one_source_model_load": True, "one_active_model_load": True,
        "split_manifest_sha256": split_sha, "policy_sha256": policy_sha, "calibration_cases_sha256": calibration_sha, "source_cases_sha256": cases_sha,
        "source": {"tool": "tools/export-qwen35-aq4-source-calibration.py", "cases": str(output.resolve()), "expected_schema": "ullm.qwen35_aq4_source_calibration.v1", "command_template": "python3 tools/export-qwen35-aq4-source-calibration.py --model-dir BF16_MODEL_DIR --split-root SPLIT_ROOT --cases CASES_JSON --output SOURCE_ARTIFACT --legacy-oracle LEGACY_SOURCE_ORACLE --expected-split-manifest-sha256 EXPECTED_SPLIT_MANIFEST_SHA256 --expected-policy-sha256 EXPECTED_POLICY_SHA256 --expected-calibration-cases-sha256 EXPECTED_CALIBRATION_CASES_SHA256 --expected-cases-sha256 EXPECTED_SOURCE_CASES_SHA256 --threads 1"},
        "active": {"tool": "target/release/ullm-aq4-fidelity-capture", "cases": str(output.resolve()), "expected_schema": "ullm.qwen35_aq4_target_calibration.v1", "command_template": "target/release/ullm-aq4-fidelity-capture --served-model-manifest ACTIVE_MANIFEST --split-root SPLIT_ROOT --source SOURCE_ARTIFACT --cases CASES_JSON --output ACTIVE_ARTIFACT --device-index DEVICE_INDEX --expected-split-manifest-sha256 EXPECTED_SPLIT_MANIFEST_SHA256 --expected-policy-sha256 EXPECTED_POLICY_SHA256 --expected-calibration-cases-sha256 EXPECTED_CALIBRATION_CASES_SHA256 --expected-served-model-manifest-sha256 EXPECTED_SERVED_MODEL_MANIFEST_SHA256 --expected-package-manifest-sha256 EXPECTED_PACKAGE_MANIFEST_SHA256 --expected-worker-binary-sha256 EXPECTED_WORKER_BINARY_SHA256 --expected-guard-sha256 EXPECTED_GUARD_SHA256 --expected-device-architecture EXPECTED_DEVICE_ARCHITECTURE --expected-quantized-artifact-revision EXPECTED_QUANTIZED_ARTIFACT_REVISION"},
        "execution_contract": {"all_m1": {"requested_m_is_label_only": True, "effective_m": 1}, "cold_batched": {"requested_m_is_execution_width": True, "effective_m": "requested_m"}, "semantic_output": "every row is the final hidden/logit result at the same full prompt context and step=0; M changes dispatch partition only", "context_reset_between_rows": True},
        "expected_active_identity": {"served_model_manifest_sha256": expected_served_sha, "package_manifest_sha256": expected_package_sha, "worker_binary_sha256": expected_worker_sha, "guard_sha256": expected_guard_sha, "device_architecture": expected_device_architecture, "quantized_artifact_revision": expected_quantized_revision},
        "bounds": {"max_rows": MAX_ROWS, "max_row_bytes": 64 * 1024, "max_vector_elements_per_row": 248320 + 4096, "max_hidden_sidecar_bytes": MAX_ROWS * 4096 * 4, "max_logits_sidecar_bytes": MAX_ROWS * 248320 * 4, "streaming": True, "host_resident_vector_rows": 1},
    }
    validate_plan(plan, expected_split_sha=expected_split_sha, expected_policy_sha=expected_policy_sha, expected_calibration_sha=expected_calibration_sha, expected_served_sha=expected_served_sha, expected_package_sha=expected_package_sha, expected_worker_sha=expected_worker_sha, expected_guard_sha=expected_guard_sha, expected_device_architecture=expected_device_architecture, expected_quantized_revision=expected_quantized_revision)
    plan_sha = atomic(plan_output, plan)
    return {"status": "ok", "row_count": MAX_ROWS, "cases": str(output), "cases_sha256": cases_sha, "plan": str(plan_output), "plan_sha256": plan_sha, "split_manifest_sha256": split_sha, "policy_sha256": policy_sha}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--expected-split-manifest-sha256", required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--expected-calibration-cases-sha256", required=True)
    parser.add_argument("--expected-served-model-manifest-sha256", required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument("--expected-worker-binary-sha256", required=True)
    parser.add_argument("--expected-guard-sha256", required=True)
    parser.add_argument("--expected-device-architecture", required=True)
    parser.add_argument("--expected-quantized-artifact-revision", required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(args.split_root, args.output, args.plan_output, expected_split_sha=args.expected_split_manifest_sha256, expected_policy_sha=args.expected_policy_sha256, expected_calibration_sha=args.expected_calibration_cases_sha256, expected_served_sha=args.expected_served_model_manifest_sha256, expected_package_sha=args.expected_package_manifest_sha256, expected_worker_sha=args.expected_worker_binary_sha256, expected_guard_sha=args.expected_guard_sha256, expected_device_architecture=args.expected_device_architecture, expected_quantized_revision=args.expected_quantized_artifact_revision), ensure_ascii=True, sort_keys=True))
        return 0
    except (PrepareError, OSError, ValueError) as error:
        print(f"Qwen3.5 AQ4 fidelity case preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
