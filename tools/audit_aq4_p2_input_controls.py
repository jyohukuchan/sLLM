#!/usr/bin/env python3
"""Audit P2 inputs, sampling, sequence layout, and a CPU AQ4 matvec control.

This is an evidence-only checker.  It compares bounded source/path oracle rows,
the deterministic pure-prefill fixture, and the gateway request fixture without
starting a model, worker, gateway, or GPU runtime.  The oracle payloads do not
contain position IDs or masks, so those fields are reported as *unobserved*
unless a fixture supplies them; the checker never infers their correctness from
context length alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ORACLE = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/source-oracle-v2"
DEFAULT_PATH_ORACLE = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/path-oracle-v2"
DEFAULT_CASE_MANIFEST = ROOT / "benchmarks/workloads/qwen35-aq4-p2-source-calibration-cases-v0.1.json"
DEFAULT_SERVED = Path("/etc/ullm/served-models/active.json")
DEFAULT_PURE_FIXTURE = ROOT / "tests/fixtures/aq4-p2-input-controls/pure-prefill.json"
DEFAULT_GATEWAY_FIXTURE = ROOT / "tests/fixtures/aq4-p2-input-controls/gateway-request.json"
VOCAB_SIZE = 248_320


class AuditError(ValueError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> bytes:
    # Match qwen35_aq4_p2_oracle.canonical_token_ids_hash: compact JSON plus
    # a terminal newline. The newline is part of the committed hash contract.
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def token_hash(token_ids: Iterable[int]) -> str:
    values = [int(item) for item in token_ids]
    if not values or any(item < 0 for item in values):
        raise AuditError("token IDs must be non-empty non-negative integers")
    return sha256_bytes(canonical(values))


def payload_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "payload.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise AuditError(f"empty oracle payload line: {path}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise AuditError("oracle payload row must be an object")
            rows.append(row)
    return rows


def _topk_contract(values: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [int(item["token_id"]) for item in values]
    logits = [float(item["logit"]) for item in values]
    finite = all(math.isfinite(value) for value in logits)
    ordered = all(
        logits[index] > logits[index + 1]
        or (logits[index] == logits[index + 1] and ids[index] < ids[index + 1])
        for index in range(len(ids) - 1)
    )
    return {"finite": finite, "tie_ordered": ordered, "token_ids": ids, "top1": ids[0] if ids else None}


def audit_oracles(source_root: Path, path_root: Path) -> dict[str, Any]:
    source_manifest = load_json(source_root / "manifest.json")
    path_manifest = load_json(path_root / "manifest.json")
    source_cases = {str(item["case_id"]): item for item in source_manifest["cases"]}
    path_cases = {str(item["case_id"]): item for item in path_manifest["cases"]}
    if source_cases != path_cases:
        raise AuditError("source/path oracle case contracts differ")
    source = {(str(row["case_id"]), int(row["step"])): row for row in payload_rows(source_root)}
    path = {(str(row["case_id"]), int(row["step"])): row for row in payload_rows(path_root)}
    expected = {(case_id, step) for case_id, item in source_cases.items() for step in range(int(item["step_count"]))}
    if set(source) != expected or set(path) != expected:
        raise AuditError("oracle case/step coverage differs from manifest")
    rows: list[dict[str, Any]] = []
    greedy_mismatches = 0
    topk_mismatches = 0
    context_mismatches = 0
    for key in sorted(expected):
        case_id, step = key
        case = source_cases[case_id]
        left, right = source[key], path[key]
        expected_context = int(case["prompt_token_count"]) + step
        # The compact source/path oracle schema intentionally omits context metadata.  The
        # expected position is still recorded from the bound case, but it is not counted as a
        # mismatch when neither side observed it.  Calibration/pure-prefill fixtures below carry
        # the actual token IDs and position/mask evidence.
        source_context_observed = "context_length" in left
        path_context_observed = "context_length" in right
        context_ok = (not source_context_observed or left.get("context_length") == expected_context) and (not path_context_observed or right.get("context_length") == expected_context)
        source_hash = left.get("context_token_ids_sha256", left.get("input_token_ids_sha256"))
        path_hash = right.get("context_token_ids_sha256", right.get("input_token_ids_sha256"))
        hashes_observed = source_hash is not None or path_hash is not None
        hashes_equal = (not hashes_observed) or source_hash == path_hash
        if not context_ok or not hashes_equal:
            context_mismatches += 1
        source_top = _topk_contract(left.get("topk", []))
        path_top = _topk_contract(right.get("topk", []))
        greedy_equal = left.get("greedy_token_id") == right.get("greedy_token_id") == source_top["top1"] == path_top["top1"]
        topk_equal = source_top["token_ids"] == path_top["token_ids"]
        if not greedy_equal:
            greedy_mismatches += 1
        if not topk_equal:
            topk_mismatches += 1
        rows.append({
            "case_id": case_id,
            "step": step,
            "prompt_tokens": int(case["prompt_token_count"]),
            "expected_context_length": expected_context,
            "source_context_length": left.get("context_length", expected_context),
            "path_context_length": right.get("context_length", expected_context),
            "context_length_observed": source_context_observed or path_context_observed,
            "source_context_token_ids_sha256": source_hash,
            "path_context_token_ids_sha256": path_hash,
            "context_token_ids_hash_observed": hashes_observed,
            "context_hash_equal": hashes_equal,
            "context_length_equal": context_ok,
            "generated_step_extension": "one_token_feedback" if step > 0 else "prefill_prompt",
            "source_greedy_token_id": left.get("greedy_token_id"),
            "path_greedy_token_id": right.get("greedy_token_id"),
            "greedy_equal": greedy_equal,
            "topk_equal": topk_equal,
            "source_topk_contract": source_top,
            "path_topk_contract": path_top,
            "position_ids_observed": False,
            "attention_mask_observed": False,
        })
    return {
        "source_schema": source_manifest.get("schema_version"),
        "path_schema": path_manifest.get("schema_version"),
        "ranking_contract_equal": source_manifest.get("ranking") == path_manifest.get("ranking"),
        "case_input_hashes": {
            case_id: str(item["prompt_token_ids_sha256"]) for case_id, item in sorted(source_cases.items())
        },
        "rows": rows,
        "row_count": len(rows),
        "context_mismatch_rows": context_mismatches,
        "greedy_mismatch_rows": greedy_mismatches,
        "topk_mismatch_rows": topk_mismatches,
        "position_ids_observed": False,
        "attention_mask_observed": False,
        "causal_semantics_status": "unobserved_in_oracle_payload",
    }


def audit_calibration_and_fixtures(case_manifest: Path, pure_path: Path, gateway_path: Path) -> dict[str, Any]:
    raw_cases = load_json(case_manifest).get("cases", [])
    calibration: dict[str, dict[str, Any]] = {}
    for item in raw_cases:
        ids = [int(value) for value in item["prompt_token_ids"]]
        calibration[str(item["case_id"])] = {
            "token_ids": ids,
            "token_count": len(ids),
            "token_ids_sha256": token_hash(ids),
            "step_count": int(item["step_count"]),
        }
    pure = load_json(pure_path)
    gateway = load_json(gateway_path)
    pure_ids = [int(value) for value in pure["prompt_token_ids"]]
    gateway_ids = [int(value) for value in gateway["prompt_token_ids"]]
    matching = [value for value in calibration.values() if value["token_ids"] == pure_ids == gateway_ids]
    if len(matching) != 1:
        raise AuditError("pure-prefill/gateway fixture IDs do not match exactly one calibration case")
    calibration_item = matching[0]
    positions = [int(value) for value in pure.get("position_ids", [])]
    expected_positions = list(range(len(pure_ids)))
    mask = pure.get("attention_mask")
    mask_expected = [[1 if column <= row else 0 for column in range(len(pure_ids))] for row in range(len(pure_ids))]
    pure_mask_ok = mask == mask_expected
    generated = [int(value) for value in gateway.get("expected_generated_token_ids", [])]
    decode_positions = [int(value) for value in gateway.get("expected_decode_position_ids", [])]
    expected_decode_positions = list(range(len(gateway_ids), len(gateway_ids) + len(generated)))
    declared_gateway_context = gateway.get("context_length")
    expected_gateway_context = len(gateway_ids) + len(generated)
    gateway_context_exact = declared_gateway_context == expected_gateway_context
    generated_steps_exact = len(generated) == int(calibration_item["step_count"])
    vocab_ok = all(0 <= value < VOCAB_SIZE for value in pure_ids + generated)
    return {
        "calibration_case": calibration_item,
        "pure_prefill": {
            "schema_version": pure.get("schema_version"),
            "token_ids": pure_ids,
            "token_ids_sha256": token_hash(pure_ids),
            "token_ids_match_calibration": pure_ids == calibration_item["token_ids"],
            "context_length": len(pure_ids),
            "position_ids": positions,
            "position_ids_expected": expected_positions,
            "position_ids_exact": positions == expected_positions,
            "attention_mask": mask,
            "causal_mask_exact": pure_mask_ok,
            "sampling": pure.get("sampling"),
        },
        "gateway_fixture": {
            "schema_version": gateway.get("schema_version"),
            "token_ids": gateway_ids,
            "token_ids_sha256": token_hash(gateway_ids),
            "token_ids_match_calibration": gateway_ids == calibration_item["token_ids"],
            "generated_token_ids": generated,
            "generated_step_count_expected": int(calibration_item["step_count"]),
            "generated_step_count_exact": generated_steps_exact,
            "decode_position_ids": decode_positions,
            "decode_position_ids_expected": expected_decode_positions,
            "decode_positions_exact": decode_positions == expected_decode_positions,
            "declared_context_length": declared_gateway_context,
            "expected_context_length": expected_gateway_context,
            "context_length_exact": gateway_context_exact,
            "context_extension": expected_gateway_context,
            "sampling": gateway.get("sampling"),
        },
        "token_ids_equal_across_controls": pure_ids == gateway_ids == calibration_item["token_ids"],
        "vocab_size": VOCAB_SIZE,
        "vocab_slicing_exact": vocab_ok,
    }


def audit_sampling(served_path: Path, case_manifest: Path, gateway_path: Path) -> dict[str, Any]:
    served = load_json(served_path)
    case_doc = load_json(case_manifest)
    gateway = load_json(gateway_path)
    case_sampling = case_doc.get("sampling", {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 0})
    served_sampling = served.get("generation", {}).get("sampling", {})
    gateway_sampling = gateway.get("sampling", {})
    effective_served = {
        "temperature": 0.0 if served_sampling.get("temperature") is False else served_sampling.get("temperature"),
        "top_p": 1.0 if served_sampling.get("top_p") is False else served_sampling.get("top_p"),
        "top_k": served_sampling.get("top_k"),
    }
    expected = {"temperature": 0.0, "top_p": 1.0, "top_k": 1}
    return {
        "case_sampling": case_sampling,
        "served_declared_sampling": served_sampling,
        "served_effective_sampling": effective_served,
        "gateway_sampling": gateway_sampling,
        "greedy_contract": expected,
        "case_matches_greedy": all(case_sampling.get(field) == value for field, value in expected.items()),
        "served_matches_greedy": effective_served == expected,
        "gateway_matches_greedy": {field: gateway_sampling.get(field) for field in expected} == expected,
        "temperature_top_p_disabled_in_served": served_sampling.get("temperature") is False and served_sampling.get("top_p") is False,
    }


def aq4_matvec_reference(indices: bytes, scale_indices: bytes, codebook: list[float], scales: list[float], tensor_scale: float, input_values: list[float], rows: int, cols: int, group_size: int, row_scales: list[float] | None = None) -> list[float]:
    if len(indices) < (rows * cols + 1) // 2 or len(scale_indices) < (rows * cols + group_size - 1) // group_size:
        raise AuditError("AQ4 fixture payload is short")
    result: list[float] = []
    for row in range(rows):
        total = 0.0
        for column in range(cols):
            element = row * cols + column
            packed = indices[element // 2]
            code = packed & 0x0F if element % 2 == 0 else packed >> 4
            group = element // group_size
            total += codebook[code] * scales[scale_indices[group]] * tensor_scale * input_values[column]
        if row_scales is not None and row < len(row_scales):
            total *= row_scales[row]
        result.append(total)
    return result


def audit_matvec() -> dict[str, Any]:
    # This is the deterministic fixture used by cpu_aq4_matvec_f32_computes_expected_values.
    indices = bytes((0x21, 0x03, 0x54))
    scale_indices = bytes((0, 1, 0))
    codebook = [float(value) for value in range(16)]
    scales = [0.5, 2.0]
    input_values = [0.5, -1.0, 2.0]
    expected = [112.5, 30.0]
    actual = aq4_matvec_reference(indices, scale_indices, codebook, scales, 10.0, input_values, 2, 3, 2)
    batch = aq4_matvec_reference(indices, scale_indices, codebook, scales, 10.0, [0.5, -1.0, 2.0], 2, 3, 2)
    return {
        "rows": 2,
        "cols": 3,
        "group_size": 2,
        "index_encoding": "idx4_low_nibble_first",
        "layout": "row_major",
        "transpose": False,
        "bias": None,
        "accumulation": "f32_scalar_sum",
        "actual": actual,
        "expected_cpu_runtime_fixture": expected,
        "exact": actual == expected,
        "batch_reference_exact": batch == expected,
    }


def run_audit(source_oracle: Path = DEFAULT_SOURCE_ORACLE, path_oracle: Path = DEFAULT_PATH_ORACLE, case_manifest: Path = DEFAULT_CASE_MANIFEST, served_manifest: Path = DEFAULT_SERVED, pure_fixture: Path = DEFAULT_PURE_FIXTURE, gateway_fixture: Path = DEFAULT_GATEWAY_FIXTURE) -> dict[str, Any]:
    oracle = audit_oracles(source_oracle, path_oracle)
    fixtures = audit_calibration_and_fixtures(case_manifest, pure_fixture, gateway_fixture)
    sampling = audit_sampling(served_manifest, case_manifest, gateway_fixture)
    matvec = audit_matvec()
    calibration_cases = load_json(case_manifest).get("cases", [])
    calibration_hashes = {
        str(item["case_id"]): token_hash(item["prompt_token_ids"])
        for item in calibration_cases
    }
    input_binding_rows = []
    for case_id, source_hash in sorted(oracle["case_input_hashes"].items()):
        calibration_hash = calibration_hashes.get(case_id)
        input_binding_rows.append(
            {
                "case_id": case_id,
                "source_path_prompt_hash": source_hash,
                "calibration_prompt_hash": calibration_hash,
                "hash_observed": calibration_hash is not None,
                "hash_equal": calibration_hash == source_hash if calibration_hash is not None else False,
            }
        )
    input_binding = {
        "source_path_vs_calibration": input_binding_rows,
        "mismatch_rows": sum(1 for row in input_binding_rows if row["hash_observed"] and not row["hash_equal"]),
        "calibration_case_count": len(calibration_hashes),
    }
    if input_binding["mismatch_rows"]:
        status = "input_binding_mismatch"
    elif (
        oracle["context_mismatch_rows"] == 0
        and fixtures["token_ids_equal_across_controls"]
        and fixtures["pure_prefill"]["position_ids_exact"]
        and fixtures["pure_prefill"]["causal_mask_exact"]
        and fixtures["gateway_fixture"]["generated_step_count_exact"]
        and fixtures["gateway_fixture"]["decode_positions_exact"]
        and fixtures["gateway_fixture"]["context_length_exact"]
        and fixtures["vocab_slicing_exact"]
        and sampling["case_matches_greedy"]
        and sampling["served_matches_greedy"]
        and sampling["gateway_matches_greedy"]
        and matvec["exact"]
    ):
        status = "controls_match_except_source_path_greedy_and_topk"
    else:
        status = "blocked_or_mismatch"
    return {
        "schema_version": "ullm.aq4_p2_input_control_audit.v1",
        "read_only": True,
        "gpu_or_service_touched": False,
        "oracle": oracle,
        "input_binding": input_binding,
        "fixtures": fixtures,
        "sampling": sampling,
        "lm_head_contract": {
            "orientation": "rows_are_vocab_tokens, columns_are_hidden_features",
            "vocab_size": VOCAB_SIZE,
            "source_path_oracle_logits_shape": [VOCAB_SIZE],
            "ranking_scope": "entire_vocabulary",
            "greedy_tie_policy": "maximum_logit_then_smallest_token_id",
        },
        "matvec": matvec,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-oracle", type=Path, default=DEFAULT_SOURCE_ORACLE)
    parser.add_argument("--path-oracle", type=Path, default=DEFAULT_PATH_ORACLE)
    parser.add_argument("--case-manifest", type=Path, default=DEFAULT_CASE_MANIFEST)
    parser.add_argument("--served-manifest", type=Path, default=DEFAULT_SERVED)
    parser.add_argument("--pure-fixture", type=Path, default=DEFAULT_PURE_FIXTURE)
    parser.add_argument("--gateway-fixture", type=Path, default=DEFAULT_GATEWAY_FIXTURE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = run_audit(args.source_oracle, args.path_oracle, args.case_manifest, args.served_manifest, args.pure_fixture, args.gateway_fixture)
        encoded = json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
        return 0 if report["status"].startswith("controls_match") else 2
    except (AuditError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"AQ4 P2 input-control audit failed: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
