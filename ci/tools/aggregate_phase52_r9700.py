#!/usr/bin/env python3
"""Publish the bounded Phase 52 R9700 long-context closeout summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any, Mapping, NoReturn, Sequence

import run_phase50_r9700_sllm as phase50


SCHEMA_VERSION = "phase52-r9700-kv-commit-summary-v1"
CASES = ("long-10001", "long-100000")
EXPECTED_MEMORY_KIND = {
    "long-10001": "virtual-contiguous",
    "long-100000": "contiguous-resident",
}
EXPECTED_CONTEXT = {"long-10001": 10_003, "long-100000": 131_072}
MAX_JSON_BYTES = 128 * 1024 * 1024


class Phase52Error(RuntimeError):
    pass


def _fail(message: str) -> NoReturn:
    raise Phase52Error(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"row is not a regular non-symlink file: {path}")
    data = path.read_bytes()
    if not data or len(data) > MAX_JSON_BYTES:
        _fail(f"row is empty or oversized: {path}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                _fail(f"{path}: duplicate JSON key {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: _fail(f"{path}: non-finite constant {token}"),
        )
    except Phase52Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"{path}: malformed JSON: {error}")
    if not isinstance(value, dict):
        _fail(f"{path}: row is not an object")
    return value


def finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label}: value is not numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        _fail(f"{label}: value is not finite and positive")
    return converted


def stats(values: Sequence[float], label: str) -> dict[str, float | int]:
    checked = [finite_positive(value, f"{label}[{index}]") for index, value in enumerate(values)]
    if len(checked) not in (3, 10):
        _fail(f"{label}: expected three or ten measured values")
    middle = float(median(checked))
    return {
        "median": middle,
        "mad": float(median(abs(value - middle) for value in checked)),
        "count": len(checked),
        "min": min(checked),
        "max": max(checked),
    }


def monitor_peak(report: Mapping[str, Any], row_path: Path) -> tuple[int, int]:
    raw = report.get("raw")
    item = raw.get("monitor_tsv") if isinstance(raw, dict) else None
    if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
        _fail(f"{row_path}: monitor identity is absent")
    path = Path(item["path"])
    if path.is_symlink() or not path.is_file() or sha256_file(path) != item["sha256"]:
        _fail(f"{row_path}: monitor artifact is absent or changed")
    hbm_peak = 0
    gtt_peak = 0
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines or lines[0] != "timestamp_ns\thbm_bytes\tgtt_bytes":
        _fail(f"{row_path}: monitor header is invalid")
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 3 or any(not field.isdigit() for field in fields):
            _fail(f"{row_path}: monitor sample is invalid")
        hbm_peak = max(hbm_peak, int(fields[1]))
        gtt_peak = max(gtt_peak, int(fields[2]))
    if hbm_peak <= 0 or gtt_peak <= 0:
        _fail(f"{row_path}: monitor has no positive samples")
    return hbm_peak, gtt_peak


def validate_external_cleanup(report: Mapping[str, Any], row_path: Path) -> None:
    process = report.get("process")
    capture = process.get("capture") if isinstance(process, dict) else None
    if not isinstance(capture, dict) or capture.get("process_group_gone") is not True:
        _fail(f"{row_path}: process group cleanup is absent")
    memory = report.get("memory")
    baseline = memory.get("baseline") if isinstance(memory, dict) else None
    settled = memory.get("settled") if isinstance(memory, dict) else None
    if not isinstance(baseline, dict) or not isinstance(settled, dict):
        _fail(f"{row_path}: external memory evidence is absent")
    if (
        settled.get("settled") is not True
        or settled.get("hbm_bytes") != baseline.get("hbm_bytes")
        or settled.get("gtt_bytes") != baseline.get("gtt_bytes")
    ):
        _fail(f"{row_path}: HBM/GTT did not return to the exact baseline")
    session_cleanup = report.get("result", {}).get("session_cleanup")
    if session_cleanup != {"retryable_cleanup": 0, "durable_quarantine": 0}:
        _fail(f"{row_path}: session cleanup is not empty")


def validate_kv_memory(sample: Mapping[str, Any], case_id: str, label: str) -> dict[str, Any]:
    memory = sample.get("memory")
    kv = memory.get("kv") if isinstance(memory, dict) else None
    layers = kv.get("layers") if isinstance(kv, dict) else None
    if not isinstance(layers, list) or not layers or kv.get("kv_layer_count") != len(layers):
        _fail(f"{label}: KV layer memory audit is absent")
    expected_kind = EXPECTED_MEMORY_KIND[case_id]
    normalized: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or layer.get("memory_kind") != expected_kind:
            _fail(f"{label}: KV layer {index} memory kind is not {expected_kind}")
        integer_fields = (
            "layer",
            "logical_capacity_tokens",
            "observed_length_tokens",
            "physical_page_bytes",
            "tokens_per_page",
            "mapped_token_capacity",
            "committed_bytes_per_plane",
        )
        if any(
            isinstance(layer.get(field), bool)
            or not isinstance(layer.get(field), int)
            or layer[field] < (0 if field == "layer" else 1)
            for field in integer_fields
        ):
            _fail(f"{label}: KV layer {index} has invalid physical metadata")
        # The terminal generated token is published but is not fed back into
        # KV; two generated tokens therefore append one decode-input token.
        expected_observed = 10_002 if case_id == "long-10001" else 100_001
        if layer["logical_capacity_tokens"] != EXPECTED_CONTEXT[case_id]:
            _fail(f"{label}: KV layer {index} logical capacity differs")
        if layer["observed_length_tokens"] != expected_observed:
            _fail(f"{label}: KV layer {index} observed length differs")
        if layer["observed_length_tokens"] > layer["mapped_token_capacity"]:
            _fail(f"{label}: KV layer {index} is published beyond mapped capacity")
        normalized.append({field: layer[field] for field in integer_fields} | {"memory_kind": expected_kind})
    committed = sum(layer["committed_bytes_per_plane"] * 2 for layer in normalized)
    if kv.get("committed_kv_bytes") != committed:
        _fail(f"{label}: aggregate KV committed bytes differ from layers")
    return {
        "memory_kind": expected_kind,
        "kv_layer_count": len(normalized),
        "logical_capacity_tokens": normalized[0]["logical_capacity_tokens"],
        "observed_length_tokens": normalized[0]["observed_length_tokens"],
        "physical_page_bytes": normalized[0]["physical_page_bytes"],
        "tokens_per_page": normalized[0]["tokens_per_page"],
        "mapped_token_capacity": min(layer["mapped_token_capacity"] for layer in normalized),
        "committed_bytes_per_plane": normalized[0]["committed_bytes_per_plane"],
        "committed_kv_bytes": committed,
    }


def summarize_row(path: Path, case_id: str) -> dict[str, Any]:
    report = load_json(path)
    if report.get("schema_version") != phase50.ROW_SCHEMA_VERSION or report.get("state") != "PASS":
        _fail(f"{path}: producer row is not PASS {phase50.ROW_SCHEMA_VERSION}")
    if (
        report.get("target") != phase50.TARGET
        or report.get("gpu_uuid") != phase50.GPU_UUID
        or report.get("gpu_bdf") != phase50.GPU_BDF
    ):
        _fail(f"{path}: R9700 target identity differs")
    row = report.get("row")
    if not isinstance(row, dict) or row.get("case_id") != case_id:
        _fail(f"{path}: case identity differs")
    expected_input = 10_001 if case_id == "long-10001" else 100_000
    if (
        row.get("input_token_count") != expected_input
        or row.get("requested_output_tokens") != 2
        or row.get("warmups") != (3 if case_id == "long-10001" else 1)
        or row.get("measured") != (10 if case_id == "long-10001" else 3)
        or row.get("context_length") != EXPECTED_CONTEXT[case_id]
        or row.get("prefill_chunk_tokens") is not None
    ):
        _fail(f"{path}: fixed Phase 52 row protocol differs")
    result = phase50.validate_result({"state": "PASS", "result": report.get("result")}, row, "bf16")
    config = result["config"]
    if config.get("prefill_chunk_selection") != "automatic" or config.get("prefill_chunk_candidates") != [2048, 512]:
        _fail(f"{path}: automatic 32 GiB candidate sequence differs")
    selected_chunk = config.get("effective_prefill_chunk_tokens")
    if selected_chunk not in (2048, 512):
        _fail(f"{path}: effective chunk is not a selected automatic candidate")
    placement_source = result.get("memory")
    placement_fields = {
        "total_memory_bytes": "placement_total_memory_bytes",
        "available_memory_bytes": "placement_available_memory_bytes",
        "required_bytes": "placement_required_bytes",
        "model_resident_bytes": "placement_model_resident_bytes",
        "request_state_bytes": "placement_request_state_bytes",
        "workspace_arena_bytes": "workspace_arena_bytes",
        "safety_reserve_bytes": "placement_safety_reserve_bytes",
    }
    if not isinstance(placement_source, dict):
        _fail(f"{path}: placement memory evidence is absent")
    placement: dict[str, int] = {}
    for output_name, source_name in placement_fields.items():
        value = placement_source.get(source_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _fail(f"{path}: placement field {source_name} is invalid")
        placement[output_name] = value
    if (
        placement["total_memory_bytes"] <= 0
        or placement["available_memory_bytes"] <= 0
        or placement["required_bytes"] <= 0
        or placement["required_bytes"] > placement["available_memory_bytes"]
    ):
        _fail(f"{path}: placement preflight does not fit the selected candidate")
    samples = list(result["warmups"]["samples"]) + list(result["measured"]["samples"])
    kv_audits = [validate_kv_memory(sample, case_id, f"{path} sample {index}") for index, sample in enumerate(samples)]
    if any(audit != kv_audits[0] for audit in kv_audits[1:]):
        _fail(f"{path}: KV physical metadata differs across repetitions")
    measured = result["measured"]["samples"]
    e2e = [sample["derived"]["e2e_ns"] for sample in measured]
    ttft = [sample["derived"]["ttft_ns"] for sample in measured]
    tpot = [value for sample in measured for value in sample["derived"]["tpot_ns"]]
    hbm_peak, gtt_peak = monitor_peak(report, path)
    validate_external_cleanup(report, path)
    generated = result["correctness_control"]["tokens"]["generated_token_ids"]
    if generated != [23066, 23066]:
        _fail(f"{path}: fixed long output is not [23066,23066]")
    raw = report.get("raw")
    input_item = raw.get("input_token_ids") if isinstance(raw, dict) else None
    if input_item is not None:
        if not isinstance(input_item, dict) or not isinstance(input_item.get("path"), str) or not isinstance(input_item.get("sha256"), str):
            _fail(f"{path}: input token artifact identity is malformed")
        input_path = Path(input_item["path"])
        if input_path.is_symlink() or not input_path.is_file() or sha256_file(input_path) != input_item["sha256"]:
            _fail(f"{path}: input token artifact is absent or changed")
    input_token_ids = row.get("input_token_ids")
    if (
        not isinstance(input_token_ids, list)
        or len(input_token_ids) != expected_input
        or any(token != 23066 for token in input_token_ids)
    ):
        _fail(f"{path}: fixed semantic input token sequence differs")
    return {
        "case_id": case_id,
        "row_id": row["row_id"],
        "input_token_count": expected_input,
        "requested_output_tokens": 2,
        "warmups": row["warmups"],
        "measured": row["measured"],
        "context_length": row["context_length"],
        "effective_prefill_chunk_tokens": selected_chunk,
        "prefill_chunk_candidates": [2048, 512],
        "placement": placement,
        "kv": kv_audits[0],
        "input_token_ids_sha256": hashlib.sha256(canonical_bytes(input_token_ids)).hexdigest(),
        "generated_token_ids_sha256": hashlib.sha256(canonical_bytes(generated)).hexdigest(),
        "metrics": {"e2e_ns": stats(e2e, f"{case_id} e2e"), "ttft_ns": stats(ttft, f"{case_id} ttft"), "tpot_ns": stats(tpot, f"{case_id} tpot")},
        "repetitions": {"e2e_ns": e2e, "ttft_ns": ttft, "tpot_ns": tpot},
        "resources": {"sysfs_hbm_peak_bytes": hbm_peak, "sysfs_gtt_peak_bytes": gtt_peak, "process_group_gone": True, "baseline_restored": True, "cleanup_failures": 0},
        "evidence": {"path": str(path.resolve()), "sha256": sha256_file(path)},
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.source_commit or len(args.source_commit) != 40 or any(char not in "0123456789abcdef" for char in args.source_commit):
        _fail("source commit must be a lowercase 40-hex Git identity")
    paths = {"long-10001": Path(args.long_10001_row), "long-100000": Path(args.long_100000_row)}
    rows = [summarize_row(paths[case], case) for case in CASES]
    raw_reports = [load_json(paths[case]) for case in CASES]
    binary = raw_reports[0].get("binary")
    model = raw_reports[0].get("model")
    lock = raw_reports[0].get("lock")
    if any(report.get("binary") != binary or report.get("model") != model or report.get("lock") != lock for report in raw_reports[1:]):
        _fail("Phase 52 rows do not share binary/model/lock identity")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "state": "PASS",
        "source_commit": args.source_commit,
        "target": phase50.TARGET,
        "gpu_uuid": phase50.GPU_UUID,
        "gpu_bdf": phase50.GPU_BDF,
        "binary_sha256": binary["sha256"],
        "model_sha256": model["sha256"],
        "model_lock_sha256": lock["sha256"],
        "producer_runner_sha256": sha256_file(Path(phase50.__file__)),
        "provider_policy": {"target": "gfx1201", "minimum_capacity_tokens": 65536, "memory_kind": "contiguous-resident"},
        "rows": rows,
        "acceptance": {"automatic_prefill": True, "all_requests_passed": True, "output_exact": True, "hip_only": True, "fallback_count": 0, "cleanup_failures": 0, "baseline_restored": True},
    }
    output = Path(args.output)
    data = canonical_bytes(summary)
    if output.exists():
        if output.is_symlink() or output.read_bytes() != data:
            _fail(f"refusing to overwrite differing output: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--long-10001-row", required=True)
    parser.add_argument("--long-100000-row", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        summary = aggregate(build_parser().parse_args(argv))
    except (OSError, Phase52Error, phase50.SessionDError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
