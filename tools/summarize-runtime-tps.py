#!/usr/bin/env python3
"""Summarize pre-SQ runtime token/s smoke JSON files as Markdown or JSONL."""

from __future__ import annotations

import argparse
import glob
import json
import re
import socket
import sys
from pathlib import Path
from typing import Any


DEFAULT_GLOB = "benchmarks/results/2026-07-06/engine/package-token-ids-generate-smoke-*.json"
SCHEMA_VERSION = "inference-benchmark-result-v0.1"
F32_BYTES = 4


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            value = json.load(input_file)
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    return value


def expand_inputs(patterns: list[str]) -> list[Path]:
    if not patterns:
        patterns = [DEFAULT_GLOB]

    paths: list[Path] = []
    for pattern in patterns:
        if any(char in pattern for char in "*?["):
            matches = [Path(match) for match in sorted(glob.glob(pattern))]
            if not matches:
                raise SystemExit(f"no files matched: {pattern}")
            paths.extend(matches)
        else:
            path = Path(pattern)
            if not path.exists():
                raise SystemExit(f"input does not exist: {path}")
            paths.append(path)

    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in paths:
        normalized = path.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_paths.append(path)
    return unique_paths


def self_attention_layer_count(raw: dict[str, Any]) -> int | None:
    layer_kinds = raw.get("layer_kinds")
    if isinstance(layer_kinds, list):
        count = sum(1 for item in layer_kinds if item == "self_attention")
        if count > 0:
            return count

    layers = raw.get("layers")
    if isinstance(layers, list) and layers:
        return len(layers)
    return None


def estimate_f32_kv_cache_bytes(raw: dict[str, Any]) -> int | None:
    memory = as_mapping(raw.get("memory"))
    self_attn = as_mapping(raw.get("self_attn"))
    blocks = parse_int(memory.get("cache_blocks"))
    block_size = parse_int(memory.get("block_size"))
    kv_heads = parse_int(self_attn.get("kv_heads"))
    key_dim = parse_int(self_attn.get("head_dim"))
    value_dim = parse_int(self_attn.get("value_dim"))
    layer_count = self_attention_layer_count(raw)

    if value_dim is None:
        value_dim = key_dim
    values = (blocks, block_size, kv_heads, key_dim, value_dim, layer_count)
    if any(value is None for value in values):
        return None

    return blocks * block_size * layer_count * kv_heads * (key_dim + value_dim) * F32_BYTES


def kv_cache_bytes(raw: dict[str, Any]) -> tuple[int | None, bool]:
    memory = as_mapping(raw.get("memory"))
    value = parse_int(memory.get("kv_cache_bytes"))
    if value is not None:
        return value, False
    estimated = estimate_f32_kv_cache_bytes(raw)
    return estimated, estimated is not None


def prompt_tokens(raw: dict[str, Any]) -> int | None:
    prefill = as_mapping(raw.get("prefill"))
    value = parse_int(prefill.get("prompt_tokens"))
    if value is not None:
        return value
    tokens = raw.get("prompt_token_ids")
    if isinstance(tokens, list):
        return len(tokens)
    return None


def generated_tokens(raw: dict[str, Any]) -> int | None:
    decode = as_mapping(raw.get("decode"))
    value = parse_int(decode.get("requested_generated_tokens"))
    if value is not None:
        return value
    tokens = raw.get("generated_token_ids")
    if isinstance(tokens, list):
        return len(tokens)
    return None


def prefill_tps(raw: dict[str, Any]) -> float | None:
    prefill = as_mapping(raw.get("prefill"))
    return parse_float(prefill.get("tps"))


def decode_tps(raw: dict[str, Any]) -> float | None:
    decode = as_mapping(raw.get("decode"))
    value = parse_float(decode.get("timed_step_tps"))
    if value is not None:
        return value
    return parse_float(decode.get("end_to_end_generated_tps"))


def wall_ms(raw: dict[str, Any], section: str) -> float | None:
    value = parse_float(as_mapping(raw.get(section)).get("wall_ms"))
    if value is not None:
        return value
    return parse_float(as_mapping(raw.get("timing_ms")).get(section))


def total_wall_seconds(raw: dict[str, Any]) -> float | None:
    throughput = as_mapping(raw.get("throughput"))
    timing_ms = as_mapping(raw.get("timing_ms"))
    value = parse_float(throughput.get("total_wall_ms"))
    if value is None:
        value = parse_float(timing_ms.get("total"))
    if value is None:
        return None
    return value / 1000.0


def total_tokens_per_second(raw: dict[str, Any]) -> float | None:
    throughput = as_mapping(raw.get("throughput"))
    value = parse_float(throughput.get("model_input_tps"))
    if value is not None:
        return value

    prompt = prompt_tokens(raw)
    generated = generated_tokens(raw)
    total_seconds = total_wall_seconds(raw)
    if prompt is None or generated is None or not total_seconds:
        return None
    return (prompt + generated) / total_seconds


def verified(raw: dict[str, Any]) -> bool | None:
    value = raw.get("verified")
    if isinstance(value, bool):
        return value
    correctness = as_mapping(raw.get("correctness"))
    value = correctness.get("verified")
    if isinstance(value, bool):
        return value
    return None


def device_label(raw: dict[str, Any]) -> str:
    name = raw.get("device_name")
    index = raw.get("device_index")
    if name is None and index is None:
        return ""
    if name is None:
        return f"idx {index}"
    if index is None:
        return str(name)
    return f"{name} (idx {index})"


def extract_date(paths: list[Path]) -> str | None:
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for path in paths:
        for part in path.parts:
            if pattern.match(part):
                return part
    return None


def case_id(path: Path) -> str:
    return f"ullm-{path.stem}"


def build_summary_row(path: Path, raw: dict[str, Any]) -> dict[str, Any]:
    kv_bytes, estimated = kv_cache_bytes(raw)
    return {
        "path": path,
        "prompt": prompt_tokens(raw),
        "generated": generated_tokens(raw),
        "device": device_label(raw),
        "decode_mode": raw.get("decode_mode"),
        "prefill_tps": prefill_tps(raw),
        "decode_tps": decode_tps(raw),
        "total_wall_seconds": total_wall_seconds(raw),
        "kv_cache_bytes": kv_bytes,
        "kv_cache_estimated": estimated,
        "verified": verified(raw),
        "raw": raw,
    }


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("prompt") if row.get("prompt") is not None else -1,
        row.get("generated") if row.get("generated") is not None else -1,
        row.get("device") or "",
        row.get("decode_mode") or "",
        str(row.get("path")),
    )


def markdown_escape(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def format_int(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return ""


def format_bool(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Pre-SQ Runtime TPS Summary",
        "",
        "| prompt | generated | device | decode_mode | prefill TPS | decode TPS | total wall seconds | KV cache bytes | verified |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    format_int(row["prompt"]),
                    format_int(row["generated"]),
                    markdown_escape(row["device"]),
                    markdown_escape(row["decode_mode"]),
                    format_float(row["prefill_tps"]),
                    format_float(row["decode_tps"]),
                    format_float(row["total_wall_seconds"]),
                    format_int(row["kv_cache_bytes"]),
                    format_bool(row["verified"]),
                ]
            )
            + " |"
        )

    if any(row["kv_cache_estimated"] for row in rows):
        lines.extend(
            [
                "",
                "KV cache bytes are read from `memory.kv_cache_bytes` when present. "
                "Null values are estimated as f32 bytes with "
                "`cache_blocks * block_size * self_attention_layers * kv_heads * "
                "(head_dim + value_dim) * 4`.",
            ]
        )
    return "\n".join(lines)


def normalized_row(path: Path, raw: dict[str, Any], run_id: str) -> dict[str, Any]:
    memory = as_mapping(raw.get("memory"))
    correctness = as_mapping(raw.get("correctness"))
    prompt = prompt_tokens(raw)
    generated = generated_tokens(raw)
    kv_bytes, kv_estimated = kv_cache_bytes(raw)
    prefill_wall_ms = wall_ms(raw, "prefill")
    decode_wall_ms = wall_ms(raw, "decode")
    total_seconds = total_wall_seconds(raw)
    decode_tokens_per_second = decode_tps(raw)
    consumed_bytes = parse_int(memory.get("vram_consumed_bytes"))
    consumed_gib = consumed_bytes / 1024**3 if consumed_bytes is not None else None
    decode_times_vram = (
        decode_tokens_per_second * consumed_gib
        if decode_tokens_per_second is not None and consumed_gib is not None
        else None
    )
    nan_or_inf = correctness.get("nan_or_inf_detected")
    is_verified = verified(raw)

    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "case_id": case_id(path),
        "status": "ok" if is_verified is not False else "failed",
        "engine": {
            "name": "uLLM",
            "version": None,
            "commit": raw.get("git_commit"),
        },
        "model": {
            "name": "Qwen3.5-9B",
            "source": "local-safetensors",
            "revision": None,
            "format": "ullm.d",
            "quantization": "qwen35_9b_p4p46_hidden3994_v1",
            "package": raw.get("package"),
        },
        "hardware": {
            "host": socket.gethostname(),
            "gpu_count": 1,
            "gpus": [
                {
                    "name": raw.get("device_name"),
                    "gfx": None,
                    "vram_bytes": parse_int(raw.get("device_total_global_mem")),
                    "ullm_device_index": parse_int(raw.get("device_index")),
                    "runtime_backend": raw.get("backend"),
                }
            ],
            "cpu": None,
            "driver": None,
            "runtime": raw.get("backend"),
        },
        "parallelism": {
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "data_parallel": 1,
        },
        "workload": {
            "context_length": parse_int(raw.get("final_sequence_len"))
            or ((prompt or 0) + (generated or 0) or None),
            "prompt_tokens": prompt,
            "generated_tokens": generated,
            "batch_size": 1,
            "concurrent_requests": 1,
            "kv_cache_dtype": "f32" if kv_bytes else None,
            "sampling": "greedy",
            "decode_mode": raw.get("decode_mode"),
        },
        "metrics": {
            "prefill_tokens_per_second": prefill_tps(raw),
            "decode_tokens_per_second": decode_tokens_per_second,
            "total_tokens_per_second": total_tokens_per_second(raw),
            "prefill_wall_time_seconds": prefill_wall_ms / 1000.0
            if prefill_wall_ms is not None
            else None,
            "decode_wall_time_seconds": decode_wall_ms / 1000.0
            if decode_wall_ms is not None
            else None,
            "total_wall_time_seconds": total_seconds,
            "time_to_first_token_ms": prefill_wall_ms,
            "time_per_output_token_ms": 1000.0 / decode_tokens_per_second
            if decode_tokens_per_second
            else None,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "vram_baseline_bytes": parse_int(memory.get("vram_baseline_bytes")),
            "vram_peak_bytes": parse_int(memory.get("vram_peak_bytes")),
            "vram_consumed_bytes": consumed_bytes,
            "decode_tokens_per_second_times_vram_consumed_gib": decode_times_vram,
            "power_watts_avg": None,
        },
        "memory": {
            "backend": raw.get("backend"),
            "sample_interval_seconds": None,
            "sample_count": 0,
            "baseline_total_bytes": parse_int(memory.get("vram_baseline_bytes")),
            "peak_total_bytes": parse_int(memory.get("vram_peak_bytes")),
            "consumed_total_bytes": consumed_bytes,
            "baseline_by_card_bytes": {},
            "peak_by_card_bytes": {},
            "consumed_by_card_bytes": {},
            "kv_cache_bytes": kv_bytes,
            "kv_cache_bytes_estimated": kv_estimated,
            "kv_cache_allocated_blocks": parse_int(memory.get("cache_blocks")),
            "kv_cache_free_blocks": None,
            "kv_cache_block_size": parse_int(memory.get("block_size")),
            "log": None,
        },
        "correctness": {
            "reference": "none",
            "reference_artifact": None,
            "logits_relative_mse": None,
            "logits_max_abs_diff": None,
            "top_k": parse_int(raw.get("top_k")),
            "top_k_agreement": None,
            "generated_prefix_matches_reference": is_verified,
            "nan_count": 0 if nan_or_inf is False else None,
            "inf_count": 0 if nan_or_inf is False else None,
            "logit_min": None,
            "logit_max": None,
        },
        "artifacts": {
            "command": None,
            "stdout_log": None,
            "stderr_log": None,
            "memory_log": None,
            "raw_json": str(path),
        },
        "error": None if is_verified is not False else {"type": "verification_failed", "message": "raw smoke JSON was not verified"},
        "notes": [
            "Normalized from package-token-ids-generate-smoke raw JSON.",
        ],
    }
    if kv_estimated and kv_bytes is not None:
        record["notes"].append("KV cache bytes estimated as f32 from self_attn and cache block metadata.")
    return record


def write_jsonl(path: Path, rows: list[dict[str, Any]], run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            record = normalized_row(row["path"], row["raw"], run_id)
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            output.write("\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize pre-SQ runtime package-token-ids smoke JSON files as a "
            "Markdown table, with optional inference-benchmark-result-v0.1-style JSONL."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=f"raw JSON files or glob patterns (default: {DEFAULT_GLOB})",
    )
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        help="write normalized inference-benchmark-result-v0.1-style JSONL rows",
    )
    parser.add_argument(
        "--run-id",
        help="run_id to use for --jsonl-out rows (default: DATE-presq-runtime-tps)",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    paths = expand_inputs(args.inputs)
    rows = sorted((build_summary_row(path, load_json(path)) for path in paths), key=sort_key)
    run_date = extract_date(paths) or "unknown-date"
    run_id = args.run_id or f"{run_date}-presq-runtime-tps"

    if args.jsonl_out:
        write_jsonl(args.jsonl_out, rows, run_id)

    print(render_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
