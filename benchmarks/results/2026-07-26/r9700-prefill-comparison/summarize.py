#!/usr/bin/env python3
"""Normalize raw prefill runs into machine-readable comparison artifacts."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
PROMPTS = (128, 512, 1024, 2048, 4095)
ROOF_BPS = 640_000_000_000

# Common SQ8_0 logical denominator, documented in accounting.md.
PROJECTION_ELEMENTS = 13_212_057_600
SQ8_PROJECTION_PAYLOAD = 13_212_057_600
SQ8_PROJECTION_SCALES = 1_612_800
SQ8_LM_HEAD_BYTES = 1_555_824_640
LM_HEAD_ELEMENTS = 777_912_320
GGUF_Q8_PROJECTION_BYTES = 14_037_811_200
GGUF_Q8_OUTPUT_BYTES = 826_531_840
KV_READ_COEFFICIENT = 1_638_400
KV_WRITE_COEFFICIENT = 327_680
ATTENTION_FLOP_COEFFICIENT = 819_200


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def work_terms(n: int) -> dict[str, int]:
    chunks = math.ceil(n / 128)
    causal_pairs = n * (n + 1) // 2
    kv_read = KV_READ_COEFFICIENT * causal_pairs
    kv_write = KV_WRITE_COEFFICIENT * n
    common = (
        chunks * (SQ8_PROJECTION_PAYLOAD + SQ8_PROJECTION_SCALES)
        + SQ8_LM_HEAD_BYTES
        + kv_read
        + kv_write
    )
    flops = (
        n * 2 * PROJECTION_ELEMENTS
        + 2 * LM_HEAD_ELEMENTS
        + ATTENTION_FLOP_COEFFICIENT * causal_pairs
    )
    return {
        "prompt_tokens": n,
        "m128_chunks": chunks,
        "causal_pairs": causal_pairs,
        "common_projection_bytes": chunks * (SQ8_PROJECTION_PAYLOAD + SQ8_PROJECTION_SCALES),
        "common_lm_head_bytes": SQ8_LM_HEAD_BYTES,
        "logical_kv_read_bytes": kv_read,
        "logical_kv_write_bytes": kv_write,
        "common_logical_bytes": common,
        "flops_lower_bound": flops,
    }


def format_aware_bytes(n: int, engine_key: str) -> int:
    chunks = math.ceil(n / 128)
    causal_pairs = n * (n + 1) // 2
    if engine_key == "ullm":
        return (
            chunks * (SQ8_PROJECTION_PAYLOAD + SQ8_PROJECTION_SCALES)
            + SQ8_LM_HEAD_BYTES
            + KV_READ_COEFFICIENT * causal_pairs
            + KV_WRITE_COEFFICIENT * n
        )
    storage_bytes = 4 if engine_key == "llama_f32" else 2
    kv_read = 40 * 40 * (128 + 128) * storage_bytes * causal_pairs
    kv_write = 40 * 8 * (128 + 128) * storage_bytes * n
    return chunks * GGUF_Q8_PROJECTION_BYTES + GGUF_Q8_OUTPUT_BYTES + kv_read + kv_write


def parse_ullm(condition_dir: Path, n: int) -> tuple[list[float], dict[str, Any]]:
    records = read_jsonl(condition_dir / "stdout.log")
    config = next(record for record in records if record.get("event") == "configuration")
    samples = [
        record
        for record in records
        if record.get("event") == "measured_region" and record.get("phase") == "prefill"
    ]
    summary = next(
        record
        for record in records
        if record.get("event") == "summary" and record.get("phase") == "prefill"
    )
    if len(samples) != 5 or config.get("prompt_tokens") != n or summary.get("units_per_repeat") != n:
        raise RuntimeError(f"invalid uLLM prefill result at N={n}")
    if config.get("device", {}).get("gcn_arch_name") != "gfx1201":
        raise RuntimeError(f"uLLM device validation failed at N={n}")
    return [float(record["elapsed_seconds"]) for record in samples], {
        "timer": "synchronized prefill advance loop",
        "excluded": ["model_load", "same_length_warmup", "request_start", "finish_and_reset"],
        "configuration": config,
        "summary": summary,
        "prefill_advance_calls": [record.get("prefill_advance_calls") for record in samples],
    }


def parse_llama(condition_dir: Path, n: int, kv: str) -> tuple[list[float], dict[str, Any]]:
    records = read_json(condition_dir / "stdout.log")
    if not isinstance(records, list) or len(records) != 1:
        raise RuntimeError(f"invalid llama.cpp JSON at N={n}, KV={kv}")
    record = records[0]
    expected = {
        "n_prompt": n,
        "n_gen": 0,
        "n_batch": n,
        "n_ubatch": 128,
        "type_k": kv,
        "type_v": kv,
        "n_gpu_layers": 999,
        "flash_attn": 1,
        "devices": "ROCm0",
        "build_commit": "68a5592",
    }
    mismatch = {
        name: {"expected": value, "actual": record.get(name)}
        for name, value in expected.items()
        if record.get(name) != value
    }
    if mismatch:
        raise RuntimeError(f"llama.cpp condition mismatch at N={n}, KV={kv}: {mismatch}")
    elapsed = [float(value) / 1e9 for value in record["samples_ns"]]
    if len(elapsed) != 5:
        raise RuntimeError(f"llama.cpp expected five samples at N={n}, KV={kv}")
    return elapsed, {
        "timer": "llama-bench prompt-only; source confirms model load and warm-up precede timer and test_prompt synchronizes",
        "raw_row": record,
    }


def marker_time(condition_dir: Path, engine_key: str) -> int | None:
    for event in read_jsonl(condition_dir / "stream-events.jsonl"):
        line = event.get("line", "")
        if engine_key == "ullm":
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if parsed.get("event") == "configuration":
                return event.get("monotonic_ns")
        elif "prompt run 1/5" in line:
            return event.get("monotonic_ns")
    return None


def thermal(condition_dir: Path, engine_key: str) -> dict[str, Any]:
    samples = read_jsonl(condition_dir / "amd-smi-metric.jsonl")
    summaries = [sample.get("summary") for sample in samples if isinstance(sample.get("summary"), dict)]
    marker = marker_time(condition_dir, engine_key)
    start_sample = next(
        (
            sample.get("summary")
            for sample in samples
            if marker is not None
            and sample.get("monotonic_started_ns", 0) >= marker
            and isinstance(sample.get("summary"), dict)
        ),
        None,
    )
    process_start = next(
        (sample.get("summary") for sample in samples if sample.get("marker") == "immediately-before-process"),
        None,
    )

    def numeric(name: str) -> list[float]:
        return [float(sample[name]) for sample in summaries if isinstance(sample.get(name), (int, float))]

    def compact(sample: Any) -> dict[str, Any] | None:
        if not isinstance(sample, dict):
            return None
        names = ("edge_c", "hotspot_c", "mem_c", "socket_power_w", "gfx_clock_mhz", "mem_clock_mhz", "throttle_status")
        return {name: sample.get(name) for name in names}

    fields = ("edge_c", "hotspot_c", "mem_c", "socket_power_w", "gfx_clock_mhz", "mem_clock_mhz")
    return {
        "sample_count": len(summaries),
        "process_start": compact(process_start),
        "timed_start_nearest_sample": compact(start_sample),
        "minimum": {field: min(numeric(field)) if numeric(field) else None for field in fields},
        "maximum": {field: max(numeric(field)) if numeric(field) else None for field in fields},
        "throttle_status_values": sorted({str(sample.get("throttle_status")) for sample in summaries if sample.get("throttle_status") is not None}),
        "timed_start_event_monotonic_ns": marker,
    }


def row(engine_key: str, n: int, kv: str) -> dict[str, Any]:
    if engine_key == "ullm":
        condition_id = f"ullm-sq8_0-f32-kv-p{n}"
        elapsed, evidence = parse_ullm(RAW / condition_id, n)
        engine = "uLLM SQ8_0"
        storage_key = "ullm"
        execution_units = evidence["prefill_advance_calls"]
    else:
        condition_id = f"llama-cpp-q8_0-{kv}-kv-p{n}"
        elapsed, evidence = parse_llama(RAW / condition_id, n, kv)
        engine = "llama.cpp Q8_0"
        storage_key = f"llama_{kv}"
        execution_units = [math.ceil(n / 128)] * len(elapsed)
    terms = work_terms(n)
    mean_seconds = statistics.fmean(elapsed)
    rates = [n / value for value in elapsed]
    common_bytes = terms["common_logical_bytes"]
    logical_gbps = common_bytes / mean_seconds / 1e9
    return {
        "id": condition_id,
        "engine_key": engine_key,
        "engine": engine,
        "kv_dtype": kv,
        "prompt_tokens": n,
        "repetitions": len(elapsed),
        "elapsed_seconds": elapsed,
        "mean_elapsed_seconds": mean_seconds,
        "tok_s": n / mean_seconds,
        "median_tok_s": statistics.median(rates),
        "sample_stdev_tok_s": statistics.stdev(rates),
        "per_repeat_tok_s": rates,
        "common_logical_bytes": common_bytes,
        "format_aware_lower_bound_bytes": format_aware_bytes(n, storage_key),
        "logical_bandwidth_gb_s": logical_gbps,
        "logical_roof_ratio_to_640_gb_s": logical_gbps / 640.0,
        "physical_hbm_efficiency": None,
        "achieved_tflop_s_lower_bound": terms["flops_lower_bound"] / mean_seconds / 1e12,
        "logical_kv_share": (terms["logical_kv_read_bytes"] + terms["logical_kv_write_bytes"]) / common_bytes,
        "work_terms": terms,
        "thermal": thermal(RAW / condition_id, engine_key),
        "prefill_execution_units": execution_units,
        "evidence": evidence,
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "prompt_tokens", "engine", "kv_dtype", "tok_s", "median_tok_s",
        "sample_stdev_tok_s", "logical_bandwidth_gb_s",
        "logical_roof_ratio_to_640_gb_s", "physical_hbm_efficiency",
        "achieved_tflop_s_lower_bound",
        "logical_kv_share", "mean_elapsed_seconds", "common_logical_bytes",
        "format_aware_lower_bound_bytes",
    ]
    with (ROOT / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in rows:
            writer.writerow({field: result[field] for field in fields})

    thermal_fields = [
        "condition_id", "utc_started", "utc_ended", "marker", "edge_c",
        "hotspot_c", "mem_c", "socket_power_w", "gfx_clock_mhz",
        "mem_clock_mhz", "throttle_status", "gfx_activity_pct", "umc_activity_pct",
    ]
    with (ROOT / "thermal-history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=thermal_fields, lineterminator="\n")
        writer.writeheader()
        for result in rows:
            for sample in read_jsonl(RAW / result["id"] / "amd-smi-metric.jsonl"):
                summary = sample.get("summary")
                if not isinstance(summary, dict):
                    continue
                writer.writerow({
                    "condition_id": result["id"],
                    "utc_started": sample.get("utc_started"),
                    "utc_ended": sample.get("utc_ended"),
                    "marker": sample.get("marker"),
                    **{field: summary.get(field) for field in thermal_fields[4:]},
                })


def write_commands(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Executed commands",
        "",
        "The credential-bearing sudo input is intentionally absent.  The one service window was launched with run-service-window.sh after the approved sudo credential priming step.  Non-secret stop/start invocations are retained under service/.",
        "",
        "## Service wrapper and final restoration",
        "",
        "    ./run-service-window.sh",
        "",
        "The wrapper's credential-free `sudo -n systemctl start ullm-openai.service` attempt is recorded in service/restore.txt and returned 1 after the sudo credential expired.  The later approved restoration is represented without the credential as:",
        "",
        "    sudo -S -p '' systemctl start ullm-openai.service",
        "",
        "Its stdin credential is intentionally not stored.  The timestamped final state and the intervening worker-EOF observation are in service/final-recovery.md.",
        "",
        "## Per-condition model commands",
        "",
    ]
    for result in rows:
        command = (RAW / result["id"] / "command.txt").read_text(encoding="utf-8").strip()
        lines.extend([f"### {result['id']}", "", "    " + command.replace("\n", "\n    "), ""])
    (ROOT / "commands.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows: list[dict[str, Any]] = []
    for n in PROMPTS:
        rows.extend((row("ullm", n, "f32"), row("llama", n, "f32"), row("llama", n, "f16")))
    for n in PROMPTS:
        ullm = next(result for result in rows if result["engine_key"] == "ullm" and result["prompt_tokens"] == n)
        for result in (item for item in rows if item["engine_key"] == "llama" and item["prompt_tokens"] == n):
            result["ratio_to_ullm_tok_s"] = result["tok_s"] / ullm["tok_s"]
            result["ullm_to_this_tok_s"] = ullm["tok_s"] / result["tok_s"]

    summary = {
        "schema_version": "ullm.r9700.prefill-comparison.summary.v1",
        "date": "2026-07-26",
        "device": {"amd_smi_gpu": 2, "pci_bdf": "0000:47:00.0", "gfx": "gfx1201", "r9700_only": True},
        "workload": {
            "prompt_lengths": list(PROMPTS),
            "repetitions": 5,
            "single_stream": True,
            "uLLM_max_new_tokens": 1,
            "llama_cpp_n_gen": 0,
            "uLLM_prefill_mode": "m128-chunk128",
            "llama_cpp_ubatch": 128,
        },
        "normalized_accounting": {
            "roof_decimal_gb_s": 640,
            "policy": "same SQ8_0 projection/scales, BF16 LM head, and F32-equivalent Q-head-expanded GQA causal KV numerator for every row; see accounting.md",
            "physical_hbm_efficiency": "unconfirmed; the common logical numerator can exceed 640 GB/s when an implementation reuses/fuses logical GQA operands",
        },
        "results": rows,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (ROOT / "r9700-prefill-comparison.jsonl").open("w", encoding="utf-8") as handle:
        for result in rows:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    write_csv(rows)
    write_commands(rows)


if __name__ == "__main__":
    main()
