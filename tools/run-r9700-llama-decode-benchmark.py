#!/usr/bin/env python3
"""Run one R9700-only llama.cpp steady-decode case with raw evidence.

The runner intentionally keeps the measurement in ``llama-bench`` rather than
timing an HTTP client.  In particular, ``-d 1028 -p 0 -n 16`` preloads an
unmeasured 1,028-token KV state and measures 16 individual M=1 decode calls.
This is the cache window used by the R9700 SQ8_0 reference: 1028 -> 1044.

It is deliberately narrow in scope: it neither starts nor stops services and
never selects an unmasked device.  The caller is responsible for the service
isolation window; this script records enough raw evidence to audit the GPU
selection and thermal conditions inside that window.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


R9700_AMD_SMI_INDEX = "2"
R9700_HIP_VISIBLE_INDEX = "1"
EXPECTED_GFX = "gfx1201"
FORBIDDEN_GFX = "gfx1030"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_text(command: list[str], env: dict[str, str]) -> str:
    return "env -u ROCR_VISIBLE_DEVICES " + " ".join(
        [f"HIP_VISIBLE_DEVICES={shlex.quote(env['HIP_VISIBLE_DEVICES'])}", shlex.join(command)]
    )


def r9700_env() -> dict[str, str]:
    env = os.environ.copy()
    # Combining ROCr and HIP masks has different ordinal semantics on this host.
    # HIP=1 was independently discovered to expose only the R9700 as ROCm0.
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env["HIP_VISIBLE_DEVICES"] = R9700_HIP_VISIBLE_INDEX
    return env


def run_capture(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, env=env, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class MetricSampler:
    def __init__(self, output: Path, interval_seconds: float, gpu: str) -> None:
        self.output = output
        self.interval_seconds = interval_seconds
        self.gpu = gpu
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.samples: list[dict[str, Any]] = []

    def snapshot(self, stage: str) -> dict[str, Any]:
        command = ["amd-smi", "metric", "--gpu", self.gpu, "--json"]
        completed = run_capture(command)
        row: dict[str, Any] = {
            "timestamp_utc": utc_now(),
            "monotonic_seconds": time.monotonic(),
            "stage": stage,
            "command": shlex.join(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        try:
            row["metric"] = json.loads(completed.stdout)
        except json.JSONDecodeError:
            row["metric"] = None
        with self.lock:
            self.samples.append(row)
            with self.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
        return row

    def start(self) -> None:
        self.snapshot("before")
        self.thread = threading.Thread(target=self._run, name="r9700-amd-smi", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self.snapshot("sample")

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(2.0, self.interval_seconds * 3))
        self.snapshot("after")


def require_r9700(llama_bench: Path, output: Path, env: dict[str, str]) -> dict[str, Any]:
    command = [str(llama_bench), "--list-devices"]
    completed = run_capture(command, env=env)
    write_text(output / "device-discovery.stdout", completed.stdout)
    write_text(output / "device-discovery.stderr", completed.stderr)
    observed = completed.stdout + completed.stderr
    valid = completed.returncode == 0 and EXPECTED_GFX in observed and FORBIDDEN_GFX not in observed
    result = {
        "command": command_text(command, env),
        "returncode": completed.returncode,
        "expected_gfx": EXPECTED_GFX,
        "forbidden_gfx": FORBIDDEN_GFX,
        "valid": valid,
    }
    write_json(output / "device-selection.json", result)
    if not valid:
        raise RuntimeError(
            "R9700-only device validation failed: expected exactly a gfx1201 discovery with no gfx1030 present"
        )
    return result


def pick_benchmark_row(stdout: str) -> dict[str, Any] | None:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list):
        return None
    for row in value:
        if isinstance(row, dict) and row.get("n_prompt") == 0 and row.get("n_gen") == 16 and row.get("n_depth") == 1028:
            return row
    return None


def sample_statistics(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    samples_ns = row.get("samples_ns")
    if not isinstance(samples_ns, list) or not all(isinstance(value, (int, float)) for value in samples_ns):
        return None
    elapsed_seconds = [float(value) / 1_000_000_000.0 for value in samples_ns]
    tps = [16.0 / value for value in elapsed_seconds]
    return {
        "repetitions": len(samples_ns),
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": tps,
        "mean_elapsed_seconds": statistics.mean(elapsed_seconds),
        "mean_duration_tokens_per_second": 16.0 / statistics.mean(elapsed_seconds),
        "mean_per_repeat_tokens_per_second": statistics.mean(tps),
        "median_tokens_per_second": statistics.median(tps),
        "sample_variance_tokens_per_second": statistics.variance(tps) if len(tps) > 1 else None,
        "population_variance_tokens_per_second": statistics.pvariance(tps) if len(tps) > 1 else 0.0,
        "sample_stdev_tokens_per_second": statistics.stdev(tps) if len(tps) > 1 else None,
        "median_elapsed_seconds": statistics.median(elapsed_seconds),
    }


def telemetry_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract the requested R9700 fields without discarding raw JSONL."""

    fields = {
        "hotspot_celsius": ("temperature", "hotspot", "value"),
        "gfx_clock_mhz": ("clock", "gfx_0", "clk", "value"),
        "socket_power_watts": ("power", "socket_power", "value"),
        "vram_used_bytes": ("memory", "vram", "used", "value"),
    }
    values: dict[str, list[float]] = {name: [] for name in fields}
    throttle_values: list[str] = []
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None

    for sample in samples:
        metric = sample.get("metric")
        gpu_data = metric.get("gpu_data") if isinstance(metric, dict) else None
        gpu = gpu_data[0] if isinstance(gpu_data, list) and gpu_data and isinstance(gpu_data[0], dict) else None
        if gpu is None:
            continue
        extracted: dict[str, Any] = {}
        for name, path in fields.items():
            current: Any = gpu
            for part in path:
                current = current.get(part) if isinstance(current, dict) else None
            extracted[name] = current
            if isinstance(current, (int, float)):
                values[name].append(float(current))
        power = gpu.get("power")
        throttle = power.get("throttle_status") if isinstance(power, dict) else None
        extracted["throttle_status"] = throttle
        if isinstance(throttle, str):
            throttle_values.append(throttle)
        if sample.get("stage") == "before":
            before = extracted
        if sample.get("stage") == "after":
            after = extracted

    return {
        "before": before,
        "after": after,
        "minimum": {name: min(series) if series else None for name, series in values.items()},
        "maximum": {name: max(series) if series else None for name, series in values.items()},
        "throttle_status_values": sorted(set(throttle_values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-bench", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv-type", choices=("f16", "f32"), required=True)
    parser.add_argument("--flash-attn", choices=("on", "off", "auto"), required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ubatch-size", type=int, default=128)
    parser.add_argument("--telemetry-interval", type=float, default=0.25)
    parser.add_argument("--amd-smi-gpu", default=R9700_AMD_SMI_INDEX)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions != 5:
        raise SystemExit("this controlled comparison requires exactly five repetitions")
    if not args.llama_bench.is_file() or not os.access(args.llama_bench, os.X_OK):
        raise SystemExit(f"llama-bench is not executable: {args.llama_bench}")
    if not args.model.is_file():
        raise SystemExit(f"model is not a file: {args.model}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing result directory: {args.output}")
    args.output.mkdir(parents=True)

    env = r9700_env()
    selection = require_r9700(args.llama_bench, args.output, env)
    command = [
        str(args.llama_bench),
        "-m", str(args.model),
        "-o", "json",
        "-r", str(args.repetitions),
        "-p", "0",
        "-n", "16",
        "-d", "1028",
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "-ctk", args.kv_type,
        "-ctv", args.kv_type,
        "-ngl", "999",
        "-sm", "none",
        "-mg", "0",
        "-dev", "ROCm0",
        "-nkvo", "0",
        "-fa", args.flash_attn,
        "-t", str(args.threads),
        "-mmp", "1",
    ]
    write_text(args.output / "command.txt", command_text(command, env) + "\n")

    telemetry = MetricSampler(args.output / "telemetry.jsonl", args.telemetry_interval, args.amd_smi_gpu)
    started = utc_now()
    monotonic_started = time.monotonic()
    telemetry.start()
    completed = run_capture(command, env=env)
    telemetry.stop()
    monotonic_finished = time.monotonic()
    finished = utc_now()
    write_text(args.output / "llama-bench.stdout", completed.stdout)
    write_text(args.output / "llama-bench.stderr", completed.stderr)

    row = pick_benchmark_row(completed.stdout)
    stats = sample_statistics(row)
    summary = {
        "schema_version": "ullm.r9700.external-engine.llama-decode.v1",
        "engine": "llama.cpp",
        "status": "ok" if completed.returncode == 0 and stats is not None else "failed",
        "device_selection": selection,
        "command": command_text(command, env),
        "benchmark_returncode": completed.returncode,
        "started_utc": started,
        "finished_utc": finished,
        "process_elapsed_seconds": monotonic_finished - monotonic_started,
        "measurement": {
            "profiled": False,
            "repetitions": args.repetitions,
            "single_stream": True,
            "prompt_tokens_in_timed_region": 0,
            "cache_length_start": 1028,
            "cache_length_end": 1044,
            "cache_length_midpoint": 1036,
            "generated_tokens_per_repeat": 16,
            "decode_batch_size": 1,
            "prefill_batch_size": args.batch_size,
            "prefill_ubatch_size": args.ubatch_size,
            "kv_type_k": args.kv_type,
            "kv_type_v": args.kv_type,
            "gpu_layers_requested": 999,
            "gpu_devices_requested": "ROCm0",
            "split_mode": "none",
            "flash_attention_requested": args.flash_attn,
            "llama_bench_default_warmup_enabled": True,
            "timing_excludes": ["depth_prefill", "tokenization", "sampling", "model_load"],
        },
        "llama_bench_row": row,
        "statistics": stats,
        "telemetry": {
            "amd_smi_gpu": args.amd_smi_gpu,
            "interval_seconds": args.telemetry_interval,
            "sample_count": len(telemetry.samples),
            "raw_jsonl": "telemetry.jsonl",
            "summary": telemetry_summary(telemetry.samples),
        },
    }
    write_json(args.output / "summary.json", summary)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
