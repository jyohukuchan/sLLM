#!/usr/bin/env python3
"""Run an external inference benchmark command and write one uLLM JSONL row."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "inference-benchmark-result-v0.1"


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


def run_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def read_rocm_info() -> dict[str, Any]:
    output = run_text(
        ["rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "vram", "--json"]
    )
    if not output:
        return {}
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {}


def read_rocm_vram() -> dict[str, dict[str, Any]]:
    data = read_rocm_info()
    result: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if not key.startswith("card") or not isinstance(value, dict):
            continue
        result[key] = {
            "used_bytes": parse_int(value.get("VRAM Total Used Memory (B)")),
            "total_bytes": parse_int(value.get("VRAM Total Memory (B)")),
            "series": value.get("Card Series"),
            "gfx": value.get("GFX Version"),
        }
    return result


def total_used_bytes(sample: dict[str, dict[str, Any]]) -> int | None:
    values = [info.get("used_bytes") for info in sample.values()]
    numeric = [value for value in values if isinstance(value, int)]
    if not numeric:
        return None
    return sum(numeric)


def used_by_card(sample: dict[str, dict[str, Any]]) -> dict[str, int]:
    return {
        card: value
        for card, info in sample.items()
        if isinstance((value := info.get("used_bytes")), int)
    }


class RocmMemoryMonitor:
    def __init__(self, log_path: Path, interval_seconds: float) -> None:
        self.log_path = log_path
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline: dict[str, dict[str, Any]] = {}
        self.peak_by_card: dict[str, int] = {}
        self.peak_total_bytes: int | None = None
        self.sample_count = 0

    def _write_sample(self, stage: str, sample: dict[str, dict[str, Any]]) -> None:
        self.sample_count += 1
        total = total_used_bytes(sample)
        if total is not None:
            self.peak_total_bytes = max(total, self.peak_total_bytes or total)
        for card, used in used_by_card(sample).items():
            self.peak_by_card[card] = max(used, self.peak_by_card.get(card, used))
        row = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "stage": stage,
            "total_used_bytes": total,
            "cards": sample,
        }
        with self.log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            output.write("\n")

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_path.exists():
            self.log_path.unlink()
        self.baseline = read_rocm_vram()
        self._write_sample("baseline", self.baseline)
        self._thread = threading.Thread(target=self._run, name="rocm-memory-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._write_sample("sample", read_rocm_vram())

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self._write_sample("final", read_rocm_vram())
        return self.summary()

    def summary(self) -> dict[str, Any]:
        baseline_by_card = used_by_card(self.baseline)
        baseline_total = total_used_bytes(self.baseline)
        consumed_by_card = {
            card: max(0, peak - baseline_by_card.get(card, 0))
            for card, peak in self.peak_by_card.items()
        }
        consumed_total = None
        if baseline_total is not None and self.peak_total_bytes is not None:
            consumed_total = max(0, self.peak_total_bytes - baseline_total)
        return {
            "backend": "rocm-smi",
            "sample_interval_seconds": self.interval_seconds,
            "sample_count": self.sample_count,
            "baseline_total_bytes": baseline_total,
            "peak_total_bytes": self.peak_total_bytes,
            "consumed_total_bytes": consumed_total,
            "baseline_by_card_bytes": baseline_by_card,
            "peak_by_card_bytes": self.peak_by_card,
            "consumed_by_card_bytes": consumed_by_card,
            "log": str(self.log_path),
        }


def rocm_driver() -> str | None:
    data = read_rocm_info()
    system = data.get("system")
    if isinstance(system, dict):
        return system.get("Driver version")
    return None


def rocm_runtime() -> str | None:
    for path in (Path("/opt/rocm/.info/version"), Path("/opt/rocm-7.2.1/.info/version")):
        if path.exists():
            return f"ROCm {path.read_text(encoding='utf-8').strip()}"
    return None


def selected_gpus(cards: list[str]) -> list[dict[str, Any]]:
    data = read_rocm_info()
    gpus: list[dict[str, Any]] = []
    for card in cards:
        info = data.get(card)
        if not isinstance(info, dict):
            continue
        gpus.append(
            {
                "name": info.get("Card Series"),
                "gfx": info.get("GFX Version"),
                "vram_bytes": parse_int(info.get("VRAM Total Memory (B)")),
                "rocm_smi_card": card,
            }
        )
    return gpus


def command_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def env_prefix(names: list[str]) -> str:
    parts: list[str] = []
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            parts.append(f"{name}={shlex.quote(value)}")
    return " ".join(parts)


def parse_vllm_metrics(stdout: str, output_json: Path | None, memory: dict[str, Any]) -> dict[str, Any]:
    elapsed = None
    total_tokens_per_second = None
    requests_per_second = None
    if output_json and output_json.exists():
        try:
            data = json.loads(output_json.read_text(encoding="utf-8"))
            elapsed = parse_float(data.get("elapsed_time"))
            total_tokens_per_second = parse_float(data.get("tokens_per_second"))
            requests_per_second = parse_float(data.get("requests_per_second"))
        except json.JSONDecodeError:
            pass

    decode_tokens_per_second = None
    match = re.search(
        r"Throughput:\s+([0-9.]+)\s+requests/s,\s+([0-9.]+)\s+total tokens/s,\s+([0-9.]+)\s+output tokens/s",
        stdout,
    )
    if match:
        requests_per_second = parse_float(match.group(1))
        total_tokens_per_second = parse_float(match.group(2))
        decode_tokens_per_second = parse_float(match.group(3))

    total_prompt_tokens = None
    match = re.search(r"Total num prompt tokens:\s+([0-9]+)", stdout)
    if match:
        total_prompt_tokens = parse_int(match.group(1))

    prefill_tokens_per_second = None
    if elapsed and total_prompt_tokens is not None and elapsed > 0:
        prefill_tokens_per_second = total_prompt_tokens / elapsed

    consumed_bytes = memory.get("consumed_total_bytes")
    consumed_gib = consumed_bytes / 1024**3 if isinstance(consumed_bytes, int) else None
    product = None
    if decode_tokens_per_second is not None and consumed_gib is not None:
        product = decode_tokens_per_second * consumed_gib

    metrics = {
        "prefill_tokens_per_second": prefill_tokens_per_second,
        "decode_tokens_per_second": decode_tokens_per_second,
        "total_tokens_per_second": total_tokens_per_second,
        "requests_per_second": requests_per_second,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "vram_baseline_bytes": memory.get("baseline_total_bytes"),
        "vram_peak_bytes": memory.get("peak_total_bytes"),
        "vram_consumed_bytes": consumed_bytes,
        "decode_tokens_per_second_times_vram_consumed_gib": product,
        "power_watts_avg": None,
    }
    return metrics


def parse_sglang_serving_metrics(output_json: Path | None, memory: dict[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if output_json and output_json.exists():
        text = output_json.read_text(encoding="utf-8")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            lines = [line for line in text.splitlines() if line.strip()]
            if lines:
                parsed = json.loads(lines[-1])
                if isinstance(parsed, dict):
                    data = parsed

    decode_tokens_per_second = parse_float(data.get("output_throughput"))
    total_tokens_per_second = parse_float(
        data.get("total_throughput", data.get("total_token_throughput"))
    )
    prefill_tokens_per_second = parse_float(data.get("input_throughput"))
    if (
        prefill_tokens_per_second is None
        and total_tokens_per_second is not None
        and decode_tokens_per_second is not None
    ):
        prefill_tokens_per_second = total_tokens_per_second - decode_tokens_per_second
    consumed_bytes = memory.get("consumed_total_bytes")
    consumed_gib = consumed_bytes / 1024**3 if isinstance(consumed_bytes, int) else None
    product = None
    if decode_tokens_per_second is not None and consumed_gib is not None:
        product = decode_tokens_per_second * consumed_gib

    return {
        "prefill_tokens_per_second": prefill_tokens_per_second,
        "decode_tokens_per_second": decode_tokens_per_second,
        "total_tokens_per_second": total_tokens_per_second,
        "requests_per_second": parse_float(data.get("request_throughput")),
        "latency_p50_ms": parse_float(
            data.get("median_e2e_latency_ms", data.get("median_e2el_ms"))
        ),
        "latency_p95_ms": parse_float(data.get("p95_e2e_latency_ms", data.get("p95_e2el_ms"))),
        "vram_baseline_bytes": memory.get("baseline_total_bytes"),
        "vram_peak_bytes": memory.get("peak_total_bytes"),
        "vram_consumed_bytes": consumed_bytes,
        "decode_tokens_per_second_times_vram_consumed_gib": product,
        "power_watts_avg": None,
    }


def default_metrics(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "prefill_tokens_per_second": None,
        "decode_tokens_per_second": None,
        "total_tokens_per_second": None,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "vram_baseline_bytes": memory.get("baseline_total_bytes"),
        "vram_peak_bytes": memory.get("peak_total_bytes"),
        "vram_consumed_bytes": memory.get("consumed_total_bytes"),
        "decode_tokens_per_second_times_vram_consumed_gib": None,
        "power_watts_avg": None,
    }


def classify_failure(returncode: int, timed_out: bool, text: str) -> tuple[str, dict[str, str]]:
    lowered = text.lower()
    if timed_out:
        return "failed", {"type": "timeout", "message": "Benchmark command timed out."}
    if "out of memory" in lowered or "memoryerror" in lowered or "hip out of memory" in lowered:
        return "oom", {"type": "oom", "message": "Benchmark command reported out-of-memory."}
    if "unsupported" in lowered or "not supported" in lowered:
        return "unsupported", {"type": "unsupported_runtime", "message": "Benchmark command reported unsupported runtime or model path."}
    return "failed", {"type": "command_failed", "message": f"Benchmark command exited with code {returncode}."}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        output.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--stdout-log", required=True, type=Path)
    parser.add_argument("--stderr-log", required=True, type=Path)
    parser.add_argument("--memory-log", required=True, type=Path)
    parser.add_argument("--engine-name", required=True)
    parser.add_argument("--engine-version")
    parser.add_argument("--engine-commit")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-source", default="huggingface")
    parser.add_argument("--model-revision")
    parser.add_argument("--model-format", required=True)
    parser.add_argument("--model-quantization", required=True)
    parser.add_argument("--gpu-card", action="append", default=[])
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--pipeline-parallel", type=int, default=1)
    parser.add_argument("--data-parallel", type=int, default=1)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--generated-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--concurrent-requests", type=int, default=1)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--parse", choices=["none", "vllm-throughput", "sglang-serving"], default="none")
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--memory-sample-interval", type=float, default=1.0)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--", dest="separator", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command after --")

    args.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    args.stderr_log.parent.mkdir(parents=True, exist_ok=True)
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        if args.result_json.exists():
            args.result_json.unlink()

    monitor = RocmMemoryMonitor(args.memory_log, args.memory_sample_interval)
    monitor.start()
    started = time.monotonic()
    timed_out = False
    with args.stdout_log.open("w", encoding="utf-8") as stdout, args.stderr_log.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(args.command, stdout=stdout, stderr=stderr, text=True)
        try:
            returncode = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                returncode = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
    elapsed_seconds = time.monotonic() - started
    memory = monitor.stop()

    stdout_text = args.stdout_log.read_text(encoding="utf-8", errors="replace")
    stderr_text = args.stderr_log.read_text(encoding="utf-8", errors="replace")
    combined = stdout_text + "\n" + stderr_text

    if returncode == 0 and not timed_out:
        status = "ok"
        error = None
    else:
        status, error = classify_failure(returncode, timed_out, combined)

    if args.parse == "vllm-throughput" and status == "ok":
        metrics = parse_vllm_metrics(stdout_text, args.result_json, memory)
    elif args.parse == "sglang-serving" and status == "ok":
        metrics = parse_sglang_serving_metrics(args.result_json, memory)
    else:
        metrics = default_metrics(memory)

    env_names = [
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "VLLM_LOGGING_LEVEL",
        "VLLM_TARGET_DEVICE",
        "SGLANG_USE_AITER",
        "ATOM_USE_UNIFIED_ATTN",
        "ATOM_USE_TRITON_GEMM",
        "AITER_ROPE_NATIVE_BACKEND",
        "AITER_LOG_LEVEL",
        "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT",
        "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT",
        "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION",
        "PYTORCH_HIP_ALLOC_CONF",
        "HSA_OVERRIDE_GFX_VERSION",
    ]
    prefix = env_prefix(env_names)
    command = command_string(args.command)
    if prefix:
        command = f"{prefix} {command}"

    row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "case_id": args.case_id,
        "status": status,
        "engine": {
            "name": args.engine_name,
            "version": args.engine_version,
            "commit": args.engine_commit,
        },
        "model": {
            "name": args.model_name,
            "source": args.model_source,
            "revision": args.model_revision,
            "format": args.model_format,
            "quantization": args.model_quantization,
        },
        "hardware": {
            "host": socket.gethostname(),
            "gpu_count": len(args.gpu_card),
            "gpus": selected_gpus(args.gpu_card),
            "cpu": None,
            "driver": rocm_driver(),
            "runtime": rocm_runtime(),
        },
        "parallelism": {
            "tensor_parallel": args.tensor_parallel,
            "pipeline_parallel": args.pipeline_parallel,
            "data_parallel": args.data_parallel,
        },
        "workload": {
            "context_length": args.context_length,
            "prompt_tokens": args.prompt_tokens,
            "generated_tokens": args.generated_tokens,
            "batch_size": args.batch_size,
            "concurrent_requests": args.concurrent_requests,
            "kv_cache_dtype": args.kv_cache_dtype,
        },
        "metrics": metrics,
        "memory": memory,
        "artifacts": {
            "command": command,
            "stdout_log": str(args.stdout_log),
            "stderr_log": str(args.stderr_log),
            "memory_log": str(args.memory_log),
            "result_json": str(args.result_json) if args.result_json else None,
            "elapsed_seconds": elapsed_seconds,
        },
        "error": error,
        "notes": args.note,
    }
    append_jsonl(args.output_jsonl, row)
    print(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
