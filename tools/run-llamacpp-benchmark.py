#!/usr/bin/env python3
"""Run llama.cpp throughput baselines and write uLLM benchmark JSONL."""

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


def csv_ints(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def parse_model(value: str) -> dict[str, str]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("model must be NAME:QUANTIZATION:PATH")
    name, quant, path = parts
    if not name or not quant or not path:
        raise argparse.ArgumentTypeError("model must be NAME:QUANTIZATION:PATH")
    return {"name": name, "quantization": quant, "path": path}


def parse_target(value: str) -> dict[str, str]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("target must be LABEL:DEVICES:SPLIT_MODE")
    label, devices, split_mode = parts
    if not label or not devices or not split_mode:
        raise argparse.ArgumentTypeError("target must be LABEL:DEVICES:SPLIT_MODE")
    return {"label": label, "devices": devices, "split_mode": split_mode}


def split_devices(value: str) -> list[str]:
    return [item for item in re.split(r"[,/]", value) if item]


def run_text(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
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


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def git_value(repo: Path, *args: str) -> str | None:
    return run_text(["git", "-C", str(repo), *args])


def read_rocm_vram() -> dict[str, dict[str, Any]]:
    output = run_text(["rocm-smi", "--showmeminfo", "vram", "--json"])
    if not output:
        return {}
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return {}

    result: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if not key.startswith("card") or not isinstance(value, dict):
            continue
        used = parse_int(value.get("VRAM Total Used Memory (B)"))
        total = parse_int(value.get("VRAM Total Memory (B)"))
        result[key] = {
            "used_bytes": used,
            "total_bytes": total,
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


def parse_devices(llama_bench: Path) -> dict[str, dict[str, Any]]:
    output = run_text([str(llama_bench), "--list-devices"]) or ""
    devices: dict[str, dict[str, Any]] = {}
    init_pattern = re.compile(r"^\s+Device\s+(\d+):\s+(.+?),\s+(gfx[0-9a-fA-F]+).*VRAM:\s+(\d+)\s+MiB")
    for line in output.splitlines():
        match = init_pattern.match(line)
        if not match:
            continue
        idx, name, gfx, mib = match.groups()
        devices[f"ROCm{idx}"] = {
            "name": name,
            "gfx": gfx,
            "vram_bytes": int(mib) * 1024 * 1024,
        }
    pattern = re.compile(r"^\s+(ROCm\d+):\s+(.+?)\s+\((\d+)\s+MiB,")
    for line in output.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        device, name, mib = match.groups()
        current = devices.get(device, {})
        devices[device] = {
            "name": name,
            "gfx": current.get("gfx"),
            "vram_bytes": current.get("vram_bytes", int(mib) * 1024 * 1024),
        }
    return devices


def rocm_driver() -> str | None:
    output = run_text(["rocm-smi", "--showdriverversion", "--json"])
    if not output:
        return None
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return None
    system = data.get("system")
    if isinstance(system, dict):
        return system.get("Driver version")
    return None


def hip_version() -> str | None:
    for command in (["/opt/rocm/bin/hipconfig", "--version"], ["hipconfig", "--version"]):
        value = run_text(command)
        if value:
            return value.splitlines()[0].strip()
    return None


def command_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def json_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return rows


def batch_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("model_filename"),
        row.get("n_batch"),
        row.get("n_ubatch"),
        row.get("type_k"),
        row.get("type_v"),
        row.get("devices"),
        row.get("split_mode"),
        row.get("flash_attn"),
    )


def case_id(*parts: Any) -> str:
    text = "-".join(str(part) for part in parts if part is not None)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def build_hardware(
    row: dict[str, Any] | None,
    target: dict[str, str],
    devices: dict[str, dict[str, Any]],
    driver: str | None,
    runtime: str | None,
) -> dict[str, Any]:
    requested = split_devices(target["devices"])
    gpu_infos = []
    for device in requested:
        info = devices.get(device, {})
        gpu_infos.append(
            {
                "name": info.get("name", device),
                "gfx": info.get("gfx"),
                "vram_bytes": info.get("vram_bytes"),
            }
        )
    return {
        "host": socket.gethostname(),
        "gpu_count": len(gpu_infos),
        "gpus": gpu_infos,
        "cpu": row.get("cpu_info") if row else None,
        "driver": driver,
        "runtime": f"ROCm HIP {runtime}" if runtime else "ROCm HIP",
    }


def build_parallelism(target: dict[str, str]) -> dict[str, int]:
    gpu_count = len(split_devices(target["devices"]))
    return {
        "tensor_parallel": gpu_count if target["split_mode"] == "tensor" else 1,
        "pipeline_parallel": 1,
        "data_parallel": 1,
    }


def build_artifacts(command: list[str], stdout_log: Path, stderr_log: Path, memory_log: Path | None = None) -> dict[str, str | None]:
    return {
        "command": command_string(command),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "memory_log": str(memory_log) if memory_log else None,
    }


def total_tokens_per_second(prompt: int, gen: int, pp_ts: float, tg_ts: float) -> float | None:
    if pp_ts <= 0 or tg_ts <= 0:
        return None
    seconds = prompt / pp_ts + gen / tg_ts
    if seconds <= 0:
        return None
    return (prompt + gen) / seconds


def gib(value: int | None) -> float | None:
    if value is None:
        return None
    return value / 1024**3


def decode_memory_product(decode_tokens_per_second: float, consumed_bytes: int | None) -> float | None:
    consumed_gib = gib(consumed_bytes)
    if consumed_gib is None:
        return None
    return decode_tokens_per_second * consumed_gib


def result_row(
    *,
    args: argparse.Namespace,
    run_id: str,
    model: dict[str, str],
    target: dict[str, str],
    prompt: int,
    gen: int,
    prefill: dict[str, Any],
    decode: dict[str, Any],
    command: list[str],
    stdout_log: Path,
    stderr_log: Path,
    memory_log: Path,
    memory: dict[str, Any] | None,
    devices: dict[str, dict[str, Any]],
    driver: str | None,
    runtime: str | None,
    engine_version: str | None,
) -> dict[str, Any]:
    pp_ts = float(prefill.get("avg_ts", 0.0))
    tg_ts = float(decode.get("avg_ts", 0.0))
    split_mode = target["split_mode"]
    gpu_count = len(split_devices(target["devices"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "case_id": case_id(
            "llamacpp",
            model["name"],
            model["quantization"],
            target["label"],
            split_mode,
            f"gpu{gpu_count}",
            f"ctx{prompt + gen}",
            f"b{prefill.get('n_batch')}",
            f"ub{prefill.get('n_ubatch')}",
            f"pp{prompt}",
            f"tg{gen}",
        ),
        "status": "ok",
        "engine": {
            "name": "llama.cpp",
            "version": engine_version,
            "commit": prefill.get("build_commit") or decode.get("build_commit"),
        },
        "model": {
            "name": model["name"],
            "source": "local",
            "revision": None,
            "format": "gguf",
            "quantization": model["quantization"],
        },
        "hardware": build_hardware(prefill, target, devices, driver, runtime),
        "parallelism": build_parallelism(target),
        "workload": {
            "context_length": prompt + gen,
            "prompt_tokens": prompt,
            "generated_tokens": gen,
            "batch_size": prefill.get("n_batch"),
            "concurrent_requests": 1,
            "kv_cache_dtype": f"{prefill.get('type_k')}/{prefill.get('type_v')}",
        },
        "metrics": {
            "prefill_tokens_per_second": pp_ts,
            "decode_tokens_per_second": tg_ts,
            "total_tokens_per_second": total_tokens_per_second(prompt, gen, pp_ts, tg_ts),
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "vram_baseline_bytes": memory.get("baseline_total_bytes") if memory else None,
            "vram_peak_bytes": memory.get("peak_total_bytes") if memory else None,
            "vram_consumed_bytes": memory.get("consumed_total_bytes") if memory else None,
            "decode_tokens_per_second_times_vram_consumed_gib": decode_memory_product(
                tg_ts,
                memory.get("consumed_total_bytes") if memory else None,
            ),
            "power_watts_avg": None,
        },
        "memory": memory,
        "artifacts": build_artifacts(command, stdout_log, stderr_log, memory_log),
        "error": None,
        "notes": [
            "llama-bench reports separate prompt-processing and token-generation rows; total token/s is computed from those two averages.",
            "context_length records prompt_tokens + generated_tokens because this llama-bench mode does not expose an independent n_ctx sweep.",
            "VRAM consumed bytes are process-window peak total used bytes minus pre-command total used bytes from rocm-smi.",
            f"llama.cpp split_mode={split_mode}; this is not the same as vLLM/SGLang pipeline parallelism.",
        ],
    }


def failure_rows(
    *,
    args: argparse.Namespace,
    run_id: str,
    model: dict[str, str],
    target: dict[str, str],
    command: list[str],
    stdout_log: Path,
    stderr_log: Path,
    memory_log: Path,
    memory: dict[str, Any] | None,
    devices: dict[str, dict[str, Any]],
    driver: str | None,
    runtime: str | None,
    engine_version: str | None,
    returncode: int,
    stderr: str,
    status_override: str | None = None,
    error_type: str | None = None,
    message_override: str | None = None,
) -> list[dict[str, Any]]:
    message = message_override or (stderr.strip().splitlines()[-1] if stderr.strip() else f"llama-bench exited with {returncode}")
    lowered = stderr.lower()
    status = status_override or (
        "oom" if "out of memory" in lowered or ("memory" in lowered and "failed" in lowered) else "failed"
    )
    rows: list[dict[str, Any]] = []
    for prompt in args.prompts:
        for gen in args.gens:
            for batch in args.batches:
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": run_id,
                        "case_id": case_id(
                            "llamacpp",
                            model["name"],
                            model["quantization"],
                            target["label"],
                            target["split_mode"],
                            f"ctx{prompt + gen}",
                            f"b{batch}",
                            f"pp{prompt}",
                            f"tg{gen}",
                        ),
                        "status": status,
                        "engine": {
                            "name": "llama.cpp",
                            "version": engine_version,
                            "commit": args.engine_commit,
                        },
                        "model": {
                            "name": model["name"],
                            "source": "local",
                            "revision": None,
                            "format": "gguf",
                            "quantization": model["quantization"],
                        },
                        "hardware": build_hardware(None, target, devices, driver, runtime),
                        "parallelism": build_parallelism(target),
                        "workload": {
                            "context_length": prompt + gen,
                            "prompt_tokens": prompt,
                            "generated_tokens": gen,
                            "batch_size": batch,
                            "concurrent_requests": 1,
                            "kv_cache_dtype": f"{args.cache_type_k}/{args.cache_type_v}",
                        },
                        "metrics": None,
                        "memory": memory,
                        "artifacts": build_artifacts(command, stdout_log, stderr_log, memory_log),
                        "error": {
                            "type": error_type or status,
                            "message": message,
                        },
                        "notes": ["llama-bench did not produce usable JSONL rows for this command."],
                    }
                )
    return rows


def run_llama_bench(
    *,
    args: argparse.Namespace,
    run_id: str,
    model: dict[str, str],
    target: dict[str, str],
    logs_dir: Path,
    devices: dict[str, dict[str, Any]],
    driver: str | None,
    runtime: str | None,
    engine_version: str | None,
) -> list[dict[str, Any]]:
    log_base = case_id("raw", model["name"], model["quantization"], target["label"], target["split_mode"])
    stdout_log = logs_dir / f"{log_base}.stdout.jsonl"
    stderr_log = logs_dir / f"{log_base}.stderr.log"
    memory_log = logs_dir / f"{log_base}.memory.jsonl"
    command = [
        str(args.llama_bench),
        "-m",
        model["path"],
        "-p",
        ",".join(str(item) for item in args.prompts),
        "-n",
        ",".join(str(item) for item in args.gens),
        "-b",
        ",".join(str(item) for item in args.batches),
        "-ub",
        str(args.ubatch),
        "-ctk",
        args.cache_type_k,
        "-ctv",
        args.cache_type_v,
        "-ngl",
        str(args.gpu_layers),
        "-sm",
        target["split_mode"],
        "-dev",
        target["devices"],
        "-r",
        str(args.repetitions),
        "-o",
        "jsonl",
    ]
    if args.flash_attn is not None:
        command.extend(["-fa", args.flash_attn])

    env = os.environ.copy()
    env.setdefault("HIP_VISIBLE_DEVICES", "0,1,2")
    monitor = None if args.no_memory_monitor else RocmMemoryMonitor(memory_log, args.memory_sample_interval)
    if monitor is not None:
        monitor.start()
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=args.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        memory = monitor.stop() if monitor is not None else None
        stdout = as_text(exc.stdout)
        stderr = as_text(exc.stderr)
        stdout_log.write_text(stdout, encoding="utf-8")
        stderr_log.write_text(stderr, encoding="utf-8")
        return failure_rows(
            args=args,
            run_id=run_id,
            model=model,
            target=target,
            command=command,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            memory_log=memory_log,
            memory=memory,
            devices=devices,
            driver=driver,
            runtime=runtime,
            engine_version=engine_version,
            returncode=-1,
            stderr=stderr,
            status_override="failed",
            error_type="timeout",
            message_override=f"llama-bench exceeded timeout_seconds={args.timeout_seconds}",
        )
    memory = monitor.stop() if monitor is not None else None
    stdout_log.write_text(completed.stdout, encoding="utf-8")
    stderr_log.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        return failure_rows(
            args=args,
            run_id=run_id,
            model=model,
            target=target,
            command=command,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            memory_log=memory_log,
            memory=memory,
            devices=devices,
            driver=driver,
            runtime=runtime,
            engine_version=engine_version,
            returncode=completed.returncode,
            stderr=completed.stderr,
        )

    rows = json_rows(completed.stdout)
    if not rows:
        return failure_rows(
            args=args,
            run_id=run_id,
            model=model,
            target=target,
            command=command,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            memory_log=memory_log,
            memory=memory,
            devices=devices,
            driver=driver,
            runtime=runtime,
            engine_version=engine_version,
            returncode=completed.returncode,
            stderr=completed.stderr,
        )

    prefills: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = {}
    decodes: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = {}
    for row in rows:
        key = batch_key(row)
        if int(row.get("n_prompt", 0)) > 0 and int(row.get("n_gen", 0)) == 0:
            prefills.setdefault(key, {})[int(row["n_prompt"])] = row
        elif int(row.get("n_prompt", 0)) == 0 and int(row.get("n_gen", 0)) > 0:
            decodes.setdefault(key, {})[int(row["n_gen"])] = row

    result: list[dict[str, Any]] = []
    for key, prompt_rows in prefills.items():
        decode_rows = decodes.get(key, {})
        for prompt in args.prompts:
            prefill = prompt_rows.get(prompt)
            if not prefill:
                continue
            for gen in args.gens:
                decode = decode_rows.get(gen)
                if not decode:
                    continue
                result.append(
                    result_row(
                        args=args,
                        run_id=run_id,
                        model=model,
                        target=target,
                        prompt=prompt,
                        gen=gen,
                        prefill=prefill,
                        decode=decode,
                        command=command,
                        stdout_log=stdout_log,
                        stderr_log=stderr_log,
                        memory_log=memory_log,
                        memory=memory,
                        devices=devices,
                        driver=driver,
                        runtime=runtime,
                        engine_version=engine_version,
                    )
                )
    if result:
        return result
    return failure_rows(
        args=args,
        run_id=run_id,
        model=model,
        target=target,
        command=command,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        memory_log=memory_log,
        memory=memory,
        devices=devices,
        driver=driver,
        runtime=runtime,
        engine_version=engine_version,
        returncode=completed.returncode,
        stderr="llama-bench JSONL had no matching prefill/decode pairs",
    )


def unsupported_rows(
    *,
    run_id: str,
    model: dict[str, str],
    target: dict[str, str],
    devices: dict[str, dict[str, Any]],
    driver: str | None,
    runtime: str | None,
    commits: dict[str, str | None],
) -> list[dict[str, Any]]:
    engines = [
        ("vLLM", "vllm", "V620 is not an initial execution target for vLLM in this project."),
        ("SGLang", "sglang", "V620 is not an initial execution target for SGLang in this project."),
        ("ROCm/ATOM", "atom", "ROCm/ATOM is recorded as reference code first; V620 execution is deferred."),
        ("TensorRT-LLM", "tensorrt-llm", "TensorRT-LLM is not an AMD GPU target."),
    ]
    rows = []
    for engine_name, key, message in engines:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "case_id": case_id(engine_name, "unsupported", target["label"], model["name"]),
                "status": "unsupported",
                "engine": {
                    "name": engine_name,
                    "version": None,
                    "commit": commits.get(key),
                },
                "model": {
                    "name": model["name"],
                    "source": "local",
                    "revision": None,
                    "format": "gguf",
                    "quantization": model["quantization"],
                },
                "hardware": build_hardware(None, target, devices, driver, runtime),
                "parallelism": {"tensor_parallel": 1, "pipeline_parallel": 1, "data_parallel": 1},
                "workload": {
                    "context_length": 640,
                    "prompt_tokens": 512,
                    "generated_tokens": 128,
                    "batch_size": 1,
                    "concurrent_requests": 1,
                    "kv_cache_dtype": "f16/f16",
                },
                "metrics": None,
                "memory": None,
                "artifacts": {"command": None, "stdout_log": None, "stderr_log": None, "memory_log": None},
                "error": {"type": "unsupported_hardware", "message": message},
                "notes": ["Unsupported rows are recorded before MI300X/NVIDIA test environments are available."],
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            output.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-bench", type=Path, default=Path("build/reference/llama.cpp-hip/bin/llama-bench"))
    parser.add_argument("--llama-repo", type=Path, default=Path("reference-src/llama.cpp"))
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--target", action="append", type=parse_target, required=True)
    parser.add_argument("--prompts", type=csv_ints, default=csv_ints("128,512,2048"))
    parser.add_argument("--gens", type=csv_ints, default=csv_ints("128,512"))
    parser.add_argument("--batches", type=csv_ints, default=csv_ints("512,2048"))
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--flash-attn", choices=["on", "off", "auto"], default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--memory-sample-interval", type=float, default=1.0)
    parser.add_argument("--no-memory-monitor", action="store_true")
    parser.add_argument("--write-unsupported", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.engine_commit = git_value(args.llama_repo, "rev-parse", "HEAD")
    return args


def main() -> int:
    args = parse_args()
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    run_id = args.run_id or f"{today}-llamacpp-rocm-baseline"
    out_dir = args.output_root / today / "llama.cpp"
    logs_dir = out_dir / "logs" / run_id
    logs_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{run_id}.jsonl"
    engine_version = git_value(args.llama_repo, "describe", "--tags", "--always", "--dirty")
    devices = parse_devices(args.llama_bench)
    driver = rocm_driver()
    runtime = hip_version()

    reference_root = args.llama_repo.parent
    commits = {
        "llama.cpp": args.engine_commit,
        "vllm": git_value(reference_root / "vllm", "rev-parse", "HEAD"),
        "sglang": git_value(reference_root / "sglang", "rev-parse", "HEAD"),
        "atom": git_value(reference_root / "atom", "rev-parse", "HEAD"),
        "tensorrt-llm": git_value(reference_root / "tensorrt-llm", "rev-parse", "HEAD"),
    }

    print(f"run_id={run_id}", file=sys.stderr)
    print(f"result_path={result_path}", file=sys.stderr)
    print(f"llama.cpp={engine_version} {args.engine_commit}", file=sys.stderr)

    if args.dry_run:
        for model in args.model:
            for target in args.target:
                print(f"would run model={model['name']} target={target['label']}", file=sys.stderr)
        return 0

    if result_path.exists():
        result_path.unlink()

    all_rows: list[dict[str, Any]] = []
    if args.write_unsupported:
        all_rows.extend(
            unsupported_rows(
                run_id=run_id,
                model=args.model[0],
                target=args.target[0],
                devices=devices,
                driver=driver,
                runtime=runtime,
                commits=commits,
            )
        )
        write_rows(result_path, all_rows)
        all_rows = []

    for model in args.model:
        for target in args.target:
            print(f"running model={model['name']} target={target['label']} split={target['split_mode']}", file=sys.stderr)
            rows = run_llama_bench(
                args=args,
                run_id=run_id,
                model=model,
                target=target,
                logs_dir=logs_dir,
                devices=devices,
                driver=driver,
                runtime=runtime,
                engine_version=engine_version,
            )
            write_rows(result_path, rows)

    print(f"wrote {sum(1 for _ in result_path.open(encoding='utf-8'))} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
