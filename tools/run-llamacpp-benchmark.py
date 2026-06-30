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


def git_value(repo: Path, *args: str) -> str | None:
    return run_text(["git", "-C", str(repo), *args])


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
    requested = [item.strip() for item in target["devices"].split(",") if item.strip()]
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
    gpu_count = len([item for item in target["devices"].split(",") if item.strip()])
    return {
        "tensor_parallel": gpu_count if target["split_mode"] == "tensor" else 1,
        "pipeline_parallel": 1,
        "data_parallel": 1,
    }


def build_artifacts(command: list[str], stdout_log: Path, stderr_log: Path) -> dict[str, str]:
    return {
        "command": command_string(command),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }


def total_tokens_per_second(prompt: int, gen: int, pp_ts: float, tg_ts: float) -> float | None:
    if pp_ts <= 0 or tg_ts <= 0:
        return None
    seconds = prompt / pp_ts + gen / tg_ts
    if seconds <= 0:
        return None
    return (prompt + gen) / seconds


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
    devices: dict[str, dict[str, Any]],
    driver: str | None,
    runtime: str | None,
    engine_version: str | None,
) -> dict[str, Any]:
    pp_ts = float(prefill.get("avg_ts", 0.0))
    tg_ts = float(decode.get("avg_ts", 0.0))
    split_mode = target["split_mode"]
    gpu_count = len([item for item in target["devices"].split(",") if item.strip()])
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
            "vram_peak_bytes": None,
            "power_watts_avg": None,
        },
        "artifacts": build_artifacts(command, stdout_log, stderr_log),
        "error": None,
        "notes": [
            "llama-bench reports separate prompt-processing and token-generation rows; total token/s is computed from those two averages.",
            "context_length records prompt_tokens + generated_tokens because this llama-bench mode does not expose an independent n_ctx sweep.",
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
    devices: dict[str, dict[str, Any]],
    driver: str | None,
    runtime: str | None,
    engine_version: str | None,
    returncode: int,
    stderr: str,
) -> list[dict[str, Any]]:
    message = stderr.strip().splitlines()[-1] if stderr.strip() else f"llama-bench exited with {returncode}"
    lowered = stderr.lower()
    status = "oom" if "out of memory" in lowered or ("memory" in lowered and "failed" in lowered) else "failed"
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
                        "artifacts": build_artifacts(command, stdout_log, stderr_log),
                        "error": {
                            "type": status,
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
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
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
                "artifacts": {"command": None, "stdout_log": None, "stderr_log": None},
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
    logs_dir = out_dir / "logs"
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
