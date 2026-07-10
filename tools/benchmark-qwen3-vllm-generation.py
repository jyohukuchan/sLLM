#!/usr/bin/env python3
"""Benchmark the fixed Qwen3-14B-FP8 M=8/G=8 vLLM generation case."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ullm.qwen3_vllm_generation_benchmark.v1"
ORACLE_EXPORTER = Path(__file__).with_name(
    "export-qwen3-vllm-generation-oracle.py"
)
DEFAULT_OUTPUT = Path(
    "/tmp/ullm-qwen3-14b-fp8-vllm-generation-throughput-m8-g8-v0.1.json"
)
PROMPT_TOKEN_IDS = tuple(range(1, 9))
EXPECTED_GENERATED_TOKEN_IDS = (353, 10, 4999, 1725, 15, 16, 17, 18)
GENERATION_STEPS = 8
WARMUP_RUNS = 3
MEASURED_REPEATS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_oracle_exporter(path: Path = ORACLE_EXPORTER) -> Any:
    spec = importlib.util.spec_from_file_location(
        "ullm_qwen3_vllm_generation_oracle_exporter", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load trusted oracle exporter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile_linear(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between zero and 100")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in ordered):
        raise ValueError("timing samples must be finite and non-negative")
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def timing_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    seconds = {
        "min": min(values),
        "p50": percentile_linear(values, 50.0),
        "p95": percentile_linear(values, 95.0),
        "max": max(values),
        "mean": sum(values) / len(values),
    }
    return {
        "count": len(values),
        "seconds": seconds,
        "milliseconds": {name: value * 1000.0 for name, value in seconds.items()},
        "percentile_method": "linear_interpolation_rank_(n-1)*p",
    }


def validate_generation_output(requests: Any) -> tuple[Any, Any]:
    if not isinstance(requests, list) or len(requests) != 1:
        raise RuntimeError("vLLM returned an unexpected request count")
    request = requests[0]
    outputs = getattr(request, "outputs", None)
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise RuntimeError("vLLM returned an unexpected completion count")
    output = outputs[0]
    token_ids = tuple(int(value) for value in output.token_ids)
    if token_ids != EXPECTED_GENERATED_TOKEN_IDS:
        raise RuntimeError(
            "vLLM generated token IDs outside the fixed oracle contract: "
            f"{list(token_ids)}"
        )
    if str(output.finish_reason) != "length":
        raise RuntimeError(
            f"vLLM finish reason was {output.finish_reason!r}, expected 'length'"
        )
    if getattr(request, "finished", None) is not True:
        raise RuntimeError("vLLM did not mark the fixed request as finished")
    return request, output


def request_output_metrics(request: Any) -> dict[str, Any]:
    metrics = getattr(request, "metrics", None)
    if metrics is None:
        return {
            "available": False,
            "unavailable_reason": "RequestOutput.metrics is None",
            "ttft_seconds": None,
            "decode_seconds": None,
            "raw_timestamps": None,
        }
    required = (
        "first_token_latency",
        "first_token_ts",
        "last_token_ts",
        "num_generation_tokens",
    )
    missing = [name for name in required if not hasattr(metrics, name)]
    if missing:
        return {
            "available": False,
            "unavailable_reason": "RequestOutput.metrics lacks: " + ", ".join(missing),
            "ttft_seconds": None,
            "decode_seconds": None,
            "raw_timestamps": None,
        }
    ttft = float(metrics.first_token_latency)
    first_token_ts = float(metrics.first_token_ts)
    last_token_ts = float(metrics.last_token_ts)
    decode = last_token_ts - first_token_ts
    if (
        not math.isfinite(ttft)
        or not math.isfinite(first_token_ts)
        or not math.isfinite(last_token_ts)
        or ttft < 0.0
        or first_token_ts <= 0.0
        or decode < 0.0
        or int(metrics.num_generation_tokens) != GENERATION_STEPS
    ):
        return {
            "available": False,
            "unavailable_reason": "RequestOutput.metrics failed the fixed timing contract",
            "ttft_seconds": None,
            "decode_seconds": None,
            "raw_timestamps": None,
        }
    return {
        "available": True,
        "unavailable_reason": None,
        "ttft_seconds": ttft,
        "decode_seconds": decode,
        "raw_timestamps": {
            "first_token_ts": first_token_ts,
            "last_token_ts": last_token_ts,
        },
    }


def summarize_request_output_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    unavailable = [
        str(record["unavailable_reason"])
        for record in records
        if record["available"] is not True
    ]
    if unavailable:
        return {
            "available": False,
            "unavailable_reason": "; ".join(sorted(set(unavailable))),
            "ttft": None,
            "decode": None,
        }
    return {
        "available": True,
        "unavailable_reason": None,
        "ttft": timing_summary([float(record["ttft_seconds"]) for record in records]),
        "decode": timing_summary(
            [float(record["decode_seconds"]) for record in records]
        ),
    }


def process_max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def publish_json_no_clobber(
    output: Path, payload: dict[str, Any], rename_noreplace: Any
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.incomplete-", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        rename_noreplace(temporary, output)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def shutdown_runtime(llm: Any) -> dict[str, Any]:
    warnings: list[str] = []
    if llm is not None:
        engine = getattr(llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "shutdown"):
            try:
                engine.shutdown()
            except Exception as error:  # pragma: no cover - runtime-specific cleanup
                warnings.append(f"engine shutdown: {error!r}")
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as error:  # pragma: no cover - runtime-specific cleanup
        warnings.append(f"distributed cleanup: {error!r}")
    return {
        "engine_shutdown_attempted": llm is not None,
        "distributed_cleanup_attempted": True,
        "torch_cuda_empty_cache_attempted": False,
        "post_cleanup_torch_memory_allocated_bytes": None,
        "post_cleanup_torch_memory_reserved_bytes": None,
        "warnings": warnings,
        "process_releases_gpu_on_exit": True,
    }


def finalize_gpu_cleanup(cleanup: dict[str, Any], torch: Any) -> None:
    gc.collect()
    if torch is None or not torch.cuda.is_available():
        return
    cleanup["torch_cuda_empty_cache_attempted"] = True
    try:
        torch.cuda.empty_cache()
        cleanup["post_cleanup_torch_memory_allocated_bytes"] = int(
            torch.cuda.memory_allocated(0)
        )
        cleanup["post_cleanup_torch_memory_reserved_bytes"] = int(
            torch.cuda.memory_reserved(0)
        )
    except Exception as error:  # pragma: no cover - runtime-specific cleanup
        cleanup["warnings"].append(f"GPU cache cleanup: {error!r}")


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    output = Path(os.path.abspath(args.output.expanduser()))
    benchmark_path = Path(__file__).resolve()
    benchmark_bytes = benchmark_path.read_bytes()
    benchmark_sha256 = hashlib.sha256(benchmark_bytes).hexdigest()
    exporter_path = ORACLE_EXPORTER.resolve()
    exporter_bytes = exporter_path.read_bytes()
    exporter_sha256 = hashlib.sha256(exporter_bytes).hexdigest()

    oracle = load_oracle_exporter(exporter_path)
    oracle.ensure_output_available(output)
    if os.environ.get("ROCR_VISIBLE_DEVICES") != "1":
        raise SystemExit("ROCR_VISIBLE_DEVICES must be exactly 1 for the local R9700")
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") not in (None, "0"):
        raise SystemExit("VLLM_ENABLE_V1_MULTIPROCESSING must be 0")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if Path(sys.executable).resolve() != oracle.DEFAULT_PYTHON.resolve():
        raise SystemExit(
            f"benchmark must run with the fixed interpreter: {oracle.DEFAULT_PYTHON}"
        )

    initial_model_snapshot = oracle.verify_model_contract(model_dir)
    config, revision, checkpoint_files = initial_model_snapshot
    oracle.load_runtime_dependencies()
    gpu = oracle.visible_gpu()
    llm: Any = None
    measured_records: list[dict[str, Any]] = []
    cleanup: dict[str, Any]
    try:
        llm = oracle.LLM(
            model=str(model_dir),
            tokenizer=str(model_dir),
            dtype="auto",
            quantization="fp8",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            max_model_len=len(PROMPT_TOKEN_IDS) + GENERATION_STEPS,
            max_num_seqs=1,
            max_num_batched_tokens=len(PROMPT_TOKEN_IDS),
            kv_cache_memory_bytes=oracle.KV_CACHE_MEMORY_BYTES,
            enforce_eager=True,
            enable_prefix_caching=False,
            async_scheduling=False,
            disable_log_stats=True,
            seed=0,
        )
        sampling = oracle.SamplingParams(
            temperature=0.0,
            max_tokens=GENERATION_STEPS,
            min_tokens=GENERATION_STEPS,
            ignore_eos=True,
            seed=0,
        )
        prompt = [{"prompt_token_ids": list(PROMPT_TOKEN_IDS)}]
        for _ in range(WARMUP_RUNS):
            validate_generation_output(llm.generate(prompt, sampling, use_tqdm=False))
        for repeat_index in range(MEASURED_REPEATS):
            started_ns = time.perf_counter_ns()
            requests = llm.generate(prompt, sampling, use_tqdm=False)
            finished_ns = time.perf_counter_ns()
            request, output_record = validate_generation_output(requests)
            measured_records.append(
                {
                    "repeat_index": repeat_index,
                    "wall_latency_seconds": (finished_ns - started_ns) / 1_000_000_000.0,
                    "generated_token_ids": [int(value) for value in output_record.token_ids],
                    "finish_reason": str(output_record.finish_reason),
                    "request_output_metrics": request_output_metrics(request),
                }
            )
    finally:
        cleanup_llm = llm
        llm = None
        cleanup = shutdown_runtime(cleanup_llm)
        del cleanup_llm
        finalize_gpu_cleanup(cleanup, oracle.torch)

    if len(measured_records) != MEASURED_REPEATS:
        raise RuntimeError("benchmark did not complete all measured repeats")
    final_model_snapshot = oracle.verify_model_contract(model_dir)
    oracle.require_unchanged(
        "checkpoint files and revision metadata",
        initial_model_snapshot,
        final_model_snapshot,
    )
    oracle.require_unchanged(
        "benchmark script",
        benchmark_sha256,
        hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
    )
    oracle.require_unchanged(
        "trusted oracle exporter",
        exporter_sha256,
        hashlib.sha256(exporter_path.read_bytes()).hexdigest(),
    )

    wall_latencies = [float(record["wall_latency_seconds"]) for record in measured_records]
    measured_seconds = sum(wall_latencies)
    request_metrics = [record["request_output_metrics"] for record in measured_records]
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "name": "Qwen/Qwen3-14B-FP8",
            "local_dir": str(model_dir),
            "revision": revision,
            "checkpoint_files": checkpoint_files,
            "config": config,
        },
        "prompt": {
            "token_ids": list(PROMPT_TOKEN_IDS),
            "position_ids": list(range(len(PROMPT_TOKEN_IDS))),
            "attention": "causal",
            "bos_inserted": False,
            "chat_template_applied": False,
        },
        "generation": {
            "method": "greedy",
            "temperature": 0.0,
            "max_new_tokens": GENERATION_STEPS,
            "min_new_tokens": GENERATION_STEPS,
            "ignore_eos": True,
            "early_stop_on_eos": False,
            "eos_token_id": int(config["eos_token_id"]),
            "expected_generated_token_ids": list(EXPECTED_GENERATED_TOKEN_IDS),
            "all_repeats_match_expected": True,
            "finish_reason": "length",
            "seed": 0,
        },
        "execution": {
            "backend": "vLLM",
            "runner": "LLM.generate",
            "model_load_in_measurement": False,
            "warmup_in_measurement": False,
            "warmup_runs": WARMUP_RUNS,
            "measured_repeats": MEASURED_REPEATS,
            "concurrent_requests": 1,
            "dtype": "bfloat16",
            "quantization": config["quantization_config"],
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "max_model_len": len(PROMPT_TOKEN_IDS) + GENERATION_STEPS,
            "max_num_seqs": 1,
            "max_num_batched_tokens": len(PROMPT_TOKEN_IDS),
            "kv_cache_memory_bytes": oracle.KV_CACHE_MEMORY_BYTES,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "async_scheduling": False,
            "v1_multiprocessing": False,
            "logprobs": None,
        },
        "benchmark": {
            "wall_latency": timing_summary(wall_latencies),
            "aggregate_measured_seconds": measured_seconds,
            "requests_per_second": MEASURED_REPEATS / measured_seconds,
            "generated_tokens_per_second": (
                MEASURED_REPEATS * GENERATION_STEPS / measured_seconds
            ),
            "total_tokens_per_second": (
                MEASURED_REPEATS
                * (len(PROMPT_TOKEN_IDS) + GENERATION_STEPS)
                / measured_seconds
            ),
            "throughput_denominator": "sum_of_measured_LLM.generate_wall_latencies",
            "request_output_metrics": summarize_request_output_metrics(request_metrics),
            "repeat_records": measured_records,
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": {
                name: oracle.package_version(name)
                for name in [
                    "vllm",
                    "torch",
                    "transformers",
                    "safetensors",
                    "accelerate",
                    "triton",
                    "numpy",
                ]
            },
            "torch_git_version": oracle.torch.version.git_version,
            "torch_hip_version": oracle.torch.version.hip,
            "rocm_version_file": (
                Path("/opt/rocm/.info/version").read_text(encoding="ascii").strip()
                if Path("/opt/rocm/.info/version").exists()
                else None
            ),
            "gpu": gpu,
            "process_max_rss_bytes": process_max_rss_bytes(),
        },
        "cleanup": cleanup,
        "trust": {
            "checkpoint_verified_before_and_after": True,
            "benchmark_script_sha256": benchmark_sha256,
            "oracle_exporter_path": str(exporter_path),
            "oracle_exporter_sha256": exporter_sha256,
            "atomic_publish": "renameat2_RENAME_NOREPLACE",
        },
    }
    publish_json_no_clobber(output, result, oracle.rename_noreplace)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": oracle.sha256_file(output),
                "p50_ms": result["benchmark"]["wall_latency"]["milliseconds"]["p50"],
                "p95_ms": result["benchmark"]["wall_latency"]["milliseconds"]["p95"],
                "requests_per_second": result["benchmark"]["requests_per_second"],
                "generated_tokens_per_second": result["benchmark"][
                    "generated_tokens_per_second"
                ],
                "total_tokens_per_second": result["benchmark"][
                    "total_tokens_per_second"
                ],
                "process_max_rss_bytes": result["environment"][
                    "process_max_rss_bytes"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
