#!/usr/bin/env python3
"""Reproducible Qwen3.5-9B EXL3 versus llama.cpp Q4_K_M benchmark.

This runner deliberately keeps the two engines' native timing boundaries visible.
It first creates one frozen prompt manifest with the upstream ``rand_prompt``
recipe, then either runs EXL3 locally or sends those token IDs to a running
llama-server.  The four rows are fixed:

* ``pp128``, ``pp512``, and ``pp2048``: one requested output token;
* ``decode128``: a 128-token prompt and 64 nominal output tokens.

Each row has one warmup and three measured requests.  EXL3 uses
``DefaultSampler`` and seed 1234.  llama.cpp receives ``min_p=0.08``,
``temperature=0.8``, ``samplers=["min_p", "temperature"]``,
``ignore_eos=true``, and ``cache_prompt=false``.

The native boundaries are recorded in every summary.  EXL3's Job implementation
turns ``max_new_tokens=64`` into 63 actual output tokens; its ``time_generate``
spans 63 forwards beginning with the final prompt-token forward.  Its
``time_prefill`` processes ``prompt_tokens - 1`` tokens, while the headline
prefill rate keeps the upstream benchmark's ``prompt_tokens`` numerator;
``processed_tokens`` is recorded separately.
llama.cpp returns 64 output tokens and reports 63 timed decode steps in
``predicted_ms``; its prompt timing includes all ``prompt_n`` tokens.  These
definitions make the report auditable without pretending that the phase
boundaries are identical.

Raw event/response files and summaries are written to the caller-selected
``--output-dir``.  Point that directory outside the repository (for example
``/tmp/qwen35-9b-exl3-bench``); this script does not add benchmark payloads to
Git.  The caller supplies any required ROCr preload, such as
``LD_PRELOAD=/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1``.  EXL3 is unloaded
normally in ``finally`` and this script never uses ``os._exit``.

Examples::

    python ci/tools/benchmark_rocm_exl3_large.py generate-prompts \
      --model-dir /work/models/qwen35-9b-exl3-4bpw \
      --prompts /tmp/qwen35-9b-exl3-bench/prompts.json

    LD_PRELOAD=/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1 \
      python ci/tools/benchmark_rocm_exl3_large.py exl3 \
      --model-dir /work/models/qwen35-9b-exl3-4bpw \
      --prompts /tmp/qwen35-9b-exl3-bench/prompts.json \
      --output-dir /tmp/qwen35-9b-exl3-bench/exl3

    python ci/tools/benchmark_rocm_exl3_large.py llama \
      --prompts /tmp/qwen35-9b-exl3-bench/prompts.json \
      --output-dir /tmp/qwen35-9b-exl3-bench/llama \
      --url http://127.0.0.1:18089

Only standard-library modules are imported before argument parsing, so
``--help`` works on a host without torch, ROCm, or exllamav3 installed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import urllib.request
from typing import Any


SCHEMA_VERSION = "rocm-exl3-large-comparison-v1"
PROMPT_SCHEMA_VERSION = "rocm-exl3-large-prompts-v1"
SEED = 1234
WARMUPS = 1
MEASURED = 3
ROWS = (
    {"case_id": "pp128", "phase": "prefill", "input_tokens": 128, "output_tokens": 1},
    {"case_id": "pp512", "phase": "prefill", "input_tokens": 512, "output_tokens": 1},
    {"case_id": "pp2048", "phase": "prefill", "input_tokens": 2048, "output_tokens": 1},
    {"case_id": "decode128", "phase": "decode", "input_tokens": 128, "output_tokens": 64},
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode_pos",
        nargs="?",
        choices=("generate-prompts", "exl3", "llama"),
        help="operation to perform (also accepted as --mode)",
    )
    parser.add_argument(
        "--mode",
        dest="mode_opt",
        choices=("generate-prompts", "exl3", "llama"),
        help="operation to perform",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="EXL3 model directory (required for generate-prompts and exl3)",
    )
    parser.add_argument(
        "--exl3-repo",
        type=Path,
        default=Path("reference/rocm_exl3"),
        help="checkout containing exllamav3 and rocm_tools/bench_model.py",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="prompt manifest to write (generate-prompts) or read (exl3/llama)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for raw events/responses and the engine summary",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:18089",
        help="llama-server base URL (llama mode)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP request timeout in seconds (llama mode)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=4096,
        help="EXL3 KV cache capacity in tokens (rounded by the upstream Cache)",
    )
    return parser


def selected_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.mode_opt and args.mode_pos and args.mode_opt != args.mode_pos:
        parser.error("positional mode and --mode disagree")
    mode = args.mode_opt or args.mode_pos
    if mode is None:
        parser.error("a mode is required (generate-prompts, exl3, or llama)")
    return mode


def require_mode_args(args: argparse.Namespace, mode: str) -> None:
    if mode in ("generate-prompts", "exl3") and args.model_dir is None:
        raise SystemExit(f"{mode} requires --model-dir")
    if mode in ("exl3", "llama") and args.output_dir is None:
        raise SystemExit(f"{mode} requires --output-dir")
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        raise SystemExit("--timeout must be a finite positive number")
    if args.cache_size <= 0:
        raise SystemExit("--cache-size must be positive")


def jsonable(value: Any) -> Any:
    """Convert an exllamav3 event to JSON without retaining its Job object."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items() if k != "job"}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    # Do not import torch just to identify a tensor.  Tensor-like objects used by
    # exllamav3 expose detach/cpu/tolist; this keeps host --help dependency-free.
    if all(hasattr(value, name) for name in ("detach", "cpu", "tolist")):
        return jsonable(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    return repr(value)


def sha256_json(value: Any) -> str:
    payload = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_manifest_cases(data: dict[str, Any]) -> list[dict[str, Any]]:
    if data.get("schema_version") != PROMPT_SCHEMA_VERSION:
        raise ValueError(f"unsupported prompt manifest schema: {data.get('schema_version')!r}")
    if data.get("seed") != SEED:
        raise ValueError(f"prompt manifest seed must be {SEED}")
    actual_vocab = data.get("tokenizer_actual_vocab_size")
    if not isinstance(actual_vocab, int) or isinstance(actual_vocab, bool) or actual_vocab <= 0:
        raise ValueError("prompt manifest tokenizer_actual_vocab_size must be a positive integer")
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError("prompt manifest cases must be a list")
    expected = {
        (row["case_id"], repeat)
        for row in ROWS
        for repeat in range(WARMUPS + MEASURED)
    }
    observed: set[tuple[str, int]] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each prompt manifest case must be an object")
        key = (case.get("case_id"), case.get("repeat"))
        if key in observed or key not in expected:
            raise ValueError(f"unexpected or duplicate prompt case: {key!r}")
        observed.add(key)
        row = next(row for row in ROWS if row["case_id"] == case["case_id"])
        if case.get("phase") != row["phase"]:
            raise ValueError(f"phase mismatch for {key!r}")
        if case.get("input_tokens") != row["input_tokens"]:
            raise ValueError(f"input length mismatch for {key!r}")
        if case.get("output_tokens") != row["output_tokens"]:
            raise ValueError(f"output length mismatch for {key!r}")
        if case.get("warmup") != (case["repeat"] < WARMUPS):
            raise ValueError(f"warmup marker mismatch for {key!r}")
        ids = case.get("token_ids")
        if not isinstance(ids, list) or len(ids) != row["input_tokens"]:
            raise ValueError(f"token_ids length mismatch for {key!r}")
        if not all(isinstance(token, int) and token >= 0 for token in ids):
            raise ValueError(f"token_ids must be non-negative integers for {key!r}")
    if observed != expected:
        missing = sorted(expected - observed)
        raise ValueError(f"prompt manifest is missing cases: {missing!r}")
    return cases


def read_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("prompt manifest root must be an object")
    return data, prompt_manifest_cases(data)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_prompts(args: argparse.Namespace) -> None:
    """Generate prompts through the checked-out upstream tokenizer and recipe."""

    repo = args.exl3_repo.resolve()
    model_dir = args.model_dir.resolve()
    if not repo.is_dir():
        raise SystemExit(f"EXL3 checkout does not exist: {repo}")
    if not model_dir.is_dir():
        raise SystemExit(f"model directory does not exist: {model_dir}")

    # Import only in this mode.  rand_prompt is imported from the read-only
    # upstream helper; its implementation is not copied into this runner.
    sys.path.insert(0, str(repo))
    try:
        torch = importlib.import_module("torch")
        exllamav3 = importlib.import_module("exllamav3")
        upstream_bench = importlib.import_module("rocm_tools.bench_model")
        Config = exllamav3.Config
        Tokenizer = exllamav3.Tokenizer
        rand_prompt = upstream_bench.rand_prompt

        config = Config.from_directory(str(model_dir))
        tokenizer = Tokenizer.from_config(config)
        rng = torch.Generator(device="cpu").manual_seed(SEED)
        cases: list[dict[str, Any]] = []
        for row in ROWS:
            for repeat in range(WARMUPS + MEASURED):
                ids = rand_prompt(tokenizer, row["input_tokens"], rng)
                ids_list = [int(token) for token in ids.reshape(-1).tolist()]
                if len(ids_list) != row["input_tokens"]:
                    raise RuntimeError(f"upstream rand_prompt returned wrong length for {row['case_id']}")
                cases.append(
                    {
                        **row,
                        "repeat": repeat,
                        "warmup": repeat < WARMUPS,
                        "token_ids": ids_list,
                        "token_ids_sha256": sha256_json(ids_list),
                    }
                )
        manifest = {
            "schema_version": PROMPT_SCHEMA_VERSION,
            "model": "Qwen/Qwen3.5-9B",
            "model_dir": str(model_dir),
            "tokenizer_actual_vocab_size": int(tokenizer.actual_vocab_size),
            "seed": SEED,
            "prompt_source": {
                "function": "rand_prompt",
                "module": "rocm_tools.bench_model",
                "path": str(repo / "rocm_tools" / "bench_model.py"),
                "semantics": "torch.randint(vocab*0.05, vocab*0.95) with CPU Generator",
            },
            "conditions": {
                "prefill_tokens": [128, 512, 2048],
                "decode_prompt_tokens": 128,
                "decode_output_tokens_nominal": 64,
                "warmups_per_row": WARMUPS,
                "measured_per_row": MEASURED,
            },
            "cases": cases,
        }
        write_json(args.prompts, manifest)
        print(json.dumps({
            "path": str(args.prompts),
            "sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
            "cases": len(cases),
            "tokenizer_vocab": int(tokenizer.actual_vocab_size),
        }, sort_keys=True), flush=True)
    finally:
        # Do not leave an imported sibling module ahead of the repository in a
        # long-lived embedding process.  A normal CLI exits immediately anyway.
        if sys.path and sys.path[0] == str(repo):
            sys.path.pop(0)


def finite_positive(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{name} is not numeric: {value!r}") from exc
    assert math.isfinite(result) and result > 0.0, f"{name} must be finite and positive: {result!r}"
    return result


def token_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):
            return token_count(value[0])
        return len(value)
    if hasattr(value, "shape"):
        shape = tuple(int(x) for x in value.shape)
        return shape[-1]
    if hasattr(value, "numel"):
        return int(value.numel())
    raise AssertionError(f"cannot determine token count from {type(value)!r}")


def summary_rows(records: list[dict[str, Any]], engine: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for expected in ROWS:
        selected = [r for r in records if r["case_id"] == expected["case_id"]]
        assert len(selected) == WARMUPS + MEASURED, (expected["case_id"], selected)
        assert sorted(int(r["repeat"]) for r in selected) == list(range(WARMUPS + MEASURED))
        measured = [r for r in selected if not r["warmup"]]
        assert len(measured) == MEASURED
        samples = [finite_positive("tokens_per_second", r["tokens_per_second"]) for r in measured]
        median = statistics.median(samples)
        spread = (max(samples) - min(samples)) / median
        assert math.isfinite(median) and median > 0.0
        assert math.isfinite(spread) and spread >= 0.0
        row = {
            "case_id": expected["case_id"],
            "phase": expected["phase"],
            "input_tokens": expected["input_tokens"],
            "nominal_output_tokens": expected["output_tokens"],
            "warmups": WARMUPS,
            "measured": MEASURED,
            "samples_tokens_per_second": samples,
            "median_tokens_per_second": median,
            "spread_percent": spread * 100.0,
            "engine": engine,
        }
        if engine == "exl3":
            row["timing_definition"] = (
                "prefill headline numerator prompt_tokens; processed_tokens="
                "prompt_tokens-cached_tokens-1; decode numerator actual new_tokens "
                "(63 for max_new_tokens=64)"
            )
        else:
            row["timing_definition"] = (
                "prefill numerator prompt_n; decode numerator predicted_n-1 "
                "(63 timed steps for n_predict=64)"
            )
        rows.append(row)
    return rows


def write_summary(
    args: argparse.Namespace,
    engine: str,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "model": "Qwen/Qwen3.5-9B",
        "prompt_manifest_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "seed": SEED,
        "conditions": manifest.get("conditions", {}),
        "warmups_per_row": WARMUPS,
        "measured_per_row": MEASURED,
        "native_timing_boundaries": {
            "exl3": {
                "prefill": "time_prefill excludes the final prompt token; headline numerator remains prompt_tokens and processed_tokens is prompt_tokens-cached_tokens-1",
                "decode": "Job max_new_tokens=64 produces 63 actual tokens; time_generate spans 63 forwards from the final prompt-token forward",
            },
            "llama": {
                "prefill": "prompt_ms includes all prompt_n tokens",
                "decode": "n_predict=64 produces 64 tokens; predicted_ms is reported as 63 timed decode steps via predicted_n-1",
            },
        },
        "rows": summary_rows(records, engine),
        "records": records,
    }
    write_json(output_dir / f"summary-{engine}.json", summary)
    print(json.dumps(summary["rows"], indent=2, sort_keys=True), flush=True)


def exl3_job_result(generator: Any, job: Any) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    # Keep shallow event references while the job is timed.  In particular, do
    # not call jsonable() or Tensor.tolist() here: serialisation can add host
    # work to a short GPU measurement.  Conversion happens after wall is read.
    events: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    started = time.perf_counter()
    generator.enqueue(job)
    while generator.num_remaining_jobs():
        for event in generator.iterate():
            events.append(dict(event))
            if event.get("stage") == "streaming" and event.get("eos"):
                final = event
    wall = time.perf_counter() - started
    if final is None:
        raise AssertionError("EXL3 job completed without an EOS event")
    return events, final, wall


def run_exl3(args: argparse.Namespace) -> None:
    manifest, cases = read_manifest(args.prompts)
    repo = args.exl3_repo.resolve()
    model_dir = args.model_dir.resolve()
    if not repo.is_dir():
        raise SystemExit(f"EXL3 checkout does not exist: {repo}")
    if not model_dir.is_dir():
        raise SystemExit(f"model directory does not exist: {model_dir}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo))
    model = None
    generator = None
    loaded = False
    try:
        torch = importlib.import_module("torch")
        exllamav3 = importlib.import_module("exllamav3")
        Config = exllamav3.Config
        Model = exllamav3.Model
        Cache = exllamav3.Cache
        Tokenizer = exllamav3.Tokenizer
        Generator = exllamav3.Generator
        Job = exllamav3.Job
        DefaultSampler = importlib.import_module("exllamav3.generator.sampler").DefaultSampler

        assert torch.version.hip and torch.cuda.is_available(), "EXL3 benchmark requires a HIP GPU"
        assert torch.cuda.device_count() == 1, "EXL3 benchmark requires exactly one visible GPU"
        properties = torch.cuda.get_device_properties(0)
        gpu_arch = getattr(properties, "gcnArchName", "")
        assert gpu_arch.split(":")[0] == "gfx1201", properties

        config = Config.from_directory(str(model_dir))
        model = Model.from_config(config)
        tokenizer = Tokenizer.from_config(config)
        assert int(tokenizer.actual_vocab_size) == manifest["tokenizer_actual_vocab_size"], (
            "tokenizer.actual_vocab_size changed between prompt generation and EXL3 runtime",
            tokenizer.actual_vocab_size,
            manifest["tokenizer_actual_vocab_size"],
        )
        cache = Cache(model, max_num_tokens=args.cache_size)
        model.load(device="cuda:0", progressbar=False)
        loaded = True
        generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

        records: list[dict[str, Any]] = []
        for case in cases:
            input_ids = torch.tensor(case["token_ids"], dtype=torch.long).reshape(1, -1)
            job = Job(
                input_ids=input_ids,
                max_new_tokens=int(case["output_tokens"]),
                sampler=DefaultSampler(),
                seed=SEED,
                # No stop conditions: the benchmark must exercise the requested
                # native budget, while llama uses ignore_eos=true for the same reason.
                stop_conditions=None,
                identifier=f"{case['case_id']}-{case['repeat']}",
            )
            events, final, wall = exl3_job_result(generator, job)
            expected_output = 1 if case["phase"] == "prefill" else 63
            streamed_output = sum(
                token_count(event.get("token_ids"))
                for event in events
                if event.get("stage") == "streaming"
            )
            assert streamed_output == int(final.get("new_tokens", 0)), final
            assert final.get("prompt_tokens") == case["input_tokens"], final
            assert final.get("cached_tokens", 0) == 0, final
            assert final.get("new_tokens") == expected_output, final
            assert final.get("eos_reason") == "max_new_tokens", final
            if case["phase"] == "prefill":
                elapsed = finite_positive("EXL3 time_prefill", final.get("time_prefill"))
                headline_tokens = int(final["prompt_tokens"])
                processed_tokens = headline_tokens - int(final.get("cached_tokens", 0)) - 1
                timed_tokens = headline_tokens
            else:
                elapsed = finite_positive("EXL3 time_generate", final.get("time_generate"))
                headline_tokens = int(final["new_tokens"])
                processed_tokens = headline_tokens
                timed_tokens = int(final["new_tokens"])
            assert headline_tokens > 0 and processed_tokens > 0
            assert timed_tokens > 0
            speed = finite_positive("EXL3 tokens_per_second", timed_tokens / elapsed)
            raw_name = f"exl3-{case['case_id']}-r{case['repeat']}.json"
            raw = {
                "engine": "exl3",
                "case": {k: v for k, v in case.items() if k != "token_ids"},
                "request": {
                    "max_new_tokens": int(case["output_tokens"]),
                    "seed": SEED,
                    "sampler": "DefaultSampler(min_p=0.08, temperature=0.8)",
                    "stop_conditions": None,
                },
                "events": events,
                "result": final,
                "streamed_output_tokens": streamed_output,
                "wall_seconds": wall,
            }
            write_json(output_dir / raw_name, raw)
            records.append(
                {
                    **{k: v for k, v in case.items() if k != "token_ids"},
                    "engine": "exl3",
                    "visible_gpu_count": 1,
                    "gpu_arch": gpu_arch,
                    "tokenizer_actual_vocab_size": int(tokenizer.actual_vocab_size),
                    "actual_output_tokens": expected_output,
                    "streamed_output_tokens": streamed_output,
                    "headline_tokens": headline_tokens,
                    "processed_tokens": processed_tokens,
                    "timed_tokens": timed_tokens,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": speed,
                    "event_file": raw_name,
                    "event_sha256": hashlib.sha256((output_dir / raw_name).read_bytes()).hexdigest(),
                    "protocol": "PASS",
                }
            )
            print(json.dumps(records[-1], sort_keys=True), flush=True)
        write_summary(args, "exl3", manifest, records)
    finally:
        if loaded and model is not None:
            # Model.unload() releases module and cache tensors through the native
            # public API.  Keep this in finally so failures also reclaim VRAM.
            model.unload()
        if generator is not None:
            del generator
        if model is not None:
            del model
        try:
            import gc

            gc.collect()
            torch.cuda.empty_cache()
        except (NameError, AttributeError, RuntimeError):
            pass
        if sys.path and sys.path[0] == str(repo):
            sys.path.pop(0)


def llama_request(url: str, body: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    endpoint = url.rstrip("/") + "/completion"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return result, time.perf_counter() - started


def run_llama(args: argparse.Namespace) -> None:
    manifest, cases = read_manifest(args.prompts)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for case in cases:
        body = {
            "prompt": case["token_ids"],
            "n_predict": int(case["output_tokens"]),
            "cache_prompt": False,
            "ignore_eos": True,
            "stream": False,
            "temperature": 0.8,
            "min_p": 0.08,
            "top_k": 0,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "repeat_last_n": 0,
            "samplers": ["min_p", "temperature"],
            "seed": SEED,
            "return_tokens": True,
        }
        response, wall = llama_request(args.url, body, args.timeout)
        timings = response.get("timings")
        if not isinstance(timings, dict):
            raise AssertionError(f"llama response has no timings: {response!r}")
        expected_output = int(case["output_tokens"])
        assert timings.get("cache_n") == 0, timings
        assert timings.get("prompt_n") == case["input_tokens"], timings
        assert response.get("tokens_predicted") == expected_output, response
        assert timings.get("predicted_n") == expected_output, timings
        output_ids = response.get("tokens")
        assert isinstance(output_ids, list) and len(output_ids) == expected_output, response
        assert response.get("stop_type") == "limit", response
        if case["phase"] == "prefill":
            elapsed_ms = finite_positive("llama prompt_ms", timings.get("prompt_ms"))
            timed_tokens = int(timings["prompt_n"])
        else:
            elapsed_ms = finite_positive("llama predicted_ms", timings.get("predicted_ms"))
            timed_tokens = int(timings["predicted_n"]) - 1
        assert timed_tokens > 0
        speed = finite_positive("llama tokens_per_second", 1000.0 * timed_tokens / elapsed_ms)
        raw_name = f"llama-{case['case_id']}-r{case['repeat']}.json"
        raw = {
            "engine": "llama",
            "case": {k: v for k, v in case.items() if k != "token_ids"},
            "request": body,
            "response": response,
            "http_wall_seconds": wall,
        }
        write_json(output_dir / raw_name, raw)
        records.append(
            {
                **{k: v for k, v in case.items() if k != "token_ids"},
                "engine": "llama",
                "actual_output_tokens": expected_output,
                "timed_tokens": timed_tokens,
                "elapsed_seconds": elapsed_ms / 1000.0,
                "tokens_per_second": speed,
                "response_file": raw_name,
                "response_sha256": hashlib.sha256((output_dir / raw_name).read_bytes()).hexdigest(),
                "protocol": "PASS",
            }
        )
        print(json.dumps(records[-1], sort_keys=True), flush=True)
    write_summary(args, "llama", manifest, records)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    mode = selected_mode(args, parser)
    require_mode_args(args, mode)
    if mode == "generate-prompts":
        generate_prompts(args)
    elif mode == "exl3":
        run_exl3(args)
    else:
        run_llama(args)


if __name__ == "__main__":
    main()
