#!/usr/bin/env python3
"""Run an uLLM package batch-throughput workload manifest."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ullm-batch-throughput-workload-v0.1"
DEFAULT_BENCHMARK_SCRIPT = Path("tools/run-external-benchmark.py")
REQUIRED_HIP_KERNEL_ENVS = {
    "ULLM_REQUIRE_HIP_AQ4_KERNEL": "1",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL": "1",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL": "1",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL": "1",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL": "1",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL": "1",
    "ULLM_REQUIRE_HIP_ADD_KERNEL": "1",
    "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL": "1",
    "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL": "1",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL": "1",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL": "1",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL": "1",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL": "1",
    "ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL": "1",
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL": "1",
    "ULLM_REQUIRE_HIP_ROPE_KERNEL": "1",
    "ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL": "1",
    "ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL": "1",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL": "1",
    "ULLM_REQUIRE_HIP_TOP1_KERNEL": "1",
}


@dataclass(frozen=True)
class WorkloadCase:
    case_id: str
    prompt_tokens: int
    generated_tokens: int
    concurrent_requests: int
    context_length: int
    prompt_token_ids_batch: str
    generated_tokens_batch: str
    warmup_runs: int
    measured_runs: int
    timeout_seconds: float | None
    notes: list[str]


@dataclass(frozen=True)
class RunCommand:
    stage: str
    case: WorkloadCase
    iteration: int
    run_dir: Path
    output_jsonl: Path
    command: list[str]
    env: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--benchmark-script", type=Path, default=DEFAULT_BENCHMARK_SCRIPT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--only-case", action="append", default=[])
    return parser.parse_args()


def as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be an object")
    return value


def as_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{label} must be a non-empty string")
    return value.strip()


def optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return as_string(value, label)


def positive_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be an integer") from exc
    if parsed <= 0:
        raise SystemExit(f"{label} must be positive")
    return parsed


def non_negative_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be an integer") from exc
    if parsed < 0:
        raise SystemExit(f"{label} must be non-negative")
    return parsed


def optional_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be numeric") from exc
    if parsed <= 0:
        raise SystemExit(f"{label} must be positive")
    return parsed


def string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SystemExit(f"{label} must be a list of strings")
    return value


def string_env(value: Any, label: str) -> dict[str, str]:
    if value is None:
        return {}
    raw = as_mapping(value, label)
    env: dict[str, str] = {}
    for key, item in raw.items():
        if not isinstance(key, str) or not key:
            raise SystemExit(f"{label} keys must be non-empty strings")
        if item is None:
            raise SystemExit(f"{label}.{key} must not be null")
        env[key] = str(item)
    return env


def read_workload(path: Path) -> dict[str, Any]:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read workload manifest {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse workload manifest {path}: {exc}") from exc
    return as_mapping(root, "workload manifest")


def manifest_string(root: dict[str, Any], key: str, default: str | None = None) -> str:
    value = root.get(key, default)
    return as_string(value, key)


def manifest_positive_int(root: dict[str, Any], key: str, default: int) -> int:
    return positive_int(root.get(key, default), key)


def manifest_non_negative_int(root: dict[str, Any], key: str, default: int) -> int:
    return non_negative_int(root.get(key, default), key)


def load_cases(root: dict[str, Any], only_case: set[str], skip_warmup: bool) -> list[WorkloadCase]:
    raw_cases = root.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SystemExit("workload manifest must contain a non-empty cases list")

    default_warmup = manifest_non_negative_int(root, "warmup_runs", 1)
    default_measured = manifest_positive_int(root, "measured_runs", 1)
    default_timeout = optional_float(root.get("timeout_seconds"), "timeout_seconds")
    cases: list[WorkloadCase] = []
    seen: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case = as_mapping(raw_case, f"cases[{index}]")
        prompt_tokens = positive_int(case.get("prompt_tokens"), f"cases[{index}].prompt_tokens")
        generated_tokens = positive_int(
            case.get("generated_tokens"),
            f"cases[{index}].generated_tokens",
        )
        concurrent_requests = positive_int(
            case.get("concurrent_requests", case.get("batch_size", 1)),
            f"cases[{index}].concurrent_requests",
        )
        case_id = optional_string(case.get("case_id"), f"cases[{index}].case_id")
        if case_id is None:
            case_id = f"pp{prompt_tokens}-tg{generated_tokens}-b{concurrent_requests}"
        if case_id in seen:
            raise SystemExit(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        if only_case and case_id not in only_case:
            continue

        context_length = positive_int(
            case.get("context_length", prompt_tokens + generated_tokens),
            f"{case_id}.context_length",
        )
        prompt_batch = optional_string(
            case.get("prompt_token_ids_batch"),
            f"{case_id}.prompt_token_ids_batch",
        )
        if prompt_batch is None:
            prompt_batch = f"len:{prompt_tokens}x{concurrent_requests}"
        generated_batch = optional_string(
            case.get("generated_tokens_batch"),
            f"{case_id}.generated_tokens_batch",
        )
        if generated_batch is None:
            generated_batch = str(generated_tokens)
        warmup_runs = non_negative_int(
            case.get("warmup_runs", default_warmup),
            f"{case_id}.warmup_runs",
        )
        if skip_warmup:
            warmup_runs = 0
        cases.append(
            WorkloadCase(
                case_id=case_id,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                concurrent_requests=concurrent_requests,
                context_length=context_length,
                prompt_token_ids_batch=prompt_batch,
                generated_tokens_batch=generated_batch,
                warmup_runs=warmup_runs,
                measured_runs=positive_int(
                    case.get("measured_runs", default_measured),
                    f"{case_id}.measured_runs",
                ),
                timeout_seconds=optional_float(
                    case.get("timeout_seconds"),
                    f"{case_id}.timeout_seconds",
                )
                or default_timeout,
                notes=string_list(case.get("notes"), f"{case_id}.notes"),
            )
        )
    if not cases:
        raise SystemExit("no cases selected")
    return cases


def current_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def merged_env(root: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    manifest_env = string_env(root.get("env"), "env")
    if bool(root.get("require_hip_kernels", False)):
        env.update(REQUIRED_HIP_KERNEL_ENVS)
    env.update(manifest_env)
    return env


def run_external_prefix(
    root: dict[str, Any],
    case: WorkloadCase,
    output_jsonl: Path,
    run: RunCommand,
) -> list[str]:
    command = [
        sys.executable,
        str(root.get("benchmark_script", DEFAULT_BENCHMARK_SCRIPT)),
        "--run-id",
        manifest_string(root, "run_id"),
        "--case-id",
        case.case_id,
        "--output-jsonl",
        str(output_jsonl),
        "--stdout-log",
        str(run.run_dir / "stdout.log"),
        "--stderr-log",
        str(run.run_dir / "stderr.log"),
        "--memory-log",
        str(run.run_dir / "memory.jsonl"),
        "--engine-name",
        manifest_string(root, "engine_name", "uLLM"),
        "--model-name",
        manifest_string(root, "model_name"),
        "--model-format",
        manifest_string(root, "model_format"),
        "--model-quantization",
        manifest_string(root, "model_quantization"),
        "--gpu-card",
        manifest_string(root, "gpu_card"),
        "--context-length",
        str(case.context_length),
        "--prompt-tokens",
        str(case.prompt_tokens),
        "--generated-tokens",
        str(case.generated_tokens),
        "--batch-size",
        str(case.concurrent_requests),
        "--concurrent-requests",
        str(case.concurrent_requests),
        "--kv-cache-dtype",
        manifest_string(root, "kv_cache_dtype", "f32"),
        "--parse",
        "ullm-package-batch-throughput",
        "--result-json",
        str(run.run_dir / "raw.json"),
        "--note",
        f"workload-stage={run.stage}",
        "--note",
        f"workload-iteration={run.iteration}",
    ]
    optional_pairs = [
        ("engine_version", "--engine-version"),
        ("engine_commit", "--engine-commit"),
        ("model_source", "--model-source"),
        ("model_revision", "--model-revision"),
        ("sq_candidate", "--sq-candidate"),
        ("candidate_artifact", "--candidate-artifact"),
        ("prefill_executor", "--prefill-executor"),
        ("resolved_prefill_executor", "--resolved-prefill-executor"),
    ]
    for key, flag in optional_pairs:
        value = optional_string(root.get(key), key)
        if value is not None:
            command.extend([flag, value])
    if "engine_commit" not in root:
        commit = current_git_commit()
        if commit:
            command.extend(["--engine-commit", commit])
    memory_sample_interval = optional_float(
        root.get("memory_sample_interval"),
        "memory_sample_interval",
    )
    if memory_sample_interval is not None:
        command.extend(["--memory-sample-interval", str(memory_sample_interval)])
    if case.timeout_seconds is not None:
        command.extend(["--timeout-seconds", str(case.timeout_seconds)])
    for note in [*string_list(root.get("notes"), "notes"), *case.notes]:
        command.extend(["--note", note])
    return command


def engine_command(root: dict[str, Any], case: WorkloadCase) -> list[str]:
    return [
        manifest_string(root, "engine", "target/release/ullm-engine"),
        "package-batch-throughput-bench",
        manifest_string(root, "package_dir"),
        str(manifest_positive_int(root, "device_index", 2)),
        str(manifest_positive_int(root, "chunk_bytes", 1024 * 1024)),
        manifest_string(root, "layers", "all"),
        case.prompt_token_ids_batch,
        case.generated_tokens_batch,
        str(manifest_positive_int(root, "top_k", 4)),
        str(manifest_positive_int(root, "lm_head_chunk_rows", 4096)),
        str(root.get("rotary_dim", 32)),
        str(root.get("rope_base", 10_000_000)),
        str(manifest_non_negative_int(root, "position_offset", 0)),
        manifest_string(root, "lm_head_mode", "gpu_resident_f32"),
        manifest_string(root, "stop_token_ids", "none"),
        manifest_string(root, "stop_token_sequences", "none"),
    ]


def build_commands(
    root: dict[str, Any],
    output_dir: Path,
    cases: list[WorkloadCase],
) -> list[RunCommand]:
    env = merged_env(root)
    commands: list[RunCommand] = []
    for case in cases:
        for stage, count, output_name in (
            ("warmup", case.warmup_runs, "warmup.jsonl"),
            ("measured", case.measured_runs, "results.jsonl"),
        ):
            for iteration in range(count):
                run_dir = output_dir / case.case_id / f"{stage}-{iteration}"
                output_jsonl = output_dir / output_name
                placeholder = RunCommand(
                    stage=stage,
                    case=case,
                    iteration=iteration,
                    run_dir=run_dir,
                    output_jsonl=output_jsonl,
                    command=[],
                    env=env,
                )
                full_command = [
                    *run_external_prefix(root, case, output_jsonl, placeholder),
                    "--",
                    *engine_command(root, case),
                ]
                commands.append(
                    RunCommand(
                        stage=stage,
                        case=case,
                        iteration=iteration,
                        run_dir=run_dir,
                        output_jsonl=output_jsonl,
                        command=full_command,
                        env=env,
                    )
                )
    return commands


def prepare_output(
    args: argparse.Namespace,
    root: dict[str, Any],
    commands: list[RunCommand],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("results.jsonl", "warmup.jsonl", "execution-plan.json"):
        path = args.output_dir / filename
        if path.exists():
            if not args.overwrite:
                raise SystemExit(f"output exists; pass --overwrite to replace: {path}")
            path.unlink()
    for command in commands:
        if command.run_dir.exists():
            if not args.overwrite:
                raise SystemExit(
                    f"run directory exists; pass --overwrite to replace: {command.run_dir}"
                )
            shutil.rmtree(command.run_dir)
        command.run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.workload_json, args.output_dir / "workload.json")
    write_execution_plan(args.output_dir / "execution-plan.json", root, commands)


def command_summary(command: RunCommand) -> dict[str, Any]:
    env_delta = {
        key: value
        for key, value in sorted(command.env.items())
        if key.startswith("ULLM_")
        or key
        in {
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "HSA_OVERRIDE_GFX_VERSION",
            "PYTORCH_HIP_ALLOC_CONF",
        }
    }
    return {
        "stage": command.stage,
        "case_id": command.case.case_id,
        "iteration": command.iteration,
        "run_dir": str(command.run_dir),
        "output_jsonl": str(command.output_jsonl),
        "command": command.command,
        "shell": shlex.join(command.command),
        "env": env_delta,
    }


def write_execution_plan(path: Path, root: dict[str, Any], commands: list[RunCommand]) -> None:
    plan = {
        "schema_version": SCHEMA_VERSION,
        "run_id": root.get("run_id"),
        "command_count": len(commands),
        "commands": [command_summary(command) for command in commands],
    }
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dry_run(root: dict[str, Any], commands: list[RunCommand]) -> int:
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": root.get("run_id"),
                "command_count": len(commands),
                "commands": [command_summary(command) for command in commands],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_commands(commands: list[RunCommand], keep_going: bool) -> int:
    failures = 0
    for command in commands:
        print(
            f"running {command.stage} {command.case.case_id} #{command.iteration}: "
            f"{command.run_dir}",
            file=sys.stderr,
        )
        completed = subprocess.run(command.command, check=False, env=command.env)
        if completed.returncode != 0:
            failures += 1
            print(
                f"failed {command.stage} {command.case.case_id} #{command.iteration}: "
                f"exit {completed.returncode}",
                file=sys.stderr,
            )
            if not keep_going:
                return completed.returncode
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    root = read_workload(args.workload_json)
    if root.get("schema_version") not in (None, SCHEMA_VERSION):
        raise SystemExit(f"unsupported workload schema_version: {root.get('schema_version')!r}")
    root["benchmark_script"] = str(args.benchmark_script)
    cases = load_cases(root, set(args.only_case), args.skip_warmup)
    commands = build_commands(root, args.output_dir, cases)
    if args.dry_run:
        return dry_run(root, commands)
    prepare_output(args, root, commands)
    return run_commands(commands, args.keep_going)


if __name__ == "__main__":
    raise SystemExit(main())
