#!/usr/bin/env python3
"""Run package-backed prefill component real-batch smoke cases."""

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


SCHEMA_VERSION = "ullm-package-prefill-component-workload-v0.1"
DEFAULT_BENCHMARK_SCRIPT = Path("tools/run-external-benchmark.py")
REQUIRED_HIP_KERNEL_ENVS = {
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL": "1",
    "ULLM_REQUIRE_HIP_ADD_KERNEL": "1",
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL": "1",
    "ULLM_REQUIRE_HIP_ROPE_KERNEL": "1",
    "ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL": "1",
    "ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL": "1",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL": "1",
}


@dataclass(frozen=True)
class WorkloadCase:
    case_id: str
    command: str
    component_args: list[str]
    prompt_tokens: int
    concurrent_requests: int
    context_length: int
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
    root = as_mapping(root, "workload manifest")
    schema = root.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise SystemExit(f"schema_version must be {SCHEMA_VERSION}, got {schema!r}")
    return root


def manifest_string(root: dict[str, Any], key: str, default: str | None = None) -> str:
    return as_string(root.get(key, default), key)


def manifest_positive_int(root: dict[str, Any], key: str, default: int) -> int:
    return positive_int(root.get(key, default), key)


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
    return completed.stdout.strip() or None


def merged_env(root: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    if bool(root.get("require_hip_kernels", False)):
        env.update(REQUIRED_HIP_KERNEL_ENVS)
    env.update(string_env(root.get("env"), "env"))
    return env


def load_cases(root: dict[str, Any], only_case: set[str], skip_warmup: bool) -> list[WorkloadCase]:
    raw_cases = root.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SystemExit("workload manifest must contain a non-empty cases list")
    default_warmup = non_negative_int(root.get("warmup_runs", 0), "warmup_runs")
    default_measured = positive_int(root.get("measured_runs", 1), "measured_runs")
    default_timeout = optional_float(root.get("timeout_seconds"), "timeout_seconds")
    cases: list[WorkloadCase] = []
    seen: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case = as_mapping(raw_case, f"cases[{index}]")
        case_id = optional_string(case.get("case_id"), f"cases[{index}].case_id")
        command = as_string(case.get("command"), f"cases[{index}].command")
        prompt_tokens = positive_int(case.get("prompt_tokens"), f"cases[{index}].prompt_tokens")
        if case_id is None:
            case_id = f"{command}-pp{prompt_tokens}"
        if case_id in seen:
            raise SystemExit(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        if only_case and case_id not in only_case:
            continue
        component_args = string_list(case.get("component_args"), f"{case_id}.component_args")
        if not component_args:
            raise SystemExit(f"{case_id}.component_args must not be empty")
        warmup_runs = non_negative_int(
            case.get("warmup_runs", default_warmup),
            f"{case_id}.warmup_runs",
        )
        if skip_warmup:
            warmup_runs = 0
        cases.append(
            WorkloadCase(
                case_id=case_id,
                command=command,
                component_args=component_args,
                prompt_tokens=prompt_tokens,
                concurrent_requests=positive_int(
                    case.get("concurrent_requests", 1),
                    f"{case_id}.concurrent_requests",
                ),
                context_length=positive_int(
                    case.get("context_length", prompt_tokens),
                    f"{case_id}.context_length",
                ),
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


def run_external_prefix(
    root: dict[str, Any],
    case: WorkloadCase,
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
        str(run.output_jsonl),
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
        manifest_string(root, "model_format", "ullm-package"),
        "--model-quantization",
        manifest_string(root, "model_quantization"),
        "--gpu-card",
        manifest_string(root, "gpu_card"),
        "--context-length",
        str(case.context_length),
        "--prompt-tokens",
        str(case.prompt_tokens),
        "--generated-tokens",
        "0",
        "--batch-size",
        str(case.concurrent_requests),
        "--concurrent-requests",
        str(case.concurrent_requests),
        "--kv-cache-dtype",
        manifest_string(root, "kv_cache_dtype", "f32"),
        "--parse",
        "ullm-component-prefill",
        "--result-json",
        str(run.run_dir / "raw.json"),
        "--note",
        f"workload-stage={run.stage}",
        "--note",
        f"workload-iteration={run.iteration}",
        "--note",
        "package-prefill-component-real-batch",
    ]
    optional_pairs = [
        ("engine_version", "--engine-version"),
        ("engine_commit", "--engine-commit"),
        ("model_source", "--model-source"),
        ("model_revision", "--model-revision"),
        ("sq_candidate", "--sq-candidate"),
        ("candidate_artifact", "--candidate-artifact"),
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
        manifest_string(root, "engine", "target/debug/ullm-engine"),
        case.command,
        manifest_string(root, "package_dir"),
        str(manifest_positive_int(root, "device_index", 2)),
        str(manifest_positive_int(root, "chunk_bytes", 1024 * 1024)),
        *case.component_args,
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
                placeholder = RunCommand(
                    stage=stage,
                    case=case,
                    iteration=iteration,
                    run_dir=run_dir,
                    output_jsonl=output_dir / output_name,
                    command=[],
                    env=env,
                )
                full_command = [
                    *run_external_prefix(root, case, placeholder),
                    "--",
                    *engine_command(root, case),
                ]
                commands.append(
                    RunCommand(
                        stage=stage,
                        case=case,
                        iteration=iteration,
                        run_dir=run_dir,
                        output_jsonl=output_dir / output_name,
                        command=full_command,
                        env=env,
                    )
                )
    return commands


def prepare_output(
    output_dir: Path,
    root: dict[str, Any],
    commands: list[RunCommand],
    overwrite: bool,
) -> None:
    if output_dir.exists():
        if not overwrite:
            raise SystemExit(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "workload.json").write_text(
        json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plan = [
        {
            "stage": run.stage,
            "case_id": run.case.case_id,
            "iteration": run.iteration,
            "run_dir": str(run.run_dir),
            "output_jsonl": str(run.output_jsonl),
            "command": run.command,
            "command_string": " ".join(shlex.quote(part) for part in run.command),
        }
        for run in commands
    ]
    (output_dir / "execution-plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def execute(commands: list[RunCommand], keep_going: bool, dry_run: bool) -> int:
    if dry_run:
        for run in commands:
            print(" ".join(shlex.quote(part) for part in run.command))
        return 0
    for run in commands:
        run.run_dir.mkdir(parents=True, exist_ok=True)
        print(" ".join(shlex.quote(part) for part in run.command), flush=True)
        completed = subprocess.run(run.command, env=run.env, check=False)
        if completed.returncode != 0 and not keep_going:
            return completed.returncode
    return 0


def main() -> int:
    args = parse_args()
    root = read_workload(args.workload_json)
    cases = load_cases(root, set(args.only_case), args.skip_warmup)
    commands = build_commands(root, args.output_dir, cases)
    prepare_output(args.output_dir, root, commands, args.overwrite)
    return execute(commands, args.keep_going, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
