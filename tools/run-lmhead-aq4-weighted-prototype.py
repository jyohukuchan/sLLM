#!/usr/bin/env python3
"""Build and guard an activation-weighted AQ4 lm_head prototype package."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "lmhead-aq4-weighted-prototype-run-v0.1"
DEFAULT_STOP_TOKEN_IDS = "248046,248044,248070,248071,248076,248056,248057,248053,248054"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", default="lmhead-aq4-weighted")
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--suite-json", type=Path, default=Path("benchmarks/prompts/pre-sq-runtime-prompt-suite-v0.3.json"))
    parser.add_argument("--prompt-file", type=Path, default=Path("benchmarks/calibration/qwen35-aq-smoke-prompts-v0.1.txt"))
    parser.add_argument("--activation-stats-dir", type=Path)
    parser.add_argument("--activation-device", default="cuda:0")
    parser.add_argument("--activation-dtype", default="bfloat16")
    parser.add_argument("--activation-max-samples", type=int, default=32)
    parser.add_argument("--activation-sequence-length", type=int, default=512)
    parser.add_argument(
        "--activation-repeat-to-length",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--module-pattern", default="lm_head$")
    parser.add_argument("--model-class", choices=("auto_model", "causal_lm"), default="causal_lm")
    parser.add_argument("--candidate", default="aq4_e4m3_g8_ts_flloyd16")
    parser.add_argument("--family", default="lm_head")
    parser.add_argument("--tensor-name", default="lm_head.weight")
    parser.add_argument("--max-elements-per-tensor", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=64)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--convert-chunk-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--scale-window", default="4")
    parser.add_argument("--tensor-scale-estimator", default="reservoir")
    parser.add_argument("--tensor-scale-reservoir-size", type=int, default=65536)
    parser.add_argument("--engine", type=Path, default=Path("target/release/ullm-engine"))
    parser.add_argument("--quant-bin", type=Path, default=Path("target/release/ullm-quant"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--suite-device",
        action="append",
        default=[],
        help="Prompt suite device as LABEL:INDEX. May be repeated.",
    )
    parser.add_argument("--generated-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--lm-head-chunk-rows", type=int, default=4096)
    parser.add_argument("--rotary-dim", type=int, default=32)
    parser.add_argument("--rope-base", type=float, default=10_000_000.0)
    parser.add_argument("--position-offset", type=int, default=0)
    parser.add_argument("--lm-head-mode", default="gpu_resident_f32")
    parser.add_argument("--stop-token-ids", default=DEFAULT_STOP_TOKEN_IDS)
    parser.add_argument("--logit-atol", type=float, default=1e-3)
    parser.add_argument(
        "--baseline-summary",
        action="append",
        default=[],
        help="Baseline prompt-suite summary as LABEL:PATH. LABEL should match a --suite-device label.",
    )
    parser.add_argument("--min-decode-tps-ratio-vs-baseline", type=float, default=1.0)
    parser.add_argument("--min-prefill-tps-ratio-vs-baseline", type=float, default=0.95)
    parser.add_argument("--max-hit-generation-limit-regression-vs-baseline", type=int, default=1)
    parser.add_argument("--max-output-warn-count", type=int, default=1)
    parser.add_argument("--max-output-not-evaluated-count", type=int, default=1)
    parser.add_argument("--max-hit-generation-limit-count", type=int, default=1)
    parser.add_argument("--max-low-unique-token-ratio-count", type=int, default=0)
    parser.add_argument("--max-prompt-echo-count", type=int, default=0)
    parser.add_argument("--skip-prompt-suite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def existing_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    candidate = repo_path(path)
    if candidate.exists():
        return candidate
    return path


def resolve_output(root: Path, child: str) -> Path:
    return root / child


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def parse_int(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def parse_inspect_output(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    metrics: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metrics[key.strip()] = parse_int(value.strip())
    return metrics


def bool_metric(metrics: dict[str, Any], key: str) -> bool:
    return metrics.get(key) is True


def int_metric(metrics: dict[str, Any], key: str) -> int | None:
    value = metrics.get(key)
    return value if isinstance(value, int) else None


def suite_gate(label: str, summary_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary = load_json_if_exists(summary_path)
    if summary is None:
        return {
            "label": label,
            "summary": str(summary_path),
            "evaluated": False,
            "passed": False,
            "reason": "summary_missing_or_invalid",
        }
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    checks = [
        ("verified_all", bool_metric(metrics, "verified_all"), True),
        (
            "output_warn_count",
            int_metric(metrics, "output_warn_count"),
            args.max_output_warn_count,
        ),
        (
            "output_not_evaluated_count",
            int_metric(metrics, "output_not_evaluated_count"),
            args.max_output_not_evaluated_count,
        ),
        (
            "hit_generation_limit_count",
            int_metric(metrics, "hit_generation_limit_count"),
            args.max_hit_generation_limit_count,
        ),
        (
            "low_unique_token_ratio_count",
            int_metric(metrics, "low_unique_token_ratio_count"),
            args.max_low_unique_token_ratio_count,
        ),
        ("prompt_echo_count", int_metric(metrics, "prompt_echo_count"), args.max_prompt_echo_count),
    ]
    failures = []
    for key, actual, expected in checks:
        if key == "verified_all":
            if actual is not expected:
                failures.append({"metric": key, "actual": actual, "expected": expected})
        elif actual is None or int(actual) > int(expected):
            failures.append({"metric": key, "actual": actual, "max": expected})
    return {
        "label": label,
        "summary": str(summary_path),
        "evaluated": True,
        "passed": not failures,
        "failures": failures,
        "metrics": metrics,
    }


def guard_gate(summary_path: Path) -> dict[str, Any]:
    summary = load_json_if_exists(summary_path)
    if summary is None:
        return {
            "summary": str(summary_path),
            "evaluated": False,
            "passed": False,
            "reason": "summary_missing_or_invalid",
        }
    checks = summary.get("checks", [])
    check_metrics = [
        {
            "name": check.get("name"),
            "passed": check.get("passed"),
            "metrics": check.get("metrics"),
        }
        for check in checks
        if isinstance(check, dict)
    ]
    return {
        "summary": str(summary_path),
        "evaluated": True,
        "passed": summary.get("passed") is True,
        "checks": check_metrics,
    }


def build_gates(
    inspect_output: Path,
    suite_summaries: list[dict[str, Any]],
    guard_summaries: list[dict[str, Any]],
    baseline_summaries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.dry_run:
        return {
            "passed": None,
            "full_quality_evaluated": False,
            "reason": "dry_run",
            "thresholds": {
                "max_output_warn_count": args.max_output_warn_count,
                "max_output_not_evaluated_count": args.max_output_not_evaluated_count,
                "max_hit_generation_limit_count": args.max_hit_generation_limit_count,
                "max_low_unique_token_ratio_count": args.max_low_unique_token_ratio_count,
                "max_prompt_echo_count": args.max_prompt_echo_count,
                "logit_atol": args.logit_atol,
                "min_decode_tps_ratio_vs_baseline": args.min_decode_tps_ratio_vs_baseline,
                "min_prefill_tps_ratio_vs_baseline": args.min_prefill_tps_ratio_vs_baseline,
                "max_hit_generation_limit_regression_vs_baseline": (
                    args.max_hit_generation_limit_regression_vs_baseline
                ),
            },
        }
    inspect_metrics = parse_inspect_output(inspect_output)
    if inspect_metrics is None:
        package_integrity = {
            "evaluated": False,
            "passed": False,
            "reason": "inspect_output_missing",
            "metrics": None,
        }
    else:
        missing = inspect_metrics.get("missing_referenced_files")
        package_integrity = {
            "evaluated": True,
            "passed": missing == 0,
            "metrics": inspect_metrics,
        }

    suite_gates = [
        suite_gate(str(item["label"]), Path(str(item["summary"])), args)
        for item in suite_summaries
    ]
    guard_gates = [guard_gate(Path(str(item["summary"]))) for item in guard_summaries]
    suite_summary_by_label = {
        str(item["label"]): Path(str(item["summary"]))
        for item in suite_summaries
    }
    baseline_gates = [
        baseline_gate(
            str(item["label"]),
            suite_summary_by_label.get(str(item["label"])),
            Path(str(item["summary"])),
            args,
        )
        for item in baseline_summaries
    ]
    evaluated_gates: list[dict[str, Any]] = [package_integrity]
    evaluated_gates.extend(suite_gates)
    evaluated_gates.extend(guard_gates)
    evaluated_gates.extend(baseline_gates)
    full_quality_evaluated = bool(suite_gates) and (len(suite_gates) < 2 or bool(guard_gates))
    return {
        "passed": all(gate.get("passed") is True for gate in evaluated_gates),
        "full_quality_evaluated": full_quality_evaluated,
        "package_integrity": package_integrity,
        "prompt_suites": suite_gates,
        "guards": guard_gates,
        "baseline_comparisons": baseline_gates,
        "thresholds": {
            "max_output_warn_count": args.max_output_warn_count,
            "max_output_not_evaluated_count": args.max_output_not_evaluated_count,
            "max_hit_generation_limit_count": args.max_hit_generation_limit_count,
            "max_low_unique_token_ratio_count": args.max_low_unique_token_ratio_count,
            "max_prompt_echo_count": args.max_prompt_echo_count,
            "logit_atol": args.logit_atol,
            "min_decode_tps_ratio_vs_baseline": args.min_decode_tps_ratio_vs_baseline,
            "min_prefill_tps_ratio_vs_baseline": args.min_prefill_tps_ratio_vs_baseline,
            "max_hit_generation_limit_regression_vs_baseline": (
                args.max_hit_generation_limit_regression_vs_baseline
            ),
        },
    }


def parse_suite_device(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise SystemExit(f"--suite-device must be LABEL:INDEX, got {value!r}")
    label, raw_index = value.split(":", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"--suite-device label must not be empty: {value!r}")
    try:
        index = int(raw_index)
    except ValueError as exc:
        raise SystemExit(f"--suite-device index must be an integer: {value!r}") from exc
    if index < 0:
        raise SystemExit(f"--suite-device index must be non-negative: {value!r}")
    return label, index


def parse_labeled_path(value: str, option: str) -> tuple[str, Path]:
    if ":" not in value:
        raise SystemExit(f"{option} must be LABEL:PATH, got {value!r}")
    label, raw_path = value.split(":", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"{option} label must not be empty: {value!r}")
    if not raw_path:
        raise SystemExit(f"{option} path must not be empty: {value!r}")
    return label, Path(raw_path)


def float_metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def ratio_or_none(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline <= 0.0:
        return None
    return candidate / baseline


def baseline_gate(
    label: str,
    candidate_summary_path: Path | None,
    baseline_summary_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if candidate_summary_path is None:
        return {
            "label": label,
            "baseline_summary": str(baseline_summary_path),
            "evaluated": False,
            "passed": False,
            "reason": "candidate_summary_missing_for_label",
        }
    candidate_summary = load_json_if_exists(candidate_summary_path)
    baseline_summary = load_json_if_exists(baseline_summary_path)
    if candidate_summary is None or baseline_summary is None:
        return {
            "label": label,
            "candidate_summary": str(candidate_summary_path),
            "baseline_summary": str(baseline_summary_path),
            "evaluated": False,
            "passed": False,
            "reason": "summary_missing_or_invalid",
        }
    candidate_metrics = candidate_summary.get("metrics", {})
    baseline_metrics = baseline_summary.get("metrics", {})
    if not isinstance(candidate_metrics, dict):
        candidate_metrics = {}
    if not isinstance(baseline_metrics, dict):
        baseline_metrics = {}

    candidate_decode = float_metric(candidate_metrics, "decode_tps_mean")
    baseline_decode = float_metric(baseline_metrics, "decode_tps_mean")
    candidate_prefill = float_metric(candidate_metrics, "prefill_tps_mean")
    baseline_prefill = float_metric(baseline_metrics, "prefill_tps_mean")
    decode_ratio = ratio_or_none(candidate_decode, baseline_decode)
    prefill_ratio = ratio_or_none(candidate_prefill, baseline_prefill)

    failures = []
    if decode_ratio is None or decode_ratio < args.min_decode_tps_ratio_vs_baseline:
        failures.append(
            {
                "metric": "decode_tps_mean_ratio",
                "actual": decode_ratio,
                "min": args.min_decode_tps_ratio_vs_baseline,
            }
        )
    if prefill_ratio is None or prefill_ratio < args.min_prefill_tps_ratio_vs_baseline:
        failures.append(
            {
                "metric": "prefill_tps_mean_ratio",
                "actual": prefill_ratio,
                "min": args.min_prefill_tps_ratio_vs_baseline,
            }
        )

    comparisons = [
        ("output_ok_count", ">="),
        ("output_warn_count", "<="),
        ("output_not_evaluated_count", "<="),
        ("low_unique_token_ratio_count", "<="),
        ("prompt_echo_count", "<="),
    ]
    for key, relation in comparisons:
        candidate_value = int_metric(candidate_metrics, key)
        baseline_value = int_metric(baseline_metrics, key)
        if candidate_value is None or baseline_value is None:
            failures.append({"metric": key, "actual": candidate_value, "baseline": baseline_value})
        elif relation == ">=" and candidate_value < baseline_value:
            failures.append({"metric": key, "actual": candidate_value, "min": baseline_value})
        elif relation == "<=" and candidate_value > baseline_value:
            failures.append({"metric": key, "actual": candidate_value, "max": baseline_value})

    candidate_hit_limit = int_metric(candidate_metrics, "hit_generation_limit_count")
    baseline_hit_limit = int_metric(baseline_metrics, "hit_generation_limit_count")
    if candidate_hit_limit is None or baseline_hit_limit is None:
        failures.append(
            {
                "metric": "hit_generation_limit_count",
                "actual": candidate_hit_limit,
                "baseline": baseline_hit_limit,
            }
        )
    else:
        max_hit_limit = baseline_hit_limit + args.max_hit_generation_limit_regression_vs_baseline
        if candidate_hit_limit > max_hit_limit:
            failures.append(
                {
                    "metric": "hit_generation_limit_count",
                    "actual": candidate_hit_limit,
                    "max": max_hit_limit,
                    "baseline": baseline_hit_limit,
                }
            )

    return {
        "label": label,
        "candidate_summary": str(candidate_summary_path),
        "baseline_summary": str(baseline_summary_path),
        "evaluated": True,
        "passed": not failures,
        "failures": failures,
        "metrics": {
            "candidate_decode_tps_mean": candidate_decode,
            "baseline_decode_tps_mean": baseline_decode,
            "decode_tps_ratio": decode_ratio,
            "candidate_prefill_tps_mean": candidate_prefill,
            "baseline_prefill_tps_mean": baseline_prefill,
            "prefill_tps_ratio": prefill_ratio,
            "candidate_output_ok_count": int_metric(candidate_metrics, "output_ok_count"),
            "baseline_output_ok_count": int_metric(baseline_metrics, "output_ok_count"),
            "candidate_output_warn_count": int_metric(candidate_metrics, "output_warn_count"),
            "baseline_output_warn_count": int_metric(baseline_metrics, "output_warn_count"),
            "candidate_hit_generation_limit_count": candidate_hit_limit,
            "baseline_hit_generation_limit_count": baseline_hit_limit,
        },
    }


def run_command(
    name: str,
    command: list[str],
    outputs: list[Path],
    args: argparse.Namespace,
    steps: list[dict[str, Any]],
) -> None:
    if args.skip_existing and outputs and all(path.exists() for path in outputs):
        steps.append(
            {
                "name": name,
                "skipped": True,
                "reason": "outputs_exist",
                "outputs": [str(path) for path in outputs],
                "command": command,
            }
        )
        print(f"skip {name}: outputs exist", file=sys.stderr)
        return

    rendered = shlex.join(command)
    steps.append(
        {
            "name": name,
            "skipped": bool(args.dry_run),
            "reason": "dry_run" if args.dry_run else None,
            "outputs": [str(path) for path in outputs],
            "command": command,
        }
    )
    print(f"run {name}: {rendered}", file=sys.stderr)
    if args.dry_run:
        return
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with code {result.returncode}: {rendered}")


def main() -> int:
    args = parse_args()
    if args.activation_max_samples < 1:
        raise SystemExit("--activation-max-samples must be >= 1")
    if args.activation_sequence_length < 1:
        raise SystemExit("--activation-sequence-length must be >= 1")
    if args.skip_prompt_suite and args.suite_device:
        raise SystemExit("--skip-prompt-suite cannot be combined with --suite-device")
    if args.suite_device and args.tokenizer_dir is None:
        raise SystemExit("--tokenizer-dir is required when --suite-device is used")

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    prompt_file = existing_path(args.prompt_file)
    suite_json = existing_path(args.suite_json)
    engine = existing_path(args.engine)
    quant_bin = existing_path(args.quant_bin)
    activation_dir = args.activation_stats_dir or resolve_output(root, "activation-stats")
    codebook_json = resolve_output(root, "lmhead-weighted-codebook.json")
    one_tensor_package = resolve_output(root, "lmhead-one-tensor.ullm.d")
    convert_summary = resolve_output(root, "lmhead-one-tensor-convert-summary.json")
    overlay_package = resolve_output(root, "package.ullm.d")
    overlay_summary = resolve_output(root, "overlay-summary.json")
    inspect_output = resolve_output(root, "inspect-package.txt")
    summary_json = args.summary_json or resolve_output(root, "summary.json")

    tools_dir = repo_root() / "tools"
    collect_script = tools_dir / "collect-activation-stats.py"
    codebook_script = tools_dir / "export-aq-family-codebooks.py"
    overlay_script = tools_dir / "overlay-ullm-prototype-package.py"
    suite_script = tools_dir / "run-package-token-prompt-suite.py"
    guard_script = tools_dir / "run-package-prompt-guard-bundle.py"

    steps: list[dict[str, Any]] = []
    activation_outputs = [
        activation_dir / "activation_second_moments.safetensors",
        activation_dir / "metadata.json",
    ]
    collect_command = [
        args.python_bin,
        str(collect_script),
        "--model-dir",
        str(args.model_dir),
        "--output-dir",
        str(activation_dir),
        "--prompt-file",
        str(prompt_file),
        "--max-samples",
        str(args.activation_max_samples),
        "--sequence-length",
        str(args.activation_sequence_length),
        "--module-pattern",
        args.module_pattern,
        "--model-class",
        args.model_class,
        "--dtype",
        args.activation_dtype,
        "--device",
        args.activation_device,
        "--run-id",
        args.run_id,
        "--note",
        "lm_head weighted aq4 prototype activation stats",
    ]
    if args.activation_repeat_to_length:
        collect_command.append("--repeat-to-length")
    for note in args.note:
        collect_command.extend(["--note", note])
    run_command("collect_activation_stats", collect_command, activation_outputs, args, steps)

    codebook_command = [
        args.python_bin,
        str(codebook_script),
        "--model-dir",
        str(args.model_dir),
        "--plan-json",
        str(args.plan_json),
        "--activation-stats",
        str(activation_dir),
        "--weighted-codebook",
        "--candidate",
        args.candidate,
        "--family",
        args.family,
        "--max-tensors",
        "1",
        "--max-tensors-per-family",
        "1",
        "--max-elements-per-tensor",
        str(args.max_elements_per_tensor),
        "--seed",
        str(args.seed),
        "--torch-threads",
        str(args.torch_threads),
        "--torch-interop-threads",
        str(args.torch_interop_threads),
        "--output",
        str(codebook_json),
        "--note",
        "lm_head weighted aq4 prototype codebook",
    ]
    for note in args.note:
        codebook_command.extend(["--note", note])
    run_command("export_weighted_codebook", codebook_command, [codebook_json], args, steps)

    convert_command = [
        str(quant_bin),
        "--convert-plan-json",
        str(args.plan_json),
        "--codebook-json",
        str(codebook_json),
        "--convert-package-output-dir",
        str(one_tensor_package),
        "--convert-package-summary-output",
        str(convert_summary),
        "--convert-family",
        args.family,
        "--convert-max-tensors",
        "1",
        "--convert-per-family",
        "1",
        "--chunk-bytes",
        str(args.convert_chunk_bytes),
        "--scale-window",
        str(args.scale_window),
        "--tensor-scale-estimator",
        args.tensor_scale_estimator,
        "--tensor-scale-reservoir-size",
        str(args.tensor_scale_reservoir_size),
        "--convert-verify",
        "--dry-run",
    ]
    if args.overwrite:
        convert_command.append("--convert-overwrite")
    run_command(
        "convert_one_tensor_package",
        convert_command,
        [one_tensor_package / "manifest.json", convert_summary],
        args,
        steps,
    )

    overlay_command = [
        args.python_bin,
        str(overlay_script),
        "--base-package",
        str(args.base_package),
        "--override-package",
        str(one_tensor_package),
        "--replace-tensor",
        args.tensor_name,
        "--output-package",
        str(overlay_package),
        "--summary-json",
        str(overlay_summary),
    ]
    if args.overwrite:
        overlay_command.append("--overwrite")
    run_command(
        "overlay_full_package",
        overlay_command,
        [overlay_package / "manifest.json", overlay_summary],
        args,
        steps,
    )

    inspect_command = [str(engine), "inspect-package", str(overlay_package)]
    if args.skip_existing and inspect_output.exists():
        steps.append(
            {
                "name": "inspect_package",
                "skipped": True,
                "reason": "outputs_exist",
                "outputs": [str(inspect_output)],
                "command": inspect_command,
            }
        )
        print("skip inspect_package: outputs exist", file=sys.stderr)
    else:
        print(f"run inspect_package: {shlex.join(inspect_command)}", file=sys.stderr)
        steps.append(
            {
                "name": "inspect_package",
                "skipped": bool(args.dry_run),
                "reason": "dry_run" if args.dry_run else None,
                "outputs": [str(inspect_output)],
                "command": inspect_command,
            }
        )
        if not args.dry_run:
            result = subprocess.run(
                inspect_command,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            inspect_output.write_text(result.stdout, encoding="utf-8")
            if result.stdout.strip():
                print(result.stdout.strip(), file=sys.stderr)
            if result.returncode != 0:
                raise SystemExit(f"inspect_package failed with code {result.returncode}")

    suite_summaries: list[dict[str, Any]] = []
    if not args.skip_prompt_suite:
        for label, device_index in [parse_suite_device(value) for value in args.suite_device]:
            suite_dir = resolve_output(root, f"prompt-suite-{label}")
            suite_summary = suite_dir / "summary.json"
            suite_command = [
                args.python_bin,
                str(suite_script),
                "--suite-json",
                str(suite_json),
                "--output-dir",
                str(suite_dir),
                "--package-dir",
                str(overlay_package),
                "--tokenizer-dir",
                str(args.tokenizer_dir),
                "--engine",
                str(engine),
                "--device-index",
                str(device_index),
                "--chunk-bytes",
                "1048576",
                "--layers",
                "all",
                "--generated-tokens",
                str(args.generated_tokens),
                "--top-k",
                str(args.top_k),
                "--lm-head-chunk-rows",
                str(args.lm_head_chunk_rows),
                "--rotary-dim",
                str(args.rotary_dim),
                "--rope-base",
                str(args.rope_base),
                "--position-offset",
                str(args.position_offset),
                "--lm-head-mode",
                args.lm_head_mode,
                "--stop-token-ids",
                args.stop_token_ids,
                "--stop-on-eos",
                "--stop-on-special-tokens",
                "--summary-json",
                "summary.json",
                "--summary-md",
                "summary.md",
            ]
            if args.overwrite:
                suite_command.append("--overwrite")
            run_command(
                f"prompt_suite_{label}",
                suite_command,
                [suite_summary],
                args,
                steps,
            )
            suite_summaries.append(
                {
                    "label": label,
                    "device_index": device_index,
                    "summary": str(suite_summary),
                }
            )

    guard_summaries: list[dict[str, Any]] = []
    if len(suite_summaries) >= 2:
        reference = suite_summaries[0]
        for candidate in suite_summaries[1:]:
            guard_dir = resolve_output(root, f"guard-{reference['label']}-{candidate['label']}")
            guard_summary = guard_dir / "guard-bundle-summary.json"
            guard_command = [
                args.python_bin,
                str(guard_script),
                "--reference-summary",
                str(reference["summary"]),
                "--candidate-summary",
                str(candidate["summary"]),
                "--reference-label",
                str(reference["label"]),
                "--candidate-label",
                str(candidate["label"]),
                "--output-dir",
                str(guard_dir),
                "--logit-atol",
                str(args.logit_atol),
                "--summary-json",
                "guard-bundle-summary.json",
                "--summary-md",
                "guard-bundle-summary.md",
            ]
            run_command(
                f"guard_{reference['label']}_{candidate['label']}",
                guard_command,
                [guard_summary],
                args,
                steps,
            )
            guard_summaries.append(
                {
                    "reference_label": reference["label"],
                    "candidate_label": candidate["label"],
                    "summary": str(guard_summary),
                }
            )

    baseline_summaries = [
        {"label": label, "summary": str(existing_path(path))}
        for label, path in [parse_labeled_path(value, "--baseline-summary") for value in args.baseline_summary]
    ]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "model_dir": str(args.model_dir),
        "base_package": str(args.base_package),
        "plan_json": str(args.plan_json),
        "candidate": args.candidate,
        "family": args.family,
        "tensor_name": args.tensor_name,
        "artifacts": {
            "activation_stats_dir": str(activation_dir),
            "codebook_json": str(codebook_json),
            "one_tensor_package": str(one_tensor_package),
            "convert_summary": str(convert_summary),
            "overlay_package": str(overlay_package),
            "overlay_summary": str(overlay_summary),
            "inspect_output": str(inspect_output),
        },
        "suite_summaries": suite_summaries,
        "guard_summaries": guard_summaries,
        "baseline_summaries": baseline_summaries,
        "gates": build_gates(inspect_output, suite_summaries, guard_summaries, baseline_summaries, args),
        "dry_run": bool(args.dry_run),
        "skip_existing": bool(args.skip_existing),
        "steps": steps,
        "notes": args.note,
    }
    write_json(summary_json, payload)
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
    print(f"wrote {summary_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
