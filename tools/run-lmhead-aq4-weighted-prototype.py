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
