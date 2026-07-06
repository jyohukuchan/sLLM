#!/usr/bin/env python3
"""Run a small prompt suite through run-package-token-prompt-bench.py."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BENCH_SCRIPT = Path("tools/run-package-token-prompt-bench.py")
CONTROL_MARKERS = ("<|", "<think>", "</think>", "user\nassistant", "assistant\n", "user\n")
TERMINAL_TEXT_ENDINGS = (".", "!", "?", "。", "！", "？", "```")


@dataclass(frozen=True)
class PromptCase:
    case_id: str
    category: str
    prompt: str
    generated_tokens: int | None
    target_prompt_tokens: int | None
    apply_chat_template: bool
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a JSON prompt suite through the text-prompt package bench wrapper "
            "and write per-prompt reports plus summary JSON/Markdown."
        )
    )
    parser.add_argument("--suite-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--bench-script", type=Path, default=DEFAULT_BENCH_SCRIPT)
    parser.add_argument("--engine", default="target/release/ullm-engine")
    parser.add_argument("--device-index", type=int, default=2)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--generated-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--lm-head-chunk-rows", type=int, default=8192)
    parser.add_argument("--rotary-dim", type=int, default=64)
    parser.add_argument("--rope-base", type=float, default=10_000_000.0)
    parser.add_argument("--position-offset", type=int, default=0)
    parser.add_argument(
        "--lm-head-mode",
        default="gpu_resident_f32",
        choices=("cpu_chunked", "gpu_resident_f32"),
    )
    parser.add_argument("--stop-token-ids")
    parser.add_argument("--stop-on-eos", action="store_true")
    parser.add_argument("--stop-on-special-tokens", action="store_true")
    parser.add_argument(
        "--require-hip-kernels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-json", default="summary.json")
    parser.add_argument("--summary-md", default="summary.md")
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Read existing per-case reports and rewrite summaries without rerunning inference",
    )
    return parser.parse_args()


def as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be an object")
    return value


def load_suite(path: Path) -> tuple[dict[str, Any], list[PromptCase]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read suite {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse suite {path}: {exc}") from exc

    if isinstance(raw, list):
        metadata: dict[str, Any] = {}
        raw_cases = raw
    else:
        root = as_mapping(raw, "suite")
        metadata = as_mapping(root.get("metadata", {}), "suite.metadata")
        raw_cases = root.get("prompts")
        if not isinstance(raw_cases, list):
            raise SystemExit("suite.prompts must be a list")

    cases: list[PromptCase] = []
    seen: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case = as_mapping(raw_case, f"prompt case {index}")
        case_id = str(case.get("id", "")).strip()
        if not case_id:
            raise SystemExit(f"prompt case {index} has no id")
        if case_id in seen:
            raise SystemExit(f"duplicate prompt id: {case_id}")
        seen.add(case_id)
        prompt = case.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise SystemExit(f"prompt case {case_id} has no prompt text")
        category = str(case.get("category", "general")).strip() or "general"
        generated_tokens = optional_positive_int(case.get("generated_tokens"), f"{case_id}.generated_tokens")
        target_prompt_tokens = optional_positive_int(
            case.get("target_prompt_tokens"),
            f"{case_id}.target_prompt_tokens",
        )
        notes = case.get("notes", [])
        if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
            raise SystemExit(f"{case_id}.notes must be a list of strings")
        cases.append(
            PromptCase(
                case_id=case_id,
                category=category,
                prompt=prompt,
                generated_tokens=generated_tokens,
                target_prompt_tokens=target_prompt_tokens,
                apply_chat_template=bool(case.get("apply_chat_template", False)),
                notes=notes,
            )
        )
    if not cases:
        raise SystemExit("suite has no prompt cases")
    return metadata, cases


def optional_positive_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be an integer") from exc
    if parsed <= 0:
        raise SystemExit(f"{label} must be positive")
    return parsed


def safe_filename(case_id: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in case_id)


def run_case(args: argparse.Namespace, case: PromptCase, output_json: Path) -> dict[str, Any]:
    if output_json.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite to replace: {output_json}")
    command = [
        sys.executable,
        str(args.bench_script),
        "--package-dir",
        args.package_dir,
        "--tokenizer-dir",
        args.tokenizer_dir,
        "--engine",
        args.engine,
        "--prompt",
        case.prompt,
        "--output-json",
        str(output_json),
        "--device-index",
        str(args.device_index),
        "--chunk-bytes",
        str(args.chunk_bytes),
        "--layers",
        args.layers,
        "--generated-tokens",
        str(case.generated_tokens or args.generated_tokens),
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
    ]
    if case.target_prompt_tokens is not None:
        command.extend(["--target-prompt-tokens", str(case.target_prompt_tokens)])
    if case.apply_chat_template:
        command.append("--apply-chat-template")
    if args.stop_token_ids:
        command.extend(["--stop-token-ids", args.stop_token_ids])
    if args.stop_on_eos:
        command.append("--stop-on-eos")
    if args.stop_on_special_tokens:
        command.append("--stop-on-special-tokens")
    command.append("--require-hip-kernels" if args.require_hip_kernels else "--no-require-hip-kernels")

    print(f"running {case.case_id} -> {output_json}", file=sys.stderr)
    result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.stdout.strip():
        print(result.stdout.strip(), file=sys.stderr)
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(f"prompt case {case.case_id} failed with code {result.returncode}")

    report = json.loads(output_json.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise SystemExit(f"{output_json}: expected JSON object")
    report.setdefault("suite_case", {})
    report["suite_case"].update({"id": case.case_id, "category": case.category, "notes": case.notes})
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def metric_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_report(case: PromptCase, report_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    decode = as_mapping(report.get("decode", {}), "report.decode")
    step = as_mapping(decode.get("step_wall_summary", {}), "report.decode.step_wall_summary")
    prefill = as_mapping(report.get("prefill", {}), "report.prefill")
    stop = as_mapping(report.get("stop", {}), "report.stop")
    text_prompt = as_mapping(report.get("text_prompt", {}), "report.text_prompt")
    decoded_text = as_mapping(report.get("decoded_text", {}), "report.decoded_text")
    generated_text = str(decoded_text.get("generated", ""))
    generated_compact = " ".join(generated_text.split())
    generated_token_ids = report.get("generated_token_ids", [])
    if not isinstance(generated_token_ids, list):
        generated_token_ids = []
    tokens = [int(token) for token in generated_token_ids]
    unique_ratio = (len(set(tokens)) / len(tokens)) if tokens else None
    requested_generated_tokens = decode.get("requested_generated_tokens")
    warnings = output_warnings(
        case,
        generated_text,
        len(tokens),
        requested_generated_tokens,
        str(stop.get("reason")),
        unique_ratio,
    )

    return {
        "id": case.case_id,
        "category": case.category,
        "report": str(report_path),
        "prompt_tokens": text_prompt.get("token_count"),
        "generated_tokens": len(tokens),
        "requested_generated_tokens": requested_generated_tokens,
        "verified": report.get("verified"),
        "stop_reason": stop.get("reason"),
        "stopped": stop.get("stopped"),
        "stopped_on_token_id": stop.get("stopped_on_token_id"),
        "prefill_tps": metric_float(prefill.get("tps")),
        "decode_tps": metric_float(step.get("all_step_tps")),
        "skip2_tps": metric_float(step.get("warmup_skip_2_step_tps")),
        "last8_tps": metric_float(step.get("last_8_step_tps")),
        "p50_ms": metric_float(step.get("p50_ms")),
        "generated_chars": len(generated_text),
        "unique_generated_token_ratio": unique_ratio,
        "output_status": "ok" if not warnings else "warn",
        "output_warnings": warnings,
        "generated_preview": generated_compact[:240],
        "notes": case.notes,
    }


def output_warnings(
    case: PromptCase,
    generated_text: str,
    generated_tokens: int,
    requested_generated_tokens: Any,
    stop_reason: str,
    unique_ratio: float | None,
) -> list[str]:
    warnings: list[str] = []
    compact = " ".join(generated_text.split())
    lowered = generated_text.lower()
    if not compact:
        warnings.append("empty_generated_text")
    try:
        requested = int(requested_generated_tokens)
    except (TypeError, ValueError):
        requested = None
    if requested is not None and generated_tokens >= requested and stop_reason == "max_generated_tokens":
        warnings.append("hit_generation_limit")
    if unique_ratio is not None and generated_tokens >= 32 and unique_ratio < 0.35:
        warnings.append("low_unique_token_ratio")
    if any(marker in lowered for marker in CONTROL_MARKERS):
        warnings.append("control_marker_text")
    prompt_prefix = " ".join(case.prompt.split())[:80]
    if prompt_prefix and prompt_prefix in compact:
        warnings.append("prompt_echo")
    if compact and generated_tokens >= 64 and not compact.endswith(TERMINAL_TEXT_ENDINGS):
        warnings.append("missing_terminal_punctuation")
    return warnings


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def category_metrics(case_summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    categories: dict[str, list[dict[str, Any]]] = {}
    for item in case_summaries:
        categories.setdefault(str(item.get("category", "general")), []).append(item)
    metrics: dict[str, dict[str, Any]] = {}
    for category, items in sorted(categories.items()):
        decode_values = [value for item in items if (value := item.get("decode_tps")) is not None]
        prefill_values = [value for item in items if (value := item.get("prefill_tps")) is not None]
        metrics[category] = {
            "case_count": len(items),
            "decode_tps_mean": mean(decode_values),
            "decode_tps_min": min(decode_values) if decode_values else None,
            "decode_tps_max": max(decode_values) if decode_values else None,
            "prefill_tps_mean": mean(prefill_values),
            "verified_all": all(item.get("verified") is True for item in items),
            "output_ok_count": sum(1 for item in items if item.get("output_status") == "ok"),
            "output_warn_count": sum(1 for item in items if item.get("output_status") == "warn"),
        }
    return metrics


def write_summary_json(
    path: Path,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    case_summaries: list[dict[str, Any]],
) -> None:
    decode_values = [value for item in case_summaries if (value := item.get("decode_tps")) is not None]
    prefill_values = [value for item in case_summaries if (value := item.get("prefill_tps")) is not None]
    payload = {
        "schema_version": "package-token-prompt-suite-summary-v0.2",
        "suite": metadata,
        "package": args.package_dir,
        "tokenizer_dir": args.tokenizer_dir,
        "device_index": args.device_index,
        "layers": args.layers,
        "generated_tokens_default": args.generated_tokens,
        "stop_policy": {
            "stop_token_ids": args.stop_token_ids,
            "stop_on_eos": args.stop_on_eos,
            "stop_on_special_tokens": args.stop_on_special_tokens,
        },
        "case_count": len(case_summaries),
        "metrics": {
            "decode_tps_mean": mean(decode_values),
            "decode_tps_min": min(decode_values) if decode_values else None,
            "decode_tps_max": max(decode_values) if decode_values else None,
            "prefill_tps_mean": mean(prefill_values),
            "verified_all": all(item.get("verified") is True for item in case_summaries),
            "stopped_count": sum(1 for item in case_summaries if item.get("stopped") is True),
            "output_ok_count": sum(1 for item in case_summaries if item.get("output_status") == "ok"),
            "output_warn_count": sum(1 for item in case_summaries if item.get("output_status") == "warn"),
            "hit_generation_limit_count": sum(
                1 for item in case_summaries if "hit_generation_limit" in item.get("output_warnings", [])
            ),
            "low_unique_token_ratio_count": sum(
                1 for item in case_summaries if "low_unique_token_ratio" in item.get("output_warnings", [])
            ),
            "prompt_echo_count": sum(1 for item in case_summaries if "prompt_echo" in item.get("output_warnings", [])),
        },
        "category_metrics": category_metrics(case_summaries),
        "cases": case_summaries,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return ""
    return str(value)


def write_summary_md(path: Path, summary_json: Path, case_summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Package Token Prompt Suite Summary",
        "",
        f"- Summary JSON: `{summary_json}`",
        "",
        "| case | category | prompt | generated | stop | status | warnings | prefill tok/s | decode tok/s | skip-2 tok/s | last-8 tok/s | p50 ms | verified | preview |",
        "| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | --- |",
    ]
    for item in case_summaries:
        lines.append(
            "| {case} | {category} | {prompt} | {generated} | {stop} | {status} | {warnings} | {prefill} | {decode} | {skip2} | {last8} | {p50} | {verified} | {preview} |".format(
                case=item["id"],
                category=fmt(item.get("category")),
                prompt=fmt(item.get("prompt_tokens")),
                generated=fmt(item.get("generated_tokens")),
                stop=fmt(item.get("stop_reason")),
                status=fmt(item.get("output_status")),
                warnings=", ".join(item.get("output_warnings", [])),
                prefill=fmt(item.get("prefill_tps")),
                decode=fmt(item.get("decode_tps")),
                skip2=fmt(item.get("skip2_tps")),
                last8=fmt(item.get("last8_tps")),
                p50=fmt(item.get("p50_ms")),
                verified=fmt(item.get("verified")),
                preview=str(item.get("generated_preview", "")).replace("|", "\\|"),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_existing_case(case: PromptCase, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    report_path = output_dir / f"{safe_filename(case.case_id)}.json"
    if not report_path.exists():
        raise SystemExit(f"missing existing report for {case.case_id}: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse {report_path}: {exc}") from exc
    if not isinstance(report, dict):
        raise SystemExit(f"{report_path}: expected JSON object")
    return report_path, report


def main() -> int:
    args = parse_args()
    if args.generated_tokens <= 0:
        raise SystemExit("--generated-tokens must be positive")
    metadata, cases = load_suite(args.suite_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_summaries: list[dict[str, Any]] = []
    for case in cases:
        report_path = args.output_dir / f"{safe_filename(case.case_id)}.json"
        if args.summarize_existing:
            report_path, report = load_existing_case(case, args.output_dir)
        else:
            report = run_case(args, case, report_path)
        case_summaries.append(summarize_report(case, report_path, report))

    summary_json = args.output_dir / args.summary_json
    summary_md = args.output_dir / args.summary_md
    write_summary_json(summary_json, args, metadata, case_summaries)
    write_summary_md(summary_md, summary_json, case_summaries)
    print(f"wrote {summary_json}", file=sys.stderr)
    print(f"wrote {summary_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
