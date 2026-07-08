#!/usr/bin/env python3
"""Run SQ FP8 overlay logits guards and summarize AQ4/SQ top-k drift."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "sq-fp8-overlay-logits-guard-result-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--source-model-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--engine", type=Path, default=Path("target/debug/ullm-engine"))
    parser.add_argument(
        "--builder", type=Path, default=Path("tools/build-sq-fp8-w8a16-artifact.py")
    )
    parser.add_argument("--device-index", type=int, default=2)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--scale-granularity", choices=("row", "row_block", "tensor"), default="row")
    parser.add_argument("--scale-block-cols", type=int, default=256)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--token-ids", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--lm-head-chunk-rows", type=int, default=4096)
    parser.add_argument("--rotary-dim")
    parser.add_argument("--rope-base")
    parser.add_argument("--position-offset")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="NAME=REGEX",
        help="SQ artifact case. May be passed more than once.",
    )
    parser.add_argument("--overwrite-artifacts", action="store_true")
    parser.add_argument("--reuse-artifacts", action="store_true")
    return parser.parse_args()


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_cases(values: list[str]) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--case must be NAME=REGEX, got {value!r}")
        name, pattern = value.split("=", 1)
        name = sanitize(name)
        if not name:
            raise SystemExit(f"--case has empty name: {value!r}")
        if name in seen:
            raise SystemExit(f"--case name is duplicated: {name}")
        try:
            re.compile(pattern)
        except re.error as err:
            raise SystemExit(f"--case {name} has invalid regex: {err}") from err
        seen.add(name)
        cases.append((name, pattern))
    if not cases:
        raise SystemExit("at least one --case is required")
    return cases


def run_to_file(command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle:
        completed = subprocess.run(
            command,
            stdout=stdout_handle,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}\n"
            f"stderr: {completed.stderr}"
        )


def engine_tail(args: argparse.Namespace) -> list[str]:
    tail = [
        str(args.device_index),
        str(args.chunk_bytes),
        args.layers,
        args.token_ids,
        str(args.top_k),
        str(args.lm_head_chunk_rows),
    ]
    if args.rotary_dim is not None:
        tail.append(args.rotary_dim)
        if args.rope_base is not None:
            tail.append(args.rope_base)
            if args.position_offset is not None:
                tail.append(args.position_offset)
    elif args.rope_base is not None or args.position_offset is not None:
        raise SystemExit("--rope-base and --position-offset require --rotary-dim")
    return tail


def top_tokens(report: dict[str, Any]) -> list[int]:
    logits = report.get("top_logits")
    if not isinstance(logits, list):
        raise SystemExit("report is missing top_logits list")
    tokens = []
    for entry in logits:
        if not isinstance(entry, dict) or "token_id" not in entry:
            raise SystemExit("top_logits entry is missing token_id")
        tokens.append(int(entry["token_id"]))
    return tokens


def top_entry_for(report: dict[str, Any], token_id: int) -> dict[str, Any] | None:
    for entry in report.get("top_logits", []):
        if isinstance(entry, dict) and int(entry.get("token_id", -1)) == token_id:
            return entry
    return None


def case_summary(
    name: str,
    artifact_dir: Path,
    baseline_report: dict[str, Any],
    sq_report: dict[str, Any],
    raw_report_path: Path,
) -> dict[str, Any]:
    baseline_top = top_tokens(baseline_report)
    sq_top = top_tokens(sq_report)
    baseline_top1 = baseline_top[0]
    sq_top1 = sq_top[0]
    sq_baseline_top1_entry = top_entry_for(sq_report, baseline_top1)
    sq_top1_entry = top_entry_for(sq_report, sq_top1)
    top1_minus_baseline_top1 = None
    if sq_baseline_top1_entry is not None and sq_top1_entry is not None:
        top1_minus_baseline_top1 = float(sq_top1_entry["logit"]) - float(
            sq_baseline_top1_entry["logit"]
        )
    timing = sq_report.get("timing_ms", {})
    return {
        "name": name,
        "artifact_dir": str(artifact_dir),
        "raw_report_json": str(raw_report_path),
        "fp8_tensor_count": (sq_report.get("sq_overlay") or {}).get("fp8_tensor_count"),
        "passthrough_tensor_count": (sq_report.get("sq_overlay") or {}).get(
            "passthrough_tensor_count"
        ),
        "baseline_top1": baseline_top1,
        "sq_top1": sq_top1,
        "top1_match": baseline_top1 == sq_top1,
        "topk_common": len(set(baseline_top) & set(sq_top)),
        "baseline_top_tokens": baseline_top,
        "sq_top_tokens": sq_top,
        "baseline_top1_rank_in_sq_topk": (
            sq_top.index(baseline_top1) + 1 if baseline_top1 in sq_top else None
        ),
        "sq_top1_minus_baseline_top1_logit": top1_minus_baseline_top1,
        "verified": sq_report.get("verified"),
        "layer_load_ms": timing.get("layer_load"),
        "total_ms": timing.get("total"),
    }


def main() -> None:
    args = parse_args()
    cases = parse_cases(args.case)
    if args.chunk_bytes <= 0:
        raise SystemExit("--chunk-bytes must be greater than zero")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be greater than zero")
    if args.lm_head_chunk_rows <= 0:
        raise SystemExit("--lm-head-chunk-rows must be greater than zero")
    if args.scale_block_cols <= 0:
        raise SystemExit("--scale-block-cols must be greater than zero")

    raw_dir = args.output_root / "raw"
    artifacts_dir = args.output_root / "artifacts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    baseline_stdout = raw_dir / "baseline.json"
    baseline_stderr = raw_dir / "baseline.stderr.txt"
    baseline_command = [
        str(args.engine),
        "package-token-ids-logits-smoke",
        str(args.package),
        *engine_tail(args),
    ]
    run_to_file(baseline_command, baseline_stdout, baseline_stderr)
    baseline_report = read_json(baseline_stdout)

    summaries = []
    for name, pattern in cases:
        artifact_dir = artifacts_dir / name
        build_stdout = raw_dir / f"{name}.build.json"
        build_stderr = raw_dir / f"{name}.build.stderr.txt"
        if not args.reuse_artifacts or not (artifact_dir / "sq_manifest.json").exists():
            build_command = [
                "python3",
                str(args.builder),
                "--source-model-dir",
                str(args.source_model_dir),
                "--output-artifact",
                str(artifact_dir),
                "--base-package",
                str(args.package),
                "--include-regex",
                pattern,
                "--scale-granularity",
                args.scale_granularity,
            ]
            if args.scale_granularity == "row_block":
                build_command.extend(["--scale-block-cols", str(args.scale_block_cols)])
            if args.overwrite_artifacts:
                build_command.append("--overwrite")
            run_to_file(build_command, build_stdout, build_stderr)

        sq_stdout = raw_dir / f"{name}.sq.json"
        sq_stderr = raw_dir / f"{name}.sq.stderr.txt"
        sq_command = [
            str(args.engine),
            "sq-fp8-token-ids-logits-smoke",
            str(args.package),
            str(artifact_dir),
            *engine_tail(args),
        ]
        run_to_file(sq_command, sq_stdout, sq_stderr)
        sq_report = read_json(sq_stdout)
        summaries.append(case_summary(name, artifact_dir, baseline_report, sq_report, sq_stdout))

    baseline_top = top_tokens(baseline_report)
    report = {
        "schema_version": SCHEMA_VERSION,
        "package": str(args.package),
        "source_model_dir": str(args.source_model_dir),
        "engine": str(args.engine),
        "device_index": args.device_index,
        "scale_granularity": args.scale_granularity,
        "scale_block_cols": args.scale_block_cols if args.scale_granularity == "row_block" else None,
        "layers": args.layers,
        "token_ids": args.token_ids,
        "top_k": args.top_k,
        "lm_head_chunk_rows": args.lm_head_chunk_rows,
        "baseline_raw_report_json": str(baseline_stdout),
        "baseline_verified": baseline_report.get("verified"),
        "baseline_top1": baseline_top[0],
        "baseline_top_tokens": baseline_top,
        "cases": summaries,
        "top1_match_count": sum(1 for item in summaries if item["top1_match"]),
        "case_count": len(summaries),
    }
    write_json(args.result_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
