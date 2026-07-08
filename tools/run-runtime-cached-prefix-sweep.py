#!/usr/bin/env python3
"""Run runtime cached-prefix attention smoke cases and save JSONL rows."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "runtime-cached-prefix-attn-sweep-v0.1"
SMOKE_PREFIX = "runtime-cached-prefix-attn-smoke "
EXECUTOR_ENVS = {
    "cached_prefix_chunked": {
        "ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_FP8_E4M3_KERNEL": "1"
    },
    "cached_prefix_flash2": {
        "ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_FP8_E4M3_FLASH2_KERNEL": "1"
    },
    "decode_loop": {"ULLM_REQUIRE_HIP_DECODE_ATTN_KERNEL": "1"},
}


@dataclass(frozen=True)
class SweepCase:
    case_id: str
    executor: str
    cached_prefix_tokens: int
    new_prefill_tokens: int
    measured_repeats: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/ullm-engine"))
    parser.add_argument("--device-index", type=int, default=2)
    parser.add_argument("--cached-prefix-tokens", default="4096,16384,65536")
    parser.add_argument("--new-tokens", default="1,16,128,512")
    parser.add_argument("--executors", default="cached_prefix_chunked")
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--value-dim", type=int, default=256)
    parser.add_argument("--measured-repeats", type=int, default=3)
    parser.add_argument("--long-measured-repeats", type=int, default=1)
    parser.add_argument("--long-prefix-threshold", type=int, default=16384)
    parser.add_argument("--long-new-token-threshold", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--max-estimated-attention-work", type=int, default=0)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def positive_int_csv(value: str, label: str) -> list[int]:
    values: list[int] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            raise SystemExit(f"{label} contains an empty item")
        try:
            parsed = int(item)
        except ValueError as exc:
            raise SystemExit(f"{label} item must be an integer: {item}") from exc
        if parsed <= 0:
            raise SystemExit(f"{label} item must be positive: {item}")
        values.append(parsed)
    return values


def executor_csv(value: str) -> list[str]:
    executors: list[str] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if item == "chunked":
            item = "cached_prefix_chunked"
        if item == "flash2":
            item = "cached_prefix_flash2"
        if item == "decode_attn_f32_loop":
            item = "decode_loop"
        if item not in EXECUTOR_ENVS:
            raise SystemExit(
                "executors must contain cached_prefix_chunked|chunked|cached_prefix_flash2|flash2|decode_loop"
            )
        executors.append(item)
    if not executors:
        raise SystemExit("executors must not be empty")
    return executors


def ensure_positive(value: int | float, label: str) -> None:
    if value <= 0:
        raise SystemExit(f"{label} must be positive")


def ensure_non_negative(value: int, label: str) -> None:
    if value < 0:
        raise SystemExit(f"{label} must be non-negative")


def estimated_attention_pairs(cached_prefix_tokens: int, new_tokens: int) -> int:
    return cached_prefix_tokens * new_tokens + new_tokens * (new_tokens + 1) // 2


def case_repeats(
    cached_prefix_tokens: int,
    new_tokens: int,
    measured_repeats: int,
    long_measured_repeats: int,
    long_prefix_threshold: int,
    long_new_token_threshold: int,
) -> int:
    if cached_prefix_tokens >= long_prefix_threshold or new_tokens >= long_new_token_threshold:
        return long_measured_repeats
    return measured_repeats


def build_cases(args: argparse.Namespace) -> list[SweepCase]:
    prefixes = positive_int_csv(args.cached_prefix_tokens, "cached-prefix-tokens")
    new_tokens_values = positive_int_csv(args.new_tokens, "new-tokens")
    executors = executor_csv(args.executors)
    ensure_non_negative(args.device_index, "device-index")
    for label in [
        "q-heads",
        "kv-heads",
        "head-dim",
        "value-dim",
        "measured-repeats",
        "long-measured-repeats",
        "long-prefix-threshold",
        "long-new-token-threshold",
        "timeout-seconds",
    ]:
        ensure_positive(getattr(args, label.replace("-", "_")), label)

    cases: list[SweepCase] = []
    for executor in executors:
        for cached_prefix_tokens in prefixes:
            for new_tokens in new_tokens_values:
                attention_work = estimated_attention_pairs(cached_prefix_tokens, new_tokens)
                if (
                    args.max_estimated_attention_work > 0
                    and attention_work > args.max_estimated_attention_work
                ):
                    continue
                repeats = case_repeats(
                    cached_prefix_tokens,
                    new_tokens,
                    args.measured_repeats,
                    args.long_measured_repeats,
                    args.long_prefix_threshold,
                    args.long_new_token_threshold,
                )
                case_id = f"{executor}-l{cached_prefix_tokens}-m{new_tokens}"
                cases.append(
                    SweepCase(
                        case_id=case_id,
                        executor=executor,
                        cached_prefix_tokens=cached_prefix_tokens,
                        new_prefill_tokens=new_tokens,
                        measured_repeats=repeats,
                    )
                )
    if not cases:
        raise SystemExit("no cases selected")
    return cases


def parse_scalar(value: str) -> Any:
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_smoke_stdout(stdout: str) -> dict[str, Any]:
    selected = ""
    for line in stdout.splitlines():
        if line.startswith(SMOKE_PREFIX):
            selected = line
    if not selected:
        raise ValueError("runtime-cached-prefix-attn-smoke output line was not found")

    row: dict[str, Any] = {}
    for field in shlex.split(selected)[1:]:
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        row[key] = parse_scalar(value)
    return row


def command_for_case(args: argparse.Namespace, case: SweepCase) -> list[str]:
    return [
        str(args.binary),
        "runtime-cached-prefix-attn-smoke",
        str(args.device_index),
        str(case.cached_prefix_tokens),
        str(case.new_prefill_tokens),
        str(case.measured_repeats),
        str(args.q_heads),
        str(args.kv_heads),
        str(args.head_dim),
        str(args.value_dim),
        case.executor,
    ]


def output_executor(case: SweepCase) -> str:
    if case.executor == "decode_loop":
        return "decode_attn_f32_loop"
    return case.executor


def run_case(args: argparse.Namespace, run_id: str, case: SweepCase) -> dict[str, Any]:
    command = command_for_case(args, case)
    env = os.environ.copy()
    env.update(EXECUTOR_ENVS[case.executor])
    common = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "case_id": case.case_id,
        "command": command,
        "required_env": EXECUTOR_ENVS[case.executor],
        "workload": {
            "prefill_mode": "cached_prefix",
            "executor": output_executor(case),
            "cached_prefix_tokens": case.cached_prefix_tokens,
            "new_prefill_tokens": case.new_prefill_tokens,
            "total_context_tokens_after_prefill": case.cached_prefix_tokens
            + case.new_prefill_tokens,
            "estimated_prefill_attention_work_tokens": estimated_attention_pairs(
                case.cached_prefix_tokens,
                case.new_prefill_tokens,
            ),
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "value_dim": args.value_dim,
            "warmup_runs": 1,
            "measured_repeats": case.measured_repeats,
        },
    }

    if args.dry_run:
        return {**common, "status": "dry_run"}

    try:
        completed = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            **common,
            "status": "timeout",
            "timeout_seconds": args.timeout_seconds,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }

    row = {
        **common,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        row["status"] = "failed"
        return row

    try:
        parsed = parse_smoke_stdout(completed.stdout)
    except ValueError as exc:
        row["status"] = "parse_failed"
        row["error"] = str(exc)
        return row

    row.update(
        {
            "status": "ok",
            "device": {
                "backend": parsed.get("backend"),
                "device_index": parsed.get("device_index"),
                "name": parsed.get("name"),
            },
            "metrics": {
                "wall_ms_mean": parsed.get("wall_ms_mean"),
                "wall_ms_min": parsed.get("wall_ms_min"),
                "wall_ms_max": parsed.get("wall_ms_max"),
                "prefill_total_input_tps": parsed.get("prefill_total_input_tps"),
                "attention_pair_tps_mean": parsed.get("attention_pair_tps_mean"),
                "cache_kv_bytes_total": parsed.get("cache_kv_bytes_total"),
                "q_bytes_total": parsed.get("q_bytes_total"),
                "output_bytes_total": parsed.get("output_bytes_total"),
            },
            "verification": {
                "verified": parsed.get("verified"),
                "verification": parsed.get("verification"),
                "sample_count": parsed.get("sample_count"),
                "sampled_max_abs_diff": parsed.get("sampled_max_abs_diff"),
            },
        }
    )
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            output.write("\n")


def format_value(value: Any, decimals: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    if value is None:
        return ""
    return str(value)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Runtime cached prefix attention sweep",
        "",
        f"- schema: `{SCHEMA_VERSION}`",
        f"- rows: {len(rows)}",
        "",
        "| status | executor | L | M | repeats | mean ms | new tok/s | pair/s | diff |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        workload = row.get("workload", {})
        metrics = row.get("metrics", {})
        verification = row.get("verification", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("status", "")),
                    str(workload.get("executor", "")),
                    format_value(workload.get("cached_prefix_tokens"), 0),
                    format_value(workload.get("new_prefill_tokens"), 0),
                    format_value(workload.get("measured_repeats"), 0),
                    format_value(metrics.get("wall_ms_mean")),
                    format_value(metrics.get("prefill_total_input_tps")),
                    format_value(metrics.get("attention_pair_tps_mean")),
                    format_value(verification.get("sampled_max_abs_diff"), 9),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    cases = build_cases(args)
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict[str, Any]] = []
    for case in cases:
        row = run_case(args, run_id, case)
        rows.append(row)
        print(
            f"{case.case_id}: {row.get('status')}",
            file=sys.stderr if row.get("status") != "ok" else sys.stdout,
        )
        if row.get("status") not in {"ok", "dry_run"} and not args.keep_going:
            break

    write_jsonl(args.output_jsonl, rows)
    if args.summary_md:
        write_summary(args.summary_md, rows)
    if any(row.get("status") not in {"ok", "dry_run"} for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
