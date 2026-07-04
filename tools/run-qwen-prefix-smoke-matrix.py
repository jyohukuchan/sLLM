#!/usr/bin/env python3
"""Run package-golden-prefix-smoke across fixture/condition matrices."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "qwen-prefix-smoke-matrix-run-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-bin", default="target/debug/ullm-engine")
    parser.add_argument("--package", required=True)
    parser.add_argument("--fixture", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="NAME[,package=PATH][,row_scale=PATH][,cell_delta=PATH]",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--chunk-bytes", default="1048576")
    parser.add_argument("--layer-start", default="0")
    parser.add_argument("--layer-end", default="12")
    parser.add_argument("--rotary-dim", default="64")
    parser.add_argument("--rope-base", default="10000000")
    parser.add_argument("--position-offset", default="0")
    parser.add_argument("--run-mode", default="actual_prefix")
    parser.add_argument("--input-dump-dir", default="none")
    parser.add_argument("--sampled-token-indices", default="none")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def parse_named_path(spec: str, flag: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f"{flag} must be LABEL=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"{flag} label must not be empty: {spec!r}")
    if not path:
        raise SystemExit(f"{flag} path must not be empty: {spec!r}")
    return label, path


def parse_condition(spec: str, default_package: str) -> dict[str, str]:
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if not parts:
        raise SystemExit("--condition must not be empty")
    condition = {
        "name": parts[0],
        "package": default_package,
        "row_scale": "none",
        "cell_delta": "none",
    }
    if not condition["name"]:
        raise SystemExit(f"--condition name must not be empty: {spec!r}")
    allowed = {"package", "row_scale", "cell_delta"}
    for part in parts[1:]:
        if "=" not in part:
            raise SystemExit(f"--condition option must be KEY=VALUE, got {part!r}")
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in allowed:
            raise SystemExit(f"unsupported --condition key {key!r}; expected one of {sorted(allowed)}")
        if not value:
            raise SystemExit(f"--condition {key} value must not be empty")
        condition[key] = value
    return condition


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unnamed"


def smoke_command(
    args: argparse.Namespace,
    package: str,
    fixture_path: str,
    report_path: Path,
    row_scale: str,
    cell_delta: str,
) -> list[str]:
    return [
        args.engine_bin,
        "package-golden-prefix-smoke",
        package,
        fixture_path,
        args.device_index,
        args.chunk_bytes,
        args.layer_start,
        args.layer_end,
        args.rotary_dim,
        args.rope_base,
        args.position_offset,
        str(report_path),
        args.run_mode,
        row_scale,
        args.input_dump_dir,
        args.sampled_token_indices,
        cell_delta,
    ]


def run_one(command: list[str], log_path: Path, dry_run: bool) -> tuple[int | None, str, str]:
    if dry_run:
        log_path.write_text("$ " + " ".join(command) + "\n", encoding="utf-8")
        return None, "", ""
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    log_path.write_text(
        "$ "
        + " ".join(command)
        + "\n\n[stdout]\n"
        + completed.stdout
        + "\n[stderr]\n"
        + completed.stderr,
        encoding="utf-8",
    )
    return completed.returncode, completed.stdout, completed.stderr


def tail(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen Prefix Smoke Matrix Run",
        "",
        f"- schema: `{summary['schema_version']}`",
        f"- dry run: `{summary['dry_run']}`",
        f"- run count: `{len(summary['runs'])}`",
        "",
        "| fixture | condition | returncode | report | log |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for run in summary["runs"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run["fixture_label"]),
                    str(run["condition"]),
                    "-" if run["returncode"] is None else str(run["returncode"]),
                    str(run["report_path"]),
                    str(run["log_path"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.fixture:
        raise SystemExit("at least one --fixture is required")
    if not args.condition:
        raise SystemExit("at least one --condition is required")
    fixtures = [parse_named_path(spec, "--fixture") for spec in args.fixture]
    conditions = [parse_condition(spec, args.package) for spec in args.condition]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for fixture_label, fixture_path in fixtures:
        for condition in conditions:
            run_label = f"{safe_label(fixture_label)}-{safe_label(condition['name'])}"
            report_path = args.output_dir / f"package-golden-prefix-{run_label}.jsonl"
            log_path = args.output_dir / f"package-golden-prefix-{run_label}.txt"
            command = smoke_command(
                args,
                condition["package"],
                fixture_path,
                report_path,
                condition["row_scale"],
                condition["cell_delta"],
            )
            returncode, stdout, stderr = run_one(command, log_path, args.dry_run)
            run = {
                "fixture_label": fixture_label,
                "fixture_path": fixture_path,
                "condition": condition["name"],
                "package": condition["package"],
                "row_scale": condition["row_scale"],
                "cell_delta": condition["cell_delta"],
                "report_path": str(report_path),
                "log_path": str(log_path),
                "command": command,
                "returncode": returncode,
                "stdout_tail": tail(stdout),
                "stderr_tail": tail(stderr),
            }
            runs.append(run)
            if returncode not in (None, 0) and not args.continue_on_error:
                break
        if runs and runs[-1]["returncode"] not in (None, 0) and not args.continue_on_error:
            break

    summary = {
        "schema_version": SCHEMA_VERSION,
        "dry_run": args.dry_run,
        "engine_bin": args.engine_bin,
        "default_package": args.package,
        "output_dir": str(args.output_dir),
        "runs": runs,
    }
    if args.summary_json:
        write_json(args.summary_json, summary)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(summary), encoding="utf-8")

    failed = [run for run in runs if run["returncode"] not in (None, 0)]
    print(f"qwen-prefix-smoke-matrix runs={len(runs)} failed={len(failed)} dry_run={args.dry_run}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
