#!/usr/bin/env python3
"""Run registered host rows locally without duplicating their suite definitions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, ROOT, load_manifests  # noqa: E402

HOST_ROWS = ("h0", "h1", "h2")
PROHIBITED_LOCAL_ATTRIBUTES = ("requires_gpu", "requires_model", "network")


def parse_rows(value: str) -> tuple[str, ...]:
    rows = tuple(item.strip() for item in value.split(",") if item.strip())
    if not rows or len(set(rows)) != len(rows):
        raise ContractError("local verification rows must be a non-empty unique list")
    unknown = sorted(set(rows) - set(HOST_ROWS))
    if unknown:
        raise ContractError(f"local verification accepts registered host rows only: {unknown}")
    return rows


def registered_rows(repo: Path, selected: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    suites, host, _paths = load_manifests(repo)
    suite_by_id = {item["suite_id"]: item for item in suites["suites"]}
    row_by_id = {item["row_id"]: item for item in host["rows"]}
    if tuple(row_by_id) != HOST_ROWS:
        raise ContractError("host matrix row order is not exactly h0, h1, h2")
    for row_id in selected:
        row = row_by_id[row_id]
        for suite_id in row["suite_ids"]:
            suite = suite_by_id[suite_id]
            attributes = suite["attributes"]
            enabled = [name for name in PROHIBITED_LOCAL_ATTRIBUTES if attributes[name]]
            if enabled:
                raise ContractError(
                    f"host row {row_id} suite {suite_id} requests prohibited local attributes: {enabled}"
                )
    return {row_id: row_by_id[row_id] for row_id in selected}


def row_command(
    repo: Path,
    output_root: Path,
    row_id: str,
    *,
    strict: bool,
    candidate_sha: str | None,
    run_id: str,
) -> list[str]:
    command = [
        sys.executable,
        str(repo / "ci/tools/run_host_suite.py"),
        "--row",
        row_id,
        "--repo",
        str(repo),
        "--output-dir",
        str(output_root / row_id),
        "--run-id",
        run_id,
        "--run-attempt",
        "1",
    ]
    if strict:
        if candidate_sha is None:
            raise ContractError("strict local verification requires a candidate SHA")
        command.extend([
            "--strict-ci",
            "--expected-reviewed-sha",
            candidate_sha,
            "--expected-tested-sha",
            candidate_sha,
            "--expected-workflow-sha",
            candidate_sha,
        ])
    else:
        command.append("--allow-dirty-local")
    return command


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, default=ROOT)
    result.add_argument("--rows", default=",".join(HOST_ROWS))
    result.add_argument("--output-root", type=Path)
    result.add_argument("--list", action="store_true", help="show registered host rows without executing them")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--allow-dirty-local", action="store_true")
    mode.add_argument("--strict", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo = args.repo.resolve()
    selected = parse_rows(args.rows)
    rows = registered_rows(repo, selected)
    if args.list:
        print(json.dumps({row_id: row["suite_ids"] for row_id, row in rows.items()}, indent=2))
        return 0
    if args.allow_dirty_local == args.strict:
        raise ContractError("choose exactly one of --allow-dirty-local or --strict")
    if args.output_root is None:
        raise ContractError("--output-root is required for execution")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise ContractError(f"local verification output root already exists: {output_root}")
    output_root.mkdir(parents=True)

    candidate_sha = None
    if args.strict:
        candidate_sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
    run_id = f"local-{int(time.time())}"
    for row_id in selected:
        completed = subprocess.run(
            row_command(
                repo,
                output_root,
                row_id,
                strict=args.strict,
                candidate_sha=candidate_sha,
                run_id=run_id,
            ),
            check=False,
        )
        if completed.returncode != 0:
            return completed.returncode

    if selected == HOST_ROWS:
        needs_path = output_root / "needs.json"
        needs_path.write_text(
            json.dumps({row_id: {"result": "success"} for row_id in HOST_ROWS}, indent=2) + "\n",
            encoding="utf-8",
        )
        aggregate = [
            sys.executable,
            str(repo / "ci/tools/aggregate_host_results.py"),
            "--repo",
            str(repo),
            "--needs-json",
            str(needs_path),
            "--artifact-dir",
            str(output_root),
            "--output-dir",
            str(output_root / "aggregate"),
            "--run-id",
            run_id,
            "--run-attempt",
            "1",
        ]
        if args.strict:
            assert candidate_sha is not None
            aggregate.extend([
                "--strict-ci",
                "--expected-reviewed-sha",
                candidate_sha,
                "--expected-tested-sha",
                candidate_sha,
                "--expected-workflow-sha",
                candidate_sha,
            ])
        else:
            aggregate.append("--allow-local-development")
        return subprocess.run(aggregate, check=False).returncode
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, subprocess.SubprocessError) as exc:
        print(f"local verification: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(3)
