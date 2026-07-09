#!/usr/bin/env python3
"""Run the prompt-suite token/logits guards as one comparison bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_SUITE_GUARD = Path("tools/compare-package-token-prompt-suite.py")
DEFAULT_LOGITS_GUARD = Path("tools/compare-package-token-logits.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run prompt-suite generated-token/top-logits guard and optional "
            "standalone logits guard, then write a small bundle summary."
        )
    )
    parser.add_argument("--reference-summary", required=True, type=Path)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suite-guard-script", type=Path, default=DEFAULT_SUITE_GUARD)
    parser.add_argument("--logits-guard-script", type=Path, default=DEFAULT_LOGITS_GUARD)
    parser.add_argument("--logit-atol", type=float, default=1e-6)
    parser.add_argument(
        "--acceptance-mode",
        choices=("strict", "behavioral"),
        default="strict",
        help="Prompt-suite acceptance mode passed to compare-package-token-prompt-suite.py",
    )
    parser.add_argument("--reference-logits", type=Path)
    parser.add_argument("--candidate-logits", type=Path)
    parser.add_argument("--summary-json", default="guard-bundle-summary.json")
    parser.add_argument("--summary-md", default="guard-bundle-summary.md")
    return parser.parse_args()


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} {path}: expected JSON object")
    return value


def run_command(command: list[str]) -> int:
    result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.stdout.strip():
        print(result.stdout.strip(), file=sys.stderr)
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    return result.returncode


def write_md(path: Path, summary_json: Path, payload: dict[str, Any]) -> None:
    checks = payload["checks"]
    lines = [
        "# Package Prompt Guard Bundle",
        "",
        f"- Summary JSON: `{summary_json}`",
        f"- Reference: `{payload['reference']['label']}`",
        f"- Candidate: `{payload['candidate']['label']}`",
        f"- Passed: `{str(payload['passed']).lower()}`",
        "",
        "| check | passed | artifact | key metrics |",
        "| --- | :---: | --- | --- |",
    ]
    for check in checks:
        metrics = check.get("metrics", {})
        key_metrics = ", ".join(f"{key}={value}" for key, value in metrics.items())
        lines.append(
            "| {name} | {passed} | `{artifact}` | {metrics} |".format(
                name=check["name"],
                passed=str(check["passed"]).lower(),
                artifact=check["json"],
                metrics=key_metrics,
            )
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_summary_path(output_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path == output_dir or path.is_relative_to(output_dir):
        return path
    output_dir_resolved = output_dir.resolve(strict=False)
    path_resolved = path.resolve(strict=False)
    if path_resolved.is_relative_to(output_dir_resolved):
        return path
    return output_dir / path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suite_guard_json = args.output_dir / "prompt-suite-token-logits-guard.json"
    suite_guard_md = args.output_dir / "prompt-suite-token-logits-guard.md"
    suite_guard_exit_code = run_command(
        [
            sys.executable,
            str(args.suite_guard_script),
            "--reference-summary",
            str(args.reference_summary),
            "--candidate-summary",
            str(args.candidate_summary),
            "--reference-label",
            args.reference_label,
            "--candidate-label",
            args.candidate_label,
            "--logit-atol",
            str(args.logit_atol),
            "--acceptance-mode",
            args.acceptance_mode,
            "--output-json",
            str(suite_guard_json),
            "--output-md",
            str(suite_guard_md),
        ]
    )
    suite_guard = load_json_object(suite_guard_json, "suite guard")
    suite_metrics = suite_guard.get("metrics", {})
    checks = [
        {
            "name": "prompt_suite_token_logits",
            "json": str(suite_guard_json),
            "md": str(suite_guard_md),
            "passed": bool(suite_metrics.get("passed")),
            "metrics": {
                "exit_code": suite_guard_exit_code,
                "acceptance_mode": suite_metrics.get("acceptance_mode"),
                "strict_passed": suite_metrics.get("strict_passed"),
                "behavioral_passed": suite_metrics.get("behavioral_passed"),
                "compared_case_count": suite_metrics.get("compared_case_count"),
                "generated_token_match_count": suite_metrics.get("generated_token_match_count"),
                "generated_text_match_count": suite_metrics.get("generated_text_match_count"),
                "generated_without_stop_text_match_count": suite_metrics.get(
                    "generated_without_stop_text_match_count"
                ),
                "top_logits_match_count": suite_metrics.get("top_logits_match_count"),
                "max_prefill_top_logit_abs_diff": suite_metrics.get("max_prefill_top_logit_abs_diff"),
                "max_decode_last_top_logit_abs_diff": suite_metrics.get("max_decode_last_top_logit_abs_diff"),
            },
        }
    ]

    if args.reference_logits is not None or args.candidate_logits is not None:
        if args.reference_logits is None or args.candidate_logits is None:
            raise SystemExit("--reference-logits and --candidate-logits must be provided together")
        logits_guard_json = args.output_dir / "standalone-logits-guard.json"
        logits_guard_md = args.output_dir / "standalone-logits-guard.md"
        logits_guard_exit_code = run_command(
            [
                sys.executable,
                str(args.logits_guard_script),
                "--reference",
                str(args.reference_logits),
                "--candidate",
                str(args.candidate_logits),
                "--reference-label",
                args.reference_label,
                "--candidate-label",
                args.candidate_label,
                "--logit-atol",
                str(args.logit_atol),
                "--output-json",
                str(logits_guard_json),
                "--output-md",
                str(logits_guard_md),
            ]
        )
        logits_guard = load_json_object(logits_guard_json, "logits guard")
        logits_metrics = logits_guard.get("metrics", {})
        checks.append(
            {
                "name": "standalone_logits",
                "json": str(logits_guard_json),
                "md": str(logits_guard_md),
                "passed": bool(logits_metrics.get("passed")),
                "metrics": {
                    "exit_code": logits_guard_exit_code,
                    "prompt_tokens": logits_metrics.get("prompt_tokens"),
                    "top_count": logits_metrics.get("top_count"),
                    "top_token_ids_match": logits_metrics.get("top_token_ids_match"),
                    "max_abs_logit_diff": logits_metrics.get("max_abs_logit_diff"),
                },
            }
        )

    payload = {
        "schema_version": "package-prompt-guard-bundle-v0.2",
        "reference": {"label": args.reference_label, "summary": str(args.reference_summary)},
        "candidate": {"label": args.candidate_label, "summary": str(args.candidate_summary)},
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
    summary_json = resolve_summary_path(args.output_dir, args.summary_json)
    summary_md = resolve_summary_path(args.output_dir, args.summary_md)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_md(summary_md, summary_json, payload)
    print(f"wrote {summary_json}", file=sys.stderr)
    print(f"wrote {summary_md}", file=sys.stderr)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
