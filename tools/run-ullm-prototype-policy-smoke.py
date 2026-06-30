#!/usr/bin/env python3
"""Run small multi-tensor ullm-quant prototype conversion smokes."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def load_codebook_keys(path: Path) -> set[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        (entry["family"], entry["candidate_id"])
        for entry in data.get("codebooks", [])
    }


def select_tensors(
    plan: dict,
    codebook_keys: set[tuple[str, str]],
    families: set[str],
    max_tensors: int,
    per_family: int,
) -> list[dict]:
    selected: list[dict] = []
    family_counts: dict[str, int] = {}
    for tensor in plan.get("tensors", []):
        if tensor.get("action") != "quantize":
            continue
        family = tensor.get("family")
        candidate = tensor.get("quant_format")
        if family not in families:
            continue
        if (family, candidate) not in codebook_keys:
            continue
        count = family_counts.get(family, 0)
        if count >= per_family:
            continue
        selected.append(tensor)
        family_counts[family] = count + 1
        if len(selected) >= max_tensors:
            break
    return selected


def parse_log(path: Path) -> dict:
    result: dict[str, str | int | float] = {}
    key_value = re.compile(r"^([A-Za-z0-9_]+)=(.*)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = key_value.match(line.strip())
        if match:
            key, value = match.groups()
            result[key] = value
            try:
                result[key] = int(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    pass
            continue
        if "Elapsed (wall clock) time" in line:
            result["elapsed_raw"] = line.split("):", maxsplit=1)[-1].strip()
        elif "Maximum resident set size (kbytes)" in line:
            result["max_rss_kib"] = int(line.rsplit(":", maxsplit=1)[-1].strip())
        elif "User time (seconds)" in line:
            result["user_seconds"] = float(line.rsplit(":", maxsplit=1)[-1].strip())
        elif "System time (seconds)" in line:
            result["system_seconds"] = float(line.rsplit(":", maxsplit=1)[-1].strip())
    return result


def run_one(args: argparse.Namespace, plan: dict, tensor: dict, index: int) -> dict:
    name = tensor["name"]
    family = tensor["family"]
    candidate = tensor["quant_format"]
    output_dir = args.prototype_root / f"{index:03d}-{sanitize(name)}.ullm.d"
    log_path = args.log_dir / f"{index:03d}-{sanitize(name)}.log"
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    command = [
        "/usr/bin/time",
        "-v",
        str(args.binary),
        "--model-dir",
        plan["model_dir"],
        "--aq-policy",
        plan["aq_policy"]["policy_id"],
        "--aq-low-format",
        plan["aq_policy"]["low_format"],
        "--aq-high-format",
        plan["aq_policy"]["high_format"],
        "--inspect-tensor",
        name,
        "--skip-inspect",
        "--inspect-aq-format",
        candidate,
        "--codebook-json",
        str(args.codebook_json),
        "--inspect-codebook-family",
        family,
        "--inspect-codebook-candidate",
        candidate,
        "--chunk-bytes",
        str(args.chunk_bytes),
        "--scale-window",
        str(args.scale_window),
        "--prototype-output-dir",
        str(output_dir),
        "--dry-run",
    ]
    if not args.verify:
        command.insert(-1, "--prototype-skip-verify")
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            command,
            cwd=args.repo_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    parsed = parse_log(log_path)
    return {
        "tensor": name,
        "family": family,
        "candidate": candidate,
        "returncode": proc.returncode,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "metrics": parsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--binary", type=Path, default=Path("target/release/ullm-quant"))
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--codebook-json", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--prototype-root", type=Path, required=True)
    parser.add_argument("--family", action="append", default=["mlp_up", "attn_k"])
    parser.add_argument("--max-tensors", type=int, default=4)
    parser.add_argument("--per-family", type=int, default=2)
    parser.add_argument("--chunk-bytes", type=int, default=1_048_576)
    parser.add_argument("--scale-window", type=int, default=4)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.plan_json = args.plan_json.resolve()
    args.codebook_json = args.codebook_json.resolve()
    args.summary_output = args.summary_output.resolve()
    args.log_dir = args.log_dir.resolve()
    args.prototype_root = args.prototype_root.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.prototype_root.mkdir(parents=True, exist_ok=True)

    plan = json.loads(args.plan_json.read_text(encoding="utf-8"))
    codebook_keys = load_codebook_keys(args.codebook_json)
    selected = select_tensors(
        plan,
        codebook_keys,
        set(args.family),
        args.max_tensors,
        args.per_family,
    )
    rows = [run_one(args, plan, tensor, index) for index, tensor in enumerate(selected)]
    summary = {
        "schema_version": "ullm-prototype-policy-smoke-v0.1",
        "plan_json": str(args.plan_json),
        "codebook_json": str(args.codebook_json),
        "prototype_root": str(args.prototype_root),
        "log_dir": str(args.log_dir),
        "families": args.family,
        "max_tensors": args.max_tensors,
        "per_family": args.per_family,
        "scale_window": args.scale_window,
        "verify": args.verify,
        "selected_count": len(selected),
        "results": rows,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
