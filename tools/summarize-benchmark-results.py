#!/usr/bin/env python3
"""Create compact Markdown tables from uLLM benchmark JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def quant_family(quantization: str) -> str:
    upper = quantization.upper()
    if upper.startswith("IQ"):
        return "I-Quant"
    if upper.startswith("UD") or "-UD-" in upper or "UD-" in upper:
        return "UD"
    if "FP8" in upper:
        return "FP8"
    if "_K" in upper or upper.endswith("_K") or "K_" in upper:
        return "K-Quant"
    return "Other"


def gib(value: Any) -> float | None:
    if not isinstance(value, int):
        return None
    return value / 1024**3


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def target_label(row: dict[str, Any]) -> str:
    hardware = row.get("hardware") or {}
    gpus = hardware.get("gpus") or []
    names = [gpu.get("name") or "GPU" for gpu in gpus]
    if not names:
        return "-"
    compact = []
    for name in names:
        if "V620" in name:
            compact.append("V620")
        elif "Graphics" in name:
            compact.append("R9700")
        else:
            compact.append(name)
    return "+".join(compact)


def markdown_table(rows: list[dict[str, Any]], include_failed: bool) -> str:
    selected = rows if include_failed else [row for row in rows if row.get("status") == "ok"]
    selected.sort(
        key=lambda row: (
            row.get("engine", {}).get("name") or "",
            row.get("model", {}).get("name") or "",
            quant_family(row.get("model", {}).get("quantization") or ""),
            row.get("model", {}).get("quantization") or "",
            target_label(row),
            row.get("workload", {}).get("prompt_tokens") or 0,
            row.get("workload", {}).get("generated_tokens") or 0,
        )
    )
    lines = [
        "| Status | Engine | Model | Family | Quant | Target | Workload | Decode tok/s | Consumed GiB | Decode x GiB | Source |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in selected:
        metrics = row.get("metrics") or {}
        model = row.get("model") or {}
        workload = row.get("workload") or {}
        consumed_bytes = metrics.get("vram_consumed_bytes")
        consumed_gib = gib(consumed_bytes)
        product = metrics.get("decode_tokens_per_second_times_vram_consumed_gib")
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("status") or "-",
                    row.get("engine", {}).get("name") or "-",
                    model.get("name") or "-",
                    quant_family(model.get("quantization") or ""),
                    model.get("quantization") or "-",
                    target_label(row),
                    f"pp{workload.get('prompt_tokens')}/tg{workload.get('generated_tokens')}/b{workload.get('batch_size')}",
                    fmt(metrics.get("decode_tokens_per_second")),
                    fmt(consumed_gib),
                    fmt(product),
                    Path(row.get("_source_file", "-")).name,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--include-failed", action="store_true")
    args = parser.parse_args()
    print(markdown_table(load_rows(args.jsonl), args.include_failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
