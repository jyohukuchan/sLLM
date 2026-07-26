#!/usr/bin/env python3
"""Summarize unprofiled wide-M overlay measurements without a speed gate.

This intentionally reads only the wall-clock summary emitted by the
full-model driver.  rocprof traces are parsed separately for dispatch counts
and are never converted into tok/s.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROMPTS = (128, 512, 1024, 2048, 4095)
WIDTHS = (128, 256, 512, 1024, 2048)
LLAMA_F32_KV_TOK_S = {
    128: 1165.756,
    512: 1195.722,
    1024: 1145.351,
    2048: 1058.379,
    4095: 1008.683,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    values = [json.loads(line) for line in lines]
    if not values or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"{path}: expected nonempty JSON object lines")
    return values


def throughput_row(run_root: Path, width: int, prompt: int) -> dict[str, Any]:
    path = run_root / "throughput" / f"m{width}-p{prompt}.jsonl"
    if not path.is_file():
        return {
            "resident_m": width,
            "prompt_tokens": prompt,
            "status": "not-run",
            "reason": "no raw driver record",
        }
    values = read_json_lines(path)
    configurations = [value for value in values if value.get("event") == "configuration"]
    summaries = [value for value in values if value.get("event") == "summary"]
    measured = [value for value in values if value.get("event") == "measured_region"]
    if len(configurations) != 1 or len(summaries) != 1:
        raise ValueError(f"{path}: expected one configuration and one summary")
    config = configurations[0]
    summary = summaries[0]
    calls = [value.get("prefill_advance_calls") for value in measured]
    if not all(isinstance(call, int) for call in calls):
        raise ValueError(f"{path}: measured rows lack prefill_advance_calls")
    tok_s = summary.get("mean_units_per_second")
    if not isinstance(tok_s, (int, float)):
        raise ValueError(f"{path}: summary lacks mean_units_per_second")
    llama = LLAMA_F32_KV_TOK_S[prompt]
    return {
        "resident_m": width,
        "prompt_tokens": prompt,
        "status": "measured",
        "raw_driver": str(path),
        "measurement_source": "synchronized full-model driver wall-clock summary",
        "unprofiled_repeats": summary.get("repeats"),
        "tok_s": float(tok_s),
        "llama_cpp_q8_0_f32_kv_tok_s": llama,
        "llama_over_ullm": llama / float(tok_s),
        "prefill_advance_calls": calls,
        "prefill_implementation": config.get("load", {}).get("prefill_implementation"),
        "device": config.get("device"),
    }


def trace_row(run_root: Path, width: int) -> dict[str, Any]:
    path = run_root / "traces" / f"m{width}-p4095-analysis.json"
    if not path.is_file():
        return {
            "resident_m": width,
            "prompt_tokens": 4095,
            "status": "not-run",
            "reason": "no trace analysis",
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: trace analysis is not an object")
    aggregate = value.get("aggregate")
    if not isinstance(aggregate, dict) or not isinstance(
        aggregate.get("attention_dispatches"), int
    ):
        raise ValueError(f"{path}: trace analysis lacks attention dispatch count")
    return {
        "resident_m": width,
        "prompt_tokens": 4095,
        "status": "traced",
        "trace_analysis": str(path),
        "attention_dispatches": aggregate["attention_dispatches"],
        "note": "trace dispatch count; profiler duration is not used as throughput",
    }


def numerical_row(run_root: Path, width: int) -> dict[str, Any]:
    if width == 128:
        path = run_root / "numerical" / "m128" / "result.json"
        return {
            "resident_m": width,
            "status": "baseline-present" if path.is_file() else "not-run",
            "result": str(path) if path.is_file() else None,
            "note": "M=128 is the numerical comparison baseline",
        }
    path = run_root / "numerical" / f"m{width}-vs-m128.json"
    if not path.is_file():
        return {
            "resident_m": width,
            "status": "not-run",
            "reason": "no baseline/candidate oracle comparison",
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    comparisons = value.get("comparisons") if isinstance(value, dict) else None
    if not isinstance(comparisons, list) or len(comparisons) != len(PROMPTS):
        raise ValueError(f"{path}: expected five oracle comparisons")
    compact: list[dict[str, Any]] = []
    for item in comparisons:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: comparison is not an object")
        hidden = item.get("final_hidden")
        logits = item.get("logits")
        generated = item.get("generated_token_ids")
        if not isinstance(hidden, dict) or not isinstance(logits, dict) or not isinstance(generated, dict):
            raise ValueError(f"{path}: comparison lacks numerical fields")
        compact.append(
            {
                "prompt_tokens": item.get("prompt_tokens"),
                "generated_token_ids_exact": generated.get("exact"),
                "final_hidden_exact_f32_le_bytes": hidden.get("exact_f32_le_bytes"),
                "final_hidden_max_abs": hidden.get("max_abs"),
                "final_hidden_relative_l2": hidden.get("relative_l2"),
                "final_hidden_nonfinite_count": hidden.get("nonfinite_count"),
                "logits_exact_f32_le_bytes": logits.get("exact_f32_le_bytes"),
                "logits_max_abs": logits.get("max_abs"),
                "logits_relative_l2": logits.get("relative_l2"),
                "logits_nonfinite_count": logits.get("nonfinite_count"),
            }
        )
    return {
        "resident_m": width,
        "status": "compared",
        "comparison": str(path),
        "comparisons": compact,
        "all_generated_token_ids_exact": all(
            item["generated_token_ids_exact"] is True for item in compact
        ),
        "all_final_hidden_f32_le_bytes_exact": all(
            item["final_hidden_exact_f32_le_bytes"] is True for item in compact
        ),
        "all_logits_f32_le_bytes_exact": all(
            item["logits_exact_f32_le_bytes"] is True for item in compact
        ),
        "note": "recorded without a scalar numerical acceptance threshold",
    }


def decode_row(run_root: Path) -> dict[str, Any]:
    path = run_root / "decode" / "m128-p1024.jsonl"
    if not path.is_file():
        return {"status": "not-run", "reason": "no decode driver record"}
    values = read_json_lines(path)
    summaries = [value for value in values if value.get("event") == "summary"]
    if len(summaries) != 1:
        raise ValueError(f"{path}: expected one decode summary")
    summary = summaries[0]
    rate = summary.get("mean_units_per_second")
    if not isinstance(rate, (int, float)):
        raise ValueError(f"{path}: decode summary lacks mean_units_per_second")
    return {
        "status": "measured",
        "raw_driver": str(path),
        "measurement_source": "synchronized full-model driver wall-clock summary",
        "prompt_tokens": 1024,
        "resident_m": 128,
        "measured_decode_tokens_per_repeat": summary.get("units_per_repeat"),
        "unprofiled_repeats": summary.get("repeats"),
        "tok_s": float(rate),
        "reference_tok_s": 27.378731,
        "ratio_to_reference": float(rate) / 27.378731,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Wide-M overlay full-model summary",
        "",
        "The tok/s column is the unprofiled, synchronized full-model driver's five-repeat wall-clock mean. It is not derived from a profiler range. The binary was built from the isolated wide-M admission overlay; see `wide-m-overlay.md`.",
        "",
        "| M | prompt | SQ8_0 tok/s | llama.cpp Q8_0 F32-KV tok/s | llama/uLLM | advances/repeat |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["throughput"]:
        if row["status"] != "measured":
            lines.append(
                f"| {row['resident_m']} | {row['prompt_tokens']} | unmeasured | {LLAMA_F32_KV_TOK_S[row['prompt_tokens']]:.3f} | — | — |"
            )
            continue
        calls = ", ".join(str(value) for value in row["prefill_advance_calls"])
        lines.append(
            f"| {row['resident_m']} | {row['prompt_tokens']} | {row['tok_s']:.3f} | "
            f"{row['llama_cpp_q8_0_f32_kv_tok_s']:.3f} | {row['llama_over_ullm']:.3f}x | {calls} |"
        )
    lines.extend(
        [
            "",
            "## N=4095 attention traces",
            "",
            "| M | cached-prefix attention dispatches | status |",
            "| ---: | ---: | --- |",
        ]
    )
    for row in summary["traces"]:
        count = row.get("attention_dispatches", "unmeasured")
        lines.append(f"| {row['resident_m']} | {count} | {row['status']} |")
    lines.extend(
        [
            "",
            "M=4096 is intentionally absent from this N=4095 grid: no-padding semantics leave it on M=1 seeds, so it is not a wider cached-prefix execution at that prompt length.",
            "",
            "## Numerical comparisons against M=128",
            "",
            "These values are recorded without a scalar numerical pass/fail threshold.",
            "",
            "| M | generated IDs exact for all prompts | final hidden F32 bytes exact for all prompts | logits F32 bytes exact for all prompts |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for row in summary["numerical"]:
        if row["resident_m"] == 128:
            lines.append("| 128 | baseline | baseline | baseline |")
        elif row["status"] == "compared":
            lines.append(
                f"| {row['resident_m']} | {row['all_generated_token_ids_exact']} | "
                f"{row['all_final_hidden_f32_le_bytes_exact']} | {row['all_logits_f32_le_bytes_exact']} |"
            )
        else:
            lines.append(f"| {row['resident_m']} | unmeasured | unmeasured | unmeasured |")
    decode = summary["decode"]
    lines.extend(
        [
            "",
            "## Decode M=128 control",
            "",
        ]
    )
    if decode["status"] == "measured":
        lines.append(
            f"Synchronized full-model decode: {decode['tok_s']:.6f} tok/s; reference "
            f"27.378731 tok/s; ratio {decode['ratio_to_reference']:.6f}x."
        )
    else:
        lines.append("Unmeasured.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    summary = {
        "schema_version": "ullm.sq8.prefill_chunk_width.overlay_summary.v1",
        "scope": "isolated lower-admission overlay; unprofiled full-model rates and trace dispatch attribution",
        "throughput": [
            throughput_row(args.run_root, width, prompt)
            for width in WIDTHS
            for prompt in PROMPTS
        ],
        "traces": [trace_row(args.run_root, width) for width in WIDTHS],
        "numerical": [numerical_row(args.run_root, width) for width in WIDTHS],
        "decode": decode_row(args.run_root),
        "not_a_gate": "numerical and text evidence are recorded separately; no scalar threshold decides acceptance",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
