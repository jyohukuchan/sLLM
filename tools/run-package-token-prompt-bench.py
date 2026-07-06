#!/usr/bin/env python3
"""Run package-token-ids-bench from a text prompt through a local tokenizer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REQUIRED_HIP_KERNEL_ENVS = {
    "ULLM_REQUIRE_HIP_AQ4_KERNEL": "1",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL": "1",
    "ULLM_REQUIRE_HIP_ADD_KERNEL": "1",
    "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL": "1",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL": "1",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL": "1",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL": "1",
    "ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL": "1",
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL": "1",
    "ULLM_REQUIRE_HIP_ROPE_KERNEL": "1",
    "ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL": "1",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL": "1",
    "ULLM_REQUIRE_HIP_TOP1_KERNEL": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize a prompt with a local Hugging Face tokenizer, run "
            "ullm-engine package-token-ids-bench, and enrich the JSON report "
            "with decoded text."
        )
    )
    parser.add_argument("--package-dir", required=True, help="Path to the .ullm.d package")
    parser.add_argument("--tokenizer-dir", required=True, help="Local Hugging Face tokenizer dir")
    parser.add_argument(
        "--engine",
        default="target/release/ullm-engine",
        help="ullm-engine binary path",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Prompt text")
    prompt_group.add_argument("--prompt-file", type=Path, help="UTF-8 prompt text file")
    parser.add_argument("--output-json", type=Path, help="Write enriched JSON report to this path")
    parser.add_argument("--device-index", type=int, default=2)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--layers", default="all", help="Layer CSV or 'all'")
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
    parser.add_argument(
        "--stop-token-ids",
        help="Comma-separated token IDs that should stop generation early",
    )
    parser.add_argument(
        "--stop-token-sequences",
        help=(
            "Semicolon-separated token ID sequences that should stop generation early "
            "(for example '198,14162,25;198,15666,25')"
        ),
    )
    parser.add_argument(
        "--stop-text",
        action="append",
        default=[],
        help="Text substring to tokenize and use as a stop token sequence; may be repeated",
    )
    parser.add_argument(
        "--stop-on-eos",
        action="store_true",
        help="Append tokenizer.eos_token_id to stop token IDs when available",
    )
    parser.add_argument(
        "--stop-on-special-tokens",
        action="store_true",
        help="Append tokenizer.all_special_ids to stop token IDs",
    )
    parser.add_argument(
        "--require-hip-kernels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the optimized HIP kernels used by the current AQ4 path",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Render the prompt through tokenizer.apply_chat_template before tokenization",
    )
    parser.add_argument("--system-prompt", help="Optional system message for chat template")
    parser.add_argument(
        "--add-generation-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass add_generation_prompt to apply_chat_template",
    )
    parser.add_argument(
        "--add-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use tokenizer special tokens for non-chat-template prompts",
    )
    parser.add_argument(
        "--skip-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip special tokens when decoding generated text",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        help="Reject prompts longer than this token count",
    )
    parser.add_argument(
        "--target-prompt-tokens",
        type=int,
        help=(
            "Repeat a non-chat prompt and truncate token IDs to this length; "
            "useful for reproducible long-prefill probes"
        ),
    )
    parser.add_argument(
        "--print-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print a compact human-readable summary to stderr",
    )
    return parser.parse_args()


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    try:
        return args.prompt_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"failed to read prompt file {args.prompt_file}: {exc}") from exc


def load_tokenizer(tokenizer_dir: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required for prompt tokenization") from exc
    return AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        trust_remote_code=True,
    )


def encode_prompt(tokenizer: Any, args: argparse.Namespace, prompt: str) -> tuple[str, list[int]]:
    if args.apply_chat_template:
        messages: list[dict[str, str]] = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": prompt})
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=args.add_generation_prompt,
        )
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=args.add_generation_prompt,
        )
        if isinstance(token_ids, Mapping):
            token_ids = token_ids.get("input_ids")
        if token_ids is None:
            raise SystemExit("chat template tokenizer returned no input_ids")
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise SystemExit("chat template tokenizer returned more than one input row")
            token_ids = token_ids[0]
        return rendered, [int(token_id) for token_id in token_ids]

    encoded = tokenizer(
        prompt,
        add_special_tokens=args.add_special_tokens,
    )
    return prompt, [int(token_id) for token_id in encoded["input_ids"]]


def adjust_prompt_token_count(
    tokenizer: Any,
    args: argparse.Namespace,
    raw_prompt: str,
    rendered_prompt: str,
    token_ids: list[int],
) -> tuple[str, list[int], dict[str, Any]]:
    target = args.target_prompt_tokens
    metadata: dict[str, Any] = {
        "target_prompt_tokens": target,
        "target_prompt_repeated": False,
        "target_prompt_truncated": False,
    }
    if target is None:
        return rendered_prompt, token_ids, metadata
    if target <= 0:
        raise SystemExit("--target-prompt-tokens must be positive")
    if args.apply_chat_template:
        raise SystemExit("--target-prompt-tokens is only supported without --apply-chat-template")
    if not token_ids:
        raise SystemExit("cannot repeat a prompt that encoded to zero tokens")

    if len(token_ids) < target:
        repeats = max(1, (target + len(token_ids) - 1) // len(token_ids))
        text = "\n".join(raw_prompt for _ in range(repeats))
        encoded = tokenizer(text, add_special_tokens=args.add_special_tokens)
        token_ids = [int(token_id) for token_id in encoded["input_ids"]]
        while len(token_ids) < target:
            text = text + "\n" + raw_prompt
            encoded = tokenizer(text, add_special_tokens=args.add_special_tokens)
            next_ids = [int(token_id) for token_id in encoded["input_ids"]]
            if len(next_ids) <= len(token_ids):
                raise SystemExit("prompt repetition did not increase token count")
            token_ids = next_ids
        metadata["target_prompt_repeated"] = True

    if len(token_ids) > target:
        token_ids = token_ids[:target]
        metadata["target_prompt_truncated"] = True

    rendered_prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
    return rendered_prompt, token_ids, metadata


def parse_stop_token_ids(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
    parsed = []
    for raw in value.split(","):
        entry = raw.strip()
        if not entry:
            raise SystemExit(f"invalid --stop-token-ids {value!r}: empty entry")
        try:
            parsed.append(int(entry))
        except ValueError as exc:
            raise SystemExit(f"invalid stop token ID {entry!r}: {exc}") from exc
    return parsed


def parse_stop_token_sequences(value: str | None) -> list[list[int]]:
    if value is None or not value.strip():
        return []
    parsed: list[list[int]] = []
    for raw_sequence in value.split(";"):
        sequence = raw_sequence.strip()
        if not sequence:
            raise SystemExit(f"invalid --stop-token-sequences {value!r}: empty sequence")
        parsed.append(parse_stop_token_ids(sequence))
    return parsed


def resolve_stop_token_ids(tokenizer: Any, args: argparse.Namespace) -> list[int]:
    stop_ids = parse_stop_token_ids(args.stop_token_ids)
    if args.stop_on_eos:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            stop_ids.append(int(eos))
    if args.stop_on_special_tokens:
        stop_ids.extend(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []))

    unique = []
    seen = set()
    for token_id in stop_ids:
        if token_id in seen:
            continue
        seen.add(token_id)
        unique.append(token_id)
    return unique


def resolve_stop_token_sequences(tokenizer: Any, args: argparse.Namespace) -> list[list[int]]:
    sequences = parse_stop_token_sequences(args.stop_token_sequences)
    for text in args.stop_text:
        encoded = tokenizer(text, add_special_tokens=False)
        token_ids = [int(token_id) for token_id in encoded["input_ids"]]
        if not token_ids:
            raise SystemExit(f"--stop-text {text!r} encoded to zero tokens")
        sequences.append(token_ids)

    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for sequence in sequences:
        key = tuple(sequence)
        if key in seen:
            continue
        seen.add(key)
        unique.append(sequence)
    return unique


def format_stop_token_sequences(stop_token_sequences: list[list[int]]) -> str:
    return ";".join(",".join(str(token_id) for token_id in sequence) for sequence in stop_token_sequences)


def run_engine(
    args: argparse.Namespace,
    token_ids: list[int],
    stop_token_ids: list[int],
    stop_token_sequences: list[list[int]],
) -> dict[str, Any]:
    token_csv = ",".join(str(token_id) for token_id in token_ids)
    command = [
        args.engine,
        "package-token-ids-bench",
        args.package_dir,
        str(args.device_index),
        str(args.chunk_bytes),
        args.layers,
        token_csv,
        str(args.generated_tokens),
        str(args.top_k),
        str(args.lm_head_chunk_rows),
        str(args.rotary_dim),
        str(args.rope_base),
        str(args.position_offset),
        args.lm_head_mode,
    ]
    if stop_token_ids:
        command.append(",".join(str(token_id) for token_id in stop_token_ids))
    elif stop_token_sequences:
        command.append("none")
    if stop_token_sequences:
        command.append(format_stop_token_sequences(stop_token_sequences))
    env = os.environ.copy()
    if args.require_hip_kernels:
        env.update(REQUIRED_HIP_KERNEL_ENVS)

    result = subprocess.run(
        command,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise SystemExit(f"ullm-engine exited with code {result.returncode}")

    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        if result.stdout:
            print(result.stdout[:4000], file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise SystemExit(f"failed to parse ullm-engine JSON: {exc}") from exc

    if not isinstance(report, dict):
        raise SystemExit("ullm-engine report was not a JSON object")
    report["_runner"] = {
        "command": command,
        "required_hip_kernel_envs": REQUIRED_HIP_KERNEL_ENVS if args.require_hip_kernels else {},
        "resolved_stop_token_ids": stop_token_ids,
        "resolved_stop_token_sequences": stop_token_sequences,
    }
    if result.stderr.strip():
        report["_runner"]["stderr"] = result.stderr.strip()
    return report


def enrich_report(
    report: dict[str, Any],
    tokenizer: Any,
    args: argparse.Namespace,
    raw_prompt: str,
    rendered_prompt: str,
    prompt_token_ids: list[int],
) -> dict[str, Any]:
    report_prompt_ids = report.get("prompt_token_ids")
    if report_prompt_ids != prompt_token_ids:
        raise SystemExit("engine report prompt_token_ids did not match tokenizer output")

    generated_token_ids = report.get("generated_token_ids")
    if not isinstance(generated_token_ids, list):
        raise SystemExit("engine report has no generated_token_ids list")
    generated_token_ids = [int(token_id) for token_id in generated_token_ids]
    full_token_ids = prompt_token_ids + generated_token_ids
    generated_without_stop_sequence_ids = generated_token_ids
    stop = report.get("stop", {})
    stopped_on_token_sequence = None
    if isinstance(stop, dict):
        stopped_on_token_sequence = stop.get("stopped_on_token_sequence")
    if isinstance(stopped_on_token_sequence, list):
        stop_sequence = [int(token_id) for token_id in stopped_on_token_sequence]
        if stop_sequence and generated_token_ids[-len(stop_sequence) :] == stop_sequence:
            generated_without_stop_sequence_ids = generated_token_ids[: -len(stop_sequence)]
    full_without_stop_sequence_ids = prompt_token_ids + generated_without_stop_sequence_ids

    report["text_prompt"] = {
        "raw": raw_prompt,
        "rendered": rendered_prompt,
        "tokenizer_dir": args.tokenizer_dir,
        "apply_chat_template": args.apply_chat_template,
        "add_generation_prompt": args.add_generation_prompt if args.apply_chat_template else None,
        "add_special_tokens": args.add_special_tokens if not args.apply_chat_template else None,
        "token_count": len(prompt_token_ids),
        "target_prompt_tokens": args.target_prompt_tokens,
    }
    report["decoded_text"] = {
        "skip_special_tokens": args.skip_special_tokens,
        "prompt": tokenizer.decode(prompt_token_ids, skip_special_tokens=args.skip_special_tokens),
        "generated": tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=args.skip_special_tokens,
        ),
        "full": tokenizer.decode(full_token_ids, skip_special_tokens=args.skip_special_tokens),
        "generated_without_stop_sequence": tokenizer.decode(
            generated_without_stop_sequence_ids,
            skip_special_tokens=args.skip_special_tokens,
        ),
        "full_without_stop_sequence": tokenizer.decode(
            full_without_stop_sequence_ids,
            skip_special_tokens=args.skip_special_tokens,
        ),
    }
    report["tokenizer"] = {
        "class": type(tokenizer).__name__,
        "model_max_length": getattr(tokenizer, "model_max_length", None),
    }
    return report


def write_or_print_report(report: dict[str, Any], output_json: Path | None) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if output_json is None:
        print(payload)
        return
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(payload + "\n", encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    prompt = report.get("text_prompt", {})
    decode = report.get("decode", {})
    step_summary = decode.get("step_wall_summary", {}) if isinstance(decode, dict) else {}
    decoded = report.get("decoded_text", {})
    print(
        "prompt_tokens={prompt_tokens} generated_tokens={generated_tokens} "
        "verified={verified} decode_tps={decode_tps} skip2_tps={skip2_tps} last8_tps={last8_tps}".format(
            prompt_tokens=prompt.get("token_count"),
            generated_tokens=len(report.get("generated_token_ids", [])),
            verified=report.get("verified"),
            decode_tps=step_summary.get("all_step_tps"),
            skip2_tps=step_summary.get("warmup_skip_2_step_tps"),
            last8_tps=step_summary.get("last_8_step_tps"),
        ),
        file=sys.stderr,
    )
    generated = decoded.get("generated_without_stop_sequence") or decoded.get("generated")
    if generated:
        compact = " ".join(str(generated).split())
        print(f"generated_text={compact[:500]}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    if args.generated_tokens < 0:
        raise SystemExit("--generated-tokens must be non-negative")
    if args.chunk_bytes <= 0:
        raise SystemExit("--chunk-bytes must be positive")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")

    raw_prompt = read_prompt(args)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    rendered_prompt, token_ids = encode_prompt(tokenizer, args, raw_prompt)
    stop_token_ids = resolve_stop_token_ids(tokenizer, args)
    stop_token_sequences = resolve_stop_token_sequences(tokenizer, args)
    rendered_prompt, token_ids, token_adjustment = adjust_prompt_token_count(
        tokenizer,
        args,
        raw_prompt,
        rendered_prompt,
        token_ids,
    )
    if not token_ids:
        raise SystemExit("prompt encoded to zero tokens")
    if args.max_prompt_tokens is not None and len(token_ids) > args.max_prompt_tokens:
        raise SystemExit(
            f"prompt token count {len(token_ids)} exceeds --max-prompt-tokens {args.max_prompt_tokens}"
        )

    report = run_engine(args, token_ids, stop_token_ids, stop_token_sequences)
    report = enrich_report(report, tokenizer, args, raw_prompt, rendered_prompt, token_ids)
    report["text_prompt"].update(token_adjustment)
    write_or_print_report(report, args.output_json)
    if args.print_summary:
        print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
