#!/usr/bin/env python3
"""Generate the lightweight promotion suite through the SQ8_1 W8A8 CPU reference.

This is deliberately an investigation-only harness.  It reuses the frozen
SQ8_1 K=32 quantizer from ``measure-sq8_1-w8a8-full-model-gate.py`` and
materializes the same reconstructed weights inside an isolated Hugging Face
process.  It does not create an artifact, worker, release, manifest, or
promotion path.  In particular, it is not a substitute for an R9700 served
model measurement.

The baseline and W8A8 candidate both greedily generate the fixed lightweight
promotion prompt suite.  Unlike the numerical gate, the resulting text and
token IDs are retained as evidence so that a large logit delta can be judged
by its observed effect on actual generations.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "docs" / "plans" / "lightweight-promotion-prompt-suite-v0.1.json"
DEFAULT_PATTERN = (
    r"(self_attn|linear_attn|mlp).*"
    r"(q_proj|k_proj|v_proj|o_proj|in_proj(_qkv|_qkvz|_ba|_[abz])?|"
    r"out_proj|gate_proj|up_proj|down_proj)$"
)


def load_module(filename: str, module_name: str) -> ModuleType:
    path = ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_module("measure-sq8_1-w8a8-full-model-gate.py", "sq8_1_w8a8_gate")
PROMOTION = load_module("lightweight_promotion.py", "sq8_1_lightweight_promotion")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def select_suite(suite: tuple[Any, ...], case_ids: tuple[str, ...]) -> tuple[Any, ...]:
    if not case_ids:
        return suite
    by_id = {case.case_id: case for case in suite}
    missing = sorted(set(case_ids) - set(by_id))
    if missing:
        raise ValueError(f"unknown prompt-suite case IDs: {', '.join(missing)}")
    return tuple(case for case in suite if case.case_id in set(case_ids))


def render_prompt(tokenizer: Any, messages: tuple[dict[str, str], ...]) -> tuple[str, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        prompt = tokenizer.apply_chat_template(list(messages), **kwargs)
    except TypeError:
        # Older template processors may not expose ``enable_thinking``.  The
        # fallback is recorded rather than silently changing the contract.
        kwargs.pop("enable_thinking")
        prompt = tokenizer.apply_chat_template(list(messages), **kwargs)
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError("chat template did not return non-empty text")
    return prompt, kwargs


def generate_case(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    case: Any,
    max_new_tokens: int,
    variant: str,
) -> dict[str, Any]:
    prompt, template_kwargs = render_prompt(tokenizer, case.messages)
    encoded = tokenizer(prompt, return_tensors="pt")
    if "input_ids" not in encoded or not torch.is_tensor(encoded["input_ids"]):
        raise RuntimeError(f"{case.case_id}: tokenizer did not return input_ids")
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None and not torch.is_tensor(attention_mask):
        raise RuntimeError(f"{case.case_id}: tokenizer returned invalid attention_mask")
    prompt_tokens = int(input_ids.shape[1])
    if prompt_tokens < 1:
        raise RuntimeError(f"{case.case_id}: empty encoded prompt")
    generation_kwargs: dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "return_dict_in_generate": False,
    }
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = int(tokenizer.pad_token_id)
    started = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
    elapsed = time.monotonic() - started
    if not torch.is_tensor(generated) or generated.ndim != 2 or generated.shape[0] != 1:
        raise RuntimeError(f"{case.case_id}: generation returned an unexpected shape")
    token_ids = [int(token) for token in generated[0, prompt_tokens:].tolist()]
    content = tokenizer.decode(token_ids, skip_special_tokens=True)
    record = {
        "schema_version": "ullm.sq8_1-w8a8-lightweight-generation.v0.1",
        "case_id": case.case_id,
        "category": case.category,
        "variant": variant,
        "request": {
            "messages": list(case.messages),
            "max_completion_tokens": max_new_tokens,
            "seed": 0,
            "sampling": {"do_sample": False},
        },
        "chat_template": {
            "rendered_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "rendered_prompt_token_count": prompt_tokens,
            "arguments": template_kwargs,
        },
        "generation": {
            "generated_token_ids": token_ids,
            "generated_token_count": len(token_ids),
            "elapsed_seconds": round(elapsed, 6),
            "stopped_before_limit": len(token_ids) < max_new_tokens,
        },
        "content": content,
        "character_count": len(content),
        "analysis": PROMOTION.analyze_text(content, case),
    }
    print(
        json.dumps(
            {
                "event": "generation_complete",
                "case_id": case.case_id,
                "variant": variant,
                "generated_tokens": len(token_ids),
                "elapsed_seconds": record["generation"]["elapsed_seconds"],
                "blocking": record["analysis"]["blocking"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return record


def selected_linear_names(model: torch.nn.Module, pattern: str) -> tuple[str, ...]:
    expression = re.compile(pattern)
    names = tuple(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and expression.search(name)
    )
    if len(names) != 248:
        raise RuntimeError(f"expected exactly 248 primary SQ8_1 projections, found {len(names)}")
    return names


def materialize_quantized_weights(model: torch.nn.Module, names: tuple[str, ...]) -> dict[str, Any]:
    """Replace only the isolated process's selected weights with SQ8_1 values.

    The original numerical gate reconstructs a static quantized weight for
    every F.linear call.  Replacing the in-memory parameter after the baseline
    pass is algebraically identical at that boundary and avoids retaining a
    second 9B-parameter copy merely to generate text.  No model file is
    modified, and the process exits after evidence has been written.
    """

    modules = dict(model.named_modules())
    aggregate = GATE.WeightAccumulator()
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, name in enumerate(names, start=1):
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError(f"selected path is not nn.Linear: {name}")
        source = module.weight.detach().to(torch.float32).contiguous()
        quantized = GATE.quantize_sq8_1(source)
        reconstructed = GATE.reconstruct_blocks(quantized)
        aggregate.add(source, quantized, reconstructed)
        with torch.no_grad():
            module.weight.copy_(reconstructed.to(dtype=module.weight.dtype))
        rows.append(
            {
                "name": name,
                "shape": [int(value) for value in module.weight.shape],
                "clipping_count": quantized.clipping_count,
                "edge_code_count": quantized.edge_code_count,
                "zero_source_block_count": quantized.zero_source_block_count,
            }
        )
        del source
        del quantized
        del reconstructed
        print(
            json.dumps(
                {"event": "weight_materialized", "completed": index, "total": len(names), "tensor": name},
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        "projection_count": len(names),
        "weight_quantization": aggregate.as_dict(),
        "per_tensor": rows,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


@contextlib.contextmanager
def patched_w8a8_activations(model: torch.nn.Module, names: tuple[str, ...]) -> Iterator[None]:
    """Apply the gate's dynamic K=32 activation reconstruction to 248 inputs."""

    modules = dict(model.named_modules())
    originals: list[tuple[torch.nn.Linear, Any]] = []
    try:
        for name in names:
            module = modules.get(name)
            if not isinstance(module, torch.nn.Linear):
                raise RuntimeError(f"selected path is not nn.Linear: {name}")
            original = module.forward

            def forward(
                current: torch.nn.Linear,
                input_value: torch.Tensor,
                *,
                _original: Any = original,
            ) -> torch.Tensor:
                if not torch.is_tensor(input_value) or not input_value.is_floating_point():
                    return _original(input_value)
                source = input_value.detach().to(torch.float32)
                if source.shape[-1] != int(current.weight.shape[1]):
                    raise ValueError(
                        f"SQ8_1 W8A8 activation width {source.shape[-1]} differs from "
                        f"weight width {current.weight.shape[1]}"
                    )
                shape = source.shape
                flat = source.reshape(-1, shape[-1]).contiguous()
                quantized = GATE.quantize_sq8_1(flat)
                reconstructed = GATE.reconstruct_blocks(quantized).reshape(shape)
                return F.linear(
                    reconstructed.to(dtype=input_value.dtype),
                    current.weight,
                    current.bias,
                )

            originals.append((module, original))
            module.forward = forward.__get__(module, torch.nn.Linear)
        yield
    finally:
        for module, original in reversed(originals):
            module.forward = original


def generate_suite(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    suite: tuple[Any, ...],
    max_tokens_override: int | None,
    variant: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for case in suite:
        maximum = max_tokens_override if max_tokens_override is not None else case.max_completion_tokens
        record = generate_case(
            model=model,
            tokenizer=tokenizer,
            case=case,
            max_new_tokens=maximum,
            variant=variant,
        )
        write_json(output_dir / f"{case.case_id}.json", record)
        records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--model-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.prompt_suite = args.prompt_suite.expanduser().resolve()
    if not args.model_dir.is_dir():
        raise SystemExit(f"missing model directory: {args.model_dir}")
    if not args.prompt_suite.is_file():
        raise SystemExit(f"missing prompt suite: {args.prompt_suite}")
    if args.output_dir.exists():
        raise SystemExit(f"refusing to use an existing output directory: {args.output_dir}")
    if args.torch_threads < 1 or args.torch_interop_threads < 1:
        raise SystemExit("thread counts must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")
    if torch.cuda.is_available():
        raise SystemExit("CPU-only generator refuses an available CUDA/HIP torch device")

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    torch.manual_seed(0)
    suite = select_suite(PROMOTION.load_suite(args.prompt_suite), tuple(args.case_id))
    if not suite:
        raise SystemExit("selected prompt suite is empty")
    args.output_dir.mkdir(parents=True, mode=0o750)
    started = utc_now()
    tool_path = Path(__file__).resolve()
    write_json(
        args.output_dir / "run-manifest.json",
        {
            "schema_version": "ullm.sq8_1-w8a8-lightweight-generation-run.v0.1",
            "started_at": started,
            "scope": {
                "candidate": "SQ8_1 explicit W8A8 CPU fake-quant reference",
                "not_a_served_candidate": True,
                "not_an_r9700_throughput_measurement": True,
                "weight_rule": "K=32 signed int8, RNE code, upward-rounded FP16 scale",
                "activation_rule": "dynamic K=32 signed int8 at selected Linear inputs",
                "lm_head": "unmodified",
            },
            "inputs": {
                "model_dir": str(args.model_dir),
                "prompt_suite": str(args.prompt_suite),
                "prompt_suite_sha256": sha256_file(args.prompt_suite),
                "tool_sha256": sha256_file(tool_path),
                "quantizer_tool": str(ROOT / "tools" / "measure-sq8_1-w8a8-full-model-gate.py"),
                "quantizer_tool_sha256": sha256_file(ROOT / "tools" / "measure-sq8_1-w8a8-full-model-gate.py"),
                "model_dtype": args.model_dtype,
                "torch_threads": args.torch_threads,
                "torch_interop_threads": args.torch_interop_threads,
                "case_ids": [case.case_id for case in suite],
                "max_new_tokens_override": args.max_new_tokens,
                "fixed_suite_complete": not args.case_id and args.max_new_tokens is None,
            },
        },
    )
    model_args = SimpleNamespace(
        model_dir=args.model_dir,
        model_class="causal_lm",
        dtype=args.model_dtype,
        trust_remote_code=args.trust_remote_code,
        device="cpu",
    )
    tokenizer, model = GATE.COLLECTOR.load_transformers_model(model_args)
    device = next(model.parameters()).device
    if device.type != "cpu" or model.training:
        raise RuntimeError(f"expected an eval CPU model, got device={device}, training={model.training}")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    names = selected_linear_names(model, DEFAULT_PATTERN)
    baseline = generate_suite(
        model=model,
        tokenizer=tokenizer,
        suite=suite,
        max_tokens_override=args.max_new_tokens,
        variant="source_reference",
        output_dir=args.output_dir / "source-reference",
    )
    weight_summary = materialize_quantized_weights(model, names)
    write_json(args.output_dir / "weight-materialization.json", weight_summary)
    with patched_w8a8_activations(model, names):
        candidate = generate_suite(
            model=model,
            tokenizer=tokenizer,
            suite=suite,
            max_tokens_override=args.max_new_tokens,
            variant="sq8_1_w8a8",
            output_dir=args.output_dir / "sq8_1-w8a8",
        )
    comparison = PROMOTION.compare_suites(suite, baseline, candidate)
    write_json(args.output_dir / "comparison.json", comparison)
    PROMOTION.write_comparison_markdown(
        args.output_dir / "comparison.md",
        suite,
        baseline,
        candidate,
        comparison,
    )
    write_json(
        args.output_dir / "run-complete.json",
        {
            "started_at": started,
            "completed_at": utc_now(),
            "case_count": len(suite),
            "comparison_passed": comparison["passed"],
            "blocking_findings": comparison["blocking_findings"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
