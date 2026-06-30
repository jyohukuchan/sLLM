#!/usr/bin/env python3
"""Collect compact activation statistics for aq weighted-error evaluation.

The tool records per-module input second moments for selected Linear modules.
It stores reductions only; raw activations are never retained.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import save_file


DEFAULT_PROMPTS = [
    "Explain the difference between prefill and decode in large language model inference.",
    "Write a short technical note about quantizing transformer weight matrices.",
    "Summarize why activation outliers matter for post-training quantization.",
    "List practical constraints for running a language model on a consumer GPU.",
]


@dataclass
class ActivationAccumulator:
    sum_sq: torch.Tensor
    sum_abs: torch.Tensor
    max_abs: torch.Tensor
    count: int = 0

    @classmethod
    def create(cls, features: int) -> "ActivationAccumulator":
        return cls(
            sum_sq=torch.zeros(features, dtype=torch.float64),
            sum_abs=torch.zeros(features, dtype=torch.float64),
            max_abs=torch.zeros(features, dtype=torch.float32),
        )

    def add(self, values: torch.Tensor) -> None:
        flat = values.detach().reshape(-1, values.shape[-1]).to(torch.float32)
        self.sum_sq += flat.square().sum(dim=0).to(torch.float64).cpu()
        self.sum_abs += flat.abs().sum(dim=0).to(torch.float64).cpu()
        self.max_abs = torch.maximum(self.max_abs, flat.abs().amax(dim=0).cpu())
        self.count += int(flat.shape[0])

    def second_moment(self) -> torch.Tensor:
        if self.count == 0:
            return torch.zeros_like(self.max_abs)
        return (self.sum_sq / float(self.count)).to(torch.float32)

    def mean_abs(self) -> torch.Tensor:
        if self.count == 0:
            return torch.zeros_like(self.max_abs)
        return (self.sum_abs / float(self.count)).to(torch.float32)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def iter_prompts(path: Path | None) -> Iterable[str]:
    if path is None:
        yield from DEFAULT_PROMPTS
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        prompt = line.strip()
        if prompt:
            yield prompt


def dtype_from_arg(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unknown dtype: {name}")


def load_transformers_model(args: argparse.Namespace):
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    model_cls = AutoModel if args.model_class == "auto_model" else AutoModelForCausalLM
    dtype = dtype_from_arg(args.dtype)
    kwargs = {"trust_remote_code": args.trust_remote_code}
    kwargs["dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    try:
        model = model_cls.from_pretrained(
            args.model_dir,
            local_files_only=True,
            **kwargs,
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = model_cls.from_pretrained(
            args.model_dir,
            local_files_only=True,
            **kwargs,
        )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.device != "auto":
        model.to(torch.device(args.device))
    model.eval()
    return tokenizer, model


def register_hooks(
    model: torch.nn.Module,
    module_pattern: re.Pattern[str],
    max_modules: int | None,
) -> tuple[dict[str, ActivationAccumulator], list[torch.utils.hooks.RemovableHandle]]:
    accumulators: dict[str, ActivationAccumulator] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str, in_features: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                return
            values = inputs[0]
            if not torch.is_tensor(values) or not values.is_floating_point():
                return
            if int(values.shape[-1]) != in_features:
                return
            accumulators[name].add(values)

        return hook

    for name, module in model.named_modules():
        if max_modules is not None and len(accumulators) >= max_modules:
            break
        if not isinstance(module, torch.nn.Linear):
            continue
        if not module_pattern.search(name):
            continue
        accumulators[name] = ActivationAccumulator.create(int(module.in_features))
        handles.append(module.register_forward_pre_hook(make_hook(name, int(module.in_features))))

    if not accumulators:
        raise SystemExit("no modules matched --module-pattern")
    return accumulators, handles


def encode_prompt(tokenizer, prompt: str, sequence_length: int, repeat_to_length: bool) -> dict[str, torch.Tensor]:
    if repeat_to_length and len(prompt) > 0:
        prompt = (prompt + "\n") * max(1, sequence_length // max(1, len(prompt.split())) // 2)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=sequence_length,
    )
    return {key: value for key, value in encoded.items() if torch.is_tensor(value)}


def tensor_dict_for_output(accumulators: dict[str, ActivationAccumulator]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name, accumulator in sorted(accumulators.items()):
        tensors[name] = accumulator.second_moment().contiguous()
        tensors[f"{name}.mean_abs"] = accumulator.mean_abs().contiguous()
        tensors[f"{name}.max_abs"] = accumulator.max_abs.contiguous()
    return tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--repeat-to-length", action="store_true")
    parser.add_argument(
        "--module-pattern",
        default=r"(self_attn|linear_attn|mlp).*(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|gate_proj|up_proj|down_proj)$",
    )
    parser.add_argument("--max-modules", type=int, default=None)
    parser.add_argument("--model-class", choices=("auto_model", "causal_lm"), default="auto_model")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-id", default="activation-stats")
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_samples < 1:
        raise SystemExit("--max-samples must be >= 1")
    if args.sequence_length < 1:
        raise SystemExit("--sequence-length must be >= 1")
    if args.max_modules is not None and args.max_modules < 1:
        raise SystemExit("--max-modules must be >= 1")

    args.model_dir = args.model_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_transformers_model(args)
    accumulators, handles = register_hooks(model, re.compile(args.module_pattern), args.max_modules)
    device = next(model.parameters()).device
    samples_seen = 0
    tokens_seen = 0

    try:
        with torch.inference_mode():
            for prompt in iter_prompts(args.prompt_file):
                if samples_seen >= args.max_samples:
                    break
                batch = encode_prompt(tokenizer, prompt, args.sequence_length, args.repeat_to_length)
                batch = {key: value.to(device) for key, value in batch.items()}
                if "input_ids" in batch:
                    tokens_seen += int(batch["input_ids"].numel())
                try:
                    model(**batch, use_cache=False)
                except TypeError as exc:
                    if "use_cache" not in str(exc):
                        raise
                    model(**batch)
                samples_seen += 1
    finally:
        for handle in handles:
            handle.remove()

    save_file(tensor_dict_for_output(accumulators), str(args.output_dir / "activation_second_moments.safetensors"))
    metadata = {
        "schema_version": "aq-activation-stats-v0.1",
        "run_id": args.run_id,
        "timestamp_utc": utc_now(),
        "model_dir": str(args.model_dir),
        "prompt_file": str(args.prompt_file.expanduser().resolve()) if args.prompt_file else None,
        "max_samples": args.max_samples,
        "samples_seen": samples_seen,
        "tokens_seen": tokens_seen,
        "sequence_length": args.sequence_length,
        "repeat_to_length": args.repeat_to_length,
        "module_pattern": args.module_pattern,
        "module_count": len(accumulators),
        "model_class": args.model_class,
        "dtype": args.dtype,
        "device": str(device),
        "notes": args.note,
        "modules": {
            name: {
                "input_features": int(accumulator.max_abs.numel()),
                "activation_count": accumulator.count,
            }
            for name, accumulator in sorted(accumulators.items())
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
