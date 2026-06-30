#!/usr/bin/env python3
"""Run a small next-token loss smoke with selected modules quantized."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


DEFAULT_PROMPT = "Explain why activation-aware quantization can improve low-bit LLM inference."


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_logit_smoke_module():
    module_path = Path(__file__).with_name("run-aq-module-logit-smoke.py")
    spec = importlib.util.spec_from_file_location("run_aq_module_logit_smoke", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_selection_modules(path: Path | None) -> list[str]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    modules = payload.get("modules")
    if not isinstance(modules, list):
        raise SystemExit(f"selection JSON missing modules list: {path}")
    names: list[str] = []
    for item in modules:
        if not isinstance(item, dict) or not isinstance(item.get("module"), str):
            raise SystemExit(f"invalid module entry in selection JSON: {path}")
        names.append(str(item["module"]))
    return names


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def repeat_prompts_to_length(tokenizer, prompts: list[str], sequence_length: int) -> list[str]:
    repeated: list[str] = []
    for prompt in prompts:
        text = prompt
        for _ in range(128):
            token_count = len(tokenizer(text, add_special_tokens=True)["input_ids"])
            if token_count >= sequence_length:
                break
            text = f"{text}\n{prompt}"
        repeated.append(text)
    return repeated


def forward_sequence_loss(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    sequence_length: int,
    device: torch.device,
) -> dict[str, float | int]:
    batch = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=sequence_length)
    input_ids = batch["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        raise ValueError("prompt has fewer than two tokens after tokenization")
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    batch = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
    with torch.inference_mode():
        try:
            logits = model(**batch, use_cache=False).logits
        except TypeError as exc:
            if "use_cache" not in str(exc):
                raise
            logits = model(**batch).logits

    shift_logits = logits[:, :-1, :].to(torch.float32)
    shift_labels = input_ids[:, 1:]
    flat_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
    ).view_as(shift_labels)
    if attention_mask is None:
        token_count = int(shift_labels.numel())
        loss = flat_loss.mean()
    else:
        mask = attention_mask[:, 1:].to(torch.float32)
        token_count = int(mask.sum().item())
        if token_count < 1:
            raise ValueError("prompt has no unmasked next-token targets")
        loss = (flat_loss * mask).sum() / mask.sum()
    return {
        "loss": float(loss.detach().cpu()),
        "token_count": token_count,
    }


def parse_args(logit_smoke) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--activation-stats", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, default=None)
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument("--variant", choices=sorted(logit_smoke.VARIANTS), action="append", default=[])
    parser.add_argument(
        "--policy",
        type=logit_smoke.parse_policy,
        action="append",
        default=[],
        help="Mixed cumulative policy as NAME=family1,family2; listed families use --policy-high-variant.",
    )
    parser.add_argument("--policy-low-variant", choices=sorted(logit_smoke.VARIANTS), default="g16_weighted")
    parser.add_argument("--policy-high-variant", choices=sorted(logit_smoke.VARIANTS), default="g8_weighted")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--repeat-to-length", action="store_true")
    parser.add_argument("--max-codebook-elements", type=int, default=262144)
    parser.add_argument("--scale-window", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=64)
    parser.add_argument(
        "--max-original-weight-mib",
        type=int,
        default=4096,
        help="Refuse cumulative runs that would keep more original weights in CPU RAM.",
    )
    parser.add_argument("--run-id", default="aq-module-loss-smoke")
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    logit_smoke = load_logit_smoke_module()
    args = parse_args(logit_smoke)
    if not args.variant and not args.policy:
        raise SystemExit("at least one --variant or --policy is required")
    if args.max_original_weight_mib < 1:
        raise SystemExit("--max-original-weight-mib must be >= 1")

    selected_modules = unique_preserve_order(load_selection_modules(args.selection_json) + args.module)
    if not selected_modules:
        raise SystemExit("at least one --module or --selection-json module is required")

    torch.set_num_threads(args.torch_threads)
    args.model_dir = args.model_dir.expanduser().resolve()
    args.activation_stats = args.activation_stats.expanduser().resolve()
    args.selection_json = args.selection_json.expanduser().resolve() if args.selection_json else None
    args.prompt_file = args.prompt_file.expanduser().resolve() if args.prompt_file else None

    sampler = logit_smoke.load_sampler_module()
    stats = logit_smoke.load_activation_stats(args.activation_stats)
    tokenizer, model = logit_smoke.load_model_and_tokenizer(args)
    device = next(model.parameters()).device
    prompts = logit_smoke.load_prompts(args.prompt_file, args.prompt, args.max_prompts)
    if args.repeat_to_length:
        prompts = repeat_prompts_to_length(tokenizer, prompts, args.sequence_length)
    policies = logit_smoke.make_policy_specs(args)

    reference_losses = [
        forward_sequence_loss(model, tokenizer, prompt, args.sequence_length, device)
        for prompt in prompts
    ]

    module_entries, total_original_bytes = logit_smoke.collect_cumulative_module_entries(
        sampler,
        model,
        selected_modules,
        stats,
        args.max_original_weight_mib,
    )
    logit_smoke.validate_policies_for_modules(policies, module_entries)
    module_names = [str(entry["name"]) for entry in module_entries]
    module_families = {str(entry["name"]): str(entry["family"]) for entry in module_entries}
    run_specs = [(variant_id, logit_smoke.VARIANTS[variant_id], None) for variant_id in args.variant]
    run_specs.extend((policy.policy_id, None, policy) for policy in policies)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:
        for spec_id, uniform_variant, policy in run_specs:
            if uniform_variant is None:
                if policy is None:
                    raise RuntimeError(f"invalid run spec: {spec_id}")
                row_variant = logit_smoke.policy_variant_payload(policy)
                row_policy = logit_smoke.policy_payload(policy, module_entries)
            else:
                row_variant = logit_smoke.variant_payload(uniform_variant)
                row_policy = None
            try:
                for entry in module_entries:
                    original = entry["original"]
                    module = entry["module"]
                    activation = entry["activation"]
                    assert isinstance(original, torch.Tensor)
                    assert isinstance(module, torch.nn.Linear)
                    assert isinstance(activation, torch.Tensor)
                    if uniform_variant is None:
                        if policy is None:
                            raise RuntimeError(f"invalid run spec: {spec_id}")
                        variant = logit_smoke.variant_for_policy(policy, str(entry["family"]))
                    else:
                        variant = uniform_variant
                    quantized = logit_smoke.quantize_weight(
                        sampler,
                        original,
                        activation,
                        variant,
                        args.max_codebook_elements,
                        args.scale_window,
                        args.seed,
                    ).to(device=module.weight.device, dtype=module.weight.dtype)
                    module.weight.data.copy_(quantized)

                rows = []
                for prompt_index, (prompt, reference) in enumerate(
                    zip(prompts, reference_losses, strict=True)
                ):
                    candidate = forward_sequence_loss(model, tokenizer, prompt, args.sequence_length, device)
                    reference_loss = float(reference["loss"])
                    candidate_loss = float(candidate["loss"])
                    rows.append(
                        {
                            "schema_version": "aq-module-loss-smoke-v0.1",
                            "run_id": args.run_id,
                            "timestamp_utc": utc_now(),
                            "status": "ok",
                            "model_dir": str(args.model_dir),
                            "activation_stats": str(args.activation_stats),
                            "selection_json": str(args.selection_json) if args.selection_json else None,
                            "module_scope": "cumulative",
                            "modules": module_names,
                            "module_families": module_families,
                            "total_original_weight_bytes": total_original_bytes,
                            "variant": row_variant,
                            "policy": row_policy,
                            "prompt": prompt,
                            "prompt_index": prompt_index,
                            "prompt_count": len(prompts),
                            "prompt_file": str(args.prompt_file) if args.prompt_file else None,
                            "sequence_length": args.sequence_length,
                            "repeat_to_length": args.repeat_to_length,
                            "metrics": {
                                "reference_loss": reference_loss,
                                "candidate_loss": candidate_loss,
                                "loss_delta": candidate_loss - reference_loss,
                                "relative_loss_delta": (candidate_loss - reference_loss)
                                / max(abs(reference_loss), 1e-30),
                                "token_count": int(candidate["token_count"]),
                            },
                            "notes": args.note,
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - keep benchmark rows self-describing.
                rows = [
                    {
                        "schema_version": "aq-module-loss-smoke-v0.1",
                        "run_id": args.run_id,
                        "timestamp_utc": utc_now(),
                        "status": "failed",
                        "model_dir": str(args.model_dir),
                        "activation_stats": str(args.activation_stats),
                        "selection_json": str(args.selection_json) if args.selection_json else None,
                        "module_scope": "cumulative",
                        "modules": module_names,
                        "module_families": module_families,
                        "total_original_weight_bytes": total_original_bytes,
                        "variant": row_variant,
                        "policy": row_policy,
                        "prompt": None,
                        "prompt_count": len(prompts),
                        "prompt_file": str(args.prompt_file) if args.prompt_file else None,
                        "sequence_length": args.sequence_length,
                        "repeat_to_length": args.repeat_to_length,
                        "metrics": {},
                        "notes": args.note,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                ]
            finally:
                logit_smoke.restore_cumulative_modules(module_entries)
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
