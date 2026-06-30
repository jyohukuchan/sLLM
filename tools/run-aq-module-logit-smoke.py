#!/usr/bin/env python3
"""Run a small logit-difference smoke with selected modules quantized."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from safetensors import safe_open


DEFAULT_PROMPT = "Explain why activation-aware quantization can improve low-bit LLM inference."


@dataclass(frozen=True)
class Variant:
    variant_id: str
    candidate_id: str
    weighted_scale_search: bool
    weighted_codebook: bool


VARIANTS = {
    "g16_unweighted": Variant("g16_unweighted", "aq4_e4m3_g16_ts_flloyd16", False, False),
    "g16_weighted": Variant("g16_weighted", "aq4_e4m3_g16_ts_flloyd16", True, True),
    "g8_weighted": Variant("g8_weighted", "aq4_e4m3_g8_ts_flloyd16", True, True),
}


@dataclass(frozen=True)
class Policy:
    policy_id: str
    low_variant: Variant
    high_variant: Variant
    high_families: frozenset[str]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_sampler_module():
    module_path = Path(__file__).with_name("run-aq-tensor-sample.py")
    spec = importlib.util.spec_from_file_location("run_aq_tensor_sample", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_activation_stats(path: Path) -> dict[str, torch.Tensor]:
    if path.is_dir():
        path = path / "activation_second_moments.safetensors"
    stats: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            stats[key] = handle.get_tensor(key).to(torch.float32).flatten().contiguous()
    return stats


def load_prompts(path: Path | None, fallback: str, max_prompts: int | None) -> list[str]:
    if path is None:
        prompts = [fallback]
    else:
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise SystemExit("no prompts to evaluate")
    return prompts


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


def load_model_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    kwargs = {"trust_remote_code": True, "dtype": dtype_from_arg(args.dtype)}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, **kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(torch.device(args.device))
    model.eval()
    return tokenizer, model


def get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"module not found: {name}")
    module = modules[name]
    if not isinstance(module, torch.nn.Linear):
        raise TypeError(f"module is not torch.nn.Linear: {name} -> {type(module).__name__}")
    return module


def module_to_tensor_name(module_name: str) -> str:
    return f"model.{module_name}.weight"


def family_for_module(sampler, module_name: str) -> str:
    return sampler.family_for_tensor(module_to_tensor_name(module_name))


def parse_policy(value: str) -> tuple[str, frozenset[str]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("policy must be NAME=family1,family2")
    name, family_list = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("policy name must not be empty")
    families = frozenset(item.strip() for item in family_list.split(",") if item.strip())
    return name, families


def make_policy_specs(args: argparse.Namespace) -> list[Policy]:
    low = VARIANTS[args.policy_low_variant]
    high = VARIANTS[args.policy_high_variant]
    return [Policy(name, low, high, families) for name, families in args.policy]


def variant_for_policy(policy: Policy, family: str) -> Variant:
    return policy.high_variant if family in policy.high_families else policy.low_variant


def variant_payload(variant: Variant) -> dict[str, object]:
    return {
        "variant_id": variant.variant_id,
        "candidate_id": variant.candidate_id,
        "weighted_scale_search": variant.weighted_scale_search,
        "weighted_codebook": variant.weighted_codebook,
    }


def policy_payload(policy: Policy, module_entries: Iterable[dict[str, object]]) -> dict[str, object]:
    module_variants = []
    for entry in module_entries:
        family = str(entry["family"])
        variant = variant_for_policy(policy, family)
        module_variants.append(
            {
                "module": str(entry["name"]),
                "family": family,
                "variant": variant_payload(variant),
            }
        )
    return {
        "policy_id": policy.policy_id,
        "low_variant": variant_payload(policy.low_variant),
        "high_variant": variant_payload(policy.high_variant),
        "high_families": sorted(policy.high_families),
        "module_variants": module_variants,
    }


def policy_variant_payload(policy: Policy) -> dict[str, object]:
    return {
        "variant_id": f"policy:{policy.policy_id}",
        "candidate_id": "mixed",
        "weighted_scale_search": "mixed",
        "weighted_codebook": "mixed",
    }


def validate_policies_for_modules(policies: list[Policy], module_entries: Iterable[dict[str, object]]) -> None:
    if not policies:
        return
    selected_families = {str(entry["family"]) for entry in module_entries}
    for policy in policies:
        matched = selected_families & policy.high_families
        if not matched:
            raise SystemExit(
                f"policy {policy.policy_id!r} high families do not match selected module families; "
                f"selected={','.join(sorted(selected_families))} "
                f"high={','.join(sorted(policy.high_families))}"
            )


def collect_cumulative_module_entries(
    sampler,
    model: torch.nn.Module,
    module_names: list[str],
    stats: dict[str, torch.Tensor],
    max_original_weight_mib: int,
) -> tuple[list[dict[str, object]], int]:
    entries: list[dict[str, object]] = []
    total_original_bytes = 0
    for module_name in module_names:
        module = get_module(model, module_name)
        original_bytes = int(module.weight.numel() * module.weight.element_size())
        total_original_bytes += original_bytes
        entries.append(
            {
                "name": module_name,
                "module": module,
                "family": family_for_module(sampler, module_name),
                "original_weight_bytes": original_bytes,
                "activation": activation_for_module(module_name, int(module.in_features), stats),
            }
        )

    limit_bytes = int(max_original_weight_mib) * 1024 * 1024
    if total_original_bytes > limit_bytes:
        mib = total_original_bytes / (1024 * 1024)
        raise SystemExit(
            f"selected cumulative modules need {mib:.1f} MiB of original-weight storage; "
            f"increase --max-original-weight-mib if this is intentional"
        )

    for entry in entries:
        module = entry["module"]
        assert isinstance(module, torch.nn.Linear)
        entry["original"] = module.weight.detach().to("cpu").clone()
    return entries, total_original_bytes


def restore_cumulative_modules(module_entries: Iterable[dict[str, object]]) -> None:
    for entry in module_entries:
        module = entry["module"]
        original = entry["original"]
        assert isinstance(module, torch.nn.Linear)
        assert isinstance(original, torch.Tensor)
        module.weight.data.copy_(original.to(device=module.weight.device, dtype=module.weight.dtype))


def activation_for_module(
    module_name: str,
    in_features: int,
    stats: dict[str, torch.Tensor],
) -> torch.Tensor:
    module_without_model = module_name.removeprefix("model.")
    language_model_name = f"language_model.{module_without_model}"
    candidates = (
        module_name,
        f"model.{module_name}",
        module_without_model,
        language_model_name,
        f"{module_name}.input_second_moment",
        f"model.{module_name}.input_second_moment",
        f"{module_without_model}.input_second_moment",
        f"{language_model_name}.input_second_moment",
    )
    for key in candidates:
        values = stats.get(key)
        if values is None:
            continue
        if values.numel() != in_features:
            raise ValueError(f"activation stats for {module_name} have {values.numel()} values, expected {in_features}")
        return values
    raise KeyError(f"activation stats missing for {module_name}")


def quantize_weight(
    sampler,
    weight: torch.Tensor,
    activation_second_moment: torch.Tensor,
    variant: Variant,
    max_codebook_elements: int,
    scale_window: int,
    seed: int,
) -> torch.Tensor:
    candidate = next(item for item in sampler.ROUND1_CANDIDATES if item.candidate_id == variant.candidate_id)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    weight_cpu = weight.detach().to("cpu", dtype=torch.float32)

    sample_groups, sample_columns = sampler.sample_groups_with_columns(
        weight_cpu,
        candidate.group_size,
        max_codebook_elements,
        generator,
    )
    sample_weights = None
    if variant.weighted_codebook or variant.weighted_scale_search:
        if sample_columns is None:
            raise ValueError("weighted quantization requires a 2D weight")
        sample_weights = activation_second_moment.index_select(0, sample_columns.flatten()).view_as(sample_groups)
    codebook_weights = sample_weights if variant.weighted_codebook else None
    codebook = sampler.codebook_from_groups(sample_groups, candidate.codebook_mode, codebook_weights)
    scales = sampler.scale_values(candidate.scale_format)
    tensor_scale = sampler.choose_tensor_scale(sample_groups, candidate, scales, codebook)

    flat = weight_cpu.flatten()
    usable = (flat.numel() // candidate.group_size) * candidate.group_size
    output = flat.clone()
    groups = flat[:usable].view(-1, candidate.group_size)
    cols = int(weight_cpu.shape[1])
    offsets = torch.arange(candidate.group_size, dtype=torch.long)

    max_code = codebook.abs().max().clamp_min(1e-12)
    for start in range(0, groups.shape[0], 65536):
        end = min(start + 65536, groups.shape[0])
        chunk = groups[start:end]
        group_ids = torch.arange(start, end, dtype=torch.long)
        columns = (group_ids[:, None] * candidate.group_size + offsets[None, :]) % cols
        group_weights = activation_second_moment.index_select(0, columns.flatten()).view_as(chunk)

        scaled = chunk / tensor_scale
        target_scale = scaled.abs().amax(dim=1) / max_code
        center = sampler.nearest_scale_indices(target_scale, scales)
        best_error = torch.full((chunk.shape[0],), torch.inf, dtype=torch.float32)
        best_recon = torch.zeros_like(chunk)

        for offset in range(-scale_window, scale_window + 1):
            idx = (center + offset).clamp(0, scales.numel() - 1)
            group_scale = scales.index_select(0, idx)
            normalized = scaled / group_scale[:, None]
            nearest = (normalized[:, :, None] - codebook[None, None, :]).abs().argmin(dim=2)
            quantized = codebook.index_select(0, nearest.flatten()).view_as(chunk)
            recon = quantized * group_scale[:, None] * tensor_scale
            square_error = (chunk - recon).square()
            if variant.weighted_scale_search:
                error = (square_error * group_weights).sum(dim=1)
            else:
                error = square_error.sum(dim=1)
            mask = error < best_error
            best_error = torch.where(mask, error, best_error)
            best_recon = torch.where(mask[:, None], recon, best_recon)
        output[:usable].view(-1, candidate.group_size)[start:end] = best_recon
    return output.view_as(weight_cpu).to(dtype=weight.dtype)


def forward_last_logits(model, tokenizer, prompt: str, sequence_length: int, device: torch.device) -> torch.Tensor:
    batch = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=sequence_length)
    batch = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
    with torch.inference_mode():
        try:
            logits = model(**batch, use_cache=False).logits
        except TypeError as exc:
            if "use_cache" not in str(exc):
                raise
            logits = model(**batch).logits
    return logits[:, -1, :].detach().to(torch.float32).cpu()


def logit_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int | bool]:
    diff = reference - candidate
    ref_prob = F.softmax(reference, dim=-1)
    cand_log_prob = F.log_softmax(candidate, dim=-1)
    kl = F.kl_div(cand_log_prob, ref_prob, reduction="batchmean")
    top_ref = torch.topk(reference, k=10, dim=-1).indices[0]
    top_candidate = torch.topk(candidate, k=10, dim=-1).indices[0]
    return {
        "mse": float(diff.square().mean()),
        "relative_mse": float(diff.square().mean() / reference.square().mean().clamp_min(1e-30)),
        "mean_abs_error": float(diff.abs().mean()),
        "max_abs_error": float(diff.abs().max()),
        "cosine_similarity": float(F.cosine_similarity(reference, candidate, dim=-1).mean()),
        "kl_ref_candidate": float(kl),
        "top1_match": bool(int(top_ref[0]) == int(top_candidate[0])),
        "top10_overlap": int(len(set(int(x) for x in top_ref) & set(int(x) for x in top_candidate))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--activation-stats", type=Path, required=True)
    parser.add_argument("--module", action="append", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), action="append", default=[])
    parser.add_argument(
        "--policy",
        type=parse_policy,
        action="append",
        default=[],
        help="Mixed cumulative policy as NAME=family1,family2; listed families use --policy-high-variant.",
    )
    parser.add_argument("--policy-low-variant", choices=sorted(VARIANTS), default="g16_weighted")
    parser.add_argument("--policy-high-variant", choices=sorted(VARIANTS), default="g8_weighted")
    parser.add_argument("--cumulative", action="store_true", help="Quantize all selected modules together per variant.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=64)
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
    parser.add_argument("--run-id", default="aq-module-logit-smoke")
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.variant and not args.policy:
        raise SystemExit("at least one --variant or --policy is required")
    if args.policy and not args.cumulative:
        raise SystemExit("--policy currently requires --cumulative")
    if args.max_original_weight_mib < 1:
        raise SystemExit("--max-original-weight-mib must be >= 1")
    torch.set_num_threads(args.torch_threads)
    args.model_dir = args.model_dir.expanduser().resolve()
    args.activation_stats = args.activation_stats.expanduser().resolve()
    args.prompt_file = args.prompt_file.expanduser().resolve() if args.prompt_file else None
    sampler = load_sampler_module()
    stats = load_activation_stats(args.activation_stats)
    tokenizer, model = load_model_and_tokenizer(args)
    device = next(model.parameters()).device
    prompts = load_prompts(args.prompt_file, args.prompt, args.max_prompts)
    policies = make_policy_specs(args)
    reference_logits = [
        forward_last_logits(model, tokenizer, prompt, args.sequence_length, device)
        for prompt in prompts
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:
        if args.cumulative:
            module_entries, total_original_bytes = collect_cumulative_module_entries(
                sampler,
                model,
                args.module,
                stats,
                args.max_original_weight_mib,
            )
            validate_policies_for_modules(policies, module_entries)
            run_specs: list[tuple[str, Variant | None, Policy | None]] = [
                (variant_id, VARIANTS[variant_id], None) for variant_id in args.variant
            ]
            run_specs.extend((policy.policy_id, None, policy) for policy in policies)
            module_names = [str(entry["name"]) for entry in module_entries]
            module_families = {str(entry["name"]): str(entry["family"]) for entry in module_entries}

            for spec_id, uniform_variant, policy in run_specs:
                if uniform_variant is None:
                    if policy is None:
                        raise RuntimeError(f"invalid run spec: {spec_id}")
                    row_variant = policy_variant_payload(policy)
                    row_policy = policy_payload(policy, module_entries)
                else:
                    row_variant = variant_payload(uniform_variant)
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
                            variant = variant_for_policy(policy, str(entry["family"]))
                        else:
                            variant = uniform_variant
                        quantized = quantize_weight(
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
                    for prompt_index, (prompt, reference) in enumerate(zip(prompts, reference_logits, strict=True)):
                        logits = forward_last_logits(model, tokenizer, prompt, args.sequence_length, device)
                        metrics = logit_metrics(reference, logits)
                        rows.append(
                            {
                                "schema_version": "aq-module-logit-smoke-v0.1",
                                "run_id": args.run_id,
                                "timestamp_utc": utc_now(),
                                "status": "ok",
                                "model_dir": str(args.model_dir),
                                "activation_stats": str(args.activation_stats),
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
                                "metrics": metrics,
                                "notes": args.note,
                            }
                        )
                except Exception as exc:  # noqa: BLE001 - keep benchmark rows self-describing.
                    rows = [
                        {
                            "schema_version": "aq-module-logit-smoke-v0.1",
                            "run_id": args.run_id,
                            "timestamp_utc": utc_now(),
                            "status": "failed",
                            "model_dir": str(args.model_dir),
                            "activation_stats": str(args.activation_stats),
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
                            "metrics": {},
                            "notes": args.note,
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        }
                    ]
                finally:
                    restore_cumulative_modules(module_entries)
                for row in rows:
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                    output.write("\n")
            return 0

        for module_name in args.module:
            module = get_module(model, module_name)
            original = module.weight.detach().to("cpu").clone()
            activation = activation_for_module(module_name, int(module.in_features), stats)
            for variant_id in args.variant:
                variant = VARIANTS[variant_id]
                try:
                    quantized = quantize_weight(
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
                    for prompt_index, (prompt, reference) in enumerate(zip(prompts, reference_logits, strict=True)):
                        logits = forward_last_logits(model, tokenizer, prompt, args.sequence_length, device)
                        metrics = logit_metrics(reference, logits)
                        rows.append(
                            {
                                "schema_version": "aq-module-logit-smoke-v0.1",
                                "run_id": args.run_id,
                                "timestamp_utc": utc_now(),
                                "status": "ok",
                                "model_dir": str(args.model_dir),
                                "activation_stats": str(args.activation_stats),
                                "module": module_name,
                                "variant": variant_payload(variant),
                                "prompt": prompt,
                                "prompt_index": prompt_index,
                                "prompt_count": len(prompts),
                                "prompt_file": str(args.prompt_file) if args.prompt_file else None,
                                "sequence_length": args.sequence_length,
                                "metrics": metrics,
                                "notes": args.note,
                            }
                        )
                except Exception as exc:  # noqa: BLE001 - keep benchmark rows self-describing.
                    rows = [
                        {
                            "schema_version": "aq-module-logit-smoke-v0.1",
                            "run_id": args.run_id,
                            "timestamp_utc": utc_now(),
                            "status": "failed",
                            "model_dir": str(args.model_dir),
                            "activation_stats": str(args.activation_stats),
                            "module": module_name,
                            "variant": variant_payload(variant),
                            "prompt": None,
                            "prompt_count": len(prompts),
                            "prompt_file": str(args.prompt_file) if args.prompt_file else None,
                            "sequence_length": args.sequence_length,
                            "metrics": {},
                            "notes": args.note,
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        }
                    ]
                finally:
                    module.weight.data.copy_(original.to(device=module.weight.device, dtype=module.weight.dtype))
                for row in rows:
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                    output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
