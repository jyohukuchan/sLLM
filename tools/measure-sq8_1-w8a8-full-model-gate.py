#!/usr/bin/env python3
"""CPU fake-quant full-model quality gate for SQ8_1 W8A8.

The instrument deliberately runs no HIP code.  It applies the frozen SQ8_1
K=32 signed-int8 / upward-FP16-scale rule to real Qwen3.5-9B transformer
projection weights and, for W8A8, to every matching projection input.  The
quantized values are reconstructed in FP32 and evaluated through a selected
floating-point ``F.linear`` reference boundary.  This measures quantization
propagation rather than a particular GPU accumulation order.

It emits only aggregate/error evidence and ranking rows.  Raw model weights,
activations, hidden states, and logits are never persisted.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from typing import Any, Iterator

import torch
import torch.nn.functional as F


DEFAULT_MODULE_PATTERN = (
    r"(self_attn|linear_attn|mlp).*"
    r"(q_proj|k_proj|v_proj|o_proj|in_proj(_qkv|_qkvz|_ba|_[abz])?|"
    r"out_proj|gate_proj|up_proj|down_proj)$"
)
GROUP_SIZE = 32
OUTLIER_EDGES = (1.0, 2.0, 4.0, 8.0)
WILSON_Z_95 = 1.959963984540054


def load_collector() -> ModuleType:
    """Load the established corpus/model loader without copying its logic."""

    path = Path(__file__).resolve().with_name("collect-activation-stats.py")
    spec = importlib.util.spec_from_file_location("sq8_1_full_gate_collector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load collector: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COLLECTOR = load_collector()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return COLLECTOR.sha256_file(path)


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


@dataclass
class QuantizedBlocks:
    """Only persistent SQ8_1 planes; reconstruction is intentionally transient."""

    codes: torch.Tensor
    scales: torch.Tensor
    original_width: int
    padded_width: int
    group_size: int
    clipping_count: int
    edge_code_count: int
    zero_source_block_count: int
    nonfinite_scale_count: int

    @property
    def group_count(self) -> int:
        return self.padded_width // self.group_size


def quantize_sq8_1(values: torch.Tensor, group_size: int = GROUP_SIZE) -> QuantizedBlocks:
    """SQ8_1 RNE code + upward-FP16 scale semantics, including zero blocks.

    A zero source block uses the format's positive sentinel scale 1.0 and zero
    codes.  This is intentionally different from carrying a zero scale through
    a diagnostic tensor, because the SQ8_1 reader requires finite positive
    scale values.
    """

    if values.ndim != 2:
        raise ValueError(f"expected [rows, K], got {tuple(values.shape)}")
    if values.shape[1] < 1:
        raise ValueError("K must be positive")
    source = values.detach().to(torch.float32)
    if not bool(torch.isfinite(source).all()):
        raise ValueError("non-finite quantization source")

    rows, width = source.shape
    padded_width = round_up(int(width), group_size)
    if padded_width != width:
        source = F.pad(source, (0, padded_width - width))
    grouped = source.view(rows, padded_width // group_size, group_size)
    raw_scales = grouped.abs().amax(dim=-1) / 127.0
    zero_source = raw_scales == 0

    stored = raw_scales.to(torch.float16)
    lower_than_raw = stored.to(torch.float32) < raw_scales
    stored = torch.where(
        lower_than_raw,
        torch.nextafter(stored, torch.full_like(stored, float("inf"))),
        stored,
    )
    # The Rust packer writes positive 1.0 for an exactly-zero source block.
    stored = torch.where(zero_source, torch.ones_like(stored), stored)
    stored_f32 = stored.to(torch.float32)
    nonfinite_scale_count = int((~torch.isfinite(stored_f32)).sum().item())
    if nonfinite_scale_count:
        raise ValueError("SQ8_1 FP16 scale overflow")
    if bool((stored_f32 <= 0).any()):
        raise ValueError("SQ8_1 stored scale is not strictly positive")

    divided = grouped / stored_f32.unsqueeze(-1)
    codes = torch.round(divided).clamp(-127, 127).to(torch.int8)
    clipping_count = int((divided.abs() > 127.0).sum().item())
    edge_code_count = int((codes.abs() == 127).sum().item())
    return QuantizedBlocks(
        codes=codes.reshape(rows, padded_width).contiguous(),
        scales=stored.contiguous(),
        original_width=int(width),
        padded_width=padded_width,
        group_size=group_size,
        clipping_count=clipping_count,
        edge_code_count=edge_code_count,
        zero_source_block_count=int(zero_source.sum().item()),
        nonfinite_scale_count=nonfinite_scale_count,
    )


def reconstruct_blocks(quantized: QuantizedBlocks) -> torch.Tensor:
    """Reconstruct FP32 values from SQ8_1 planes; callers release this promptly."""

    rows = quantized.codes.shape[0]
    grouped_codes = quantized.codes.view(rows, quantized.group_count, quantized.group_size)
    reconstructed = grouped_codes.to(torch.float32) * quantized.scales.to(torch.float32).unsqueeze(-1)
    return reconstructed.reshape(rows, quantized.padded_width)[:, : quantized.original_width]


@dataclass
class ErrorAccumulator:
    value_count: int = 0
    reference_sumsq: float = 0.0
    candidate_sumsq: float = 0.0
    error_sumsq: float = 0.0
    absolute_error_sum: float = 0.0
    maximum_absolute_error: float = 0.0
    dot_product: float = 0.0
    nonfinite_count: int = 0

    def add(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        if reference.shape != candidate.shape:
            raise ValueError(f"comparison shape mismatch: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
        ref = reference.detach().to(torch.float32)
        cand = candidate.detach().to(torch.float32)
        finite = torch.isfinite(ref) & torch.isfinite(cand)
        if not bool(finite.all()):
            self.nonfinite_count += int((~finite).sum().item())
            raise ValueError("non-finite comparison value")
        error = cand - ref
        self.value_count += int(ref.numel())
        self.reference_sumsq += float(ref.to(torch.float64).square().sum().item())
        self.candidate_sumsq += float(cand.to(torch.float64).square().sum().item())
        self.error_sumsq += float(error.to(torch.float64).square().sum().item())
        self.absolute_error_sum += float(error.abs().to(torch.float64).sum().item())
        if error.numel():
            self.maximum_absolute_error = max(self.maximum_absolute_error, float(error.abs().amax().item()))
        self.dot_product += float((ref.to(torch.float64) * cand.to(torch.float64)).sum().item())

    def as_dict(self) -> dict[str, Any]:
        relative_l2 = math.sqrt(self.error_sumsq / self.reference_sumsq) if self.reference_sumsq else 0.0
        cosine_denominator = math.sqrt(self.reference_sumsq * self.candidate_sumsq)
        return {
            "value_count": self.value_count,
            "reference_sumsq": self.reference_sumsq,
            "candidate_sumsq": self.candidate_sumsq,
            "error_sumsq": self.error_sumsq,
            "relative_l2_error": relative_l2,
            "mean_absolute_error": self.absolute_error_sum / self.value_count if self.value_count else 0.0,
            "maximum_absolute_error": self.maximum_absolute_error,
            "cosine_similarity": self.dot_product / cosine_denominator if cosine_denominator else 1.0,
            "nonfinite_count": self.nonfinite_count,
        }


@dataclass
class QuantizationAccumulator:
    value_error: ErrorAccumulator = field(default_factory=ErrorAccumulator)
    block_count: int = 0
    clipping_count: int = 0
    edge_code_count: int = 0
    zero_source_block_count: int = 0
    nonfinite_scale_count: int = 0
    bypass_block_count: int = 0
    outlier_bin_counts: Counter[str] = field(default_factory=Counter)

    def add(
        self,
        source: torch.Tensor,
        quantized: QuantizedBlocks,
        effective_reconstructed: torch.Tensor,
        bypass_mask: torch.Tensor | None,
    ) -> None:
        self.value_error.add(source, effective_reconstructed)
        self.block_count += int(quantized.scales.numel())
        self.clipping_count += quantized.clipping_count
        self.edge_code_count += quantized.edge_code_count
        self.zero_source_block_count += quantized.zero_source_block_count
        self.nonfinite_scale_count += quantized.nonfinite_scale_count
        grouped = source.detach().to(torch.float32).view(source.shape[0], -1, quantized.group_size)
        rms = grouped.square().mean(dim=-1).sqrt()
        maximum = grouped.abs().amax(dim=-1)
        ratio = torch.where(rms > 0, maximum / rms, torch.zeros_like(rms))
        self.outlier_bin_counts["[0,1)"] += int((ratio < OUTLIER_EDGES[0]).sum().item())
        self.outlier_bin_counts["[1,2)"] += int(((ratio >= 1) & (ratio < 2)).sum().item())
        self.outlier_bin_counts["[2,4)"] += int(((ratio >= 2) & (ratio < 4)).sum().item())
        self.outlier_bin_counts["[4,8)"] += int(((ratio >= 4) & (ratio < 8)).sum().item())
        self.outlier_bin_counts["[8,inf)"] += int((ratio >= 8).sum().item())
        if bypass_mask is not None:
            self.bypass_block_count += int(bypass_mask.sum().item())

    def as_dict(self) -> dict[str, Any]:
        error = self.value_error.as_dict()
        return {
            "activation_value_error": error,
            "block_count": self.block_count,
            "true_clipping_count": self.clipping_count,
            "true_clipping_rate": self.clipping_count / error["value_count"] if error["value_count"] else 0.0,
            "edge_code_count": self.edge_code_count,
            "edge_code_rate": self.edge_code_count / error["value_count"] if error["value_count"] else 0.0,
            "zero_source_block_count": self.zero_source_block_count,
            "nonfinite_scale_count": self.nonfinite_scale_count,
            "outlier_bin_counts": dict(sorted(self.outlier_bin_counts.items())),
            "outlier_bin_rates": {
                name: count / self.block_count if self.block_count else 0.0
                for name, count in sorted(self.outlier_bin_counts.items())
            },
            "bypass_block_count": self.bypass_block_count,
            "bypass_block_rate": self.bypass_block_count / self.block_count if self.block_count else 0.0,
        }


@dataclass
class WeightAccumulator:
    error: ErrorAccumulator = field(default_factory=ErrorAccumulator)
    tensor_count: int = 0
    clipping_count: int = 0
    edge_code_count: int = 0
    zero_source_block_count: int = 0
    nonfinite_scale_count: int = 0
    code_min: int = 127
    code_max: int = -127

    def add(self, source: torch.Tensor, quantized: QuantizedBlocks, reconstructed: torch.Tensor) -> None:
        self.error.add(source, reconstructed)
        self.tensor_count += 1
        self.clipping_count += quantized.clipping_count
        self.edge_code_count += quantized.edge_code_count
        self.zero_source_block_count += quantized.zero_source_block_count
        self.nonfinite_scale_count += quantized.nonfinite_scale_count
        self.code_min = min(self.code_min, int(quantized.codes.min().item()))
        self.code_max = max(self.code_max, int(quantized.codes.max().item()))

    def as_dict(self) -> dict[str, Any]:
        result = self.error.as_dict()
        result.update(
            {
                "tensor_count": self.tensor_count,
                "true_clipping_count": self.clipping_count,
                "edge_code_count": self.edge_code_count,
                "zero_source_block_count": self.zero_source_block_count,
                "nonfinite_scale_count": self.nonfinite_scale_count,
                "code_min": self.code_min if self.tensor_count else None,
                "code_max": self.code_max if self.tensor_count else None,
            }
        )
        return result


@dataclass
class PreparedLinear:
    name: str
    module: torch.nn.Linear
    quantized_weight: QuantizedBlocks
    original_forward: Any


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    mode: str
    include_lm_head: bool


@dataclass
class CandidateMetrics:
    name: str
    logits: ErrorAccumulator = field(default_factory=ErrorAccumulator)
    layers: dict[str, ErrorAccumulator] = field(default_factory=dict)
    activation: QuantizationAccumulator | None = None
    token_count: int = 0
    top1_match_count: int = 0
    top1_total: int = 0
    top10_intersection_total: int = 0
    reference_top1_in_candidate_top10_count: int = 0
    kl_sum: float = 0.0
    mismatch_count: int = 0
    allowed_near_margin_mismatch_count: int = 0
    mismatch_margins: list[float] = field(default_factory=list)
    mismatch_margin_bins: Counter[str] = field(default_factory=Counter)
    per_prompt: list[dict[str, Any]] = field(default_factory=list)
    mismatch_rows: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.activation is None and self.name in {"w8a8", "outlier_bypass_ge4", "all_linear_w8a8"}:
            self.activation = QuantizationAccumulator()

    def layer_accumulator(self, name: str) -> ErrorAccumulator:
        return self.layers.setdefault(name, ErrorAccumulator())

    def margin_bin(self, margin: float) -> str:
        if margin <= 0.001:
            return "[0,0.001]"
        if margin <= 0.01:
            return "(0.001,0.01]"
        if margin <= 0.05:
            return "(0.01,0.05]"
        if margin <= 0.1:
            return "(0.05,0.1]"
        return ">(0.1)"

    def add_logits(
        self,
        record_id: str,
        reference: torch.Tensor,
        candidate: torch.Tensor,
    ) -> None:
        self.logits.add(reference, candidate)
        ref = reference.to(torch.float32)
        cand = candidate.to(torch.float32)
        reference_topk_values, reference_topk = torch.topk(ref, k=10, dim=-1)
        candidate_topk = torch.topk(cand, k=10, dim=-1).indices
        reference_top1 = reference_topk[:, 0]
        candidate_top1 = candidate_topk[:, 0]
        matches = reference_top1 == candidate_top1
        top1_total = int(reference_top1.numel())
        top1_matches = int(matches.sum().item())
        intersection = (reference_topk.unsqueeze(-1) == candidate_topk.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
        retained = (candidate_topk == reference_top1.unsqueeze(-1)).any(dim=-1)
        reference_log_prob = F.log_softmax(ref, dim=-1)
        candidate_log_prob = F.log_softmax(cand, dim=-1)
        kl = torch.sum(reference_log_prob.exp() * (reference_log_prob - candidate_log_prob), dim=-1)
        prompt_delta = cand - ref
        denom = float(ref.to(torch.float64).square().sum().item())
        prompt_rel_l2 = math.sqrt(float(prompt_delta.to(torch.float64).square().sum().item()) / denom) if denom else 0.0
        prompt_kl = float(kl.mean().item())
        self.token_count += top1_total
        self.top1_total += top1_total
        self.top1_match_count += top1_matches
        self.top10_intersection_total += int(intersection.sum().item())
        self.reference_top1_in_candidate_top10_count += int(retained.sum().item())
        self.kl_sum += float(kl.sum().item())
        self.per_prompt.append(
            {
                "candidate": self.name,
                "record_id": record_id,
                "scored_positions": top1_total,
                "logits_relative_l2_error": prompt_rel_l2,
                "logits_maximum_absolute_error": float(prompt_delta.abs().amax().item()),
                "mean_token_kl_reference_to_candidate": prompt_kl,
                "top1_match_count": top1_matches,
                "top1_total": top1_total,
                "top1_agreement_rate": top1_matches / top1_total if top1_total else 1.0,
                "top10_overlap_rate": float(intersection.to(torch.float32).mean().item() / 10.0),
                "reference_top1_retained_in_candidate_top10_count": int(retained.sum().item()),
            }
        )
        mismatch_positions = torch.nonzero(~matches, as_tuple=False).flatten()
        for position in mismatch_positions.tolist():
            reference_top1_id = int(reference_top1[position].item())
            reference_top2_id = int(reference_topk[position, 1].item())
            candidate_top1_id = int(candidate_top1[position].item())
            margin = float((reference_topk_values[position, 0] - reference_topk_values[position, 1]).item())
            candidate_rank = int(
                torch.nonzero(candidate_topk[position] == reference_top1[position], as_tuple=False)[0].item() + 1
            ) if bool((candidate_topk[position] == reference_top1[position]).any()) else None
            allowed = candidate_top1_id == reference_top2_id and margin <= 0.05
            self.mismatch_count += 1
            self.allowed_near_margin_mismatch_count += int(allowed)
            self.mismatch_margins.append(margin)
            self.mismatch_margin_bins[self.margin_bin(margin)] += 1
            self.mismatch_rows.append(
                {
                    "candidate": self.name,
                    "record_id": record_id,
                    "position": int(position),
                    "reference_top1_token_id": reference_top1_id,
                    "reference_top2_token_id": reference_top2_id,
                    "candidate_top1_token_id": candidate_top1_id,
                    "reference_top1_minus_top2_margin": margin,
                    "reference_top1_candidate_top10_rank": candidate_rank,
                    "allowed_near_margin_swap": allowed,
                }
            )

    def as_dict(self) -> dict[str, Any]:
        logits = self.logits.as_dict()
        agreement = self.top1_match_count / self.top1_total if self.top1_total else 1.0
        margins = sorted(self.mismatch_margins)

        def percentile(p: float) -> float | None:
            if not margins:
                return None
            index = min(len(margins) - 1, max(0, math.ceil(len(margins) * p) - 1))
            return margins[index]

        return {
            "candidate": self.name,
            "logits": logits,
            "mean_token_kl_reference_to_candidate": self.kl_sum / self.token_count if self.token_count else 0.0,
            "top1_match_count": self.top1_match_count,
            "top1_total": self.top1_total,
            "top1_agreement_rate": agreement,
            "top1_agreement_wilson_lower_95": wilson_lower_bound(self.top1_match_count, self.top1_total),
            "top10_overlap_rate": self.top10_intersection_total / (10 * self.top1_total) if self.top1_total else 1.0,
            "reference_top1_retained_in_candidate_top10_count": self.reference_top1_in_candidate_top10_count,
            "reference_top1_retained_in_candidate_top10_rate": self.reference_top1_in_candidate_top10_count / self.top1_total if self.top1_total else 1.0,
            "mismatch_count": self.mismatch_count,
            "allowed_near_margin_mismatch_count": self.allowed_near_margin_mismatch_count,
            "disallowed_mismatch_count": self.mismatch_count - self.allowed_near_margin_mismatch_count,
            "mismatch_margin_distribution": {
                "count": len(margins),
                "minimum": margins[0] if margins else None,
                "p50": percentile(0.50),
                "p90": percentile(0.90),
                "maximum": margins[-1] if margins else None,
                "bins": dict(sorted(self.mismatch_margin_bins.items())),
            },
            "layers": {name: accumulator.as_dict() for name, accumulator in sorted(self.layers.items())},
            "activation_quantization": self.activation.as_dict() if self.activation else None,
        }


def wilson_lower_bound(successes: int, total: int) -> float:
    if total == 0:
        return 1.0
    p = successes / total
    z2 = WILSON_Z_95 * WILSON_Z_95
    denominator = 1.0 + z2 / total
    centre = p + z2 / (2.0 * total)
    adjustment = WILSON_Z_95 * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return (centre - adjustment) / denominator


def extract_hidden(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                return extract_hidden(item)
            except TypeError:
                continue
    raise TypeError(f"cannot extract hidden tensor from {type(output)!r}")


class LayerCapture:
    """Forward hooks that retain one reference sample and aggregate candidates."""

    def __init__(self, model: torch.nn.Module):
        self.reference: dict[str, list[torch.Tensor]] = {}
        self.mode: str | None = None
        self.metrics: CandidateMetrics | None = None
        self.valid_lengths: list[int] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        layer_names = [name for name, _ in model.named_modules() if re.fullmatch(r"model\.layers\.\d+", name)]
        if len(layer_names) != 32:
            raise RuntimeError(f"expected 32 decoder layers, found {len(layer_names)}")
        if not hasattr(model, "model") or not hasattr(model.model, "norm"):
            raise RuntimeError("Qwen3.5 model norm path is unavailable")
        layer_names.append("model.norm")
        self.names = layer_names
        for name in layer_names:
            module = model.get_submodule(name)
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = extract_hidden(output)
            if hidden.ndim < 3:
                raise ValueError(f"{name}: expected [batch, tokens, hidden], got {tuple(hidden.shape)}")
            if hidden.shape[0] != len(self.valid_lengths):
                raise ValueError(f"{name}: batch size {hidden.shape[0]} does not match {len(self.valid_lengths)}")
            if self.mode == "reference":
                self.reference[name] = [
                    hidden[index, :valid_length].detach().to(torch.float32).clone()
                    for index, valid_length in enumerate(self.valid_lengths)
                ]
            elif self.mode == "candidate":
                if self.metrics is None or name not in self.reference:
                    raise RuntimeError(f"{name}: missing candidate/reference capture state")
                for index, valid_length in enumerate(self.valid_lengths):
                    self.metrics.layer_accumulator(name).add(
                        self.reference[name][index],
                        hidden[index, :valid_length].detach().to(torch.float32),
                    )

        return hook

    def capture_reference(self, valid_lengths: list[int]) -> None:
        self.reference.clear()
        self.valid_lengths = valid_lengths
        self.mode = "reference"
        self.metrics = None

    def capture_candidate(self, metrics: CandidateMetrics, valid_lengths: list[int]) -> None:
        self.valid_lengths = valid_lengths
        self.mode = "candidate"
        self.metrics = metrics

    def validate_reference(self) -> None:
        missing = sorted(set(self.names) - set(self.reference))
        if missing:
            raise RuntimeError(f"reference forward did not visit: {missing}")

    def validate_candidate(self, metrics: CandidateMetrics) -> None:
        missing = sorted(set(self.names) - set(metrics.layers))
        if missing:
            raise RuntimeError(f"candidate {metrics.name} did not visit: {missing}")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def select_examples(path: Path, requested: int) -> list[dict[str, Any]]:
    examples = list(COLLECTOR.iter_examples(path))
    if len(examples) < requested:
        raise ValueError(f"corpus contains {len(examples)} records, need {requested}")
    indices = [index * (len(examples) - 1) // (requested - 1) for index in range(requested)] if requested > 1 else [0]
    selected = [examples[index] for index in indices]
    if len({str(example["record_id"]) for example in selected}) != requested:
        raise RuntimeError("evenly spaced selection produced duplicate record IDs")
    return selected


def prepare_weights(
    model: torch.nn.Module,
    selected_names: set[str],
    lm_head_name: str,
    include_lm_head: bool,
) -> tuple[dict[str, PreparedLinear], dict[str, Any]]:
    prepared: dict[str, PreparedLinear] = {}
    primary_accumulator = WeightAccumulator()
    all_accumulator = WeightAccumulator()
    modules = dict(model.named_modules())
    names = sorted(selected_names | ({lm_head_name} if include_lm_head else set()))
    for index, name in enumerate(names, start=1):
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError(f"selected path is not nn.Linear: {name}")
        source = module.weight.detach().to(torch.float32).contiguous()
        quantized = quantize_sq8_1(source)
        reconstructed = reconstruct_blocks(quantized)
        all_accumulator.add(source, quantized, reconstructed)
        if name in selected_names:
            primary_accumulator.add(source, quantized, reconstructed)
        del reconstructed
        del source
        prepared[name] = PreparedLinear(name, module, quantized, module.forward)
        print(
            json.dumps(
                {"event": "weight_prequantized", "completed": index, "total": len(names), "tensor": name},
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
    return prepared, {
        "primary_248_projections": primary_accumulator.as_dict(),
        "all_linear_249": all_accumulator.as_dict() if include_lm_head else None,
    }


def outlier_mask(source: torch.Tensor, group_size: int, threshold: float) -> torch.Tensor:
    grouped = source.view(source.shape[0], -1, group_size)
    rms = grouped.square().mean(dim=-1).sqrt()
    maximum = grouped.abs().amax(dim=-1)
    ratio = torch.where(rms > 0, maximum / rms, torch.zeros_like(rms))
    return ratio >= threshold


def candidate_linear_forward(
    prepared: PreparedLinear,
    spec: CandidateSpec,
    activation_accumulator: QuantizationAccumulator | None,
    bypass_threshold: float,
    input_value: torch.Tensor,
) -> torch.Tensor:
    if not torch.is_tensor(input_value) or not input_value.is_floating_point():
        return prepared.original_forward(input_value)
    source = input_value.detach().to(torch.float32)
    original_shape = source.shape
    if source.shape[-1] != prepared.quantized_weight.original_width:
        raise ValueError(f"{prepared.name}: unexpected input width {source.shape[-1]}")
    if spec.mode == "control":
        # This intentionally has the exact F.linear operand boundary of
        # nn.Linear.forward.  It is the strict harness control; it catches a
        # monkeypatch/hook wiring error before a quantization result is used.
        return F.linear(input_value, prepared.module.weight, prepared.module.bias)
    else:
        weight = reconstruct_blocks(prepared.quantized_weight)
        if spec.mode in {"w8a8", "outlier_bypass_ge4"}:
            flat = source.reshape(-1, source.shape[-1]).contiguous()
            activation = quantize_sq8_1(flat)
            reconstructed = reconstruct_blocks(activation)
            bypass = None
            if spec.mode == "outlier_bypass_ge4":
                bypass = outlier_mask(flat, activation.group_size, bypass_threshold)
                grouped_source = flat.view(flat.shape[0], activation.group_count, activation.group_size)
                grouped_reconstructed = reconstructed.view(flat.shape[0], activation.group_count, activation.group_size)
                reconstructed = torch.where(bypass.unsqueeze(-1), grouped_source, grouped_reconstructed).reshape_as(flat)
            if activation_accumulator is None:
                raise RuntimeError("W8A8 candidate lacks activation accumulator")
            activation_accumulator.add(flat, activation, reconstructed, bypass)
            effective_input = reconstructed.reshape(original_shape)
        elif spec.mode == "w8a16":
            effective_input = source
        else:
            raise RuntimeError(f"unsupported candidate mode: {spec.mode}")
    weight = weight.to(dtype=input_value.dtype)
    effective_input = effective_input.to(dtype=input_value.dtype)
    bias = prepared.module.bias
    return F.linear(effective_input, weight, bias)


@contextlib.contextmanager
def patched_candidate(
    prepared: dict[str, PreparedLinear],
    primary_names: set[str],
    lm_head_name: str,
    spec: CandidateSpec,
    activation_accumulator: QuantizationAccumulator | None,
    bypass_threshold: float,
) -> Iterator[None]:
    selected = sorted(primary_names | ({lm_head_name} if spec.include_lm_head else set()))
    originals: list[tuple[torch.nn.Linear, Any]] = []
    try:
        for name in selected:
            item = prepared[name]

            def forward(module: torch.nn.Linear, input_value: torch.Tensor, _item: PreparedLinear = item) -> torch.Tensor:
                return candidate_linear_forward(_item, spec, activation_accumulator, bypass_threshold, input_value)

            originals.append((item.module, item.module.forward))
            item.module.forward = MethodType(forward, item.module)
        yield
    finally:
        for module, original in reversed(originals):
            module.forward = original


def candidate_specs(run_all_linear_stress: bool) -> list[CandidateSpec]:
    specs = [
        CandidateSpec("control", "control", False),
        CandidateSpec("w8a16", "w8a16", False),
        CandidateSpec("w8a8", "w8a8", False),
        CandidateSpec("outlier_bypass_ge4", "outlier_bypass_ge4", False),
    ]
    if run_all_linear_stress:
        specs.extend(
            [
                CandidateSpec("all_linear_w8a16", "w8a16", True),
                CandidateSpec("all_linear_w8a8", "w8a8", True),
            ]
        )
    return specs


def run_measurement(args: argparse.Namespace) -> tuple[dict[str, CandidateMetrics], dict[str, Any], dict[str, Any]]:
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    torch.manual_seed(args.seed)
    model_args = SimpleNamespace(
        model_dir=args.model_dir,
        model_class="causal_lm",
        dtype=args.model_dtype,
        trust_remote_code=args.trust_remote_code,
        device="cpu",
    )
    tokenizer, model = COLLECTOR.load_transformers_model(model_args)
    device = next(model.parameters()).device
    if device.type != "cpu":
        raise RuntimeError(f"CPU-only gate refused non-CPU model: {device}")
    if model.training:
        raise RuntimeError("model must be in eval mode")

    module_pattern = re.compile(args.module_pattern)
    primary_names = {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and module_pattern.search(name)
    }
    lm_head_name = "lm_head"
    if len(primary_names) != 248:
        raise RuntimeError(f"expected exactly 248 primary SQ8_1 projections, found {len(primary_names)}")
    lm_head = dict(model.named_modules()).get(lm_head_name)
    if not isinstance(lm_head, torch.nn.Linear):
        raise RuntimeError("expected lm_head to be nn.Linear")
    selected = select_examples(args.prompt_file, args.max_samples)
    domain_counts = Counter(str(example.get("domain", "unknown")) for example in selected)
    if not args.allow_incomplete_coverage:
        if args.max_samples != 20 or len(domain_counts) != 5 or any(count != 4 for count in domain_counts.values()):
            raise RuntimeError(f"frozen gate requires 20 records / five domains x4, got {dict(domain_counts)}")

    capture = LayerCapture(model)
    specs = candidate_specs(args.run_all_linear_stress)
    metrics = {spec.name: CandidateMetrics(spec.name) for spec in specs}
    start = time.monotonic()
    try:
        prepared, weight_summary = prepare_weights(model, primary_names, lm_head_name, args.run_all_linear_stress)
        valid_tokens = 0
        record_ids: list[str] = []
        with torch.inference_mode():
            for batch_index, start_index in enumerate(range(0, len(selected), args.batch_size), start=1):
                examples = selected[start_index : start_index + args.batch_size]
                batch, render_kinds = COLLECTOR.encode_examples(tokenizer, examples, args.sequence_length, False)
                batch = {key: value.to(device) for key, value in batch.items()}
                attention_mask = batch.get("attention_mask")
                valid_lengths = (
                    [int(value) for value in attention_mask.sum(dim=1).tolist()]
                    if attention_mask is not None
                    else [int(batch["input_ids"].shape[1])] * len(examples)
                )
                if any(value < 1 for value in valid_lengths):
                    raise RuntimeError(f"batch {batch_index}: at least one record has no valid tokens")
                batch_record_ids = [str(example["record_id"]) for example in examples]
                record_ids.extend(batch_record_ids)
                print(
                    json.dumps(
                        {
                            "event": "batch_begin",
                            "batch_index": batch_index,
                            "batch_total": math.ceil(len(selected) / args.batch_size),
                            "record_ids": batch_record_ids,
                            "valid_tokens": valid_lengths,
                            "render_kinds": render_kinds,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                capture.capture_reference(valid_lengths)
                reference_outputs = model(**batch, use_cache=False)
                reference_logits = getattr(reference_outputs, "logits", None)
                if reference_logits is None:
                    raise RuntimeError("causal LM output has no logits")
                reference_logits = reference_logits.detach().clone()
                capture.validate_reference()
                valid_tokens += sum(valid_lengths)
                for spec in specs:
                    candidate = metrics[spec.name]
                    capture.capture_candidate(candidate, valid_lengths)
                    with patched_candidate(
                        prepared,
                        primary_names,
                        lm_head_name,
                        spec,
                        candidate.activation,
                        args.outlier_bypass_threshold,
                    ):
                        outputs = model(**batch, use_cache=False)
                    candidate_logits = getattr(outputs, "logits", None)
                    if candidate_logits is None:
                        raise RuntimeError(f"{spec.name}: causal LM output has no logits")
                    for row_index, (record_id, valid_length) in enumerate(zip(batch_record_ids, valid_lengths, strict=True)):
                        candidate.add_logits(
                            record_id,
                            reference_logits[row_index, :valid_length],
                            candidate_logits[row_index, :valid_length],
                        )
                    capture.validate_candidate(candidate)
                    print(
                        json.dumps(
                            {
                                "event": "candidate_complete",
                                "candidate": spec.name,
                                "record_ids": batch_record_ids,
                                "batch_index": batch_index,
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                del reference_logits
                capture.reference.clear()
        run = {
            "samples_seen": len(selected),
            "valid_scored_positions": valid_tokens,
            "domain_counts": dict(sorted(domain_counts.items())),
            "processed_record_ids_sha256": hashlib.sha256("\n".join(record_ids).encode("utf-8")).hexdigest(),
            "rendering": "official chat template when messages are supplied; plain text otherwise",
            "elapsed_seconds": time.monotonic() - start,
            "primary_projection_count": len(primary_names),
            "all_linear_count": len(primary_names) + 1,
        }
        return metrics, weight_summary, run
    finally:
        capture.close()


def maximum_layer_metric(candidate: dict[str, Any], key: str) -> tuple[str, float]:
    rows = candidate["layers"]
    return max(((name, float(row[key])) for name, row in rows.items()), key=lambda item: item[1])


def worst_prompt(rows: list[dict[str, Any]], key: str) -> float:
    return max((float(row[key]) for row in rows), default=0.0)


def check(name: str, observed: Any, threshold: str, passed: bool, note: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "observed": observed, "threshold": threshold, "passed": passed}
    if note:
        result["note"] = note
    return result


def numeric_w8a8_checks(w8a16: dict[str, Any], w8a8: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    logits = w8a8["logits"]
    final_hidden = w8a8["layers"]["model.norm"]
    w8a16_logits_l2 = float(w8a16["logits"]["relative_l2_error"])
    w8a16_final_l2 = float(w8a16["layers"]["model.norm"]["relative_l2_error"])
    maximum_layer_name, maximum_layer_l2 = maximum_layer_metric(w8a8, "relative_l2_error")
    return [
        check(
            f"{prefix}_aggregate_logits_relative_l2",
            logits["relative_l2_error"],
            "<= 0.060",
            float(logits["relative_l2_error"]) <= 0.060,
        ),
        check(
            f"{prefix}_worst_prompt_logits_relative_l2",
            worst_prompt(w8a8.get("per_prompt", []), "logits_relative_l2_error"),
            "<= 0.080",
            worst_prompt(w8a8.get("per_prompt", []), "logits_relative_l2_error") <= 0.080,
        ),
        check(
            f"{prefix}_logits_max_abs",
            logits["maximum_absolute_error"],
            "<= 1.0",
            float(logits["maximum_absolute_error"]) <= 1.0,
        ),
        check(
            f"{prefix}_mean_token_kl",
            w8a8["mean_token_kl_reference_to_candidate"],
            "<= 0.005",
            float(w8a8["mean_token_kl_reference_to_candidate"]) <= 0.005,
        ),
        check(
            f"{prefix}_worst_prompt_mean_kl",
            worst_prompt(w8a8.get("per_prompt", []), "mean_token_kl_reference_to_candidate"),
            "<= 0.010",
            worst_prompt(w8a8.get("per_prompt", []), "mean_token_kl_reference_to_candidate") <= 0.010,
        ),
        check(
            f"{prefix}_incremental_logits_penalty_ratio",
            float(logits["relative_l2_error"]) / w8a16_logits_l2 if w8a16_logits_l2 else None,
            "W8A8 <= 1.60 * W8A16",
            float(logits["relative_l2_error"]) <= 1.60 * w8a16_logits_l2,
        ),
        check(
            f"{prefix}_incremental_logits_penalty_absolute",
            float(logits["relative_l2_error"]) - w8a16_logits_l2,
            "W8A8 <= W8A16 + 0.020",
            float(logits["relative_l2_error"]) <= w8a16_logits_l2 + 0.020,
        ),
        check(
            f"{prefix}_maximum_layer_relative_l2",
            {"layer": maximum_layer_name, "relative_l2_error": maximum_layer_l2},
            "<= 0.080",
            maximum_layer_l2 <= 0.080,
        ),
        check(
            f"{prefix}_final_hidden_relative_l2",
            final_hidden["relative_l2_error"],
            "<= 0.060",
            float(final_hidden["relative_l2_error"]) <= 0.060,
        ),
        check(
            f"{prefix}_final_hidden_max_abs",
            final_hidden["maximum_absolute_error"],
            "<= 1.0",
            float(final_hidden["maximum_absolute_error"]) <= 1.0,
        ),
        check(
            f"{prefix}_incremental_final_hidden_penalty_ratio",
            float(final_hidden["relative_l2_error"]) / w8a16_final_l2 if w8a16_final_l2 else None,
            "W8A8 <= 1.60 * W8A16",
            float(final_hidden["relative_l2_error"]) <= 1.60 * w8a16_final_l2,
        ),
        check(
            f"{prefix}_incremental_final_hidden_penalty_absolute",
            float(final_hidden["relative_l2_error"]) - w8a16_final_l2,
            "W8A8 <= W8A16 + 0.020",
            float(final_hidden["relative_l2_error"]) <= w8a16_final_l2 + 0.020,
        ),
    ]


def evaluate_gate(
    metrics: dict[str, CandidateMetrics],
    weight_summary: dict[str, Any],
    run: dict[str, Any],
    criteria_sha256: str,
    allow_incomplete_coverage: bool,
) -> dict[str, Any]:
    candidates = {name: item.as_dict() for name, item in metrics.items()}
    for item in metrics.values():
        candidates[item.name]["per_prompt"] = item.per_prompt
    control = candidates["control"]
    w8a16 = candidates["w8a16"]
    w8a8 = candidates["w8a8"]
    primary_weight = weight_summary["primary_248_projections"]
    checks: list[dict[str, Any]] = []
    coverage_ok = (
        run["samples_seen"] == 20
        and run["valid_scored_positions"] >= 4000
        and len(run["domain_counts"]) == 5
        and all(count == 4 for count in run["domain_counts"].values())
    )
    checks.append(
        check(
            "coverage",
            {
                "samples_seen": run["samples_seen"],
                "valid_scored_positions": run["valid_scored_positions"],
                "domain_counts": run["domain_counts"],
            },
            "20 records; five domains x4; >=4,000 scored positions",
            coverage_ok if not allow_incomplete_coverage else False,
            "test-only incomplete coverage is never a production gate pass" if allow_incomplete_coverage else None,
        )
    )
    checks.extend(
        [
            check(
                "weight_storage_validity",
                {
                    "true_clipping_count": primary_weight["true_clipping_count"],
                    "nonfinite_scale_count": primary_weight["nonfinite_scale_count"],
                    "code_min": primary_weight["code_min"],
                    "code_max": primary_weight["code_max"],
                },
                "zero clipping; finite scales; code range [-127,127]",
                primary_weight["true_clipping_count"] == 0
                and primary_weight["nonfinite_scale_count"] == 0
                and primary_weight["code_min"] >= -127
                and primary_weight["code_max"] <= 127,
            ),
            check(
                "control_logits_relative_l2",
                control["logits"]["relative_l2_error"],
                "<= 1e-5",
                float(control["logits"]["relative_l2_error"]) <= 1e-5,
            ),
            check(
                "control_logits_max_abs",
                control["logits"]["maximum_absolute_error"],
                "<= 2e-5",
                float(control["logits"]["maximum_absolute_error"]) <= 2e-5,
            ),
            check(
                "control_final_hidden_relative_l2",
                control["layers"]["model.norm"]["relative_l2_error"],
                "<= 1e-5",
                float(control["layers"]["model.norm"]["relative_l2_error"]) <= 1e-5,
            ),
            check(
                "control_final_hidden_max_abs",
                control["layers"]["model.norm"]["maximum_absolute_error"],
                "<= 2e-5",
                float(control["layers"]["model.norm"]["maximum_absolute_error"]) <= 2e-5,
            ),
            check(
                "w8a16_aggregate_logits_relative_l2",
                w8a16["logits"]["relative_l2_error"],
                "<= 0.040",
                float(w8a16["logits"]["relative_l2_error"]) <= 0.040,
            ),
            check(
                "w8a16_worst_prompt_logits_relative_l2",
                worst_prompt(w8a16["per_prompt"], "logits_relative_l2_error"),
                "<= 0.060",
                worst_prompt(w8a16["per_prompt"], "logits_relative_l2_error") <= 0.060,
            ),
        ]
    )
    activation = w8a8["activation_quantization"]
    checks.append(
        check(
            "w8a8_activation_storage_validity",
            {
                "true_clipping_count": activation["true_clipping_count"],
                "nonfinite_scale_count": activation["nonfinite_scale_count"],
            },
            "zero clipping; finite scales",
            activation["true_clipping_count"] == 0 and activation["nonfinite_scale_count"] == 0,
        )
    )
    checks.extend(numeric_w8a8_checks(w8a16, w8a8, "w8a8"))
    checks.extend(
        [
            check(
                "w8a8_top10_overlap",
                w8a8["top10_overlap_rate"],
                ">= 0.950",
                float(w8a8["top10_overlap_rate"]) >= 0.950,
            ),
            check(
                "w8a8_reference_top1_retained_in_top10",
                w8a8["reference_top1_retained_in_candidate_top10_rate"],
                "1.0",
                float(w8a8["reference_top1_retained_in_candidate_top10_rate"]) == 1.0,
            ),
            check(
                "w8a8_top1_agreement",
                {
                    "rate": w8a8["top1_agreement_rate"],
                    "wilson_lower_95": w8a8["top1_agreement_wilson_lower_95"],
                    "disallowed_mismatch_count": w8a8["disallowed_mismatch_count"],
                },
                "rate >= 0.990; Wilson lower >= 0.985; every mismatch allowed near-margin runner-up swap",
                float(w8a8["top1_agreement_rate"]) >= 0.990
                and float(w8a8["top1_agreement_wilson_lower_95"]) >= 0.985
                and int(w8a8["disallowed_mismatch_count"]) == 0,
            ),
        ]
    )
    primary_passed = all(item["passed"] for item in checks)

    outlier = candidates["outlier_bypass_ge4"]
    gap_base = max(0.0, float(w8a8["logits"]["relative_l2_error"]) - float(w8a16["logits"]["relative_l2_error"]))
    gap_bypass = max(0.0, float(outlier["logits"]["relative_l2_error"]) - float(w8a16["logits"]["relative_l2_error"]))
    fraction_removed = (gap_base - gap_bypass) / gap_base if gap_base else 0.0
    bypass_numeric = all(item["passed"] for item in numeric_w8a8_checks(w8a16, outlier, "outlier_bypass_ge4"))
    outlier_assessment = {
        "base_w8a8_to_w8a16_logit_l2_gap": gap_base,
        "bypass_to_w8a16_logit_l2_gap": gap_bypass,
        "gap_fraction_removed": fraction_removed,
        "numeric_gate_pass_with_upper_bound_bypass": bypass_numeric,
        "promising_outlier_side_route": bypass_numeric or fraction_removed >= 0.50,
        "rule": "promising only if bypass passes all numeric W8A8 gates or removes >=50% of the W8A8-to-W8A16 aggregate-logit-L2 gap",
    }
    all_linear_status: dict[str, Any] | None = None
    if "all_linear_w8a8" in candidates:
        all_w8a16 = candidates["all_linear_w8a16"]
        all_w8a8 = candidates["all_linear_w8a8"]
        all_checks = numeric_w8a8_checks(all_w8a16, all_w8a8, "all_linear_w8a8")
        all_checks.extend(
            [
                check(
                    "all_linear_w8a8_top10_overlap",
                    all_w8a8["top10_overlap_rate"],
                    ">= 0.950",
                    float(all_w8a8["top10_overlap_rate"]) >= 0.950,
                ),
                check(
                    "all_linear_w8a8_top1_agreement",
                    {
                        "rate": all_w8a8["top1_agreement_rate"],
                        "wilson_lower_95": all_w8a8["top1_agreement_wilson_lower_95"],
                        "disallowed_mismatch_count": all_w8a8["disallowed_mismatch_count"],
                    },
                    "rate >= 0.990; Wilson lower >= 0.985; zero disallowed mismatch",
                    float(all_w8a8["top1_agreement_rate"]) >= 0.990
                    and float(all_w8a8["top1_agreement_wilson_lower_95"]) >= 0.985
                    and int(all_w8a8["disallowed_mismatch_count"]) == 0,
                ),
            ]
        )
        all_linear_status = {"passed": all(item["passed"] for item in all_checks), "checks": all_checks}

    return {
        "criteria_sha256": criteria_sha256,
        "primary_scope_status": "pass" if primary_passed else "no-go",
        "primary_scope_passed": primary_passed,
        "primary_checks": checks,
        "outlier_attribution": outlier_assessment,
        "supplementary_all_linear_stress": all_linear_status,
        "candidates": candidates,
    }


def render_readme(summary: dict[str, Any]) -> str:
    decision = summary["gate"]["primary_scope_status"]
    candidates = summary["gate"]["candidates"]
    lines = [
        "# SQ8_1 W8A8 full-model quality gate",
        "",
        f"**Primary 248-projection decision: `{decision}`.**",
        "",
        "## Scope",
        "",
        "- CPU-only Qwen3.5-9B floating-point reference and fake-quant candidates; no GPU or service was used.",
        "- Primary scope quantizes all 248 selected transformer projections. `lm_head` remains unmodified FP32 there; the separate 249-Linear stress scope adds it explicitly.",
        "- W8A8 uses per-token K=32 signed symmetric int8 activations and per-row K=32 signed symmetric int8 weights, with RNE codes and upward-rounded FP16 scales.",
        "- SQ8_1 values are reconstructed in FP32, then passed through the same floating-point `F.linear` operand boundary as the reference. This is a full-model quantization-propagation gate, not a GPU accumulation-order or performance result.",
        "",
        "## Full-model logits",
        "",
        "| candidate | relative L2 | max abs | mean KL | top-1 agreement | top-10 overlap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("control", "w8a16", "w8a8", "outlier_bypass_ge4", "all_linear_w8a16", "all_linear_w8a8"):
        if name not in candidates:
            continue
        item = candidates[name]
        logits = item["logits"]
        lines.append(
            f"| {name} | {logits['relative_l2_error']:.8g} | {logits['maximum_absolute_error']:.8g} | "
            f"{item['mean_token_kl_reference_to_candidate']:.8g} | "
            f"{item['top1_match_count']}/{item['top1_total']} ({item['top1_agreement_rate']:.6%}) | "
            f"{item['top10_overlap_rate']:.6%} |"
        )
    w8a16 = candidates["w8a16"]
    w8a8 = candidates["w8a8"]
    lines.extend(
        [
            "",
            "## Hidden-state propagation",
            "",
            "| candidate | final hidden relative L2 | final hidden max abs | worst layer (relative L2) |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for name in ("w8a16", "w8a8", "outlier_bypass_ge4", "all_linear_w8a16", "all_linear_w8a8"):
        if name not in candidates:
            continue
        item = candidates[name]
        final_hidden = item["layers"]["model.norm"]
        worst_name, worst_value = maximum_layer_metric(item, "relative_l2_error")
        lines.append(
            f"| {name} | {final_hidden['relative_l2_error']:.8g} | {final_hidden['maximum_absolute_error']:.8g} | {worst_name}: {worst_value:.8g} |"
        )
    outlier = summary["gate"]["outlier_attribution"]
    activation = w8a8["activation_quantization"]
    lines.extend(
        [
            "",
            "## Outliers",
            "",
            f"- Base W8A8 K=32 outlier blocks `[4,8)`: `{activation['outlier_bin_rates'].get('[4,8)', 0.0):.6%}`; `[8,inf)`: `{activation['outlier_bin_rates'].get('[8,inf)', 0.0):.6%}`.",
            f"- The diagnostic `outlier_bypass_ge4` bypassed `{candidates['outlier_bypass_ge4']['activation_quantization']['bypass_block_rate']:.6%}` of activation blocks and removed `{outlier['gap_fraction_removed']:.6%}` of the W8A8-to-W8A16 aggregate-logit-L2 gap.",
            f"- Outlier-side-route outlook under the frozen rule: `{'promising' if outlier['promising_outlier_side_route'] else 'not supported by this run'}`.",
            "",
            "## Gate checks",
            "",
            "| check | observed | threshold | pass |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in summary["gate"]["primary_checks"]:
        observed = json.dumps(item["observed"], ensure_ascii=False, sort_keys=True) if isinstance(item["observed"], (dict, list)) else str(item["observed"])
        lines.append(f"| {item['name']} | {observed} | {item['threshold']} | {'yes' if item['passed'] else 'NO'} |")
    lines.extend(
        [
            "",
            "The complete frozen contract is `gate-criteria.md`; reproducibility/provenance is in `measurement-manifest.json`.  Per-prompt values, every layer, mismatch margins, and activation outlier counts are retained as structured evidence in this directory. Raw model tensors are intentionally not written.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--criteria-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--model-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--module-pattern", default=DEFAULT_MODULE_PATTERN)
    parser.add_argument("--torch-threads", type=int, default=32)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outlier-bypass-threshold", type=float, default=4.0)
    parser.add_argument("--run-all-linear-stress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-incomplete-coverage", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_samples < 1 or args.batch_size < 1 or args.sequence_length < 1:
        raise SystemExit("--max-samples, --batch-size, and --sequence-length must be positive")
    if args.torch_threads < 1 or args.torch_interop_threads < 1:
        raise SystemExit("thread counts must be positive")
    if args.outlier_bypass_threshold <= 0:
        raise SystemExit("--outlier-bypass-threshold must be positive")
    for attr in ("model_dir", "prompt_file", "corpus_manifest", "criteria_file", "output_dir"):
        setattr(args, attr, getattr(args, attr).expanduser().resolve())
    if not args.model_dir.is_dir():
        raise SystemExit(f"missing model directory: {args.model_dir}")
    for path in (args.prompt_file, args.corpus_manifest, args.criteria_file):
        if not path.is_file():
            raise SystemExit(f"missing input file: {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        allowed = {"gate-criteria.md"}
        existing = {item.name for item in args.output_dir.iterdir()}
        if existing != allowed or args.criteria_file != args.output_dir / "gate-criteria.md":
            raise SystemExit(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        raise SystemExit("CPU-only measurement refuses an available CUDA/HIP torch device")

    tool_path = Path(__file__).resolve()
    start_revision = COLLECTOR.git_revision()
    start_utc = utc_now()
    metrics, weight_summary, run = run_measurement(args)
    criteria_sha = sha256_file(args.criteria_file)
    gate = evaluate_gate(metrics, weight_summary, run, criteria_sha, args.allow_incomplete_coverage)
    end_utc = utc_now()
    manifest = {
        "schema_version": "ullm.sq8_1-w8a8-full-model-gate.v0.1",
        "status": "completed",
        "timestamp_utc_start": start_utc,
        "timestamp_utc_end": end_utc,
        "cpu_only": True,
        "gpu_execution": "not used",
        "command": sys.argv,
        "cwd": str(Path.cwd()),
        "git_revision_at_start": start_revision,
        "git_revision_at_finalize": COLLECTOR.git_revision(),
        "measurement_tool": {"path": str(tool_path), "sha256": sha256_file(tool_path)},
        "reused_activation_collector": {
            "path": str(tool_path.with_name("collect-activation-stats.py")),
            "sha256": sha256_file(tool_path.with_name("collect-activation-stats.py")),
            "reused_functions": ["load_transformers_model", "iter_examples", "encode_examples", "sha256_file", "git_revision"],
        },
        "criteria": {"path": str(args.criteria_file), "sha256": criteria_sha, "frozen_before_measurement": True},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_hip": torch.version.hip,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        },
        "model": {
            "path": str(args.model_dir),
            "config_sha256": sha256_file(args.model_dir / "config.json"),
            "weight_index_sha256": sha256_file(args.model_dir / "model.safetensors.index.json"),
            "tokenizer_config_sha256": sha256_file(args.model_dir / "tokenizer_config.json"),
        },
        "corpus": {
            "prompt_file": str(args.prompt_file),
            "prompt_file_sha256": sha256_file(args.prompt_file),
            "corpus_manifest": str(args.corpus_manifest),
            "corpus_manifest_sha256": sha256_file(args.corpus_manifest),
            "selection": "deterministic evenly spaced record indices across the complete prompt file",
        },
        "quantization": {
            "format": "SQ8_1 diagnostic full-model fake-quant",
            "group_size": GROUP_SIZE,
            "symmetric": True,
            "zero_point": None,
            "code_range": [-127, 127],
            "rounding": "torch.round nearest-even (RNE)",
            "scale": "per-row K=32 max(abs(x))/127 stored FP16 rounded upward; exact-zero blocks store positive 1.0 scale",
            "compute": "FP32 reconstruction followed by the reference FP32 F.linear operand/output boundary",
            "outlier_bypass": "diagnostic only: source activation blocks with max(abs)/RMS >= threshold bypass dynamic activation quantization",
        },
        "scope": {
            "module_pattern": args.module_pattern,
            "primary_projection_count_expected": 248,
            "lm_head_stress_enabled": args.run_all_linear_stress,
            "lm_head_primary_scope": "FP32 / unmodified",
            "primary_scope": "all matching transformer projection weights W8; matching inputs W8 for W8A8",
        },
        "run": {
            **run,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "model_dtype": args.model_dtype,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
            "seed": args.seed,
            "outlier_bypass_threshold": args.outlier_bypass_threshold,
            "allow_incomplete_coverage": args.allow_incomplete_coverage,
        },
    }
    summary = {
        "schema_version": manifest["schema_version"],
        "measurement_manifest": "measurement-manifest.json",
        "gate_criteria": "gate-criteria.md",
        "weight_quantization": weight_summary,
        "gate": gate,
    }
    (args.output_dir / "measurement-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "layer-metrics.json").write_text(
        json.dumps(
            {name: payload["layers"] for name, payload in gate["candidates"].items()},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "outlier-analysis.json").write_text(
        json.dumps(
            {
                "base_w8a8_activation": gate["candidates"]["w8a8"]["activation_quantization"],
                "outlier_bypass_ge4_activation": gate["candidates"]["outlier_bypass_ge4"]["activation_quantization"],
                "attribution": gate["outlier_attribution"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "per-prompt.jsonl").open("w", encoding="utf-8") as handle:
        for name in sorted(gate["candidates"]):
            for row in gate["candidates"][name]["per_prompt"]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "top1-mismatches.jsonl").open("w", encoding="utf-8") as handle:
        for name in sorted(metrics):
            for row in metrics[name].mismatch_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "README.md").write_text(render_readme(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "complete",
                "output_dir": str(args.output_dir),
                "primary_scope_status": gate["primary_scope_status"],
                "valid_scored_positions": run["valid_scored_positions"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
