#!/usr/bin/env python3
"""Measure SQ8_1 W8A8 activation error from real model pre-hook tensors.

This is deliberately a CPU-only measurement instrument, not an SQ8_1
quantizer, artifact reader, or GPU kernel.  It imports and reuses the
importance-score activation collector for model loading, corpus decoding, and
the same Linear-module naming convention.  Unlike that collector it observes
the raw pre-hook tensors transiently so that a dynamic per-token int8 scale can
be evaluated.  Raw activation tensors are not written to disk.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_MODULE_PATTERN = (
    r"(self_attn|linear_attn|mlp).*"
    r"(q_proj|k_proj|v_proj|o_proj|in_proj(_qkv|_qkvz|_ba|_[abz])?|"
    r"out_proj|gate_proj|up_proj|down_proj)$"
)
DEFAULT_GROUP_SIZES = (16, 32, 64, 128)
SCALE_ENCODINGS = ("float16_rne", "float16_ceil", "bfloat16_rne", "bfloat16_ceil")
OUTLIER_BIN_EDGES = (1.0, 2.0, 4.0, 8.0)


def load_activation_collector() -> ModuleType:
    """Load the existing importance-score collector without duplicating it."""

    path = Path(__file__).resolve().with_name("collect-activation-stats.py")
    spec = importlib.util.spec_from_file_location("sq8_1_activation_collector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load existing activation collector: {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve postponed annotations through sys.modules while the
    # imported collector is executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COLLECTOR = load_activation_collector()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path | None) -> str | None:
    return COLLECTOR.sha256_file(path)


def finite_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


def select_evenly(length: int, limit: int, device: torch.device) -> torch.Tensor:
    if length < 1:
        return torch.empty(0, dtype=torch.long, device=device)
    count = min(length, limit)
    if count == length:
        return torch.arange(length, dtype=torch.long, device=device)
    if count == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    # Integer arithmetic keeps the sample deterministic across torch versions.
    return torch.arange(count, dtype=torch.long, device=device) * (length - 1) // (count - 1)


@dataclass
class QuantizedBlocks:
    codes: torch.Tensor
    scales: torch.Tensor
    reconstructed: torch.Tensor
    original_width: int
    padded_width: int
    positive_scale_count: int
    stored_zero_positive_scale_count: int
    stored_nonfinite_scale_count: int
    clipping_count: int
    saturation_count: int


def quantize_symmetric_per_token(
    values: torch.Tensor,
    group_size: int,
    scale_encoding: str,
) -> QuantizedBlocks:
    """RNE, signed int8, max-abs block scale with actual stored-scale semantics."""

    if values.ndim != 2:
        raise ValueError(f"expected [tokens, K], got {tuple(values.shape)}")
    if values.shape[1] < 1:
        raise ValueError("K must be positive")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("non-finite activation or weight encountered")

    source = values.to(dtype=torch.float32)
    token_count, width = source.shape
    group_count = (width + group_size - 1) // group_size
    padded_width = group_count * group_size
    if padded_width != width:
        source = F.pad(source, (0, padded_width - width))
    grouped = source.view(token_count, group_count, group_size)
    raw_scales = grouped.abs().amax(dim=-1) / 127.0
    positive = raw_scales > 0

    try:
        scale_dtype, scale_rounding = scale_encoding.rsplit("_", 1)
    except ValueError as exc:
        raise ValueError(f"unsupported scale encoding {scale_encoding}") from exc
    if scale_dtype == "float16":
        stored = raw_scales.to(torch.float16)
    elif scale_dtype == "bfloat16":
        stored = raw_scales.to(torch.bfloat16)
    elif scale_dtype == "float32":
        stored = raw_scales
    else:
        raise ValueError(f"unsupported scale dtype {scale_dtype}")
    if scale_rounding == "ceil":
        # RNE can round a max-derived scale down, so its actual stored value
        # clips the source maximum.  The format can keep the identical FP16 or
        # BF16 payload while selecting the next representable positive scale.
        lower_than_raw = stored.to(torch.float32) < raw_scales
        stored = torch.where(
            lower_than_raw,
            torch.nextafter(stored, torch.full_like(stored, float("inf"))),
            stored,
        )
    elif scale_rounding != "rne":
        raise ValueError(f"unsupported scale rounding {scale_rounding}")
    stored_f32 = stored.to(torch.float32)
    invalid_stored = positive & ((stored_f32 <= 0) | ~torch.isfinite(stored_f32))
    # A zero/inf stored scale is a format error, not an excuse to divide by
    # zero.  The code is made zero and the condition is reported explicitly.
    safe_scales = torch.where(invalid_stored, torch.ones_like(stored_f32), stored_f32)
    pre_round = grouped / safe_scales.unsqueeze(-1)
    pre_round = torch.where(invalid_stored.unsqueeze(-1), torch.zeros_like(pre_round), pre_round)
    clipping = ((pre_round > 127.0) | (pre_round < -127.0)).sum()
    codes = torch.clamp(torch.round(pre_round), -127, 127).to(torch.int8)
    saturation = (codes.abs() == 127).sum()
    reconstructed = (codes.to(torch.float32) * stored_f32.unsqueeze(-1)).reshape(token_count, padded_width)
    return QuantizedBlocks(
        codes=codes.reshape(token_count, padded_width),
        scales=stored_f32,
        reconstructed=reconstructed[:, :width],
        original_width=width,
        padded_width=padded_width,
        positive_scale_count=int(positive.sum().item()),
        stored_zero_positive_scale_count=int((positive & (stored_f32 == 0)).sum().item()),
        stored_nonfinite_scale_count=int((positive & ~torch.isfinite(stored_f32)).sum().item()),
        clipping_count=int(clipping.item()),
        saturation_count=int(saturation.item()),
    )


@dataclass
class ErrorAccumulator:
    value_count: int = 0
    reference_sumsq: float = 0.0
    error_sumsq: float = 0.0
    absolute_error_sum: float = 0.0
    max_absolute_error: float = 0.0
    clipping_count: int = 0
    saturation_count: int = 0
    positive_scale_count: int = 0
    stored_zero_positive_scale_count: int = 0
    stored_nonfinite_scale_count: int = 0

    def add(self, reference: torch.Tensor, quantized: QuantizedBlocks) -> None:
        if tuple(reference.shape) != tuple(quantized.reconstructed.shape):
            raise ValueError("reference/reconstructed shape mismatch")
        error = quantized.reconstructed - reference.to(torch.float32)
        self.value_count += int(reference.numel())
        self.reference_sumsq += float(reference.to(torch.float64).square().sum().item())
        self.error_sumsq += float(error.to(torch.float64).square().sum().item())
        self.absolute_error_sum += float(error.abs().to(torch.float64).sum().item())
        if error.numel():
            self.max_absolute_error = max(self.max_absolute_error, float(error.abs().amax().item()))
        self.clipping_count += quantized.clipping_count
        self.saturation_count += quantized.saturation_count
        self.positive_scale_count += quantized.positive_scale_count
        self.stored_zero_positive_scale_count += quantized.stored_zero_positive_scale_count
        self.stored_nonfinite_scale_count += quantized.stored_nonfinite_scale_count

    def as_dict(self) -> dict[str, Any]:
        relative_l2 = math.sqrt(self.error_sumsq / self.reference_sumsq) if self.reference_sumsq else 0.0
        return {
            "value_count": self.value_count,
            "reference_sumsq": self.reference_sumsq,
            "error_sumsq": self.error_sumsq,
            "relative_l2_error": relative_l2,
            "mean_absolute_error": self.absolute_error_sum / self.value_count if self.value_count else 0.0,
            "maximum_absolute_error": self.max_absolute_error,
            "true_clipping_rate": self.clipping_count / self.value_count if self.value_count else 0.0,
            "edge_code_rate": self.saturation_count / self.value_count if self.value_count else 0.0,
            "positive_scale_count": self.positive_scale_count,
            "stored_zero_positive_scale_count": self.stored_zero_positive_scale_count,
            "stored_nonfinite_scale_count": self.stored_nonfinite_scale_count,
        }


@dataclass
class OutlierBins:
    group_count: list[int] = field(default_factory=lambda: [0] * 4)
    value_count: list[int] = field(default_factory=lambda: [0] * 4)
    reference_sumsq: list[float] = field(default_factory=lambda: [0.0] * 4)
    error_sumsq: list[float] = field(default_factory=lambda: [0.0] * 4)
    max_ratio: float = 0.0

    def add(self, reference: torch.Tensor, quantized: QuantizedBlocks, group_size: int) -> None:
        padded = F.pad(reference.to(torch.float32), (0, quantized.padded_width - reference.shape[1]))
        grouped = padded.view(reference.shape[0], -1, group_size)
        reconstruction = F.pad(
            quantized.reconstructed, (0, quantized.padded_width - quantized.original_width)
        ).view(reference.shape[0], -1, group_size)
        rms = grouped.square().mean(dim=-1).sqrt()
        ratios = torch.where(rms > 0, grouped.abs().amax(dim=-1) / rms, torch.zeros_like(rms))
        group_error_sse = (reconstruction - grouped).to(torch.float64).square().sum(dim=-1)
        group_reference_sse = grouped.to(torch.float64).square().sum(dim=-1)
        self.max_ratio = max(self.max_ratio, float(ratios.max().item()))
        bucket = torch.bucketize(ratios.reshape(-1), torch.tensor(OUTLIER_BIN_EDGES[1:], device=ratios.device))
        for index in range(4):
            mask = bucket == index
            if not bool(mask.any()):
                continue
            self.group_count[index] += int(mask.sum().item())
            # This is exact for the models measured here (all K are multiples
            # of 32); tails are separately recorded in the format design.
            self.value_count[index] += int(mask.sum().item()) * group_size
            self.reference_sumsq[index] += float(group_reference_sse.reshape(-1)[mask].sum().item())
            self.error_sumsq[index] += float(group_error_sse.reshape(-1)[mask].sum().item())

    def as_dict(self) -> dict[str, Any]:
        labels = ("[1,2)", "[2,4)", "[4,8)", "[8,inf)")
        rows = []
        for index, label in enumerate(labels):
            rel_l2 = (
                math.sqrt(self.error_sumsq[index] / self.reference_sumsq[index])
                if self.reference_sumsq[index]
                else 0.0
            )
            rows.append(
                {
                    "max_abs_over_rms_bin": label,
                    "group_count": self.group_count[index],
                    "value_count": self.value_count[index],
                    "relative_l2_error": rel_l2,
                }
            )
        return {"maximum_group_max_abs_over_rms": self.max_ratio, "bins": rows}


@dataclass
class LinearOutputAccumulator:
    output_value_count: int = 0
    reference_sumsq: float = 0.0
    errors_sumsq: dict[str, float] = field(
        default_factory=lambda: {"activation_only": 0.0, "w8a16": 0.0, "w8a8": 0.0}
    )
    errors_absolute_sum: dict[str, float] = field(
        default_factory=lambda: {"activation_only": 0.0, "w8a16": 0.0, "w8a8": 0.0}
    )
    errors_max_abs: dict[str, float] = field(
        default_factory=lambda: {"activation_only": 0.0, "w8a16": 0.0, "w8a8": 0.0}
    )

    def add(self, reference: torch.Tensor, candidates: dict[str, torch.Tensor]) -> None:
        self.output_value_count += int(reference.numel())
        self.reference_sumsq += float(reference.to(torch.float64).square().sum().item())
        for name, value in candidates.items():
            error = value - reference
            self.errors_sumsq[name] += float(error.to(torch.float64).square().sum().item())
            self.errors_absolute_sum[name] += float(error.abs().to(torch.float64).sum().item())
            if error.numel():
                self.errors_max_abs[name] = max(self.errors_max_abs[name], float(error.abs().amax().item()))

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "output_value_count": self.output_value_count,
            "reference_sumsq": self.reference_sumsq,
        }
        for name in sorted(self.errors_sumsq):
            result[name] = {
                "error_sumsq": self.errors_sumsq[name],
                "relative_l2_error": (
                    math.sqrt(self.errors_sumsq[name] / self.reference_sumsq)
                    if self.reference_sumsq
                    else 0.0
                ),
                "mean_absolute_error": (
                    self.errors_absolute_sum[name] / self.output_value_count
                    if self.output_value_count
                    else 0.0
                ),
                "maximum_absolute_error": self.errors_max_abs[name],
            }
        return result


@dataclass
class WeightSample:
    row_indices: torch.Tensor
    reference_rows: torch.Tensor
    quantized: QuantizedBlocks
    quantized_rows: torch.Tensor
    weight_relative_l2_error: float


class TensorMeasurement:
    def __init__(self, name: str, input_features: int, group_sizes: tuple[int, ...], primary_group: int):
        self.name = name
        self.input_features = input_features
        self.group_sizes = group_sizes
        self.primary_group = primary_group
        self.activation = {
            (group_size, scale_encoding): ErrorAccumulator()
            for group_size in group_sizes
            for scale_encoding in SCALE_ENCODINGS
        }
        self.outlier_bins = OutlierBins()
        self.channel_sumsq = torch.zeros(input_features, dtype=torch.float64)
        self.channel_max_abs = torch.zeros(input_features, dtype=torch.float32)
        self.channel_token_count = 0
        self.sampled_token_count = 0
        self.output = LinearOutputAccumulator()
        self.weight_sample: WeightSample | None = None

    def add_activation(self, values: torch.Tensor) -> dict[tuple[int, str], QuantizedBlocks]:
        source = values.to(torch.float32)
        if source.ndim != 2 or source.shape[1] != self.input_features:
            raise ValueError(f"{self.name}: unexpected activation shape {tuple(source.shape)}")
        self.sampled_token_count += int(source.shape[0])
        self.channel_sumsq += source.to(torch.float64).square().sum(dim=0).cpu()
        self.channel_max_abs = torch.maximum(self.channel_max_abs, source.abs().amax(dim=0).cpu())
        self.channel_token_count += int(source.shape[0])
        quantized: dict[tuple[int, str], QuantizedBlocks] = {}
        for group_size in self.group_sizes:
            for scale_encoding in SCALE_ENCODINGS:
                item = quantize_symmetric_per_token(source, group_size, scale_encoding)
                self.activation[(group_size, scale_encoding)].add(source, item)
                quantized[(group_size, scale_encoding)] = item
        primary = quantized[(self.primary_group, "float16_ceil")]
        self.outlier_bins.add(source, primary, self.primary_group)
        return quantized

    def ensure_weight_sample(self, module: torch.nn.Linear, output_rows: int) -> WeightSample:
        if self.weight_sample is not None:
            return self.weight_sample
        if module.weight.ndim != 2 or module.weight.shape[1] != self.input_features:
            raise ValueError(f"{self.name}: unexpected linear weight shape {tuple(module.weight.shape)}")
        rows = select_evenly(int(module.weight.shape[0]), output_rows, module.weight.device)
        reference_rows = module.weight.detach().index_select(0, rows).to(torch.float32).contiguous()
        quantized = quantize_symmetric_per_token(reference_rows, self.primary_group, "float16_ceil")
        reconstructed = quantized.reconstructed
        denominator = float(reference_rows.to(torch.float64).square().sum().item())
        numerator = float((reconstructed - reference_rows).to(torch.float64).square().sum().item())
        self.weight_sample = WeightSample(
            row_indices=rows.cpu(),
            reference_rows=reference_rows,
            quantized=quantized,
            quantized_rows=reconstructed,
            weight_relative_l2_error=math.sqrt(numerator / denominator) if denominator else 0.0,
        )
        return self.weight_sample

    def as_dict(self) -> dict[str, Any]:
        channel_rms = (self.channel_sumsq / self.channel_token_count).sqrt() if self.channel_token_count else self.channel_sumsq
        channel_ratio = torch.where(channel_rms > 0, self.channel_max_abs.to(torch.float64) / channel_rms, torch.zeros_like(channel_rms))
        if channel_ratio.numel():
            top_index = int(channel_ratio.argmax().item())
            sorted_ratio = torch.sort(channel_ratio).values
            p99_index = min(sorted_ratio.numel() - 1, max(0, math.ceil(sorted_ratio.numel() * 0.99) - 1))
            channel_outlier = {
                "top_channel_index": top_index,
                "top_channel_max_abs_over_rms": float(channel_ratio[top_index].item()),
                "top_channel_max_abs": float(self.channel_max_abs[top_index].item()),
                "median_channel_max_abs_over_rms": float(torch.median(sorted_ratio).item()),
                "p99_channel_max_abs_over_rms": float(sorted_ratio[p99_index].item()),
            }
        else:
            channel_outlier = {}
        activation = {
            f"k{group_size}_{scale_encoding}": self.activation[(group_size, scale_encoding)].as_dict()
            for group_size in self.group_sizes
            for scale_encoding in SCALE_ENCODINGS
        }
        result: dict[str, Any] = {
            "tensor": self.name,
            "input_features": self.input_features,
            "sampled_token_count": self.sampled_token_count,
            "activation": activation,
            "outlier_channel": channel_outlier,
            "outlier_group_distribution_primary_k32_fp16_ceil": self.outlier_bins.as_dict(),
            "linear_output_sample": self.output.as_dict(),
        }
        if self.weight_sample is not None:
            result["weight_sample"] = {
                "sampled_output_rows": int(self.weight_sample.row_indices.numel()),
                "row_indices": [int(value) for value in self.weight_sample.row_indices.tolist()],
                "weight_relative_l2_error": self.weight_sample.weight_relative_l2_error,
                "weight_scale_encoding": "float16_ceil",
                "weight_group_size": self.primary_group,
            }
        return result


def int8_block_dot(activation: QuantizedBlocks, weight: QuantizedBlocks) -> torch.Tensor:
    """Exact int32 K-block partials followed by FP32 scale products."""

    if activation.padded_width != weight.padded_width:
        raise ValueError("activation/weight padded K widths differ")
    group_size = activation.padded_width // activation.scales.shape[1]
    if group_size != weight.padded_width // weight.scales.shape[1]:
        raise ValueError("activation/weight group sizes differ")
    groups = activation.scales.shape[1]
    qa = activation.codes.view(activation.codes.shape[0], groups, group_size)
    qw = weight.codes.view(weight.codes.shape[0], groups, group_size)
    # [groups, tokens, K] @ [groups, K, rows] -> int32 partial dot products.
    partial = torch.bmm(
        qa.permute(1, 0, 2).contiguous().to(torch.int32),
        qw.permute(1, 2, 0).contiguous().to(torch.int32),
    )
    return (
        partial.to(torch.float32)
        * activation.scales.transpose(0, 1).unsqueeze(-1)
        * weight.scales.transpose(0, 1).unsqueeze(1)
    ).sum(dim=0)


def aggregate_error(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    reference_sumsq = 0.0
    error_sumsq = 0.0
    value_count = 0
    max_abs = 0.0
    for row in rows:
        output = row["linear_output_sample"]
        item = output[key]
        count = int(output["output_value_count"])
        reference_sumsq += float(output["reference_sumsq"])
        error_sumsq += float(item["error_sumsq"])
        value_count += count
        max_abs = max(max_abs, float(item["maximum_absolute_error"]))
    return {
        "sampled_output_value_count": value_count,
        "relative_l2_error": math.sqrt(error_sumsq / reference_sumsq) if reference_sumsq else 0.0,
        "maximum_absolute_error": max_abs,
        "aggregation_note": "exact aggregate of retained per-tensor output SSE and reference SSE",
    }


def aggregate_activation(rows: list[dict[str, Any]], group_size: int, scale_encoding: str) -> dict[str, Any]:
    value_count = 0
    error_sumsq = 0.0
    reference_sumsq = 0.0
    max_abs = 0.0
    true_clip_count_proxy = 0.0
    edge_count_proxy = 0.0
    zero_scales = 0
    nonfinite_scales = 0
    for row in rows:
        item = row["activation"][f"k{group_size}_{scale_encoding}"]
        count = int(item["value_count"])
        value_count += count
        error_sumsq += float(item["error_sumsq"])
        reference_sumsq += float(item["reference_sumsq"])
        max_abs = max(max_abs, float(item["maximum_absolute_error"]))
        true_clip_count_proxy += count * float(item["true_clipping_rate"])
        edge_count_proxy += count * float(item["edge_code_rate"])
        zero_scales += int(item["stored_zero_positive_scale_count"])
        nonfinite_scales += int(item["stored_nonfinite_scale_count"])
    return {
        "value_count": value_count,
        "relative_l2_error": math.sqrt(error_sumsq / reference_sumsq) if reference_sumsq else 0.0,
        "maximum_absolute_error": max_abs,
        "true_clipping_rate": true_clip_count_proxy / value_count if value_count else 0.0,
        "edge_code_rate": edge_count_proxy / value_count if value_count else 0.0,
        "stored_zero_positive_scale_count": zero_scales,
        "stored_nonfinite_scale_count": nonfinite_scales,
        "aggregation_note": "exact aggregate of retained per-tensor activation SSE and reference SSE",
    }


def make_measurement_hooks(
    measurements: dict[str, TensorMeasurement],
    module_pattern: re.Pattern[str],
    samples_per_call: int,
    output_rows: int,
    enabled: dict[str, bool],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str, measurement: TensorMeasurement):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not enabled["value"] or not inputs or not isinstance(module, torch.nn.Linear):
                return
            input_value = inputs[0]
            if not torch.is_tensor(input_value) or not input_value.is_floating_point():
                return
            if input_value.shape[-1] != measurement.input_features:
                raise ValueError(f"{name}: module input width changed")
            flat = input_value.detach().reshape(-1, input_value.shape[-1])
            rows = select_evenly(int(flat.shape[0]), samples_per_call, flat.device)
            sampled = flat.index_select(0, rows).to(torch.float32)
            quantized = measurement.add_activation(sampled)
            weight = measurement.ensure_weight_sample(module, output_rows)
            activation = quantized[(measurement.primary_group, "float16_ceil")]
            reference = sampled @ weight.reference_rows.t()
            activation_only = activation.reconstructed @ weight.reference_rows.t()
            w8a16 = sampled @ weight.quantized_rows.t()
            w8a8 = int8_block_dot(activation, weight.quantized)
            measurement.output.add(
                reference,
                {"activation_only": activation_only, "w8a16": w8a16, "w8a8": w8a8},
            )

        return hook

    for name, module in measurements["__model__"].named_modules():
        if not isinstance(module, torch.nn.Linear) or not module_pattern.search(name):
            continue
        measurement = TensorMeasurement(
            name=name,
            input_features=int(module.in_features),
            group_sizes=measurements["__group_sizes__"],
            primary_group=measurements["__primary_group__"],
        )
        measurements[name] = measurement
        handles.append(module.register_forward_pre_hook(make_hook(name, measurement)))
    if len(handles) == 0:
        raise RuntimeError("no Linear modules matched --module-pattern")
    return handles


def make_logit_quantization_hooks(
    model: torch.nn.Module,
    module_pattern: re.Pattern[str],
    group_size: int,
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
        if not inputs or not torch.is_tensor(inputs[0]) or not inputs[0].is_floating_point():
            return None
        original = inputs[0]
        flat = original.detach().reshape(-1, original.shape[-1]).to(torch.float32)
        quantized = quantize_symmetric_per_token(flat, group_size, "float16_ceil")
        return (quantized.reconstructed.reshape_as(original).to(dtype=original.dtype),)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module_pattern.search(name):
            handles.append(module.register_forward_pre_hook(hook))
    if not handles:
        raise RuntimeError("no modules matched while installing logit hooks")
    return handles


def parse_group_sizes(raw: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0 or value % 4 for value in values):
        raise argparse.ArgumentTypeError("group sizes must be positive multiples of 4")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--group-sizes", type=parse_group_sizes, default=DEFAULT_GROUP_SIZES)
    parser.add_argument("--primary-group", type=int, default=32)
    parser.add_argument("--samples-per-module-call", type=int, default=8)
    parser.add_argument("--output-rows-per-module", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=16)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--module-pattern", default=DEFAULT_MODULE_PATTERN)
    parser.add_argument("--logit-samples", type=int, default=1)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_model_measurement(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    torch.manual_seed(args.seed)
    model_args = SimpleNamespace(
        model_dir=args.model_dir,
        model_class="causal_lm",
        dtype="bfloat16",
        trust_remote_code=args.trust_remote_code,
        device="cpu",
    )
    tokenizer, model = COLLECTOR.load_transformers_model(model_args)
    device = next(model.parameters()).device
    if device.type != "cpu":
        raise RuntimeError(f"CPU-only probe refused non-CPU model device: {device}")
    if model.training:
        raise RuntimeError("model must be in eval mode")

    module_pattern = re.compile(args.module_pattern)
    measurements: dict[str, Any] = {
        "__model__": model,
        "__group_sizes__": args.group_sizes,
        "__primary_group__": args.primary_group,
    }
    enabled = {"value": True}
    handles = make_measurement_hooks(
        measurements, module_pattern, args.samples_per_module_call, args.output_rows_per_module, enabled
    )
    samples_seen = 0
    tokens_seen = 0
    domain_counts: Counter[str] = Counter()
    record_ids: list[str] = []
    reference_logit_batch: dict[str, torch.Tensor] | None = None
    reference_logits: torch.Tensor | None = None
    reference_logit_positions: torch.Tensor | None = None
    try:
        examples_iter = iter(COLLECTOR.iter_examples(args.prompt_file))
        with torch.inference_mode():
            while samples_seen < args.max_samples:
                try:
                    example = next(examples_iter)
                except StopIteration:
                    break
                batch, _ = COLLECTOR.encode_examples(tokenizer, [example], args.sequence_length, False)
                batch = {key: value.to(device) for key, value in batch.items()}
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    tokens_seen += int(attention_mask.sum().item())
                else:
                    tokens_seen += int(batch["input_ids"].numel())
                outputs = model(**batch, use_cache=False)
                if samples_seen < args.logit_samples:
                    logits = getattr(outputs, "logits", None)
                    if logits is None:
                        raise RuntimeError("causal-LM output does not expose logits")
                    valid = int(attention_mask[0].sum().item()) if attention_mask is not None else logits.shape[1]
                    start = max(0, valid - min(valid, 16))
                    reference_logit_batch = {key: value.detach().clone() for key, value in batch.items()}
                    reference_logits = logits[0, start:valid].detach().to(torch.float32).cpu()
                    reference_logit_positions = torch.arange(start, valid, dtype=torch.long)
                samples_seen += 1
                domain_counts[str(example.get("domain", "unknown"))] += 1
                record_ids.append(str(example["record_id"]))
                print(
                    json.dumps(
                        {
                            "event": "sample_complete",
                            "samples_seen": samples_seen,
                            "tokens_seen": tokens_seen,
                            "record_id": record_ids[-1],
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()
    measurements.pop("__model__")
    measurements.pop("__group_sizes__")
    measurements.pop("__primary_group__")
    rows = [measurements[name].as_dict() for name in sorted(measurements)]

    logit_result: dict[str, Any] = {"status": "not_run"}
    if reference_logit_batch is not None and reference_logits is not None and reference_logit_positions is not None:
        quant_handles = make_logit_quantization_hooks(model, module_pattern, args.primary_group)
        try:
            with torch.inference_mode():
                quantized_outputs = model(**reference_logit_batch, use_cache=False)
            quantized_logits = getattr(quantized_outputs, "logits", None)
            if quantized_logits is None:
                raise RuntimeError("quantized causal-LM output does not expose logits")
            candidate = quantized_logits[0].index_select(0, reference_logit_positions.to(quantized_logits.device)).to(torch.float32).cpu()
            delta = candidate - reference_logits
            denom = float(reference_logits.to(torch.float64).square().sum().item())
            rel_l2 = math.sqrt(float(delta.to(torch.float64).square().sum().item()) / denom) if denom else 0.0
            reference_top1 = reference_logits.argmax(dim=-1)
            candidate_top1 = candidate.argmax(dim=-1)
            reference_log_prob = torch.log_softmax(reference_logits, dim=-1)
            candidate_log_prob = torch.log_softmax(candidate, dim=-1)
            kl = torch.sum(reference_log_prob.exp() * (reference_log_prob - candidate_log_prob), dim=-1)
            logit_result = {
                "status": "completed_activation_only",
                "scope": "all selected Linear inputs dynamically int8/F16-scale quantized; weights remain BF16",
                "prompt_count": 1,
                "token_positions": [int(value) for value in reference_logit_positions.tolist()],
                "logit_value_count": int(delta.numel()),
                "relative_l2_error": rel_l2,
                "mean_absolute_error": float(delta.abs().mean().item()),
                "maximum_absolute_error": float(delta.abs().amax().item()),
                "mean_token_kl_reference_to_activation_quantized": float(kl.mean().item()),
                "top1_match_count": int((reference_top1 == candidate_top1).sum().item()),
                "top1_total": int(reference_top1.numel()),
            }
        finally:
            for handle in quant_handles:
                handle.remove()
    run_info = {
        "samples_seen": samples_seen,
        "tokens_seen": tokens_seen,
        "domain_counts": dict(sorted(domain_counts.items())),
        "processed_record_ids_sha256": hashlib.sha256("\n".join(record_ids).encode("utf-8")).hexdigest(),
        "module_count": len(rows),
    }
    return rows, run_info, logit_result


def render_report(summary: dict[str, Any]) -> str:
    aggregate = summary["aggregate_activation_error"]
    output = summary["aggregate_linear_output_error"]
    lines = [
        "# SQ8_1 W8A8 activation quantization error — CPU measurement",
        "",
        "## Scope",
        "",
        "- CPU-only real-Qwen3.5-9B forward pre-hook measurement; no HIP runtime API or GPU was used.",
        "- The existing `tools/collect-activation-stats.py` loader, corpus parser, and Linear-module convention were reused.",
        "- Raw activations were quantized in-process and discarded. This directory contains only aggregate/error evidence.",
        "- `SQ8_1` activation rule: per-token, contiguous K block, symmetric signed int8, RNE codes, `s=max(abs(x))/127`, stored FP16 scale rounded upward to a representable value.",
        "",
        "## Activation error",
        "",
        "| K | scale | sampled values | relative L2 | max abs error | true clipping rate | edge-code rate |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, item in aggregate.items():
        lines.append(
            f"| {key.split('_')[0][1:]} | {key.split('_', 1)[1]} | {item['value_count']} | "
            f"{item['relative_l2_error']:.8g} | {item['maximum_absolute_error']:.8g} | "
            f"{item['true_clipping_rate']:.8g} | {item['edge_code_rate']:.8g} |"
        )
    lines.extend(
        [
            "",
            "`true clipping` counts values outside the post-storage-scale [-127,127] range before clamp. "
            "`edge-code` is the fraction represented by ±127 and is intentionally reported separately. The `ceil` scale policy preserves scale positivity and avoids an RNE-down-rounding clip without changing scale bytes.",
            "",
            "## Sampled linear-output error",
            "",
            "Each selected Linear tensor uses deterministic evenly spaced raw-token and output-row samples. "
            "`W8A16` quantizes only weights; `W8A8` uses int32 block dots and applies the upward-rounded FP16 activation and weight scales once per K=32 partial.",
            "",
            "| path | sampled outputs | relative L2 | max abs error |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, item in output.items():
        lines.append(
            f"| {name} | {item['sampled_output_value_count']} | {item['relative_l2_error']:.8g} | {item['maximum_absolute_error']:.8g} |"
        )
    logit = summary["activation_only_logit_impact"]
    lines.extend(["", "## Activation-only logit smoke", ""])
    if logit["status"] == "completed_activation_only":
        lines.extend(
            [
                f"- Relative L2: `{logit['relative_l2_error']:.8g}`; max abs: `{logit['maximum_absolute_error']:.8g}`; mean KL: `{logit['mean_token_kl_reference_to_activation_quantized']:.8g}`.",
                f"- Top-1 matches: `{logit['top1_match_count']}/{logit['top1_total']}`.",
                "- This is activation-only. Full-model `SQ8_1` W8A8 logits remain unmeasured here because no production weight reader/kernel was implemented.",
            ]
        )
    else:
        lines.append("- Not run; see `summary.json`.")
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "The exact command line, file hashes, model/corpus provenance, selected-row rule, thread settings, and tool hashes are in `measurement-manifest.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.max_samples < 1 or args.sequence_length < 1:
        raise SystemExit("--max-samples and --sequence-length must be positive")
    if args.primary_group not in args.group_sizes:
        raise SystemExit("--primary-group must be one of --group-sizes")
    if args.primary_group != 32 or 32 not in args.group_sizes:
        raise SystemExit("this SQ8_1 diagnostic fixes its primary path at K=32")
    if args.samples_per_module_call < 1 or args.output_rows_per_module < 1:
        raise SystemExit("sampling limits must be positive")
    if args.torch_threads < 1 or args.torch_interop_threads < 1:
        raise SystemExit("thread counts must be positive")
    if args.logit_samples < 0:
        raise SystemExit("--logit-samples must be non-negative")

    for attr in ("model_dir", "prompt_file", "corpus_manifest", "output_dir"):
        setattr(args, attr, getattr(args, attr).expanduser().resolve())
    if not args.model_dir.is_dir():
        raise SystemExit(f"missing model directory: {args.model_dir}")
    for path in (args.prompt_file, args.corpus_manifest):
        if not path.is_file():
            raise SystemExit(f"missing input file: {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Explicitly refuse a GPU torch build that has moved the model to an
    # accelerator. The caller also passes CPU-only environment masks.
    if torch.cuda.is_available():
        raise SystemExit("CPU-only measurement refuses an available torch CUDA/HIP device")

    implementation_sha = sha256_file(Path(__file__).resolve())
    collector_sha = sha256_file(Path(__file__).resolve().with_name("collect-activation-stats.py"))
    start_revision = COLLECTOR.git_revision()
    rows, run_info, logit_result = run_model_measurement(args)

    aggregate_activation_error = {
        f"k{group_size}_{scale_encoding}": aggregate_activation(rows, group_size, scale_encoding)
        for group_size in args.group_sizes
        for scale_encoding in SCALE_ENCODINGS
    }
    aggregate_linear_output_error = {
        name: aggregate_error(rows, name) for name in ("activation_only", "w8a16", "w8a8")
    }
    outliers = sorted(
        (
            {
                "tensor": row["tensor"],
                **row["outlier_channel"],
                "tensor_k32_fp16_ceil_relative_l2": row["activation"]["k32_float16_ceil"]["relative_l2_error"],
                "tensor_k32_fp16_ceil_edge_code_rate": row["activation"]["k32_float16_ceil"]["edge_code_rate"],
            }
            for row in rows
        ),
        key=lambda item: item.get("top_channel_max_abs_over_rms", 0.0),
        reverse=True,
    )
    manifest = {
        "schema_version": "ullm.sq8_1-w8a8-activation-error.v0.1",
        "timestamp_utc": utc_now(),
        "status": "completed",
        "cpu_only": True,
        "gpu_execution": "not used",
        "command": sys.argv,
        "cwd": str(Path.cwd()),
        "git_revision_at_start": start_revision,
        "git_revision_at_finalize": COLLECTOR.git_revision(),
        "measurement_tool": {"path": str(Path(__file__).resolve()), "sha256": implementation_sha},
        "reused_activation_collector": {
            "path": str(Path(__file__).resolve().with_name("collect-activation-stats.py")),
            "sha256": collector_sha,
            "reused_functions": ["load_transformers_model", "iter_examples", "encode_examples", "sha256_file", "git_revision"],
        },
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
        },
        "quantization": {
            "format": "SQ8_1 diagnostic W8A8",
            "symmetric": True,
            "zero_point": None,
            "rounding": "torch.round nearest-even (RNE)",
            "code_range": [-127, 127],
            "activation_scale": "per-token contiguous K block max(abs(x))/127, stored FP16 rounded upward unless table says otherwise",
            "weight_scale": "per-output-row contiguous K=32 block max(abs(w))/127, stored FP16 rounded upward",
            "group_sizes_evaluated": list(args.group_sizes),
            "primary_group_size": args.primary_group,
            "raw_activation_storage": "not retained; values are aggregated inside forward pre-hooks",
        },
        "sampling": {
            "module_pattern": args.module_pattern,
            "activation_token_rows_per_module_call": args.samples_per_module_call,
            "output_rows_per_module": args.output_rows_per_module,
            "activation_row_selection": "evenly spaced valid flattened rows within each single-example forward",
            "output_row_selection": "evenly spaced output rows of each Linear weight matrix",
            "linear_output_scope": "sampled matmul without bias; W8A8 uses exact int32 per-K partial dots then FP32 scale products",
        },
        "run": {**run_info, "sequence_length": args.sequence_length, "torch_threads": args.torch_threads, "torch_interop_threads": args.torch_interop_threads, "seed": args.seed},
    }
    summary = {
        "schema_version": manifest["schema_version"],
        "measurement_manifest": "measurement-manifest.json",
        "aggregate_activation_error": aggregate_activation_error,
        "aggregate_linear_output_error": aggregate_linear_output_error,
        "activation_only_logit_impact": logit_result,
        "top_outlier_channels": outliers[:32],
        "coverage": {"tensor_count": len(rows), "sampled_tokens_per_tensor": {row["tensor"]: row["sampled_token_count"] for row in rows}},
    }
    (args.output_dir / "measurement-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per-tensor.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "top-outlier-channels.json").write_text(
        json.dumps(outliers[:32], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"event": "complete", "output_dir": str(args.output_dir), "module_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
