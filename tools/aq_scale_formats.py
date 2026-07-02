#!/usr/bin/env python3
"""Shared AQ local-scale format helpers."""

from __future__ import annotations

import re

import torch


_UE_FORMAT_RE = re.compile(r"^u?e(\d+)m(\d+)$")


def parse_unsigned_em_format(scale_format: str) -> tuple[int, int] | None:
    """Parse positive E/M scale formats.

    In AQ local-scale context, both ``e4m3`` and ``ue4m3`` name unsigned,
    positive-only scale tables. The ``u`` prefix is preferred in docs when the
    sign-bit policy matters.
    """

    match = _UE_FORMAT_RE.fullmatch(scale_format.lower())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def decode_e8m0() -> torch.Tensor:
    codes = torch.arange(0, 255, dtype=torch.float32)
    return torch.pow(torch.tensor(2.0, dtype=torch.float32), codes - 127.0)


def decode_unsigned_em(exp_bits: int, mant_bits: int) -> torch.Tensor:
    if exp_bits < 1:
        raise ValueError("exp_bits must be >= 1")
    if mant_bits < 0:
        raise ValueError("mant_bits must be >= 0")

    bias = (1 << (exp_bits - 1)) - 1
    max_exp = (1 << exp_bits) - 1
    mant_count = 1 << mant_bits
    values: list[float] = []
    for exp in range(max_exp):
        for mant in range(mant_count):
            if exp == 0:
                if mant == 0:
                    continue
                value = (mant / float(mant_count)) * (2.0 ** (1 - bias))
            else:
                value = (1.0 + mant / float(mant_count)) * (2.0 ** (exp - bias))
            values.append(value)
    return torch.tensor(sorted(set(values)), dtype=torch.float32)


def scale_values(scale_format: str) -> torch.Tensor:
    fmt = scale_format.lower()
    if fmt == "e8m0":
        return decode_e8m0()
    parsed = parse_unsigned_em_format(fmt)
    if parsed is not None:
        exp_bits, mant_bits = parsed
        return decode_unsigned_em(exp_bits, mant_bits)
    raise ValueError(f"unknown scale format: {scale_format}")


def scale_format_dominates(upper: str, lower: str, *, strict: bool = True) -> bool:
    """Return whether an unsigned E/M format should contain another."""

    upper_parsed = parse_unsigned_em_format(upper)
    lower_parsed = parse_unsigned_em_format(lower)
    if upper_parsed is None or lower_parsed is None:
        return False
    upper_exp, upper_mant = upper_parsed
    lower_exp, lower_mant = lower_parsed
    if upper_exp < lower_exp or upper_mant < lower_mant:
        return False
    if strict and upper_exp == lower_exp and upper_mant == lower_mant:
        return False
    return True


def scale_complexity(scale_format: str) -> tuple[int, int, str]:
    parsed = parse_unsigned_em_format(scale_format)
    if parsed is None:
        return (999, 999, scale_format)
    return (parsed[0], parsed[1], scale_format)


def scale_subset_index_map(source_scales: torch.Tensor, target_scales: torch.Tensor) -> torch.Tensor:
    """Map every source scale value to an exactly matching target index."""

    target_by_value = {float(value): idx for idx, value in enumerate(target_scales.to(torch.float32).tolist())}
    mapped: list[int] = []
    missing: list[float] = []
    for value in source_scales.to(torch.float32).tolist():
        idx = target_by_value.get(float(value))
        if idx is None:
            missing.append(float(value))
            continue
        mapped.append(idx)
    if missing:
        preview = ", ".join(f"{value:.9g}" for value in missing[:8])
        raise ValueError(f"target scale table does not contain {len(missing)} source values: {preview}")
    return torch.tensor(mapped, dtype=torch.long)
