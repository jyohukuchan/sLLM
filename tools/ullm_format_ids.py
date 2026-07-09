"""Canonical uLLM public format identifiers and legacy aliases."""

from __future__ import annotations


FORMAT_AQ4_0 = "AQ4_0"
FORMAT_SQ8_0 = "SQ8_0"


def canonical_format_id(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    lower = text.lower()
    if lower in {"aq4", "aq4_0", "aq4-prototype-current-runtime"} or lower.startswith("aq4_"):
        return FORMAT_AQ4_0
    if lower in {"sq", "sq8_0", "sq-format-v0.1"} or lower.startswith("sq-fp8"):
        return FORMAT_SQ8_0
    return None


def canonical_or_original(value: str) -> str:
    return canonical_format_id(value) or value


def is_legacy_alias(value: str, canonical: str) -> bool:
    return value != canonical and canonical_format_id(value) == canonical
