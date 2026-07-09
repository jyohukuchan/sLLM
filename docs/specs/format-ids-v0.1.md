# uLLM Format IDs v0.1

## Purpose

This document defines the public format identifiers used by uLLM runtime records and artifacts.

## Public IDs

| ID | Meaning |
| --- | --- |
| `AQ4_0` | Existing AQ4 format, version 0. The `4` denotes approximate bits per parameter. |
| `SQ8_0` | FP8 E4M3 SQ format, version 0. The `8` denotes approximate bits per parameter. |

## Legacy Aliases

Existing result and prototype strings are accepted as aliases so old artifacts remain readable.

`AQ4_0` aliases include:

- `aq4`
- `aq4-prototype-current-runtime`
- existing internal codebook candidates beginning with `aq4_`

`SQ8_0` aliases include:

- `sq`
- `sq-format-v0.1`
- implementation lineage IDs beginning with `sq-fp8`, such as `sq-fp8-w8a16-r9700-v0`

## SQ8_0 Validation Meaning

`SQ8_0` is an adopted FP8 E4M3 format. Quality guards comparing SQ8_0 against AQ4_0 are diagnostic
regression checks, not proof that FP8 itself is an acceptable quantization family.

An SQ8_0 implementation is valid when the artifact is decoded and executed according to the SQ8_0
format specification and the relevant implementation regression checks pass.
