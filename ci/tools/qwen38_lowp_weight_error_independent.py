#!/usr/bin/env python3
"""Independent NumPy measurement of BF16-relative weight quantization error.

Clean-room stage-0 evidence for the low-precision scale-selection plan
(``docs/plans/active/2026/09/11-20/low-precision-scale-selection.md``).

Independence: every quantization rule and every error number below is
implemented here from the format specification.  This tool deliberately does
not import or read ``ci/tools/qwen38_weight_error.py``.  GGUF files are opened
only to read the ``sllm.tensor_recipe`` field (which logical tensors are
quantized, with which encoding) and the tensor type table; no GGUF payload is
used to compute an error.  Errors are recomputed from the BF16 source weights.

Rules implemented (the block axis is the last / K axis):

  MXFP8 (E4M3FN elements, 32-element blocks, shared E8M0 micro-scale)
    old / floor       scale = 2**(floor(log2(amax)) - 8)              (clips)
    new / no-clipping smallest 2**k with 2**k * 448 >= amax
  MXFP6 (E3M2 elements, 32-element blocks, shared E8M0 micro-scale)
    old / floor       scale = 2**(floor(log2(amax)) - 4)
    new / no-clipping smallest 2**k with 2**k * 28 >= amax
    MXFP4 (E2M1 elements, 32-element blocks, shared E8M0 micro-scale)
    floor  2**(floor(log2(amax)) - 2)
    ceil   2**(ceil(log2(amax)) - 2)   (may overshoot by up to one octave)
    even   floor, exponent bumped by one when the block max mantissa >= 1.75
    best   per block the smaller-SSE of floor and floor+1 (tie -> floor)
    no_clip smallest 2**k with 2**k * 6 >= amax (smallest non-saturating
           power of two; reported so the earlier "ceil ~= 11.8%" scratch number
           can be attributed to a rule)
  NVFP4 (E2M1 elements, 16-element blocks, E4M3FN per-block scale, fp32
         per-tensor global scale g = amax_tensor / 6 / 448 for weights; the
         global scale is never changed)
    old / nearest       code = round-to-nearest-even E4M3(amax_block / 6 / g)
    new / adjacent-best better of the two E4M3 codes adjacent to
                        amax_block / 6 / g by block SSE (tie -> smaller code)
    intermediate        upper (ceil) adjacent code, and the scale codes stored
                        in the NVFP4 artifact itself

Units: rel-RMS = sqrt(mean((q - w)**2)) / sqrt(mean(w**2)) with w the BF16
weight; saturation is the fraction of quantized elements that land on the
element format's maximum finite magnitude.

Everything is streamed one tensor at a time in whole-block chunks, so all
elements are processed by default.  ``--sample-row-stride`` optionally keeps
whole rows only, which never splits a 32/16-element quantization block.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
import time

import numpy as np

# --------------------------------------------------------------------------
# Format definitions
# --------------------------------------------------------------------------

U32 = np.uint32
MAG_MASK = U32(0x7FFFFFFF)

E8M0_BIAS = 127
E8M0_MAX_CODE = 254  # code 255 marks NaN and must never carry a finite scale

# element format -> (exponent bits, mantissa bits, exponent bias,
#                    max finite magnitude)
ELEM_FORMATS = {
    "e4m3fn": (4, 3, 7, 448.0),
    "e3m2": (3, 2, 3, 28.0),
    "e2m1": (2, 1, 1, 6.0),
}

MX_FORMATS = {
    "mxfp8": {
        "elem": "e4m3fn",
        "block": 32,
        "emax": 8,
        "gguf_encoding": "mxfp8-e4m3-block32-e8m0",
    },
    "mxfp6": {
        "elem": "e3m2",
        "block": 32,
        "emax": 4,
        "gguf_encoding": "mxfp6-e3m2-block32-e8m0",
    },
    "mxfp4": {"elem": "e2m1", "block": 32, "emax": 2, "gguf_encoding": None},
}

MX_RULES = {
    "mxfp8": ["old", "new"],
    "mxfp6": ["old", "new"],
    "mxfp4": ["floor", "ceil", "even", "best", "no_clip"],
}

# Which two rule names the headline "old -> new" columns of the report use.
REPORT_RULES = {
    "mxfp8": ("old", "new"),
    "mxfp6": ("old", "new"),
    "mxfp4": ("floor", "best"),
    "nvfp4": ("old_nearest", "new_adjacent_best"),
}

NVFP4_BLOCK = 16
NVFP4_RULES = [
    "old_nearest",
    "new_adjacent_best",
    "intermediate_ceil_adjacent",
    "intermediate_artifact_stored_scale",
]

GGUF_TYPE_NAMES = {24: "I8", 30: "BF16"}


def elem_parts(key):
    """(mantissa bits, minimum normal exponent, max finite magnitude)."""
    _exponent_bits, mantissa_bits, bias, max_mag = ELEM_FORMATS[key]
    return mantissa_bits, 1 - bias, max_mag


def element_codes(key):
    """Ascending (codes, values) for the non-negative finite magnitudes."""
    exponent_bits, mantissa_bits, bias, max_mag = ELEM_FORMATS[key]
    codes, values = [], []
    for code in range(1 << (exponent_bits + mantissa_bits)):
        exponent = code >> mantissa_bits
        mantissa = code & ((1 << mantissa_bits) - 1)
        if exponent == 0:
            value = mantissa * 2.0 ** (1 - bias - mantissa_bits)
        else:
            value = (1.0 + mantissa / float(1 << mantissa_bits)) * 2.0 ** (exponent - bias)
        if not math.isfinite(value) or value > max_mag:
            continue  # unrepresentable / NaN codes (E4M3FN code 0x7f)
        codes.append(code)
        values.append(value)
    return np.array(codes, dtype=np.int32), np.array(values, dtype=np.float64)


def element_code_value_table(key):
    """code -> magnitude for every code; NaN codes map to numpy nan."""
    exponent_bits, mantissa_bits = ELEM_FORMATS[key][0], ELEM_FORMATS[key][1]
    codes, values = element_codes(key)
    table = np.full(1 << (exponent_bits + mantissa_bits), np.nan, dtype=np.float64)
    table[codes] = values
    return table


# --------------------------------------------------------------------------
# Quantizers
# --------------------------------------------------------------------------


def quantize_rne(x, key):
    """Round the signed float32 array ``x`` to element format ``key``.

    Rounding is round-to-nearest-even and the magnitude saturates at the
    format's maximum finite magnitude.

    Returns ``(q, qmag)`` where ``q`` keeps the sign of ``x`` and ``qmag`` is
    the non-negative quantized magnitude (the format maximum for saturated
    elements).
    """
    mantissa_bits, emin, max_mag = elem_parts(key)
    x = np.ascontiguousarray(x, dtype=np.float32)
    shift = 23 - mantissa_bits
    magnitude = np.bitwise_and(x.view(U32), MAG_MASK)
    bits = magnitude + U32((1 << (shift - 1)) - 1)
    bits += np.bitwise_and(magnitude >> U32(shift), U32(1))
    np.bitwise_and(bits, U32((0xFFFFFFFF << shift) & 0xFFFFFFFF), out=bits)
    qmag = bits.view(np.float32)
    subnormal_bits = (emin + 127) << 23
    if int(magnitude.min()) < subnormal_bits:
        subnormal = magnitude < U32(subnormal_bits)
        quantized_subnormal = np.rint(
            np.abs(x[subnormal]) * np.float32(2.0 ** (mantissa_bits - emin))
        )
        quantized_subnormal *= np.float32(2.0 ** (emin - mantissa_bits))
        qmag = qmag.copy()
        qmag[subnormal] = quantized_subnormal
    np.minimum(qmag, np.float32(max_mag), out=qmag)
    return np.copysign(qmag, x), qmag


def floor_log2(magnitudes):
    """floor(log2(magnitude)) for a float32 array of non-negative magnitudes.

    Every non-zero magnitude produced by BF16 weights is a normal float32, so
    the exponent field gives the answer directly; 0.0 maps to -127, which is
    also the E8M0 code-0 exponent.
    """
    bits = np.ascontiguousarray(magnitudes, dtype=np.float32).view(U32)
    return np.bitwise_and(bits >> U32(23), U32(0xFF)).astype(np.int64) - 127


def ceil_log2(magnitudes):
    """ceil(log2(magnitude)); 0.0 maps to -127."""
    bits = np.ascontiguousarray(magnitudes, dtype=np.float32).view(U32)
    exponent = np.bitwise_and(bits >> U32(23), U32(0xFF)).astype(np.int64) - 127
    exact_power_of_two = np.bitwise_and(bits, U32(0x7FFFFF)) == 0
    return np.where(exact_power_of_two, exponent, exponent + 1)


def no_clip_exponents(magnitudes, max_mag):
    """Smallest integer k with 2**k * max_mag >= magnitude.

    An exact power of two takes the exact (smaller) exponent, so this is the
    smallest E8M0 value that does not saturate, with no overshoot.
    """
    magnitudes = np.ascontiguousarray(magnitudes, dtype=np.float32)
    ratio = magnitudes.astype(np.float64) / np.float64(max_mag)
    mantissa, exponent = np.frexp(ratio)
    candidate = np.where(mantissa == 0.5, exponent - 1, exponent).astype(np.int64)
    return np.where(magnitudes.view(U32) == 0, -127, candidate)


def e8m0_pairs(exponents):
    """(scale, inverse_scale, clamped_count) for an exponent array.

    The E8M0 code is clamped into [0, 254]; code 255 (NaN) is never produced.
    """
    code = exponents + E8M0_BIAS
    clamped = (code < 0) | (code > E8M0_MAX_CODE)
    code = np.clip(code, 0, E8M0_MAX_CODE).astype(np.int32) - E8M0_BIAS
    scale = np.ldexp(np.float32(1.0), code)
    inverse = np.ldexp(np.float32(1.0), -code)
    return scale, inverse, int(np.count_nonzero(clamped))


# --------------------------------------------------------------------------
# Minimal safetensors reader (plain NumPy; no tensor framework needed)
# --------------------------------------------------------------------------


class SafeTensors:
    """Reader for the safetensors container (header JSON + raw data)."""

    DTYPES = {
        "F64": "<f8",
        "F32": "<f4",
        "F16": "<f2",
        "BF16": "<u2",
        "I64": "<i8",
        "I32": "<i4",
        "I16": "<i2",
        "I8": "i1",
        "U8": "u1",
        "U16": "<u2",
        "U32": "<u4",
        "U64": "<u8",
        "BOOL": "?",
        "F8_E4M3": "u1",
        "F8_E5M2": "u1",
    }

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            if not 0 < header_len < (1 << 30):
                raise ValueError("%s: implausible safetensors header length %d" % (path, header_len))
            header = json.loads(handle.read(header_len))
        self.data_start = 8 + header_len
        self.metadata = header.get("__metadata__", {})
        self.tensors = {k: v for k, v in header.items() if k != "__metadata__"}

    def names(self):
        return list(self.tensors)

    def dtype(self, name):
        return self.tensors[name]["dtype"]

    def shape(self, name):
        return tuple(self.tensors[name]["shape"])

    def raw(self, name):
        """Zero-copy memmap view of one tensor's bytes."""
        info = self.tensors[name]
        dtype = self.DTYPES[info["dtype"]]
        start, end = info["data_offsets"]
        count = (end - start) // np.dtype(dtype).itemsize
        expected = 1
        for dim in info["shape"]:
            expected *= dim
        if count != expected:
            raise ValueError("%s: byte range does not match shape" % name)
        view = np.memmap(
            self.path,
            dtype=dtype,
            mode="r",
            offset=self.data_start + start,
            shape=(count,),
        )
        return view.reshape(info["shape"])


def bf16_to_f32(values):
    """BF16 bit pattern (uint16) -> float32."""
    return (np.asarray(values, dtype=U32) << U32(16)).view(np.float32)


# --------------------------------------------------------------------------
# GGUF recipe / tensor type audit
# --------------------------------------------------------------------------


def read_gguf_recipe(path):
    """Return (recipe, bindings, tensor_type_map, tensor_count) for a GGUF."""
    import gguf  # third-party reader used only for GGUF metadata

    reader = gguf.GGUFReader(path)
    field = reader.fields.get("sllm.tensor_recipe")
    if field is None:
        raise SystemExit("%s: no sllm.tensor_recipe field" % path)
    text = bytes(field.parts[field.data[0]]).decode("utf-8")
    if not text.strip().startswith("{") or not text.strip().endswith("}"):
        raise SystemExit("%s: tensor recipe is not a complete JSON object" % path)
    recipe = json.loads(text)
    tensor_types = {tensor.name: int(tensor.tensor_type) for tensor in reader.tensors}
    return recipe, recipe["bindings"], tensor_types, len(reader.tensors)


def gguf_audit(path, recipe, bindings, tensor_types):
    """Compare the recipe's quantized set with what the GGUF really stores."""
    quantized = {}
    encoding_counts = {}
    for binding in bindings:
        name = binding["logical_tensor"]
        quantized[name] = tuple(int(d) for d in binding["logical_shape"])
        encoding = binding.get("encoding")
        encoding_counts[encoding] = encoding_counts.get(encoding, 0) + 1
    type_counts = {}
    for value in tensor_types.values():
        key = GGUF_TYPE_NAMES.get(value, str(value))
        type_counts[key] = type_counts.get(key, 0) + 1
    scale_suffix = ".sllm.scale.block32_e8m0"
    i8_value = sorted(n for n, t in tensor_types.items() if t == 24 and not n.endswith(scale_suffix))
    i8_scale = sorted(n for n, t in tensor_types.items() if t == 24 and n.endswith(scale_suffix))
    bf16_names = sorted(n for n, t in tensor_types.items() if t == 30)
    return quantized, {
        "gguf_path": path,
        "gguf_tensor_count": len(tensor_types),
        "gguf_type_counts": type_counts,
        "unexpected_type_codes": sorted({t for t in tensor_types.values() if t not in (24, 30)}),
        "recipe_binding_count": len(bindings),
        "recipe_encoding_counts": encoding_counts,
        "quantized_value_tensors_typed_i8": len(quantized) - len(
            sorted(n for n in quantized if tensor_types.get(n) != 24)
        ),
        "quantized_value_tensors_missing_or_mistyped": sorted(
            n for n in quantized if tensor_types.get(n) != 24
        ),
        "scale_tensors_declared": sum(len(b.get("scales", [])) for b in bindings),
        "scale_tensors_missing": sorted(
            s["tensor"]
            for binding in bindings
            for s in binding.get("scales", [])
            if s["tensor"] not in tensor_types
        ),
        "i8_value_tensor_count": len(i8_value),
        "i8_scale_tensor_count": len(i8_scale),
        "i8_value_tensors_not_in_recipe": sorted(set(i8_value) - set(quantized)),
        "recipe_value_tensors_not_i8": sorted(set(quantized) - set(i8_value)),
        "sets_match": sorted(i8_value) == sorted(quantized),
        "bf16_retained_count": len(bf16_names),
        "bf16_retained_tensors": bf16_names,
        "known_unconsumed_tensor_count": len(recipe.get("known_unconsumed_tensors", [])),
        "known_unconsumed_tensors": sorted(recipe.get("known_unconsumed_tensors", [])),
    }


# --------------------------------------------------------------------------
# Per-rule accumulation helpers
# --------------------------------------------------------------------------


def new_accumulator(rules):
    return {rule: {"sse": 0.0, "sum_sq": 0.0, "sat": 0, "elements": 0} for rule in rules}


def accumulate(acc, qn, qmag, scale, w, max_mag):
    """Accumulate SSE / saturation for one rule over a chunk (real units)."""
    delta = qn * scale
    delta -= w
    flat = delta.reshape(-1)
    acc["sse"] += float(np.einsum("i,i->", flat, flat, dtype=np.float64))
    acc["sat"] += int(np.count_nonzero(qmag >= np.float32(max_mag)))
    acc["elements"] += int(qmag.size)


def accumulate_sum_sq(acc, w):
    flat = np.ascontiguousarray(w).reshape(-1)
    acc["sum_sq"] += float(np.einsum("i,i->", flat, flat, dtype=np.float64))


def select_rule(acc, cand_a, cand_b, mask, w, max_mag):
    """Accumulate the per-element choice between two quantized candidates."""
    qn = np.where(mask, cand_a[0], cand_b[0])
    qmag = np.where(mask, cand_a[1], cand_b[1])
    scale = np.where(mask, cand_a[2], cand_b[2])
    accumulate(acc, qn, qmag, scale, w, max_mag)


# --------------------------------------------------------------------------
# Streaming helpers
# --------------------------------------------------------------------------


def row_chunks(rows, k, chunk_elements, row_stride):
    """Yield whole-row selections; whole quantization blocks stay intact."""
    per_chunk = max(1, int(chunk_elements) // k)
    for start in range(0, rows, per_chunk):
        stop = min(start + per_chunk, rows)
        if row_stride > 1:
            yield np.arange(start, stop, row_stride)
        else:
            yield slice(start, stop)


def prepare_chunk(w):
    """Sanitize non-finite values and report how many were seen."""
    bits = np.bitwise_and(w.view(U32), MAG_MASK)
    nonfinite = int(np.count_nonzero((bits >> U32(23)) >= U32(255)))
    if nonfinite:
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return w, nonfinite


# --------------------------------------------------------------------------
# MX family measurement
# --------------------------------------------------------------------------


def measure_mx_tensor(raw_bf16, shape, want, args):
    """Measure the requested MX rule sets for one BF16 tensor."""
    block = 32
    k = int(shape[-1])
    rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    if k % block:
        raise ValueError("K=%d is not a multiple of the %d-element MX block" % (k, block))
    rules = []
    for fmt, enabled in want.items():
        if enabled:
            rules.extend("%s_%s" % (fmt, rule) for rule in MX_RULES[fmt])
    acc = new_accumulator(rules)
    clamped = {fmt: 0 for fmt in want}
    nonfinite = 0
    for selection in row_chunks(rows, k, args.chunk_elements, args.sample_row_stride):
        chunk = np.ascontiguousarray(raw_bf16[selection])
        w2, count = prepare_chunk(bf16_to_f32(chunk))
        nonfinite += count
        n, width = w2.shape
        blocks = width // block
        w3 = w2.reshape(n, blocks, block)
        accumulate_sum_sq(acc[rules[0]], w2)
        for rule in rules[1:]:
            acc[rule]["sum_sq"] = acc[rules[0]]["sum_sq"]
        magnitude = np.bitwise_and(w3.view(U32), MAG_MASK)
        amax = magnitude.max(axis=2).view(np.float32)
        flat_exponent = floor_log2(amax)
        if want.get("mxfp8"):
            spec = MX_FORMATS["mxfp8"]
            _mantissa, _emin, max_mag = elem_parts(spec["elem"])
            candidates = (
                ("old", flat_exponent - spec["emax"]),
                ("new", no_clip_exponents(amax, max_mag)),
            )
            for rule, exponents in candidates:
                scale, inverse, count = e8m0_pairs(exponents)
                clamped["mxfp8"] += count
                qn, qmag = quantize_rne(w3 * inverse[..., None], spec["elem"])
                accumulate(acc["mxfp8_%s" % rule], qn, qmag, scale[..., None], w3, max_mag)
        if want.get("mxfp6"):
            spec = MX_FORMATS["mxfp6"]
            _mantissa, _emin, max_mag = elem_parts(spec["elem"])
            candidates = (
                ("old", flat_exponent - spec["emax"]),
                ("new", no_clip_exponents(amax, max_mag)),
            )
            for rule, exponents in candidates:
                scale, inverse, count = e8m0_pairs(exponents)
                clamped["mxfp6"] += count
                qn, qmag = quantize_rne(w3 * inverse[..., None], spec["elem"])
                accumulate(acc["mxfp6_%s" % rule], qn, qmag, scale[..., None], w3, max_mag)
        if want.get("mxfp4"):
            spec = MX_FORMATS["mxfp4"]
            _mantissa, _emin, max_mag = elem_parts(spec["elem"])
            floor_exponents = flat_exponent - spec["emax"]
            quantized = {}
            for label, exponents in (
                ("floor", floor_exponents),
                ("bumped", floor_exponents + 1),
                ("no_clip", no_clip_exponents(amax, max_mag)),
            ):
                scale, inverse, count = e8m0_pairs(exponents)
                clamped["mxfp4"] += count
                qn, qmag = quantize_rne(w3 * inverse[..., None], spec["elem"])
                quantized[label] = (qn, qmag, scale[..., None])
            floor_cand = quantized["floor"]
            bumped_cand = quantized["bumped"]
            accumulate(acc["mxfp4_floor"], floor_cand[0], floor_cand[1], floor_cand[2], w3, max_mag)
            no_clip_cand = quantized["no_clip"]
            accumulate(
                acc["mxfp4_no_clip"], no_clip_cand[0], no_clip_cand[1], no_clip_cand[2], w3, max_mag
            )
            exact_power_of_two = np.bitwise_and(amax.view(U32), U32(0x7FFFFF)) == 0
            select_rule(acc["mxfp4_ceil"], floor_cand, bumped_cand, exact_power_of_two[..., None], w3, max_mag)
            even_threshold = np.ldexp(np.float32(1.75), flat_exponent.astype(np.int32))
            select_rule(acc["mxfp4_even"], bumped_cand, floor_cand, (amax >= even_threshold)[..., None], w3, max_mag)
            delta_floor = floor_cand[0] * floor_cand[2] - w3
            delta_bumped = bumped_cand[0] * bumped_cand[2] - w3
            sse_floor = np.einsum("ijk,ijk->ij", delta_floor, delta_floor, dtype=np.float64)
            sse_bumped = np.einsum("ijk,ijk->ij", delta_bumped, delta_bumped, dtype=np.float64)
            select_rule(acc["mxfp4_best"], floor_cand, bumped_cand, (sse_floor <= sse_bumped)[..., None], w3, max_mag)
    return acc, {"e8m0_clamped_scales": clamped, "nonfinite_elements": nonfinite}


# --------------------------------------------------------------------------
# NVFP4 measurement
# --------------------------------------------------------------------------


NVFP4_SCALE_VALUES_F32 = element_codes("e4m3fn")[1].astype(np.float32)
NVFP4_SCALE_TABLE = element_code_value_table("e4m3fn")


def nvfp4_tensor_amax(raw_bf16, args):
    """Largest finite BF16 magnitude of the tensor."""
    rows, k = int(raw_bf16.shape[0]), int(raw_bf16.shape[1])
    best = 0.0
    for selection in row_chunks(rows, k, args.chunk_elements, args.sample_row_stride):
        chunk = np.ascontiguousarray(raw_bf16[selection])
        w2 = bf16_to_f32(chunk)
        bits = np.bitwise_and(w2.view(U32), MAG_MASK)
        finite = (bits >> U32(23)) < U32(255)
        if not finite.all():
            w2 = np.where(finite, w2, np.float32(0.0))
        best = max(best, float(np.abs(w2).max()))
    return np.float32(best)


def nvfp4_global_scale(amax_tensor):
    """Spec global scale for weights: g = amax_tensor / 6 / 448."""
    return np.float32(np.float32(amax_tensor) / np.float32(6.0) / np.float32(448.0))


def measure_nvfp4_tensor(raw_bf16, shape, stored_scale_codes, artifact_wgs, args):
    """Measure NVFP4 old / new / intermediate rules for one BF16 tensor."""
    block = NVFP4_BLOCK
    rows, k = int(shape[0]), int(shape[1])
    if k % block:
        raise ValueError("K=%d is not a multiple of the %d-element NVFP4 block" % (k, block))
    if tuple(stored_scale_codes.shape) != (rows, k // block):
        raise ValueError(
            "stored weight_scale shape %s does not match (%d, %d)"
            % (tuple(stored_scale_codes.shape), rows, k // block)
        )
    _mantissa, _emin, max_mag = elem_parts("e2m1")
    amax_tensor = nvfp4_tensor_amax(raw_bf16, args)
    g = nvfp4_global_scale(amax_tensor)
    inverse_g = (np.float32(1.0) / g) if g > 0 else np.float32(0.0)
    acc = new_accumulator(NVFP4_RULES)
    diag = {
        "tensor_amax": float(amax_tensor),
        "spec_global_scale_g": float(g),
        "spec_inverse_g": float(inverse_g),
        "artifact_weight_global_scale": None if artifact_wgs is None else float(artifact_wgs),
        "blocks": 0,
        "stored_codes": 0,
        "stored_codes_equal_spec_rne": 0,
        "stored_codes_in_adjacent_pair": 0,
        "old_code_outside_adjacent_pair": 0,
        "invalid_stored_codes": 0,
        "negative_stored_sign_bits": 0,
        "stored_scale_zero_blocks": 0,
        "nonfinite_elements": 0,
    }
    rows_per_chunk = max(1, int(args.chunk_elements) // k)
    for start in range(0, rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, rows)
        if args.sample_row_stride > 1:
            selection = np.arange(start, stop, args.sample_row_stride)
        else:
            selection = slice(start, stop)
        chunk = np.ascontiguousarray(raw_bf16[selection])
        w2, count = prepare_chunk(bf16_to_f32(chunk))
        diag["nonfinite_elements"] += count
        stored = np.ascontiguousarray(stored_scale_codes[selection]).astype(np.uint8)
        n = w2.shape[0]
        blocks = k // block
        w3 = w2.reshape(n, blocks, block)
        accumulate_sum_sq(acc[NVFP4_RULES[0]], w2)
        for rule in NVFP4_RULES[1:]:
            acc[rule]["sum_sq"] = acc[NVFP4_RULES[0]]["sum_sq"]
        magnitude = np.bitwise_and(w3.view(U32), MAG_MASK)
        amax_block = magnitude.max(axis=2).view(np.float32)
        if g > 0:
            raw = (amax_block / np.float32(6.0)) / g
        else:
            raw = np.zeros_like(amax_block)
        flat_raw = raw.reshape(-1)
        # The two E4M3 codes adjacent to the raw block scale: lo is the largest
        # code whose value is <= raw, hi the smallest whose value is >= raw.
        # When raw lands exactly on a code the two coincide, so "best of the
        # adjacent pair" degrades to the nearest code for that block.
        last = NVFP4_SCALE_VALUES_F32.size - 1
        hi_idx = np.clip(np.searchsorted(NVFP4_SCALE_VALUES_F32, flat_raw, side="left"), 0, last)
        lo_idx = np.clip(np.searchsorted(NVFP4_SCALE_VALUES_F32, flat_raw, side="right") - 1, 0, last)
        _q, rne_magnitude = quantize_rne(flat_raw, "e4m3fn")
        old_idx = np.clip(
            np.searchsorted(NVFP4_SCALE_VALUES_F32, rne_magnitude, side="left"),
            0,
            NVFP4_SCALE_VALUES_F32.size - 1,
        )
        stored_flat = stored.reshape(-1).astype(np.int64)
        sign_bits = stored_flat >= 128
        diag["negative_stored_sign_bits"] += int(np.count_nonzero(sign_bits))
        diag["invalid_stored_codes"] += int(np.count_nonzero(stored_flat == 0x7F))
        stored_idx = np.where(sign_bits, stored_flat - 128, stored_flat)
        stored_values = NVFP4_SCALE_TABLE[np.clip(stored_idx, 0, NVFP4_SCALE_TABLE.size - 1)]
        stored_values = np.where(np.isfinite(stored_values), stored_values, 0.0)
        diag["blocks"] += int(old_idx.size)
        diag["stored_codes"] += int(stored_idx.size)
        diag["stored_codes_equal_spec_rne"] += int(np.count_nonzero(stored_idx == old_idx))
        diag["stored_codes_in_adjacent_pair"] += int(
            np.count_nonzero((stored_idx == lo_idx) | (stored_idx == hi_idx))
        )
        diag["old_code_outside_adjacent_pair"] += int(
            np.count_nonzero((old_idx != lo_idx) & (old_idx != hi_idx))
        )
        diag["stored_scale_zero_blocks"] += int(np.count_nonzero(stored_values == 0))
        lo_value = NVFP4_SCALE_VALUES_F32[lo_idx].reshape(n, blocks)
        hi_value = NVFP4_SCALE_VALUES_F32[hi_idx].reshape(n, blocks)
        old_value = NVFP4_SCALE_VALUES_F32[old_idx].reshape(n, blocks)
        artifact_value = stored_values.astype(np.float32).reshape(n, blocks)
        if artifact_wgs:
            artifact_real = artifact_value / np.float32(artifact_wgs)
        else:
            artifact_real = artifact_value * g

        def quantize_with(code_values, real_scale, norm_scale=None):
            safe = np.where(code_values > 0, code_values, np.float32(1.0))
            inverse = np.where(code_values > 0, np.float32(1.0) / safe, np.float32(0.0))
            normalized = w3 * (inverse_g if norm_scale is None else norm_scale)
            qn, qmag = quantize_rne(normalized * inverse[..., None], "e2m1")
            return qn, qmag, real_scale[..., None]

        cand_lo = quantize_with(lo_value, lo_value * g)
        cand_hi = quantize_with(hi_value, hi_value * g)
        accumulate(acc["intermediate_ceil_adjacent"], cand_hi[0], cand_hi[1], cand_hi[2], w3, max_mag)
        old_is_lo = (old_idx == lo_idx).reshape(n, blocks)[..., None]
        select_rule(acc["old_nearest"], cand_lo, cand_hi, old_is_lo, w3, max_mag)
        delta_lo = cand_lo[0] * cand_lo[2] - w3
        delta_hi = cand_hi[0] * cand_hi[2] - w3
        sse_lo = np.einsum("ijk,ijk->ij", delta_lo, delta_lo, dtype=np.float64)
        sse_hi = np.einsum("ijk,ijk->ij", delta_hi, delta_hi, dtype=np.float64)
        select_rule(acc["new_adjacent_best"], cand_lo, cand_hi, (sse_lo <= sse_hi)[..., None], w3, max_mag)
        # The artifact's own dequantization divides by its stored global scale,
        # so that rule normalizes with the stored scale, not the spec's g.
        artifact_norm = np.float32(artifact_wgs) if artifact_wgs else inverse_g
        cand_artifact = quantize_with(artifact_value, artifact_real, artifact_norm)
        accumulate(
            acc["intermediate_artifact_stored_scale"],
            cand_artifact[0],
            cand_artifact[1],
            cand_artifact[2],
            w3,
            max_mag,
        )
    return acc, diag


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------


def finalize_rule(stats):
    elements = stats["elements"]
    sum_sq = stats["sum_sq"]
    return {
        "rel_rms": math.sqrt(stats["sse"] / sum_sq) if sum_sq > 0 else 0.0,
        "saturation_fraction": (stats["sat"] / elements) if elements else 0.0,
        "saturation_elements": stats["sat"],
        "sse": stats["sse"],
        "sum_sq": sum_sq,
    }


def aggregate(rule_stats_list):
    totals = {}
    for stats in rule_stats_list:
        for rule, values in stats.items():
            entry = totals.setdefault(rule, {"sse": 0.0, "sum_sq": 0.0, "sat": 0, "elements": 0})
            entry["sse"] += values["sse"]
            entry["sum_sq"] += values["sum_sq"]
            entry["sat"] += values["sat"]
            entry["elements"] += values["elements"]
    out = {}
    for rule, entry in totals.items():
        out[rule] = {
            "rel_rms": math.sqrt(entry["sse"] / entry["sum_sq"]) if entry["sum_sq"] > 0 else 0.0,
            "saturation_fraction": (entry["sat"] / entry["elements"]) if entry["elements"] else 0.0,
            "saturation_elements": entry["sat"],
            "elements": entry["elements"],
            "sse": entry["sse"],
            "sum_sq": entry["sum_sq"],
        }
    return out


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------


def reference_rne_magnitude(magnitudes, codes, values):
    """Table reference: nearest representable magnitude, ties to even code."""
    index = np.searchsorted(values, magnitudes, side="left")
    hi = np.clip(index, 0, values.size - 1)
    lo = np.clip(index - 1, 0, values.size - 1)
    delta_lo = magnitudes - values[lo]
    delta_hi = values[hi] - magnitudes
    take_lo = (delta_lo < delta_hi) | ((delta_lo == delta_hi) & ((codes[lo] % 2) == 0))
    return np.where(take_lo, values[lo], values[hi])


def self_test(seed=20260919):
    """Validate the bit-trick quantizer and the scale rules."""
    result = {"status": "pass", "checks": [], "mismatches": []}
    rng = np.random.default_rng(seed)
    for key in ("e4m3fn", "e3m2", "e2m1"):
        codes, values = element_codes(key)
        max_mag = elem_parts(key)[2]
        samples = [
            rng.uniform(0.0, max_mag * 2.0, size=200000),
            rng.uniform(0.0, 0.02, size=100000),
            rng.uniform(0.0, 0.5, size=100000),
            np.linspace(0.0, max_mag * 1.5, 20000),
            values * 0.999,
            values,
            values * 1.001,
            (values[:-1] + values[1:]) / 2.0,
            np.array([0.0, 1e-30, 2.0 ** -130, float(np.finfo(np.float32).tiny)]),
        ]
        magnitudes = np.concatenate([np.asarray(s, dtype=np.float64) for s in samples])
        signed = np.concatenate([magnitudes, -magnitudes]).astype(np.float32)
        _q, qmag = quantize_rne(signed, key)
        reference = reference_rne_magnitude(np.abs(signed).astype(np.float64), codes, values)
        mismatch = int(np.count_nonzero(qmag.astype(np.float64) != reference))
        result["checks"].append({"check": "rne_vs_table:%s" % key, "samples": int(signed.size), "mismatches": mismatch})
        if mismatch:
            result["status"] = "fail"
            bad = np.nonzero(qmag.astype(np.float64) != reference)[0][:5]
            result["mismatches"].append(
                {"check": "rne_vs_table:%s" % key, "examples": [float(signed[i]) for i in bad]}
            )
    boundary = np.array(
        [0.0, 2.0 ** -20, 1.0, 1.5, 1.75, 2.0, 3.5, 447.0, 448.0, 449.0, 512.0, 1024.0],
        dtype=np.float32,
    )
    expectations = {
        "floor_log2": (floor_log2(boundary), [-127, -20, 0, 0, 0, 1, 1, 8, 8, 8, 9, 10]),
        "ceil_log2": (ceil_log2(boundary), [-127, -20, 0, 1, 1, 1, 2, 9, 9, 9, 9, 10]),
        "no_clip_448": (
            no_clip_exponents(boundary, 448.0),
            [-127, -28, -8, -8, -8, -7, -7, 0, 0, 1, 1, 2],
        ),
        "no_clip_28": (
            no_clip_exponents(boundary, 28.0),
            [-127, -24, -4, -4, -4, -3, -3, 4, 4, 5, 5, 6],
        ),
        "no_clip_6": (
            no_clip_exponents(boundary, 6.0),
            [-127, -22, -2, -2, -1, -1, 0, 7, 7, 7, 7, 8],
        ),
    }
    for name, (got, want) in expectations.items():
        want_array = np.array(want, dtype=np.int64)
        mismatch = int(np.count_nonzero(got != want_array))
        result["checks"].append({"check": name, "samples": int(boundary.size), "mismatches": mismatch})
        if mismatch:
            result["status"] = "fail"
            result["mismatches"].append({"check": name, "got": got.tolist(), "want": want})
    return result


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------


def render_markdown(result, print_tensor_table=False):
    lines = []
    lines.append(
        "| format | tensors | elements | old rel-RMS % | new rel-RMS % | saturation old % | saturation new % |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name in ("mxfp8", "mxfp6", "mxfp4", "nvfp4"):
        entry = result["formats"].get(name)
        if not entry or "aggregate" not in entry:
            continue
        agg = entry["aggregate"]
        old_rule, new_rule = REPORT_RULES[name]
        lines.append(
            "| %s | %d | %d | %s %.4f | %s %.4f | %.4f | %.4f |"
            % (
                name,
                entry["tensor_count"],
                entry["element_count"],
                old_rule + "=",
                100.0 * agg[old_rule]["rel_rms"],
                new_rule + "=",
                100.0 * agg[new_rule]["rel_rms"],
                100.0 * agg[old_rule]["saturation_fraction"],
                100.0 * agg[new_rule]["saturation_fraction"],
            )
        )
    lines.append("")
    lines.append("| format | intermediate / alternative rules (rel-RMS %, saturation %) |")
    lines.append("| --- | --- |")
    for name in ("mxfp8", "mxfp6", "mxfp4", "nvfp4"):
        entry = result["formats"].get(name)
        if not entry or "aggregate" not in entry:
            continue
        parts = []
        for rule, stats in entry["aggregate"].items():
            if rule in REPORT_RULES[name]:
                continue
            parts.append(
                "%s=%.4f,\\ sat %.4f" % (rule, 100.0 * stats["rel_rms"], 100.0 * stats["saturation_fraction"])
            )
        lines.append("| %s | %s |" % (name, "; ".join(parts) if parts else "-"))
    if print_tensor_table:
        lines.append("")
        lines.append("| format | tensor | shape | elements | old % | new % |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for name in ("mxfp8", "mxfp6", "mxfp4", "nvfp4"):
            entry = result["formats"].get(name)
            if not entry or "aggregate" not in entry:
                continue
            old_rule, new_rule = REPORT_RULES[name]
            for tensor in entry["tensors"]:
                lines.append(
                    "| %s | %s | %s | %d | %.4f | %.4f |"
                    % (
                        name,
                        tensor["name"],
                        "x".join(str(d) for d in tensor["shape"]),
                        tensor["elements"],
                        100.0 * tensor["rules"][old_rule]["rel_rms"],
                        100.0 * tensor["rules"][new_rule]["rel_rms"],
                    )
                )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Independent low-precision weight error measurement.")
    parser.add_argument(
        "--bf16-root",
        default="/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-BF16",
        help="BF16 safetensors source directory (with model.safetensors.index.json)",
    )
    parser.add_argument(
        "--mxfp8-gguf",
        default="/home/homelab1/datapool/qwen38-kld-20260918/sllm-mxfp8/Qwen3.8-27B-MXFP8.gguf",
    )
    parser.add_argument(
        "--mxfp6-gguf",
        default="/home/homelab1/datapool/qwen38-kld-20260918/sllm-mxfp6/Qwen3.8-27B-MXFP6.gguf",
    )
    parser.add_argument(
        "--nvfp4-root",
        default="/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-NVFP4",
    )
    parser.add_argument(
        "--nvfp4-identity",
        default="/home/homelab1/datapool/qwen38-kld-20260918/nvfp4-source-identity.json",
    )
    parser.add_argument("--model-lock", default="docs/models/locks/qwen3.8-27b-bf16.json")
    parser.add_argument(
        "--output",
        default="/home/homelab1/datapool/qwen38-kld-20260918/results/lowp-weight-error-independent.json",
    )
    parser.add_argument("--formats", default="mxfp8,mxfp6,mxfp4,nvfp4", help="comma-separated subset")
    parser.add_argument("--chunk-elements", type=int, default=8_000_000, help="elements per streamed chunk")
    parser.add_argument(
        "--sample-row-stride",
        type=int,
        default=1,
        help="use every Nth row (whole blocks only); 1 = all elements",
    )
    parser.add_argument("--max-tensors", type=int, default=0, help="limit tensors per format (0 = all)")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="worker processes for the per-tensor scan (1 = in-process); results are independent of this",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--print-tensor-table", action="store_true", help="print the per-tensor table too")
    return parser.parse_args(argv)


def load_bf16_index(root):
    with open(os.path.join(root, "model.safetensors.index.json")) as handle:
        index = json.load(handle)
    return index["weight_map"]


# --------------------------------------------------------------------------
# Optional multi-process fan-out
#
# Each worker owns whole tensors and returns only per-rule SSE / sum-of-squares
# / saturation counters, so the parent-side aggregation order (sorted tensor
# names) is identical to the single-process path and the result is
# deterministic regardless of how many workers run.
# --------------------------------------------------------------------------

_WORKER = {}


def _worker_init(bf16_root, bf16_map, args, nvfp4_file=None):
    _WORKER.clear()
    _WORKER.update(
        {
            "bf16_root": bf16_root,
            "bf16_map": bf16_map,
            "args": args,
            "nvfp4_file": nvfp4_file,
            "bf16_handles": {},
            "store": None,
        }
    )


def _worker_bf16(name):
    handles = _WORKER["bf16_handles"]
    shard = _WORKER["bf16_map"][name]
    if shard not in handles:
        handles[shard] = SafeTensors(os.path.join(_WORKER["bf16_root"], shard))
    return handles[shard].raw(name)


def _worker_mx_tensor(job):
    name, shape, want = job
    try:
        acc, diag = measure_mx_tensor(_worker_bf16(name), shape, want, _WORKER["args"])
    except ValueError as exc:
        return name, None, None, str(exc)
    return name, acc, diag, None


def _worker_nvfp4_tensor(job):
    base, source_name, shape = job
    if _WORKER["store"] is None:
        _WORKER["store"] = SafeTensors(_WORKER["nvfp4_file"])
    store = _WORKER["store"]
    scale_codes = store.raw(base + "_scale")
    wgs = store.raw(base + "_global_scale").astype(np.float32).reshape(-1)
    try:
        acc, diag = measure_nvfp4_tensor(
            _worker_bf16(source_name), shape, scale_codes, float(wgs[0]), _WORKER["args"]
        )
    except ValueError as exc:
        return base, None, None, str(exc)
    return base, acc, diag, None


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    started = time.time()
    wanted = [name.strip() for name in args.formats.split(",") if name.strip()]
    result = {
        "schema": "sllm-lowp-weight-error-independent-v1",
        "tool": os.path.relpath(os.path.abspath(__file__)),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "provenance": (
            "Clean-room NumPy implementation written from the format spec in "
            "docs/plans/active/2026/09/11-20/low-precision-scale-selection.md; "
            "ci/tools/qwen38_weight_error.py was not read or imported."
        ),
        "inputs": {
            "bf16_root": args.bf16_root,
            "mxfp8_gguf": args.mxfp8_gguf,
            "mxfp6_gguf": args.mxfp6_gguf,
            "nvfp4_root": args.nvfp4_root,
            "nvfp4_identity": args.nvfp4_identity,
            "model_lock": args.model_lock,
        },
        "sampling": {
            "row_stride": args.sample_row_stride,
            "chunk_elements": args.chunk_elements,
            "jobs": max(1, int(args.jobs)),
            "note": (
                "all elements of every quantized matrix are processed in whole 32/16-element "
                "blocks unless row_stride > 1, which keeps whole rows and therefore whole blocks"
            ),
        },
        "self_test": self_test(),
        "formats": {},
        "audit": {},
        "unclassified": [],
    }
    print("[self-test] %s" % result["self_test"]["status"], flush=True)
    if result["self_test"]["status"] != "pass":
        print(json.dumps(result["self_test"], indent=2), flush=True)
    if args.max_tensors:
        result["sampling"]["max_tensors_per_format"] = args.max_tensors

    bf16_map = load_bf16_index(args.bf16_root)
    handles = {}

    def bf16_tensor(name):
        shard = bf16_map[name]
        if shard not in handles:
            handles[shard] = SafeTensors(os.path.join(args.bf16_root, shard))
        return handles[shard].raw(name)

    mx_sets = {}
    for fmt in ("mxfp8", "mxfp6"):
        if fmt not in wanted:
            continue
        path = args.mxfp8_gguf if fmt == "mxfp8" else args.mxfp6_gguf
        _recipe, bindings, types, tensor_count = read_gguf_recipe(path)
        quantized, audit = gguf_audit(path, _recipe, bindings, types)
        audit["gguf_tensor_count"] = tensor_count
        result["audit"]["%s_gguf" % fmt] = audit
        mx_sets[fmt] = quantized
        print(
            "[audit] %s: %d bindings, %d I8 value tensors, %d BF16 retained, sets_match=%s"
            % (
                fmt,
                len(bindings),
                audit["i8_value_tensor_count"],
                audit["bf16_retained_count"],
                audit["sets_match"],
            ),
            flush=True,
        )
    if "mxfp4" in wanted:
        if "mxfp8" in mx_sets:
            mx_sets["mxfp4"] = dict(mx_sets["mxfp8"])
        else:
            result["formats"]["mxfp4"] = {
                "status": "not measured: no MXFP4 artifact in this run and --formats excluded mxfp8"
            }

    mx_measured = [fmt for fmt in ("mxfp8", "mxfp6", "mxfp4") if fmt in mx_sets]
    if mx_measured:
        reference_set = mx_sets[mx_measured[0]]
        if not all(sorted(mx_sets[f]) == sorted(reference_set) for f in mx_measured):
            result["unclassified"].append(
                {
                    "issue": "MX GGUF quantized sets differ between formats",
                    "detail": {f: len(mx_sets[f]) for f in mx_measured},
                }
            )
        names = sorted(reference_set)
        if args.max_tensors:
            names = names[: args.max_tensors]
        want = {fmt: True for fmt in mx_measured}
        per_format = {
            fmt: {
                "encoding": MX_FORMATS[fmt]["gguf_encoding"],
                "quantized_set_source": (
                    args.mxfp8_gguf if fmt in ("mxfp8", "mxfp4") else args.mxfp6_gguf
                )
                + (" (MX quantized tensor set; no MXFP4 artifact in this run)" if fmt == "mxfp4" else ""),
                "block_elements": MX_FORMATS[fmt]["block"],
                "tensors": [],
                "stats": [],
            }
            for fmt in mx_measured
        }
        print("[mx] measuring %d tensors for %s" % (len(names), ",".join(mx_measured)), flush=True)
        jobs = max(1, int(args.jobs))
        measured = {}
        errors = {}
        if jobs > 1 and len(names) > 1:
            from concurrent.futures import ProcessPoolExecutor

            queued = [name for name in names if name in bf16_map]
            for name in names:
                if name not in bf16_map:
                    errors[name] = ("tensor missing from BF16 index", None)
            work = [(name, tuple(int(d) for d in reference_set[name]), want) for name in queued]
            work.sort(key=lambda job: -int(np.prod(job[1])))
            print("[mx] fanning out over %d workers" % jobs, flush=True)
            with ProcessPoolExecutor(
                max_workers=jobs,
                initializer=_worker_init,
                initargs=(args.bf16_root, dict(bf16_map), args),
            ) as pool:
                for count, (name, acc, diag, error) in enumerate(
                    pool.map(_worker_mx_tensor, work, chunksize=1), 1
                ):
                    if error is not None:
                        errors[name] = (error, list(reference_set[name]))
                    else:
                        measured[name] = (acc, diag)
                    if args.progress_every and (
                        count % args.progress_every == 0 or count == len(work)
                    ):
                        elapsed = time.time() - started
                        print(
                            "[mx] %d/%d tensors (%.2f tensors/s, %.1f min elapsed)"
                            % (count, len(work), count / elapsed if elapsed else 0.0, elapsed / 60.0),
                            flush=True,
                        )
        else:
            for count, name in enumerate(names, 1):
                if name not in bf16_map:
                    errors[name] = ("tensor missing from BF16 index", None)
                    continue
                shape = reference_set[name]
                try:
                    acc, diag = measure_mx_tensor(bf16_tensor(name), shape, want, args)
                except ValueError as exc:
                    errors[name] = (str(exc), list(shape))
                    continue
                measured[name] = (acc, diag)
                if args.progress_every and (count % args.progress_every == 0 or count == len(names)):
                    elapsed = time.time() - started
                    print(
                        "[mx] %d/%d tensors (%.2f tensors/s, %.1f min elapsed)"
                        % (count, len(names), count / elapsed if elapsed else 0.0, elapsed / 60.0),
                        flush=True,
                    )
        for name in names:
            if name in errors:
                message, shape = errors[name]
                entry = {"issue": message, "tensor": name}
                if shape:
                    entry["shape"] = shape
                result["unclassified"].append(entry)
                continue
            if name not in measured:
                continue
            shape = reference_set[name]
            acc, diag = measured[name]
            elements = int(np.prod(shape))
            for fmt in mx_measured:
                rules = {
                    rule: acc["%s_%s" % (fmt, rule)]
                    for rule in MX_RULES[fmt]
                    if "%s_%s" % (fmt, rule) in acc
                }
                if not rules:
                    continue
                per_format[fmt]["tensors"].append(
                    {
                        "name": name,
                        "shape": list(shape),
                        "elements": elements,
                        "rules": {rule: finalize_rule(values) for rule, values in rules.items()},
                    }
                )
                per_format[fmt]["stats"].append(rules)
        for fmt in mx_measured:
            per_format[fmt]["aggregate"] = aggregate(per_format[fmt]["stats"])
            per_format[fmt]["tensor_count"] = len(per_format[fmt]["tensors"])
            per_format[fmt]["element_count"] = sum(t["elements"] for t in per_format[fmt]["tensors"])
            per_format[fmt]["rules"] = MX_RULES[fmt]
            del per_format[fmt]["stats"]
            result["formats"][fmt] = per_format[fmt]

    if "nvfp4" in wanted:
        result["formats"]["nvfp4"] = measure_nvfp4(args, bf16_map, bf16_tensor, result)

    measured = {name: entry for name, entry in result["formats"].items() if "aggregate" in entry}
    result["totals"] = {
        "formats_measured": list(measured),
        "elements_scanned_per_format": {name: entry["element_count"] for name, entry in measured.items()},
        "elements_scanned_total": sum(entry["element_count"] for entry in measured.values()),
        "unique_weight_elements_scanned": sum(
            entry["element_count"] for name, entry in measured.items() if name in ("mxfp8", "nvfp4")
        ),
        "note": "the MXFP8 / MXFP6 / MXFP4 rows share one tensor set and one scan",
        "elapsed_seconds": time.time() - started,
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=1, sort_keys=False)
        handle.write("\n")
    print("[json] wrote %s" % args.output, flush=True)
    print()
    print(render_markdown(result, print_tensor_table=args.print_tensor_table))
    return 0


def measure_nvfp4(args, bf16_map, bf16_tensor, result):
    root = args.nvfp4_root
    index_path = os.path.join(root, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return {"status": "artifact not located", "detail": "%s not found" % index_path}
    store = SafeTensors(os.path.join(root, "model.safetensors"))
    names = store.names()
    packed = sorted(name for name in names if name.endswith(".weight_packed"))
    if args.max_tensors:
        packed = packed[: args.max_tensors]
    # The artifact is a hybrid: only ``.weight_packed`` tensors are NVFP4.  The
    # remaining linears ship as FP8 E4M3 weights with BF16 per-row scales, and a
    # handful of tensors stay BF16.  Those are reported, not scanned as NVFP4.
    fp8_e4m3_weights = sorted(
        name for name in names if name.endswith(".weight") and store.dtype(name) == "F8_E4M3"
    )
    bf16_per_row_scales = sorted(
        name for name in names if name.endswith(".weight_scale") and store.dtype(name) == "BF16"
    )
    module_kinds = {}
    for name in packed:
        parts = name.split(".")
        kind = parts[-2] if len(parts) >= 2 else name
        module_kinds[kind] = module_kinds.get(kind, 0) + 1
    audit = {
        "artifact_root": root,
        "safetensors_file": os.path.join(root, "model.safetensors"),
        "tensor_count": len(names),
        "weight_packed_count": len(packed),
        "weight_packed_module_kinds": module_kinds,
        "weight_scale_count": sum(1 for n in names if n.endswith(".weight_scale")),
        "weight_global_scale_count": sum(1 for n in names if n.endswith(".weight_global_scale")),
        "input_global_scale_count": sum(1 for n in names if n.endswith(".input_global_scale")),
        "fp8_e4m3_weight_count": len(fp8_e4m3_weights),
        "fp8_e4m3_bf16_per_row_scale_count": len(bf16_per_row_scales),
        "fp8_e4m3_lm_head": [n for n in fp8_e4m3_weights if n.startswith("lm_head.")],
        "fp8_e4m3_module_kinds": sorted({n.split(".")[-2] for n in fp8_e4m3_weights}),
        "dtype_counts": {},
    }
    for name in names:
        dtype = store.dtype(name)
        audit["dtype_counts"][dtype] = audit["dtype_counts"].get(dtype, 0) + 1
    if os.path.exists(args.nvfp4_identity):
        with open(args.nvfp4_identity) as handle:
            audit["source_identity"] = json.load(handle)
    tensors = []
    stats = []
    diagnostics = []
    missing_source = []
    missing_scale = []
    shape_mismatch = []
    plan = []
    for packed_name in packed:
        base = packed_name[: -len("_packed")]
        # base is already the BF16 source tensor name (``prefix.weight``); the
        # sibling metadata keeps that stem and appends ``_scale`` /``_global_scale``.
        source_name = base
        scale_name = base + "_scale"
        wgs_name = base + "_global_scale"
        if source_name not in bf16_map:
            missing_source.append(source_name)
            continue
        if scale_name not in store.tensors or wgs_name not in store.tensors:
            missing_scale.append(base)
            continue
        packed_shape = store.shape(packed_name)
        rows = int(packed_shape[0])
        k = int(packed_shape[1]) * 2
        shape = (rows, k)
        scale_codes = store.raw(scale_name)
        if tuple(scale_codes.shape) != (rows, k // NVFP4_BLOCK):
            shape_mismatch.append(
                {
                    "tensor": base,
                    "packed_shape": list(packed_shape),
                    "weight_scale_shape": list(scale_codes.shape),
                    "expected_weight_scale_shape": [rows, k // NVFP4_BLOCK],
                }
            )
            continue
        plan.append((base, source_name, shape))
    jobs = max(1, int(args.jobs))
    outcomes = {}
    errors = {}
    if jobs > 1 and len(plan) > 1:
        from concurrent.futures import ProcessPoolExecutor

        work = sorted(plan, key=lambda item: -int(item[2][0] * item[2][1]))
        print("[nvfp4] fanning out over %d workers" % jobs, flush=True)
        with ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_worker_init,
            initargs=(args.bf16_root, dict(bf16_map), args, audit["safetensors_file"]),
        ) as pool:
            for count, (base, acc, diag, error) in enumerate(
                pool.map(_worker_nvfp4_tensor, work, chunksize=1), 1
            ):
                if error is not None:
                    errors[base] = error
                else:
                    outcomes[base] = (acc, diag)
                if args.progress_every and (count % args.progress_every == 0 or count == len(work)):
                    print("[nvfp4] %d/%d tensors" % (count, len(work)), flush=True)
    else:
        for count, (base, source_name, shape) in enumerate(plan, 1):
            wgs = store.raw(base + "_global_scale").astype(np.float32).reshape(-1)
            try:
                acc, diag = measure_nvfp4_tensor(
                    bf16_tensor(source_name), shape, store.raw(base + "_scale"), float(wgs[0]), args
                )
            except ValueError as exc:
                errors[base] = str(exc)
                continue
            outcomes[base] = (acc, diag)
            if args.progress_every and (count % args.progress_every == 0 or count == len(plan)):
                print("[nvfp4] %d/%d tensors" % (count, len(plan)), flush=True)
    for base, source_name, shape in plan:
        if base in errors:
            result["unclassified"].append({"issue": errors[base], "tensor": base})
            continue
        if base not in outcomes:
            continue
        acc, diag = outcomes[base]
        rows, k = shape
        tensors.append(
            {
                "name": base,
                "source_bf16_tensor": source_name,
                "shape": list(shape),
                "elements": rows * k,
                "rules": {rule: finalize_rule(acc[rule]) for rule in NVFP4_RULES},
            }
        )
        stats.append({rule: acc[rule] for rule in NVFP4_RULES})
        diagnostics.append(dict(diag, tensor=base))
    audit["missing_bf16_source_tensors"] = missing_source
    audit["tensors_missing_weight_scale_or_global_scale"] = missing_scale
    audit["shape_mismatches"] = shape_mismatch
    excluded_elements = 0
    for name in fp8_e4m3_weights:
        excluded_elements += int(np.prod(store.shape(name)))
    audit["excluded_from_nvfp4_scan"] = {
        "reason": (
            "artifact ships these linears as FP8 E4M3 weights with BF16 per-row (real) "
            "scales, not as block-16 NVFP4 with E4M3 block scales"
        ),
        "tensor_count": len(fp8_e4m3_weights),
        "element_count": excluded_elements,
        "names": fp8_e4m3_weights,
    }
    result["audit"]["nvfp4"] = audit
    return {
        "artifact": "unsloth/Qwen3.8-27B-NVFP4 (safetensors)",
        "quantized_set_source": audit["safetensors_file"],
        "quantized_set_definition": "tensors ending in .weight_packed (NVFP4 blocks of 16 along K)",
        "excluded_from_scan": {
            "kind": "fp8_e4m3_weight_with_bf16_per_row_scale",
            "tensor_count": len(fp8_e4m3_weights),
            "element_count": excluded_elements,
            "note": "not NVFP4; listed in audit.nvfp4.excluded_from_nvfp4_scan",
        },
        "global_scale_rule": (
            "g = tensor_amax / 6 / 448 (spec); the artifact's own weight_global_scale is used "
            "for the stored-scale rule"
        ),
        "block_elements": NVFP4_BLOCK,
        "tensor_count": len(tensors),
        "element_count": sum(t["elements"] for t in tensors),
        "rules": NVFP4_RULES,
        "tensors": tensors,
        "aggregate": aggregate(stats),
        "diagnostics": diagnostics,
    }


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
