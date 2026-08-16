#!/usr/bin/env python3
"""Independent NumPy oracle for Qwen3.5 sparse-MoE routing metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct

import numpy as np

EXPERT_COUNT = 256
SELECTED_EXPERT_COUNT = 8
MX_BLOCK_SIZE = 32
E2M1_POSITIVE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


class MoeOracleError(ValueError):
    """A fail-closed oracle input or numerical contract violation."""


def round_to_bf16(values: np.ndarray) -> np.ndarray:
    """Return FP32 values rounded to BF16 with round-to-nearest-even."""

    source = np.asarray(values, dtype=np.float32)
    bits = source.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) & np.uint32(0xFFFF0000)).view(np.float32)


def router_logits(
    hidden: np.ndarray,
    weight: np.ndarray,
    *,
    bf16_inputs: bool = True,
    bf16_weight: bool = True,
) -> np.ndarray:
    """Compute `[M, E]` router logits with explicit FP32 accumulation."""

    hidden = np.asarray(hidden, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise MoeOracleError("router matrix shapes differ")
    if hidden.shape[0] == 0 or weight.shape[0] == 0:
        raise MoeOracleError("router matrix has a zero extent")
    if not np.isfinite(hidden).all() or not np.isfinite(weight).all():
        raise MoeOracleError("router matrix contains a nonfinite value")
    if bf16_inputs:
        hidden = round_to_bf16(hidden)
    if bf16_weight:
        weight = round_to_bf16(weight)
    return np.einsum("mk,ek->me", hidden, weight, dtype=np.float32)


def route_logits(logits: np.ndarray, selected: int = SELECTED_EXPERT_COUNT) -> dict[str, np.ndarray]:
    """Apply FP32 softmax, deterministic top-k, renormalization and grouping."""

    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise MoeOracleError("router logits must be a nonempty rank-2 matrix")
    if selected <= 0 or selected > logits.shape[1]:
        raise MoeOracleError("selected expert count is out of range")
    if not np.isfinite(logits).all():
        raise MoeOracleError("router logits contain a nonfinite value")

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted, dtype=np.float32)
    probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True, dtype=np.float32)
    expert_axis = np.arange(logits.shape[1], dtype=np.int64)
    ids = np.empty((logits.shape[0], selected), dtype=np.uint16)
    weights = np.empty((logits.shape[0], selected), dtype=np.float32)
    for token in range(logits.shape[0]):
        order = np.lexsort((expert_axis, -probabilities[token]))[:selected]
        chosen = probabilities[token, order]
        chosen = chosen / np.sum(chosen, dtype=np.float32)
        ids[token] = order.astype(np.uint16)
        weights[token] = chosen

    counts = np.bincount(ids.reshape(-1), minlength=logits.shape[1]).astype(np.uint32)
    offsets = np.empty(logits.shape[1] + 1, dtype=np.uint32)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.uint32, out=offsets[1:])
    pair_experts = ids.reshape(-1)
    pair_tokens = np.repeat(np.arange(logits.shape[0], dtype=np.uint32), selected)
    pair_slots = np.tile(np.arange(selected, dtype=np.uint16), logits.shape[0])
    grouping = np.lexsort((pair_slots, pair_tokens, pair_experts))
    return {
        "expert_ids": ids,
        "expert_weights": weights,
        "expert_counts": counts,
        "expert_offsets": offsets,
        "grouped_token_ids": pair_tokens[grouping],
        "grouped_topk_slots": pair_slots[grouping],
    }


def shared_expert_gate(hidden: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Compute the Qwen3.5 shared-expert sigmoid gate in FP32."""

    logits = router_logits(hidden, weight, bf16_inputs=True, bf16_weight=True)
    if logits.shape[1] != 1:
        raise MoeOracleError("shared expert gate must have one output")
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-logits, dtype=np.float32))


def decode_e8m0(codes: np.ndarray) -> np.ndarray:
    """Decode OCP E8M0 bytes, rejecting its reserved NaN encoding."""

    codes = np.asarray(codes, dtype=np.uint8)
    if np.any(codes == np.uint8(255)):
        raise MoeOracleError("E8M0 scale contains the reserved NaN encoding")
    bits = codes.astype(np.uint32) << np.uint32(23)
    bits = np.where(codes == 0, np.uint32(0x00400000), bits).astype(np.uint32)
    return bits.view(np.float32)


def decode_e2m1(codes: np.ndarray) -> np.ndarray:
    """Decode packed-element E2M1 codes without interpreting scale data."""

    codes = np.asarray(codes, dtype=np.uint8)
    magnitude = E2M1_POSITIVE[(codes & np.uint8(7)).astype(np.intp)]
    return np.where((codes & np.uint8(8)) == 0, magnitude, -magnitude).astype(np.float32)


def encode_e2m1(values: np.ndarray) -> np.ndarray:
    """Round FP32 to E2M1 using nearest, ties-to-even, and saturation."""

    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise MoeOracleError("MXFP4 element input contains a nonfinite value")
    magnitude = np.abs(values)[..., np.newaxis]
    errors = np.abs(magnitude - E2M1_POSITIVE)
    minimum = np.min(errors, axis=-1, keepdims=True)
    candidates = errors == minimum
    # Codes with an even least-significant digit win an exact midpoint tie.
    even = (np.arange(8, dtype=np.uint8) & np.uint8(1)) == 0
    preferred = candidates & even
    selected = np.where(
        np.any(preferred, axis=-1),
        np.argmax(preferred, axis=-1),
        np.argmax(candidates, axis=-1),
    ).astype(np.uint8)
    sign = np.where(np.signbit(values), np.uint8(8), np.uint8(0))
    return selected | sign


def mxfp4_even_scale_codes(maximum: np.ndarray) -> np.ndarray:
    """Apply the artifact's Quark `even` E8M0 scale calculation exactly."""

    maximum = np.asarray(maximum, dtype=np.float32)
    if np.any(maximum < 0) or not np.isfinite(maximum).all():
        raise MoeOracleError("MXFP4 block maximum must be finite and nonnegative")
    bits = maximum.view(np.uint32)
    rounded_exponent = (bits + np.uint32(0x00200000)) & np.uint32(0x7F800000)
    code = (rounded_exponent >> np.uint32(23)).astype(np.int32) - 2
    code = np.clip(code, 0, 254).astype(np.uint8)
    return np.where(maximum == 0, np.uint8(0), code).astype(np.uint8)


def quantize_mxfp4_rows(values: np.ndarray) -> dict[str, np.ndarray]:
    """Quantize rank-2 FP32/BF16 rows to row-padded OCP MXFP4 storage."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise MoeOracleError("MXFP4 input must be a nonempty rank-2 matrix")
    if not np.isfinite(values).all():
        raise MoeOracleError("MXFP4 input contains a nonfinite value")
    rows, columns = values.shape
    blocks = (columns + MX_BLOCK_SIZE - 1) // MX_BLOCK_SIZE
    padded = np.zeros((rows, blocks * MX_BLOCK_SIZE), dtype=np.float32)
    padded[:, :columns] = values
    grouped = padded.reshape(rows, blocks, MX_BLOCK_SIZE)
    scale_codes = mxfp4_even_scale_codes(np.max(np.abs(grouped), axis=-1))
    scales = decode_e8m0(scale_codes)
    codes = encode_e2m1(grouped / scales[..., np.newaxis]).reshape(rows, -1)[:, :columns]
    packed = np.zeros((rows, (columns + 1) // 2), dtype=np.uint8)
    packed[:, :] = codes[:, 0::2]
    if columns > 1:
        packed[:, : columns // 2] |= codes[:, 1::2] << np.uint8(4)
    decoded = decode_e2m1(codes) * np.repeat(scales, MX_BLOCK_SIZE, axis=1)[:, :columns]
    return {
        "packed": packed,
        "scale_codes": scale_codes,
        "element_codes": codes,
        "decoded": decoded.astype(np.float32),
    }


def decode_mxfp4_rows(
    packed: np.ndarray, scale_codes: np.ndarray, columns: int
) -> np.ndarray:
    """Decode row-padded packed MXFP4 plus a separate E8M0 scale plane."""

    packed = np.asarray(packed, dtype=np.uint8)
    scale_codes = np.asarray(scale_codes, dtype=np.uint8)
    if packed.ndim != 2 or scale_codes.ndim != 2 or columns <= 0:
        raise MoeOracleError("MXFP4 packed/scale planes have an invalid shape")
    rows = packed.shape[0]
    blocks = (columns + MX_BLOCK_SIZE - 1) // MX_BLOCK_SIZE
    if packed.shape[1] != (columns + 1) // 2 or scale_codes.shape != (rows, blocks):
        raise MoeOracleError("MXFP4 packed/scale plane lengths differ from the logical shape")
    codes = np.empty((rows, columns), dtype=np.uint8)
    codes[:, 0::2] = packed[:, : (columns + 1) // 2] & np.uint8(0x0F)
    if columns > 1:
        codes[:, 1::2] = packed[:, : columns // 2] >> np.uint8(4)
    scales = np.repeat(decode_e8m0(scale_codes), MX_BLOCK_SIZE, axis=1)[:, :columns]
    return (decode_e2m1(codes) * scales).astype(np.float32)


def mxfp4_w4a4_matmul(activation: np.ndarray, weight: np.ndarray) -> dict[str, np.ndarray]:
    """Quantize activation dynamically and apply an already-MXFP4 weight matrix."""

    activation_q = quantize_mxfp4_rows(round_to_bf16(activation))
    weight_q = quantize_mxfp4_rows(round_to_bf16(weight))
    output = np.einsum(
        "mk,nk->mn", activation_q["decoded"], weight_q["decoded"], dtype=np.float32
    )
    return {"output": output.astype(np.float32), "activation": activation_q, "weight": weight_q}


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


LAYER_BLOB_BYTES = 434_114_560


class SafeTensorReader:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.weight_map = json.loads(
            (root / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.headers: dict[str, tuple[int, dict[str, object]]] = {}

    def tensor(self, name: str) -> bytes:
        file_name = self.weight_map[name]
        if file_name not in self.headers:
            with (self.root / file_name).open("rb") as handle:
                header_size = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_size))
            self.headers[file_name] = (8 + header_size, header)
        data_start, header = self.headers[file_name]
        entry = header[name]
        begin, end = entry["data_offsets"]
        with (self.root / file_name).open("rb") as handle:
            handle.seek(data_start + begin)
            value = handle.read(end - begin)
        if len(value) != end - begin:
            raise MoeOracleError(f"short safetensors read: {name}")
        return value


def bf16_bytes_to_f32(value: bytes, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.frombuffer(value, dtype="<u2").astype(np.uint32) << np.uint32(16)
    return raw.view(np.float32).reshape(shape)


def actual_mxfp4(reader: SafeTensorReader, stem: str, shape: tuple[int, int]) -> np.ndarray:
    rows, columns = shape
    packed = np.frombuffer(reader.tensor(f"{stem}.weight"), dtype=np.uint8).reshape(
        rows, (columns + 1) // 2
    )
    scales = np.frombuffer(reader.tensor(f"{stem}.weight_scale"), dtype=np.uint8).reshape(
        rows, (columns + 31) // 32
    )
    return decode_mxfp4_rows(packed, scales, columns)


def build_actual_layer_fixture(
    root: pathlib.Path,
    layer: int,
    tokens: int,
    expert_start: int,
    blob_output: pathlib.Path,
    hidden_output: pathlib.Path,
    expected_output: pathlib.Path,
) -> dict[str, object]:
    if layer < 0 or layer >= 40:
        raise MoeOracleError("layer must be in [0,40)")
    if tokens <= 0 or expert_start < 0 or expert_start + SELECTED_EXPERT_COUNT > EXPERT_COUNT:
        raise MoeOracleError("actual fixture token/expert range is invalid")
    reader = SafeTensorReader(root)
    prefix = f"model.language_model.layers.{layer}.mlp"
    blob = bytearray()
    for projection in ("gate_proj", "up_proj"):
        for expert in range(EXPERT_COUNT):
            blob.extend(reader.tensor(f"{prefix}.experts.{expert}.{projection}.weight"))
        for expert in range(EXPERT_COUNT):
            blob.extend(reader.tensor(f"{prefix}.experts.{expert}.{projection}.weight_scale"))
    for expert in range(EXPERT_COUNT):
        blob.extend(reader.tensor(f"{prefix}.experts.{expert}.down_proj.weight"))
    for expert in range(EXPERT_COUNT):
        blob.extend(reader.tensor(f"{prefix}.experts.{expert}.down_proj.weight_scale"))
    for name in (
        "shared_expert.gate_proj.weight",
        "shared_expert.up_proj.weight",
        "shared_expert.down_proj.weight",
        "shared_expert_gate.weight",
    ):
        blob.extend(reader.tensor(f"{prefix}.{name}"))
    if len(blob) != LAYER_BLOB_BYTES:
        raise MoeOracleError(f"layer blob length differs: {len(blob)}")
    blob_output.write_bytes(blob)

    hidden = np.array(
        [
            [((column * 17 + token * 29) % 101 - 50) / 64.0 for column in range(2048)]
            for token in range(tokens)
        ],
        dtype=np.float32,
    )
    hidden = round_to_bf16(hidden)
    hidden_output.write_bytes((hidden.view(np.uint32) >> np.uint32(16)).astype("<u2").tobytes())
    activation = quantize_mxfp4_rows(hidden)["decoded"]
    routed = np.zeros((tokens, 2048), dtype=np.float32)
    selected_experts = list(range(expert_start, expert_start + SELECTED_EXPERT_COUNT))
    for expert in selected_experts:
        base = f"{prefix}.experts.{expert}"
        gate = activation @ actual_mxfp4(
            reader, f"{base}.gate_proj", (512, 2048)
        ).T
        up = activation @ actual_mxfp4(
            reader, f"{base}.up_proj", (512, 2048)
        ).T
        gate = round_to_bf16(gate)
        up = round_to_bf16(up)
        intermediate = round_to_bf16((gate / (1.0 + np.exp(-gate, dtype=np.float32))) * up)
        intermediate_q = quantize_mxfp4_rows(intermediate)["decoded"]
        down = intermediate_q @ actual_mxfp4(
            reader, f"{base}.down_proj", (2048, 512)
        ).T
        routed += round_to_bf16(down) / np.float32(SELECTED_EXPERT_COUNT)
    shared_gate_w = bf16_bytes_to_f32(
        reader.tensor(f"{prefix}.shared_expert.gate_proj.weight"), (512, 2048)
    )
    shared_up_w = bf16_bytes_to_f32(
        reader.tensor(f"{prefix}.shared_expert.up_proj.weight"), (512, 2048)
    )
    shared_down_w = bf16_bytes_to_f32(
        reader.tensor(f"{prefix}.shared_expert.down_proj.weight"), (2048, 512)
    )
    shared_selector = bf16_bytes_to_f32(
        reader.tensor(f"{prefix}.shared_expert_gate.weight"), (1, 2048)
    )
    shared_gate = round_to_bf16(hidden @ shared_gate_w.T)
    shared_up = round_to_bf16(hidden @ shared_up_w.T)
    shared_intermediate = round_to_bf16(
        (shared_gate / (1.0 + np.exp(-shared_gate, dtype=np.float32))) * shared_up
    )
    shared_down = round_to_bf16(shared_intermediate @ shared_down_w.T)
    selector_logit = np.einsum("mk,k->m", hidden, shared_selector[0], dtype=np.float32)
    selector = np.float32(1.0) / (np.float32(1.0) + np.exp(-selector_logit, dtype=np.float32))
    expected = round_to_bf16(routed + shared_down * selector[:, np.newaxis])
    expected_output.write_bytes(expected.astype("<f4").tobytes())
    return {
        "schema_version": "sllm-qwen35-moe-layer-fixture-v1",
        "layer": layer,
        "tokens": tokens,
        "blob_bytes": len(blob),
        "blob_sha256": hashlib.sha256(blob).hexdigest(),
        "hidden_sha256": _digest(hidden),
        "expected_sha256": _digest(expected),
        "selected_experts": selected_experts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=33)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=EXPERT_COUNT)
    parser.add_argument("--selected", type=int, default=SELECTED_EXPERT_COUNT)
    parser.add_argument("--seed", type=int, default=1902)
    parser.add_argument("--artifact", type=pathlib.Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-start", type=int, default=0)
    parser.add_argument("--blob-output", type=pathlib.Path)
    parser.add_argument("--hidden-output", type=pathlib.Path)
    parser.add_argument("--expected-output", type=pathlib.Path)
    args = parser.parse_args()
    fixture_paths = (args.blob_output, args.hidden_output, args.expected_output)
    if args.artifact is not None or any(path is not None for path in fixture_paths):
        if args.artifact is None or any(path is None for path in fixture_paths):
            raise MoeOracleError(
                "artifact fixture requires --artifact, --blob-output, --hidden-output and --expected-output"
            )
        print(
            json.dumps(
                build_actual_layer_fixture(
                    args.artifact,
                    args.layer,
                    args.tokens,
                    args.expert_start,
                    args.blob_output,
                    args.hidden_output,
                    args.expected_output,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.tokens <= 0 or args.hidden <= 0 or args.experts <= 0:
        raise MoeOracleError("dimensions must be positive")
    rng = np.random.default_rng(args.seed)
    hidden = rng.normal(0.0, 0.25, size=(args.tokens, args.hidden)).astype(np.float32)
    weight = rng.normal(0.0, 0.02, size=(args.experts, args.hidden)).astype(np.float32)
    logits = router_logits(hidden, weight)
    route = route_logits(logits, args.selected)
    print(
        json.dumps(
            {
                "schema_version": "sllm-qwen35-moe-router-oracle-v1",
                "tokens": args.tokens,
                "hidden": args.hidden,
                "experts": args.experts,
                "selected": args.selected,
                "logits_sha256": _digest(logits),
                **{f"{name}_sha256": _digest(value) for name, value in route.items()},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
