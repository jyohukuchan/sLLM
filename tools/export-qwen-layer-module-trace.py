#!/usr/bin/env python3
"""Export full-precision Qwen layer module traces for golden fixture inputs.

The script instantiates the model structure on the meta device, loads only the
selected decoder-layer weights from safetensors, and runs those layers on golden
``before`` tensors. This avoids loading the full model while still using the
Transformers implementation for module-level full-precision references.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from aq_scale_formats import scale_values
from safetensors import safe_open


SCHEMA_VERSION = "qwen-layer-module-trace-v0.5"


TOP_ABS_FEATURES = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--package-dir",
        type=Path,
        help="Optional uLLM package directory for AQ4 row-dot comparison.",
    )
    parser.add_argument(
        "--input-override-dir",
        type=Path,
        help="Optional directory containing layer-XXXX-input.f32 files to use instead of fixture before tensors.",
    )
    parser.add_argument("--layers", required=True, help="Comma-separated layer indices.")
    parser.add_argument("--hidden-index", type=int, default=3994)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def parse_layers(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("--layers requires at least one layer")
    return list(dict.fromkeys(values))


def torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def resolve_layers(model: torch.nn.Module) -> torch.nn.ModuleList | list[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    raise RuntimeError("cannot find decoder layers")


def resolve_rotary_embedding(model: torch.nn.Module) -> torch.nn.Module | None:
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        return model.model.rotary_emb
    if hasattr(model, "language_model") and hasattr(model.language_model, "rotary_emb"):
        return model.language_model.rotary_emb
    return None


def read_fixture_metadata(fixture: Path) -> dict[str, Any]:
    return json.loads((fixture / "metadata.json").read_text(encoding="utf-8"))


def fixture_layer_entry(metadata: dict[str, Any], layer_index: int) -> dict[str, Any]:
    for item in metadata.get("layers", []):
        if int(item.get("layer_index", -1)) == layer_index:
            return item
    raise KeyError(f"fixture does not contain layer {layer_index}")


def read_f32_tensor(path: Path, shape: list[int]) -> np.ndarray:
    data = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} values, expected {expected}")
    return data.reshape(shape)


def read_layer_input_tensor(
    fixture: Path,
    entry: dict[str, Any],
    layer_index: int,
    input_override_dir: Path | None,
) -> tuple[np.ndarray, str]:
    before_shape = [int(value) for value in entry["before_shape"]]
    if input_override_dir is not None:
        override_path = input_override_dir / f"layer-{layer_index:04d}-input.f32"
        if override_path.exists():
            metadata_path = input_override_dir / f"layer-{layer_index:04d}-input.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if int(metadata.get("layer_index", -1)) != layer_index:
                    raise ValueError(f"{metadata_path} layer_index does not match {layer_index}")
                if metadata.get("dtype") != "float32":
                    raise ValueError(f"{metadata_path} dtype is {metadata.get('dtype')}, expected float32")
                metadata_shape = [int(value) for value in metadata.get("shape", [])]
                if metadata_shape != before_shape:
                    raise ValueError(f"{metadata_path} shape {metadata_shape} does not match fixture {before_shape}")
            return read_f32_tensor(override_path, before_shape), str(override_path)
    fixture_path = fixture / str(entry["before_file"])
    return read_f32_tensor(fixture_path, before_shape), str(fixture_path)


def build_weight_file_map(model_dir: Path) -> dict[str, Path]:
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} does not contain a weight_map")
    return {name: model_dir / str(filename) for name, filename in weight_map.items()}


def read_package_manifest(package_dir: Path | None) -> dict[str, dict[str, Any]] | None:
    if package_dir is None:
        return None
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(item.get("name")): item for item in manifest.get("tensors", [])}


def read_f32_file(path: Path) -> torch.Tensor:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} length is not divisible by 4")
    return torch.tensor(struct.unpack(f"<{len(data) // 4}f", data), dtype=torch.float32)


def read_file_window(path: Path, offset: int, length: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(length)
    if len(data) != length:
        raise ValueError(f"{path} returned {len(data)} bytes at offset {offset}, expected {length}")
    return data


def decode_idx4_low_nibble_first(data: bytes, elements: int) -> torch.Tensor:
    packed = torch.tensor(list(data), dtype=torch.uint8)
    indices = torch.empty(packed.numel() * 2, dtype=torch.long)
    indices[0::2] = (packed & 0x0F).to(torch.long)
    indices[1::2] = ((packed >> 4) & 0x0F).to(torch.long)
    return indices[:elements]


def dequantize_package_row(package_dir: Path, tensor: dict[str, Any], row_index: int) -> np.ndarray:
    shape = tensor.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"{tensor.get('name')} shape is not 2D: {shape}")
    rows = int(shape[0])
    cols = int(shape[1])
    if row_index < 0 or row_index >= rows:
        raise ValueError(f"row {row_index} is outside shape {shape} for {tensor.get('name')}")

    group_size = int(tensor["group_size"])
    if cols % group_size != 0:
        raise ValueError(f"{tensor.get('name')} row cols {cols} is not divisible by group size {group_size}")

    row_start = row_index * cols
    index_data = read_file_window(package_dir / str(tensor["index_file"]), row_start // 2, math.ceil(cols / 2))
    scale_data = read_file_window(package_dir / str(tensor["scale_file"]), row_start // group_size, cols // group_size)
    codebook = read_f32_file(package_dir / str(tensor["codebook_file"]))
    if codebook.numel() != 16:
        raise ValueError(f"{tensor.get('codebook_file')} has {codebook.numel()} entries, expected 16")

    indices = decode_idx4_low_nibble_first(index_data, cols)
    scale_indices = torch.tensor(list(scale_data), dtype=torch.long)
    scales = scale_values(str(tensor["scale_format"])).to(torch.float32)
    combined_scales = scales[scale_indices] * float(tensor["tensor_scale"])
    row = codebook[indices] * combined_scales.repeat_interleave(group_size)[:cols]
    return row.numpy().astype(np.float32, copy=False)


def load_layer_state(model_dir: Path, weight_files: dict[str, Path], layer_index: int) -> dict[str, torch.Tensor]:
    prefix = f"model.language_model.layers.{layer_index}."
    selected: dict[Path, list[tuple[str, str]]] = defaultdict(list)
    for tensor_name, path in weight_files.items():
        if tensor_name.startswith(prefix):
            selected[path].append((tensor_name, tensor_name[len(prefix) :]))
    if not selected:
        raise ValueError(f"no safetensors weights found for layer {layer_index}")

    state: dict[str, torch.Tensor] = {}
    for path, names in selected.items():
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for full_name, local_name in names:
                state[local_name] = handle.get_tensor(full_name)
    return state


def compare_arrays(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: actual={actual.shape} expected={expected.shape}")
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    abs_diff = np.abs(diff)
    return {
        "count": int(diff.size),
        "mse": float(np.mean(diff * diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_abs_diff": float(np.max(abs_diff)),
    }


def capture_first_tensor(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                return capture_first_tensor(item)
            except TypeError:
                pass
    if isinstance(output, dict):
        for item in output.values():
            try:
                return capture_first_tensor(item)
            except TypeError:
                pass
    raise TypeError("module output did not contain a tensor")


def to_numpy_f32(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()


def tensor_to_numpy_f32(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()


def point_trace(
    token_index: int,
    hidden_index: int,
    before: np.ndarray,
    expected_after: np.ndarray,
    actual_after: np.ndarray,
    attention_output: np.ndarray,
    attention_block_output: np.ndarray,
    post_normed: np.ndarray,
    mlp_output: np.ndarray,
) -> dict[str, float | int]:
    idx = (0, token_index, hidden_index)
    actual_delta = float(actual_after[idx] - before[idx])
    expected_delta = float(expected_after[idx] - before[idx])
    output_diff = float(actual_after[idx] - expected_after[idx])
    return {
        "token_index": token_index,
        "hidden_index": hidden_index,
        "input": float(before[idx]),
        "attention_output": float(attention_output[idx]),
        "attention_block_output": float(attention_block_output[idx]),
        "post_normed": float(post_normed[idx]),
        "mlp_output": float(mlp_output[idx]),
        "actual_delta": actual_delta,
        "expected_delta": expected_delta,
        "delta_diff": actual_delta - expected_delta,
        "actual_output": float(actual_after[idx]),
        "expected_output": float(expected_after[idx]),
        "output_diff": output_diff,
        "abs_output_diff": abs(output_diff),
    }


def vector_summary(
    values: np.ndarray,
    token_index: int,
    sampled_feature_indices: list[int] | None = None,
    sampled_group_width: int | None = None,
) -> dict[str, Any]:
    values = np.asarray(values)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    finite_count = int(finite_values.size)
    nonfinite_count = int(values.size - finite_count)

    if finite_count == 0:
        min_value = 0.0
        max_value = 0.0
        max_abs = 0.0
        max_abs_index = 0
        mean = 0.0
        abs_mean = 0.0
        variance = 0.0
        stddev = 0.0
        rms = 0.0
        l2_norm = 0.0
    else:
        abs_values = np.abs(finite_values)
        min_value = float(finite_values.min())
        max_value = float(finite_values.max())
        finite_abs_values = np.abs(finite_values)
        max_abs_finite_index = int(np.argmax(finite_abs_values))
        finite_indices = np.nonzero(finite_mask)[0]
        max_abs_index = int(finite_indices[max_abs_finite_index])
        max_abs = float(finite_abs_values[max_abs_finite_index])
        mean = float(finite_values.mean())
        abs_mean = float(abs_values.mean())
        variance = float(finite_values.var())
        stddev = float(finite_values.std())
        rms = float(np.sqrt(np.mean(np.square(finite_values, dtype=np.float64), dtype=np.float64)))
        l2_norm = float(np.linalg.norm(finite_values.astype(np.float64)))

    if finite_values.size == 0:
        top_indices = np.array([], dtype=int)
        top_values = np.array([], dtype=np.float64)
    else:
        finite_abs = np.abs(finite_values)
        abs_desc_idx = np.argsort(finite_abs)[::-1]
        top_finite_indices = abs_desc_idx[: min(TOP_ABS_FEATURES, finite_values.size)]
        finite_indices = np.nonzero(finite_mask)[0]
        top_indices = finite_indices[top_finite_indices]
        top_values = finite_values[top_finite_indices]

    top_abs_features = [
        {
            "feature_index": int(index),
            "value": float(top_values[offset]),
            "abs_value": float(abs(top_values[offset])),
        }
        for offset, index in enumerate(top_indices)
    ]
    sampled_features = []
    if sampled_feature_indices:
        for index in sorted({int(index) for index in sampled_feature_indices}):
            if index < 0 or index >= values.size:
                continue
            value = float(values[index])
            sampled_features.append(
                {
                    "feature_index": index,
                    "value": value,
                    "abs_value": abs(value),
                }
            )
            if sampled_group_width is not None and sampled_group_width > 0:
                group_index = index // sampled_group_width
                group_start = group_index * sampled_group_width
                group_end = min(group_start + sampled_group_width, values.size)
                if group_start < group_end:
                    sampled_features[-1].update(
                        {
                            "group_index": group_index,
                            "group_offset": index - group_start,
                            "group_width": group_end - group_start,
                            "group_stats": sampled_group_stats(values[group_start:group_end]),
                        }
                    )

    summary = {
        "token_index": token_index,
        "feature_count": int(values.size),
        "stats": {
            "count": int(values.size),
            "finite_count": finite_count,
            "nonfinite_count": nonfinite_count,
            "mean": mean,
            "abs_mean": abs_mean,
            "variance": variance,
            "stddev": stddev,
            "rms": rms,
            "l2_norm": l2_norm,
            "min": min_value,
            "max": max_value,
            "max_abs": max_abs,
            "max_abs_index": max_abs_index,
        },
        "top_abs_features": top_abs_features,
    }
    if sampled_features:
        summary["sampled_features"] = sampled_features
    return summary


def sampled_group_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.size == 0:
        return {
            "count": int(values.size),
            "finite_count": 0,
            "nonfinite_count": int(values.size),
            "mean": 0.0,
            "abs_mean": 0.0,
            "rms": 0.0,
            "min": 0.0,
            "max": 0.0,
            "max_abs": 0.0,
            "max_abs_index": 0,
        }
    finite_abs = np.abs(finite_values)
    finite_indices = np.nonzero(finite_mask)[0]
    max_abs_finite_index = int(np.argmax(finite_abs))
    return {
        "count": int(values.size),
        "finite_count": int(finite_values.size),
        "nonfinite_count": int(values.size - finite_values.size),
        "mean": float(finite_values.mean()),
        "abs_mean": float(finite_abs.mean()),
        "rms": float(np.sqrt(np.mean(finite_values.astype(np.float64) * finite_values.astype(np.float64)))),
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
        "max_abs": float(finite_abs[max_abs_finite_index]),
        "max_abs_index": int(finite_indices[max_abs_finite_index]),
    }


def top_abs_feature_indices(values: np.ndarray, limit: int = TOP_ABS_FEATURES) -> list[int]:
    values = np.asarray(values)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.size == 0:
        return []
    finite_abs = np.abs(finite_values)
    abs_desc_idx = np.argsort(finite_abs)[::-1]
    top_finite_indices = abs_desc_idx[: min(limit, finite_values.size)]
    finite_indices = np.nonzero(finite_mask)[0]
    return [int(index) for index in finite_indices[top_finite_indices]]


def mapped_hot_feature_indices(
    values: np.ndarray,
    feature_dim: int,
    hidden: int,
    attention_hot_feature_indices: list[int],
) -> list[int]:
    if feature_dim == hidden:
        return list(attention_hot_feature_indices)
    head_width = 128
    if hidden % head_width == 0:
        value_heads = hidden // head_width
        if feature_dim == value_heads:
            indices = sorted({int(index) // head_width for index in attention_hot_feature_indices})
            indices = [index for index in indices if 0 <= index < feature_dim]
            if indices:
                return indices
        elif feature_dim % head_width == 0:
            feature_heads = feature_dim // head_width
            if feature_heads > 0 and feature_heads <= value_heads and value_heads % feature_heads == 0:
                value_heads_per_feature_head = value_heads // feature_heads
                indices = []
                for feature_index in attention_hot_feature_indices:
                    value_head = int(feature_index) // head_width
                    head_offset = int(feature_index) % head_width
                    feature_head = value_head // value_heads_per_feature_head
                    mapped_index = feature_head * head_width + head_offset
                    if 0 <= mapped_index < feature_dim:
                        indices.append(mapped_index)
                indices = sorted(set(indices))
                if indices:
                    return indices
        if feature_dim > hidden:
            v_base = feature_dim - hidden
            indices = sorted(
                {
                    v_base + int(feature_index)
                    for feature_index in attention_hot_feature_indices
                    if 0 <= v_base + int(feature_index) < feature_dim
                }
            )
            if indices:
                return indices
    return top_abs_feature_indices(values)


def hot_vector_projection_summary(
    values: np.ndarray,
    token_index: int,
    feature_dim: int,
    name: str,
    sampled_feature_indices: list[int] | None = None,
    sampled_group_width: int | None = None,
) -> dict[str, Any] | None:
    if not isinstance(values, np.ndarray):
        return None
    if values.ndim != 1:
        raise ValueError(f"{name} expected 1D hot token vector, got {values.ndim}")
    feature_count = values.size
    if feature_dim > feature_count:
        raise ValueError(f"{name} feature_dim={feature_dim} is larger than available {feature_count}")

    vector = values[:feature_dim].astype(np.float32)
    return vector_summary(vector, token_index, sampled_feature_indices, sampled_group_width)


def hot_input_vector_summaries(
    attention_projection_input: np.ndarray,
    mlp_activation: np.ndarray,
    token_index: int,
    hidden: int,
) -> dict[str, Any]:
    attention_vector = attention_projection_input[0, token_index][:hidden].astype(np.float32)
    attention_sampled_features = top_abs_feature_indices(attention_vector)
    mlp_vector = mlp_activation[0, token_index].astype(np.float32)
    mlp_sampled_features = top_abs_feature_indices(mlp_vector)
    hidden_group_width = 128 if hidden % 128 == 0 else None
    return {
        "attention_projection_input": hot_vector_projection_summary(
            attention_projection_input[0, token_index],
            token_index,
            hidden,
            "attention_projection_input",
            attention_sampled_features,
            hidden_group_width,
        ),
        "mlp_activation": hot_vector_projection_summary(
            mlp_activation[0, token_index],
            token_index,
            mlp_activation.shape[2],
            "mlp_activation",
            mlp_sampled_features,
            None,
        ),
    }


def add_token_vector_summary(
    target: dict[str, Any],
    name: str,
    values: np.ndarray,
    token_index: int,
    sampled_feature_indices: list[int] | None = None,
    sampled_group_width: int | None = None,
) -> None:
    if values.ndim != 3:
        raise ValueError(f"{name} expected [batch,seq,features], got {values.shape}")
    target[name] = hot_vector_projection_summary(
        values[0, token_index],
        token_index,
        values.shape[2],
        name,
        sampled_feature_indices,
        sampled_group_width,
    )


def causal_conv1d(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError(f"conv input expected [batch,seq,channels], got {values.shape}")
    if weight.ndim == 3:
        weight = weight[:, 0, :]
    if weight.ndim != 2:
        raise ValueError(f"conv weight expected [channels,kernel], got {weight.shape}")
    batch, sequence_len, channels = values.shape
    if weight.shape[0] != channels:
        raise ValueError(f"conv channels mismatch: input={channels} weight={weight.shape[0]}")
    kernel_size = weight.shape[1]
    output = np.zeros((batch, sequence_len, channels), dtype=np.float32)
    for kernel in range(kernel_size):
        left_padding = kernel_size - 1 - kernel
        if left_padding >= sequence_len:
            continue
        output[:, left_padding:, :] += values[:, : sequence_len - left_padding, :] * weight[:, kernel]
    return output.astype(np.float32)


def causal_conv1d_silu(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return silu(causal_conv1d(values, weight))


def silu(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return (values / (1.0 + np.exp(-values))).astype(np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def flatten_batch_sequence_features(name: str, values: np.ndarray) -> np.ndarray:
    if values.ndim == 3:
        return values
    if values.ndim == 4:
        return values.reshape(values.shape[0], values.shape[1], -1)
    raise ValueError(f"{name} expected [batch,seq,...] tensor, got {values.shape}")


def split_self_attention_q_gate(q_projection: np.ndarray, head_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if q_projection.ndim != 3:
        raise ValueError(f"q projection expected [batch,seq,features], got {q_projection.shape}")
    if head_dim <= 0 or q_projection.shape[2] % (head_dim * 2) != 0:
        raise ValueError(
            f"q projection feature count {q_projection.shape[2]} is not divisible by 2*head_dim={head_dim * 2}"
        )
    batch, sequence_len, _features = q_projection.shape
    grouped = q_projection.reshape(batch, sequence_len, -1, head_dim * 2)
    query, gate = np.split(grouped, 2, axis=-1)
    return query.reshape(batch, sequence_len, -1), gate.reshape(batch, sequence_len, -1)


def rmsnorm_by_head(values: np.ndarray, weight: np.ndarray, epsilon: float) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError(f"RMSNorm input expected [batch,seq,hidden], got {values.shape}")
    weight = weight.astype(np.float32).reshape(-1)
    if weight.size == 0:
        raise ValueError("RMSNorm weight must be non-empty")
    if values.shape[2] % weight.size != 0:
        raise ValueError(f"hidden={values.shape[2]} is not divisible by RMSNorm width={weight.size}")
    flat = values.astype(np.float32).reshape(-1, weight.size)
    mean_square = np.mean(flat * flat, axis=1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(mean_square + np.float32(epsilon))
    normed = flat * inv_rms * weight[None, :]
    return normed.reshape(values.shape).astype(np.float32)


def linear_attn_gate_beta(a: np.ndarray, b: np.ndarray, a_log: np.ndarray, dt_bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gate = -np.exp(a_log.astype(np.float32))[None, None, :] * np.logaddexp(
        0.0,
        a.astype(np.float32) + dt_bias.astype(np.float32)[None, None, :],
    )
    beta = 1.0 / (1.0 + np.exp(-b.astype(np.float32)))
    return gate.astype(np.float32), beta.astype(np.float32)


def split_linear_attn_qkv_for_recurrent(
    conv_output: np.ndarray,
    key_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    q_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if conv_output.ndim != 3:
        raise ValueError(f"conv output expected [batch,seq,features], got {conv_output.shape}")
    q_features = key_heads * key_dim
    k_features = key_heads * key_dim
    v_features = value_heads * value_dim
    expected_features = q_features + k_features + v_features
    if conv_output.shape[2] != expected_features:
        raise ValueError(f"conv output feature mismatch: got {conv_output.shape[2]} expected {expected_features}")

    q = conv_output[:, :, :q_features].astype(np.float32).reshape(
        conv_output.shape[0],
        conv_output.shape[1],
        key_heads,
        key_dim,
    )
    k = conv_output[:, :, q_features : q_features + k_features].astype(np.float32).reshape(
        conv_output.shape[0],
        conv_output.shape[1],
        key_heads,
        key_dim,
    )
    v = conv_output[:, :, q_features + k_features :].astype(np.float32)
    q_norm = np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + np.float32(1e-6))
    k_norm = np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + np.float32(1e-6))
    q = (q / q_norm) * np.float32(q_scale)
    k = k / k_norm
    q = q.reshape(conv_output.shape[0], conv_output.shape[1], q_features)
    k = k.reshape(conv_output.shape[0], conv_output.shape[1], k_features)
    return q.astype(np.float32), k.astype(np.float32), v.astype(np.float32)


def row_dot_trace(
    name: str,
    input_values: np.ndarray,
    module_output: np.ndarray,
    source_row: np.ndarray,
    package_row: np.ndarray | None,
    hidden_index: int,
) -> dict[str, Any]:
    if input_values.ndim != 3:
        raise ValueError(f"{name} input must be [batch,seq,features], got {input_values.shape}")
    if module_output.ndim != 3:
        raise ValueError(f"{name} output must be [batch,seq,hidden], got {module_output.shape}")
    sequence_len = input_values.shape[1]
    traces = []
    for token_index in range(sequence_len):
        vector = input_values[0, token_index].astype(np.float64)
        source_dot = float(np.dot(vector, source_row.astype(np.float64)))
        output_value = float(module_output[0, token_index, hidden_index])
        trace: dict[str, Any] = {
            "token_index": token_index,
            "hidden_index": hidden_index,
            "source_row_dot": source_dot,
            "module_output": output_value,
            "source_row_dot_error": source_dot - output_value,
        }
        if package_row is not None:
            package_dot = float(np.dot(vector, package_row.astype(np.float64)))
            trace.update(
                {
                    "package_row_dot": package_dot,
                    "package_row_dot_error_vs_source_row": package_dot - source_dot,
                    "package_row_dot_error_vs_module": package_dot - output_value,
                }
            )
        traces.append(trace)

    worst_source = max(traces, key=lambda item: abs(float(item["source_row_dot_error"])))
    worst_package = None
    if package_row is not None:
        worst_package = max(
            traces,
            key=lambda item: abs(float(item["package_row_dot_error_vs_source_row"])),
        )
    return {
        "name": name,
        "input_shape": list(input_values.shape),
        "source_row_l2_norm": float(np.linalg.norm(source_row.astype(np.float64))),
        "package_row_l2_norm": None if package_row is None else float(np.linalg.norm(package_row.astype(np.float64))),
        "per_token": traces,
        "worst_source_dot_error": worst_source,
        "worst_package_row_error": worst_package,
    }


def projection_row_dot_trace(
    name: str,
    input_values: np.ndarray,
    module_output: np.ndarray,
    source_weight: torch.Tensor,
    package_dir: Path | None,
    package_tensor: dict[str, Any] | None,
    token_index: int,
) -> dict[str, Any]:
    if input_values.ndim != 3:
        raise ValueError(f"{name} input must be [batch,seq,features], got {input_values.shape}")
    if module_output.ndim != 3:
        raise ValueError(f"{name} output must be [batch,seq,features], got {module_output.shape}")
    if source_weight.ndim != 2:
        raise ValueError(f"{name} source weight must be 2D, got {tuple(source_weight.shape)}")

    output_summary = vector_summary(module_output[0, token_index], token_index)
    selected_features = [
        int(item["feature_index"])
        for item in output_summary["top_abs_features"]
        if isinstance(item, dict) and "feature_index" in item
    ]
    vector = input_values[0, token_index].astype(np.float64)
    traces = []
    for feature_index in selected_features:
        source_row = tensor_to_numpy_f32(source_weight[feature_index])
        source_dot = float(np.dot(vector, source_row.astype(np.float64)))
        output_value = float(module_output[0, token_index, feature_index])
        trace: dict[str, Any] = {
            "token_index": token_index,
            "feature_index": feature_index,
            "source_row_dot": source_dot,
            "module_output": output_value,
            "source_row_dot_error": source_dot - output_value,
        }
        if package_dir is not None and package_tensor is not None:
            package_row = dequantize_package_row(package_dir, package_tensor, feature_index)
            package_dot = float(np.dot(vector, package_row.astype(np.float64)))
            trace.update(
                {
                    "package_row_dot": package_dot,
                    "package_row_dot_error_vs_source_row": package_dot - source_dot,
                    "package_row_dot_error_vs_module": package_dot - output_value,
                }
            )
        traces.append(trace)

    worst_package = None
    if traces and package_dir is not None and package_tensor is not None:
        worst_package = max(
            traces,
            key=lambda item: abs(float(item["package_row_dot_error_vs_source_row"])),
        )
    return {
        "name": name,
        "input_shape": list(input_values.shape),
        "output_shape": list(module_output.shape),
        "token_index": token_index,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "per_feature": traces,
        "worst_package_row_error": worst_package,
    }


def run_linear_attention_layer_trace(
    model_dir: Path,
    fixture: Path,
    metadata: dict[str, Any],
    weight_files: dict[str, Path],
    package_dir: Path | None,
    package_tensors: dict[str, dict[str, Any]] | None,
    input_override_dir: Path | None,
    layer: torch.nn.Module,
    layer_index: int,
    hidden_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    layer_type = getattr(layer, "layer_type", None)
    if layer_type != "linear_attention":
        raise ValueError(f"layer {layer_index} is {layer_type}; this trace currently supports linear_attention only")

    entry = fixture_layer_entry(metadata, layer_index)
    after_shape = [int(value) for value in entry["after_shape"]]
    before, input_source = read_layer_input_tensor(fixture, entry, layer_index, input_override_dir)
    expected_after = read_f32_tensor(fixture / str(entry["after_file"]), after_shape)
    if before.ndim != 3:
        raise ValueError(f"expected [batch,seq,hidden] fixture shape, got {before.shape}")
    if hidden_index < 0 or hidden_index >= before.shape[-1]:
        raise ValueError(f"hidden index {hidden_index} outside fixture hidden size {before.shape[-1]}")

    layer.to_empty(device=device)
    state = load_layer_state(model_dir, weight_files, layer_index)
    layer.load_state_dict(state, strict=True)
    layer.to(device=device, dtype=dtype)
    layer.eval()

    captured: dict[str, np.ndarray] = {}

    def hook(name: str):
        def _hook(_module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            captured[name] = to_numpy_f32(capture_first_tensor(output))

        return _hook

    def pre_hook(name: str):
        def _hook(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
            captured[name] = to_numpy_f32(capture_first_tensor(inputs))

        return _hook

    handles = [
        layer.input_layernorm.register_forward_hook(hook("attention_input_normed")),
        layer.linear_attn.register_forward_hook(hook("attention_output")),
        layer.linear_attn.in_proj_qkv.register_forward_hook(hook("attention_qkv_projection")),
        layer.linear_attn.in_proj_z.register_forward_hook(hook("attention_z_projection")),
        layer.linear_attn.in_proj_a.register_forward_hook(hook("attention_a_projection")),
        layer.linear_attn.in_proj_b.register_forward_hook(hook("attention_b_projection")),
        layer.linear_attn.norm.register_forward_pre_hook(pre_hook("attention_recurrent_flat")),
        layer.linear_attn.norm.register_forward_hook(hook("attention_gated_normed")),
        layer.linear_attn.out_proj.register_forward_pre_hook(pre_hook("attention_projection_input")),
        layer.post_attention_layernorm.register_forward_hook(hook("post_normed")),
        layer.mlp.down_proj.register_forward_pre_hook(pre_hook("mlp_activation")),
        layer.mlp.register_forward_hook(hook("mlp_output")),
    ]
    hidden_states = torch.from_numpy(before).to(device=device, dtype=dtype)
    empty_position_embeddings = (
        torch.empty(0, device=device, dtype=dtype),
        torch.empty(0, device=device, dtype=dtype),
    )
    with torch.inference_mode():
        try:
            output = layer(
                hidden_states,
                position_embeddings=empty_position_embeddings,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
            )
        finally:
            for handle in handles:
                handle.remove()
    actual_after = to_numpy_f32(capture_first_tensor(output))
    attention_input_normed = captured["attention_input_normed"]
    attention_qkv_projection = captured["attention_qkv_projection"]
    attention_z_projection = captured["attention_z_projection"]
    attention_a_projection = captured["attention_a_projection"]
    attention_b_projection = captured["attention_b_projection"]
    attention_conv_pre_silu = causal_conv1d(
        attention_qkv_projection,
        tensor_to_numpy_f32(state["linear_attn.conv1d.weight"]),
    )
    attention_conv = silu(attention_conv_pre_silu)
    value_dim = int(state["linear_attn.norm.weight"].numel())
    value_heads = int(before.shape[2] // value_dim)
    key_dim = value_dim
    key_heads = int((attention_qkv_projection.shape[2] - before.shape[2]) // (2 * key_dim))
    q_scale = 1.0 / math.sqrt(float(key_dim))
    attention_recurrent_q, attention_recurrent_k, attention_recurrent_v = split_linear_attn_qkv_for_recurrent(
        attention_conv,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        q_scale,
    )
    attention_gate, attention_beta = linear_attn_gate_beta(
        attention_a_projection,
        attention_b_projection,
        tensor_to_numpy_f32(state["linear_attn.A_log"]),
        tensor_to_numpy_f32(state["linear_attn.dt_bias"]),
    )
    attention_recurrent = captured["attention_recurrent_flat"].reshape(before.shape[0], before.shape[1], -1)
    attention_pre_gate_normed = rmsnorm_by_head(
        attention_recurrent,
        tensor_to_numpy_f32(state["linear_attn.norm.weight"]),
        1e-6,
    )
    attention_gate_silu = silu(attention_z_projection)
    attention_gated_normed = captured["attention_gated_normed"].reshape(before.shape[0], before.shape[1], -1)
    attention_projection_input = captured["attention_projection_input"]
    attention_output = captured["attention_output"]
    post_normed = captured["post_normed"]
    mlp_activation = captured["mlp_activation"]
    mlp_output = captured["mlp_output"]
    attention_block_output = before + attention_output

    expected_delta_for_hidden = expected_after[:, :, hidden_index] - before[:, :, hidden_index]
    token_index = int(np.argmax(np.abs(expected_delta_for_hidden.reshape(-1))) % before.shape[1])
    per_token = [
        point_trace(
            token,
            hidden_index,
            before,
            expected_after,
            actual_after,
            attention_output,
            attention_block_output,
            post_normed,
            mlp_output,
        )
        for token in range(before.shape[1])
    ]
    out_tensor_name = f"model.language_model.layers.{layer_index}.linear_attn.out_proj.weight"
    down_tensor_name = f"model.language_model.layers.{layer_index}.mlp.down_proj.weight"
    qkv_tensor_name = f"model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight"
    z_tensor_name = f"model.language_model.layers.{layer_index}.linear_attn.in_proj_z.weight"
    a_tensor_name = f"model.language_model.layers.{layer_index}.linear_attn.in_proj_a.weight"
    b_tensor_name = f"model.language_model.layers.{layer_index}.linear_attn.in_proj_b.weight"
    package_out_row = None
    package_down_row = None
    package_qkv_tensor = None
    package_z_tensor = None
    package_a_tensor = None
    package_b_tensor = None
    if package_dir is not None and package_tensors is not None:
        package_out_tensor = package_tensors.get(out_tensor_name)
        package_down_tensor = package_tensors.get(down_tensor_name)
        package_qkv_tensor = package_tensors.get(qkv_tensor_name)
        package_z_tensor = package_tensors.get(z_tensor_name)
        package_a_tensor = package_tensors.get(a_tensor_name)
        package_b_tensor = package_tensors.get(b_tensor_name)
        if package_out_tensor is not None:
            package_out_row = dequantize_package_row(package_dir, package_out_tensor, hidden_index)
        if package_down_tensor is not None:
            package_down_row = dequantize_package_row(package_dir, package_down_tensor, hidden_index)

    row_dot = {
        "attention_out_proj": row_dot_trace(
            "attention_out_proj",
            attention_projection_input,
            attention_output,
            tensor_to_numpy_f32(state["linear_attn.out_proj.weight"][hidden_index]),
            package_out_row,
            hidden_index,
        ),
        "mlp_down_proj": row_dot_trace(
            "mlp_down_proj",
            mlp_activation,
            mlp_output,
            tensor_to_numpy_f32(state["mlp.down_proj.weight"][hidden_index]),
            package_down_row,
            hidden_index,
        ),
    }
    projection_row_dot = {
        "attention_qkv_projection": projection_row_dot_trace(
            "attention_qkv_projection",
            attention_input_normed,
            attention_qkv_projection,
            state["linear_attn.in_proj_qkv.weight"],
            package_dir,
            package_qkv_tensor,
            token_index,
        ),
        "attention_z_projection": projection_row_dot_trace(
            "attention_z_projection",
            attention_input_normed,
            attention_z_projection,
            state["linear_attn.in_proj_z.weight"],
            package_dir,
            package_z_tensor,
            token_index,
        ),
        "attention_a_projection": projection_row_dot_trace(
            "attention_a_projection",
            attention_input_normed,
            attention_a_projection,
            state["linear_attn.in_proj_a.weight"],
            package_dir,
            package_a_tensor,
            token_index,
        ),
        "attention_b_projection": projection_row_dot_trace(
            "attention_b_projection",
            attention_input_normed,
            attention_b_projection,
            state["linear_attn.in_proj_b.weight"],
            package_dir,
            package_b_tensor,
            token_index,
        ),
    }
    hot_input_vectors = hot_input_vector_summaries(
        attention_projection_input,
        mlp_activation,
        token_index,
        before.shape[2],
    )
    for name, values in (
        ("attention_input_normed", attention_input_normed),
        ("attention_qkv_projection", attention_qkv_projection),
        ("attention_z_projection", attention_z_projection),
        ("attention_gate_silu", attention_gate_silu),
        ("attention_a_projection", attention_a_projection),
        ("attention_b_projection", attention_b_projection),
        ("attention_conv_pre_silu", attention_conv_pre_silu),
        ("attention_conv", attention_conv),
        ("attention_recurrent_q", attention_recurrent_q),
        ("attention_recurrent_k", attention_recurrent_k),
        ("attention_recurrent_v", attention_recurrent_v),
        ("attention_gate", attention_gate),
        ("attention_beta", attention_beta),
        ("attention_recurrent", attention_recurrent),
        ("attention_pre_gate_normed", attention_pre_gate_normed),
        ("attention_gated_normed", attention_gated_normed),
    ):
        sampled_features = None
        if values.shape[2] == before.shape[2]:
            sampled_features = [
                int(item["feature_index"])
                for item in hot_input_vectors["attention_projection_input"]["top_abs_features"]
            ]
        else:
            sampled_features = mapped_hot_feature_indices(
                values[0, token_index],
                values.shape[2],
                before.shape[2],
                [
                    int(item["feature_index"])
                    for item in hot_input_vectors["attention_projection_input"]["top_abs_features"]
                ],
            )
        sampled_group_width = 128 if values.shape[2] % 128 == 0 else None
        add_token_vector_summary(
            hot_input_vectors,
            name,
            values,
            token_index,
            sampled_features,
            sampled_group_width,
        )
    per_token_hot_input_vectors = [
        {
            "token_index": token,
            **hot_input_vector_summaries(
                attention_projection_input,
                mlp_activation,
                token,
                before.shape[2],
            ),
        }
        for token in range(before.shape[1])
    ]
    for item in per_token_hot_input_vectors:
        token = int(item["token_index"])
        for name, values in (
            ("attention_input_normed", attention_input_normed),
            ("attention_qkv_projection", attention_qkv_projection),
            ("attention_z_projection", attention_z_projection),
            ("attention_gate_silu", attention_gate_silu),
            ("attention_a_projection", attention_a_projection),
            ("attention_b_projection", attention_b_projection),
            ("attention_conv_pre_silu", attention_conv_pre_silu),
            ("attention_conv", attention_conv),
            ("attention_recurrent_q", attention_recurrent_q),
            ("attention_recurrent_k", attention_recurrent_k),
            ("attention_recurrent_v", attention_recurrent_v),
            ("attention_gate", attention_gate),
            ("attention_beta", attention_beta),
            ("attention_recurrent", attention_recurrent),
            ("attention_pre_gate_normed", attention_pre_gate_normed),
            ("attention_gated_normed", attention_gated_normed),
        ):
            sampled_features = None
            if values.shape[2] == before.shape[2]:
                sampled_features = [
                    int(feature["feature_index"])
                    for feature in item["attention_projection_input"]["top_abs_features"]
                ]
            else:
                sampled_features = mapped_hot_feature_indices(
                    values[0, token],
                    values.shape[2],
                    before.shape[2],
                    [
                        int(feature["feature_index"])
                        for feature in item["attention_projection_input"]["top_abs_features"]
                    ],
                )
            sampled_group_width = 128 if values.shape[2] % 128 == 0 else None
            add_token_vector_summary(
                item,
                name,
                values,
                token,
                sampled_features,
                sampled_group_width,
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "command": "export-qwen-layer-module-trace",
        "model_dir": str(model_dir),
        "fixture": str(fixture),
        "input_source": input_source,
        "input_override_dir": None if input_override_dir is None else str(input_override_dir),
        "package_dir": None if package_dir is None else str(package_dir),
        "layer_index": layer_index,
        "layer_type": layer_type,
        "hidden_index": hidden_index,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "shape": list(actual_after.shape),
        "fixture_match": compare_arrays(actual_after, expected_after),
        "max_expected_delta_trace": per_token[token_index],
        "per_token_hidden_trace": per_token,
        "module_contribution": {
            "hot_input_vectors": hot_input_vectors,
            "per_token_hot_input_vectors": per_token_hot_input_vectors,
        },
        "row_dot": row_dot,
        "projection_row_dot": projection_row_dot,
    }


def run_self_attention_layer_trace(
    model_dir: Path,
    fixture: Path,
    metadata: dict[str, Any],
    weight_files: dict[str, Path],
    package_dir: Path | None,
    package_tensors: dict[str, dict[str, Any]] | None,
    input_override_dir: Path | None,
    layer: torch.nn.Module,
    layer_index: int,
    hidden_index: int,
    device: torch.device,
    dtype: torch.dtype,
    rotary_template: torch.nn.Module | None,
) -> dict[str, Any]:
    layer_type = getattr(layer, "layer_type", None)
    if layer_type != "full_attention":
        raise ValueError(f"layer {layer_index} is {layer_type}; this trace expected full_attention")
    if rotary_template is None:
        raise ValueError("full_attention trace requires a model rotary embedding module")

    entry = fixture_layer_entry(metadata, layer_index)
    after_shape = [int(value) for value in entry["after_shape"]]
    before, input_source = read_layer_input_tensor(fixture, entry, layer_index, input_override_dir)
    expected_after = read_f32_tensor(fixture / str(entry["after_file"]), after_shape)
    if before.ndim != 3:
        raise ValueError(f"expected [batch,seq,hidden] fixture shape, got {before.shape}")
    if hidden_index < 0 or hidden_index >= before.shape[-1]:
        raise ValueError(f"hidden index {hidden_index} outside fixture hidden size {before.shape[-1]}")

    layer.to_empty(device=device)
    state = load_layer_state(model_dir, weight_files, layer_index)
    layer.load_state_dict(state, strict=True)
    layer.to(device=device, dtype=dtype)
    layer.eval()

    rotary_embedding = type(rotary_template)(rotary_template.config, device=device)
    rotary_embedding.to(device=device, dtype=dtype)
    rotary_embedding.eval()

    captured: dict[str, np.ndarray] = {}

    def hook(name: str):
        def _hook(_module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            captured[name] = to_numpy_f32(capture_first_tensor(output))

        return _hook

    def pre_hook(name: str):
        def _hook(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
            captured[name] = to_numpy_f32(capture_first_tensor(inputs))

        return _hook

    handles = [
        layer.input_layernorm.register_forward_hook(hook("attention_input_normed")),
        layer.self_attn.register_forward_hook(hook("attention_output")),
        layer.self_attn.q_proj.register_forward_hook(hook("self_attention_q_projection")),
        layer.self_attn.k_proj.register_forward_hook(hook("self_attention_k_projection")),
        layer.self_attn.v_proj.register_forward_hook(hook("self_attention_v_projection")),
        layer.self_attn.q_norm.register_forward_hook(hook("self_attention_q_normed")),
        layer.self_attn.k_norm.register_forward_hook(hook("self_attention_k_normed")),
        layer.self_attn.o_proj.register_forward_pre_hook(pre_hook("attention_projection_input")),
        layer.post_attention_layernorm.register_forward_hook(hook("post_normed")),
        layer.mlp.down_proj.register_forward_pre_hook(pre_hook("mlp_activation")),
        layer.mlp.register_forward_hook(hook("mlp_output")),
    ]
    hidden_states = torch.from_numpy(before).to(device=device, dtype=dtype)
    batch_size = hidden_states.shape[0]
    sequence_len = hidden_states.shape[1]
    position_ids = torch.arange(sequence_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    rotary_position_ids = position_ids[None, ...].expand(3, batch_size, -1)
    with torch.inference_mode():
        try:
            position_embeddings = rotary_embedding(hidden_states, rotary_position_ids)
            output = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
            )
        finally:
            for handle in handles:
                handle.remove()
    actual_after = to_numpy_f32(capture_first_tensor(output))

    attention_input_normed = captured["attention_input_normed"]
    self_attention_q_projection = captured["self_attention_q_projection"]
    self_attention_k_projection = captured["self_attention_k_projection"]
    self_attention_v_projection = captured["self_attention_v_projection"]
    self_attention_q_normed = flatten_batch_sequence_features(
        "self_attention_q_normed",
        captured["self_attention_q_normed"],
    )
    self_attention_k_normed = flatten_batch_sequence_features(
        "self_attention_k_normed",
        captured["self_attention_k_normed"],
    )
    head_dim = int(layer.self_attn.head_dim)
    self_attention_query_projection, self_attention_gate_projection = split_self_attention_q_gate(
        self_attention_q_projection,
        head_dim,
    )
    self_attention_gate_sigmoid = sigmoid(self_attention_gate_projection)
    attention_projection_input = captured["attention_projection_input"]
    attention_output = captured["attention_output"]
    post_normed = captured["post_normed"]
    mlp_activation = captured["mlp_activation"]
    mlp_output = captured["mlp_output"]
    attention_block_output = before + attention_output

    expected_delta_for_hidden = expected_after[:, :, hidden_index] - before[:, :, hidden_index]
    token_index = int(np.argmax(np.abs(expected_delta_for_hidden.reshape(-1))) % before.shape[1])
    per_token = [
        point_trace(
            token,
            hidden_index,
            before,
            expected_after,
            actual_after,
            attention_output,
            attention_block_output,
            post_normed,
            mlp_output,
        )
        for token in range(before.shape[1])
    ]

    o_tensor_name = f"model.language_model.layers.{layer_index}.self_attn.o_proj.weight"
    down_tensor_name = f"model.language_model.layers.{layer_index}.mlp.down_proj.weight"
    q_tensor_name = f"model.language_model.layers.{layer_index}.self_attn.q_proj.weight"
    k_tensor_name = f"model.language_model.layers.{layer_index}.self_attn.k_proj.weight"
    v_tensor_name = f"model.language_model.layers.{layer_index}.self_attn.v_proj.weight"
    package_o_row = None
    package_down_row = None
    package_q_tensor = None
    package_k_tensor = None
    package_v_tensor = None
    if package_dir is not None and package_tensors is not None:
        package_o_tensor = package_tensors.get(o_tensor_name)
        package_down_tensor = package_tensors.get(down_tensor_name)
        package_q_tensor = package_tensors.get(q_tensor_name)
        package_k_tensor = package_tensors.get(k_tensor_name)
        package_v_tensor = package_tensors.get(v_tensor_name)
        if package_o_tensor is not None:
            package_o_row = dequantize_package_row(package_dir, package_o_tensor, hidden_index)
        if package_down_tensor is not None:
            package_down_row = dequantize_package_row(package_dir, package_down_tensor, hidden_index)

    row_dot = {
        "self_attention_o_proj": row_dot_trace(
            "self_attention_o_proj",
            attention_projection_input,
            attention_output,
            tensor_to_numpy_f32(state["self_attn.o_proj.weight"][hidden_index]),
            package_o_row,
            hidden_index,
        ),
        "mlp_down_proj": row_dot_trace(
            "mlp_down_proj",
            mlp_activation,
            mlp_output,
            tensor_to_numpy_f32(state["mlp.down_proj.weight"][hidden_index]),
            package_down_row,
            hidden_index,
        ),
    }
    projection_row_dot = {
        "self_attention_q_projection": projection_row_dot_trace(
            "self_attention_q_projection",
            attention_input_normed,
            self_attention_q_projection,
            state["self_attn.q_proj.weight"],
            package_dir,
            package_q_tensor,
            token_index,
        ),
        "self_attention_k_projection": projection_row_dot_trace(
            "self_attention_k_projection",
            attention_input_normed,
            self_attention_k_projection,
            state["self_attn.k_proj.weight"],
            package_dir,
            package_k_tensor,
            token_index,
        ),
        "self_attention_v_projection": projection_row_dot_trace(
            "self_attention_v_projection",
            attention_input_normed,
            self_attention_v_projection,
            state["self_attn.v_proj.weight"],
            package_dir,
            package_v_tensor,
            token_index,
        ),
    }

    hot_input_vectors = hot_input_vector_summaries(
        attention_projection_input,
        mlp_activation,
        token_index,
        before.shape[2],
    )
    summary_values = (
        ("attention_input_normed", attention_input_normed),
        ("self_attention_q_projection", self_attention_q_projection),
        ("self_attention_query_projection", self_attention_query_projection),
        ("self_attention_gate_projection", self_attention_gate_projection),
        ("self_attention_gate_sigmoid", self_attention_gate_sigmoid),
        ("self_attention_k_projection", self_attention_k_projection),
        ("self_attention_v_projection", self_attention_v_projection),
        ("self_attention_q_normed", self_attention_q_normed),
        ("self_attention_k_normed", self_attention_k_normed),
    )
    for name, values in summary_values:
        if values.shape[2] == before.shape[2]:
            sampled_features = [
                int(item["feature_index"])
                for item in hot_input_vectors["attention_projection_input"]["top_abs_features"]
            ]
        else:
            sampled_features = mapped_hot_feature_indices(
                values[0, token_index],
                values.shape[2],
                before.shape[2],
                [
                    int(item["feature_index"])
                    for item in hot_input_vectors["attention_projection_input"]["top_abs_features"]
                ],
            )
        sampled_group_width = 128 if values.shape[2] % 128 == 0 else None
        add_token_vector_summary(
            hot_input_vectors,
            name,
            values,
            token_index,
            sampled_features,
            sampled_group_width,
        )

    per_token_hot_input_vectors = [
        {
            "token_index": token,
            **hot_input_vector_summaries(
                attention_projection_input,
                mlp_activation,
                token,
                before.shape[2],
            ),
        }
        for token in range(before.shape[1])
    ]
    for item in per_token_hot_input_vectors:
        token = int(item["token_index"])
        for name, values in summary_values:
            if values.shape[2] == before.shape[2]:
                sampled_features = [
                    int(feature["feature_index"])
                    for feature in item["attention_projection_input"]["top_abs_features"]
                ]
            else:
                sampled_features = mapped_hot_feature_indices(
                    values[0, token],
                    values.shape[2],
                    before.shape[2],
                    [
                        int(feature["feature_index"])
                        for feature in item["attention_projection_input"]["top_abs_features"]
                    ],
                )
            sampled_group_width = 128 if values.shape[2] % 128 == 0 else None
            add_token_vector_summary(
                item,
                name,
                values,
                token,
                sampled_features,
                sampled_group_width,
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "command": "export-qwen-layer-module-trace",
        "model_dir": str(model_dir),
        "fixture": str(fixture),
        "input_source": input_source,
        "input_override_dir": None if input_override_dir is None else str(input_override_dir),
        "package_dir": None if package_dir is None else str(package_dir),
        "layer_index": layer_index,
        "layer_type": layer_type,
        "hidden_index": hidden_index,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "shape": list(actual_after.shape),
        "fixture_match": compare_arrays(actual_after, expected_after),
        "max_expected_delta_trace": per_token[token_index],
        "per_token_hidden_trace": per_token,
        "module_contribution": {
            "hot_input_vectors": hot_input_vectors,
            "per_token_hot_input_vectors": per_token_hot_input_vectors,
        },
        "row_dot": row_dot,
        "projection_row_dot": projection_row_dot,
    }


def run_layer_trace(
    model_dir: Path,
    fixture: Path,
    metadata: dict[str, Any],
    weight_files: dict[str, Path],
    package_dir: Path | None,
    package_tensors: dict[str, dict[str, Any]] | None,
    input_override_dir: Path | None,
    layer: torch.nn.Module,
    layer_index: int,
    hidden_index: int,
    device: torch.device,
    dtype: torch.dtype,
    rotary_template: torch.nn.Module | None,
) -> dict[str, Any]:
    layer_type = getattr(layer, "layer_type", None)
    if layer_type == "linear_attention":
        return run_linear_attention_layer_trace(
            model_dir,
            fixture,
            metadata,
            weight_files,
            package_dir,
            package_tensors,
            input_override_dir,
            layer,
            layer_index,
            hidden_index,
            device,
            dtype,
        )
    if layer_type == "full_attention":
        return run_self_attention_layer_trace(
            model_dir,
            fixture,
            metadata,
            weight_files,
            package_dir,
            package_tensors,
            input_override_dir,
            layer,
            layer_index,
            hidden_index,
            device,
            dtype,
            rotary_template,
        )
    raise ValueError(f"layer {layer_index} is {layer_type}; unsupported layer type")


def fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return str(value)


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| layer | hidden | token | fixture_max_abs | expected_delta | attn | mlp | actual_delta | output_diff | out_pkg_err | down_pkg_err |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        trace = row["max_expected_delta_trace"]
        token = int(trace["token_index"])
        row_dot = row.get("row_dot", {})
        out_projection = "attention_out_proj" if "attention_out_proj" in row_dot else "self_attention_o_proj"
        out_entries = row_dot.get(out_projection, {}).get("per_token", [])
        down_entries = row_dot.get("mlp_down_proj", {}).get("per_token", [])
        out_row = out_entries[token] if token < len(out_entries) else {}
        down_row = down_entries[token] if token < len(down_entries) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["layer_index"]),
                    str(row["hidden_index"]),
                    str(trace["token_index"]),
                    fmt(row["fixture_match"]["max_abs_diff"]),
                    fmt(trace["expected_delta"]),
                    fmt(trace["attention_output"]),
                    fmt(trace["mlp_output"]),
                    fmt(trace["actual_delta"]),
                    fmt(trace["output_diff"]),
                    fmt(out_row.get("package_row_dot_error_vs_source_row")),
                    fmt(down_row.get("package_row_dot_error_vs_source_row")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    fixture = args.fixture.expanduser().resolve()
    package_dir = args.package_dir.expanduser().resolve() if args.package_dir else None
    input_override_dir = args.input_override_dir.expanduser().resolve() if args.input_override_dir else None
    layers = parse_layers(args.layers)
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        str(model_dir),
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
    model_layers = resolve_layers(model)
    rotary_template = resolve_rotary_embedding(model)
    metadata = read_fixture_metadata(fixture)
    weight_files = build_weight_file_map(model_dir)
    package_tensors = read_package_manifest(package_dir)

    rows = []
    for layer_index in layers:
        if layer_index < 0 or layer_index >= len(model_layers):
            raise ValueError(f"layer {layer_index} outside model range 0..{len(model_layers) - 1}")
        rows.append(
            run_layer_trace(
                model_dir,
                fixture,
                metadata,
                weight_files,
                package_dir,
                package_tensors,
                input_override_dir,
                model_layers[layer_index],
                layer_index,
                args.hidden_index,
                device,
                dtype,
                rotary_template,
            )
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(rows), encoding="utf-8")
    print(f"qwen-layer-module-trace rows={len(rows)} output={args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
