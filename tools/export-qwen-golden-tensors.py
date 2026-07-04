#!/usr/bin/env python3
"""Export selected Qwen decoder-layer activations from Hugging Face Transformers.

The script captures hidden-state input/output for selected decoder layers and stores
them as raw little-endian float32 binaries plus a small JSON metadata file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch


FIXTURE_FORMAT = "qwen_golden_tensor_v1"
FIXTURE_FORMAT_VERSION = "0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to Hugging Face model directory")
    parser.add_argument(
        "--token-ids",
        type=str,
        required=True,
        help="Comma-separated token ids to run through the model",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--layers",
        type=str,
        help="Comma-separated 0-based decoder layer indices",
    )
    selection.add_argument(
        "--layer-range",
        type=str,
        help="Layer range as START:END (END exclusive)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory for fixture files")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="torch device name (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Model dtype for load-time casting",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to transformers loader",
    )
    return parser.parse_args()


def parse_comma_int_list(raw: str, name: str, *, allow_negative: bool = False) -> list[int]:
    parts = [item.strip() for item in raw.split(",") if item.strip() != ""]
    if not parts:
        raise ValueError(f"--{name} requires at least one integer")
    values: list[int] = []
    for part in parts:
        value = int(part)
        if value < 0 and not allow_negative:
            raise ValueError(f"--{name} must be non-negative: {value}")
        values.append(value)
    return values


def parse_layer_range(raw: str) -> tuple[int, int]:
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(f"--layer-range must be START:END, got: {raw}")
    start_raw, end_raw = parts
    if start_raw == "" or end_raw == "":
        raise ValueError(f"--layer-range must be START:END, got: {raw}")
    start = int(start_raw)
    end = int(end_raw)
    if start < 0 or end < 0:
        raise ValueError(f"--layer-range values must be non-negative, got: {start}:{end}")
    if start >= end:
        raise ValueError(f"--layer-range requires START < END, got: {start}:{end}")
    return (start, end)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(name)


def parse_dtype(name: str) -> torch.dtype | None:
    if name == "auto":
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def resolve_layers(model: torch.nn.Module) -> torch.nn.ModuleList | list[torch.nn.Module]:
    if hasattr(model, "model"):
        model_layers = getattr(model, "model")
        if hasattr(model_layers, "layers"):
            return getattr(model_layers, "layers")
    if hasattr(model, "language_model"):
        language_layers = getattr(model, "language_model")
        if hasattr(language_layers, "layers"):
            return getattr(language_layers, "layers")
    raise RuntimeError("cannot find decoder layers on model (expected model.model.layers or model.language_model.layers)")


def first_tensor(obj: object) -> torch.Tensor | None:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            value = first_tensor(item)
            if value is not None:
                return value
    if isinstance(obj, dict):
        for item in obj.values():
            value = first_tensor(item)
            if value is not None:
                return value
    return None


def to_cpu_f32_contig(tensor: torch.Tensor) -> np.ndarray:
    np_tensor = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return np_tensor.astype("<f4", copy=False)


def capture_tensor_cpu_f32(tensor: torch.Tensor) -> np.ndarray:
    return to_cpu_f32_contig(tensor)


def write_f32_file(path: Path, array: np.ndarray) -> list[int]:
    array.ravel().tofile(path)
    return [int(x) for x in array.shape]


def load_model(model_dir: Path, device: torch.device, dtype_name: str, trust_remote_code: bool):
    from transformers import AutoModelForCausalLM

    requested_dtype = parse_dtype(dtype_name)
    kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": True,
    }
    if requested_dtype is not None:
        kwargs["torch_dtype"] = requested_dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            **{k: v for k, v in kwargs.items() if v is not None},
        )
    except TypeError as exc:
        if requested_dtype is None or "torch_dtype" not in str(exc):
            raise
        # Older transformers versions may accept `dtype` instead.
        kwargs.pop("torch_dtype")
        kwargs["dtype"] = requested_dtype
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            **kwargs,
        )
    model.to(device)
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    token_ids = parse_comma_int_list(args.token_ids, "token-ids")
    is_prefix_mode = args.layer_range is not None
    if is_prefix_mode:
        layer_start, layer_end_exclusive = parse_layer_range(args.layer_range)
        layer_indices = list(range(layer_start, layer_end_exclusive))
    else:
        layer_indices = parse_comma_int_list(args.layers, "layers")

    if len(token_ids) < 1:
        raise SystemExit("--token-ids must include at least one id")
    if len(layer_indices) < 1:
        raise SystemExit("--layers / --layer-range must include at least one index")

    args.model_dir = args.model_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model = load_model(args.model_dir, device, args.dtype, args.trust_remote_code)
    layers = resolve_layers(model)
    layer_count = len(layers)
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= layer_count:
            raise SystemExit(
                f"{'--layer-range' if is_prefix_mode else '--layers'} contains out-of-range index "
                f"{layer_idx} (0..{layer_count - 1})"
            )

    selected = list(dict.fromkeys(layer_indices))
    before: dict[int, np.ndarray] = {}
    after: dict[int, np.ndarray] = {}

    def make_pre_hook(layer_index: int) -> Callable:
        def pre_hook(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
            if layer_index in before:
                return
            if not inputs:
                return
            tensor = first_tensor(inputs[0]) if len(inputs) > 0 else None
            if tensor is not None:
                before[layer_index] = capture_tensor_cpu_f32(tensor)

        return pre_hook

    def make_post_hook(layer_index: int) -> Callable:
        def post_hook(_module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            if layer_index in after:
                return
            tensor = first_tensor(output)
            if tensor is not None:
                after[layer_index] = capture_tensor_cpu_f32(tensor)

        return post_hook

    handles = []
    for layer_idx in selected:
        handles.append(layers[layer_idx].register_forward_pre_hook(make_pre_hook(layer_idx)))
        handles.append(layers[layer_idx].register_forward_hook(make_post_hook(layer_idx)))

    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    sequence_len = input_ids.shape[1]
    attention_mask = torch.ones((1, sequence_len), dtype=torch.long, device=device)
    position_ids = torch.arange(sequence_len, dtype=torch.long, device=device).unsqueeze(0)

    with torch.inference_mode():
        try:
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        finally:
            for handle in handles:
                handle.remove()
    missing_before = [idx for idx in selected if idx not in before]
    missing_after = [idx for idx in selected if idx not in after]
    if missing_before or missing_after:
        raise SystemExit(
            "failed to capture hidden states for layer(s): "
            f"before={missing_before} after={missing_after}"
        )

    first = before[selected[0]]
    hidden_size = int(first.shape[-1])
    metadata_layers: list[dict] = []
    for layer_idx in selected:
        before_file = f"layer_{layer_idx}_before.f32"
        after_file = f"layer_{layer_idx}_after.f32"
        before_shape = write_f32_file(args.output / before_file, before[layer_idx])
        after_shape = write_f32_file(args.output / after_file, after[layer_idx])
        if before_shape != after_shape:
            raise RuntimeError(f"layer {layer_idx} shape mismatch: before={before_shape}, after={after_shape}")
        metadata_layers.append(
            {
                "layer_index": layer_idx,
                "before_file": before_file,
                "after_file": after_file,
                "before_shape": before_shape,
                "after_shape": after_shape,
                "dtype": "float32",
            }
        )

    model_type = getattr(model.config, "model_type", None)
    if model.config is not None and getattr(model.config, "torch_dtype", None) is not None:
        model_dtype = str(model.config.torch_dtype)
    else:
        model_dtype = str(torch.float32 if args.dtype == "auto" else parse_dtype(args.dtype))

    metadata = {
        "format": FIXTURE_FORMAT,
        "format_version": FIXTURE_FORMAT_VERSION,
        "model_dir": str(args.model_dir),
        "model_type": model_type,
        "dtype": model_dtype,
        "token_ids": token_ids,
        "position_ids": list(range(sequence_len)),
        "sequence_len": sequence_len,
        "hidden_size": hidden_size,
        "layers": metadata_layers,
        "fixture_kind": "prefix" if is_prefix_mode else "layers",
        "export_command": " ".join(sys.argv),
        "torch_version": torch.__version__,
    }
    if is_prefix_mode:
        metadata["layer_start"] = layer_start
        metadata["layer_end_exclusive"] = layer_end_exclusive

    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
