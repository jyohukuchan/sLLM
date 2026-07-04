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
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


SCHEMA_VERSION = "qwen-layer-module-trace-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
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


def build_weight_file_map(model_dir: Path) -> dict[str, Path]:
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} does not contain a weight_map")
    return {name: model_dir / str(filename) for name, filename in weight_map.items()}


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


def run_layer_trace(
    model_dir: Path,
    fixture: Path,
    metadata: dict[str, Any],
    weight_files: dict[str, Path],
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
    before_shape = [int(value) for value in entry["before_shape"]]
    after_shape = [int(value) for value in entry["after_shape"]]
    before = read_f32_tensor(fixture / str(entry["before_file"]), before_shape)
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

    handles = [
        layer.linear_attn.register_forward_hook(hook("attention_output")),
        layer.post_attention_layernorm.register_forward_hook(hook("post_normed")),
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
    attention_output = captured["attention_output"]
    post_normed = captured["post_normed"]
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
    return {
        "schema_version": SCHEMA_VERSION,
        "command": "export-qwen-layer-module-trace",
        "model_dir": str(model_dir),
        "fixture": str(fixture),
        "layer_index": layer_index,
        "layer_type": layer_type,
        "hidden_index": hidden_index,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "shape": list(actual_after.shape),
        "fixture_match": compare_arrays(actual_after, expected_after),
        "max_expected_delta_trace": per_token[token_index],
        "per_token_hidden_trace": per_token,
    }


def fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return str(value)


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| layer | hidden | token | fixture_max_abs | expected_delta | attn | mlp | actual_delta | output_diff |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        trace = row["max_expected_delta_trace"]
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
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    fixture = args.fixture.expanduser().resolve()
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
    metadata = read_fixture_metadata(fixture)
    weight_files = build_weight_file_map(model_dir)

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
                model_layers[layer_index],
                layer_index,
                args.hidden_index,
                device,
                dtype,
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
