#!/usr/bin/env python3
"""Trace package q/k/v projection error through Qwen full self-attention."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from aq_scale_formats import scale_values


SCHEMA_VERSION = "qwen-self-attention-propagation-v0.1"


def load_trace_module() -> Any:
    path = Path(__file__).with_name("export-qwen-layer-module-trace.py")
    spec = importlib.util.spec_from_file_location("qwen_layer_module_trace", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument(
        "--input-override-dir",
        type=Path,
        help="Optional directory containing layer-XXXX-input.f32 files to use instead of fixture before tensors.",
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--hidden-index",
        action="append",
        type=int,
        required=True,
        help="Hidden row to summarize. Can be passed multiple times.",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--token-index", action="append", type=int, help="Token index for per-feature stage tracing.")
    parser.add_argument("--feature-index", action="append", type=int, help="Feature index for per-feature stage tracing.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def dequantize_package_matrix(package_dir: Path, tensor: dict[str, Any]) -> np.ndarray:
    shape = tensor.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"{tensor.get('name')} shape is not 2D: {shape}")
    rows = int(shape[0])
    cols = int(shape[1])
    elements = rows * cols

    packed = np.fromfile(package_dir / str(tensor["index_file"]), dtype=np.uint8)
    indices = np.empty(packed.size * 2, dtype=np.uint8)
    indices[0::2] = packed & 0x0F
    indices[1::2] = packed >> 4
    indices = indices[:elements]
    if indices.size != elements:
        raise ValueError(f"{tensor.get('name')} index payload has {indices.size} elements, expected {elements}")

    codebook = np.fromfile(package_dir / str(tensor["codebook_file"]), dtype="<f4")
    if codebook.size != 16:
        raise ValueError(f"{tensor.get('codebook_file')} has {codebook.size} entries, expected 16")

    scale_indices = np.fromfile(package_dir / str(tensor["scale_file"]), dtype=np.uint8)
    group_size = int(tensor["group_size"])
    expected_groups = (elements + group_size - 1) // group_size
    if scale_indices.size != expected_groups:
        raise ValueError(
            f"{tensor.get('name')} scale payload has {scale_indices.size} groups, expected {expected_groups}"
        )
    scales = scale_values(str(tensor["scale_format"])).numpy().astype(np.float32)
    combined_scales = scales[scale_indices] * np.float32(tensor["tensor_scale"])
    per_element_scales = np.repeat(combined_scales, group_size)[:elements]
    values = codebook[indices].astype(np.float32, copy=False) * per_element_scales
    return values.reshape(rows, cols)


def package_projection(
    package_dir: Path,
    manifest: dict[str, dict[str, Any]],
    normed: torch.Tensor,
    tensor_name: str,
) -> torch.Tensor:
    package_weight = dequantize_package_matrix(package_dir, manifest[tensor_name])
    weight = torch.from_numpy(package_weight)
    output = torch.matmul(
        normed.float().reshape(-1, normed.shape[-1]),
        weight.t(),
    ).reshape(normed.shape[0], normed.shape[1], -1)
    del weight, package_weight
    gc.collect()
    return output


def tensor_stats(values: np.ndarray) -> dict[str, float | int]:
    values = values.astype(np.float64, copy=False)
    abs_values = np.abs(values)
    return {
        "count": int(values.size),
        "mse": float(np.mean(values * values)),
        "mean_abs": float(np.mean(abs_values)),
        "max_abs": float(np.max(abs_values)),
    }


def self_attention_o_input(
    layer: torch.nn.Module,
    rotary: torch.nn.Module,
    hidden_size: int,
    q_projection: torch.Tensor,
    k_projection: torch.Tensor,
    v_projection: torch.Tensor,
    dtype: torch.dtype,
    return_stages: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35

    batch_size, sequence_len, _features = q_projection.shape
    head_dim = int(layer.self_attn.head_dim)
    hidden_shape = (batch_size, sequence_len, -1, head_dim)

    q_grouped = q_projection.view(batch_size, sequence_len, -1, head_dim * 2)
    query, gate = torch.chunk(q_grouped, 2, dim=-1)
    gate = gate.reshape(batch_size, sequence_len, -1)

    query_projection = query.reshape(batch_size, sequence_len, -1)
    gate_projection = gate
    query = layer.self_attn.q_norm(query.reshape(hidden_shape).to(dtype)).transpose(1, 2)
    key = layer.self_attn.k_norm(k_projection.view(hidden_shape).to(dtype)).transpose(1, 2)
    value = v_projection.view(hidden_shape).to(dtype).transpose(1, 2)
    query_normed = query.transpose(1, 2).reshape(batch_size, sequence_len, -1)
    key_normed = key.transpose(1, 2).reshape(batch_size, sequence_len, -1)
    value_projected = value.transpose(1, 2).reshape(batch_size, sequence_len, -1)

    position_ids = torch.arange(sequence_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    rotary_position_ids = position_ids[None, ...].expand(3, batch_size, -1)
    rotary_input = torch.empty((batch_size, sequence_len, hidden_size), dtype=dtype)
    cos, sin = rotary(rotary_input, rotary_position_ids)
    query, key = qwen35.apply_rotary_pos_emb(query, key, cos.to(query.dtype), sin.to(key.dtype))
    query_rope = query.transpose(1, 2).reshape(batch_size, sequence_len, -1)
    key_rope = key.transpose(1, 2).reshape(batch_size, sequence_len, -1)

    attn_output, _ = sdpa_attention_forward(
        layer.self_attn,
        query,
        key,
        value,
        attention_mask=None,
        dropout=0.0,
        scaling=layer.self_attn.scaling,
    )
    raw_attention = attn_output.reshape(batch_size, sequence_len, -1).contiguous()
    gate_sigmoid = torch.sigmoid(gate.to(raw_attention.dtype))
    o_input = raw_attention * gate_sigmoid
    if return_stages:
        return {
            "query_projection": query_projection.float(),
            "gate_projection": gate_projection.float(),
            "key_projection": k_projection.float(),
            "value_projection": v_projection.float(),
            "query_normed": query_normed.float(),
            "key_normed": key_normed.float(),
            "query_rope": query_rope.float(),
            "key_rope": key_rope.float(),
            "raw_attention": raw_attention.float(),
            "gate_sigmoid": gate_sigmoid.float(),
            "o_input": o_input.float(),
        }
    return o_input.float()


def capture_layer_o_input(
    trace: Any,
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.Tensor,
) -> np.ndarray:
    captured: dict[str, np.ndarray] = {}

    def pre_hook(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        captured["o_input"] = trace.to_numpy_f32(trace.capture_first_tensor(inputs))

    handle = layer.self_attn.o_proj.register_forward_pre_hook(pre_hook)
    with torch.inference_mode():
        try:
            _ = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
            )
        finally:
            handle.remove()
    return captured["o_input"]


def summarize_hidden_rows(
    trace: Any,
    package_dir: Path,
    manifest: dict[str, dict[str, Any]],
    state: dict[str, torch.Tensor],
    layer_index: int,
    hidden_indices: list[int],
    source_o_input: np.ndarray,
    package_o_input: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    o_tensor_name = f"model.language_model.layers.{layer_index}.self_attn.o_proj.weight"
    for hidden_index in hidden_indices:
        source_row = state["self_attn.o_proj.weight"][hidden_index].float().numpy()
        package_row = trace.dequantize_package_row(package_dir, manifest[o_tensor_name], hidden_index)
        per_token = []
        for token_index in range(source_o_input.shape[1]):
            source_source = float(np.dot(source_o_input[0, token_index], source_row))
            package_input_source_row = float(np.dot(package_o_input[0, token_index], source_row))
            package_package = float(np.dot(package_o_input[0, token_index], package_row))
            per_token.append(
                {
                    "token_index": token_index,
                    "source_input_source_o_row": source_source,
                    "package_input_source_o_row": package_input_source_row,
                    "package_input_package_o_row": package_package,
                    "input_error_using_source_o_row": package_input_source_row - source_source,
                    "total_error_with_package_o_row": package_package - source_source,
                }
            )
        worst_input = max(per_token, key=lambda item: abs(float(item["input_error_using_source_o_row"])))
        worst_total = max(per_token, key=lambda item: abs(float(item["total_error_with_package_o_row"])))
        rows.append(
            {
                "hidden_index": hidden_index,
                "source_o_row_l2_norm": float(np.linalg.norm(source_row.astype(np.float64))),
                "package_o_row_l2_norm": float(np.linalg.norm(package_row.astype(np.float64))),
                "worst_input_error_using_source_o_row": worst_input,
                "worst_total_error_with_package_o_row": worst_total,
                "per_token": per_token,
            }
        )
    return rows


def summarize_feature_traces(
    source_stages: dict[str, torch.Tensor],
    package_stages: dict[str, torch.Tensor],
    token_indices: list[int],
    feature_indices: list[int],
) -> list[dict[str, Any]]:
    traces = []
    for token_index in token_indices:
        for feature_index in feature_indices:
            stage_rows = []
            for name in source_stages.keys():
                source = source_stages[name]
                package = package_stages.get(name)
                if package is None:
                    continue
                if source.ndim != 3 or package.ndim != 3:
                    continue
                if token_index < 0 or token_index >= source.shape[1] or token_index >= package.shape[1]:
                    continue
                if feature_index < 0 or feature_index >= source.shape[2] or feature_index >= package.shape[2]:
                    continue
                source_value = float(source[0, token_index, feature_index].item())
                package_value = float(package[0, token_index, feature_index].item())
                stage_rows.append(
                    {
                        "stage": name,
                        "token_index": token_index,
                        "feature_index": feature_index,
                        "source_value": source_value,
                        "package_value": package_value,
                        "diff": package_value - source_value,
                        "abs_diff": abs(package_value - source_value),
                    }
                )
            traces.append(
                {
                    "token_index": token_index,
                    "feature_index": feature_index,
                    "stages": stage_rows,
                }
            )
    return traces


def attention_feature_breakdown(
    stages: dict[str, torch.Tensor],
    token_index: int,
    feature_index: int,
    head_dim: int,
    softmax_scale: float,
) -> dict[str, Any] | None:
    query_rope = stages["query_rope"]
    key_rope = stages["key_rope"]
    value_projection = stages["value_projection"]
    raw_attention = stages["raw_attention"]
    gate_projection = stages["gate_projection"]
    gate_sigmoid = stages["gate_sigmoid"]
    o_input = stages["o_input"]
    if (
        query_rope.ndim != 3
        or key_rope.ndim != 3
        or value_projection.ndim != 3
        or raw_attention.ndim != 3
        or head_dim <= 0
    ):
        return None
    sequence_len = int(query_rope.shape[1])
    if token_index < 0 or token_index >= sequence_len:
        return None
    q_width = int(query_rope.shape[2])
    k_width = int(key_rope.shape[2])
    if q_width % head_dim != 0 or k_width % head_dim != 0:
        return None
    q_heads = q_width // head_dim
    kv_heads = k_width // head_dim
    if kv_heads == 0 or q_heads == 0 or q_heads % kv_heads != 0:
        return None
    value_width = int(value_projection.shape[2])
    if value_width % kv_heads != 0:
        return None
    value_dim = value_width // kv_heads
    attention_width = q_heads * value_dim
    if feature_index < 0 or feature_index >= attention_width:
        return None
    q_head = feature_index // value_dim
    value_offset = feature_index % value_dim
    kv_head = q_head // (q_heads // kv_heads)

    q_start = q_head * head_dim
    q_end = q_start + head_dim
    k_start = kv_head * head_dim
    k_end = k_start + head_dim
    v_feature = kv_head * value_dim + value_offset

    q_vector = query_rope[0, token_index, q_start:q_end].float()
    scores = []
    dots = []
    for source_token in range(token_index + 1):
        k_vector = key_rope[0, source_token, k_start:k_end].float()
        dot = float(torch.dot(q_vector, k_vector).item())
        dots.append(dot)
        scores.append(dot * softmax_scale)
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    weights = torch.softmax(score_tensor, dim=0)

    source_tokens = []
    computed_attention_output = 0.0
    for source_token, (dot, score, weight) in enumerate(
        zip(dots, scores, weights.tolist(), strict=True)
    ):
        v_value = float(value_projection[0, source_token, v_feature].item())
        contribution = float(weight * v_value)
        computed_attention_output += contribution
        source_tokens.append(
            {
                "source_token_index": source_token,
                "dot": dot,
                "score": score,
                "softmax_weight": float(weight),
                "v_value": v_value,
                "weighted_v_contribution": contribution,
            }
        )

    raw_value = float(raw_attention[0, token_index, feature_index].item())
    gate_value = float(gate_projection[0, token_index, feature_index].item())
    gate_sigmoid_value = float(gate_sigmoid[0, token_index, feature_index].item())
    o_input_value = float(o_input[0, token_index, feature_index].item())
    return {
        "token_index": token_index,
        "feature_index": feature_index,
        "q_head": q_head,
        "kv_head": kv_head,
        "q_per_kv": q_heads // kv_heads,
        "value_offset": value_offset,
        "head_dim": head_dim,
        "value_dim": value_dim,
        "softmax_scale": softmax_scale,
        "computed_attention_output": computed_attention_output,
        "raw_attention": raw_value,
        "computed_minus_raw_attention": computed_attention_output - raw_value,
        "gate_projection": gate_value,
        "gate_sigmoid": gate_sigmoid_value,
        "o_input": o_input_value,
        "computed_o_input": computed_attention_output * gate_sigmoid_value,
        "computed_minus_o_input": computed_attention_output * gate_sigmoid_value - o_input_value,
        "source_tokens": source_tokens,
    }


def summarize_attention_breakdowns(
    source_stages: dict[str, torch.Tensor],
    package_stages: dict[str, torch.Tensor],
    token_indices: list[int],
    feature_indices: list[int],
    head_dim: int,
    softmax_scale: float,
) -> list[dict[str, Any]]:
    breakdowns = []
    for token_index in token_indices:
        for feature_index in feature_indices:
            source = attention_feature_breakdown(
                source_stages,
                token_index,
                feature_index,
                head_dim,
                softmax_scale,
            )
            package = attention_feature_breakdown(
                package_stages,
                token_index,
                feature_index,
                head_dim,
                softmax_scale,
            )
            breakdowns.append(
                {
                    "token_index": token_index,
                    "feature_index": feature_index,
                    "source": source,
                    "package": package,
                }
            )
    return breakdowns


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 9) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}g}"
    return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "| hidden | worst_input_token | input_error_source_o_row | worst_total_token | total_error_package_o_row |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["hidden_rows"]:
        worst_input = row["worst_input_error_using_source_o_row"]
        worst_total = row["worst_total_error_with_package_o_row"]
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                row["hidden_index"],
                worst_input["token_index"],
                fmt(worst_input["input_error_using_source_o_row"]),
                worst_total["token_index"],
                fmt(worst_total["total_error_with_package_o_row"]),
            )
        )
    lines.extend(
        [
            "",
            "| stage | mse | mean_abs | max_abs |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, stats in payload["stage_diff"].items():
        lines.append(
            "| {} | {} | {} | {} |".format(
                name,
                fmt(stats["mse"]),
                fmt(stats["mean_abs"]),
                fmt(stats["max_abs"]),
            )
        )
    feature_traces = payload.get("feature_traces")
    if feature_traces:
        lines.extend(
            [
                "",
                "## Feature Traces",
                "",
                "| token | feature | stage | source | package | diff |",
                "| ---: | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for trace in feature_traces:
            for stage in trace["stages"]:
                lines.append(
                    "| {} | {} | {} | {} | {} | {} |".format(
                        stage["token_index"],
                        stage["feature_index"],
                        stage["stage"],
                        fmt(stage["source_value"]),
                        fmt(stage["package_value"]),
                        fmt(stage["diff"]),
                    )
                )
    attention_breakdowns = payload.get("attention_breakdowns")
    if attention_breakdowns:
        for breakdown in attention_breakdowns:
            lines.extend(
                [
                    "",
                    "## Attention Breakdown token={} feature={}".format(
                        breakdown["token_index"],
                        breakdown["feature_index"],
                    ),
                    "",
                    "| replay | q_head | kv_head | value_offset | raw_attention | computed | gate_sigmoid | o_input |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for replay_name in ("source", "package"):
                replay = breakdown.get(replay_name)
                if replay is None:
                    continue
                lines.append(
                    "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        replay_name,
                        replay["q_head"],
                        replay["kv_head"],
                        replay["value_offset"],
                        fmt(replay["raw_attention"]),
                        fmt(replay["computed_attention_output"]),
                        fmt(replay["gate_sigmoid"]),
                        fmt(replay["o_input"]),
                    )
                )
            lines.extend(
                [
                    "",
                    "| replay | source_token | score | weight | v_value | contribution |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for replay_name in ("source", "package"):
                replay = breakdown.get(replay_name)
                if replay is None:
                    continue
                for row in replay["source_tokens"]:
                    lines.append(
                        "| {} | {} | {} | {} | {} | {} |".format(
                            replay_name,
                            row["source_token_index"],
                            fmt(row["score"]),
                            fmt(row["softmax_weight"]),
                            fmt(row["v_value"]),
                            fmt(row["weighted_v_contribution"]),
                        )
                    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    dtype = torch_dtype(args.dtype)
    trace = load_trace_module()

    model_dir = args.model_dir.expanduser().resolve()
    fixture = args.fixture.expanduser().resolve()
    package_dir = args.package_dir.expanduser().resolve()
    input_override_dir = args.input_override_dir.expanduser().resolve() if args.input_override_dir else None

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        str(model_dir),
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
    layers = trace.resolve_layers(model)
    layer = layers[args.layer]
    if getattr(layer, "layer_type", None) != "full_attention":
        raise ValueError(f"layer {args.layer} is {getattr(layer, 'layer_type', None)}; expected full_attention")
    rotary_template = trace.resolve_rotary_embedding(model)
    if rotary_template is None:
        raise ValueError("model does not expose a rotary embedding")

    metadata = trace.read_fixture_metadata(fixture)
    entry = trace.fixture_layer_entry(metadata, args.layer)
    before, input_source = trace.read_layer_input_tensor(fixture, entry, args.layer, input_override_dir)
    if before.ndim != 3:
        raise ValueError(f"expected [batch,seq,hidden] fixture shape, got {before.shape}")

    weight_files = trace.build_weight_file_map(model_dir)
    state = trace.load_layer_state(model_dir, weight_files, args.layer)
    layer.to_empty(device="cpu")
    layer.load_state_dict(state, strict=True)
    layer.to(device="cpu", dtype=dtype)
    layer.eval()

    rotary = type(rotary_template)(rotary_template.config, device=torch.device("cpu")).to(dtype=dtype)
    manifest = trace.read_package_manifest(package_dir)
    if manifest is None:
        raise ValueError("package manifest is missing")

    hidden_states = torch.from_numpy(before).to(dtype=dtype)
    sequence_len = hidden_states.shape[1]
    position_ids = torch.arange(sequence_len, dtype=torch.long).unsqueeze(0).expand(hidden_states.shape[0], -1)
    rotary_position_ids = position_ids[None, ...].expand(3, hidden_states.shape[0], -1)
    position_embeddings = rotary(hidden_states, rotary_position_ids)

    with torch.inference_mode():
        attention_input_normed = layer.input_layernorm(hidden_states)
        source_q = layer.self_attn.q_proj(attention_input_normed)
        source_k = layer.self_attn.k_proj(attention_input_normed)
        source_v = layer.self_attn.v_proj(attention_input_normed)
        source_stages = self_attention_o_input(
            layer,
            rotary,
            before.shape[-1],
            source_q,
            source_k,
            source_v,
            dtype,
            return_stages=True,
        )
        source_o_input = source_stages["o_input"]
        captured_source_o_input = capture_layer_o_input(
            trace,
            layer,
            hidden_states,
            position_embeddings,
            position_ids,
        )
        package_q = package_projection(
            package_dir,
            manifest,
            attention_input_normed,
            f"model.language_model.layers.{args.layer}.self_attn.q_proj.weight",
        )
        package_k = package_projection(
            package_dir,
            manifest,
            attention_input_normed,
            f"model.language_model.layers.{args.layer}.self_attn.k_proj.weight",
        )
        package_v = package_projection(
            package_dir,
            manifest,
            attention_input_normed,
            f"model.language_model.layers.{args.layer}.self_attn.v_proj.weight",
        )
        package_stages = self_attention_o_input(
            layer,
            rotary,
            before.shape[-1],
            package_q,
            package_k,
            package_v,
            torch.float32,
            return_stages=True,
        )
        package_o_input = package_stages["o_input"]

    source_o_input_np = source_o_input.numpy()
    package_o_input_np = package_o_input.numpy()
    captured_diff = source_o_input_np - captured_source_o_input
    stage_diff = {
        "source_o_input_replay_vs_layer_hook": tensor_stats(captured_diff),
        "package_q_projection_vs_source": tensor_stats(package_q.numpy() - source_q.float().numpy()),
        "package_k_projection_vs_source": tensor_stats(package_k.numpy() - source_k.float().numpy()),
        "package_v_projection_vs_source": tensor_stats(package_v.numpy() - source_v.float().numpy()),
        "package_o_input_vs_source": tensor_stats(package_o_input_np - source_o_input_np),
    }

    hidden_indices = list(dict.fromkeys(int(index) for index in args.hidden_index))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "command": "analyze-qwen-self-attention-propagation",
        "model_dir": str(model_dir),
        "fixture": str(fixture),
        "package_dir": str(package_dir),
        "input_source": input_source,
        "input_override_dir": None if input_override_dir is None else str(input_override_dir),
        "layer_index": args.layer,
        "hidden_indices": hidden_indices,
        "dtype": args.dtype,
        "stage_diff": stage_diff,
        "hidden_rows": summarize_hidden_rows(
            trace,
            package_dir,
            manifest,
            state,
            args.layer,
            hidden_indices,
            source_o_input_np,
            package_o_input_np,
        ),
    }
    if args.token_index and args.feature_index:
        token_indices = list(dict.fromkeys(int(index) for index in args.token_index))
        feature_indices = list(dict.fromkeys(int(index) for index in args.feature_index))
        payload["feature_traces"] = summarize_feature_traces(
            source_stages,
            package_stages,
            token_indices,
            feature_indices,
        )
        payload["attention_breakdowns"] = summarize_attention_breakdowns(
            source_stages,
            package_stages,
            token_indices,
            feature_indices,
            int(layer.self_attn.head_dim),
            float(layer.self_attn.scaling),
        )
    write_json(args.summary_json, payload)
    if args.markdown is not None:
        write_markdown(args.markdown, payload)
    print(f"qwen-self-attention-propagation hidden_rows={len(hidden_indices)} output={args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
