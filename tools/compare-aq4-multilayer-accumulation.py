#!/usr/bin/env python3
"""CPU-only chained AQ4/BF16 layer comparison for the H8 diagnostic.

The Rust side uses the production standalone AQ4 decoder for one layer at a
time and streams each f32 layer output.  This tool calculates the matching
BF16 source layer, compares that frame immediately, and retains only the
current sequence plus aggregate metrics/fixed coordinates.  It intentionally
does not invoke a GPU, a resident service, or a full-model runtime.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO

import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[1]
HYBRID_PATH = ROOT / "tools" / "compare-aq4-layer0-hybrid.py"
SPEC = importlib.util.spec_from_file_location("compare_aq4_layer0_hybrid", HYBRID_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"failed loading shared hybrid comparator: {HYBRID_PATH}")
HYBRID = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HYBRID
SPEC.loader.exec_module(HYBRID)


SCHEMA = "ullm.aq4_multilayer_accumulation.source_compare.v3"
AQ4_SCHEMA = "ullm.aq4_multilayer_accumulation.aq4_cpu.v3"
HIDDEN = 4096
INTERMEDIATE = 12288
QKV = 8192
KEY_HEADS = 16
VALUE_HEADS = 32
KEY_DIM = 128
VALUE_DIM = 128
CONV_KERNEL = 4
SELF_Q_HEADS = 16
SELF_KV_HEADS = 4
SELF_HEAD_DIM = 256
SELF_VALUE_DIM = 256
SELF_Q_ROWS = SELF_Q_HEADS * SELF_HEAD_DIM * 2
SELF_KV_ROWS = SELF_KV_HEADS * SELF_HEAD_DIM
SELF_VALUE_ROWS = SELF_KV_HEADS * SELF_VALUE_DIM
ROTARY_DIM = 64
ROPE_BASE = 10_000_000.0
INPUT_EPS = 1e-6
ATTN_EPS = 1e-6
SOURCE_POST_EPS = 1e-6
MAX_LINE_BYTES = 64 * 1024
FINAL_RELATIVE_L2 = 0.615
FINAL_NORM_TENSOR = "model.language_model.norm.weight"
LM_HEAD_TENSOR = "lm_head.weight"
LM_HEAD_SAMPLE_ROWS = HYBRID.DIAGNOSTIC_LOGIT_ROWS


def parse_layer_range(raw: str) -> tuple[int, int]:
    try:
        start_raw, end_raw = raw.split(":", 1)
        start, end = int(start_raw), int(end_raw)
    except ValueError as error:
        raise ValueError("--chain-layer-range must be START:END") from error
    if start < 0 or end <= start:
        raise ValueError("--chain-layer-range must contain at least two ascending nonnegative layers")
    return start, end


def source_config(source_model: Path) -> dict[str, Any]:
    raw = json.loads((source_model / "config.json").read_text(encoding="utf-8"))
    config = raw.get("text_config")
    if not isinstance(config, dict):
        raise ValueError("source config has no text_config")
    required = {
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": 32,
        "num_attention_heads": SELF_Q_HEADS,
        "num_key_value_heads": SELF_KV_HEADS,
        "head_dim": SELF_HEAD_DIM,
        "linear_num_key_heads": KEY_HEADS,
        "linear_num_value_heads": VALUE_HEADS,
        "linear_key_head_dim": KEY_DIM,
        "linear_value_head_dim": VALUE_DIM,
        "linear_conv_kernel_dim": CONV_KERNEL,
        "rms_norm_eps": SOURCE_POST_EPS,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"source config {key} differs: {config.get(key)!r}")
    rope = config.get("rope_parameters")
    if not isinstance(rope, dict) or rope.get("rope_theta") != ROPE_BASE or rope.get("partial_rotary_factor") != 0.25:
        raise ValueError("source config RoPE parameters differ")
    types = config.get("layer_types")
    if not isinstance(types, list) or len(types) != 32 or any(item not in ("linear_attention", "full_attention") for item in types):
        raise ValueError("source config layer_types differs")
    expected_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8
    if types != expected_types:
        raise ValueError("source config layer_types does not match Qwen3.5-9B hybrid topology")
    return {
        "config_path": str(source_model / "config.json"),
        "config_sha256": HYBRID.sha256_file(source_model / "config.json"),
        "layer_types": types,
        "self_attention_indices": [index for index, kind in enumerate(types) if kind == "full_attention"],
        "rotary_dim": ROTARY_DIM,
        "rope_base": ROPE_BASE,
    }


def tensor(loader: HYBRID.SourceLoader, name: str, shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    return loader.tensor_by_name(name, shape, dtype)


def layer_name(layer_index: int, suffix: str) -> str:
    return f"model.language_model.layers.{layer_index}.{suffix}"


def source_linear_layer(loader: HYBRID.SourceLoader, layer_index: int, residual: torch.Tensor) -> torch.Tensor:
    if residual.ndim != 2 or residual.shape[1] != HIDDEN or residual.dtype != torch.bfloat16:
        raise ValueError(f"invalid BF16 source residual for linear layer {layer_index}")
    sequence = int(residual.shape[0])
    input_norm_name = layer_name(layer_index, "input_layernorm.weight")
    input_norm_weight = tensor(loader, input_norm_name, [HIDDEN], torch.bfloat16)
    input_normed = HYBRID.source_rmsnorm(residual, input_norm_weight, INPUT_EPS)
    del input_norm_weight
    qkv = functional.linear(
        input_normed,
        tensor(loader, layer_name(layer_index, "linear_attn.in_proj_qkv.weight"), [QKV, HIDDEN], torch.bfloat16),
    )
    z = functional.linear(
        input_normed,
        tensor(loader, layer_name(layer_index, "linear_attn.in_proj_z.weight"), [HIDDEN, HIDDEN], torch.bfloat16),
    )
    a = functional.linear(
        input_normed,
        tensor(loader, layer_name(layer_index, "linear_attn.in_proj_a.weight"), [VALUE_HEADS, HIDDEN], torch.bfloat16),
    )
    b = functional.linear(
        input_normed,
        tensor(loader, layer_name(layer_index, "linear_attn.in_proj_b.weight"), [VALUE_HEADS, HIDDEN], torch.bfloat16),
    )
    conv_weight = tensor(
        loader,
        layer_name(layer_index, "linear_attn.conv1d.weight"),
        [QKV, 1, CONV_KERNEL],
        torch.bfloat16,
    )
    conv_pre = functional.conv1d(
        qkv.transpose(0, 1).unsqueeze(0), conv_weight, padding=CONV_KERNEL - 1, groups=QKV
    )[:, :, :sequence].squeeze(0).transpose(0, 1).contiguous()
    del conv_weight
    conv_silu = functional.silu(conv_pre)
    q_raw = conv_silu[:, : KEY_HEADS * KEY_DIM].reshape(sequence, KEY_HEADS, KEY_DIM)
    k_raw = conv_silu[:, KEY_HEADS * KEY_DIM : KEY_HEADS * KEY_DIM * 2].reshape(sequence, KEY_HEADS, KEY_DIM)
    v = conv_silu[:, KEY_HEADS * KEY_DIM * 2 :].reshape(sequence, VALUE_HEADS, VALUE_DIM)
    q_l2 = HYBRID.source_qk_l2norm(q_raw)
    k_l2 = HYBRID.source_qk_l2norm(k_raw)
    a_log = tensor(loader, layer_name(layer_index, "linear_attn.A_log"), [VALUE_HEADS], torch.float32)
    dt_bias = tensor(loader, layer_name(layer_index, "linear_attn.dt_bias"), [VALUE_HEADS], torch.bfloat16)
    gate = -a_log.float().exp().unsqueeze(0) * functional.softplus(a.float() + dt_bias.float().unsqueeze(0))
    beta = torch.sigmoid(b)
    del a_log, dt_bias, qkv, conv_pre, conv_silu, a, b
    attention_weight = tensor(loader, layer_name(layer_index, "linear_attn.norm.weight"), [VALUE_DIM], torch.float32)
    out_weight = tensor(loader, layer_name(layer_index, "linear_attn.out_proj.weight"), [HIDDEN, HIDDEN], torch.bfloat16)
    post_weight = tensor(loader, layer_name(layer_index, "post_attention_layernorm.weight"), [HIDDEN], torch.bfloat16)
    recurrent_state = torch.zeros((VALUE_HEADS, KEY_DIM, VALUE_DIM), dtype=torch.float32)
    attention_residuals: list[torch.Tensor] = []
    post_normed: list[torch.Tensor] = []
    for timestep in range(sequence):
        q_repeated = q_l2[timestep].repeat_interleave(2, dim=0).float() * HYBRID.Q_SCALE
        k_repeated = k_l2[timestep].repeat_interleave(2, dim=0).float()
        v_repeated = v[timestep].float()
        recurrent_state.mul_(gate[timestep].exp().reshape(VALUE_HEADS, 1, 1))
        memory = (recurrent_state * k_repeated.unsqueeze(-1)).sum(dim=-2)
        delta = (v_repeated - memory) * beta[timestep].float().unsqueeze(-1)
        recurrent_state.add_(k_repeated.unsqueeze(-1) * delta.unsqueeze(-2))
        recurrent = (recurrent_state * q_repeated.unsqueeze(-1)).sum(dim=-2).to(dtype=torch.bfloat16)
        _, _, composed = HYBRID.source_attention_gated_norm(
            recurrent, z[timestep].reshape(VALUE_HEADS, VALUE_DIM), attention_weight
        )
        attention_projection = functional.linear(composed.reshape(1, HIDDEN), out_weight).reshape(-1)
        attention_residual = residual[timestep] + attention_projection
        attention_residuals.append(attention_residual)
        post_normed.append(HYBRID.source_rmsnorm(attention_residual.reshape(1, HIDDEN), post_weight, SOURCE_POST_EPS).reshape(-1))
    del attention_weight, out_weight, post_weight, q_l2, k_l2, v, z, gate, beta, recurrent_state
    post_batch = torch.stack(post_normed, dim=0)
    gate_weight = tensor(loader, layer_name(layer_index, "mlp.gate_proj.weight"), [INTERMEDIATE, HIDDEN], torch.bfloat16)
    mlp_gate = functional.linear(post_batch, gate_weight)
    del gate_weight
    up_weight = tensor(loader, layer_name(layer_index, "mlp.up_proj.weight"), [INTERMEDIATE, HIDDEN], torch.bfloat16)
    mlp_up = functional.linear(post_batch, up_weight)
    del up_weight
    down_weight = tensor(loader, layer_name(layer_index, "mlp.down_proj.weight"), [HIDDEN, INTERMEDIATE], torch.bfloat16)
    outputs = []
    for timestep in range(sequence):
        activation = functional.silu(mlp_gate[timestep]) * mlp_up[timestep]
        mlp_output = functional.linear(activation.reshape(1, INTERMEDIATE), down_weight).reshape(-1)
        outputs.append(attention_residuals[timestep] + mlp_output)
    del down_weight, mlp_gate, mlp_up, attention_residuals, post_normed, post_batch
    return torch.stack(outputs, dim=0).contiguous()


def source_rope(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[2] != SELF_HEAD_DIM:
        raise ValueError("invalid source RoPE tensor geometry")
    sequence = int(hidden.shape[0])
    positions = torch.arange(sequence, dtype=torch.float32)
    inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM))
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cat((freqs, freqs), dim=-1).cos().to(dtype=hidden.dtype).unsqueeze(1)
    sin = torch.cat((freqs, freqs), dim=-1).sin().to(dtype=hidden.dtype).unsqueeze(1)
    first, second = hidden[..., : ROTARY_DIM // 2], hidden[..., ROTARY_DIM // 2 : ROTARY_DIM]
    rotated = torch.cat((-second, first), dim=-1)
    result = hidden.clone()
    result[..., :ROTARY_DIM] = hidden[..., :ROTARY_DIM] * cos + rotated * sin
    return result


def source_self_attention_layer(loader: HYBRID.SourceLoader, layer_index: int, residual: torch.Tensor) -> torch.Tensor:
    if residual.ndim != 2 or residual.shape[1] != HIDDEN or residual.dtype != torch.bfloat16:
        raise ValueError(f"invalid BF16 source residual for self-attention layer {layer_index}")
    sequence = int(residual.shape[0])
    input_norm_weight = tensor(loader, layer_name(layer_index, "input_layernorm.weight"), [HIDDEN], torch.bfloat16)
    input_normed = HYBRID.source_rmsnorm(residual, input_norm_weight, INPUT_EPS)
    del input_norm_weight
    q_projected = functional.linear(
        input_normed,
        tensor(loader, layer_name(layer_index, "self_attn.q_proj.weight"), [SELF_Q_ROWS, HIDDEN], torch.bfloat16),
    ).reshape(sequence, SELF_Q_HEADS, 2, SELF_HEAD_DIM)
    q_raw, gate = q_projected.unbind(dim=2)
    del q_projected
    k_projected = functional.linear(
        input_normed,
        tensor(loader, layer_name(layer_index, "self_attn.k_proj.weight"), [SELF_KV_ROWS, HIDDEN], torch.bfloat16),
    ).reshape(sequence, SELF_KV_HEADS, SELF_HEAD_DIM)
    v_projected = functional.linear(
        input_normed,
        tensor(loader, layer_name(layer_index, "self_attn.v_proj.weight"), [SELF_VALUE_ROWS, HIDDEN], torch.bfloat16),
    ).reshape(sequence, SELF_KV_HEADS, SELF_VALUE_DIM)
    q_norm_weight = tensor(loader, layer_name(layer_index, "self_attn.q_norm.weight"), [SELF_HEAD_DIM], torch.bfloat16)
    k_norm_weight = tensor(loader, layer_name(layer_index, "self_attn.k_norm.weight"), [SELF_HEAD_DIM], torch.bfloat16)
    q_normed = HYBRID.source_rmsnorm(q_raw, q_norm_weight, SOURCE_POST_EPS)
    k_normed = HYBRID.source_rmsnorm(k_projected, k_norm_weight, SOURCE_POST_EPS)
    del q_norm_weight, k_norm_weight, q_raw, k_projected
    q_rope = source_rope(q_normed)
    k_rope = source_rope(k_normed)
    del q_normed, k_normed
    q = q_rope.permute(1, 0, 2).unsqueeze(0)
    k = k_rope.permute(1, 0, 2).unsqueeze(0)
    v = v_projected.permute(1, 0, 2).unsqueeze(0)
    k_repeated = k.repeat_interleave(SELF_Q_HEADS // SELF_KV_HEADS, dim=1)
    v_repeated = v.repeat_interleave(SELF_Q_HEADS // SELF_KV_HEADS, dim=1)
    scores = torch.matmul(q, k_repeated.transpose(2, 3)) * (SELF_HEAD_DIM ** -0.5)
    mask = torch.triu(
        torch.full((sequence, sequence), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16), diagonal=1
    )
    probabilities = torch.softmax(scores + mask, dim=-1, dtype=torch.float32).to(dtype=torch.bfloat16)
    attention_output = torch.matmul(probabilities, v_repeated).transpose(1, 2).contiguous().reshape(sequence, HIDDEN)
    del q_rope, k_rope, v_projected, q, k, v, k_repeated, v_repeated, scores, mask, probabilities
    gated_attention = attention_output * torch.sigmoid(gate.reshape(sequence, HIDDEN))
    del attention_output, gate
    o_weight = tensor(loader, layer_name(layer_index, "self_attn.o_proj.weight"), [HIDDEN, HIDDEN], torch.bfloat16)
    attention_projection = functional.linear(gated_attention, o_weight)
    del o_weight, gated_attention
    attention_residual = residual + attention_projection
    post_weight = tensor(loader, layer_name(layer_index, "post_attention_layernorm.weight"), [HIDDEN], torch.bfloat16)
    post = HYBRID.source_rmsnorm(attention_residual, post_weight, SOURCE_POST_EPS)
    del post_weight
    gate_weight = tensor(loader, layer_name(layer_index, "mlp.gate_proj.weight"), [INTERMEDIATE, HIDDEN], torch.bfloat16)
    mlp_gate = functional.linear(post, gate_weight)
    del gate_weight
    up_weight = tensor(loader, layer_name(layer_index, "mlp.up_proj.weight"), [INTERMEDIATE, HIDDEN], torch.bfloat16)
    mlp_up = functional.linear(post, up_weight)
    del up_weight, post
    down_weight = tensor(loader, layer_name(layer_index, "mlp.down_proj.weight"), [HIDDEN, INTERMEDIATE], torch.bfloat16)
    mlp_output = functional.linear(functional.silu(mlp_gate) * mlp_up, down_weight)
    del down_weight, mlp_gate, mlp_up
    return (attention_residual + mlp_output).contiguous()


def source_final_rmsnorm(loader: HYBRID.SourceLoader, residual: torch.Tensor) -> torch.Tensor:
    """Runs Qwen3.5's additive final RMSNorm on the current BF16 sequence.

    The checkpoint's final norm is an instance of `Qwen3_5RMSNorm`, whose
    forward equation is `normalized * (1 + weight)`.  Its raw weights are not
    interchangeable with a direct-weight RMSNorm, so keep this source-side
    contract explicit instead of reusing the AQ4 package's runtime handling.
    """
    if residual.ndim != 2 or residual.shape[1] != HIDDEN or residual.dtype != torch.bfloat16:
        raise ValueError("invalid BF16 source residual for final RMSNorm")
    weight = tensor(loader, FINAL_NORM_TENSOR, [HIDDEN], torch.bfloat16)
    result = HYBRID.source_rmsnorm(residual, weight, SOURCE_POST_EPS)
    del weight
    return result.contiguous()


class ChainReader:
    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.accumulators: dict[int, HYBRID.MetricAccumulator] = {}
        self.terminal_accumulators: dict[str, HYBRID.MetricAccumulator] = {}

    def expect(self, layer_index: int, layer_kind: str, case: dict[str, Any], timestep: int, reference: torch.Tensor) -> None:
        raw = self.source.readline(MAX_LINE_BYTES + 1)
        if not raw:
            raise ValueError(f"AQ4 stream ended before layer {layer_index} output")
        if len(raw) > MAX_LINE_BYTES or not raw.endswith(b"\n"):
            raise ValueError("AQ4 chain stream header is oversized or unterminated")
        header = json.loads(raw)
        expected = {
            "kind": "chain_layer_output",
            "layer_index": layer_index,
            "layer_kind": layer_kind,
            "case_id": case["case_id"],
            "step": case["step"],
            "context_token_ids_sha256": case["context_token_ids_sha256"],
            "context_length": case["context_length"],
            "timestep": timestep,
            "dtype": "f32le",
            "shape": [HIDDEN],
            "bytes": HIDDEN * 4,
        }
        if header != expected:
            raise ValueError(f"AQ4 chain stream identity/order differs for layer {layer_index}: {header}")
        payload = self.source.read(HIDDEN * 4)
        if len(payload) != HIDDEN * 4:
            raise ValueError(f"AQ4 chain stream payload is short for layer {layer_index}")
        actual = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone()
        metric_header = dict(header)
        metric_header["stage"] = f"layer_{layer_index}_output"
        self.accumulators.setdefault(layer_index, HYBRID.MetricAccumulator()).update(metric_header, actual, reference.reshape(-1))

    def expect_terminal(
        self,
        stage: str,
        measurement_scope: str,
        coordinates: tuple[int, ...],
        case: dict[str, Any],
        timestep: int,
        reference: torch.Tensor,
    ) -> None:
        raw = self.source.readline(MAX_LINE_BYTES + 1)
        if not raw:
            raise ValueError(f"AQ4 stream ended before terminal {stage}")
        if len(raw) > MAX_LINE_BYTES or not raw.endswith(b"\n"):
            raise ValueError("AQ4 terminal stream header is oversized or unterminated")
        header = json.loads(raw)
        elements = int(reference.numel())
        expected = {
            "kind": "chain_terminal_output",
            "stage": stage,
            "measurement_scope": measurement_scope,
            "coordinates": list(coordinates),
            "case_id": case["case_id"],
            "step": case["step"],
            "context_token_ids_sha256": case["context_token_ids_sha256"],
            "context_length": case["context_length"],
            "timestep": timestep,
            "dtype": "f32le",
            "shape": [elements],
            "bytes": elements * 4,
        }
        if header != expected:
            raise ValueError(f"AQ4 terminal stream identity/order differs for {stage}: {header}")
        payload = self.source.read(elements * 4)
        if len(payload) != elements * 4:
            raise ValueError(f"AQ4 terminal stream payload is short for {stage}")
        actual = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone()
        self.terminal_accumulators.setdefault(stage, HYBRID.MetricAccumulator()).update(
            header, actual, reference.reshape(-1)
        )

    def expect_end(self) -> None:
        line = self.source.readline(MAX_LINE_BYTES + 1)
        if line != b'{"kind":"end"}\n':
            raise ValueError("AQ4 chain stream has no valid terminal frame")
        if self.source.read(1):
            raise ValueError("AQ4 chain stream has trailing bytes")


def extrapolate(layer_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(item["aggregate"]["relative_l2"]) for item in layer_metrics]
    indices = [int(item["layer_index"]) for item in layer_metrics]
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("invalid layer relative L2 values for extrapolation")
    last_index, last = indices[-1], values[-1]
    linear = last * (32.0 / (last_index + 1.0))
    positive = [value for value in values if value > 0.0]
    if len(values) >= 2 and values[0] > 0.0 and last > 0.0:
        geometric_ratio = (last / values[0]) ** (1.0 / (len(values) - 1))
        geometric = last * geometric_ratio ** (31 - last_index)
    else:
        geometric_ratio = None
        geometric = None
    increments = [right - left for left, right in zip(values, values[1:])]
    ratios = [right / left for left, right in zip(values, values[1:]) if left > 0.0]
    monotonic = all(right >= left for left, right in zip(values, values[1:]))
    # Ratios alone are misleading for an exactly linear curve with a positive
    # intercept (for example .04, .08, .12, .16).  Classify acceleration from
    # the increment trend, and reserve the geometric continuation for a
    # materially growing increment.
    if len(increments) >= 2 and monotonic and increments[-1] > increments[0] * 1.25:
        shape = "superlinear"
        chosen_model = "geometric"
    elif monotonic:
        shape = "approximately_linear_or_sublinear"
        chosen_model = "linear"
    else:
        shape = "nonmonotonic_or_layer_jump"
        chosen_model = "linear_conservative"
    complete_decoder_stack = indices == list(range(32))
    if complete_decoder_stack:
        chosen_model = "observed_full_decoder_stack"
        chosen = last
    else:
        chosen = geometric if chosen_model == "geometric" and geometric is not None else linear
    fraction = chosen / FINAL_RELATIVE_L2
    if fraction >= 0.8:
        verdict = "explains"
    elif fraction >= 0.25:
        verdict = "partially_explains"
    else:
        verdict = "does_not_explain"
    return {
        "observed_layer_indices": indices,
        "observed_relative_l2": values,
        "complete_decoder_stack": complete_decoder_stack,
        "increment_per_observed_transition": increments,
        "multiplicative_ratio_per_observed_transition": ratios,
        "shape": shape,
        "assumptions": {
            "linear": "relative L2 grows proportionally to completed decoder layers, anchored at zero before layer 0",
            "geometric": "the geometric mean ratio across observed completed layers persists through layer 31",
        },
        "linear_extrapolated_relative_l2_at_layer31": linear,
        "geometric_mean_ratio": geometric_ratio,
        "geometric_extrapolated_relative_l2_at_layer31": geometric,
        "chosen_model": chosen_model,
        "chosen_extrapolated_relative_l2_at_layer31": chosen,
        "observed_production_final_relative_l2": FINAL_RELATIVE_L2,
        "chosen_fraction_of_production_final": fraction,
        "verdict": verdict,
    }


def terminal_boundary_assessment(
    layer_metrics: list[dict[str, Any]], terminal_metrics: list[dict[str, Any]]
) -> dict[str, Any]:
    """Records stage-boundary facts without treating sampled logits as full vocab."""
    if not terminal_metrics:
        return {"status": "not_measured"}
    final_norm = next((item for item in terminal_metrics if item["stage"] == "final_norm"), None)
    lm_head = next((item for item in terminal_metrics if item["stage"] == "lm_head"), None)
    if final_norm is None or lm_head is None:
        raise ValueError("terminal stages are incomplete")
    if not layer_metrics or int(layer_metrics[-1]["layer_index"]) != 31:
        raise ValueError("terminal stages require an observed layer 31")
    decoder = layer_metrics[-1]["aggregate"]
    norm = final_norm["aggregate"]
    decoder_l2 = float(decoder["relative_l2"])
    norm_l2 = float(norm["relative_l2"])
    return {
        "status": "measured",
        "pre_final_norm_stage": "decoder_layer_31_output",
        "pre_final_norm_relative_l2": decoder_l2,
        "final_norm_relative_l2": norm_l2,
        "final_norm_delta_relative_l2": norm_l2 - decoder_l2,
        "final_norm_ratio_to_layer31": norm_l2 / decoder_l2 if decoder_l2 else None,
        "lm_head_measurement_scope": lm_head["measurement_scope"],
        "lm_head_coordinates": lm_head["coordinates"],
        "lm_head_sampled_relative_l2": lm_head["aggregate"]["relative_l2"],
        "lm_head_sampled_cosine": lm_head["aggregate"]["cosine"],
        "lm_head_sampled_max_abs": lm_head["aggregate"]["max_abs"],
        "interpretation_limit": "LM-head values are fixed-row samples, not a full-vocabulary relative L2. The final-norm boundary is full-hidden and directly comparable with layer 31.",
    }


def write_growth_artifacts(
    output: Path,
    metrics: list[dict[str, Any]],
    extrapolation: dict[str, Any],
    terminal_metrics: list[dict[str, Any]],
) -> list[Path]:
    csv_path = output / "growth-curve.csv"
    rows = ["stage_order,stage,layer_index,kind,measurement_scope,coordinates,relative_l2,cosine,max_abs,records"]
    for item in metrics:
        aggregate = item["aggregate"]
        rows.append(
            f"{item['layer_index']},decoder_layer,{item['layer_index']},{item['kind']},full_hidden,,{aggregate['relative_l2']:.12g},{aggregate['cosine']:.12g},{aggregate['max_abs']:.12g},{aggregate['records']}"
        )
    for offset, item in enumerate(terminal_metrics, start=len(metrics)):
        aggregate = item["aggregate"]
        coordinates = "|".join(str(value) for value in item["coordinates"])
        rows.append(
            f"{offset},{item['stage']},,{item['kind']},{item['measurement_scope']},{coordinates},{aggregate['relative_l2']:.12g},{aggregate['cosine']:.12g},{aggregate['max_abs']:.12g},{aggregate['records']}"
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    md_path = output / "growth-curve.md"
    lines = [
        "# AQ4 multi-layer accumulation growth curve",
        "",
        "| stage | kind | scope | relative L2 | cosine | max abs | records |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in metrics:
        aggregate = item["aggregate"]
        lines.append(
            f"| layer {item['layer_index']} | {item['kind']} | full hidden | {aggregate['relative_l2']:.6f} | {aggregate['cosine']:.6f} | {aggregate['max_abs']:.6f} | {aggregate['records']} |"
        )
    for item in terminal_metrics:
        aggregate = item["aggregate"]
        coordinate_note = (
            "" if not item["coordinates"] else f"; rows {','.join(str(value) for value in item['coordinates'])}"
        )
        lines.append(
            f"| {item['stage']} | {item['kind']} | {item['measurement_scope']}{coordinate_note} | {aggregate['relative_l2']:.6f} | {aggregate['cosine']:.6f} | {aggregate['max_abs']:.6f} | {aggregate['records']} |"
        )
    lines.extend(
        [
            "",
            f"Shape: `{extrapolation['shape']}`; selected model: `{extrapolation['chosen_model']}`.",
            f"Layer-31 {'observation' if extrapolation['complete_decoder_stack'] else 'extrapolation'}: `{extrapolation['chosen_extrapolated_relative_l2_at_layer31']:.6f}` vs observed production final `{FINAL_RELATIVE_L2:.6f}` ({extrapolation['chosen_fraction_of_production_final']:.1%}); verdict: `{extrapolation['verdict']}`.",
            f"Linear extrapolation: `{extrapolation['linear_extrapolated_relative_l2_at_layer31']:.6f}`.",
            f"Geometric extrapolation: `{extrapolation['geometric_extrapolated_relative_l2_at_layer31']!r}` (mean ratio `{extrapolation['geometric_mean_ratio']!r}`).",
            "",
            "Final norm is a full-hidden comparison. LM head is explicitly fixed-row sampled, not a full-vocabulary comparison.",
            "This is a CPU-only diagnostic, not a production-path or GPU-kernel measurement.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, md_path]


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.aq4_output.exists():
        raise ValueError("refusing to overwrite comparison or AQ4 output")
    if not args.chain_binary.is_file():
        raise ValueError(f"chain binary is missing: {args.chain_binary}")
    start, end = parse_layer_range(args.chain_layer_range)
    if start != 0:
        raise ValueError(
            "the CPU chain starts from the embedded residual, so --chain-layer-range must start at 0"
        )
    if args.chain_include_final_norm_lm_head and end != 31:
        raise ValueError("--chain-include-final-norm-lm-head requires --chain-layer-range 0:31")
    source_topology = source_config(args.source_model)
    selected_types = source_topology["layer_types"][start : end + 1]
    header, cases, input_sha = HYBRID.load_hybrid_input(args.hybrid_input)
    source_sequences = [(case, HYBRID.read_residual(args.hybrid_input, case)) for case in cases]
    args.output.mkdir(parents=True)
    stderr_path = args.output / "aq4.stderr.log"
    command = [
        str(args.chain_binary),
        "--package",
        str(args.package),
        "--hybrid-input",
        str(args.hybrid_input),
        "--chain-layer-range",
        args.chain_layer_range,
        "--output",
        str(args.aq4_output),
        "--chunk-bytes",
        str(args.chunk_bytes),
        "--stage-stream-stdout",
    ]
    if args.post_norm_epsilon_source_control:
        command.append("--post-norm-epsilon-source-control")
    if args.chain_include_final_norm_lm_head:
        command.append("--chain-include-final-norm-lm-head")
    loader = HYBRID.SourceLoader(args.source_model)
    with stderr_path.open("xb") as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr)
        if process.stdout is None:
            raise ValueError("failed to capture AQ4 chain stream")
        reader = ChainReader(process.stdout)
        try:
            for layer_index, source_kind in zip(range(start, end + 1), selected_types):
                if source_kind == "linear_attention":
                    next_sequences = [(case, source_linear_layer(loader, layer_index, sequence)) for case, sequence in source_sequences]
                    stream_kind = "linear_attention"
                else:
                    next_sequences = [(case, source_self_attention_layer(loader, layer_index, sequence)) for case, sequence in source_sequences]
                    stream_kind = "self_attention"
                for case, sequence in next_sequences:
                    for timestep in range(case["context_length"]):
                        reader.expect(layer_index, stream_kind, case, timestep, sequence[timestep])
                source_sequences = next_sequences
            if args.chain_include_final_norm_lm_head:
                final_sequences = [(case, source_final_rmsnorm(loader, sequence)) for case, sequence in source_sequences]
                for case, sequence in final_sequences:
                    for timestep in range(case["context_length"]):
                        reader.expect_terminal(
                            "final_norm", "full_hidden", (), case, timestep, sequence[timestep]
                        )
                lm_head_rows = loader.lm_head_rows(LM_HEAD_SAMPLE_ROWS)
                for case, sequence in final_sequences:
                    logits = functional.linear(sequence, lm_head_rows)
                    for timestep in range(case["context_length"]):
                        reader.expect_terminal(
                            "lm_head",
                            "fixed_logit_rows",
                            LM_HEAD_SAMPLE_ROWS,
                            case,
                            timestep,
                            logits[timestep],
                        )
                del lm_head_rows, final_sequences
            reader.expect_end()
        except BaseException:
            process.kill()
            process.wait()
            raise
        returncode = process.wait()
    if returncode != 0:
        detail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise ValueError(f"AQ4 chain binary failed with status {returncode}: {detail}")
    aq4_report_path = args.aq4_output / "aq4-report.json"
    if not aq4_report_path.is_file():
        raise ValueError("AQ4 chain report is missing")
    aq4 = json.loads(aq4_report_path.read_text(encoding="utf-8"))
    if aq4.get("schema_version") != AQ4_SCHEMA or aq4.get("status") != "valid":
        raise ValueError("AQ4 chain report schema/status differs")
    if aq4["input"]["consumed_sha256"] != input_sha or aq4["input"]["rows"] != len(cases):
        raise ValueError("AQ4 chain report input binding differs")
    topology = aq4["chain"]["layers"]
    expected_topology = [
        {"layer_index": index, "kind": "self_attention" if kind == "full_attention" else "linear_attention"}
        for index, kind in zip(range(start, end + 1), selected_types)
    ]
    if topology != expected_topology:
        raise ValueError(f"AQ4 manifest-derived topology differs from source config: {topology}")
    metrics = []
    for item in expected_topology:
        accumulator = reader.accumulators.get(item["layer_index"])
        if accumulator is None:
            raise ValueError(f"missing metrics for layer {item['layer_index']}")
        metrics.append({**item, "aggregate": accumulator.report()})
    terminal_metrics: list[dict[str, Any]] = []
    if args.chain_include_final_norm_lm_head:
        expected_terminal_contract = [
            ("final_norm", "final_rmsnorm", "full_hidden", ()),
            ("lm_head", "aq4_lm_head_projection", "fixed_logit_rows", LM_HEAD_SAMPLE_ROWS),
        ]
        if aq4["chain"].get("includes_final_norm_lm_head") is not True:
            raise ValueError("AQ4 chain report did not record requested terminal stages")
        if aq4["chain"].get("final_norm_tensor") != FINAL_NORM_TENSOR:
            raise ValueError("AQ4 chain final norm tensor differs")
        if aq4["chain"].get("lm_head_tensor") != LM_HEAD_TENSOR:
            raise ValueError("AQ4 chain LM-head tensor differs")
        if aq4["chain"].get("lm_head_weight_representation") != "aq4_dequantized_fixed_rows":
            raise ValueError("AQ4 chain LM-head weight representation differs")
        if aq4["chain"].get("lm_head_sample_rows") != list(LM_HEAD_SAMPLE_ROWS):
            raise ValueError("AQ4 chain LM-head sample rows differ")
        report_terminals = aq4.get("terminal_summaries")
        if not isinstance(report_terminals, list) or len(report_terminals) != len(expected_terminal_contract):
            raise ValueError("AQ4 chain terminal summary count differs")
        for stage, kind, scope, coordinates in expected_terminal_contract:
            accumulator = reader.terminal_accumulators.get(stage)
            if accumulator is None:
                raise ValueError(f"missing metrics for terminal {stage}")
            report_terminal = next((item for item in report_terminals if item.get("stage") == stage), None)
            if not isinstance(report_terminal, dict):
                raise ValueError(f"AQ4 chain terminal summary is missing {stage}")
            if (
                report_terminal.get("measurement_scope") != scope
                or report_terminal.get("coordinates") != list(coordinates)
                or report_terminal.get("output", {}).get("elements_per_record")
                != (HIDDEN if stage == "final_norm" else len(coordinates))
            ):
                raise ValueError(f"AQ4 chain terminal summary contract differs for {stage}")
            terminal_metrics.append(
                {
                    "stage": stage,
                    "kind": kind,
                    "measurement_scope": scope,
                    "coordinates": list(coordinates),
                    "aggregate": accumulator.report(),
                }
            )
    elif aq4["chain"].get("includes_final_norm_lm_head") is not False:
        raise ValueError("AQ4 chain report unexpectedly includes terminal stages")
    extrapolation = extrapolate(metrics)
    boundary_assessment = terminal_boundary_assessment(metrics, terminal_metrics)
    growth_paths = write_growth_artifacts(args.output, metrics, extrapolation, terminal_metrics)
    result = {
        "schema_version": SCHEMA,
        "status": "valid",
        "classification": "complete_decoder_and_terminal_measured"
        if args.chain_include_final_norm_lm_head
        else extrapolation["verdict"],
        "promotion": False,
        "holdout": "not_run",
        "policy_evaluation": "policy_not_evaluated",
        "device": "cpu-only",
        "input": {"path": str(args.hybrid_input), "sha256": input_sha, "schema": header["schema_version"], "rows": len(cases)},
        "aq4_probe": {
            "binary": str(args.chain_binary),
            "binary_sha256": HYBRID.sha256_file(args.chain_binary),
            "command": command,
            "report_path": str(aq4_report_path),
            "report_sha256": HYBRID.sha256_file(aq4_report_path),
            "package_root": aq4["package_root"],
            "package_manifest_sha256": aq4["package_manifest_sha256"],
            "post_rms_epsilon": aq4["chain"]["post_rms_epsilon"],
            "post_rms_epsilon_mode": aq4["chain"]["post_rms_epsilon_mode"],
            "includes_final_norm_lm_head": aq4["chain"]["includes_final_norm_lm_head"],
            "lm_head_weight_representation": aq4["chain"].get("lm_head_weight_representation"),
        },
        "source_model": loader.identity(),
        "topology": {"source_config": source_topology, "selected_layers": expected_topology},
        "comparison_contract": "Each AQ4 f32 decoder/final-norm output is compared immediately with the matching BF16 source output. LM head applies AQ4-dequantized fixed token rows only, then compares those sampled logits with BF16 source rows. Only the current source layer sequence and per-stage aggregate/fixed-coordinate metrics are retained; no full all-layer hidden/state or full vocabulary tensor is retained.",
        "layer_metrics": metrics,
        "terminal_metrics": terminal_metrics,
        "boundary_assessment": boundary_assessment,
        "growth_curve": extrapolation,
        "growth_artifacts": [{"path": str(path), "sha256": HYBRID.sha256_file(path)} for path in growth_paths],
    }
    report_path = args.output / "comparison.json"
    report_path.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_paths = [report_path, *growth_paths]
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{HYBRID.sha256_file(path)}  {path.name}\n" for path in checksum_paths), encoding="ascii"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-binary", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--hybrid-input", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--chain-layer-range", required=True)
    parser.add_argument("--aq4-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--post-norm-epsilon-source-control", action="store_true")
    parser.add_argument("--chain-include-final-norm-lm-head", action="store_true")
    args = parser.parse_args()
    try:
        if args.chunk_bytes <= 0:
            raise ValueError("chunk bytes must be positive")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        result = run_compare(args)
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        print(f"AQ4 multi-layer accumulation comparison failed: {error}")
        return 1
    print(json.dumps({"status": result["status"], "growth_curve": result["growth_curve"]}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
