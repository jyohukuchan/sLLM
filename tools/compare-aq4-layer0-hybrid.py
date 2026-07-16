#!/usr/bin/env python3
"""CPU-only BF16-source comparison for the AQ4 layer-0 hybrid diagnostic.

The Rust probe sends one f32 stage tensor at a time over stdout.  This tool
replays the same bounded context on CPU with BF16 source tensors, consumes the
matching frame immediately, and retains only aggregate metrics plus fixed
coordinate samples.  It never writes a full hidden state or full vocabulary
logit tensor to the artifact directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable

import torch
import torch.nn.functional as functional
from safetensors import safe_open


SCHEMA = "ullm.aq4_layer0_hybrid_diagnostic.source_compare.v1"
INPUT_SCHEMA = "ullm.aq4_layer0_hybrid_input_jsonl.v1"
AQ4_SCHEMA = "ullm.aq4_layer0_hybrid_diagnostic.aq4_cpu.v1"
HIDDEN = 4096
QKV = 8192
KEY_HEADS = 16
VALUE_HEADS = 32
KEY_DIM = 128
VALUE_DIM = 128
CONV_KERNEL = 4
INTERMEDIATE = 12288
STATE_ELEMENTS = VALUE_HEADS * KEY_DIM * VALUE_DIM
INPUT_EPS = 1e-6
ATTN_EPS = 1e-6
SOURCE_POST_EPS = 1e-6
Q_SCALE = 1.0 / math.sqrt(KEY_DIM)
MAX_LINE_BYTES = 64 * 1024
MAX_CONTEXT = 512
DIAGNOSTIC_LOGIT_ROWS = tuple(range(32)) + (220, 41330)

TENSORS = {
    "input_norm": "model.language_model.layers.0.input_layernorm.weight",
    "qkv": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    "z": "model.language_model.layers.0.linear_attn.in_proj_z.weight",
    "a": "model.language_model.layers.0.linear_attn.in_proj_a.weight",
    "b": "model.language_model.layers.0.linear_attn.in_proj_b.weight",
    "conv": "model.language_model.layers.0.linear_attn.conv1d.weight",
    "a_log": "model.language_model.layers.0.linear_attn.A_log",
    "dt_bias": "model.language_model.layers.0.linear_attn.dt_bias",
    "attn_norm": "model.language_model.layers.0.linear_attn.norm.weight",
    "out": "model.language_model.layers.0.linear_attn.out_proj.weight",
    "post_norm": "model.language_model.layers.0.post_attention_layernorm.weight",
    "mlp_gate": "model.language_model.layers.0.mlp.gate_proj.weight",
    "mlp_up": "model.language_model.layers.0.mlp.up_proj.weight",
    "mlp_down": "model.language_model.layers.0.mlp.down_proj.weight",
    "lm_head": "lm_head.weight",
}

EXPECTED_SHAPES = {
    "input_norm": [HIDDEN],
    "qkv": [QKV, HIDDEN],
    "z": [HIDDEN, HIDDEN],
    "a": [VALUE_HEADS, HIDDEN],
    "b": [VALUE_HEADS, HIDDEN],
    "conv": [QKV, 1, CONV_KERNEL],
    "a_log": [VALUE_HEADS],
    "dt_bias": [VALUE_HEADS],
    "attn_norm": [VALUE_DIM],
    "out": [HIDDEN, HIDDEN],
    "post_norm": [HIDDEN],
    "mlp_gate": [INTERMEDIATE, HIDDEN],
    "mlp_up": [INTERMEDIATE, HIDDEN],
    "mlp_down": [HIDDEN, INTERMEDIATE],
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_token_ids_hash(token_ids: list[int]) -> str:
    return sha256_bytes((json.dumps(token_ids, separators=(",", ":")) + "\n").encode("ascii"))


def safe_relative(root: Path, raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("hybrid residual path must be a relative child path")
    return root / candidate


def load_hybrid_input(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    lines = [line for line in raw.splitlines() if line]
    if not lines:
        raise ValueError("hybrid input is empty")
    header = json.loads(lines[0])
    expected_header = {
        "kind",
        "schema_version",
        "tensor_name",
        "dtype",
        "shape",
        "residual_encoding",
        "source_model_index_sha256",
    }
    if not isinstance(header, dict) or set(header) != expected_header:
        raise ValueError("hybrid input header fields differ")
    if (
        header["kind"] != "header"
        or header["schema_version"] != INPUT_SCHEMA
        or header["tensor_name"] != "model.language_model.embed_tokens.weight"
        or header["dtype"] != "f32"
        or header["shape"] != [HIDDEN]
        or header["residual_encoding"] != "f32le_row_major"
    ):
        raise ValueError("hybrid input header contract differs")
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    expected_case = {
        "kind",
        "case_id",
        "step",
        "context_token_ids",
        "context_token_ids_sha256",
        "context_length",
        "residual_path",
        "residual_sha256",
        "residual_shape",
        "residual_dtype",
    }
    for raw_case in lines[1:]:
        case = json.loads(raw_case)
        if not isinstance(case, dict) or set(case) != expected_case:
            raise ValueError("hybrid input case fields differ")
        tokens = case["context_token_ids"]
        if (
            case["kind"] != "case"
            or not isinstance(case["case_id"], str)
            or not case["case_id"]
            or not isinstance(case["step"], int)
            or case["step"] < 0
            or not isinstance(tokens, list)
            or not tokens
            or len(tokens) > MAX_CONTEXT
            or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in tokens)
            or case["context_length"] != len(tokens)
            or case["residual_shape"] != [len(tokens), HIDDEN]
            or case["residual_dtype"] != "f32le"
            or canonical_token_ids_hash(tokens) != case["context_token_ids_sha256"]
        ):
            raise ValueError(f"hybrid input case contract differs: {case.get('case_id')}")
        key = (case["case_id"], case["step"])
        if key in seen:
            raise ValueError("duplicate hybrid case/step")
        seen.add(key)
        cases.append(case)
    if not cases:
        raise ValueError("hybrid input has no cases")
    return header, cases, digest


def read_residual(input_path: Path, case: dict[str, Any]) -> torch.Tensor:
    sidecar = safe_relative(input_path.parent, case["residual_path"])
    raw = sidecar.read_bytes()
    expected_bytes = case["context_length"] * HIDDEN * 4
    if len(raw) != expected_bytes or sha256_bytes(raw) != case["residual_sha256"]:
        raise ValueError(f"hybrid residual identity differs: {case['case_id']}")
    values = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
    if values.numel() != case["context_length"] * HIDDEN or not bool(torch.isfinite(values).all()):
        raise ValueError("hybrid residual values are invalid")
    return values.reshape(case["context_length"], HIDDEN).to(dtype=torch.bfloat16)


def fixed_coordinates(elements: int) -> list[int]:
    candidates = (0, 1, 31, 127, 1024, 2048, 4095, elements - 1)
    return list(dict.fromkeys(index for index in candidates if 0 <= index < elements))


@dataclass
class MetricAccumulator:
    records: int = 0
    elements_per_record: int | None = None
    diff_sq: float = 0.0
    reference_sq: float = 0.0
    actual_sq: float = 0.0
    dot: float = 0.0
    max_abs: float = 0.0
    samples: list[dict[str, Any]] = field(default_factory=list)

    def update(self, header: dict[str, Any], actual: torch.Tensor, reference: torch.Tensor) -> None:
        actual64 = actual.detach().to(dtype=torch.float64, device="cpu").flatten()
        reference64 = reference.detach().to(dtype=torch.float64, device="cpu").flatten()
        if actual64.numel() != reference64.numel() or not bool(torch.isfinite(actual64).all()) or not bool(torch.isfinite(reference64).all()):
            raise ValueError(f"non-finite or mismatched stage tensor: {header['stage']}")
        elements = int(actual64.numel())
        if self.elements_per_record is None:
            self.elements_per_record = elements
        elif self.elements_per_record != elements:
            raise ValueError(f"stage geometry changed: {header['stage']}")
        difference = actual64 - reference64
        record_diff_sq = float(torch.dot(difference, difference).item())
        record_reference_sq = float(torch.dot(reference64, reference64).item())
        record_actual_sq = float(torch.dot(actual64, actual64).item())
        record_dot = float(torch.dot(actual64, reference64).item())
        record_max_abs = float(difference.abs().max().item())
        self.records += 1
        self.diff_sq += record_diff_sq
        self.reference_sq += record_reference_sq
        self.actual_sq += record_actual_sq
        self.dot += record_dot
        self.max_abs = max(self.max_abs, record_max_abs)
        coordinates = fixed_coordinates(elements)
        actual32 = actual64.to(dtype=torch.float32)
        reference32 = reference64.to(dtype=torch.float32)
        self.samples.append(
            {
                "case_id": header["case_id"],
                "step": header["step"],
                "context_token_ids_sha256": header["context_token_ids_sha256"],
                "context_length": header["context_length"],
                "timestep": header["timestep"],
                "elements": elements,
                "coordinates": coordinates,
                "aq4_values": [float(actual32[index].item()) for index in coordinates],
                "source_values": [float(reference32[index].item()) for index in coordinates],
                "abs_diff_values": [float((actual32[index] - reference32[index]).abs().item()) for index in coordinates],
                "max_abs": record_max_abs,
                "relative_l2": math.sqrt(record_diff_sq) / math.sqrt(record_reference_sq)
                if record_reference_sq
                else (0.0 if record_diff_sq == 0.0 else math.inf),
                "cosine": record_dot / math.sqrt(record_actual_sq * record_reference_sq)
                if record_actual_sq and record_reference_sq
                else None,
            }
        )

    def report(self) -> dict[str, Any]:
        relative_l2 = math.sqrt(self.diff_sq) / math.sqrt(self.reference_sq) if self.reference_sq else (0.0 if self.diff_sq == 0.0 else math.inf)
        cosine = self.dot / math.sqrt(self.actual_sq * self.reference_sq) if self.actual_sq and self.reference_sq else None
        return {
            "records": self.records,
            "elements_per_record": self.elements_per_record,
            "max_abs": self.max_abs,
            "relative_l2": relative_l2,
            "cosine": cosine,
            "samples": self.samples,
        }


class StageReader:
    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.accumulators: dict[str, MetricAccumulator] = {}

    def expect(self, case: dict[str, Any], timestep: int, stage: str, reference: torch.Tensor) -> None:
        header_raw = self.source.readline(MAX_LINE_BYTES + 1)
        if not header_raw:
            raise ValueError(f"AQ4 stage stream ended before {stage}")
        if len(header_raw) > MAX_LINE_BYTES or not header_raw.endswith(b"\n"):
            raise ValueError("AQ4 stage stream header is oversized or unterminated")
        header = json.loads(header_raw)
        expected = {
            "kind": "stage",
            "case_id": case["case_id"],
            "step": case["step"],
            "context_token_ids_sha256": case["context_token_ids_sha256"],
            "context_length": case["context_length"],
            "timestep": timestep,
            "stage": stage,
            "dtype": "f32le",
            "shape": [int(reference.numel())],
            "bytes": int(reference.numel()) * 4,
        }
        if header != expected:
            raise ValueError(f"AQ4 stage stream identity/order differs for {stage}: {header}")
        payload = self.source.read(expected["bytes"])
        if len(payload) != expected["bytes"]:
            raise ValueError(f"AQ4 stage stream payload is short for {stage}")
        actual = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone()
        self.accumulators.setdefault(stage, MetricAccumulator()).update(header, actual, reference)

    def expect_end(self) -> None:
        line = self.source.readline(MAX_LINE_BYTES + 1)
        if line != b'{"kind":"end"}\n':
            raise ValueError("AQ4 stage stream has no valid terminal frame")
        if self.source.read(1):
            raise ValueError("AQ4 stage stream has trailing bytes")

    def reports(self) -> dict[str, dict[str, Any]]:
        return {stage: accumulator.report() for stage, accumulator in sorted(self.accumulators.items())}


class SourceLoader:
    def __init__(self, source_model: Path) -> None:
        self.source_model = source_model
        self.index_path = source_model / "model.safetensors.index.json"
        index = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.weight_map = index.get("weight_map")
        if not isinstance(self.weight_map, dict):
            raise ValueError("source model index has no weight_map")
        self.used_shards: dict[str, str] = {}

    def tensor(self, key: str) -> torch.Tensor:
        name = TENSORS[key]
        if key == "a_log" or key == "attn_norm":
            expected_dtype = torch.float32
        else:
            expected_dtype = torch.bfloat16
        return self.tensor_by_name(name, EXPECTED_SHAPES[key], expected_dtype)

    def tensor_by_name(
        self,
        name: str,
        expected_shape: list[int],
        expected_dtype: torch.dtype,
    ) -> torch.Tensor:
        if name not in self.weight_map:
            raise ValueError(f"source model is missing {name}")
        shard = self.source_model / self.weight_map[name]
        if not shard.is_file():
            raise ValueError(f"source shard is missing: {shard}")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            value = handle.get_tensor(name)
        if list(value.shape) != expected_shape:
            raise ValueError(f"source geometry differs for {name}: {list(value.shape)}")
        if value.dtype != expected_dtype:
            raise ValueError(f"source dtype differs for {name}: {value.dtype}")
        # The chained diagnostic can draw several tensors from one shard.  Hash
        # each shard once for the identity record instead of re-reading its
        # multi-gigabyte contents for every tensor.
        if shard.name not in self.used_shards:
            self.used_shards[shard.name] = sha256_file(shard)
        return value

    def lm_head_rows(self, rows: tuple[int, ...]) -> torch.Tensor:
        name = TENSORS["lm_head"]
        shard = self.source_model / self.weight_map[name]
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            values = [view[row : row + 1] for row in rows]
        result = torch.cat(values, dim=0).contiguous()
        if list(result.shape) != [len(rows), HIDDEN] or result.dtype != torch.bfloat16:
            raise ValueError("source LM-head selected row geometry/dtype differs")
        if shard.name not in self.used_shards:
            self.used_shards[shard.name] = sha256_file(shard)
        return result

    def identity(self) -> dict[str, Any]:
        config_path = self.source_model / "config.json"
        return {
            "model_dir": str(self.source_model),
            "index_path": str(self.index_path),
            "index_sha256": sha256_file(self.index_path),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "used_shards": [{"name": name, "sha256": digest} for name, digest in sorted(self.used_shards.items())],
        }


def source_rmsnorm(hidden: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    normalized = hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return (normalized * (1.0 + weight.float())).to(dtype=hidden.dtype)


def source_attention_gated_norm(recurrent: torch.Tensor, z: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = recurrent.reshape(-1, VALUE_DIM)
    normalized = rows.float() * torch.rsqrt(rows.float().pow(2).mean(dim=-1, keepdim=True) + ATTN_EPS)
    head_norm = (weight.float() * normalized.to(dtype=recurrent.dtype)).reshape_as(recurrent)
    z_silu = functional.silu(z.to(dtype=torch.float32))
    composed = (head_norm * z_silu).to(dtype=recurrent.dtype)
    return head_norm, z_silu, composed


def source_qk_l2norm(values: torch.Tensor) -> torch.Tensor:
    return values * torch.rsqrt((values * values).sum(dim=-1, keepdim=True) + 1e-6)


def source_case(
    reader: StageReader,
    loader: SourceLoader,
    input_path: Path,
    case: dict[str, Any],
) -> torch.Tensor:
    residual = read_residual(input_path, case)
    sequence = case["context_length"]

    input_norm_weight = loader.tensor("input_norm")
    input_normed = source_rmsnorm(residual, input_norm_weight, INPUT_EPS)
    del input_norm_weight
    qkv_weight = loader.tensor("qkv")
    qkv = functional.linear(input_normed, qkv_weight)
    del qkv_weight
    z_weight = loader.tensor("z")
    z = functional.linear(input_normed, z_weight)
    del z_weight
    a_weight = loader.tensor("a")
    a = functional.linear(input_normed, a_weight)
    del a_weight
    b_weight = loader.tensor("b")
    b = functional.linear(input_normed, b_weight)
    del b_weight
    conv_weight = loader.tensor("conv")
    conv_pre = functional.conv1d(
        qkv.transpose(0, 1).unsqueeze(0), conv_weight, padding=CONV_KERNEL - 1, groups=QKV
    )[:, :, :sequence].squeeze(0).transpose(0, 1).contiguous()
    del conv_weight
    conv_silu = functional.silu(conv_pre)
    q_raw = conv_silu[:, : KEY_HEADS * KEY_DIM].reshape(sequence, KEY_HEADS, KEY_DIM)
    k_raw = conv_silu[:, KEY_HEADS * KEY_DIM : KEY_HEADS * KEY_DIM * 2].reshape(sequence, KEY_HEADS, KEY_DIM)
    v = conv_silu[:, KEY_HEADS * KEY_DIM * 2 :].reshape(sequence, VALUE_HEADS, VALUE_DIM)
    q_l2 = source_qk_l2norm(q_raw)
    k_l2 = source_qk_l2norm(k_raw)
    a_log = loader.tensor("a_log")
    dt_bias = loader.tensor("dt_bias")
    gate = -a_log.float().exp().unsqueeze(0) * functional.softplus(a.float() + dt_bias.float().unsqueeze(0))
    beta = torch.sigmoid(b)
    del a_log, dt_bias
    attention_weight = loader.tensor("attn_norm")
    out_weight = loader.tensor("out")
    post_weight = loader.tensor("post_norm")

    conv_state = torch.zeros((CONV_KERNEL, QKV), dtype=torch.float32)
    recurrent_state = torch.zeros((VALUE_HEADS, KEY_DIM, VALUE_DIM), dtype=torch.float32)
    attention_residuals: list[torch.Tensor] = []
    post_normed: list[torch.Tensor] = []
    for timestep in range(sequence):
        reader.expect(case, timestep, "input_rmsnorm", input_normed[timestep])
        reader.expect(case, timestep, "qkv_dequant_row_scale", qkv[timestep])
        reader.expect(case, timestep, "z_dequant_row_scale", z[timestep])
        reader.expect(case, timestep, "a_dequant_row_scale", a[timestep])
        reader.expect(case, timestep, "b_dequant_row_scale", b[timestep])
        conv_state = torch.roll(conv_state, shifts=-1, dims=0)
        conv_state[-1].copy_(qkv[timestep].float())
        reader.expect(case, timestep, "conv_state_after", conv_state)
        reader.expect(case, timestep, "conv_pre_silu", conv_pre[timestep])
        reader.expect(case, timestep, "conv_silu", conv_silu[timestep])
        q_stage = (q_l2[timestep].float() * Q_SCALE).reshape(-1)
        k_stage = k_l2[timestep].float().reshape(-1)
        v_stage = v[timestep].reshape(-1)
        reader.expect(case, timestep, "q_after_l2norm", q_stage)
        reader.expect(case, timestep, "k_after_l2norm", k_stage)
        reader.expect(case, timestep, "v_after_split", v_stage)
        reader.expect(case, timestep, "recurrent_gate", gate[timestep])
        reader.expect(case, timestep, "recurrent_beta", beta[timestep])
        q_repeated = q_l2[timestep].repeat_interleave(2, dim=0).float() * Q_SCALE
        k_repeated = k_l2[timestep].repeat_interleave(2, dim=0).float()
        v_repeated = v[timestep].float()
        recurrent_state.mul_(gate[timestep].exp().reshape(VALUE_HEADS, 1, 1))
        memory = (recurrent_state * k_repeated.unsqueeze(-1)).sum(dim=-2)
        delta = (v_repeated - memory) * beta[timestep].float().unsqueeze(-1)
        recurrent_state.add_(k_repeated.unsqueeze(-1) * delta.unsqueeze(-2))
        recurrent = (recurrent_state * q_repeated.unsqueeze(-1)).sum(dim=-2).to(dtype=torch.bfloat16)
        reader.expect(case, timestep, "recurrent_state_after", recurrent_state)
        reader.expect(case, timestep, "recurrent_output", recurrent.reshape(-1))
        head_norm, z_silu, composed = source_attention_gated_norm(recurrent, z[timestep].reshape(VALUE_HEADS, VALUE_DIM), attention_weight)
        reader.expect(case, timestep, "attention_head_rmsnorm", head_norm.reshape(-1))
        reader.expect(case, timestep, "z_silu", z_silu.reshape(-1))
        reader.expect(case, timestep, "gate_composed", composed.reshape(-1))
        attention_projection = functional.linear(composed.reshape(1, HIDDEN), out_weight).reshape(-1)
        reader.expect(case, timestep, "attention_projection", attention_projection)
        attention_residual = residual[timestep] + attention_projection
        reader.expect(case, timestep, "attention_residual", attention_residual)
        post = source_rmsnorm(attention_residual.reshape(1, HIDDEN), post_weight, SOURCE_POST_EPS).reshape(-1)
        reader.expect(case, timestep, "post_norm", post)
        attention_residuals.append(attention_residual)
        post_normed.append(post)
    del attention_weight, out_weight, post_weight

    post_batch = torch.stack(post_normed, dim=0)
    gate_weight = loader.tensor("mlp_gate")
    mlp_gate = functional.linear(post_batch, gate_weight)
    del gate_weight
    up_weight = loader.tensor("mlp_up")
    mlp_up = functional.linear(post_batch, up_weight)
    del up_weight
    down_weight = loader.tensor("mlp_down")
    final_output: torch.Tensor | None = None
    for timestep in range(sequence):
        reader.expect(case, timestep, "mlp_gate_projection", mlp_gate[timestep])
        reader.expect(case, timestep, "mlp_up_projection", mlp_up[timestep])
        gate_silu = functional.silu(mlp_gate[timestep])
        reader.expect(case, timestep, "mlp_gate_silu", gate_silu)
        activation = gate_silu * mlp_up[timestep]
        reader.expect(case, timestep, "mlp_activation", activation)
        mlp_output = functional.linear(activation.reshape(1, INTERMEDIATE), down_weight).reshape(-1)
        reader.expect(case, timestep, "mlp_output", mlp_output)
        layer_output = attention_residuals[timestep] + mlp_output
        reader.expect(case, timestep, "layer_output", layer_output)
        if timestep + 1 == sequence:
            final_output = layer_output
    del down_weight
    if final_output is None:
        raise ValueError("source hybrid case has no final layer output")
    return final_output


def source_lm_head_readout(reader: StageReader, loader: SourceLoader, finals: list[tuple[dict[str, Any], torch.Tensor]]) -> None:
    rows = loader.lm_head_rows(DIAGNOSTIC_LOGIT_ROWS)
    for case, final_output in finals:
        logits = functional.linear(final_output.reshape(1, HIDDEN), rows).reshape(-1)
        reader.expect(case, case["context_length"] - 1, "diagnostic_lm_head_readout_logits", logits)
    del rows


def run_hybrid_compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ValueError(f"refusing to overwrite comparison output: {args.output}")
    if args.aq4_output.exists():
        raise ValueError(f"refusing to overwrite AQ4 output: {args.aq4_output}")
    if not args.hybrid_binary.is_file():
        raise ValueError(f"hybrid binary is missing: {args.hybrid_binary}")
    header, cases, input_sha = load_hybrid_input(args.hybrid_input)
    for case in cases:
        _ = read_residual(args.hybrid_input, case)
    args.output.mkdir(parents=True)
    stderr_path = args.output / "aq4.stderr.log"
    command = [
        str(args.hybrid_binary),
        "--package",
        str(args.package),
        "--hybrid-input",
        str(args.hybrid_input),
        "--output",
        str(args.aq4_output),
        "--chunk-bytes",
        str(args.chunk_bytes),
        "--stage-stream-stdout",
    ]
    if args.post_norm_epsilon_source_control:
        command.append("--post-norm-epsilon-source-control")
    loader = SourceLoader(args.source_model)
    with stderr_path.open("xb") as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr)
        if process.stdout is None:
            raise ValueError("failed to capture AQ4 hybrid stage stream")
        reader = StageReader(process.stdout)
        try:
            finals = [(case, source_case(reader, loader, args.hybrid_input, case)) for case in cases]
            source_lm_head_readout(reader, loader, finals)
            reader.expect_end()
        except BaseException:
            process.kill()
            process.wait()
            raise
        returncode = process.wait()
    if returncode != 0:
        detail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise ValueError(f"AQ4 hybrid binary failed with status {returncode}: {detail}")
    aq4_report_path = args.aq4_output / "aq4-report.json"
    if not aq4_report_path.is_file():
        raise ValueError("AQ4 hybrid report is missing")
    aq4 = json.loads(aq4_report_path.read_text(encoding="utf-8"))
    if aq4.get("schema_version") != AQ4_SCHEMA or aq4.get("status") != "valid":
        raise ValueError("AQ4 hybrid report schema/status differs")
    if aq4["input"]["consumed_sha256"] != input_sha or aq4["input"]["rows"] != len(cases):
        raise ValueError("AQ4 hybrid report input binding differs")
    stage_reports = reader.reports()
    layer_output = stage_reports.get("layer_output")
    logits = stage_reports.get("diagnostic_lm_head_readout_logits")
    if layer_output is None or logits is None:
        raise ValueError("hybrid comparison lacks final hidden or diagnostic readout metrics")
    result = {
        "schema_version": SCHEMA,
        "status": "valid",
        "classification": "unclassified",
        "promotion": False,
        "holdout": "not_run",
        "policy_evaluation": "policy_not_evaluated",
        "thresholds": None,
        "device": "cpu-only",
        "input": {
            "path": str(args.hybrid_input),
            "sha256": input_sha,
            "schema": header["schema_version"],
            "rows": len(cases),
            "cases": [
                {
                    "case_id": case["case_id"],
                    "step": case["step"],
                    "context_length": case["context_length"],
                    "context_token_ids_sha256": case["context_token_ids_sha256"],
                    "residual_sha256": case["residual_sha256"],
                }
                for case in cases
            ],
        },
        "aq4_probe": {
            "binary": str(args.hybrid_binary),
            "binary_sha256": sha256_file(args.hybrid_binary),
            "command": command[:-1] + ["--stage-stream-stdout"],
            "report_path": str(aq4_report_path),
            "report_sha256": sha256_file(aq4_report_path),
            "package_root": aq4["package_root"],
            "package_manifest_sha256": aq4["package_manifest_sha256"],
            "post_rms_epsilon": aq4["one_at_a_time_hybrid"]["post_rms_epsilon"],
            "post_rms_epsilon_mode": aq4["one_at_a_time_hybrid"]["post_rms_epsilon_mode"],
        },
        "source_model": loader.identity(),
        "source_formula": {
            "input_rms_epsilon": INPUT_EPS,
            "attention_rms_epsilon": ATTN_EPS,
            "post_rms_epsilon": SOURCE_POST_EPS,
            "precision": "source embedding/projections/Conv/MLP use BF16 tensors; source recurrent update is the CPU fallback's per-token f32 recurrent_gated_delta_rule state update and returns BF16 hidden output.",
            "rope": "not_applicable: layer0 is linear attention",
        },
        "comparison_contract": "Each AQ4 f32 stage frame is matched immediately with the corresponding BF16 source stage. Persistent report contains only aggregate metrics and fixed-coordinate samples; full hidden/state/logit tensors are discarded after the comparison update.",
        "stages": stage_reports,
        "endpoints": {
            "layer0_hidden": layer_output,
            "diagnostic_lm_head_readout_logits": logits,
        },
        "notes": [
            "diagnostic_lm_head_readout_logits applies selected LM-head rows directly to layer0 output; it is not a full-model vocabulary logit.",
            "AQ4 post RMSNorm epsilon is intentionally read from the AQ4 report. The optional source-epsilon control changes only this CPU diagnostic invocation and never changes production runtime configuration.",
        ],
    }
    report_path = args.output / "comparison.json"
    report_path.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "SHA256SUMS").write_text(f"{sha256_file(report_path)}  comparison.json\n", encoding="ascii")
    return result


def synthetic_hybrid_fixture() -> dict[str, torch.Tensor]:
    """Small independent fixture used by the dedicated hybrid formula test.

    It deliberately uses one key/value head, width two, a two-tap Conv1d,
    and scalar MLP.  The production dimensions are not required to verify the
    Conv -> SiLU -> recurrent -> gated norm -> residual expression itself.
    """
    residual = torch.tensor([1.0, -2.0], dtype=torch.float32)
    input_weight = torch.tensor([1.5, 0.5], dtype=torch.float32)
    input_norm = residual * torch.rsqrt((residual * residual).mean() + 1e-6) * input_weight
    qkv = torch.tensor([0.5, -0.5, 1.0, 2.0, -1.0, 0.25], dtype=torch.float32)
    conv_weight = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    conv_state = torch.zeros((2, 6), dtype=torch.float32)
    conv_state = torch.roll(conv_state, shifts=-1, dims=0)
    conv_state[-1] = qkv
    conv_pre = (conv_state.transpose(0, 1) * conv_weight).sum(dim=1)
    conv = functional.silu(conv_pre)
    q = conv[:2]
    k = conv[2:4]
    v = conv[4:]
    q = q * torch.rsqrt((q * q).sum() + 1e-6) / math.sqrt(2)
    k = k * torch.rsqrt((k * k).sum() + 1e-6)
    gate = torch.tensor([-0.25], dtype=torch.float32)
    beta = torch.tensor([0.75], dtype=torch.float32)
    state = torch.zeros((1, 2, 2), dtype=torch.float32)
    state = state * gate.exp().reshape(1, 1, 1)
    memory = (state * k.reshape(1, 2, 1)).sum(dim=-2)
    delta = (v.reshape(1, 2) - memory) * beta.reshape(1, 1)
    state = state + k.reshape(1, 2, 1) * delta.reshape(1, 1, 2)
    recurrent = (state * q.reshape(1, 2, 1)).sum(dim=-2).reshape(-1)
    z = torch.tensor([0.5, -1.0], dtype=torch.float32)
    head_norm = recurrent * torch.rsqrt((recurrent * recurrent).mean() + 1e-6)
    composed = head_norm * functional.silu(z)
    attention_projection = torch.tensor([[1.0, 0.0], [0.0, 1.0]]) @ composed
    attention_residual = residual + attention_projection
    post_norm = attention_residual * torch.rsqrt((attention_residual * attention_residual).mean() + 1e-6)
    mlp_gate = post_norm.sum().reshape(1)
    mlp_up = torch.tensor([2.0], dtype=torch.float32)
    activation = functional.silu(mlp_gate) * mlp_up
    mlp_output = torch.tensor([activation.item(), -activation.item()], dtype=torch.float32)
    layer_output = attention_residual + mlp_output
    return {
        "input_norm": input_norm,
        "conv_state": conv_state,
        "conv_pre": conv_pre,
        "q": q,
        "k": k,
        "state": state,
        "recurrent": recurrent,
        "gate_composed": composed,
        "attention_residual": attention_residual,
        "post_norm": post_norm,
        "layer_output": layer_output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-binary", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--hybrid-input", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--aq4-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument(
        "--post-norm-epsilon-source-control",
        action="store_true",
        help="diagnostic-only: pass source post-norm epsilon 1e-6 to the CPU AQ4 probe",
    )
    args = parser.parse_args()
    try:
        if args.chunk_bytes <= 0:
            raise ValueError("chunk bytes must be positive")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        result = run_hybrid_compare(args)
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        print(f"layer0 hybrid comparison failed: {error}")
        return 1
    print(json.dumps({"status": result["status"], "layer0_hidden": result["endpoints"]["layer0_hidden"]}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
