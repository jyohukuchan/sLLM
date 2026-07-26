#!/usr/bin/env python3
"""Run a bounded, CPU-only source-vs-AQ4_0 Qwen3.5-35B MoE forward.

The 35B checkpoint cannot be loaded as one Hugging Face model on this host.
This diagnostic instead keeps only one decoder layer and the experts selected
by a small fixed prefill in memory.  It executes all 40 decoder layers twice:

* source: the selected expert rows are read directly from BF16 safetensors;
* AQ4_0: the selected expert rows are decoded from the completed package.

All non-expert decoder tensors are read from the same source checkpoint for
both passes.  The final package verifier separately establishes that these
raw passthrough files are byte-identical.  Thus a changed router decision here
can only be caused by accumulated routed-expert quantization error, rather
than by a duplicated full-model load or a raw-weight mismatch.

This is deliberately a bounded layer-streaming diagnostic, not a corpus,
campaign, bitwise gate, or serving path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np


class ValidationError(RuntimeError):
    """Raised for a deterministic input/package contract failure."""


AQ4_CODEBOOK_ENTRIES = 16
ROUTED_GATE_UP = "moe_routed_gate_up"
ROUTED_DOWN = "moe_routed_down"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--token-ids",
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated text token IDs for the bounded prefill.",
    )
    parser.add_argument(
        "--right-mode",
        choices=("aq4_0", "source"),
        default="aq4_0",
        help="Use source as the right-hand control, or decode AQ4_0 expert rows.",
    )
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_token_ids(value: str, vocab_size: int) -> list[int]:
    try:
        token_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValidationError(f"--token-ids must be comma-separated integers: {value!r}") from exc
    if not token_ids:
        raise ValidationError("--token-ids must contain at least one token")
    if len(token_ids) > 16:
        raise ValidationError("--token-ids is intentionally bounded to at most 16 tokens")
    if any(token < 0 or token >= vocab_size for token in token_ids):
        raise ValidationError(f"--token-ids contains a value outside [0, {vocab_size})")
    return token_ids


class SourceReader:
    def __init__(self, model_dir: Path, weight_map: dict[str, str]):
        self.model_dir = model_dir
        self.weight_map = weight_map

    def _path(self, name: str) -> Path:
        try:
            return self.model_dir / self.weight_map[name]
        except KeyError as exc:
            raise ValidationError(f"source checkpoint does not contain {name}") from exc

    def tensor(self, name: str):
        from safetensors import safe_open

        with safe_open(self._path(name), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def rows(self, name: str, expert: int):
        from safetensors import safe_open

        with safe_open(self._path(name), framework="pt", device="cpu") as handle:
            return handle.get_slice(name)[expert].contiguous()

    def embedding_rows(self, name: str, token_ids: list[int]):
        from safetensors import safe_open

        with safe_open(self._path(name), framework="pt", device="cpu") as handle:
            return handle.get_slice(name)[token_ids].contiguous()


def e4m3_scale_values() -> np.ndarray:
    values: list[float] = []
    exponent_bits, mantissa_bits = 4, 3
    bias = (1 << (exponent_bits - 1)) - 1
    max_exponent = (1 << exponent_bits) - 1
    mantissa_count = 1 << mantissa_bits
    for exponent in range(max_exponent):
        for mantissa in range(mantissa_count):
            if exponent == 0:
                if mantissa:
                    values.append((mantissa / mantissa_count) * 2.0 ** (1 - bias))
            else:
                values.append((1.0 + mantissa / mantissa_count) * 2.0 ** (exponent - bias))
    return np.asarray(sorted(set(values)), dtype=np.float32)


class Aq4Tensor:
    def __init__(self, package_dir: Path, manifest: dict[str, Any]):
        self.name = str(manifest["name"])
        self.shape = tuple(int(value) for value in manifest["shape"])
        self.group_size = int(manifest["group_size"])
        if len(self.shape) != 3 or self.group_size not in (8, 16):
            raise ValidationError(f"unexpected AQ4_0 expert tensor contract: {self.name} {self.shape}")
        self.rows = self.shape[1]
        self.columns = self.shape[2]
        self.tensor_scale = np.float32(manifest["tensor_scale"])
        self.indices = np.memmap(package_dir / str(manifest["index_file"]), dtype=np.uint8, mode="r")
        self.scale_indices = np.memmap(package_dir / str(manifest["scale_file"]), dtype=np.uint8, mode="r")
        self.codebook = np.fromfile(package_dir / str(manifest["codebook_file"]), dtype="<f4")
        if self.codebook.size != AQ4_CODEBOOK_ENTRIES:
            raise ValidationError(f"invalid codebook length for {self.name}")
        self.scales = e4m3_scale_values()

    def expert(self, expert: int):
        import torch

        experts, rows, columns = self.shape
        if expert < 0 or expert >= experts:
            raise ValidationError(f"expert {expert} is outside {self.name}")
        elements = rows * columns
        start = expert * elements
        packed = np.asarray(self.indices[start // 2 : (start + elements) // 2], dtype=np.uint8)
        codebook_indices = np.empty(elements, dtype=np.uint8)
        codebook_indices[0::2] = packed & np.uint8(0x0F)
        codebook_indices[1::2] = packed >> np.uint8(4)
        scale_start = start // self.group_size
        scale_ids = np.asarray(
            self.scale_indices[scale_start : scale_start + elements // self.group_size], dtype=np.intp
        )
        if np.any(scale_ids >= self.scales.size):
            raise ValidationError(f"invalid E4M3 scale index in {self.name}")
        expanded = np.repeat(self.scales[scale_ids] * self.tensor_scale, self.group_size)
        decoded = (self.codebook[codebook_indices] * expanded).reshape(rows, columns).copy()
        return torch.from_numpy(decoded)


class ExpertProvider:
    def __init__(
        self,
        mode: str,
        source: SourceReader,
        layer_idx: int,
        package_gate_up: Aq4Tensor,
        package_down: Aq4Tensor,
    ):
        self.mode = mode
        self.source = source
        self.gate_name = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        self.down_name = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
        self.package_gate_up = package_gate_up
        self.package_down = package_down
        self.cache: dict[int, tuple[Any, Any]] = {}

    def weights(self, expert: int):
        if expert not in self.cache:
            if self.mode == "source":
                gate_up = self.source.rows(self.gate_name, expert).float()
                down = self.source.rows(self.down_name, expert).float()
            elif self.mode == "aq4_0":
                gate_up = self.package_gate_up.expert(expert)
                down = self.package_down.expert(expert)
            else:  # pragma: no cover - constructor-only contract
                raise ValidationError(f"unknown provider mode {self.mode}")
            self.cache[expert] = (gate_up, down)
        return self.cache[expert]


class RouteTracker:
    def __init__(self):
        self.selected_ids: list[list[int]] | None = None


def streamed_experts_module(provider: ExpertProvider, tracker: RouteTracker):
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class StreamedExperts(nn.Module):
        def forward(self, hidden_states, top_k_index, top_k_weights):
            tracker.selected_ids = top_k_index.detach().to(dtype=torch.int32).cpu().tolist()
            final_hidden_states = torch.zeros_like(hidden_states)
            for expert in torch.unique(top_k_index).tolist():
                coordinates = (top_k_index == expert).nonzero(as_tuple=False)
                token_indices = coordinates[:, 0]
                slot_indices = coordinates[:, 1]
                gate_up, down = provider.weights(int(expert))
                current = hidden_states.index_select(0, token_indices)
                gate, up = functional.linear(current, gate_up).chunk(2, dim=-1)
                output = functional.linear(functional.silu(gate) * up, down)
                output = output * top_k_weights[token_indices, slot_indices].float().unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, output.to(final_hidden_states.dtype))
            return final_hidden_states

    return StreamedExperts()


def load_base_layer(layer, source: SourceReader, layer_idx: int) -> None:
    """Load only non-routed-expert decoder state as F32 for both tracks."""
    prefix = f"model.language_model.layers.{layer_idx}."
    state: dict[str, Any] = {}
    for local_name in layer.state_dict():
        if local_name.startswith("mlp.experts."):
            continue
        state[local_name] = source.tensor(prefix + local_name).float()
    missing, unexpected = layer.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValidationError(
            f"layer {layer_idx} non-expert state load differs: missing={missing}, unexpected={unexpected}"
        )


def causal_mask(sequence_length: int):
    import torch

    blocked = torch.full((sequence_length, sequence_length), torch.finfo(torch.float32).min)
    return torch.triu(blocked, diagonal=1).view(1, 1, sequence_length, sequence_length)


def relative_l2(left, right) -> float:
    import torch

    numerator = torch.linalg.vector_norm((left - right).reshape(-1))
    denominator = torch.linalg.vector_norm(left.reshape(-1))
    return float((numerator / denominator).item()) if float(denominator) else 0.0


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise ValidationError("--threads must be positive")
    args.model_dir = args.model_dir.resolve()
    args.package_dir = args.package_dir.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValidationError(f"refusing to overwrite output {args.output}")
    if not (args.model_dir / "config.json").is_file() or not (args.model_dir / "model.safetensors.index.json").is_file():
        raise ValidationError("--model-dir must contain config.json and model.safetensors.index.json")
    if not (args.package_dir / "manifest.json").is_file():
        raise ValidationError("--package-dir must contain manifest.json")

    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeTextRotaryEmbedding,
    )

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # The interpreter may already have selected its inter-op pool.  It does
        # not affect correctness, and the recorded intra-op setting is enough.
        pass
    root_config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    text_config_raw = root_config.get("text_config")
    if root_config.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"] or not isinstance(text_config_raw, dict):
        raise ValidationError("source model is not the expected Qwen3.5-35B-A3B MoE text checkpoint")
    config = Qwen3_5MoeTextConfig(**text_config_raw)
    config._attn_implementation = "eager"
    token_ids = parse_token_ids(args.token_ids, int(config.vocab_size))
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValidationError("source safetensors index lacks weight_map")
    package_manifest = json.loads((args.package_dir / "manifest.json").read_text(encoding="utf-8"))
    package_tensors = {str(row["name"]): row for row in package_manifest.get("tensors", [])}
    if len(package_tensors) != 80:
        raise ValidationError("package does not have exactly 80 quantized routed tensors")

    source = SourceReader(args.model_dir, {str(key): str(value) for key, value in weight_map.items()})
    embedding_name = "model.language_model.embed_tokens.weight"
    source_hidden = source.embedding_rows(embedding_name, token_ids).float().unsqueeze(0)
    aq4_hidden = source_hidden.clone()
    source_cache = DynamicCache(config=config)
    aq4_cache = DynamicCache(config=config)
    position_ids = torch.arange(len(token_ids), dtype=torch.long).view(1, 1, -1).expand(3, 1, -1)
    rope = Qwen3_5MoeTextRotaryEmbedding(config)
    full_attention_mask = causal_mask(len(token_ids))
    rows: list[dict[str, Any]] = []
    started = time.monotonic()

    with torch.no_grad():
        for layer_idx, layer_type in enumerate(config.layer_types):
            gate_name = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
            down_name = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
            try:
                package_gate = Aq4Tensor(args.package_dir, package_tensors[gate_name])
                package_down = Aq4Tensor(args.package_dir, package_tensors[down_name])
            except KeyError as exc:
                raise ValidationError(f"package lacks routed tensor {exc.args[0]}") from exc

            # Construction transiently creates the HF rank-3 expert Parameters;
            # replace them before loading state so only selected rows are used.
            layer = Qwen3_5MoeDecoderLayer(config, layer_idx)
            bootstrap_tracker = RouteTracker()
            layer.mlp.experts = streamed_experts_module(
                ExpertProvider("source", source, layer_idx, package_gate, package_down), bootstrap_tracker
            )
            layer.eval()
            load_base_layer(layer, source, layer_idx)
            mask = full_attention_mask if layer_type == "full_attention" else None

            source_tracker = RouteTracker()
            layer.mlp.experts = streamed_experts_module(
                ExpertProvider("source", source, layer_idx, package_gate, package_down), source_tracker
            )
            source_output = layer(
                source_hidden,
                position_embeddings=rope(source_hidden, position_ids),
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=source_cache,
            )
            aq4_tracker = RouteTracker()
            layer.mlp.experts = streamed_experts_module(
                ExpertProvider(args.right_mode, source, layer_idx, package_gate, package_down), aq4_tracker
            )
            aq4_output = layer(
                aq4_hidden,
                position_embeddings=rope(aq4_hidden, position_ids),
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=aq4_cache,
            )
            if source_tracker.selected_ids is None or aq4_tracker.selected_ids is None:
                raise ValidationError(f"layer {layer_idx} did not execute its MoE router")
            # A router's ordered top-k sequence can differ solely because two
            # selected experts swap rank.  The expert module's weighted sum is
            # order-independent, so retain that strict ordered comparison for
            # reproducibility while also recording the semantically relevant
            # selected-expert-set comparison.
            ordered_changed_tokens = sum(
                int(source_ids != aq4_ids)
                for source_ids, aq4_ids in zip(source_tracker.selected_ids, aq4_tracker.selected_ids)
            )
            set_changed_tokens = sum(
                int(set(source_ids) != set(aq4_ids))
                for source_ids, aq4_ids in zip(source_tracker.selected_ids, aq4_tracker.selected_ids)
            )
            rows.append(
                {
                    "layer": layer_idx,
                    "layer_type": layer_type,
                    "source_topk_ids": source_tracker.selected_ids,
                    "aq4_0_topk_ids": aq4_tracker.selected_ids,
                    # Kept as a backwards-compatible alias for the strict
                    # ordered result.  Consumers should use the explicit
                    # fields below when judging expert-selection invariance.
                    "topk_changed_tokens": ordered_changed_tokens,
                    "topk_order_changed_tokens": ordered_changed_tokens,
                    "topk_selected_set_changed_tokens": set_changed_tokens,
                    "layer_relative_l2": relative_l2(source_output, aq4_output),
                    "layer_max_abs_error": float(torch.max(torch.abs(source_output - aq4_output)).item()),
                }
            )
            source_hidden = source_output.detach()
            aq4_hidden = aq4_output.detach()
            del layer

    ordered_changed_tokens = sum(int(row["topk_order_changed_tokens"]) for row in rows)
    set_changed_tokens = sum(int(row["topk_selected_set_changed_tokens"]) for row in rows)
    result = {
        "schema_version": "ullm.qwen35_moe_aq4_streaming_forward.v1",
        "scope": "CPU-only, one-layer-at-a-time decoder prefill; no full checkpoint materialization, no final norm/head, no serving",
        "model_dir": str(args.model_dir),
        "source_config_sha256": sha256_file(args.model_dir / "config.json"),
        "package_dir": str(args.package_dir),
        "package_manifest_sha256": sha256_file(args.package_dir / "manifest.json"),
        "token_ids": token_ids,
        "tokens": len(token_ids),
        "layers": rows,
        "topk": {
            "layers_checked": len(rows),
            "tokens_checked": len(rows) * len(token_ids),
            "topk_changed_tokens": ordered_changed_tokens,
            "all_topk_ids_equal": ordered_changed_tokens == 0,
            "topk_order_changed_tokens": ordered_changed_tokens,
            "all_topk_ordered_ids_equal": ordered_changed_tokens == 0,
            "topk_selected_set_changed_tokens": set_changed_tokens,
            "all_topk_selected_sets_equal": set_changed_tokens == 0,
        },
        "final_hidden_relative_l2": relative_l2(source_hidden, aq4_hidden),
        "final_hidden_max_abs_error": float(torch.max(torch.abs(source_hidden - aq4_hidden)).item()),
        "arithmetic": {
            "nonexpert_weights": "source BF16 tensors converted to F32 for both tracks; package verifier establishes raw passthrough identity",
            "routed_expert_source": "selected BF16 expert rows streamed from safetensors",
            "right_routed_expert": (
                "selected AQ4_0 rows decoded from idx4/E4M3/codebook package payload"
                if args.right_mode == "aq4_0"
                else "selected BF16 expert rows streamed from the same safetensors source (control)"
            ),
            "router": "same raw router values on both tracks; route divergence, if any, is caused by upstream quantized routed-expert outputs",
        },
        "threads": args.threads,
        "right_mode": args.right_mode,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    write_json(args.output, result)
    print(args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"validate-qwen35-moe-aq4-streaming-forward: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
