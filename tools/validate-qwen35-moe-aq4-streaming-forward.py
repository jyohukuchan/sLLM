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
from collections.abc import Mapping
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
    parser.add_argument(
        "--generation-suite",
        type=Path,
        help=(
            "Run a bounded greedy generation suite instead of the fixed-token "
            "forward diagnostic. The JSON object must contain a cases array with "
            "id, messages, and max_completion_tokens fields."
        ),
    )
    parser.add_argument(
        "--generation-markdown",
        type=Path,
        help="Human-readable side-by-side generation evidence (required with --generation-suite).",
    )
    parser.add_argument(
        "--generation-max-prompt-tokens",
        type=int,
        default=64,
        help="Fail rather than truncate a rendered generation prompt above this token count.",
    )
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


def write_new_text(path: Path, value: str) -> None:
    """Write an evidence file once; generation runs must not replace prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


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


def causal_mask(sequence_length: int, past_tokens: int = 0):
    import torch

    if sequence_length < 1 or past_tokens < 0:
        raise ValidationError("causal mask dimensions must be non-negative with at least one query token")
    key_length = past_tokens + sequence_length
    key_positions = torch.arange(key_length).view(1, key_length)
    query_positions = (past_tokens + torch.arange(sequence_length)).view(sequence_length, 1)
    blocked = torch.full((sequence_length, key_length), torch.finfo(torch.float32).min)
    return torch.where(key_positions > query_positions, blocked, torch.zeros_like(blocked)).view(
        1, 1, sequence_length, key_length
    )


def relative_l2(left, right) -> float:
    import torch

    numerator = torch.linalg.vector_norm((left - right).reshape(-1))
    denominator = torch.linalg.vector_norm(left.reshape(-1))
    return float((numerator / denominator).item()) if float(denominator) else 0.0


def normalize_token_ids(value: Any) -> list[int]:
    """Accept the list / one-batch shapes returned by the local tokenizer."""
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValidationError("chat template did not return a token-ID list")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValidationError("chat template returned multiple token sequences")
        value = value[0]
    token_ids: list[int] = []
    for index, token_id in enumerate(value):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValidationError(f"chat template token {index} is not an integer")
        token_ids.append(token_id)
    if not token_ids:
        raise ValidationError("chat template returned no tokens")
    return token_ids


def load_generation_suite(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a deliberately small local prompt suite without applying policy judgement."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read generation suite {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise ValidationError("generation suite must be a JSON object with a cases array")
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in raw["cases"]:
        if not isinstance(row, dict):
            raise ValidationError("generation suite contains a non-object case")
        case_id = row.get("id")
        messages = row.get("messages")
        max_completion_tokens = row.get("max_completion_tokens")
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise ValidationError("generation suite case IDs must be non-empty and unique")
        if not isinstance(messages, list) or not messages or not all(isinstance(item, dict) for item in messages):
            raise ValidationError(f"{case_id}: messages must be a non-empty array of objects")
        if not isinstance(max_completion_tokens, int) or not 1 <= max_completion_tokens <= 32:
            raise ValidationError(f"{case_id}: max_completion_tokens must be between 1 and 32")
        seen_ids.add(case_id)
        cases.append(
            {
                "id": case_id,
                "category": str(row.get("category", "unspecified")),
                "expect": row.get("expect", {}),
                "messages": messages,
                "max_completion_tokens": max_completion_tokens,
            }
        )
    if not cases:
        raise ValidationError("generation suite contains no cases")
    return raw, cases


class RouteStatistics:
    """Aggregate path-dependent route changes as an observation, never a gate."""

    def __init__(self):
        self.tokens_checked = 0
        self.ordered_changes = 0
        self.selected_set_changes = 0

    def add(self, source_ids: list[list[int]], right_ids: list[list[int]]) -> None:
        if len(source_ids) != len(right_ids):
            raise ValidationError("source/right route trackers produced different token counts")
        self.tokens_checked += len(source_ids)
        self.ordered_changes += sum(int(left != right) for left, right in zip(source_ids, right_ids))
        self.selected_set_changes += sum(
            int(set(left) != set(right)) for left, right in zip(source_ids, right_ids)
        )

    def result(self) -> dict[str, Any]:
        return {
            "tokens_checked": self.tokens_checked,
            "topk_order_changed_tokens": self.ordered_changes,
            "topk_selected_set_changed_tokens": self.selected_set_changes,
        }


def text_screen(text: str) -> dict[str, Any]:
    """Record only high-confidence text symptoms; semantic judgement stays human-readable."""
    controls = sorted(
        {f"U+{ord(char):04X}" for char in text if ord(char) < 32 and char not in "\n\r\t"}
    )
    repeated: str | None = None
    compact = " ".join(text.split())
    # A three-times-consecutive substring is an intentionally conservative loop signal.
    for width in range(2, min(17, len(compact) // 3 + 1)):
        for start in range(0, len(compact) - width * 3 + 1):
            fragment = compact[start : start + width]
            if fragment.strip() and compact[start : start + width * 3] == fragment * 3:
                repeated = fragment
                break
        if repeated is not None:
            break
    return {
        "empty": not text.strip(),
        "replacement_character": "\ufffd" in text,
        "control_characters": controls,
        "threefold_consecutive_fragment": repeated,
        "characters": len(text),
    }


def generation_markdown(result: dict[str, Any]) -> str:
    """Create the audit-facing side-by-side view from the machine-readable result."""
    right_name = "AQ4_0" if result["right_mode"] == "aq4_0" else "source control"
    lines = [
        "# Qwen3.5-35B-A3B source vs AQ4_0 CPU streaming generation",
        "",
        "This is a bounded CPU-only, one-decoder-layer-at-a-time greedy generation check. "
        "It uses source BF16 checkpoint values converted to F32 arithmetic on both tracks; "
        "only the right track's routed experts are decoded from the `AQ4_0` package. "
        "The final RMSNorm and `lm_head` are raw source/passthrough weights.",
        "",
        "The suite is intentionally shortened from the 10-case lightweight-promotion suite "
        "for CPU feasibility. It is evidence for package reclassification, not a serving or promotion run. "
        "Greedy-token equality and route-set equality are recorded as observations, not pass/fail rules.",
        "",
        f"- right track: `{right_name}`",
        f"- threads: `{result['threads']}`",
        f"- wall time: `{result['wall_seconds']:.3f} s`",
        "",
        "## Side-by-side outputs",
    ]
    for case in result["cases"]:
        lines.extend(["", f"### {case['id']} ({case['category']})", "", "Prompt messages:", "", "```text"])
        for message in case["messages"]:
            lines.append(f"[{message.get('role', 'unknown')}] {message.get('content', '')}")
        lines.extend(
            [
                "```",
                "",
                "Source (nonquantized routed experts):",
                "",
                "`````text",
                case["source"]["generated_text"],
                "`````",
                "",
                f"{right_name}:",
                "",
                "`````text",
                case["right"]["generated_text"],
                "`````",
                "",
                "Observations (not thresholds):",
                "",
                f"- generated tokens: source `{len(case['source']['generated_token_ids'])}`, "
                f"{right_name} `{len(case['right']['generated_token_ids'])}`",
                f"- source-greedy token matches: `{case['comparison']['greedy_token_matches']}`/"
                f"`{case['comparison']['greedy_steps_compared']}`",
                f"- route observations during this path: selected-set "
                f"`{case['route_observation']['topk_selected_set_changed_tokens']}`/"
                f"`{case['route_observation']['tokens_checked']}`, ordered "
                f"`{case['route_observation']['topk_order_changed_tokens']}`/"
                f"`{case['route_observation']['tokens_checked']}`",
                f"- source-greedy conditional NLL: source "
                f"`{case['comparison']['source_greedy_nll_mean_source']:.6f}`, {right_name} "
                f"`{case['comparison']['source_greedy_nll_mean_right']:.6f}` (descriptive only)",
                f"- automatic symptom screen: source `{json.dumps(case['source']['screen'], ensure_ascii=False)}`, "
                f"{right_name} `{json.dumps(case['right']['screen'], ensure_ascii=False)}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Important retained observation",
            "",
            "The original same-input 8-token × 40-layer prefill evidence remains the canonical route observation: "
            "selected expert sets changed 105/320 and ordered top-k changed 238/320 for source vs `AQ4_0`; "
            "the source-vs-source control changed 0/320. This generation record does not treat those rates as a quality gate.",
            "",
        ]
    )
    return "\n".join(lines)


def stream_decoder_pair(
    *,
    config,
    source: SourceReader,
    package_dir: Path,
    package_tensors: dict[str, Any],
    right_mode: str,
    rope,
    source_hidden,
    right_hidden,
    source_cache,
    right_cache,
    past_tokens: int,
    route_stats: RouteStatistics,
):
    """Execute one prefill/decode chunk for both tracks without full-model materialization."""
    import torch
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer

    if source_hidden.shape != right_hidden.shape or source_hidden.ndim != 3:
        raise ValidationError("source/right generation hidden tensors must share [batch, tokens, hidden] shape")
    sequence_length = int(source_hidden.shape[1])
    position_ids = (
        torch.arange(past_tokens, past_tokens + sequence_length, dtype=torch.long)
        .view(1, 1, -1)
        .expand(3, 1, -1)
    )
    full_attention_mask = causal_mask(sequence_length, past_tokens)
    for layer_idx, layer_type in enumerate(config.layer_types):
        gate_name = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        down_name = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
        try:
            package_gate = Aq4Tensor(package_dir, package_tensors[gate_name])
            package_down = Aq4Tensor(package_dir, package_tensors[down_name])
        except KeyError as exc:
            raise ValidationError(f"package lacks routed tensor {exc.args[0]}") from exc
        layer = Qwen3_5MoeDecoderLayer(config, layer_idx)
        # Avoid retaining the constructor's full rank-3 expert Parameters.
        layer.mlp.experts = streamed_experts_module(
            ExpertProvider("source", source, layer_idx, package_gate, package_down), RouteTracker()
        )
        layer.eval()
        load_base_layer(layer, source, layer_idx)
        attention_mask = full_attention_mask if layer_type == "full_attention" else None

        source_tracker = RouteTracker()
        layer.mlp.experts = streamed_experts_module(
            ExpertProvider("source", source, layer_idx, package_gate, package_down), source_tracker
        )
        source_output = layer(
            source_hidden,
            position_embeddings=rope(source_hidden, position_ids),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=source_cache,
        )
        right_tracker = RouteTracker()
        layer.mlp.experts = streamed_experts_module(
            ExpertProvider(right_mode, source, layer_idx, package_gate, package_down), right_tracker
        )
        right_output = layer(
            right_hidden,
            position_embeddings=rope(right_hidden, position_ids),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=right_cache,
        )
        if source_tracker.selected_ids is None or right_tracker.selected_ids is None:
            raise ValidationError(f"layer {layer_idx} did not execute its MoE router")
        route_stats.add(source_tracker.selected_ids, right_tracker.selected_ids)
        source_hidden = source_output.detach()
        right_hidden = right_output.detach()
        del layer
    return source_hidden, right_hidden


def logits_from_hidden(final_norm, lm_head, hidden):
    import torch.nn.functional as functional

    logits = functional.linear(final_norm(hidden[:, -1:, :]), lm_head)[0, 0]
    if not bool(logits.isfinite().all()):
        raise ValidationError("generation produced non-finite logits")
    return logits


def token_nll(logits, token_id: int) -> float:
    import torch

    return float((torch.logsumexp(logits, dim=-1) - logits[token_id]).item())


def run_generation(args: argparse.Namespace) -> int:
    """Run the package-quality evidence path with actual tokenizer-rendered text."""
    if args.generation_suite is None or args.generation_markdown is None:
        raise ValidationError("--generation-suite and --generation-markdown must be provided together")
    if args.generation_max_prompt_tokens < 1:
        raise ValidationError("--generation-max-prompt-tokens must be positive")
    args.generation_suite = args.generation_suite.resolve()
    args.generation_markdown = args.generation_markdown.resolve()
    if args.generation_markdown.exists():
        raise ValidationError(f"refusing to overwrite generation Markdown {args.generation_markdown}")
    suite_raw, suite_cases = load_generation_suite(args.generation_suite)

    import torch
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeRMSNorm,
        Qwen3_5MoeTextRotaryEmbedding,
    )

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    root_config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    text_config_raw = root_config.get("text_config")
    if root_config.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"] or not isinstance(
        text_config_raw, dict
    ):
        raise ValidationError("source model is not the expected Qwen3.5-35B-A3B MoE text checkpoint")
    config = Qwen3_5MoeTextConfig(**text_config_raw)
    config._attn_implementation = "eager"
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValidationError("source safetensors index lacks weight_map")
    package_manifest = json.loads((args.package_dir / "manifest.json").read_text(encoding="utf-8"))
    package_tensors = {str(row["name"]): row for row in package_manifest.get("tensors", [])}
    if len(package_tensors) != 80:
        raise ValidationError("package does not have exactly 80 quantized routed tensors")
    source = SourceReader(args.model_dir, {str(key): str(value) for key, value in weight_map.items()})
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    embedding_name = "model.language_model.embed_tokens.weight"
    final_norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    final_norm.load_state_dict({"weight": source.tensor("model.language_model.norm.weight").float()}, strict=True)
    final_norm.eval()
    # This is a raw passthrough tensor in the completed package. Keeping one F32 copy avoids
    # an additional 1-GiB BF16 safetensors read at every generated token.
    lm_head = source.tensor("lm_head.weight").float()
    rope = Qwen3_5MoeTextRotaryEmbedding(config)
    eos_ids = set(tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, list) else [tokenizer.eos_token_id])
    eos_ids.discard(None)
    cases: list[dict[str, Any]] = []
    started = time.monotonic()

    with torch.inference_mode():
        for case in suite_cases:
            try:
                rendered_prompt = tokenizer.apply_chat_template(
                    case["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                raw_ids = tokenizer.apply_chat_template(
                    case["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
            except Exception as exc:  # pragma: no cover - tokenizer implementation detail
                raise ValidationError(f"{case['id']}: local chat-template rendering failed: {exc}") from exc
            if not isinstance(rendered_prompt, str) or not rendered_prompt:
                raise ValidationError(f"{case['id']}: chat template returned an empty rendered prompt")
            prompt_ids = normalize_token_ids(raw_ids)
            if len(prompt_ids) > args.generation_max_prompt_tokens:
                raise ValidationError(
                    f"{case['id']}: rendered prompt has {len(prompt_ids)} tokens, above "
                    f"--generation-max-prompt-tokens={args.generation_max_prompt_tokens}; refusing to truncate"
                )
            source_cache = DynamicCache(config=config)
            right_cache = DynamicCache(config=config)
            source_hidden = source.embedding_rows(embedding_name, prompt_ids).float().unsqueeze(0)
            right_hidden = source_hidden.clone()
            route_stats = RouteStatistics()
            source_hidden, right_hidden = stream_decoder_pair(
                config=config,
                source=source,
                package_dir=args.package_dir,
                package_tensors=package_tensors,
                right_mode=args.right_mode,
                rope=rope,
                source_hidden=source_hidden,
                right_hidden=right_hidden,
                source_cache=source_cache,
                right_cache=right_cache,
                past_tokens=0,
                route_stats=route_stats,
            )
            initial_hidden_relative_l2 = relative_l2(source_hidden, right_hidden)
            source_generated: list[int] = []
            right_generated: list[int] = []
            source_nll: list[float] = []
            right_nll_on_source: list[float] = []
            greedy_matches = 0
            stop_reason = "max_completion_tokens"
            past_tokens = len(prompt_ids)
            for _ in range(case["max_completion_tokens"]):
                source_logits = logits_from_hidden(final_norm, lm_head, source_hidden)
                right_logits = logits_from_hidden(final_norm, lm_head, right_hidden)
                source_token = int(torch.argmax(source_logits).item())
                right_token = int(torch.argmax(right_logits).item())
                source_generated.append(source_token)
                right_generated.append(right_token)
                source_nll.append(token_nll(source_logits, source_token))
                right_nll_on_source.append(token_nll(right_logits, source_token))
                greedy_matches += int(source_token == right_token)
                if source_token in eos_ids or right_token in eos_ids:
                    stop_reason = "source_eos" if source_token in eos_ids else "right_eos"
                    break
                source_hidden = source.embedding_rows(embedding_name, [source_token]).float().unsqueeze(0)
                right_hidden = source.embedding_rows(embedding_name, [right_token]).float().unsqueeze(0)
                source_hidden, right_hidden = stream_decoder_pair(
                    config=config,
                    source=source,
                    package_dir=args.package_dir,
                    package_tensors=package_tensors,
                    right_mode=args.right_mode,
                    rope=rope,
                    source_hidden=source_hidden,
                    right_hidden=right_hidden,
                    source_cache=source_cache,
                    right_cache=right_cache,
                    past_tokens=past_tokens,
                    route_stats=route_stats,
                )
                past_tokens += 1
            source_text = tokenizer.decode(
                source_generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            right_text = tokenizer.decode(right_generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            cases.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "expect": case["expect"],
                    "messages": case["messages"],
                    "rendered_prompt": rendered_prompt,
                    "prompt_token_ids": prompt_ids,
                    "prompt_token_count": len(prompt_ids),
                    "max_completion_tokens": case["max_completion_tokens"],
                    "stop_reason": stop_reason,
                    "source": {
                        "generated_token_ids": source_generated,
                        "generated_text": source_text,
                        "screen": text_screen(source_text),
                    },
                    "right": {
                        "generated_token_ids": right_generated,
                        "generated_text": right_text,
                        "screen": text_screen(right_text),
                    },
                    "comparison": {
                        "greedy_token_matches": greedy_matches,
                        "greedy_steps_compared": len(source_generated),
                        "source_greedy_nll_mean_source": sum(source_nll) / len(source_nll),
                        "source_greedy_nll_mean_right": sum(right_nll_on_source) / len(right_nll_on_source),
                        "initial_final_hidden_relative_l2": initial_hidden_relative_l2,
                        "final_decode_hidden_relative_l2": relative_l2(source_hidden, right_hidden),
                    },
                    "route_observation": route_stats.result(),
                }
            )
    result = {
        "schema_version": "ullm.qwen35_moe_aq4_streaming_generation.v1",
        "scope": (
            "CPU-only, one-layer-at-a-time decoder streaming with tokenizer-rendered text and greedy decode; "
            "no full checkpoint materialization, no uLLM loader, no service"
        ),
        "quality_policy": {
            "primary_criterion": "human-readable generated-text quality",
            "not_quality_gates": [
                "greedy token exact match",
                "expert selected-set/top-k equality",
                "conditional NLL observations",
            ],
        },
        "model_dir": str(args.model_dir),
        "source_config_sha256": sha256_file(args.model_dir / "config.json"),
        "package_dir": str(args.package_dir),
        "package_manifest_sha256": sha256_file(args.package_dir / "manifest.json"),
        "right_mode": args.right_mode,
        "arithmetic": {
            "nonexpert_weights": "source BF16 tensors converted to F32 for both tracks; completed package verifier establishes raw-passthrough identity",
            "source_routed_experts": "selected BF16 expert rows streamed from safetensors and converted to F32",
            "right_routed_experts": (
                "selected AQ4_0 rows decoded from idx4/E4M3/codebook package payload"
                if args.right_mode == "aq4_0"
                else "selected BF16 expert rows streamed from the same safetensors source"
            ),
            "final_norm_and_lm_head": "raw source/passthrough tensors, evaluated in F32",
        },
        "suite": {
            "path": str(args.generation_suite),
            "sha256": sha256_file(args.generation_suite),
            "declared_schema_version": suite_raw.get("schema_version"),
            "shortened_for_cpu": bool(suite_raw.get("shortened_for_cpu", False)),
            "chat_template_arguments": {"add_generation_prompt": True, "enable_thinking": False},
        },
        "cases": cases,
        "threads": args.threads,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    write_json(args.output, result)
    write_new_text(args.generation_markdown, generation_markdown(result))
    print(args.output)
    print(args.generation_markdown)
    return 0


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
    if args.generation_suite is not None or args.generation_markdown is not None:
        return run_generation(args)

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
