#!/usr/bin/env python3
"""Capture a small, real Qwen3.5 MoE router fixture from local HF code.

This is deliberately narrower than ``architecture_hf_trace.py``: it reads the
actual BF16 router tensor from one safetensors shard, invokes the installed
``Qwen3_5MoeTopKRouter`` class, and writes raw inputs/expected top-k outputs
for the independent Rust MoE runtime verifier. It never loads a full 35B model
or an uLLM package.

The fixture has a normal real-weight case and an all-zero exact-tie case. The
latter records what this particular PyTorch build happens to return, but is
not treated as a portable semantic contract because PyTorch documents top-k
tie ordering as unstable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


class FixtureError(RuntimeError):
    """A short, deterministic fixture-generation failure."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--grouped-rows", type=int, default=37)
    parser.add_argument("--grouped-cols", type=int, default=71)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array, dtype: str) -> str:
    import numpy as np

    value = np.ascontiguousarray(array, dtype=dtype)
    return hashlib.sha256(value.tobytes()).hexdigest()


def write_array(path: Path, array, dtype: str) -> None:
    import numpy as np

    value = np.ascontiguousarray(array, dtype=dtype)
    value.tofile(path)


def write_fixture(
    directory: Path,
    hidden,
    router_weight,
    router_weight_raw_bf16,
    selected_scores,
    selected_indices,
    router_logits,
    *,
    tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    directory.joinpath("shape.txt").write_text(
        f"{tokens} {hidden_size} {num_experts} {top_k}\n", encoding="utf-8"
    )
    write_array(directory / "hidden.f32", hidden, "<f4")
    write_array(directory / "router.f32", router_weight, "<f4")
    write_array(directory / "router.bf16", router_weight_raw_bf16, "<u2")
    write_array(directory / "expected_scores.f32", selected_scores, "<f4")
    write_array(directory / "expected_indices.i32", selected_indices, "<i4")
    write_array(directory / "router_logits.f32", router_logits, "<f4")


def write_grouped_gemm_fixture(
    directory: Path,
    hidden,
    gate_up_slice,
    source_expert_ids,
    *,
    rows: int,
    cols: int,
) -> dict[str, object]:
    """Write a compact real 3-D expert slice for ABI layout validation.

    The local expert axis is deliberately remapped to `[0, 1]`; metadata
    preserves the original checkpoint expert IDs. Assignment IDs visit both
    local experts out of source order, exercising the `[expert,row,column]`
    addressing used by grouped GEMM without copying a whole 1.5-GiB layer.
    """
    import torch

    if tuple(gate_up_slice.shape) != (2, rows, cols):
        raise FixtureError(
            f"grouped slice shape {tuple(gate_up_slice.shape)} != {(2, rows, cols)}"
        )
    assignment_count = min(int(hidden.shape[0]), 3)
    assignment_ids = torch.tensor([1, 0, 1][:assignment_count], dtype=torch.int32)
    input_rows = hidden[:assignment_count, :cols].to(dtype=torch.float32).contiguous()
    expected = torch.bmm(
        gate_up_slice[assignment_ids.to(dtype=torch.long)].to(dtype=torch.float32),
        input_rows.unsqueeze(-1),
    ).squeeze(-1)
    directory.mkdir(parents=True, exist_ok=False)
    directory.joinpath("shape.txt").write_text(
        f"{assignment_count} 2 {rows} {cols}\n", encoding="utf-8"
    )
    write_array(directory / "weights.bf16", gate_up_slice.view(torch.uint16).numpy(), "<u2")
    write_array(directory / "expert_ids.i32", assignment_ids.numpy(), "<i4")
    write_array(directory / "input.f32", input_rows.numpy(), "<f4")
    write_array(directory / "expected.f32", expected.numpy(), "<f4")
    return {
        "source_expert_ids": [int(value) for value in source_expert_ids],
        "local_assignment_expert_ids": assignment_ids.tolist(),
        "shape": [assignment_count, 2, rows, cols],
        "weight_dtype": str(gate_up_slice.dtype),
        "weights_raw_bf16_sha256": array_sha256(gate_up_slice.view(torch.uint16).numpy(), "<u2"),
        "expected_f32_sha256": array_sha256(expected.numpy(), "<f4"),
    }


def main() -> int:
    args = parse_args()
    if args.layer < 0:
        raise FixtureError("--layer must be non-negative")
    if args.tokens < 1 or args.tokens > 32:
        raise FixtureError("--tokens must be in 1..32")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FixtureError(f"refusing to overwrite existing output {output}")
    model_dir = args.model_dir.expanduser().resolve()
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FixtureError("model directory must contain config.json and model.safetensors.index.json")

    try:
        import numpy as np
        import torch
        import transformers
        from safetensors import safe_open
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter
    except Exception as error:  # pragma: no cover - environment dependency
        raise FixtureError(f"local Python dependencies are unavailable: {error}") from error

    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise FixtureError("config has no text_config object")
    if config.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"]:
        raise FixtureError(f"unexpected architecture {config.get('architectures')!r}")
    hidden_size = int(text_config["hidden_size"])
    num_experts = int(text_config["num_experts"])
    top_k = int(text_config["num_experts_per_tok"])
    moe_intermediate_size = int(text_config["moe_intermediate_size"])
    if args.grouped_rows < 1 or args.grouped_rows > 2 * moe_intermediate_size:
        raise FixtureError("--grouped-rows is outside the gate/up row range")
    if args.grouped_cols < 1 or args.grouped_cols > hidden_size:
        raise FixtureError("--grouped-cols is outside the hidden-state column range")
    weight_name = f"model.language_model.layers.{args.layer}.mlp.gate.weight"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or weight_name not in weight_map:
        raise FixtureError(f"missing router tensor {weight_name}")
    shard = model_dir / str(weight_map[weight_name])

    with safe_open(shard, framework="pt", device="cpu") as handle:
        router_weight = handle.get_tensor(weight_name)
    if tuple(router_weight.shape) != (num_experts, hidden_size):
        raise FixtureError(
            f"router shape {tuple(router_weight.shape)} != {(num_experts, hidden_size)}"
        )
    if router_weight.dtype != torch.bfloat16:
        raise FixtureError(f"router dtype {router_weight.dtype} is not BF16")

    router_config = SimpleNamespace(
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        hidden_size=hidden_size,
    )
    router = Qwen3_5MoeTopKRouter(router_config).to(dtype=torch.bfloat16, device="cpu")
    with torch.no_grad():
        router.weight.copy_(router_weight)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    hidden = torch.randn((args.tokens, hidden_size), generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    with torch.no_grad():
        router_logits, selected_scores, selected_indices = router(hidden)

    tie_router = Qwen3_5MoeTopKRouter(router_config).to(dtype=torch.bfloat16, device="cpu")
    with torch.no_grad():
        tie_router.weight.zero_()
        tie_logits, tie_scores, tie_indices = tie_router(torch.zeros_like(hidden[:1]))

    output.mkdir(parents=True)
    write_fixture(
        output / "real",
        hidden.to(dtype=torch.float32).numpy(),
        router_weight.to(dtype=torch.float32).numpy(),
        router_weight.view(torch.uint16).numpy(),
        selected_scores.to(dtype=torch.float32).numpy(),
        selected_indices.to(dtype=torch.int32).numpy(),
        router_logits.to(dtype=torch.float32).numpy(),
        tokens=args.tokens,
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
    )
    gate_up_name = f"model.language_model.layers.{args.layer}.mlp.experts.gate_up_proj"
    if gate_up_name not in weight_map:
        raise FixtureError(f"missing expert tensor {gate_up_name}")
    source_expert_ids = selected_indices[0, :2].to(dtype=torch.long).tolist()
    with safe_open(model_dir / str(weight_map[gate_up_name]), framework="pt", device="cpu") as handle:
        gate_up = handle.get_slice(gate_up_name)[source_expert_ids]
    grouped_metadata = write_grouped_gemm_fixture(
        output / "expert-grouped-gemm",
        hidden,
        gate_up[:, : args.grouped_rows, : args.grouped_cols].contiguous(),
        source_expert_ids,
        rows=args.grouped_rows,
        cols=args.grouped_cols,
    )
    write_fixture(
        output / "exact-tie",
        torch.zeros_like(hidden[:1], dtype=torch.float32).numpy(),
        tie_router.weight.detach().to(dtype=torch.float32).numpy(),
        tie_router.weight.detach().view(torch.uint16).numpy(),
        tie_scores.to(dtype=torch.float32).numpy(),
        tie_indices.to(dtype=torch.int32).numpy(),
        tie_logits.to(dtype=torch.float32).numpy(),
        tokens=1,
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
    )
    metadata = {
        "schema": "ullm.qwen35_moe_hf_routing_fixture.v1",
        "model_dir": str(model_dir),
        "config_sha256": sha256(config_path),
        "router_weight_name": weight_name,
        "router_weight_shard": shard.name,
        "router_weight_shape": list(router_weight.shape),
        "router_weight_dtype": str(router_weight.dtype),
        "router_weight_raw_bf16_sha256": array_sha256(router_weight.view(torch.uint16).numpy(), "<u2"),
        "tokens": args.tokens,
        "hidden_size": hidden_size,
        "num_experts": num_experts,
        "top_k": top_k,
        "seed": args.seed,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "router_logits_dtype": str(router_logits.dtype),
        "selected_scores_dtype": str(selected_scores.dtype),
        "real_selected_expert_ids": selected_indices.to(dtype=torch.int32).tolist(),
        "real_selected_scores_f32": selected_scores.to(dtype=torch.float32).tolist(),
        "exact_tie_observed_expert_ids": tie_indices.to(dtype=torch.int32).tolist(),
        "expert_grouped_gemm": {
            "source_tensor": gate_up_name,
            "source_tensor_shape": [num_experts, 2 * moe_intermediate_size, hidden_size],
            **grouped_metadata,
        },
        "tie_note": "PyTorch topk tie ordering is not stable; exact-tie is diagnostic only.",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FixtureError as error:
        print(f"qwen35_moe_hf_routing_reference: {error}", file=sys.stderr)
        raise SystemExit(1)
