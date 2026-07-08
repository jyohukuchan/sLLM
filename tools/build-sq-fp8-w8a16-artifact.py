#!/usr/bin/env python3
"""Build an FP8 W8A16 SQ candidate artifact from a safetensors model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


SCHEMA_VERSION = "sq-fp8-artifact-v0.1"
POLICY_SCHEMA_VERSION = "sq-fp8-policy-v0.1"
DEFAULT_CANDIDATE_ID = "sq-fp8-w8a16-r9700-v0"
FP8_E4M3_MAX = 448.0


@dataclass(frozen=True)
class SourceTensor:
    name: str
    source_file: Path
    dtype: str
    shape: list[int]

    @property
    def elements(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total


@dataclass(frozen=True)
class SelectedTensor:
    source: SourceTensor
    family: str
    payload_file: Path
    scale_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", required=True, type=Path)
    parser.add_argument("--output-artifact", required=True, type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--base-package", type=Path)
    parser.add_argument("--policy-json", type=Path)
    parser.add_argument("--scale-granularity", choices=("row", "row_block", "tensor"))
    parser.add_argument("--scale-block-cols", type=int)
    parser.add_argument("--activation-dtype", default="bf16_or_f32")
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--include-regex", action="append", default=[])
    parser.add_argument("--exclude-regex", action="append", default=[])
    parser.add_argument("--max-tensors", type=int, default=0)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_policy(path: Path) -> dict[str, Any]:
    policy = read_json(path)
    schema_version = policy.get("schema_version")
    if schema_version != POLICY_SCHEMA_VERSION:
        raise SystemExit(
            f"policy schema_version must be {POLICY_SCHEMA_VERSION!r}, got {schema_version!r}"
        )
    return policy


def resolve_policy_args(args: argparse.Namespace) -> dict[str, Any] | None:
    policy = load_policy(args.policy_json) if args.policy_json is not None else None

    if args.candidate_id is None:
        args.candidate_id = (
            str(policy.get("candidate_id"))
            if policy and policy.get("candidate_id")
            else DEFAULT_CANDIDATE_ID
        )

    if policy is not None and not args.include_regex:
        fp8_selection = policy.get("fp8_selection")
        if not isinstance(fp8_selection, dict):
            raise SystemExit("policy fp8_selection must be an object")
        include_regex = fp8_selection.get("include_regex")
        if not isinstance(include_regex, str) or not include_regex:
            raise SystemExit("policy fp8_selection.include_regex must be a non-empty string")
        args.include_regex.append(include_regex)

    policy_scale = policy.get("scale") if policy is not None else None
    if policy_scale is not None and not isinstance(policy_scale, dict):
        raise SystemExit("policy scale must be an object")

    if args.scale_granularity is None:
        args.scale_granularity = (
            str(policy_scale.get("granularity"))
            if policy_scale and policy_scale.get("granularity")
            else "row"
        )
    if args.scale_granularity not in {"row", "row_block", "tensor"}:
        raise SystemExit(f"unsupported scale granularity: {args.scale_granularity}")

    if args.scale_block_cols is None:
        args.scale_block_cols = (
            int(policy_scale.get("block_cols"))
            if policy_scale and policy_scale.get("block_cols") is not None
            else 256
        )
    if args.scale_block_cols <= 0:
        raise SystemExit("scale-block-cols must be positive")

    return policy


def policy_manifest_entry(policy_json: Path, policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": policy.get("schema_version"),
        "policy_json": str(policy_json),
        "policy_id": policy.get("policy_id"),
        "status": policy.get("status"),
        "fp8_selection": policy.get("fp8_selection"),
        "fallback_policy": policy.get("fallback_policy"),
        "prompt_bundle_result": policy.get("prompt_bundle_result"),
    }


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def relative_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_files(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if isinstance(weight_map, dict):
            files = sorted({model_dir / str(value) for value in weight_map.values()})
            return [path for path in files if path.is_file()]
    return sorted(model_dir.glob("*.safetensors"))


def collect_tensors(model_dir: Path) -> list[SourceTensor]:
    tensors: list[SourceTensor] = []
    for source_file in source_files(model_dir):
        with safe_open(source_file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                view = handle.get_slice(name)
                tensors.append(
                    SourceTensor(
                        name=name,
                        source_file=source_file,
                        dtype=str(view.get_dtype()),
                        shape=[int(dim) for dim in view.get_shape()],
                    )
                )
    tensors.sort(key=lambda item: item.name)
    return tensors


def tensor_family(name: str) -> str | None:
    if name == "lm_head.weight":
        return "lm_head"
    if name == "model.language_model.embed_tokens.weight":
        return "embed"
    suffix_map = {
        ".self_attn.q_proj.weight": "attn_q",
        ".self_attn.k_proj.weight": "attn_k",
        ".self_attn.v_proj.weight": "attn_v",
        ".self_attn.o_proj.weight": "attn_o",
        ".mlp.gate_proj.weight": "mlp_gate",
        ".mlp.up_proj.weight": "mlp_up",
        ".mlp.down_proj.weight": "mlp_down",
        ".linear_attn.in_proj_a.weight": "linear_attn_a",
        ".linear_attn.in_proj_b.weight": "linear_attn_b",
        ".linear_attn.in_proj_qkv.weight": "linear_attn_qkv",
        ".linear_attn.in_proj_z.weight": "linear_attn_z",
        ".linear_attn.out_proj.weight": "linear_attn_out",
    }
    for suffix, family in suffix_map.items():
        if name.endswith(suffix):
            return family
    return None


def is_default_target(tensor: SourceTensor) -> bool:
    if len(tensor.shape) != 2:
        return False
    if tensor_family(tensor.name) is None:
        return False
    if tensor.name.startswith("model.visual"):
        return False
    if tensor.name.startswith("mtp_"):
        return False
    return True


def compile_patterns(values: list[str], label: str) -> list[re.Pattern[str]]:
    patterns = []
    for value in values:
        try:
            patterns.append(re.compile(value))
        except re.error as err:
            raise SystemExit(f"invalid {label} pattern {value!r}: {err}") from err
    return patterns


def selected_tensors(
    tensors: list[SourceTensor],
    output: Path,
    include_patterns: list[re.Pattern[str]],
    exclude_patterns: list[re.Pattern[str]],
    max_tensors: int,
) -> tuple[list[SelectedTensor], list[dict[str, Any]]]:
    selected: list[SelectedTensor] = []
    passthrough: list[dict[str, Any]] = []
    for tensor in tensors:
        included = (
            any(pattern.search(tensor.name) for pattern in include_patterns)
            if include_patterns
            else is_default_target(tensor)
        )
        excluded = any(pattern.search(tensor.name) for pattern in exclude_patterns)
        if max_tensors > 0 and len(selected) >= max_tensors:
            included = False
        if included and not excluded:
            stem = sanitize(tensor.name)
            selected.append(
                SelectedTensor(
                    source=tensor,
                    family=tensor_family(tensor.name) or "custom",
                    payload_file=Path("fp8") / f"{stem}.fp8_e4m3",
                    scale_file=Path("scales") / f"{stem}.scale_f32",
                )
            )
            continue
        reason = "not_selected"
        if excluded:
            reason = "excluded_by_regex"
        elif tensor_family(tensor.name) is None:
            reason = "not_fp8_target_family"
        elif len(tensor.shape) != 2:
            reason = "not_2d_weight"
        passthrough.append(
            {
                "name": tensor.name,
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "elements": tensor.elements,
                "source_file": str(tensor.source_file),
                "reason": reason,
            }
        )
    return selected, passthrough


def dtype_size(dtype: str) -> int:
    return {
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }.get(dtype.upper(), 0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_selected_tensor(
    tensor: SelectedTensor,
    output: Path,
    scale_granularity: str,
    scale_block_cols: int,
    row_chunk: int,
) -> dict[str, Any]:
    if not hasattr(torch, "float8_e4m3fn"):
        raise SystemExit("this PyTorch build does not provide torch.float8_e4m3fn")
    if row_chunk <= 0:
        raise SystemExit("row-chunk must be positive")
    if scale_block_cols <= 0:
        raise SystemExit("scale-block-cols must be positive")
    source = tensor.source
    if len(source.shape) != 2:
        raise SystemExit(f"FP8 target must be 2D: {source.name}")
    rows, cols = source.shape
    payload_path = output / tensor.payload_file
    scale_path = output / tensor.scale_file
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    scale_path.parent.mkdir(parents=True, exist_ok=True)

    with safe_open(source.source_file, framework="pt", device="cpu") as handle:
        view = handle.get_slice(source.name)
        with payload_path.open("wb") as payload_handle, scale_path.open("wb") as scale_handle:
            if scale_granularity == "tensor":
                full = view[:].to(torch.float32)
                scale = full.abs().max() / FP8_E4M3_MAX
                scale = torch.where(scale > 0, scale, torch.tensor(1.0, dtype=torch.float32))
                normalized = (full / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
                encoded = normalized.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
                payload_handle.write(encoded.cpu().numpy().tobytes())
                scale_handle.write(scale.reshape(1).cpu().numpy().astype("float32").tobytes())
            elif scale_granularity == "row_block":
                for start in range(0, rows, row_chunk):
                    end = min(start + row_chunk, rows)
                    chunk = view[start:end].to(torch.float32)
                    encoded_blocks = []
                    scale_blocks = []
                    for col_start in range(0, cols, scale_block_cols):
                        col_end = min(col_start + scale_block_cols, cols)
                        block = chunk[:, col_start:col_end]
                        scale = block.abs().amax(dim=1) / FP8_E4M3_MAX
                        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                        normalized = (block / scale[:, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
                        encoded_blocks.append(
                            normalized.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
                        )
                        scale_blocks.append(scale)
                    encoded = torch.cat(encoded_blocks, dim=1).contiguous()
                    scales = torch.stack(scale_blocks, dim=1).contiguous()
                    payload_handle.write(encoded.cpu().numpy().tobytes())
                    scale_handle.write(scales.cpu().numpy().astype("float32").tobytes())
            else:
                for start in range(0, rows, row_chunk):
                    end = min(start + row_chunk, rows)
                    chunk = view[start:end].to(torch.float32)
                    scale = chunk.abs().amax(dim=1) / FP8_E4M3_MAX
                    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                    normalized = (chunk / scale[:, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
                    encoded = normalized.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
                    payload_handle.write(encoded.cpu().numpy().tobytes())
                    scale_handle.write(scale.contiguous().cpu().numpy().astype("float32").tobytes())

    payload_bytes = payload_path.stat().st_size
    scale_bytes = scale_path.stat().st_size
    return {
        "payload_bytes": payload_bytes,
        "scale_bytes": scale_bytes,
        "payload_sha256": file_sha256(payload_path),
        "scale_sha256": file_sha256(scale_path),
    }


def tensor_manifest_entry(
    tensor: SelectedTensor,
    output: Path,
    scale_granularity: str,
    scale_block_cols: int,
    metadata_only: bool,
) -> dict[str, Any]:
    source = tensor.source
    rows = source.shape[0] if len(source.shape) >= 1 else 1
    cols = source.shape[1] if len(source.shape) >= 2 else 1
    if scale_granularity == "tensor":
        scale_elements = 1
    elif scale_granularity == "row_block":
        scale_elements = rows * ((cols + scale_block_cols - 1) // scale_block_cols)
    else:
        scale_elements = rows
    payload_bytes = source.elements
    scale_bytes = scale_elements * 4
    entry = {
        "name": source.name,
        "family": tensor.family,
        "source_dtype": source.dtype,
        "shape": source.shape,
        "elements": source.elements,
        "source_file": str(source.source_file),
        "payload_dtype": "fp8_e4m3",
        "payload_file": relative_path(tensor.payload_file),
        "payload_bytes": payload_bytes,
        "scale_granularity": scale_granularity,
        "scale_dtype": "f32",
        "scale_file": relative_path(tensor.scale_file),
        "scale_elements": scale_elements,
        "scale_bytes": scale_bytes,
    }
    if scale_granularity == "row_block":
        entry["scale_block_cols"] = scale_block_cols
    if not metadata_only:
        payload_path = output / tensor.payload_file
        scale_path = output / tensor.scale_file
        entry["payload_bytes"] = payload_path.stat().st_size
        entry["scale_bytes"] = scale_path.stat().st_size
        entry["payload_sha256"] = file_sha256(payload_path)
        entry["scale_sha256"] = file_sha256(scale_path)
    return entry


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.source_model_dir
    if not model_dir.is_dir():
        raise SystemExit(f"source model dir does not exist: {model_dir}")
    tensors = collect_tensors(model_dir)
    include_patterns = compile_patterns(args.include_regex, "include-regex")
    exclude_patterns = compile_patterns(args.exclude_regex, "exclude-regex")
    selected, passthrough = selected_tensors(
        tensors,
        args.output_artifact,
        include_patterns,
        exclude_patterns,
        args.max_tensors,
    )

    if not args.dry_run:
        if args.output_artifact.exists():
            if not args.overwrite:
                raise SystemExit(f"output artifact already exists: {args.output_artifact}")
            shutil.rmtree(args.output_artifact)
        args.output_artifact.mkdir(parents=True)

    encoded_by_name: dict[str, dict[str, Any]] = {}
    if not args.metadata_only and not args.dry_run:
        for tensor in selected:
            encoded_by_name[tensor.source.name] = encode_selected_tensor(
                tensor,
                args.output_artifact,
                args.scale_granularity,
                args.scale_block_cols,
                args.row_chunk,
            )

    fp8_entries = []
    for tensor in selected:
        entry = tensor_manifest_entry(
            tensor,
            args.output_artifact,
            args.scale_granularity,
            args.scale_block_cols,
            args.metadata_only or args.dry_run,
        )
        entry.update(encoded_by_name.get(tensor.source.name, {}))
        fp8_entries.append(entry)

    fp8_payload_bytes = sum(int(entry["payload_bytes"]) for entry in fp8_entries)
    fp8_scale_bytes = sum(int(entry["scale_bytes"]) for entry in fp8_entries)
    passthrough_bytes_estimate = sum(
        int(item["elements"]) * dtype_size(str(item["dtype"])) for item in passthrough
    )
    largest_materialized = max(
        (int(entry["elements"]) * 4 for entry in fp8_entries),
        default=0,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "candidate": {
            "id": args.candidate_id,
            "weight_payload_dtype": "fp8_e4m3",
            "activation_dtype": args.activation_dtype,
            "scale_granularity": args.scale_granularity,
            "scale_dtype": "f32",
        },
        "source": {
            "model_dir": str(model_dir),
            "base_package": str(args.base_package) if args.base_package else None,
        },
        "storage": {
            "fp8_tensor_count": len(fp8_entries),
            "passthrough_tensor_count": len(passthrough),
            "fp8_payload_bytes": fp8_payload_bytes,
            "fp8_scale_bytes": fp8_scale_bytes,
            "passthrough_source_bytes_estimate": passthrough_bytes_estimate,
            "compact_resident_bytes_estimate": fp8_payload_bytes
            + fp8_scale_bytes
            + passthrough_bytes_estimate,
            "materialized_working_set_bytes_estimate": largest_materialized,
        },
        "fp8_tensors": fp8_entries,
        "passthrough_tensors": passthrough,
        "notes": [
            "This artifact is the first FP8 W8A16 SQ candidate payload/metadata path.",
            "Runtime execution support is staged after metadata and payload generation are verified.",
        ],
    }
    if getattr(args, "policy_payload", None) is not None and args.policy_json is not None:
        manifest["policy"] = policy_manifest_entry(args.policy_json, args.policy_payload)
    return manifest


def main() -> int:
    args = parse_args()
    args.policy_payload = resolve_policy_args(args)
    manifest = build_manifest(args)
    if args.summary_json is not None and not args.dry_run:
        write_json(args.summary_json, manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    write_json(args.output_artifact / "sq_manifest.json", manifest)
    print(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "artifact": str(args.output_artifact),
        "fp8_tensor_count": manifest["storage"]["fp8_tensor_count"],
        "passthrough_tensor_count": manifest["storage"]["passthrough_tensor_count"],
        "compact_resident_bytes_estimate": manifest["storage"]["compact_resident_bytes_estimate"],
        "metadata_only": args.metadata_only,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
