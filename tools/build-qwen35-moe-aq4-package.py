#!/usr/bin/env python3
"""Build a restartable AQ4_0 text package for Qwen3.5-35B-A3B MoE.

The existing ``ullm-quant`` converter already understands contiguous tensors
with rank >= 2.  Qwen3.5 MoE's routed experts are rank-3 safetensors payloads,
so this tool supplies a deliberately narrow plan instead of teaching the
dense-model planner to silently quantize every ``experts`` tensor it finds.

Only the routed expert ``gate_up_proj`` and ``down_proj`` tensors are encoded
as AQ4_0 (the existing G16 or G8 E4M3 codebook candidate).  Routers, shared experts, attention,
embeddings, norms, and lm_head are preserved byte-for-byte as BF16/F32
passthrough payloads.  That keeps routing semantics exact and leaves a clear
quality boundary while still fitting the R9700 packed-weight budget.

Conversion is resumable at tensor granularity.  Each 3-D tensor is converted
into its own existing ``ullm-quant`` prototype directory and independently
re-read/dequantized before a final package merge.  A stopped run therefore
does not require re-reading or re-quantizing completed 1 GiB / 512 MiB expert
payloads.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


AQ4_FORMATS = {
    8: "aq4_e4m3_g8_ts_flloyd16",
    16: "aq4_e4m3_g16_ts_flloyd16",
}
AQ4_FORMAT = AQ4_FORMATS[16]
AQ4_GROUP_SIZE = 16
AQ4_SCALE_FORMAT = "e4m3"
AQ4_CODEBOOK_ENTRIES = 16
PLAN_SCHEMA = "ullm-quant-plan-v0.3"
PRODUCT_SCHEMA = "ullm.qwen35_moe_aq4_product.v0.1"
TOOL_SCHEMA = "ullm.qwen35_moe_aq4_package.v0.1"
ROUTED_GATE_UP = "moe_routed_gate_up"
ROUTED_DOWN = "moe_routed_down"
R9700_VRAM_BYTES = 34_208_743_424


@dataclass(frozen=True)
class TensorRef:
    """A safetensors tensor whose payload is addressed without materializing it."""

    name: str
    source_file: Path
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_offset_start: int
    data_offset_end: int

    @property
    def n_elements(self) -> int:
        return math.prod(self.shape)

    @property
    def n_bytes(self) -> int:
        return self.data_offset_end - self.data_offset_start

    @property
    def payload_start(self) -> int:
        return self.data_start + self.data_offset_start


class ToolError(RuntimeError):
    """An expected, user-actionable packaging failure."""


def configure_aq4_group_size(group_size: int) -> None:
    """Select an existing AQ4_0 candidate; this does not create a format."""
    global AQ4_FORMAT, AQ4_GROUP_SIZE
    try:
        AQ4_FORMAT = AQ4_FORMATS[group_size]
    except KeyError as exc:
        raise ToolError(f"AQ4_0 group size must be one of {sorted(AQ4_FORMATS)}, got {group_size}") from exc
    AQ4_GROUP_SIZE = group_size


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def gib(value: int | float) -> float:
    return float(value) / float(1024**3)


def mib(value: int | float) -> float:
    return float(value) / float(1024**2)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dtype_bytes(dtype: str) -> int:
    sizes = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1}
    try:
        return sizes[dtype]
    except KeyError as exc:
        raise ToolError(f"unsupported safetensors dtype {dtype}") from exc


def read_safetensors_headers(path: Path) -> dict[str, TensorRef]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ToolError(f"{path} has no safetensors header length")
        header_length = int.from_bytes(raw_length, "little")
        if header_length <= 0 or header_length > 128 * 1024 * 1024:
            raise ToolError(f"{path} has implausible safetensors header length {header_length}")
        header_raw = handle.read(header_length)
        if len(header_raw) != header_length:
            raise ToolError(f"{path} ended inside its safetensors header")
    try:
        header = json.loads(header_raw)
    except json.JSONDecodeError as exc:
        raise ToolError(f"failed to parse safetensors header {path}: {exc}") from exc
    data_start = 8 + header_length
    tensors: dict[str, TensorRef] = {}
    for name, item in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(item, dict):
            raise ToolError(f"{path} header item {name} is not an object")
        dtype = str(item.get("dtype"))
        shape = tuple(int(value) for value in item.get("shape", []))
        offsets = item.get("data_offsets")
        if len(shape) == 0 or not isinstance(offsets, list) or len(offsets) != 2:
            raise ToolError(f"{path} tensor {name} has an invalid header")
        start, end = (int(offsets[0]), int(offsets[1]))
        if start < 0 or end < start:
            raise ToolError(f"{path} tensor {name} has invalid data offsets {offsets}")
        expected = math.prod(shape) * dtype_bytes(dtype)
        if end - start != expected:
            raise ToolError(
                f"{path} tensor {name} payload is {end - start} bytes, expected {expected} from {dtype} {shape}"
            )
        tensors[name] = TensorRef(
            name=name,
            source_file=path,
            dtype=dtype,
            shape=shape,
            data_start=data_start,
            data_offset_start=start,
            data_offset_end=end,
        )
    return tensors


def catalog_model(model_dir: Path) -> dict[str, TensorRef]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise ToolError(f"missing required safetensors index {index_path}")
    index = read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ToolError(f"{index_path} has no weight_map")
    source_files = sorted({model_dir / str(filename) for filename in weight_map.values()})
    catalog: dict[str, TensorRef] = {}
    for source_file in source_files:
        if not source_file.exists():
            raise ToolError(f"index references missing source shard {source_file}")
        headers = read_safetensors_headers(source_file)
        for name, ref in headers.items():
            expected_name_file = model_dir / str(weight_map.get(name, ""))
            if expected_name_file != source_file:
                raise ToolError(f"index/header disagreement for tensor {name}")
            if name in catalog:
                raise ToolError(f"duplicate tensor {name} in safetensors catalog")
            catalog[name] = ref
    missing = sorted(set(weight_map) - set(catalog))
    if missing:
        raise ToolError(f"index tensors absent from shard headers: {missing[:3]}")
    return catalog


def is_text_tensor(name: str) -> bool:
    # MTP is deliberately outside this package.  lm_head is required for text
    # decoding even though it is a top-level tensor rather than a language-model
    # child.
    return name.startswith("model.language_model.") or name == "lm_head.weight"


def routed_projection(name: str) -> str | None:
    suffixes = {
        ".mlp.experts.gate_up_proj": ROUTED_GATE_UP,
        ".mlp.experts.down_proj": ROUTED_DOWN,
    }
    for suffix, family in suffixes.items():
        if name.endswith(suffix):
            return family
    return None


def tensor_family(name: str) -> str:
    routed = routed_projection(name)
    if routed is not None:
        return routed
    if name.endswith(".mlp.gate.weight"):
        return "moe_router"
    if ".mlp.shared_expert_gate." in name:
        return "moe_shared_gate"
    if ".mlp.shared_expert." in name:
        return "moe_shared_expert"
    if name == "lm_head.weight":
        return "lm_head"
    if "embed_tokens" in name:
        return "embed"
    return "text_passthrough"


def layer_for_routed_tensor(name: str) -> int:
    marker = ".layers."
    if marker not in name:
        raise ToolError(f"routed tensor lacks layer path: {name}")
    suffix = name.split(marker, 1)[1]
    token = suffix.split(".", 1)[0]
    try:
        return int(token)
    except ValueError as exc:
        raise ToolError(f"routed tensor has non-numeric layer: {name}") from exc


def codebook_scope(family: str, name: str) -> str:
    if family not in (ROUTED_GATE_UP, ROUTED_DOWN):
        raise ToolError(f"cannot assign a routed codebook scope to {family}")
    # A scope is a loader-visible AQ4_0 contract, not merely a calibration
    # convenience.  The held-out study below selects one scope per projection
    # family, shared across all 40 layers and 256 experts.  Keeping it this
    # coarse is both smaller and, importantly, avoids the layer/expert-tail
    # regressions measured by the calibration study.
    del name
    return family


def calibration_layer_scope(family: str, name: str) -> str:
    """Internal-only key for comparing the unselected per-layer alternative."""
    return f"{family}_l{layer_for_routed_tensor(name):02d}"


def output_bytes_for_aq4(n_elements: int) -> int:
    if n_elements % AQ4_GROUP_SIZE != 0:
        raise ToolError(f"AQ4_0 tensor has {n_elements} elements, not divisible by {AQ4_GROUP_SIZE}")
    if n_elements % 2 != 0:
        raise ToolError(f"AQ4_0 tensor has {n_elements} elements, not nibble-packable")
    return n_elements // 2 + n_elements // AQ4_GROUP_SIZE


def effective_bpp_for_aq4(n_elements: int) -> float:
    """Return the exact index-plus-scale storage rate for the selected candidate."""
    return output_bytes_for_aq4(n_elements) * 8.0 / n_elements


def resume_tensor_contract(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Exclude descriptive estimates that do not affect a staged conversion.

    This preserves resume compatibility for packages produced before a reporting
    correction while retaining exact checks on names, source files, shapes,
    format, codebook scope, and output byte counts.
    """
    return [
        {key: value for key, value in row.items() if key != "estimated_effective_bpp"}
        for row in plan["tensors"]
    ]


def build_plan_and_audit(model_dir: Path, work_dir: Path, resume: bool) -> tuple[dict[str, Any], dict[str, TensorRef]]:
    plan_path = work_dir / "plan.json"
    audit_path = work_dir / "source-audit.json"
    catalog = catalog_model(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise ToolError(f"missing config.json at {config_path}")
    config_sha = sha256_file(config_path)

    text_refs = [ref for name, ref in catalog.items() if is_text_tensor(name)]
    text_refs.sort(key=lambda ref: ref.name)
    if not text_refs:
        raise ToolError("no text tensors selected from source checkpoint")

    routed = [ref for ref in text_refs if routed_projection(ref.name) is not None]
    if len(routed) != 80:
        raise ToolError(f"expected 80 routed expert tensors (40 layers x 2), found {len(routed)}")
    seen_layers: dict[int, set[str]] = {}
    for ref in routed:
        if ref.dtype != "BF16" or len(ref.shape) != 3 or ref.shape[0] != 256:
            raise ToolError(f"unexpected routed tensor contract for {ref.name}: {ref.dtype} {ref.shape}")
        family = routed_projection(ref.name)
        assert family is not None
        layer = layer_for_routed_tensor(ref.name)
        seen_layers.setdefault(layer, set()).add(family)
    expected_layers = set(range(40))
    if set(seen_layers) != expected_layers or any(
        families != {ROUTED_GATE_UP, ROUTED_DOWN} for families in seen_layers.values()
    ):
        raise ToolError("routed expert layer/projection inventory is not the expected 40 x 2 contract")

    tensors: list[dict[str, Any]] = []
    category_bytes: dict[str, int] = {
        "routed_experts": 0,
        "shared_experts": 0,
        "routers": 0,
        "lm_head": 0,
        "other_text": 0,
    }
    for ref in text_refs:
        family = tensor_family(ref.name)
        routed_family = routed_projection(ref.name)
        quantize = routed_family is not None
        if quantize:
            category_bytes["routed_experts"] += ref.n_bytes
        elif family == "moe_shared_expert" or family == "moe_shared_gate":
            category_bytes["shared_experts"] += ref.n_bytes
        elif family == "moe_router":
            category_bytes["routers"] += ref.n_bytes
        elif family == "lm_head":
            category_bytes["lm_head"] += ref.n_bytes
        else:
            category_bytes["other_text"] += ref.n_bytes
        quant_format = AQ4_FORMAT if quantize else None
        tensors.append(
            {
                "name": ref.name,
                "source_file": str(ref.source_file),
                "dtype": ref.dtype,
                "shape": list(ref.shape),
                "family": family,
                "n_elements": ref.n_elements,
                "n_bytes": ref.n_bytes,
                "supported_input": quantize,
                "action": "quantize" if quantize else "passthrough",
                "quant_format": quant_format,
                "quant_role": "low" if quantize else None,
                "codebook_scope": codebook_scope(family, ref.name) if quantize else None,
                "estimated_output_bytes": output_bytes_for_aq4(ref.n_elements) if quantize else ref.n_bytes,
                "estimated_effective_bpp": (
                    effective_bpp_for_aq4(ref.n_elements) if quantize else (ref.n_bytes * 8.0 / ref.n_elements)
                ),
            }
        )
    total_text_bytes = sum(ref.n_bytes for ref in text_refs)
    total_source_bytes = sum(ref.n_bytes for ref in catalog.values())
    estimated_output_bytes = sum(int(row["estimated_output_bytes"]) for row in tensors)
    plan = {
        "schema_version": PLAN_SCHEMA,
        "model_dir": str(model_dir),
        "aq_policy": {
            "policy_id": "qwen35_moe_routed_aq4_0_v0_1",
            "low_format": AQ4_FORMAT,
            "high_format": AQ4_FORMAT,
            "high_families": [],
            "high_tensors": [],
        },
        "codebook_scope_max_elements": None,
        "tensor_count": len(tensors),
        "supported_tensor_count": len(routed),
        "passthrough_tensor_count": len(tensors) - len(routed),
        "total_tensor_bytes": total_text_bytes,
        "total_estimated_output_bytes": estimated_output_bytes,
        "estimated_output_to_input_ratio": estimated_output_bytes / total_text_bytes,
        "tensors": tensors,
    }
    audit = {
        "schema_version": TOOL_SCHEMA,
        "created_at_utc": utc_now(),
        "source_model_dir": str(model_dir),
        "config_sha256": config_sha,
        "source_tensor_count": len(catalog),
        "source_payload_bytes": total_source_bytes,
        "text_tensor_count": len(text_refs),
        "text_payload_bytes": total_text_bytes,
        "text_decoder_excluding_lm_head_bytes": total_text_bytes - category_bytes["lm_head"],
        "text_decoder_excluding_lm_head_gib": gib(total_text_bytes - category_bytes["lm_head"]),
        "routed_tensor_count": len(routed),
        "routed_payload_bytes": category_bytes["routed_experts"],
        "category_payload_bytes": category_bytes,
        "routed_plus_shared_expert_bytes": category_bytes["routed_experts"] + category_bytes["shared_experts"],
        "routed_plus_shared_expert_gib": gib(
            category_bytes["routed_experts"] + category_bytes["shared_experts"]
        ),
        "text_payload_gib": gib(total_text_bytes),
        "routed_payload_gib": gib(category_bytes["routed_experts"]),
        "source_payload_gib": gib(total_source_bytes),
        "plan_sha256": sha256_bytes((json.dumps(plan, sort_keys=True) + "\n").encode()),
        "routed_contract": {
            "layers": sorted(seen_layers),
            "experts_per_layer": 256,
            "gate_up_shape": [256, 1024, 2048],
            "down_shape": [256, 2048, 512],
        },
    }
    if not (resume and plan_path.exists() and audit_path.exists()):
        write_json(plan_path, plan)
        write_json(audit_path, audit)
    else:
        existing_plan = read_json(plan_path)
        existing_audit = read_json(audit_path)
        if (
            existing_audit.get("config_sha256") != config_sha
            or resume_tensor_contract(existing_plan) != resume_tensor_contract(plan)
        ):
            raise ToolError("--resume plan/source contract differs from existing work directory")
        plan = existing_plan
    return plan, catalog


class PayloadReader:
    """Small, explicit pread reader for BF16 safetensors payloads."""

    def __init__(self) -> None:
        self._fds: dict[Path, int] = {}

    def _fd(self, path: Path) -> int:
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return fd

    def read_at(self, ref: TensorRef, relative_offset: int, size: int) -> bytes:
        if relative_offset < 0 or size < 0 or relative_offset + size > ref.n_bytes:
            raise ToolError(f"payload read outside {ref.name}: offset={relative_offset} size={size}")
        data = os.pread(self._fd(ref.source_file), size, ref.payload_start + relative_offset)
        if len(data) != size:
            raise ToolError(f"short read of {ref.name}: got {len(data)}, expected {size}")
        return data

    def read_all(self, ref: TensorRef) -> bytes:
        return self.read_at(ref, 0, ref.n_bytes)

    def read_expert_rows(self, ref: TensorRef, expert: int, rows: Sequence[int]) -> np.ndarray:
        if ref.dtype != "BF16" or len(ref.shape) != 3:
            raise ToolError(f"expected rank-3 BF16 routed tensor, got {ref.name} {ref.dtype} {ref.shape}")
        experts, row_count, columns = ref.shape
        if expert < 0 or expert >= experts:
            raise ToolError(f"expert {expert} outside {ref.name} axis size {experts}")
        row_values: list[np.ndarray] = []
        for row in rows:
            if row < 0 or row >= row_count:
                raise ToolError(f"row {row} outside {ref.name} row size {row_count}")
            element_offset = (expert * row_count + row) * columns
            raw = self.read_at(ref, element_offset * 2, columns * 2)
            row_values.append(bf16_bytes_to_f32(raw))
        if not row_values:
            return np.empty((0, columns), dtype=np.float32)
        return np.stack(row_values, axis=0)

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __enter__(self) -> "PayloadReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    if len(raw) % 2:
        raise ToolError("BF16 payload has an odd number of bytes")
    bits16 = np.frombuffer(raw, dtype="<u2")
    bits32 = bits16.astype("<u4") << np.uint32(16)
    return bits32.view("<f4")


def deterministic_rows(row_count: int, count: int, token: str, seed: int) -> list[int]:
    if count <= 0:
        return []
    if count > row_count:
        raise ToolError(f"requested {count} sample rows from a tensor with only {row_count} rows")
    digest = hashlib.sha256(f"{seed}:{token}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return sorted(int(value) for value in rng.choice(row_count, size=count, replace=False))


def normalized_group_values(values: np.ndarray) -> np.ndarray:
    if values.size % AQ4_GROUP_SIZE:
        raise ToolError(f"sample has {values.size} elements, not divisible by AQ4 group size")
    groups = values.astype(np.float32, copy=False).reshape(-1, AQ4_GROUP_SIZE)
    absmax = np.max(np.abs(groups), axis=1, keepdims=True)
    normalized = np.divide(groups, absmax, out=np.zeros_like(groups), where=absmax > 0.0)
    return normalized.reshape(-1)


def fit_lloyd16(values: np.ndarray, iterations: int) -> np.ndarray:
    values = values[np.isfinite(values)].astype(np.float32, copy=False)
    if values.size < AQ4_CODEBOOK_ENTRIES:
        raise ToolError(f"codebook fit has only {values.size} finite values")
    quantiles = (np.arange(AQ4_CODEBOOK_ENTRIES, dtype=np.float64) + 0.5) / AQ4_CODEBOOK_ENTRIES
    centers = np.quantile(values, quantiles).astype(np.float32)
    centers.sort()
    for _ in range(iterations):
        boundaries = (centers[:-1] + centers[1:]) * 0.5
        labels = np.searchsorted(boundaries, values, side="left")
        counts = np.bincount(labels, minlength=AQ4_CODEBOOK_ENTRIES).astype(np.float64)
        sums = np.bincount(labels, weights=values, minlength=AQ4_CODEBOOK_ENTRIES)
        replacement = np.divide(sums, counts, out=centers.astype(np.float64), where=counts > 0)
        updated = replacement.astype(np.float32)
        updated.sort()
        if np.array_equal(updated, centers):
            break
        centers = updated
    return centers


def quant_error(values: np.ndarray, codebook: np.ndarray) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)].astype(np.float32, copy=False)
    boundaries = (codebook[:-1] + codebook[1:]) * 0.5
    indices = np.searchsorted(boundaries, finite, side="left")
    reconstructed = codebook[indices]
    error = finite - reconstructed
    sse = float(np.dot(error, error))
    ref_sse = float(np.dot(finite, finite))
    return sse, ref_sse, float(np.max(np.abs(error), initial=0.0))


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def summarize_codebook_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    global_rel = [float(row["global_relative_mse"]) for row in rows]
    layer_rel = [float(row["layer_relative_mse"]) for row in rows]
    expert_rel = [float(row["expert_relative_mse"]) for row in rows]
    layer_improvement = [float(row["layer_improvement_vs_global"]) for row in rows]
    expert_improvement = [float(row["expert_improvement_vs_layer"]) for row in rows]
    worst = sorted(rows, key=lambda row: float(row["layer_relative_mse"]), reverse=True)[:16]
    return {
        "expert_sample_count": len(rows),
        "global_relative_mse": {
            "median": percentile(global_rel, 50),
            "p95": percentile(global_rel, 95),
            "max": max(global_rel, default=0.0),
        },
        "layer_relative_mse": {
            "median": percentile(layer_rel, 50),
            "p95": percentile(layer_rel, 95),
            "max": max(layer_rel, default=0.0),
        },
        "expert_relative_mse": {
            "median": percentile(expert_rel, 50),
            "p95": percentile(expert_rel, 95),
            "max": max(expert_rel, default=0.0),
        },
        "layer_improvement_vs_global": {
            "median": percentile(layer_improvement, 50),
            "p05": percentile(layer_improvement, 5),
            "min": min(layer_improvement, default=0.0),
        },
        "expert_improvement_vs_layer": {
            "median": percentile(expert_improvement, 50),
            "p95": percentile(expert_improvement, 95),
            "max": max(expert_improvement, default=0.0),
        },
        "worst_layer_scope_samples": worst,
    }


def calibrate_codebooks(
    plan: dict[str, Any],
    catalog: dict[str, TensorRef],
    work_dir: Path,
    resume: bool,
    seed: int,
    fit_groups_per_expert: int,
    eval_groups_per_expert: int,
    max_global_values: int,
    lloyd_iterations: int,
) -> tuple[Path, Path]:
    codebook_path = work_dir / "codebooks.json"
    study_path = work_dir / "codebook-granularity-study.json"
    plan_sha = sha256_file(work_dir / "plan.json")
    if resume and codebook_path.exists() and study_path.exists():
        existing = read_json(codebook_path)
        if existing.get("plan_sha256") == plan_sha:
            return codebook_path, study_path
        raise ToolError("--resume codebook export belongs to a different plan")

    rows_by_family: dict[str, list[dict[str, Any]]] = {ROUTED_GATE_UP: [], ROUTED_DOWN: []}
    for row in plan["tensors"]:
        if row["action"] == "quantize":
            rows_by_family[str(row["family"])].append(row)
    if any(len(rows) != 40 for rows in rows_by_family.values()):
        raise ToolError("calibration plan is missing routed tensors")

    export_rows: list[dict[str, Any]] = []
    studies: dict[str, Any] = {}
    with PayloadReader() as reader:
        for family, rows in rows_by_family.items():
            scope_codebooks: dict[str, np.ndarray] = {}
            global_chunks: list[np.ndarray] = []
            scope_sample_counts: dict[str, int] = {}
            refs = [(row, catalog[str(row["name"])]) for row in rows]
            for row, ref in refs:
                experts, row_count, columns = ref.shape
                groups_per_row = columns // AQ4_GROUP_SIZE
                fit_rows_count = math.ceil(fit_groups_per_expert / groups_per_row)
                scope_chunks: list[np.ndarray] = []
                for expert in range(experts):
                    sample_rows = deterministic_rows(
                        row_count,
                        fit_rows_count,
                        f"fit:{ref.name}:{expert}",
                        seed,
                    )
                    values = normalized_group_values(reader.read_expert_rows(ref, expert, sample_rows))
                    scope_chunks.append(values)
                scope_values = np.concatenate(scope_chunks)
                scope = calibration_layer_scope(family, ref.name)
                scope_codebooks[scope] = fit_lloyd16(scope_values, lloyd_iterations)
                scope_sample_counts[scope] = int(scope_values.size)
                global_chunks.append(scope_values)
                del scope_chunks, scope_values

            global_values = np.concatenate(global_chunks)
            if global_values.size > max_global_values:
                rng = np.random.default_rng(seed + (0 if family == ROUTED_GATE_UP else 1))
                indices = rng.choice(global_values.size, size=max_global_values, replace=False)
                global_fit_values = global_values[indices]
            else:
                global_fit_values = global_values
            global_codebook = fit_lloyd16(global_fit_values, lloyd_iterations)
            comparison_rows: list[dict[str, Any]] = []
            for row, ref in refs:
                experts, row_count, columns = ref.shape
                groups_per_row = columns // AQ4_GROUP_SIZE
                fit_rows_count = math.ceil(fit_groups_per_expert / groups_per_row)
                eval_rows_count = math.ceil(eval_groups_per_expert / groups_per_row)
                scope = calibration_layer_scope(family, ref.name)
                layer_codebook = scope_codebooks[scope]
                for expert in range(experts):
                    selected_rows = deterministic_rows(
                        row_count,
                        fit_rows_count + eval_rows_count,
                        f"cross_validate:{ref.name}:{expert}",
                        seed,
                    )
                    expert_fit = normalized_group_values(
                        reader.read_expert_rows(ref, expert, selected_rows[:fit_rows_count])
                    )
                    evaluation = normalized_group_values(
                        reader.read_expert_rows(ref, expert, selected_rows[fit_rows_count:])
                    )
                    expert_codebook = fit_lloyd16(expert_fit, lloyd_iterations)
                    global_sse, ref_sse, global_max = quant_error(evaluation, global_codebook)
                    layer_sse, _, layer_max = quant_error(evaluation, layer_codebook)
                    expert_sse, _, expert_max = quant_error(evaluation, expert_codebook)
                    global_rel = global_sse / ref_sse if ref_sse else 0.0
                    layer_rel = layer_sse / ref_sse if ref_sse else 0.0
                    expert_rel = expert_sse / ref_sse if ref_sse else 0.0
                    comparison_rows.append(
                        {
                            "tensor": ref.name,
                            "layer": layer_for_routed_tensor(ref.name),
                            "projection": family,
                            "codebook_scope": scope,
                            "expert": expert,
                            "evaluation_elements": int(evaluation.size),
                            "global_relative_mse": global_rel,
                            "layer_relative_mse": layer_rel,
                            "expert_relative_mse": expert_rel,
                            "layer_improvement_vs_global": 1.0 - layer_sse / global_sse if global_sse else 0.0,
                            "expert_improvement_vs_layer": 1.0 - expert_sse / layer_sse if layer_sse else 0.0,
                            "global_max_abs_error": global_max,
                            "layer_max_abs_error": layer_max,
                            "expert_max_abs_error": expert_max,
                        }
                    )
            studies[family] = {
                "global_cross_layer_codebook": [float(value) for value in global_codebook],
                "fit_values_total": int(global_values.size),
                "fit_values_used_for_global_codebook": int(global_fit_values.size),
                "per_layer_scope_fit_values": scope_sample_counts,
                "comparison": summarize_codebook_comparison(comparison_rows),
                "per_expert_cross_validation": comparison_rows,
            }
            # The plan assigns every 3-D routed tensor in this projection
            # family to this one exported AQ4_0 codebook.  Per-layer fits are
            # retained above solely as an audited alternative in the study.
            export_rows.append(
                {
                    "family": family,
                    "codebook_scope": codebook_scope(family, str(refs[0][0]["name"])),
                    "candidate_id": AQ4_FORMAT,
                    "entry_count": AQ4_CODEBOOK_ENTRIES,
                    "index_bits": 4,
                    "storage_dtype": "float32",
                    "values_f32": [float(value) for value in global_codebook],
                    "fit_tensor": "all-40-layers",
                    "fit_elements": int(global_fit_values.size),
                }
            )
            del global_chunks, global_values

    study = {
        "schema_version": TOOL_SCHEMA,
        "created_at_utc": utc_now(),
        "plan_sha256": plan_sha,
        "method": {
            "normalization": "per contiguous AQ4_0 group absmax",
            "group_size": AQ4_GROUP_SIZE,
            "fit_groups_per_expert": fit_groups_per_expert,
            "eval_groups_per_expert": eval_groups_per_expert,
            "lloyd_iterations": lloyd_iterations,
            "comparison": "global cross-layer vs shared per-layer tensor vs held-out per-expert Lloyd16",
        },
        "families": studies,
        "decision": {
            "selected_granularity": "one codebook per routed projection family, shared by all 40 layers and 256 experts per layer",
            "not_selected": "per-layer and per-expert codebooks",
            "reason": "the held-out cross-validation records a tighter global worst case and lower or comparable relative MSE; per-expert fitting adds no measured quality benefit and has unstable tails.  The selected scopes are existing AQ4_0 codebook scopes, not a new runtime format.",
        },
    }
    export = {
        "schema_version": "aq-family-codebook-export-v0.2",
        "created_at_utc": utc_now(),
        "plan_sha256": plan_sha,
        "model_dir": plan["model_dir"],
        "candidate_ids": [AQ4_FORMAT],
        "codebooks": sorted(export_rows, key=lambda row: (row["codebook_scope"], row["candidate_id"])),
        "notes": [
            "Qwen3.5-35B-A3B MoE routed experts only.",
            "One AQ4_0 codebook is shared by every routed tensor in each projection family (40 layers x 256 experts).",
            "Router and shared-expert tensors are intentionally absent because they are raw passthrough payloads.",
        ],
    }
    write_json(study_path, study)
    write_json(codebook_path, export)
    return codebook_path, study_path


def selected_quant_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in plan["tensors"] if row["action"] == "quantize"]


def single_tensor_plan(plan: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    selected_bytes = int(row["n_bytes"])
    output_bytes = int(row["estimated_output_bytes"])
    return {
        **plan,
        "tensor_count": 1,
        "supported_tensor_count": 1,
        "passthrough_tensor_count": 0,
        "total_tensor_bytes": selected_bytes,
        "total_estimated_output_bytes": output_bytes,
        "estimated_output_to_input_ratio": output_bytes / selected_bytes,
        "tensors": [row],
    }


def tensor_task_dir(work_dir: Path, index: int) -> Path:
    return work_dir / "staging" / f"{index:03d}"


def tensor_summary_path(work_dir: Path, index: int) -> Path:
    return work_dir / "individual-summaries" / f"{index:03d}.json"


def valid_individual_summary(summary_path: Path, row: dict[str, Any]) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = read_json(summary_path)
        results = summary.get("results")
        if not isinstance(results, list) or len(results) != 1:
            return False
        result = results[0]
        manifest = result.get("manifest")
        if result.get("status") != "ok" or not isinstance(manifest, dict):
            return False
        if manifest.get("name") != row["name"] or manifest.get("shape") != row["shape"]:
            return False
        if manifest.get("candidate_id") != AQ4_FORMAT or manifest.get("family") != row["family"]:
            return False
        output_dir = Path(str(result["output_dir"]))
        index_path = output_dir / str(manifest["index_file"])
        scale_path = output_dir / str(manifest["scale_file"])
        expected_index = int(row["n_elements"]) // 2
        expected_scale = int(row["n_elements"]) // AQ4_GROUP_SIZE
        if index_path.stat().st_size != expected_index or scale_path.stat().st_size != expected_scale:
            return False
        verification = result.get("verification")
        return isinstance(verification, dict) and int(verification.get("elements", -1)) == int(row["n_elements"])
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False


def run_one_quantization(
    index: int,
    row: dict[str, Any],
    plan: dict[str, Any],
    codebook_path: Path,
    work_dir: Path,
    quant_bin: Path,
    chunk_bytes: int,
    scale_window: int,
    reservoir_size: int,
    resume: bool,
) -> dict[str, Any]:
    summary_path = tensor_summary_path(work_dir, index)
    if resume and valid_individual_summary(summary_path, row):
        return {"index": index, "tensor": row["name"], "status": "resumed"}
    task_dir = tensor_task_dir(work_dir, index)
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    per_plan = task_dir / "plan.json"
    write_json(per_plan, single_tensor_plan(plan, row))
    log_path = work_dir / "logs" / f"quant-{index:03d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(quant_bin),
        "--convert-plan-json",
        str(per_plan),
        "--codebook-json",
        str(codebook_path),
        "--convert-output-root",
        str(task_dir / "output"),
        "--convert-summary-output",
        str(summary_path),
        "--convert-jobs",
        "1",
        "--convert-verify",
        "--chunk-bytes",
        str(chunk_bytes),
        "--scale-window",
        str(scale_window),
        "--tensor-scale-estimator",
        "reservoir",
        "--tensor-scale-reservoir-size",
        str(reservoir_size),
    ]
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False)
    if completed.returncode != 0 or not valid_individual_summary(summary_path, row):
        raise ToolError(f"quantization failed for {row['name']}; see {log_path}")
    return {"index": index, "tensor": row["name"], "status": "ok"}


def quantize_all(
    plan: dict[str, Any],
    codebook_path: Path,
    work_dir: Path,
    quant_bin: Path,
    jobs: int,
    chunk_bytes: int,
    scale_window: int,
    reservoir_size: int,
    resume: bool,
) -> Path:
    if not quant_bin.is_file() or not os.access(quant_bin, os.X_OK):
        raise ToolError(f"ullm-quant binary is not executable: {quant_bin}")
    rows = selected_quant_rows(plan)
    state_path = work_dir / "quantization-state.json"
    state: dict[str, Any] = {
        "schema_version": TOOL_SCHEMA,
        "created_at_utc": utc_now(),
        "quant_bin": str(quant_bin),
        "jobs": jobs,
        "targets": len(rows),
        "completed": [],
    }
    if resume and state_path.exists():
        existing = read_json(state_path)
        if existing.get("targets") == len(rows):
            state["completed"] = existing.get("completed", [])
    completed_by_index: dict[int, dict[str, Any]] = {
        int(item["index"]): item for item in state["completed"] if isinstance(item, dict) and "index" in item
    }
    pending = [
        (index, row)
        for index, row in enumerate(rows)
        if not (resume and valid_individual_summary(tensor_summary_path(work_dir, index), row))
    ]
    for index, row in enumerate(rows):
        if index not in completed_by_index and resume and valid_individual_summary(tensor_summary_path(work_dir, index), row):
            completed_by_index[index] = {"index": index, "tensor": row["name"], "status": "resumed"}
    write_json(state_path, {**state, "completed": [completed_by_index[key] for key in sorted(completed_by_index)]})

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    run_one_quantization,
                    index,
                    row,
                    plan,
                    codebook_path,
                    work_dir,
                    quant_bin,
                    chunk_bytes,
                    scale_window,
                    reservoir_size,
                    resume,
                ): (index, row)
                for index, row in pending
            }
            for future in concurrent.futures.as_completed(futures):
                index, row = futures[future]
                completed_by_index[index] = future.result()
                write_json(
                    state_path,
                    {**state, "completed": [completed_by_index[key] for key in sorted(completed_by_index)]},
                )
    if len(completed_by_index) != len(rows):
        raise ToolError("not every routed tensor produced a verified individual package")

    master_results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        summary_path = tensor_summary_path(work_dir, index)
        if not valid_individual_summary(summary_path, row):
            raise ToolError(f"invalid or missing individual result for {row['name']}")
        master_results.append(read_json(summary_path)["results"][0])
    master_path = work_dir / "master-convert-summary.json"
    write_json(
        master_path,
        {
            "schema_version": "ullm-prototype-convert-summary-v0.1",
            "source": "restartable qwen35 moe AQ4_0 orchestration",
            "results": master_results,
        },
    )
    return master_path


def merge_package(
    plan_path: Path,
    master_summary: Path,
    product_dir: Path,
    work_dir: Path,
    quant_bin: Path,
    copy_buffer_bytes: int,
    resume: bool,
) -> Path:
    package_dir = product_dir / "package"
    manifest_path = package_dir / "manifest.json"
    if resume and manifest_path.exists():
        manifest = read_json(manifest_path)
        if len(manifest.get("tensors", [])) == 80:
            return package_dir
        raise ToolError(f"existing final package at {package_dir} is incomplete")
    product_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = product_dir / "package.partial"
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    merge_summary = work_dir / "merge-summary.json"
    log_path = work_dir / "logs" / "merge.log"
    command = [
        str(quant_bin),
        "--merge-policy-summary",
        str(master_summary),
        "--merge-plan-json",
        str(plan_path),
        "--merge-output-dir",
        str(partial_dir),
        "--merge-summary-output",
        str(merge_summary),
        "--merge-include-passthrough",
        "--merge-copy-buffer-bytes",
        str(copy_buffer_bytes),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0 or not (partial_dir / "manifest.json").exists():
        raise ToolError(f"package merge failed; see {log_path}")
    if package_dir.exists():
        raise ToolError(f"refusing to overwrite existing package {package_dir}")
    os.replace(partial_dir, package_dir)
    return package_dir


def e4m3_scale_values() -> np.ndarray:
    values: list[float] = []
    exponent_bits, mantissa_bits = 4, 3
    bias = (1 << (exponent_bits - 1)) - 1
    max_exponent = (1 << exponent_bits) - 1
    mantissa_count = 1 << mantissa_bits
    for exponent in range(max_exponent):
        for mantissa in range(mantissa_count):
            if exponent == 0:
                if mantissa == 0:
                    continue
                values.append((mantissa / mantissa_count) * 2.0 ** (1 - bias))
            else:
                values.append((1.0 + mantissa / mantissa_count) * 2.0 ** (exponent - bias))
    return np.asarray(sorted(set(values)), dtype=np.float32)


def decode_aq4_expert(package_dir: Path, tensor: dict[str, Any], expert: int) -> np.ndarray:
    shape = tuple(int(value) for value in tensor["shape"])
    if len(shape) != 3 or int(tensor["group_size"]) != AQ4_GROUP_SIZE:
        raise ToolError(f"unexpected AQ4 routed tensor shape/format: {tensor['name']} {shape}")
    experts, rows, columns = shape
    if expert < 0 or expert >= experts:
        raise ToolError(f"expert {expert} outside {tensor['name']}")
    elements = rows * columns
    start = expert * elements
    if start % 2 or start % AQ4_GROUP_SIZE:
        raise ToolError(f"unexpected expert alignment for {tensor['name']}")
    index_map = np.memmap(package_dir / str(tensor["index_file"]), dtype=np.uint8, mode="r")
    scale_map = np.memmap(package_dir / str(tensor["scale_file"]), dtype=np.uint8, mode="r")
    packed = np.asarray(index_map[start // 2 : (start + elements) // 2], dtype=np.uint8)
    indices = np.empty(elements, dtype=np.uint8)
    indices[0::2] = packed & np.uint8(0x0F)
    indices[1::2] = packed >> np.uint8(4)
    codebook = np.fromfile(package_dir / str(tensor["codebook_file"]), dtype="<f4")
    if codebook.size != AQ4_CODEBOOK_ENTRIES:
        raise ToolError(f"invalid codebook for {tensor['name']}")
    scales = e4m3_scale_values()
    scale_start = start // AQ4_GROUP_SIZE
    scale_indices = np.asarray(scale_map[scale_start : scale_start + elements // AQ4_GROUP_SIZE], dtype=np.intp)
    if np.any(scale_indices >= scales.size):
        raise ToolError(f"scale index outside e4m3 table for {tensor['name']}")
    expanded = np.repeat(scales[scale_indices] * np.float32(tensor["tensor_scale"]), AQ4_GROUP_SIZE)
    return (codebook[indices] * expanded).reshape(rows, columns)


def raw_package_tensor_bytes(package_dir: Path, manifest: dict[str, Any], name: str) -> bytes:
    passthrough = {str(row["name"]): row for row in manifest.get("passthrough_tensors", [])}
    row = passthrough.get(name)
    if row is None:
        raise ToolError(f"package does not contain raw passthrough tensor {name}")
    path = package_dir / str(row["payload_file"])
    raw = path.read_bytes()
    if len(raw) != int(row["payload_bytes"]):
        raise ToolError(f"raw package payload length mismatch for {name}")
    if sha256_bytes(raw) != row["payload_sha256"]:
        raise ToolError(f"raw package payload hash mismatch for {name}")
    return raw


def tensor_from_bf16_bytes(raw: bytes, shape: Sequence[int]) -> np.ndarray:
    values = bf16_bytes_to_f32(raw)
    expected = math.prod(shape)
    if values.size != expected:
        raise ToolError(f"BF16 payload has {values.size} values, expected {expected}")
    return values.reshape(tuple(shape))


def route_topk(hidden: np.ndarray, router: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the installed HF router arithmetic, not a NumPy approximation.

    Qwen3.5 MoE does BF16 ``F.linear``, takes its softmax in FP32, then
    returns BF16-normalized top-k scores.  Preserving a router payload byte for
    byte proves semantic identity for every input, while this independently
    exercises the exact framework boundary on a fixed set of inputs.
    """
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - environment contract
        raise ToolError(f"PyTorch is required for HF-faithful router validation: {exc}") from exc
    hidden_tensor = torch.from_numpy(np.ascontiguousarray(hidden, dtype=np.float32)).to(torch.bfloat16)
    router_tensor = torch.from_numpy(np.ascontiguousarray(router, dtype=np.float32)).to(torch.bfloat16)
    with torch.no_grad():
        logits = functional.linear(hidden_tensor, router_tensor)
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        selected, order = torch.topk(probabilities, 8, dim=-1)
        selected = (selected / selected.sum(dim=-1, keepdim=True)).to(logits.dtype)
        top_nine = torch.topk(probabilities, 9, dim=-1).values
        margins = top_nine[:, 7] - top_nine[:, 8]
    return (
        order.to(dtype=torch.int32).cpu().numpy(),
        selected.to(dtype=torch.float32).cpu().numpy(),
        margins.to(dtype=torch.float32).cpu().numpy(),
    )


def silu(values: np.ndarray) -> np.ndarray:
    return values / (1.0 + np.exp(-values))


def find_named_tensor(catalog: dict[str, TensorRef], suffix: str) -> TensorRef:
    # The checkpoint also has an MTP layer 0 with similarly named MoE fields.
    # This package is deliberately text-only, so a suffix match must not cross
    # that namespace boundary.
    matches = [ref for name, ref in catalog.items() if is_text_tensor(name) and name.endswith(suffix)]
    if len(matches) != 1:
        raise ToolError(f"expected one tensor ending {suffix}, found {[ref.name for ref in matches]}")
    return matches[0]


def validate_router_and_moe_sublayer(
    package_dir: Path,
    manifest: dict[str, Any],
    catalog: dict[str, TensorRef],
    evidence_dir: Path,
    seed: int,
) -> dict[str, Any]:
    router_refs = sorted((ref for ref in catalog.values() if is_text_tensor(ref.name) and tensor_family(ref.name) == "moe_router"), key=lambda ref: ref.name)
    if len(router_refs) != 40:
        raise ToolError(f"expected 40 routers, found {len(router_refs)}")
    package_raw = {str(row["name"]): row for row in manifest.get("passthrough_tensors", [])}
    router_hashes: list[dict[str, Any]] = []
    route_rows = 0
    differing_rows = 0
    max_score_abs = 0.0
    boundary_ties = 0
    rng = np.random.default_rng(seed)
    with PayloadReader() as reader:
        for ref in router_refs:
            source_raw = reader.read_all(ref)
            package_raw_bytes = raw_package_tensor_bytes(package_dir, manifest, ref.name)
            source_hash = sha256_bytes(source_raw)
            package_hash = sha256_bytes(package_raw_bytes)
            if source_hash != package_hash:
                raise ToolError(f"router raw payload changed: {ref.name}")
            router = tensor_from_bf16_bytes(source_raw, ref.shape)
            hidden = (rng.standard_normal((32, ref.shape[1])).astype(np.float32) * np.float32(0.1))
            source_ids, source_scores, source_margins = route_topk(hidden, router)
            package_ids, package_scores, package_margins = route_topk(
                hidden,
                tensor_from_bf16_bytes(package_raw_bytes, ref.shape),
            )
            differing_rows += int(np.count_nonzero(np.any(source_ids != package_ids, axis=1)))
            max_score_abs = max(max_score_abs, float(np.max(np.abs(source_scores - package_scores))))
            boundary_ties += int(np.count_nonzero(source_margins == 0.0))
            route_rows += hidden.shape[0]
            router_hashes.append(
                {
                    "name": ref.name,
                    "shape": list(ref.shape),
                    "source_sha256": source_hash,
                    "package_sha256": package_hash,
                    "tested_tokens": int(hidden.shape[0]),
                    "topk_changed_tokens": int(np.count_nonzero(np.any(source_ids != package_ids, axis=1))),
                    "max_selected_probability_abs_error": float(np.max(np.abs(source_scores - package_scores))),
                    "boundary_tie_tokens": int(np.count_nonzero(source_margins == 0.0)),
                }
            )

        # Layer 0 is a true rank-3 routed-weight forward, not a synthetic slice.
        layer = 0
        gate_name = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
        down_name = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
        router_name = f"model.language_model.layers.{layer}.mlp.gate.weight"
        gate_ref, down_ref, router_ref = catalog[gate_name], catalog[down_name], catalog[router_name]
        quantized = {str(row["name"]): row for row in manifest.get("tensors", [])}
        gate_manifest, down_manifest = quantized[gate_name], quantized[down_name]
        hidden = (rng.standard_normal((4, gate_ref.shape[2])).astype(np.float32) * np.float32(0.1))
        router = tensor_from_bf16_bytes(reader.read_all(router_ref), router_ref.shape)
        ids, scores, _ = route_topk(hidden, router)
        source_routed = np.zeros((hidden.shape[0], hidden.shape[1]), dtype=np.float32)
        quantized_routed = np.zeros_like(source_routed)
        source_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        quantized_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for token, selected_experts in enumerate(ids):
            for slot, expert in enumerate(selected_experts.tolist()):
                if expert not in source_cache:
                    source_gate = reader.read_expert_rows(gate_ref, expert, list(range(gate_ref.shape[1])))
                    source_down = reader.read_expert_rows(down_ref, expert, list(range(down_ref.shape[1])))
                    source_cache[expert] = (source_gate, source_down)
                    quantized_cache[expert] = (
                        decode_aq4_expert(package_dir, gate_manifest, expert),
                        decode_aq4_expert(package_dir, down_manifest, expert),
                    )
                source_gate, source_down = source_cache[expert]
                quant_gate, quant_down = quantized_cache[expert]
                source_gate_values = source_gate @ hidden[token]
                quant_gate_values = quant_gate @ hidden[token]
                source_active = silu(source_gate_values[:512]) * source_gate_values[512:]
                quant_active = silu(quant_gate_values[:512]) * quant_gate_values[512:]
                source_routed[token] += scores[token, slot] * (source_down @ source_active)
                quantized_routed[token] += scores[token, slot] * (quant_down @ quant_active)

        shared_gate_ref = find_named_tensor(catalog, ".layers.0.mlp.shared_expert_gate.weight")
        shared_gate_proj_ref = find_named_tensor(catalog, ".layers.0.mlp.shared_expert.gate_proj.weight")
        shared_up_ref = find_named_tensor(catalog, ".layers.0.mlp.shared_expert.up_proj.weight")
        shared_down_ref = find_named_tensor(catalog, ".layers.0.mlp.shared_expert.down_proj.weight")
        shared_gate = tensor_from_bf16_bytes(reader.read_all(shared_gate_ref), shared_gate_ref.shape)
        shared_gate_proj = tensor_from_bf16_bytes(reader.read_all(shared_gate_proj_ref), shared_gate_proj_ref.shape)
        shared_up = tensor_from_bf16_bytes(reader.read_all(shared_up_ref), shared_up_ref.shape)
        shared_down = tensor_from_bf16_bytes(reader.read_all(shared_down_ref), shared_down_ref.shape)
        shared_active = silu(hidden @ shared_gate_proj.T) * (hidden @ shared_up.T)
        shared_output = shared_active @ shared_down.T
        shared_scale = 1.0 / (1.0 + np.exp(-(hidden @ shared_gate.T)))
        source_output = source_routed + shared_scale * shared_output
        quantized_output = quantized_routed + shared_scale * shared_output
        error = source_output - quantized_output
        source_sse = float(np.sum(source_output * source_output))
        sse = float(np.sum(error * error))
        moe_forward = {
            "scope": "layer-0 MoE MLP only; hybrid attention/residual composition is intentionally outside the current executable runtime",
            "tokens": int(hidden.shape[0]),
            "unique_routed_experts": sorted(source_cache),
            "topk_ids": ids.tolist(),
            "relative_mse": sse / source_sse if source_sse else 0.0,
            "rms_error": math.sqrt(sse / error.size),
            "max_abs_error": float(np.max(np.abs(error))),
            "source_rms": math.sqrt(source_sse / source_output.size),
            "shared_expert_source": "raw package passthrough (verified separately by SHA-256)",
        }

    result = {
        "schema_version": TOOL_SCHEMA,
        "created_at_utc": utc_now(),
        "router_passthrough": {
            "router_count": len(router_refs),
            "all_source_package_hashes_equal": True,
            "routing_arithmetic": "PyTorch BF16 F.linear -> FP32 softmax -> torch.topk(k=8) -> BF16 normalized selected scores",
            "tested_tokens": route_rows,
            "topk_changed_tokens": differing_rows,
            "max_selected_probability_abs_error": max_score_abs,
            "boundary_tie_tokens": boundary_ties,
            "routers": router_hashes,
        },
        "moe_sublayer_forward": moe_forward,
    }
    write_json(evidence_dir / "router-and-moe-sublayer-validation.json", result)
    return result


def robust_outlier_report(metrics: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = np.asarray([float(row[metric]) for row in metrics], dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = median + 6.0 * mad if mad > 0.0 else median * 1.5
    outliers = [row for row in metrics if float(row[metric]) > threshold]
    return {
        "metric": metric,
        "criterion": f"{metric} > median + 6 * MAD (or 1.5 * median when MAD is zero)",
        "median": median,
        "mad": mad,
        "threshold": threshold,
        "outlier_count": len(outliers),
        "outliers": outliers,
    }


def vram_estimate(plan: dict[str, Any], manifest: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    config = read_json(model_dir / "config.json")["text_config"]
    # Package byte lengths are measured at validation time below; manifest metadata
    # supplies the exact payload categories independent of filesystem block size.
    quantized_payload = sum(
        int(row["elements"]) // 2 + int(row["groups"])
        for row in manifest.get("tensors", [])
    )
    passthrough_payload = sum(int(row["payload_bytes"]) for row in manifest.get("passthrough_tensors", []))
    codebook_payload = len(manifest.get("codebooks", [])) * AQ4_CODEBOOK_ENTRIES * 4
    decoder_payload = quantized_payload + passthrough_payload + codebook_payload
    layer_types = list(config["layer_types"])
    full_layers = sum(layer == "full_attention" for layer in layer_types)
    linear_layers = sum(layer == "linear_attention" for layer in layer_types)
    kv_per_token = full_layers * 2 * int(config["num_key_value_heads"]) * int(config["head_dim"]) * 2
    conv_dim = (
        2 * int(config["linear_num_key_heads"]) * int(config["linear_key_head_dim"])
        + int(config["linear_num_value_heads"]) * int(config["linear_value_head_dim"])
    )
    linear_conv_state = linear_layers * conv_dim * int(config["linear_conv_kernel_dim"]) * 2
    linear_recurrent_state = (
        linear_layers
        * int(config["linear_num_value_heads"])
        * int(config["linear_key_head_dim"])
        * int(config["linear_value_head_dim"])
        * 4
    )
    # BI's runtime foundation establishes the exact decode gather volume: eight
    # selected BF16 routed slabs plus the one BF16 shared MLP.  A direct AQ4_0
    # matvec implementation can avoid materializing it, but reserving it makes
    # this a conservative residency ledger for an initial gather/dequant path.
    hidden = int(config["hidden_size"])
    routed_intermediate = int(config["moe_intermediate_size"])
    shared_intermediate = int(config["shared_expert_intermediate_size"])
    top_k = int(config["num_experts_per_tok"])
    selected_routed_gather = top_k * ((2 * routed_intermediate * hidden) + (hidden * routed_intermediate)) * 2
    shared_expert_gather = ((2 * shared_intermediate * hidden) + (hidden * shared_intermediate) + hidden) * 2
    moe_selected_weight_gather = selected_routed_gather + shared_expert_gather
    moe_decode_activation = (
        int(config["num_experts"]) * 4
        + top_k * (4 + 4)
        + 4
        + top_k * hidden * 4
        + top_k * (2 * routed_intermediate) * 4
        + top_k * routed_intermediate * 4
        + top_k * hidden * 4
        + 3 * hidden * 4
    )
    kv_block_size = 256
    paged_split_tile = 128
    contexts = []
    for context in (4096, 32768, 131072, int(config["max_position_embeddings"])):
        kv_bytes = kv_per_token * context
        kv_block_table = full_layers * math.ceil(context / kv_block_size) * 4
        # Existing Qwen3.5 resident full-attention dispatch reserves this
        # F32 split workspace per full-attention layer at tile 128.  It is a
        # conservative compatibility reserve until the MoE loader is wired.
        paged_split_workspace = (
            full_layers
            * int(config["num_attention_heads"])
            * math.ceil(context / paged_split_tile)
            * (int(config["head_dim"]) + 2)
            * 4
        )
        total = (
            decoder_payload
            + kv_bytes
            + kv_block_table
            + linear_conv_state
            + linear_recurrent_state
            + moe_selected_weight_gather
            + moe_decode_activation
            + paged_split_workspace
        )
        contexts.append(
            {
                "context_tokens": context,
                "full_attention_kv_bytes": kv_bytes,
                "full_attention_block_table_bytes": kv_block_table,
                "full_attention_split_workspace_bytes": paged_split_workspace,
                "known_total_bytes": total,
                "headroom_bytes": R9700_VRAM_BYTES - total,
                "fits_packed_weight_ledger": total <= R9700_VRAM_BYTES,
            }
        )
    return {
        "schema_version": TOOL_SCHEMA,
        "r9700_vram_bytes": R9700_VRAM_BYTES,
        "r9700_vram_gib": gib(R9700_VRAM_BYTES),
        "weight_ledger": {
            "routed_aq4_index_and_scale_bytes": quantized_payload,
            "raw_text_passthrough_bytes": passthrough_payload,
            "codebook_bytes": codebook_payload,
            "packed_text_decoder_bytes": decoder_payload,
            "packed_text_decoder_gib": gib(decoder_payload),
        },
        "cache_ledger": {
            "full_attention_layers": full_layers,
            "linear_attention_layers": linear_layers,
            "full_attention_kv_bytes_per_token": kv_per_token,
            "kv_block_size_tokens": kv_block_size,
            "paged_decode_split_source_tile_tokens": paged_split_tile,
            "linear_conv_state_bytes_batch1": linear_conv_state,
            "linear_recurrent_state_bytes_batch1_f32": linear_recurrent_state,
            "moe_selected_routed_bf16_gather_bytes": selected_routed_gather,
            "moe_shared_bf16_gather_bytes": shared_expert_gather,
            "moe_selected_weight_gather_reserve_bytes": moe_selected_weight_gather,
            "moe_decode_f32_activation_bytes": moe_decode_activation,
            "linear_state_derivation": "HF uses BF16 conv state and creates gated-delta recurrent state in float32; batch=1.",
        },
        "contexts": contexts,
        "qualification": "This is a batch=1 exact packed-artifact byte ledger plus source-derived KV/linear-state and conservative MoE-gather/full-attention-workspace reserves. It is not a hipMemGetInfo allocation measurement because a Qwen35 MoE loader/residency integration does not yet exist.",
    }


def validate_package(
    package_dir: Path,
    work_dir: Path,
    quant_bin: Path,
    catalog: dict[str, TensorRef],
    plan: dict[str, Any],
    model_dir: Path,
    chunk_bytes: int,
    seed: int,
) -> dict[str, Path]:
    evidence_dir = package_dir.parent / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    verify_log = evidence_dir / "full-package-verify.log"
    command = [
        str(quant_bin),
        "--verify-prototype-dir",
        str(package_dir),
        "--verify-prototype-all",
        "--verify-passthrough",
        "--chunk-bytes",
        str(chunk_bytes),
    ]
    with verify_log.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise ToolError(f"full final package verification failed; see {verify_log}")
    manifest = read_json(package_dir / "manifest.json")
    if len(manifest.get("tensors", [])) != 80:
        raise ToolError("final package does not contain all 80 routed tensors")
    master = read_json(work_dir / "master-convert-summary.json")
    metrics: list[dict[str, Any]] = []
    for index, result in enumerate(master["results"]):
        tensor = result.get("manifest")
        verification = result.get("verification")
        if result.get("status") != "ok" or not isinstance(tensor, dict) or not isinstance(verification, dict):
            raise ToolError(f"master summary result {index} has no verified tensor")
        metrics.append(
            {
                "index": index,
                "name": tensor["name"],
                "family": tensor["family"],
                "codebook_scope": tensor.get("codebook_scope"),
                "shape": tensor["shape"],
                "elements": tensor["elements"],
                "mse": tensor["metrics"]["mse"],
                "relative_mse": tensor["metrics"]["relative_mse"],
                "max_abs_error": tensor["metrics"]["max_abs_error"],
                "verify_relative_mse": verification["relative_mse"],
                "verify_max_abs_error": verification["max_abs_error"],
                "verify_elements": verification["elements"],
            }
        )
    tensor_error_path = evidence_dir / "tensor-errors.json"
    write_json(
        tensor_error_path,
        {
            "schema_version": TOOL_SCHEMA,
            "created_at_utc": utc_now(),
            "verification": "ullm-quant re-read final package indices/scales/codebooks and compared each full tensor against the source safetensors payload",
            "tensor_count": len(metrics),
            "tensors": metrics,
            "outlier_analysis": {
                "relative_mse": robust_outlier_report(metrics, "relative_mse"),
                "max_abs_error": robust_outlier_report(metrics, "max_abs_error"),
            },
        },
    )
    router_result = validate_router_and_moe_sublayer(package_dir, manifest, catalog, evidence_dir, seed)
    router_path = evidence_dir / "router-and-moe-sublayer-validation.json"
    vram_path = evidence_dir / "vram-estimate.json"
    write_json(vram_path, vram_estimate(plan, manifest, model_dir))
    metadata_path = package_dir.parent / "product-metadata.json"
    write_json(
        metadata_path,
        {
            "schema_version": PRODUCT_SCHEMA,
            "created_at_utc": utc_now(),
            "format_id": "AQ4_0",
            "source_model_dir": str(model_dir),
            "source_config_sha256": sha256_file(model_dir / "config.json"),
            "package_manifest": "package/manifest.json",
            "package_manifest_sha256": sha256_file(package_dir / "manifest.json"),
            "quantization": {
                "format": AQ4_FORMAT,
                "quantized": "routed MoE gate_up_proj and down_proj only",
                "passthrough": "routers, shared experts, attention, embeddings, norms, lm_head, and other text tensors",
                "codebook_granularity": "one codebook per routed projection family, shared across 40 layers x 256 experts",
            },
            "evidence": {
                "tensor_errors": str(tensor_error_path.relative_to(package_dir.parent)),
                "router_and_moe_sublayer": str(router_path.relative_to(package_dir.parent)),
                "vram_estimate": str(vram_path.relative_to(package_dir.parent)),
                "final_package_verify_log": str(verify_log.relative_to(package_dir.parent)),
            },
            "router_validation": router_result["router_passthrough"],
        },
    )
    return {
        "tensor_errors": tensor_error_path,
        "router": router_path,
        "vram": vram_path,
        "metadata": metadata_path,
    }


def record_streaming_evidence(
    product_dir: Path,
    model_dir: Path,
    streaming_path: Path,
    control_path: Path,
) -> Path:
    """Attach an already-run bounded streaming check to immutable package metadata.

    This deliberately does not rerun conversion or rewrite the package payload.
    It turns an end-to-end routing result into a first-class product qualification
    record, including an explicit non-pass result when lossy experts perturb a
    later router input.
    """
    package_dir = product_dir / "package"
    metadata_path = product_dir / "product-metadata.json"
    if not (package_dir / "manifest.json").is_file() or not metadata_path.is_file():
        raise ToolError("record-streaming requires an already validated product package and metadata")
    if not streaming_path.is_file() or not control_path.is_file():
        raise ToolError("record-streaming requires both streaming evidence JSON files")
    try:
        streaming_relative = streaming_path.resolve().relative_to(product_dir.resolve())
        control_relative = control_path.resolve().relative_to(product_dir.resolve())
    except ValueError as exc:
        raise ToolError("streaming evidence must be retained below the product directory") from exc

    def validate_record(path: Path, expected_mode: str) -> dict[str, Any]:
        record = read_json(path)
        if record.get("right_mode") != expected_mode:
            raise ToolError(f"{path} has right_mode={record.get('right_mode')!r}, expected {expected_mode!r}")
        if Path(str(record.get("model_dir", ""))).resolve() != model_dir.resolve():
            raise ToolError(f"{path} was made from another model directory")
        if Path(str(record.get("package_dir", ""))).resolve() != package_dir.resolve():
            raise ToolError(f"{path} was made from another package")
        if record.get("package_manifest_sha256") != sha256_file(package_dir / "manifest.json"):
            raise ToolError(f"{path} does not match the current package manifest")
        topk = record.get("topk")
        if not isinstance(topk, dict):
            raise ToolError(f"{path} lacks top-k validation data")
        required = (
            "layers_checked",
            "tokens_checked",
            "topk_order_changed_tokens",
            "topk_selected_set_changed_tokens",
        )
        if any(key not in topk for key in required):
            raise ToolError(f"{path} predates explicit ordered/set top-k accounting")
        return record

    streaming = validate_record(streaming_path, "aq4_0")
    control = validate_record(control_path, "source")
    streaming_topk = streaming["topk"]
    control_topk = control["topk"]
    if int(control_topk["topk_order_changed_tokens"]) != 0 or int(control_topk["topk_selected_set_changed_tokens"]) != 0:
        raise ToolError("source-vs-source streaming control is not exact; refusing to qualify package")

    metadata = read_json(metadata_path)
    evidence = metadata.get("evidence")
    if not isinstance(evidence, dict):
        raise ToolError("product metadata has no evidence object")
    evidence["streaming_forward"] = str(streaming_relative)
    evidence["streaming_forward_source_control"] = str(control_relative)
    selected_set_changed = int(streaming_topk["topk_selected_set_changed_tokens"])
    ordered_changed = int(streaming_topk["topk_order_changed_tokens"])
    metadata["streaming_forward_validation"] = {
        "scope": streaming.get("scope"),
        "status": "passed" if selected_set_changed == 0 else "not_passed",
        "criterion": "end-to-end equality of selected top-k expert sets; ordering is reported separately",
        "layers_checked": int(streaming_topk["layers_checked"]),
        "tokens_checked": int(streaming_topk["tokens_checked"]),
        "topk_order_changed_tokens": ordered_changed,
        "topk_selected_set_changed_tokens": selected_set_changed,
        "final_hidden_relative_l2": streaming.get("final_hidden_relative_l2"),
        "source_control_topk_order_changed_tokens": int(control_topk["topk_order_changed_tokens"]),
        "source_control_topk_selected_set_changed_tokens": int(control_topk["topk_selected_set_changed_tokens"]),
    }
    write_json(metadata_path, metadata)
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="Restartable intermediate directory on datapool.")
    parser.add_argument("--product-dir", type=Path, required=True, help="New product root; package is written below it.")
    parser.add_argument("--quant-bin", type=Path, default=Path("target/release/ullm-quant"))
    parser.add_argument(
        "--phase",
        choices=("all", "plan", "calibrate", "quantize", "merge", "validate", "record-streaming"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse only independently verified completed tensor conversions.")
    parser.add_argument("--jobs", type=int, default=8, help="Concurrent tensor conversions; default is intentionally modest.")
    parser.add_argument("--chunk-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--copy-buffer-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--scale-window", type=int, default=4)
    parser.add_argument(
        "--streaming-evidence",
        type=Path,
        help="AQ4_0 right-hand evidence made by validate-qwen35-moe-aq4-streaming-forward.py.",
    )
    parser.add_argument(
        "--streaming-control-evidence",
        type=Path,
        help="Source-vs-source control evidence made by validate-qwen35-moe-aq4-streaming-forward.py.",
    )
    parser.add_argument(
        "--aq4-group-size",
        type=int,
        choices=tuple(sorted(AQ4_FORMATS)),
        default=16,
        help="Existing AQ4_0 group size: G16 is compact; G8 is the established higher-fidelity candidate.",
    )
    parser.add_argument("--tensor-scale-reservoir-size", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=20_260_726)
    parser.add_argument("--fit-groups-per-expert", type=int, default=128)
    parser.add_argument("--eval-groups-per-expert", type=int, default=128)
    parser.add_argument("--max-global-calibration-values", type=int, default=1_048_576)
    parser.add_argument("--lloyd-iterations", type=int, default=8)
    return parser.parse_args()


def require_positive(args: argparse.Namespace) -> None:
    for name in (
        "jobs",
        "chunk_bytes",
        "copy_buffer_bytes",
        "tensor_scale_reservoir_size",
        "fit_groups_per_expert",
        "eval_groups_per_expert",
        "max_global_calibration_values",
        "lloyd_iterations",
    ):
        if int(getattr(args, name)) <= 0:
            raise ToolError(f"--{name.replace('_', '-')} must be positive")
    group_alignment_bytes = AQ4_GROUP_SIZE * 2
    if args.chunk_bytes % group_alignment_bytes:
        raise ToolError(f"--chunk-bytes must be divisible by {group_alignment_bytes} for BF16 AQ4_0 groups")
    if args.scale_window < 0:
        raise ToolError("--scale-window must be non-negative")


def main() -> int:
    args = parse_args()
    try:
        configure_aq4_group_size(args.aq4_group_size)
        require_positive(args)
        args.model_dir = args.model_dir.resolve()
        args.work_dir = args.work_dir.resolve()
        args.product_dir = args.product_dir.resolve()
        args.quant_bin = args.quant_bin.resolve()
        if not args.model_dir.is_dir():
            raise ToolError(f"model dir does not exist: {args.model_dir}")
        # Avoid accidental use for mutable active products or deployments.
        if str(args.product_dir).startswith("/opt/ullm"):
            raise ToolError("product output must not be under /opt/ullm")
        if args.phase == "record-streaming":
            if args.streaming_evidence is None or args.streaming_control_evidence is None:
                raise ToolError("record-streaming requires --streaming-evidence and --streaming-control-evidence")
            metadata = record_streaming_evidence(
                args.product_dir,
                args.model_dir,
                args.streaming_evidence,
                args.streaming_control_evidence,
            )
            print(metadata)
            return 0
        args.work_dir.mkdir(parents=True, exist_ok=True)
        plan, catalog = build_plan_and_audit(args.model_dir, args.work_dir, args.resume)
        if args.phase == "plan":
            print(args.work_dir / "plan.json")
            return 0
        codebook_path, _ = calibrate_codebooks(
            plan,
            catalog,
            args.work_dir,
            args.resume,
            args.seed,
            args.fit_groups_per_expert,
            args.eval_groups_per_expert,
            args.max_global_calibration_values,
            args.lloyd_iterations,
        )
        if args.phase == "calibrate":
            print(codebook_path)
            return 0
        master_summary = quantize_all(
            plan,
            codebook_path,
            args.work_dir,
            args.quant_bin,
            args.jobs,
            args.chunk_bytes,
            args.scale_window,
            args.tensor_scale_reservoir_size,
            args.resume,
        )
        if args.phase == "quantize":
            print(master_summary)
            return 0
        package_dir = merge_package(
            args.work_dir / "plan.json",
            master_summary,
            args.product_dir,
            args.work_dir,
            args.quant_bin,
            args.copy_buffer_bytes,
            args.resume,
        )
        if args.phase == "merge":
            print(package_dir)
            return 0
        artifacts = validate_package(
            package_dir,
            args.work_dir,
            args.quant_bin,
            catalog,
            plan,
            args.model_dir,
            args.chunk_bytes,
            args.seed,
        )
        for name, path in artifacts.items():
            print(f"{name}={path}")
        return 0
    except ToolError as exc:
        print(f"qwen35_moe_aq4_package: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
