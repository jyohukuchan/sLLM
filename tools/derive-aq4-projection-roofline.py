#!/usr/bin/env python3
"""Derive AQ4_0 Qwen3.5-9B projection roofline lower bounds from package bytes.

The calculation is intentionally a physical payload-read lower bound.  It counts
each non-MTP AQ4 index payload and scale-index payload once per decode token,
then compares that byte count with marker-attributed AQ4 projection kernel time.
It does not pretend that codebook traffic, activations, writes, or cache effects
are zero; those omissions make the resulting bandwidth efficiency an upper-bound
style diagnostic rather than a hardware-counter measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path


class RooflineError(RuntimeError):
    pass


KERNEL_FAMILIES = {
    "ullm_aq4_matvec_qkv_z_gate_beta_f32_kernel": (
        "linear_attn_qkv",
        "linear_attn_z",
        "linear_attn_a",
        "linear_attn_b",
    ),
    "ullm_aq4_matvec_triple_f32_kernel": ("attn_q", "attn_k", "attn_v"),
    "ullm_aq4_matvec_silu_mul_f32_kernel": ("mlp_gate", "mlp_up"),
    "ullm_aq4_matvec_add_f32_kernel": ("linear_attn_out", "attn_o", "mlp_down"),
    "ullm_aq4_matvec_f32_kernel": ("lm_head",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--bandwidth-gbps", type=float, default=640.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RooflineError(f"failed to read {path}: {error}") from error


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        json.dump(value, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def regular_decode_tensor(tensor: dict[str, object]) -> bool:
    name = tensor.get("name")
    return isinstance(name, str) and not name.startswith("mtp.")


def tensor_bytes(manifest_path: Path) -> tuple[dict[str, dict[str, int]], dict[str, object]]:
    manifest = read_json(manifest_path)
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        raise RooflineError("package manifest does not contain a tensor list")
    root = manifest_path.parent
    families: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_elements = 0
    total_bytes = 0
    total_tensors = 0
    for tensor in tensors:
        if not isinstance(tensor, dict) or not regular_decode_tensor(tensor):
            continue
        family = tensor.get("family")
        index_file = tensor.get("index_file")
        scale_file = tensor.get("scale_file")
        elements = tensor.get("elements")
        groups = tensor.get("groups")
        group_size = tensor.get("group_size")
        if not isinstance(family, str) or not all(
            isinstance(value, (str, int))
            for value in (index_file, scale_file, elements, groups, group_size)
        ):
            raise RooflineError("AQ4 tensor manifest entry is incomplete")
        if not isinstance(index_file, str) or not isinstance(scale_file, str):
            raise RooflineError("AQ4 tensor payload paths are invalid")
        if not isinstance(elements, int) or not isinstance(groups, int) or not isinstance(group_size, int):
            raise RooflineError("AQ4 tensor geometry is invalid")
        index_bytes = os.path.getsize(root / index_file)
        scale_bytes = os.path.getsize(root / scale_file)
        if index_bytes * 2 != elements:
            raise RooflineError(f"{tensor.get('name')}: idx4 payload byte count does not equal elements / 2")
        if scale_bytes != groups or elements != groups * group_size:
            raise RooflineError(f"{tensor.get('name')}: scale-index payload geometry is inconsistent")
        family_row = families[family]
        family_row["tensor_count"] += 1
        family_row["elements"] += elements
        family_row["index_bytes"] += index_bytes
        family_row["scale_index_bytes"] += scale_bytes
        family_row["physical_weight_bytes"] += index_bytes + scale_bytes
        total_elements += elements
        total_bytes += index_bytes + scale_bytes
        total_tensors += 1
    required = {family for group in KERNEL_FAMILIES.values() for family in group}
    missing = sorted(required.difference(families))
    if missing:
        raise RooflineError(f"decode package is missing expected AQ4 families: {missing}")
    for row in families.values():
        row["effective_bpp"] = 8.0 * row["physical_weight_bytes"] / row["elements"]
    return dict(families), {
        "decode_tensor_count": total_tensors,
        "elements": total_elements,
        "physical_weight_bytes": total_bytes,
        "effective_bpp": 8.0 * total_bytes / total_elements,
        "excluded_mtp_tensor_policy": "exclude names beginning with mtp.; normal AQ4_0 decode does not dispatch MTP layers",
    }


def kernel_timings(accounting: dict[str, object]) -> tuple[dict[str, dict[str, object]], int]:
    measurement = accounting.get("measurement")
    module_kernel = accounting.get("module_kernel")
    if not isinstance(measurement, dict) or not isinstance(module_kernel, dict):
        raise RooflineError("wall-time accounting JSON has an incompatible schema")
    step_count = measurement.get("decode_step_count")
    entries = module_kernel.get("kernels")
    if not isinstance(step_count, int) or step_count <= 0 or not isinstance(entries, list):
        raise RooflineError("wall-time accounting JSON has invalid kernel timing data")
    by_name = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            by_name[entry["name"]] = entry
    missing = sorted(set(KERNEL_FAMILIES).difference(by_name))
    if missing:
        raise RooflineError(f"trace lacks expected AQ4 projection kernels: {missing}")
    return by_name, step_count


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.bandwidth_gbps <= 0:
        raise RooflineError("--bandwidth-gbps must be positive")
    accounting = read_json(args.accounting)
    families, total = tensor_bytes(args.package_manifest)
    timing_by_name, step_count = kernel_timings(accounting)
    bandwidth_bytes_per_ns = args.bandwidth_gbps
    per_kernel = []
    mapped_total_bytes = 0
    mapped_total_ns = 0.0
    for name, family_names in KERNEL_FAMILIES.items():
        byte_count = sum(families[family]["physical_weight_bytes"] for family in family_names)
        elements = sum(families[family]["elements"] for family in family_names)
        timed_ns = int(timing_by_name[name]["inclusive_ns"]) / step_count
        lower_bound_ns = byte_count / bandwidth_bytes_per_ns
        effective_bandwidth = byte_count / timed_ns
        per_kernel.append(
            {
                "kernel": name,
                "families": list(family_names),
                "physical_weight_bytes_per_token": byte_count,
                "elements_per_token": elements,
                "effective_bpp": 8.0 * byte_count / elements,
                "observed_kernel_ns_per_token": timed_ns,
                "bandwidth_roofline_ns_per_token": lower_bound_ns,
                "effective_bandwidth_gbps": effective_bandwidth,
                "efficiency_of_bandwidth_roofline": effective_bandwidth / args.bandwidth_gbps,
                "max_kernel_speedup_to_payload_bandwidth_floor": timed_ns / lower_bound_ns,
            }
        )
        mapped_total_bytes += byte_count
        mapped_total_ns += timed_ns
    if mapped_total_bytes != total["physical_weight_bytes"]:
        raise RooflineError(
            f"kernel mapping accounts for {mapped_total_bytes} bytes, expected {total['physical_weight_bytes']}"
        )
    module_total_ns = int(accounting["module_kernel"]["inclusive_ns"]) / step_count
    payload_floor_ns = mapped_total_bytes / bandwidth_bytes_per_ns
    return {
        "schema_version": "ullm.aq4_projection_roofline.v1",
        "inputs": {
            "package_manifest": str(args.package_manifest),
            "accounting": str(args.accounting),
            "bandwidth_gbps": args.bandwidth_gbps,
        },
        "method": {
            "weight_read_lower_bound": (
                "Each regular Qwen3.5 decoder/lm-head AQ4 tensor's idx4 payload and one-byte "
                "scale index payload are counted once. This excludes codebook/activation/output "
                "traffic and therefore is a lower bound, not a hardware-counter byte total."
            ),
            "AQ4_0_effective_bpp": "4 bits/index + 8/group_size bits for u8 scale-table index",
        },
        "payload": {"by_family": families, "decode_total": total},
        "projection_kernels": per_kernel,
        "projection_total": {
            "physical_weight_bytes_per_token": mapped_total_bytes,
            "observed_kernel_ns_per_token": mapped_total_ns,
            "bandwidth_roofline_ns_per_token": payload_floor_ns,
            "effective_bandwidth_gbps": mapped_total_bytes / mapped_total_ns,
            "efficiency_of_bandwidth_roofline": (mapped_total_bytes / mapped_total_ns) / args.bandwidth_gbps,
            "max_kernel_speedup_to_payload_bandwidth_floor": mapped_total_ns / payload_floor_ns,
            "module_kernel_ns_per_token": module_total_ns,
        },
    }


def main() -> int:
    args = parse_args()
    try:
        write_json(args.output, run(args))
    except RooflineError as error:
        print(f"derive-aq4-projection-roofline: {error}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
