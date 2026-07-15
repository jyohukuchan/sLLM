#!/usr/bin/env python3
"""Compare layer-0 raw AQ4 family outputs with BF16 source weights.

The AQ4 sidecars are produced by the CPU-only Rust diagnostic.  This tool
loads one source tensor at a time through ``safe_open`` and keeps only one
weight and one output row resident at a time.  It reports metrics only;
thresholds, promotion, and holdout policy are deliberately not evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


SCHEMA = "ullm.aq4_layer0_family_isolation.source_compare.v1"
AQ4_SCHEMA = "ullm.aq4_layer0_family_isolation.aq4_cpu.v1"
INPUT_SCHEMA = "ullm.aq4_layer0_input_normed_jsonl.v1"
FAMILIES = {
    "qkv": ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", 8192),
    "z": ("model.language_model.layers.0.linear_attn.in_proj_z.weight", 4096),
    "a": ("model.language_model.layers.0.linear_attn.in_proj_a.weight", 32),
    "b": ("model.language_model.layers.0.linear_attn.in_proj_b.weight", 32),
}
INPUT_COLS = 4096
MAX_ROWS = 4096


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def f32_bytes(values: torch.Tensor) -> bytes:
    return values.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy().tobytes()


def source_tensor_bytes(values: torch.Tensor) -> bytes:
    # safetensors stores BF16 little-endian payloads on this host.  Hash the
    # logical source tensor before converting it to f32 for the comparison.
    # NumPy does not expose a native bfloat16 scalar on this host, so hash the
    # identical 16-bit payload through a uint16 view.
    return values.detach().to(device="cpu").contiguous().view(torch.uint16).numpy().tobytes()


def read_input(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    lines = [line for line in raw.splitlines() if line]
    if not lines:
        raise ValueError("input is empty")
    header = json.loads(lines[0])
    if header.get("kind") != "header" or header.get("schema_version") != INPUT_SCHEMA:
        raise ValueError("input header schema differs")
    if header.get("dtype") != "f32" or header.get("shape") != [INPUT_COLS]:
        raise ValueError("input must be f32 [4096]")
    cases: list[dict[str, Any]] = []
    for line in lines[1:]:
        case = json.loads(line)
        values = case.get("values")
        if case.get("kind") != "case" or not isinstance(values, list) or len(values) != INPUT_COLS:
            raise ValueError(f"invalid input case {case.get('case_id')}")
        actual = sha256_bytes(struct.pack(f"<{len(values)}f", *values))
        if case.get("input_sha256") != actual:
            raise ValueError(f"input hash mismatch for {case.get('case_id')}")
        cases.append(case)
    if not cases or len(cases) > MAX_ROWS:
        raise ValueError("input case count is outside bounds")
    return header, cases, digest


def metric_row(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    actual = actual.to(dtype=torch.float64, device="cpu").flatten()
    reference = reference.to(dtype=torch.float64, device="cpu").flatten()
    if actual.numel() != reference.numel():
        raise ValueError("AQ4/source row geometry differs")
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(reference).all()):
        return {
            "max_abs": None,
            "relative_l2": None,
            "cosine": None,
            "nonfinite": True,
        }
    diff = actual - reference
    max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
    diff_norm = float(torch.linalg.vector_norm(diff).item())
    ref_norm = float(torch.linalg.vector_norm(reference).item())
    actual_norm = float(torch.linalg.vector_norm(actual).item())
    relative_l2 = diff_norm / ref_norm if ref_norm else (0.0 if diff_norm == 0.0 else math.inf)
    cosine = float(torch.dot(actual, reference).item() / (actual_norm * ref_norm)) if actual_norm and ref_norm else None
    return {
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "nonfinite": False,
    }


def aggregate(rows: list[dict[str, Any]], actual_rows: list[torch.Tensor], reference_rows: list[torch.Tensor]) -> dict[str, Any]:
    finite_rows = [row for row in rows if not row["nonfinite"]]
    diff_sq = 0.0
    ref_sq = 0.0
    dot = 0.0
    actual_sq = 0.0
    max_abs = 0.0
    for actual, reference in zip(actual_rows, reference_rows):
        actual64 = actual.to(dtype=torch.float64, device="cpu").flatten()
        ref64 = reference.to(dtype=torch.float64, device="cpu").flatten()
        if not bool(torch.isfinite(actual64).all()) or not bool(torch.isfinite(ref64).all()):
            continue
        diff = actual64 - ref64
        diff_sq += float(torch.dot(diff, diff).item())
        ref_sq += float(torch.dot(ref64, ref64).item())
        actual_sq += float(torch.dot(actual64, actual64).item())
        dot += float(torch.dot(actual64, ref64).item())
        max_abs = max(max_abs, float(diff.abs().max().item()))
    relative_l2 = math.sqrt(diff_sq) / math.sqrt(ref_sq) if ref_sq else (0.0 if diff_sq == 0.0 else math.inf)
    cosine = dot / math.sqrt(actual_sq * ref_sq) if actual_sq and ref_sq else None
    return {
        "rows": len(rows),
        "finite_rows": len(finite_rows),
        "nonfinite_rows": len(rows) - len(finite_rows),
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "thresholds": None,
        "policy_evaluation": "policy_not_evaluated",
    }


def compare_family(
    family: str,
    tensor_name: str,
    expected_rows: int,
    package_report: dict[str, Any],
    output_dir: Path,
    source_dir: Path,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    output = package_report["outputs"].get(family)
    if not output or output["shape"] != [expected_rows] or len(output["cases"]) != len(cases):
        raise ValueError(f"AQ4 output report for {family} is incomplete")
    sidecar = Path(output["path"])
    if not sidecar.is_absolute():
        sidecar = output_dir / sidecar.name
    if not sidecar.is_file():
        raise ValueError(f"AQ4 sidecar missing: {sidecar}")
    if sha256_file(sidecar) != output["sha256"]:
        raise ValueError(f"AQ4 sidecar hash changed: {sidecar}")
    index = json_load(source_dir / "model.safetensors.index.json")
    shard_name = index["weight_map"][tensor_name]
    shard_path = source_dir / shard_name
    source_index_sha = sha256_file(source_dir / "model.safetensors.index.json")
    source_rows: list[torch.Tensor] = []
    actual_rows: list[torch.Tensor] = []
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        source_weight = handle.get_tensor(tensor_name)
        if list(source_weight.shape) != [expected_rows, INPUT_COLS] or source_weight.dtype != torch.bfloat16:
            raise ValueError(f"source tensor geometry/dtype differs for {family}")
        source_tensor_sha = sha256_bytes(source_tensor_bytes(source_weight))
        with sidecar.open("rb") as stream:
            for case_index, case in enumerate(cases):
                raw = stream.read(expected_rows * 4)
                if len(raw) != expected_rows * 4:
                    raise ValueError(f"AQ4 sidecar row {case_index} is short for {family}")
                actual = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
                input_tensor = torch.tensor(case["values"], dtype=torch.float32)
                # Explicit contract: same f32 input, BF16 source weight cast to
                # f32 before an f32 CPU matmul.  No model-level state is inferred.
                reference = torch.matmul(input_tensor, source_weight.to(dtype=torch.float32).transpose(0, 1))
                actual_rows.append(actual)
                source_rows.append(reference)
    per_row = []
    for case, actual, reference in zip(cases, actual_rows, source_rows):
        metrics = metric_row(actual, reference)
        per_row.append(
            {
                "case_id": case["case_id"],
                "step": case["step"],
                "input_sha256": case["input_sha256"],
                **metrics,
                "actual_sha256": sha256_bytes(f32_bytes(actual)),
                "source_sha256": sha256_bytes(f32_bytes(reference)),
            }
        )
    return {
        "family": family,
        "tensor_name": tensor_name,
        "shape": [expected_rows, INPUT_COLS],
        "source": {
            "model_dir": str(source_dir),
            "index_path": str(source_dir / "model.safetensors.index.json"),
            "index_sha256": source_index_sha,
            "shard_path": str(shard_path),
            "shard_bytes": shard_path.stat().st_size,
            "tensor_dtype": "BF16",
            "tensor_shape": [expected_rows, INPUT_COLS],
            "tensor_payload_sha256": source_tensor_sha,
            "operation": "input_f32_matmul_weight_bf16_cast_f32_accumulate_f32",
        },
        "per_row": per_row,
        "aggregate": aggregate(per_row, actual_rows, source_rows),
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    package_report = json_load(args.aq4_report)
    if package_report.get("schema_version") != AQ4_SCHEMA or package_report.get("status") != "valid":
        raise ValueError("AQ4 report schema/status is invalid")
    header, cases, input_sha = read_input(args.input)
    if package_report["input"]["consumed_sha256"] != input_sha:
        raise ValueError("AQ4 report input digest differs")
    if package_report["input"]["rows"] != len(cases):
        raise ValueError("AQ4 report row count differs")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)
    family_reports = []
    for family, (tensor_name, rows) in FAMILIES.items():
        family_reports.append(compare_family(family, tensor_name, rows, package_report, args.aq4_report.parent, args.source_model, cases))
    # This is a diagnostic ranking only, not a policy verdict.  The largest
    # aggregate relative-L2 mismatch is the most useful next family candidate;
    # max-absolute is reported separately because tensor row counts differ.
    relative_l2_family = max(family_reports, key=lambda item: item["aggregate"]["relative_l2"])
    max_abs_family = max(family_reports, key=lambda item: item["aggregate"]["max_abs"])
    result = {
        "schema_version": SCHEMA,
        "status": "valid",
        "classification": "unclassified",
        "promotion": False,
        "holdout": "not_run",
        "policy_evaluation": "policy_not_evaluated",
        "thresholds": None,
        "input": {
            "path": str(args.input),
            "schema": header["schema_version"],
            "dtype": header["dtype"],
            "shape": header["shape"],
            "rows": len(cases),
            "sha256": input_sha,
        },
        "aq4_probe": {
            "report_path": str(args.aq4_report),
            "report_sha256": sha256_file(args.aq4_report),
            "package_root": package_report["package_root"],
            "package_manifest_sha256": package_report["package_manifest_sha256"],
        },
        "source_model": {
            "model_dir": str(args.source_model),
            "index_path": str(args.source_model / "model.safetensors.index.json"),
            "index_sha256": sha256_file(args.source_model / "model.safetensors.index.json"),
        },
        "families": family_reports,
        "dominant_family_candidate": {
            "metric": "aggregate.relative_l2",
            "family": relative_l2_family["family"],
            "value": relative_l2_family["aggregate"]["relative_l2"],
            "max_abs_family": max_abs_family["family"],
            "max_abs_value": max_abs_family["aggregate"]["max_abs"],
            "status": "diagnostic_candidate_only",
        },
        "dominance_rule": "not_inferred_without_policy_thresholds; compare aggregate relative_l2/max_abs/cosine only",
        "one_at_a_time_hybrid": package_report["one_at_a_time_hybrid"],
        "notes": [
            "Only raw per-family matvecs were compared. Layer norm, gate/beta transforms, recurrent state, residual, and hidden contribution are not synthesized.",
            "AQ4 and source outputs use identical f32 input rows; source BF16 weights are explicitly cast to f32 for the bounded CPU matmul.",
        ],
    }
    report_path = args.output / "comparison.json"
    report_path.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "SHA256SUMS").write_text(
        f"{sha256_file(report_path)}  comparison.json\n", encoding="ascii"
    )
    artifact_root = args.output.parent
    checksum_lines = []
    for path in sorted(artifact_root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        checksum_lines.append(
            f"{sha256_file(path)}  {path.relative_to(artifact_root).as_posix()}\n"
        )
    (artifact_root / "SHA256SUMS").write_text("".join(checksum_lines), encoding="ascii")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aq4-report", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"layer0 family comparison failed: {error}")
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
