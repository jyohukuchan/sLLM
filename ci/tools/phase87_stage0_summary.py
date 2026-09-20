#!/usr/bin/env python3
"""Assemble the Phase 87 Stage 0 decode accounting evidence.

The profile and counter runs are deliberately separate: kernel duration comes
from the ordinary ``rocprofv3 --kernel-trace`` run, while read-request bytes
come from the perturbing counter run.  This tool joins them only after exact
``(Kernel_Name, Grid_Size_X)`` invocation counts and benchmark token output
have been checked.  Counter bytes are GL2C/EA interface request bytes and are
never labelled physical DRAM bytes.  Copy bandwidth is a mixed read+write
proxy used for a signed roofline score; it is not a strict memory ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "phase87-stage0-summary-v1"
TARGETS = ("gfx1030", "gfx1201")
MODES = ("off", "on")
REQUIRED = tuple((target, mode) for target in TARGETS for mode in MODES)


class SummaryError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise SummaryError(message)


def read_json(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file():
        fail(f"missing {label}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} {path}: {exc}")


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        fail(f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"invalid {label} line {line_number}: {exc}")
        if not isinstance(value, dict):
            fail(f"{label} line {line_number} is not an object")
        rows.append(value)
    if not rows:
        fail(f"{label} is empty: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_run(report: dict[str, Any], label: str) -> dict[str, Any]:
    if report.get("state") != "PASS":
        fail(f"{label} report is not PASS")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        fail(f"{label} must contain exactly one benchmark row")
    runs = rows[0].get("runs")
    if not isinstance(runs, list) or not runs:
        fail(f"{label} has no runs")
    measured = [run for run in runs if isinstance(run, dict) and run.get("sample_kind") == "measured"]
    if len(measured) != 1:
        fail(f"{label} must contain exactly one measured run")
    run = measured[0]
    audit = run.get("audit")
    if not isinstance(audit, dict) or audit.get("all_dispatches_hip") is not True or audit.get("fallback_used") is not False:
        fail(f"{label} audit is not HIP-only/no-fallback")
    generated = run.get("generated_tokens")
    if not isinstance(generated, list) or not generated:
        fail(f"{label} generated token list is missing")
    return {
        "target": report.get("target"),
        "mtp_requested": bool(isinstance(report.get("mtp"), dict) and report["mtp"].get("requested") is True),
        "model_sha256": report.get("model", {}).get("model_sha256") if isinstance(report.get("model"), dict) else None,
        "fixture_sha256": report.get("fixture", {}).get("sha256") if isinstance(report.get("fixture"), dict) else None,
        "prompt_tokens": rows[0].get("prompt_tokens"),
        "output_tokens": rows[0].get("output_tokens"),
        "decode_transition_count": run.get("decode_transition_count"),
        "generated_tokens": generated,
        "generated_tokens_sha256": run.get("generated_tokens_sha256"),
        "audit_kernel_dispatch_count": audit.get("kernel_dispatch_count"),
    }


def load_pair(root: Path, target: str, mode: str) -> dict[str, Any]:
    profile_dir = root / f"profile-{target}" / f"profile-mtp-{mode}"
    counter_dir = root / f"counter-{target}" / f"counter-mtp-{mode}"
    profile_analysis_path = profile_dir / "analysis.json"
    counter_analysis_path = counter_dir / "analysis.json"
    profile_breakdown_path = profile_dir / "breakdown.json"
    for path, label in (
        (profile_analysis_path, "profile analysis"),
        (counter_analysis_path, "counter analysis"),
        (profile_breakdown_path, "profile breakdown"),
    ):
        if not path.is_file():
            fail(f"missing final {label} for {target}/{mode}: {path}")
    profile = read_json(profile_analysis_path, "profile analysis")
    counter = read_json(counter_analysis_path, "counter analysis")
    breakdown = read_json(profile_breakdown_path, "profile breakdown")
    if profile.get("state") != "PASS" or counter.get("state") != "PASS" or breakdown.get("state") != "PASS":
        fail(f"analysis state is not PASS for {target}/{mode}")
    p_report_path = root / f"profile-{target}" / f"profile-mtp-{mode}" / "report.json"
    c_report_path = root / f"counter-{target}" / f"counter-mtp-{mode}" / "report.json"
    p_run = report_run(read_json(p_report_path, "profile benchmark report"), f"profile {target}/{mode}")
    c_run = report_run(read_json(c_report_path, "counter benchmark report"), f"counter {target}/{mode}")
    for field in ("target", "mtp_requested", "model_sha256", "fixture_sha256", "prompt_tokens", "output_tokens", "decode_transition_count"):
        if p_run.get(field) != c_run.get(field):
            fail(f"profile/counter benchmark mismatch {target}/{mode}: {field}")
    if p_run["generated_tokens"] != c_run["generated_tokens"]:
        fail(f"profile/counter generated token mismatch {target}/{mode}")
    if profile["decode_segment"].get("dispatch_count") != counter["decode_segment"].get("dispatch_count"):
        fail(f"profile/counter decode segment dispatch mismatch {target}/{mode}")
    if p_run["decode_transition_count"] != 127:
        fail(f"unexpected committed decode transition count for {target}/{mode}: {p_run['decode_transition_count']}")
    measured = counter.get("measured_dram_read", {})
    if measured.get("state") not in {"available_interface_request_estimate", "partial_interface_request_estimate"}:
        fail(f"counter read-request evidence unavailable for {target}/{mode}: {measured}")
    if measured.get("incomplete_instance_count") != 0:
        fail(f"counter coverage incomplete for {target}/{mode}")
    return {
        "target": target,
        "mtp": mode == "on",
        "profile": profile,
        "counter": counter,
        "breakdown": breakdown,
        "profile_run": p_run,
        "counter_run": c_run,
        "inputs": {str(path): sha256(path) for path in (profile_analysis_path, counter_analysis_path, profile_breakdown_path, p_report_path, c_report_path)},
    }


def load_shapes(root: Path) -> dict[tuple[int, int, str], int]:
    path = root / "linear-tensor-shapes.json"
    value = read_json(path, "linear tensor shape metadata")
    if not isinstance(value, dict):
        fail("linear tensor shape metadata is not an object")
    counts: Counter[tuple[int, int, str]] = Counter()
    for section in value.values():
        if not isinstance(section, dict):
            continue
        for tensor in section.get("tensors", []):
            if not isinstance(tensor, dict):
                continue
            try:
                counts[(int(tensor["k"]), int(tensor["n"]), str(tensor["format"]))] += 1
            except (KeyError, TypeError, ValueError):
                fail("linear tensor shape metadata contains an invalid tensor")
    if not counts:
        fail("linear tensor shape metadata has no tensors")
    return dict(counts)


def load_copy(root: Path, target: str) -> list[dict[str, Any]]:
    streaming = root / f"copy-streaming-{target}.jsonl"
    path = streaming
    rows = read_jsonl(path, "copy-streaming output")
    if len(rows) != 73:
        fail(f"copy-streaming {target} has {len(rows)} rows; expected 73")
    for row in rows:
        if row.get("status") != "PASS" or row.get("target") != target:
            fail(f"copy-streaming {target} contains a non-PASS row")
        if int(row["payload_bytes"]) >= 1024**2 and int(row.get("working_set_bytes", 0)) < 512 * 1024**2:
            fail(f"copy-streaming {target} does not exceed the large-cache working set")
        if not math.isfinite(copy_bw(row)) or copy_bw(row) <= 0:
            fail(f"copy-streaming {target} has invalid bandwidth")
    return rows


def copy_path(root: Path, target: str) -> Path:
    streaming = root / f"copy-streaming-{target}.jsonl"
    return streaming


def copy_bw(row: dict[str, Any]) -> float:
    # The probe prints GiB/s.  Summary scores use decimal GB/s.
    return float(row["effective_gib_per_s"]) * (1024.0**3) / 1_000_000_000.0


def copy_match(rows: list[dict[str, Any]], candidates: list[tuple[int, int, int, str]], *, payload_range: tuple[int, int] | None = None) -> dict[str, Any] | None:
    matched: list[dict[str, Any]] = []
    for row in rows:
        if payload_range is not None:
            payload = int(row.get("payload_bytes", -1))
            if row.get("encoding") == "bytes" and payload_range[0] <= payload <= payload_range[1]:
                matched.append(row)
        else:
            key = (int(row["m"]), int(row["k"]), int(row["n"]), str(row["encoding"]))
            if key in candidates:
                matched.append(row)
    if not matched:
        return None
    speeds = [copy_bw(row) for row in matched]
    return {
        "state": "available",
        "row_count": len(matched),
        "copy_gb_s_min": min(speeds),
        "copy_gb_s_max": max(speeds),
        "reference_rows": [
            {"m": row.get("m"), "k": row.get("k"), "n": row.get("n"), "encoding": row.get("encoding"), "payload_bytes": row.get("payload_bytes"), "median_ms": row.get("median_ms"), "effective_gb_s": copy_bw(row)}
            for row in matched
        ],
    }


def matrix_candidates(target: str, mtp: bool, kernel_name: str, grid: int, shape_counts: dict[tuple[int, int, str], int]) -> tuple[str, list[tuple[int, int, int, str]]] | None:
    base_m = 3 if mtp else 1
    if "nvfp4" in kernel_name.lower() and "to_nvfp4" not in kernel_name.lower():
        dimensions = {139264: (5120, 17408), 40960: (17408, 5120)}
        if grid in dimensions:
            k, n = dimensions[grid]
            return "nvfp4_matmul", [(base_m, k, n, "nvfp4")]
    lower = kernel_name.lower()
    if "fp8" in lower or (target == "gfx1201" and "cijk_" in lower and "f8bs" in lower):
        if target == "gfx1030":
            if grid == 1986560:
                k, n = 5120, 248320
            else:
                match = re.search(r"k(\d+)n(\d+)", lower)
                if not match:
                    return None
                k, n = int(match.group(1)), int(match.group(2))
        else:
            mapping = {20480: (5120, 10240), 12288: (5120, 6144), 10240: (6144, 5120), 8192: (5120, 1024), 24576: (5120, 12288), 34816: (5120, 17408), 2048: (5120, 1024), 124160: (5120, 248320)}
            if grid not in mapping:
                return None
            k, n = mapping[grid]
            # Resolve these two K values in assemble_pair using the artifact
            # occurrence counts and independently observed target forwards.
            if grid == 10240:
                return "fp8_matmul", [(base_m, candidate_k, 5120, "fp8")
                                      for candidate_k in (6144, 17408)]
        ms = [1, 3] if (mtp and n == 248320) else [base_m]
        return "fp8_matmul", [(m, k, n, "fp8") for m in ms]
    return None


def bf16_candidates(mtp: bool, kernel_name: str, grid: int, shape_counts: dict[tuple[int, int, str], int]) -> tuple[str, list[tuple[int, int, int, str]]] | None:
    if grid % 256 != 0:
        return None
    n = grid // 256
    possible = sorted((k, n, "bf16") for k, candidate_n, fmt in shape_counts if fmt == "bf16" and candidate_n == n)
    if not possible:
        return None
    # The v4 kernel is a single-row matvec. Target GDN A/B projections use
    # serial_rows at M=3 during verify; BF16 companion matvecs remain M=1.
    ms = [3] if mtp and "serial_rows" in kernel_name else [1]
    return "bf16_matmul", [(m, k, n, "bf16") for m in ms for k, _n, _fmt in possible]


def quant_candidates(target: str, mtp: bool, kernel_name: str, grid: int) -> tuple[str, list[tuple[int, int, int, str]]] | None:
    lower = kernel_name.lower()
    if "to_nvfp4" in lower:
        candidates = []
        for k in (5120, 6144, 17408):
            if grid % (2 * k) == 0:
                candidates.append((grid // (2 * k), k, 1, "bytes"))
        # n=1 is a byte-payload copy key; actual copy matching is handled below.
        return "activation_quantize", candidates
    if "to_fp8" in lower:
        ms = [1, 3] if mtp else [1]
        return "activation_quantize", [(m, k, 1, "bytes") for m in ms for k in (5120, 6144, 17408)]
    return None


def attention_reference(rows: list[dict[str, Any]], kernel_name: str) -> dict[str, Any] | None:
    lower = kernel_name.lower()
    if "stage1" in lower:
        return copy_match(rows, [], payload_range=(17_300_000, 17_600_000))
    if "stage2" in lower:
        return copy_match(rows, [], payload_range=(792576, 792576))
    return None


def priority_for(family: str, kernel_name: str) -> str:
    lower = kernel_name.lower()
    if family == "full_attention" and ("stage1" in lower or "stage2" in lower):
        return "attention_stage1_stage2"
    if family in {"nvfp4_matmul", "fp8_matmul", "bf16_matmul"}:
        return "matmul"
    if family == "activation_quantize":
        return "quantization"
    return "nonpriority"


def assemble_pair(pair: dict[str, Any], copy_rows: list[dict[str, Any]], shape_counts: dict[tuple[int, int, str], int]) -> dict[str, Any]:
    target = pair["target"]
    mtp = pair["mtp"]
    profile_kernels = pair["profile"]["decode_kernels"]
    counter_grid = pair["counter"]["measured_dram_read"].get("per_kernel_grid", [])
    counter_map = {(str(row["kernel_name"]), int(row["grid_size"])): row for row in counter_grid}
    profile_map = {(str(row["kernel_name"]), int(row["grid_x"])): row for row in profile_kernels}
    if set(profile_map) != set(counter_map):
        fail(f"profile/counter kernel grid set mismatch for {target}/{('on' if mtp else 'off')}")
    total_duration_ns = sum(int(row["duration_ns"]) for row in profile_kernels)
    nv_wide_calls = sum(int(row["dispatch_count"]) for row in profile_kernels
                       if row["family"] == "nvfp4_matmul" and int(row["grid_x"]) == 139264)
    if nv_wide_calls <= 0 or nv_wide_calls % 112:
        fail("target-forward count is not proven by the 112 NVFP4 gate/up matrices")
    target_forwards = nv_wide_calls // 112
    transitions = int(pair["profile_run"]["decode_transition_count"])
    rows: list[dict[str, Any]] = []
    for key in sorted(profile_map):
        profile_row = profile_map[key]
        counter_row = counter_map[key]
        if int(profile_row["dispatch_count"]) != int(counter_row["complete_instance_count"]):
            fail(f"profile/counter invocation mismatch {target}/{mtp}: {key} {profile_row['dispatch_count']} != {counter_row['complete_instance_count']}")
        duration_ns = int(profile_row["duration_ns"])
        read_bytes = float(counter_row["read_request_bytes"])
        duration_s = duration_ns / 1_000_000_000.0
        effective_gb_s = read_bytes / duration_s / 1_000_000_000.0 if duration_s else None
        family = str(profile_row["family"])
        kernel_name, grid = key
        candidates: list[tuple[int, int, int, str]] = []
        candidate_kind: str | None = None
        shape_info = matrix_candidates(target, mtp, kernel_name, grid, shape_counts) if family in {"nvfp4_matmul", "fp8_matmul"} else None
        if target == "gfx1201" and family == "fp8_matmul" and grid == 10240:
            calls = int(profile_row["dispatch_count"])
            # Same N and launch grid, different K. Prove the mapping against
            # the artifact's 64 output projections and eight MLP down matrices,
            # rather than inferring it from a library implementation flag.
            if calls == 64 * target_forwards:
                shape_info = ("fp8_matmul", [(3 if mtp else 1, 6144, 5120, "fp8")])
            elif calls == 8 * target_forwards:
                shape_info = ("fp8_matmul", [(3 if mtp else 1, 17408, 5120, "fp8")])
            else:
                fail("ambiguous gfx1201 FP8 N5120 shape: tensor occurrence counts do not match")
        if shape_info is None:
            shape_info = bf16_candidates(mtp, kernel_name, grid, shape_counts) if family == "bf16_matmul" else None
        if shape_info is None:
            shape_info = quant_candidates(target, mtp, kernel_name, grid) if family == "activation_quantize" else None
        if shape_info is not None:
            candidate_kind, candidates = shape_info
            for _m, candidate_k, candidate_n, encoding in candidates:
                if encoding in {"nvfp4", "fp8", "bf16"} and (candidate_k, candidate_n, encoding) not in shape_counts:
                    fail(f"shape candidate is absent from metadata {target}/{mtp}/{key}: {_m}x{candidate_k}x{candidate_n}:{encoding}")
        copy_reference = attention_reference(copy_rows, kernel_name) if family == "full_attention" else None
        if copy_reference is None and candidates:
            # Quantization candidates encode byte payload as (m, k, 1, bytes).
            if candidates[0][3] == "bytes":
                payloads = [2 * m * k for m, k, _n, _fmt in candidates]
                matched = [row for row in copy_rows if row.get("encoding") == "bytes" and int(row.get("payload_bytes", -1)) in payloads]
                if not matched:
                    copy_reference = None
                else:
                    speeds = [copy_bw(row) for row in matched]
                    copy_reference = {"state": "available", "row_count": len(matched), "copy_gb_s_min": min(speeds), "copy_gb_s_max": max(speeds), "candidate_payload_bytes": sorted(set(payloads))}
            else:
                copy_reference = copy_match(copy_rows, candidates)
        if copy_reference is None and candidates:
            fail(f"no copy-warm reference for eligible kernel {target}/{mtp}/{key}")
        time_share = duration_ns / total_duration_ns if total_duration_ns else 0.0
        score = None
        if copy_reference is not None and effective_gb_s is not None:
            score = time_share * (float(copy_reference["copy_gb_s_max"]) - effective_gb_s)
        rows.append({
            "kernel_name": kernel_name,
            "grid_x": grid,
            "family": family,
            "priority": priority_for(family, kernel_name),
            "invocation_count": int(profile_row["dispatch_count"]),
            "duration_ms": duration_ns / 1_000_000.0,
            "duration_ms_per_transition": duration_ns / 1_000_000.0 / transitions,
            "time_share": time_share,
            "read_request_bytes": read_bytes,
            "read_request_bytes_per_transition": read_bytes / transitions,
            "effective_gb_s": effective_gb_s,
            "copy_reference": copy_reference,
            "copy_score_signed": score,
            "shape_candidates": [{"m": m, "k": k, "n": n, "encoding": fmt} for m, k, n, fmt in candidates],
            "shape_candidate_kind": candidate_kind,
        })
    return {
        "target": target,
        "mtp": mtp,
        "inputs": pair["inputs"],
        "committed_decode_transitions": transitions,
        "kernel_duration_source": "ordinary rocprof kernel trace; no counter timing",
        "read_source": "separate rocprof GL2C/EA interface request counter run; not physical DRAM",
        "copy_source": "copy-warm mixed read+write proxy; not a strict ceiling",
        "total_kernel_duration_ms": total_duration_ns / 1_000_000.0,
        "time_share_domain": "sum of decode GPU kernel durations; host idle is reported separately",
        "target_forward_count": target_forwards,
        "rows": rows,
        "generated_tokens_sha256": pair["profile_run"]["generated_tokens_sha256"],
    }


def load_baseline(root: Path) -> list[dict[str, Any]]:
    path = root / "baseline-summary.json"
    value = read_json(path, "baseline summary")
    if not isinstance(value, list):
        fail("baseline summary is not a list")
    result = []
    for row in value:
        if not isinstance(row, dict) or row.get("status") != "PASS":
            fail("baseline summary contains a non-PASS row")
        result.append({"target": row.get("target"), "mtp": row.get("mtp"), "decode_tokens_per_second": row.get("summary", {}).get("decode_tokens_per_second"), "report": row.get("report")})
    return result


def assemble(root: Path) -> dict[str, Any]:
    shape_counts = load_shapes(root)
    pairs = [load_pair(root, target, mode) for target, mode in REQUIRED]
    copy_rows = {target: load_copy(root, target) for target in TARGETS}
    assembled = [assemble_pair(pair, copy_rows[pair["target"]], shape_counts) for pair in pairs]
    for item in assembled:
        selected_copy = copy_path(root, item["target"])
        item["copy_source"] = str(selected_copy)
        item["copy_source_sha256"] = sha256(selected_copy)
    evidence_paths = [root / name for name in (
        "initial-state.json", "binaries.json", "profile-binaries.json",
        "baseline-summary.json", "baseline-route-source.json", "kld-corpus.json",
        "counter-coverage.json", "profile-output-equivalence.json",
        "copy-source-identity.json", "kld-gfx1201/execution.json",
        "kld-gfx1201/fp16-vs-bf16.json", "kld-gfx1201/kv-mxfp8-e4-vs-bf16.json",
    )]
    for target in TARGETS:
        evidence_paths.extend(root / f"{kind}-{target}/execution.json"
                              for kind in ("baseline", "profile", "counter"))
        evidence_paths.append(root / f"copy-streaming-{target}-execution.json")
    return {
        "schema": SCHEMA,
        "state": "PASS",
        "scope": "Qwen3.8 single-request decode Stage 0; four profile/counter pairs",
        "identity_scope": "local dirty-tree measurement, not a release candidate",
        "evidence_index": [{"path": str(path), "sha256": sha256(path)}
                           for path in evidence_paths],
        "counter_domain": "GL2C/EA interface read-request bytes; not physical DRAM",
        "copy_domain": "mixed read+write device-copy proxy; not a strict bandwidth ceiling",
        "baseline_unprofiled": load_baseline(root),
        "pairs": assembled,
        "shape_inventory": {"path": str(root / "linear-tensor-shapes.json"), "unique_shape_count": len(shape_counts), "format_counts": dict(Counter(fmt for _k, _n, fmt in shape_counts for _ in range(shape_counts[(_k, _n, fmt)])))},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = assemble(args.root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except SummaryError as exc:
        print(f"phase87 stage0 summary: FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2
    print(f"PASS {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
