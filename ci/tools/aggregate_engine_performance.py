#!/usr/bin/env python3
"""Aggregate every fixed Phase 5 direct-engine row into compact JSON/CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, canonical_bytes  # noqa: E402
from engine_performance_common import (  # noqa: E402
    AGGREGATE_SCHEMA_PATH,
    AGGREGATE_VERSION,
    CLAIMS,
    DIRECT_SCHEMA_PATH,
    MATRIX_PATH,
    TARGETS,
    aggregate_build_identity,
    aggregate_source_identity,
    cache_digest,
    expected_device,
    expected_model,
    load_matrix,
    metric_values,
    publish_aggregate_bundle,
    read_json,
    resolved_row,
    schema_validate,
    sha256_file,
    summary_stats,
    validate_cli_result,
    verify_aggregate_bundle,
)
import run_engine_performance as runner  # noqa: E402


METRIC_ORDER = (
    "ttft_ns", "prefill_ns", "tpot_ns", "decode_token_per_s", "prefill_token_per_s",
    "e2e_ns", "resident_vram_bytes", "peak_vram_bytes",
)


def _fail(message: str) -> None:
    raise ContractError(message)


def _external_sha(path: Path, label: str) -> str:
    return sha256_file(path, label)


def _validate_recorded_observation(
    observation: Any, target: str, phase: str, manifest_path: Path,
) -> dict[str, Any]:
    """Validate producer-emitted process evidence without consulting current env."""
    expected_process_keys = {
        "available", "reliable", "state", "gpu_processes", "residual_runner_children",
    }
    if not isinstance(observation, dict) or set(observation) != {"selected_device", "health", "process"}:
        _fail(f"{manifest_path}: {phase} health observation is incomplete")
    if observation["selected_device"] != expected_device(target):
        _fail(f"{manifest_path}: {phase} selected-device identity drifted")
    if observation["health"] != {
        "available": True, "reliable": True, "state": "OK", "ras_uncorrectable_count": 0,
    }:
        _fail(f"{manifest_path}: {phase} health changed or is not reliable")
    process = observation["process"]
    if not isinstance(process, dict) or set(process) != expected_process_keys or any(
        process.get(key) != value for key, value in {
            "available": True, "reliable": True, "state": "CLEAN",
            "residual_runner_children": [],
        }.items()
    ):
        _fail(f"{manifest_path}: {phase} process evidence is not clean")
    gpu_processes = process["gpu_processes"]
    if not isinstance(gpu_processes, list):
        _fail(f"{manifest_path}: {phase} GPU process evidence is malformed")
    if not gpu_processes:
        return observation
    marker = gpu_processes[0]
    if not isinstance(marker, dict) or set(marker) != {"allowlisted_pids"}:
        _fail(f"{manifest_path}: {phase} GPU process allowlist is malformed")
    allowed_pids = marker["allowlisted_pids"]
    if not isinstance(allowed_pids, list) or not allowed_pids or any(
        not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or pid > runner.MAX_LINUX_PID
        for pid in allowed_pids
    ) or len(set(allowed_pids)) != len(allowed_pids):
        _fail(f"{manifest_path}: {phase} GPU process allowlist PID identity is malformed")
    records: list[Any] = []
    for entry in gpu_processes[1:]:
        if not isinstance(entry, dict) or set(entry) != {"record", "record_sha256"}:
            _fail(f"{manifest_path}: {phase} GPU process record evidence is malformed")
        records.append(entry["record"])
    try:
        canonical = runner._allowed_process_observation(records, tuple(allowed_pids))
    except (ContractError, ValueError) as exc:
        _fail(f"{manifest_path}: {phase} GPU process evidence is invalid: {exc}")
    if canonical != gpu_processes:
        _fail(f"{manifest_path}: {phase} GPU process evidence is not canonical")
    return observation


def _validate_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
    row: dict[str, Any],
    matrix_path: Path,
    matrix_digest: str,
    *,
    verify_external_digests: bool = True,
) -> tuple[dict[str, Any], str, str, str, str, str, dict[str, Any]]:
    schema_validate(manifest, DIRECT_SCHEMA_PATH, "performance evidence manifest", "manifest")
    if manifest["state"] != "PASS" or manifest["failure_reason"] is not None or manifest["required"] is not False or manifest["claims"] != CLAIMS:
        _fail(f"{manifest_path}: failed/non-baseline manifest cannot be aggregated")
    if manifest["row_id"] != row["row_id"] or manifest["matrix"]["matrix_id"] != "engine-performance-direct-v1" or manifest["matrix"]["sha256"] != matrix_digest:
        _fail(f"{manifest_path}: stale row or matrix identity")
    if Path(manifest["matrix"]["path"]).resolve() != matrix_path.resolve():
        _fail(f"{manifest_path}: matrix path is not the selected fixed matrix")
    model = expected_model(row["model_size"])
    if manifest["model_lock"]["fingerprint"] != model["lock_fingerprint"]:
        _fail(f"{manifest_path}: model lock fingerprint does not match the row")
    build_identity = manifest["build_identity"]
    if not isinstance(build_identity.get("source_root"), str) or not isinstance(build_identity.get("source_base_revision"), str) or not isinstance(build_identity.get("semantic_tree"), str):
        _fail(f"{manifest_path}: build identity v2 source fields are missing")
    if not re.fullmatch(r"[0-9a-f]{40}", build_identity["source_base_revision"]) or not re.fullmatch(r"[0-9a-f]{40}", build_identity["semantic_tree"]):
        _fail(f"{manifest_path}: build identity v2 source fields are malformed")
    source_root = Path(build_identity["source_root"])
    if not source_root.is_dir():
        _fail(f"{manifest_path}: build identity source root is unavailable")
    runner._validate_source_base(source_root, build_identity["source_base_revision"])
    runner._validate_semantic_tree(source_root, build_identity["semantic_tree"])
    observations = manifest["observations"]
    for phase in ("pre", "post"):
        _validate_recorded_observation(observations[phase], row["target"], phase, manifest_path)
    if not runner._observations_have_stable_authorization(observations["pre"], observations["post"]):
        _fail(f"{manifest_path}: pre/post health or process authorization differs")
    if manifest["execution"] != {"exit_code": 0, "timed_out": False, "timeout_seconds": row["timeout_seconds"], "stderr_bytes": 0, "term_sent": False, "kill_sent": False, "process_group_gone": True}:
        _fail(f"{manifest_path}: execution is timed out, noisy, or not cleaned")
    if manifest["cleanup"] != {"pre_process_clean": True, "post_process_clean": True, "process_group_gone": True, "retryable_cleanup": 0, "durable_quarantine": 0}:
        _fail(f"{manifest_path}: cleanup is not fail-closed")
    binary = Path(manifest["binary"]["path"])
    lock = Path(manifest["model_lock"]["path"])
    cache = Path(manifest["model_cache"]["path"])
    raw_path = Path(manifest["raw_artifact"]["path"])
    raw, raw_bytes, raw_sha = read_json(raw_path, "raw benchmark result")
    if raw_sha != manifest["raw_artifact"]["sha256"] or len(raw_bytes) != manifest["raw_artifact"]["bytes"]:
        _fail(f"{manifest_path}: raw result digest/size was tampered")
    if verify_external_digests:
        validated_build, build_manifest_sha = runner._validate_build_manifest(Path(build_identity["path"]), binary, row["target"], source_root)
        if build_manifest_sha != build_identity["sha256"] or any(build_identity[key] != validated_build[key] for key in ("source_root", "source_base_revision", "semantic_tree", "build_inputs_digest", "build_configuration", "target", "backend", "rocm_release", "rocm_root", "binary_sha256")):
            _fail(f"{manifest_path}: build identity manifest is stale or not bound to the evidence manifest")
        binary_sha = _external_sha(binary, "benchmark binary")
        lock_sha = _external_sha(lock, "model lock")
        cache_sha = cache_digest(cache)
        lock_document, _, _ = read_json(lock, "model lock")
        if not isinstance(lock_document, dict) or not isinstance(lock_document.get("model"), dict):
            _fail(f"{manifest_path}: model lock identity is malformed")
        lock_model = lock_document["model"]
        if lock_model.get("repo_id") != model["repo_id"] or lock_model.get("resolved_revision") != model["resolved_revision"] or lock_document.get("fingerprint") != model["lock_fingerprint"]:
            _fail(f"{manifest_path}: model lock content is stale or wrong")
        if binary_sha != manifest["binary"]["sha256"] or binary.stat().st_size != manifest["binary"]["bytes"]:
            _fail(f"{manifest_path}: benchmark binary digest/size was tampered")
        if lock_sha != manifest["model_lock"]["sha256"]:
            _fail(f"{manifest_path}: model lock digest was tampered")
        if cache_sha != manifest["model_cache"]["sha256"]:
            _fail(f"{manifest_path}: model cache digest was tampered")
    else:
        binary_sha = manifest["binary"]["sha256"]
        lock_sha = manifest["model_lock"]["sha256"]
        cache_sha = manifest["model_cache"]["sha256"]
    validate_cli_result(raw, row)
    return raw, raw_sha, binary_sha, lock_sha, manifest["model_lock"]["fingerprint"], cache_sha, build_identity


def _graph_bytes(rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["order", "row_id", "model_size", "case_id", "target", "input_tokens", "requested_output_tokens", "metric", "median", "p10", "p90", "mad", "min", "max", "count"])
    for row in rows:
        for metric in METRIC_ORDER:
            stats = row["metrics"][metric]
            writer.writerow([
                row["order"], row["row_id"], row["model_size"], row["case_id"], row["target"],
                row["input_tokens"], row["requested_output_tokens"], metric,
                stats["median"], stats["p10"], stats["p90"], stats["mad"], stats["min"], stats["max"], stats["count"],
            ])
    return stream.getvalue().encode("utf-8")


def aggregate_manifests(
    manifest_paths: list[Path],
    output_dir: Path,
    *,
    matrix_path: Path = MATRIX_PATH,
    verify_external_digests: bool = True,
) -> dict[str, Any]:
    matrix, matrix_digest = load_matrix(matrix_path)
    rows_by_id = {row["row_id"]: row for row in matrix["rows"]}
    expected_ids = [row["row_id"] for row in matrix["rows"]]
    if len(manifest_paths) != len(expected_ids):
        _fail(f"performance aggregate requires exactly {len(expected_ids)} manifests, got {len(manifest_paths)}")
    if len({path.resolve() for path in manifest_paths}) != len(manifest_paths):
        _fail("performance aggregate has duplicate manifest paths")
    collected: dict[str, dict[str, Any]] = {}
    binary_by_target: dict[str, str] = {}
    lock_by_model: dict[str, str] = {}
    fingerprint_by_model: dict[str, str] = {}
    cache_by_model: dict[str, str] = {}
    source_identity: dict[str, Any] | None = None
    build_identity_by_target: dict[str, dict[str, Any]] = {}
    raw_hashes: set[str] = set()
    for manifest_path in manifest_paths:
        manifest, _, manifest_sha = read_json(manifest_path, "performance evidence manifest")
        if not isinstance(manifest, dict):
            _fail(f"{manifest_path}: manifest is not an object")
        row_id = manifest.get("row_id")
        if row_id not in rows_by_id:
            _fail(f"{manifest_path}: row is unknown or stale")
        if row_id in collected:
            _fail(f"duplicate performance row: {row_id}")
        raw, raw_sha, binary_sha, lock_sha, fingerprint, cache_sha, build_identity = _validate_manifest(manifest_path, manifest, rows_by_id[row_id], matrix_path, matrix_digest, verify_external_digests=verify_external_digests)
        if raw_sha in raw_hashes:
            _fail(f"duplicate raw result digest: {raw_sha}")
        raw_hashes.add(raw_sha)
        row = rows_by_id[row_id]
        resolved = resolved_row(row)
        target = row["target"]
        model_size = row["model_size"]
        this_source_identity = aggregate_source_identity(build_identity)
        this_build_identity = aggregate_build_identity(build_identity)
        if source_identity is not None and this_source_identity != source_identity:
            _fail("mixed source identity across performance rows")
        if target in build_identity_by_target and this_build_identity != build_identity_by_target[target]:
            _fail(f"mixed complete build identity for {target}")
        if target in binary_by_target and binary_by_target[target] != binary_sha:
            _fail(f"mixed benchmark binary identity for {target}")
        if model_size in lock_by_model and lock_by_model[model_size] != lock_sha:
            _fail(f"mixed model lock identity for {model_size}")
        if model_size in fingerprint_by_model and fingerprint_by_model[model_size] != fingerprint:
            _fail(f"mixed model fingerprint for {model_size}")
        if model_size in cache_by_model and cache_by_model[model_size] != cache_sha:
            _fail(f"mixed model cache identity for {model_size}")
        binary_by_target[target] = binary_sha
        source_identity = this_source_identity
        build_identity_by_target[target] = this_build_identity
        lock_by_model[model_size] = lock_sha
        fingerprint_by_model[model_size] = fingerprint
        cache_by_model[model_size] = cache_sha
        values = metric_values(raw, row)
        collected[row_id] = {
            "order": row["order"], "row_id": row_id, "model_size": model_size, "case_id": row["case_id"],
            "input_token_sequence": row["input_token_sequence"], "input_token_ids": resolved["input_token_ids"],
            "target": target,
            "input_tokens": row["input_tokens"], "requested_output_tokens": row["requested_output_tokens"],
            "timeout_seconds": row["timeout_seconds"],
            "manifest_sha256": manifest_sha, "raw_result_sha256": raw_sha, "binary_sha256": binary_sha,
            "model_lock_sha256": lock_sha, "model_lock_fingerprint": fingerprint,
            "warmup_count": 3, "sample_count": len(raw["measured"]["samples"]),
            "metrics": {metric: summary_stats(values[metric]) for metric in METRIC_ORDER},
        }
    if set(collected) != set(expected_ids):
        _fail(f"performance aggregate has missing rows: {sorted(set(expected_ids) - set(collected))}")
    ordered_rows = [collected[row_id] for row_id in expected_ids]
    graph = _graph_bytes(ordered_rows)
    output_dir = output_dir.resolve()
    graph_path = output_dir / "graph.csv"
    summary_path = output_dir / "summary.json"
    if source_identity is None:
        _fail("performance aggregate has no common source identity")
    summary = {
        "benchmark_schema_version": AGGREGATE_VERSION,
        "state": "PASS",
        "claims": dict(CLAIMS),
        "matrix": {"path": str(matrix_path), "matrix_id": "engine-performance-direct-v1", "sha256": matrix_digest},
        "expected_rows": expected_ids,
        "rows": ordered_rows,
        "identity": {
            "source": source_identity,
            "build_identity_by_target": {target: build_identity_by_target[target] for target in TARGETS},
            "binary_sha256_by_target": {target: binary_by_target[target] for target in TARGETS},
            "model_lock_sha256_by_model": {size: lock_by_model[size] for size in ("2B", "4B", "9B")},
            "model_lock_fingerprint_by_model": {size: fingerprint_by_model[size] for size in ("2B", "4B", "9B")},
            "model_cache_sha256_by_model": {size: cache_by_model[size] for size in ("2B", "4B", "9B")},
        },
        "counts": {"expected_rows": 22, "collected_rows": len(ordered_rows), "passed_rows": len(ordered_rows), "expected_samples": 220, "collected_samples": sum(row["sample_count"] for row in ordered_rows)},
        "graph_csv": {"path": str(graph_path), "sha256": hashlib.sha256(graph).hexdigest(), "bytes": len(graph), "row_count": len(ordered_rows) * len(METRIC_ORDER)},
    }
    schema_validate(summary, AGGREGATE_SCHEMA_PATH, "performance aggregate")
    summary_bytes = canonical_bytes(summary)
    summary_sidecar = f"{hashlib.sha256(summary_bytes).hexdigest()}  summary.json\n".encode("ascii")
    graph_sidecar = f"{hashlib.sha256(graph).hexdigest()}  graph.csv\n".encode("ascii")
    publish_aggregate_bundle(
        output_dir,
        {
            "graph.csv": graph,
            "summary.json": summary_bytes,
            "graph.csv.sha256": graph_sidecar,
            "summary.json.sha256": summary_sidecar,
        },
        "performance aggregate",
    )
    verify_aggregate_bundle(output_dir, "performance aggregate")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = aggregate_manifests(args.manifests, args.output_dir, matrix_path=args.matrix)
    except (ContractError, OSError, ValueError) as exc:
        print(f"engine-performance aggregate: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"engine-performance aggregate: {summary['state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
