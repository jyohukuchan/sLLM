#!/usr/bin/env python3
"""Aggregate the two fixed Phase 5 render/tokenize rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, canonical_bytes  # noqa: E402
import engine_performance_common as contracts  # noqa: E402
import run_engine_performance_render as render  # noqa: E402
import run_engine_performance as direct_runner  # noqa: E402
import aggregate_engine_performance as direct_aggregate  # noqa: E402


METRIC_ORDER = (
    "ttft_ns", "prefill_ns", "tpot_ns", "decode_token_per_s", "e2e_ns",
    "resident_vram_bytes", "peak_vram_bytes",
)


def _fail(message: str) -> None:
    raise ContractError(message)


def _metric_values(result: dict[str, Any]) -> dict[str, list[int | float]]:
    values: dict[str, list[int | float]] = {metric: [] for metric in METRIC_ORDER}
    for sample in result["measured"]["samples"]:
        derived = sample["derived"]
        values["ttft_ns"].append(derived["ttft_ns"])
        values["prefill_ns"].append(derived["prefill_ns"])
        if derived["tpot_ns"]:
            values["tpot_ns"].append(contracts.statistics_median(derived["tpot_ns"]))
        if derived["decode_tokens_per_second"] is not None:
            values["decode_token_per_s"].append(derived["decode_tokens_per_second"])
        values["e2e_ns"].append(derived["e2e_ns"])
    values["resident_vram_bytes"].append(result["memory"]["resident_vram_bytes"])
    values["peak_vram_bytes"].append(result["memory"]["peak_vram_bytes"])
    return values


def _validate_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
    row: dict[str, Any],
    matrix_path: Path,
    matrix_digest: str,
    *,
    verify_external_digests: bool = True,
) -> tuple[dict[str, Any], str, str, str, str, str, dict[str, Any]]:
    contracts.schema_validate(manifest, render.SCHEMA_PATH, "render/tokenize performance evidence manifest", "manifest")
    contracts.validate_manifest_evidence(manifest)
    if manifest["state"] != "PASS" or manifest["failure_reason"] is not None or manifest["required"] is not False or manifest["claims"] != render.CLAIMS:
        _fail(f"{manifest_path}: failed/non-baseline manifest cannot be aggregated")
    if manifest["row_id"] != row["row_id"] or manifest["matrix"]["matrix_id"] != render.VERSION or manifest["matrix"]["sha256"] != matrix_digest:
        _fail(f"{manifest_path}: stale row or matrix identity")
    if Path(manifest["matrix"]["path"]).resolve() != matrix_path.resolve():
        _fail(f"{manifest_path}: matrix path is not the selected fixed matrix")
    model = render.expected_model()
    if manifest["model_lock"]["fingerprint"] != model["lock_fingerprint"]:
        _fail(f"{manifest_path}: model lock fingerprint does not match the row")
    expected_device = render.expected_device(row["target"])
    build_identity = manifest["build_identity"]
    source_root = Path(build_identity["source_root"])
    if not source_root.is_dir():
        _fail(f"{manifest_path}: build identity source root is unavailable")
    try:
        direct_runner._validate_source_base(source_root, build_identity["source_base_revision"])
        direct_runner._validate_semantic_tree(source_root, build_identity["semantic_tree"])
    except (ContractError, OSError, ValueError) as exc:
        _fail(f"{manifest_path}: build identity source fields are stale: {exc}")
    for phase in ("pre", "post"):
        observation = manifest["observations"][phase]
        direct_aggregate._validate_recorded_observation(
            observation, row["target"], phase, manifest_path,
        )
        if observation["selected_device"] != expected_device:
            _fail(f"{manifest_path}: {phase} selected-device evidence is wrong")
    if not direct_runner._observations_have_stable_authorization(
        manifest["observations"]["pre"], manifest["observations"]["post"],
    ):
        _fail(f"{manifest_path}: pre/post health/process authorization differs")
    expected_execution = {"exit_code": 0, "timed_out": False, "timeout_seconds": row["timeout_seconds"], "stderr_bytes": 0, "term_sent": False, "kill_sent": False, "process_group_gone": True}
    if manifest["execution"] != expected_execution:
        _fail(f"{manifest_path}: execution is timed out, noisy, or not cleaned")
    if manifest["cleanup"] != {"pre_process_clean": True, "post_process_clean": True, "process_group_gone": True, "retryable_cleanup": 0, "durable_quarantine": 0}:
        _fail(f"{manifest_path}: cleanup is not fail-closed")
    checks = manifest["evidence"].get("checks", {})
    required_checks = ("exact_identity", "static_identity_unchanged", "profile_unchanged", "limits_unchanged", "performance_level_unchanged", "vram_auxiliary_complete", "process_ownership", "loader_paths_verified", "process_group_cleanup")
    if any(checks.get(name) is not True for name in required_checks) or checks.get("explicit_violation") is not False or checks.get("monitor_errors") != 0:
        _fail(f"{manifest_path}: runtime health evidence is not fail-closed")
    raw_path = Path(manifest["raw_artifact"]["path"])
    raw, raw_bytes, raw_sha = contracts.read_json(raw_path, "raw render/tokenize result", direct_runner_max_raw_bytes())
    if raw_sha != manifest["raw_artifact"]["sha256"] or len(raw_bytes) != manifest["raw_artifact"]["bytes"]:
        _fail(f"{manifest_path}: raw result digest/size was tampered")
    if verify_external_digests:
        binary = Path(manifest["binary"]["path"])
        lock = Path(manifest["model_lock"]["path"])
        cache = Path(manifest["model_cache"]["path"])
        validated_build, build_manifest_sha = direct_runner._validate_build_manifest(Path(build_identity["path"]), binary, row["target"], source_root)
        if build_manifest_sha != build_identity["sha256"] or any(build_identity[key] != validated_build[key] for key in ("source_root", "source_base_revision", "semantic_tree", "build_inputs_digest", "build_configuration", "target", "backend", "rocm_release", "rocm_root", "binary_sha256")):
            _fail(f"{manifest_path}: build identity manifest is stale or not bound to the evidence manifest")
        binary_sha = contracts.sha256_file(binary, "benchmark binary")
        lock_sha = contracts.sha256_file(lock, "model lock")
        cache_sha = contracts.cache_digest(cache)
        lock_document, _, _ = contracts.read_json(lock, "model lock")
        if not isinstance(lock_document, dict) or not isinstance(lock_document.get("model"), dict) or lock_document["model"].get("repo_id") != model["repo_id"] or lock_document["model"].get("resolved_revision") != model["resolved_revision"] or lock_document.get("fingerprint") != model["lock_fingerprint"]:
            _fail(f"{manifest_path}: model lock content is stale or wrong")
        if binary_sha != manifest["binary"]["sha256"] or binary.stat().st_size != manifest["binary"]["bytes"]:
            _fail(f"{manifest_path}: benchmark binary digest/size was tampered")
        if lock_sha != manifest["model_lock"]["sha256"] or cache_sha != manifest["model_cache"]["sha256"]:
            _fail(f"{manifest_path}: model lock/cache digest was tampered")
    else:
        binary_sha = manifest["binary"]["sha256"]
        lock_sha = manifest["model_lock"]["sha256"]
        cache_sha = manifest["model_cache"]["sha256"]
    render.validate_cli_result(raw, row)
    return raw, raw_sha, binary_sha, lock_sha, manifest["model_lock"]["fingerprint"], cache_sha, build_identity


def direct_runner_max_raw_bytes() -> int:
    """Keep the raw-artifact bound identical to the health runner."""
    return direct_runner.MAX_RAW_BYTES


def _graph_bytes(rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["order", "row_id", "model_size", "case_id", "target", "input_tokens", "requested_output_tokens", "metric", "median", "p10", "p90", "mad", "min", "max", "count"])
    for row in rows:
        for metric in METRIC_ORDER:
            stats = row["metrics"][metric]
            writer.writerow([row["order"], row["row_id"], row["model_size"], row["case_id"], row["target"], row["input_tokens"], row["requested_output_tokens"], metric, stats["median"], stats["p10"], stats["p90"], stats["mad"], stats["min"], stats["max"], stats["count"]])
    return stream.getvalue().encode("utf-8")


def aggregate_manifests(
    manifest_paths: list[Path],
    output_dir: Path,
    *,
    matrix_path: Path = render.MATRIX_PATH,
    verify_external_digests: bool = True,
) -> dict[str, Any]:
    matrix, matrix_digest = render.load_matrix(matrix_path)
    rows_by_id = {row["row_id"]: row for row in matrix["rows"]}
    expected_ids = [row["row_id"] for row in matrix["rows"]]
    if len(manifest_paths) != 2 or len({path.resolve() for path in manifest_paths}) != 2:
        _fail("render/tokenize aggregate requires exactly two distinct manifests")
    collected: dict[str, dict[str, Any]] = {}
    binary_by_target: dict[str, str] = {}
    raw_hashes: set[str] = set()
    lock_sha: str | None = None
    lock_fingerprint: str | None = None
    cache_sha: str | None = None
    source_identity: dict[str, Any] | None = None
    build_identity_by_target: dict[str, dict[str, Any]] = {}
    for manifest_path in manifest_paths:
        manifest, _, manifest_sha = contracts.read_json(manifest_path, "render/tokenize performance evidence manifest")
        if not isinstance(manifest, dict) or manifest.get("row_id") not in rows_by_id:
            _fail(f"{manifest_path}: row is unknown or stale")
        row_id = manifest["row_id"]
        if row_id in collected:
            _fail(f"duplicate render/tokenize performance row: {row_id}")
        raw, raw_sha, binary_sha, this_lock_sha, fingerprint, this_cache_sha, build_identity = _validate_manifest(manifest_path, manifest, rows_by_id[row_id], matrix_path, matrix_digest, verify_external_digests=verify_external_digests)
        if raw_sha in raw_hashes:
            _fail(f"duplicate raw render/tokenize result digest: {raw_sha}")
        raw_hashes.add(raw_sha)
        row = rows_by_id[row_id]
        target = row["target"]
        this_source_identity = contracts.aggregate_source_identity(build_identity)
        this_build_identity = contracts.aggregate_build_identity(build_identity)
        if source_identity is not None and this_source_identity != source_identity:
            _fail("mixed source identity across render/tokenize rows")
        if target in build_identity_by_target and this_build_identity != build_identity_by_target[target]:
            _fail(f"mixed complete build identity for {target}")
        if target in binary_by_target and binary_by_target[target] != binary_sha:
            _fail(f"mixed benchmark binary identity for {target}")
        if lock_sha is not None and this_lock_sha != lock_sha:
            _fail("mixed model lock identity across render/tokenize rows")
        if lock_fingerprint is not None and fingerprint != lock_fingerprint:
            _fail("mixed model fingerprint across render/tokenize rows")
        if cache_sha is not None and this_cache_sha != cache_sha:
            _fail("mixed model cache identity across render/tokenize rows")
        binary_by_target[target] = binary_sha
        source_identity = this_source_identity
        build_identity_by_target[target] = this_build_identity
        lock_sha, lock_fingerprint, cache_sha = this_lock_sha, fingerprint, this_cache_sha
        values = _metric_values(raw)
        collected[row_id] = {
            "order": row["order"], "row_id": row_id, "model_size": row["model_size"], "case_id": row["case_id"], "target": target,
            "input_token_ids": list(render.INPUT_TOKEN_IDS), "input_tokens": row["input_tokens"], "requested_output_tokens": row["requested_output_tokens"],
            "manifest_sha256": manifest_sha, "raw_result_sha256": raw_sha, "binary_sha256": binary_sha,
            "model_lock_sha256": this_lock_sha, "model_lock_fingerprint": fingerprint, "warmup_count": 3, "sample_count": 10,
            "metrics": {metric: contracts.summary_stats(values[metric]) for metric in METRIC_ORDER},
        }
    if set(collected) != set(expected_ids):
        _fail(f"render/tokenize aggregate has missing rows: {sorted(set(expected_ids) - set(collected))}")
    ordered_rows = [collected[row_id] for row_id in expected_ids]
    graph = _graph_bytes(ordered_rows)
    output_dir = output_dir.resolve()
    graph_path = output_dir / "graph.csv"
    summary_path = output_dir / "summary.json"
    if source_identity is None:
        _fail("render/tokenize aggregate has no common source identity")
    summary = {
        "benchmark_schema_version": render.AGGREGATE_VERSION,
        "state": "PASS",
        "claims": dict(render.CLAIMS),
        "matrix": {"path": str(matrix_path), "matrix_id": render.VERSION, "sha256": matrix_digest},
        "expected_rows": expected_ids,
        "rows": ordered_rows,
        "identity": {
            "source": source_identity,
            "build_identity_by_target": {target: build_identity_by_target[target] for target in render.TARGETS},
            "binary_sha256_by_target": {target: binary_by_target[target] for target in render.TARGETS},
            "model_lock_sha256": lock_sha,
            "model_lock_fingerprint": lock_fingerprint,
            "model_cache_sha256": cache_sha,
        },
        "counts": {"expected_rows": 2, "collected_rows": 2, "passed_rows": 2, "expected_samples": 20, "collected_samples": 20},
        "graph_csv": {"path": str(graph_path), "sha256": hashlib.sha256(graph).hexdigest(), "bytes": len(graph), "row_count": len(ordered_rows) * len(METRIC_ORDER)},
    }
    contracts.schema_validate(summary, render.SCHEMA_PATH, "render/tokenize performance aggregate")
    summary_bytes = canonical_bytes(summary)
    contracts.publish_aggregate_bundle(
        output_dir,
        {
            "graph.csv": graph,
            "summary.json": summary_bytes,
            "graph.csv.sha256": f"{hashlib.sha256(graph).hexdigest()}  graph.csv\n".encode("ascii"),
            "summary.json.sha256": f"{hashlib.sha256(summary_bytes).hexdigest()}  summary.json\n".encode("ascii"),
        },
        "render/tokenize aggregate",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=render.MATRIX_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = aggregate_manifests(args.manifests, args.output_dir, matrix_path=args.matrix)
    except (ContractError, OSError, ValueError) as exc:
        print(f"engine-performance-render aggregate: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"engine-performance-render aggregate: {summary['state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
