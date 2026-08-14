#!/usr/bin/env python3
"""Run one bounded render/tokenize Phase 5 row.

The render lane owns its closed two-row matrix and result semantics while it
reuses the direct runner's bounded process, AMD-SMI health, loader evidence,
and cleanup implementation.  This keeps the health boundary shared without
making a render result look like a pretokenized direct result.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, canonical_bytes  # noqa: E402
import engine_performance_common as contracts  # noqa: E402
import run_engine_performance as direct_runner  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "ci/matrix/engine-performance-render-v1.json"
SCHEMA_PATH = ROOT / "ci/schema/engine-performance-render-v1.schema.json"
VERSION = "engine-performance-render-v1"
AGGREGATE_VERSION = "engine-performance-render-aggregate-v1"
MATRIX_RELATIVE = "ci/matrix/engine-performance-render-v1.json"
TARGETS = ("gfx1030", "gfx1201")
ROW_IDS = tuple(f"engine-performance-render-4b-{target}-chat-hello" for target in TARGETS)
INPUT_TOKEN_IDS = [248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
RENDERED_PROMPT = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
RENDERED_PROMPT_SHA256 = "13071b0f0e23e97681f6f39247dbb715973302580660a81dbae994fe8064e7d3"
MODEL_SIZE = "4B"
CASE_ID = "chat-hello"
REQUESTED_OUTPUT_TOKENS = 17
TIMEOUT_SECONDS = 5400
CLAIMS = dict(contracts.CLAIMS)
PROTOCOL = {
    "backend": "hip",
    "dtype": "BF16",
    "batch_size": 1,
    "warmup_requests": 3,
    "measured_requests": 10,
    "max_new_tokens": REQUESTED_OUTPUT_TOKENS,
    "greedy": True,
    "thinking": "disabled",
    "add_generation_prompt": True,
    "message": "user:Hello",
    "stop_token_ids": [248046, 248044],
    "visible_stop_tokens": False,
}


def _fail(message: str) -> None:
    raise ContractError(message)


def _expected_rows() -> list[dict[str, Any]]:
    return [
        {
            "order": index,
            "row_id": ROW_IDS[index],
            "model_size": MODEL_SIZE,
            "case_id": CASE_ID,
            "message": "user:Hello",
            "thinking": "disabled",
            "add_generation_prompt": True,
            "input_token_ids": list(INPUT_TOKEN_IDS),
            "input_tokens": len(INPUT_TOKEN_IDS),
            "requested_output_tokens": REQUESTED_OUTPUT_TOKENS,
            "target": target,
            "timeout_seconds": TIMEOUT_SECONDS,
        }
        for index, target in enumerate(TARGETS)
    ]


def _expected_render_contract() -> dict[str, Any]:
    return {
        "renderer_version": 1,
        "template_filename": "chat_template.jinja",
        "template_sha256": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        "tokenizer_sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "message": {"role": "user", "content": "Hello"},
        "thinking": "disabled",
        "add_generation_prompt": True,
        "rendered_prompt": RENDERED_PROMPT,
        "rendered_prompt_sha256": RENDERED_PROMPT_SHA256,
        "input_token_ids": list(INPUT_TOKEN_IDS),
    }


def validate_matrix_document(matrix: Any) -> dict[str, Any]:
    contracts.schema_validate(matrix, SCHEMA_PATH, "render/tokenize performance matrix")
    if not isinstance(matrix, dict):
        _fail("render/tokenize matrix is not an object")
    expected_top = {
        "schema_version", "matrix_id", "revision", "suite_id", "tier", "required", "claims",
        "protocol", "render_contract", "targets", "model", "rows",
    }
    if set(matrix) != expected_top:
        _fail("render/tokenize matrix keys differ from the closed contract")
    if matrix["schema_version"] != VERSION or matrix["matrix_id"] != VERSION or matrix["revision"] != 1:
        _fail("render/tokenize matrix version or identity is stale")
    if matrix["suite_id"] != "h0-engine-performance-render-tokenize-contract" or matrix["tier"] != "tier_p1" or matrix["required"] is not False:
        _fail("render/tokenize matrix suite/tier/required contract drifted")
    if matrix["claims"] != CLAIMS or matrix["protocol"] != PROTOCOL:
        _fail("render/tokenize matrix claims or protocol drifted")
    if matrix["render_contract"] != _expected_render_contract():
        _fail("render/tokenize renderer/tokenizer contract drifted")
    expected_targets = [dict(contracts.expected_device(target), order=index) for index, target in enumerate(TARGETS)]
    if matrix["targets"] != expected_targets:
        _fail("render/tokenize target mapping is not canonical")
    expected_model = dict(contracts.expected_model(MODEL_SIZE), order=0, model_size=MODEL_SIZE)
    if matrix["model"] != expected_model:
        _fail("render/tokenize model identity is not the locked Qwen3.5-4B contract")
    if matrix["rows"] != _expected_rows():
        _fail("render/tokenize rows are missing, reordered, duplicated, or changed")
    return matrix


def load_matrix(path: Path = MATRIX_PATH) -> tuple[dict[str, Any], str]:
    matrix, _, digest = contracts.read_json(path, "render/tokenize performance matrix", 8 * 1024 * 1024)
    return validate_matrix_document(matrix), digest


def resolved_row(row: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(row)
    expected = {item["row_id"]: item for item in _expected_rows()}.get(row.get("row_id"))
    if expected is None or row != expected:
        _fail(f"row is not an exact member of the render/tokenize matrix: {row.get('row_id')}")
    return row


def expected_device(target: str) -> dict[str, Any]:
    if target not in TARGETS:
        _fail(f"unknown canonical render target: {target}")
    return contracts.expected_device(target)


def expected_model() -> dict[str, str]:
    return contracts.expected_model(MODEL_SIZE)


def _expected_command(binary: Path, row: Mapping[str, Any], lock: Path, cache: Path) -> list[str]:
    row = resolved_row(row)
    return [
        str(binary), "benchmark", "--lane", "render-tokenize", "--lock", str(lock), "--cache", str(cache),
        "--row-id", row["row_id"], "--model-size", row["model_size"], "--case-id", row["case_id"],
        "--message", row["message"], "--thinking", row["thinking"],
        "--max-new-tokens", str(row["requested_output_tokens"]), "--device-index", "0",
        "--target", row["target"], "--greedy", "--warmups", "3", "--measured", "10",
    ]


def _validate_snapshot_accounting(snapshot: Any, label: str) -> None:
    contracts.validate_snapshot(snapshot, label)
    categories = [snapshot[name] for name in ("model_resident", "request_state", "workspace")]
    if sum(item["current_bytes"] for item in categories) != snapshot["current_bytes"]:
        _fail(f"{label} category current bytes do not sum to total current bytes")
    if snapshot["high_water_bytes"] < snapshot["current_bytes"]:
        _fail(f"{label} total high-water bytes are below current bytes")
    for category in categories:
        if category["high_water_bytes"] < category["current_bytes"]:
            _fail(f"{label} category high-water bytes are below current bytes")


def _control_fields(value: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return value["tokens"], value["stop"], value["audit"]


def _validate_control_equality(control: Mapping[str, Any], sample: Mapping[str, Any]) -> None:
    control_tokens, control_stop, control_audit = _control_fields(control)
    sample_tokens, sample_stop, sample_audit = _control_fields(sample)
    if control_tokens != sample_tokens or control_stop != sample_stop:
        _fail("render/tokenize sample token or stop control differs from correctness control")
    for field in (
        "selected_backend", "target", "device_index", "model_fingerprint", "plan_digest",
        "fallback_used", "all_dispatches_hip", "submission_count", "kernel_dispatch_count",
        "segment_count", "boundary_count",
    ):
        if control_audit[field] != sample_audit[field]:
            _fail(f"render/tokenize sample audit control differs at {field}")


EXPECTED_COMPARISON = {
    "mode": "exact",
    "scope": "every_warmup_and_measured_sample",
    "token_fields": ["input_token_ids", "generated_token_ids", "visible_token_ids", "decode_input_token_ids"],
    "stop_fields": ["version", "reason_version", "kind", "token_id"],
    "dispatch_fields": [
        "selected_backend", "target", "device_index", "model_fingerprint", "plan_digest",
        "fallback_used", "all_dispatches_hip", "submission_count", "kernel_dispatch_count",
        "segment_count", "boundary_count",
    ],
    "dispatch_count_rule": "exact_when_token_and_stop_fields_match",
}


def _fallback_observation(target: str) -> dict[str, Any]:
    return direct_runner._fallback_observation(target)


def _fallback_phase_evidence(target: str) -> dict[str, Any]:
    return direct_runner._fallback_phase_evidence(target)


def _validate_manifest(manifest: Mapping[str, Any], label: str) -> None:
    contracts.schema_validate(dict(manifest), SCHEMA_PATH, label, "manifest")
    contracts.validate_manifest_evidence(dict(manifest))


def _durable_failure_manifest(
    row: Mapping[str, Any], matrix_path: Path, matrix_digest: str, output_dir: Path, reason: str,
    *, binary: Path, build_manifest: Path | None, build_document: Mapping[str, Any] | None = None,
    capture: Mapping[str, Any] | None = None, pre: Mapping[str, Any] | None = None,
    post: Mapping[str, Any] | None = None, pre_evidence: Mapping[str, Any] | None = None,
    post_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a render FAIL record using the direct runner's evidence shape."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw-result.json"
    manifest_path = output_dir / "report.json"
    if raw_path.exists() or manifest_path.exists() or raw_path.is_symlink() or manifest_path.is_symlink():
        _fail(f"benchmark output directory already contains an evidence file: {output_dir}")
    raw_bytes = canonical_bytes({"benchmark_schema_version": VERSION, "state": "FAIL", "failure_reason": reason})
    direct_runner._write(raw_path, raw_bytes, "failure raw result")
    try:
        binary_path = binary.resolve()
        binary_sha = contracts.sha256_file(binary_path, "benchmark binary", max_bytes=direct_runner.MAX_BINARY_BYTES) if binary_path.is_file() and not binary_path.is_symlink() else "0" * 64
        binary_bytes = max(1, binary_path.stat().st_size) if binary_path.is_file() and not binary_path.is_symlink() else 1
    except OSError:
        binary_path, binary_sha, binary_bytes = binary, "0" * 64, 1
    document = dict(build_document or {})
    build_path = build_manifest.resolve() if build_manifest is not None else output_dir / "build-identity.json"
    build_sha = "0" * 64
    if build_manifest is not None and build_manifest.is_file() and not build_manifest.is_symlink():
        try:
            build_sha = contracts.sha256_file(build_manifest.resolve(), "build identity manifest", max_bytes=4 * 1024 * 1024)
        except ContractError:
            pass
    build = {
        "path": str(build_path), "sha256": build_sha,
        "source_root": document.get("source_root", str(ROOT.resolve())),
        "source_base_revision": document.get("source_base_revision", "0" * 40),
        "semantic_tree": document.get("semantic_tree", "0" * 40),
        "build_inputs_digest": document.get("build_inputs_digest", "sha256:" + "0" * 64),
        "build_configuration": contracts.expected_build_configuration(row["target"]),
        "target": row["target"], "backend": "hip", "rocm_release": direct_runner.ROCM_RELEASE,
        "rocm_root": direct_runner.ROCM_ROOT, "binary_sha256": binary_sha,
    }
    fallback_pre = dict(pre or _fallback_observation(row["target"]))
    fallback_post = dict(post or _fallback_observation(row["target"]))
    fallback_pre_evidence = dict(pre_evidence or _fallback_phase_evidence(row["target"]))
    fallback_post_evidence = dict(post_evidence or _fallback_phase_evidence(row["target"]))
    evidence = direct_runner._failed_evidence(fallback_pre_evidence, fallback_post_evidence, capture or {}, reason)
    capture = capture or {}
    process_gone = capture.get("process_group_gone") is True
    failure_exit_code = capture.get("exit_code")
    if failure_exit_code is not None and not contracts.is_int(failure_exit_code):
        failure_exit_code = None
    failure_stderr = capture.get("stderr", b"")
    if not isinstance(failure_stderr, (bytes, bytearray)):
        failure_stderr = b""
    execution = {
        "exit_code": failure_exit_code, "timed_out": bool(capture.get("timed_out", False)),
        "timeout_seconds": row["timeout_seconds"], "stderr_bytes": len(failure_stderr),
        "term_sent": bool(capture.get("term_sent", False)), "kill_sent": bool(capture.get("kill_sent", False)),
        "process_group_gone": process_gone,
    }
    cleanup = {
        "pre_process_clean": fallback_pre.get("process", {}).get("state") == "CLEAN",
        "post_process_clean": fallback_post.get("process", {}).get("state") == "CLEAN",
        "process_group_gone": process_gone,
        "retryable_cleanup": 0 if process_gone else 1, "durable_quarantine": 0,
    }
    manifest = {
        "benchmark_schema_version": VERSION, "record_kind": "evidence_manifest", "state": "FAIL",
        "required": False, "failure_reason": reason, "row_id": row["row_id"], "claims": dict(CLAIMS),
        "matrix": {"path": str(matrix_path), "matrix_id": VERSION, "sha256": matrix_digest},
        "binary": {"path": str(binary_path), "sha256": binary_sha, "bytes": binary_bytes},
        "build_identity": build,
        "model_lock": {"path": str((ROOT / expected_model()["lock_path"]).resolve()), "sha256": "0" * 64, "fingerprint": expected_model()["lock_fingerprint"]},
        "model_cache": {"path": str(output_dir / "model-cache"), "sha256": "0" * 64},
        "raw_artifact": {"path": str(raw_path), "sha256": hashlib.sha256(raw_bytes).hexdigest(), "bytes": len(raw_bytes)},
        "observations": {"pre": fallback_pre, "post": fallback_post}, "evidence": evidence,
        "execution": execution, "cleanup": cleanup,
    }
    _validate_manifest(manifest, "failure render/tokenize performance evidence manifest")
    encoded = canonical_bytes(manifest)
    direct_runner._write(manifest_path, encoded, "failure render/tokenize performance evidence manifest")
    direct_runner._write(manifest_path.with_name("report.json.sha256"), f"{hashlib.sha256(encoded).hexdigest()}  report.json\n".encode("ascii"), "failure manifest digest sidecar")
    direct_runner._write(raw_path.with_name("raw-result.json.sha256"), f"{hashlib.sha256(raw_bytes).hexdigest()}  raw-result.json\n".encode("ascii"), "failure raw result digest sidecar")
    return manifest


def validate_cli_result(result: Any, row: Mapping[str, Any], *, schema: bool = True) -> dict[str, Any]:
    if schema:
        contracts.schema_validate(result, SCHEMA_PATH, "render/tokenize engine result")
    if not isinstance(result, dict):
        _fail("render/tokenize engine result is not an object")
    row = resolved_row(row)
    model = expected_model()
    device = expected_device(row["target"])
    expected_row = {
        "row_id": row["row_id"], "model_size": MODEL_SIZE, "case_id": CASE_ID,
        "input_token_ids": list(INPUT_TOKEN_IDS), "input_token_count": 13,
        "requested_output_tokens": REQUESTED_OUTPUT_TOKENS,
    }
    if result["benchmark_schema_version"] != VERSION or result["lane"] != "render-tokenize" or result["state"] != "PASS":
        _fail("render/tokenize result lane or schema identity is stale")
    if result["row"] != expected_row:
        _fail("render/tokenize result row/token identity does not match the matrix")
    identities = result["identities"]
    expected_model_identity = {
        "model_size": MODEL_SIZE, "repo_id": model["repo_id"],
        "resolved_revision": model["resolved_revision"], "lock_fingerprint": model["lock_fingerprint"],
    }
    if identities["engine"] != "sllm" or identities["backend"] != "hip" or identities["device_index"] != device["logical_device_index"] or identities["target"] != row["target"]:
        _fail("render/tokenize engine/device identity is stale")
    if identities["model"] != expected_model_identity or identities["binding"]["model_fingerprint"] != model["lock_fingerprint"]:
        _fail("render/tokenize model/binding identity is stale")
    if result["config"] != {
        "input_token_ids": list(INPUT_TOKEN_IDS), "input_token_count": 13,
        "max_new_tokens": REQUESTED_OUTPUT_TOKENS, "greedy": True, "warmups": 3, "measured": 10,
        "tokenizer": True, "render": True,
        "stop_policy": {"stop_token_ids": [248046, 248044], "visible_stop_tokens": False},
    }:
        _fail("render/tokenize result config does not match the fixed matrix")
    load = result["model_load"]
    if load["event"] != "model_load" or load["start_ns"] != 0 or load["load_count"] != 1 or contracts.safe_difference(load["model_ready_ns"], load["start_ns"], "model load") != load["duration_ns"]:
        _fail("render/tokenize model load arithmetic or count is invalid")
    _validate_snapshot_accounting(result["memory"]["model_ready"], "model-ready memory")
    _validate_snapshot_accounting(result["memory"]["after_model_drop"], "post-model memory")
    memory = result["memory"]
    ready = memory["model_ready"]
    after_drop = memory["after_model_drop"]
    if (
        memory["model_resident_high_water_bytes"] != ready["model_resident"]["high_water_bytes"]
        or memory["resident_vram_bytes"] != ready["model_resident"]["high_water_bytes"]
        or memory["resident_vram_source"] != "model_resident_allocator_high_water"
        or memory["peak_vram_bytes"] != after_drop["high_water_bytes"]
        or memory["peak_vram_bytes"] < memory["resident_vram_bytes"]
        or memory["peak_source"] != "runtime_allocator"
    ):
        _fail("render/tokenize memory identity or high-water mark is invalid")
    if ready["request_state"]["current_bytes"] != 0 or ready["workspace"]["current_bytes"] != 0 or ready["model_resident"]["current_bytes"] == 0:
        _fail("render/tokenize model-ready memory does not contain only the resident model")
    if any(after_drop[key]["current_bytes"] != 0 for key in ("model_resident", "request_state", "workspace")) or after_drop["current_bytes"] != 0:
        _fail("render/tokenize model lifecycle memory is not cleaned")
    audit = result["audit"]
    if audit["selected_backend"] != "hip" or audit["target"] != row["target"] or audit["device_index"] != 0 or audit["fallback_used"] is not False or audit["all_dispatches_hip"] is not True:
        _fail("render/tokenize aggregate audit is not HIP-only and fallback-free")
    if audit["model_load_count"] != 1 or audit["request_model_load_count"] != 0 or audit["model_reused"] is not True or audit["sample_count"] != 13 or audit["correctness_control_request_count"] != 1 or audit["total_request_count"] != 14:
        _fail("render/tokenize model-resident/request-local audit is invalid")
    if result["cleanup"] != {
        "correctness_control_request_count": 1, "warmup_request_count": 3, "measured_request_count": 10,
        "request_cleanup_count": 14, "performance_sample_count": 13, "all_requests_dropped": True,
        "correctness_control_dropped": True, "retryable_cleanup": 0, "durable_quarantine": 0,
    } or result["session_cleanup"] != {"retryable_cleanup": 0, "durable_quarantine": 0}:
        _fail("render/tokenize cleanup is not empty")
    control = result["correctness_control"]
    if control["tokens"]["input_token_ids"] != INPUT_TOKEN_IDS or control["tokens"]["input_token_ids"] != result["config"]["input_token_ids"]:
        _fail("render/tokenize correctness control input IDs are not the locked frontend IDs")
    if (
        control["label"] != "correctness-only"
        or control["execution_path"] != "normal-untimed"
        or control["timing_instrumentation"] != "off"
        or control["included_in_performance_statistics"] is not False
        or control["comparison"] != EXPECTED_COMPARISON
    ):
        _fail("render/tokenize correctness-control execution/comparison contract is invalid")
    control_tokens = control["tokens"]
    stop_policy = {"stop_token_ids": [248046, 248044], "visible_stop_tokens": False}
    if not control_tokens["generated_token_ids"] or control_tokens["decode_input_token_ids"] != control_tokens["generated_token_ids"][:-1] or control_tokens["visible_token_ids"] != [token for token in control_tokens["generated_token_ids"] if token not in stop_policy["stop_token_ids"]]:
        _fail("render/tokenize correctness-control token semantics are invalid")
    if any(not contracts.is_int(token) or token < 0 or token > contracts.MAX_TOKEN_ID for token in control_tokens["generated_token_ids"]):
        _fail("render/tokenize correctness-control token ID is outside the locked tokenizer vocabulary")
    contracts.validate_stop_semantics(control_tokens["generated_token_ids"], control["stop"], stop_policy, row["requested_output_tokens"], "render/tokenize correctness-control")
    _validate_snapshot_accounting(control["memory"]["request_start"], "correctness-control request memory")
    _validate_snapshot_accounting(control["memory"]["after_cleanup"], "correctness-control cleanup memory")
    if control["memory"]["after_cleanup"]["request_state"]["current_bytes"] != 0 or control["memory"]["after_cleanup"]["workspace"]["current_bytes"] != 0 or control["memory"]["after_cleanup"]["model_resident"]["current_bytes"] != control["memory"]["request_start"]["model_resident"]["current_bytes"]:
        _fail("render/tokenize correctness-control cleanup memory is not empty")
    sample_submission_count = 0
    sample_dispatch_count = 0
    for group, expected_count in (("warmups", 3), ("measured", 10)):
        if result[group]["count"] != expected_count or len(result[group]["samples"]) != expected_count:
            _fail(f"render/tokenize {group} sample count is not {expected_count}")
        for expected_index, sample in enumerate(result[group]["samples"]):
            if sample["cleanup"]["sample_index"] != expected_index:
                _fail(f"render/tokenize {group} samples are missing, duplicated, or reordered")
            _validate_snapshot_accounting(sample["memory"]["request_start"], f"{group} request memory")
            _validate_snapshot_accounting(sample["memory"]["after_cleanup"], f"{group} cleanup memory")
            enriched = dict(sample)
            enriched.update({
                "_target": row["target"], "_device_index": 0,
                "_model_fingerprint": model["lock_fingerprint"],
                "_plan_digest": identities["binding"]["plan_digest"],
            })
            contracts.expected_sample(enriched, INPUT_TOKEN_IDS, stop_policy, row["requested_output_tokens"])
            _validate_control_equality(control, sample)
            sample_submission_count += sample["audit"]["submission_count"]
            sample_dispatch_count += sample["audit"]["kernel_dispatch_count"]
    if audit["submission_count"] != sample_submission_count or audit["kernel_dispatch_count"] != sample_dispatch_count:
        _fail("render/tokenize aggregate dispatch counts do not match samples")
    return result


def _raw_bytes(capture: Mapping[str, Any]) -> bytes:
    value = capture.get("stdout", b"")
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    _fail("benchmark stdout capture is not bytes")


def run_row(
    row_id: str,
    binary: Path,
    model_lock: Path,
    model_cache: Path,
    output_dir: Path,
    *,
    build_manifest: Path | None = None,
    matrix_path: Path = MATRIX_PATH,
    repo: Path | None = None,
    command_runner: Callable[[list[str], Mapping[str, str], Path, int], dict[str, Any]] | None = None,
    observation_provider: Callable[[str, str], dict[str, Any]] | None = None,
    evidence_provider: Callable[[str, str], dict[str, Any]] | None = None,
    tool_provider: Callable[[], dict[str, str]] | None = None,
) -> dict[str, Any]:
    repo = (repo or ROOT).resolve()
    matrix, matrix_digest = load_matrix(matrix_path)
    rows = {row["row_id"]: row for row in matrix["rows"]}
    if row_id not in rows:
        _fail(f"row is not in the closed render/tokenize matrix: {row_id}")
    row = resolved_row(rows[row_id])
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw-result.json"
    manifest_path = output_dir / "report.json"
    if raw_path.exists() or manifest_path.exists() or raw_path.is_symlink() or manifest_path.is_symlink():
        _fail("benchmark output directory already contains an evidence file")
    binary_input = binary
    try:
        binary = direct_runner._regular_executable(binary.resolve())
        lock = direct_runner._resolve_repo_path(str(model_lock), repo, "model lock")
        cache = direct_runner._resolve_repo_path(str(model_cache), repo, "model cache")
        lock_document, lock_digest = direct_runner._validate_lock(lock, MODEL_SIZE, repo)
        cache_hash = direct_runner._validate_cache(cache, lock_document)
        binary_digest = contracts.sha256_file(binary, "benchmark binary", max_bytes=direct_runner.MAX_BINARY_BYTES)
        if build_manifest is None:
            _fail("an immutable build identity manifest is required")
        build_document, build_digest = direct_runner._validate_build_manifest(build_manifest, binary, row["target"], repo)
    except (ContractError, OSError, ValueError) as exc:
        return _durable_failure_manifest(row, matrix_path, matrix_digest, output_dir, str(exc), binary=binary_input, build_manifest=build_manifest)
    observer = observation_provider or direct_runner._amd_smi_observation
    evidence_observer = evidence_provider or direct_runner._amd_smi_phase_evidence
    pre = _fallback_observation(row["target"])
    pre_evidence = _fallback_phase_evidence(row["target"])
    try:
        pre = direct_runner.validate_observation(observer(row["target"], "pre"), row["target"], "pre")
        pre_evidence = evidence_observer(row["target"], "pre")
    except (ContractError, OSError, ValueError) as exc:
        return _durable_failure_manifest(row, matrix_path, matrix_digest, output_dir, f"pre-run health/evidence failed: {exc}", binary=binary, build_manifest=build_manifest, build_document=build_document, pre=pre, pre_evidence=pre_evidence)
    command = _expected_command(binary, row, lock, cache)
    environment = direct_runner._execution_environment(row_id, row["target"])
    capture: dict[str, Any] = {
        "stdout": b"", "stderr": b"", "exit_code": None, "timed_out": False,
        "term_sent": False, "kill_sent": False, "process_group_gone": False,
        "monitor": {"samples": [], "errors": []},
    }
    post = _fallback_observation(row["target"])
    post_evidence = _fallback_phase_evidence(row["target"])
    reasons: list[str] = []
    try:
        if command_runner is not None:
            candidate = command_runner(command, environment, repo, row["timeout_seconds"])
        else:
            candidate = direct_runner._execute_bounded(
                command, environment, repo, row["timeout_seconds"],
                monitor_provider=direct_runner._amd_smi_monitor_sample, monitor_target=row["target"],
            )
        if not isinstance(candidate, dict):
            _fail("benchmark execution returned a non-object capture")
        capture = candidate
    except (ContractError, OSError, ValueError) as exc:
        reasons.append(f"benchmark execution failed: {exc}")
    try:
        post = direct_runner.validate_observation(observer(row["target"], "post"), row["target"], "post")
    except (ContractError, OSError, ValueError) as exc:
        reasons.append(f"post-run health failed: {exc}")
    try:
        post_evidence = evidence_observer(row["target"], "post")
    except (ContractError, OSError, ValueError) as exc:
        reasons.append(f"post-run evidence failed: {exc}")
    try:
        direct_runner._write(raw_path, _raw_bytes(capture), "raw result")
        raw_digest = contracts.sha256_file(raw_path, "raw result", max_bytes=direct_runner.MAX_RAW_BYTES)
        raw_bytes = raw_path.stat().st_size
    except (ContractError, OSError, ValueError) as exc:
        if not raw_path.exists() and not manifest_path.exists():
            return _durable_failure_manifest(row, matrix_path, matrix_digest, output_dir, f"raw result persistence failed: {exc}", binary=binary, build_manifest=build_manifest, build_document=build_document, pre=pre, post=post, pre_evidence=pre_evidence, post_evidence=post_evidence, capture=capture)
        raise
    try:
        raw, _, _ = contracts.read_json(raw_path, "raw result", direct_runner.MAX_RAW_BYTES)
        validate_cli_result(raw, row)
    except (ContractError, OSError, ValueError) as exc:
        reasons.append(str(exc))
    stderr_value = capture.get("stderr", b"")
    if not isinstance(stderr_value, (bytes, bytearray)):
        reasons.append("benchmark stderr capture is not bytes")
        stderr_value = b""
    if capture.get("exit_code") != 0:
        reasons.append(f"benchmark process exited with {capture.get('exit_code')}")
    if capture.get("timed_out"):
        reasons.append("benchmark process timed out")
    if stderr_value != b"":
        reasons.append("benchmark stderr was not empty")
    if capture.get("term_sent") or capture.get("kill_sent") or capture.get("process_group_gone") is not True:
        reasons.append("benchmark process group cleanup was not clean")
    if pre != post:
        reasons.append("pre/post health or process observations differ")
    try:
        tool = tool_provider() if tool_provider else {"path": direct_runner.AMD_SMI_EXECUTABLE, **direct_runner._amd_smi_version()}
        evidence = direct_runner._build_evidence(pre_evidence, post_evidence, capture, row["target"], tool)
    except (ContractError, OSError, ValueError) as exc:
        evidence = direct_runner._failed_evidence(pre_evidence, post_evidence, capture, str(exc))
        reasons.append(f"runtime evidence validation failed: {exc}")
    try:
        binary_after = contracts.sha256_file(binary, "benchmark binary after run", max_bytes=direct_runner.MAX_BINARY_BYTES)
        if binary_after != binary_digest:
            reasons.append("benchmark binary changed during the run")
    except (ContractError, OSError, ValueError) as exc:
        reasons.append(f"benchmark binary post-run validation failed: {exc}")
    try:
        build_after = contracts.sha256_file(build_manifest, "build identity manifest after run", max_bytes=4 * 1024 * 1024)
        if build_after != build_digest:
            reasons.append("build identity manifest changed during the run")
    except (ContractError, OSError, ValueError) as exc:
        reasons.append(f"build identity manifest post-run validation failed: {exc}")
    try:
        cache_after = contracts.cache_digest(cache)
    except (ContractError, OSError, ValueError) as exc:
        cache_after = "0" * 64
        reasons.append(str(exc))
    if cache_after != cache_hash:
        reasons.append("model cache changed during the run")
    cleanup = {
        "pre_process_clean": pre["process"]["state"] == "CLEAN",
        "post_process_clean": post["process"]["state"] == "CLEAN",
        "process_group_gone": capture.get("process_group_gone") is True,
        "retryable_cleanup": 0 if capture.get("process_group_gone") is True and post["process"]["state"] == "CLEAN" else 1,
        "durable_quarantine": 0,
    }
    exit_code = capture.get("exit_code")
    if exit_code is not None and not contracts.is_int(exit_code):
        reasons.append("benchmark exit code capture is not an integer")
        exit_code = None
    execution = {
        "exit_code": exit_code, "timed_out": bool(capture.get("timed_out", False)),
        "timeout_seconds": row["timeout_seconds"], "stderr_bytes": len(stderr_value),
        "term_sent": bool(capture.get("term_sent", False)), "kill_sent": bool(capture.get("kill_sent", False)),
        "process_group_gone": capture.get("process_group_gone") is True,
    }
    if cleanup["retryable_cleanup"] != 0 or cleanup["durable_quarantine"] != 0 or not cleanup["pre_process_clean"] or not cleanup["post_process_clean"]:
        reasons.append("render/tokenize cleanup is not fail-closed")
    manifest = {
        "benchmark_schema_version": VERSION,
        "record_kind": "evidence_manifest",
        "state": "PASS" if not reasons else "FAIL",
        "required": False,
        "failure_reason": "; ".join(reasons) if reasons else None,
        "row_id": row_id,
        "claims": dict(CLAIMS),
        "matrix": {"path": str(matrix_path), "matrix_id": VERSION, "sha256": matrix_digest},
        "binary": {"path": str(binary), "sha256": binary_digest, "bytes": binary.stat().st_size},
        "build_identity": {
            "path": str(build_manifest.resolve()), "sha256": build_digest,
            "source_root": build_document["source_root"], "source_base_revision": build_document["source_base_revision"],
            "semantic_tree": build_document["semantic_tree"],
            "build_inputs_digest": build_document["build_inputs_digest"],
            "build_configuration": build_document["build_configuration"], "target": build_document["target"],
            "backend": build_document["backend"], "rocm_release": build_document["rocm_release"],
            "rocm_root": build_document["rocm_root"], "binary_sha256": build_document["binary_sha256"],
        },
        "model_lock": {"path": str(lock), "sha256": lock_digest, "fingerprint": expected_model()["lock_fingerprint"]},
        "model_cache": {"path": str(cache), "sha256": cache_hash},
        "raw_artifact": {"path": str(raw_path), "sha256": raw_digest, "bytes": raw_bytes},
        "observations": {"pre": pre, "post": post},
        "evidence": evidence,
        "execution": execution,
        "cleanup": cleanup,
    }
    try:
        _validate_manifest(manifest, "render/tokenize performance evidence manifest")
    except ContractError as exc:
        reasons.append(f"render/tokenize manifest validation failed: {exc}")
        manifest["state"] = "FAIL"
        manifest["failure_reason"] = "; ".join(reasons)
        manifest["evidence"] = direct_runner._failed_evidence(pre_evidence, post_evidence, capture, manifest["failure_reason"])
        _validate_manifest(manifest, "failed render/tokenize performance evidence manifest")
    encoded = canonical_bytes(manifest)
    direct_runner._write(manifest_path, encoded, "render/tokenize performance evidence manifest")
    direct_runner._write(manifest_path.with_name("report.json.sha256"), f"{hashlib.sha256(encoded).hexdigest()}  report.json\n".encode("ascii"), "manifest digest sidecar")
    direct_runner._write(raw_path.with_name("raw-result.json.sha256"), f"{raw_digest}  raw-result.json\n".encode("ascii"), "raw result digest sidecar")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--repo", type=Path, default=ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = run_row(args.row, args.binary, args.model_lock, args.model_cache, args.output_dir, build_manifest=args.build_manifest, matrix_path=args.matrix, repo=args.repo)
    except (ContractError, OSError, ValueError) as exc:
        print(f"engine-performance-render: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"engine-performance-render: {manifest['state']}")
    return 0 if manifest["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
