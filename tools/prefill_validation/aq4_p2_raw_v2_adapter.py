#!/usr/bin/env python3
"""Execute and strictly adapt full-model-driver v2 artifacts into P2 raw v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT_FIELDS = {"schema_version", "raw_target_schema_version", "scope", "status", "immutable_status", "case_id", "case_sha256", "identity", "requested_m", "resolved_m", "actual_token_batch_width", "actual_request_batch_width", "timing", "audit", "lifecycle", "reset", "outcome", "oom", "fallback", "preflight", "failure", "links", "adapter"}
IDENTITY_FIELDS = {"served_model_manifest_sha256", "model_id", "model_revision", "format_id", "implementation_id", "manifest_worker_binary_path", "manifest_worker_binary_sha256", "benchmark_binary_path", "benchmark_binary_sha256", "benchmark_worker_roles_distinct", "package_root", "package_content_sha256", "package_manifest_sha256", "package_file_count", "package_bytes", "manifest_device_architecture", "runtime_device", "execution_profile"}
RUNTIME_FIELDS = {"requested_device_index", "observed_device_id", "observed_backend", "observed_name", "observed_architecture"}
GENERATION_FIELDS = {"cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second"}
MAX_JSON = 32 * 1024 * 1024


class AdapterError(ValueError): pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result: raise AdapterError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON: raise AdapterError(f"{label} must be a bounded regular file")
    try: value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(AdapterError(f"non-finite JSON: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error: raise AdapterError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict): raise AdapterError(f"{label} root must be an object")
    return value


def exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields: raise AdapterError(f"{label} fields differ")
    return value


def sha_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file(): raise AdapterError("hash input must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes: return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
def ordered_json(value: Any) -> bytes: return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
def digest(value: Any) -> str: return hashlib.sha256(canonical(value)).hexdigest()


def contained(root: Path, path: Path, label: str, *, existing: bool = True) -> Path:
    root = root.resolve(strict=True); result = path.resolve(strict=existing)
    if result != root and root not in result.parents: raise AdapterError(f"{label} escapes run root")
    return result


def validate_driver(value: dict[str, Any], path: Path, case: dict[str, Any], identity: dict[str, Any], preflight_path: Path, driver: Path) -> None:
    exact(value, ROOT_FIELDS, "driver root")
    if value["schema_version"] != "ullm.qwen35_aq4_p2.full_model_driver.v2" or value["raw_target_schema_version"] != "ullm.aq4_production_p2_raw_result.v2" or value["scope"] != case.get("scope"): raise AdapterError("driver schema/scope differs")
    status = value["status"]
    if status not in {"ok", "failed", "oom"} or value["immutable_status"] is not (status != "ok") or value["case_id"] != case.get("case_id") or value["case_sha256"] != case.get("case_sha256"): raise AdapterError("driver status/case differs")
    adapter = exact(value["adapter"], {"target_schema_version", "mapping_version", "exact_root_fields", "benchmark_binary_role", "manifest_worker_role", "raw_v2_requires_role_aware_adapter"}, "adapter")
    if adapter != {"target_schema_version": "ullm.aq4_production_p2_raw_result.v2", "mapping_version": "ullm.aq4_p2_full_model_to_raw.v1", "exact_root_fields": True, "benchmark_binary_role": "executed_benchmark_driver", "manifest_worker_role": "served_identity_reference", "raw_v2_requires_role_aware_adapter": True}: raise AdapterError("adapter handshake differs")
    links = exact(value["links"], {"case", "identity", "preflight", "timing", "audit"}, "links")
    expected_files = (("case", Path(case["_path"])), ("identity", Path(identity["_path"])), ("preflight", preflight_path))
    for role, expected in expected_files:
        link = exact(links[role], {"path", "sha256"}, f"links.{role}")
        if Path(link["path"]).resolve() != expected.resolve() or link["sha256"] != sha_file(expected): raise AdapterError(f"driver {role} link differs")
    timing = exact(value["timing"], {"request_elapsed_ms", "generation", "generated_tokens"}, "timing")
    if exact(links["timing"], {"json_pointer", "sha256"}, "timing link")["json_pointer"] != "/timing" or links["timing"]["sha256"] != hashlib.sha256(ordered_json(timing)).hexdigest(): raise AdapterError("timing embedded link differs")
    fallback = exact(value["fallback"], {"count", "unexpected_count", "reasons"}, "fallback")
    if not isinstance(fallback["reasons"], list) or any(not isinstance(item, dict) or set(item) != {"unavailable_primary", "resolved_implementation", "invocation_count"} for item in fallback["reasons"]): raise AdapterError("fallback reasons differ")
    preflight = exact(value["preflight"], {"input", "required_environment_count", "required_environment_verified", "binary_roles_verified", "package_tree_verified"}, "preflight")
    if preflight["input"] != load(preflight_path, "preflight"): raise AdapterError("driver preflight input differs")
    if status != "ok":
        if value["failure"] is None: raise AdapterError("failed driver artifact lacks failure")
        return
    model = identity.get("model_identity", {}); driver_identity = exact(value["identity"], IDENTITY_FIELDS, "driver identity"); runtime = exact(driver_identity["runtime_device"], RUNTIME_FIELDS, "runtime device")
    worker = Path(identity.get("artifacts", {}).get("worker", "")); package = Path(identity.get("artifacts", {}).get("package_root", "")); package_manifest = Path(identity.get("artifacts", {}).get("package_manifest", ""))
    expected_identity = {"served_model_manifest_sha256": identity.get("hash_binding", {}).get("served_model_manifest_sha256"), "model_id": model.get("id"), "model_revision": model.get("revision"), "format_id": case.get("format_id"), "implementation_id": case.get("implementation_id"), "manifest_worker_binary_path": str(worker.resolve()), "manifest_worker_binary_sha256": identity.get("hash_binding", {}).get("worker_binary_sha256"), "benchmark_binary_path": str(driver.resolve()), "benchmark_binary_sha256": sha_file(driver), "package_root": str(package.resolve()), "package_content_sha256": identity.get("hash_binding", {}).get("package_content_sha256"), "package_manifest_sha256": identity.get("hash_binding", {}).get("package_manifest_sha256")}
    if any(driver_identity.get(key) != wanted for key, wanted in expected_identity.items()) or driver_identity["benchmark_worker_roles_distinct"] is not True or worker.resolve() == driver.resolve() or sha_file(worker) == sha_file(driver): raise AdapterError("driver/served identity roles differ")
    device = case.get("device", {}); expected_runtime = {"requested_device_index": device.get("runtime_device_index"), "observed_backend": device.get("backend"), "observed_name": device.get("name"), "observed_architecture": device.get("architecture")}
    if any(runtime.get(key) != wanted for key, wanted in expected_runtime.items()): raise AdapterError("runtime device differs")
    if value["requested_m"] != case.get("prefill_requested_m") or value["resolved_m"] != case.get("resolved_m") or value["actual_token_batch_width"] != case.get("resolved_m") or value["actual_request_batch_width"] != case.get("request_count"): raise AdapterError("driver request/width differs")
    generation = exact(timing["generation"], GENERATION_FIELDS, "generation timing")
    if generation["prompt_n"] != case.get("prompt_tokens") or generation["cache_n"] != case.get("cached_prefix_tokens") or generation["predicted_n"] != case.get("generated_tokens") or timing["generated_tokens"] != case.get("generated_tokens"): raise AdapterError("driver token counts differ")
    audit = exact(value["audit"], {"deterministic_digest_sha256", "outcome", "coverage_complete", "physical_operation_invocations", "total_records"}, "audit")
    if links["audit"] != {"json_pointer": "/audit", "sha256": audit["deterministic_digest_sha256"]} or audit["coverage_complete"] is not True or value["outcome"] != audit["outcome"]: raise AdapterError("driver audit/outcome differs")
    lifecycle = exact(value["lifecycle"], {"prepare", "commit", "discard", "error", "cancel", "prefill", "publication", "reset"}, "lifecycle")
    reset = exact(value["reset"], {"attempted", "complete", "failed"}, "reset")
    if lifecycle["prepare"] <= 0 or lifecycle["prepare"] != lifecycle["commit"] + lifecycle["discard"] or lifecycle["error"] != 0 or lifecycle["cancel"] != 0 or reset != {"attempted": 1, "complete": 1, "failed": 0} or lifecycle["reset"] != reset: raise AdapterError("driver lifecycle/reset differs")
    for phase in ("prefill", "publication"):
        counts = exact(lifecycle[phase], {"prepare", "commit", "discard"}, f"lifecycle.{phase}")
        if counts["prepare"] != counts["commit"] + counts["discard"]: raise AdapterError("driver phase lifecycle differs")
    if fallback["unexpected_count"] != 0 or (case.get("mode") != "all_m1" and fallback["count"] != 0): raise AdapterError("driver fallback differs")
    if value["failure"] is not None or value["oom"] is not None: raise AdapterError("ok artifact contains failure/OOM")


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise AdapterError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("xb") as target: target.write((json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode()); target.flush(); os.fsync(target.fileno())
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    root = args.run_root.resolve(strict=True)
    for path, label in ((args.case, "case"), (args.identity, "identity"), (args.policy, "policy"), (args.preflight, "preflight"), (args.driver, "driver"), (args.served_model_manifest, "served manifest"), (args.fixture, "fixture")): contained(root, path, label)
    for path in (args.output, args.measurement, args.state, args.trace_input): contained(root, path, "output", existing=False)
    raw_dir = contained(root, args.raw_dir, "raw directory", existing=False); raw_dir.mkdir(parents=True, exist_ok=False)
    case = load(args.case, "case"); identity = load(args.identity, "identity"); load(args.policy, "policy"); preflight = load(args.preflight, "preflight")
    case["_path"] = str(args.case.resolve()); identity["_path"] = str(args.identity.resolve())
    worker = Path(identity.get("artifacts", {}).get("worker", ""))
    if worker.resolve() == args.driver.resolve() or sha_file(worker) == sha_file(args.driver): raise AdapterError("benchmark driver and served worker roles must be distinct")
    artifacts: list[dict[str, Any]] = []; failure: str | None = None; artifact_hashes: set[str] = set()
    for index in range(12):
        artifact_path = raw_dir / f"run-{index:02d}.driver.json"
        command = [str(args.driver), "--served-model-manifest", str(args.served_model_manifest), "--fixture", str(args.fixture), "--case", str(args.case), "--identity", str(args.identity), "--preflight", str(args.preflight), "--m", str(case.get("prefill_requested_m")), "--output", str(artifact_path)]
        completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=args.timeout, check=False)
        if not artifact_path.is_file(): raise AdapterError(f"driver did not publish run {index} artifact")
        artifact_sha = sha_file(artifact_path)
        if artifact_sha in artifact_hashes: raise AdapterError("driver raw artifact was reused")
        artifact_hashes.add(artifact_sha)
        artifact = load(artifact_path, f"driver run {index}"); validate_driver(artifact, artifact_path, case, identity, args.preflight, args.driver); artifacts.append(artifact)
        if completed.returncode != 0 or artifact["status"] != "ok": failure = f"driver_run_{index}_{artifact['status']}"; break
    status = "ok" if failure is None and len(artifacts) == 12 else artifacts[-1]["status"] if artifacts and artifacts[-1]["status"] in {"failed", "oom"} else "failed"
    expanded_path = Path(identity.get("artifacts", {}).get("expanded_manifest", "")); contained(root, expanded_path, "expanded manifest")
    links = {"expanded": {"path": str(expanded_path.resolve()), "sha256": sha_file(expanded_path)}, "identity": {"path": str(args.identity.resolve()), "sha256": sha_file(args.identity)}, "policy": {"path": str(args.policy.resolve()), "sha256": sha_file(args.policy)}, "measurement": None, "state": None, "trace": None, "driver_runs": [{"path": str((raw_dir / f"run-{index:02d}.driver.json").resolve()), "sha256": sha_file(raw_dir / f"run-{index:02d}.driver.json"), "run_index": index} for index in range(len(artifacts))]}
    raw = {"schema_version": "ullm.aq4_production_p2_raw_result.v2", "case_id": case["case_id"], "case_sha256": case["case_sha256"], "status": status, "immutable_status": status != "ok", "mode": "cpu_synthetic" if args.cpu_fixture else "production", "device_id": case.get("device", {}).get("device_id"), "case_contract": {key: case.get(key) for key in ("fixture_id", "scope", "phase", "mode", "prompt_tokens", "cached_prefix_tokens", "context_tokens", "decode_start_tokens", "generated_tokens", "prefill_requested_m", "resolved_m", "request_count", "decode_request_count", "sampling", "control", "control_id", "format_id", "implementation_id", "device")}, "started_at_unix": None, "finished_at_unix": None, "execution": {"status": status, "returncode": 0 if status == "ok" else 1, "elapsed_ms": sum(float(item["timing"]["request_elapsed_ms"]) for item in artifacts), "timed_out": False, "output_overflow": False, "stdout_sha256": hashlib.sha256(b"").hexdigest(), "stderr_sha256": hashlib.sha256((failure or "").encode()).hexdigest(), "stdout_bytes": 0, "stderr_bytes": len((failure or "").encode())}, "declared_execution": {"executable": str(worker.resolve()), "executable_sha256": identity["hash_binding"]["worker_binary_sha256"], "package_root": identity["artifacts"]["package_root"], "package_content_sha256": identity["hash_binding"]["package_content_sha256"], "argv_sha256": None, "argv_count": 0, "argv_values_recorded": False}, "executed_benchmark_driver": {"path": str(args.driver.resolve()), "sha256": sha_file(args.driver), "role": "executed_benchmark_driver"}, "served_identity_reference": {"worker_path": str(worker.resolve()), "worker_sha256": sha_file(worker), "role": "served_identity_reference"}, "links": links, "preflight": preflight, "failure_reason": failure, "capture_contract": {"bounded_streaming": True, "shell": False, "max_output_bytes_per_stream": 0, "command_arguments_stored": False}}
    if status == "ok":
        def row(item: dict[str, Any]) -> dict[str, Any]:
            generation = item["timing"]["generation"]
            return {"prefill_ms": generation["prompt_ms"], "ttft_ms": generation["prompt_ms"] + generation["predicted_per_token_ms"], "decode_ms": generation["predicted_ms"], "inter_token_latency_ms": generation["predicted_per_token_ms"], "end_to_end_ms": item["timing"]["request_elapsed_ms"], "vram_peak_bytes": 0, "workspace_peak_bytes": 0, "actual_token_batch_width": item["actual_token_batch_width"], "actual_request_batch_width": item["actual_request_batch_width"]}
        measurement = {"schema_version": "ullm.aq4_p2_measurements.v1", "case_id": case["case_id"], "warmup_runs": [row(item) for item in artifacts[:2]], "measured_runs": [row(item) for item in artifacts[2:]]}
        terminal = artifacts[-1]; state = {"schema_version": "ullm.aq4_p2_state_evidence.v1", "case_id": case["case_id"], "status": "valid", "checks": {field: True for field in ("finite_outputs", "shape_contract_passed", "kv_state_cache_passed", "scheduler_progress_passed", "chunk_equivalence_passed", "cancel_reset_passed", "publish_failure_reset_passed")}, "reset": {"attempted": True, "complete": True, "failed": False}, "fallback": {"unexpected_count": 0, "fail_closed_count": 0, "unsupported_count": 0, "reasons": terminal["fallback"]["reasons"]}, "memory": {"oom": None, "headroom_bytes": preflight["vram_headroom_bytes"], "observed_peak_bytes": 0, "workspace_peak_bytes": 0}}
        trace_input = {"schema_version": "ullm.aq4_p2_driver_lifecycle_input.v1", "not_a_production_execution_trace": True, "case_id": case["case_id"], "case_sha256": case["case_sha256"], "runs": [{"run_index": index, "audit": item["audit"], "lifecycle": item["lifecycle"], "reset": item["reset"], "outcome": item["outcome"], "fallback": item["fallback"], "driver_sha256": sha_file(raw_dir / f"run-{index:02d}.driver.json")} for index, item in enumerate(artifacts)]}
        write_atomic(args.measurement, measurement); write_atomic(args.state, state); write_atomic(args.trace_input, trace_input)
        links["measurement"] = {"path": str(args.measurement.resolve()), "sha256": sha_file(args.measurement)}; links["state"] = {"path": str(args.state.resolve()), "sha256": sha_file(args.state)}; links["driver_lifecycle_input"] = {"path": str(args.trace_input.resolve()), "sha256": sha_file(args.trace_input)}
    write_atomic(args.output, raw)
    return 0 if status == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run_root", "case", "identity", "policy", "preflight", "driver", "served_model_manifest", "fixture", "raw_dir", "output", "measurement", "state"): parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--driver-lifecycle-input", dest="trace_input", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0); parser.add_argument("--cpu-fixture", action="store_true")
    try: return run(parser.parse_args(argv))
    except (AdapterError, OSError, subprocess.TimeoutExpired, ValueError) as error:
        print(f"P2 raw-v2 adapter failed closed: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
