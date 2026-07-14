#!/usr/bin/env python3
"""Run the representative AQ4 P2 full-model target profile through one resident driver.

The driver protocol is deliberately tiny and hash-only: one child process announces one model
load, then receives case/run commands.  A real GPU driver can implement this protocol later; the
planner and fake-driver tests are CPU-only and never touch a service or device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 64 * 1024 * 1024
CASE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SCHEMA = "ullm.aq4_p2_resident_batch.v1"
DRIVER_SCHEMA = "ullm.aq4_p2_resident_driver.v1"
WARMUP_RUNS = 2
MEASURED_RUNS = 10
READY_IDENTITY_KEYS = {
    "binary_sha256",
    "build_git_commit",
    "protocol",
    "package_manifest_sha256",
    "runtime_device",
    "guard_set_sha256",
}
RUNTIME_DEVICE_KEYS = {
    "runtime_device_index",
    "device_id",
    "backend",
    "name",
    "architecture",
}


class BatchError(ValueError):
    pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        if key in value:
            raise BatchError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise BatchError(f"{label} must be a bounded regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(BatchError(f"non-finite JSON: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BatchError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise BatchError(f"{label} root must be an object")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise BatchError(f"{label} must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def case_hash(case: dict[str, Any]) -> str:
    value = json.loads(json.dumps(case))
    value["case_sha256"] = None
    return sha_bytes(canonical(value))


def atomic_write(path: Path, value: Any) -> None:
    if os.path.lexists(path):
        raise BatchError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    try:
        with temporary.open("x", encoding="utf-8") as target:
            json.dump(value, target, ensure_ascii=True, sort_keys=True, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise BatchError(f"refusing to overwrite {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def select_target_cases(expanded: dict[str, Any], fixture_index: dict[str, Any]) -> list[dict[str, Any]]:
    if expanded.get("schema_version") != "ullm.aq4_production_p2_expanded.v2":
        raise BatchError("expanded manifest schema differs")
    if fixture_index.get("schema_version") != "ullm.aq4_p2_fixture_index.v1":
        raise BatchError("fixture index schema differs")
    cases = expanded.get("cases")
    if not isinstance(cases, list):
        raise BatchError("expanded cases are missing")
    selected = [
        case for case in cases
        if isinstance(case, dict)
        and case.get("stage_id") == "representative"
        and case.get("scope") == "full_model"
        and case.get("phase") == "cold_prefill"
        and case.get("device", {}).get("device_id") == "r9700-rdna4"
        and case.get("control_id") == "aq4_0_target"
    ]
    if len(selected) != 84:
        raise BatchError(f"representative full_model target profile must contain 84 cases, got {len(selected)}")
    index_cases = fixture_index.get("cases")
    if not isinstance(index_cases, list):
        raise BatchError("fixture index cases are missing")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in index_cases:
        if not isinstance(entry, dict) or not isinstance(entry.get("case_id"), str) or not entry["case_id"] or entry["case_id"] in by_id:
            raise BatchError("fixture index contains invalid or duplicate case IDs")
        by_id[entry["case_id"]] = entry
    if len(by_id) != fixture_index.get("case_count"):
        raise BatchError("fixture index case coverage differs")
    for case in selected:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or CASE_ID_RE.fullmatch(case_id) is None or case.get("case_sha256") != case_hash(case):
            raise BatchError(f"selected case identity differs: {case_id}")
        entry = by_id.get(case_id)
        if not isinstance(entry, dict) or entry.get("case_sha256") != case.get("case_sha256") or entry.get("prompt_tokens") != case.get("prompt_tokens") or entry.get("context_tokens") != case.get("context_tokens") or entry.get("generated_tokens") != case.get("generated_tokens"):
            raise BatchError(f"fixture index does not bind selected case: {case_id}")
        fixture_path = Path(entry.get("fixture_path", ""))
        if sha_file(fixture_path, "fixture") != entry.get("fixture_sha256"):
            raise BatchError(f"fixture hash differs: {case_id}")
    return sorted(selected, key=lambda case: case["case_id"])


def _send(process: subprocess.Popen[str], message: dict[str, Any]) -> None:
    if process.stdin is None:
        raise BatchError("resident driver stdin is unavailable")
    process.stdin.write(json.dumps(message, ensure_ascii=True, sort_keys=True) + "\n")
    process.stdin.flush()


def _recv(process: subprocess.Popen[str], timeout: float) -> dict[str, Any]:
    if process.stdout is None:
        raise BatchError("resident driver stdout is unavailable")
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    if not ready:
        raise BatchError("resident driver response timed out")
    line = process.stdout.readline()
    if not line:
        raise BatchError("resident driver exited before response")
    value = json.loads(line, object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(BatchError(f"non-finite driver number: {item}")))
    if not isinstance(value, dict):
        raise BatchError("resident driver response is not an object")
    return value


def _validate_ready_identity(value: Any, identity: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != READY_IDENTITY_KEYS:
        raise BatchError("resident driver identity fields differ")
    for field in ("binary_sha256", "package_manifest_sha256", "guard_set_sha256"):
        if not isinstance(value[field], str) or SHA256_RE.fullmatch(value[field]) is None:
            raise BatchError(f"resident driver identity.{field} is invalid")
    if not isinstance(value["build_git_commit"], str) or GIT_SHA_RE.fullmatch(value["build_git_commit"]) is None:
        raise BatchError("resident driver identity.build_git_commit is invalid")
    if value["protocol"] != DRIVER_SCHEMA:
        raise BatchError("resident driver identity protocol differs")
    runtime = value["runtime_device"]
    if not isinstance(runtime, dict) or set(runtime) != RUNTIME_DEVICE_KEYS:
        raise BatchError("resident driver runtime device fields differ")
    if type(runtime["runtime_device_index"]) is not int or runtime["runtime_device_index"] < 0:
        raise BatchError("resident driver runtime device index is invalid")
    if not isinstance(runtime["device_id"], (str, int)) or isinstance(runtime["device_id"], bool) or (isinstance(runtime["device_id"], str) and not runtime["device_id"]):
        raise BatchError("resident driver runtime device ID is invalid")
    for field in ("backend", "name", "architecture"):
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise BatchError(f"resident driver runtime device {field} is invalid")
    bound = identity.get("resident_driver_identity")
    if not isinstance(bound, dict) or set(bound) != READY_IDENTITY_KEYS:
        raise BatchError("identity file lacks resident driver identity")
    if bound != value:
        raise BatchError("resident driver identity differs from identity file")
    bound_hashes = identity.get("hash_binding", {})
    if isinstance(bound_hashes, dict) and bound_hashes.get("package_manifest_sha256") != value["package_manifest_sha256"]:
        raise BatchError("resident package manifest identity differs")
    if identity.get("build_git_commit") not in (None, value["build_git_commit"]):
        raise BatchError("resident build commit identity differs")
    for case in cases:
        device = case.get("device")
        if not isinstance(device, dict):
            raise BatchError(f"case device identity is missing: {case.get('case_id')}")
        for field in RUNTIME_DEVICE_KEYS:
            if field in device and device[field] != runtime[field]:
                raise BatchError(f"resident runtime device differs from case: {case['case_id']}")
    return value


def validate_ready(value: dict[str, Any], identity: dict[str, Any], cases: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if set(value) != {"event", "schema_version", "model_loads", "resident_session_id", "driver_identity"} or value.get("event") != "ready" or value.get("schema_version") != DRIVER_SCHEMA or type(value.get("model_loads")) is not int or value.get("model_loads") != 1 or not isinstance(value.get("resident_session_id"), str) or not value["resident_session_id"]:
        raise BatchError("resident driver did not prove one model load")
    ready_identity = _validate_ready_identity(value["driver_identity"], identity, cases)
    return value["resident_session_id"], ready_identity


def validate_run(value: dict[str, Any], case: dict[str, Any], session_id: str) -> dict[str, Any]:
    if value.get("event") != "run_complete" or value.get("resident_session_id") != session_id or value.get("status") not in {"ok", "failed", "oom"}:
        raise BatchError(f"resident driver run identity/status differs: {case['case_id']}")
    if type(value.get("elapsed_ms")) not in {int, float} or value["elapsed_ms"] < 0:
        raise BatchError("resident driver elapsed_ms is invalid")
    reset = value.get("reset")
    if not isinstance(reset, dict) or reset != {"attempted": 1, "complete": 1, "failed": 0}:
        raise BatchError(f"resident driver reset is not complete: {case['case_id']}")
    if value["status"] == "ok":
        audit = value.get("audit")
        resource = value.get("resource")
        if not isinstance(audit, dict) or audit.get("coverage_complete") is not True or not isinstance(audit.get("deterministic_digest_sha256"), str) or not isinstance(resource, dict) or not resource.get("samples") or not isinstance(resource.get("peak"), dict):
            raise BatchError(f"resident driver terminal audit/resource is incomplete: {case['case_id']}")
        if value.get("actual_token_batch_width") != case.get("resolved_m") or value.get("actual_request_batch_width") != case.get("request_count"):
            raise BatchError(f"resident driver actual width differs: {case['case_id']}")
    return value


def make_case_raw(case: dict[str, Any], fixture_entry: dict[str, Any], identity_link: dict[str, str], policy_link: dict[str, str], run_id: str, baseline_kind: str, session_id: str, driver_identity: dict[str, Any], runs: list[dict[str, Any]], failure_reason: str | None = None) -> dict[str, Any]:
    status = "ok" if not failure_reason and all(run["status"] == "ok" for run in runs) else "oom" if any(run["status"] == "oom" for run in runs) else "failed"
    terminal = {
        "audit_digests": [run.get("audit", {}).get("deterministic_digest_sha256") for run in runs if isinstance(run.get("audit"), dict)],
        "reset_count": sum(1 for run in runs if run.get("reset") == {"attempted": 1, "complete": 1, "failed": 0}),
        "all_resets_complete": all(run.get("reset") == {"attempted": 1, "complete": 1, "failed": 0} for run in runs),
    }
    return {
        "schema_version": "ullm.aq4_p2_resident_batch_raw.v1",
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "status": status,
        "immutable_status": status != "ok",
        "baseline_identity": {
            "run_id": run_id,
            "kind": baseline_kind,
            "identity_file": identity_link,
        },
        "resident": {"session_id": session_id, "model_loads": 1, "driver_identity": driver_identity, "case_reset_count": sum(1 for run in runs if run.get("reset") == {"attempted": 1, "complete": 1, "failed": 0})},
        "workload": {key: case.get(key) for key in ("scope", "phase", "mode", "prompt_tokens", "cached_prefix_tokens", "context_tokens", "prefill_requested_m", "resolved_m", "request_count", "generated_tokens")},
        "schedule": {"warmup_runs": WARMUP_RUNS, "measured_runs": MEASURED_RUNS, "completed_runs": len(runs)},
        "runs": runs,
        "terminal": terminal,
        "failure_reason": failure_reason,
        "links": {"fixture": {"path": fixture_entry["fixture_path"], "sha256": fixture_entry["fixture_sha256"]}, "identity": identity_link, "policy": policy_link},
    }


def build_plan(cases: list[dict[str, Any]], expanded_path: Path, fixture_index_path: Path, run_id: str, baseline_kind: str, identity: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    token_sum = sum(int(case["prompt_tokens"]) for case in cases) * (WARMUP_RUNS + MEASURED_RUNS)
    return {
        "schema_version": SCHEMA,
        "status": "dry_run",
        "scope": "full_model",
        "case_count": len(cases),
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "transaction_count": len(cases) * (WARMUP_RUNS + MEASURED_RUNS),
        "prompt_tokens_across_transactions": token_sum,
        "resident_model_loads": 1,
        "baseline_identity": {
            "run_id": run_id,
            "kind": baseline_kind,
            "identity_file": {"path": str(identity.get("_path", "")), "sha256": identity.get("_sha256")},
            "served_model_manifest_sha256": identity.get("hash_binding", {}).get("served_model_manifest_sha256"),
            "worker_binary_sha256": identity.get("hash_binding", {}).get("worker_binary_sha256"),
            "build_git_commit": identity.get("build_git_commit"),
        },
        "links": {"expanded": {"path": str(expanded_path), "sha256": sha_file(expanded_path, "expanded")}, "fixture_index": {"path": str(fixture_index_path), "sha256": sha_file(fixture_index_path, "fixture index")}, "policy": {"path": str(policy.get("_path", "")), "sha256": policy.get("_sha256")}},
    }


def run_batch(args: argparse.Namespace) -> int:
    expanded = load(args.expanded, "expanded")
    fixture_index = load(args.fixture_index, "fixture index")
    identity = load(args.identity, "identity")
    policy = load(args.policy, "policy")
    identity_link = {"path": str(args.identity.resolve()), "sha256": sha_file(args.identity, "identity")}
    policy_link = {"path": str(args.policy.resolve()), "sha256": sha_file(args.policy, "policy")}
    identity["_path"], identity["_sha256"] = str(args.identity.resolve()), identity_link["sha256"]
    policy["_path"], policy["_sha256"] = str(args.policy.resolve()), policy_link["sha256"]
    if args.baseline_kind not in {"active-production", "p3-current-head"}:
        raise BatchError("baseline kind must identify one immutable build/run")
    cases = select_target_cases(expanded, fixture_index)
    plan = build_plan(cases, args.expanded, args.fixture_index, args.run_id, args.baseline_kind, identity, policy)
    if args.dry_run:
        atomic_write(args.output_dir / "resident-batch.plan.json", plan)
        return 0
    if not args.driver_command:
        raise BatchError("--driver-command is required unless --dry-run is set")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    process = subprocess.Popen(args.driver_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=False, bufsize=1)
    session_id, driver_identity = validate_ready(_recv(process, args.timeout), identity, cases)
    by_id = {entry["case_id"]: entry for entry in fixture_index["cases"]}
    completed_cases = 0
    try:
        for case in cases:
            fixture_entry = by_id[case["case_id"]]
            _send(process, {"command": "case_begin", "case_id": case["case_id"], "case_sha256": case["case_sha256"], "fixture_path": fixture_entry["fixture_path"], "fixture_sha256": fixture_entry["fixture_sha256"], "requested_m": case["prefill_requested_m"], "resolved_m": case["resolved_m"]})
            begin = _recv(process, args.timeout)
            if begin.get("event") != "case_ready" or begin.get("resident_session_id") != session_id:
                raise BatchError(f"resident driver case begin failed: {case['case_id']}")
            runs: list[dict[str, Any]] = []
            for run_index in range(WARMUP_RUNS + MEASURED_RUNS):
                run_kind = "warmup" if run_index < WARMUP_RUNS else "measured"
                _send(process, {"command": "run", "case_id": case["case_id"], "run_index": run_index, "run_kind": run_kind})
                value = validate_run(_recv(process, args.timeout), case, session_id)
                value["run_index"], value["run_kind"] = run_index, run_kind
                runs.append(value)
                if value["status"] == "oom":
                    break
            _send(process, {"command": "case_end", "case_id": case["case_id"]})
            end = _recv(process, args.timeout)
            if end.get("event") != "case_complete" or end.get("resident_session_id") != session_id:
                raise BatchError(f"resident driver case end failed: {case['case_id']}")
            raw = make_case_raw(case, fixture_entry, identity_link, policy_link, args.run_id, args.baseline_kind, session_id, driver_identity, runs, "resident_driver_oom" if any(item["status"] == "oom" for item in runs) else None)
            atomic_write(args.output_dir / f"{case['case_id']}.raw.json", raw)
            completed_cases += 1
            if raw["status"] == "oom":
                raise BatchError(f"resident driver OOM at {case['case_id']}; remaining cases were not executed")
    finally:
        try:
            _send(process, {"command": "shutdown"})
        except (BatchError, OSError):
            pass
        try:
            process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
    atomic_write(args.output_dir / "resident-batch.summary.json", {**plan, "status": "complete", "completed_cases": completed_cases})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--fixture-index", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-kind", choices=("active-production", "p3-current-head"), required=True)
    parser.add_argument("--driver-command", nargs="+")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run_batch(args)
    except (BatchError, OSError, subprocess.SubprocessError) as error:
        print(f"AQ4 P2 resident batch failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
