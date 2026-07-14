#!/usr/bin/env python3
"""Run the representative AQ4 P2 production-server matrix through the active gateway.

This runner intentionally owns the HTTP boundary separately from the direct resident driver.
It records the active manifest/worker/PID identity before each case, sends only a fixed request
shape, and keeps stream/non-stream, resource, trace, release, reset, 429, and M failures in
case-scoped immutable raw artifacts.  Tests use a local fake HTTP server; this tool never starts
or restarts a service by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
CASE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SCHEMA = "ullm.aq4_p2_active_gateway_batch.v1"
IDENTITY_SCHEMA = "ullm.aq4_p2_active_gateway_identity.v1"
WARMUP_RUNS = 2
MEASURED_RUNS = 10
TRANSPORTS = {"stream", "nonstream"}
RUNTIME_DEVICE_KEYS = {"runtime_device_index", "device_id", "backend", "name", "architecture"}
GATEWAY_IDENTITY_KEYS = {
    "served_model_manifest_sha256",
    "worker_binary_sha256",
    "guard_set_sha256",
    "gateway_pid",
    "gateway_starttime_ticks",
    "runtime_device",
    "m_capability",
}


class GatewayBatchError(ValueError):
    pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        if key in value:
            raise GatewayBatchError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def reject_constant(item: str) -> Any:
    raise GatewayBatchError(f"non-finite JSON value: {item}")


def load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise GatewayBatchError(f"{label} must be a bounded regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError, GatewayBatchError) as error:
        raise GatewayBatchError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise GatewayBatchError(f"{label} root must be an object")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise GatewayBatchError(f"{label} must be a regular file")
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
        raise GatewayBatchError(f"refusing to overwrite {path}")
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
        raise GatewayBatchError(f"refusing to overwrite {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def select_target_cases(expanded: dict[str, Any], fixture_index: dict[str, Any]) -> list[dict[str, Any]]:
    if expanded.get("schema_version") != "ullm.aq4_production_p2_expanded.v2":
        raise GatewayBatchError("expanded manifest schema differs")
    if fixture_index.get("schema_version") != "ullm.aq4_p2_fixture_index.v1":
        raise GatewayBatchError("fixture index schema differs")
    cases = expanded.get("cases")
    index_cases = fixture_index.get("cases")
    if not isinstance(cases, list) or not isinstance(index_cases, list):
        raise GatewayBatchError("expanded or fixture index cases are missing")
    selected = [
        case for case in cases
        if isinstance(case, dict)
        and case.get("stage_id") == "representative"
        and case.get("scope") == "production_server"
        and case.get("phase") == "cold_prefill"
        and case.get("device", {}).get("device_id") == "r9700-rdna4"
        and case.get("control_id") == "aq4_0_target"
    ]
    if len(selected) != 48:
        raise GatewayBatchError(f"representative production_server target profile must contain 48 cases, got {len(selected)}")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in index_cases:
        if not isinstance(entry, dict) or not isinstance(entry.get("case_id"), str) or not entry["case_id"] or entry["case_id"] in by_id:
            raise GatewayBatchError("fixture index contains invalid or duplicate case IDs")
        by_id[entry["case_id"]] = entry
    if len(by_id) != fixture_index.get("case_count"):
        raise GatewayBatchError("fixture index case coverage differs")
    for case in selected:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or CASE_ID_RE.fullmatch(case_id) is None or case.get("case_sha256") != case_hash(case):
            raise GatewayBatchError(f"selected case identity differs: {case_id}")
        entry = by_id.get(case_id)
        if not isinstance(entry, dict) or any(entry.get(field) != case.get(field) for field in ("case_sha256", "prompt_tokens", "context_tokens", "generated_tokens")):
            raise GatewayBatchError(f"fixture index does not bind selected case: {case_id}")
        fixture_path = Path(entry.get("fixture_path", ""))
        if sha_file(fixture_path, "fixture") != entry.get("fixture_sha256"):
            raise GatewayBatchError(f"fixture hash differs: {case_id}")
    return sorted(selected, key=lambda item: item["case_id"])


def _runtime_device(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != RUNTIME_DEVICE_KEYS:
        raise GatewayBatchError(f"{label} runtime device fields differ")
    if type(value["runtime_device_index"]) is not int or value["runtime_device_index"] < 0:
        raise GatewayBatchError(f"{label} runtime device index is invalid")
    if not isinstance(value["device_id"], (str, int)) or isinstance(value["device_id"], bool) or (isinstance(value["device_id"], str) and not value["device_id"]):
        raise GatewayBatchError(f"{label} runtime device id is invalid")
    for field in ("backend", "name", "architecture"):
        if not isinstance(value[field], str) or not value[field]:
            raise GatewayBatchError(f"{label} runtime device {field} is invalid")
    return value


def _m_capability(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"configurable", "fixed_m"} or type(value["configurable"]) is not bool:
        raise GatewayBatchError(f"{label} M capability fields differ")
    if value["fixed_m"] is not None and (type(value["fixed_m"]) is not int or value["fixed_m"] <= 0):
        raise GatewayBatchError(f"{label} fixed M is invalid")
    if value["configurable"] and value["fixed_m"] is not None:
        raise GatewayBatchError(f"{label} configurable M cannot have a fixed value")
    if not value["configurable"] and value["fixed_m"] is None:
        raise GatewayBatchError(f"{label} non-configurable M lacks fixed value")
    return value


def validate_identity(value: Any, identity: dict[str, Any], cases: list[dict[str, Any]], pinned_pid: tuple[int, int] | None = None) -> tuple[dict[str, Any], tuple[int, int]]:
    if not isinstance(value, dict) or value.get("schema_version") != IDENTITY_SCHEMA or value.get("status") != "ready" or set(value) != {"schema_version", "status", *GATEWAY_IDENTITY_KEYS}:
        raise GatewayBatchError("active gateway identity envelope differs")
    for field in ("served_model_manifest_sha256", "worker_binary_sha256", "guard_set_sha256"):
        if not isinstance(value[field], str) or SHA256_RE.fullmatch(value[field]) is None:
            raise GatewayBatchError(f"active gateway identity.{field} is invalid")
    if type(value["gateway_pid"]) is not int or value["gateway_pid"] <= 0 or type(value["gateway_starttime_ticks"]) is not int or value["gateway_starttime_ticks"] <= 0:
        raise GatewayBatchError("active gateway PID identity is invalid")
    runtime = _runtime_device(value["runtime_device"], "active gateway")
    capability = _m_capability(value["m_capability"], "active gateway")
    bound = identity.get("active_gateway_identity")
    if not isinstance(bound, dict) or set(bound) != {"served_model_manifest_sha256", "worker_binary_sha256", "guard_set_sha256", "runtime_device", "m_capability"}:
        raise GatewayBatchError("identity file lacks active gateway identity")
    if any(value[field] != bound[field] for field in ("served_model_manifest_sha256", "worker_binary_sha256", "guard_set_sha256")) or runtime != bound["runtime_device"] or capability != bound["m_capability"]:
        raise GatewayBatchError("active gateway identity differs from identity file")
    if pinned_pid is not None and (value["gateway_pid"], value["gateway_starttime_ticks"]) != pinned_pid:
        raise GatewayBatchError("active gateway PID changed during matrix")
    for case in cases:
        device = case.get("device")
        if not isinstance(device, dict):
            raise GatewayBatchError(f"case device identity is missing: {case.get('case_id')}")
        for field in RUNTIME_DEVICE_KEYS:
            if field in device and device[field] != runtime[field]:
                raise GatewayBatchError(f"active gateway runtime device differs from case: {case['case_id']}")
    return value, (value["gateway_pid"], value["gateway_starttime_ticks"])


class GatewayHttp:
    def __init__(self, base_url: str, timeout: float, transport: str):
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise GatewayBatchError("base URL must be an HTTP(S) origin")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, bytes, str]:
        if not path.startswith("/") or ".." in Path(path).parts:
            raise GatewayBatchError("gateway path is invalid")
        data = None if body is None else canonical(body)
        headers = {"Accept": "application/json", "User-Agent": "ullm-aq4-p2-active-gateway/1"}
        if data is not None:
            headers["Content-Type"] = "application/json"
            if self.transport == "stream" and path == "/v1/chat/completions":
                headers["Accept"] = "text/event-stream"
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read(MAX_RESPONSE_BYTES + 1)
                status = int(response.status)
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as error:
            payload = error.read(MAX_RESPONSE_BYTES + 1)
            status = int(error.code)
            content_type = error.headers.get("Content-Type", "") if error.headers else ""
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise GatewayBatchError(f"gateway HTTP request failed: {error}") from error
        if len(payload) > MAX_RESPONSE_BYTES:
            raise GatewayBatchError("gateway response exceeds bound")
        return status, payload, content_type

    def ready(self) -> dict[str, Any]:
        status, payload, _ = self._request("GET", "/readyz")
        if status != 200:
            raise GatewayBatchError(f"gateway readyz status {status}")
        value = json.loads(payload, object_pairs_hook=pairs, parse_constant=reject_constant)
        if not isinstance(value, dict) or value.get("status") != "ready":
            raise GatewayBatchError("gateway readyz body differs")
        status, payload, _ = self._request("GET", "/v1/identity")
        if status != 200:
            raise GatewayBatchError(f"gateway identity status {status}")
        value = json.loads(payload, object_pairs_hook=pairs, parse_constant=reject_constant)
        if not isinstance(value, dict):
            raise GatewayBatchError("gateway identity body differs")
        return value

    def case_identity(self) -> dict[str, Any]:
        return self.ready()

    def request(self, case: dict[str, Any], fixture_entry: dict[str, Any], run_index: int) -> dict[str, Any]:
        payload = {
            "model": "ullm-qwen3.5-9b-aq4",
            "prompt_fixture_sha256": fixture_entry["fixture_sha256"],
            "prompt_tokens": case["prompt_tokens"],
            "context_tokens": case["context_tokens"],
            "prefill_requested_m": case["prefill_requested_m"],
            "max_tokens": case["generated_tokens"],
            "stream": self.transport == "stream",
            "run_index": run_index,
        }
        status, raw, content_type = self._request("POST", "/v1/chat/completions", payload)
        result: dict[str, Any] = {"http_status": status, "content_type": content_type, "response_sha256": sha_bytes(raw), "stream": self.transport == "stream"}
        if status != 200:
            result["status"] = "failed"
            result["failure_reason"] = "gateway_http_429" if status == 429 else "gateway_http_error"
            return result
        if self.transport == "stream":
            lines = raw.decode("utf-8").splitlines()
            events = [line[5:].strip() for line in lines if line.startswith("data:")]
            if not events or events[-1] != "[DONE]":
                result["status"] = "failed"
                result["failure_reason"] = "stream_terminal_missing"
                return result
            try:
                value = json.loads(events[-2], object_pairs_hook=pairs, parse_constant=reject_constant)
            except (IndexError, UnicodeError, json.JSONDecodeError, GatewayBatchError) as error:
                result["status"] = "failed"
                result["failure_reason"] = f"stream_json_invalid:{error}"
                return result
        else:
            try:
                value = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)
            except (UnicodeError, json.JSONDecodeError, GatewayBatchError) as error:
                result["status"] = "failed"
                result["failure_reason"] = f"nonstream_json_invalid:{error}"
                return result
        if not isinstance(value, dict) or not isinstance(value.get("request_id"), str) or not value["request_id"] or value.get("actual_m") != case["prefill_requested_m"]:
            result["status"] = "failed"
            result["failure_reason"] = "gateway_response_identity_or_m_differs"
            return result
        usage = value.get("usage")
        if not isinstance(usage, dict) or usage.get("prompt_tokens") != case["prompt_tokens"]:
            result["status"] = "failed"
            result["failure_reason"] = "gateway_prompt_usage_differs"
            return result
        evidence_status, evidence_raw, _ = self._request("GET", f"/v1/evidence/{urllib.parse.quote(value['request_id'], safe='')}")
        if evidence_status != 200:
            result["status"] = "failed"
            result["failure_reason"] = "gateway_evidence_missing"
            return result
        try:
            evidence = json.loads(evidence_raw, object_pairs_hook=pairs, parse_constant=reject_constant)
        except (UnicodeError, json.JSONDecodeError, GatewayBatchError) as error:
            result["status"] = "failed"
            result["failure_reason"] = f"gateway_evidence_invalid:{error}"
            return result
        if not isinstance(evidence, dict) or not isinstance(evidence.get("trace_sidecar"), dict) or not isinstance(evidence.get("resource"), dict) or not evidence["resource"].get("samples") or not isinstance(evidence["resource"].get("peak"), dict) or evidence.get("release", {}).get("complete") is not True or evidence.get("reset") != {"attempted": 1, "complete": 1, "failed": 0}:
            result["status"] = "failed"
            result["failure_reason"] = "gateway_terminal_evidence_incomplete"
            return result
        result.update({"status": "ok", "request_id": value["request_id"], "actual_m": value["actual_m"], "usage": usage, "trace_sidecar": evidence["trace_sidecar"], "resource": evidence["resource"], "release": evidence["release"], "reset": evidence["reset"]})
        return result

    def release_case(self, case_id: str, request_ids: list[str]) -> dict[str, Any]:
        status, raw, _ = self._request("POST", "/v1/cases/release", {"case_id": case_id, "request_ids": request_ids})
        if status != 200:
            return {"status": "failed", "http_status": status, "failure_reason": "gateway_release_http_error", "response_sha256": sha_bytes(raw)}
        try:
            value = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)
        except (UnicodeError, json.JSONDecodeError, GatewayBatchError) as error:
            return {"status": "failed", "http_status": status, "failure_reason": f"gateway_release_json_invalid:{error}", "response_sha256": sha_bytes(raw)}
        if not isinstance(value, dict) or value.get("case_id") != case_id or value.get("release", {}).get("complete") is not True or value.get("reset") != {"attempted": 1, "complete": 1, "failed": 0}:
            return {"status": "failed", "http_status": status, "failure_reason": "gateway_release_reset_incomplete", "response_sha256": sha_bytes(raw)}
        return {"status": "ok", "http_status": status, "release": value["release"], "reset": value["reset"], "response_sha256": sha_bytes(raw)}


def make_raw(case: dict[str, Any], fixture_entry: dict[str, Any], identity_link: dict[str, str], policy_link: dict[str, str], run_id: str, transport: str, gateway_identities: list[dict[str, Any]], m_configuration: dict[str, Any], runs: list[dict[str, Any]], release: dict[str, Any], failure_reason: str | None = None) -> dict[str, Any]:
    status = "unsupported" if m_configuration.get("status") == "unsupported" else "ok" if not failure_reason and release.get("status") == "ok" and runs and all(run.get("status") == "ok" for run in runs) else "oom" if any(run.get("status") == "oom" for run in runs) else "failed"
    return {
        "schema_version": "ullm.aq4_p2_active_gateway_raw.v1",
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "status": status,
        "immutable_status": status != "ok",
        "run_id": run_id,
        "transport": transport,
        "gateway_identities": gateway_identities,
        "m_configuration": m_configuration,
        "workload": {key: case.get(key) for key in ("scope", "phase", "mode", "prompt_tokens", "context_tokens", "prefill_requested_m", "resolved_m", "request_count", "generated_tokens")},
        "schedule": {"warmup_runs": WARMUP_RUNS, "measured_runs": MEASURED_RUNS, "completed_runs": len(runs)},
        "runs": runs,
        "terminal": {"release": release, "reset": release.get("reset"), "trace_sidecars": [run.get("trace_sidecar") for run in runs if run.get("trace_sidecar") is not None]},
        "failure_reason": failure_reason,
        "links": {"fixture": {"path": fixture_entry["fixture_path"], "sha256": fixture_entry["fixture_sha256"]}, "identity": identity_link, "policy": policy_link},
    }


def build_plan(cases: list[dict[str, Any]], expanded: Path, fixture_index: Path, identity: dict[str, Any], policy: dict[str, Any], run_id: str, transport: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "dry_run",
        "scope": "production_server",
        "transport": transport,
        "case_count": len(cases),
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "transaction_count": len(cases) * (WARMUP_RUNS + MEASURED_RUNS),
        "prompt_tokens_across_transactions": sum(int(case["prompt_tokens"]) for case in cases) * (WARMUP_RUNS + MEASURED_RUNS),
        "gateway_model_loads": "active_service",
        "baseline_identity": {"run_id": run_id, "kind": "active-production-gateway", "served_model_manifest_sha256": identity.get("active_gateway_identity", {}).get("served_model_manifest_sha256"), "worker_binary_sha256": identity.get("active_gateway_identity", {}).get("worker_binary_sha256"), "guard_set_sha256": identity.get("active_gateway_identity", {}).get("guard_set_sha256"), "identity_file": {"path": str(identity.get("_path", "")), "sha256": identity.get("_sha256")}},
        "links": {"expanded": {"path": str(expanded), "sha256": sha_file(expanded, "expanded")}, "fixture_index": {"path": str(fixture_index), "sha256": sha_file(fixture_index, "fixture index")}, "policy": {"path": str(policy.get("_path", "")), "sha256": policy.get("_sha256")}},
    }


def run_batch(args: argparse.Namespace) -> int:
    if args.transport not in TRANSPORTS:
        raise GatewayBatchError("transport must be stream or nonstream")
    expanded = load(args.expanded, "expanded")
    fixture_index = load(args.fixture_index, "fixture index")
    identity = load(args.identity, "identity")
    policy = load(args.policy, "policy")
    identity_link = {"path": str(args.identity.resolve()), "sha256": sha_file(args.identity, "identity")}
    policy_link = {"path": str(args.policy.resolve()), "sha256": sha_file(args.policy, "policy")}
    identity["_path"], identity["_sha256"] = identity_link["path"], identity_link["sha256"]
    policy["_path"], policy["_sha256"] = policy_link["path"], policy_link["sha256"]
    cases = select_target_cases(expanded, fixture_index)
    plan = build_plan(cases, args.expanded, args.fixture_index, identity, policy, args.run_id, args.transport)
    if args.dry_run:
        atomic_write(args.output_dir / "active-gateway-batch.plan.json", plan)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    client = GatewayHttp(args.base_url, args.timeout, args.transport)
    initial_identity = client.ready()
    _, pinned_pid = validate_identity(initial_identity, identity, cases)
    gateway_identities: list[dict[str, Any]] = []
    by_id = {entry["case_id"]: entry for entry in fixture_index["cases"]}
    completed_cases = 0
    try:
        for case in cases:
            current = client.case_identity()
            current, current_pid = validate_identity(current, identity, [case], pinned_pid)
            gateway_identities.append(current)
            entry = by_id[case["case_id"]]
            m_configuration = {"requested_m": case["prefill_requested_m"], "status": "configured", "actual_m": case["prefill_requested_m"], "reason_code": None}
            capability = current.get("m_capability")
            if isinstance(capability, dict) and capability.get("configurable") is False:
                fixed_m = capability.get("fixed_m")
                if fixed_m != case["prefill_requested_m"]:
                    m_configuration = {"requested_m": case["prefill_requested_m"], "status": "unsupported", "actual_m": fixed_m, "reason_code": "gateway_m_configuration_unavailable"}
            runs: list[dict[str, Any]] = []
            release: dict[str, Any] = {"status": "not_attempted"}
            if m_configuration["status"] == "configured":
                for run_index in range(WARMUP_RUNS + MEASURED_RUNS):
                    started = time.monotonic()
                    run = client.request(case, entry, run_index)
                    run["run_index"] = run_index
                    run["run_kind"] = "warmup" if run_index < WARMUP_RUNS else "measured"
                    run["elapsed_ms"] = (time.monotonic() - started) * 1000.0
                    runs.append(run)
                release = client.release_case(case["case_id"], [run["request_id"] for run in runs if run.get("request_id")])
            raw = make_raw(case, entry, identity_link, policy_link, args.run_id, args.transport, [current], m_configuration, runs, release, next((run.get("failure_reason") for run in runs if run.get("status") != "ok"), None) or (release.get("failure_reason") if release.get("status") != "ok" else None))
            atomic_write(args.output_dir / f"{case['case_id']}.raw.json", raw)
            completed_cases += 1
    finally:
        pass
    atomic_write(args.output_dir / "active-gateway-batch.summary.json", {**plan, "status": "complete", "completed_cases": completed_cases})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--fixture-index", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--transport", choices=sorted(TRANSPORTS), required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run_batch(args)
    except (GatewayBatchError, OSError, urllib.error.URLError) as error:
        print(f"AQ4 P2 active gateway batch failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
