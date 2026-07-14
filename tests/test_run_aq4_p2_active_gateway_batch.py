from __future__ import annotations

import http.server
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aq4_active_gateway_batch", ROOT / "tools/run-aq4-p2-active-gateway-batch.py")
assert SPEC and SPEC.loader
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)

RUNTIME = {
    "runtime_device_index": 1,
    "device_id": "r9700-rdna4",
    "backend": "hip",
    "name": "AMD Radeon Graphics",
    "architecture": "gfx1201",
}
CAPABILITY = {"configurable": True, "fixed_m": None}
BOUND = {
    "served_model_manifest_sha256": "a" * 64,
    "worker_binary_sha256": "b" * 64,
    "guard_set_sha256": "c" * 64,
    "runtime_device": RUNTIME,
    "m_capability": CAPABILITY,
}


def _case(tmp_path: Path, prompt: int, requested_m: int, mode: str, index: int) -> tuple[dict, dict]:
    case = {
        "case_id": f"p2-representative-production_server-cold_prefill-{mode}-n{prompt}-m{requested_m}-r9700-aq4_0_target-{index}",
        "case_sha256": None,
        "stage_id": "representative",
        "scope": "production_server",
        "phase": "cold_prefill",
        "mode": mode,
        "prompt_tokens": prompt,
        "cached_prefix_tokens": 0,
        "context_tokens": prompt,
        "prefill_requested_m": requested_m,
        "resolved_m": 1 if mode == "all_m1" else requested_m,
        "request_count": 1,
        "generated_tokens": 0,
        "control_id": "aq4_0_target",
        "device": RUNTIME,
    }
    case["case_sha256"] = BATCH.case_hash(case)
    fixture = tmp_path / f"{case['case_id']}.fixture.json"
    fixture.write_text(json.dumps({"cases": [{"case_id": case["case_id"], "prompt_token_ids": [1] * prompt, "step_count": 1}]}), encoding="utf-8")
    entry = {
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "fixture_path": str(fixture),
        "fixture_sha256": BATCH.sha_file(fixture, "fixture"),
        "prompt_tokens": prompt,
        "context_tokens": prompt,
        "generated_tokens": 0,
    }
    return case, entry


def _bundle(tmp_path: Path, capability: dict | None = None) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    expanded, index, identity, policy = (tmp_path / name for name in ("expanded.json", "fixture-index.json", "identity.json", "policy.json"))
    cases, entries = [], []
    ordinal = 0
    for prompt in (1011, 1024, 1339, 2048):
        for requested_m in (1, 8, 16, 32, 64, 128):
            for mode in ("all_m1", "cold_batched"):
                case, entry = _case(tmp_path, prompt, requested_m, mode, ordinal)
                cases.append(case)
                entries.append(entry)
                ordinal += 1
    expanded.write_text(json.dumps({"schema_version": "ullm.aq4_production_p2_expanded.v2", "cases": cases}), encoding="utf-8")
    index.write_text(json.dumps({"schema_version": "ullm.aq4_p2_fixture_index.v1", "case_count": len(entries), "cases": entries}), encoding="utf-8")
    bound = {**BOUND, "m_capability": capability or CAPABILITY}
    identity.write_text(json.dumps({"schema_version": "ullm.aq4_production_p2_identity.v2", "status": "bound", "active_gateway_identity": bound, "hash_binding": {"served_model_manifest_sha256": bound["served_model_manifest_sha256"], "worker_binary_sha256": bound["worker_binary_sha256"]}}), encoding="utf-8")
    policy.write_text(json.dumps({"schema_version": "ullm.aq4_production_p2_threshold_policy.v1", "status": "bound"}), encoding="utf-8")
    return expanded, index, identity, policy


class _GatewayState:
    def __init__(self, *, transport: str, drift: str | None = None, reset_bad: bool = False, error_429: bool = False, capability: dict | None = None):
        self.transport = transport
        self.drift = drift
        self.reset_bad = reset_bad
        self.error_429 = error_429
        self.capability = capability or CAPABILITY
        self.identity_calls = 0
        self.request_count = 0


def _server(state: _GatewayState) -> tuple[http.server.ThreadingHTTPServer, threading.Thread]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            pass

        def _json(self, status: int, value: dict) -> None:
            raw = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path == "/readyz":
                self._json(200, {"status": "ready"})
                return
            if self.path == "/v1/identity":
                state.identity_calls += 1
                manifest = "0" * 64 if state.drift == "identity" else BOUND["served_model_manifest_sha256"]
                pid = 5000 + (1 if state.drift == "pid" and state.identity_calls > 2 else 0)
                self._json(200, {"schema_version": BATCH.IDENTITY_SCHEMA, "status": "ready", **BOUND, "m_capability": state.capability, "served_model_manifest_sha256": manifest, "gateway_pid": pid, "gateway_starttime_ticks": 9000})
                return
            if self.path.startswith("/v1/evidence/"):
                request_id = self.path.rsplit("/", 1)[-1]
                self._json(200, {"trace_sidecar": {"schema_version": "ullm.production_execution_trace.v1", "trace_id": f"trace-{request_id}"}, "resource": {"samples": [{"monotonic_ms": 1}], "peak": {"vram_used_bytes": 1}}, "release": {"complete": True}, "reset": {"attempted": 1, "complete": 1, "failed": 0}})
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size) or b"{}")
            if self.path == "/v1/chat/completions":
                state.request_count += 1
                if state.error_429 and body.get("run_index") == 0:
                    self._json(429, {"error": "busy"})
                    return
                request_id = f"req-{state.request_count}"
                value = {"request_id": request_id, "actual_m": body["prefill_requested_m"], "usage": {"prompt_tokens": body["prompt_tokens"]}}
                if body.get("stream"):
                    raw = f"data: {json.dumps(value)}\n\ndata: [DONE]\n\n".encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                else:
                    self._json(200, value)
                return
            if self.path == "/v1/cases/release":
                value = {"case_id": body["case_id"], "release": {"complete": True}}
                if not state.reset_bad:
                    value["reset"] = {"attempted": 1, "complete": 1, "failed": 0}
                self._json(200, value)
                return
            self._json(404, {"error": "not found"})

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _run(bundle: tuple[Path, Path, Path, Path], output: Path, server: http.server.ThreadingHTTPServer, transport: str) -> subprocess.CompletedProcess[str]:
    expanded, index, identity, policy = bundle
    command = [sys.executable, str(ROOT / "tools/run-aq4-p2-active-gateway-batch.py"), "--expanded", str(expanded), "--fixture-index", str(index), "--identity", str(identity), "--policy", str(policy), "--output-dir", str(output), "--run-id", f"gateway-{transport}", "--base-url", f"http://127.0.0.1:{server.server_port}", "--transport", transport]
    return subprocess.run(command, text=True, capture_output=True)


def test_gateway_dry_run_is_exact_48_target_matrix(tmp_path: Path) -> None:
    expanded, index, identity, policy = _bundle(tmp_path / "plan")
    output = tmp_path / "plan-run"
    command = [sys.executable, str(ROOT / "tools/run-aq4-p2-active-gateway-batch.py"), "--expanded", str(expanded), "--fixture-index", str(index), "--identity", str(identity), "--policy", str(policy), "--output-dir", str(output), "--run-id", "gateway-plan", "--base-url", "http://127.0.0.1:1", "--transport", "stream", "--dry-run"]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    plan = json.loads((output / "active-gateway-batch.plan.json").read_text())
    assert plan["case_count"] == 48
    assert plan["transaction_count"] == 48 * 12
    assert plan["prompt_tokens_across_transactions"] == 780_768
    assert plan["transport"] == "stream"


def test_gateway_stream_and_nonstream_record_identity_trace_resource_release(tmp_path: Path) -> None:
    for transport in ("stream", "nonstream"):
        state = _GatewayState(transport=transport)
        server, _thread = _server(state)
        try:
            completed = _run(_bundle(tmp_path / transport), tmp_path / f"{transport}-run", server, transport)
            assert completed.returncode == 0, completed.stderr
            raws = list((tmp_path / f"{transport}-run").glob("*.raw.json"))
            assert len(raws) == 48
            sample = json.loads(raws[0].read_text())
            assert sample["status"] == "ok"
            assert sample["transport"] == transport
            assert sample["gateway_identities"][0]["gateway_pid"] == 5000
            assert sample["runs"][0]["trace_sidecar"]["schema_version"] == "ullm.production_execution_trace.v1"
            assert sample["terminal"]["reset"] == {"attempted": 1, "complete": 1, "failed": 0}
        finally:
            server.shutdown()


def test_gateway_identity_and_pid_drift_fail_closed(tmp_path: Path) -> None:
    for drift in ("identity", "pid"):
        state = _GatewayState(transport="nonstream", drift=drift)
        server, _thread = _server(state)
        try:
            completed = _run(_bundle(tmp_path / drift), tmp_path / f"{drift}-run", server, "nonstream")
            assert completed.returncode != 0
            assert state.request_count == (0 if drift == "identity" else 12)
        finally:
            server.shutdown()


def test_gateway_reset_missing_and_429_are_recorded(tmp_path: Path) -> None:
    state = _GatewayState(transport="nonstream", reset_bad=True, error_429=True)
    server, _thread = _server(state)
    try:
        completed = _run(_bundle(tmp_path / "mutations"), tmp_path / "mutation-run", server, "nonstream")
        assert completed.returncode == 0
        sample = json.loads(next((tmp_path / "mutation-run").glob("*.raw.json")).read_text())
        assert sample["status"] == "failed"
        assert sample["runs"][0]["failure_reason"] == "gateway_http_429"
        assert sample["terminal"]["release"]["failure_reason"] == "gateway_release_reset_incomplete"
    finally:
        server.shutdown()


def test_gateway_fixed_m_is_unsupported_not_misreported(tmp_path: Path) -> None:
    capability = {"configurable": False, "fixed_m": 1}
    state = _GatewayState(transport="nonstream", capability=capability)
    server, _thread = _server(state)
    try:
        completed = _run(_bundle(tmp_path / "fixed-m", capability), tmp_path / "fixed-m-run", server, "nonstream")
        assert completed.returncode == 0
        raws = list((tmp_path / "fixed-m-run").glob("*.raw.json"))
        assert len(raws) == 48
        unsupported = [json.loads(path.read_text()) for path in raws if json.loads(path.read_text())["status"] == "unsupported"]
        assert len(unsupported) == 40
        assert unsupported[0]["m_configuration"]["reason_code"] == "gateway_m_configuration_unavailable"
        assert state.request_count == 8 * 12
    finally:
        server.shutdown()
