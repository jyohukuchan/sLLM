#!/usr/bin/env python3
"""Run fail-closed Phase 6 A6 service evidence on one exact AMD target."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


def post(base_url: str, payload: dict[str, object], *, stream: bool = False) -> tuple[bytes, int]:
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic_ns()
    with urllib.request.urlopen(request, timeout=300) as response:
        expected = "text/event-stream" if stream else "application/json"
        if response.status != 200 or response.headers.get_content_type() != expected:
            raise RuntimeError("service response status/content type differs")
        body = response.read()
    return body, time.monotonic_ns() - started


def parse_sse(body: bytes) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    terminal = 0
    for block in body.decode("utf-8").split("\n\n"):
        if not block:
            continue
        data = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if len(data) != 1:
            raise RuntimeError("SSE event is not one data line")
        if data[0] == "[DONE]":
            terminal += 1
        else:
            chunks.append(json.loads(data[0]))
    if terminal != 1 or not chunks:
        raise RuntimeError("SSE stream does not have one terminal and JSON chunks")
    ids = {chunk.get("id") for chunk in chunks}
    if len(ids) != 1 or None in ids:
        raise RuntimeError("SSE chunk identity is not stable")
    if chunks[0]["choices"][0]["delta"].get("role") != "assistant":
        raise RuntimeError("SSE first chunk is not assistant role")
    if chunks[-1]["choices"][0].get("finish_reason") not in {"stop", "length"}:
        raise RuntimeError("SSE final chunk has no profile finish reason")
    return chunks


def request_payload(max_tokens: int, *, stream: bool = False, stop: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": "qwen3.5-4b",
        "messages": [{"role": "user", "content": "Reply with one word."}],
        "temperature": 0,
        "max_completion_tokens": max_tokens,
        "stream": stream,
    }
    if stop is not None:
        payload["stop"] = stop
    return payload


def phase5_render_payload() -> dict[str, object]:
    """Return the exact Phase 5 render/tokenize baseline request identity."""
    return {
        "model": "qwen3.5-4b",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
        "max_completion_tokens": 17,
        "stream": False,
    }


def disconnect_after_content(host: str, port: int) -> None:
    payload = json.dumps(
        {
            "model": "qwen3.5-4b",
            "messages": [{"role": "user", "content": "Count slowly."}],
            "temperature": 0,
            "max_completion_tokens": 17,
            "stream": True,
        }
    ).encode("utf-8")
    request = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
    ).encode("ascii") + payload
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.settimeout(300)
        connection.sendall(request)
        received = b""
        while received.count(b"\n\n") < 2:
            chunk = connection.recv(4096)
            if not chunk:
                raise RuntimeError("service closed before first content SSE event")
            received += chunk
        if b'"role":"assistant"' not in received or b'"content":"' not in received:
            raise RuntimeError("disconnect probe did not observe role and content events")
        # Context-manager close is the intentional client disconnect.


def official_client_smoke(python: Path, base_url: str) -> dict[str, object]:
    script = r'''
import json,sys
from openai import OpenAI
client=OpenAI(api_key="local-test-key",base_url=sys.argv[1]+"/v1",timeout=300.0)
r=client.chat.completions.create(model="qwen3.5-4b",messages=[{"role":"user","content":"Reply with one word."}],temperature=0,max_completion_tokens=1)
chunks=list(client.chat.completions.create(model="qwen3.5-4b",messages=[{"role":"user","content":"Reply with one word."}],temperature=0,max_completion_tokens=1,stream=True))
print(json.dumps({"version":__import__("openai").__version__,"object":r.object,"role":r.choices[0].message.role,"finish_reason":r.choices[0].finish_reason,"stream_chunks":len(chunks),"stream_finish_reason":chunks[-1].choices[0].finish_reason},sort_keys=True))
'''
    result = subprocess.run(
        [str(python), "-c", script, base_url],
        check=True,
        capture_output=True,
        text=True,
        timeout=660,
    )
    value = json.loads(result.stdout.strip().splitlines()[-1])
    if value["object"] != "chat.completion" or value["role"] != "assistant":
        raise RuntimeError("official OpenAI client non-stream response differs")
    if value["finish_reason"] not in {"stop", "length"} or value["stream_finish_reason"] not in {"stop", "length"}:
        raise RuntimeError("official OpenAI client finish reason differs")
    return value


def amd_smi(command: str, gpu: int | None = None) -> object:
    argv = ["amd-smi", command]
    if gpu is not None:
        argv += ["--gpu", str(gpu)]
    argv.append("--json")
    result = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


def process_count(document: object, gpu: int) -> int:
    if not isinstance(document, list):
        raise RuntimeError("amd-smi process output is not a list")
    row = next((entry for entry in document if entry.get("gpu") == gpu), None)
    if not isinstance(row, dict) or not isinstance(row.get("process_list"), list):
        raise RuntimeError("amd-smi process row is absent")
    return sum(1 for value in row["process_list"] if "No running processes" not in value.get("process_info", ""))


def wait_ready(process: subprocess.Popen[str], timeout: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = process.stdout.readline() if process.stdout is not None else ""
        if line:
            value = json.loads(line)
            if value.get("event") == "ready":
                return value
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise RuntimeError(f"service exited before ready: {stderr}")
    raise TimeoutError("service did not become ready")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--amd-smi-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--target", choices=("gfx1030", "gfx1201"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--client-python", type=Path, required=True)
    args = parser.parse_args()
    pre_static = amd_smi("static", args.amd_smi_index)
    pre_metric = amd_smi("metric", args.amd_smi_index)
    pre_processes = process_count(amd_smi("process"), args.amd_smi_index)
    env = dict(os.environ)
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        env.pop(name, None)
    env["ROCR_VISIBLE_DEVICES"] = args.gpu_uuid
    env["LD_LIBRARY_PATH"] = "/opt/rocm/lib:/opt/rocm/lib64" + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    command = [
        str(args.binary),
        "--lock", str(args.lock),
        "--cache", str(args.cache),
        "--device-index", str(args.device_index),
        "--target", args.target,
        "--listen", f"127.0.0.1:{args.port}",
        "--model", "qwen3.5-4b",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        ready = wait_ready(process, 180)
        if ready.get("target") != args.target:
            raise RuntimeError("ready target differs")
        base_url = f"http://127.0.0.1:{args.port}"
        non_stream_body, non_stream_ns = post(base_url, request_payload(1))
        non_stream = json.loads(non_stream_body)
        text = non_stream["choices"][0]["message"]["content"]
        if non_stream["object"] != "chat.completion" or not text:
            raise RuntimeError("non-stream smoke differs")
        stream_body, stream_ns = post(base_url, request_payload(1, stream=True), stream=True)
        chunks = parse_sse(stream_body)
        visible = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
        if visible != text or chunks[-1]["usage"] != non_stream["usage"]:
            raise RuntimeError("stream/non-stream text or usage differs")
        phase5_body, phase5_http_ns = post(base_url, phase5_render_payload())
        phase5_response = json.loads(phase5_body)
        if phase5_response["usage"]["prompt_tokens"] != 13:
            raise RuntimeError("Phase 5 render baseline request identity differs")
        stop_body, _ = post(base_url, request_payload(1, stop=text))
        stop_response = json.loads(stop_body)
        if stop_response["choices"][0]["finish_reason"] != "stop" or stop_response["choices"][0]["message"]["content"]:
            raise RuntimeError("stop-string smoke differs")
        boundary_capacities = []
        prompt_tokens = int(non_stream["usage"]["prompt_tokens"])
        for capacity in (1023, 1024, 1025):
            max_tokens = capacity - prompt_tokens
            if max_tokens < 1:
                raise RuntimeError("page-boundary max token derivation failed")
            body, _ = post(base_url, request_payload(max_tokens, stop=text))
            if json.loads(body)["choices"][0]["finish_reason"] != "stop":
                raise RuntimeError("page-boundary stop request did not terminate immediately")
            boundary_capacities.append(capacity)
        client = official_client_smoke(args.client_python, base_url)
        disconnect_after_content("127.0.0.1", args.port)
        recovery_body, recovery_ns = post(base_url, request_payload(1))
        if json.loads(recovery_body)["object"] != "chat.completion":
            raise RuntimeError("post-disconnect recovery failed")

        first_elapsed: dict[str, int] = {}
        second_elapsed: dict[str, int] = {}
        barrier = threading.Barrier(3)
        def queued(name: str, destination: dict[str, int]) -> None:
            barrier.wait()
            _, elapsed = post(base_url, request_payload(1))
            destination[name] = elapsed
        one = threading.Thread(target=queued, args=("one", first_elapsed))
        two = threading.Thread(target=queued, args=("two", second_elapsed))
        one.start(); two.start(); barrier.wait(); one.join(); two.join()
        queue_pair = sorted([first_elapsed["one"], second_elapsed["two"]])
        queue_wait_residual_ns = max(0, queue_pair[1] - queue_pair[0])
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=180)
    shutdown_events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    shutdown = next((value["report"] for value in shutdown_events if value.get("event") == "shutdown_audit"), None)
    if not isinstance(shutdown, dict):
        raise RuntimeError(f"service produced no shutdown audit: {stderr}")
    if shutdown["target"] != args.target or shutdown["final_current_bytes"] != 0:
        raise RuntimeError("shutdown target or allocation cleanup differs")
    audits = shutdown["requests"]
    completed = [audit for audit in audits if audit["outcome"] == "completed"]
    cancelled = [audit for audit in audits if audit["outcome"] == "cancelled"]
    if not completed or not cancelled:
        raise RuntimeError("completed/cancelled request audit is absent")
    for audit in completed:
        if audit["selected_backend"] != "hip" or audit["target"] != args.target or audit["fallback_used"] or not audit["all_dispatches_hip"]:
            raise RuntimeError("completed request is not exact HIP/no-fallback")
        if audit["cleanup_request_state_bytes"] != 0 or audit["cleanup_workspace_bytes"] != 0:
            raise RuntimeError("completed request cleanup differs")
    disconnected = cancelled[-1]
    if disconnected["selected_backend"] != "hip" or disconnected["submission_count"] is None:
        raise RuntimeError("disconnect did not occur after real HIP dispatch")
    if disconnected["cleanup_request_state_bytes"] != 0 or disconnected["cleanup_workspace_bytes"] != 0:
        raise RuntimeError("disconnect request cleanup differs")
    boundary = [audit for audit in completed if audit["logical_kv_capacity_tokens"] in {1023, 1024, 1025}]
    if sorted(audit["logical_kv_capacity_tokens"] for audit in boundary) != [1023, 1024, 1025]:
        raise RuntimeError("full-model service page-boundary audits are absent")
    if any(audit["physical_page_bytes"] <= 0 or audit["committed_kv_bytes"] <= 0 for audit in boundary):
        raise RuntimeError("page-boundary physical KV evidence is absent")
    post_metric = amd_smi("metric", args.amd_smi_index)
    post_processes = process_count(amd_smi("process"), args.amd_smi_index)
    if post_processes != pre_processes:
        raise RuntimeError("GPU process count did not return to pre-run value")
    # Request order is fixed up to the final two concurrent queue probes.
    non_stream_backend_ns = completed[0]["elapsed_ns"]
    stream_backend_ns = completed[1]["elapsed_ns"]
    phase5_backend_ns = completed[2]["elapsed_ns"]
    report = {
        "schema_version": "openai-a6-gpu-evidence-v1",
        "result": "PASS",
        "target": args.target,
        "device_index": args.device_index,
        "amd_smi_index": args.amd_smi_index,
        "visibility": {"selector": "ROCR_VISIBLE_DEVICES", "gpu_uuid": args.gpu_uuid},
        "ready": ready,
        "model_fingerprint": shutdown["model_fingerprint"],
        "plan_digest": shutdown["plan_digest"],
        "official_client": client,
        "raw_sse_chunks": len(chunks),
        "boundary_capacities": boundary_capacities,
        "disconnect": disconnected,
        "overhead": {
            "non_stream_json_residual_ns": max(0, non_stream_ns - non_stream_backend_ns),
            "stream_sse_residual_ns": max(0, stream_ns - stream_backend_ns),
            "queue_wait_residual_ns": queue_wait_residual_ns,
            "post_disconnect_recovery_http_ns": recovery_ns,
            "phase5_render_case": {
                "case_id": "chat-hello",
                "input_tokens": 13,
                "requested_output_tokens": 17,
                "phase5_matrix_revision": 1,
                "service_backend_elapsed_ns": phase5_backend_ns,
                "service_http_elapsed_ns": phase5_http_ns,
                "json_http_residual_ns": max(0, phase5_http_ns - phase5_backend_ns),
            },
        },
        "shutdown": shutdown,
        "health": {
            "pre_process_count": pre_processes,
            "post_process_count": post_processes,
            "pre_static": pre_static,
            "pre_metric": pre_metric,
            "post_metric": post_metric,
        },
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
