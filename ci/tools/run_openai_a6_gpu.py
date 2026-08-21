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


def reasoning_payload(max_tokens: int, *, stream: bool = False) -> dict[str, object]:
    """Build the bounded raw request for the opt-in separated reasoning extension."""
    payload = request_payload(max_tokens, stream=stream)
    payload["sllm"] = {"thinking": "enabled", "separate_reasoning": True}
    return payload


def seeded_sampling_payload(
    max_tokens: int, *, seed: int, stream: bool = False
) -> dict[str, object]:
    """Build a non-greedy request whose explicit seed must replay exactly."""
    payload = request_payload(max_tokens, stream=stream)
    payload.update({"temperature": 0.8, "top_p": 0.9, "seed": seed})
    return payload


def validate_reasoning_response(response: object) -> tuple[str, str]:
    """Return separated reasoning/content, rejecting malformed or leaked tags."""
    if not isinstance(response, dict) or response.get("object") != "chat.completion":
        raise RuntimeError("reasoning non-stream response object differs")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise RuntimeError("reasoning non-stream choices differ")
    message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise RuntimeError("reasoning non-stream assistant message differs")
    if "reasoning_content" not in message:
        raise RuntimeError("reasoning non-stream response omitted reasoning_content")
    reasoning = message.get("reasoning_content")
    content = message.get("content")
    if not isinstance(reasoning, str) or not isinstance(content, str):
        raise RuntimeError("reasoning non-stream fields are not strings")
    if "<think>" in reasoning or "</think>" in reasoning or "<think>" in content or "</think>" in content:
        raise RuntimeError("reasoning tags leaked into separated response")
    if not reasoning and not content:
        raise RuntimeError("reasoning response contains neither reasoning nor content")
    finish_reason = choices[0].get("finish_reason")
    if finish_reason not in {"stop", "length"}:
        raise RuntimeError("reasoning non-stream finish reason differs")
    return reasoning, content


def parse_reasoning_sse(chunks: list[dict[str, object]]) -> tuple[str, str]:
    """Validate SSE delta separation and return concatenated reasoning/content."""
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for chunk in chunks:
        choices = chunk.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise RuntimeError("reasoning SSE choices differ")
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            raise RuntimeError("reasoning SSE delta differs")
        reasoning = delta.get("reasoning_content")
        content = delta.get("content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise RuntimeError("reasoning SSE field is not a string")
        if content is not None and not isinstance(content, str):
            raise RuntimeError("content SSE field is not a string")
        if reasoning and content:
            raise RuntimeError("reasoning and content share one SSE delta")
        if reasoning:
            reasoning_parts.append(reasoning)
        if content:
            content_parts.append(content)
        if any(tag in value for value in (reasoning or "", content or "") for tag in ("<think>", "</think>")):
            raise RuntimeError("reasoning tags leaked into separated SSE")
    reasoning = "".join(reasoning_parts)
    content = "".join(content_parts)
    if not reasoning and not content:
        raise RuntimeError("reasoning SSE contains neither reasoning nor content")
    return reasoning, content


def validate_seeded_response(response: object) -> tuple[str, object]:
    """Validate the public response shape used by the seeded replay probe."""
    if not isinstance(response, dict) or response.get("object") != "chat.completion":
        raise RuntimeError("seeded response object differs")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise RuntimeError("seeded response choices differ")
    message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise RuntimeError("seeded assistant message differs")
    content = message.get("content")
    usage = response.get("usage")
    if not isinstance(content, str) or not content or not isinstance(usage, dict):
        raise RuntimeError("seeded response content or usage differs")
    if choices[0].get("finish_reason") not in {"stop", "length"}:
        raise RuntimeError("seeded finish reason differs")
    return content, usage


def build_server_command(
    binary: Path,
    gguf: Path,
    derived_lock: Path,
    device_index: int,
    target: str,
    port: int,
) -> list[str]:
    """Construct the public GGUF server invocation used by the lifecycle probe."""
    return [
        str(binary),
        "--gguf", str(gguf),
        "--derived-lock", str(derived_lock),
        "--device-index", str(device_index),
        "--target", target,
        "--listen", f"127.0.0.1:{port}",
        "--model", "qwen3.5-4b",
    ]


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


def optional_amd_smi(command: str, gpu: int | None = None) -> object:
    """Capture provider-blocked telemetry as unavailable, never as zero."""
    try:
        return amd_smi(command, gpu)
    except subprocess.CalledProcessError as exc:
        return {
            "state": "unavailable",
            "command": command,
            "returncode": exc.returncode,
            "stderr": exc.stderr.strip(),
        }


def metric_observation(target: str, gpu: int) -> object:
    if target == "gfx942":
        return optional_amd_smi("metric", gpu)
    return amd_smi("metric", gpu)


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
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--derived-lock", type=Path, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--amd-smi-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--target", choices=("gfx1030", "gfx1201", "gfx942"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--client-python", type=Path, required=True)
    args = parser.parse_args()
    pre_static = amd_smi("static", args.amd_smi_index)
    pre_metric = metric_observation(args.target, args.amd_smi_index)
    pre_processes = process_count(amd_smi("process"), args.amd_smi_index)
    env = dict(os.environ)
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        env.pop(name, None)
    env["ROCR_VISIBLE_DEVICES"] = args.gpu_uuid
    env["LD_LIBRARY_PATH"] = "/opt/rocm/lib:/opt/rocm/lib64" + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    command = build_server_command(
        args.binary,
        args.gguf,
        args.derived_lock,
        args.device_index,
        args.target,
        args.port,
    )
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
        reasoning_body, reasoning_http_ns = post(base_url, reasoning_payload(17))
        reasoning_response = json.loads(reasoning_body)
        reasoning, reasoning_content = validate_reasoning_response(reasoning_response)
        reasoning_stream_body, reasoning_stream_http_ns = post(
            base_url, reasoning_payload(17, stream=True), stream=True
        )
        reasoning_chunks = parse_sse(reasoning_stream_body)
        reasoning_stream, reasoning_stream_content = parse_reasoning_sse(reasoning_chunks)
        if (reasoning_stream, reasoning_stream_content) != (reasoning, reasoning_content):
            raise RuntimeError("reasoning stream/non-stream split differs")
        if reasoning_chunks[-1].get("usage") != reasoning_response.get("usage"):
            raise RuntimeError("reasoning stream/non-stream usage differs")
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
        sampling_seed = 1902
        seeded_body, seeded_http_ns = post(
            base_url, seeded_sampling_payload(17, seed=sampling_seed)
        )
        seeded_response = json.loads(seeded_body)
        seeded_text, seeded_usage = validate_seeded_response(seeded_response)
        seeded_replay_body, seeded_replay_http_ns = post(
            base_url, seeded_sampling_payload(17, seed=sampling_seed)
        )
        seeded_replay_response = json.loads(seeded_replay_body)
        seeded_replay_text, seeded_replay_usage = validate_seeded_response(seeded_replay_response)
        if (seeded_replay_text, seeded_replay_usage) != (seeded_text, seeded_usage):
            raise RuntimeError("explicit seeded sampling did not replay exactly")
        disconnect_after_content("127.0.0.1", args.port)
        recovery_body, recovery_ns = post(base_url, request_payload(1))
        if json.loads(recovery_body)["object"] != "chat.completion":
            raise RuntimeError("post-disconnect recovery failed")

        first_elapsed: dict[str, int] = {}
        second_elapsed: dict[str, int] = {}
        queue_errors: list[BaseException] = []
        barrier = threading.Barrier(3)
        def queued(name: str, destination: dict[str, int]) -> None:
            try:
                barrier.wait()
                _, elapsed = post(base_url, request_payload(1))
                destination[name] = elapsed
            except BaseException as error:
                queue_errors.append(error)
        one = threading.Thread(target=queued, args=("one", first_elapsed))
        two = threading.Thread(target=queued, args=("two", second_elapsed))
        one.start(); two.start(); barrier.wait(); one.join(); two.join()
        if queue_errors or set(first_elapsed) != {"one"} or set(second_elapsed) != {"two"}:
            raise RuntimeError("two-concurrent request probe failed")
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
    post_metric = metric_observation(args.target, args.amd_smi_index)
    post_processes = process_count(amd_smi("process"), args.amd_smi_index)
    if post_processes != pre_processes:
        raise RuntimeError("GPU process count did not return to pre-run value")
    # The first two completed audits are the raw non-stream/SSE probes; the
    # fifth is the fixed render/tokenize baseline (reasoning probes occupy
    # slots three and four).  Validate its prompt identity before accounting.
    if len(completed) < 5:
        raise RuntimeError("service completed-audit sequence is incomplete")
    non_stream_backend_ns = completed[0]["elapsed_ns"]
    stream_backend_ns = completed[1]["elapsed_ns"]
    phase5_audit = completed[4]
    if phase5_audit["prompt_tokens"] != phase5_response["usage"]["prompt_tokens"]:
        raise RuntimeError("Phase 5 render audit identity differs")
    phase5_backend_ns = phase5_audit["elapsed_ns"]
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
        "reasoning": {
            "non_stream_reasoning_chars": len(reasoning),
            "non_stream_content_chars": len(reasoning_content),
            "stream_reasoning_chars": len(reasoning_stream),
            "stream_content_chars": len(reasoning_stream_content),
            "sse_chunks": len(reasoning_chunks),
        },
        "seeded_sampling": {
            "seed": sampling_seed,
            "temperature": 0.8,
            "top_p": 0.9,
            "replays": 2,
            "completion_chars": len(seeded_text),
        },
        "boundary_capacities": boundary_capacities,
        "disconnect": disconnected,
        "overhead": {
            "non_stream_json_residual_ns": max(0, non_stream_ns - non_stream_backend_ns),
            "stream_sse_residual_ns": max(0, stream_ns - stream_backend_ns),
            "reasoning_non_stream_http_ns": reasoning_http_ns,
            "reasoning_stream_http_ns": reasoning_stream_http_ns,
            "seeded_sampling_http_ns": seeded_http_ns + seeded_replay_http_ns,
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
