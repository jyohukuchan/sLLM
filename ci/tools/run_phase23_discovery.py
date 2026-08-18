#!/usr/bin/env python3
"""Bounded Phase 23 full-model measurement and summary helper.

Raw reports stay under a caller-owned local artifact directory.  The checked-in
Phase 23 summary contains only bounded distributions and immutable digests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, ROOT, canonical_bytes  # noqa: E402


SCHEMA_VERSION = "phase23-discovery-run-v1"
MODEL = ROOT / "docs/models/locks/qwen3.5-4b-bf16.json"
GGUF = Path("/home/homelab1/.cache/sllm/derived/phase20-audit-qwen35-bf16.gguf")
DERIVED_LOCK = Path(
    "/home/homelab1/.cache/sllm/derived/phase20-audit-qwen35-bf16.lock.json"
)
LLAMA_COMPARISON = (
    ROOT
    / ".local-artifacts/phase5/phase5-final-f1fd321a/llama-aggregate/comparison.json"
)
TARGETS = {
    "gfx1030": {
        "uuid": "GPU-76a08c022586fed6",
        "bdf": "0000:03:00.0",
        "product": "AMD Radeon Pro V620",
    },
    "gfx1201": {
        "uuid": "GPU-a8e9ddefa2d60f55",
        "bdf": "0000:07:00.0",
        "product": "AMD Radeon AI PRO R9700",
    },
}
VISIBILITY = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
MAX_OUTPUT = 64 * 1024 * 1024


def fail(message: str) -> None:
    raise ContractError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        fail(f"required regular file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_OUTPUT:
        fail(f"JSON input is missing, unsafe, or too large: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        fail(f"cannot read JSON {path}: {error}")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def distribution(values: list[int | float]) -> dict[str, int | float]:
    if not values:
        fail("cannot summarize an empty distribution")
    ordered = sorted(values)
    middle = statistics.median(ordered)
    deviations = [abs(value - middle) for value in ordered]
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": middle,
        "max": ordered[-1],
        "mad": statistics.median(deviations),
    }


def health_snapshot() -> bytes:
    command = [
        "rocm-smi",
        "--showproductname",
        "--showuniqueid",
        "--showuse",
        "--showmemuse",
        "--showpids",
        "--showtemp",
        "--showpower",
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0 or completed.stderr:
        fail("rocm-smi health snapshot failed")
    return completed.stdout


def verify_report(report: Any, target: str) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("state") != "PASS":
        fail("sLLM benchmark did not report PASS")
    if report.get("benchmark_schema_version") != "engine-performance-render-v1":
        fail("sLLM benchmark schema identity changed")
    identities = report.get("identities")
    audit = report.get("audit")
    measured = report.get("measured")
    cleanup = report.get("session_cleanup")
    if not isinstance(identities, dict) or identities.get("target") != target:
        fail("sLLM benchmark target identity changed")
    if not isinstance(audit, dict) or audit.get("selected_backend") != "hip":
        fail("sLLM benchmark did not select HIP")
    if audit.get("fallback_used") is not False or audit.get("all_dispatches_hip") is not True:
        fail("sLLM benchmark used fallback or non-HIP dispatch")
    if not isinstance(cleanup, dict) or cleanup != {"retryable_cleanup": 0, "durable_quarantine": 0}:
        fail("sLLM benchmark session cleanup was not terminal-zero")
    if not isinstance(measured, dict) or measured.get("count") != 10:
        fail("sLLM benchmark did not return ten measured requests")
    samples = measured.get("samples")
    if not isinstance(samples, list) or len(samples) != 10:
        fail("sLLM benchmark measured sample array is incomplete")
    for sample in samples:
        if not isinstance(sample, dict):
            fail("sLLM benchmark sample is not an object")
        sample_audit = sample.get("audit")
        if not isinstance(sample_audit, dict) or sample_audit.get("fallback_used") is not False:
            fail("sLLM benchmark sample audit is incomplete")
        if sample.get("cleanup", {}).get("allocator_cleanup_validated") is not True:
            fail("sLLM benchmark sample cleanup was not validated")
    return measured


def run_sllm(
    binary: Path,
    target: str,
    output_dir: Path,
    max_new_tokens: int,
    message_text: str,
) -> dict[str, Any]:
    if target not in TARGETS:
        fail(f"unsupported Phase 23 target: {target}")
    if binary.is_symlink() or not os.access(binary, os.X_OK):
        fail(f"sLLM binary is not an executable regular path: {binary}")
    if not message_text or len(message_text.encode("utf-8")) > 1024 * 1024:
        fail("Phase 23 message must contain 1..1048576 UTF-8 bytes")
    for path in (GGUF, DERIVED_LOCK, MODEL):
        sha256_file(path)
    message_digest = sha256_bytes(message_text.encode("utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    before = health_snapshot()
    environment = dict(os.environ)
    for name in VISIBILITY:
        environment.pop(name, None)
    environment["ROCR_VISIBLE_DEVICES"] = TARGETS[target]["uuid"]
    environment["LD_LIBRARY_PATH"] = "/opt/rocm/lib"
    command = [
        str(binary.resolve()),
        "benchmark",
        "--gguf",
        str(GGUF),
        "--derived-lock",
        str(DERIVED_LOCK),
        "--lane",
        "render-tokenize",
        "--row-id",
        f"phase23-qwen35-4b-{target}-{message_digest[:12]}",
        "--model-size",
        "4B",
        "--case-id",
        f"phase23-{message_digest[:12]}",
        "--message",
        "user:" + message_text,
        "--thinking",
        "disabled",
        "--max-new-tokens",
        str(max_new_tokens),
        "--device-index",
        "0",
        "--target",
        target,
        "--greedy",
        "--warmups",
        "3",
        "--measured",
        "10",
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        timeout=1800,
    )
    after = health_snapshot()
    if completed.returncode != 0:
        fail(f"sLLM benchmark failed with status {completed.returncode}: {completed.stderr[-4000:]!r}")
    if completed.stderr:
        fail("sLLM benchmark wrote unexpected stderr")
    if len(completed.stdout) > MAX_OUTPUT:
        fail("sLLM benchmark output exceeded the bounded limit")
    try:
        report = json.loads(completed.stdout)
    except (UnicodeError, ValueError) as error:
        fail(f"sLLM benchmark output is not valid JSON: {error}")
    measured = verify_report(report, target)
    samples = measured["samples"]
    derived = [sample["derived"] for sample in samples]
    events = [sample["events"] for sample in samples]
    summary = {
        "ttft_ns": distribution([item["ttft_ns"] for item in derived]),
        "prefill_ns": distribution([item["prefill_ns"] for item in derived]),
        "e2e_ns": distribution([item["e2e_ns"] for item in derived]),
        "decode_tokens_per_second": distribution(
            [item["decode_tokens_per_second"] for item in derived if item["decode_tokens_per_second"] is not None]
        ),
        "request_to_prefill_submit_ns": distribution(
            [item["prefill_submit_ns"] - item["request_start_ns"] for item in events]
        ),
        "prefill_complete_to_first_token_ns": distribution(
            [item["first_token_ns"] - item["prefill_complete_ns"] for item in events]
        ),
        "stop_to_cleanup_ns": distribution([item["cleanup_ns"] - item["stop_ns"] for item in events]),
    }
    raw_sha = sha256_bytes(completed.stdout)
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "state": "PASS",
        "target": target,
        "gpu": TARGETS[target],
        "binary": {"path": str(binary.resolve()), "sha256": sha256_file(binary)},
        "model": {
            "gguf_sha256": sha256_file(GGUF),
            "derived_lock_sha256": sha256_file(DERIVED_LOCK),
            "source_lock_sha256": sha256_file(MODEL),
        },
        "protocol": {
            "message_sha256": message_digest,
            "thinking": "disabled",
            "warmups": 3,
            "measured": 10,
            "max_new_tokens": max_new_tokens,
            "greedy": True,
            "comparison_class": "E1-system-equivalent",
        },
        "summary": summary,
        "audit": {
            "input_tokens": report["row"]["input_token_count"],
            "submission_count": report["audit"]["submission_count"],
            "kernel_dispatch_count": report["audit"]["kernel_dispatch_count"],
            "fallback_used": False,
            "all_dispatches_hip": True,
            "cleanup_terminal_zero": True,
        },
        "artifacts": {
            "raw_result_sha256": raw_sha,
            "health_before_sha256": sha256_bytes(before),
            "health_after_sha256": sha256_bytes(after),
        },
    }
    raw_path = output_dir / "raw-result.json"
    evidence_path = output_dir / "evidence.json"
    atomic_write(raw_path, completed.stdout)
    atomic_write(output_dir / "health-before.txt", before)
    atomic_write(output_dir / "health-after.txt", after)
    atomic_write(evidence_path, canonical_bytes(evidence) + b"\n")
    atomic_write(output_dir / "evidence.json.sha256", (sha256_file(evidence_path) + "\n").encode())
    return evidence


def post_chat(base_url: str, *, stream: bool) -> tuple[Any, int]:
    payload = {
        "model": "qwen3.5-4b",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
        "max_completion_tokens": 17,
        "stream": stream,
    }
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=canonical_bytes(payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic_ns()
    with urllib.request.urlopen(request, timeout=300) as response:
        expected = "text/event-stream" if stream else "application/json"
        if response.status != 200 or response.headers.get_content_type() != expected:
            fail("Phase 23 service response status or content type changed")
        body = response.read()
    elapsed = time.monotonic_ns() - started
    if len(body) > MAX_OUTPUT:
        fail("Phase 23 service response exceeded the bounded limit")
    if not stream:
        result = json.loads(body)
        if result.get("object") != "chat.completion":
            fail("Phase 23 non-stream response object changed")
        return result, elapsed
    chunks: list[dict[str, Any]] = []
    terminal = 0
    for block in body.decode("utf-8").split("\n\n"):
        if not block:
            continue
        data = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if len(data) != 1:
            fail("Phase 23 SSE event is not one data line")
        if data[0] == "[DONE]":
            terminal += 1
        else:
            chunks.append(json.loads(data[0]))
    if terminal != 1 or not chunks:
        fail("Phase 23 SSE response is incomplete")
    return chunks, elapsed


def wait_ready(process: subprocess.Popen[str], timeout: float) -> tuple[dict[str, Any], int]:
    started = time.monotonic_ns()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = process.stdout.readline() if process.stdout is not None else ""
        if line:
            value = json.loads(line)
            if value.get("event") == "ready":
                return value, time.monotonic_ns() - started
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr is not None else ""
            fail(f"Phase 23 service exited before ready: {stderr[-4000:]}")
    fail("Phase 23 service did not become ready")


def verify_service_audit(audit: Any, target: str) -> None:
    if not isinstance(audit, dict) or audit.get("outcome") != "completed":
        fail("Phase 23 service request did not complete")
    if audit.get("target") != target or audit.get("selected_backend") != "hip":
        fail("Phase 23 service request selected the wrong target or backend")
    if audit.get("fallback_used") is not False or audit.get("all_dispatches_hip") is not True:
        fail("Phase 23 service request used fallback or non-HIP dispatch")
    if audit.get("cleanup_request_state_bytes") != 0 or audit.get("cleanup_workspace_bytes") != 0:
        fail("Phase 23 service request cleanup was not terminal-zero")


def run_api(binary: Path, target: str, output_dir: Path, port: int) -> dict[str, Any]:
    if target not in TARGETS:
        fail(f"unsupported Phase 23 target: {target}")
    if binary.is_symlink() or not os.access(binary, os.X_OK):
        fail(f"server binary is not an executable regular path: {binary}")
    output_dir.mkdir(parents=True, exist_ok=True)
    before = health_snapshot()
    environment = dict(os.environ)
    for name in VISIBILITY:
        environment.pop(name, None)
    environment["ROCR_VISIBLE_DEVICES"] = TARGETS[target]["uuid"]
    environment["LD_LIBRARY_PATH"] = "/opt/rocm/lib"
    command = [
        str(binary.resolve()),
        "--gguf", str(GGUF),
        "--derived-lock", str(DERIVED_LOCK),
        "--device-index", "0",
        "--target", target,
        "--listen", f"127.0.0.1:{port}",
        "--model", "qwen3.5-4b",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    remaining_stdout = ""
    stderr = ""
    non_stream_http: list[int] = []
    stream_http: list[int] = []
    try:
        ready, startup_ns = wait_ready(process, 180)
        if ready.get("target") != target:
            fail("Phase 23 service ready target changed")
        base_url = f"http://127.0.0.1:{port}"
        for index in range(13):
            result, elapsed = post_chat(base_url, stream=False)
            if result.get("usage", {}).get("prompt_tokens") != 13:
                fail("Phase 23 service prompt identity changed")
            if index >= 3:
                non_stream_http.append(elapsed)
        for index in range(13):
            _, elapsed = post_chat(base_url, stream=True)
            if index >= 3:
                stream_http.append(elapsed)
        pair: dict[str, int] = {}
        barrier = threading.Barrier(3)

        def queued(name: str) -> None:
            barrier.wait()
            _, elapsed = post_chat(base_url, stream=False)
            pair[name] = elapsed

        one = threading.Thread(target=queued, args=("one",))
        two = threading.Thread(target=queued, args=("two",))
        one.start()
        two.start()
        barrier.wait()
        one.join()
        two.join()
        concurrent_http = sorted(pair.values())
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        try:
            remaining_stdout, stderr = process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            fail("Phase 23 service did not terminate after SIGINT")
    if process.returncode != 0 or stderr:
        fail(f"Phase 23 service shutdown failed: status={process.returncode} stderr={stderr[-4000:]}")
    events = [json.loads(line) for line in remaining_stdout.splitlines() if line.strip()]
    shutdown = next((event.get("report") for event in events if event.get("event") == "shutdown_audit"), None)
    if not isinstance(shutdown, dict) or shutdown.get("final_current_bytes") != 0:
        fail("Phase 23 service shutdown audit is absent or nonzero")
    audits = shutdown.get("requests")
    if not isinstance(audits, list) or len(audits) != 28:
        fail("Phase 23 service request audit count changed")
    for audit in audits:
        verify_service_audit(audit, target)
    non_stream_backend = [audit["elapsed_ns"] for audit in audits[3:13]]
    stream_backend = [audit["elapsed_ns"] for audit in audits[16:26]]
    after = health_snapshot()
    evidence = {
        "schema_version": "phase23-api-run-v1",
        "state": "PASS",
        "target": target,
        "gpu": TARGETS[target],
        "binary": {"path": str(binary.resolve()), "sha256": sha256_file(binary)},
        "model": {
            "gguf_sha256": sha256_file(GGUF),
            "derived_lock_sha256": sha256_file(DERIVED_LOCK),
            "source_lock_sha256": sha256_file(MODEL),
        },
        "protocol": {
            "input_tokens": 13,
            "requested_output_tokens": 17,
            "warmups": 3,
            "measured": 10,
            "greedy": True,
            "comparison_class": "E1-system-equivalent",
        },
        "summary": {
            "cold_start_to_ready_ns": startup_ns,
            "non_stream_http_ns": distribution(non_stream_http),
            "non_stream_backend_ns": distribution(non_stream_backend),
            "non_stream_transport_residual_ns": distribution(
                [max(0, outer - inner) for outer, inner in zip(non_stream_http, non_stream_backend)]
            ),
            "sse_http_ns": distribution(stream_http),
            "sse_backend_ns": distribution(stream_backend),
            "sse_transport_residual_ns": distribution(
                [max(0, outer - inner) for outer, inner in zip(stream_http, stream_backend)]
            ),
            "concurrency_2_http_ns": distribution(concurrent_http),
            "concurrency_2_skew_ns": concurrent_http[1] - concurrent_http[0],
        },
        "audit": {
            "request_count": len(audits),
            "fallback_used": False,
            "all_dispatches_hip": True,
            "cleanup_terminal_zero": True,
        },
        "artifacts": {
            "health_before_sha256": sha256_bytes(before),
            "health_after_sha256": sha256_bytes(after),
            "shutdown_audit_sha256": sha256_bytes(canonical_bytes(shutdown)),
        },
    }
    atomic_write(output_dir / "shutdown-audit.json", canonical_bytes(shutdown) + b"\n")
    atomic_write(output_dir / "health-before.txt", before)
    atomic_write(output_dir / "health-after.txt", after)
    evidence_path = output_dir / "evidence.json"
    atomic_write(evidence_path, canonical_bytes(evidence) + b"\n")
    atomic_write(output_dir / "evidence.json.sha256", (sha256_file(evidence_path) + "\n").encode())
    return evidence


def run_cold(binary: Path, target: str, output_dir: Path, port: int) -> dict[str, Any]:
    if target not in TARGETS:
        fail(f"unsupported Phase 23 target: {target}")
    if binary.is_symlink() or not os.access(binary, os.X_OK):
        fail(f"server binary is not an executable regular path: {binary}")
    output_dir.mkdir(parents=True, exist_ok=True)
    before = health_snapshot()
    environment = dict(os.environ)
    for name in VISIBILITY:
        environment.pop(name, None)
    environment["ROCR_VISIBLE_DEVICES"] = TARGETS[target]["uuid"]
    environment["LD_LIBRARY_PATH"] = "/opt/rocm/lib"
    durations: list[int] = []
    shutdown_digests: list[str] = []
    for index in range(5):
        command = [
            str(binary.resolve()),
            "--gguf", str(GGUF),
            "--derived-lock", str(DERIVED_LOCK),
            "--device-index", "0",
            "--target", target,
            "--listen", f"127.0.0.1:{port + index}",
            "--model", "qwen3.5-4b",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        try:
            ready, elapsed = wait_ready(process, 180)
            if ready.get("target") != target:
                fail("Phase 23 cold-start ready target changed")
            durations.append(elapsed)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=180)
        if process.returncode != 0 or stderr:
            fail(f"Phase 23 cold-start shutdown failed: {stderr[-4000:]}")
        events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        shutdown = next(
            (event.get("report") for event in events if event.get("event") == "shutdown_audit"),
            None,
        )
        if not isinstance(shutdown, dict):
            fail("Phase 23 cold-start shutdown audit is absent")
        if shutdown.get("final_current_bytes") != 0 or shutdown.get("requests") != []:
            fail("Phase 23 cold-start shutdown audit is not request-free terminal-zero")
        shutdown_digests.append(sha256_bytes(canonical_bytes(shutdown)))
    after = health_snapshot()
    evidence = {
        "schema_version": "phase23-cold-start-v1",
        "state": "PASS",
        "target": target,
        "gpu": TARGETS[target],
        "binary": {"path": str(binary.resolve()), "sha256": sha256_file(binary)},
        "model": {
            "gguf_sha256": sha256_file(GGUF),
            "derived_lock_sha256": sha256_file(DERIVED_LOCK),
            "source_lock_sha256": sha256_file(MODEL),
        },
        "protocol": {
            "fresh_process_each_sample": True,
            "os_page_cache": "uncontrolled-warm",
            "measured": 5,
            "requests_per_process": 0,
        },
        "summary": {"process_start_to_model_ready_ns": distribution(durations)},
        "audit": {"fallback_used": False, "cleanup_terminal_zero": True},
        "artifacts": {
            "health_before_sha256": sha256_bytes(before),
            "health_after_sha256": sha256_bytes(after),
            "shutdown_audit_sha256": shutdown_digests,
        },
    }
    evidence_path = output_dir / "evidence.json"
    atomic_write(output_dir / "health-before.txt", before)
    atomic_write(output_dir / "health-after.txt", after)
    atomic_write(evidence_path, canonical_bytes(evidence) + b"\n")
    atomic_write(output_dir / "evidence.json.sha256", (sha256_file(evidence_path) + "\n").encode())
    return evidence


def profiler_category(name: str) -> str:
    if "matmul_bf16_fp32_tiled16" in name:
        return "prefill_matmul"
    if name.startswith("Cijk_"):
        return "hipblas_gemm"
    if "matmul_" in name:
        return "decode_or_recurrent_matvec"
    if "linear_attention" in name:
        return "linear_attention"
    if "attention" in name or "kv_state" in name:
        return "full_attention_and_kv"
    if "argmax" in name or "sampling" in name:
        return "sampling"
    if "rmsnorm" in name or "norm" in name:
        return "normalization"
    if "elementwise" in name or "embedding" in name:
        return "elementwise_and_embedding"
    if "rocclr_" in name:
        return "runtime_copy_fill"
    return "other"


def aggregate_profile(profile_dir: Path, target: str, output_dir: Path) -> dict[str, Any]:
    def one(pattern: str) -> Path:
        matches = sorted(profile_dir.glob(pattern))
        if len(matches) != 1:
            fail(f"Phase 23 profile expected one {pattern}, found {len(matches)}")
        return matches[0]

    kernel_path = one("*_kernel_stats.csv")
    hip_path = one("*_hip_api_stats.csv")
    copy_path = one("*_memory_copy_stats.csv")
    trace_path = one("*_kernel_trace.csv")
    with kernel_path.open(newline="", encoding="utf-8") as stream:
        kernels = list(csv.DictReader(stream))
    with hip_path.open(newline="", encoding="utf-8") as stream:
        hip_apis = list(csv.DictReader(stream))
    with copy_path.open(newline="", encoding="utf-8") as stream:
        copies = list(csv.DictReader(stream))
    total_ns = sum(int(row["TotalDurationNs"]) for row in kernels)
    if total_ns <= 0:
        fail("Phase 23 profiler kernel duration is empty")
    categories: dict[str, dict[str, int]] = {}
    for row in kernels:
        name = profiler_category(row["Name"])
        value = categories.setdefault(name, {"calls": 0, "total_duration_ns": 0})
        value["calls"] += int(row["Calls"])
        value["total_duration_ns"] += int(row["TotalDurationNs"])
    category_rows = [
        {
            "category": name,
            **value,
            "device_time_share": value["total_duration_ns"] / total_ns,
        }
        for name, value in categories.items()
    ]
    category_rows.sort(key=lambda row: row["total_duration_ns"], reverse=True)
    evidence = {
        "schema_version": "phase23-profiler-aggregate-v1",
        "state": "PASS",
        "target": target,
        "observer_effect": "rocprofv3-runtime-trace; wall time is diagnostic-only",
        "kernel": {
            "calls": sum(int(row["Calls"]) for row in kernels),
            "total_duration_ns": total_ns,
            "categories": category_rows,
        },
        "hip_api_top": [
            {
                "name": row["Name"],
                "calls": int(row["Calls"]),
                "total_duration_ns": int(row["TotalDurationNs"]),
            }
            for row in hip_apis[:12]
        ],
        "memory_copy": [
            {
                "name": row["Name"],
                "calls": int(row["Calls"]),
                "total_duration_ns": int(row["TotalDurationNs"]),
            }
            for row in copies
        ],
        "artifacts": {
            "kernel_stats_sha256": sha256_file(kernel_path),
            "hip_api_stats_sha256": sha256_file(hip_path),
            "memory_copy_stats_sha256": sha256_file(copy_path),
            "kernel_trace_sha256": sha256_file(trace_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "profile-aggregate.json"
    atomic_write(evidence_path, canonical_bytes(evidence) + b"\n")
    atomic_write(
        output_dir / "profile-aggregate.json.sha256",
        (sha256_file(evidence_path) + "\n").encode(),
    )
    return evidence


def contract_only() -> dict[str, Any]:
    for path in (MODEL, GGUF, DERIVED_LOCK, LLAMA_COMPARISON):
        sha256_file(path)
    llama = read_json(LLAMA_COMPARISON)
    if llama.get("state") != "PASS" or len(llama.get("rows", [])) != 14:
        fail("immutable Phase 5 llama.cpp comparison is incomplete")
    targets = {row.get("target") for row in llama["rows"]}
    if targets != set(TARGETS):
        fail("immutable Phase 5 llama.cpp comparison does not cover both targets")
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "PASS",
        "llama_comparison_sha256": sha256_file(LLAMA_COMPARISON),
        "model_lock_sha256": sha256_file(MODEL),
        "gguf_sha256": sha256_file(GGUF),
        "derived_lock_sha256": sha256_file(DERIVED_LOCK),
        "targets": list(TARGETS),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    modes = result.add_mutually_exclusive_group(required=True)
    modes.add_argument("--contract-only", action="store_true")
    modes.add_argument("--run-sllm", action="store_true")
    modes.add_argument("--run-api", action="store_true")
    modes.add_argument("--run-cold", action="store_true")
    modes.add_argument("--aggregate-profile", action="store_true")
    result.add_argument("--binary", type=Path)
    result.add_argument("--target", choices=TARGETS)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--max-new-tokens", type=int, default=17)
    result.add_argument("--message-text", default="Hello")
    result.add_argument("--port", type=int)
    result.add_argument("--profile-dir", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.contract_only:
            result = contract_only()
        elif args.aggregate_profile:
            if args.profile_dir is None or args.target is None or args.output_dir is None:
                fail("--aggregate-profile requires --profile-dir, --target, and --output-dir")
            result = aggregate_profile(args.profile_dir, args.target, args.output_dir)
        else:
            if args.binary is None or args.target is None or args.output_dir is None:
                fail("execution requires --binary, --target, and --output-dir")
            if args.run_api or args.run_cold:
                if args.port is None:
                    fail("service execution requires --port")
                if args.run_cold:
                    result = run_cold(args.binary, args.target, args.output_dir, args.port)
                else:
                    result = run_api(args.binary, args.target, args.output_dir, args.port)
            else:
                if not 2 <= args.max_new_tokens <= 128:
                    fail("--max-new-tokens must be in [2,128]")
                result = run_sllm(
                    args.binary,
                    args.target,
                    args.output_dir,
                    args.max_new_tokens,
                    args.message_text,
                )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except (ContractError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"phase23-discovery: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
