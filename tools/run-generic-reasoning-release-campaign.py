#!/usr/bin/env python3
"""Collect hash-only generic reasoning release cases from an active v2 service.

The campaign is intentionally fail-closed: it requires a validated v2
served-model manifest and an exclusive gfx1201 worker before sending a request.
It sends one streamed case for each release mode, correlates the response with
the gateway lifecycle observer, samples the target GPU and service RSS, and
atomically publishes bounded cases/lifecycle artifacts. Response and prompt
contents are used only in memory for quality checks and are never persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import select
import socket
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SERVED_MODEL_VALIDATOR_PATH = ROOT / "tools/validate-served-model.py"
DEFAULT_FIXTURES = ROOT / "tests/fixtures/generic-reasoning-release-v0.1/prompts.json"
DEFAULT_OBSERVER = Path("/run/ullm/lifecycle-observer.sock")
DEFAULT_ENDPOINT = "http://172.20.0.1:8000/v1/chat/completions"
DEFAULT_HTTP_IMAGE = "sha256:5dce198cca467ce79994ed65e01d03882238f9efdd16a8c6f4bc55151c8a4a54"
TARGET_GFX = "gfx1201"
TARGET_GPU_INDEX = "1"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_FIXTURE_BYTES = 1 * 1024 * 1024
MAX_EVENT_BYTES = 65_536
IMAGE_RE = re.compile(r"(?:[A-Za-z0-9][A-Za-z0-9._/:+-]*@)?sha256:[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
LIFECYCLE_SCHEMA = "ullm.gateway.lifecycle.v1"
CAMPAIGN_SCHEMA = "ullm.generic_reasoning_release_campaign.v1"
MODES = ("disabled", "budget-32", "budget-128", "budget-256", "unbounded")
MODE_BUDGETS: dict[str, int | None] = {
    "disabled": 0,
    "budget-32": 32,
    "budget-128": 128,
    "budget-256": 256,
    "unbounded": None,
}
_SERVED_MODEL_MODULE_NAME = "_ullm_generic_campaign_served_model_validator"


class CampaignError(RuntimeError):
    """Raised when a release campaign cannot safely produce evidence."""


@dataclass(frozen=True, slots=True)
class Fixture:
    identifier: str
    prompt: str
    expected_answer: str


@dataclass(frozen=True, slots=True)
class ResourceSample:
    rss_bytes: int
    vram_bytes: int
    temperature_c: float
    power_w: float


@dataclass(frozen=True, slots=True)
class StreamResult:
    status: int
    completion_id: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    usage_timings: dict[str, Any]
    answer_text: str
    reasoning_text: str
    sse_chunk_count: int
    first_reasoning_ms: float | None
    first_answer_ms: float | None
    latency_ms: float


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular(path: Path, label: str, maximum: int) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise CampaignError(f"{label} is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CampaignError(f"{label} is not a regular non-symlink file")
    if metadata.st_size <= 0 or metadata.st_size > maximum:
        raise CampaignError(f"{label} exceeds its size bound")
    try:
        value = path.read_bytes()
    except OSError as error:
        raise CampaignError(f"{label} cannot be read") from error
    if len(value) != metadata.st_size or len(value) > maximum:
        raise CampaignError(f"{label} changed while being read")
    return value


def _strict_json(raw: bytes, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise CampaignError(f"{label} contains duplicate fields")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"{label} is not strict JSON") from error


def _load_fixtures(path: Path) -> dict[str, Fixture]:
    value = _strict_json(_read_regular(path, "fixture suite", MAX_FIXTURE_BYTES), "fixture suite")
    if not isinstance(value, dict) or set(value) != {"fixtures"}:
        raise CampaignError("fixture suite fields differ")
    raw_fixtures = value["fixtures"]
    if not isinstance(raw_fixtures, list) or not raw_fixtures:
        raise CampaignError("fixture suite is empty")
    result: dict[str, Fixture] = {}
    for raw in raw_fixtures:
        if not isinstance(raw, dict) or set(raw) != {"id", "prompt", "expected_answer"}:
            raise CampaignError("fixture fields differ")
        identifier = raw["id"]
        prompt = raw["prompt"]
        answer = raw["expected_answer"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or len(identifier.encode("utf-8")) > 512
            or identifier in result
            or not isinstance(prompt, str)
            or not prompt
            or len(prompt.encode("utf-8")) > 65_536
            or not isinstance(answer, str)
            or not answer
            or len(answer.encode("utf-8")) > 65_536
        ):
            raise CampaignError("fixture text is invalid")
        result[identifier] = Fixture(identifier, prompt, answer)
    return result


def _load_served_model_validator() -> ModuleType:
    existing = sys.modules.get(_SERVED_MODEL_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        _SERVED_MODEL_MODULE_NAME, SERVED_MODEL_VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise CampaignError("served-model validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SERVED_MODEL_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_SERVED_MODEL_MODULE_NAME, None)
        raise
    return module


def _validate_manifest(path: Path) -> dict[str, Any]:
    document = _strict_json(_read_regular(path, "served-model manifest", 1_048_576), "manifest")
    if not isinstance(document, dict) or document.get("schema_version") != "ullm.served_model.v2":
        raise CampaignError("served-model manifest is not v2")
    worker = document.get("worker")
    public = document.get("public")
    if not isinstance(worker, dict) or worker.get("protocol") != "ullm.worker.v2":
        raise CampaignError("served-model manifest worker protocol is not v2")
    if not isinstance(public, dict) or not isinstance(public.get("id"), str):
        raise CampaignError("served-model manifest public identity is missing")
    if "reasoning" not in document:
        raise CampaignError("served-model manifest reasoning dialect is missing")
    try:
        summary = _load_served_model_validator().validation_summary(path)
    except Exception as error:
        raise CampaignError("served-model manifest failed validation") from error
    if summary.get("worker", {}).get("protocol") != "ullm.worker.v2":
        raise CampaignError("served-model validator did not return v2")
    if summary.get("worker", {}).get("device") != TARGET_GFX:
        raise CampaignError("served-model manifest does not target gfx1201")
    return summary


def _read_gpu_processes(rocm_smi: str = "rocm-smi") -> dict[str, Any]:
    try:
        result = subprocess.run(
            [rocm_smi, "--showpids", "--json"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CampaignError("ROCm GPU preflight failed") from error
    if result.returncode != 0:
        raise CampaignError("ROCm GPU preflight returned nonzero")
    value = _strict_json(result.stdout.encode("utf-8"), "ROCm GPU preflight")
    processes = value.get("system") if isinstance(value, dict) else None
    if not isinstance(processes, dict):
        raise CampaignError("ROCm GPU preflight process data is missing")
    positive: list[dict[str, Any]] = []
    for pid, description in processes.items():
        if not isinstance(pid, str) or not pid.startswith("PID") or not isinstance(description, str):
            raise CampaignError("ROCm GPU process identity is malformed")
        fields = [field.strip() for field in description.split(",")]
        if len(fields) < 3:
            raise CampaignError("ROCm GPU process description is incomplete")
        try:
            vram = int(fields[2])
        except ValueError as error:
            raise CampaignError("ROCm GPU process VRAM is invalid") from error
        visible = {item.strip() for item in fields[1].split(",") if item.strip()}
        if TARGET_GPU_INDEX in visible and vram > 0:
            positive.append(
                {"pid": pid[3:], "process": fields[0], "gpu_index": TARGET_GPU_INDEX, "vram_bytes": vram}
            )
    if not positive:
        raise CampaignError("target R9700 has no resident v2 worker")
    if any(item["process"] == "llama-server" for item in positive):
        raise CampaignError("llama.cpp is resident; release campaign requires an exclusive R9700")
    if any(item["process"] != "ullm-aq4-worker" for item in positive):
        raise CampaignError("an unexpected process owns the target R9700")
    return {"tool": f"{rocm_smi} --showpids --json", "gpu_index": TARGET_GPU_INDEX, "positive_vram_processes": positive}


def _hash_process_executable(pid: str) -> str:
    link = Path(f"/proc/{pid}/exe")
    try:
        target = Path(os.readlink(link))
        metadata = target.stat()
    except OSError as error:
        raise CampaignError("target worker executable cannot be inspected") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise CampaignError("target worker executable is not a regular file")
    digest = hashlib.sha256()
    try:
        with target.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise CampaignError("target worker executable cannot be hashed") from error
    return digest.hexdigest()


def _bind_gpu_processes(preflight: dict[str, Any], expected_binary_sha256: str) -> None:
    for process in preflight["positive_vram_processes"]:
        observed = _hash_process_executable(process["pid"])
        if observed != expected_binary_sha256:
            raise CampaignError("resident GPU worker binary differs from the v2 manifest")
        process["binary_sha256"] = observed


def _service_pid(service: str, systemctl: str = "systemctl") -> int:
    try:
        result = subprocess.run(
            [systemctl, "show", service, "--property=MainPID", "--value"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CampaignError("service PID lookup failed") from error
    try:
        pid = int(result.stdout.strip())
    except ValueError as error:
        raise CampaignError("service MainPID is invalid") from error
    if result.returncode != 0 or pid <= 0:
        raise CampaignError("service is not running")
    return pid


def _rss_bytes(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
    except OSError as error:
        raise CampaignError("service RSS cannot be sampled") from error
    for line in raw.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) == 3 and fields[1].isdigit() and fields[2] == "kB":
                return int(fields[1]) * 1024
    raise CampaignError("service RSS field is missing")


def _resource_sample(service: str, rocm_smi: str, systemctl: str) -> ResourceSample:
    try:
        result = subprocess.run(
            [rocm_smi, "--showproductname", "--showmeminfo", "vram", "--showtemp", "--showpower", "--json"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CampaignError("resource sample failed") from error
    if result.returncode != 0:
        raise CampaignError("resource sample returned nonzero")
    value = _strict_json(result.stdout.encode("utf-8"), "resource sample")
    cards = value if isinstance(value, dict) else {}
    target: dict[str, Any] | None = None
    for card in cards.values():
        if isinstance(card, dict) and card.get("GFX Version") == TARGET_GFX:
            target = card
            break
    if target is None:
        raise CampaignError("gfx1201 resource sample is missing")
    try:
        vram = int(target["VRAM Total Used Memory (B)"])
        temperature = float(target["Temperature (Sensor edge) (C)"])
        power = float(target["Average Graphics Package Power (W)"])
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError("gfx1201 resource sample fields are invalid") from error
    return ResourceSample(
        rss_bytes=_rss_bytes(_service_pid(service, systemctl)),
        vram_bytes=vram,
        temperature_c=temperature,
        power_w=power,
    )


class LifecycleObserver:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.socket: socket.socket | None = None
        self._bound = False

    def open(self) -> None:
        if self.path.exists() or self.path.is_symlink():
            raise CampaignError("lifecycle observer socket already exists")
        try:
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self.socket.bind(os.fspath(self.path))
            os.chmod(self.path, 0o600)
            self._bound = True
        except OSError as error:
            self.close()
            raise CampaignError("lifecycle observer socket could not be opened") from error

    def wait_release(self, completion_id: str, timeout_seconds: float) -> dict[str, Any]:
        if self.socket is None:
            raise CampaignError("lifecycle observer is not open")
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CampaignError("matching request_released lifecycle event timed out")
            readable, _, _ = select.select([self.socket], [], [], min(remaining, 1.0))
            if not readable:
                continue
            try:
                raw = self.socket.recv(MAX_EVENT_BYTES)
            except OSError as error:
                raise CampaignError("lifecycle observer read failed") from error
            value = _strict_json(raw, "lifecycle event")
            if not isinstance(value, dict) or value.get("schema_version") != LIFECYCLE_SCHEMA:
                continue
            if value.get("event") == "request_released" and value.get("completion_id") == completion_id:
                return value

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self._bound:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
            self._bound = False


def _mode_fields(mode: str) -> dict[str, Any]:
    if mode == "disabled":
        return {"reasoning_effort": "none"}
    if mode == "unbounded":
        return {"thinking_budget_tokens": -1}
    budget = MODE_BUDGETS.get(mode)
    if budget is None:
        raise CampaignError("unknown release mode")
    return {"thinking_budget_tokens": budget}


def _request_body(model_id: str, mode: str, fixture: Fixture, stream: bool = True) -> bytes:
    value: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": fixture.prompt}],
        "max_completion_tokens": 512,
        "temperature": 0,
        "stream": stream,
    }
    if stream:
        value["stream_options"] = {"include_usage": True}
    value.update(_mode_fields(mode))
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _docker_command(
    *, docker: str, image: str, key_file: Path, endpoint: str, network: str
) -> list[str]:
    script = (
        "set -eu; key=$(cat /run/secrets/ullm-api-key); config=$(mktemp); "
        "trap 'rm -f \"$config\"' EXIT; umask 077; "
        "printf 'header = \"Authorization: Bearer %s\"\\n' \"$key\" > \"$config\"; "
        "exec curl --config \"$config\" --silent --show-error --no-buffer "
        "-H 'Content-Type: application/json' -H 'Accept: text/event-stream' "
        "-w '\\n__ULLM_HTTP_STATUS__%{http_code}\\n' --data-binary @- "
        + endpoint
    )
    return [
        docker,
        "run",
        "--rm",
        "-i",
        "--network",
        network,
        "-v",
        f"{key_file.resolve(strict=True)}:/run/secrets/ullm-api-key:ro",
        "--entrypoint",
        "sh",
        image,
        "-c",
        script,
    ]


def _stream_request(
    body: bytes,
    *,
    command: list[str],
    timeout_seconds: float,
) -> StreamResult:
    started_ns = time.monotonic_ns()
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as error:
        raise CampaignError("Docker HTTP probe could not start") from error
    assert process.stdin is not None and process.stdout is not None
    try:
        process.stdin.write(body)
        process.stdin.close()
        lines: list[bytes] = []
        total_bytes = 0
        first_reasoning_ns: int | None = None
        first_answer_ns: int | None = None
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        completion_id: str | None = None
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        timings: dict[str, Any] = {}
        chunks = 0
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CampaignError("Docker HTTP probe timed out")
            readable, _, _ = select.select([process.stdout], [], [], min(remaining, 1.0))
            if not readable:
                continue
            line = process.stdout.readline()
            if not line:
                break
            total_bytes += len(line)
            if total_bytes > MAX_RESPONSE_BYTES:
                raise CampaignError("HTTP response exceeds its size bound")
            lines.append(line)
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                continue
            value = _strict_json(payload, "SSE data")
            if not isinstance(value, dict):
                raise CampaignError("SSE data is not an object")
            chunks += 1
            if isinstance(value.get("id"), str):
                completion_id = value["id"]
            if isinstance(value.get("timings"), dict):
                timings = value["timings"]
            choices = value.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]
                if isinstance(choice, dict):
                    if isinstance(choice.get("finish_reason"), str):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta")
                    if isinstance(delta, dict):
                        reasoning = delta.get("reasoning_content")
                        content = delta.get("content")
                        if isinstance(reasoning, str) and reasoning:
                            if first_reasoning_ns is None:
                                first_reasoning_ns = time.monotonic_ns()
                            reasoning_parts.append(reasoning)
                        if isinstance(content, str) and content:
                            if first_answer_ns is None:
                                first_answer_ns = time.monotonic_ns()
                            answer_parts.append(content)
            if isinstance(value.get("usage"), dict):
                usage = value["usage"]
        return_code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
    except CampaignError:
        process.kill()
        process.wait(timeout=15.0)
        raise
    except (BrokenPipeError, OSError, subprocess.TimeoutExpired) as error:
        process.kill()
        process.wait(timeout=15.0)
        raise CampaignError("Docker HTTP probe failed") from error
    if return_code != 0:
        raise CampaignError("Docker HTTP probe returned nonzero")
    raw = b"".join(lines)
    marker = b"\n__ULLM_HTTP_STATUS__"
    marker_position = raw.rfind(marker)
    if marker_position < 0:
        raise CampaignError("Docker HTTP probe returned no HTTP status")
    try:
        status = int(raw[marker_position + len(marker) :].strip())
    except ValueError as error:
        raise CampaignError("Docker HTTP probe returned an invalid HTTP status") from error
    if completion_id is None or finish_reason is None or usage is None:
        raise CampaignError("SSE response is missing completion metadata")
    try:
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
        reasoning_tokens = int(usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0))
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError("SSE usage accounting is invalid") from error
    ended_ns = time.monotonic_ns()
    return StreamResult(
        status=status,
        completion_id=completion_id,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        usage_timings=timings,
        answer_text="".join(answer_parts),
        reasoning_text="".join(reasoning_parts),
        sse_chunk_count=chunks,
        first_reasoning_ms=(
            (first_reasoning_ns - started_ns) / 1_000_000 if first_reasoning_ns is not None else None
        ),
        first_answer_ms=(
            (first_answer_ns - started_ns) / 1_000_000 if first_answer_ns is not None else None
        ),
        latency_ms=(ended_ns - started_ns) / 1_000_000,
    )


def _timing(result: StreamResult, forced_end_tokens: int) -> dict[str, float | None]:
    prompt_rate = result.usage_timings.get("prompt_per_second")
    if not isinstance(prompt_rate, (int, float)) or prompt_rate <= 0:
        prompt_ms = result.usage_timings.get("prompt_ms")
        prompt_rate = (
            result.prompt_tokens / (float(prompt_ms) / 1000)
            if isinstance(prompt_ms, (int, float)) and prompt_ms > 0
            else None
        )
    answer_tokens = result.completion_tokens - result.reasoning_tokens - forced_end_tokens
    reasoning_rate: float | None = None
    if result.reasoning_tokens > 0 and result.first_reasoning_ms is not None and result.first_answer_ms is not None:
        duration = (result.first_answer_ms - result.first_reasoning_ms) / 1000
        if duration > 0:
            reasoning_rate = result.reasoning_tokens / duration
    answer_rate: float | None = None
    if result.first_answer_ms is not None:
        duration = (result.latency_ms - result.first_answer_ms) / 1000
        if duration > 0:
            answer_rate = answer_tokens / duration
    decode_rate = result.completion_tokens / (result.latency_ms / 1000) if result.latency_ms > 0 else None
    return {
        "prefill_tokens_per_second": float(prompt_rate) if prompt_rate is not None else None,
        "first_reasoning_token_ms": result.first_reasoning_ms,
        "first_answer_token_ms": result.first_answer_ms,
        "reasoning_decode_tokens_per_second": reasoning_rate,
        "answer_decode_tokens_per_second": answer_rate,
        "decode_tokens_per_second": decode_rate,
        "latency_ms": result.latency_ms,
    }


def _case_and_lifecycle(
    *, mode: str, fixture: Fixture, result: StreamResult, release: dict[str, Any], before: ResourceSample, after: ResourceSample
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if result.status != 200 or result.finish_reason not in {"stop", "length"}:
        raise CampaignError(f"release case {mode} HTTP or finish contract failed")
    if release.get("completion_id") != result.completion_id or release.get("outcome") != result.finish_reason:
        raise CampaignError(f"release case {mode} lifecycle identity differs")
    if release.get("prompt_tokens") != result.prompt_tokens or release.get("completion_tokens") != result.completion_tokens:
        raise CampaignError(f"release case {mode} lifecycle token count differs")
    if release.get("reset_complete") is not True:
        raise CampaignError(f"release case {mode} did not reset completely")
    forced = release.get("forced_end_tokens", 0) if mode != "disabled" else 0
    if not isinstance(forced, int) or forced < 0:
        raise CampaignError(f"release case {mode} forced-end accounting is invalid")
    reasoning = result.reasoning_tokens if mode != "disabled" else 0
    if mode != "disabled" and release.get("reasoning_tokens") != reasoning:
        raise CampaignError(f"release case {mode} reasoning accounting differs")
    answer_tokens = result.completion_tokens - reasoning - forced
    if answer_tokens < 1:
        raise CampaignError(f"release case {mode} has no answer tokens")
    budget = MODE_BUDGETS[mode]
    overshoot = max(0, reasoning - budget) if budget is not None and mode != "disabled" else 0
    case_id = f"generic-reasoning-{mode}"
    case = {
        "id": case_id,
        "mode": mode,
        "prompt_fixture_id": fixture.identifier,
        "prompt_sha256": _sha256(fixture.prompt.encode("utf-8")),
        "stream": True,
        "http_status": result.status,
        "sse_chunk_count": result.sse_chunk_count,
        "finish_reason": result.finish_reason,
        "raw": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "reasoning_tokens": reasoning,
            "forced_end_tokens": forced,
            "answer_tokens": answer_tokens,
            "budget_overshoot": overshoot,
            "empty_answer": not bool(result.answer_text),
            "usage_completion_tokens": result.completion_tokens,
        },
        "timing": _timing(result, forced),
        "resource": {
            "rss_delta_bytes": max(0, after.rss_bytes - before.rss_bytes),
            "vram_delta_bytes": max(0, after.vram_bytes - before.vram_bytes),
            "gpu_temperature_c": after.temperature_c,
            "power_w": after.power_w,
        },
        "quality": {
            "correct": result.answer_text.strip() == fixture.expected_answer,
            "score": 1.0 if result.answer_text.strip() == fixture.expected_answer else 0.0,
        },
    }
    lifecycle = {
        "case_id": case_id,
        "stream": True,
        "outcome": result.finish_reason,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "reset_complete": True,
        "reasoning_tokens": None if mode == "disabled" else reasoning,
        "forced_end_tokens": None if mode == "disabled" else forced,
        "admit_to_start_ns": release.get("admit_to_start_ns"),
        "start_to_release_ns": release.get("start_to_release_ns"),
        "admit_to_release_ns": release.get("admit_to_release_ns"),
    }
    if not all(isinstance(lifecycle[field], int) and lifecycle[field] >= 0 for field in (
        "admit_to_start_ns", "start_to_release_ns", "admit_to_release_ns"
    )):
        raise CampaignError(f"release case {mode} lifecycle timing is invalid")
    sample = {"case_id": case_id, "before": before.__dict__ if hasattr(before, "__dict__") else {
        "rss_bytes": before.rss_bytes, "vram_bytes": before.vram_bytes, "temperature_c": before.temperature_c, "power_w": before.power_w
    }, "after": {
        "rss_bytes": after.rss_bytes, "vram_bytes": after.vram_bytes, "temperature_c": after.temperature_c, "power_w": after.power_w
    }}
    return case, lifecycle, sample


def _write_json(path: Path, value: Any) -> None:
    encoded = (json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n").encode("ascii")
    with path.open("wb") as destination:
        destination.write(encoded)
        destination.flush()
        os.fsync(destination.fileno())


def execute(
    *, output_dir: Path, manifest: Path, fixture_suite: Path, token_file: Path, http_image: str,
    endpoint: str = DEFAULT_ENDPOINT, network: str = "open-webui-network", service: str = "ullm-openai.service",
    observer_socket: Path = DEFAULT_OBSERVER, docker: str = "docker", rocm_smi: str = "rocm-smi",
    systemctl: str = "systemctl", timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise CampaignError("campaign output directory already exists")
    if IMAGE_RE.fullmatch(http_image) is None:
        raise CampaignError("HTTP image must be an immutable Docker SHA-256 identity")
    token = _read_regular(token_file, "OpenWebUI API key file", 65_536)
    del token
    manifest_summary = _validate_manifest(manifest)
    fixtures = _load_fixtures(fixture_suite)
    missing = [mode for mode in MODES if f"generic-reasoning-{mode}" not in fixtures]
    if missing:
        raise CampaignError("fixture suite lacks modes: " + ",".join(missing))
    gpu_preflight = _read_gpu_processes(rocm_smi)
    _bind_gpu_processes(gpu_preflight, manifest_summary["worker"]["binary_sha256"])
    command = _docker_command(docker=docker, image=http_image, key_file=token_file, endpoint=endpoint, network=network)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f".{output_dir.name}.incomplete-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o700, parents=True)
    observer = LifecycleObserver(observer_socket)
    cases: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    try:
        observer.open()
        for mode in MODES:
            fixture = fixtures[f"generic-reasoning-{mode}"]
            before = _resource_sample(service, rocm_smi, systemctl)
            result = _stream_request(
                _request_body(manifest_summary["model_id"], mode, fixture),
                command=command,
                timeout_seconds=timeout_seconds,
            )
            release = observer.wait_release(result.completion_id, timeout_seconds)
            after = _resource_sample(service, rocm_smi, systemctl)
            case, event, sample = _case_and_lifecycle(
                mode=mode, fixture=fixture, result=result, release=release, before=before, after=after
            )
            cases.append(case)
            lifecycle.append(event)
            samples.append(sample)
        _write_json(stage / "cases.json", cases)
        _write_json(stage / "lifecycle.json", {"schema_version": "ullm.generic_reasoning_lifecycle_evidence.v1", "events": lifecycle})
        with (stage / "resource-samples.jsonl").open("w", encoding="ascii") as output:
            for sample in samples:
                output.write(json.dumps(sample, ensure_ascii=True, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        _write_json(stage / "summary.json", {
            "schema_version": CAMPAIGN_SCHEMA,
            "status": "incomplete",
            "raw_bodies_stored": False,
            "case_count": len(cases),
            "modes": list(MODES),
            "manifest_sha256": manifest_summary["manifest_sha256"],
            "model_id": manifest_summary["model_id"],
            "worker_binary_sha256": manifest_summary["worker"]["binary_sha256"],
            "gpu_exclusive_preflight": gpu_preflight,
        })
        stage.chmod(0o700)
        os.replace(stage, output_dir)
    except BaseException:
        import shutil

        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        observer.close()
    return {"schema_version": CAMPAIGN_SCHEMA, "output_dir": os.fspath(output_dir.resolve()), "case_count": len(cases), "modes": list(MODES)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixture-suite", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--http-image", required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--network", default="open-webui-network")
    parser.add_argument("--service", default="ullm-openai.service")
    parser.add_argument("--observer-socket", type=Path, default=DEFAULT_OBSERVER)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--rocm-smi", default="rocm-smi")
    parser.add_argument("--systemctl", default="systemctl")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = execute(
            output_dir=args.output_dir, manifest=args.manifest, fixture_suite=args.fixture_suite,
            token_file=args.token_file, http_image=args.http_image, endpoint=args.endpoint,
            network=args.network, service=args.service, observer_socket=args.observer_socket,
            docker=args.docker, rocm_smi=args.rocm_smi, systemctl=args.systemctl,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as error:
        print(f"Generic reasoning release campaign failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
