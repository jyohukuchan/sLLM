from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/run-generic-reasoning-release-campaign.py"
VALIDATOR_PATH = ROOT / "tools/validate-generic-reasoning-release.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generic_reasoning_release_campaign", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generic_reasoning_release_validator_for_campaign", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_validator()


def test_modes_and_request_body_are_explicit_and_bounded() -> None:
    fixture = TOOL.Fixture("fixture", "hello", "ok")

    assert TOOL._mode_fields("disabled") == {"reasoning_effort": "none"}
    assert TOOL._mode_fields("budget-32") == {"thinking_budget_tokens": 32}
    assert TOOL._mode_fields("budget-128") == {"thinking_budget_tokens": 128}
    assert TOOL._mode_fields("budget-256") == {"thinking_budget_tokens": 256}
    assert TOOL._mode_fields("unbounded") == {"thinking_budget_tokens": -1}

    body = json.loads(TOOL._request_body("model", "budget-128", fixture))
    assert body["model"] == "model"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["max_completion_tokens"] == 512
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["thinking_budget_tokens"] == 128

    with pytest.raises(TOOL.CampaignError, match="unknown release mode"):
        TOOL._mode_fields("invalid")


def test_nonstream_response_is_bounded_and_matches_release_case_contract() -> None:
    payload = {
        "id": "completion-nonstream",
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        },
        "timings": {"prompt_per_second": 100.0, "predicted_per_second": 80.0},
    }
    output = json.dumps(payload, separators=(",", ":")).encode("ascii")
    marker = b"\n__ULLM_HTTP_STATUS__200\n"
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdin.buffer.read(); sys.stdout.buffer.write(" + repr(output + marker) + ")",
    ]
    result = TOOL._nonstream_request(b"request", command=command, timeout_seconds=2.0)

    assert result.stream is False
    assert result.status == 200
    assert result.completion_id == "completion-nonstream"
    assert result.sse_chunk_count == 0
    assert result.answer_text == "ok"
    assert result.reasoning_tokens == 0

    fixture = TOOL.Fixture("fixture", "hello", "ok")
    release = {
        "completion_id": result.completion_id,
        "outcome": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "reset_complete": True,
        "admit_to_start_ns": 1,
        "start_to_release_ns": 2,
        "admit_to_release_ns": 3,
    }
    sample = TOOL.ResourceSample(100, 200, 50.0, 100.0)
    case, lifecycle, _ = TOOL._case_and_lifecycle(
        mode="disabled", fixture=fixture, result=result, release=release,
        before=sample, after=sample,
    )
    assert case["id"] == "generic-reasoning-disabled-nonstream"
    assert case["stream"] is False
    assert case["sse_chunk_count"] == 0
    assert case["timing"]["answer_decode_tokens_per_second"] == 80.0
    assert VALIDATOR._validate_case(case) == "disabled"
    assert VALIDATOR._validate_lifecycle({
        "schema_version": VALIDATOR.LIFECYCLE_SCHEMA_VERSION,
        "events": [lifecycle],
    }, {case["id"]: case})["case_ids"] == {case["id"]}


def test_stream_and_nonstream_semantics_are_compared_without_persisting_text() -> None:
    fields = {
        "status": 200,
        "completion_id": "different-id",
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "reasoning_tokens": 1,
        "usage_timings": {},
        "answer_text": "ok",
        "reasoning_text": "step",
        "sse_chunk_count": 3,
        "first_reasoning_ms": 1.0,
        "first_answer_ms": 2.0,
        "latency_ms": 3.0,
    }
    stream = TOOL.StreamResult(**fields, stream=True)
    nonstream = TOOL.StreamResult(**fields, stream=False)

    TOOL._assert_transport_match("budget-32", stream, nonstream)
    nonstream_mismatch = TOOL.StreamResult(**{**fields, "answer_text": "different"}, stream=False)
    with pytest.raises(TOOL.CampaignError, match="stream/non-stream contract differs"):
        TOOL._assert_transport_match("budget-32", stream, nonstream_mismatch)


def test_manifest_preflight_rejects_v1_before_external_validation(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "ullm.served_model.v1"}), encoding="ascii")

    with pytest.raises(TOOL.CampaignError, match="not v2"):
        TOOL._validate_manifest(manifest)


def test_immutable_http_image_is_required(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_bytes(b"opaque-token")

    with pytest.raises(TOOL.CampaignError, match="immutable Docker"):
        TOOL.execute(
            output_dir=tmp_path / "out",
            manifest=tmp_path / "missing-manifest",
            fixture_suite=TOOL.DEFAULT_FIXTURES,
            token_file=token,
            http_image="curl:latest",
        )


def test_gpu_process_identity_is_bound_to_manifest_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    preflight = {
        "positive_vram_processes": [{"pid": "123", "process": "ullm-aq4-worker"}]
    }
    monkeypatch.setattr(TOOL, "_hash_process_executable", lambda _pid: "a" * 64)

    TOOL._bind_gpu_processes(preflight, "a" * 64)
    assert preflight["positive_vram_processes"][0]["binary_sha256"] == "a" * 64

    monkeypatch.setattr(TOOL, "_hash_process_executable", lambda _pid: "b" * 64)
    with pytest.raises(TOOL.CampaignError, match="differs from the v2 manifest"):
        TOOL._bind_gpu_processes(preflight, "a" * 64)


def test_lifecycle_observer_correlates_release_and_removes_socket(tmp_path: Path) -> None:
    path = tmp_path / "observer.sock"
    observer = TOOL.LifecycleObserver(path)
    observer.open()
    sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

    def send() -> None:
        sender.sendto(
            json.dumps(
                {
                    "schema_version": TOOL.LIFECYCLE_SCHEMA,
                    "event": "request_released",
                    "completion_id": "matching",
                }
            ).encode("ascii"),
            str(path),
        )

    thread = threading.Thread(target=send)
    thread.start()
    assert observer.wait_release("matching", 2.0)["completion_id"] == "matching"
    thread.join()
    sender.close()
    observer.close()
    assert not path.exists()
