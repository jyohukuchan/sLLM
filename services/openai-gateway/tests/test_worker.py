from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import pytest

from ullm_openai_gateway.app import _cancel_stream_and_drain
from ullm_openai_gateway.schemas import EOS_TOKEN_IDS
from ullm_openai_gateway.worker import (
    WorkerBusy,
    WorkerConfig,
    WorkerFatal,
    WorkerGenerationRequest,
    WorkerNotReady,
    WorkerSupervisor,
)


FAKE_WORKER = r"""
import json
import os
import sys
import time

mode = os.environ.get("FAKE_WORKER_MODE", "normal")
if mode == "no_ready":
    time.sleep(10)
ready = {
    "schema_version": "ullm.worker.v1",
    "type": "ready",
    "model": "ullm-qwen3-14b-sq8",
    "model_revision": "9a283b4a5efbc09ce247e0ae5b02b744739e525a",
    "artifact_content_sha256": "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147",
    "package_manifest_sha256": "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb",
    "device": "gfx1201",
    "execution_profile": "rdna4_w8a8_block_ck",
    "context_length": 4096,
    "max_new_tokens": 512,
}
if mode == "bad_ready":
    ready["model"] = "wrong"
print(json.dumps(ready, separators=(",", ":")), flush=True)
if mode == "eof_after_ready":
    sys.exit(0)

active = None
for line in sys.stdin:
    command = json.loads(line)
    kind = command["type"]
    if kind == "generate":
        active = command
        request_id = command["request_id"]
        prompt_tokens = len(command["prompt_token_ids"])
        print(json.dumps({
            "schema_version": "ullm.worker.v1",
            "type": "started",
            "request_id": request_id,
            "prompt_tokens": prompt_tokens,
        }, separators=(",", ":")), flush=True)
        if mode == "no_progress":
            time.sleep(10)
            continue
        progress_values = [prompt_tokens] if mode == "skip_progress" else list(range(128, prompt_tokens + 1, 128))
        if not progress_values or progress_values[-1] != prompt_tokens:
            progress_values.append(prompt_tokens)
        for processed in progress_values:
            print(json.dumps({
                "schema_version": "ullm.worker.v1",
                "type": "progress",
                "request_id": request_id,
                "phase": "prefill",
                "processed_prompt_tokens": processed,
            }, separators=(",", ":")), flush=True)
        if mode in {"wait_cancel", "cancel_hang"}:
            continue
        time.sleep(float(os.environ.get("FAKE_WORKER_DELAY", "0")))
        maximum = command["max_new_tokens"]
        outcome = "length" if maximum == 1 else "stop"
        token_id = 151645 if mode == "length_eos" else (42 if outcome == "length" else 151645)
        token_index = 1 if mode == "token_gap" else 0
        print(json.dumps({
            "schema_version": "ullm.worker.v1",
            "type": "token",
            "request_id": request_id,
            "index": token_index,
            "token_id": token_id,
        }, separators=(",", ":")), flush=True)
        completion_tokens = 1
        if mode == "token_after_eos":
            print(json.dumps({
                "schema_version": "ullm.worker.v1",
                "type": "token",
                "request_id": request_id,
                "index": 1,
                "token_id": 42,
            }, separators=(",", ":")), flush=True)
            outcome = "length"
            completion_tokens = 2
        print(json.dumps({
            "schema_version": "ullm.worker.v1",
            "type": "released",
            "request_id": request_id,
            "outcome": outcome,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reset_complete": True,
        }, separators=(",", ":")), flush=True)
        active = None
    elif kind == "cancel":
        if mode == "cancel_hang":
            time.sleep(10)
            continue
        time.sleep(float(os.environ.get("FAKE_CANCEL_DELAY", "0")))
        print(json.dumps({
            "schema_version": "ullm.worker.v1",
            "type": "released",
            "request_id": command["request_id"],
            "outcome": "cancelled",
            "cancel_reason": command["reason"],
            "prompt_tokens": len(active["prompt_token_ids"]),
            "completion_tokens": 0,
            "reset_complete": True,
        }, separators=(",", ":")), flush=True)
        active = None
    elif kind == "shutdown":
        sys.exit(0)
"""


def write_fake_worker(tmp_path: Path) -> Path:
    path = tmp_path / "fake-worker.py"
    path.write_text(FAKE_WORKER, encoding="ascii")
    return path


def config(
    tmp_path: Path,
    *,
    mode: str = "normal",
    delay: float = 0.0,
    cancel_delay: float = 0.0,
    lock_file: Path | None = None,
    startup_timeout: float = 1.0,
    request_timeout: float = 1.0,
    progress_timeout: float = 1.0,
    cancel_timeout: float = 1.0,
) -> WorkerConfig:
    environment = dict(os.environ)
    environment["FAKE_WORKER_MODE"] = mode
    environment["FAKE_WORKER_DELAY"] = str(delay)
    environment["FAKE_CANCEL_DELAY"] = str(cancel_delay)
    return WorkerConfig(
        command=(sys.executable, str(write_fake_worker(tmp_path))),
        lock_file=lock_file or (tmp_path / "gpu.lock"),
        environment=environment,
        startup_timeout_seconds=startup_timeout,
        request_timeout_seconds=request_timeout,
        progress_timeout_seconds=progress_timeout,
        cancel_timeout_seconds=cancel_timeout,
        terminate_grace_seconds=0.1,
    )


def generation(
    maximum: int = 2, prompt_tokens: int = 3, *, stream: bool = False
) -> WorkerGenerationRequest:
    return WorkerGenerationRequest(
        prompt_token_ids=tuple(range(prompt_tokens)),
        max_new_tokens=maximum,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        stream=stream,
    )


def test_worker_is_reused_and_busy_request_is_not_queued(tmp_path: Path) -> None:
    async def scenario() -> None:
        supervisor = WorkerSupervisor(
            config(tmp_path, delay=0.15), fatal_exit=lambda _: None
        )
        await supervisor.launch()
        await supervisor.wait_ready()
        process_id = supervisor.process_id
        first = await supervisor.admit(generation())
        with pytest.raises(WorkerBusy):
            await supervisor.admit(generation())
        first_result = await supervisor.wait(first)
        assert first_result.outcome == "stop"
        assert first_result.token_ids == (EOS_TOKEN_IDS[0],)
        second = await supervisor.admit(generation(1))
        second_result = await supervisor.wait(second)
        assert second_result.outcome == "length"
        assert second_result.token_ids == (42,)
        assert supervisor.process_id == process_id
        await supervisor.shutdown()

    asyncio.run(scenario())


def test_release_log_is_structured_and_omits_content(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    completion_id = "chatcmpl-0123456789abcdef0123456789abcdef"

    async def scenario() -> None:
        supervisor = WorkerSupervisor(config(tmp_path), fatal_exit=lambda _: None)
        await supervisor.launch()
        await supervisor.wait_ready()
        request = generation()
        request = WorkerGenerationRequest(
            prompt_token_ids=request.prompt_token_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
            completion_id=completion_id,
        )
        result = await supervisor.wait(await supervisor.admit(request))
        assert result.outcome == "stop"
        await supervisor.shutdown()

    caplog.set_level(logging.INFO, logger="uvicorn.error")
    asyncio.run(scenario())
    records = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.getMessage().startswith("{")
    ]
    assert [record["event"] for record in records] == [
        "request_admitted",
        "request_started",
        "request_progress",
        "request_first_token",
        "request_released",
    ]
    assert [record["observed_monotonic_ns"] for record in records] == sorted(
        record["observed_monotonic_ns"] for record in records
    )
    released = [
        record for record in records if record.get("event") == "request_released"
    ]
    assert len(released) == 1
    record = released[0]
    assert set(record) == {
        "schema_version",
        "event",
        "observed_monotonic_ns",
        "request_id",
        "completion_id",
        "stream",
        "outcome",
        "cancel_reason",
        "prompt_tokens",
        "completion_tokens",
        "reset_complete",
        "admit_to_start_ns",
        "start_to_release_ns",
        "admit_to_release_ns",
    }
    assert record["schema_version"] == "ullm.gateway.lifecycle.v1"
    assert record["observed_monotonic_ns"] > 0
    assert record["request_id"].startswith("req-")
    assert record["completion_id"] == completion_id
    assert record["outcome"] == "stop"
    assert record["reset_complete"] is True
    assert record["prompt_tokens"] == 3
    assert record["completion_tokens"] == 1
    assert 0 <= record["admit_to_start_ns"] <= record["admit_to_release_ns"]
    assert 0 <= record["start_to_release_ns"] <= record["admit_to_release_ns"]
    serialized = json.dumps(records, sort_keys=True)
    for forbidden in ("prompt_token_ids", "token_ids", "test-secret"):
        assert forbidden not in serialized


def test_cancel_log_precedes_matching_cancelled_release(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async def scenario() -> None:
        supervisor = WorkerSupervisor(
            config(tmp_path, mode="wait_cancel"), fatal_exit=lambda _: None
        )
        await supervisor.launch()
        await supervisor.wait_ready()
        handle = await supervisor.admit(generation(stream=True))
        await supervisor.wait_started(handle)
        await supervisor.cancel(handle, "operator")
        assert (await supervisor.wait(handle)).outcome == "cancelled"
        await supervisor.shutdown()

    caplog.set_level(logging.INFO, logger="uvicorn.error")
    asyncio.run(scenario())
    records = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.getMessage().startswith("{")
    ]
    cancel = next(
        record for record in records if record["event"] == "request_cancel_requested"
    )
    released = next(
        record for record in records if record["event"] == "request_released"
    )
    assert cancel["request_id"] == released["request_id"]
    assert cancel["reason"] == "operator"
    assert released["outcome"] == "cancelled"
    assert released["cancel_reason"] == "operator"
    assert released["reset_complete"] is True
    assert cancel["observed_monotonic_ns"] <= released["observed_monotonic_ns"]


def test_slot_remains_busy_from_cancel_until_matching_release(tmp_path: Path) -> None:
    async def scenario() -> None:
        supervisor = WorkerSupervisor(
            config(tmp_path, mode="wait_cancel", cancel_delay=0.1),
            fatal_exit=lambda _: None,
        )
        await supervisor.launch()
        await supervisor.wait_ready()
        handle = await supervisor.admit(generation())
        await asyncio.sleep(0.02)
        await supervisor.cancel(handle, "client_disconnect")
        with pytest.raises(WorkerBusy):
            await supervisor.admit(generation())
        result = await supervisor.wait(handle)
        assert result.outcome == "cancelled"
        await supervisor.shutdown()

    asyncio.run(scenario())


def test_gpu_singleton_lock_rejects_second_supervisor(tmp_path: Path) -> None:
    async def scenario() -> None:
        lock = tmp_path / "shared.lock"
        first = WorkerSupervisor(
            config(tmp_path / "first", lock_file=lock), fatal_exit=lambda _: None
        )
        second = WorkerSupervisor(
            config(tmp_path / "second", lock_file=lock), fatal_exit=lambda _: None
        )
        await first.launch()
        await first.wait_ready()
        with pytest.raises(WorkerBusy):
            await second.launch()
        await first.shutdown()

    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mode",
    [
        "bad_ready",
        "token_gap",
        "eof_after_ready",
        "token_after_eos",
        "length_eos",
        "skip_progress",
    ],
)
def test_identity_protocol_and_eof_fail_closed(tmp_path: Path, mode: str) -> None:
    async def scenario() -> None:
        exits: list[int] = []
        supervisor = WorkerSupervisor(
            config(tmp_path, mode=mode), fatal_exit=exits.append
        )
        await supervisor.launch()
        if mode in {"token_gap", "token_after_eos", "length_eos", "skip_progress"}:
            await supervisor.wait_ready()
            request = generation(
                maximum=1 if mode == "length_eos" else 2,
                prompt_tokens=256 if mode == "skip_progress" else 3,
            )
            handle = await supervisor.admit(request)
            with pytest.raises(WorkerFatal):
                await supervisor.wait(handle)
        else:
            try:
                await supervisor.wait_ready()
            except WorkerFatal:
                pass
        for _ in range(100):
            if exits:
                break
            await asyncio.sleep(0.01)
        assert exits == [1]
        assert supervisor.ready is False
        await supervisor.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mode,timeout_name",
    [("no_progress", "progress"), ("cancel_hang", "cancel")],
)
def test_hard_watchdogs_terminate_worker(
    tmp_path: Path, mode: str, timeout_name: str
) -> None:
    async def scenario() -> None:
        exits: list[int] = []
        supervisor = WorkerSupervisor(
            config(
                tmp_path,
                mode=mode,
                request_timeout=1.0,
                progress_timeout=0.05,
                cancel_timeout=0.05,
            ),
            fatal_exit=exits.append,
        )
        await supervisor.launch()
        await supervisor.wait_ready()
        handle = await supervisor.admit(generation())
        if timeout_name == "cancel":
            await asyncio.sleep(0.02)
            await supervisor.cancel(handle, "client_disconnect")
        with pytest.raises(WorkerFatal):
            await supervisor.wait(handle)
        for _ in range(100):
            if exits:
                break
            await asyncio.sleep(0.01)
        assert exits == [1]
        assert supervisor.ready is False
        await supervisor.shutdown()

    asyncio.run(scenario())


def test_startup_watchdog_terminates_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        exits: list[int] = []
        supervisor = WorkerSupervisor(
            config(tmp_path, mode="no_ready", startup_timeout=0.05),
            fatal_exit=exits.append,
        )
        await supervisor.launch()
        with pytest.raises(WorkerFatal):
            await supervisor.wait_ready()
        for _ in range(100):
            if exits:
                break
            await asyncio.sleep(0.01)
        assert exits == [1]
        await supervisor.shutdown()

    asyncio.run(scenario())


def test_total_request_watchdog_terminates_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        exits: list[int] = []
        supervisor = WorkerSupervisor(
            config(
                tmp_path,
                mode="wait_cancel",
                request_timeout=0.05,
                progress_timeout=1.0,
            ),
            fatal_exit=exits.append,
        )
        await supervisor.launch()
        await supervisor.wait_ready()
        handle = await supervisor.admit(generation())
        with pytest.raises(WorkerFatal):
            await supervisor.wait(handle)
        for _ in range(100):
            if exits:
                break
            await asyncio.sleep(0.01)
        assert exits == [1]
        await supervisor.shutdown()

    asyncio.run(scenario())


def test_cancel_losing_to_buffered_terminal_and_stale_error_is_recoverable(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        exits: list[int] = []
        supervisor = WorkerSupervisor(
            config(tmp_path, mode="cancel_hang"), fatal_exit=exits.append
        )
        await supervisor.launch()
        await supervisor.wait_ready()
        handle = await supervisor.admit(generation())
        await asyncio.sleep(0.02)
        await supervisor.cancel(handle, "client_disconnect")
        await supervisor._handle_event(
            {
                "schema_version": "ullm.worker.v1",
                "type": "token",
                "request_id": handle.request_id,
                "index": 0,
                "token_id": EOS_TOKEN_IDS[0],
            }
        )
        await supervisor._handle_event(
            {
                "schema_version": "ullm.worker.v1",
                "type": "released",
                "request_id": handle.request_id,
                "outcome": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "reset_complete": True,
            }
        )
        result = await supervisor.wait(handle)
        assert result.outcome == "stop"
        await supervisor._handle_event(
            {
                "schema_version": "ullm.worker.v1",
                "type": "error",
                "request_id": handle.request_id,
                "code": "unknown_request",
                "recoverable": True,
                "message": "request is no longer active",
            }
        )
        assert exits == []
        await supervisor.shutdown()

    asyncio.run(scenario())


def test_shutdown_pipe_error_still_reaps_worker_and_releases_lock(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        exits: list[int] = []
        supervisor = WorkerSupervisor(config(tmp_path), fatal_exit=exits.append)
        await supervisor.launch()
        await supervisor.wait_ready()

        async def failed_write(_: object) -> None:
            raise OSError("injected pipe failure")

        supervisor._write_command = failed_write  # type: ignore[method-assign]
        await supervisor.shutdown()
        assert supervisor._process is not None
        assert supervisor._process.returncode is not None
        assert supervisor._lock_descriptor is None
        assert exits == []

    asyncio.run(scenario())


def test_fatal_cleanup_waits_for_http_error_attempt_ack(tmp_path: Path) -> None:
    async def scenario() -> None:
        exits: list[int] = []
        supervisor = WorkerSupervisor(
            config(tmp_path, mode="token_gap"), fatal_exit=exits.append
        )
        await supervisor.launch()
        await supervisor.wait_ready()
        handle = await supervisor.admit(generation())
        with pytest.raises(WorkerFatal):
            await supervisor.wait(handle)
        await asyncio.sleep(0.05)
        assert exits == []
        await supervisor.acknowledge_fatal_response()
        for _ in range(100):
            if exits:
                break
            await asyncio.sleep(0.01)
        assert exits == [1]
        await supervisor.shutdown()

    asyncio.run(scenario())


def test_idle_fatal_still_waits_for_http_lifecycle_ack(tmp_path: Path) -> None:
    async def scenario() -> None:
        exits: list[int] = []
        supervisor = WorkerSupervisor(config(tmp_path), fatal_exit=exits.append)
        await supervisor.launch()
        await supervisor.wait_ready()
        fatal = asyncio.create_task(supervisor._fatal("injected idle failure"))
        await supervisor.wait_fatal()
        await asyncio.sleep(0.05)
        assert exits == []
        await supervisor.acknowledge_fatal_response()
        await fatal
        assert exits == [1]
        await supervisor.shutdown()

    asyncio.run(scenario())


def test_gateway_fatal_request_poison_is_synchronous_after_release(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        exits: list[int] = []
        supervisor = WorkerSupervisor(config(tmp_path), fatal_exit=exits.append)
        await supervisor.launch()
        await supervisor.wait_ready()
        handle = await supervisor.admit(generation())
        assert (await supervisor.wait(handle)).outcome == "stop"
        supervisor.request_fatal("injected post-release gateway failure")
        assert supervisor.failed
        assert not supervisor.ready
        with pytest.raises(WorkerNotReady):
            await supervisor.admit(generation())
        await supervisor.acknowledge_fatal_response()
        for _ in range(100):
            if exits:
                break
            await asyncio.sleep(0.01)
        assert exits == [1]
        await supervisor.shutdown()

    asyncio.run(scenario())


def test_stream_queue_overflow_requests_immediate_slow_client_cancel(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        supervisor = WorkerSupervisor(
            config(tmp_path, mode="cancel_hang"), fatal_exit=lambda _: None
        )
        await supervisor.launch()
        await supervisor.wait_ready()
        handle = await supervisor.admit(generation(maximum=64, stream=True))
        await supervisor.wait_started(handle)
        assert handle.stream_state is not None
        for _ in range(100):
            active = supervisor._active
            if active is not None and active.processed_prompt_tokens == 3:
                break
            await asyncio.sleep(0.01)
        assert supervisor._active is not None
        assert supervisor._active.processed_prompt_tokens == 3
        for index in range(33):
            await supervisor._handle_event(
                {
                    "schema_version": "ullm.worker.v1",
                    "type": "token",
                    "request_id": handle.request_id,
                    "index": index,
                    "token_id": 42,
                }
            )
        for _ in range(100):
            active = supervisor._active
            if active is not None and active.cancel_reason == "slow_client":
                break
            await asyncio.sleep(0.01)
        assert handle.stream_state.token_queue.qsize() == 32
        assert handle.stream_state.aborted_reason == "slow_client"
        assert handle.stream_state.aborted.is_set()
        assert supervisor._active is not None
        assert supervisor._active.cancel_reason == "slow_client"
        await supervisor.shutdown()
        if handle._future.done():
            handle._future.exception()

    asyncio.run(scenario())


def test_stream_queue_abort_and_http_cleanup_share_slow_client_reason(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        supervisor = WorkerSupervisor(
            config(tmp_path, mode="wait_cancel"), fatal_exit=lambda _: None
        )
        cancel_commands: list[dict[str, object]] = []
        original_write = supervisor._write_command

        async def record_write(command: dict[str, object]) -> None:
            if command.get("type") == "cancel":
                cancel_commands.append(command)
            await original_write(command)

        supervisor._write_command = record_write  # type: ignore[method-assign]
        await supervisor.launch()
        await supervisor.wait_ready()
        handle = await supervisor.admit(generation(maximum=64, stream=True))
        await supervisor.wait_started(handle)
        assert handle.stream_state is not None
        for _ in range(32):
            handle.stream_state.token_queue.put_nowait(42)
        active = supervisor._active
        assert active is not None
        supervisor._publish_stream_token(active, 42)
        assert handle.stream_state.aborted_reason == "slow_client"
        cleanup = asyncio.create_task(_cancel_stream_and_drain(supervisor, handle))
        result = await asyncio.wait_for(supervisor.wait(handle), timeout=1.0)
        await cleanup
        assert result.outcome == "cancelled"
        assert len(cancel_commands) == 1
        assert cancel_commands[0]["reason"] == "slow_client"
        await supervisor.shutdown()

    asyncio.run(scenario())
