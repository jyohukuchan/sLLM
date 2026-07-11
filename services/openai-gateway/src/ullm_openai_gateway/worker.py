"""Resident SQ8 worker supervision and strict JSONL event validation."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import signal
import socket
import stat
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .schemas import EOS_TOKEN_IDS, MODEL_ID, TOP_K
from .settings import GatewaySettings


WORKER_SCHEMA = "ullm.worker.v1"
MODEL_REVISION = "9a283b4a5efbc09ce247e0ae5b02b744739e525a"
ARTIFACT_CONTENT_SHA256 = (
    "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147"
)
PACKAGE_MANIFEST_SHA256 = (
    "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb"
)
DEVICE = "gfx1201"
EXECUTION_PROFILE = "rdna4_w8a8_block_ck"
CONTEXT_LENGTH = 4_096
MAX_NEW_TOKENS = 512
VOCAB_SIZE = 151_936
MAX_EVENT_BYTES = 65_536
WRITE_TIMEOUT_SECONDS = 5.0
FATAL_RESPONSE_ATTEMPT_SECONDS = 0.25
KILL_WAIT_SECONDS = 2.0
PREFILL_PROGRESS_TOKENS = 128
STREAM_TOKEN_QUEUE_SIZE = 32
LIFECYCLE_LOG_SCHEMA = "ullm.gateway.lifecycle.v1"
LIFECYCLE_OBSERVER_SOCKET = Path("/run/ullm/lifecycle-observer.sock")

_LIFECYCLE_LOGGER = logging.getLogger("uvicorn.error")

HIP_GUARDS = (
    "ULLM_REQUIRE_HIP_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL",
    "ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL",
    "ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
    "ULLM_REQUIRE_HIP_ROPE_KERNEL",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL",
)


class WorkerError(RuntimeError):
    """Base class for gateway-to-worker failures."""


class WorkerNotReady(WorkerError):
    pass


class WorkerBusy(WorkerError):
    pass


class WorkerFatal(WorkerError):
    pass


class WorkerProtocolError(WorkerError):
    pass


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    command: tuple[str, ...]
    lock_file: Path
    environment: dict[str, str]
    startup_timeout_seconds: float = 600.0
    request_timeout_seconds: float = 180.0
    progress_timeout_seconds: float = 30.0
    cancel_timeout_seconds: float = 5.0
    terminate_grace_seconds: float = 2.0

    @classmethod
    def from_settings(cls, settings: GatewaySettings) -> "WorkerConfig":
        environment = dict(os.environ)
        environment["HIP_VISIBLE_DEVICES"] = "1"
        for name in HIP_GUARDS:
            environment[name] = "1"
        return cls(
            command=(
                str(settings.worker_binary),
                "--artifact",
                str(settings.artifact_dir),
                "--package",
                str(settings.package_dir),
            ),
            lock_file=settings.gpu_lock_file,
            environment=environment,
        )


@dataclass(frozen=True, slots=True)
class WorkerGenerationRequest:
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    stream: bool = False
    completion_id: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerGenerationResult:
    request_id: str
    outcome: str
    prompt_tokens: int
    token_ids: tuple[int, ...]


@dataclass(slots=True)
class WorkerStreamState:
    token_queue: asyncio.Queue[int]
    aborted_reason: str | None = None
    aborted: asyncio.Event = field(default_factory=asyncio.Event)

    def abort(self, reason: str) -> None:
        if self.aborted_reason is None:
            self.aborted_reason = reason
            self.aborted.set()


@dataclass(frozen=True, slots=True)
class GenerationHandle:
    request_id: str
    _generation: int
    _future: asyncio.Future[WorkerGenerationResult]
    _started_future: asyncio.Future[None] | None = None
    stream_state: WorkerStreamState | None = None


@dataclass(slots=True)
class _ActiveRequest:
    request_id: str
    generation: int
    request: WorkerGenerationRequest
    future: asyncio.Future[WorkerGenerationResult]
    started_future: asyncio.Future[None]
    stream_state: WorkerStreamState | None
    admitted_monotonic_ns: int
    started: bool = False
    started_monotonic_ns: int | None = None
    processed_prompt_tokens: int = 0
    token_ids: list[int] = field(default_factory=list)
    terminal_outcome: str | None = None
    cancel_reason: str | None = None
    last_progress: float = 0.0
    progress_wakeup: asyncio.Event = field(default_factory=asyncio.Event)
    total_watchdog: asyncio.Task[None] | None = None
    progress_watchdog: asyncio.Task[None] | None = None
    cancel_watchdog: asyncio.Task[None] | None = None


class WorkerSupervisor:
    def __init__(
        self,
        config: WorkerConfig,
        *,
        fatal_exit: Callable[[int], None] | None = None,
    ) -> None:
        self._config = config
        self._fatal_exit = fatal_exit if fatal_exit is not None else os._exit
        self._process: asyncio.subprocess.Process | None = None
        self._state = "shutdown"
        self._state_lock = asyncio.Lock()
        self._protocol_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._fatal_lock = asyncio.Lock()
        self._fatal_started = False
        self._fatal_reserved = False
        self._fatal_event = asyncio.Event()
        self._stopping = False
        self._active: _ActiveRequest | None = None
        self._generation = 0
        self._ready_future: asyncio.Future[None] | None = None
        self._fatal_response_attempted = asyncio.Event()
        self._lock_descriptor: int | None = None
        self._stale_cancel_request_id: str | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def ready(self) -> bool:
        return (
            self._state == "ready"
            and not self._fatal_reserved
            and not self._fatal_started
        )

    @property
    def process_id(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def failed(self) -> bool:
        return self._fatal_reserved or self._fatal_started

    async def launch(self) -> None:
        if self._process is not None:
            raise WorkerError("worker supervisor has already launched")
        self._lock_descriptor = _acquire_singleton_lock(self._config.lock_file)
        self._state = "loading"
        loop = asyncio.get_running_loop()
        self._ready_future = loop.create_future()
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._config.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._config.environment,
                start_new_session=True,
                limit=MAX_EVENT_BYTES + 1,
            )
        except BaseException:
            self._state = "failed"
            self._release_singleton_lock()
            raise
        self._track(self._stdout_loop(), "ullm-worker-stdout")
        self._track(self._stderr_loop(), "ullm-worker-stderr")
        self._track(self._wait_loop(), "ullm-worker-wait")
        self._track(self._startup_watchdog(), "ullm-worker-startup-watchdog")

    async def wait_ready(self) -> None:
        if self._ready_future is None:
            raise WorkerNotReady("worker has not launched")
        await asyncio.shield(self._ready_future)

    async def admit(self, request: WorkerGenerationRequest) -> GenerationHandle:
        _validate_generation_request(request)
        loop = asyncio.get_running_loop()
        async with self._state_lock:
            if not self.ready:
                raise WorkerNotReady("worker is not ready")
            if self._active is not None:
                raise WorkerBusy("worker already has an active request")
            self._fatal_response_attempted.clear()
            self._generation += 1
            request_id = f"req-{uuid.uuid4().hex}"
            future: asyncio.Future[WorkerGenerationResult] = loop.create_future()
            started_future: asyncio.Future[None] = loop.create_future()
            stream_state = (
                WorkerStreamState(asyncio.Queue(maxsize=STREAM_TOKEN_QUEUE_SIZE))
                if request.stream
                else None
            )
            active = _ActiveRequest(
                request_id=request_id,
                generation=self._generation,
                request=request,
                future=future,
                started_future=started_future,
                stream_state=stream_state,
                admitted_monotonic_ns=time.monotonic_ns(),
                last_progress=loop.time(),
            )
            self._active = active
            active.total_watchdog = self._track(
                self._request_watchdog(request_id, active.generation),
                f"ullm-request-{active.generation}-deadline",
            )
        command = {
            "schema_version": WORKER_SCHEMA,
            "type": "generate",
            "request_id": request_id,
            "prompt_token_ids": list(request.prompt_token_ids),
            "max_new_tokens": request.max_new_tokens,
            "sampling": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": TOP_K,
                "seed": request.seed,
            },
            "eos_token_ids": list(EOS_TOKEN_IDS),
        }
        _log_lifecycle(
            "request_admitted",
            request_id=active.request_id,
            completion_id=active.request.completion_id,
            stream=active.request.stream,
            prompt_tokens=len(active.request.prompt_token_ids),
            max_completion_tokens=active.request.max_new_tokens,
        )
        try:
            await self._write_command(command)
        except BaseException as error:
            if self._reserve_fatal():
                self._track(
                    self._fatal("worker command write failed"),
                    "ullm-worker-command-write-fatal",
                )
            raise WorkerFatal("worker command write failed") from error
        active.progress_watchdog = self._track(
            self._progress_watchdog(request_id, active.generation),
            f"ullm-request-{active.generation}-progress",
        )
        return GenerationHandle(
            request_id,
            active.generation,
            future,
            started_future,
            stream_state,
        )

    async def wait(self, handle: GenerationHandle) -> WorkerGenerationResult:
        return await asyncio.shield(handle._future)

    async def wait_started(self, handle: GenerationHandle) -> None:
        if handle._started_future is None:
            return
        await asyncio.shield(handle._started_future)

    async def wait_fatal(self) -> None:
        await self._fatal_event.wait()

    async def cancel(self, handle: GenerationHandle, reason: str) -> None:
        if reason not in {"client_disconnect", "slow_client", "shutdown", "operator"}:
            raise ValueError("invalid worker cancellation reason")
        async with self._protocol_lock:
            active = self._active
            if (
                active is None
                or active.request_id != handle.request_id
                or active.generation != handle._generation
            ):
                if handle._future.done():
                    return
                raise WorkerProtocolError("generation handle is not active")
            if active.terminal_outcome is not None:
                return
            if active.cancel_reason is not None:
                return
            try:
                await self._write_command(
                    {
                        "schema_version": WORKER_SCHEMA,
                        "type": "cancel",
                        "request_id": handle.request_id,
                        "reason": reason,
                    }
                )
            except BaseException as error:
                if self._reserve_fatal():
                    self._track(
                        self._fatal("worker cancel write failed"),
                        "ullm-worker-cancel-write-fatal",
                    )
                raise WorkerFatal("worker cancel write failed") from error
            active.cancel_reason = reason
            _log_lifecycle(
                "request_cancel_requested",
                request_id=active.request_id,
                completion_id=active.request.completion_id,
                stream=active.request.stream,
                reason=reason,
                admit_to_cancel_ns=(time.monotonic_ns() - active.admitted_monotonic_ns),
            )
            active.cancel_watchdog = self._track(
                self._cancel_watchdog(active.request_id, active.generation),
                f"ullm-request-{active.generation}-cancel",
            )

    async def acknowledge_fatal_response(self) -> None:
        self._fatal_response_attempted.set()

    def request_fatal(self, reason: str) -> None:
        if not self._reserve_fatal():
            return
        self._fatal_response_attempted.clear()
        self._track(
            self._fatal(reason),
            "ullm-gateway-requested-fatal",
        )

    async def shutdown(self) -> None:
        if self._process is None:
            self._release_singleton_lock()
            return
        self._stopping = True
        self._state = "shutdown"
        interrupted: BaseException | None = None
        try:
            if self._process.returncode is None:
                try:
                    await self._write_command(
                        {"schema_version": WORKER_SCHEMA, "type": "shutdown"}
                    )
                    await asyncio.wait_for(
                        asyncio.shield(self._process.wait()),
                        self._config.cancel_timeout_seconds,
                    )
                except BaseException as error:
                    interrupted = error
            if self._process.returncode is None:
                await self._terminate_worker()
        finally:
            for task in tuple(self._tasks):
                task.cancel()
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
            self._tasks.clear()
            active = self._active
            if active is not None and not active.future.done():
                active.future.set_exception(WorkerFatal("gateway shut down"))
            if active is not None and not active.started_future.done():
                active.started_future.set_exception(WorkerFatal("gateway shut down"))
            self._active = None
            self._close_worker_stdin()
            if self._process.returncode is not None:
                self._release_singleton_lock()
            else:
                self._fatal_exit(1)
                raise WorkerFatal("worker termination was not confirmed")
        if isinstance(interrupted, asyncio.CancelledError):
            raise interrupted

    def _track(self, coroutine: Any, name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _stdout_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    if not self._stopping:
                        await self._fatal("unexpected worker stdout EOF")
                    return
                if len(line) > MAX_EVENT_BYTES or not line.endswith(b"\n"):
                    raise WorkerProtocolError("worker event framing is invalid")
                value = _decode_event(line[:-1])
                await self._handle_event(value)
        except asyncio.CancelledError:
            raise
        except BaseException:
            await self._fatal("worker stdout protocol failed")

    async def _stderr_loop(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        try:
            while await self._process.stderr.read(64 * 1024):
                pass
        except asyncio.CancelledError:
            raise
        except BaseException:
            if not self._stopping:
                await self._fatal("worker stderr drain failed")

    async def _wait_loop(self) -> None:
        assert self._process is not None
        await self._process.wait()
        if not self._stopping:
            await self._fatal("worker process exited")

    async def _startup_watchdog(self) -> None:
        assert self._ready_future is not None
        try:
            await asyncio.wait_for(
                asyncio.shield(self._ready_future),
                self._config.startup_timeout_seconds,
            )
        except asyncio.TimeoutError:
            await self._fatal("worker ready deadline exceeded")
        except WorkerFatal:
            return

    async def _request_watchdog(self, request_id: str, generation: int) -> None:
        await asyncio.sleep(self._config.request_timeout_seconds)
        if self._matches_active(request_id, generation):
            await self._fatal("worker request deadline exceeded")

    async def _progress_watchdog(self, request_id: str, generation: int) -> None:
        while True:
            active = self._matching_active(request_id, generation)
            if active is None:
                return
            remaining = self._config.progress_timeout_seconds - (
                asyncio.get_running_loop().time() - active.last_progress
            )
            if remaining <= 0:
                await self._fatal("worker progress deadline exceeded")
                return
            try:
                await asyncio.wait_for(active.progress_wakeup.wait(), remaining)
                active.progress_wakeup.clear()
            except asyncio.TimeoutError:
                if self._matches_active(request_id, generation):
                    await self._fatal("worker progress deadline exceeded")
                return

    async def _cancel_watchdog(self, request_id: str, generation: int) -> None:
        await asyncio.sleep(self._config.cancel_timeout_seconds)
        if self._matches_active(request_id, generation):
            await self._fatal("worker cancel-to-release deadline exceeded")

    async def _write_command(self, value: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise WorkerFatal("worker stdin is unavailable")
        payload = (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )
        async with self._write_lock:
            process.stdin.write(payload)
            await asyncio.wait_for(process.stdin.drain(), WRITE_TIMEOUT_SECONDS)

    async def _handle_event(self, value: dict[str, Any]) -> None:
        async with self._protocol_lock:
            await self._handle_event_locked(value)

    async def _handle_event_locked(self, value: dict[str, Any]) -> None:
        if value.get("schema_version") != WORKER_SCHEMA:
            raise WorkerProtocolError("worker schema version differs")
        event_type = value.get("type")
        if event_type == "ready":
            await self._handle_ready(value)
        elif event_type == "started":
            self._handle_started(value)
        elif event_type == "progress":
            self._handle_progress(value)
        elif event_type == "token":
            self._handle_token(value)
        elif event_type == "released":
            await self._handle_released(value)
        elif event_type == "error":
            _validate_error_event(value)
            if (
                value.get("recoverable") is True
                and value.get("code") == "unknown_request"
                and value.get("request_id") == self._stale_cancel_request_id
            ):
                self._stale_cancel_request_id = None
                return
            raise WorkerProtocolError("worker emitted an unexpected error event")
        else:
            raise WorkerProtocolError("worker event type is invalid")

    async def _handle_ready(self, value: dict[str, Any]) -> None:
        expected = {
            "schema_version": WORKER_SCHEMA,
            "type": "ready",
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "artifact_content_sha256": ARTIFACT_CONTENT_SHA256,
            "package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
            "device": DEVICE,
            "execution_profile": EXECUTION_PROFILE,
            "context_length": CONTEXT_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
        }
        if value != expected or self._state != "loading" or self._active is not None:
            raise WorkerProtocolError("worker ready identity or ordering differs")
        self._state = "ready"
        assert self._ready_future is not None
        if self._ready_future.done():
            raise WorkerProtocolError("worker emitted ready more than once")
        self._ready_future.set_result(None)

    def _handle_started(self, value: dict[str, Any]) -> None:
        _require_exact_keys(
            value, {"schema_version", "type", "request_id", "prompt_tokens"}
        )
        request_id = value.get("request_id")
        if request_id != self._stale_cancel_request_id:
            self._stale_cancel_request_id = None
        active = self._require_active_id(request_id)
        if active.started:
            raise WorkerProtocolError("worker emitted started more than once")
        if _integer(value.get("prompt_tokens")) != len(active.request.prompt_token_ids):
            raise WorkerProtocolError("worker started prompt count differs")
        active.started = True
        active.started_monotonic_ns = time.monotonic_ns()
        _log_lifecycle(
            "request_started",
            observed_monotonic_ns=active.started_monotonic_ns,
            request_id=active.request_id,
            completion_id=active.request.completion_id,
            stream=active.request.stream,
            prompt_tokens=len(active.request.prompt_token_ids),
            admit_to_start_ns=(
                active.started_monotonic_ns - active.admitted_monotonic_ns
            ),
        )
        if active.started_future.done():
            raise WorkerProtocolError("worker started future completed too early")
        active.started_future.set_result(None)
        self._mark_progress(active)

    def _handle_progress(self, value: dict[str, Any]) -> None:
        _require_exact_keys(
            value,
            {
                "schema_version",
                "type",
                "request_id",
                "phase",
                "processed_prompt_tokens",
            },
        )
        active = self._require_active_id(value.get("request_id"))
        processed = _integer(value.get("processed_prompt_tokens"))
        expected = min(
            active.processed_prompt_tokens + PREFILL_PROGRESS_TOKENS,
            len(active.request.prompt_token_ids),
        )
        if (
            not active.started
            or value.get("phase") != "prefill"
            or processed != expected
            or active.token_ids
        ):
            raise WorkerProtocolError("worker progress event violates ordering")
        active.processed_prompt_tokens = processed
        _log_lifecycle(
            "request_progress",
            request_id=active.request_id,
            completion_id=active.request.completion_id,
            phase="prefill",
            processed_prompt_tokens=processed,
            prompt_tokens=len(active.request.prompt_token_ids),
        )
        self._mark_progress(active)

    def _handle_token(self, value: dict[str, Any]) -> None:
        _require_exact_keys(
            value, {"schema_version", "type", "request_id", "index", "token_id"}
        )
        active = self._require_active_id(value.get("request_id"))
        index = _integer(value.get("index"))
        token_id = _integer(value.get("token_id"))
        if (
            not active.started
            or active.processed_prompt_tokens != len(active.request.prompt_token_ids)
            or index != len(active.token_ids)
            or not 0 <= token_id < VOCAB_SIZE
            or len(active.token_ids) >= active.request.max_new_tokens
            or active.terminal_outcome is not None
        ):
            raise WorkerProtocolError("worker token event violates counters")
        active.token_ids.append(token_id)
        if index == 0:
            _log_lifecycle(
                "request_first_token",
                request_id=active.request_id,
                completion_id=active.request.completion_id,
                stream=active.request.stream,
                completion_tokens=1,
            )
        if token_id in EOS_TOKEN_IDS:
            active.terminal_outcome = "stop"
        elif len(active.token_ids) == active.request.max_new_tokens:
            active.terminal_outcome = "length"
        self._publish_stream_token(active, token_id)
        self._mark_progress(active)

    async def _handle_released(self, value: dict[str, Any]) -> None:
        base = {
            "schema_version",
            "type",
            "request_id",
            "outcome",
            "prompt_tokens",
            "completion_tokens",
            "reset_complete",
        }
        outcome = value.get("outcome")
        keys = base | ({"cancel_reason"} if outcome == "cancelled" else set())
        _require_exact_keys(value, keys)
        active = self._require_active_id(value.get("request_id"))
        prompt_tokens = _integer(value.get("prompt_tokens"))
        completion_tokens = _integer(value.get("completion_tokens"))
        if (
            not active.started
            or outcome not in {"stop", "length", "cancelled"}
            or value.get("reset_complete") is not True
            or prompt_tokens != len(active.request.prompt_token_ids)
            or completion_tokens != len(active.token_ids)
        ):
            raise WorkerProtocolError("worker release event violates counters")
        if outcome in {"stop", "length"} and active.terminal_outcome != outcome:
            raise WorkerProtocolError(
                "worker release outcome differs from terminal token"
            )
        if outcome == "cancelled" and active.terminal_outcome is not None:
            raise WorkerProtocolError("cancelled release followed a terminal token")
        if outcome == "cancelled":
            reason = value.get("cancel_reason")
            expected_reason = active.cancel_reason
            if expected_reason is None and self._stopping:
                expected_reason = "shutdown"
            if reason != expected_reason:
                raise WorkerProtocolError("cancel release reason differs")
        elif active.cancel_reason is not None:
            self._stale_cancel_request_id = active.request_id
        result = WorkerGenerationResult(
            request_id=active.request_id,
            outcome=outcome,
            prompt_tokens=prompt_tokens,
            token_ids=tuple(active.token_ids),
        )
        released_monotonic_ns = time.monotonic_ns()
        started_monotonic_ns = active.started_monotonic_ns
        if started_monotonic_ns is None:
            raise WorkerProtocolError("worker release lacks a start timestamp")
        await self._finish_active(active, result)
        _log_lifecycle(
            "request_released",
            observed_monotonic_ns=released_monotonic_ns,
            request_id=active.request_id,
            completion_id=active.request.completion_id,
            stream=active.request.stream,
            outcome=outcome,
            cancel_reason=active.cancel_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reset_complete=True,
            admit_to_start_ns=(started_monotonic_ns - active.admitted_monotonic_ns),
            start_to_release_ns=released_monotonic_ns - started_monotonic_ns,
            admit_to_release_ns=(released_monotonic_ns - active.admitted_monotonic_ns),
        )

    async def _finish_active(
        self, active: _ActiveRequest, result: WorkerGenerationResult
    ) -> None:
        async with self._state_lock:
            if self._active is not active:
                raise WorkerProtocolError("worker release does not own the active slot")
            self._active = None
        for task in (
            active.total_watchdog,
            active.progress_watchdog,
            active.cancel_watchdog,
        ):
            if task is not None:
                task.cancel()
        if not active.future.done():
            active.future.set_result(result)

    def _require_active_id(self, request_id: Any) -> _ActiveRequest:
        if not isinstance(request_id, str):
            raise WorkerProtocolError("worker request ID is invalid")
        active = self._active
        if active is None or active.request_id != request_id:
            raise WorkerProtocolError("worker event request ID is not active")
        return active

    def _mark_progress(self, active: _ActiveRequest) -> None:
        active.last_progress = asyncio.get_running_loop().time()
        active.progress_wakeup.set()

    def _publish_stream_token(self, active: _ActiveRequest, token_id: int) -> None:
        stream = active.stream_state
        if stream is None or stream.aborted_reason is not None:
            return
        try:
            stream.token_queue.put_nowait(token_id)
        except asyncio.QueueFull:
            stream.abort("slow_client")
            self._track(
                self._cancel_slow_client(active),
                f"ullm-request-{active.generation}-slow-client",
            )

    async def _cancel_slow_client(self, active: _ActiveRequest) -> None:
        handle = GenerationHandle(
            active.request_id,
            active.generation,
            active.future,
            active.started_future,
            active.stream_state,
        )
        try:
            await self.cancel(handle, "slow_client")
        except WorkerError:
            return

    def _matches_active(self, request_id: str, generation: int) -> bool:
        return self._matching_active(request_id, generation) is not None

    def _matching_active(
        self, request_id: str, generation: int
    ) -> _ActiveRequest | None:
        active = self._active
        if (
            active is not None
            and active.request_id == request_id
            and active.generation == generation
        ):
            return active
        return None

    async def _fatal(self, reason: str) -> None:
        active: _ActiveRequest | None = None
        async with self._fatal_lock:
            if self._fatal_started:
                return
            self._fatal_reserved = True
            self._fatal_started = True
            self._fatal_event.set()
            self._state = "failed"
            ready = self._ready_future
            if ready is not None and not ready.done():
                ready.set_exception(WorkerFatal("worker failed before readiness"))
            active = self._active
            now_monotonic_ns = time.monotonic_ns()
            _log_lifecycle(
                "worker_fatal",
                request_id=active.request_id if active is not None else None,
                completion_id=(
                    active.request.completion_id if active is not None else None
                ),
                reason=reason,
                admit_to_fatal_ns=(
                    now_monotonic_ns - active.admitted_monotonic_ns
                    if active is not None
                    else None
                ),
            )
            if active is not None and not active.future.done():
                active.future.set_exception(WorkerFatal("resident worker failed"))
            if active is not None and not active.started_future.done():
                active.started_future.set_exception(
                    WorkerFatal("resident worker failed before start")
                )
        try:
            await asyncio.wait_for(
                self._fatal_response_attempted.wait(),
                FATAL_RESPONSE_ATTEMPT_SECONDS,
            )
        except asyncio.TimeoutError:
            pass
        try:
            await self._terminate_worker()
        finally:
            self._fatal_exit(1)

    def _reserve_fatal(self) -> bool:
        if self._fatal_reserved or self._fatal_started:
            return False
        self._fatal_reserved = True
        self._state = "failed"
        self._fatal_event.set()
        return True

    async def _terminate_worker(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        self._signal_worker(process, signal.SIGTERM)
        try:
            await asyncio.wait_for(
                asyncio.shield(process.wait()),
                self._config.terminate_grace_seconds,
            )
            return
        except asyncio.TimeoutError:
            pass
        self._signal_worker(process, signal.SIGKILL)
        try:
            await asyncio.wait_for(
                asyncio.shield(process.wait()),
                KILL_WAIT_SECONDS,
            )
        except asyncio.TimeoutError as error:
            raise WorkerFatal("worker did not exit after SIGKILL") from error

    @staticmethod
    def _signal_worker(
        process: asyncio.subprocess.Process, requested_signal: signal.Signals
    ) -> None:
        try:
            os.killpg(process.pid, requested_signal)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
        try:
            process.send_signal(requested_signal)
        except ProcessLookupError:
            return
        except OSError as error:
            raise WorkerFatal("failed to signal resident worker") from error

    def _close_worker_stdin(self) -> None:
        process = self._process
        if process is None or process.stdin is None:
            return
        try:
            process.stdin.close()
        except OSError:
            pass

    def _release_singleton_lock(self) -> None:
        if self._lock_descriptor is not None:
            os.close(self._lock_descriptor)
            self._lock_descriptor = None


def _log_lifecycle(event: str, **fields: Any) -> None:
    value = {
        "schema_version": LIFECYCLE_LOG_SCHEMA,
        "event": event,
        "observed_monotonic_ns": time.monotonic_ns(),
        **fields,
    }
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    _LIFECYCLE_LOGGER.info("%s", payload)
    _notify_lifecycle_observer(payload.encode("ascii"))


def _notify_lifecycle_observer(payload: bytes) -> None:
    """Best-effort low-latency mirror; the systemd journal remains authoritative."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as observer:
            observer.setblocking(False)
            observer.sendto(payload, os.fspath(LIFECYCLE_OBSERVER_SOCKET))
    except OSError:
        # No observer is the normal product state and must not affect inference.
        return


def _acquire_singleton_lock(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise WorkerError("failed to open the GPU singleton lock") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise WorkerError("GPU singleton lock is not a regular file")
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
            raise WorkerError("GPU singleton lock ownership or mode is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise WorkerBusy("another process owns the GPU singleton lock") from error
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _validate_generation_request(request: WorkerGenerationRequest) -> None:
    if (
        not request.prompt_token_ids
        or len(request.prompt_token_ids) + request.max_new_tokens > CONTEXT_LENGTH
        or not 1 <= request.max_new_tokens <= MAX_NEW_TOKENS
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 0 <= item < VOCAB_SIZE
            for item in request.prompt_token_ids
        )
        or isinstance(request.temperature, bool)
        or not isinstance(request.temperature, (int, float))
        or not 0.0 <= float(request.temperature) <= 2.0
        or isinstance(request.top_p, bool)
        or not isinstance(request.top_p, (int, float))
        or not 0.0 < float(request.top_p) <= 1.0
        or isinstance(request.seed, bool)
        or not isinstance(request.seed, int)
        or not -(2**63) <= request.seed < 2**63
        or not isinstance(request.stream, bool)
        or (
            request.completion_id is not None
            and (
                not isinstance(request.completion_id, str)
                or len(request.completion_id) != 41
                or not request.completion_id.startswith("chatcmpl-")
                or any(
                    character not in "0123456789abcdef"
                    for character in request.completion_id[9:]
                )
            )
        )
    ):
        raise WorkerProtocolError("generation request violates worker limits")


def _decode_event(raw: bytes) -> dict[str, Any]:
    if raw.endswith(b"\r"):
        raise WorkerProtocolError("worker event uses noncanonical CRLF framing")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise WorkerProtocolError("worker event is not valid UTF-8") from error

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise WorkerProtocolError("worker event contains a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                WorkerProtocolError("worker event contains a non-finite number")
            ),
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as error:
        raise WorkerProtocolError("worker event is not valid JSON") from error
    if not isinstance(value, dict):
        raise WorkerProtocolError("worker event root is not an object")
    return value


def _require_exact_keys(value: dict[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise WorkerProtocolError("worker event field set differs")


def _integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkerProtocolError("worker event counter is invalid")
    return value


def _validate_error_event(value: dict[str, Any]) -> None:
    _require_exact_keys(
        value,
        {"schema_version", "type", "request_id", "code", "recoverable", "message"},
    )
    if value.get("request_id") is not None and not isinstance(
        value.get("request_id"), str
    ):
        raise WorkerProtocolError("worker error request ID is invalid")
    if not isinstance(value.get("code"), str) or not isinstance(
        value.get("recoverable"), bool
    ):
        raise WorkerProtocolError("worker error metadata is invalid")
    message = value.get("message")
    if not isinstance(message, str) or len(message.encode("utf-8")) > 1_024:
        raise WorkerProtocolError("worker error message is invalid")
