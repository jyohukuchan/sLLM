"""FastAPI application for the frozen OpenAI Chat Completions subset."""

from __future__ import annotations

import asyncio
import hmac
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, cast

from anyio import CancelScope
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHttpException
from starlette.types import Receive, Scope, Send

from .errors import (
    ApiError,
    context_length_exceeded,
    invalid_request,
)
from .schemas import (
    CONTEXT_LENGTH,
    MODEL_ID,
    NormalizedChatRequest,
    NormalizedMessage,
    decode_json_object,
    normalize_chat_request,
)
from .settings import GatewaySettings, read_api_key
from .tokenizer import (
    FrozenQwen3Tokenizer,
    StableIncrementalDecoder,
    TokenizedPrompt,
    TokenizerError,
)
from .worker import (
    GenerationHandle,
    WorkerBusy,
    WorkerFatal,
    WorkerGenerationRequest,
    WorkerGenerationResult,
    WorkerNotReady,
    WorkerSupervisor,
    WorkerConfig,
)


BODY_LIMIT_BYTES = 2_097_152
STREAM_SEND_TIMEOUT_SECONDS = 5.0
FATAL_SEND_TIMEOUT_SECONDS = 0.20


class _StreamAborted(Exception):
    pass


class _TerminalSendInterrupted(Exception):
    pass


class _RequestGate:
    def __init__(self) -> None:
        self._claimed = False

    def claim(self) -> _RequestLease | None:
        if self._claimed:
            return None
        self._claimed = True
        return _RequestLease(self)

    def _release(self) -> None:
        self._claimed = False


class _RequestLease:
    def __init__(self, gate: _RequestGate) -> None:
        self._gate = gate
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._gate._release()


class _LifecycleResponse(Response):
    def __init__(self, inner: Response, lease: _RequestLease) -> None:
        self._inner = inner
        self._lease = lease
        self.status_code = inner.status_code
        self.media_type = inner.media_type
        self.background = None
        self.raw_headers = inner.raw_headers
        self.body = getattr(inner, "body", b"")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await self._inner(scope, receive, send)
        finally:
            self._lease.release()


def create_app(
    settings: GatewaySettings | None = None,
    *,
    tokenizer: Any | None = None,
    worker: Any | None = None,
    api_key: bytes | None = None,
) -> FastAPI:
    configured = settings or GatewaySettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if tokenizer is None or worker is None:
            configured.validate_paths()
        loaded_key = (
            api_key if api_key is not None else read_api_key(configured.api_key_file)
        )
        if not loaded_key:
            raise RuntimeError("configured API key is empty")
        loaded_tokenizer = tokenizer or FrozenQwen3Tokenizer.load(
            configured.tokenizer_dir
        )
        stream_tokenizer = tokenizer or FrozenQwen3Tokenizer.load(
            configured.tokenizer_dir
        )
        loaded_worker = worker or WorkerSupervisor(
            WorkerConfig.from_settings(configured)
        )
        tokenizer_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ullm-tokenizer",
        )
        stream_tokenizer_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ullm-stream-tokenizer",
        )
        app.state.api_key = loaded_key
        app.state.tokenizer = loaded_tokenizer
        app.state.tokenizer_lock = asyncio.Lock()
        app.state.tokenizer_executor = tokenizer_executor
        app.state.stream_tokenizer = stream_tokenizer
        app.state.stream_tokenizer_lock = asyncio.Lock()
        app.state.stream_tokenizer_executor = stream_tokenizer_executor
        app.state.worker = loaded_worker
        app.state.request_gate = _RequestGate()
        await loaded_worker.launch()
        try:
            yield
        finally:
            await loaded_worker.shutdown()
            tokenizer_executor.shutdown(wait=True, cancel_futures=True)
            stream_tokenizer_executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(
        title="uLLM OpenAI Gateway",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
        redirect_slashes=False,
    )

    @app.exception_handler(ApiError)
    async def api_error_handler(_: Request, error: ApiError) -> JSONResponse:
        return _api_error_response(error)

    @app.exception_handler(StarletteHttpException)
    async def route_error_handler(
        _: Request, error: StarletteHttpException
    ) -> JSONResponse:
        status = error.status_code if error.status_code in {404, 405} else 400
        return JSONResponse(
            status_code=status,
            content=invalid_request(
                "The requested method or path is not supported."
            ).envelope(),
        )

    @app.exception_handler(Exception)
    async def internal_error_handler(_: Request, __: Exception) -> JSONResponse:
        return _internal_error_response()

    @app.get("/healthz")
    async def health(request: Request) -> JSONResponse:
        _reject_query(request)
        return JSONResponse({"status": "ok"})

    @app.get("/readyz")
    async def ready(request: Request) -> JSONResponse:
        _reject_query(request)
        if request.app.state.worker.ready:
            return JSONResponse({"status": "ready"})
        return JSONResponse({"status": "not_ready"}, status_code=503)

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        _authenticate(request)
        _reject_query(request)
        return JSONResponse(
            {
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "ullm"}],
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        _authenticate(request)
        _reject_query(request)
        raw = await _read_json_body(request)
        normalized = normalize_chat_request(decode_json_object(raw))
        return await _serve_chat_completion(request, normalized)

    return app


async def _serve_chat_completion(
    request: Request,
    normalized: NormalizedChatRequest,
) -> Response:
    try:
        prompt = await _render_prompt(request, normalized.messages)
    except TokenizerError as error:
        raise ApiError(
            500,
            "server_error",
            "internal_error",
            "The request could not be processed.",
        ) from error
    if len(prompt.token_ids) + normalized.max_completion_tokens > CONTEXT_LENGTH:
        raise context_length_exceeded()

    worker = cast(WorkerSupervisor, request.app.state.worker)
    if not worker.ready:
        raise _model_not_ready_error()
    gate = cast(_RequestGate, request.app.state.request_gate)
    lease = gate.claim()
    if lease is None:
        raise _worker_busy_error()
    try:
        response = await _serve_claimed_chat_completion(request, normalized, prompt)
    except WorkerFatal:
        response = _fatal_worker_response(request)
    except ApiError as error:
        response = _api_error_response(error)
    except Exception:
        response = _internal_error_response()
    except BaseException:
        lease.release()
        raise
    return _LifecycleResponse(response, lease)


async def _serve_claimed_chat_completion(
    request: Request,
    normalized: NormalizedChatRequest,
    prompt: TokenizedPrompt,
) -> Response:
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    worker_request = WorkerGenerationRequest(
        prompt_token_ids=prompt.token_ids,
        max_new_tokens=normalized.max_completion_tokens,
        temperature=normalized.temperature,
        top_p=normalized.top_p,
        seed=normalized.seed,
        stream=normalized.stream,
        completion_id=completion_id,
    )
    try:
        handle = await request.app.state.worker.admit(worker_request)
    except WorkerNotReady as error:
        raise _model_not_ready_error() from error
    except WorkerBusy as error:
        raise _worker_busy_error() from error
    except WorkerFatal:
        return _fatal_worker_response(request)

    if normalized.stream:
        worker = cast(WorkerSupervisor, request.app.state.worker)
        try:
            started = await _wait_for_stream_start(request, worker, handle)
        except WorkerFatal:
            return _fatal_worker_response(request)
        if not started:
            return Response(status_code=499)
        response = _WorkerStreamingResponse(
            _stream_completion(
                request,
                handle,
                completion_id,
                created,
                normalized.include_usage,
            ),
            worker=worker,
            handle=handle,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        return response

    try:
        result = await _wait_for_nonstream_result(request, handle)
    except WorkerFatal:
        return _fatal_worker_response(request)
    if result is None:
        return Response(status_code=499)
    if result.outcome not in {"stop", "length"}:
        raise ApiError(
            500,
            "server_error",
            "internal_error",
            "The generation failed.",
        )
    try:
        content = await _decode_completion_while_healthy(request, result.token_ids)
    except WorkerFatal:
        return _fatal_worker_response(request)
    except TokenizerError as error:
        raise ApiError(
            500,
            "server_error",
            "internal_error",
            "The generation failed.",
        ) from error
    completion_tokens = len(result.token_ids)
    value = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "logprobs": None,
                "finish_reason": result.outcome,
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": result.prompt_tokens + completion_tokens,
        },
    }
    timings = _completion_timings(result)
    if timings is not None:
        value["timings"] = timings
    return JSONResponse(value)


def _worker_busy_error() -> ApiError:
    return ApiError(
        429,
        "rate_limit_error",
        "request_busy",
        "The model is serving another request.",
        headers={"Retry-After": "1"},
    )


def _model_not_ready_error() -> ApiError:
    return ApiError(
        503,
        "server_error",
        "model_not_ready",
        "The model is not ready.",
    )


def _api_error_response(error: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content=error.envelope(),
        headers=error.headers,
    )


def _internal_error_response() -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "The request could not be processed.",
                "type": "server_error",
                "param": None,
                "code": "internal_error",
            }
        },
    )


def _reject_query(request: Request) -> None:
    if request.scope.get("query_string", b""):
        raise invalid_request("Query parameters are not supported.")


def _authenticate(request: Request) -> None:
    values = [
        value
        for name, value in request.scope.get("headers", [])
        if name.lower() == b"authorization"
    ]
    valid = False
    if len(values) == 1:
        value = values[0]
        if len(value) > 7 and value[:6].lower() == b"bearer" and value[6:7] == b" ":
            valid = hmac.compare_digest(value[7:], request.app.state.api_key)
    if not valid:
        raise ApiError(
            401,
            "invalid_request_error",
            "invalid_api_key",
            "The supplied API key is invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def _read_json_body(request: Request) -> bytes:
    content_types = [
        value
        for name, value in request.scope.get("headers", [])
        if name.lower() == b"content-type"
    ]
    if len(content_types) != 1:
        raise invalid_request("Content-Type must be application/json.")
    media_type = content_types[0].split(b";", 1)[0].strip().lower()
    if media_type != b"application/json":
        raise invalid_request("Content-Type must be application/json.")

    body = bytearray()
    async for chunk in request.stream():
        remaining = BODY_LIMIT_BYTES - len(body)
        if len(chunk) > remaining:
            raise invalid_request("The request body exceeds 2 MiB.")
        body.extend(chunk)
    return bytes(body)


async def _render_prompt(
    request: Request, messages: tuple[NormalizedMessage, ...]
) -> TokenizedPrompt:
    lock = cast(asyncio.Lock, request.app.state.tokenizer_lock)
    executor = cast(ThreadPoolExecutor, request.app.state.tokenizer_executor)
    tokenizer = request.app.state.tokenizer
    async with lock:
        return cast(
            TokenizedPrompt,
            await asyncio.get_running_loop().run_in_executor(
                executor, tokenizer.render, messages
            ),
        )


async def _decode_completion(request: Request, token_ids: tuple[int, ...]) -> str:
    lock = cast(asyncio.Lock, request.app.state.tokenizer_lock)
    executor = cast(ThreadPoolExecutor, request.app.state.tokenizer_executor)
    tokenizer = request.app.state.tokenizer
    async with lock:
        return cast(
            str,
            await asyncio.get_running_loop().run_in_executor(
                executor, tokenizer.decode, token_ids
            ),
        )


async def _decode_completion_while_healthy(
    request: Request, token_ids: tuple[int, ...]
) -> str:
    worker = cast(WorkerSupervisor, request.app.state.worker)
    decode_task = asyncio.create_task(_decode_completion(request, token_ids))
    fatal_task = asyncio.create_task(worker.wait_fatal())
    try:
        done, _ = await asyncio.wait(
            {decode_task, fatal_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if fatal_task in done:
            raise WorkerFatal("resident worker failed before JSON response")
        return decode_task.result()
    finally:
        if not decode_task.done():
            decode_task.cancel()
        fatal_task.cancel()
        with CancelScope(shield=True):
            await asyncio.gather(decode_task, fatal_task, return_exceptions=True)


async def _decode_stream_token(
    request: Request, decoder: StableIncrementalDecoder, token_id: int
) -> str:
    lock = cast(asyncio.Lock, request.app.state.stream_tokenizer_lock)
    executor = cast(ThreadPoolExecutor, request.app.state.stream_tokenizer_executor)
    async with lock:
        return await asyncio.get_running_loop().run_in_executor(
            executor, decoder.push, token_id
        )


async def _finish_stream_decode(
    request: Request, decoder: StableIncrementalDecoder
) -> str:
    lock = cast(asyncio.Lock, request.app.state.stream_tokenizer_lock)
    executor = cast(ThreadPoolExecutor, request.app.state.stream_tokenizer_executor)
    async with lock:
        return await asyncio.get_running_loop().run_in_executor(
            executor, decoder.finish
        )


class _WorkerStreamingResponse(StreamingResponse):
    def __init__(
        self,
        content: AsyncIterator[bytes],
        *,
        worker: WorkerSupervisor,
        handle: GenerationHandle,
        media_type: str,
        headers: dict[str, str],
    ) -> None:
        super().__init__(content, media_type=media_type, headers=headers)
        self._worker = worker
        self._handle = handle

    async def stream_response(self, send: Send) -> None:
        stream = self._handle.stream_state
        if stream is None:
            raise RuntimeError("streaming response has no worker stream state")
        iterator = self.body_iterator.__aiter__()
        fatal_task = asyncio.create_task(self._worker.wait_fatal())
        abort_task = asyncio.create_task(stream.aborted.wait())
        committed = False
        terminal_sent = False
        try:
            self._raise_preheader_failure()
            first = cast(
                bytes,
                await _wait_stream_operation(
                    iterator.__anext__(), fatal_task, abort_task
                ),
            )
            self._raise_preheader_failure()
            await _wait_stream_operation(
                send(
                    {
                        "type": "http.response.start",
                        "status": self.status_code,
                        "headers": self.raw_headers,
                    }
                ),
                fatal_task,
                abort_task,
                timeout=STREAM_SEND_TIMEOUT_SECONDS,
                prefer_operation=True,
            )
            committed = True
            await _wait_stream_operation(
                send(
                    {
                        "type": "http.response.body",
                        "body": first,
                        "more_body": True,
                    }
                ),
                fatal_task,
                abort_task,
                timeout=STREAM_SEND_TIMEOUT_SECONDS,
            )
            while True:
                try:
                    chunk = cast(
                        bytes,
                        await _wait_stream_operation(
                            iterator.__anext__(), fatal_task, abort_task
                        ),
                    )
                except StopAsyncIteration:
                    break
                await _wait_stream_operation(
                    send(
                        {
                            "type": "http.response.body",
                            "body": chunk,
                            "more_body": True,
                        }
                    ),
                    fatal_task,
                    abort_task,
                    timeout=STREAM_SEND_TIMEOUT_SECONDS,
                    prefer_operation=chunk == b"data: [DONE]\n\n",
                    terminal_operation=chunk == b"data: [DONE]\n\n",
                )
                if chunk == b"data: [DONE]\n\n":
                    terminal_sent = True
                    break
            if not terminal_sent:
                self._worker.request_fatal("SSE stream ended without a terminal marker")
                raise WorkerFatal("SSE stream ended without a terminal marker")
            await _send_with_timeout(
                send,
                {"type": "http.response.body", "body": b"", "more_body": False},
                STREAM_SEND_TIMEOUT_SECONDS,
            )
        except WorkerFatal:
            if committed:
                await _attempt_postheader_error(send, self._worker)
            else:
                await _attempt_preheader_error(send, self._worker)
        except _TerminalSendInterrupted:
            try:
                await _attempt_stream_close(send)
            finally:
                await self._worker.acknowledge_fatal_response()
        except _StreamAborted:
            if not committed:
                await _attempt_preheader_error(
                    send, self._worker, acknowledge_fatal=False
                )
            else:
                await _attempt_stream_close(send)
        finally:
            fatal_task.cancel()
            abort_task.cancel()
            with CancelScope(shield=True):
                await asyncio.gather(fatal_task, abort_task, return_exceptions=True)
                closer = getattr(iterator, "aclose", None)
                if closer is not None:
                    await closer()

    def _raise_preheader_failure(self) -> None:
        future = self._handle._future
        if future.done():
            if future.cancelled():
                raise WorkerFatal("generation was cancelled before SSE headers")
            exception = future.exception()
            if exception is not None:
                if isinstance(exception, WorkerFatal):
                    raise exception
                raise WorkerFatal("generation failed before SSE headers") from exception
        if self._worker.failed:
            if not future.done():
                future.add_done_callback(_consume_future_exception)
            raise WorkerFatal("resident worker failed before SSE headers")


def _consume_future_exception(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


async def _wait_stream_operation(
    operation: Any,
    fatal_task: asyncio.Task[None],
    abort_task: asyncio.Task[bool],
    *,
    timeout: float | None = None,
    prefer_operation: bool = False,
    terminal_operation: bool = False,
) -> Any:
    operation_task = asyncio.ensure_future(operation)
    try:
        done, _ = await asyncio.wait(
            {operation_task, fatal_task, abort_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if prefer_operation and operation_task in done:
            try:
                return operation_task.result()
            except (OSError, ConnectionError) as error:
                raise _StreamAborted from error
        if fatal_task in done:
            if terminal_operation:
                raise _TerminalSendInterrupted
            raise WorkerFatal("resident worker failed during SSE response")
        if abort_task in done or operation_task not in done:
            raise _StreamAborted
        try:
            return operation_task.result()
        except (OSError, ConnectionError) as error:
            raise _StreamAborted from error
    finally:
        if not operation_task.done():
            operation_task.cancel()
        with CancelScope(shield=True):
            await asyncio.gather(operation_task, return_exceptions=True)


async def _send_with_timeout(
    send: Send, message: dict[str, Any], timeout: float
) -> None:
    try:
        await asyncio.wait_for(send(message), timeout)
    except (asyncio.TimeoutError, OSError, ConnectionError) as error:
        raise _StreamAborted from error


async def _attempt_preheader_error(
    send: Send,
    worker: WorkerSupervisor,
    *,
    acknowledge_fatal: bool = True,
) -> None:
    response = JSONResponse(status_code=500, content=_fatal_error_content())
    try:
        async with asyncio.timeout(FATAL_SEND_TIMEOUT_SECONDS):
            await send(
                {
                    "type": "http.response.start",
                    "status": response.status_code,
                    "headers": response.raw_headers,
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": response.body,
                    "more_body": False,
                }
            )
    except (TimeoutError, OSError, ConnectionError):
        pass
    finally:
        if acknowledge_fatal:
            await worker.acknowledge_fatal_response()


async def _attempt_postheader_error(send: Send, worker: WorkerSupervisor) -> None:
    try:
        async with asyncio.timeout(FATAL_SEND_TIMEOUT_SECONDS):
            await send(
                {
                    "type": "http.response.body",
                    "body": _stream_error_record(),
                    "more_body": False,
                }
            )
    except (TimeoutError, OSError, ConnectionError):
        pass
    finally:
        await worker.acknowledge_fatal_response()


async def _attempt_stream_close(send: Send) -> None:
    try:
        async with asyncio.timeout(FATAL_SEND_TIMEOUT_SECONDS):
            await send({"type": "http.response.body", "body": b"", "more_body": False})
    except (TimeoutError, OSError, ConnectionError):
        pass


async def _stream_completion(
    request: Request,
    handle: GenerationHandle,
    completion_id: str,
    created: int,
    include_usage: bool,
) -> AsyncIterator[bytes]:
    worker = cast(WorkerSupervisor, request.app.state.worker)
    stream = handle.stream_state
    if stream is None:
        raise RuntimeError("streaming request has no worker stream state")
    tokenizer = request.app.state.stream_tokenizer
    decoder = StableIncrementalDecoder(tokenizer)
    result_task: asyncio.Task[WorkerGenerationResult] = asyncio.create_task(
        worker.wait(handle)
    )
    queue_task: asyncio.Task[int] | None = None
    try:
        yield _sse_record(_role_chunk(completion_id, created))
        while True:
            if stream.aborted_reason is not None:
                try:
                    await asyncio.shield(result_task)
                except WorkerFatal:
                    pass
                return

            if result_task.done():
                break

            try:
                token_id = stream.token_queue.get_nowait()
            except asyncio.QueueEmpty:
                queue_task = asyncio.create_task(stream.token_queue.get())
                done, _ = await asyncio.wait(
                    {queue_task, result_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if queue_task in done:
                    token_id = queue_task.result()
                    queue_task = None
                else:
                    queue_task.cancel()
                    await asyncio.gather(queue_task, return_exceptions=True)
                    queue_task = None
                    break

            suffix = await _decode_stream_token(request, decoder, token_id)
            if stream.aborted_reason is not None:
                continue
            if suffix:
                yield _sse_record(_content_chunk(completion_id, created, suffix))

        result = result_task.result()
        if stream.aborted_reason is not None or result.outcome == "cancelled":
            return
        while True:
            try:
                token_id = stream.token_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            suffix = await _decode_stream_token(request, decoder, token_id)
            if suffix:
                yield _sse_record(_content_chunk(completion_id, created, suffix))
        suffix = await _finish_stream_decode(request, decoder)
        if suffix:
            yield _sse_record(_content_chunk(completion_id, created, suffix))
        timings = _completion_timings(result)
        final_chunk = _final_chunk(completion_id, created, result.outcome)
        if timings is not None and not include_usage:
            final_chunk["timings"] = timings
        yield _sse_record(final_chunk)
        if include_usage:
            yield _sse_record(
                _usage_chunk(completion_id, created, result, timings=timings)
            )
        yield b"data: [DONE]\n\n"
    except WorkerFatal:
        raise
    except TokenizerError as error:
        worker.request_fatal("incremental detokenization failed")
        raise WorkerFatal("incremental detokenization failed") from error
    except asyncio.CancelledError:
        raise
    except Exception as error:
        worker.request_fatal("stream response generation failed")
        raise WorkerFatal("stream response generation failed") from error
    finally:
        with CancelScope(shield=True):
            if queue_task is not None:
                queue_task.cancel()
                await asyncio.gather(queue_task, return_exceptions=True)
            if not handle._future.done():
                if worker.failed:
                    handle._future.add_done_callback(_consume_future_exception)
                else:
                    await _cancel_stream_and_drain(worker, handle)
            if not result_task.done():
                result_task.cancel()
            await asyncio.gather(result_task, return_exceptions=True)


async def _cancel_stream_and_drain(
    worker: WorkerSupervisor, handle: GenerationHandle
) -> None:
    stream = handle.stream_state
    reason = (
        stream.aborted_reason
        if stream is not None and stream.aborted_reason == "slow_client"
        else "client_disconnect"
    )
    with CancelScope(shield=True):
        try:
            await worker.cancel(handle, reason)
            await worker.wait(handle)
        except WorkerFatal:
            return


async def _wait_for_stream_start(
    request: Request,
    worker: WorkerSupervisor,
    handle: GenerationHandle,
) -> bool:
    started = asyncio.create_task(worker.wait_started(handle))
    disconnect = asyncio.create_task(_wait_for_disconnect(request, started))
    try:
        done, _ = await asyncio.wait(
            {started, disconnect},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnect in done and disconnect.result():
            await _cancel_stream_and_drain(worker, handle)
            return False
        if started in done:
            if await request.is_disconnected():
                await _cancel_stream_and_drain(worker, handle)
                return False
            await started
            return True
        await started
        return True
    except asyncio.CancelledError:
        await _cancel_stream_and_drain(worker, handle)
        raise
    finally:
        disconnect.cancel()
        if not started.done():
            started.cancel()
        await asyncio.gather(disconnect, started, return_exceptions=True)


def _sse_record(value: dict[str, Any]) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return b"data: " + payload + b"\n\n"


def _chunk_base(completion_id: str, created: int) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
    }


def _role_chunk(completion_id: str, created: int) -> dict[str, Any]:
    value = _chunk_base(completion_id, created)
    value["choices"] = [
        {
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "logprobs": None,
            "finish_reason": None,
        }
    ]
    return value


def _content_chunk(completion_id: str, created: int, content: str) -> dict[str, Any]:
    if not content:
        raise ValueError("SSE content chunk must be nonempty")
    value = _chunk_base(completion_id, created)
    value["choices"] = [
        {
            "index": 0,
            "delta": {"content": content},
            "logprobs": None,
            "finish_reason": None,
        }
    ]
    return value


def _final_chunk(
    completion_id: str, created: int, finish_reason: str
) -> dict[str, Any]:
    value = _chunk_base(completion_id, created)
    value["choices"] = [
        {
            "index": 0,
            "delta": {},
            "logprobs": None,
            "finish_reason": finish_reason,
        }
    ]
    return value


def _usage_chunk(
    completion_id: str,
    created: int,
    result: WorkerGenerationResult,
    *,
    timings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completion_tokens = len(result.token_ids)
    value = _chunk_base(completion_id, created)
    value["choices"] = []
    value["usage"] = {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": result.prompt_tokens + completion_tokens,
    }
    if timings is not None:
        value["timings"] = timings
    return value


def _completion_timings(result: WorkerGenerationResult) -> dict[str, Any] | None:
    timings = result.timings
    if timings is None:
        return None
    if result.outcome == "stop":
        termination_reason = "eos_token"
    elif result.prompt_tokens + len(result.token_ids) == CONTEXT_LENGTH:
        termination_reason = "context_length"
    else:
        termination_reason = "max_tokens"
    return {
        "cache_n": timings.cache_n,
        "prompt_n": timings.prompt_n,
        "prompt_ms": timings.prompt_ms,
        "prompt_per_token_ms": timings.prompt_per_token_ms,
        "prompt_per_second": timings.prompt_per_second,
        "predicted_n": timings.predicted_n,
        "predicted_ms": timings.predicted_ms,
        "predicted_per_token_ms": timings.predicted_per_token_ms,
        "predicted_per_second": timings.predicted_per_second,
        "finish_reason": result.outcome,
        "termination_reason": termination_reason,
    }


def _stream_error_record() -> bytes:
    return _sse_record(_fatal_error_content())


def _fatal_error_content() -> dict[str, Any]:
    return {
        "error": {
            "message": "The generation failed.",
            "type": "server_error",
            "param": None,
            "code": "internal_error",
        }
    }


def _fatal_worker_response(request: Request) -> JSONResponse:
    worker = cast(WorkerSupervisor, request.app.state.worker)
    return JSONResponse(
        status_code=500,
        content=_fatal_error_content(),
        background=BackgroundTask(worker.acknowledge_fatal_response),
    )


async def _wait_for_nonstream_result(
    request: Request, handle: GenerationHandle
) -> WorkerGenerationResult | None:
    worker = cast(WorkerSupervisor, request.app.state.worker)
    generation: asyncio.Task[WorkerGenerationResult] = asyncio.create_task(
        worker.wait(handle)
    )
    disconnect = asyncio.create_task(_wait_for_disconnect(request, generation))
    try:
        done, _ = await asyncio.wait(
            {generation, disconnect}, return_when=asyncio.FIRST_COMPLETED
        )
        if generation in done:
            disconnect.cancel()
            await asyncio.gather(disconnect, return_exceptions=True)
            return generation.result()
        if disconnect.result():
            await worker.cancel(handle, "client_disconnect")
            try:
                await asyncio.shield(generation)
            except WorkerFatal:
                pass
            return None
        return await generation
    except asyncio.CancelledError:
        try:
            await worker.cancel(handle, "client_disconnect")
            await asyncio.shield(generation)
        except WorkerFatal:
            pass
        raise
    finally:
        disconnect.cancel()
        await asyncio.gather(disconnect, return_exceptions=True)


async def _wait_for_disconnect(
    request: Request, generation: asyncio.Future[Any]
) -> bool:
    while not generation.done():
        if await request.is_disconnected():
            return True
        await asyncio.sleep(0.05)
    return False
