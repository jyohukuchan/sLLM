"""FastAPI application for the frozen non-streaming OpenAI subset."""

from __future__ import annotations

import asyncio
import hmac
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHttpException

from .errors import (
    ApiError,
    context_length_exceeded,
    invalid_request,
    unsupported_parameter,
)
from .schemas import (
    CONTEXT_LENGTH,
    MODEL_ID,
    NormalizedMessage,
    decode_json_object,
    normalize_chat_request,
)
from .settings import GatewaySettings, read_api_key
from .tokenizer import FrozenQwen3Tokenizer, TokenizedPrompt, TokenizerError
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
        loaded_worker = worker or WorkerSupervisor(
            WorkerConfig.from_settings(configured)
        )
        tokenizer_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ullm-tokenizer",
        )
        app.state.api_key = loaded_key
        app.state.tokenizer = loaded_tokenizer
        app.state.tokenizer_lock = asyncio.Lock()
        app.state.tokenizer_executor = tokenizer_executor
        app.state.worker = loaded_worker
        await loaded_worker.launch()
        try:
            yield
        finally:
            await loaded_worker.shutdown()
            tokenizer_executor.shutdown(wait=True, cancel_futures=True)

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
        return JSONResponse(
            status_code=error.status_code,
            content=error.envelope(),
            headers=error.headers,
        )

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
        if normalized.stream:
            raise unsupported_parameter("stream")
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

        worker_request = WorkerGenerationRequest(
            prompt_token_ids=prompt.token_ids,
            max_new_tokens=normalized.max_completion_tokens,
            temperature=normalized.temperature,
            top_p=normalized.top_p,
            seed=normalized.seed,
        )
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        try:
            handle = await request.app.state.worker.admit(worker_request)
        except WorkerNotReady as error:
            raise ApiError(
                503,
                "server_error",
                "model_not_ready",
                "The model is not ready.",
            ) from error
        except WorkerBusy as error:
            raise ApiError(
                429,
                "rate_limit_error",
                "request_busy",
                "The model is serving another request.",
                headers={"Retry-After": "1"},
            ) from error
        except WorkerFatal:
            return _fatal_worker_response(request)

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
            content = await _decode_completion(request, result.token_ids)
        except TokenizerError as error:
            raise ApiError(
                500,
                "server_error",
                "internal_error",
                "The generation failed.",
            ) from error
        completion_tokens = len(result.token_ids)
        return JSONResponse(
            {
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
        )

    return app


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


def _fatal_worker_response(request: Request) -> JSONResponse:
    worker = cast(WorkerSupervisor, request.app.state.worker)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "The generation failed.",
                "type": "server_error",
                "param": None,
                "code": "internal_error",
            }
        },
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
    request: Request, generation: asyncio.Task[WorkerGenerationResult]
) -> bool:
    while not generation.done():
        if await request.is_disconnected():
            return True
        await asyncio.sleep(0.05)
    return False
