from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from ullm_openai_gateway.app import (
    _decode_stream_token,
    _render_prompt,
    _stream_completion,
    _wait_for_stream_start,
    create_app,
)
from ullm_openai_gateway.schemas import EOS_TOKEN_IDS, MODEL_ID, NormalizedMessage
from ullm_openai_gateway.settings import GatewaySettings
from ullm_openai_gateway.tokenizer import (
    StableIncrementalDecoder,
    TokenizedPrompt,
    TokenizerError,
)
from ullm_openai_gateway.worker import (
    GenerationHandle,
    WorkerBusy,
    WorkerGenerationResult,
    WorkerFatal,
    WorkerNotReady,
    WorkerStreamState,
)


API_KEY = b"test-secret"
AUTH = {"Authorization": "Bearer test-secret"}


class FakeTokenizer:
    def __init__(self, prompt_tokens: int = 3) -> None:
        self.prompt_tokens = prompt_tokens

    def render(self, _: Any) -> TokenizedPrompt:
        return TokenizedPrompt("rendered", tuple(range(self.prompt_tokens)))

    def decode(self, token_ids: Any) -> str:
        values = tuple(token_ids)
        if values == (EOS_TOKEN_IDS[0],):
            return ""
        if values == (101,):
            return "日本語"
        if values == (101, 102):
            return "日本語の長い応答"
        assert values == (101, EOS_TOKEN_IDS[0])
        return "日本語の応答"


class FakeWorker:
    def __init__(
        self,
        *,
        outcome: str = "stop",
        token_ids: tuple[int, ...] = (101, EOS_TOKEN_IDS[0]),
    ) -> None:
        self.ready = True
        self.busy = False
        self.generate_count = 0
        self.requests: list[Any] = []
        self._generation = 0
        self._failed = False
        self._fatal_event = asyncio.Event()
        self.fatal_response_ack_count = 0
        self.outcome = outcome
        self.token_ids = token_ids

    async def launch(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def admit(self, request: Any) -> GenerationHandle:
        if not self.ready:
            raise WorkerNotReady
        if self.busy:
            raise WorkerBusy
        self.generate_count += 1
        self.requests.append(request)
        self._generation += 1
        future: asyncio.Future[WorkerGenerationResult] = (
            asyncio.get_running_loop().create_future()
        )
        started: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        started.set_result(None)
        stream_state = None
        if request.stream:
            stream_state = WorkerStreamState(asyncio.Queue(maxsize=32))
            for token_id in self.token_ids:
                stream_state.token_queue.put_nowait(token_id)
        future.set_result(
            WorkerGenerationResult(
                request_id=f"req-{self._generation}",
                outcome=self.outcome,
                prompt_tokens=len(request.prompt_token_ids),
                token_ids=self.token_ids,
            )
        )
        return GenerationHandle(
            f"req-{self._generation}",
            self._generation,
            future,
            started,
            stream_state,
        )

    async def wait(self, handle: GenerationHandle) -> WorkerGenerationResult:
        return await handle._future

    async def wait_started(self, handle: GenerationHandle) -> None:
        if handle._started_future is not None:
            await handle._started_future

    @property
    def failed(self) -> bool:
        return self._failed

    async def wait_fatal(self) -> None:
        await self._fatal_event.wait()

    async def cancel(self, _: GenerationHandle, __: str) -> None:
        return None

    async def acknowledge_fatal_response(self) -> None:
        self.fatal_response_ack_count += 1

    def request_fatal(self, _: str) -> None:
        return None


class FatalFakeWorker(FakeWorker):
    def __init__(self) -> None:
        super().__init__()
        self.fatal_response_acknowledged = False

    async def admit(self, request: Any) -> GenerationHandle:
        self.generate_count += 1
        self._generation += 1
        future: asyncio.Future[WorkerGenerationResult] = (
            asyncio.get_running_loop().create_future()
        )
        future.set_exception(WorkerFatal("injected worker failure"))
        return GenerationHandle(f"req-{self._generation}", self._generation, future)

    async def acknowledge_fatal_response(self) -> None:
        await super().acknowledge_fatal_response()
        self.fatal_response_acknowledged = True


class StreamingFatalFakeWorker(FatalFakeWorker):
    async def admit(self, request: Any) -> GenerationHandle:
        self.generate_count += 1
        self._generation += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future[WorkerGenerationResult] = loop.create_future()
        started: asyncio.Future[None] = loop.create_future()
        started.set_result(None)
        stream_state = WorkerStreamState(asyncio.Queue(maxsize=32))
        stream_state.token_queue.put_nowait(101)
        future.set_exception(WorkerFatal("injected streaming failure"))
        return GenerationHandle(
            f"req-{self._generation}",
            self._generation,
            future,
            started,
            stream_state,
        )


class PostHeaderFatalFakeWorker(FatalFakeWorker):
    async def admit(self, request: Any) -> GenerationHandle:
        self.generate_count += 1
        self._generation += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future[WorkerGenerationResult] = loop.create_future()
        started: asyncio.Future[None] = loop.create_future()
        started.set_result(None)
        stream_state = WorkerStreamState(asyncio.Queue(maxsize=32))

        def fail() -> None:
            self._failed = True
            self._fatal_event.set()
            future.set_exception(WorkerFatal("injected post-header failure"))

        loop.call_later(0.01, fail)
        return GenerationHandle(
            f"req-{self._generation}",
            self._generation,
            future,
            started,
            stream_state,
        )


class ReleasedThenFatalFakeWorker(FatalFakeWorker):
    async def admit(self, request: Any) -> GenerationHandle:
        handle = await FakeWorker.admit(self, request)

        def fail() -> None:
            self._failed = True
            self._fatal_event.set()

        asyncio.get_running_loop().call_later(0.01, fail)
        return handle


class CancelableFakeWorker(FakeWorker):
    def __init__(self) -> None:
        super().__init__()
        self.handle: GenerationHandle | None = None
        self.cancel_reason: str | None = None

    async def make_handle(self, *, started_ready: bool = True) -> GenerationHandle:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[WorkerGenerationResult] = loop.create_future()
        started: asyncio.Future[None] = loop.create_future()
        if started_ready:
            started.set_result(None)
        self.handle = GenerationHandle(
            "req-cancel",
            1,
            future,
            started,
            WorkerStreamState(asyncio.Queue(maxsize=32)),
        )
        return self.handle

    async def cancel(self, handle: GenerationHandle, reason: str) -> None:
        self.cancel_reason = reason
        if not handle._future.done():
            handle._future.set_result(
                WorkerGenerationResult(
                    request_id=handle.request_id,
                    outcome="cancelled",
                    prompt_tokens=3,
                    token_ids=(),
                )
            )


class PendingStreamFakeWorker(CancelableFakeWorker):
    async def admit(self, request: Any) -> GenerationHandle:
        self.generate_count += 1
        self.requests.append(request)
        self._generation += 1
        return await self.make_handle()


class GatewayFatalPendingWorker(PendingStreamFakeWorker):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_entered = False

    async def admit(self, request: Any) -> GenerationHandle:
        handle = await super().admit(request)
        assert handle.stream_state is not None
        handle.stream_state.token_queue.put_nowait(101)
        return handle

    async def cancel(self, handle: GenerationHandle, reason: str) -> None:
        self.cancel_entered = True
        await asyncio.sleep(0.4)
        await super().cancel(handle, reason)

    def request_fatal(self, _: str) -> None:
        self._failed = True
        self._fatal_event.set()


class SlowTokenizer(FakeTokenizer):
    def __init__(self, started: threading.Event) -> None:
        super().__init__()
        self.started = started
        self.render_count = 0

    def render(self, messages: Any) -> TokenizedPrompt:
        self.render_count += 1
        self.started.set()
        time.sleep(0.2)
        return super().render(messages)


def settings(tmp_path: Path) -> GatewaySettings:
    return GatewaySettings(
        worker_binary=tmp_path / "worker",
        artifact_dir=tmp_path / "artifact",
        package_dir=tmp_path / "package",
        tokenizer_dir=tmp_path / "tokenizer",
        api_key_file=tmp_path / "key",
        gpu_lock_file=tmp_path / "lock",
    )


def body(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "こんにちは"}],
        "max_tokens": 8,
        "temperature": 0,
        "seed": 7,
    }
    value.update(updates)
    return value


def chat_scope(payload: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"authorization", b"Bearer test-secret"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(raw)).encode("ascii")),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }
    return scope, raw


@pytest.fixture
def fake_worker() -> FakeWorker:
    return FakeWorker()


@pytest.fixture
def client(tmp_path: Path, fake_worker: FakeWorker) -> TestClient:
    app = create_app(
        settings(tmp_path),
        tokenizer=FakeTokenizer(),
        worker=fake_worker,
        api_key=API_KEY,
    )
    with TestClient(app) as instance:
        yield instance


def test_health_readiness_and_exact_model_list(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}
    response = client.get("/v1/models", headers=AUTH)
    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "ullm"}],
    }


def test_authentication_precedes_body_parsing(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_nonstream_completion_has_exact_shape_and_usage(
    client: TestClient, fake_worker: FakeWorker
) -> None:
    response = client.post("/v1/chat/completions", headers=AUTH, json=body())
    assert response.status_code == 200
    value = response.json()
    assert value["id"].startswith("chatcmpl-") and len(value["id"]) == 41
    assert value["object"] == "chat.completion"
    assert value["model"] == MODEL_ID
    assert value["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "日本語の応答"},
            "logprobs": None,
            "finish_reason": "stop",
        }
    ]
    assert value["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert fake_worker.generate_count == 1
    assert fake_worker.requests[0].temperature == 0
    assert fake_worker.requests[0].seed == 7
    assert fake_worker.requests[0].completion_id == value["id"]


@pytest.mark.parametrize(
    "payload,code",
    [
        (
            {"model": MODEL_ID, "messages": [{"role": "assistant", "content": "x"}]},
            "invalid_request_error",
        ),
        (body(model="missing"), "model_not_found"),
        (body(tools=[]), "unsupported_parameter"),
    ],
)
def test_rejections_are_openai_shaped_and_do_not_mutate_worker(
    client: TestClient,
    fake_worker: FakeWorker,
    payload: dict[str, Any],
    code: str,
) -> None:
    response = client.post("/v1/chat/completions", headers=AUTH, json=payload)
    assert response.status_code in {400, 404}
    assert set(response.json()) == {"error"}
    assert response.json()["error"]["code"] == code
    assert response.status_code != 422
    assert fake_worker.generate_count == 0


def test_duplicate_json_key_is_rejected_without_worker_mutation(
    client: TestClient, fake_worker: FakeWorker
) -> None:
    raw = b'{"model":"ullm-qwen3-14b-sq8","model":"other","messages":[]}'
    response = client.post(
        "/v1/chat/completions",
        headers={**AUTH, "Content-Type": "application/json"},
        content=raw,
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request_error"
    assert fake_worker.generate_count == 0


def test_context_boundary_is_checked_before_worker_admission(
    tmp_path: Path, fake_worker: FakeWorker
) -> None:
    accepted_app = create_app(
        settings(tmp_path),
        tokenizer=FakeTokenizer(4_088),
        worker=fake_worker,
        api_key=API_KEY,
    )
    with TestClient(accepted_app) as accepted:
        assert (
            accepted.post(
                "/v1/chat/completions", headers=AUTH, json=body(max_tokens=8)
            ).status_code
            == 200
        )
    assert fake_worker.generate_count == 1

    rejected_worker = FakeWorker()
    rejected_app = create_app(
        settings(tmp_path),
        tokenizer=FakeTokenizer(4_089),
        worker=rejected_worker,
        api_key=API_KEY,
    )
    with TestClient(rejected_app) as rejected:
        response = rejected.post(
            "/v1/chat/completions", headers=AUTH, json=body(max_tokens=8)
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"
    assert rejected_worker.generate_count == 0


def test_not_ready_and_busy_mapping(
    client: TestClient, fake_worker: FakeWorker
) -> None:
    fake_worker.ready = False
    response = client.post("/v1/chat/completions", headers=AUTH, json=body())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_ready"
    fake_worker.ready = True
    fake_worker.busy = True
    response = client.post("/v1/chat/completions", headers=AUTH, json=body())
    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"
    assert response.json()["error"]["code"] == "request_busy"
    assert client.get("/readyz").status_code == 200


def test_invalid_request_precedes_readiness_and_busy(
    client: TestClient, fake_worker: FakeWorker
) -> None:
    fake_worker.ready = False
    response = client.post(
        "/v1/chat/completions", headers=AUTH, json=body(model="missing")
    )
    assert response.status_code == 404
    fake_worker.ready = True
    fake_worker.busy = True
    response = client.post(
        "/v1/chat/completions", headers=AUTH, json=body(max_tokens=0)
    )
    assert response.status_code == 400
    assert fake_worker.generate_count == 0


def test_query_and_content_type_are_rejected(client: TestClient) -> None:
    assert client.get("/healthz?x=1").status_code == 400
    response = client.post(
        "/v1/chat/completions",
        headers={**AUTH, "Content-Type": "text/plain"},
        content=json.dumps(body()),
    )
    assert response.status_code == 400


def test_duplicate_authorization_and_unknown_route_are_shaped(
    client: TestClient,
) -> None:
    response = client.get(
        "/v1/models",
        headers=[
            ("Authorization", "Bearer test-secret"),
            ("Authorization", "Bearer test-secret"),
        ],
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"
    response = client.get("/v1/unknown")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "invalid_request_error"


def test_actual_openwebui_nonstream_fixture_is_accepted(
    client: TestClient, fake_worker: FakeWorker
) -> None:
    fixture_path = (
        Path(__file__).resolve().parents[3]
        / "tests/fixtures/sq8-serving-v0.1/openwebui/nonstream-request.json"
    )
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    response = client.post("/v1/chat/completions", headers=AUTH, json=payload)
    assert response.status_code == 200
    assert fake_worker.generate_count == 1


def test_oversized_body_is_rejected_without_worker_mutation(
    client: TestClient, fake_worker: FakeWorker
) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers={**AUTH, "Content-Type": "application/json"},
        content=b"{" + b" " * 2_097_152 + b"}",
    )
    assert response.status_code == 400
    assert fake_worker.generate_count == 0


def test_trailing_slash_does_not_redirect_or_bypass_auth(client: TestClient) -> None:
    response = client.get("/healthz/", follow_redirects=False)
    assert response.status_code == 404
    response = client.get("/v1/models/", follow_redirects=False)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "invalid_request_error"


def test_tokenizer_work_does_not_starve_health_endpoint(tmp_path: Path) -> None:
    started = threading.Event()
    fake_worker = FakeWorker()
    app = create_app(
        settings(tmp_path),
        tokenizer=SlowTokenizer(started),
        worker=fake_worker,
        api_key=API_KEY,
    )
    with TestClient(app) as instance:
        responses: list[int] = []
        thread = threading.Thread(
            target=lambda: responses.append(
                instance.post(
                    "/v1/chat/completions", headers=AUTH, json=body()
                ).status_code
            )
        )
        thread.start()
        assert started.wait(timeout=1.0)
        before = time.monotonic()
        assert instance.get("/healthz").status_code == 200
        assert time.monotonic() - before < 0.1
        thread.join(timeout=1.0)
        assert responses == [200]


def test_stream_decode_executor_isolated_from_prompt_render(tmp_path: Path) -> None:
    class BlockingRenderTokenizer(FakeTokenizer):
        def __init__(self) -> None:
            super().__init__()
            self.render_started = threading.Event()
            self.render_release = threading.Event()

        def render(self, messages: Any) -> TokenizedPrompt:
            self.render_started.set()
            assert self.render_release.wait(timeout=1.0)
            return super().render(messages)

    async def scenario() -> None:
        tokenizer = BlockingRenderTokenizer()
        app = create_app(
            settings(tmp_path),
            tokenizer=tokenizer,
            worker=FakeWorker(),
            api_key=API_KEY,
        )
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [],
                "query_string": b"",
                "app": app,
            }
        )
        async with app.router.lifespan_context(app):
            render = asyncio.create_task(
                _render_prompt(request, (NormalizedMessage("user", "hello"),))
            )
            assert await asyncio.to_thread(tokenizer.render_started.wait, 1.0)
            decoder = StableIncrementalDecoder(tokenizer)  # type: ignore[arg-type]
            suffix = await asyncio.wait_for(
                _decode_stream_token(request, decoder, 101), timeout=0.1
            )
            assert suffix == "日本語"
            tokenizer.render_release.set()
            await render

    asyncio.run(scenario())


def test_fatal_worker_response_is_attempted_before_ack(tmp_path: Path) -> None:
    fake_worker = FatalFakeWorker()
    app = create_app(
        settings(tmp_path),
        tokenizer=FakeTokenizer(),
        worker=fake_worker,
        api_key=API_KEY,
    )
    with TestClient(app) as instance:
        response = instance.post("/v1/chat/completions", headers=AUTH, json=body())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert fake_worker.fatal_response_acknowledged is True


def test_nonstream_release_then_fatal_beats_slow_decode(tmp_path: Path) -> None:
    class SlowDecodeTokenizer(FakeTokenizer):
        def decode(self, token_ids: Any) -> str:
            time.sleep(0.2)
            return super().decode(token_ids)

    fake_worker = ReleasedThenFatalFakeWorker()
    app = create_app(
        settings(tmp_path),
        tokenizer=SlowDecodeTokenizer(),
        worker=fake_worker,
        api_key=API_KEY,
    )
    with TestClient(app) as instance:
        response = instance.post("/v1/chat/completions", headers=AUTH, json=body())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert fake_worker.fatal_response_acknowledged is True


def parse_sse(payload: str) -> list[str]:
    records = payload.split("\n\n")
    assert records[-1] == ""
    return [record.removeprefix("data: ") for record in records[:-1]]


def test_stream_completion_has_exact_order_content_and_usage(
    client: TestClient, fake_worker: FakeWorker
) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json=body(stream=True, stream_options={"include_usage": True}),
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    records = parse_sse(response.text)
    assert records[-1] == "[DONE]"
    chunks = [json.loads(record) for record in records[:-1]]
    identity = {
        (chunk["id"], chunk["created"], chunk["model"], chunk["object"])
        for chunk in chunks
    }
    assert len(identity) == 1
    assert chunks[0]["choices"][0]["delta"] == {
        "role": "assistant",
        "content": "",
    }
    content = "".join(chunk["choices"][0]["delta"]["content"] for chunk in chunks[1:-2])
    assert content == "日本語の応答"
    assert chunks[-2]["choices"][0]["delta"] == {}
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert fake_worker.fatal_response_ack_count == 0


def test_actual_openwebui_stream_fixture_is_accepted(client: TestClient) -> None:
    fixture_path = (
        Path(__file__).resolve().parents[3]
        / "tests/fixtures/sq8-serving-v0.1/openwebui/stream-request.json"
    )
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    response = client.post("/v1/chat/completions", headers=AUTH, json=payload)
    assert response.status_code == 200
    records = parse_sse(response.text)
    assert records[-1] == "[DONE]"
    chunks = [json.loads(record) for record in records[:-1]]
    assert all("usage" not in chunk for chunk in chunks)


@pytest.mark.parametrize(
    "outcome,token_ids,expected_content,expected_completion_tokens",
    [
        ("length", (101, 102), "日本語の長い応答", 2),
        ("stop", (EOS_TOKEN_IDS[0],), "", 1),
    ],
)
def test_stream_length_and_eos_only_terminal_contract(
    tmp_path: Path,
    outcome: str,
    token_ids: tuple[int, ...],
    expected_content: str,
    expected_completion_tokens: int,
) -> None:
    fake_worker = FakeWorker(outcome=outcome, token_ids=token_ids)
    app = create_app(
        settings(tmp_path),
        tokenizer=FakeTokenizer(),
        worker=fake_worker,
        api_key=API_KEY,
    )
    with TestClient(app) as instance:
        response = instance.post(
            "/v1/chat/completions",
            headers=AUTH,
            json=body(stream=True, stream_options={"include_usage": True}),
        )
    assert response.status_code == 200
    records = parse_sse(response.text)
    assert records[-1] == "[DONE]"
    chunks = [json.loads(record) for record in records[:-1]]
    content = "".join(
        choice["delta"]["content"]
        for chunk in chunks
        for choice in chunk["choices"]
        if "content" in choice["delta"] and choice["delta"].get("role") is None
    )
    assert content == expected_content
    assert chunks[-2]["choices"][0]["finish_reason"] == outcome
    assert chunks[-1]["usage"]["completion_tokens"] == expected_completion_tokens


def test_stream_failure_before_headers_returns_json_error(
    tmp_path: Path,
) -> None:
    fake_worker = StreamingFatalFakeWorker()
    app = create_app(
        settings(tmp_path),
        tokenizer=FakeTokenizer(),
        worker=fake_worker,
        api_key=API_KEY,
    )
    with TestClient(app) as instance:
        response = instance.post(
            "/v1/chat/completions", headers=AUTH, json=body(stream=True)
        )
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "internal_error"
    assert fake_worker.fatal_response_acknowledged is True


def test_stream_failure_after_headers_emits_one_error_and_no_done(
    tmp_path: Path,
) -> None:
    fake_worker = PostHeaderFatalFakeWorker()
    app = create_app(
        settings(tmp_path),
        tokenizer=FakeTokenizer(),
        worker=fake_worker,
        api_key=API_KEY,
    )
    with TestClient(app) as instance:
        response = instance.post(
            "/v1/chat/completions", headers=AUTH, json=body(stream=True)
        )
    assert response.status_code == 200
    records = parse_sse(response.text)
    assert records[-1] != "[DONE]"
    errors = [json.loads(record) for record in records if "error" in record]
    assert len(errors) == 1
    assert errors[0]["error"]["code"] == "internal_error"
    assert fake_worker.fatal_response_acknowledged is True


def test_fatal_interrupts_blocked_response_start_before_commit(tmp_path: Path) -> None:
    async def scenario() -> None:
        fake_worker = PostHeaderFatalFakeWorker()
        app = create_app(
            settings(tmp_path),
            tokenizer=FakeTokenizer(),
            worker=fake_worker,
            api_key=API_KEY,
        )
        scope, raw = chat_scope(body(stream=True))
        request_sent = False
        no_disconnect = asyncio.Event()
        statuses: list[int] = []

        async def receive() -> dict[str, Any]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": raw, "more_body": False}
            await no_disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] != "http.response.start":
                return
            content_type = dict(message["headers"])[b"content-type"]
            if content_type.startswith(b"text/event-stream"):
                await asyncio.sleep(0.4)
            statuses.append(message["status"])

        before = time.monotonic()
        async with app.router.lifespan_context(app):
            await app(scope, receive, send)
        assert time.monotonic() - before < 0.25
        assert statuses == [500]
        assert fake_worker.fatal_response_acknowledged is True

    asyncio.run(scenario())


def test_decoder_fatal_skips_normal_cancel_before_sse_error(tmp_path: Path) -> None:
    class FailingDecodeTokenizer(FakeTokenizer):
        def decode(self, _: Any) -> str:
            raise TokenizerError("injected decode failure")

    async def scenario() -> None:
        fake_worker = GatewayFatalPendingWorker()
        app = create_app(
            settings(tmp_path),
            tokenizer=FailingDecodeTokenizer(),
            worker=fake_worker,
            api_key=API_KEY,
        )
        scope, raw = chat_scope(body(stream=True))
        request_sent = False
        no_disconnect = asyncio.Event()
        response_body = bytearray()

        async def receive() -> dict[str, Any]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": raw, "more_body": False}
            await no_disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.body":
                response_body.extend(message.get("body", b""))

        before = time.monotonic()
        async with app.router.lifespan_context(app):
            await app(scope, receive, send)
        elapsed = time.monotonic() - before
        records = parse_sse(response_body.decode("utf-8"))
        assert elapsed < 0.25
        assert records[-1] != "[DONE]"
        assert json.loads(records[-1])["error"]["code"] == "internal_error"
        assert not fake_worker.cancel_entered
        assert fake_worker.fatal_response_ack_count == 1
        assert fake_worker.handle is not None
        fake_worker.handle._future.cancel()

    asyncio.run(scenario())


def test_stream_generator_close_cancels_and_drains_worker() -> None:
    async def scenario() -> None:
        fake_worker = CancelableFakeWorker()
        handle = await fake_worker.make_handle()
        app = FastAPI()
        executor = ThreadPoolExecutor(max_workers=1)
        app.state.worker = fake_worker
        app.state.tokenizer = FakeTokenizer()
        app.state.tokenizer_lock = asyncio.Lock()
        app.state.tokenizer_executor = executor
        app.state.stream_tokenizer = app.state.tokenizer
        app.state.stream_tokenizer_lock = asyncio.Lock()
        app.state.stream_tokenizer_executor = executor
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [],
                "query_string": b"",
                "app": app,
            }
        )
        generator = _stream_completion(request, handle, "chatcmpl-test", 1, False)
        first = await generator.__anext__()
        assert (
            json.loads(first.removeprefix(b"data: ").strip())["choices"][0]["delta"][
                "role"
            ]
            == "assistant"
        )
        await generator.aclose()
        assert fake_worker.cancel_reason == "client_disconnect"
        assert handle._future.done()
        executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_stream_generator_close_preserves_slow_client_cancel_reason() -> None:
    async def scenario() -> None:
        fake_worker = CancelableFakeWorker()
        handle = await fake_worker.make_handle()
        assert handle.stream_state is not None
        handle.stream_state.abort("slow_client")
        app = FastAPI()
        executor = ThreadPoolExecutor(max_workers=1)
        app.state.worker = fake_worker
        app.state.stream_tokenizer = FakeTokenizer()
        app.state.stream_tokenizer_lock = asyncio.Lock()
        app.state.stream_tokenizer_executor = executor
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [],
                "query_string": b"",
                "app": app,
            }
        )
        generator = _stream_completion(request, handle, "chatcmpl-test", 1, False)
        await generator.__anext__()
        await generator.aclose()
        assert fake_worker.cancel_reason == "slow_client"
        assert handle._future.done()
        executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_disconnect_before_worker_started_cancels_and_drains() -> None:
    async def scenario() -> None:
        fake_worker = CancelableFakeWorker()
        handle = await fake_worker.make_handle(started_ready=False)
        app = FastAPI()
        app.state.worker = fake_worker

        async def receive() -> dict[str, str]:
            return {"type": "http.disconnect"}

        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [],
                "query_string": b"",
                "app": app,
            },
            receive=receive,
        )
        assert await _wait_for_stream_start(request, fake_worker, handle) is False  # type: ignore[arg-type]
        assert fake_worker.cancel_reason == "client_disconnect"
        assert handle._future.done()

    asyncio.run(scenario())


def test_disconnect_observed_with_started_event_still_cancels() -> None:
    async def scenario() -> None:
        fake_worker = CancelableFakeWorker()
        handle = await fake_worker.make_handle(started_ready=True)
        app = FastAPI()
        app.state.worker = fake_worker

        async def receive() -> dict[str, str]:
            return {"type": "http.disconnect"}

        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [],
                "query_string": b"",
                "app": app,
            },
            receive=receive,
        )
        assert await _wait_for_stream_start(request, fake_worker, handle) is False  # type: ignore[arg-type]
        assert fake_worker.cancel_reason == "client_disconnect"
        assert handle._future.done()

    asyncio.run(scenario())


def test_asgi_23_disconnect_shields_worker_cancel_and_drain(tmp_path: Path) -> None:
    async def scenario() -> None:
        fake_worker = PendingStreamFakeWorker()
        app = create_app(
            settings(tmp_path),
            tokenizer=FakeTokenizer(),
            worker=fake_worker,
            api_key=API_KEY,
        )
        scope, raw = chat_scope(body(stream=True))
        request_sent = False
        disconnect = asyncio.Event()
        messages: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": raw, "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            messages.append(message)
            if (
                message["type"] == "http.response.body"
                and message.get("more_body") is True
            ):
                disconnect.set()

        async with app.router.lifespan_context(app):
            await app(scope, receive, send)
        assert messages[0]["status"] == 200
        assert fake_worker.cancel_reason == "client_disconnect"
        assert fake_worker.handle is not None
        assert fake_worker.handle._future.done()

    asyncio.run(scenario())


def test_blocked_send_is_cancelled_when_stream_queue_aborts(tmp_path: Path) -> None:
    async def scenario() -> None:
        fake_worker = PendingStreamFakeWorker()
        app = create_app(
            settings(tmp_path),
            tokenizer=FakeTokenizer(),
            worker=fake_worker,
            api_key=API_KEY,
        )
        scope, raw = chat_scope(body(stream=True))
        request_sent = False
        never_disconnect = asyncio.Event()
        blocked_send_cancelled = asyncio.Event()
        response_status: int | None = None

        async def receive() -> dict[str, Any]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": raw, "more_body": False}
            await never_disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message["status"]
                return
            if message.get("more_body") is not True:
                return
            assert fake_worker.handle is not None
            assert fake_worker.handle.stream_state is not None
            asyncio.get_running_loop().call_later(
                0.01, fake_worker.handle.stream_state.abort, "slow_client"
            )
            try:
                await asyncio.Event().wait()
            finally:
                blocked_send_cancelled.set()

        async with app.router.lifespan_context(app):
            await app(scope, receive, send)
        assert response_status == 200
        assert blocked_send_cancelled.is_set()
        assert fake_worker.cancel_reason == "slow_client"
        assert fake_worker.handle is not None
        assert fake_worker.handle._future.done()

    asyncio.run(scenario())


def test_http_lifecycle_gate_prevents_post_release_request_overlap(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        fake_worker = FakeWorker()
        app = create_app(
            settings(tmp_path),
            tokenizer=FakeTokenizer(prompt_tokens=4_000),
            worker=fake_worker,
            api_key=API_KEY,
        )
        first_scope, first_raw = chat_scope(body(stream=True, max_tokens=8))
        first_request_sent = False
        no_disconnect = asyncio.Event()
        first_body_blocked = asyncio.Event()
        release_first_body = asyncio.Event()

        async def first_receive() -> dict[str, Any]:
            nonlocal first_request_sent
            if not first_request_sent:
                first_request_sent = True
                return {
                    "type": "http.request",
                    "body": first_raw,
                    "more_body": False,
                }
            await no_disconnect.wait()
            return {"type": "http.disconnect"}

        async def first_send(message: dict[str, Any]) -> None:
            if (
                message["type"] == "http.response.body"
                and message.get("more_body") is True
                and not first_body_blocked.is_set()
            ):
                first_body_blocked.set()
                await release_first_body.wait()

        async def invoke(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            scope, raw = chat_scope(payload)
            request_sent = False
            status = 0
            response_body = bytearray()

            async def receive() -> dict[str, Any]:
                nonlocal request_sent
                if not request_sent:
                    request_sent = True
                    return {
                        "type": "http.request",
                        "body": raw,
                        "more_body": False,
                    }
                return {"type": "http.disconnect"}

            async def send(message: dict[str, Any]) -> None:
                nonlocal status
                if message["type"] == "http.response.start":
                    status = message["status"]
                elif message["type"] == "http.response.body":
                    response_body.extend(message.get("body", b""))

            await app(scope, receive, send)
            return status, json.loads(response_body)

        async with app.router.lifespan_context(app):
            first = asyncio.create_task(app(first_scope, first_receive, first_send))
            await asyncio.wait_for(first_body_blocked.wait(), timeout=1.0)
            collision_status, collision = await invoke(body(max_tokens=8))
            assert collision_status == 429
            assert collision["error"]["code"] == "request_busy"
            overflow_status, overflow = await invoke(body(max_tokens=100))
            assert overflow_status == 400
            assert overflow["error"]["code"] == "context_length_exceeded"
            assert fake_worker.generate_count == 1
            release_first_body.set()
            await first
            recovery_status, _ = await invoke(body(max_tokens=8))
            assert recovery_status == 200
            assert fake_worker.generate_count == 2

    asyncio.run(scenario())


def test_http_lifecycle_gate_covers_post_claim_error_body(tmp_path: Path) -> None:
    class FailOnceDecodeTokenizer(FakeTokenizer):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        def decode(self, token_ids: Any) -> str:
            if not self.failed:
                self.failed = True
                raise TokenizerError("injected final decode failure")
            return super().decode(token_ids)

    async def scenario() -> None:
        fake_worker = FakeWorker()
        app = create_app(
            settings(tmp_path),
            tokenizer=FailOnceDecodeTokenizer(),
            worker=fake_worker,
            api_key=API_KEY,
        )
        first_scope, first_raw = chat_scope(body())
        first_request_sent = False
        error_body_blocked = asyncio.Event()
        release_error_body = asyncio.Event()
        first_status = 0

        async def first_receive() -> dict[str, Any]:
            nonlocal first_request_sent
            if not first_request_sent:
                first_request_sent = True
                return {
                    "type": "http.request",
                    "body": first_raw,
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

        async def first_send(message: dict[str, Any]) -> None:
            nonlocal first_status
            if message["type"] == "http.response.start":
                first_status = message["status"]
            elif message["type"] == "http.response.body":
                error_body_blocked.set()
                await release_error_body.wait()

        async def invoke() -> int:
            scope, raw = chat_scope(body())
            request_sent = False
            status = 0

            async def receive() -> dict[str, Any]:
                nonlocal request_sent
                if not request_sent:
                    request_sent = True
                    return {
                        "type": "http.request",
                        "body": raw,
                        "more_body": False,
                    }
                return {"type": "http.disconnect"}

            async def send(message: dict[str, Any]) -> None:
                nonlocal status
                if message["type"] == "http.response.start":
                    status = message["status"]

            await app(scope, receive, send)
            return status

        async with app.router.lifespan_context(app):
            first = asyncio.create_task(app(first_scope, first_receive, first_send))
            await asyncio.wait_for(error_body_blocked.wait(), timeout=1.0)
            assert first_status == 500
            assert await invoke() == 429
            assert fake_worker.generate_count == 1
            release_error_body.set()
            await first
            assert await invoke() == 200
            assert fake_worker.generate_count == 2

    asyncio.run(scenario())


def test_done_send_wins_over_simultaneous_later_worker_failure(tmp_path: Path) -> None:
    async def scenario() -> None:
        fake_worker = FakeWorker()
        app = create_app(
            settings(tmp_path),
            tokenizer=FakeTokenizer(),
            worker=fake_worker,
            api_key=API_KEY,
        )
        scope, raw = chat_scope(body(stream=True))
        request_sent = False
        no_disconnect = asyncio.Event()
        body_bytes = bytearray()

        async def receive() -> dict[str, Any]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": raw, "more_body": False}
            await no_disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] != "http.response.body":
                return
            chunk = message.get("body", b"")
            body_bytes.extend(chunk)
            if chunk == b"data: [DONE]\n\n":
                fake_worker._failed = True
                fake_worker._fatal_event.set()
                for _ in range(4):
                    await asyncio.sleep(0)

        async with app.router.lifespan_context(app):
            await app(scope, receive, send)
        records = parse_sse(body_bytes.decode("utf-8"))
        assert records[-1] == "[DONE]"
        assert all('"error"' not in record for record in records)
        assert fake_worker.fatal_response_ack_count == 1

    asyncio.run(scenario())
