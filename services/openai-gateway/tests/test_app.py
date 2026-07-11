from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ullm_openai_gateway.app import create_app
from ullm_openai_gateway.schemas import EOS_TOKEN_IDS, MODEL_ID
from ullm_openai_gateway.settings import GatewaySettings
from ullm_openai_gateway.tokenizer import TokenizedPrompt
from ullm_openai_gateway.worker import (
    GenerationHandle,
    WorkerBusy,
    WorkerGenerationResult,
    WorkerFatal,
    WorkerNotReady,
)


API_KEY = b"test-secret"
AUTH = {"Authorization": "Bearer test-secret"}


class FakeTokenizer:
    def __init__(self, prompt_tokens: int = 3) -> None:
        self.prompt_tokens = prompt_tokens

    def render(self, _: Any) -> TokenizedPrompt:
        return TokenizedPrompt("rendered", tuple(range(self.prompt_tokens)))

    def decode(self, token_ids: Any) -> str:
        assert tuple(token_ids) == (101, EOS_TOKEN_IDS[0])
        return "日本語の応答"


class FakeWorker:
    def __init__(self) -> None:
        self.ready = True
        self.busy = False
        self.generate_count = 0
        self.requests: list[Any] = []
        self._generation = 0

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
        future.set_result(
            WorkerGenerationResult(
                request_id=f"req-{self._generation}",
                outcome="stop",
                prompt_tokens=len(request.prompt_token_ids),
                token_ids=(101, EOS_TOKEN_IDS[0]),
            )
        )
        return GenerationHandle(f"req-{self._generation}", self._generation, future)

    async def wait(self, handle: GenerationHandle) -> WorkerGenerationResult:
        return await handle._future

    async def cancel(self, _: GenerationHandle, __: str) -> None:
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
        self.fatal_response_acknowledged = True


class SlowTokenizer(FakeTokenizer):
    def __init__(self, started: threading.Event) -> None:
        super().__init__()
        self.started = started

    def render(self, messages: Any) -> TokenizedPrompt:
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


@pytest.mark.parametrize(
    "payload,code",
    [
        (
            {"model": MODEL_ID, "messages": [{"role": "assistant", "content": "x"}]},
            "invalid_request_error",
        ),
        (body(model="missing"), "model_not_found"),
        (body(tools=[]), "unsupported_parameter"),
        (body(stream=True), "unsupported_parameter"),
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
