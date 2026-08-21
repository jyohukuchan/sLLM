use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use axum::Router;
use serde_json::Value;
use sllm_frontend::GenerationCancellationV1;
use sllm_server::{
    BackendCompletionV1, BackendErrorV1, BackendTokenLogprobV1, BackendTopLogprobV1,
    ChatCompletionRequestV1, ChatGenerationBackendV1, FinishReasonV1, GenerationDeltaSinkV1,
    ModelRegistryEntryV1, ModelRegistryV1, SchedulerConfigV1, SchedulerV1, ServerConfigV1,
    TokenUsageV1, build_router_v1,
};

// Selected response/usage/finish, first-role/final-chunk, authentication, and
// disconnect test ideas are adapted to sLLM's narrower profile from llama.cpp
// commit f5919bf458ef190468b5c329bb293f8a54a1e69c. The exact upstream files,
// blobs, license, local digest, and modifications are recorded in
// THIRD_PARTY_NOTICES.md; no llama-specific request fields are accepted here.

const FINGERPRINT: &str = "sha256:0000000000000000000000000000000000000000000000000000000000000000";

struct ScriptBackend {
    deltas: Vec<String>,
    finish_reason: FinishReasonV1,
    usage: TokenUsageV1,
    fail_after: Option<usize>,
    cancelled: Arc<AtomicUsize>,
}

impl ChatGenerationBackendV1 for ScriptBackend {
    fn generate(
        &self,
        request: &ChatCompletionRequestV1,
        cancellation: &GenerationCancellationV1,
        sink: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        for (index, delta) in self.deltas.iter().enumerate() {
            if cancellation.is_cancelled() {
                self.cancelled.fetch_add(1, Ordering::Relaxed);
                return Err(BackendErrorV1::new("cancelled"));
            }
            sink.publish(delta)?;
            if self.fail_after == Some(index + 1) {
                return Err(BackendErrorV1::new("scripted backend failure"));
            }
        }
        if request.logprobs().is_some_and(|options| options.enabled()) {
            sink.publish_logprobs(vec![BackendTokenLogprobV1 {
                token: "ok".to_owned(),
                bytes: Some(vec![111, 107]),
                logprob: -0.25,
                top_logprobs: vec![BackendTopLogprobV1 {
                    token: "alt".to_owned(),
                    bytes: Some(vec![97, 108, 116]),
                    logprob: -1.5,
                }],
            }])?;
        }
        Ok(BackendCompletionV1 {
            finish_reason: self.finish_reason,
            usage: self.usage,
            matched_stop: None,
        })
    }
}

struct BlockingBackend {
    started: AtomicUsize,
    release: AtomicBool,
    cancelled: AtomicUsize,
}

impl BlockingBackend {
    fn new() -> Self {
        Self {
            started: AtomicUsize::new(0),
            release: AtomicBool::new(false),
            cancelled: AtomicUsize::new(0),
        }
    }
}

impl ChatGenerationBackendV1 for BlockingBackend {
    fn generate(
        &self,
        _: &ChatCompletionRequestV1,
        cancellation: &GenerationCancellationV1,
        sink: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        self.started.fetch_add(1, Ordering::Release);
        while !self.release.load(Ordering::Acquire) {
            if cancellation.is_cancelled() {
                self.cancelled.fetch_add(1, Ordering::Relaxed);
                return Err(BackendErrorV1::new("cancelled"));
            }
            thread::sleep(Duration::from_millis(2));
        }
        sink.publish("ok")?;
        Ok(BackendCompletionV1 {
            finish_reason: FinishReasonV1::Stop,
            usage: TokenUsageV1::new(1, 1).unwrap(),
            matched_stop: None,
        })
    }
}

struct EndlessBackend {
    cancelled: Arc<AtomicBool>,
}

struct FailOnceBackend {
    calls: AtomicUsize,
}

impl ChatGenerationBackendV1 for FailOnceBackend {
    fn generate(
        &self,
        _: &ChatCompletionRequestV1,
        _: &GenerationCancellationV1,
        sink: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        if self.calls.fetch_add(1, Ordering::AcqRel) == 0 {
            return Err(BackendErrorV1::new("first request failed"));
        }
        sink.publish("healthy")?;
        Ok(BackendCompletionV1 {
            finish_reason: FinishReasonV1::Stop,
            usage: TokenUsageV1::new(1, 1).unwrap(),
            matched_stop: None,
        })
    }
}

impl ChatGenerationBackendV1 for EndlessBackend {
    fn generate(
        &self,
        _: &ChatCompletionRequestV1,
        cancellation: &GenerationCancellationV1,
        sink: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        let payload = "x".repeat(16 * 1024);
        loop {
            if cancellation.is_cancelled() {
                self.cancelled.store(true, Ordering::Release);
                return Err(BackendErrorV1::new("cancelled"));
            }
            if let Err(error) = sink.publish(&payload) {
                self.cancelled.store(true, Ordering::Release);
                return Err(error);
            }
        }
    }
}

fn router(
    backend: Arc<dyn ChatGenerationBackendV1>,
    queue_capacity: usize,
    event_capacity: usize,
    bearer: Option<&str>,
) -> (Router, SchedulerV1) {
    router_with_config(
        backend,
        queue_capacity,
        event_capacity,
        ServerConfigV1::new(bearer.map(str::to_owned)).unwrap(),
    )
}

fn router_with_config(
    backend: Arc<dyn ChatGenerationBackendV1>,
    queue_capacity: usize,
    event_capacity: usize,
    config: ServerConfigV1,
) -> (Router, SchedulerV1) {
    let entry = ModelRegistryEntryV1::new("qwen-test", 1_700_000_000, "sllm", FINGERPRINT, backend)
        .unwrap();
    let registry = ModelRegistryV1::new(vec![entry]).unwrap();
    let scheduler = SchedulerV1::new(
        SchedulerConfigV1::new(queue_capacity, event_capacity, Duration::from_secs(5)).unwrap(),
    );
    let app = build_router_v1(registry, scheduler.clone(), config);
    (app, scheduler)
}

async fn serve(app: Router) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (address, task)
}

fn request_bytes(method: &str, path: &str, body: &[u8], extra_headers: &[&str]) -> Vec<u8> {
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: {}\r\n",
        body.len()
    );
    for header in extra_headers {
        request.push_str(header);
        request.push_str("\r\n");
    }
    request.push_str("\r\n");
    let mut bytes = request.into_bytes();
    bytes.extend_from_slice(body);
    bytes
}

async fn raw_http(address: SocketAddr, request: Vec<u8>) -> RawResponse {
    tokio::task::spawn_blocking(move || {
        let mut socket = TcpStream::connect(address).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        socket.write_all(&request).unwrap();
        let mut response = Vec::new();
        socket.read_to_end(&mut response).unwrap();
        RawResponse::parse(&response)
    })
    .await
    .unwrap()
}

struct RawResponse {
    status: u16,
    headers: String,
    body: Vec<u8>,
}

impl RawResponse {
    fn parse(bytes: &[u8]) -> Self {
        let split = bytes
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .unwrap_or_else(|| panic!("HTTP response has no header terminator: {bytes:?}"));
        let headers = String::from_utf8(bytes[..split].to_vec()).unwrap();
        let status = headers
            .lines()
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let raw_body = &bytes[split + 4..];
        let body = if headers
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked")
        {
            decode_chunked(raw_body)
        } else {
            raw_body.to_vec()
        };
        Self {
            status,
            headers,
            body,
        }
    }

    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap()
    }
}

fn decode_chunked(mut input: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    loop {
        let line_end = input
            .windows(2)
            .position(|window| window == b"\r\n")
            .unwrap();
        let size_text = std::str::from_utf8(&input[..line_end]).unwrap();
        let size = usize::from_str_radix(size_text.split(';').next().unwrap(), 16).unwrap();
        input = &input[line_end + 2..];
        if size == 0 {
            break;
        }
        output.extend_from_slice(&input[..size]);
        input = &input[size + 2..];
    }
    output
}

fn valid_body(stream: bool) -> Vec<u8> {
    format!(
        r#"{{"model":"qwen-test","messages":[{{"role":"user","content":"hello"}}],"stream":{stream},"max_completion_tokens":17}}"#
    )
    .into_bytes()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn models_non_stream_and_sse_share_text_usage_and_finish_reason() {
    let cancelled = Arc::new(AtomicUsize::new(0));
    let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(ScriptBackend {
        deltas: vec!["hel".to_owned(), "lo終".to_owned()],
        finish_reason: FinishReasonV1::Stop,
        usage: TokenUsageV1::new(3, 2).unwrap(),
        fail_after: None,
        cancelled,
    });
    let (app, scheduler) = router(backend, 4, 2, None);
    let (address, server) = serve(app).await;

    let models = raw_http(address, request_bytes("GET", "/v1/models", b"", &[])).await;
    assert_eq!(models.status, 200);
    assert_eq!(models.json()["object"], "list");
    assert_eq!(models.json()["data"][0]["id"], "qwen-test");
    assert_eq!(models.json()["data"][0]["created"], 1_700_000_000_u64);

    let non_stream = raw_http(
        address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            &valid_body(false),
            &["Content-Type: application/json"],
        ),
    )
    .await;
    assert_eq!(non_stream.status, 200);
    let response = non_stream.json();
    assert_eq!(response["object"], "chat.completion");
    assert_eq!(response["choices"][0]["message"]["content"], "hello終");
    assert_eq!(response["choices"][0]["finish_reason"], "stop");
    assert_eq!(response["usage"]["prompt_tokens"], 3);
    assert_eq!(response["usage"]["completion_tokens"], 2);
    assert_eq!(response["usage"]["total_tokens"], 5);

    let streaming = raw_http(
        address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            &valid_body(true),
            &["Content-Type: application/json"],
        ),
    )
    .await;
    assert_eq!(streaming.status, 200);
    assert!(
        streaming
            .headers
            .to_ascii_lowercase()
            .contains("text/event-stream")
    );
    let body = String::from_utf8(streaming.body).unwrap();
    assert!(body.starts_with("data: {\"id\":"));
    assert!(body.contains(r#""delta":{"role":"assistant","content":""}"#));
    assert!(body.contains(r#""delta":{"content":"hel"}"#));
    assert!(body.contains(r#""delta":{"content":"lo終"}"#));
    assert!(body.contains(r#""finish_reason":"stop""#));
    assert!(body.contains(r#""usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}"#));
    assert!(body.ends_with("data: [DONE]\n\n"));

    scheduler.shutdown();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn requested_logprobs_are_mapped_for_buffered_and_streaming_responses() {
    let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(ScriptBackend {
        deltas: vec!["ok".to_owned()],
        finish_reason: FinishReasonV1::Stop,
        usage: TokenUsageV1::new(2, 1).unwrap(),
        fail_after: None,
        cancelled: Arc::new(AtomicUsize::new(0)),
    });
    let (app, scheduler) = router(backend, 4, 4, None);
    let (address, server) = serve(app).await;
    let body = |stream| {
        format!(
            r#"{{"model":"qwen-test","messages":[{{"role":"user","content":"hello"}}],"stream":{stream},"logprobs":true,"top_logprobs":1,"max_completion_tokens":2}}"#
        )
    };

    let buffered = raw_http(
        address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            body(false).as_bytes(),
            &["Content-Type: application/json"],
        ),
    )
    .await;
    assert_eq!(buffered.status, 200);
    let json = buffered.json();
    assert_eq!(json["choices"][0]["logprobs"]["content"][0]["token"], "ok");
    assert_eq!(
        json["choices"][0]["logprobs"]["content"][0]["bytes"],
        serde_json::json!([111, 107])
    );
    assert_eq!(
        json["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0]["token"],
        "alt"
    );

    let streaming = raw_http(
        address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            body(true).as_bytes(),
            &["Content-Type: application/json"],
        ),
    )
    .await;
    assert_eq!(streaming.status, 200);
    let text = String::from_utf8(streaming.body).unwrap();
    assert!(text.contains(r#""logprobs":{"content":[{"token":"ok""#));
    assert!(text.ends_with("data: [DONE]\n\n"));

    scheduler.shutdown();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn multiple_choices_use_independent_slots_and_aggregate_usage() {
    let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(ScriptBackend {
        deltas: vec!["choice".to_owned()],
        finish_reason: FinishReasonV1::Stop,
        usage: TokenUsageV1::new(3, 2).unwrap(),
        fail_after: None,
        cancelled: Arc::new(AtomicUsize::new(0)),
    });
    let (app, scheduler) = router(backend, 8, 4, None);
    let (address, server) = serve(app).await;
    let body = br#"{"model":"qwen-test","messages":[{"role":"user","content":"hello"}],"n":2,"max_completion_tokens":17}"#;

    let response = raw_http(
        address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            body,
            &["Content-Type: application/json"],
        ),
    )
    .await;
    assert_eq!(response.status, 200);
    let response = response.json();
    assert_eq!(response["choices"].as_array().unwrap().len(), 2);
    assert_eq!(response["choices"][0]["index"], 0);
    assert_eq!(response["choices"][1]["index"], 1);
    assert_eq!(response["choices"][0]["message"]["content"], "choice");
    assert_eq!(response["choices"][1]["message"]["content"], "choice");
    assert_eq!(response["usage"]["prompt_tokens"], 3);
    assert_eq!(response["usage"]["completion_tokens"], 4);
    assert_eq!(response["usage"]["total_tokens"], 7);

    let stream_body = br#"{"model":"qwen-test","messages":[{"role":"user","content":"hello"}],"n":2,"stream":true,"max_completion_tokens":17}"#;
    let response = raw_http(
        address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            stream_body,
            &["Content-Type: application/json"],
        ),
    )
    .await;
    assert_eq!(response.status, 200);
    let events = String::from_utf8(response.body)
        .unwrap()
        .split("\n\n")
        .filter_map(|block| block.strip_prefix("data: "))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    assert_eq!(events.iter().filter(|event| *event == "[DONE]").count(), 1);
    let chunks = events
        .iter()
        .filter(|event| event.as_str() != "[DONE]")
        .map(|event| serde_json::from_str::<Value>(event).unwrap())
        .collect::<Vec<_>>();
    let role_indices = chunks
        .iter()
        .filter(|chunk| chunk["choices"][0]["delta"]["role"] == "assistant")
        .map(|chunk| chunk["choices"][0]["index"].as_u64().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(role_indices, [0, 1]);
    let final_chunk = chunks.last().unwrap();
    assert_eq!(final_chunk["choices"][0]["index"], 1);
    assert_eq!(final_chunk["usage"]["total_tokens"], 7);

    scheduler.shutdown();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn strict_http_error_matrix_uses_profile_envelopes() {
    let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(ScriptBackend {
        deltas: vec!["x".to_owned()],
        finish_reason: FinishReasonV1::Length,
        usage: TokenUsageV1::new(1, 1).unwrap(),
        fail_after: None,
        cancelled: Arc::new(AtomicUsize::new(0)),
    });
    let (app, scheduler) = router(backend, 2, 2, Some("secret"));
    let (address, server) = serve(app).await;
    let cases = [
        (
            request_bytes("GET", "/v1/models", b"", &[]),
            401,
            "invalid_api_key",
        ),
        (
            request_bytes(
                "POST",
                "/v1/chat/completions",
                &valid_body(false),
                &["Authorization: Bearer secret", "Content-Type: text/plain"],
            ),
            415,
            "unsupported_media_type",
        ),
        (
            request_bytes(
                "POST",
                "/v1/chat/completions",
                b"{",
                &[
                    "Authorization: Bearer secret",
                    "Content-Type: application/json",
                ],
            ),
            400,
            "invalid_json",
        ),
        (
            request_bytes(
                "POST",
                "/v1/chat/completions",
                br#"{"model":"missing","messages":[{"role":"user","content":"x"}]}"#,
                &[
                    "Authorization: Bearer secret",
                    "Content-Type: application/json",
                ],
            ),
            404,
            "model_not_found",
        ),
        (
            request_bytes(
                "POST",
                "/v1/chat/completions",
                br#"{"model":"qwen-test","messages":[{"role":"user","content":"x"}],"tools":null}"#,
                &[
                    "Authorization: Bearer secret",
                    "Content-Type: application/json",
                ],
            ),
            400,
            "unsupported_parameter",
        ),
    ];
    for (request, expected_status, expected_code) in cases {
        let response = raw_http(address, request).await;
        assert_eq!(response.status, expected_status);
        assert_eq!(response.json()["error"]["code"], expected_code);
    }

    let oversized = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\nContent-Length: 100663297\r\nConnection: close\r\n\r\n".to_vec();
    let response = raw_http(address, oversized).await;
    assert_eq!(response.status, 413);
    assert_eq!(response.json()["error"]["code"], "request_too_large");

    scheduler.shutdown();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queue_full_is_429_and_fifo_work_continues() {
    let backend = Arc::new(BlockingBackend::new());
    let erased: Arc<dyn ChatGenerationBackendV1> = backend.clone();
    let (app, scheduler) = router(erased, 1, 1, None);
    let (address, server) = serve(app).await;
    let request = request_bytes(
        "POST",
        "/v1/chat/completions",
        &valid_body(false),
        &["Content-Type: application/json"],
    );
    let first = tokio::spawn(raw_http(address, request.clone()));
    wait_until(Duration::from_secs(1), || {
        backend.started.load(Ordering::Acquire) == 1
    })
    .await;
    let second = tokio::spawn(raw_http(address, request.clone()));
    tokio::time::sleep(Duration::from_millis(20)).await;
    let third = raw_http(address, request).await;
    assert_eq!(third.status, 429);
    assert_eq!(third.json()["error"]["code"], "rate_limit_exceeded");
    backend.release.store(true, Ordering::Release);
    assert_eq!(first.await.unwrap().status, 200);
    assert_eq!(second.await.unwrap().status, 200);
    assert_eq!(backend.started.load(Ordering::Acquire), 2);

    scheduler.shutdown();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backend_failure_does_not_poison_the_model_owner() {
    let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(FailOnceBackend {
        calls: AtomicUsize::new(0),
    });
    let (app, scheduler) = router(backend, 2, 2, None);
    let (address, server) = serve(app).await;
    let request = request_bytes(
        "POST",
        "/v1/chat/completions",
        &valid_body(false),
        &["Content-Type: application/json"],
    );

    let failed = raw_http(address, request.clone()).await;
    assert_eq!(failed.status, 500);
    assert_eq!(failed.json()["error"]["code"], "generation_failed");
    let healthy = raw_http(address, request).await;
    assert_eq!(healthy.status, 200);
    assert_eq!(
        healthy.json()["choices"][0]["message"]["content"],
        "healthy"
    );

    scheduler.shutdown();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn empty_delta_stop_first_and_unicode_length_streams_are_well_formed() {
    let stop_backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(ScriptBackend {
        deltas: vec![String::new()],
        finish_reason: FinishReasonV1::Stop,
        usage: TokenUsageV1::new(2, 1).unwrap(),
        fail_after: None,
        cancelled: Arc::new(AtomicUsize::new(0)),
    });
    let (app, scheduler) = router(stop_backend, 2, 1, None);
    let (address, server) = serve(app).await;
    let response = raw_http(
        address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            &valid_body(true),
            &["Content-Type: application/json"],
        ),
    )
    .await;
    let body = String::from_utf8(response.body).unwrap();
    assert_eq!(body.matches(r#""content":"""#).count(), 1);
    assert!(body.contains(r#""finish_reason":"stop""#));
    assert!(body.ends_with("data: [DONE]\n\n"));
    scheduler.shutdown();
    server.abort();

    let length_backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(ScriptBackend {
        deltas: vec!["終".to_owned(), "端".to_owned()],
        finish_reason: FinishReasonV1::Length,
        usage: TokenUsageV1::new(2, 17).unwrap(),
        fail_after: None,
        cancelled: Arc::new(AtomicUsize::new(0)),
    });
    let (app, scheduler) = router(length_backend, 2, 1, None);
    let (address, server) = serve(app).await;
    let response = raw_http(
        address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            &valid_body(true),
            &["Content-Type: application/json"],
        ),
    )
    .await;
    let body = String::from_utf8(response.body).unwrap();
    assert!(body.contains(r#""content":"終""#));
    assert!(body.contains(r#""content":"端""#));
    assert!(body.contains(r#""finish_reason":"length""#));
    assert!(body.contains(r#""completion_tokens":17"#));
    assert!(body.ends_with("data: [DONE]\n\n"));
    scheduler.shutdown();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn post_header_failure_is_terminal_error_without_done() {
    let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(ScriptBackend {
        deltas: vec!["partial".to_owned(), "unused".to_owned()],
        finish_reason: FinishReasonV1::Stop,
        usage: TokenUsageV1::new(1, 1).unwrap(),
        fail_after: Some(1),
        cancelled: Arc::new(AtomicUsize::new(0)),
    });
    let (app, scheduler) = router(backend, 2, 2, None);
    let (address, server) = serve(app).await;
    let response = raw_http(
        address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            &valid_body(true),
            &["Content-Type: application/json"],
        ),
    )
    .await;
    let body = String::from_utf8(response.body).unwrap();
    assert_eq!(response.status, 200);
    assert!(body.contains("partial"));
    assert!(body.contains("generation_failed"));
    assert!(!body.contains("[DONE]"));

    scheduler.shutdown();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_disconnect_cancels_active_generation() {
    let cancelled = Arc::new(AtomicBool::new(false));
    let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(EndlessBackend {
        cancelled: Arc::clone(&cancelled),
    });
    let (app, scheduler) = router(backend, 2, 1, None);
    let (address, server) = serve(app).await;
    let request = request_bytes(
        "POST",
        "/v1/chat/completions",
        &valid_body(true),
        &["Content-Type: application/json"],
    );
    tokio::task::spawn_blocking(move || {
        let mut socket = TcpStream::connect(address).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        socket.write_all(&request).unwrap();
        let mut response_prefix = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !response_prefix
            .windows(4)
            .any(|window| window == b"\r\n\r\n")
        {
            let count = socket.read(&mut buffer).unwrap();
            assert!(count > 0, "server closed before streaming response headers");
            response_prefix.extend_from_slice(&buffer[..count]);
        }
        socket.shutdown(Shutdown::Both).unwrap();
    })
    .await
    .unwrap();
    wait_until(Duration::from_secs(2), || cancelled.load(Ordering::Acquire)).await;
    assert!(cancelled.load(Ordering::Acquire));

    scheduler.shutdown();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_profile_fixture_matches_raw_http_and_sse() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/openai_chat_profile_v1.json"
    ))
    .unwrap();
    assert_eq!(
        fixture["official_openapi"]["commit"],
        "117ce5680e4269f6656a4fd70d28f9755630d938"
    );
    let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(ScriptBackend {
        deltas: fixture["positive"]["stream"]["content_deltas"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect(),
        finish_reason: FinishReasonV1::Stop,
        usage: TokenUsageV1::new(3, 2).unwrap(),
        fail_after: None,
        cancelled: Arc::new(AtomicUsize::new(0)),
    });
    let (app, scheduler) = router(backend, 8, 4, None);
    let (address, server) = serve(app).await;

    for case in fixture["negative"].as_array().unwrap() {
        let body = match &case["body"] {
            Value::String(value) => value.as_bytes().to_vec(),
            value => serde_json::to_vec(value).unwrap(),
        };
        let response = raw_http(
            address,
            request_bytes(
                "POST",
                "/v1/chat/completions",
                &body,
                &["Content-Type: application/json"],
            ),
        )
        .await;
        assert_eq!(response.status, case["status"].as_u64().unwrap() as u16);
        let envelope = response.json();
        assert_eq!(envelope["error"]["code"], case["code"]);
        assert_eq!(envelope["error"]["param"], case["param"]);
    }

    let mut request = fixture["positive"]["request"].clone();
    request["stream"] = Value::Bool(false);
    let response = raw_http(
        address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            &serde_json::to_vec(&request).unwrap(),
            &["Content-Type: application/json"],
        ),
    )
    .await;
    assert_eq!(response.status, 200);
    let response = response.json();
    let expected = &fixture["positive"]["non_stream"];
    assert_eq!(response["object"], expected["object"]);
    assert_eq!(response["choices"][0]["index"], expected["choice_index"]);
    assert_eq!(response["choices"][0]["message"]["role"], expected["role"]);
    assert_eq!(
        response["choices"][0]["message"]["content"],
        expected["content"]
    );
    assert_eq!(
        response["choices"][0]["finish_reason"],
        expected["finish_reason"]
    );
    assert_eq!(response["usage"]["total_tokens"], expected["total_tokens"]);

    request["stream"] = Value::Bool(true);
    let response = raw_http(
        address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            &serde_json::to_vec(&request).unwrap(),
            &["Content-Type: application/json"],
        ),
    )
    .await;
    assert_eq!(response.status, 200);
    let events = String::from_utf8(response.body)
        .unwrap()
        .split("\n\n")
        .filter(|block| !block.is_empty())
        .map(|block| block.strip_prefix("data: ").unwrap().to_owned())
        .collect::<Vec<_>>();
    assert_eq!(events.last().unwrap(), "[DONE]");
    let chunks = events[..events.len() - 1]
        .iter()
        .map(|event| serde_json::from_str::<Value>(event).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "hel");
    assert_eq!(chunks[2]["choices"][0]["delta"]["content"], "lo終");
    let final_chunk = chunks.last().unwrap();
    assert_eq!(final_chunk["choices"][0]["finish_reason"], "stop");
    assert_eq!(final_chunk["usage"]["total_tokens"], 5);
    let id = &chunks[0]["id"];
    assert!(chunks.iter().all(|chunk| chunk["id"] == *id));

    scheduler.shutdown();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reasoning_extension_separates_tags_for_non_stream_and_sse() {
    let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(ScriptBackend {
        deltas: vec![
            "<thi".to_owned(),
            "nk>\nrea".to_owned(),
            "son</thi".to_owned(),
            "nk>\n\nans".to_owned(),
            "wer".to_owned(),
        ],
        finish_reason: FinishReasonV1::Stop,
        usage: TokenUsageV1::new(3, 5).unwrap(),
        fail_after: None,
        cancelled: Arc::new(AtomicUsize::new(0)),
    });
    let (app, scheduler) = router(backend, 4, 2, None);
    let (address, server) = serve(app).await;
    let body = |stream| {
        serde_json::to_vec(&serde_json::json!({
            "model": "qwen-test",
            "messages": [{"role": "user", "content": "why"}],
            "stream": stream,
            "max_completion_tokens": 17,
            "sllm": {"thinking": "enabled", "separate_reasoning": true}
        }))
        .unwrap()
    };

    let non_stream = raw_http(
        address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            &body(false),
            &["Content-Type: application/json"],
        ),
    )
    .await;
    assert_eq!(non_stream.status, 200);
    let response = non_stream.json();
    assert_eq!(
        response["choices"][0]["message"]["reasoning_content"],
        "reason"
    );
    assert_eq!(response["choices"][0]["message"]["content"], "answer");
    assert!(
        !String::from_utf8(non_stream.body)
            .unwrap()
            .contains("<think>")
    );

    let streaming = raw_http(
        address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            &body(true),
            &["Content-Type: application/json"],
        ),
    )
    .await;
    assert_eq!(streaming.status, 200);
    let stream_body = String::from_utf8(streaming.body).unwrap();
    let chunks = stream_body
        .split("\n\n")
        .filter_map(|block| block.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str::<Value>(data).unwrap())
        .collect::<Vec<_>>();
    let reasoning = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["reasoning_content"].as_str())
        .collect::<String>();
    let content = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect::<String>();
    assert_eq!(reasoning, "reason");
    assert_eq!(content, "answer");
    assert!(!stream_body.contains("<think>"));
    assert!(stream_body.ends_with("data: [DONE]\n\n"));

    scheduler.shutdown();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn openwebui_max_tokens_alias_requires_the_compatibility_profile() {
    let make_backend = || -> Arc<dyn ChatGenerationBackendV1> {
        Arc::new(ScriptBackend {
            deltas: vec!["ok".to_owned()],
            finish_reason: FinishReasonV1::Stop,
            usage: TokenUsageV1::new(1, 1).unwrap(),
            fail_after: None,
            cancelled: Arc::new(AtomicUsize::new(0)),
        })
    };
    let body =
        br#"{"model":"qwen-test","messages":[{"role":"user","content":"hi"}],"max_tokens":17}"#;

    let (strict_app, strict_scheduler) = router(make_backend(), 2, 2, None);
    let (strict_address, strict_server) = serve(strict_app).await;
    let strict = raw_http(
        strict_address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            body,
            &["Content-Type: application/json"],
        ),
    )
    .await;
    assert_eq!(strict.status, 400);
    assert_eq!(strict.json()["error"]["param"], "max_tokens");
    strict_scheduler.shutdown();
    strict_server.abort();

    let config = ServerConfigV1::openwebui_compatible(None).unwrap();
    let (compatible_app, compatible_scheduler) = router_with_config(make_backend(), 2, 2, config);
    let (compatible_address, compatible_server) = serve(compatible_app).await;
    let compatible = raw_http(
        compatible_address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            body,
            &["Content-Type: application/json"],
        ),
    )
    .await;
    assert_eq!(compatible.status, 200);
    assert_eq!(compatible.json()["choices"][0]["message"]["content"], "ok");
    compatible_scheduler.shutdown();
    compatible_server.abort();
}

async fn wait_until(timeout: Duration, condition: impl Fn() -> bool) {
    let started = Instant::now();
    while !condition() && started.elapsed() < timeout {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}
