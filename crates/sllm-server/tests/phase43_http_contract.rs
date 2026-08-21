use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::Router;
use serde_json::{Value, json};
use sllm_frontend::GenerationCancellationV1;
use sllm_server::{
    BackendCompletionV1, BackendErrorV1, ChatCompletionRequestV1, ChatGenerationBackendV1,
    FinishReasonV1, GenerationDeltaSinkV1, ModelRegistryEntryV1, ModelRegistryV1, ResumableStoreV1,
    SchedulerConfigV1, SchedulerV1, ServerConfigV1, TokenUsageV1, build_router_v1,
};

const FINGERPRINT: &str = "sha256:0000000000000000000000000000000000000000000000000000000000000000";

struct ProtocolBackend {
    output: &'static str,
    tool_capable: bool,
    matched_stop: Option<&'static str>,
}

struct FailingBackend;

struct CancellationBackend {
    started: AtomicBool,
    cancelled: AtomicBool,
}

impl CancellationBackend {
    fn new() -> Self {
        Self {
            started: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
        }
    }
}

impl ChatGenerationBackendV1 for CancellationBackend {
    fn generate(
        &self,
        _: &ChatCompletionRequestV1,
        cancellation: &GenerationCancellationV1,
        _: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        self.started.store(true, Ordering::Release);
        while !cancellation.is_cancelled() {
            std::thread::sleep(Duration::from_millis(1));
        }
        self.cancelled.store(true, Ordering::Release);
        Err(BackendErrorV1::new("cancelled"))
    }
}

impl ChatGenerationBackendV1 for FailingBackend {
    fn generate(
        &self,
        _: &ChatCompletionRequestV1,
        _: &GenerationCancellationV1,
        sink: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        sink.publish("untrusted partial output")?;
        Err(BackendErrorV1::new(
            "SECRET backend path /models/private and tool arguments",
        ))
    }
}

impl ChatGenerationBackendV1 for ProtocolBackend {
    fn tool_protocol_v1_available(&self) -> bool {
        self.tool_capable
    }

    fn generate(
        &self,
        _: &ChatCompletionRequestV1,
        cancellation: &GenerationCancellationV1,
        sink: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        if cancellation.is_cancelled() {
            return Err(BackendErrorV1::new("cancelled"));
        }
        sink.publish(self.output)?;
        Ok(BackendCompletionV1 {
            finish_reason: FinishReasonV1::Stop,
            usage: TokenUsageV1::new(3, 2).unwrap(),
            matched_stop: self.matched_stop.map(str::to_owned),
        })
    }
}

fn entry(alias: &str, output: &'static str, tool_capable: bool) -> ModelRegistryEntryV1 {
    let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(ProtocolBackend {
        output,
        tool_capable,
        matched_stop: None,
    });
    ModelRegistryEntryV1::new(alias, 1_700_000_000, "sllm", FINGERPRINT, backend).unwrap()
}

fn router(resumable: bool) -> (Router, SchedulerV1) {
    let mut entries = vec![
        entry("text", "ok", true),
        entry(
            "tool",
            r#"{"type":"tool_calls","calls":[{"name":"lookup","arguments":{"q":"Tokyo"}}]}"#,
            true,
        ),
        entry(
            "dangerous-tool",
            r#"{"type":"tool_calls","calls":[{"name":"lookup","arguments":{"command":"touch /tmp/sllm-phase43-must-not-exist"}}]}"#,
            true,
        ),
        entry(
            "parallel",
            r#"{"type":"tool_calls","calls":[{"name":"lookup","arguments":{"q":"a"}},{"name":"lookup","arguments":{"q":"b"}}]}"#,
            true,
        ),
        entry("no-tools", "ok", false),
    ];
    let stop_backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(ProtocolBackend {
        output: "ok",
        tool_capable: true,
        matched_stop: Some("--"),
    });
    entries.push(
        ModelRegistryEntryV1::new("stop", 1_700_000_000, "sllm", FINGERPRINT, stop_backend)
            .unwrap(),
    );
    let failing: Arc<dyn ChatGenerationBackendV1> = Arc::new(FailingBackend);
    entries.push(
        ModelRegistryEntryV1::new("failure", 1_700_000_000, "sllm", FINGERPRINT, failing).unwrap(),
    );
    let registry = ModelRegistryV1::new(entries).unwrap();
    let scheduler = SchedulerV1::new(SchedulerConfigV1::new(8, 8, Duration::from_secs(5)).unwrap());
    let config = if resumable {
        ServerConfigV1::default().with_resumable_store(ResumableStoreV1::new(8, 64).unwrap())
    } else {
        ServerConfigV1::default()
    };
    (
        build_router_v1(registry, scheduler.clone(), config),
        scheduler,
    )
}

fn cancellation_router() -> (Router, SchedulerV1, Arc<CancellationBackend>) {
    let backend = Arc::new(CancellationBackend::new());
    let erased: Arc<dyn ChatGenerationBackendV1> = backend.clone();
    let entry =
        ModelRegistryEntryV1::new("cancel", 1_700_000_000, "sllm", FINGERPRINT, erased).unwrap();
    let registry = ModelRegistryV1::new(vec![entry]).unwrap();
    let scheduler = SchedulerV1::new(SchedulerConfigV1::new(1, 1, Duration::from_secs(5)).unwrap());
    (
        build_router_v1(registry, scheduler.clone(), ServerConfigV1::default()),
        scheduler,
        backend,
    )
}

async fn serve(app: Router) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (address, task)
}

fn request(method: &str, path: &str, headers: &[(&str, &str)], body: &[u8]) -> Vec<u8> {
    let mut value = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: {}\r\n",
        body.len()
    );
    for (name, value_) in headers {
        value.push_str(name);
        value.push_str(": ");
        value.push_str(value_);
        value.push_str("\r\n");
    }
    value.push_str("\r\n");
    let mut value = value.into_bytes();
    value.extend_from_slice(body);
    value
}

fn json_request(path: &str, body: &[u8]) -> Vec<u8> {
    request("POST", path, &[("Content-Type", "application/json")], body)
}

fn anthropic_request(body: &[u8]) -> Vec<u8> {
    request(
        "POST",
        "/v1/messages",
        &[
            ("Content-Type", "application/json"),
            ("anthropic-version", "2023-06-01"),
        ],
        body,
    )
}

async fn raw_http(address: SocketAddr, request: Vec<u8>) -> (u16, String) {
    tokio::task::spawn_blocking(move || {
        let mut socket = TcpStream::connect(address).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        socket.write_all(&request).unwrap();
        let mut response = Vec::new();
        socket.read_to_end(&mut response).unwrap();
        let split = response
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .unwrap();
        let headers = String::from_utf8_lossy(&response[..split]).into_owned();
        let status = headers
            .lines()
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let raw_body = &response[split + 4..];
        let body = if headers
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked")
        {
            String::from_utf8_lossy(&decode_chunked(raw_body)).into_owned()
        } else {
            String::from_utf8_lossy(raw_body).into_owned()
        };
        (status, body)
    })
    .await
    .unwrap()
}

fn decode_chunked(mut input: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    loop {
        let line_end = input
            .windows(2)
            .position(|window| window == b"\r\n")
            .unwrap();
        let size = usize::from_str_radix(
            std::str::from_utf8(&input[..line_end])
                .unwrap()
                .split(';')
                .next()
                .unwrap(),
            16,
        )
        .unwrap();
        input = &input[line_end + 2..];
        if size == 0 {
            break;
        }
        output.extend_from_slice(&input[..size]);
        input = &input[size + 2..];
    }
    output
}

fn tool(name: &str) -> Value {
    json!({
        "type": "function",
        "name": name,
        "description": "read-only fixture",
        "parameters": {
            "type": "object",
            "properties": {
                "q": {"type": "string"},
                "command": {"type": "string"}
            },
            "additionalProperties": false
        }
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn phase43_nonstream_profiles_tool_roundtrip_and_no_execution() {
    let (app, scheduler) = router(false);
    let (address, server) = serve(app).await;

    let (status, body) = raw_http(
        address,
        json_request(
            "/v1/responses",
            br#"{"model":"text","input":"hello","store":false}"#,
        ),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let response: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(response["object"], "response");
    assert_eq!(response["model"], "text");
    assert_eq!(response["output_text"], "ok");
    assert!(response["id"].as_str().unwrap().starts_with("resp_"));
    assert_eq!(response["usage"]["total_tokens"], 5);

    let tool_body = json!({
        "model":"tool", "input":"look up Tokyo", "store":false,
        "tools":[tool("lookup")], "tool_choice":"required",
        "parallel_tool_calls":false
    });
    let (status, body) = raw_http(
        address,
        json_request("/v1/responses", &serde_json::to_vec(&tool_body).unwrap()),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let response: Value = serde_json::from_str(&body).unwrap();
    let call = &response["output"][0];
    assert_eq!(call["type"], "function_call");
    assert_eq!(call["name"], "lookup");
    assert_eq!(call["arguments"], r#"{"q":"Tokyo"}"#);
    let call_id = call["call_id"].as_str().unwrap();

    let roundtrip = json!({
        "model":"tool", "store":false,
        "input":[
            {"type":"message","role":"user","content":[{"type":"input_text","text":"look up Tokyo"}]},
            {"type":"function_call","call_id":call_id,"name":"lookup","arguments":"{\"q\":\"Tokyo\"}"},
            {"type":"function_call_output","call_id":call_id,"output":"sunny"}
        ],
        "tools":[tool("lookup")], "tool_choice":"required",
        "parallel_tool_calls":false
    });
    let (status, body) = raw_http(
        address,
        json_request("/v1/responses", &serde_json::to_vec(&roundtrip).unwrap()),
    )
    .await;
    assert_eq!(status, 200, "{body}");

    let unsupported = json!({
        "model":"no-tools", "input":"x", "store":false,
        "tools":[tool("lookup")]
    });
    let (status, body) = raw_http(
        address,
        json_request("/v1/responses", &serde_json::to_vec(&unsupported).unwrap()),
    )
    .await;
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("unsupported_parameter"));

    let misplaced_system = json!({
        "model":"text", "store":false,
        "input":[
            {"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]},
            {"type":"message","role":"system","content":[{"type":"input_text","text":"late"}]}
        ]
    });
    let (status, body) = raw_http(
        address,
        json_request(
            "/v1/responses",
            &serde_json::to_vec(&misplaced_system).unwrap(),
        ),
    )
    .await;
    assert_eq!(status, 400, "{body}");
    assert!(!body.contains("late"));

    let marker = PathBuf::from("/tmp/sllm-phase43-must-not-exist");
    let _ = fs::remove_file(&marker);
    let dangerous = json!({
        "model":"dangerous-tool", "input":"x", "store":false,
        "tools":[tool("lookup")], "tool_choice":"required"
    });
    let (status, body) = raw_http(
        address,
        json_request("/v1/responses", &serde_json::to_vec(&dangerous).unwrap()),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("touch /tmp/sllm-phase43-must-not-exist"));
    assert!(
        !marker.exists(),
        "generated arguments must remain inert data"
    );

    let (status, body) = raw_http(
        address,
        json_request(
            "/v1/messages",
            br#"{"model":"text","max_tokens":8,"messages":[{"role":"user","content":"hello"}]}"#,
        ),
    )
    .await;
    assert_eq!(status, 400, "{body}");
    let error: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(error["type"], "error");
    assert_eq!(error["error"]["type"], "invalid_request_error");

    let (status, body) = raw_http(
        address,
        anthropic_request(
            br#"{"model":"stop","max_tokens":8,"stop_sequences":["--"],"messages":[{"role":"user","content":"hello"}]}"#,
        ),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let message: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(message["type"], "message");
    assert_eq!(message["model"], "stop");
    assert_eq!(message["stop_reason"], "stop_sequence");
    assert_eq!(message["stop_sequence"], "--");

    let anthropic_tool = json!({
        "model":"tool", "max_tokens":8,
        "messages":[{"role":"user","content":"look up Tokyo"}],
        "tools":[{"name":"lookup","description":"read-only fixture","input_schema":{"type":"object","properties":{"q":{"type":"string"}},"additionalProperties":false}}],
        "tool_choice":{"type":"any","disable_parallel_tool_use":true}
    });
    let (status, body) = raw_http(
        address,
        anthropic_request(&serde_json::to_vec(&anthropic_tool).unwrap()),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let message: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(message["content"][0]["type"], "tool_use");
    assert_eq!(message["stop_reason"], "tool_use");

    let sequential = json!({
        "model":"parallel", "max_tokens":8,
        "messages":[{"role":"user","content":"twice"}],
        "tools":[{"name":"lookup","input_schema":{"type":"object","properties":{"q":{"type":"string"}},"additionalProperties":false}}],
        "tool_choice":{"type":"any","disable_parallel_tool_use":true}
    });
    let (status, body) = raw_http(
        address,
        anthropic_request(&serde_json::to_vec(&sequential).unwrap()),
    )
    .await;
    assert_eq!(status, 500, "{body}");
    assert!(body.contains("api_error"));

    let (status, body) = raw_http(
        address,
        json_request(
            "/v1/responses",
            br#"{"model":"failure","input":"hello","store":false}"#,
        ),
    )
    .await;
    assert_eq!(status, 500, "{body}");
    assert!(body.contains("generation failed"));
    assert!(!body.contains("SECRET"));
    assert!(!body.contains("/models/private"));

    scheduler.shutdown();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn phase43_named_streams_and_resumable_replay_are_ordered() {
    let (app, scheduler) = router(true);
    let (address, server) = serve(app).await;

    let (status, body) = raw_http(
        address,
        json_request(
            "/v1/responses",
            br#"{"model":"text","input":"hello","store":false,"stream":true}"#,
        ),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let created = body.find("event: response.created").unwrap();
    let delta = body.find("event: response.output_text.delta").unwrap();
    let completed = body.find("event: response.completed").unwrap();
    assert!(created < delta && delta < completed);
    assert!(!body.contains("[DONE]"));

    let (status, body) = raw_http(
        address,
        json_request(
            "/v1/responses",
            br#"{"model":"failure","input":"hello","store":false,"stream":true}"#,
        ),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("event: error"));
    assert!(!body.contains("response.completed"));
    assert!(!body.contains("SECRET"));
    assert!(!body.contains("untrusted partial output"));

    let (status, body) = raw_http(
        address,
        anthropic_request(
            br#"{"model":"text","max_tokens":8,"messages":[{"role":"user","content":"hello"}],"stream":true}"#,
        ),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let start = body.find("event: message_start").unwrap();
    let delta = body.find("event: content_block_delta").unwrap();
    let stop = body.find("event: message_stop").unwrap();
    assert!(start < delta && delta < stop);
    assert!(!body.contains("[DONE]"));

    let (status, body) = raw_http(
        address,
        json_request(
            "/v1/responses",
            br#"{"model":"text","input":"hello","store":false,"max_output_tokens":41,"stream":true,"sllm":{"resumable":true}}"#,
        ),
    )
    .await;
    assert_eq!(status, 400, "{body}");

    let (status, body) = raw_http(
        address,
        json_request(
            "/v1/responses",
            br#"{"model":"text","input":"hello","store":false,"max_output_tokens":40,"stream":true,"sllm":{"resumable":true}}"#,
        ),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("id: 1\n"));
    let response_id = body
        .lines()
        .find_map(|line| {
            let data = line.strip_prefix("data: ")?;
            let value: Value = serde_json::from_str(data).ok()?;
            (value["type"] == "response.created")
                .then(|| value["response"]["id"].as_str().map(str::to_owned))
                .flatten()
        })
        .unwrap();
    let replay_path = format!("/v1/responses/{response_id}/events");
    let (status, replay) = raw_http(
        address,
        request("GET", &replay_path, &[("Last-Event-ID", "1")], b""),
    )
    .await;
    assert_eq!(status, 200, "{replay}");
    assert!(!replay.contains("id: 1\n"));
    assert!(replay.contains("event: response.completed"));

    let (status, body) = raw_http(
        address,
        request(
            "GET",
            &replay_path,
            &[("Last-Event-ID", "not-a-number")],
            b"",
        ),
    )
    .await;
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("Last-Event-ID"));

    scheduler.shutdown();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn phase43_stream_disconnect_cancels_the_shared_scheduler_request() {
    let (app, scheduler, backend) = cancellation_router();
    let (address, server) = serve(app).await;
    let request = json_request(
        "/v1/responses",
        br#"{"model":"cancel","input":"hello","store":false,"stream":true}"#,
    );
    let (drop_tx, drop_rx) = std::sync::mpsc::channel();
    let client = tokio::task::spawn_blocking(move || {
        let mut socket = TcpStream::connect(address).unwrap();
        socket.write_all(&request).unwrap();
        drop_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        drop(socket);
    });

    for _ in 0..200 {
        if backend.started.load(Ordering::Acquire) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(backend.started.load(Ordering::Acquire));
    drop_tx.send(()).unwrap();
    client.await.unwrap();
    for _ in 0..200 {
        if backend.cancelled.load(Ordering::Acquire) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(
        backend.cancelled.load(Ordering::Acquire),
        "dropping a Phase43 SSE consumer must cancel generation"
    );

    scheduler.shutdown();
    server.abort();
}
