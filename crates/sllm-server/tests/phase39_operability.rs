//! Host-only Phase 39 operability coverage.
//!
//! These tests intentionally use a deterministic backend and a loopback HTTP
//! listener.  They exercise the deployment surface without requiring a model,
//! GPU, or external credentials.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Router;
use serde_json::Value;
use sllm_frontend::GenerationCancellationV1;
use sllm_server::{
    BackendCompletionV1, BackendErrorV1, ChatCompletionRequestV1, ChatGenerationBackendV1,
    CredentialRoleV1, CredentialStoreV1, FinishReasonV1, GenerationDeltaSinkV1,
    ModelRegistryEntryV1, ModelRegistryV1, ResumableStoreV1, SchedulerConfigV1, SchedulerV1,
    ServerConfigV1, ServerLifecycleStateV1, ServerLifecycleV1, ServerMetricsV1, TokenUsageV1,
    build_router_v1,
};

const FINGERPRINT: &str = "sha256:0000000000000000000000000000000000000000000000000000000000000000";

struct ScriptBackend {
    deltas: Vec<String>,
    delay: Duration,
}

impl ChatGenerationBackendV1 for ScriptBackend {
    fn generate(
        &self,
        _: &ChatCompletionRequestV1,
        cancellation: &GenerationCancellationV1,
        sink: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        for delta in &self.deltas {
            if cancellation.is_cancelled() {
                return Err(BackendErrorV1::new("cancelled"));
            }
            if !self.delay.is_zero() {
                thread::sleep(self.delay);
            }
            sink.publish(delta)?;
        }
        Ok(BackendCompletionV1 {
            finish_reason: FinishReasonV1::Stop,
            usage: TokenUsageV1::new(2, self.deltas.len() as u64).unwrap(),
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
        })
    }
}

fn registry(backend: Arc<dyn ChatGenerationBackendV1>) -> ModelRegistryV1 {
    ModelRegistryV1::new(vec![
        ModelRegistryEntryV1::new("qwen-test", 1_700_000_000, "sllm", FINGERPRINT, backend)
            .unwrap(),
    ])
    .unwrap()
}

fn make_scheduler(queue_capacity: usize) -> SchedulerV1 {
    SchedulerV1::new(SchedulerConfigV1::new(queue_capacity, 4, Duration::from_secs(5)).unwrap())
}

fn body(stream: bool, resumable: bool) -> Vec<u8> {
    format!(
        r#"{{"model":"qwen-test","messages":[{{"role":"user","content":"hello"}}],"stream":{stream},"max_completion_tokens":17,"sllm":{{"resumable":{resumable}}}}}"#
    )
    .into_bytes()
}

async fn serve(app: Router) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (address, task)
}

fn request_bytes(method: &str, path: &str, body: &[u8], headers: &[&str]) -> Vec<u8> {
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: {}\r\n",
        body.len()
    );
    for header in headers {
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

    fn text(&self) -> String {
        String::from_utf8(self.body.clone()).unwrap()
    }

    fn has_header(&self, name: &str, value: &str) -> bool {
        self.headers.lines().any(|line| {
            line.split_once(':').is_some_and(|(key, actual)| {
                key.eq_ignore_ascii_case(name) && actual.trim() == value
            })
        })
    }
}

fn decode_chunked(mut input: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    loop {
        let line_end = input
            .windows(2)
            .position(|window| window == b"\r\n")
            .unwrap();
        let size_text = std::str::from_utf8(
            input[..line_end]
                .split(|byte| *byte == b';')
                .next()
                .unwrap(),
        )
        .unwrap();
        let size = usize::from_str_radix(size_text, 16).unwrap();
        input = &input[line_end + 2..];
        if size == 0 {
            break;
        }
        output.extend_from_slice(&input[..size]);
        input = &input[size + 2..];
    }
    output
}

async fn wait_until<F>(timeout: Duration, mut predicate: F)
where
    F: FnMut() -> bool,
{
    let deadline = tokio::time::Instant::now() + timeout;
    while !predicate() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "condition did not become true"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn admin_headers() -> [&'static str; 1] {
    ["Authorization: Bearer admin-key"]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn health_readiness_metrics_and_props_are_bounded_and_authenticated() {
    let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(ScriptBackend {
        deltas: vec!["safe".to_owned()],
        delay: Duration::ZERO,
    });
    let lifecycle = ServerLifecycleV1::new(ServerLifecycleStateV1::Loading);
    let credentials = CredentialStoreV1::from_keys([
        (CredentialRoleV1::User, "user-key"),
        (CredentialRoleV1::Admin, "admin-key"),
    ])
    .unwrap();
    let config = ServerConfigV1::new(None)
        .unwrap()
        .with_credentials(credentials)
        .with_lifecycle(lifecycle.clone())
        .with_metrics(ServerMetricsV1::new(["qwen-test"]).unwrap());
    let scheduler = make_scheduler(2);
    let (address, server) = serve(build_router_v1(
        registry(backend),
        scheduler.clone(),
        config,
    ))
    .await;

    let health = raw_http(address, request_bytes("GET", "/healthz", b"", &[])).await;
    assert_eq!(health.status, 200);
    assert_eq!(health.json()["state"], "loading");
    let not_ready = raw_http(address, request_bytes("GET", "/readyz", b"", &[])).await;
    assert_eq!(not_ready.status, 503);
    let unauth_props = raw_http(address, request_bytes("GET", "/props", b"", &[])).await;
    assert_eq!(unauth_props.status, 401);
    let props = raw_http(
        address,
        request_bytes("GET", "/props", b"", &admin_headers()),
    )
    .await;
    assert_eq!(props.status, 200);
    assert_eq!(props.json()["schema_version"], "sllm-server-props-v1");
    lifecycle.transition(ServerLifecycleStateV1::Ready);
    let ready = raw_http(address, request_bytes("GET", "/readyz", b"", &[])).await;
    assert_eq!(ready.status, 200);

    let response = raw_http(
        address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            &body(false, false),
            &[
                "Authorization: Bearer user-key",
                "Content-Type: application/json",
            ],
        ),
    )
    .await;
    assert_eq!(response.status, 200);
    let metrics = raw_http(
        address,
        request_bytes("GET", "/metrics", b"", &admin_headers()),
    )
    .await;
    assert_eq!(metrics.status, 200);
    let text = metrics.text();
    assert!(text.contains("sllm_http_responses_total"));
    assert!(text.contains("sllm_requests_total"));
    assert!(text.contains("sllm_model_ready{model=\"qwen-test\"} 1"));
    assert!(!text.contains("hello"));
    assert!(!text.contains("user-key"));

    scheduler.shutdown();
    server.abort();

    let disabled_scheduler = make_scheduler(1);
    let disabled = build_router_v1(
        registry(Arc::new(ScriptBackend {
            deltas: vec!["x".to_owned()],
            delay: Duration::ZERO,
        })),
        disabled_scheduler.clone(),
        ServerConfigV1::default(),
    );
    let (disabled_address, disabled_server) = serve(disabled).await;
    let disabled_metrics =
        raw_http(disabled_address, request_bytes("GET", "/metrics", b"", &[])).await;
    assert_eq!(disabled_metrics.status, 404);
    disabled_scheduler.shutdown();
    disabled_server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn admin_slots_list_and_cancel_queued_and_active_requests() {
    let backend = Arc::new(BlockingBackend::new());
    let erased: Arc<dyn ChatGenerationBackendV1> = backend.clone();
    let scheduler = make_scheduler(2);
    let config = ServerConfigV1::new(None).unwrap().with_credentials(
        CredentialStoreV1::from_keys([(CredentialRoleV1::Admin, "admin-key")]).unwrap(),
    );
    let (address, server) =
        serve(build_router_v1(registry(erased), scheduler.clone(), config)).await;
    let request = || {
        request_bytes(
            "POST",
            "/v1/chat/completions",
            &body(false, false),
            &[
                "Content-Type: application/json",
                "Authorization: Bearer admin-key",
            ],
        )
    };

    let first = tokio::spawn(raw_http(address, request()));
    wait_until(Duration::from_secs(1), || {
        backend.started.load(Ordering::Acquire) == 1
    })
    .await;
    let second = tokio::spawn(raw_http(address, request()));
    wait_until(Duration::from_secs(1), || {
        scheduler.snapshot().slots.len() == 2
    })
    .await;
    let slots = raw_http(
        address,
        request_bytes("GET", "/slots", b"", &admin_headers()),
    )
    .await;
    assert_eq!(slots.status, 200);
    let slots_json = slots.json();
    let values = slots_json["slots"].as_array().unwrap();
    assert_eq!(values.len(), 2);
    let queued_id = values
        .iter()
        .find(|value| value["state"] == "queued")
        .and_then(|value| value["id"].as_u64())
        .unwrap();
    let active_id = values
        .iter()
        .find(|value| value["state"] == "active")
        .and_then(|value| value["id"].as_u64())
        .unwrap();
    let cancel = raw_http(
        address,
        request_bytes(
            "POST",
            &format!("/admin/slots/{queued_id}/cancel"),
            b"",
            &admin_headers(),
        ),
    )
    .await;
    assert_eq!(cancel.status, 200);
    assert_eq!(cancel.json()["state"], "cancelled");

    let cancel_active = raw_http(
        address,
        request_bytes(
            "POST",
            &format!("/admin/slots/{active_id}/cancel"),
            b"",
            &admin_headers(),
        ),
    )
    .await;
    assert_eq!(cancel_active.status, 200);
    assert_eq!(first.await.unwrap().status, 409);
    assert_eq!(second.await.unwrap().status, 409);
    wait_until(Duration::from_secs(1), || {
        backend.cancelled.load(Ordering::Acquire) >= 1
    })
    .await;
    scheduler.shutdown();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cors_uses_an_exact_allowlist_and_preflight() {
    let scheduler = make_scheduler(1);
    let config = ServerConfigV1::default()
        .with_cors_origins(["https://allowed.example"])
        .unwrap();
    let app = build_router_v1(
        registry(Arc::new(ScriptBackend {
            deltas: vec!["x".to_owned()],
            delay: Duration::ZERO,
        })),
        scheduler.clone(),
        config,
    );
    let (address, server) = serve(app).await;
    let allowed = raw_http(
        address,
        request_bytes(
            "OPTIONS",
            "/v1/models",
            b"",
            &[
                "Origin: https://allowed.example",
                "Access-Control-Request-Method: GET",
                "Access-Control-Request-Headers: authorization",
            ],
        ),
    )
    .await;
    assert!(allowed.status == 200 || allowed.status == 204);
    assert!(allowed.has_header("access-control-allow-origin", "https://allowed.example"));
    let denied = raw_http(
        address,
        request_bytes(
            "GET",
            "/v1/models",
            b"",
            &["Origin: https://denied.example"],
        ),
    )
    .await;
    assert!(!denied.has_header("access-control-allow-origin", "https://denied.example"));
    scheduler.shutdown();
    server.abort();
}

fn sse_ids(text: &str) -> Vec<u64> {
    text.lines()
        .filter_map(|line| line.strip_prefix("id: "))
        .map(|id| id.parse::<u64>().unwrap())
        .collect()
}

fn chat_id(text: &str) -> String {
    text.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .find_map(|data| {
            serde_json::from_str::<Value>(data)
                .ok()
                .and_then(|value| value["id"].as_str().map(str::to_owned))
        })
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn resumable_sse_reconnects_without_duplicates_and_rejects_bad_cursors() {
    let scheduler = make_scheduler(2);
    let replay = ResumableStoreV1::new(4, 16).unwrap();
    let app = build_router_v1(
        registry(Arc::new(ScriptBackend {
            deltas: vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
            delay: Duration::from_millis(2),
        })),
        scheduler.clone(),
        ServerConfigV1::default().with_resumable_store(replay.clone()),
    );
    let (address, server) = serve(app).await;
    let initial = raw_http(
        address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            &body(true, true),
            &["Content-Type: application/json"],
        ),
    )
    .await;
    assert_eq!(initial.status, 200);
    let initial_text = initial.text();
    let ids = sse_ids(&initial_text);
    assert!(ids.len() >= 6);
    let mut unique = ids.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), ids.len());
    let id = chat_id(&initial_text);
    let reconnect = raw_http(
        address,
        request_bytes(
            "GET",
            &format!("/v1/chat/completions/{id}/events"),
            b"",
            &["Last-Event-ID: 1"],
        ),
    )
    .await;
    assert_eq!(reconnect.status, 200);
    let reconnect_ids = sse_ids(&reconnect.text());
    assert!(!reconnect_ids.is_empty());
    assert!(reconnect_ids.iter().all(|event_id| *event_id > 1));
    let unknown = raw_http(
        address,
        request_bytes("GET", "/v1/chat/completions/unknown/events", b"", &[]),
    )
    .await;
    assert_eq!(unknown.status, 404);

    let manual = "manual-phase39";
    replay.create(manual).unwrap();
    for value in 1..=17 {
        replay
            .append(manual, value.to_string(), value == 17)
            .unwrap();
    }
    let out_of_range = raw_http(
        address,
        request_bytes(
            "GET",
            &format!("/v1/chat/completions/{manual}/events"),
            b"",
            &["Last-Event-ID: 0"],
        ),
    )
    .await;
    assert_eq!(out_of_range.status, 416);
    scheduler.shutdown();
    server.abort();

    let disabled_scheduler = make_scheduler(1);
    let disabled = build_router_v1(
        registry(Arc::new(ScriptBackend {
            deltas: vec!["x".to_owned()],
            delay: Duration::ZERO,
        })),
        disabled_scheduler.clone(),
        ServerConfigV1::default(),
    );
    let (disabled_address, disabled_server) = serve(disabled).await;
    let response = raw_http(
        disabled_address,
        request_bytes(
            "POST",
            "/v1/chat/completions",
            &body(true, true),
            &["Content-Type: application/json"],
        ),
    )
    .await;
    assert_eq!(response.status, 400);
    disabled_scheduler.shutdown();
    disabled_server.abort();
}

#[test]
fn credential_roles_and_key_file_reload_are_separated() {
    let store = CredentialStoreV1::from_keys([
        (CredentialRoleV1::User, "user-key"),
        (CredentialRoleV1::Admin, "admin-key"),
    ])
    .unwrap();
    assert!(store.authorize_user(Some("Bearer user-key")));
    assert!(!store.authorize_admin(Some("Bearer user-key")));
    assert!(store.authorize_admin(Some("Bearer admin-key")));

    let path = std::env::temp_dir().join(format!(
        "sllm-phase39-keys-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&path, "admin:old-admin\nuser:old-user\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    let reloaded = CredentialStoreV1::from_key_file(&path).unwrap();
    assert!(reloaded.authorize_admin(Some("Bearer old-admin")));
    std::fs::write(&path, "admin:new-admin\n").unwrap();
    reloaded.reload().unwrap();
    assert!(!reloaded.authorize_admin(Some("Bearer old-admin")));
    assert!(reloaded.authorize_admin(Some("Bearer new-admin")));
    let _ = std::fs::remove_file(path);
}
