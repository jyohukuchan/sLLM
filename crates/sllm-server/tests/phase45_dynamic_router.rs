use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use sllm_frontend::GenerationCancellationV1;
use sllm_server::{
    BackendCompletionV1, BackendErrorV1, ChatCompletionRequestV1, ChatGenerationBackendV1,
    CredentialRoleV1, CredentialStoreV1, FinishReasonV1, GenerationDeltaSinkV1,
    ModelLifecycleConfigV1, ModelLifecycleDescriptorV1, ModelLifecycleLoadedV1,
    ModelLifecycleRegistryV1, ModelRegistryEntryV1, SchedulerConfigV1, SchedulerV1, ServerConfigV1,
    TokenUsageV1, build_dynamic_router_v1,
};

const FINGERPRINT: &str = "sha256:0000000000000000000000000000000000000000000000000000000000000000";

struct Backend;
impl ChatGenerationBackendV1 for Backend {
    fn generate(
        &self,
        _: &ChatCompletionRequestV1,
        _: &GenerationCancellationV1,
        sink: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        sink.publish("ok")?;
        Ok(BackendCompletionV1 {
            finish_reason: FinishReasonV1::Stop,
            usage: TokenUsageV1::new(1, 1).unwrap(),
            matched_stop: None,
        })
    }
}

fn dynamic_router(always_fail: bool) -> Router {
    let owner = Arc::new(
        ModelRegistryEntryV1::new("dyn", 1, "sllm", FINGERPRINT, Arc::new(Backend)).unwrap(),
    );
    let descriptor = ModelLifecycleDescriptorV1::new(
        "dyn",
        FINGERPRINT,
        "sha256:1111111111111111111111111111111111111111111111111111111111111111",
        "adapter:none-v1",
        1,
    )
    .unwrap();
    let owner_for_loader = Arc::clone(&owner);
    let lifecycle = ModelLifecycleRegistryV1::new_with_fns(
        [descriptor],
        move |descriptor: &ModelLifecycleDescriptorV1| {
            if always_fail {
                Err::<ModelLifecycleLoadedV1, _>(())
            } else {
                Ok(ModelLifecycleLoadedV1::new(
                    Arc::clone(&owner_for_loader),
                    descriptor.declared_resident_bytes(),
                    FINGERPRINT,
                    descriptor.identity().plan_identity(),
                    "adapter:none-v1",
                )
                .unwrap())
            }
        },
        |_loaded: ModelLifecycleLoadedV1| Ok::<_, ()>(()),
        ModelLifecycleConfigV1::new(8).unwrap(),
    )
    .unwrap();
    let scheduler = SchedulerV1::new(SchedulerConfigV1::new(2, 2, Duration::from_secs(2)).unwrap());
    let credentials = CredentialStoreV1::from_keys([
        (CredentialRoleV1::Admin, "admin-key"),
        (CredentialRoleV1::User, "user-key"),
    ])
    .unwrap();
    build_dynamic_router_v1(
        lifecycle,
        scheduler,
        ServerConfigV1::default().with_credentials(credentials),
    )
}

async fn serve(app: Router) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (address, task)
}

fn request(path: &str, headers: &[(&str, &str)], body: &[u8]) -> Vec<u8> {
    let mut value = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: {}\r\n",
        body.len()
    );
    for (name, value_) in headers {
        value.push_str(name);
        value.push_str(": ");
        value.push_str(value_);
        value.push_str("\r\n");
    }
    value.push_str("\r\n");
    let mut request = value.into_bytes();
    request.extend_from_slice(body);
    request
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
        let headers = String::from_utf8_lossy(&response[..split]);
        let status = headers
            .lines()
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        (
            status,
            String::from_utf8_lossy(&response[split + 4..]).into_owned(),
        )
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn dynamic_admin_requires_admin_and_rejects_body() {
    let (address, task) = serve(dynamic_router(false)).await;
    let (status, _) = raw_http(address, request("/admin/models/dyn/preload", &[], &[])).await;
    assert_eq!(status, 401);
    let headers = [("Authorization", "Bearer admin-key")];
    let (status, _) = raw_http(
        address,
        request("/admin/models/dyn/preload", &headers, b"{}"),
    )
    .await;
    assert_eq!(status, 400);
    let (status, _) = raw_http(address, request("/admin/models/dyn/preload", &headers, &[])).await;
    assert_eq!(status, 200);
    let (status, _) = raw_http(address, request("/admin/models/dyn/unload", &headers, &[])).await;
    assert_eq!(status, 200);
    task.abort();
}

#[tokio::test]
async fn unknown_and_quarantined_aliases_are_deterministic_503_or_404() {
    let (address, task) = serve(dynamic_router(true)).await;
    let body = br#"{"model":"missing","max_completion_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#;
    let (status, _response_body) = raw_http(
        address,
        request(
            "/v1/chat/completions",
            &[
                ("Authorization", "Bearer user-key"),
                ("Content-Type", "application/json"),
            ],
            body,
        ),
    )
    .await;
    assert_eq!(status, 404);
    let body =
        br#"{"model":"dyn","max_completion_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#;
    let (status, _) = raw_http(
        address,
        request(
            "/v1/chat/completions",
            &[
                ("Authorization", "Bearer user-key"),
                ("Content-Type", "application/json"),
            ],
            body,
        ),
    )
    .await;
    assert_eq!(status, 503);
    let (status, _) = raw_http(
        address,
        request(
            "/admin/models/dyn/preload",
            &[("Authorization", "Bearer admin-key")],
            &[],
        ),
    )
    .await;
    assert_eq!(status, 503);
    task.abort();
}

#[tokio::test]
async fn multi_choice_loader_failure_preserves_lifecycle_status() {
    let (address, task) = serve(dynamic_router(true)).await;
    let body = br#"{"model":"dyn","n":2,"max_completion_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#;
    let (status, response_body) = raw_http(
        address,
        request(
            "/v1/chat/completions",
            &[
                ("Authorization", "Bearer user-key"),
                ("Content-Type", "application/json"),
            ],
            body,
        ),
    )
    .await;
    assert_eq!(status, 503);
    assert!(response_body.contains("generation_failed"));
    task.abort();
}
