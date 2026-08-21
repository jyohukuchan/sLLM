use std::sync::Arc;
use std::time::Duration;

use sllm_frontend::GenerationCancellationV1;
use sllm_server::{
    BackendCompletionV1, BackendErrorV1, ChatCompletionRequestV1, ChatGenerationBackendV1,
    FinishReasonV1, GenerationDeltaSinkV1, ModelRegistryEntryV1, ModelRegistryV1,
    SchedulerConfigV1, SchedulerV1, ServerConfigV1, TokenUsageV1, build_router_v1,
};

const DEFAULT_ADDRESS: &str = "127.0.0.1:18080";
const MODEL_ALIAS: &str = "qwen-test";
const LOCK_FINGERPRINT: &str =
    "sha256:0000000000000000000000000000000000000000000000000000000000000000";

struct FixtureBackend;

impl ChatGenerationBackendV1 for FixtureBackend {
    fn generate(
        &self,
        _: &ChatCompletionRequestV1,
        cancellation: &GenerationCancellationV1,
        sink: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        if cancellation.is_cancelled() {
            return Err(BackendErrorV1::new("fixture request was cancelled"));
        }
        sink.publish("fixture ")?;
        sink.publish("response")?;
        Ok(BackendCompletionV1 {
            finish_reason: FinishReasonV1::Stop,
            usage: TokenUsageV1::new(3, 2)
                .map_err(|error| BackendErrorV1::new(error.to_string()))?,
            matched_stop: None,
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let address = std::env::var("SLLM_FIXTURE_ADDR").unwrap_or_else(|_| DEFAULT_ADDRESS.to_owned());
    let listener = tokio::net::TcpListener::bind(&address).await?;
    let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(FixtureBackend);
    let entry = ModelRegistryEntryV1::new(
        MODEL_ALIAS,
        1_700_000_000,
        "sllm",
        LOCK_FINGERPRINT,
        backend,
    )?;
    let registry = ModelRegistryV1::new(vec![entry])?;
    let scheduler = SchedulerV1::new(SchedulerConfigV1::new(4, 2, Duration::from_secs(30))?);
    let app = build_router_v1(registry, scheduler.clone(), ServerConfigV1::default());
    let shutdown_scheduler = scheduler.clone();
    let shutdown = async move {
        let _ = tokio::signal::ctrl_c().await;
        shutdown_scheduler.shutdown();
    };

    println!("sLLM profile-v1 fixture listening on http://{address}/v1");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    scheduler.shutdown();
    Ok(())
}
