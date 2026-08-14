//! OpenAI-compatible Chat Completions profile-v1 server.
//!
//! The HTTP layer owns strict wire validation, admission, response framing,
//! and transport cancellation. Model execution remains behind a synchronous
//! backend trait so one bounded worker can own the initial single-GPU runtime.

mod api;
mod production;
mod runtime;
mod service;

pub use api::{
    ApiErrorV1, ChatCompletionRequestV1, ChatMessageV1, ErrorCodeV1, FinishReasonV1, TokenUsageV1,
};
pub use production::{
    ProductionRequestAuditV1, ProductionShutdownAuditV1, QwenBackendConfigV1, QwenChatBackendV1,
};
pub use runtime::{
    BackendCompletionV1, BackendErrorV1, ChatGenerationBackendV1, GenerationDeltaSinkV1,
    ModelRegistryEntryV1, ModelRegistryV1, SchedulerConfigV1, SchedulerV1,
};
pub use service::{ServerConfigV1, build_router_v1};

/// Version of the server dependency and ownership boundary fixed in Phase 6.
pub const SERVER_RUNTIME_CONTRACT_VERSION: u32 = 1;

#[cfg(test)]
mod tests {
    use super::SERVER_RUNTIME_CONTRACT_VERSION;

    #[test]
    fn runtime_contract_version_is_nonzero() {
        assert_ne!(SERVER_RUNTIME_CONTRACT_VERSION, 0);
    }
}
