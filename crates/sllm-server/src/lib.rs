//! OpenAI-compatible Chat Completions profile-v1 server.
//!
//! The HTTP layer owns strict wire validation, admission, response framing,
//! and transport cancellation. Model execution remains behind a synchronous
//! backend trait so one bounded worker can own the initial single-GPU runtime.

mod api;
mod batching;
mod lifecycle;
mod metrics;
mod phase42_api;
mod production;
mod resume;
mod runtime;
mod security;
mod service;

pub use api::{
    ApiErrorV1, ChatCompatibilityProfileV1, ChatCompletionRequestV1, ChatMessageV1,
    DrySamplingConfigV1, DynamicTemperatureConfigV1, ErrorCodeV1, FinishReasonV1,
    JsonSchemaFormatV1, LogitBiasV1, LogprobOptionsV1, MirostatSamplingConfigV1,
    ReasoningOptionsV1, ResponseFormatV1, SamplerExtensionConfigV1, TokenUsageV1,
    XtcSamplingConfigV1,
};
pub use lifecycle::{ServerLifecycleStateV1, ServerLifecycleV1};
pub use metrics::{
    CancellationReasonV1, HttpEndpointV1, MetricsConfigError, MetricsRequestHandleV1,
    RequestOutcomeV1, ServerMetricsV1,
};
pub use phase42_api::{
    ApplyTemplateRequestV1, CompletionRequestV1, DetokenizeRequestV1, EmbeddingEncodingFormatV1,
    EmbeddingRequestV1, InfillRequestV1, InputTokensInputV1, InputTokensRequestV1,
    PHASE42_PROFILE_VERSION, PromptV1, RerankRequestV1, TemplateMessageV1, TemplateRoleV1,
    TokenizeRequestV1,
};
pub use production::{
    CheckpointStartupConfigV1, ContextWindowStartupConfigV1, DraftStartupConfigV1,
    Gemma4BackendConfigV1, Gemma4ChatBackendV1, Phase41ProductionConfigV1,
    PrefixCacheStartupConfigV1, ProductionCheckpointOperationV1, ProductionCheckpointResultV1,
    ProductionDraftProviderV1, ProductionPhase41AuditV1, ProductionPrefixCacheResultV1,
    ProductionRequestAuditV1, ProductionShutdownAuditV1, QwenBackendConfigV1, QwenChatBackendV1,
};
pub use resume::{ReplayErrorV1, ReplayEventV1, ReplayReadV1, ResumableStoreV1};
pub use runtime::{
    BackendCompletionV1, BackendEmbeddingBatchV1, BackendEmbeddingInputV1,
    BackendEmbeddingRequestV1, BackendEmbeddingVectorV1, BackendErrorV1, BackendInfillCapabilityV1,
    BackendMemoryCategorySnapshotV1, BackendObservabilitySnapshotV1, BackendTokenLogprobV1,
    BackendTopLogprobV1, ChatGenerationBackendV1, GenerationDeltaSinkV1, ModelRegistryEntryV1,
    ModelRegistryV1, SchedulerConfigV1, SchedulerSlotSnapshotV1, SchedulerSlotStateV1,
    SchedulerSnapshotV1, SchedulerV1,
};
pub use security::{CredentialErrorV1, CredentialRoleV1, CredentialStoreV1};
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
