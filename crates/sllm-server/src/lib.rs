//! OpenAI-compatible Chat Completions profile-v1 server.
//!
//! The HTTP layer owns strict wire validation, admission, response framing,
//! and transport cancellation. Model execution remains behind a synchronous
//! backend trait so one bounded worker can own the initial single-GPU runtime.

mod api;
mod batching;
mod lifecycle;
mod metrics;
mod model_lifecycle;
mod model_manifest;
mod phase42_api;
mod phase43_api;
mod phase43_service;
mod phase43_transport;
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
pub use model_lifecycle::{
    MAX_CONFIGURED_ALIASES_V1, MAX_IDENTITY_BYTES_V1, MAX_LOADED_MODELS_V1, ModelLifecycleConfigV1,
    ModelLifecycleDescriptorV1, ModelLifecycleErrorV1, ModelLifecycleIdentityV1,
    ModelLifecycleLeaseV1, ModelLifecycleLoadedV1, ModelLifecycleLoaderErrorV1,
    ModelLifecycleLoaderFnsV1, ModelLifecycleLoaderV1, ModelLifecycleRegistryV1,
    ModelLifecycleSnapshotV1, ModelLifecycleStateV1, model_lifecycle_loader_from_fns,
};
pub use model_manifest::{
    MAX_MODEL_MANIFEST_ADAPTERS_V1, MAX_MODEL_MANIFEST_ALIAS_BYTES_V1,
    MAX_MODEL_MANIFEST_ARTIFACTS_V1, MAX_MODEL_MANIFEST_BYTES_V1,
    MAX_MODEL_MANIFEST_CONTROL_VECTORS_V1, MAX_MODEL_MANIFEST_MODELS_V1,
    MODEL_MANIFEST_SCHEMA_VERSION_V1, ModelArtifactManifestV1, ModelManifestEntryV1,
    ModelManifestErrorV1, ModelManifestV1, parse_model_manifest_v1, read_model_manifest_v1,
};
pub use phase42_api::{
    ApplyTemplateRequestV1, CompletionRequestV1, DetokenizeRequestV1, EmbeddingEncodingFormatV1,
    EmbeddingRequestV1, InfillRequestV1, InputTokensInputV1, InputTokensRequestV1,
    PHASE42_PROFILE_VERSION, PromptV1, RerankRequestV1, TemplateMessageV1, TemplateRoleV1,
    TokenizeRequestV1,
};
pub use phase43_api::{
    ANTHROPIC_API_VERSION_V1, AnthropicContentBlockV1, AnthropicMessageV1,
    AnthropicMessagesRequestV1, AnthropicRoleV1, AnthropicSystemV1,
    PHASE43_ANTHROPIC_PROFILE_VERSION, PHASE43_RESPONSES_PROFILE_VERSION, Phase43ApiErrorV1,
    Phase43ErrorCodeV1, ResponsesInputItemV1, ResponsesInputV1, ResponsesMessageRoleV1,
    ResponsesReasoningEffortV1, ResponsesRequestV1, ResponsesTextPartKindV1, ResponsesTextPartV1,
    SllmExtensionsV1, ToolChoiceV1 as Phase43ToolChoiceV1,
    ToolDefinitionV1 as Phase43ToolDefinitionV1, parse_anthropic_request_v1,
    parse_responses_request_v1, validate_anthropic_version_header,
};
pub use phase43_transport::{
    AnthropicStreamBuilderV1, Phase43CompletedOutputV1, Phase43FinishReasonV1, Phase43SseEventV1,
    Phase43ToolCallV1, Phase43TransportError, Phase43UsageV1, ResponsesStreamBuilderV1,
    anthropic_non_stream_v1, responses_non_stream_v1,
};
pub use production::{
    CheckpointStartupConfigV1, ContextWindowStartupConfigV1, DraftStartupConfigV1,
    Gemma4BackendConfigV1, Gemma4ChatBackendV1, KvCacheExplicitSourceV1, KvCacheSelectionReportV1,
    Phase41ProductionConfigV1, PrefixCacheStartupConfigV1, ProductionCheckpointOperationV1,
    ProductionCheckpointResultV1, ProductionDraftProviderV1, ProductionPhase41AuditV1,
    ProductionPrefixCacheResultV1, ProductionRequestAuditV1, ProductionShutdownAuditV1,
    QwenAdapterArtifactConfigV1, QwenAdapterCatalogConfigV1, QwenBackendConfigV1,
    QwenChatBackendV1, QwenPersistentChatFinishReasonV1, QwenPersistentChatSessionConfigV1,
    QwenPersistentChatSessionV1, QwenPersistentChatTurnRequestV1, QwenPersistentChatTurnResultV1,
    dynamic_model_plan_digest_preflight, qwen_adapter_catalog_identity_preflight,
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
pub use service::{ServerConfigV1, build_dynamic_router_v1, build_router_v1};

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
