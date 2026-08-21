//! Production model backends for the profile-v1 transport.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    AdapterModelDimsV1, AdapterRequestSetV1, AllocationSnapshot, Backend, CheckpointIdentity,
    CheckpointStore, CompiledGrammar, ContextPositionPolicyV1, ContextWindowStateV1,
    ControlVectorLockV1, ControlVectorSelectionV1, DerivedGgufLock, DraftProposalV1,
    DrySamplingConfigV1 as CoreDrySamplingConfigV1, DynamicTemperatureV1, EmbeddingPoolV1,
    ExecutionSession, ExecutionSessionRequest, GEMMA4_HIDDEN_SIZE,
    GEMMA4_RECOMMENDED_CONTEXT_TOKENS, Gemma4ModelLock, Gemma4PrefixForkAuditV1,
    Gemma4PrefixStateV1, Gemma4ResidentModel, KvCacheEncoding, LogitBiasV1 as CoreLogitBiasV1,
    LoraAdapterLockV1, LoraAdapterSelectionV1, MirostatModeV1,
    MirostatSamplingConfigV1 as CoreMirostatSamplingConfigV1, ModelLock, NgramDraftProviderV1,
    OsSamplingRandom, PrefixCacheConfigV1, PrefixCacheKeyV1, PrefixCacheV1, PrefixCacheValueV1,
    PrefixEntryIdV1, PrefixKvLayoutV1, PrefixLeaseV1, PrefixLookupKind, PrefixStateIdentityV1,
    QWEN35_HIDDEN_SIZE, QWEN35_RECOMMENDED_CONTEXT_TOKENS, QwenComponentSelection,
    QwenExecutionRequest, QwenGraph, QwenGraphStateDescriptor, QwenMultimodalImageEmbedding,
    QwenMultimodalPrompt, QwenPrefixForkAuditV1, QwenPrefixStateV1, QwenResidentModel,
    QwenVisionExecutionInput, QwenVisionManifest, QwenVisionResidentModel, ReviewedModelLock,
    SamplerChainConfigV1, SessionCheckpoint, VerifiedCache, VerifiedControlVectorPayloadV1,
    VerifiedFp8Sidecar, VerifiedGgufGemmaSource, VerifiedGgufQwen35Moe, VerifiedGgufWeightSource,
    VerifiedLoraPayloadV1, VerifiedNvfp4Sidecar, VerifiedQwen35Moe, WeightLoadPlan,
    XtcSamplingConfigV1 as CoreXtcSamplingConfigV1, assemble_gguf_qwen35_multimodal_prompt,
    assemble_qwen35_multimodal_prompt, build_gguf_qwen35_moe_weight_load_plan,
    build_qwen35_fp8_fnuz_graph, build_qwen35_fp8_graph, build_qwen35_gguf_fp8_graph,
    build_qwen35_gguf_moe_execution_graph, build_qwen35_graph_with_kv_cache_encoding,
    build_qwen35_graph_with_position_payload_mode, build_qwen35_moe_execution_graph,
    build_qwen35_mtp_graph, build_qwen35_multimodal_graph, build_qwen35_nvfp4_graph,
    build_verified_gguf_gemma_weight_load_plan, build_verified_gguf_qwen_weight_load_plan,
    build_verified_gguf_qwen35_vision_manifest, builtin_reviewed_model_lock,
    parse_control_vector_lock_v1, parse_lora_lock_v1, qwen_graph_memory_estimate,
    qwen_prefill_chunk_candidates, qwen35_moe_generation_stop_policy, read_derived_gguf_lock,
    verify_derived_gguf, verify_gguf_qwen35_moe,
};
use sllm_frontend::{
    ApplyTemplateResultV1, DecodeModeV1, GenerationCancellationV1, GenerationExecutorV1,
    GenerationInputV1, GenerationOutputSinkV1, GenerationResultV1, GenerationServiceError,
    GenerationServiceV1, GenerationStepV1, GenerationStopPolicyV1, GenerationTextFrontendV1,
    PreparedGenerationInputV1, Qwen35ChatMessageV1, Qwen35ChatTemplateV1, Qwen35RenderOptionsV1,
    QwenMtpGenerationExecutorV1, ReasoningPolicyV1, SpeculativeGenerationAdapterV1,
    SpeculativeGenerationExecutorV1, ThinkingModeV1, TokenIdsV1, TokenizeOptionsV1,
    TokenizeResultV1, TokenizerFrontendV1, TokenizerUtilityServiceV1,
    gemma4_generation_stop_policy,
};
use sllm_hip::HipBackend;

use crate::api::{ChatContentPartV1, GenerationRequestInputV1, ResponseFormatV1};
use crate::{
    BackendCompletionV1, BackendEmbeddingBatchV1, BackendEmbeddingInputV1,
    BackendEmbeddingRequestV1, BackendEmbeddingVectorV1, BackendErrorV1,
    BackendMemoryCategorySnapshotV1, BackendObservabilitySnapshotV1, BackendTokenLogprobV1,
    BackendTopLogprobV1, ChatCompletionRequestV1, ChatGenerationBackendV1, FinishReasonV1,
    GenerationDeltaSinkV1, TokenUsageV1,
};

const MAX_RETAINED_REQUEST_AUDITS: usize = 64;
const GEMMA4_RAW_CHAT_MAX_BYTES: usize = 16 * 1024 * 1024;
const GEMMA4_STATIC_FP8_KV_BYTES_PER_TOKEN: u64 = 172_032;

fn validate_generation_token_ids(
    tokenizer: &TokenizerFrontendV1,
    token_ids: &[u32],
    field: &str,
) -> Result<Vec<u32>, BackendErrorV1> {
    if token_ids.is_empty() {
        return Err(BackendErrorV1::new(format!(
            "{field} must contain at least one token"
        )));
    }
    for &token_id in token_ids {
        if tokenizer.token_byte_table().entry(token_id).is_none() {
            return Err(BackendErrorV1::new(format!(
                "{field} contains token ID {token_id} outside the verified vocabulary"
            )));
        }
    }
    Ok(token_ids.to_vec())
}

fn qwen_generation_prompt(
    request: &ChatCompletionRequestV1,
    service: &GenerationServiceV1<'_>,
    tokenizer: &TokenizerFrontendV1,
) -> Result<PreparedGenerationInputV1, BackendErrorV1> {
    match request.input() {
        GenerationRequestInputV1::Chat => service
            .prepare_input_plan(&GenerationInputV1::Messages {
                messages: request
                    .messages()
                    .iter()
                    .map(|message| message.inner().clone())
                    .collect(),
                options: Qwen35RenderOptionsV1 {
                    add_generation_prompt: true,
                    thinking: request.reasoning().thinking(),
                },
            })
            .map_err(|error| {
                BackendErrorV1::new(format!("generation input preparation failed: {error}"))
            }),
        GenerationRequestInputV1::ChatWithAssistantPrefill(assistant_prefill) => service
            .prepare_input_plan(&GenerationInputV1::MessagesWithAssistantPrefill {
                messages: request
                    .messages()
                    .iter()
                    .map(|message| message.inner().clone())
                    .collect(),
                options: Qwen35RenderOptionsV1 {
                    add_generation_prompt: true,
                    thinking: request.reasoning().thinking(),
                },
                assistant_prefill: assistant_prefill.clone(),
            })
            .map_err(|error| {
                BackendErrorV1::new(format!("generation input preparation failed: {error}"))
            }),
        GenerationRequestInputV1::RawText(text) => service
            .prepare_input_plan(&GenerationInputV1::Prompt(text.clone()))
            .map_err(|error| {
                BackendErrorV1::new(format!("generation input preparation failed: {error}"))
            }),
        GenerationRequestInputV1::RawTextWithAssistantPrefill {
            prompt,
            assistant_prefill,
        } => service
            .prepare_input_plan(&GenerationInputV1::PromptWithAssistantPrefill {
                prompt: prompt.clone(),
                assistant_prefill: assistant_prefill.clone(),
            })
            .map_err(|error| {
                BackendErrorV1::new(format!("generation input preparation failed: {error}"))
            }),
        GenerationRequestInputV1::TokenIds(token_ids) => {
            let token_ids = validate_generation_token_ids(tokenizer, token_ids, "token_ids")?;
            PreparedGenerationInputV1::from_token_ids(token_ids, Vec::new())
                .map_err(|error| BackendErrorV1::new(error.to_string()))
        }
        GenerationRequestInputV1::Infill { .. } => Err(BackendErrorV1::new(
            "infill is not supported by the current Qwen model lock (FIM capability absent)",
        )),
    }
}

fn gemma_generation_prompt(
    request: &ChatCompletionRequestV1,
    service: &GenerationServiceV1<'_>,
    tokenizer: &TokenizerFrontendV1,
) -> Result<PreparedGenerationInputV1, BackendErrorV1> {
    match request.input() {
        GenerationRequestInputV1::Chat => {
            let rendered = render_gemma4_raw_messages(request.messages())?;
            service
                .prepare_input_plan(&GenerationInputV1::Prompt(rendered))
                .map_err(|error| {
                    BackendErrorV1::new(format!("generation input preparation failed: {error}"))
                })
        }
        GenerationRequestInputV1::ChatWithAssistantPrefill(assistant_prefill) => {
            let rendered = render_gemma4_raw_messages(request.messages())?;
            service
                .prepare_input_plan(&GenerationInputV1::PromptWithAssistantPrefill {
                    prompt: rendered,
                    assistant_prefill: assistant_prefill.clone(),
                })
                .map_err(|error| {
                    BackendErrorV1::new(format!("generation input preparation failed: {error}"))
                })
        }
        GenerationRequestInputV1::RawText(text) => service
            .prepare_input_plan(&GenerationInputV1::Prompt(text.clone()))
            .map_err(|error| {
                BackendErrorV1::new(format!("generation input preparation failed: {error}"))
            }),
        GenerationRequestInputV1::RawTextWithAssistantPrefill {
            prompt,
            assistant_prefill,
        } => service
            .prepare_input_plan(&GenerationInputV1::PromptWithAssistantPrefill {
                prompt: prompt.clone(),
                assistant_prefill: assistant_prefill.clone(),
            })
            .map_err(|error| {
                BackendErrorV1::new(format!("generation input preparation failed: {error}"))
            }),
        GenerationRequestInputV1::TokenIds(token_ids) => {
            let token_ids = validate_generation_token_ids(tokenizer, token_ids, "token_ids")?;
            PreparedGenerationInputV1::from_token_ids(token_ids, Vec::new())
                .map_err(|error| BackendErrorV1::new(error.to_string()))
        }
        GenerationRequestInputV1::Infill { .. } => Err(BackendErrorV1::new(
            "infill is not supported by the current Gemma model lock (FIM capability absent)",
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_with_optional_assistant_prefill(
    service: &GenerationServiceV1<'_>,
    executor: &mut impl GenerationExecutorV1,
    input_token_ids: &[u32],
    assistant_prefill_token_ids: &[u32],
    generation: &sllm_frontend::GenerationConfigV1,
    cancellation: &GenerationCancellationV1,
    random: &mut OsSamplingRandom,
    sink: &mut OutputSinkAdapterV1<'_>,
) -> Result<GenerationResultV1, GenerationServiceError> {
    if assistant_prefill_token_ids.is_empty() {
        service.generate_tokens_with_sink(
            executor,
            input_token_ids,
            generation,
            cancellation,
            random,
            sink,
        )
    } else {
        service.generate_tokens_with_assistant_prefill_sink(
            executor,
            input_token_ids,
            assistant_prefill_token_ids,
            generation,
            cancellation,
            random,
            sink,
        )
    }
}

fn qwen_embedding_graph_for_rows(
    state: &QwenBackendStateV1,
    target_rows: u64,
    state_capacity: u64,
) -> Result<QwenGraph, BackendErrorV1> {
    if let Some(artifact) = &state.moe_artifact {
        build_qwen35_moe_execution_graph(artifact, &state.plan, target_rows, state_capacity)
            .map_err(|error| BackendErrorV1::new(format!("embedding graph failed: {error}")))
    } else if let Some(source) = &state.gguf_moe {
        build_qwen35_gguf_moe_execution_graph(source, &state.plan, target_rows, state_capacity)
            .map_err(|error| BackendErrorV1::new(format!("embedding graph failed: {error}")))
    } else if let Some(sidecar) = &state.nvfp4_sidecar {
        let lock = state.lock.as_ref().ok_or_else(|| {
            BackendErrorV1::new("NVFP4 embedding requires the reviewed dense Qwen lock")
        })?;
        build_qwen35_nvfp4_graph(lock, &state.plan, sidecar, target_rows, state_capacity)
            .map_err(|error| BackendErrorV1::new(format!("embedding graph failed: {error}")))
    } else if let Some(source) = state
        .gguf_source
        .as_ref()
        .filter(|source| source.has_fp8_recipe())
    {
        let lock = state.lock.as_ref().ok_or_else(|| {
            BackendErrorV1::new("GGUF FP8 embedding requires the reviewed dense Qwen lock")
        })?;
        build_qwen35_gguf_fp8_graph(
            lock,
            &state.plan,
            source,
            target_rows,
            state_capacity,
            gguf_fp8_dtype(
                state
                    .fp8_provider
                    .as_deref()
                    .ok_or_else(|| BackendErrorV1::new("GGUF FP8 provider is absent"))?,
            ),
            state.kv_cache_encoding,
        )
        .map_err(|error| BackendErrorV1::new(format!("embedding graph failed: {error}")))
    } else {
        let lock = state.lock.as_ref().ok_or_else(|| {
            BackendErrorV1::new("dense embedding requires the reviewed Qwen lock")
        })?;
        match (&state.sidecar, state.fp8_provider.as_deref()) {
            (Some(_), Some("converted-bf16")) | (None, None) => {
                build_qwen35_graph_with_kv_cache_encoding(
                    lock,
                    &state.plan,
                    target_rows,
                    state_capacity,
                    state.kv_cache_encoding,
                )
                .map_err(|error| BackendErrorV1::new(format!("embedding graph failed: {error}")))
            }
            (Some(sidecar), Some("native-fnuz")) => {
                build_qwen35_fp8_fnuz_graph(lock, &state.plan, sidecar, target_rows, state_capacity)
                    .map_err(|error| {
                        BackendErrorV1::new(format!("embedding graph failed: {error}"))
                    })
            }
            (Some(sidecar), Some(_)) => {
                build_qwen35_fp8_graph(lock, &state.plan, sidecar, target_rows, state_capacity)
                    .map_err(|error| {
                        BackendErrorV1::new(format!("embedding graph failed: {error}"))
                    })
            }
            _ => Err(BackendErrorV1::new(
                "validated Qwen embedding state has no supported weight source",
            )),
        }
    }
}

fn qwen_embedding_graph(
    state: &QwenBackendStateV1,
    token_count: u64,
) -> Result<QwenGraph, BackendErrorV1> {
    let total_memory = state
        .session
        .total_memory_bytes()
        .map_err(|error| BackendErrorV1::new(error.to_string()))?
        .ok_or_else(|| BackendErrorV1::new("backend omitted total device memory"))?;
    let available_memory = state
        .session
        .available_memory_bytes()
        .map_err(|error| BackendErrorV1::new(error.to_string()))?
        .ok_or_else(|| BackendErrorV1::new("backend omitted available device memory"))?;
    let candidates = qwen_prefill_chunk_candidates(total_memory, token_count)
        .map_err(|error| BackendErrorV1::new(error.to_string()))?;
    let mut rejected = Vec::new();
    for target_rows in candidates {
        let graph = qwen_embedding_graph_for_rows(state, target_rows, token_count)?;
        let estimate = qwen_graph_memory_estimate(&graph, &state.plan, total_memory)
            .map_err(|error| BackendErrorV1::new(error.to_string()))?;
        let incremental = estimate
            .required_bytes()
            .checked_sub(estimate.model_resident_bytes())
            .ok_or_else(|| BackendErrorV1::new("embedding placement estimate underflowed"))?;
        if incremental <= available_memory {
            return Ok(graph);
        }
        rejected.push(format!("{target_rows}:{incremental}"));
    }
    Err(BackendErrorV1::new(format!(
        "no embedding prefill chunk fits available device memory {available_memory}; candidates chunk:incremental-required [{}]",
        rejected.join(",")
    )))
}

fn qwen_embed_one(
    state: &QwenBackendStateV1,
    token_ids: &[u32],
    cancellation: &GenerationCancellationV1,
) -> Result<BackendEmbeddingVectorV1, BackendErrorV1> {
    if cancellation.is_cancelled() {
        return Err(BackendErrorV1::new("embedding cancelled"));
    }
    let token_count = u64::try_from(token_ids.len())
        .map_err(|_| BackendErrorV1::new("embedding token count overflowed u64"))?;
    let graph = qwen_embedding_graph(state, token_count)?;
    let mut owner = state
        .resident
        .new_request_for_session(Arc::clone(&state.session), graph)
        .map_err(|error| {
            BackendErrorV1::new(format!("embedding request provisioning failed: {error}"))
        })?;
    let token_ids_i32 = token_ids
        .iter()
        .map(|&token_id| {
            i32::try_from(token_id)
                .map_err(|_| BackendErrorV1::new("embedding token ID does not fit i32"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output = owner
        .prefill_with_embeddings(&token_ids_i32)
        .map_err(|error| BackendErrorV1::new(format!("embedding execution failed: {error}")))?;
    let audit = owner
        .audit_snapshot()
        .map_err(|error| BackendErrorV1::new(format!("embedding audit failed: {error}")))?;
    if audit.selected_backend() != "hip"
        || audit.target() != state.target
        || audit.fallback_used()
        || !audit.all_dispatches_hip()
    {
        return Err(BackendErrorV1::new(
            "embedding dispatch audit is not exact HIP/no-fallback",
        ));
    }
    let rows = output
        .embeddings_bf16()
        .ok_or_else(|| BackendErrorV1::new("embedding execution omitted final hidden rows"))?;
    let pooled = EmbeddingPoolV1::new()
        .pool_bf16(rows, token_ids.len(), QWEN35_HIDDEN_SIZE)
        .map_err(|error| BackendErrorV1::new(format!("embedding pooling failed: {error}")))?;
    if cancellation.is_cancelled() {
        owner.cancel();
        return Err(BackendErrorV1::new("embedding cancelled"));
    }
    BackendEmbeddingVectorV1::new(pooled.as_slice().to_vec(), token_count)
}

fn gemma_embed_one(
    state: &Gemma4BackendStateV1,
    token_ids: &[u32],
    cancellation: &GenerationCancellationV1,
) -> Result<BackendEmbeddingVectorV1, BackendErrorV1> {
    if cancellation.is_cancelled() {
        return Err(BackendErrorV1::new("embedding cancelled"));
    }
    let token_count = u64::try_from(token_ids.len())
        .map_err(|_| BackendErrorV1::new("embedding token count overflowed u64"))?;
    let mut owner = state
        .resident
        .new_request_for_session(Arc::clone(&state.session), token_count, token_count)
        .map_err(|error| {
            BackendErrorV1::new(format!("embedding request provisioning failed: {error}"))
        })?;
    let token_ids_i32 = token_ids
        .iter()
        .map(|&token_id| {
            i32::try_from(token_id)
                .map_err(|_| BackendErrorV1::new("embedding token ID does not fit i32"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output = owner
        .prefill_with_embeddings(&token_ids_i32)
        .map_err(|error| BackendErrorV1::new(format!("embedding execution failed: {error}")))?;
    let audit = owner
        .audit_snapshot()
        .map_err(|error| BackendErrorV1::new(format!("embedding audit failed: {error}")))?;
    if audit.target() != state.target || audit.fallback_used() {
        return Err(BackendErrorV1::new(
            "embedding dispatch audit is not exact HIP/no-fallback",
        ));
    }
    let rows = output
        .embeddings_bf16()
        .ok_or_else(|| BackendErrorV1::new("embedding execution omitted final hidden rows"))?;
    let pooled = EmbeddingPoolV1::new()
        .pool_bf16(rows, token_ids.len(), GEMMA4_HIDDEN_SIZE as usize)
        .map_err(|error| BackendErrorV1::new(format!("embedding pooling failed: {error}")))?;
    if cancellation.is_cancelled() {
        owner.cancel();
        return Err(BackendErrorV1::new("embedding cancelled"));
    }
    BackendEmbeddingVectorV1::new(pooled.as_slice().to_vec(), token_count)
}

fn generation_config_for_request(
    request: &ChatCompletionRequestV1,
    tokenizer: &TokenizerFrontendV1,
    reasoning_close_token_ids: Option<&[u32]>,
) -> Result<sllm_frontend::GenerationConfigV1, BackendErrorV1> {
    let mut generation = request.generation().clone();
    let mut chain = SamplerChainConfigV1::new(generation.sampling());
    let mut advanced = false;

    if let Some(bias) = request.logit_bias() {
        chain = chain
            .with_logit_bias(
                bias.entries()
                    .iter()
                    .map(|(&token_id, &bias)| CoreLogitBiasV1 { token_id, bias })
                    .collect(),
            )
            .map_err(|error| BackendErrorV1::new(format!("logit bias failed: {error}")))?;
        advanced = true;
    }
    if let Some(logprobs) = request.logprobs() {
        chain = chain.with_return_logprobs(logprobs.enabled());
        chain = chain
            .with_top_logprobs(usize::from(logprobs.top_logprobs().unwrap_or(0)))
            .map_err(|error| BackendErrorV1::new(format!("logprobs failed: {error}")))?;
        advanced |= logprobs.enabled();
    }
    if let Some(extension) = request.sampler() {
        if let Some(top_k) = extension.top_k() {
            chain = chain
                .with_top_k(top_k as usize)
                .map_err(|error| BackendErrorV1::new(format!("top-k failed: {error}")))?;
        }
        if let Some(min_p) = extension.min_p() {
            chain = chain
                .with_min_p(min_p)
                .map_err(|error| BackendErrorV1::new(format!("min-p failed: {error}")))?;
        }
        if let Some(typical_p) = extension.typical_p() {
            chain = chain
                .with_typical_p(typical_p)
                .map_err(|error| BackendErrorV1::new(format!("typical-p failed: {error}")))?;
        }
        if let Some(penalty) = extension.repeat_penalty() {
            chain = chain
                .with_repeat_penalty(penalty, extension.repeat_last_n() as usize)
                .map_err(|error| BackendErrorV1::new(format!("repeat penalty failed: {error}")))?;
        }
        if let Some(dynamic) = extension.dynamic_temperature() {
            let center = generation.sampling().temperature().max(0.01);
            let minimum = (center - dynamic.range()).max(0.01);
            let maximum = (center + dynamic.range()).min(2.0).max(minimum);
            let dynamic = DynamicTemperatureV1::new(minimum, maximum, dynamic.exponent()).map_err(
                |error| BackendErrorV1::new(format!("dynamic temperature failed: {error}")),
            )?;
            chain = chain.with_dynamic_temperature(dynamic).map_err(|error| {
                BackendErrorV1::new(format!("dynamic temperature failed: {error}"))
            })?;
        }
        if let Some(dry) = extension.dry() {
            let mut breakers = Vec::with_capacity(dry.sequence_breakers().len());
            for breaker in dry.sequence_breakers() {
                let tokens = tokenizer
                    .encode_without_special_tokens(breaker)
                    .map_err(|error| {
                        BackendErrorV1::new(format!("DRY sequence breaker failed: {error}"))
                    })?;
                if tokens.is_empty() {
                    return Err(BackendErrorV1::new(
                        "DRY sequence breaker produced no token IDs",
                    ));
                }
                breakers.push(tokens.as_slice().to_vec());
            }
            let dry = CoreDrySamplingConfigV1::new(
                dry.multiplier(),
                dry.base(),
                dry.allowed_length() as usize,
                breakers,
            )
            .and_then(|value| value.with_penalty_last_n(dry.penalty_last_n() as usize))
            .map_err(|error| BackendErrorV1::new(format!("DRY sampling failed: {error}")))?;
            chain = chain
                .with_dry(dry)
                .map_err(|error| BackendErrorV1::new(format!("DRY sampling failed: {error}")))?;
        }
        if let Some(xtc) = extension.xtc() {
            let xtc = CoreXtcSamplingConfigV1::new(
                xtc.probability(),
                xtc.threshold(),
                xtc.min_keep() as usize,
            )
            .map_err(|error| BackendErrorV1::new(format!("XTC sampling failed: {error}")))?;
            chain = chain
                .with_xtc(xtc)
                .map_err(|error| BackendErrorV1::new(format!("XTC sampling failed: {error}")))?;
        }
        if let Some(mirostat) = extension.mirostat() {
            let mode = if mirostat.version() == 1 {
                MirostatModeV1::V1
            } else {
                MirostatModeV1::V2
            };
            let mirostat = CoreMirostatSamplingConfigV1::new(
                mode,
                mirostat.tau(),
                mirostat.eta(),
                2.0 * mirostat.tau(),
            )
            .map_err(|error| BackendErrorV1::new(format!("Mirostat failed: {error}")))?;
            chain = chain
                .with_mirostat(mirostat)
                .map_err(|error| BackendErrorV1::new(format!("Mirostat failed: {error}")))?;
        }
        generation = generation.with_ignore_stop_tokens(extension.ignore_eos());
        advanced = true;
    }

    if advanced {
        generation = generation
            .with_sampler_chain(chain)
            .map_err(|error| BackendErrorV1::new(format!("sampler chain failed: {error}")))?;
    }
    if let Some(response_format) = request.response_format() {
        let grammar = match response_format {
            ResponseFormatV1::Text => None,
            ResponseFormatV1::JsonObject => Some(CompiledGrammar::json_object()),
            ResponseFormatV1::JsonSchema(schema) => {
                Some(CompiledGrammar::from_json_schema(schema.schema()))
            }
        }
        .transpose()
        .map_err(|error| BackendErrorV1::new(format!("structured output failed: {error}")))?;
        if let Some(grammar) = grammar {
            generation = generation.with_grammar(grammar);
        }
    }
    if request.reasoning().enabled() {
        let close_token_ids = reasoning_close_token_ids
            .ok_or_else(|| BackendErrorV1::new("Qwen reasoning close marker is unavailable"))?;
        let policy = qwen_reasoning_policy_for_request(request, close_token_ids)?;
        generation = generation
            .with_reasoning(policy)
            .map_err(|error| BackendErrorV1::new(format!("reasoning policy failed: {error}")))?;
    } else if request.reasoning().max_reasoning_tokens().is_some() {
        return Err(BackendErrorV1::new(
            "reasoning budget requires an enabled reasoning mode",
        ));
    }
    Ok(generation)
}

/// Resolves and validates the reviewed Qwen close marker once while opening
/// the model.  Request admission only reuses this immutable token sequence,
/// so a missing or non-round-tripping marker cannot reach scheduler/GPU work.
fn validate_qwen_reasoning_close_marker(
    tokenizer: &TokenizerFrontendV1,
) -> Result<Vec<u32>, BackendErrorV1> {
    let close = tokenizer
        .encode_without_special_tokens("</think>")
        .map_err(|_| BackendErrorV1::new("Qwen reasoning close marker is unavailable"))?;
    if close.is_empty() {
        return Err(BackendErrorV1::new(
            "Qwen reasoning close marker is unavailable",
        ));
    }
    let close_ids = close.as_slice().to_vec();
    let decoded = tokenizer
        .decode(
            &TokenIdsV1::from_slice(&close_ids),
            DecodeModeV1::PreserveSpecialTokens,
        )
        .map_err(|_| BackendErrorV1::new("Qwen reasoning close marker is unavailable"))?;
    if decoded != "</think>" {
        return Err(BackendErrorV1::new(
            "Qwen reasoning close marker is unavailable",
        ));
    }
    Ok(close_ids)
}

fn qwen_reasoning_policy_for_request(
    request: &ChatCompletionRequestV1,
    close_token_ids: &[u32],
) -> Result<ReasoningPolicyV1, BackendErrorV1> {
    // Raw completion prompts never receive the reasoning policy from the
    // public wire adapter.  The sole non-chat exception is the internal
    // `protocol` bit set by Phase 43 after it has rendered a reviewed,
    // bounded protocol/tool envelope through `from_protocol_text`.
    let chat_input = matches!(
        request.input(),
        GenerationRequestInputV1::Chat | GenerationRequestInputV1::ChatWithAssistantPrefill(_)
    );
    if !chat_input && !request.reasoning().protocol() {
        return Err(BackendErrorV1::new(
            "reasoning mode requires a chat input with the reviewed Qwen template",
        ));
    }
    if close_token_ids.is_empty() {
        return Err(BackendErrorV1::new(
            "Qwen reasoning close marker is unavailable",
        ));
    }
    ReasoningPolicyV1::from_thinking(
        request.reasoning().thinking(),
        request.reasoning().max_reasoning_tokens(),
        close_token_ids.to_vec(),
    )
    .map_err(|error| BackendErrorV1::new(format!("reasoning policy is invalid: {error}")))
}

fn publish_generation_logprobs(
    request: &ChatCompletionRequestV1,
    tokenizer: &TokenizerFrontendV1,
    result: &sllm_frontend::GenerationResultV1,
    sink: &mut dyn GenerationDeltaSinkV1,
) -> Result<(), BackendErrorV1> {
    if !request.logprobs().is_some_and(|options| options.enabled()) {
        return Ok(());
    }
    let table = tokenizer.token_byte_table();
    let token_metadata = |token_id: u32, logprob: f64| {
        let entry = table.entry(token_id).ok_or_else(|| {
            BackendErrorV1::new(format!(
                "logprob token {token_id} is outside the vocabulary"
            ))
        })?;
        Ok(BackendTopLogprobV1 {
            token: entry.piece().unwrap_or("").to_owned(),
            bytes: entry.bytes().map(<[u8]>::to_vec),
            logprob,
        })
    };
    let values = result
        .selections()
        .iter()
        .map(|selection| {
            let selected = token_metadata(selection.token_id, selection.logprob)?;
            let top_logprobs = selection
                .top_logprobs
                .iter()
                .map(|value| token_metadata(value.token_id, value.logprob))
                .collect::<Result<Vec<_>, BackendErrorV1>>()?;
            Ok(BackendTokenLogprobV1 {
                token: selected.token,
                bytes: selected.bytes,
                logprob: selected.logprob,
                top_logprobs,
            })
        })
        .collect::<Result<Vec<_>, BackendErrorV1>>()?;
    sink.publish_logprobs(values)
}

fn observability_snapshot_from_allocation(
    snapshot: AllocationSnapshot,
) -> BackendObservabilitySnapshotV1 {
    if snapshot.poisoned() {
        return BackendObservabilitySnapshotV1::default();
    }
    let model_resident = snapshot.model_resident();
    let request_kv = snapshot.request_state();
    let workspace_arena = snapshot.workspace();
    BackendObservabilitySnapshotV1 {
        model_resident: BackendMemoryCategorySnapshotV1 {
            current_bytes: model_resident.current_bytes(),
            high_water_bytes: model_resident.high_water_bytes(),
        },
        request_kv: BackendMemoryCategorySnapshotV1 {
            current_bytes: request_kv.current_bytes(),
            high_water_bytes: request_kv.high_water_bytes(),
        },
        workspace_arena: BackendMemoryCategorySnapshotV1 {
            current_bytes: workspace_arena.current_bytes(),
            high_water_bytes: workspace_arena.high_water_bytes(),
        },
        total: BackendMemoryCategorySnapshotV1 {
            current_bytes: snapshot.current_bytes(),
            high_water_bytes: snapshot.high_water_bytes(),
        },
    }
}

fn select_gguf_fp8_provider(target: &str) -> Result<&'static str, String> {
    match target {
        "gfx1201" => Ok("gguf-native"),
        "gfx942" => Ok("native-fnuz"),
        _ => Err(format!(
            "embedded E4M3FN GGUF recipe requires exact gfx1201 native or gfx942 native-fnuz provider; exact target {target} is unsupported"
        )),
    }
}

fn gguf_fp8_dtype(provider: &str) -> sllm_core::DType {
    match provider {
        "gguf-native" => sllm_core::DType::F8E4M3Fn,
        "native-fnuz" => sllm_core::DType::F8E4M3FnuZ,
        _ => unreachable!("validated GGUF FP8 provider"),
    }
}

pub const MAX_PREFIX_CACHE_ENTRIES_V1: u16 = 256;
pub const MAX_PREFIX_CACHE_LOGICAL_TOKENS_V1: u64 = 1_048_576;
pub const MAX_DRAFT_NGRAM_ORDER_V1: u8 = 16;
pub const MAX_DRAFT_WIDTH_V1: u8 = 8;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum PrefixCacheStartupConfigV1 {
    #[default]
    Disabled,
    Enabled {
        max_entries: u16,
        max_logical_tokens: u64,
        max_resident_bytes: u64,
    },
}

impl PrefixCacheStartupConfigV1 {
    pub fn validate(&self) -> Result<(), BackendErrorV1> {
        match self {
            Self::Disabled => Ok(()),
            Self::Enabled {
                max_entries,
                max_logical_tokens,
                max_resident_bytes,
            } if (1..=MAX_PREFIX_CACHE_ENTRIES_V1).contains(max_entries)
                && (1..=MAX_PREFIX_CACHE_LOGICAL_TOKENS_V1).contains(max_logical_tokens)
                && *max_resident_bytes != 0 =>
            {
                Ok(())
            }
            Self::Enabled { .. } => Err(BackendErrorV1::new(
                "prefix cache limits must be entries 1..=256, logical tokens 1..=1048576, and nonzero resident bytes",
            )),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum ContextWindowStartupConfigV1 {
    #[default]
    Disabled,
    KeepPrefixRecentV1 {
        keep_prefix: u64,
        keep_recent: u64,
    },
}

impl ContextWindowStartupConfigV1 {
    pub fn validate(&self) -> Result<(), BackendErrorV1> {
        match self {
            Self::Disabled => Ok(()),
            Self::KeepPrefixRecentV1 {
                keep_prefix,
                keep_recent,
            } => {
                let retained = keep_prefix.checked_add(*keep_recent).ok_or_else(|| {
                    BackendErrorV1::new("context keep-prefix/recent sum overflowed")
                })?;
                if retained == 0 {
                    return Err(BackendErrorV1::new(
                        "keep-prefix-recent-v1 must retain at least one token",
                    ));
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum CheckpointStartupConfigV1 {
    #[default]
    Disabled,
    Enabled {
        directory: PathBuf,
        quota_bytes: u64,
        load_name: Option<String>,
        save_name: Option<String>,
    },
}

impl CheckpointStartupConfigV1 {
    pub fn validate(&self) -> Result<(), BackendErrorV1> {
        match self {
            Self::Disabled => Ok(()),
            Self::Enabled {
                directory,
                quota_bytes,
                load_name,
                save_name,
            } => {
                if directory.as_os_str().is_empty() || *quota_bytes == 0 {
                    return Err(BackendErrorV1::new(
                        "checkpoint directory and nonzero quota are required when enabled",
                    ));
                }
                if load_name.is_none() && save_name.is_none() {
                    return Err(BackendErrorV1::new(
                        "checkpoint enablement requires an explicit load or save name",
                    ));
                }
                if load_name.is_some() && save_name.is_some() {
                    return Err(BackendErrorV1::new(
                        "checkpoint load and save names cannot be enabled together",
                    ));
                }
                for name in load_name.iter().chain(save_name.iter()) {
                    validate_checkpoint_name(name)?;
                }
                Ok(())
            }
        }
    }

    /// Startup-only validation seam. It proves that an explicitly requested
    /// load target exists and is a regular non-symlink file without reading or
    /// exposing checkpoint contents or its path in an error.
    pub fn validate_startup_load_exists(&self) -> Result<(), BackendErrorV1> {
        self.validate()?;
        let Self::Enabled {
            directory,
            load_name: Some(load_name),
            ..
        } = self
        else {
            return Ok(());
        };
        let metadata = std::fs::symlink_metadata(directory.join(format!("{load_name}.ckpt")))
            .map_err(|_| BackendErrorV1::new("configured checkpoint load target is unavailable"))?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(BackendErrorV1::new(
                "configured checkpoint load target is unavailable",
            ));
        }
        Ok(())
    }
}

fn validate_checkpoint_name(name: &str) -> Result<(), BackendErrorV1> {
    if name.is_empty()
        || name.len() > 128
        || name == "."
        || name == ".."
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
    {
        return Err(BackendErrorV1::new("checkpoint name is invalid"));
    }
    Ok(())
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum DraftStartupConfigV1 {
    #[default]
    Disabled,
    MtpAuto,
    Ngram {
        order: u8,
        width: u8,
    },
    External {
        model_identity: String,
        tokenizer_identity: String,
        vocabulary_size: u32,
        width: u8,
    },
}

impl DraftStartupConfigV1 {
    pub fn validate(&self) -> Result<(), BackendErrorV1> {
        match self {
            Self::Disabled | Self::MtpAuto => Ok(()),
            Self::Ngram { order, width }
                if (1..=MAX_DRAFT_NGRAM_ORDER_V1).contains(order)
                    && (1..=MAX_DRAFT_WIDTH_V1).contains(width) =>
            {
                Ok(())
            }
            Self::Ngram { .. } => Err(BackendErrorV1::new(
                "ngram order must be 1..=16 and draft width must be 1..=8",
            )),
            Self::External {
                model_identity,
                tokenizer_identity,
                vocabulary_size,
                width,
            } => {
                if !valid_draft_identity(model_identity)
                    || !valid_draft_identity(tokenizer_identity)
                    || *vocabulary_size == 0
                    || !(1..=MAX_DRAFT_WIDTH_V1).contains(width)
                {
                    return Err(BackendErrorV1::new(
                        "external draft identity, vocabulary, or width is invalid",
                    ));
                }
                Ok(())
            }
        }
    }
}

fn valid_draft_identity(value: &str) -> bool {
    !value.is_empty() && value.len() <= 1024 && !value.as_bytes().contains(&0)
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Phase41ProductionConfigV1 {
    pub prefix_cache: PrefixCacheStartupConfigV1,
    pub context_window: ContextWindowStartupConfigV1,
    pub checkpoint: CheckpointStartupConfigV1,
    pub draft: DraftStartupConfigV1,
}

impl Phase41ProductionConfigV1 {
    pub fn validate(&self) -> Result<(), BackendErrorV1> {
        self.prefix_cache.validate()?;
        self.context_window.validate()?;
        self.checkpoint.validate()?;
        self.draft.validate()
    }

    pub fn validate_startup(&self) -> Result<(), BackendErrorV1> {
        self.validate()?;
        self.checkpoint.validate_startup_load_exists()
    }
}

fn validate_qwen_phase41_operational_config(
    config: &Phase41ProductionConfigV1,
) -> Result<(), BackendErrorV1> {
    if matches!(config.draft, DraftStartupConfigV1::External { .. }) {
        return Err(BackendErrorV1::new(
            "external draft requires an independently provisioned executor; configuration-only identity cannot start production inference",
        ));
    }
    if !matches!(config.prefix_cache, PrefixCacheStartupConfigV1::Disabled)
        && matches!(config.draft, DraftStartupConfigV1::MtpAuto)
    {
        return Err(BackendErrorV1::new(
            "prefix cache and MTP-auto cannot be combined until the MTP owner accepts a prefixed target state",
        ));
    }
    if !matches!(config.checkpoint, CheckpointStartupConfigV1::Disabled)
        && (!matches!(config.prefix_cache, PrefixCacheStartupConfigV1::Disabled)
            || !matches!(
                config.context_window,
                ContextWindowStartupConfigV1::Disabled
            )
            || !matches!(config.draft, DraftStartupConfigV1::Disabled))
    {
        return Err(BackendErrorV1::new(
            "Qwen prompt checkpoint cannot be combined with prefix cache, context shifting, or draft execution",
        ));
    }
    Ok(())
}

fn validate_gemma_phase41_operational_config(
    config: &Phase41ProductionConfigV1,
) -> Result<(), BackendErrorV1> {
    if !matches!(config.checkpoint, CheckpointStartupConfigV1::Disabled)
        && (!matches!(config.prefix_cache, PrefixCacheStartupConfigV1::Disabled)
            || !matches!(
                config.context_window,
                ContextWindowStartupConfigV1::Disabled
            )
            || !matches!(config.draft, DraftStartupConfigV1::Disabled))
    {
        return Err(BackendErrorV1::new(
            "Gemma prompt checkpoint cannot be combined with prefix cache, context shifting, or draft execution",
        ));
    }
    match config.draft {
        DraftStartupConfigV1::Disabled => Ok(()),
        DraftStartupConfigV1::MtpAuto => Err(BackendErrorV1::new(
            "Gemma production does not have a provisioned MTP draft executor",
        )),
        DraftStartupConfigV1::Ngram { .. } => Err(BackendErrorV1::new(
            "ngram speculative verification is currently implemented only by the Qwen target",
        )),
        DraftStartupConfigV1::External { .. } => Err(BackendErrorV1::new(
            "external draft requires an independently provisioned executor; configuration-only identity cannot start production inference",
        )),
    }
}

#[derive(Clone, Debug)]
pub struct QwenAdapterArtifactConfigV1 {
    pub alias: String,
    pub lock_path: PathBuf,
    pub payload_path: PathBuf,
}

#[derive(Clone, Debug, Default)]
pub struct QwenAdapterCatalogConfigV1 {
    pub lora: Vec<QwenAdapterArtifactConfigV1>,
    pub control_vectors: Vec<QwenAdapterArtifactConfigV1>,
}

#[derive(Clone, Debug)]
pub struct QwenBackendConfigV1 {
    pub gguf_path: PathBuf,
    pub derived_lock_path: PathBuf,
    pub device_index: u32,
    pub target: String,
    pub completion_timeout: Duration,
    pub shutdown_timeout: Duration,
    pub context_length: u32,
    pub kv_cache_encoding: KvCacheEncoding,
    pub phase41: Phase41ProductionConfigV1,
    pub adapter_catalog: Option<QwenAdapterCatalogConfigV1>,
}

/// Configuration for one persistent dense-BF16 Qwen chat owner.  The owner
/// creates one resident HIP model/session and keeps the current opaque
/// checkpoint between turns; checkpoint files remain managed by the core
/// `CheckpointStore` rather than by the CLI.
#[derive(Clone, Debug)]
pub struct QwenPersistentChatSessionConfigV1 {
    pub backend: QwenBackendConfigV1,
    pub checkpoint_directory: PathBuf,
    pub checkpoint_quota_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenPersistentChatTurnRequestV1 {
    pub messages: Vec<Qwen35ChatMessageV1>,
    pub max_new_tokens: u32,
    pub stop_sequences: Vec<String>,
    pub reverse_prompts: Vec<String>,
    pub thinking: ThinkingModeV1,
    pub reasoning_budget: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenPersistentChatFinishReasonV1 {
    Stop,
    ReversePrompt,
    Length,
}

impl QwenPersistentChatFinishReasonV1 {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::ReversePrompt => "reverse_prompt",
            Self::Length => "length",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenPersistentChatTurnResultV1 {
    pub text: String,
    pub reasoning: Option<String>,
    pub finish_reason: QwenPersistentChatFinishReasonV1,
    pub usage: TokenUsageV1,
}

impl QwenBackendConfigV1 {
    pub fn validate(&self) -> Result<(), BackendErrorV1> {
        if self.target.is_empty()
            || !self.target.is_ascii()
            || self.completion_timeout.is_zero()
            || self.shutdown_timeout.is_zero()
            || self.context_length == 0
        {
            return Err(BackendErrorV1::new(
                "Qwen backend target, context length, and timeouts must be valid and nonzero",
            ));
        }
        if let Some(catalog) = &self.adapter_catalog {
            validate_qwen_adapter_catalog_config(catalog)?;
        }
        self.phase41.validate()?;
        Ok(())
    }
}

fn validate_qwen_adapter_catalog_config(
    catalog: &QwenAdapterCatalogConfigV1,
) -> Result<(), BackendErrorV1> {
    if catalog.lora.len() > 8 || catalog.control_vectors.len() > 8 {
        return Err(BackendErrorV1::new(
            "Qwen adapter catalog supports at most 8 LoRA and 8 control artifacts",
        ));
    }
    let mut aliases = BTreeSet::new();
    for list in [&catalog.lora, &catalog.control_vectors] {
        for pair in list.windows(2) {
            if pair[0].alias >= pair[1].alias {
                return Err(BackendErrorV1::new(
                    "Qwen adapter catalog aliases must be strictly sorted and unique",
                ));
            }
        }
    }
    for artifact in catalog.lora.iter().chain(catalog.control_vectors.iter()) {
        if artifact.alias.is_empty()
            || artifact.alias.len() > 128
            || !artifact
                .alias
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        {
            return Err(BackendErrorV1::new(format!(
                "Qwen adapter alias {} must match [A-Za-z0-9._-]",
                artifact.alias
            )));
        }
        if !aliases.insert(artifact.alias.as_str()) {
            return Err(BackendErrorV1::new(format!(
                "Qwen adapter alias {} is duplicated",
                artifact.alias
            )));
        }
        if artifact.lock_path.as_os_str().is_empty() || artifact.payload_path.as_os_str().is_empty()
        {
            return Err(BackendErrorV1::new(format!(
                "Qwen adapter {} requires lock and payload paths",
                artifact.alias
            )));
        }
    }
    Ok(())
}

const MAX_QWEN_ADAPTER_LOCK_BYTES: u64 = 1 << 20;
const MAX_QWEN_ADAPTER_PAYLOAD_BYTES: u64 = 1 << 30;

fn adapter_file_identity(metadata: &std::fs::Metadata) -> (u64, u64, u64) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        (metadata.dev(), metadata.ino(), metadata.len())
    }
    #[cfg(not(unix))]
    {
        (0, 0, metadata.len())
    }
}

fn read_qwen_adapter_file(path: &std::path::Path, maximum: u64) -> Result<Vec<u8>, BackendErrorV1> {
    let before = fs::symlink_metadata(path)
        .map_err(|error| BackendErrorV1::new(format!("adapter file metadata failed: {error}")))?;
    if !before.is_file() || before.file_type().is_symlink() || before.len() > maximum {
        return Err(BackendErrorV1::new(
            "adapter file is not a bounded regular file",
        ));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    }
    let file: File = options
        .open(path)
        .map_err(|error| BackendErrorV1::new(format!("adapter file open failed: {error}")))?;
    let opened = file
        .metadata()
        .map_err(|error| BackendErrorV1::new(format!("adapter file metadata failed: {error}")))?;
    if !opened.is_file()
        || adapter_file_identity(&before) != adapter_file_identity(&opened)
        || opened.len() > maximum
    {
        return Err(BackendErrorV1::new(
            "adapter file changed before bounded read",
        ));
    }
    let read_limit = maximum
        .checked_add(1)
        .ok_or_else(|| BackendErrorV1::new("adapter file bound overflowed"))?;
    let mut bytes = Vec::new();
    (&file)
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| BackendErrorV1::new(format!("adapter file read failed: {error}")))?;
    let after = file
        .metadata()
        .map_err(|error| BackendErrorV1::new(format!("adapter file metadata failed: {error}")))?;
    if adapter_file_identity(&opened) != adapter_file_identity(&after)
        || bytes.len() as u64 != after.len()
        || bytes.len() as u64 > maximum
    {
        return Err(BackendErrorV1::new(
            "adapter file changed during bounded read",
        ));
    }
    Ok(bytes)
}

fn sha256_prefixed(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::from("sha256:");
    for byte in digest {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

/// Rebuilds the exact runtime weight-plan identity from a verified derived
/// artifact without opening a HIP session.  The derived-lock fingerprint is
/// an artifact identity; this digest is the plan identity used by prefix and
/// checkpoint state, and must therefore be kept as a separate field.
pub fn dynamic_model_plan_digest_preflight(
    gguf_path: &Path,
    derived: &DerivedGgufLock,
) -> Result<String, BackendErrorV1> {
    let verified = verify_derived_gguf(derived.clone(), gguf_path)
        .map_err(|error| BackendErrorV1::new(format!("GGUF verification failed: {error}")))?;
    if derived.semantic_model_id.starts_with("qwen35moe:") {
        let source = verify_gguf_qwen35_moe(verified).map_err(|error| {
            BackendErrorV1::new(format!("Qwen MoE GGUF validation failed: {error}"))
        })?;
        return build_gguf_qwen35_moe_weight_load_plan(&source)
            .map(|plan| plan.digest_hex())
            .map_err(|error| BackendErrorV1::new(format!("Qwen MoE load plan failed: {error}")));
    }
    let reviewed =
        builtin_reviewed_model_lock(&derived.source_lock_fingerprints).map_err(|error| {
            BackendErrorV1::new(format!("built-in model lock resolution failed: {error}"))
        })?;
    match reviewed {
        ReviewedModelLock::Qwen35(lock) => build_verified_gguf_qwen_weight_load_plan(
            &lock,
            verified,
            QwenComponentSelection::TEXT_ONLY,
        )
        .map(|(_, plan)| plan.digest_hex())
        .map_err(|error| BackendErrorV1::new(format!("verified Qwen load plan failed: {error}"))),
        ReviewedModelLock::Gemma4(lock) => {
            build_verified_gguf_gemma_weight_load_plan(&lock, verified)
                .map(|(_, plan)| plan.digest_hex())
                .map_err(|error| {
                    BackendErrorV1::new(format!("verified Gemma load plan failed: {error}"))
                })
        }
    }
}

/// Computes the same path-independent identity used by a loaded Qwen adapter
/// catalog, while only reading bounded, non-symlink lock/payload files. The
/// backend repeats model/plan verification when opening the resident model.
pub fn qwen_adapter_catalog_identity_preflight(
    config: &QwenAdapterCatalogConfigV1,
) -> Result<String, BackendErrorV1> {
    validate_qwen_adapter_catalog_config(config)?;
    let mut entries = Vec::with_capacity(config.lora.len() + config.control_vectors.len());
    for artifact in &config.lora {
        let lock_bytes = read_qwen_adapter_file(&artifact.lock_path, MAX_QWEN_ADAPTER_LOCK_BYTES)
            .map_err(|error| {
            BackendErrorV1::new(format!(
                "LoRA adapter {} lock read failed: {error}",
                artifact.alias
            ))
        })?;
        let lock = parse_lora_lock_v1(&lock_bytes).map_err(|error| {
            BackendErrorV1::new(format!(
                "LoRA adapter {} lock verification failed: {error}",
                artifact.alias
            ))
        })?;
        if lock.payload_size == 0 || lock.payload_size > MAX_QWEN_ADAPTER_PAYLOAD_BYTES {
            return Err(BackendErrorV1::new(format!(
                "LoRA adapter {} payload size is outside the bounded limit",
                artifact.alias
            )));
        }
        let payload =
            read_qwen_adapter_file(&artifact.payload_path, lock.payload_size).map_err(|error| {
                BackendErrorV1::new(format!(
                    "LoRA adapter {} payload read failed: {error}",
                    artifact.alias
                ))
            })?;
        if payload.len() as u64 != lock.payload_size
            || !sha256_prefixed(&payload).eq_ignore_ascii_case(&lock.payload_sha256)
        {
            return Err(BackendErrorV1::new(format!(
                "LoRA adapter {} payload hash or size differs",
                artifact.alias
            )));
        }
        let canonical = serde_json::to_vec(&lock).map_err(|error| {
            BackendErrorV1::new(format!("adapter lock canonicalization failed: {error}"))
        })?;
        entries.push(QwenAdapterCatalogIdentityEntry {
            kind: "lora",
            alias: artifact.alias.clone(),
            artifact_id: lock.artifact_id,
            lock_sha256: sha256_prefixed(&canonical),
            payload_sha256: lock.payload_sha256,
            payload_size: lock.payload_size,
        });
    }
    for artifact in &config.control_vectors {
        let lock_bytes = read_qwen_adapter_file(&artifact.lock_path, MAX_QWEN_ADAPTER_LOCK_BYTES)
            .map_err(|error| {
            BackendErrorV1::new(format!(
                "control-vector {} lock read failed: {error}",
                artifact.alias
            ))
        })?;
        let lock = parse_control_vector_lock_v1(&lock_bytes).map_err(|error| {
            BackendErrorV1::new(format!(
                "control-vector {} lock verification failed: {error}",
                artifact.alias
            ))
        })?;
        if lock.payload_size == 0 || lock.payload_size > MAX_QWEN_ADAPTER_PAYLOAD_BYTES {
            return Err(BackendErrorV1::new(format!(
                "control-vector {} payload size is outside the bounded limit",
                artifact.alias
            )));
        }
        let payload =
            read_qwen_adapter_file(&artifact.payload_path, lock.payload_size).map_err(|error| {
                BackendErrorV1::new(format!(
                    "control-vector {} payload read failed: {error}",
                    artifact.alias
                ))
            })?;
        if payload.len() as u64 != lock.payload_size
            || !sha256_prefixed(&payload).eq_ignore_ascii_case(&lock.payload_sha256)
        {
            return Err(BackendErrorV1::new(format!(
                "control-vector {} payload hash or size differs",
                artifact.alias
            )));
        }
        let canonical = serde_json::to_vec(&lock).map_err(|error| {
            BackendErrorV1::new(format!("adapter lock canonicalization failed: {error}"))
        })?;
        entries.push(QwenAdapterCatalogIdentityEntry {
            kind: "control-vector",
            alias: artifact.alias.clone(),
            artifact_id: lock.artifact_id,
            lock_sha256: sha256_prefixed(&canonical),
            payload_sha256: lock.payload_sha256,
            payload_size: lock.payload_size,
        });
    }
    Ok(qwen_adapter_catalog_identity_from_entries(entries))
}

fn load_qwen_adapter_catalog(
    config: Option<&QwenAdapterCatalogConfigV1>,
    lock: &ModelLock,
    plan: &WeightLoadPlan,
) -> Result<Option<QwenAdapterCatalogV1>, BackendErrorV1> {
    let Some(config) = config else {
        return Ok(None);
    };
    let dims = AdapterModelDimsV1::new(
        lock.model.architecture.text_config.hidden_size,
        lock.model.architecture.text_config.num_hidden_layers,
    )
    .map_err(|error| BackendErrorV1::new(error.to_string()))?;
    let mut lora = BTreeMap::new();
    for artifact in &config.lora {
        let lock_bytes = read_qwen_adapter_file(&artifact.lock_path, MAX_QWEN_ADAPTER_LOCK_BYTES)
            .map_err(|error| {
            BackendErrorV1::new(format!(
                "LoRA adapter {} lock read failed: {error}",
                artifact.alias
            ))
        })?;
        let adapter_lock =
            serde_json::from_slice::<LoraAdapterLockV1>(&lock_bytes).map_err(|error| {
                BackendErrorV1::new(format!(
                    "LoRA adapter {} lock parse failed: {error}",
                    artifact.alias
                ))
            })?;
        if adapter_lock.payload_size == 0
            || adapter_lock.payload_size > MAX_QWEN_ADAPTER_PAYLOAD_BYTES
        {
            return Err(BackendErrorV1::new(format!(
                "LoRA adapter {} payload size is outside the bounded limit",
                artifact.alias
            )));
        }
        let payload = read_qwen_adapter_file(&artifact.payload_path, adapter_lock.payload_size)
            .map_err(|error| {
                BackendErrorV1::new(format!(
                    "LoRA adapter {} payload read failed: {error}",
                    artifact.alias
                ))
            })?;
        if payload.len() as u64 != adapter_lock.payload_size {
            return Err(BackendErrorV1::new(format!(
                "LoRA adapter {} payload size changed during read",
                artifact.alias
            )));
        }
        let verified = VerifiedLoraPayloadV1::from_bytes(
            &lock_bytes,
            Arc::<[u8]>::from(payload),
            lock.fingerprint(),
            plan,
        )
        .map_err(|error| {
            BackendErrorV1::new(format!(
                "LoRA adapter {} verification failed: {error}",
                artifact.alias
            ))
        })?;
        lora.insert(artifact.alias.clone(), Arc::new(verified));
    }
    let mut control_vectors = BTreeMap::new();
    for artifact in &config.control_vectors {
        let lock_bytes = read_qwen_adapter_file(&artifact.lock_path, MAX_QWEN_ADAPTER_LOCK_BYTES)
            .map_err(|error| {
            BackendErrorV1::new(format!(
                "control-vector {} lock read failed: {error}",
                artifact.alias
            ))
        })?;
        let adapter_lock =
            serde_json::from_slice::<ControlVectorLockV1>(&lock_bytes).map_err(|error| {
                BackendErrorV1::new(format!(
                    "control-vector {} lock parse failed: {error}",
                    artifact.alias
                ))
            })?;
        if adapter_lock.payload_size == 0
            || adapter_lock.payload_size > MAX_QWEN_ADAPTER_PAYLOAD_BYTES
        {
            return Err(BackendErrorV1::new(format!(
                "control-vector {} payload size is outside the bounded limit",
                artifact.alias
            )));
        }
        let payload = read_qwen_adapter_file(&artifact.payload_path, adapter_lock.payload_size)
            .map_err(|error| {
                BackendErrorV1::new(format!(
                    "control-vector {} payload read failed: {error}",
                    artifact.alias
                ))
            })?;
        if payload.len() as u64 != adapter_lock.payload_size {
            return Err(BackendErrorV1::new(format!(
                "control-vector {} payload size changed during read",
                artifact.alias
            )));
        }
        let verified = VerifiedControlVectorPayloadV1::from_bytes(
            &lock_bytes,
            Arc::<[u8]>::from(payload),
            lock.fingerprint(),
            plan,
            dims,
        )
        .map_err(|error| {
            BackendErrorV1::new(format!(
                "control-vector {} verification failed: {error}",
                artifact.alias
            ))
        })?;
        control_vectors.insert(artifact.alias.clone(), Arc::new(verified));
    }
    Ok(Some(QwenAdapterCatalogV1 {
        lora,
        control_vectors,
    }))
}

fn resolve_qwen_adapters(
    catalog: Option<&QwenAdapterCatalogV1>,
    request: &crate::api::ModelVariantRequestV1,
) -> Result<AdapterRequestSetV1, BackendErrorV1> {
    if request.adapters().is_empty() && request.control_vectors().is_empty() {
        return Ok(AdapterRequestSetV1::disabled());
    }
    let catalog = catalog.ok_or_else(|| {
        BackendErrorV1::new("request selected adapters but no Qwen adapter catalog is configured")
    })?;
    let lora = request
        .adapters()
        .iter()
        .map(|selection| {
            let artifact = catalog.lora.get(selection.name()).ok_or_else(|| {
                BackendErrorV1::new(format!("unknown Qwen LoRA adapter {}", selection.name()))
            })?;
            Ok(LoraAdapterSelectionV1 {
                alias: selection.name().to_owned(),
                artifact: Arc::clone(artifact),
                scale: selection.scale(),
            })
        })
        .collect::<Result<Vec<_>, BackendErrorV1>>()?;
    let controls = request
        .control_vectors()
        .iter()
        .map(|selection| {
            let artifact = catalog
                .control_vectors
                .get(selection.name())
                .ok_or_else(|| {
                    BackendErrorV1::new(format!("unknown Qwen control vector {}", selection.name()))
                })?;
            Ok(ControlVectorSelectionV1 {
                alias: selection.name().to_owned(),
                artifact: Arc::clone(artifact),
                scale: selection.scale(),
            })
        })
        .collect::<Result<Vec<_>, BackendErrorV1>>()?;
    AdapterRequestSetV1::new(lora, controls)
        .map_err(|error| BackendErrorV1::new(format!("Qwen adapter request rejected: {error}")))
}

#[derive(Clone, Debug)]
pub struct Gemma4BackendConfigV1 {
    pub gguf_path: PathBuf,
    pub derived_lock_path: PathBuf,
    pub device_index: u32,
    pub target: String,
    pub completion_timeout: Duration,
    pub shutdown_timeout: Duration,
    pub context_length: u32,
    pub phase41: Phase41ProductionConfigV1,
}

impl Gemma4BackendConfigV1 {
    pub fn validate(&self) -> Result<(), BackendErrorV1> {
        if self.target.is_empty()
            || !self.target.is_ascii()
            || self.completion_timeout.is_zero()
            || self.shutdown_timeout.is_zero()
            || self.context_length == 0
        {
            return Err(BackendErrorV1::new(
                "Gemma backend target, context length, and timeouts must be valid and nonzero",
            ));
        }
        self.phase41.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ProductionRequestAuditV1 {
    pub outcome: String,
    pub target: String,
    pub weight_encoding: String,
    pub kv_cache_encoding: String,
    pub fp8_provider: Option<String>,
    pub prompt_tokens: u64,
    pub requested_max_completion_tokens: u32,
    pub completion_tokens: Option<u64>,
    pub elapsed_ns: u64,
    pub selected_backend: Option<String>,
    pub fallback_used: Option<bool>,
    pub all_dispatches_hip: Option<bool>,
    pub submission_count: Option<u64>,
    pub kernel_dispatch_count: Option<u64>,
    pub full_attention_layers: usize,
    pub linear_attention_layers: usize,
    pub logical_kv_capacity_tokens: Option<u64>,
    pub observed_kv_length_tokens: Option<u64>,
    pub physical_page_bytes: Option<u64>,
    pub kv_memory_kind: Option<String>,
    pub tokens_per_page: Option<u64>,
    pub mapped_kv_capacity_tokens: Option<u64>,
    pub committed_kv_bytes: Option<u64>,
    pub prefill_chunk_capacity_tokens: Option<u64>,
    pub prefill_chunk_count: Option<u64>,
    pub placement_total_memory_bytes: Option<u64>,
    pub placement_available_memory_bytes: Option<u64>,
    pub placement_required_bytes: Option<u64>,
    pub placement_incremental_required_bytes: Option<u64>,
    pub workspace_separate_allocation_bytes: Option<u64>,
    pub workspace_arena_bytes: Option<u64>,
    pub allocated_request_state_bytes: u64,
    pub allocated_workspace_bytes: u64,
    pub cleanup_request_state_bytes: u64,
    pub cleanup_workspace_bytes: u64,
    pub phase41: ProductionPhase41AuditV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProductionPrefixCacheResultV1 {
    Miss,
    ExactHit,
    PartialHit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProductionCheckpointOperationV1 {
    Load,
    Save,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProductionCheckpointResultV1 {
    Succeeded,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProductionDraftProviderV1 {
    Mtp,
    Ngram,
    External,
}

/// Bounded Phase 41 request evidence. Absence means that the corresponding
/// opt-in facility was disabled or did not run. This deliberately contains no
/// path, checkpoint name, model identity, token ID, or token content.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ProductionPhase41AuditV1 {
    pub prefix_cache_result: Option<ProductionPrefixCacheResultV1>,
    pub prefix_shared_pages: u64,
    pub prefix_cow_pages: u64,
    pub prefix_copied_bytes: u64,
    pub checkpoint_operation: Option<ProductionCheckpointOperationV1>,
    pub checkpoint_result: Option<ProductionCheckpointResultV1>,
    pub context_shift_count: u64,
    pub draft_provider: Option<ProductionDraftProviderV1>,
    pub draft_proposed_tokens: u64,
    pub draft_accepted_tokens: u64,
    pub draft_rejected_tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProductionShutdownAuditV1 {
    pub schema_version: String,
    pub target: String,
    pub model_fingerprint: String,
    pub plan_digest: String,
    pub model_ready_current_bytes: u64,
    pub final_current_bytes: u64,
    pub final_request_state_bytes: u64,
    pub final_workspace_bytes: u64,
    pub retryable_cleanup: usize,
    pub durable_quarantine: usize,
    pub requests: Vec<ProductionRequestAuditV1>,
}

enum ShutdownStateV1 {
    Active,
    Pending {
        session: Arc<ExecutionSession>,
        model_ready_current_bytes: u64,
    },
    Complete(ProductionShutdownAuditV1),
}

pub struct QwenChatBackendV1 {
    state: Mutex<Option<QwenBackendStateV1>>,
    audits: Mutex<Vec<ProductionRequestAuditV1>>,
    shutdown: Mutex<ShutdownStateV1>,
    shutdown_timeout: Duration,
    identity: BackendIdentityV1,
}

enum QwenPrefixCacheRuntimeV1 {
    Disabled,
    Enabled {
        index: PrefixCacheV1,
        states: HashMap<PrefixEntryIdV1, QwenPrefixStateV1>,
    },
}

enum GemmaPrefixCacheRuntimeV1 {
    Disabled,
    Enabled {
        index: PrefixCacheV1,
        states: HashMap<PrefixEntryIdV1, Gemma4PrefixStateV1>,
    },
}

struct QwenPrefixHitV1 {
    prefix: QwenPrefixStateV1,
    lease: PrefixLeaseV1,
    matched_tokens: Vec<u32>,
    kind: PrefixLookupKind,
}

struct QwenCheckpointRuntimeV1 {
    store: Arc<CheckpointStore>,
    loaded: Option<Arc<SessionCheckpoint>>,
    save_name: Option<String>,
}

struct QwenCapturedChatCheckpointV1 {
    checkpoint: SessionCheckpoint,
    text: String,
    reasoning: Option<String>,
    prompt_tokens: u64,
}

struct QwenCheckpointSaveV1 {
    store: Arc<CheckpointStore>,
    name: String,
    identity: CheckpointIdentity,
    prompt_tokens: Vec<u32>,
    status: Arc<AtomicU8>,
}

const CHECKPOINT_STATUS_NONE: u8 = 0;
const CHECKPOINT_STATUS_SUCCEEDED: u8 = 1;
const CHECKPOINT_STATUS_FAILED: u8 = 2;

struct GemmaPrefixHitV1 {
    prefix: Gemma4PrefixStateV1,
    lease: PrefixLeaseV1,
    matched_tokens: Vec<u32>,
    kind: PrefixLookupKind,
}

impl QwenPrefixCacheRuntimeV1 {
    fn new(config: &PrefixCacheStartupConfigV1) -> Result<Self, BackendErrorV1> {
        match config {
            PrefixCacheStartupConfigV1::Disabled => Ok(Self::Disabled),
            PrefixCacheStartupConfigV1::Enabled {
                max_entries,
                max_logical_tokens,
                max_resident_bytes,
            } => Ok(Self::Enabled {
                index: PrefixCacheV1::new(
                    PrefixCacheConfigV1::new(
                        usize::from(*max_entries),
                        *max_logical_tokens,
                        *max_resident_bytes,
                    )
                    .map_err(|error| BackendErrorV1::new(error.to_string()))?,
                ),
                states: HashMap::new(),
            }),
        }
    }

    fn baseline_bytes(&self) -> Result<u64, BackendErrorV1> {
        match self {
            Self::Disabled => Ok(0),
            Self::Enabled { states, .. } => checked_prefix_request_state_baseline(
                states
                    .values()
                    .map(|prefix| prefix.fork_audit().destination_owned_bytes()),
            ),
        }
    }

    fn lookup(
        &self,
        identity: &PrefixStateIdentityV1,
        tokens: &[u32],
    ) -> Result<Option<QwenPrefixHitV1>, BackendErrorV1> {
        let Self::Enabled { index, states } = self else {
            return Ok(None);
        };
        let Some(result) = index
            .lookup(identity, tokens)
            .map_err(|error| BackendErrorV1::new(error.to_string()))?
        else {
            return Ok(None);
        };
        let prefix = states
            .get(&result.lease().entry_id())
            .cloned()
            .ok_or_else(|| {
                BackendErrorV1::new("prefix index referenced an absent Qwen state owner")
            })?;
        let matched_tokens = result.lease().tokens().to_vec();
        let kind = result.kind();
        let lease = result.into_lease();
        Ok(Some(QwenPrefixHitV1 {
            prefix,
            lease,
            matched_tokens,
            kind,
        }))
    }

    fn publish(
        &mut self,
        identity: PrefixStateIdentityV1,
        tokens: &[u32],
        prefix: QwenPrefixStateV1,
    ) -> Result<(), BackendErrorV1> {
        let Self::Enabled { index, states } = self else {
            return Ok(());
        };
        let logical_tokens = u64::try_from(tokens.len())
            .map_err(|_| BackendErrorV1::new("prefix token count overflowed u64"))?;
        let audit = prefix.fork_audit();
        let id = index
            .publish(
                PrefixCacheKeyV1::new(identity, tokens)
                    .map_err(|error| BackendErrorV1::new(error.to_string()))?,
                PrefixCacheValueV1::new(
                    logical_tokens,
                    audit.cache_resident_bytes(),
                    *prefix.graph_semantics_digest(),
                )
                .map_err(|error| BackendErrorV1::new(error.to_string()))?,
            )
            .map_err(|error| BackendErrorV1::new(error.to_string()))?;
        states.insert(id, prefix);
        reconcile_prefix_states(index, states)?;
        Ok(())
    }
}

impl GemmaPrefixCacheRuntimeV1 {
    fn new(config: &PrefixCacheStartupConfigV1) -> Result<Self, BackendErrorV1> {
        match config {
            PrefixCacheStartupConfigV1::Disabled => Ok(Self::Disabled),
            PrefixCacheStartupConfigV1::Enabled {
                max_entries,
                max_logical_tokens,
                max_resident_bytes,
            } => Ok(Self::Enabled {
                index: PrefixCacheV1::new(
                    PrefixCacheConfigV1::new(
                        usize::from(*max_entries),
                        *max_logical_tokens,
                        *max_resident_bytes,
                    )
                    .map_err(|error| BackendErrorV1::new(error.to_string()))?,
                ),
                states: HashMap::new(),
            }),
        }
    }

    fn baseline_bytes(&self) -> Result<u64, BackendErrorV1> {
        match self {
            Self::Disabled => Ok(0),
            Self::Enabled { states, .. } => checked_prefix_request_state_baseline(
                states
                    .values()
                    .map(|prefix| prefix.fork_audit().destination_owned_bytes()),
            ),
        }
    }

    fn lookup(
        &self,
        identity: &PrefixStateIdentityV1,
        tokens: &[u32],
    ) -> Result<Option<GemmaPrefixHitV1>, BackendErrorV1> {
        let Self::Enabled { index, states } = self else {
            return Ok(None);
        };
        let Some(result) = index
            .lookup(identity, tokens)
            .map_err(|error| BackendErrorV1::new(error.to_string()))?
        else {
            return Ok(None);
        };
        let prefix = states
            .get(&result.lease().entry_id())
            .cloned()
            .ok_or_else(|| {
                BackendErrorV1::new("prefix index referenced an absent Gemma state owner")
            })?;
        let matched_tokens = result.lease().tokens().to_vec();
        let kind = result.kind();
        let lease = result.into_lease();
        Ok(Some(GemmaPrefixHitV1 {
            prefix,
            lease,
            matched_tokens,
            kind,
        }))
    }

    fn publish(
        &mut self,
        identity: PrefixStateIdentityV1,
        tokens: &[u32],
        prefix: Gemma4PrefixStateV1,
    ) -> Result<(), BackendErrorV1> {
        let Self::Enabled { index, states } = self else {
            return Ok(());
        };
        let logical_tokens = u64::try_from(tokens.len())
            .map_err(|_| BackendErrorV1::new("prefix token count overflowed u64"))?;
        let audit = prefix.fork_audit();
        let id = index
            .publish(
                PrefixCacheKeyV1::new(identity, tokens)
                    .map_err(|error| BackendErrorV1::new(error.to_string()))?,
                PrefixCacheValueV1::new(
                    logical_tokens,
                    audit.cache_resident_bytes(),
                    *prefix.plan_digest(),
                )
                .map_err(|error| BackendErrorV1::new(error.to_string()))?,
            )
            .map_err(|error| BackendErrorV1::new(error.to_string()))?;
        states.insert(id, prefix);
        reconcile_prefix_states(index, states)?;
        Ok(())
    }
}

fn reconcile_prefix_states<T>(
    index: &PrefixCacheV1,
    states: &mut HashMap<PrefixEntryIdV1, T>,
) -> Result<(), BackendErrorV1> {
    let published = index
        .published_entry_ids()
        .map_err(|error| BackendErrorV1::new(error.to_string()))?
        .into_iter()
        .collect::<HashSet<_>>();
    states.retain(|id, _| published.contains(id));
    Ok(())
}

fn checked_prefix_request_state_baseline(
    destination_owned_bytes: impl IntoIterator<Item = u64>,
) -> Result<u64, BackendErrorV1> {
    destination_owned_bytes
        .into_iter()
        .try_fold(0_u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or_else(|| BackendErrorV1::new("prefix request-state baseline overflowed u64"))
        })
}

struct QwenBackendStateV1 {
    lock: Option<ModelLock>,
    moe_artifact: Option<Arc<VerifiedQwen35Moe>>,
    gguf_moe: Option<Arc<VerifiedGgufQwen35Moe>>,
    reasoning_close_token_ids: Vec<u32>,
    stop_policy: GenerationStopPolicyV1,
    tokenizer: TokenizerFrontendV1,
    renderer: Qwen35ChatTemplateV1,
    plan: WeightLoadPlan,
    resident: QwenResidentModel,
    mtp_resident: Option<QwenResidentModel>,
    mtp_plan: Option<WeightLoadPlan>,
    session: Arc<ExecutionSession>,
    target: String,
    model_ready_current_bytes: u64,
    sidecar: Option<Arc<VerifiedFp8Sidecar>>,
    nvfp4_sidecar: Option<Arc<VerifiedNvfp4Sidecar>>,
    fp8_provider: Option<String>,
    cache: Option<Arc<VerifiedCache>>,
    gguf_source: Option<Arc<VerifiedGgufWeightSource>>,
    vision_manifest: Option<QwenVisionManifest>,
    vision_resident: Option<QwenVisionResidentModel>,
    completion_timeout: Duration,
    kv_cache_encoding: KvCacheEncoding,
    phase41: Phase41ProductionConfigV1,
    prefix_cache: QwenPrefixCacheRuntimeV1,
    checkpoint: Option<QwenCheckpointRuntimeV1>,
    // Dense BF16 checkpoint identities use the stable KV descriptor from the
    // resident seed graph.  Request graphs may vary their token row count,
    // but keep the same state capacity and descriptor semantics.
    persistent_checkpoint_descriptor_digest: Option<[u8; 32]>,
    persistent_capture_requested: bool,
    persistent_capture: Option<QwenCapturedChatCheckpointV1>,
    adapter_catalog: Option<QwenAdapterCatalogV1>,
}

struct QwenAdapterCatalogV1 {
    lora: BTreeMap<String, Arc<VerifiedLoraPayloadV1>>,
    control_vectors: BTreeMap<String, Arc<VerifiedControlVectorPayloadV1>>,
}

struct QwenAdapterCatalogIdentityEntry {
    kind: &'static str,
    alias: String,
    artifact_id: String,
    lock_sha256: String,
    payload_sha256: String,
    payload_size: u64,
}

fn qwen_adapter_catalog_identity_from_entries(
    entries: impl IntoIterator<Item = QwenAdapterCatalogIdentityEntry>,
) -> String {
    let entries = entries.into_iter().collect::<Vec<_>>();
    if entries.is_empty() {
        return "adapter:none-v1".to_owned();
    }
    let mut digest = Sha256::new();
    digest.update(b"sllm-qwen-adapter-catalog-v1");
    for entry in entries {
        digest.update([if entry.kind == "lora" { 1 } else { 2 }]);
        digest.update((entry.alias.len() as u64).to_le_bytes());
        digest.update(entry.alias.as_bytes());
        let identity = format!(
            "{}:{}:{}:{}:{}:{}",
            entry.kind,
            entry.artifact_id,
            entry.lock_sha256,
            entry.payload_sha256,
            entry.payload_size,
            "v1"
        );
        digest.update((identity.len() as u64).to_le_bytes());
        digest.update(identity.as_bytes());
    }
    let digest = digest.finalize();
    let mut output = String::from("adapter:catalog-v1:sha256:");
    for byte in digest {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

impl QwenAdapterCatalogV1 {
    fn identity(&self) -> String {
        let entries = self
            .lora
            .iter()
            .map(|(alias, artifact)| QwenAdapterCatalogIdentityEntry {
                kind: "lora",
                alias: alias.clone(),
                artifact_id: artifact.identity().artifact_id().to_owned(),
                lock_sha256: artifact.identity().lock_sha256().to_owned(),
                payload_sha256: artifact.identity().payload_sha256().to_owned(),
                payload_size: artifact.identity().payload_size(),
            })
            .chain(self.control_vectors.iter().map(|(alias, artifact)| {
                QwenAdapterCatalogIdentityEntry {
                    kind: "control-vector",
                    alias: alias.clone(),
                    artifact_id: artifact.identity().artifact_id().to_owned(),
                    lock_sha256: artifact.identity().lock_sha256().to_owned(),
                    payload_sha256: artifact.identity().payload_sha256().to_owned(),
                    payload_size: artifact.identity().payload_size(),
                }
            }));
        qwen_adapter_catalog_identity_from_entries(entries)
    }
}

pub struct Gemma4ChatBackendV1 {
    state: Mutex<Option<Gemma4BackendStateV1>>,
    audits: Mutex<Vec<ProductionRequestAuditV1>>,
    shutdown: Mutex<ShutdownStateV1>,
    shutdown_timeout: Duration,
    identity: BackendIdentityV1,
}

struct Gemma4BackendStateV1 {
    _lock: Gemma4ModelLock,
    tokenizer: TokenizerFrontendV1,
    stop_policy: GenerationStopPolicyV1,
    plan: WeightLoadPlan,
    resident: Gemma4ResidentModel,
    session: Arc<ExecutionSession>,
    target: String,
    model_ready_current_bytes: u64,
    weight_encoding: String,
    kv_bytes_per_token: u64,
    phase41: Phase41ProductionConfigV1,
    prefix_cache: GemmaPrefixCacheRuntimeV1,
    checkpoint: Option<QwenCheckpointRuntimeV1>,
    checkpoint_descriptor_digest: Option<[u8; 32]>,
}

#[derive(Clone)]
struct BackendIdentityV1 {
    target: String,
    model_fingerprint: String,
    plan_digest: String,
    model_ready_current_bytes: u64,
    context_length: u32,
    recommended_context_tokens: u32,
}

fn context_policy_identity_version(config: &ContextWindowStartupConfigV1) -> u32 {
    match config {
        ContextWindowStartupConfigV1::Disabled => 0,
        ContextWindowStartupConfigV1::KeepPrefixRecentV1 { .. } => {
            sllm_core::CONTEXT_POSITION_POLICY_VERSION_V1
        }
    }
}

fn qwen_prefix_identity(
    state: &QwenBackendStateV1,
    graph: &QwenGraph,
    state_capacity: u64,
    adapter_identity: &str,
) -> Result<PrefixStateIdentityV1, BackendErrorV1> {
    let descriptor = graph
        .states()
        .iter()
        .find_map(|state| match state.descriptor() {
            QwenGraphStateDescriptor::Kv(descriptor) => Some(descriptor),
            QwenGraphStateDescriptor::Linear(_) => None,
        })
        .ok_or_else(|| BackendErrorV1::new("Qwen prefix identity has no KV descriptor"))?;
    let layout = descriptor.layout();
    let heads = u32::try_from(layout.heads())
        .map_err(|_| BackendErrorV1::new("Qwen KV head count overflowed u32"))?;
    let head_dim = u32::try_from(layout.head_dim())
        .map_err(|_| BackendErrorV1::new("Qwen KV head dimension overflowed u32"))?;
    let derived_identity = format!("{}:capacity={state_capacity}", state.plan.digest_hex());
    let renderer_identity = format!(
        "qwen35-chat-v{}:{}",
        state.renderer.version(),
        state.renderer.consistency_label()
    );
    let target_semantics = format!(
        "{}:{}",
        state.target,
        state.fp8_provider.as_deref().unwrap_or("bf16")
    );
    PrefixStateIdentityV1::new(
        state.plan.lock_fingerprint.as_bytes(),
        derived_identity.as_bytes(),
        adapter_identity.as_bytes(),
        renderer_identity.as_bytes(),
        state.tokenizer.snapshot().fingerprint().as_bytes(),
        state.kv_cache_encoding,
        PrefixKvLayoutV1::new(heads, head_dim)
            .map_err(|error| BackendErrorV1::new(error.to_string()))?,
        target_semantics.as_bytes(),
        context_policy_identity_version(&state.phase41.context_window),
    )
    .map_err(|error| BackendErrorV1::new(error.to_string()))
}

fn qwen_checkpoint_kv_descriptor_digest(graph: &QwenGraph) -> [u8; 32] {
    let mut descriptors = BTreeMap::new();
    for state in graph.states() {
        if let QwenGraphStateDescriptor::Kv(descriptor) = state.descriptor() {
            descriptors.insert(state.layer(), descriptor);
        }
    }
    let mut digest = Sha256::new();
    digest.update(b"sllm-qwen-kv-descriptor-v1");
    digest.update((descriptors.len() as u64).to_le_bytes());
    for (layer, descriptor) in descriptors {
        digest.update(layer.to_le_bytes());
        digest.update(descriptor.layer_id().to_le_bytes());
        digest.update(descriptor.capacity().to_le_bytes());
        digest.update((descriptor.layout().heads() as u64).to_le_bytes());
        digest.update((descriptor.layout().head_dim() as u64).to_le_bytes());
        let encoding = match descriptor.cache_encoding() {
            KvCacheEncoding::Fp16 => 0_u8,
            KvCacheEncoding::Fp8E4M3Fn => 1_u8,
            KvCacheEncoding::Fp8E4M3FnStatic => 2_u8,
            KvCacheEncoding::Nvfp4 => 3_u8,
        };
        digest.update([encoding]);
        if let Some((key, value)) = descriptor.static_fp8_scales() {
            digest.update([1]);
            digest.update(key.to_bits().to_le_bytes());
            digest.update(value.to_bits().to_le_bytes());
        } else {
            digest.update([0]);
        }
    }
    digest.finalize().into()
}

fn qwen_checkpoint_context_policy_digest() -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"sllm-qwen-checkpoint-context-disabled-v1");
    digest.finalize().into()
}

fn gemma_checkpoint_context_policy_digest() -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"sllm-gemma4-checkpoint-context-disabled-v1");
    digest.finalize().into()
}

fn gemma_checkpoint_kv_descriptor_digest(
    graph: &sllm_core::Gemma4Graph,
    source: &VerifiedGgufGemmaSource,
) -> Result<[u8; 32], BackendErrorV1> {
    let mut full_layers = BTreeMap::new();
    let mut sliding_layers = BTreeMap::new();
    for descriptor in graph.kv_descriptors() {
        if let Some(retention_window) = descriptor.retention_window {
            sliding_layers.insert(
                descriptor.layer,
                (
                    descriptor.heads,
                    descriptor.head_dim,
                    descriptor.capacity,
                    retention_window,
                ),
            );
            continue;
        }
        let scale = source.kv_scale(descriptor.layer).ok_or_else(|| {
            BackendErrorV1::new("Gemma checkpoint descriptor scale is unavailable")
        })?;
        let state_descriptor = sllm_core::KvStateDescriptor::new_with_static_fp8(
            descriptor.layer,
            descriptor.capacity,
            usize::try_from(descriptor.heads)
                .map_err(|_| BackendErrorV1::new("Gemma checkpoint head count overflowed"))?,
            usize::try_from(descriptor.head_dim)
                .map_err(|_| BackendErrorV1::new("Gemma checkpoint head dimension overflowed"))?,
            scale.key_decode_scale(),
            scale.value_decode_scale(),
        )
        .map_err(|_| BackendErrorV1::new("Gemma checkpoint descriptor construction failed"))?;
        full_layers.insert(descriptor.layer, state_descriptor);
    }
    if full_layers.is_empty() {
        return Err(BackendErrorV1::new(
            "Gemma checkpoint graph has no full-attention KV descriptors",
        ));
    }
    let mut digest = Sha256::new();
    digest.update(b"sllm-gemma4-checkpoint-descriptors-v1");
    digest.update([3_u8]);
    digest.update((full_layers.len() as u64).to_le_bytes());
    for (&layer, descriptor) in &full_layers {
        digest.update([1]);
        digest.update(layer.to_le_bytes());
        digest.update(descriptor.layer_id().to_le_bytes());
        digest.update(descriptor.capacity().to_le_bytes());
        digest.update((descriptor.layout().heads() as u64).to_le_bytes());
        digest.update((descriptor.layout().head_dim() as u64).to_le_bytes());
        let (key, value) = descriptor
            .static_fp8_scales()
            .ok_or_else(|| BackendErrorV1::new("Gemma checkpoint descriptor scale is absent"))?;
        digest.update([1]);
        digest.update(key.to_bits().to_le_bytes());
        digest.update(value.to_bits().to_le_bytes());
    }
    digest.update((sliding_layers.len() as u64).to_le_bytes());
    for (&layer, &(heads, head_dim, capacity, retention_window)) in &sliding_layers {
        digest.update([2]);
        digest.update(layer.to_le_bytes());
        digest.update(heads.to_le_bytes());
        digest.update(head_dim.to_le_bytes());
        digest.update(capacity.to_le_bytes());
        digest.update(retention_window.to_le_bytes());
        digest.update(b"bf16-unquantized-key-value");
    }
    Ok(digest.finalize().into())
}

fn gemma_checkpoint_identity(
    model_fingerprint: &str,
    plan: &WeightLoadPlan,
    tokenizer: &TokenizerFrontendV1,
    target: &str,
    descriptor_digest: [u8; 32],
    tokens: &[u32],
) -> Result<CheckpointIdentity, BackendErrorV1> {
    CheckpointIdentity::for_tokens(
        model_fingerprint.to_owned(),
        format!("derived-artifact:{}", plan.digest_hex()),
        "adapter:none-v1",
        "gemma4-raw-chat-v1",
        tokenizer.snapshot().fingerprint().to_owned(),
        format!("{target}:mixed-nvfp4-w4a4-fp8-w8a8"),
        plan.digest_hex(),
        tokens,
        KvCacheEncoding::Fp8E4M3FnStatic,
        descriptor_digest,
        gemma_checkpoint_context_policy_digest(),
    )
    .map_err(|_| BackendErrorV1::new("Gemma checkpoint identity construction failed"))
}

fn build_gemma_checkpoint_runtime(
    config: &CheckpointStartupConfigV1,
    graph: &sllm_core::Gemma4Graph,
    plan: &WeightLoadPlan,
    tokenizer: &TokenizerFrontendV1,
    target: &str,
    descriptor_digest: [u8; 32],
) -> Result<Option<QwenCheckpointRuntimeV1>, BackendErrorV1> {
    let CheckpointStartupConfigV1::Enabled {
        directory,
        quota_bytes,
        load_name,
        save_name,
    } = config
    else {
        return Ok(None);
    };
    let store = Arc::new(
        CheckpointStore::new(directory, *quota_bytes)
            .map_err(|_| BackendErrorV1::new("Gemma checkpoint store initialization failed"))?,
    );
    let loaded = load_name
        .as_deref()
        .map(|name| {
            store
                .load_validated(name)
                .map(Arc::new)
                .map_err(|_| BackendErrorV1::new("Gemma checkpoint load failed"))
        })
        .transpose()?;
    if let Some(checkpoint) = loaded.as_ref() {
        let expected = gemma_checkpoint_identity(
            graph.lock_fingerprint(),
            plan,
            tokenizer,
            target,
            descriptor_digest,
            &checkpoint.payload.token_history,
        )?;
        if checkpoint.header.identity != expected {
            return Err(BackendErrorV1::new(
                "Gemma checkpoint identity differs from the running model",
            ));
        }
    }
    Ok(Some(QwenCheckpointRuntimeV1 {
        store,
        loaded,
        save_name: save_name.clone(),
    }))
}

#[allow(clippy::too_many_arguments)]
fn qwen_checkpoint_identity(
    graph: &QwenGraph,
    plan: &WeightLoadPlan,
    tokenizer: &TokenizerFrontendV1,
    renderer: &Qwen35ChatTemplateV1,
    target: &str,
    fp8_provider: Option<&str>,
    adapter_identity: &str,
    kv_cache_encoding: KvCacheEncoding,
    tokens: &[u32],
) -> Result<CheckpointIdentity, BackendErrorV1> {
    qwen_checkpoint_identity_with_descriptor_digest(
        graph.model_fingerprint(),
        plan,
        tokenizer,
        renderer,
        target,
        fp8_provider,
        adapter_identity,
        kv_cache_encoding,
        qwen_checkpoint_kv_descriptor_digest(graph),
        tokens,
    )
}

#[allow(clippy::too_many_arguments)]
fn qwen_checkpoint_identity_with_descriptor_digest(
    model_fingerprint: &str,
    plan: &WeightLoadPlan,
    tokenizer: &TokenizerFrontendV1,
    renderer: &Qwen35ChatTemplateV1,
    target: &str,
    fp8_provider: Option<&str>,
    adapter_identity: &str,
    kv_cache_encoding: KvCacheEncoding,
    descriptor_digest: [u8; 32],
    tokens: &[u32],
) -> Result<CheckpointIdentity, BackendErrorV1> {
    let renderer_identity = format!(
        "qwen35-chat-v{}:{}",
        renderer.version(),
        renderer.consistency_label()
    );
    let target_semantics = format!("{}:{}", target, fp8_provider.unwrap_or("bf16"));
    CheckpointIdentity::for_tokens(
        model_fingerprint.to_owned(),
        format!("derived-artifact:{}", plan.digest_hex()),
        adapter_identity,
        renderer_identity,
        tokenizer.snapshot().fingerprint().to_owned(),
        target_semantics,
        plan.digest_hex(),
        tokens,
        kv_cache_encoding,
        descriptor_digest,
        qwen_checkpoint_context_policy_digest(),
    )
    .map_err(|_| BackendErrorV1::new("Qwen checkpoint identity construction failed"))
}

#[allow(clippy::too_many_arguments)]
fn build_qwen_checkpoint_runtime(
    config: &CheckpointStartupConfigV1,
    graph: &QwenGraph,
    _plan: &WeightLoadPlan,
    _tokenizer: &TokenizerFrontendV1,
    _renderer: &Qwen35ChatTemplateV1,
    _target: &str,
    fp8_provider: Option<&str>,
    _kv_cache_encoding: KvCacheEncoding,
) -> Result<Option<QwenCheckpointRuntimeV1>, BackendErrorV1> {
    let CheckpointStartupConfigV1::Enabled {
        directory,
        quota_bytes,
        load_name,
        save_name,
    } = config
    else {
        return Ok(None);
    };
    if fp8_provider.is_some() || graph.is_mtp() || graph.is_multimodal() {
        return Err(BackendErrorV1::new(
            "Qwen prompt checkpoint requires the dense BF16 text graph",
        ));
    }
    let store = Arc::new(
        CheckpointStore::new(directory, *quota_bytes)
            .map_err(|_| BackendErrorV1::new("Qwen checkpoint store initialization failed"))?,
    );
    let loaded = load_name
        .as_deref()
        .map(|name| {
            store
                .load_validated(name)
                .map(Arc::new)
                .map_err(|_| BackendErrorV1::new("Qwen checkpoint load failed"))
        })
        .transpose()?;
    Ok(Some(QwenCheckpointRuntimeV1 {
        store,
        loaded,
        save_name: save_name.clone(),
    }))
}

fn gemma_prefix_identity(
    state: &Gemma4BackendStateV1,
) -> Result<PrefixStateIdentityV1, BackendErrorV1> {
    let renderer_identity = b"gemma4-raw-chat-v1";
    PrefixStateIdentityV1::new(
        state.plan.lock_fingerprint.as_bytes(),
        state.plan.digest_hex().as_bytes(),
        b"adapter:none-v1",
        renderer_identity,
        state.tokenizer.snapshot().fingerprint().as_bytes(),
        KvCacheEncoding::Fp8E4M3FnStatic,
        PrefixKvLayoutV1::new(1, 512).map_err(|error| BackendErrorV1::new(error.to_string()))?,
        state.target.as_bytes(),
        context_policy_identity_version(&state.phase41.context_window),
    )
    .map_err(|error| BackendErrorV1::new(error.to_string()))
}

fn require_request_memory_baseline(
    snapshot: AllocationSnapshot,
    expected_request_state_bytes: u64,
    boundary: &str,
) -> Result<(), BackendErrorV1> {
    if snapshot.poisoned()
        || snapshot.request_state().current_bytes() != expected_request_state_bytes
        || snapshot.workspace().current_bytes() != 0
    {
        return Err(BackendErrorV1::new(format!(
            "{boundary} differs from the prefix-cache request-state baseline"
        )));
    }
    Ok(())
}

fn production_prefix_kind(kind: PrefixLookupKind) -> ProductionPrefixCacheResultV1 {
    match kind {
        PrefixLookupKind::ExactHit => ProductionPrefixCacheResultV1::ExactHit,
        PrefixLookupKind::PartialHit => ProductionPrefixCacheResultV1::PartialHit,
        PrefixLookupKind::Miss => ProductionPrefixCacheResultV1::Miss,
    }
}

fn production_ngram_provider(order: u8) -> Result<NgramDraftProviderV1, BackendErrorV1> {
    NgramDraftProviderV1::new(1, usize::from(order))
        .map_err(|error| BackendErrorV1::new(error.to_string()))
}

fn require_prompt_only_prefix(
    committed_length: u64,
    prompt_token_count: usize,
) -> Result<(), GenerationServiceError> {
    let prompt_token_count = u64::try_from(prompt_token_count).map_err(|_| {
        GenerationServiceError::Execution("prompt token count overflowed u64".to_owned())
    })?;
    if committed_length != prompt_token_count {
        return Err(GenerationServiceError::Execution(
            "prefix publication included state outside the immutable prompt".to_owned(),
        ));
    }
    Ok(())
}

impl QwenChatBackendV1 {
    pub fn open(config: QwenBackendConfigV1) -> Result<Self, BackendErrorV1> {
        config.validate()?;
        validate_qwen_phase41_operational_config(&config.phase41)?;
        let derived = read_derived_gguf_lock(&config.derived_lock_path).map_err(|error| {
            BackendErrorV1::new(format!("derived GGUF lock validation failed: {error}"))
        })?;
        if derived.semantic_model_id.starts_with("qwen35moe:") {
            if !matches!(
                config.phase41.checkpoint,
                CheckpointStartupConfigV1::Disabled
            ) {
                return Err(BackendErrorV1::new(
                    "Qwen prompt checkpoint requires the dense BF16 text graph",
                ));
            }
            if config.kv_cache_encoding != KvCacheEncoding::Fp16 {
                return Err(BackendErrorV1::new(
                    "Qwen MoE currently requires FP16 KV cache",
                ));
            }
            return Self::open_gguf_moe(config, derived);
        }
        let lock = match builtin_reviewed_model_lock(&derived.source_lock_fingerprints).map_err(
            |error| BackendErrorV1::new(format!("built-in model lock resolution failed: {error}")),
        )? {
            ReviewedModelLock::Qwen35(lock) => lock,
            ReviewedModelLock::Gemma4(_) => {
                return Err(BackendErrorV1::new(
                    "Qwen backend requires a derived GGUF for a reviewed Qwen model",
                ));
            }
        };
        let verified = verify_derived_gguf(derived, &config.gguf_path)
            .map_err(|error| BackendErrorV1::new(format!("GGUF verification failed: {error}")))?;
        let (source, plan) = build_verified_gguf_qwen_weight_load_plan(
            &lock,
            verified,
            QwenComponentSelection::TEXT_ONLY,
        )
        .map_err(|error| {
            BackendErrorV1::new(format!("verified GGUF model load plan failed: {error}"))
        })?;
        let adapter_catalog =
            load_qwen_adapter_catalog(config.adapter_catalog.as_ref(), &lock, &plan)?;
        if source.has_fp8_recipe() && adapter_catalog.is_some() {
            return Err(BackendErrorV1::new(
                "Qwen adapter catalog requires the dense BF16 GGUF artifact",
            ));
        }
        let source = Arc::new(source);
        let tokenizer =
            TokenizerFrontendV1::from_qwen35_gguf(&lock, source.gguf()).map_err(|error| {
                BackendErrorV1::new(format!("verified tokenizer construction failed: {error}"))
            })?;
        let reasoning_close_token_ids = validate_qwen_reasoning_close_marker(&tokenizer)?;
        let renderer =
            Qwen35ChatTemplateV1::from_qwen35_gguf(&lock, source.gguf()).map_err(|error| {
                BackendErrorV1::new(format!(
                    "verified chat renderer construction failed: {error}"
                ))
            })?;
        let gguf_fp8_provider = source
            .has_fp8_recipe()
            .then(|| select_gguf_fp8_provider(&config.target))
            .transpose()
            .map_err(BackendErrorV1::new)?;
        let seed_graph = if source.has_fp8_recipe() {
            build_qwen35_gguf_fp8_graph(
                &lock,
                &plan,
                &source,
                1,
                u64::from(config.context_length),
                gguf_fp8_dtype(gguf_fp8_provider.expect("validated GGUF FP8 provider")),
                config.kv_cache_encoding,
            )
        } else {
            build_qwen35_graph_with_kv_cache_encoding(
                &lock,
                &plan,
                1,
                u64::from(config.context_length),
                config.kv_cache_encoding,
            )
        }
        .map_err(|error| {
            BackendErrorV1::new(format!("resident seed graph construction failed: {error}"))
        })?;
        let checkpoint_graph = seed_graph.clone();
        let backend = HipBackend::connect()
            .map_err(|error| BackendErrorV1::new(format!("HIP backend is unavailable: {error}")))?;
        let session_request = ExecutionSessionRequest::new(config.device_index, &config.target)
            .map_err(|error| BackendErrorV1::new(format!("HIP session request failed: {error}")))?;
        let session = backend
            .open_execution_session(session_request)
            .map_err(|error| {
                BackendErrorV1::new(format!("exact HIP execution session failed: {error}"))
            })?;
        let resident = QwenResidentModel::new_gguf(
            Arc::clone(&session),
            seed_graph,
            plan.clone(),
            Arc::clone(&source),
            config.completion_timeout,
        )
        .map_err(|error| BackendErrorV1::new(format!("resident model load failed: {error}")))?;
        let vision_manifest = if lock.fingerprint() == sllm_core::QWEN35_4B_FINGERPRINT {
            Some(
                build_verified_gguf_qwen35_vision_manifest(&lock, &source).map_err(|error| {
                    BackendErrorV1::new(format!("GGUF vision manifest validation failed: {error}"))
                })?,
            )
        } else {
            None
        };
        let (mtp_resident, mtp_plan) = if !source.has_fp8_recipe()
            && config.target == "gfx1201"
            && config.kv_cache_encoding == KvCacheEncoding::Fp16
            && lock.fingerprint() == sllm_core::QWEN35_4B_FINGERPRINT
        {
            let mtp_plan = source
                .build_qwen_weight_load_plan(&lock, QwenComponentSelection::MTP_ONLY)
                .map_err(|error| {
                    BackendErrorV1::new(format!("GGUF MTP load plan validation failed: {error}"))
                })?;
            let mtp_graph = build_qwen35_mtp_graph(&lock, &mtp_plan, 1).map_err(|error| {
                BackendErrorV1::new(format!("GGUF MTP resident graph failed: {error}"))
            })?;
            let mtp_resident = QwenResidentModel::new_gguf(
                Arc::clone(&session),
                mtp_graph,
                mtp_plan.clone(),
                Arc::clone(&source),
                config.completion_timeout,
            )
            .map_err(|error| {
                BackendErrorV1::new(format!("GGUF MTP resident load failed: {error}"))
            })?;
            (Some(mtp_resident), Some(mtp_plan))
        } else {
            (None, None)
        };
        let ready = session.memory_snapshot();
        require_clean_request_memory(ready, "model-ready")?;
        let model_ready_current_bytes = ready.model_resident().current_bytes();
        if model_ready_current_bytes == 0 || ready.current_bytes() != model_ready_current_bytes {
            return Err(BackendErrorV1::new(
                "model-ready allocation accounting is not resident-only",
            ));
        }
        let fp8_provider = gguf_fp8_provider.map(str::to_owned);
        let identity = BackendIdentityV1 {
            target: config.target.clone(),
            model_fingerprint: lock.fingerprint().to_owned(),
            plan_digest: plan.digest_hex(),
            model_ready_current_bytes,
            context_length: config.context_length,
            recommended_context_tokens: QWEN35_RECOMMENDED_CONTEXT_TOKENS as u32,
        };
        let prefix_cache = QwenPrefixCacheRuntimeV1::new(&config.phase41.prefix_cache)?;
        let checkpoint = build_qwen_checkpoint_runtime(
            &config.phase41.checkpoint,
            &checkpoint_graph,
            &plan,
            &tokenizer,
            &renderer,
            &config.target,
            fp8_provider.as_deref(),
            config.kv_cache_encoding,
        )?;
        Ok(Self {
            state: Mutex::new(Some(QwenBackendStateV1 {
                reasoning_close_token_ids,
                stop_policy: lock.generation_stop_policy().clone(),
                lock: Some(lock),
                moe_artifact: None,
                gguf_moe: None,
                tokenizer,
                renderer,
                plan,
                resident,
                mtp_resident,
                mtp_plan,
                session,
                target: config.target,
                model_ready_current_bytes,
                sidecar: None,
                nvfp4_sidecar: None,
                fp8_provider,
                cache: None,
                gguf_source: Some(source),
                vision_manifest,
                vision_resident: None,
                completion_timeout: config.completion_timeout,
                kv_cache_encoding: config.kv_cache_encoding,
                phase41: config.phase41,
                prefix_cache,
                checkpoint,
                persistent_checkpoint_descriptor_digest: Some(
                    qwen_checkpoint_kv_descriptor_digest(&checkpoint_graph),
                ),
                persistent_capture_requested: false,
                persistent_capture: None,
                adapter_catalog,
            })),
            audits: Mutex::new(Vec::new()),
            shutdown: Mutex::new(ShutdownStateV1::Active),
            shutdown_timeout: config.shutdown_timeout,
            identity,
        })
    }

    fn open_gguf_moe(
        config: QwenBackendConfigV1,
        derived: sllm_core::DerivedGgufLock,
    ) -> Result<Self, BackendErrorV1> {
        if config.adapter_catalog.is_some() {
            return Err(BackendErrorV1::new(
                "Qwen adapter catalog is supported only by dense BF16 Qwen",
            ));
        }
        validate_qwen_phase41_operational_config(&config.phase41)?;
        let verified = verify_derived_gguf(derived, &config.gguf_path)
            .map_err(|error| BackendErrorV1::new(format!("GGUF verification failed: {error}")))?;
        let source = Arc::new(verify_gguf_qwen35_moe(verified).map_err(|error| {
            BackendErrorV1::new(format!("Qwen3.5 MoE GGUF validation failed: {error}"))
        })?);
        let tokenizer = TokenizerFrontendV1::from_qwen35_moe_gguf(&source).map_err(|error| {
            BackendErrorV1::new(format!("MoE tokenizer construction failed: {error}"))
        })?;
        let reasoning_close_token_ids = validate_qwen_reasoning_close_marker(&tokenizer)?;
        let renderer = Qwen35ChatTemplateV1::from_qwen35_moe_gguf(&source)
            .map_err(|error| BackendErrorV1::new(format!("MoE chat renderer failed: {error}")))?;
        let plan = build_gguf_qwen35_moe_weight_load_plan(&source).map_err(|error| {
            BackendErrorV1::new(format!("MoE GGUF load plan validation failed: {error}"))
        })?;
        let seed_graph =
            build_qwen35_gguf_moe_execution_graph(&source, &plan, 1, 1).map_err(|error| {
                BackendErrorV1::new(format!("MoE GGUF resident graph failed: {error}"))
            })?;
        let backend = HipBackend::connect()
            .map_err(|error| BackendErrorV1::new(format!("HIP backend is unavailable: {error}")))?;
        let session = backend
            .open_execution_session(
                ExecutionSessionRequest::new(config.device_index, &config.target).map_err(
                    |error| BackendErrorV1::new(format!("HIP session request failed: {error}")),
                )?,
            )
            .map_err(|error| BackendErrorV1::new(format!("HIP session failed: {error}")))?;
        let resident = QwenResidentModel::new_gguf_moe(
            Arc::clone(&session),
            seed_graph,
            plan.clone(),
            Arc::clone(&source),
            config.completion_timeout,
        )
        .map_err(|error| BackendErrorV1::new(format!("MoE GGUF resident load failed: {error}")))?;
        let ready = session.memory_snapshot();
        require_clean_request_memory(ready, "MoE GGUF model-ready")?;
        let model_ready_current_bytes = ready.model_resident().current_bytes();
        if model_ready_current_bytes == 0 || ready.current_bytes() != model_ready_current_bytes {
            return Err(BackendErrorV1::new(
                "MoE GGUF model-ready accounting is not resident-only",
            ));
        }
        let identity = BackendIdentityV1 {
            target: config.target.clone(),
            model_fingerprint: sllm_core::QWEN35_MOE_MODEL_FINGERPRINT.to_owned(),
            plan_digest: plan.digest_hex(),
            model_ready_current_bytes,
            context_length: config.context_length,
            recommended_context_tokens: QWEN35_RECOMMENDED_CONTEXT_TOKENS as u32,
        };
        let prefix_cache = QwenPrefixCacheRuntimeV1::new(&config.phase41.prefix_cache)?;
        Ok(Self {
            state: Mutex::new(Some(QwenBackendStateV1 {
                reasoning_close_token_ids,
                lock: None,
                moe_artifact: None,
                gguf_moe: Some(source),
                stop_policy: qwen35_moe_generation_stop_policy(),
                tokenizer,
                renderer,
                plan,
                resident,
                mtp_resident: None,
                mtp_plan: None,
                session,
                target: config.target,
                model_ready_current_bytes,
                sidecar: None,
                nvfp4_sidecar: None,
                fp8_provider: Some("ocp-mxfp4-w4a4-mixed".to_owned()),
                cache: None,
                gguf_source: None,
                vision_manifest: None,
                vision_resident: None,
                completion_timeout: config.completion_timeout,
                kv_cache_encoding: KvCacheEncoding::Fp16,
                phase41: config.phase41,
                prefix_cache,
                checkpoint: None,
                persistent_checkpoint_descriptor_digest: None,
                persistent_capture_requested: false,
                persistent_capture: None,
                adapter_catalog: None,
            })),
            audits: Mutex::new(Vec::new()),
            shutdown: Mutex::new(ShutdownStateV1::Active),
            shutdown_timeout: config.shutdown_timeout,
            identity,
        })
    }

    fn validate_persistent_chat_state(state: &QwenBackendStateV1) -> Result<(), BackendErrorV1> {
        if state.lock.is_none()
            || state.moe_artifact.is_some()
            || state.gguf_moe.is_some()
            || state.mtp_resident.is_some()
            || state.mtp_plan.is_some()
            || state.sidecar.is_some()
            || state.nvfp4_sidecar.is_some()
            || state.fp8_provider.is_some()
            || state
                .gguf_source
                .as_ref()
                .is_none_or(|source| source.has_fp8_recipe())
            || state.kv_cache_encoding != KvCacheEncoding::Fp16
            || !matches!(
                state.phase41.prefix_cache,
                PrefixCacheStartupConfigV1::Disabled
            )
            || !matches!(
                state.phase41.context_window,
                ContextWindowStartupConfigV1::Disabled
            )
            || !matches!(
                state.phase41.checkpoint,
                CheckpointStartupConfigV1::Disabled
            )
            || !matches!(state.phase41.draft, DraftStartupConfigV1::Disabled)
        {
            return Err(BackendErrorV1::new(
                "persistent chat requires a dense BF16 Qwen text runtime with Phase 41 disabled",
            ));
        }
        Ok(())
    }

    fn validate_persistent_checkpoint_identity(
        state: &QwenBackendStateV1,
        checkpoint: &SessionCheckpoint,
    ) -> Result<(), BackendErrorV1> {
        let descriptor_digest = state
            .persistent_checkpoint_descriptor_digest
            .ok_or_else(|| {
                BackendErrorV1::new(
                    "persistent chat checkpoint identity requires the dense BF16 graph",
                )
            })?;
        let expected = qwen_checkpoint_identity_with_descriptor_digest(
            state
                .lock
                .as_ref()
                .expect("validated dense Qwen lock")
                .fingerprint(),
            &state.plan,
            &state.tokenizer,
            &state.renderer,
            &state.target,
            state.fp8_provider.as_deref(),
            "adapter:none-v1",
            state.kv_cache_encoding,
            descriptor_digest,
            &checkpoint.payload.token_history,
        )?;
        if checkpoint.header.identity != expected {
            return Err(BackendErrorV1::new(
                "Qwen checkpoint identity differs from the running model",
            ));
        }
        Ok(())
    }

    fn install_persistent_checkpoint(
        &self,
        store: Arc<CheckpointStore>,
        loaded: Option<&SessionCheckpoint>,
    ) -> Result<(), BackendErrorV1> {
        let mut state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Qwen backend state is poisoned"))?;
        let state = state_guard
            .as_mut()
            .ok_or_else(|| BackendErrorV1::new("Qwen backend is shut down"))?;
        Self::validate_persistent_chat_state(state)?;
        if let Some(checkpoint) = loaded {
            Self::validate_persistent_checkpoint_identity(state, checkpoint)?;
        }
        state.checkpoint = Some(QwenCheckpointRuntimeV1 {
            store,
            loaded: loaded.map(|checkpoint| Arc::new(checkpoint.clone())),
            save_name: None,
        });
        state.persistent_capture_requested = false;
        state.persistent_capture = None;
        Ok(())
    }

    fn arm_persistent_checkpoint_capture(&self) -> Result<(), BackendErrorV1> {
        let mut state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Qwen backend state is poisoned"))?;
        let state = state_guard
            .as_mut()
            .ok_or_else(|| BackendErrorV1::new("Qwen backend is shut down"))?;
        Self::validate_persistent_chat_state(state)?;
        if state.checkpoint.is_none() {
            return Err(BackendErrorV1::new(
                "persistent chat checkpoint store is not installed",
            ));
        }
        state.persistent_capture_requested = true;
        state.persistent_capture = None;
        Ok(())
    }

    fn take_persistent_checkpoint_capture(
        &self,
    ) -> Result<Option<QwenCapturedChatCheckpointV1>, BackendErrorV1> {
        let mut state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Qwen backend state is poisoned"))?;
        let state = state_guard
            .as_mut()
            .ok_or_else(|| BackendErrorV1::new("Qwen backend is shut down"))?;
        state.persistent_capture_requested = false;
        Ok(state.persistent_capture.take())
    }

    pub fn request_audits(&self) -> Vec<ProductionRequestAuditV1> {
        self.audits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn model_fingerprint(&self) -> &str {
        &self.identity.model_fingerprint
    }

    /// Returns the exact verified runtime weight-plan digest used by state
    /// identities.  This is intentionally distinct from the derived-lock
    /// artifact fingerprint.
    pub fn plan_digest(&self) -> &str {
        &self.identity.plan_digest
    }

    /// Returns the verified, path-independent identity of the offline adapter
    /// catalog loaded with this backend. Empty catalogs use the stable disabled
    /// identity and never expose filesystem paths or payload bytes.
    pub fn adapter_catalog_identity(&self) -> Result<String, BackendErrorV1> {
        let state = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Qwen backend state is poisoned"))?;
        Ok(state
            .as_ref()
            .and_then(|state| state.adapter_catalog.as_ref())
            .map_or_else(
                || "adapter:none-v1".to_owned(),
                QwenAdapterCatalogV1::identity,
            ))
    }

    pub const fn context_length(&self) -> u32 {
        self.identity.context_length
    }

    pub const fn recommended_context_tokens(&self) -> u32 {
        self.identity.recommended_context_tokens
    }

    pub fn target(&self) -> &str {
        &self.identity.target
    }

    /// Returns nonblocking redacted memory accounting for operational metrics.
    /// A busy or already-shut-down backend reports the safe all-zero default.
    pub fn observability_snapshot(&self) -> BackendObservabilitySnapshotV1 {
        self.state
            .try_lock()
            .ok()
            .and_then(|state| state.as_ref().map(|state| state.session.memory_snapshot()))
            .map(observability_snapshot_from_allocation)
            .unwrap_or_default()
    }

    pub fn shutdown(&self) -> Result<ProductionShutdownAuditV1, BackendErrorV1> {
        let pending = {
            let status = self
                .shutdown
                .lock()
                .map_err(|_| BackendErrorV1::new("Qwen shutdown state is poisoned"))?;
            match &*status {
                ShutdownStateV1::Complete(audit) => return Ok(audit.clone()),
                ShutdownStateV1::Pending {
                    session,
                    model_ready_current_bytes,
                } => Some((Arc::clone(session), *model_ready_current_bytes)),
                ShutdownStateV1::Active => None,
            }
        };
        let (session, model_ready_current_bytes) = if let Some(pending) = pending {
            pending
        } else {
            let state = self
                .state
                .lock()
                .map_err(|_| BackendErrorV1::new("Qwen backend state is poisoned"))?
                .take()
                .ok_or_else(|| BackendErrorV1::new("Qwen backend is already shut down"))?;
            let QwenBackendStateV1 {
                resident,
                mtp_resident,
                vision_resident,
                prefix_cache,
                session,
                model_ready_current_bytes,
                ..
            } = state;
            drop(prefix_cache);
            drop(resident);
            drop(mtp_resident);
            drop(vision_resident);
            (session, model_ready_current_bytes)
        };
        let mark_pending = |session: &Arc<ExecutionSession>| {
            if let Ok(mut status) = self.shutdown.lock() {
                *status = ShutdownStateV1::Pending {
                    session: Arc::clone(session),
                    model_ready_current_bytes,
                };
            }
        };
        let before_shutdown = session.memory_snapshot();
        if before_shutdown.current_bytes() != 0 {
            mark_pending(&session);
            return Err(BackendErrorV1::new(format!(
                "resident drop left {} tracked device bytes",
                before_shutdown.current_bytes()
            )));
        }
        let report = match session.shutdown(self.shutdown_timeout) {
            Ok(report) => report,
            Err(error) => {
                mark_pending(&session);
                return Err(BackendErrorV1::new(format!(
                    "HIP session shutdown failed: {error}"
                )));
            }
        };
        let final_memory = session.memory_snapshot();
        if final_memory.current_bytes() != 0
            || report.retryable_cleanup != 0
            || report.durable_quarantine != 0
        {
            mark_pending(&session);
            return Err(BackendErrorV1::new(
                "HIP session shutdown did not reach a zero-cleanup terminal state",
            ));
        }
        let audit = ProductionShutdownAuditV1 {
            schema_version: "openai-chat-production-shutdown-v1".to_owned(),
            target: self.identity.target.clone(),
            model_fingerprint: self.identity.model_fingerprint.clone(),
            plan_digest: self.identity.plan_digest.clone(),
            model_ready_current_bytes,
            final_current_bytes: final_memory.current_bytes(),
            final_request_state_bytes: final_memory.request_state().current_bytes(),
            final_workspace_bytes: final_memory.workspace().current_bytes(),
            retryable_cleanup: report.retryable_cleanup,
            durable_quarantine: report.durable_quarantine,
            requests: self.request_audits(),
        };
        if let Ok(mut status) = self.shutdown.lock() {
            *status = ShutdownStateV1::Complete(audit.clone());
        }
        Ok(audit)
    }

    fn record_audit(&self, audit: ProductionRequestAuditV1) {
        let mut audits = self
            .audits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if audits.len() == MAX_RETAINED_REQUEST_AUDITS {
            audits.remove(0);
        }
        audits.push(audit);
    }
}

impl Gemma4ChatBackendV1 {
    pub fn open(config: Gemma4BackendConfigV1) -> Result<Self, BackendErrorV1> {
        config.validate()?;
        validate_gemma_phase41_operational_config(&config.phase41)?;
        let derived = read_derived_gguf_lock(&config.derived_lock_path).map_err(|error| {
            BackendErrorV1::new(format!("derived GGUF lock validation failed: {error}"))
        })?;
        let lock = match builtin_reviewed_model_lock(&derived.source_lock_fingerprints).map_err(
            |error| BackendErrorV1::new(format!("built-in model lock resolution failed: {error}")),
        )? {
            ReviewedModelLock::Gemma4(lock) => lock,
            ReviewedModelLock::Qwen35(_) => {
                return Err(BackendErrorV1::new(
                    "Gemma backend requires a derived GGUF for a reviewed Gemma 4 model",
                ));
            }
        };
        let verified = verify_derived_gguf(derived, &config.gguf_path)
            .map_err(|error| BackendErrorV1::new(format!("GGUF verification failed: {error}")))?;
        let (source, plan) =
            build_verified_gguf_gemma_weight_load_plan(&lock, verified).map_err(|error| {
                BackendErrorV1::new(format!("GGUF Gemma load plan failed: {error}"))
            })?;
        let source = Arc::new(source);
        let tokenizer =
            TokenizerFrontendV1::from_gemma4_gguf(&lock, source.gguf()).map_err(|error| {
                BackendErrorV1::new(format!(
                    "verified Gemma tokenizer construction failed: {error}"
                ))
            })?;
        let checkpoint_graph =
            sllm_core::build_gemma4_graph(&lock, &plan, 1, 0, u64::from(config.context_length))
                .map_err(|error| {
                    BackendErrorV1::new(format!("Gemma checkpoint graph failed: {error}"))
                })?;
        let checkpoint_descriptor_digest =
            gemma_checkpoint_kv_descriptor_digest(&checkpoint_graph, source.as_ref())?;
        let checkpoint = build_gemma_checkpoint_runtime(
            &config.phase41.checkpoint,
            &checkpoint_graph,
            &plan,
            &tokenizer,
            &config.target,
            checkpoint_descriptor_digest,
        )?;
        let stop_policy = gemma4_generation_stop_policy(&lock).map_err(|error| {
            BackendErrorV1::new(format!("Gemma stop policy construction failed: {error}"))
        })?;
        let backend = HipBackend::connect()
            .map_err(|error| BackendErrorV1::new(format!("HIP backend is unavailable: {error}")))?;
        let session_request = ExecutionSessionRequest::new(config.device_index, &config.target)
            .map_err(|error| BackendErrorV1::new(format!("HIP session request failed: {error}")))?;
        let session = backend
            .open_execution_session(session_request)
            .map_err(|error| {
                BackendErrorV1::new(format!("exact HIP execution session failed: {error}"))
            })?;
        let resident = Gemma4ResidentModel::new_gguf_quantized(
            Arc::clone(&session),
            lock.clone(),
            plan.clone(),
            source,
            config.completion_timeout,
        )
        .map_err(|error| BackendErrorV1::new(format!("resident model load failed: {error}")))?;
        let ready = session.memory_snapshot();
        require_clean_request_memory(ready, "model-ready")?;
        let model_ready_current_bytes = ready.model_resident().current_bytes();
        if model_ready_current_bytes == 0 || ready.current_bytes() != model_ready_current_bytes {
            return Err(BackendErrorV1::new(
                "Gemma model-ready allocation accounting is not resident-only",
            ));
        }
        let identity = BackendIdentityV1 {
            target: config.target.clone(),
            model_fingerprint: lock.fingerprint().to_owned(),
            plan_digest: plan.digest_hex(),
            model_ready_current_bytes,
            context_length: config.context_length,
            recommended_context_tokens: GEMMA4_RECOMMENDED_CONTEXT_TOKENS as u32,
        };
        let prefix_cache = GemmaPrefixCacheRuntimeV1::new(&config.phase41.prefix_cache)?;
        Ok(Self {
            state: Mutex::new(Some(Gemma4BackendStateV1 {
                _lock: lock,
                tokenizer,
                stop_policy,
                plan,
                resident,
                session,
                target: config.target,
                model_ready_current_bytes,
                weight_encoding: "mixed-nvfp4-w4a4-fp8-w8a8".to_owned(),
                kv_bytes_per_token: GEMMA4_STATIC_FP8_KV_BYTES_PER_TOKEN,
                phase41: config.phase41,
                prefix_cache,
                checkpoint,
                checkpoint_descriptor_digest: Some(checkpoint_descriptor_digest),
            })),
            audits: Mutex::new(Vec::new()),
            shutdown: Mutex::new(ShutdownStateV1::Active),
            shutdown_timeout: config.shutdown_timeout,
            identity,
        })
    }

    pub fn request_audits(&self) -> Vec<ProductionRequestAuditV1> {
        self.audits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn model_fingerprint(&self) -> &str {
        &self.identity.model_fingerprint
    }

    /// Returns the exact verified runtime weight-plan digest used by state
    /// identities.  This is intentionally distinct from the derived-lock
    /// artifact fingerprint.
    pub fn plan_digest(&self) -> &str {
        &self.identity.plan_digest
    }

    pub const fn context_length(&self) -> u32 {
        self.identity.context_length
    }

    pub const fn recommended_context_tokens(&self) -> u32 {
        self.identity.recommended_context_tokens
    }

    pub fn target(&self) -> &str {
        &self.identity.target
    }

    /// Returns nonblocking redacted memory accounting for operational metrics.
    /// A busy or already-shut-down backend reports the safe all-zero default.
    pub fn observability_snapshot(&self) -> BackendObservabilitySnapshotV1 {
        self.state
            .try_lock()
            .ok()
            .and_then(|state| state.as_ref().map(|state| state.session.memory_snapshot()))
            .map(observability_snapshot_from_allocation)
            .unwrap_or_default()
    }

    pub fn shutdown(&self) -> Result<ProductionShutdownAuditV1, BackendErrorV1> {
        let pending = {
            let status = self
                .shutdown
                .lock()
                .map_err(|_| BackendErrorV1::new("Gemma shutdown state is poisoned"))?;
            match &*status {
                ShutdownStateV1::Complete(audit) => return Ok(audit.clone()),
                ShutdownStateV1::Pending {
                    session,
                    model_ready_current_bytes,
                } => Some((Arc::clone(session), *model_ready_current_bytes)),
                ShutdownStateV1::Active => None,
            }
        };
        let (session, model_ready_current_bytes) = if let Some(pending) = pending {
            pending
        } else {
            let state = self
                .state
                .lock()
                .map_err(|_| BackendErrorV1::new("Gemma backend state is poisoned"))?
                .take()
                .ok_or_else(|| BackendErrorV1::new("Gemma backend is already shut down"))?;
            let Gemma4BackendStateV1 {
                resident,
                prefix_cache,
                session,
                model_ready_current_bytes,
                ..
            } = state;
            drop(prefix_cache);
            drop(resident);
            (session, model_ready_current_bytes)
        };
        let mark_pending = |session: &Arc<ExecutionSession>| {
            if let Ok(mut status) = self.shutdown.lock() {
                *status = ShutdownStateV1::Pending {
                    session: Arc::clone(session),
                    model_ready_current_bytes,
                };
            }
        };
        let before_shutdown = session.memory_snapshot();
        if before_shutdown.current_bytes() != 0 {
            mark_pending(&session);
            return Err(BackendErrorV1::new(format!(
                "Gemma resident drop left {} tracked device bytes",
                before_shutdown.current_bytes()
            )));
        }
        let report = match session.shutdown(self.shutdown_timeout) {
            Ok(report) => report,
            Err(error) => {
                mark_pending(&session);
                return Err(BackendErrorV1::new(format!(
                    "HIP session shutdown failed: {error}"
                )));
            }
        };
        let final_memory = session.memory_snapshot();
        if final_memory.current_bytes() != 0
            || report.retryable_cleanup != 0
            || report.durable_quarantine != 0
        {
            mark_pending(&session);
            return Err(BackendErrorV1::new(
                "HIP session shutdown did not reach a zero-cleanup terminal state",
            ));
        }
        let audit = ProductionShutdownAuditV1 {
            schema_version: "openai-chat-production-shutdown-v1".to_owned(),
            target: self.identity.target.clone(),
            model_fingerprint: self.identity.model_fingerprint.clone(),
            plan_digest: self.identity.plan_digest.clone(),
            model_ready_current_bytes: self.identity.model_ready_current_bytes,
            final_current_bytes: final_memory.current_bytes(),
            final_request_state_bytes: final_memory.request_state().current_bytes(),
            final_workspace_bytes: final_memory.workspace().current_bytes(),
            retryable_cleanup: report.retryable_cleanup,
            durable_quarantine: report.durable_quarantine,
            requests: self.request_audits(),
        };
        if let Ok(mut status) = self.shutdown.lock() {
            *status = ShutdownStateV1::Complete(audit.clone());
        }
        Ok(audit)
    }

    fn record_audit(&self, audit: ProductionRequestAuditV1) {
        let mut audits = self
            .audits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if audits.len() == MAX_RETAINED_REQUEST_AUDITS {
            audits.remove(0);
        }
        audits.push(audit);
    }
}

impl ChatGenerationBackendV1 for QwenChatBackendV1 {
    fn observability_snapshot(&self) -> BackendObservabilitySnapshotV1 {
        QwenChatBackendV1::observability_snapshot(self)
    }

    fn embedding_dimension(&self) -> Option<u32> {
        Some(QWEN35_HIDDEN_SIZE as u32)
    }

    fn reviewed_chat_template_available(&self) -> bool {
        true
    }

    fn tool_protocol_v1_available(&self) -> bool {
        true
    }

    fn validate_embedding_input(
        &self,
        input: &BackendEmbeddingInputV1,
    ) -> Result<u64, BackendErrorV1> {
        let state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Qwen backend state is poisoned"))?;
        let state = state_guard
            .as_ref()
            .ok_or_else(|| BackendErrorV1::new("Qwen backend is shut down"))?;
        let tokens = match input {
            BackendEmbeddingInputV1::Text(text) => state
                .tokenizer
                .encode(text)
                .map_err(|error| BackendErrorV1::new(error.to_string()))?
                .len(),
            BackendEmbeddingInputV1::TokenIds(tokens) => {
                validate_generation_token_ids(&state.tokenizer, tokens, "input")?.len()
            }
        };
        let tokens = u64::try_from(tokens)
            .map_err(|_| BackendErrorV1::new("embedding token count overflowed u64"))?;
        if tokens == 0 || tokens > u64::from(self.identity.context_length) {
            return Err(BackendErrorV1::new(format!(
                "embedding input token count {tokens} must be in [1,{}]",
                self.identity.context_length
            )));
        }
        Ok(tokens)
    }

    fn tokenize_utility(
        &self,
        text: &str,
        options: TokenizeOptionsV1,
    ) -> Result<TokenizeResultV1, BackendErrorV1> {
        let state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Qwen backend state is poisoned"))?;
        let state = state_guard
            .as_ref()
            .ok_or_else(|| BackendErrorV1::new("Qwen backend is shut down"))?;
        TokenizerUtilityServiceV1::new(&state.tokenizer, Some(&state.renderer))
            .tokenize(text, options)
            .map_err(|error| BackendErrorV1::new(error.to_string()))
    }

    fn detokenize_utility(
        &self,
        token_ids: &[u32],
        mode: DecodeModeV1,
    ) -> Result<String, BackendErrorV1> {
        let state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Qwen backend state is poisoned"))?;
        let state = state_guard
            .as_ref()
            .ok_or_else(|| BackendErrorV1::new("Qwen backend is shut down"))?;
        TokenizerUtilityServiceV1::new(&state.tokenizer, Some(&state.renderer))
            .detokenize_ids(token_ids, mode)
            .map_err(|error| BackendErrorV1::new(error.to_string()))
    }

    fn apply_template_utility(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<ApplyTemplateResultV1, BackendErrorV1> {
        let state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Qwen backend state is poisoned"))?;
        let state = state_guard
            .as_ref()
            .ok_or_else(|| BackendErrorV1::new("Qwen backend is shut down"))?;
        TokenizerUtilityServiceV1::new(&state.tokenizer, Some(&state.renderer))
            .apply_template(messages, options)
            .map_err(|error| BackendErrorV1::new(error.to_string()))
    }

    fn tokenize_infill_content(&self, text: &str) -> Result<Vec<u32>, BackendErrorV1> {
        let state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Qwen backend state is poisoned"))?;
        let state = state_guard
            .as_ref()
            .ok_or_else(|| BackendErrorV1::new("Qwen backend is shut down"))?;
        state
            .tokenizer
            .encode_without_special_tokens(text)
            .map(|tokens| tokens.as_slice().to_vec())
            .map_err(|error| BackendErrorV1::new(error.to_string()))
    }

    fn embed(
        &self,
        request: &BackendEmbeddingRequestV1,
        cancellation: &GenerationCancellationV1,
    ) -> Result<BackendEmbeddingBatchV1, BackendErrorV1> {
        let mut state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Qwen backend state is poisoned"))?;
        let state = state_guard
            .as_mut()
            .ok_or_else(|| BackendErrorV1::new("Qwen backend is shut down"))?;
        let ready = state.session.memory_snapshot();
        require_request_memory_baseline(
            ready,
            state.prefix_cache.baseline_bytes()?,
            "Qwen embedding admission",
        )?;
        if ready.model_resident().current_bytes() != state.model_ready_current_bytes {
            return Err(BackendErrorV1::new(
                "model-resident accounting changed before embedding admission",
            ));
        }
        let mut vectors = Vec::with_capacity(request.inputs().len());
        for input in request.inputs() {
            if cancellation.is_cancelled() {
                return Err(BackendErrorV1::new("embedding cancelled"));
            }
            let token_ids = match input {
                BackendEmbeddingInputV1::Text(text) => state
                    .tokenizer
                    .encode(text)
                    .map_err(|error| {
                        BackendErrorV1::new(format!("embedding tokenization failed: {error}"))
                    })?
                    .as_slice()
                    .to_vec(),
                BackendEmbeddingInputV1::TokenIds(token_ids) => {
                    validate_generation_token_ids(&state.tokenizer, token_ids, "input")?
                }
            };
            if u64::try_from(token_ids.len()).unwrap_or(u64::MAX)
                > u64::from(self.identity.context_length)
            {
                return Err(BackendErrorV1::new(format!(
                    "embedding input exceeds the configured context length {}",
                    self.identity.context_length
                )));
            }
            let vector = qwen_embed_one(state, &token_ids, cancellation);
            let cleanup = state.session.memory_snapshot();
            require_request_memory_baseline(
                cleanup,
                state.prefix_cache.baseline_bytes()?,
                "Qwen embedding cleanup",
            )
            .and_then(|()| {
                if cleanup.model_resident().current_bytes() == state.model_ready_current_bytes {
                    Ok(())
                } else {
                    Err(BackendErrorV1::new(
                        "model-resident accounting changed after Qwen embedding cleanup",
                    ))
                }
            })?;
            vectors.push(vector?);
        }
        BackendEmbeddingBatchV1::new(QWEN35_HIDDEN_SIZE as u32, vectors)
    }

    fn generate(
        &self,
        request: &ChatCompletionRequestV1,
        cancellation: &GenerationCancellationV1,
        sink: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        let started = Instant::now();
        let mut state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Qwen backend state is poisoned"))?;
        let state = state_guard
            .as_mut()
            .ok_or_else(|| BackendErrorV1::new("Qwen backend is shut down"))?;
        let ready = state.session.memory_snapshot();
        require_request_memory_baseline(
            ready,
            state.prefix_cache.baseline_bytes()?,
            "request admission",
        )?;
        if ready.model_resident().current_bytes() != state.model_ready_current_bytes {
            return Err(BackendErrorV1::new(
                "model-resident accounting changed before request admission",
            ));
        }

        let service = GenerationServiceV1::new(
            &state.tokenizer,
            Some(&state.renderer),
            &state.stop_policy,
        )
        .map_err(|error| BackendErrorV1::new(format!("generation service failed: {error}")))?;
        let mut generation = generation_config_for_request(
            request,
            &state.tokenizer,
            Some(&state.reasoning_close_token_ids),
        )?;
        let requires_logits = generation
            .sampler_chain()
            .map_or(generation.sampling().requires_logits(), |chain| {
                chain.requires_logits()
            })
            || generation.grammar().is_some()
            // Reasoning policies intersect the candidate set and may force a
            // marker token, so the device path must expose logits even when
            // the user selected greedy legacy sampling.
            || generation
                .reasoning()
                .is_some_and(ReasoningPolicyV1::is_enabled);
        let requires_randomness = generation.sampler_chain().map_or(
            generation.sampling().requires_logits(),
            SamplerChainConfigV1::requires_randomness,
        );
        let resolved_sampling_seed = requires_randomness
            .then(|| OsSamplingRandom::resolve_seed(request.sampling_seed()))
            .transpose()
            .map_err(|error| BackendErrorV1::new(format!("sampling seed failed: {error}")))?;
        if let Some(seed) = resolved_sampling_seed {
            generation = generation.with_device_selector_seed(seed);
        }
        let prepared_prompt = qwen_generation_prompt(request, &service, &state.tokenizer)?;
        let assistant_prefill_tokens = prepared_prompt.assistant_prefill_token_ids().to_vec();
        let prompt = prepared_prompt.token_ids().to_vec();
        let adapter_request =
            resolve_qwen_adapters(state.adapter_catalog.as_ref(), request.model_variant())?;
        let adapters_enabled = adapter_request.identity() != "adapter:none-v1";
        if adapters_enabled
            && (state.moe_artifact.is_some()
                || state.gguf_moe.is_some()
                || state.sidecar.is_some()
                || state.nvfp4_sidecar.is_some()
                || state
                    .gguf_source
                    .as_ref()
                    .is_some_and(|source| source.has_fp8_recipe()))
        {
            return Err(BackendErrorV1::new(
                "Qwen adapters require the dense BF16 text execution path",
            ));
        }
        let loaded_checkpoint = state
            .checkpoint
            .as_ref()
            .and_then(|runtime| runtime.loaded.clone());
        if loaded_checkpoint.is_some() && !assistant_prefill_tokens.is_empty() {
            return Err(BackendErrorV1::new(
                "assistant prefill cannot be combined with a loaded Qwen checkpoint",
            ));
        }
        let checkpoint_suffix = loaded_checkpoint
            .as_ref()
            .map(|checkpoint| {
                let prefix = &checkpoint.payload.token_history;
                if prompt.len() <= prefix.len() || !prompt.starts_with(prefix) {
                    return Err(BackendErrorV1::new(
                        "Qwen checkpoint request must extend the loaded prompt token prefix",
                    ));
                }
                Ok(prompt[prefix.len()..].to_vec())
            })
            .transpose()?;
        let processed_images = request
            .messages()
            .iter()
            .flat_map(|message| message.parts())
            .filter_map(|part| match part {
                ChatContentPartV1::Image(image) => Some(image),
                ChatContentPartV1::Text(_) => None,
            })
            .collect::<Vec<_>>();
        let multimodal_prompt = if processed_images.is_empty() {
            None
        } else {
            if adapters_enabled {
                return Err(BackendErrorV1::new(
                    "Qwen adapters are currently text-only and reject multimodal requests",
                ));
            }
            if state.moe_artifact.is_some() || state.gguf_moe.is_some() {
                return Err(BackendErrorV1::new(
                    "Qwen3.5 MoE production path is text-only",
                ));
            }
            if state.sidecar.is_some() || state.nvfp4_sidecar.is_some() {
                return Err(BackendErrorV1::new(
                    "vision requests currently require the BF16 text artifact",
                ));
            }
            let vision_manifest = state.vision_manifest.clone().ok_or_else(|| {
                BackendErrorV1::new("vision requests require the fixed Qwen3.5-4B model")
            })?;
            if state.vision_resident.is_none() {
                state.vision_resident = Some(
                    match (&state.cache, &state.gguf_source) {
                        (Some(cache), None) => QwenVisionResidentModel::new(
                            Arc::clone(&state.session),
                            Arc::clone(cache),
                            vision_manifest.clone(),
                            state.completion_timeout,
                        ),
                        (None, Some(source)) => QwenVisionResidentModel::new_gguf(
                            Arc::clone(&state.session),
                            Arc::clone(source),
                            vision_manifest.clone(),
                            state.completion_timeout,
                        ),
                        _ => {
                            return Err(BackendErrorV1::new(
                                "vision requires exactly one verified dense weight source",
                            ));
                        }
                    }
                    .map_err(|error| {
                        BackendErrorV1::new(format!("vision resident load failed: {error}"))
                    })?,
                );
                let ready = state.session.memory_snapshot();
                require_clean_request_memory(ready, "vision model-ready")?;
                state.model_ready_current_bytes = ready.model_resident().current_bytes();
            }
            let vision = state
                .vision_resident
                .as_ref()
                .expect("vision resident was initialized");
            let images = processed_images
                .iter()
                .map(|image| {
                    let output = vision
                        .execute(&QwenVisionExecutionInput {
                            grid_thw: image.grid_thw,
                            patch_width: image.patch_width,
                            patches: image.patches.clone(),
                        })
                        .map_err(|error| {
                            BackendErrorV1::new(format!("vision encode failed: {error}"))
                        })?;
                    Ok(QwenMultimodalImageEmbedding {
                        grid_thw: image.grid_thw,
                        embeddings_bf16: output.embeddings_bf16().to_vec(),
                    })
                })
                .collect::<Result<Vec<_>, BackendErrorV1>>()?;
            let assembled = match (&state.cache, &state.gguf_source) {
                (Some(cache), None) => assemble_qwen35_multimodal_prompt(
                    cache,
                    &prompt,
                    vision_manifest.image_pad_token,
                    &images,
                ),
                (None, Some(source)) => assemble_gguf_qwen35_multimodal_prompt(
                    source,
                    &prompt,
                    vision_manifest.image_pad_token,
                    &images,
                ),
                _ => {
                    return Err(BackendErrorV1::new(
                        "multimodal assembly requires exactly one verified dense weight source",
                    ));
                }
            }
            .map_err(|error| {
                BackendErrorV1::new(format!("multimodal prompt assembly failed: {error}"))
            })?;
            Some(assembled)
        };
        if multimodal_prompt.is_some()
            && !matches!(
                state.phase41.prefix_cache,
                PrefixCacheStartupConfigV1::Disabled
            )
        {
            return Err(BackendErrorV1::new(
                "prefix cache is currently text-only and rejects multimodal requests",
            ));
        }
        if loaded_checkpoint.is_some() && multimodal_prompt.is_some() {
            return Err(BackendErrorV1::new(
                "Qwen prompt checkpoint continuation is text-only",
            ));
        }
        if multimodal_prompt.is_some()
            && matches!(state.phase41.draft, DraftStartupConfigV1::Ngram { .. })
        {
            return Err(BackendErrorV1::new(
                "ngram draft verification is currently text-only",
            ));
        }
        let prompt_tokens = u64::try_from(prompt.len())
            .map_err(|_| BackendErrorV1::new("prompt token count overflowed u64"))?;
        let context_policy = match &state.phase41.context_window {
            ContextWindowStartupConfigV1::Disabled => None,
            ContextWindowStartupConfigV1::KeepPrefixRecentV1 {
                keep_prefix,
                keep_recent,
            } => Some(
                ContextPositionPolicyV1::keep_prefix_recent_v1(*keep_prefix, *keep_recent)
                    .map_err(|error| BackendErrorV1::new(error.to_string()))?,
            ),
        };
        if adapters_enabled && context_policy.is_some() {
            return Err(BackendErrorV1::new(
                "Qwen adapters cannot be combined with context-window shifting",
            ));
        }
        if context_policy.is_some() && generation.device_selector_seed().is_some() {
            return Err(BackendErrorV1::new(
                "context-window shifting cannot be combined with device-selector sampling",
            ));
        }
        if context_policy.is_some()
            && (multimodal_prompt.is_some()
                || state.moe_artifact.is_some()
                || state.gguf_moe.is_some()
                || state
                    .gguf_source
                    .as_ref()
                    .is_some_and(|source| source.has_fp8_recipe())
                || state.sidecar.is_some()
                || state.nvfp4_sidecar.is_some()
                || state.kv_cache_encoding != KvCacheEncoding::Fp16
                || !matches!(
                    state.phase41.prefix_cache,
                    PrefixCacheStartupConfigV1::Disabled
                )
                || !matches!(state.phase41.draft, DraftStartupConfigV1::Disabled))
        {
            return Err(BackendErrorV1::new(
                "context-window shifting currently supports only dense BF16 text without prefix cache, MTP, ngram, multimodal, or quantized variants",
            ));
        }
        let state_capacity = if state.checkpoint.is_some()
            || context_policy.is_some()
            || !matches!(
                state.phase41.prefix_cache,
                PrefixCacheStartupConfigV1::Disabled
            ) {
            u64::from(self.identity.context_length)
        } else {
            prompt_tokens
                .checked_add(u64::from(generation.max_new_tokens()))
                .ok_or_else(|| BackendErrorV1::new("request state capacity overflowed u64"))?
        };
        let initial_context_decision = context_policy
            .map(|policy| {
                policy
                    .plan_initial(prompt_tokens, state_capacity)
                    .map_err(|error| BackendErrorV1::new(error.to_string()))
            })
            .transpose()?;
        if requires_logits
            && initial_context_decision.is_some_and(|decision| decision.requires_shift())
        {
            return Err(BackendErrorV1::new(
                "context-window shifting with an oversized prompt does not support an initial logits readback",
            ));
        }
        if state_capacity > u64::from(self.identity.context_length) {
            return Err(BackendErrorV1::new(format!(
                "request requires {state_capacity} context tokens but the server was started with --context-length {}",
                self.identity.context_length
            )));
        }
        if multimodal_prompt.is_some() && state.kv_cache_encoding != KvCacheEncoding::Fp16 {
            return Err(BackendErrorV1::new(
                "multimodal Qwen requests currently require FP16 KV cache",
            ));
        }
        let placement_total_memory_bytes = state
            .session
            .total_memory_bytes()
            .map_err(|error| BackendErrorV1::new(error.to_string()))?
            .ok_or_else(|| BackendErrorV1::new("backend omitted total device memory"))?;
        let placement_available_memory_bytes = state
            .session
            .available_memory_bytes()
            .map_err(|error| BackendErrorV1::new(error.to_string()))?
            .ok_or_else(|| BackendErrorV1::new("backend omitted available device memory"))?;
        let graph_prompt_tokens = initial_context_decision
            .map(|decision| decision.proposed_state().logical_length())
            .unwrap_or(prompt_tokens);
        let chunk_candidates =
            if initial_context_decision.is_some_and(|decision| decision.requires_shift()) {
                vec![graph_prompt_tokens]
            } else if multimodal_prompt.is_some() {
                vec![prompt_tokens]
            } else {
                qwen_prefill_chunk_candidates(placement_total_memory_bytes, graph_prompt_tokens)
                    .map_err(|error| BackendErrorV1::new(error.to_string()))?
            };
        let mtp_target = matches!(state.phase41.draft, DraftStartupConfigV1::MtpAuto)
            && state.mtp_resident.is_some()
            && !requires_logits
            && multimodal_prompt.is_none();
        if adapters_enabled && mtp_target {
            return Err(BackendErrorV1::new(
                "Qwen adapters cannot be combined with MTP draft execution",
            ));
        }
        let ngram_draft_width = match &state.phase41.draft {
            DraftStartupConfigV1::Ngram { width, .. } if !requires_logits => usize::from(*width),
            _ => 0,
        };
        if requires_logits && matches!(state.phase41.draft, DraftStartupConfigV1::Ngram { .. }) {
            return Err(BackendErrorV1::new(
                "ngram speculative verification currently requires exact greedy generation",
            ));
        }
        let build_graph = |chunk_rows: u64| {
            let target_rows = if mtp_target {
                chunk_rows.max(2)
            } else if ngram_draft_width != 0 {
                chunk_rows.max(ngram_draft_width as u64 + 1)
            } else {
                chunk_rows
            };
            if let Some(artifact) = &state.moe_artifact {
                build_qwen35_moe_execution_graph(artifact, &state.plan, target_rows, state_capacity)
            } else if let Some(source) = &state.gguf_moe {
                build_qwen35_gguf_moe_execution_graph(
                    source,
                    &state.plan,
                    target_rows,
                    state_capacity,
                )
            } else if multimodal_prompt.is_some() {
                build_qwen35_multimodal_graph(
                    state.lock.as_ref().expect("dense Qwen lock"),
                    &state.plan,
                    prompt_tokens,
                    state_capacity,
                )
            } else if let Some(nvfp4_sidecar) = &state.nvfp4_sidecar {
                build_qwen35_nvfp4_graph(
                    state.lock.as_ref().expect("dense Qwen lock"),
                    &state.plan,
                    nvfp4_sidecar,
                    target_rows,
                    state_capacity,
                )
            } else if let Some(source) = state
                .gguf_source
                .as_ref()
                .filter(|source| source.has_fp8_recipe())
            {
                build_qwen35_gguf_fp8_graph(
                    state.lock.as_ref().expect("dense Qwen lock"),
                    &state.plan,
                    source,
                    target_rows,
                    state_capacity,
                    gguf_fp8_dtype(
                        state
                            .fp8_provider
                            .as_deref()
                            .expect("GGUF FP8 state has a selected provider"),
                    ),
                    state.kv_cache_encoding,
                )
            } else {
                match (&state.sidecar, state.fp8_provider.as_deref()) {
                    (Some(_), Some("converted-bf16")) | (None, None) => {
                        build_qwen35_graph_with_kv_cache_encoding(
                            state.lock.as_ref().expect("dense Qwen lock"),
                            &state.plan,
                            target_rows,
                            state_capacity,
                            state.kv_cache_encoding,
                        )
                    }
                    (Some(sidecar), Some("native-fnuz")) => build_qwen35_fp8_fnuz_graph(
                        state.lock.as_ref().expect("dense Qwen lock"),
                        &state.plan,
                        sidecar,
                        target_rows,
                        state_capacity,
                    ),
                    (Some(sidecar), Some(_)) => build_qwen35_fp8_graph(
                        state.lock.as_ref().expect("dense Qwen lock"),
                        &state.plan,
                        sidecar,
                        target_rows,
                        state_capacity,
                    ),
                    _ => unreachable!("validated FP8 server state has a selected provider"),
                }
            }
        };
        let mut rejected = Vec::new();
        let mut selected = None;
        for chunk_rows in chunk_candidates {
            let graph = build_graph(chunk_rows)
                .map_err(|error| BackendErrorV1::new(format!("request graph failed: {error}")))?;
            let estimate =
                qwen_graph_memory_estimate(&graph, &state.plan, placement_total_memory_bytes)
                    .map_err(|error| BackendErrorV1::new(error.to_string()))?;
            let incremental_required = estimate
                .required_bytes()
                .checked_sub(estimate.model_resident_bytes())
                .ok_or_else(|| {
                    BackendErrorV1::new("request placement underflowed graph model-resident bytes")
                })?;
            if incremental_required <= placement_available_memory_bytes {
                selected = Some((graph, estimate, incremental_required));
                break;
            }
            rejected.push(format!("{chunk_rows}:{incremental_required}"));
        }
        let (graph, placement, placement_incremental_required_bytes) = selected.ok_or_else(|| {
            BackendErrorV1::new(format!(
                "no prefill chunk fits available device memory {}; candidates chunk:incremental-required [{}]",
                placement_available_memory_bytes,
                rejected.join(",")
            ))
        })?;
        let prefill_chunk_capacity_tokens = graph.token_count();
        let checkpoint_identity = loaded_checkpoint
            .as_ref()
            .map(|checkpoint| {
                let expected = qwen_checkpoint_identity(
                    &graph,
                    &state.plan,
                    &state.tokenizer,
                    &state.renderer,
                    &state.target,
                    state.fp8_provider.as_deref(),
                    adapter_request.identity(),
                    state.kv_cache_encoding,
                    &checkpoint.payload.token_history,
                )?;
                if checkpoint.header.identity != expected {
                    return Err(BackendErrorV1::new(
                        "Qwen checkpoint identity differs from the request graph",
                    ));
                }
                Ok(expected)
            })
            .transpose()?;
        let prefix_cache_enabled = !matches!(
            state.phase41.prefix_cache,
            PrefixCacheStartupConfigV1::Disabled
        );
        let prefix_cache_eligible = prefix_cache_enabled
            && assistant_prefill_tokens.is_empty()
            && !requires_logits
            && multimodal_prompt.is_none()
            && match state.phase41.prefix_cache {
                PrefixCacheStartupConfigV1::Disabled => false,
                PrefixCacheStartupConfigV1::Enabled {
                    max_logical_tokens, ..
                } => prompt_tokens <= max_logical_tokens,
            };
        let prefix_identity = prefix_cache_eligible
            .then(|| {
                qwen_prefix_identity(state, &graph, state_capacity, adapter_request.identity())
            })
            .transpose()?;
        let prefix_hit = prefix_identity
            .as_ref()
            .map(|identity| state.prefix_cache.lookup(identity, &prompt))
            .transpose()?
            .flatten();
        let mut phase41_audit = ProductionPhase41AuditV1 {
            prefix_cache_result: prefix_cache_eligible.then_some(
                prefix_hit
                    .as_ref()
                    .map_or(ProductionPrefixCacheResultV1::Miss, |hit| {
                        production_prefix_kind(hit.kind)
                    }),
            ),
            ..ProductionPhase41AuditV1::default()
        };
        if loaded_checkpoint.is_some() {
            phase41_audit.checkpoint_operation = Some(ProductionCheckpointOperationV1::Load);
            phase41_audit.checkpoint_result = Some(ProductionCheckpointResultV1::Succeeded);
        }
        if let Some(hit) = prefix_hit.as_ref() {
            let audit = hit.prefix.fork_audit();
            phase41_audit.prefix_shared_pages = audit.shared_pages();
            phase41_audit.prefix_copied_bytes = audit.copied_bytes();
        }
        let prefix_continuation = prefix_hit.is_some();
        let checkpoint_save_status = Arc::new(AtomicU8::new(CHECKPOINT_STATUS_NONE));
        let checkpoint_save = if loaded_checkpoint.is_none() && !prefix_continuation {
            state
                .checkpoint
                .as_ref()
                .and_then(|runtime| runtime.save_name.as_ref())
                .map(|name| {
                    Ok(QwenCheckpointSaveV1 {
                        store: state
                            .checkpoint
                            .as_ref()
                            .expect("checkpoint runtime exists for save name")
                            .store
                            .clone(),
                        name: name.clone(),
                        identity: qwen_checkpoint_identity(
                            &graph,
                            &state.plan,
                            &state.tokenizer,
                            &state.renderer,
                            &state.target,
                            state.fp8_provider.as_deref(),
                            adapter_request.identity(),
                            state.kv_cache_encoding,
                            &prompt,
                        )?,
                        prompt_tokens: prompt.clone(),
                        status: Arc::clone(&checkpoint_save_status),
                    })
                })
                .transpose()?
        } else {
            None
        };
        let checkpoint_save_requested = checkpoint_save.is_some();
        let persistent_capture_graph = state.persistent_capture_requested.then(|| graph.clone());
        let (owner, prefix_hit) = if let (Some(checkpoint), Some(identity)) =
            (loaded_checkpoint.as_ref(), checkpoint_identity.as_ref())
        {
            let owner = state
                .resident
                .new_request_from_checkpoint_with_adapters(
                    checkpoint,
                    graph,
                    identity,
                    adapter_request.clone(),
                )
                .map_err(|_| BackendErrorV1::new("Qwen checkpoint request provisioning failed"))?;
            (owner, None)
        } else if let Some(hit) = prefix_hit {
            let owner = state
                .resident
                .new_request_from_prefix_with_adapters(&hit.prefix, graph, adapter_request.clone())
                .map_err(|error| {
                    BackendErrorV1::new(format!("prefix request provisioning failed: {error}"))
                })?;
            (owner, Some(hit))
        } else {
            let owner = state
                .resident
                .new_request_for_session_with_adapters(
                    Arc::clone(&state.session),
                    graph,
                    adapter_request,
                )
                .map_err(|error| {
                    BackendErrorV1::new(format!("request provisioning failed: {error}"))
                })?;
            (owner, None)
        };
        let execution_prompt = checkpoint_suffix.as_deref().unwrap_or(&prompt);
        let mut allocated = state.session.memory_snapshot();
        let mut random =
            OsSamplingRandom::for_randomness_and_seed(requires_randomness, resolved_sampling_seed)
                .map_err(|error| BackendErrorV1::new(format!("sampling source failed: {error}")))?;
        let mut output_sink = OutputSinkAdapterV1 { inner: sink };
        let mut post_cow_error = None;
        let (outcome, dispatch, memory, prefill_chunk_count, published_prefix) =
            if let Some(multimodal_prompt) = multimodal_prompt.as_ref() {
                let mut owner = owner;
                let mut executor = QwenMultimodalExecutorV1 {
                    inner: &mut owner,
                    prompt: multimodal_prompt,
                    prefilled: false,
                };
                let outcome = generate_with_optional_assistant_prefill(
                    &service,
                    &mut executor,
                    execution_prompt,
                    &assistant_prefill_tokens,
                    &generation,
                    cancellation,
                    &mut random,
                    &mut output_sink,
                );
                let dispatch = owner.audit_snapshot().ok();
                let memory = owner.memory_audit_snapshot().ok();
                let prefill_chunk_count = owner.prefill_chunk_count();
                drop(owner);
                (outcome, dispatch, memory, Some(prefill_chunk_count), None)
            } else if let (true, Some(mtp_resident), Some(mtp_plan)) =
                (mtp_target, &state.mtp_resident, &state.mtp_plan)
            {
                let mtp_graph = build_qwen35_mtp_graph(
                    state.lock.as_ref().expect("MTP requires dense Qwen lock"),
                    mtp_plan,
                    state_capacity,
                )
                .map_err(|error| {
                    BackendErrorV1::new(format!("MTP request graph failed: {error}"))
                })?;
                let mtp_owner = mtp_resident
                    .new_request_for_session(Arc::clone(&state.session), mtp_graph)
                    .map_err(|error| {
                        BackendErrorV1::new(format!("MTP request provisioning failed: {error}"))
                    })?;
                allocated = state.session.memory_snapshot();
                let mut executor = SpeculativeGenerationAdapterV1::new(
                    QwenMtpGenerationExecutorV1::new_with_draft_width(owner, mtp_owner, 1)
                        .map_err(|error| BackendErrorV1::new(error.to_string()))?,
                );
                let outcome = generate_with_optional_assistant_prefill(
                    &service,
                    &mut executor,
                    execution_prompt,
                    &assistant_prefill_tokens,
                    &generation,
                    cancellation,
                    &mut random,
                    &mut output_sink,
                );
                let dispatch = executor.inner().target().audit_snapshot().ok();
                let memory = executor.inner().target().memory_audit_snapshot().ok();
                let prefill_chunk_count = executor.inner().target().prefill_chunk_count();
                let proposed = executor.inner().proposed_draft_tokens();
                let accepted = executor.inner().accepted_draft_tokens();
                phase41_audit.draft_provider = Some(ProductionDraftProviderV1::Mtp);
                phase41_audit.draft_proposed_tokens = proposed;
                phase41_audit.draft_accepted_tokens = accepted;
                phase41_audit.draft_rejected_tokens = proposed.saturating_sub(accepted);
                drop(executor);
                (outcome, dispatch, memory, Some(prefill_chunk_count), None)
            } else if let DraftStartupConfigV1::Ngram { order, width } = &state.phase41.draft {
                let target = match prefix_hit {
                    Some(hit) => {
                        QwenPrefixGenerationExecutorV1::from_hit(owner, hit, usize::from(*width))
                    }
                    None => QwenPrefixGenerationExecutorV1::fresh(
                        owner,
                        usize::from(*width),
                        prefix_cache_eligible,
                    ),
                };
                let provider = production_ngram_provider(*order)?;
                let mut executor = SpeculativeGenerationAdapterV1::with_provider_and_draft_width(
                    target,
                    provider,
                    usize::from(*width),
                )
                .map_err(|error| BackendErrorV1::new(error.to_string()))?;
                let outcome = generate_with_optional_assistant_prefill(
                    &service,
                    &mut executor,
                    execution_prompt,
                    &assistant_prefill_tokens,
                    &generation,
                    cancellation,
                    &mut random,
                    &mut output_sink,
                );
                let dispatch = executor.inner().inner().audit_snapshot().ok();
                let memory = executor.inner().inner().memory_audit_snapshot().ok();
                let prefill_chunk_count = executor.inner().inner().prefill_chunk_count();
                let accounting = executor.accounting();
                phase41_audit.draft_provider = Some(ProductionDraftProviderV1::Ngram);
                phase41_audit.draft_proposed_tokens = accounting.proposed_tokens();
                phase41_audit.draft_accepted_tokens = accounting.accepted_tokens();
                phase41_audit.draft_rejected_tokens = accounting.rejected_tokens();
                phase41_audit.context_shift_count = executor.inner().context_shift_count();
                if prefix_continuation {
                    match executor.inner().refresh_prefix_fork_audit() {
                        Ok(audit) => {
                            phase41_audit.prefix_cow_pages = phase41_audit
                                .prefix_shared_pages
                                .saturating_sub(audit.shared_pages());
                            phase41_audit.prefix_shared_pages = audit.shared_pages();
                            phase41_audit.prefix_copied_bytes = audit.copied_bytes();
                        }
                        Err(error) => {
                            post_cow_error = Some(BackendErrorV1::new(format!(
                                "Qwen prefix COW accounting failed: {error}"
                            )));
                            executor.cancel();
                        }
                    }
                }
                let published_prefix = executor.inner_mut().take_published_prefix();
                drop(executor);
                (
                    outcome,
                    dispatch,
                    memory,
                    Some(prefill_chunk_count),
                    published_prefix,
                )
            } else {
                let mut executor = match prefix_hit {
                    Some(hit) => QwenPrefixGenerationExecutorV1::from_hit(owner, hit, 1),
                    None => {
                        let executor =
                            QwenPrefixGenerationExecutorV1::fresh(owner, 1, prefix_cache_eligible);
                        match checkpoint_save {
                            Some(save) => executor.with_checkpoint_save(save),
                            None => executor,
                        }
                    }
                };
                if let Some(policy) = context_policy {
                    let lock = state.lock.as_ref().ok_or_else(|| {
                        BackendErrorV1::new("context shifting requires the dense Qwen lock")
                    })?;
                    executor = executor.with_context_shift(
                        state.resident.clone(),
                        lock.clone(),
                        state.plan.clone(),
                        policy,
                        state_capacity,
                    );
                }
                let mut outcome = generate_with_optional_assistant_prefill(
                    &service,
                    &mut executor,
                    execution_prompt,
                    &assistant_prefill_tokens,
                    &generation,
                    cancellation,
                    &mut random,
                    &mut output_sink,
                );
                // Capture request observability before canonical rebase.  The
                // rebase owner is a bookkeeping prefix for the next turn;
                // publishing its empty-generation audit would hide the actual
                // user request's dispatch and memory work.
                let generation_dispatch = executor.inner().audit_snapshot().ok();
                let generation_memory = executor.inner().memory_audit_snapshot().ok();
                let generation_prefill_chunk_count = executor.inner().prefill_chunk_count();
                if state.persistent_capture_requested {
                    if let Ok(result) = outcome.as_ref() {
                        let graph = persistent_capture_graph
                            .as_ref()
                            .expect("persistent capture graph was cloned before ownership");
                        let token_history = qwen_persistent_history_tokens(state, request, result)
                            .and_then(|token_history| {
                                executor
                                    .rebase_persistent_owner(
                                        &state.resident,
                                        Arc::clone(&state.session),
                                        graph.clone(),
                                        &token_history,
                                    )
                                    .map(|()| token_history)
                            });
                        match token_history.and_then(|token_history| {
                            capture_qwen_persistent_checkpoint(
                                state,
                                graph,
                                &executor,
                                &token_history,
                                prompt.len(),
                                result,
                            )
                        }) {
                            Ok(captured) => state.persistent_capture = Some(captured),
                            Err(error) => outcome = Err(error),
                        }
                    }
                    state.persistent_capture_requested = false;
                }
                let dispatch = generation_dispatch;
                let memory = generation_memory;
                let prefill_chunk_count = generation_prefill_chunk_count;
                phase41_audit.context_shift_count = executor.context_shift_count();
                if prefix_continuation {
                    match executor.refresh_prefix_fork_audit() {
                        Ok(audit) => {
                            phase41_audit.prefix_cow_pages = phase41_audit
                                .prefix_shared_pages
                                .saturating_sub(audit.shared_pages());
                            phase41_audit.prefix_shared_pages = audit.shared_pages();
                            phase41_audit.prefix_copied_bytes = audit.copied_bytes();
                        }
                        Err(error) => {
                            post_cow_error = Some(BackendErrorV1::new(format!(
                                "Qwen prefix COW accounting failed: {error}"
                            )));
                            executor.cancel();
                        }
                    }
                }
                let published_prefix = executor.take_published_prefix();
                drop(executor);
                (
                    outcome,
                    dispatch,
                    memory,
                    Some(prefill_chunk_count),
                    published_prefix,
                )
            };
        if checkpoint_save_requested {
            phase41_audit.checkpoint_operation = Some(ProductionCheckpointOperationV1::Save);
            phase41_audit.checkpoint_result = Some(
                if checkpoint_save_status.load(Ordering::Acquire) == CHECKPOINT_STATUS_SUCCEEDED {
                    ProductionCheckpointResultV1::Succeeded
                } else {
                    ProductionCheckpointResultV1::Failed
                },
            );
        }
        if let (Some(identity), Some(prefix)) = (prefix_identity, published_prefix) {
            let _ = state.prefix_cache.publish(identity, &prompt, prefix);
        }
        let cleanup = state.session.memory_snapshot();
        let cleanup_result = require_request_memory_baseline(
            cleanup,
            state.prefix_cache.baseline_bytes()?,
            "request cleanup",
        )
        .and_then(|()| {
            if cleanup.model_resident().current_bytes() == state.model_ready_current_bytes {
                Ok(())
            } else {
                Err(BackendErrorV1::new(
                    "model-resident accounting changed after request cleanup",
                ))
            }
        });
        let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let generation_result = outcome
            .map_err(|error| BackendErrorV1::new(format!("generation failed: {error}")))
            .and_then(|result| {
                let dispatch = dispatch.as_ref().ok_or_else(|| {
                    BackendErrorV1::new("completed generation has no dispatch audit")
                })?;
                if dispatch.selected_backend() != "hip"
                    || dispatch.target() != state.target
                    || dispatch.fallback_used()
                    || !dispatch.all_dispatches_hip()
                {
                    return Err(BackendErrorV1::new(
                        "completed generation dispatch audit is not exact HIP/no-fallback",
                    ));
                }
                if memory.is_none() {
                    return Err(BackendErrorV1::new(
                        "completed generation has no physical-memory audit",
                    ));
                }
                publish_generation_logprobs(request, &state.tokenizer, &result, output_sink.inner)?;
                let finish_reason = match result.finish_reason() {
                    sllm_frontend::FinishReasonV1::Stop => FinishReasonV1::Stop,
                    sllm_frontend::FinishReasonV1::Length => FinishReasonV1::Length,
                };
                let usage = result.usage();
                Ok(BackendCompletionV1 {
                    finish_reason,
                    usage: TokenUsageV1::new(usage.prompt_tokens(), usage.completion_tokens())
                        .map_err(|error| BackendErrorV1::new(error.to_string()))?,
                    matched_stop: result.matched_stop().map(str::to_owned),
                })
            });
        let result = match (post_cow_error, generation_result, cleanup_result) {
            (Some(error), _, _) => Err(error),
            (None, _, Err(error)) => Err(error),
            (None, result, Ok(())) => result,
        };
        let first_kv = memory.as_ref().and_then(|audit| audit.kv_layers().first());
        let committed_kv_bytes = memory
            .as_ref()
            .and_then(|audit| audit.committed_kv_bytes().ok());
        let completion_tokens = result
            .as_ref()
            .ok()
            .map(|value| value.usage.completion_tokens);
        self.record_audit(ProductionRequestAuditV1 {
            outcome: if cancellation.is_cancelled() {
                "cancelled".to_owned()
            } else if result.is_ok() {
                "completed".to_owned()
            } else {
                "failed".to_owned()
            },
            target: state.target.clone(),
            weight_encoding: match state.fp8_provider.as_deref() {
                None => "bf16".to_owned(),
                Some("converted-bf16") => "bf16-converted-from-ocp-e4m3fn".to_owned(),
                Some("native-fnuz") => "e4m3fnuz-converted-from-ocp-e4m3fn-outer-f32".to_owned(),
                Some("nvfp4-packed-dequant") => "nvfp4-e2m1-block16-e4m3fn-tensor-f32".to_owned(),
                Some("ocp-mxfp4-w4a4-mixed") => "ocp-mxfp4-e2m1-block32-e8m0-mixed".to_owned(),
                Some(_) => "ocp-e4m3fn-outer-f32".to_owned(),
            },
            kv_cache_encoding: match state.kv_cache_encoding {
                KvCacheEncoding::Fp16 => "fp16",
                KvCacheEncoding::Fp8E4M3Fn => "fp8",
                KvCacheEncoding::Fp8E4M3FnStatic => "fp8-static",
                KvCacheEncoding::Nvfp4 => "nvfp4",
            }
            .to_owned(),
            fp8_provider: state.fp8_provider.clone(),
            prompt_tokens,
            requested_max_completion_tokens: generation.max_new_tokens(),
            completion_tokens,
            elapsed_ns,
            selected_backend: dispatch
                .as_ref()
                .map(|audit| audit.selected_backend().to_owned()),
            fallback_used: dispatch.as_ref().map(|audit| audit.fallback_used()),
            all_dispatches_hip: dispatch.as_ref().map(|audit| audit.all_dispatches_hip()),
            submission_count: dispatch.as_ref().map(|audit| audit.submission_count()),
            kernel_dispatch_count: dispatch.as_ref().map(|audit| audit.kernel_dispatch_count()),
            full_attention_layers: memory.as_ref().map_or(0, |audit| audit.kv_layers().len()),
            linear_attention_layers: memory
                .as_ref()
                .map_or(0, |audit| audit.linear_attention_layers()),
            logical_kv_capacity_tokens: first_kv.map(|layer| layer.logical_capacity_tokens()),
            observed_kv_length_tokens: first_kv.map(|layer| layer.observed_length_tokens()),
            physical_page_bytes: first_kv.map(|layer| layer.physical().physical_page_bytes()),
            kv_memory_kind: first_kv.map(|layer| match layer.physical().memory_kind() {
                sllm_core::KvMemoryKind::VirtualContiguous => "virtual-contiguous".to_owned(),
                sllm_core::KvMemoryKind::ContiguousResident => "contiguous-resident".to_owned(),
            }),
            tokens_per_page: first_kv.map(|layer| layer.physical().tokens_per_page()),
            mapped_kv_capacity_tokens: first_kv
                .map(|layer| layer.physical().mapped_token_capacity()),
            committed_kv_bytes,
            prefill_chunk_capacity_tokens: Some(prefill_chunk_capacity_tokens),
            prefill_chunk_count,
            placement_total_memory_bytes: Some(placement_total_memory_bytes),
            placement_available_memory_bytes: Some(placement_available_memory_bytes),
            placement_required_bytes: Some(placement.required_bytes()),
            placement_incremental_required_bytes: Some(placement_incremental_required_bytes),
            workspace_separate_allocation_bytes: Some(placement.workspace_baseline_bytes()),
            workspace_arena_bytes: Some(placement.workspace_arena_bytes()),
            allocated_request_state_bytes: allocated.request_state().current_bytes(),
            allocated_workspace_bytes: allocated.workspace().current_bytes(),
            cleanup_request_state_bytes: cleanup.request_state().current_bytes(),
            cleanup_workspace_bytes: cleanup.workspace().current_bytes(),
            phase41: phase41_audit,
        });
        result
    }
}

struct PersistentChatSinkV1;

impl GenerationDeltaSinkV1 for PersistentChatSinkV1 {
    fn publish(&mut self, _delta: &str) -> Result<(), BackendErrorV1> {
        Ok(())
    }
}

#[derive(Default)]
struct PersistentCheckpointStateV1 {
    current: Option<SessionCheckpoint>,
    pending: Option<SessionCheckpoint>,
}

impl PersistentCheckpointStateV1 {
    fn stage(&mut self, checkpoint: SessionCheckpoint) -> Result<(), BackendErrorV1> {
        if self.pending.is_some() {
            return Err(BackendErrorV1::new(
                "persistent chat has an uncommitted turn",
            ));
        }
        self.pending = Some(checkpoint);
        Ok(())
    }

    fn candidate_with_conversation(
        &self,
        conversation: &[u8],
    ) -> Result<SessionCheckpoint, BackendErrorV1> {
        let mut candidate = self.pending.clone().ok_or_else(|| {
            BackendErrorV1::new("persistent chat has no pending turn to checkpoint")
        })?;
        candidate.payload.conversation = conversation.to_vec();
        candidate
            .validate()
            .map_err(|error| BackendErrorV1::new(error.to_string()))?;
        Ok(candidate)
    }

    fn promote(&mut self, checkpoint: SessionCheckpoint) {
        self.current = Some(checkpoint);
        self.pending = None;
    }
}

fn persistent_chat_finish_reason(
    completion: &BackendCompletionV1,
    reverse_prompts: &[String],
) -> QwenPersistentChatFinishReasonV1 {
    if completion
        .matched_stop
        .as_deref()
        .is_some_and(|stop| reverse_prompts.iter().any(|reverse| reverse == stop))
    {
        QwenPersistentChatFinishReasonV1::ReversePrompt
    } else {
        match completion.finish_reason {
            FinishReasonV1::Stop => QwenPersistentChatFinishReasonV1::Stop,
            FinishReasonV1::Length => QwenPersistentChatFinishReasonV1::Length,
        }
    }
}

/// Persistent multi-turn owner for the dense BF16 Qwen text path.
pub struct QwenPersistentChatSessionV1 {
    backend: QwenChatBackendV1,
    store: Arc<CheckpointStore>,
    checkpoints: PersistentCheckpointStateV1,
}

impl QwenPersistentChatSessionV1 {
    pub fn open(config: QwenPersistentChatSessionConfigV1) -> Result<Self, BackendErrorV1> {
        if config.checkpoint_directory.as_os_str().is_empty()
            || config.checkpoint_quota_bytes == 0
            || !matches!(
                config.backend.phase41.prefix_cache,
                PrefixCacheStartupConfigV1::Disabled
            )
            || !matches!(
                config.backend.phase41.context_window,
                ContextWindowStartupConfigV1::Disabled
            )
            || !matches!(
                config.backend.phase41.checkpoint,
                CheckpointStartupConfigV1::Disabled
            )
            || !matches!(config.backend.phase41.draft, DraftStartupConfigV1::Disabled)
        {
            return Err(BackendErrorV1::new(
                "persistent chat requires a nonempty checkpoint store and all Phase 41 features disabled",
            ));
        }
        let store = Arc::new(
            CheckpointStore::new(&config.checkpoint_directory, config.checkpoint_quota_bytes)
                .map_err(|error| BackendErrorV1::new(error.to_string()))?,
        );
        let backend = QwenChatBackendV1::open(config.backend)?;
        if let Err(error) = backend.install_persistent_checkpoint(Arc::clone(&store), None) {
            let _ = backend.shutdown();
            return Err(error);
        }
        Ok(Self {
            backend,
            store,
            checkpoints: PersistentCheckpointStateV1::default(),
        })
    }

    pub fn turn(
        &mut self,
        request: QwenPersistentChatTurnRequestV1,
    ) -> Result<QwenPersistentChatTurnResultV1, BackendErrorV1> {
        let cancellation = GenerationCancellationV1::new();
        self.turn_with_cancellation(request, &cancellation)
    }

    pub fn turn_with_cancellation(
        &mut self,
        request: QwenPersistentChatTurnRequestV1,
        cancellation: &GenerationCancellationV1,
    ) -> Result<QwenPersistentChatTurnResultV1, BackendErrorV1> {
        if self.checkpoints.pending.is_some() {
            return Err(BackendErrorV1::new(
                "persistent chat has an uncommitted turn; call commit_turn or save_checkpoint",
            ));
        }
        let reverse_prompts = request.reverse_prompts.clone();
        let api_request = ChatCompletionRequestV1::from_persistent_chat(
            "sllm-qwen35".to_owned(),
            request.messages,
            request.max_new_tokens,
            request.stop_sequences,
            reverse_prompts.clone(),
            request.thinking,
            request.reasoning_budget,
        )
        .map_err(|error| BackendErrorV1::new(error.to_string()))?;
        self.backend.install_persistent_checkpoint(
            Arc::clone(&self.store),
            self.checkpoints.current.as_ref(),
        )?;
        self.backend.arm_persistent_checkpoint_capture()?;
        let mut sink = PersistentChatSinkV1;
        let completion = self.backend.generate(&api_request, cancellation, &mut sink);
        let captured = self.backend.take_persistent_checkpoint_capture()?;
        let completion = completion?;
        let captured = captured.ok_or_else(|| {
            BackendErrorV1::new("persistent chat generation completed without a checkpoint")
        })?;
        let finish_reason = persistent_chat_finish_reason(&completion, &reverse_prompts);
        self.checkpoints.stage(captured.checkpoint)?;
        let usage = TokenUsageV1::new(captured.prompt_tokens, completion.usage.completion_tokens)
            .map_err(|error| BackendErrorV1::new(error.to_string()))?;
        Ok(QwenPersistentChatTurnResultV1 {
            text: captured.text,
            reasoning: captured.reasoning,
            finish_reason,
            usage,
        })
    }

    pub fn load_checkpoint(&mut self, name: &str) -> Result<Vec<u8>, BackendErrorV1> {
        if self.checkpoints.pending.is_some() {
            return Err(BackendErrorV1::new(
                "cannot load a checkpoint while a turn is pending commit",
            ));
        }
        let checkpoint = self
            .store
            .load_validated(name)
            .map_err(|error| BackendErrorV1::new(error.to_string()))?;
        self.backend
            .install_persistent_checkpoint(Arc::clone(&self.store), Some(&checkpoint))?;
        let conversation = checkpoint.payload.conversation.clone();
        self.checkpoints.current = Some(checkpoint);
        Ok(conversation)
    }

    /// Commits the most recent successful turn for ordinary interactive use
    /// when no named checkpoint save was requested.  The caller supplies the
    /// canonical transcript bytes so the checkpoint conversation and KV/token
    /// history advance atomically.  A successful named save already promotes
    /// the same pending checkpoint, so this is idempotent.
    pub fn commit_turn(&mut self, conversation: &[u8]) -> Result<(), BackendErrorV1> {
        if self.checkpoints.pending.is_none() {
            return Ok(());
        }
        let candidate = self.checkpoints.candidate_with_conversation(conversation)?;
        self.backend
            .install_persistent_checkpoint(Arc::clone(&self.store), Some(&candidate))?;
        self.checkpoints.promote(candidate);
        Ok(())
    }

    /// Discards a failed/unwanted turn and reinstalls the last committed
    /// checkpoint, leaving the persistent owner ready for another turn.
    pub fn discard_pending_turn(&mut self) -> Result<(), BackendErrorV1> {
        self.checkpoints.pending = None;
        self.backend.install_persistent_checkpoint(
            Arc::clone(&self.store),
            self.checkpoints.current.as_ref(),
        )
    }

    pub fn save_checkpoint(
        &mut self,
        name: &str,
        conversation: &[u8],
    ) -> Result<(), BackendErrorV1> {
        let previous = self.checkpoints.current.clone();
        let candidate = self.checkpoints.candidate_with_conversation(conversation)?;
        self.backend
            .install_persistent_checkpoint(Arc::clone(&self.store), Some(&candidate))?;
        if let Err(error) = self.store.save(name, &candidate) {
            let _ = self
                .backend
                .install_persistent_checkpoint(Arc::clone(&self.store), previous.as_ref());
            return Err(BackendErrorV1::new(error.to_string()));
        }
        self.checkpoints.promote(candidate);
        Ok(())
    }

    pub fn shutdown(&self) -> Result<ProductionShutdownAuditV1, BackendErrorV1> {
        self.backend.shutdown()
    }
}

impl ChatGenerationBackendV1 for Gemma4ChatBackendV1 {
    fn observability_snapshot(&self) -> BackendObservabilitySnapshotV1 {
        Gemma4ChatBackendV1::observability_snapshot(self)
    }

    fn embedding_dimension(&self) -> Option<u32> {
        Some(GEMMA4_HIDDEN_SIZE as u32)
    }

    fn validate_embedding_input(
        &self,
        input: &BackendEmbeddingInputV1,
    ) -> Result<u64, BackendErrorV1> {
        let state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Gemma backend state is poisoned"))?;
        let state = state_guard
            .as_ref()
            .ok_or_else(|| BackendErrorV1::new("Gemma backend is shut down"))?;
        let tokens = match input {
            BackendEmbeddingInputV1::Text(text) => state
                .tokenizer
                .encode(text)
                .map_err(|error| BackendErrorV1::new(error.to_string()))?
                .len(),
            BackendEmbeddingInputV1::TokenIds(tokens) => {
                validate_generation_token_ids(&state.tokenizer, tokens, "input")?.len()
            }
        };
        let tokens = u64::try_from(tokens)
            .map_err(|_| BackendErrorV1::new("embedding token count overflowed u64"))?;
        if tokens == 0 || tokens > u64::from(self.identity.context_length) {
            return Err(BackendErrorV1::new(format!(
                "embedding input token count {tokens} must be in [1,{}]",
                self.identity.context_length
            )));
        }
        Ok(tokens)
    }

    fn tokenize_utility(
        &self,
        text: &str,
        options: TokenizeOptionsV1,
    ) -> Result<TokenizeResultV1, BackendErrorV1> {
        let state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Gemma backend state is poisoned"))?;
        let state = state_guard
            .as_ref()
            .ok_or_else(|| BackendErrorV1::new("Gemma backend is shut down"))?;
        TokenizerUtilityServiceV1::new(&state.tokenizer, None)
            .tokenize(text, options)
            .map_err(|error| BackendErrorV1::new(error.to_string()))
    }

    fn detokenize_utility(
        &self,
        token_ids: &[u32],
        mode: DecodeModeV1,
    ) -> Result<String, BackendErrorV1> {
        let state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Gemma backend state is poisoned"))?;
        let state = state_guard
            .as_ref()
            .ok_or_else(|| BackendErrorV1::new("Gemma backend is shut down"))?;
        TokenizerUtilityServiceV1::new(&state.tokenizer, None)
            .detokenize_ids(token_ids, mode)
            .map_err(|error| BackendErrorV1::new(error.to_string()))
    }

    fn apply_template_utility(
        &self,
        _messages: &[Qwen35ChatMessageV1],
        _options: Qwen35RenderOptionsV1,
    ) -> Result<ApplyTemplateResultV1, BackendErrorV1> {
        Err(BackendErrorV1::new(
            "Gemma 4 has no reviewed chat template in the model lock",
        ))
    }

    fn tokenize_infill_content(&self, text: &str) -> Result<Vec<u32>, BackendErrorV1> {
        let state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Gemma backend state is poisoned"))?;
        let state = state_guard
            .as_ref()
            .ok_or_else(|| BackendErrorV1::new("Gemma backend is shut down"))?;
        state
            .tokenizer
            .encode_without_special_tokens(text)
            .map(|tokens| tokens.as_slice().to_vec())
            .map_err(|error| BackendErrorV1::new(error.to_string()))
    }

    fn embed(
        &self,
        request: &BackendEmbeddingRequestV1,
        cancellation: &GenerationCancellationV1,
    ) -> Result<BackendEmbeddingBatchV1, BackendErrorV1> {
        let mut state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Gemma backend state is poisoned"))?;
        let state = state_guard
            .as_mut()
            .ok_or_else(|| BackendErrorV1::new("Gemma backend is shut down"))?;
        let ready = state.session.memory_snapshot();
        require_request_memory_baseline(
            ready,
            state.prefix_cache.baseline_bytes()?,
            "Gemma embedding admission",
        )?;
        if ready.model_resident().current_bytes() != state.model_ready_current_bytes {
            return Err(BackendErrorV1::new(
                "model-resident accounting changed before Gemma embedding admission",
            ));
        }
        let mut vectors = Vec::with_capacity(request.inputs().len());
        for input in request.inputs() {
            if cancellation.is_cancelled() {
                return Err(BackendErrorV1::new("embedding cancelled"));
            }
            let token_ids = match input {
                BackendEmbeddingInputV1::Text(text) => state
                    .tokenizer
                    .encode(text)
                    .map_err(|error| {
                        BackendErrorV1::new(format!("embedding tokenization failed: {error}"))
                    })?
                    .as_slice()
                    .to_vec(),
                BackendEmbeddingInputV1::TokenIds(token_ids) => {
                    validate_generation_token_ids(&state.tokenizer, token_ids, "input")?
                }
            };
            if u64::try_from(token_ids.len()).unwrap_or(u64::MAX)
                > u64::from(self.identity.context_length)
            {
                return Err(BackendErrorV1::new(format!(
                    "embedding input exceeds the configured context length {}",
                    self.identity.context_length
                )));
            }
            let vector = gemma_embed_one(state, &token_ids, cancellation);
            let cleanup = state.session.memory_snapshot();
            require_request_memory_baseline(
                cleanup,
                state.prefix_cache.baseline_bytes()?,
                "Gemma embedding cleanup",
            )
            .and_then(|()| {
                if cleanup.model_resident().current_bytes() == state.model_ready_current_bytes {
                    Ok(())
                } else {
                    Err(BackendErrorV1::new(
                        "model-resident accounting changed after Gemma embedding cleanup",
                    ))
                }
            })?;
            vectors.push(vector?);
        }
        BackendEmbeddingBatchV1::new(GEMMA4_HIDDEN_SIZE as u32, vectors)
    }

    fn generate(
        &self,
        request: &ChatCompletionRequestV1,
        cancellation: &GenerationCancellationV1,
        sink: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        let started = Instant::now();
        if request.reasoning().enabled() || request.reasoning().separate_reasoning() {
            return Err(BackendErrorV1::new(
                "Gemma 4 base raw-text profile does not support reasoning mode",
            ));
        }
        let mut state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Gemma backend state is poisoned"))?;
        let state = state_guard
            .as_mut()
            .ok_or_else(|| BackendErrorV1::new("Gemma backend is shut down"))?;
        let ready = state.session.memory_snapshot();
        require_request_memory_baseline(
            ready,
            state.prefix_cache.baseline_bytes()?,
            "Gemma request admission",
        )?;
        if ready.model_resident().current_bytes() != state.model_ready_current_bytes {
            return Err(BackendErrorV1::new(
                "Gemma model-resident accounting changed before request admission",
            ));
        }

        let service = GenerationServiceV1::new(&state.tokenizer, None, &state.stop_policy)
            .map_err(|error| BackendErrorV1::new(format!("generation service failed: {error}")))?;
        let mut generation = generation_config_for_request(request, &state.tokenizer, None)?;
        let requires_logits = generation
            .sampler_chain()
            .map_or(generation.sampling().requires_logits(), |chain| {
                chain.requires_logits()
            })
            || generation.grammar().is_some();
        let requires_randomness = generation.sampler_chain().map_or(
            generation.sampling().requires_logits(),
            SamplerChainConfigV1::requires_randomness,
        );
        let resolved_sampling_seed = requires_randomness
            .then(|| OsSamplingRandom::resolve_seed(request.sampling_seed()))
            .transpose()
            .map_err(|error| BackendErrorV1::new(format!("sampling seed failed: {error}")))?;
        if let Some(seed) = resolved_sampling_seed {
            generation = generation.with_device_selector_seed(seed);
        }
        let prepared_prompt = gemma_generation_prompt(request, &service, &state.tokenizer)?;
        let assistant_prefill_tokens = prepared_prompt.assistant_prefill_token_ids().to_vec();
        let prompt = prepared_prompt.token_ids().to_vec();
        let loaded_checkpoint = state
            .checkpoint
            .as_ref()
            .and_then(|runtime| runtime.loaded.clone());
        if loaded_checkpoint.is_some() && !assistant_prefill_tokens.is_empty() {
            return Err(BackendErrorV1::new(
                "assistant prefill cannot be combined with a loaded Gemma checkpoint",
            ));
        }
        let checkpoint_suffix = loaded_checkpoint
            .as_ref()
            .map(|checkpoint| {
                let prefix = &checkpoint.payload.token_history;
                if prompt.len() <= prefix.len() || !prompt.starts_with(prefix) {
                    return Err(BackendErrorV1::new(
                        "Gemma checkpoint request must extend the loaded prompt token prefix",
                    ));
                }
                Ok(prompt[prefix.len()..].to_vec())
            })
            .transpose()?;
        let prompt_tokens = u64::try_from(prompt.len())
            .map_err(|_| BackendErrorV1::new("prompt token count overflowed u64"))?;
        let context_policy = match &state.phase41.context_window {
            ContextWindowStartupConfigV1::Disabled => None,
            ContextWindowStartupConfigV1::KeepPrefixRecentV1 {
                keep_prefix,
                keep_recent,
            } => Some(
                ContextPositionPolicyV1::keep_prefix_recent_v1(*keep_prefix, *keep_recent)
                    .map_err(|error| BackendErrorV1::new(error.to_string()))?,
            ),
        };
        if context_policy.is_some() && generation.device_selector_seed().is_some() {
            return Err(BackendErrorV1::new(
                "context-window shifting cannot be combined with device-selector sampling",
            ));
        }
        if context_policy.is_some()
            && !matches!(
                state.phase41.prefix_cache,
                PrefixCacheStartupConfigV1::Disabled
            )
        {
            return Err(BackendErrorV1::new(
                "Gemma context-window shifting cannot be combined with prefix cache",
            ));
        }
        let state_capacity = if context_policy.is_some() || state.checkpoint.is_some() {
            u64::from(self.identity.context_length)
        } else {
            prompt_tokens
                .checked_add(u64::from(generation.max_new_tokens()))
                .ok_or_else(|| BackendErrorV1::new("request state capacity overflowed u64"))?
        };
        if state_capacity > u64::from(self.identity.context_length) {
            return Err(BackendErrorV1::new(format!(
                "request requires {state_capacity} context tokens but the server was started with --context-length {}",
                self.identity.context_length
            )));
        }
        let checkpoint_identity = loaded_checkpoint
            .as_ref()
            .map(|checkpoint| {
                let descriptor_digest = state.checkpoint_descriptor_digest.ok_or_else(|| {
                    BackendErrorV1::new("Gemma checkpoint descriptor is unavailable")
                })?;
                let expected = gemma_checkpoint_identity(
                    state._lock.fingerprint(),
                    &state.plan,
                    &state.tokenizer,
                    &state.target,
                    descriptor_digest,
                    &checkpoint.payload.token_history,
                )?;
                if checkpoint.header.identity != expected {
                    return Err(BackendErrorV1::new(
                        "Gemma checkpoint identity differs from the request graph",
                    ));
                }
                Ok(expected)
            })
            .transpose()?;
        let prefix_cache_enabled = !matches!(
            state.phase41.prefix_cache,
            PrefixCacheStartupConfigV1::Disabled
        );
        let prefix_cache_eligible = prefix_cache_enabled
            && assistant_prefill_tokens.is_empty()
            && !requires_logits
            && match state.phase41.prefix_cache {
                PrefixCacheStartupConfigV1::Disabled => false,
                PrefixCacheStartupConfigV1::Enabled {
                    max_logical_tokens, ..
                } => prompt_tokens <= max_logical_tokens,
            };
        let prefix_identity = prefix_cache_eligible
            .then(|| gemma_prefix_identity(state))
            .transpose()?;
        let prefix_hit = prefix_identity
            .as_ref()
            .map(|identity| state.prefix_cache.lookup(identity, &prompt))
            .transpose()?
            .flatten();
        let mut phase41_audit = ProductionPhase41AuditV1 {
            prefix_cache_result: prefix_cache_eligible.then_some(
                prefix_hit
                    .as_ref()
                    .map_or(ProductionPrefixCacheResultV1::Miss, |hit| {
                        production_prefix_kind(hit.kind)
                    }),
            ),
            ..ProductionPhase41AuditV1::default()
        };
        if loaded_checkpoint.is_some() {
            phase41_audit.checkpoint_operation = Some(ProductionCheckpointOperationV1::Load);
            phase41_audit.checkpoint_result = Some(ProductionCheckpointResultV1::Succeeded);
        }
        if let Some(hit) = prefix_hit.as_ref() {
            let audit = hit.prefix.fork_audit();
            phase41_audit.prefix_shared_pages = audit.shared_pages();
            phase41_audit.prefix_copied_bytes = audit.copied_bytes();
        }
        let prefix_continuation = prefix_hit.is_some();
        let checkpoint_save_status = Arc::new(AtomicU8::new(CHECKPOINT_STATUS_NONE));
        let checkpoint_save = if loaded_checkpoint.is_none() && !prefix_continuation {
            state
                .checkpoint
                .as_ref()
                .and_then(|runtime| runtime.save_name.as_ref())
                .map(|name| {
                    let descriptor_digest =
                        state.checkpoint_descriptor_digest.ok_or_else(|| {
                            BackendErrorV1::new("Gemma checkpoint descriptor is unavailable")
                        })?;
                    Ok(QwenCheckpointSaveV1 {
                        store: state
                            .checkpoint
                            .as_ref()
                            .expect("checkpoint runtime exists for save name")
                            .store
                            .clone(),
                        name: name.clone(),
                        identity: gemma_checkpoint_identity(
                            state._lock.fingerprint(),
                            &state.plan,
                            &state.tokenizer,
                            &state.target,
                            descriptor_digest,
                            &prompt,
                        )?,
                        prompt_tokens: prompt.clone(),
                        status: Arc::clone(&checkpoint_save_status),
                    })
                })
                .transpose()?
        } else {
            None
        };
        let checkpoint_save_requested = checkpoint_save.is_some();
        let initial_graph_tokens = context_policy
            .and_then(|policy| {
                policy
                    .plan_initial(prompt_tokens, state_capacity)
                    .ok()
                    .map(|decision| decision.proposed_state().logical_length())
            })
            .unwrap_or(prompt_tokens);
        if requires_logits && context_policy.is_some() && initial_graph_tokens < prompt_tokens {
            return Err(BackendErrorV1::new(
                "context-window shifting with an oversized prompt does not support an initial logits readback",
            ));
        }
        let mut executor = if let (Some(checkpoint), Some(identity)) =
            (loaded_checkpoint.as_ref(), checkpoint_identity.as_ref())
        {
            let suffix_tokens = u64::try_from(
                checkpoint_suffix
                    .as_ref()
                    .expect("checkpoint suffix validated before owner creation")
                    .len(),
            )
            .map_err(|_| BackendErrorV1::new("Gemma checkpoint suffix length overflowed"))?;
            let owner = state
                .resident
                .new_request_from_checkpoint(checkpoint, identity, suffix_tokens, state_capacity)
                .map_err(|_| BackendErrorV1::new("Gemma checkpoint request provisioning failed"))?;
            GemmaPrefixGenerationExecutorV1::fresh(owner, false)
        } else if let Some(hit) = prefix_hit {
            let suffix_tokens = prompt_tokens
                .checked_sub(hit.prefix.committed_length())
                .ok_or_else(|| BackendErrorV1::new("Gemma prefix length exceeded prompt length"))?;
            let owner = state
                .resident
                .new_request_from_prefix(&hit.prefix, suffix_tokens.max(1), state_capacity)
                .map_err(|error| {
                    BackendErrorV1::new(format!("prefix request provisioning failed: {error}"))
                })?;
            GemmaPrefixGenerationExecutorV1::from_hit(owner, hit)
        } else {
            let owner = state
                .resident
                .new_request_for_session(
                    Arc::clone(&state.session),
                    initial_graph_tokens,
                    state_capacity,
                )
                .map_err(|error| {
                    BackendErrorV1::new(format!("request provisioning failed: {error}"))
                })?;
            {
                let executor = GemmaPrefixGenerationExecutorV1::fresh(owner, prefix_cache_eligible);
                if let Some(policy) = context_policy {
                    executor.with_context_shift(state.resident.clone(), policy, state_capacity)
                } else {
                    executor
                }
            }
        };
        if checkpoint_save_requested {
            // The save owner is always the fresh text owner; prefix/context/draft
            // combinations are rejected during startup validation.
            executor = executor.with_checkpoint_save(
                checkpoint_save
                    .expect("checkpoint save request was recorded before owner creation"),
            );
        }
        let allocated = state.session.memory_snapshot();
        let mut random =
            OsSamplingRandom::for_randomness_and_seed(requires_randomness, resolved_sampling_seed)
                .map_err(|error| BackendErrorV1::new(format!("sampling source failed: {error}")))?;
        let mut output_sink = OutputSinkAdapterV1 { inner: sink };
        let mut post_cow_error = None;
        let outcome = generate_with_optional_assistant_prefill(
            &service,
            &mut executor,
            checkpoint_suffix.as_deref().unwrap_or(&prompt),
            &assistant_prefill_tokens,
            &generation,
            cancellation,
            &mut random,
            &mut output_sink,
        );
        let dispatch = executor.inner().audit_snapshot().ok();
        let observed_length = executor.inner().committed_length();
        phase41_audit.context_shift_count = executor.context_shift_count();
        if prefix_continuation {
            match executor.refresh_prefix_fork_audit() {
                Ok(audit) => {
                    phase41_audit.prefix_cow_pages = phase41_audit
                        .prefix_shared_pages
                        .saturating_sub(audit.shared_pages());
                    phase41_audit.prefix_shared_pages = audit.shared_pages();
                    phase41_audit.prefix_copied_bytes = audit.copied_bytes();
                }
                Err(error) => {
                    post_cow_error = Some(BackendErrorV1::new(format!(
                        "Gemma prefix COW accounting failed: {error}"
                    )));
                    executor.cancel();
                }
            }
        }
        let published_prefix = executor.take_published_prefix();
        if checkpoint_save_requested {
            phase41_audit.checkpoint_operation = Some(ProductionCheckpointOperationV1::Save);
            phase41_audit.checkpoint_result = Some(
                if checkpoint_save_status.load(Ordering::Acquire) == CHECKPOINT_STATUS_SUCCEEDED {
                    ProductionCheckpointResultV1::Succeeded
                } else {
                    ProductionCheckpointResultV1::Failed
                },
            );
        }
        drop(executor);
        if let (Some(identity), Some(prefix)) = (prefix_identity, published_prefix) {
            let _ = state.prefix_cache.publish(identity, &prompt, prefix);
        }
        let cleanup = state.session.memory_snapshot();
        let cleanup_result = require_request_memory_baseline(
            cleanup,
            state.prefix_cache.baseline_bytes()?,
            "Gemma request cleanup",
        )
        .and_then(|()| {
            if cleanup.model_resident().current_bytes() == state.model_ready_current_bytes {
                Ok(())
            } else {
                Err(BackendErrorV1::new(
                    "Gemma model-resident accounting changed after request cleanup",
                ))
            }
        });
        let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let generation_result = outcome
            .map_err(|error| BackendErrorV1::new(format!("generation failed: {error}")))
            .and_then(|result| {
                let dispatch = dispatch.as_ref().ok_or_else(|| {
                    BackendErrorV1::new("completed Gemma generation has no dispatch audit")
                })?;
                if dispatch.target() != state.target || dispatch.fallback_used() {
                    return Err(BackendErrorV1::new(
                        "completed Gemma generation is not exact HIP/no-fallback",
                    ));
                }
                publish_generation_logprobs(request, &state.tokenizer, &result, output_sink.inner)?;
                let finish_reason = match result.finish_reason() {
                    sllm_frontend::FinishReasonV1::Stop => FinishReasonV1::Stop,
                    sllm_frontend::FinishReasonV1::Length => FinishReasonV1::Length,
                };
                let usage = result.usage();
                Ok(BackendCompletionV1 {
                    finish_reason,
                    usage: TokenUsageV1::new(usage.prompt_tokens(), usage.completion_tokens())
                        .map_err(|error| BackendErrorV1::new(error.to_string()))?,
                    matched_stop: result.matched_stop().map(str::to_owned),
                })
            });
        let result = match (post_cow_error, generation_result, cleanup_result) {
            (Some(error), _, _) => Err(error),
            (None, _, Err(error)) => Err(error),
            (None, result, Ok(())) => result,
        };
        let completion_tokens = result
            .as_ref()
            .ok()
            .map(|value| value.usage.completion_tokens);
        self.record_audit(ProductionRequestAuditV1 {
            outcome: if cancellation.is_cancelled() {
                "cancelled".to_owned()
            } else if result.is_ok() {
                "completed".to_owned()
            } else {
                "failed".to_owned()
            },
            target: state.target.clone(),
            weight_encoding: state.weight_encoding.clone(),
            kv_cache_encoding: "fp8-static".to_owned(),
            fp8_provider: None,
            prompt_tokens,
            requested_max_completion_tokens: generation.max_new_tokens(),
            completion_tokens,
            elapsed_ns,
            selected_backend: dispatch.as_ref().map(|_| "hip".to_owned()),
            fallback_used: dispatch.as_ref().map(|audit| audit.fallback_used()),
            all_dispatches_hip: dispatch.as_ref().map(|audit| !audit.fallback_used()),
            submission_count: dispatch.as_ref().map(|audit| audit.submission_count()),
            kernel_dispatch_count: dispatch.as_ref().map(|audit| audit.kernel_dispatch_count()),
            full_attention_layers: 8,
            linear_attention_layers: 0,
            logical_kv_capacity_tokens: Some(state_capacity),
            observed_kv_length_tokens: Some(observed_length),
            physical_page_bytes: None,
            kv_memory_kind: Some("contiguous-resident".to_owned()),
            tokens_per_page: None,
            mapped_kv_capacity_tokens: Some(state_capacity),
            committed_kv_bytes: observed_length.checked_mul(state.kv_bytes_per_token),
            prefill_chunk_capacity_tokens: None,
            prefill_chunk_count: None,
            placement_total_memory_bytes: None,
            placement_available_memory_bytes: None,
            placement_required_bytes: None,
            placement_incremental_required_bytes: None,
            workspace_separate_allocation_bytes: None,
            workspace_arena_bytes: None,
            allocated_request_state_bytes: allocated.request_state().current_bytes(),
            allocated_workspace_bytes: allocated.workspace().current_bytes(),
            cleanup_request_state_bytes: cleanup.request_state().current_bytes(),
            cleanup_workspace_bytes: cleanup.workspace().current_bytes(),
            phase41: phase41_audit,
        });
        result
    }
}

fn render_gemma4_raw_messages(messages: &[crate::ChatMessageV1]) -> Result<String, BackendErrorV1> {
    let mut rendered = String::new();
    for message in messages {
        let (role, content) = match message.inner() {
            Qwen35ChatMessageV1::System { content } => ("System", content),
            Qwen35ChatMessageV1::User { content } => ("User", content),
            Qwen35ChatMessageV1::Assistant {
                content,
                reasoning_content,
            } => {
                if reasoning_content.is_some() {
                    return Err(BackendErrorV1::new(
                        "Gemma 4 base raw-text profile rejects reasoning history",
                    ));
                }
                ("Assistant", content)
            }
        };
        rendered.push_str(role);
        rendered.push_str(": ");
        rendered.push_str(content);
        rendered.push('\n');
        if rendered.len() > GEMMA4_RAW_CHAT_MAX_BYTES {
            return Err(BackendErrorV1::new(
                "Gemma raw chat transcript exceeds the host byte limit",
            ));
        }
    }
    rendered.push_str("Assistant:");
    if rendered.len() > GEMMA4_RAW_CHAT_MAX_BYTES {
        return Err(BackendErrorV1::new(
            "Gemma raw chat transcript exceeds the host byte limit",
        ));
    }
    Ok(rendered)
}

struct OutputSinkAdapterV1<'a> {
    inner: &'a mut dyn GenerationDeltaSinkV1,
}

struct QwenPrefixGenerationExecutorV1 {
    inner: QwenExecutionRequest,
    matched_tokens: Option<Vec<u32>>,
    _lease: Option<PrefixLeaseV1>,
    published_prefix: Option<QwenPrefixStateV1>,
    publish_prefix: bool,
    draft_width: usize,
    speculative_block_pending: bool,
    context_shift: Option<QwenContextShiftRuntimeV1>,
    checkpoint_save: Option<QwenCheckpointSaveV1>,
}

struct QwenContextShiftRuntimeV1 {
    resident: QwenResidentModel,
    lock: ModelLock,
    plan: WeightLoadPlan,
    policy: ContextPositionPolicyV1,
    state: ContextWindowStateV1,
    history: Vec<i32>,
    capacity: u64,
}

fn capture_qwen_persistent_checkpoint(
    state: &QwenBackendStateV1,
    graph: &QwenGraph,
    executor: &QwenPrefixGenerationExecutorV1,
    token_history: &[u32],
    prompt_tokens: usize,
    result: &GenerationResultV1,
) -> Result<QwenCapturedChatCheckpointV1, GenerationServiceError> {
    // The persistent owner was rebased to the renderer's completed-history
    // token sequence.  Selected stop/reverse markers and hidden reasoning are
    // therefore absent, while usage still counts every selected token.
    let committed = executor.inner().committed_length();
    if token_history.len() as u64 != committed {
        return Err(GenerationServiceError::Execution(
            "persistent checkpoint token history does not match committed KV length".to_owned(),
        ));
    }
    let identity = qwen_checkpoint_identity(
        graph,
        &state.plan,
        &state.tokenizer,
        &state.renderer,
        &state.target,
        state.fp8_provider.as_deref(),
        executor.inner().adapter_identity(),
        state.kv_cache_encoding,
        token_history,
    )
    .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
    let conversation = state
        .checkpoint
        .as_ref()
        .and_then(|runtime| runtime.loaded.as_ref())
        .map_or_else(Vec::new, |checkpoint| {
            checkpoint.payload.conversation.clone()
        });
    let token_count =
        u64::try_from(token_history.len()).map_err(|_| GenerationServiceError::CountOverflow)?;
    let checkpoint = executor
        .inner()
        .checkpoint(
            identity,
            token_history,
            &conversation,
            &[],
            &[],
            &[],
            token_count,
            token_count,
            1,
        )
        .map_err(|_| {
            GenerationServiceError::Execution("persistent checkpoint capture failed".to_owned())
        })?;
    let reasoning = persistent_reasoning_text(state, result)?;
    Ok(QwenCapturedChatCheckpointV1 {
        checkpoint,
        text: result.output_text().to_owned(),
        reasoning,
        prompt_tokens: u64::try_from(prompt_tokens)
            .map_err(|_| GenerationServiceError::CountOverflow)?,
    })
}

fn qwen_persistent_history_tokens(
    state: &QwenBackendStateV1,
    request: &ChatCompletionRequestV1,
    result: &GenerationResultV1,
) -> Result<Vec<u32>, GenerationServiceError> {
    let mut messages = request
        .messages()
        .iter()
        .map(|message| message.inner().clone())
        .collect::<Vec<_>>();
    messages.push(Qwen35ChatMessageV1::assistant(
        result.output_text().to_owned(),
        None,
    ));
    let rendered = state
        .renderer
        .render_history_prefix(&messages)
        .map_err(|_| GenerationServiceError::Render)?;
    state
        .tokenizer
        .encode_generation(&rendered)
        .map_err(|_| GenerationServiceError::Tokenize)
}

fn persistent_reasoning_text(
    state: &QwenBackendStateV1,
    result: &GenerationResultV1,
) -> Result<Option<String>, GenerationServiceError> {
    let reasoning_ids = result.reasoning_token_ids();
    let reasoning_ids = if state.reasoning_close_token_ids.is_empty() {
        reasoning_ids
    } else {
        reasoning_ids
            .strip_suffix(state.reasoning_close_token_ids.as_slice())
            .unwrap_or(reasoning_ids)
    };
    if reasoning_ids.is_empty() {
        return Ok(None);
    }
    let text = state
        .tokenizer
        .decode_generation(reasoning_ids)
        .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
    Ok((!text.is_empty()).then_some(text))
}

impl QwenPrefixGenerationExecutorV1 {
    fn fresh(inner: QwenExecutionRequest, draft_width: usize, publish_prefix: bool) -> Self {
        Self {
            inner,
            matched_tokens: None,
            _lease: None,
            published_prefix: None,
            publish_prefix,
            draft_width,
            speculative_block_pending: false,
            context_shift: None,
            checkpoint_save: None,
        }
    }

    fn from_hit(inner: QwenExecutionRequest, hit: QwenPrefixHitV1, draft_width: usize) -> Self {
        Self {
            inner,
            matched_tokens: Some(hit.matched_tokens),
            _lease: Some(hit.lease),
            published_prefix: None,
            publish_prefix: false,
            draft_width,
            speculative_block_pending: false,
            context_shift: None,
            checkpoint_save: None,
        }
    }

    fn with_context_shift(
        mut self,
        resident: QwenResidentModel,
        lock: ModelLock,
        plan: WeightLoadPlan,
        policy: ContextPositionPolicyV1,
        capacity: u64,
    ) -> Self {
        self.context_shift = Some(QwenContextShiftRuntimeV1 {
            resident,
            lock,
            plan,
            policy,
            state: ContextWindowStateV1::new(0, 0, 0),
            history: Vec::new(),
            capacity,
        });
        self
    }

    fn with_checkpoint_save(mut self, checkpoint: QwenCheckpointSaveV1) -> Self {
        self.checkpoint_save = Some(checkpoint);
        self
    }

    fn save_checkpoint_after_prefill(
        &mut self,
        prompt_token_count: usize,
    ) -> Result<(), GenerationServiceError> {
        let Some(checkpoint) = self.checkpoint_save.take() else {
            return Ok(());
        };
        if checkpoint.prompt_tokens.len() != prompt_token_count {
            checkpoint
                .status
                .store(CHECKPOINT_STATUS_FAILED, Ordering::Release);
            return Err(GenerationServiceError::Execution(
                "checkpoint prompt length changed before save".to_owned(),
            ));
        }
        let prompt_len = u64::try_from(prompt_token_count).map_err(|_| {
            checkpoint
                .status
                .store(CHECKPOINT_STATUS_FAILED, Ordering::Release);
            GenerationServiceError::CountOverflow
        })?;
        let result = self
            .inner
            .checkpoint(
                checkpoint.identity,
                &checkpoint.prompt_tokens,
                &[],
                &[],
                &[],
                &[],
                prompt_len,
                prompt_len,
                1,
            )
            .map_err(|_| GenerationServiceError::Execution("checkpoint capture failed".to_owned()))
            .and_then(|checkpoint_payload| {
                checkpoint
                    .store
                    .save(&checkpoint.name, &checkpoint_payload)
                    .map(|_| ())
                    .map_err(|_| {
                        GenerationServiceError::Execution("checkpoint save failed".to_owned())
                    })
            });
        if result.is_ok() {
            checkpoint
                .status
                .store(CHECKPOINT_STATUS_SUCCEEDED, Ordering::Release);
        } else {
            checkpoint
                .status
                .store(CHECKPOINT_STATUS_FAILED, Ordering::Release);
        }
        result
    }

    fn inner(&self) -> &QwenExecutionRequest {
        &self.inner
    }

    fn rebase_persistent_owner(
        &mut self,
        resident: &QwenResidentModel,
        session: Arc<ExecutionSession>,
        graph: QwenGraph,
        token_ids: &[u32],
    ) -> Result<(), GenerationServiceError> {
        if self.matched_tokens.is_some()
            || self.context_shift.is_some()
            || self.speculative_block_pending
        {
            return Err(GenerationServiceError::Execution(
                "persistent rebase requires a plain Qwen request owner".to_owned(),
            ));
        }
        if token_ids.is_empty() {
            return Err(GenerationServiceError::EmptyPromptTokens);
        }
        let mut owner = resident
            .new_request_for_session(session, graph)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let token_ids = token_ids
            .iter()
            .map(|&token_id| {
                i32::try_from(token_id).map_err(|_| GenerationServiceError::TokenIdOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        owner
            .prefill(&token_ids)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        self.inner = owner;
        Ok(())
    }

    fn context_shift_count(&self) -> u64 {
        self.context_shift
            .as_ref()
            .map_or(0, |context| context.state.shift_count())
    }

    fn refresh_prefix_fork_audit(&self) -> Result<QwenPrefixForkAuditV1, GenerationServiceError> {
        self.inner
            .refresh_prefix_fork_audit()
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))
    }

    fn take_published_prefix(&mut self) -> Option<QwenPrefixStateV1> {
        self.published_prefix.take()
    }

    fn publish_fresh_prefix(
        &mut self,
        prompt_token_count: usize,
    ) -> Result<(), GenerationServiceError> {
        if self.publish_prefix && self.matched_tokens.is_none() {
            let prefix = self
                .inner
                .prefix_state()
                .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
            require_prompt_only_prefix(prefix.committed_length(), prompt_token_count)?;
            self.published_prefix = Some(prefix);
        }
        Ok(())
    }

    fn prefix_prefill(
        &mut self,
        input_token_ids: &[u32],
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let Some(matched_tokens) = self.matched_tokens.as_deref() else {
            if let Some(context) = self.context_shift.as_mut() {
                let token_history = input_token_ids
                    .iter()
                    .map(|&token| {
                        i32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if token_history.len() as u64 > context.capacity {
                    let decision = context
                        .policy
                        .plan_initial(token_history.len() as u64, context.capacity)
                        .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
                    let retained = decision
                        .retained_token_ids(&token_history)
                        .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
                    let graph = build_qwen35_graph_with_position_payload_mode(
                        &context.lock,
                        &context.plan,
                        retained.len() as u64,
                        context.capacity,
                        sllm_core::AttentionPreprocessPositionPayloadModeV1::Explicit,
                    )
                    .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
                    context
                        .policy
                        .validate_adapter(graph.context_adapter_capabilities())
                        .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
                    let (owner, output) = context
                        .resident
                        .new_request_from_context_shift(
                            graph,
                            decision,
                            ContextWindowStateV1::new(
                                token_history.len() as u64,
                                token_history.len() as u64,
                                0,
                            ),
                            &token_history,
                        )
                        .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
                    self.inner = owner;
                    context.state = decision.proposed_state();
                    context.history = retained;
                    return qwen_step_from_output(
                        &output,
                        output.token_ids().len().saturating_sub(1),
                    );
                }
                context.state = ContextWindowStateV1::new(
                    token_history.len() as u64,
                    token_history.len() as u64,
                    0,
                );
                context.history = token_history;
            }
            let step = GenerationExecutorV1::prefill(
                &mut self.inner,
                input_token_ids,
                include_last_logits,
            )?;
            self.publish_fresh_prefix(input_token_ids.len())?;
            self.save_checkpoint_after_prefill(input_token_ids.len())?;
            return Ok(step);
        };
        if include_last_logits || !input_token_ids.starts_with(matched_tokens) {
            return Err(GenerationServiceError::Execution(
                "prefix continuation requires exact greedy prompt semantics".to_owned(),
            ));
        }
        let suffix = input_token_ids[matched_tokens.len()..]
            .iter()
            .map(|&token| i32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        let output = self
            .inner
            .continue_from_prefix(&suffix)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        qwen_step_from_output(&output, output.token_ids().len().saturating_sub(1))
    }
}

impl GenerationExecutorV1 for QwenPrefixGenerationExecutorV1 {
    fn prefill(
        &mut self,
        input_token_ids: &[u32],
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        self.prefix_prefill(input_token_ids, include_last_logits)
    }

    fn decode(
        &mut self,
        token_id: u32,
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let output = GenerationExecutorV1::decode(&mut self.inner, token_id, include_last_logits)?;
        if let Some(context) = self.context_shift.as_mut() {
            context.history.push(
                i32::try_from(token_id).map_err(|_| GenerationServiceError::TokenIdOverflow)?,
            );
            context.state = context
                .state
                .after_append(1, context.capacity)
                .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        }
        Ok(output)
    }

    fn before_decode(&mut self, _token_id: u32) -> Result<(), GenerationServiceError> {
        let Some(context) = self.context_shift.as_mut() else {
            return Ok(());
        };
        let decision = context
            .policy
            .plan(context.state, context.capacity, 1)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        if !decision.requires_shift() {
            return Ok(());
        }
        let retained = decision
            .retained_token_ids(&context.history)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let graph = build_qwen35_graph_with_position_payload_mode(
            &context.lock,
            &context.plan,
            retained.len() as u64,
            context.capacity,
            sllm_core::AttentionPreprocessPositionPayloadModeV1::Explicit,
        )
        .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        context
            .policy
            .validate_adapter(graph.context_adapter_capabilities())
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let (owner, _output) = context
            .resident
            .new_request_from_context_shift(graph, decision, context.state, &context.history)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        self.inner = owner;
        context.state = decision.proposed_state();
        context.history = retained;
        Ok(())
    }

    fn supports_device_selector(&self) -> bool {
        self.matched_tokens.is_none() && self.context_shift.is_none()
    }

    fn prefill_with_device_selector(
        &mut self,
        input_token_ids: &[u32],
        selector: &sllm_core::DeviceTokenSelectorRequestV1,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        if self.matched_tokens.is_some() || self.context_shift.is_some() {
            return Err(GenerationServiceError::DeviceSelectorUnsupported);
        }
        let step = GenerationExecutorV1::prefill_with_device_selector(
            &mut self.inner,
            input_token_ids,
            selector,
        )?;
        self.publish_fresh_prefix(input_token_ids.len())?;
        self.save_checkpoint_after_prefill(input_token_ids.len())?;
        Ok(step)
    }

    fn decode_with_device_selector(
        &mut self,
        token_id: u32,
        selector: &sllm_core::DeviceTokenSelectorRequestV1,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        if self.context_shift.is_some() {
            return Err(GenerationServiceError::DeviceSelectorUnsupported);
        }
        let output =
            GenerationExecutorV1::decode_with_device_selector(&mut self.inner, token_id, selector)?;
        if let Some(context) = self.context_shift.as_mut() {
            context.history.push(
                i32::try_from(token_id).map_err(|_| GenerationServiceError::TokenIdOverflow)?,
            );
            context.state = context
                .state
                .after_append(1, context.capacity)
                .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        }
        Ok(output)
    }

    fn cancel(&mut self) {
        self.inner.cancel();
    }
}

impl SpeculativeGenerationExecutorV1 for QwenPrefixGenerationExecutorV1 {
    fn speculative_decode_greedy(
        &mut self,
        _pending_token: u32,
    ) -> Result<Vec<GenerationStepV1>, GenerationServiceError> {
        Err(GenerationServiceError::Execution(
            "ngram target verification requires an explicit proposal".to_owned(),
        ))
    }

    fn has_draft_provider(&self) -> bool {
        true
    }

    fn speculative_draft_width(&self) -> usize {
        self.draft_width
    }

    fn speculative_decode_with_proposal(
        &mut self,
        pending_token: u32,
        proposal: &DraftProposalV1,
    ) -> Result<Vec<GenerationStepV1>, GenerationServiceError> {
        if self.speculative_block_pending {
            return Err(GenerationServiceError::Execution(
                "previous ngram target block is still pending".to_owned(),
            ));
        }
        if proposal.token_ids().is_empty() || proposal.token_ids().len() > self.draft_width {
            return Err(GenerationServiceError::Execution(
                "ngram proposal width exceeds the target graph".to_owned(),
            ));
        }
        let mut input = Vec::with_capacity(proposal.token_ids().len() + 1);
        input.push(
            i32::try_from(pending_token).map_err(|_| GenerationServiceError::TokenIdOverflow)?,
        );
        input.extend(
            proposal
                .token_ids()
                .iter()
                .map(|&token| {
                    i32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow)
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        let output = self
            .inner
            .decode_block_with_mtp_state(&input)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        if output.token_ids().len() != input.len() {
            return Err(GenerationServiceError::MissingDeviceArgmax);
        }
        let accepted = proposal
            .token_ids()
            .iter()
            .zip(output.token_ids())
            .take_while(|(draft, target)| i32::try_from(**draft).ok() == Some(**target))
            .count();
        let committed_rows = accepted + 1;
        let steps = (0..committed_rows)
            .map(|row| qwen_step_from_output(&output, row))
            .collect::<Result<Vec<_>, _>>()?;
        self.speculative_block_pending = true;
        Ok(steps)
    }

    fn finalize_speculative_decode(
        &mut self,
        committed_input_rows: usize,
    ) -> Result<(), GenerationServiceError> {
        if !self.speculative_block_pending {
            return Err(GenerationServiceError::Execution(
                "no ngram target block is pending".to_owned(),
            ));
        }
        self.inner
            .resolve_decode_block(committed_input_rows)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        self.speculative_block_pending = false;
        Ok(())
    }
}

struct GemmaPrefixGenerationExecutorV1 {
    inner: sllm_core::Gemma4ExecutionRequest,
    matched_tokens: Option<Vec<u32>>,
    _lease: Option<PrefixLeaseV1>,
    published_prefix: Option<Gemma4PrefixStateV1>,
    publish_prefix: bool,
    context_shift: Option<GemmaContextShiftRuntimeV1>,
    checkpoint_save: Option<QwenCheckpointSaveV1>,
}

struct GemmaContextShiftRuntimeV1 {
    resident: Gemma4ResidentModel,
    policy: ContextPositionPolicyV1,
    state: ContextWindowStateV1,
    history: Vec<i32>,
    capacity: u64,
}

impl GemmaPrefixGenerationExecutorV1 {
    fn fresh(inner: sllm_core::Gemma4ExecutionRequest, publish_prefix: bool) -> Self {
        Self {
            inner,
            matched_tokens: None,
            _lease: None,
            published_prefix: None,
            publish_prefix,
            context_shift: None,
            checkpoint_save: None,
        }
    }

    fn from_hit(inner: sllm_core::Gemma4ExecutionRequest, hit: GemmaPrefixHitV1) -> Self {
        Self {
            inner,
            matched_tokens: Some(hit.matched_tokens),
            _lease: Some(hit.lease),
            published_prefix: None,
            publish_prefix: false,
            context_shift: None,
            checkpoint_save: None,
        }
    }

    fn with_context_shift(
        mut self,
        resident: Gemma4ResidentModel,
        policy: ContextPositionPolicyV1,
        capacity: u64,
    ) -> Self {
        self.context_shift = Some(GemmaContextShiftRuntimeV1 {
            resident,
            policy,
            state: ContextWindowStateV1::new(0, 0, 0),
            history: Vec::new(),
            capacity,
        });
        self
    }

    fn with_checkpoint_save(mut self, checkpoint: QwenCheckpointSaveV1) -> Self {
        self.checkpoint_save = Some(checkpoint);
        self
    }

    fn save_checkpoint_after_prefill(
        &mut self,
        prompt_token_count: usize,
    ) -> Result<(), GenerationServiceError> {
        let Some(checkpoint) = self.checkpoint_save.take() else {
            return Ok(());
        };
        if checkpoint.prompt_tokens.len() != prompt_token_count {
            checkpoint
                .status
                .store(CHECKPOINT_STATUS_FAILED, Ordering::Release);
            return Err(GenerationServiceError::Execution(
                "checkpoint prompt length changed before save".to_owned(),
            ));
        }
        let prompt_len = u64::try_from(prompt_token_count).map_err(|_| {
            checkpoint
                .status
                .store(CHECKPOINT_STATUS_FAILED, Ordering::Release);
            GenerationServiceError::CountOverflow
        })?;
        let result = self
            .inner
            .checkpoint(
                checkpoint.identity,
                &checkpoint.prompt_tokens,
                &[],
                &[],
                &[],
                &[],
                prompt_len,
                prompt_len,
                1,
            )
            .map_err(|_| GenerationServiceError::Execution("checkpoint capture failed".to_owned()))
            .and_then(|checkpoint_payload| {
                checkpoint
                    .store
                    .save(&checkpoint.name, &checkpoint_payload)
                    .map(|_| ())
                    .map_err(|_| {
                        GenerationServiceError::Execution("checkpoint save failed".to_owned())
                    })
            });
        checkpoint.status.store(
            if result.is_ok() {
                CHECKPOINT_STATUS_SUCCEEDED
            } else {
                CHECKPOINT_STATUS_FAILED
            },
            Ordering::Release,
        );
        result
    }

    fn inner(&self) -> &sllm_core::Gemma4ExecutionRequest {
        &self.inner
    }

    fn context_shift_count(&self) -> u64 {
        self.context_shift
            .as_ref()
            .map_or(0, |context| context.state.shift_count())
    }

    fn refresh_prefix_fork_audit(&self) -> Result<Gemma4PrefixForkAuditV1, GenerationServiceError> {
        self.inner
            .refresh_prefix_fork_audit()
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))
    }

    fn take_published_prefix(&mut self) -> Option<Gemma4PrefixStateV1> {
        self.published_prefix.take()
    }

    fn publish_fresh_prefix(
        &mut self,
        prompt_token_count: usize,
    ) -> Result<(), GenerationServiceError> {
        if self.publish_prefix && self.matched_tokens.is_none() {
            let prefix = self
                .inner
                .prefix_state()
                .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
            require_prompt_only_prefix(prefix.committed_length(), prompt_token_count)?;
            self.published_prefix = Some(prefix);
        }
        Ok(())
    }
}

impl GenerationExecutorV1 for GemmaPrefixGenerationExecutorV1 {
    fn prefill(
        &mut self,
        input_token_ids: &[u32],
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        if let Some(matched_tokens) = self.matched_tokens.as_deref() {
            if include_last_logits || !input_token_ids.starts_with(matched_tokens) {
                return Err(GenerationServiceError::Execution(
                    "Gemma prefix continuation requires exact greedy prompt semantics".to_owned(),
                ));
            }
            let suffix = input_token_ids[matched_tokens.len()..]
                .iter()
                .map(|&token| {
                    i32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let output = self
                .inner
                .continue_from_prefix(&suffix)
                .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
            return gemma_step_from_output(&output, output.token_ids().len().saturating_sub(1));
        }
        if let Some(context) = self.context_shift.as_mut() {
            let history = input_token_ids
                .iter()
                .map(|&token| {
                    i32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if history.len() as u64 > context.capacity {
                let decision = context
                    .policy
                    .plan_initial(history.len() as u64, context.capacity)
                    .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
                context
                    .policy
                    .validate_adapter(Gemma4ResidentModel::context_adapter_capabilities())
                    .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
                let retained_history = decision
                    .retained_token_ids(&history)
                    .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
                let (owner, output) = context
                    .resident
                    .new_request_from_context_shift(
                        decision,
                        ContextWindowStateV1::new(history.len() as u64, history.len() as u64, 0),
                        &history,
                        context.capacity,
                    )
                    .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
                self.inner = owner;
                context.state = decision.proposed_state();
                context.history = retained_history;
                return gemma_step_from_output(&output, output.token_ids().len().saturating_sub(1));
            }
            context.state =
                ContextWindowStateV1::new(history.len() as u64, history.len() as u64, 0);
            context.history = history;
        }
        let step =
            GenerationExecutorV1::prefill(&mut self.inner, input_token_ids, include_last_logits)?;
        self.publish_fresh_prefix(input_token_ids.len())?;
        self.save_checkpoint_after_prefill(input_token_ids.len())?;
        Ok(step)
    }

    fn decode(
        &mut self,
        token_id: u32,
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let output = GenerationExecutorV1::decode(&mut self.inner, token_id, include_last_logits)?;
        if let Some(context) = self.context_shift.as_mut() {
            context.history.push(
                i32::try_from(token_id).map_err(|_| GenerationServiceError::TokenIdOverflow)?,
            );
            context.state = context
                .state
                .after_append(1, context.capacity)
                .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        }
        Ok(output)
    }

    fn before_decode(&mut self, _token_id: u32) -> Result<(), GenerationServiceError> {
        let Some(context) = self.context_shift.as_mut() else {
            return Ok(());
        };
        let decision = context
            .policy
            .plan(context.state, context.capacity, 1)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        if !decision.requires_shift() {
            return Ok(());
        }
        context
            .policy
            .validate_adapter(Gemma4ResidentModel::context_adapter_capabilities())
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let retained = decision
            .retained_token_ids(&context.history)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let (owner, _output) = context
            .resident
            .new_request_from_context_shift(
                decision,
                context.state,
                &context.history,
                context.capacity,
            )
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        self.inner = owner;
        context.state = decision.proposed_state();
        context.history = retained;
        Ok(())
    }

    fn supports_device_selector(&self) -> bool {
        self.matched_tokens.is_none() && self.context_shift.is_none()
    }

    fn prefill_with_device_selector(
        &mut self,
        input_token_ids: &[u32],
        selector: &sllm_core::DeviceTokenSelectorRequestV1,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        if self.matched_tokens.is_some() || self.context_shift.is_some() {
            return Err(GenerationServiceError::DeviceSelectorUnsupported);
        }
        let step = GenerationExecutorV1::prefill_with_device_selector(
            &mut self.inner,
            input_token_ids,
            selector,
        )?;
        self.publish_fresh_prefix(input_token_ids.len())?;
        self.save_checkpoint_after_prefill(input_token_ids.len())?;
        Ok(step)
    }

    fn decode_with_device_selector(
        &mut self,
        token_id: u32,
        selector: &sllm_core::DeviceTokenSelectorRequestV1,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        if self.context_shift.is_some() {
            return Err(GenerationServiceError::DeviceSelectorUnsupported);
        }
        let output =
            GenerationExecutorV1::decode_with_device_selector(&mut self.inner, token_id, selector)?;
        if let Some(context) = self.context_shift.as_mut() {
            context.history.push(
                i32::try_from(token_id).map_err(|_| GenerationServiceError::TokenIdOverflow)?,
            );
            context.state = context
                .state
                .after_append(1, context.capacity)
                .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        }
        Ok(output)
    }

    fn cancel(&mut self) {
        self.inner.cancel();
    }
}

fn qwen_step_from_output(
    output: &sllm_core::QwenExecutionOutput,
    row: usize,
) -> Result<GenerationStepV1, GenerationServiceError> {
    let token = *output
        .token_ids()
        .get(row)
        .ok_or(GenerationServiceError::MissingDeviceArgmax)?;
    Ok(GenerationStepV1::new(
        u32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow)?,
        output.last_logits().map(<[f32]>::to_vec),
    ))
}

fn gemma_step_from_output(
    output: &sllm_core::Gemma4ExecutionOutput,
    row: usize,
) -> Result<GenerationStepV1, GenerationServiceError> {
    let token = *output
        .token_ids()
        .get(row)
        .ok_or(GenerationServiceError::MissingDeviceArgmax)?;
    Ok(GenerationStepV1::new(
        u32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow)?,
        output.last_logits().map(<[f32]>::to_vec),
    ))
}

struct QwenMultimodalExecutorV1<'a> {
    inner: &'a mut QwenExecutionRequest,
    prompt: &'a QwenMultimodalPrompt,
    prefilled: bool,
}

impl GenerationExecutorV1 for QwenMultimodalExecutorV1<'_> {
    fn prefill(
        &mut self,
        input_token_ids: &[u32],
        _include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        if self.prefilled {
            return Err(GenerationServiceError::Execution(
                "multimodal prefill was requested twice".to_owned(),
            ));
        }
        let token_ids = input_token_ids
            .iter()
            .map(|token| i32::try_from(*token).map_err(|_| GenerationServiceError::TokenIdOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        let output = self
            .inner
            .prefill_multimodal_with_last_logits(
                &token_ids,
                &self.prompt.embeddings_bf16,
                &self.prompt.positions,
            )
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        self.prefilled = true;
        let device_argmax = output
            .token_ids()
            .last()
            .copied()
            .ok_or(GenerationServiceError::MissingDeviceArgmax)
            .and_then(|token| {
                u32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow)
            })?;
        Ok(GenerationStepV1::new(
            device_argmax,
            output.last_logits().map(ToOwned::to_owned),
        ))
    }

    fn decode(
        &mut self,
        token_id: u32,
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let token = i32::try_from(token_id).map_err(|_| GenerationServiceError::TokenIdOverflow)?;
        let output = if include_last_logits {
            self.inner.decode_with_last_logits(token)
        } else {
            self.inner.decode(token)
        }
        .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let device_argmax = output
            .token_ids()
            .last()
            .copied()
            .ok_or(GenerationServiceError::MissingDeviceArgmax)
            .and_then(|token| {
                u32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow)
            })?;
        Ok(GenerationStepV1::new(
            device_argmax,
            output.last_logits().map(ToOwned::to_owned),
        ))
    }

    fn cancel(&mut self) {
        self.inner.cancel();
    }
}

impl GenerationOutputSinkV1 for OutputSinkAdapterV1<'_> {
    fn publish(&mut self, delta: &str) -> Result<(), GenerationServiceError> {
        self.inner
            .publish(delta)
            .map_err(|error| GenerationServiceError::Output(error.to_string()))
    }

    fn publish_reasoning(&mut self, delta: &str) -> Result<(), GenerationServiceError> {
        // The runtime's existing response splitter consumes the raw stream
        // and routes this channel into reasoning_content.  Keep visible
        // output on `publish` so generic frontend sinks remain fail-closed.
        self.inner
            .publish(delta)
            .map_err(|error| GenerationServiceError::Output(error.to_string()))
    }
}

fn require_clean_request_memory(
    snapshot: AllocationSnapshot,
    boundary: &str,
) -> Result<(), BackendErrorV1> {
    if snapshot.poisoned()
        || snapshot.request_state().current_bytes() != 0
        || snapshot.workspace().current_bytes() != 0
    {
        return Err(BackendErrorV1::new(format!(
            "{boundary} has nonzero or poisoned request allocation accounting"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChatMessageV1;
    use sllm_core::CheckpointPayload;

    fn checkpoint_test_directory(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "sllm-phase41-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn lifecycle_checkpoint(tokens: &[u32], conversation: &[u8]) -> SessionCheckpoint {
        let identity = CheckpointIdentity::for_tokens(
            format!("sha256:{}", "0".repeat(64)),
            "artifact",
            "adapter",
            "renderer",
            "tokenizer",
            "gfx1201",
            "plan",
            tokens,
            KvCacheEncoding::Fp16,
            [0; 32],
            [0; 32],
        )
        .unwrap();
        let payload = CheckpointPayload {
            token_history: tokens.to_vec(),
            conversation: conversation.to_vec(),
            ..CheckpointPayload::default()
        };
        SessionCheckpoint::new(
            identity,
            tokens.len() as u64,
            tokens.len() as u64,
            1,
            payload,
        )
        .unwrap()
    }

    #[test]
    fn phase41_defaults_are_disabled_and_valid() {
        let config = Phase41ProductionConfigV1::default();
        assert_eq!(config.prefix_cache, PrefixCacheStartupConfigV1::Disabled);
        assert_eq!(
            config.context_window,
            ContextWindowStartupConfigV1::Disabled
        );
        assert_eq!(config.checkpoint, CheckpointStartupConfigV1::Disabled);
        assert_eq!(config.draft, DraftStartupConfigV1::Disabled);
        config.validate_startup().unwrap();
        validate_qwen_phase41_operational_config(&config).unwrap();
        validate_gemma_phase41_operational_config(&config).unwrap();
        assert_eq!(
            QwenPrefixCacheRuntimeV1::new(&config.prefix_cache)
                .unwrap()
                .baseline_bytes()
                .unwrap(),
            0
        );
        assert_eq!(
            GemmaPrefixCacheRuntimeV1::new(&config.prefix_cache)
                .unwrap()
                .baseline_bytes()
                .unwrap(),
            0
        );
    }

    #[test]
    fn qwen_adapter_catalog_requires_strict_sorted_aliases() {
        let artifact = |alias: &str| QwenAdapterArtifactConfigV1 {
            alias: alias.to_owned(),
            lock_path: PathBuf::from("lock.json"),
            payload_path: PathBuf::from("payload.bin"),
        };
        let unsorted = QwenAdapterCatalogConfigV1 {
            lora: vec![artifact("zeta"), artifact("alpha")],
            control_vectors: Vec::new(),
        };
        assert!(validate_qwen_adapter_catalog_config(&unsorted).is_err());
        let duplicate_across_kinds = QwenAdapterCatalogConfigV1 {
            lora: vec![artifact("same")],
            control_vectors: vec![artifact("same")],
        };
        assert!(validate_qwen_adapter_catalog_config(&duplicate_across_kinds).is_err());
    }

    #[test]
    fn qwen_adapter_resolver_is_disabled_or_fails_closed_without_catalog() {
        let empty = crate::api::ModelVariantRequestV1::default();
        assert_eq!(
            resolve_qwen_adapters(None, &empty).unwrap().identity(),
            "adapter:none-v1"
        );
        let selection = crate::api::ArtifactScaleSelectionV1::new("missing".to_owned(), 1.0)
            .expect("bounded test selection");
        let request = crate::api::ModelVariantRequestV1::from_parts(vec![selection], Vec::new())
            .expect("sorted test selection");
        assert!(resolve_qwen_adapters(None, &request).is_err());
        assert_eq!(
            QwenAdapterCatalogV1 {
                lora: BTreeMap::new(),
                control_vectors: BTreeMap::new(),
            }
            .identity(),
            "adapter:none-v1"
        );
        assert_eq!(
            qwen_adapter_catalog_identity_preflight(&QwenAdapterCatalogConfigV1::default())
                .unwrap(),
            "adapter:none-v1"
        );
    }

    #[test]
    fn persistent_checkpoint_lifecycle_supports_fresh_promote_and_pending_rollback() {
        let mut state = PersistentCheckpointStateV1::default();
        let first = lifecycle_checkpoint(&[1, 2], b"first");
        state.stage(first.clone()).unwrap();
        assert!(state.current.is_none());
        let saved = state.candidate_with_conversation(b"saved").unwrap();
        assert_eq!(saved.payload.conversation, b"saved");
        assert!(state.current.is_none());
        state.promote(saved.clone());
        assert_eq!(state.current, Some(saved));
        assert!(state.pending.is_none());

        let second = lifecycle_checkpoint(&[1, 2, 3], b"second");
        state.stage(second).unwrap();
        let previous = state.current.clone();
        state.pending = None;
        assert_eq!(state.current, previous);
    }

    #[test]
    fn persistent_checkpoint_candidate_failure_keeps_current_and_pending() {
        let mut state = PersistentCheckpointStateV1::default();
        let current = lifecycle_checkpoint(&[1], b"current");
        let pending = lifecycle_checkpoint(&[1, 2], b"pending");
        state.current = Some(current.clone());
        state.stage(pending.clone()).unwrap();
        let oversized = vec![0_u8; 16 * 1024 * 1024 + 1];
        assert!(state.candidate_with_conversation(&oversized).is_err());
        assert_eq!(state.current, Some(current));
        assert_eq!(state.pending, Some(pending));
        assert!(state.stage(lifecycle_checkpoint(&[3], b"blocked")).is_err());
    }

    #[test]
    fn persistent_reverse_prompt_finish_reason_is_distinct_from_stop() {
        let usage = TokenUsageV1::new(4, 3).unwrap();
        let completion = BackendCompletionV1 {
            finish_reason: FinishReasonV1::Stop,
            usage,
            matched_stop: Some("User:".to_owned()),
        };
        assert_eq!(
            persistent_chat_finish_reason(&completion, &["User:".to_owned()]),
            QwenPersistentChatFinishReasonV1::ReversePrompt
        );
        assert_eq!(
            persistent_chat_finish_reason(&completion, &["Other:".to_owned()]),
            QwenPersistentChatFinishReasonV1::Stop
        );
        assert_eq!(
            QwenPersistentChatFinishReasonV1::ReversePrompt.as_str(),
            "reverse_prompt"
        );
    }

    #[test]
    fn qwen_reasoning_policy_allows_only_reviewed_protocol_raw_text() {
        let completion = crate::phase42_api::parse_completion_request(
            br#"{"model":"qwen","prompt":"ordinary raw"}"#,
        )
        .unwrap();
        let ordinary = ChatCompletionRequestV1::from_completion(
            &completion,
            GenerationRequestInputV1::RawText("ordinary raw".to_owned()),
        )
        .unwrap();
        assert!(qwen_reasoning_policy_for_request(&ordinary, &[7]).is_err());

        let reviewed = ChatCompletionRequestV1::from_protocol_text(
            "qwen".to_owned(),
            "reviewed protocol envelope".to_owned(),
            None,
            8,
            0.0,
            1.0,
            Vec::new(),
            false,
            false,
            true,
            None,
            None,
        )
        .unwrap();
        assert!(qwen_reasoning_policy_for_request(&reviewed, &[7]).is_ok());
    }

    #[test]
    fn operational_draft_validation_fails_closed_without_silent_fallback() {
        let external = Phase41ProductionConfigV1 {
            draft: DraftStartupConfigV1::External {
                model_identity: "draft-model-v1".to_owned(),
                tokenizer_identity: "tokenizer-v1".to_owned(),
                vocabulary_size: 1024,
                width: 2,
            },
            ..Phase41ProductionConfigV1::default()
        };
        external.validate_startup().unwrap();
        for error in [
            validate_qwen_phase41_operational_config(&external).unwrap_err(),
            validate_gemma_phase41_operational_config(&external).unwrap_err(),
        ] {
            assert!(
                error
                    .to_string()
                    .contains("independently provisioned executor")
            );
        }

        let qwen_prefix_mtp = Phase41ProductionConfigV1 {
            prefix_cache: PrefixCacheStartupConfigV1::Enabled {
                max_entries: 1,
                max_logical_tokens: 1,
                max_resident_bytes: 1,
            },
            draft: DraftStartupConfigV1::MtpAuto,
            ..Phase41ProductionConfigV1::default()
        };
        qwen_prefix_mtp.validate_startup().unwrap();
        assert!(
            validate_qwen_phase41_operational_config(&qwen_prefix_mtp)
                .unwrap_err()
                .to_string()
                .contains("cannot be combined")
        );

        for draft in [
            DraftStartupConfigV1::MtpAuto,
            DraftStartupConfigV1::Ngram { order: 2, width: 2 },
        ] {
            let config = Phase41ProductionConfigV1 {
                draft,
                ..Phase41ProductionConfigV1::default()
            };
            config.validate_startup().unwrap();
            assert!(validate_gemma_phase41_operational_config(&config).is_err());
        }
    }

    #[test]
    fn qwen_checkpoint_rejects_other_phase41_state_owners() {
        let checkpoint = CheckpointStartupConfigV1::Enabled {
            directory: PathBuf::from("checkpoints"),
            quota_bytes: 1024,
            load_name: None,
            save_name: Some("prompt".to_owned()),
        };
        for config in [
            Phase41ProductionConfigV1 {
                checkpoint: checkpoint.clone(),
                prefix_cache: PrefixCacheStartupConfigV1::Enabled {
                    max_entries: 1,
                    max_logical_tokens: 1,
                    max_resident_bytes: 1,
                },
                ..Phase41ProductionConfigV1::default()
            },
            Phase41ProductionConfigV1 {
                checkpoint: checkpoint.clone(),
                context_window: ContextWindowStartupConfigV1::KeepPrefixRecentV1 {
                    keep_prefix: 1,
                    keep_recent: 1,
                },
                ..Phase41ProductionConfigV1::default()
            },
            Phase41ProductionConfigV1 {
                checkpoint: checkpoint.clone(),
                draft: DraftStartupConfigV1::Ngram { order: 2, width: 1 },
                ..Phase41ProductionConfigV1::default()
            },
        ] {
            assert!(validate_qwen_phase41_operational_config(&config).is_err());
            assert!(validate_gemma_phase41_operational_config(&config).is_err());
        }
        let enabled = Phase41ProductionConfigV1 {
            checkpoint,
            ..Phase41ProductionConfigV1::default()
        };
        assert!(validate_qwen_phase41_operational_config(&enabled).is_ok());
        assert!(validate_gemma_phase41_operational_config(&enabled).is_ok());
    }

    #[test]
    fn checkpoint_load_and_save_names_are_exclusive() {
        let config = CheckpointStartupConfigV1::Enabled {
            directory: PathBuf::from("checkpoints"),
            quota_bytes: 1024,
            load_name: Some("load".to_owned()),
            save_name: Some("save".to_owned()),
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn enabled_prefix_runtime_starts_with_zero_owned_baseline() {
        let config = PrefixCacheStartupConfigV1::Enabled {
            max_entries: 1,
            max_logical_tokens: 1,
            max_resident_bytes: 1,
        };
        assert_eq!(
            QwenPrefixCacheRuntimeV1::new(&config)
                .unwrap()
                .baseline_bytes()
                .unwrap(),
            0
        );
        assert_eq!(
            GemmaPrefixCacheRuntimeV1::new(&config)
                .unwrap()
                .baseline_bytes()
                .unwrap(),
            0
        );
        assert_eq!(
            checked_prefix_request_state_baseline([0, 17, 23]).unwrap(),
            40
        );
        assert!(checked_prefix_request_state_baseline([u64::MAX, 1]).is_err());
    }

    #[test]
    fn prefix_lookup_kinds_map_to_bounded_production_audit_values() {
        assert_eq!(
            production_prefix_kind(PrefixLookupKind::Miss),
            ProductionPrefixCacheResultV1::Miss
        );
        assert_eq!(
            production_prefix_kind(PrefixLookupKind::PartialHit),
            ProductionPrefixCacheResultV1::PartialHit
        );
        assert_eq!(
            production_prefix_kind(PrefixLookupKind::ExactHit),
            ProductionPrefixCacheResultV1::ExactHit
        );
    }

    #[test]
    fn production_ngram_order_is_a_longest_suffix_search_bound() {
        let provider = production_ngram_provider(7).unwrap();
        assert_eq!(provider.min_order(), 1);
        assert_eq!(provider.max_order(), 7);
    }

    #[test]
    fn fresh_prefix_publication_accepts_prompt_only_state() {
        require_prompt_only_prefix(3, 3).unwrap();
        for committed_length in [2, 4] {
            assert!(require_prompt_only_prefix(committed_length, 3).is_err());
        }
    }

    #[test]
    fn phase41_prefix_and_draft_limits_check_both_boundaries() {
        for max_entries in [1, MAX_PREFIX_CACHE_ENTRIES_V1] {
            PrefixCacheStartupConfigV1::Enabled {
                max_entries,
                max_logical_tokens: MAX_PREFIX_CACHE_LOGICAL_TOKENS_V1,
                max_resident_bytes: 1,
            }
            .validate()
            .unwrap();
        }
        for max_entries in [0, MAX_PREFIX_CACHE_ENTRIES_V1 + 1] {
            assert!(
                PrefixCacheStartupConfigV1::Enabled {
                    max_entries,
                    max_logical_tokens: 1,
                    max_resident_bytes: 1,
                }
                .validate()
                .is_err()
            );
        }
        for order in [1, MAX_DRAFT_NGRAM_ORDER_V1] {
            for width in [1, MAX_DRAFT_WIDTH_V1] {
                DraftStartupConfigV1::Ngram { order, width }
                    .validate()
                    .unwrap();
            }
        }
        for (order, width) in [
            (0, 1),
            (MAX_DRAFT_NGRAM_ORDER_V1 + 1, 1),
            (1, 0),
            (1, MAX_DRAFT_WIDTH_V1 + 1),
        ] {
            assert!(
                DraftStartupConfigV1::Ngram { order, width }
                    .validate()
                    .is_err()
            );
        }
    }

    #[test]
    fn context_policy_rejects_empty_and_overflowing_retention() {
        ContextWindowStartupConfigV1::KeepPrefixRecentV1 {
            keep_prefix: 0,
            keep_recent: 1,
        }
        .validate()
        .unwrap();
        assert!(
            ContextWindowStartupConfigV1::KeepPrefixRecentV1 {
                keep_prefix: 0,
                keep_recent: 0,
            }
            .validate()
            .is_err()
        );
        assert!(
            ContextWindowStartupConfigV1::KeepPrefixRecentV1 {
                keep_prefix: u64::MAX,
                keep_recent: 1,
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn checkpoint_load_seam_fails_closed_without_disclosing_the_target() {
        let directory = checkpoint_test_directory("load");
        std::fs::create_dir(&directory).unwrap();
        let config = CheckpointStartupConfigV1::Enabled {
            directory: directory.clone(),
            quota_bytes: 1,
            load_name: Some("missing-load".to_owned()),
            save_name: None,
        };
        let error = config
            .validate_startup_load_exists()
            .unwrap_err()
            .to_string();
        assert_eq!(error, "configured checkpoint load target is unavailable");
        assert!(!error.contains(directory.to_string_lossy().as_ref()));
        assert!(!error.contains("missing-load"));

        std::fs::write(directory.join("missing-load.ckpt"), b"typed seam only").unwrap();
        config.validate_startup_load_exists().unwrap();
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn checkpoint_names_are_bounded_and_cannot_escape_the_directory() {
        for name in [
            "",
            ".",
            "..",
            "../escape",
            "contains/slash",
            "contains space",
        ] {
            assert!(
                CheckpointStartupConfigV1::Enabled {
                    directory: PathBuf::from("checkpoints"),
                    quota_bytes: 1,
                    load_name: None,
                    save_name: Some(name.to_owned()),
                }
                .validate()
                .is_err(),
                "accepted invalid name {name:?}"
            );
        }
        CheckpointStartupConfigV1::Enabled {
            directory: PathBuf::from("checkpoints"),
            quota_bytes: 1,
            load_name: None,
            save_name: Some("safe_NAME-1.0".to_owned()),
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn phase41_audit_is_bounded_and_redacted() {
        let json = serde_json::to_value(ProductionPhase41AuditV1 {
            prefix_cache_result: Some(ProductionPrefixCacheResultV1::PartialHit),
            prefix_shared_pages: 3,
            prefix_cow_pages: 1,
            prefix_copied_bytes: 4096,
            checkpoint_operation: Some(ProductionCheckpointOperationV1::Load),
            checkpoint_result: Some(ProductionCheckpointResultV1::Succeeded),
            context_shift_count: 2,
            draft_provider: Some(ProductionDraftProviderV1::External),
            draft_proposed_tokens: 7,
            draft_accepted_tokens: 5,
            draft_rejected_tokens: 2,
        })
        .unwrap();
        assert_eq!(json["prefix_cache_result"], "partial-hit");
        assert_eq!(json["checkpoint_operation"], "load");
        assert_eq!(json["checkpoint_result"], "succeeded");
        assert_eq!(json["draft_provider"], "external");
        let serialized = serde_json::to_string(&json).unwrap();
        for forbidden in [
            "path",
            "directory",
            "identity",
            "checkpoint_name",
            "token_id",
        ] {
            assert!(!serialized.contains(forbidden), "audit exposed {forbidden}");
        }
    }

    #[test]
    fn embedded_gguf_fp8_provider_accepts_native_and_fnuz_targets() {
        assert_eq!(select_gguf_fp8_provider("gfx1201").unwrap(), "gguf-native");
        assert_eq!(select_gguf_fp8_provider("gfx942").unwrap(), "native-fnuz");
        assert_eq!(
            gguf_fp8_dtype(select_gguf_fp8_provider("gfx1201").unwrap()),
            sllm_core::DType::F8E4M3Fn
        );
        assert_eq!(
            gguf_fp8_dtype(select_gguf_fp8_provider("gfx942").unwrap()),
            sllm_core::DType::F8E4M3FnuZ
        );
    }

    #[test]
    fn embedded_gguf_fp8_provider_rejects_unsupported_exact_targets() {
        for target in ["gfx1030", "gfx1200", "gfx942:sramecc+:xnack-", "unknown"] {
            let error = select_gguf_fp8_provider(target).unwrap_err();
            assert!(error.contains(target), "{error}");
            assert!(error.contains("native-fnuz"), "{error}");
        }
    }

    fn message(inner: Qwen35ChatMessageV1) -> ChatMessageV1 {
        let content = match &inner {
            Qwen35ChatMessageV1::System { content }
            | Qwen35ChatMessageV1::User { content }
            | Qwen35ChatMessageV1::Assistant { content, .. } => content.clone(),
        };
        ChatMessageV1 {
            inner,
            parts: vec![crate::api::ChatContentPartV1::Text(content)],
        }
    }

    #[test]
    fn gemma_raw_transcript_is_versioned_by_exact_roles_and_unicode() {
        let rendered = render_gemma4_raw_messages(&[
            message(Qwen35ChatMessageV1::system("方針")),
            message(Qwen35ChatMessageV1::user("こんにちは🌙")),
            message(Qwen35ChatMessageV1::assistant("了解", None)),
            message(Qwen35ChatMessageV1::user("続けて")),
        ])
        .unwrap();
        assert_eq!(
            rendered,
            "System: 方針\nUser: こんにちは🌙\nAssistant: 了解\nUser: 続けて\nAssistant:"
        );
        assert!(
            render_gemma4_raw_messages(&[message(Qwen35ChatMessageV1::assistant(
                "visible",
                Some("hidden".to_owned()),
            ))])
            .is_err()
        );
    }

    #[test]
    fn gemma_raw_transcript_checks_both_sides_of_the_byte_cap() {
        let overhead = "User: \nAssistant:".len();
        let accepted = "x".repeat(GEMMA4_RAW_CHAT_MAX_BYTES - overhead);
        assert_eq!(
            render_gemma4_raw_messages(&[message(Qwen35ChatMessageV1::user(accepted))])
                .unwrap()
                .len(),
            GEMMA4_RAW_CHAT_MAX_BYTES
        );
        let rejected = "x".repeat(GEMMA4_RAW_CHAT_MAX_BYTES - overhead + 1);
        assert!(
            render_gemma4_raw_messages(&[message(Qwen35ChatMessageV1::user(rejected))]).is_err()
        );
    }
}
