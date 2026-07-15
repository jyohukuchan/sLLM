use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::process::{Command, ExitCode};
use std::time::Instant;
use ullm_engine::aq4_package_runtime::{
    PackageAq4ResidentMatvec, SQ8_0_MODEL_ARCH_QWEN_FAMILY, SqFp8ProjectionDispatches,
    SqFp8ProjectionTelemetry, reset_sq_fp8_projection_telemetry,
    snapshot_sq_fp8_projection_telemetry,
};
#[cfg(test)]
use ullm_engine::aq4_package_runtime::{
    SqFp8ProjectionDispatch, dispatchable_sq8_projection_gpu_name,
    select_sq_fp8_projection_implementation_id,
};
use ullm_engine::backend_dispatch::{
    BackendImplementation, BackendRequest, Sq8ProjectionFamily, select_backend,
};
#[cfg(test)]
use ullm_engine::backend_dispatch::{
    Sq8ProjectionMatvecOperation as SqFp8ProjectionMatvecOperation,
    sq8_0_projection_descriptor_family,
};
use ullm_engine::decode_runner::{
    Qwen3DecoderLayerDecodeBatchInput, Qwen3DecoderLayerDecodeInputLayout,
    Qwen3DecoderLayerDecodeSequenceView, Qwen3DecoderLayerRequestDecodeRunner,
    Qwen3DecoderLayerStackRequestDecodeRunner, Qwen3SelfAttnDecodeBatchInput,
    Qwen3SelfAttnRequestDecodeRunner, qwen3_decoder_layer_decode_batch_inputs_from_sequences,
    qwen3_decoder_layer_prefill_input_from_sequence,
};
use ullm_engine::decoder::{
    PagedDecodeShape, PagedKvCacheReadback, Qwen3DecoderLayerRuntimeWeights,
    Qwen3DecoderLayerSequenceOutput, Qwen3MlpRuntimeWeights, Qwen3PostAttentionRuntimeWeights,
    Qwen3SelfAttnRuntimePreparedSequence, Qwen3SelfAttnRuntimePreparedSequenceForPagedDecode,
    Qwen3SelfAttnRuntimeShape, Qwen3SelfAttnRuntimeWeights, pack_paged_kv_cache_for_block_table,
    qwen3_causal_attn_to_host_f32, qwen3_decoder_layer_sequence_to_host_f32,
    qwen3_headwise_rmsnorm_to_host_f32, qwen3_rope_to_host_f32,
    qwen3_self_attn_block_sequence_to_host_f32,
    qwen3_self_attn_prepare_sequence_for_paged_decode_f32, qwen3_self_attn_runtime_shape,
    split_qwen3_self_attn_q_projection,
};
use ullm_engine::format_id::{FORMAT_AQ4_0, FORMAT_SQ8_0};
use ullm_engine::golden::{GoldenTensorFixture, compare_f32_slices};
use ullm_engine::host_bytes::{decode_f32_le_values, encode_f32_to_bytes, encode_u32_to_bytes};
use ullm_engine::loader::{
    LoadOptions, LoadedPayload, PassthroughF32Data, WeightRegistry,
    effective_rmsnorm_weight_values, load_package_tensor_prefix, materialize_config,
    materialize_selected_aq4_matrix, matrix_shape_rows_cols, read_named_passthrough_f32,
    read_named_passthrough_f32_rows,
    read_passthrough_payload_f32_bytes, resolve_passthrough_dtype,
    validate_passthrough_shape_elements,
};
use ullm_engine::package::{
    ReferencedFile, ReferencedFileRole, TensorSelector, list_tensor_payload_bundles,
    select_tensor_payload_bundle,
};
use ullm_engine::qwen3_loader::{
    Qwen3PackageModelDecodePlan, Qwen3PackageModelRuntime, Qwen3PackageModelStackRequest,
    Qwen3PackageSqOverlay, qwen3_decoder_layer_runtime_weights_from_package,
    qwen3_package_decoder_layer_runtime_from_package,
    qwen3_package_decoder_layer_runtime_from_package_with_sq_overlay,
    qwen3_package_model_run_prefill_batch_from_sequences,
    qwen3_package_model_run_ready_batch_from_sequences, qwen3_package_model_stack_runner,
    qwen3_self_attn_runtime_weights_from_package,
};
use ullm_engine::qwen35_aq4_layer_runtime::{
    MixedRequestStateBatchStepItem, PackageLinearAttnComponentStepMs,
    PackageLinearAttnResidentStepBatchLayer, PackageLinearAttnResidentStepLayer,
    PackageMixedRequestStateLayer, PackageSelfAttnComponentStepMs,
    PackageSelfAttnResidentStepBatchLayer, PackageSelfAttnResidentStepLayer, format_f32_preview,
    reset_sq_diagnostic_host_staging_telemetry, runtime_host_linear_attn_gate_beta_f32,
    runtime_host_linear_attn_recurrent_f32, snapshot_sq_diagnostic_host_staging_telemetry,
};
use ullm_engine::qwen35_aq4_head_runtime::{
    PackageEmbeddingRuntime, PackageFinalNormRuntime, PackageLmHeadMode, PackageLmHeadRuntime,
    PackageTokenLogit, QWEN3_EMBED_TOKENS_TENSOR, QWEN3_FINAL_NORM_TENSOR,
    QWEN3_LM_HEAD_TENSOR, copy_f32_values_to_runtime_buffer_chunked,
    package_embedding_shape, package_lm_head_top_k_from_rows,
};
use ullm_engine::qwen35_package_contract::{
    PackageDecoderLayerKind, matched_stop_token_id, matched_stop_token_sequence,
    package_decoder_layer_kind, package_layer_entries_are_contiguous,
    package_layer_entries_for_indices,
    package_self_attention_layer_indices, parse_stop_token_ids, parse_stop_token_sequences,
    select_package_layer_indices,
};
#[cfg(test)]
use ullm_engine::qwen35_package_contract::{
    PackageManifestLayerEntry, package_manifest_layer_entries,
};
use ullm_engine::scheduler::{
    KvBlockAllocator, KvBlockAllocatorStats, Request, RequestId, SchedulerDecodeRequest,
    SchedulerState,
};
use ullm_engine::sq::{
    materialize_sq_fp8_tensor_rows_to_runtime_f32, read_sq_fp8_artifact,
    select_sq_fp8_tensor_index, sq_fp8_tensor_rows_cols,
};


fn sq_fp8_projection_boundary(telemetry: SqFp8ProjectionTelemetry) -> String {
    let mut boundaries = Vec::new();
    if telemetry.single_matvec_count > 0 {
        boundaries.push("single");
    }
    if telemetry.batch_matvec_count > 0 {
        boundaries.push("batch");
    }
    if telemetry.pair_matvec_count > 0 {
        boundaries.push("pair");
    }
    if telemetry.triple_matvec_count > 0 {
        boundaries.push("triple");
    }
    if boundaries.is_empty() {
        "none".to_string()
    } else {
        boundaries.join("+")
    }
}


fn sq_fp8_projection_implementation_ids(
    telemetry: SqFp8ProjectionTelemetry,
    dispatches: SqFp8ProjectionDispatches,
) -> String {
    let mut selected = Vec::new();
    if telemetry.single_matvec_count > 0 {
        selected.push(format!(
            "{}={}",
            dispatches.single.label(),
            dispatches.single.implementation_id
        ));
    }
    if telemetry.batch_matvec_count > 0 {
        selected.push(format!(
            "{}={}",
            dispatches.batch.label(),
            dispatches.batch.implementation_id
        ));
    }
    if telemetry.pair_matvec_count > 0 {
        selected.push(format!(
            "{}={}",
            dispatches.pair.label(),
            dispatches.pair.implementation_id
        ));
    }
    if telemetry.triple_matvec_count > 0 {
        selected.push(format!(
            "{}={}",
            dispatches.triple.label(),
            dispatches.triple.implementation_id
        ));
    }
    if selected.is_empty() {
        "none".to_string()
    } else {
        selected.join(",")
    }
}

fn sq_fp8_projection_kernel_families(
    telemetry: SqFp8ProjectionTelemetry,
    dispatches: SqFp8ProjectionDispatches,
) -> String {
    let mut selected = Vec::new();
    if telemetry.single_matvec_count > 0 {
        selected.push(format!(
            "{}={}",
            dispatches.single.label(),
            dispatches
                .single
                .family
                .map(Sq8ProjectionFamily::id)
                .unwrap_or("none")
        ));
    }
    if telemetry.batch_matvec_count > 0 {
        selected.push(format!(
            "{}={}",
            dispatches.batch.label(),
            dispatches
                .batch
                .family
                .map(Sq8ProjectionFamily::id)
                .unwrap_or("none")
        ));
    }
    if telemetry.pair_matvec_count > 0 {
        selected.push(format!(
            "{}={}",
            dispatches.pair.label(),
            dispatches.pair.family.map(Sq8ProjectionFamily::id).unwrap_or("none")
        ));
    }
    if telemetry.triple_matvec_count > 0 {
        selected.push(format!(
            "{}={}",
            dispatches.triple.label(),
            dispatches
                .triple
                .family
                .map(Sq8ProjectionFamily::id)
                .unwrap_or("none")
        ));
    }
    if selected.is_empty() {
        "none".to_string()
    } else {
        selected.join(",")
    }
}

#[cfg(test)]
const RDNA4_GFX1201_R9700_NAME: &str = "AMD Radeon Graphics";
#[cfg(test)]
const RDNA4_GFX1201_R9700_CANONICAL_NAME: &str = "Radeon_AI_PRO_R9700";


fn main() -> ExitCode {
    match env::args().nth(1).as_deref() {
        Some("inspect-devices") => inspect_devices(),
        Some("runtime-smoke") => runtime_smoke(),
        Some("runtime-memory-smoke") => runtime_memory_smoke(env::args().nth(2)),
        Some("runtime-stream-smoke") => runtime_stream_smoke(env::args().nth(2)),
        Some("runtime-copy-smoke") => runtime_copy_smoke(env::args().nth(2)),
        Some("runtime-rmsnorm-smoke") => runtime_rmsnorm_smoke(env::args().nth(2)),
        Some("runtime-silu-mul-smoke") => runtime_silu_mul_smoke(env::args().nth(2)),
        Some("runtime-sigmoid-mul-smoke") => runtime_sigmoid_mul_smoke(env::args().nth(2)),
        Some("runtime-add-smoke") => runtime_add_smoke(env::args().nth(2)),
        Some("runtime-rope-smoke") => runtime_rope_smoke(env::args().nth(2)),
        Some("runtime-causal-attn-smoke") => runtime_causal_attn_smoke(env::args().nth(2)),
        Some("runtime-causal-attn-batch-smoke") => runtime_causal_attn_batch_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
            env::args().nth(10),
        ),
        Some("runtime-decode-attn-smoke") => runtime_decode_attn_smoke(env::args().nth(2)),
        Some("runtime-cached-prefix-attn-smoke") => runtime_cached_prefix_attn_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
            env::args().nth(10),
            env::args().nth(11),
        ),
        Some("runtime-wmma-fp8-probe-smoke") => runtime_wmma_fp8_probe_smoke(env::args().nth(2)),
        Some("runtime-wmma-fp8-qk-probe-smoke") => runtime_wmma_fp8_qk_probe_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
        ),
        Some("runtime-rocwmma-fp8-qk-probe-smoke") => runtime_rocwmma_fp8_qk_probe_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
        ),
        Some("runtime-rocwmma-fp8-attn-probe-smoke") => {
            runtime_rocwmma_fp8_attn_probe_smoke(env::args().nth(2), env::args().nth(3))
        }
        Some("runtime-paged-decode-attn-smoke") => {
            runtime_paged_decode_attn_smoke(env::args().nth(2))
        }
        Some("runtime-paged-kv-write-smoke") => runtime_paged_kv_write_smoke(env::args().nth(2)),
        Some("runtime-scheduler-paged-decode-smoke") => {
            runtime_scheduler_paged_decode_smoke(env::args().nth(2))
        }
        Some("runtime-scheduler-layer-decode-smoke") => {
            runtime_scheduler_layer_decode_smoke(env::args().nth(2))
        }
        Some("runtime-kv-paged-decode-smoke") => {
            runtime_kv_paged_decode_attn_smoke(env::args().nth(2))
        }
        Some("runtime-depthwise-conv1d-smoke") => {
            runtime_depthwise_conv1d_smoke(env::args().nth(2))
        }
        Some("runtime-linear-attn-gate-beta-smoke") => {
            runtime_linear_attn_gate_beta_smoke(env::args().nth(2))
        }
        Some("runtime-linear-attn-recurrent-smoke") => {
            runtime_linear_attn_recurrent_smoke(env::args().nth(2))
        }
        Some("runtime-mlp-smoke") => runtime_mlp_smoke(env::args().nth(2)),
        Some("inspect-package") => inspect_package(env::args().nth(2)),
        Some("package-layer-kind-inventory-smoke") => {
            package_layer_kind_inventory_smoke(env::args().nth(2), env::args().nth(3))
        }
        Some("package-load-smoke") => package_load_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("package-tensor-load-smoke") => package_tensor_load_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("package-weight-register-smoke") => package_weight_register_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("package-weight-register-many-smoke") => package_weight_register_many_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("package-materialize-smoke") => package_materialize_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("sq-fp8-materialize-smoke") => sq_fp8_materialize_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-mlp-smoke") => package_mlp_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("package-materialize-matvec-smoke") => package_materialize_matvec_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("package-aq4-matvec-smoke") => package_aq4_matvec_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-rmsnorm-smoke") => package_rmsnorm_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-rmsnorm-mlp-smoke") => package_rmsnorm_mlp_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-mlp-block-smoke") => package_mlp_block_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-proj-smoke") => package_linear_attn_proj_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-self-attn-proj-smoke") => package_self_attn_proj_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-self-attn-qk-norm-smoke") => package_self_attn_qk_norm_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("package-self-attn-rope-smoke") => package_self_attn_rope_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
        ),
        Some("package-self-attn-attention-smoke") => package_self_attn_attention_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
        ),
        Some("package-self-attn-decode-smoke") => package_self_attn_decode_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
        ),
        Some("package-self-attn-block-smoke") => package_self_attn_block_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
        ),
        Some("package-self-attn-mlp-block-smoke") => package_self_attn_mlp_block_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
        ),
        Some("package-self-attn-mlp-block-scheduler-smoke") => {
            package_self_attn_mlp_block_scheduler_smoke(
                env::args().nth(2),
                env::args().nth(3),
                env::args().nth(4),
                env::args().nth(5),
                env::args().nth(6),
                env::args().nth(7),
                env::args().nth(8),
                env::args().nth(9),
            )
        }
        Some("package-self-attn-mlp-block-model-loop-smoke") => {
            package_self_attn_mlp_block_model_loop_smoke(
                env::args().nth(2),
                env::args().nth(3),
                env::args().nth(4),
                env::args().nth(5),
                env::args().nth(6),
                env::args().nth(7),
                env::args().nth(8),
                env::args().nth(9),
                env::args().nth(10),
            )
        }
        Some("package-token-ids-model-loop-smoke") => package_token_ids_model_loop_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
            env::args().nth(10),
            env::args().nth(11),
            env::args().nth(12),
        ),
        Some("package-token-ids-mixed-request-state-smoke") => {
            package_token_ids_mixed_request_state_smoke(
                env::args().nth(2),
                env::args().nth(3),
                env::args().nth(4),
                env::args().nth(5),
                env::args().nth(6),
                env::args().nth(7),
                env::args().nth(8),
                env::args().nth(9),
                env::args().nth(10),
                env::args().nth(11),
                env::args().nth(12),
            )
        }
        Some("sq-fp8-token-ids-mixed-request-state-smoke") => {
            sq_fp8_token_ids_mixed_request_state_smoke(
                env::args().nth(2),
                env::args().nth(3),
                env::args().nth(4),
                env::args().nth(5),
                env::args().nth(6),
                env::args().nth(7),
                env::args().nth(8),
                env::args().nth(9),
                env::args().nth(10),
                env::args().nth(11),
                env::args().nth(12),
                env::args().nth(13),
            )
        }
        Some("sq-fp8-token-ids-offline-serving-throughput") => {
            sq_fp8_token_ids_offline_serving_throughput(
                env::args().nth(2),
                env::args().nth(3),
                env::args().nth(4),
                env::args().nth(5),
                env::args().nth(6),
                env::args().nth(7),
                env::args().nth(8),
                env::args().nth(9),
                env::args().nth(10),
                env::args().nth(11),
                env::args().nth(12),
                env::args().nth(13),
            )
        }
        Some("sq-fp8-token-ids-model-loop-smoke") => sq_fp8_token_ids_model_loop_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
            env::args().nth(10),
            env::args().nth(11),
            env::args().nth(12),
            env::args().nth(13),
        ),
        Some("package-token-ids-logits-smoke") => package_token_ids_logits_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
            env::args().nth(10),
            env::args().nth(11),
        ),
        Some("sq-fp8-token-ids-logits-smoke") => sq_fp8_token_ids_logits_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
            env::args().nth(10),
            env::args().nth(11),
            env::args().nth(12),
        ),
        Some("package-token-ids-generate-smoke" | "package-token-ids-bench") => {
            package_token_ids_generate_smoke(
                env::args().nth(2),
                env::args().nth(3),
                env::args().nth(4),
                env::args().nth(5),
                env::args().nth(6),
                env::args().nth(7),
                env::args().nth(8),
                env::args().nth(9),
                env::args().nth(10),
                env::args().nth(11),
                env::args().nth(12),
                env::args().nth(13),
                env::args().nth(14),
                env::args().nth(15),
            )
        }
        Some("sq-fp8-token-ids-generate-smoke" | "sq-fp8-token-ids-bench") => {
            sq_fp8_token_ids_generate_smoke(
                env::args().nth(2),
                env::args().nth(3),
                env::args().nth(4),
                env::args().nth(5),
                env::args().nth(6),
                env::args().nth(7),
                env::args().nth(8),
                env::args().nth(9),
                env::args().nth(10),
                env::args().nth(11),
                env::args().nth(12),
                env::args().nth(13),
                env::args().nth(14),
                env::args().nth(15),
                env::args().nth(16),
            )
        }
        Some("package-batch-throughput-bench") => package_batch_throughput_bench(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
            env::args().nth(10),
            env::args().nth(11),
            env::args().nth(12),
            env::args().nth(13),
            env::args().nth(14),
            env::args().nth(15),
        ),
        Some("package-prefill-rmsnorm-batch-smoke") => package_prefill_rmsnorm_batch_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
        ),
        Some("package-prefill-aq4-matvec-batch-smoke") => package_prefill_aq4_matvec_batch_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
        ),
        Some("package-self-attn-qkv-rope-batch-smoke") => package_self_attn_qkv_rope_batch_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
            env::args().nth(10),
        ),
        Some("package-self-attn-attention-batch-smoke") => package_self_attn_attention_batch_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
            env::args().nth(10),
        ),
        Some("package-self-attn-block-batch-smoke") => package_self_attn_block_batch_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
            env::args().nth(10),
        ),
        Some("package-self-attn-layer-batch-smoke") => package_self_attn_layer_batch_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
            env::args().nth(10),
        ),
        Some("sq-fp8-package-self-attn-layer-batch-smoke") => {
            sq_fp8_package_self_attn_layer_batch_smoke(
                env::args().nth(2),
                env::args().nth(3),
                env::args().nth(4),
                env::args().nth(5),
                env::args().nth(6),
                env::args().nth(7),
                env::args().nth(8),
                env::args().nth(9),
                env::args().nth(10),
                env::args().nth(11),
                env::args().nth(12),
            )
        }
        Some("sq-fp8-package-self-attn-stack-batch-smoke") => {
            sq_fp8_package_self_attn_stack_batch_smoke(
                env::args().nth(2),
                env::args().nth(3),
                env::args().nth(4),
                env::args().nth(5),
                env::args().nth(6),
                env::args().nth(7),
                env::args().nth(8),
                env::args().nth(9),
                env::args().nth(10),
                env::args().nth(11),
                env::args().nth(12),
                env::args().nth(13),
            )
        }
        Some("package-linear-attn-proj-batch-smoke") => package_linear_attn_proj_batch_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
        ),
        Some("package-linear-attn-qkv-prepare-batch-smoke") => {
            package_linear_attn_qkv_prepare_batch_smoke(
                env::args().nth(2),
                env::args().nth(3),
                env::args().nth(4),
                env::args().nth(5),
                env::args().nth(6),
                env::args().nth(7),
            )
        }
        Some("package-linear-attn-recurrent-batch-smoke") => {
            package_linear_attn_recurrent_batch_smoke(
                env::args().nth(2),
                env::args().nth(3),
                env::args().nth(4),
                env::args().nth(5),
                env::args().nth(6),
                env::args().nth(7),
            )
        }
        Some("package-linear-attn-post-batch-smoke") => package_linear_attn_post_batch_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
        ),
        Some("package-linear-attn-attention-batch-smoke") => {
            package_linear_attn_attention_batch_smoke(
                env::args().nth(2),
                env::args().nth(3),
                env::args().nth(4),
                env::args().nth(5),
                env::args().nth(6),
                env::args().nth(7),
            )
        }
        Some("package-linear-attn-mlp-batch-smoke") => package_linear_attn_mlp_batch_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
        ),
        Some("package-linear-attn-layer-batch-smoke") => package_linear_attn_layer_batch_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
        ),
        Some("package-layer-golden-smoke") => package_layer_golden_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
        ),
        Some("package-golden-prefix-smoke") => package_golden_prefix_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
            env::args().nth(10),
            env::args().nth(11),
            env::args().nth(12),
            env::args().nth(13),
            env::args().nth(14),
            env::args().nth(15),
            env::args().nth(16),
        ),
        Some("package-linear-attn-qkv-norm-smoke") => package_linear_attn_qkv_norm_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("package-linear-attn-conv1d-smoke") => package_linear_attn_conv1d_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-gate-beta-smoke") => package_linear_attn_gate_beta_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-recurrent-smoke") => package_linear_attn_recurrent_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-post-smoke") => package_linear_attn_post_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-workflow-smoke") => package_linear_attn_workflow_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-block-smoke") => package_linear_attn_block_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-mlp-block-smoke") => package_linear_attn_mlp_block_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-z-hybrid-diagnostic") => {
            package_linear_attn_z_hybrid_diagnostic(
                env::args().nth(2),
                env::args().nth(3),
                env::args().nth(4),
                env::args().nth(5),
                env::args().nth(6),
                env::args().nth(7),
                env::args().nth(8),
            )
        }
        Some("package-linear-attn-qkv-hybrid-diagnostic") => {
            package_linear_attn_qkv_hybrid_diagnostic(
                env::args().nth(2),
                env::args().nth(3),
                env::args().nth(4),
                env::args().nth(5),
                env::args().nth(6),
                env::args().nth(7),
                env::args().nth(8),
            )
        }
        Some("package-linear-attn-stateful-step-smoke") => package_linear_attn_stateful_step_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-request-state-smoke") => package_linear_attn_request_state_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
        ),
        Some("package-linear-attn-aux-smoke") => package_linear_attn_aux_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-materialize-bench") => package_materialize_bench(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("-h") | Some("--help") | None => {
            print_help();
            ExitCode::SUCCESS
        }
        Some(command) => {
            eprintln!("unknown command: {command}");
            print_help();
            ExitCode::from(2)
        }
    }
}

fn inspect_devices() -> ExitCode {
    println!("uLLM runtime ABI {}", ullm_runtime_sys::abi_version());
    let count = match ullm_runtime_sys::device_count() {
        Ok(count) => count,
        Err(err) => {
            eprintln!("failed to query device count: {err}");
            return ExitCode::from(1);
        }
    };
    println!("devices: {count}");
    for index in 0..count {
        match ullm_runtime_sys::device_info(index) {
            Ok(info) => {
                println!(
                    "[{index}] backend={} id={} name=\"{}\" mem={} compute={}.{} arch=\"{}\" flags={}",
                    info.backend,
                    info.device_id,
                    info.name,
                    info.total_global_mem,
                    info.compute_major,
                    info.compute_minor,
                    info.gcn_arch_name,
                    info.flags
                );
            }
            Err(err) => {
                eprintln!("failed to query device {index}: {err}");
                return ExitCode::from(1);
            }
        }
    }
    ExitCode::SUCCESS
}

fn runtime_smoke() -> ExitCode {
    let lhs = [1.0_f32, 2.0, 3.5, -4.0];
    let rhs = [10.0_f32, -2.0, 0.5, 4.0];
    let out = match ullm_runtime_sys::smoke_add_f32(&lhs, &rhs) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("runtime smoke failed: {err}");
            return ExitCode::from(1);
        }
    };
    println!("runtime-smoke add_f32 output: {out:?}");
    if out == [11.0, 0.0, 4.0, 0.0] {
        ExitCode::SUCCESS
    } else {
        eprintln!("runtime smoke produced unexpected output");
        ExitCode::from(1)
    }
}

fn runtime_memory_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let bytes = 4 * 1024 * 1024;
    let buffer = match context.alloc_buffer(bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let actual = match buffer.size() {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("failed to query runtime buffer size: {err}");
            return ExitCode::from(1);
        }
    };
    println!(
        "runtime-memory-smoke backend={} device_index={} name=\"{}\" bytes={}",
        info.backend, device_index, info.name, actual
    );
    if actual == bytes {
        ExitCode::SUCCESS
    } else {
        eprintln!("runtime memory smoke returned unexpected buffer size");
        ExitCode::from(1)
    }
}

fn runtime_stream_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream: {err}");
        return ExitCode::from(1);
    }
    println!(
        "runtime-stream-smoke backend={} device_index={} name=\"{}\" synchronized=true",
        info.backend, device_index, info.name
    );
    ExitCode::SUCCESS
}

fn runtime_copy_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };
    let bytes = 4096_usize;
    let mut buffer = match context.alloc_buffer(bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let input: Vec<u8> = (0..bytes)
        .map(|index| (index.wrapping_mul(31).wrapping_add(7) & 0xff) as u8)
        .collect();
    if let Err(err) = buffer.copy_from_host(0, &input, Some(&mut stream)) {
        eprintln!("failed to copy host data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after host copy: {err}");
        return ExitCode::from(1);
    }

    let mut output = vec![0_u8; bytes];
    if let Err(err) = buffer.copy_to_host(0, &mut output, Some(&mut stream)) {
        eprintln!("failed to copy runtime buffer data back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after device copy: {err}");
        return ExitCode::from(1);
    }
    if input != output {
        eprintln!("runtime copy smoke returned mismatched bytes");
        return ExitCode::from(1);
    }
    println!(
        "runtime-copy-smoke backend={} device_index={} name=\"{}\" bytes={} verified=true",
        info.backend, device_index, info.name, bytes
    );
    ExitCode::SUCCESS
}

fn runtime_rmsnorm_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let elements = 4_usize;
    let epsilon = 1e-5_f32;
    let input = [1.0_f32, 2.0, -3.0, 4.0];
    let weight = [0.5_f32, 1.0, 1.5, -2.0];
    let expected = {
        let mean_square = input.iter().map(|value| value * value).sum::<f32>() / elements as f32;
        let inv_rms = 1.0_f32 / (mean_square + epsilon).sqrt();
        input
            .iter()
            .zip(weight.iter())
            .map(|(input_value, weight_value)| input_value * inv_rms * weight_value)
            .collect::<Vec<_>>()
    };

    let mut input_bytes = Vec::with_capacity(elements * std::mem::size_of::<f32>());
    let mut weight_bytes = Vec::with_capacity(elements * std::mem::size_of::<f32>());
    for value in &input {
        input_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in &weight {
        weight_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let output_bytes = elements * std::mem::size_of::<f32>();

    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate input runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy input data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copy: {err}");
        return ExitCode::from(1);
    }

    let mut weight_buffer = match context.alloc_buffer(weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate weight runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = weight_buffer.copy_from_host(0, &weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy weight data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after weight copy: {err}");
        return ExitCode::from(1);
    }

    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = ullm_runtime_sys::rmsnorm_f32(
        &input_buffer,
        &weight_buffer,
        elements,
        epsilon,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime rmsnorm_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after rmsnorm: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy rmsnorm result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);

    if output.len() != expected.len()
        || expected
            .iter()
            .zip(output.iter())
            .any(|(lhs, rhs)| (*lhs - *rhs).abs() > 1e-5_f32)
    {
        eprintln!(
            "runtime rmsnorm smoke produced unexpected output: output={:?} expected={:?}",
            output, expected
        );
        return ExitCode::from(1);
    }
    println!(
        "runtime-rmsnorm-smoke backend={} device_index={} name=\"{}\" elements={} epsilon={} output={} verified=true",
        info.backend,
        device_index,
        info.name,
        elements,
        epsilon,
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_silu_mul_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let elements = 4_usize;
    let epsilon = 1e-5_f32;
    let gate = [-1.0_f32, 0.0_f32, 1.0_f32, 2.0_f32];
    let up = [3.0_f32, -4.0_f32, 5.0_f32, 6.0_f32];
    let expected = gate
        .iter()
        .zip(up.iter())
        .map(|(gate_value, up_value)| {
            let gate_value = *gate_value;
            gate_value * (1.0_f32 / (1.0_f32 + (-gate_value).exp())) * *up_value
        })
        .collect::<Vec<_>>();

    let mut gate_bytes = Vec::with_capacity(elements * std::mem::size_of::<f32>());
    let mut up_bytes = Vec::with_capacity(elements * std::mem::size_of::<f32>());
    for value in &gate {
        gate_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in &up {
        up_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let output_bytes = elements * std::mem::size_of::<f32>();

    let mut gate_buffer = match context.alloc_buffer(gate_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = gate_buffer.copy_from_host(0, &gate_bytes, Some(&mut stream)) {
        eprintln!("failed to copy gate data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate copy: {err}");
        return ExitCode::from(1);
    }

    let mut up_buffer = match context.alloc_buffer(up_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate up runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = up_buffer.copy_from_host(0, &up_bytes, Some(&mut stream)) {
        eprintln!("failed to copy up data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after up copy: {err}");
        return ExitCode::from(1);
    }

    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = ullm_runtime_sys::silu_mul_f32(
        &gate_buffer,
        &up_buffer,
        elements,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime silu_mul_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after silu_mul: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy silu_mul result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);

    if output.len() != expected.len()
        || expected
            .iter()
            .zip(output.iter())
            .any(|(lhs, rhs)| (*lhs - *rhs).abs() > epsilon)
    {
        eprintln!(
            "runtime silu mul smoke produced unexpected output: output={:?} expected={:?}",
            output, expected
        );
        return ExitCode::from(1);
    }
    println!(
        "runtime-silu-mul-smoke backend={} device_index={} name=\"{}\" elements={} output={} verified=true",
        info.backend,
        device_index,
        info.name,
        elements,
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_sigmoid_mul_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let elements = 4_usize;
    let epsilon = 1e-5_f32;
    let gate = [-1.0_f32, 0.0_f32, 1.0_f32, 2.0_f32];
    let input = [3.0_f32, -4.0_f32, 5.0_f32, 6.0_f32];
    let expected = runtime_host_sigmoid_mul_f32(&gate, &input);

    let gate_bytes = encode_f32_to_bytes(&gate);
    let input_bytes = encode_f32_to_bytes(&input);
    let output_bytes = elements * std::mem::size_of::<f32>();

    let mut gate_buffer = match context.alloc_buffer(gate_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = gate_buffer.copy_from_host(0, &gate_bytes, Some(&mut stream)) {
        eprintln!("failed to copy gate data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate copy: {err}");
        return ExitCode::from(1);
    }

    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate input runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy input data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copy: {err}");
        return ExitCode::from(1);
    }

    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = ullm_runtime_sys::sigmoid_mul_f32(
        &gate_buffer,
        &input_buffer,
        elements,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime sigmoid_mul_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after sigmoid_mul: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy sigmoid_mul result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);

    if output.len() != expected.len()
        || expected
            .iter()
            .zip(output.iter())
            .any(|(lhs, rhs)| (*lhs - *rhs).abs() > epsilon)
    {
        eprintln!(
            "runtime sigmoid mul smoke produced unexpected output: output={:?} expected={:?}",
            output, expected
        );
        return ExitCode::from(1);
    }
    println!(
        "runtime-sigmoid-mul-smoke backend={} device_index={} name=\"{}\" elements={} output={} verified=true",
        info.backend,
        device_index,
        info.name,
        elements,
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_add_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let lhs = [-1.0_f32, 0.0, 1.0, 2.0, 8.5, -13.25, 0.125, 64.0];
    let rhs = [3.0_f32, -4.0, 5.0, 6.0, -0.25, 2.0, -0.5, -63.5];
    let expected = runtime_host_add_f32(&lhs, &rhs);
    let lhs_bytes = encode_f32_to_bytes(&lhs);
    let rhs_bytes = encode_f32_to_bytes(&rhs);
    let output_bytes = lhs.len() * std::mem::size_of::<f32>();

    let mut lhs_buffer = match context.alloc_buffer(lhs_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate lhs runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut rhs_buffer = match context.alloc_buffer(rhs_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate rhs runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = lhs_buffer.copy_from_host(0, &lhs_bytes, Some(&mut stream)) {
        eprintln!("failed to copy lhs data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = rhs_buffer.copy_from_host(0, &rhs_bytes, Some(&mut stream)) {
        eprintln!("failed to copy rhs data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copies: {err}");
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::add_f32(
        &lhs_buffer,
        &rhs_buffer,
        lhs.len(),
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime add_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after add_f32: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy add_f32 result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);
    let tolerance = 1e-6_f32;
    if output.len() != expected.len()
        || expected
            .iter()
            .zip(output.iter())
            .any(|(lhs, rhs)| (*lhs - *rhs).abs() > tolerance)
    {
        eprintln!(
            "runtime add smoke produced unexpected output: output={:?} expected={:?}",
            output, expected
        );
        return ExitCode::from(1);
    }

    println!(
        "runtime-add-smoke backend={} device_index={} name=\"{}\" elements={} output={} verified=true",
        info.backend,
        device_index,
        info.name,
        lhs.len(),
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_rope_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let sequence_len = 2_usize;
    let heads = 2_usize;
    let head_dim = 6_usize;
    let rotary_dim = 4_usize;
    let position_offset = 3_usize;
    let rope_base = 10000.0_f32;
    let elements = sequence_len * heads * head_dim;
    let input = (0..elements)
        .map(|index| (index as f32 - 11.0) / 7.0)
        .collect::<Vec<_>>();
    let expected = runtime_host_rope_f32(
        &input,
        sequence_len,
        heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    );
    let input_bytes = encode_f32_to_bytes(&input);
    let output_bytes = input.len() * std::mem::size_of::<f32>();

    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate input runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy RoPE input into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copy: {err}");
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::rope_f32(
        &input_buffer,
        sequence_len,
        heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime rope_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after rope_f32: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy rope_f32 result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);
    let tolerance = if info.backend == "cpu" {
        1e-5_f32
    } else {
        1e-4_f32
    };
    if output.len() != expected.len() {
        eprintln!(
            "runtime RoPE smoke produced unexpected output length: output={} expected={}",
            output.len(),
            expected.len()
        );
        return ExitCode::from(1);
    }
    let mut max_abs_diff = 0.0_f32;
    for (actual, expected_value) in output.iter().zip(expected.iter()) {
        let diff = (*actual - *expected_value).abs();
        if diff > tolerance {
            eprintln!(
                "runtime RoPE smoke produced unexpected output: max_abs_diff={diff} tolerance={tolerance} output={:?} expected={:?}",
                output, expected
            );
            return ExitCode::from(1);
        }
        max_abs_diff = max_abs_diff.max(diff);
    }

    println!(
        "runtime-rope-smoke backend={} device_index={} name=\"{}\" sequence_len={} heads={} head_dim={} rotary_dim={} position_offset={} rope_base={} output={} max_abs_diff={max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        sequence_len,
        heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_causal_attn_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let sequence_len = 3_usize;
    let q_heads = 4_usize;
    let kv_heads = 2_usize;
    let head_dim = 3_usize;
    let value_dim = 2_usize;
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q = (0..sequence_len * q_heads * head_dim)
        .map(|index| (index as f32 - 8.0) / 11.0)
        .collect::<Vec<_>>();
    let k = (0..sequence_len * kv_heads * head_dim)
        .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
        .collect::<Vec<_>>();
    let v = (0..sequence_len * kv_heads * value_dim)
        .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
        .collect::<Vec<_>>();
    let expected = runtime_host_causal_attn_f32(
        &q,
        &k,
        &v,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let q_bytes = encode_f32_to_bytes(&q);
    let k_bytes = encode_f32_to_bytes(&k);
    let v_bytes = encode_f32_to_bytes(&v);
    let output_bytes = expected.len() * std::mem::size_of::<f32>();

    let mut q_buffer = match context.alloc_buffer(q_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate q runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut k_buffer = match context.alloc_buffer(k_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate k runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_buffer = match context.alloc_buffer(v_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate v runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate causal attention output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = q_buffer.copy_from_host(0, &q_bytes, Some(&mut stream)) {
        eprintln!("failed to copy causal attention q input: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = k_buffer.copy_from_host(0, &k_bytes, Some(&mut stream)) {
        eprintln!("failed to copy causal attention k input: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_buffer.copy_from_host(0, &v_bytes, Some(&mut stream)) {
        eprintln!("failed to copy causal attention v input: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!(
            "failed to synchronize runtime stream after causal attention input copies: {err}"
        );
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::causal_attn_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime causal_attn_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after causal_attn_f32: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy causal_attn_f32 result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after causal attention output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);
    let tolerance = if info.backend == "cpu" {
        1e-5_f32
    } else {
        1e-4_f32
    };
    let max_abs_diff = match verify_f32_close(
        "runtime causal attention smoke",
        &output,
        &expected,
        tolerance,
        tolerance,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}; output={output:?} expected={expected:?}");
            return ExitCode::from(1);
        }
    };

    println!(
        "runtime-causal-attn-smoke backend={} device_index={} name=\"{}\" sequence_len={} q_heads={} kv_heads={} head_dim={} value_dim={} softmax_scale={softmax_scale:.9} output={} max_abs_diff={max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_causal_attn_batch_smoke(
    device_index: Option<String>,
    batch_count: Option<String>,
    sequence_len: Option<String>,
    measured_repeats: Option<String>,
    q_heads: Option<String>,
    kv_heads: Option<String>,
    head_dim: Option<String>,
    value_dim: Option<String>,
    executor: Option<String>,
) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let batch_count = match parse_optional_usize(batch_count, 2, "batch count") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("batch count must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let sequence_len = match parse_optional_usize(sequence_len, 128, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let measured_repeats = match parse_optional_usize(measured_repeats, 3, "measured repeats") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("measured repeats must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let q_heads = match parse_optional_usize(q_heads, 16, "q heads") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("q heads must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let kv_heads = match parse_optional_usize(kv_heads, 4, "kv heads") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("kv heads must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let head_dim = match parse_optional_usize(head_dim, 256, "head dim") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("head dim must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let value_dim = match parse_optional_usize(value_dim, 256, "value dim") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("value dim must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let executor = match RuntimeCausalAttnBatchExecutor::parse(executor) {
        Ok(executor) => executor,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(2);
        }
    };

    match runtime_causal_attn_batch_smoke_impl(
        device_index,
        batch_count,
        sequence_len,
        measured_repeats,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        executor,
    ) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn runtime_causal_attn_batch_smoke_impl(
    device_index: u32,
    batch_count: usize,
    sequence_len: usize,
    measured_repeats: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    executor: RuntimeCausalAttnBatchExecutor,
) -> Result<String, String> {
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(format!(
            "runtime batched causal attention requires q_heads to be a multiple of kv_heads: q_heads={q_heads} kv_heads={kv_heads}"
        ));
    }
    let q_head_sequence_per_request = sequence_len
        .checked_mul(q_heads)
        .ok_or_else(|| "runtime batched causal attention q head-sequence overflows".to_string())?;
    let kv_head_sequence_per_request = sequence_len
        .checked_mul(kv_heads)
        .ok_or_else(|| "runtime batched causal attention kv head-sequence overflows".to_string())?;
    let q_elements_per_request = q_head_sequence_per_request
        .checked_mul(head_dim)
        .ok_or_else(|| "runtime batched causal attention q elements overflow".to_string())?;
    let k_elements_per_request = kv_head_sequence_per_request
        .checked_mul(head_dim)
        .ok_or_else(|| "runtime batched causal attention k elements overflow".to_string())?;
    let v_elements_per_request = kv_head_sequence_per_request
        .checked_mul(value_dim)
        .ok_or_else(|| "runtime batched causal attention v elements overflow".to_string())?;
    let output_elements_per_request = q_head_sequence_per_request
        .checked_mul(value_dim)
        .ok_or_else(|| "runtime batched causal attention output elements overflow".to_string())?;
    let q_elements = q_elements_per_request
        .checked_mul(batch_count)
        .ok_or_else(|| "runtime batched causal attention q batch elements overflow".to_string())?;
    let k_elements = k_elements_per_request
        .checked_mul(batch_count)
        .ok_or_else(|| "runtime batched causal attention k batch elements overflow".to_string())?;
    let v_elements = v_elements_per_request
        .checked_mul(batch_count)
        .ok_or_else(|| "runtime batched causal attention v batch elements overflow".to_string())?;
    let output_elements = output_elements_per_request
        .checked_mul(batch_count)
        .ok_or_else(|| {
            "runtime batched causal attention output batch elements overflow".to_string()
        })?;
    let q_bytes_total = checked_f32_byte_len(q_elements, "runtime batched causal attention q")?;
    let k_bytes_total = checked_f32_byte_len(k_elements, "runtime batched causal attention k")?;
    let v_bytes_total = checked_f32_byte_len(v_elements, "runtime batched causal attention v")?;
    let output_bytes_total =
        checked_f32_byte_len(output_elements, "runtime batched causal attention output")?;
    let runtime_buffer_bytes_total = q_bytes_total
        .checked_add(k_bytes_total)
        .and_then(|value| value.checked_add(v_bytes_total))
        .and_then(|value| value.checked_add(output_bytes_total))
        .ok_or_else(|| "runtime batched causal attention total byte count overflows".to_string())?;
    let max_runtime_buffer_bytes = 8_usize
        .checked_mul(1024)
        .and_then(|value| value.checked_mul(1024))
        .and_then(|value| value.checked_mul(1024))
        .ok_or_else(|| {
            "runtime batched causal attention buffer byte limit overflows".to_string()
        })?;
    if runtime_buffer_bytes_total > max_runtime_buffer_bytes {
        return Err(format!(
            "runtime batched causal attention would allocate {runtime_buffer_bytes_total} bytes across q/k/v/output buffers, above limit {max_runtime_buffer_bytes}; reduce batch_count, sequence_len, or dimensions"
        ));
    }

    let estimated_attention_pairs = (batch_count as u128)
        .checked_mul(sequence_len as u128)
        .and_then(|value| value.checked_mul(sequence_len.checked_add(1)? as u128))
        .map(|value| value / 2)
        .ok_or_else(|| "runtime batched causal attention pair count overflows".to_string())?;
    let prefill_total_input_tokens = batch_count.checked_mul(sequence_len).ok_or_else(|| {
        "runtime batched causal attention total input tokens overflow".to_string()
    })?;
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q = (0..q_elements)
        .map(|index| synthetic_attention_f32(index, 17))
        .collect::<Vec<_>>();
    let k = (0..k_elements)
        .map(|index| synthetic_attention_f32(index, 31))
        .collect::<Vec<_>>();
    let v = (0..v_elements)
        .map(|index| synthetic_attention_f32(index, 47))
        .collect::<Vec<_>>();

    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let mut q_buffer = context
        .alloc_buffer(q_bytes_total)
        .map_err(|err| format!("failed to allocate batched causal attention q buffer: {err}"))?;
    let mut k_buffer = context
        .alloc_buffer(k_bytes_total)
        .map_err(|err| format!("failed to allocate batched causal attention k buffer: {err}"))?;
    let mut v_buffer = context
        .alloc_buffer(v_bytes_total)
        .map_err(|err| format!("failed to allocate batched causal attention v buffer: {err}"))?;
    let mut output_buffer = context.alloc_buffer(output_bytes_total).map_err(|err| {
        format!("failed to allocate batched causal attention output buffer: {err}")
    })?;

    copy_f32_values_to_runtime_buffer_chunked(
        &mut q_buffer,
        &q,
        &mut stream,
        "runtime batched causal attention q",
    )?;
    copy_f32_values_to_runtime_buffer_chunked(
        &mut k_buffer,
        &k,
        &mut stream,
        "runtime batched causal attention k",
    )?;
    copy_f32_values_to_runtime_buffer_chunked(
        &mut v_buffer,
        &v,
        &mut stream,
        "runtime batched causal attention v",
    )?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize runtime batched causal attention input copies: {err}")
    })?;

    let mut run_causal_attn_batch =
        |stream: &mut ullm_runtime_sys::RuntimeStream| -> Result<(), String> {
            match executor {
                RuntimeCausalAttnBatchExecutor::Default => ullm_runtime_sys::causal_attn_batch_f32(
                    &q_buffer,
                    &k_buffer,
                    &v_buffer,
                    batch_count,
                    sequence_len,
                    q_heads,
                    kv_heads,
                    head_dim,
                    value_dim,
                    softmax_scale,
                    &mut output_buffer,
                    Some(stream),
                )
                .map_err(|err| format!("failed to run runtime causal_attn_batch_f32: {err}")),
                RuntimeCausalAttnBatchExecutor::Flash2 => {
                    ullm_runtime_sys::causal_attn_batch_f32_flash2(
                        &q_buffer,
                        &k_buffer,
                        &v_buffer,
                        batch_count,
                        sequence_len,
                        q_heads,
                        kv_heads,
                        head_dim,
                        value_dim,
                        softmax_scale,
                        &mut output_buffer,
                        Some(stream),
                    )
                    .map_err(|err| {
                        format!("failed to run runtime causal_attn_batch_f32_flash2: {err}")
                    })
                }
            }
        };

    run_causal_attn_batch(&mut stream)?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize runtime batched causal attention warmup: {err}")
    })?;

    let mut measured_ms = Vec::with_capacity(measured_repeats);
    for _ in 0..measured_repeats {
        let started = Instant::now();
        run_causal_attn_batch(&mut stream)?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize runtime batched causal attention measured run: {err}")
        })?;
        measured_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let wall_ms = measured_ms.iter().sum::<f64>() / measured_ms.len() as f64;
    let wall_ms_min = measured_ms
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .unwrap_or(wall_ms);
    let wall_ms_max = measured_ms
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(wall_ms);

    let mut sample_batches = vec![0, batch_count / 2, batch_count - 1];
    sample_batches.sort_unstable();
    sample_batches.dedup();
    let sample_points = causal_attention_sample_points(sequence_len, q_heads, value_dim);
    if sample_points.is_empty() {
        return Err("runtime batched causal attention sampled verification has no points".into());
    }
    let mut max_abs_diff = 0.0_f32;
    let mut sample_count = 0_usize;
    let mut output_preview = Vec::new();
    for batch_index in sample_batches.iter().copied() {
        let q_start = batch_index
            .checked_mul(q_elements_per_request)
            .ok_or_else(|| {
                "runtime batched causal attention q sample start overflows".to_string()
            })?;
        let k_start = batch_index
            .checked_mul(k_elements_per_request)
            .ok_or_else(|| {
                "runtime batched causal attention k sample start overflows".to_string()
            })?;
        let v_start = batch_index
            .checked_mul(v_elements_per_request)
            .ok_or_else(|| {
                "runtime batched causal attention v sample start overflows".to_string()
            })?;
        let q_slice = &q[q_start..q_start + q_elements_per_request];
        let k_slice = &k[k_start..k_start + k_elements_per_request];
        let v_slice = &v[v_start..v_start + v_elements_per_request];
        for (timestep, q_head, value_index) in sample_points.iter().copied() {
            let expected = runtime_host_causal_attn_f32_sample(
                q_slice,
                k_slice,
                v_slice,
                sequence_len,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
                timestep,
                q_head,
                value_index,
            )
            .ok_or_else(|| {
                "runtime batched causal attention failed to build sampled reference".to_string()
            })?;
            let output_element_index = batch_index
                .checked_mul(output_elements_per_request)
                .and_then(|value| {
                    value.checked_add((timestep * q_heads + q_head) * value_dim + value_index)
                })
                .ok_or_else(|| {
                    "runtime batched causal attention sampled output index overflows".to_string()
                })?;
            let actual = read_runtime_buffer_f32_scalar(
                &output_buffer,
                &mut stream,
                output_element_index,
                "runtime batched causal attention sampled output",
            )?;
            if output_preview.len() < 8 {
                output_preview.push(actual);
            }
            let diff = (actual - expected).abs();
            let tolerance = 2e-4_f32.max(expected.abs() * 2e-4_f32);
            if diff > tolerance {
                return Err(format!(
                    "runtime batched causal attention sampled guard failed: batch={batch_index} timestep={timestep} q_head={q_head} value_index={value_index} max_abs_diff={diff:.9} tolerance={tolerance:.9}"
                ));
            }
            max_abs_diff = max_abs_diff.max(diff);
            sample_count += 1;
        }
    }
    let attention_pair_tps = if wall_ms > 0.0 {
        Some((estimated_attention_pairs as f64) / wall_ms * 1000.0)
    } else {
        None
    };

    Ok(format!(
        "runtime-causal-attn-batch-smoke backend={} device_index={} name=\"{}\" prefill_mode=cold executor={} batching_mode=real batch_count={} concurrent_requests={} prompt_tokens_per_request={} prefill_total_input_tokens={} q_heads={} kv_heads={} head_dim={} value_dim={} softmax_scale={softmax_scale:.9} estimated_prefill_attention_work_tokens={} q_bytes_total={} k_bytes_total={} v_bytes_total={} output_bytes_total={} runtime_buffer_bytes_total={} warmup_runs=1 measured_repeats={} wall_ms_mean={:.6} wall_ms_min={:.6} wall_ms_max={:.6} prefill_total_input_tps={} attention_pair_tps_mean={} request_parallelism={} token_parallelism={} verification=sampled sample_count={} sampled_max_abs_diff={max_abs_diff:.9} output_preview={} verified=true",
        info.backend,
        device_index,
        info.name,
        executor.label(),
        batch_count,
        batch_count,
        sequence_len,
        prefill_total_input_tokens,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        estimated_attention_pairs,
        q_bytes_total,
        k_bytes_total,
        v_bytes_total,
        output_bytes_total,
        runtime_buffer_bytes_total,
        measured_repeats,
        wall_ms,
        wall_ms_min,
        wall_ms_max,
        tps(prefill_total_input_tokens, wall_ms)
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "null".to_string()),
        attention_pair_tps
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "null".to_string()),
        batch_count,
        sequence_len,
        sample_count,
        format_f32_preview(&output_preview),
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeCausalAttnBatchExecutor {
    Default,
    Flash2,
}

impl RuntimeCausalAttnBatchExecutor {
    fn parse(value: Option<String>) -> Result<Self, String> {
        match value.as_deref().unwrap_or("default") {
            "causal_attn_batch_f32" | "default" => Ok(Self::Default),
            "flash2" | "causal_attn_batch_f32_flash2" => Ok(Self::Flash2),
            other => Err(format!(
                "runtime batched causal attention executor must be causal_attn_batch_f32|default|flash2|causal_attn_batch_f32_flash2, got {other}"
            )),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Default => "causal_attn_batch_f32",
            Self::Flash2 => "causal_attn_batch_f32_flash2",
        }
    }
}

fn runtime_decode_attn_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let cache_len = 3_usize;
    let q_heads = 4_usize;
    let kv_heads = 2_usize;
    let head_dim = 3_usize;
    let value_dim = 2_usize;
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q = (0..q_heads * head_dim)
        .map(|index| (index as f32 - 8.0) / 11.0)
        .collect::<Vec<_>>();
    let k = (0..cache_len * kv_heads * head_dim)
        .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
        .collect::<Vec<_>>();
    let v = (0..cache_len * kv_heads * value_dim)
        .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
        .collect::<Vec<_>>();
    let expected = runtime_host_decode_attn_f32(
        &q,
        &k,
        &v,
        cache_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let q_bytes = encode_f32_to_bytes(&q);
    let k_bytes = encode_f32_to_bytes(&k);
    let v_bytes = encode_f32_to_bytes(&v);
    let output_bytes = expected.len() * std::mem::size_of::<f32>();

    let mut q_buffer = match context.alloc_buffer(q_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate decode attention q runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut k_buffer = match context.alloc_buffer(k_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate decode attention k cache runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_buffer = match context.alloc_buffer(v_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate decode attention v cache runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate decode attention output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = q_buffer.copy_from_host(0, &q_bytes, Some(&mut stream)) {
        eprintln!("failed to copy decode attention q input: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = k_buffer.copy_from_host(0, &k_bytes, Some(&mut stream)) {
        eprintln!("failed to copy decode attention k cache: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_buffer.copy_from_host(0, &v_bytes, Some(&mut stream)) {
        eprintln!("failed to copy decode attention v cache: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!(
            "failed to synchronize runtime stream after decode attention input copies: {err}"
        );
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::decode_attn_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        cache_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime decode_attn_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after decode_attn_f32: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy decode_attn_f32 result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after decode attention output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);
    let tolerance = if info.backend == "cpu" {
        1e-5_f32
    } else {
        1e-4_f32
    };
    let max_abs_diff = match verify_f32_close(
        "runtime decode attention smoke",
        &output,
        &expected,
        tolerance,
        tolerance,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}; output={output:?} expected={expected:?}");
            return ExitCode::from(1);
        }
    };

    println!(
        "runtime-decode-attn-smoke backend={} device_index={} name=\"{}\" cache_len={} q_heads={} kv_heads={} head_dim={} value_dim={} softmax_scale={softmax_scale:.9} output={} max_abs_diff={max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        cache_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

const RDNA4_FP8_AUTO_ROCWMMA_NEW_TOKEN_THRESHOLD: usize = 64;

const RUNTIME_CACHED_PREFIX_DISPATCH_OPERATION: &str = "cached_prefix_attention";
const RUNTIME_CACHED_PREFIX_DISPATCH_PHASE: &str = "prefill";
const RUNTIME_CACHED_PREFIX_DISPATCH_IMPLEMENTATIONS: &[BackendImplementation<'static>] = &[
    BackendImplementation {
        id: "cached_prefix_chunked",
        operation: RUNTIME_CACHED_PREFIX_DISPATCH_OPERATION,
        phase: RUNTIME_CACHED_PREFIX_DISPATCH_PHASE,
        format_id: None,
        model_arch: None,
        gpu_arch: None,
        gpu_name: None,
        priority: 0,
    },
    BackendImplementation {
        id: "cached_prefix_rdna4_fp8_auto",
        operation: RUNTIME_CACHED_PREFIX_DISPATCH_OPERATION,
        phase: RUNTIME_CACHED_PREFIX_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch: None,
        gpu_arch: Some("RDNA4"),
        gpu_name: None,
        priority: 10,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeCachedPrefixAttnExecutor {
    Chunked,
    Flash2,
    Flash2Fp8Q,
    RocwmmaFp8,
    Rdna4Fp8Auto,
    DecodeLoop,
}

impl RuntimeCachedPrefixAttnExecutor {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cached_prefix_chunked" | "chunked" => Ok(Self::Chunked),
            "cached_prefix_flash2" | "flash2" => Ok(Self::Flash2),
            "cached_prefix_flash2_fp8q" | "flash2_fp8q" | "fp8q_flash2" => Ok(Self::Flash2Fp8Q),
            "cached_prefix_rocwmma_fp8" | "rocwmma_fp8" | "cached_prefix_rocwmma" | "rocwmma" => {
                Ok(Self::RocwmmaFp8)
            }
            "cached_prefix_rdna4_fp8_auto" | "rdna4_fp8_auto" | "fp8_auto" => {
                Ok(Self::Rdna4Fp8Auto)
            }
            "decode_attn_f32_loop" | "decode_loop" => Ok(Self::DecodeLoop),
            other => Err(format!(
                "runtime cached prefix attention executor must be cached_prefix_chunked|chunked|cached_prefix_flash2|flash2|cached_prefix_flash2_fp8q|flash2_fp8q|cached_prefix_rocwmma_fp8|rocwmma_fp8|cached_prefix_rdna4_fp8_auto|rdna4_fp8_auto|decode_loop, got {other}"
            )),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Chunked => "cached_prefix_chunked",
            Self::Flash2 => "cached_prefix_flash2",
            Self::Flash2Fp8Q => "cached_prefix_flash2_fp8q",
            Self::RocwmmaFp8 => "cached_prefix_rocwmma_fp8",
            Self::Rdna4Fp8Auto => "cached_prefix_rdna4_fp8_auto",
            Self::DecodeLoop => "decode_attn_f32_loop",
        }
    }

    fn resolved_label(
        self,
        new_tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
    ) -> &'static str {
        match self {
            Self::Rdna4Fp8Auto
                if !rdna4_fp8_auto_uses_rocwmma(
                    new_tokens, q_heads, kv_heads, head_dim, value_dim,
                ) =>
            {
                Self::Flash2Fp8Q.label()
            }
            Self::Rdna4Fp8Auto => Self::RocwmmaFp8.label(),
            other => other.label(),
        }
    }

    fn uses_fp8_q(self) -> bool {
        matches!(
            self,
            Self::Flash2Fp8Q | Self::RocwmmaFp8 | Self::Rdna4Fp8Auto
        )
    }

    fn requires_fp8_kv(self) -> bool {
        matches!(
            self,
            Self::Flash2Fp8Q | Self::RocwmmaFp8 | Self::Rdna4Fp8Auto
        )
    }

    fn resolves_to_flash2_fp8q(
        self,
        new_tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
    ) -> bool {
        matches!(self, Self::Flash2Fp8Q)
            || matches!(self, Self::Rdna4Fp8Auto)
                && !rdna4_fp8_auto_uses_rocwmma(
                    new_tokens, q_heads, kv_heads, head_dim, value_dim,
                )
    }

    fn resolves_to_rocwmma_fp8(
        self,
        new_tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
    ) -> bool {
        matches!(self, Self::RocwmmaFp8)
            || matches!(self, Self::Rdna4Fp8Auto)
                && rdna4_fp8_auto_uses_rocwmma(
                    new_tokens, q_heads, kv_heads, head_dim, value_dim,
                )
    }
}

fn rdna4_fp8_auto_uses_rocwmma(
    new_tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
) -> bool {
    new_tokens >= RDNA4_FP8_AUTO_ROCWMMA_NEW_TOKEN_THRESHOLD
        && head_dim.is_multiple_of(16)
        && value_dim.is_multiple_of(16)
        && (q_heads / kv_heads).is_multiple_of(16)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeCachedPrefixKvCacheDtype {
    F32,
    Fp8E4m3,
}

impl RuntimeCachedPrefixKvCacheDtype {
    fn parse(value: Option<String>) -> Result<Self, String> {
        match value.as_deref().unwrap_or("fp8_e4m3") {
            "f32" | "fp32" => Ok(Self::F32),
            "fp8" | "fp8_e4m3" | "e4m3" => Ok(Self::Fp8E4m3),
            other => Err(format!(
                "runtime cached prefix attention kv cache dtype must be fp8_e4m3|fp8|f32, got {other}"
            )),
        }
    }

    fn is_label(value: &str) -> bool {
        matches!(value, "f32" | "fp32" | "fp8" | "fp8_e4m3" | "e4m3")
    }

    fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Fp8E4m3 => "fp8_e4m3",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RuntimeCachedPrefixDispatchSelection {
    executor: RuntimeCachedPrefixAttnExecutor,
    source: &'static str,
    implementation_id: &'static str,
    request_format_id: Option<&'static str>,
    request_gpu_arch: Option<&'static str>,
}

fn runtime_device_gpu_arch(info: &ullm_runtime_sys::DeviceInfo) -> Option<&'static str> {
    let gcn_arch = info.gcn_arch_name.to_ascii_lowercase();
    let name = info.name.to_ascii_lowercase();
    if info.compute_major == 12 || gcn_arch.starts_with("gfx12") || name.contains("r9700") {
        Some("RDNA4")
    } else if info.compute_major == 10 || gcn_arch.starts_with("gfx10") || name.contains("v620") {
        Some("RDNA2")
    } else {
        None
    }
}

fn select_runtime_cached_prefix_executor(
    requested: Option<RuntimeCachedPrefixAttnExecutor>,
    kv_cache_dtype: RuntimeCachedPrefixKvCacheDtype,
    info: &ullm_runtime_sys::DeviceInfo,
) -> Result<RuntimeCachedPrefixDispatchSelection, String> {
    if let Some(executor) = requested {
        return Ok(RuntimeCachedPrefixDispatchSelection {
            executor,
            source: "cli_override",
            implementation_id: executor.label(),
            request_format_id: None,
            request_gpu_arch: runtime_device_gpu_arch(info),
        });
    }

    let request_format_id = match kv_cache_dtype {
        RuntimeCachedPrefixKvCacheDtype::F32 => None,
        RuntimeCachedPrefixKvCacheDtype::Fp8E4m3 => Some(FORMAT_SQ8_0),
    };
    let request_gpu_arch = runtime_device_gpu_arch(info);
    let request = BackendRequest {
        operation: RUNTIME_CACHED_PREFIX_DISPATCH_OPERATION,
        phase: RUNTIME_CACHED_PREFIX_DISPATCH_PHASE,
        format_id: request_format_id,
        model_arch: None,
        gpu_arch: request_gpu_arch,
        gpu_name: Some(info.name.as_str()),
    };
    let implementation = select_backend(&request, RUNTIME_CACHED_PREFIX_DISPATCH_IMPLEMENTATIONS)
        .ok_or_else(|| "no cached-prefix attention backend implementation matched".to_string())?;
    let executor = RuntimeCachedPrefixAttnExecutor::parse(implementation.id)?;
    Ok(RuntimeCachedPrefixDispatchSelection {
        executor,
        source: "backend_dispatch",
        implementation_id: implementation.id,
        request_format_id,
        request_gpu_arch,
    })
}

fn runtime_cached_prefix_attn_smoke(
    device_index: Option<String>,
    cached_prefix_tokens: Option<String>,
    new_tokens: Option<String>,
    measured_repeats: Option<String>,
    q_heads: Option<String>,
    kv_heads: Option<String>,
    head_dim: Option<String>,
    value_dim: Option<String>,
    executor: Option<String>,
    kv_cache_dtype: Option<String>,
) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let cached_prefix_tokens =
        match parse_optional_usize(cached_prefix_tokens, 4096, "cached prefix tokens") {
            Ok(value) => value,
            Err(code) => return code,
        };
    let new_tokens = match parse_optional_usize(new_tokens, 16, "new tokens") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("new tokens must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let measured_repeats = match parse_optional_usize(measured_repeats, 3, "measured repeats") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("measured repeats must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let q_heads = match parse_optional_usize(q_heads, 16, "q heads") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("q heads must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let kv_heads = match parse_optional_usize(kv_heads, 4, "kv heads") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("kv heads must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let head_dim = match parse_optional_usize(head_dim, 256, "head dim") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("head dim must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let value_dim = match parse_optional_usize(value_dim, 256, "value dim") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("value dim must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let (executor_arg, kv_cache_dtype_arg) = match (executor, kv_cache_dtype) {
        (Some(value), None) if RuntimeCachedPrefixKvCacheDtype::is_label(&value) => {
            (None, Some(value))
        }
        (executor, kv_cache_dtype) => (executor, kv_cache_dtype),
    };
    let executor = match executor_arg {
        Some(value) => match RuntimeCachedPrefixAttnExecutor::parse(&value) {
            Ok(value) => Some(value),
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(2);
            }
        },
        None => None,
    };
    let kv_cache_dtype = match RuntimeCachedPrefixKvCacheDtype::parse(kv_cache_dtype_arg) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(2);
        }
    };

    match runtime_cached_prefix_attn_smoke_impl(
        device_index,
        cached_prefix_tokens,
        new_tokens,
        measured_repeats,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        executor,
        kv_cache_dtype,
    ) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn runtime_cached_prefix_attn_smoke_impl(
    device_index: u32,
    cached_prefix_tokens: usize,
    new_tokens: usize,
    measured_repeats: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    executor: Option<RuntimeCachedPrefixAttnExecutor>,
    kv_cache_dtype: RuntimeCachedPrefixKvCacheDtype,
) -> Result<String, String> {
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(format!(
            "runtime cached prefix attention requires q_heads to be a multiple of kv_heads: q_heads={q_heads} kv_heads={kv_heads}"
        ));
    }
    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let dispatch_selection = select_runtime_cached_prefix_executor(executor, kv_cache_dtype, &info)?;
    let executor = dispatch_selection.executor;
    if executor == RuntimeCachedPrefixAttnExecutor::DecodeLoop
        && kv_cache_dtype != RuntimeCachedPrefixKvCacheDtype::F32
    {
        return Err(
            "runtime cached prefix attention decode_loop executor currently supports only f32 kv cache; use cached_prefix_chunked for fp8_e4m3"
                .to_string(),
        );
    }
    if executor.requires_fp8_kv() && kv_cache_dtype != RuntimeCachedPrefixKvCacheDtype::Fp8E4m3 {
        return Err(format!(
            "runtime cached prefix attention {} executor requires fp8_e4m3 kv cache",
            executor.label()
        ));
    }
    if executor.resolves_to_flash2_fp8q(new_tokens, q_heads, kv_heads, head_dim, value_dim) {
        if kv_cache_dtype != RuntimeCachedPrefixKvCacheDtype::Fp8E4m3 {
            return Err(
                "runtime cached prefix attention flash2_fp8q executor requires fp8_e4m3 kv cache"
                    .to_string(),
            );
        }
        if value_dim > 256 {
            return Err(
                "runtime cached prefix attention flash2_fp8q executor currently requires value_dim <= 256"
                    .to_string(),
            );
        }
    }
    if executor.resolves_to_rocwmma_fp8(new_tokens, q_heads, kv_heads, head_dim, value_dim) {
        if !head_dim.is_multiple_of(16) || !value_dim.is_multiple_of(16) {
            return Err(
                "runtime cached prefix attention rocwmma_fp8 executor currently requires head_dim and value_dim to be multiples of 16"
                    .to_string(),
            );
        }
        if !(q_heads / kv_heads).is_multiple_of(16) {
            return Err(
                "runtime cached prefix attention rocwmma_fp8 executor requires q_heads/kv_heads to be a multiple of 16"
                    .to_string(),
            );
        }
    }
    let total_context_tokens = cached_prefix_tokens
        .checked_add(new_tokens)
        .ok_or_else(|| "runtime cached prefix attention total context overflows".to_string())?;
    let q_elements = q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "runtime cached prefix attention q element count overflows".to_string())?;
    let output_elements = q_heads.checked_mul(value_dim).ok_or_else(|| {
        "runtime cached prefix attention output element count overflows".to_string()
    })?;
    let q_sequence_elements = new_tokens.checked_mul(q_elements).ok_or_else(|| {
        "runtime cached prefix attention q sequence element count overflows".to_string()
    })?;
    let kv_head_context = total_context_tokens.checked_mul(kv_heads).ok_or_else(|| {
        "runtime cached prefix attention kv head-context count overflows".to_string()
    })?;
    let k_cache_elements = kv_head_context.checked_mul(head_dim).ok_or_else(|| {
        "runtime cached prefix attention k cache element count overflows".to_string()
    })?;
    let v_cache_elements = kv_head_context.checked_mul(value_dim).ok_or_else(|| {
        "runtime cached prefix attention v cache element count overflows".to_string()
    })?;
    let estimated_attention_pairs = cached_prefix_attention_pairs(cached_prefix_tokens, new_tokens)
        .ok_or_else(|| "runtime cached prefix attention pair count overflows".to_string())?;
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let host_input_elements = q_sequence_elements
        .checked_add(k_cache_elements)
        .and_then(|value| value.checked_add(v_cache_elements))
        .ok_or_else(|| {
            "runtime cached prefix attention host input element count overflows".to_string()
        })?;
    let host_input_bytes = checked_f32_byte_len(
        host_input_elements,
        "runtime cached prefix attention synthetic host input",
    )?;
    let max_host_input_bytes = 8_usize
        .checked_mul(1024)
        .and_then(|value| value.checked_mul(1024))
        .and_then(|value| value.checked_mul(1024))
        .ok_or_else(|| "runtime cached prefix attention host input limit overflows".to_string())?;
    if host_input_bytes > max_host_input_bytes {
        return Err(format!(
            "runtime cached prefix attention synthetic host input would allocate {host_input_bytes} bytes, above limit {max_host_input_bytes}; reduce cached_prefix_tokens/new_tokens or dimensions"
        ));
    }

    let q_sequence = (0..q_sequence_elements)
        .map(|index| synthetic_attention_f32(index, 17))
        .collect::<Vec<_>>();
    let k_cache = (0..k_cache_elements)
        .map(|index| synthetic_attention_f32(index, 31))
        .collect::<Vec<_>>();
    let v_cache = (0..v_cache_elements)
        .map(|index| synthetic_attention_f32(index, 47))
        .collect::<Vec<_>>();
    let mut q_sequence_fp8 = Vec::new();
    let mut q_sequence_reference_storage = Vec::new();
    let mut q_sequence_scale = 1.0_f32;
    let executor_uses_fp8_q = executor.uses_fp8_q();
    if executor_uses_fp8_q {
        let quantized = fp8_e4m3_quantize(&q_sequence);
        q_sequence_fp8 = quantized.encoded;
        q_sequence_scale = quantized.scale;
        q_sequence_reference_storage = quantized.decoded;
    }
    let q_sequence_reference = if executor_uses_fp8_q {
        q_sequence_reference_storage.as_slice()
    } else {
        q_sequence.as_slice()
    };
    let mut k_cache_fp8 = Vec::new();
    let mut v_cache_fp8 = Vec::new();
    let mut k_cache_reference_storage = Vec::new();
    let mut v_cache_reference_storage = Vec::new();
    let mut k_cache_scale = 1.0_f32;
    let mut v_cache_scale = 1.0_f32;
    if kv_cache_dtype == RuntimeCachedPrefixKvCacheDtype::Fp8E4m3 {
        let quantized = fp8_e4m3_quantize(&k_cache);
        k_cache_fp8 = quantized.encoded;
        k_cache_scale = quantized.scale;
        k_cache_reference_storage = quantized.decoded;

        let quantized = fp8_e4m3_quantize(&v_cache);
        v_cache_fp8 = quantized.encoded;
        v_cache_scale = quantized.scale;
        v_cache_reference_storage = quantized.decoded;
    }
    let k_cache_reference = if kv_cache_dtype == RuntimeCachedPrefixKvCacheDtype::Fp8E4m3 {
        k_cache_reference_storage.as_slice()
    } else {
        k_cache.as_slice()
    };
    let v_cache_reference = if kv_cache_dtype == RuntimeCachedPrefixKvCacheDtype::Fp8E4m3 {
        v_cache_reference_storage.as_slice()
    } else {
        v_cache.as_slice()
    };

    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let k_cache_buffer_bytes = match kv_cache_dtype {
        RuntimeCachedPrefixKvCacheDtype::F32 => {
            checked_f32_byte_len(k_cache_elements, "runtime cached prefix attention k cache")?
        }
        RuntimeCachedPrefixKvCacheDtype::Fp8E4m3 => k_cache_elements,
    };
    let v_cache_buffer_bytes = match kv_cache_dtype {
        RuntimeCachedPrefixKvCacheDtype::F32 => {
            checked_f32_byte_len(v_cache_elements, "runtime cached prefix attention v cache")?
        }
        RuntimeCachedPrefixKvCacheDtype::Fp8E4m3 => v_cache_elements,
    };
    let mut k_cache_buffer = context.alloc_buffer(k_cache_buffer_bytes).map_err(|err| {
        format!("failed to allocate runtime cached prefix attention k cache: {err}")
    })?;
    let mut v_cache_buffer = context.alloc_buffer(v_cache_buffer_bytes).map_err(|err| {
        format!("failed to allocate runtime cached prefix attention v cache: {err}")
    })?;
    match kv_cache_dtype {
        RuntimeCachedPrefixKvCacheDtype::F32 => {
            let k_bytes = encode_f32_to_bytes(&k_cache);
            k_cache_buffer
                .copy_from_host(0, &k_bytes, Some(&mut stream))
                .map_err(|err| {
                    format!("failed to copy runtime cached prefix attention k cache: {err}")
                })?;
            let v_bytes = encode_f32_to_bytes(&v_cache);
            v_cache_buffer
                .copy_from_host(0, &v_bytes, Some(&mut stream))
                .map_err(|err| {
                    format!("failed to copy runtime cached prefix attention v cache: {err}")
                })?;
        }
        RuntimeCachedPrefixKvCacheDtype::Fp8E4m3 => {
            k_cache_buffer
                .copy_from_host(0, &k_cache_fp8, Some(&mut stream))
                .map_err(|err| {
                    format!("failed to copy runtime cached prefix attention fp8 k cache: {err}")
                })?;
            v_cache_buffer
                .copy_from_host(0, &v_cache_fp8, Some(&mut stream))
                .map_err(|err| {
                    format!("failed to copy runtime cached prefix attention fp8 v cache: {err}")
                })?;
        }
    }

    let q_token_f32_bytes =
        checked_f32_byte_len(q_elements, "runtime cached prefix attention q token")?;
    let q_sequence_buffer_bytes = if executor_uses_fp8_q {
        q_sequence_elements
    } else {
        checked_f32_byte_len(
            q_sequence_elements,
            "runtime cached prefix attention q sequence",
        )?
    };
    let output_bytes = checked_f32_byte_len(
        output_elements,
        "runtime cached prefix attention output token",
    )?;
    let output_sequence_elements = new_tokens.checked_mul(output_elements).ok_or_else(|| {
        "runtime cached prefix attention output sequence element count overflows".to_string()
    })?;
    let output_sequence_bytes = checked_f32_byte_len(
        output_sequence_elements,
        "runtime cached prefix attention output sequence",
    )?;
    let mut q_sequence_buffer = context
        .alloc_buffer(q_sequence_buffer_bytes)
        .map_err(|err| {
            format!("failed to allocate runtime cached prefix attention q sequence: {err}")
        })?;
    if executor_uses_fp8_q {
        q_sequence_buffer
            .copy_from_host(0, &q_sequence_fp8, Some(&mut stream))
            .map_err(|err| {
                format!("failed to copy runtime cached prefix attention fp8 q sequence: {err}")
            })?;
    } else {
        let q_sequence_bytes_host = encode_f32_to_bytes(&q_sequence);
        q_sequence_buffer
            .copy_from_host(0, &q_sequence_bytes_host, Some(&mut stream))
            .map_err(|err| {
                format!("failed to copy runtime cached prefix attention q sequence: {err}")
            })?;
    }
    let mut output_sequence_buffer =
        context.alloc_buffer(output_sequence_bytes).map_err(|err| {
            format!("failed to allocate runtime cached prefix attention output sequence: {err}")
        })?;
    let mut q_buffers = Vec::with_capacity(new_tokens);
    let mut output_buffers = Vec::with_capacity(new_tokens);
    for token_index in 0..new_tokens {
        let q_start = token_index
            .checked_mul(q_elements)
            .ok_or_else(|| "runtime cached prefix attention q start overflows".to_string())?;
        let q_end = q_start
            .checked_add(q_elements)
            .ok_or_else(|| "runtime cached prefix attention q end overflows".to_string())?;
        let mut q_buffer = context.alloc_buffer(q_token_f32_bytes).map_err(|err| {
            format!(
                "failed to allocate runtime cached prefix attention q token {token_index}: {err}"
            )
        })?;
        let q_token_bytes = encode_f32_to_bytes(&q_sequence[q_start..q_end]);
        q_buffer
            .copy_from_host(0, &q_token_bytes, Some(&mut stream))
            .map_err(|err| {
                format!(
                    "failed to copy runtime cached prefix attention q token {token_index}: {err}"
                )
            })?;
        q_buffers.push(q_buffer);
        output_buffers.push(context.alloc_buffer(output_bytes).map_err(|err| {
            format!("failed to allocate runtime cached prefix attention output token {token_index}: {err}")
        })?);
    }
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize runtime cached prefix attention setup: {err}")
    })?;

    let measured_ms = {
        let mut run_cached_prefix_attn =
            |stream: &mut ullm_runtime_sys::RuntimeStream| -> Result<(), String> {
                match executor {
                    RuntimeCachedPrefixAttnExecutor::Chunked => match kv_cache_dtype {
                        RuntimeCachedPrefixKvCacheDtype::F32 => {
                            ullm_runtime_sys::cached_prefix_attn_f32(
                                &q_sequence_buffer,
                                &k_cache_buffer,
                                &v_cache_buffer,
                                cached_prefix_tokens,
                                new_tokens,
                                q_heads,
                                kv_heads,
                                head_dim,
                                value_dim,
                                softmax_scale,
                                &mut output_sequence_buffer,
                                Some(stream),
                            )
                            .map_err(|err| {
                                format!(
                                    "failed to run runtime cached prefix chunked attention: {err}"
                                )
                            })?;
                        }
                        RuntimeCachedPrefixKvCacheDtype::Fp8E4m3 => {
                            ullm_runtime_sys::cached_prefix_attn_fp8_e4m3(
                                    &q_sequence_buffer,
                                    &k_cache_buffer,
                                    &v_cache_buffer,
                                    cached_prefix_tokens,
                                    new_tokens,
                                    q_heads,
                                    kv_heads,
                                    head_dim,
                                    value_dim,
                                    softmax_scale,
                                    k_cache_scale,
                                    v_cache_scale,
                                    &mut output_sequence_buffer,
                                    Some(stream),
                                )
                                .map_err(|err| {
                                    format!(
                                        "failed to run runtime fp8 e4m3 cached prefix chunked attention: {err}"
                                    )
                                })?;
                        }
                    },
                    RuntimeCachedPrefixAttnExecutor::Flash2 => match kv_cache_dtype {
                        RuntimeCachedPrefixKvCacheDtype::F32 => {
                            ullm_runtime_sys::cached_prefix_attn_f32_flash2(
                                &q_sequence_buffer,
                                &k_cache_buffer,
                                &v_cache_buffer,
                                cached_prefix_tokens,
                                new_tokens,
                                q_heads,
                                kv_heads,
                                head_dim,
                                value_dim,
                                softmax_scale,
                                &mut output_sequence_buffer,
                                Some(stream),
                            )
                            .map_err(|err| {
                                format!(
                                    "failed to run runtime f32 cached prefix flash2 attention: {err}"
                                )
                            })?;
                        }
                        RuntimeCachedPrefixKvCacheDtype::Fp8E4m3 => {
                            ullm_runtime_sys::cached_prefix_attn_fp8_e4m3_flash2(
                                &q_sequence_buffer,
                                &k_cache_buffer,
                                &v_cache_buffer,
                                cached_prefix_tokens,
                                new_tokens,
                                q_heads,
                                kv_heads,
                                head_dim,
                                value_dim,
                                softmax_scale,
                                k_cache_scale,
                                v_cache_scale,
                                &mut output_sequence_buffer,
                                Some(stream),
                            )
                            .map_err(|err| {
                                format!(
                                    "failed to run runtime fp8 e4m3 cached prefix flash2 attention: {err}"
                                )
                            })?;
                        }
                    },
                    RuntimeCachedPrefixAttnExecutor::Flash2Fp8Q => {
                        ullm_runtime_sys::cached_prefix_attn_fp8_e4m3_flash2_fp8q(
                            &q_sequence_buffer,
                            &k_cache_buffer,
                            &v_cache_buffer,
                            cached_prefix_tokens,
                            new_tokens,
                            q_heads,
                            kv_heads,
                            head_dim,
                            value_dim,
                            softmax_scale,
                            q_sequence_scale,
                            k_cache_scale,
                            v_cache_scale,
                            &mut output_sequence_buffer,
                            Some(stream),
                        )
                        .map_err(|err| {
                            format!(
                                "failed to run runtime fp8 e4m3 cached prefix flash2 fp8q attention: {err}"
                            )
                        })?;
                    }
                    RuntimeCachedPrefixAttnExecutor::RocwmmaFp8 => {
                        ullm_runtime_sys::cached_prefix_attn_fp8_e4m3_rocwmma(
                            &q_sequence_buffer,
                            &k_cache_buffer,
                            &v_cache_buffer,
                            cached_prefix_tokens,
                            new_tokens,
                            q_heads,
                            kv_heads,
                            head_dim,
                            value_dim,
                            softmax_scale,
                            q_sequence_scale,
                            k_cache_scale,
                            v_cache_scale,
                            &mut output_sequence_buffer,
                            Some(stream),
                        )
                        .map_err(|err| {
                            format!(
                                "failed to run runtime fp8 e4m3 cached prefix rocwmma attention: {err}"
                            )
                        })?;
                    }
                    RuntimeCachedPrefixAttnExecutor::Rdna4Fp8Auto => {
                        if !executor.resolves_to_rocwmma_fp8(
                            new_tokens, q_heads, kv_heads, head_dim, value_dim,
                        ) {
                            ullm_runtime_sys::cached_prefix_attn_fp8_e4m3_flash2_fp8q(
                                &q_sequence_buffer,
                                &k_cache_buffer,
                                &v_cache_buffer,
                                cached_prefix_tokens,
                                new_tokens,
                                q_heads,
                                kv_heads,
                                head_dim,
                                value_dim,
                                softmax_scale,
                                q_sequence_scale,
                                k_cache_scale,
                                v_cache_scale,
                                &mut output_sequence_buffer,
                                Some(stream),
                            )
                            .map_err(|err| {
                                format!(
                                    "failed to run runtime fp8 e4m3 cached prefix rdna4 auto flash2 fp8q attention: {err}"
                                )
                            })?;
                        } else {
                            ullm_runtime_sys::cached_prefix_attn_fp8_e4m3_rocwmma(
                                &q_sequence_buffer,
                                &k_cache_buffer,
                                &v_cache_buffer,
                                cached_prefix_tokens,
                                new_tokens,
                                q_heads,
                                kv_heads,
                                head_dim,
                                value_dim,
                                softmax_scale,
                                q_sequence_scale,
                                k_cache_scale,
                                v_cache_scale,
                                &mut output_sequence_buffer,
                                Some(stream),
                            )
                            .map_err(|err| {
                                format!(
                                    "failed to run runtime fp8 e4m3 cached prefix rdna4 auto rocwmma attention: {err}"
                                )
                            })?;
                        }
                    }
                    RuntimeCachedPrefixAttnExecutor::DecodeLoop => {
                        for (token_index, (q_buffer, output_buffer)) in
                            q_buffers.iter().zip(output_buffers.iter_mut()).enumerate()
                        {
                            let cache_len = cached_prefix_tokens
                                .checked_add(token_index)
                                .and_then(|value| value.checked_add(1))
                                .ok_or_else(|| {
                                    "runtime cached prefix attention step cache length overflows"
                                        .to_string()
                                })?;
                            ullm_runtime_sys::decode_attn_f32(
                                q_buffer,
                                &k_cache_buffer,
                                &v_cache_buffer,
                                cache_len,
                                q_heads,
                                kv_heads,
                                head_dim,
                                value_dim,
                                softmax_scale,
                                output_buffer,
                                Some(stream),
                            )
                            .map_err(|err| {
                                format!(
                                    "failed to run runtime cached prefix attention step {token_index}: {err}"
                                )
                            })?;
                        }
                    }
                }
                Ok(())
            };

        run_cached_prefix_attn(&mut stream)?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize runtime cached prefix attention warmup: {err}")
        })?;

        let mut measured_ms = Vec::with_capacity(measured_repeats);
        for _ in 0..measured_repeats {
            let started = Instant::now();
            run_cached_prefix_attn(&mut stream)?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize runtime cached prefix attention measured run: {err}")
            })?;
            measured_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        measured_ms
    };

    let wall_ms = measured_ms.iter().sum::<f64>() / measured_ms.len() as f64;
    let wall_ms_min = measured_ms
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .unwrap_or(wall_ms);
    let wall_ms_max = measured_ms
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(wall_ms);

    let sample_steps = cached_prefix_sample_steps(new_tokens);
    let mut max_abs_diff = 0.0_f32;
    let mut sample_count = 0_usize;
    let mut output_preview = Vec::new();
    let chunked_output_sequence = if matches!(
        executor,
        RuntimeCachedPrefixAttnExecutor::Chunked
            | RuntimeCachedPrefixAttnExecutor::Flash2
            | RuntimeCachedPrefixAttnExecutor::Flash2Fp8Q
            | RuntimeCachedPrefixAttnExecutor::RocwmmaFp8
            | RuntimeCachedPrefixAttnExecutor::Rdna4Fp8Auto
    ) {
        Some(read_runtime_buffer_f32(
            &output_sequence_buffer,
            &mut stream,
            output_sequence_elements,
            "runtime cached prefix attention chunked output sequence",
        )?)
    } else {
        None
    };
    let resolved_executor = executor.resolved_label(new_tokens, q_heads, kv_heads, head_dim, value_dim);
    for (sample_index, token_index) in sample_steps.iter().copied().enumerate() {
        let output = if let Some(output_sequence) = chunked_output_sequence.as_ref() {
            let output_start = token_index.checked_mul(output_elements).ok_or_else(|| {
                "runtime cached prefix attention sampled output start overflows".to_string()
            })?;
            let output_end = output_start.checked_add(output_elements).ok_or_else(|| {
                "runtime cached prefix attention sampled output end overflows".to_string()
            })?;
            output_sequence[output_start..output_end].to_vec()
        } else {
            read_runtime_buffer_f32(
                &output_buffers[token_index],
                &mut stream,
                output_elements,
                "runtime cached prefix attention sampled output",
            )?
        };
        if token_index == new_tokens - 1 {
            output_preview = output[..output.len().min(8)].to_vec();
        }
        let q_head = match sample_index % 3 {
            0 => 0,
            1 => q_heads / 2,
            _ => q_heads - 1,
        };
        let value_index = match sample_index % 3 {
            0 => 0,
            1 => value_dim / 2,
            _ => value_dim - 1,
        };
        let q_start = token_index.checked_mul(q_elements).ok_or_else(|| {
            "runtime cached prefix attention sampled q start overflows".to_string()
        })?;
        let q_end = q_start
            .checked_add(q_elements)
            .ok_or_else(|| "runtime cached prefix attention sampled q end overflows".to_string())?;
        let cache_len = cached_prefix_tokens
            .checked_add(token_index)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| {
                "runtime cached prefix attention sampled cache length overflows".to_string()
            })?;
        let expected = runtime_host_decode_attn_f32_sample(
            &q_sequence_reference[q_start..q_end],
            k_cache_reference,
            v_cache_reference,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            q_head,
            value_index,
        )
        .ok_or_else(|| {
            "runtime cached prefix attention failed to build sampled reference".to_string()
        })?;
        let actual = output[q_head * value_dim + value_index];
        max_abs_diff = max_abs_diff.max((actual - expected).abs());
        sample_count += 1;
    }
    if output_preview.is_empty() {
        let last_output = if let Some(output_sequence) = chunked_output_sequence.as_ref() {
            let output_start = (new_tokens - 1)
                .checked_mul(output_elements)
                .ok_or_else(|| {
                    "runtime cached prefix attention last output start overflows".to_string()
                })?;
            let output_end = output_start.checked_add(output_elements).ok_or_else(|| {
                "runtime cached prefix attention last output end overflows".to_string()
            })?;
            output_sequence[output_start..output_end].to_vec()
        } else {
            read_runtime_buffer_f32(
                &output_buffers[new_tokens - 1],
                &mut stream,
                output_elements,
                "runtime cached prefix attention last output",
            )?
        };
        output_preview = last_output[..last_output.len().min(8)].to_vec();
    }
    if max_abs_diff > 2e-4_f32 {
        return Err(format!(
            "runtime cached prefix attention sampled guard failed: max_abs_diff={max_abs_diff:.9}"
        ));
    }

    let cache_kv_elements_total = k_cache_elements
        .checked_add(v_cache_elements)
        .ok_or_else(|| "runtime cached prefix attention kv byte count overflows".to_string())?;
    let cache_kv_bytes_total = match kv_cache_dtype {
        RuntimeCachedPrefixKvCacheDtype::F32 => checked_f32_byte_len(
            cache_kv_elements_total,
            "runtime cached prefix attention total kv cache",
        )?,
        RuntimeCachedPrefixKvCacheDtype::Fp8E4m3 => cache_kv_elements_total,
    };
    let q_bytes_total = if executor_uses_fp8_q {
        q_sequence_elements
    } else {
        checked_f32_byte_len(
            q_sequence_elements,
            "runtime cached prefix attention total q sequence",
        )?
    };
    let output_bytes_total = checked_f32_byte_len(
        new_tokens
            .checked_mul(output_elements)
            .ok_or_else(|| "runtime cached prefix attention total output overflows".to_string())?,
        "runtime cached prefix attention total output",
    )?;
    let attention_pair_tps = if wall_ms > 0.0 {
        Some((estimated_attention_pairs as f64) / wall_ms * 1000.0)
    } else {
        None
    };
    let dispatch_format_id = dispatch_selection.request_format_id.unwrap_or("none");
    let dispatch_gpu_arch = dispatch_selection.request_gpu_arch.unwrap_or("unknown");

    Ok(format!(
        "runtime-cached-prefix-attn-smoke backend={} device_index={} name=\"{}\" prefill_mode=cached_prefix executor={} resolved_executor={} executor_selection={} selected_implementation_id={} dispatch_operation={} dispatch_phase={} dispatch_format_id={} dispatch_gpu_arch={} kv_cache_dtype={} cached_prefix_tokens={} new_prefill_tokens={} total_context_tokens_after_prefill={} q_heads={} kv_heads={} head_dim={} value_dim={} softmax_scale={softmax_scale:.9} q_sequence_scale={q_sequence_scale:.9} k_cache_scale={k_cache_scale:.9} v_cache_scale={v_cache_scale:.9} estimated_prefill_attention_work_tokens={} cache_kv_bytes_total={} q_bytes_total={} output_bytes_total={} warmup_runs=1 measured_repeats={} wall_ms_mean={:.6} wall_ms_min={:.6} wall_ms_max={:.6} prefill_total_input_tps={} attention_pair_tps_mean={} verification=sampled sample_count={} sampled_max_abs_diff={max_abs_diff:.9} output_preview={} verified=true",
        info.backend,
        device_index,
        info.name,
        executor.label(),
        resolved_executor,
        dispatch_selection.source,
        dispatch_selection.implementation_id,
        RUNTIME_CACHED_PREFIX_DISPATCH_OPERATION,
        RUNTIME_CACHED_PREFIX_DISPATCH_PHASE,
        dispatch_format_id,
        dispatch_gpu_arch,
        kv_cache_dtype.label(),
        cached_prefix_tokens,
        new_tokens,
        total_context_tokens,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        estimated_attention_pairs,
        cache_kv_bytes_total,
        q_bytes_total,
        output_bytes_total,
        measured_repeats,
        wall_ms,
        wall_ms_min,
        wall_ms_max,
        tps(new_tokens, wall_ms)
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "null".to_string()),
        attention_pair_tps
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "null".to_string()),
        sample_count,
        format_f32_preview(&output_preview),
    ))
}

fn synthetic_attention_f32(index: usize, salt: usize) -> f32 {
    let value = index.wrapping_mul(37).wrapping_add(salt.wrapping_mul(101)) % 1024;
    ((value as f32) / 512.0 - 1.0) * 0.25
}

struct Fp8E4m3Quantized {
    encoded: Vec<u8>,
    scale: f32,
    decoded: Vec<f32>,
}

fn fp8_e4m3_to_f32_unscaled(value: u8) -> f32 {
    let sign = value >> 7;
    let exponent = (value >> 3) & 0x0f;
    let mantissa = value & 0x07;
    let magnitude = if exponent == 0 {
        f32::from(mantissa) * 0.001953125
    } else {
        (1.0 + f32::from(mantissa) * 0.125) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    if sign == 0 { magnitude } else { -magnitude }
}

fn fp8_e4m3_encode_scaled(value: f32, scale: f32) -> u8 {
    if value == 0.0 || !value.is_finite() {
        return 0;
    }
    let sign: u8 = if value.is_sign_negative() { 0x80 } else { 0x00 };
    let magnitude = (value.abs() / scale).min(240.0);
    if magnitude < 0.001953125 {
        return 0;
    }
    if magnitude < 0.015625 {
        let mantissa = (magnitude / 0.001953125).round().clamp(0.0, 7.0) as u8;
        if mantissa == 0 {
            return 0;
        }
        return sign | mantissa;
    }
    let mut exponent = magnitude.log2().floor() as i32;
    let mut mantissa = ((magnitude / 2.0_f32.powi(exponent) - 1.0) * 8.0).round() as i32;
    if mantissa == 8 {
        exponent += 1;
        mantissa = 0;
    }
    if exponent > 7 {
        return sign | 0x77;
    }
    let biased_exponent = (exponent + 7).clamp(1, 14) as u8;
    sign | (biased_exponent << 3) | (mantissa.clamp(0, 7) as u8)
}

fn fp8_e4m3_quantize(values: &[f32]) -> Fp8E4m3Quantized {
    let max_abs = values.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 240.0 };
    let encoded = values
        .iter()
        .copied()
        .map(|value| fp8_e4m3_encode_scaled(value, scale))
        .collect::<Vec<_>>();
    let decoded = encoded
        .iter()
        .copied()
        .map(|value| fp8_e4m3_to_f32_unscaled(value) * scale)
        .collect::<Vec<_>>();
    Fp8E4m3Quantized {
        encoded,
        scale,
        decoded,
    }
}

fn cached_prefix_attention_pairs(cached_prefix_tokens: usize, new_tokens: usize) -> Option<u128> {
    let prefix = (cached_prefix_tokens as u128).checked_mul(new_tokens as u128)?;
    let chunk = (new_tokens as u128).checked_mul(new_tokens.checked_add(1)? as u128)? / 2;
    prefix.checked_add(chunk)
}

fn cached_prefix_sample_steps(new_tokens: usize) -> Vec<usize> {
    let mut steps = vec![0, new_tokens / 2, new_tokens - 1];
    steps.sort_unstable();
    steps.dedup();
    steps
}

#[allow(clippy::too_many_arguments)]
fn runtime_host_decode_attn_f32_sample(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    cache_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    q_head: usize,
    value_index: usize,
) -> Option<f32> {
    if cache_len == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || value_dim == 0
        || q_head >= q_heads
        || value_index >= value_dim
        || !q_heads.is_multiple_of(kv_heads)
        || q.len() != q_heads * head_dim
        || k_cache.len() < cache_len * kv_heads * head_dim
        || v_cache.len() < cache_len * kv_heads * value_dim
    {
        return None;
    }
    let q_per_kv = q_heads / kv_heads;
    let kv_head = q_head / q_per_kv;
    let q_base = q_head * head_dim;
    let mut scores = Vec::with_capacity(cache_len);
    for source_timestep in 0..cache_len {
        let k_base = (source_timestep * kv_heads + kv_head) * head_dim;
        let score = (0..head_dim)
            .map(|dim| q[q_base + dim] * k_cache[k_base + dim])
            .sum::<f32>()
            * softmax_scale;
        scores.push(score);
    }
    let max_score = scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |max, score| max.max(score));
    let mut denominator = 0.0_f32;
    let mut weighted = 0.0_f32;
    for (source_timestep, score) in scores.iter().copied().enumerate() {
        let weight = (score - max_score).exp();
        let v_index = (source_timestep * kv_heads + kv_head) * value_dim + value_index;
        denominator += weight;
        weighted += weight * v_cache[v_index];
    }
    Some(weighted / denominator)
}

fn runtime_paged_decode_attn_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let cache_len = 5_usize;
    let block_size = 2_usize;
    let cache_blocks = 4_usize;
    let q_heads = 4_usize;
    let kv_heads = 2_usize;
    let head_dim = 3_usize;
    let value_dim = 2_usize;
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q = (0..q_heads * head_dim)
        .map(|index| (index as f32 - 8.0) / 11.0)
        .collect::<Vec<_>>();
    let k = (0..cache_blocks * block_size * kv_heads * head_dim)
        .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
        .collect::<Vec<_>>();
    let v = (0..cache_blocks * block_size * kv_heads * value_dim)
        .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
        .collect::<Vec<_>>();
    let block_table = vec![2_u32, 0_u32, 3_u32];
    let expected = runtime_host_paged_decode_attn_f32(
        &q,
        &k,
        &v,
        &block_table,
        cache_len,
        block_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let q_bytes = encode_f32_to_bytes(&q);
    let k_bytes = encode_f32_to_bytes(&k);
    let v_bytes = encode_f32_to_bytes(&v);
    let block_table_bytes = encode_u32_to_bytes(&block_table);
    let output_bytes = expected.len() * std::mem::size_of::<f32>();

    let mut q_buffer = match context.alloc_buffer(q_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged decode attention q runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut k_buffer = match context.alloc_buffer(k_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged decode attention k cache runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_buffer = match context.alloc_buffer(v_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged decode attention v cache runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut block_table_buffer = match context.alloc_buffer(block_table_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!(
                "failed to allocate paged decode attention block table runtime buffer: {err}"
            );
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged decode attention output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = q_buffer.copy_from_host(0, &q_bytes, Some(&mut stream)) {
        eprintln!("failed to copy paged decode attention q input: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = k_buffer.copy_from_host(0, &k_bytes, Some(&mut stream)) {
        eprintln!("failed to copy paged decode attention k cache: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_buffer.copy_from_host(0, &v_bytes, Some(&mut stream)) {
        eprintln!("failed to copy paged decode attention v cache: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = block_table_buffer.copy_from_host(0, &block_table_bytes, Some(&mut stream)) {
        eprintln!("failed to copy paged decode attention block table: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!(
            "failed to synchronize runtime stream after paged decode attention input copies: {err}"
        );
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::paged_decode_attn_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        &block_table_buffer,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime paged_decode_attn_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after paged_decode_attn_f32: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy paged_decode_attn_f32 result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!(
            "failed to synchronize runtime stream after paged decode attention output copy: {err}"
        );
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);
    let tolerance = if info.backend == "cpu" {
        1e-5_f32
    } else {
        1e-4_f32
    };
    let max_abs_diff = match verify_f32_close(
        "runtime paged decode attention smoke",
        &output,
        &expected,
        tolerance,
        tolerance,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}; output={output:?} expected={expected:?}");
            return ExitCode::from(1);
        }
    };

    println!(
        "runtime-paged-decode-attn-smoke backend={} device_index={} name=\"{}\" cache_len={} block_size={} cache_blocks={} block_table={:?} q_heads={} kv_heads={} head_dim={} value_dim={} softmax_scale={softmax_scale:.9} output={} max_abs_diff={max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        cache_len,
        block_size,
        cache_blocks,
        block_table,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_paged_kv_write_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let cache_len = 3_usize;
    let block_size = 2_usize;
    let scheduled = match allocate_fragmented_paged_decode_blocks(cache_len, block_size) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let ScheduledPagedDecodeBlocks {
        block_table,
        cache_blocks,
        allocator_stats: stats,
        request_id: scheduler_request_id,
        prefill_tokens: scheduler_prefill_tokens,
        max_new_tokens: scheduler_max_new_tokens,
        cached_tokens: scheduler_cached_tokens,
        generated_tokens: scheduler_generated_tokens,
        active_len: scheduler_active_len,
    } = scheduled;
    let kv_heads = 2_usize;
    let head_dim = 3_usize;
    let value_dim = 2_usize;
    let logical_k = (0..cache_len * kv_heads * head_dim)
        .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
        .collect::<Vec<_>>();
    let logical_v = (0..cache_len * kv_heads * value_dim)
        .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
        .collect::<Vec<_>>();
    let shape = PagedDecodeShape {
        block_size,
        cache_blocks,
        q_heads: kv_heads,
        kv_heads,
        head_dim,
        value_dim,
    };
    let PagedKvCacheReadback {
        k: expected_k_cache,
        v: expected_v_cache,
    } = match pack_paged_kv_cache_for_block_table(
        &logical_k,
        &logical_v,
        &block_table,
        cache_len,
        shape,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let k_token_bytes = kv_heads * head_dim * std::mem::size_of::<f32>();
    let v_token_bytes = kv_heads * value_dim * std::mem::size_of::<f32>();
    let k_cache_bytes = expected_k_cache.len() * std::mem::size_of::<f32>();
    let v_cache_bytes = expected_v_cache.len() * std::mem::size_of::<f32>();
    let block_table_bytes = encode_u32_to_bytes(&block_table);
    let zero_k_cache = vec![0_u8; k_cache_bytes];
    let zero_v_cache = vec![0_u8; v_cache_bytes];

    let mut k_token_buffer = match context.alloc_buffer(k_token_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged KV write k token buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_token_buffer = match context.alloc_buffer(v_token_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged KV write v token buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut block_table_buffer = match context.alloc_buffer(block_table_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged KV write block table buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut k_cache_buffer = match context.alloc_buffer(k_cache_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged KV write k cache buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_cache_buffer = match context.alloc_buffer(v_cache_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged KV write v cache buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = block_table_buffer.copy_from_host(0, &block_table_bytes, Some(&mut stream)) {
        eprintln!("failed to copy paged KV write block table: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = k_cache_buffer.copy_from_host(0, &zero_k_cache, Some(&mut stream)) {
        eprintln!("failed to initialize paged KV write k cache: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_cache_buffer.copy_from_host(0, &zero_v_cache, Some(&mut stream)) {
        eprintln!("failed to initialize paged KV write v cache: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize paged KV write initial copies: {err}");
        return ExitCode::from(1);
    }

    for timestep in 0..cache_len {
        let k_start = timestep * kv_heads * head_dim;
        let k_end = k_start + kv_heads * head_dim;
        let v_start = timestep * kv_heads * value_dim;
        let v_end = v_start + kv_heads * value_dim;
        let k_bytes = encode_f32_to_bytes(&logical_k[k_start..k_end]);
        let v_bytes = encode_f32_to_bytes(&logical_v[v_start..v_end]);
        if let Err(err) = k_token_buffer.copy_from_host(0, &k_bytes, Some(&mut stream)) {
            eprintln!("failed to copy paged KV write timestep {timestep} k: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = v_token_buffer.copy_from_host(0, &v_bytes, Some(&mut stream)) {
            eprintln!("failed to copy paged KV write timestep {timestep} v: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize paged KV write timestep {timestep} inputs: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::paged_kv_write_f32(
            &k_token_buffer,
            &v_token_buffer,
            &block_table_buffer,
            timestep,
            block_size,
            cache_blocks,
            kv_heads,
            head_dim,
            value_dim,
            &mut k_cache_buffer,
            &mut v_cache_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run paged_kv_write_f32 for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize paged KV write timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
    }

    let mut k_cache_raw = vec![0_u8; k_cache_bytes];
    let mut v_cache_raw = vec![0_u8; v_cache_bytes];
    if let Err(err) = k_cache_buffer.copy_to_host(0, &mut k_cache_raw, Some(&mut stream)) {
        eprintln!("failed to copy paged KV write k cache back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_cache_buffer.copy_to_host(0, &mut v_cache_raw, Some(&mut stream)) {
        eprintln!("failed to copy paged KV write v cache back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize paged KV write readback: {err}");
        return ExitCode::from(1);
    }
    let k_cache = decode_f32_le_values(&k_cache_raw);
    let v_cache = decode_f32_le_values(&v_cache_raw);
    let k_max_abs_diff = match verify_f32_close(
        "runtime paged KV write k cache",
        &k_cache,
        &expected_k_cache,
        1e-5,
        1e-5,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let v_max_abs_diff = match verify_f32_close(
        "runtime paged KV write v cache",
        &v_cache,
        &expected_v_cache,
        1e-5,
        1e-5,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    println!(
        "runtime-paged-kv-write-smoke backend={} device_index={} name=\"{}\" cache_len={} block_size={} cache_blocks={} block_table={:?} scheduler_request_id={} scheduler_prefill_tokens={} scheduler_max_new_tokens={} scheduler_cached_tokens={} scheduler_generated_tokens={} scheduler_active_len={} free_blocks={} allocated_block_count={} free_runs={} largest_free_run={} kv_heads={} head_dim={} value_dim={} k_cache_preview={} v_cache_preview={} k_max_abs_diff={k_max_abs_diff:.9} v_max_abs_diff={v_max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        cache_len,
        block_size,
        cache_blocks,
        block_table,
        scheduler_request_id.0,
        scheduler_prefill_tokens,
        scheduler_max_new_tokens,
        scheduler_cached_tokens,
        scheduler_generated_tokens,
        scheduler_active_len,
        stats.free_blocks,
        stats.allocated_blocks,
        stats.free_runs,
        stats.largest_free_run,
        kv_heads,
        head_dim,
        value_dim,
        format_f32_preview(&k_cache[..8.min(k_cache.len())]),
        format_f32_preview(&v_cache[..8.min(v_cache.len())]),
    );
    ExitCode::SUCCESS
}

struct SyntheticSchedulerPagedDecodeRun {
    request_id: RequestId,
    prompt_tokens: usize,
    max_new_tokens: usize,
    total_tokens: usize,
    block_table: Vec<u32>,
    q_sequence: Vec<f32>,
    k_sequence: Vec<f32>,
    v_sequence: Vec<f32>,
    expected_k_cache: Vec<f32>,
    expected_v_cache: Vec<f32>,
    decode_steps: usize,
    attention_max_abs_diff: f32,
    k_cache_max_abs_diff: f32,
    v_cache_max_abs_diff: f32,
}

struct SchedulerLayerDecodeState {
    request_id: RequestId,
    prompt_tokens: usize,
    max_new_tokens: usize,
    total_tokens: usize,
    block_table: Vec<u32>,
    q_sequence: Vec<f32>,
    k_sequence: Vec<f32>,
    v_sequence: Vec<f32>,
    output_gate_sequence: Option<Vec<f32>>,
    residual_sequence: Vec<f32>,
    decode_steps: usize,
}

struct SchedulerLayerDecodeRun {
    state: SchedulerLayerDecodeState,
    checks: SchedulerLayerDecodeSmokeChecks,
}

impl std::ops::Deref for SchedulerLayerDecodeRun {
    type Target = SchedulerLayerDecodeState;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl std::ops::DerefMut for SchedulerLayerDecodeRun {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

struct SchedulerLayerDecodeSmokeChecks {
    expected: Qwen3DecoderLayerSequenceOutput,
    attention_max_abs_diff: f32,
    projection_input_max_abs_diff: f32,
    projected_max_abs_diff: f32,
    block_max_abs_diff: f32,
    post_norm_max_abs_diff: f32,
    mlp_max_abs_diff: f32,
    layer_max_abs_diff: f32,
    k_cache_max_abs_diff: f32,
    v_cache_max_abs_diff: f32,
}

impl SchedulerLayerDecodeSmokeChecks {
    fn new(expected: Qwen3DecoderLayerSequenceOutput) -> Self {
        Self {
            expected,
            attention_max_abs_diff: 0.0,
            projection_input_max_abs_diff: 0.0,
            projected_max_abs_diff: 0.0,
            block_max_abs_diff: 0.0,
            post_norm_max_abs_diff: 0.0,
            mlp_max_abs_diff: 0.0,
            layer_max_abs_diff: 0.0,
            k_cache_max_abs_diff: 0.0,
            v_cache_max_abs_diff: 0.0,
        }
    }
}

fn synthetic_scheduler_decode_values(
    request_index: usize,
    total_tokens: usize,
    token_elements: usize,
    salt: usize,
) -> Vec<f32> {
    let mut values = Vec::with_capacity(total_tokens * token_elements);
    for token in 0..total_tokens {
        for element in 0..token_elements {
            let angle = (request_index as f32 + 1.0_f32) * 0.173_f32
                + token as f32 * 0.119_f32
                + element as f32 * 0.037_f32
                + salt as f32 * 0.071_f32;
            values.push(angle.sin() * 0.25_f32 + angle.cos() * 0.05_f32);
        }
    }
    values
}

fn scheduler_layer_decode_run(
    runs: &[SchedulerLayerDecodeRun],
    request_id: RequestId,
) -> Option<&SchedulerLayerDecodeRun> {
    runs.iter().find(|run| run.request_id == request_id)
}

fn scheduler_layer_decode_run_mut(
    runs: &mut [SchedulerLayerDecodeRun],
    request_id: RequestId,
) -> Option<&mut SchedulerLayerDecodeRun> {
    runs.iter_mut().find(|run| run.request_id == request_id)
}

fn scheduler_layer_decode_sequence_view(
    run: &SchedulerLayerDecodeRun,
) -> Qwen3DecoderLayerDecodeSequenceView<'_> {
    Qwen3DecoderLayerDecodeSequenceView {
        request_id: run.request_id,
        q_sequence: &run.q_sequence,
        k_sequence: &run.k_sequence,
        v_sequence: &run.v_sequence,
        output_gate_sequence: run.output_gate_sequence.as_deref(),
        residual_sequence: &run.residual_sequence,
    }
}

fn runtime_f32_buffer_from_values(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    label: &str,
    values: &[f32],
) -> Result<ullm_runtime_sys::RuntimeBuffer, String> {
    let mut buffer = context
        .alloc_buffer(
            values
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| format!("{label} byte size overflows"))?,
        )
        .map_err(|err| format!("failed to allocate {label}: {err}"))?;
    buffer
        .copy_from_host(0, &encode_f32_to_bytes(values), Some(stream))
        .map_err(|err| format!("failed to copy {label}: {err}"))?;
    Ok(buffer)
}

fn synthetic_scheduler_decoder_layer_weights(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    shape: PagedDecodeShape,
    hidden: usize,
    intermediate: usize,
) -> Result<Qwen3DecoderLayerRuntimeWeights, String> {
    if hidden == 0 || intermediate == 0 {
        return Err("synthetic decoder layer hidden/intermediate must be nonzero".to_string());
    }
    let q_rows = shape.q_elements()?;
    let k_rows = shape.k_token_elements()?;
    let v_rows = shape.v_token_elements()?;
    let attention_elements = shape.output_elements()?;
    let q_matrix = synthetic_scheduler_decode_values(0, q_rows, hidden, 11);
    let k_matrix = synthetic_scheduler_decode_values(0, k_rows, hidden, 12);
    let v_matrix = synthetic_scheduler_decode_values(0, v_rows, hidden, 13);
    let o_matrix = synthetic_scheduler_decode_values(0, hidden, attention_elements, 14);
    let post_norm_weight = (0..hidden)
        .map(|index| 0.75_f32 + index as f32 * 0.03125_f32)
        .collect::<Vec<_>>();
    let mlp_gate_matrix = synthetic_scheduler_decode_values(0, intermediate, hidden, 15);
    let mlp_up_matrix = synthetic_scheduler_decode_values(0, intermediate, hidden, 16);
    let mlp_down_matrix = synthetic_scheduler_decode_values(0, hidden, intermediate, 17);

    let weights = Qwen3DecoderLayerRuntimeWeights {
        self_attn: Qwen3SelfAttnRuntimeWeights {
            q_rows,
            q_cols: hidden,
            k_rows,
            v_rows,
            o_rows: hidden,
            o_cols: attention_elements,
            head_dim: shape.head_dim,
            kv_heads: shape.kv_heads,
            value_dim: shape.value_dim,
            q_matrix: runtime_f32_buffer_from_values(
                context,
                stream,
                "synthetic Qwen3 decoder layer q matrix",
                &q_matrix,
            )?,
            k_matrix: runtime_f32_buffer_from_values(
                context,
                stream,
                "synthetic Qwen3 decoder layer k matrix",
                &k_matrix,
            )?,
            v_matrix: runtime_f32_buffer_from_values(
                context,
                stream,
                "synthetic Qwen3 decoder layer v matrix",
                &v_matrix,
            )?,
            o_matrix: runtime_f32_buffer_from_values(
                context,
                stream,
                "synthetic Qwen3 decoder layer o matrix",
                &o_matrix,
            )?,
        },
        post_attention: Qwen3PostAttentionRuntimeWeights {
            hidden,
            intermediate,
            post_norm_weight: runtime_f32_buffer_from_values(
                context,
                stream,
                "synthetic Qwen3 decoder layer post norm weight",
                &post_norm_weight,
            )?,
            mlp: Qwen3MlpRuntimeWeights {
                gate_rows: intermediate,
                gate_cols: hidden,
                gate_matrix: runtime_f32_buffer_from_values(
                    context,
                    stream,
                    "synthetic Qwen3 decoder layer MLP gate matrix",
                    &mlp_gate_matrix,
                )?,
                up_matrix: runtime_f32_buffer_from_values(
                    context,
                    stream,
                    "synthetic Qwen3 decoder layer MLP up matrix",
                    &mlp_up_matrix,
                )?,
                down_matrix: runtime_f32_buffer_from_values(
                    context,
                    stream,
                    "synthetic Qwen3 decoder layer MLP down matrix",
                    &mlp_down_matrix,
                )?,
            },
        },
    };
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize synthetic decoder layer weights: {err}"))?;
    Ok(weights)
}

fn synthetic_scheduler_decode_run(
    runs: &[SyntheticSchedulerPagedDecodeRun],
    request_id: RequestId,
) -> Option<&SyntheticSchedulerPagedDecodeRun> {
    runs.iter().find(|run| run.request_id == request_id)
}

fn synthetic_scheduler_decode_run_mut(
    runs: &mut [SyntheticSchedulerPagedDecodeRun],
    request_id: RequestId,
) -> Option<&mut SyntheticSchedulerPagedDecodeRun> {
    runs.iter_mut().find(|run| run.request_id == request_id)
}

#[allow(clippy::too_many_arguments)]
fn run_synthetic_scheduler_decode_step(
    runner: &mut Qwen3SelfAttnRequestDecodeRunner,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    run: &mut SyntheticSchedulerPagedDecodeRun,
    timestep: usize,
    block_size: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    label: &str,
) -> Result<(), String> {
    let q_token_elements = q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| format!("{label} q token element count overflows"))?;
    let k_token_elements = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| format!("{label} k token element count overflows"))?;
    let v_token_elements = kv_heads
        .checked_mul(value_dim)
        .ok_or_else(|| format!("{label} v token element count overflows"))?;
    let q_start = timestep
        .checked_mul(q_token_elements)
        .ok_or_else(|| format!("{label} q slice start overflows"))?;
    let q_end = q_start
        .checked_add(q_token_elements)
        .ok_or_else(|| format!("{label} q slice end overflows"))?;
    let k_start = timestep
        .checked_mul(k_token_elements)
        .ok_or_else(|| format!("{label} k slice start overflows"))?;
    let k_end = k_start
        .checked_add(k_token_elements)
        .ok_or_else(|| format!("{label} k slice end overflows"))?;
    let v_start = timestep
        .checked_mul(v_token_elements)
        .ok_or_else(|| format!("{label} v slice start overflows"))?;
    let v_end = v_start
        .checked_add(v_token_elements)
        .ok_or_else(|| format!("{label} v slice end overflows"))?;

    let step = runner
        .run_prefill_step(
            stream,
            Qwen3SelfAttnDecodeBatchInput {
                request_id: run.request_id,
                q: &run.q_sequence[q_start..q_end],
                k: &run.k_sequence[k_start..k_end],
                v: &run.v_sequence[v_start..v_end],
            },
        )
        .map_err(|err| {
            format!(
                "{label} failed to run request {:?} timestep {timestep}: {err}",
                run.request_id
            )
        })?;
    if step.cache_position != timestep {
        return Err(format!(
            "{label} request {:?} wrote cache position {}, expected {timestep}",
            run.request_id, step.cache_position
        ));
    }
    if step.cache_len != timestep + 1 {
        return Err(format!(
            "{label} request {:?} reported cache_len {}, expected {}",
            run.request_id,
            step.cache_len,
            timestep + 1
        ));
    }

    let expected = runtime_host_paged_decode_attn_f32(
        &run.q_sequence[q_start..q_end],
        &run.expected_k_cache,
        &run.expected_v_cache,
        &run.block_table,
        timestep + 1,
        block_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let max_abs_diff = verify_f32_close(
        &format!("{label} request {:?} timestep {timestep}", run.request_id),
        &step.attention_output,
        &expected,
        1e-4,
        1e-4,
    )?;
    run.attention_max_abs_diff = run.attention_max_abs_diff.max(max_abs_diff);
    Ok(())
}

fn synthetic_layer_expected_slice<'a>(
    values: &'a [f32],
    timestep: usize,
    elements: usize,
    label: &str,
) -> Result<&'a [f32], String> {
    let start = timestep
        .checked_mul(elements)
        .ok_or_else(|| format!("{label} slice start overflows"))?;
    let end = start
        .checked_add(elements)
        .ok_or_else(|| format!("{label} slice end overflows"))?;
    values
        .get(start..end)
        .ok_or_else(|| format!("{label} slice {start}..{end} is out of bounds"))
}

fn verify_scheduler_layer_step_output(
    label: &str,
    run: &mut SchedulerLayerDecodeRun,
    step: &ullm_engine::decode_runner::Qwen3DecoderLayerDecodeBatchOutput,
    hidden: usize,
    attention_elements: usize,
) -> Result<(), String> {
    let timestep = step.cache_position;
    if step.cache_len != timestep + 1 {
        return Err(format!(
            "{label} request {:?} cache_len {} did not match timestep + 1 ({})",
            run.request_id,
            step.cache_len,
            timestep + 1
        ));
    }
    let attention_expected = synthetic_layer_expected_slice(
        &run.checks.expected.attention_output,
        timestep,
        attention_elements,
        label,
    )?;
    let projection_input_expected = synthetic_layer_expected_slice(
        &run.checks.expected.attention_projection_input,
        timestep,
        attention_elements,
        label,
    )?;
    let projected_expected = synthetic_layer_expected_slice(
        &run.checks.expected.projected_output,
        timestep,
        hidden,
        label,
    )?;
    let block_expected =
        synthetic_layer_expected_slice(&run.checks.expected.block_output, timestep, hidden, label)?;
    let post_norm_expected =
        synthetic_layer_expected_slice(&run.checks.expected.post_normed, timestep, hidden, label)?;
    let mlp_expected =
        synthetic_layer_expected_slice(&run.checks.expected.mlp_output, timestep, hidden, label)?;
    let layer_expected =
        synthetic_layer_expected_slice(&run.checks.expected.layer_output, timestep, hidden, label)?;

    run.checks.attention_max_abs_diff = run.checks.attention_max_abs_diff.max(verify_f32_close(
        &format!(
            "{label} request {:?} attention timestep {timestep}",
            run.request_id
        ),
        &step.attention_output,
        attention_expected,
        1e-4,
        1e-4,
    )?);
    run.checks.projection_input_max_abs_diff =
        run.checks
            .projection_input_max_abs_diff
            .max(verify_f32_close(
                &format!(
                    "{label} request {:?} projection input timestep {timestep}",
                    run.request_id
                ),
                &step.attention_projection_input,
                projection_input_expected,
                1e-4,
                1e-4,
            )?);
    run.checks.projected_max_abs_diff = run.checks.projected_max_abs_diff.max(verify_f32_close(
        &format!(
            "{label} request {:?} projected timestep {timestep}",
            run.request_id
        ),
        &step.projected_output,
        projected_expected,
        1e-4,
        1e-4,
    )?);
    run.checks.block_max_abs_diff = run.checks.block_max_abs_diff.max(verify_f32_close(
        &format!(
            "{label} request {:?} block timestep {timestep}",
            run.request_id
        ),
        &step.block_output,
        block_expected,
        1e-4,
        1e-4,
    )?);
    run.checks.post_norm_max_abs_diff = run.checks.post_norm_max_abs_diff.max(verify_f32_close(
        &format!(
            "{label} request {:?} post norm timestep {timestep}",
            run.request_id
        ),
        &step.post_normed,
        post_norm_expected,
        1e-4,
        1e-4,
    )?);
    run.checks.mlp_max_abs_diff = run.checks.mlp_max_abs_diff.max(verify_f32_close(
        &format!(
            "{label} request {:?} MLP timestep {timestep}",
            run.request_id
        ),
        &step.mlp_output,
        mlp_expected,
        1e-4,
        1e-4,
    )?);
    run.checks.layer_max_abs_diff = run.checks.layer_max_abs_diff.max(verify_f32_close(
        &format!(
            "{label} request {:?} layer timestep {timestep}",
            run.request_id
        ),
        &step.layer_output,
        layer_expected,
        1e-4,
        1e-4,
    )?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_scheduler_layer_prefill_step(
    runner: &mut Qwen3DecoderLayerRequestDecodeRunner<'_>,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    run: &mut SchedulerLayerDecodeRun,
    timestep: usize,
    q_token_elements: usize,
    k_token_elements: usize,
    v_token_elements: usize,
    attention_elements: usize,
    hidden: usize,
    label: &str,
) -> Result<(), String> {
    let input_layout = Qwen3DecoderLayerDecodeInputLayout {
        q_token_elements,
        k_token_elements,
        v_token_elements,
        attention_elements,
        hidden,
    };
    let input = qwen3_decoder_layer_prefill_input_from_sequence(
        scheduler_layer_decode_sequence_view(run),
        timestep,
        input_layout,
        label,
    )?;
    let step = runner.run_prefill_step(stream, input)?;
    if step.cache_position != timestep {
        return Err(format!(
            "{label} request {:?} wrote cache_position {}, expected {timestep}",
            run.request_id, step.cache_position
        ));
    }
    verify_scheduler_layer_step_output(label, run, &step, hidden, attention_elements)
}

#[allow(clippy::too_many_arguments)]
fn run_scheduler_layer_stack_prefill_batch(
    runner: &mut Qwen3DecoderLayerStackRequestDecodeRunner<'_>,
    layer_index: usize,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    runs: &mut [SchedulerLayerDecodeRun],
    timestep: usize,
    decode: Qwen3PackageModelDecodePlan,
    label: &str,
) -> Result<usize, String> {
    let active_indices = runs
        .iter()
        .enumerate()
        .filter_map(|(run_index, run)| (timestep < run.prompt_tokens).then_some(run_index))
        .collect::<Vec<_>>();
    if active_indices.is_empty() {
        return Ok(0);
    }
    let sequences = active_indices
        .iter()
        .map(|run_index| scheduler_layer_decode_sequence_view(&runs[*run_index]))
        .collect::<Vec<_>>();
    let outputs = qwen3_package_model_run_prefill_batch_from_sequences(
        runner,
        layer_index,
        stream,
        &sequences,
        timestep,
        decode,
        label,
    )?;
    if outputs.len() != active_indices.len() {
        return Err(format!(
            "{label} prefill batch produced {} outputs, expected {}",
            outputs.len(),
            active_indices.len()
        ));
    }

    for output in outputs {
        let run = scheduler_layer_decode_run_mut(runs, output.request_id).ok_or_else(|| {
            format!(
                "{label} prefill batch returned unknown request {:?}",
                output.request_id
            )
        })?;
        if output.cache_position != timestep {
            return Err(format!(
                "{label} request {:?} wrote cache_position {}, expected {timestep}",
                output.request_id, output.cache_position
            ));
        }
        verify_scheduler_layer_step_output(
            label,
            run,
            &output,
            decode.hidden,
            decode.attention_elements,
        )?;
    }
    Ok(active_indices.len())
}

#[allow(clippy::too_many_arguments)]
fn run_synthetic_scheduler_ready_batch(
    runner: &mut Qwen3SelfAttnRequestDecodeRunner,
    scheduler: &mut SchedulerState,
    runs: &mut [SyntheticSchedulerPagedDecodeRun],
    stream: &mut ullm_runtime_sys::RuntimeStream,
    expected_ids: &[RequestId],
    max_requests: usize,
    block_size: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    label: &str,
) -> Result<usize, String> {
    let ready = scheduler
        .ready_decode_batch(max_requests)
        .map_err(|err| format!("{label} failed to prepare ready decode batch: {err}"))?;
    let ready_ids = ready
        .iter()
        .map(|request| request.request.id)
        .collect::<Vec<_>>();
    if ready_ids != expected_ids {
        return Err(format!(
            "{label} ready request ids {:?} did not match expected {:?}",
            ready_ids, expected_ids
        ));
    }

    for request in &ready {
        let run = synthetic_scheduler_decode_run(runs, request.request.id)
            .ok_or_else(|| format!("{label} request {:?} has no decode run", request.request.id))?;
        let expected_cache_position =
            run.prompt_tokens
                .checked_add(run.decode_steps)
                .ok_or_else(|| {
                    format!(
                        "{label} request {:?} cache position overflows",
                        run.request_id
                    )
                })?;
        if request.cache_position != expected_cache_position {
            return Err(format!(
                "{label} request {:?} cache_position {} did not match expected {}",
                request.request.id, request.cache_position, expected_cache_position
            ));
        }
        if request.next_cache_len != request.cache_position + 1 {
            return Err(format!(
                "{label} request {:?} next_cache_len {} did not match cache_position + 1",
                request.request.id, request.next_cache_len
            ));
        }
        let expected_remaining = run
            .max_new_tokens
            .checked_sub(run.decode_steps)
            .ok_or_else(|| format!("{label} request {:?} decode step overflow", run.request_id))?;
        if request.remaining_new_tokens != expected_remaining {
            return Err(format!(
                "{label} request {:?} remaining_new_tokens {} did not match expected {}",
                request.request.id, request.remaining_new_tokens, expected_remaining
            ));
        }
        if request.allocation.blocks != run.block_table {
            return Err(format!(
                "{label} request {:?} block table {:?} did not match run block table {:?}",
                request.request.id, request.allocation.blocks, run.block_table
            ));
        }
    }

    let outputs = {
        let q_token_elements = q_heads
            .checked_mul(head_dim)
            .ok_or_else(|| format!("{label} q token element count overflows"))?;
        let k_token_elements = kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| format!("{label} k token element count overflows"))?;
        let v_token_elements = kv_heads
            .checked_mul(value_dim)
            .ok_or_else(|| format!("{label} v token element count overflows"))?;
        let mut inputs = Vec::with_capacity(ready.len());
        for request in &ready {
            let run =
                synthetic_scheduler_decode_run(runs, request.request.id).ok_or_else(|| {
                    format!(
                        "{label} request {:?} disappeared while preparing decode input",
                        request.request.id
                    )
                })?;
            let q_start = request
                .cache_position
                .checked_mul(q_token_elements)
                .ok_or_else(|| format!("{label} q slice start overflows"))?;
            let q_end = q_start
                .checked_add(q_token_elements)
                .ok_or_else(|| format!("{label} q slice end overflows"))?;
            let k_start = request
                .cache_position
                .checked_mul(k_token_elements)
                .ok_or_else(|| format!("{label} k slice start overflows"))?;
            let k_end = k_start
                .checked_add(k_token_elements)
                .ok_or_else(|| format!("{label} k slice end overflows"))?;
            let v_start = request
                .cache_position
                .checked_mul(v_token_elements)
                .ok_or_else(|| format!("{label} v slice start overflows"))?;
            let v_end = v_start
                .checked_add(v_token_elements)
                .ok_or_else(|| format!("{label} v slice end overflows"))?;
            inputs.push(Qwen3SelfAttnDecodeBatchInput {
                request_id: request.request.id,
                q: &run.q_sequence[q_start..q_end],
                k: &run.k_sequence[k_start..k_end],
                v: &run.v_sequence[v_start..v_end],
            });
        }
        runner.run_ready_batch(stream, scheduler, &ready, &inputs)?
    };

    for output in outputs {
        let run = synthetic_scheduler_decode_run_mut(runs, output.request_id).ok_or_else(|| {
            format!(
                "{label} advanced request {:?} disappeared",
                output.request_id
            )
        })?;
        run.decode_steps = run.decode_steps.checked_add(1).ok_or_else(|| {
            format!(
                "{label} request {:?} decode step count overflows",
                run.request_id
            )
        })?;
        let q_token_elements = q_heads
            .checked_mul(head_dim)
            .ok_or_else(|| format!("{label} q token element count overflows"))?;
        let q_start = output
            .cache_position
            .checked_mul(q_token_elements)
            .ok_or_else(|| format!("{label} q expected slice start overflows"))?;
        let q_end = q_start
            .checked_add(q_token_elements)
            .ok_or_else(|| format!("{label} q expected slice end overflows"))?;
        let expected = runtime_host_paged_decode_attn_f32(
            &run.q_sequence[q_start..q_end],
            &run.expected_k_cache,
            &run.expected_v_cache,
            &run.block_table,
            output.cache_len,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );
        let max_abs_diff = verify_f32_close(
            &format!(
                "{label} request {:?} timestep {}",
                run.request_id, output.cache_position
            ),
            &output.attention_output,
            &expected,
            1e-4,
            1e-4,
        )?;
        run.attention_max_abs_diff = run.attention_max_abs_diff.max(max_abs_diff);
    }
    Ok(expected_ids.len())
}

#[allow(clippy::too_many_arguments)]
fn run_scheduler_layer_ready_batch(
    runner: &mut Qwen3DecoderLayerRequestDecodeRunner<'_>,
    scheduler: &mut SchedulerState,
    runs: &mut [SchedulerLayerDecodeRun],
    stream: &mut ullm_runtime_sys::RuntimeStream,
    expected_ids: &[RequestId],
    max_requests: usize,
    q_token_elements: usize,
    k_token_elements: usize,
    v_token_elements: usize,
    attention_elements: usize,
    hidden: usize,
    advance_scheduler: bool,
    label: &str,
) -> Result<usize, String> {
    let ready = scheduler
        .ready_decode_batch(max_requests)
        .map_err(|err| format!("{label} failed to prepare ready decode batch: {err}"))?;
    let ready_ids = ready
        .iter()
        .map(|request| request.request.id)
        .collect::<Vec<_>>();
    if ready_ids != expected_ids {
        return Err(format!(
            "{label} ready request ids {:?} did not match expected {:?}",
            ready_ids, expected_ids
        ));
    }

    for request in &ready {
        let run = scheduler_layer_decode_run(runs, request.request.id)
            .ok_or_else(|| format!("{label} request {:?} has no layer run", request.request.id))?;
        let expected_cache_position =
            run.prompt_tokens
                .checked_add(run.decode_steps)
                .ok_or_else(|| {
                    format!(
                        "{label} request {:?} cache position overflows",
                        run.request_id
                    )
                })?;
        if request.cache_position != expected_cache_position {
            return Err(format!(
                "{label} request {:?} cache_position {} did not match expected {}",
                request.request.id, request.cache_position, expected_cache_position
            ));
        }
        if request.next_cache_len != request.cache_position + 1 {
            return Err(format!(
                "{label} request {:?} next_cache_len {} did not match cache_position + 1",
                request.request.id, request.next_cache_len
            ));
        }
        let expected_remaining = run
            .max_new_tokens
            .checked_sub(run.decode_steps)
            .ok_or_else(|| format!("{label} request {:?} decode step overflow", run.request_id))?;
        if request.remaining_new_tokens != expected_remaining {
            return Err(format!(
                "{label} request {:?} remaining_new_tokens {} did not match expected {}",
                request.request.id, request.remaining_new_tokens, expected_remaining
            ));
        }
        if request.allocation.blocks != run.block_table {
            return Err(format!(
                "{label} request {:?} block table {:?} did not match run block table {:?}",
                request.request.id, request.allocation.blocks, run.block_table
            ));
        }
    }

    let outputs = {
        let input_layout = Qwen3DecoderLayerDecodeInputLayout {
            q_token_elements,
            k_token_elements,
            v_token_elements,
            attention_elements,
            hidden,
        };
        let sequences = runs
            .iter()
            .map(scheduler_layer_decode_sequence_view)
            .collect::<Vec<_>>();
        let inputs = qwen3_decoder_layer_decode_batch_inputs_from_sequences(
            &ready,
            &sequences,
            input_layout,
            label,
        )?;
        if advance_scheduler {
            runner.run_ready_batch(stream, scheduler, &ready, &inputs)?
        } else {
            runner.run_ready_batch_without_advance(stream, scheduler, &ready, &inputs)?
        }
    };

    for output in outputs {
        let run = scheduler_layer_decode_run_mut(runs, output.request_id).ok_or_else(|| {
            format!(
                "{label} advanced request {:?} disappeared",
                output.request_id
            )
        })?;
        run.decode_steps = run.decode_steps.checked_add(1).ok_or_else(|| {
            format!(
                "{label} request {:?} decode step count overflows",
                run.request_id
            )
        })?;
        verify_scheduler_layer_step_output(label, run, &output, hidden, attention_elements)?;
    }
    Ok(expected_ids.len())
}

#[allow(clippy::too_many_arguments)]
fn run_scheduler_layer_stack_ready_batch(
    runner: &mut Qwen3DecoderLayerStackRequestDecodeRunner<'_>,
    scheduler: &mut SchedulerState,
    runs_by_layer: &mut [Vec<SchedulerLayerDecodeRun>],
    stream: &mut ullm_runtime_sys::RuntimeStream,
    ready: &[SchedulerDecodeRequest],
    decode: Qwen3PackageModelDecodePlan,
    label: &str,
) -> Result<usize, String> {
    if ready.is_empty() {
        return Ok(0);
    }

    for (layer_position, runs) in runs_by_layer.iter().enumerate() {
        for request in ready {
            let run = scheduler_layer_decode_run(runs, request.request.id).ok_or_else(|| {
                format!(
                    "{label} layer {layer_position} request {:?} has no layer run",
                    request.request.id
                )
            })?;
            let expected_cache_position = run
                .prompt_tokens
                .checked_add(run.decode_steps)
                .ok_or_else(|| {
                    format!(
                        "{label} layer {layer_position} request {:?} cache position overflows",
                        run.request_id
                    )
                })?;
            if request.cache_position != expected_cache_position {
                return Err(format!(
                    "{label} layer {layer_position} request {:?} cache_position {} did not match expected {}",
                    request.request.id, request.cache_position, expected_cache_position
                ));
            }
            if request.next_cache_len != request.cache_position + 1 {
                return Err(format!(
                    "{label} layer {layer_position} request {:?} next_cache_len {} did not match cache_position + 1",
                    request.request.id, request.next_cache_len
                ));
            }
            let expected_remaining = run
                .max_new_tokens
                .checked_sub(run.decode_steps)
                .ok_or_else(|| {
                    format!(
                        "{label} layer {layer_position} request {:?} decode step overflow",
                        run.request_id
                    )
                })?;
            if request.remaining_new_tokens != expected_remaining {
                return Err(format!(
                    "{label} layer {layer_position} request {:?} remaining_new_tokens {} did not match expected {}",
                    request.request.id, request.remaining_new_tokens, expected_remaining
                ));
            }
            if request.allocation.blocks != run.block_table {
                return Err(format!(
                    "{label} layer {layer_position} request {:?} block table {:?} did not match run block table {:?}",
                    request.request.id, request.allocation.blocks, run.block_table
                ));
            }
        }
    }

    let layer_sequences = runs_by_layer
        .iter()
        .map(|runs| {
            runs.iter()
                .map(scheduler_layer_decode_sequence_view)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let layer_sequence_refs = layer_sequences
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let outputs_by_layer = qwen3_package_model_run_ready_batch_from_sequences(
        runner,
        stream,
        scheduler,
        ready,
        decode,
        &layer_sequence_refs,
        label,
    )?;
    drop(layer_sequence_refs);
    drop(layer_sequences);

    for (layer_position, outputs) in outputs_by_layer.into_iter().enumerate() {
        let runs = runs_by_layer
            .get_mut(layer_position)
            .ok_or_else(|| format!("{label} layer {layer_position} output has no run list"))?;
        for output in outputs {
            let run = scheduler_layer_decode_run_mut(runs, output.request_id).ok_or_else(|| {
                format!(
                    "{label} layer {layer_position} advanced request {:?} disappeared",
                    output.request_id
                )
            })?;
            run.decode_steps = run.decode_steps.checked_add(1).ok_or_else(|| {
                format!(
                    "{label} layer {layer_position} request {:?} decode step count overflows",
                    run.request_id
                )
            })?;
            verify_scheduler_layer_step_output(
                label,
                run,
                &output,
                decode.hidden,
                decode.attention_elements,
            )?;
        }
    }
    Ok(ready.len())
}

fn runtime_scheduler_paged_decode_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    match runtime_scheduler_paged_decode_smoke_impl(device_index) {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn runtime_scheduler_paged_decode_smoke_impl(device_index: u32) -> Result<String, String> {
    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let block_size = 2_usize;
    let cache_blocks = 8_usize;
    let q_heads = 4_usize;
    let kv_heads = 2_usize;
    let head_dim = 3_usize;
    let value_dim = 2_usize;
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let requests = vec![
        Request::new(201, 2, 2),
        Request::new(202, 3, 1),
        Request::new(203, 1, 0),
    ];

    let mut scheduler = SchedulerState::with_block_size(cache_blocks as u32, block_size as u32);
    for request in &requests {
        scheduler.enqueue(request.clone());
    }
    let mut decode_runner = Qwen3SelfAttnRequestDecodeRunner::new();
    let mut allocated = scheduler
        .pop_prefill_batch_with_allocation(usize::MAX)
        .map_err(|err| format!("failed to allocate synthetic scheduler decode batch: {err}"))?;
    if allocated.len() != requests.len() {
        return Err(format!(
            "synthetic scheduler decode selected {} requests, expected {}",
            allocated.len(),
            requests.len()
        ));
    }

    let mut runs = Vec::with_capacity(allocated.len());
    for (request_index, scheduled) in allocated.drain(..).enumerate() {
        let request = scheduled.request;
        let block_table = scheduled.allocation.blocks;
        let total_tokens = request
            .prompt_tokens
            .checked_add(request.max_new_tokens)
            .ok_or_else(|| format!("request {:?} total token count overflows", request.id))?;
        if total_tokens == 0 {
            return Err(format!("request {:?} has no tokens to decode", request.id));
        }

        let q_sequence =
            synthetic_scheduler_decode_values(request_index, total_tokens, q_heads * head_dim, 1);
        let k_sequence =
            synthetic_scheduler_decode_values(request_index, total_tokens, kv_heads * head_dim, 2);
        let v_sequence =
            synthetic_scheduler_decode_values(request_index, total_tokens, kv_heads * value_dim, 3);
        let shape = PagedDecodeShape {
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
        };
        let PagedKvCacheReadback {
            k: expected_k_cache,
            v: expected_v_cache,
        } = pack_paged_kv_cache_for_block_table(
            &k_sequence,
            &v_sequence,
            &block_table,
            total_tokens,
            shape,
        )?;
        decode_runner.insert_request(
            &mut context,
            &mut stream,
            request.id,
            shape,
            block_table.clone(),
            softmax_scale,
        )?;
        let mut run = SyntheticSchedulerPagedDecodeRun {
            request_id: request.id,
            prompt_tokens: request.prompt_tokens,
            max_new_tokens: request.max_new_tokens,
            total_tokens,
            block_table,
            q_sequence,
            k_sequence,
            v_sequence,
            expected_k_cache,
            expected_v_cache,
            decode_steps: 0,
            attention_max_abs_diff: 0.0,
            k_cache_max_abs_diff: 0.0,
            v_cache_max_abs_diff: 0.0,
        };
        for timestep in 0..run.prompt_tokens {
            run_synthetic_scheduler_decode_step(
                &mut decode_runner,
                &mut stream,
                &mut run,
                timestep,
                block_size,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
                "runtime scheduler paged decode prefill",
            )?;
        }
        scheduler.complete_prefill(run.request_id).map_err(|err| {
            format!(
                "failed to complete synthetic prefill {:?}: {err}",
                run.request_id
            )
        })?;
        runs.push(run);
    }

    let first_batch_ready = run_synthetic_scheduler_ready_batch(
        &mut decode_runner,
        &mut scheduler,
        &mut runs,
        &mut stream,
        &[RequestId(201), RequestId(202)],
        8,
        block_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        "runtime scheduler paged decode first batch",
    )?;
    let second_batch_ready = run_synthetic_scheduler_ready_batch(
        &mut decode_runner,
        &mut scheduler,
        &mut runs,
        &mut stream,
        &[RequestId(201)],
        8,
        block_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        "runtime scheduler paged decode second batch",
    )?;
    let final_ready = scheduler
        .ready_decode_batch(8)
        .map_err(|err| format!("failed to query final ready decode batch: {err}"))?
        .len();
    if final_ready != 0 {
        return Err(format!(
            "runtime scheduler paged decode final ready count {final_ready}, expected 0"
        ));
    }

    for run in &mut runs {
        let active = scheduler.active_request(run.request_id).ok_or_else(|| {
            format!(
                "request {:?} missing from active scheduler state",
                run.request_id
            )
        })?;
        if active.cached_tokens != run.total_tokens {
            return Err(format!(
                "request {:?} cached_tokens {} did not match total_tokens {}",
                run.request_id, active.cached_tokens, run.total_tokens
            ));
        }
        if active.generated_tokens != run.max_new_tokens {
            return Err(format!(
                "request {:?} generated_tokens {} did not match max_new_tokens {}",
                run.request_id, active.generated_tokens, run.max_new_tokens
            ));
        }
        let cache = decode_runner
            .read_cache_to_host(run.request_id, &mut stream)
            .map_err(|err| {
                format!(
                    "failed to read synthetic cache for {:?}: {err}",
                    run.request_id
                )
            })?;
        run.k_cache_max_abs_diff = verify_f32_close(
            &format!(
                "runtime scheduler paged decode request {:?} k cache",
                run.request_id
            ),
            &cache.k,
            &run.expected_k_cache,
            1e-5,
            1e-5,
        )?;
        run.v_cache_max_abs_diff = verify_f32_close(
            &format!(
                "runtime scheduler paged decode request {:?} v cache",
                run.request_id
            ),
            &cache.v,
            &run.expected_v_cache,
            1e-5,
            1e-5,
        )?;
    }

    let stats = scheduler.allocator_stats();
    let request_ids = runs.iter().map(|run| run.request_id.0).collect::<Vec<_>>();
    let block_tables = runs
        .iter()
        .map(|run| run.block_table.clone())
        .collect::<Vec<_>>();
    let cached_tokens = runs
        .iter()
        .map(|run| {
            scheduler
                .active_request(run.request_id)
                .map(|active| active.cached_tokens)
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let generated_tokens = runs
        .iter()
        .map(|run| {
            scheduler
                .active_request(run.request_id)
                .map(|active| active.generated_tokens)
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let prompt_tokens = runs.iter().map(|run| run.prompt_tokens).collect::<Vec<_>>();
    let max_new_tokens = runs
        .iter()
        .map(|run| run.max_new_tokens)
        .collect::<Vec<_>>();
    let decode_steps = runs.iter().map(|run| run.decode_steps).collect::<Vec<_>>();
    let attention_max_abs_diff = runs
        .iter()
        .map(|run| run.attention_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let k_cache_max_abs_diff = runs
        .iter()
        .map(|run| run.k_cache_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let v_cache_max_abs_diff = runs
        .iter()
        .map(|run| run.v_cache_max_abs_diff)
        .fold(0.0_f32, f32::max);

    Ok(format!(
        "runtime-scheduler-paged-decode-smoke backend={} device_index={} name=\"{}\" request_count={} request_ids={:?} prompt_tokens={:?} max_new_tokens={:?} block_size={} cache_blocks={} block_tables={:?} first_batch_ready={} second_batch_ready={} final_ready={} decode_steps={:?} cached_tokens={:?} generated_tokens={:?} active_len={} free_blocks={} allocated_block_count={} free_runs={} largest_free_run={} q_heads={} kv_heads={} head_dim={} value_dim={} softmax_scale={softmax_scale:.9} attention_max_abs_diff={attention_max_abs_diff:.9} k_cache_max_abs_diff={k_cache_max_abs_diff:.9} v_cache_max_abs_diff={v_cache_max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        runs.len(),
        request_ids,
        prompt_tokens,
        max_new_tokens,
        block_size,
        cache_blocks,
        block_tables,
        first_batch_ready,
        second_batch_ready,
        final_ready,
        decode_steps,
        cached_tokens,
        generated_tokens,
        scheduler.active_len(),
        stats.free_blocks,
        stats.allocated_blocks,
        stats.free_runs,
        stats.largest_free_run,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
    ))
}

fn runtime_scheduler_layer_decode_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    match runtime_scheduler_layer_decode_smoke_impl(device_index) {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn runtime_scheduler_layer_decode_smoke_impl(device_index: u32) -> Result<String, String> {
    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let block_size = 2_usize;
    let cache_blocks = 8_usize;
    let shape = PagedDecodeShape {
        block_size,
        cache_blocks,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 2,
        value_dim: 2,
    };
    let hidden = 4_usize;
    let intermediate = 5_usize;
    let softmax_scale = 1.0_f32 / (shape.head_dim as f32).sqrt();
    let mlp_epsilon = 1e-5_f32;
    let q_token_elements = shape.q_elements()?;
    let k_token_elements = shape.k_token_elements()?;
    let v_token_elements = shape.v_token_elements()?;
    let attention_elements = shape.output_elements()?;
    let weights = synthetic_scheduler_decoder_layer_weights(
        &mut context,
        &mut stream,
        shape,
        hidden,
        intermediate,
    )?;
    let requests = vec![
        Request::new(201, 2, 2),
        Request::new(202, 3, 1),
        Request::new(203, 1, 0),
    ];

    let mut scheduler = SchedulerState::with_block_size(cache_blocks as u32, block_size as u32);
    for request in &requests {
        scheduler.enqueue(request.clone());
    }
    let mut runner = Qwen3DecoderLayerRequestDecodeRunner::new();
    let mut allocated = scheduler
        .pop_prefill_batch_with_allocation(usize::MAX)
        .map_err(|err| format!("failed to allocate synthetic scheduler layer batch: {err}"))?;
    if allocated.len() != requests.len() {
        return Err(format!(
            "synthetic scheduler layer selected {} requests, expected {}",
            allocated.len(),
            requests.len()
        ));
    }

    let mut runs = Vec::with_capacity(allocated.len());
    for (request_index, scheduled) in allocated.drain(..).enumerate() {
        let request = scheduled.request;
        let block_table = scheduled.allocation.blocks;
        let total_tokens = request
            .prompt_tokens
            .checked_add(request.max_new_tokens)
            .ok_or_else(|| format!("request {:?} total token count overflows", request.id))?;
        if total_tokens == 0 {
            return Err(format!("request {:?} has no tokens to decode", request.id));
        }

        let q_sequence =
            synthetic_scheduler_decode_values(request_index, total_tokens, q_token_elements, 21);
        let k_sequence =
            synthetic_scheduler_decode_values(request_index, total_tokens, k_token_elements, 22);
        let v_sequence =
            synthetic_scheduler_decode_values(request_index, total_tokens, v_token_elements, 23);
        let gate_sequence =
            synthetic_scheduler_decode_values(request_index, total_tokens, attention_elements, 24);
        let residual_sequence =
            synthetic_scheduler_decode_values(request_index, total_tokens, hidden, 25);
        let expected = qwen3_decoder_layer_sequence_to_host_f32(
            &weights,
            &mut context,
            &mut stream,
            shape,
            &block_table,
            softmax_scale,
            mlp_epsilon,
            &q_sequence,
            &k_sequence,
            &v_sequence,
            Some(&gate_sequence),
            &residual_sequence,
            total_tokens,
        )?;

        runner.insert_request(
            &mut context,
            &mut stream,
            request.id,
            &weights,
            shape,
            block_table.clone(),
            softmax_scale,
            mlp_epsilon,
        )?;
        let mut run = SchedulerLayerDecodeRun {
            state: SchedulerLayerDecodeState {
                request_id: request.id,
                prompt_tokens: request.prompt_tokens,
                max_new_tokens: request.max_new_tokens,
                total_tokens,
                block_table,
                q_sequence,
                k_sequence,
                v_sequence,
                output_gate_sequence: Some(gate_sequence),
                residual_sequence,
                decode_steps: 0,
            },
            checks: SchedulerLayerDecodeSmokeChecks::new(expected),
        };
        for timestep in 0..run.prompt_tokens {
            run_scheduler_layer_prefill_step(
                &mut runner,
                &mut stream,
                &mut run,
                timestep,
                q_token_elements,
                k_token_elements,
                v_token_elements,
                attention_elements,
                hidden,
                "runtime scheduler layer decode prefill",
            )?;
        }
        scheduler.complete_prefill(run.request_id).map_err(|err| {
            format!(
                "failed to complete synthetic layer prefill {:?}: {err}",
                run.request_id
            )
        })?;
        runs.push(run);
    }

    let first_batch_ready = run_scheduler_layer_ready_batch(
        &mut runner,
        &mut scheduler,
        &mut runs,
        &mut stream,
        &[RequestId(201), RequestId(202)],
        8,
        q_token_elements,
        k_token_elements,
        v_token_elements,
        attention_elements,
        hidden,
        true,
        "runtime scheduler layer decode first batch",
    )?;
    let second_batch_ready = run_scheduler_layer_ready_batch(
        &mut runner,
        &mut scheduler,
        &mut runs,
        &mut stream,
        &[RequestId(201)],
        8,
        q_token_elements,
        k_token_elements,
        v_token_elements,
        attention_elements,
        hidden,
        true,
        "runtime scheduler layer decode second batch",
    )?;
    let final_ready = scheduler
        .ready_decode_batch(8)
        .map_err(|err| format!("failed to query final layer ready decode batch: {err}"))?
        .len();
    if final_ready != 0 {
        return Err(format!(
            "runtime scheduler layer decode final ready count {final_ready}, expected 0"
        ));
    }

    for run in &mut runs {
        let active = scheduler.active_request(run.request_id).ok_or_else(|| {
            format!(
                "layer request {:?} missing from active scheduler state",
                run.request_id
            )
        })?;
        if active.cached_tokens != run.total_tokens {
            return Err(format!(
                "layer request {:?} cached_tokens {} did not match total_tokens {}",
                run.request_id, active.cached_tokens, run.total_tokens
            ));
        }
        if active.generated_tokens != run.max_new_tokens {
            return Err(format!(
                "layer request {:?} generated_tokens {} did not match max_new_tokens {}",
                run.request_id, active.generated_tokens, run.max_new_tokens
            ));
        }
        let cache = runner
            .read_cache_to_host(run.request_id, &mut stream)
            .map_err(|err| {
                format!(
                    "failed to read synthetic layer cache for {:?}: {err}",
                    run.request_id
                )
            })?;
        run.checks.k_cache_max_abs_diff = verify_f32_close(
            &format!(
                "runtime scheduler layer decode request {:?} k cache",
                run.request_id
            ),
            &cache.k,
            &run.checks.expected.paged_cache.k,
            1e-5,
            1e-5,
        )?;
        run.checks.v_cache_max_abs_diff = verify_f32_close(
            &format!(
                "runtime scheduler layer decode request {:?} v cache",
                run.request_id
            ),
            &cache.v,
            &run.checks.expected.paged_cache.v,
            1e-5,
            1e-5,
        )?;
    }

    let stats = scheduler.allocator_stats();
    let request_ids = runs.iter().map(|run| run.request_id.0).collect::<Vec<_>>();
    let block_tables = runs
        .iter()
        .map(|run| run.block_table.clone())
        .collect::<Vec<_>>();
    let cached_tokens = runs
        .iter()
        .map(|run| {
            scheduler
                .active_request(run.request_id)
                .map(|active| active.cached_tokens)
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let generated_tokens = runs
        .iter()
        .map(|run| {
            scheduler
                .active_request(run.request_id)
                .map(|active| active.generated_tokens)
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let prompt_tokens = runs.iter().map(|run| run.prompt_tokens).collect::<Vec<_>>();
    let max_new_tokens = runs
        .iter()
        .map(|run| run.max_new_tokens)
        .collect::<Vec<_>>();
    let decode_steps = runs.iter().map(|run| run.decode_steps).collect::<Vec<_>>();
    let attention_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.attention_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let projection_input_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.projection_input_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let projected_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.projected_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let block_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.block_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let post_norm_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.post_norm_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let mlp_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.mlp_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let layer_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.layer_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let k_cache_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.k_cache_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let v_cache_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.v_cache_max_abs_diff)
        .fold(0.0_f32, f32::max);

    Ok(format!(
        "runtime-scheduler-layer-decode-smoke backend={} device_index={} name=\"{}\" request_count={} request_ids={:?} prompt_tokens={:?} max_new_tokens={:?} block_size={} cache_blocks={} block_tables={:?} first_batch_ready={} second_batch_ready={} final_ready={} decode_steps={:?} cached_tokens={:?} generated_tokens={:?} active_len={} free_blocks={} allocated_block_count={} free_runs={} largest_free_run={} q_heads={} kv_heads={} head_dim={} value_dim={} hidden={} intermediate={} softmax_scale={softmax_scale:.9} mlp_epsilon={mlp_epsilon:.9} attention_max_abs_diff={attention_max_abs_diff:.9} projection_input_max_abs_diff={projection_input_max_abs_diff:.9} projected_max_abs_diff={projected_max_abs_diff:.9} block_max_abs_diff={block_max_abs_diff:.9} post_norm_max_abs_diff={post_norm_max_abs_diff:.9} mlp_max_abs_diff={mlp_max_abs_diff:.9} layer_max_abs_diff={layer_max_abs_diff:.9} k_cache_max_abs_diff={k_cache_max_abs_diff:.9} v_cache_max_abs_diff={v_cache_max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        runs.len(),
        request_ids,
        prompt_tokens,
        max_new_tokens,
        block_size,
        cache_blocks,
        block_tables,
        first_batch_ready,
        second_batch_ready,
        final_ready,
        decode_steps,
        cached_tokens,
        generated_tokens,
        scheduler.active_len(),
        stats.free_blocks,
        stats.allocated_blocks,
        stats.free_runs,
        stats.largest_free_run,
        shape.q_heads,
        shape.kv_heads,
        shape.head_dim,
        shape.value_dim,
        hidden,
        intermediate,
    ))
}

fn runtime_kv_paged_decode_attn_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let mut allocator = KvBlockAllocator::with_block_size(4, 2);
    let fragment = match allocator.allocate(RequestId(10), 3) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to allocate fragmenting KV blocks: {err}");
            return ExitCode::from(1);
        }
    };
    let freed = allocator.free_request(fragment.request_id);
    if freed != fragment.blocks.len() {
        eprintln!(
            "freed KV block count {freed} does not match allocated fragment blocks {}",
            fragment.blocks.len()
        );
        return ExitCode::from(1);
    }
    let cache_len = 3_usize;
    let block_size = allocator.block_size_tokens() as usize;
    let block_count = (cache_len - 1) / block_size + 1;
    let allocation = match allocator.allocate(RequestId(11), block_count) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to allocate decode KV blocks: {err}");
            return ExitCode::from(1);
        }
    };
    let block_table = allocation.blocks;
    let cache_blocks = allocator.total_blocks() as usize;
    let stats = allocator.stats();

    let q_heads = 4_usize;
    let kv_heads = 2_usize;
    let head_dim = 3_usize;
    let value_dim = 2_usize;
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q = (0..q_heads * head_dim)
        .map(|index| (index as f32 - 8.0) / 11.0)
        .collect::<Vec<_>>();
    let logical_k = (0..cache_len * kv_heads * head_dim)
        .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
        .collect::<Vec<_>>();
    let logical_v = (0..cache_len * kv_heads * value_dim)
        .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
        .collect::<Vec<_>>();
    let decode_shape = PagedDecodeShape {
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
    };
    let PagedKvCacheReadback {
        k: paged_k_cache,
        v: paged_v_cache,
    } = match pack_paged_kv_cache_for_block_table(
        &logical_k,
        &logical_v,
        &block_table,
        cache_len,
        decode_shape,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (output, max_abs_diff) = match runtime_paged_decode_attn_verify(
        &mut context,
        &mut stream,
        &q,
        &paged_k_cache,
        &paged_v_cache,
        &block_table,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        "runtime kv paged decode attention smoke",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let contiguous_expected = runtime_host_decode_attn_f32(
        &q,
        &logical_k,
        &logical_v,
        cache_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let logical_paged_max_abs_diff = match verify_f32_close(
        "runtime kv paged decode attention logical-vs-paged",
        &output,
        &contiguous_expected,
        1e-4,
        1e-4,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    println!(
        "runtime-kv-paged-decode-smoke backend={} device_index={} name=\"{}\" cache_len={} block_size={} cache_blocks={} allocated_blocks={:?} free_blocks={} allocated_block_count={} free_runs={} largest_free_run={} q_heads={} kv_heads={} head_dim={} value_dim={} softmax_scale={softmax_scale:.9} paged_k_cache_preview={} paged_v_cache_preview={} output={} max_abs_diff={max_abs_diff:.9} logical_paged_max_abs_diff={logical_paged_max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        cache_len,
        block_size,
        cache_blocks,
        block_table,
        stats.free_blocks,
        stats.allocated_blocks,
        stats.free_runs,
        stats.largest_free_run,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        format_f32_preview(&paged_k_cache[..8.min(paged_k_cache.len())]),
        format_f32_preview(&paged_v_cache[..8.min(paged_v_cache.len())]),
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_depthwise_conv1d_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let channels = 3_usize;
    let sequence_len = 5_usize;
    let kernel_size = 3_usize;
    let input = [
        1.0_f32, 0.5, -1.0, 2.0, 1.0, 0.5, 3.0, -0.5, 0.5, 4.0, -1.0, 1.5, 5.0, 0.0, -2.0,
    ];
    let weight = [1.0_f32, -1.0, 2.0, 0.5, 1.0, -0.5, -1.0, 1.0, 1.5];
    let expected =
        runtime_host_depthwise_conv1d_f32(&input, &weight, channels, sequence_len, kernel_size);
    if expected.is_empty() {
        eprintln!("failed to build deterministic depthwise conv1d reference");
        return ExitCode::from(1);
    }

    let input_bytes = encode_f32_to_bytes(&input);
    let weight_bytes = encode_f32_to_bytes(&weight);
    let output_bytes = input_bytes.len();

    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate depthwise conv1d input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut weight_buffer = match context.alloc_buffer(weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate depthwise conv1d weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate depthwise conv1d output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy depthwise conv1d input data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = weight_buffer.copy_from_host(0, &weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy depthwise conv1d weight data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after depthwise conv1d input copy: {err}");
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::depthwise_conv1d_f32(
        &input_buffer,
        &weight_buffer,
        channels,
        sequence_len,
        kernel_size,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime depthwise_conv1d_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after depthwise conv1d: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy depthwise conv1d result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after depthwise conv1d output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);

    let mut max_abs_diff = 0.0_f32;
    for (lhs, rhs) in output.iter().zip(expected.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-5_f32 {
            eprintln!(
                "runtime depthwise conv1d smoke produced unexpected output: max_abs_diff={diff} output={:?} expected={:?}",
                output, expected
            );
            return ExitCode::from(1);
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }
    println!(
        "runtime-depthwise-conv1d-smoke backend={} device_index={} name=\"{}\" channels={} sequence_len={} kernel_size={} output={} max_abs_diff={max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        channels,
        sequence_len,
        kernel_size,
        format_f32_preview(&output[..8.min(output.len())])
    );
    ExitCode::SUCCESS
}

fn runtime_wmma_fp8_probe_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    if info.backend != "hip" {
        eprintln!(
            "runtime wmma fp8 probe smoke requires a HIP device: backend={} device={}",
            info.backend, info.name
        );
        return ExitCode::from(1);
    }
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let mut output = match context.alloc_buffer(4) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate runtime wmma probe output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = ullm_runtime_sys::wmma_fp8_probe(&mut output, Some(&mut stream)) {
        eprintln!("failed to run runtime wmma fp8 probe: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after wmma fp8 probe: {err}");
        return ExitCode::from(1);
    }

    let mut raw = [0_u8; 4];
    if let Err(err) = output.copy_to_host(0, &mut raw, Some(&mut stream)) {
        eprintln!("failed to copy runtime wmma fp8 probe marker back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after wmma fp8 probe copyback: {err}");
        return ExitCode::from(1);
    }

    let marker = u32::from_le_bytes(raw);
    if marker == 0 {
        eprintln!(
            "runtime wmma fp8 probe marker is zero: backend={} device={} compute={}.{} arch={}",
            info.backend, info.name, info.compute_major, info.compute_minor, info.gcn_arch_name
        );
        return ExitCode::from(1);
    }

    println!(
        "runtime-wmma-fp8-probe-smoke backend={} device={} compute={}.{} arch={} marker=0x{marker:08x} verified=true",
        info.backend, info.name, info.compute_major, info.compute_minor, info.gcn_arch_name
    );
    ExitCode::SUCCESS
}

fn runtime_fp8_qk_probe_smoke(
    command_name: &'static str,
    command_label: &'static str,
    call_label: &'static str,
    device_index: Option<String>,
    pattern: Option<String>,
    preview_count: Option<String>,
    probe: Fp8QkProbeKernel,
) -> ExitCode {
    let selected_device_index = match select_rdna4_device(device_index, command_name) {
        Ok(Some(index)) => index,
        Ok(None) => return ExitCode::SUCCESS,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(selected_device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    if info.backend != "hip" {
        eprintln!(
            "{call_label} requires a HIP device: backend={} device={}",
            info.backend, info.name
        );
        return ExitCode::from(1);
    }
    if !is_rdna4_device(&info) {
        eprintln!(
            "{call_label} requires RDNA4: backend={} device={} compute={}.{} arch={}",
            info.backend, info.name, info.compute_major, info.compute_minor, info.gcn_arch_name
        );
        return ExitCode::from(1);
    }
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let pattern = match parse_fp8_qk_pattern(pattern, command_name) {
        Ok(pattern) => pattern,
        Err(code) => return code,
    };
    let preview_count = match parse_optional_usize(preview_count, 16, "preview count") {
        Ok(value) => value.min(16 * 16),
        Err(code) => return code,
    };

    let tile_bytes = 16_usize * 16_usize;
    let output_bytes = tile_bytes * std::mem::size_of::<f32>();
    let mut q_buffer = match context.alloc_buffer(tile_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!(
                "failed to allocate runtime {} q buffer: {err}",
                command_label
            );
            return ExitCode::from(1);
        }
    };
    let mut k_buffer = match context.alloc_buffer(tile_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!(
                "failed to allocate runtime {} k buffer: {err}",
                command_label
            );
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!(
                "failed to allocate runtime {} qk output buffer: {err}",
                command_label
            );
            return ExitCode::from(1);
        }
    };

    let (q_bytes, k_bytes) = fp8_qk_probe_pattern_bytes(&pattern);
    if let Err(err) = q_buffer.copy_from_host(0, &q_bytes, Some(&mut stream)) {
        eprintln!("failed to copy runtime {} q input: {err}", command_label);
        return ExitCode::from(1);
    }
    if let Err(err) = k_buffer.copy_from_host(0, &k_bytes, Some(&mut stream)) {
        eprintln!("failed to copy runtime {} k input: {err}", command_label);
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!(
            "failed to synchronize runtime {} q/k copy: {err}",
            command_label
        );
        return ExitCode::from(1);
    }

    if let Err(err) = probe(&q_buffer, &k_buffer, &mut output_buffer, Some(&mut stream)) {
        eprintln!("failed to run {command_name}: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after {command_name}: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy {command_name} output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after {command_name} output copy: {err}");
        return ExitCode::from(1);
    }

    let output = decode_f32_le_values(&output_raw);
    let mut max_abs = 0.0_f32;
    for value in &output {
        if !value.is_finite() {
            eprintln!("{command_name} output is not finite");
            return ExitCode::from(1);
        }
        let abs = value.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    if !output.iter().any(|value| *value != 0.0) {
        eprintln!("{command_name} output is all zero");
        return ExitCode::from(1);
    }
    let preview = format_f32_preview(&output[..preview_count.min(output.len())]);

    println!(
        "{command_name} backend={} device={} compute={}.{} arch={} pattern={} max_abs={max_abs:.9} preview_count={} preview={} finite=true nonzero=true verified=true",
        info.backend,
        info.name,
        info.compute_major,
        info.compute_minor,
        info.gcn_arch_name,
        pattern.as_str(),
        preview_count,
        preview
    );
    ExitCode::SUCCESS
}

fn runtime_wmma_fp8_qk_probe_smoke(
    device_index: Option<String>,
    pattern: Option<String>,
    preview_count: Option<String>,
) -> ExitCode {
    runtime_fp8_qk_probe_smoke(
        "runtime-wmma-fp8-qk-probe-smoke",
        "wmma",
        "runtime wmma fp8 qk probe smoke",
        device_index,
        pattern,
        preview_count,
        ullm_runtime_sys::wmma_fp8_qk_probe,
    )
}

fn runtime_rocwmma_fp8_qk_probe_smoke(
    device_index: Option<String>,
    pattern: Option<String>,
    preview_count: Option<String>,
) -> ExitCode {
    runtime_fp8_qk_probe_smoke(
        "runtime-rocwmma-fp8-qk-probe-smoke",
        "rocwmma",
        "runtime rocwmma fp8 qk probe smoke",
        device_index,
        pattern,
        preview_count,
        ullm_runtime_sys::rocwmma_fp8_qk_probe,
    )
}

fn runtime_rocwmma_fp8_attn_probe_smoke(
    device_index: Option<String>,
    pattern: Option<String>,
) -> ExitCode {
    let command_name = "runtime-rocwmma-fp8-attn-probe-smoke";
    let selected_device_index = match select_rdna4_device(device_index, command_name) {
        Ok(Some(index)) => index,
        Ok(None) => return ExitCode::SUCCESS,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(selected_device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    if info.backend != "hip" {
        eprintln!(
            "runtime rocwmma fp8 attention probe smoke requires a HIP device: backend={} device={}",
            info.backend, info.name
        );
        return ExitCode::from(1);
    }
    if !is_rdna4_device(&info) {
        eprintln!(
            "runtime rocwmma fp8 attention probe smoke requires RDNA4: backend={} device={} compute={}.{} arch={}",
            info.backend, info.name, info.compute_major, info.compute_minor, info.gcn_arch_name
        );
        return ExitCode::from(1);
    }
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let pattern = match parse_fp8_qk_pattern(pattern, command_name) {
        Ok(pattern) => pattern,
        Err(code) => return code,
    };
    let (q_bytes, k_bytes, v_values, expected) = fp8_attn_probe_inputs(&pattern);
    let v_bytes = encode_f32_to_bytes(&v_values);
    let output_bytes = 16_usize * 16_usize * std::mem::size_of::<f32>();

    let mut q_buffer = match context.alloc_buffer(q_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate runtime rocwmma attention q buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut k_buffer = match context.alloc_buffer(k_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate runtime rocwmma attention k buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_buffer = match context.alloc_buffer(v_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate runtime rocwmma attention v buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate runtime rocwmma attention output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = q_buffer.copy_from_host(0, &q_bytes, Some(&mut stream)) {
        eprintln!("failed to copy runtime rocwmma attention q input: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = k_buffer.copy_from_host(0, &k_bytes, Some(&mut stream)) {
        eprintln!("failed to copy runtime rocwmma attention k input: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_buffer.copy_from_host(0, &v_bytes, Some(&mut stream)) {
        eprintln!("failed to copy runtime rocwmma attention v input: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime rocwmma attention input copy: {err}");
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::rocwmma_fp8_attn_probe(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run {command_name}: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after {command_name}: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy {command_name} output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after {command_name} output copy: {err}");
        return ExitCode::from(1);
    }

    let output = decode_f32_le_values(&output_raw);
    if output.iter().any(|value| !value.is_finite()) {
        eprintln!("{command_name} output is not finite");
        return ExitCode::from(1);
    }
    let max_abs_diff = match verify_f32_close(command_name, &output, &expected, 2.0e-3, 2.0e-3) {
        Ok(diff) => diff,
        Err(err) => {
            eprintln!("{err}");
            eprintln!(
                "actual={}",
                format_f32_preview(&output[..16.min(output.len())])
            );
            eprintln!(
                "expected={}",
                format_f32_preview(&expected[..16.min(expected.len())])
            );
            return ExitCode::from(1);
        }
    };

    println!(
        "{command_name} backend={} device={} compute={}.{} arch={} pattern={} q_shape=16x16_fp8 k_shape=32x16_fp8 v_shape=32x16_f32 output_shape=16x16_f32 max_abs_diff={max_abs_diff:.9} output_preview={} expected_preview={} verified=true",
        info.backend,
        info.name,
        info.compute_major,
        info.compute_minor,
        info.gcn_arch_name,
        pattern.as_str(),
        format_f32_preview(&output[..16.min(output.len())]),
        format_f32_preview(&expected[..16.min(expected.len())])
    );
    ExitCode::SUCCESS
}

type Fp8QkProbeKernel = fn(
    &ullm_runtime_sys::RuntimeBuffer,
    &ullm_runtime_sys::RuntimeBuffer,
    &mut ullm_runtime_sys::RuntimeBuffer,
    Option<&mut ullm_runtime_sys::RuntimeStream>,
) -> Result<(), String>;

enum Fp8QkPattern {
    Ones,
    Layout,
}

impl Fp8QkPattern {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Ones => "ones",
            Self::Layout => "layout",
        }
    }
}

fn parse_fp8_qk_pattern(raw: Option<String>, command_name: &str) -> Result<Fp8QkPattern, ExitCode> {
    match raw {
        Some(raw) => {
            let value = raw.split_once('=').map_or(raw.as_str(), |(_, value)| value);
            match value {
                "ones" => Ok(Fp8QkPattern::Ones),
                "layout" => Ok(Fp8QkPattern::Layout),
                _ => {
                    eprintln!("{command_name}: invalid pattern {raw}; expected ones|layout");
                    Err(ExitCode::from(2))
                }
            }
        }
        None => Ok(Fp8QkPattern::Ones),
    }
}

fn select_rdna4_device(
    device_index: Option<String>,
    command_name: &str,
) -> Result<Option<u32>, ExitCode> {
    let device_count = match ullm_runtime_sys::device_count() {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to query device count: {err}");
            return Err(ExitCode::from(1));
        }
    };
    let selected = match device_index {
        Some(raw) => match parse_optional_device_index(Some(raw)) {
            Ok(value) => value,
            Err(code) => return Err(code),
        },
        None => match (0..device_count).find(|&index| {
            ullm_runtime_sys::device_info(index)
                .map(|info| is_rdna4_device(&info))
                .unwrap_or(false)
        }) {
            Some(index) => index,
            None => {
                println!("{command_name}: no rdna4 device found; skipped");
                return Ok(None);
            }
        },
    };
    Ok(Some(selected))
}

fn fp8_qk_probe_pattern_bytes(pattern: &Fp8QkPattern) -> (Vec<u8>, Vec<u8>) {
    let tile_bytes = 16_usize * 16_usize;
    match pattern {
        Fp8QkPattern::Ones => (vec![0x38_u8; tile_bytes], vec![0x38_u8; tile_bytes]),
        Fp8QkPattern::Layout => {
            let mut q_values = vec![0.0_f32; tile_bytes];
            let mut k_values = vec![0.0_f32; tile_bytes];
            for row in 0..16 {
                let offset = row * 16;
                q_values[offset] = row as f32;
                q_values[offset + 1] = 1.0;
            }
            for col in 0..16 {
                let offset = col * 16;
                k_values[offset] = 16.0;
                k_values[offset + 1] = col as f32;
            }
            let q_bytes = q_values
                .into_iter()
                .map(|value| fp8_e4m3_encode_scaled(value, 1.0))
                .collect();
            let k_bytes = k_values
                .into_iter()
                .map(|value| fp8_e4m3_encode_scaled(value, 1.0))
                .collect();
            (q_bytes, k_bytes)
        }
    }
}

fn fp8_attn_probe_inputs(pattern: &Fp8QkPattern) -> (Vec<u8>, Vec<u8>, Vec<f32>, Vec<f32>) {
    let q_tokens = 16_usize;
    let kv_tokens = 32_usize;
    let head_dim = 16_usize;
    let q_elements = q_tokens * head_dim;
    let k_elements = kv_tokens * head_dim;
    let (q_bytes, k_bytes) = match pattern {
        Fp8QkPattern::Ones => (vec![0x38_u8; q_elements], vec![0x38_u8; k_elements]),
        Fp8QkPattern::Layout => {
            let mut q_values = vec![0.0_f32; q_elements];
            let mut k_values = vec![0.0_f32; k_elements];
            for row in 0..q_tokens {
                let offset = row * head_dim;
                q_values[offset] = row as f32;
                q_values[offset + 1] = 1.0;
            }
            for token in 0..kv_tokens {
                let offset = token * head_dim;
                k_values[offset] = 16.0;
                k_values[offset + 1] = token as f32;
            }
            let q_bytes = q_values
                .into_iter()
                .map(|value| fp8_e4m3_encode_scaled(value, 1.0))
                .collect();
            let k_bytes = k_values
                .into_iter()
                .map(|value| fp8_e4m3_encode_scaled(value, 1.0))
                .collect();
            (q_bytes, k_bytes)
        }
    };
    let v_values = (0..kv_tokens * head_dim)
        .map(|index| {
            let token = index / head_dim;
            let value = index % head_dim;
            (((token * 17 + value * 5) % 97) as f32 - 48.0) * 0.03125
        })
        .collect::<Vec<_>>();
    let expected = fp8_attn_probe_reference(&q_bytes, &k_bytes, &v_values);
    (q_bytes, k_bytes, v_values, expected)
}

fn fp8_attn_probe_reference(q: &[u8], k: &[u8], v: &[f32]) -> Vec<f32> {
    let q_tokens = 16_usize;
    let kv_tokens = 32_usize;
    let head_dim = 16_usize;
    let mut output = vec![0.0_f32; q_tokens * head_dim];
    for row in 0..q_tokens {
        let mut scores = vec![0.0_f32; kv_tokens];
        let mut max_score = f32::NEG_INFINITY;
        for token in 0..kv_tokens {
            let mut score = 0.0_f32;
            for dim in 0..head_dim {
                let q_value = fp8_e4m3_to_f32_unscaled(q[row * head_dim + dim]);
                let k_value = fp8_e4m3_to_f32_unscaled(k[token * head_dim + dim]);
                score += q_value * k_value;
            }
            scores[token] = score;
            max_score = max_score.max(score);
        }
        let mut denominator = 0.0_f32;
        for token in 0..kv_tokens {
            let weight = (scores[token] - max_score).exp();
            denominator += weight;
            for value in 0..head_dim {
                output[row * head_dim + value] += weight * v[token * head_dim + value];
            }
        }
        for value in 0..head_dim {
            output[row * head_dim + value] /= denominator;
        }
    }
    output
}

fn runtime_linear_attn_gate_beta_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let heads = 3_usize;
    let sequence_len = 4_usize;
    let a = [
        -2.0_f32, -0.5, 0.25, 0.75, 1.0, -1.25, 2.0, -3.0, 0.5, 21.0, -20.0, 4.0,
    ];
    let b = [
        -4.0_f32, -1.0, 0.0, 0.5, 1.0, 3.0, -2.0, 2.5, -0.25, 4.0, -3.5, 1.5,
    ];
    let a_log = [-0.75_f32, 0.0, 0.5];
    let dt_bias = [0.25_f32, -0.5, 1.25];
    let (expected_gate, expected_beta) =
        runtime_host_linear_attn_gate_beta_f32(&a, &b, &a_log, &dt_bias, heads, sequence_len);
    if expected_gate.is_empty() || expected_beta.is_empty() {
        eprintln!("failed to build deterministic linear attention gate beta reference");
        return ExitCode::from(1);
    }

    let a_bytes = encode_f32_to_bytes(&a);
    let b_bytes = encode_f32_to_bytes(&b);
    let a_log_bytes = encode_f32_to_bytes(&a_log);
    let dt_bias_bytes = encode_f32_to_bytes(&dt_bias);
    let output_bytes = a_bytes.len();

    let mut a_buffer = match context.alloc_buffer(a_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate a runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut b_buffer = match context.alloc_buffer(b_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate b runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut a_log_buffer = match context.alloc_buffer(a_log_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate A_log runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut dt_bias_buffer = match context.alloc_buffer(dt_bias_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate dt_bias runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut gate_output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut beta_output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate beta output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = a_buffer.copy_from_host(0, &a_bytes, Some(&mut stream)) {
        eprintln!("failed to copy a data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = b_buffer.copy_from_host(0, &b_bytes, Some(&mut stream)) {
        eprintln!("failed to copy b data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = a_log_buffer.copy_from_host(0, &a_log_bytes, Some(&mut stream)) {
        eprintln!("failed to copy A_log data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = dt_bias_buffer.copy_from_host(0, &dt_bias_bytes, Some(&mut stream)) {
        eprintln!("failed to copy dt_bias data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate beta input copy: {err}");
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::linear_attn_gate_beta_f32(
        &a_buffer,
        &b_buffer,
        &a_log_buffer,
        &dt_bias_buffer,
        heads,
        sequence_len,
        &mut gate_output_buffer,
        &mut beta_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime linear_attn_gate_beta_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after linear attention gate beta: {err}");
        return ExitCode::from(1);
    }

    let mut gate_output_raw = vec![0_u8; output_bytes];
    let mut beta_output_raw = vec![0_u8; output_bytes];
    if let Err(err) = gate_output_buffer.copy_to_host(0, &mut gate_output_raw, Some(&mut stream)) {
        eprintln!("failed to copy gate output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = beta_output_buffer.copy_to_host(0, &mut beta_output_raw, Some(&mut stream)) {
        eprintln!("failed to copy beta output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate beta output copy: {err}");
        return ExitCode::from(1);
    }
    let gate_output = decode_f32_le_values(&gate_output_raw);
    let beta_output = decode_f32_le_values(&beta_output_raw);

    let mut max_abs_diff = 0.0_f32;
    for ((gate, expected_gate), (beta, expected_beta)) in gate_output
        .iter()
        .zip(expected_gate.iter())
        .zip(beta_output.iter().zip(expected_beta.iter()))
    {
        let gate_diff = (gate - expected_gate).abs();
        let beta_diff = (beta - expected_beta).abs();
        let diff = gate_diff.max(beta_diff);
        if diff > 1e-5_f32 {
            eprintln!(
                "runtime linear attention gate beta smoke produced unexpected output: max_abs_diff={diff} gate={:?} expected_gate={:?} beta={:?} expected_beta={:?}",
                gate_output, expected_gate, beta_output, expected_beta
            );
            return ExitCode::from(1);
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }
    println!(
        "runtime-linear-attn-gate-beta-smoke backend={} device_index={} name=\"{}\" heads={} sequence_len={} gate={} beta={} max_abs_diff={max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        heads,
        sequence_len,
        format_f32_preview(&gate_output[..8.min(gate_output.len())]),
        format_f32_preview(&beta_output[..8.min(beta_output.len())]),
    );
    ExitCode::SUCCESS
}

fn runtime_linear_attn_recurrent_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let key_heads = 1_usize;
    let value_heads = 2_usize;
    let sequence_len = 3_usize;
    let key_dim = 3_usize;
    let value_dim = 2_usize;
    let q = [0.25_f32, -0.5, 0.75, 0.6, -0.2, 0.3, -0.1, 0.8, -0.35];
    let k = [-0.3_f32, 0.4, 0.2, 0.1, 0.2, -0.6, 0.55, -0.1, 0.25];
    let v = [
        0.7_f32, -0.2, -0.5, 0.4, 0.25, 0.3, -0.1, -0.6, 0.9, 0.05, -0.35, 0.8,
    ];
    let gate = [-0.2_f32, -0.5, -0.1, -0.3, -0.7, -0.05];
    let beta = [0.8_f32, 0.6, 0.5, 0.9, 0.7, 0.4];
    let initial_state = [
        0.01_f32, -0.02, 0.03, 0.04, -0.01, 0.02, -0.03, 0.05, 0.02, -0.04, 0.01, 0.03,
    ];
    let mut expected_state = initial_state.to_vec();
    let expected_output = runtime_host_linear_attn_recurrent_f32(
        &q,
        &k,
        &v,
        &gate,
        &beta,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        &mut expected_state,
    );
    if expected_output.is_empty() {
        eprintln!("failed to build deterministic linear attention recurrent reference");
        return ExitCode::from(1);
    }

    let q_bytes = encode_f32_to_bytes(&q);
    let k_bytes = encode_f32_to_bytes(&k);
    let v_bytes = encode_f32_to_bytes(&v);
    let gate_bytes = encode_f32_to_bytes(&gate);
    let beta_bytes = encode_f32_to_bytes(&beta);
    let state_bytes = encode_f32_to_bytes(&initial_state);
    let output_bytes = v_bytes.len();

    let mut q_buffer = match context.alloc_buffer(q_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate q runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut k_buffer = match context.alloc_buffer(k_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate k runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_buffer = match context.alloc_buffer(v_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate v runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut gate_buffer = match context.alloc_buffer(gate_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut beta_buffer = match context.alloc_buffer(beta_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate beta runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut state_buffer = match context.alloc_buffer(state_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate state runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = q_buffer.copy_from_host(0, &q_bytes, Some(&mut stream)) {
        eprintln!("failed to copy q data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = k_buffer.copy_from_host(0, &k_bytes, Some(&mut stream)) {
        eprintln!("failed to copy k data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_buffer.copy_from_host(0, &v_bytes, Some(&mut stream)) {
        eprintln!("failed to copy v data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = gate_buffer.copy_from_host(0, &gate_bytes, Some(&mut stream)) {
        eprintln!("failed to copy gate data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = beta_buffer.copy_from_host(0, &beta_bytes, Some(&mut stream)) {
        eprintln!("failed to copy beta data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = state_buffer.copy_from_host(0, &state_bytes, Some(&mut stream)) {
        eprintln!("failed to copy state data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after recurrent input copy: {err}");
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::linear_attn_recurrent_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        &gate_buffer,
        &beta_buffer,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        &mut state_buffer,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime linear_attn_recurrent_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after linear attention recurrent: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    let mut final_state_raw = vec![0_u8; state_bytes.len()];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy recurrent output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = state_buffer.copy_to_host(0, &mut final_state_raw, Some(&mut stream)) {
        eprintln!("failed to copy recurrent state back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after recurrent output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);
    let final_state = decode_f32_le_values(&final_state_raw);

    let mut max_abs_diff = 0.0_f32;
    for (lhs, rhs) in output.iter().zip(expected_output.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-5_f32 {
            eprintln!(
                "runtime linear attention recurrent smoke produced unexpected output: max_abs_diff={diff} output={:?} expected={:?}",
                output, expected_output
            );
            return ExitCode::from(1);
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }
    for (lhs, rhs) in final_state.iter().zip(expected_state.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-5_f32 {
            eprintln!(
                "runtime linear attention recurrent smoke produced unexpected state: max_abs_diff={diff} state={:?} expected={:?}",
                final_state, expected_state
            );
            return ExitCode::from(1);
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }
    println!(
        "runtime-linear-attn-recurrent-smoke backend={} device_index={} name=\"{}\" key_heads={} value_heads={} sequence_len={} key_dim={} value_dim={} output={} state={} max_abs_diff={max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        format_f32_preview(&output[..8.min(output.len())]),
        format_f32_preview(&final_state[..8.min(final_state.len())]),
    );
    ExitCode::SUCCESS
}

fn runtime_mlp_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    const HIDDEN: usize = 4;
    const INTERMEDIATE: usize = 6;
    let epsilon = 1e-5_f32;

    let input = [0.45_f32, -1.20_f32, 0.95_f32, -0.35_f32];
    let norm_weight = [1.10_f32, -0.75_f32, 0.90_f32, 1.25_f32];
    let gate_matrix = [
        0.25_f32, -0.40_f32, 0.55_f32, 0.33_f32, //
        -0.60_f32, 0.80_f32, 0.45_f32, -0.70_f32, //
        1.10_f32, 0.20_f32, -0.30_f32, 0.45_f32, //
        0.65_f32, -0.55_f32, 0.85_f32, -0.10_f32, //
        -0.20_f32, 0.33_f32, 0.77_f32, -0.91_f32, //
        0.44_f32, -0.88_f32, 0.12_f32, 0.56_f32, //
    ];
    let up_matrix = [
        -0.30_f32, 0.70_f32, 0.90_f32, -0.50_f32, //
        1.05_f32, -0.95_f32, 0.25_f32, 0.60_f32, //
        0.20_f32, -0.15_f32, 0.40_f32, 1.10_f32, //
        -0.80_f32, 0.65_f32, 0.55_f32, -0.45_f32, //
        0.30_f32, 0.30_f32, 0.30_f32, 0.30_f32, //
        -0.25_f32, 1.20_f32, -1.10_f32, 0.45_f32, //
    ];
    let down_matrix = [
        0.50_f32, -0.30_f32, 0.70_f32, -0.60_f32, 0.40_f32, 0.20_f32, //
        0.10_f32, 0.90_f32, -0.40_f32, 0.80_f32, -0.15_f32, 0.60_f32, //
        -0.70_f32, 0.65_f32, 0.20_f32, 0.25_f32, 1.05_f32, -0.80_f32, //
        0.45_f32, -0.10_f32, -0.55_f32, 0.30_f32, 0.50_f32, 0.85_f32, //
    ];

    let expected_normed = runtime_host_rmsnorm_f32(&input, &norm_weight, epsilon);
    let expected_gate =
        runtime_host_matvec_f32(&gate_matrix, &expected_normed, INTERMEDIATE, HIDDEN);
    let expected_up = runtime_host_matvec_f32(&up_matrix, &expected_normed, INTERMEDIATE, HIDDEN);
    let expected_activated = runtime_host_silu_mul_f32(&expected_gate, &expected_up);
    let expected_output =
        runtime_host_matvec_f32(&down_matrix, &expected_activated, HIDDEN, INTERMEDIATE);

    let hidden_bytes = HIDDEN * std::mem::size_of::<f32>();
    let intermediate_bytes = INTERMEDIATE * std::mem::size_of::<f32>();
    let gate_matrix_byte_count = gate_matrix.len().checked_mul(std::mem::size_of::<f32>());
    if gate_matrix_byte_count.is_none() {
        eprintln!("gate matrix byte size overflows");
        return ExitCode::from(1);
    }
    let up_matrix_byte_count = up_matrix.len().checked_mul(std::mem::size_of::<f32>());
    if up_matrix_byte_count.is_none() {
        eprintln!("up matrix byte size overflows");
        return ExitCode::from(1);
    }
    let down_matrix_byte_count = down_matrix.len().checked_mul(std::mem::size_of::<f32>());
    if down_matrix_byte_count.is_none() {
        eprintln!("down matrix byte size overflows");
        return ExitCode::from(1);
    }
    let gate_matrix_bytes = gate_matrix_byte_count.unwrap();
    let up_matrix_bytes = up_matrix_byte_count.unwrap();
    let down_matrix_bytes = down_matrix_byte_count.unwrap();

    let mut input_buffer = match context.alloc_buffer(input.len() * std::mem::size_of::<f32>()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let input_bytes = encode_f32_to_bytes(&input);
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy input data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copy: {err}");
        return ExitCode::from(1);
    }

    let mut norm_weight_buffer =
        match context.alloc_buffer(norm_weight.len() * std::mem::size_of::<f32>()) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate norm weight buffer: {err}");
                return ExitCode::from(1);
            }
        };
    let norm_weight_bytes = encode_f32_to_bytes(&norm_weight);
    if let Err(err) = norm_weight_buffer.copy_from_host(0, &norm_weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy norm weight data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after norm weight copy: {err}");
        return ExitCode::from(1);
    }

    let mut gate_matrix_buffer = match context.alloc_buffer(gate_matrix_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate matrix buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let gate_matrix_bytes = encode_f32_to_bytes(&gate_matrix);
    if let Err(err) = gate_matrix_buffer.copy_from_host(0, &gate_matrix_bytes, Some(&mut stream)) {
        eprintln!("failed to copy gate matrix into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate matrix copy: {err}");
        return ExitCode::from(1);
    }

    let mut up_matrix_buffer = match context.alloc_buffer(up_matrix_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate up matrix buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let up_matrix_bytes = encode_f32_to_bytes(&up_matrix);
    if let Err(err) = up_matrix_buffer.copy_from_host(0, &up_matrix_bytes, Some(&mut stream)) {
        eprintln!("failed to copy up matrix into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after up matrix copy: {err}");
        return ExitCode::from(1);
    }

    let mut down_matrix_buffer = match context.alloc_buffer(down_matrix_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate down matrix buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let down_matrix_bytes = encode_f32_to_bytes(&down_matrix);
    if let Err(err) = down_matrix_buffer.copy_from_host(0, &down_matrix_bytes, Some(&mut stream)) {
        eprintln!("failed to copy down matrix into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after down matrix copy: {err}");
        return ExitCode::from(1);
    }

    let mut normed_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate normed output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::rmsnorm_f32(
        &input_buffer,
        &norm_weight_buffer,
        HIDDEN,
        epsilon,
        &mut normed_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime rmsnorm_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after rmsnorm: {err}");
        return ExitCode::from(1);
    }

    let mut gate_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &gate_matrix_buffer,
        &normed_buffer,
        INTERMEDIATE,
        HIDDEN,
        &mut gate_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime gate matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate matvec: {err}");
        return ExitCode::from(1);
    }

    let mut up_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate up output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &up_matrix_buffer,
        &normed_buffer,
        INTERMEDIATE,
        HIDDEN,
        &mut up_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime up matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after up matvec: {err}");
        return ExitCode::from(1);
    }

    let mut activated_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate activated output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::silu_mul_f32(
        &gate_buffer,
        &up_buffer,
        INTERMEDIATE,
        &mut activated_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime silu_mul_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after silu_mul: {err}");
        return ExitCode::from(1);
    }

    let mut output_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &down_matrix_buffer,
        &activated_buffer,
        HIDDEN,
        INTERMEDIATE,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime down matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output matvec: {err}");
        return ExitCode::from(1);
    }

    let mut output_bytes = vec![0_u8; hidden_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_bytes, Some(&mut stream)) {
        eprintln!("failed to copy runtime output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_bytes);

    if output.len() != expected_output.len()
        || expected_output
            .iter()
            .zip(output.iter())
            .any(|(expected, actual)| (*expected - *actual).abs() > 1e-4_f32)
    {
        eprintln!(
            "runtime mlp smoke produced unexpected output: output={:?} expected={:?}",
            output, expected_output
        );
        return ExitCode::from(1);
    }

    println!(
        "runtime-mlp-smoke backend={} device_index={} name=\"{}\" hidden={} intermediate={} output={} verified=true",
        info.backend,
        device_index,
        info.name,
        HIDDEN,
        INTERMEDIATE,
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn inspect_package(path: Option<String>) -> ExitCode {
    let Some(path) = path else {
        eprintln!("inspect-package requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let summary = match ullm_engine::package::inspect_package(path) {
        Ok(summary) => summary,
        Err(err) => {
            eprintln!("failed to inspect package: {err}");
            return ExitCode::from(1);
        }
    };
    println!("package: {}", summary.package_dir.display());
    println!(
        "schema: {}",
        summary
            .schema_version
            .unwrap_or_else(|| "unknown".to_string())
    );
    if let Some(source) = summary.source_model_dir {
        println!("source_model_dir: {source}");
    }
    println!("quantized_tensors: {}", summary.quantized_tensors);
    println!("passthrough_tensors: {}", summary.passthrough_tensors);
    println!("codebooks: {}", summary.codebooks);
    println!("quantized_elements: {}", summary.quantized_elements);
    println!("passthrough_elements: {}", summary.passthrough_elements);
    println!("referenced_files: {}", summary.referenced_files);
    println!("referenced_file_bytes: {}", summary.referenced_file_bytes);
    println!(
        "missing_referenced_files: {}",
        summary.missing_referenced_files
    );
    println!(
        "declared_passthrough_payload_bytes: {}",
        summary.declared_passthrough_payload_bytes
    );
    ExitCode::SUCCESS
}

fn package_layer_kind_inventory_smoke(
    path: Option<String>,
    layer_indices: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-layer-kind-inventory-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let layer_indices = match parse_package_token_ids_layer_indices_for_package(
        &path,
        layer_indices.or_else(|| Some("manifest-all".to_string())),
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let entries = match package_layer_entries_for_indices(&path, &layer_indices) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let self_attention_count = entries
        .iter()
        .filter(|entry| entry.kind == PackageDecoderLayerKind::SelfAttention)
        .count();
    let linear_attention_count = entries.len().saturating_sub(self_attention_count);
    let contiguous = package_layer_entries_are_contiguous(&entries);
    let layers = entries
        .iter()
        .map(|entry| entry.layer_index)
        .collect::<Vec<_>>();
    let layer_kinds = entries
        .iter()
        .map(|entry| entry.kind.as_str())
        .collect::<Vec<_>>();
    let report = serde_json::json!({
        "schema_version": "package-layer-kind-inventory-smoke-v0.1",
        "package": path,
        "layers": layers,
        "layer_kinds": layer_kinds,
        "layer_count": entries.len(),
        "self_attention_count": self_attention_count,
        "linear_attention_count": linear_attention_count,
        "mixed_attention": self_attention_count > 0 && linear_attention_count > 0,
        "contiguous_layer_indices": contiguous,
        "real_batch_requirements": [
            "self_attention requires per-request paged KV state and ready-batch decode inputs",
            "linear_attention requires per-request recurrent state and Conv1d history",
            "full mixed-attention runner must preserve manifest layer order across both layer kinds",
            "result rows must not be promoted to SQ throughput decisions until prefill/decode are real request-batch"
        ],
        "verified": true,
    });
    match serde_json::to_string_pretty(&report) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("failed to encode package layer kind inventory: {err}");
            ExitCode::from(1)
        }
    }
}

fn package_load_smoke(
    path: Option<String>,
    device_index: Option<String>,
    max_bytes: Option<String>,
    payload_role: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-load-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let max_bytes = match parse_optional_usize(max_bytes, 1024 * 1024, "max bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("max bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let payload_role = match parse_optional_payload_role(payload_role) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let summary = match ullm_engine::package::inspect_package(&path) {
        Ok(summary) => summary,
        Err(err) => {
            eprintln!("failed to inspect package: {err}");
            return ExitCode::from(1);
        }
    };
    let selected = match ullm_engine::package::select_existing_referenced_file(&path, payload_role)
    {
        Ok(selected) => selected,
        Err(err) => {
            eprintln!("failed to select package payload: {err}");
            return ExitCode::from(1);
        }
    };
    let data = match read_bounded_file(&selected.absolute_path, max_bytes) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    if data.is_empty() {
        eprintln!("selected payload produced zero bytes after applying max-bytes");
        return ExitCode::from(1);
    }

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };
    let mut buffer = match context.alloc_buffer(data.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = buffer.copy_from_host(0, &data, Some(&mut stream)) {
        eprintln!("failed to copy package payload into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after package payload load: {err}");
        return ExitCode::from(1);
    }
    let mut output = vec![0_u8; data.len()];
    if let Err(err) = buffer.copy_to_host(0, &mut output, Some(&mut stream)) {
        eprintln!("failed to copy package payload back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after package payload readback: {err}");
        return ExitCode::from(1);
    }
    if data != output {
        eprintln!("package payload roundtrip produced mismatched bytes");
        return ExitCode::from(1);
    }
    println!(
        "package-load-smoke package={} schema={} role={} file={} file_bytes={} copied_bytes={} owner_index={} owner_name=\"{}\" backend={} device_index={} name=\"{}\" verified=true",
        summary.package_dir.display(),
        summary
            .schema_version
            .unwrap_or_else(|| "unknown".to_string()),
        selected.role.as_str(),
        selected.relative_path,
        selected.bytes,
        data.len(),
        selected
            .owner_index
            .map(|index| index.to_string())
            .unwrap_or_else(|| "none".to_string()),
        selected.owner_name.unwrap_or_else(|| "none".to_string()),
        info.backend,
        device_index,
        info.name
    );
    ExitCode::SUCCESS
}

fn package_tensor_load_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    tensor_selector: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-tensor-load-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let selector = TensorSelector::parse(tensor_selector.as_deref());
    let bundle = match ullm_engine::package::select_tensor_payload_bundle(&path, &selector) {
        Ok(bundle) => bundle,
        Err(err) => {
            eprintln!("failed to select package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let index_summary =
        match roundtrip_file_chunks(&mut context, &mut stream, &bundle.index_file, chunk_bytes) {
            Ok(summary) => summary,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
    let scale_summary =
        match roundtrip_file_chunks(&mut context, &mut stream, &bundle.scale_file, chunk_bytes) {
            Ok(summary) => summary,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
    let codebook_summary = match roundtrip_file_chunks(
        &mut context,
        &mut stream,
        &bundle.codebook_file,
        chunk_bytes,
    ) {
        Ok(summary) => summary,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    println!(
        "package-tensor-load-smoke package={} tensor_index={} tensor=\"{}\" dtype={} family={} candidate_id={} elements={} groups={} backend={} device_index={} name=\"{}\" chunk_bytes={} verified=true",
        path,
        bundle.tensor_index,
        bundle.tensor_name,
        bundle.dtype.as_deref().unwrap_or("unknown"),
        bundle.family.as_deref().unwrap_or("unknown"),
        bundle.candidate_id.as_deref().unwrap_or("unknown"),
        bundle.elements,
        bundle.groups,
        info.backend,
        device_index,
        info.name,
        chunk_bytes
    );
    print_file_roundtrip_summary("tensor-index", &bundle.index_file, &index_summary);
    print_file_roundtrip_summary("tensor-scale", &bundle.scale_file, &scale_summary);
    print_file_roundtrip_summary("tensor-codebook", &bundle.codebook_file, &codebook_summary);
    ExitCode::SUCCESS
}

fn package_weight_register_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    tensor_selector: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-weight-register-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let selector = TensorSelector::parse(tensor_selector.as_deref());
    let bundle = match ullm_engine::package::select_tensor_payload_bundle(&path, &selector) {
        Ok(bundle) => bundle,
        Err(err) => {
            eprintln!("failed to select package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };
    let mut registry = WeightRegistry::new();
    let registry_index = match registry.load_and_insert(
        &mut context,
        &mut stream,
        &bundle,
        LoadOptions {
            chunk_bytes,
            verify: true,
        },
    ) {
        Ok(index) => index,
        Err(err) => {
            eprintln!("failed to register package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };
    let Some(loaded) = registry.get(registry_index) else {
        eprintln!("registered tensor disappeared from weight registry");
        return ExitCode::from(1);
    };

    println!(
        "package-weight-register-smoke package={} registry_index={} registry_tensors={} registry_payload_bytes={} resident_payload_bytes={} codebook_payloads={} tensor_index={} tensor=\"{}\" dtype={} family={} candidate_id={} elements={} groups={} backend={} device_index={} name=\"{}\" chunk_bytes={} verified=true",
        path,
        registry_index,
        registry.len(),
        registry.total_payload_bytes(),
        registry.resident_payload_bytes(),
        registry.codebook_payloads(),
        loaded.tensor_index,
        loaded.tensor_name,
        loaded.dtype.as_deref().unwrap_or("unknown"),
        loaded.family.as_deref().unwrap_or("unknown"),
        loaded.candidate_id.as_deref().unwrap_or("unknown"),
        loaded.elements,
        loaded.groups,
        info.backend,
        device_index,
        info.name,
        chunk_bytes
    );
    print_loaded_payload_summary(&loaded.index);
    print_loaded_payload_summary(&loaded.scale);
    print_loaded_payload_summary(&loaded.codebook);
    ExitCode::SUCCESS
}

fn package_weight_register_many_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    max_tensors: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-weight-register-many-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let max_tensors = match parse_optional_usize(max_tensors, 2, "max tensors") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("max tensors must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };
    let loaded_package = match load_package_tensor_prefix(
        &mut context,
        &mut stream,
        &path,
        max_tensors,
        LoadOptions {
            chunk_bytes,
            verify: true,
        },
    ) {
        Ok(loaded) => loaded,
        Err(err) => {
            eprintln!("failed to register package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };
    let registry = loaded_package.registry();

    println!(
        "package-weight-register-many-smoke package={} selected_tensors={} package_tensors={} registry_tensors={} registry_payload_bytes={} resident_payload_bytes={} codebook_payloads={} backend={} device_index={} name=\"{}\" chunk_bytes={} verified=true",
        path,
        loaded_package.loaded_tensor_count,
        loaded_package.summary.quantized_tensors,
        registry.len(),
        registry.total_payload_bytes(),
        registry.resident_payload_bytes(),
        registry.codebook_payloads(),
        info.backend,
        device_index,
        info.name,
        chunk_bytes
    );
    for &registry_index in &loaded_package.registry_indices {
        let Some(loaded) = registry.get(registry_index) else {
            eprintln!("registered tensor disappeared from weight registry");
            return ExitCode::from(1);
        };
        println!(
            "  registered_tensor registry_index={} tensor_index={} tensor=\"{}\" bytes={}",
            registry_index,
            loaded.tensor_index,
            loaded.tensor_name,
            loaded.total_payload_bytes()
        );
    }
    ExitCode::SUCCESS
}

fn package_materialize_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    tensor_selector: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-materialize-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let selector = TensorSelector::parse(tensor_selector.as_deref());
    let bundle = match ullm_engine::package::select_tensor_payload_bundle(&path, &selector) {
        Ok(bundle) => bundle,
        Err(err) => {
            eprintln!("failed to select package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };
    let mut registry = WeightRegistry::new();
    let registry_index = match registry.load_and_insert(
        &mut context,
        &mut stream,
        &bundle,
        LoadOptions {
            chunk_bytes,
            verify: true,
        },
    ) {
        Ok(index) => index,
        Err(err) => {
            eprintln!("failed to register package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };
    let Some(loaded) = registry.get(registry_index) else {
        eprintln!("registered tensor disappeared from weight registry");
        return ExitCode::from(1);
    };

    let materialize = match materialize_config(loaded) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let mut output = match context.alloc_buffer(materialize.output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate materialized output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::aq4_dequant_f32(
        loaded.index.buffer.as_ref(),
        loaded.scale.buffer.as_ref(),
        loaded.codebook.buffer.as_ref(),
        &materialize.scale_values,
        materialize.group_size,
        materialize.tensor_scale,
        materialize.elements,
        &mut output,
        Some(&mut stream),
    ) {
        eprintln!("failed to materialize AQ4 tensor: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after materialize: {err}");
        return ExitCode::from(1);
    }

    let preview_count = materialize.elements.min(8);
    let mut preview_bytes = vec![0_u8; preview_count * std::mem::size_of::<f32>()];
    if let Err(err) = output.copy_to_host(0, &mut preview_bytes, Some(&mut stream)) {
        eprintln!("failed to copy materialized preview back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after preview copy: {err}");
        return ExitCode::from(1);
    }
    let preview = decode_f32_le_values(&preview_bytes);
    println!(
        "package-materialize-smoke package={} registry_index={} tensor_index={} tensor=\"{}\" elements={} output_bytes={} scale_format={} scale_count={} group_size={} tensor_scale={:.9} backend={} device_index={} name=\"{}\" preview={} verified=true",
        path,
        registry_index,
        loaded.tensor_index,
        loaded.tensor_name,
        materialize.elements,
        materialize.output_bytes,
        materialize.scale_format,
        materialize.scale_values.len(),
        materialize.group_size,
        materialize.tensor_scale,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&preview)
    );
    ExitCode::SUCCESS
}

fn sq_fp8_materialize_smoke(
    path: Option<String>,
    device_index: Option<String>,
    tensor_selector: Option<String>,
    row_count: Option<String>,
    start_row: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("sq-fp8-materialize-smoke requires an SQ FP8 artifact directory");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let artifact = match read_sq_fp8_artifact(&path) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read SQ FP8 artifact: {err}");
            return ExitCode::from(1);
        }
    };
    let selector = TensorSelector::parse(tensor_selector.as_deref());
    let tensor_index = match select_sq_fp8_tensor_index(&artifact.manifest, &selector) {
        Ok(index) => index,
        Err(err) => {
            eprintln!("failed to select SQ FP8 tensor: {err}");
            return ExitCode::from(1);
        }
    };
    let tensor = &artifact.manifest.fp8_tensors[tensor_index];
    let (rows, cols) = match sq_fp8_tensor_rows_cols(tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("invalid SQ FP8 tensor entry: {err}");
            return ExitCode::from(1);
        }
    };
    let start_row = match parse_optional_usize(start_row, 0, "start row") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let row_count = match row_count {
        Some(value) => match parse_optional_usize(Some(value), 0, "row count") {
            Ok(value) if value > 0 => value,
            Ok(_) => {
                eprintln!("row count must be greater than zero");
                return ExitCode::from(2);
            }
            Err(code) => return code,
        },
        None => rows.saturating_sub(start_row).min(4),
    };
    if row_count == 0 {
        eprintln!("SQ FP8 row range is empty: start_row={start_row} rows={rows}");
        return ExitCode::from(2);
    }

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };
    let materialized = match materialize_sq_fp8_tensor_rows_to_runtime_f32(
        &mut context,
        &mut stream,
        &artifact,
        &selector,
        start_row,
        row_count,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize SQ FP8 tensor rows: {err}");
            return ExitCode::from(1);
        }
    };
    let output_bytes = match materialized
        .values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
    {
        Some(bytes) => bytes,
        None => {
            eprintln!("SQ FP8 materialized output byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut roundtrip_bytes = vec![0_u8; output_bytes];
    if let Err(err) = materialized
        .buffer
        .copy_to_host(0, &mut roundtrip_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy SQ FP8 materialized rows back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after SQ FP8 readback: {err}");
        return ExitCode::from(1);
    }
    let roundtrip = decode_f32_le_values(&roundtrip_bytes);
    if roundtrip.len() != materialized.values.len() {
        eprintln!(
            "SQ FP8 readback element count mismatch: expected {} got {}",
            materialized.values.len(),
            roundtrip.len()
        );
        return ExitCode::from(1);
    }
    let roundtrip_max_abs_diff = materialized
        .values
        .iter()
        .zip(roundtrip.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f32, f32::max);
    if roundtrip_max_abs_diff != 0.0 {
        eprintln!("SQ FP8 runtime roundtrip mismatch: max_abs_diff={roundtrip_max_abs_diff:.9}");
        return ExitCode::from(1);
    }
    let sq_candidate_legacy = if artifact.manifest.candidate.id == FORMAT_SQ8_0 {
        None
    } else {
        Some(artifact.manifest.candidate.id.as_str())
    };
    let sq_implementation_id = artifact
        .manifest
        .candidate
        .implementation_id
        .as_deref()
        .or(sq_candidate_legacy)
        .unwrap_or("none");
    let sq_candidate_legacy = sq_candidate_legacy.unwrap_or("none");
    let preview_count = roundtrip.len().min(8);
    println!(
        "sq-fp8-materialize-smoke artifact={} schema={} format_id={} candidate={} candidate_legacy={} sq_format_id={} sq_implementation_id={} tensor_index={} tensor=\"{}\" family={} source_dtype={} shape=[{},{}] rows={} cols={} start_row={} row_count={} materialized_elements={} output_bytes={} payload_dtype={} scale_granularity={} scale_dtype={} backend={} device_index={} name=\"{}\" preview={} roundtrip_max_abs_diff={roundtrip_max_abs_diff:.9} verified=true",
        path,
        artifact.manifest.schema_version,
        FORMAT_SQ8_0,
        FORMAT_SQ8_0,
        sq_candidate_legacy,
        FORMAT_SQ8_0,
        sq_implementation_id,
        materialized.tensor_index,
        materialized.tensor_name,
        tensor.family,
        tensor.source_dtype,
        materialized.rows,
        materialized.cols,
        rows,
        cols,
        materialized.start_row,
        materialized.row_count,
        materialized.values.len(),
        output_bytes,
        tensor.payload_dtype,
        tensor.scale_granularity,
        tensor.scale_dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&roundtrip[..preview_count])
    );
    ExitCode::SUCCESS
}

fn package_materialize_bench(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    tensor_selector: Option<String>,
    repeats: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-materialize-bench requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let repeats = match parse_optional_usize(repeats, 20, "repeats") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("repeats must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let selector = TensorSelector::parse(tensor_selector.as_deref());
    let bundle = match ullm_engine::package::select_tensor_payload_bundle(&path, &selector) {
        Ok(bundle) => bundle,
        Err(err) => {
            eprintln!("failed to select package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };
    let mut registry = WeightRegistry::new();
    let registry_index = match registry.load_and_insert(
        &mut context,
        &mut stream,
        &bundle,
        LoadOptions {
            chunk_bytes,
            verify: true,
        },
    ) {
        Ok(index) => index,
        Err(err) => {
            eprintln!("failed to register package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };
    let Some(loaded) = registry.get(registry_index) else {
        eprintln!("registered tensor disappeared from weight registry");
        return ExitCode::from(1);
    };
    let materialize = match materialize_config(loaded) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let mut output = match context.alloc_buffer(materialize.output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate materialized output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = ullm_runtime_sys::aq4_dequant_f32(
        loaded.index.buffer.as_ref(),
        loaded.scale.buffer.as_ref(),
        loaded.codebook.buffer.as_ref(),
        &materialize.scale_values,
        materialize.group_size,
        materialize.tensor_scale,
        materialize.elements,
        &mut output,
        Some(&mut stream),
    ) {
        eprintln!("failed to warm up AQ4 materialize: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize warmup materialize: {err}");
        return ExitCode::from(1);
    }

    let mut elapsed_ms = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        if let Err(err) = ullm_runtime_sys::aq4_dequant_f32(
            loaded.index.buffer.as_ref(),
            loaded.scale.buffer.as_ref(),
            loaded.codebook.buffer.as_ref(),
            &materialize.scale_values,
            materialize.group_size,
            materialize.tensor_scale,
            materialize.elements,
            &mut output,
            Some(&mut stream),
        ) {
            eprintln!("failed to materialize AQ4 tensor during benchmark: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize materialize benchmark: {err}");
            return ExitCode::from(1);
        }
        elapsed_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    elapsed_ms.sort_by(|left, right| left.total_cmp(right));
    let mean_ms = elapsed_ms.iter().sum::<f64>() / elapsed_ms.len() as f64;
    let min_ms = elapsed_ms[0];
    let p50_ms = elapsed_ms[elapsed_ms.len() / 2];
    let p95_index = ((elapsed_ms.len() - 1) * 95) / 100;
    let p95_ms = elapsed_ms[p95_index];
    let output_gib = materialize.output_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
    let output_gib_per_s = if mean_ms > 0.0 {
        output_gib / (mean_ms / 1000.0)
    } else {
        0.0
    };
    println!(
        "package-materialize-bench package={} registry_index={} tensor_index={} tensor=\"{}\" elements={} output_bytes={} scale_format={} scale_count={} group_size={} tensor_scale={:.9} backend={} device_index={} name=\"{}\" repeats={} mean_ms={:.6} min_ms={:.6} p50_ms={:.6} p95_ms={:.6} output_gib_per_s={:.6} verified=true",
        path,
        registry_index,
        loaded.tensor_index,
        loaded.tensor_name,
        materialize.elements,
        materialize.output_bytes,
        materialize.scale_format,
        materialize.scale_values.len(),
        materialize.group_size,
        materialize.tensor_scale,
        info.backend,
        device_index,
        info.name,
        repeats,
        mean_ms,
        min_ms,
        p50_ms,
        p95_ms,
        output_gib_per_s
    );
    ExitCode::SUCCESS
}

fn package_materialize_matvec_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    tensor_selector: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-materialize-matvec-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let selector = TensorSelector::parse(tensor_selector.as_deref());
    let bundle = match ullm_engine::package::select_tensor_payload_bundle(&path, &selector) {
        Ok(bundle) => bundle,
        Err(err) => {
            eprintln!("failed to select package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };
    let mut registry = WeightRegistry::new();
    let registry_index = match registry.load_and_insert(
        &mut context,
        &mut stream,
        &bundle,
        LoadOptions {
            chunk_bytes,
            verify: true,
        },
    ) {
        Ok(index) => index,
        Err(err) => {
            eprintln!("failed to register package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };
    let Some(loaded) = registry.get(registry_index) else {
        eprintln!("registered tensor disappeared from weight registry");
        return ExitCode::from(1);
    };
    let materialize = match materialize_config(loaded) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (rows, cols) = match matrix_shape_rows_cols(&loaded.shape, materialize.elements) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let mut matrix = match context.alloc_buffer(materialize.output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate materialized matrix buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::aq4_dequant_f32(
        loaded.index.buffer.as_ref(),
        loaded.scale.buffer.as_ref(),
        loaded.codebook.buffer.as_ref(),
        &materialize.scale_values,
        materialize.group_size,
        materialize.tensor_scale,
        materialize.elements,
        &mut matrix,
        Some(&mut stream),
    ) {
        eprintln!("failed to materialize AQ4 tensor: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after materialize: {err}");
        return ExitCode::from(1);
    }

    let mut input = Vec::with_capacity(cols);
    for i in 0..cols {
        input.push(((i % 17) as f32 - 8.0) / 16.0);
    }
    let input_byte_count = match cols.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("matvec input byte size overflows");
            return ExitCode::from(1);
        }
    };
    let output_byte_count = match rows.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("matvec output byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut input_bytes = Vec::with_capacity(input_byte_count);
    for value in &input {
        input_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy input vector into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copy: {err}");
        return ExitCode::from(1);
    }

    let mut output = match context.alloc_buffer(output_byte_count) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &matrix,
        &input_buffer,
        rows,
        cols,
        &mut output,
        Some(&mut stream),
    ) {
        eprintln!("failed to run matvec f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after matvec: {err}");
        return ExitCode::from(1);
    }

    let preview_count = rows.min(8);
    let mut preview_bytes = vec![0_u8; preview_count * std::mem::size_of::<f32>()];
    if let Err(err) = output.copy_to_host(0, &mut preview_bytes, Some(&mut stream)) {
        eprintln!("failed to copy matvec preview back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after preview copy: {err}");
        return ExitCode::from(1);
    }
    let preview = decode_f32_le_values(&preview_bytes);
    println!(
        "package-materialize-matvec-smoke package={} registry_index={} tensor_index={} tensor=\"{}\" elements={} rows={} cols={} output_bytes={} scale_format={} scale_count={} group_size={} tensor_scale={:.9} backend={} device_index={} name=\"{}\" preview={} verified=true",
        path,
        registry_index,
        loaded.tensor_index,
        loaded.tensor_name,
        materialize.elements,
        rows,
        cols,
        output_byte_count,
        materialize.scale_format,
        materialize.scale_values.len(),
        materialize.group_size,
        materialize.tensor_scale,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&preview)
    );
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use ullm_engine::backend_dispatch::{
        SQ8_0_MATVEC_BATCH_R9700_DIRECT_ID, SQ8_0_MATVEC_BATCH_RDNA4_DIRECT_ID,
        SQ8_0_MATVEC_PAIR_GENERIC_DIRECT_ID, SQ8_0_MATVEC_R9700_DIRECT_ID,
        SQ8_0_MATVEC_TRIPLE_R9700_DIRECT_ID, SQ8_0_MATVEC_TRIPLE_RDNA4_DIRECT_ID,
    };
    use ullm_runtime_sys::DeviceInfo;

    fn sq_fp8_projection_dispatch_non_direct_fixture(
        operation: SqFp8ProjectionMatvecOperation,
    ) -> SqFp8ProjectionDispatch {
        let implementation_id = match operation {
            SqFp8ProjectionMatvecOperation::Single => "sq8_0_matvec_generic_legacy",
            SqFp8ProjectionMatvecOperation::Batch => "sq8_0_matvec_batch_generic_legacy",
            SqFp8ProjectionMatvecOperation::Pair => "sq8_0_matvec_pair_generic_legacy",
            SqFp8ProjectionMatvecOperation::Triple => "sq8_0_matvec_triple_generic_legacy",
        };
        SqFp8ProjectionDispatch {
            operation,
            implementation_id,
            family: sq8_0_projection_descriptor_family(implementation_id),
        }
    }

    fn sq_fp8_projection_dispatch_fixture(
        operation: SqFp8ProjectionMatvecOperation,
        implementation_id: &'static str,
    ) -> SqFp8ProjectionDispatch {
        SqFp8ProjectionDispatch {
            operation,
            implementation_id,
            family: sq8_0_projection_descriptor_family(implementation_id),
        }
    }

    #[test]
    fn sq_fp8_projection_model_arch_does_not_change_default_selection() {
        let info = DeviceInfo {
            device_id: 0,
            backend: "cpu".to_string(),
            name: "unknown".to_string(),
            total_global_mem: 0,
            compute_major: 12,
            compute_minor: 0,
            gcn_arch_name: "gfx12".to_string(),
            flags: 0,
        };
        for operation in &[
            SqFp8ProjectionMatvecOperation::Single,
            SqFp8ProjectionMatvecOperation::Batch,
            SqFp8ProjectionMatvecOperation::Pair,
            SqFp8ProjectionMatvecOperation::Triple,
        ] {
            assert_eq!(
                select_sq_fp8_projection_implementation_id(*operation, &info, None),
                select_sq_fp8_projection_implementation_id(
                    *operation,
                    &info,
                    Some(SQ8_0_MODEL_ARCH_QWEN_FAMILY)
                )
            );
        }
    }

    #[test]
    fn sq_fp8_projection_dispatch_rejects_non_direct_family_for_single_batch_pair_and_triple_boundaries() {
        for operation in &[
            SqFp8ProjectionMatvecOperation::Single,
            SqFp8ProjectionMatvecOperation::Batch,
            SqFp8ProjectionMatvecOperation::Pair,
            SqFp8ProjectionMatvecOperation::Triple,
        ] {
            let dispatch = sq_fp8_projection_dispatch_non_direct_fixture(*operation);
            assert_eq!(dispatch.family, None);
            let err = dispatch
                .require_direct_family("sq_fp8_projection_boundary")
                .expect_err("expected non-direct projection family to be rejected");
            assert!(err.contains("has no direct kernel family"));
            assert!(err.contains(operation.label()));
            assert!(err.contains(dispatch.implementation_id));
        }
    }

    #[test]
    fn sq_fp8_projection_kernel_families_returns_none_without_projection_boundary() {
        let dispatches = SqFp8ProjectionDispatches {
            single: sq_fp8_projection_dispatch_fixture(
                SqFp8ProjectionMatvecOperation::Single,
                SQ8_0_MATVEC_R9700_DIRECT_ID,
            ),
            batch: sq_fp8_projection_dispatch_fixture(
                SqFp8ProjectionMatvecOperation::Batch,
                SQ8_0_MATVEC_BATCH_R9700_DIRECT_ID,
            ),
            pair: sq_fp8_projection_dispatch_fixture(
                SqFp8ProjectionMatvecOperation::Pair,
                SQ8_0_MATVEC_PAIR_GENERIC_DIRECT_ID,
            ),
            triple: sq_fp8_projection_dispatch_fixture(
                SqFp8ProjectionMatvecOperation::Triple,
                SQ8_0_MATVEC_TRIPLE_R9700_DIRECT_ID,
            ),
        };
        let telemetry = SqFp8ProjectionTelemetry::default();
        assert_eq!(
            sq_fp8_projection_kernel_families(telemetry, dispatches),
            "none"
        );
    }

    #[test]
    fn sq_fp8_projection_kernel_families_tracks_direct_boundaries() {
        let dispatches = SqFp8ProjectionDispatches {
            single: sq_fp8_projection_dispatch_fixture(
                SqFp8ProjectionMatvecOperation::Single,
                SQ8_0_MATVEC_R9700_DIRECT_ID,
            ),
            batch: sq_fp8_projection_dispatch_fixture(
                SqFp8ProjectionMatvecOperation::Batch,
                SQ8_0_MATVEC_BATCH_R9700_DIRECT_ID,
            ),
            pair: sq_fp8_projection_dispatch_fixture(
                SqFp8ProjectionMatvecOperation::Pair,
                SQ8_0_MATVEC_PAIR_GENERIC_DIRECT_ID,
            ),
            triple: sq_fp8_projection_dispatch_fixture(
                SqFp8ProjectionMatvecOperation::Triple,
                SQ8_0_MATVEC_TRIPLE_RDNA4_DIRECT_ID,
            ),
        };
        let telemetry = SqFp8ProjectionTelemetry {
            single_matvec_count: 1,
            batch_matvec_count: 2,
            pair_matvec_count: 0,
            triple_matvec_count: 3,
        };
        assert_eq!(
            sq_fp8_projection_kernel_families(telemetry, dispatches),
            "single=direct,batch=direct,triple=direct"
        );
    }

    #[test]
    fn dispatchable_sq8_projection_gpu_name_normalizes_amd_radeon_graphics_rdna4_r9700() {
        let info = DeviceInfo {
            device_id: 0,
            backend: "hip".to_string(),
            name: RDNA4_GFX1201_R9700_NAME.to_string(),
            total_global_mem: 32 * 1024 * 1024 * 1024,
            compute_major: 12,
            compute_minor: 0,
            gcn_arch_name: "".to_string(),
            flags: 0,
        };
        assert_eq!(
            dispatchable_sq8_projection_gpu_name(&info).as_ref(),
            RDNA4_GFX1201_R9700_CANONICAL_NAME
        );
        assert_eq!(
            select_sq_fp8_projection_implementation_id(
                SqFp8ProjectionMatvecOperation::Single,
                &info,
                Some(SQ8_0_MODEL_ARCH_QWEN_FAMILY),
            ),
            SQ8_0_MATVEC_R9700_DIRECT_ID
        );
    }

    #[test]
    fn dispatchable_sq8_projection_gpu_name_leaves_non_matching_rocm_amd_graphics_name() {
        let info = DeviceInfo {
            device_id: 0,
            backend: "hip".to_string(),
            name: RDNA4_GFX1201_R9700_NAME.to_string(),
            total_global_mem: 16 * 1024 * 1024 * 1024,
            compute_major: 12,
            compute_minor: 0,
            gcn_arch_name: "gfx1201".to_string(),
            flags: 0,
        };
        assert_eq!(
            dispatchable_sq8_projection_gpu_name(&info).as_ref(),
            RDNA4_GFX1201_R9700_NAME
        );
        assert_eq!(
            select_sq_fp8_projection_implementation_id(
                SqFp8ProjectionMatvecOperation::Batch,
                &info,
                Some(SQ8_0_MODEL_ARCH_QWEN_FAMILY),
            ),
            SQ8_0_MATVEC_BATCH_RDNA4_DIRECT_ID
        );
    }
}
