//! Backend-independent runtime contracts for sLLM.
//!
//! Phase 1 deliberately contains descriptors and control-plane behavior only.
//! It does not allocate model data, emulate a GPU, or execute numerical work.

mod backend;
mod dtype;
mod execution;
mod fake;
mod final_output;
mod fp8;
mod fp8_sidecar;
mod handles;
mod kv_state;
mod linear_attention;
mod model;
mod op;
mod prepared_execution;
mod qwen_execution;
mod qwen_graph;
mod registry;
mod sampling;
mod tensor;
mod weights;

pub use backend::{
    Backend, BackendCapabilities, BackendError, BackendSupport, ExecutionReceipt,
    MaterializedTensor,
};
pub use dtype::{DType, Encoding, EncodingError, Fp8ResidentRepresentation, Fp8ScaleGranularity};
pub use execution::{
    AdapterResource, AllocationCategory, AllocationCategorySnapshot, AllocationSnapshot,
    BoundSemanticOp, BufferRange, BufferReadback, CausalAttentionSubmission, DispatchEvidence,
    ExecutionAdapterAccess, ExecutionBuffer, ExecutionBufferId,
    ExecutionCausalAttentionSubmissionAdapter, ExecutionError, ExecutionKvStateSubmissionAdapter,
    ExecutionLinearAttentionSubmissionAdapter, ExecutionQueue, ExecutionQueueId,
    ExecutionReadbackAdapter, ExecutionSession, ExecutionSessionAdapter, ExecutionSessionId,
    ExecutionSessionRequest, ExecutionState, ExecutionSubmissionAdapter, ExecutionTransferAdapter,
    KvState, KvStateAppendSubmission, KvStateId, LinearAttentionBindings, LinearAttentionState,
    LinearAttentionStateId, LinearAttentionSubmission, OwnedTensorBinding, PrepareSupport,
    PreparedOperation, PreparedOperationId, Readback, ShutdownReport, Submission, Transfer,
};
pub use fake::{FakeBackend, MAX_FAKE_MATERIALIZATION_BYTES};
pub use final_output::{
    QWEN35_EMBEDDING_TENSOR, QWEN35_HIDDEN_SIZE, QWEN35_VOCAB_SIZE, QwenFinalOutputBindings,
};
pub use fp8::{
    E4M3FN_MAX, E4M3FNUZ_MAX, Fp8Error, Fp8Provider, Fp8ProviderRejection, Fp8ProviderRequest,
    QuantizedFp8, convert_e4m3fn_to_e4m3fnuz, decode_e4m3fn, decode_e4m3fnuz, encode_e4m3fn,
    encode_e4m3fnuz, quantize_e4m3fn_k_blocks, quantize_e4m3fn_outer_rows, select_fp8_provider,
};
pub use fp8_sidecar::{Fp8SidecarError, Fp8SidecarTensor, VerifiedFp8Sidecar, verify_fp8_sidecar};
pub use handles::{
    AccessMode, BufferHandle, BufferUse, CompletionLease, EventHandle, InFlightSubmission,
    QueueHandle,
};
pub use kv_state::{
    CausalAttentionDescriptor, KvMemoryKind, KvPhysicalMemorySnapshot, KvStateAppendRequest,
    KvStateDescriptor, KvStateError, KvStateLayout, KvStateSnapshot,
};
pub use linear_attention::{
    LinearAttentionDescriptor, LinearAttentionError, LinearAttentionLayout, LinearAttentionRequest,
    LinearAttentionStateDescriptor, LinearAttentionStateSnapshot,
};
pub use model::{
    AccumulationDType, BaseModel, BudgetBoundary, ClassificationStatus, ComponentMetadata,
    ComponentStatus, ConfigEos, ExcludedFile, FrontendAssetKind, GenerationConfig,
    GenerationStopPolicyV1, LayerSchedule, LayerType, LicenseInfo, LockedFile, LockedModel,
    MaxNewTokensZero, ModelArchitecture, ModelError, ModelLock, NormalizationContract,
    NormalizationKind, PromptEvaluation, QWEN35_2B_FINGERPRINT, QWEN35_2B_REPO_ID,
    QWEN35_2B_REVISION, QWEN35_4B_FINGERPRINT, QWEN35_4B_REPO_ID, QWEN35_4B_REVISION,
    QWEN35_9B_FINGERPRINT, QWEN35_9B_REPO_ID, QWEN35_9B_REVISION, Qwen35ReviewedSpec,
    RopeParameters, RopeType, ScaleMode, SliceContract, StopEvaluation, StopIdentity,
    StopTokenHandling, TensorClassification, TensorContract, TensorDType, TensorDescriptor,
    TextConfig, TokenizerContract, TokenizerEos, VerifiedCache, VerifiedFile, fingerprint_for_json,
    parse_model_lock, qwen35_reviewed_spec, read_model_lock, reviewed_qwen35_spec,
    validate_model_config, verify_model_cache,
};
pub use op::{
    ArgmaxTensor, AttentionPreprocessContract, AttentionPreprocessPacking,
    AttentionPreprocessPositionMode, AttentionPreprocessTensor, ElementwiseTensor, OpError,
    RmsNormAliasPolicy, RmsNormContract, RmsNormEpsilon, RmsNormScaleMode, RmsNormTensor,
    SemanticOp, SemanticOpDescriptor, SemanticOpKind,
};
pub use prepared_execution::{
    ExecutionBoundaryKind, PreparedCachePolicy, PreparedDynamicIdentity, PreparedExecutionAudit,
    PreparedExecutionError, PreparedExecutionPlan, PreparedPlanNode, PreparedTransition,
};
pub use qwen_execution::{
    QwenExecutionAudit, QwenExecutionError, QwenExecutionOutput, QwenExecutionRequest,
    QwenKvLayerMemoryAudit, QwenRequestMemoryAudit, QwenResidentModel,
};
pub use qwen_graph::{
    QWEN35_LAYER_COUNT, QWEN35_LAYER_TYPES, QWEN35_MAX_POSITION_EMBEDDINGS,
    QWEN35_PLAN_ENTRY_COUNT, QWEN35_REQUIRED_WEIGHT_COUNT, QwenGraph, QwenGraphDispatchError,
    QwenGraphError, QwenGraphNode, QwenGraphNodeKind, QwenGraphState, QwenGraphStateDescriptor,
    QwenGraphStateKind, QwenGraphTensor, QwenGraphTensorBacking, QwenGraphWeightBinding,
    build_qwen35_fp8_fnuz_graph, build_qwen35_fp8_graph, build_qwen35_graph,
};
pub use registry::{BACKEND_REGISTRY, BackendRegistration, backend_registry};
pub use sampling::{
    OsSamplingRandom, ProfileSamplerV1, SamplingError, SamplingParametersV1, SamplingRandomSource,
};
pub use tensor::{TensorError, TensorView};
pub use weights::{
    WEIGHT_LOAD_CHUNK_BYTES, WeightClassification, WeightConsumer, WeightConsumerKey,
    WeightLoadChunk, WeightLoadEntry, WeightLoadPlan, WeightPlanError, WeightUploadError,
    WeightUploadReceipt, WeightUploadRequest, build_verified_weight_load_plan,
    build_weight_load_plan, upload_verified_weight,
};

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn dtype_and_encoding_are_independent() {
        let view = TensorView::contiguous(DType::Bf16, &[3, 5, 7]).expect("valid view");

        assert_eq!(view.dtype(), DType::Bf16);
        assert_eq!(view.encoding(), Encoding::Unquantized);
        assert_eq!(view.element_count(), 105);
        assert_eq!(view.payload_bytes(), 210);
        assert_eq!(view.end_offset(), 210);
        assert_eq!(view.span_bytes(), 210);
    }

    #[test]
    fn contiguous_view_uses_element_strides_and_handles_zero_extent() {
        let view = TensorView::contiguous(DType::F32, &[3, 5, 7]).expect("valid view");
        assert_eq!(view.strides(), &[35, 7, 1]);
        assert!(view.is_contiguous());

        let empty = TensorView::contiguous(DType::F32, &[0, 7]).expect("valid empty view");
        assert_eq!(empty.element_count(), 0);
        assert_eq!(empty.span_bytes(), 0);
    }

    #[test]
    fn tensor_offsets_obey_dtype_alignment_but_packed_nvfp4_is_byte_aligned() {
        assert!(matches!(
            TensorView::new(DType::F32, Encoding::Unquantized, &[1], &[1], 2),
            Err(TensorError::MisalignedOffset {
                offset: 2,
                alignment: 4
            })
        ));
        assert!(TensorView::new(DType::Bf16, Encoding::Unquantized, &[1], &[1], 2).is_ok());
        assert!(
            TensorView::new(
                DType::U8,
                Encoding::Nvfp4 {
                    block_size: 16,
                    scale_dtype: DType::F32,
                },
                &[17],
                &[1],
                1,
            )
            .is_ok()
        );
    }

    #[test]
    fn scalar_zero_and_nvfp4_boundaries_are_explicit() {
        let scalar = TensorView::contiguous(DType::F32, &[]).expect("scalar is one element");
        assert_eq!(scalar.element_count(), 1);
        assert_eq!(scalar.span_bytes(), 4);

        let empty = TensorView::with_encoding(
            DType::U8,
            Encoding::Nvfp4 {
                block_size: 16,
                scale_dtype: DType::F16,
            },
            &[0],
        )
        .expect("zero extent is representable");
        assert_eq!(empty.element_count(), 0);
        assert_eq!(empty.span_bytes(), 0);

        let fifteen = TensorView::with_encoding(
            DType::U8,
            Encoding::Nvfp4 {
                block_size: 16,
                scale_dtype: DType::F16,
            },
            &[15],
        )
        .expect("first block");
        let sixteen = TensorView::with_encoding(
            DType::U8,
            Encoding::Nvfp4 {
                block_size: 16,
                scale_dtype: DType::F16,
            },
            &[16],
        )
        .expect("first block boundary");
        let seventeen = TensorView::with_encoding(
            DType::U8,
            Encoding::Nvfp4 {
                block_size: 16,
                scale_dtype: DType::F16,
            },
            &[17],
        )
        .expect("second block boundary");
        assert_eq!(fifteen.span_bytes(), 8 + 2);
        assert_eq!(sixteen.span_bytes(), 8 + 2);
        assert_eq!(seventeen.span_bytes(), 9 + 4);
    }

    #[test]
    fn tensor_shape_and_span_overflow_fail_closed() {
        assert!(matches!(
            TensorView::contiguous(DType::F32, &[usize::MAX, 2]),
            Err(TensorError::ShapeOverflow)
        ));
        assert!(matches!(
            TensorView::new(DType::F32, Encoding::Unquantized, &[1], &[1], u64::MAX - 3),
            Err(TensorError::SizeOverflow)
        ));
        assert_eq!(
            Encoding::Nvfp4 {
                block_size: 16,
                scale_dtype: DType::F16,
            }
            .storage_bytes(DType::U8, u64::MAX),
            Ok(0xA000_0000_0000_0000)
        );
        assert_eq!(
            Encoding::Nvfp4 {
                block_size: 2,
                scale_dtype: DType::F16,
            }
            .storage_bytes(DType::U8, u64::MAX),
            Err(EncodingError::SizeOverflow)
        );
        assert_eq!(
            Encoding::Nvfp4 {
                block_size: 1,
                scale_dtype: DType::F32,
            }
            .storage_bytes(DType::U8, u64::MAX),
            Err(EncodingError::SizeOverflow)
        );
        assert!(matches!(
            TensorView::with_encoding(
                DType::U8,
                Encoding::Nvfp4 {
                    block_size: 0,
                    scale_dtype: DType::F16,
                },
                &[1],
            ),
            Err(TensorError::InvalidEncoding(EncodingError::ZeroBlockSize))
        ));
    }

    #[test]
    fn access_modes_and_completion_lease_hold_opaque_resources() {
        assert_eq!(
            AccessMode::Read.join(AccessMode::Write),
            AccessMode::ReadWrite
        );
        assert_eq!(
            AccessMode::ReadWrite.join(AccessMode::Read),
            AccessMode::ReadWrite
        );
        assert!(QueueHandle::from_raw(0).is_none());

        let queue = Arc::new(QueueHandle::from_raw(11).expect("non-zero queue handle"));
        let event = Arc::new(EventHandle::from_raw(13).expect("non-zero event handle"));
        let buffer = Arc::new(BufferHandle::from_raw(17).expect("non-zero buffer handle"));
        let buffer_use = BufferUse::new(Arc::clone(&buffer), AccessMode::ReadWrite);
        let lease =
            CompletionLease::new(Arc::clone(&queue), Arc::clone(&event), [buffer_use.clone()]);

        assert_eq!(lease.queue().raw(), 11);
        assert_eq!(lease.event().raw(), 13);
        assert_eq!(lease.buffers(), &[buffer_use]);
        assert!(lease.holds_buffer(&buffer));
        assert_eq!(Arc::strong_count(&queue), 2);
        assert_eq!(Arc::strong_count(&event), 2);
        assert_eq!(Arc::strong_count(&buffer), 2);
        drop(lease);
        assert_eq!(Arc::strong_count(&queue), 1);
        assert_eq!(Arc::strong_count(&event), 1);
        assert_eq!(Arc::strong_count(&buffer), 1);
    }

    #[test]
    fn fake_backend_accepts_exact_limit_and_rejects_one_byte_over() {
        let backend = FakeBackend::new();
        let exact_elements = MAX_FAKE_MATERIALIZATION_BYTES / 2;
        let exact = TensorView::contiguous(DType::Bf16, &[exact_elements as usize])
            .expect("exact limit is representable");
        assert_eq!(
            backend
                .materialize(&exact)
                .expect("exact limit accepted")
                .byte_len(),
            MAX_FAKE_MATERIALIZATION_BYTES
        );

        let over = TensorView::contiguous(DType::Bf16, &[(exact_elements + 1) as usize])
            .expect("one byte over is representable");
        assert_eq!(
            backend
                .materialize(&over)
                .expect_err("one byte over is rejected"),
            BackendError::MaterializationTooLarge {
                requested_bytes: MAX_FAKE_MATERIALIZATION_BYTES + 2,
                max_bytes: MAX_FAKE_MATERIALIZATION_BYTES,
            }
        );
    }

    #[test]
    fn fake_backend_never_executes_numerical_operations() {
        let input = TensorView::contiguous(DType::Bf16, &[3, 5]).expect("valid input");
        let output = TensorView::contiguous(DType::Bf16, &[3, 5]).expect("valid output");
        let operation = SemanticOp::new(
            SemanticOpKind::Add,
            vec![input.clone(), input],
            vec![output],
        )
        .expect("valid add descriptor");
        let backend = FakeBackend::new();

        assert!(matches!(
            backend.execute(&operation),
            Err(BackendError::NumericalExecutionUnsupported)
        ));
        assert!(matches!(
            backend.open_execution_session(
                ExecutionSessionRequest::new(0, "fake").expect("valid session request")
            ),
            Err(ExecutionError::ExecutionUnavailable {
                backend: "fake",
                ..
            })
        ));
    }

    #[test]
    fn semantic_operations_reject_wrong_arity_and_metadata() {
        let f32_2x2 = TensorView::contiguous(DType::F32, &[2, 2]).expect("valid tensor");
        let f32_2x3 = TensorView::contiguous(DType::F32, &[2, 3]).expect("valid tensor");
        let f16_2x2 = TensorView::contiguous(DType::F16, &[2, 2]).expect("valid tensor");

        assert!(matches!(
            SemanticOpDescriptor::new(SemanticOpKind::Copy, vec![], vec![f32_2x2.clone()]),
            Err(OpError::Arity { .. })
        ));
        assert!(matches!(
            SemanticOpDescriptor::new(
                SemanticOpKind::Copy,
                vec![f32_2x2.clone()],
                vec![f32_2x3.clone()],
            ),
            Err(OpError::CopyMetadataMismatch)
        ));
        assert!(matches!(
            SemanticOpDescriptor::new(
                SemanticOpKind::Add,
                vec![f32_2x2.clone(), f16_2x2.clone()],
                vec![f32_2x2.clone()],
            ),
            Err(OpError::ElementwiseMetadataMismatch)
        ));
        assert!(matches!(
            SemanticOpDescriptor::new(
                SemanticOpKind::Matmul,
                vec![f32_2x2.clone(), f32_2x3],
                vec![f32_2x2],
            ),
            Err(OpError::MatmulShapeMismatch)
        ));
        assert!(matches!(
            SemanticOpDescriptor::new(
                SemanticOpKind::Copy,
                vec![TensorView::contiguous(DType::F32, &[2, 2]).expect("valid tensor")],
                vec![TensorView::contiguous(DType::F32, &[2, 2]).expect("valid tensor")],
            ),
            Err(OpError::ElementwiseUnsupportedDType { .. })
        ));
        let valid_copy = SemanticOpDescriptor::new(
            SemanticOpKind::Copy,
            vec![TensorView::contiguous(DType::Bf16, &[2, 2]).expect("valid tensor")],
            vec![TensorView::contiguous(DType::Bf16, &[2, 2]).expect("valid tensor")],
        )
        .expect("valid copy descriptor");
        assert_eq!(valid_copy.arity(), (1, 1));
        assert_eq!(SemanticOpKind::Copy.arity(), (1, 1));
        assert_eq!(SemanticOpKind::Add.arity(), (2, 1));
        assert_eq!(SemanticOpKind::Embedding.arity(), (2, 1));
        assert_eq!(SemanticOpKind::Matmul.arity(), (2, 1));
        assert_eq!(SemanticOpKind::SiluMul.arity(), (2, 1));
        assert_eq!(SemanticOpKind::SigmoidMul.arity(), (2, 1));
        assert_eq!(SemanticOpKind::RmsNorm.arity(), (2, 1));
        assert_eq!(SemanticOpKind::Argmax.arity(), (1, 1));
    }

    #[test]
    fn argmax_freezes_rank_dtype_layout_shape_and_vocab_boundaries() {
        for (m, v) in [
            (1_usize, 1_usize),
            (3, 3),
            (17, 17),
            (1, 255),
            (3, 256),
            (17, 257),
            (1, 248_320),
            (1, 1_048_576),
        ] {
            let descriptor = SemanticOpDescriptor::new(
                SemanticOpKind::Argmax,
                vec![TensorView::contiguous(DType::Bf16, &[m, v]).unwrap()],
                vec![TensorView::contiguous(DType::I32, &[m]).unwrap()],
            )
            .expect("valid greedy argmax descriptor");
            assert_eq!(descriptor.kind(), SemanticOpKind::Argmax);
            assert_eq!(descriptor.arity(), (1, 1));
        }

        let valid_logits = TensorView::contiguous(DType::Bf16, &[3, 17]).unwrap();
        let valid_output = TensorView::contiguous(DType::I32, &[3]).unwrap();
        let error = |logits, output| {
            SemanticOpDescriptor::new(SemanticOpKind::Argmax, vec![logits], vec![output])
                .expect_err("invalid argmax descriptor")
        };

        assert_eq!(
            error(
                TensorView::contiguous(DType::Bf16, &[17]).unwrap(),
                valid_output.clone(),
            ),
            OpError::ArgmaxRankMismatch {
                tensor: ArgmaxTensor::Logits,
            }
        );
        assert_eq!(
            error(
                valid_logits.clone(),
                TensorView::contiguous(DType::I32, &[3, 1]).unwrap(),
            ),
            OpError::ArgmaxRankMismatch {
                tensor: ArgmaxTensor::Output,
            }
        );
        assert_eq!(
            error(
                TensorView::contiguous(DType::Bf16, &[0, 17]).unwrap(),
                TensorView::contiguous(DType::I32, &[0]).unwrap(),
            ),
            OpError::ArgmaxZeroExtent {
                tensor: ArgmaxTensor::Logits,
            }
        );
        assert_eq!(
            error(
                TensorView::new(DType::Bf16, Encoding::Unquantized, &[3, 17], &[18, 1], 0,)
                    .unwrap(),
                valid_output.clone(),
            ),
            OpError::ArgmaxNonContiguous {
                tensor: ArgmaxTensor::Logits,
            }
        );
        assert_eq!(
            error(
                TensorView::contiguous(DType::F16, &[3, 17]).unwrap(),
                valid_output.clone(),
            ),
            OpError::ArgmaxUnsupportedDType {
                tensor: ArgmaxTensor::Logits,
                expected: DType::Bf16,
                actual: DType::F16,
            }
        );
        assert_eq!(
            error(
                valid_logits.clone(),
                TensorView::contiguous(DType::F32, &[3]).unwrap(),
            ),
            OpError::ArgmaxUnsupportedDType {
                tensor: ArgmaxTensor::Output,
                expected: DType::I32,
                actual: DType::F32,
            }
        );
        let encoded = TensorView::with_encoding(
            DType::U8,
            Encoding::Nvfp4 {
                block_size: 32,
                scale_dtype: DType::Bf16,
            },
            &[3, 17],
        )
        .unwrap();
        assert_eq!(
            error(encoded, valid_output.clone()),
            OpError::ArgmaxUnsupportedEncoding {
                tensor: ArgmaxTensor::Logits,
                actual: Encoding::Nvfp4 {
                    block_size: 32,
                    scale_dtype: DType::Bf16,
                },
            }
        );
        assert_eq!(
            error(
                valid_logits,
                TensorView::contiguous(DType::I32, &[2]).unwrap(),
            ),
            OpError::ArgmaxShapeMismatch
        );
        assert_eq!(
            error(
                TensorView::contiguous(DType::Bf16, &[1, 1_048_577]).unwrap(),
                TensorView::contiguous(DType::I32, &[1]).unwrap(),
            ),
            OpError::ArgmaxVocabTooLarge { vocab: 1_048_577 }
        );
    }

    #[test]
    fn kv_state_is_not_accidentally_added_to_semantic_op_kind() {
        let semantic_kinds = [
            SemanticOpKind::Copy,
            SemanticOpKind::Add,
            SemanticOpKind::Embedding,
            SemanticOpKind::Matmul,
            SemanticOpKind::SiluMul,
            SemanticOpKind::SigmoidMul,
            SemanticOpKind::RmsNorm,
            SemanticOpKind::AttentionPreprocess,
        ];
        assert_eq!(semantic_kinds.len(), 8);
        assert!(
            semantic_kinds
                .iter()
                .all(|kind| !kind.name().contains("kv") && !kind.name().contains("state"))
        );
        assert_eq!(KvStateLayout::HEADS, 4);
    }

    #[test]
    fn baseline_elementwise_rejects_scalar_zero_strided_dtype_and_encoding() {
        let expect_copy_error = |view: TensorView, expected: OpError| {
            assert_eq!(
                SemanticOpDescriptor::new(SemanticOpKind::Copy, vec![view.clone()], vec![view],),
                Err(expected)
            );
        };

        expect_copy_error(
            TensorView::contiguous(DType::Bf16, &[]).expect("representable scalar"),
            OpError::ElementwiseRankZero {
                kind: SemanticOpKind::Copy,
                tensor: ElementwiseTensor::Input0,
            },
        );
        expect_copy_error(
            TensorView::contiguous(DType::Bf16, &[3, 0, 5]).expect("representable zero extent"),
            OpError::ElementwiseZeroExtent {
                kind: SemanticOpKind::Copy,
                tensor: ElementwiseTensor::Input0,
            },
        );
        expect_copy_error(
            TensorView::new(DType::Bf16, Encoding::Unquantized, &[2, 3], &[4, 1], 0)
                .expect("representable strided view"),
            OpError::ElementwiseNonContiguous {
                kind: SemanticOpKind::Copy,
                tensor: ElementwiseTensor::Input0,
            },
        );
        expect_copy_error(
            TensorView::contiguous(DType::F16, &[7]).expect("representable f16 view"),
            OpError::ElementwiseUnsupportedDType {
                kind: SemanticOpKind::Copy,
                tensor: ElementwiseTensor::Input0,
                actual: DType::F16,
            },
        );
        expect_copy_error(
            TensorView::with_encoding(
                DType::U8,
                Encoding::Nvfp4 {
                    block_size: 32,
                    scale_dtype: DType::Bf16,
                },
                &[33],
            )
            .expect("representable encoded view"),
            OpError::ElementwiseUnsupportedEncoding {
                kind: SemanticOpKind::Copy,
                tensor: ElementwiseTensor::Input0,
                actual: Encoding::Nvfp4 {
                    block_size: 32,
                    scale_dtype: DType::Bf16,
                },
            },
        );
    }

    #[test]
    fn embedding_requires_bf16_weight_i32_ids_and_exact_output() {
        let weight = TensorView::contiguous(DType::Bf16, &[7, 3]).unwrap();
        let ids = TensorView::contiguous(DType::I32, &[2]).unwrap();
        let output = TensorView::contiguous(DType::Bf16, &[2, 3]).unwrap();
        let descriptor = SemanticOpDescriptor::new(
            SemanticOpKind::Embedding,
            vec![weight.clone(), ids.clone()],
            vec![output],
        )
        .unwrap();
        assert_eq!(descriptor.kind(), SemanticOpKind::Embedding);
        assert!(!DType::I32.is_float());
        assert_eq!(DType::I32.size_bytes(), 4);

        assert_eq!(
            SemanticOpDescriptor::new(
                SemanticOpKind::Embedding,
                vec![
                    weight.clone(),
                    TensorView::contiguous(DType::F32, &[2]).unwrap()
                ],
                vec![TensorView::contiguous(DType::Bf16, &[2, 3]).unwrap()],
            ),
            Err(OpError::EmbeddingTensorContractMismatch)
        );
        assert_eq!(
            SemanticOpDescriptor::new(
                SemanticOpKind::Embedding,
                vec![weight.clone(), ids.clone()],
                vec![TensorView::contiguous(DType::Bf16, &[2, 4]).unwrap()],
            ),
            Err(OpError::EmbeddingOutputShapeMismatch)
        );
        assert_eq!(
            SemanticOpDescriptor::new(
                SemanticOpKind::Embedding,
                vec![weight, TensorView::contiguous(DType::I32, &[0]).unwrap()],
                vec![TensorView::contiguous(DType::Bf16, &[0, 3]).unwrap()],
            ),
            Err(OpError::EmbeddingZeroExtent)
        );
    }

    #[test]
    fn matmul_and_silu_mul_fix_bf16_layout_shape_and_accumulation_boundary() {
        let activation = TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap();
        let weight = TensorView::contiguous(DType::Bf16, &[7, 5]).unwrap();
        let output = TensorView::contiguous(DType::Bf16, &[3, 7]).unwrap();
        let descriptor = SemanticOpDescriptor::new(
            SemanticOpKind::Matmul,
            vec![activation.clone(), weight.clone()],
            vec![output],
        )
        .unwrap();
        assert_eq!(descriptor.kind(), SemanticOpKind::Matmul);

        assert_eq!(
            SemanticOpDescriptor::new(
                SemanticOpKind::Matmul,
                vec![activation.clone(), weight],
                vec![TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap()],
            ),
            Err(OpError::MatmulShapeMismatch)
        );
        assert_eq!(
            SemanticOpDescriptor::new(
                SemanticOpKind::Matmul,
                vec![
                    TensorView::contiguous(DType::F16, &[3, 5]).unwrap(),
                    TensorView::contiguous(DType::Bf16, &[7, 5]).unwrap(),
                ],
                vec![TensorView::contiguous(DType::Bf16, &[3, 7]).unwrap()],
            ),
            Err(OpError::MatmulActivationOutputContract)
        );

        let fp8_weight = TensorView::with_encoding(
            DType::F8E4M3Fn,
            Encoding::Fp8Scaled {
                granularity: Fp8ScaleGranularity::OuterDimension,
                scale_dtype: DType::F32,
                resident: Fp8ResidentRepresentation::PackedBytes,
            },
            &[7, 5],
        )
        .unwrap();
        SemanticOpDescriptor::new(
            SemanticOpKind::Matmul,
            vec![activation.clone(), fp8_weight],
            vec![TensorView::contiguous(DType::Bf16, &[3, 7]).unwrap()],
        )
        .expect("valid OCP E4M3FN W8A8 matmul descriptor");

        let unsupported_fp8 = TensorView::with_encoding(
            DType::F8E4M3Fn,
            Encoding::Fp8Scaled {
                granularity: Fp8ScaleGranularity::KBlock { block_size: 128 },
                scale_dtype: DType::F32,
                resident: Fp8ResidentRepresentation::PackedBytes,
            },
            &[7, 5],
        )
        .unwrap();
        assert_eq!(
            SemanticOpDescriptor::new(
                SemanticOpKind::Matmul,
                vec![activation, unsupported_fp8],
                vec![TensorView::contiguous(DType::Bf16, &[3, 7]).unwrap()],
            ),
            Err(OpError::MatmulWeightContract)
        );

        let gate = TensorView::contiguous(DType::Bf16, &[3, 17]).unwrap();
        let silu_mul = SemanticOpDescriptor::new(
            SemanticOpKind::SiluMul,
            vec![gate.clone(), gate.clone()],
            vec![gate],
        )
        .unwrap();
        assert_eq!(silu_mul.kind(), SemanticOpKind::SiluMul);
    }

    #[test]
    fn sigmoid_mul_is_distinct_and_freezes_output_gate_shape_and_o_proj_handoff() {
        for m in [1_usize, 3, 17, 255, 256, 257] {
            let gate = TensorView::contiguous(DType::Bf16, &[m, 16, 256]).unwrap();
            let value = TensorView::contiguous(DType::Bf16, &[m, 16, 256]).unwrap();
            let output = TensorView::new(
                DType::Bf16,
                Encoding::Unquantized,
                &[m, 16, 256],
                &[4096, 256, 1],
                2,
            )
            .unwrap();
            let descriptor = SemanticOpDescriptor::new(
                SemanticOpKind::SigmoidMul,
                vec![gate, value],
                vec![output.clone()],
            )
            .unwrap();
            assert_ne!(descriptor.kind(), SemanticOpKind::SiluMul);
            let handoff = descriptor
                .sigmoid_mul_o_proj_input_view()
                .expect("sigmoid output has an o_proj handoff");
            assert_eq!(handoff.shape(), &[m, 4096]);
            assert_eq!(handoff.strides(), &[4096, 1]);
            assert_eq!(handoff.byte_offset(), output.byte_offset());
            assert_eq!(handoff.payload_bytes(), output.payload_bytes());
            assert!(handoff.is_contiguous());
        }

        let valid = TensorView::contiguous(DType::Bf16, &[3, 16, 256]).unwrap();
        for wrong_shape in [
            vec![3, 4096],
            vec![3, 4, 256],
            vec![3, 16, 255],
            vec![3, 16, 256, 1],
        ] {
            let wrong = TensorView::contiguous(DType::Bf16, &wrong_shape).unwrap();
            assert_eq!(
                SemanticOpDescriptor::new(
                    SemanticOpKind::SigmoidMul,
                    vec![wrong.clone(), wrong.clone()],
                    vec![wrong],
                ),
                Err(OpError::SigmoidMulShapeMismatch)
            );
        }
        assert_eq!(
            SemanticOpDescriptor::new(
                SemanticOpKind::SigmoidMul,
                vec![
                    valid.clone(),
                    TensorView::contiguous(DType::Bf16, &[3, 16, 255]).unwrap(),
                ],
                vec![valid],
            ),
            Err(OpError::ElementwiseMetadataMismatch)
        );
    }

    #[test]
    fn rms_norm_requires_explicit_contract_and_exposes_fixed_baseline() {
        let activation = TensorView::contiguous(DType::Bf16, &[2, 3]).expect("valid activation");
        let scale = TensorView::contiguous(DType::Bf16, &[3]).expect("valid scale");
        let output = TensorView::contiguous(DType::Bf16, &[2, 3]).expect("valid output");

        assert!(matches!(
            SemanticOpDescriptor::new(
                SemanticOpKind::RmsNorm,
                vec![activation.clone(), scale.clone()],
                vec![output.clone()],
            ),
            Err(OpError::RmsNormContractRequired)
        ));

        let operation = SemanticOpDescriptor::new_rms_norm(
            vec![activation, scale],
            vec![output],
            1.0e-6,
            RmsNormScaleMode::OffsetOne,
        )
        .expect("valid RMSNorm descriptor");
        let contract = operation.rms_norm_contract().expect("RMSNorm contract");
        assert_eq!(contract.scale_mode(), RmsNormScaleMode::OffsetOne);
        assert_eq!(contract.epsilon().value().to_bits(), 1.0e-6_f32.to_bits());
        assert_eq!(contract.accumulation_dtype(), DType::F32);
        assert_eq!(contract.output_dtype(), DType::Bf16);
        assert_eq!(contract.alias_policy(), RmsNormAliasPolicy::Unsupported);
        assert_eq!(contract.effective_scale(0.25), 1.25);
        assert_eq!(
            RmsNormContract::new(1.0e-6, RmsNormScaleMode::OffsetOne),
            Ok(contract)
        );
    }

    #[test]
    fn rms_norm_rejects_rank_zero_zero_stride_dtype_encoding_and_shape_errors() {
        let scale = TensorView::contiguous(DType::Bf16, &[3]).expect("valid scale");
        let output = TensorView::contiguous(DType::Bf16, &[2, 3]).expect("valid output");
        let make = |activation: TensorView, scale: TensorView, output: TensorView| {
            SemanticOpDescriptor::new_rms_norm(
                vec![activation, scale],
                vec![output],
                1.0e-6,
                RmsNormScaleMode::OffsetOne,
            )
        };

        let scalar = TensorView::contiguous(DType::Bf16, &[]).expect("valid scalar");
        assert!(matches!(
            make(
                scalar,
                scale.clone(),
                TensorView::contiguous(DType::Bf16, &[]).unwrap()
            ),
            Err(OpError::RmsNormRankZero {
                tensor: RmsNormTensor::Activation
            })
        ));

        let zero = TensorView::contiguous(DType::Bf16, &[2, 0]).expect("zero extent view");
        assert!(matches!(
            make(zero, scale.clone(), output.clone()),
            Err(OpError::RmsNormZeroExtent {
                tensor: RmsNormTensor::Activation
            })
        ));

        let strided = TensorView::new(DType::Bf16, Encoding::Unquantized, &[2, 3], &[4, 1], 0)
            .expect("valid strided view");
        assert!(matches!(
            make(
                strided,
                scale.clone(),
                TensorView::contiguous(DType::Bf16, &[2, 3]).unwrap()
            ),
            Err(OpError::RmsNormNonContiguous {
                tensor: RmsNormTensor::Activation
            })
        ));

        let wrong_dtype = TensorView::contiguous(DType::F32, &[2, 3]).expect("valid tensor");
        assert!(matches!(
            make(wrong_dtype, scale.clone(), output.clone()),
            Err(OpError::RmsNormUnsupportedDType {
                tensor: RmsNormTensor::Activation,
                actual: DType::F32
            })
        ));

        let packed_scale = TensorView::with_encoding(
            DType::U8,
            Encoding::Nvfp4 {
                block_size: 16,
                scale_dtype: DType::F32,
            },
            &[3],
        )
        .expect("valid packed descriptor");
        assert!(matches!(
            make(
                TensorView::contiguous(DType::Bf16, &[2, 3]).unwrap(),
                packed_scale,
                output.clone()
            ),
            Err(OpError::RmsNormUnsupportedEncoding {
                tensor: RmsNormTensor::RawScale,
                actual: Encoding::Nvfp4 {
                    block_size: 16,
                    scale_dtype: DType::F32,
                }
            })
        ));

        let wrong_output = TensorView::contiguous(DType::Bf16, &[2, 4]).unwrap();
        assert!(matches!(
            make(
                TensorView::contiguous(DType::Bf16, &[2, 3]).unwrap(),
                scale.clone(),
                wrong_output
            ),
            Err(OpError::RmsNormOutputShapeMismatch)
        ));

        let rank_two_scale = TensorView::contiguous(DType::Bf16, &[1, 3]).unwrap();
        assert!(matches!(
            make(
                TensorView::contiguous(DType::Bf16, &[2, 3]).unwrap(),
                rank_two_scale,
                output.clone()
            ),
            Err(OpError::RmsNormScaleRankMismatch)
        ));

        let wrong_scale = TensorView::contiguous(DType::Bf16, &[4]).unwrap();
        assert!(matches!(
            make(
                TensorView::contiguous(DType::Bf16, &[2, 3]).unwrap(),
                wrong_scale,
                output
            ),
            Err(OpError::RmsNormScaleShapeMismatch)
        ));
    }

    #[test]
    fn rms_norm_accepts_aligned_nonzero_offset_without_inferring_alias() {
        let offset = TensorView::new(DType::Bf16, Encoding::Unquantized, &[2, 3], &[3, 1], 2)
            .expect("aligned offset");
        let scale = TensorView::contiguous(DType::Bf16, &[3]).expect("valid scale");
        let output = TensorView::contiguous(DType::Bf16, &[2, 3]).expect("valid output");
        assert_eq!(offset.payload_bytes(), 12);
        assert_eq!(offset.end_offset(), 14);
        assert!(offset.is_contiguous());
        assert!(
            SemanticOpDescriptor::new_rms_norm(
                vec![offset, scale],
                vec![output],
                1.0e-6,
                RmsNormScaleMode::OffsetOne,
            )
            .is_ok()
        );

        // Both zero-offset views are valid descriptors even though TensorView
        // cannot identify whether they came from the same backing buffer.
        let valid = SemanticOpDescriptor::new_rms_norm(
            vec![
                TensorView::contiguous(DType::Bf16, &[2, 3]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[3]).unwrap(),
            ],
            vec![TensorView::contiguous(DType::Bf16, &[2, 3]).unwrap()],
            1.0e-6,
            RmsNormScaleMode::OffsetOne,
        )
        .expect("buffer identity is outside TensorView");
        assert_eq!(
            valid.rms_norm_contract().expect("contract").alias_policy(),
            RmsNormAliasPolicy::Unsupported
        );
    }

    #[test]
    fn rms_norm_rejects_invalid_epsilon_and_fake_backend_stays_numerically_unsupported() {
        for epsilon in [0.0_f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                RmsNormContract::new(epsilon, RmsNormScaleMode::OffsetOne),
                Err(OpError::RmsNormInvalidEpsilon { .. })
            ));
        }

        let operation = SemanticOpDescriptor::new_rms_norm(
            vec![
                TensorView::contiguous(DType::Bf16, &[1, 3]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[3]).unwrap(),
            ],
            vec![TensorView::contiguous(DType::Bf16, &[1, 3]).unwrap()],
            1.0e-6,
            RmsNormScaleMode::OffsetOne,
        )
        .expect("valid RMSNorm");
        let backend = FakeBackend::new();
        assert!(matches!(
            backend.execute(&operation),
            Err(BackendError::NumericalExecutionUnsupported)
        ));
        assert!(matches!(
            backend.supports(&operation),
            BackendSupport::Unsupported { .. }
        ));
    }

    fn attention_preprocess_views(m: usize) -> (Vec<TensorView>, Vec<TensorView>) {
        (
            vec![
                TensorView::contiguous(DType::Bf16, &[m, 16, 512]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[m, 4, 256]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[16, 256]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[4, 256]).unwrap(),
                TensorView::contiguous(DType::I32, &[m]).unwrap(),
            ],
            vec![
                TensorView::contiguous(DType::Bf16, &[m, 16, 256]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[m, 16, 256]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[m, 4, 256]).unwrap(),
            ],
        )
    }

    fn attention_preprocess_descriptor(
        m: usize,
        mode: AttentionPreprocessPositionMode,
        start_position: i64,
    ) -> Result<SemanticOpDescriptor, OpError> {
        let (inputs, outputs) = attention_preprocess_views(m);
        let contract = AttentionPreprocessContract::new_qwen3_5(mode, start_position, m as u64)?;
        SemanticOpDescriptor::new_attention_preprocess(inputs, outputs, contract)
    }

    fn attention_preprocess_non_contiguous(dtype: DType, shape: &[usize]) -> TensorView {
        let mut strides = vec![0; shape.len()];
        let mut stride = 1;
        for (dimension, current_stride) in shape.iter().zip(strides.iter_mut()).rev() {
            *current_stride = stride;
            stride *= *dimension;
        }
        strides[0] += 1;
        TensorView::new(dtype, Encoding::Unquantized, shape, &strides, 0).unwrap()
    }

    #[test]
    fn attention_preprocess_freezes_layout_numeric_and_position_contract() {
        for m in [1_usize, 3, 17] {
            for start_position in [0_i64, 1, 3, 255, 256, 257] {
                let mode = if start_position == 0 {
                    AttentionPreprocessPositionMode::Prefill
                } else {
                    AttentionPreprocessPositionMode::DecodeContinuation
                };
                let descriptor = attention_preprocess_descriptor(m, mode, start_position)
                    .expect("valid C3a1 descriptor");
                assert_eq!(descriptor.kind(), SemanticOpKind::AttentionPreprocess);
                assert_eq!(descriptor.arity(), (5, 3));
                let contract = descriptor
                    .attention_preprocess_contract()
                    .expect("C3a1 contract");
                assert_eq!(
                    contract.packing(),
                    AttentionPreprocessPacking::HeadInterleavedQGate
                );
                assert_eq!(contract.position_mode(), mode);
                assert_eq!(contract.start_position(), start_position as u32);
                assert_eq!(contract.token_count(), m as u32);
                assert_eq!(contract.epsilon().bits(), 1.0e-6_f32.to_bits());
                assert_eq!(contract.scale_mode(), RmsNormScaleMode::OffsetOne);
                assert_eq!(contract.accumulation_dtype(), DType::F32);
                assert_eq!(contract.output_dtype(), DType::Bf16);
                assert_eq!(contract.rotary_dim(), 64);
                assert_eq!(contract.rope_theta(), 10_000_000.0);
                assert!(contract.mrope_interleaved());
                assert_eq!(contract.mrope_sections(), [11, 11, 10]);
                assert_eq!(contract.max_position_embeddings(), 262_144);
            }
        }
    }

    #[test]
    fn attention_preprocess_rejects_wrong_tensor_contracts_and_flat_split() {
        let replace = |index: usize, replacement: TensorView| {
            let (mut inputs, mut outputs) = attention_preprocess_views(3);
            if index < inputs.len() {
                inputs[index] = replacement;
            } else {
                outputs[index - inputs.len()] = replacement;
            }
            let contract = AttentionPreprocessContract::new_qwen3_5(
                AttentionPreprocessPositionMode::Prefill,
                0,
                3,
            )
            .unwrap();
            assert!(
                SemanticOpDescriptor::new_attention_preprocess(inputs, outputs, contract).is_err()
            );
        };

        // Every C3a1 tensor has an independent fixed head/width/rank shape.
        for (index, shape) in [
            (0, vec![3, 15, 512]),
            (0, vec![3, 16, 511]),
            (0, vec![3, 16]),
            (1, vec![3, 3, 256]),
            (1, vec![3, 4, 255]),
            (1, vec![3, 4]),
            (2, vec![15, 256]),
            (2, vec![16, 255]),
            (2, vec![16]),
            (3, vec![3, 256]),
            (3, vec![4, 255]),
            (3, vec![4]),
            (4, vec![3, 1]),
            (4, vec![4]),
            (5, vec![3, 15, 256]),
            (5, vec![3, 16, 255]),
            (5, vec![3, 16]),
            (6, vec![3, 15, 256]),
            (6, vec![3, 16, 255]),
            (6, vec![3, 16]),
            (7, vec![3, 3, 256]),
            (7, vec![3, 4, 255]),
            (7, vec![3, 4]),
        ] {
            let dtype = if index == 4 { DType::I32 } else { DType::Bf16 };
            replace(index, TensorView::contiguous(dtype, &shape).unwrap());
        }

        // The forbidden flat-half representation cannot satisfy rank three
        // head-interleaved storage, even though it carries the same element count.
        replace(0, TensorView::contiguous(DType::Bf16, &[3, 8192]).unwrap());

        for index in 0..8 {
            let shape = match index {
                0 => vec![3, 16, 512],
                1 => vec![3, 4, 256],
                2 => vec![16, 256],
                3 => vec![4, 256],
                4 => vec![3],
                5 | 6 => vec![3, 16, 256],
                7 => vec![3, 4, 256],
                _ => unreachable!(),
            };
            replace(index, TensorView::contiguous(DType::F32, &shape).unwrap());
            replace(
                index,
                TensorView::with_encoding(
                    DType::U8,
                    Encoding::Nvfp4 {
                        block_size: 16,
                        scale_dtype: DType::F32,
                    },
                    &shape,
                )
                .unwrap(),
            );
            replace(
                index,
                attention_preprocess_non_contiguous(
                    if index == 4 { DType::I32 } else { DType::Bf16 },
                    &shape,
                ),
            );
        }

        for index in 0..8 {
            let mut shape = match index {
                0 => vec![3, 16, 512],
                1 => vec![3, 4, 256],
                2 => vec![16, 256],
                3 => vec![4, 256],
                4 => vec![3],
                5 | 6 => vec![3, 16, 256],
                7 => vec![3, 4, 256],
                _ => unreachable!(),
            };
            shape[0] = 0;
            replace(
                index,
                TensorView::contiguous(if index == 4 { DType::I32 } else { DType::Bf16 }, &shape)
                    .unwrap(),
            );
        }
    }

    #[test]
    fn attention_preprocess_rejects_missing_or_changed_config_and_position_boundaries() {
        let (inputs, outputs) = attention_preprocess_views(1);
        assert!(matches!(
            SemanticOpDescriptor::new(SemanticOpKind::AttentionPreprocess, inputs, outputs,),
            Err(OpError::AttentionPreprocessContractRequired)
        ));
        assert!(matches!(
            AttentionPreprocessContract::new_qwen3_5(
                AttentionPreprocessPositionMode::Prefill,
                1,
                1,
            ),
            Err(OpError::AttentionPreprocessPositionReset { .. })
        ));
        assert!(matches!(
            AttentionPreprocessContract::new_qwen3_5(
                AttentionPreprocessPositionMode::DecodeContinuation,
                0,
                1,
            ),
            Err(OpError::AttentionPreprocessPositionReset { .. })
        ));
        assert!(matches!(
            AttentionPreprocessContract::new_qwen3_5(
                AttentionPreprocessPositionMode::Prefill,
                -1,
                1,
            ),
            Err(OpError::AttentionPreprocessNegativePosition { .. })
        ));
        assert!(matches!(
            AttentionPreprocessContract::new_qwen3_5(
                AttentionPreprocessPositionMode::Prefill,
                0,
                0,
            ),
            Err(OpError::AttentionPreprocessZeroTokenCount)
        ));
        assert!(matches!(
            AttentionPreprocessContract::new_qwen3_5(
                AttentionPreprocessPositionMode::DecodeContinuation,
                1,
                u64::MAX,
            ),
            Err(OpError::AttentionPreprocessPositionOverflow)
        ));
        assert!(
            AttentionPreprocessContract::new_qwen3_5(
                AttentionPreprocessPositionMode::DecodeContinuation,
                262_143,
                1,
            )
            .is_ok()
        );
        assert!(matches!(
            AttentionPreprocessContract::new_qwen3_5(
                AttentionPreprocessPositionMode::DecodeContinuation,
                262_143,
                2,
            ),
            Err(OpError::AttentionPreprocessPositionOutOfRange { .. })
        ));
        assert!(matches!(
            AttentionPreprocessContract::new_qwen3_5(
                AttentionPreprocessPositionMode::DecodeContinuation,
                262_144,
                1,
            ),
            Err(OpError::AttentionPreprocessPositionOutOfRange { .. })
        ));

        let make = |epsilon,
                    accumulation_dtype,
                    output_dtype,
                    rotary_dim,
                    rope_theta,
                    mrope_interleaved,
                    mrope_sections,
                    max_position_embeddings| {
            AttentionPreprocessContract::new(
                AttentionPreprocessPacking::HeadInterleavedQGate,
                AttentionPreprocessPositionMode::Prefill,
                0,
                1,
                epsilon,
                RmsNormScaleMode::OffsetOne,
                accumulation_dtype,
                output_dtype,
                rotary_dim,
                rope_theta,
                mrope_interleaved,
                mrope_sections,
                max_position_embeddings,
            )
        };
        assert!(
            make(
                2.0e-6,
                DType::F32,
                DType::Bf16,
                64,
                10_000_000.0,
                true,
                [11, 11, 10],
                262_144
            )
            .is_err()
        );
        assert!(
            make(
                1.0e-6,
                DType::F16,
                DType::Bf16,
                64,
                10_000_000.0,
                true,
                [11, 11, 10],
                262_144
            )
            .is_err()
        );
        assert!(
            make(
                1.0e-6,
                DType::F32,
                DType::F32,
                64,
                10_000_000.0,
                true,
                [11, 11, 10],
                262_144
            )
            .is_err()
        );
        assert!(
            make(
                1.0e-6,
                DType::F32,
                DType::Bf16,
                32,
                10_000_000.0,
                true,
                [11, 11, 10],
                262_144
            )
            .is_err()
        );
        assert!(
            make(
                1.0e-6,
                DType::F32,
                DType::Bf16,
                64,
                1_000_000.0,
                true,
                [11, 11, 10],
                262_144
            )
            .is_err()
        );
        assert!(
            make(
                1.0e-6,
                DType::F32,
                DType::Bf16,
                64,
                10_000_000.0,
                false,
                [11, 11, 10],
                262_144
            )
            .is_err()
        );
        assert!(
            make(
                1.0e-6,
                DType::F32,
                DType::Bf16,
                64,
                10_000_000.0,
                true,
                [11, 10, 11],
                262_144
            )
            .is_err()
        );
        assert!(
            make(
                1.0e-6,
                DType::F32,
                DType::Bf16,
                64,
                10_000_000.0,
                true,
                [11, 11, 10],
                262_143
            )
            .is_err()
        );
    }

    #[test]
    fn static_registry_contains_only_the_explicit_phase_one_fake_backend() {
        assert_eq!(backend_registry().len(), 1);
        assert_eq!(backend_registry()[0].name(), "fake");
        assert_eq!(BACKEND_REGISTRY.as_ptr(), backend_registry().as_ptr());
    }
}
