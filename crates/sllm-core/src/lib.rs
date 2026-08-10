//! Backend-independent runtime contracts for sLLM.
//!
//! Phase 1 deliberately contains descriptors and control-plane behavior only.
//! It does not allocate model data, emulate a GPU, or execute numerical work.

mod backend;
mod dtype;
mod execution;
mod fake;
mod handles;
mod model;
mod op;
mod registry;
mod tensor;
mod weights;

pub use backend::{
    Backend, BackendCapabilities, BackendError, BackendSupport, ExecutionReceipt,
    MaterializedTensor,
};
pub use dtype::{DType, Encoding, EncodingError};
pub use execution::{
    AdapterResource, BoundSemanticOp, BufferRange, BufferReadback, DispatchEvidence,
    ExecutionAdapterAccess, ExecutionBuffer, ExecutionBufferId, ExecutionError, ExecutionQueue,
    ExecutionQueueId, ExecutionReadbackAdapter, ExecutionSession, ExecutionSessionAdapter,
    ExecutionSessionId, ExecutionSessionRequest, ExecutionState, ExecutionSubmissionAdapter,
    ExecutionTransferAdapter, OwnedTensorBinding, PrepareSupport, PreparedOperation,
    PreparedOperationId, Readback, ShutdownReport, Submission, Transfer,
};
pub use fake::{FakeBackend, MAX_FAKE_MATERIALIZATION_BYTES};
pub use handles::{
    AccessMode, BufferHandle, BufferUse, CompletionLease, EventHandle, InFlightSubmission,
    QueueHandle,
};
pub use model::{
    AccumulationDType, BaseModel, BudgetBoundary, ClassificationStatus, ComponentMetadata,
    ComponentStatus, ConfigEos, ExcludedFile, FrontendAssetKind, GenerationConfig,
    GenerationStopPolicyV1, LayerSchedule, LayerType, LicenseInfo, LockedFile, LockedModel,
    MaxNewTokensZero, ModelArchitecture, ModelError, ModelLock, NormalizationContract,
    NormalizationKind, PromptEvaluation, ScaleMode, SliceContract, StopEvaluation, StopIdentity,
    StopTokenHandling, TensorClassification, TensorContract, TensorDType, TensorDescriptor,
    TextConfig, TokenizerContract, TokenizerEos, VerifiedCache, VerifiedFile, fingerprint_for_json,
    parse_model_lock, read_model_lock, validate_model_config, verify_model_cache,
};
pub use op::{
    OpError, RmsNormAliasPolicy, RmsNormContract, RmsNormEpsilon, RmsNormScaleMode, RmsNormTensor,
    SemanticOp, SemanticOpDescriptor, SemanticOpKind,
};
pub use registry::{BACKEND_REGISTRY, BackendRegistration, backend_registry};
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
        let input = TensorView::contiguous(DType::F32, &[3, 5]).expect("valid input");
        let output = TensorView::contiguous(DType::F32, &[3, 5]).expect("valid output");
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
        let valid_copy = SemanticOpDescriptor::new(
            SemanticOpKind::Copy,
            vec![TensorView::contiguous(DType::F32, &[2, 2]).expect("valid tensor")],
            vec![TensorView::contiguous(DType::F32, &[2, 2]).expect("valid tensor")],
        )
        .expect("valid copy descriptor");
        assert_eq!(valid_copy.arity(), (1, 1));
        assert_eq!(SemanticOpKind::Copy.arity(), (1, 1));
        assert_eq!(SemanticOpKind::Add.arity(), (2, 1));
        assert_eq!(SemanticOpKind::Matmul.arity(), (2, 1));
        assert_eq!(SemanticOpKind::RmsNorm.arity(), (2, 1));
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

    #[test]
    fn static_registry_contains_only_the_explicit_phase_one_fake_backend() {
        assert_eq!(backend_registry().len(), 1);
        assert_eq!(backend_registry()[0].name(), "fake");
        assert_eq!(BACKEND_REGISTRY.as_ptr(), backend_registry().as_ptr());
    }
}
