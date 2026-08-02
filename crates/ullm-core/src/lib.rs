//! Backend-independent runtime contracts for uLLM.
//!
//! Phase 1 deliberately contains descriptors and control-plane behavior only.
//! It does not allocate model data, emulate a GPU, or execute numerical work.

mod backend;
mod dtype;
mod fake;
mod handles;
mod op;
mod registry;
mod tensor;

pub use backend::{
    Backend, BackendCapabilities, BackendError, BackendSupport, ExecutionReceipt,
    MaterializedTensor,
};
pub use dtype::{DType, Encoding, EncodingError};
pub use fake::{FakeBackend, MAX_FAKE_MATERIALIZATION_BYTES};
pub use handles::{
    AccessMode, BufferHandle, BufferUse, CompletionLease, EventHandle, InFlightSubmission,
    QueueHandle,
};
pub use op::{OpError, SemanticOp, SemanticOpDescriptor, SemanticOpKind};
pub use registry::{BACKEND_REGISTRY, BackendRegistration, backend_registry};
pub use tensor::{TensorError, TensorView};

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
    }

    #[test]
    fn static_registry_contains_only_the_explicit_phase_one_fake_backend() {
        assert_eq!(backend_registry().len(), 1);
        assert_eq!(backend_registry()[0].name(), "fake");
        assert_eq!(BACKEND_REGISTRY.as_ptr(), backend_registry().as_ptr());
    }
}
