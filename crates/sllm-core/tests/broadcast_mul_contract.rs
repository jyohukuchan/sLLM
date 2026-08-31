use sllm_core::{DType, Encoding, OpError, SemanticOpDescriptor, SemanticOpKind, TensorView};

fn contiguous(dtype: DType, shape: &[usize]) -> TensorView {
    TensorView::contiguous(dtype, shape).expect("test tensor view")
}

#[test]
fn broadcast_mul_accepts_bf16_m_by_h_and_h_for_token_boundaries() {
    for rows in [1, 3] {
        let descriptor = SemanticOpDescriptor::new(
            SemanticOpKind::BroadcastMul,
            vec![
                contiguous(DType::Bf16, &[rows, 2_816]),
                contiguous(DType::Bf16, &[2_816]),
            ],
            vec![contiguous(DType::Bf16, &[rows, 2_816])],
        )
        .expect("broadcast multiplication contract");
        assert_eq!(descriptor.kind(), SemanticOpKind::BroadcastMul);
    }
}

#[test]
fn broadcast_mul_rejects_shape_dtype_encoding_and_stride_mismatches() {
    let matrix = contiguous(DType::Bf16, &[3, 2_816]);
    let output = matrix.clone();
    assert_eq!(
        SemanticOpDescriptor::new(
            SemanticOpKind::BroadcastMul,
            vec![matrix.clone(), contiguous(DType::Bf16, &[2_815])],
            vec![output.clone()],
        ),
        Err(OpError::BroadcastMulShapeMismatch)
    );
    assert!(matches!(
        SemanticOpDescriptor::new(
            SemanticOpKind::BroadcastMul,
            vec![matrix.clone(), contiguous(DType::F32, &[2_816])],
            vec![output.clone()],
        ),
        Err(OpError::BroadcastMulUnsupportedDType { .. })
    ));
    let encoded = TensorView::new(
        DType::U8,
        Encoding::Nvfp4 {
            block_size: 16,
            scale_dtype: DType::F8E4M3Fn,
        },
        &[2_816],
        &[1],
        0,
    )
    .expect("encoded view");
    assert!(matches!(
        SemanticOpDescriptor::new(
            SemanticOpKind::BroadcastMul,
            vec![matrix.clone(), encoded],
            vec![output.clone()],
        ),
        Err(OpError::BroadcastMulUnsupportedEncoding { .. })
    ));
    let strided = TensorView::new(
        DType::Bf16,
        Encoding::Unquantized,
        &[3, 2_816],
        &[2_817, 1],
        0,
    )
    .expect("non-contiguous view");
    assert!(matches!(
        SemanticOpDescriptor::new(
            SemanticOpKind::BroadcastMul,
            vec![strided, contiguous(DType::Bf16, &[2_816])],
            vec![output],
        ),
        Err(OpError::BroadcastMulNonContiguous { .. })
    ));
}
