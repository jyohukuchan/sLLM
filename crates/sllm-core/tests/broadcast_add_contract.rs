use sllm_core::{DType, Encoding, OpError, SemanticOpDescriptor, SemanticOpKind, TensorView};

#[test]
fn broadcast_add_accepts_rank2_activation_and_rank1_vector() {
    let input = TensorView::contiguous(DType::Bf16, &[3, 17]).unwrap();
    let vector = TensorView::contiguous(DType::Bf16, &[17]).unwrap();
    let output = TensorView::contiguous(DType::Bf16, &[3, 17]).unwrap();
    let descriptor = SemanticOpDescriptor::new(
        SemanticOpKind::BroadcastAdd,
        vec![input, vector].into_iter().collect(),
        vec![output],
    )
    .unwrap();
    assert_eq!(descriptor.kind(), SemanticOpKind::BroadcastAdd);
    assert_eq!(descriptor.arity(), (2, 1));
}

#[test]
fn broadcast_add_rejects_wrong_vector_width_dtype_and_layout() {
    let input = TensorView::contiguous(DType::Bf16, &[3, 17]).unwrap();
    let output = TensorView::contiguous(DType::Bf16, &[3, 17]).unwrap();

    let wrong_width = TensorView::contiguous(DType::Bf16, &[16]).unwrap();
    assert_eq!(
        SemanticOpDescriptor::new(
            SemanticOpKind::BroadcastAdd,
            vec![input.clone(), wrong_width],
            vec![output.clone()],
        ),
        Err(OpError::BroadcastAddShapeMismatch)
    );

    let wrong_dtype = TensorView::contiguous(DType::F32, &[17]).unwrap();
    assert!(matches!(
        SemanticOpDescriptor::new(
            SemanticOpKind::BroadcastAdd,
            vec![input.clone(), wrong_dtype],
            vec![output.clone()],
        ),
        Err(OpError::BroadcastAddUnsupportedDType { .. })
    ));

    let non_contiguous =
        TensorView::new(DType::Bf16, Encoding::Unquantized, &[3, 17], &[18, 1], 0).unwrap();
    assert!(matches!(
        SemanticOpDescriptor::new(
            SemanticOpKind::BroadcastAdd,
            vec![
                non_contiguous,
                TensorView::contiguous(DType::Bf16, &[17]).unwrap()
            ],
            vec![output],
        ),
        Err(OpError::BroadcastAddNonContiguous { .. })
    ));
}
