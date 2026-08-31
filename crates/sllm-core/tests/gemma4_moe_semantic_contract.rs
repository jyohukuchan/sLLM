use sllm_core::{
    DType, OpError, SemanticOpDescriptor, SemanticOpKind, SparseMoeContract, TensorView,
};

fn contiguous(dtype: DType, shape: &[usize]) -> TensorView {
    TensorView::contiguous(dtype, shape).expect("test tensor view")
}

#[test]
fn gemma_route_accepts_token_one_and_three_canonical_layouts() {
    for (tokens, metadata_bytes) in [(1, 1_160), (3, 1_416)] {
        let descriptor = SemanticOpDescriptor::new(
            SemanticOpKind::MoeRoute,
            vec![contiguous(DType::Bf16, &[tokens, 128])],
            vec![contiguous(DType::U8, &[metadata_bytes])],
        )
        .expect("fixed Gemma route contract");
        assert_eq!(descriptor.kind(), SemanticOpKind::MoeRoute);
    }
}

#[test]
fn gemma_route_rejects_expert_and_metadata_shape_mismatches() {
    assert_eq!(
        SemanticOpDescriptor::new(
            SemanticOpKind::MoeRoute,
            vec![contiguous(DType::Bf16, &[3, 127])],
            vec![contiguous(DType::U8, &[1_416])],
        ),
        Err(OpError::MoeRouteTensorContractMismatch)
    );
    assert_eq!(
        SemanticOpDescriptor::new(
            SemanticOpKind::MoeRoute,
            vec![contiguous(DType::Bf16, &[3, 128])],
            vec![contiguous(DType::U8, &[1_415])],
        ),
        Err(OpError::MoeRouteTensorContractMismatch)
    );
}

#[test]
fn gemma_expert_requires_separate_hidden_route_and_v2_blob() {
    let descriptor = SemanticOpDescriptor::new(
        SemanticOpKind::MoeExpert,
        vec![
            contiguous(DType::Bf16, &[3, 2_816]),
            contiguous(DType::U8, &[1_416]),
            contiguous(DType::U8, &[428_215_552]),
        ],
        vec![contiguous(DType::Bf16, &[3, 2_816])],
    )
    .expect("fixed Gemma expert v2 contract");
    assert_eq!(descriptor.kind(), SemanticOpKind::MoeExpert);

    assert_eq!(
        SemanticOpDescriptor::new(
            SemanticOpKind::MoeExpert,
            vec![
                contiguous(DType::Bf16, &[3, 2_816]),
                contiguous(DType::U8, &[1_416]),
                contiguous(DType::U8, &[434_114_560]),
            ],
            vec![contiguous(DType::Bf16, &[3, 2_816])],
        ),
        Err(OpError::MoeExpertTensorContractMismatch)
    );
}

#[test]
fn qwen_sparse_moe_v1_constructor_and_blob_contract_are_unchanged() {
    let contract =
        SparseMoeContract::new(2_048, 256, 8, 512, 512, true).expect("existing Qwen v1 contract");
    SemanticOpDescriptor::new_sparse_moe(
        vec![
            contiguous(DType::Bf16, &[3, 2_048]),
            contiguous(DType::Bf16, &[256, 2_048]),
            contiguous(DType::U8, &[434_114_560]),
        ],
        vec![contiguous(DType::Bf16, &[3, 2_048])],
        contract,
    )
    .expect("Qwen sparse MoE v1 remains accepted");

    assert_eq!(
        SemanticOpDescriptor::new_sparse_moe(
            vec![
                contiguous(DType::Bf16, &[3, 2_048]),
                contiguous(DType::Bf16, &[256, 2_048]),
                contiguous(DType::U8, &[428_215_552]),
            ],
            vec![contiguous(DType::Bf16, &[3, 2_048])],
            contract,
        ),
        Err(OpError::SparseMoeTensorContractMismatch)
    );
}
