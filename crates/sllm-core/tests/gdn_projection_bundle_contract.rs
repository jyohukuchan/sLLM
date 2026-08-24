use sllm_core::{
    DType, Encoding, GdnProjectionBundleContractV1, OpError, SemanticOpDescriptor, TensorView,
};

const M: usize = 1;
const K: usize = 2560;
const WIDTHS: [usize; 4] = [8192, 4096, 32, 32];

fn valid_parts() -> (TensorView, Vec<TensorView>, Vec<TensorView>) {
    let activation = TensorView::contiguous(DType::Bf16, &[M, K]).unwrap();
    let weights = WIDTHS
        .into_iter()
        .map(|width| TensorView::contiguous(DType::Bf16, &[width, K]).unwrap())
        .collect();
    let outputs = WIDTHS
        .into_iter()
        .map(|width| TensorView::contiguous(DType::Bf16, &[M, width]).unwrap())
        .collect();
    (activation, weights, outputs)
}

fn descriptor(
    activation: TensorView,
    weights: Vec<TensorView>,
    outputs: Vec<TensorView>,
) -> Result<SemanticOpDescriptor, OpError> {
    let mut inputs = Vec::with_capacity(5);
    inputs.push(activation);
    inputs.extend(weights);
    SemanticOpDescriptor::new_gdn_projection_bundle(
        inputs,
        outputs,
        GdnProjectionBundleContractV1::qwen35(),
    )
}

#[test]
fn gdn_projection_bundle_accepts_fixed_qwen35_roles() {
    let (activation, weights, outputs) = valid_parts();
    let descriptor = descriptor(activation, weights, outputs).unwrap();
    assert_eq!(descriptor.inputs().len(), 5);
    assert_eq!(descriptor.outputs().len(), 4);
}

#[test]
fn gdn_projection_bundle_rejects_non_decode_batch_and_role_widths() {
    let (activation, weights, outputs) = valid_parts();
    let batched = TensorView::contiguous(DType::Bf16, &[2, K]).unwrap();
    assert_eq!(
        descriptor(batched, weights.clone(), outputs.clone()),
        Err(OpError::GdnProjectionBundleShapeMismatch)
    );

    // The four weight/output positions are fixed roles; exchanging qkv and z
    // cannot silently select a generic multi-column implementation.
    let mut swapped_weights = weights;
    swapped_weights.swap(0, 1);
    assert_eq!(
        descriptor(activation, swapped_weights, outputs),
        Err(OpError::GdnProjectionBundleShapeMismatch)
    );
}

#[test]
fn gdn_projection_bundle_rejects_dtype_layout_and_zero_extent() {
    let (activation, weights, outputs) = valid_parts();
    let mut wrong_dtype = weights.clone();
    wrong_dtype[0] = TensorView::contiguous(DType::F32, &[WIDTHS[0], K]).unwrap();
    assert!(matches!(
        descriptor(activation.clone(), wrong_dtype, outputs.clone()),
        Err(OpError::GdnProjectionBundleUnsupportedDType { actual: DType::F32 })
    ));

    let non_contiguous = TensorView::new(
        DType::Bf16,
        Encoding::Unquantized,
        &[WIDTHS[0], K],
        &[K + 1, 1],
        0,
    )
    .unwrap();
    let mut wrong_layout = weights.clone();
    wrong_layout[0] = non_contiguous;
    assert_eq!(
        descriptor(activation.clone(), wrong_layout, outputs.clone()),
        Err(OpError::GdnProjectionBundleNonContiguous)
    );

    let zero = TensorView::contiguous(DType::Bf16, &[0, K]).unwrap();
    assert_eq!(
        descriptor(zero, weights, outputs),
        Err(OpError::GdnProjectionBundleZeroExtent)
    );
}
