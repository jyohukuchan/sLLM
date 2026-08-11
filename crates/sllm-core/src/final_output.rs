//! Typed Stage C host composition for Qwen3.5 embedding and final output.
//!
//! This is deliberately not a model graph. It only proves that already
//! validated semantic operations are connected through exact owned views and
//! that the checkpoint's embedding weight is the same read-only device range
//! used by the tied output projection.

use std::sync::Arc;

use crate::{
    AccessMode, BoundSemanticOp, DType, Encoding, ExecutionError, OwnedTensorBinding,
    RmsNormScaleMode, SemanticOpKind, TensorDType, WeightClassification, WeightConsumer,
    WeightConsumerKey, WeightLoadEntry,
};

pub const QWEN35_HIDDEN_SIZE: usize = 2_560;
pub const QWEN35_VOCAB_SIZE: usize = 248_320;
pub const QWEN35_EMBEDDING_TENSOR: &str = "model.language_model.embed_tokens.weight";

/// The checked host-side composition around the fixed Qwen3.5 embedding and
/// final RMSNorm -> tied Matmul -> Argmax path.
#[derive(Clone, Debug)]
pub struct QwenFinalOutputBindings {
    embedding: Arc<BoundSemanticOp>,
    final_rmsnorm: Arc<BoundSemanticOp>,
    tied_projection: Arc<BoundSemanticOp>,
    argmax: Arc<BoundSemanticOp>,
}

impl QwenFinalOutputBindings {
    pub fn new(
        tied_weight_entry: &WeightLoadEntry,
        embedding: Arc<BoundSemanticOp>,
        final_rmsnorm: Arc<BoundSemanticOp>,
        tied_projection: Arc<BoundSemanticOp>,
        argmax: Arc<BoundSemanticOp>,
    ) -> Result<Self, ExecutionError> {
        validate_tied_weight_entry(tied_weight_entry)?;
        require_kind(&embedding, SemanticOpKind::Embedding, "input embedding")?;
        require_kind(&final_rmsnorm, SemanticOpKind::RmsNorm, "final RMSNorm")?;
        require_kind(
            &tied_projection,
            SemanticOpKind::Matmul,
            "tied output projection",
        )?;
        require_kind(&argmax, SemanticOpKind::Argmax, "greedy Argmax")?;

        let embedding_weight = &embedding.inputs()[0];
        require_binding(
            embedding_weight,
            DType::Bf16,
            &[QWEN35_VOCAB_SIZE, QWEN35_HIDDEN_SIZE],
            AccessMode::Read,
            "input embedding weight",
        )?;
        require_binding(
            &embedding.inputs()[1],
            DType::I32,
            &[embedding.inputs()[1].view().shape()[0]],
            AccessMode::Read,
            "input embedding token IDs",
        )?;
        let embedding_rows = embedding.inputs()[1].view().shape()[0];
        require_binding(
            &embedding.outputs()[0],
            DType::Bf16,
            &[embedding_rows, QWEN35_HIDDEN_SIZE],
            AccessMode::Write,
            "input embedding output",
        )?;

        let final_rows = final_rmsnorm.inputs()[0].view().shape()[0];
        for (binding, shape, access, role) in [
            (
                &final_rmsnorm.inputs()[0],
                &[final_rows, QWEN35_HIDDEN_SIZE][..],
                AccessMode::Read,
                "final RMSNorm activation",
            ),
            (
                &final_rmsnorm.inputs()[1],
                &[QWEN35_HIDDEN_SIZE][..],
                AccessMode::Read,
                "final RMSNorm raw scale",
            ),
            (
                &final_rmsnorm.outputs()[0],
                &[final_rows, QWEN35_HIDDEN_SIZE][..],
                AccessMode::Write,
                "final RMSNorm output",
            ),
        ] {
            require_binding(binding, DType::Bf16, shape, access, role)?;
        }
        let norm_contract = final_rmsnorm
            .descriptor()
            .rms_norm_contract()
            .ok_or_else(|| invalid("final RMSNorm is missing its numeric contract"))?;
        if norm_contract.epsilon().bits() != 1.0e-6_f32.to_bits()
            || norm_contract.scale_mode() != RmsNormScaleMode::OffsetOne
        {
            return Err(invalid(
                "final RMSNorm must use epsilon 1e-6 and offset-one scale",
            ));
        }

        let projection_weight = &tied_projection.inputs()[1];
        for (binding, shape, access, role) in [
            (
                &tied_projection.inputs()[0],
                &[final_rows, QWEN35_HIDDEN_SIZE][..],
                AccessMode::Read,
                "tied projection activation",
            ),
            (
                projection_weight,
                &[QWEN35_VOCAB_SIZE, QWEN35_HIDDEN_SIZE][..],
                AccessMode::Read,
                "tied projection weight",
            ),
            (
                &tied_projection.outputs()[0],
                &[final_rows, QWEN35_VOCAB_SIZE][..],
                AccessMode::Write,
                "tied projection logits",
            ),
        ] {
            require_binding(binding, DType::Bf16, shape, access, role)?;
        }
        require_binding(
            &argmax.inputs()[0],
            DType::Bf16,
            &[final_rows, QWEN35_VOCAB_SIZE],
            AccessMode::Read,
            "Argmax logits",
        )?;
        require_binding(
            &argmax.outputs()[0],
            DType::I32,
            &[final_rows],
            AccessMode::Write,
            "Argmax token IDs",
        )?;

        require_same_range(
            embedding_weight,
            projection_weight,
            "input embedding and tied output projection weights",
        )?;
        require_same_range(
            &final_rmsnorm.outputs()[0],
            &tied_projection.inputs()[0],
            "final RMSNorm output and tied projection activation",
        )?;
        require_same_range(
            &tied_projection.outputs()[0],
            &argmax.inputs()[0],
            "tied projection logits and Argmax input",
        )?;

        Ok(Self {
            embedding,
            final_rmsnorm,
            tied_projection,
            argmax,
        })
    }

    pub fn embedding(&self) -> &Arc<BoundSemanticOp> {
        &self.embedding
    }

    pub fn final_rmsnorm(&self) -> &Arc<BoundSemanticOp> {
        &self.final_rmsnorm
    }

    pub fn tied_projection(&self) -> &Arc<BoundSemanticOp> {
        &self.tied_projection
    }

    pub fn argmax(&self) -> &Arc<BoundSemanticOp> {
        &self.argmax
    }

    pub fn tied_weight(&self) -> &OwnedTensorBinding {
        &self.embedding.inputs()[0]
    }
}

fn validate_tied_weight_entry(entry: &WeightLoadEntry) -> Result<(), ExecutionError> {
    let expected_consumer = Some(WeightConsumerKey {
        layer: None,
        role: WeightConsumer::EmbeddingAndTiedOutput,
    });
    if entry.tensor_name != QWEN35_EMBEDDING_TENSOR
        || entry.classification != WeightClassification::Required
        || entry.consumer != expected_consumer
        || entry.dtype != TensorDType::Bf16
        || entry.shape != [QWEN35_VOCAB_SIZE as u64, QWEN35_HIDDEN_SIZE as u64]
    {
        return Err(invalid(
            "weight entry is not the required Qwen embedding-and-tied-output BF16 matrix",
        ));
    }
    Ok(())
}

fn require_kind(
    operation: &BoundSemanticOp,
    expected: SemanticOpKind,
    role: &'static str,
) -> Result<(), ExecutionError> {
    if operation.descriptor().kind() != expected {
        return Err(ExecutionError::DescriptorBindingMismatch { role });
    }
    Ok(())
}

fn require_binding(
    binding: &OwnedTensorBinding,
    dtype: DType,
    shape: &[usize],
    access: AccessMode,
    role: &'static str,
) -> Result<(), ExecutionError> {
    let view = binding.view();
    if view.dtype() != dtype
        || view.encoding() != Encoding::Unquantized
        || view.shape() != shape
        || !view.is_contiguous()
    {
        return Err(ExecutionError::DescriptorBindingMismatch { role });
    }
    if binding.access() != access {
        return Err(ExecutionError::AccessViolation {
            role,
            required: access,
            actual: binding.access(),
        });
    }
    Ok(())
}

fn require_same_range(
    left: &OwnedTensorBinding,
    right: &OwnedTensorBinding,
    role: &'static str,
) -> Result<(), ExecutionError> {
    if left.buffer().id() != right.buffer().id() || left.view() != right.view() {
        return Err(ExecutionError::DescriptorBindingMismatch { role });
    }
    Ok(())
}

fn invalid(reason: impl Into<String>) -> ExecutionError {
    ExecutionError::InvalidRequest {
        reason: reason.into(),
    }
}
