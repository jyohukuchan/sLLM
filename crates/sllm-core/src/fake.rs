use crate::{
    Backend, BackendCapabilities, BackendError, BackendSupport, ExecutionReceipt,
    MaterializedTensor, SemanticOp, TensorView,
};

pub const MAX_FAKE_MATERIALIZATION_BYTES: u64 = 16 * 1024 * 1024;

/// A metadata-only backend for host contract tests. It never stores or computes tensor data.
#[derive(Clone, Copy, Debug, Default)]
pub struct FakeBackend;

impl FakeBackend {
    pub const fn new() -> Self {
        Self
    }
}

impl Backend for FakeBackend {
    fn name(&self) -> &'static str {
        "fake"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            metadata_only: true,
            numerical_execution: false,
            max_materialization_bytes: Some(MAX_FAKE_MATERIALIZATION_BYTES),
        }
    }

    fn supports(&self, _operation: &SemanticOp) -> BackendSupport {
        BackendSupport::Unsupported {
            reason: "FakeBackend does not execute numerical operations",
        }
    }

    fn materialize(&self, view: &TensorView) -> Result<MaterializedTensor, BackendError> {
        if view.span_bytes() > MAX_FAKE_MATERIALIZATION_BYTES {
            return Err(BackendError::MaterializationTooLarge {
                requested_bytes: view.span_bytes(),
                max_bytes: MAX_FAKE_MATERIALIZATION_BYTES,
            });
        }
        Ok(MaterializedTensor::metadata_only(view.clone()))
    }

    fn execute(&self, _operation: &SemanticOp) -> Result<ExecutionReceipt, BackendError> {
        Err(BackendError::NumericalExecutionUnsupported)
    }
}
