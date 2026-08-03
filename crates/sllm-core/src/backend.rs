use std::fmt;

use crate::{OpError, SemanticOp, TensorError, TensorView};

pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> BackendCapabilities;
    fn supports(&self, operation: &SemanticOp) -> BackendSupport;
    fn materialize(&self, view: &TensorView) -> Result<MaterializedTensor, BackendError>;
    fn execute(&self, operation: &SemanticOp) -> Result<ExecutionReceipt, BackendError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendCapabilities {
    pub metadata_only: bool,
    pub numerical_execution: bool,
    pub max_materialization_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendSupport {
    Supported,
    Unsupported { reason: &'static str },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterializedTensor {
    view: TensorView,
    metadata_only: bool,
}

impl MaterializedTensor {
    pub const fn metadata_only(view: TensorView) -> Self {
        Self {
            view,
            metadata_only: true,
        }
    }

    pub const fn view(&self) -> &TensorView {
        &self.view
    }

    pub const fn byte_len(&self) -> u64 {
        self.view.span_bytes()
    }

    pub const fn is_metadata_only(&self) -> bool {
        self.metadata_only
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionReceipt {
    operation: &'static str,
}

impl ExecutionReceipt {
    pub const fn new(operation: &'static str) -> Self {
        Self { operation }
    }

    pub const fn operation(&self) -> &'static str {
        self.operation
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BackendError {
    InvalidTensor(TensorError),
    InvalidOperation(OpError),
    MaterializationTooLarge {
        requested_bytes: u64,
        max_bytes: u64,
    },
    NumericalExecutionUnsupported,
    BackendUnavailable {
        name: &'static str,
    },
    UnsupportedOperation {
        name: &'static str,
    },
}

impl fmt::Display for BackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTensor(error) => write!(formatter, "invalid tensor: {error}"),
            Self::InvalidOperation(error) => write!(formatter, "invalid operation: {error}"),
            Self::MaterializationTooLarge {
                requested_bytes,
                max_bytes,
            } => write!(
                formatter,
                "metadata materialization requested {requested_bytes} bytes, limit is {max_bytes}"
            ),
            Self::NumericalExecutionUnsupported => {
                formatter.write_str("numerical execution is unavailable in Phase 1")
            }
            Self::BackendUnavailable { name } => write!(formatter, "backend {name} is unavailable"),
            Self::UnsupportedOperation { name } => {
                write!(formatter, "operation {name} is unsupported")
            }
        }
    }
}

impl std::error::Error for BackendError {}

impl From<TensorError> for BackendError {
    fn from(error: TensorError) -> Self {
        Self::InvalidTensor(error)
    }
}

impl From<OpError> for BackendError {
    fn from(error: OpError) -> Self {
        Self::InvalidOperation(error)
    }
}
