use std::fmt;

use crate::TensorView;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SemanticOpKind {
    Copy,
    Add,
    Matmul,
}

impl SemanticOpKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Copy => "copy",
            Self::Add => "add",
            Self::Matmul => "matmul",
        }
    }

    pub const fn arity(self) -> (usize, usize) {
        match self {
            Self::Copy => (1, 1),
            Self::Add => (2, 1),
            Self::Matmul => (2, 1),
        }
    }
}

/// A backend-independent operation contract containing only tensor metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticOpDescriptor {
    kind: SemanticOpKind,
    inputs: Vec<TensorView>,
    outputs: Vec<TensorView>,
}

/// Short name for the semantic operation descriptor used by backend traits.
pub type SemanticOp = SemanticOpDescriptor;

impl SemanticOpDescriptor {
    pub fn new(
        kind: SemanticOpKind,
        inputs: Vec<TensorView>,
        outputs: Vec<TensorView>,
    ) -> Result<Self, OpError> {
        let descriptor = Self {
            kind,
            inputs,
            outputs,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub const fn kind(&self) -> SemanticOpKind {
        self.kind
    }

    pub const fn arity(&self) -> (usize, usize) {
        self.kind.arity()
    }

    pub fn inputs(&self) -> &[TensorView] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[TensorView] {
        &self.outputs
    }

    pub fn validate(&self) -> Result<(), OpError> {
        let (expected_inputs, expected_outputs) = self.arity();
        if self.inputs.len() != expected_inputs || self.outputs.len() != expected_outputs {
            return Err(OpError::Arity {
                kind: self.kind,
                expected_inputs,
                actual_inputs: self.inputs.len(),
                expected_outputs,
                actual_outputs: self.outputs.len(),
            });
        }

        match self.kind {
            SemanticOpKind::Copy => {
                if !same_metadata(&self.inputs[0], &self.outputs[0]) {
                    return Err(OpError::CopyMetadataMismatch);
                }
            }
            SemanticOpKind::Add => {
                if !same_metadata(&self.inputs[0], &self.inputs[1])
                    || !same_metadata(&self.inputs[0], &self.outputs[0])
                {
                    return Err(OpError::ElementwiseMetadataMismatch);
                }
            }
            SemanticOpKind::Matmul => {
                let left = self.inputs[0].shape();
                let right = self.inputs[1].shape();
                let output = self.outputs[0].shape();
                if left.len() != 2
                    || right.len() != 2
                    || output.len() != 2
                    || left[1] != right[0]
                    || output != [left[0], right[1]]
                    || self.inputs[0].dtype() != self.inputs[1].dtype()
                    || self.inputs[0].dtype() != self.outputs[0].dtype()
                    || self.inputs[0].encoding() != self.inputs[1].encoding()
                    || self.inputs[0].encoding() != self.outputs[0].encoding()
                {
                    return Err(OpError::MatmulShapeMismatch);
                }
            }
        }
        Ok(())
    }
}

fn same_metadata(left: &TensorView, right: &TensorView) -> bool {
    left.dtype() == right.dtype()
        && left.encoding() == right.encoding()
        && left.shape() == right.shape()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpError {
    Arity {
        kind: SemanticOpKind,
        expected_inputs: usize,
        actual_inputs: usize,
        expected_outputs: usize,
        actual_outputs: usize,
    },
    CopyMetadataMismatch,
    ElementwiseMetadataMismatch,
    MatmulShapeMismatch,
}

impl fmt::Display for OpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Arity {
                kind,
                expected_inputs,
                actual_inputs,
                expected_outputs,
                actual_outputs,
            } => write!(
                formatter,
                "{} expects {expected_inputs} input(s) and {expected_outputs} output(s), got {actual_inputs} and {actual_outputs}",
                kind.name()
            ),
            Self::CopyMetadataMismatch => {
                formatter.write_str("copy input and output metadata differ")
            }
            Self::ElementwiseMetadataMismatch => {
                formatter.write_str("elementwise operand metadata differ")
            }
            Self::MatmulShapeMismatch => {
                formatter.write_str("matmul shapes or dtypes are incompatible")
            }
        }
    }
}

impl std::error::Error for OpError {}
