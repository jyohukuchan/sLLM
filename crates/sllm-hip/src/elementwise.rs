//! Safe BF16 elementwise wrappers shared by reviewed model adapters.

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_core::{SemanticOpDescriptor, SemanticOpKind};
use sllm_hip_sys as sys;

use crate::runtime::{
    Completion, CompletionState, Context, Queue, RuntimeError, RuntimeStatus,
    enqueue_elementwise_cleanup, ensure_ok, release_elementwise_plan_once, sink,
};
use crate::{HipBackend, TensorBinding};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ElementwiseOperation {
    Copy,
    Add,
    BroadcastAdd,
    BroadcastMul,
    ScalarMul,
    SiluMul,
    GeluTanhMul,
    SigmoidMul,
    TanhSoftcap,
}

impl ElementwiseOperation {
    const fn raw(self) -> sys::sllm_elementwise_operation_t {
        match self {
            Self::Copy => sys::SLLM_ELEMENTWISE_OPERATION_COPY,
            Self::Add => sys::SLLM_ELEMENTWISE_OPERATION_ADD,
            Self::BroadcastAdd => sys::SLLM_ELEMENTWISE_OPERATION_BROADCAST_ADD,
            Self::BroadcastMul => sys::SLLM_ELEMENTWISE_OPERATION_BROADCAST_MUL,
            Self::ScalarMul => sys::SLLM_ELEMENTWISE_OPERATION_SCALAR_MUL,
            Self::SiluMul => sys::SLLM_ELEMENTWISE_OPERATION_SILU_MUL,
            Self::GeluTanhMul => sys::SLLM_ELEMENTWISE_OPERATION_GELU_TANH_MUL,
            Self::SigmoidMul => sys::SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL,
            Self::TanhSoftcap => sys::SLLM_ELEMENTWISE_OPERATION_TANH_SOFTCAP,
        }
    }

    fn from_semantic(kind: SemanticOpKind) -> Result<Self, RuntimeError> {
        match kind {
            SemanticOpKind::Copy => Ok(Self::Copy),
            SemanticOpKind::Add => Ok(Self::Add),
            SemanticOpKind::BroadcastAdd => Ok(Self::BroadcastAdd),
            SemanticOpKind::BroadcastMul => Ok(Self::BroadcastMul),
            SemanticOpKind::ScalarMul => Ok(Self::ScalarMul),
            SemanticOpKind::SiluMul => Ok(Self::SiluMul),
            SemanticOpKind::GeluTanhMul => Ok(Self::GeluTanhMul),
            SemanticOpKind::SigmoidMul => Ok(Self::SigmoidMul),
            SemanticOpKind::TanhSoftcap => Ok(Self::TanhSoftcap),
            _ => Err(RuntimeError::local(
                RuntimeStatus::InvalidElementwiseDescriptor,
                "semantic descriptor is not a supported elementwise operation",
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ElementwiseDescriptor {
    operation: ElementwiseOperation,
    input0: TensorBinding,
    input1: Option<TensorBinding>,
    output: TensorBinding,
    semantic: Arc<SemanticOpDescriptor>,
}

impl ElementwiseDescriptor {
    pub fn copy(input: TensorBinding, output: TensorBinding) -> Result<Self, sllm_core::OpError> {
        let semantic = Arc::new(SemanticOpDescriptor::new(
            SemanticOpKind::Copy,
            vec![input.view().clone()],
            vec![output.view().clone()],
        )?);
        Ok(Self {
            operation: ElementwiseOperation::Copy,
            input0: input,
            input1: None,
            output,
            semantic,
        })
    }

    pub fn add(
        input0: TensorBinding,
        input1: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, sllm_core::OpError> {
        let semantic = Arc::new(SemanticOpDescriptor::new(
            SemanticOpKind::Add,
            vec![input0.view().clone(), input1.view().clone()],
            vec![output.view().clone()],
        )?);
        Ok(Self {
            operation: ElementwiseOperation::Add,
            input0,
            input1: Some(input1),
            output,
            semantic,
        })
    }

    pub fn broadcast_add(
        input: TensorBinding,
        vector: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, sllm_core::OpError> {
        let semantic = Arc::new(SemanticOpDescriptor::new(
            SemanticOpKind::BroadcastAdd,
            vec![input.view().clone(), vector.view().clone()],
            vec![output.view().clone()],
        )?);
        Ok(Self {
            operation: ElementwiseOperation::BroadcastAdd,
            input0: input,
            input1: Some(vector),
            output,
            semantic,
        })
    }

    pub fn broadcast_mul(
        input: TensorBinding,
        vector: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, sllm_core::OpError> {
        let semantic = Arc::new(SemanticOpDescriptor::new(
            SemanticOpKind::BroadcastMul,
            vec![input.view().clone(), vector.view().clone()],
            vec![output.view().clone()],
        )?);
        Ok(Self {
            operation: ElementwiseOperation::BroadcastMul,
            input0: input,
            input1: Some(vector),
            output,
            semantic,
        })
    }

    pub fn silu_mul(
        gate: TensorBinding,
        up: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, sllm_core::OpError> {
        let semantic = Arc::new(SemanticOpDescriptor::new(
            SemanticOpKind::SiluMul,
            vec![gate.view().clone(), up.view().clone()],
            vec![output.view().clone()],
        )?);
        Ok(Self {
            operation: ElementwiseOperation::SiluMul,
            input0: gate,
            input1: Some(up),
            output,
            semantic,
        })
    }

    pub fn scalar_mul(
        input: TensorBinding,
        scalar: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, sllm_core::OpError> {
        let semantic = Arc::new(SemanticOpDescriptor::new(
            SemanticOpKind::ScalarMul,
            vec![input.view().clone(), scalar.view().clone()],
            vec![output.view().clone()],
        )?);
        Ok(Self {
            operation: ElementwiseOperation::ScalarMul,
            input0: input,
            input1: Some(scalar),
            output,
            semantic,
        })
    }

    pub fn gelu_tanh_mul(
        gate: TensorBinding,
        up: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, sllm_core::OpError> {
        let semantic = Arc::new(SemanticOpDescriptor::new(
            SemanticOpKind::GeluTanhMul,
            vec![gate.view().clone(), up.view().clone()],
            vec![output.view().clone()],
        )?);
        Ok(Self {
            operation: ElementwiseOperation::GeluTanhMul,
            input0: gate,
            input1: Some(up),
            output,
            semantic,
        })
    }

    pub fn tanh_softcap(
        input: TensorBinding,
        cap: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, sllm_core::OpError> {
        let semantic = Arc::new(SemanticOpDescriptor::new(
            SemanticOpKind::TanhSoftcap,
            vec![input.view().clone(), cap.view().clone()],
            vec![output.view().clone()],
        )?);
        Ok(Self {
            operation: ElementwiseOperation::TanhSoftcap,
            input0: input,
            input1: Some(cap),
            output,
            semantic,
        })
    }

    pub fn sigmoid_mul(
        gate: TensorBinding,
        attention_value: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, sllm_core::OpError> {
        let semantic = Arc::new(SemanticOpDescriptor::new(
            SemanticOpKind::SigmoidMul,
            vec![gate.view().clone(), attention_value.view().clone()],
            vec![output.view().clone()],
        )?);
        Ok(Self {
            operation: ElementwiseOperation::SigmoidMul,
            input0: gate,
            input1: Some(attention_value),
            output,
            semantic,
        })
    }

    /// Rebinds the validated sigmoid output storage as contiguous
    /// `[M,4096]`, the activation layout accepted by the existing `o_proj`
    /// matmul descriptor. This is a zero-copy metadata handoff; sequencing the
    /// later matmul after completion remains the caller's responsibility.
    pub fn sigmoid_mul_o_proj_input(&self) -> Option<TensorBinding> {
        let view = self.semantic.sigmoid_mul_o_proj_input_view()?;
        Some(self.output.buffer().binding(view))
    }

    pub fn semantic(&self) -> &SemanticOpDescriptor {
        self.semantic.as_ref()
    }

    pub(crate) fn from_validated_semantic(
        semantic: Arc<SemanticOpDescriptor>,
        inputs: Vec<TensorBinding>,
        output: TensorBinding,
    ) -> Result<Self, RuntimeError> {
        semantic.validate().map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidElementwiseDescriptor,
                format!("invalid validated semantic elementwise descriptor: {error}"),
            )
        })?;
        let operation = ElementwiseOperation::from_semantic(semantic.kind())?;
        let expected_inputs = match operation {
            ElementwiseOperation::Copy => 1,
            ElementwiseOperation::Add => 2,
            ElementwiseOperation::BroadcastAdd => 2,
            ElementwiseOperation::BroadcastMul => 2,
            ElementwiseOperation::ScalarMul => 2,
            ElementwiseOperation::SiluMul => 2,
            ElementwiseOperation::GeluTanhMul => 2,
            ElementwiseOperation::SigmoidMul => 2,
            ElementwiseOperation::TanhSoftcap => 2,
        };
        if inputs.len() != expected_inputs || semantic.outputs().len() != 1 {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidElementwiseDescriptor,
                "semantic elementwise descriptor has noncanonical arity",
            ));
        }
        if inputs
            .iter()
            .zip(semantic.inputs())
            .any(|(binding, view)| binding.view() != view)
            || output.view() != &semantic.outputs()[0]
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidElementwiseDescriptor,
                "bound HIP tensor views differ from the core elementwise descriptor",
            ));
        }
        let mut inputs = inputs.into_iter();
        let input0 = inputs.next().expect("validated elementwise arity");
        let input1 = inputs.next();
        Ok(Self {
            operation,
            input0,
            input1,
            output,
            semantic,
        })
    }

    fn raw(&self) -> Result<sys::sllm_elementwise_desc_t, RuntimeError> {
        let input1 = match &self.input1 {
            Some(binding) => binding.raw()?,
            None => zero_binding(),
        };
        Ok(sys::sllm_elementwise_desc_t {
            struct_size: size_of::<sys::sllm_elementwise_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_ELEMENTWISE_VERSION,
            operation: self.operation.raw(),
            reserved: [0; 4],
            input0: self.input0.raw()?,
            input1,
            output: self.output.raw()?,
        })
    }
}

fn zero_binding() -> sys::sllm_tensor_binding_t {
    sys::sllm_tensor_binding_t {
        struct_size: 0,
        abi_version: 0,
        buffer: std::ptr::null(),
        byte_offset: 0,
        dtype: 0,
        encoding: 0,
        rank: 0,
        reserved0: 0,
        shape: [0; 8],
        stride_elements: [0; 8],
        reserved: [0; 2],
    }
}

struct PreparedElementwiseOwners {
    context: Context,
    descriptor: ElementwiseDescriptor,
}

pub(crate) struct PreparedElementwiseState {
    raw: NonNull<sys::sllm_elementwise_plan_t>,
    owners: PreparedElementwiseOwners,
}

// SAFETY: the native plan is an opaque registry token and its owner graph is
// immutable. Native state transitions are serialized by the registry and
// context accounting lock.
unsafe impl Send for PreparedElementwiseState {}
// SAFETY: shared access exposes no unsynchronized native mutation.
unsafe impl Sync for PreparedElementwiseState {}

impl Drop for PreparedElementwiseState {
    fn drop(&mut self) {
        let (status, remaining) = release_elementwise_plan_once(self.raw);
        if let Some(remaining) = remaining {
            enqueue_elementwise_cleanup(
                remaining,
                self.owners.context.clone(),
                self.owners.descriptor.clone(),
                status,
            );
        }
    }
}

#[derive(Clone)]
pub struct PreparedElementwise {
    state: Arc<PreparedElementwiseState>,
}

// SAFETY: the state has the Send/Sync guarantees documented above.
unsafe impl Send for PreparedElementwise {}
// SAFETY: the state has the Send/Sync guarantees documented above.
unsafe impl Sync for PreparedElementwise {}

impl std::fmt::Debug for PreparedElementwise {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedElementwise")
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ElementwiseDispatchInfo {
    pub abi_version: u32,
    pub info_version: u32,
    pub dispatch_id: u64,
    pub operation: ElementwiseOperation,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
    pub element_count: u64,
    pub backend: u32,
    pub fallback_allowed: bool,
    pub fallback_used: bool,
    pub kernel_symbol: String,
    pub device_symbol: String,
    pub gcn_arch_name: String,
}

fn read_c_string(value: &[core::ffi::c_char]) -> String {
    let length = value
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(value.len());
    value[..length]
        .iter()
        .map(|byte| *byte as u8)
        .map(char::from)
        .collect()
}

fn dispatch_info_from_raw(
    info: &sys::sllm_elementwise_dispatch_info_t,
) -> Result<ElementwiseDispatchInfo, RuntimeError> {
    let operation = match info.operation {
        sys::SLLM_ELEMENTWISE_OPERATION_COPY => ElementwiseOperation::Copy,
        sys::SLLM_ELEMENTWISE_OPERATION_ADD => ElementwiseOperation::Add,
        sys::SLLM_ELEMENTWISE_OPERATION_BROADCAST_ADD => ElementwiseOperation::BroadcastAdd,
        sys::SLLM_ELEMENTWISE_OPERATION_BROADCAST_MUL => ElementwiseOperation::BroadcastMul,
        sys::SLLM_ELEMENTWISE_OPERATION_SCALAR_MUL => ElementwiseOperation::ScalarMul,
        sys::SLLM_ELEMENTWISE_OPERATION_SILU_MUL => ElementwiseOperation::SiluMul,
        sys::SLLM_ELEMENTWISE_OPERATION_GELU_TANH_MUL => ElementwiseOperation::GeluTanhMul,
        sys::SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL => ElementwiseOperation::SigmoidMul,
        sys::SLLM_ELEMENTWISE_OPERATION_TANH_SOFTCAP => ElementwiseOperation::TanhSoftcap,
        _ => {
            return Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "native elementwise dispatch returned an unknown operation",
            ));
        }
    };
    Ok(ElementwiseDispatchInfo {
        abi_version: info.abi_version,
        info_version: info.info_version,
        dispatch_id: info.dispatch_id,
        operation,
        dispatch_count: info.dispatch_count,
        kernel_id: info.kernel_id,
        workgroup_size_x: info.workgroup_size_x,
        grid_size_x: info.grid_size_x,
        element_count: info.element_count,
        backend: info.backend,
        fallback_allowed: info.fallback_allowed != 0,
        fallback_used: info.fallback_used != 0,
        kernel_symbol: read_c_string(&info.kernel_symbol),
        device_symbol: read_c_string(&info.device_symbol),
        gcn_arch_name: read_c_string(&info.gcn_arch_name),
    })
}

pub struct ElementwiseSubmission {
    completion: Completion,
    _plan: Arc<PreparedElementwiseState>,
}

impl std::fmt::Debug for ElementwiseSubmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ElementwiseSubmission")
            .finish_non_exhaustive()
    }
}

impl ElementwiseSubmission {
    pub fn query(&mut self) -> Result<CompletionState, RuntimeError> {
        self.completion.query()
    }

    pub fn wait(&mut self, timeout: std::time::Duration) -> Result<CompletionState, RuntimeError> {
        self.completion.wait(timeout)
    }

    pub fn kernel_elapsed_ns(&mut self) -> Result<u64, RuntimeError> {
        self.completion.kernel_elapsed_ns()
    }

    pub(crate) fn finalize_after_token(
        &mut self,
        fence_token: u64,
    ) -> Result<CompletionState, RuntimeError> {
        self.completion.finalize_after_token(fence_token)
    }
}

impl HipBackend {
    pub fn prepare_elementwise(
        &self,
        context: &Context,
        descriptor: ElementwiseDescriptor,
    ) -> Result<PreparedElementwise, RuntimeError> {
        let raw_descriptor = descriptor.raw()?;
        let mut raw_plan = std::ptr::null_mut();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_elementwise_prepare(
                context.raw_handle()?.as_ptr(),
                &raw_descriptor,
                &mut raw_plan,
                &mut error_sink,
            )
        };
        ensure_ok(status, &buffer, error_sink.message_length)?;
        let raw = NonNull::new(raw_plan).ok_or_else(|| {
            RuntimeError::new(
                RuntimeStatus::InternalError,
                "native elementwise prepare returned a null plan on success".to_owned(),
            )
        })?;
        Ok(PreparedElementwise {
            state: Arc::new(PreparedElementwiseState {
                raw,
                owners: PreparedElementwiseOwners {
                    context: context.clone(),
                    descriptor,
                },
            }),
        })
    }
}

impl PreparedElementwise {
    pub(crate) fn raw_plan_handle(&self) -> *const sys::sllm_elementwise_plan_t {
        self.state.raw.as_ptr()
    }

    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<(ElementwiseSubmission, ElementwiseDispatchInfo), RuntimeError> {
        let mut info = sys::sllm_elementwise_dispatch_info_t {
            struct_size: size_of::<sys::sllm_elementwise_dispatch_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_ELEMENTWISE_DISPATCH_INFO_VERSION,
            backend: 0,
            dispatch_id: 0,
            operation: 0,
            dispatch_count: 0,
            kernel_id: 0,
            workgroup_size_x: 0,
            grid_size_x: 0,
            fallback_allowed: 0,
            fallback_used: 0,
            element_count: 0,
            kernel_symbol: [0; sys::SLLM_HIP_ELEMENTWISE_KERNEL_SYMBOL_MAX as usize],
            device_symbol: [0; sys::SLLM_HIP_ELEMENTWISE_DEVICE_SYMBOL_MAX as usize],
            gcn_arch_name: [0; sys::SLLM_HIP_MAX_GCN_ARCH_NAME as usize],
            reserved: [0; 8],
        };
        let mut error_buffer = [0_u8; 256];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_completion = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_elementwise_execute(
                self.state.raw.as_ptr(),
                queue.raw_handle()?.as_ptr(),
                &mut raw_completion,
                &mut info,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        let raw_completion = NonNull::new(raw_completion).ok_or_else(|| {
            RuntimeError::new(
                RuntimeStatus::InternalError,
                "native elementwise execute returned a null completion on success".to_owned(),
            )
        })?;
        let completion = Completion::from_native(
            raw_completion,
            &self.state.owners.context,
            queue,
            self.state.owners.descriptor.input0.buffer(),
            0,
            false,
        );
        let dispatch_info = match dispatch_info_from_raw(&info) {
            Ok(dispatch_info) => dispatch_info,
            Err(error) => {
                drop(completion);
                return Err(error);
            }
        };
        Ok((
            ElementwiseSubmission {
                completion,
                _plan: Arc::clone(&self.state),
            },
            dispatch_info,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sllm_core::{DType, TensorView};

    #[test]
    fn elementwise_descriptors_lower_to_one_versioned_abi_family_and_o_proj_handoff() {
        assert_eq!(size_of::<sys::sllm_elementwise_desc_t>(), 584);
        let context = Context::test_without_native();
        let input0 = crate::Buffer::test_without_native(&context);
        let input1 = crate::Buffer::test_without_native(&context);
        let output = crate::Buffer::test_without_native(&context);
        let projected_output = crate::Buffer::test_without_native(&context);
        let view = TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap();
        let copy =
            ElementwiseDescriptor::copy(input0.binding(view.clone()), output.binding(view.clone()))
                .unwrap();
        assert_eq!(copy.operation, ElementwiseOperation::Copy);
        assert_eq!(copy.semantic().kind(), SemanticOpKind::Copy);
        let empty = zero_binding();
        assert!(empty.buffer.is_null());
        assert_eq!(empty.struct_size, 0);

        let add = ElementwiseDescriptor::add(
            input0.binding(view.clone()),
            input1.binding(view.clone()),
            output.binding(view),
        )
        .unwrap();
        assert_eq!(add.operation, ElementwiseOperation::Add);
        assert_eq!(add.semantic().kind(), SemanticOpKind::Add);
        assert!(add.input1.is_some());

        let broadcast_add = ElementwiseDescriptor::broadcast_add(
            input0.binding(TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap()),
            input1.binding(TensorView::contiguous(DType::Bf16, &[5]).unwrap()),
            output.binding(TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap()),
        )
        .unwrap();
        assert_eq!(broadcast_add.operation, ElementwiseOperation::BroadcastAdd);
        assert_eq!(
            broadcast_add.semantic().kind(),
            SemanticOpKind::BroadcastAdd
        );

        let broadcast_mul = ElementwiseDescriptor::broadcast_mul(
            input0.binding(TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap()),
            input1.binding(TensorView::contiguous(DType::Bf16, &[5]).unwrap()),
            output.binding(TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap()),
        )
        .unwrap();
        assert_eq!(broadcast_mul.operation, ElementwiseOperation::BroadcastMul);
        assert_eq!(
            broadcast_mul.semantic().kind(),
            SemanticOpKind::BroadcastMul
        );

        let silu_mul = ElementwiseDescriptor::silu_mul(
            input0.binding(TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap()),
            input1.binding(TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap()),
            output.binding(TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap()),
        )
        .unwrap();
        assert_eq!(silu_mul.operation, ElementwiseOperation::SiluMul);
        assert_eq!(silu_mul.semantic().kind(), SemanticOpKind::SiluMul);

        let scalar_view = TensorView::contiguous(DType::Bf16, &[1]).unwrap();
        let scalar_mul = ElementwiseDescriptor::scalar_mul(
            input0.binding(TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap()),
            input1.binding(scalar_view.clone()),
            output.binding(TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap()),
        )
        .unwrap();
        assert_eq!(scalar_mul.operation, ElementwiseOperation::ScalarMul);
        assert_eq!(scalar_mul.semantic().kind(), SemanticOpKind::ScalarMul);

        let gelu_tanh_mul = ElementwiseDescriptor::gelu_tanh_mul(
            input0.binding(TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap()),
            input1.binding(TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap()),
            output.binding(TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap()),
        )
        .unwrap();
        assert_eq!(gelu_tanh_mul.operation, ElementwiseOperation::GeluTanhMul);
        assert_eq!(gelu_tanh_mul.semantic().kind(), SemanticOpKind::GeluTanhMul);

        let tanh_softcap = ElementwiseDescriptor::tanh_softcap(
            input0.binding(TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap()),
            input1.binding(scalar_view),
            output.binding(TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap()),
        )
        .unwrap();
        assert_eq!(tanh_softcap.operation, ElementwiseOperation::TanhSoftcap);
        assert_eq!(tanh_softcap.semantic().kind(), SemanticOpKind::TanhSoftcap);

        let gate_view = TensorView::contiguous(DType::Bf16, &[3, 16, 256]).unwrap();
        let sigmoid_mul = ElementwiseDescriptor::sigmoid_mul(
            input0.binding(gate_view.clone()),
            input1.binding(gate_view.clone()),
            output.binding(gate_view),
        )
        .unwrap();
        assert_eq!(sigmoid_mul.operation, ElementwiseOperation::SigmoidMul);
        assert_eq!(sigmoid_mul.semantic().kind(), SemanticOpKind::SigmoidMul);
        let o_proj_input = sigmoid_mul
            .sigmoid_mul_o_proj_input()
            .expect("validated sigmoid output gate has an o_proj input view");
        assert_eq!(o_proj_input.view().shape(), &[3, 4096]);
        assert_eq!(o_proj_input.view().strides(), &[4096, 1]);
        let weight = input1.binding(TensorView::contiguous(DType::Bf16, &[2560, 4096]).unwrap());
        let projected =
            projected_output.binding(TensorView::contiguous(DType::Bf16, &[3, 2560]).unwrap());
        let matmul = crate::MatmulDescriptor::new(o_proj_input, weight, projected)
            .expect("sigmoid output gate storage is accepted by the existing matmul path");
        assert_eq!(matmul.semantic().inputs()[0].shape(), &[3, 4096]);
    }
}
