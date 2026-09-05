//! Safe lowering for the additive fused residual-add/RMSNorm ABI.

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_core::{RmsNormScaleMode, SemanticOpDescriptor};
use sllm_hip_sys as sys;

use crate::HipBackend;
use crate::rmsnorm::TensorBinding;
use crate::runtime::{Completion, CompletionState, Context, Queue, RuntimeError, ensure_ok, sink};

#[derive(Clone, Debug)]
pub struct ResidualRmsNormDescriptor {
    residual: TensorBinding,
    addend: TensorBinding,
    raw_scale: TensorBinding,
    residual_output: TensorBinding,
    output: TensorBinding,
    semantic: Arc<SemanticOpDescriptor>,
}

impl ResidualRmsNormDescriptor {
    pub(crate) fn from_validated_semantic(
        semantic: Arc<SemanticOpDescriptor>,
        residual: TensorBinding,
        addend: TensorBinding,
        raw_scale: TensorBinding,
        residual_output: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, RuntimeError> {
        semantic.validate().map_err(|error| {
            RuntimeError::new(
                crate::RuntimeStatus::InvalidRmsNormDescriptor,
                error.to_string(),
            )
        })?;
        if semantic.kind() != sllm_core::SemanticOpKind::ResidualRmsNorm
            || semantic.inputs().len() != 3
            || semantic.outputs().len() != 2
            || semantic.inputs()
                != [
                    residual.view().clone(),
                    addend.view().clone(),
                    raw_scale.view().clone(),
                ]
            || semantic.outputs() != [residual_output.view().clone(), output.view().clone()]
        {
            return Err(RuntimeError::local(
                crate::RuntimeStatus::InvalidRmsNormDescriptor,
                "bound HIP tensor views differ from fused semantic descriptor",
            ));
        }
        Ok(Self {
            residual,
            addend,
            raw_scale,
            residual_output,
            output,
            semantic,
        })
    }

    pub fn semantic(&self) -> &SemanticOpDescriptor {
        self.semantic.as_ref()
    }

    fn raw(&self) -> Result<sys::sllm_residual_rmsnorm_desc_t, RuntimeError> {
        let contract = self.semantic.residual_rms_norm_contract().ok_or_else(|| {
            RuntimeError::local(
                crate::RuntimeStatus::InvalidRmsNormDescriptor,
                "fused contract absent",
            )
        })?;
        Ok(sys::sllm_residual_rmsnorm_desc_t {
            struct_size: size_of::<sys::sllm_residual_rmsnorm_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_RESIDUAL_RMSNORM_VERSION,
            accumulation_dtype: sys::SLLM_RMSNORM_ACCUMULATION_F32,
            scale_mode: match contract.scale_mode() {
                RmsNormScaleMode::OffsetOne => sys::SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE,
                RmsNormScaleMode::Direct => sys::SLLM_RMSNORM_SCALE_MODE_DIRECT,
            },
            alias_policy: sys::SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP,
            epsilon_bits: contract.epsilon().bits(),
            reserved: [0; 3],
            residual: self.residual.raw()?,
            addend: self.addend.raw()?,
            raw_scale: self.raw_scale.raw()?,
            residual_output: self.residual_output.raw()?,
            output: self.output.raw()?,
        })
    }
}

struct Owners {
    context: Context,
    descriptor: ResidualRmsNormDescriptor,
}

pub(crate) struct PreparedResidualRmsNormState {
    pub(crate) raw: NonNull<sys::sllm_residual_rmsnorm_plan_t>,
    owners: Owners,
}
unsafe impl Send for PreparedResidualRmsNormState {}
unsafe impl Sync for PreparedResidualRmsNormState {}
impl Drop for PreparedResidualRmsNormState {
    fn drop(&mut self) {
        let mut raw = self.raw.as_ptr();
        let mut bytes = [0_u8; 256];
        let mut error_sink = sink(&mut bytes);
        let _ = unsafe { sys::sllm_residual_rmsnorm_plan_release(&mut raw, &mut error_sink) };
    }
}

#[derive(Clone)]
pub struct PreparedResidualRmsNorm {
    pub(crate) state: Arc<PreparedResidualRmsNormState>,
}
unsafe impl Send for PreparedResidualRmsNorm {}
unsafe impl Sync for PreparedResidualRmsNorm {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidualRmsNormDispatchInfo {
    pub abi_version: u32,
    pub info_version: u32,
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub row_count: u64,
    pub normalized_size: u64,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
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
        .map(|byte| *byte as u8 as char)
        .collect()
}

pub struct ResidualRmsNormSubmission {
    completion: Completion,
    _plan: Arc<PreparedResidualRmsNormState>,
}
impl ResidualRmsNormSubmission {
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
        token: u64,
    ) -> Result<CompletionState, RuntimeError> {
        self.completion.finalize_after_token(token)
    }
}

impl HipBackend {
    pub fn prepare_residual_rms_norm(
        &self,
        context: &Context,
        descriptor: ResidualRmsNormDescriptor,
    ) -> Result<PreparedResidualRmsNorm, RuntimeError> {
        let raw_descriptor = descriptor.raw()?;
        let mut raw_plan = std::ptr::null_mut();
        let mut bytes = [0_u8; 256];
        let mut error_sink = sink(&mut bytes);
        let status = unsafe {
            sys::sllm_residual_rmsnorm_prepare(
                context.raw_handle()?.as_ptr(),
                &raw_descriptor,
                &mut raw_plan,
                &mut error_sink,
            )
        };
        ensure_ok(status, &bytes, error_sink.message_length)?;
        let raw = NonNull::new(raw_plan).ok_or_else(|| {
            RuntimeError::local(
                crate::RuntimeStatus::InternalError,
                "fused prepare returned null plan",
            )
        })?;
        Ok(PreparedResidualRmsNorm {
            state: Arc::new(PreparedResidualRmsNormState {
                raw,
                owners: Owners {
                    context: context.clone(),
                    descriptor,
                },
            }),
        })
    }
}

impl PreparedResidualRmsNorm {
    pub(crate) fn raw_plan_handle(&self) -> *const sys::sllm_residual_rmsnorm_plan_t {
        self.state.raw.as_ptr()
    }

    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<(ResidualRmsNormSubmission, ResidualRmsNormDispatchInfo), RuntimeError> {
        let mut info = sys::sllm_residual_rmsnorm_dispatch_info_t {
            struct_size: size_of::<sys::sllm_residual_rmsnorm_dispatch_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: 1,
            backend: 0,
            dispatch_id: 0,
            dispatch_count: 0,
            kernel_id: 0,
            workgroup_size_x: 0,
            grid_size_x: 0,
            row_count: 0,
            normalized_size: 0,
            fallback_allowed: 0,
            fallback_used: 0,
            kernel_symbol: [0; 64],
            device_symbol: [0; 64],
            gcn_arch_name: [0; 64],
            reserved: [0; 8],
        };
        let mut bytes = [0_u8; 256];
        let mut error_sink = sink(&mut bytes);
        let mut completion = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_residual_rmsnorm_execute(
                self.state.raw.as_ptr(),
                queue.raw_handle()?.as_ptr(),
                &mut completion,
                &mut info,
                &mut error_sink,
            )
        };
        ensure_ok(status, &bytes, error_sink.message_length)?;
        let completion = NonNull::new(completion).ok_or_else(|| {
            RuntimeError::local(
                crate::RuntimeStatus::InternalError,
                "fused execute returned null completion",
            )
        })?;
        let completion = Completion::from_native(
            completion,
            &self.state.owners.context,
            queue,
            self.state.owners.descriptor.residual.buffer(),
            0,
            false,
        );
        let dispatch = ResidualRmsNormDispatchInfo {
            abi_version: info.abi_version,
            info_version: info.info_version,
            dispatch_id: info.dispatch_id,
            dispatch_count: info.dispatch_count,
            kernel_id: info.kernel_id,
            workgroup_size_x: info.workgroup_size_x,
            grid_size_x: info.grid_size_x,
            row_count: info.row_count,
            normalized_size: info.normalized_size,
            backend: info.backend,
            fallback_allowed: info.fallback_allowed != 0,
            fallback_used: info.fallback_used != 0,
            kernel_symbol: read_c_string(&info.kernel_symbol),
            device_symbol: read_c_string(&info.device_symbol),
            gcn_arch_name: read_c_string(&info.gcn_arch_name),
        };
        Ok((
            ResidualRmsNormSubmission {
                completion,
                _plan: Arc::clone(&self.state),
            },
            dispatch,
        ))
    }
}
