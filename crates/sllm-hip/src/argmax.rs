//! Safe BF16 greedy argmax preparation and asynchronous baseline execution.

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_core::{SemanticOpDescriptor, SemanticOpKind};
use sllm_hip_sys as sys;

use crate::runtime::{
    Completion, CompletionState, Context, Queue, RuntimeError, RuntimeStatus,
    enqueue_argmax_cleanup, ensure_ok, release_argmax_plan_once, sink,
};
use crate::{HipBackend, TensorBinding};

#[derive(Clone, Debug)]
pub struct ArgmaxDescriptor {
    logits: TensorBinding,
    output: TensorBinding,
    semantic: Arc<SemanticOpDescriptor>,
}

impl ArgmaxDescriptor {
    pub fn new(logits: TensorBinding, output: TensorBinding) -> Result<Self, sllm_core::OpError> {
        let semantic = Arc::new(SemanticOpDescriptor::new(
            SemanticOpKind::Argmax,
            vec![logits.view().clone()],
            vec![output.view().clone()],
        )?);
        Ok(Self {
            logits,
            output,
            semantic,
        })
    }

    pub fn semantic(&self) -> &SemanticOpDescriptor {
        self.semantic.as_ref()
    }

    pub(crate) fn from_validated_semantic(
        semantic: Arc<SemanticOpDescriptor>,
        logits: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, RuntimeError> {
        semantic.validate().map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidArgmaxDescriptor,
                format!("invalid validated argmax descriptor: {error}"),
            )
        })?;
        if semantic.kind() != SemanticOpKind::Argmax
            || semantic.inputs().len() != 1
            || semantic.outputs().len() != 1
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgmaxDescriptor,
                "semantic descriptor is not a canonical argmax operation",
            ));
        }
        if logits.view() != &semantic.inputs()[0] || output.view() != &semantic.outputs()[0] {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgmaxDescriptor,
                "bound HIP tensor views differ from the core argmax descriptor",
            ));
        }
        Ok(Self {
            logits,
            output,
            semantic,
        })
    }

    fn raw(&self) -> Result<sys::sllm_argmax_desc_t, RuntimeError> {
        Ok(sys::sllm_argmax_desc_t {
            struct_size: size_of::<sys::sllm_argmax_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_ARGMAX_VERSION,
            reserved: [0; 5],
            logits: self.logits.raw()?,
            output: self.output.raw()?,
        })
    }
}

struct PreparedArgmaxOwners {
    context: Context,
    descriptor: ArgmaxDescriptor,
}

struct PreparedArgmaxState {
    raw: NonNull<sys::sllm_argmax_plan_t>,
    owners: PreparedArgmaxOwners,
}

// SAFETY: the native plan is an opaque registry token. Native transitions are
// serialized by the public registry/accounting lock and the retained owner
// graph is immutable.
unsafe impl Send for PreparedArgmaxState {}
unsafe impl Sync for PreparedArgmaxState {}

impl Drop for PreparedArgmaxState {
    fn drop(&mut self) {
        let (status, remaining) = release_argmax_plan_once(self.raw);
        if let Some(remaining) = remaining {
            enqueue_argmax_cleanup(
                remaining,
                self.owners.context.clone(),
                self.owners.descriptor.clone(),
                status,
            );
        }
    }
}

#[derive(Clone)]
pub struct PreparedArgmax {
    state: Arc<PreparedArgmaxState>,
}

// SAFETY: the state has the Send/Sync guarantees documented above.
unsafe impl Send for PreparedArgmax {}
unsafe impl Sync for PreparedArgmax {}

impl std::fmt::Debug for PreparedArgmax {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedArgmax")
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArgmaxDispatchInfo {
    pub abi_version: u32,
    pub info_version: u32,
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
    pub row_count: u64,
    pub vocab_size: u64,
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

fn dispatch_info_from_raw(info: &sys::sllm_argmax_dispatch_info_t) -> ArgmaxDispatchInfo {
    ArgmaxDispatchInfo {
        abi_version: info.abi_version,
        info_version: info.info_version,
        dispatch_id: info.dispatch_id,
        dispatch_count: info.dispatch_count,
        kernel_id: info.kernel_id,
        workgroup_size_x: info.workgroup_size_x,
        grid_size_x: info.grid_size_x,
        row_count: info.row_count,
        vocab_size: info.vocab_size,
        backend: info.backend,
        fallback_allowed: info.fallback_allowed != 0,
        fallback_used: info.fallback_used != 0,
        kernel_symbol: read_c_string(&info.kernel_symbol),
        device_symbol: read_c_string(&info.device_symbol),
        gcn_arch_name: read_c_string(&info.gcn_arch_name),
    }
}

pub struct ArgmaxSubmission {
    completion: Completion,
    _plan: Arc<PreparedArgmaxState>,
}

impl std::fmt::Debug for ArgmaxSubmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ArgmaxSubmission")
            .finish_non_exhaustive()
    }
}

impl ArgmaxSubmission {
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
    pub fn prepare_argmax(
        &self,
        context: &Context,
        descriptor: ArgmaxDescriptor,
    ) -> Result<PreparedArgmax, RuntimeError> {
        let raw_descriptor = descriptor.raw()?;
        let mut raw_plan = std::ptr::null_mut();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_argmax_prepare(
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
                "native argmax prepare returned a null plan on success".to_owned(),
            )
        })?;
        Ok(PreparedArgmax {
            state: Arc::new(PreparedArgmaxState {
                raw,
                owners: PreparedArgmaxOwners {
                    context: context.clone(),
                    descriptor,
                },
            }),
        })
    }
}

impl PreparedArgmax {
    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<(ArgmaxSubmission, ArgmaxDispatchInfo), RuntimeError> {
        let mut info = sys::sllm_argmax_dispatch_info_t {
            struct_size: size_of::<sys::sllm_argmax_dispatch_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_ARGMAX_DISPATCH_INFO_VERSION,
            backend: 0,
            dispatch_id: 0,
            dispatch_count: 0,
            kernel_id: 0,
            workgroup_size_x: 0,
            grid_size_x: 0,
            row_count: 0,
            vocab_size: 0,
            fallback_allowed: 0,
            fallback_used: 0,
            kernel_symbol: [0; sys::SLLM_HIP_ARGMAX_KERNEL_SYMBOL_MAX as usize],
            device_symbol: [0; sys::SLLM_HIP_ARGMAX_DEVICE_SYMBOL_MAX as usize],
            gcn_arch_name: [0; sys::SLLM_HIP_MAX_GCN_ARCH_NAME as usize],
            reserved: [0; 8],
        };
        let mut error_buffer = [0_u8; 256];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_completion = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_argmax_execute(
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
                "native argmax execute returned a null completion on success".to_owned(),
            )
        })?;
        let dispatch_info = dispatch_info_from_raw(&info);
        let completion = Completion::from_native(
            raw_completion,
            &self.state.owners.context,
            queue,
            self.state.owners.descriptor.logits.buffer(),
            0,
            false,
        );
        Ok((
            ArgmaxSubmission {
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

    #[test]
    fn public_argmax_submission_exposes_native_completion_timing() {
        let timing: fn(&mut ArgmaxSubmission) -> Result<u64, RuntimeError> =
            ArgmaxSubmission::kernel_elapsed_ns;
        let _ = timing;
    }
}
