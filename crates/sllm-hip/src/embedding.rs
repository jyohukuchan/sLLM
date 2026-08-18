//! Safe single-GPU BF16 embedding gather wrapper.

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_core::{SemanticOpDescriptor, SemanticOpKind};
use sllm_hip_sys as sys;

use crate::runtime::{
    Completion, CompletionState, Context, Queue, RuntimeError, RuntimeStatus,
    enqueue_embedding_cleanup, ensure_ok, release_embedding_plan_once, sink,
};
use crate::{HipBackend, TensorBinding};

#[derive(Clone, Debug)]
pub struct EmbeddingDescriptor {
    weight: TensorBinding,
    token_ids: TensorBinding,
    output: TensorBinding,
    semantic: Arc<SemanticOpDescriptor>,
}

impl EmbeddingDescriptor {
    pub fn new(
        weight: TensorBinding,
        token_ids: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, sllm_core::OpError> {
        let semantic = Arc::new(SemanticOpDescriptor::new(
            SemanticOpKind::Embedding,
            vec![weight.view().clone(), token_ids.view().clone()],
            vec![output.view().clone()],
        )?);
        Ok(Self {
            weight,
            token_ids,
            output,
            semantic,
        })
    }

    pub fn semantic(&self) -> &SemanticOpDescriptor {
        self.semantic.as_ref()
    }

    pub(crate) fn from_validated_semantic(
        semantic: Arc<SemanticOpDescriptor>,
        weight: TensorBinding,
        token_ids: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, RuntimeError> {
        semantic.validate().map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidEmbeddingDescriptor,
                format!("invalid validated embedding descriptor: {error}"),
            )
        })?;
        if semantic.kind() != SemanticOpKind::Embedding
            || semantic.inputs().len() != 2
            || semantic.outputs().len() != 1
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidEmbeddingDescriptor,
                "semantic descriptor is not a canonical embedding operation",
            ));
        }
        if weight.view() != &semantic.inputs()[0]
            || token_ids.view() != &semantic.inputs()[1]
            || output.view() != &semantic.outputs()[0]
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidEmbeddingDescriptor,
                "bound HIP tensor views differ from the core embedding descriptor",
            ));
        }
        Ok(Self {
            weight,
            token_ids,
            output,
            semantic,
        })
    }

    fn raw(&self) -> Result<sys::sllm_embedding_desc_t, RuntimeError> {
        Ok(sys::sllm_embedding_desc_t {
            struct_size: size_of::<sys::sllm_embedding_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_EMBEDDING_VERSION,
            reserved: [0; 5],
            weight: self.weight.raw()?,
            token_ids: self.token_ids.raw()?,
            output: self.output.raw()?,
        })
    }
}

struct PreparedEmbeddingOwners {
    context: Context,
    descriptor: EmbeddingDescriptor,
}

struct PreparedEmbeddingState {
    raw: NonNull<sys::sllm_embedding_plan_t>,
    owners: PreparedEmbeddingOwners,
}

// SAFETY: the native plan is an opaque registry token. Its immutable owner
// graph and native transitions follow the same serialized contract as the
// existing numeric plans.
unsafe impl Send for PreparedEmbeddingState {}
unsafe impl Sync for PreparedEmbeddingState {}

impl Drop for PreparedEmbeddingState {
    fn drop(&mut self) {
        let (status, remaining) = release_embedding_plan_once(self.raw);
        if let Some(remaining) = remaining {
            enqueue_embedding_cleanup(
                remaining,
                self.owners.context.clone(),
                self.owners.descriptor.clone(),
                status,
            );
        }
    }
}

#[derive(Clone)]
pub struct PreparedEmbedding {
    state: Arc<PreparedEmbeddingState>,
}

impl std::fmt::Debug for PreparedEmbedding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedEmbedding")
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmbeddingDispatchInfo {
    pub abi_version: u32,
    pub info_version: u32,
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
    pub token_count: u64,
    pub hidden_size: u64,
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

pub struct EmbeddingSubmission {
    completion: Completion,
    _plan: Arc<PreparedEmbeddingState>,
}

impl std::fmt::Debug for EmbeddingSubmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EmbeddingSubmission")
            .finish_non_exhaustive()
    }
}

impl EmbeddingSubmission {
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
    pub fn prepare_embedding(
        &self,
        context: &Context,
        descriptor: EmbeddingDescriptor,
    ) -> Result<PreparedEmbedding, RuntimeError> {
        let raw_descriptor = descriptor.raw()?;
        let mut raw_plan = std::ptr::null_mut();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_embedding_prepare(
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
                "native embedding prepare returned a null plan on success".to_owned(),
            )
        })?;
        Ok(PreparedEmbedding {
            state: Arc::new(PreparedEmbeddingState {
                raw,
                owners: PreparedEmbeddingOwners {
                    context: context.clone(),
                    descriptor,
                },
            }),
        })
    }
}

impl PreparedEmbedding {
    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<(EmbeddingSubmission, EmbeddingDispatchInfo), RuntimeError> {
        let mut info = sys::sllm_embedding_dispatch_info_t {
            struct_size: size_of::<sys::sllm_embedding_dispatch_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_EMBEDDING_DISPATCH_INFO_VERSION,
            backend: 0,
            dispatch_id: 0,
            dispatch_count: 0,
            kernel_id: 0,
            workgroup_size_x: 0,
            grid_size_x: 0,
            fallback_allowed: 0,
            fallback_used: 0,
            token_count: 0,
            hidden_size: 0,
            vocab_size: 0,
            kernel_symbol: [0; sys::SLLM_HIP_EMBEDDING_KERNEL_SYMBOL_MAX as usize],
            device_symbol: [0; sys::SLLM_HIP_EMBEDDING_DEVICE_SYMBOL_MAX as usize],
            gcn_arch_name: [0; sys::SLLM_HIP_MAX_GCN_ARCH_NAME as usize],
            reserved: [0; 8],
        };
        let mut error_buffer = [0_u8; 256];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_completion = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_embedding_execute(
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
                "native embedding execute returned a null completion on success".to_owned(),
            )
        })?;
        let dispatch = EmbeddingDispatchInfo {
            abi_version: info.abi_version,
            info_version: info.info_version,
            dispatch_id: info.dispatch_id,
            dispatch_count: info.dispatch_count,
            kernel_id: info.kernel_id,
            workgroup_size_x: info.workgroup_size_x,
            grid_size_x: info.grid_size_x,
            token_count: info.token_count,
            hidden_size: info.hidden_size,
            vocab_size: info.vocab_size,
            backend: info.backend,
            fallback_allowed: info.fallback_allowed != 0,
            fallback_used: info.fallback_used != 0,
            kernel_symbol: read_c_string(&info.kernel_symbol),
            device_symbol: read_c_string(&info.device_symbol),
            gcn_arch_name: read_c_string(&info.gcn_arch_name),
        };
        let completion = Completion::from_native(
            raw_completion,
            &self.state.owners.context,
            queue,
            self.state.owners.descriptor.weight.buffer(),
            0,
            false,
        );
        Ok((
            EmbeddingSubmission {
                completion,
                _plan: Arc::clone(&self.state),
            },
            dispatch,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sllm_core::{DType, TensorView};

    #[test]
    fn descriptor_requires_i32_ids_and_exact_output_shape() {
        let context = Context::test_without_native();
        let weight = crate::Buffer::test_without_native(&context);
        let ids = crate::Buffer::test_without_native(&context);
        let output = crate::Buffer::test_without_native(&context);
        let descriptor = EmbeddingDescriptor::new(
            weight.binding(TensorView::contiguous(DType::Bf16, &[7, 3]).unwrap()),
            ids.binding(TensorView::contiguous(DType::I32, &[2]).unwrap()),
            output.binding(TensorView::contiguous(DType::Bf16, &[2, 3]).unwrap()),
        )
        .unwrap();
        assert_eq!(descriptor.semantic().kind(), SemanticOpKind::Embedding);
        assert_eq!(size_of::<sys::sllm_embedding_desc_t>(), 584);
    }
}
