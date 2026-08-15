//! Safe model-neutral windowed/full causal-attention execution.

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_core::{SemanticOpDescriptor, SemanticOpKind, WindowedCausalAttentionContract};
use sllm_hip_sys as sys;

use crate::runtime::{
    Completion, CompletionState, Context, Queue, RuntimeError, RuntimeStatus,
    enqueue_windowed_attention_cleanup, ensure_ok, release_windowed_attention_plan_once, sink,
};
use crate::{HipBackend, TensorBinding};

#[derive(Clone, Debug)]
pub struct WindowedAttentionDescriptor {
    query: TensorBinding,
    key: TensorBinding,
    value: TensorBinding,
    output: TensorBinding,
    semantic: Arc<SemanticOpDescriptor>,
}

impl WindowedAttentionDescriptor {
    pub fn new(
        query: TensorBinding,
        key: TensorBinding,
        value: TensorBinding,
        output: TensorBinding,
        contract: WindowedCausalAttentionContract,
    ) -> Result<Self, sllm_core::OpError> {
        let semantic = Arc::new(SemanticOpDescriptor::new_causal_attention(
            vec![
                query.view().clone(),
                key.view().clone(),
                value.view().clone(),
            ],
            vec![output.view().clone()],
            contract,
        )?);
        Ok(Self {
            query,
            key,
            value,
            output,
            semantic,
        })
    }

    pub fn semantic(&self) -> &SemanticOpDescriptor {
        self.semantic.as_ref()
    }

    pub(crate) fn from_validated_semantic(
        semantic: Arc<SemanticOpDescriptor>,
        query: TensorBinding,
        key: TensorBinding,
        value: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, RuntimeError> {
        semantic.validate().map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidWindowedAttentionDescriptor,
                format!("invalid validated windowed attention descriptor: {error}"),
            )
        })?;
        if semantic.kind() != SemanticOpKind::CausalAttention
            || semantic.inputs().len() != 3
            || semantic.outputs().len() != 1
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidWindowedAttentionDescriptor,
                "semantic descriptor is not canonical windowed causal attention",
            ));
        }
        let input_views = [query.view(), key.view(), value.view()];
        if input_views
            .iter()
            .zip(semantic.inputs())
            .any(|(actual, expected)| *actual != expected)
            || output.view() != &semantic.outputs()[0]
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidWindowedAttentionDescriptor,
                "bound HIP tensor views differ from the core attention descriptor",
            ));
        }
        Ok(Self {
            query,
            key,
            value,
            output,
            semantic,
        })
    }

    fn raw(&self) -> Result<sys::sllm_windowed_attention_desc_t, RuntimeError> {
        let contract = self.semantic.causal_attention_contract().ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidWindowedAttentionDescriptor,
                "windowed causal-attention contract is absent",
            )
        })?;
        Ok(sys::sllm_windowed_attention_desc_t {
            struct_size: size_of::<sys::sllm_windowed_attention_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_WINDOWED_ATTENTION_VERSION,
            reserved0: 0,
            start_position: contract.start_position(),
            expected_kv_length: contract.expected_kv_length(),
            sliding_window: contract.sliding_window().unwrap_or(0),
            q_heads: contract.q_heads(),
            kv_heads: contract.kv_heads(),
            head_dim: contract.head_dim(),
            scaling_bits: contract.scaling_bits(),
            reserved: [0; 4],
            query: self.query.raw()?,
            key: self.key.raw()?,
            value: self.value.raw()?,
            output: self.output.raw()?,
        })
    }
}

struct PreparedWindowedAttentionOwners {
    context: Context,
    descriptor: WindowedAttentionDescriptor,
}

pub(crate) struct PreparedWindowedAttentionState {
    raw: NonNull<sys::sllm_windowed_attention_plan_t>,
    owners: PreparedWindowedAttentionOwners,
}

unsafe impl Send for PreparedWindowedAttentionState {}
unsafe impl Sync for PreparedWindowedAttentionState {}

impl Drop for PreparedWindowedAttentionState {
    fn drop(&mut self) {
        let (status, remaining) = release_windowed_attention_plan_once(self.raw);
        if let Some(remaining) = remaining {
            enqueue_windowed_attention_cleanup(
                remaining,
                self.owners.context.clone(),
                self.owners.descriptor.clone(),
                status,
            );
        }
    }
}

#[derive(Clone)]
pub struct PreparedWindowedAttention {
    state: Arc<PreparedWindowedAttentionState>,
}

unsafe impl Send for PreparedWindowedAttention {}
unsafe impl Sync for PreparedWindowedAttention {}

impl std::fmt::Debug for PreparedWindowedAttention {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedWindowedAttention")
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowedAttentionDispatchInfo {
    pub abi_version: u32,
    pub info_version: u32,
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
    pub query_count: u64,
    pub start_position: u64,
    pub committed_kv_length: u64,
    pub sliding_window: u64,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub scaling_bits: u32,
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
    info: &sys::sllm_windowed_attention_dispatch_info_t,
) -> WindowedAttentionDispatchInfo {
    WindowedAttentionDispatchInfo {
        abi_version: info.abi_version,
        info_version: info.info_version,
        dispatch_id: info.dispatch_id,
        dispatch_count: info.dispatch_count,
        kernel_id: info.kernel_id,
        workgroup_size_x: info.workgroup_size_x,
        grid_size_x: info.grid_size_x,
        query_count: info.query_count,
        start_position: info.start_position,
        committed_kv_length: info.committed_kv_length,
        sliding_window: info.sliding_window,
        q_heads: info.q_heads,
        kv_heads: info.kv_heads,
        head_dim: info.head_dim,
        scaling_bits: info.scaling_bits,
        backend: info.backend,
        fallback_allowed: info.fallback_allowed != 0,
        fallback_used: info.fallback_used != 0,
        kernel_symbol: read_c_string(&info.kernel_symbol),
        device_symbol: read_c_string(&info.device_symbol),
        gcn_arch_name: read_c_string(&info.gcn_arch_name),
    }
}

pub struct WindowedAttentionSubmission {
    completion: Completion,
    _plan: Arc<PreparedWindowedAttentionState>,
}

impl WindowedAttentionSubmission {
    pub fn query(&mut self) -> Result<CompletionState, RuntimeError> {
        self.completion.query()
    }

    pub fn wait(&mut self, timeout: std::time::Duration) -> Result<CompletionState, RuntimeError> {
        self.completion.wait(timeout)
    }

    pub fn kernel_elapsed_ns(&mut self) -> Result<u64, RuntimeError> {
        self.completion.kernel_elapsed_ns()
    }
}

impl HipBackend {
    pub fn prepare_windowed_attention(
        &self,
        context: &Context,
        descriptor: WindowedAttentionDescriptor,
    ) -> Result<PreparedWindowedAttention, RuntimeError> {
        let raw_descriptor = descriptor.raw()?;
        let mut raw_plan = std::ptr::null_mut();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_windowed_attention_prepare(
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
                "native windowed attention prepare returned a null plan".to_owned(),
            )
        })?;
        Ok(PreparedWindowedAttention {
            state: Arc::new(PreparedWindowedAttentionState {
                raw,
                owners: PreparedWindowedAttentionOwners {
                    context: context.clone(),
                    descriptor,
                },
            }),
        })
    }
}

impl PreparedWindowedAttention {
    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<(WindowedAttentionSubmission, WindowedAttentionDispatchInfo), RuntimeError> {
        let mut info = sys::sllm_windowed_attention_dispatch_info_t {
            struct_size: size_of::<sys::sllm_windowed_attention_dispatch_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_WINDOWED_ATTENTION_DISPATCH_INFO_VERSION,
            backend: 0,
            dispatch_id: 0,
            dispatch_count: 0,
            kernel_id: 0,
            workgroup_size_x: 0,
            grid_size_x: 0,
            query_count: 0,
            start_position: 0,
            committed_kv_length: 0,
            sliding_window: 0,
            q_heads: 0,
            kv_heads: 0,
            head_dim: 0,
            scaling_bits: 0,
            fallback_allowed: 0,
            fallback_used: 0,
            kernel_symbol: [0; sys::SLLM_HIP_WINDOWED_ATTENTION_KERNEL_SYMBOL_MAX as usize],
            device_symbol: [0; sys::SLLM_HIP_WINDOWED_ATTENTION_DEVICE_SYMBOL_MAX as usize],
            gcn_arch_name: [0; sys::SLLM_HIP_MAX_GCN_ARCH_NAME as usize],
            reserved: [0; 8],
        };
        let mut error_buffer = [0_u8; 256];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_completion = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_windowed_attention_execute(
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
                "native windowed attention execute returned a null completion".to_owned(),
            )
        })?;
        let dispatch_info = dispatch_info_from_raw(&info);
        let completion = Completion::from_native(
            raw_completion,
            &self.state.owners.context,
            queue,
            self.state.owners.descriptor.query.buffer(),
            0,
            false,
        );
        Ok((
            WindowedAttentionSubmission {
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
    use crate::Buffer;
    use sllm_core::{DType, TensorView};

    #[test]
    fn semantic_lowering_preserves_sliding_window_contract() {
        let context = Context::test_without_native();
        let buffers = std::array::from_fn::<_, 4, _>(|_| Buffer::test_without_native(&context));
        let contract = WindowedCausalAttentionContract::new(3, 1, 6, 2, 3, 5, Some(4), 1.0)
            .expect("valid non-aligned attention contract");
        let descriptor = WindowedAttentionDescriptor::new(
            buffers[0].binding(TensorView::contiguous(DType::Bf16, &[3, 3, 6]).unwrap()),
            buffers[1].binding(TensorView::contiguous(DType::Bf16, &[5, 1, 6]).unwrap()),
            buffers[2].binding(TensorView::contiguous(DType::Bf16, &[5, 1, 6]).unwrap()),
            buffers[3].binding(TensorView::contiguous(DType::Bf16, &[3, 3, 6]).unwrap()),
            contract,
        )
        .expect("valid attention descriptor");
        assert_eq!(
            descriptor.semantic().kind(),
            SemanticOpKind::CausalAttention
        );
        assert_eq!(
            descriptor.semantic().causal_attention_contract(),
            Some(contract)
        );
    }
}
