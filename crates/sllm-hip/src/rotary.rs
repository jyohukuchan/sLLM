//! Safe split-half rotary preparation and asynchronous execution.

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_core::{SemanticOpDescriptor, SemanticOpKind, SplitHalfRotaryContract};
use sllm_hip_sys as sys;

use crate::runtime::{
    Completion, CompletionState, Context, Queue, RuntimeError, RuntimeStatus,
    enqueue_rotary_cleanup, ensure_ok, release_rotary_plan_once, sink,
};
use crate::{HipBackend, TensorBinding};

#[derive(Clone, Debug)]
pub struct RotaryDescriptor {
    query: TensorBinding,
    key: TensorBinding,
    positions: TensorBinding,
    query_output: TensorBinding,
    key_output: TensorBinding,
    semantic: Arc<SemanticOpDescriptor>,
}

impl RotaryDescriptor {
    pub fn new(
        query: TensorBinding,
        key: TensorBinding,
        positions: TensorBinding,
        query_output: TensorBinding,
        key_output: TensorBinding,
        contract: SplitHalfRotaryContract,
    ) -> Result<Self, sllm_core::OpError> {
        let semantic = Arc::new(SemanticOpDescriptor::new_rotary(
            vec![
                query.view().clone(),
                key.view().clone(),
                positions.view().clone(),
            ],
            vec![query_output.view().clone(), key_output.view().clone()],
            contract,
        )?);
        Ok(Self {
            query,
            key,
            positions,
            query_output,
            key_output,
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
        positions: TensorBinding,
        query_output: TensorBinding,
        key_output: TensorBinding,
    ) -> Result<Self, RuntimeError> {
        semantic.validate().map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidRotaryDescriptor,
                format!("invalid validated rotary descriptor: {error}"),
            )
        })?;
        if semantic.kind() != SemanticOpKind::Rotary
            || semantic.inputs().len() != 3
            || semantic.outputs().len() != 2
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidRotaryDescriptor,
                "semantic descriptor is not a canonical split-half rotary operation",
            ));
        }
        let input_views = [query.view(), key.view(), positions.view()];
        if input_views
            .iter()
            .zip(semantic.inputs())
            .any(|(actual, expected)| *actual != expected)
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidRotaryDescriptor,
                "bound HIP input tensor views differ from the core rotary descriptor",
            ));
        }
        let output_views = [query_output.view(), key_output.view()];
        if output_views
            .iter()
            .zip(semantic.outputs())
            .any(|(actual, expected)| *actual != expected)
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidRotaryDescriptor,
                "bound HIP output tensor views differ from the core rotary descriptor",
            ));
        }
        Ok(Self {
            query,
            key,
            positions,
            query_output,
            key_output,
            semantic,
        })
    }

    fn raw(&self) -> Result<sys::sllm_rotary_desc_t, RuntimeError> {
        let contract = self.semantic.rotary_contract().ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidRotaryDescriptor,
                "split-half rotary contract is absent",
            )
        })?;
        Ok(sys::sllm_rotary_desc_t {
            struct_size: size_of::<sys::sllm_rotary_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_ROTARY_VERSION,
            reserved0: 0,
            start_position: u64::from(contract.start_position()),
            q_heads: contract.q_heads(),
            kv_heads: contract.kv_heads(),
            head_dim: contract.head_dim(),
            rotary_dim: contract.rotary_dim(),
            theta_bits: contract.theta_bits(),
            max_position: contract.max_position_embeddings(),
            reserved: [0; 2],
            query: self.query.raw()?,
            key: self.key.raw()?,
            positions: self.positions.raw()?,
            query_output: self.query_output.raw()?,
            key_output: self.key_output.raw()?,
        })
    }
}

struct PreparedRotaryOwners {
    context: Context,
    descriptor: RotaryDescriptor,
}

pub(crate) struct PreparedRotaryState {
    raw: NonNull<sys::sllm_rotary_plan_t>,
    owners: PreparedRotaryOwners,
}

// SAFETY: the opaque plan token is serialized by the native registry and its
// retained owner graph is immutable.
unsafe impl Send for PreparedRotaryState {}
unsafe impl Sync for PreparedRotaryState {}

impl Drop for PreparedRotaryState {
    fn drop(&mut self) {
        let (status, remaining) = release_rotary_plan_once(self.raw);
        if let Some(remaining) = remaining {
            enqueue_rotary_cleanup(
                remaining,
                self.owners.context.clone(),
                self.owners.descriptor.clone(),
                status,
            );
        }
    }
}

#[derive(Clone)]
pub struct PreparedRotary {
    state: Arc<PreparedRotaryState>,
}

unsafe impl Send for PreparedRotary {}
unsafe impl Sync for PreparedRotary {}

impl std::fmt::Debug for PreparedRotary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedRotary")
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RotaryDispatchInfo {
    pub abi_version: u32,
    pub info_version: u32,
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
    pub token_count: u64,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub start_position: u32,
    pub max_position: u32,
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

fn dispatch_info_from_raw(info: &sys::sllm_rotary_dispatch_info_t) -> RotaryDispatchInfo {
    RotaryDispatchInfo {
        abi_version: info.abi_version,
        info_version: info.info_version,
        dispatch_id: info.dispatch_id,
        dispatch_count: info.dispatch_count,
        kernel_id: info.kernel_id,
        workgroup_size_x: info.workgroup_size_x,
        grid_size_x: info.grid_size_x,
        token_count: info.token_count,
        q_heads: info.q_heads,
        kv_heads: info.kv_heads,
        head_dim: info.head_dim,
        rotary_dim: info.rotary_dim,
        start_position: info.start_position,
        max_position: info.max_position,
        backend: info.backend,
        fallback_allowed: info.fallback_allowed != 0,
        fallback_used: info.fallback_used != 0,
        kernel_symbol: read_c_string(&info.kernel_symbol),
        device_symbol: read_c_string(&info.device_symbol),
        gcn_arch_name: read_c_string(&info.gcn_arch_name),
    }
}

pub struct RotarySubmission {
    completion: Completion,
    _plan: Arc<PreparedRotaryState>,
}

impl std::fmt::Debug for RotarySubmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RotarySubmission")
            .finish_non_exhaustive()
    }
}

impl RotarySubmission {
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
    pub fn prepare_rotary(
        &self,
        context: &Context,
        descriptor: RotaryDescriptor,
    ) -> Result<PreparedRotary, RuntimeError> {
        let raw_descriptor = descriptor.raw()?;
        let mut raw_plan = std::ptr::null_mut();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_rotary_prepare(
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
                "native rotary prepare returned a null plan on success".to_owned(),
            )
        })?;
        Ok(PreparedRotary {
            state: Arc::new(PreparedRotaryState {
                raw,
                owners: PreparedRotaryOwners {
                    context: context.clone(),
                    descriptor,
                },
            }),
        })
    }
}

impl PreparedRotary {
    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<(RotarySubmission, RotaryDispatchInfo), RuntimeError> {
        let mut info = sys::sllm_rotary_dispatch_info_t {
            struct_size: size_of::<sys::sllm_rotary_dispatch_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_ROTARY_DISPATCH_INFO_VERSION,
            backend: 0,
            dispatch_id: 0,
            dispatch_count: 0,
            kernel_id: 0,
            workgroup_size_x: 0,
            grid_size_x: 0,
            token_count: 0,
            q_heads: 0,
            kv_heads: 0,
            head_dim: 0,
            rotary_dim: 0,
            start_position: 0,
            max_position: 0,
            fallback_allowed: 0,
            fallback_used: 0,
            kernel_symbol: [0; sys::SLLM_HIP_ROTARY_KERNEL_SYMBOL_MAX as usize],
            device_symbol: [0; sys::SLLM_HIP_ROTARY_DEVICE_SYMBOL_MAX as usize],
            gcn_arch_name: [0; sys::SLLM_HIP_MAX_GCN_ARCH_NAME as usize],
            reserved: [0; 8],
        };
        let mut error_buffer = [0_u8; 256];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_completion = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_rotary_execute(
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
                "native rotary execute returned a null completion on success".to_owned(),
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
            RotarySubmission {
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
    fn semantic_lowering_preserves_split_half_contract() {
        let context = Context::test_without_native();
        let buffers = std::array::from_fn::<_, 5, _>(|_| Buffer::test_without_native(&context));
        let contract = SplitHalfRotaryContract::new(3, 1, 6, 4, 10_000.0, 257, 3, 262_144)
            .expect("valid non-aligned rotary contract");
        let descriptor = RotaryDescriptor::new(
            buffers[0].binding(TensorView::contiguous(DType::Bf16, &[3, 3, 6]).unwrap()),
            buffers[1].binding(TensorView::contiguous(DType::Bf16, &[3, 1, 6]).unwrap()),
            buffers[2].binding(TensorView::contiguous(DType::I32, &[3]).unwrap()),
            buffers[3].binding(TensorView::contiguous(DType::Bf16, &[3, 3, 6]).unwrap()),
            buffers[4].binding(TensorView::contiguous(DType::Bf16, &[3, 1, 6]).unwrap()),
            contract,
        )
        .expect("valid rotary descriptor");
        assert_eq!(descriptor.semantic().kind(), SemanticOpKind::Rotary);
        assert_eq!(descriptor.semantic().rotary_contract(), Some(contract));
    }
}
