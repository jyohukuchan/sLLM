//! Safe active-pair Qwen3.5 and Gemma 4 MoE expert execution over the public
//! HIP ABI.

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_core::{DType, Encoding};
use sllm_hip_sys as sys;

use crate::runtime::{
    Completion, CompletionState, Context, Queue, RuntimeError, RuntimeStatus, ensure_ok, sink,
};
use crate::{HipBackend, TensorBinding};

pub const fn moe_expert_workspace_bytes(token_count: u64) -> Option<u64> {
    token_count.checked_mul(12_484)
}

pub const fn gemma4_moe_expert_workspace_bytes(token_count: u64) -> Option<u64> {
    token_count.checked_mul(sys::SLLM_HIP_GEMMA4_MOE_EXPERT_WORKSPACE_BYTES_PER_TOKEN)
}

#[derive(Clone, Debug)]
pub struct MoeExpertDescriptor {
    hidden: TensorBinding,
    routing_metadata: TensorBinding,
    layer_blob: TensorBinding,
    workspace: TensorBinding,
    output: TensorBinding,
    token_count: u64,
    op_version: u32,
}

impl MoeExpertDescriptor {
    pub fn new(
        hidden: TensorBinding,
        routing_metadata: TensorBinding,
        layer_blob: TensorBinding,
        workspace: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, RuntimeError> {
        Self::new_fixed(
            hidden,
            routing_metadata,
            layer_blob,
            workspace,
            output,
            sys::SLLM_HIP_MOE_EXPERT_VERSION,
            sys::SLLM_HIP_MOE_EXPERT_HIDDEN_SIZE as u64,
            sys::SLLM_HIP_MOE_EXPERT_COUNT as u64,
            sys::SLLM_HIP_MOE_EXPERT_TOPK as u64,
            sys::SLLM_HIP_MOE_EXPERT_LAYER_BLOB_BYTES,
            moe_expert_workspace_bytes,
            "MoE expert bindings differ from the fixed Qwen3.5 layer contract",
        )
    }

    pub fn new_gemma4(
        hidden: TensorBinding,
        routing_metadata: TensorBinding,
        layer_blob: TensorBinding,
        workspace: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, RuntimeError> {
        Self::new_fixed(
            hidden,
            routing_metadata,
            layer_blob,
            workspace,
            output,
            sys::SLLM_HIP_MOE_EXPERT_GEMMA4_VERSION,
            sys::SLLM_HIP_GEMMA4_MOE_EXPERT_HIDDEN_SIZE as u64,
            sys::SLLM_HIP_GEMMA4_MOE_EXPERT_COUNT as u64,
            sys::SLLM_HIP_GEMMA4_MOE_EXPERT_TOPK as u64,
            sys::SLLM_HIP_GEMMA4_MOE_EXPERT_LAYER_BLOB_BYTES,
            gemma4_moe_expert_workspace_bytes,
            "MoE expert bindings differ from the fixed Gemma 4 26B-A4B routed-expert contract",
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_fixed(
        hidden: TensorBinding,
        routing_metadata: TensorBinding,
        layer_blob: TensorBinding,
        workspace: TensorBinding,
        output: TensorBinding,
        op_version: u32,
        hidden_size: u64,
        expert_count: u64,
        top_k: u64,
        layer_blob_bytes: u64,
        workspace_size: fn(u64) -> Option<u64>,
        contract_error: &'static str,
    ) -> Result<Self, RuntimeError> {
        let tokens = hidden.view().shape().first().copied().unwrap_or(0) as u64;
        let route_bytes = tokens
            .checked_mul(top_k)
            .and_then(|pairs| pairs.checked_mul(16))
            .and_then(|bytes| {
                expert_count
                    .checked_mul(4)
                    .and_then(|counts| {
                        expert_count
                            .checked_add(1)
                            .and_then(|offsets| offsets.checked_mul(4))
                            .and_then(|offsets| counts.checked_add(offsets))
                    })
                    .and_then(|grouped| bytes.checked_add(grouped))
            })
            .and_then(|bytes| bytes.checked_add(4))
            .ok_or_else(|| {
                RuntimeError::local(RuntimeStatus::InvalidArgument, "MoE layout overflow")
            })?;
        let workspace_bytes = workspace_size(tokens).ok_or_else(|| {
            RuntimeError::local(RuntimeStatus::InvalidArgument, "MoE workspace overflow")
        })?;
        let matrix = |binding: &TensorBinding| {
            binding.view().shape() == [tokens as usize, hidden_size as usize]
                && binding.view().dtype() == DType::Bf16
                && binding.view().encoding() == Encoding::Unquantized
                && binding.view().is_contiguous()
        };
        let bytes = |binding: &TensorBinding, length: u64| {
            binding.view().shape() == [length as usize]
                && binding.view().dtype() == DType::U8
                && binding.view().encoding() == Encoding::Unquantized
                && binding.view().is_contiguous()
        };
        if tokens == 0
            || tokens > sys::SLLM_HIP_MOE_EXPERT_MAX_TOKENS
            || !matrix(&hidden)
            || !matrix(&output)
            || !bytes(&routing_metadata, route_bytes)
            || !bytes(&layer_blob, layer_blob_bytes)
            || !bytes(&workspace, workspace_bytes)
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                contract_error,
            ));
        }
        Ok(Self {
            hidden,
            routing_metadata,
            layer_blob,
            workspace,
            output,
            token_count: tokens,
            op_version,
        })
    }

    pub const fn token_count(&self) -> u64 {
        self.token_count
    }

    fn raw(&self) -> Result<sys::sllm_moe_expert_desc_t, RuntimeError> {
        Ok(sys::sllm_moe_expert_desc_t {
            struct_size: size_of::<sys::sllm_moe_expert_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: self.op_version,
            reserved0: 0,
            reserved: [0; 4],
            hidden: self.hidden.raw()?,
            routing_metadata: self.routing_metadata.raw()?,
            layer_blob: self.layer_blob.raw()?,
            workspace: self.workspace.raw()?,
            output: self.output.raw()?,
        })
    }
}

struct PreparedMoeExpertState {
    raw: NonNull<sys::sllm_moe_expert_plan_t>,
    context: Context,
    descriptor: MoeExpertDescriptor,
}

unsafe impl Send for PreparedMoeExpertState {}
unsafe impl Sync for PreparedMoeExpertState {}

impl Drop for PreparedMoeExpertState {
    fn drop(&mut self) {
        let mut raw = self.raw.as_ptr();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe { sys::sllm_moe_expert_plan_release(&mut raw, &mut error_sink) };
        debug_assert_eq!(status, sys::SLLM_STATUS_OK);
    }
}

#[derive(Clone)]
pub struct PreparedMoeExpert {
    state: Arc<PreparedMoeExpertState>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MoeExpertDispatchInfo {
    pub abi_version: u32,
    pub info_version: u32,
    pub backend: u32,
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
    pub token_count: u64,
    pub active_pair_count: u64,
    pub workspace_bytes: u64,
    pub selected_expert_count: u32,
    pub shared_expert_count: u32,
    pub fallback_allowed: bool,
    pub fallback_used: bool,
    pub gcn_arch_name: String,
    pub kernel_symbol: String,
    pub device_symbol: String,
}

fn c_string(value: &[core::ffi::c_char]) -> String {
    let end = value
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(value.len());
    value[..end]
        .iter()
        .map(|byte| char::from(*byte as u8))
        .collect()
}

pub struct MoeExpertSubmission {
    completion: Completion,
    _plan: Arc<PreparedMoeExpertState>,
}

impl MoeExpertSubmission {
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
    pub fn prepare_moe_expert(
        &self,
        context: &Context,
        descriptor: MoeExpertDescriptor,
    ) -> Result<PreparedMoeExpert, RuntimeError> {
        let raw_descriptor = descriptor.raw()?;
        let mut raw = std::ptr::null_mut();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_moe_expert_prepare(
                context.raw_handle()?.as_ptr(),
                &raw_descriptor,
                &mut raw,
                &mut error_sink,
            )
        };
        ensure_ok(status, &buffer, error_sink.message_length)?;
        let raw = NonNull::new(raw).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native MoE expert plan is null",
            )
        })?;
        Ok(PreparedMoeExpert {
            state: Arc::new(PreparedMoeExpertState {
                raw,
                context: context.clone(),
                descriptor,
            }),
        })
    }
}

impl PreparedMoeExpert {
    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<(MoeExpertSubmission, MoeExpertDispatchInfo), RuntimeError> {
        let mut info: sys::sllm_moe_expert_dispatch_info_t = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<sys::sllm_moe_expert_dispatch_info_t>() as u32;
        info.abi_version = sys::SLLM_HIP_ABI_VERSION;
        info.info_version = sys::SLLM_HIP_MOE_EXPERT_DISPATCH_INFO_VERSION;
        let mut completion = std::ptr::null_mut();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_moe_expert_execute(
                self.state.raw.as_ptr(),
                queue.raw_handle()?.as_ptr(),
                &mut completion,
                &mut info,
                &mut error_sink,
            )
        };
        ensure_ok(status, &buffer, error_sink.message_length)?;
        let completion = NonNull::new(completion).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native MoE expert completion is null",
            )
        })?;
        let dispatch = MoeExpertDispatchInfo {
            abi_version: info.abi_version,
            info_version: info.info_version,
            backend: info.backend,
            dispatch_id: info.dispatch_id,
            dispatch_count: info.dispatch_count,
            kernel_id: info.kernel_id,
            workgroup_size_x: info.workgroup_size_x,
            grid_size_x: info.grid_size_x,
            token_count: info.token_count,
            active_pair_count: info.active_pair_count,
            workspace_bytes: info.workspace_bytes,
            selected_expert_count: info.selected_expert_count,
            shared_expert_count: info.shared_expert_count,
            fallback_allowed: info.fallback_allowed != 0,
            fallback_used: info.fallback_used != 0,
            gcn_arch_name: c_string(&info.gcn_arch_name),
            kernel_symbol: c_string(&info.kernel_symbol),
            device_symbol: c_string(&info.device_symbol),
        };
        let completion = Completion::from_native(
            completion,
            &self.state.context,
            queue,
            self.state.descriptor.hidden.buffer(),
            0,
            false,
        );
        Ok((
            MoeExpertSubmission {
                completion,
                _plan: Arc::clone(&self.state),
            },
            dispatch,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{gemma4_moe_expert_workspace_bytes, moe_expert_workspace_bytes};

    #[test]
    fn versioned_workspace_sizes_keep_qwen_v1_unchanged() {
        assert_eq!(moe_expert_workspace_bytes(1), Some(12_484));
        assert_eq!(moe_expert_workspace_bytes(3), Some(37_452));
        assert_eq!(gemma4_moe_expert_workspace_bytes(1), Some(27_104));
        assert_eq!(gemma4_moe_expert_workspace_bytes(3), Some(81_312));
        assert_eq!(
            gemma4_moe_expert_workspace_bytes(65_536),
            Some(1_776_287_744)
        );
    }

    #[test]
    fn versioned_workspace_sizes_reject_overflow() {
        assert_eq!(
            gemma4_moe_expert_workspace_bytes(u64::MAX / 27_104 + 1),
            None
        );
        assert_eq!(moe_expert_workspace_bytes(u64::MAX / 12_484 + 1), None);
    }
}
