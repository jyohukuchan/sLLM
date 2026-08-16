//! Safe deterministic sparse-MoE routing over the public HIP ABI.

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_core::{DType, Encoding};
use sllm_hip_sys as sys;

use crate::runtime::{
    Completion, CompletionState, Context, Queue, RuntimeError, RuntimeStatus, ensure_ok, sink,
};
use crate::{HipBackend, TensorBinding};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoeRouteLayout {
    pub pair_count: u64,
    pub ids_offset: u64,
    pub weights_offset: u64,
    pub counts_offset: u64,
    pub offsets_offset: u64,
    pub grouped_tokens_offset: u64,
    pub grouped_slots_offset: u64,
    pub status_offset: u64,
    pub metadata_bytes: u64,
}

impl MoeRouteLayout {
    pub fn new(
        token_count: u64,
        expert_count: u64,
        selected_expert_count: u32,
    ) -> Result<Self, RuntimeError> {
        if token_count == 0
            || token_count > sys::SLLM_HIP_MOE_ROUTE_MAX_TOKENS
            || expert_count == 0
            || expert_count > sys::SLLM_HIP_MOE_ROUTE_MAX_EXPERTS
            || selected_expert_count == 0
            || selected_expert_count > sys::SLLM_HIP_MOE_ROUTE_MAX_SELECTED
            || u64::from(selected_expert_count) > expert_count
        {
            return Err(RuntimeError::local(
                RuntimeStatus::Unsupported,
                "sparse MoE routing dimensions exceed the public HIP contract",
            ));
        }
        let pair_count = token_count
            .checked_mul(u64::from(selected_expert_count))
            .ok_or_else(|| {
                RuntimeError::local(RuntimeStatus::InvalidArgument, "MoE pair count overflow")
            })?;
        let bytes = |count: u64| {
            count.checked_mul(4).ok_or_else(|| {
                RuntimeError::local(RuntimeStatus::InvalidArgument, "MoE layout overflow")
            })
        };
        let ids_offset = 0;
        let weights_offset = bytes(pair_count)?;
        let counts_offset = weights_offset
            .checked_add(bytes(pair_count)?)
            .ok_or_else(|| {
                RuntimeError::local(RuntimeStatus::InvalidArgument, "MoE layout overflow")
            })?;
        let offsets_offset = counts_offset
            .checked_add(bytes(expert_count)?)
            .ok_or_else(|| {
                RuntimeError::local(RuntimeStatus::InvalidArgument, "MoE layout overflow")
            })?;
        let grouped_tokens_offset = offsets_offset
            .checked_add(bytes(expert_count + 1)?)
            .ok_or_else(|| {
                RuntimeError::local(RuntimeStatus::InvalidArgument, "MoE layout overflow")
            })?;
        let grouped_slots_offset = grouped_tokens_offset
            .checked_add(bytes(pair_count)?)
            .ok_or_else(|| {
                RuntimeError::local(RuntimeStatus::InvalidArgument, "MoE layout overflow")
            })?;
        let status_offset = grouped_slots_offset
            .checked_add(bytes(pair_count)?)
            .ok_or_else(|| {
                RuntimeError::local(RuntimeStatus::InvalidArgument, "MoE layout overflow")
            })?;
        let metadata_bytes = status_offset.checked_add(4).ok_or_else(|| {
            RuntimeError::local(RuntimeStatus::InvalidArgument, "MoE layout overflow")
        })?;
        Ok(Self {
            pair_count,
            ids_offset,
            weights_offset,
            counts_offset,
            offsets_offset,
            grouped_tokens_offset,
            grouped_slots_offset,
            status_offset,
            metadata_bytes,
        })
    }
}

#[derive(Clone, Debug)]
pub struct MoeRouteDescriptor {
    logits: TensorBinding,
    metadata: TensorBinding,
    selected_expert_count: u32,
    layout: MoeRouteLayout,
}

impl MoeRouteDescriptor {
    pub fn new(
        logits: TensorBinding,
        metadata: TensorBinding,
        selected_expert_count: u32,
    ) -> Result<Self, RuntimeError> {
        let logits_view = logits.view();
        if logits_view.shape().len() != 2
            || logits_view.dtype() != DType::Bf16
            || logits_view.encoding() != Encoding::Unquantized
            || !logits_view.is_contiguous()
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                "MoE route logits must be contiguous unquantized BF16 [M,E]",
            ));
        }
        let layout = MoeRouteLayout::new(
            logits_view.shape()[0] as u64,
            logits_view.shape()[1] as u64,
            selected_expert_count,
        )?;
        let metadata_view = metadata.view();
        if metadata_view.shape() != [layout.metadata_bytes as usize]
            || metadata_view.dtype() != DType::U8
            || metadata_view.encoding() != Encoding::Unquantized
            || !metadata_view.is_contiguous()
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                "MoE route metadata binding differs from the reviewed byte layout",
            ));
        }
        Ok(Self {
            logits,
            metadata,
            selected_expert_count,
            layout,
        })
    }

    pub const fn layout(&self) -> MoeRouteLayout {
        self.layout
    }

    fn raw(&self) -> Result<sys::sllm_moe_route_desc_t, RuntimeError> {
        Ok(sys::sllm_moe_route_desc_t {
            struct_size: size_of::<sys::sllm_moe_route_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_MOE_ROUTE_VERSION,
            selected_expert_count: self.selected_expert_count,
            reserved: [0; 4],
            logits: self.logits.raw()?,
            metadata: self.metadata.raw()?,
        })
    }
}

struct PreparedMoeRouteState {
    raw: NonNull<sys::sllm_moe_route_plan_t>,
    context: Context,
    descriptor: MoeRouteDescriptor,
}

unsafe impl Send for PreparedMoeRouteState {}
unsafe impl Sync for PreparedMoeRouteState {}

impl Drop for PreparedMoeRouteState {
    fn drop(&mut self) {
        let mut raw = self.raw.as_ptr();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe { sys::sllm_moe_route_plan_release(&mut raw, &mut error_sink) };
        debug_assert_eq!(status, sys::SLLM_STATUS_OK);
    }
}

#[derive(Clone)]
pub struct PreparedMoeRoute {
    state: Arc<PreparedMoeRouteState>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MoeRouteDispatchInfo {
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub token_count: u64,
    pub expert_count: u64,
    pub pair_count: u64,
    pub selected_expert_count: u32,
    pub fallback_allowed: bool,
    pub fallback_used: bool,
    pub gcn_arch_name: String,
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

pub struct MoeRouteSubmission {
    completion: Completion,
    _plan: Arc<PreparedMoeRouteState>,
}

impl MoeRouteSubmission {
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
    pub fn prepare_moe_route(
        &self,
        context: &Context,
        descriptor: MoeRouteDescriptor,
    ) -> Result<PreparedMoeRoute, RuntimeError> {
        let raw_descriptor = descriptor.raw()?;
        let mut raw = std::ptr::null_mut();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_moe_route_prepare(
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
                "native MoE route prepare returned a null plan",
            )
        })?;
        Ok(PreparedMoeRoute {
            state: Arc::new(PreparedMoeRouteState {
                raw,
                context: context.clone(),
                descriptor,
            }),
        })
    }
}

impl PreparedMoeRoute {
    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<(MoeRouteSubmission, MoeRouteDispatchInfo), RuntimeError> {
        let mut info: sys::sllm_moe_route_dispatch_info_t = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<sys::sllm_moe_route_dispatch_info_t>() as u32;
        info.abi_version = sys::SLLM_HIP_ABI_VERSION;
        info.info_version = sys::SLLM_HIP_MOE_ROUTE_DISPATCH_INFO_VERSION;
        let mut raw_completion = std::ptr::null_mut();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_moe_route_execute(
                self.state.raw.as_ptr(),
                queue.raw_handle()?.as_ptr(),
                &mut raw_completion,
                &mut info,
                &mut error_sink,
            )
        };
        ensure_ok(status, &buffer, error_sink.message_length)?;
        let raw_completion = NonNull::new(raw_completion).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native MoE route execute returned a null completion",
            )
        })?;
        let dispatch = MoeRouteDispatchInfo {
            dispatch_id: info.dispatch_id,
            dispatch_count: info.dispatch_count,
            kernel_id: info.kernel_id,
            token_count: info.token_count,
            expert_count: info.expert_count,
            pair_count: info.pair_count,
            selected_expert_count: info.selected_expert_count,
            fallback_allowed: info.fallback_allowed != 0,
            fallback_used: info.fallback_used != 0,
            gcn_arch_name: c_string(&info.gcn_arch_name),
        };
        let completion = Completion::from_native(
            raw_completion,
            &self.state.context,
            queue,
            self.state.descriptor.logits.buffer(),
            0,
            false,
        );
        Ok((
            MoeRouteSubmission {
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

    #[test]
    fn reviewed_layout_has_expected_offsets() {
        let layout = MoeRouteLayout::new(3, 256, 8).unwrap();
        assert_eq!(layout.pair_count, 24);
        assert_eq!(layout.ids_offset, 0);
        assert_eq!(layout.weights_offset, 96);
        assert_eq!(layout.counts_offset, 192);
        assert_eq!(layout.offsets_offset, 1216);
        assert_eq!(layout.grouped_tokens_offset, 2244);
        assert_eq!(layout.grouped_slots_offset, 2340);
        assert_eq!(layout.status_offset, 2436);
        assert_eq!(layout.metadata_bytes, 2440);
    }
}
