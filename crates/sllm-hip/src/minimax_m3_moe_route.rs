//! Safe MiniMax M3 sigmoid top-4 routing over its dedicated public HIP ABI.

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_core::{DType, Encoding, TensorView};
use sllm_hip_sys as sys;

use crate::runtime::{
    Completion, CompletionState, Context, Queue, RuntimeError, RuntimeStatus, ensure_ok, sink,
};
use crate::{HipBackend, TensorBinding};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MiniMaxM3MoeRouteStatus {
    Ok,
    Nonfinite,
    ZeroNormalizer,
}

impl MiniMaxM3MoeRouteStatus {
    pub const fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            sys::SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_OK => Some(Self::Ok),
            sys::SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_NONFINITE => Some(Self::Nonfinite),
            sys::SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_ZERO_NORMALIZER => Some(Self::ZeroNormalizer),
            _ => None,
        }
    }

    pub const fn as_raw(self) -> i32 {
        match self {
            Self::Ok => sys::SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_OK,
            Self::Nonfinite => sys::SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_NONFINITE,
            Self::ZeroNormalizer => sys::SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_ZERO_NORMALIZER,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MiniMaxM3MoeRouteLayout {
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

impl MiniMaxM3MoeRouteLayout {
    pub fn new(token_count: u64) -> Result<Self, RuntimeError> {
        if token_count == 0 || token_count > sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_MAX_TOKENS {
            return Err(RuntimeError::local(
                RuntimeStatus::Unsupported,
                "MiniMax M3 MoE route token count is outside 1..=65536",
            ));
        }
        let selected = u64::from(sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_SELECTED_EXPERT_COUNT);
        let experts = sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_EXPERT_COUNT;
        let pair_count = token_count
            .checked_mul(selected)
            .ok_or_else(layout_overflow)?;
        let word_bytes = |count: u64| count.checked_mul(4).ok_or_else(layout_overflow);
        let ids_offset = 0;
        let weights_offset = word_bytes(pair_count)?;
        let counts_offset = weights_offset
            .checked_add(word_bytes(pair_count)?)
            .ok_or_else(layout_overflow)?;
        let offsets_offset = counts_offset
            .checked_add(word_bytes(experts)?)
            .ok_or_else(layout_overflow)?;
        let grouped_tokens_offset = offsets_offset
            .checked_add(word_bytes(
                experts.checked_add(1).ok_or_else(layout_overflow)?,
            )?)
            .ok_or_else(layout_overflow)?;
        let grouped_slots_offset = grouped_tokens_offset
            .checked_add(word_bytes(pair_count)?)
            .ok_or_else(layout_overflow)?;
        let status_offset = grouped_slots_offset
            .checked_add(word_bytes(pair_count)?)
            .ok_or_else(layout_overflow)?;
        let metadata_bytes = status_offset.checked_add(4).ok_or_else(layout_overflow)?;
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

fn layout_overflow() -> RuntimeError {
    RuntimeError::local(
        RuntimeStatus::MetadataOverflow,
        "MiniMax M3 MoE route metadata layout overflowed",
    )
}

fn exact_contiguous_view(view: &TensorView, dtype: DType, shape: &[usize]) -> bool {
    view.dtype() == dtype
        && view.encoding() == Encoding::Unquantized
        && view.shape() == shape
        && view.is_contiguous()
}

fn validate_logits_view(view: &TensorView) -> Result<u64, RuntimeError> {
    if view.shape().len() != 2
        || view.shape()[1] != sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_EXPERT_COUNT as usize
        || !exact_contiguous_view(view, DType::F32, view.shape())
    {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidTensorBinding,
            "MiniMax M3 MoE route logits must be contiguous unquantized F32 [M,128]",
        ));
    }
    let token_count = u64::try_from(view.shape()[0]).map_err(|_| layout_overflow())?;
    MiniMaxM3MoeRouteLayout::new(token_count)?;
    Ok(token_count)
}

#[derive(Clone, Debug)]
pub struct MiniMaxM3MoeRouteDescriptor {
    logits: TensorBinding,
    selection_bias: TensorBinding,
    metadata: TensorBinding,
    token_count: u64,
    layout: MiniMaxM3MoeRouteLayout,
}

impl MiniMaxM3MoeRouteDescriptor {
    pub fn new(
        logits: TensorBinding,
        selection_bias: TensorBinding,
        metadata: TensorBinding,
    ) -> Result<Self, RuntimeError> {
        let token_count = validate_logits_view(logits.view())?;
        let experts = sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_EXPERT_COUNT as usize;
        if !exact_contiguous_view(selection_bias.view(), DType::F32, &[experts]) {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidTensorBinding,
                "MiniMax M3 MoE route bias must be contiguous unquantized F32 [128]",
            ));
        }
        let layout = MiniMaxM3MoeRouteLayout::new(token_count)?;
        let metadata_bytes =
            usize::try_from(layout.metadata_bytes).map_err(|_| layout_overflow())?;
        if !exact_contiguous_view(metadata.view(), DType::U8, &[metadata_bytes])
            || metadata.view().byte_offset() % 4 != 0
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidTensorBinding,
                "MiniMax M3 MoE route metadata differs from the exact aligned U8 layout",
            ));
        }
        Ok(Self {
            logits,
            selection_bias,
            metadata,
            token_count,
            layout,
        })
    }

    pub const fn token_count(&self) -> u64 {
        self.token_count
    }

    pub const fn layout(&self) -> MiniMaxM3MoeRouteLayout {
        self.layout
    }

    fn raw(&self) -> Result<sys::sllm_minimax_m3_moe_route_desc_t, RuntimeError> {
        Ok(sys::sllm_minimax_m3_moe_route_desc_t {
            struct_size: size_of::<sys::sllm_minimax_m3_moe_route_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_VERSION,
            selected_expert_count: sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_SELECTED_EXPERT_COUNT,
            reserved: [0; 4],
            logits: self.logits.raw()?,
            selection_bias: self.selection_bias.raw()?,
            metadata: self.metadata.raw()?,
        })
    }

    pub fn query(&self) -> Result<MiniMaxM3MoeRouteQueryInfo, RuntimeError> {
        let descriptor = self.raw()?;
        let mut info: sys::sllm_minimax_m3_moe_route_query_info_t = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<sys::sllm_minimax_m3_moe_route_query_info_t>() as u32;
        info.abi_version = sys::SLLM_HIP_ABI_VERSION;
        info.info_version = sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_QUERY_INFO_VERSION;
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_minimax_m3_moe_route_query(&descriptor, &mut info, &mut error_sink)
        };
        ensure_ok(status, &buffer, error_sink.message_length)?;
        let info = MiniMaxM3MoeRouteQueryInfo::from_raw(&info)?;
        if info.token_count != self.token_count
            || info.expert_count != sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_EXPERT_COUNT
            || info.pair_count != self.layout.pair_count
            || info.metadata_bytes != self.layout.metadata_bytes
            || info.selected_expert_count
                != sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_SELECTED_EXPERT_COUNT
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "native MiniMax M3 MoE route query differs from the Rust contract",
            ));
        }
        Ok(info)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MiniMaxM3MoeRouteQueryInfo {
    pub abi_version: u32,
    pub info_version: u32,
    pub token_count: u64,
    pub expert_count: u64,
    pub pair_count: u64,
    pub metadata_bytes: u64,
    pub selected_expert_count: u32,
}

impl MiniMaxM3MoeRouteQueryInfo {
    fn from_raw(info: &sys::sllm_minimax_m3_moe_route_query_info_t) -> Result<Self, RuntimeError> {
        if info.struct_size != size_of::<sys::sllm_minimax_m3_moe_route_query_info_t>() as u32
            || info.abi_version != sys::SLLM_HIP_ABI_VERSION
            || info.info_version != sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_QUERY_INFO_VERSION
            || info.reserved != [0; 8]
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "native MiniMax M3 MoE route query returned an invalid ABI record",
            ));
        }
        Ok(Self {
            abi_version: info.abi_version,
            info_version: info.info_version,
            token_count: info.token_count,
            expert_count: info.expert_count,
            pair_count: info.pair_count,
            metadata_bytes: info.metadata_bytes,
            selected_expert_count: info.selected_expert_count,
        })
    }
}

struct PreparedMiniMaxM3MoeRouteState {
    raw: NonNull<sys::sllm_minimax_m3_moe_route_plan_t>,
    context: Context,
    descriptor: MiniMaxM3MoeRouteDescriptor,
    query: MiniMaxM3MoeRouteQueryInfo,
}

unsafe impl Send for PreparedMiniMaxM3MoeRouteState {}
unsafe impl Sync for PreparedMiniMaxM3MoeRouteState {}

impl Drop for PreparedMiniMaxM3MoeRouteState {
    fn drop(&mut self) {
        let mut raw = self.raw.as_ptr();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status =
            unsafe { sys::sllm_minimax_m3_moe_route_plan_release(&mut raw, &mut error_sink) };
        debug_assert_eq!(status, sys::SLLM_STATUS_OK);
    }
}

#[derive(Clone)]
pub struct PreparedMiniMaxM3MoeRoute {
    state: Arc<PreparedMiniMaxM3MoeRouteState>,
}

impl std::fmt::Debug for PreparedMiniMaxM3MoeRoute {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedMiniMaxM3MoeRoute")
            .field("token_count", &self.state.query.token_count)
            .finish_non_exhaustive()
    }
}

impl HipBackend {
    pub fn prepare_minimax_m3_moe_route(
        &self,
        context: &Context,
        descriptor: MiniMaxM3MoeRouteDescriptor,
    ) -> Result<PreparedMiniMaxM3MoeRoute, RuntimeError> {
        let query = descriptor.query()?;
        let raw_descriptor = descriptor.raw()?;
        let mut raw_plan = std::ptr::null_mut();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_minimax_m3_moe_route_prepare(
                context.raw_handle()?.as_ptr(),
                &raw_descriptor,
                &mut raw_plan,
                &mut error_sink,
            )
        };
        ensure_ok(status, &buffer, error_sink.message_length)?;
        let raw = NonNull::new(raw_plan).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native MiniMax M3 MoE route prepare returned a null plan",
            )
        })?;
        Ok(PreparedMiniMaxM3MoeRoute {
            state: Arc::new(PreparedMiniMaxM3MoeRouteState {
                raw,
                context: context.clone(),
                descriptor,
                query,
            }),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MiniMaxM3MoeRouteDispatchInfo {
    pub abi_version: u32,
    pub info_version: u32,
    pub backend: u32,
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
    pub token_count: u64,
    pub expert_count: u64,
    pub pair_count: u64,
    pub selected_expert_count: u32,
    pub fallback_allowed: bool,
    pub fallback_used: bool,
    pub kernel_symbol: String,
    pub device_symbol: String,
    pub gcn_arch_name: String,
}

impl MiniMaxM3MoeRouteDispatchInfo {
    fn from_raw(
        info: &sys::sllm_minimax_m3_moe_route_dispatch_info_t,
    ) -> Result<Self, RuntimeError> {
        if info.struct_size != size_of::<sys::sllm_minimax_m3_moe_route_dispatch_info_t>() as u32
            || info.abi_version != sys::SLLM_HIP_ABI_VERSION
            || info.info_version != sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_DISPATCH_INFO_VERSION
            || info.fallback_allowed > 1
            || info.fallback_used > 1
            || info.reserved0 != 0
            || info.reserved != [0; 8]
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "native MiniMax M3 MoE route dispatch returned an invalid ABI record",
            ));
        }
        Ok(Self {
            abi_version: info.abi_version,
            info_version: info.info_version,
            backend: info.backend,
            dispatch_id: info.dispatch_id,
            dispatch_count: info.dispatch_count,
            kernel_id: info.kernel_id,
            workgroup_size_x: info.workgroup_size_x,
            grid_size_x: info.grid_size_x,
            token_count: info.token_count,
            expert_count: info.expert_count,
            pair_count: info.pair_count,
            selected_expert_count: info.selected_expert_count,
            fallback_allowed: info.fallback_allowed != 0,
            fallback_used: info.fallback_used != 0,
            kernel_symbol: c_string(&info.kernel_symbol),
            device_symbol: c_string(&info.device_symbol),
            gcn_arch_name: c_string(&info.gcn_arch_name),
        })
    }
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

pub struct MiniMaxM3MoeRouteSubmission {
    completion: Completion,
    _plan: Arc<PreparedMiniMaxM3MoeRouteState>,
}

impl std::fmt::Debug for MiniMaxM3MoeRouteSubmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MiniMaxM3MoeRouteSubmission")
            .finish_non_exhaustive()
    }
}

impl MiniMaxM3MoeRouteSubmission {
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

impl PreparedMiniMaxM3MoeRoute {
    pub fn query_info(&self) -> MiniMaxM3MoeRouteQueryInfo {
        self.state.query
    }

    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<(MiniMaxM3MoeRouteSubmission, MiniMaxM3MoeRouteDispatchInfo), RuntimeError> {
        let mut info: sys::sllm_minimax_m3_moe_route_dispatch_info_t =
            unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<sys::sllm_minimax_m3_moe_route_dispatch_info_t>() as u32;
        info.abi_version = sys::SLLM_HIP_ABI_VERSION;
        info.info_version = sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_DISPATCH_INFO_VERSION;
        let mut raw_completion = std::ptr::null_mut();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_minimax_m3_moe_route_execute(
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
                "native MiniMax M3 MoE route execute returned a null completion",
            )
        })?;
        let completion = Completion::from_native(
            raw_completion,
            &self.state.context,
            queue,
            self.state.descriptor.metadata.buffer(),
            0,
            false,
        );
        let dispatch = MiniMaxM3MoeRouteDispatchInfo::from_raw(&info)?;
        let query = self.state.query;
        if dispatch.backend != sys::SLLM_BACKEND_HIP
            || dispatch.dispatch_count != 2
            || dispatch.kernel_id != sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_KERNEL_ID_SIGMOID_TOP4_V1
            || dispatch.workgroup_size_x != sys::SLLM_HIP_MINIMAX_M3_MOE_ROUTE_WORKGROUP_SIZE
            || u64::from(dispatch.grid_size_x) != query.token_count
            || dispatch.token_count != query.token_count
            || dispatch.expert_count != query.expert_count
            || dispatch.pair_count != query.pair_count
            || dispatch.selected_expert_count != query.selected_expert_count
            || dispatch.fallback_allowed
            || dispatch.fallback_used
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "native MiniMax M3 MoE route dispatch differs from the queried no-fallback contract",
            ));
        }
        Ok((
            MiniMaxM3MoeRouteSubmission {
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
    fn reviewed_m3_layout_is_exact() {
        let layout = MiniMaxM3MoeRouteLayout::new(3).unwrap();
        assert_eq!(layout.pair_count, 12);
        assert_eq!(layout.weights_offset, 48);
        assert_eq!(layout.counts_offset, 96);
        assert_eq!(layout.offsets_offset, 608);
        assert_eq!(layout.grouped_tokens_offset, 1_124);
        assert_eq!(layout.grouped_slots_offset, 1_172);
        assert_eq!(layout.status_offset, 1_220);
        assert_eq!(layout.metadata_bytes, 1_224);
    }

    #[test]
    fn boundaries_and_wrong_views_fail_closed() {
        assert!(MiniMaxM3MoeRouteLayout::new(0).is_err());
        assert!(MiniMaxM3MoeRouteLayout::new(1).is_ok());
        assert!(MiniMaxM3MoeRouteLayout::new(65_536).is_ok());
        assert!(MiniMaxM3MoeRouteLayout::new(65_537).is_err());
        assert!(
            validate_logits_view(&TensorView::contiguous(DType::F32, &[3, 128]).unwrap()).is_ok()
        );
        assert!(
            validate_logits_view(&TensorView::contiguous(DType::Bf16, &[3, 128]).unwrap()).is_err()
        );
        assert!(
            validate_logits_view(&TensorView::contiguous(DType::F32, &[3, 127]).unwrap()).is_err()
        );
    }

    #[test]
    fn status_mapping_rejects_unknown_values() {
        assert_eq!(
            MiniMaxM3MoeRouteStatus::from_raw(0),
            Some(MiniMaxM3MoeRouteStatus::Ok)
        );
        assert_eq!(
            MiniMaxM3MoeRouteStatus::from_raw(2),
            Some(MiniMaxM3MoeRouteStatus::ZeroNormalizer)
        );
        assert_eq!(MiniMaxM3MoeRouteStatus::from_raw(3), None);
    }
}
