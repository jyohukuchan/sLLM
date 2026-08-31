//! Safe DeepSeek V4 score/hash MoE routing over its dedicated public HIP ABI.

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
pub enum DeepSeekV4MoeRouteMode {
    Score,
    Hash,
}

impl DeepSeekV4MoeRouteMode {
    pub fn from_raw(raw: sys::sllm_deepseek_v4_moe_route_mode_t) -> Result<Self, RuntimeError> {
        match raw {
            sys::SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE => Ok(Self::Score),
            sys::SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_HASH => Ok(Self::Hash),
            _ => Err(RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                "DeepSeek V4 MoE route mode is unsupported",
            )),
        }
    }

    pub const fn as_raw(self) -> sys::sllm_deepseek_v4_moe_route_mode_t {
        match self {
            Self::Score => sys::SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE,
            Self::Hash => sys::SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_HASH,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepSeekV4MoeRouteStatus {
    Ok,
    Nonfinite,
    ExpertOutOfRange,
    DuplicateExpert,
    ZeroNormalizer,
}

impl DeepSeekV4MoeRouteStatus {
    pub const fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            sys::SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_OK => Some(Self::Ok),
            sys::SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_NONFINITE => Some(Self::Nonfinite),
            sys::SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_EXPERT_OUT_OF_RANGE => {
                Some(Self::ExpertOutOfRange)
            }
            sys::SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_DUPLICATE_EXPERT => Some(Self::DuplicateExpert),
            sys::SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_ZERO_NORMALIZER => Some(Self::ZeroNormalizer),
            _ => None,
        }
    }

    pub const fn as_raw(self) -> i32 {
        match self {
            Self::Ok => sys::SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_OK,
            Self::Nonfinite => sys::SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_NONFINITE,
            Self::ExpertOutOfRange => sys::SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_EXPERT_OUT_OF_RANGE,
            Self::DuplicateExpert => sys::SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_DUPLICATE_EXPERT,
            Self::ZeroNormalizer => sys::SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_ZERO_NORMALIZER,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MoeRouteLayout {
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

impl DeepSeekV4MoeRouteLayout {
    pub fn new(token_count: u64) -> Result<Self, RuntimeError> {
        if token_count == 0 || token_count > sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_MAX_TOKENS {
            return Err(RuntimeError::local(
                RuntimeStatus::Unsupported,
                "DeepSeek V4 MoE route token count is outside 1..=65536",
            ));
        }
        let selected = u64::from(sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_SELECTED_EXPERT_COUNT);
        let experts = sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_EXPERT_COUNT;
        let pair_count = token_count
            .checked_mul(selected)
            .ok_or_else(layout_overflow)?;
        let i32_bytes = |count: u64| count.checked_mul(4).ok_or_else(layout_overflow);
        let ids_offset = 0;
        let weights_offset = i32_bytes(pair_count)?;
        let counts_offset = weights_offset
            .checked_add(i32_bytes(pair_count)?)
            .ok_or_else(layout_overflow)?;
        let offsets_offset = counts_offset
            .checked_add(i32_bytes(experts)?)
            .ok_or_else(layout_overflow)?;
        let grouped_tokens_offset = offsets_offset
            .checked_add(i32_bytes(
                experts.checked_add(1).ok_or_else(layout_overflow)?,
            )?)
            .ok_or_else(layout_overflow)?;
        let grouped_slots_offset = grouped_tokens_offset
            .checked_add(i32_bytes(pair_count)?)
            .ok_or_else(layout_overflow)?;
        let status_offset = grouped_slots_offset
            .checked_add(i32_bytes(pair_count)?)
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
        "DeepSeek V4 MoE route metadata layout overflowed",
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
        || view.shape()[1] != sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_EXPERT_COUNT as usize
        || !exact_contiguous_view(view, DType::Bf16, view.shape())
    {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidTensorBinding,
            "DeepSeek V4 MoE route logits must be contiguous unquantized BF16 [M,256]",
        ));
    }
    let token_count = u64::try_from(view.shape()[0]).map_err(|_| {
        RuntimeError::local(
            RuntimeStatus::MetadataOverflow,
            "DeepSeek V4 MoE route token count does not fit u64",
        )
    })?;
    DeepSeekV4MoeRouteLayout::new(token_count)?;
    Ok(token_count)
}

fn validate_scale(routed_scale: f32) -> Result<(), RuntimeError> {
    if !routed_scale.is_finite() || routed_scale <= 0.0 {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidArgument,
            "DeepSeek V4 MoE routed scale must be finite and strictly positive",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
enum DeepSeekV4MoeRouteInput {
    Score { selection_bias: TensorBinding },
    Hash { hash_expert_ids: TensorBinding },
}

impl DeepSeekV4MoeRouteInput {
    const fn mode(&self) -> DeepSeekV4MoeRouteMode {
        match self {
            Self::Score { .. } => DeepSeekV4MoeRouteMode::Score,
            Self::Hash { .. } => DeepSeekV4MoeRouteMode::Hash,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeepSeekV4MoeRouteDescriptor {
    logits: TensorBinding,
    input: DeepSeekV4MoeRouteInput,
    metadata: TensorBinding,
    renormalize: bool,
    routed_scale: f32,
    token_count: u64,
    layout: DeepSeekV4MoeRouteLayout,
}

impl DeepSeekV4MoeRouteDescriptor {
    pub fn new_score(
        logits: TensorBinding,
        selection_bias: TensorBinding,
        metadata: TensorBinding,
        renormalize: bool,
        routed_scale: f32,
    ) -> Result<Self, RuntimeError> {
        let token_count = validate_logits_view(logits.view())?;
        let expert_count = sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_EXPERT_COUNT as usize;
        if !exact_contiguous_view(selection_bias.view(), DType::F32, &[expert_count]) {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidTensorBinding,
                "DeepSeek V4 score route bias must be contiguous unquantized F32 [256]",
            ));
        }
        Self::finish(
            logits,
            DeepSeekV4MoeRouteInput::Score { selection_bias },
            metadata,
            renormalize,
            routed_scale,
            token_count,
        )
    }

    pub fn new_hash(
        logits: TensorBinding,
        hash_expert_ids: TensorBinding,
        metadata: TensorBinding,
        renormalize: bool,
        routed_scale: f32,
    ) -> Result<Self, RuntimeError> {
        let token_count = validate_logits_view(logits.view())?;
        let token_count_usize = usize::try_from(token_count).map_err(|_| {
            RuntimeError::local(
                RuntimeStatus::MetadataOverflow,
                "DeepSeek V4 MoE route token count does not fit usize",
            )
        })?;
        let selected = sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_SELECTED_EXPERT_COUNT as usize;
        if !exact_contiguous_view(
            hash_expert_ids.view(),
            DType::I32,
            &[token_count_usize, selected],
        ) {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidTensorBinding,
                "DeepSeek V4 hash route IDs must be contiguous unquantized I32 [M,6]",
            ));
        }
        Self::finish(
            logits,
            DeepSeekV4MoeRouteInput::Hash { hash_expert_ids },
            metadata,
            renormalize,
            routed_scale,
            token_count,
        )
    }

    fn finish(
        logits: TensorBinding,
        input: DeepSeekV4MoeRouteInput,
        metadata: TensorBinding,
        renormalize: bool,
        routed_scale: f32,
        token_count: u64,
    ) -> Result<Self, RuntimeError> {
        validate_scale(routed_scale)?;
        let layout = DeepSeekV4MoeRouteLayout::new(token_count)?;
        let metadata_bytes = usize::try_from(layout.metadata_bytes).map_err(|_| {
            RuntimeError::local(
                RuntimeStatus::MetadataOverflow,
                "DeepSeek V4 MoE route metadata size does not fit usize",
            )
        })?;
        if !exact_contiguous_view(metadata.view(), DType::U8, &[metadata_bytes])
            || metadata.view().byte_offset() % 4 != 0
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidTensorBinding,
                "DeepSeek V4 MoE route metadata differs from the exact aligned U8 layout",
            ));
        }
        Ok(Self {
            logits,
            input,
            metadata,
            renormalize,
            routed_scale,
            token_count,
            layout,
        })
    }

    pub const fn mode(&self) -> DeepSeekV4MoeRouteMode {
        self.input.mode()
    }

    pub const fn token_count(&self) -> u64 {
        self.token_count
    }

    pub const fn layout(&self) -> DeepSeekV4MoeRouteLayout {
        self.layout
    }

    pub const fn renormalize(&self) -> bool {
        self.renormalize
    }

    pub const fn routed_scale(&self) -> f32 {
        self.routed_scale
    }

    fn raw(&self) -> Result<sys::sllm_deepseek_v4_moe_route_desc_t, RuntimeError> {
        let inactive = zero_tensor_binding();
        let (selection_bias, hash_expert_ids) = match &self.input {
            DeepSeekV4MoeRouteInput::Score { selection_bias } => (selection_bias.raw()?, inactive),
            DeepSeekV4MoeRouteInput::Hash { hash_expert_ids } => (inactive, hash_expert_ids.raw()?),
        };
        Ok(sys::sllm_deepseek_v4_moe_route_desc_t {
            struct_size: size_of::<sys::sllm_deepseek_v4_moe_route_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_VERSION,
            mode: self.mode().as_raw(),
            selected_expert_count: sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_SELECTED_EXPERT_COUNT,
            renormalize: u32::from(self.renormalize),
            routed_scale: self.routed_scale,
            reserved0: 0,
            reserved: [0; 4],
            logits: self.logits.raw()?,
            selection_bias,
            hash_expert_ids,
            metadata: self.metadata.raw()?,
        })
    }

    pub fn query(&self) -> Result<DeepSeekV4MoeRouteQueryInfo, RuntimeError> {
        let descriptor = self.raw()?;
        let mut info: sys::sllm_deepseek_v4_moe_route_query_info_t = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<sys::sllm_deepseek_v4_moe_route_query_info_t>() as u32;
        info.abi_version = sys::SLLM_HIP_ABI_VERSION;
        info.info_version = sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_QUERY_INFO_VERSION;
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_deepseek_v4_moe_route_query(&descriptor, &mut info, &mut error_sink)
        };
        ensure_ok(status, &buffer, error_sink.message_length)?;
        let info = DeepSeekV4MoeRouteQueryInfo::from_raw(&info)?;
        self.ensure_query_matches(&info)?;
        Ok(info)
    }

    fn ensure_query_matches(&self, info: &DeepSeekV4MoeRouteQueryInfo) -> Result<(), RuntimeError> {
        if info.abi_version != sys::SLLM_HIP_ABI_VERSION
            || info.info_version != sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_QUERY_INFO_VERSION
            || info.mode != self.mode()
            || info.token_count != self.token_count
            || info.expert_count != sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_EXPERT_COUNT
            || info.pair_count != self.layout.pair_count
            || info.metadata_bytes != self.layout.metadata_bytes
            || info.selected_expert_count
                != sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_SELECTED_EXPERT_COUNT
            || info.renormalize != self.renormalize
            || info.routed_scale.to_bits() != self.routed_scale.to_bits()
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "native DeepSeek V4 MoE route query differs from the Rust contract",
            ));
        }
        Ok(())
    }
}

fn zero_tensor_binding() -> sys::sllm_tensor_binding_t {
    // The dedicated ABI requires every byte of the inactive binding to be zero.
    unsafe { std::mem::zeroed() }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepSeekV4MoeRouteQueryInfo {
    pub abi_version: u32,
    pub info_version: u32,
    pub mode: DeepSeekV4MoeRouteMode,
    pub token_count: u64,
    pub expert_count: u64,
    pub pair_count: u64,
    pub metadata_bytes: u64,
    pub selected_expert_count: u32,
    pub renormalize: bool,
    pub routed_scale: f32,
}

impl DeepSeekV4MoeRouteQueryInfo {
    fn from_raw(info: &sys::sllm_deepseek_v4_moe_route_query_info_t) -> Result<Self, RuntimeError> {
        if info.struct_size != size_of::<sys::sllm_deepseek_v4_moe_route_query_info_t>() as u32
            || info.abi_version != sys::SLLM_HIP_ABI_VERSION
            || info.info_version != sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_QUERY_INFO_VERSION
            || info.renormalize > 1
            || info.reserved0 != 0
            || info.reserved != [0; 8]
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "native DeepSeek V4 MoE route query returned an invalid ABI record",
            ));
        }
        Ok(Self {
            abi_version: info.abi_version,
            info_version: info.info_version,
            mode: DeepSeekV4MoeRouteMode::from_raw(info.mode)?,
            token_count: info.token_count,
            expert_count: info.expert_count,
            pair_count: info.pair_count,
            metadata_bytes: info.metadata_bytes,
            selected_expert_count: info.selected_expert_count,
            renormalize: info.renormalize != 0,
            routed_scale: info.routed_scale,
        })
    }
}

struct PreparedDeepSeekV4MoeRouteState {
    raw: NonNull<sys::sllm_deepseek_v4_moe_route_plan_t>,
    context: Context,
    descriptor: DeepSeekV4MoeRouteDescriptor,
    query: DeepSeekV4MoeRouteQueryInfo,
}

unsafe impl Send for PreparedDeepSeekV4MoeRouteState {}
unsafe impl Sync for PreparedDeepSeekV4MoeRouteState {}

impl Drop for PreparedDeepSeekV4MoeRouteState {
    fn drop(&mut self) {
        let mut raw = self.raw.as_ptr();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status =
            unsafe { sys::sllm_deepseek_v4_moe_route_plan_release(&mut raw, &mut error_sink) };
        debug_assert_eq!(status, sys::SLLM_STATUS_OK);
    }
}

#[derive(Clone)]
pub struct PreparedDeepSeekV4MoeRoute {
    state: Arc<PreparedDeepSeekV4MoeRouteState>,
}

impl std::fmt::Debug for PreparedDeepSeekV4MoeRoute {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedDeepSeekV4MoeRoute")
            .field("mode", &self.state.query.mode)
            .field("token_count", &self.state.query.token_count)
            .finish_non_exhaustive()
    }
}

impl HipBackend {
    pub fn prepare_deepseek_v4_moe_route(
        &self,
        context: &Context,
        descriptor: DeepSeekV4MoeRouteDescriptor,
    ) -> Result<PreparedDeepSeekV4MoeRoute, RuntimeError> {
        let query = descriptor.query()?;
        let raw_descriptor = descriptor.raw()?;
        let mut raw_plan = std::ptr::null_mut();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_deepseek_v4_moe_route_prepare(
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
                "native DeepSeek V4 MoE route prepare returned a null plan",
            )
        })?;
        Ok(PreparedDeepSeekV4MoeRoute {
            state: Arc::new(PreparedDeepSeekV4MoeRouteState {
                raw,
                context: context.clone(),
                descriptor,
                query,
            }),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MoeRouteDispatchInfo {
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
    pub mode: DeepSeekV4MoeRouteMode,
    pub renormalize: bool,
    pub fallback_allowed: bool,
    pub fallback_used: bool,
    pub kernel_symbol: String,
    pub device_symbol: String,
    pub gcn_arch_name: String,
}

impl DeepSeekV4MoeRouteDispatchInfo {
    fn from_raw(
        info: &sys::sllm_deepseek_v4_moe_route_dispatch_info_t,
    ) -> Result<Self, RuntimeError> {
        if info.struct_size != size_of::<sys::sllm_deepseek_v4_moe_route_dispatch_info_t>() as u32
            || info.abi_version != sys::SLLM_HIP_ABI_VERSION
            || info.info_version != sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_DISPATCH_INFO_VERSION
            || info.renormalize > 1
            || info.fallback_allowed > 1
            || info.fallback_used > 1
            || info.reserved0 != 0
            || info.reserved != [0; 8]
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "native DeepSeek V4 MoE route dispatch returned an invalid ABI record",
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
            mode: DeepSeekV4MoeRouteMode::from_raw(info.mode)?,
            renormalize: info.renormalize != 0,
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

pub struct DeepSeekV4MoeRouteSubmission {
    completion: Completion,
    _plan: Arc<PreparedDeepSeekV4MoeRouteState>,
}

impl std::fmt::Debug for DeepSeekV4MoeRouteSubmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeepSeekV4MoeRouteSubmission")
            .finish_non_exhaustive()
    }
}

impl DeepSeekV4MoeRouteSubmission {
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

impl PreparedDeepSeekV4MoeRoute {
    pub fn query_info(&self) -> DeepSeekV4MoeRouteQueryInfo {
        self.state.query
    }

    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<(DeepSeekV4MoeRouteSubmission, DeepSeekV4MoeRouteDispatchInfo), RuntimeError> {
        let mut info: sys::sllm_deepseek_v4_moe_route_dispatch_info_t =
            unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<sys::sllm_deepseek_v4_moe_route_dispatch_info_t>() as u32;
        info.abi_version = sys::SLLM_HIP_ABI_VERSION;
        info.info_version = sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_DISPATCH_INFO_VERSION;
        let mut raw_completion = std::ptr::null_mut();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_deepseek_v4_moe_route_execute(
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
                "native DeepSeek V4 MoE route execute returned a null completion",
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
        let dispatch = DeepSeekV4MoeRouteDispatchInfo::from_raw(&info)?;
        self.ensure_dispatch_matches(&dispatch)?;
        Ok((
            DeepSeekV4MoeRouteSubmission {
                completion,
                _plan: Arc::clone(&self.state),
            },
            dispatch,
        ))
    }

    fn ensure_dispatch_matches(
        &self,
        info: &DeepSeekV4MoeRouteDispatchInfo,
    ) -> Result<(), RuntimeError> {
        let query = self.state.query;
        let expected_kernel = match query.mode {
            DeepSeekV4MoeRouteMode::Score => sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_KERNEL_ID_SCORE_V1,
            DeepSeekV4MoeRouteMode::Hash => sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_KERNEL_ID_HASH_V1,
        };
        if info.abi_version != sys::SLLM_HIP_ABI_VERSION
            || info.info_version != sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_DISPATCH_INFO_VERSION
            || info.backend != sys::SLLM_BACKEND_HIP
            || info.dispatch_count != 2
            || info.kernel_id != expected_kernel
            || info.workgroup_size_x != sys::SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_WORKGROUP_SIZE
            || u64::from(info.grid_size_x) != query.token_count
            || info.token_count != query.token_count
            || info.expert_count != query.expert_count
            || info.pair_count != query.pair_count
            || info.selected_expert_count != query.selected_expert_count
            || info.mode != query.mode
            || info.renormalize != query.renormalize
            || info.fallback_allowed
            || info.fallback_used
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "native DeepSeek V4 MoE route dispatch differs from the queried no-fallback contract",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, offset_of, size_of};

    use sllm_core::{DType, TensorView};
    use sllm_hip_sys as sys;

    use crate::{Buffer, runtime::Context};

    use super::{
        DeepSeekV4MoeRouteDescriptor, DeepSeekV4MoeRouteInput, DeepSeekV4MoeRouteLayout,
        DeepSeekV4MoeRouteMode, DeepSeekV4MoeRouteStatus, validate_logits_view, validate_scale,
        zero_tensor_binding,
    };

    #[test]
    fn reviewed_m3_layout_is_exactly_2344_bytes() {
        let layout = DeepSeekV4MoeRouteLayout::new(3).unwrap();
        assert_eq!(layout.pair_count, 18);
        assert_eq!(layout.ids_offset, 0);
        assert_eq!(layout.weights_offset, 72);
        assert_eq!(layout.counts_offset, 144);
        assert_eq!(layout.offsets_offset, 1_168);
        assert_eq!(layout.grouped_tokens_offset, 2_196);
        assert_eq!(layout.grouped_slots_offset, 2_268);
        assert_eq!(layout.status_offset, 2_340);
        assert_eq!(layout.metadata_bytes, 2_344);
    }

    #[test]
    fn token_count_boundaries_fail_closed() {
        assert!(DeepSeekV4MoeRouteLayout::new(0).is_err());
        assert!(DeepSeekV4MoeRouteLayout::new(1).is_ok());
        assert!(DeepSeekV4MoeRouteLayout::new(65_536).is_ok());
        assert!(DeepSeekV4MoeRouteLayout::new(65_537).is_err());
    }

    #[test]
    fn invalid_scale_shape_and_mode_are_rejected() {
        for scale in [0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            assert!(validate_scale(scale).is_err());
        }
        assert!(validate_scale(1.5).is_ok());

        let valid = TensorView::contiguous(DType::Bf16, &[3, 256]).unwrap();
        assert_eq!(validate_logits_view(&valid).unwrap(), 3);
        for shape in [&[3, 255][..], &[3, 256, 1][..], &[0, 256][..]] {
            let invalid = TensorView::contiguous(DType::Bf16, shape).unwrap();
            assert!(validate_logits_view(&invalid).is_err());
        }
        let wrong_dtype = TensorView::contiguous(DType::F32, &[3, 256]).unwrap();
        assert!(validate_logits_view(&wrong_dtype).is_err());
        let strided = TensorView::new(
            DType::Bf16,
            sllm_core::Encoding::Unquantized,
            &[3, 256],
            &[257, 1],
            0,
        )
        .unwrap();
        assert!(validate_logits_view(&strided).is_err());

        assert!(DeepSeekV4MoeRouteMode::from_raw(0).is_err());
        assert!(DeepSeekV4MoeRouteMode::from_raw(3).is_err());
        assert_eq!(
            DeepSeekV4MoeRouteMode::from_raw(sys::SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE).unwrap(),
            DeepSeekV4MoeRouteMode::Score
        );
    }

    #[test]
    fn constructors_retain_only_the_mode_active_binding() {
        let context = Context::test_without_native();
        let buffer = Buffer::test_without_native(&context);
        let binding =
            |dtype, shape: &[usize]| buffer.binding(TensorView::contiguous(dtype, shape).unwrap());
        let logits = binding(DType::Bf16, &[3, 256]);
        let metadata = binding(DType::U8, &[2_344]);

        let score = DeepSeekV4MoeRouteDescriptor::new_score(
            logits.clone(),
            binding(DType::F32, &[256]),
            metadata.clone(),
            true,
            1.5,
        )
        .unwrap();
        assert_eq!(score.mode(), DeepSeekV4MoeRouteMode::Score);
        assert!(matches!(score.input, DeepSeekV4MoeRouteInput::Score { .. }));

        let hash = DeepSeekV4MoeRouteDescriptor::new_hash(
            logits.clone(),
            binding(DType::I32, &[3, 6]),
            metadata.clone(),
            false,
            1.25,
        )
        .unwrap();
        assert_eq!(hash.mode(), DeepSeekV4MoeRouteMode::Hash);
        assert!(matches!(hash.input, DeepSeekV4MoeRouteInput::Hash { .. }));

        assert!(
            DeepSeekV4MoeRouteDescriptor::new_score(
                logits.clone(),
                binding(DType::F32, &[255]),
                metadata.clone(),
                true,
                1.5,
            )
            .is_err()
        );
        assert!(
            DeepSeekV4MoeRouteDescriptor::new_hash(
                logits.clone(),
                binding(DType::I32, &[3, 5]),
                metadata.clone(),
                true,
                1.5,
            )
            .is_err()
        );
        assert!(
            DeepSeekV4MoeRouteDescriptor::new_score(
                logits,
                binding(DType::F32, &[256]),
                metadata,
                true,
                f32::NAN,
            )
            .is_err()
        );
    }

    #[test]
    fn inactive_binding_is_all_zero() {
        let binding = zero_tensor_binding();
        assert_eq!(binding.struct_size, 0);
        assert_eq!(binding.abi_version, 0);
        assert!(binding.buffer.is_null());
        assert_eq!(binding.byte_offset, 0);
        assert_eq!(binding.dtype, 0);
        assert_eq!(binding.encoding, 0);
        assert_eq!(binding.rank, 0);
        assert_eq!(binding.reserved0, 0);
        assert_eq!(binding.shape, [0; 8]);
        assert_eq!(binding.stride_elements, [0; 8]);
        assert_eq!(binding.reserved, [0; 2]);
    }

    #[test]
    fn status_values_round_trip() {
        for status in [
            DeepSeekV4MoeRouteStatus::Ok,
            DeepSeekV4MoeRouteStatus::Nonfinite,
            DeepSeekV4MoeRouteStatus::ExpertOutOfRange,
            DeepSeekV4MoeRouteStatus::DuplicateExpert,
            DeepSeekV4MoeRouteStatus::ZeroNormalizer,
        ] {
            assert_eq!(
                DeepSeekV4MoeRouteStatus::from_raw(status.as_raw()),
                Some(status)
            );
        }
        assert_eq!(DeepSeekV4MoeRouteStatus::from_raw(5), None);
    }

    #[test]
    fn dedicated_abi_layout_matches_the_c_header() {
        assert_eq!(size_of::<sys::sllm_tensor_binding_t>(), 184);
        assert_eq!(align_of::<sys::sllm_tensor_binding_t>(), 8);
        assert_eq!(size_of::<sys::sllm_deepseek_v4_moe_route_desc_t>(), 784);
        assert_eq!(align_of::<sys::sllm_deepseek_v4_moe_route_desc_t>(), 8);
        assert_eq!(
            offset_of!(sys::sllm_deepseek_v4_moe_route_desc_t, logits),
            48
        );
        assert_eq!(
            offset_of!(sys::sllm_deepseek_v4_moe_route_desc_t, selection_bias),
            232
        );
        assert_eq!(
            offset_of!(sys::sllm_deepseek_v4_moe_route_desc_t, hash_expert_ids),
            416
        );
        assert_eq!(
            offset_of!(sys::sllm_deepseek_v4_moe_route_desc_t, metadata),
            600
        );
        assert_eq!(
            size_of::<sys::sllm_deepseek_v4_moe_route_query_info_t>(),
            96
        );
        assert_eq!(
            align_of::<sys::sllm_deepseek_v4_moe_route_query_info_t>(),
            8
        );
        assert_eq!(
            offset_of!(sys::sllm_deepseek_v4_moe_route_query_info_t, token_count),
            16
        );
        assert_eq!(
            size_of::<sys::sllm_deepseek_v4_moe_route_dispatch_info_t>(),
            312
        );
        assert_eq!(
            align_of::<sys::sllm_deepseek_v4_moe_route_dispatch_info_t>(),
            8
        );
        assert_eq!(
            offset_of!(
                sys::sllm_deepseek_v4_moe_route_dispatch_info_t,
                kernel_symbol
            ),
            88
        );
    }
}
