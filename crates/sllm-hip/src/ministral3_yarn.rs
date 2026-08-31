//! Safe Rust wrapper for the fixed Ministral 3 BF16 YaRN RoPE ABI.

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_hip_sys as sys;

use crate::runtime::{Completion, CompletionState, Context, Queue, RuntimeError, ensure_ok, sink};
use crate::{HipBackend, TensorBinding};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Ministral3YarnPositionMode {
    Contiguous,
    Explicit,
}

#[derive(Clone, Debug)]
pub struct Ministral3YarnDescriptor {
    query: TensorBinding,
    key: TensorBinding,
    positions: TensorBinding,
    query_output: TensorBinding,
    key_output: TensorBinding,
    start_position: u64,
    position_mode: Ministral3YarnPositionMode,
}

impl Ministral3YarnDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        query: TensorBinding,
        key: TensorBinding,
        positions: TensorBinding,
        query_output: TensorBinding,
        key_output: TensorBinding,
        start_position: u64,
        position_mode: Ministral3YarnPositionMode,
    ) -> Result<Self, RuntimeError> {
        if start_position >= u64::from(sys::SLLM_HIP_MINISTRAL3_YARN_MAX_POSITION) {
            return Err(RuntimeError::local(
                crate::RuntimeStatus::InvalidArgument,
                "Ministral3 YaRN start position is outside the fixed context",
            ));
        }
        Ok(Self {
            query,
            key,
            positions,
            query_output,
            key_output,
            start_position,
            position_mode,
        })
    }

    fn raw(&self) -> Result<sys::sllm_ministral3_yarn_desc_t, RuntimeError> {
        Ok(sys::sllm_ministral3_yarn_desc_t {
            struct_size: size_of::<sys::sllm_ministral3_yarn_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_MINISTRAL3_YARN_VERSION,
            position_payload_mode: match self.position_mode {
                Ministral3YarnPositionMode::Contiguous => {
                    sys::SLLM_HIP_POSITION_PAYLOAD_MODE_CONTIGUOUS_V1
                }
                Ministral3YarnPositionMode::Explicit => {
                    sys::SLLM_HIP_POSITION_PAYLOAD_MODE_EXPLICIT_V1
                }
            },
            start_position: self.start_position,
            q_heads: sys::SLLM_HIP_MINISTRAL3_YARN_Q_HEADS,
            kv_heads: sys::SLLM_HIP_MINISTRAL3_YARN_KV_HEADS,
            head_dim: sys::SLLM_HIP_MINISTRAL3_YARN_HEAD_DIM,
            rotary_dim: sys::SLLM_HIP_MINISTRAL3_YARN_ROTARY_DIM,
            theta_bits: 1_000_000.0_f32.to_bits(),
            factor_bits: 16.0_f32.to_bits(),
            original_context: sys::SLLM_HIP_MINISTRAL3_YARN_ORIGINAL_CONTEXT,
            max_position: sys::SLLM_HIP_MINISTRAL3_YARN_MAX_POSITION,
            beta_fast_bits: 32.0_f32.to_bits(),
            beta_slow_bits: 1.0_f32.to_bits(),
            query_scale_beta_bits: 0.1_f32.to_bits(),
            reserved: [0; 5],
            query: self.query.raw()?,
            key: self.key.raw()?,
            positions: self.positions.raw()?,
            query_output: self.query_output.raw()?,
            key_output: self.key_output.raw()?,
        })
    }
}

struct PreparedOwners {
    context: Context,
    descriptor: Ministral3YarnDescriptor,
}

struct PreparedState {
    raw: NonNull<sys::sllm_ministral3_yarn_plan_t>,
    owners: PreparedOwners,
}

// SAFETY: native plan handles are process-local opaque tokens and the owner
// graph is immutable while a prepared operation is shared.
unsafe impl Send for PreparedState {}
unsafe impl Sync for PreparedState {}

impl Drop for PreparedState {
    fn drop(&mut self) {
        let mut raw = self.raw.as_ptr();
        let mut error_bytes = [0_u8; 256];
        let mut error_sink = sink(&mut error_bytes);
        // The owning Arc is retained by each submission, so a plan reaches
        // this path only after all native completions have been finalized.
        let _ = unsafe { sys::sllm_ministral3_yarn_plan_release(&mut raw, &mut error_sink) };
        let _ = &self.owners.context;
    }
}

#[derive(Clone)]
pub struct PreparedMinistral3Yarn {
    state: Arc<PreparedState>,
}

unsafe impl Send for PreparedMinistral3Yarn {}
unsafe impl Sync for PreparedMinistral3Yarn {}

impl std::fmt::Debug for PreparedMinistral3Yarn {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedMinistral3Yarn")
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ministral3YarnDispatchInfo {
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

fn dispatch_info_from_raw(
    info: &sys::sllm_ministral3_yarn_dispatch_info_t,
) -> Ministral3YarnDispatchInfo {
    Ministral3YarnDispatchInfo {
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

pub struct Ministral3YarnSubmission {
    completion: Completion,
    _plan: Arc<PreparedState>,
}

impl std::fmt::Debug for Ministral3YarnSubmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Ministral3YarnSubmission")
            .finish_non_exhaustive()
    }
}

impl Ministral3YarnSubmission {
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
    pub fn prepare_ministral3_yarn(
        &self,
        context: &Context,
        descriptor: Ministral3YarnDescriptor,
    ) -> Result<PreparedMinistral3Yarn, RuntimeError> {
        let raw_descriptor = descriptor.raw()?;
        let mut raw_plan = std::ptr::null_mut();
        let mut error_bytes = [0_u8; 256];
        let mut error_sink = sink(&mut error_bytes);
        let status = unsafe {
            sys::sllm_ministral3_yarn_prepare(
                context.raw_handle()?.as_ptr(),
                &raw_descriptor,
                &mut raw_plan,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_bytes, error_sink.message_length)?;
        let raw = NonNull::new(raw_plan).ok_or_else(|| {
            RuntimeError::local(
                crate::RuntimeStatus::InternalError,
                "native Ministral3 YaRN prepare returned a null plan",
            )
        })?;
        Ok(PreparedMinistral3Yarn {
            state: Arc::new(PreparedState {
                raw,
                owners: PreparedOwners {
                    context: context.clone(),
                    descriptor,
                },
            }),
        })
    }
}

impl PreparedMinistral3Yarn {
    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<(Ministral3YarnSubmission, Ministral3YarnDispatchInfo), RuntimeError> {
        let mut info = sys::sllm_ministral3_yarn_dispatch_info_t {
            struct_size: size_of::<sys::sllm_ministral3_yarn_dispatch_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_MINISTRAL3_YARN_DISPATCH_INFO_VERSION,
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
            kernel_symbol: [0; sys::SLLM_HIP_MINISTRAL3_YARN_KERNEL_SYMBOL_MAX as usize],
            device_symbol: [0; sys::SLLM_HIP_MINISTRAL3_YARN_DEVICE_SYMBOL_MAX as usize],
            gcn_arch_name: [0; sys::SLLM_HIP_MAX_GCN_ARCH_NAME as usize],
            reserved: [0; 8],
        };
        let mut error_bytes = [0_u8; 256];
        let mut error_sink = sink(&mut error_bytes);
        let mut raw_completion = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_ministral3_yarn_execute(
                self.state.raw.as_ptr(),
                queue.raw_handle()?.as_ptr(),
                &mut raw_completion,
                &mut info,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_bytes, error_sink.message_length)?;
        let raw_completion = NonNull::new(raw_completion).ok_or_else(|| {
            RuntimeError::local(
                crate::RuntimeStatus::InternalError,
                "native Ministral3 YaRN execute returned a null completion",
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
            Ministral3YarnSubmission {
                completion,
                _plan: Arc::clone(&self.state),
            },
            dispatch_info,
        ))
    }
}
