//! C4 request-local linear-attention state and transactional execution.

use std::ffi::c_char;
use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use sllm_core::{
    ExecutionSessionId, LinearAttentionLayout, LinearAttentionRequest,
    LinearAttentionStateDescriptor, LinearAttentionStateId, LinearAttentionStateSnapshot,
    StateForkAuditV1, StateForkModeV1,
};
use sllm_hip_sys as sys;

use crate::Buffer;
use crate::rmsnorm::TensorBinding;
use crate::runtime::{
    CompletionState, Context, Queue, RuntimeError, RuntimeStatus, completion_from_opaque_token,
    enqueue_linear_attention_completion_cleanup, enqueue_linear_attention_state_cleanup, ensure_ok,
    finalize_completion_after, gcn_arch_matches, logical_gcn_arch_name,
    release_linear_attention_completion_once, release_linear_attention_state_once, result_error,
    sink,
};

const ERROR_CAPACITY: usize = 256;
const MAX_FINITE_TIMEOUT_MS: u32 = u32::MAX - 1;
const KERNEL_SYMBOL: &str = "linear_attention.gdn.v1";
const CONV_DEVICE_SYMBOL: &str = "sllm_linear_attention_causal_conv_silu_v1";
const RECURRENT_DEVICE_SYMBOL: &str = "sllm_linear_attention_recurrent_gated_norm_v1";
const COLUMN_KERNEL_SYMBOL: &str = "linear_attention.gdn.column_state.v2";
const COLUMN_RECURRENT_DEVICE_SYMBOL: &str = "sllm_linear_attention_recurrent_column_state_v2";

struct LinearAttentionStateInner {
    raw: usize,
    context: Context,
    session_id: ExecutionSessionId,
    state_id: LinearAttentionStateId,
    descriptor: LinearAttentionStateDescriptor,
    last_generation: AtomicU64,
}

impl Drop for LinearAttentionStateInner {
    fn drop(&mut self) {
        let Some(raw) = NonNull::new(self.raw as *mut sys::sllm_linear_attention_state_t) else {
            return;
        };
        let (status, remaining) = release_linear_attention_state_once(raw);
        if let Some(remaining) = remaining {
            enqueue_linear_attention_state_cleanup(remaining, self.context.clone(), status);
        }
    }
}

#[derive(Clone)]
pub(crate) struct LinearAttentionStateResource {
    inner: Arc<LinearAttentionStateInner>,
}

impl LinearAttentionStateResource {
    pub(crate) fn create(
        context: &Context,
        session_id: ExecutionSessionId,
        state_id: LinearAttentionStateId,
        descriptor: LinearAttentionStateDescriptor,
    ) -> Result<Self, RuntimeError> {
        if descriptor.capacity() > sys::SLLM_HIP_LINEAR_ATTENTION_MAX_CAPACITY {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidLinearAttentionStateDescriptor,
                "linear-attention capacity exceeds the bounded native contract",
            ));
        }
        let info = sys::sllm_linear_attention_state_create_info_t {
            struct_size: size_of::<sys::sllm_linear_attention_state_create_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            session_id: session_id.raw(),
            layer_id: descriptor.layer_id(),
            flags: 0,
            capacity_tokens: descriptor.capacity(),
            qk_heads: descriptor.layout().qk_heads() as u32,
            value_heads: descriptor.layout().value_heads() as u32,
            head_dim: descriptor.layout().head_dim() as u32,
            conv_kernel_size: descriptor.layout().conv_kernel_size() as u32,
        };
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_state = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_linear_attention_state_create(
                context.raw_handle()?.as_ptr(),
                &info,
                &mut raw_state,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        let raw_state = NonNull::new(raw_state).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native linear-attention state create returned null on success",
            )
        })?;
        Ok(Self {
            inner: Arc::new(LinearAttentionStateInner {
                raw: raw_state.as_ptr() as usize,
                context: context.clone(),
                session_id,
                state_id,
                descriptor,
                last_generation: AtomicU64::new(0),
            }),
        })
    }

    fn raw_handle(&self) -> Result<NonNull<sys::sllm_linear_attention_state_t>, RuntimeError> {
        NonNull::new(self.inner.raw as *mut sys::sllm_linear_attention_state_t).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "linear-attention state was already released",
            )
        })
    }

    pub(crate) fn fork(
        &self,
        state_id: LinearAttentionStateId,
        descriptor: LinearAttentionStateDescriptor,
    ) -> Result<(Self, StateForkAuditV1), RuntimeError> {
        if descriptor.layer_id() != self.inner.descriptor.layer_id()
            || descriptor.layout() != self.inner.descriptor.layout()
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidLinearAttentionStateDescriptor,
                "native linear fork requires matching layer/layout metadata",
            ));
        }
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut info = sys::sllm_state_fork_info_t {
            struct_size: size_of::<sys::sllm_state_fork_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_STATE_FORK_INFO_VERSION,
            mode: 0,
            source_state_identity: 0,
            child_state_identity: 0,
            source_owned_bytes: 0,
            child_owned_bytes: 0,
            copied_bytes: 0,
            shared_bytes: 0,
            published_length: 0,
            page_bytes: 0,
            reserved: [0; 4],
        };
        let mut raw_child = std::ptr::null_mut();
        let destination_info = sys::sllm_linear_attention_state_create_info_t {
            struct_size: size_of::<sys::sllm_linear_attention_state_create_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            session_id: self.inner.session_id.raw(),
            layer_id: descriptor.layer_id(),
            flags: 0,
            capacity_tokens: descriptor.capacity(),
            qk_heads: descriptor.layout().qk_heads() as u32,
            value_heads: descriptor.layout().value_heads() as u32,
            head_dim: descriptor.layout().head_dim() as u32,
            conv_kernel_size: descriptor.layout().conv_kernel_size() as u32,
        };
        let status = unsafe {
            sys::sllm_linear_attention_state_fork(
                self.raw_handle()?.as_ptr(),
                &destination_info,
                &mut raw_child,
                &mut info,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        let raw_child = NonNull::new(raw_child).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native linear fork returned a null child handle on success",
            )
        })?;
        if info.mode != sys::SLLM_HIP_STATE_FORK_MODE_DEVICE_COPY {
            let mut child_handle = raw_child.as_ptr();
            let _ = unsafe {
                sys::sllm_linear_attention_state_release(&mut child_handle, &mut error_sink)
            };
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidLinearAttentionStateDescriptor,
                "native linear fork did not use device copy mode",
            ));
        }
        let audit = StateForkAuditV1::new(
            StateForkModeV1::DeviceCopy,
            info.published_length,
            0,
            info.copied_bytes,
            info.child_owned_bytes,
        )
        .map_err(|error| {
            let mut child_handle = raw_child.as_ptr();
            let _ = unsafe {
                sys::sllm_linear_attention_state_release(&mut child_handle, &mut error_sink)
            };
            RuntimeError::new(
                RuntimeStatus::InvalidLinearAttentionStateDescriptor,
                format!("native linear fork audit failed core validation: {error}"),
            )
        })?;
        Ok((
            Self {
                inner: Arc::new(LinearAttentionStateInner {
                    raw: raw_child.as_ptr() as usize,
                    context: self.inner.context.clone(),
                    session_id: self.inner.session_id,
                    state_id,
                    descriptor,
                    last_generation: AtomicU64::new(
                        self.inner.last_generation.load(Ordering::Acquire),
                    ),
                }),
            },
            audit,
        ))
    }

    pub(crate) fn snapshot(&self) -> Result<LinearAttentionStateSnapshot, RuntimeError> {
        let mut info = empty_view_info();
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe {
            sys::sllm_linear_attention_state_query(
                self.raw_handle()?.as_ptr(),
                &mut info,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        self.validate_view(&info)?;
        record_monotonic_generation(&self.inner.last_generation, info.generation)?;
        LinearAttentionStateSnapshot::new(
            self.inner.session_id,
            self.inner.state_id,
            self.inner.descriptor,
            info.observed_length,
        )
        .map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidLinearAttentionStateDescriptor,
                format!("native linear-attention snapshot failed core validation: {error}"),
            )
        })
    }

    pub(crate) fn rewind_last(
        &self,
        expected_length: u64,
        rewind_length: u64,
    ) -> Result<(), RuntimeError> {
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe {
            sys::sllm_linear_attention_state_rewind_last(
                self.raw_handle()?.as_ptr(),
                expected_length,
                rewind_length,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)
    }

    pub(crate) fn export_chunk(
        &self,
        plane: u32,
        byte_offset: u64,
        destination: &mut [u8],
    ) -> Result<(), RuntimeError> {
        if destination.is_empty() {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                "linear export chunk must not be empty",
            ));
        }
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let chunk = sys::sllm_state_chunk_t {
            struct_size: size_of::<sys::sllm_state_chunk_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_STATE_FORK_INFO_VERSION,
            plane,
            reserved0: 0,
            reserved1: 0,
            byte_offset,
            byte_length: destination.len() as u64,
            host_pointer: destination.as_mut_ptr().cast(),
            host_capacity: destination.len() as u64,
            reserved: [0; 4],
        };
        let status = unsafe {
            sys::sllm_linear_attention_state_export(
                self.raw_handle()?.as_ptr(),
                &chunk,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)
    }

    pub(crate) fn import_chunk(
        &self,
        plane: u32,
        byte_offset: u64,
        source: &[u8],
    ) -> Result<(), RuntimeError> {
        if source.is_empty() {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                "linear import chunk must not be empty",
            ));
        }
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let chunk = sys::sllm_state_chunk_t {
            struct_size: size_of::<sys::sllm_state_chunk_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_STATE_FORK_INFO_VERSION,
            plane,
            reserved0: 0,
            reserved1: 0,
            byte_offset,
            byte_length: source.len() as u64,
            host_pointer: source.as_ptr().cast_mut().cast(),
            host_capacity: source.len() as u64,
            reserved: [0; 4],
        };
        let status = unsafe {
            sys::sllm_linear_attention_state_import(
                self.raw_handle()?.as_ptr(),
                &chunk,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)
    }

    pub(crate) fn image_query(&self) -> Result<sys::sllm_state_image_info_t, RuntimeError> {
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut info = sys::sllm_state_image_info_t {
            struct_size: size_of::<sys::sllm_state_image_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_STATE_FORK_INFO_VERSION,
            reserved0: 0,
            session_id: 0,
            layer_id: 0,
            dtype: 0,
            encoding: 0,
            active_slot: 0,
            capacity_tokens: 0,
            published_length: 0,
            generation: 0,
            plane_count: 0,
            reserved: [0; 7],
        };
        let status = unsafe {
            sys::sllm_linear_attention_state_image_query(
                self.raw_handle()?.as_ptr(),
                &mut info,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        Ok(info)
    }

    pub(crate) fn image_plane_size(&self, plane: u32) -> Result<u64, RuntimeError> {
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut size_bytes = 0_u64;
        let status = unsafe {
            sys::sllm_linear_attention_state_image_plane_size(
                self.raw_handle()?.as_ptr(),
                plane,
                &mut size_bytes,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        if size_bytes == 0 {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidLinearAttentionStateDescriptor,
                "native linear image plane size is zero",
            ));
        }
        Ok(size_bytes)
    }

    pub(crate) fn import_finalize(
        &self,
        info: &sys::sllm_state_image_info_t,
    ) -> Result<(), RuntimeError> {
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe {
            sys::sllm_linear_attention_state_import_finalize(
                self.raw_handle()?.as_ptr(),
                info,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)
    }

    fn validate_view(
        &self,
        info: &sys::sllm_linear_attention_view_info_t,
    ) -> Result<(), RuntimeError> {
        let layout = self.inner.descriptor.layout();
        let valid = info.struct_size as usize
            == size_of::<sys::sllm_linear_attention_view_info_t>()
            && info.abi_version == sys::SLLM_HIP_ABI_VERSION
            && info.info_version == sys::SLLM_HIP_LINEAR_ATTENTION_VIEW_INFO_VERSION
            && info.reserved0 == 0
            && info.session_id == self.inner.session_id.raw()
            && info.layer_id == self.inner.descriptor.layer_id()
            && info.conv_state_dtype == sys::SLLM_TENSOR_DTYPE_BF16
            && info.recurrent_state_dtype == sys::SLLM_TENSOR_DTYPE_F32
            && info.encoding == sys::SLLM_TENSOR_ENCODING_UNQUANTIZED
            && info.active_slot <= 1
            && info.capacity_tokens == self.inner.descriptor.capacity()
            && info.observed_length <= info.capacity_tokens
            && info.context_identity != 0
            && info.state_identity == self.inner.raw as u64
            && info.conv_state_shape == layout.conv_state_shape()
            && info.recurrent_state_shape == layout.recurrent_state_shape()
            && info.reserved.iter().all(|value| *value == 0);
        if valid {
            Ok(())
        } else {
            Err(RuntimeError::local(
                RuntimeStatus::InvalidLinearAttentionStateDescriptor,
                "native linear-attention view metadata violated the fixed contract",
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute(
        &self,
        queue: &Queue,
        bindings: [&TensorBinding; 9],
        request: LinearAttentionRequest,
    ) -> Result<(LinearAttentionCompletion, LinearAttentionEvidence), RuntimeError> {
        if request.state_id() != self.inner.state_id
            || request.state_descriptor() != self.inner.descriptor
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidLinearAttentionDescriptor,
                "Rust linear-attention request is not the admitted state transition",
            ));
        }
        let [
            qkv,
            z,
            b_input,
            a_input,
            conv_weight,
            a_log,
            dt_bias,
            norm_weight,
            output,
        ] = bindings;
        let descriptor = request.descriptor();
        let native = sys::sllm_linear_attention_desc_t {
            struct_size: size_of::<sys::sllm_linear_attention_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_LINEAR_ATTENTION_VERSION,
            reserved0: 0,
            start_position: descriptor.start_position(),
            expected_length: descriptor.expected_length(),
            state: self.raw_handle()?.as_ptr(),
            qkv: qkv.raw()?,
            z: z.raw()?,
            b_input: b_input.raw()?,
            a_input: a_input.raw()?,
            conv_weight: conv_weight.raw()?,
            a_log: a_log.raw()?,
            dt_bias: dt_bias.raw()?,
            norm_weight: norm_weight.raw()?,
            output: output.raw()?,
            reserved: [0; 4],
        };
        let mut info = empty_dispatch_info();
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_completion = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_linear_attention_execute(
                self.inner.context.raw_handle()?.as_ptr(),
                queue.raw_handle()?.as_ptr(),
                &native,
                &mut raw_completion,
                &mut info,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        let raw_completion = NonNull::new(raw_completion).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native linear-attention execute returned null completion on success",
            )
        })?;
        let evidence = validate_dispatch(
            &info,
            request.descriptor(),
            self.inner.descriptor.layout(),
            self.inner.context.expected_target(),
        )?;
        let retained = bindings.map(|binding| binding.buffer().clone()).to_vec();
        Ok((
            LinearAttentionCompletion {
                raw: Some(raw_completion.as_ptr() as usize),
                context: self.inner.context.clone(),
                queue: queue.clone(),
                buffers: retained,
                state: self.clone(),
                terminal: false,
                canceled: false,
            },
            evidence,
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LinearAttentionEvidence {
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub recurrent_kernel_id: u32,
    pub workgroup_size_x: u32,
    pub recurrent_grid_size_x: u32,
    pub token_count: u64,
    pub kernel_symbol: String,
    pub recurrent_device_symbol: String,
    pub target: String,
}

pub(crate) struct LinearAttentionCompletion {
    raw: Option<usize>,
    context: Context,
    queue: Queue,
    buffers: Vec<Buffer>,
    state: LinearAttentionStateResource,
    terminal: bool,
    canceled: bool,
}

impl LinearAttentionCompletion {
    pub(crate) fn query(&mut self) -> Result<CompletionState, RuntimeError> {
        self.call_completion(None)
    }

    pub(crate) fn wait(&mut self, timeout: Duration) -> Result<CompletionState, RuntimeError> {
        self.call_completion(Some(timeout))
    }

    pub(crate) fn finalize_after_token(
        &mut self,
        fence_token: u64,
    ) -> Result<CompletionState, RuntimeError> {
        let state = finalize_completion_after(
            self.raw_handle()?,
            completion_from_opaque_token(fence_token)?,
        )?;
        self.terminal = state != CompletionState::Pending;
        Ok(state)
    }

    fn raw_handle(&self) -> Result<NonNull<sys::sllm_completion_t>, RuntimeError> {
        self.raw
            .and_then(|raw| NonNull::new(raw as *mut sys::sllm_completion_t))
            .ok_or_else(|| {
                RuntimeError::local(
                    RuntimeStatus::InvalidHandle,
                    "linear-attention completion was already released",
                )
            })
    }

    fn cancel(&mut self) -> Result<(), RuntimeError> {
        if self.terminal || self.canceled {
            return Ok(());
        }
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe {
            sys::sllm_linear_attention_cancel(
                self.state.raw_handle()?.as_ptr(),
                self.raw_handle()?.as_ptr(),
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        self.canceled = true;
        Ok(())
    }

    fn call_completion(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<CompletionState, RuntimeError> {
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut result = completion_result();
        let status = unsafe {
            match timeout {
                Some(timeout) => sys::sllm_completion_wait(
                    self.raw_handle()?.as_ptr(),
                    timeout_millis(timeout),
                    &mut result,
                    &mut error_sink,
                ),
                None => sys::sllm_completion_query(
                    self.raw_handle()?.as_ptr(),
                    &mut result,
                    &mut error_sink,
                ),
            }
        };
        let state = completion_state(result.state)?;
        if state != CompletionState::Pending {
            self.terminal = true;
        }
        let runtime_status = RuntimeStatus::from_raw(status);
        if runtime_status == RuntimeStatus::Ok {
            return Ok(state);
        }
        Err(result_error(
            runtime_status.raw(),
            &error_buffer,
            error_sink.message_length,
        ))
    }
}

impl Drop for LinearAttentionCompletion {
    fn drop(&mut self) {
        if !self.terminal && !self.canceled {
            let _ = self.cancel();
        }
        let Some(raw_value) = self.raw.take() else {
            return;
        };
        let Some(raw) = NonNull::new(raw_value as *mut sys::sllm_completion_t) else {
            return;
        };
        let (status, remaining) = release_linear_attention_completion_once(raw);
        if let Some(remaining) = remaining {
            enqueue_linear_attention_completion_cleanup(
                remaining,
                self.context.clone(),
                self.queue.clone(),
                self.buffers.clone(),
                self.state.clone(),
                status,
            );
        }
    }
}

fn empty_view_info() -> sys::sllm_linear_attention_view_info_t {
    sys::sllm_linear_attention_view_info_t {
        struct_size: size_of::<sys::sllm_linear_attention_view_info_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_LINEAR_ATTENTION_VIEW_INFO_VERSION,
        reserved0: 0,
        session_id: 0,
        layer_id: 0,
        conv_state_dtype: 0,
        recurrent_state_dtype: 0,
        encoding: 0,
        active_slot: 0,
        capacity_tokens: 0,
        observed_length: 0,
        generation: 0,
        context_identity: 0,
        state_identity: 0,
        conv_state_shape: [0; 2],
        recurrent_state_shape: [0; 3],
        reserved: [0; 4],
    }
}

fn empty_dispatch_info() -> sys::sllm_linear_attention_dispatch_info_t {
    sys::sllm_linear_attention_dispatch_info_t {
        struct_size: size_of::<sys::sllm_linear_attention_dispatch_info_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_LINEAR_ATTENTION_DISPATCH_INFO_VERSION,
        backend: 0,
        dispatch_id: 0,
        dispatch_count: 0,
        conv_kernel_id: 0,
        recurrent_kernel_id: 0,
        workgroup_size_x: 0,
        conv_grid_size_x: 0,
        recurrent_grid_size_x: 0,
        token_count: 0,
        start_position: 0,
        expected_length: 0,
        qk_heads: 0,
        value_heads: 0,
        head_dim: 0,
        fallback_allowed: 0,
        fallback_used: 0,
        kernel_symbol: [0; 64],
        conv_device_symbol: [0; 64],
        recurrent_device_symbol: [0; 64],
        gcn_arch_name: [0; 64],
        reserved: [0; 8],
    }
}

fn validate_dispatch(
    info: &sys::sllm_linear_attention_dispatch_info_t,
    descriptor: sllm_core::LinearAttentionDescriptor,
    layout: LinearAttentionLayout,
    expected_target: Option<&str>,
) -> Result<LinearAttentionEvidence, RuntimeError> {
    let observed_target = read_c_string(&info.gcn_arch_name);
    let target = logical_gcn_arch_name(&observed_target).to_owned();
    let kernel_symbol = read_c_string(&info.kernel_symbol);
    let conv_device_symbol = read_c_string(&info.conv_device_symbol);
    let recurrent_device_symbol = read_c_string(&info.recurrent_device_symbol);
    let expected_conv_grid = convolution_grid_size(descriptor.token_count(), layout);
    let force_baseline =
        std::env::var_os("SLLM_GDN_FORCE_BASELINE").is_some_and(|value| value == "1");
    let use_column_provider = descriptor.token_count() >= 128
        && !force_baseline
        && matches!(
            logical_gcn_arch_name(&observed_target),
            "gfx1030" | "gfx1201"
        );
    let valid = info.struct_size as usize
        == size_of::<sys::sllm_linear_attention_dispatch_info_t>()
        && info.abi_version == sys::SLLM_HIP_ABI_VERSION
        && info.info_version == sys::SLLM_HIP_LINEAR_ATTENTION_DISPATCH_INFO_VERSION
        && info.backend == sys::SLLM_BACKEND_HIP
        && info.dispatch_id != 0
        && info.dispatch_count == if use_column_provider { 4 } else { 2 }
        && info.conv_kernel_id == sys::SLLM_HIP_LINEAR_ATTENTION_KERNEL_ID_CAUSAL_CONV_SILU_V1
        && info.recurrent_kernel_id
            == sys::SLLM_HIP_LINEAR_ATTENTION_KERNEL_ID_RECURRENT_GATED_NORM_V1
        && info.workgroup_size_x == sys::SLLM_HIP_LINEAR_ATTENTION_WORKGROUP_SIZE
        && expected_conv_grid == Some(info.conv_grid_size_x)
        && info.recurrent_grid_size_x
            == if use_column_provider {
                (layout.value_heads() * layout.head_dim() / 4) as u32
            } else {
                layout.value_heads() as u32
            }
        && info.token_count == descriptor.token_count()
        && info.start_position == descriptor.start_position()
        && info.expected_length == descriptor.expected_length()
        && info.qk_heads == layout.qk_heads() as u32
        && info.value_heads == layout.value_heads() as u32
        && info.head_dim == layout.head_dim() as u32
        && info.fallback_allowed == 0
        && info.fallback_used == 0
        && kernel_symbol
            == if use_column_provider {
                COLUMN_KERNEL_SYMBOL
            } else {
                KERNEL_SYMBOL
            }
        && conv_device_symbol == CONV_DEVICE_SYMBOL
        && recurrent_device_symbol
            == if use_column_provider {
                COLUMN_RECURRENT_DEVICE_SYMBOL
            } else {
                RECURRENT_DEVICE_SYMBOL
            }
        && expected_target.is_none_or(|expected| gcn_arch_matches(expected, &observed_target))
        && info.reserved.iter().all(|value| *value == 0);
    if !valid {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidLinearAttentionDescriptor,
            "native linear-attention dispatch metadata violated the fixed contract",
        ));
    }
    Ok(LinearAttentionEvidence {
        dispatch_id: info.dispatch_id,
        dispatch_count: info.dispatch_count,
        recurrent_kernel_id: info.recurrent_kernel_id,
        workgroup_size_x: info.workgroup_size_x,
        recurrent_grid_size_x: info.recurrent_grid_size_x,
        token_count: info.token_count,
        kernel_symbol,
        recurrent_device_symbol,
        target,
    })
}

fn convolution_grid_size(token_count: u64, layout: LinearAttentionLayout) -> Option<u32> {
    let elements = token_count
        .checked_add(layout.conv_history() as u64)?
        .checked_mul(layout.qkv_width() as u64)?;
    let workgroup = sys::SLLM_HIP_LINEAR_ATTENTION_WORKGROUP_SIZE as u64;
    let grid = elements.checked_add(workgroup - 1)? / workgroup;
    u32::try_from(grid).ok()
}

fn record_monotonic_generation(
    last_generation: &AtomicU64,
    generation: u64,
) -> Result<(), RuntimeError> {
    let previous = last_generation.fetch_max(generation, Ordering::AcqRel);
    if generation < previous {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidLinearAttentionStateDescriptor,
            "native linear-attention generation moved backwards",
        ));
    }
    Ok(())
}

fn read_c_string<const N: usize>(bytes: &[c_char; N]) -> String {
    let length = bytes.iter().position(|byte| *byte == 0).unwrap_or(N);
    String::from_utf8_lossy(
        &bytes[..length]
            .iter()
            .map(|byte| *byte as u8)
            .collect::<Vec<_>>(),
    )
    .into_owned()
}

fn completion_result() -> sys::sllm_completion_result_t {
    sys::sllm_completion_result_t {
        struct_size: size_of::<sys::sllm_completion_result_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        state: sys::SLLM_COMPLETION_STATE_PENDING,
        reserved0: 0,
        transfer_size_bytes: 0,
        available_bytes: 0,
        reserved: [0; 4],
    }
}

fn completion_state(raw: u32) -> Result<CompletionState, RuntimeError> {
    match raw {
        sys::SLLM_COMPLETION_STATE_PENDING => Ok(CompletionState::Pending),
        sys::SLLM_COMPLETION_STATE_SUCCESS => Ok(CompletionState::Success),
        sys::SLLM_COMPLETION_STATE_FAILURE => Ok(CompletionState::Failure),
        _ => Err(RuntimeError::local(
            RuntimeStatus::InternalError,
            "native completion returned an unknown state",
        )),
    }
}

fn timeout_millis(timeout: Duration) -> u32 {
    u32::try_from(timeout.as_millis()).unwrap_or(MAX_FINITE_TIMEOUT_MS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::thread;

    fn put<const N: usize>(destination: &mut [c_char; N], value: &str) {
        for (output, input) in destination.iter_mut().zip(value.bytes()) {
            *output = input as c_char;
        }
    }

    fn valid_dispatch(
        descriptor: sllm_core::LinearAttentionDescriptor,
    ) -> sys::sllm_linear_attention_dispatch_info_t {
        let mut info = empty_dispatch_info();
        info.backend = sys::SLLM_BACKEND_HIP;
        info.dispatch_id = 7;
        let use_column_provider = descriptor.token_count() >= 128;
        info.dispatch_count = if use_column_provider { 4 } else { 2 };
        info.conv_kernel_id = sys::SLLM_HIP_LINEAR_ATTENTION_KERNEL_ID_CAUSAL_CONV_SILU_V1;
        info.recurrent_kernel_id = sys::SLLM_HIP_LINEAR_ATTENTION_KERNEL_ID_RECURRENT_GATED_NORM_V1;
        info.workgroup_size_x = sys::SLLM_HIP_LINEAR_ATTENTION_WORKGROUP_SIZE;
        let layout = LinearAttentionLayout::default();
        info.conv_grid_size_x = convolution_grid_size(descriptor.token_count(), layout).unwrap();
        info.recurrent_grid_size_x = if use_column_provider {
            sys::SLLM_HIP_LINEAR_ATTENTION_VALUE_HEADS * sys::SLLM_HIP_LINEAR_ATTENTION_HEAD_DIM / 4
        } else {
            sys::SLLM_HIP_LINEAR_ATTENTION_VALUE_HEADS
        };
        info.token_count = descriptor.token_count();
        info.start_position = descriptor.start_position();
        info.expected_length = descriptor.expected_length();
        info.qk_heads = sys::SLLM_HIP_LINEAR_ATTENTION_QK_HEADS;
        info.value_heads = sys::SLLM_HIP_LINEAR_ATTENTION_VALUE_HEADS;
        info.head_dim = sys::SLLM_HIP_LINEAR_ATTENTION_HEAD_DIM;
        put(
            &mut info.kernel_symbol,
            if use_column_provider {
                COLUMN_KERNEL_SYMBOL
            } else {
                KERNEL_SYMBOL
            },
        );
        put(&mut info.conv_device_symbol, CONV_DEVICE_SYMBOL);
        put(
            &mut info.recurrent_device_symbol,
            if use_column_provider {
                COLUMN_RECURRENT_DEVICE_SYMBOL
            } else {
                RECURRENT_DEVICE_SYMBOL
            },
        );
        put(&mut info.gcn_arch_name, "gfx1201");
        info
    }

    #[test]
    fn dispatch_requires_exact_symbols_and_convolution_grid_formula() {
        let descriptor = sllm_core::LinearAttentionDescriptor::new(3, 17, 20).unwrap();
        let layout = LinearAttentionLayout::default();
        let info = valid_dispatch(descriptor);
        assert!(validate_dispatch(&info, descriptor, layout, Some("gfx1201")).is_ok());
        assert_eq!(info.conv_grid_size_x, ((17_u32 + 3) * 8_192).div_ceil(128));

        let mut wrong_grid = info;
        wrong_grid.conv_grid_size_x += 1;
        assert!(validate_dispatch(&wrong_grid, descriptor, layout, Some("gfx1201")).is_err());

        for field in 0..3 {
            let mut wrong_symbol = info;
            match field {
                0 => put(&mut wrong_symbol.kernel_symbol, "linear_attention.gdn.v2"),
                1 => put(
                    &mut wrong_symbol.conv_device_symbol,
                    "sllm_linear_attention_causal_conv_silu_v2",
                ),
                _ => put(
                    &mut wrong_symbol.recurrent_device_symbol,
                    "sllm_linear_attention_recurrent_gated_norm_v2",
                ),
            }
            assert!(validate_dispatch(&wrong_symbol, descriptor, layout, Some("gfx1201")).is_err());
        }
    }

    #[test]
    fn column_provider_starts_at_the_long_prefill_boundary() {
        let layout = LinearAttentionLayout::default();
        for token_count in [127_u64, 128, 129] {
            let descriptor =
                sllm_core::LinearAttentionDescriptor::new(0, token_count, token_count).unwrap();
            let info = valid_dispatch(descriptor);
            assert!(validate_dispatch(&info, descriptor, layout, Some("gfx1201")).is_ok());
            assert_eq!(info.dispatch_count, if token_count >= 128 { 4 } else { 2 });
        }
    }

    #[test]
    fn generation_tracker_is_atomic_monotonic_under_concurrent_observations() {
        let generation = Arc::new(AtomicU64::new(0));
        let gate = Arc::new(Barrier::new(7));
        let mut workers = Vec::new();
        for observed in [1_u64, 3, 17, 255, 256, 257] {
            let generation = Arc::clone(&generation);
            let gate = Arc::clone(&gate);
            workers.push(thread::spawn(move || {
                gate.wait();
                let _ = record_monotonic_generation(&generation, observed);
            }));
        }
        gate.wait();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(generation.load(Ordering::Acquire), 257);
        assert!(record_monotonic_generation(&generation, 256).is_err());
        assert_eq!(generation.load(Ordering::Acquire), 257);
        assert!(record_monotonic_generation(&generation, 258).is_ok());
        assert_eq!(generation.load(Ordering::Acquire), 258);
    }
}
