//! C3a2 request-local FP16 KV state ownership and append transactions.
//!
//! Native KV handles are deliberately kept opaque. The only Rust-visible
//! state is copied metadata; the two device allocations and their strides
//! remain native-owned. The sendable resource token is an erased-core
//! lifetime boundary, while the direct state/view owner types carry an Rc
//! marker so they cannot be moved or shared as thread-affine native views.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, Weak};
use std::time::Duration;

use sllm_core::{
    DType, Encoding, ExecutionSessionId, KvStateAppendRequest, KvStateDescriptor, KvStateId,
    KvStateSnapshot,
};
use sllm_hip_sys as sys;

use crate::Buffer;
use crate::rmsnorm::TensorBinding;
use crate::runtime::{
    CompletionState, Context, Queue, RuntimeError, RuntimeStatus,
    enqueue_causal_completion_cleanup, enqueue_kv_completion_cleanup, enqueue_kv_state_cleanup,
    enqueue_kv_view_cleanup, ensure_ok, release_causal_completion_once, release_kv_completion_once,
    release_kv_state_once, release_kv_view_once, result_error, sink,
};

const ERROR_CAPACITY: usize = 256;
const MAX_FINITE_TIMEOUT_MS: u32 = u32::MAX - 1;
const KERNEL_SYMBOL: &str = "kv_state.bf16_to_f16_transpose.v1";
const DEVICE_SYMBOL: &str = "sllm_kv_state_bf16_to_f16_transpose_v1";

struct KvStateInner {
    raw: usize,
    context: Context,
    session_id: ExecutionSessionId,
    state_id: KvStateId,
    descriptor: KvStateDescriptor,
    last_generation: AtomicU64,
}

type EvidenceKey = (u64, u64);
type EvidenceResourceMap = HashMap<EvidenceKey, Weak<KvStateInner>>;

static EVIDENCE_RESOURCES: OnceLock<Mutex<EvidenceResourceMap>> = OnceLock::new();

fn evidence_resources() -> &'static Mutex<EvidenceResourceMap> {
    EVIDENCE_RESOURCES.get_or_init(|| Mutex::new(EvidenceResourceMap::new()))
}

impl Drop for KvStateInner {
    fn drop(&mut self) {
        if let Ok(mut resources) = evidence_resources().lock() {
            resources.remove(&(self.session_id.raw(), self.state_id.raw()));
        }
        let Some(raw) = NonNull::new(self.raw as *mut sys::sllm_kv_state_t) else {
            return;
        };
        let (status, remaining) = release_kv_state_once(raw);
        if let Some(remaining) = remaining {
            enqueue_kv_state_cleanup(remaining, self.context.clone(), status);
        }
    }
}

/// Sendable opaque ownership token used by the erased core resource.
///
/// It contains no dereferenceable pointer or writable storage. The native
/// registry owns synchronization and handle validation; the final Arc drop
/// is the only point that releases the state handle.
#[derive(Clone)]
pub(crate) struct KvStateResource {
    inner: Arc<KvStateInner>,
}

impl KvStateResource {
    pub(crate) fn create(
        context: &Context,
        session_id: ExecutionSessionId,
        state_id: KvStateId,
        descriptor: KvStateDescriptor,
    ) -> Result<Self, RuntimeError> {
        if descriptor.capacity() > sys::SLLM_HIP_KV_MAX_CAPACITY {
            return Err(RuntimeError::local(
                RuntimeStatus::KvCapacityExceeded,
                "KV capacity exceeds the bounded native contract",
            ));
        }
        let context_raw = context.raw_handle()?;
        let info = sys::sllm_kv_state_create_info_t {
            struct_size: size_of::<sys::sllm_kv_state_create_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            session_id: session_id.raw(),
            layer_id: descriptor.layer_id(),
            flags: 0,
            capacity_tokens: descriptor.capacity(),
            reserved: [0; 4],
        };
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_state = std::ptr::null_mut();
        let raw = unsafe {
            sys::sllm_kv_state_create(context_raw.as_ptr(), &info, &mut raw_state, &mut error_sink)
        };
        ensure_ok(raw, &error_buffer, error_sink.message_length)?;
        let raw_state = NonNull::new(raw_state).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native KV state create returned a null handle on success",
            )
        })?;
        let resource = Self {
            inner: Arc::new(KvStateInner {
                raw: raw_state.as_ptr() as usize,
                context: context.clone(),
                session_id,
                state_id,
                descriptor,
                last_generation: AtomicU64::new(0),
            }),
        };
        evidence_resources()
            .lock()
            .map_err(|_| {
                RuntimeError::local(
                    RuntimeStatus::InternalError,
                    "KV evidence resource registry is poisoned",
                )
            })?
            .insert(
                (session_id.raw(), state_id.raw()),
                Arc::downgrade(&resource.inner),
            );
        Ok(resource)
    }

    fn raw_handle(&self) -> Result<NonNull<sys::sllm_kv_state_t>, RuntimeError> {
        NonNull::new(self.inner.raw as *mut sys::sllm_kv_state_t).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "KV state was already released",
            )
        })
    }

    pub(crate) fn snapshot(&self) -> Result<KvStateSnapshot, RuntimeError> {
        let view = NativeKvSnapshotOwner::create(self)?;
        let info = view.query()?;
        validate_view_info(
            &info,
            &self.inner.context,
            self.inner.raw,
            self.inner.session_id,
            self.inner.descriptor,
        )?;
        let previous = self.inner.last_generation.load(Ordering::Acquire);
        if info.generation < previous {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvStateDescriptor,
                "native KV snapshot generation moved backwards",
            ));
        }
        self.inner
            .last_generation
            .store(info.generation, Ordering::Release);
        KvStateSnapshot::new(
            self.inner.session_id,
            self.inner.state_id,
            self.inner.descriptor,
            info.observed_length,
        )
        .map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidKvStateDescriptor,
                format!("native KV snapshot failed core validation: {error}"),
            )
        })
    }

    pub(crate) fn readback(
        &self,
        plane: u32,
        byte_offset: u64,
        destination: &mut [u8],
    ) -> Result<(), RuntimeError> {
        let view = NativeKvSnapshotOwner::create(self)?;
        let info = view.query()?;
        validate_view_info(
            &info,
            &self.inner.context,
            self.inner.raw,
            self.inner.session_id,
            self.inner.descriptor,
        )?;
        view.readback(plane, byte_offset, destination)
    }

    pub(crate) fn append(
        &self,
        queue: &Queue,
        key: &TensorBinding,
        value: &TensorBinding,
        request: KvStateAppendRequest,
    ) -> Result<(KvAppendCompletion, KvAppendEvidence), RuntimeError> {
        if request.state_id() != self.inner.state_id
            || request.descriptor() != self.inner.descriptor
            || request.start_position() != request.expected_length()
            || request.end_position() > self.inner.descriptor.capacity()
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvAppendDescriptor,
                "Rust KV append metadata is not the admitted state request",
            ));
        }
        let key_raw = key.raw()?;
        let value_raw = value.raw()?;
        validate_append_binding(key)?;
        validate_append_binding(value)?;
        let descriptor = sys::sllm_kv_append_desc_t {
            struct_size: size_of::<sys::sllm_kv_append_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            append_version: sys::SLLM_HIP_KV_STATE_VERSION,
            reserved0: 0,
            expected_length: request.expected_length(),
            start_position: request.start_position(),
            key_input: key_raw,
            value_input: value_raw,
            reserved: [0; 4],
        };
        let state_raw = self.raw_handle()?;
        let queue_raw = queue.raw_handle()?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_completion = std::ptr::null_mut();
        let mut append_info = empty_append_info();
        let raw = unsafe {
            sys::sllm_kv_state_append(
                state_raw.as_ptr(),
                queue_raw.as_ptr(),
                &descriptor,
                &mut raw_completion,
                &mut append_info,
                &mut error_sink,
            )
        };
        ensure_ok(raw, &error_buffer, error_sink.message_length)?;
        let raw_completion = NonNull::new(raw_completion).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native KV append returned a null completion on success",
            )
        })?;
        let completion = KvAppendCompletion {
            raw: Some(raw_completion.as_ptr() as usize),
            context: self.inner.context.clone(),
            queue: queue.clone(),
            key: key.buffer().clone(),
            value: value.buffer().clone(),
            state: self.clone(),
            terminal: false,
            canceled: false,
        };
        let evidence = match validate_append_info(
            &append_info,
            &self.inner.context,
            request,
            self.inner.descriptor,
        ) {
            Ok(evidence) => evidence,
            Err(error) => {
                drop(completion);
                return Err(error);
            }
        };
        Ok((completion, evidence))
    }

    pub(crate) fn causal_attention(
        &self,
        queue: &Queue,
        query: &TensorBinding,
        output: &TensorBinding,
        start_position: u64,
        expected_kv_length: u64,
    ) -> Result<(CausalAttentionCompletion, CausalAttentionEvidence), RuntimeError> {
        validate_causal_attention_binding(query)?;
        validate_causal_attention_binding(output)?;
        let descriptor = sys::sllm_causal_attention_desc_t {
            struct_size: size_of::<sys::sllm_causal_attention_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_CAUSAL_ATTENTION_VERSION,
            reserved0: 0,
            start_position,
            expected_kv_length,
            kv_state: self.raw_handle()?.as_ptr(),
            query: query.raw()?,
            output: output.raw()?,
            reserved: [0; 4],
        };
        let mut dispatch_info = empty_causal_attention_info();
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_completion = std::ptr::null_mut();
        let raw = unsafe {
            sys::sllm_causal_attention_execute(
                self.inner.context.raw_handle()?.as_ptr(),
                queue.raw_handle()?.as_ptr(),
                &descriptor,
                &mut raw_completion,
                &mut dispatch_info,
                &mut error_sink,
            )
        };
        ensure_ok(raw, &error_buffer, error_sink.message_length)?;
        let raw_completion = NonNull::new(raw_completion).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native causal attention returned a null completion on success",
            )
        })?;
        let completion = CausalAttentionCompletion {
            raw: Some(raw_completion.as_ptr() as usize),
            context: self.inner.context.clone(),
            queue: queue.clone(),
            _query: query.buffer().clone(),
            _output: output.buffer().clone(),
            state: self.clone(),
            terminal: false,
        };
        let evidence = match validate_causal_attention_info(
            &dispatch_info,
            &self.inner.context,
            start_position,
            expected_kv_length,
            self.inner.descriptor,
        ) {
            Ok(evidence) => evidence,
            Err(error) => {
                drop(completion);
                return Err(error);
            }
        };
        Ok((completion, evidence))
    }
}

pub(crate) fn resource_for_evidence(session_id: u64, state_id: u64) -> Option<KvStateResource> {
    evidence_resources()
        .lock()
        .ok()?
        .get(&(session_id, state_id))
        .and_then(Weak::upgrade)
        .map(|inner| KvStateResource { inner })
}

/// Direct native state owner. It is intentionally not Send or Sync.
#[derive(Clone)]
pub(crate) struct NativeKvStateOwner {
    _resource: KvStateResource,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl NativeKvStateOwner {
    pub(crate) fn new(resource: KvStateResource) -> Self {
        Self {
            _resource: resource,
            _not_send_sync: PhantomData,
        }
    }
}

struct NativeKvSnapshotOwner {
    raw: Option<NonNull<sys::sllm_kv_view_t>>,
    context: Context,
    _state: NativeKvStateOwner,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl NativeKvSnapshotOwner {
    fn create(state: &KvStateResource) -> Result<Self, RuntimeError> {
        let state_raw = state.raw_handle()?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_view = std::ptr::null_mut();
        let raw = unsafe {
            sys::sllm_kv_state_snapshot(state_raw.as_ptr(), &mut raw_view, &mut error_sink)
        };
        ensure_ok(raw, &error_buffer, error_sink.message_length)?;
        let raw_view = NonNull::new(raw_view).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native KV snapshot returned a null view on success",
            )
        })?;
        Ok(Self {
            raw: Some(raw_view),
            context: state.inner.context.clone(),
            _state: NativeKvStateOwner::new(state.clone()),
            _not_send_sync: PhantomData,
        })
    }

    fn query(&self) -> Result<sys::sllm_kv_view_info_t, RuntimeError> {
        let raw = self.raw.ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "KV snapshot view was released",
            )
        })?;
        let mut info = empty_view_info();
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe { sys::sllm_kv_view_query(raw.as_ptr(), &mut info, &mut error_sink) };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        Ok(info)
    }

    fn readback(
        &self,
        plane: u32,
        byte_offset: u64,
        destination: &mut [u8],
    ) -> Result<(), RuntimeError> {
        if destination.is_empty() {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                "KV evidence readback destination is empty",
            ));
        }
        let raw = self.raw.ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "KV snapshot view was released",
            )
        })?;
        let byte_length = u64::try_from(destination.len()).map_err(|_| {
            RuntimeError::local(
                RuntimeStatus::MetadataOverflow,
                "KV evidence readback destination is too large",
            )
        })?;
        let request = sys::evidence::sllm_hip_kv_readback_request_t {
            struct_size: size_of::<sys::evidence::sllm_hip_kv_readback_request_t>() as u32,
            abi_version: sys::evidence::SLLM_HIP_KV_EVIDENCE_ABI_VERSION,
            view: raw.as_ptr(),
            plane,
            reserved0: 0,
            byte_offset,
            byte_length,
            host_capacity: byte_length,
            host_output: destination.as_mut_ptr(),
            reserved: [0; 4],
        };
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe { sys::evidence::sllm_hip_kv_view_readback(&request, &mut error_sink) };
        ensure_ok(status, &error_buffer, error_sink.message_length)
    }
}

impl Drop for NativeKvSnapshotOwner {
    fn drop(&mut self) {
        let Some(raw) = self.raw.take() else {
            return;
        };
        let (status, remaining) = release_kv_view_once(raw);
        if let Some(remaining) = remaining {
            enqueue_kv_view_cleanup(remaining, self.context.clone(), status);
        }
    }
}

/// Metadata returned by one accepted native append.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvAppendEvidence {
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
    pub start_position: u64,
    pub token_count: u64,
    pub end_position: u64,
    pub commit_allowed: bool,
    pub fallback_allowed: bool,
    pub fallback_used: bool,
    pub kernel_symbol: String,
    pub device_symbol: String,
    pub target: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CausalAttentionEvidence {
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
    pub query_count: u64,
    pub start_position: u64,
    pub committed_kv_length: u64,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub scale_denominator: u32,
    pub fallback_allowed: bool,
    pub fallback_used: bool,
    pub kernel_symbol: String,
    pub device_symbol: String,
    pub target: String,
}

/// Sendable append completion retaining every native dependency.
pub(crate) struct KvAppendCompletion {
    raw: Option<usize>,
    context: Context,
    queue: Queue,
    key: Buffer,
    value: Buffer,
    state: KvStateResource,
    terminal: bool,
    canceled: bool,
}

pub(crate) struct CausalAttentionCompletion {
    raw: Option<usize>,
    context: Context,
    queue: Queue,
    _query: Buffer,
    _output: Buffer,
    state: KvStateResource,
    terminal: bool,
}

impl CausalAttentionCompletion {
    pub(crate) fn query(&mut self) -> Result<CompletionState, RuntimeError> {
        self.call_completion(None)
    }

    pub(crate) fn wait(&mut self, timeout: Duration) -> Result<CompletionState, RuntimeError> {
        self.call_completion(Some(timeout))
    }

    fn raw_handle(&self) -> Result<NonNull<sys::sllm_completion_t>, RuntimeError> {
        let raw = self.raw.ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "causal attention completion was already released",
            )
        })?;
        NonNull::new(raw as *mut sys::sllm_completion_t).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "causal attention completion had a null opaque handle",
            )
        })
    }

    fn call_completion(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<CompletionState, RuntimeError> {
        let raw = self.raw_handle()?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut result = completion_result();
        let status = unsafe {
            match timeout {
                Some(timeout) => sys::sllm_completion_wait(
                    raw.as_ptr(),
                    timeout_millis(timeout),
                    &mut result,
                    &mut error_sink,
                ),
                None => sys::sllm_completion_query(raw.as_ptr(), &mut result, &mut error_sink),
            }
        };
        let state = completion_state(result.state)?;
        if state != CompletionState::Pending {
            self.terminal = true;
        }
        let status = RuntimeStatus::from_raw(status);
        if status == RuntimeStatus::Ok {
            return Ok(state);
        }
        if state == CompletionState::Pending
            && matches!(status, RuntimeStatus::Pending | RuntimeStatus::Timeout)
        {
            return Err(result_error(
                status.raw(),
                &error_buffer,
                error_sink.message_length,
            ));
        }
        Err(result_error(
            status.raw(),
            &error_buffer,
            error_sink.message_length,
        ))
    }
}

impl Drop for CausalAttentionCompletion {
    fn drop(&mut self) {
        let Some(raw_value) = self.raw.take() else {
            return;
        };
        let Some(raw) = NonNull::new(raw_value as *mut sys::sllm_completion_t) else {
            return;
        };
        let (status, remaining) = release_causal_completion_once(raw);
        if let Some(remaining) = remaining {
            enqueue_causal_completion_cleanup(
                remaining,
                self.context.clone(),
                self.queue.clone(),
                self._query.clone(),
                self._output.clone(),
                self.state.clone(),
                status,
            );
        }
    }
}

impl KvAppendCompletion {
    pub(crate) fn query(&mut self) -> Result<CompletionState, RuntimeError> {
        self.call_completion(None)
    }

    pub(crate) fn wait(&mut self, timeout: Duration) -> Result<CompletionState, RuntimeError> {
        self.call_completion(Some(timeout))
    }

    pub(crate) fn cancel(&mut self) -> Result<(), RuntimeError> {
        if self.terminal || self.canceled {
            return Ok(());
        }
        let raw = self.raw_handle()?;
        let state = self.state.raw_handle()?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe {
            sys::sllm_kv_state_append_cancel(state.as_ptr(), raw.as_ptr(), &mut error_sink)
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        self.canceled = true;
        Ok(())
    }

    fn raw_handle(&self) -> Result<NonNull<sys::sllm_completion_t>, RuntimeError> {
        let raw = self.raw.ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "KV append completion was already released",
            )
        })?;
        NonNull::new(raw as *mut sys::sllm_completion_t).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "KV append completion had a null opaque handle",
            )
        })
    }

    fn call_completion(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<CompletionState, RuntimeError> {
        let raw = self.raw_handle()?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut result = completion_result();
        let status = unsafe {
            match timeout {
                Some(timeout) => sys::sllm_completion_wait(
                    raw.as_ptr(),
                    timeout_millis(timeout),
                    &mut result,
                    &mut error_sink,
                ),
                None => sys::sllm_completion_query(raw.as_ptr(), &mut result, &mut error_sink),
            }
        };
        let state = completion_state(result.state)?;
        if state != CompletionState::Pending {
            self.terminal = true;
        }
        let status = RuntimeStatus::from_raw(status);
        if status == RuntimeStatus::Ok {
            return Ok(state);
        }
        if state == CompletionState::Pending
            && matches!(status, RuntimeStatus::Pending | RuntimeStatus::Timeout)
        {
            return Err(result_error(
                status.raw(),
                &error_buffer,
                error_sink.message_length,
            ));
        }
        Err(result_error(
            status.raw(),
            &error_buffer,
            error_sink.message_length,
        ))
    }
}

impl Drop for KvAppendCompletion {
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
        let (status, remaining) = release_kv_completion_once(raw);
        if let Some(remaining) = remaining {
            enqueue_kv_completion_cleanup(
                remaining,
                self.context.clone(),
                self.queue.clone(),
                self.key.clone(),
                self.value.clone(),
                self.state.clone(),
                status,
            );
        }
    }
}

fn empty_view_info() -> sys::sllm_kv_view_info_t {
    sys::sllm_kv_view_info_t {
        struct_size: size_of::<sys::sllm_kv_view_info_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_KV_VIEW_INFO_VERSION,
        reserved0: 0,
        session_id: 0,
        layer_id: 0,
        dtype: 0,
        encoding: 0,
        head_count: 0,
        head_dim: 0,
        reserved1: 0,
        capacity_tokens: 0,
        observed_length: 0,
        generation: 0,
        context_identity: 0,
        state_identity: 0,
        k_stride_elements: [0; 3],
        v_stride_elements: [0; 3],
        reserved: [0; 4],
    }
}

fn empty_append_info() -> sys::sllm_kv_append_info_t {
    sys::sllm_kv_append_info_t {
        struct_size: size_of::<sys::sllm_kv_append_info_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_KV_APPEND_INFO_VERSION,
        backend: 0,
        dispatch_id: 0,
        dispatch_count: 0,
        kernel_id: 0,
        workgroup_size_x: 0,
        grid_size_x: 0,
        start_position: 0,
        token_count: 0,
        end_position: 0,
        commit_allowed: 0,
        fallback_allowed: 0,
        fallback_used: 0,
        reserved0: 0,
        kernel_symbol: [0; 64],
        device_symbol: [0; 64],
        gcn_arch_name: [0; 64],
        reserved: [0; 8],
    }
}

fn empty_causal_attention_info() -> sys::sllm_causal_attention_dispatch_info_t {
    sys::sllm_causal_attention_dispatch_info_t {
        struct_size: size_of::<sys::sllm_causal_attention_dispatch_info_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION,
        backend: 0,
        dispatch_id: 0,
        dispatch_count: 0,
        kernel_id: 0,
        workgroup_size_x: 0,
        grid_size_x: 0,
        query_count: 0,
        start_position: 0,
        committed_kv_length: 0,
        q_heads: 0,
        kv_heads: 0,
        head_dim: 0,
        scale_denominator: 0,
        fallback_allowed: 0,
        fallback_used: 0,
        kernel_symbol: [0; 64],
        device_symbol: [0; 64],
        gcn_arch_name: [0; 64],
        reserved: [0; 8],
    }
}

fn validate_view_info(
    info: &sys::sllm_kv_view_info_t,
    context: &Context,
    raw_state: usize,
    session_id: ExecutionSessionId,
    descriptor: KvStateDescriptor,
) -> Result<(), RuntimeError> {
    let expected_stride = descriptor
        .capacity()
        .checked_mul(sys::SLLM_HIP_KV_HEAD_DIM as u64)
        .ok_or_else(|| {
            RuntimeError::local(RuntimeStatus::MetadataOverflow, "KV stride overflow")
        })?;
    if info.struct_size != size_of::<sys::sllm_kv_view_info_t>() as u32
        || info.abi_version != sys::SLLM_HIP_ABI_VERSION
        || info.info_version != sys::SLLM_HIP_KV_VIEW_INFO_VERSION
        || info.session_id != session_id.raw()
        || info.layer_id != descriptor.layer_id()
        || info.dtype != sys::SLLM_TENSOR_DTYPE_F16
        || info.encoding != sys::SLLM_TENSOR_ENCODING_UNQUANTIZED
        || info.head_count != sys::SLLM_HIP_KV_HEAD_COUNT
        || info.head_dim != sys::SLLM_HIP_KV_HEAD_DIM
        || info.capacity_tokens != descriptor.capacity()
        || info.context_identity != context.raw_handle()?.as_ptr() as usize as u64
        || info.state_identity != raw_state as u64
        || info.k_stride_elements != [expected_stride, 256, 1]
        || info.v_stride_elements != [expected_stride, 256, 1]
        || info.reserved0 != 0
        || info.reserved1 != 0
        || info.reserved != [0; 4]
    {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidKvStateDescriptor,
            "native KV snapshot metadata is not the fixed FP16 [4, capacity, 256] layout",
        ));
    }
    if info.observed_length > descriptor.capacity() {
        return Err(RuntimeError::local(
            RuntimeStatus::KvCapacityExceeded,
            "native KV snapshot length exceeds capacity",
        ));
    }
    Ok(())
}

fn validate_append_binding(binding: &TensorBinding) -> Result<(), RuntimeError> {
    let view = binding.view();
    if view.dtype() != DType::Bf16
        || view.encoding() != Encoding::Unquantized
        || view.shape().len() != 3
        || view.shape()[1] != 4
        || view.shape()[2] != 256
        || view.strides() != [4 * 256, 256, 1]
    {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidKvAppendDescriptor,
            "KV append input must be contiguous BF16 [M, 4, 256]",
        ));
    }
    Ok(())
}

fn validate_causal_attention_binding(binding: &TensorBinding) -> Result<(), RuntimeError> {
    let view = binding.view();
    if view.dtype() != DType::Bf16
        || view.encoding() != Encoding::Unquantized
        || view.shape().len() != 3
        || view.shape()[1] != 16
        || view.shape()[2] != 256
        || view.strides() != [16 * 256, 256, 1]
        || view.shape()[0] == 0
    {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidCausalAttentionDescriptor,
            "causal attention Q/output must be contiguous BF16 [M, 16, 256]",
        ));
    }
    Ok(())
}

fn validate_append_info(
    info: &sys::sllm_kv_append_info_t,
    context: &Context,
    request: KvStateAppendRequest,
    descriptor: KvStateDescriptor,
) -> Result<KvAppendEvidence, RuntimeError> {
    let target = c_string(&info.gcn_arch_name);
    let expected_target = context.expected_target();
    let expected_grid = request
        .token_count()
        .checked_mul(4)
        .and_then(|value| u32::try_from(value).ok());
    if info.struct_size != size_of::<sys::sllm_kv_append_info_t>() as u32
        || info.abi_version != sys::SLLM_HIP_ABI_VERSION
        || info.info_version != sys::SLLM_HIP_KV_APPEND_INFO_VERSION
        || info.backend != sys::SLLM_BACKEND_HIP
        || info.dispatch_id == 0
        || info.dispatch_count != 1
        || info.kernel_id != sys::SLLM_HIP_KV_KERNEL_ID_BF16_TO_F16_TRANSPOSE_V1
        || info.workgroup_size_x != sys::SLLM_HIP_KV_WORKGROUP_SIZE
        || Some(info.grid_size_x) != expected_grid
        || info.start_position != request.start_position()
        || info.token_count != request.token_count()
        || info.end_position != request.end_position()
        || info.commit_allowed != 1
        || info.fallback_allowed != 0
        || info.fallback_used != 0
        || c_string(&info.kernel_symbol) != KERNEL_SYMBOL
        || c_string(&info.device_symbol) != DEVICE_SYMBOL
        || info.reserved0 != 0
        || info.reserved != [0; 8]
        || info.end_position > descriptor.capacity()
        || expected_target.is_some_and(|expected| expected != target)
    {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidKvAppendDescriptor,
            "native KV append metadata failed exact-target/no-fallback validation",
        ));
    }
    Ok(KvAppendEvidence {
        dispatch_id: info.dispatch_id,
        dispatch_count: info.dispatch_count,
        kernel_id: info.kernel_id,
        workgroup_size_x: info.workgroup_size_x,
        grid_size_x: info.grid_size_x,
        start_position: info.start_position,
        token_count: info.token_count,
        end_position: info.end_position,
        commit_allowed: info.commit_allowed == 1,
        fallback_allowed: info.fallback_allowed == 1,
        fallback_used: info.fallback_used == 1,
        kernel_symbol: c_string(&info.kernel_symbol),
        device_symbol: c_string(&info.device_symbol),
        target,
    })
}

fn validate_causal_attention_info(
    info: &sys::sllm_causal_attention_dispatch_info_t,
    context: &Context,
    start_position: u64,
    committed_kv_length: u64,
    descriptor: KvStateDescriptor,
) -> Result<CausalAttentionEvidence, RuntimeError> {
    let target = c_string(&info.gcn_arch_name);
    let query_count = committed_kv_length
        .checked_sub(start_position)
        .ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::CausalAttentionLengthMismatch,
                "causal attention evidence range underflowed",
            )
        })?;
    let expected_grid = query_count
        .checked_mul(16)
        .and_then(|value| u32::try_from(value).ok());
    let expected_target = context.expected_target();
    if info.struct_size != size_of::<sys::sllm_causal_attention_dispatch_info_t>() as u32
        || info.abi_version != sys::SLLM_HIP_ABI_VERSION
        || info.info_version != sys::SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION
        || info.backend != sys::SLLM_BACKEND_HIP
        || info.dispatch_id == 0
        || info.dispatch_count != 1
        || info.kernel_id != sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_STABLE_SOFTMAX_V1
        || info.workgroup_size_x != sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE
        || Some(info.grid_size_x) != expected_grid
        || info.query_count != query_count
        || info.start_position != start_position
        || info.committed_kv_length != committed_kv_length
        || info.q_heads != sys::SLLM_HIP_CAUSAL_ATTENTION_Q_HEADS
        || info.kv_heads != sys::SLLM_HIP_CAUSAL_ATTENTION_KV_HEADS
        || info.head_dim != sys::SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM
        || info.scale_denominator != sys::SLLM_HIP_CAUSAL_ATTENTION_SCALE_DENOMINATOR
        || info.fallback_allowed != 0
        || info.fallback_used != 0
        || c_string(&info.kernel_symbol) != "causal_attention.stable_softmax_gqa.v1"
        || c_string(&info.device_symbol) != "sllm_causal_attention_stable_softmax_gqa_v1"
        || info.reserved != [0; 8]
        || committed_kv_length > descriptor.capacity()
        || expected_target.is_some_and(|expected| expected != target)
    {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidCausalAttentionDescriptor,
            "native causal attention metadata failed exact-target/no-fallback validation",
        ));
    }
    Ok(CausalAttentionEvidence {
        dispatch_id: info.dispatch_id,
        dispatch_count: info.dispatch_count,
        kernel_id: info.kernel_id,
        workgroup_size_x: info.workgroup_size_x,
        grid_size_x: info.grid_size_x,
        query_count: info.query_count,
        start_position: info.start_position,
        committed_kv_length: info.committed_kv_length,
        q_heads: info.q_heads,
        kv_heads: info.kv_heads,
        head_dim: info.head_dim,
        scale_denominator: info.scale_denominator,
        fallback_allowed: info.fallback_allowed != 0,
        fallback_used: info.fallback_used != 0,
        kernel_symbol: c_string(&info.kernel_symbol),
        device_symbol: c_string(&info.device_symbol),
        target,
    })
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
            "native KV completion returned an unknown state",
        )),
    }
}

fn timeout_millis(timeout: Duration) -> u32 {
    let millis = timeout.as_millis();
    if millis >= u128::from(MAX_FINITE_TIMEOUT_MS) {
        MAX_FINITE_TIMEOUT_MS
    } else {
        millis as u32
    }
}

fn c_string(value: &[std::ffi::c_char]) -> String {
    let length = value
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(value.len());
    value[..length]
        .iter()
        .map(|byte| *byte as u8 as char)
        .collect()
}

/// Exact BF16-to-FP16 round-to-nearest-even conversion used by the bounded
/// evidence oracle. It is independent of device storage.
pub fn bf16_to_f16_bits(bits: u16) -> u16 {
    f32_to_f16_bits(u32::from(bits) << 16)
}

fn f32_to_f16_bits(bits: u32) -> u16 {
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x7f_ff_ff;
    if exponent == 0xff {
        return sign | if fraction == 0 { 0x7c00 } else { 0x7e00 };
    }
    let unbiased = exponent - 127;
    if unbiased < -24 {
        return sign;
    }
    if unbiased < -14 {
        let shift = (-unbiased - 14) as u32;
        let mantissa = fraction | 0x80_0000;
        return sign | round_shift(mantissa, 13 + shift) as u16;
    }
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    let half_exponent = (unbiased + 15) as u16;
    let rounded = round_shift(fraction, 13);
    if rounded >= 0x400 {
        let next_exponent = half_exponent + 1;
        return if next_exponent >= 0x1f {
            sign | 0x7c00
        } else {
            sign | (next_exponent << 10)
        };
    }
    sign | (half_exponent << 10) | rounded as u16
}

fn round_shift(value: u32, shift: u32) -> u32 {
    let truncated = value >> shift;
    let remainder = value & ((1_u32 << shift) - 1);
    let halfway = 1_u32 << (shift - 1);
    truncated + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0))
}

/// Exact expected placement in a native [4, capacity, 256] allocation.
pub fn expected_storage_offset(
    capacity: u64,
    start_position: u64,
    token: u64,
    head: u64,
    dim: u64,
) -> Option<u64> {
    if capacity == 0 || head >= 4 || dim >= 256 {
        return None;
    }
    start_position
        .checked_add(token)
        .filter(|position| *position < capacity)
        .and_then(|position| {
            head.checked_mul(capacity)?
                .checked_mul(256)?
                .checked_add(position.checked_mul(256)?.checked_add(dim)?)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_oracle_covers_special_and_rounding_cases() {
        assert_eq!(bf16_to_f16_bits(0x0000), 0x0000);
        assert_eq!(bf16_to_f16_bits(0x8000), 0x8000);
        assert_eq!(bf16_to_f16_bits(0x7f80), 0x7c00);
        assert_eq!(bf16_to_f16_bits(0xff80), 0xfc00);
        assert_eq!(bf16_to_f16_bits(0x7fc1), 0x7e00);
        assert_eq!(bf16_to_f16_bits(0x3f80), 0x3c00);
        assert_eq!(bf16_to_f16_bits(0x3f81), 0x3c08);
    }

    #[test]
    fn placement_is_head_major_capacity_stride_and_rejects_boundaries() {
        assert_eq!(expected_storage_offset(257, 17, 3, 0, 0), Some(20 * 256));
        assert_eq!(
            expected_storage_offset(257, 17, 3, 1, 255),
            Some(257 * 256 + 20 * 256 + 255)
        );
        assert_eq!(expected_storage_offset(257, 257, 0, 0, 0), None);
        assert_eq!(expected_storage_offset(257, 0, 0, 4, 0), None);
    }

    #[test]
    fn direct_native_owners_are_not_send_or_sync() {
        static_assertions::assert_not_impl_any!(NativeKvStateOwner: Send, Sync);
        static_assertions::assert_not_impl_any!(NativeKvSnapshotOwner: Send, Sync);
        static_assertions::assert_impl_all!(KvStateResource: Send, Sync);
        static_assertions::assert_impl_all!(KvAppendCompletion: Send);
        static_assertions::assert_impl_all!(CausalAttentionCompletion: Send);
    }

    #[test]
    fn abi_layout_fields_have_expected_rust_sizes() {
        assert_eq!(size_of::<sys::sllm_kv_state_create_info_t>(), 48);
        assert_eq!(size_of::<sys::sllm_kv_view_info_t>(), 152);
        assert_eq!(size_of::<sys::sllm_kv_append_desc_t>(), 416);
        assert_eq!(size_of::<sys::sllm_kv_append_info_t>(), 304);
        assert_eq!(size_of::<sys::sllm_causal_attention_desc_t>(), 424);
        assert_eq!(size_of::<sys::sllm_causal_attention_dispatch_info_t>(), 312);
    }
}
