//! Owned HIP graph spans over already-prepared stateless operations.
//!
//! Graph capture is deliberately a native operation.  This module only
//! supplies the queue and immutable prepared-plan handles, retains every
//! provider owner for the lifetime of the graph, and wraps the aggregate
//! completion returned by graph replay.  It does not expose a raw stream or a
//! capture-begin operation through the public stateless API. The private
//! whole-decode capture guard below also retains stateful provider owners.

use core::any::Any;
use core::ffi::c_void;
use std::cell::Cell;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_hip_sys as sys;

use crate::kv_state::{CausalAttentionCompletion, KvAppendCompletion};
use crate::linear_attention::{LinearAttentionCompletion, LinearAttentionStateResource};
use crate::runtime::{
    Completion, Queue, RuntimeError, RuntimeStatus, enqueue_graph_span_cleanup, ensure_ok,
    release_graph_span_once, sink,
};
use crate::{
    PreparedElementwise, PreparedMatmul, PreparedQwen38ProjectionPack2, PreparedResidualRmsNorm,
    PreparedRmsNorm,
};

const ERROR_CAPACITY: usize = 256;

thread_local! {
    static WHOLE_DECODE_CAPTURE_QUEUE: Cell<Option<usize>> = const { Cell::new(None) };
}

struct CaptureQueueScope {
    queue: usize,
}

impl CaptureQueueScope {
    // The raw helper is used by the thread-local scope unit test and mirrors
    // the native begin path; production capture calls the checked sequence in
    // `WholeDecodeCapture::begin` directly.
    #[allow(dead_code)]
    fn enter_raw(queue: usize) -> Result<Self, RuntimeError> {
        Self::ensure_vacant()?;
        Ok(Self::activate_raw(queue))
    }

    fn ensure_vacant() -> Result<(), RuntimeError> {
        if WHOLE_DECODE_CAPTURE_QUEUE.with(|active| active.get().is_some()) {
            return Err(RuntimeError::local(
                RuntimeStatus::Busy,
                "a whole-decode capture is already active on this thread",
            ));
        }
        Ok(())
    }

    fn activate_raw(queue: usize) -> Self {
        WHOLE_DECODE_CAPTURE_QUEUE.with(|active| {
            debug_assert!(active.get().is_none());
            active.set(Some(queue));
        });
        Self { queue }
    }
}

impl Drop for CaptureQueueScope {
    fn drop(&mut self) {
        WHOLE_DECODE_CAPTURE_QUEUE.with(|active| {
            if active.get() == Some(self.queue) {
                active.set(None);
            }
        });
    }
}

pub(crate) fn whole_decode_capture_active_on(queue: &Queue) -> Result<bool, RuntimeError> {
    let queue = queue.raw_handle()?.as_ptr() as usize;
    Ok(WHOLE_DECODE_CAPTURE_QUEUE.with(|active| active.get() == Some(queue)))
}

/// A prepared stateless plan that can participate in a graph span.
///
/// The enum is crate-private so callers cannot pass arbitrary native pointers
/// into the graph API.  The public prepared-plan types remain the ownership
/// boundary; cloning a variant clones its provider's Arc-backed plan state.
#[derive(Clone)]
pub(crate) enum GraphSpanPlan {
    RmsNorm(PreparedRmsNorm),
    ResidualRmsNorm(PreparedResidualRmsNorm),
    Elementwise(PreparedElementwise),
    Matmul(PreparedMatmul),
    Qwen38ProjectionPack2(PreparedQwen38ProjectionPack2),
}

impl GraphSpanPlan {
    fn raw_handle(&self) -> *const c_void {
        match self {
            Self::RmsNorm(plan) => plan.raw_plan_handle().cast(),
            Self::ResidualRmsNorm(plan) => plan.raw_plan_handle().cast(),
            Self::Elementwise(plan) => plan.raw_plan_handle().cast(),
            Self::Matmul(plan) => plan.raw_plan_handle().cast(),
            Self::Qwen38ProjectionPack2(plan) => plan.raw_plan_handle().cast(),
        }
    }
}

#[derive(Default)]
pub(crate) struct GraphSpanOwners {
    plans: Vec<GraphSpanPlan>,
    captured: Vec<Completion>,
    command_bindings: Vec<crate::TensorBinding>,
    _control: Option<crate::TensorBinding>,
}

struct GraphSpanInner {
    // This field is released explicitly in Drop while queue and owners are
    // still alive.  Native release also performs its own in-flight safety
    // accounting when a replay completion is pending.
    raw: NonNull<sys::sllm_graph_span_t>,
    queue: Queue,
    owners: GraphSpanOwners,
    node_count: u64,
    kernel_node_count: u64,
}

// SAFETY: the native graph handle is synchronized by the public runtime and
// the retained queue/provider owners are Send + Sync.  Graph replay returns a
// distinct completion for each call, while release is serialized by the
// Arc-backed final owner in Drop.
unsafe impl Send for GraphSpanInner {}
unsafe impl Sync for GraphSpanInner {}

impl Drop for GraphSpanInner {
    fn drop(&mut self) {
        let (status, remaining) = release_graph_span_once(self.raw);
        if let Some(remaining) = remaining {
            // The native release contract leaves a BUSY/PENDING handle live.
            // Transfer that handle together with its queue and provider owners
            // to the existing bounded cleanup/reaper path; dropping either
            // here would invalidate native graph dependencies.
            enqueue_graph_span_cleanup(
                remaining,
                self.queue.clone(),
                std::mem::take(&mut self.owners),
                status,
            );
        }
    }
}

/// A reusable, request-owned graph span for prepared stateless operations.
///
/// Creation invokes the native warm/capture/instantiate path.  It does not
/// execute an operation or mutate a bound output.  Clones share the native
/// graph and its retained provider owners; the native graph is released after
/// the last clone and before those owners are dropped.
#[derive(Clone)]
pub struct GraphSpan {
    inner: Arc<GraphSpanInner>,
}

impl std::fmt::Debug for GraphSpan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GraphSpan")
            .field("node_count", &self.node_count())
            .field(
                "operation_count",
                &(self.inner.owners.plans.len() + self.inner.owners.captured.len()),
            )
            .finish_non_exhaustive()
    }
}

impl GraphSpan {
    /// Capture and instantiate a graph from prepared plans without replaying
    /// it.  The native layer validates that all plan handles belong to the
    /// supplied queue/context and pins the same dependencies independently.
    pub(crate) fn create(queue: &Queue, plans: &[GraphSpanPlan]) -> Result<Self, RuntimeError> {
        if plans.is_empty() {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                "graph span requires at least one prepared plan",
            ));
        }
        let prepared_plan_count = u64::try_from(plans.len()).map_err(|_| {
            RuntimeError::local(
                RuntimeStatus::MetadataOverflow,
                "graph span plan count does not fit the public ABI",
            )
        })?;
        let raw_handles: Vec<*const c_void> = plans.iter().map(GraphSpanPlan::raw_handle).collect();

        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_graph = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_graph_span_create(
                queue.raw_handle()?.as_ptr(),
                raw_handles.as_ptr(),
                prepared_plan_count,
                &mut raw_graph,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        let raw_graph = NonNull::new(raw_graph).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native graph span create returned a null handle on success",
            )
        })?;

        let mut node_count = 0_u64;
        let mut node_error_buffer = [0_u8; ERROR_CAPACITY];
        let mut node_error_sink = sink(&mut node_error_buffer);
        let node_status = unsafe {
            sys::sllm_graph_span_node_count(
                raw_graph.as_ptr(),
                &mut node_count,
                &mut node_error_sink,
            )
        };
        if let Err(error) = ensure_ok(
            node_status,
            &node_error_buffer,
            node_error_sink.message_length,
        ) {
            let (release_status, remaining) = release_graph_span_once(raw_graph);
            if let Some(remaining) = remaining {
                // A failed metadata query must retain the same queue and
                // prepared-plan owners as the normal Drop path.  Native
                // release may leave a graph live while a prior capture/replay
                // is still in flight.
                enqueue_graph_span_cleanup(
                    remaining,
                    queue.clone(),
                    GraphSpanOwners {
                        plans: plans.to_vec(),
                        ..Default::default()
                    },
                    release_status,
                );
            }
            return Err(error);
        }

        Ok(Self {
            inner: Arc::new(GraphSpanInner {
                raw: raw_graph,
                queue: queue.clone(),
                owners: GraphSpanOwners {
                    plans: plans.to_vec(),
                    ..Default::default()
                },
                node_count,
                kernel_node_count: node_count,
            }),
        })
    }

    /// Number of all native HIP graph nodes in the instantiated span,
    /// including kernel, memcpy, and event nodes.
    pub fn node_count(&self) -> u64 {
        self.inner.node_count
    }

    /// Number of native HIP kernel nodes in the instantiated span.
    pub fn kernel_node_count(&self) -> u64 {
        self.inner.kernel_node_count
    }

    /// Replay the graph and return its aggregate, eventless completion.
    pub(crate) fn execute(&self) -> Result<Completion, RuntimeError> {
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_completion = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_graph_span_execute(
                self.inner.raw.as_ptr(),
                &mut raw_completion,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        let raw_completion = NonNull::new(raw_completion).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native graph span execute returned a null completion on success",
            )
        })?;
        Ok(Completion::from_native_with_keepalive(
            raw_completion,
            &self.inner.queue,
            Arc::new(self.clone()),
        ))
    }

    pub fn queue(&self) -> &Queue {
        &self.inner.queue
    }

    pub(crate) fn publish_state_metadata(
        &self,
        expected_initial_position: u64,
        final_position: u64,
        successful_generations: u64,
    ) -> Result<(), RuntimeError> {
        let mut buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_graph_span_publish_state_metadata(
                self.inner.raw.as_ptr(),
                expected_initial_position,
                final_position,
                successful_generations,
                &mut error_sink,
            )
        };
        ensure_ok(status, &buffer, error_sink.message_length)
    }
}

/// Incremental capture stays on its creating thread because HIP capture and
/// the native capture owner use thread-local state. Only the finished graph
/// can cross a thread boundary.
pub(crate) struct WholeDecodeCapture {
    raw: Option<NonNull<sys::sllm_graph_span_t>>,
    queue: Queue,
    owners: GraphSpanOwners,
    _queue_scope: CaptureQueueScope,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl WholeDecodeCapture {
    pub(crate) fn begin(
        context: &crate::Context,
        queue: &Queue,
        control: &crate::TensorBinding,
    ) -> Result<Self, RuntimeError> {
        CaptureQueueScope::ensure_vacant()?;
        let queue_identity = queue.raw_handle()?.as_ptr() as usize;
        let binding = control.raw()?;
        let mut buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut buffer);
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_graph_span_begin_capture(
                context.raw_handle()?.as_ptr(),
                queue.raw_handle()?.as_ptr(),
                &binding,
                &mut raw,
                &mut error_sink,
            )
        };
        ensure_ok(status, &buffer, error_sink.message_length)?;
        let raw = NonNull::new(raw).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "whole capture returned a null graph",
            )
        })?;
        // Register only after native HIP capture is live. No wrapper may use
        // projected request/evidence ranges outside this exact queue scope.
        let queue_scope = CaptureQueueScope::activate_raw(queue_identity);
        Ok(Self {
            raw: Some(raw),
            queue: queue.clone(),
            owners: GraphSpanOwners {
                _control: Some(control.clone()),
                ..Default::default()
            },
            _queue_scope: queue_scope,
            _thread: std::marker::PhantomData,
        })
    }

    // Retain is kept for the legacy completion-transfer path.  Whole-decode
    // capture currently transfers markers through `capture_completion`, but
    // this owner-preserving path remains part of the private adapter surface.
    #[allow(dead_code)]
    pub(crate) fn retain(&mut self, mut completion: Completion) -> Result<(), RuntimeError> {
        self.owners.captured.try_reserve(1).map_err(|_| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "capture owner allocation failed",
            )
        })?;
        let result = completion.transfer_to_graph(self.raw.expect("live capture"));
        // Preserve provider owners on failure as well; abort/release handles
        // native cleanup before this vector is dropped.
        self.owners.captured.push(completion);
        result
    }

    /// Transfers a completion already retained by a core submission owner.
    /// The owner itself remains alive in `ExecutionGraphSpan`; this method only
    /// moves the native marker handle into the graph's marker list.
    pub(crate) fn capture_completion(
        &mut self,
        completion: &mut Completion,
    ) -> Result<(), RuntimeError> {
        completion.transfer_to_graph(self.raw.expect("live capture"))
    }

    /// Transfers one stateful completion whose Rust wrapper owns an opaque
    /// native handle rather than the general `Completion` wrapper.
    pub(crate) fn capture_opaque_completion(
        &mut self,
        raw: &mut Option<usize>,
    ) -> Result<(), RuntimeError> {
        let mut raw_pointer = raw.ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "stateful completion was already released",
            )
        })? as *mut sys::sllm_completion_t;
        let mut buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_graph_span_capture_marker(
                self.raw.expect("live capture").as_ptr(),
                &mut raw_pointer,
                &mut error_sink,
            )
        };
        *raw = NonNull::new(raw_pointer).map(|handle| handle.as_ptr() as usize);
        ensure_ok(status, &buffer, error_sink.message_length)?;
        if raw.is_some() {
            return Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "capture marker success did not consume its opaque handle",
            ));
        }
        Ok(())
    }

    pub(crate) fn command(
        &mut self,
        mut command: sys::sllm_graph_span_decode_command_desc_t,
        bindings: [Option<&crate::TensorBinding>; 4],
    ) -> Result<(), RuntimeError> {
        self.owners.command_bindings.try_reserve(4).map_err(|_| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "capture command owner allocation failed",
            )
        })?;
        // The private ABI uses an all-zero binding for absent operands. Its
        // fields are integer scalars, integer arrays, and nullable pointers.
        let mut raw_bindings: [sys::sllm_tensor_binding_t; 4] = unsafe { std::mem::zeroed() };
        for (index, binding) in bindings.into_iter().enumerate() {
            if let Some(binding) = binding {
                raw_bindings[index] = binding.raw()?;
                self.owners.command_bindings.push(binding.clone());
            }
        }
        command.struct_size = std::mem::size_of_val(&command) as u32;
        command.abi_version = sys::SLLM_HIP_ABI_VERSION;
        command.info_version = 1;
        command.input0 = raw_bindings[0];
        command.input1 = raw_bindings[1];
        command.output = raw_bindings[2];
        command.stop_ids = raw_bindings[3];
        let mut buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_graph_span_decode_command(
                self.raw.expect("live capture").as_ptr(),
                &command,
                &mut error_sink,
            )
        };
        ensure_ok(status, &buffer, error_sink.message_length)
    }

    pub(crate) fn bind_linear_state(
        &mut self,
        state: &LinearAttentionStateResource,
        checkpoint_conv: Option<&crate::TensorBinding>,
        checkpoint_recurrent: Option<&crate::TensorBinding>,
        token_count: u32,
        checkpoint_rows: u32,
    ) -> Result<(), RuntimeError> {
        let conv = checkpoint_conv
            .map(|binding| {
                self.owners.command_bindings.push(binding.clone());
                binding.raw()
            })
            .transpose()?;
        let recurrent = checkpoint_recurrent
            .map(|binding| {
                self.owners.command_bindings.push(binding.clone());
                binding.raw()
            })
            .transpose()?;
        let conv_ptr = conv
            .as_ref()
            .map_or(std::ptr::null(), |binding| binding as *const _);
        let recurrent_ptr = recurrent
            .as_ref()
            .map_or(std::ptr::null(), |binding| binding as *const _);
        let mut buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_graph_span_bind_linear_state(
                self.raw.expect("live capture").as_ptr(),
                state.raw_handle()?.as_ptr(),
                conv_ptr,
                recurrent_ptr,
                token_count,
                checkpoint_rows,
                &mut error_sink,
            )
        };
        ensure_ok(status, &buffer, error_sink.message_length)
    }

    pub(crate) fn select_linear_state(
        &mut self,
        state: &LinearAttentionStateResource,
        token_count: u32,
    ) -> Result<(), RuntimeError> {
        let mut buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_graph_span_select_linear_state(
                self.raw.expect("live capture").as_ptr(),
                state.raw_handle()?.as_ptr(),
                token_count,
                &mut error_sink,
            )
        };
        ensure_ok(status, &buffer, error_sink.message_length)
    }

    pub(crate) fn finish(mut self) -> Result<GraphSpan, RuntimeError> {
        let mut info = sys::sllm_graph_span_capture_info_t {
            struct_size: std::mem::size_of::<sys::sllm_graph_span_capture_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: 1,
            ..Default::default()
        };
        let mut buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut buffer);
        let raw = self.raw.expect("live capture");
        let status =
            unsafe { sys::sllm_graph_span_end_capture(raw.as_ptr(), &mut info, &mut error_sink) };
        ensure_ok(status, &buffer, error_sink.message_length)?;
        self.raw = None;
        Ok(GraphSpan {
            inner: Arc::new(GraphSpanInner {
                raw,
                queue: self.queue.clone(),
                owners: std::mem::take(&mut self.owners),
                node_count: info.actual_node_count,
                kernel_node_count: info.kernel_node_count,
            }),
        })
    }
}

impl sllm_core::ExecutionCaptureMarker for WholeDecodeCapture {
    fn capture_completion(
        &mut self,
        completion: &mut dyn Any,
    ) -> Result<(), sllm_core::ExecutionError> {
        if let Some(completion) = completion.downcast_mut::<Completion>() {
            return self
                .capture_completion(completion)
                .map_err(map_runtime_error);
        }
        if let Some(completion) = completion.downcast_mut::<KvAppendCompletion>() {
            return completion.capture_marker(self).map_err(map_runtime_error);
        }
        if let Some(completion) = completion.downcast_mut::<CausalAttentionCompletion>() {
            return completion.capture_marker(self).map_err(map_runtime_error);
        }
        if let Some(completion) = completion.downcast_mut::<LinearAttentionCompletion>() {
            return completion.capture_marker(self).map_err(map_runtime_error);
        }
        Err(sllm_core::ExecutionError::Unsupported {
            reason: "HIP capture marker received an unknown completion owner".to_owned(),
        })
    }
}

fn map_runtime_error(error: RuntimeError) -> sllm_core::ExecutionError {
    sllm_core::ExecutionError::BackendStatus {
        status: error.status().raw(),
        diagnostic: error.to_string(),
    }
}

impl Drop for WholeDecodeCapture {
    fn drop(&mut self) {
        let Some(raw) = self.raw.take() else {
            return;
        };
        let mut buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut buffer);
        // End capture before any provider owner can be released. The native
        // failed-capture path quarantines unsafe markers instead of executing
        // or reporting them as successfully completed operations.
        let mut raw_pointer = raw.as_ptr();
        unsafe {
            sys::sllm_graph_span_abort_capture(&mut raw_pointer, &mut error_sink);
        }
        let Some(raw) = NonNull::new(raw_pointer) else {
            return;
        };
        let (status, remaining) = release_graph_span_once(raw);
        if let Some(remaining) = remaining {
            enqueue_graph_span_cleanup(
                remaining,
                self.queue.clone(),
                std::mem::take(&mut self.owners),
                status,
            );
        }
    }
}

#[cfg(test)]
mod capture_scope_tests {
    use super::*;

    #[test]
    fn capture_queue_scope_is_thread_local_exact_and_drop_bounded() {
        assert!(!WHOLE_DECODE_CAPTURE_QUEUE.with(|active| active.get().is_some()));
        {
            let _scope = CaptureQueueScope::enter_raw(17).unwrap();
            assert_eq!(
                WHOLE_DECODE_CAPTURE_QUEUE.with(|active| active.get()),
                Some(17)
            );
            assert!(CaptureQueueScope::enter_raw(17).is_err());
            assert!(CaptureQueueScope::enter_raw(19).is_err());
        }
        assert!(!WHOLE_DECODE_CAPTURE_QUEUE.with(|active| active.get().is_some()));
    }
}
