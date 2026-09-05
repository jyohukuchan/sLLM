//! Owned HIP graph spans over already-prepared stateless operations.
//!
//! Graph capture is deliberately a native operation.  This module only
//! supplies the queue and immutable prepared-plan handles, retains every
//! provider owner for the lifetime of the graph, and wraps the aggregate
//! completion returned by graph replay.  It does not expose a raw stream or a
//! capture-begin operation.

use core::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_hip_sys as sys;

use crate::runtime::{
    Completion, Queue, RuntimeError, RuntimeStatus, enqueue_graph_span_cleanup, ensure_ok,
    release_graph_span_once, sink,
};
use crate::{
    PreparedElementwise, PreparedMatmul, PreparedQwen38ProjectionPack2, PreparedResidualRmsNorm,
    PreparedRmsNorm,
};

const ERROR_CAPACITY: usize = 256;

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

struct GraphSpanInner {
    // This field is released explicitly in Drop while queue and owners are
    // still alive.  Native release also performs its own in-flight safety
    // accounting when a replay completion is pending.
    raw: NonNull<sys::sllm_graph_span_t>,
    queue: Queue,
    owners: Vec<GraphSpanPlan>,
    node_count: u64,
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
            .field("operation_count", &self.inner.owners.len())
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
                    plans.to_vec(),
                    release_status,
                );
            }
            return Err(error);
        }

        Ok(Self {
            inner: Arc::new(GraphSpanInner {
                raw: raw_graph,
                queue: queue.clone(),
                owners: plans.to_vec(),
                node_count,
            }),
        })
    }

    /// Number of native HIP graph nodes in the instantiated span.
    pub fn node_count(&self) -> u64 {
        self.inner.node_count
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
}
