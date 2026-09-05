use super::*;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct DropCounts {
    buffer: Arc<AtomicUsize>,
    queue: Arc<AtomicUsize>,
    prepared: Arc<AtomicUsize>,
    graph: Arc<AtomicUsize>,
}

impl Default for DropCounts {
    fn default() -> Self {
        Self {
            buffer: Arc::new(AtomicUsize::new(0)),
            queue: Arc::new(AtomicUsize::new(0)),
            prepared: Arc::new(AtomicUsize::new(0)),
            graph: Arc::new(AtomicUsize::new(0)),
        }
    }
}

struct DropMarker(Arc<AtomicUsize>);

impl Drop for DropMarker {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

struct NoopSubmission;

impl ExecutionSubmissionAdapter for NoopSubmission {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        Ok(ExecutionState::Success)
    }

    fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        Ok(ExecutionState::Success)
    }

    fn start_output_readback(
        &mut self,
        _access: &ExecutionAdapterAccess<'_>,
        _output: &OwnedTensorBinding,
    ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
        Err(ExecutionError::Unsupported {
            reason: "graph span test does not read output".to_owned(),
        })
    }
}

struct GraphAdapter {
    native_kernel_nodes: u64,
    capture_calls: AtomicUsize,
    replay_calls: AtomicUsize,
    submit_calls: AtomicUsize,
    fallback_calls: AtomicUsize,
    capture_error: Mutex<Option<ExecutionError>>,
    replay_error: Mutex<Option<ExecutionError>>,
    drops: Arc<DropCounts>,
}

impl GraphAdapter {
    fn new(native_kernel_nodes: u64) -> (Arc<Self>, Arc<DropCounts>) {
        let drops = Arc::new(DropCounts::default());
        let adapter = Arc::new(Self {
            native_kernel_nodes,
            capture_calls: AtomicUsize::new(0),
            replay_calls: AtomicUsize::new(0),
            submit_calls: AtomicUsize::new(0),
            fallback_calls: AtomicUsize::new(0),
            capture_error: Mutex::new(None),
            replay_error: Mutex::new(None),
            drops: Arc::clone(&drops),
        });
        (adapter, drops)
    }

    fn fail_capture(&self) {
        *self.capture_error.lock().expect("capture error lock") =
            Some(ExecutionError::BackendStatus {
                status: 41,
                diagnostic: "capture failed".to_owned(),
            });
    }

    fn fail_replay(&self) {
        *self.replay_error.lock().expect("replay error lock") =
            Some(ExecutionError::BackendStatus {
                status: 42,
                diagnostic: "replay failed".to_owned(),
            });
    }
}

impl ExecutionSessionAdapter for GraphAdapter {
    fn max_transfer_bytes(&self) -> u64 {
        4096
    }

    fn supports(&self, _descriptor: &SemanticOpDescriptor) -> PrepareSupport {
        PrepareSupport::Supported
    }

    fn create_queue(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
    ) -> Result<AdapterResource, ExecutionError> {
        Ok(AdapterResource::new(DropMarker(Arc::clone(
            &self.drops.queue,
        ))))
    }

    fn allocate(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        _size_bytes: u64,
    ) -> Result<AdapterResource, ExecutionError> {
        Ok(AdapterResource::new(DropMarker(Arc::clone(
            &self.drops.buffer,
        ))))
    }

    fn prepare(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        _operation: &BoundSemanticOp,
    ) -> Result<AdapterResource, ExecutionError> {
        Ok(AdapterResource::new(DropMarker(Arc::clone(
            &self.drops.prepared,
        ))))
    }

    fn submit(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        _prepared: &PreparedOperation,
        _queue: &ExecutionQueue,
    ) -> Result<(Box<dyn ExecutionSubmissionAdapter>, DispatchEvidence), ExecutionError> {
        self.submit_calls.fetch_add(1, Ordering::Relaxed);
        Ok((Box::new(NoopSubmission), evidence(1, 1, "eager")))
    }

    fn create_graph_span(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        _queue: &ExecutionQueue,
        _operations: &[PreparedOperation],
    ) -> Result<(AdapterResource, u64), ExecutionError> {
        self.capture_calls.fetch_add(1, Ordering::Relaxed);
        if let Some(error) = self
            .capture_error
            .lock()
            .expect("capture error lock")
            .take()
        {
            return Err(error);
        }
        Ok((
            AdapterResource::new(DropMarker(Arc::clone(&self.drops.graph))),
            self.native_kernel_nodes,
        ))
    }

    fn submit_graph_span(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        _span: &ExecutionGraphSpan,
    ) -> Result<Box<dyn ExecutionSubmissionAdapter>, ExecutionError> {
        self.replay_calls.fetch_add(1, Ordering::Relaxed);
        if let Some(error) = self.replay_error.lock().expect("replay error lock").take() {
            return Err(error);
        }
        Ok(Box::new(NoopSubmission))
    }

    fn upload(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        _queue: &ExecutionQueue,
        _destination: &BufferRange,
        _bytes: Arc<[u8]>,
    ) -> Result<Box<dyn ExecutionTransferAdapter>, ExecutionError> {
        Err(ExecutionError::Unsupported {
            reason: "graph span test does not upload".to_owned(),
        })
    }

    fn readback(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        _queue: &ExecutionQueue,
        _source: &BufferRange,
    ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
        Err(ExecutionError::Unsupported {
            reason: "graph span test does not readback".to_owned(),
        })
    }

    fn shutdown(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        _deadline: Duration,
    ) -> Result<ShutdownReport, ExecutionError> {
        Ok(ShutdownReport {
            retryable_cleanup: 0,
            durable_quarantine: 0,
        })
    }
}

fn evidence(dispatch_count: u32, kernel_id: u32, symbol: &str) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: 1,
        info_version: 1,
        dispatch_id: u64::from(kernel_id),
        dispatch_count,
        kernel_id,
        workgroup_size_x: 256,
        grid_size_x: 1,
        row_count: 1,
        normalized_size: 1,
        backend: 1,
        fallback_allowed: false,
        fallback_used: false,
        kernel_symbol: symbol.to_owned(),
        device_symbol: symbol.to_owned(),
        target: "graph-test".to_owned(),
    }
}

fn operation(session: &ExecutionSession) -> (Vec<ExecutionBuffer>, PreparedOperation) {
    let activation = TensorView::contiguous(DType::Bf16, &[1, 4]).unwrap();
    let scale = TensorView::contiguous(DType::Bf16, &[4]).unwrap();
    let output = TensorView::contiguous(DType::Bf16, &[1, 4]).unwrap();
    let activation_buffer = session.allocate(activation.end_offset()).unwrap();
    let scale_buffer = session.allocate(scale.end_offset()).unwrap();
    let output_buffer = session.allocate(output.end_offset()).unwrap();
    let descriptor = Arc::new(
        SemanticOpDescriptor::new_rms_norm(
            vec![activation.clone(), scale.clone()],
            vec![output.clone()],
            1.0e-6,
            crate::RmsNormScaleMode::OffsetOne,
        )
        .unwrap(),
    );
    let bound = Arc::new(
        BoundSemanticOp::new(
            descriptor,
            vec![
                session
                    .bind(&activation_buffer, activation, AccessMode::Read)
                    .unwrap(),
                session
                    .bind(&scale_buffer, scale, AccessMode::Read)
                    .unwrap(),
            ],
            vec![
                session
                    .bind(&output_buffer, output, AccessMode::Write)
                    .unwrap(),
            ],
        )
        .unwrap(),
    );
    let prepared = session.prepare(bound).unwrap();
    (
        vec![activation_buffer, scale_buffer, output_buffer],
        prepared,
    )
}

#[test]
fn graph_span_validates_identity_counts_and_native_nodes() {
    let (adapter, _drops) = GraphAdapter::new(3);
    let adapter_for_session: Arc<dyn ExecutionSessionAdapter> = adapter.clone();
    let session = ExecutionSession::new("graph-test", adapter_for_session);
    let queue = session.create_queue().unwrap();
    let (buffers, prepared) = operation(&session);
    let logical = vec![("layer.0.norm".to_owned(), evidence(1, 1, "norm"))];

    assert!(matches!(
        session.create_graph_span(&queue, &[], &[]),
        Err(ExecutionError::InvalidRequest { .. })
    ));
    assert!(matches!(
        session.create_graph_span(&queue, std::slice::from_ref(&prepared), &[]),
        Err(ExecutionError::InvalidRequest { .. })
    ));
    assert_eq!(adapter.capture_calls.load(Ordering::Relaxed), 0);

    let graph = session
        .create_graph_span(&queue, std::slice::from_ref(&prepared), &logical)
        .unwrap();
    assert_eq!(graph.queue().id(), queue.id());
    assert_eq!(graph.operations().len(), 1);
    assert_eq!(graph.operations()[0].id(), prepared.id());
    assert_eq!(graph.native_kernel_nodes(), 3);

    let foreign = ExecutionSession::new("graph-test", GraphAdapter::new(3).0);
    let foreign_queue = foreign.create_queue().unwrap();
    assert!(matches!(
        session.create_graph_span(&foreign_queue, std::slice::from_ref(&prepared), &logical),
        Err(ExecutionError::WrongQueue { .. })
    ));
    let (_foreign_buffers, foreign_prepared) = operation(&foreign);
    assert!(matches!(
        session.create_graph_span(&queue, std::slice::from_ref(&foreign_prepared), &logical),
        Err(ExecutionError::WrongSession { .. })
    ));

    let (zero_adapter, _zero_drops) = GraphAdapter::new(0);
    let zero_session = ExecutionSession::new("graph-test", zero_adapter.clone());
    let zero_queue = zero_session.create_queue().unwrap();
    let (_zero_buffers, zero_prepared) = operation(&zero_session);
    assert!(matches!(
        zero_session.create_graph_span(&zero_queue, std::slice::from_ref(&zero_prepared), &logical),
        Err(ExecutionError::InvalidRequest { .. })
    ));
    assert_eq!(zero_adapter.capture_calls.load(Ordering::Relaxed), 1);
    drop(graph);
    drop(buffers);
}

#[test]
fn graph_capture_does_not_execute_and_backend_failures_do_not_fallback() {
    let (adapter, _drops) = GraphAdapter::new(2);
    let adapter_for_session: Arc<dyn ExecutionSessionAdapter> = adapter.clone();
    let session = ExecutionSession::new("graph-test", adapter_for_session);
    let queue = session.create_queue().unwrap();
    let (_buffers, prepared) = operation(&session);
    let logical = vec![("layer.0.norm".to_owned(), evidence(2, 7, "norm"))];

    let graph = session
        .create_graph_span(&queue, std::slice::from_ref(&prepared), &logical)
        .unwrap();
    assert_eq!(adapter.capture_calls.load(Ordering::Relaxed), 1);
    assert_eq!(adapter.submit_calls.load(Ordering::Relaxed), 0);
    assert_eq!(adapter.replay_calls.load(Ordering::Relaxed), 0);

    adapter.fail_replay();
    assert!(matches!(
        session.submit_graph_span(&graph),
        Err(ExecutionError::BackendStatus { status: 42, .. })
    ));
    assert_eq!(adapter.replay_calls.load(Ordering::Relaxed), 1);
    assert_eq!(adapter.submit_calls.load(Ordering::Relaxed), 0);
    assert_eq!(adapter.fallback_calls.load(Ordering::Relaxed), 0);

    let (failing_adapter, _failing_drops) = GraphAdapter::new(2);
    failing_adapter.fail_capture();
    let failing_session = ExecutionSession::new("graph-test", failing_adapter.clone());
    let failing_queue = failing_session.create_queue().unwrap();
    let (_failing_buffers, failing_prepared) = operation(&failing_session);
    assert!(matches!(
        failing_session.create_graph_span(
            &failing_queue,
            std::slice::from_ref(&failing_prepared),
            &logical,
        ),
        Err(ExecutionError::BackendStatus { status: 41, .. })
    ));
    assert_eq!(failing_adapter.submit_calls.load(Ordering::Relaxed), 0);
}

#[test]
fn graph_span_and_pending_submission_retain_owned_resources() {
    let (adapter, drops) = GraphAdapter::new(2);
    let (session, queue, buffers, prepared, graph) = {
        let adapter_for_session: Arc<dyn ExecutionSessionAdapter> = adapter.clone();
        let session = ExecutionSession::new("graph-test", adapter_for_session);
        let queue = session.create_queue().unwrap();
        let (buffers, prepared) = operation(&session);
        let logical = vec![("layer.0.norm".to_owned(), evidence(1, 3, "norm"))];
        let graph = session
            .create_graph_span(&queue, std::slice::from_ref(&prepared), &logical)
            .unwrap();
        (session, queue, buffers, prepared, graph)
    };
    let mut submission = session.submit_graph_span(&graph).unwrap();
    drop(graph);
    drop(prepared);
    drop(buffers);
    drop(queue);
    drop(session);
    assert_eq!(drops.graph.load(Ordering::Relaxed), 0);
    assert_eq!(drops.prepared.load(Ordering::Relaxed), 0);
    assert_eq!(drops.buffer.load(Ordering::Relaxed), 0);
    assert_eq!(drops.queue.load(Ordering::Relaxed), 0);

    assert_eq!(
        submission.wait(Duration::ZERO).unwrap(),
        ExecutionState::Success
    );
    drop(submission);
    assert_eq!(drops.graph.load(Ordering::Relaxed), 1);
    assert_eq!(drops.prepared.load(Ordering::Relaxed), 1);
    assert_eq!(drops.buffer.load(Ordering::Relaxed), 3);
    assert_eq!(drops.queue.load(Ordering::Relaxed), 1);
}
