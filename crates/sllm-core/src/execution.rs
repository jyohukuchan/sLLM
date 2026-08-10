//! Owned, backend-neutral execution-session contracts.
//!
//! This module is deliberately separate from the Phase 1 `Backend` control
//! plane.  A semantic descriptor alone cannot identify device storage or hold
//! the resources needed by an asynchronous completion, so numerical backends
//! enter here through an owned session and binding graph.

use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::{AccessMode, SemanticOpDescriptor, SemanticOpKind, TensorView};

static NEXT_EXECUTION_ID: AtomicU64 = AtomicU64::new(1);

fn next_execution_id() -> u64 {
    loop {
        let id = NEXT_EXECUTION_ID.fetch_add(1, Ordering::Relaxed);
        if id != 0 {
            return id;
        }
    }
}

macro_rules! execution_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        #[repr(transparent)]
        pub struct $name(u64);

        impl $name {
            const fn new(raw: u64) -> Self {
                Self(raw)
            }

            pub const fn raw(self) -> u64 {
                self.0
            }
        }
    };
}

execution_id!(ExecutionSessionId);
execution_id!(ExecutionBufferId);
execution_id!(ExecutionQueueId);
execution_id!(PreparedOperationId);

/// Exact device selection supplied when opening an owned backend session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionSessionRequest {
    device_index: u32,
    expected_target: String,
}

impl ExecutionSessionRequest {
    pub fn new(
        device_index: u32,
        expected_target: impl Into<String>,
    ) -> Result<Self, ExecutionError> {
        let expected_target = expected_target.into();
        if expected_target.is_empty()
            || !expected_target.is_ascii()
            || expected_target.as_bytes().contains(&0)
        {
            return Err(ExecutionError::InvalidRequest {
                reason: "expected target must be non-empty ASCII without NUL".to_owned(),
            });
        }
        Ok(Self {
            device_index,
            expected_target,
        })
    }

    pub const fn device_index(&self) -> u32 {
        self.device_index
    }

    pub fn expected_target(&self) -> &str {
        &self.expected_target
    }
}

/// Preparation support is intentionally narrower than a numerical-result
/// promise.  It only says whether a session can accept the descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PrepareSupport {
    Supported,
    Unsupported { reason: String },
}

/// Terminal or pending state shared by submissions and transfers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionState {
    Pending,
    Success,
    Failure,
}

/// Backend-neutral dispatch metadata returned only after a successful submit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DispatchEvidence {
    pub abi_version: u32,
    pub info_version: u32,
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
    pub row_count: u64,
    pub normalized_size: u64,
    pub backend: u32,
    pub fallback_allowed: bool,
    pub fallback_used: bool,
    pub kernel_symbol: String,
    pub device_symbol: String,
    pub target: String,
}

/// Cleanup observations returned from an explicitly closed execution session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShutdownReport {
    pub retryable_cleanup: usize,
    pub durable_quarantine: usize,
}

/// Errors in the owned execution boundary.  Backend diagnostics are copied
/// into owned strings so no C ABI or borrowed asynchronous state leaks out.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionError {
    ExecutionUnavailable {
        backend: &'static str,
        reason: String,
    },
    Unsupported {
        reason: String,
    },
    InvalidRequest {
        reason: String,
    },
    WrongBackend {
        expected: &'static str,
        actual: &'static str,
    },
    WrongSession {
        expected: ExecutionSessionId,
        actual: ExecutionSessionId,
    },
    WrongQueue {
        expected: ExecutionSessionId,
        actual: ExecutionSessionId,
    },
    DescriptorBindingMismatch {
        role: &'static str,
    },
    OutOfBounds {
        buffer: ExecutionBufferId,
        end_offset: u64,
        size_bytes: u64,
    },
    AccessViolation {
        role: &'static str,
        required: AccessMode,
        actual: AccessMode,
    },
    AliasOverlap {
        left: &'static str,
        right: &'static str,
    },
    InvalidRange {
        reason: String,
    },
    Busy,
    NotReady,
    Closing,
    CleanupQuarantined,
    BackendStatus {
        status: u32,
        diagnostic: String,
    },
    AsyncFailure {
        status: u32,
        diagnostic: String,
    },
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExecutionUnavailable { backend, reason } => {
                write!(
                    formatter,
                    "backend {backend} execution is unavailable: {reason}"
                )
            }
            Self::Unsupported { reason }
            | Self::InvalidRequest { reason }
            | Self::InvalidRange { reason } => formatter.write_str(reason),
            Self::WrongBackend { expected, actual } => {
                write!(
                    formatter,
                    "resource backend {actual} does not match {expected}"
                )
            }
            Self::WrongSession { expected, actual } => write!(
                formatter,
                "resource session {} does not match session {}",
                actual.raw(),
                expected.raw()
            ),
            Self::WrongQueue { expected, actual } => write!(
                formatter,
                "queue session {} does not match session {}",
                actual.raw(),
                expected.raw()
            ),
            Self::DescriptorBindingMismatch { role } => {
                write!(
                    formatter,
                    "{role} binding does not exactly match its descriptor view"
                )
            }
            Self::OutOfBounds {
                buffer,
                end_offset,
                size_bytes,
            } => write!(
                formatter,
                "buffer {} interval ends at {end_offset}, beyond {size_bytes}",
                buffer.raw()
            ),
            Self::AccessViolation {
                role,
                required,
                actual,
            } => write!(
                formatter,
                "{role} requires {required:?} access, binding has {actual:?}"
            ),
            Self::AliasOverlap { left, right } => {
                write!(formatter, "RMSNorm {left} and {right} bindings overlap")
            }
            Self::Busy => formatter.write_str("execution resource is busy"),
            Self::NotReady => formatter.write_str("execution completion is not ready"),
            Self::Closing => formatter.write_str("execution session is closing"),
            Self::CleanupQuarantined => {
                formatter.write_str("execution cleanup entered durable quarantine")
            }
            Self::BackendStatus { status, diagnostic } => {
                write!(formatter, "backend status {status}: {diagnostic}")
            }
            Self::AsyncFailure { status, diagnostic } => {
                write!(
                    formatter,
                    "asynchronous backend failure {status}: {diagnostic}"
                )
            }
        }
    }
}

impl std::error::Error for ExecutionError {}

/// Type-erased, adapter-owned resource payload.  It is intentionally only
/// constructible by backend adapters and is never exposed by public resource
/// wrappers.
pub struct AdapterResource {
    payload: Arc<dyn Any + Send + Sync>,
}

impl AdapterResource {
    pub fn new<T>(payload: T) -> Self
    where
        T: Any + Send + Sync,
    {
        Self {
            payload: Arc::new(payload),
        }
    }
}

/// Backend implementation hook for an owned execution session.  It may only
/// obtain its opaque resources through the checked downcast accessors on the
/// session passed to each method.
pub trait ExecutionSessionAdapter: Send + Sync {
    /// Maximum byte count accepted by one H2D or D2H transfer.
    fn max_transfer_bytes(&self) -> u64;

    fn supports(&self, descriptor: &SemanticOpDescriptor) -> PrepareSupport;

    fn create_queue(
        &self,
        access: &ExecutionAdapterAccess<'_>,
    ) -> Result<AdapterResource, ExecutionError>;

    fn allocate(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        size_bytes: u64,
    ) -> Result<AdapterResource, ExecutionError>;

    fn prepare(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        operation: &BoundSemanticOp,
    ) -> Result<AdapterResource, ExecutionError>;

    fn submit(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        prepared: &PreparedOperation,
        queue: &ExecutionQueue,
    ) -> Result<(Box<dyn ExecutionSubmissionAdapter>, DispatchEvidence), ExecutionError>;

    fn upload(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        queue: &ExecutionQueue,
        destination: &BufferRange,
        bytes: Arc<[u8]>,
    ) -> Result<Box<dyn ExecutionTransferAdapter>, ExecutionError>;

    fn readback(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        queue: &ExecutionQueue,
        source: &BufferRange,
    ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError>;

    fn shutdown(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        deadline: Duration,
    ) -> Result<ShutdownReport, ExecutionError>;
}

/// Adapter-owned mutable submission state.  It is intentionally `Send` but
/// not `Sync`; core exposes it only through the single-owner `Submission`.
pub trait ExecutionSubmissionAdapter: Send {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError>;
    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError>;
    fn start_output_readback(
        &mut self,
        access: &ExecutionAdapterAccess<'_>,
        output: &OwnedTensorBinding,
    ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError>;
}

/// Adapter-owned mutable transfer state.
pub trait ExecutionTransferAdapter: Send {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError>;
    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError>;
}

/// Adapter-owned mutable D2H readback state.
pub trait ExecutionReadbackAdapter: Send {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError>;
    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError>;
    fn read_into(&mut self, destination: &mut [u8]) -> Result<u64, ExecutionError>;
}

struct ExecutionSessionState {
    id: ExecutionSessionId,
    backend: &'static str,
    adapter: Arc<dyn ExecutionSessionAdapter>,
    closing: AtomicBool,
    // `supports` is a control-plane callback, but it must share an admission
    // boundary with shutdown.  Holding this mutex across the callback gives
    // both operations a total order: an admitted support query completes
    // before shutdown can establish closing, and a shutdown that establishes
    // closing prevents a later adapter call.
    adapter_admission: Mutex<()>,
}

/// An owned backend execution context.  Its public API never exposes a
/// backend native handle, borrowed host pointer, or asynchronous raw state.
#[derive(Clone)]
pub struct ExecutionSession {
    state: Arc<ExecutionSessionState>,
}

impl fmt::Debug for ExecutionSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutionSession")
            .field("id", &self.id())
            .field("backend", &self.backend_name())
            .finish_non_exhaustive()
    }
}

impl ExecutionSession {
    /// Backend adapters create sessions through this constructor.  The core
    /// owns all subsequent public wrappers and identity checks.
    pub fn new(backend: &'static str, adapter: Arc<dyn ExecutionSessionAdapter>) -> Self {
        Self {
            state: Arc::new(ExecutionSessionState {
                id: ExecutionSessionId::new(next_execution_id()),
                backend,
                adapter,
                closing: AtomicBool::new(false),
                adapter_admission: Mutex::new(()),
            }),
        }
    }

    pub fn id(&self) -> ExecutionSessionId {
        self.state.id
    }

    pub fn backend_name(&self) -> &'static str {
        self.state.backend
    }

    pub fn max_transfer_bytes(&self) -> Result<u64, ExecutionError> {
        self.ensure_open()?;
        let limit = self.state.adapter.max_transfer_bytes();
        if limit == 0 {
            return Err(ExecutionError::InvalidRange {
                reason: "backend transfer limit must be non-zero".to_owned(),
            });
        }
        Ok(limit)
    }

    pub fn supports(&self, descriptor: &SemanticOpDescriptor) -> PrepareSupport {
        let _admission = match self.state.adapter_admission.lock() {
            Ok(guard) => guard,
            Err(_) => {
                return PrepareSupport::Unsupported {
                    reason: "execution session lifecycle is unavailable".to_owned(),
                };
            }
        };
        if self.state.closing.load(Ordering::Acquire) {
            return PrepareSupport::Unsupported {
                reason: "execution session is closing".to_owned(),
            };
        }
        self.state.adapter.supports(descriptor)
    }

    pub fn create_queue(&self) -> Result<ExecutionQueue, ExecutionError> {
        self.ensure_open()?;
        let access = ExecutionAdapterAccess { session: self };
        let resource = self.state.adapter.create_queue(&access)?;
        Ok(ExecutionQueue {
            state: Arc::clone(&self.state),
            id: ExecutionQueueId::new(next_execution_id()),
            payload: resource.payload,
        })
    }

    pub fn allocate(&self, size_bytes: u64) -> Result<ExecutionBuffer, ExecutionError> {
        self.ensure_open()?;
        if size_bytes == 0 {
            return Err(ExecutionError::InvalidRange {
                reason: "execution buffer size must be non-zero".to_owned(),
            });
        }
        let access = ExecutionAdapterAccess { session: self };
        let resource = self.state.adapter.allocate(&access, size_bytes)?;
        Ok(ExecutionBuffer {
            state: Arc::clone(&self.state),
            id: ExecutionBufferId::new(next_execution_id()),
            size_bytes,
            payload: resource.payload,
        })
    }

    pub fn bind(
        &self,
        buffer: &ExecutionBuffer,
        view: TensorView,
        access: AccessMode,
    ) -> Result<OwnedTensorBinding, ExecutionError> {
        self.ensure_open()?;
        self.ensure_buffer(buffer)?;
        ensure_view_in_bounds(buffer, &view)?;
        Ok(OwnedTensorBinding {
            buffer: buffer.clone(),
            view,
            access,
        })
    }

    pub fn prepare(
        &self,
        operation: Arc<BoundSemanticOp>,
    ) -> Result<PreparedOperation, ExecutionError> {
        self.ensure_open()?;
        self.ensure_operation(&operation)?;
        let access = ExecutionAdapterAccess { session: self };
        let resource = self.state.adapter.prepare(&access, &operation)?;
        Ok(PreparedOperation {
            state: Arc::clone(&self.state),
            id: PreparedOperationId::new(next_execution_id()),
            operation,
            payload: resource.payload,
        })
    }

    pub fn submit(
        &self,
        prepared: &PreparedOperation,
        queue: &ExecutionQueue,
    ) -> Result<Submission, ExecutionError> {
        self.ensure_open()?;
        self.ensure_prepared(prepared)?;
        self.ensure_queue(queue)?;
        let access = ExecutionAdapterAccess { session: self };
        let (inner, dispatch) = self.state.adapter.submit(&access, prepared, queue)?;
        Ok(Submission {
            state: Arc::clone(&self.state),
            prepared: prepared.clone(),
            queue: queue.clone(),
            dispatch,
            completion_state: ExecutionState::Pending,
            inner,
        })
    }

    pub fn upload(
        &self,
        queue: &ExecutionQueue,
        destination: BufferRange,
        bytes: Arc<[u8]>,
    ) -> Result<Transfer, ExecutionError> {
        self.ensure_open()?;
        self.ensure_queue(queue)?;
        self.ensure_buffer(destination.buffer())?;
        if bytes.is_empty() || destination.size_bytes() != bytes.len() as u64 {
            return Err(ExecutionError::InvalidRange {
                reason: "upload range must be non-zero and exactly match the owned byte payload"
                    .to_owned(),
            });
        }
        if destination.size_bytes() > self.max_transfer_bytes()? {
            return Err(ExecutionError::InvalidRange {
                reason: "upload range exceeds the backend transfer limit".to_owned(),
            });
        }
        let inner = self.state.adapter.upload(
            &ExecutionAdapterAccess { session: self },
            queue,
            &destination,
            Arc::clone(&bytes),
        )?;
        Ok(Transfer {
            state: Arc::clone(&self.state),
            queue: queue.clone(),
            destination,
            bytes,
            completion_state: ExecutionState::Pending,
            inner,
        })
    }

    pub fn readback(
        &self,
        queue: &ExecutionQueue,
        source: BufferRange,
    ) -> Result<BufferReadback, ExecutionError> {
        self.ensure_open()?;
        self.ensure_queue(queue)?;
        self.ensure_buffer(source.buffer())?;
        if source.size_bytes() > self.max_transfer_bytes()? {
            return Err(ExecutionError::InvalidRange {
                reason: "readback range exceeds the backend transfer limit".to_owned(),
            });
        }
        let inner = self.state.adapter.readback(
            &ExecutionAdapterAccess { session: self },
            queue,
            &source,
        )?;
        Ok(BufferReadback {
            state: Arc::clone(&self.state),
            queue: queue.clone(),
            source,
            completion_state: ExecutionState::Pending,
            inner,
        })
    }

    /// Close the session before asking the backend to drain its owned work.
    /// A failure remains fail-closed: reopening the same session is forbidden.
    pub fn shutdown(&self, deadline: Duration) -> Result<ShutdownReport, ExecutionError> {
        let _admission = self
            .state
            .adapter_admission
            .lock()
            .map_err(|_| ExecutionError::Closing)?;
        self.state.closing.store(true, Ordering::Release);
        let access = ExecutionAdapterAccess { session: self };
        self.state.adapter.shutdown(&access, deadline)
    }

    fn downcast_buffer_payload<'a, T: Any + Send + Sync>(
        &self,
        buffer: &'a ExecutionBuffer,
    ) -> Result<&'a T, ExecutionError> {
        self.ensure_buffer(buffer)?;
        buffer
            .payload
            .downcast_ref::<T>()
            .ok_or(ExecutionError::WrongBackend {
                expected: self.backend_name(),
                actual: buffer.backend_name(),
            })
    }

    fn downcast_queue_payload<'a, T: Any + Send + Sync>(
        &self,
        queue: &'a ExecutionQueue,
    ) -> Result<&'a T, ExecutionError> {
        self.ensure_queue(queue)?;
        queue
            .payload
            .downcast_ref::<T>()
            .ok_or(ExecutionError::WrongBackend {
                expected: self.backend_name(),
                actual: queue.backend_name(),
            })
    }

    fn downcast_prepared_payload<'a, T: Any + Send + Sync>(
        &self,
        prepared: &'a PreparedOperation,
    ) -> Result<&'a T, ExecutionError> {
        self.ensure_prepared(prepared)?;
        prepared
            .payload
            .downcast_ref::<T>()
            .ok_or(ExecutionError::WrongBackend {
                expected: self.backend_name(),
                actual: prepared.backend_name(),
            })
    }

    fn ensure_open(&self) -> Result<(), ExecutionError> {
        if self.state.closing.load(Ordering::Acquire) {
            Err(ExecutionError::Closing)
        } else {
            Ok(())
        }
    }

    fn ensure_buffer(&self, buffer: &ExecutionBuffer) -> Result<(), ExecutionError> {
        ensure_identity(
            self.backend_name(),
            self.id(),
            buffer.backend_name(),
            buffer.session_id(),
        )
    }

    fn ensure_queue(&self, queue: &ExecutionQueue) -> Result<(), ExecutionError> {
        match ensure_identity(
            self.backend_name(),
            self.id(),
            queue.backend_name(),
            queue.session_id(),
        ) {
            Err(ExecutionError::WrongSession { expected, actual }) => {
                Err(ExecutionError::WrongQueue { expected, actual })
            }
            result => result,
        }
    }

    fn ensure_operation(&self, operation: &BoundSemanticOp) -> Result<(), ExecutionError> {
        ensure_identity(
            self.backend_name(),
            self.id(),
            operation.backend_name,
            operation.session_id,
        )
    }

    fn ensure_prepared(&self, prepared: &PreparedOperation) -> Result<(), ExecutionError> {
        ensure_identity(
            self.backend_name(),
            self.id(),
            prepared.backend_name(),
            prepared.session_id(),
        )
    }
}

/// Core-issued access to adapter payloads.  It is created only while a
/// backend adapter callback is executing, so external callers cannot unwrap
/// the opaque public session resources into backend-native objects.
pub struct ExecutionAdapterAccess<'a> {
    session: &'a ExecutionSession,
}

impl ExecutionAdapterAccess<'_> {
    pub fn session_id(&self) -> ExecutionSessionId {
        self.session.id()
    }

    pub fn backend_name(&self) -> &'static str {
        self.session.backend_name()
    }

    pub fn downcast_buffer_payload<'a, T: Any + Send + Sync>(
        &self,
        buffer: &'a ExecutionBuffer,
    ) -> Result<&'a T, ExecutionError> {
        self.session.downcast_buffer_payload(buffer)
    }

    pub fn downcast_queue_payload<'a, T: Any + Send + Sync>(
        &self,
        queue: &'a ExecutionQueue,
    ) -> Result<&'a T, ExecutionError> {
        self.session.downcast_queue_payload(queue)
    }

    pub fn downcast_prepared_payload<'a, T: Any + Send + Sync>(
        &self,
        prepared: &'a PreparedOperation,
    ) -> Result<&'a T, ExecutionError> {
        self.session.downcast_prepared_payload(prepared)
    }
}

fn ensure_identity(
    expected_backend: &'static str,
    expected_session: ExecutionSessionId,
    actual_backend: &'static str,
    actual_session: ExecutionSessionId,
) -> Result<(), ExecutionError> {
    if expected_backend != actual_backend {
        return Err(ExecutionError::WrongBackend {
            expected: expected_backend,
            actual: actual_backend,
        });
    }
    if expected_session != actual_session {
        return Err(ExecutionError::WrongSession {
            expected: expected_session,
            actual: actual_session,
        });
    }
    Ok(())
}

/// Opaque, session-owned device allocation.
#[derive(Clone)]
pub struct ExecutionBuffer {
    state: Arc<ExecutionSessionState>,
    id: ExecutionBufferId,
    size_bytes: u64,
    payload: Arc<dyn Any + Send + Sync>,
}

impl fmt::Debug for ExecutionBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutionBuffer")
            .field("id", &self.id)
            .field("session_id", &self.session_id())
            .field("size_bytes", &self.size_bytes)
            .finish_non_exhaustive()
    }
}

impl ExecutionBuffer {
    pub const fn id(&self) -> ExecutionBufferId {
        self.id
    }

    pub fn session_id(&self) -> ExecutionSessionId {
        self.state.id
    }

    pub fn backend_name(&self) -> &'static str {
        self.state.backend
    }

    pub const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub fn range(&self, offset_bytes: u64, size_bytes: u64) -> Result<BufferRange, ExecutionError> {
        BufferRange::new(self.clone(), offset_bytes, size_bytes)
    }
}

/// Opaque, session-owned execution queue.
#[derive(Clone)]
pub struct ExecutionQueue {
    state: Arc<ExecutionSessionState>,
    id: ExecutionQueueId,
    payload: Arc<dyn Any + Send + Sync>,
}

impl fmt::Debug for ExecutionQueue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutionQueue")
            .field("id", &self.id)
            .field("session_id", &self.session_id())
            .finish_non_exhaustive()
    }
}

impl ExecutionQueue {
    pub const fn id(&self) -> ExecutionQueueId {
        self.id
    }

    pub fn session_id(&self) -> ExecutionSessionId {
        self.state.id
    }

    pub fn backend_name(&self) -> &'static str {
        self.state.backend
    }
}

/// A checked half-open byte interval in one owned execution buffer.
#[derive(Clone, Debug)]
pub struct BufferRange {
    buffer: ExecutionBuffer,
    offset_bytes: u64,
    size_bytes: u64,
}

impl BufferRange {
    pub fn new(
        buffer: ExecutionBuffer,
        offset_bytes: u64,
        size_bytes: u64,
    ) -> Result<Self, ExecutionError> {
        if size_bytes == 0 {
            return Err(ExecutionError::InvalidRange {
                reason: "buffer range must be non-zero".to_owned(),
            });
        }
        let end_offset =
            offset_bytes
                .checked_add(size_bytes)
                .ok_or_else(|| ExecutionError::InvalidRange {
                    reason: "buffer range end overflowed u64".to_owned(),
                })?;
        if end_offset > buffer.size_bytes() {
            return Err(ExecutionError::OutOfBounds {
                buffer: buffer.id(),
                end_offset,
                size_bytes: buffer.size_bytes(),
            });
        }
        Ok(Self {
            buffer,
            offset_bytes,
            size_bytes,
        })
    }

    pub fn buffer(&self) -> &ExecutionBuffer {
        &self.buffer
    }

    pub const fn offset_bytes(&self) -> u64 {
        self.offset_bytes
    }

    pub const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub const fn end_offset(&self) -> u64 {
        self.offset_bytes + self.size_bytes
    }
}

/// One owned buffer/view/access binding.  Construction is only through its
/// session, and `BoundSemanticOp` repeats all checks defensively.
#[derive(Clone, Debug)]
pub struct OwnedTensorBinding {
    buffer: ExecutionBuffer,
    view: TensorView,
    access: AccessMode,
}

impl OwnedTensorBinding {
    pub fn buffer(&self) -> &ExecutionBuffer {
        &self.buffer
    }

    pub fn view(&self) -> &TensorView {
        &self.view
    }

    pub const fn access(&self) -> AccessMode {
        self.access
    }
}

/// A semantic operation paired with the exact owned storage bindings it will
/// use.  This is the point where descriptor-only alias uncertainty becomes a
/// concrete buffer-identity validation.
#[derive(Clone, Debug)]
pub struct BoundSemanticOp {
    descriptor: Arc<SemanticOpDescriptor>,
    inputs: Vec<OwnedTensorBinding>,
    outputs: Vec<OwnedTensorBinding>,
    backend_name: &'static str,
    session_id: ExecutionSessionId,
}

impl BoundSemanticOp {
    pub fn new(
        descriptor: Arc<SemanticOpDescriptor>,
        inputs: Vec<OwnedTensorBinding>,
        outputs: Vec<OwnedTensorBinding>,
    ) -> Result<Self, ExecutionError> {
        descriptor
            .validate()
            .map_err(|error| ExecutionError::Unsupported {
                reason: format!("invalid semantic descriptor: {error}"),
            })?;
        if inputs.len() != descriptor.inputs().len() || outputs.len() != descriptor.outputs().len()
        {
            return Err(ExecutionError::DescriptorBindingMismatch {
                role: "RMSNorm binding arity",
            });
        }
        let first = inputs.first().or_else(|| outputs.first()).ok_or(
            ExecutionError::DescriptorBindingMismatch {
                role: "operation has no bindings",
            },
        )?;
        let backend_name = first.buffer.backend_name();
        let session_id = first.buffer.session_id();

        for (index, binding) in inputs.iter().enumerate() {
            validate_binding_identity(backend_name, session_id, binding)?;
            ensure_view_in_bounds(&binding.buffer, &binding.view)?;
            if binding.view != descriptor.inputs()[index] {
                return Err(ExecutionError::DescriptorBindingMismatch {
                    role: input_role(descriptor.kind(), index),
                });
            }
            if !binding.access.permits_read() {
                return Err(ExecutionError::AccessViolation {
                    role: input_role(descriptor.kind(), index),
                    required: AccessMode::Read,
                    actual: binding.access,
                });
            }
        }
        for (index, binding) in outputs.iter().enumerate() {
            validate_binding_identity(backend_name, session_id, binding)?;
            ensure_view_in_bounds(&binding.buffer, &binding.view)?;
            if binding.view != descriptor.outputs()[index] {
                return Err(ExecutionError::DescriptorBindingMismatch {
                    role: output_role(descriptor.kind(), index),
                });
            }
            if !binding.access.permits_write() {
                return Err(ExecutionError::AccessViolation {
                    role: output_role(descriptor.kind(), index),
                    required: AccessMode::Write,
                    actual: binding.access,
                });
            }
        }

        if descriptor.kind() == SemanticOpKind::RmsNorm {
            validate_rmsnorm_nonoverlap(&inputs, &outputs)?;
        }

        Ok(Self {
            descriptor,
            inputs,
            outputs,
            backend_name,
            session_id,
        })
    }

    pub fn descriptor(&self) -> &Arc<SemanticOpDescriptor> {
        &self.descriptor
    }

    pub fn inputs(&self) -> &[OwnedTensorBinding] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[OwnedTensorBinding] {
        &self.outputs
    }

    pub fn session_id(&self) -> ExecutionSessionId {
        self.session_id
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend_name
    }
}

fn input_role(kind: SemanticOpKind, index: usize) -> &'static str {
    match (kind, index) {
        (SemanticOpKind::RmsNorm, 0) => "RMSNorm activation",
        (SemanticOpKind::RmsNorm, 1) => "RMSNorm raw scale",
        _ => "input",
    }
}

fn output_role(kind: SemanticOpKind, index: usize) -> &'static str {
    match (kind, index) {
        (SemanticOpKind::RmsNorm, 0) => "RMSNorm output",
        _ => "output",
    }
}

fn validate_binding_identity(
    expected_backend: &'static str,
    expected_session: ExecutionSessionId,
    binding: &OwnedTensorBinding,
) -> Result<(), ExecutionError> {
    ensure_identity(
        expected_backend,
        expected_session,
        binding.buffer.backend_name(),
        binding.buffer.session_id(),
    )
}

fn ensure_view_in_bounds(
    buffer: &ExecutionBuffer,
    view: &TensorView,
) -> Result<(), ExecutionError> {
    if view.end_offset() > buffer.size_bytes() {
        return Err(ExecutionError::OutOfBounds {
            buffer: buffer.id(),
            end_offset: view.end_offset(),
            size_bytes: buffer.size_bytes(),
        });
    }
    Ok(())
}

fn validate_rmsnorm_nonoverlap(
    inputs: &[OwnedTensorBinding],
    outputs: &[OwnedTensorBinding],
) -> Result<(), ExecutionError> {
    let entries = [
        ("activation", &inputs[0]),
        ("raw scale", &inputs[1]),
        ("output", &outputs[0]),
    ];
    for left_index in 0..entries.len() {
        for right_index in left_index + 1..entries.len() {
            let (left_name, left) = entries[left_index];
            let (right_name, right) = entries[right_index];
            if left.buffer.id() == right.buffer.id()
                && intervals_overlap(
                    left.view.byte_offset(),
                    left.view.end_offset(),
                    right.view.byte_offset(),
                    right.view.end_offset(),
                )
            {
                return Err(ExecutionError::AliasOverlap {
                    left: left_name,
                    right: right_name,
                });
            }
        }
    }
    Ok(())
}

fn intervals_overlap(left_start: u64, left_end: u64, right_start: u64, right_end: u64) -> bool {
    left_start < right_end && right_start < left_end
}

/// An opaque prepared operation.  It retains the complete bound semantic
/// graph until both the prepared operation and all derived submissions drop.
#[derive(Clone)]
pub struct PreparedOperation {
    state: Arc<ExecutionSessionState>,
    id: PreparedOperationId,
    operation: Arc<BoundSemanticOp>,
    payload: Arc<dyn Any + Send + Sync>,
}

impl fmt::Debug for PreparedOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedOperation")
            .field("id", &self.id)
            .field("session_id", &self.session_id())
            .field("kind", &self.operation.descriptor().kind())
            .finish_non_exhaustive()
    }
}

impl PreparedOperation {
    pub const fn id(&self) -> PreparedOperationId {
        self.id
    }

    pub fn session_id(&self) -> ExecutionSessionId {
        self.state.id
    }

    pub fn backend_name(&self) -> &'static str {
        self.state.backend
    }

    pub fn operation(&self) -> &Arc<BoundSemanticOp> {
        &self.operation
    }
}

/// A single-observer asynchronous submission.  It is `Send` but deliberately
/// not `Sync`: polling, waiting, and starting readback all require `&mut self`.
pub struct Submission {
    state: Arc<ExecutionSessionState>,
    prepared: PreparedOperation,
    queue: ExecutionQueue,
    dispatch: DispatchEvidence,
    completion_state: ExecutionState,
    inner: Box<dyn ExecutionSubmissionAdapter>,
}

impl fmt::Debug for Submission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Submission")
            .field("session_id", &self.state.id)
            .field("prepared", &self.prepared.id())
            .field("queue", &self.queue.id())
            .field("completion_state", &self.completion_state)
            .finish_non_exhaustive()
    }
}

impl Submission {
    pub fn dispatch(&self) -> &DispatchEvidence {
        &self.dispatch
    }

    pub const fn state(&self) -> ExecutionState {
        self.completion_state
    }

    pub fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        let state = self.inner.query()?;
        self.completion_state = state;
        Ok(state)
    }

    pub fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        let state = self.inner.wait(timeout)?;
        self.completion_state = state;
        Ok(state)
    }

    pub fn start_output_readback(
        &mut self,
        output_index: usize,
    ) -> Result<Readback, ExecutionError> {
        if self.completion_state != ExecutionState::Success {
            return Err(ExecutionError::NotReady);
        }
        let output = self
            .prepared
            .operation()
            .outputs()
            .get(output_index)
            .ok_or_else(|| ExecutionError::InvalidRange {
                reason: "output binding index is outside the prepared operation".to_owned(),
            })?
            .clone();
        let session = ExecutionSession {
            state: Arc::clone(&self.state),
        };
        let access = ExecutionAdapterAccess { session: &session };
        let inner = self.inner.start_output_readback(&access, &output)?;
        Ok(Readback {
            state: Arc::clone(&self.state),
            queue: self.queue.clone(),
            output,
            completion_state: ExecutionState::Pending,
            inner,
        })
    }
}

/// An asynchronous owned upload.  The source byte slice is kept alive until
/// its terminal state is observed or its adapter cleanup takes ownership.
pub struct Transfer {
    state: Arc<ExecutionSessionState>,
    queue: ExecutionQueue,
    destination: BufferRange,
    bytes: Arc<[u8]>,
    completion_state: ExecutionState,
    inner: Box<dyn ExecutionTransferAdapter>,
}

impl fmt::Debug for Transfer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Transfer")
            .field("session_id", &self.state.id)
            .field("queue", &self.queue.id())
            .field("destination", &self.destination)
            .field("source_bytes", &self.bytes.len())
            .field("completion_state", &self.completion_state)
            .finish_non_exhaustive()
    }
}

impl Transfer {
    pub const fn state(&self) -> ExecutionState {
        self.completion_state
    }

    pub fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        let state = self.inner.query()?;
        self.completion_state = state;
        Ok(state)
    }

    pub fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        let state = self.inner.wait(timeout)?;
        self.completion_state = state;
        Ok(state)
    }
}

/// An asynchronous owned D2H transfer.  Its bytes become readable only after
/// terminal success; no borrowed host staging pointer is exposed.
pub struct Readback {
    state: Arc<ExecutionSessionState>,
    queue: ExecutionQueue,
    output: OwnedTensorBinding,
    completion_state: ExecutionState,
    inner: Box<dyn ExecutionReadbackAdapter>,
}

/// An asynchronous D2H transfer for an arbitrary checked buffer range.
///
/// This is deliberately separate from semantic-output [`Readback`]. The
/// source range, queue, and session remain owned until terminal observation.
pub struct BufferReadback {
    state: Arc<ExecutionSessionState>,
    queue: ExecutionQueue,
    source: BufferRange,
    completion_state: ExecutionState,
    inner: Box<dyn ExecutionReadbackAdapter>,
}

impl fmt::Debug for BufferReadback {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BufferReadback")
            .field("session_id", &self.state.id)
            .field("queue", &self.queue.id())
            .field("source", &self.source)
            .field("completion_state", &self.completion_state)
            .finish_non_exhaustive()
    }
}

impl BufferReadback {
    pub const fn state(&self) -> ExecutionState {
        self.completion_state
    }

    pub fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        let state = self.inner.query()?;
        self.completion_state = state;
        Ok(state)
    }

    pub fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        let state = self.inner.wait(timeout)?;
        self.completion_state = state;
        Ok(state)
    }

    pub fn read_into(&mut self, destination: &mut [u8]) -> Result<u64, ExecutionError> {
        if self.completion_state != ExecutionState::Success {
            return Err(ExecutionError::NotReady);
        }
        let capacity =
            u64::try_from(destination.len()).map_err(|_| ExecutionError::InvalidRange {
                reason: "readback destination length does not fit u64".to_owned(),
            })?;
        if capacity != self.source.size_bytes() {
            return Err(ExecutionError::InvalidRange {
                reason: "readback destination must exactly match the source range".to_owned(),
            });
        }
        let copied = self.inner.read_into(destination)?;
        if copied != self.source.size_bytes() {
            return Err(ExecutionError::InvalidRange {
                reason: "backend readback byte count differs from the source range".to_owned(),
            });
        }
        Ok(copied)
    }
}

impl fmt::Debug for Readback {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Readback")
            .field("session_id", &self.state.id)
            .field("queue", &self.queue.id())
            .field("output", &self.output)
            .field("completion_state", &self.completion_state)
            .finish_non_exhaustive()
    }
}

impl Readback {
    pub const fn state(&self) -> ExecutionState {
        self.completion_state
    }

    pub fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        let state = self.inner.query()?;
        self.completion_state = state;
        Ok(state)
    }

    pub fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        let state = self.inner.wait(timeout)?;
        self.completion_state = state;
        Ok(state)
    }

    pub fn read_into(&mut self, destination: &mut [u8]) -> Result<u64, ExecutionError> {
        if self.completion_state != ExecutionState::Success {
            return Err(ExecutionError::NotReady);
        }
        self.inner.read_into(destination)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Barrier, Mutex};
    use std::thread;

    #[derive(Default)]
    struct TestAdapter {
        support_calls: AtomicUsize,
        support_in_progress: AtomicBool,
        support_shutdown_overlap: AtomicBool,
        shutdown_calls: AtomicUsize,
        upload_calls: AtomicUsize,
        readback_calls: AtomicUsize,
        support_gate: Mutex<Option<SupportGate>>,
    }

    struct SupportGate {
        entered: Arc<Barrier>,
        proceed: Arc<Barrier>,
    }

    impl TestAdapter {
        fn block_next_support(&self, entered: Arc<Barrier>, proceed: Arc<Barrier>) {
            *self.support_gate.lock().expect("support gate lock") =
                Some(SupportGate { entered, proceed });
        }
    }

    struct TestSubmission;
    struct TestTransfer;
    struct TestReadback {
        bytes: Vec<u8>,
    }

    impl ExecutionSessionAdapter for TestAdapter {
        fn max_transfer_bytes(&self) -> u64 {
            256
        }

        fn supports(&self, descriptor: &SemanticOpDescriptor) -> PrepareSupport {
            self.support_calls.fetch_add(1, Ordering::Relaxed);
            self.support_in_progress.store(true, Ordering::SeqCst);
            if let Some(gate) = self.support_gate.lock().expect("support gate lock").take() {
                gate.entered.wait();
                gate.proceed.wait();
            }
            self.support_in_progress.store(false, Ordering::SeqCst);
            if descriptor.kind() == SemanticOpKind::RmsNorm {
                PrepareSupport::Supported
            } else {
                PrepareSupport::Unsupported {
                    reason: "test adapter only prepares RMSNorm".to_owned(),
                }
            }
        }

        fn create_queue(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
        ) -> Result<AdapterResource, ExecutionError> {
            Ok(AdapterResource::new(()))
        }

        fn allocate(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            size_bytes: u64,
        ) -> Result<AdapterResource, ExecutionError> {
            let size_bytes =
                usize::try_from(size_bytes).map_err(|_| ExecutionError::InvalidRange {
                    reason: "test buffer size does not fit usize".to_owned(),
                })?;
            Ok(AdapterResource::new(Mutex::new(vec![0_u8; size_bytes])))
        }

        fn prepare(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _operation: &BoundSemanticOp,
        ) -> Result<AdapterResource, ExecutionError> {
            Ok(AdapterResource::new(()))
        }

        fn submit(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _prepared: &PreparedOperation,
            _queue: &ExecutionQueue,
        ) -> Result<(Box<dyn ExecutionSubmissionAdapter>, DispatchEvidence), ExecutionError>
        {
            Ok((
                Box::new(TestSubmission),
                DispatchEvidence {
                    abi_version: 1,
                    info_version: 1,
                    dispatch_id: 1,
                    dispatch_count: 1,
                    kernel_id: 1,
                    workgroup_size_x: 1,
                    grid_size_x: 1,
                    row_count: 1,
                    normalized_size: 1,
                    backend: 0,
                    fallback_allowed: false,
                    fallback_used: false,
                    kernel_symbol: "test".to_owned(),
                    device_symbol: "test".to_owned(),
                    target: "test".to_owned(),
                },
            ))
        }

        fn upload(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            _queue: &ExecutionQueue,
            destination: &BufferRange,
            bytes: Arc<[u8]>,
        ) -> Result<Box<dyn ExecutionTransferAdapter>, ExecutionError> {
            self.upload_calls.fetch_add(1, Ordering::Relaxed);
            let start = usize::try_from(destination.offset_bytes()).map_err(|_| {
                ExecutionError::InvalidRange {
                    reason: "test upload offset does not fit usize".to_owned(),
                }
            })?;
            let end = start + bytes.len();
            let mut storage = access
                .downcast_buffer_payload::<Mutex<Vec<u8>>>(destination.buffer())?
                .lock()
                .map_err(|_| ExecutionError::Busy)?;
            storage[start..end].copy_from_slice(&bytes);
            Ok(Box::new(TestTransfer))
        }

        fn readback(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            _queue: &ExecutionQueue,
            source: &BufferRange,
        ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
            self.readback_calls.fetch_add(1, Ordering::Relaxed);
            let storage = access
                .downcast_buffer_payload::<Mutex<Vec<u8>>>(source.buffer())?
                .lock()
                .map_err(|_| ExecutionError::Busy)?;
            let start = usize::try_from(source.offset_bytes()).map_err(|_| {
                ExecutionError::InvalidRange {
                    reason: "test readback offset does not fit usize".to_owned(),
                }
            })?;
            let size =
                usize::try_from(source.size_bytes()).map_err(|_| ExecutionError::InvalidRange {
                    reason: "test readback size does not fit usize".to_owned(),
                })?;
            Ok(Box::new(TestReadback {
                bytes: storage[start..start + size].to_vec(),
            }))
        }

        fn shutdown(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _deadline: Duration,
        ) -> Result<ShutdownReport, ExecutionError> {
            if self.support_in_progress.load(Ordering::SeqCst) {
                self.support_shutdown_overlap.store(true, Ordering::SeqCst);
            }
            self.shutdown_calls.fetch_add(1, Ordering::Relaxed);
            Ok(ShutdownReport {
                retryable_cleanup: 0,
                durable_quarantine: 0,
            })
        }
    }

    impl ExecutionSubmissionAdapter for TestSubmission {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn start_output_readback(
            &mut self,
            _access: &ExecutionAdapterAccess<'_>,
            output: &OwnedTensorBinding,
        ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
            let size = usize::try_from(output.view().payload_bytes()).map_err(|_| {
                ExecutionError::InvalidRange {
                    reason: "test output size does not fit usize".to_owned(),
                }
            })?;
            Ok(Box::new(TestReadback {
                bytes: vec![0x5a; size],
            }))
        }
    }

    impl ExecutionTransferAdapter for TestTransfer {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }
    }

    impl ExecutionReadbackAdapter for TestReadback {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn read_into(&mut self, destination: &mut [u8]) -> Result<u64, ExecutionError> {
            if destination.len() != self.bytes.len() {
                return Err(ExecutionError::InvalidRange {
                    reason: "test readback destination length mismatch".to_owned(),
                });
            }
            destination.copy_from_slice(&self.bytes);
            Ok(destination.len() as u64)
        }
    }

    fn session(name: &'static str) -> ExecutionSession {
        ExecutionSession::new(name, Arc::new(TestAdapter::default()))
    }

    fn rmsnorm_views(offsets: [u64; 3]) -> (TensorView, TensorView, TensorView) {
        let activation = TensorView::new(
            crate::DType::Bf16,
            crate::Encoding::Unquantized,
            &[2, 3],
            &[3, 1],
            offsets[0],
        )
        .expect("valid activation");
        let scale = TensorView::new(
            crate::DType::Bf16,
            crate::Encoding::Unquantized,
            &[3],
            &[1],
            offsets[1],
        )
        .expect("valid scale");
        let output = TensorView::new(
            crate::DType::Bf16,
            crate::Encoding::Unquantized,
            &[2, 3],
            &[3, 1],
            offsets[2],
        )
        .expect("valid output");
        (activation, scale, output)
    }

    fn rmsnorm_descriptor(
        activation: TensorView,
        scale: TensorView,
        output: TensorView,
    ) -> Arc<SemanticOpDescriptor> {
        Arc::new(
            SemanticOpDescriptor::new_rms_norm(
                vec![activation, scale],
                vec![output],
                1.0e-6,
                crate::RmsNormScaleMode::OffsetOne,
            )
            .expect("valid RMSNorm descriptor"),
        )
    }

    #[test]
    fn bounded_buffer_readback_round_trips_and_rejects_before_adapter_submission() {
        let adapter = Arc::new(TestAdapter::default());
        let test_session = ExecutionSession::new("test", adapter.clone());
        let queue = test_session.create_queue().unwrap();
        let buffer = test_session.allocate(520).unwrap();
        assert_eq!(test_session.max_transfer_bytes().unwrap(), 256);

        for size in [1_usize, 3, 17, 255, 256] {
            let bytes: Vec<u8> = (0..size).map(|index| (index % 251) as u8).collect();
            let range = buffer.range(7, size as u64).unwrap();
            let mut upload = test_session
                .upload(&queue, range.clone(), Arc::<[u8]>::from(bytes.clone()))
                .unwrap();
            assert_eq!(
                upload.wait(Duration::ZERO).unwrap(),
                ExecutionState::Success
            );

            let mut readback = test_session.readback(&queue, range).unwrap();
            let mut exact = vec![0_u8; size];
            assert!(matches!(
                readback.read_into(&mut exact),
                Err(ExecutionError::NotReady)
            ));
            assert_eq!(
                readback.wait(Duration::ZERO).unwrap(),
                ExecutionState::Success
            );
            let mut short = vec![0_u8; size - 1];
            let mut long = vec![0_u8; size + 1];
            assert!(matches!(
                readback.read_into(&mut short),
                Err(ExecutionError::InvalidRange { .. })
            ));
            assert!(matches!(
                readback.read_into(&mut long),
                Err(ExecutionError::InvalidRange { .. })
            ));
            assert_eq!(readback.read_into(&mut exact).unwrap(), size as u64);
            assert_eq!(exact, bytes);
        }
        assert_eq!(adapter.upload_calls.load(Ordering::Relaxed), 5);
        assert_eq!(adapter.readback_calls.load(Ordering::Relaxed), 5);

        let too_large = buffer.range(0, 257).unwrap();
        assert!(matches!(
            test_session.readback(&queue, too_large.clone()),
            Err(ExecutionError::InvalidRange { .. })
        ));
        assert!(matches!(
            test_session.upload(&queue, too_large, Arc::<[u8]>::from(vec![0_u8; 257])),
            Err(ExecutionError::InvalidRange { .. })
        ));
        assert_eq!(adapter.upload_calls.load(Ordering::Relaxed), 5);
        assert_eq!(adapter.readback_calls.load(Ordering::Relaxed), 5);

        let other = session("test");
        let other_queue = other.create_queue().unwrap();
        let other_buffer = other.allocate(17).unwrap();
        assert!(matches!(
            test_session.readback(&other_queue, buffer.range(0, 1).unwrap()),
            Err(ExecutionError::WrongQueue { .. })
        ));
        assert!(matches!(
            test_session.readback(&queue, other_buffer.range(0, 1).unwrap()),
            Err(ExecutionError::WrongSession { .. })
        ));
        assert!(matches!(
            buffer.range(0, 0),
            Err(ExecutionError::InvalidRange { .. })
        ));
        assert!(matches!(
            buffer.range(u64::MAX, 2),
            Err(ExecutionError::InvalidRange { .. })
        ));

        let pending = test_session
            .readback(&queue, buffer.range(0, 1).unwrap())
            .unwrap();
        drop(pending);
        test_session.shutdown(Duration::ZERO).unwrap();
        assert!(matches!(
            test_session.readback(&queue, buffer.range(0, 1).unwrap()),
            Err(ExecutionError::Closing)
        ));
    }

    #[test]
    fn supports_after_shutdown_returns_exact_closing_reason_without_adapter_call() {
        let adapter = Arc::new(TestAdapter::default());
        let session = ExecutionSession::new("test", adapter.clone());
        let (activation, scale, output) = rmsnorm_views([0, 12, 18]);
        let descriptor = rmsnorm_descriptor(activation, scale, output);

        assert_eq!(session.supports(&descriptor), PrepareSupport::Supported);
        assert_eq!(adapter.support_calls.load(Ordering::Relaxed), 1);

        session
            .shutdown(Duration::ZERO)
            .expect("test adapter shutdown succeeds");
        assert_eq!(
            session.supports(&descriptor),
            PrepareSupport::Unsupported {
                reason: "execution session is closing".to_owned(),
            }
        );
        assert_eq!(adapter.support_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn shutdown_waits_for_admitted_support_adapter_call_without_overlap() {
        let adapter = Arc::new(TestAdapter::default());
        let session = Arc::new(ExecutionSession::new("test", adapter.clone()));
        let (activation, scale, output) = rmsnorm_views([0, 12, 18]);
        let descriptor = rmsnorm_descriptor(activation, scale, output);
        let entered = Arc::new(Barrier::new(2));
        let proceed = Arc::new(Barrier::new(2));
        adapter.block_next_support(Arc::clone(&entered), Arc::clone(&proceed));

        let support_session = Arc::clone(&session);
        let support_descriptor = Arc::clone(&descriptor);
        let support_thread = thread::spawn(move || support_session.supports(&support_descriptor));
        entered.wait();

        let shutdown_ready = Arc::new(Barrier::new(2));
        let shutdown_session = Arc::clone(&session);
        let shutdown_ready_for_thread = Arc::clone(&shutdown_ready);
        let shutdown_thread = thread::spawn(move || {
            shutdown_ready_for_thread.wait();
            shutdown_session.shutdown(Duration::ZERO)
        });
        shutdown_ready.wait();

        assert!(matches!(
            session.state.adapter_admission.try_lock(),
            Err(std::sync::TryLockError::WouldBlock)
        ));
        proceed.wait();

        assert_eq!(
            support_thread.join().expect("support thread completed"),
            PrepareSupport::Supported
        );
        assert_eq!(
            shutdown_thread.join().expect("shutdown thread completed"),
            Ok(ShutdownReport {
                retryable_cleanup: 0,
                durable_quarantine: 0,
            })
        );
        assert!(!adapter.support_shutdown_overlap.load(Ordering::SeqCst));
        assert_eq!(adapter.shutdown_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn descriptor_arc_identity_survives_bound_and_prepared_operations() {
        let session = session("test");
        let activation_buffer = session.allocate(64).unwrap();
        let scale_buffer = session.allocate(64).unwrap();
        let output_buffer = session.allocate(64).unwrap();
        let (activation, scale, output) = rmsnorm_views([0, 12, 18]);
        let descriptor = rmsnorm_descriptor(activation.clone(), scale.clone(), output.clone());
        let bound = Arc::new(
            BoundSemanticOp::new(
                Arc::clone(&descriptor),
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

        assert!(Arc::ptr_eq(&descriptor, bound.descriptor()));
        let prepared = session.prepare(Arc::clone(&bound)).unwrap();
        assert!(Arc::ptr_eq(&descriptor, prepared.operation().descriptor()));
    }

    #[test]
    fn binding_layer_checks_exact_views_identity_bounds_and_access() {
        let first = session("test-a");
        let second = session("test-b");
        let first_buffer = first.allocate(64).expect("first buffer");
        let second_buffer = second.allocate(64).expect("second buffer");
        let (activation, scale, output) = rmsnorm_views([0, 12, 18]);
        let descriptor = rmsnorm_descriptor(activation.clone(), scale.clone(), output.clone());

        let input = first
            .bind(&first_buffer, activation.clone(), AccessMode::Read)
            .expect("activation binding");
        let scale_binding = first
            .bind(&first_buffer, scale.clone(), AccessMode::Read)
            .expect("scale binding");
        let output_binding = first
            .bind(&first_buffer, output.clone(), AccessMode::Write)
            .expect("output binding");
        assert!(
            BoundSemanticOp::new(
                Arc::clone(&descriptor),
                vec![input.clone(), scale_binding.clone()],
                vec![output_binding.clone()]
            )
            .is_ok()
        );

        let wrong_view = first
            .bind(
                &first_buffer,
                TensorView::new(
                    crate::DType::Bf16,
                    crate::Encoding::Unquantized,
                    &[2, 3],
                    &[3, 1],
                    2,
                )
                .unwrap(),
                AccessMode::Read,
            )
            .expect("wrong view can bind");
        assert!(matches!(
            BoundSemanticOp::new(
                Arc::clone(&descriptor),
                vec![wrong_view, scale_binding.clone()],
                vec![output_binding.clone()]
            ),
            Err(ExecutionError::DescriptorBindingMismatch { .. })
        ));

        let foreign = second
            .bind(&second_buffer, scale.clone(), AccessMode::Read)
            .expect("foreign binding");
        assert!(matches!(
            BoundSemanticOp::new(
                Arc::clone(&descriptor),
                vec![input.clone(), foreign],
                vec![output_binding.clone()]
            ),
            Err(ExecutionError::WrongBackend { .. })
        ));

        let same_backend_other_session = session("test-a");
        let same_backend_other_buffer = same_backend_other_session.allocate(64).unwrap();
        let foreign_same_backend = same_backend_other_session
            .bind(&same_backend_other_buffer, scale.clone(), AccessMode::Read)
            .unwrap();
        assert!(matches!(
            BoundSemanticOp::new(
                Arc::clone(&descriptor),
                vec![input.clone(), foreign_same_backend],
                vec![output_binding.clone()]
            ),
            Err(ExecutionError::WrongSession { .. })
        ));

        let write_input = first
            .bind(&first_buffer, activation, AccessMode::Write)
            .expect("write-only input binding");
        assert!(matches!(
            BoundSemanticOp::new(
                Arc::clone(&descriptor),
                vec![write_input, scale_binding.clone()],
                vec![output_binding.clone()]
            ),
            Err(ExecutionError::AccessViolation { .. })
        ));

        let read_output = first
            .bind(&first_buffer, output, AccessMode::Read)
            .expect("read-only output binding");
        assert!(matches!(
            BoundSemanticOp::new(descriptor, vec![input, scale_binding], vec![read_output]),
            Err(ExecutionError::AccessViolation { .. })
        ));

        let too_large = TensorView::new(
            crate::DType::Bf16,
            crate::Encoding::Unquantized,
            &[2, 3],
            &[3, 1],
            54,
        )
        .expect("well-formed external view");
        assert!(matches!(
            first.bind(&first_buffer, too_large, AccessMode::Read),
            Err(ExecutionError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn rmsnorm_overlap_rejects_every_pair_but_allows_touching_ranges() {
        let session = session("test");
        let buffer = session.allocate(64).unwrap();
        let (activation, scale, output) = rmsnorm_views([0, 12, 18]);
        let touching = rmsnorm_descriptor(activation.clone(), scale.clone(), output.clone());
        let valid = BoundSemanticOp::new(
            touching,
            vec![
                session.bind(&buffer, activation, AccessMode::Read).unwrap(),
                session.bind(&buffer, scale, AccessMode::Read).unwrap(),
            ],
            vec![session.bind(&buffer, output, AccessMode::Write).unwrap()],
        );
        assert!(
            valid.is_ok(),
            "half-open touching ranges must stay disjoint"
        );

        for (offsets, expected_left, expected_right) in [
            ([0, 4, 18], "activation", "raw scale"),
            ([0, 12, 4], "activation", "output"),
            ([0, 12, 14], "raw scale", "output"),
        ] {
            let (activation, scale, output) = rmsnorm_views(offsets);
            let descriptor = rmsnorm_descriptor(activation.clone(), scale.clone(), output.clone());
            assert!(matches!(
                BoundSemanticOp::new(
                    descriptor,
                    vec![
                        session.bind(&buffer, activation, AccessMode::Read).unwrap(),
                        session.bind(&buffer, scale, AccessMode::Read).unwrap(),
                    ],
                    vec![session.bind(&buffer, output, AccessMode::Write).unwrap()],
                ),
                Err(ExecutionError::AliasOverlap { left, right })
                    if left == expected_left && right == expected_right
            ));
        }
    }

    #[test]
    fn prepared_submission_transfer_and_readback_keep_single_session_contract() {
        let test_session = session("test");
        let queue = test_session.create_queue().unwrap();
        let activation_buffer = test_session.allocate(64).unwrap();
        let scale_buffer = test_session.allocate(64).unwrap();
        let output_buffer = test_session.allocate(64).unwrap();
        let (activation, scale, output) = rmsnorm_views([0, 0, 0]);
        let descriptor = rmsnorm_descriptor(activation.clone(), scale.clone(), output.clone());
        let operation = Arc::new(
            BoundSemanticOp::new(
                descriptor,
                vec![
                    test_session
                        .bind(&activation_buffer, activation, AccessMode::Read)
                        .unwrap(),
                    test_session
                        .bind(&scale_buffer, scale, AccessMode::Read)
                        .unwrap(),
                ],
                vec![
                    test_session
                        .bind(&output_buffer, output, AccessMode::Write)
                        .unwrap(),
                ],
            )
            .unwrap(),
        );
        let prepared = test_session.prepare(operation).unwrap();
        let mut submission = test_session.submit(&prepared, &queue).unwrap();
        assert!(matches!(
            submission.start_output_readback(0),
            Err(ExecutionError::NotReady)
        ));
        assert_eq!(
            submission.wait(Duration::ZERO).unwrap(),
            ExecutionState::Success
        );
        let mut readback = submission.start_output_readback(0).unwrap();
        assert_eq!(
            readback.wait(Duration::ZERO).unwrap(),
            ExecutionState::Success
        );
        let mut bytes = [0_u8; 12];
        assert_eq!(readback.read_into(&mut bytes).unwrap(), 12);
        assert!(bytes.iter().all(|byte| *byte == 0x5a));

        let range = activation_buffer.range(0, 12).unwrap();
        let mut transfer = test_session
            .upload(&queue, range, Arc::<[u8]>::from([1_u8; 12]))
            .unwrap();
        assert_eq!(
            transfer.wait(Duration::ZERO).unwrap(),
            ExecutionState::Success
        );

        let other = session("test");
        let other_queue = other.create_queue().unwrap();
        assert!(matches!(
            test_session.submit(&prepared, &other_queue),
            Err(ExecutionError::WrongQueue { .. })
        ));
    }
}
