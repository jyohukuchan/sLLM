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

use crate::kv_state::{
    CausalAttentionDescriptor, KvStateAppendRequest, KvStateDescriptor, KvStateLayout,
    KvStateSnapshot,
};
use crate::linear_attention::{
    LinearAttentionDescriptor, LinearAttentionLayout, LinearAttentionRequest,
    LinearAttentionStateDescriptor, LinearAttentionStateSnapshot,
};
use crate::{AccessMode, DType, Encoding, SemanticOpDescriptor, SemanticOpKind, TensorView};

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
            pub(crate) const fn new(raw: u64) -> Self {
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
execution_id!(KvStateId);
execution_id!(LinearAttentionStateId);

/// The accounting bucket for one session-owned device allocation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AllocationCategory {
    ModelResident,
    RequestState,
    Workspace,
}

/// Current and high-water bytes for one allocation bucket.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AllocationCategorySnapshot {
    current_bytes: u64,
    high_water_bytes: u64,
}

impl AllocationCategorySnapshot {
    pub const fn current_bytes(self) -> u64 {
        self.current_bytes
    }

    pub const fn high_water_bytes(self) -> u64 {
        self.high_water_bytes
    }
}

/// Checked session allocation accounting. The category snapshots are exact
/// for allocations admitted through this execution boundary; backend memory
/// outside the boundary is intentionally not inferred.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AllocationSnapshot {
    model_resident: AllocationCategorySnapshot,
    request_state: AllocationCategorySnapshot,
    workspace: AllocationCategorySnapshot,
    current_bytes: u64,
    high_water_bytes: u64,
    poisoned: bool,
}

impl AllocationSnapshot {
    pub const fn model_resident(self) -> AllocationCategorySnapshot {
        self.model_resident
    }

    pub const fn request_state(self) -> AllocationCategorySnapshot {
        self.request_state
    }

    pub const fn workspace(self) -> AllocationCategorySnapshot {
        self.workspace
    }

    pub const fn current_bytes(self) -> u64 {
        self.current_bytes
    }

    pub const fn high_water_bytes(self) -> u64 {
        self.high_water_bytes
    }

    pub const fn poisoned(self) -> bool {
        self.poisoned
    }
}

#[derive(Debug)]
struct AllocationAccountingState {
    buckets: [AllocationCategorySnapshot; 3],
    current_bytes: u64,
    high_water_bytes: u64,
    poisoned: bool,
}

impl Default for AllocationAccountingState {
    fn default() -> Self {
        const EMPTY: AllocationCategorySnapshot = AllocationCategorySnapshot {
            current_bytes: 0,
            high_water_bytes: 0,
        };
        Self {
            buckets: [EMPTY; 3],
            current_bytes: 0,
            high_water_bytes: 0,
            poisoned: false,
        }
    }
}

#[derive(Debug, Default)]
struct AllocationAccounting {
    state: Mutex<AllocationAccountingState>,
}

impl AllocationAccounting {
    fn snapshot(&self) -> AllocationSnapshot {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        AllocationSnapshot {
            model_resident: state.buckets[0],
            request_state: state.buckets[1],
            workspace: state.buckets[2],
            current_bytes: state.current_bytes,
            high_water_bytes: state.high_water_bytes,
            poisoned: state.poisoned,
        }
    }

    fn reserve(
        self: &Arc<Self>,
        category: AllocationCategory,
        size_bytes: u64,
    ) -> Result<Arc<AllocationLease>, ExecutionError> {
        if size_bytes == 0 {
            return Err(ExecutionError::InvalidRange {
                reason: "allocation accounting cannot reserve zero bytes".to_owned(),
            });
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| ExecutionError::AllocationAccountingPoisoned)?;
        if state.poisoned {
            return Err(ExecutionError::AllocationAccountingPoisoned);
        }
        let index = category.index();
        let current = state.buckets[index].current_bytes;
        let next_category = match current.checked_add(size_bytes) {
            Some(value) => value,
            None => {
                state.poisoned = true;
                return Err(ExecutionError::AllocationAccountingOverflow);
            }
        };
        let next_total = match state.current_bytes.checked_add(size_bytes) {
            Some(value) => value,
            None => {
                state.poisoned = true;
                return Err(ExecutionError::AllocationAccountingOverflow);
            }
        };
        state.buckets[index].current_bytes = next_category;
        state.buckets[index].high_water_bytes =
            state.buckets[index].high_water_bytes.max(next_category);
        state.current_bytes = next_total;
        state.high_water_bytes = state.high_water_bytes.max(next_total);
        drop(state);
        Ok(Arc::new(AllocationLease {
            accounting: Arc::clone(self),
            category,
            size_bytes,
        }))
    }

    fn release(&self, category: AllocationCategory, size_bytes: u64) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let index = category.index();
        let Some(next_category) = state.buckets[index].current_bytes.checked_sub(size_bytes) else {
            state.poisoned = true;
            return;
        };
        let Some(next_total) = state.current_bytes.checked_sub(size_bytes) else {
            state.poisoned = true;
            return;
        };
        state.buckets[index].current_bytes = next_category;
        state.current_bytes = next_total;
    }
}

impl AllocationCategory {
    const fn index(self) -> usize {
        match self {
            Self::ModelResident => 0,
            Self::RequestState => 1,
            Self::Workspace => 2,
        }
    }
}

struct AllocationLease {
    accounting: Arc<AllocationAccounting>,
    category: AllocationCategory,
    size_bytes: u64,
}

impl Drop for AllocationLease {
    fn drop(&mut self) {
        self.accounting.release(self.category, self.size_bytes);
    }
}

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
    WrongKvState {
        expected: KvStateId,
        actual: KvStateId,
    },
    WrongLinearAttentionState {
        expected: LinearAttentionStateId,
        actual: LinearAttentionStateId,
    },
    StaleKvLength {
        expected: u64,
        actual: u64,
    },
    StaleLinearAttentionLength {
        expected: u64,
        actual: u64,
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
    AllocationAccountingOverflow,
    AllocationAccountingPoisoned,
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
            Self::WrongKvState { expected, actual } => write!(
                formatter,
                "KV state {} does not match state {}",
                actual.raw(),
                expected.raw()
            ),
            Self::WrongLinearAttentionState { expected, actual } => write!(
                formatter,
                "linear-attention state {} does not match state {}",
                actual.raw(),
                expected.raw()
            ),
            Self::StaleKvLength { expected, actual } => write!(
                formatter,
                "stale KV length: expected {expected}, backend reports {actual}"
            ),
            Self::StaleLinearAttentionLength { expected, actual } => write!(
                formatter,
                "stale linear-attention length: expected {expected}, backend reports {actual}"
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
                write!(formatter, "semantic {left} and {right} bindings overlap")
            }
            Self::Busy => formatter.write_str("execution resource is busy"),
            Self::NotReady => formatter.write_str("execution completion is not ready"),
            Self::Closing => formatter.write_str("execution session is closing"),
            Self::CleanupQuarantined => {
                formatter.write_str("execution cleanup entered durable quarantine")
            }
            Self::AllocationAccountingOverflow => {
                formatter.write_str("execution allocation accounting overflowed")
            }
            Self::AllocationAccountingPoisoned => {
                formatter.write_str("execution allocation accounting is poisoned")
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

    /// Free device memory observed immediately before the session was opened.
    /// Backends that cannot report it return `None`; callers that require a
    /// fail-closed placement preflight must reject that absence explicitly.
    fn available_memory_bytes(&self) -> Option<u64> {
        None
    }

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

    /// Creates one request-local full-attention KV state resource.  Adapters
    /// that do not implement C3a2 remain source-compatible and reject it by
    /// default; core never allocates a CPU substitute.
    fn create_kv_state(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        _state_id: KvStateId,
        _descriptor: KvStateDescriptor,
    ) -> Result<AdapterResource, ExecutionError> {
        Err(ExecutionError::Unsupported {
            reason: "backend does not support request-local KV state".to_owned(),
        })
    }

    /// Reads the backend-owned authoritative length and identity metadata.
    fn kv_state_snapshot(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        _state: &KvState,
    ) -> Result<KvStateSnapshot, ExecutionError> {
        Err(ExecutionError::Unsupported {
            reason: "backend does not support request-local KV state snapshots".to_owned(),
        })
    }

    /// Enqueues a transactional append.  The backend owns the authoritative
    /// length transition; core supplies already-admitted bindings and request
    /// metadata, but never performs numerical or state updates itself.
    fn append_kv_state(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        _state: &KvState,
        _queue: &ExecutionQueue,
        _key: &OwnedTensorBinding,
        _value: &OwnedTensorBinding,
        _request: &KvStateAppendRequest,
    ) -> Result<(Box<dyn ExecutionKvStateSubmissionAdapter>, DispatchEvidence), ExecutionError>
    {
        Err(ExecutionError::Unsupported {
            reason: "backend does not support request-local KV state append".to_owned(),
        })
    }

    /// Enqueues causal GQA attention against an immutable committed KV
    /// snapshot. The adapter owns the native state lifetime until terminal
    /// completion cleanup.
    #[allow(clippy::too_many_arguments)]
    fn execute_causal_attention(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        _state: &KvState,
        _queue: &ExecutionQueue,
        _query: &OwnedTensorBinding,
        _output: &OwnedTensorBinding,
        _descriptor: CausalAttentionDescriptor,
    ) -> Result<
        (
            Box<dyn ExecutionCausalAttentionSubmissionAdapter>,
            DispatchEvidence,
        ),
        ExecutionError,
    > {
        Err(ExecutionError::Unsupported {
            reason: "backend does not support causal GQA attention".to_owned(),
        })
    }

    /// Creates one request-local C4 convolution/recurrent state. Adapters
    /// without native linear attention reject it rather than substituting CPU
    /// storage or execution.
    fn create_linear_attention_state(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        _state_id: LinearAttentionStateId,
        _descriptor: LinearAttentionStateDescriptor,
    ) -> Result<AdapterResource, ExecutionError> {
        Err(ExecutionError::Unsupported {
            reason: "backend does not support request-local linear-attention state".to_owned(),
        })
    }

    fn linear_attention_state_snapshot(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        _state: &LinearAttentionState,
    ) -> Result<LinearAttentionStateSnapshot, ExecutionError> {
        Err(ExecutionError::Unsupported {
            reason: "backend does not support linear-attention state snapshots".to_owned(),
        })
    }

    fn execute_linear_attention(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        _state: &LinearAttentionState,
        _queue: &ExecutionQueue,
        _bindings: &LinearAttentionBindings,
        _request: LinearAttentionRequest,
    ) -> Result<
        (
            Box<dyn ExecutionLinearAttentionSubmissionAdapter>,
            DispatchEvidence,
        ),
        ExecutionError,
    > {
        Err(ExecutionError::Unsupported {
            reason: "backend does not support linear-attention execution".to_owned(),
        })
    }
}

/// Adapter-owned mutable submission state.  It is intentionally `Send` but
/// not `Sync`; core exposes it only through the single-owner `Submission`.
pub trait ExecutionSubmissionAdapter: Send {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError>;
    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError>;
    fn kernel_elapsed_ns(&mut self) -> Result<Option<u64>, ExecutionError> {
        Ok(None)
    }
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

/// Adapter-owned mutable state append completion.  It is separate from a
/// prepared semantic-op submission because a KV append has no stateless output
/// descriptor or output readback operation.
pub trait ExecutionKvStateSubmissionAdapter: Send {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError>;
    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError>;
}

/// Adapter-owned mutable causal-attention completion. It is separate from a
/// stateless semantic submission because it retains a request-local KV state.
pub trait ExecutionCausalAttentionSubmissionAdapter: Send {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError>;
    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError>;
}

/// Adapter-owned completion for one transactional linear-attention state
/// transition.
pub trait ExecutionLinearAttentionSubmissionAdapter: Send {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError>;
    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError>;
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
    allocation_accounting: Arc<AllocationAccounting>,
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
                allocation_accounting: Arc::new(AllocationAccounting::default()),
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

    pub fn available_memory_bytes(&self) -> Result<Option<u64>, ExecutionError> {
        self.ensure_open()?;
        Ok(self.state.adapter.available_memory_bytes())
    }

    /// Returns exact current and high-water accounting for this session.
    /// This remains readable after shutdown so callers can verify cleanup.
    pub fn allocation_snapshot(&self) -> AllocationSnapshot {
        self.state.allocation_accounting.snapshot()
    }

    /// Alias used by benchmark/reporting callers that expose memory rather
    /// than allocator terminology.
    pub fn memory_snapshot(&self) -> AllocationSnapshot {
        self.allocation_snapshot()
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
        self.allocate_with_category(size_bytes, AllocationCategory::Workspace)
    }

    /// Allocates a checked device buffer in an explicit accounting bucket.
    pub fn allocate_with_category(
        &self,
        size_bytes: u64,
        category: AllocationCategory,
    ) -> Result<ExecutionBuffer, ExecutionError> {
        self.ensure_open()?;
        if size_bytes == 0 {
            return Err(ExecutionError::InvalidRange {
                reason: "execution buffer size must be non-zero".to_owned(),
            });
        }
        let allocation = self
            .state
            .allocation_accounting
            .reserve(category, size_bytes)?;
        let access = ExecutionAdapterAccess { session: self };
        let resource = match self.state.adapter.allocate(&access, size_bytes) {
            Ok(resource) => resource,
            Err(error) => {
                drop(allocation);
                return Err(error);
            }
        };
        Ok(ExecutionBuffer {
            state: Arc::clone(&self.state),
            id: ExecutionBufferId::new(next_execution_id()),
            size_bytes,
            payload: resource.payload,
            _allocation: allocation,
        })
    }

    /// Creates a backend-owned request-local full-attention KV state.
    ///
    /// This is a state-resource operation, not a `SemanticOpDescriptor` and
    /// not a prepared stateless operation.  An adapter that has not adopted
    /// C3a2 returns the default unsupported error.
    pub fn create_kv_state(
        &self,
        descriptor: KvStateDescriptor,
    ) -> Result<KvState, ExecutionError> {
        self.ensure_open()?;
        let allocation = kv_state_allocation_bytes(descriptor)
            .map(|bytes| {
                self.state
                    .allocation_accounting
                    .reserve(AllocationCategory::RequestState, bytes)
            })
            .transpose()?;
        let id = KvStateId::new(next_execution_id());
        let resource = match self.state.adapter.create_kv_state(
            &ExecutionAdapterAccess { session: self },
            id,
            descriptor,
        ) {
            Ok(resource) => resource,
            Err(error) => {
                drop(allocation);
                return Err(error);
            }
        };
        Ok(KvState {
            state: Arc::clone(&self.state),
            id,
            descriptor,
            payload: resource.payload,
            append_in_flight: Arc::new(AtomicBool::new(false)),
            attention_in_flight: Arc::new(AtomicBool::new(false)),
            operation_admission: Arc::new(Mutex::new(())),
            _allocation: allocation,
        })
    }

    /// Returns the backend-owned authoritative length and typed identity.
    pub fn kv_state_snapshot(&self, state: &KvState) -> Result<KvStateSnapshot, ExecutionError> {
        self.ensure_open()?;
        self.ensure_kv_state(state)?;
        let snapshot = self
            .state
            .adapter
            .kv_state_snapshot(&ExecutionAdapterAccess { session: self }, state)?;
        validate_kv_state_snapshot(self, state, snapshot)
    }

    /// Creates a backend-owned request-local linear-attention state.
    pub fn create_linear_attention_state(
        &self,
        descriptor: LinearAttentionStateDescriptor,
    ) -> Result<LinearAttentionState, ExecutionError> {
        self.ensure_open()?;
        let allocation = linear_state_allocation_bytes(descriptor)
            .map(|bytes| {
                self.state
                    .allocation_accounting
                    .reserve(AllocationCategory::RequestState, bytes)
            })
            .transpose()?;
        let id = LinearAttentionStateId::new(next_execution_id());
        let resource = match self.state.adapter.create_linear_attention_state(
            &ExecutionAdapterAccess { session: self },
            id,
            descriptor,
        ) {
            Ok(resource) => resource,
            Err(error) => {
                drop(allocation);
                return Err(error);
            }
        };
        Ok(LinearAttentionState {
            state: Arc::clone(&self.state),
            id,
            descriptor,
            payload: resource.payload,
            execution_in_flight: Arc::new(AtomicBool::new(false)),
            _allocation: allocation,
        })
    }

    pub fn linear_attention_state_snapshot(
        &self,
        state: &LinearAttentionState,
    ) -> Result<LinearAttentionStateSnapshot, ExecutionError> {
        self.ensure_open()?;
        self.ensure_linear_attention_state(state)?;
        let snapshot = self
            .state
            .adapter
            .linear_attention_state_snapshot(&ExecutionAdapterAccess { session: self }, state)?;
        validate_linear_attention_state_snapshot(self, state, snapshot)
    }

    /// Admits one ordered convolution/recurrent state transition. The output
    /// and inactive state slot may be written asynchronously, but publication
    /// of the new state is a backend completion responsibility.
    pub fn linear_attention(
        &self,
        state: &LinearAttentionState,
        queue: &ExecutionQueue,
        bindings: LinearAttentionBindings,
        descriptor: LinearAttentionDescriptor,
    ) -> Result<LinearAttentionSubmission, ExecutionError> {
        self.ensure_open()?;
        self.ensure_linear_attention_state(state)?;
        self.ensure_queue(queue)?;
        validate_linear_attention_bindings(self, &bindings, descriptor, state.descriptor.layout())?;
        if descriptor.expected_length() > state.capacity() {
            return Err(ExecutionError::InvalidRange {
                reason: "linear-attention transition exceeds state capacity".to_owned(),
            });
        }
        if state
            .execution_in_flight
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(ExecutionError::Busy);
        }
        let snapshot = match self.linear_attention_state_snapshot(state) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                state.execution_in_flight.store(false, Ordering::Release);
                return Err(error);
            }
        };
        if snapshot.length() != descriptor.start_position() {
            state.execution_in_flight.store(false, Ordering::Release);
            return Err(ExecutionError::StaleLinearAttentionLength {
                expected: descriptor.start_position(),
                actual: snapshot.length(),
            });
        }
        let request = LinearAttentionRequest::new(state.id, state.descriptor, descriptor);
        let (inner, dispatch) = match self.state.adapter.execute_linear_attention(
            &ExecutionAdapterAccess { session: self },
            state,
            queue,
            &bindings,
            request,
        ) {
            Ok(result) => result,
            Err(error) => {
                state.execution_in_flight.store(false, Ordering::Release);
                return Err(error);
            }
        };
        Ok(LinearAttentionSubmission {
            state: state.clone(),
            queue: queue.clone(),
            bindings,
            request,
            dispatch,
            completion_state: ExecutionState::Pending,
            inner: Some(inner),
        })
    }

    /// Admits and submits one transactional K/V append.  All core checks,
    /// including the authoritative length comparison, occur before the
    /// adapter append callback.  The returned type retains the state and both
    /// input bindings until completion is observed or it is dropped.
    #[allow(clippy::too_many_arguments)]
    pub fn append_kv_state(
        &self,
        state: &KvState,
        queue: &ExecutionQueue,
        key: OwnedTensorBinding,
        value: OwnedTensorBinding,
        expected_length: u64,
        start_position: u64,
    ) -> Result<KvStateAppendSubmission, ExecutionError> {
        self.ensure_open()?;
        self.ensure_kv_state(state)?;
        self.ensure_queue(queue)?;

        let token_count =
            validate_kv_append_bindings(self, &key, &value, state.descriptor.layout())?;
        if start_position != expected_length {
            return Err(ExecutionError::InvalidRange {
                reason: "KV append start position must equal expected length".to_owned(),
            });
        }
        let end_position = start_position.checked_add(token_count).ok_or_else(|| {
            ExecutionError::InvalidRange {
                reason: "KV append end position overflowed u64".to_owned(),
            }
        })?;
        if end_position > state.descriptor.capacity() {
            return Err(ExecutionError::InvalidRange {
                reason: "KV append exceeds state capacity".to_owned(),
            });
        }

        {
            let _admission = state
                .operation_admission
                .lock()
                .map_err(|_| ExecutionError::Busy)?;
            if state.attention_in_flight.load(Ordering::Acquire)
                || state
                    .append_in_flight
                    .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                    .is_err()
            {
                return Err(ExecutionError::Busy);
            }
        }

        let snapshot = match self.kv_state_snapshot(state) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                state.append_in_flight.store(false, Ordering::Release);
                return Err(error);
            }
        };
        if snapshot.length() != expected_length {
            state.append_in_flight.store(false, Ordering::Release);
            return Err(ExecutionError::StaleKvLength {
                expected: expected_length,
                actual: snapshot.length(),
            });
        }

        let request = KvStateAppendRequest::new(
            state.id,
            state.descriptor,
            token_count,
            expected_length,
            start_position,
        );
        let (inner, dispatch) = match self.state.adapter.append_kv_state(
            &ExecutionAdapterAccess { session: self },
            state,
            queue,
            &key,
            &value,
            &request,
        ) {
            Ok(inner) => inner,
            Err(error) => {
                state.append_in_flight.store(false, Ordering::Release);
                return Err(error);
            }
        };

        Ok(KvStateAppendSubmission {
            state: state.clone(),
            queue: queue.clone(),
            key,
            value,
            request,
            dispatch,
            completion_state: ExecutionState::Pending,
            inner: Some(inner),
        })
    }

    /// Validates and submits causal GQA attention against the exact session,
    /// queue, Q/output bindings, and committed state identity.
    #[allow(clippy::too_many_arguments)]
    pub fn causal_attention(
        &self,
        state: &KvState,
        queue: &ExecutionQueue,
        query: OwnedTensorBinding,
        output: OwnedTensorBinding,
        descriptor: CausalAttentionDescriptor,
    ) -> Result<CausalAttentionSubmission, ExecutionError> {
        self.ensure_open()?;
        self.ensure_kv_state(state)?;
        self.ensure_queue(queue)?;
        validate_causal_attention_bindings(
            self,
            &query,
            &output,
            descriptor,
            state.descriptor.layout(),
        )?;
        if descriptor.expected_kv_length() > state.capacity() {
            return Err(ExecutionError::InvalidRange {
                reason: "causal attention snapshot length exceeds KV capacity".to_owned(),
            });
        }
        {
            let _admission = state
                .operation_admission
                .lock()
                .map_err(|_| ExecutionError::Busy)?;
            if state.append_in_flight.load(Ordering::Acquire)
                || state
                    .attention_in_flight
                    .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                    .is_err()
            {
                return Err(ExecutionError::Busy);
            }
        }
        let snapshot = match self.kv_state_snapshot(state) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                state.attention_in_flight.store(false, Ordering::Release);
                return Err(error);
            }
        };
        if snapshot.length() != descriptor.expected_kv_length() {
            state.attention_in_flight.store(false, Ordering::Release);
            return Err(ExecutionError::StaleKvLength {
                expected: descriptor.expected_kv_length(),
                actual: snapshot.length(),
            });
        }
        let (inner, dispatch) = match self.state.adapter.execute_causal_attention(
            &ExecutionAdapterAccess { session: self },
            state,
            queue,
            &query,
            &output,
            descriptor,
        ) {
            Ok(result) => result,
            Err(error) => {
                state.attention_in_flight.store(false, Ordering::Release);
                return Err(error);
            }
        };
        Ok(CausalAttentionSubmission {
            state: state.clone(),
            queue: queue.clone(),
            query,
            output,
            descriptor,
            dispatch,
            attention_in_flight: Arc::clone(&state.attention_in_flight),
            completion_state: ExecutionState::Pending,
            inner: Some(inner),
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

    fn downcast_kv_state_payload<'a, T: Any + Send + Sync>(
        &self,
        state: &'a KvState,
    ) -> Result<&'a T, ExecutionError> {
        self.ensure_kv_state(state)?;
        state
            .payload
            .downcast_ref::<T>()
            .ok_or(ExecutionError::WrongBackend {
                expected: self.backend_name(),
                actual: state.backend_name(),
            })
    }

    fn downcast_linear_attention_state_payload<'a, T: Any + Send + Sync>(
        &self,
        state: &'a LinearAttentionState,
    ) -> Result<&'a T, ExecutionError> {
        self.ensure_linear_attention_state(state)?;
        state
            .payload
            .downcast_ref::<T>()
            .ok_or(ExecutionError::WrongBackend {
                expected: self.backend_name(),
                actual: state.backend_name(),
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

    fn ensure_kv_state(&self, state: &KvState) -> Result<(), ExecutionError> {
        ensure_identity(
            self.backend_name(),
            self.id(),
            state.backend_name(),
            state.session_id(),
        )
    }

    fn ensure_linear_attention_state(
        &self,
        state: &LinearAttentionState,
    ) -> Result<(), ExecutionError> {
        ensure_identity(
            self.backend_name(),
            self.id(),
            state.backend_name(),
            state.session_id(),
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

    pub fn downcast_kv_state_payload<'a, T: Any + Send + Sync>(
        &self,
        state: &'a KvState,
    ) -> Result<&'a T, ExecutionError> {
        self.session.downcast_kv_state_payload(state)
    }

    pub fn downcast_linear_attention_state_payload<'a, T: Any + Send + Sync>(
        &self,
        state: &'a LinearAttentionState,
    ) -> Result<&'a T, ExecutionError> {
        self.session.downcast_linear_attention_state_payload(state)
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

fn checked_shape_bytes(dtype: DType, shape: &[u64]) -> Result<u64, ExecutionError> {
    let elements = shape.iter().try_fold(1_u64, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| ExecutionError::InvalidRange {
                reason: "state allocation element count overflowed".to_owned(),
            })
    })?;
    elements
        .checked_mul(dtype.size_bytes())
        .ok_or_else(|| ExecutionError::InvalidRange {
            reason: "state allocation byte count overflowed".to_owned(),
        })
}

fn kv_state_allocation_bytes(descriptor: KvStateDescriptor) -> Option<u64> {
    let one = checked_shape_bytes(descriptor.dtype(), &descriptor.storage_shape()).ok()?;
    one.checked_mul(2)
}

fn linear_state_allocation_bytes(descriptor: LinearAttentionStateDescriptor) -> Option<u64> {
    let layout = descriptor.layout();
    let convolution = checked_shape_bytes(
        crate::linear_attention::LinearAttentionLayout::CONV_STATE_DTYPE,
        &layout.conv_state_shape(),
    )
    .ok()?;
    let recurrent = checked_shape_bytes(
        crate::linear_attention::LinearAttentionLayout::RECURRENT_STATE_DTYPE,
        &layout.recurrent_state_shape(),
    )
    .ok()?;
    convolution.checked_add(recurrent)
}

/// Opaque, session-owned device allocation.
#[derive(Clone)]
pub struct ExecutionBuffer {
    state: Arc<ExecutionSessionState>,
    id: ExecutionBufferId,
    size_bytes: u64,
    payload: Arc<dyn Any + Send + Sync>,
    _allocation: Arc<AllocationLease>,
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

/// Opaque, backend-owned request-local full-attention KV state.
#[derive(Clone)]
pub struct KvState {
    state: Arc<ExecutionSessionState>,
    id: KvStateId,
    descriptor: KvStateDescriptor,
    payload: Arc<dyn Any + Send + Sync>,
    append_in_flight: Arc<AtomicBool>,
    attention_in_flight: Arc<AtomicBool>,
    operation_admission: Arc<Mutex<()>>,
    _allocation: Option<Arc<AllocationLease>>,
}

impl fmt::Debug for KvState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KvState")
            .field("id", &self.id)
            .field("session_id", &self.session_id())
            .field("layer_id", &self.descriptor.layer_id())
            .field("capacity", &self.descriptor.capacity())
            .finish_non_exhaustive()
    }
}

impl KvState {
    pub const fn id(&self) -> KvStateId {
        self.id
    }

    pub fn session_id(&self) -> ExecutionSessionId {
        self.state.id
    }

    pub fn backend_name(&self) -> &'static str {
        self.state.backend
    }

    pub const fn descriptor(&self) -> KvStateDescriptor {
        self.descriptor
    }

    pub const fn layer_id(&self) -> u32 {
        self.descriptor.layer_id()
    }

    pub const fn capacity(&self) -> u64 {
        self.descriptor.capacity()
    }

    pub fn snapshot(&self, session: &ExecutionSession) -> Result<KvStateSnapshot, ExecutionError> {
        session.kv_state_snapshot(self)
    }
}

/// Opaque, backend-owned request-local short-convolution and recurrent state.
#[derive(Clone)]
pub struct LinearAttentionState {
    state: Arc<ExecutionSessionState>,
    id: LinearAttentionStateId,
    descriptor: LinearAttentionStateDescriptor,
    payload: Arc<dyn Any + Send + Sync>,
    execution_in_flight: Arc<AtomicBool>,
    _allocation: Option<Arc<AllocationLease>>,
}

impl fmt::Debug for LinearAttentionState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LinearAttentionState")
            .field("id", &self.id)
            .field("session_id", &self.session_id())
            .field("layer_id", &self.descriptor.layer_id())
            .field("capacity", &self.descriptor.capacity())
            .finish_non_exhaustive()
    }
}

impl LinearAttentionState {
    pub const fn id(&self) -> LinearAttentionStateId {
        self.id
    }

    pub fn session_id(&self) -> ExecutionSessionId {
        self.state.id
    }

    pub fn backend_name(&self) -> &'static str {
        self.state.backend
    }

    pub const fn descriptor(&self) -> LinearAttentionStateDescriptor {
        self.descriptor
    }

    pub const fn layer_id(&self) -> u32 {
        self.descriptor.layer_id()
    }

    pub const fn capacity(&self) -> u64 {
        self.descriptor.capacity()
    }

    pub fn snapshot(
        &self,
        session: &ExecutionSession,
    ) -> Result<LinearAttentionStateSnapshot, ExecutionError> {
        session.linear_attention_state_snapshot(self)
    }
}

/// Owned projected inputs, weights, and output for one C4 state transition.
/// Construction is cheap; exact shape/access/alias validation occurs when it
/// is submitted through an execution session.
pub struct LinearAttentionBindings {
    qkv: OwnedTensorBinding,
    z: OwnedTensorBinding,
    b_input: OwnedTensorBinding,
    a_input: OwnedTensorBinding,
    conv_weight: OwnedTensorBinding,
    a_log: OwnedTensorBinding,
    dt_bias: OwnedTensorBinding,
    norm_weight: OwnedTensorBinding,
    output: OwnedTensorBinding,
}

impl LinearAttentionBindings {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qkv: OwnedTensorBinding,
        z: OwnedTensorBinding,
        b_input: OwnedTensorBinding,
        a_input: OwnedTensorBinding,
        conv_weight: OwnedTensorBinding,
        a_log: OwnedTensorBinding,
        dt_bias: OwnedTensorBinding,
        norm_weight: OwnedTensorBinding,
        output: OwnedTensorBinding,
    ) -> Self {
        Self {
            qkv,
            z,
            b_input,
            a_input,
            conv_weight,
            a_log,
            dt_bias,
            norm_weight,
            output,
        }
    }

    pub fn qkv(&self) -> &OwnedTensorBinding {
        &self.qkv
    }
    pub fn z(&self) -> &OwnedTensorBinding {
        &self.z
    }
    pub fn b_input(&self) -> &OwnedTensorBinding {
        &self.b_input
    }
    pub fn a_input(&self) -> &OwnedTensorBinding {
        &self.a_input
    }
    pub fn conv_weight(&self) -> &OwnedTensorBinding {
        &self.conv_weight
    }
    pub fn a_log(&self) -> &OwnedTensorBinding {
        &self.a_log
    }
    pub fn dt_bias(&self) -> &OwnedTensorBinding {
        &self.dt_bias
    }
    pub fn norm_weight(&self) -> &OwnedTensorBinding {
        &self.norm_weight
    }
    pub fn output(&self) -> &OwnedTensorBinding {
        &self.output
    }
}

impl fmt::Debug for LinearAttentionBindings {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LinearAttentionBindings")
            .field("qkv", &self.qkv.buffer().id())
            .field("output", &self.output.buffer().id())
            .finish_non_exhaustive()
    }
}

/// Completion owner for one transactional C4 state transition.
pub struct LinearAttentionSubmission {
    state: LinearAttentionState,
    queue: ExecutionQueue,
    bindings: LinearAttentionBindings,
    request: LinearAttentionRequest,
    dispatch: DispatchEvidence,
    completion_state: ExecutionState,
    inner: Option<Box<dyn ExecutionLinearAttentionSubmissionAdapter>>,
}

impl fmt::Debug for LinearAttentionSubmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LinearAttentionSubmission")
            .field("state", &self.state.id())
            .field("queue", &self.queue.id())
            .field("token_count", &self.request.descriptor().token_count())
            .field("completion_state", &self.completion_state)
            .finish_non_exhaustive()
    }
}

impl LinearAttentionSubmission {
    pub fn state(&self) -> &LinearAttentionState {
        &self.state
    }
    pub fn bindings(&self) -> &LinearAttentionBindings {
        &self.bindings
    }
    pub const fn request(&self) -> LinearAttentionRequest {
        self.request
    }
    pub fn dispatch(&self) -> &DispatchEvidence {
        &self.dispatch
    }
    pub const fn completion_state(&self) -> ExecutionState {
        self.completion_state
    }

    pub fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        let Some(inner) = self.inner.as_mut() else {
            return Ok(self.completion_state);
        };
        let completion = inner.query()?;
        self.record_completion(completion);
        Ok(completion)
    }

    pub fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        let Some(inner) = self.inner.as_mut() else {
            return Ok(self.completion_state);
        };
        let completion = inner.wait(timeout)?;
        self.record_completion(completion);
        Ok(completion)
    }

    fn record_completion(&mut self, completion: ExecutionState) {
        self.completion_state = completion;
        if completion != ExecutionState::Pending {
            // Keep core admission closed while adapter drop releases the native
            // completion or transfers its cleanup ownership. Native admission
            // remains the final safety boundary if that cleanup is quarantined.
            drop(self.inner.take());
            self.state
                .execution_in_flight
                .store(false, Ordering::Release);
        }
    }
}

impl Drop for LinearAttentionSubmission {
    fn drop(&mut self) {
        // Backend cancellation/cleanup runs before a new transition may be
        // admitted for this state.
        drop(self.inner.take());
        self.state
            .execution_in_flight
            .store(false, Ordering::Release);
    }
}

/// Distinct asynchronous completion for a transactional KV append.
///
/// The state, queue, and both input bindings are intentionally retained in
/// this owner even though the adapter receives only borrowed views while the
/// append callback runs.
pub struct KvStateAppendSubmission {
    state: KvState,
    queue: ExecutionQueue,
    key: OwnedTensorBinding,
    value: OwnedTensorBinding,
    request: KvStateAppendRequest,
    dispatch: DispatchEvidence,
    completion_state: ExecutionState,
    inner: Option<Box<dyn ExecutionKvStateSubmissionAdapter>>,
}

impl fmt::Debug for KvStateAppendSubmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KvStateAppendSubmission")
            .field("state", &self.state.id())
            .field("queue", &self.queue.id())
            .field("token_count", &self.request.token_count())
            .field("completion_state", &self.completion_state)
            .finish_non_exhaustive()
    }
}

impl KvStateAppendSubmission {
    pub fn state(&self) -> &KvState {
        &self.state
    }

    pub fn key_binding(&self) -> &OwnedTensorBinding {
        &self.key
    }

    pub fn value_binding(&self) -> &OwnedTensorBinding {
        &self.value
    }

    pub const fn request(&self) -> KvStateAppendRequest {
        self.request
    }

    pub fn dispatch(&self) -> &DispatchEvidence {
        &self.dispatch
    }

    pub const fn completion_state(&self) -> ExecutionState {
        self.completion_state
    }

    pub fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        let state = self
            .inner
            .as_mut()
            .expect("KV submission adapter remains owned until drop")
            .query()?;
        self.record_completion(state);
        Ok(state)
    }

    pub fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        let state = self
            .inner
            .as_mut()
            .expect("KV submission adapter remains owned until drop")
            .wait(timeout)?;
        self.record_completion(state);
        Ok(state)
    }

    fn record_completion(&mut self, state: ExecutionState) {
        self.completion_state = state;
        if state != ExecutionState::Pending {
            self.state.append_in_flight.store(false, Ordering::Release);
        }
    }
}

impl Drop for KvStateAppendSubmission {
    fn drop(&mut self) {
        // Revoke or transfer backend commit/cleanup ownership before allowing a
        // concurrent core append to be admitted for the same state.
        drop(self.inner.take());
        self.state.append_in_flight.store(false, Ordering::Release);
    }
}

/// Distinct asynchronous completion for one causal GQA attention dispatch.
/// The retained state clone prevents native K/V release while the completion
/// owner (or its cleanup reaper) still exists.
pub struct CausalAttentionSubmission {
    state: KvState,
    queue: ExecutionQueue,
    query: OwnedTensorBinding,
    output: OwnedTensorBinding,
    descriptor: CausalAttentionDescriptor,
    dispatch: DispatchEvidence,
    attention_in_flight: Arc<AtomicBool>,
    completion_state: ExecutionState,
    inner: Option<Box<dyn ExecutionCausalAttentionSubmissionAdapter>>,
}

impl Drop for CausalAttentionSubmission {
    fn drop(&mut self) {
        drop(self.inner.take());
        self.attention_in_flight.store(false, Ordering::Release);
    }
}

impl fmt::Debug for CausalAttentionSubmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CausalAttentionSubmission")
            .field("state", &self.state.id())
            .field("queue", &self.queue.id())
            .field("query_count", &self.descriptor.query_count())
            .field("completion_state", &self.completion_state)
            .finish_non_exhaustive()
    }
}

impl CausalAttentionSubmission {
    pub fn state(&self) -> &KvState {
        &self.state
    }

    pub fn query_binding(&self) -> &OwnedTensorBinding {
        &self.query
    }

    pub fn output_binding(&self) -> &OwnedTensorBinding {
        &self.output
    }

    pub const fn descriptor(&self) -> CausalAttentionDescriptor {
        self.descriptor
    }

    pub const fn completion_state(&self) -> ExecutionState {
        self.completion_state
    }

    pub fn dispatch(&self) -> &DispatchEvidence {
        &self.dispatch
    }

    pub fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        let state = self
            .inner
            .as_mut()
            .expect("attention submission adapter remains owned until drop")
            .query()?;
        self.completion_state = state;
        Ok(state)
    }

    pub fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        let state = self
            .inner
            .as_mut()
            .expect("attention submission adapter remains owned until drop")
            .wait(timeout)?;
        self.completion_state = state;
        Ok(state)
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
                role: "semantic operation binding arity",
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

        if matches!(
            descriptor.kind(),
            SemanticOpKind::Copy
                | SemanticOpKind::Add
                | SemanticOpKind::SiluMul
                | SemanticOpKind::SigmoidMul
        ) {
            validate_elementwise_nonoverlap(descriptor.kind(), &inputs, &outputs)?;
        } else if descriptor.kind() == SemanticOpKind::Embedding {
            validate_nonoverlap(&[
                ("embedding weight", &inputs[0]),
                ("embedding token IDs", &inputs[1]),
                ("embedding output", &outputs[0]),
            ])?;
        } else if descriptor.kind() == SemanticOpKind::Matmul {
            validate_nonoverlap(&[
                ("matmul activation", &inputs[0]),
                ("matmul weight", &inputs[1]),
                ("matmul output", &outputs[0]),
            ])?;
        } else if descriptor.kind() == SemanticOpKind::RmsNorm {
            validate_rmsnorm_nonoverlap(&inputs, &outputs)?;
        } else if descriptor.kind() == SemanticOpKind::AttentionPreprocess {
            validate_nonoverlap(&[
                ("attention_preprocess packed Q/gate", &inputs[0]),
                ("attention_preprocess K", &inputs[1]),
                ("attention_preprocess Q raw scale", &inputs[2]),
                ("attention_preprocess K raw scale", &inputs[3]),
                ("attention_preprocess positions", &inputs[4]),
                ("attention_preprocess Q output", &outputs[0]),
                ("attention_preprocess gate output", &outputs[1]),
                ("attention_preprocess K output", &outputs[2]),
            ])?;
        } else if descriptor.kind() == SemanticOpKind::Argmax {
            validate_nonoverlap(&[
                ("argmax logits", &inputs[0]),
                ("argmax output", &outputs[0]),
            ])?;
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
        (SemanticOpKind::Copy, 0) => "copy input",
        (SemanticOpKind::Add, 0) => "add input 0",
        (SemanticOpKind::Add, 1) => "add input 1",
        (SemanticOpKind::Embedding, 0) => "embedding weight",
        (SemanticOpKind::Embedding, 1) => "embedding token IDs",
        (SemanticOpKind::Matmul, 0) => "matmul activation",
        (SemanticOpKind::Matmul, 1) => "matmul weight",
        (SemanticOpKind::SiluMul, 0) => "silu_mul gate",
        (SemanticOpKind::SiluMul, 1) => "silu_mul up",
        (SemanticOpKind::RmsNorm, 0) => "RMSNorm activation",
        (SemanticOpKind::RmsNorm, 1) => "RMSNorm raw scale",
        (SemanticOpKind::Argmax, 0) => "argmax logits",
        (SemanticOpKind::AttentionPreprocess, 0) => "attention_preprocess packed Q/gate",
        (SemanticOpKind::AttentionPreprocess, 1) => "attention_preprocess K",
        (SemanticOpKind::AttentionPreprocess, 2) => "attention_preprocess Q raw scale",
        (SemanticOpKind::AttentionPreprocess, 3) => "attention_preprocess K raw scale",
        (SemanticOpKind::AttentionPreprocess, 4) => "attention_preprocess positions",
        _ => "input",
    }
}

fn output_role(kind: SemanticOpKind, index: usize) -> &'static str {
    match (kind, index) {
        (SemanticOpKind::Copy, 0) => "copy output",
        (SemanticOpKind::Add, 0) => "add output",
        (SemanticOpKind::Embedding, 0) => "embedding output",
        (SemanticOpKind::Matmul, 0) => "matmul output",
        (SemanticOpKind::SiluMul, 0) => "silu_mul output",
        (SemanticOpKind::RmsNorm, 0) => "RMSNorm output",
        (SemanticOpKind::Argmax, 0) => "argmax output",
        (SemanticOpKind::AttentionPreprocess, 0) => "attention_preprocess Q output",
        (SemanticOpKind::AttentionPreprocess, 1) => "attention_preprocess gate output",
        (SemanticOpKind::AttentionPreprocess, 2) => "attention_preprocess K output",
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

fn validate_elementwise_nonoverlap(
    kind: SemanticOpKind,
    inputs: &[OwnedTensorBinding],
    outputs: &[OwnedTensorBinding],
) -> Result<(), ExecutionError> {
    let entries: Vec<(&str, &OwnedTensorBinding)> = match kind {
        SemanticOpKind::Copy => vec![("copy input", &inputs[0]), ("copy output", &outputs[0])],
        SemanticOpKind::Add => vec![
            ("add input 0", &inputs[0]),
            ("add input 1", &inputs[1]),
            ("add output", &outputs[0]),
        ],
        SemanticOpKind::SiluMul => vec![
            ("silu_mul gate", &inputs[0]),
            ("silu_mul up", &inputs[1]),
            ("silu_mul output", &outputs[0]),
        ],
        SemanticOpKind::SigmoidMul => vec![
            ("sigmoid_mul gate", &inputs[0]),
            ("sigmoid_mul attention value", &inputs[1]),
            ("sigmoid_mul output", &outputs[0]),
        ],
        _ => unreachable!("elementwise overlap is only used by copy/add"),
    };
    validate_nonoverlap(&entries)
}

fn validate_nonoverlap(
    entries: &[(&'static str, &OwnedTensorBinding)],
) -> Result<(), ExecutionError> {
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

fn validate_kv_state_snapshot(
    session: &ExecutionSession,
    state: &KvState,
    snapshot: KvStateSnapshot,
) -> Result<KvStateSnapshot, ExecutionError> {
    if snapshot.session_id() != session.id() {
        return Err(ExecutionError::WrongSession {
            expected: session.id(),
            actual: snapshot.session_id(),
        });
    }
    if snapshot.state_id() != state.id() {
        return Err(ExecutionError::WrongKvState {
            expected: state.id(),
            actual: snapshot.state_id(),
        });
    }
    if snapshot.descriptor() != state.descriptor() {
        return Err(ExecutionError::InvalidRequest {
            reason: "backend KV snapshot descriptor does not match the state".to_owned(),
        });
    }
    if snapshot.length() > state.capacity() {
        return Err(ExecutionError::InvalidRequest {
            reason: "backend KV snapshot length exceeds state capacity".to_owned(),
        });
    }
    Ok(snapshot)
}

fn validate_kv_append_bindings(
    session: &ExecutionSession,
    key: &OwnedTensorBinding,
    value: &OwnedTensorBinding,
    layout: KvStateLayout,
) -> Result<u64, ExecutionError> {
    validate_kv_append_binding(session, key, "KV key", layout)?;
    validate_kv_append_binding(session, value, "KV value", layout)?;

    let key_shape = key.view().shape();
    let value_shape = value.view().shape();
    if key_shape[0] != value_shape[0] {
        return Err(ExecutionError::InvalidRequest {
            reason: "KV key and value token counts must match".to_owned(),
        });
    }
    if key.buffer().id() == value.buffer().id()
        && intervals_overlap(
            key.view().byte_offset(),
            key.view().end_offset(),
            value.view().byte_offset(),
            value.view().end_offset(),
        )
    {
        return Err(ExecutionError::AliasOverlap {
            left: "KV key",
            right: "KV value",
        });
    }
    u64::try_from(key_shape[0]).map_err(|_| ExecutionError::InvalidRequest {
        reason: "KV token count does not fit u64".to_owned(),
    })
}

fn validate_causal_attention_bindings(
    session: &ExecutionSession,
    query: &OwnedTensorBinding,
    output: &OwnedTensorBinding,
    descriptor: CausalAttentionDescriptor,
    layout: KvStateLayout,
) -> Result<(), ExecutionError> {
    let query_heads =
        layout
            .heads()
            .checked_mul(4)
            .ok_or_else(|| ExecutionError::InvalidRequest {
                reason: "causal attention query head count overflowed".to_owned(),
            })?;
    for (role, binding, required) in [
        ("causal attention query", query, AccessMode::Read),
        ("causal attention output", output, AccessMode::Write),
    ] {
        validate_binding_identity(session.backend_name(), session.id(), binding)?;
        ensure_view_in_bounds(binding.buffer(), binding.view())?;
        let permitted = match required {
            AccessMode::Read => binding.access().permits_read(),
            AccessMode::Write => binding.access().permits_write(),
            AccessMode::ReadWrite => {
                binding.access().permits_read() && binding.access().permits_write()
            }
        };
        if !permitted {
            return Err(ExecutionError::AccessViolation {
                role,
                required,
                actual: binding.access(),
            });
        }
        let view = binding.view();
        if view.dtype() != DType::Bf16 || view.encoding() != Encoding::Unquantized {
            return Err(ExecutionError::InvalidRequest {
                reason: format!("{role} must be contiguous unquantized BF16"),
            });
        }
        let query_count = usize::try_from(descriptor.query_count()).map_err(|_| {
            ExecutionError::InvalidRequest {
                reason: "causal attention query count does not fit the host index type".to_owned(),
            }
        })?;
        let shape = view.shape();
        let strides = view.strides();
        if shape != [query_count, query_heads, layout.head_dim()]
            || strides != [query_heads * layout.head_dim(), layout.head_dim(), 1]
            || !view.is_contiguous()
        {
            return Err(ExecutionError::InvalidRequest {
                reason: format!(
                    "{role} shape and strides must match the contiguous reviewed query-head layout"
                ),
            });
        }
    }
    if query.buffer().id() == output.buffer().id()
        && intervals_overlap(
            query.view().byte_offset(),
            query.view().end_offset(),
            output.view().byte_offset(),
            output.view().end_offset(),
        )
    {
        return Err(ExecutionError::AliasOverlap {
            left: "causal attention query",
            right: "causal attention output",
        });
    }
    Ok(())
}

fn validate_linear_attention_state_snapshot(
    session: &ExecutionSession,
    state: &LinearAttentionState,
    snapshot: LinearAttentionStateSnapshot,
) -> Result<LinearAttentionStateSnapshot, ExecutionError> {
    if snapshot.session_id() != session.id() {
        return Err(ExecutionError::WrongSession {
            expected: session.id(),
            actual: snapshot.session_id(),
        });
    }
    if snapshot.state_id() != state.id() {
        return Err(ExecutionError::WrongLinearAttentionState {
            expected: state.id(),
            actual: snapshot.state_id(),
        });
    }
    if snapshot.descriptor() != state.descriptor() {
        return Err(ExecutionError::InvalidRequest {
            reason: "backend linear-attention snapshot descriptor does not match the state"
                .to_owned(),
        });
    }
    if snapshot.length() > state.capacity() {
        return Err(ExecutionError::InvalidRequest {
            reason: "backend linear-attention snapshot length exceeds state capacity".to_owned(),
        });
    }
    Ok(snapshot)
}

fn validate_linear_attention_bindings(
    session: &ExecutionSession,
    bindings: &LinearAttentionBindings,
    descriptor: LinearAttentionDescriptor,
    layout: LinearAttentionLayout,
) -> Result<(), ExecutionError> {
    let token_count =
        usize::try_from(descriptor.token_count()).map_err(|_| ExecutionError::InvalidRequest {
            reason: "linear-attention token count does not fit the host index type".to_owned(),
        })?;
    let qkv_shape = [token_count, layout.qkv_width()];
    let output_shape = [token_count, layout.output_width()];
    let scalar_shape = [token_count, layout.value_heads()];
    validate_linear_attention_binding(
        session,
        bindings.qkv(),
        "linear attention qkv",
        AccessMode::Read,
        DType::Bf16,
        &qkv_shape,
    )?;
    validate_linear_attention_binding(
        session,
        bindings.z(),
        "linear attention z",
        AccessMode::Read,
        DType::Bf16,
        &output_shape,
    )?;
    validate_linear_attention_binding(
        session,
        bindings.b_input(),
        "linear attention b input",
        AccessMode::Read,
        DType::Bf16,
        &scalar_shape,
    )?;
    validate_linear_attention_binding(
        session,
        bindings.a_input(),
        "linear attention a input",
        AccessMode::Read,
        DType::Bf16,
        &scalar_shape,
    )?;
    validate_linear_attention_binding(
        session,
        bindings.conv_weight(),
        "linear attention convolution weight",
        AccessMode::Read,
        DType::Bf16,
        &[layout.qkv_width(), 1, layout.conv_kernel_size()],
    )?;
    validate_linear_attention_binding(
        session,
        bindings.a_log(),
        "linear attention A_log",
        AccessMode::Read,
        DType::F32,
        &[layout.value_heads()],
    )?;
    validate_linear_attention_binding(
        session,
        bindings.dt_bias(),
        "linear attention dt_bias",
        AccessMode::Read,
        DType::Bf16,
        &[layout.value_heads()],
    )?;
    validate_linear_attention_binding(
        session,
        bindings.norm_weight(),
        "linear attention norm weight",
        AccessMode::Read,
        DType::F32,
        &[layout.head_dim()],
    )?;
    validate_linear_attention_binding(
        session,
        bindings.output(),
        "linear attention output",
        AccessMode::Write,
        DType::Bf16,
        &output_shape,
    )?;

    validate_nonoverlap(&[
        ("linear attention qkv", bindings.qkv()),
        ("linear attention z", bindings.z()),
        ("linear attention b input", bindings.b_input()),
        ("linear attention a input", bindings.a_input()),
        (
            "linear attention convolution weight",
            bindings.conv_weight(),
        ),
        ("linear attention A_log", bindings.a_log()),
        ("linear attention dt_bias", bindings.dt_bias()),
        ("linear attention norm weight", bindings.norm_weight()),
        ("linear attention output", bindings.output()),
    ])
}

fn validate_linear_attention_binding(
    session: &ExecutionSession,
    binding: &OwnedTensorBinding,
    role: &'static str,
    required_access: AccessMode,
    dtype: DType,
    shape: &[usize],
) -> Result<(), ExecutionError> {
    validate_binding_identity(session.backend_name(), session.id(), binding)?;
    ensure_view_in_bounds(binding.buffer(), binding.view())?;
    let permitted = match required_access {
        AccessMode::Read => binding.access().permits_read(),
        AccessMode::Write => binding.access().permits_write(),
        AccessMode::ReadWrite => {
            binding.access().permits_read() && binding.access().permits_write()
        }
    };
    if !permitted {
        return Err(ExecutionError::AccessViolation {
            role,
            required: required_access,
            actual: binding.access(),
        });
    }
    let view = binding.view();
    if view.dtype() != dtype
        || view.encoding() != Encoding::Unquantized
        || view.shape() != shape
        || !view.is_contiguous()
    {
        return Err(ExecutionError::InvalidRequest {
            reason: format!("{role} must be contiguous unquantized {dtype:?} with shape {shape:?}"),
        });
    }
    Ok(())
}

fn validate_kv_append_binding(
    session: &ExecutionSession,
    binding: &OwnedTensorBinding,
    role: &'static str,
    layout: KvStateLayout,
) -> Result<(), ExecutionError> {
    validate_binding_identity(session.backend_name(), session.id(), binding)?;
    ensure_view_in_bounds(binding.buffer(), binding.view())?;
    if !binding.access().permits_read() {
        return Err(ExecutionError::AccessViolation {
            role,
            required: AccessMode::Read,
            actual: binding.access(),
        });
    }
    let shape = binding.view().shape();
    if shape.len() != 3 || shape[1] != layout.heads() || shape[2] != layout.head_dim() {
        return Err(ExecutionError::InvalidRequest {
            reason: format!("{role} must match the state KV head layout"),
        });
    }
    if shape[0] == 0 {
        return Err(ExecutionError::InvalidRequest {
            reason: format!("{role} token count must be non-zero"),
        });
    }
    if binding.view().dtype() != crate::DType::Bf16 {
        return Err(ExecutionError::InvalidRequest {
            reason: format!("{role} must use BF16 storage"),
        });
    }
    if binding.view().encoding() != crate::Encoding::Unquantized {
        return Err(ExecutionError::InvalidRequest {
            reason: format!("{role} must use unquantized encoding"),
        });
    }
    if !binding.view().is_contiguous() {
        return Err(ExecutionError::InvalidRequest {
            reason: format!("{role} must be row-major contiguous"),
        });
    }
    Ok(())
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

    /// Backend event timing for the submitted kernel, when available.  The
    /// submission must have reached a terminal state; host-only adapters may
    /// return `None` and must not fabricate GPU time.
    pub fn kernel_elapsed_ns(&mut self) -> Result<Option<u64>, ExecutionError> {
        if self.completion_state != ExecutionState::Success {
            return Err(ExecutionError::NotReady);
        }
        self.inner.kernel_elapsed_ns()
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

    struct DropPayload(Arc<AtomicUsize>);

    impl Drop for DropPayload {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    struct FakeStateEntry {
        id: KvStateId,
        descriptor: crate::KvStateDescriptor,
        length: u64,
    }

    struct FakeStateStore {
        entries: Mutex<Vec<FakeStateEntry>>,
        linear_entries: Mutex<Vec<FakeLinearStateEntry>>,
        append_drops: Arc<AtomicUsize>,
        append_calls: AtomicUsize,
        linear_drops: Arc<AtomicUsize>,
        linear_calls: AtomicUsize,
        linear_drop_saw_core_admission: AtomicBool,
        append_drop_saw_core_admission: AtomicBool,
        wrong_snapshot_identity: AtomicBool,
    }

    struct FakeLinearStateEntry {
        id: LinearAttentionStateId,
        descriptor: LinearAttentionStateDescriptor,
        length: u64,
    }

    struct KvStateAdapter {
        base: TestAdapter,
        store: Arc<FakeStateStore>,
        state_resource_drops: Arc<AtomicUsize>,
        buffer_drops: Arc<AtomicUsize>,
        causal_attention_calls: AtomicUsize,
        causal_attention_drops: Arc<AtomicUsize>,
    }

    impl KvStateAdapter {
        fn new() -> Self {
            Self {
                base: TestAdapter::default(),
                store: Arc::new(FakeStateStore {
                    entries: Mutex::new(Vec::new()),
                    linear_entries: Mutex::new(Vec::new()),
                    append_drops: Arc::new(AtomicUsize::new(0)),
                    append_calls: AtomicUsize::new(0),
                    linear_drops: Arc::new(AtomicUsize::new(0)),
                    linear_calls: AtomicUsize::new(0),
                    linear_drop_saw_core_admission: AtomicBool::new(false),
                    append_drop_saw_core_admission: AtomicBool::new(false),
                    wrong_snapshot_identity: AtomicBool::new(false),
                }),
                state_resource_drops: Arc::new(AtomicUsize::new(0)),
                buffer_drops: Arc::new(AtomicUsize::new(0)),
                causal_attention_calls: AtomicUsize::new(0),
                causal_attention_drops: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    struct FakeKvSubmission {
        store: Arc<FakeStateStore>,
        request: crate::KvStateAppendRequest,
        core_append_in_flight: Arc<AtomicBool>,
        complete: bool,
    }

    impl FakeKvSubmission {
        fn finish(&mut self) -> Result<ExecutionState, ExecutionError> {
            if !self.complete {
                let mut entries = self
                    .store
                    .entries
                    .lock()
                    .map_err(|_| ExecutionError::Busy)?;
                let entry = entries
                    .iter_mut()
                    .find(|entry| entry.id == self.request.state_id())
                    .ok_or(ExecutionError::WrongKvState {
                        expected: self.request.state_id(),
                        actual: KvStateId::new(1),
                    })?;
                if entry.length != self.request.expected_length() {
                    return Err(ExecutionError::StaleKvLength {
                        expected: self.request.expected_length(),
                        actual: entry.length,
                    });
                }
                entry.length = self.request.end_position();
                self.complete = true;
            }
            Ok(ExecutionState::Success)
        }
    }

    impl Drop for FakeKvSubmission {
        fn drop(&mut self) {
            self.store.append_drop_saw_core_admission.store(
                self.core_append_in_flight.load(Ordering::Acquire),
                Ordering::Release,
            );
            self.store.append_drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl ExecutionKvStateSubmissionAdapter for FakeKvSubmission {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            self.finish()
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            self.finish()
        }
    }

    struct FakeCausalAttentionSubmission {
        drops: Arc<AtomicUsize>,
    }

    impl Drop for FakeCausalAttentionSubmission {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl ExecutionCausalAttentionSubmissionAdapter for FakeCausalAttentionSubmission {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }
    }

    struct FakeLinearAttentionSubmission {
        store: Arc<FakeStateStore>,
        request: LinearAttentionRequest,
        core_execution_in_flight: Arc<AtomicBool>,
        complete: bool,
    }

    impl FakeLinearAttentionSubmission {
        fn finish(&mut self) -> Result<ExecutionState, ExecutionError> {
            if !self.complete {
                let mut entries = self
                    .store
                    .linear_entries
                    .lock()
                    .map_err(|_| ExecutionError::Busy)?;
                let entry = entries
                    .iter_mut()
                    .find(|entry| entry.id == self.request.state_id())
                    .ok_or(ExecutionError::WrongLinearAttentionState {
                        expected: self.request.state_id(),
                        actual: LinearAttentionStateId::new(1),
                    })?;
                let descriptor = self.request.descriptor();
                if entry.length != descriptor.start_position() {
                    return Err(ExecutionError::StaleLinearAttentionLength {
                        expected: descriptor.start_position(),
                        actual: entry.length,
                    });
                }
                entry.length = descriptor.expected_length();
                self.complete = true;
            }
            Ok(ExecutionState::Success)
        }
    }

    impl Drop for FakeLinearAttentionSubmission {
        fn drop(&mut self) {
            self.store.linear_drop_saw_core_admission.store(
                self.core_execution_in_flight.load(Ordering::Acquire),
                Ordering::Release,
            );
            self.store.linear_drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl ExecutionLinearAttentionSubmissionAdapter for FakeLinearAttentionSubmission {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            self.finish()
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            self.finish()
        }
    }

    impl ExecutionSessionAdapter for KvStateAdapter {
        fn max_transfer_bytes(&self) -> u64 {
            self.base.max_transfer_bytes()
        }

        fn supports(&self, descriptor: &SemanticOpDescriptor) -> PrepareSupport {
            self.base.supports(descriptor)
        }

        fn create_queue(
            &self,
            access: &ExecutionAdapterAccess<'_>,
        ) -> Result<AdapterResource, ExecutionError> {
            self.base.create_queue(access)
        }

        fn allocate(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _size_bytes: u64,
        ) -> Result<AdapterResource, ExecutionError> {
            Ok(AdapterResource::new(DropPayload(Arc::clone(
                &self.buffer_drops,
            ))))
        }

        fn prepare(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            operation: &BoundSemanticOp,
        ) -> Result<AdapterResource, ExecutionError> {
            self.base.prepare(access, operation)
        }

        fn submit(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            prepared: &PreparedOperation,
            queue: &ExecutionQueue,
        ) -> Result<(Box<dyn ExecutionSubmissionAdapter>, DispatchEvidence), ExecutionError>
        {
            self.base.submit(access, prepared, queue)
        }

        fn upload(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            queue: &ExecutionQueue,
            destination: &BufferRange,
            bytes: Arc<[u8]>,
        ) -> Result<Box<dyn ExecutionTransferAdapter>, ExecutionError> {
            self.base.upload(access, queue, destination, bytes)
        }

        fn readback(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            queue: &ExecutionQueue,
            source: &BufferRange,
        ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
            self.base.readback(access, queue, source)
        }

        fn shutdown(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            deadline: Duration,
        ) -> Result<ShutdownReport, ExecutionError> {
            self.base.shutdown(access, deadline)
        }

        fn create_kv_state(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state_id: KvStateId,
            descriptor: crate::KvStateDescriptor,
        ) -> Result<AdapterResource, ExecutionError> {
            self.store
                .entries
                .lock()
                .map_err(|_| ExecutionError::Busy)?
                .push(FakeStateEntry {
                    id: state_id,
                    descriptor,
                    length: 0,
                });
            Ok(AdapterResource::new(DropPayload(Arc::clone(
                &self.state_resource_drops,
            ))))
        }

        fn kv_state_snapshot(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            state: &KvState,
        ) -> Result<crate::KvStateSnapshot, ExecutionError> {
            let entries = self
                .store
                .entries
                .lock()
                .map_err(|_| ExecutionError::Busy)?;
            let entry = entries.iter().find(|entry| entry.id == state.id()).ok_or(
                ExecutionError::WrongKvState {
                    expected: state.id(),
                    actual: KvStateId::new(1),
                },
            )?;
            let state_id = if self.store.wrong_snapshot_identity.load(Ordering::Relaxed) {
                KvStateId::new(state.id().raw().checked_add(1).unwrap_or(1))
            } else {
                state.id()
            };
            crate::KvStateSnapshot::new(
                access.session_id(),
                state_id,
                entry.descriptor,
                entry.length,
            )
            .map_err(|error| ExecutionError::InvalidRequest {
                reason: error.to_string(),
            })
        }

        fn append_kv_state(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state: &KvState,
            _queue: &ExecutionQueue,
            _key: &OwnedTensorBinding,
            _value: &OwnedTensorBinding,
            request: &crate::KvStateAppendRequest,
        ) -> Result<(Box<dyn ExecutionKvStateSubmissionAdapter>, DispatchEvidence), ExecutionError>
        {
            let entries = self
                .store
                .entries
                .lock()
                .map_err(|_| ExecutionError::Busy)?;
            let entry = entries.iter().find(|entry| entry.id == state.id()).ok_or(
                ExecutionError::WrongKvState {
                    expected: state.id(),
                    actual: request.state_id(),
                },
            )?;
            if entry.length != request.expected_length() {
                return Err(ExecutionError::StaleKvLength {
                    expected: request.expected_length(),
                    actual: entry.length,
                });
            }
            self.store.append_calls.fetch_add(1, Ordering::Relaxed);
            Ok((
                Box::new(FakeKvSubmission {
                    store: Arc::clone(&self.store),
                    request: *request,
                    core_append_in_flight: Arc::clone(&state.append_in_flight),
                    complete: false,
                }),
                DispatchEvidence {
                    abi_version: 1,
                    info_version: 1,
                    dispatch_id: 1,
                    dispatch_count: 1,
                    kernel_id: 1,
                    workgroup_size_x: 256,
                    grid_size_x: (request.token_count() * 4) as u32,
                    row_count: request.token_count(),
                    normalized_size: 1024,
                    backend: 1,
                    fallback_allowed: false,
                    fallback_used: false,
                    kernel_symbol: "kv_state.bf16_to_f16_token_major.v2".to_owned(),
                    device_symbol: "fake_kv_append".to_owned(),
                    target: "fake".to_owned(),
                },
            ))
        }

        fn execute_causal_attention(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            state: &KvState,
            _queue: &ExecutionQueue,
            _query: &OwnedTensorBinding,
            _output: &OwnedTensorBinding,
            descriptor: CausalAttentionDescriptor,
        ) -> Result<
            (
                Box<dyn ExecutionCausalAttentionSubmissionAdapter>,
                DispatchEvidence,
            ),
            ExecutionError,
        > {
            let snapshot = self.kv_state_snapshot(access, state)?;
            if snapshot.length() != descriptor.expected_kv_length() {
                return Err(ExecutionError::StaleKvLength {
                    expected: descriptor.expected_kv_length(),
                    actual: snapshot.length(),
                });
            }
            self.causal_attention_calls.fetch_add(1, Ordering::Relaxed);
            Ok((
                Box::new(FakeCausalAttentionSubmission {
                    drops: Arc::clone(&self.causal_attention_drops),
                }),
                DispatchEvidence {
                    abi_version: 1,
                    info_version: 1,
                    dispatch_id: 2,
                    dispatch_count: 1,
                    kernel_id: 2,
                    workgroup_size_x: 256,
                    grid_size_x: (descriptor.query_count() * 16) as u32,
                    row_count: descriptor.query_count(),
                    normalized_size: 256,
                    backend: 1,
                    fallback_allowed: false,
                    fallback_used: false,
                    kernel_symbol: "causal_attention.stable_softmax_gqa.v1".to_owned(),
                    device_symbol: "fake_causal_attention".to_owned(),
                    target: "fake".to_owned(),
                },
            ))
        }

        fn create_linear_attention_state(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state_id: LinearAttentionStateId,
            descriptor: LinearAttentionStateDescriptor,
        ) -> Result<AdapterResource, ExecutionError> {
            self.store
                .linear_entries
                .lock()
                .map_err(|_| ExecutionError::Busy)?
                .push(FakeLinearStateEntry {
                    id: state_id,
                    descriptor,
                    length: 0,
                });
            Ok(AdapterResource::new(DropPayload(Arc::clone(
                &self.state_resource_drops,
            ))))
        }

        fn linear_attention_state_snapshot(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            state: &LinearAttentionState,
        ) -> Result<LinearAttentionStateSnapshot, ExecutionError> {
            let entries = self
                .store
                .linear_entries
                .lock()
                .map_err(|_| ExecutionError::Busy)?;
            let entry = entries.iter().find(|entry| entry.id == state.id()).ok_or(
                ExecutionError::WrongLinearAttentionState {
                    expected: state.id(),
                    actual: LinearAttentionStateId::new(1),
                },
            )?;
            LinearAttentionStateSnapshot::new(
                access.session_id(),
                state.id(),
                entry.descriptor,
                entry.length,
            )
            .map_err(|error| ExecutionError::InvalidRequest {
                reason: error.to_string(),
            })
        }

        fn execute_linear_attention(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state: &LinearAttentionState,
            _queue: &ExecutionQueue,
            _bindings: &LinearAttentionBindings,
            request: LinearAttentionRequest,
        ) -> Result<
            (
                Box<dyn ExecutionLinearAttentionSubmissionAdapter>,
                DispatchEvidence,
            ),
            ExecutionError,
        > {
            self.store.linear_calls.fetch_add(1, Ordering::Relaxed);
            Ok((
                Box::new(FakeLinearAttentionSubmission {
                    store: Arc::clone(&self.store),
                    request,
                    core_execution_in_flight: Arc::clone(&state.execution_in_flight),
                    complete: false,
                }),
                DispatchEvidence {
                    abi_version: 1,
                    info_version: 1,
                    dispatch_id: 3,
                    dispatch_count: 2,
                    kernel_id: 3,
                    workgroup_size_x: 128,
                    grid_size_x: LinearAttentionLayout::VALUE_HEADS as u32,
                    row_count: request.descriptor().token_count(),
                    normalized_size: LinearAttentionLayout::HEAD_DIM as u64,
                    backend: 1,
                    fallback_allowed: false,
                    fallback_used: false,
                    kernel_symbol: "linear_attention.gdn.v1".to_owned(),
                    device_symbol: "fake_linear_attention".to_owned(),
                    target: "fake".to_owned(),
                },
            ))
        }
    }

    fn session(name: &'static str) -> ExecutionSession {
        ExecutionSession::new(name, Arc::new(TestAdapter::default()))
    }

    #[test]
    fn allocation_accounting_is_checked_and_raii_returns_request_bytes() {
        let test_session = session("test");
        let resident = test_session
            .allocate_with_category(17, AllocationCategory::ModelResident)
            .unwrap();
        let workspace = test_session.allocate(3).unwrap();
        let snapshot = test_session.allocation_snapshot();
        assert_eq!(snapshot.model_resident().current_bytes(), 17);
        assert_eq!(snapshot.workspace().current_bytes(), 3);
        assert_eq!(snapshot.high_water_bytes(), 20);
        drop(workspace);
        assert_eq!(
            test_session.memory_snapshot().workspace().current_bytes(),
            0
        );
        assert_eq!(
            test_session
                .memory_snapshot()
                .model_resident()
                .current_bytes(),
            17
        );
        drop(resident);
        let final_snapshot = test_session.memory_snapshot();
        assert_eq!(final_snapshot.current_bytes(), 0);
        assert_eq!(final_snapshot.high_water_bytes(), 20);
        assert!(!final_snapshot.poisoned());
    }

    #[test]
    fn allocation_accounting_poison_is_sticky_after_overflow() {
        let accounting = Arc::new(AllocationAccounting::default());
        let huge = accounting
            .reserve(AllocationCategory::Workspace, u64::MAX)
            .unwrap();
        assert!(matches!(
            accounting.reserve(AllocationCategory::Workspace, 1),
            Err(ExecutionError::AllocationAccountingOverflow)
        ));
        assert!(accounting.snapshot().poisoned);
        assert!(matches!(
            accounting.reserve(AllocationCategory::Workspace, 1),
            Err(ExecutionError::AllocationAccountingPoisoned)
        ));
        drop(huge);
        assert!(accounting.snapshot().poisoned);
    }

    #[test]
    fn allocation_accounting_poison_is_sticky_after_underflow() {
        let accounting = Arc::new(AllocationAccounting::default());
        accounting
            .reserve(AllocationCategory::Workspace, 3)
            .unwrap();
        accounting.release(AllocationCategory::Workspace, 4);
        assert!(accounting.snapshot().poisoned);
    }

    #[test]
    fn adapters_without_stateful_methods_remain_source_compatible_and_unsupported() {
        let test_session = session("test");
        let descriptor = crate::KvStateDescriptor::new(0, 1).unwrap();

        assert!(matches!(
            test_session.create_kv_state(descriptor),
            Err(ExecutionError::Unsupported { reason })
                if reason.contains("request-local KV state")
        ));
        assert!(matches!(
            test_session.create_linear_attention_state(
                LinearAttentionStateDescriptor::new(0, 1).unwrap()
            ),
            Err(ExecutionError::Unsupported { reason })
                if reason.contains("linear-attention state")
        ));
    }

    fn kv_session() -> (ExecutionSession, Arc<KvStateAdapter>) {
        let adapter = Arc::new(KvStateAdapter::new());
        let session_adapter: Arc<dyn ExecutionSessionAdapter> = adapter.clone();
        (ExecutionSession::new("kv-test", session_adapter), adapter)
    }

    #[test]
    fn allocation_accounting_includes_both_kv_planes_and_linear_state() {
        let (session, _) = kv_session();
        let kv_descriptor = KvStateDescriptor::new(0, 3).unwrap();
        let linear_descriptor = LinearAttentionStateDescriptor::new(1, 3).unwrap();
        let kv = session.create_kv_state(kv_descriptor).unwrap();
        let linear = session
            .create_linear_attention_state(linear_descriptor)
            .unwrap();
        let expected = kv_state_allocation_bytes(kv_descriptor).unwrap()
            + linear_state_allocation_bytes(linear_descriptor).unwrap();
        assert_eq!(
            session.memory_snapshot().request_state().current_bytes(),
            expected
        );
        drop(kv);
        drop(linear);
        assert_eq!(session.memory_snapshot().request_state().current_bytes(), 0);
    }

    fn kv_binding(
        session: &ExecutionSession,
        dtype: crate::DType,
        encoding: crate::Encoding,
        shape: &[usize],
        strides: &[usize],
        access: AccessMode,
    ) -> OwnedTensorBinding {
        let view = TensorView::new(dtype, encoding, shape, strides, 0).unwrap();
        let buffer = session.allocate(view.end_offset().max(1)).unwrap();
        session.bind(&buffer, view, access).unwrap()
    }

    fn valid_kv_bindings(
        session: &ExecutionSession,
        token_count: usize,
    ) -> (OwnedTensorBinding, OwnedTensorBinding) {
        let shape = [token_count, KvStateLayout::HEADS, KvStateLayout::HEAD_DIM];
        let strides = [
            KvStateLayout::HEADS * KvStateLayout::HEAD_DIM,
            KvStateLayout::HEAD_DIM,
            1,
        ];
        (
            kv_binding(
                session,
                crate::DType::Bf16,
                crate::Encoding::Unquantized,
                &shape,
                &strides,
                AccessMode::Read,
            ),
            kv_binding(
                session,
                crate::DType::Bf16,
                crate::Encoding::Unquantized,
                &shape,
                &strides,
                AccessMode::Read,
            ),
        )
    }

    fn linear_binding(
        session: &ExecutionSession,
        dtype: DType,
        shape: &[usize],
        access: AccessMode,
    ) -> OwnedTensorBinding {
        let view = TensorView::with_encoding(dtype, Encoding::Unquantized, shape).unwrap();
        let buffer = session.allocate(view.end_offset().max(1)).unwrap();
        session.bind(&buffer, view, access).unwrap()
    }

    fn valid_linear_bindings(
        session: &ExecutionSession,
        token_count: usize,
    ) -> LinearAttentionBindings {
        LinearAttentionBindings::new(
            linear_binding(
                session,
                DType::Bf16,
                &[token_count, LinearAttentionLayout::QKV_WIDTH],
                AccessMode::Read,
            ),
            linear_binding(
                session,
                DType::Bf16,
                &[token_count, LinearAttentionLayout::OUTPUT_WIDTH],
                AccessMode::Read,
            ),
            linear_binding(
                session,
                DType::Bf16,
                &[token_count, LinearAttentionLayout::VALUE_HEADS],
                AccessMode::Read,
            ),
            linear_binding(
                session,
                DType::Bf16,
                &[token_count, LinearAttentionLayout::VALUE_HEADS],
                AccessMode::Read,
            ),
            linear_binding(
                session,
                DType::Bf16,
                &[
                    LinearAttentionLayout::QKV_WIDTH,
                    1,
                    LinearAttentionLayout::CONV_KERNEL_SIZE,
                ],
                AccessMode::Read,
            ),
            linear_binding(
                session,
                DType::F32,
                &[LinearAttentionLayout::VALUE_HEADS],
                AccessMode::Read,
            ),
            linear_binding(
                session,
                DType::Bf16,
                &[LinearAttentionLayout::VALUE_HEADS],
                AccessMode::Read,
            ),
            linear_binding(
                session,
                DType::F32,
                &[LinearAttentionLayout::HEAD_DIM],
                AccessMode::Read,
            ),
            linear_binding(
                session,
                DType::Bf16,
                &[token_count, LinearAttentionLayout::OUTPUT_WIDTH],
                AccessMode::Write,
            ),
        )
    }

    fn qwen_tied_weight_entry() -> crate::WeightLoadEntry {
        crate::WeightLoadEntry {
            tensor_name: crate::QWEN35_EMBEDDING_TENSOR.to_owned(),
            classification: crate::WeightClassification::Required,
            consumer: Some(crate::WeightConsumerKey {
                layer: None,
                role: crate::WeightConsumer::EmbeddingAndTiedOutput,
            }),
            dtype: crate::TensorDType::Bf16,
            shape: vec![
                crate::QWEN35_VOCAB_SIZE as u64,
                crate::QWEN35_HIDDEN_SIZE as u64,
            ],
            source_file: "model-00001-of-00002.safetensors".to_owned(),
            locked_file_size: 1,
            locked_file_sha256: "0".repeat(64),
            source_range: [0, 1],
            destination_start: Some(0),
            chunks: Vec::new(),
        }
    }

    fn qwen_bound_op(
        descriptor: SemanticOpDescriptor,
        inputs: Vec<OwnedTensorBinding>,
        outputs: Vec<OwnedTensorBinding>,
    ) -> Arc<BoundSemanticOp> {
        Arc::new(
            BoundSemanticOp::new(Arc::new(descriptor), inputs, outputs)
                .expect("valid Qwen host binding"),
        )
    }

    fn qwen_final_output_fixture() -> (
        ExecutionSession,
        crate::WeightLoadEntry,
        Arc<BoundSemanticOp>,
        Arc<BoundSemanticOp>,
        Arc<BoundSemanticOp>,
        Arc<BoundSemanticOp>,
    ) {
        let (session, _) = kv_session();
        let m = 17;
        let embedding_tokens = 3;
        let weight_view = TensorView::contiguous(
            DType::Bf16,
            &[crate::QWEN35_VOCAB_SIZE, crate::QWEN35_HIDDEN_SIZE],
        )
        .unwrap();
        let weight_buffer = session.allocate(weight_view.end_offset()).unwrap();
        let token_view = TensorView::contiguous(DType::I32, &[embedding_tokens]).unwrap();
        let token_buffer = session.allocate(token_view.end_offset()).unwrap();
        let embedding_output_view =
            TensorView::contiguous(DType::Bf16, &[embedding_tokens, crate::QWEN35_HIDDEN_SIZE])
                .unwrap();
        let embedding_output_buffer = session
            .allocate(embedding_output_view.end_offset())
            .unwrap();
        let embedding = qwen_bound_op(
            SemanticOpDescriptor::new(
                SemanticOpKind::Embedding,
                vec![weight_view.clone(), token_view.clone()],
                vec![embedding_output_view.clone()],
            )
            .unwrap(),
            vec![
                session
                    .bind(&weight_buffer, weight_view.clone(), AccessMode::Read)
                    .unwrap(),
                session
                    .bind(&token_buffer, token_view, AccessMode::Read)
                    .unwrap(),
            ],
            vec![
                session
                    .bind(
                        &embedding_output_buffer,
                        embedding_output_view,
                        AccessMode::Write,
                    )
                    .unwrap(),
            ],
        );

        let final_activation =
            TensorView::contiguous(DType::Bf16, &[m, crate::QWEN35_HIDDEN_SIZE]).unwrap();
        let final_activation_buffer = session.allocate(final_activation.end_offset()).unwrap();
        let final_scale =
            TensorView::contiguous(DType::Bf16, &[crate::QWEN35_HIDDEN_SIZE]).unwrap();
        let final_scale_buffer = session.allocate(final_scale.end_offset()).unwrap();
        let norm_output = final_activation.clone();
        let norm_output_buffer = session.allocate(norm_output.end_offset()).unwrap();
        let final_rmsnorm = qwen_bound_op(
            SemanticOpDescriptor::new_rms_norm(
                vec![final_activation.clone(), final_scale.clone()],
                vec![norm_output.clone()],
                1.0e-6,
                crate::RmsNormScaleMode::OffsetOne,
            )
            .unwrap(),
            vec![
                session
                    .bind(&final_activation_buffer, final_activation, AccessMode::Read)
                    .unwrap(),
                session
                    .bind(&final_scale_buffer, final_scale, AccessMode::Read)
                    .unwrap(),
            ],
            vec![
                session
                    .bind(&norm_output_buffer, norm_output.clone(), AccessMode::Write)
                    .unwrap(),
            ],
        );

        let logits = TensorView::contiguous(DType::Bf16, &[m, crate::QWEN35_VOCAB_SIZE]).unwrap();
        let logits_buffer = session.allocate(logits.end_offset()).unwrap();
        let tied_projection = qwen_bound_op(
            SemanticOpDescriptor::new(
                SemanticOpKind::Matmul,
                vec![norm_output.clone(), weight_view.clone()],
                vec![logits.clone()],
            )
            .unwrap(),
            vec![
                session
                    .bind(&norm_output_buffer, norm_output, AccessMode::Read)
                    .unwrap(),
                session
                    .bind(&weight_buffer, weight_view, AccessMode::Read)
                    .unwrap(),
            ],
            vec![
                session
                    .bind(&logits_buffer, logits.clone(), AccessMode::Write)
                    .unwrap(),
            ],
        );
        let token_ids = TensorView::contiguous(DType::I32, &[m]).unwrap();
        let token_ids_buffer = session.allocate(token_ids.end_offset()).unwrap();
        let argmax = qwen_bound_op(
            SemanticOpDescriptor::new(
                SemanticOpKind::Argmax,
                vec![logits.clone()],
                vec![token_ids.clone()],
            )
            .unwrap(),
            vec![
                session
                    .bind(&logits_buffer, logits, AccessMode::Read)
                    .unwrap(),
            ],
            vec![
                session
                    .bind(&token_ids_buffer, token_ids, AccessMode::Write)
                    .unwrap(),
            ],
        );
        (
            session,
            qwen_tied_weight_entry(),
            embedding,
            final_rmsnorm,
            tied_projection,
            argmax,
        )
    }

    #[test]
    fn qwen_final_output_contract_accepts_exact_tied_buffer_and_model_descriptors() {
        let (_, entry, embedding, final_rmsnorm, tied_projection, argmax) =
            qwen_final_output_fixture();
        let composition = crate::QwenFinalOutputBindings::new(
            &entry,
            Arc::clone(&embedding),
            Arc::clone(&final_rmsnorm),
            Arc::clone(&tied_projection),
            Arc::clone(&argmax),
        )
        .expect("exact Qwen final output composition");

        assert_eq!(
            composition.tied_weight().buffer().id(),
            tied_projection.inputs()[1].buffer().id()
        );
        assert_eq!(
            composition.tied_weight().view(),
            tied_projection.inputs()[1].view()
        );
        assert_eq!(
            final_rmsnorm.descriptor().inputs()[0].shape(),
            &[17, crate::QWEN35_HIDDEN_SIZE]
        );
        assert_eq!(
            tied_projection.descriptor().outputs()[0].shape(),
            &[17, crate::QWEN35_VOCAB_SIZE]
        );
        assert_eq!(argmax.descriptor().outputs()[0].dtype(), DType::I32);
    }

    #[test]
    fn qwen_final_output_contract_rejects_different_weight_buffer_range_and_access() {
        let (session, entry, embedding, final_rmsnorm, tied_projection, argmax) =
            qwen_final_output_fixture();
        let weight_view = tied_projection.inputs()[1].view().clone();
        let other_weight_buffer = session.allocate(weight_view.end_offset()).unwrap();
        let different_buffer_projection = qwen_bound_op(
            tied_projection.descriptor().as_ref().clone(),
            vec![
                tied_projection.inputs()[0].clone(),
                session
                    .bind(&other_weight_buffer, weight_view.clone(), AccessMode::Read)
                    .unwrap(),
            ],
            tied_projection.outputs().to_vec(),
        );
        assert!(matches!(
            crate::QwenFinalOutputBindings::new(
                &entry,
                Arc::clone(&embedding),
                Arc::clone(&final_rmsnorm),
                different_buffer_projection,
                Arc::clone(&argmax),
            ),
            Err(ExecutionError::DescriptorBindingMismatch { .. })
        ));

        let shifted_weight = TensorView::new(
            DType::Bf16,
            Encoding::Unquantized,
            &[crate::QWEN35_VOCAB_SIZE, crate::QWEN35_HIDDEN_SIZE],
            &[crate::QWEN35_HIDDEN_SIZE, 1],
            2,
        )
        .unwrap();
        let shifted_weight_buffer = session.allocate(shifted_weight.end_offset()).unwrap();
        let shifted_projection = qwen_bound_op(
            SemanticOpDescriptor::new(
                SemanticOpKind::Matmul,
                vec![
                    tied_projection.inputs()[0].view().clone(),
                    shifted_weight.clone(),
                ],
                vec![tied_projection.outputs()[0].view().clone()],
            )
            .unwrap(),
            vec![
                tied_projection.inputs()[0].clone(),
                session
                    .bind(&shifted_weight_buffer, shifted_weight, AccessMode::Read)
                    .unwrap(),
            ],
            tied_projection.outputs().to_vec(),
        );
        assert!(matches!(
            crate::QwenFinalOutputBindings::new(
                &entry,
                Arc::clone(&embedding),
                Arc::clone(&final_rmsnorm),
                shifted_projection,
                Arc::clone(&argmax),
            ),
            Err(ExecutionError::DescriptorBindingMismatch { .. })
        ));

        let read_write_projection = qwen_bound_op(
            tied_projection.descriptor().as_ref().clone(),
            vec![
                tied_projection.inputs()[0].clone(),
                session
                    .bind(
                        embedding.inputs()[0].buffer(),
                        weight_view,
                        AccessMode::ReadWrite,
                    )
                    .unwrap(),
            ],
            tied_projection.outputs().to_vec(),
        );
        assert!(matches!(
            crate::QwenFinalOutputBindings::new(
                &entry,
                embedding,
                final_rmsnorm,
                read_write_projection,
                argmax,
            ),
            Err(ExecutionError::AccessViolation { .. })
        ));
    }

    #[test]
    fn qwen_final_output_contract_rejects_wrong_model_shape_layout_dtype_and_weight_role() {
        let (session, mut entry, embedding, final_rmsnorm, tied_projection, argmax) =
            qwen_final_output_fixture();
        entry.consumer = Some(crate::WeightConsumerKey {
            layer: None,
            role: crate::WeightConsumer::FinalNorm,
        });
        assert!(matches!(
            crate::QwenFinalOutputBindings::new(
                &entry,
                Arc::clone(&embedding),
                Arc::clone(&final_rmsnorm),
                Arc::clone(&tied_projection),
                Arc::clone(&argmax),
            ),
            Err(ExecutionError::InvalidRequest { .. })
        ));

        let wrong_weight = TensorView::contiguous(
            DType::Bf16,
            &[crate::QWEN35_VOCAB_SIZE - 1, crate::QWEN35_HIDDEN_SIZE],
        )
        .unwrap();
        let wrong_logits =
            TensorView::contiguous(DType::Bf16, &[17, crate::QWEN35_VOCAB_SIZE - 1]).unwrap();
        let wrong_weight_buffer = session.allocate(wrong_weight.end_offset()).unwrap();
        let wrong_logits_buffer = session.allocate(wrong_logits.end_offset()).unwrap();
        let wrong_shape_projection = qwen_bound_op(
            SemanticOpDescriptor::new(
                SemanticOpKind::Matmul,
                vec![
                    tied_projection.inputs()[0].view().clone(),
                    wrong_weight.clone(),
                ],
                vec![wrong_logits.clone()],
            )
            .unwrap(),
            vec![
                tied_projection.inputs()[0].clone(),
                session
                    .bind(&wrong_weight_buffer, wrong_weight, AccessMode::Read)
                    .unwrap(),
            ],
            vec![
                session
                    .bind(&wrong_logits_buffer, wrong_logits, AccessMode::Write)
                    .unwrap(),
            ],
        );
        let entry = qwen_tied_weight_entry();
        assert!(matches!(
            crate::QwenFinalOutputBindings::new(
                &entry,
                embedding,
                final_rmsnorm,
                wrong_shape_projection,
                argmax,
            ),
            Err(ExecutionError::DescriptorBindingMismatch { .. })
        ));

        assert!(SemanticOpDescriptor::new(
            SemanticOpKind::Matmul,
            vec![
                TensorView::contiguous(DType::Bf16, &[17, crate::QWEN35_HIDDEN_SIZE]).unwrap(),
                TensorView::contiguous(
                    DType::F16,
                    &[crate::QWEN35_VOCAB_SIZE, crate::QWEN35_HIDDEN_SIZE],
                )
                .unwrap(),
            ],
            vec![
                TensorView::contiguous(DType::Bf16, &[17, crate::QWEN35_VOCAB_SIZE]).unwrap(),
            ],
        )
        .is_err());
        assert!(
            SemanticOpDescriptor::new(
                SemanticOpKind::Argmax,
                vec![
                    TensorView::new(
                        DType::Bf16,
                        Encoding::Unquantized,
                        &[17, crate::QWEN35_VOCAB_SIZE],
                        &[crate::QWEN35_VOCAB_SIZE + 1, 1],
                        0,
                    )
                    .unwrap(),
                ],
                vec![TensorView::contiguous(DType::I32, &[17]).unwrap()],
            )
            .is_err()
        );
    }

    #[test]
    fn linear_attention_covers_boundaries_and_transactional_publication() {
        for token_count in [1_usize, 3, 17, 255, 256, 257] {
            let (session, adapter) = kv_session();
            let queue = session.create_queue().unwrap();
            let state = session
                .create_linear_attention_state(
                    LinearAttentionStateDescriptor::new(5, token_count as u64).unwrap(),
                )
                .unwrap();
            let descriptor =
                LinearAttentionDescriptor::new(0, token_count as u64, token_count as u64).unwrap();
            let mut submission = session
                .linear_attention(
                    &state,
                    &queue,
                    valid_linear_bindings(&session, token_count),
                    descriptor,
                )
                .unwrap();
            assert_eq!(state.snapshot(&session).unwrap().length(), 0);
            assert_eq!(submission.request().descriptor(), descriptor);
            assert_eq!(submission.dispatch().dispatch_count, 2);
            assert_eq!(
                submission.wait(Duration::ZERO).unwrap(),
                ExecutionState::Success
            );
            assert_eq!(
                submission.wait(Duration::ZERO).unwrap(),
                ExecutionState::Success
            );
            assert!(
                adapter
                    .store
                    .linear_drop_saw_core_admission
                    .load(Ordering::Acquire)
            );
            assert_eq!(adapter.store.linear_drops.load(Ordering::Relaxed), 1);
            assert_eq!(
                state.snapshot(&session).unwrap().length(),
                token_count as u64
            );
            assert_eq!(adapter.store.linear_calls.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn linear_attention_rejects_concurrency_and_drop_does_not_publish() {
        let (session, adapter) = kv_session();
        let queue = session.create_queue().unwrap();
        let state = session
            .create_linear_attention_state(LinearAttentionStateDescriptor::new(0, 3).unwrap())
            .unwrap();
        let descriptor = LinearAttentionDescriptor::new(0, 1, 1).unwrap();
        let first = session
            .linear_attention(
                &state,
                &queue,
                valid_linear_bindings(&session, 1),
                descriptor,
            )
            .unwrap();
        assert!(matches!(
            session.linear_attention(
                &state,
                &queue,
                valid_linear_bindings(&session, 1),
                descriptor,
            ),
            Err(ExecutionError::Busy)
        ));
        drop(first);
        assert!(
            adapter
                .store
                .linear_drop_saw_core_admission
                .load(Ordering::Acquire)
        );
        assert_eq!(state.snapshot(&session).unwrap().length(), 0);
        assert_eq!(adapter.store.linear_drops.load(Ordering::Relaxed), 1);

        let mut second = session
            .linear_attention(
                &state,
                &queue,
                valid_linear_bindings(&session, 1),
                descriptor,
            )
            .unwrap();
        assert_eq!(second.query().unwrap(), ExecutionState::Success);
        assert_eq!(second.query().unwrap(), ExecutionState::Success);
        assert!(
            adapter
                .store
                .linear_drop_saw_core_admission
                .load(Ordering::Acquire)
        );
        assert_eq!(state.snapshot(&session).unwrap().length(), 1);
        let next_descriptor = LinearAttentionDescriptor::new(1, 1, 2).unwrap();
        let next = session
            .linear_attention(
                &state,
                &queue,
                valid_linear_bindings(&session, 1),
                next_descriptor,
            )
            .expect("terminal query cleaned the backend owner before reopening admission");
        drop(next);
    }

    #[test]
    fn linear_attention_rejects_wrong_contract_alias_and_capacity_before_dispatch() {
        let (session, adapter) = kv_session();
        let queue = session.create_queue().unwrap();
        let state = session
            .create_linear_attention_state(LinearAttentionStateDescriptor::new(0, 3).unwrap())
            .unwrap();
        let descriptor = LinearAttentionDescriptor::new(0, 1, 1).unwrap();

        let mut wrong_dtype = valid_linear_bindings(&session, 1);
        wrong_dtype.qkv = linear_binding(
            &session,
            DType::F16,
            &[1, LinearAttentionLayout::QKV_WIDTH],
            AccessMode::Read,
        );
        assert!(matches!(
            session.linear_attention(&state, &queue, wrong_dtype, descriptor),
            Err(ExecutionError::InvalidRequest { reason }) if reason.contains("Bf16")
        ));

        let mut wrong_access = valid_linear_bindings(&session, 1);
        wrong_access.output = linear_binding(
            &session,
            DType::Bf16,
            &[1, LinearAttentionLayout::OUTPUT_WIDTH],
            AccessMode::Read,
        );
        assert!(matches!(
            session.linear_attention(&state, &queue, wrong_access, descriptor),
            Err(ExecutionError::AccessViolation { .. })
        ));

        let mut alias = valid_linear_bindings(&session, 1);
        let z_view = TensorView::with_encoding(
            DType::Bf16,
            Encoding::Unquantized,
            &[1, LinearAttentionLayout::OUTPUT_WIDTH],
        )
        .unwrap();
        alias.z = session
            .bind(alias.qkv.buffer(), z_view, AccessMode::Read)
            .unwrap();
        assert!(matches!(
            session.linear_attention(&state, &queue, alias, descriptor),
            Err(ExecutionError::AliasOverlap { .. })
        ));

        let capacity_state = session
            .create_linear_attention_state(LinearAttentionStateDescriptor::new(1, 1).unwrap())
            .unwrap();
        assert!(matches!(
            session.linear_attention(
                &capacity_state,
                &queue,
                valid_linear_bindings(&session, 3),
                LinearAttentionDescriptor::new(0, 3, 3).unwrap(),
            ),
            Err(ExecutionError::InvalidRange { .. })
        ));
        assert_eq!(adapter.store.linear_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn kv_append_covers_non_aligned_capacity_and_exact_end_boundaries() {
        for token_count in [1_usize, 3, 17, 255, 256, 257] {
            let (session, adapter) = kv_session();
            let queue = session.create_queue().unwrap();
            let descriptor = crate::KvStateDescriptor::new(3, token_count as u64).unwrap();
            let state = session.create_kv_state(descriptor).unwrap();
            let (key, value) = valid_kv_bindings(&session, token_count);
            let mut append = session
                .append_kv_state(&state, &queue, key, value, 0, 0)
                .unwrap();
            assert_eq!(append.request().token_count(), token_count as u64);
            assert_eq!(append.request().end_position(), token_count as u64);
            assert_eq!(append.dispatch().backend, 1);
            assert_eq!(append.dispatch().dispatch_count, 1);
            assert!(!append.dispatch().fallback_allowed);
            assert!(!append.dispatch().fallback_used);
            assert_eq!(append.query().unwrap(), ExecutionState::Success);
            assert_eq!(
                session.kv_state_snapshot(&state).unwrap().length(),
                token_count as u64
            );
            assert_eq!(adapter.store.append_calls.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn kv_append_covers_start_and_end_position_boundaries() {
        let (session, _adapter) = kv_session();
        let queue = session.create_queue().unwrap();
        let state = session
            .create_kv_state(crate::KvStateDescriptor::new(0, 257).unwrap())
            .unwrap();
        for (start, token_count, end) in [
            (0_u64, 1_usize, 1_u64),
            (1, 2, 3),
            (3, 14, 17),
            (17, 238, 255),
            (255, 1, 256),
            (256, 1, 257),
        ] {
            let (key, value) = valid_kv_bindings(&session, token_count);
            let mut append = session
                .append_kv_state(&state, &queue, key, value, start, start)
                .unwrap();
            assert_eq!(append.request().end_position(), end);
            assert_eq!(
                append.wait(Duration::ZERO).unwrap(),
                ExecutionState::Success
            );
        }
        assert_eq!(state.snapshot(&session).unwrap().length(), 257);
    }

    #[test]
    fn kv_append_rejects_capacity_overflow_and_position_overflow_before_backend_append() {
        let (session, adapter) = kv_session();
        let queue = session.create_queue().unwrap();
        let state = session
            .create_kv_state(crate::KvStateDescriptor::new(0, 1).unwrap())
            .unwrap();
        let (key, value) = valid_kv_bindings(&session, 1);
        assert!(matches!(
            session.append_kv_state(&state, &queue, key, value, 1, 1),
            Err(ExecutionError::InvalidRange { .. })
        ));
        assert_eq!(adapter.store.append_calls.load(Ordering::Relaxed), 0);

        let state = session
            .create_kv_state(crate::KvStateDescriptor::new(1, u64::MAX).unwrap())
            .unwrap();
        let (key, value) = valid_kv_bindings(&session, 1);
        assert!(matches!(
            session.append_kv_state(&state, &queue, key, value, u64::MAX, u64::MAX),
            Err(ExecutionError::InvalidRange { reason })
                if reason.contains("overflow")
        ));
        assert_eq!(adapter.store.append_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn kv_append_uses_snapshot_for_stale_admission_and_rejects_concurrent_append() {
        let (session, adapter) = kv_session();
        let queue = session.create_queue().unwrap();
        let state = session
            .create_kv_state(crate::KvStateDescriptor::new(0, 3).unwrap())
            .unwrap();
        let (key, value) = valid_kv_bindings(&session, 1);
        let mut first = session
            .append_kv_state(&state, &queue, key, value, 0, 0)
            .unwrap();

        let (key, value) = valid_kv_bindings(&session, 1);
        assert!(matches!(
            session.append_kv_state(&state, &queue, key, value, 0, 0),
            Err(ExecutionError::Busy)
        ));
        assert_eq!(adapter.store.append_calls.load(Ordering::Relaxed), 1);

        assert_eq!(first.wait(Duration::ZERO).unwrap(), ExecutionState::Success);
        assert_eq!(session.kv_state_snapshot(&state).unwrap().length(), 1);
        let (key, value) = valid_kv_bindings(&session, 1);
        assert!(matches!(
            session.append_kv_state(&state, &queue, key, value, 0, 0),
            Err(ExecutionError::StaleKvLength {
                expected: 0,
                actual: 1
            })
        ));
        assert_eq!(adapter.store.append_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn kv_append_rejects_wrong_session_state_identity_and_contract_without_append() {
        let (session, adapter) = kv_session();
        let queue = session.create_queue().unwrap();
        let state = session
            .create_kv_state(crate::KvStateDescriptor::new(2, 4).unwrap())
            .unwrap();

        adapter
            .store
            .wrong_snapshot_identity
            .store(true, Ordering::Relaxed);
        assert!(matches!(
            session.kv_state_snapshot(&state),
            Err(ExecutionError::WrongKvState { .. })
        ));
        adapter
            .store
            .wrong_snapshot_identity
            .store(false, Ordering::Relaxed);

        let (foreign_session, _) = kv_session();
        let foreign_state = foreign_session
            .create_kv_state(crate::KvStateDescriptor::new(2, 4).unwrap())
            .unwrap();
        assert!(matches!(
            session.kv_state_snapshot(&foreign_state),
            Err(ExecutionError::WrongSession { .. })
        ));

        let invalid = [
            kv_binding(
                &session,
                crate::DType::Bf16,
                crate::Encoding::Unquantized,
                &[1, 16, 256],
                &[4096, 256, 1],
                AccessMode::Read,
            ),
            kv_binding(
                &session,
                crate::DType::F16,
                crate::Encoding::Unquantized,
                &[1, 4, 256],
                &[1024, 256, 1],
                AccessMode::Read,
            ),
            kv_binding(
                &session,
                crate::DType::U8,
                crate::Encoding::Nvfp4 {
                    block_size: 32,
                    scale_dtype: crate::DType::Bf16,
                },
                &[1, 4, 256],
                &[1024, 256, 1],
                AccessMode::Read,
            ),
            kv_binding(
                &session,
                crate::DType::Bf16,
                crate::Encoding::Unquantized,
                &[1, 4, 256],
                &[1024, 257, 1],
                AccessMode::Read,
            ),
            kv_binding(
                &session,
                crate::DType::Bf16,
                crate::Encoding::Unquantized,
                &[0, 4, 256],
                &[1024, 256, 1],
                AccessMode::Read,
            ),
        ];
        let (mismatch_key, _) = valid_kv_bindings(&session, 1);
        let (_, mismatch_value) = valid_kv_bindings(&session, 3);
        assert!(
            session
                .append_kv_state(&state, &queue, mismatch_key, mismatch_value, 0, 0)
                .is_err()
        );
        for invalid_key in invalid {
            let (_, value) = valid_kv_bindings(&session, 1);
            assert!(
                session
                    .append_kv_state(&state, &queue, invalid_key, value, 0, 0)
                    .is_err()
            );
        }
        let (key, value) = valid_kv_bindings(&session, 1);
        let wrong_access = kv_binding(
            &session,
            crate::DType::Bf16,
            crate::Encoding::Unquantized,
            &[1, 4, 256],
            &[1024, 256, 1],
            AccessMode::Write,
        );
        assert!(matches!(
            session.append_kv_state(&state, &queue, wrong_access, value, 0, 0),
            Err(ExecutionError::AccessViolation { .. })
        ));
        let (foreign_key, foreign_value) = valid_kv_bindings(&foreign_session, 1);
        assert!(matches!(
            session.append_kv_state(&state, &queue, foreign_key, foreign_value, 0, 0),
            Err(ExecutionError::WrongSession { .. })
        ));
        drop(key);
        assert_eq!(adapter.store.append_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn kv_append_submission_owns_inputs_and_state_until_drop() {
        let (session, adapter) = kv_session();
        let queue = session.create_queue().unwrap();
        let state = session
            .create_kv_state(crate::KvStateDescriptor::new(0, 1).unwrap())
            .unwrap();
        let (key, value) = valid_kv_bindings(&session, 1);
        let submission = session
            .append_kv_state(&state, &queue, key, value, 0, 0)
            .unwrap();
        assert_eq!(adapter.buffer_drops.load(Ordering::Relaxed), 0);
        assert_eq!(adapter.store.append_drops.load(Ordering::Relaxed), 0);
        drop(submission);
        assert_eq!(adapter.buffer_drops.load(Ordering::Relaxed), 2);
        assert_eq!(adapter.store.append_drops.load(Ordering::Relaxed), 1);
        assert!(
            adapter
                .store
                .append_drop_saw_core_admission
                .load(Ordering::Acquire)
        );
        assert_eq!(adapter.state_resource_drops.load(Ordering::Relaxed), 0);
        drop(state);
        assert_eq!(adapter.state_resource_drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn causal_attention_has_exact_shape_and_exclusive_state_admission() {
        let (session, adapter) = kv_session();
        let queue = session.create_queue().unwrap();
        let state = session
            .create_kv_state(crate::KvStateDescriptor::new(0, 3).unwrap())
            .unwrap();
        let (key, value) = valid_kv_bindings(&session, 1);
        let mut append = session
            .append_kv_state(&state, &queue, key, value, 0, 0)
            .unwrap();
        assert_eq!(
            append.wait(Duration::ZERO).unwrap(),
            ExecutionState::Success
        );

        let query = kv_binding(
            &session,
            crate::DType::Bf16,
            crate::Encoding::Unquantized,
            &[1, 16, 256],
            &[4096, 256, 1],
            AccessMode::Read,
        );
        let output = kv_binding(
            &session,
            crate::DType::Bf16,
            crate::Encoding::Unquantized,
            &[1, 16, 256],
            &[4096, 256, 1],
            AccessMode::Write,
        );
        let descriptor = CausalAttentionDescriptor::new(0, 1, 1).unwrap();
        let attention = session
            .causal_attention(&state, &queue, query, output, descriptor)
            .unwrap();
        assert_eq!(adapter.causal_attention_calls.load(Ordering::Relaxed), 1);
        assert_eq!(attention.dispatch().grid_size_x, 16);

        let (blocked_key, blocked_value) = valid_kv_bindings(&session, 1);
        assert!(matches!(
            session.append_kv_state(&state, &queue, blocked_key, blocked_value, 1, 1),
            Err(ExecutionError::Busy)
        ));
        let second_query = kv_binding(
            &session,
            crate::DType::Bf16,
            crate::Encoding::Unquantized,
            &[1, 16, 256],
            &[4096, 256, 1],
            AccessMode::Read,
        );
        let second_output = kv_binding(
            &session,
            crate::DType::Bf16,
            crate::Encoding::Unquantized,
            &[1, 16, 256],
            &[4096, 256, 1],
            AccessMode::Write,
        );
        assert!(matches!(
            session.causal_attention(&state, &queue, second_query, second_output, descriptor,),
            Err(ExecutionError::Busy)
        ));
        assert_eq!(adapter.causal_attention_drops.load(Ordering::Relaxed), 0);
        drop(attention);
        assert_eq!(adapter.causal_attention_drops.load(Ordering::Relaxed), 1);

        let (key, value) = valid_kv_bindings(&session, 1);
        let mut append = session
            .append_kv_state(&state, &queue, key, value, 1, 1)
            .unwrap();
        assert_eq!(
            append.wait(Duration::ZERO).unwrap(),
            ExecutionState::Success
        );
        assert_eq!(session.kv_state_snapshot(&state).unwrap().length(), 2);
    }

    #[test]
    fn causal_attention_rejects_invalid_descriptor_and_bindings() {
        assert!(matches!(
            CausalAttentionDescriptor::new(0, 0, 0),
            Err(crate::KvStateError::ZeroQueryCount)
        ));
        assert!(matches!(
            CausalAttentionDescriptor::new(u64::MAX, 1, 0),
            Err(crate::KvStateError::LengthOverflow)
        ));
        assert!(matches!(
            CausalAttentionDescriptor::new(2, 3, 4),
            Err(crate::KvStateError::LengthMismatch { .. })
        ));

        let (session, _adapter) = kv_session();
        let queue = session.create_queue().unwrap();
        let state = session
            .create_kv_state(crate::KvStateDescriptor::new(0, 3).unwrap())
            .unwrap();
        let query = kv_binding(
            &session,
            crate::DType::Bf16,
            crate::Encoding::Unquantized,
            &[1, 16, 256],
            &[4096, 256, 1],
            AccessMode::Read,
        );
        let wrong_output = kv_binding(
            &session,
            crate::DType::F16,
            crate::Encoding::Unquantized,
            &[1, 16, 256],
            &[4096, 256, 1],
            AccessMode::Write,
        );
        assert!(matches!(
            session.causal_attention(
                &state,
                &queue,
                query,
                wrong_output,
                CausalAttentionDescriptor::new(0, 1, 1).unwrap(),
            ),
            Err(ExecutionError::InvalidRequest { reason })
                if reason.contains("BF16")
        ));

        let query = kv_binding(
            &session,
            crate::DType::Bf16,
            crate::Encoding::Unquantized,
            &[1, 16, 256],
            &[4096, 256, 1],
            AccessMode::Read,
        );
        let wrong_stride = kv_binding(
            &session,
            crate::DType::Bf16,
            crate::Encoding::Unquantized,
            &[1, 16, 256],
            &[4096, 257, 1],
            AccessMode::Write,
        );
        assert!(matches!(
            session.causal_attention(
                &state,
                &queue,
                query,
                wrong_stride,
                CausalAttentionDescriptor::new(0, 1, 1).unwrap(),
            ),
            Err(ExecutionError::InvalidRequest { reason })
                if reason.contains("strides")
        ));

        let capacity_state = session
            .create_kv_state(crate::KvStateDescriptor::new(0, 1).unwrap())
            .unwrap();
        let query = kv_binding(
            &session,
            crate::DType::Bf16,
            crate::Encoding::Unquantized,
            &[1, 16, 256],
            &[4096, 256, 1],
            AccessMode::Read,
        );
        let output = kv_binding(
            &session,
            crate::DType::Bf16,
            crate::Encoding::Unquantized,
            &[1, 16, 256],
            &[4096, 256, 1],
            AccessMode::Write,
        );
        assert!(matches!(
            session.causal_attention(
                &capacity_state,
                &queue,
                query,
                output,
                CausalAttentionDescriptor::new(1, 1, 2).unwrap(),
            ),
            Err(ExecutionError::InvalidRange { .. })
        ));
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

    fn attention_preprocess_views_with_offsets(
        offsets: [u64; 8],
    ) -> (Vec<TensorView>, Vec<TensorView>) {
        let make = |dtype, shape: &[usize], strides: &[usize], offset| {
            TensorView::new(dtype, crate::Encoding::Unquantized, shape, strides, offset)
                .expect("valid C3a1 view")
        };
        (
            vec![
                make(
                    crate::DType::Bf16,
                    &[1, 16, 512],
                    &[8192, 512, 1],
                    offsets[0],
                ),
                make(
                    crate::DType::Bf16,
                    &[1, 4, 256],
                    &[1024, 256, 1],
                    offsets[1],
                ),
                make(crate::DType::Bf16, &[16, 256], &[256, 1], offsets[2]),
                make(crate::DType::Bf16, &[4, 256], &[256, 1], offsets[3]),
                make(crate::DType::I32, &[1], &[1], offsets[4]),
            ],
            vec![
                make(
                    crate::DType::Bf16,
                    &[1, 16, 256],
                    &[4096, 256, 1],
                    offsets[5],
                ),
                make(
                    crate::DType::Bf16,
                    &[1, 16, 256],
                    &[4096, 256, 1],
                    offsets[6],
                ),
                make(
                    crate::DType::Bf16,
                    &[1, 4, 256],
                    &[1024, 256, 1],
                    offsets[7],
                ),
            ],
        )
    }

    fn bind_attention_preprocess(offsets: [u64; 8]) -> Result<BoundSemanticOp, ExecutionError> {
        let session = session("test");
        let buffer = session.allocate(100_000).expect("test buffer");
        let (inputs, outputs) = attention_preprocess_views_with_offsets(offsets);
        let descriptor = Arc::new(
            SemanticOpDescriptor::new_attention_preprocess(
                inputs.clone(),
                outputs.clone(),
                crate::AttentionPreprocessContract::new_qwen3_5(
                    crate::AttentionPreprocessPositionMode::Prefill,
                    0,
                    1,
                )
                .expect("C3a1 contract"),
            )
            .expect("C3a1 descriptor"),
        );
        let input_bindings = inputs
            .into_iter()
            .map(|view| session.bind(&buffer, view, crate::AccessMode::Read))
            .collect::<Result<Vec<_>, _>>()?;
        let output_bindings = outputs
            .into_iter()
            .map(|view| session.bind(&buffer, view, crate::AccessMode::Write))
            .collect::<Result<Vec<_>, _>>()?;
        BoundSemanticOp::new(descriptor, input_bindings, output_bindings)
    }

    #[test]
    fn attention_preprocess_binding_rejects_every_overlap_but_accepts_touching_ranges() {
        let lengths = [16_384_u64, 2_048, 8_192, 2_048, 4, 8_192, 8_192, 2_048];
        let mut touching = [0_u64; 8];
        for index in 1..touching.len() {
            touching[index] = touching[index - 1] + lengths[index - 1];
        }
        assert!(bind_attention_preprocess(touching).is_ok());

        for left in 0..8 {
            for right in left + 1..8 {
                let mut overlapping = touching;
                overlapping[right] = overlapping[left];
                assert!(
                    bind_attention_preprocess(overlapping).is_err(),
                    "overlap pair {left}/{right} must be rejected"
                );
            }
        }
    }

    #[test]
    fn elementwise_overlap_rejects_aliases_but_allows_touching_ranges() {
        let session = session("test");
        let buffer = session.allocate(64).unwrap();
        let view = |offset| {
            TensorView::new(
                crate::DType::Bf16,
                crate::Encoding::Unquantized,
                &[3],
                &[1],
                offset,
            )
            .expect("valid elementwise view")
        };

        let copy_input = view(0);
        let copy_output = view(6);
        let copy = Arc::new(
            SemanticOpDescriptor::new(
                SemanticOpKind::Copy,
                vec![copy_input.clone()],
                vec![copy_output.clone()],
            )
            .unwrap(),
        );
        assert!(
            BoundSemanticOp::new(
                copy,
                vec![session.bind(&buffer, copy_input, AccessMode::Read).unwrap()],
                vec![
                    session
                        .bind(&buffer, copy_output, AccessMode::Write)
                        .unwrap()
                ],
            )
            .is_ok(),
            "half-open touching copy ranges must stay disjoint"
        );

        let add_views = [view(0), view(4), view(12)];
        let add = Arc::new(
            SemanticOpDescriptor::new(
                SemanticOpKind::Add,
                vec![add_views[0].clone(), add_views[1].clone()],
                vec![add_views[2].clone()],
            )
            .unwrap(),
        );
        assert!(matches!(
            BoundSemanticOp::new(
                add,
                vec![
                    session
                        .bind(&buffer, add_views[0].clone(), AccessMode::Read)
                        .unwrap(),
                    session
                        .bind(&buffer, add_views[1].clone(), AccessMode::Read)
                        .unwrap(),
                ],
                vec![
                    session
                        .bind(&buffer, add_views[2].clone(), AccessMode::Write)
                        .unwrap()
                ],
            ),
            Err(ExecutionError::AliasOverlap {
                left: "add input 0",
                right: "add input 1"
            })
        ));

        let gate = TensorView::new(
            crate::DType::Bf16,
            crate::Encoding::Unquantized,
            &[1, 16, 256],
            &[4096, 256, 1],
            0,
        )
        .unwrap();
        let attention_value = TensorView::new(
            crate::DType::Bf16,
            crate::Encoding::Unquantized,
            &[1, 16, 256],
            &[4096, 256, 1],
            8192,
        )
        .unwrap();
        let sigmoid_output = gate.clone();
        let sigmoid_mul = Arc::new(
            SemanticOpDescriptor::new(
                SemanticOpKind::SigmoidMul,
                vec![gate.clone(), attention_value.clone()],
                vec![sigmoid_output.clone()],
            )
            .unwrap(),
        );
        let sigmoid_buffer = session.allocate(16_384).unwrap();
        assert!(matches!(
            BoundSemanticOp::new(
                sigmoid_mul,
                vec![
                    session
                        .bind(&sigmoid_buffer, gate, AccessMode::Read)
                        .unwrap(),
                    session
                        .bind(&sigmoid_buffer, attention_value, AccessMode::Read)
                        .unwrap(),
                ],
                vec![
                    session
                        .bind(&sigmoid_buffer, sigmoid_output, AccessMode::Write)
                        .unwrap(),
                ],
            ),
            Err(ExecutionError::AliasOverlap {
                left: "sigmoid_mul gate",
                right: "sigmoid_mul output"
            })
        ));

        let copy_input = view(0);
        let copy_output = view(4);
        let copy = Arc::new(
            SemanticOpDescriptor::new(
                SemanticOpKind::Copy,
                vec![copy_input.clone()],
                vec![copy_output.clone()],
            )
            .unwrap(),
        );
        assert!(matches!(
            BoundSemanticOp::new(
                copy,
                vec![session.bind(&buffer, copy_input, AccessMode::Read).unwrap()],
                vec![
                    session
                        .bind(&buffer, copy_output, AccessMode::Write)
                        .unwrap()
                ],
            ),
            Err(ExecutionError::AliasOverlap {
                left: "copy input",
                right: "copy output"
            })
        ));
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
