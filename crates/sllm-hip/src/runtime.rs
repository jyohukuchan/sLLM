//! Safe ownership wrappers for the additive public HIP runtime ABI.
//!
//! The wrappers use `Arc` and are `Send + Sync`.  Native calls are serialized
//! by the public runtime registry/accounting locks, while Rust only exposes
//! mutable completion state through `&mut self`.  An in-flight completion keeps the context, queue, and
//! buffer `Arc`s alive until the caller observes a terminal result.

use std::cell::RefCell;
use std::ffi::{c_char, c_void};
use std::fmt;
use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(test)]
use std::sync::{Condvar, Mutex};
#[cfg(test)]
use std::thread::ThreadId;
use std::time::Duration;

use sllm_hip_sys as sys;

const ERROR_CAPACITY: usize = 256;
const MAX_FINITE_TIMEOUT_MS: u32 = u32::MAX - 1;
const DROP_WAIT_TIMEOUT_MS: u32 = 0;

#[cfg(test)]
static FORCED_RMSNORM_PLAN_RELEASE: Mutex<Option<(RuntimeStatus, bool)>> = Mutex::new(None);
#[cfg(test)]
static FORCED_MATMUL_PLAN_RELEASE: Mutex<Option<(RuntimeStatus, bool)>> = Mutex::new(None);
#[cfg(test)]
static FORCED_ATTENTION_PREPROCESS_PLAN_RELEASE: Mutex<Option<(RuntimeStatus, bool)>> =
    Mutex::new(None);
#[cfg(test)]
pub(crate) static CLEANUP_TEST_SERIAL: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeStatus {
    Ok,
    InvalidArgument,
    BufferTooSmall,
    Unsupported,
    HipUnavailable,
    InvalidAbiVersion,
    ReservedNonzero,
    InternalError,
    Pending,
    Timeout,
    InvalidHandle,
    DeviceMismatch,
    HipRuntimeError,
    Busy,
    NotReady,
    InvalidRmsNormDescriptor,
    InvalidTensorBinding,
    ZeroExtent,
    ShapeMismatch,
    StrideMismatch,
    MetadataOverflow,
    BufferOutOfBounds,
    MisalignedOffset,
    UnsupportedDType,
    UnsupportedEncoding,
    InvalidEpsilon,
    UnsupportedScaleMode,
    AliasOverlap,
    ContextOrDeviceMismatch,
    InvalidElementwiseDescriptor,
    InvalidEmbeddingDescriptor,
    TokenIdOutOfRange,
    InvalidMatmulDescriptor,
    InvalidAttentionPreprocessDescriptor,
    PositionPayloadMismatch,
    InvalidKvStateDescriptor,
    InvalidKvAppendDescriptor,
    KvLengthMismatch,
    KvCapacityExceeded,
    InvalidCausalAttentionDescriptor,
    CausalAttentionLengthMismatch,
    CausalAttentionStateBusy,
    Unknown(u32),
}

impl RuntimeStatus {
    pub const fn from_raw(raw: u32) -> Self {
        match raw {
            sys::SLLM_STATUS_OK => Self::Ok,
            sys::SLLM_STATUS_INVALID_ARGUMENT => Self::InvalidArgument,
            sys::SLLM_STATUS_BUFFER_TOO_SMALL => Self::BufferTooSmall,
            sys::SLLM_STATUS_UNSUPPORTED => Self::Unsupported,
            sys::SLLM_STATUS_HIP_UNAVAILABLE => Self::HipUnavailable,
            sys::SLLM_STATUS_INVALID_ABI_VERSION => Self::InvalidAbiVersion,
            sys::SLLM_STATUS_RESERVED_NONZERO => Self::ReservedNonzero,
            sys::SLLM_STATUS_INTERNAL_ERROR => Self::InternalError,
            sys::SLLM_STATUS_PUBLIC_PENDING => Self::Pending,
            sys::SLLM_STATUS_PUBLIC_TIMEOUT => Self::Timeout,
            sys::SLLM_STATUS_PUBLIC_INVALID_HANDLE => Self::InvalidHandle,
            sys::SLLM_STATUS_PUBLIC_DEVICE_MISMATCH => Self::DeviceMismatch,
            sys::SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR => Self::HipRuntimeError,
            sys::SLLM_STATUS_PUBLIC_BUSY => Self::Busy,
            sys::SLLM_STATUS_PUBLIC_NOT_READY => Self::NotReady,
            sys::SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR => Self::InvalidRmsNormDescriptor,
            sys::SLLM_STATUS_INVALID_TENSOR_BINDING => Self::InvalidTensorBinding,
            sys::SLLM_STATUS_ZERO_EXTENT => Self::ZeroExtent,
            sys::SLLM_STATUS_SHAPE_MISMATCH => Self::ShapeMismatch,
            sys::SLLM_STATUS_STRIDE_MISMATCH => Self::StrideMismatch,
            sys::SLLM_STATUS_METADATA_OVERFLOW => Self::MetadataOverflow,
            sys::SLLM_STATUS_BUFFER_OUT_OF_BOUNDS => Self::BufferOutOfBounds,
            sys::SLLM_STATUS_MISALIGNED_OFFSET => Self::MisalignedOffset,
            sys::SLLM_STATUS_UNSUPPORTED_DTYPE => Self::UnsupportedDType,
            sys::SLLM_STATUS_UNSUPPORTED_ENCODING => Self::UnsupportedEncoding,
            sys::SLLM_STATUS_INVALID_EPSILON => Self::InvalidEpsilon,
            sys::SLLM_STATUS_UNSUPPORTED_SCALE_MODE => Self::UnsupportedScaleMode,
            sys::SLLM_STATUS_ALIAS_OVERLAP => Self::AliasOverlap,
            sys::SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH => Self::ContextOrDeviceMismatch,
            sys::SLLM_STATUS_INVALID_ELEMENTWISE_DESCRIPTOR => Self::InvalidElementwiseDescriptor,
            sys::SLLM_STATUS_INVALID_EMBEDDING_DESCRIPTOR => Self::InvalidEmbeddingDescriptor,
            sys::SLLM_STATUS_TOKEN_ID_OUT_OF_RANGE => Self::TokenIdOutOfRange,
            sys::SLLM_STATUS_INVALID_MATMUL_DESCRIPTOR => Self::InvalidMatmulDescriptor,
            sys::SLLM_STATUS_INVALID_ATTENTION_PREPROCESS_DESCRIPTOR => {
                Self::InvalidAttentionPreprocessDescriptor
            }
            sys::SLLM_STATUS_POSITION_PAYLOAD_MISMATCH => Self::PositionPayloadMismatch,
            sys::SLLM_STATUS_INVALID_KV_STATE_DESCRIPTOR => Self::InvalidKvStateDescriptor,
            sys::SLLM_STATUS_INVALID_KV_APPEND_DESCRIPTOR => Self::InvalidKvAppendDescriptor,
            sys::SLLM_STATUS_KV_LENGTH_MISMATCH => Self::KvLengthMismatch,
            sys::SLLM_STATUS_KV_CAPACITY_EXCEEDED => Self::KvCapacityExceeded,
            sys::SLLM_STATUS_INVALID_CAUSAL_ATTENTION_DESCRIPTOR => {
                Self::InvalidCausalAttentionDescriptor
            }
            sys::SLLM_STATUS_CAUSAL_ATTENTION_LENGTH_MISMATCH => {
                Self::CausalAttentionLengthMismatch
            }
            sys::SLLM_STATUS_CAUSAL_ATTENTION_STATE_BUSY => Self::CausalAttentionStateBusy,
            other => Self::Unknown(other),
        }
    }

    pub const fn raw(self) -> u32 {
        match self {
            Self::Ok => sys::SLLM_STATUS_OK,
            Self::InvalidArgument => sys::SLLM_STATUS_INVALID_ARGUMENT,
            Self::BufferTooSmall => sys::SLLM_STATUS_BUFFER_TOO_SMALL,
            Self::Unsupported => sys::SLLM_STATUS_UNSUPPORTED,
            Self::HipUnavailable => sys::SLLM_STATUS_HIP_UNAVAILABLE,
            Self::InvalidAbiVersion => sys::SLLM_STATUS_INVALID_ABI_VERSION,
            Self::ReservedNonzero => sys::SLLM_STATUS_RESERVED_NONZERO,
            Self::InternalError => sys::SLLM_STATUS_INTERNAL_ERROR,
            Self::Pending => sys::SLLM_STATUS_PUBLIC_PENDING,
            Self::Timeout => sys::SLLM_STATUS_PUBLIC_TIMEOUT,
            Self::InvalidHandle => sys::SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            Self::DeviceMismatch => sys::SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            Self::HipRuntimeError => sys::SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
            Self::Busy => sys::SLLM_STATUS_PUBLIC_BUSY,
            Self::NotReady => sys::SLLM_STATUS_PUBLIC_NOT_READY,
            Self::InvalidRmsNormDescriptor => sys::SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR,
            Self::InvalidTensorBinding => sys::SLLM_STATUS_INVALID_TENSOR_BINDING,
            Self::ZeroExtent => sys::SLLM_STATUS_ZERO_EXTENT,
            Self::ShapeMismatch => sys::SLLM_STATUS_SHAPE_MISMATCH,
            Self::StrideMismatch => sys::SLLM_STATUS_STRIDE_MISMATCH,
            Self::MetadataOverflow => sys::SLLM_STATUS_METADATA_OVERFLOW,
            Self::BufferOutOfBounds => sys::SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            Self::MisalignedOffset => sys::SLLM_STATUS_MISALIGNED_OFFSET,
            Self::UnsupportedDType => sys::SLLM_STATUS_UNSUPPORTED_DTYPE,
            Self::UnsupportedEncoding => sys::SLLM_STATUS_UNSUPPORTED_ENCODING,
            Self::InvalidEpsilon => sys::SLLM_STATUS_INVALID_EPSILON,
            Self::UnsupportedScaleMode => sys::SLLM_STATUS_UNSUPPORTED_SCALE_MODE,
            Self::AliasOverlap => sys::SLLM_STATUS_ALIAS_OVERLAP,
            Self::ContextOrDeviceMismatch => sys::SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH,
            Self::InvalidElementwiseDescriptor => sys::SLLM_STATUS_INVALID_ELEMENTWISE_DESCRIPTOR,
            Self::InvalidEmbeddingDescriptor => sys::SLLM_STATUS_INVALID_EMBEDDING_DESCRIPTOR,
            Self::TokenIdOutOfRange => sys::SLLM_STATUS_TOKEN_ID_OUT_OF_RANGE,
            Self::InvalidMatmulDescriptor => sys::SLLM_STATUS_INVALID_MATMUL_DESCRIPTOR,
            Self::InvalidAttentionPreprocessDescriptor => {
                sys::SLLM_STATUS_INVALID_ATTENTION_PREPROCESS_DESCRIPTOR
            }
            Self::PositionPayloadMismatch => sys::SLLM_STATUS_POSITION_PAYLOAD_MISMATCH,
            Self::InvalidKvStateDescriptor => sys::SLLM_STATUS_INVALID_KV_STATE_DESCRIPTOR,
            Self::InvalidKvAppendDescriptor => sys::SLLM_STATUS_INVALID_KV_APPEND_DESCRIPTOR,
            Self::KvLengthMismatch => sys::SLLM_STATUS_KV_LENGTH_MISMATCH,
            Self::KvCapacityExceeded => sys::SLLM_STATUS_KV_CAPACITY_EXCEEDED,
            Self::InvalidCausalAttentionDescriptor => {
                sys::SLLM_STATUS_INVALID_CAUSAL_ATTENTION_DESCRIPTOR
            }
            Self::CausalAttentionLengthMismatch => {
                sys::SLLM_STATUS_CAUSAL_ATTENTION_LENGTH_MISMATCH
            }
            Self::CausalAttentionStateBusy => sys::SLLM_STATUS_CAUSAL_ATTENTION_STATE_BUSY,
            Self::Unknown(raw) => raw,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeError {
    status: RuntimeStatus,
    message: String,
}

impl RuntimeError {
    pub(crate) fn new(status: RuntimeStatus, message: String) -> Self {
        Self { status, message }
    }

    pub(crate) fn local(status: RuntimeStatus, message: &'static str) -> Self {
        Self::new(status, message.to_owned())
    }

    pub const fn status(&self) -> RuntimeStatus {
        self.status
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", status_name(self.status), self.message)
    }
}

impl std::error::Error for RuntimeError {}

fn status_name(status: RuntimeStatus) -> &'static str {
    match status {
        RuntimeStatus::Ok => "ok",
        RuntimeStatus::InvalidArgument => "invalid argument",
        RuntimeStatus::BufferTooSmall => "buffer too small",
        RuntimeStatus::Unsupported => "unsupported",
        RuntimeStatus::HipUnavailable => "HIP unavailable",
        RuntimeStatus::InvalidAbiVersion => "invalid ABI version",
        RuntimeStatus::ReservedNonzero => "reserved field is non-zero",
        RuntimeStatus::InternalError => "internal error",
        RuntimeStatus::Pending => "completion pending",
        RuntimeStatus::Timeout => "completion timeout",
        RuntimeStatus::InvalidHandle => "invalid or stale handle",
        RuntimeStatus::DeviceMismatch => "device or context mismatch",
        RuntimeStatus::HipRuntimeError => "HIP runtime error",
        RuntimeStatus::Busy => "resource is busy",
        RuntimeStatus::NotReady => "resource is not ready",
        RuntimeStatus::InvalidRmsNormDescriptor => "invalid RMSNorm descriptor",
        RuntimeStatus::InvalidTensorBinding => "invalid tensor binding",
        RuntimeStatus::ZeroExtent => "zero tensor extent",
        RuntimeStatus::ShapeMismatch => "tensor shape mismatch",
        RuntimeStatus::StrideMismatch => "tensor stride mismatch",
        RuntimeStatus::MetadataOverflow => "tensor metadata overflow",
        RuntimeStatus::BufferOutOfBounds => "tensor interval is out of bounds",
        RuntimeStatus::MisalignedOffset => "misaligned tensor offset",
        RuntimeStatus::UnsupportedDType => "unsupported tensor dtype",
        RuntimeStatus::UnsupportedEncoding => "unsupported tensor encoding",
        RuntimeStatus::InvalidEpsilon => "invalid RMSNorm epsilon",
        RuntimeStatus::UnsupportedScaleMode => "unsupported RMSNorm scale mode",
        RuntimeStatus::AliasOverlap => "overlapping semantic tensor intervals",
        RuntimeStatus::ContextOrDeviceMismatch => "semantic context or device mismatch",
        RuntimeStatus::InvalidElementwiseDescriptor => "invalid elementwise descriptor",
        RuntimeStatus::InvalidEmbeddingDescriptor => "invalid embedding descriptor",
        RuntimeStatus::TokenIdOutOfRange => "embedding token ID is out of range",
        RuntimeStatus::InvalidMatmulDescriptor => "invalid matmul descriptor",
        RuntimeStatus::InvalidAttentionPreprocessDescriptor => {
            "invalid attention preprocess descriptor"
        }
        RuntimeStatus::PositionPayloadMismatch => "attention preprocess position payload mismatch",
        RuntimeStatus::InvalidKvStateDescriptor => "invalid KV state descriptor",
        RuntimeStatus::InvalidKvAppendDescriptor => "invalid KV append descriptor",
        RuntimeStatus::KvLengthMismatch => "KV state length mismatch",
        RuntimeStatus::KvCapacityExceeded => "KV state capacity exceeded",
        RuntimeStatus::InvalidCausalAttentionDescriptor => "invalid causal attention descriptor",
        RuntimeStatus::CausalAttentionLengthMismatch => "causal attention length mismatch",
        RuntimeStatus::CausalAttentionStateBusy => "causal attention state is busy",
        RuntimeStatus::Unknown(_) => "unknown public runtime status",
    }
}

pub(crate) fn sink(buffer: &mut [u8; ERROR_CAPACITY]) -> sys::sllm_error_sink_t {
    sys::sllm_error_sink_t {
        struct_size: size_of::<sys::sllm_error_sink_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        message: buffer.as_mut_ptr().cast(),
        message_capacity: buffer.len() as u64,
        message_length: 0,
        reserved: [0, 0],
    }
}

pub(crate) fn diagnostic(buffer: &[u8; ERROR_CAPACITY], length: u64) -> String {
    let length = match usize::try_from(length) {
        Ok(value) => value,
        Err(_) => buffer.len(),
    }
    .min(buffer.len().saturating_sub(1));
    String::from_utf8_lossy(&buffer[..length]).into_owned()
}

pub(crate) fn result_error(raw: u32, buffer: &[u8; ERROR_CAPACITY], length: u64) -> RuntimeError {
    RuntimeError::new(RuntimeStatus::from_raw(raw), diagnostic(buffer, length))
}

pub(crate) fn ensure_ok(
    raw: u32,
    buffer: &[u8; ERROR_CAPACITY],
    length: u64,
) -> Result<(), RuntimeError> {
    if RuntimeStatus::from_raw(raw) == RuntimeStatus::Ok {
        Ok(())
    } else {
        Err(result_error(raw, buffer, length))
    }
}

fn copy_c_string(destination: &mut [c_char], value: &str) -> Result<(), RuntimeError> {
    if value.is_empty() || value.as_bytes().contains(&0) || value.len() >= destination.len() {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidArgument,
            "gcnArchName must be non-empty, NUL-free, and fit the ABI field",
        ));
    }
    destination.fill(0);
    for (slot, byte) in destination.iter_mut().zip(value.bytes()) {
        *slot = byte as c_char;
    }
    Ok(())
}

fn read_c_string(value: &[c_char]) -> String {
    let length = match value.iter().position(|byte| *byte == 0) {
        Some(value) => value,
        None => value.len(),
    };
    value[..length]
        .iter()
        .map(|byte| *byte as u8)
        .collect::<Vec<u8>>()
        .into_iter()
        .map(char::from)
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo {
    pub device_index: u32,
    pub visible_device_count: u32,
    pub total_memory_bytes: u64,
    pub wavefront_size: u32,
    pub name: String,
    pub gcn_arch_name: String,
}

fn device_info_from_raw(info: &sys::sllm_device_info_t) -> DeviceInfo {
    DeviceInfo {
        device_index: info.device_index,
        visible_device_count: info.visible_device_count,
        total_memory_bytes: info.total_memory_bytes,
        wavefront_size: info.wavefront_size,
        name: read_c_string(&info.name),
        gcn_arch_name: read_c_string(&info.gcn_arch_name),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionState {
    Pending,
    Success,
    Failure,
}

impl CompletionState {
    fn from_raw(raw: u32) -> Result<Self, RuntimeError> {
        match raw {
            sys::SLLM_COMPLETION_STATE_PENDING => Ok(Self::Pending),
            sys::SLLM_COMPLETION_STATE_SUCCESS => Ok(Self::Success),
            sys::SLLM_COMPLETION_STATE_FAILURE => Ok(Self::Failure),
            _ => Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "native completion returned an unknown state",
            )),
        }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CleanupDisposition {
    Recoverable,
    Poisoned,
}

enum PendingCleanup {
    Context {
        raw: Option<NonNull<sys::sllm_context_t>>,
        disposition: CleanupDisposition,
    },
    Queue {
        raw: Option<NonNull<sys::sllm_queue_t>>,
        context: Arc<ContextInner>,
        disposition: CleanupDisposition,
    },
    Buffer {
        raw: Option<NonNull<sys::sllm_buffer_t>>,
        context: Arc<ContextInner>,
        disposition: CleanupDisposition,
    },
    Event {
        raw: Option<NonNull<sys::sllm_event_t>>,
        context: Arc<ContextInner>,
        disposition: CleanupDisposition,
    },
    Completion {
        raw: Option<NonNull<sys::sllm_completion_t>>,
        context: Arc<ContextInner>,
        queue: Arc<QueueInner>,
        buffer: Arc<BufferInner>,
        disposition: CleanupDisposition,
    },
    KvState {
        raw: Option<NonNull<sys::sllm_kv_state_t>>,
        context: Context,
        disposition: CleanupDisposition,
    },
    KvView {
        raw: Option<NonNull<sys::sllm_kv_view_t>>,
        context: Context,
        disposition: CleanupDisposition,
    },
    KvCompletion {
        raw: Option<NonNull<sys::sllm_completion_t>>,
        context: Context,
        queue: Queue,
        key: Buffer,
        value: Buffer,
        state: crate::kv_state::KvStateResource,
        disposition: CleanupDisposition,
    },
    CausalCompletion {
        raw: Option<NonNull<sys::sllm_completion_t>>,
        context: Context,
        queue: Queue,
        query: Buffer,
        output: Buffer,
        state: crate::kv_state::KvStateResource,
        disposition: CleanupDisposition,
    },
    RmsNormPlan {
        raw: Option<NonNull<sys::sllm_rmsnorm_plan_t>>,
        context: Arc<ContextInner>,
        descriptor: Box<crate::rmsnorm::RmsNormDescriptor>,
        disposition: CleanupDisposition,
    },
    ElementwisePlan {
        raw: Option<NonNull<sys::sllm_elementwise_plan_t>>,
        context: Arc<ContextInner>,
        descriptor: Box<crate::elementwise::ElementwiseDescriptor>,
        disposition: CleanupDisposition,
    },
    EmbeddingPlan {
        raw: Option<NonNull<sys::sllm_embedding_plan_t>>,
        context: Arc<ContextInner>,
        descriptor: Box<crate::embedding::EmbeddingDescriptor>,
        disposition: CleanupDisposition,
    },
    MatmulPlan {
        raw: Option<NonNull<sys::sllm_matmul_plan_t>>,
        context: Arc<ContextInner>,
        descriptor: Box<crate::matmul::MatmulDescriptor>,
        disposition: CleanupDisposition,
    },
    AttentionPreprocessPlan {
        raw: Option<NonNull<sys::sllm_attention_preprocess_plan_t>>,
        context: Arc<ContextInner>,
        descriptor: Box<crate::attention_preprocess::AttentionPreprocessDescriptor>,
        disposition: CleanupDisposition,
    },
}

struct CleanupOwner {
    recoverable: Vec<CleanupRecord>,
}

/* A cleanup record is counted before it is made visible in TLS.  Taking it
 * for an attempt moves only this wrapper; the record's count stays live until
 * the attempt returns terminal success or it is transferred to quarantine.
 * Rust's ownership of the wrapper makes a second terminal decrement
 * impossible without an explicit new record. */
struct CleanupRecord {
    cleanup: PendingCleanup,
}

enum CleanupAttempt {
    Complete,
    Retry(CleanupRecord),
}

impl CleanupRecord {
    fn accepted(cleanup: PendingCleanup) -> Result<Self, PendingCleanup> {
        if checked_increment(&PENDING_CLEANUP_ITEMS, CleanupCasTarget::PendingIncrement).is_err() {
            record_cleanup_accounting_error();
            return Err(cleanup);
        }
        Ok(Self { cleanup })
    }

    fn try_once(self) -> CleanupAttempt {
        let Self { cleanup } = self;
        match cleanup.try_once() {
            Some(cleanup) => CleanupAttempt::Retry(Self { cleanup }),
            None => CleanupAttempt::Complete,
        }
    }
}

impl CleanupOwner {
    const fn new() -> Self {
        Self {
            recoverable: Vec::new(),
        }
    }

    fn push(&mut self, record: CleanupRecord) -> Result<(), CleanupRecord> {
        if record.cleanup.is_poisoned() || self.recoverable.try_reserve(1).is_err() {
            return Err(record);
        }
        self.recoverable.push(record);
        Ok(())
    }

    fn take_recoverable(&mut self) -> Vec<CleanupRecord> {
        std::mem::take(&mut self.recoverable)
    }
}

impl Drop for CleanupOwner {
    fn drop(&mut self) {
        /* This is the thread-exit handoff.  Do not drop the vectors: dropping
         * their elements would drop Arc dependencies and can re-enter this
         * TLS key while Rust is destroying it.  Leaking the backing storage is
         * an explicit process-lifetime quarantine of both raw tokens and the
         * dependency graph.  No allocation, FFI call, lock, or TLS lookup is
         * performed from this destructor. */
        let recoverable = std::mem::take(&mut self.recoverable);
        for record in recoverable {
            quarantine_from_tls(record);
        }
    }
}

thread_local! {
    static CLEANUP_REAPER: RefCell<CleanupOwner> = const { RefCell::new(CleanupOwner::new()) };
}

static PENDING_CLEANUP_ITEMS: AtomicUsize = AtomicUsize::new(0);
static DURABLE_QUARANTINE_ITEMS: AtomicUsize = AtomicUsize::new(0);
/* Once set, this sentinel is a durable process-lifetime accounting bucket for
 * every quarantine that cannot be represented by the saturated exact count.
 * It is published before ownership is forgotten and is never cleared outside
 * serialized tests. */
static DURABLE_QUARANTINE_OVERFLOW: AtomicUsize = AtomicUsize::new(0);
static CLEANUP_ACCOUNTING_ERRORS: AtomicUsize = AtomicUsize::new(0);

/* Bit 0 is an exclusive drain barrier.  The remaining bits count cleanup
 * handoffs that have been accepted but are not yet published or terminal.
 * A drain sets the barrier with compare_exchange, waits for already accepted
 * handoffs, and then samples the pending count while new handoffs are forced
 * directly to quarantine.  This closes both the take/requeue window and the
 * increment-after-publication window without moving Arc-bearing records across
 * threads. */
const CLEANUP_DRAIN_BIT: usize = 1;
const CLEANUP_HANDOFF_UNIT: usize = 2;
/* Every internal compare-exchange loop has this fixed production bound.  A
 * caller-provided drain bound is additionally capped here, so usize::MAX can
 * never turn a nominally bounded cleanup operation into an unbounded retry. */
const CLEANUP_CAS_BOUND: usize = 16;
static CLEANUP_HANDOFF_STATE: AtomicUsize = AtomicUsize::new(0);
/* A handoff increments this generation before removing its state ticket.  It
 * never wraps: saturation leaves the handoff active and records a sticky
 * accounting error. */
static CLEANUP_HANDOFF_EPOCH: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CheckedAtomicError {
    Saturated,
    Underflow,
    Contended,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CleanupCasTarget {
    PendingIncrement,
    PendingDecrement,
    DurableIncrement,
    AccountingError,
    HandoffEpochIncrement,
    HandoffBegin,
    HandoffRollback,
    HandoffEnd,
    HandoffDrainEnter,
    HandoffDrainExit,
}

#[cfg(test)]
struct CasFailureSpec {
    thread: ThreadId,
    target: CleanupCasTarget,
    remaining: usize,
}

#[cfg(test)]
static FORCED_CAS_FAILURES: Mutex<Vec<CasFailureSpec>> = Mutex::new(Vec::new());

#[cfg(test)]
struct CasFailureGuard {
    thread: ThreadId,
    target: CleanupCasTarget,
}

#[cfg(test)]
impl CasFailureGuard {
    fn new(target: CleanupCasTarget, attempts: usize) -> Self {
        assert!(attempts != 0 && attempts <= CLEANUP_CAS_BOUND);
        let thread = std::thread::current().id();
        FORCED_CAS_FAILURES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(CasFailureSpec {
                thread,
                target,
                remaining: attempts,
            });
        Self { thread, target }
    }
}

#[cfg(test)]
impl Drop for CasFailureGuard {
    fn drop(&mut self) {
        FORCED_CAS_FAILURES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|spec| spec.thread != self.thread || spec.target != self.target);
    }
}

#[cfg(test)]
fn cleanup_cas_failure_injected(target: CleanupCasTarget) -> bool {
    let thread = std::thread::current().id();
    let mut failures = FORCED_CAS_FAILURES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(index) = failures
        .iter()
        .position(|spec| spec.thread == thread && spec.target == target)
    else {
        return false;
    };
    let spec = &mut failures[index];
    spec.remaining -= 1;
    if spec.remaining == 0 {
        failures.remove(index);
    }
    true
}

#[cfg(not(test))]
const fn cleanup_cas_failure_injected(_target: CleanupCasTarget) -> bool {
    false
}

#[cfg(test)]
fn clear_cleanup_cas_failures() {
    FORCED_CAS_FAILURES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
}

fn checked_increment(
    counter: &AtomicUsize,
    target: CleanupCasTarget,
) -> Result<usize, CheckedAtomicError> {
    let mut current = counter.load(Ordering::Acquire);
    for _ in 0..CLEANUP_CAS_BOUND {
        let next = current
            .checked_add(1)
            .ok_or(CheckedAtomicError::Saturated)?;
        if cleanup_cas_failure_injected(target) {
            continue;
        }
        match counter.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Ok(next),
            Err(observed) => current = observed,
        }
    }
    Err(CheckedAtomicError::Contended)
}

fn checked_decrement(
    counter: &AtomicUsize,
    target: CleanupCasTarget,
) -> Result<usize, CheckedAtomicError> {
    let mut current = counter.load(Ordering::Acquire);
    for _ in 0..CLEANUP_CAS_BOUND {
        let next = current
            .checked_sub(1)
            .ok_or(CheckedAtomicError::Underflow)?;
        if cleanup_cas_failure_injected(target) {
            continue;
        }
        match counter.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Ok(next),
            Err(observed) => current = observed,
        }
    }
    Err(CheckedAtomicError::Contended)
}

fn record_cleanup_accounting_error() {
    let mut current = CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire);
    for _ in 0..CLEANUP_CAS_BOUND {
        let Some(next) = current.checked_add(1) else {
            return;
        };
        if cleanup_cas_failure_injected(CleanupCasTarget::AccountingError) {
            continue;
        }
        match CLEANUP_ACCOUNTING_ERRORS.compare_exchange_weak(
            current,
            next,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
    /* Contention must not make an error disappear.  Saturating the sticky
     * public counter loses multiplicity but preserves fail-closed state. */
    CLEANUP_ACCOUNTING_ERRORS.store(usize::MAX, Ordering::Release);
}

fn cleanup_handoff_state_is_valid(state: usize) -> bool {
    (state & !CLEANUP_DRAIN_BIT) % CLEANUP_HANDOFF_UNIT == 0
}

enum CleanupHandoff {
    Publish,
    Quarantine,
    QuarantineUncounted,
}

fn cleanup_handoff_is_counted(handoff: &CleanupHandoff) -> bool {
    matches!(
        handoff,
        CleanupHandoff::Publish | CleanupHandoff::Quarantine
    )
}

#[cfg(test)]
struct ReapPauseHook {
    thread: ThreadId,
    entered: Arc<TimedGate>,
    release: Arc<TimedGate>,
    panic_after_signal: bool,
}

#[cfg(test)]
struct TimedGate {
    signaled: Mutex<bool>,
    ready: Condvar,
}

#[cfg(test)]
impl TimedGate {
    fn new() -> Self {
        Self {
            signaled: Mutex::new(false),
            ready: Condvar::new(),
        }
    }

    fn signal(&self) {
        *self
            .signaled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        self.ready.notify_all();
    }

    fn wait_for(&self, timeout: Duration) -> bool {
        let signaled = self
            .signaled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *signaled {
            return true;
        }
        let (signaled, _) = self
            .ready
            .wait_timeout_while(signaled, timeout, |ready| !*ready)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *signaled
    }

    fn wait(&self) -> bool {
        self.wait_for(Duration::from_secs(2))
    }
}

#[cfg(test)]
static REAP_PAUSE_HOOK: Mutex<Option<ReapPauseHook>> = Mutex::new(None);

#[cfg(test)]
static FORCE_RETRY_THREAD: Mutex<Option<ThreadId>> = Mutex::new(None);

#[cfg(test)]
static DURABLE_FORGET_PAUSE_HOOK: Mutex<Option<ReapPauseHook>> = Mutex::new(None);

#[cfg(test)]
static POST_CAS_PAUSE_HOOK: Mutex<Option<ReapPauseHook>> = Mutex::new(None);

/* Own the cleanup across the interval in which the handoff state ticket is
 * live.  In particular, the hook below is allowed to panic in tests.  The
 * guard must publish the raw token and its Arc graph before removing the state
 * ticket; otherwise a concurrent drain could observe the old counts after the
 * ticket disappeared and report a false success. */
struct CleanupHandoffGuard {
    cleanup: Option<PendingCleanup>,
    record: Option<CleanupRecord>,
    pending_ticket: bool,
    handoff_started: bool,
    active: bool,
}

impl CleanupHandoffGuard {
    fn for_cleanup(cleanup: PendingCleanup) -> Self {
        Self {
            cleanup: Some(cleanup),
            record: None,
            pending_ticket: false,
            handoff_started: false,
            active: true,
        }
    }

    fn for_record(record: CleanupRecord) -> Self {
        Self {
            cleanup: None,
            record: Some(record),
            pending_ticket: true,
            handoff_started: false,
            active: true,
        }
    }

    fn mark_handoff_started(&mut self) {
        self.handoff_started = true;
    }

    fn accept_pending(&mut self) -> bool {
        let Some(cleanup) = self.cleanup.take() else {
            return self.record.is_some();
        };
        match CleanupRecord::accepted(cleanup) {
            Ok(record) => {
                self.record = Some(record);
                self.pending_ticket = true;
                true
            }
            Err(cleanup) => {
                self.cleanup = Some(cleanup);
                false
            }
        }
    }

    fn take_record(&mut self) -> Option<CleanupRecord> {
        self.record.take()
    }

    fn restore_record(&mut self, record: CleanupRecord) {
        debug_assert!(self.record.is_none());
        self.record = Some(record);
    }

    fn quarantine_durable(&mut self) {
        let cleanup = self
            .cleanup
            .take()
            .or_else(|| self.record.take().map(|record| record.cleanup));
        if let Some(cleanup) = cleanup {
            durable_quarantine(cleanup);
        }
    }

    fn finish_pending(&mut self) {
        if self.pending_ticket {
            self.pending_ticket = false;
            finish_pending_cleanup();
        }
    }

    fn quarantine_owned(&mut self) {
        self.quarantine_durable();
        pause_after_durable_before_pending_decrement_for_test();
        self.finish_pending();
    }

    fn commit_record_transfer(&mut self) {
        debug_assert!(self.record.is_none());
        self.pending_ticket = false;
    }

    fn disarm(&mut self) {
        debug_assert!(self.cleanup.is_none());
        debug_assert!(self.record.is_none());
        debug_assert!(!self.pending_ticket);
        self.active = false;
    }
}

impl Drop for CleanupHandoffGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        /* Keep this order.  `quarantine_owned` is durable before the pending
         * ticket is removed, and both happen before the aborted state ticket
         * is rolled back.  The rollback deliberately does not publish an
         * epoch: this handoff did not complete publication. */
        self.quarantine_owned();
        if self.handoff_started {
            rollback_cleanup_handoff();
        }
    }
}

fn begin_cleanup_handoff(recovery: &mut CleanupHandoffGuard) -> CleanupHandoff {
    let mut current = CLEANUP_HANDOFF_STATE.load(Ordering::Acquire);
    for _ in 0..CLEANUP_CAS_BOUND {
        let Some(next) = current.checked_add(CLEANUP_HANDOFF_UNIT) else {
            record_cleanup_accounting_error();
            return CleanupHandoff::QuarantineUncounted;
        };
        if !cleanup_handoff_state_is_valid(current) {
            record_cleanup_accounting_error();
            return CleanupHandoff::QuarantineUncounted;
        }
        let handoff = if current & CLEANUP_DRAIN_BIT != 0 {
            CleanupHandoff::Quarantine
        } else {
            CleanupHandoff::Publish
        };
        if cleanup_cas_failure_injected(CleanupCasTarget::HandoffBegin) {
            continue;
        }
        match CLEANUP_HANDOFF_STATE.compare_exchange_weak(
            current,
            next,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                recovery.mark_handoff_started();
                pause_after_handoff_start_for_test();
                return handoff;
            }
            Err(observed) => current = observed,
        }
    }
    record_cleanup_accounting_error();
    CleanupHandoff::QuarantineUncounted
}

fn rollback_cleanup_handoff() {
    let mut current = CLEANUP_HANDOFF_STATE.load(Ordering::Acquire);
    for _ in 0..CLEANUP_CAS_BOUND {
        if !cleanup_handoff_state_is_valid(current)
            || current & !CLEANUP_DRAIN_BIT < CLEANUP_HANDOFF_UNIT
        {
            record_cleanup_accounting_error();
            return;
        }
        if cleanup_cas_failure_injected(CleanupCasTarget::HandoffRollback) {
            continue;
        }
        match CLEANUP_HANDOFF_STATE.compare_exchange_weak(
            current,
            current - CLEANUP_HANDOFF_UNIT,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
    record_cleanup_accounting_error();
}

fn end_cleanup_handoff() {
    /* Publication completion linearizes through the epoch before its active
     * state ticket is removed.  If the epoch cannot advance, retaining the
     * ticket makes every later drain fail closed. */
    if checked_increment(
        &CLEANUP_HANDOFF_EPOCH,
        CleanupCasTarget::HandoffEpochIncrement,
    )
    .is_err()
    {
        record_cleanup_accounting_error();
        return;
    }
    let mut current = CLEANUP_HANDOFF_STATE.load(Ordering::Acquire);
    for _ in 0..CLEANUP_CAS_BOUND {
        if !cleanup_handoff_state_is_valid(current)
            || current & !CLEANUP_DRAIN_BIT < CLEANUP_HANDOFF_UNIT
        {
            record_cleanup_accounting_error();
            return;
        }
        if cleanup_cas_failure_injected(CleanupCasTarget::HandoffEnd) {
            continue;
        }
        match CLEANUP_HANDOFF_STATE.compare_exchange_weak(
            current,
            current - CLEANUP_HANDOFF_UNIT,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
    record_cleanup_accounting_error();
}

struct CleanupDrainGuard;

impl Drop for CleanupDrainGuard {
    fn drop(&mut self) {
        exit_cleanup_drain();
    }
}

fn enter_cleanup_drain(max_attempts: usize) -> Result<CleanupDrainGuard, RuntimeError> {
    let attempt_budget = max_attempts.saturating_add(1).min(CLEANUP_CAS_BOUND);
    for attempt in 0..attempt_budget {
        let current = CLEANUP_HANDOFF_STATE.load(Ordering::Acquire);
        if !cleanup_handoff_state_is_valid(current) {
            record_cleanup_accounting_error();
            return Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "cleanup handoff barrier is inconsistent",
            ));
        }
        if current & CLEANUP_DRAIN_BIT != 0 {
            if attempt + 1 == attempt_budget {
                return Err(RuntimeError::local(
                    RuntimeStatus::Busy,
                    "cleanup drain is already active",
                ));
            }
            std::thread::yield_now();
            continue;
        }
        if cleanup_cas_failure_injected(CleanupCasTarget::HandoffDrainEnter) {
            if attempt + 1 == attempt_budget {
                return Err(RuntimeError::local(
                    RuntimeStatus::Busy,
                    "cleanup drain could not acquire its barrier",
                ));
            }
            continue;
        }
        if CLEANUP_HANDOFF_STATE
            .compare_exchange_weak(
                current,
                current | CLEANUP_DRAIN_BIT,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            break;
        }
        if attempt + 1 == attempt_budget {
            return Err(RuntimeError::local(
                RuntimeStatus::Busy,
                "cleanup drain could not acquire its barrier",
            ));
        }
    }

    for attempt in 0..attempt_budget {
        let current = CLEANUP_HANDOFF_STATE.load(Ordering::Acquire);
        if !cleanup_handoff_state_is_valid(current) {
            exit_cleanup_drain();
            record_cleanup_accounting_error();
            return Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "cleanup handoff barrier is inconsistent",
            ));
        }
        if current & !CLEANUP_DRAIN_BIT == 0 {
            return Ok(CleanupDrainGuard);
        }
        if attempt + 1 == attempt_budget {
            exit_cleanup_drain();
            return Err(RuntimeError::local(
                RuntimeStatus::Busy,
                "cleanup handoffs did not quiesce within the drain bound",
            ));
        }
        std::thread::yield_now();
    }

    unreachable!("bounded cleanup drain loop must return");
}

fn exit_cleanup_drain() {
    let mut current = CLEANUP_HANDOFF_STATE.load(Ordering::Acquire);
    for _ in 0..CLEANUP_CAS_BOUND {
        if !cleanup_handoff_state_is_valid(current) || current & CLEANUP_DRAIN_BIT == 0 {
            record_cleanup_accounting_error();
            return;
        }
        if cleanup_cas_failure_injected(CleanupCasTarget::HandoffDrainExit) {
            continue;
        }
        match CLEANUP_HANDOFF_STATE.compare_exchange_weak(
            current,
            current & !CLEANUP_DRAIN_BIT,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
    record_cleanup_accounting_error();
}

fn finish_pending_cleanup() {
    if checked_decrement(&PENDING_CLEANUP_ITEMS, CleanupCasTarget::PendingDecrement).is_err() {
        record_cleanup_accounting_error();
    }
}

fn quarantine_pending_record(record: CleanupRecord) {
    let mut recovery = CleanupHandoffGuard::for_record(record);
    let handoff = begin_cleanup_handoff(&mut recovery);
    /* Durable exact accounting or its sticky overflow bucket is published
     * before the pending ticket is removed. */
    recovery.quarantine_durable();
    pause_after_durable_before_pending_decrement_for_test();
    recovery.finish_pending();
    if cleanup_handoff_is_counted(&handoff) {
        end_cleanup_handoff();
    }
    recovery.disarm();
}

fn quarantine_from_tls(record: CleanupRecord) {
    quarantine_pending_record(record);
}

#[cfg(test)]
fn pause_after_reap_take_for_test() {
    let current_thread = std::thread::current().id();
    let hook = {
        let mut configured = REAP_PAUSE_HOOK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if configured
            .as_ref()
            .is_some_and(|hook| hook.thread == current_thread)
        {
            configured.take()
        } else {
            None
        }
    };
    if let Some(hook) = hook {
        hook.entered.signal();
        assert!(hook.release.wait(), "cleanup test hook release timed out");
    }
}

#[cfg(test)]
static HANDOFF_PAUSE_HOOK: Mutex<Option<ReapPauseHook>> = Mutex::new(None);

#[cfg(test)]
static QUARANTINE_PAUSE_HOOK: Mutex<Option<ReapPauseHook>> = Mutex::new(None);

#[cfg(test)]
static SNAPSHOT_PAUSE_HOOK: Mutex<Option<ReapPauseHook>> = Mutex::new(None);

#[cfg(test)]
fn take_owned_pause_hook(hooks: &Mutex<Option<ReapPauseHook>>) -> Option<ReapPauseHook> {
    let current_thread = std::thread::current().id();
    let mut configured = hooks
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if configured
        .as_ref()
        .is_some_and(|hook| hook.thread == current_thread)
    {
        configured.take()
    } else {
        None
    }
}

#[cfg(test)]
fn pause_after_handoff_start_for_test() {
    if let Some(hook) = take_owned_pause_hook(&HANDOFF_PAUSE_HOOK) {
        hook.entered.signal();
        if hook.panic_after_signal {
            panic!("intentional cleanup handoff pause-hook panic");
        }
        assert!(hook.release.wait(), "cleanup test hook release timed out");
    }
}

#[cfg(not(test))]
fn pause_after_handoff_start_for_test() {}

#[cfg(test)]
fn pause_after_durable_before_pending_decrement_for_test() {
    if let Some(hook) = take_owned_pause_hook(&QUARANTINE_PAUSE_HOOK) {
        hook.entered.signal();
        assert!(hook.release.wait(), "cleanup test hook release timed out");
    }
}

#[cfg(not(test))]
fn pause_after_durable_before_pending_decrement_for_test() {}

#[cfg(test)]
fn pause_after_cleanup_snapshot_for_test() {
    if let Some(hook) = take_owned_pause_hook(&SNAPSHOT_PAUSE_HOOK) {
        hook.entered.signal();
        assert!(hook.release.wait(), "cleanup test hook release timed out");
    }
}

#[cfg(test)]
fn pause_after_durable_before_forget_for_test() {
    if let Some(hook) = take_owned_pause_hook(&DURABLE_FORGET_PAUSE_HOOK) {
        hook.entered.signal();
        /* A failed observation must let the worker finish; the owning test
         * thread reports the missing gate separately.  This keeps a test
         * hook timeout from leaving the handoff ticket active while unwinding. */
        let _ = hook.release.wait();
    }
}

#[cfg(not(test))]
fn pause_after_durable_before_forget_for_test() {}

#[cfg(test)]
fn pause_after_cleanup_noop_cas_for_test() {
    if let Some(hook) = take_owned_pause_hook(&POST_CAS_PAUSE_HOOK) {
        hook.entered.signal();
        assert!(hook.release.wait(), "cleanup test hook release timed out");
    }
}

#[cfg(not(test))]
fn pause_after_cleanup_noop_cas_for_test() {}

#[cfg(not(test))]
fn pause_after_cleanup_snapshot_for_test() {}

#[cfg(not(test))]
fn pause_after_reap_take_for_test() {}

#[cfg(test)]
fn force_retry_once_for_test() -> bool {
    let current_thread = std::thread::current().id();
    let mut configured = FORCE_RETRY_THREAD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if configured.as_ref() == Some(&current_thread) {
        *configured = None;
        true
    } else {
        false
    }
}

#[cfg(not(test))]
fn force_retry_once_for_test() -> bool {
    false
}

#[cfg(test)]
fn clear_cleanup_test_hooks() {
    for hooks in [
        &REAP_PAUSE_HOOK,
        &HANDOFF_PAUSE_HOOK,
        &QUARANTINE_PAUSE_HOOK,
        &SNAPSHOT_PAUSE_HOOK,
        &DURABLE_FORGET_PAUSE_HOOK,
        &POST_CAS_PAUSE_HOOK,
    ] {
        hooks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
    }
    FORCE_RETRY_THREAD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    clear_cleanup_cas_failures();
    clear_forced_rmsnorm_plan_release_for_test();
    clear_forced_matmul_plan_release_for_test();
}

struct DurableCleanupOwner {
    cleanup: Option<PendingCleanup>,
}

impl DurableCleanupOwner {
    fn new(cleanup: PendingCleanup) -> Self {
        Self {
            cleanup: Some(cleanup),
        }
    }
}

impl Drop for DurableCleanupOwner {
    fn drop(&mut self) {
        /* Once durable accounting has been published, the only safe unwind
         * action is to retain the ownership graph.  This destructor performs
         * no allocation, FFI, lock, or TLS access. */
        if let Some(cleanup) = self.cleanup.take() {
            std::mem::forget(cleanup);
        }
    }
}

fn durable_quarantine(cleanup: PendingCleanup) -> bool {
    let mut owner = DurableCleanupOwner::new(cleanup);
    let incremented = checked_increment(
        &DURABLE_QUARANTINE_ITEMS,
        CleanupCasTarget::DurableIncrement,
    )
    .is_ok();
    if !incremented {
        /* This sentinel is the durable accounting representation for this and
         * all later overflowed quarantines.  Publish it before forgetting the
         * ownership graph. */
        DURABLE_QUARANTINE_OVERFLOW.store(1, Ordering::Release);
        record_cleanup_accounting_error();
    }
    pause_after_durable_before_forget_for_test();
    /* Keep the raw token and every Arc dependency alive without allocation,
     * TLS lookup, or a fallible ownership handoff. */
    std::mem::forget(owner.cleanup.take());
    incremented
}

fn enqueue_cleanup(cleanup: PendingCleanup) {
    let poisoned = cleanup.is_poisoned();
    let mut recovery = CleanupHandoffGuard::for_cleanup(cleanup);
    let handoff = begin_cleanup_handoff(&mut recovery);
    if poisoned || !matches!(&handoff, CleanupHandoff::Publish) {
        recovery.quarantine_owned();
        if cleanup_handoff_is_counted(&handoff) {
            end_cleanup_handoff();
        }
        recovery.disarm();
        return;
    }
    if !recovery.accept_pending() {
        recovery.quarantine_owned();
        end_cleanup_handoff();
        recovery.disarm();
        return;
    }
    let mut pending = recovery.take_record();
    let inserted = CLEANUP_REAPER
        .try_with(|reaper| {
            let Ok(mut reaper) = reaper.try_borrow_mut() else {
                return false;
            };
            let Some(record) = pending.take() else {
                return true;
            };
            match reaper.push(record) {
                Ok(()) => true,
                Err(record) => {
                    pending = Some(record);
                    false
                }
            }
        })
        .unwrap_or(false);
    if !inserted {
        if let Some(record) = pending.take() {
            recovery.restore_record(record);
            recovery.quarantine_durable();
            pause_after_durable_before_pending_decrement_for_test();
            recovery.finish_pending();
        }
    } else {
        recovery.commit_record_transfer();
    }
    end_cleanup_handoff();
    recovery.disarm();
}

fn requeue_cleanup(record: CleanupRecord) {
    let mut recovery = CleanupHandoffGuard::for_record(record);
    let handoff = begin_cleanup_handoff(&mut recovery);
    if !matches!(&handoff, CleanupHandoff::Publish) {
        recovery.quarantine_durable();
        pause_after_durable_before_pending_decrement_for_test();
        recovery.finish_pending();
        if cleanup_handoff_is_counted(&handoff) {
            end_cleanup_handoff();
        }
        recovery.disarm();
        return;
    }
    let mut pending = recovery.take_record();
    let inserted = CLEANUP_REAPER
        .try_with(|reaper| {
            let Ok(mut reaper) = reaper.try_borrow_mut() else {
                return false;
            };
            let Some(record) = pending.take() else {
                return true;
            };
            match reaper.push(record) {
                Ok(()) => true,
                Err(record) => {
                    pending = Some(record);
                    false
                }
            }
        })
        .unwrap_or(false);
    if !inserted {
        if let Some(record) = pending.take() {
            recovery.restore_record(record);
            recovery.quarantine_durable();
            pause_after_durable_before_pending_decrement_for_test();
            recovery.finish_pending();
        }
    } else {
        recovery.commit_record_transfer();
    }
    end_cleanup_handoff();
    recovery.disarm();
}

impl PendingCleanup {
    fn is_poisoned(&self) -> bool {
        match self {
            Self::Context { disposition, .. }
            | Self::Queue { disposition, .. }
            | Self::Buffer { disposition, .. }
            | Self::Event { disposition, .. }
            | Self::Completion { disposition, .. }
            | Self::KvState { disposition, .. }
            | Self::KvView { disposition, .. }
            | Self::KvCompletion { disposition, .. }
            | Self::CausalCompletion { disposition, .. }
            | Self::RmsNormPlan { disposition, .. }
            | Self::ElementwisePlan { disposition, .. }
            | Self::EmbeddingPlan { disposition, .. }
            | Self::MatmulPlan { disposition, .. }
            | Self::AttentionPreprocessPlan { disposition, .. } => {
                *disposition == CleanupDisposition::Poisoned
            }
        }
    }
}

fn classify_release<T>(
    status: RuntimeStatus,
    remaining: Option<NonNull<T>>,
) -> (Option<NonNull<T>>, CleanupDisposition, bool) {
    if status == RuntimeStatus::Ok && remaining.is_none() {
        return (None, CleanupDisposition::Recoverable, true);
    }
    /* These statuses are returned before native destruction consumes the
     * token.  Retrying is therefore ownership-safe.  A HIP runtime error is
     * retryable only while the native call still returns the token; native
     * destruction failures consume it and return None. */
    if remaining.is_some()
        && matches!(
            status,
            RuntimeStatus::Busy
                | RuntimeStatus::NotReady
                | RuntimeStatus::Pending
                | RuntimeStatus::Timeout
                | RuntimeStatus::HipRuntimeError
        )
    {
        return (remaining, CleanupDisposition::Recoverable, false);
    }
    /* Every other result is ambiguous or terminal.  Never retry it even if
     * the caller still has a token; a null token means native quarantine
     * already consumed it, while a non-null token is retained conservatively. */
    (remaining, CleanupDisposition::Poisoned, false)
}

fn release_context_once(
    raw: NonNull<sys::sllm_context_t>,
) -> (RuntimeStatus, Option<NonNull<sys::sllm_context_t>>) {
    let mut error_buffer = [0_u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut error_buffer);
    let mut native = raw.as_ptr();
    let status = unsafe { sys::sllm_context_release(&mut native, &mut error_sink) };
    (RuntimeStatus::from_raw(status), NonNull::new(native))
}

fn release_queue_once(
    raw: NonNull<sys::sllm_queue_t>,
) -> (RuntimeStatus, Option<NonNull<sys::sllm_queue_t>>) {
    let mut error_buffer = [0_u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut error_buffer);
    let mut native = raw.as_ptr();
    let status = unsafe { sys::sllm_queue_release(&mut native, &mut error_sink) };
    (RuntimeStatus::from_raw(status), NonNull::new(native))
}

fn release_buffer_once(
    raw: NonNull<sys::sllm_buffer_t>,
) -> (RuntimeStatus, Option<NonNull<sys::sllm_buffer_t>>) {
    let mut error_buffer = [0_u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut error_buffer);
    let mut native = raw.as_ptr();
    let status = unsafe { sys::sllm_buffer_release(&mut native, &mut error_sink) };
    (RuntimeStatus::from_raw(status), NonNull::new(native))
}

fn release_event_once(
    raw: NonNull<sys::sllm_event_t>,
) -> (RuntimeStatus, Option<NonNull<sys::sllm_event_t>>) {
    let mut error_buffer = [0_u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut error_buffer);
    let mut native = raw.as_ptr();
    let status = unsafe { sys::sllm_event_release(&mut native, &mut error_sink) };
    (RuntimeStatus::from_raw(status), NonNull::new(native))
}

fn release_completion_once(
    raw: NonNull<sys::sllm_completion_t>,
) -> (RuntimeStatus, Option<NonNull<sys::sllm_completion_t>>) {
    let mut error_buffer = [0_u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut error_buffer);
    let mut native = raw.as_ptr();
    let status = unsafe { sys::sllm_completion_release(&mut native, &mut error_sink) };
    (RuntimeStatus::from_raw(status), NonNull::new(native))
}

pub(crate) fn release_kv_state_once(
    raw: NonNull<sys::sllm_kv_state_t>,
) -> (RuntimeStatus, Option<NonNull<sys::sllm_kv_state_t>>) {
    let mut error_buffer = [0_u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut error_buffer);
    let mut native = raw.as_ptr();
    let status = unsafe { sys::sllm_kv_state_release(&mut native, &mut error_sink) };
    (RuntimeStatus::from_raw(status), NonNull::new(native))
}

pub(crate) fn release_kv_view_once(
    raw: NonNull<sys::sllm_kv_view_t>,
) -> (RuntimeStatus, Option<NonNull<sys::sllm_kv_view_t>>) {
    let mut error_buffer = [0_u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut error_buffer);
    let mut native = raw.as_ptr();
    let status = unsafe { sys::sllm_kv_view_release(&mut native, &mut error_sink) };
    (RuntimeStatus::from_raw(status), NonNull::new(native))
}

pub(crate) fn release_kv_completion_once(
    raw: NonNull<sys::sllm_completion_t>,
) -> (RuntimeStatus, Option<NonNull<sys::sllm_completion_t>>) {
    release_completion_once(raw)
}

pub(crate) fn enqueue_kv_state_cleanup(
    raw: NonNull<sys::sllm_kv_state_t>,
    context: Context,
    status: RuntimeStatus,
) {
    let (_, disposition, _) = classify_release(status, Some(raw));
    enqueue_cleanup(PendingCleanup::KvState {
        raw: Some(raw),
        context,
        disposition,
    });
}

pub(crate) fn enqueue_kv_view_cleanup(
    raw: NonNull<sys::sllm_kv_view_t>,
    context: Context,
    status: RuntimeStatus,
) {
    let (_, disposition, _) = classify_release(status, Some(raw));
    enqueue_cleanup(PendingCleanup::KvView {
        raw: Some(raw),
        context,
        disposition,
    });
}

pub(crate) fn enqueue_kv_completion_cleanup(
    raw: NonNull<sys::sllm_completion_t>,
    context: Context,
    queue: Queue,
    key: Buffer,
    value: Buffer,
    state: crate::kv_state::KvStateResource,
    status: RuntimeStatus,
) {
    let (_, disposition, _) = classify_release(status, Some(raw));
    enqueue_cleanup(PendingCleanup::KvCompletion {
        raw: Some(raw),
        context,
        queue,
        key,
        value,
        state,
        disposition,
    });
}

pub(crate) fn release_causal_completion_once(
    raw: NonNull<sys::sllm_completion_t>,
) -> (RuntimeStatus, Option<NonNull<sys::sllm_completion_t>>) {
    release_completion_once(raw)
}

pub(crate) fn enqueue_causal_completion_cleanup(
    raw: NonNull<sys::sllm_completion_t>,
    context: Context,
    queue: Queue,
    query: Buffer,
    output: Buffer,
    state: crate::kv_state::KvStateResource,
    status: RuntimeStatus,
) {
    let (_, disposition, _) = classify_release(status, Some(raw));
    enqueue_cleanup(PendingCleanup::CausalCompletion {
        raw: Some(raw),
        context,
        queue,
        query,
        output,
        state,
        disposition,
    });
}

pub(crate) fn release_rmsnorm_plan_once(
    raw: NonNull<sys::sllm_rmsnorm_plan_t>,
) -> (RuntimeStatus, Option<NonNull<sys::sllm_rmsnorm_plan_t>>) {
    #[cfg(test)]
    if let Some((status, consumed)) = FORCED_RMSNORM_PLAN_RELEASE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
    {
        return (status, if consumed { None } else { Some(raw) });
    }

    let mut error_buffer = [0_u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut error_buffer);
    let mut native = raw.as_ptr();
    let status = unsafe { sys::sllm_rmsnorm_plan_release(&mut native, &mut error_sink) };
    (RuntimeStatus::from_raw(status), NonNull::new(native))
}

pub(crate) fn enqueue_rmsnorm_cleanup(
    raw: NonNull<sys::sllm_rmsnorm_plan_t>,
    context: Context,
    descriptor: crate::rmsnorm::RmsNormDescriptor,
    status: RuntimeStatus,
) {
    let (_, disposition, _) = classify_release(status, Some(raw));
    enqueue_cleanup(PendingCleanup::RmsNormPlan {
        raw: Some(raw),
        context: Arc::clone(&context.inner),
        descriptor: Box::new(descriptor),
        disposition,
    });
}

pub(crate) fn release_elementwise_plan_once(
    raw: NonNull<sys::sllm_elementwise_plan_t>,
) -> (RuntimeStatus, Option<NonNull<sys::sllm_elementwise_plan_t>>) {
    let mut error_buffer = [0_u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut error_buffer);
    let mut native = raw.as_ptr();
    let status = unsafe { sys::sllm_elementwise_plan_release(&mut native, &mut error_sink) };
    (RuntimeStatus::from_raw(status), NonNull::new(native))
}

pub(crate) fn enqueue_elementwise_cleanup(
    raw: NonNull<sys::sllm_elementwise_plan_t>,
    context: Context,
    descriptor: crate::elementwise::ElementwiseDescriptor,
    status: RuntimeStatus,
) {
    let (_, disposition, _) = classify_release(status, Some(raw));
    enqueue_cleanup(PendingCleanup::ElementwisePlan {
        raw: Some(raw),
        context: Arc::clone(&context.inner),
        descriptor: Box::new(descriptor),
        disposition,
    });
}

pub(crate) fn release_embedding_plan_once(
    raw: NonNull<sys::sllm_embedding_plan_t>,
) -> (RuntimeStatus, Option<NonNull<sys::sllm_embedding_plan_t>>) {
    let mut error_buffer = [0_u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut error_buffer);
    let mut native = raw.as_ptr();
    let status = unsafe { sys::sllm_embedding_plan_release(&mut native, &mut error_sink) };
    (RuntimeStatus::from_raw(status), NonNull::new(native))
}

pub(crate) fn enqueue_embedding_cleanup(
    raw: NonNull<sys::sllm_embedding_plan_t>,
    context: Context,
    descriptor: crate::embedding::EmbeddingDescriptor,
    status: RuntimeStatus,
) {
    let (_, disposition, _) = classify_release(status, Some(raw));
    enqueue_cleanup(PendingCleanup::EmbeddingPlan {
        raw: Some(raw),
        context: Arc::clone(&context.inner),
        descriptor: Box::new(descriptor),
        disposition,
    });
}

pub(crate) fn release_matmul_plan_once(
    raw: NonNull<sys::sllm_matmul_plan_t>,
) -> (RuntimeStatus, Option<NonNull<sys::sllm_matmul_plan_t>>) {
    #[cfg(test)]
    if let Some((status, consumed)) = FORCED_MATMUL_PLAN_RELEASE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
    {
        return (status, if consumed { None } else { Some(raw) });
    }

    let mut error_buffer = [0_u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut error_buffer);
    let mut native = raw.as_ptr();
    let status = unsafe { sys::sllm_matmul_plan_release(&mut native, &mut error_sink) };
    (RuntimeStatus::from_raw(status), NonNull::new(native))
}

pub(crate) fn enqueue_matmul_cleanup(
    raw: NonNull<sys::sllm_matmul_plan_t>,
    context: Context,
    descriptor: crate::matmul::MatmulDescriptor,
    status: RuntimeStatus,
) {
    let (_, disposition, _) = classify_release(status, Some(raw));
    enqueue_cleanup(PendingCleanup::MatmulPlan {
        raw: Some(raw),
        context: Arc::clone(&context.inner),
        descriptor: Box::new(descriptor),
        disposition,
    });
}

pub(crate) fn release_attention_preprocess_plan_once(
    raw: NonNull<sys::sllm_attention_preprocess_plan_t>,
) -> (
    RuntimeStatus,
    Option<NonNull<sys::sllm_attention_preprocess_plan_t>>,
) {
    #[cfg(test)]
    if let Some((status, consumed)) = FORCED_ATTENTION_PREPROCESS_PLAN_RELEASE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
    {
        return (status, if consumed { None } else { Some(raw) });
    }

    let mut error_buffer = [0_u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut error_buffer);
    let mut native = raw.as_ptr();
    let status =
        unsafe { sys::sllm_attention_preprocess_plan_release(&mut native, &mut error_sink) };
    (RuntimeStatus::from_raw(status), NonNull::new(native))
}

pub(crate) fn enqueue_attention_preprocess_cleanup(
    raw: NonNull<sys::sllm_attention_preprocess_plan_t>,
    context: Context,
    descriptor: crate::attention_preprocess::AttentionPreprocessDescriptor,
    status: RuntimeStatus,
) {
    let (_, disposition, _) = classify_release(status, Some(raw));
    enqueue_cleanup(PendingCleanup::AttentionPreprocessPlan {
        raw: Some(raw),
        context: Arc::clone(&context.inner),
        descriptor: Box::new(descriptor),
        disposition,
    });
}

#[cfg(test)]
pub(crate) fn force_rmsnorm_plan_release_for_test(status: RuntimeStatus, consumed: bool) {
    *FORCED_RMSNORM_PLAN_RELEASE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((status, consumed));
}

#[cfg(test)]
pub(crate) fn clear_forced_rmsnorm_plan_release_for_test() {
    FORCED_RMSNORM_PLAN_RELEASE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
}

#[cfg(test)]
pub(crate) fn force_matmul_plan_release_for_test(status: RuntimeStatus, consumed: bool) {
    *FORCED_MATMUL_PLAN_RELEASE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((status, consumed));
}

#[cfg(test)]
pub(crate) fn clear_forced_matmul_plan_release_for_test() {
    FORCED_MATMUL_PLAN_RELEASE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
}

#[cfg(test)]
pub(crate) fn force_attention_preprocess_plan_release_for_test(
    status: RuntimeStatus,
    consumed: bool,
) {
    *FORCED_ATTENTION_PREPROCESS_PLAN_RELEASE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((status, consumed));
}

#[cfg(test)]
pub(crate) fn clear_forced_attention_preprocess_plan_release_for_test() {
    FORCED_ATTENTION_PREPROCESS_PLAN_RELEASE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
}

impl PendingCleanup {
    fn try_once(self) -> Option<Self> {
        /* The host stub cannot produce a retryable native BUSY result.  This
         * per-thread test seam is used only to keep a live record through the
         * real TLS handoff/borrow machinery; production always reaches the
         * native release/query calls below. */
        if force_retry_once_for_test() {
            return Some(self);
        }
        match self {
            Self::Context { raw, disposition } => {
                let Some(raw_handle) = raw else {
                    return if disposition == CleanupDisposition::Poisoned {
                        Some(Self::Context { raw, disposition })
                    } else {
                        None
                    };
                };
                if disposition == CleanupDisposition::Poisoned {
                    return Some(Self::Context { raw, disposition });
                }
                let (status, remaining) = release_context_once(raw_handle);
                let (remaining, disposition, done) = classify_release(status, remaining);
                if done {
                    None
                } else {
                    Some(Self::Context {
                        raw: remaining,
                        disposition,
                    })
                }
            }
            Self::Queue {
                raw,
                context,
                disposition,
            } => {
                let Some(raw_handle) = raw else {
                    return if disposition == CleanupDisposition::Poisoned {
                        Some(Self::Queue {
                            raw,
                            context,
                            disposition,
                        })
                    } else {
                        None
                    };
                };
                if disposition == CleanupDisposition::Poisoned {
                    return Some(Self::Queue {
                        raw,
                        context,
                        disposition,
                    });
                }
                let (status, remaining) = release_queue_once(raw_handle);
                let (remaining, disposition, done) = classify_release(status, remaining);
                if done {
                    None
                } else {
                    Some(Self::Queue {
                        raw: remaining,
                        context,
                        disposition,
                    })
                }
            }
            Self::Buffer {
                raw,
                context,
                disposition,
            } => {
                let Some(raw_handle) = raw else {
                    return if disposition == CleanupDisposition::Poisoned {
                        Some(Self::Buffer {
                            raw,
                            context,
                            disposition,
                        })
                    } else {
                        None
                    };
                };
                if disposition == CleanupDisposition::Poisoned {
                    return Some(Self::Buffer {
                        raw,
                        context,
                        disposition,
                    });
                }
                let (status, remaining) = release_buffer_once(raw_handle);
                let (remaining, disposition, done) = classify_release(status, remaining);
                if done {
                    None
                } else {
                    Some(Self::Buffer {
                        raw: remaining,
                        context,
                        disposition,
                    })
                }
            }
            Self::Event {
                raw,
                context,
                disposition,
            } => {
                let Some(raw_handle) = raw else {
                    return if disposition == CleanupDisposition::Poisoned {
                        Some(Self::Event {
                            raw,
                            context,
                            disposition,
                        })
                    } else {
                        None
                    };
                };
                if disposition == CleanupDisposition::Poisoned {
                    return Some(Self::Event {
                        raw,
                        context,
                        disposition,
                    });
                }
                let (status, remaining) = release_event_once(raw_handle);
                let (remaining, disposition, done) = classify_release(status, remaining);
                if done {
                    None
                } else {
                    Some(Self::Event {
                        raw: remaining,
                        context,
                        disposition,
                    })
                }
            }
            Self::Completion {
                raw,
                context,
                queue,
                buffer,
                disposition,
            } => {
                let Some(raw_handle) = raw else {
                    return if disposition == CleanupDisposition::Poisoned {
                        Some(Self::Completion {
                            raw,
                            context,
                            queue,
                            buffer,
                            disposition,
                        })
                    } else {
                        None
                    };
                };
                if disposition == CleanupDisposition::Poisoned {
                    return Some(Self::Completion {
                        raw,
                        context,
                        queue,
                        buffer,
                        disposition,
                    });
                }
                let mut error_buffer = [0_u8; ERROR_CAPACITY];
                let mut error_sink = sink(&mut error_buffer);
                let mut result = Completion::result();
                let wait_status = unsafe {
                    sys::sllm_completion_wait(
                        raw_handle.as_ptr(),
                        DROP_WAIT_TIMEOUT_MS,
                        &mut result,
                        &mut error_sink,
                    )
                };
                let status = RuntimeStatus::from_raw(wait_status);
                let state = CompletionState::from_raw(result.state);
                if !matches!(state, Ok(CompletionState::Success)) {
                    let pending = matches!(state, Ok(CompletionState::Pending))
                        && matches!(
                            status,
                            RuntimeStatus::Pending | RuntimeStatus::Timeout | RuntimeStatus::Busy
                        );
                    let disposition = if pending {
                        CleanupDisposition::Recoverable
                    } else {
                        CleanupDisposition::Poisoned
                    };
                    return Some(Self::Completion {
                        raw: Some(raw_handle),
                        context,
                        queue,
                        buffer,
                        disposition,
                    });
                }
                if status != RuntimeStatus::Ok {
                    return Some(Self::Completion {
                        raw: Some(raw_handle),
                        context,
                        queue,
                        buffer,
                        disposition: CleanupDisposition::Poisoned,
                    });
                }
                let (release_status, remaining) = release_completion_once(raw_handle);
                let (remaining, disposition, done) = classify_release(release_status, remaining);
                if done {
                    None
                } else {
                    Some(Self::Completion {
                        raw: remaining,
                        context,
                        queue,
                        buffer,
                        disposition,
                    })
                }
            }
            Self::KvState {
                raw,
                context,
                disposition,
            } => {
                let Some(raw_handle) = raw else {
                    return if disposition == CleanupDisposition::Poisoned {
                        Some(Self::KvState {
                            raw,
                            context,
                            disposition,
                        })
                    } else {
                        None
                    };
                };
                if disposition == CleanupDisposition::Poisoned {
                    return Some(Self::KvState {
                        raw: Some(raw_handle),
                        context,
                        disposition,
                    });
                }
                let (status, remaining) = release_kv_state_once(raw_handle);
                let (remaining, disposition, done) = classify_release(status, remaining);
                if done {
                    None
                } else {
                    Some(Self::KvState {
                        raw: remaining,
                        context,
                        disposition,
                    })
                }
            }
            Self::KvView {
                raw,
                context,
                disposition,
            } => {
                let Some(raw_handle) = raw else {
                    return if disposition == CleanupDisposition::Poisoned {
                        Some(Self::KvView {
                            raw,
                            context,
                            disposition,
                        })
                    } else {
                        None
                    };
                };
                if disposition == CleanupDisposition::Poisoned {
                    return Some(Self::KvView {
                        raw: Some(raw_handle),
                        context,
                        disposition,
                    });
                }
                let (status, remaining) = release_kv_view_once(raw_handle);
                let (remaining, disposition, done) = classify_release(status, remaining);
                if done {
                    None
                } else {
                    Some(Self::KvView {
                        raw: remaining,
                        context,
                        disposition,
                    })
                }
            }
            Self::KvCompletion {
                raw,
                context,
                queue,
                key,
                value,
                state,
                disposition,
            } => {
                let Some(raw_handle) = raw else {
                    return if disposition == CleanupDisposition::Poisoned {
                        Some(Self::KvCompletion {
                            raw,
                            context,
                            queue,
                            key,
                            value,
                            state,
                            disposition,
                        })
                    } else {
                        None
                    };
                };
                if disposition == CleanupDisposition::Poisoned {
                    return Some(Self::KvCompletion {
                        raw: Some(raw_handle),
                        context,
                        queue,
                        key,
                        value,
                        state,
                        disposition,
                    });
                }
                let mut error_buffer = [0_u8; ERROR_CAPACITY];
                let mut error_sink = sink(&mut error_buffer);
                let mut result = Completion::result();
                let wait_status = unsafe {
                    sys::sllm_completion_wait(
                        raw_handle.as_ptr(),
                        DROP_WAIT_TIMEOUT_MS,
                        &mut result,
                        &mut error_sink,
                    )
                };
                let status = RuntimeStatus::from_raw(wait_status);
                let state_result = CompletionState::from_raw(result.state);
                if !matches!(state_result, Ok(CompletionState::Success)) {
                    let pending = matches!(state_result, Ok(CompletionState::Pending))
                        && matches!(
                            status,
                            RuntimeStatus::Pending | RuntimeStatus::Timeout | RuntimeStatus::Busy
                        );
                    return Some(Self::KvCompletion {
                        raw: Some(raw_handle),
                        context,
                        queue,
                        key,
                        value,
                        state,
                        disposition: if pending {
                            CleanupDisposition::Recoverable
                        } else {
                            CleanupDisposition::Poisoned
                        },
                    });
                }
                if status != RuntimeStatus::Ok {
                    return Some(Self::KvCompletion {
                        raw: Some(raw_handle),
                        context,
                        queue,
                        key,
                        value,
                        state,
                        disposition: CleanupDisposition::Poisoned,
                    });
                }
                let (release_status, remaining) = release_kv_completion_once(raw_handle);
                let (remaining, disposition, done) = classify_release(release_status, remaining);
                if done {
                    None
                } else {
                    Some(Self::KvCompletion {
                        raw: remaining,
                        context,
                        queue,
                        key,
                        value,
                        state,
                        disposition,
                    })
                }
            }
            Self::CausalCompletion {
                raw,
                context,
                queue,
                query,
                output,
                state,
                disposition,
            } => {
                let Some(raw_handle) = raw else {
                    return if disposition == CleanupDisposition::Poisoned {
                        Some(Self::CausalCompletion {
                            raw,
                            context,
                            queue,
                            query,
                            output,
                            state,
                            disposition,
                        })
                    } else {
                        None
                    };
                };
                if disposition == CleanupDisposition::Poisoned {
                    return Some(Self::CausalCompletion {
                        raw: Some(raw_handle),
                        context,
                        queue,
                        query,
                        output,
                        state,
                        disposition,
                    });
                }
                let mut error_buffer = [0_u8; ERROR_CAPACITY];
                let mut error_sink = sink(&mut error_buffer);
                let mut result = Completion::result();
                let wait_status = unsafe {
                    sys::sllm_completion_wait(
                        raw_handle.as_ptr(),
                        DROP_WAIT_TIMEOUT_MS,
                        &mut result,
                        &mut error_sink,
                    )
                };
                let status = RuntimeStatus::from_raw(wait_status);
                let state_result = CompletionState::from_raw(result.state);
                if !matches!(state_result, Ok(CompletionState::Success)) {
                    let pending = matches!(state_result, Ok(CompletionState::Pending))
                        && matches!(
                            status,
                            RuntimeStatus::Pending | RuntimeStatus::Timeout | RuntimeStatus::Busy
                        );
                    return Some(Self::CausalCompletion {
                        raw: Some(raw_handle),
                        context,
                        queue,
                        query,
                        output,
                        state,
                        disposition: if pending {
                            CleanupDisposition::Recoverable
                        } else {
                            CleanupDisposition::Poisoned
                        },
                    });
                }
                if status != RuntimeStatus::Ok {
                    return Some(Self::CausalCompletion {
                        raw: Some(raw_handle),
                        context,
                        queue,
                        query,
                        output,
                        state,
                        disposition: CleanupDisposition::Poisoned,
                    });
                }
                let (release_status, remaining) = release_causal_completion_once(raw_handle);
                let (remaining, disposition, done) = classify_release(release_status, remaining);
                if done {
                    None
                } else {
                    Some(Self::CausalCompletion {
                        raw: remaining,
                        context,
                        queue,
                        query,
                        output,
                        state,
                        disposition,
                    })
                }
            }
            Self::RmsNormPlan {
                raw,
                context,
                descriptor,
                disposition,
            } => {
                let Some(raw_handle) = raw else {
                    return if disposition == CleanupDisposition::Poisoned {
                        Some(Self::RmsNormPlan {
                            raw,
                            context,
                            descriptor,
                            disposition,
                        })
                    } else {
                        None
                    };
                };
                if disposition == CleanupDisposition::Poisoned {
                    return Some(Self::RmsNormPlan {
                        raw: Some(raw_handle),
                        context,
                        descriptor,
                        disposition,
                    });
                }
                let (status, remaining) = release_rmsnorm_plan_once(raw_handle);
                /* Native plan release consumes the caller token when it
                 * returns a recognized ownership-ambiguous failure and
                 * transfers the complete plan to its durable quarantine. */
                let remaining = remaining?;
                let (remaining, disposition, done) = classify_release(status, Some(remaining));
                if done {
                    None
                } else {
                    Some(Self::RmsNormPlan {
                        raw: remaining,
                        context,
                        descriptor,
                        disposition,
                    })
                }
            }
            Self::ElementwisePlan {
                raw,
                context,
                descriptor,
                disposition,
            } => {
                let Some(raw_handle) = raw else {
                    return if disposition == CleanupDisposition::Poisoned {
                        Some(Self::ElementwisePlan {
                            raw,
                            context,
                            descriptor,
                            disposition,
                        })
                    } else {
                        None
                    };
                };
                if disposition == CleanupDisposition::Poisoned {
                    return Some(Self::ElementwisePlan {
                        raw: Some(raw_handle),
                        context,
                        descriptor,
                        disposition,
                    });
                }
                let (status, remaining) = release_elementwise_plan_once(raw_handle);
                let remaining = remaining?;
                let (remaining, disposition, done) = classify_release(status, Some(remaining));
                if done {
                    None
                } else {
                    Some(Self::ElementwisePlan {
                        raw: remaining,
                        context,
                        descriptor,
                        disposition,
                    })
                }
            }
            Self::EmbeddingPlan {
                raw,
                context,
                descriptor,
                disposition,
            } => {
                let Some(raw_handle) = raw else {
                    return if disposition == CleanupDisposition::Poisoned {
                        Some(Self::EmbeddingPlan {
                            raw,
                            context,
                            descriptor,
                            disposition,
                        })
                    } else {
                        None
                    };
                };
                if disposition == CleanupDisposition::Poisoned {
                    return Some(Self::EmbeddingPlan {
                        raw: Some(raw_handle),
                        context,
                        descriptor,
                        disposition,
                    });
                }
                let (status, remaining) = release_embedding_plan_once(raw_handle);
                let remaining = remaining?;
                let (remaining, disposition, done) = classify_release(status, Some(remaining));
                if done {
                    None
                } else {
                    Some(Self::EmbeddingPlan {
                        raw: remaining,
                        context,
                        descriptor,
                        disposition,
                    })
                }
            }
            Self::MatmulPlan {
                raw,
                context,
                descriptor,
                disposition,
            } => {
                let Some(raw_handle) = raw else {
                    return if disposition == CleanupDisposition::Poisoned {
                        Some(Self::MatmulPlan {
                            raw,
                            context,
                            descriptor,
                            disposition,
                        })
                    } else {
                        None
                    };
                };
                if disposition == CleanupDisposition::Poisoned {
                    return Some(Self::MatmulPlan {
                        raw: Some(raw_handle),
                        context,
                        descriptor,
                        disposition,
                    });
                }
                let (status, remaining) = release_matmul_plan_once(raw_handle);
                let remaining = remaining?;
                let (remaining, disposition, done) = classify_release(status, Some(remaining));
                if done {
                    None
                } else {
                    Some(Self::MatmulPlan {
                        raw: remaining,
                        context,
                        descriptor,
                        disposition,
                    })
                }
            }
            Self::AttentionPreprocessPlan {
                raw,
                context,
                descriptor,
                disposition,
            } => {
                let Some(raw_handle) = raw else {
                    return if disposition == CleanupDisposition::Poisoned {
                        Some(Self::AttentionPreprocessPlan {
                            raw,
                            context,
                            descriptor,
                            disposition,
                        })
                    } else {
                        None
                    };
                };
                if disposition == CleanupDisposition::Poisoned {
                    return Some(Self::AttentionPreprocessPlan {
                        raw: Some(raw_handle),
                        context,
                        descriptor,
                        disposition,
                    });
                }
                let (status, remaining) = release_attention_preprocess_plan_once(raw_handle);
                let remaining = remaining?;
                let (remaining, disposition, done) = classify_release(status, Some(remaining));
                if done {
                    None
                } else {
                    Some(Self::AttentionPreprocessPlan {
                        raw: remaining,
                        context,
                        descriptor,
                        disposition,
                    })
                }
            }
        }
    }
}

fn reap_pending_cleanup() {
    let pending = CLEANUP_REAPER
        .try_with(|reaper| {
            reaper
                .try_borrow_mut()
                .ok()
                .map(|mut reaper| reaper.take_recoverable())
        })
        .unwrap_or(None);
    let Some(pending) = pending else {
        /* AccessError means the TLS key is being destroyed or already has
         * been destroyed.  A borrow failure means another cleanup operation
         * is active.  In both cases ownership stays where it is. */
        return;
    };
    pause_after_reap_take_for_test();
    let mut retained = Vec::new();
    for record in pending {
        match record.try_once() {
            CleanupAttempt::Complete => finish_pending_cleanup(),
            CleanupAttempt::Retry(record) => {
                if record.cleanup.is_poisoned() {
                    quarantine_pending_record(record);
                } else if retained.try_reserve(1).is_ok() {
                    retained.push(record);
                } else {
                    quarantine_pending_record(record);
                }
            }
        }
    }
    for record in retained {
        requeue_cleanup(record);
    }
}

struct ContextInner {
    raw: Option<NonNull<sys::sllm_context_t>>,
    expected_target: Option<Arc<str>>,
    #[cfg(test)]
    drop_probe: Option<Arc<AtomicUsize>>,
}

// SAFETY: opaque tokens are never dereferenced in Rust.  Native public
// runtime operations serialize registry/accounting access; final release is
// reached only after the last Arc reference, and Completion mutation requires
// an exclusive `&mut self`.
unsafe impl Send for ContextInner {}
// SAFETY: see the Send rationale above.
unsafe impl Sync for ContextInner {}

impl ContextInner {
    #[cfg(test)]
    fn new(raw: Option<NonNull<sys::sllm_context_t>>) -> Self {
        Self {
            raw,
            expected_target: None,
            #[cfg(test)]
            drop_probe: None,
        }
    }

    fn new_with_target(raw: NonNull<sys::sllm_context_t>, target: &str) -> Self {
        Self {
            raw: Some(raw),
            expected_target: Some(Arc::from(target)),
            #[cfg(test)]
            drop_probe: None,
        }
    }

    #[cfg(test)]
    fn new_with_drop_probe(
        raw: Option<NonNull<sys::sllm_context_t>>,
        drop_probe: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            raw,
            expected_target: None,
            drop_probe: Some(drop_probe),
        }
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        #[cfg(test)]
        if let Some(drop_probe) = self.drop_probe.take() {
            drop_probe.fetch_add(1, Ordering::Relaxed);
        }
        let Some(raw) = self.raw.take() else {
            return;
        };
        let (status, remaining) = release_context_once(raw);
        let (remaining, disposition, done) = classify_release(status, remaining);
        if !done {
            enqueue_cleanup(PendingCleanup::Context {
                raw: remaining,
                disposition,
            });
        }
    }
}

#[derive(Clone)]
pub struct Context {
    inner: Arc<ContextInner>,
}

impl fmt::Debug for Context {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Context").finish_non_exhaustive()
    }
}

impl Context {
    pub(crate) fn raw_handle(&self) -> Result<NonNull<sys::sllm_context_t>, RuntimeError> {
        self.inner.raw.ok_or_else(|| {
            RuntimeError::local(RuntimeStatus::InvalidHandle, "context was already released")
        })
    }

    pub(crate) fn expected_target(&self) -> Option<&str> {
        self.inner.expected_target.as_deref()
    }

    #[cfg(test)]
    pub(crate) fn test_without_native() -> Self {
        Self {
            inner: Arc::new(ContextInner::new(None)),
        }
    }
    pub fn device_count() -> Result<u32, RuntimeError> {
        reap_pending_cleanup();
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut count = 0_u32;
        let raw = unsafe { sys::sllm_device_count(&mut count, &mut error_sink) };
        ensure_ok(raw, &error_buffer, error_sink.message_length).map(|()| count)
    }

    pub fn query_device(device_index: u32) -> Result<DeviceInfo, RuntimeError> {
        reap_pending_cleanup();
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut info = sys::sllm_device_info_t {
            struct_size: size_of::<sys::sllm_device_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            device_index: 0,
            visible_device_count: 0,
            total_memory_bytes: 0,
            wavefront_size: 0,
            reserved0: 0,
            name: [0; 128],
            gcn_arch_name: [0; 64],
            reserved: [0; 4],
        };
        let raw = unsafe { sys::sllm_device_query(device_index, &mut info, &mut error_sink) };
        ensure_ok(raw, &error_buffer, error_sink.message_length)
            .map(|()| device_info_from_raw(&info))
    }

    pub fn create(device_index: u32, expected_gcn_arch_name: &str) -> Result<Self, RuntimeError> {
        reap_pending_cleanup();
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut info = sys::sllm_context_create_info_t {
            struct_size: size_of::<sys::sllm_context_create_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            device_index,
            flags: 0,
            expected_gcn_arch_name: [0; 64],
            reserved: [0; 4],
        };
        copy_c_string(&mut info.expected_gcn_arch_name, expected_gcn_arch_name)?;
        let mut raw_context = std::ptr::null_mut();
        let raw = unsafe { sys::sllm_context_create(&info, &mut raw_context, &mut error_sink) };
        ensure_ok(raw, &error_buffer, error_sink.message_length)?;
        let raw = NonNull::new(raw_context).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native context create returned a null handle on success",
            )
        })?;
        Ok(Self {
            inner: Arc::new(ContextInner::new_with_target(raw, expected_gcn_arch_name)),
        })
    }

    pub fn reap_cleanup() {
        reap_pending_cleanup();
    }

    /// Return `(retryable_pending, durable_quarantine)` cleanup counts.
    /// Durable quarantine is intentionally retained for process lifetime.
    pub fn cleanup_counts() -> (usize, usize) {
        (
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
        )
    }

    pub fn pending_cleanup_count() -> usize {
        Self::cleanup_counts().0
    }

    pub fn durable_quarantine_count() -> usize {
        Self::cleanup_counts().1
    }

    /// Report whether at least one retained cleanup could not be represented
    /// by the saturated exact durable count.
    pub fn durable_quarantine_overflowed() -> bool {
        DURABLE_QUARANTINE_OVERFLOW.load(Ordering::Acquire) != 0
    }

    pub fn cleanup_accounting_error_count() -> usize {
        CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire)
    }

    pub fn checked_cleanup_counts() -> Result<(usize, usize), RuntimeError> {
        if Self::durable_quarantine_overflowed() {
            return Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "cleanup durable quarantine count overflowed",
            ));
        }
        if Self::cleanup_accounting_error_count() != 0 {
            return Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "cleanup accounting is inconsistent",
            ));
        }
        Ok(Self::cleanup_counts())
    }

    /// Make at most `max_attempts` nonblocking cleanup passes on this thread.
    pub fn drain_cleanup(max_attempts: usize) -> Result<(usize, usize), RuntimeError> {
        let reap_budget = max_attempts.min(CLEANUP_CAS_BOUND);
        for _ in 0..reap_budget {
            reap_pending_cleanup();
            if PENDING_CLEANUP_ITEMS.load(Ordering::Acquire) == 0 {
                break;
            }
        }
        let _drain = enter_cleanup_drain(max_attempts)?;
        let mut counts = None;
        let validation_budget = max_attempts.saturating_add(1).min(CLEANUP_CAS_BOUND);
        for attempt in 0..validation_budget {
            let state = CLEANUP_HANDOFF_STATE.load(Ordering::Acquire);
            if !cleanup_handoff_state_is_valid(state) {
                record_cleanup_accounting_error();
                break;
            }
            if state != CLEANUP_DRAIN_BIT {
                if attempt + 1 == validation_budget {
                    break;
                }
                std::thread::yield_now();
                continue;
            }
            let epoch = CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire);
            let snapshot = Self::cleanup_counts();
            pause_after_cleanup_snapshot_for_test();
            /* The successful no-op CAS is the snapshot linearization point:
             * it proves that the drain barrier was still the sampled state at
             * its RMW point.  The final acquire epoch load is only an ABA
             * validation.  A handoff accepted before the CAS and still active
             * makes the CAS fail; a completed handoff before or after the CAS
             * changes the epoch and rejects the snapshot.  A handoff accepted
             * after the successful CAS may coexist with Ok if it remains
             * incomplete through the final validation; exit_cleanup_drain
             * then preserves its exact state ticket. */
            let state_stable = CLEANUP_HANDOFF_STATE
                .compare_exchange(state, state, Ordering::AcqRel, Ordering::Acquire)
                .is_ok();
            if state_stable {
                pause_after_cleanup_noop_cas_for_test();
            }
            let epoch_stable = CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire) == epoch;
            if state_stable && epoch_stable {
                counts = Some(snapshot);
                break;
            }
            if attempt + 1 != validation_budget {
                std::thread::yield_now();
            }
        }
        drop(_drain);
        if Self::durable_quarantine_overflowed() {
            return Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "cleanup durable quarantine count overflowed",
            ));
        }
        if Self::cleanup_accounting_error_count() != 0 {
            return Err(RuntimeError::local(
                RuntimeStatus::InternalError,
                "cleanup accounting is inconsistent",
            ));
        }
        counts.ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::Busy,
                "cleanup counts did not stabilize within the drain bound",
            )
        })
    }

    /// Drain retryable cleanup and fail closed if any retryable entry remains.
    pub fn shutdown_cleanup(max_attempts: usize) -> Result<(usize, usize), RuntimeError> {
        let counts = Self::drain_cleanup(max_attempts)?;
        if counts.0 != 0 {
            Err(RuntimeError::local(
                RuntimeStatus::Busy,
                "cleanup shutdown incomplete: retryable entries remain",
            ))
        } else {
            Ok(counts)
        }
    }
}

struct QueueInner {
    raw: Option<NonNull<sys::sllm_queue_t>>,
    context: Arc<ContextInner>,
}

// SAFETY: QueueInner contains immutable opaque native identity plus an Arc
// context; native lifetime transitions are synchronized by the C ABI.
unsafe impl Send for QueueInner {}
// SAFETY: see the Send rationale above.
unsafe impl Sync for QueueInner {}

impl Drop for QueueInner {
    fn drop(&mut self) {
        let Some(raw) = self.raw.take() else {
            return;
        };
        let (status, remaining) = release_queue_once(raw);
        let (remaining, disposition, done) = classify_release(status, remaining);
        if !done {
            enqueue_cleanup(PendingCleanup::Queue {
                raw: remaining,
                context: Arc::clone(&self.context),
                disposition,
            });
        }
    }
}

#[derive(Clone)]
pub struct Queue {
    inner: Arc<QueueInner>,
}

impl fmt::Debug for Queue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Queue").finish_non_exhaustive()
    }
}

impl Queue {
    pub(crate) fn raw_handle(&self) -> Result<NonNull<sys::sllm_queue_t>, RuntimeError> {
        self.inner.raw.ok_or_else(|| {
            RuntimeError::local(RuntimeStatus::InvalidHandle, "queue was already released")
        })
    }

    pub fn create(context: &Context) -> Result<Self, RuntimeError> {
        reap_pending_cleanup();
        let context_raw = context.inner.raw.ok_or_else(|| {
            RuntimeError::local(RuntimeStatus::InvalidHandle, "context was already released")
        })?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let info = sys::sllm_queue_create_info_t {
            struct_size: size_of::<sys::sllm_queue_create_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            flags: 0,
            reserved: [0; 5],
        };
        let mut raw_queue = std::ptr::null_mut();
        let raw = unsafe {
            sys::sllm_queue_create(context_raw.as_ptr(), &info, &mut raw_queue, &mut error_sink)
        };
        ensure_ok(raw, &error_buffer, error_sink.message_length)?;
        let raw_queue = NonNull::new(raw_queue).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native queue create returned a null handle on success",
            )
        })?;
        Ok(Self {
            inner: Arc::new(QueueInner {
                raw: Some(raw_queue),
                context: Arc::clone(&context.inner),
            }),
        })
    }

    pub fn copy_to_device(
        &self,
        buffer: &Buffer,
        data: &[u8],
        buffer_offset_bytes: u64,
    ) -> Result<Completion, RuntimeError> {
        submit_copy(
            self,
            buffer,
            data.as_ptr().cast_mut().cast(),
            data.len(),
            buffer_offset_bytes,
            false,
        )
    }

    pub fn copy_to_host(
        &self,
        buffer: &Buffer,
        size_bytes: u64,
        buffer_offset_bytes: u64,
    ) -> Result<Completion, RuntimeError> {
        let size_bytes = usize::try_from(size_bytes).map_err(|_| {
            RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                "copy size does not fit the host usize",
            )
        })?;
        submit_copy(
            self,
            buffer,
            std::ptr::null_mut(),
            size_bytes,
            buffer_offset_bytes,
            true,
        )
    }
}

struct BufferInner {
    raw: Option<NonNull<sys::sllm_buffer_t>>,
    _context: Arc<ContextInner>,
}

// SAFETY: BufferInner owns no dereferenceable Rust pointer.  Its final native
// release is serialized by Arc uniqueness and the native registry.
unsafe impl Send for BufferInner {}
// SAFETY: see the Send rationale above.
unsafe impl Sync for BufferInner {}

impl Drop for BufferInner {
    fn drop(&mut self) {
        let Some(raw) = self.raw.take() else {
            return;
        };
        let (status, remaining) = release_buffer_once(raw);
        let (remaining, disposition, done) = classify_release(status, remaining);
        if !done {
            enqueue_cleanup(PendingCleanup::Buffer {
                raw: remaining,
                context: Arc::clone(&self._context),
                disposition,
            });
        }
    }
}

#[derive(Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
}

impl fmt::Debug for Buffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Buffer").finish_non_exhaustive()
    }
}

impl Buffer {
    pub(crate) fn raw_handle(&self) -> Result<NonNull<sys::sllm_buffer_t>, RuntimeError> {
        self.inner.raw.ok_or_else(|| {
            RuntimeError::local(RuntimeStatus::InvalidHandle, "buffer was already released")
        })
    }

    #[cfg(test)]
    pub(crate) fn test_without_native(context: &Context) -> Self {
        Self {
            inner: Arc::new(BufferInner {
                raw: None,
                _context: Arc::clone(&context.inner),
            }),
        }
    }

    /// Bind a semantic tensor view to this owned opaque buffer.  The returned
    /// binding retains the buffer identity; callers cannot fabricate one.
    pub fn binding(&self, view: sllm_core::TensorView) -> crate::rmsnorm::TensorBinding {
        crate::rmsnorm::TensorBinding::from_buffer(self.clone(), view)
    }
    pub fn allocate(context: &Context, size_bytes: u64) -> Result<Self, RuntimeError> {
        reap_pending_cleanup();
        if size_bytes == 0 || size_bytes > sys::SLLM_HIP_MAX_TRANSFER_BYTES * 16 {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                "buffer size is outside the bounded public runtime range",
            ));
        }
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let context_raw = context.inner.raw.ok_or_else(|| {
            RuntimeError::local(RuntimeStatus::InvalidHandle, "context was already released")
        })?;
        let info = sys::sllm_buffer_create_info_t {
            struct_size: size_of::<sys::sllm_buffer_create_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            size_bytes,
            alignment_bytes: 0,
            flags: 0,
            reserved: [0; 5],
        };
        let mut raw_buffer = std::ptr::null_mut();
        let raw = unsafe {
            sys::sllm_buffer_create(
                context_raw.as_ptr(),
                &info,
                &mut raw_buffer,
                &mut error_sink,
            )
        };
        ensure_ok(raw, &error_buffer, error_sink.message_length)?;
        let raw_buffer = NonNull::new(raw_buffer).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native buffer create returned a null handle on success",
            )
        })?;
        Ok(Self {
            inner: Arc::new(BufferInner {
                raw: Some(raw_buffer),
                _context: Arc::clone(&context.inner),
            }),
        })
    }

    pub fn size_bytes(&self) -> Result<u64, RuntimeError> {
        reap_pending_cleanup();
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut size_bytes = 0_u64;
        let raw = unsafe {
            sys::sllm_buffer_size(
                self.inner
                    .raw
                    .ok_or_else(|| {
                        RuntimeError::local(
                            RuntimeStatus::InvalidHandle,
                            "buffer was already released",
                        )
                    })?
                    .as_ptr(),
                &mut size_bytes,
                &mut error_sink,
            )
        };
        ensure_ok(raw, &error_buffer, error_sink.message_length).map(|()| size_bytes)
    }
}

struct EventInner {
    raw: Option<NonNull<sys::sllm_event_t>>,
    _context: Arc<ContextInner>,
}

// SAFETY: EventInner follows the same opaque-token and native synchronization
// contract as BufferInner.
unsafe impl Send for EventInner {}
// SAFETY: see the Send rationale above.
unsafe impl Sync for EventInner {}

impl Drop for EventInner {
    fn drop(&mut self) {
        let Some(raw) = self.raw.take() else {
            return;
        };
        let (status, remaining) = release_event_once(raw);
        let (remaining, disposition, done) = classify_release(status, remaining);
        if !done {
            enqueue_cleanup(PendingCleanup::Event {
                raw: remaining,
                context: Arc::clone(&self._context),
                disposition,
            });
        }
    }
}

#[derive(Clone)]
pub struct Event {
    _inner: Arc<EventInner>,
}

impl fmt::Debug for Event {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Event").finish_non_exhaustive()
    }
}

impl Event {
    pub fn create(context: &Context) -> Result<Self, RuntimeError> {
        reap_pending_cleanup();
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let context_raw = context.inner.raw.ok_or_else(|| {
            RuntimeError::local(RuntimeStatus::InvalidHandle, "context was already released")
        })?;
        let mut raw_event = std::ptr::null_mut();
        let raw = unsafe {
            sys::sllm_event_create(context_raw.as_ptr(), &mut raw_event, &mut error_sink)
        };
        ensure_ok(raw, &error_buffer, error_sink.message_length)?;
        let raw_event = NonNull::new(raw_event).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native event create returned a null handle on success",
            )
        })?;
        Ok(Self {
            _inner: Arc::new(EventInner {
                raw: Some(raw_event),
                _context: Arc::clone(&context.inner),
            }),
        })
    }
}

pub struct Completion {
    raw: Option<NonNull<sys::sllm_completion_t>>,
    _context: Arc<ContextInner>,
    _queue: Arc<QueueInner>,
    _buffer: Arc<BufferInner>,
    transfer_size_bytes: u64,
    d2h: bool,
    terminal: bool,
    safe_to_release: bool,
}

// SAFETY: Completion's public state-changing methods require `&mut self`.
// Its opaque native handle and retained Arc graph are released only by Drop;
// native operations protect shared lifetime/accounting state.
unsafe impl Send for Completion {}
// SAFETY: shared access cannot mutate Completion's Rust fields.
unsafe impl Sync for Completion {}

impl fmt::Debug for Completion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Completion")
            .field("terminal", &self.terminal)
            .field("transfer_size_bytes", &self.transfer_size_bytes)
            .finish_non_exhaustive()
    }
}

impl Completion {
    pub(crate) fn from_native(
        raw: NonNull<sys::sllm_completion_t>,
        context: &Context,
        queue: &Queue,
        buffer: &Buffer,
        transfer_size_bytes: u64,
        d2h: bool,
    ) -> Self {
        Self {
            raw: Some(raw),
            _context: Arc::clone(&context.inner),
            _queue: Arc::clone(&queue.inner),
            _buffer: Arc::clone(&buffer.inner),
            transfer_size_bytes,
            d2h,
            terminal: false,
            safe_to_release: false,
        }
    }

    fn result() -> sys::sllm_completion_result_t {
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

    fn update_state(
        &mut self,
        result: &sys::sllm_completion_result_t,
    ) -> Result<CompletionState, RuntimeError> {
        let state = CompletionState::from_raw(result.state)?;
        self.terminal = !matches!(state, CompletionState::Pending);
        self.safe_to_release = matches!(state, CompletionState::Success);
        Ok(state)
    }

    pub fn query(&mut self) -> Result<CompletionState, RuntimeError> {
        reap_pending_cleanup();
        let raw_completion = self.raw.ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "completion was already released",
            )
        })?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut result = Self::result();
        let raw = unsafe {
            sys::sllm_completion_query(raw_completion.as_ptr(), &mut result, &mut error_sink)
        };
        let state = self.update_state(&result)?;
        let status = RuntimeStatus::from_raw(raw);
        if status == RuntimeStatus::Ok || status == RuntimeStatus::Pending {
            if status == RuntimeStatus::Ok && state == CompletionState::Success {
                self.safe_to_release = true;
            }
            Ok(state)
        } else {
            Err(result_error(raw, &error_buffer, error_sink.message_length))
        }
    }

    pub fn wait(&mut self, timeout: Duration) -> Result<CompletionState, RuntimeError> {
        reap_pending_cleanup();
        let raw_completion = self.raw.ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "completion was already released",
            )
        })?;
        let timeout_ms = timeout_millis(timeout);
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut result = Self::result();
        let raw = unsafe {
            sys::sllm_completion_wait(
                raw_completion.as_ptr(),
                timeout_ms,
                &mut result,
                &mut error_sink,
            )
        };
        let state = self.update_state(&result)?;
        let status = RuntimeStatus::from_raw(raw);
        if status == RuntimeStatus::Ok {
            if state == CompletionState::Success {
                self.safe_to_release = true;
            }
            Ok(state)
        } else {
            Err(result_error(raw, &error_buffer, error_sink.message_length))
        }
    }

    pub fn read_into(&mut self, destination: &mut [u8]) -> Result<u64, RuntimeError> {
        reap_pending_cleanup();
        if !self.terminal {
            return Err(RuntimeError::local(
                RuntimeStatus::NotReady,
                "completion must be terminal before reading output",
            ));
        }
        if !self.d2h {
            return Err(RuntimeError::local(
                RuntimeStatus::Unsupported,
                "H2D completion has no host output",
            ));
        }
        let raw_completion = self.raw.ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "completion was already released",
            )
        })?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut bytes_written = 0_u64;
        let capacity = match u64::try_from(destination.len()) {
            Ok(value) => value,
            Err(_) => {
                return Err(RuntimeError::local(
                    RuntimeStatus::InvalidArgument,
                    "destination length does not fit the public ABI",
                ));
            }
        };
        let raw = unsafe {
            sys::sllm_completion_read(
                raw_completion.as_ptr(),
                destination.as_mut_ptr().cast::<c_void>(),
                capacity,
                &mut bytes_written,
                &mut error_sink,
            )
        };
        ensure_ok(raw, &error_buffer, error_sink.message_length).map(|()| bytes_written)
    }

    pub fn kernel_elapsed_ns(&mut self) -> Result<u64, RuntimeError> {
        reap_pending_cleanup();
        let raw_completion = self.raw.ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "completion was already released",
            )
        })?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut timing = sys::sllm_completion_timing_t {
            struct_size: size_of::<sys::sllm_completion_timing_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            valid: 0,
            reserved0: 0,
            elapsed_ns: 0,
            reserved: [0; 4],
        };
        let raw = unsafe {
            sys::sllm_completion_timing(raw_completion.as_ptr(), &mut timing, &mut error_sink)
        };
        ensure_ok(raw, &error_buffer, error_sink.message_length)?;
        if timing.valid != 1 || timing.elapsed_ns == 0 {
            return Err(RuntimeError::local(
                RuntimeStatus::HipRuntimeError,
                "public completion timing did not return a positive elapsed time",
            ));
        }
        Ok(timing.elapsed_ns)
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        let Some(raw) = self.raw.take() else {
            return;
        };
        let cleanup = PendingCleanup::Completion {
            raw: Some(raw),
            context: Arc::clone(&self._context),
            queue: Arc::clone(&self._queue),
            buffer: Arc::clone(&self._buffer),
            disposition: CleanupDisposition::Recoverable,
        };
        if let Some(cleanup) = cleanup.try_once() {
            enqueue_cleanup(cleanup);
        }
    }
}

fn submit_copy(
    queue: &Queue,
    buffer: &Buffer,
    host_pointer: *mut c_void,
    size_bytes: usize,
    buffer_offset_bytes: u64,
    d2h: bool,
) -> Result<Completion, RuntimeError> {
    reap_pending_cleanup();
    if size_bytes == 0 {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidArgument,
            "copy size must be non-zero",
        ));
    }
    let size_bytes_u64 = match u64::try_from(size_bytes) {
        Ok(value) => value,
        Err(_) => {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                "copy size does not fit the public ABI",
            ));
        }
    };
    if size_bytes_u64 > sys::SLLM_HIP_MAX_TRANSFER_BYTES {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidArgument,
            "copy size exceeds the public bounded transfer limit",
        ));
    }
    let mut error_buffer = [0_u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut error_buffer);
    let transfer = sys::sllm_transfer_desc_t {
        struct_size: size_of::<sys::sllm_transfer_desc_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        host_pointer,
        buffer_offset_bytes,
        size_bytes: size_bytes_u64,
        reserved: [0; 4],
    };
    let queue_raw = queue.inner.raw.ok_or_else(|| {
        RuntimeError::local(RuntimeStatus::InvalidHandle, "queue was already released")
    })?;
    let buffer_raw = buffer.inner.raw.ok_or_else(|| {
        RuntimeError::local(RuntimeStatus::InvalidHandle, "buffer was already released")
    })?;
    let mut raw_completion = std::ptr::null_mut();
    let raw = unsafe {
        if d2h {
            sys::sllm_buffer_copy_d2h(
                queue_raw.as_ptr(),
                buffer_raw.as_ptr(),
                &transfer,
                &mut raw_completion,
                &mut error_sink,
            )
        } else {
            sys::sllm_buffer_copy_h2d(
                queue_raw.as_ptr(),
                buffer_raw.as_ptr(),
                &transfer,
                &mut raw_completion,
                &mut error_sink,
            )
        }
    };
    ensure_ok(raw, &error_buffer, error_sink.message_length)?;
    let raw_completion = NonNull::new(raw_completion).ok_or_else(|| {
        RuntimeError::local(
            RuntimeStatus::InternalError,
            "native copy returned a null completion on success",
        )
    })?;
    Ok(Completion {
        raw: Some(raw_completion),
        _context: Arc::clone(&queue.inner.context),
        _queue: Arc::clone(&queue.inner),
        _buffer: Arc::clone(&buffer.inner),
        transfer_size_bytes: size_bytes_u64,
        d2h,
        terminal: false,
        safe_to_release: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    static_assertions::assert_impl_all!(Context: Send, Sync);
    static_assertions::assert_impl_all!(Queue: Send, Sync);
    static_assertions::assert_impl_all!(Buffer: Send, Sync);
    static_assertions::assert_impl_all!(Event: Send, Sync);
    static_assertions::assert_impl_all!(Completion: Send, Sync);

    fn raw_sink(buffer: &mut [u8]) -> sys::sllm_error_sink_t {
        sys::sllm_error_sink_t {
            struct_size: size_of::<sys::sllm_error_sink_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            message: buffer.as_mut_ptr().cast(),
            message_capacity: buffer.len() as u64,
            message_length: 0,
            reserved: [0; 2],
        }
    }

    fn context_info() -> sys::sllm_context_create_info_t {
        let mut info = sys::sllm_context_create_info_t {
            struct_size: size_of::<sys::sllm_context_create_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            device_index: 0,
            flags: 0,
            expected_gcn_arch_name: [0; 64],
            reserved: [0; 4],
        };
        info.expected_gcn_arch_name[0] = b'g' as c_char;
        info.expected_gcn_arch_name[1] = b'f' as c_char;
        info.expected_gcn_arch_name[2] = b'x' as c_char;
        info.expected_gcn_arch_name[3] = b'1' as c_char;
        info.expected_gcn_arch_name[4] = b'2' as c_char;
        info.expected_gcn_arch_name[5] = b'0' as c_char;
        info.expected_gcn_arch_name[6] = b'1' as c_char;
        info
    }

    #[test]
    fn public_status_range_does_not_use_private_evidence_constants() {
        assert_eq!(RuntimeStatus::Pending.raw(), 0x100);
        assert_eq!(RuntimeStatus::Timeout.raw(), 0x101);
        assert_eq!(RuntimeStatus::InvalidHandle.raw(), 0x102);
    }

    #[test]
    fn finite_timeout_never_uses_the_infinite_sentinel() {
        assert_eq!(timeout_millis(Duration::from_millis(0)), 0);
        assert_eq!(
            timeout_millis(Duration::from_millis(u64::from(u32::MAX - 1))),
            u32::MAX - 1
        );
        assert_ne!(
            timeout_millis(Duration::from_millis(u64::from(u32::MAX))),
            u32::MAX
        );
        assert_ne!(timeout_millis(Duration::MAX), u32::MAX);
    }

    #[test]
    fn drop_reaper_keeps_ownership_across_three_busy_results() {
        let raw = NonNull::<sys::sllm_queue_t>::dangling();
        let mut remaining = Some(raw);
        for _ in 0..3 {
            let (next, disposition, done) = classify_release(RuntimeStatus::Busy, remaining);
            assert_eq!(disposition, CleanupDisposition::Recoverable);
            assert!(!done);
            remaining = next;
            assert!(remaining.is_some());
        }
        let (remaining, disposition, done) =
            classify_release::<sys::sllm_queue_t>(RuntimeStatus::Ok, None);
        assert!(done);
        assert_eq!(disposition, CleanupDisposition::Recoverable);
        assert!(remaining.is_none());
    }

    #[test]
    fn fatal_wait_and_release_are_terminal_poison_without_retry_spin() {
        let raw = NonNull::<sys::sllm_completion_t>::dangling();
        let (remaining, disposition, done) =
            classify_release(RuntimeStatus::InternalError, Some(raw));
        assert!(!done);
        assert_eq!(disposition, CleanupDisposition::Poisoned);
        assert!(remaining.is_some());

        let (remaining, disposition, done) =
            classify_release::<sys::sllm_completion_t>(RuntimeStatus::HipRuntimeError, None);
        assert!(!done);
        assert_eq!(disposition, CleanupDisposition::Poisoned);
        assert!(remaining.is_none());
    }

    #[test]
    fn pre_destruction_runtime_error_is_retryable_but_destroy_error_is_not() {
        let raw = NonNull::<sys::sllm_queue_t>::dangling();
        let (remaining, disposition, done) =
            classify_release(RuntimeStatus::HipRuntimeError, Some(raw));
        assert!(!done);
        assert_eq!(disposition, CleanupDisposition::Recoverable);
        assert_eq!(remaining, Some(raw));

        let (remaining, disposition, done) =
            classify_release::<sys::sllm_queue_t>(RuntimeStatus::HipRuntimeError, None);
        assert!(!done);
        assert_eq!(disposition, CleanupDisposition::Poisoned);
        assert!(remaining.is_none());
    }

    #[test]
    fn destroy_success_is_consumed_before_accounting_failure() {
        let raw = NonNull::<sys::sllm_event_t>::dangling();
        let (remaining, disposition, done) =
            classify_release::<sys::sllm_event_t>(RuntimeStatus::Ok, None);
        assert!(done);
        assert!(remaining.is_none());
        assert_eq!(disposition, CleanupDisposition::Recoverable);
        let (remaining, disposition, done) =
            classify_release(RuntimeStatus::InternalError, Some(raw));
        assert!(!done);
        assert_eq!(disposition, CleanupDisposition::Poisoned);
        assert!(remaining.is_some());
    }

    #[test]
    fn partial_submission_and_stale_token_are_retained_explicitly() {
        let raw = NonNull::<sys::sllm_completion_t>::dangling();
        let cleanup = PendingCleanup::Completion {
            raw: Some(raw),
            context: Arc::new(ContextInner::new(None)),
            queue: Arc::new(QueueInner {
                raw: None,
                context: Arc::new(ContextInner::new(None)),
            }),
            buffer: Arc::new(BufferInner {
                raw: None,
                _context: Arc::new(ContextInner::new(None)),
            }),
            disposition: CleanupDisposition::Poisoned,
        };
        assert!(cleanup.try_once().is_some());
    }

    #[test]
    fn thread_exit_handoff_keeps_poisoned_production_cleanup_durable() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _state = CleanupTestStateGuard::new();
        let before = DURABLE_QUARANTINE_ITEMS.load(Ordering::Relaxed);
        let completed = Arc::new(TimedGate::new());
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::spawn(move || {
            let _completed = CompletionSignal(worker_completed);
            enqueue_cleanup(PendingCleanup::Queue {
                raw: Some(NonNull::dangling()),
                context: Arc::new(ContextInner::new(None)),
                disposition: CleanupDisposition::Poisoned,
            });
        });
        let worker = TestWorker::new(worker, &completed, &[]);
        join_worker(worker, "cleanup TLS exit worker")
            .expect("cleanup thread must not panic during TLS destruction");
        assert!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Relaxed) > before,
            "TLS exit must hand off raw token and dependency ownership"
        );
    }

    thread_local! {
        static OLDER_TLS_COMPLETION: RefCell<Option<Completion>> = const { RefCell::new(None) };
    }

    struct CleanupTestStateGuard {
        pending: usize,
        durable: usize,
        durable_overflow: usize,
        accounting_errors: usize,
        handoff_state: usize,
        handoff_epoch: usize,
    }

    impl CleanupTestStateGuard {
        fn new() -> Self {
            Self {
                pending: PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
                durable: DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
                durable_overflow: DURABLE_QUARANTINE_OVERFLOW.load(Ordering::Acquire),
                accounting_errors: CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
                handoff_state: CLEANUP_HANDOFF_STATE.load(Ordering::Acquire),
                handoff_epoch: CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire),
            }
        }
    }

    impl Drop for CleanupTestStateGuard {
        fn drop(&mut self) {
            PENDING_CLEANUP_ITEMS.store(self.pending, Ordering::Release);
            DURABLE_QUARANTINE_ITEMS.store(self.durable, Ordering::Release);
            DURABLE_QUARANTINE_OVERFLOW.store(self.durable_overflow, Ordering::Release);
            CLEANUP_ACCOUNTING_ERRORS.store(self.accounting_errors, Ordering::Release);
            CLEANUP_HANDOFF_STATE.store(self.handoff_state, Ordering::Release);
            CLEANUP_HANDOFF_EPOCH.store(self.handoff_epoch, Ordering::Release);
        }
    }

    struct CleanupTestHookGuard {
        _state: CleanupTestStateGuard,
    }

    impl CleanupTestHookGuard {
        fn new() -> Self {
            Self {
                _state: CleanupTestStateGuard::new(),
            }
        }
    }

    impl Drop for CleanupTestHookGuard {
        fn drop(&mut self) {
            clear_cleanup_test_hooks();
        }
    }

    struct CompletionSignal(Arc<TimedGate>);

    impl Drop for CompletionSignal {
        fn drop(&mut self) {
            self.0.signal();
        }
    }

    struct TestWorker<T> {
        handle: Option<std::thread::JoinHandle<T>>,
        completed: Arc<TimedGate>,
        release_gates: Vec<Arc<TimedGate>>,
    }

    impl<T> TestWorker<T> {
        fn new(
            handle: std::thread::JoinHandle<T>,
            completed: &Arc<TimedGate>,
            release_gates: &[&Arc<TimedGate>],
        ) -> Self {
            Self {
                handle: Some(handle),
                completed: Arc::clone(completed),
                release_gates: release_gates.iter().map(|gate| Arc::clone(gate)).collect(),
            }
        }

        fn signal_releases(&self) {
            for gate in &self.release_gates {
                gate.signal();
            }
        }

        fn force_join(&mut self) {
            self.signal_releases();
            /* CompletionSignal gives this wait a bounded observation point.
             * The final join is a safety barrier: dropping a JoinHandle would
             * detach a worker that could still mutate process-global test
             * hooks after CleanupTestStateGuard restores them. */
            let _ = self.completed.wait();
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    impl<T> Drop for TestWorker<T> {
        fn drop(&mut self) {
            if self.handle.is_some() {
                self.force_join();
            }
        }
    }

    fn join_worker<T>(mut worker: TestWorker<T>, description: &str) -> std::thread::Result<T> {
        if !worker.completed.wait() {
            worker.force_join();
            panic!("{description}: worker did not complete within the test bound");
        }
        worker
            .handle
            .take()
            .expect("test worker handle must be present")
            .join()
    }

    fn wait_for_gate<T>(entered: &TimedGate, worker: &mut TestWorker<T>, description: &str) {
        if !entered.wait() {
            worker.force_join();
            panic!("{description}: hook was not reached within the test bound");
        }
    }

    #[test]
    fn gate_timeout_joins_worker_before_test_state_unwind() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let hooks = CleanupTestHookGuard::new();
        let entered = Arc::new(TimedGate::new());
        let release = Arc::new(TimedGate::new());
        let completed = Arc::new(TimedGate::new());
        let worker_release = Arc::clone(&release);
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::spawn(move || {
            let _completed = CompletionSignal(worker_completed);
            assert!(worker_release.wait_for(Duration::from_secs(10)));
            *FORCE_RETRY_THREAD
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                Some(std::thread::current().id());
        });
        let mut worker = TestWorker::new(worker, &completed, &[&release]);
        let timed_out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            wait_for_gate(&entered, &mut worker, "intentional worker gate timeout");
        }));
        assert!(timed_out.is_err());
        assert!(
            FORCE_RETRY_THREAD
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_some()
        );
        /* `wait_for_gate` has already joined the worker.  Clearing the hook
         * before the state guard restores globals is therefore deterministic. */
        drop(worker);
        clear_cleanup_test_hooks();
        drop(hooks);
        assert!(
            FORCE_RETRY_THREAD
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_none()
        );
    }

    fn live_queue_cleanup(
        probe: Arc<AtomicUsize>,
        disposition: CleanupDisposition,
    ) -> (
        PendingCleanup,
        Arc<ContextInner>,
        NonNull<sys::sllm_queue_t>,
    ) {
        let context = Arc::new(ContextInner::new_with_drop_probe(None, probe));
        let raw = NonNull::<sys::sllm_queue_t>::dangling();
        (
            PendingCleanup::Queue {
                raw: Some(raw),
                context: Arc::clone(&context),
                disposition,
            },
            context,
            raw,
        )
    }

    #[test]
    fn late_tls_drop_of_production_completion_is_durable_and_non_panicking() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_quarantine = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let drop_probe = Arc::new(AtomicUsize::new(0));
        let drop_probe_observer = Arc::clone(&drop_probe);
        let completed = Arc::new(TimedGate::new());
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::Builder::new()
            .name("c5-late-tls-access-error".to_owned())
            .spawn(move || {
                let _completed = CompletionSignal(worker_completed);
                let context = Arc::new(ContextInner::new_with_drop_probe(None, drop_probe));
                let queue = Arc::new(QueueInner {
                    raw: None,
                    context: Arc::clone(&context),
                });
                let buffer = Arc::new(BufferInner {
                    raw: None,
                    _context: Arc::clone(&context),
                });
                let completion = Completion {
                    raw: Some(NonNull::dangling()),
                    _context: Arc::clone(&context),
                    _queue: Arc::clone(&queue),
                    _buffer: Arc::clone(&buffer),
                    transfer_size_bytes: 1,
                    d2h: false,
                    terminal: false,
                    safe_to_release: false,
                };
                /* The host stub cannot return BUSY.  Force the cleanup
                 * boundary to retain this live record once, then let
                 * Completion::drop call the actual CLEANUP_REAPER.try_with
                 * after the TLS key has been destroyed. */
                *FORCE_RETRY_THREAD
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                    Some(std::thread::current().id());
                /* Initialize the older holder first.  Rust then destroys the
                 * reaper key first at thread exit, which exercises the real
                 * late TLS path rather than directly enqueueing a fake item. */
                OLDER_TLS_COMPLETION.with(|_| {});
                Context::reap_cleanup();
                OLDER_TLS_COMPLETION.with(|slot| *slot.borrow_mut() = Some(completion));
            });
        let worker = TestWorker::new(
            worker.expect("late TLS test thread must start"),
            &completed,
            &[],
        );
        join_worker(worker, "late TLS production cleanup worker")
            .expect("late TLS production cleanup must not panic or abort");

        assert_eq!(drop_probe_observer.load(Ordering::Acquire), 0);
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_quarantine + 1,
            "late TLS retryable cleanup must be quarantined exactly once"
        );
    }

    #[test]
    fn tls_teardown_and_shutdown_observe_the_same_durable_transition() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_quarantine = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let entered = Arc::new(TimedGate::new());
        let release = Arc::new(TimedGate::new());
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        let completed = Arc::new(TimedGate::new());
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::spawn(move || {
            let _completed = CompletionSignal(worker_completed);
            let thread = std::thread::current().id();
            *QUARANTINE_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread,
                entered: worker_entered,
                release: worker_release,
                panic_after_signal: false,
            });
            enqueue_cleanup(PendingCleanup::Queue {
                raw: Some(NonNull::dangling()),
                context: Arc::new(ContextInner::new(None)),
                disposition: CleanupDisposition::Recoverable,
            });
        });
        let mut worker = TestWorker::new(worker, &completed, &[&release]);

        wait_for_gate(&entered, &mut worker, "TLS teardown worker");
        assert_eq!(
            (
                PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
                DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire)
            ),
            (before_pending + 1, before_quarantine + 1)
        );
        assert!(matches!(
            Context::shutdown_cleanup(0),
            Err(error) if error.status() == RuntimeStatus::Busy
        ));

        release.signal();
        join_worker(worker, "TLS teardown worker")
            .expect("TLS teardown worker must not panic or leak the handoff barrier");
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_quarantine + 1
        );
        assert!(Context::shutdown_cleanup(0).is_ok());
    }

    #[test]
    fn reaper_borrow_failure_preserves_pending_entry() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_quarantine = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        enqueue_cleanup(PendingCleanup::Queue {
            raw: Some(NonNull::dangling()),
            context: Arc::new(ContextInner::new(None)),
            disposition: CleanupDisposition::Recoverable,
        });
        let queued_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        assert_eq!(queued_pending, before_pending + 1);
        CLEANUP_REAPER.with(|reaper| {
            let borrow = reaper.borrow_mut();
            reap_pending_cleanup();
            assert_eq!(
                PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
                queued_pending
            );
            drop(borrow);
        });
        *FORCE_RETRY_THREAD
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(std::thread::current().id());
        reap_pending_cleanup();
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            queued_pending,
            "retry requeue must retain the original accounting ticket"
        );
        reap_pending_cleanup();
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_quarantine + 1
        );
    }

    #[test]
    fn concurrent_drain_cannot_report_success_while_retry_is_in_flight() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_quarantine = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let entered = Arc::new(TimedGate::new());
        let release = Arc::new(TimedGate::new());
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        let completed = Arc::new(TimedGate::new());
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::Builder::new()
            .name("c5-cleanup-race-worker".to_owned())
            .spawn(move || {
                let _completed = CompletionSignal(worker_completed);
                let thread = std::thread::current().id();
                *FORCE_RETRY_THREAD
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(thread);
                *REAP_PAUSE_HOOK
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                    thread,
                    entered: worker_entered,
                    release: worker_release,
                    panic_after_signal: false,
                });
                enqueue_cleanup(PendingCleanup::Queue {
                    raw: Some(NonNull::dangling()),
                    context: Arc::new(ContextInner::new(None)),
                    disposition: CleanupDisposition::Recoverable,
                });
                reap_pending_cleanup();
                /* The forced retry was requeued without changing the
                 * accounting ticket.  A final pass reaches the host stub's
                 * terminal invalid-handle disposition and quarantines it. */
                reap_pending_cleanup();
            })
            .expect("cleanup race worker must start");
        let mut worker = TestWorker::new(worker, &completed, &[&release]);

        wait_for_gate(&entered, &mut worker, "cleanup race worker");
        let drain_a_done = Arc::new(TimedGate::new());
        let drain_a_done_worker = Arc::clone(&drain_a_done);
        let drain_a = std::thread::spawn(move || {
            let _completed = CompletionSignal(drain_a_done_worker);
            Context::shutdown_cleanup(0)
        });
        let drain_a = TestWorker::new(drain_a, &drain_a_done, &[]);
        let drain_b_done = Arc::new(TimedGate::new());
        let drain_b_done_worker = Arc::clone(&drain_b_done);
        let drain_b = std::thread::spawn(move || {
            let _completed = CompletionSignal(drain_b_done_worker);
            Context::shutdown_cleanup(0)
        });
        let drain_b = TestWorker::new(drain_b, &drain_b_done, &[]);
        let first = join_worker(drain_a, "first concurrent drain")
            .expect("first concurrent drain must not panic");
        let second = join_worker(drain_b, "second concurrent drain")
            .expect("second concurrent drain must not panic");
        for result in [first, second] {
            assert!(
                matches!(&result, Err(error) if error.status() == RuntimeStatus::Busy),
                "drain must fail closed while a taken record is in flight: {result:?}"
            );
        }
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending + 1
        );
        release.signal();
        join_worker(worker, "cleanup race worker").expect("cleanup race worker must not panic");
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_quarantine + 1
        );
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            0,
            "cleanup accounting must not underflow or double-decrement"
        );
    }

    #[test]
    fn pending_to_durable_transition_is_visible_to_shutdown_before_decrement() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_durable = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let entered_take = Arc::new(TimedGate::new());
        let release_take = Arc::new(TimedGate::new());
        let entered_quarantine = Arc::new(TimedGate::new());
        let release_quarantine = Arc::new(TimedGate::new());
        let worker_entered_take = Arc::clone(&entered_take);
        let worker_release_take = Arc::clone(&release_take);
        let worker_entered_quarantine = Arc::clone(&entered_quarantine);
        let worker_release_quarantine = Arc::clone(&release_quarantine);
        let completed = Arc::new(TimedGate::new());
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::spawn(move || {
            let _completed = CompletionSignal(worker_completed);
            let thread = std::thread::current().id();
            *FORCE_RETRY_THREAD
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(thread);
            *REAP_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread,
                entered: worker_entered_take,
                release: worker_release_take,
                panic_after_signal: false,
            });
            *QUARANTINE_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread,
                entered: worker_entered_quarantine,
                release: worker_release_quarantine,
                panic_after_signal: false,
            });
            enqueue_cleanup(PendingCleanup::Queue {
                raw: Some(NonNull::dangling()),
                context: Arc::new(ContextInner::new(None)),
                disposition: CleanupDisposition::Recoverable,
            });
            reap_pending_cleanup();
        });
        let mut worker = TestWorker::new(worker, &completed, &[&release_take, &release_quarantine]);
        wait_for_gate(&entered_take, &mut worker, "transition worker take");
        let drain = enter_cleanup_drain(0).expect("test drain must own its barrier");
        release_take.signal();
        wait_for_gate(
            &entered_quarantine,
            &mut worker,
            "transition worker quarantine",
        );
        assert_eq!(
            (
                PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
                DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire)
            ),
            (before_pending + 1, before_durable + 1),
            "durable ownership must be observable before pending is removed"
        );
        let shutdown = Context::shutdown_cleanup(0);
        assert!(matches!(
            shutdown,
            Err(error) if error.status() == RuntimeStatus::Busy
        ));
        release_quarantine.signal();
        join_worker(worker, "transition worker").expect("transition worker must not panic");
        drop(drain);
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable + 1
        );
    }

    #[test]
    fn publication_while_drain_bit_is_active_is_bounded_and_not_lost() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let before_durable = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let drain = enter_cleanup_drain(0).expect("test drain must own its barrier");
        let entered = Arc::new(TimedGate::new());
        let release = Arc::new(TimedGate::new());
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        let completed = Arc::new(TimedGate::new());
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::spawn(move || {
            let _completed = CompletionSignal(worker_completed);
            let thread = std::thread::current().id();
            *HANDOFF_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread,
                entered: worker_entered,
                release: worker_release,
                panic_after_signal: false,
            });
            enqueue_cleanup(PendingCleanup::Queue {
                raw: Some(NonNull::dangling()),
                context: Arc::new(ContextInner::new(None)),
                disposition: CleanupDisposition::Poisoned,
            });
        });
        let mut worker = TestWorker::new(worker, &completed, &[&release]);
        wait_for_gate(&entered, &mut worker, "publication worker");
        let shutdown = Context::shutdown_cleanup(0);
        assert!(matches!(
            shutdown,
            Err(error) if error.status() == RuntimeStatus::Busy
        ));
        release.signal();
        join_worker(worker, "publication worker").expect("publication worker must not panic");
        drop(drain);
        assert!(Context::shutdown_cleanup(0).is_ok());
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable + 1
        );
    }

    #[test]
    fn drain_snapshot_rejects_completed_handoff_state_aba() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_durable = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let before_epoch = CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire);
        let entered = Arc::new(TimedGate::new());
        let release = Arc::new(TimedGate::new());
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        let completed = Arc::new(TimedGate::new());
        let worker_completed = Arc::clone(&completed);
        let drain = std::thread::spawn(move || {
            let _completed = CompletionSignal(worker_completed);
            let thread = std::thread::current().id();
            *SNAPSHOT_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread,
                entered: worker_entered,
                release: worker_release,
                panic_after_signal: false,
            });
            Context::drain_cleanup(1)
        });
        let mut drain = TestWorker::new(drain, &completed, &[&release]);

        wait_for_gate(&entered, &mut drain, "snapshot drain");
        assert_eq!(
            CLEANUP_HANDOFF_STATE.load(Ordering::Acquire),
            CLEANUP_DRAIN_BIT
        );
        enqueue_cleanup(PendingCleanup::Queue {
            raw: Some(NonNull::dangling()),
            context: Arc::new(ContextInner::new(None)),
            disposition: CleanupDisposition::Poisoned,
        });
        assert_eq!(
            CLEANUP_HANDOFF_STATE.load(Ordering::Acquire),
            CLEANUP_DRAIN_BIT,
            "the old state value must be restored to exercise the ABA"
        );
        assert_eq!(
            CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire),
            before_epoch + 1
        );
        release.signal();

        let counts = join_worker(drain, "snapshot drain")
            .expect("snapshot drain must not panic")
            .expect("completed handoff ABA must be retried, not accepted");
        assert_eq!(counts, (before_pending, before_durable + 1));
    }

    #[test]
    fn drain_success_after_post_cas_incomplete_handoff_preserves_ticket() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_durable = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let before_epoch = CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire);
        let before_state = CLEANUP_HANDOFF_STATE.load(Ordering::Acquire);
        assert_eq!(before_state, 0, "serialized cleanup state must be idle");

        let post_cas_entered = Arc::new(TimedGate::new());
        let post_cas_release = Arc::new(TimedGate::new());
        let drain_done = Arc::new(TimedGate::new());
        let drain_post_cas_entered = Arc::clone(&post_cas_entered);
        let drain_post_cas_release = Arc::clone(&post_cas_release);
        let drain_completed = Arc::clone(&drain_done);
        let drain = std::thread::spawn(move || {
            let _completed = CompletionSignal(drain_completed);
            *POST_CAS_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread: std::thread::current().id(),
                entered: drain_post_cas_entered,
                release: drain_post_cas_release,
                panic_after_signal: false,
            });
            Context::drain_cleanup(1)
        });
        let mut drain = TestWorker::new(drain, &drain_done, &[&post_cas_release]);

        wait_for_gate(&post_cas_entered, &mut drain, "post-CAS drain");

        let handoff_entered = Arc::new(TimedGate::new());
        let handoff_release = Arc::new(TimedGate::new());
        let worker_done = Arc::new(TimedGate::new());
        let worker_handoff_entered = Arc::clone(&handoff_entered);
        let worker_handoff_release = Arc::clone(&handoff_release);
        let worker_completed = Arc::clone(&worker_done);
        let worker = std::thread::spawn(move || {
            let _completed = CompletionSignal(worker_completed);
            *HANDOFF_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread: std::thread::current().id(),
                entered: worker_handoff_entered,
                release: worker_handoff_release,
                panic_after_signal: false,
            });
            enqueue_cleanup(PendingCleanup::Queue {
                raw: Some(NonNull::dangling()),
                context: Arc::new(ContextInner::new(None)),
                disposition: CleanupDisposition::Poisoned,
            });
        });
        let mut worker = TestWorker::new(worker, &worker_done, &[&handoff_release]);

        wait_for_gate(&handoff_entered, &mut worker, "post-CAS handoff");
        assert_eq!(
            CLEANUP_HANDOFF_STATE.load(Ordering::Acquire),
            CLEANUP_DRAIN_BIT + CLEANUP_HANDOFF_UNIT,
            "an incomplete post-CAS handoff must retain its active ticket"
        );

        post_cas_release.signal();
        let counts = join_worker(drain, "post-CAS drain")
            .expect("post-CAS drain must not panic")
            .expect("incomplete post-CAS handoff may coexist with Ok");
        assert_eq!(counts, (before_pending, before_durable));
        assert_eq!(
            CLEANUP_HANDOFF_STATE.load(Ordering::Acquire),
            CLEANUP_HANDOFF_UNIT,
            "drain exit must preserve the exact handoff ticket"
        );

        handoff_release.signal();
        join_worker(worker, "post-CAS handoff worker")
            .expect("post-CAS handoff worker must not panic");
        assert_eq!(CLEANUP_HANDOFF_STATE.load(Ordering::Acquire), before_state);
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable + 1
        );
        assert_eq!(
            CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire),
            before_epoch + 1
        );
        assert!(Context::shutdown_cleanup(0).is_ok());
    }

    #[test]
    fn zero_attempt_drain_returns_busy_without_spinning() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        enqueue_cleanup(PendingCleanup::Queue {
            raw: Some(NonNull::dangling()),
            context: Arc::new(ContextInner::new(None)),
            disposition: CleanupDisposition::Recoverable,
        });
        assert!(matches!(
            Context::shutdown_cleanup(0),
            Err(error) if error.status() == RuntimeStatus::Busy
        ));
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending + 1
        );
        reap_pending_cleanup();
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
    }

    #[test]
    fn cas_failure_injection_is_thread_and_owner_isolated() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let _failure = CasFailureGuard::new(CleanupCasTarget::PendingIncrement, CLEANUP_CAS_BOUND);
        assert!(!cleanup_cas_failure_injected(
            CleanupCasTarget::DurableIncrement
        ));
        let other_thread =
            std::thread::spawn(|| cleanup_cas_failure_injected(CleanupCasTarget::PendingIncrement))
                .join()
                .expect("CAS failure isolation worker must not panic");
        assert!(!other_thread);
        assert!(cleanup_cas_failure_injected(
            CleanupCasTarget::PendingIncrement
        ));
    }

    #[test]
    fn contended_pending_acceptance_retains_non_null_cleanup_and_fails_shutdown() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let _state = CleanupTestStateGuard::new();
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_durable = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let before_errors = CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire);
        let probe = Arc::new(AtomicUsize::new(0));
        let (cleanup, owner, raw) =
            live_queue_cleanup(Arc::clone(&probe), CleanupDisposition::Recoverable);
        let _failure = CasFailureGuard::new(CleanupCasTarget::PendingIncrement, CLEANUP_CAS_BOUND);
        enqueue_cleanup(cleanup);

        assert_eq!(raw, NonNull::<sys::sllm_queue_t>::dangling());
        assert_eq!(Arc::strong_count(&owner), 2);
        assert_eq!(probe.load(Ordering::Acquire), 0);
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable + 1
        );
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors + 1
        );
        assert!(Context::checked_cleanup_counts().is_err());
        assert!(matches!(
            Context::shutdown_cleanup(0),
            Err(error) if error.status() == RuntimeStatus::InternalError
        ));
    }

    #[test]
    fn contended_pending_decrement_retains_durable_record_without_false_zero() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let _state = CleanupTestStateGuard::new();
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_durable = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let before_epoch = CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire);
        let before_errors = CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire);
        let probe = Arc::new(AtomicUsize::new(0));
        let (cleanup, owner, raw) =
            live_queue_cleanup(Arc::clone(&probe), CleanupDisposition::Recoverable);
        let record = match CleanupRecord::accepted(cleanup) {
            Ok(record) => record,
            Err(_) => panic!("pending ticket must be accepted before decrement injection"),
        };
        let _failure = CasFailureGuard::new(CleanupCasTarget::PendingDecrement, CLEANUP_CAS_BOUND);
        quarantine_pending_record(record);

        assert_eq!(raw, NonNull::<sys::sllm_queue_t>::dangling());
        assert_eq!(Arc::strong_count(&owner), 2);
        assert_eq!(probe.load(Ordering::Acquire), 0);
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending + 1
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable + 1
        );
        assert_eq!(CLEANUP_HANDOFF_STATE.load(Ordering::Acquire), 0);
        assert_eq!(
            CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire),
            before_epoch + 1
        );
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors + 1
        );
        assert!(Context::checked_cleanup_counts().is_err());
        assert!(matches!(
            Context::shutdown_cleanup(0),
            Err(error) if error.status() == RuntimeStatus::InternalError
        ));
    }

    #[test]
    fn contended_handoff_state_quarantines_without_losing_ownership() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let _state = CleanupTestStateGuard::new();
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_durable = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let before_epoch = CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire);
        let before_state = CLEANUP_HANDOFF_STATE.load(Ordering::Acquire);
        let before_errors = CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire);
        let probe = Arc::new(AtomicUsize::new(0));
        let (cleanup, owner, raw) =
            live_queue_cleanup(Arc::clone(&probe), CleanupDisposition::Recoverable);
        let _failure = CasFailureGuard::new(CleanupCasTarget::HandoffBegin, CLEANUP_CAS_BOUND);
        enqueue_cleanup(cleanup);

        assert_eq!(raw, NonNull::<sys::sllm_queue_t>::dangling());
        assert_eq!(Arc::strong_count(&owner), 2);
        assert_eq!(probe.load(Ordering::Acquire), 0);
        assert_eq!(CLEANUP_HANDOFF_STATE.load(Ordering::Acquire), before_state);
        assert_eq!(CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire), before_epoch);
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable + 1
        );
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors + 1
        );
        assert!(Context::checked_cleanup_counts().is_err());
        assert!(matches!(
            Context::shutdown_cleanup(0),
            Err(error) if error.status() == RuntimeStatus::InternalError
        ));
    }

    #[test]
    fn accounting_underflow_rejects_shutdown_and_restores_state() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _state = CleanupTestStateGuard::new();
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_durable = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let before_errors = CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire);
        finish_pending_cleanup();
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors + 1
        );
        let result = Context::shutdown_cleanup(0);
        assert!(matches!(
            result,
            Err(error) if error.status() == RuntimeStatus::InternalError
        ));
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable
        );
    }

    #[test]
    fn durable_overflow_is_checked_and_shutdown_fails_closed() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let before_durable = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let before_errors = CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire);
        DURABLE_QUARANTINE_ITEMS.store(usize::MAX, Ordering::Release);
        let entered = Arc::new(TimedGate::new());
        let release = Arc::new(TimedGate::new());
        let completed = Arc::new(TimedGate::new());
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        let worker_probe = Arc::new(AtomicUsize::new(0));
        let probe = Arc::clone(&worker_probe);
        let (evidence_tx, evidence_rx) = std::sync::mpsc::channel();
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::spawn(move || {
            let _completed = CompletionSignal(worker_completed);
            *DURABLE_FORGET_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread: std::thread::current().id(),
                entered: worker_entered,
                release: worker_release,
                panic_after_signal: false,
            });
            let (cleanup, owner, raw) = live_queue_cleanup(probe, CleanupDisposition::Poisoned);
            evidence_tx
                .send((raw.as_ptr() as usize, Arc::strong_count(&owner)))
                .expect("durable-only saturation evidence receiver must remain live");
            assert!(!durable_quarantine(cleanup));
            assert_eq!(Arc::strong_count(&owner), 2);
        });
        let mut worker = TestWorker::new(worker, &completed, &[&release]);
        let (raw_address, strong_count) = evidence_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("durable-only saturation worker must publish live ownership evidence");
        wait_for_gate(&entered, &mut worker, "durable-only overflow before forget");
        assert_ne!(raw_address, 0);
        assert_eq!(strong_count, 2);
        assert_eq!(worker_probe.load(Ordering::Acquire), 0);
        assert_eq!(DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire), usize::MAX);
        assert_eq!(DURABLE_QUARANTINE_OVERFLOW.load(Ordering::Acquire), 1);
        assert!(Context::durable_quarantine_overflowed());
        assert!(Context::checked_cleanup_counts().is_err());
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors + 1
        );
        assert!(matches!(
            Context::shutdown_cleanup(0),
            Err(error) if error.status() == RuntimeStatus::InternalError
        ));
        release.signal();
        join_worker(worker, "durable-only saturation worker")
            .expect("durable-only saturation worker must not panic");
        assert_eq!(worker_probe.load(Ordering::Acquire), 0);
        assert_eq!(DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire), usize::MAX);
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors + 1
        );
        assert!(matches!(
            Context::shutdown_cleanup(0),
            Err(error) if error.status() == RuntimeStatus::InternalError
        ));
        assert_ne!(before_durable, usize::MAX);
    }

    #[test]
    fn pending_and_durable_saturation_publish_overflow_before_forget() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_durable = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let before_errors = CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire);
        PENDING_CLEANUP_ITEMS.store(usize::MAX, Ordering::Release);
        DURABLE_QUARANTINE_ITEMS.store(usize::MAX, Ordering::Release);

        let entered = Arc::new(TimedGate::new());
        let release = Arc::new(TimedGate::new());
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        let worker_probe = Arc::new(AtomicUsize::new(0));
        let probe = Arc::clone(&worker_probe);
        let (evidence_tx, evidence_rx) = std::sync::mpsc::channel();
        let completed = Arc::new(TimedGate::new());
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::spawn(move || {
            let _completed = CompletionSignal(worker_completed);
            let thread = std::thread::current().id();
            *DURABLE_FORGET_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread,
                entered: worker_entered,
                release: worker_release,
                panic_after_signal: false,
            });
            let (cleanup, owner, raw) = live_queue_cleanup(probe, CleanupDisposition::Recoverable);
            evidence_tx
                .send((raw.as_ptr() as usize, Arc::strong_count(&owner)))
                .expect("saturation ownership evidence receiver must remain live");
            enqueue_cleanup(cleanup);
            assert_eq!(Arc::strong_count(&owner), 2);
        });
        let mut worker = TestWorker::new(worker, &completed, &[&release]);

        let (raw_address, strong_count) = evidence_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("saturation worker must publish live ownership evidence");
        wait_for_gate(&entered, &mut worker, "durable overflow before forget");
        assert_ne!(raw_address, 0);
        assert_eq!(strong_count, 2);
        assert_eq!(worker_probe.load(Ordering::Acquire), 0);
        assert_eq!(DURABLE_QUARANTINE_OVERFLOW.load(Ordering::Acquire), 1);
        release.signal();
        join_worker(worker, "saturation ownership worker")
            .expect("saturation ownership worker must not panic");

        assert_eq!(PENDING_CLEANUP_ITEMS.load(Ordering::Acquire), usize::MAX);
        assert_eq!(DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire), usize::MAX);
        assert!(Context::durable_quarantine_overflowed());
        assert!(Context::checked_cleanup_counts().is_err());
        assert_eq!(worker_probe.load(Ordering::Acquire), 0);
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors + 2
        );
        assert!(matches!(
            Context::shutdown_cleanup(0),
            Err(error) if error.status() == RuntimeStatus::InternalError
        ));
        assert_ne!(before_pending, usize::MAX);
        assert_ne!(before_durable, usize::MAX);
    }

    #[test]
    fn saturated_accounting_error_counter_stays_sticky_and_fails_closed() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _state = CleanupTestStateGuard::new();
        CLEANUP_ACCOUNTING_ERRORS.store(usize::MAX, Ordering::Release);
        record_cleanup_accounting_error();
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            usize::MAX
        );
        assert!(Context::checked_cleanup_counts().is_err());
        assert!(matches!(
            Context::shutdown_cleanup(0),
            Err(error) if error.status() == RuntimeStatus::InternalError
        ));
    }

    #[test]
    fn saturated_handoff_epoch_retains_state_ticket_and_fails_closed() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _state = CleanupTestStateGuard::new();
        let before_state = CLEANUP_HANDOFF_STATE.load(Ordering::Acquire);
        let before_durable = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let before_errors = CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire);
        CLEANUP_HANDOFF_EPOCH.store(usize::MAX, Ordering::Release);

        let probe = Arc::new(AtomicUsize::new(0));
        let (cleanup, owner, raw) =
            live_queue_cleanup(Arc::clone(&probe), CleanupDisposition::Poisoned);
        enqueue_cleanup(cleanup);

        assert_eq!(raw, NonNull::<sys::sllm_queue_t>::dangling());
        assert_eq!(Arc::strong_count(&owner), 2);
        assert_eq!(probe.load(Ordering::Acquire), 0);
        assert_eq!(CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire), usize::MAX);
        assert_eq!(
            CLEANUP_HANDOFF_STATE.load(Ordering::Acquire),
            before_state + CLEANUP_HANDOFF_UNIT
        );
        assert!(matches!(
            Context::shutdown_cleanup(0),
            Err(error)
                if matches!(error.status(), RuntimeStatus::Busy | RuntimeStatus::InternalError)
        ));
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable + 1
        );
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors + 1
        );
        assert_eq!(
            CLEANUP_HANDOFF_STATE.load(Ordering::Acquire),
            before_state + CLEANUP_HANDOFF_UNIT
        );
    }

    #[test]
    fn handoff_hook_panic_before_idle_pending_acceptance_retains_recoverable_cleanup() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let _state = CleanupTestStateGuard::new();
        let before_state = CLEANUP_HANDOFF_STATE.load(Ordering::Acquire);
        let before_epoch = CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire);
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_durable = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let before_errors = CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire);
        let entered = Arc::new(TimedGate::new());
        let release = Arc::new(TimedGate::new());
        let entered_quarantine = Arc::new(TimedGate::new());
        let release_quarantine = Arc::new(TimedGate::new());
        let completed = Arc::new(TimedGate::new());
        let worker_probe = Arc::new(AtomicUsize::new(0));
        let probe = Arc::clone(&worker_probe);
        let (evidence_tx, evidence_rx) = std::sync::mpsc::channel();
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        let worker_entered_quarantine = Arc::clone(&entered_quarantine);
        let worker_release_quarantine = Arc::clone(&release_quarantine);
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::spawn(move || {
            let _completed = CompletionSignal(worker_completed);
            let thread = std::thread::current().id();
            *HANDOFF_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread,
                entered: worker_entered,
                release: worker_release,
                panic_after_signal: true,
            });
            *QUARANTINE_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread,
                entered: worker_entered_quarantine,
                release: worker_release_quarantine,
                panic_after_signal: false,
            });
            let (cleanup, owner, raw) = live_queue_cleanup(probe, CleanupDisposition::Recoverable);
            evidence_tx
                .send((raw.as_ptr() as usize, Arc::strong_count(&owner)))
                .expect("idle panic ownership evidence receiver must remain live");
            enqueue_cleanup(cleanup);
        });
        let mut worker = TestWorker::new(worker, &completed, &[&release, &release_quarantine]);
        let (raw_address, strong_count) = evidence_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("idle panic worker must publish live ownership evidence");
        wait_for_gate(&entered, &mut worker, "idle panic handoff hook");
        wait_for_gate(
            &entered_quarantine,
            &mut worker,
            "idle panic durable-before-pending hook",
        );
        assert_eq!(
            CLEANUP_HANDOFF_STATE.load(Ordering::Acquire),
            before_state + CLEANUP_HANDOFF_UNIT
        );
        assert_eq!(CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire), before_epoch);
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable + 1
        );
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors
        );
        assert!(matches!(
            Context::shutdown_cleanup(0),
            Err(error) if error.status() == RuntimeStatus::Busy
        ));

        release_quarantine.signal();
        let result = join_worker(worker, "idle panic worker");
        assert!(
            result.is_err(),
            "the real pause hook must panic in the worker"
        );
        assert_eq!(CLEANUP_HANDOFF_STATE.load(Ordering::Acquire), before_state);
        assert_eq!(CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire), before_epoch);
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable + 1
        );
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors
        );
        assert_ne!(raw_address, 0);
        assert_eq!(strong_count, 2);
        assert_eq!(worker_probe.load(Ordering::Acquire), 0);
        assert!(Context::shutdown_cleanup(0).is_ok());
    }

    #[test]
    fn handoff_hook_panic_with_drain_bit_retains_direct_quarantine() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let before_state = CLEANUP_HANDOFF_STATE.load(Ordering::Acquire);
        let before_epoch = CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire);
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_durable = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let before_errors = CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire);
        let drain = enter_cleanup_drain(0).expect("test drain must own its barrier");
        assert_eq!(
            CLEANUP_HANDOFF_STATE.load(Ordering::Acquire),
            before_state | CLEANUP_DRAIN_BIT
        );

        let entered = Arc::new(TimedGate::new());
        let release = Arc::new(TimedGate::new());
        let entered_quarantine = Arc::new(TimedGate::new());
        let release_quarantine = Arc::new(TimedGate::new());
        let completed = Arc::new(TimedGate::new());
        let worker_probe = Arc::new(AtomicUsize::new(0));
        let probe = Arc::clone(&worker_probe);
        let (evidence_tx, evidence_rx) = std::sync::mpsc::channel();
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        let worker_entered_quarantine = Arc::clone(&entered_quarantine);
        let worker_release_quarantine = Arc::clone(&release_quarantine);
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::spawn(move || {
            let _completed = CompletionSignal(worker_completed);
            let thread = std::thread::current().id();
            *HANDOFF_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread,
                entered: worker_entered,
                release: worker_release,
                panic_after_signal: true,
            });
            *QUARANTINE_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread,
                entered: worker_entered_quarantine,
                release: worker_release_quarantine,
                panic_after_signal: false,
            });
            let (cleanup, owner, raw) = live_queue_cleanup(probe, CleanupDisposition::Recoverable);
            evidence_tx
                .send((raw.as_ptr() as usize, Arc::strong_count(&owner)))
                .expect("drain-bit panic ownership evidence receiver must remain live");
            enqueue_cleanup(cleanup);
        });
        let mut worker = TestWorker::new(worker, &completed, &[&release, &release_quarantine]);
        let (raw_address, strong_count) = evidence_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("drain-bit panic worker must publish live ownership evidence");
        wait_for_gate(&entered, &mut worker, "drain-bit panic handoff hook");
        wait_for_gate(
            &entered_quarantine,
            &mut worker,
            "drain-bit panic durable-before-pending hook",
        );
        assert_eq!(
            CLEANUP_HANDOFF_STATE.load(Ordering::Acquire),
            (before_state | CLEANUP_DRAIN_BIT) + CLEANUP_HANDOFF_UNIT
        );
        assert_eq!(CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire), before_epoch);
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable + 1
        );
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors
        );
        release_quarantine.signal();
        let result = join_worker(worker, "drain-bit panic handoff worker");
        assert!(
            result.is_err(),
            "the real pause hook must panic in the worker"
        );
        assert_eq!(
            CLEANUP_HANDOFF_STATE.load(Ordering::Acquire),
            before_state | CLEANUP_DRAIN_BIT,
            "the drain bit must survive the aborted handoff without a false zero"
        );
        assert_eq!(CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire), before_epoch);
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable + 1
        );
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors
        );
        assert_ne!(raw_address, 0);
        assert_eq!(strong_count, 2);
        assert_eq!(worker_probe.load(Ordering::Acquire), 0);
        assert!(matches!(
            Context::shutdown_cleanup(0),
            Err(error) if error.status() == RuntimeStatus::Busy
        ));
        drop(drain);
        assert_eq!(CLEANUP_HANDOFF_STATE.load(Ordering::Acquire), before_state);
        assert!(Context::shutdown_cleanup(0).is_ok());
        assert_eq!(worker_probe.load(Ordering::Acquire), 0);
    }

    #[test]
    fn handoff_hook_panic_during_drain_bit_requeue_retains_pending_ticket_order() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let _state = CleanupTestStateGuard::new();
        let before_state = CLEANUP_HANDOFF_STATE.load(Ordering::Acquire);
        let before_epoch = CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire);
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_durable = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let before_errors = CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire);
        let drain = enter_cleanup_drain(0).expect("test drain must own its barrier");

        let entered_handoff = Arc::new(TimedGate::new());
        let release_handoff = Arc::new(TimedGate::new());
        let entered_quarantine = Arc::new(TimedGate::new());
        let release_quarantine = Arc::new(TimedGate::new());
        let completed = Arc::new(TimedGate::new());
        let worker_probe = Arc::new(AtomicUsize::new(0));
        let probe = Arc::clone(&worker_probe);
        let (evidence_tx, evidence_rx) = std::sync::mpsc::channel();
        let worker_entered_handoff = Arc::clone(&entered_handoff);
        let worker_release_handoff = Arc::clone(&release_handoff);
        let worker_entered_quarantine = Arc::clone(&entered_quarantine);
        let worker_release_quarantine = Arc::clone(&release_quarantine);
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::spawn(move || {
            let _completed = CompletionSignal(worker_completed);
            let thread = std::thread::current().id();
            *HANDOFF_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread,
                entered: worker_entered_handoff,
                release: worker_release_handoff,
                panic_after_signal: true,
            });
            *QUARANTINE_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread,
                entered: worker_entered_quarantine,
                release: worker_release_quarantine,
                panic_after_signal: false,
            });
            let (cleanup, owner, raw) = live_queue_cleanup(probe, CleanupDisposition::Recoverable);
            evidence_tx
                .send((raw.as_ptr() as usize, Arc::strong_count(&owner)))
                .expect("drain-bit requeue panic evidence receiver must remain live");
            let record = match CleanupRecord::accepted(cleanup) {
                Ok(record) => record,
                Err(_) => panic!("drain-bit requeue test must accept its pending ticket"),
            };
            requeue_cleanup(record);
        });
        let mut worker =
            TestWorker::new(worker, &completed, &[&release_handoff, &release_quarantine]);
        let (raw_address, strong_count) = evidence_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("drain-bit requeue worker must publish ownership evidence");
        wait_for_gate(
            &entered_handoff,
            &mut worker,
            "drain-bit requeue handoff hook",
        );
        wait_for_gate(
            &entered_quarantine,
            &mut worker,
            "drain-bit requeue durable-before-pending hook",
        );
        assert_eq!(
            CLEANUP_HANDOFF_STATE.load(Ordering::Acquire),
            (before_state | CLEANUP_DRAIN_BIT) + CLEANUP_HANDOFF_UNIT
        );
        assert_eq!(CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire), before_epoch);
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending + 1
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable + 1
        );
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors
        );
        assert!(matches!(
            Context::shutdown_cleanup(0),
            Err(error) if error.status() == RuntimeStatus::Busy
        ));

        release_quarantine.signal();
        let result = join_worker(worker, "drain-bit requeue panic worker");
        assert!(
            result.is_err(),
            "the real pause hook must panic in the worker"
        );
        assert_eq!(
            CLEANUP_HANDOFF_STATE.load(Ordering::Acquire),
            before_state | CLEANUP_DRAIN_BIT
        );
        assert_eq!(CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire), before_epoch);
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable + 1
        );
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors
        );
        assert_ne!(raw_address, 0);
        assert_eq!(strong_count, 2);
        assert_eq!(worker_probe.load(Ordering::Acquire), 0);

        drop(drain);
        assert_eq!(CLEANUP_HANDOFF_STATE.load(Ordering::Acquire), before_state);
        assert!(Context::shutdown_cleanup(0).is_ok());
    }

    #[test]
    fn handoff_hook_panic_during_requeue_retains_existing_pending_ticket() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let before_state = CLEANUP_HANDOFF_STATE.load(Ordering::Acquire);
        let before_epoch = CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire);
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_durable = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let before_errors = CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire);
        let entered = Arc::new(TimedGate::new());
        let release = Arc::new(TimedGate::new());
        let entered_quarantine = Arc::new(TimedGate::new());
        let release_quarantine = Arc::new(TimedGate::new());
        let completed = Arc::new(TimedGate::new());
        let worker_probe = Arc::new(AtomicUsize::new(0));
        let probe = Arc::clone(&worker_probe);
        let (evidence_tx, evidence_rx) = std::sync::mpsc::channel();
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        let worker_entered_quarantine = Arc::clone(&entered_quarantine);
        let worker_release_quarantine = Arc::clone(&release_quarantine);
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::spawn(move || {
            let _completed = CompletionSignal(worker_completed);
            let thread = std::thread::current().id();
            let (cleanup, owner, raw) = live_queue_cleanup(probe, CleanupDisposition::Recoverable);
            evidence_tx
                .send((raw.as_ptr() as usize, Arc::strong_count(&owner)))
                .expect("requeue panic ownership evidence receiver must remain live");
            let record = match CleanupRecord::accepted(cleanup) {
                Ok(record) => record,
                Err(_) => panic!("requeue panic test must accept its pending ticket"),
            };
            *HANDOFF_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread,
                entered: worker_entered,
                release: worker_release,
                panic_after_signal: true,
            });
            *QUARANTINE_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread,
                entered: worker_entered_quarantine,
                release: worker_release_quarantine,
                panic_after_signal: false,
            });
            requeue_cleanup(record);
        });
        let mut worker = TestWorker::new(worker, &completed, &[&release, &release_quarantine]);
        let (raw_address, strong_count) = evidence_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("requeue panic worker must publish live ownership evidence");
        wait_for_gate(&entered, &mut worker, "requeue panic handoff hook");
        wait_for_gate(
            &entered_quarantine,
            &mut worker,
            "requeue panic durable-before-pending hook",
        );
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending + 1
        );
        assert_eq!(
            CLEANUP_HANDOFF_STATE.load(Ordering::Acquire),
            before_state + CLEANUP_HANDOFF_UNIT
        );
        assert_eq!(CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire), before_epoch);
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable + 1
        );
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors
        );
        assert!(matches!(
            Context::shutdown_cleanup(0),
            Err(error) if error.status() == RuntimeStatus::Busy
        ));
        release_quarantine.signal();
        let result = join_worker(worker, "requeue panic worker");
        assert!(
            result.is_err(),
            "the real pause hook must panic in the worker"
        );
        assert_eq!(CLEANUP_HANDOFF_STATE.load(Ordering::Acquire), before_state);
        assert_eq!(CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire), before_epoch);
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable + 1
        );
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors
        );
        assert_ne!(raw_address, 0);
        assert_eq!(strong_count, 2);
        assert_eq!(worker_probe.load(Ordering::Acquire), 0);
        assert!(Context::shutdown_cleanup(0).is_ok());
    }

    #[test]
    fn hooks_are_owner_isolated_and_panic_cleanup_is_deterministic() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hooks = CleanupTestHookGuard::new();
        let other_thread = std::thread::spawn(|| std::thread::current().id())
            .join()
            .unwrap();
        *REAP_PAUSE_HOOK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
            thread: other_thread,
            entered: Arc::new(TimedGate::new()),
            release: Arc::new(TimedGate::new()),
            panic_after_signal: false,
        });
        pause_after_reap_take_for_test();
        assert!(
            REAP_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_some()
        );
        clear_cleanup_test_hooks();
        let before_state = CLEANUP_HANDOFF_STATE.load(Ordering::Acquire);
        let before_epoch = CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire);
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_durable = DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire);
        let before_errors = CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire);
        let entered = Arc::new(TimedGate::new());
        let release = Arc::new(TimedGate::new());
        let completed = Arc::new(TimedGate::new());
        let worker_probe = Arc::new(AtomicUsize::new(0));
        let probe = Arc::clone(&worker_probe);
        let (evidence_tx, evidence_rx) = std::sync::mpsc::channel();
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::spawn(move || {
            let _completed = CompletionSignal(worker_completed);
            *HANDOFF_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ReapPauseHook {
                thread: std::thread::current().id(),
                entered: worker_entered,
                release: worker_release,
                panic_after_signal: true,
            });
            let (cleanup, owner, raw) = live_queue_cleanup(probe, CleanupDisposition::Poisoned);
            evidence_tx
                .send((raw.as_ptr() as usize, Arc::strong_count(&owner)))
                .expect("panic handoff ownership evidence receiver must remain live");
            enqueue_cleanup(cleanup);
        });
        let mut worker = TestWorker::new(worker, &completed, &[&release]);
        let (raw_address, strong_count) = evidence_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("panic handoff worker must publish live ownership evidence");
        wait_for_gate(&entered, &mut worker, "panic handoff hook");
        let result = join_worker(worker, "panic handoff worker");
        assert!(
            result.is_err(),
            "the real pause hook must panic in the worker"
        );
        assert_eq!(CLEANUP_HANDOFF_STATE.load(Ordering::Acquire), before_state);
        assert_eq!(CLEANUP_HANDOFF_EPOCH.load(Ordering::Acquire), before_epoch);
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            DURABLE_QUARANTINE_ITEMS.load(Ordering::Acquire),
            before_durable + 1
        );
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors
        );
        assert_ne!(raw_address, 0);
        assert_eq!(strong_count, 2);
        assert_eq!(worker_probe.load(Ordering::Acquire), 0);
        assert!(
            REAP_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_none()
        );
        assert!(
            HANDOFF_PAUSE_HOOK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_none()
        );
        assert!(
            FORCE_RETRY_THREAD
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_none()
        );
    }

    #[test]
    fn terminal_success_decrements_one_accepted_ticket_exactly_once() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _state = CleanupTestStateGuard::new();
        let before_pending = PENDING_CLEANUP_ITEMS.load(Ordering::Acquire);
        let before_errors = CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire);
        /* The host stub deliberately never creates a native context or
         * completion, so it cannot provide a genuine native successful
         * release lifecycle here.  This test covers the host-visible
         * accounting terminal path; the production-TU native lifecycle suite
         * in native/hip/tests/public_runtime_host_test.cpp is the relevant
         * success evidence for actual handles. */
        let record = match CleanupRecord::accepted(PendingCleanup::Context {
            raw: None,
            disposition: CleanupDisposition::Recoverable,
        }) {
            Ok(record) => record,
            Err(_) => panic!("test cleanup ticket must be accepted"),
        };
        assert!(matches!(record.try_once(), CleanupAttempt::Complete));
        finish_pending_cleanup();
        assert_eq!(
            PENDING_CLEANUP_ITEMS.load(Ordering::Acquire),
            before_pending
        );
        assert_eq!(
            CLEANUP_ACCOUNTING_ERRORS.load(Ordering::Acquire),
            before_errors
        );
    }

    #[test]
    fn kv_state_and_snapshot_cleanup_have_no_host_fallback_path() {
        let _serial = CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _state = CleanupTestStateGuard::new();
        let context = Context::test_without_native();
        let state_record = match CleanupRecord::accepted(PendingCleanup::KvState {
            raw: None,
            context: context.clone(),
            disposition: CleanupDisposition::Recoverable,
        }) {
            Ok(record) => record,
            Err(_) => panic!("KV state cleanup ticket must be accepted"),
        };
        assert!(matches!(state_record.try_once(), CleanupAttempt::Complete));
        finish_pending_cleanup();
        let view_record = match CleanupRecord::accepted(PendingCleanup::KvView {
            raw: None,
            context,
            disposition: CleanupDisposition::Recoverable,
        }) {
            Ok(record) => record,
            Err(_) => panic!("KV snapshot cleanup ticket must be accepted"),
        };
        assert!(matches!(view_record.try_once(), CleanupAttempt::Complete));
        finish_pending_cleanup();
    }

    #[test]
    fn host_stub_is_unavailable_without_cpu_success() {
        let result = Context::device_count();
        assert!(matches!(
            result,
            Err(error)
                if error.status() == RuntimeStatus::HipUnavailable
                    && error.message().contains("unavailable")
        ));
    }

    #[test]
    fn context_selection_rejects_empty_arch_before_ffi() {
        let result = Context::create(0, "");
        assert!(matches!(
            result,
            Err(error) if error.status() == RuntimeStatus::InvalidArgument
        ));
    }

    #[test]
    fn host_public_abi_fails_closed_for_null_size_version_reserved_and_unavailable() {
        let mut message = [0_u8; ERROR_CAPACITY];
        let mut error_sink = raw_sink(&mut message);
        let mut handle = std::ptr::null_mut();
        let raw =
            unsafe { sys::sllm_context_create(std::ptr::null(), &mut handle, &mut error_sink) };
        assert_eq!(RuntimeStatus::from_raw(raw), RuntimeStatus::InvalidArgument);
        assert!(handle.is_null());

        let mut info = context_info();
        info.struct_size -= 1;
        let raw = unsafe { sys::sllm_context_create(&info, &mut handle, &mut error_sink) };
        assert_eq!(RuntimeStatus::from_raw(raw), RuntimeStatus::InvalidArgument);
        assert!(handle.is_null());

        let mut info = context_info();
        info.abi_version += 1;
        let raw = unsafe { sys::sllm_context_create(&info, &mut handle, &mut error_sink) };
        assert_eq!(
            RuntimeStatus::from_raw(raw),
            RuntimeStatus::InvalidAbiVersion
        );
        assert!(handle.is_null());

        let mut info = context_info();
        info.reserved[0] = 1;
        let raw = unsafe { sys::sllm_context_create(&info, &mut handle, &mut error_sink) };
        assert_eq!(RuntimeStatus::from_raw(raw), RuntimeStatus::ReservedNonzero);
        assert!(handle.is_null());

        let info = context_info();
        let raw = unsafe { sys::sllm_context_create(&info, &mut handle, &mut error_sink) };
        assert_eq!(RuntimeStatus::from_raw(raw), RuntimeStatus::HipUnavailable);
        assert!(handle.is_null());
    }

    #[test]
    fn host_public_abi_reports_truncation_and_rejects_stale_double_release() {
        let mut message = [0_u8; 1];
        let mut error_sink = raw_sink(&mut message);
        let mut count = u32::MAX;
        let raw = unsafe { sys::sllm_device_count(&mut count, &mut error_sink) };
        assert_eq!(RuntimeStatus::from_raw(raw), RuntimeStatus::BufferTooSmall);
        assert_eq!(count, 0);
        assert!(error_sink.message_length > 1);

        let mut fake = std::ptr::dangling_mut::<sys::sllm_context_t>();
        let mut full_message = [0_u8; ERROR_CAPACITY];
        let mut full_sink = raw_sink(&mut full_message);
        let raw = unsafe { sys::sllm_context_release(&mut fake, &mut full_sink) };
        assert_eq!(RuntimeStatus::from_raw(raw), RuntimeStatus::InvalidHandle);
        let raw = unsafe { sys::sllm_context_release(&mut fake, &mut full_sink) };
        assert_eq!(RuntimeStatus::from_raw(raw), RuntimeStatus::InvalidHandle);
    }
}
