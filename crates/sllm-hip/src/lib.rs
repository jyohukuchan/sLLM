//! Safe Rust access to the versioned HIP ABI.
//!
//! The legacy [`sllm_core::Backend`] methods remain a Phase 1 control-plane
//! boundary and retain their unavailable/non-executing contract.  The
//! additive owned execution session is the typed
//! typed `Context`/`Buffer`/prepared semantic-operation paths below; it does
//! not change the meaning of those legacy methods.

use std::fmt;
use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Duration;

use sllm_core::{
    Backend, BackendCapabilities, BackendError, BackendSupport, ExecutionError, ExecutionReceipt,
    ExecutionSession, ExecutionSessionRequest, MaterializedTensor, SemanticOp, TensorView,
};
use sllm_hip_sys as sys;

mod argmax;
mod attention_preprocess;
mod bridge;
mod elementwise;
mod embedding;
mod kv_state;
mod linear_attention;
mod matmul;
mod rmsnorm;
mod rotary;
mod runtime;
mod windowed_attention;

pub use argmax::{ArgmaxDescriptor, ArgmaxDispatchInfo, ArgmaxSubmission, PreparedArgmax};
pub use attention_preprocess::{
    AttentionPreprocessDescriptor, AttentionPreprocessDispatchInfo, AttentionPreprocessSubmission,
    PreparedAttentionPreprocess,
};
pub use elementwise::{
    ElementwiseDescriptor, ElementwiseDispatchInfo, ElementwiseOperation, ElementwiseSubmission,
    PreparedElementwise,
};
pub use embedding::{
    EmbeddingDescriptor, EmbeddingDispatchInfo, EmbeddingSubmission, PreparedEmbedding,
};
pub use kv_state::{
    CausalAttentionEvidence, KvAppendEvidence, bf16_to_f16_bits, expected_storage_offset,
};
pub use matmul::{MatmulDescriptor, MatmulDispatchInfo, MatmulSubmission, PreparedMatmul};
pub use rmsnorm::{
    PreparedRmsNorm, RmsNormDescriptor, RmsNormDispatchInfo, RmsNormSubmission, TensorBinding,
};
pub use rotary::{PreparedRotary, RotaryDescriptor, RotaryDispatchInfo, RotarySubmission};
pub use runtime::{
    Buffer, Completion, CompletionState, Context, DeviceInfo, Event, Queue, RuntimeError,
    RuntimeStatus,
};
pub use windowed_attention::{
    PreparedWindowedAttention, WindowedAttentionDescriptor, WindowedAttentionDispatchInfo,
    WindowedAttentionSubmission,
};

const ERROR_CAPACITY: usize = 256;
const MAX_FINITE_TIMEOUT_MS: u32 = u32::MAX - 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    Ok,
    InvalidArgument,
    BufferTooSmall,
    Unsupported,
    HipUnavailable,
    InvalidAbiVersion,
    ReservedNonzero,
    InternalError,
    Timeout,
    InvalidHandle,
    ZeroDispatch,
    RuntimeError,
    DispatchContract,
    Unknown(u32),
}

impl Status {
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
            sys::evidence::SLLM_STATUS_HIP_TIMEOUT => Self::Timeout,
            sys::evidence::SLLM_STATUS_HIP_INVALID_HANDLE => Self::InvalidHandle,
            sys::evidence::SLLM_STATUS_HIP_ZERO_DISPATCH => Self::ZeroDispatch,
            sys::evidence::SLLM_STATUS_HIP_RUNTIME_ERROR => Self::RuntimeError,
            sys::evidence::SLLM_STATUS_HIP_DISPATCH_CONTRACT => Self::DispatchContract,
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
            Self::Timeout => sys::evidence::SLLM_STATUS_HIP_TIMEOUT,
            Self::InvalidHandle => sys::evidence::SLLM_STATUS_HIP_INVALID_HANDLE,
            Self::ZeroDispatch => sys::evidence::SLLM_STATUS_HIP_ZERO_DISPATCH,
            Self::RuntimeError => sys::evidence::SLLM_STATUS_HIP_RUNTIME_ERROR,
            Self::DispatchContract => sys::evidence::SLLM_STATUS_HIP_DISPATCH_CONTRACT,
            Self::Unknown(raw) => raw,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HipError {
    Status { status: Status, message: String },
    HipUnavailable { message: String },
}

impl HipError {
    pub fn status(&self) -> Status {
        match self {
            Self::Status { status, .. } => *status,
            Self::HipUnavailable { .. } => Status::HipUnavailable,
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Status { message, .. } | Self::HipUnavailable { message } => message,
        }
    }
}

impl fmt::Display for HipError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}: {}",
            status_name(self.status()),
            self.message()
        )
    }
}

impl std::error::Error for HipError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Version {
    pub abi_version: u32,
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendProbe {
    pub backend: u32,
    pub available: bool,
    pub hip_runtime_present: bool,
    pub diagnostic: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContextProbe {
    pub context_present: bool,
    pub hip_available: bool,
    pub diagnostic: String,
}

fn status_name(status: Status) -> &'static str {
    match status {
        Status::Ok => "ok",
        Status::InvalidArgument => "invalid argument",
        Status::BufferTooSmall => "buffer too small",
        Status::Unsupported => "unsupported",
        Status::HipUnavailable => "HIP unavailable",
        Status::InvalidAbiVersion => "invalid ABI version",
        Status::ReservedNonzero => "reserved field is non-zero",
        Status::InternalError => "internal error",
        Status::Timeout => "HIP evidence timeout",
        Status::InvalidHandle => "invalid HIP evidence handle",
        Status::ZeroDispatch => "HIP evidence performed zero dispatches",
        Status::RuntimeError => "HIP runtime error",
        Status::DispatchContract => "HIP evidence dispatch contract violation",
        Status::Unknown(_) => "unknown native status",
    }
}

fn sink(buffer: &mut [u8; ERROR_CAPACITY]) -> sys::sllm_error_sink_t {
    sys::sllm_error_sink_t {
        struct_size: size_of::<sys::sllm_error_sink_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        message: buffer.as_mut_ptr().cast(),
        message_capacity: buffer.len() as u64,
        message_length: 0,
        reserved: [0, 0],
    }
}

fn diagnostic(buffer: &[u8; ERROR_CAPACITY], length: u64) -> String {
    let length = usize::try_from(length)
        .unwrap_or(buffer.len())
        .min(buffer.len().saturating_sub(1));
    String::from_utf8_lossy(&buffer[..length]).into_owned()
}

fn error_from_raw(raw: u32, message: String) -> HipError {
    let status = Status::from_raw(raw);
    if status == Status::HipUnavailable {
        HipError::HipUnavailable { message }
    } else {
        HipError::Status { status, message }
    }
}

fn ensure_ok(raw: u32, message: String) -> Result<(), HipError> {
    if Status::from_raw(raw) == Status::Ok {
        Ok(())
    } else {
        Err(error_from_raw(raw, message))
    }
}

pub fn abi_version() -> Result<u32, HipError> {
    let mut buffer = [0u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut buffer);
    let mut version = 0;
    let raw = unsafe { sys::sllm_get_abi_version(&mut version, &mut error_sink) };
    ensure_ok(raw, diagnostic(&buffer, error_sink.message_length)).map(|()| version)
}

pub fn version() -> Result<Version, HipError> {
    let mut buffer = [0u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut buffer);
    let mut info = sys::sllm_version_info_t {
        struct_size: size_of::<sys::sllm_version_info_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        major: 0,
        minor: 0,
        patch: 0,
        reserved: [0; 3],
    };
    let raw = unsafe { sys::sllm_query_version(&mut info, &mut error_sink) };
    ensure_ok(raw, diagnostic(&buffer, error_sink.message_length)).map(|()| Version {
        abi_version: info.abi_version,
        major: info.major,
        minor: info.minor,
        patch: info.patch,
    })
}

pub fn backend_probe() -> Result<BackendProbe, HipError> {
    let mut buffer = [0u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut buffer);
    let mut result = sys::sllm_backend_probe_result_t {
        struct_size: size_of::<sys::sllm_backend_probe_result_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        backend: 0,
        available: 0,
        hip_runtime_present: 0,
        reserved: [0; 3],
    };
    let raw =
        unsafe { sys::sllm_backend_probe(sys::SLLM_BACKEND_HIP, &mut result, &mut error_sink) };
    let status = Status::from_raw(raw);
    if status != Status::Ok && status != Status::HipUnavailable {
        return Err(error_from_raw(
            raw,
            diagnostic(&buffer, error_sink.message_length),
        ));
    }
    Ok(BackendProbe {
        backend: result.backend,
        available: result.available != 0,
        hip_runtime_present: result.hip_runtime_present != 0,
        diagnostic: diagnostic(&buffer, error_sink.message_length),
    })
}

pub fn context_probe() -> Result<ContextProbe, HipError> {
    let mut buffer = [0u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut buffer);
    let mut result = sys::sllm_context_probe_result_t {
        struct_size: size_of::<sys::sllm_context_probe_result_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        context_present: 0,
        hip_available: 0,
        reserved: [0; 4],
    };
    let raw = unsafe { sys::sllm_context_probe(std::ptr::null(), &mut result, &mut error_sink) };
    let status = Status::from_raw(raw);
    if status != Status::Ok && status != Status::HipUnavailable {
        return Err(error_from_raw(
            raw,
            diagnostic(&buffer, error_sink.message_length),
        ));
    }
    Ok(ContextProbe {
        context_present: result.context_present != 0,
        hip_available: result.hip_available != 0,
        diagnostic: diagnostic(&buffer, error_sink.message_length),
    })
}

/// Independent Rust oracle for the private model-free diagnostic transform.
/// This is evidence plumbing, not a semantic operation or a CPU fallback.
pub const EVIDENCE_EXPECTED_XOR: u8 = 0x5a;
pub const EVIDENCE_CASE_SIZES: [usize; 6] = [1, 3, 17, 255, 256, 257];

pub fn expected_evidence_output(input: &[u8]) -> Vec<u8> {
    input
        .iter()
        .map(|value| *value ^ EVIDENCE_EXPECTED_XOR)
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceReport {
    pub output_size: usize,
    pub allocation_count: u64,
    pub copy_count: u64,
    pub dispatch_count: u64,
    pub selected_backend: u32,
    pub fallback_used: bool,
    pub terminal: bool,
}

pub struct EvidenceCompletion {
    handle: Option<NonNull<sys::evidence::sllm_hip_evidence_completion_t>>,
}

impl fmt::Debug for EvidenceCompletion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EvidenceCompletion")
            .finish_non_exhaustive()
    }
}

impl EvidenceCompletion {
    pub fn wait(
        &mut self,
        output: &mut [u8],
        timeout: Duration,
    ) -> Result<EvidenceReport, HipError> {
        if self.handle.is_none() {
            return Err(HipError::Status {
                status: Status::InvalidHandle,
                message: "evidence completion was already consumed".to_owned(),
            });
        }
        let timeout_ms = timeout_millis(timeout);
        let output_capacity = u64::try_from(output.len()).map_err(|_| HipError::Status {
            status: Status::InvalidArgument,
            message: "evidence output is too large for the HIP ABI".to_owned(),
        })?;
        let handle = match self.handle.take() {
            Some(handle) => handle,
            None => {
                return Err(HipError::Status {
                    status: Status::InvalidHandle,
                    message: "evidence completion was already consumed".to_owned(),
                });
            }
        };
        let mut error_buffer = [0u8; ERROR_CAPACITY];
        let mut error_sink = evidence_sink(&mut error_buffer);
        let mut result = evidence_result();
        let raw = unsafe {
            sys::evidence::sllm_hip_evidence_wait(
                handle.as_ptr(),
                timeout_ms,
                output.as_mut_ptr(),
                output_capacity,
                &mut result,
                &mut error_sink,
            )
        };
        ensure_ok(raw, diagnostic(&error_buffer, error_sink.message_length))?;
        let output_size = usize::try_from(result.output_size).map_err(|_| HipError::Status {
            status: Status::RuntimeError,
            message: "HIP evidence result size does not fit in usize".to_owned(),
        })?;
        Ok(EvidenceReport {
            output_size,
            allocation_count: result.allocation_count,
            copy_count: result.copy_count,
            dispatch_count: result.dispatch_count,
            selected_backend: result.selected_backend,
            fallback_used: result.fallback_used != 0,
            terminal: result.terminal != 0,
        })
    }
}

impl Drop for EvidenceCompletion {
    fn drop(&mut self) {
        let Some(handle) = self.handle.take() else {
            return;
        };
        let mut raw_handle = handle.as_ptr();
        let mut error_buffer = [0u8; ERROR_CAPACITY];
        let mut error_sink = evidence_sink(&mut error_buffer);
        let _ =
            unsafe { sys::evidence::sllm_hip_evidence_destroy(&mut raw_handle, &mut error_sink) };
    }
}

pub fn submit_evidence(input: &[u8]) -> Result<EvidenceCompletion, HipError> {
    if input.is_empty() {
        return Err(HipError::Status {
            status: Status::InvalidArgument,
            message: "evidence input must not be empty".to_owned(),
        });
    }
    let input_size = u64::try_from(input.len()).map_err(|_| HipError::Status {
        status: Status::InvalidArgument,
        message: "evidence input is too large for the HIP ABI".to_owned(),
    })?;
    let request = sys::evidence::sllm_hip_evidence_request_t {
        struct_size: size_of::<sys::evidence::sllm_hip_evidence_request_t>() as u32,
        abi_version: sys::evidence::SLLM_HIP_EVIDENCE_ABI_VERSION,
        input: input.as_ptr(),
        input_size,
        reserved: [0; 4],
    };
    let mut handle = std::ptr::null_mut();
    let mut error_buffer = [0u8; ERROR_CAPACITY];
    let mut error_sink = evidence_sink(&mut error_buffer);
    let raw =
        unsafe { sys::evidence::sllm_hip_evidence_submit(&request, &mut handle, &mut error_sink) };
    ensure_ok(raw, diagnostic(&error_buffer, error_sink.message_length))?;
    NonNull::new(handle)
        .map(|handle| EvidenceCompletion {
            handle: Some(handle),
        })
        .ok_or_else(|| HipError::Status {
            status: Status::InternalError,
            message: "native evidence submit returned a null completion".to_owned(),
        })
}

fn timeout_millis(timeout: Duration) -> u32 {
    let millis = timeout.as_millis();
    if millis >= u128::from(MAX_FINITE_TIMEOUT_MS) {
        MAX_FINITE_TIMEOUT_MS
    } else {
        u32::try_from(millis).unwrap_or(MAX_FINITE_TIMEOUT_MS)
    }
}

fn validate_evidence_contract(
    input: &[u8],
    output: &[u8],
    report: &EvidenceReport,
) -> Result<(), HipError> {
    if report.dispatch_count == 0 {
        return Err(HipError::Status {
            status: Status::ZeroDispatch,
            message: "evidence result contains zero kernel dispatches".to_owned(),
        });
    }
    if report.selected_backend != sys::SLLM_BACKEND_HIP
        || report.fallback_used
        || !report.terminal
        || report.output_size != input.len()
        || report.allocation_count != 2
        || report.copy_count != 2
        || report.dispatch_count != 1
        || output != expected_evidence_output(input)
    {
        return Err(HipError::Status {
            status: Status::DispatchContract,
            message: "evidence result failed the exact no-fallback byte contract".to_owned(),
        });
    }
    Ok(())
}

pub fn run_evidence(
    input: &[u8],
    timeout: Duration,
) -> Result<(Vec<u8>, EvidenceReport), HipError> {
    let mut completion = submit_evidence(input)?;
    let mut output = vec![0u8; input.len()];
    let report = completion.wait(&mut output, timeout)?;
    validate_evidence_contract(input, &output, &report)?;
    Ok((output, report))
}

fn evidence_sink(buffer: &mut [u8; ERROR_CAPACITY]) -> sys::evidence::sllm_error_sink_t {
    sys::evidence::sllm_error_sink_t {
        struct_size: size_of::<sys::evidence::sllm_error_sink_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        message: buffer.as_mut_ptr().cast(),
        message_capacity: buffer.len() as u64,
        message_length: 0,
        reserved: [0, 0],
    }
}

fn evidence_result() -> sys::evidence::sllm_hip_evidence_result_t {
    sys::evidence::sllm_hip_evidence_result_t {
        struct_size: size_of::<sys::evidence::sllm_hip_evidence_result_t>() as u32,
        abi_version: sys::evidence::SLLM_HIP_EVIDENCE_ABI_VERSION,
        output_size: 0,
        allocation_count: 0,
        copy_count: 0,
        dispatch_count: 0,
        selected_backend: 0,
        fallback_used: 0,
        terminal: 0,
        reserved: [0; 4],
    }
}

/// A typed backend handle is only constructible after an available HIP probe.
/// Its legacy [`Backend`] methods remain the Phase 1 unavailable contract;
/// owned numerical work is opened separately through `open_execution_session`.
#[derive(Clone, Copy, Debug)]
pub struct HipBackend {
    _private: (),
}

impl HipBackend {
    pub fn connect() -> Result<Self, HipError> {
        let probe = backend_probe()?;
        if !probe.available {
            return Err(HipError::HipUnavailable {
                message: probe.diagnostic,
            });
        }
        Ok(Self { _private: () })
    }
}

/// Evidence-binary-only bridge to the private native KV readback ABI.
///
/// This is not a public runtime capability: it accepts only a core-issued
/// session/state pair and copies into caller-owned host storage. The native
/// implementation retains all opaque-handle, snapshot, and lifetime checks.
#[doc(hidden)]
pub fn read_kv_storage_for_evidence(
    session: &ExecutionSession,
    state: &sllm_core::KvState,
    plane: u32,
    byte_offset: u64,
    destination: &mut [u8],
) -> Result<(), ExecutionError> {
    if session.backend_name() != "hip" || state.backend_name() != "hip" {
        return Err(ExecutionError::WrongBackend {
            expected: "hip",
            actual: state.backend_name(),
        });
    }
    if state.session_id() != session.id() {
        return Err(ExecutionError::WrongSession {
            expected: session.id(),
            actual: state.session_id(),
        });
    }
    session.max_transfer_bytes()?;
    let resource = kv_state::resource_for_evidence(session.id().raw(), state.id().raw())
        .ok_or_else(|| ExecutionError::BackendStatus {
            status: sys::SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            diagnostic: "HIP KV evidence state is stale or unavailable".to_owned(),
        })?;
    resource
        .readback(plane, byte_offset, destination)
        .map_err(map_evidence_runtime_error)
}

fn map_evidence_runtime_error(error: RuntimeError) -> ExecutionError {
    match error.status() {
        RuntimeStatus::HipUnavailable => ExecutionError::ExecutionUnavailable {
            backend: "hip",
            reason: error.message().to_owned(),
        },
        RuntimeStatus::Busy => ExecutionError::Busy,
        RuntimeStatus::NotReady => ExecutionError::NotReady,
        _ => ExecutionError::BackendStatus {
            status: error.status().raw(),
            diagnostic: error.message().to_owned(),
        },
    }
}

impl Backend for HipBackend {
    fn name(&self) -> &'static str {
        "hip"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            metadata_only: false,
            numerical_execution: false,
            max_materialization_bytes: None,
        }
    }

    fn supports(&self, _operation: &SemanticOp) -> BackendSupport {
        BackendSupport::Unsupported {
            reason: "HIP backend is unavailable in Phase 1",
        }
    }

    fn materialize(&self, _view: &TensorView) -> Result<MaterializedTensor, BackendError> {
        Err(BackendError::BackendUnavailable { name: self.name() })
    }

    fn execute(&self, _operation: &SemanticOp) -> Result<ExecutionReceipt, BackendError> {
        Err(BackendError::BackendUnavailable { name: self.name() })
    }

    fn open_execution_session(
        &self,
        request: ExecutionSessionRequest,
    ) -> Result<Arc<ExecutionSession>, ExecutionError> {
        bridge::open_execution_session(*self, request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sink_for(buffer: &mut [u8]) -> sys::sllm_error_sink_t {
        sys::sllm_error_sink_t {
            struct_size: size_of::<sys::sllm_error_sink_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            message: buffer.as_mut_ptr().cast(),
            message_capacity: buffer.len() as u64,
            message_length: 0,
            reserved: [0, 0],
        }
    }

    fn backend_result() -> sys::sllm_backend_probe_result_t {
        sys::sllm_backend_probe_result_t {
            struct_size: size_of::<sys::sllm_backend_probe_result_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            backend: 0,
            available: 0,
            hip_runtime_present: 0,
            reserved: [0; 3],
        }
    }

    fn version_info() -> sys::sllm_version_info_t {
        sys::sllm_version_info_t {
            struct_size: size_of::<sys::sllm_version_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            major: 0,
            minor: 0,
            patch: 0,
            reserved: [0; 3],
        }
    }

    fn context_probe_result() -> sys::sllm_context_probe_result_t {
        sys::sllm_context_probe_result_t {
            struct_size: size_of::<sys::sllm_context_probe_result_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            context_present: 0,
            hip_available: 0,
            reserved: [0; 4],
        }
    }

    #[test]
    fn status_mapping_preserves_unknown_codes() {
        assert_eq!(
            Status::from_raw(sys::SLLM_STATUS_HIP_UNAVAILABLE),
            Status::HipUnavailable
        );
        assert_eq!(
            Status::from_raw(sys::evidence::SLLM_STATUS_HIP_ZERO_DISPATCH),
            Status::ZeroDispatch
        );
        assert_eq!(
            Status::ZeroDispatch.raw(),
            sys::evidence::SLLM_STATUS_HIP_ZERO_DISPATCH
        );
        assert_eq!(Status::from_raw(99), Status::Unknown(99));
    }

    #[test]
    fn evidence_oracle_covers_required_non_aligned_boundaries() {
        assert_eq!(EVIDENCE_CASE_SIZES, [1, 3, 17, 255, 256, 257]);
        for size in EVIDENCE_CASE_SIZES {
            let input: Vec<u8> = (0..size).map(|index| index as u8).collect();
            let expected = expected_evidence_output(&input);
            assert_eq!(expected.len(), size);
            assert_eq!(expected[0], input[0] ^ EVIDENCE_EXPECTED_XOR);
            assert_eq!(expected[size - 1], input[size - 1] ^ EVIDENCE_EXPECTED_XOR);
        }
    }

    #[test]
    fn evidence_contract_rejects_zero_and_multiple_dispatches() {
        let input = [1_u8, 3, 17];
        let output = expected_evidence_output(&input);
        let mut report = EvidenceReport {
            output_size: input.len(),
            allocation_count: 2,
            copy_count: 2,
            dispatch_count: 1,
            selected_backend: sys::SLLM_BACKEND_HIP,
            fallback_used: false,
            terminal: true,
        };
        assert!(validate_evidence_contract(&input, &output, &report).is_ok());
        report.dispatch_count = 0;
        let error = validate_evidence_contract(&input, &output, &report)
            .expect_err("zero dispatch must fail closed");
        assert_eq!(error.status(), Status::ZeroDispatch);
        report.dispatch_count = 2;
        let error = validate_evidence_contract(&input, &output, &report)
            .expect_err("non-exact dispatch count must fail");
        assert_eq!(error.status(), Status::DispatchContract);
    }

    #[test]
    fn timeout_conversion_is_bounded_and_zero_is_an_immediate_poll() {
        assert_eq!(timeout_millis(Duration::ZERO), 0);
        assert_eq!(timeout_millis(Duration::from_nanos(999_999)), 0);
        assert_eq!(timeout_millis(Duration::from_millis(1)), 1);
        assert_eq!(
            timeout_millis(Duration::from_millis(u64::from(u32::MAX) + 1)),
            u32::MAX - 1
        );
        assert_eq!(
            timeout_millis(Duration::from_millis(u64::from(u32::MAX))),
            u32::MAX - 1
        );
    }

    #[test]
    fn private_evidence_abi_layout_and_constants_are_explicit() {
        use std::mem::{align_of, offset_of};

        type Request = sys::evidence::sllm_hip_evidence_request_t;
        type Result = sys::evidence::sllm_hip_evidence_result_t;
        type KvReadback = sys::evidence::sllm_hip_kv_readback_request_t;
        assert_eq!(size_of::<Request>(), 40);
        assert_eq!(align_of::<Request>(), 8);
        assert_eq!(offset_of!(Request, struct_size), 0);
        assert_eq!(offset_of!(Request, abi_version), 4);
        assert_eq!(offset_of!(Request, input), 8);
        assert_eq!(offset_of!(Request, input_size), 16);
        assert_eq!(offset_of!(Request, reserved), 24);
        assert_eq!(size_of::<Result>(), 72);
        assert_eq!(align_of::<Result>(), 8);
        assert_eq!(offset_of!(Result, struct_size), 0);
        assert_eq!(offset_of!(Result, abi_version), 4);
        assert_eq!(offset_of!(Result, output_size), 8);
        assert_eq!(offset_of!(Result, allocation_count), 16);
        assert_eq!(offset_of!(Result, copy_count), 24);
        assert_eq!(offset_of!(Result, dispatch_count), 32);
        assert_eq!(offset_of!(Result, selected_backend), 40);
        assert_eq!(offset_of!(Result, fallback_used), 44);
        assert_eq!(offset_of!(Result, terminal), 48);
        assert_eq!(offset_of!(Result, reserved), 52);
        assert_eq!(size_of::<KvReadback>(), 72);
        assert_eq!(align_of::<KvReadback>(), 8);
        assert_eq!(offset_of!(KvReadback, struct_size), 0);
        assert_eq!(offset_of!(KvReadback, abi_version), 4);
        assert_eq!(offset_of!(KvReadback, view), 8);
        assert_eq!(offset_of!(KvReadback, plane), 16);
        assert_eq!(offset_of!(KvReadback, byte_offset), 24);
        assert_eq!(offset_of!(KvReadback, byte_length), 32);
        assert_eq!(offset_of!(KvReadback, host_capacity), 40);
        assert_eq!(offset_of!(KvReadback, host_output), 48);
        assert_eq!(offset_of!(KvReadback, reserved), 56);
        type ErrorSink = sys::evidence::sllm_error_sink_t;
        assert_eq!(size_of::<ErrorSink>(), 48);
        assert_eq!(align_of::<ErrorSink>(), 8);
        assert_eq!(offset_of!(ErrorSink, struct_size), 0);
        assert_eq!(offset_of!(ErrorSink, abi_version), 4);
        assert_eq!(offset_of!(ErrorSink, message), 8);
        assert_eq!(offset_of!(ErrorSink, message_capacity), 16);
        assert_eq!(offset_of!(ErrorSink, message_length), 24);
        assert_eq!(offset_of!(ErrorSink, reserved), 32);
        assert_eq!(sys::evidence::SLLM_HIP_EVIDENCE_ABI_VERSION, 1);
        assert_eq!(sys::evidence::SLLM_HIP_EVIDENCE_TRANSFORM_XOR, 0x5a);
        assert_eq!(sys::evidence::SLLM_STATUS_HIP_TIMEOUT, 8);
        assert_eq!(sys::evidence::SLLM_STATUS_HIP_INVALID_HANDLE, 9);
        assert_eq!(sys::evidence::SLLM_STATUS_HIP_ZERO_DISPATCH, 10);
        assert_eq!(sys::evidence::SLLM_STATUS_HIP_DISPATCH_CONTRACT, 12);
        assert_eq!(sys::evidence::SLLM_HIP_KV_EVIDENCE_ABI_VERSION, 1);
        assert_eq!(sys::evidence::SLLM_HIP_KV_EVIDENCE_PLANE_K, 0);
        assert_eq!(sys::evidence::SLLM_HIP_KV_EVIDENCE_PLANE_V, 1);
        assert_eq!(
            sys::evidence::SLLM_HIP_KV_EVIDENCE_MAX_READBACK_BYTES,
            536_870_912
        );
    }

    #[test]
    fn host_stub_evidence_is_unavailable_and_never_cpu_fallback() {
        let input = [1_u8, 3, 17, 255, 0, 257_u16 as u8];
        let error = run_evidence(&input, Duration::from_millis(100)).expect_err("stub must fail");
        assert_eq!(error.status(), Status::HipUnavailable);
        assert!(error.message().contains("CPU fallback"));
        assert!(submit_evidence(&[]).is_err());
    }

    #[test]
    fn private_evidence_abi_validates_request_and_handle_inputs_on_stub() {
        let mut message = [0_u8; ERROR_CAPACITY];
        let mut error_sink = evidence_sink(&mut message);
        let mut handle = std::ptr::null_mut();
        let input = [1_u8, 2, 3];
        let mut request = sys::evidence::sllm_hip_evidence_request_t {
            struct_size: size_of::<sys::evidence::sllm_hip_evidence_request_t>() as u32,
            abi_version: sys::evidence::SLLM_HIP_EVIDENCE_ABI_VERSION,
            input: input.as_ptr(),
            input_size: input.len() as u64,
            reserved: [0; 4],
        };

        let raw = unsafe {
            sys::evidence::sllm_hip_evidence_submit(std::ptr::null(), &mut handle, &mut error_sink)
        };
        assert_eq!(Status::from_raw(raw), Status::InvalidArgument);

        request.struct_size -= 1;
        let raw = unsafe {
            sys::evidence::sllm_hip_evidence_submit(&request, &mut handle, &mut error_sink)
        };
        assert_eq!(Status::from_raw(raw), Status::InvalidArgument);

        request.struct_size = size_of::<sys::evidence::sllm_hip_evidence_request_t>() as u32;
        request.abi_version += 1;
        let raw = unsafe {
            sys::evidence::sllm_hip_evidence_submit(&request, &mut handle, &mut error_sink)
        };
        assert_eq!(Status::from_raw(raw), Status::InvalidAbiVersion);

        request.abi_version = sys::evidence::SLLM_HIP_EVIDENCE_ABI_VERSION;
        request.reserved[0] = 1;
        let raw = unsafe {
            sys::evidence::sllm_hip_evidence_submit(&request, &mut handle, &mut error_sink)
        };
        assert_eq!(Status::from_raw(raw), Status::ReservedNonzero);

        let raw = unsafe {
            sys::evidence::sllm_hip_evidence_wait(
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                &mut error_sink,
            )
        };
        assert_eq!(Status::from_raw(raw), Status::InvalidHandle);

        let raw = unsafe {
            sys::evidence::sllm_hip_evidence_destroy(std::ptr::null_mut(), &mut error_sink)
        };
        assert_eq!(Status::from_raw(raw), Status::InvalidArgument);

        let mut stale = std::ptr::null_mut();
        let raw = unsafe { sys::evidence::sllm_hip_evidence_destroy(&mut stale, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidHandle);

        let mut fake = std::ptr::dangling_mut::<sys::evidence::sllm_hip_evidence_completion_t>();
        let raw = unsafe { sys::evidence::sllm_hip_evidence_destroy(&mut fake, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidHandle);
        let raw = unsafe { sys::evidence::sllm_hip_evidence_destroy(&mut fake, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidHandle);
    }

    #[test]
    fn host_stub_reports_unavailability_without_fallback() {
        let probe = backend_probe().expect("unavailability is a probe result, not a call failure");
        assert!(!probe.available);
        assert!(!probe.hip_runtime_present);
        assert!(probe.diagnostic.contains("unavailable"));
        assert!(matches!(
            HipBackend::connect(),
            Err(HipError::HipUnavailable { .. })
        ));
    }

    #[test]
    fn generic_backend_preserves_legacy_contract_while_owned_session_is_additive() {
        let backend = HipBackend { _private: () };
        let activation =
            TensorView::contiguous(sllm_core::DType::Bf16, &[2, 3]).expect("valid activation");
        let scale = TensorView::contiguous(sllm_core::DType::Bf16, &[3]).expect("valid scale");
        let output = TensorView::contiguous(sllm_core::DType::Bf16, &[2, 3]).expect("valid output");
        let operation = sllm_core::SemanticOp::new_rms_norm(
            vec![activation.clone(), scale],
            vec![output],
            1.0e-6,
            sllm_core::RmsNormScaleMode::OffsetOne,
        )
        .expect("valid RMSNorm operation");

        assert!(!backend.capabilities().numerical_execution);
        assert_eq!(
            backend.supports(&operation),
            BackendSupport::Unsupported {
                reason: "HIP backend is unavailable in Phase 1",
            }
        );
        assert_eq!(
            backend.materialize(&activation),
            Err(BackendError::BackendUnavailable { name: "hip" })
        );
        assert_eq!(
            backend.execute(&operation),
            Err(BackendError::BackendUnavailable { name: "hip" })
        );
    }

    #[test]
    fn version_query_is_explicit_and_stable() {
        assert_eq!(abi_version().expect("version query succeeds"), 1);
        assert_eq!(
            version().expect("version query succeeds"),
            Version {
                abi_version: 1,
                major: 0,
                minor: 1,
                patch: 0,
            }
        );
        let probe = context_probe().expect("unavailability is a probe result");
        assert!(!probe.context_present);
        assert!(!probe.hip_available);
        assert!(probe.diagnostic.contains("unavailable"));
    }

    #[test]
    fn native_query_version_rejects_null_reserved_wrong_abi_and_undersized_output() {
        let mut message = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut message);

        let raw = unsafe { sys::sllm_query_version(std::ptr::null_mut(), &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidArgument);
        assert!(diagnostic(&message, error_sink.message_length).contains("version output"));

        let mut info = version_info();
        info.reserved[0] = 1;
        let raw = unsafe { sys::sllm_query_version(&mut info, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::ReservedNonzero);
        assert!(diagnostic(&message, error_sink.message_length).contains("reserved"));

        let mut info = version_info();
        info.abi_version += 1;
        let raw = unsafe { sys::sllm_query_version(&mut info, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidAbiVersion);
        assert!(diagnostic(&message, error_sink.message_length).contains("ABI"));

        let mut info = version_info();
        info.struct_size -= 1;
        let raw = unsafe { sys::sllm_query_version(&mut info, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidArgument);
        assert!(diagnostic(&message, error_sink.message_length).contains("struct_size"));
    }

    #[test]
    fn native_query_version_initializes_output_and_maps_success() {
        let mut message = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut message);
        let mut info = version_info();
        info.major = u32::MAX;
        info.minor = u32::MAX;
        info.patch = u32::MAX;

        let raw = unsafe { sys::sllm_query_version(&mut info, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::Ok);
        assert_eq!(info.abi_version, sys::SLLM_HIP_ABI_VERSION);
        assert_eq!(info.major, sys::SLLM_HIP_LIBRARY_VERSION_MAJOR);
        assert_eq!(info.minor, sys::SLLM_HIP_LIBRARY_VERSION_MINOR);
        assert_eq!(info.patch, sys::SLLM_HIP_LIBRARY_VERSION_PATCH);
        assert_eq!(info.reserved, [0; 3]);
        assert_eq!(error_sink.message_length, 0);
    }

    #[test]
    fn native_context_probe_rejects_null_reserved_wrong_abi_and_undersized_result() {
        let mut message = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut message);

        let raw = unsafe {
            sys::sllm_context_probe(std::ptr::null(), std::ptr::null_mut(), &mut error_sink)
        };
        assert_eq!(Status::from_raw(raw), Status::InvalidArgument);
        assert!(diagnostic(&message, error_sink.message_length).contains("context probe"));

        let mut result = context_probe_result();
        result.reserved[0] = 1;
        let raw =
            unsafe { sys::sllm_context_probe(std::ptr::null(), &mut result, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::ReservedNonzero);
        assert!(diagnostic(&message, error_sink.message_length).contains("reserved"));

        let mut result = context_probe_result();
        result.abi_version += 1;
        let raw =
            unsafe { sys::sllm_context_probe(std::ptr::null(), &mut result, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidAbiVersion);
        assert!(diagnostic(&message, error_sink.message_length).contains("ABI"));

        let mut result = context_probe_result();
        result.struct_size -= 1;
        let raw =
            unsafe { sys::sllm_context_probe(std::ptr::null(), &mut result, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidArgument);
        assert!(diagnostic(&message, error_sink.message_length).contains("struct_size"));
    }

    #[test]
    fn native_context_probe_initializes_null_context_output_and_maps_unavailable() {
        let mut message = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut message);
        let mut result = context_probe_result();
        result.context_present = u32::MAX;
        result.hip_available = u32::MAX;

        let raw =
            unsafe { sys::sllm_context_probe(std::ptr::null(), &mut result, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::HipUnavailable);
        assert_eq!(result.context_present, 0);
        assert_eq!(result.hip_available, 0);
        assert_eq!(result.reserved, [0; 4]);
        assert!(diagnostic(&message, error_sink.message_length).contains("unavailable"));
    }

    #[test]
    fn native_probe_rejects_nonzero_reserved_fields() {
        let mut buffer = [0u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut buffer);
        let mut result = sys::sllm_backend_probe_result_t {
            struct_size: size_of::<sys::sllm_backend_probe_result_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            backend: 0,
            available: 0,
            hip_runtime_present: 0,
            reserved: [1, 0, 0],
        };

        let raw =
            unsafe { sys::sllm_backend_probe(sys::SLLM_BACKEND_HIP, &mut result, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::ReservedNonzero);
        assert!(diagnostic(&buffer, error_sink.message_length).contains("reserved"));
    }

    #[test]
    fn native_abi_rejects_null_and_wrong_abi_inputs() {
        let raw = unsafe { sys::sllm_get_abi_version(std::ptr::null_mut(), std::ptr::null_mut()) };
        assert_eq!(Status::from_raw(raw), Status::InvalidArgument);

        let mut untouched = [0xA5_u8; ERROR_CAPACITY];
        let mut wrong_sink = sink(&mut untouched);
        wrong_sink.abi_version = sys::SLLM_HIP_ABI_VERSION + 1;
        let raw = unsafe { sys::sllm_get_abi_version(std::ptr::null_mut(), &mut wrong_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidAbiVersion);
        assert!(untouched.iter().all(|byte| *byte == 0xA5));

        let mut message = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut message);
        let mut result = backend_result();
        result.struct_size -= 1;
        let raw =
            unsafe { sys::sllm_backend_probe(sys::SLLM_BACKEND_HIP, &mut result, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidArgument);
        assert!(diagnostic(&message, error_sink.message_length).contains("struct_size"));

        let mut result = backend_result();
        result.abi_version += 1;
        let raw =
            unsafe { sys::sllm_backend_probe(sys::SLLM_BACKEND_HIP, &mut result, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidAbiVersion);
    }

    #[test]
    fn native_errors_report_required_length_and_buffer_too_small() {
        let expected = "HIP backend is unavailable in Phase 1 host stub";
        let mut result = backend_result();
        let mut short = [0xA5_u8; 1];
        let mut short_sink = sink_for(&mut short);
        let raw =
            unsafe { sys::sllm_backend_probe(sys::SLLM_BACKEND_HIP, &mut result, &mut short_sink) };
        assert_eq!(Status::from_raw(raw), Status::BufferTooSmall);
        assert_eq!(short_sink.message_length as usize, expected.len());
        assert_eq!(short, [0]);

        let mut no_storage = sink_for(&mut []);
        let raw =
            unsafe { sys::sllm_backend_probe(sys::SLLM_BACKEND_HIP, &mut result, &mut no_storage) };
        assert_eq!(Status::from_raw(raw), Status::BufferTooSmall);
        assert_eq!(no_storage.message_length as usize, expected.len());

        let mut full = vec![0_u8; expected.len() + 1];
        let mut full_sink = sink_for(&mut full);
        let raw =
            unsafe { sys::sllm_backend_probe(sys::SLLM_BACKEND_HIP, &mut result, &mut full_sink) };
        assert_eq!(Status::from_raw(raw), Status::HipUnavailable);
        assert_eq!(full_sink.message_length as usize, expected.len());
        assert_eq!(&full[..expected.len()], expected.as_bytes());
        assert_eq!(full[expected.len()], 0);

        let mut invalid_sink = sink_for(&mut full);
        invalid_sink.message = std::ptr::null_mut();
        let raw = unsafe {
            sys::sllm_backend_probe(sys::SLLM_BACKEND_HIP, &mut result, &mut invalid_sink)
        };
        assert_eq!(Status::from_raw(raw), Status::InvalidArgument);

        let raw = unsafe {
            sys::sllm_backend_probe(sys::SLLM_BACKEND_HIP, &mut result, std::ptr::null_mut())
        };
        assert_eq!(Status::from_raw(raw), Status::HipUnavailable);
    }

    #[test]
    fn native_probe_maps_unknown_backend_without_execution() {
        let mut message = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut message);
        let mut result = backend_result();
        let raw = unsafe { sys::sllm_backend_probe(999, &mut result, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::Unsupported);
        assert_eq!(result.backend, 0);
        assert!(diagnostic(&message, error_sink.message_length).contains("unknown backend"));
    }
}
