//! Safe Rust access to the versioned HIP ABI.
//!
//! The Phase 1 native library is a host stub. Probing it is useful for callers
//! that need an explicit capability result, but it must not be mistaken for a
//! working HIP backend.

use std::fmt;
use std::mem::size_of;

use ullm_core::{
    Backend, BackendCapabilities, BackendError, BackendSupport, ExecutionReceipt,
    MaterializedTensor, SemanticOp, TensorView,
};
use ullm_hip_sys as sys;

const ERROR_CAPACITY: usize = 256;

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
    Unknown(u32),
}

impl Status {
    pub const fn from_raw(raw: u32) -> Self {
        match raw {
            sys::ULLM_STATUS_OK => Self::Ok,
            sys::ULLM_STATUS_INVALID_ARGUMENT => Self::InvalidArgument,
            sys::ULLM_STATUS_BUFFER_TOO_SMALL => Self::BufferTooSmall,
            sys::ULLM_STATUS_UNSUPPORTED => Self::Unsupported,
            sys::ULLM_STATUS_HIP_UNAVAILABLE => Self::HipUnavailable,
            sys::ULLM_STATUS_INVALID_ABI_VERSION => Self::InvalidAbiVersion,
            sys::ULLM_STATUS_RESERVED_NONZERO => Self::ReservedNonzero,
            sys::ULLM_STATUS_INTERNAL_ERROR => Self::InternalError,
            other => Self::Unknown(other),
        }
    }

    pub const fn raw(self) -> u32 {
        match self {
            Self::Ok => sys::ULLM_STATUS_OK,
            Self::InvalidArgument => sys::ULLM_STATUS_INVALID_ARGUMENT,
            Self::BufferTooSmall => sys::ULLM_STATUS_BUFFER_TOO_SMALL,
            Self::Unsupported => sys::ULLM_STATUS_UNSUPPORTED,
            Self::HipUnavailable => sys::ULLM_STATUS_HIP_UNAVAILABLE,
            Self::InvalidAbiVersion => sys::ULLM_STATUS_INVALID_ABI_VERSION,
            Self::ReservedNonzero => sys::ULLM_STATUS_RESERVED_NONZERO,
            Self::InternalError => sys::ULLM_STATUS_INTERNAL_ERROR,
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
        Status::Unknown(_) => "unknown native status",
    }
}

fn sink(buffer: &mut [u8; ERROR_CAPACITY]) -> sys::ullm_error_sink_t {
    sys::ullm_error_sink_t {
        struct_size: size_of::<sys::ullm_error_sink_t>() as u32,
        abi_version: sys::ULLM_HIP_ABI_VERSION,
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
    let raw = unsafe { sys::ullm_get_abi_version(&mut version, &mut error_sink) };
    ensure_ok(raw, diagnostic(&buffer, error_sink.message_length)).map(|()| version)
}

pub fn version() -> Result<Version, HipError> {
    let mut buffer = [0u8; ERROR_CAPACITY];
    let mut error_sink = sink(&mut buffer);
    let mut info = sys::ullm_version_info_t {
        struct_size: size_of::<sys::ullm_version_info_t>() as u32,
        abi_version: sys::ULLM_HIP_ABI_VERSION,
        major: 0,
        minor: 0,
        patch: 0,
        reserved: [0; 3],
    };
    let raw = unsafe { sys::ullm_query_version(&mut info, &mut error_sink) };
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
    let mut result = sys::ullm_backend_probe_result_t {
        struct_size: size_of::<sys::ullm_backend_probe_result_t>() as u32,
        abi_version: sys::ULLM_HIP_ABI_VERSION,
        backend: 0,
        available: 0,
        hip_runtime_present: 0,
        reserved: [0; 3],
    };
    let raw =
        unsafe { sys::ullm_backend_probe(sys::ULLM_BACKEND_HIP, &mut result, &mut error_sink) };
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
    let mut result = sys::ullm_context_probe_result_t {
        struct_size: size_of::<sys::ullm_context_probe_result_t>() as u32,
        abi_version: sys::ULLM_HIP_ABI_VERSION,
        context_present: 0,
        hip_available: 0,
        reserved: [0; 4],
    };
    let raw = unsafe { sys::ullm_context_probe(std::ptr::null(), &mut result, &mut error_sink) };
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

/// A typed backend handle is only constructible after an available HIP probe.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sink_for(buffer: &mut [u8]) -> sys::ullm_error_sink_t {
        sys::ullm_error_sink_t {
            struct_size: size_of::<sys::ullm_error_sink_t>() as u32,
            abi_version: sys::ULLM_HIP_ABI_VERSION,
            message: buffer.as_mut_ptr().cast(),
            message_capacity: buffer.len() as u64,
            message_length: 0,
            reserved: [0, 0],
        }
    }

    fn backend_result() -> sys::ullm_backend_probe_result_t {
        sys::ullm_backend_probe_result_t {
            struct_size: size_of::<sys::ullm_backend_probe_result_t>() as u32,
            abi_version: sys::ULLM_HIP_ABI_VERSION,
            backend: 0,
            available: 0,
            hip_runtime_present: 0,
            reserved: [0; 3],
        }
    }

    fn version_info() -> sys::ullm_version_info_t {
        sys::ullm_version_info_t {
            struct_size: size_of::<sys::ullm_version_info_t>() as u32,
            abi_version: sys::ULLM_HIP_ABI_VERSION,
            major: 0,
            minor: 0,
            patch: 0,
            reserved: [0; 3],
        }
    }

    fn context_probe_result() -> sys::ullm_context_probe_result_t {
        sys::ullm_context_probe_result_t {
            struct_size: size_of::<sys::ullm_context_probe_result_t>() as u32,
            abi_version: sys::ULLM_HIP_ABI_VERSION,
            context_present: 0,
            hip_available: 0,
            reserved: [0; 4],
        }
    }

    #[test]
    fn status_mapping_preserves_unknown_codes() {
        assert_eq!(
            Status::from_raw(sys::ULLM_STATUS_HIP_UNAVAILABLE),
            Status::HipUnavailable
        );
        assert_eq!(Status::from_raw(99), Status::Unknown(99));
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

        let raw = unsafe { sys::ullm_query_version(std::ptr::null_mut(), &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidArgument);
        assert!(diagnostic(&message, error_sink.message_length).contains("version output"));

        let mut info = version_info();
        info.reserved[0] = 1;
        let raw = unsafe { sys::ullm_query_version(&mut info, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::ReservedNonzero);
        assert!(diagnostic(&message, error_sink.message_length).contains("reserved"));

        let mut info = version_info();
        info.abi_version += 1;
        let raw = unsafe { sys::ullm_query_version(&mut info, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidAbiVersion);
        assert!(diagnostic(&message, error_sink.message_length).contains("ABI"));

        let mut info = version_info();
        info.struct_size -= 1;
        let raw = unsafe { sys::ullm_query_version(&mut info, &mut error_sink) };
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

        let raw = unsafe { sys::ullm_query_version(&mut info, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::Ok);
        assert_eq!(info.abi_version, sys::ULLM_HIP_ABI_VERSION);
        assert_eq!(info.major, sys::ULLM_HIP_LIBRARY_VERSION_MAJOR);
        assert_eq!(info.minor, sys::ULLM_HIP_LIBRARY_VERSION_MINOR);
        assert_eq!(info.patch, sys::ULLM_HIP_LIBRARY_VERSION_PATCH);
        assert_eq!(info.reserved, [0; 3]);
        assert_eq!(error_sink.message_length, 0);
    }

    #[test]
    fn native_context_probe_rejects_null_reserved_wrong_abi_and_undersized_result() {
        let mut message = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut message);

        let raw = unsafe {
            sys::ullm_context_probe(std::ptr::null(), std::ptr::null_mut(), &mut error_sink)
        };
        assert_eq!(Status::from_raw(raw), Status::InvalidArgument);
        assert!(diagnostic(&message, error_sink.message_length).contains("context probe"));

        let mut result = context_probe_result();
        result.reserved[0] = 1;
        let raw =
            unsafe { sys::ullm_context_probe(std::ptr::null(), &mut result, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::ReservedNonzero);
        assert!(diagnostic(&message, error_sink.message_length).contains("reserved"));

        let mut result = context_probe_result();
        result.abi_version += 1;
        let raw =
            unsafe { sys::ullm_context_probe(std::ptr::null(), &mut result, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidAbiVersion);
        assert!(diagnostic(&message, error_sink.message_length).contains("ABI"));

        let mut result = context_probe_result();
        result.struct_size -= 1;
        let raw =
            unsafe { sys::ullm_context_probe(std::ptr::null(), &mut result, &mut error_sink) };
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
            unsafe { sys::ullm_context_probe(std::ptr::null(), &mut result, &mut error_sink) };
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
        let mut result = sys::ullm_backend_probe_result_t {
            struct_size: size_of::<sys::ullm_backend_probe_result_t>() as u32,
            abi_version: sys::ULLM_HIP_ABI_VERSION,
            backend: 0,
            available: 0,
            hip_runtime_present: 0,
            reserved: [1, 0, 0],
        };

        let raw =
            unsafe { sys::ullm_backend_probe(sys::ULLM_BACKEND_HIP, &mut result, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::ReservedNonzero);
        assert!(diagnostic(&buffer, error_sink.message_length).contains("reserved"));
    }

    #[test]
    fn native_abi_rejects_null_and_wrong_abi_inputs() {
        let raw = unsafe { sys::ullm_get_abi_version(std::ptr::null_mut(), std::ptr::null_mut()) };
        assert_eq!(Status::from_raw(raw), Status::InvalidArgument);

        let mut untouched = [0xA5_u8; ERROR_CAPACITY];
        let mut wrong_sink = sink(&mut untouched);
        wrong_sink.abi_version = sys::ULLM_HIP_ABI_VERSION + 1;
        let raw = unsafe { sys::ullm_get_abi_version(std::ptr::null_mut(), &mut wrong_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidAbiVersion);
        assert!(untouched.iter().all(|byte| *byte == 0xA5));

        let mut message = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut message);
        let mut result = backend_result();
        result.struct_size -= 1;
        let raw =
            unsafe { sys::ullm_backend_probe(sys::ULLM_BACKEND_HIP, &mut result, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidArgument);
        assert!(diagnostic(&message, error_sink.message_length).contains("struct_size"));

        let mut result = backend_result();
        result.abi_version += 1;
        let raw =
            unsafe { sys::ullm_backend_probe(sys::ULLM_BACKEND_HIP, &mut result, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::InvalidAbiVersion);
    }

    #[test]
    fn native_errors_report_required_length_and_buffer_too_small() {
        let expected = "HIP backend is unavailable in Phase 1 host stub";
        let mut result = backend_result();
        let mut short = [0xA5_u8; 1];
        let mut short_sink = sink_for(&mut short);
        let raw =
            unsafe { sys::ullm_backend_probe(sys::ULLM_BACKEND_HIP, &mut result, &mut short_sink) };
        assert_eq!(Status::from_raw(raw), Status::BufferTooSmall);
        assert_eq!(short_sink.message_length as usize, expected.len());
        assert_eq!(short, [0]);

        let mut no_storage = sink_for(&mut []);
        let raw =
            unsafe { sys::ullm_backend_probe(sys::ULLM_BACKEND_HIP, &mut result, &mut no_storage) };
        assert_eq!(Status::from_raw(raw), Status::BufferTooSmall);
        assert_eq!(no_storage.message_length as usize, expected.len());

        let mut full = vec![0_u8; expected.len() + 1];
        let mut full_sink = sink_for(&mut full);
        let raw =
            unsafe { sys::ullm_backend_probe(sys::ULLM_BACKEND_HIP, &mut result, &mut full_sink) };
        assert_eq!(Status::from_raw(raw), Status::HipUnavailable);
        assert_eq!(full_sink.message_length as usize, expected.len());
        assert_eq!(&full[..expected.len()], expected.as_bytes());
        assert_eq!(full[expected.len()], 0);

        let mut invalid_sink = sink_for(&mut full);
        invalid_sink.message = std::ptr::null_mut();
        let raw = unsafe {
            sys::ullm_backend_probe(sys::ULLM_BACKEND_HIP, &mut result, &mut invalid_sink)
        };
        assert_eq!(Status::from_raw(raw), Status::InvalidArgument);

        let raw = unsafe {
            sys::ullm_backend_probe(sys::ULLM_BACKEND_HIP, &mut result, std::ptr::null_mut())
        };
        assert_eq!(Status::from_raw(raw), Status::HipUnavailable);
    }

    #[test]
    fn native_probe_maps_unknown_backend_without_execution() {
        let mut message = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut message);
        let mut result = backend_result();
        let raw = unsafe { sys::ullm_backend_probe(999, &mut result, &mut error_sink) };
        assert_eq!(Status::from_raw(raw), Status::Unsupported);
        assert_eq!(result.backend, 0);
        assert!(diagnostic(&message, error_sink.message_length).contains("unknown backend"));
    }
}
