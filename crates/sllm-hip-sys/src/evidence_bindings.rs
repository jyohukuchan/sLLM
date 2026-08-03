use core::ffi::c_char;

pub const SLLM_HIP_EVIDENCE_ABI_VERSION: u32 = 1;
pub const SLLM_HIP_EVIDENCE_TRANSFORM_XOR: u8 = 0x5a;

pub const SLLM_STATUS_HIP_TIMEOUT: u32 = 8;
pub const SLLM_STATUS_HIP_INVALID_HANDLE: u32 = 9;
pub const SLLM_STATUS_HIP_ZERO_DISPATCH: u32 = 10;
pub const SLLM_STATUS_HIP_RUNTIME_ERROR: u32 = 11;
pub const SLLM_STATUS_HIP_DISPATCH_CONTRACT: u32 = 12;

#[repr(C)]
pub struct sllm_hip_evidence_completion_t {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct sllm_hip_evidence_request_t {
    pub struct_size: u32,
    pub abi_version: u32,
    pub input: *const u8,
    pub input_size: u64,
    pub reserved: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct sllm_hip_evidence_result_t {
    pub struct_size: u32,
    pub abi_version: u32,
    pub output_size: u64,
    /// Successful device `hipMalloc` allocations; excludes `hipHostMalloc`.
    pub allocation_count: u64,
    /// Successful `hipMemcpyAsync` transfers; excludes CPU `memcpy` operations.
    pub copy_count: u64,
    pub dispatch_count: u64,
    pub selected_backend: u32,
    pub fallback_used: u32,
    pub terminal: u32,
    pub reserved: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct sllm_error_sink_t {
    pub struct_size: u32,
    pub abi_version: u32,
    pub message: *mut c_char,
    pub message_capacity: u64,
    pub message_length: u64,
    pub reserved: [u64; 2],
}

unsafe extern "C" {
    pub fn sllm_hip_evidence_submit(
        request: *const sllm_hip_evidence_request_t,
        completion: *mut *mut sllm_hip_evidence_completion_t,
        error_sink: *mut sllm_error_sink_t,
    ) -> u32;
    pub fn sllm_hip_evidence_wait(
        completion: *mut sllm_hip_evidence_completion_t,
        timeout_ms: u32,
        output: *mut u8,
        output_capacity: u64,
        result: *mut sllm_hip_evidence_result_t,
        error_sink: *mut sllm_error_sink_t,
    ) -> u32;
    pub fn sllm_hip_evidence_destroy(
        completion: *mut *mut sllm_hip_evidence_completion_t,
        error_sink: *mut sllm_error_sink_t,
    ) -> u32;
}
