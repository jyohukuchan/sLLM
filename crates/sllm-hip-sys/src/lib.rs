#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

mod bindings;

pub use bindings::*;

/// Private model-free G1 evidence ABI. This is not the installed public ABI.
#[doc(hidden)]
#[allow(non_camel_case_types, non_upper_case_globals)]
pub mod evidence {
    include!("evidence_bindings.rs");
}

/// Build-time identity embedded in the dedicated G2 executable dependency.
/// The host builder and validator independently recompute this identity from
/// the reviewed Cargo/build.rs/CMake source set before accepting an artifact.
#[doc(hidden)]
pub mod g2_build_identity {
    include!(concat!(env!("OUT_DIR"), "/rmsnorm_g2_build_identity.rs"));
}

/// Metadata for the checked-in ABI mirror. Regeneration is intentionally explicit.
pub mod binding_metadata {
    pub const SOURCE: &str = "include/sllm/hip.h";
    pub const ABI_VERSION: u32 = 1;
    pub const TOOL: &str = "bindgen 0.71.1 (checked-in output)";
    pub const OPTIONS: &str = "--allowlist-type '^sllm_.*' --allowlist-function '^sllm_.*' --allowlist-var '^SLLM_.*' --no-layout-tests";
}
