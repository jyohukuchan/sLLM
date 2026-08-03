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

/// Metadata for the checked-in ABI mirror. Regeneration is intentionally explicit.
pub mod binding_metadata {
    pub const SOURCE: &str = "include/ullm/hip.h";
    pub const ABI_VERSION: u32 = 1;
    pub const TOOL: &str = "bindgen 0.71.1 (checked-in output)";
    pub const OPTIONS: &str = "--allowlist-type '^ullm_.*' --allowlist-function '^ullm_.*' --allowlist-var '^ULLM_.*' --no-layout-tests";
}
