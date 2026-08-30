//! Host-side artifact, benchmark, quality, and bounded-debug tooling.
//!
//! These tools never act as a numerical fallback for the HIP runtime.  They
//! consume verified artifacts or explicit bounded fixtures and emit versioned,
//! digest-bound reports.

mod artifact;
mod benchmark;
mod debug_dump;
mod quality;
mod tool_manifest;

pub use artifact::*;
pub use benchmark::*;
pub use debug_dump::*;
pub use quality::*;
pub use tool_manifest::*;
