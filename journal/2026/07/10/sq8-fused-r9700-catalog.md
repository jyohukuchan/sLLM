2026-07-10

- Purpose: extend the SQ8_0 fused projection descriptor catalog toward R9700-specific optimization without enabling runtime active dispatch.
- Change: add R9700 fused descriptor constants and catalog entries in `backend_dispatch.rs`.
- R9700 catalog entries use `gpu_arch=Some("RDNA4")`, `gpu_name=Some("Radeon_AI_PRO_R9700")`, and `priority=20`.
- Tests: catalog coverage includes Generic/RDNA4/R9700; fused catalog entries remain absent from active SQ8_0 projection dispatch; selecting from the catalog can choose the R9700 fused entry.
- Verification: `cargo test -p ullm-engine backend_dispatch --lib` passed with 22 tests.
