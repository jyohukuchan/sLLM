# AQ4 gfx1201 register-tiled batch GEMM

- Added experimental BM4 and BM8 register-tiled AQ4 batch kernels for exact gfx1201/group16 shapes. A 256-thread block maps eight lanes to each of 32 rows; each lane decodes complete group16 weights once and accumulates 4 or 8 token outputs in registers.
- The kernels use width-8 shuffle reduction within four independent row subgroups per wave32. They use no LDS, no K-loop barriers, and no global workspace. gfx1201 O3 code-object metadata reports BM4/BM8 VGPR counts `19/24`, SGPR counts `50/60`, zero spills, and zero fixed group/private segment bytes.
- Dispatch remains experimental. `ULLM_EXPERIMENTAL_HIP_AQ4_REGISTER_BM=4` selects BM4 for batch counts at least 4; value `8` selects BM8 for batch counts at least 8. Valid register settings take precedence over the older LDS flag. An invalid register value is fail-closed to Legacy; an absent register setting allows the separately guarded LDS experiment.
- Supported candidate geometry is exact `gfx1201`, `group_size=16`, rows divisible by 32, and cols divisible by 128. Batch tails are guarded. Environment variables absent, small batches, ragged rows/K, unsupported groups, and other architectures select Legacy.
- The C/Rust classifier now distinguishes Legacy, LDS BM8, register BM4, and register BM8. Enum additions are additive and the runtime ABI remains version 1.
- Verification: gfx1201 C++17 O3 device compile passed with width-8 shuffle; host C++ no-run compile passed; `cargo fmt --all --check`; `cargo test -p ullm-runtime-sys -- --test-threads=1` passed 148 tests; `git diff --check` passed. Conditional gfx1201 differential coverage includes BM4/BM8, M=4/8/16/32/64/127/128, rows 32/64, cols 128/256, scale table 7, tensor scale, signed inputs, and row scales present/absent.
- Commit: `e1c9877` (`runtime: add experimental AQ4 register batch GEMM`).

## Forced BM8 runtime ABI

- Added the additive C/Rust `aq4_matvec_batch_register_bm8_f32` ABI for typed registry calls. It directly invokes the cached BM8 kernel without reading experiment environment variables and never falls back to Legacy or LDS.
- The forced path accepts only HIP exact gfx1201, group16, batch at least 8, rows divisible by 32, and cols divisible by 128. CPU and unsupported geometry are rejected before launch with output unchanged. HIP scale metadata is not copied to host or synchronized per call.
- CPU reject/no-fallback/output-preservation regression passed. The conditional gfx1201 differential now calls the forced wrapper directly for M=8/16/32/64/127/128 across rows 32/64 and cols 128/256, while CPU expected output still uses the existing reference batch ABI.
- Full runtime-sys result: 149 passed. Host no-run compile, fmt, diff check, and exact gfx1201 O3 device compile passed. BM8 metadata remains wave32, VGPR 24, SGPR 60, zero spills, and zero LDS/private segment bytes.
- Forced ABI commit: `17ceb66` (`runtime: expose forced AQ4 register BM8 ABI`).
