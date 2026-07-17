# AQ4 gfx1201 tiled batch GEMM runtime

- Added a cached HIPRTC `ullm_aq4_gemm_tiled_f32_kernel` using BM=8, BN=32, BK=128 and 256 threads. Weight tiles are dequantized once into LDS per token block and reused across the batch-token outputs.
- Production dispatch is exact `gfx1201`, `group_size=16`, `batch_count>=8`, `rows % 32 == 0`, and `cols % 128 == 0`. Batch tails are guarded in-kernel; ragged rows/K and all other architectures use the legacy kernel.
- Added a shape classifier for tests/registry consumers while leaving the existing matvec-batch ABI and ABI version unchanged. CPU scale-index metadata is validated before output writes; HIP validation remains a load-time/package responsibility to avoid per-call D2H synchronization.
- R9700 resident smoke showed the initial tile was a severe regression (`M127 56.29` vs `116.61`, `M128 56.88` vs `116.56`, `M256 56.61` vs `115.59` tok/s legacy), so tiled dispatch is now opt-in only via `ULLM_EXPERIMENTAL_HIP_AQ4_TILED_GEMM=1`; production defaults to Legacy pending a measured redesign.
- Tests: `cargo fmt --all --check`; `cargo test -p ullm-runtime-sys -- --test-threads=1` (148 passed); C++ compile via `cargo test --no-run`. HIP differential coverage is conditional on an available exact gfx1201 device.
- Commit: `4ab1181` (`runtime: add gfx1201 AQ4 tiled batch GEMM`).
