# Runtime RDNA4 FP8 rocWMMA QK probe v1

Date: 2026-07-08

Scope:

- Device under test: R9700/RDNA4, runtime device index `2`, backend `hip`, name `AMD Radeon Graphics`, compute `12.0`.
- Negative controls: V620/RDNA2 runtime device index `1`, CPU runtime device index `0`.
- Purpose: verify a row-major 16x16 FP8 Q*K^T path through rocWMMA fragment load/store before using WMMA inside uLLM FlashAttention-style kernels.

## Implementation Shape

- C ABI: `ullm_runtime_rocwmma_fp8_qk_probe`.
- Rust FFI wrapper: `rocwmma_fp8_qk_probe`.
- CLI smoke: `runtime-rocwmma-fp8-qk-probe-smoke [DEVICE_INDEX] [PATTERN=ones|layout] [PREVIEW_COUNT]`.
- HIPRTC kernel: `ullm_rocwmma_fp8_qk_probe_kernel`.
- RDNA4 path uses `rocwmma::fragment`, `load_matrix_sync`, `mma_sync`, and `store_matrix_sync`.
- Q input layout: `rocwmma::row_major`.
- K input layout for Q*K^T: `rocwmma::col_major` over the row-major K tile bytes.
- Output layout: `rocwmma::row_major`.
- HIPRTC compile options now support extra include paths for rocWMMA headers.

This probe is separate from `ullm_runtime_wmma_fp8_qk_probe`. The raw builtin probe remains useful for inspecting register order, while the rocWMMA probe verifies the practical matrix API path needed by attention kernels.

## Results

```text
cargo test -p ullm-runtime-sys rocwmma_fp8_qk_probe -- --test-threads=1
running 2 tests
test tests::cpu_rocwmma_fp8_qk_probe_outputs_finite_nonzero_values ... ok
test tests::first_hip_rocwmma_fp8_qk_probe_outputs_finite_nonzero_values_when_available ... ok
```

```text
target/release/ullm-engine runtime-rocwmma-fp8-qk-probe-smoke 2 ones 16
runtime-rocwmma-fp8-qk-probe-smoke backend=hip device=AMD Radeon Graphics compute=12.0 arch= pattern=ones max_abs=16.000000000 preview_count=16 preview=[16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000] finite=true nonzero=true verified=true
```

```text
target/release/ullm-engine runtime-rocwmma-fp8-qk-probe-smoke 2 layout 64
runtime-rocwmma-fp8-qk-probe-smoke backend=hip device=AMD Radeon Graphics compute=12.0 arch= pattern=layout max_abs=255.000000000 preview_count=64 preview=[0.0000000,1.0000000,2.0000000,3.0000000,4.0000000,5.0000000,6.0000000,7.0000000,8.0000000,9.0000000,10.0000000,11.0000000,12.0000000,13.0000000,14.0000000,15.0000000,16.0000000,17.0000000,18.0000000,19.0000000,20.0000000,21.0000000,22.0000000,23.0000000,24.0000000,25.0000000,26.0000000,27.0000000,28.0000000,29.0000000,30.0000000,31.0000000,32.0000000,33.0000000,34.0000000,35.0000000,36.0000000,37.0000000,38.0000000,39.0000000,40.0000000,41.0000000,42.0000000,43.0000000,44.0000000,45.0000000,46.0000000,47.0000000,48.0000000,49.0000000,50.0000000,51.0000000,52.0000000,53.0000000,54.0000000,55.0000000,56.0000000,57.0000000,58.0000000,59.0000000,60.0000000,61.0000000,62.0000000,63.0000000] finite=true nonzero=true verified=true
```

```text
target/release/ullm-engine runtime-rocwmma-fp8-qk-probe-smoke
runtime-rocwmma-fp8-qk-probe-smoke backend=hip device=AMD Radeon Graphics compute=12.0 arch= pattern=ones max_abs=16.000000000 preview_count=16 preview=[16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000] finite=true nonzero=true verified=true
```

```text
target/release/ullm-engine runtime-rocwmma-fp8-qk-probe-smoke 1 ones 16
runtime rocwmma fp8 qk probe smoke requires RDNA4: backend=hip device=AMD Radeon Pro V620 compute=10.3 arch=
```

## Interpretation

- The rocWMMA fragment path fixes the practical layout problem seen in the raw builtin QK probe.
- The all-ones pattern produces the expected dot-product value `16.0`.
- The non-uniform `layout` pattern produces row-major `16*row+col`; the first 64 outputs are `0..63` and the max is `255`.
- This means uLLM can use rocWMMA load/store semantics as the first implementation path for RDNA4 FP8 QK tiles instead of manually reverse-engineering the raw accumulator order before any useful FlashAttention kernel can be built.
- The remaining FlashAttention work is still nontrivial: QK tiles must be integrated with causal/cached-prefix masks, online softmax, V accumulation, and per-head batching. This probe only proves the 16x16 FP8 QK primitive.

## Verification

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo test -p ullm-runtime-sys rocwmma_fp8_qk_probe -- --test-threads=1`
- `cargo build -p ullm-engine --release`
- R9700 ones smoke: `target/release/ullm-engine runtime-rocwmma-fp8-qk-probe-smoke 2 ones 16`
- R9700 layout smoke: `target/release/ullm-engine runtime-rocwmma-fp8-qk-probe-smoke 2 layout 64`
- Default RDNA4 selection: `target/release/ullm-engine runtime-rocwmma-fp8-qk-probe-smoke`
- V620 negative control: `target/release/ullm-engine runtime-rocwmma-fp8-qk-probe-smoke 1 ones 16`
