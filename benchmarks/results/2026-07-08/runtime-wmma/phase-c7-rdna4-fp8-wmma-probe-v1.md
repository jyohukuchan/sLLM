# Runtime RDNA4 FP8 WMMA probe v1

Date: 2026-07-08

Scope:

- Device under test: R9700/RDNA4, runtime device index `2`, backend `hip`, name `AMD Radeon Graphics`, compute `12.0`.
- Negative control: V620/RDNA2, runtime device index `1`, backend `hip`, name `AMD Radeon Pro V620`, compute `10.3`.
- ROCm/HIP: HIP `7.2.53211-e1a6bc5663`, AMD clang `22.0.0git`, `/opt/rocm-7.2.1`.
- Purpose: verify that uLLM HIPRTC kernels can directly call the RDNA4 FP8 WMMA builtin before integrating WMMA/MFMA-style QK/V accumulation into FlashAttention2-style kernels.

## Implementation Shape

- C ABI: `ullm_runtime_wmma_fp8_probe`.
- Rust FFI wrapper: `wmma_fp8_probe`.
- CLI smoke: `runtime-wmma-fp8-probe-smoke [DEVICE_INDEX]`.
- HIPRTC kernel: `ullm_wmma_fp8_probe_kernel`.
- The RDNA4 path calls `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`.
- Non-RDNA4 HIP devices write marker `0`, so they are not treated as a successful FP8 WMMA probe.
- CPU writes a separate nonzero marker only for API plumbing tests; it is not a GPU FP8 WMMA validation.

## Results

```text
target/release/ullm-engine inspect-devices
uLLM runtime ABI 1
devices: 4
[0] backend=cpu id=0 name="host CPU fallback" mem=0 compute=0.0 arch="" flags=1
[1] backend=hip id=0 name="AMD Radeon Pro V620" mem=32195477504 compute=10.3 arch="" flags=70253211
[2] backend=hip id=1 name="AMD Radeon Graphics" mem=34208743424 compute=12.0 arch="" flags=70253211
[3] backend=hip id=2 name="AMD Radeon Pro V620" mem=32195477504 compute=10.3 arch="" flags=70253211
```

```text
target/release/ullm-engine runtime-wmma-fp8-probe-smoke 2
runtime-wmma-fp8-probe-smoke backend=hip device=AMD Radeon Graphics compute=12.0 arch= marker=0x90c26ee1 verified=true

target/release/ullm-engine runtime-wmma-fp8-probe-smoke 2
runtime-wmma-fp8-probe-smoke backend=hip device=AMD Radeon Graphics compute=12.0 arch= marker=0x914ed1f3 verified=true
```

```text
target/release/ullm-engine runtime-wmma-fp8-probe-smoke 1
runtime wmma fp8 probe marker is zero: backend=hip device=AMD Radeon Pro V620 compute=10.3 arch=
```

## Interpretation

- uLLM can compile and launch a HIPRTC kernel that directly calls the RDNA4 FP8 WMMA builtin.
- The exact marker value is not a numerical correctness result and may vary; the pass condition is a nonzero marker on RDNA4.
- This confirms the next FlashAttention2-like step can use a local RDNA4 microkernel experiment instead of requiring a full external FlashAttention2 port.
- This probe does not measure attention speed and does not prove that the lane layout is correct for QK/V accumulation.
- The next useful benchmark is a small QK tile microkernel with known inputs, sampled diff, and a scalar reference before replacing scalar dot loops in cached-prefix/cold-prefill attention.

## Verification

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo test -p ullm-runtime-sys wmma_fp8_probe -- --test-threads=1`
- `cargo build -p ullm-engine --release`
- R9700 smoke: `target/release/ullm-engine runtime-wmma-fp8-probe-smoke 2`
- V620 negative control: `target/release/ullm-engine runtime-wmma-fp8-probe-smoke 1`
