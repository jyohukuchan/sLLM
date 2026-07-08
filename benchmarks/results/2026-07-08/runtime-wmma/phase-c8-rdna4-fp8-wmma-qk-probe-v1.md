# Runtime RDNA4 FP8 WMMA QK probe v1

Date: 2026-07-08

Scope:

- Device under test: R9700/RDNA4, runtime device index `2`, backend `hip`, name `AMD Radeon Graphics`, compute `12.0`.
- Negative controls: V620/RDNA2 runtime device index `1`, CPU runtime device index `0`.
- Purpose: move from a marker-only FP8 WMMA builtin probe to a small QK tile arithmetic probe before replacing scalar dot loops in FlashAttention2-style attention kernels.

## Implementation Shape

- C ABI: `ullm_runtime_wmma_fp8_qk_probe`.
- Rust FFI wrapper: `wmma_fp8_qk_probe`.
- CLI smoke: `runtime-wmma-fp8-qk-probe-smoke [DEVICE_INDEX]`.
- Input: fixed 16x16 FP8 E4M3 byte tile for Q and fixed 16x16 FP8 E4M3 byte tile for K.
- Output: 16x16 F32 accumulator tile, currently stored in raw WMMA accumulator register order on HIP.
- HIPRTC kernel: `ullm_wmma_fp8_qk_probe_kernel`.
- RDNA4 path calls `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`.
- CPU path computes row-major Q*K^T and is used for API plumbing tests, not as the CLI target.

The initial smoke fills Q and K with FP8 E4M3 `1.0` (`0x38`). This avoids depending on the final accumulator layout: every 16-element dot product should be `16.0` regardless of output element ordering.

## Results

```text
cargo test -p ullm-runtime-sys wmma_fp8_qk_probe -- --test-threads=1
running 2 tests
test tests::cpu_wmma_fp8_qk_probe_outputs_finite_nonzero_values ... ok
test tests::first_hip_wmma_fp8_qk_probe_outputs_finite_nonzero_values_when_available ... ok
```

```text
target/release/ullm-engine runtime-wmma-fp8-qk-probe-smoke 2
runtime-wmma-fp8-qk-probe-smoke backend=hip device=AMD Radeon Graphics compute=12.0 arch= pattern=ones max_abs=16.000000000 preview=[16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000,16.0000000] finite=true nonzero=true verified=true
```

```text
target/release/ullm-engine runtime-wmma-fp8-qk-probe-smoke 2 layout
runtime-wmma-fp8-qk-probe-smoke backend=hip device=AMD Radeon Graphics compute=12.0 arch= pattern=layout max_abs=374.000000000 preview=[136.0000000,0.0000000,168.0000000,0.0000000,200.0000000,0.0000000,232.0000000,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000] finite=true nonzero=true verified=true
```

The CLI was then extended with `PREVIEW_COUNT`, and `layout 256` produced this raw accumulator pattern:

```text
target/release/ullm-engine runtime-wmma-fp8-qk-probe-smoke 2 layout 256
runtime-wmma-fp8-qk-probe-smoke backend=hip device=AMD Radeon Graphics compute=12.0 arch= pattern=layout max_abs=374.000000000 preview_count=256 preview=[136.0000000,0.0000000,168.0000000,0.0000000,200.0000000,0.0000000,232.0000000,0.0000000,...,278.0000000,0.0000000,310.0000000,0.0000000,342.0000000,0.0000000,374.0000000,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000] finite=true nonzero=true verified=true
```

Observed nonzero structure:

- Raw output indices `16*c + {0,2,4,6}` for `c=0..7` contain `136+2*c`, `168+2*c`, `200+2*c`, `232+2*c`.
- Raw output indices `128 + 16*c + {0,2,4,6}` for `c=0..7` contain `264+2*c`, `296+2*c`, `328+2*c`, `360+2*c`.
- All other observed elements are zero.

```text
target/release/ullm-engine runtime-wmma-fp8-qk-probe-smoke 1
runtime wmma fp8 qk probe smoke requires RDNA4: backend=hip device=AMD Radeon Pro V620 compute=10.3 arch=
```

```text
target/release/ullm-engine runtime-wmma-fp8-qk-probe-smoke 0
runtime wmma fp8 qk probe smoke requires a HIP device: backend=cpu device=host CPU fallback
```

## Interpretation

- R9700 can run a 16x16x16 FP8 WMMA operation through uLLM HIPRTC and produce the expected all-ones QK accumulator magnitude.
- This is stronger than the marker-only probe because the output value depends on FP8 WMMA arithmetic.
- The `layout` pattern is intentionally non-uniform. CPU row-major Q*K^T would produce values `16*row+col`, but the current HIP preview does not match row-major order.
- `layout 256` reaches `374`, while CPU row-major Q*K^T would be bounded by `255`. This means the current direct contiguous byte packing is not just a differently ordered row-major result; the A/B input register packing is also not row-major.
- This is still not a complete QK microkernel for attention because both input register packing and output accumulator order must be mapped before arbitrary Q/K tiles can be compared to row-major Q*K^T.
- The next required step is to identify lane/register output layout with non-uniform Q/K input and compare against CPU row-major Q*K^T.

## Verification

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo test -p ullm-runtime-sys wmma_fp8_qk_probe -- --test-threads=1`
- `cargo build -p ullm-engine --release`
- R9700 smoke: `target/release/ullm-engine runtime-wmma-fp8-qk-probe-smoke 2`
- R9700 layout dump: `target/release/ullm-engine runtime-wmma-fp8-qk-probe-smoke 2 layout 256`
- V620 negative control: `target/release/ullm-engine runtime-wmma-fp8-qk-probe-smoke 1`
- CPU negative control: `target/release/ullm-engine runtime-wmma-fp8-qk-probe-smoke 0`
