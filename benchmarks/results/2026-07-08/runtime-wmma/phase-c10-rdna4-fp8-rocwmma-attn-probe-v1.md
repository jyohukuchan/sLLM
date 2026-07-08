# Runtime RDNA4 FP8 rocWMMA attention probe v1

Date: 2026-07-08

Scope:

- Device under test: R9700/RDNA4, runtime device index `2`, backend `hip`, name `AMD Radeon Graphics`, compute `12.0`.
- Negative control: V620/RDNA2 runtime device index `1`.
- Purpose: verify a minimal FlashAttention-style path using rocWMMA FP8 QK, online softmax, and F32 V accumulation before integrating into cached-prefix/cold-prefill attention kernels.

## Implementation Shape

- C ABI: `ullm_runtime_rocwmma_fp8_attn_probe`.
- Rust FFI wrapper: `rocwmma_fp8_attn_probe`.
- CLI smoke: `runtime-rocwmma-fp8-attn-probe-smoke [DEVICE_INDEX] [PATTERN=ones|layout]`.
- Fixed input/output shapes:
  - Q: `16x16` FP8 E4M3 bytes.
  - K: `32x16` FP8 E4M3 bytes.
  - V: `32x16` F32.
  - output: `16x16` F32.
- The HIPRTC kernel runs a single 32-lane block, computes two 16x16 QK tiles through rocWMMA, and performs per-row online softmax with V accumulation.
- The CLI computes an independent CPU reference from the same quantized Q/K bytes and V values, then reports `max_abs_diff`.

## Results

```text
cargo test -p ullm-runtime-sys rocwmma_fp8_attn_probe -- --test-threads=1
running 2 tests
test tests::cpu_rocwmma_fp8_attn_probe_outputs_finite_nonzero_values ... ok
test tests::first_hip_rocwmma_fp8_attn_probe_outputs_finite_nonzero_values_when_available ... ok
```

```text
target/release/ullm-engine runtime-rocwmma-fp8-attn-probe-smoke 2 ones
runtime-rocwmma-fp8-attn-probe-smoke backend=hip device=AMD Radeon Graphics compute=12.0 arch= pattern=ones q_shape=16x16_fp8 k_shape=32x16_fp8 v_shape=32x16_f32 output_shape=16x16_f32 max_abs_diff=0.000000000 output_preview=[-0.0859375,-0.0244141,-0.0576172,-0.0908203,-0.0292969,0.0322266,-0.0009766,0.0605469,0.1220703,0.0888672,0.0556641,0.0224609,-0.0107422,-0.0439453,0.0175781,-0.0156250] expected_preview=[-0.0859375,-0.0244141,-0.0576172,-0.0908203,-0.0292969,0.0322266,-0.0009766,0.0605469,0.1220703,0.0888672,0.0556641,0.0224609,-0.0107422,-0.0439453,0.0175781,-0.0156250] verified=true
```

```text
target/release/ullm-engine runtime-rocwmma-fp8-attn-probe-smoke 2 layout
runtime-rocwmma-fp8-attn-probe-smoke backend=hip device=AMD Radeon Graphics compute=12.0 arch= pattern=layout q_shape=16x16_fp8 k_shape=32x16_fp8 v_shape=32x16_f32 output_shape=16x16_f32 max_abs_diff=0.000000119 output_preview=[-0.3190792,-0.1628311,-0.0488643,0.1072809,0.2635306,0.4197807,0.5337475,0.6898926,0.8461425,0.9966701,1.1529059,-0.9994264,-0.8488988,-0.6926630,-0.5364131,-0.6925957] expected_preview=[-0.3190792,-0.1628311,-0.0488643,0.1072809,0.2635307,0.4197807,0.5337475,0.6898926,0.8461425,0.9966701,1.1529059,-0.9994264,-0.8488989,-0.6926630,-0.5364131,-0.6925957] verified=true
```

```text
target/release/ullm-engine runtime-rocwmma-fp8-attn-probe-smoke 1 ones
runtime rocwmma fp8 attention probe smoke requires RDNA4: backend=hip device=AMD Radeon Pro V620 compute=10.3 arch=
```

## Interpretation

- The rocWMMA QK path can be connected to online softmax and V accumulation without breaking numerical output on the fixed probe.
- This is still a standalone probe, not a production attention kernel. It uses a single block, fixed shapes, F32 V, and no causal/cached-prefix masking.
- The next implementation step is to move this structure into the existing cached-prefix/cold-prefill flash2 kernels with real head/token/block indexing and sampled diff checks.

## Verification

- `cargo fmt --all`
- `git diff --check`
- `cargo check -p ullm-engine`
- `cargo test -p ullm-runtime-sys rocwmma_fp8_attn_probe -- --test-threads=1`
- `cargo build -p ullm-engine --release`
- R9700 ones smoke: `target/release/ullm-engine runtime-rocwmma-fp8-attn-probe-smoke 2 ones`
- R9700 layout smoke: `target/release/ullm-engine runtime-rocwmma-fp8-attn-probe-smoke 2 layout`
- V620 negative control: `target/release/ullm-engine runtime-rocwmma-fp8-attn-probe-smoke 1 ones`
