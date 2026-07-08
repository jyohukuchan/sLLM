# Phase C11: RDNA4 FP8 rocWMMA Cached-Prefix v1

## Summary

- Added a first cached-prefix attention path that uses FP8 Q/K/V bytes, rocWMMA FP8 QK tiles, online softmax, and V accumulation in one HIPRTC kernel.
- Scope is intentionally narrow: RDNA4/gfx12 only, `head_dim=16`, `value_dim=16`, and `q_heads/kv_heads` must be a multiple of 16.
- The runtime smoke executor is `cached_prefix_rocwmma_fp8`.
- Q is quantized to FP8 for this executor, so `q_bytes_total` is one byte per Q element and the sampled reference uses decoded FP8 Q.

## Verification

```text
cargo fmt --all --check
git diff --check
cargo check -p ullm-engine
cargo test -p ullm-runtime-sys cached_prefix_attn_fp8_e4m3_rocwmma -- --test-threads=1
cargo build -p ullm-engine --release
```

The targeted runtime-sys test passed both CPU fallback and RDNA4 HIP execution:

```text
test tests::cpu_cached_prefix_attn_fp8_e4m3_rocwmma_computes_expected_values ... ok
test tests::first_hip_cached_prefix_attn_fp8_e4m3_rocwmma_computes_expected_values_when_available ... ok
```

V620/RDNA2 rejected the executor as expected:

```text
failed to run runtime fp8 e4m3 cached prefix rocwmma attention: fp8 e4m3 cached prefix rocWMMA attention requires RDNA4/gfx12 HIP device
```

## R9700 Smoke Results

Device: R9700, runtime device index `2`.

| executor | L | M | q_heads | kv_heads | dim | repeats | wall_ms_mean | input tok/s | attention pair tok/s | sampled max abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cached_prefix_rocwmma_fp8 | 3 | 2 | 16 | 1 | 16 | 3 | 0.046388 | 43114.908417 | 194017.087875 | 0.000000001 |
| cached_prefix_rocwmma_fp8 | 31 | 3 | 16 | 1 | 16 | 3 | 0.053074 | 56524.497089 | 1865308.403937 | 0.000000001 |
| cached_prefix_rocwmma_fp8 | 31 | 3 | 32 | 2 | 16 | 3 | 0.055704 | 53856.096510 | 1777251.184834 | 0.000000002 |
| cached_prefix_rocwmma_fp8 | 4096 | 16 | 16 | 1 | 16 | 5 | 0.911433 | 17554.769785 | 72053552.580705 | 0.000000025 |
| cached_prefix_flash2 | 4096 | 16 | 16 | 1 | 16 | 5 | 3.252488 | 4919.310699 | 20191310.763249 | 0.000000010 |
| cached_prefix_chunked | 4096 | 16 | 16 | 1 | 16 | 5 | 3.778853 | 4234.088797 | 17378817.467691 | 0.000000002 |
| cached_prefix_rocwmma_fp8 | 16384 | 128 | 16 | 1 | 16 | 1 | 3.559969 | 35955.369274 | 591411891.508044 | 0.000000150 |
| cached_prefix_rocwmma_fp8 | 65536 | 512 | 16 | 1 | 16 | 1 | 14.627084 | 35003.559151 | 2302971665.439263 | 0.000000415 |

## Notes

- The comparison against `cached_prefix_flash2` and `cached_prefix_chunked` is not perfectly apples-to-apples because the new rocWMMA path quantizes Q to FP8 while the older FP8 K/V paths keep Q as F32.
- The important milestone is that the FA2-like structure is now connected to the cached-prefix measurement path, not just a standalone probe.
- The next implementation step is widening beyond the current fixed `16x16` tile shape and moving toward Q row grouping/KV tile reuse for real model head dimensions.
