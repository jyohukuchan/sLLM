# Runtime cached prefix FP8 E4M3 KV cache v1

Date: 2026-07-08

Scope:

- Device: R9700/RDNA4, runtime device index `2`, backend `hip`, name `AMD Radeon Graphics`.
- Workload: cached prefix prefill with Qwen3.5-like shape `q_heads=16`, `kv_heads=4`, `head_dim=256`, `value_dim=256`.
- Executor: `cached_prefix_chunked`.
- FP8 format: per-tensor scaled E4M3 byte cache for K and V; Q and output remain F32.
- Implementation scope: runtime cached-prefix attention path only. Paged decode and package model decode state still use F32 K/V cache.
- Base commit before this change: `1aae138`.

Command shape:

```bash
ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_FP8_E4M3_KERNEL=1 \
  target/release/ullm-engine runtime-cached-prefix-attn-smoke \
  2 L M REPEATS 16 4 256 256 cached_prefix_chunked fp8_e4m3

ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_KERNEL=1 \
  target/release/ullm-engine runtime-cached-prefix-attn-smoke \
  2 L M REPEATS 16 4 256 256 cached_prefix_chunked f32
```

The smoke defaults to `kv_cache_dtype=fp8_e4m3` when the dtype argument is omitted.

## FP8 results

| L | M | repeats | mean ms | min ms | max ms | new tok/s | pair/s mean | KV bytes | sampled diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 16 | 3 | 4.514144 | 4.224012 | 4.792753 | 3544.414803 | 14548050.560782 | 8421376 | 0 |
| 4096 | 128 | 3 | 34.415441 | 33.513757 | 35.158578 | 3719.260782 | 15473984.482721 | 8650752 | 0 |
| 4096 | 512 | 3 | 138.495114 | 138.191465 | 138.995601 | 3696.881312 | 16090675.911039 | 9437184 | 0 |
| 16384 | 16 | 3 | 17.576305 | 17.489169 | 17.721485 | 910.316492 | 14922363.089063 | 33587200 | 0 |
| 16384 | 128 | 3 | 133.929520 | 133.864949 | 134.050213 | 955.726562 | 15720268.352787 | 33816576 | 0 |
| 16384 | 512 | 3 | 538.731734 | 533.394509 | 542.827543 | 950.380251 | 15814802.558617 | 34603008 | 0 |
| 65536 | 16 | 7 | 145.119930 | 84.429518 | 163.741796 | 110.253636 | 7226519.458322 | 134250496 | 0 |
| 65536 | 128 | 3 | 1279.703884 | 1252.990304 | 1298.654594 | 100.023139 | 6561567.955523 | 134479872 | 0 |
| 65536 | 512 | 3 | 5113.254823 | 5034.055066 | 5159.111604 | 100.131916 | 6587929.052250 | 135266304 | 0 |

## Same-build F32 comparison

| L | M | repeats | mean ms | min ms | max ms | new tok/s | pair/s mean | KV bytes | sampled diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 16 | 3 | 4.199371 | 3.919691 | 4.414242 | 3810.094719 | 15638533.773950 | 33685504 | 0 |
| 4096 | 128 | 3 | 30.809383 | 30.170527 | 31.500364 | 4154.578493 | 17285123.820883 | 34603008 | 0 |
| 4096 | 512 | 3 | 129.121143 | 127.840343 | 129.813204 | 3965.268492 | 17258831.111803 | 37748736 | 0 |
| 16384 | 16 | 3 | 19.420526 | 19.219328 | 19.787831 | 823.870565 | 13505298.234365 | 134348800 | 0 |
| 16384 | 128 | 3 | 172.190438 | 171.150760 | 173.100620 | 743.362997 | 12227206.251720 | 135266304 | 0 |
| 16384 | 512 | 3 | 666.420772 | 653.344247 | 676.354781 | 768.283376 | 12784619.511022 | 138412032 | 0 |
| 65536 | 16 | 3 | 83.547650 | 78.540646 | 86.271817 | 191.507480 | 12552262.042271 | 537001984 | 0 |
| 65536 | 128 | 3 | 668.713389 | 661.425821 | 672.721901 | 191.412348 | 12556745.748343 | 537919488 | 0 |
| 65536 | 512 | 3 | 2509.902446 | 2484.038522 | 2544.537880 | 203.991992 | 13421143.141912 | 541065216 | 0 |

## FP8 / F32 speed ratio

| L | M | FP8 tok/s | F32 tok/s | ratio | KV byte ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 16 | 3544.414803 | 3810.094719 | 0.930x | 0.250x |
| 4096 | 128 | 3719.260782 | 4154.578493 | 0.895x | 0.250x |
| 4096 | 512 | 3696.881312 | 3965.268492 | 0.932x | 0.250x |
| 16384 | 16 | 910.316492 | 823.870565 | 1.105x | 0.250x |
| 16384 | 128 | 955.726562 | 743.362997 | 1.286x | 0.250x |
| 16384 | 512 | 950.380251 | 768.283376 | 1.237x | 0.250x |
| 65536 | 16 | 110.253636 | 191.507480 | 0.576x | 0.250x |
| 65536 | 128 | 100.023139 | 191.412348 | 0.522x | 0.250x |
| 65536 | 512 | 100.131916 | 203.991992 | 0.491x | 0.250x |

## Interpretation

- FP8 E4M3 K/V cache reduces resident K/V cache bytes to exactly `25%` of the F32 path for this layout.
- Correctness guard passed in all measured cases against references computed from the decoded FP8 K/V values.
- The current FP8 path is not uniformly faster:
  - `L=4096` is `7-11%` slower than F32.
  - `L=16384` is `10-29%` faster than F32.
  - `L=65536` is about `42-51%` slower by mean tok/s.
- The long-prefix regression is likely dequant cost amplified by the cached-prefix kernel structure. The current kernel still decodes FP8 K repeatedly while computing score and value accumulation. Reducing K score recomputation, caching decoded values per tile, or moving to a tiled/FlashAttention-like kernel is the next optimization direction.
- `L=65536,M=16` has large repeat variance; the repeat-7 mean is saved, but the min run is much closer to the F32 mean. Do not use only this single small-M mean to judge FP8 viability.

## Verification

- `cargo fmt --all --check`
- `cargo check -p ullm-runtime-sys`
- `cargo check -p ullm-engine`
- `cargo test -p ullm-runtime-sys cached_prefix_attn_fp8_e4m3 -- --test-threads=1`
- `cargo build -p ullm-engine --release`
- R9700 FP8 smokes with `ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_FP8_E4M3_KERNEL=1`
- R9700 F32 comparison smokes with `ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_KERNEL=1`
