# Runtime cached prefix FP8 E4M3 KV cache builtin convert v2

Date: 2026-07-08

Scope:

- Device: R9700/RDNA4, runtime device index `2`, backend `hip`, name `AMD Radeon Graphics`.
- Workload: cached prefix prefill with `q_heads=16`, `kv_heads=4`, `head_dim=256`, `value_dim=256`.
- Executor: `cached_prefix_chunked`.
- Change from v1: device-side FP8 E4M3 to F32 conversion now uses `__builtin_amdgcn_cvt_f32_fp8` on `__gfx1200__` / `__gfx1201__`.
- Non-gfx12 builds keep the previous bit-pattern fallback.
- Scale is still applied as a separate F32 multiply. `__builtin_amdgcn_cvt_scalef32_f32_fp8` is not available for gfx1200 in the installed ROCm 7.2 compiler.

## ISA probe

The local ROCm 7.2 compiler lowers a minimal gfx1200 kernel:

```cpp
__builtin_amdgcn_cvt_f32_fp8(x, 0)
```

to:

```asm
v_cvt_f32_fp8_e32
```

The packed builtin lowers to:

```asm
v_cvt_pk_f32_fp8_e32
```

CK has a note that the packed builtin can produce incorrect results on gfx12 in at least one code path, so this runtime change uses the scalar conversion only.

## FP8 builtin results

Command shape:

```bash
ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_FP8_E4M3_KERNEL=1 \
  target/release/ullm-engine runtime-cached-prefix-attn-smoke \
  2 L M 3 16 4 256 256 cached_prefix_chunked fp8_e4m3
```

| L | M | repeats | mean ms | min ms | max ms | new tok/s | pair/s mean | KV bytes | sampled diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 16 | 3 | 4.292718 | 4.014467 | 4.506805 | 3727.241985 | 15298464.725731 | 8421376 | 0 |
| 4096 | 128 | 3 | 31.644946 | 31.482477 | 31.953764 | 4044.879710 | 16828722.033528 | 8650752 | 0 |
| 4096 | 512 | 3 | 131.048745 | 129.782030 | 131.800485 | 3906.943176 | 17004970.173503 | 9437184 | 0 |
| 16384 | 16 | 3 | 15.562430 | 15.533720 | 15.585001 | 1028.117052 | 16853408.778847 | 33587200 | 0 |
| 16384 | 128 | 3 | 124.919123 | 124.546993 | 125.449098 | 1024.662976 | 16854168.961930 | 33816576 | 0 |
| 16384 | 512 | 3 | 499.328782 | 495.390746 | 502.458245 | 1025.376502 | 17062777.687027 | 34603008 | 0 |
| 65536 | 16 | 3 | 112.763136 | 79.315767 | 159.000780 | 141.890342 | 9300131.533234 | 134250496 | 0 |
| 65536 | 128 | 3 | 1225.180136 | 1193.930103 | 1260.213892 | 104.474433 | 6853575.040332 | 134479872 | 0 |
| 65536 | 512 | 3 | 4908.104574 | 4701.718765 | 5016.236354 | 104.317256 | 6863293.047676 | 135266304 | 0 |

## Same-build F32 comparison

Command shape:

```bash
ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_KERNEL=1 \
  target/release/ullm-engine runtime-cached-prefix-attn-smoke \
  2 L M 3 16 4 256 256 cached_prefix_chunked f32
```

| L | M | repeats | mean ms | min ms | max ms | new tok/s | pair/s mean | KV bytes | sampled diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 16 | 3 | 4.201120 | 3.937526 | 4.391444 | 3808.507905 | 15632020.696701 | 33685504 | 0 |
| 4096 | 128 | 3 | 30.979069 | 30.341778 | 31.516798 | 4131.822082 | 17190445.772600 | 34603008 | 0 |
| 4096 | 512 | 3 | 128.673577 | 128.230236 | 129.082020 | 3979.060917 | 17318862.642429 | 37748736 | 0 |
| 16384 | 16 | 3 | 19.467442 | 19.205981 | 19.857942 | 821.885059 | 13472750.837479 | 134348800 | 0 |
| 16384 | 128 | 3 | 170.708524 | 168.666673 | 172.389805 | 749.816103 | 12333350.173604 | 135266304 | 0 |
| 16384 | 512 | 3 | 671.855301 | 669.151889 | 675.868962 | 762.068855 | 12681206.782649 | 138412032 | 0 |
| 65536 | 16 | 3 | 83.326828 | 78.075317 | 86.044670 | 192.014989 | 12585526.476539 | 537001984 | 0 |
| 65536 | 128 | 3 | 669.227408 | 665.459338 | 671.441168 | 191.265329 | 12547101.185345 | 537919488 | 0 |
| 65536 | 512 | 3 | 2506.124152 | 2489.046163 | 2519.969381 | 204.299535 | 13441377.185211 | 541065216 | 0 |

## Ratios

| L | M | FP8 builtin / F32 tok/s | FP8 builtin / v1 bit-cvt tok/s | KV byte ratio vs F32 |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 16 | 0.979x | 1.052x | 0.250x |
| 4096 | 128 | 0.979x | 1.088x | 0.250x |
| 4096 | 512 | 0.982x | 1.057x | 0.250x |
| 16384 | 16 | 1.251x | 1.129x | 0.250x |
| 16384 | 128 | 1.367x | 1.072x | 0.250x |
| 16384 | 512 | 1.346x | 1.079x | 0.250x |
| 65536 | 16 | 0.739x | 1.287x | 0.250x |
| 65536 | 128 | 0.546x | 1.045x | 0.250x |
| 65536 | 512 | 0.511x | 1.042x | 0.250x |

## Interpretation

- Dedicated FP8 conversion improves the FP8 path across the entire saved grid.
- The improvement is largest at `L=65536,M=16`, but that case is still variance-heavy.
- `L=4096` is now almost tied with F32, at about `98%` of F32 tok/s while using `25%` of K/V bytes.
- `L=16384` is now clearly faster than F32, reaching `1.25-1.37x`.
- `L=65536` remains slower than F32 for larger chunks. This points to the cached-prefix kernel structure, not only conversion instruction choice:
  - FP8 K is still converted repeatedly while scanning the long prefix.
  - The current kernel does not tile and reuse decoded K/V values.
  - Scale is still a separate multiply because gfx1200 does not expose the scale-convert builtin in this compiler target.

## Verification

- `cargo fmt --all --check`
- `cargo test -p ullm-runtime-sys cached_prefix_attn_fp8_e4m3 -- --test-threads=1`
- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine --release`
- R9700 FP8 smokes with `ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_FP8_E4M3_KERNEL=1`
- R9700 F32 smokes with `ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_KERNEL=1`
