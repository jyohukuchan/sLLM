# Runtime cached prefix attention online softmax v1

Date: 2026-07-07

Change:

- `ullm_cached_prefix_attn_f32_kernel` now uses an online softmax path
  when `value_dim <= blockDim.x`.
- The measured Qwen3.5-like shape is `value_dim=256`, `blockDim.x=256`.
- The previous shared-score kernel still recomputed q/k scores for max,
  denominator, and value accumulation. This change reduces those score
  dot-product passes from three to one for the normal cached prefix path.
- The previous three-pass path remains as fallback for larger value dims.

Command shape:

```bash
ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_KERNEL=1 \
  target/release/ullm-engine runtime-cached-prefix-attn-smoke \
  2 L M REPEATS 16 4 256 256 cached_prefix_chunked
```

Device:

- index: `2`
- backend: `hip`
- name: `AMD Radeon Graphics`
- scope: R9700/RDNA4

## Results

| L | M | repeats | old mean ms | new mean ms | speedup | new tok/s | new pair/s | diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1 | 3 | 47.865108 | 4.039271 | 11.850x | 247.569405 | 1014291.851649 | 0 |
| 4096 | 16 | 3 | 55.839281 | 4.193457 | 13.316x | 3815.467763 | 15660587.434186 | 0 |
| 4096 | 128 | 1 | 435.542576 | 30.812317 | 14.135x | 4154.182887 | 17283477.902684 | 0 |
| 16384 | 1 | 1 | 200.889494 | 18.882177 | 10.639x | 52.959995 | 867749.518501 | 0 |
| 16384 | 16 | 1 | 303.245627 | 19.256383 | 15.748x | 830.893320 | 13620418.746345 | 0 |
| 16384 | 128 | 1 | 1748.356354 | 170.666624 | 10.244x | 750.000188 | 12336378.084095 | 0 |
| 65536 | 1 | 1 | 911.030189 | 76.723657 | 11.874x | 13.033790 | 854195.466725 | 0 |
| 65536 | 16 | 1 | 1738.452511 | 78.523667 | 22.139x | 203.760224 | 13355362.020982 | 0 |
| 65536 | 128 | 1 | 7959.297024 | 673.121102 | 11.824x | 190.158947 | 12474522.006591 | 0 |

## Interpretation

- The cached prefix component improved by `10.2-22.1x` over the prior
  shared-score kernel across the saved Phase C4 `L/M` grid.
- `M=16/128` now reaches roughly `12.3-17.3M` attention pairs/s, close to
  the cold causal attention online-softmax component range.
- `M=1` remains much lower in pair/s because it is decode-like and does not
  expose enough token-level parallelism.
- Long prefix `L=65536, M=128` moved from `7959.297024 ms` to
  `673.121102 ms`, so it is now practical enough for repeated SQ candidate
  comparison runs.
- Further cached prefix work should focus on batch/request dimensions,
  `M=1` decode-like boundary, and K/V reuse/coalescing rather than repeated
  score recomputation.

## Verification

- `cargo fmt --all --check`
- `cargo test -p ullm-runtime-sys cached_prefix_attn -- --test-threads=1`
- `cargo build -p ullm-engine --release`
- R9700 smokes with `ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_KERNEL=1`
