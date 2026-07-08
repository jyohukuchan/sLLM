# Phase C18: RDNA4 FP8 Cached-Prefix Auto Executor v1

## Summary

- Added `cached_prefix_rdna4_fp8_auto` to `runtime-cached-prefix-attn-smoke`.
- The auto executor routes `new_prefill_tokens < 64` to `cached_prefix_flash2_fp8q` and `new_prefill_tokens >= 64` to `cached_prefix_rocwmma_fp8`.
- The smoke output now includes `resolved_executor` so the selected path is visible in logs and sweep JSONL.
- Updated `tools/run-runtime-cached-prefix-sweep.py` to accept the auto executor, record `workload.resolved_executor`, and only expand `--rocwmma-value-group-widths` for auto cases that resolve to rocWMMA.

This is a routing integration of the Phase C17 observation, not a final multi-query-token FlashAttention2 kernel. It makes the best current RDNA4 FP8 cached-prefix path available as a single executor for SQ candidate measurement.

## Verification

```text
cargo fmt --all --check
cargo check -p ullm-engine
python3 -m py_compile tools/run-runtime-cached-prefix-sweep.py
git diff --check -- crates/ullm-engine/src/main.rs tools/run-runtime-cached-prefix-sweep.py
cargo build -p ullm-engine --release
```

Sweep case expansion check:

```text
python3 tools/run-runtime-cached-prefix-sweep.py --binary target/release/ullm-engine --device-index 2 --cached-prefix-tokens 4096 --new-tokens 16,128 --executors cached_prefix_rdna4_fp8_auto --kv-cache-dtype fp8_e4m3 --q-heads 16 --kv-heads 1 --head-dim 256 --value-dim 256 --rocwmma-value-group-widths auto,16 --output-jsonl /tmp/ullm-rdna4-auto-dryrun.jsonl --summary-md /tmp/ullm-rdna4-auto-dryrun.md --dry-run
```

Dry-run output:

```text
cached_prefix_rdna4_fp8_auto-fp8_e4m3-l4096-m16: dry_run
cached_prefix_rdna4_fp8_auto-fp8_e4m3-l4096-m128-vgwauto: dry_run
cached_prefix_rdna4_fp8_auto-fp8_e4m3-l4096-m128-vgw16: dry_run
```

## R9700 Smoke Results

Device: R9700, runtime device index `2`.

Shape: `L=4096`, `q_heads=16,kv_heads=1,head_dim=256,value_dim=256`, `kv_cache_dtype=fp8_e4m3`, `measured_repeats=2`.

| M | executor | resolved executor | wall ms mean | input tok/s | pair/s | sampled max abs diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 16 | `cached_prefix_rdna4_fp8_auto` | `cached_prefix_flash2_fp8q` | 4.254524 | 3760.703167 | 15435806.148444 | 0.000002176 |
| 16 | `cached_prefix_flash2_fp8q` | `cached_prefix_flash2_fp8q` | 4.280219 | 3738.126484 | 15343140.152408 | 0.000002176 |
| 16 | `cached_prefix_rocwmma_fp8` | `cached_prefix_rocwmma_fp8` | 6.978157 | 2292.869020 | 9411080.891416 | 0.000000719 |
| 64 | `cached_prefix_rdna4_fp8_auto` | `cached_prefix_rocwmma_fp8` | 16.039908 | 3990.047823 | 16472912.438151 | 0.000000719 |
| 64 | `cached_prefix_flash2_fp8q` | `cached_prefix_flash2_fp8q` | 18.344833 | 3488.720775 | 14403183.719361 | 0.000002190 |
| 64 | `cached_prefix_rocwmma_fp8` | `cached_prefix_rocwmma_fp8` | 14.338819 | 4463.407884 | 18427179.448071 | 0.000000719 |
| 128 | `cached_prefix_rdna4_fp8_auto` | `cached_prefix_rocwmma_fp8` | 16.143782 | 7928.749286 | 32987561.402898 | 0.000000723 |
| 128 | `cached_prefix_flash2_fp8q` | `cached_prefix_flash2_fp8q` | 27.364147 | 4677.653659 | 19461378.048097 | 0.000002202 |
| 128 | `cached_prefix_rocwmma_fp8` | `cached_prefix_rocwmma_fp8` | 16.206428 | 7898.100680 | 32860047.877299 | 0.000000723 |

## Interpretation

- `M=16` resolves to scalar FP8-Q flash2 and matches the explicit `cached_prefix_flash2_fp8q` path.
- `M=64` and `M=128` resolve to rocWMMA. Both are faster than explicit scalar FP8-Q flash2 on this smoke shape.
- The `M=64` auto run had more variance than the explicit rocWMMA run, but it selected the correct path and passed the sampled guard.
- This gives SQ candidate sweeps a single RDNA4 FP8 cached-prefix executor while keeping `resolved_executor` visible for analysis.

## Next Action

- Use `cached_prefix_rdna4_fp8_auto` as the default R9700 FP8 cached-prefix executor for near-term SQ candidate measurement.
- Keep `cached_prefix_flash2_fp8q` and `cached_prefix_rocwmma_fp8` as explicit comparison axes when tuning the threshold or value-group policy.
- The next kernel-level optimization remains multi-query-token tiling, especially to reduce the short-chunk gap without losing the rocWMMA advantage at larger chunks.
