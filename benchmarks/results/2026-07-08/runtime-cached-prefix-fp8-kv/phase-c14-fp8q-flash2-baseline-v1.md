# Phase C14: FP8 Q Cached-Prefix Flash2 Baseline v1

## Summary

- Added `cached_prefix_flash2_fp8q`, a scalar FlashAttention2-style cached-prefix executor that reads Q/K/V as FP8 E4M3 byte tensors and writes F32 output.
- The existing `cached_prefix_flash2` executor remains the F32-Q / FP8-KV baseline.
- The goal is to separate FP8 Q input/dequant overhead from the slower RDNA4 rocWMMA prototype.

## Verification

```text
cargo fmt --all --check
cargo check -p ullm-engine
cargo test -p ullm-runtime-sys cached_prefix_attn_fp8_e4m3_flash2_fp8q -- --test-threads=1
cargo build -p ullm-engine --release
python3 tools/run-runtime-cached-prefix-sweep.py --executors cached_prefix_flash2_fp8q --cached-prefix-tokens 4096 --new-tokens 16 --kv-cache-dtype fp8_e4m3 --output-jsonl /tmp/ullm-cached-prefix-fp8q-dryrun.jsonl --dry-run
git diff --check -- runtime/src/ullm_runtime_hiprtc_sources.inc runtime/src/ullm_runtime.cpp runtime/src/ullm_runtime_api_attention.inc runtime/include/ullm_runtime.h crates/ullm-runtime-sys/src/lib.rs crates/ullm-engine/src/main.rs tools/run-runtime-cached-prefix-sweep.py
```

Note: the full repository `git diff --check` is still blocked by an unrelated dirty `README.md` trailing whitespace change outside this work.

## R9700 Smoke Results

Device: R9700, runtime device index `2`.

Shape: `L=4096,M=16,q_heads=16,kv_heads=1,head_dim=256,value_dim=256`, `kv_cache_dtype=fp8_e4m3`, `repeats=3`.

| executor | Q dtype | q bytes | wall_ms_mean | wall_ms_min | input tok/s | attention pair/s | sampled max abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cached_prefix_flash2 | F32 | 262144 | 3.856106 | 3.645929 | 4149.263890 | 17030653.637863 | 0.000000611 |
| cached_prefix_flash2_fp8q | FP8 E4M3 | 65536 | 4.123693 | 3.855122 | 3880.016943 | 15925529.541479 | 0.000002176 |
| cached_prefix_rocwmma_fp8 | FP8 E4M3 | 65536 | 20.403873 | 19.484865 | 784.164837 | 3218604.572138 | 0.000000719 |

## Interpretation

- Switching Q from F32 to FP8 in the scalar flash2 executor reduces Q input bytes by `4x`.
- The measured speed cost is small compared with the rocWMMA prototype: `cached_prefix_flash2_fp8q` is about `1.07x` slower than F32-Q flash2, while rocWMMA is about `4.95x` slower than FP8-Q flash2 on this shape.
- Therefore the current rocWMMA bottleneck is not explained by FP8 Q dequant alone. The remaining issue is the rocWMMA kernel structure, especially repeated QK/softmax work across value groups and insufficient K/V tile reuse.

## Next Action

Use `cached_prefix_flash2_fp8q` as the near-term FP8-Q baseline while continuing the RDNA4 FlashAttention2-like path. The next optimization should move toward multi-query-row/tile reuse rather than treating FP8 Q conversion as the primary bottleneck.
