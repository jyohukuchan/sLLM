# Phase C16: RDNA4 FP8 rocWMMA Value Group Sweep Axis v1

## Summary

- Added `--rocwmma-value-group-widths auto|16|32|64|128|256` to `tools/run-runtime-cached-prefix-sweep.py`.
- The sweep axis applies only to `cached_prefix_rocwmma_fp8`.
- `auto` leaves `ULLM_ROCWMMA_CACHED_PREFIX_VALUE_GROUP_WIDTH` unset, so the runtime heuristic chooses the width.
- Numeric widths set `ULLM_ROCWMMA_CACHED_PREFIX_VALUE_GROUP_WIDTH` per case and are recorded in JSONL `required_env`.
- Summary markdown now includes a `rocWMMA value group` column.

## Verification

```text
python3 -m py_compile tools/run-runtime-cached-prefix-sweep.py
python3 tools/run-runtime-cached-prefix-sweep.py --executors cached_prefix_rocwmma_fp8 --cached-prefix-tokens 4096 --new-tokens 16,128 --kv-cache-dtype fp8_e4m3 --q-heads 16 --kv-heads 1 --rocwmma-value-group-widths auto,16,64 --output-jsonl /tmp/ullm-rocwmma-vgw-dryrun.jsonl --summary-md /tmp/ullm-rocwmma-vgw-dryrun.md --dry-run
python3 tools/run-runtime-cached-prefix-sweep.py --executors cached_prefix_rocwmma_fp8 --cached-prefix-tokens 4096 --new-tokens 16,128 --kv-cache-dtype fp8_e4m3 --q-heads 16 --kv-heads 1 --head-dim 256 --value-dim 256 --measured-repeats 2 --long-measured-repeats 2 --rocwmma-value-group-widths auto,16,64 --output-jsonl /tmp/ullm-rocwmma-vgw-sweep.jsonl --summary-md /tmp/ullm-rocwmma-vgw-sweep.md --timeout-seconds 120 --keep-going
git diff --check -- tools/run-runtime-cached-prefix-sweep.py
```

Note: the full repository `git diff --check` is still blocked by an unrelated dirty `README.md` trailing whitespace change outside this work.

## R9700 Script-Driven Sweep

Device: R9700, runtime device index `2`.

Shape: `L=4096,q_heads=16,kv_heads=1,head_dim=256,value_dim=256`, `kv_cache_dtype=fp8_e4m3`, `repeats=2`.

| M | rocWMMA value group | required env | wall_ms_mean | input tok/s | attention pair/s | sampled max abs diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 16 | auto | `{}` | 17.804503 | 898.649066 | 3688505.093346 | 0.000000719 |
| 16 | 16 | `ULLM_ROCWMMA_CACHED_PREFIX_VALUE_GROUP_WIDTH=16` | 18.245964 | 876.906233 | 3599261.633990 | 0.000000719 |
| 16 | 64 | `ULLM_ROCWMMA_CACHED_PREFIX_VALUE_GROUP_WIDTH=64` | 20.665049 | 774.254153 | 3177926.168963 | 0.000000719 |
| 128 | auto | `{}` | 15.295860 | 8368.277429 | 34816218.244675 | 0.000000723 |
| 128 | 16 | `ULLM_ROCWMMA_CACHED_PREFIX_VALUE_GROUP_WIDTH=16` | 48.100149 | 2661.114418 | 11071566.535064 | 0.000000723 |
| 128 | 64 | `ULLM_ROCWMMA_CACHED_PREFIX_VALUE_GROUP_WIDTH=64` | 15.443554 | 8288.247640 | 34483254.307914 | 0.000000723 |

## Interpretation

- The benchmark harness can now reproduce the value-group tradeoff without hand-written shell loops.
- The current heuristic tracks the useful side of the tradeoff for these two cases:
  - `M=16`: auto chooses the short-chunk path and stays near the value-group-16 result.
  - `M=128`: auto chooses the larger-chunk path and stays near the value-group-64 result.
- This makes the RDNA4 FP8 rocWMMA path easier to include in larger SQ candidate sweeps.

## Next Action

Use the new sweep axis when testing longer cached-prefix grids, especially `M={16,128,512}` and `L={4096,16384,65536}`, before making the next multi-query-token tile change.
