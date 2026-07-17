# RDNA4 FP8 cached-prefix auto executor

## 前回の要点

- `cached_prefix_flash2_fp8q` と `cached_prefix_rocwmma_fp8` の比較で、短いchunkはscalar FP8-Q flash2、数十token以上のchunkはrocWMMAが有利になり始めることが分かった。
- Phase C17の長prefix gridでは、少なくとも `M=16` はflash2_fp8q、`M=128/512` はrocWMMAを使い分ける必要があった。

## 今回の変更点

- `runtime-cached-prefix-attn-smoke` に `cached_prefix_rdna4_fp8_auto` を追加した。
- auto executorは `new_prefill_tokens < 64` で `cached_prefix_flash2_fp8q`、`new_prefill_tokens >= 64` で `cached_prefix_rocwmma_fp8` に解決する。
- smoke出力へ `resolved_executor` を追加した。
- `tools/run-runtime-cached-prefix-sweep.py` にauto executor対応を追加し、JSONLとsummaryで解決先を保存するようにした。
- R9700 smokeで `M=16/64/128` を確認し、全て sampled guard が通った。

## 確認した値

`L=4096,q_heads=16,kv_heads=1,head_dim=256,value_dim=256,kv_cache_dtype=fp8_e4m3,measured_repeats=2`:

| M | auto resolved | auto ms | explicit flash2_fp8q ms | explicit rocWMMA ms |
| ---: | --- | ---: | ---: | ---: |
| 16 | `cached_prefix_flash2_fp8q` | 4.254524 | 4.280219 | 6.978157 |
| 64 | `cached_prefix_rocwmma_fp8` | 16.039908 | 18.344833 | 14.338819 |
| 128 | `cached_prefix_rocwmma_fp8` | 16.143782 | 27.364147 | 16.206428 |

## 次の行動

1. SQ候補のR9700 FP8 cached-prefix測定では、まず `cached_prefix_rdna4_fp8_auto` を使う。
2. 閾値やvalue group調整が必要な場合は、明示executorを併走させる。
3. kernelとして次に進めるなら、短chunkを改善するmulti-query-token tile化を扱う。
