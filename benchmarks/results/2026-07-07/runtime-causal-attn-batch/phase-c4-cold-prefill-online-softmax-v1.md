# Runtime causal attention online softmax v1

Date: 2026-07-07

Change:

- `ullm_causal_attn_f32_kernel` and `ullm_causal_attn_batch_f32_kernel`
  now use an online softmax path when `value_dim <= blockDim.x`.
- The common Qwen3.5 shape on R9700 is `value_dim=256`, `blockDim.x=256`.
- This reduces q/k score dot-product passes from three to one for the
  normal prefill attention shape.
- The previous three-pass path remains as fallback for larger value dims.

## Runtime Batched Causal Attention

Command shape:

```bash
ULLM_REQUIRE_HIP_CAUSAL_ATTN_BATCH_KERNEL=1 \
  target/release/ullm-engine runtime-causal-attn-batch-smoke \
  2 B N 3 16 4 256 256
```

| B | N | old mean ms | new mean ms | speedup | new input tok/s | new attention pair/s | sample diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 18.603645 | 7.239056 | 2.570x | 70727.453966 | 18141591.942375 | 0 |
| 4 | 512 | 68.139151 | 27.982984 | 2.435x | 73187.334131 | 18772551.204689 | 0 |
| 8 | 512 | 135.607344 | 55.488527 | 2.444x | 73817.061761 | 18934076.341788 | 0 |
| 1 | 2048 | 274.000056 | 118.997799 | 2.303x | 17210.402306 | 17632057.162021 | 0 |
| 4 | 2048 | 1095.987421 | 460.720775 | 2.379x | 17780.834836 | 18216465.289789 | 0 |
| 8 | 2048 | 2208.166702 | 908.260240 | 2.431x | 18038.882770 | 18480835.397837 | 0 |
| 1 | 4096 | 1127.648016 | 464.365975 | 2.428x | 8820.629037 | 18069058.582561 | 0 |
| 4 | 4096 | 4649.452562 | 1860.373915 | 2.499x | 8806.831715 | 18040794.768930 | 0 |
| 1 | 8192 | n/a | 1887.833494 | n/a | 4339.365747 | 17776211.782796 | 0 |

## Package Self-Attention Attention Batch

Command shape:

```bash
ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 \
ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL=1 \
ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1 \
  target/release/ullm-engine package-self-attn-attention-batch-smoke \
  /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d \
  2 1048576 3 len:N 3
```

| N | old mean ms | new mean ms | speedup | new token/s | attention verification | attention diff |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 512 | 281.601274 | 33.528531 | 8.399x | 15270.576724 | full | 0.000011325 |
| 2048 | n/a | 227.980683 | n/a | 8983.217245 | sampled | 0.000000238 |
| 4096 | n/a | 694.683770 | n/a | 5896.207994 | sampled | 0.000000298 |
| 8192 | n/a | 2402.205139 | n/a | 3410.200015 | sampled | 0.000000164 |

## Package Self-Attention Layer Batch

Command shape:

```bash
ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 \
ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL=1 \
ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1 \
  target/release/ullm-engine package-self-attn-layer-batch-smoke \
  /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d \
  2 1048576 3 len:N 3
```

| N | old mean ms | new mean ms | speedup | new token/s | layer diff |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 141.768580 | 133.188978 | 1.064x | 3844.161940 | 0 |
| 2048 | 777.894286 | 628.195305 | 1.238x | 3260.132609 | 0 |
| 4096 | 2182.970006 | 1518.104339 | 1.438x | 2698.101768 | 0 |
| 8192 | 6892.180390 | 4162.250951 | 1.656x | 1968.165807 | 0 |

## Interpretation

- The raw runtime causal attention component improved by roughly `2.3-2.5x`
  across the saved `N=512/2048/4096` batch-width grid.
- Attention pair throughput moved from roughly `7.2-7.7M pair/s` to roughly
  `17.8-18.9M pair/s`.
- The self-attention attention package smoke at `N=512` dropped from
  `281.601274 ms` to `33.528531 ms`.
- The full self-attention layer partial smoke improves more strongly as
  context grows: `N=8192` moved from `6892.180390 ms` to `4162.250951 ms`.
- Batch width still does not create superlinear throughput gains. The new
  kernel mostly reduces repeated score work per query/head. Further prefill
  gains require tiled/block causal attention and better K/V reuse across
  neighboring timesteps or heads.

## Verification

- `cargo fmt --all --check`
- `cargo test -p ullm-runtime-sys causal_attn -- --test-threads=1`
- `cargo build -p ullm-engine --release`
- R9700 smokes with required HIP kernel flags shown above
