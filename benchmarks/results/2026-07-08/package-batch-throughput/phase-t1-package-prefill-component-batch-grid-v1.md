# T1 Package Prefill Component Batch Grid v1

Date: 2026-07-08

## Summary

This extends the package-backed prefill component runner with a batch-width case.

The new `component_args_template` field lets a workload case derive component arguments from
`prompt_tokens * concurrent_requests`. For this smoke, the package AQ4 `k_proj` component is run with
flattened token parallelism:

- workload `batch_size=1`, `prompt_tokens=2` -> component `len:2`
- workload `batch_size=4`, `prompt_tokens=2` -> component `len:8`

This is still not full package total throughput. It is a package-backed component row that records
both the requested workload batch and the executor's actual token/request parallelism.

## Command

```text
python3 tools/run-package-prefill-component-workload.py \
  --workload-json benchmarks/workloads/r9700-aq4-package-prefill-component-real-batch-smoke.json \
  --output-dir benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-package-prefill-component-batch-grid-v1 \
  --overwrite
```

## Result

| case | workload batch | prompt/request | component tokens | token parallelism | request parallelism | prefill tok/s | sampled max abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `aq4-r9700-package-kproj-batch-prefill-pp2` | `1` | `2` | `2` | `2` | `1` | `19143.885443` | `0.000000101` |
| `aq4-r9700-package-kproj-batch-prefill-b4-pp2` | `4` | `2` | `8` | `8` | `1` | `65513.626834` | `0.000000123` |

Both rows reported:

- `status=ok`
- `batching.mode=real`
- `batching.prefill_real_batch=true`
- `batching.prefill_executor=aq4_matvec_batch_f32`
- `correctness.verified_all=true`

## Interpretation

The B=4 row preserves:

- `workload.batch_size=4`
- `workload.concurrent_requests=4`
- `workload.prompt_tokens_per_request=[2,2,2,2]`
- `workload.component_total_input_tokens=8`
- `batching.prefill_executor_token_parallelism=8`
- `batching.prefill_executor_request_parallelism=1`

This distinction matters. The component executor is real-batch over tokens, but it is not yet a
request-batch scheduler and it does not measure decode or end-to-end throughput.

## Next Action

1. Add package-backed layer component cases where request boundaries matter, especially self-attention.
2. Add a true request-batch package prefill path before using these rows for SQ throughput decisions.
3. Add decode/end-to-end total throughput rows after prefill request-batch accounting is stable.
