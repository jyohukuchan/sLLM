# T1 Package Prefill Component Real-Batch Runner v1

Date: 2026-07-08

## Summary

This adds a package-backed prefill component workload runner and validates that its output can be
converted into `inference-benchmark-result-v0.1` JSONL with real-batch metadata.

This is not final package total throughput. It proves that a `.ullm.d` package component can be run
through the T1 real-batch result path before whole-model prefill/decode scheduling is connected.

## Command

```text
python3 tools/run-package-prefill-component-workload.py \
  --workload-json benchmarks/workloads/r9700-aq4-package-prefill-component-real-batch-smoke.json \
  --output-dir benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-package-prefill-component-runner-v1 \
  --overwrite
```

## Result

| field | value |
| --- | --- |
| status | `ok` |
| command | `package-prefill-aq4-matvec-batch-smoke` |
| package | `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d` |
| tensor | `model.language_model.layers.3.self_attn.k_proj.weight` |
| device | R9700, runtime device index `2` |
| prompt tokens | `2` |
| batching mode | `real` |
| prefill real batch | `true` |
| executor | `aq4_matvec_batch_f32` |
| request parallelism | `1` |
| token parallelism | `2` |
| prefill total input tok/s | `19063.596157` |
| wall ms mean | `0.104912` |
| sampled max abs diff | `0.000000101` |
| verified | `true` |

Artifacts:

- `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-package-prefill-component-runner-v1/results.jsonl`
- `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-package-prefill-component-runner-v1/workload.json`
- `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-package-prefill-component-runner-v1/execution-plan.json`

## Interpretation

`tools/run-package-prefill-component-workload.py` is the first runner that starts from a `.ullm.d`
package path and emits a real-batch prefill component row through `tools/run-external-benchmark.py
--parse ullm-component-prefill`.

The current smoke covers token-parallel prefill for one request. It does not yet provide:

- request-batch prefill across `batch=1/4/8`
- decode real batch
- end-to-end package total throughput
- SQ FP8 candidate throughput

## Next Action

1. Add broader package-backed component cases, especially self-attention layer and linear-attention layer batch smokes.
2. Extend package-level runner work from component prefill rows to full package prefill/decode total throughput.
3. Use only full package total-throughput rows for AQ4/SQ performance decisions.
