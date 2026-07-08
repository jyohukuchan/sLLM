# T1 Model-Loop Hybrid Throughput Smoke v1

## Summary

This records the first JSONL-converted model-loop throughput smoke for the package stack runner.

It is not final full language-model throughput. The command uses a synthetic selected-layer stack
over layers `3,7`, but it proves that the `Qwen3DecoderLayerStackRequestDecodeRunner` path can emit
timed throughput fields and preserve real decode batching in `inference-benchmark-result-v0.1`.

Result directory:

- `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-model-loop-hybrid-throughput-smoke-v1/`

Command:

```bash
python3 tools/run-external-benchmark.py \
  --parse ullm-model-loop-throughput \
  -- target/debug/ullm-engine package-self-attn-mlp-block-model-loop-smoke \
  /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d \
  2 1048576 3,7 3 32 10000000 3
```

## Result

| case | layers | requests | prefill real | decode real | decode request parallelism | prefill total tok/s | decode generated tok/s | end-to-end tok/s | VRAM consumed bytes | verified |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `aq4-r9700-model-loop-layers3-7-seq3` | `3,7` | 3 | false | true | 2 | 78.702126 | 78.214266 | 78.492300 | 1519312896 | true |

## Interpretation

- `tools/run-external-benchmark.py --parse ullm-model-loop-throughput` now converts model-loop
  key-value stdout into `inference-benchmark-result-v0.1`.
- The row preserves `batching.mode=hybrid`, `prefill_real_batch=false`,
  `decode_real_batch=true`, and `decode_executor_request_parallelism=2`.
- CSV fields such as `prompt_tokens_csv`, `generated_tokens_csv`, and `layers_csv` avoid losing
  workload shape when key-value stdout also contains debug arrays with spaces.
- The next T1 step is to move from this synthetic selected-layer stack to the real full-package
  token path, then emit `batching.mode=real` rows for `batch=1/4/8`.
