# T1 Component Prefill Real Batch Parser v1

Date: 2026-07-08

## Summary

`tools/run-external-benchmark.py` can now parse uLLM component prefill real-batch smoke output and
write an `inference-benchmark-result-v0.1` JSONL row.

This is not yet full package real-batch throughput. It is the first T1 bridge that allows a real
batch prefill component to flow through the same result schema as package batch throughput rows.

## Command

```text
python3 tools/run-external-benchmark.py \
  --run-id phase-t1-component-prefill-real-batch-parser-v1 \
  --case-id runtime-causal-attn-batch-b2-n32 \
  --parse ullm-component-prefill \
  --result-json benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-component-prefill-real-batch-parser-v1/raw.json \
  -- \
  target/debug/ullm-engine runtime-causal-attn-batch-smoke 2 2 32 1 4 1 16 16 flash2
```

## Result

| field | value |
| --- | ---: |
| status | `ok` |
| prefill total input tokens | `64` |
| prefill total input tok/s | `850713.136872` |
| attention pair/s | `14036766.758384` |
| batching mode | `real` |
| prefill real batch | `true` |
| request parallelism | `2` |
| token parallelism | `32` |
| sampled max abs diff | `0.000000008` |
| verified all | `true` |

## Artifacts

- `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-component-prefill-real-batch-parser-v1/results.jsonl`
- `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-component-prefill-real-batch-parser-v1/raw.json`
- `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-component-prefill-real-batch-parser-v1/stdout.log`
- `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-component-prefill-real-batch-parser-v1/stderr.log`
- `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-component-prefill-real-batch-parser-v1/memory.jsonl`

## Interpretation

The parser preserves:

- `metrics.prefill_total_input_tokens_per_second`
- `metrics.attention_pair_tps_mean`
- `workload.prompt_tokens_per_request`
- `workload.estimated_prefill_attention_work_tokens`
- `batching.mode`
- `batching.prefill_real_batch`
- `batching.prefill_executor_request_parallelism`
- `batching.prefill_executor_token_parallelism`
- sampled correctness fields

This closes a schema gap for real-batch component rows. The remaining T1 gap is connecting real
batch prefill/decode executors to the full package throughput runner.
