| Status | Engine | Model | Family | Quant | SQ mode | Impl | Target | Workload | Batching | Prefill total tok/s | Decode total tok/s | End-to-end tok/s | Consumed GiB | Decode x GiB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ok | uLLM | Qwen3.5-9B | FP8 | SQ8_0 | direct_fp8_dequant_matvec | single=sq8_0_matvec_r9700_direct,triple=sq8_0_matvec_triple_r9700_direct | R9700 | pp2/tg1/b2 | grouped | 63.61 | 150.41 | 17.88 | 0.92 | 137.67 | `results.jsonl` |
