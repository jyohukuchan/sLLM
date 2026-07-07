# Package Prompt Guard Bundle

- Summary JSON: `benchmarks/results/2026-07-07/engine/prompt-suite-qwen35-qk-rope-kvwrite-fused-r9700-v620-compare-atol1e-3/guard-bundle-summary.json`
- Reference: `qk-rope-kvwrite-fused-r9700`
- Candidate: `qk-rope-kvwrite-fused-v620`
- Passed: `true`

| check | passed | artifact | key metrics |
| --- | :---: | --- | --- |
| prompt_suite_token_logits | true | `benchmarks/results/2026-07-07/engine/prompt-suite-qwen35-qk-rope-kvwrite-fused-r9700-v620-compare-atol1e-3/prompt-suite-token-logits-guard.json` | compared_case_count=7, generated_token_match_count=7, top_logits_match_count=7, max_prefill_top_logit_abs_diff=2.86102294921875e-06, max_decode_last_top_logit_abs_diff=8.869171142578125e-05 |
