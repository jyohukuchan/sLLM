# Package Prompt Guard Bundle

- Summary JSON: `benchmarks/results/2026-07-06/engine/prompt-suite-aq4-pagedattn-r9700-v620-v0.3-guard-bundle/guard-bundle-summary.json`
- Reference: `R9700-RDNA4`
- Candidate: `V620-RDNA2`
- Passed: `true`

| check | passed | artifact | key metrics |
| --- | :---: | --- | --- |
| prompt_suite_token_logits | true | `benchmarks/results/2026-07-06/engine/prompt-suite-aq4-pagedattn-r9700-v620-v0.3-guard-bundle/prompt-suite-token-logits-guard.json` | compared_case_count=7, generated_token_match_count=7, top_logits_match_count=7, max_prefill_top_logit_abs_diff=0.0, max_decode_last_top_logit_abs_diff=0.0 |
| standalone_logits | true | `benchmarks/results/2026-07-06/engine/prompt-suite-aq4-pagedattn-r9700-v620-v0.3-guard-bundle/standalone-logits-guard.json` | prompt_tokens=25, top_count=8, top_token_ids_match=True, max_abs_logit_diff=0.0 |
