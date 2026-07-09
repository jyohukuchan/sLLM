# Package Prompt Guard Bundle

- Summary JSON: `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-prompt-suite-text-guard-smoke-v0.1/guard/guard-bundle-summary.json`
- Reference: `aq4`
- Candidate: `sq-layer23-k16`
- Passed: `false`

| check | passed | artifact | key metrics |
| --- | :---: | --- | --- |
| prompt_suite_token_logits | false | `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-prompt-suite-text-guard-smoke-v0.1/guard/prompt-suite-token-logits-guard.json` | exit_code=1, acceptance_mode=strict, strict_passed=False, behavioral_passed=True, compared_case_count=1, generated_token_match_count=1, generated_text_match_count=1, generated_without_stop_text_match_count=1, top_logits_match_count=0, max_prefill_top_logit_abs_diff=0.04891014099121094, max_decode_last_top_logit_abs_diff=0.1147451400756836 |
