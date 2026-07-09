# Package Prompt Guard Bundle

- Summary JSON: `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/qwen3-14b-sq8-prompt-suite-smoke-rope128-theta1e6/guard-self-behavioral/guard-bundle-summary.json`
- Reference: `qwen3-14b-sq8-smoke-self-reference`
- Candidate: `qwen3-14b-sq8-smoke`
- Passed: `true`

| check | passed | artifact | key metrics |
| --- | :---: | --- | --- |
| prompt_suite_token_logits | true | `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/qwen3-14b-sq8-prompt-suite-smoke-rope128-theta1e6/guard-self-behavioral/prompt-suite-token-logits-guard.json` | exit_code=0, acceptance_mode=behavioral, strict_passed=True, behavioral_passed=True, compared_case_count=1, generated_token_match_count=1, generated_text_match_count=1, generated_without_stop_text_match_count=1, top_logits_match_count=1, max_prefill_top_logit_abs_diff=0.0, max_decode_last_top_logit_abs_diff=0.0 |
