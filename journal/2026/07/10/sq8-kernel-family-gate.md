2026-07-10

- Purpose: make the SQ8_0 comparison kernel-family gate validate each reported projection boundary.
- Change: `--require-ullm-sq-kernel-families` now checks comma-separated `boundary=family` entries and rejects missing, empty, malformed, or `none` family values.
- Future family names are intentionally allowed, so non-direct/fused families can be reported without changing the gate.
- Verification: `python3 -m unittest tests.test_summarize_sq8_vllm_batch_grid` passed with 35 tests; the refreshed uLLM/vLLM normalized comparison still passes with `batch=direct` rows.
