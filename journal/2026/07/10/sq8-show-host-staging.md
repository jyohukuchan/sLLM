2026-07-10

- Purpose: make the SQ8_0/vLLM comparison table show host-staging residue directly.
- Change: `--show-sq-details` now adds `SQ staging ops` and `SQ staging MiB`.
- Current refreshed Qwen3-14B-FP8 rows show `0/72`, `0/120`, and `0/216` staging ops for b2/b4/b8.
- Verification: unit tests, py_compile, diff check, and the refreshed normalized comparison command passed.
