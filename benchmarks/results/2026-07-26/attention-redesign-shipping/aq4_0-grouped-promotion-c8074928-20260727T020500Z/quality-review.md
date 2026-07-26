# `AQ4_0` grouped candidate text-quality review

This is a same-model comparison: active `AQ4_0` P3 versus the
`AQ4_0` 4:1×256 grouped-decode candidate.  It is not related to the separate
`SQ8_0` Qwen3-14B quality review.

`tools/promote-served-model.py --yes` captured the fixed ten-prompt suite from
the active runtime, atomically switched to candidate manifest
`69a5e1eb2e7713a1d017332539a587b9a13cf925cbfb28d7c89719ba6709ec2e`,
restarted the service once, and captured the same suite again.

- All 10 active and candidate requests completed successfully.
- The candidate had no request failure, empty completion, repetition,
  garbling, extreme-length, code-structure, or language-abandonment finding.
- The candidate response text exactly matched the active response for all 10
  cases.  The 1.000 exact-match rate is retained as a diagnostic observation,
  not a promotion threshold.
- `comparison.md` contains the actual prompt/output pairs.  Reading the
  code and Japanese multi-turn examples found no new candidate degradation;
  they are byte-for-byte the active output.

The promotion outcome is `activated`; the active manifest is the candidate
manifest above.  The exact rollback P3 bytes and one-command rollback metadata
are retained in the root-owned promotion state transaction referenced by
`outcome.json`.  No rollback was run after a successful activation.
