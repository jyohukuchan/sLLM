# Lightweight promotion attempt 2 — activated

This is the successful generic lightweight-promotion run for the P3-only `AQ4_0` candidate.
The tool captured all ten fixed prompt-suite outputs from the old active worker, atomically
activated the candidate, waited through the gateway's bounded readiness retries, then captured
all ten candidate outputs.

## Result

- Status: `activated`
- Active candidate manifest SHA-256:
  `a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49`
- Saved rollback manifest SHA-256:
  `c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4`
- Generic service operation: one successful `systemctl restart`; no StartLimit recovery.
- Candidate readiness: ten bounded probes; the tenth observed health/ready/models all HTTP 200.
- Comparison: 10 cases, zero blocking findings, and nonempty generated text for Japanese,
  English, Python, JavaScript, long-summary, multi-turn, translation, and reasoning prompts.

`comparison.md` contains the complete side-by-side generated text.  The diagnostic exact-match
rate was 1.000 for this deterministic suite, but it was not used as an approval criterion: the
promotion decision is the absence of empty, garbled, repetitive, abandoned, or otherwise blocking
candidate responses in the saved real generations.

`rollback-preflight.json` records a read-only preflight after activation.  It confirms that the
old manifest bytes remain available and are strictly different from the active candidate.  No
rollback was executed.
