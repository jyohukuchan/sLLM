# Lightweight recheck of rejected projection / format candidates

This result set applies `docs/plans/lightweight-promotion-policy-v0.1.md`:
actual generated text is the quality criterion; exact logits and top-1
agreement are diagnostic only.  The scope excludes attention work owned by
依頼BH.

| Candidate | R9700 full-model decode result | Lightweight text result | Disposition |
| --- | --- | --- | --- |
| `SQ8_0` private handwritten gfx1201 WMMA projection | **9.624875977 tok/s** versus matched CK **14.662647430 tok/s** (34.357857% slower) | Not run: speed-first stops a non-faster candidate. | do not promote |
| `SQ8_1` W8A8 | not measurable: no gfx1201 W8A8 dispatch, served selector, manifest, or worker | no-go: it says `NaN` is truthy / `Boolean(NaN)` is true, which is factually false | do not promote |
| `SQ9_0` | no current candidate | no current candidate | deferred future option |
| old handwritten `SQ8_0` attempt-2 | superseded revision | not rerun | not a current candidate |

## Evidence index

- [candidate inventory](candidate-inventory.md)
- [registered served-candidate validation](registered-candidate-validation.md)
- [`SQ8_1` W8A8 CPU text suite and human review](sq8_1-w8a8-cpu-fixed-suite/review.md)
- [`SQ8_1` speed disposition](sq8_1-w8a8-speed.md)
- [current active `AQ4_0` fixed-suite capture](active-aq4_0-fixed-suite/)
- [current active `AQ4_0` versus `SQ8_1` side-by-side output](active-aq4_0-vs-sq8_1-w8a8/)
- [complete recheck report](recheck-report.md)
- [first WMMA attempt status](sq8_0-handwritten-wmma-speed-first/attempt-status.md)
- [foreign R9700 contention record](r9700-contention-before-retry.md)

The active manifest is never edited in this result set.  Any eligible future
promotion must use the generic promotion / rollback tools, not an ad-hoc
candidate path.
