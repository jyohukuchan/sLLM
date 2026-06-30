# Start aq validation

## Done

- Added `docs/plans/aq-validation-plan-v0.1.md`.
- Added `docs/specs/aq-experiment-result-v0.1.md`.
- Added `tools/run-aq-tensor-sample.py` for first-pass tensor sampling.
- Reframed the first `aq` step as validation of candidate axes, not as a stable format decision.
- Confirmed reference points with Firecrawl:
  - OCP MX specification landing page for MX-style E8M0/block scaling.
  - NVIDIA NVFP4 blog for 16-value micro-block, E4M3 scale, and tensor-level scale.

## Notes

- Initial `aq` payload assumption is 4bit indices plus 16-entry codebook.
- Scale candidates are E8M0, E5M2, unsigned E5M3, and E4M3.
- Initial group sizes are 16 and 32, with 64 as a low-overhead candidate.
- Initial codebook modes are `zero_free15` and `symmetric7`; `free16` is kept for upper-bound sampling.
- The first objective is BF16 weight reconstruction error. Activation-aware weighting and perplexity come after tensor-level narrowing.
- The current helper initializes codebooks per tensor sample. Family-level codebook aggregation is still a follow-up task.

## Next

- Add family-level codebook aggregation to the Python sampler.
- Run Round 1 on sampled Qwen3-14B tensor families.
