# SQ8 worker acceptance v0.2

This directory records the successful standalone P8-C acceptance run on the
physical Radeon AI PRO R9700. The raw stream is the source of truth; the
independent validator reconstructed every gate from it without using the
producer's in-memory decisions.

## Identity

- Git commit: `4e627bc537ce493cbe6a7387144229331d943b03`
- Tracked worktree at start and finish: clean
- Release worker SHA-256: `145a5351db3957130200276314853e394d0fd206a69e2eab260c01141411b950`
- Contract: `docs/specs/sq8-worker-acceptance-v0.2.md`
- Raw schema: `ullm.sq8.worker_acceptance.raw.v2`
- Validation schema: `ullm.sq8.worker_acceptance.validation.v2`

## Result

- Independent validation: pass, with no gate errors
- Cancellation: 2 warmups and 10 measured samples
- Measured cancellation upper-bound p50: `88,639,018.5 ns`
- Measured cancellation upper-bound p95: `145,297,962.95 ns`
- Maximum across all 34 cancellations: `298,216,883 ns`
- Resource schedule: 100 requests and 505 post-release samples
- Final R9700 VRAM delta: `0 bytes`
- Final worker RSS delta: `0 bytes`
- Theil-Sen VRAM slope: `0 bytes/request`
- Theil-Sen worker RSS slope: `0 bytes/request`
- KFD stable snapshots: 641, with zero retries
- Worker exit: code 0 in `62,034,509 ns`

The preceding v0.1 attempt stopped fail-closed at resource request 87 after a
short-lived non-worker KFD PID directory disappeared during observation. It was
kept only as incomplete external diagnostic evidence and is not part of this
successful bundle.

## Validation

From the repository root:

```bash
tools/validate-sq8-worker-acceptance.py \
  benchmarks/results/2026-07-11/sq8-worker-acceptance-v0.2/raw.jsonl \
  --expected-git-commit 4e627bc537ce493cbe6a7387144229331d943b03 \
  --expected-binary-sha256 145a5351db3957130200276314853e394d0fd206a69e2eab260c01141411b950
```

Verify the copied artifacts with `sha256sum -c SHA256SUMS`.
