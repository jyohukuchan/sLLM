# Benchmark Results

Benchmark result files are stored as JSON Lines.

The current schema is documented in:

```text
docs/specs/inference-benchmark-result-v0.1.md
```

Suggested layout:

```text
benchmarks/results/YYYY-MM-DD/<engine>/<run_id>.jsonl
benchmarks/results/YYYY-MM-DD/<engine>/logs/
```

Large raw logs should not be committed unless they are small enough to be useful as reproducible setup records.
