# CPU strict-F32 v0.2 reference capture

This directory is the resumable, CPU-only capture root for the frozen SQ8 v0.2
artifact-F32 control.  Its static launcher plan binds the frozen gate SHA-256
`64a43c032570bed8086e3c441b0774cc470c5ab1e8c67f99e02af2b6307f72bf`, the
artifact/package paths, the worker executable hash, fixed seed `0`, 8 threads
per process, 8 processes, physical-core affinity, and GPU-invisible environment
variables.  This README is not a completion indicator; use `launcher-progress.json`
and each case's `run.json` for the live/final state.

The launcher has one process own one causal case and only parallelizes the 17
independent cases.  Every worker is pinned to a disjoint set of physical
Threadripper cores and uses the CPU strict-F32 reference; it invokes no GPU,
runtime context, or service control.

## Coverage layout

- `cases/sequential_m1/` contains the seven primary streams and five mandatory
  boundary cases.  Their prompt-plus-forced-decode forward count is 16,437.
- `cases/m128_chunks_with_declared_tail/` contains the five required M=128
  inputs and has 12,416 forwards.  It records every complete 128-token boundary
  and each final tail in `m128-checkpoints.json`.
- Together the frozen schedule has 28,853 forwards.  The seven primary streams
  contribute 4,096 forced decode positions in total; prompt forwards are kept
  separately rather than treated as decode samples.

Each published forward directory contains `logits.f32le`, `final-hidden.f32le`,
all 40 `layers/layer-XX-hidden.f32le` payloads, and `metadata.json` carrying the
token IDs, geometry, finiteness metadata, and content SHA-256 values.  Payloads
are first written under `.staging` and atomically renamed into the final
position directory.  `progress.json` is atomically updated after each position.
On a completed case, `SHA256SUMS` covers the run plan, progress/receipt files,
teacher-forced token stream, checkpoint receipt where applicable, and every
published payload/metadata file; `run.json` records that manifest's SHA-256.

## Interruption and resume

Run the same launcher invocation with `--resume`.  It refuses a changed static
launcher plan, including changed worker binary or frozen-gate hash.  A resumed
worker verifies each already-published position's metadata/content hashes,
replays it only to reconstruct causal F32 KV state, and does not rewrite its
capture.  A missing or mismatched position fails closed rather than being
silently accepted.  Thus resume avoids output recapture while preserving the
causal state required to continue after the last checkpoint.

The serial-versus-parallel 8-thread verification is recorded in
[`parallel-vs-serial-t8-v1.json`](parallel-vs-serial-t8-v1.json): four causal
positions and all 168 tensor payloads were byte-identical with eight workers
running concurrently.
