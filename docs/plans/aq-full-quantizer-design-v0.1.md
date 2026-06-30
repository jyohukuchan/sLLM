# aq full quantizer design v0.1

## Purpose

The Python aq sampler is only for tensor-level search. Full model conversion needs a dedicated CPU-multithreaded quantizer because quantization time is dominated by scanning, scale search, codebook assignment, nibble packing, and output writes across billions of weights.

This design targets the first production-style aq converter for BF16/FP16 safetensors input.

## Initial Requirements

- Input:
  - HF safetensors model directory.
  - `model.safetensors.index.json` must be supported.
  - BF16 and FP16 tensors must be supported first.

- Output:
  - Temporary multi-file directory output is acceptable until final `.ullm` container structure is fixed.
  - Output must preserve enough metadata to reconstruct:
    - tensor names and shapes,
    - family names,
    - index bit width,
    - packed indices,
    - scale format and group size,
    - tensor/family scale,
    - LUT/codebook values and granularity.

- Threading:
  - Thread count must be explicit and recorded.
  - Default on WRX80 should start at 64 worker threads, not 128 SMT threads.
  - There must be one main compute thread pool to avoid oversubscription.
  - I/O thread count must be separate and small.

- Memory:
  - Never materialize the whole model in RAM.
  - Avoid materializing full dequantized tensors.
  - Process tensors in row/block-aligned chunks.
  - Bound peak RSS with a user-facing `--max-working-memory` option.

## Recommended Implementation Split

Use Rust for orchestration and C++20 for numeric kernels.

Rust:

- CLI and config parsing.
- safetensors metadata planning.
- task scheduling and progress reporting.
- output manifest writing.
- per-run log and metrics.

C++20:

- BF16/FP16 to FP32 chunk decode.
- group amax calculation.
- codebook assignment.
- scale selection.
- 4-bit index packing.
- per-thread reductions for codebook optimization.

Rust can call C++ kernels through a narrow C ABI. Do not expose model-format logic across the FFI boundary; pass flat typed buffers and simple structs.

## Pipeline

### Phase 0: Plan

Read only metadata first:

1. Parse safetensors index.
2. Build `TensorPlan` records:
   - source file,
   - tensor name,
   - dtype,
   - shape,
   - family,
   - target candidate.
3. Reject unsupported tensors early or mark pass-through.
4. Estimate output size and peak working memory.

Initial implementation status:

- `crates/ullm-quant` can read safetensors headers without loading tensor payloads.
- Plan output paths:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-plan-qwen35-9b-all-g16.json`
  - `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-plan-qwen35-9b-p4p6.json`
  - `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-plan-qwen35-9b-p4p9.json`
  - `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-plan-qwen35-9b-all-g8.json`
- Qwen3.5-9B plan result:
  - total tensors: `775`
  - default quantize tensors: `255`
  - passthrough tensors: `520`
  - total tensor bytes: `19306216416`
- The planner currently quantizes known text linear families only and
  passes through embeddings, lm head, vision tensors, convolution tensors, MTP
  tensors, and unknown families.
- Plan schema `ullm-quant-plan-v0.3` records aq policy assignment, estimated
  output bytes, and estimated effective bpp.
- Current best policy candidate is `p4p6`:
  - high format `aq4_e4m3_g8_ts_flloyd16` for `attn_k`, `attn_o`, `attn_v`,
    and `linear_attn_out`,
  - low format `aq4_e4m3_g16_ts_flloyd16` for other quantizable families.
- Policy size summary:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-policy-size-summary-qwen35-9b.json`
  - all-g16 estimated output bytes: `9059400672`
  - p4p6 estimated output bytes: `9098722272`
  - p4p9 estimated output bytes: `9325214688`
  - all-g8 estimated output bytes: `9504914400`
- `p4p6` is only `39321600` bytes above all-g16 in the current payload estimate
  and performed best in the 10-module logit smoke.

### Phase 1: Calibration

Goal: build candidate codebooks and optional family/tensor scales.

For each selected tensor:

1. Stream deterministic sample chunks.
2. Convert BF16/FP16 chunks to FP32.
3. Split into contiguous groups.
4. Compute group amax.
5. Normalize values by group amax.
6. Accumulate per-family sample buffers or histograms.

For Lloyd-style codebooks:

1. Initialize centers from quantiles or histogram percentiles.
2. Run fixed iterations.
3. Use per-thread accumulators:
   - count per center,
   - sum per center,
   - optional squared error per center.
4. Merge per-thread accumulators deterministically.

Avoid pushing all sampled values into one giant vector for full conversion. The Python sampler can do that, but the production path should use bounded reservoir samples or histograms.

### Phase 2: Quantization

For each tensor:

1. Open source safetensor.
2. Iterate row/block-aligned chunks.
3. Decode BF16/FP16 to FP32.
4. For each group:
   - compute amax,
   - choose tensor/family scale,
   - choose nearest representable group scale,
   - assign each value to nearest codebook entry,
   - optionally scan nearby scale values for lower group error.
5. Pack two 4-bit indices into one byte.
6. Write packed indices and scales to output.
7. Record per-tensor metrics:
   - relative MSE sample,
   - max abs error sample,
   - saturation rate,
   - scale min/max hit rate,
   - zero preservation rate if applicable.

### Phase 3: Verification

After writing:

1. Re-read selected output chunks.
2. Dequantize them with the output metadata.
3. Compare against source chunks.
4. Fail the run if measured metrics diverge from in-flight metrics.

## Parallel Work Units

Use tensor chunks as the outer work unit.

Recommended initial chunking:

- Large 2D tensors:
  - split by rows,
  - keep row slices aligned to scale group layout,
  - target 64-256 MiB decoded FP32 working set per worker batch only if memory allows.
- Small tensors:
  - process whole tensor as one task,
  - avoid scheduling overhead by bundling small tensors.

Per worker scratch:

- FP32 decode buffer.
- group amax buffer.
- scale index buffer.
- packed index buffer.
- local metric accumulators.

Do not allocate scratch buffers inside the innermost loop.

## SIMD Strategy

Initial CPU target assumes AVX-512 and INT8 support.

Prioritize kernels in this order:

1. BF16 to FP32 decode.
2. group amax.
3. nearest codebook assignment for 16-entry LUT.
4. scale search over nearby E4M3/E5M2 values.
5. 4-bit packing.

The 16-entry codebook assignment should use a layout that broadcasts codebook entries and processes multiple values per vector. For group size 16, one group naturally maps well to AVX-512 lanes.

## Threading Rules

- `--threads`: compute workers.
- `--io-threads`: read/write helpers.
- `--calibration-threads`: optional override, default same as `--threads`.
- `--pin-threads`: later option for NUMA experiments.

Rules:

- Never combine Rayon parallelism, OpenMP, and a library thread pool without explicitly limiting all but one.
- If using Rust Rayon for scheduling, C++ kernels should be single-call vector kernels, not OpenMP regions.
- If using a C++ thread pool, Rust should submit coarse tasks and wait.
- Log all thread settings in the output manifest.

## First Candidate To Implement

Implement the `p4p6` policy first, using this low-budget candidate for most
quantizable families:

```text
aq4_e4m3_g16_ts_flloyd16
```

Reason:

- It is the current best 4.5 bpp tensor-level candidate.
- It slightly beats sampled UD `Q4_K` rows and clearly beats sampled NVFP4 rows.
- Family-level LUT did not show a penalty in the first 3-tensor/family check.
- It maps naturally to 16-value groups.

And this high-budget candidate for `attn_k`, `attn_o`, `attn_v`, and
`linear_attn_out`:

```text
aq4_e4m3_g8_ts_flloyd16
```

Reason:

- It gives a 5.0 bpp accuracy point.
- It improved the current mixed-policy smoke when used selectively.

Rust implementation status:

- Chunked safetensors payload reader exists.
- `--inspect-tensor` can read real tensor payloads in bounded chunks and compute
  FNV-1a64 checksums.
- BF16/F32 numeric stats can be computed during chunk inspection.
- `--inspect-aq-format` can compute group count, group absmax stats, and direct
  scale-index dry-run metrics for aq formats.
- Real Qwen3.5 inspection outputs:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-inspect-qwen35-9b-layer0-mlp-up.txt`
  - `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-inspect-qwen35-9b-layer3-attn-k-g8.txt`
- Fixed sampled codebook export for Rust dry-runs:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-codebooks-qwen35-9b-mlp-up-attn-k-weighted.json`
  - exported families: `mlp_up`, `attn_k`
  - exported candidates: `aq4_e4m3_g16_ts_flloyd16`, `aq4_e4m3_g8_ts_flloyd16`
  - source tool: `tools/export-aq-family-codebooks.py`
- `--codebook-json` plus `--inspect-codebook-family` and
  `--inspect-codebook-candidate` can load exported 16-entry family LUTs.
- One-tensor quantization dry-run exists for streamed real payloads:
  - `model.language_model.layers.0.mlp.up_proj.weight` with
    `aq4_e4m3_g16_ts_flloyd16`: relative MSE `0.006231116836`.
  - `model.language_model.layers.3.self_attn.k_proj.weight` with
    `aq4_e4m3_g8_ts_flloyd16`: relative MSE `0.004610619768`.
- The dry-run now supports `_ts_` tensor-scale estimation and
  `--scale-window N` per-group scale search.
- Scale-window 4 results:
  - `mlp_up` g16 relative MSE improved to `0.005283509762` with tensor scale
    `0.014789051376`.
  - `attn_k` g8 relative MSE improved to `0.003677692937` with tensor scale
    `0.018260609359`.
- Python reference tool:
  - `tools/verify-aq-one-tensor.py`
  - validates the Rust chunk path against a chunked PyTorch implementation
    using the same exported family codebook and scale-window setting.
- Prototype one-tensor output exists via `--prototype-output-dir`:
  - writes `manifest.json`, `idx4` packed indices, `scale_u8` scale indices,
    and an F32 little-endian codebook file,
  - re-reads the prototype files and dequantizes against the source safetensors
    payload,
  - fails if re-read relative MSE differs from the in-flight metric by more
    than `1e-9`.
- First real prototype output:
  - `benchmarks/results/2026-07-01/aq/prototype-qwen35-9b-layer3-attn-k-g8-scale-window4.ullm.d/`
  - tensor: `model.language_model.layers.3.self_attn.k_proj.weight`
  - relative MSE and re-read relative MSE: `0.003677692937`
  - idx4 bytes: `2097152`
  - scale bytes: `524288`
  - wall time: `1.71 s`
  - peak RSS: `8232 KiB`
- `--skip-inspect` and `--prototype-skip-verify` allow write-only benchmark
  runs without duplicate inspection or re-read verification.
- Larger scalar Rust prototype benchmark:
  - tensor: `model.language_model.layers.0.mlp.up_proj.weight`
  - candidate: `aq4_e4m3_g16_ts_flloyd16`
  - relative MSE: `0.005283509762`
  - idx4 bytes: `25165824`
  - scale bytes: `3145728`
  - wall time: `8.76 s`
  - peak RSS: `21560 KiB`
  - throughput: about `5.75M` elements/s
  - note: source tensor is read twice because tensor-scale estimation is still
    a pre-pass.
- First C++20 chunk kernel:
  - ABI entry: `ullm_aq_quantize_chunk_v1`
  - request struct carries `struct_size`, explicit dtype id, pointers, buffer
    sizes, group size, scale table, codebook, tensor scale, and scale window.
  - owns best-scale search, nearest-codebook assignment, idx4 packing,
    scale-index output, and chunk metric accumulation.
  - Rust still owns safetensors I/O, tensor-scale estimation, manifest writing,
    and prototype re-read/dequant verification.
  - currently supports BF16 only; other dtype ids return unsupported status.
  - requires a 16-entry codebook, `scale_count <= 256`, group-aligned input
    chunks, and preallocated output buffers.
- C++ scalar baseline benchmark:
  - `mlp_up` g16 write-only: `7.13 s`, `21516 KiB`, about `7.06M`
    elements/s, same relative MSE `0.005283509762`.
  - scalar Rust write-only baseline was `8.76 s`, so the C++ chunk path is
    about `1.23x` faster before SIMD/multithreading.
  - `attn_k` g8 write + re-read verification succeeded with relative MSE
    `0.003677692937`.

## Output Directory Prototype

Until `.ullm` single-file layout is fixed, use:

```text
model.ullm.d/
  manifest.json
  tensors/
    <tensor-id>.idx4
    <tensor-id>.scale
  codebooks/
    <family-or-tensor-id>.bf16
  metrics/
    quantization.jsonl
```

This is not the final container format. It is a low-risk converter target for validation.

## Metrics To Record

Per run:

- source model path,
- source revision if known,
- quantizer git commit,
- candidate id,
- thread settings,
- wall time,
- peak RSS,
- input bytes,
- output bytes,
- effective bpp.

Per tensor:

- tensor name,
- shape,
- family,
- input dtype,
- output index bytes,
- output scale bytes,
- relative MSE sample,
- cosine similarity sample,
- max abs error sample,
- saturation rate,
- quantization seconds,
- write seconds.

## Tests

Unit tests:

- BF16 decode matches known values.
- E4M3 scale table matches PyTorch float8 conversion for finite values.
- 4-bit packing/unpacking round trips.
- all-zero group quantizes without NaN.
- group sizes 8, 16, 32, 64 work.
- tensors whose element count is not divisible by group size are either padded with explicit metadata or rejected.

Integration tests:

- Small synthetic safetensors model.
- One real Qwen3.5 tensor.
- Compare C++ quantizer output against Python sampler for the same candidate and seed.
- Verify deterministic output across repeated runs with the same thread count.

Performance tests:

- elements/s per tensor family.
- GB/s input decode.
- scale-search ns/group.
- RSS under multiple chunk sizes.
- scaling from 1, 8, 32, 64, and 128 threads.

## Immediate Steps

1. Add F16 support to `quantize_chunk_v1` after BF16 semantics are stable.
2. Add larger golden tests that compare C++ chunk output against Python or Rust
   scalar output across multiple random seeds.
3. Avoid the tensor-scale pre-pass where possible by either storing group amax
   summaries or fusing estimation with a bounded histogram.
4. Add SIMD kernels after the scalar C++ semantics are locked.
5. Extend from one tensor to all tensors selected by the p4p6 plan.
6. Run a full Qwen3.5-9B conversion once RSS, throughput, and one-tensor
   reconstruction metrics are acceptable.
