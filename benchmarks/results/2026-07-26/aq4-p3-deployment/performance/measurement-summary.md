# AQ4_0 P3 deployment performance measurement

Date: 2026-07-26

## Candidate identity and isolation

- Source commit: `c4c9a9b344fc10e9a77ab0ded3293469d21b2f72` (P3 endpoint)
- Worker: `ullm-aq4-worker`, SHA-256 `ba8c46d6eee81d508f4b2e744ec05d8743a46bf44100ec66257c8d8ae739e265`
- Product package manifest SHA-256: `a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad`
- Device: R9700 / `gfx1201` only (`HIP_VISIBLE_DEVICES=1`); V620 was not used.
- Service: `ullm-openai.service` was stopped for the direct timings.  No `llama-server`,
  `llama-bench`, or other timing worker held the GPU when the two measurements started.
- Model package, tokenizer, quantization package, resident execution profile, and all 36 P3
  required-kernel guards were those intended for the served candidate.  The HTTP gateway was
  intentionally not included in these kernel-path timings.

The product manifest is byte-identical to the active production product package manifest, so
the measurements do not substitute a model or quantization artifact.

## Results

| Path | Candidate result | Historical P3 reference | Difference | Assessment |
|---|---:|---:|---:|---|
| Prefill, 2,048 tokens / chunk width 128 | **970.6107 tok/s** | 982.3835 tok/s | -1.198% | Near reproduction; not exact numerical reproduction |
| Decode, cache C=1339, 32 measured steps | **73.4568 tok/s** | 74.29 tok/s | -1.122% | Near reproduction; not exact numerical reproduction |

The prefill run emitted:

```json
{"chunk_width":128,"elapsed_seconds":2.110011763,"tokens":2048,"tokens_per_second":970.6107027044095}
```

The decode profile emitted a mean step of `0.01361345265625` seconds and all 32 observed token
IDs were `4445`.  Its cache range was C=1339 through C=1371, after six warmup steps.

## Comparability and limits

The historical P3 readings used a materially different thermal/load condition: approximately
85°C junction temperature, maximum fixed core clock, and a 5.3 GB resident llama comparison
process.  This isolated deployment run began with edge/junction/memory temperatures of
50/51/50°C and completed at 59/65/60°C, with that comparison process absent.  Therefore the
1.20% and 1.12% deltas are reported rather than hidden, and the two datasets are not asserted
to be a strict environment-controlled A/B comparison.

The requested historic `56.6%` decode efficiency cannot be independently reproduced from a
tracked raw theoretical denominator.  Using the user-supplied rounded denominator of
131.2 tok/s gives a diagnostic ratio of `73.4568 / 131.2 = 55.99%`; this is not used as a gate
and remains **unconfirmed** as an efficiency measurement.

## Commands and guard contract

The detached worktree was built with:

```text
CARGO_BUILD_JOBS=16 \
CARGO_TARGET_DIR=/home/homelab1/coding-local/ultimateLLM/uLLM-aq4-p3-deployment-build-target-c4c9a9b3 \
cargo build --release -p ullm-engine --bin ullm-aq4-worker \
  --bin ullm-aq4-e2e-prefill-timing --bin ullm-aq4-decode-step-profile
```

Both direct binaries were invoked with the 36 required P3 kernel guard environment variables
from `release-inputs/candidate-manifest.json`, on the product package above.  The six P3 guards
new relative to the active 30-guard manifest were:

```text
ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_GROUP8_KERNEL
ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_KERNEL
ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_KERNEL
ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_RAGGED_M_KERNEL
ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_RAGGED_M_KERNEL
ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_WMMA_KERNEL
```
