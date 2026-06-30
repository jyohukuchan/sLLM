# aq quantization research

## Work

- Downloaded and verified `Qwen/Qwen3.5-9B` BF16 safetensors.
- Downloaded and verified `unsloth/Qwen3.5-9B-GGUF` `Qwen3.5-9B-UD-Q4_K_XL.gguf`.
- Reused existing local `AxionML-Qwen3.5-9B-NVFP4` safetensors.
- Added NVFP4 tensor error comparison tooling.
- Added GGUF tensor error comparison tooling via llama.cpp `gguf-py`.
- Extended aq tensor sampler:
  - explicit CPU thread recording,
  - Qwen3.5 `linear_attn` family labels,
  - family-balanced tensor selection,
  - Lloyd-refined codebook candidates,
  - group-size sweep candidates.
- Added CPU full-quantizer design:
  - `docs/plans/aq-full-quantizer-design-v0.1.md`

## Results

- NVFP4 baseline:
  - result: `benchmarks/results/2026-07-01/aq/2026-07-01-nvfp4-error-qwen35-9b.jsonl`
  - mean relative MSE: `0.008996`
  - mean cosine similarity: `0.995502`

- Unsloth Dynamic Q4_K_XL reliable subset:
  - result: `benchmarks/results/2026-07-01/aq/2026-07-01-udq4kxl-error-qwen35-9b-reordered.jsonl`
  - mean relative MSE: `0.002857`
  - mean cosine similarity: `0.998570`
  - Qwen3.5 linear-attention comparison now applies llama.cpp's V-head reorder to the HF reference. The old unreordered run remains as an invalid comparison artifact.

- aq round2 best:
  - result: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-round2-qwen35-9b-balanced.jsonl`
  - best candidate: `aq4_e4m3_g16_ts_flloyd16`
  - mean relative MSE: `0.005255`
  - scale-window 16 rerun: `0.005235`, only a small improvement.

- aq group-size sweep:
  - result: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-round3-qwen35-9b-group-sizes.jsonl`
  - g64 / 4.125 bpp: `0.008292`
  - g32 / 4.25 bpp: `0.006873`
  - g16 / 4.50 bpp: `0.005244`
  - g8 / 5.00 bpp: `0.003573`

- aq family-level LUT:
  - result: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-lut-qwen35-9b-balanced.jsonl`
  - g16 / 4.50 bpp free Lloyd16: `0.005241`
  - sample-local g16 result was `0.005244`; no meaningful penalty was observed with 3 tensors per family.
  - wide result: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-lut-qwen35-9b-wide.jsonl`
  - with up to 8 tensors per family, g16 / 4.50 bpp free Lloyd16 was `0.005268`, still close to the sample-local value.

- CPU full quantizer design:
  - Rust orchestration plus C++20 numeric kernels.
  - One explicit compute thread pool; avoid Rayon/OpenMP/thread-pool oversubscription.
  - Chunked tensor processing to avoid whole-model RAM use.
  - First implementation target: `aq4_e4m3_g16_ts_flloyd16`, plus `g8` as a 5.0 bpp accuracy point.

- `ullm-quant` skeleton:
  - added a Rust workspace with `crates/ullm-quant`.
  - added a C++20 CPU kernel stub behind a C ABI.
  - smoke kernel packs 4-bit nibbles and is called from Rust.
  - `cargo test -p ullm-quant` passed.
  - `cargo run -p ullm-quant -- --dry-run --threads 64 --io-threads 2 --max-working-memory-mib 4096` printed `pack_smoke=ok [16, 33, 15, 120]`.

- Firecrawl literature survey:
  - created `docs/research/quantization-method-survey-2026-07-01.md`.
  - checked GPTQ, SmoothQuant, AWQ, OmniQuant, AQLM, QuIP#, QuaRot, and FP4/MXFP4 papers through arXiv pages.
  - Firecrawl search returned empty results for exact paper-title queries; direct arXiv scraping worked.
  - main conclusion: tensor MSE is not enough; activation-weighted error and logit/perplexity smoke tests are needed before treating an aq row as a format candidate.

- Activation-aware plan:
  - created `docs/plans/aq-activation-aware-validation-v0.1.md`.
  - next tool targets are `tools/collect-activation-stats.py` and `tools/run-aq-weighted-sample.py`.
  - activation stats should store only streaming reductions such as per-input-channel second moments, not raw activations.

## Current Interpretation

Concrete measurement should continue in parallel with quantizer optimization. A separate long theory-only phase is not useful now, but full-model conversion will require a dedicated CPU-multithreaded quantizer implementation.

The current aq result is promising at 4.5 bpp: it beats sampled NVFP4 and slightly beats sampled UD `Q4_K` rows. The family-level LUT result remained close even at up to 8 tensors per family, so the next uncertainty is not obvious LUT instability. The larger risk is activation sensitivity and model-level behavior.

## Next

- Start activation-weighted and logit-level checks for the current top aq candidates.
- Extend `ullm-quant` from skeleton to safetensors metadata planning.
- Add a small model-level check after tensor-level candidate narrowing.
