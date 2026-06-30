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

- Activation-aware tooling:
  - added `tools/collect-activation-stats.py`.
  - added `tools/run-aq-weighted-sample.py`.
  - extended `tools/run-aq-tensor-sample.py` with `--activation-stats`, `weighted_mse`, and `weighted_relative_mse`.
  - `python3 -m py_compile tools/run-aq-tensor-sample.py tools/run-aq-weighted-sample.py tools/collect-activation-stats.py` passed.
  - one-tensor smoke with unit activation weights on `model.language_model.layers.14.mlp.down_proj.weight` passed; weighted relative MSE was `0.005159932654350996`.
  - default Python has `torch 2.12.0+cpu` and no CUDA/ROCm-visible device.
  - real CPU activation-stat smoke succeeded for `language_model.layers.0.mlp.down_proj`, with 1 prompt and 15 tokens.
  - result: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-weighted-smoke-qwen35-9b.jsonl`
  - weighted smoke metrics for `aq4_e4m3_g16_ts_flloyd16`: unweighted relative MSE `0.0051582370266549235`, weighted relative MSE `0.004603734239935875`.
  - full activation collection should use an R9700-capable environment; small CPU smoke runs are feasible.

- R9700 activation-weighted comparison:
  - collected R9700 stats with `build/envs/vllm-rocm-nightly`, `ROCR_VISIBLE_DEVICES=1`, 4 prompts, 1403 tokens, 152 modules.
  - aq g16 unweighted scale search: mean relative MSE `0.005269024`, weighted relative MSE `0.008698592`.
  - aq g16 weighted scale search: mean relative MSE `0.005972846`, weighted relative MSE `0.004922713`.
  - aq g8 weighted scale search: mean relative MSE `0.004234023`, weighted relative MSE `0.003684397`.
  - ModelOpt NVFP4 family4 weighted relative MSE: `0.010255294`.
  - Unsloth Dynamic Q4_K_XL mixed family4 weighted relative MSE: `0.002460200` at mean `5.4668` bpp; it stores `linear_attn_out` as `Q8_0`.
  - interpretation: weighted scale search is a real aq optimization lever; Unsloth's result points toward family-specific bpp policy.

- R9700 calib32 stability check:
  - added `benchmarks/calibration/qwen35-aq-smoke-prompts-v0.1.txt`.
  - collected 32 prompts / 14061 tokens / 152 modules in `benchmarks/results/2026-07-01/aq/activation-r9700-calib32-qwen35-9b-s512/`.
  - aq g16 weighted scale search weighted relative MSE: `0.004622421`.
  - aq g8 weighted scale search weighted relative MSE: `0.003439578`.
  - ModelOpt NVFP4 weighted relative MSE: `0.009864150`.
  - Unsloth Dynamic Q4_K_XL mixed weighted relative MSE: `0.002471176`.
  - direction remained stable versus the 4-prompt smoke.

- Weighted codebook and family policy:
  - added activation-weighted Lloyd support through `--weighted-codebook`.
  - aq g16 weighted scale + codebook weighted relative MSE: `0.004038034`.
  - aq g8 weighted scale + codebook weighted relative MSE: `0.002821072`.
  - combined param-weighted result: aq all-g16 `0.003798456` at 4.5 bpp; aq all-g8 `0.002582475` at 5.0 bpp.
  - sampled UD Q4_K_XL mixed combined result: `0.002364278` at parameter-weighted bpp `5.206019`.
  - simple family policy with g8 on `attn_k,attn_o,attn_v,linear_attn_out` gave combined weighted relative MSE `0.003053866` at bpp `4.592593`.
  - policy artifact: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-policy-r9700-calib32-qwen35-9b.json`.
  - next evidence should be model-level logit/perplexity, not only tensor metrics.

- Module-level logit smoke:
  - added `tools/run-aq-module-logit-smoke.py`.
  - quantized only `model.layers.0.linear_attn.out_proj` and compared final-token logits against BF16.
  - result: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-linear-attn-out-r9700-calib32-qwen35-9b.jsonl`.
  - g16 unweighted logit relative MSE: `0.002045509`.
  - g16 weighted scale + codebook logit relative MSE: `0.000198949`.
  - g8 weighted scale + codebook logit relative MSE: `0.000101244`.
  - all three preserved top1 and top10 on the single prompt.
  - 8-prompt follow-up result: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-linear-attn-out-r9700-calib32-qwen35-9b-prompts8.jsonl`.
  - 8-prompt mean logit relative MSE: g16 unweighted `0.002274514`, g16 weighted `0.000214926`, g8 weighted `0.000253724`.
  - 8-prompt mean KL: g16 unweighted `0.005510745`, g16 weighted `0.000705097`, g8 weighted `0.000899909`.
  - note: g16 weighted ranked slightly better than g8 weighted on the logit smoke despite g8's better tensor weighted MSE.
  - extra module smoke result: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-extra-modules-r9700-calib32-qwen35-9b-prompts8.jsonl`.
  - `model.layers.0.mlp.up_proj`: g8 weighted was best by mean relative MSE `0.000162605` and KL `0.000821250`.
  - `model.layers.3.self_attn.v_proj`: g8 weighted was best by mean relative MSE `0.000222151`, but g16 unweighted had lower KL `0.001237670` than weighted variants.
  - interpretation: weighted variants are promising, but per-family/per-metric behavior differs.
  - cumulative3 result: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-cumulative3-r9700-calib32-qwen35-9b-prompts8.jsonl`.
  - cumulative3 mean logit relative MSE: g16 unweighted `0.002544046`, g16 weighted `0.000297915`, g8 weighted `0.000249932`.
  - cumulative3 mean KL: g16 unweighted `0.005718965`, g16 weighted `0.001522995`, g8 weighted `0.001281331`.
  - added mixed family policy support to `tools/run-aq-module-logit-smoke.py` via `--policy NAME=family1,family2`.
  - cumulative runs now store selected original weights on CPU and have `--max-original-weight-mib` as a guard.
  - layer0 mixed policy result: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-layer0-policy-r9700-calib32-qwen35-9b-prompts8.jsonl`.
  - layer0 mean logit relative MSE: all-g16 `0.000299323`, all-g8 `0.000198038`, p4p6 `0.000302250`, p4p9 `0.000192349`.
  - policy5 mixed policy result: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-policy5-r9700-calib32-qwen35-9b-prompts8.jsonl`.
  - policy5 mean logit relative MSE: all-g16 `0.000286738`, all-g8 `0.000284312`, p4p6 `0.000225818`, p4p9 `0.000284312`.
  - interpretation: p4p6 improved policy5 logit relative MSE by keeping `mlp_up` at g16 while using g8 for `attn_k,attn_o,attn_v,linear_attn_out`; KL still did not improve, so this is an ordering signal, not a quality conclusion.
  - added `tools/select-aq-logit-smoke-modules.py` to generate reproducible module sets from activation stats.
  - policy10 selection: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-logit-smoke-selection-policy10.json`.
  - policy10 result: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-policy10-r9700-calib32-qwen35-9b-prompts8.jsonl`.
  - policy10 mean logit relative MSE: all-g16 `0.000398490`, all-g8 `0.000426076`, p4p6 `0.000369140`, p4p9 `0.000426076`.
  - policy10 mean KL: all-g16 `0.001178040`, all-g8 `0.001987679`, p4p6 `0.001034530`, p4p9 `0.001987679`.
  - interpretation: p4p6 beat all-g16 and all-g8 on both logit relative MSE and KL for this 10-module smoke; `mlp_up` should remain a lower-priority g8 family until broader model-level evidence says otherwise.

- `ullm-quant` metadata planner:
  - added safetensors header planning without reading tensor payloads.
  - generated `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-plan-qwen35-9b.json`.
  - Qwen3.5-9B plan: total tensors `775`, default quantize tensors `255`, passthrough tensors `520`, total tensor bytes `19306216416`.
  - default quantize target is known text linear families only; embeddings, lm head, vision, conv, MTP, and unknown tensors pass through for now.
  - added aq policy planning options: `--aq-policy all-g16|all-g8|p4p6|p4p9|custom`, `--aq-high-family`, `--aq-low-format`, `--aq-high-format`.
  - generated p4p6 plan: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-plan-qwen35-9b-p4p6.json`.
  - p4p6 plan schema `ullm-quant-plan-v0.3`; low tensors `204`, high tensors `51`, passthrough tensors `520`.
  - p4p6 estimated output bytes: low `3655729152`, high `393216000`, passthrough `5049777120`, total `9098722272`; input tensor bytes `19306216416`, estimated output/input ratio `0.471285`.
  - added safetensors payload chunk reader in Rust so future conversion can read tensor bytes by offset/length instead of loading whole tensors.
  - `cargo test -p ullm-quant` passes 5 tests, including a handcrafted safetensors chunk-read test.
  - real-model chunk inspection output: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-inspect-qwen35-9b-layer0-mlp-up.txt`.
  - inspected `model.language_model.layers.0.mlp.up_proj.weight`: BF16 `[12288, 4096]`, payload `100663296` bytes, chunk size `1048576`, chunks `96`, FNV-1a64 `16e6f2e89dfb833b`.
  - chunked numeric stats for the same tensor: elements `50331648`, min `-0.104980469`, max `0.131835938`, mean_abs `0.008388692`, max_abs `0.131835938`, NaN count `0`.
  - chunked aq group stats for `aq4_e4m3_g16_ts_flloyd16`: group_size `16`, groups `3145728`, group_absmax_mean `0.021983589`, group_absmax_max `0.131835938`.
  - direct E4M3 scale dry-run on group absmax: scale_count `119`, scale_index_min `2`, scale_index_max `31`, clamped_low `0`, clamped_high `0`, mean_relative_scale_error `0.024264209`.
  - attention high-format inspection output: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-inspect-qwen35-9b-layer3-attn-k-g8.txt`.
  - inspected `model.language_model.layers.3.self_attn.k_proj.weight` with `aq4_e4m3_g8_ts_flloyd16`: BF16 `[1024, 4096]`, group_size `8`, groups `524288`, group_absmax_mean `0.028590068`, group_absmax_max `0.277343750`, scale_index_min `0`, scale_index_max `40`, clamped_low `1`, clamped_high `0`, mean_relative_scale_error `0.024841432`.
  - added `tools/export-aq-family-codebooks.py` to export sampled family codebook values for Rust dry-runs.
  - exported codebook artifact: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-codebooks-qwen35-9b-mlp-up-attn-k-weighted.json`; families `mlp_up,attn_k`; candidates `aq4_e4m3_g16_ts_flloyd16,aq4_e4m3_g8_ts_flloyd16`; weighted codebook with calib32 activation stats.
  - Rust codebook inspection output: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-inspect-codebook-mlp-up-g16.txt`; loaded `mlp_up` + `aq4_e4m3_g16_ts_flloyd16`, 16 entries, min `-0.966317832`, max `0.968481123`.
  - generated policy size summary: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-policy-size-summary-qwen35-9b.json`.
  - estimated output bytes: all-g16 `9059400672`, p4p6 `9098722272`, p4p9 `9325214688`, all-g8 `9504914400`.
  - size interpretation: p4p6 is only `39321600` bytes above all-g16, while p4p9 is `265814016` bytes above all-g16.
  - updated `docs/plans/aq-full-quantizer-design-v0.1.md` with current p4p6 policy, plan size results, chunk inspection status, and revised immediate Rust implementation steps.
  - added exported codebook loading to `ullm-quant` inspection.
  - added streamed one-tensor quantization dry-run: direct E4M3 group scale plus nearest 4-bit family LUT assignment.
  - dry-run result for `model.language_model.layers.0.mlp.up_proj.weight` with `aq4_e4m3_g16_ts_flloyd16`: relative MSE `0.006231116836`, max abs error `0.006380409`, output path `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-dry-run-qwen35-9b-layer0-mlp-up-g16.txt`.
  - dry-run result for `model.language_model.layers.3.self_attn.k_proj.weight` with `aq4_e4m3_g8_ts_flloyd16`: relative MSE `0.004610619768`, max abs error `0.012256019`, output path `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-dry-run-qwen35-9b-layer3-attn-k-g8.txt`.
  - `cargo test -p ullm-quant` passes 9 tests after adding a direct nearest-codebook reconstruction unit test.
  - added `_ts_` tensor-scale estimation and `--scale-window` per-group scale search to `ullm-quant`.
  - scale-window 4 result for `mlp_up` g16: tensor scale `0.014789051376`, relative MSE `0.005283509762`, improved groups `1612071`, output path `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-dry-run-qwen35-9b-layer0-mlp-up-g16-scale-window4.txt`.
  - scale-window 4 result for `attn_k` g8: tensor scale `0.018260609359`, relative MSE `0.003677692937`, improved groups `259635`, output path `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-dry-run-qwen35-9b-layer3-attn-k-g8-scale-window4.txt`.
  - added `tools/verify-aq-one-tensor.py` as a chunked Python reference for one full tensor with exported family codebooks.
  - Python reference matched Rust for `attn_k` index counts exactly; `mlp_up` relative MSE matched within about `1.3e-10`, with only tiny count differences from tensor-scale rounding/ties.
  - `cargo test -p ullm-quant` passes 10 tests; `python3 -m py_compile tools/verify-aq-one-tensor.py` passes.
  - added `--prototype-output-dir` to `ullm-quant` for one inspected tensor.
  - prototype output writes `manifest.json`, packed idx4 indices, u8 scale indices, and f32-le codebook values.
  - added re-read/dequant verification for prototype output; it fails if verified relative MSE differs from manifest relative MSE by more than `1e-9`.
  - first real prototype output: `benchmarks/results/2026-07-01/aq/prototype-qwen35-9b-layer3-attn-k-g8-scale-window4.ullm.d/`.
  - prototype tensor `model.language_model.layers.3.self_attn.k_proj.weight`: idx4 bytes `2097152`, scale bytes `524288`, relative MSE `0.003677692937`, verified relative MSE `0.003677692937`.
  - prototype write log: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-prototype-write-qwen35-9b-layer3-attn-k-g8-scale-window4.txt`; elapsed `1.71 s`, max RSS `8232 KiB`.
  - added `--skip-inspect` and `--prototype-skip-verify` so write-only prototype benchmarks do not rerun duplicate inspection or re-read verification.
  - larger scalar Rust write-only benchmark: `model.language_model.layers.0.mlp.up_proj.weight`, `aq4_e4m3_g16_ts_flloyd16`, output written under `/tmp`, log retained at `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-prototype-write-benchmark-qwen35-9b-layer0-mlp-up-g16-scale-window4.txt`.
  - `mlp_up` write-only benchmark: relative MSE `0.005283509762`, idx4 bytes `25165824`, scale bytes `3145728`, elapsed `8.76 s`, max RSS `21560 KiB`, throughput about `5.75M` elements/s.
  - added first C++20 BF16 chunk quantization kernel behind `ullm_aq_quantize_bf16_chunk`.
  - C++ kernel owns best-scale search, nearest-codebook assignment, idx4 packing, scale-index output, and chunk metrics; Rust still owns safetensors I/O, tensor-scale estimation, manifest writing, and verification.
  - C++ kernel write-only benchmark for `mlp_up` g16: relative MSE `0.005283509762`, elapsed `7.13 s`, max RSS `21516 KiB`, about `7.06M` elements/s, log `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-prototype-write-benchmark-cxx-qwen35-9b-layer0-mlp-up-g16-scale-window4.txt`.
  - C++ kernel real-tensor re-read verification for `attn_k` g8 succeeded: relative MSE and verified relative MSE `0.003677692937`, elapsed `0.74 s`, max RSS `8220 KiB`, log `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-prototype-cxx-verify-qwen35-9b-layer3-attn-k-g8-scale-window4.txt`.
  - changed the C++ entry used by Rust to `ullm_aq_quantize_chunk_v1`, with a request struct that includes `struct_size`, dtype id, pointers, buffer sizes, group size, scale table, codebook, tensor scale, and scale window.
  - `quantize_chunk_v1` currently supports BF16 only; unsupported dtype ids return `-5`.
  - `cargo test -p ullm-quant` passes 16 tests including C++ scale-window/packing, unsupported-dtype/output-buffer validation, all-zero groups, NaN handling, invalid scale/codebook/layout validation, and a BF16 chunk golden that compares C++ metrics against the Rust scalar dry-run.
  - added `--tensor-scale-override` for prototype output to skip exact tensor-scale estimation when a correct tensor scale is already known.
  - C++ one-pass `mlp_up` g16 benchmark with tensor scale override `0.014789051376`: relative MSE `0.005283509762`, elapsed `6.99 s`, max RSS `4180 KiB`, about `7.20M` elements/s, log `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-prototype-write-benchmark-cxx-onepass-qwen35-9b-layer0-mlp-up-g16-scale-window4.txt`.
  - interpretation: one-pass override only improved wall time slightly versus C++ pre-pass (`7.13 s -> 6.99 s`), but significantly reduced peak RSS (`21516 KiB -> 4180 KiB`).

## Current Interpretation

Concrete measurement should continue in parallel with quantizer optimization. A separate long theory-only phase is not useful now, but full-model conversion will require a dedicated CPU-multithreaded quantizer implementation.

The current aq result is promising at 4.5 bpp: it beats sampled NVFP4 and slightly beats sampled UD `Q4_K` rows. The family-level LUT result remained close even at up to 8 tensors per family, so the next uncertainty is not obvious LUT instability. The larger risk is activation sensitivity and model-level behavior.

## Next

- Add F16 support to `quantize_chunk_v1` after BF16 semantics are stable.
- Add larger C++ vs Python/Rust golden tests across random seeds and output bytes.
- Replace exact tensor-scale pre-pass with a lower-memory estimator or scheduling strategy for multi-tensor conversion.
- Add SIMD kernels after scalar C++ semantics are locked.
- Extend the output path from one tensor to all tensors selected by the p4p6 plan.
