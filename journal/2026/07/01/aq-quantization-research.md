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
  - default quantize target is known linear families; Qwen3.5 MTP linear weights are quantized when their names match `mlp_*` or `self_attn.*_proj`. Embeddings, lm head, vision, conv, normalization tensors, non-linear MTP tensors, and unknown tensors pass through for now.
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
  - added named aq policy presets `p4p46_inproj` and `p4p65_inproj` to `ullm-quant`; short aliases `p4p46` and `p4p65` use the same high-family sets.
  - generated p4p46 plan: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-plan-qwen35-9b-p4p46-inproj.json`; high tensors `114`, low tensors `141`, passthrough tensors `520`, estimated output bytes `9121922016`, output/input ratio `0.472486`.
  - generated p4p65 plan: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-plan-qwen35-9b-p4p65-inproj.json`; high tensors `123`, low tensors `132`, passthrough tensors `520`, estimated output bytes `9149447136`, output/input ratio `0.473912`.
  - generated updated size summary: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-policy-size-summary-qwen35-9b-inproj.json`.
  - size interpretation: p4p46 is `23199744` estimated bytes above p4p6; p4p65 is `50724864` estimated bytes above p4p6. Both are much smaller than p4p9's delta.
  - verification after policy preset change: `cargo fmt -p ullm-quant`, `cargo test -p ullm-quant`, and `cargo build -p ullm-quant --release` pass. Tests now pass 22 unit tests.
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
  - added F16 decode support to Rust numeric stats and C++ `quantize_chunk_v1`.
  - `cargo test -p ullm-quant` passes 18 tests including F16 numeric decode and F16 C++ quantization/packing smoke.
  - added `tools/run-ullm-prototype-policy-smoke.py` to drive multiple single-tensor prototype conversions from a plan JSON and exported codebook JSON.
  - p4p6 smoke summary: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-mlp-up-attn-k.json`; logs in `benchmarks/results/2026-07-01/aq/prototype-policy-smoke-qwen35-9b-p4p6-mlp-up-attn-k-logs/`; binary prototype dirs written under `/tmp`.
  - p4p6 smoke converted 4 tensors: layer0/layer1 `mlp_up` g16 and layer11/layer15 `attn_k` g8. Relative MSE range `0.003702330162` to `0.005288028063`; elapsed times about `7.4-7.8 s` for `mlp_up` and `0.73 s` for `attn_k`.
  - added `tools/merge-ullm-prototype-dirs.py` to merge per-tensor prototype dirs into one shared `.ullm.d` directory.
  - merged p4p6 smoke summary: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-qwen35-9b-p4p6-mlp-up-attn-k.json`; output under `/tmp/ullm-prototype-policy-smoke-qwen35-9b-p4p6-mlp-up-attn-k-merged.ullm.d`; tensor count `4`, shared codebooks `2`, total file bytes `61872608`.
  - added `--verify-prototype-dir` and `--verify-prototype-all` to verify existing prototype manifests.
  - merged 4-tensor prototype verify log: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-verify-qwen35-9b-p4p6-mlp-up-attn-k.txt`; all 4 tensors verified with matching relative MSE; elapsed `0.74 s`, max RSS `29764 KiB`.
  - extended `tools/export-aq-family-codebooks.py` with explicit `--missing-activation-stats unweighted` fallback for weighted codebook export.
  - full p4p6 family codebook export: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-codebooks-qwen35-9b-p4p6-families-weighted.json`; log `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-codebooks-qwen35-9b-p4p6-families-weighted.log`.
  - full export contains 24 codebooks: 12 families times 2 candidates. 16 are activation-weighted; 8 use `unweighted_missing_activation_stats` because current activation stats include `linear_attn.out_proj` but not `linear_attn.in_proj_qkv/a/b/z`.
  - full export resource use: elapsed `11.31 s`, max RSS `617952 KiB`.
  - expanded p4p6 prototype smoke to one tensor per quantized family. Summary: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-all-families.json`; driver log `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-all-families-driver.log`; per-tensor logs in `benchmarks/results/2026-07-01/aq/prototype-policy-smoke-qwen35-9b-p4p6-all-families-logs/`.
  - all-family smoke converted and verified 12/12 tensors. Relative MSE ranged from `0.003642895769` (`attn_o` g8) to `0.005458763018` (`linear_attn_b` g16); largest per-tensor RSS was `31148 KiB`; driver elapsed `42.75 s`.
  - merged all-family smoke summary: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-qwen35-9b-p4p6-all-families.json`; output under `/tmp/ullm-prototype-policy-smoke-qwen35-9b-p4p6-all-families-merged.ullm.d`; tensor count `12`, codebook count `12`, total file bytes `158503771`.
  - merged all-family verify log: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-verify-qwen35-9b-p4p6-all-families.txt`; all 12 tensors verified; elapsed `2.16 s`, max RSS `101196 KiB`.
  - widened the p4p6 prototype smoke to two tensors per quantized family. Summary: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-family2.json`; driver log `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-family2-driver.log`; logs in `benchmarks/results/2026-07-01/aq/prototype-policy-smoke-qwen35-9b-p4p6-family2-logs/`.
  - family2 smoke converted and verified 24/24 tensors. Relative MSE ranged from `0.003639662156` (`attn_o` g8) to `0.005540913549` (`linear_attn_b` g16); largest per-tensor RSS was `30852 KiB`; driver elapsed `1:27.16`.
  - merged family2 summary: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-qwen35-9b-p4p6-family2.json`; output under `/tmp/ullm-prototype-policy-smoke-qwen35-9b-p4p6-family2-merged.ullm.d`; tensor count `24`, codebook count `12`, total file bytes `317004099`.
  - merged family2 verify log: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-verify-qwen35-9b-p4p6-family2.txt`; all 24 tensors verified; elapsed `4.09 s`, max RSS `101216 KiB`.
  - widened the p4p6 prototype smoke to four tensors per quantized family. Summary: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-family4.json`; driver log `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-family4-driver.log`; logs in `benchmarks/results/2026-07-01/aq/prototype-policy-smoke-qwen35-9b-p4p6-family4-logs/`.
  - family4 smoke converted and verified 48/48 tensors. Relative MSE ranged from `0.003639662156` (`attn_o` g8) to `0.005741676939` (`linear_attn_b` g16); largest per-tensor RSS was `32076 KiB`; driver elapsed `2:45.31`.
  - merged family4 summary: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-qwen35-9b-p4p6-family4.json`; output under `/tmp/ullm-prototype-policy-smoke-qwen35-9b-p4p6-family4-merged.ullm.d`; tensor count `48`, codebook count `12`, total file bytes `634004817`.
  - merged family4 verify log: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-verify-qwen35-9b-p4p6-family4.txt`; all 48 tensors verified; elapsed `8.12 s`, max RSS `101252 KiB`.
  - ran the full p4p6 quantized-tensor prototype conversion. Summary: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-full-quantized.json`; driver log `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-full-quantized-driver.log`; logs in `benchmarks/results/2026-07-01/aq/prototype-policy-smoke-qwen35-9b-p4p6-full-quantized-logs/`.
  - full quantized conversion selected all 255 p4p6 quantized tensors, including 7 MTP linear tensors, and skipped per-tensor re-read verification; all 255 returned success. Relative MSE ranged from `0.003639662156` (`attn_o`) to `0.005783048676` (`linear_attn_a`). Driver elapsed `17:23.16`, max RSS `22616 KiB`, parts directory size `3.8 GiB`.
  - merged full quantized summary: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-qwen35-9b-p4p6-full-quantized.json`; output under `/tmp/ullm-prototype-policy-smoke-qwen35-9b-p4p6-full-quantized-merged.ullm.d`; tensor count `255`, codebook count `12`, total file bytes `4049329404`.
  - merged full quantized verify log: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-verify-qwen35-9b-p4p6-full-quantized.txt`; all 255 tensors verified; elapsed `47.48 s`, max RSS `103892 KiB`; relative-MSE values matched summary with max delta about `1e-12`.
  - extended `tools/merge-ullm-prototype-dirs.py` with optional `--include-passthrough`, `--plan-json`, and `--copy-buffer-bytes`. It streams safetensors payloads for passthrough tensors into `passthrough/*.raw` and records them in top-level manifest field `passthrough_tensors`.
  - full-package prototype summary: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-qwen35-9b-p4p6-full-package.json`; merge log `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-qwen35-9b-p4p6-full-package.log`; output under `/tmp/ullm-prototype-policy-smoke-qwen35-9b-p4p6-full-package.ullm.d`.
  - full-package prototype has 255 quantized tensors, 520 passthrough tensors, 12 codebooks, passthrough payload bytes `5049777120`, total file bytes `9099409599`, directory size `8.5 GiB`. Merge elapsed `8.71 s`, max RSS `36240 KiB`.
  - existing Rust verifier accepts the passthrough-extended manifest and verifies the 255 quantized tensors while ignoring `passthrough_tensors`: log `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-verify-qwen35-9b-p4p6-full-package.txt`; elapsed `48.63 s`, max RSS `103296 KiB`.
  - added explicit Rust passthrough verification via `--verify-passthrough`; `sha2` dependency added for SHA-256. The verifier streams payload files and checks byte length plus `payload_sha256`.
  - passthrough verification log: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-verify-passthrough-qwen35-9b-p4p6-full-package.txt`; verified 255 quantized tensors and 520 passthrough tensors; passthrough payload bytes `5049777120`; elapsed `55.37 s`, max RSS `104596 KiB`.
  - `cargo test -p ullm-quant` passes 20 tests after adding manifest default and hex encoding tests.
  - moved prototype merge behavior into `ullm-quant` itself.
  - added merge CLI flags: `--merge-policy-summary`, `--merge-plan-json`, `--merge-output-dir`, `--merge-summary-output`, `--merge-include-passthrough`, `--merge-copy-buffer-bytes`, and `--merge-overwrite`.
  - Rust merge copies quantized index/scale files, deduplicates codebooks by `(family,candidate_id)`, and streams passthrough safetensors payloads while computing SHA-256.
  - added a unit fixture covering two quantized tensors sharing one codebook plus one passthrough tensor copied with a tiny buffer.
  - test status after merge implementation: `cargo fmt -p ullm-quant --check`, `cargo test -p ullm-quant`, and `cargo build -p ullm-quant --release` pass. `cargo test -p ullm-quant` now passes 21 tests.
  - Rust full-quantized merge summary: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-rust-merged-qwen35-9b-p4p6-full-quantized.json`; log `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-rust-merged-qwen35-9b-p4p6-full-quantized.log`; output `/tmp/ullm-prototype-policy-smoke-qwen35-9b-p4p6-full-quantized-rust-merged.ullm.d`.
  - Rust full-quantized merge result: 255 tensors, 12 codebooks, total file bytes `4049329123`, directory size `3.8 GiB`, elapsed `1.55 s`, max RSS `2076 KiB`.
  - Rust full-quantized verify log: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-rust-merged-verify-qwen35-9b-p4p6-full-quantized.txt`; all 255 tensors verified; elapsed `52.55 s`, max RSS `102216 KiB`.
  - Rust full-package merge summary: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-rust-merged-qwen35-9b-p4p6-full-package.json`; log `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-rust-merged-qwen35-9b-p4p6-full-package.log`; output `/tmp/ullm-prototype-policy-smoke-qwen35-9b-p4p6-full-package-rust-merged.ullm.d`.
  - Rust full-package merge result: 255 quantized tensors, 520 passthrough tensors, 12 codebooks, passthrough payload bytes `5049777120`, total file bytes `9099409318`, directory size `8.5 GiB`, elapsed `8.77 s`, max RSS `12372 KiB`.
  - Rust full-package verify log: `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-rust-merged-verify-passthrough-qwen35-9b-p4p6-full-package.txt`; verified 255 quantized tensors and 520 passthrough tensors; elapsed `54.52 s`, max RSS `103288 KiB`.
  - Python and Rust merged manifest JSON files differ by 281 bytes because serde and Python format a few float values differently. Tensor files, codebook files, passthrough byte counts, and SHA-256 verification are consistent.
- Linear-attention in-projection activation stats:
  - fixed `tools/collect-activation-stats.py` default regex so it matches Qwen3.5 `linear_attn.in_proj_qkv`, `in_proj_a`, `in_proj_b`, and `in_proj_z`.
  - reran R9700 activation stats with calib32 prompts: `benchmarks/results/2026-07-01/aq/activation-r9700-calib32-qwen35-9b-s512-inproj/`; log `benchmarks/results/2026-07-01/aq/activation-r9700-calib32-qwen35-9b-s512-inproj.log`.
  - stats result: 32 prompts, 14,061 tokens, 248 matched modules, 744 safetensors stat keys, and 24 modules each for `linear_attn.in_proj_qkv/a/b/z/out_proj`; elapsed `33.02 s`, max RSS `15902624 KiB`.
  - reran full-family weighted codebook export with `--missing-activation-stats error`: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-codebooks-qwen35-9b-p4p6-families-weighted-inproj.json`; 24/24 codebooks weighted, fallback rows `0`, elapsed `12.30 s`, max RSS `618104 KiB`.
  - reran family4 weighted tensor sample: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-weighted-scale-codebook-r9700-calib32-inproj-qwen35-9b-family4.jsonl`; 96 rows, 0 failures, elapsed `15.44 s`, max RSS `633212 KiB`.
  - policy summary: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-policy-r9700-calib32-inproj-qwen35-9b-family4.json`.
  - new family4 tensor policy estimates: all-g16 combined weighted relative MSE `0.003225949` at 4.5 bpp; cap 4.60 with `attn_o,attn_v,linear_attn_a,linear_attn_b,linear_attn_out,linear_attn_z` high gives `0.002340079` at bpp `4.598865`; cap 4.70 with `attn_k,attn_o,attn_v,linear_attn_a,linear_attn_b,linear_attn_out,linear_attn_qkv,linear_attn_z` high gives `0.002139622` at bpp `4.666982`; all-g8 gives `0.001886067` at 5.0 bpp.
  - in-proj12 logit smoke selection: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-logit-smoke-selection-inproj12.json`.
  - in-proj12 logit smoke result: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-inproj12-r9700-calib32-qwen35-9b-prompts8.jsonl`; log `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-inproj12-r9700-calib32-qwen35-9b-prompts8.log`.
  - in-proj12 scope: layer0/layer1 `linear_attn.in_proj_a/b/qkv/z`, `linear_attn.out_proj`, and `mlp.up_proj`; 8 prompts; total original selected weight bytes `470810624`; elapsed `5:39.00`; max RSS `16365640 KiB`.
  - in-proj12 mean logit relative MSE / KL: all-g16 `0.000347698` / `0.001960925`; all-g8 `0.000402234` / `0.001103623`; old p4p6 `0.000416993` / `0.001656361`; p4p46_inproj `0.000349033` / `0.001686816`; p4p65_inproj `0.000361837` / `0.001986985`; p4p10_inproj `0.000403207` / `0.001373638`.
  - interpretation: tensor metrics strongly prefer promoting several in-proj families to g8, but this in-proj-heavy logit smoke still ranks all-g16 best by relative MSE and all-g8 best by KL. `p4p46_inproj` is the best mixed policy by relative MSE and clearly improves over old p4p6 in this scope, but it needs a wider module/perplexity run before replacing p4p6.
  - wider self-attn + in-proj selection: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-logit-smoke-selection-inproj22-selfattn.json`.
  - wider self-attn + in-proj logit smoke result: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-inproj22-selfattn-r9700-calib32-qwen35-9b-prompts8.jsonl`; log `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-inproj22-selfattn-r9700-calib32-qwen35-9b-prompts8.log`.
  - wider scope: layer0/layer12 `linear_attn.in_proj_a/b/qkv/z`, `linear_attn.out_proj`, `mlp.up_proj`; layer3/layer7 `self_attn.q/k/v/o_proj`, `mlp.up_proj`; 22 modules; total original selected weight bytes `907018240`; elapsed `11:11.54`; max RSS `16367660 KiB`.
  - wider mean logit relative MSE / KL: all-g16 `0.000579478` / `0.001758983`; all-g8 `0.000452192` / `0.002217304`; p4p6 `0.000392221` / `0.001303061`; p4p46_inproj `0.000384154` / `0.001293804`; p4p65_inproj `0.000412484` / `0.001134097`; p4p70_inproj `0.000387565` / `0.001182557`; p4p80_inproj `0.000426451` / `0.001241760`.
  - interpretation update: with dense self-attn included, mixed policies beat all-g16/all-g8 on both relative MSE and KL. `p4p46_inproj` is best by relative MSE, `p4p65_inproj` is best by KL, and `p4p70_inproj` is close by relative MSE with full top10 overlap. Keep p4p6 as conservative baseline, but p4p46/p4p65 should become named follow-up candidates.
  - added `tools/run-aq-module-loss-smoke.py` for cumulative next-token cross-entropy smoke. It reuses the existing logit-smoke quantization/policy implementation and adds optional `--repeat-to-length`.
  - short-prompt loss smoke result: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-loss-smoke-inproj22-selfattn-r9700-calib32-qwen35-9b-prompts8.jsonl`; log `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-loss-smoke-inproj22-selfattn-r9700-calib32-qwen35-9b-prompts8.log`; 40 rows, only 111 target tokens, elapsed `9:11.85`, max RSS `16370264 KiB`.
  - short-prompt token-weighted loss delta: all-g16 `+0.001532838`, all-g8 `-0.004225286`, p4p6 `-0.000037107`, p4p46 `-0.011064199`, p4p65 `+0.003953395`. This run is too small for policy selection.
  - repeat128 loss smoke result: `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-loss-smoke-inproj22-selfattn-r9700-calib32-qwen35-9b-prompts8-repeat128.jsonl`; log `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-loss-smoke-inproj22-selfattn-r9700-calib32-qwen35-9b-prompts8-repeat128.log`; 40 rows, 1016 target tokens, elapsed `10:44.75`, max RSS `16363884 KiB`.
  - repeat128 token-weighted loss delta: all-g16 `+0.001027819`, all-g8 `-0.004386369`, p4p6 `-0.011532098`, p4p46 `-0.006359033`, p4p65 `-0.004686363`.
  - interpretation update: repeat128 loss ranks p4p6 best and p4p46 second. Negative deltas on repeated prompts are not quality proof, but the relative ordering keeps p4p6 as the conservative full-conversion baseline while keeping p4p46 as the strongest in-proj follow-up.

## Current Interpretation

Concrete measurement should continue in parallel with quantizer optimization. A separate long theory-only phase is not useful now, but full-model conversion will require a dedicated CPU-multithreaded quantizer implementation.

The current aq result is promising at 4.5 bpp: it beats sampled NVFP4 and slightly beats sampled UD `Q4_K` rows. The family-level LUT result remained close even at up to 8 tensors per family, so the next uncertainty is not obvious LUT instability. The larger risk is activation sensitivity and model-level behavior. The in-proj stats fix removed an unweighted fallback, and the wider self-attn smoke supports p4p46/p4p65 as real policy candidates, but the repeated-prompt loss smoke still favors p4p6 as the conservative baseline.

## Next

- Add larger C++ vs Python/Rust golden tests across random seeds and output bytes.
- Run full-policy prototype conversion for p4p6, p4p46, and p4p65, using the in-proj-weighted codebooks.
- Run a less artificial perplexity or next-token loss smoke for p4p6, p4p46, and p4p65 on a real text calibration set.
- Replace exact tensor-scale pre-pass with a lower-memory estimator or scheduling strategy for multi-tensor conversion.
- Add SIMD kernels after scalar C++ semantics are locked.
- Replace the current per-tensor temporary conversion driver with a single `ullm-quant` full-conversion command.
- Decide whether manifest JSON needs canonical float/text formatting or whether semantic JSON plus payload hashes are sufficient for the prototype.
