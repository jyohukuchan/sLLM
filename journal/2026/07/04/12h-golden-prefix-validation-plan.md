# 12h golden prefix validation plan

## Progress

- Added `uLLM-project/docs/plans/12h-golden-prefix-validation-plan-v0.1.md`.
- The plan follows the previous 1-layer golden smoke and scopes the next larger task to contiguous decoder layer prefix validation.
- Main target is a `package-golden-prefix-smoke` path that runs `0..4` first, then expands to `0..8`, `0..12`, `seq16`, and optional V620 validation.
- Added terms to `uLLM-project/docs/words.txt`:
  - `golden prefix fixture`
  - `package golden prefix smoke`
- Started executing the plan.
- Confirmed `uLLM-project/journal/` is absent and the outer `ultimateLLM/journal/` tree is present.
- Added exporter support for `--layer-range START:END` through a worker subagent.
- Added `GoldenTensorFixture` contiguous range accessors and prefix metadata fields through a worker subagent.
- Added initial `package-golden-prefix-smoke` CLI integration in `crates/ullm-engine/src/main.rs`.
- The prefix smoke loads a contiguous package model runtime, runs each layer in order, passes actual layer output to the next layer, and can write JSONL report entries.
- During execution, `0..4` exposed that Qwen3.5-9B uses mixed layer kinds:
  - layers `0,1,2,4,5,6,...` are `linear_attn`
  - layers `3,7,11,...` are `self_attn`
- Updated `package-golden-prefix-smoke` to detect the package layer kind and run mixed prefixes:
  - linear attention layers use the package linear attention MLP block sequence path with the fixture/current residual as input.
  - self attention layers use the existing Qwen3 package decoder layer runtime path.
- Added numeric classification in JSONL:
  - `possible_quantization_error` for finite nonzero drift within the current broad band.
  - `numeric_drift` for larger drift such as low cosine or high MSE.

## Rationale

- The previous task finished quickly because the existing package loader and decoder layer runtime were already strong enough for a 1-layer connection.
- A prefix task is more likely to consume a full 12h because it adds layer-to-layer residual chaining, fixture schema expansion, streaming/windowed package load decisions, per-layer machine-readable reports, and CPU/R9700/V620 validation.

## Verification

- `python3 -m py_compile tools/export-qwen-golden-tensors.py` passed.
- `cargo fmt --all --check` passed.
- `cargo check -p ullm-engine` passed.
- `cargo test -p ullm-engine golden -- --test-threads=1` passed.
- `cargo build -p ullm-engine` passed.
- Baseline CPU `package-layer-golden-smoke` on existing layer 3 seq8 fixture passed:
  - `mse=0.010949263712`
  - `mean_abs_diff=0.075410757`
  - `max_abs_diff=6.772741318`
  - `cosine_similarity=0.916806314`
- `cargo test -p ullm-engine -- --test-threads=1` passed.
- `cargo test --workspace -- --test-threads=1` passed.

## Fixtures

- `uLLM-project/benchmarks/golden/2026-07-04/qwen35-9b-prefix0-4-seq8/`
  - layer range: `0..4`
  - layer kinds: `linear_attn, linear_attn, linear_attn, self_attn`
  - shape: `[1, 8, 4096]`
- `uLLM-project/benchmarks/golden/2026-07-04/qwen35-9b-prefix0-8-seq8/`
  - layer range: `0..8`
  - layer kinds: `linear_attn, linear_attn, linear_attn, self_attn, linear_attn, linear_attn, linear_attn, self_attn`
  - shape: `[1, 8, 4096]`

Exporter command pattern:

```bash
HIP_VISIBLE_DEVICES=1 TRANSFORMERS_OFFLINE=1 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  build/envs/sglang-rocm/bin/python tools/export-qwen-golden-tensors.py \
  --model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B \
  --token-ids 1,2,3,4,5,6,7,8 \
  --layer-range 0:4 \
  --output benchmarks/golden/2026-07-04/qwen35-9b-prefix0-4-seq8 \
  --device cuda:0 \
  --dtype bfloat16
```

## Results

Primary `0..4`, `seq8`:

- CPU fallback `0`
  - result: `uLLM-project/benchmarks/results/2026-07-04/engine/package-golden-prefix-smoke-cpu-prefix0-4-seq8.txt`
  - JSONL: `uLLM-project/benchmarks/results/2026-07-04/engine/package-golden-prefix-smoke-cpu-prefix0-4-seq8.jsonl`
  - `max_mse=0.036098765300`
  - `max_mean_abs_diff=0.122858104`
  - `max_abs_diff=15.920772552`
  - `min_cosine_similarity=0.504583529`
  - classes: all `possible_quantization_error`
- R9700/RDNA4 `2`
  - result: `uLLM-project/benchmarks/results/2026-07-04/engine/package-golden-prefix-smoke-r9700-prefix0-4-seq8.txt`
  - JSONL: `uLLM-project/benchmarks/results/2026-07-04/engine/package-golden-prefix-smoke-r9700-prefix0-4-seq8.jsonl`
  - `max_mse=0.036098755031`
  - `max_mean_abs_diff=0.122858101`
  - `max_abs_diff=15.920771599`
  - `min_cosine_similarity=0.504583619`
  - classes: all `possible_quantization_error`
- V620/RDNA2 `1`
  - result: `uLLM-project/benchmarks/results/2026-07-04/engine/package-golden-prefix-smoke-v620-prefix0-4-seq8.txt`
  - JSONL: `uLLM-project/benchmarks/results/2026-07-04/engine/package-golden-prefix-smoke-v620-prefix0-4-seq8.jsonl`
  - metrics matched R9700 at the recorded precision.

Expansion `0..8`, `seq8`:

- CPU fallback `0`
  - `max_mse=0.264669271684`
  - `max_mean_abs_diff=0.228727875`
  - `max_abs_diff=44.618324280`
  - `min_cosine_similarity=-0.083518201`
  - classes: layers `0..3` `possible_quantization_error`, layers `4..7` `numeric_drift`
- R9700/RDNA4 `2`
  - `max_mse=0.264669283836`
  - `max_mean_abs_diff=0.228727879`
  - `max_abs_diff=44.618328094`
  - `min_cosine_similarity=-0.083517839`
  - classes: layers `0..3` `possible_quantization_error`, layers `4..7` `numeric_drift`

Interpretation:

- CPU/R9700/V620 are aligned closely, so the prefix drift is not currently a backend divergence.
- Drift grows sharply after the second self-attention boundary in `0..8`; this is now visible per layer in JSONL.
- The current implementation still records metrics instead of failing on numeric drift. That is intentional for AQ package/reference comparison.

## Next

- If this is executed as a `/goal`, start with `seq8`, `layer_start=0`, `layer_end=4`.
- Keep memory usage explicit. If resident multi-layer f32 weights pressure GPU memory, switch to layer or 2-layer window execution instead of widening the range.
- Next useful action is to decide whether `numeric_drift` after layer 4 is acceptable AQ accumulation or whether layer 4 linear attention needs deeper reference decomposition.
