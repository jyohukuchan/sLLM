# Package layer golden smoke

## Progress

- Moved repo-local `uLLM-project/journal/` files into the outer `ultimateLLM/journal/` tree.
- Added `journal/` to `uLLM-project/.gitignore` so future notes stay outside the Git repo.
- Baseline checks passed:
  - `cargo fmt --all --check`
  - `cargo check -p ullm-engine`
  - CPU `package-self-attn-mlp-block-model-loop-smoke` on the Qwen3.5 p4p6 package with layers `3,7,11`, sequence length `3`.
- Added `tools/export-qwen-golden-tensors.py` through a worker subagent and adjusted metadata to emit `before_shape` / `after_shape`.
- Added Rust `golden` module through a worker subagent:
  - directory fixture metadata loader
  - raw f32 little-endian payload reader
  - MSE, mean absolute diff, max absolute diff, cosine similarity metrics
- Started CLI integration for `ullm-engine package-layer-golden-smoke`.

## Current verification

- `python3 tools/export-qwen-golden-tensors.py --help` passed.
- `python3 -m py_compile tools/export-qwen-golden-tensors.py` passed.
- `cargo test -p ullm-engine golden -- --test-threads=1` passed.
- `cargo fmt --all --check` passed.
- `cargo check -p ullm-engine` passed.
- `cargo build -p ullm-engine` passed.
- `cargo test -p ullm-engine -- --test-threads=1` passed.
- `cargo test --workspace -- --test-threads=1` passed.
- `git diff --check` passed.

## Fixture

Created fixture:

- `uLLM-project/benchmarks/golden/2026-07-04/qwen35-9b-layer3-seq8/`

Exporter command:

```bash
HIP_VISIBLE_DEVICES=1 TRANSFORMERS_OFFLINE=1 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  build/envs/sglang-rocm/bin/python tools/export-qwen-golden-tensors.py \
  --model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B \
  --token-ids 1,2,3,4,5,6,7,8 \
  --layers 3 \
  --output benchmarks/golden/2026-07-04/qwen35-9b-layer3-seq8 \
  --device cuda:0 \
  --dtype bfloat16
```

Fixture summary:

- layer: `3`
- sequence length: `8`
- hidden size: `4096`
- shape: `[1, 8, 4096]`
- payload size: about `264K`

## Results

Result files:

- `uLLM-project/benchmarks/results/2026-07-04/engine/package-layer-golden-smoke-cpu-layer3-seq8.txt`
- `uLLM-project/benchmarks/results/2026-07-04/engine/package-layer-golden-smoke-r9700-layer3-seq8.txt`
- `uLLM-project/benchmarks/results/2026-07-04/engine/package-layer-golden-smoke-v620-layer3-seq8.txt`

CPU fallback `0`:

- `mse=0.010949263712`
- `mean_abs_diff=0.075410757`
- `max_abs_diff=6.772741318`
- `cosine_similarity=0.916806314`
- `verified=true`

R9700/RDNA4 runtime device `2`:

- `mse=0.010949259393`
- `mean_abs_diff=0.075410756`
- `max_abs_diff=6.772733688`
- `cosine_similarity=0.916806348`
- `verified=true`

V620/RDNA2 runtime device `1`:

- `mse=0.010949259393`
- `mean_abs_diff=0.075410756`
- `max_abs_diff=6.772733688`
- `cosine_similarity=0.916806348`
- `verified=true`

Interpretation:

- CPU and HIP metrics are effectively aligned, so the runtime backend path is not showing a CPU/HIP divergence for this fixture.
- The nonzero delta is expected to include AQ package quantization error and any reference/runtime implementation differences. The new smoke records metrics rather than treating numeric mismatch as an immediate failure.

## Next

- Broaden fixture coverage to more layers such as `0,3,7,11` if needed.
- Use this boundary before attempting full prompt generation, tokenizer integration, final RMSNorm, lm_head, or sampling.
