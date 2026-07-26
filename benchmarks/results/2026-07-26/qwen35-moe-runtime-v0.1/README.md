# Qwen3.5-35B-A3B MoE runtime v0.1 evidence

This directory contains compact, reproducible correctness reports for the
loader-independent MoE runtime foundation. It intentionally does not commit
raw fixture files that duplicate the local checkpoint's BF16 router weights.

Source checkpoint contract:

- model: `Qwen/Qwen3.5-35B-A3B` local BF16 checkpoint;
- config SHA-256: `5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944`;
- layer-0 router raw BF16 SHA-256:
  `d55bd73f2cfdb0bd87a2228e6b1894757b9ca3a6fdd6ecac7418cb2a829fbac3`.

`hf-routing-v5/metadata.json` records the HF-selected IDs and scores for three
inputs. Regenerate its local raw fixture with:

```bash
python3 tools/qwen35_moe_hf_routing_reference.py \
  --model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen3.5-35B-A3B-BF16 \
  --output benchmarks/results/2026-07-26/qwen35-moe-runtime-v0.1/hf-routing-NEW \
  --layer 0 --tokens 3 --seed 20260726
```

The reports show:

- `cpu-runtime-full-verify-v5.json`: every F32 and raw-BF16 synthetic MoE
  stage is bit-identical between the Rust CPU reference and CPU C ABI on both
  the prefill grouped-GEMM path and the separate decode GEMM path;
- `gpu-runtime-full-verify-v5.json`: the same stages on R9700/gfx1201 versus
  the CPU reference;
- `cpu-hf-routing-verify-v5.json` and `gpu-hf-routing-verify-v5.json`: the
  real HF router IDs/scores match exactly;
- `cpu-hf-prefill-grouped-gemm-verify-v5.json` and
  `gpu-hf-prefill-grouped-gemm-verify-v5.json`: a real source expert 3-D BF16 slice,
  deliberately reordered by local grouped IDs, matches HF F32 expected values
  and the prefill ABI exactly;
- `cpu-hf-decode-gemm-verify-v5.json` and
  `gpu-hf-decode-gemm-verify-v5.json`: all eight real first-token selected
  expert slabs from the layer-0 BF16 tensor match HF F32 expected values and
  the separate decode ABI exactly;
- `cpu-hf-routing-exact-tie-v5.json`: an intentional all-tie diagnostic is
  flagged rather than treated as an ordering contract;
- `architecture-hf-trace-self-test/`: the existing trace harness rejects the
  intentional layer-3 corruption at its first affected layer.

The GPU reports were correctness runs only. They used ROCm's isolated R9700
token (`HIP_VISIBLE_DEVICES=1`, `ULLM_HIP_VISIBLE_DEVICES=1`) and contain no
timing result.
