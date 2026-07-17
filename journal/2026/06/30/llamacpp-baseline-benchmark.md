# llama.cpp baseline benchmark

## Done

- Updated local `reference-src/llama.cpp` from `86b94708f224` to `6c5de1cc83537bce5616ed08474f6fe119973a27` (`b9844`) and rebuilt `llama-bench` with HIP.
- Added `tools/run-llamacpp-benchmark.py` to convert `llama-bench` JSONL into the uLLM benchmark schema.
- Added timeout support and corrected multi-device handling to use slash-separated llama.cpp device strings.
- Collected Qwen3.5 Q4_K_M baseline rows for V620, R9700, V620 x2, and V620/R9700/V620 layer split.
- Recorded V620 unsupported rows for vLLM, SGLang, ROCm/ATOM, and TensorRT-LLM.
- Tried Qwen3.5-27B FP8_E4M3 and FP8_E5M2 on V620/R9700; both failed to load under the current llama.cpp build.

## Outputs

- `uLLM-project/benchmarks/results/2026-06-30/llama.cpp/summary.md`
- `uLLM-project/benchmarks/results/2026-06-30/llama.cpp/*.jsonl`

## Notes

- The first multi-GPU attempt used comma-separated devices and was removed because llama.cpp interpreted it as a device sweep. Correct multi-device runs use `ROCm0/ROCm2` and `ROCm0/ROCm1/ROCm2`.
- Full-grid multi-GPU sweeps are expensive; representative multi-GPU runs are enough for the first baseline.
