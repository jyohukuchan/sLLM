# Reference source inventory v0.1

## Purpose

This document records local reference source checkouts used for architecture and benchmark research. These checkouts are not part of uLLM source distribution and are ignored by Git.

Fetch command:

```bash
tools/fetch-reference-sources.sh
```

Local path:

```text
reference-src/
```

## Inventory

| Project | Local path | Commit | Size | License state | Use |
| --- | --- | --- | --- | --- | --- |
| llama.cpp | `reference-src/llama.cpp` | `6c5de1cc8353` | 191M | MIT license found | GGUF, CPU/GPU layout, quantization comparison |
| vLLM | `reference-src/vllm` | `5b4cb6952310` | 144M | Apache-2.0 license found | serving architecture, scheduler, paged attention, benchmark comparison |
| SGLang | `reference-src/sglang` | `3add35e26dc0` | 125M | Apache-2.0 license found | high-throughput serving, RadixAttention, prefill/decode disaggregation |
| ATOM | `reference-src/atom` | `cce1a6e56dcd` | 15M | MIT license found | ROCm/HIP kernel and serving comparison |
| TensorRT-LLM | `reference-src/tensorrt-llm` | `92147d6e01d7` | 320M | Apache-2.0 license found, with third-party notices | FP8 serving, engine/runtime comparison, benchmark comparison |

## Rules

- Do not copy code from `reference-src/` into uLLM.
- If an implementation detail matters, write a design note that describes the idea without copying source.
- If third-party code must be imported, create an ADR first.
- Re-run the inventory when reference repositories are updated.
