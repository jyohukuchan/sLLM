# sLLM GGUF format contract

## Status and scope

This document is the implemented Phase 20 format authority for the public model
container. The bounded reader, deterministic writer, converters, derived lock,
and GGUF-only public runtime were completed on 2026-08-17.

The public artifact is one GGUF file containing the model metadata, tensor
payloads, tokenizer, vocabulary, special-token data, and chat template needed by
the supported runtime path. Safetensors, quantization sidecars, and external
frontend files remain conversion inputs only.

## Base container

- GGUF magic bytes: `GGUF`.
- GGUF version: `3`.
- Byte order: little-endian for the initial Linux runtime.
- Default and required Phase 20 tensor-data alignment: 32 bytes.
- Distribution shape: one file. Split GGUF is outside the initial contract.
- Initial standard architecture values: `qwen35`, `qwen35moe`, and `gemma4`.
- Standard metadata keys are used wherever the pinned source defines the same
  semantic value, including `general.architecture`, `general.alignment`,
  tokenizer fields, token IDs, and `tokenizer.chat_template`.

The inspected base is the MIT-licensed llama.cpp `b10453`, full commit
`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`. Exact inspected file hashes are in
`ci/matrix/phase20-gguf-a0-v1.json`. Inspection does not by itself create a code
import. A later direct reuse must add the provenance event and notices required
by `docs/provenance/README.md` before release.

## Tensor encoding boundary

| sLLM semantic encoding | GGUF representation fixed by A0 | Conversion |
| --- | --- | --- |
| unquantized BF16 | standard `GGML_TYPE_BF16` (`30`), block 1 / 2 bytes | preserve BF16 bits |
| OCP MXFP4 E2M1, block 32, E8M0 scale | standard `GGML_TYPE_MXFP4` (`39`), block 32 / 17 bytes | lossless interleave of scale byte and packed codes |
| NVFP4 E2M1, block 16, E4M3 scale, outer scale | standard `GGML_TYPE_NVFP4` (`40`), super-block 64 / 36 bytes plus named scale tensors | lossless grouping of four source blocks; preserve scale/code bits |
| FP8 E4M3FN plus channel BF16 scale | no standard type in the pinned source | versioned sLLM extension required in A1; no dequantized substitution |

`MXFP4` and `NVFP4` are not interchangeable. Their block size, scale encoding,
packing, and outer-scale rules remain distinct. The converter may move bits into
the standard block layout but must not run a new quantizer. Source and output
logical values are checked by independent decode in a later work unit.

The pinned enum has no FP8 tensor type. sLLM therefore stores FP8 value planes as
standard I8 carrier tensors and binds them to scale tensors with versioned
`sllm.fp8.*` metadata. Readers unaware of the extension can still inspect the
GGUF structure; sLLM rejects missing, unknown, or ambiguous extension bindings.
No dequantized BF16, F16, or Q8_0 substitute is produced.

## Container-neutral lowering

The GGUF reader produces the same internal contracts as the reviewed source
importers. Container-specific offsets and names do not enter execution planning.

- Qwen3.5 dense: reviewed model identity, 738-tensor catalog, text plan, MTP 15
  tensors, vision 297 tensors, tokenizer/chat metadata.
- Gemma 4 BF16: reviewed identity, 677 physical tensors, 666 loadable text
  tensors, architecture/frontend metadata.
- Gemma 4 NVFP4 mixed: 1,389 physical to 677 logical tensors; 144 NVFP4 MLP,
  184 FP8 attention, 48-layer static FP8 KV recipe, BF16/ignored remainder.
- Qwen3.5 MoE MXFP4: reviewed text inventory of 62,053 tensors, 493-entry load
  plan, expert-axis recipe, tokenizer/chat metadata. Vision 333 and MTP 785
  source tensors stay known-unconsumed until their existing execution scope is
  intentionally enabled.

The machine-readable manifest contains the exact fingerprints, revisions,
counts, and recipe digests. A converter or loader must reject a mismatch rather
than infer a nearby layout.

## Derived identity

A GGUF output is a derived artifact of one or more reviewed source locks. Its
model lock records:

- every source lock fingerprint;
- converter repository and full commit;
- complete arguments and effective configuration, including defaults;
- relevant environment and dependency identity;
- output path, size, and SHA-256;
- GGUF metadata and tensor-catalog digests;
- semantic model identity and the encoding/recipe digest lowered to runtime.

The semantic model identity is preserved across source and GGUF containers. The
container digest changes, while aliases move only through an explicit lock
change. Runtime verification binds the opened GGUF descriptor before metadata
or tensor reads and does not reopen the path after verification.

## Fail-closed parser requirements

Before allocation or GPU work, the reader rejects unsupported version or byte
order, duplicate metadata or tensor names, unknown architecture/type/extension,
integer overflow, invalid dimension/block multiple, range overlap, range beyond
EOF, bad alignment, truncated string/array/table, and incomplete or ambiguous
recipe bindings. A GGUF failure never falls back to an unverified safetensors or
sidecar path.

## Implemented boundary

The public CLI and server accept exactly one GGUF plus its derived lock. Source
safetensors and sidecars remain converter/development inputs only. Runtime
verification hashes the GGUF, validates metadata and tensor ranges, and retains
the verified open descriptor for payload reads; it never falls back to a source
container.
