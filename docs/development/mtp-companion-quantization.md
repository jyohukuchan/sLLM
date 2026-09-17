# Qwen3.8 MTP companion quantization

This document describes the opt-in quantized MTP companion for the reviewed
Unsloth Qwen3.8-27B NVFP4 artifact. The target model, shared embedding/output
head, KV cache, and MTP control flow remain the same. Only the companion's
eight projection matrices are supplied by the sidecar:

- `mtp.fc.weight`
- `mtp.layers.0.self_attn.q_proj.weight`
- `mtp.layers.0.self_attn.k_proj.weight`
- `mtp.layers.0.self_attn.v_proj.weight`
- `mtp.layers.0.self_attn.o_proj.weight`
- `mtp.layers.0.mlp.gate_proj.weight`
- `mtp.layers.0.mlp.up_proj.weight`
- `mtp.layers.0.mlp.down_proj.weight`

The seven companion norm tensors remain BF16. The target NVFP4/FP8 weights,
shared BF16 embedding, shared FP8 output head, and KV encoding are not
re-quantized by this feature.

## Encodings and current status

The converter supports these two packed MX sidecar recipes:

| encoding | value/activation | scale and accumulation |
| --- | --- | --- |
| MXFP8 | E4M3 W8A8 | K-axis block 32, E8M0 scale, FP32 accumulation, BF16 output |
| MXFP6 | E3M2 W6A6 | K-axis block 32, E8M0 scale, FP32 accumulation, BF16 output |

For the Phase85 activation-error ablation, it also accepts
`--encoding bf16-roundtrip-mxfp8` and `--encoding bf16-roundtrip-mxfp6`.
These diagnostic recipes quantize the same eight matrices, dequantize their
values, and store them as BF16. The runtime consequently uses the existing
BF16 matrix and activation path. This isolates weight quantization from the
packed MX activation path; it is not an implementation of a packed W8A16 or
W6A16 kernel, and its BF16 execution also applies during prefix priming.

Each diagnostic tensor records the all-element BF16 roundtrip result,
nonfinite values, underflow, overflow, bit mismatches, and the first affected
positions. The two recipe names and combined digests remain distinct even
though both use BF16 payloads. The existing packed MX payloads and recipe
names are unchanged. Shared embedding/head and norms are still excluded.
These recipes do not change the BF16 default companion selection.

The Phase85 native M=1 A16 path is a separate opt-in:
`SLLM_MX_WA_M1_A16=1` keeps BF16 activations for single-row MXFP8/MXFP6
matmuls while reading the existing packed sidecar. It selects kernel ID101
(W8A16) or ID102 (W6A16) and omits the activation quantizer and its workspace.
The runtime reads the switch when preparing the operation. Multi-row
operations retain W8A8/W6A6, including the existing WMMA selection.
Unset or `0` restores the ordinary activation path;
`SLLM_MX_WA_M1_FORCE_BASELINE=1` takes precedence and selects the legacy
baseline. No sidecar reconversion is needed.

A16 changes numerical results relative to A8/A6 and can change MTP proposal
acceptance and the generated sequence. Both exact GPUs passed the six real
matrix shapes and boundary cases against sampled independent FP32 oracles.
The full-model speed benefit depends on the GPU, format, and priming setup;
the BF16 companion remains the default. See the
[experiment plan](../plans/archive/2026/09/11-20/phase85-m1-a16-mtp.md)
and [measurement summary](../../ci/matrix/phase85-a16-mtp-results-v1.json).

For the bounded Phase85 benchmark, `SLLM_PHASE85_MTP_PRIMING_TIMING=1`
records per-call prefix row counts and host wall times. It is disabled by
default and does not change arithmetic or kernel routing. The final ABBA
comparison found practically equal prefix time; the earlier small positive
difference remains in the experiment history.

The sidecar is opt-in. Omitting `--mtp-weights` keeps the bundled BF16 MTP
companion and is the comparison and rollback path. The Qwen3.8 KV default is
separately standard OCP MXFP8 E4; `--kv-cache-encoding fp16` is its explicit
rollback and does not select BF16 MTP weights.

MXFP8 is integrated into the normal loader, graph, resident model, CLI, chat,
and server paths. The initial full comparison did not satisfy the speed
condition for making the quantized companion the default, so BF16 remains the
default. An MXFP6 sidecar can be generated and loaded for bounded development
checks, but its comprehensive GPU quality/performance evaluation was not run
after the MXFP8 speed condition failed. Both exact GPUs passed the bounded operator and public API correctness smoke,
including SSE, cancellation after MTP progress, recovery and cleanup. This is
not a comprehensive MXFP6 quality/performance approval.

## Creating a sidecar

The converter reads the verified source artifact and the embedded reviewed
model lock. Both paths below are the actual Phase84 inputs; generated sidecars
are local artifacts and must not be committed:

```bash
cargo run --release -p sllm-cli --bin sllm-convert-qwen38-mtp -- \
  --artifact-root /home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-NVFP4 \
  --encoding mxfp8 \
  --output-dir /home/homelab1/coding-local/sLLM/.local-artifacts/phase84/mxfp8

cargo run --release -p sllm-cli --bin sllm-convert-qwen38-mtp -- \
  --artifact-root /home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-NVFP4 \
  --encoding mxfp6 \
  --output-dir /home/homelab1/coding-local/sLLM/.local-artifacts/phase84/mxfp6
```

`--artifact-root` and `--output-dir` must be absolute, and the output
directory must not already exist. Each successful conversion publishes exactly
`manifest.json` and `payload.safetensors` after conversion and readback
verification. To regenerate an existing sidecar, choose a new output
directory or remove the old generated directory after confirming it is not in
use.

The converter output includes the encoding, a manifest fingerprint, and a
combined recipe digest. The Phase84 generated identities were:

| sidecar | manifest fingerprint | combined recipe digest |
| --- | --- | --- |
| MXFP8 | `sha256:5a75528a271fed2856ca521a61403d2e044f8861946ebd68fd3863a784e01470` | `sha256:6ef86986eb689c460c397a79b714f2193bc1f3da999ed7c01495b68c1e235858` |
| MXFP6 | `sha256:103b03b0642b238d74c99ca0c38201fcef882e4887f1719bdfb6569e7acd74bc` | `sha256:f85ff503a6e807e34872b6bd93ac16a33c9f2cbfe1f937992f0d87ba4fc5722a` |

The runtime records the selected encoding and digest in its audit data. Graph
and resident construction checks that the sidecar is bound to the same source
artifact, model-lock fingerprint, base recipe, and eight expected matrix
shapes. A changed or truncated manifest/payload, source hash mismatch, unknown
or missing tensor, wrong dtype/shape/range, or payload mutation after
verification is an artifact-integrity error. Startup or tensor upload fails
closed without silently falling back to a different precision.

## Using a sidecar from the public paths

The value of `--mtp-weights` is the sidecar directory, not the individual
`manifest.json` or `payload.safetensors` file. It is accepted only with the
Qwen3.8 NVFP4 source, logical device index 0, and exact target `gfx1030` or
`gfx1201`.

For one-shot generation:

```bash
sllm generate \
  --qwen38-nvfp4 /absolute/path/Qwen3.8-27B-NVFP4 \
  --mtp-weights /absolute/path/qwen38-mtp-mxfp8 \
  --prompt "Implement a small Rust function and explain the tests." \
  --max-new-tokens 128 \
  --device-index 0 --target gfx1030
```

For the interactive chat command, use the same model and sidecar options:

```bash
sllm chat \
  --qwen38-nvfp4 /absolute/path/Qwen3.8-27B-NVFP4 \
  --mtp-weights /absolute/path/qwen38-mtp-mxfp8 \
  --device-index 0 --target gfx1030 \
  --prompt "Review this patch for correctness."
```

For the HTTP server, select `mtp-auto` when a sidecar is supplied:

```bash
sllm-server \
  --qwen38-nvfp4 /absolute/path/Qwen3.8-27B-NVFP4 \
  --mtp-weights /absolute/path/qwen38-mtp-mxfp8 \
  --draft mtp-auto \
  --device-index 0 --target gfx1201
```

The Qwen3.8 public profile fixes sampling to temperature `1.0`, `top_p 0.95`,
`top_k 20`, and zero penalties. MTP width remains the fixed width 2; the sidecar changes
the companion matrix representation, not speculative decoding policy.

Supplying a sidecar while disabling MTP is rejected explicitly. For the CLI
and chat command, `--mtp-weights` cannot be combined with
`--mtp-draft-width 0`; for the server it cannot be combined with
`--draft disabled`. Removing `--mtp-weights` restores the BF16 companion and
allows target-only execution. The sidecar option is also rejected for the
generic `--gguf`, `--models`, or library-only server paths.

The authoritative implementation and measurement scope are tracked in the
[Phase84 plan](../plans/archive/2026/09/1-10/phase84-mtp-weight-quantization.md)
and its [measurement history](../history/2026/09/11-20/phase84-mtp-weight-quantization.md).
