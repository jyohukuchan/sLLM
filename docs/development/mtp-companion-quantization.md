# Qwen3.8 MTP companion quantization

This document describes the reviewed quantized MTP companion for the reviewed
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

The converter supports these packed sidecar recipes:

| encoding | value/activation | scale and accumulation |
| --- | --- | --- |
| MXFP8 | E4M3 W8A8 | K-axis block 32, E8M0 scale, FP32 accumulation, BF16 output |
| NVFP4 | E2M1 W4A4 | K-axis block 16, E4M3FN block scale, FP32 weight tensor scale and calibrated activation scale, FP32 accumulation, BF16 output |

Phase 87 WU-3S retired the MXFP6 **MTP companion** sidecar. Previously
generated packed MXFP6 sidecars are rejected with an explicit retired-format
error; no other companion encoding is selected as a fallback. This retirement
does not remove MXFP6 body-model support or its shared low-precision kernel.

Phase 87 Stage 3 adds NVFP4 as the reviewed default sidecar. Its weight planes are made
by sLLM's `quantize_nvfp4_weights` from the eight BF16 source matrices.
Activation scale `g` is calibrated from BF16 companion inputs on six fixed
English, Japanese, and Chinese prompts that do not overlap the
`mtp-bench-v1` evaluation inputs. The five input sites are the fusion input,
attention Q/K/V input, attention output projection input, MLP gate/up input,
and MLP down input. Q/K/V share one `g`, as do gate/up. The sidecar stores
the **resident** value `g = f32(amax / (6 × 448))`; no reciprocal is applied
when it is uploaded. The source Unsloth artifact's `input_global_scale`
uses a different, reciprocal convention and is not used to calibrate this
companion.

For the Phase85 activation-error ablation, it also accepts
`--encoding bf16-roundtrip-mxfp8`. This diagnostic recipe quantizes the same
eight matrices, dequantizes their values, and stores them as BF16. The runtime consequently uses the existing
BF16 matrix and activation path. This isolates weight quantization from the
packed MX activation path; it is not an implementation of a packed W8A16 or
W6A16 kernel, and its BF16 execution also applies during prefix priming.

Each diagnostic tensor records the all-element BF16 roundtrip result,
nonfinite values, underflow, overflow, bit mismatches, and the first affected
positions. The historical MXFP6 roundtrip recipe is retired along with the
packed MXFP6 companion. Shared embedding/head and norms are still excluded.
This diagnostic recipe does not change the production companion selection; it
uses the ordinary BF16 matrix path for its comparison.

The former MXFP8 W8A16, MXFP6 W6A16, and NVFP4 W4A16 execution paths were
removed in [Phase 87 Stage 4](../history/2026/09/21-30/phase87-stage4.md).
The BF16 roundtrip diagnostic sidecar above still uses the ordinary BF16
companion matrix path; they do not restore those removed low-precision A16
kernels. The historical A16 experiment is recorded in the
[Phase85 plan](../plans/archive/2026/09/11-20/phase85-m1-a16-mtp.md).

For the bounded Phase85 benchmark, `SLLM_PHASE85_MTP_PRIMING_TIMING=1`
records per-call prefix row counts and host wall times. It is disabled by
default and does not change arithmetic or kernel routing. The final ABBA
comparison found practically equal prefix time; the earlier small positive
difference remains in the experiment history.

When Qwen3.8 MTP is enabled and `--mtp-weights` is omitted, the production
shared backend resolves the reviewed NVFP4 sidecar at
`<artifact_root>/.sllm/mtp-nvfp4-v1/`. It validates the NVFP4 encoding,
artifact/model-lock binding, and combined recipe digest
`sha256:d9698c41954ef7b53a2937c0f662ac2a273f1bdc40c602f77d4928b63de991e1`.
Missing, corrupt, or mismatched sidecars fail clearly without selecting BF16 or
another encoding as a fallback. An explicit `--mtp-weights` directory still
overrides this default. MTP disabled is unaffected.

The Qwen3.8 KV default is separately standard OCP MXFP8 E4;
`--kv-cache-encoding fp16` is its explicit rollback and does not select MTP
weights. BF16 remains the benchmark control and source/binary rollback path.

MXFP8 is integrated into the normal loader, graph, resident model, CLI, chat,
and server paths. Phase 87 Stage 3 compared BF16, MXFP6, and NVFP4 on the same
`mtp-bench-v1` Tier A 26 conditions and both exact GPUs. NVFP4's expected
p/q acceptance was 0.80–1.01 percentage points below MXFP6 and 1.08–1.37
points below BF16 across the two frozen token streams. The 5-point rule for
requiring MXFP6 companion support was not triggered. In the completed WU-3S, NVFP4 improved
the acceptance-adjusted M4 by 1.61–1.76% across both GPUs and frozen streams,
but normal whole-graph AB/BA decode TPOT regressed by 0.48–0.63% on V620 and
13.27–13.28% on R9700. The common adoption rules would have retained BF16,
but the user explicitly instructed this Stage 3 decision to ignore those rules
and make NVFP4 the default. The one-time prefill exception and accepted
trade-off are in
[Phase 87 WU-3S](../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#段階3-mtp-companionの形式).
See the [Stage 3 evidence](../history/2026/09/21-30/phase87-stage3.md).
The [WU-3P follow-up](../history/2026/09/21-30/phase87-mtp-nvfp4-prefix-prefill.md)
then reduced the NVFP4 MTP prefix preparation from 7.527 to 0.287 seconds on
V620 and from 6.018 to 0.201 seconds on R9700 for coding8192/128. Normal
prefill fell from 43.515 to 36.198 and 20.746 to 14.916 seconds, respectively.
Both exact GPUs passed earlier bounded operator and public API correctness
smoke for MXFP6; those are historical results for the retired MTP sidecar.

## Creating a sidecar

The converter reads the verified source artifact and the embedded reviewed
model lock. Generated sidecars are local artifacts and must not be committed:

```bash
cargo run --release -p sllm-cli --bin sllm-convert-qwen38-mtp -- \
  --artifact-root /home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-NVFP4 \
  --encoding mxfp8 \
  --output-dir /home/homelab1/coding-local/sLLM/.local-artifacts/phase84/mxfp8

```

`--artifact-root` and `--output-dir` must be absolute, and the output
directory must not already exist. Each successful conversion publishes exactly
`manifest.json` and `payload.safetensors` after conversion and readback
verification. To regenerate an existing sidecar, choose a new output
directory or remove the old generated directory after confirming it is not in
use.

For NVFP4, a held-out activation-scale manifest is mandatory. Its five BF16
input maxima, eight derived scales, GPU report digests, and calibration input
manifest digest are checked before conversion. Generate it with
`ci/tools/phase87_stage3_calibrate_nvfp4.py`, then pass the absolute path:

```bash
cargo run --release -p sllm-cli --bin sllm-convert-qwen38-mtp -- \
  --artifact-root /home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-NVFP4 \
  --encoding nvfp4 \
  --activation-scale-manifest /absolute/path/activation-scales.json \
  --output-dir /home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-NVFP4/.sllm/mtp-nvfp4-v1
```

The converter output includes the encoding, a manifest fingerprint, and a
combined recipe digest. The measured local sidecar identities were:

| sidecar | manifest fingerprint | combined recipe digest |
| --- | --- | --- |
| MXFP8 | `sha256:5a75528a271fed2856ca521a61403d2e044f8861946ebd68fd3863a784e01470` | `sha256:6ef86986eb689c460c397a79b714f2193bc1f3da999ed7c01495b68c1e235858` |
| MXFP6 (retired MTP sidecar; historical) | `sha256:103b03b0642b238d74c99ca0c38201fcef882e4887f1719bdfb6569e7acd74bc` | `sha256:f85ff503a6e807e34872b6bd93ac16a33c9f2cbfe1f937992f0d87ba4fc5722a` |
| NVFP4 (Phase 87 Stage 3) | `sha256:b6334312e4d64404d7a31618005a707d317a8ba5ff53a8feccb639c2cbbd32c4` | `sha256:d9698c41954ef7b53a2937c0f662ac2a273f1bdc40c602f77d4928b63de991e1` |

The reviewed production copy is installed as
`<artifact_root>/.sllm/mtp-nvfp4-v1/`. The generated local payload is about
228 MB and is not tracked in Git.

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

Use `/absolute/path/qwen38-mtp-nvfp4` as `--mtp-weights` for the calibrated
NVFP4 companion. The same directory option applies to chat and server mode.

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
`top_k 20`, and zero penalties. MTP width defaults to 2; width 3 or 4 can be
selected explicitly with `--mtp-draft-width`. The sidecar changes the companion
matrix representation independently of the selected width.

Supplying a sidecar while disabling MTP is rejected explicitly. For the CLI
and chat command, `--mtp-weights` cannot be combined with
`--mtp-draft-width 0`; for the server it cannot be combined with
`--draft disabled`. Removing `--mtp-weights` selects the default NVFP4 sidecar
described above (the BF16 companion is no longer selectable at run time; it
remains the benchmark control and the source/binary rollback path), and
disabling MTP allows target-only execution. The sidecar option is also rejected for the
generic `--gguf`, `--models`, or library-only server paths.

The authoritative implementation and measurement scope are tracked in the
[Phase84 plan](../plans/archive/2026/09/1-10/phase84-mtp-weight-quantization.md)
and its [measurement history](../history/2026/09/11-20/phase84-mtp-weight-quantization.md).
