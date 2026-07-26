# Qwen3.5-35B-A3B MoE runtime foundation v0.1

> Status: implemented baseline design. 2026-07-26 JST.
>
> Scope: loader-independent routing, gather/scatter, grouped GEMM, and shared-expert
> execution primitives. This document deliberately does not claim that the complete
> Qwen3.5 hybrid-attention model is executable yet.

## Evidence and source contract

The source checkpoint was read directly at
`/home/homelab1/datapool/ai_models/safetensors/Qwen3.5-35B-A3B-BF16`.
Its `config.json` SHA-256 is
`5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944`.
The observed top-level architecture is
`Qwen3_5MoeForConditionalGeneration`; its text config is
`qwen3_5_moe_text`.

| Contract item | Observed value |
| --- | --- |
| text hidden size / decoder layers | 2048 / 40 |
| experts / selected experts per token | 256 / 8 |
| routed expert intermediate width | 512 |
| shared expert intermediate width | 512 |
| activation | SiLU |
| expert `gate_up_proj` | BF16 `[256, 1024, 2048]` |
| expert `down_proj` | BF16 `[256, 2048, 512]` |
| router `mlp.gate.weight` | BF16 `[256, 2048]` |
| shared gate/up/down | separate BF16 `[512,2048]`, `[512,2048]`, `[2048,512]` |
| shared-expert gate | BF16 `[1,2048]` |
| dense/MoE mix in text decoder | no dense replacement layers observed: all 40 text layers have the 3-D routed tensors and all shared tensors |

The source’s MTP and vision tensors are outside this text-decoder executor
scope. `mtp.layers.0` has a different, per-expert tensor naming/layout and is
not silently treated as a text decoder layer.

The installed local Transformers 5.12.1 source defines routing as:

1. `router_logits = linear(hidden, gate.weight)`;
2. `softmax(router_logits, dtype=float32)`;
3. `torch.topk(..., k=8)`;
4. divide the eight selected probabilities by their sum;
5. cast the selected values back to the router-logit dtype;
6. evaluate each selected expert and add the weighted results;
7. add `sigmoid(shared_expert_gate(hidden)) * shared_expert(hidden)`.

The router weight is not normalized before this softmax. The selected weights
are normalized **after** top-k, not before it.

`torch.topk` documents no stable tie ordering. On this host’s PyTorch
2.12.0+cpu, equal 256-way inputs happened to return
`[172,169,170,171,174,175,168,173]` for `k=8`, while smaller equal inputs
gave different shape-dependent orders. That observation is not an executable
model contract. The baseline exposes a per-token boundary-tie flag and its
full forward path rejects a tie at the kth probability rather than inventing
a different semantic order. The real-weight HF fixture has no boundary tie.

## Runtime ABI and tensor layout

The baseline is intentionally separate from package loading. It accepts
resident runtime buffers and a `F32` or raw IEEE BF16 weight dtype. All
activation, routing-score, and output buffers are F32; BF16 source weights are
read without a mandatory F32 model-wide expansion. For a BF16 router, the
baseline explicitly round-trips the F32 activation and resulting logit to BF16
before HF's F32 softmax, then round-trips the normalized selected score to
BF16. This reproduces the observed HF `F.linear(BF16,BF16)` / router boundary
while preserving F32 buffer ownership in the runtime.

The primitive ABI is split into these operations:

| Primitive | Inputs | Outputs |
| --- | --- | --- |
| `moe_route` | `[M,H]` hidden, `[E,H]` router | `[M,K]` scores, `[M,K]` expert IDs, `[M]` boundary-tie flags |
| `moe_gather` | `[M,H]`, `K` | `[M*K,H]` assignment-major activations |
| `moe_decode_gemm` | one token's `[K,C]` selected activations, selected IDs, `[E,R,C]` weight | `[K,R]` |
| `moe_grouped_gemm` | prefill assignment-major activations, IDs, `[E,R,C]` weight | `[M*K,R]` |
| `moe_gated_silu` | `[M*K,2I]` | `[M*K,I]` |
| `moe_scatter_weighted` | `[M*K,H]`, scores | `[M,H]` routed reduction |
| `moe_sigmoid_gate` | `[M]` shared gates, `[M,H]` shared output | `[M,H]` |

The layout exactly preserves the safetensors row-major layout:
`expert * rows_per_expert * cols + row * cols + col`. Thus routed gate/up
uses `R=2I=1024, C=H=2048`, and routed down uses `R=H=2048, C=I=512`.
The shared expert uses `E=1`: on decode it uses the decode GEMM path; on
prefill it uses the grouped path. It does not need a dense-only implementation.

## Data flow

```text
hidden[M,H]
  | router GEMV + F32 softmax/top-k/renormalize
  +--> expert_id[M,K], route_weight[M,K], tie_flag[M]
  | gather (token repeated K times)
  +--> assignment_hidden[M*K,H]
       | grouped gate/up GEMM [E,2I,H]
       +--> gate_up[M*K,2I] -- SiLU(gate) * up --> active[M*K,I]
            | grouped down GEMM [E,H,I]
            +--> expert_out[M*K,H] -- weighted scatter/reduce --> routed[M,H]

hidden[M,H] -- shared gate/up/down MLP --> shared[M,H]
hidden[M,H] -- shared gate projection --> shared_gate[M]
sigmoid(shared_gate) * shared + routed --> output[M,H]
```

The CPU reference composes the same materialized stages. The GPU baseline has
separate decode and prefill GEMM launch boundaries plus simple kernels for the
other primitives. Prefill retains assignment-major rows for direct reference
comparison; correctness and inspectability take priority over tiling
efficiency. It therefore has seven correctness kernels: route, activation
gather, decode GEMM, prefill grouped GEMM, gated-SiLU, weighted scatter, and
shared sigmoid gate.

## Decode and prefill are distinct plans

### Decode (`M=1`)

There are exactly eight routed assignments plus one shared expert. The
baseline has a separate `moe_decode_gemm` ABI and HIP kernel, which accepts
only the selected expert IDs and `K` activation rows; it is never dispatched
through the prefill grouped kernel. Physical collection of the eight BF16
expert slabs belongs to the later residency layer: this correctness substrate
indexes the selected slabs in a provided `[E,R,C]` buffer and deliberately
does not invent an offload/copy policy. The complete routed reservoir is 1.5
GiB per text layer, while the eight selected routed slabs are 48 MiB per token
(`8 * (1024*2048 + 2048*512) * 2`); the shared gate/up/down adds about 6 MiB.
Thus even decode's approximately 54 MiB of selected MLP weight traffic per
layer is bandwidth-dominated, not arithmetic-dominated. A later residency
implementation must physically gather those slabs before the decode GEMMs;
it must remain separate from prefill’s grouped plan.

### Prefill (`M=N`)

There are `N*8` assignments. The baseline sorts neither assignments nor
weights: `expert_id[a]` selects its group inside a correctness-first grouped
GEMM. A production prefill specialization should histogram/prefix-sum by
expert, permute assignment rows into contiguous expert groups, invoke an
expert-grouped GEMM for each nonempty group, then unpermute/scatter. This is
the path that needs variable-M grouped GEMM tuning; it must not be folded into
the decode gather path.

## Workspace budget

For F32 activations the baseline’s per-layer transient allocation is:

| Buffer | Formula | Decode `M=1` | Prefill example `M=128` |
| --- | ---: | ---: | ---: |
| router logits (diagnostic only) | `M*E*4` | 1 KiB | 128 KiB |
| IDs + scores + flags | `M*K*(4+4)+M*4` | 68 B | 8.5 KiB |
| gathered hidden | `M*K*H*4` | 64 KiB | 8 MiB |
| gate/up | `M*K*(2I)*4` | 32 KiB | 4 MiB |
| activated intermediate | `M*K*I*4` | 16 KiB | 2 MiB |
| per-assignment down output | `M*K*H*4` | 64 KiB | 8 MiB |
| routed + shared + final outputs | `3*M*H*4` | 24 KiB | 3 MiB |

The M=128 total excluding optional logits is about 25 MiB. It is negligible
relative to the uncompressed weights; a future resident implementation still
needs a separately budgeted KV cache and attention workspace.

## Actual R9700 residency decision

`amd-smi` reports the permitted gfx1201/R9700 as 34,208,743,424 bytes
(31.859 GiB) VRAM. A safetensors-header audit gives:

| Scope | Raw BF16 / source bytes | Difference from R9700 VRAM |
| --- | ---: | ---: |
| text decoder | 68,304,112,256 B / 63.613 GiB | short 31.754 GiB |
| routed + shared expert weights | 64,676,331,520 B / 60.235 GiB | short 28.375 GiB |
| complete checkpoint (text + vision + MTP + other) | 71,903,655,008 B / 66.965 GiB | short 35.106 GiB |

Consequently a full raw-BF16 resident 35B inference cannot be attempted on
this R9700. This task does not silently introduce quantization, CPU offload,
or partial-layer generation as a substitute. It verifies the primitives on
small resident slices and leaves a BF16-aware streaming/offload or quantized
weight policy as an explicit later integration decision.

## Non-goals and promotion boundary

- No changes are made to `AQ4_0`, `SQ8_0`, attention kernels, or the active
  serving manifest.
- This is not a Qwen3.5 full-model executor: hybrid linear/full attention,
  mRoPE, Q output gating, model loading, tokenizer/vision handling, MTP, and
  KV state remain separate work.
- The verification uses the lightweight HF trace machinery and dedicated
  routing fixtures; it does not invoke FP32 reference corpus, campaign, or
  bitwise-promotion gates.

## Implemented baseline and evidence

The public runtime C ABI now has loader-independent `moe_route`, `moe_gather`,
`moe_decode_gemm`, `moe_grouped_gemm`, `moe_gated_silu`, `moe_scatter_weighted`, and
`moe_sigmoid_gate` operations. Their CPU implementation is the reference;
the optional `rocm-moe-gfx1201` feature compiles simple static HIP kernels
only for the permitted gfx1201 device. The GPU path fails closed when that
feature is absent or the selected device is not gfx1201.

`moe_runtime_verify` exercises every materialized stage on both F32 and raw
BF16 weight storage through two synthetic paths: prefill (`M=5`) uses
`moe_grouped_gemm`, and decode (`M=1`) uses only `moe_decode_gemm`. CPU C ABI
versus the Rust reference is bit-identical for all stages in both paths. On
the R9700, each path's final-output maximum absolute difference was
`2.384185791e-7`; the largest raw-BF16-stage difference was
`3.576278687e-7` (prefill shared gate/up). This is a correctness check, not a
timing measurement.

For the real layer-0 router fixture, generated directly with the installed
HF `Qwen3_5MoeTopKRouter`, three BF16 inputs selected:

```text
[52, 148, 101, 178, 151, 128, 116, 166]
[171, 96, 196, 226, 76, 123, 80, 117]
[23, 225, 71, 21, 250, 163, 84, 102]
```

Both CPU and R9700 runtime routing matched those IDs and the BF16-normalized
scores exactly (`max_abs = 0`). An exact-tie fixture is separately flagged and
not claimed as an HF ordering match. The compact result reports are under
`benchmarks/results/2026-07-26/qwen35-moe-runtime-v0.1/`; raw fixture tensors
remain local because they duplicate checkpoint weights.

The same fixture generator captures two compact real 3-D layer-0
`gate_up_proj` slices. The prefill slice uses source experts `[52,148]`, raw
BF16 shape `[2,37,71]`, and deliberately reordered local IDs `[1,0,1]`.
The decode slice uses the real first-token top-8 source experts
`[52,148,101,178,151,128,116,166]`, raw BF16 `[8,37,71]`, and local IDs
`[0,1,2,3,4,5,6,7]`. In both cases HF F32 expected values, the raw-BF16 CPU
reference, CPU C ABI, and the respective R9700 GEMM kernel agreed exactly
(`max_abs = 0`). This validates expert-axis/row/column addressing without
checking a copied full 1.5-GiB expert reservoir into source control.

`tools/architecture_hf_trace.py self-test` was also run unchanged and rejected
its deliberate layer-3 corruption at `step-0000__layer-0003`. A full 35B HF
capture was not launched on this shared host: the raw checkpoint needs
66.965 GiB before framework overhead, while the measured host memory available
at validation time was about 43 GiB. There is no complete uLLM hybrid-attention
candidate to compare yet in any case; the direct HF router fixture is the
numeric source for this loader-independent layer of the work.
