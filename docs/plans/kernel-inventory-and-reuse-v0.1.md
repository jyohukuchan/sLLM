# GPU kernel inventory and reuse, v0.1

Date: 2026-07-27.  Scope is source currently reachable from `runtime/src`, including the
SQ8_0/SQ8_1 HIPRTC sources and separately compiled CK, handwritten, FP32-reference, and
MoE HIP files.  This is an inventory, not a benchmark.  “Production” below means the active
`/etc/ullm/served-models/active.json` AQ4 worker, not every compiled feature.

## Answers first

**A. 32 logical kernel entry points are evidenced as launched by the AQ4 production-family
end-to-end trace.**  The trace is the union of the four checked AQ4 e2e profiles at
`benchmarks/results/2026-07-18/qwen35-9b-aq4-production-opt-v0.1/p3/*/raw/e2e-post-wmma_kernel_trace.csv`.
It names the 32 rows marked **P** below.  The active manifest independently selects the same
family (`format_id: "AQ4_0"`, `implementation_id: "qwen35_aq4_rdna4_v1"`, gfx1201, and
`paged_decode_attention.kernel: "aq4_gqa_grouped_split"`, `split_tile: 128`).  This is strong
evidence of the selected family, but it is not a live capture of the currently running process;
the latter is **unverified**.  The count is deliberately the trace's distinct symbols, not
launches per token or macro-generated dtype spellings.

**B. Zero of the four cited wins is inherently tied to one model's shape; all four are
implemented with shape/architecture gates.**  The P3 wave-shuffle + uint4 SQ8 code accepts
`rows` and `cols`, but the fast body is literally `#if defined(__gfx1030__)`; gfx1201 takes the
LDS-tree `#else` body.  GQA co-scheduling is a generic idea, but its fast body requires
`q_per_kv == 4`, `head_dim == 256`, `value_dim == 256`; split-KV/flash has a scalar fallback,
but the grouped fast body has those same constants; `matvec_add` accepts its matrix arguments
but has a compile-time rows-per-block/cache contract.  Thus the useful first work is parameterise
or multi-specialise the dispatch/fast bodies, not replace the algorithms.

**C. The smallest Gemma4 change is a new admission/dispatch descriptor that sends one supported
Gemma4 full-attention geometry to the existing generic split-decode scalar body, with Gemma's
actual `q_heads`, `kv_heads`, dimensions, scale, and no Q-gate.**  No new GPU math is needed for
correctness: that body accepts dimensions as arguments and handles any `head_dim,value_dim <=256`.
It will not select the Qwen grouped fast body.  A complete Gemma4 solution still needs its
local/full window, shared-K/V map, PLE and ties handled by its executor; those are the larger
performance/correctness issues, not prerequisites to call the scalar attention kernel.

**D. The immediate blocker is the Rust executor/dispatch layer, with the loader now a secondary
closed-set gate; it is not primarily that every kernel is Qwen-fixed.**  The prior claim that the
loader reads neither `config.json` nor `architectures` is **refuted for this revision**:
`model_config.rs` reads `model_dir/config.json`, requires exactly one architecture, and recognises
Qwen3, Gemma4, Qwen3.5 dense and Qwen3.5 MoE.  Both `qwen35_aq4_model_runtime.rs` and
`qwen35_moe_aq4_runtime.rs` call `load_model_config_from_package`.  But admission for the fastest
attention ABI is explicitly Qwen3.5/gfx1201/16Q/4KV/256/256/block-256, while Gemma4's executor
has distinct local/full/shared-KV semantics and the MoE runtime rejects the 35B-A3B full layer's
16Q/2KV plus two-channel Q layout.  The SQ8 resident stack is separately fixed to Qwen3-14B;
it is not the active AQ4 worker.

### Shape tally (95 literal source-level entry-point names)

| Classification | Count | Meaning |
|---|---:|---|
| GENERIC | 9 | Elementwise/conversion kernels whose element count or matrix dimensions arrive as arguments. |
| PARAMETRIC | 65 | Correct only within a fixed tile/wave/LDS/format/launch envelope, or has a generic fallback but no generic fast path. |
| QWEN3-FIXED | 21 | A Qwen3/Qwen3.5 layout or the Qwen3-14B SQ8 reference execution is baked into the callable path. |

The 95 is a reproducible logical-name count from:

```sh
rg -n --glob '*.{hip,hip.cpp,cu,cpp,inc}' '__global__' runtime/src \
| perl -ne 'if(/__global__.*?\\b(?:void|__launch_bounds__\\([^)]*\\)\\s+void)\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\(/){print "$1\\n"}' \
| sort -u | nl -ba
```

There are additionally three macro families that emit eight K/V dtype symbols each.  They are
listed as families, not silently omitted: `qwen35_qk_norm_rope_paged_kv_write_typed_*`,
`paged_decode_attn_typed_*`/`split_typed_*`, and `paged_{kv_write,causal_gqa}_chunk_typed_*`.
Expanding them produces 32 physical dtype-specific symbols (the split partial is part of the
second family).  They share a body and classification; treating each spelling as an independent
algorithm would inflate the total without adding a distinct entry point design.  If those three
family rows are included in the displayed-inventory tally, it is **9 GENERIC / 67 PARAMETRIC /
22 QWEN3-FIXED (98 logical rows)**; the 95-row source-name tally above deliberately excludes
the macro-family headings.

## Evidence keys and call path

Paths are evidence locations only; kernels are always identified by name, because runtime source
files are being split concurrently.

* **E1 — manifest and Rust selection.** `/etc/ullm/served-models/active.json` says
  `"kernel": "aq4_gqa_grouped_split"` and `"split_tile": 128`.  `served_model.rs` accepts
  `"aq4_gqa_grouped_split" => split_tile == 128`.
* **E2 — decode call path.** `qwen35_aq4_layer_runtime.rs` resolves
  `HipPagedDecodeAttentionSplit{,SigmoidGate}F32`, calls
  `execute_paged_decode_attention_split_{,sigmoid_gate_}f32` in
  `backend_operation_registry.rs`, then the C ABI
  `paged_decode_attn_split_{,sigmoid_gate_}f32` and the runtime launcher lookup for
  `ullm_paged_decode_attn_split_partial_f32_kernel` plus
  `ullm_paged_decode_attn_split_merge_f32_kernel`.  The source quote for the fast subpath is:
  `if (blockDim.x == 256u && q_per_kv == 4ull && head_dim == 256ull && value_dim == 256ull`.
* **E3 — prefill call path.** The same layer runtime invokes paged KV write and paged causal GQA
  through the operation registry/C ABI.  The WMMA admission guard in
  `backend_operation_registry.rs` is:
  `q_heads != 16 || kv_heads != 4 || head_dim != 256 || value_dim != 256 || block_size != 256`.
* **E4 — projection/primitive path.** `qwen35_aq4_layer_runtime.rs` invokes `ullm_runtime_sys`
  AQ4 matvec, fused QKV, norm, linear-attention, and top-1 C-ABI functions; their HIPRTC module
  lookups are in `ullm_runtime_parts/part_00.inc` and `part_01.inc`.  The active manifest's
  `ULLM_REQUIRE_HIP_*` list is compilation/fail-closed evidence, not launch evidence.
* **E5 — SQ8 fast-gate refutation.** `sq8_0_matvec_hiprtc.inc` contains
  `#if defined(__gfx1030__)` immediately before the `__launch_bounds__(256)` wave32/uint4
  definition, and the legacy `__shared__ float partial[256];` definition under `#else`.
  The active worker is gfx1201.  The generic SQ8 matvec rows are therefore compiled/reachable,
  but are not in the AQ4 trace and are not active-production launches.
* **E6 — typed-family envelope.** The typed split source says
  `if (blockDim.x != 256u || head_dim > 256ull || value_dim > 256ull) return;`.
* **E7 — Qwen3-14B reference.** `sq8_fp32_gpu_reference_gfx1201.hip.cpp` hardcodes
  `constexpr size_t kQHeads = 40u;`, `kKvHeads = 8u;`, and `kHeadDim = 128u;`.
  `ullm-sq8-gpu-fp32-reference` and `sq8_gpu_fp32_reference.rs` own this numerical oracle.
* **E8 — loader correction.** `model_config.rs` says `/// Reads and resolves model_dir/config.json`,
  reads the file with `fs::read`, and rejects an unknown `architectures` value.  Its package helper
  says it refuses to assume Qwen3 when `source_model_dir` is absent.
* **E9 — Gemma/MoE semantic blockers.** `model_config.rs` says
  `Gemma4TextExecutor does not implement attention_k_eq_v=true alternate attention`; its descriptor
  and executor separately model `layer_types`, shared K/V and local/full geometry.  The MoE AQ4
  runtime requires `layer.attention.q_heads != 16 || layer.attention.kv_heads != 2 ||
  layer.attention.head_dim != 256 || layer.attention.value_dim != 256` to reject, and its
  Q projection layout checker describes the Qwen3.5 gated layout.

### Legend

`P` = in the 32-symbol AQ4 e2e trace. `B` = benchmark/probe only. `O` = numerical-oracle/
validation gate, not dead code. `R` = runtime-reachable alternative, but absent from that trace.
`—` = no call site found outside definition/compilation; this is deliberately not called “dead”
without a whole-program link audit.  `E#` cites the quoted evidence above.  **P rows use the
same Gemma/MoE answer unless a cell says otherwise:** Gemma is blocked first by executor semantics
(local/full window, PLE, shared K/V, tied heads/embeddings and scale) and lacks Qwen Q-gate;
MoE is blocked by 16Q×256×2 = 8192 two-channel Q and expert routing.  Where an entry is generic,
that answer means “the kernel could be called, but neither current executor dispatches it.”

## Inventory

| Kernel entry point | Family | Launch/reachability | Shape verdict and code evidence | Gemma4 / 35B-A3B use |
|---|---|---|---|---|
| `add_in_place_f32` | primitive | O — SQ8 GPU FP32 oracle (E7) | GENERIC — `elements` argument | Oracle only |
| `attention_exp_and_sum_f32` | attention | O — SQ8 GPU FP32 oracle | QWEN3-FIXED — E7 fixed Qwen3-14B session geometry | Oracle only |
| `attention_scores_and_max_f32` | attention | O — SQ8 GPU FP32 oracle | QWEN3-FIXED — E7 `kQHeads = 40u` | Oracle only |
| `attention_weighted_values_f32` | attention | O — SQ8 GPU FP32 oracle | QWEN3-FIXED — E7 `kHeadDim = 128u` | Oracle only |
| `bf16_to_f32` | primitive | B/R — CK helper | GENERIC — element count argument | Not selected by AQ4 |
| `bf16_vector_to_f32` | primitive | O — SQ8 GPU FP32 oracle | GENERIC — `elements` argument | Oracle only |
| `copy_kv_f32` | KV | O — SQ8 GPU FP32 oracle | QWEN3-FIXED — E7 `kKvHeads = 8u` | Oracle only |
| `dequant_ocp_block128x128_to_bf16` | primitive | B — gfx942 CK control | PARAMETRIC — `row / 128ull`, `col / 128ull` | CK test only |
| `dequant_ocp_row_k128_to_bf16` | primitive | B — gfx942 CK control | PARAMETRIC — `cols / 128ull` | CK test only |
| `dequant_sq8_ocp_block128_to_f32` | primitive | O — SQ8 GPU FP32 oracle | QWEN3-FIXED — E7 `kFp8Block = 128u` session | Oracle only |
| `embedding_row_bf16_to_f32` | primitive | O — SQ8 GPU FP32 oracle | QWEN3-FIXED — E7 fixed session widths | Oracle only |
| `fnuz_fragment_probe_kernel` | probe | B — gfx942 A-prime probe | PARAMETRIC — CK fragment target | Probe only |
| `mark_nonfinite_f32` | primitive | O — SQ8 GPU FP32 oracle | GENERIC — `elements` argument | Oracle only |
| `quantize_activation_block128` | primitive | B/R — CK gfx1201 | PARAMETRIC — `k / 128u`, launch `dim3(128u)` | SQ8 CK only |
| `rmsnorm_serial_f32` | primitive | O — SQ8 GPU FP32 oracle | QWEN3-FIXED — E7 `kHidden` session contract | Oracle only |
| `rope_split_half_f32` | RoPE | O — SQ8 GPU FP32 oracle | QWEN3-FIXED — E7 `kHeadDim = 128u` | Oracle only |
| `silu_mul_in_place_f32` | primitive | O — SQ8 GPU FP32 oracle | QWEN3-FIXED — fixed `kIntermediate` session | Oracle only |
| `ullm_add_f32_kernel` | primitive | P (E4 trace) | GENERIC — `elements` argument | Could run; executor/layout excludes it |
| `ullm_aq4_dequant_f32_kernel` | AQ4 projection | R/B | GENERIC — `group_size` and `elements` are arguments | AQ4_0 payload required, otherwise reusable |
| `ullm_aq4_gemm_register_bm4_f32_kernel` | AQ4 projection | B/R | PARAMETRIC — register BM4 tile | AQ4 format/dispatch |
| `ullm_aq4_gemm_register_bm8_f32_kernel` | AQ4 projection | P | PARAMETRIC — BM8 register tile | Gemma BF16; MoE needs gated/expert dispatch |
| `ullm_aq4_gemm_register_bm8_group8_f32_kernel` | AQ4 projection | P | PARAMETRIC — group-8 register tile | Same |
| `ullm_aq4_gemm_tiled_f32_kernel` | AQ4 projection | B/R | PARAMETRIC — tiled launch | AQ4 format/dispatch |
| `ullm_aq4_gemm_wmma_group8_prototype_f32_kernel` | AQ4 projection | P | PARAMETRIC — WMMA/group-8 tile | Same |
| `ullm_aq4_gemm_wmma_prototype_f32_kernel` | AQ4 projection | P | PARAMETRIC — WMMA tile | Same |
| `ullm_aq4_gemm_wmma_prototype_v3_f32_kernel` | AQ4 projection | B/R | PARAMETRIC — prototype WMMA tile | Benchmark/prototype only |
| `ullm_aq4_matvec_add_f32_kernel` | AQ4 projection | P | PARAMETRIC — compile-time rows-per-block (`ULLM_AQ4_MATVEC_ADD_RPB`) | Gemma format; MoE routing/layout |
| `ullm_aq4_matvec_batch_f32_kernel` | AQ4 projection | P | PARAMETRIC — batch/register dispatch tile | Gemma format; MoE routing/layout |
| `ullm_aq4_matvec_f32_kernel` | AQ4 projection | P | PARAMETRIC — 256-thread reduction/format | Gemma format; MoE routing/layout |
| `ullm_aq4_matvec_gate_beta_f32_kernel` | AQ4 projection | P | PARAMETRIC — fused gate/beta layout | Gemma lacks Qwen gate; MoE linear-specific layout |
| `ullm_aq4_matvec_pair_f32_kernel` | AQ4 projection | P | PARAMETRIC — two-matrix fused launch | Gemma format; MoE routing/layout |
| `ullm_aq4_matvec_qkv_z_gate_beta_f32_kernel` | AQ4 projection | P | QWEN3-FIXED — two-channel Q/gate-beta fused layout (E9) | Gemma has incompatible Q/K/V semantics; MoE's 8192 Q rows requires its own layout |
| `ullm_aq4_matvec_silu_mul_f32_kernel` | AQ4 projection | P | PARAMETRIC — fused AQ4 gate/up rows-per-block | Gemma uses GELU; MoE has expert MLP routing |
| `ullm_aq4_matvec_top1_f32_kernel` | AQ4 projection | R/B | PARAMETRIC — partial-count/top-1 reduction | Generic concept, AQ4 ABI only |
| `ullm_aq4_matvec_triple_f32_kernel` | AQ4 projection | P | PARAMETRIC — three-matrix fused launch | Gemma format; MoE routing/layout |
| `ullm_aq4_row_f32_kernel` | AQ4 projection | R/B | GENERIC — `rows`, `cols`, `row_index`, `group_size` arguments | AQ4 payload required, otherwise reusable |
| `ullm_bf16_row_f32_kernel` | primitive | P | PARAMETRIC — 256-thread row reduction | Gemma could reuse after dispatch; MoE could too |
| `ullm_cached_prefix_attn_f32_flash2_gqa_grouped_kernel` | attention | B/R | PARAMETRIC — grouped GQA tile | Gemma local/full/shared-KV; MoE Q layout |
| `ullm_cached_prefix_attn_f32_flash2_kernel` | attention | B/R | PARAMETRIC — Flash2 tile/LDS | Executor does not select it |
| `ullm_cached_prefix_attn_f32_kernel` | attention | B/R | PARAMETRIC — block reduction | Executor does not select it |
| `ullm_cached_prefix_attn_fp8_e4m3_flash2_kernel` | attention | B/R | PARAMETRIC — FP8 cache/tile | Different cache/semantics |
| `ullm_cached_prefix_attn_fp8_e4m3_rocwmma_kernel` | attention | B/probe | PARAMETRIC — rocWMMA/GQA ratio gate | Different cache/semantics |
| `ullm_causal_attn_batch_f32_flash2_kernel` | attention | B/R | PARAMETRIC — Flash2 tile | Generic semantic, no production dispatch |
| `ullm_causal_attn_batch_f32_kernel` | attention | B/R | PARAMETRIC — 256-thread block | Generic semantic, no production dispatch |
| `ullm_causal_attn_f32_flash2_kernel` | attention | B/R | PARAMETRIC — Flash2 tile | Generic semantic, no production dispatch |
| `ullm_causal_attn_f32_kernel` | attention | B/R | PARAMETRIC — 256-thread block | Generic semantic, no production dispatch |
| `ullm_decode_attn_f32_kernel` | attention | B/R | PARAMETRIC — one CTA/head reduction | Generic semantic, no production dispatch |
| `ullm_depthwise_conv1d_f32_kernel` | linear-attn | B/R | PARAMETRIC — convolution width/tile | Gemma has no linear-attn; MoE has its own linear state |
| `ullm_linear_attn_gate_beta_f32_kernel` | linear-attn | P | QWEN3-FIXED — Qwen3.5 gate/beta fusion | Gemma no linear-attn; MoE's linear layout differs |
| `ullm_linear_attn_qkv_conv_step_silu_f32_kernel` | linear-attn | B/R | QWEN3-FIXED — fused QKV/SiLU layout | Same |
| `ullm_linear_attn_qkv_prepare_batch_f32_kernel` | linear-attn | P | QWEN3-FIXED — Qwen3.5 QKV preparation layout | Same |
| `ullm_linear_attn_qkv_prepare_batch_update_history_f32_kernel` | linear-attn | P | QWEN3-FIXED — Qwen3.5 history layout | Same |
| `ullm_linear_attn_qkv_prepare_f32_kernel` | linear-attn | P | QWEN3-FIXED — Qwen3.5 QKV preparation layout | Same |
| `ullm_linear_attn_qkv_split_l2norm_f32_kernel` | linear-attn | B/R | QWEN3-FIXED — split QKV/L2-norm layout | Same |
| `ullm_linear_attn_recurrent_f32_kernel` | linear-attn | P | PARAMETRIC — recurrent state dimensions/CTA | Gemma no linear-attn; MoE needs 16/32 head state |
| `ullm_matvec_bf16_f32_kernel` | primitive | R/B | PARAMETRIC — row/CTA reduction | Generic semantic but no AQ4 dispatch |
| `ullm_matvec_f32_kernel` | primitive | R/B | PARAMETRIC — row/CTA reduction | Generic semantic but no AQ4 dispatch |
| `ullm_moe_decode_gemm_f32_kernel` | MoE | R/B — MoE runtime | PARAMETRIC — `kThreads = 256u`, expert-gemm ABI | Gemma no MoE; 35B routes to it, not AQ4 dense path |
| `ullm_moe_gated_silu_f32_kernel` | MoE | R/B | PARAMETRIC — 256-thread gate tile | Same |
| `ullm_moe_gather_f32_kernel` | MoE | R/B | PARAMETRIC — token/expert packing | Same |
| `ullm_moe_prefill_grouped_gemm_f32_kernel` | MoE | R/B | PARAMETRIC — grouped-expert tile | Same |
| `ullm_moe_route_f32_kernel` | MoE | R/B | PARAMETRIC — `kMaxExperts = 256u` | Same |
| `ullm_moe_scatter_weighted_f32_kernel` | MoE | R/B | PARAMETRIC — 256-thread scatter | Same |
| `ullm_moe_sigmoid_gate_f32_kernel` | MoE | R/B | PARAMETRIC — MoE gate ABI | Same |
| `ullm_paged_causal_gqa_chunk_f32_kernel` | attention/KV | P | PARAMETRIC — `blockDim.x != 256u || head_dim > 256ull || value_dim > 256ull` (E6) | Gemma dispatch/semantics; MoE Q layout |
| `ullm_paged_causal_gqa_chunk_wmma_f32_kernel` | attention/KV | P | QWEN3-FIXED — E3 exact `16/4/256/256/256` admission | Gemma differs; MoE is 16Q/2KV and gated Q |
| `ullm_paged_decode_attn_f32_kernel` | attention/KV | P | PARAMETRIC — 256-thread direct reduction | Generic fallback candidate; MoE executor differs |
| `ullm_paged_decode_attn_split_merge_f32_kernel` | attention/KV | P | PARAMETRIC — source tile/workspace ABI | Gemma can use scalar split after dispatch; MoE differs |
| `ullm_paged_decode_attn_split_partial_f32_kernel` | attention/KV | P | PARAMETRIC — E2/E6; generic scalar fallback but fast 4×256 GQA | Gemma scalar only; MoE 16/2 and gate layout |
| `ullm_paged_kv_write_chunk_f32_kernel` | KV | P | PARAMETRIC — 256-thread typed KV payload tile | Gemma cache policy; MoE layout |
| `ullm_paged_kv_write_f32_kernel` | KV | P | PARAMETRIC — paged payload/block ABI | Gemma cache policy; MoE layout |
| `ullm_qwen35_qk_norm_rope_batch_f32_kernel` | RoPE/KV | P | QWEN3-FIXED — Qwen3.5 fused Q/K norm + gated-Q layout | Gemma distinct RoPE/norm; MoE mRoPE/layout |
| `ullm_qwen35_qk_norm_rope_f32_kernel` | RoPE/KV | R/B | QWEN3-FIXED — Qwen3.5 fused layout | Same |
| `ullm_qwen35_qk_norm_rope_paged_kv_write_f32_kernel` | RoPE/KV | P | QWEN3-FIXED — Qwen3.5 fused layout/cache write | Same |
| `ullm_qwen35_split_q_gate_f32_kernel` | RoPE/KV | R/B | QWEN3-FIXED — Q projection is split into Q/gate halves | Gemma has no Qwen gate; MoE uses two-channel Q |
| `ullm_rmsnorm_f32_kernel` | primitive | R/B | PARAMETRIC — CTA reduction | Gemma norm convention/dispatch; MoE can reuse mechanically |
| `ullm_rocwmma_fp8_attn_probe_kernel` | probe | B | PARAMETRIC — rocWMMA target | Probe only |
| `ullm_rocwmma_fp8_qk_probe_kernel` | probe | B | PARAMETRIC — rocWMMA target | Probe only |
| `ullm_rope_f32_kernel` | RoPE | R/B | PARAMETRIC — rotary/tile contract | Gemma needs its own RoPE semantics |
| `ullm_segmented_rmsnorm_f32_kernel` | primitive | P | PARAMETRIC — `partial[256]` reduction | Gemma possible after executor work; MoE possible |
| `ullm_segmented_rmsnorm_silu_mul_f32_kernel` | primitive | P | PARAMETRIC — segmented layout/SiLU | Gemma GELU; MoE expert MLP |
| `ullm_sigmoid_mul_f32_kernel` | primitive | R/B | GENERIC — `elements` argument | Could run; Gemma does not use Qwen gate |
| `ullm_silu_mul_f32_kernel` | primitive | P | GENERIC — `elements` argument | Gemma GELU; MoE may use gated SiLU |
| `ullm_sq8_1_matvec_w8a16_f32_kernel` | primitive | B/R — SQ8_1 runtime | PARAMETRIC — `__launch_bounds__(256)`/wave32 | SQ8_1 only |
| `ullm_sq8_1_matvec_w8a8_explicit_f32_kernel` | primitive | B/R — SQ8_1 runtime | PARAMETRIC — `__launch_bounds__(256)`/wave32 | SQ8_1 only |
| `ullm_sq8_1_matvec_w8a8_explicit_fallback_f32_kernel` | primitive | B/R — SQ8_1 runtime | PARAMETRIC — `__launch_bounds__(256)` fallback | SQ8_1 only |
| `ullm_sq8_handwritten_gfx1201_m1_wmma_kernel` | primitive | B/R — SQ8 handwritten | PARAMETRIC — `kWmmaTile = 16u`, M=1 | SQ8-only/probe path |
| `ullm_sq_fp8_matvec_batch_f32_kernel` | primitive | R/B, **not P** (E5) | PARAMETRIC — fast `#if defined(__gfx1030__)`; otherwise LDS tree | Not active AQ4; generic SQ8 input shapes |
| `ullm_sq_fp8_matvec_f32_kernel` | primitive | R/B, **not P** (E5) | PARAMETRIC — same gfx1030 gate | Not active AQ4; generic SQ8 input shapes |
| `ullm_sq_fp8_matvec_pair_f32_kernel` | primitive | R/B, **not P** | PARAMETRIC — 256-thread reduction | Not active AQ4 |
| `ullm_sq_fp8_matvec_triple_f32_kernel` | primitive | R/B, **not P** | PARAMETRIC — 256-thread reduction | Not active AQ4 |
| `ullm_top1_f32_kernel` | primitive | P | PARAMETRIC — top-1 partial reduction | Could reuse after executor/format change |
| `ullm_top1_pairs_f32_kernel` | primitive | R/B | PARAMETRIC — paired top-1 reduction | Could reuse after executor/format change |
| `ullm_wmma_fp8_probe_kernel` | probe | B | PARAMETRIC — WMMA target | Probe only |
| `ullm_wmma_fp8_qk_probe_kernel` | probe | B | PARAMETRIC — WMMA target | Probe only |
| `qwen35_qk_norm_rope_paged_kv_write_typed_*` (8) | RoPE/KV | R — typed C ABI | QWEN3-FIXED — macro calls Qwen3.5 fused implementation; dtype is compile-time | Gemma semantics; MoE mRoPE/two-channel Q |
| `paged_decode_attn_typed_*` and `paged_decode_attn_split_typed_*` (16) | attention/KV | R — typed C ABI | PARAMETRIC — E2/E6, fast branch is 4Q/256 | Gemma scalar path possible after dispatch; MoE layout |
| `paged_kv_write_chunk_typed_*` and `paged_causal_gqa_chunk_typed_*` (16) | KV/attention | R — typed C ABI | PARAMETRIC — `blockDim.x != 256u || head_dim > 256ull || value_dim > 256ull` | Gemma cache/executor; MoE layout |

## What the inventory says about the hypothesis

The hypothesis is only partly confirmed.  It is false that the fast-kernel source is uniformly
hardcoded to one model: projection dimensions, cache dimensions and most attention dimensions are
ABI arguments, and the split attention body contains a scalar fallback.  It is true that the
*admission to the fast implementations* is fixed to Qwen3.5 geometry and that several fused
linear-attention/RoPE layouts are genuinely Qwen-specific.  The more consequential lock-in is
above the kernels:

1. The AQ4 dispatch has exact geometry/feature guards (E2/E3), so changing model geometry loses
   the fast branch even where a correct scalar body exists.
2. Gemma4's executor needs local/full scheduling, shared-K/V ownership, PLE and its norm/RoPE
   conventions.  Merely changing a kernel's `head_dim` constant would silently mis-execute it.
3. The 35B-A3B MoE requires both 8192-row two-channel Q handling and expert routing; its seven
   MoE kernels are a separate reachable family, not a route to the dense AQ4 projection path.
4. SQ8's resident Qwen3-14B runtime is fixed, but this does not explain the active Qwen3.5 AQ4
   worker's behaviour.  It is evidence against importing SQ8 performance conclusions wholesale.

### Reproducibility / limitations

No GPU workload was run.  The `P` evidence is a stored ROCprof trace, and the active worker binary
is an immutable release external to the worktree.  To settle the remaining live-release uncertainty,
capture a read-only kernel-name trace of one production prefill and one decode under the active
manifest and compare its unique symbols with the 32 `P` rows.  Do not infer a launch from a
`ULLM_REQUIRE_HIP_*` environment guard or from `hipModuleGetFunction` alone.
