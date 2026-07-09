# SQ8_0 Implementation Plan v0.1

## 前回の要点

- `FQ8_0` は会話上のtypoであり、正式なpublic format IDは `SQ8_0` とする。
- `SQ8_0` はFP8 E4M3を使う正式採用formatであり、FP8そのものの品質採否候補ではない。
- 旧 `sq-fp8-w8a16-r9700-v0` はpublic format IDではなく、`SQ8_0` のimplementation lineage / legacy aliasとして扱う。
- 既存実装では、artifact生成、manifest検証、row / row-block scale、SQ FP8 direct matvec、batch、pair、triple、selected-layer model-loop overlayまで進んでいる。
- 旧T2のstrict top1 / top-k / text guardは、今後は採用gateではなく実装回帰診断として扱う。

## 今回の変更点

- 旧FP8 SQ候補計画を、`SQ8_0` の正式実装計画として再定義する。
- 実装対象、非対象、runtime境界、backend dispatch方針、C++ kernel構成、回帰検証、完了条件を固定する。
- 直近の実装順を、既存部品を壊さずに進められる小さい単位へ分解する。

## 次の行動

1. 旧CLI名とartifact schemaを互換維持しながら、表示・結果・docs上のpublic format表記を `SQ8_0` へ寄せる。
2. CLI-localのSQ FP8 resident overlay処理をlibrary moduleへ移し、package/model-loop runtimeから再利用できる境界にする。
3. `backend_dispatch` を実runtime pathへ接続し、まずSQ8_0 matvec系とcached-prefix attention executor選択に使う。

## Objective

`SQ8_0` を、Qwen3/Qwen3.5系LM weightのFP8 E4M3実行formatとして実装する。

ここでの「実装完了」は、FP8の一般的な品質を証明することではない。次の条件を満たすことを意味する。

- `SQ8_0` artifactを仕様どおり生成できる。
- runtimeがartifactを仕様どおり読み、resident payload / scaleを保持できる。
- 対応GPUでは、F32 materialize fallbackではなくSQ8_0 direct pathで主要projectionを実行できる。
- model-loop / batch throughput / prompt-suite回帰診断で、実装ミスを検出できる。
- 結果schemaではpublic format ID `SQ8_0` とimplementation lineageを分けて記録できる。

## Naming And Compatibility

| item | rule |
| --- | --- |
| public format ID | `SQ8_0` |
| legacy aliases | `sq`, `sq-format-v0.1`, `sq-fp8*` |
| current implementation lineage | `sq-fp8-w8a16-r9700-v0` |
| initial payload dtype | `fp8_e4m3` |
| initial activation contract | W8A16-style: FP8 weights, BF16/F32 activations depending on runtime boundary |
| initial scale dtype | `f32` |
| initial scale layouts | `tensor`, `row`, `row_block`, and mixed per-tensor policy |

Compatibility rule:

- Readers must accept old `sq-fp8*` IDs.
- New artifacts and result rows should emit `candidate.id = "SQ8_0"` and `candidate.format_id = "SQ8_0"`.
- If a legacy implementation ID is used, it must be preserved as `candidate.implementation_id` or `legacy_*` metadata.

## Scope

Initial target:

- model family: Qwen3/Qwen3.5 decoder-only LM path;
- tensor kind: language-model 2D weight tensors;
- GPU target: R9700/RDNA4 first;
- runtime path: package-backed inference, selected-layer and model-loop smoke, then full-package throughput;
- storage: sidecar SQ8_0 artifact first, later integrated `.ullm.d` package format.

Initial tensor families:

- embedding and lm_head;
- self-attention `q/k/v/o`;
- Qwen3.5 self-attention output gate support when present;
- linear-attention projections when present;
- MLP `gate/up/down`.

Deferred:

- visual tower and MTP tensors;
- tensor parallel layout;
- full server API;
- tokenizer integration beyond existing package prompt tools;
- RDNA2/V620 native FP8 optimization. RDNA2 may use fallback/dequant paths, but it is not a blocker for R9700/RDNA4 `SQ8_0` implementation.

## Current Implementation State

Done:

- `AQ4_0` / `SQ8_0` format ID canonicalization and legacy alias handling.
- SQ FP8 artifact builder emits `SQ8_0` public metadata.
- Artifact manifest supports FP8 payload files, F32 scales, row-block scale, mixed policy metadata, and passthrough tensors.
- Runtime artifact parsing and selected tensor materialization smoke exist.
- SQ FP8 runtime APIs exist for:
  - single matvec;
  - batch matvec;
  - pair matvec;
  - triple matvec.
- `PackageAq4ResidentMatvec` has an `SqFp8` storage variant in the CLI smoke/runtime harness path.
- SQ8_0 selected-layer model-loop bridge exists through `sq-fp8-token-ids-model-loop-smoke`.
- R9700 cached-prefix FP8 KV cache and FlashAttention2-style executor experiments exist.
- RDNA4 FP8 WMMA probes exist.

Partial:

- SQ8_0 resident overlay logic is still too CLI-local.
- Runtime result labels still use many `sq-fp8-*` command names.
- Direct SQ8_0 execution is available at projection API boundaries, but not all higher-level fused layer boundaries are SQ8_0-aware.
- `backend_dispatch` exists as a typed selector, but is not yet wired into real runtime operations.
- C++ runtime still has broad `.inc` chunks rather than explicit model/GPU/format implementation modules.

Not done:

- Integrated SQ8_0 package format inside `.ullm.d`.
- Full-package SQ8_0 real-batch throughput row with artifact load/materialization accounting.
- C++ implementation registry that can choose kernels by model architecture, GPU architecture, GPU name, phase, and format.
- SQ8_0 fused MLP / attention-specific direct kernels beyond matvec pair/triple boundaries.
- Stable regression suite named around `SQ8_0` rather than old SQ FP8 candidate language.

## Runtime Architecture

### Artifact Boundary

`sq_manifest.json` remains the source of truth for v0.1.

Required fields:

- public candidate ID and format ID;
- implementation ID;
- payload dtype;
- scale dtype and layout;
- tensor entries with shape, family, payload file, scale file, byte counts, and checksums;
- passthrough entries with explicit reasons;
- storage estimates;
- optional policy block.

Implementation rule:

- The runtime must not infer scale layout from file naming.
- Per-tensor manifest entries are authoritative.
- Row-block scale must validate `scale_block_cols > 0` and expected scale element count.

### Rust Loader Boundary

Create or extend a library-side SQ8_0 runtime module so CLI smokes stop owning the core behavior.

Target module options:

- `crates/ullm-engine/src/sq_runtime.rs`, or
- an expanded `crates/ullm-engine/src/qwen3_loader.rs` boundary if the behavior is Qwen-specific.

The preferred split is:

- `sq.rs`: artifact schema, manifest validation, FP8 decode helpers, row selection utilities.
- `sq_runtime.rs`: resident payload/scale loading, runtime buffer registration, SQ8_0 matvec storage references.
- `qwen3_loader.rs`: mapping model tensor names to Qwen3/Qwen3.5 layer runtime weights and SQ8_0 overlay application.

### Runtime Kernel Boundary

Keep these C ABI functions as the first stable SQ8_0 W8A16 surface:

- `ullm_runtime_sq_fp8_matvec_f32`
- `ullm_runtime_sq_fp8_matvec_batch_f32`
- `ullm_runtime_sq_fp8_matvec_pair_f32`
- `ullm_runtime_sq_fp8_matvec_triple_f32`

Next kernel boundaries:

- SQ8_0 gate/up fused MLP projection when both tensors are SQ8_0 and the chosen policy includes them.
- SQ8_0 Q/K/V fused projection for self-attention when all required tensors are SQ8_0.
- Optional mixed AQ4/SQ8_0 fused boundary only after all-SQ8_0 fused boundary is stable.

Fallback rule:

- CPU/HIP staging fallback is allowed for correctness smoke.
- Performance rows must mark fallback explicitly with `sq_execution_mode`.
- A result must not be counted as native SQ8_0 throughput if it used F32 materialized fallback for the measured projection path.

## Backend Dispatch Plan

Use the existing Rust `backend_dispatch` model as the high-level selector.

The request should include:

- operation: e.g. `matvec`, `matvec_batch`, `self_attn_qkv`, `mlp_gate_up`, `cached_prefix_attention`;
- phase: `prefill`, `decode`, `component`, `materialize`, or `smoke`;
- format ID: `SQ8_0`, `AQ4_0`, or unquantized;
- model architecture: e.g. `qwen3`, `qwen3.5`;
- GPU architecture: e.g. `RDNA4`, `RDNA2`, `Ampere`;
- GPU name: e.g. `Radeon AI PRO R9700`, `A100_80GB`;
- implementation priority.

Selection rules:

- Exact GPU-name implementation beats GPU-architecture default.
- GPU-architecture implementation beats generic implementation.
- Format-specific implementation beats generic implementation when other specificity is equal.
- Model-specific implementation beats model-agnostic implementation when other specificity is equal.
- Explicit priority resolves remaining ties.

First real connections:

1. cached-prefix attention executor resolution:
   - RDNA4 + SQ8_0/FP8 KV + prefill -> `cached_prefix_rdna4_fp8_auto`;
   - fallback -> existing chunked/F32 path.
2. SQ8_0 projection matvec resolution:
   - RDNA4 + SQ8_0 + batch prefill -> direct SQ8_0 batch/pair/triple kernel;
   - CPU or unsupported GPU -> staging/materialized fallback with explicit telemetry.
3. future C++ implementation registry:
   - C++ kernel families should expose implementation IDs that Rust can select or report.

## C++ Runtime Organization

Short term:

- Keep `runtime/src/ullm_runtime.cpp` as the translation unit entrypoint.
- Keep current `.inc` split while adding clearer kernel-family boundaries.
- Add descriptors in C++ or Rust-side metadata before splitting into multiple `.cpp` files.

Medium term:

```text
runtime/src/
  ullm_runtime.cpp
  ullm_runtime_api_*.inc
  kernels/
    sq8_0/
      sq8_0_matvec_rdna4.inc
      sq8_0_matvec_cpu.inc
    aq4_0/
      aq4_0_matvec_rdna4.inc
    attention/
      cached_prefix_rdna4_fp8.inc
      cached_prefix_generic.inc
```

Do not introduce model/GPU `if` chains inside hot API wrappers. Implementation choice should happen through descriptors or a registry-style resolver.

## Milestones

### M0: Naming Foundation

Status: done.

Deliverables:

- `format_id.rs`;
- `tools/ullm_format_ids.py`;
- `docs/specs/format-ids-v0.1.md`;
- SQ artifact builder emits `SQ8_0`.

### M1: SQ8_0 Plan And Spec Cleanup

Status: this plan starts it.

Deliverables:

- this plan;
- update old SQ FP8 plan language when touched;
- keep legacy CLI names documented as compatibility names;
- make new result summaries use `SQ8_0` public format wording.

### M2: Artifact Hardening

Status: partial.

Deliverables:

- manifest checksum validation required for generated payloads;
- deterministic artifact generation test for a tiny fixture;
- policy JSON validation errors with stable messages;
- artifact rebuild command recorded in result metadata;
- sidecar artifact path and future package-integrated path documented separately.

Verification:

```text
python3 -m unittest tests.test_build_sq_fp8_artifact_policy tests.test_ullm_format_ids
python3 -m py_compile tools/build-sq-fp8-w8a16-artifact.py tools/ullm_format_ids.py
```

### M3: Library-Side Resident SQ8_0 Loader

Status: done for the current v0.1 resident loader boundary.

Deliverables:

- move resident SQ FP8 payload/scale buffer loading out of CLI-only `main_parts`;
- expose a reusable SQ8_0 overlay loader for Qwen3 package runtime;
- preserve direct matvec storage references without requiring F32 materialization;
- keep telemetry fields for selected tensors, passthrough tensors, scale layout, and execution mode.

Verification:

```text
cargo test -p ullm-engine sq_runtime -- --test-threads=1
cargo check -p ullm-engine
```

### M4: Native SQ8_0 Projection Execution

Status: partial.

Done:

- single, batch, pair, and triple SQ FP8 matvec APIs exist.
- mixed request-state rows report the projection boundary counters and the selected SQ8_0 projection
  implementation IDs resolved through `backend_dispatch`.
- full mixed request-state smoke can run supported SQ8_0 tensors through the direct resident
  dequant matvec path without whole-tensor F32 materialization.

Remaining:

- extend any remaining package/model-loop rows that still miss direct SQ8_0 projection telemetry;
- store `sq_fp8_single_matvec_count`, `sq_fp8_batch_matvec_count`, `sq_fp8_pair_matvec_count`, and `sq_fp8_triple_matvec_count` in any remaining throughput rows that do not yet expose them;
- ensure performance summaries reject accidental `materialized_f32_fallback` rows unless explicitly marked as fallback.

Verification:

```text
cargo test -p ullm-runtime-sys cpu_sq_fp8_matvec -- --test-threads=1
cargo test -p ullm-runtime-sys first_hip_sq_fp8_matvec -- --test-threads=1
cargo check -p ullm-engine
```

For required HIP validation:

```text
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1 cargo test -p ullm-runtime-sys first_hip_sq_fp8_matvec_f32_computes_expected_values_when_available -- --test-threads=1
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1 cargo test -p ullm-runtime-sys first_hip_sq_fp8_matvec_batch_f32_computes_expected_values_when_available -- --test-threads=1
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_PAIR_KERNEL=1 cargo test -p ullm-runtime-sys first_hip_sq_fp8_matvec_pair_f32_computes_expected_values_when_available -- --test-threads=1
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1 cargo test -p ullm-runtime-sys first_hip_sq_fp8_matvec_triple_f32_computes_expected_values_when_available -- --test-threads=1
```

### M5: Model-Loop And Batch Integration

Status: partial.

Deliverables:

- selected-layer SQ8_0 model-loop bridge for len4/case_a/case_b prompt bundle;
- full mixed request-state path with SQ8_0 overlay metadata;
- batch throughput rows with artifact load excluded/included fields explicitly separated;
- final logits guard and behavioral prompt-suite result recorded as regression diagnostics.

Minimum result fields:

- `format_id = SQ8_0`;
- `implementation_id`;
- `sq_artifact`;
- `sq_fp8_tensor_count`;
- `sq_passthrough_tensor_count`;
- `sq_execution_mode`;
- projection boundary counters;
- artifact load/materialization wall time;
- prefill/decode/end-to-end total tok/s;
- prompt-suite regression status.

### M6: Dispatch Integration

Status: partial.

Done:

- cached-prefix attention smoke uses `backend_dispatch` when the CLI executor is not explicitly
  overridden.
- SQ8_0 projection execution records operation-level implementation IDs for single, batch, pair, and
  triple matvec boundaries.
- `PackageAq4ResidentMatvec` now carries SQ8_0 projection dispatch decisions to the single, batch,
  pair, and triple matvec execution boundaries before calling the existing runtime kernels.
- runtime GPU-architecture detection maps `compute_major == 12` to `RDNA4`, so R9700 dispatch rows
  resolve to `sq8_0_matvec_*_rdna4_direct` instead of generic direct IDs.

Remaining:

- use dispatch-selected SQ8_0 projection implementations to select between multiple C++ kernel
  families once multiple implementations exist;
- add registry entries for higher-level SQ8_0 fused projection kernels;
- keep selected implementation IDs in all result rows that represent dispatch-selected execution.

Example intent:

```text
format_id == SQ8_0 && gpu_arch == RDNA4 && operation == self_attn_qkv
  -> sq8_0_self_attn_qkv_rdna4_v0

gpu_arch == RDNA4 && gpu_name == Radeon_AI_PRO_R9700 && operation == cached_prefix_attention
  -> cached_prefix_rdna4_fp8_auto
```

### M7: C++ Kernel Descriptor Split

Status: planned.

Deliverables:

- define implementation descriptor naming convention;
- move SQ8_0 kernel source snippets into SQ8_0-specific `.inc` files;
- make build.rs track new include files;
- keep one translation unit until helper ownership is clear;
- later split `.cpp` files only after compile/link behavior is stable.

### M8: Regression Suite

Status: partial.

Regression categories:

- schema/format regression;
- artifact generation regression;
- runtime decode regression;
- direct-kernel unit regression;
- package model-loop regression;
- throughput regression;
- behavioral prompt-suite regression.

Important rule:

- Strict AQ4_0-vs-SQ8_0 token/logit equality is diagnostic, not a proof of FP8 quality and not a universal blocker.
- A failing diagnostic must say whether it indicates an implementation bug, an expected FP8 numerical difference, or an unsupported policy/layout.

### M9: SQ8_0 v0 Completion Criteria

`SQ8_0` v0 implementation is complete when:

- public format metadata is consistently emitted as `SQ8_0`;
- legacy aliases remain readable;
- a reproducible SQ8_0 artifact can be generated from policy JSON;
- runtime can load SQ8_0 payload/scale resident buffers without whole-model F32 expansion;
- R9700/RDNA4 uses direct SQ8_0 projection kernels for selected supported tensors;
- result rows clearly report native vs fallback SQ8_0 execution;
- backend dispatch records which implementation was selected;
- package/model-loop regression suite passes with documented tolerances;
- docs/specs describe the implementation as adopted SQ8_0, not as an FP8 quality experiment.

### M10: vLLM + FP8 External Baseline Comparison

Status: planned for the later half of the SQ8_0 implementation cycle.

Purpose:

- Compare uLLM `SQ8_0` against an established FP8 serving/runtime baseline.
- Keep this as a performance and systems comparison, not a proof that FP8 quantization is acceptable.
- Use `vLLM + Qwen3-14B-FP8` as the primary external baseline because it is the closest existing
  FP8 runtime target already represented in local plans and benchmark records.

Prerequisites:

- M5 model-loop / batch integration is stable enough to produce repeatable uLLM rows.
- M6 dispatch records the selected uLLM implementation ID.
- uLLM rows clearly distinguish native SQ8_0 direct execution from fallback execution.
- `tools/run-external-benchmark.py` can emit comparable `inference-benchmark-result-v0.1` rows for
  both uLLM and vLLM.

Baseline target:

```text
engine: vLLM
model: Qwen/Qwen3-14B-FP8
local model path: ~/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8/
primary GPU: R9700/RDNA4
environment: build/envs/vllm-rocm-nightly or the current working vLLM ROCm env
```

Workload grid:

- smoke: prompt `16`, generated `8`, batch/concurrency `1`;
- representative decode: prompt `512`, generated `128`, batch/concurrency `1`;
- cached-prefix or long-prefill probe when uLLM has the matching SQ8_0 path;
- later batch grid: concurrent requests `1, 2, 4, 8` only after uLLM and vLLM rows are both stable.

Metrics:

- prefill tok/s;
- decode tok/s;
- total tok/s;
- latency p50/p95 when available;
- VRAM baseline, peak, and consumed bytes;
- decode tok/s times consumed GiB;
- backend/executor name;
- model dtype, quantization, KV cache dtype, and failure reason if unsupported.

Comparison rules:

- vLLM comparison happens after uLLM SQ8_0 rows are implementation-valid; it must not block earlier
  artifact or direct-kernel work.
- If vLLM cannot run the exact FP8 model or local GPU target, record an `unsupported` or `failed`
  row rather than omitting it.
- Do not compare uLLM selected-layer rows against full vLLM serving rows as equivalent results.
  Selected-layer uLLM rows are path-connectivity diagnostics only.
- Full-package uLLM rows can be compared against vLLM serving rows when prompt/generation lengths,
  batch/concurrency, KV cache dtype, and model target are documented.
- Differences in tokenizer/server overhead should be recorded. When possible, keep a raw model-loop
  uLLM row and a server-style vLLM row as separate comparison classes.

Expected outputs:

- JSONL rows in `benchmarks/results/YYYY-MM-DD/external/` or a later SQ8_0 comparison directory.
- Markdown summary with one table for successful comparable rows and one table for unsupported or
  failed rows.
- A note linking to `docs/plans/r9700-qwen3-14b-fp8-external-engine-plan-v0.1.md` and the exact
  vLLM environment used.

## Immediate Work Queue

1. Add `sq_runtime.rs` or equivalent library boundary. Done.
   - Move SQ resident payload/scale loading and `SqFp8` storage refs out of CLI-local code.
2. Add an SQ8_0 implementation result label. Done.
   - Keep CLI command compatibility, but result rows should report public `format_id = SQ8_0`.
3. Wire `backend_dispatch` into cached-prefix executor resolution. Done.
   - This is the smallest real dispatch connection and exercises GPU arch/name selection.
4. Wire `backend_dispatch` into SQ8_0 projection execution. Partial.
   - Operation-level reporting is done.
   - Dispatch decisions now reach the direct matvec execution boundary.
   - C++ kernel-family switching remains a follow-up.
5. Generate one fresh SQ8_0 artifact from policy JSON and run. Done.
   - materialize smoke;
   - selected-layer model-loop smoke;
   - mixed request-state throughput row.
6. Update old docs opportunistically.
   - Do not rewrite all historical T2 logs.
   - New plans/results should use `SQ8_0` terminology.
7. After the first implementation-valid full-package SQ8_0 rows exist, run the M10 vLLM + FP8
   comparison grid and save unsupported/failure rows explicitly when vLLM cannot match the target.

## Risks

| risk | impact | handling |
| --- | --- | --- |
| old SQ FP8 wording remains in historical docs | confusion about whether FP8 is adopted | keep history unchanged but add current SQ8_0 plan and specs |
| CLI-local SQ overlay logic grows further | hard to reuse in package/model-loop runtime | move resident loader into library before adding more paths |
| direct SQ8_0 matvec is correct but not fast enough | format appears slow due to kernel maturity | report implementation ID and direct/fallback mode separately |
| backend dispatch is added too broadly | hard-to-debug selection bugs | connect one operation at a time with tests |
| C++ split breaks anonymous namespace/helper ownership | build instability | keep one translation unit until descriptors are stable |
| strict AQ4 equality blocks valid FP8 behavior | false negatives | keep exact equality as regression diagnostic, use implementation correctness and behavioral suite for gating |
| vLLM comparison is run before uLLM rows are comparable | misleading performance conclusions | only compare full-package SQ8_0 rows with documented workload and execution mode |
