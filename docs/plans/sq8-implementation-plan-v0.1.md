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

Status: planned.

Deliverables:

- move resident SQ FP8 payload/scale buffer loading out of CLI-only `main_parts`;
- expose a reusable SQ8_0 overlay loader for Qwen3 package runtime;
- preserve direct matvec storage references without requiring F32 materialization;
- keep telemetry fields for selected tensors, passthrough tensors, scale layout, and execution mode.

Verification:

```text
cargo test -p ullm-engine sq -- --test-threads=1
cargo check -p ullm-engine
```

### M4: Native SQ8_0 Projection Execution

Status: partial.

Done:

- single, batch, pair, and triple SQ FP8 matvec APIs exist.

Remaining:

- make full model-loop path use direct SQ8_0 projection kernels when tensors are resident SQ8_0;
- store `sq_fp8_single_matvec_count`, `sq_fp8_batch_matvec_count`, `sq_fp8_pair_matvec_count`, and `sq_fp8_triple_matvec_count` in all relevant throughput rows;
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

Status: planned.

Deliverables:

- connect `backend_dispatch` to at least one production-like runtime path;
- add registry entries for SQ8_0 RDNA4 projection kernels;
- add registry entries for cached-prefix FP8 executor;
- record selected implementation ID in result rows.

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

## Immediate Work Queue

1. Add `sq_runtime.rs` or equivalent library boundary.
   - Move SQ resident payload/scale loading and `SqFp8` storage refs out of CLI-local code.
2. Add an SQ8_0 implementation result label.
   - Keep CLI command compatibility, but result rows should report public `format_id = SQ8_0`.
3. Wire `backend_dispatch` into cached-prefix executor resolution.
   - This is the smallest real dispatch connection and exercises GPU arch/name selection.
4. Wire `backend_dispatch` into SQ8_0 projection execution.
   - Start with operation-level reporting before changing kernel selection semantics.
5. Generate one fresh SQ8_0 artifact from policy JSON and run:
   - materialize smoke;
   - selected-layer model-loop smoke;
   - mixed request-state throughput row.
6. Update old docs opportunistically.
   - Do not rewrite all historical T2 logs.
   - New plans/results should use `SQ8_0` terminology.

## Risks

| risk | impact | handling |
| --- | --- | --- |
| old SQ FP8 wording remains in historical docs | confusion about whether FP8 is adopted | keep history unchanged but add current SQ8_0 plan and specs |
| CLI-local SQ overlay logic grows further | hard to reuse in package/model-loop runtime | move resident loader into library before adding more paths |
| direct SQ8_0 matvec is correct but not fast enough | format appears slow due to kernel maturity | report implementation ID and direct/fallback mode separately |
| backend dispatch is added too broadly | hard-to-debug selection bugs | connect one operation at a time with tests |
| C++ split breaks anonymous namespace/helper ownership | build instability | keep one translation unit until descriptors are stable |
| strict AQ4 equality blocks valid FP8 behavior | false negatives | keep exact equality as regression diagnostic, use implementation correctness and behavioral suite for gating |
