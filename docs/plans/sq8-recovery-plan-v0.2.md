# SQ8_0 Recovery And Performance Plan v0.2

Status: active

Date: 2026-07-10

This plan supersedes the execution order in `sq8-implementation-plan-v0.1.md`. The v0.1 document remains as implementation history.

Schedule basis: acceptance-driven. There is no fixed completion deadline. Work advances only when the current phase satisfies its evidence and acceptance requirements.

## 前回の要点

- resident loader、40層projection接続、real-request batch、D2D handoff、telemetryは再利用できる。
- 現Qwen3-14B-FP8 sidecarはsourceの128x128 `weight_scale_inv`を適用しておらず、元checkpointと数学的に異なる。
- 現batch kernelは重みをbatchごとに読み直すscalar W8A16 matvecであり、b2/b4/b8のtotal throughputが伸びない。
- 既存uLLM same-model結果は接続診断へ降格し、品質・性能比較には使わない。

## 今回の変更点

- 目標を「短時間で40層serving parityを作る」から「source correctnessとR9700上のFP8高速化可能性を段階的に証明する」へ変更する。
- canonical artifact、device向けprepack、reference execution、optimized executionを別の契約として扱う。
- full model統合より前に、source round-trip、1 linear oracle、実shape component benchmarkを必須にする。
- 各phaseに成果物、合格条件、停止条件を設定する。
- 固定の時間枠は設けず、初期component検証、最低限の機能完成、追加性能改善を分離する。

## 次の行動

最初の作業単位は、Qwen3-14B-FP8のF8 payloadと`weight_scale_inv`を保持するcanonical artifact仕様、および実checkpoint tensorを使うround-trip golden testである。これが通るまでGPU性能kernelや40層統合へ進まない。

## 1. Objective

SQ8_0について、次の順序で証拠を積み上げる。

1. 元checkpointのFP8 weightを正しく表現できる。
2. CPU referenceとGPU referenceが同じlinear演算を行う。
3. R9700上でactivation quantization込みのFP8 componentがbatch scalingする。
4. 1 decoder layer、40層、prefill、実生成へ段階的に統合する。
5. 最後に同条件のvLLM比較を行う。

FP8一般の品質採否は対象にしない。実装がsourceの量子化契約を守っていることと、対象GPU上で効率的に実行できることを確認する。

## 2. Fixed Target

Initial target:

- model: Qwen3-14B-FP8
- GPU: Radeon AI PRO R9700 / `gfx1201`
- weight format: FP8 E4M3
- source scale layout: 128x128 block
- activation scheme: dynamic
- initial batch dimension: M=`1,2,4,8`
- extended batch dimension: M=`16,32,128`

Representative projection shapes:

| projection class | N | K |
| --- | ---: | ---: |
| q/o projection | 5120 | 5120 |
| k/v projection | 1024 | 5120 |
| gate/up projection | 17408 | 5120 |
| down projection | 5120 | 17408 |

The initial component checkpoint uses only the q/o projection shape. The other shapes follow after that checkpoint because one shape cannot represent KV and MLP behavior.

Before each benchmark, record:

- uLLM git commit and dirty state;
- model path plus config and tensor-file hashes;
- ROCm, driver, compiler, and selected library versions;
- GPU name, arch, clock/temperature observation, and device index;
- random seed, input hash, warmup count, repeat count, and command line.

## 3. Non-Goals Before The Initial Component Gate

- 40-layer integration;
- full artifact generation for every tensor;
- QKV or gate/up fusion;
- scheduler and HTTP serving;
- lm_head and sampling integration;
- vLLM end-to-end parity;
- result schema expansion unrelated to component evidence;
- new dispatch descriptors without a corresponding implementation.

## 4. Contract Separation

### 4.1 Canonical Artifact

The canonical artifact preserves model semantics and is independent of a specific kernel layout.

Required fields:

- exact F8 payload;
- corresponding 2D block scale payload;
- block shape, logical tensor shape, dtype, source tensor name, and checksums;
- explicit scale meaning and reconstruction formula;
- source model/config identity;
- schema version.

An F8 source weight without its scale is invalid. Weight/scale pairs must be one-to-one, with missing, duplicate, and incompatible shapes rejected.

### 4.2 Device Prepack Cache

Kernel-specific transposition, swizzle, padding, or packed scales are derived data. Store them in a versioned cache keyed by canonical artifact hash, GPU arch, implementation ID, and packing version.

Changing a kernel layout must not change the canonical artifact contract.

### 4.3 Execution Profiles

- `reference_w8a16`: BF16/F32 activation plus correct block-scale dequantization. This is a correctness path, not a performance claim.
- `rdna4_w8a8_block`: dynamic FP8 activation plus block-scaled FP8 GEMM. This is the target performance path.

Result rows must identify the profile. A reference profile cannot be promoted as an optimized SQ8_0 result.

## 5. Phase Plan

### P0: Quarantine And Reproducibility Contract

Tasks:

- mark 2026-07-09/10 uLLM Qwen3-14B-FP8 same-model rows as invalid for quality/performance conclusions without deleting history;
- freeze model/config identity, one q/o tensor, one deterministic activation fixture, and benchmark commands;
- add an acceptance matrix for artifact, numerical correctness, kernel family, fallback, and performance;
- distinguish connection diagnostics from implementation-valid and performance-valid results.

Deliverables:

- quarantine note or machine-readable validity marker;
- reproducibility metadata fixture;
- fixed input and expected source reconstruction hash.

Acceptance:

- rerunning the fixture selects the same source tensors and produces identical input hashes;
- old invalid rows cannot pass the implementation-valid comparison gate.

Stop condition:

- do not begin performance work if model identity or source tensor pairing is ambiguous.

### P1: Source-Correct Canonical Artifact

Tasks:

- extend or replace the v0.1 row/row-block scale schema with an explicit 2D block-scale layout;
- import the F8 payload and matching `weight_scale_inv` without dequantize/requantize loss;
- validate block counts, edge blocks, dtype, tensor name, shape, and checksums;
- keep BF16-to-SQ8 quantization as a separate builder mode from F8 checkpoint import;
- use chunked/streaming reads and atomic output replacement so full generation does not require all tensors in RAM.

Tests:

- tiny synthetic F8 + 2D-scale fixture;
- missing, duplicate, wrong-shape, and wrong-dtype scale rejection;
- one real q/o projection tensor from Qwen3-14B-FP8;
- deterministic rebuild and checksum verification.

Acceptance:

- F8 payload and scale round-trip are byte exact;
- direct source reconstruction and artifact reconstruction match element-for-element in F32 for sampled and boundary blocks;
- all expected weight/scale pairs are accounted for before a full artifact can be finalized.

Stop condition:

- one reconstruction mismatch blocks full artifact generation and all GPU performance work.

### P2: Reference Correctness Path

Tasks:

- implement a small CPU oracle for canonical block-scale reconstruction and linear output;
- make `reference_w8a16` consume the canonical artifact correctly;
- compare one q/o projection on CPU and GPU with fixed BF16/F32 activations;
- return typed correctness metrics rather than parsing stdout.

Acceptance:

- no NaN/Inf;
- the GPU reference output passes a tolerance frozen before optimization begins;
- the report includes artifact hash, implementation profile, fallback state, and error metrics;
- the current scalar kernel is retained only if it is source-correct and labeled as reference.

Stop condition:

- do not start W8A8 optimization until one linear layer matches the oracle.

### P3: R9700 Execution Route Selection

Evaluate in this order:

1. hipBLASLt;
2. Composable Kernel;
3. rocWMMA;
4. direct HIP kernel only when the existing libraries cannot satisfy the required operation.

Check:

- gfx1201 support;
- FP8xFP8 input and supported accumulation/output type;
- block-scale and scale-layout requirements;
- transpose and alignment constraints;
- whether execution silently falls back to BF16 or a non-matrix path;
- whether actual Qwen projection shapes run.

Acceptance:

- select one concrete implementation route or record a bounded reason that a direct kernel is required;
- profiler evidence identifies matrix instructions for any route claimed as matrix-core FP8.

Stop condition:

- stop evaluating alternatives after one route satisfies the required operation, numerical gate, and P4 performance gate;
- if no library route supports the operation, record the unsupported constraints and define the smallest direct-kernel work package before implementation;
- do not evaluate another library merely for theoretical peak performance unless profiler evidence shows that the selected route cannot meet P4;
- implicit BF16 fallback is not an FP8 performance proof.

### P4: One-Projection Optimized Component

Tasks:

- implement CPU dynamic activation quantization as the optimized-path oracle;
- implement GPU activation quantization and include its cost in end-to-end component timing;
- run FP8 GEMM using the selected P3 route;
- benchmark kernel-only and quantization-inclusive latency separately;
- use a dedicated M=1 GEMV path when appropriate and tiled GEMM for M>=2;
- profile memory traffic, occupancy, launch count, and matrix instructions.

Provisional numerical gate, frozen before GPU result inspection:

- relative L2 error <= `5e-3` against the CPU optimized-path oracle;
- cosine similarity >= `0.9999`;
- no NaN/Inf and no fallback.

Performance gate:

- report p50 latency and aggregate throughput after fixed warmup/repeats;
- activation quantization time is included in the promotion number;
- profiler evidence confirms the intended FP8 matrix path and rules out cache-only artifacts;
- aggregate throughput increases beyond the repeated-measurement noise band from M=2 to M=8;
- the optimized component is faster than `reference_w8a16` for the same shape and inputs.

Recommended initial target:

- M=8 aggregate throughput >= `2.5x` M=2 for the first q/o shape.

Missing the recommended ratio does not block source-correct functional integration when scaling is non-flat and the remaining bottleneck is explained by profiler evidence. It does block claiming that the target performance has been reached.

Stop condition:

- if b2-b8 remains flat, stop optimized-path integration and save the profiler-backed bottleneck result;
- a source-correct `reference_w8a16` integration may continue for functional validation, but it must remain labeled as reference;
- do not compensate for a failed component gate with host-staging, schema, descriptor, or serving work.

### P5: Shape Expansion And One Decoder Layer

Entry gates:

- functional path: P1 and P2 green, with the P3 route decision recorded;
- optimized path: the P4 required performance gate is also green.

Tasks:

- extend the component to k/v, gate/up, and down projection shapes;
- select dispatch by shape and M using measured results;
- quantize a shared input once where QKV or gate/up consume the same activation;
- integrate one complete decoder layer;
- compare intermediate tensors and final layer output with an independent oracle.

Acceptance:

- all four projection classes pass source-correct numerical gates;
- no projection uses an unreported fallback;
- one decoder layer passes intermediate and final-output checks;
- any projection labeled optimized has passed the P4 required performance gate;
- optimized layer latency improves over the source-correct reference path when an optimized layer is present.

Stop condition:

- do not scale an optimized path to 40 layers while any dominant projection shape is unverified or flat;
- the source-correct reference path may proceed to full-model functional validation without an optimized label.

### P6: Full Model And Prefill

Entry gates:

- functional path: one source-correct decoder layer green;
- optimized path: one optimized decoder layer green.

Tasks:

- extend resident execution and D2D handoff to all 40 layers;
- group prompt tokens into M dimensions or bounded chunks instead of executing only timestep batches;
- preserve separate decode dispatch for small M;
- add lm_head, output-health checks, and source-correct end-to-end logits verification;
- then evaluate QKV and gate/up fusion based on profiler evidence.

Acceptance:

- all expected projections use the selected profile with complete counters;
- no host staging in the measured steady-state path;
- prefill launch count reflects token batching/chunking rather than 16 separate full-stack steps;
- output health is evaluated and end-to-end logits pass the frozen regression gate.

### P7: Real Generation And External Comparison

Entry gate: source-correct full model green

Tasks:

- implement typed model-runner results;
- connect lm_head, sampling, next-token feedback, EOS, request completion times, and scheduler-ready batches;
- measure offline throughput before adding HTTP transport;
- run the same model, prompt/generated shapes, context, and generation semantics in uLLM and vLLM;
- keep component, model-loop diagnostic, offline serving, and online serving result classes separate.

Acceptance:

- generated token IDs feed the next decode step;
- request-level latency and aggregate throughput are derived from the real generation loop;
- comparison gates verify model identity, workload identity, execution profile, output health, fallback state, and artifact hash;
- vLLM comparison contains no quarantined uLLM rows.

## 6. Initial Validation Checkpoint

This checkpoint decides, with reproducible evidence, whether source-correct SQ8_0 can use an efficient FP8 path on the R9700. It has no fixed duration.

| order | work | required output |
| ---: | --- | --- |
| 1 | P0 contract and quarantine | fixed hashes, input, commands, and validity rules |
| 2 | P1 canonical one-tensor artifact | byte-exact payload/scale round-trip and source golden |
| 3 | P2 one-linear reference | CPU/GPU correctness report |
| 4 | P3 capability check | selected library/kernel route with profiler evidence or explicit blocker |
| 5 | P4 q/o projection component | dynamic activation quantization, M grid, correctness, and batch scaling |
| 6 | result freeze | commands, metrics, profiler record, and next-route decision |

Checkpoint outcomes:

- Green: source-correct artifact plus one q/o projection shows valid FP8 execution and batch scaling. Continue to P5.
- Yellow: artifact/reference are correct, but available library routes are unsupported or do not scale. Implement the bounded direct-kernel work package, then repeat P4.
- Red: artifact round-trip or one-linear correctness is not valid. Remain in P1/P2 and do not begin performance integration.

No later phase may be compressed or skipped because an earlier phase took longer than expected.

## 7. Minimum Functional Completion

The minimum functional SQ8_0 implementation is complete when all of the following are true:

- the full Qwen3-14B-FP8 canonical artifact accounts for every expected weight/scale pair and passes reconstruction checks;
- all four representative projection classes pass source-correct numerical gates and use measured dispatch decisions;
- one complete decoder layer passes independent intermediate and final-output checks;
- all 40 layers run through resident buffers and D2D handoff without an unreported fallback;
- the model runs from resident SQ8_0 payload/scale buffers without whole-model F32 expansion;
- prefill batches or chunks prompt tokens instead of executing only one timestep at a time;
- lm_head, sampling, next-token feedback, EOS, and request completion are part of the real generation loop;
- output health, artifact identity, execution profile, fallback state, and request-level metrics are reported through typed results;
- an implementation-valid uLLM functional baseline is recorded.

The minimum functional completion does not require:

- exact performance parity with vLLM;
- every possible QKV, gate/up, normalization, or activation fusion;
- peak tuning for every M and projection shape;
- HTTP transport;
- RDNA2/V620 native FP8 optimization.

Performance differences found at this point become measured follow-up work, not reasons to keep changing the minimum implementation without an identified bottleneck.

### 7.1 Optimized v0 Completion

The R9700 optimized v0 is complete when:

- all dominant projection classes use `rdna4_w8a8_block` or another explicitly named native FP8 profile without an unreported fallback;
- profiler evidence confirms native FP8 matrix instructions;
- prefill uses token batching or chunking and decode has a measured small-M path;
- steady-state host staging is zero on the measured path;
- activation quantization-inclusive execution is faster than `reference_w8a16` and batch scaling is not flat;
- the recommended M=8/M=2 target is either met or the remaining gap is documented with profiler evidence;
- a same-condition vLLM result is recorded for context, without requiring exact performance parity.

## 8. Optimization Scope Control

Additional optimization starts only when one of these entry conditions is true:

- an existing P4-P7 performance gate is not met;
- profiler evidence identifies the operation as at least 10% of measured end-to-end runtime;
- the current path has a structural defect such as flat batch scaling, repeated full-weight reads, avoidable host synchronization, or an implicit fallback.

Every optimization work package must state before implementation:

- the measured bottleneck;
- the expected affected metric and minimum useful gain;
- the files and execution boundary in scope;
- the correctness tests that must remain green;
- the rollback condition.

Optimization stops when:

- the functional completion conditions and existing performance gates are satisfied, and the next idea has less than 5% estimated end-to-end impact;
- two consecutive work packages against the same bottleneck each produce less than 3% repeatable end-to-end improvement;
- the profiler shows that another subsystem is now dominant;
- an optimization increases complexity without a repeatable measured gain.

Exceptions require a correctness, compatibility, or maintainability reason recorded before the work begins. Descriptor additions, counter changes, and benchmark-schema changes are not optimization progress by themselves.

## 9. Global Promotion Rules

- Correctness precedes performance.
- Component evidence precedes layer integration.
- One-layer evidence precedes 40-layer integration.
- Full-model correctness precedes serving comparison.
- A descriptor, counter, or coverage ratio is not performance evidence by itself.
- A self-reference guard is not an independent oracle.
- A result with unevaluated output health cannot be promoted.
- A result using an implicit or unreported fallback cannot be labeled native SQ8_0 performance.
- Any change to artifact semantics, activation quantization, accumulation type, or prepack version reruns the relevant lower-level gates.

## 10. Reusable And Deferred Work

Reuse now:

- `sq_runtime` resident ownership and buffer lifetime;
- D2D layer handoff primitives;
- batch API and scheduler integration points;
- fallback and projection counters;
- benchmark storage and comparison-gate structure.

Keep as reference only:

- current scalar SQ8_0 batch matvec;
- current timestep-batched full-stack diagnostic;
- 2026-07-09/10 Qwen3-14B-FP8 uLLM rows.

Defer until their entry gates:

- fused descriptor activation;
- further host-staging cleanup;
- serving parser/schema expansion;
- HTTP serving;
- final vLLM grid.

## 11. Completion Definition

The SQ8_0 functional implementation and R9700 optimized v0 have separate completion states defined above. This recovery plan is complete when both states are satisfied:

- canonical artifacts preserve exact source FP8 payload and 2D block scales;
- source, CPU reference, GPU reference, and optimized execution have independent passing evidence;
- actual Qwen projection shapes use measured, source-correct dispatch decisions;
- one decoder layer and the full model pass oracle/output-health checks;
- prefill uses token batching or chunking and decode uses an appropriate small-M path;
- real generation drives subsequent token inputs;
- external comparisons use the same model and workload and exclude quarantined rows, without requiring exact vLLM parity;
- any remaining performance gap is recorded as a profiler-backed backlog item rather than an open-ended completion blocker.
