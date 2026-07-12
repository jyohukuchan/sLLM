# ADR 0004: Model graph and state schema boundary

## Status

Accepted for staged implementation under the generic production inference optimization plan.

## Context

The current serving paths prove that resident weights, per-layer state, paged
KV cache writes, recurrent state, and device operations can work together.
They do not yet provide a common execution contract. Qwen3 package loading
derives a dense decoder from Qwen tensor names, while Qwen3.5 AQ4 discovers
self-attention and linear-attention layers and stores their execution state in
different resident layer types. The scheduler owns request progress and block
allocation, but it does not describe a multi-request prefill chunk to a model
executor.

This creates two risks. First, making prefill batched by adding another
Qwen3.5/AQ4 loop would copy scheduler, state, and backend-selection behavior
into model-specific code. Second, treating an AQ4/SQ8 format choice as a model
topology choice would prevent one graph from using different numerical formats
or backend implementations.

ADR 0002 assigns orchestration and request lifetime to Rust and backend memory
and kernels to the C++ runtime. ADR 0003 assigns paged-KV block tables and
lifetime to Rust while the backend owns the per-layer payload. This ADR extends
those boundaries to all request state and defines the input to a generic,
batched and chunked executor.

## Decision

### 1. Keep five independent axes

The common execution contract MUST keep the following axes separate:

| Axis | It describes | It does not describe |
| --- | --- | --- |
| Model topology | Node order, data dependencies, semantic attributes, logical shapes | GPU kernel or quantization choice |
| Execution phase | Cold prefill, cached-prefix prefill, decode, and future verify/speculative phases | Model family |
| State kind | Paged or sliding KV, recurrent state, convolution history, position and future cross-attention state | Tensor names or hardware allocation policy |
| Numerical format | Weight, activation, and state dtype/format such as F32, BF16, AQ4_0, or SQ8_0 | Graph semantics |
| Backend capability | Backend/device, supported shape buckets, state layouts, workspace limits, and operator implementations | Public model identity |

A generic executor MUST NOT select its control flow by model ID, package name,
AQ4, SQ8, Qwen3, or Qwen3.5. An architecture-specific semantic difference MUST
be represented by a graph-node attribute or by a newly defined graph operator.
An implementation may use an optional architecture constraint as a registry
selection preference only when a generic-compatible fallback exists.

### 2. ModelGraph declares topology and semantics

`ModelGraph` is an immutable, backend-independent description of the model
topology. It contains ordered nodes, typed value edges, logical shape
constraints, node attributes, and references to logical weights and state
entries. It does not own device buffers or physical tensor names.

The initial semantic node vocabulary is:

- embedding;
- norm;
- linear and fused linear group;
- rotary position transform;
- dense attention;
- recurrent attention;
- activation;
- gated MLP;
- residual;
- final norm;
- LM head; and
- sampling.

The vocabulary is deliberately open for MoE router/expert groups,
sliding-window attention, cross attention, convolution or SSM scan,
multimodal adapters, and MTP/speculative heads. Those extensions MUST add an
explicit semantic operator or attribute and MUST NOT be encoded as an
undocumented model-name branch in the generic executor.

### 3. ModelAdapter is the architecture-specific boundary

`ModelAdapter` converts a package or upstream model description into
`ModelGraph`, `WeightBindings`, and `StateSchema`. It MAY:

- map package/HF tensor names to logical weight IDs;
- derive ordered layers, graph nodes, operator attributes, and logical shapes;
- declare architecture-specific attention, RoPE, activation, residual, and
  recurrent semantics;
- declare request/layer state requirements and the model/tokenizer contract;
- validate that the discovered tensors satisfy the declared graph before any
  backend allocation; and
- derive a graph for legacy packages that do not contain graph metadata.

`ModelAdapter` MUST NOT:

- call HIP, CPU, or other backend APIs;
- allocate device buffers or choose workspace sizes;
- own a token-by-token, prefill, decode, or sampling loop;
- choose chunk sizes, request buckets, or scheduler policy;
- name a concrete kernel implementation;
- mutate scheduler progress, state lifetime, or commit state; or
- handle HTTP, SSE, worker protocol, gateway, or OpenWebUI behavior.

This permits a Qwen3 adapter to describe dense attention and a Qwen3.5 adapter
to describe an ordered mixture of dense and recurrent attention without making
the executor Qwen-specific.

### 4. WeightBindings separate logical weights from physical payloads

`WeightBindings` maps each graph logical weight ID to one or more physical
payload descriptions. A physical description includes package tensor location,
shape, storage dtype, quantization format and scale metadata, checksums, and
any backend-usable layout metadata. It may represent a single tensor, a fused
group, a passthrough tensor, or an overlay.

Graph nodes refer only to logical IDs. Therefore changing AQ4_0 to SQ8_0,
materializing a weight, or adding a hardware-native packed layout updates a
binding and backend plan rather than model topology. The binding validator MUST
prove shape, format, and required auxiliary-payload compatibility before device
allocation.

### 5. StateSchema describes logical request state

`StateSchema` is an immutable list of state entries referenced by graph nodes.
Each entry declares its state kind, logical shape, numerical format, ownership
scope, initialization, mutation, commit, reset, and optional
snapshot/restore contract. Initial state kinds are:

- paged KV cache;
- recurrent state;
- convolution history; and
- position and cache-length state.

The schema may declare block size, sliding-window behavior, per-layer shape,
and compatibility constraints, but it does not contain a physical GPU pointer.
Rust owns opaque state handles, request and layer lifetime, block tables, and
their allocation/release transaction. The backend owns the physical KV,
recurrent, convolution, and workspace payloads addressed by those handles.
This generalizes ADR 0003 without moving backend payload ownership into Rust.

### 6. ExecutionBatch is the only batch-planner to executor input

`ExecutionBatch` represents one prepared, compatible unit of work. It MUST
contain:

- execution phase;
- request IDs;
- packed-token or hidden-value offsets and lengths;
- per-request prefix length and absolute positions;
- state handles and, where applicable, block tables;
- source and destination chunk ranges;
- a graph/state/format/backend compatibility key; and
- a commit nonce that identifies the scheduler and state version validated
  before execution.

It may additionally carry padding/mask metadata, shape-bucket identity,
workspace reservation, cancellation observations, and trace IDs. The planner
first groups compatible requests and chooses a bounded chunk. The executor
performs graph lowering and backend work without advancing visible request
state. Only a successful commit whose nonce still matches may advance scheduler
prefill/decode counters and make KV, recurrent, convolution, position, RNG, or
sampling changes visible. Cancellation and errors discard the uncommitted
batch or follow the existing reset path.

The first implementation MAY require a rectangular batch with a common chunk
width. It MUST retain offsets, lengths, and prefix lengths in the contract so
that length buckets and ragged batches can be added without another model API.

### 7. Validate graph, state, and workspace before GPU allocation

The adapter and planner MUST validate graph topology, value shapes,
WeightBindings, StateSchema, request block capacity, compatibility key, and a
bounded workspace estimate before allocating GPU payloads or submitting work.
The estimate includes resident state, per-batch activation buffers, temporary
attention storage, materialized weights, and required safety headroom.

The planner MUST reject or split an oversized batch before allocation. It MUST
not construct a full attention matrix for long prefill solely to satisfy this
contract; long contexts require streamed or chunked execution. An expected
capacity shortage is reported as an explicit admission/planning outcome. An
unexpected allocation failure is recorded as OOM and must not be silently
replaced by a smaller successful batch in the same evidence row.

### 8. Backend operation selection is downstream of graph lowering

Lowering selects operations by semantic operator, phase, input/output layout,
weight/activation/state numerical format, state layout, shape bucket, and
backend capability. A backend registry supplies candidate implementations,
capability probes, workspace estimators, and fallback chains. The registry may
reuse the existing operation/phase/format/model/GPU matching mechanism, but it
MUST register runnable operations rather than only static descriptor names.

The generic executor owns buffer lifetime, event ordering, backend fallback,
state prepare/commit/reset, cancellation propagation, and execution tracing.
It is the common path used by the resident `InferenceSession`, worker, gateway,
and OpenWebUI; a component-only invocation cannot satisfy production evidence.

## Rejected alternatives

### Add a Qwen3.5-specific batched prefill loop

Rejected because it would duplicate request-state ownership, chunk policy,
rollback, backend selection, and validation. It would also leave Qwen3/SQ8 and
future hybrid models on different execution contracts.

### Encode topology in quantization or backend identifiers

Rejected because AQ4_0 and SQ8_0 are numerical-format choices, while dense,
recurrent, and hybrid attention are model semantics. Combining them prevents a
single graph from evaluating multiple formats and makes fallback selection
ambiguous.

### Make ModelAdapter a runtime plugin that owns execution

Rejected because it reintroduces backend allocation, scheduler mutation, and
model-ID branching in every adapter. Adapters declare; the generic executor
plans, lowers, executes, and commits.

### Keep only per-request fixed resident state

Rejected because it cannot express request buckets, independent block tables,
or a common commit boundary for batched prefill. Per-request state remains a
valid backend payload layout, but it must be reachable through `StateSchema`
and opaque state handles.

### Require graph metadata in every existing package immediately

Rejected because existing packages must remain usable during migration. Legacy
adapters derive the graph and bindings from validated tensor metadata; a future
package schema may provide an equivalent serialized graph.

## Consequences

- New model support begins with an adapter, graph/state validation, and
  bindings rather than a new serving loop.
- New numerical formats normally change bindings and backend registry entries,
  not public API, worker protocol, or graph topology.
- Backend work must expose capability and workspace information early enough
  for admission control, and must preserve the Rust/C++ ownership split of ADR
  0002.
- Scheduler evolution must add prepared multi-request prefill commit support;
  its existing request IDs and block allocations remain the source of lifetime
  truth.
- Existing single-request paths remain valid compatibility implementations but
  cannot be cited as real-batch production prefill until they use an
  `ExecutionBatch` and report the resolved plan.
- Graph, binding, state-schema, compatibility-key, executor, and fallback
  identities become required execution-trace fields.

## Migration

1. P0 fixes this ADR and companion registry/evidence contracts without changing
   package payloads or serving behavior.
2. P1 implements Rust types and a CPU reference executor for the initial node
   vocabulary. It introduces graph, binding, state, and workspace validation
   before device allocation.
3. A Qwen3 dense adapter derives a graph and paged-KV schema from current
   package conventions. A Qwen3.5 hybrid adapter derives a graph containing
   dense-attention and recurrent-attention nodes in manifest order. Both must
   use the same generic executor API.
4. Existing packages continue through adapter derivation. A later package
   schema may serialize graph/state metadata and digest it; the loader must
   validate equivalence with the adapter contract during the transition.
5. The first production vertical slice uses a bounded rectangular prefill batch
   and a fallback single-request plan. It connects the selected graph executor
   to the resident session before any performance claim. Ragged chunks,
   sliding-window state, MoE, and new backend operators follow incrementally.

## Validation

P0 review and subsequent implementation MUST demonstrate all of the following:

1. A Qwen3 dense decoder and a Qwen3.5 hybrid decoder are expressible with the
   same `ModelGraph`, `WeightBindings`, `StateSchema`, and `ExecutionBatch`
   types. The difference is node sequence and attributes, not an executor
   model-ID branch.
2. Qwen3 dense attention declares paged-KV and position state. Qwen3.5 hybrid
   execution declares paged-KV/position state only for dense-attention layers
   and recurrent/conv state for the applicable layers.
3. Binding validation rejects missing, duplicate, shape-incompatible, or
   format-incompatible physical tensors before any GPU allocation.
4. Batch validation rejects incompatible graph, state layout, numerical format,
   backend capability, stale block table, stale prefix position, invalid chunk
   range, invalid nonce, and insufficient workspace before execution.
5. A failed or cancelled batch changes neither scheduler-visible progress nor
   committed backend state. A successful batch advances every included request
   exactly once through the commit nonce.
6. CPU reference and selected backend paths compare graph outputs and state
   transitions on small dense and hybrid fixtures. Long-context tests use
   chunked/streamed inputs and record planned versus observed workspace and
   state bytes.
7. Production traces prove whether an execution was component, full-model, or
   resident-server scope, and identify the resolved graph/backend plan.

## References

- `docs/plans/generic-production-inference-optimization-plan-v0.1.md`
- `docs/decisions/0002-inference-engine-language-boundary.md`
- `docs/decisions/0003-kv-cache-block-layout.md`
- `docs/decisions/0005-backend-operation-registry.md`
- `docs/specs/production-execution-trace-v0.1.md`
- `docs/specs/prefill-validation-v0.1.md`
- `crates/ullm-engine/src/qwen35_aq4_model_runtime.rs`
- `crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs`
- `crates/ullm-engine/src/qwen35_package_contract.rs`
- `crates/ullm-engine/src/scheduler.rs`
