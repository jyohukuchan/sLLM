# ADR 0005: Backend operation registry

## Status

Accepted for staged implementation under the generic production inference optimization plan.

## Context

uLLM already has reusable runtime operations for AQ4 and SQ8 projections,
batched causal attention, cached-prefix attention, and recurrent preparation.
It also has a small dispatch matcher that considers operation, phase, format,
model architecture, GPU architecture, GPU name, and priority. That matcher is
currently a static descriptor catalog: it does not prove that an entry is
runnable, estimate its workspace, express its transaction behavior, or record
why a fallback was selected.

The generic model graph and state schema boundary in ADR 0004 requires a
different contract. Graph lowering must ask for a semantic operation with a
known layout and state contract; it must not choose a Qwen-, AQ4-, SQ8-, or
GPU-specific kernel in the executor. The batch planner must decide whether a
plan fits memory before device allocation, and stateful operations must not
make a partially executed request visible.

ADR 0002 keeps backend memory and kernels behind the runtime ABI. ADR 0003 and
ADR 0004 keep Rust responsible for request lifetime, block tables, state
handles, and commit. This ADR defines the typed registry that connects a
lowered operation to an executable backend implementation without weakening
those ownership rules.

## Decision

### 1. Registry role and ownership

`BackendOperationRegistry` is an immutable, typed catalog of runnable backend
implementations. It is consulted by the batch planner during graph lowering.
The planner resolves every operation, implementation ID, workspace estimate,
fallback chain, and transaction contract before execution begins. The generic
executor receives that resolved plan and MUST NOT repeat selection with an
ad-hoc branch or environment-variable decision.

The registry describes implementations; it does not contain model topology,
package tensor names, scheduler policy, public model identity, or request
lifetime. `ModelGraph`, `WeightBindings`, `StateSchema`, and
`ExecutionBatch` remain the owners defined by ADR 0004.

### 2. Registry key

An operation lookup key MUST include the following typed fields:

```text
OpKind
+ Phase
+ input and output layouts
+ weight, activation, and state format
+ state layout
+ shape bucket
+ backend and GPU capability
+ optional architecture constraint
```

`OpKind` is a semantic graph operation, not a kernel name. `Phase` distinguishes
cold prefill, cached-prefix prefill, decode, and future phases. Layout fields
include rank, logical axes, strides or packed representation, and any ragged or
mask contract required by the operation. Numerical-format fields distinguish
storage and compute representation. State layout identifies, for example,
paged-KV block semantics, recurrent state, or convolution history.

The optional architecture constraint can select a compatible specialized
implementation, but model ID, public model name, package path, and tokenizer
identity MUST NOT be required key fields. A specialized entry is eligible only
when it implements exactly the requested semantic operation and layout
contract. An architecture constraint cannot redefine those semantics.

### 3. Runnable implementation descriptor

Every registry entry MUST be a runnable `ImplementationDescriptor`, not a
string-only catalog record. Its stable fields are:

- immutable implementation ID and semantic version;
- semantic `OpKind`, supported phases, layouts, numerical formats, state
  layouts, and shape buckets;
- capability predicate over backend, device, GPU architecture, concrete GPU,
  ABI version, and required runtime features;
- workspace estimator that returns an exact requirement or a checked upper
  bound split into persistent and temporary bytes;
- typed execute entry that accepts the lowered operation and reserved buffers;
- correctness and promotion status;
- selection priority and declared fallback compatibility;
- determinism, external side-effect, and state prepare/execute/commit
  properties; and
- trace metadata sufficient to identify the binary/runtime build and resolved
  implementation.

The execute entry may reach the C ABI through `ullm-runtime-sys`, a CPU
reference implementation, or a future backend ABI. It MUST receive buffers and
state handles supplied by the resolved plan; it MUST NOT allocate unbounded
workspace, inspect a model name, or independently select another kernel.

Promotion status is distinct from capability. An entry may be compiled and
capable but remain reference-only, diagnostic-only, or not promotion-approved.
Only an implementation with the required correctness and promotion status is
eligible for a production plan.

### 4. Separate graph lowering from kernel selection

Graph lowering converts semantic nodes and an `ExecutionBatch` into typed
lowered operations with concrete shapes, layouts, logical weights, state
handles, and commit requirements. It does not inspect GPU-specific kernel
names. Registry selection then resolves candidate descriptors against that
lowered operation and device capability.

The batch planner performs both lowering and selection while preparing the
batch. It reserves the selected workspace and emits a plan containing ordered
execute entries and their declared transaction boundaries. The executor runs
only that plan. It may fail an entry, but it must return to the declared
fallback or fail-closed path rather than silently re-run selection.

### 5. Selection, fallback, and fail-closed rules

Selection proceeds in this order:

1. Reject descriptors whose semantic operation, phase, layout, numerical
   format, state layout, shape bucket, ABI, or capability predicate does not
   match.
2. Reject descriptors whose workspace estimate exceeds the admitted budget or
   whose promotion status is insufficient for the requested scope.
3. Rank the remaining entries by specificity and then declared priority.
4. If the highest rank remains tied, report an ambiguous selection error. The
   registry MUST NOT depend on declaration order to break a tie.
5. Select the resulting entry, or use a declared compatible generic fallback.

Every semantic operation supported in production MUST have a generic fallback
for each supported backend/format contract, unless the served-model contract
explicitly marks the operation unsupported. Architecture-specific overrides
MUST name that fallback and prove semantic, layout, numerical, and state
transaction compatibility with it.

`Unsupported` means no descriptor satisfies the requested contract on the
admitted backend; it is a normal, explicit outcome and is recorded in the
trace. `Fallback` means the selected primary is unavailable before execution
and a declared compatible fallback has been selected; the trace MUST record the
primary ID, fallback ID, and reason. `FailClosed` is required when no compatible
fallback exists, selection is ambiguous, capability or ABI validation fails,
the workspace admission is insufficient, the selected implementation loses
promotion eligibility, or a stateful execution fails after preparation.

There is no silent fallback. Environment variables may enable a required-kernel
guard, diagnostics, or test fault injection, but they MUST NOT be the sole
source of operation selection. The resolved implementation and every fallback
decision are written to the production execution trace before state commit.

### 6. Stateful operation transaction contract

Every descriptor declares one of these side-effect classes:

- pure/read-only;
- output-only with no request state mutation; or
- transactional stateful.

A transactional stateful operation MUST expose either separate `prepare`,
`execute`, and `commit` entries or an equivalent runtime transaction contract.
`prepare` validates state handles, positions, block tables, capacity, and the
commit nonce without making request-visible changes. `execute` writes only
reserved temporary or pending state. `commit` publishes all affected state
only after the enclosing batch succeeds and its nonce remains current. Reset or
abort discards pending state.

An operation that mutates paged KV, recurrent state, convolution history,
position, RNG, or sampling state cannot be registered as pure. A kernel that
cannot meet this contract is restricted to a diagnostic scope until it gains a
transaction-compatible wrapper. Partial state updates must never become
visible after a failed batch or cancellation.

### 7. Workspace and admission

Workspace estimation occurs before GPU allocation and before scheduler state
commit. Descriptors classify bytes as:

- persistent resident bytes, including weights and long-lived state;
- persistent per-request state bytes;
- temporary per-batch activation and staging bytes; and
- temporary operator workspace bytes.

An estimator returns either an exact value or a checked upper bound for the
resolved shape bucket. The planner aggregates those values, required alignment,
and configured safety headroom against currently reserved device memory. It
must reject, split, or defer a batch before allocation if the budget is
insufficient. Admission failure performs no allocation and no state mutation.

The registry does not permit an implementation to allocate a full attention
matrix merely because it was admitted for a long prompt. Long prefill requires
the declared streamed or chunked layout. Unexpected allocation failure is an
OOM execution outcome, is recorded with estimate and observed request, and is
not hidden by retrying a smaller batch under the same evidence identity.

### 8. Registration lifecycle and backend extension

The initial registry uses compile-time registration in Rust and a versioned
runtime ABI. Descriptor IDs are unique within a registry semantic version.
Startup validates duplicate IDs, supported ABI versions, descriptor schema,
capability probes, fallback references, and generic-fallback coverage before a
served model becomes ready.

A future CPU, HIP, CUDA, or plugin backend supplies descriptors and executable
entries through a versioned registration boundary. A plugin must declare its
registry schema version, runtime ABI version range, capability probe, supported
formats/layouts, workspace estimator, transaction properties, and promotion
status. The host rejects an incompatible or incomplete plugin before it can
participate in selection. Runtime ABI changes follow ADR 0002 and require an
explicit compatible version range.

Static immutable tables are acceptable for compile-time registration. A static
global mutable registry is not. The registry is constructed, validated, and
frozen before serving; request execution cannot add, remove, or mutate entries.

## Rejected alternatives

### Keep a string-only descriptor catalog

Rejected because an ID, operation, and priority cannot prove executability,
workspace safety, fallback equivalence, promotion status, or state behavior.
The current projection catalog is useful migration input but is insufficient as
the production registry.

### Let the executor choose kernels with `if` branches or environment variables

Rejected because it makes production behavior impossible to plan, test, or
trace consistently. It also permits worker-specific drift after admission.

### Require model ID in every registry key

Rejected because it makes numerical formats and model topology inseparable and
duplicates generic operators for each model. Optional architecture constraints
remain available for semantically compatible optimizations.

### Fall back whenever an implementation returns an error

Rejected because an error after a stateful kernel has begun may have changed
pending state or exposed a runtime failure. Fallback is allowed only before
execution or through a transaction contract that proves rollback; otherwise the
batch fails closed.

### Use a process-global mutable plugin registry

Rejected because concurrent mutation creates non-deterministic selection,
unvalidated duplicate IDs, and request-to-request behavior changes. The initial
registry is compile-time and frozen; later plugins use validated startup
registration.

## Consequences

- A production plan has a reproducible implementation and workspace decision
  before any GPU work starts.
- Backend optimization work must supply a descriptor, capability probe,
  estimator, transaction contract, and validation evidence rather than only a
  new kernel symbol.
- CPU reference implementations become first-class generic fallbacks and
  correctness baselines, although a served-model contract may explicitly mark
  them unsupported for a performance scope.
- Production traces distinguish unsupported hardware, planned fallback,
  admission rejection, runtime failure, and OOM.
- Existing environment guards remain useful checks, but selection moves to the
  typed resolved plan.

## Migration

1. Preserve `backend_dispatch.rs` as a compatibility matcher while introducing
   typed semantic `OpKind`, phase, layout, format, capability, workspace, and
   transaction fields beside it.
2. Wrap current CPU and HIP runtime operations exposed by `ullm-runtime-sys` as
   runnable descriptors. Begin with AQ4 projection/batched projection, SQ8
   projection, causal attention, cached-prefix attention, and recurrent
   preparation.
3. Translate existing RDNA4 and R9700 priority rules into capability predicates
   and explicit priorities. Keep CPU and HIP generic fallbacks where their
   operation contracts are supported.
4. Migrate AQ4 and SQ8 dispatch call sites to request a lowered semantic
   operation. Retain their current implementation IDs as stable migration IDs
   and record whether a descriptor is reference, diagnostic, or production
   approved.
5. Move the existing static projection and fused-operation catalogs into
   compile-time registrations only after each entry has an execute target,
   workspace estimator, and fallback contract. Planned-but-unrunnable entries
   remain outside the active registry.
6. Add plugin registration only after the compile-time registry, trace format,
   and ABI validation tests are stable.

## Validation

The registry implementation and every new backend registration MUST test:

1. specificity and declared priority choose the expected descriptor;
2. an equal highest specificity/priority pair is rejected as ambiguous;
3. a model-independent generic fallback is selected when an optional
   architecture-specific override is unavailable;
4. unsupported capability and missing fallback return explicit unsupported or
   fail-closed outcomes without executing a kernel;
5. checked workspace arithmetic rejects overflow, insufficient headroom, and
   admission failure before allocation;
6. execution traces contain selected ID/version, capability summary, workspace
   estimate, promotion status, fallback ID/reason, and terminal outcome;
7. a transactional stateful fake operation commits every request exactly once
   on success and publishes no partial state on execute failure, stale nonce,
   or cancellation; and
8. duplicate stable IDs, invalid fallback references, ABI mismatch, and
   incomplete capability declarations reject registration at startup.

CPU reference, HIP, RDNA4, R9700, AQ4, and SQ8 migration tests additionally
verify that a descriptor is selected only for its declared format, layout, and
capability, never solely because a model ID or environment variable happens to
match.

## References

- `docs/plans/generic-production-inference-optimization-plan-v0.1.md`
- `docs/decisions/0002-inference-engine-language-boundary.md`
- `docs/decisions/0003-kv-cache-block-layout.md`
- `docs/decisions/0004-model-graph-and-state-schema.md`
- `docs/specs/production-execution-trace-v0.1.md`
- `docs/specs/prefill-validation-v0.1.md`
- `crates/ullm-engine/src/backend_dispatch.rs`
- `crates/ullm-engine/src/aq4_package_runtime.rs`
- `crates/ullm-engine/src/sq8_stack_runtime.rs`
- `crates/ullm-runtime-sys/src/lib.rs`
- `crates/ullm-runtime-sys/src/lib_parts/part_00.rs`
