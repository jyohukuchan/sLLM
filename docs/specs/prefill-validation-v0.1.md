# Prefill Validation Evidence v0.1

Status: Accepted as the P0 validation contract for staged implementation; no performance implementation is implied.

## 1. Purpose

This document defines the evidence contract for validating a prefill
optimization from an operator experiment through a production OpenWebUI request.
It is deliberately independent of a model architecture, quantization format, or
backend implementation.

The contract has four goals:

- preserve an all-M=1 path as the implementation/path oracle;
- compare a selected optimized path with an independently captured source oracle;
- prevent component-only measurements from being presented as full-model or
  product performance; and
- bind a case, a result, and a production execution trace by hashes so a
  promotion decision is reproducible.

The schema for a validation result is `ullm.prefill_validation.v1`.
It supplements and never replaces
`docs/specs/inference-benchmark-result-v0.1.md`. Throughput rows remain in the
existing JSONL schema; this artifact links to those rows and adds the
correctness, trace, policy, and promotion decision that the generic benchmark
schema intentionally does not define.

This specification is the P0 evidence contract for
`docs/plans/generic-production-inference-optimization-plan-v0.1.md`.

## 2. Normative language and terms

The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

- **M**: the number of new prompt tokens in one prefill execution unit. M is
  token parallelism inside a request; it is not the number of concurrent
  requests.
- **all-M=1**: a complete prompt processed one token at a time through the
  maintained reference path.
- **cold prefill**: prompt processing with no cached prefix for the request.
- **cached-prefix prefill**: processing new prompt tokens while retaining and
  attending to an already committed prefix.
- **source oracle**: a separately captured reference result from the declared
  source model/runtime. It is not produced by the candidate executor.
- **path oracle**: the all-M=1 result from the same artifact and model
  configuration as the candidate.
- **promotion**: allowing an implementation to be selected for the served
  product. A component or diagnostic result cannot promote an implementation.

## 3. Trust boundary

The producer may execute a candidate and write raw evidence, but its
`passed`, `verified`, `promotion_eligible`, or similarly named conclusion is
not authoritative. An independent validator MUST reconstruct every acceptance
decision from the files and hashes below.

The validator MUST at least:

1. reject duplicate JSON keys, non-finite numbers, symlinks where a regular
   file is required, path escape, and file/hash mismatches;
2. verify every declared case, result, trace, oracle, policy, binary, manifest,
   package, and artifact identity before using a derived metric;
3. recompute numerical, token, scheduler, cache/state, percentile, and
   regression decisions from raw evidence rather than trusting producer
   summaries; and
4. reject an input whose trace, result, case, scope, model, format, backend,
   or power-condition identity differs from the declared comparison baseline.

The existing SQ8 oracle trust rules remain authoritative for its frozen
fixtures: `docs/specs/sq8-serving-oracle-v0.1.md` and
`tools/validate-sq8-serving-runtime-oracle.py` validate payload identities and
metrics independently of an exporter. This specification generalizes the
linkage pattern; it does not relax those rules.

## 4. Evidence layout and publication

Suggested run layout:

```text
benchmarks/results/YYYY-MM-DD/<run-id>/prefill-validation/
  cases/<case-id>.case.json
  results/<case-id>.result.json
  traces/<trace-id>.json
  raw/<case-id>/...
  policies/<policy-id>.json
  oracles/<oracle-id>/...
  SHA256SUMS
```

- A case manifest is immutable input describing the planned matrix cell.
- A result artifact has schema `ullm.prefill_validation.v1` and references the
  case manifest, benchmark row file, raw captures, policies, oracles, and
  execution trace by SHA-256.
- A production trace is the artifact defined by
  `docs/specs/production-execution-trace-v0.1.md`. Until that specification is
  available, producers MUST retain the complete raw resolved-executor evidence
  needed to create it; they MUST NOT claim production promotion.
- Existing JSONL benchmark files keep their original schema and append-only
  behavior. `record_sha256` is the SHA-256 of the exact UTF-8 JSONL line,
  excluding its terminating LF. `file_sha256` is the SHA-256 of the complete
  JSONL file captured by the result.

Producers MUST write a file to a sibling `.incomplete` path, flush and fsync it,
then atomically rename it only after the complete scheduled evidence is present.
An incomplete, missing, or extra scheduled artifact makes the run ineligible.
Existing evidence MUST NOT be overwritten.

## 5. Case identity

A case identity is the tuple below. It MUST be canonical JSON in a case
manifest, and the result MUST contain its SHA-256.

```text
model identity + tokenizer identity + product/package/artifact identity
+ format ID + implementation ID + backend/GPU capability identity
+ build identity + power-condition identity
+ scope + phase + baseline mode + requested/resolved M
+ prompt/context/decode-start/request shape + sampling contract
+ policy identity + source-oracle identity
```

`case_id` is a stable, human-readable label. It is not sufficient as an
identity by itself. A changed model revision, runner binary, backend driver,
requested implementation, resolved fallback, power condition, or source oracle
creates a different case.

## 6. Complete result example

The following is a complete illustrative `ullm.prefill_validation.v1` result.
The hashes are placeholders. A real result MUST use regular-file paths within
the run root and lowercase 64-character SHA-256 values.

```json
{
  "schema_version": "ullm.prefill_validation.v1",
  "run_id": "2026-07-12-qwen35-aq4-prefill-m16-r9700",
  "case_id": "qwen35-9b-aq4-r9700-cold-p1024-m16",
  "status": "ok",
  "scope": "full_model",
  "case": {
    "path": "../cases/qwen35-9b-aq4-r9700-cold-p1024-m16.case.json",
    "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  },
  "identity": {
    "model": {
      "family": "Qwen3.5",
      "name": "Qwen3.5-9B",
      "revision": "aq4-cli-compat-v0.1",
      "tokenizer_sha256": "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "context_limit": 4096
    },
    "format": {
      "format_id": "AQ4_0",
      "implementation_id": "qwen35_aq4_rdna4_prefill_m16_v1",
      "kv_cache_dtype": "f32"
    },
    "backend": {
      "name": "hip",
      "gpu_architecture": "RDNA4",
      "gpu_name": "Radeon AI PRO R9700",
      "device_identity_sha256": "2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    },
    "build": {
      "git_commit": "0123456789abcdef0123456789abcdef01234567",
      "worktree_clean": true,
      "binary_sha256": "3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "served_model_manifest_sha256": "4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "package_manifest_sha256": "5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "artifact_content_sha256": null
    },
    "power_condition": {
      "policy_id": "r9700-default-power-v1",
      "capture_path": "../raw/environment/power.json",
      "capture_sha256": "6123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  },
  "workload": {
    "phase": "cold_prefill",
    "baseline_mode": "cold_batched",
    "requested_chunk_tokens": 16,
    "resolved_chunk_tokens": 16,
    "prompt_tokens_per_request": [1024],
    "cached_prefix_tokens_per_request": [0],
    "context_tokens_after_prefill_per_request": [1024],
    "decode_start_tokens_per_request": [1024],
    "generated_tokens_per_request": [64],
    "request_count": 1,
    "concurrent_requests": 1,
    "sampling": {
      "mode": "greedy",
      "temperature": 0.0,
      "top_p": 1.0,
      "top_k": 1,
      "seed": 0
    }
  },
  "evidence": {
    "inference_benchmark_result": {
      "schema_version": "inference-benchmark-result-v0.1",
      "path": "../benchmark.jsonl",
      "file_sha256": "7123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "case_id": "qwen35-9b-aq4-r9700-cold-p1024-m16",
      "record_sha256": "8123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    },
    "execution_trace": {
      "schema_version": "ullm.production_execution_trace.v1",
      "trace_id": "trace-qwen35-9b-aq4-r9700-p1024-m16",
      "path": "../traces/trace-qwen35-9b-aq4-r9700-p1024-m16.json",
      "sha256": "9123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "scope": "full_model"
    },
    "raw_evidence": [
      {
        "role": "timing_samples",
        "path": "../raw/qwen35-9b-aq4-r9700-cold-p1024-m16/timing.json",
        "sha256": "a123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
      },
      {
        "role": "state_snapshots",
        "path": "../raw/qwen35-9b-aq4-r9700-cold-p1024-m16/state.json",
        "sha256": "b123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
      }
    ]
  },
  "oracles": {
    "path_oracle": {
      "mode": "all_m1",
      "result_path": "../results/qwen35-9b-aq4-r9700-cold-p1024-m1.result.json",
      "result_sha256": "c123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    },
    "source_oracle": {
      "oracle_id": "qwen35-9b-bf16-source-v1",
      "path": "../oracles/qwen35-9b-bf16-source-v1/metadata.json",
      "sha256": "d123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "independent": true
    },
    "threshold_policy": {
      "policy_id": "qwen35-aq4-r9700-correctness-v1",
      "path": "../policies/qwen35-aq4-r9700-correctness-v1.json",
      "sha256": "e123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  },
  "correctness": {
    "finite": true,
    "shape_contract_passed": true,
    "path_oracle_passed": true,
    "source_oracle_passed": true,
    "final_hidden": {
      "relative_l2": 0.001,
      "cosine_similarity": 0.9999,
      "max_abs": 0.01
    },
    "logits": {
      "relative_l2": 0.002,
      "cosine_similarity": 0.9998,
      "max_abs": 0.02,
      "top_1_exact": true,
      "top_k": 10,
      "top_k_overlap": 10
    },
    "greedy_tokens_exact": true,
    "kv_state_cache_passed": true,
    "scheduler_progress_passed": true,
    "chunk_equivalence_passed": true,
    "cancel_reset_passed": true,
    "publish_failure_reset_passed": true
  },
  "performance": {
    "warmup_runs": 2,
    "measured_runs": 10,
    "percentile_method": "linear_interpolation_rank_(n-1)*p",
    "prefill_tokens_per_second_p50": 450.0,
    "prefill_tokens_per_second_p95": 430.0,
    "ttft_ms_p50": null,
    "ttft_ms_p95": null,
    "decode_tokens_per_second_p50": 70.0,
    "inter_token_latency_ms_p95": 20.0,
    "end_to_end_tokens_per_second": 200.0,
    "vram_baseline_bytes": 1000,
    "vram_peak_bytes": 2000,
    "workspace_estimate_bytes": 500,
    "workspace_peak_bytes": 480,
    "actual_token_batch_width_p50": 16,
    "actual_request_batch_width_p50": 1,
    "fallback_count": 0,
    "fallback_reasons": []
  },
  "regression": {
    "baseline_result_path": "../results/qwen35-9b-aq4-r9700-cold-p1024-m1.result.json",
    "baseline_result_sha256": "c123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "prefill_p50_change_percent": 600.0,
    "prefill_p95_change_percent": 590.0,
    "vram_limit_bytes": 3000,
    "new_oom": false,
    "passed": true
  },
  "promotion": {
    "eligible": false,
    "reason_codes": ["scope_full_model_not_production_server"],
    "required_next_scope": "production_server"
  },
  "error": null,
  "notes": []
}
```

`status` is one of `ok`, `failed`, `oom`, `unsupported`, or `skipped`.
`unsupported` means a declared capability is absent; `skipped` means the
planned case was intentionally not run; neither may be silently removed from a
matrix. A non-`ok` result MUST retain case, identity, policy, trace-or-absence
reason, and error evidence.

## 7. Execution scopes

Every result and trace has exactly one scope:

| Scope | Meaning | Promotion eligibility |
| --- | --- | --- |
| `component` | One operator or a partial layer boundary. | Never eligible. |
| `full_model` | The complete model graph and request state execute, but not necessarily through the served worker. | Never sufficient by itself. |
| `production_server` | The resident worker used by the served manifest executes the complete request; direct API and OpenWebUI evidence can be attached. | Required for promotion. |

A result MUST NOT be relabeled from `component` to `full_model` or
`production_server` because its numbers were copied into another report. The
trace scope, trace digest, resolved executor, actual token/request widths, and
state lifecycle must prove the same scope. In particular, a logical batch or a
component batch does not prove that a production request used the same batch
implementation.

The referenced production trace MUST use the exact
`ullm.production_execution_trace.v1` contract. Promotion additionally requires
its `status="ok"`, `scope="production_server"`,
`verification.independent_validation.status="valid"`, a complete reset in
`state_commit.reset`, no OOM, and no unexpected, unsupported, or fail-closed
fallback in its `fallback` object. A prefill-validation result cannot replace
any of those trace-level checks with copied summary fields.

## 8. Oracle layers and threshold policy

### 8.1 Layer 0: structural guard

Every scope MUST verify input/output shape, dtype/layout, finite values, token
range, absolute positions, causal visibility, and declared state geometry before
numerical comparison. No numerical tolerance permits a malformed or non-finite
payload.

### 8.2 Layer 1: path oracle

For every optimized case, the matching all-M=1 case is mandatory. It uses the
same model, tokenizer, artifact/package, format, backend, sampling contract,
prompt, context, and generated-token shape. The candidate may differ only in
the declared prefill implementation, resolved chunk plan, and declared
workspace.

The three baseline modes are:

- `all_m1`: mandatory path oracle and fallback;
- `cold_batched`: cold causal prefill with no request prefix; and
- `cached_prefix_chunked`: a nonempty cached prefix plus a new chunk.

`cold_batched` and `cached_prefix_chunked` MUST be compared with `all_m1` at
each supported M. Cached-prefix validation must prove both prefix visibility and
in-chunk causal visibility; it must not reuse an empty-cache-only operation.

### 8.3 Layer 2: independent source oracle

The source oracle is separately captured from the declared source model/runtime,
with its model, tokenizer, generation, device, exporter, and payload identities
anchored outside candidate-controlled summaries. It MUST provide the payloads
needed for the configured gate: at least final hidden state/logits and greedy
tokens for short correctness cases, plus finite/logit-range sanity for long
throughput cases.

### 8.4 Versioned format policy

Numerical tolerances, source-oracle requirements, absolute latency budgets, and
hardware limits belong to a versioned threshold-policy artifact. A policy MUST
state its model/format/backend applicability, source artifact identities,
metric definitions, acceptance thresholds, effective date, and SHA-256.

The Qwen3-14B SQ8 source and chunk thresholds in
`docs/specs/sq8-serving-oracle-v0.1.md` and
`tools/validate-sq8-serving-chunks.py` remain specific to their frozen SQ8
artifact and fixture set. They MUST NOT be applied to Qwen3.5 AQ4, another
SQ8 artifact, another model revision, or another backend unless a policy
explicitly declares that compatibility and is independently reviewed.

## 9. Correctness gates

An `ok` candidate is correctness-eligible only when all applicable checks pass:

1. finite values and exact shape/dtype/layout contracts;
2. final hidden and logits numerical metrics, including relative L2, cosine,
   and maximum absolute error, against both required oracle layers;
3. exact greedy generated tokens through the recorded finish boundary;
4. deterministic top-k ranking and the policy-required top-k agreement;
5. KV, recurrent, convolution, cache length, absolute position, and block-table
   state equivalence at every committed execution unit;
6. scheduler request ownership, prompt progress, execution-unit width, and
   generated-token counters consistent with the trace;
7. chunk-boundary equivalence for all enabled M values, including a nonempty
   cached-prefix case when that mode is enabled;
8. cancellation after each supported boundary, normal reset, EOS/length finish,
   and a subsequent request proving the baseline is restored; and
9. a publish failure after token preparation proving that sampling, scheduler,
   generated-token count, cache/state ownership, and reset behavior remain
   correct.

For production eligibility, these gates must run through the resident session,
not through a recreated CLI-only execution path.

## 10. Performance gates and regression stop

Performance evidence MUST record prefill p50/p95 throughput, TTFT p50/p95 when
network serving is in scope, decode throughput and p95 inter-token latency,
end-to-end throughput/latency, VRAM baseline and peak, estimated and observed
workspace, fallback count/reasons, and actual token/request batch widths from
the execution trace. The validator derives the latter values from trace
`phases`, `operator_resolutions`, `fallback`, and `memory` fields; a benchmark
label or a producer-computed total is not a substitute.

Performance comparisons are valid only when model/tokenizer/product/artifact,
binary, backend/GPU/driver, requested workload, source policy, and declared
power condition match. A baseline with a different identity is diagnostic only.

Unless a stricter approved policy applies, promotion MUST stop when a comparable
cell has any of the following:

- prefill p50 throughput regresses by more than 5 percent;
- prefill p95 throughput regresses by more than 10 percent;
- measured VRAM or workspace exceeds the policy limit;
- a previously non-OOM case becomes `oom`; or
- an undeclared fallback is selected, or the actual batch width does not meet
  the scope's declared minimum.

Model/GPU-specific absolute TTFT, decode, ITL, and memory budgets MUST be in the
hash-bound policy, not inferred from these generic rules.

## 11. Required matrix

The initial matrix is a coverage requirement, not permission to run every cell
concurrently.

| Axis | Required values |
| --- | --- |
| model topology | Qwen3 dense self-attention; Qwen3.5 hybrid recurrent plus self-attention |
| numerical format | AQ4_0; SQ8_0; an available BF16, FP16, or F32 reference |
| prefill M | 1, 8, 16, 32, 64, 128 |
| baseline mode | `all_m1`, `cold_batched`, `cached_prefix_chunked` |
| prompt/context | 1, 8, 32, 128, 512, 1024, 2048, 3584, and the model context-limit edge |
| decode start context | 16, 512, 1024, 1339, 2048, 3584 where within the model limit |
| backend capability | CPU reference; HIP R9700/RDNA4 mandatory; V620/RDNA2 recorded as supported, `unsupported`, or skipped by capability policy |
| scope | component, full_model, production_server |

The M=1 cell is mandatory for every optimization cell. Unsupported V620 cases
MUST be recorded with a capability reason; they do not invalidate an R9700-only
claim, but they cannot support a V620 compatibility claim.

## 12. OOM handling

The validation harness MUST use one GPU inference process at a time. Before an
execution it MUST record a preflight estimate for weights, persistent state, KV,
workspace, temporary buffers, and configured VRAM headroom. A case without
headroom is `skipped` only when the preflight policy explicitly rejects it;
otherwise an allocation failure is `oom`.

Oracle captures and comparisons MUST stream tensors/logits in bounded chunks and
release per-case data before the next case. A harness MUST NOT retain a matrix of
full logits, duplicate the model for each case, or materialize a full attention
matrix merely for validation.

An OOM result is immutable evidence. It MUST NOT be replaced, hidden, or
rewritten as a success by running a smaller prompt, M, context, or batch. The
smaller case is a new case with a new identity.

## 13. Qwen3.5 AQ4 promotion minimum

For Qwen3.5-9B AQ4 on the approved R9700 production profile, promotion from
the current tokenwise resident path requires all generic gates plus:

1. prompt 1011 production prefill is at least 5 times the recorded tokenwise
   baseline of 63.638 tok/s;
2. prompt 2048 production prefill is at least 5 times its recorded tokenwise
   baseline, without OOM;
3. prompt 1024 has a target of at least 1000 tok/s. This is a target, not a
   substitute for the two minimum gates;
4. decode starting at context 1339 improves on the recorded 42.64 tok/s by at
   least 25 percent, or a prefill-only promotion proves no more than 5 percent
   decode regression at that context and records a separate decode plan; and
5. short-context decode p50 does not regress by more than 5 percent.

The policy used for these thresholds MUST bind the baseline artifact hashes and
the exact production identity. The quoted values do not authorize application to
another model, format, GPU, or worker binary.

## 14. Production rollout

Promotion advances only in this order:

1. **Component:** verify one operator or layer boundary against sampled/CPU
   reference. Record `scope=component`; promotion is forbidden.
2. **Full model:** run all-M=1 and M>1 through one complete resident request
   state, including chunk boundaries, state commit, cancellation, publish
   failure, and reset. Record `scope=full_model`; promotion is still forbidden.
3. **Direct worker:** execute the selected served-model manifest through the
   resident worker and record `scope=production_server` trace evidence.
4. **API/SSE:** verify direct non-stream and streaming OpenAI-compatible calls,
   EOS/length/overflow, cancellation/recovery, and exact lifecycle correlation.
5. **OpenWebUI:** verify the same active manifest through OpenWebUI, including
   visible content, Stop-button cancellation, recovery, resource soak, and the
   planned worker failure/restart path.
6. **Canary activation:** activate only after independent validation passes;
   retain the prior manifest and roll back atomically on any failed gate.

The final result MUST reference an execution trace with
`scope=production_server`, `status=ok`, and
`verification.independent_validation.status=valid`; a trace from a component,
offline tool, different worker, or different manifest cannot satisfy this
requirement. The validator must also reconstruct the trace's identity, phase,
fallback, memory, state-commit, and ready/release checks defined in
`docs/specs/production-execution-trace-v0.1.md`.

## 15. Canonical TTFT and production timing

For all new generic production evidence, TTFT uses this single definition:

```text
start = immediately after the synchronous write containing the final request-body byte returns
end   = receipt time of the final raw socket chunk required to parse the first
        SSE data object with a non-empty choices[0].delta.content string
```

Socket reads are not SSE boundaries. The raw chunks are authoritative; no
synthetic spacing may be introduced. The matching gateway
`request_first_token` lifecycle observation MUST be no later than the client
observation on the same monotonic clock. The request then closes only after that
observation and MUST be correlated with cancellation and
`request_released(reset_complete=true)` before the next sample.

Each TTFT cell is exactly two warmups followed by ten measured requests. The
ten measured values use the frozen linear-interpolation percentile method
`linear_interpolation_rank_(n-1)*p`. This is the canonical P0 schedule for new
generic evidence and matches the production release schedule in
`docs/specs/sq8-openwebui-release-v0.1.md`.

`tools/validate-sq8-serving-performance.py` retains its existing SQ8-specific
five-measured-run contract until explicitly migrated. It MUST NOT be silently
used as a validator for this new 2+10 generic schedule or for AQ4 thresholds.

## 16. Independent validation

The validator writes a new validation artifact rather than modifying producer
output. It must retain the hashes of every consumed case, result, trace, policy,
source oracle, raw capture, benchmark row, and baseline. Missing or malformed
links fail closed.

## 17. Compatibility

This specification is compatible with existing evidence as follows:

- `inference-benchmark-result-v0.1` remains the performance-row schema;
- SQ8 source, M8/M32/M128 chunk, session, and OpenWebUI release validators
  remain valid for their frozen contracts;
- existing component rows remain useful diagnostics but are explicitly
  ineligible for production promotion; and
- no existing schema version, field meaning, fixture identity, or historical
  threshold is changed by this document.

Future adapters may emit `ullm.prefill_validation.v1` links for old evidence
only when every required case/result/trace/policy identity can be supplied. An
adapter MUST record unavailable evidence as unavailable; it MUST NOT invent a
production trace or infer a passed gate from a producer summary.

## 18. References

- `docs/plans/generic-production-inference-optimization-plan-v0.1.md`
- `docs/decisions/0004-model-graph-and-state-schema.md`
- `docs/decisions/0005-backend-operation-registry.md`
- `docs/specs/production-execution-trace-v0.1.md`
- `docs/specs/inference-benchmark-result-v0.1.md`
- `docs/specs/sq8-serving-oracle-v0.1.md`
- `docs/specs/sq8-openwebui-release-v0.1.md`
