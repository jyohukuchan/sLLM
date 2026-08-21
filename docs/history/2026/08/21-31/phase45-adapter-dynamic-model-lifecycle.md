# Phase 45 history: adapter・control vector・dynamic model lifecycle

## Outcome

Phase 45 completed on 2026-08-22 as a host/API/CLI and RDNA GPU integration phase. The
strict `sllm-model-manifest-v1` parser admits only verified offline regular
files with no-symlink, bounded size/digest-race checks before backend or GPU
allocation. LoRA and control-vector contracts are limited to reviewed dense
BF16 Qwen artifacts; base lock, derived plan, target tensor, shape, dtype,
rank/range, artifact digest, canonical order and finite scale are checked
before admission. Unsupported model/dtype, wrong base, missing tensor,
shape/rank/range mismatch, duplicate/order violations and non-finite scales
fail closed.

The registry now owns `unloaded`, `loading`, `ready`, `draining`, `failed`, and
`quarantined` states with alias limits, resident quota, coalesced immutable
identity loads, linearizable leases, idle-only eviction, last-owner drain, and
explicit quarantine clearing. The router resolves aliases to verified model
and adapter/control identity without exposing paths, prompts, artifact bytes or
credentials. Existing Chat, Completions, Responses and Anthropic generation
semantics remain unchanged. The CLI/server expose the same `--models` manifest;
admin actions are alias-only and the `sllm.adapters`/
`sllm.control_vectors` request extension preserves ordered selection.

## Verification

- The Phase 45 machine profile, Draft 2020-12 schema, dependency-free validator,
  and mutation/duplicate/non-finite JSON tests pass and are registered in the
  H0 CI suite.
- Focused host tests cover adapter/control contracts and BroadcastAdd shape,
  stride, range, finite-value and disabled-identity behavior; model manifest,
  lifecycle, dynamic router, CLI/server manifest, alias-only admin, registry
  lease/drain/quota/LRU/quarantine, and cross-list duplicate rejection paths
  are covered by the affected Rust/API tests.
- The cumulative host checks recorded for this closeout include 57 `sllm-cli`
  tests, 10 `sllm-server` binary tests, 24 Phase 43 API contract tests,
  warning-free clippy for the affected CLI/server binaries, and the Phase 45
  profile validator plus 9 Python mutation/contract tests.
- Exact final release builds passed on `gfx1030` (V620, 16,588 ms) and `gfx1201`
  (R9700, 18,001 ms). Qwen BF16 disabled/LoRA/control/combined cases were each
  bitwise-identical across two runs. Both targets were HIP-only with fallback
  false, resident `8,411,592,192` bytes, request/workspace baseline restored,
  pre/final allocations 0, retryable/quarantine 0. BroadcastAdd standalone
  (`M=1/3`, `H=17`, mismatch 0, cleanup PASS) also passed on both targets.
- Compact identity prefixes, dispatch counts (492/497/495/500), elapsed values,
  and the no-raw-artifact policy are recorded in the
  [GPU summary](../../../../../ci/matrix/phase45-adapter-lifecycle-gpu-summary-v1.json).
  `gfx942`/MI300X runtime remains deferred until a fresh VM/session; no compile
  or host evidence is used to promote that target.
- No llama.cpp source was directly reused. The pinned `b10453` revision is a
  behavior/reference comparison only.

## Known limitations and handoff

- The RDNA full-model adapter/control smoke evidence is fixed in the compact
  summary; raw model/output artifacts are intentionally not tracked.
- MI300X Phase 37/38 optimization resumes only after a fresh exact runtime and
  baseline. Phase 46 conversion/quantization/benchmark tools, Phase 47
  tool/MCP execution, and Phase 48 WebUI remain later scopes.

## References

- [Archived Phase plan](../../../../plans/archive/2026/08/21-31/phase45-adapter-dynamic-model-lifecycle.md)
- [Machine profile](../../../../../tests/fixtures/phase45_adapter_lifecycle_v1.json)
- [Main plan](../../../../plans/main-plan.md)
- [Phase 37+ roadmap](../../../../plans/active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)
- [Runtime architecture](../../../../architecture/runtime.md)
- [Model lock](../../../../models/model-lock.md)
- [OpenAI compatibility](../../../../api/openai-compatibility.md)
- [Credentials](../../../../security/credentials.md)
