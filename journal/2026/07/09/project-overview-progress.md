# uLLM project overview and progress

Date: 2026-07-09

## Scope

- Workspace root: `/home/homelab1/coding-local/ultimateLLM`
- GitHub-connected repository: `uLLM-project/`
- Remote: `git@github.com:jyohukuchan/uLLM-project.git`
- Branch state at review time: `main` at `3138206 Add behavioral SQ prompt guard mode`, aligned with `origin/main`.
- Local uncommitted change: `uLLM-project/README.md` only.
- Ignored local-heavy paths include `build/`, `target/`, `reference-src/`, result logs, and pycache directories.

## Project Summary

uLLM is an from-scratch LLM inference engine project. The public direction is broad hardware compatibility with current local focus on AMD RDNA2/RDNA4. The current implementation is a Rust workspace with C++20/HIP runtime pieces.

Main workspace crates:

- `crates/ullm-engine`: runtime-facing engine, package loader, scheduler, Qwen3/Qwen3.5 decoder execution, AQ/SQ runtime support, and many smoke/benchmark CLI commands.
- `crates/ullm-quant`: prototype AQ converter and package builder. It reads safetensors metadata and can write prototype `.ullm.d` package outputs with bounded chunking and optional tensor-level parallelism.
- `crates/ullm-runtime-sys`: Rust FFI wrapper and build bridge for `runtime/`.

Runtime:

- `runtime/include/ullm_runtime.h` exposes the C ABI.
- `runtime/src/*.inc` contains core buffer/stream APIs, primitive kernels, AQ4 kernels, attention kernels, linear-attention kernels, and smoke/probe APIs.
- HIP is loaded dynamically. CPU fallback exists for some paths, but the active development target is local AMD GPUs.

Docs/results:

- `docs/plans/`, `docs/specs/`, and `docs/research/` hold the active design and evidence trail.
- `benchmarks/results/` holds most tracked files. The repo has thousands of benchmark and golden result artifacts.
- `tools/` contains benchmark runners, guard comparators, SQ/AQ artifact builders, and analysis utilities.

## Current Progress

AQ4 prototype:

- Current narrow claim: single-request Qwen3.5-9B AQ4 prototype runs locally on R9700/RDNA4 and V620/RDNA2.
- Controlled v0.3 prompt suite evidence records mean decode around `19.796 tok/s` on R9700 and `15.434 tok/s` on V620, with 6 quality-scored outputs ok and 1 timing probe not evaluated.
- Cross-device prompt-suite guard has generated-token and top-logit agreement across 7 cases.
- Known limits: no production API, no tensor parallel, no continuous batching, greedy-only measured suite, tokenizer handled by Python wrapper, no independent CPU/external final-logits proof yet.

SQ/FP8 work:

- Current main activity is SQ FP8 candidate evaluation on R9700/RDNA4.
- `sq-fp8-w8a16-r9700-v0` is an evaluation artifact boundary, not the final SQ format.
- SQ artifact support includes FP8 E4M3 payloads, f32 scales, row and row-block scale metadata, policy JSON input, and partial runtime materialization/loading.
- Latest direction separates strict exact match from behavioral promotion:
  - strict generated token/text/logit/top1 matching is kept for drift diagnosis;
  - `acceptance_mode=behavioral` is used as the forward gate when candidate output remains usable.
- Current passing diagnostic branch reaches layer23 `k16`; layer23 `q/v` and layer27 `k` variants are stored as failure guards.
- `kup6_gate5_down5` is treated as a six-layer regression subset/reference point, not a full SQ policy.

Performance/evaluation:

- Earlier long `512/256` pre-SQ path was decode dominated around `0.14 tok/s`, mostly due to CPU/chunked lm_head and smoke-only verification overhead, so repeated long decode on that path is low value.
- Later runtime improvements added GPU-resident lm_head paths, breakdowns, resident gate/beta buffers, and async HIP enqueueing.
- Cached-prefix FP8 K/V cache can reduce K/V bytes to about 25 percent of f32; R9700 speed impact depends on prefix length and kernel structure.
- FlashAttention2-style cached-prefix/cold-prefill experiments improved component throughput and are considered sufficient to stop blocking SQ candidate evaluation.
- RDNA4 FP8 WMMA/rocWMMA probes exist and validate FP8 QK probe paths on R9700 while rejecting V620/RDNA2 as expected.

## Code Shape

Important `ullm-engine` modules:

- `scheduler.rs`: request state, KV block allocator, ready decode batch API.
- `decode_runner.rs`: request-owned self-attention and decoder-layer runners.
- `decoder.rs`: Qwen3 self-attention, layer runtime weights, RMSNorm, RoPE, causal attention, MLP/layer helpers.
- `loader.rs`: package tensor loading, passthrough BF16/F32 reads, AQ4 materialization, registry logic.
- `package.rs`: `.ullm.d` package inspection and payload selection.
- `qwen3_loader.rs`: Qwen3 package runtime loading, SQ overlay handling, model stack runner helpers.
- `sq.rs`: SQ FP8 artifact manifest, validation, materialization, compact bytes, FP8 decode.
- `aq.rs`: AQ scale helpers.
- `golden.rs`: golden fixture comparison.
- `main.rs`: large CLI surface for runtime smoke, package smoke, token-id generation/logits, throughput, and guard paths.

## Verified During This Review

- `cargo test -p ullm-engine sq -- --test-threads=1`
  - passed: 4 SQ tests.
- `python3 -m unittest tests.test_compare_package_guards tests.test_sq_candidate_runtime_row tests.test_sq_fp8_overlay_acceptance`
  - passed: 10 tests.

Full workspace tests and GPU smoke commands were not rerun in this review.

## Next Likely Actions

1. Treat `README.md` as an active uncommitted documentation update and decide whether to refine/commit it.
2. Continue SQ FP8 batch throughput and memory comparison using behavioral prompt-suite guard as the promotion gate.
3. Keep strict top1/token/text/logit guards as diagnostics, especially around layer23/layer27 SQ boundary cases.
4. Record SQ candidate rows with compact resident bytes, materialized working-set bytes, materialization time, prefill/decode throughput, and guard bundle results.
5. Avoid rerunning long decode grids until the runtime path or SQ candidate materially changes.
