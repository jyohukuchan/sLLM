# AGENTS.md

## Project

- sLLM is an MIT-licensed LLM inference engine. GPU operations use C++/HIP;
  other backends and the upper layers use Rust.
- `sLLM.md` is the detailed, Git-excluded requirements document.
  `docs/plans/main-plan.md` is the tracked plan for product, architecture,
  compatibility decisions, progress, and unresolved questions.
- Read `docs/plans/main-plan.md` before beginning work. If `sLLM.md` and the
  main plan disagree, ask the user; do not silently merge the difference.

## Authority and proposal policy

- Authority, from highest to lowest, is: the current explicit user
  instruction; `sLLM.md`; this `AGENTS.md`; approved decisions in
  `docs/plans/main-plan.md`; the active task-local plan; and historical facts.
  A lower level or history fact cannot create a blocker.
- An AI proposal is nonblocking. Without explicit user approval, it cannot
  introduce a hard gate, independent review requirement, broad or GPU rerun,
  security boundary, reuse restriction, blocking stage, finer work unit, or
  larger immutable-evidence requirement. Record nonblocking proposals with
  their origin, scope, cost, and expiry when they matter.
- The default mode is trusted-solo-development. External-contribution and
  release policies are separate lanes; inactive rules do not block work.
- The main agent may investigate and implement directly. Delegate only when
  parallelism, isolation, or specialist context is useful; neither delegation
  nor a particular Codex invocation method is a completion gate.
- Future changes to `AGENTS.md` or `sLLM.md` still require explicit user
  confirmation.

## Work lanes and identity

- Draft: run focused relevant tests, allow a dirty local tree, and require
  neither immutable identity nor independent review.
- Integration: run affected checks and one integration review. Re-review only
  the findings that changed; do not start a fresh review at every checkpoint.
- Release/push: use a clean candidate with an immutable final identity and all
  relevant final gates. Use the `push` skill for project-wide review, minimal
  commit organization, and GitHub publication when publication is requested.
- Docs-only: run markdown, link, and consistency checks. Do not add a
  docs-only deployment or docs-only closeout stage.
- Semantic/build identity is separate from a Git commit. Reuse docs-only
  evidence only when source, build inputs, toolchain, model lock, and artifact
  are unchanged and the mapping has been checked.

## Acceptance, findings, and review

- Freeze acceptance criteria before implementation. Correctness and security
  defects may block. New process requirements are follow-ups unless the user
  approves them as acceptance criteria.
- Classify findings as: correctness/security blocker; release evidence;
  process improvement; optional hardening; or style/docs.
- Design review is optional and reserved for high-risk ABI or kernel changes.
  Use one integration review, focused re-review of findings, and one cumulative
  release review. Per-checkpoint fresh review and docs-only closeout are
  abolished.
- Stop new review or verification and replan the same work unit when it is
  rejected twice, review time exceeds implementation time, functional progress
  stops for more than one hour, verification/docs exceed 30% of the work, the
  estimate exceeds 1.5x, or a gate or acceptance criterion changes.
- Tests should include non-aligned values and both sides of relevant boundaries,
  not only powers of two or a single convenient case.

## GPU evidence and deployment

- Before GPU or software compatibility work, read the relevant compatibility
  documents listed below. Synchronize them with the main plan when a supported
  target or toolchain decision changes.
- GPU proof is fail-closed: CPU emulation or fallback, timeout, crash, and zero
  test selection are never GPU PASS. Evidence must name the exact target and
  use a numerical oracle. Immutable evidence is required when relevant at
  integration or release; draft work is not thereby blocked.
- Keep CPU CI to host contracts, tiny numerical oracles, and compile-only
  checks. Do not use CPU CI to claim full-model, GPU-scale, or GPU-kernel
  correctness.
- Deployment smoke and health checks apply only to a deployable service or
  runtime that is in scope. An absent deployment target never blocks a
  library, tool, or documentation push.
- Monitor long-running commands and report their health. Do not terminate a
  progressing process solely because an arbitrary checkpoint clock elapsed.

## External code and provenance

- Consider llama.cpp direct reuse before implementing clean-room code; it is
  allowed under `docs/provenance/README.md`.
- Provenance is required for release/distribution, not as a human-review gate
  or a provenance-only follow-up commit at each checkpoint. A pending import
  commit is acceptable in development and must be resolved for release.
- AI similarity/provenance checking is allowed at integration. vLLM and other
  non-llama sources remain no-copy references. Keep inspection notes separate
  from implementation; separate agents are optional.

## Repository safeguards

- Do not track models, binaries, raw traces/profiles, large model slices, or
  generated artifacts; follow `docs/development/repository-hygiene.md`.
- Do not edit `README.md`; use `README-AI-manuscript.md` instead. New
  `.gitignore` lines are allowed without prior approval, but changing existing
  lines requires approval.
- Never edit `passwords.txt`. Follow `docs/security/credentials.md` for
  credentials and scoped `sudo -n` operations.
- Keep plans under `docs/plans/active` until complete or abandoned, then move
  them to `docs/plans/archive`; put detailed changes in the matching
  `docs/history` partition and link plan/history pairs at their ends.

## Canonical references

- GPU compatibility: `docs/compatibility/gpu.md`,
  `docs/compatibility/amd-gpu.md`, `docs/compatibility/software.md`
- Runtime architecture: `docs/architecture/runtime.md`
- Model locking: `docs/models/model-lock.md`
- OpenAI compatibility: `docs/api/openai-compatibility.md`
- CI and tests: `docs/plans/active/2026/08/1-10/ci-test-strategy.md`
- Provenance: `docs/provenance/README.md`
