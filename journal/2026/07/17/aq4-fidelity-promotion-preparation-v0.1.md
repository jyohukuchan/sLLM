# AQ4 fidelity-fix promotion preparation v0.1

## Result

Prepared the CPU-only portion of the AQ4 fidelity-fix promotion at
f1a3cf4c86978b3b8900396a0b6a8caff90b97f1. No systemd, Docker, sudo,
activation, GPU, R9700 lock, or V620 action was performed.

The current active manifest was read and validated without modification:

- path: /etc/ullm/served-models/active.json
- schema/protocol: ullm.served_model.v2 / ullm.worker.v2
- public model: ullm-qwen3.5-9b-aq4
- active source: ae8b2bb7c2735f4dc761773957bf45f470dd5a8c
- active manifest SHA-256:
  feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44
- active worker SHA-256:
  177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d

## Prepared artifacts

- Clean detached source worktree:
  /home/homelab1/coding-local/ultimateLLM/uLLM-aq4-fidelity-promotion-source-f1a3cf4c
- Candidate worker, mode 0555 / nlink 1:
  /home/homelab1/coding-local/ultimateLLM/uLLM-aq4-fidelity-promotion-release-f1a3cf4c/ullm-aq4-worker
  SHA-256 1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b
- Legacy engine used only by the resident promotion evidence runner, mode 0555
  / nlink 1:
  /home/homelab1/coding-local/ultimateLLM/uLLM-aq4-fidelity-promotion-release-f1a3cf4c/ullm-engine
  SHA-256 d1c18362c6253294d37e7258434d877752c5052ab677ecfd35f1a7928b64b433
- New AQ4-only profile:
  deploy/served-models/qwen35-9b-aq4-reasoning-f1a3cf4c.profile.json

The candidate manifest and release bundle were not fabricated. The generator
was invoked once before a receipt existed and failed closed; it left no
candidate output behind.

## Verification

- Detached release builds succeeded with only the known C++
  subobject-linkage warnings.
- Expanded deployment/promotion/release/P2/browser-gate regression selection:
  212 passed, 41 subtests passed.
- The f1a3cf4c P2 baseline/staging path-oracle dry-run returned
  dry_run_valid with gpu_or_service_action=none. It opened no model, HIP
  device, service, or R9700 lock; its planned output remained absent.

## Receipt-linked accepted risk

The future receipt path is:

    /home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/promotion-reasoning-v2-fidelity-f1a3cf4c.json

This journal and the promotion runbook are the receipt-linked note because
ullm.aq4_resident_promotion.v1 has no field for an approval annotation.

The formal P2 result remains a formal no-go: 7/8 metrics passed, but
token_agreement_rate was 20/24 and its Wilson lower bound 0.676 was below the
0.899 requirement. The fidelity plan records the user's 2026-07-17 decision
to accept the residual near-margin differences as expected AQ4 4-bit
quantization noise. Any receipt at the path above must be called
user-approved accepted-risk, never formal Gate success.

## Parent handoff blockers

1. The existing bootstrap-v2 path rejects this v2-to-v2 candidate because the
   candidate worker SHA differs from the current v2 active worker. A complete
   bundle cannot be made until candidate-active evidence exists, so a parent
   decision on a reviewed bridge or policy change is required.
2. run-sq8-direct-cancel-gate.py is hard-coded to the SQ8 model and cannot
   produce AQ4 cancellation evidence. No SQ8 code or tooling was changed.
3. The current v2 browser runner/bundle implementation still demands a
   provider-switch artifact although the release policy says that comparison is
   no longer required. The parent must decide whether to run the
   implementation-enforced R9700-only switch or align policy/tooling first.

The exact conditional gate, bundle, and final AQ4-to-AQ4 activation commands
are in docs/plans/aq4-fidelity-root-cause-and-fix-plan-v0.1-promotion-runbook-v0.1.md.

## 2026-07-17 tooling follow-up

The user approved two bounded tooling decisions without authorizing any live
service action:

- A differing-worker v2-to-v2 evidence bootstrap is available only through
  `--bootstrap-v2 --authorize-differing-worker-v2-bootstrap` with a non-empty
  `--authorization-note`. The ordinary default rejection remains in force.
  The note and both worker/manifest identities are written as a mode-0600
  sidecar next to the bootstrap backup. This bridge is only for collecting
  candidate-active soak/stop/failure/browser evidence; the original active
  bytes must then be restored before normal complete-bundle activation.
- Browser smoke now permits a v2 no-switch record: exactly two candidate-model
  provider requests and no switch-specific fields. Supplying both switch model
  arguments retains the existing four-request uLLM -> llama.cpp -> uLLM path.
  The browser validator and generic release-bundle preparation accept either
  gate-eligible v2 shape, while retaining v1 read compatibility.

The SQ8-only cancel gate is exempt for this AQ4 promotion. No SQ8 code,
manifest, or tooling was changed. No systemd, Docker, GPU, sudo, R9700 lock,
or V620 operation was performed. Related pytest selection: 56 passed.
