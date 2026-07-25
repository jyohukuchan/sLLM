# AQ4_0 runtime-hardening locked activation v0.1

Status: implemented control-path contract. This document authorizes no active-manifest mutation, authorization consumption, service window, or promotion campaign.

## Scope

The route is dedicated `AQ4_0` to `AQ4_0`: it does not import, call, or share an execution entry point with the SQ8 final route. Both manifest sides must have model ID `ullm-qwen3.5-9b-aq4`, format `AQ4_0`, protocol `ullm.worker.v2`, and the same exact worker SHA-256. Their worker, product, and tokenizer paths must differ, so hardening changes the closure rather than model semantics.

Unlike SQ8, no cross-model campaign authorization is consumed. Fresh AQ4 bundle v1 collection is after hardened AQ4 live proof, not an activation input. This simplification makes protected path and source sealing the central safety property.

## Schemas and source pin

`tools/aq4_runtime_hardening_activation.py` publishes canonical root-owned `0444`, single-link records:

| Purpose | Schema |
| --- | --- |
| Plan | `ullm.aq4_runtime_hardening_activation_plan.v1` |
| Intent | `ullm.aq4_runtime_hardening_activation_intent.v1` |
| Outcome | `ullm.aq4_runtime_hardening_activation_outcome.v1` |
| Recovery | `ullm.aq4_runtime_hardening_activation_recovery.v1` |
| Manual rollback | `ullm.aq4_runtime_hardening_rollback_outcome.v1` |
| Failed-attempt audit | `ullm.aq4_runtime_hardening_recovery_attempt.v1` |
| Live proof | `ullm.aq4_runtime_hardening_live_proof.v1` |
| Plan preflight | `ullm.aq4_runtime_hardening_activation_preflight.v1` |

The plan binds control-source commit/tree/tool bytes, promotion source commit/tree, protected runtime seals, saved rollback bytes, legacy hashes, credential/unit/environment seals, operation executable hashes, lock identity, epoch, and every output destination. The detached clean standalone control source seals exactly these tools:

- `tools/aq4_runtime_hardening_activation.py`
- `tools/prepare-aq4-runtime-hardening-activation.py`
- `tools/run-aq4-runtime-hardening-activation.py`
- `tools/rollback-aq4-runtime-hardening-activation.py`

The reviewed implementation commit is recorded in the promotion plan. Later documentation-only commits do not change this runtime source pin.

## Atomic boundary

Immutable records use a pinned parent dirfd, a private single-link temporary file, file and parent `fsync`, and `renameat2(RENAME_NOREPLACE)`. No temporary hard link is created. A post-rename error is treated as committed only after the exact destination is durably reopened.

Under `/etc/ullm/served-models/.active.json.activation.lock`, candidate bytes are staged in the active-manifest directory and swapped with pinned-dirfd `renameat2(RENAME_EXCHANGE)`. The frozen candidate is never mutated. The old bytes remain in the staging inode until candidate proof or exact restoration. Reading both names after a fault prevents a committed rename from being misreported as pre-commit.

Credential, source, operations, unit/environment, candidate runtime, and exact active/rollback seals are checked before intent and again immediately before exchange. The immutable successful outcome is the commit boundary; no fallible source/runtime check follows it.

## Operations and recovery

The sealed operations document is `ullm.aq4_runtime_hardening_activation_operations.v1` with `candidate_reconcile`, `candidate_live_proof`, `rollback_reconcile`, and `rollback_live_proof`. A live observation binds plan SHA and epoch to active bytes, AQ4 model and worker path/hash, `ullm-openai.service` active/running state, boot/PID/PPID/starttime/executable identity, and all five gateway/OpenWebUI endpoints.

Before success receipt, failure restores exact saved AQ4 bytes under the same lock and requires rollback reconciliation/live proof. Legacy worker/product/tokenizer/receipt hashes must still validate before a rollback-health claim. SIGKILL/power loss after intent, `failed_restore`, and `rollback_incomplete` require exact plan SHA plus literal `RECOVER AQ4_RUNTIME_HARDENING`. Failed recovery attempts write unique audits only, preserving the successful receipt pathname for retry. Manual rollback additionally requires exact candidate-active bytes and `ROLLBACK AQ4_RUNTIME_HARDENING`.

Wrappers default to read-only preflight. Execution requires exact plan SHA plus `ACTIVATE AQ4_RUNTIME_HARDENING`; a human must still inspect plan/candidate/rollback hashes, maintenance window, and destinations immediately before the sole swap. The route has no reference to `llama-qwen35-udq4.service` or `gdm3`.

## Bundle v1

`tools/prepare-generic-reasoning-release-bundle.py` now uses no-replace publication for v1 as well as v2. Production CLI defaults to `--required-uid 0`, rejects unsafe/non-root parents, writes root-owned `0444` nlink-one output, then validates it. `validate-generic-reasoning-release-bundle.py --require-immutable-publication --required-uid 0` proves owner, mode, link count, and stable post-validation bytes; use it for the post-hardening AQ4 bundle v1 step.
