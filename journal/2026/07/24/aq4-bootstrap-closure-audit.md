# AQ4_0 bootstrap closure audit

Date: 2026-07-24

Status: **content-consistent and live, but neither historically closed by
complete bundle v1 nor admissible as a root-owned final-activation runtime
closure.**

This audit was read-only with respect to production.  It did not change
`/etc/ullm/served-models/active.json`, any candidate or release, a systemd
unit, service state, ownership, or permissions.  It did not execute a worker,
contact a GPU, or inspect authentication/JWT material.  The only new files are
the retrospective audit artifacts in this repository.

## Exact production reference

The immutable reference for this audit was the live
`/etc/ullm/served-models/active.json`:

| Item | Observed value |
|---|---|
| file identity | root:root, mode `0644`, nlink 1, 4,459 bytes |
| manifest SHA-256 | `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a` |
| schema / format / protocol | `ullm.served_model.v2` / `AQ4_0` / `ullm.worker.v2` |
| public model | `ullm-qwen3.5-9b-aq4` |
| candidate equality | byte-equal to `/etc/ullm/served-models/candidates/qwen35-9b-aq4-reasoning-fidelity-f1a3cf4c.json` |
| worker | `/home/homelab1/coding-local/ultimateLLM/uLLM-aq4-fidelity-promotion-release-f1a3cf4c/ullm-aq4-worker` |
| worker identity | homelab1:homelab1, mode `0555`, nlink 1, 4,223,912 bytes |
| worker SHA-256 | `1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b` |
| promotion source commit | `0cd760568e197e1adb4c4df3d6149591a912f709` |
| promotion receipt SHA-256 | `1b36fc880bf1510185eaad7887c9aed33f69df223036271e4bfba4bb43f16e8b` |
| promotion evidence SHA-256 | `fbeb061c5aa852dd6449700a2a151f3315da6b00e5310c758062a19dc1362f60` |
| package manifest SHA-256 | `a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad` |
| tokenizer aggregate SHA-256 | `b959f4b4ac6b6272f390747e393b1adcf1c55f14d78daeb37d6dedb9598a49d9` |

`tools/validate-served-model.py` accepted the live manifest.  All declared
worker, receipt, promotion-evidence, package-manifest, and tokenizer-member
hashes matched the bytes at their declared absolute paths.

Read-only `systemctl show` and `/proc` inspection observed the service
active/running with gateway PID 1440198 and worker PID 1440624.  The worker
argv named the live active manifest, `/proc/1440624/exe` resolved to the
declared release binary, and the opened executable hash matched
`1f93f215...`.  These PIDs are a point-in-time observation, not a durable
identity claim.

### Two distinct source commits

The worker build and promotion admission lineages must not be conflated:

- The worker and legacy engine were built while the detached build worktree
  was at `f1a3cf4c86978b3b8900396a0b6a8caff90b97f1`.  The worktree reflog,
  release mtimes, and the AQ4 promotion runbook corroborate this.
- The evidence/tooling worktree was then advanced to clean detached
  `0cd760568e197e1adb4c4df3d6149591a912f709`.  The promotion evidence,
  receipt, candidate, and active manifest all bind this latter commit.
- No Rust/Cargo input changed between those commits.  Only deployment,
  browser-gate, activation, evidence-runner, tests, and documentation files
  changed.

Consequently, any retrospective bundle-v1 evidence must use `0cd76056...` as
both `source_commit` and `active_promotion_source_commit`.  The historical
runbook's earlier `SOURCE_COMMIT=f1a3cf4c...` instruction no longer matches
the produced receipt or active manifest.

## What the differing-worker bootstrap authorized

Commit `9c9a6f2972d09ec74a85c58498840f5b7fcee304` added a deliberately
temporary escape hatch to `tools/activate-served-model.py`.  It applies only
when all ordinary bootstrap safety checks pass, the active and candidate are
both v2 with the same public model ID, and their worker hashes differ.  That
one worker-equality check can be relaxed only with all of:

- `--bootstrap-v2`;
- no `--release-bundle`;
- a fresh absolute `--bootstrap-backup`;
- `--authorize-differing-worker-v2-bootstrap`; and
- a non-empty, bounded, control-character-free `--authorization-note`.

It retains manifest validation, activation locking, stable unit/environment
reads, inactive-service checks, an external exact old-active backup, atomic
replacement, hooks, and exact-byte rollback on hook failure.  The
authorization flag is rejected for v1 active manifests, same-worker v2
manifests, and a differing public model ID.  It is not a cross-model or
AQ4-to-SQ8 path.

The escape hatch publishes a mode-`0600`, no-replace sidecar with schema
`ullm.served_model.v2_differing_worker_bootstrap_authorization.v1`.  It binds
the note, temporary-evidence-only purpose, required restoration statement,
old/candidate manifest hashes, and old/candidate worker hashes.  It does not
record a bundle, unit/environment hashes, actor, attempt ID, hook outcome, or
successful live state.  Its existence therefore cannot prove a normal
bundle-gated activation.

Three backup/sidecar attempts remain.  The v4 backup is root:root `0644`,
has the old-active SHA-256
`feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`,
and its adjacent sidecar is root:root `0600`, nlink 1, 1,013 bytes.  Creation
ordering makes v4 the overwhelmingly likely successful attempt, but the
sidecar is unreadable without elevated privileges and all of these files sit
below user-owned result ancestry.  A root operator must read and hash the
sidecar before treating even its fields as confirmed.

## Retrospective bundle-v1 audit

The AQ4 v1 bundle requires six exact artifact slots:

1. complete release evidence;
2. its independently recomputed validator report;
3. complete browser evidence;
4. its independently recomputed validator report;
5. AQ4 promotion evidence; and
6. its bound promotion receipt.

It additionally binds the candidate manifest, worker, tokenizer aggregate,
content-addressed OpenWebUI image, aligned source commit, old active manifest,
systemd unit, and environment file.  Normal activation recomputes these
bindings against live paths.

The current AQ4 output contains only the ten-case HTTP/SSE campaign.  No
current browser evidence, validator reports, or complete bundle v1 exists.
The preserved cases cover all five reasoning modes, have ten matching
lifecycle events, all quality checks true, and all required timing fields.

To determine whether that core slot was intrinsically missing or merely never
assembled, a fresh local `git clone --no-hardlinks` was detached at the exact
promotion source `0cd76056...`.  The historical preparer consumed the
preserved cases/lifecycle and the live manifest/worker in a temporary output.
Both the exact historical validator and the current validator reported:

- schema `ullm.generic_reasoning_release_validator.v1`;
- 10 cases and 10 lifecycle events;
- all five modes;
- `structurally_valid=true`;
- `git_worktree_clean=true`; and
- `gate_eligible=true`, with no reasons.

The resulting immutable copy is
`benchmarks/results/2026-07-24/aq4-bootstrap-closure-audit-v0.1/release-evidence-retrospective.json`.
Its SHA-256 is
`5f729be13c53aac071bf55f0642432db97be5fcbc3caa8c9eb78078c8d9af3f2`.
The current no-replace publisher produced
`release-evidence-retrospective-validation.json`, SHA-256
`ed3db25dbf1305faa4b83a630a0e4d0162ae6523913f5fc0aef9615b52f5aaeb`.
This proves the historical core release-evidence slot can be reconstructed
for the exact current old-path identity.  It does not supply browser evidence,
build a bundle, or convert the past bootstrap into a normal activation.

The only pre-existing complete AQ4 bundle v1 is the older
`release-bundle-ae8b2bb-20260714-final.json`, SHA-256
`3a6307adf2a7cc32c4412407bc9430a44ad24b4db427b272f352e1eb86c21534`.
It remains structurally valid and gate eligible, but binds source
`ae8b2bb7...`, manifest `feb3190d...`, and worker `177f3106...`.  It is not
applicable to the current AQ4 bytes.

Even a counterfactual current bundle cannot alter history.  A bundle whose
rollback manifest is the preserved old active would be rejected by normal
activation now because the live active bytes already equal the candidate.
Only a dedicated read-only counterfactual validator could answer whether all
historical inputs would have matched at that earlier boundary.

## Root-owned runtime-closure result

The final SQ8 activation policy applies an additional runtime seal that
historical bundle v1 did not provide.  Running that exact seal predicate
against the current manifest with `required_uid=0` failed at the worker with
`runtime artifact ancestry owner is untrusted`.  The remaining reachable
assets independently fail as follows:

| Closure | Observed state | Result |
|---|---|---|
| AQ4 release | user-owned `0755` directory below user-owned `0775` ancestry; two user-owned `0555` binaries | fail |
| promotion receipt/evidence | user-owned `0644` below a user-owned `0775` product root | fail |
| package | 4 user-owned `0775` directories; 1 manifest plus 1,044 payload files, all user-owned `0664`; 7,700,872,459 regular-file bytes | fail |
| complete product root | 9 directories, 1,167 regular files, one `artifact -> package` symlink | fail |
| tokenizer root | user-owned `0775`; 34 user-owned `0664` files | fail |
| AQ4 source | linked worktree with a user-owned `.git` pointer file, not a standalone sealed clone | fail |

The runtime verifier recursively seals the entire declared `product.root`,
not only `package/`.  The symlink and unrelated historical entries therefore
remain in scope.  Changing only leaf modes or ownership, copying only the
worker, or copying only `package/` cannot close the current manifest.

The correct remediation is a separately reviewed AQ4-to-AQ4 runtime-hardening
promotion:

1. no-hardlink copy a purpose-built minimal worker/product/tokenizer closure
   below protected root-owned ancestry outside `/home`;
2. create a root-owned standalone source clone at the exact AQ4 promotion
   commit;
3. collect fresh path-bound promotion evidence and receipt;
4. freeze a new AQ4 manifest naming only the protected absolute paths;
5. activate it through its own locked rollback/live-proof route; and
6. collect fresh AQ4 release/browser campaigns and complete bundle v1 for
   that new manifest.

Existing AQ4 evidence cannot be copied into this closure because it binds the
old absolute paths and manifest hash.  None of these live steps was performed
by this audit.

## Fixed rollback inputs observed

- old active backup SHA-256:
  `feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`
- old active worker SHA-256:
  `177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d`
- `/etc/systemd/system/ullm-openai.service` SHA-256:
  `f0239713b16b3bf31cfd12a98f506e77e55af9b31abf58352f4e437e1cdee552`
- `/etc/ullm/openai-gateway-manifest.env` SHA-256:
  `68dd3a027fa86aaa8f5649bf55f34c32b818afb49a9e35e272f5dc6a1e5fb835`

These hashes are audit observations.  The bootstrap sidecar did not bind the
unit or environment hashes.

## Verification

The escape-hatch and bundle-v1 regression set passed without cache writes:

```text
PYTHONDONTWRITEBYTECODE=1 uv run --project services/openai-gateway \
  pytest -q -p no:cacheprovider \
  tests/test_activate_served_model.py \
  tests/test_prepare_generic_reasoning_release_bundle.py \
  tests/test_validate_generic_reasoning_release_bundle.py
```

Result: `68 passed`.

The retrospective evidence is regenerated only from a clean detached
`0cd76056...` source and the exact paths/hashes fixed at the top of this
record.  It must not be regenerated from current HEAD and relabeled as AQ4
source evidence.
