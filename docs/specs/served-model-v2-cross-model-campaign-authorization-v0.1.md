# Served-model v2 cross-model campaign authorization v0.1

Status: superseded, unissued draft

Date: 2026-07-24

This exact-three-campaign draft was superseded by
`served-model-v2-cross-model-campaign-authorization-v0.2.md` before any v0.1
authorization or claim was issued. It is retained only to make the version
boundary auditable. Production tooling must reject every v0.1 authorization,
claim, outcome, and recovery schema.

This contract authorizes exactly one temporary candidate-active evidence
window from the current `AQ4_0` served model to the independent `SQ8_0`
served model. It is not a final activation authorization. It is unrelated to
the historical Qwen3.5 AQ4 SQ8-overlay authorization lineage.

## 1. Authorization

The authorization schema is
`ullm.served_model.v2_cross_model_campaign_authorization.v1`. Its exact root
fields are:

```text
schema_version
authorization_id
issued_at
expires_at
max_attempts
authorization_note
purpose
required_final_route
source
before
candidate
campaigns
rollback
prior_outcome
```

`issued_at` and `expires_at` are canonical whole-second UTC timestamps.
`max_attempts` is exactly `1`. `purpose` is
`temporary_candidate_active_evidence_collection_only`, and
`required_final_route` is
`restore_exact_aq4_then_bundle_v2_activation`.

`source` contains the exact full Git commit and tree. `before` binds model ID
`ullm-qwen3.5-9b-aq4`, format `AQ4_0`, active-manifest SHA-256, worker
SHA-256, and full promotion source commit. `candidate` binds model ID
`ullm-qwen3-14b-sq8`, format `SQ8_0`, worker protocol `ullm.worker.v2`,
candidate-manifest SHA-256, worker SHA-256, full promotion source commit, and
promotion-receipt SHA-256. The candidate promotion commit equals
`source.commit`.

`campaigns` has exactly `sq8_full`, `reasoning_release`, and
`reasoning_browser`. Each contains a distinct run ID and absolute final output
path. `rollback` contains the absolute fresh AQ4 backup path plus the systemd
unit and environment-file SHA-256 values. `prior_outcome` is null for the
first attempt or an exact immutable path/SHA-256 reference to the preceding
failed outcome for an operator-authorized successor.

The authorization is strict canonical JSON with one trailing LF. It is an
absolute, root-owned, regular non-symlink file with mode `0444`, link count
one, bounded stable-read identity, and no extra or duplicate fields. It must
be unexpired when claimed.

## 2. One-shot claim

The claim schema is
`ullm.served_model.v2_cross_model_campaign_claim.v1`. The exact fields are:

```text
schema_version
authorization_id
authorization_path
authorization_sha256
claimed_at
attempt
max_attempts
```

The claim path is derived only as:

```text
/var/lib/ullm/served-model-campaign-claims/<authorization-sha256>.claim.json
```

There is no caller-selected registry or claim name. The window wrapper
validates the authorization and atomically publishes this claim with
no-replace semantics before its first operational side effect. The claim is
root-owned, mode `0444`, link count one, canonical, and stable-rehashed after
publication. An existing destination means the authorization is consumed.
It remains consumed after every later preflight, switch, campaign,
restoration, interruption, or outcome-publication failure.

All window and campaign processes load the same authorization-derived claim.
They compare the source/tree, before and candidate hashes, worker and receipt
hashes, rollback path, campaign name, run ID, and final path with the
authorization. A fresh caller-selected output cannot create a retry.

The standalone claim CLI is deliberately non-operational. Only the locked
campaign runner may consume a claim. Loading a consumed claim, outcome, or
recovery receipt uses archival validation and remains possible after
`expires_at`; expiry cannot make the durable audit trail unreadable.

## 3. Transaction and outcome

The candidate-active claim is usable only by the locked cross-model campaign
transaction. Its CLI accepts no command vectors: it derives the exact
`ullm.served_model.v2_cross_model_campaign_plan.v1` from the authorization
and authorized source tree. The plan fixes the production paths, service,
secret-file locations, content-addressed images, source tools, run IDs,
outputs, and v2 active-binding arguments. Every stage re-pins the
source/tree, manifests, worker, receipt, unit, environment, claim, and output
identities.

The transaction holds the served-model activation lock from AQ4 snapshot
through candidate activation, all three campaigns, exact AQ4 restoration,
reverse reconciliation, structured live checks, outcome publication, and
lock close. Switch and restore use a pinned parent directory descriptor and
exact-current `renameat2(RENAME_EXCHANGE)`, verify both exchanged inode/byte
identities, and `fsync` the directory. `expires_at` is the hard
candidate-active deadline, but expiry or a signal never shortens the
unconditional restoration path. Commands run under a Linux child subreaper;
failure, timeout, or interruption terminates and reaps their descendants.

Every exit attempts to publish
`ullm.served_model.v2_cross_model_campaign_outcome.v1` at:

```text
/var/lib/ullm/served-model-campaign-outcomes/<authorization-sha256>.outcome.json
```

The outcome binds the authorization and claim hashes, stage results, candidate
observations, campaign manifest/evidence/report hashes when available, exact
restored AQ4 hash, reverse reconciliation, and final model/health checks.
Its exact root fields are:

```text
schema_version
authorization_id
authorization_path
authorization_sha256
claim_path
claim_sha256
started_at
completed_at
status
failure_stage
stages
candidate_observations
campaigns
restoration
```

`status` is exactly one of `succeeded_restored`, `failed_restored`, or
`failed_restore`. `stages` has the exact keys `claim`, `lock`, `preflight`,
`backup`, `candidate_activation`, `candidate_reconciliation`,
`candidate_checks`, `sq8_full`, `reasoning_release`, `reasoning_browser`,
`aq4_restore`, `reverse_reconciliation`, and `final_checks`. Each value is
`pending`, `passed`, `failed`, or `skipped` while the transaction is in
memory; a published outcome contains no `pending` value. A successful outcome
has every stage `passed`; a failed outcome names a stage whose value is
`failed`.

Each non-null campaign result binds its authorized run ID and path, whether
the result is a file or directory, the SHA-256 of its canonical file-tree
inventory, artifact count, total bytes, and selected manifest/evidence/report
hashes. Candidate observations record an ordered stage name, the freshly read
active-manifest hash, and the result of exact byte comparison.

`restoration` has the exact fields `expected_manifest_sha256`,
`displaced_manifest_sha256`, `observed_manifest_sha256`, `bytes_equal`,
`reverse_reconciliation_passed`, `final_checks_passed`, `model_id`,
`format_id`, `worker_binary_sha256`, and `proof`. Both
`succeeded_restored` and `failed_restored` must prove exact AQ4 bytes,
successful reverse reconciliation and final checks, model
`ullm-qwen3.5-9b-aq4`, format `AQ4_0`, the authorized AQ4 worker hash, and a
non-null `ullm.served_model.v2_cross_model_restoration_proof.v1`.
`failed_restore` means that complete proof could not be made; it never
authorizes bundle assembly.

The proof binds authorization/claim, exact active bytes, service state, boot
ID/restart count, gateway and worker PID/PPID/starttime/live executable hash,
and HTTP 200 results for Gateway health/ready/models and OpenWebUI
health/models. Both model listings must contain exactly AQ4, and the observed
process/service epoch must remain stable. Command exit status alone is not a
restoration proof.

The outcome is strict canonical JSON, mode 0444, link count one, and published
with atomic no-replace semantics at the authorization-derived path. A second
publication is rejected. A successor authorization validates the referenced
outcome's complete shape and canonical immutable bytes, not merely its schema
tag.
Outcome publication never releases or deletes the claim. A successor
authorization must bind this immutable outcome through `prior_outcome`.

Only a successful outcome that proves exact AQ4 restoration may feed bundle
v2 assembly. Final activation uses the complete bundle-v2 route and never
reuses this authorization or claim.

## 4. Locked recovery and execution boundary

`recover-served-model-v2-cross-model-campaign.py` is the recovery route for a
consumed claim whose transaction did not prove restoration. It is read-only
unless `--execute-recovery` and the exact authorization SHA-256 are supplied.
It loads the claim after expiry, uses the same fixed plan and activation lock,
re-pins the source/candidate/AQ4 backup/unit/environment identities, refuses
blind replacement of an unknown active entry, and publishes exactly one
immutable
`ullm.served_model.v2_cross_model_campaign_recovery.v1` receipt at
`/var/lib/ullm/served-model-campaign-outcomes/<authorization-sha256>.recovery.json`.
`restored` requires the same exact bytes, reconciliation, worker identity,
and structured live AQ4 proof.

This draft was exercised only with private files and mocked commands before
being superseded. No production authorization was issued or claimed, no
production `active.json` or `ullm-openai.service` was changed, and no GPU
campaign was run under this schema.
