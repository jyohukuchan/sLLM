# Served-model v2 cross-model campaign authorization v0.2

Status: ratified implementation contract

Date: 2026-07-24

This contract authorizes exactly one temporary evidence window from the
current independent `AQ4_0` served model to the independent Qwen3-14B-FP8
`SQ8_0` served model, followed by exact `AQ4_0` restoration and fresh `AQ4_0`
release evidence. It is not a final activation authorization. It is unrelated
to the historical Qwen3.5 `AQ4_0` partial-FP8 “SQ8 overlay” lineage.

This document supersedes v0.1 before any v0.1 authorization was issued. V0.1
remains a historical exact-three-campaign draft and must not be accepted as a
v0.2 authorization, claim, outcome, bundle input, or final-activation input.

## 1. Authorization

The schema is
`ullm.served_model.v2_cross_model_campaign_authorization.v2`. The exact root
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
aq4_release
candidate
campaigns
rollback
prior_outcome
```

`issued_at` and `expires_at` are canonical whole-second UTC timestamps and
the authorization must be current when claimed. `max_attempts` is exactly
`1`. `purpose` is
`temporary_candidate_active_evidence_collection_only`.
`required_final_route` is
`restore_exact_aq4_then_bundle_v2_activation`.

`source` binds the full Git commit and tree of the SQ8 release source.
`before` has exactly:

```text
model_id
format_id
manifest_sha256
worker_protocol
worker_binary_path
worker_binary_sha256
promotion_source_commit
promotion_receipt_path
promotion_receipt_sha256
```

It permits only `ullm-qwen3.5-9b-aq4`, `AQ4_0`, and `ullm.worker.v2`.
The worker and promotion receipt are existing absolute regular non-symlink
inputs. Their bytes are rehashed during preflight and throughout execution.

`aq4_release` has exactly:

```text
source
openwebui_image
promotion_evidence
promotion_receipt
release_evidence_path
release_validator_path
browser_validator_path
```

Its `source` contains the absolute detached clean AQ4 source root and its full
commit/tree; its commit equals `before.promotion_source_commit`.
`openwebui_image` equals the fixed content-addressed OpenWebUI server image
in the source-bound plan. The plan separately fixes the immutable Playwright
browser-runner image; the two roles are never interchangeable. Each
promotion reference contains exact
`source_path`, fresh authorized copy `path`, and SHA-256. The receipt and
evidence are validated as one relative-path-preserving AQ4 promotion pair.
The source pair is never overwritten; the transaction publishes immutable
copies below the AQ4 bundle root.

`candidate` has exactly:

```text
model_id
format_id
manifest_sha256
worker_protocol
worker_binary_sha256
promotion_source_commit
promotion_receipt_sha256
```

It permits only `ullm-qwen3-14b-sq8`, `SQ8_0`, and `ullm.worker.v2`.
Its promotion source commit equals `source.commit`.

`campaigns` has exactly six entries, each with a unique `run_id` and absolute
fresh `final_path`:

```text
sq8_full
reasoning_release
reasoning_browser
aq4_reasoning_release
aq4_reasoning_browser
aq4_bundle
```

The first three are the fresh candidate-active SQ8 campaign. After exact AQ4
restoration and reverse reconciliation, the next two collect fresh AQ4
reasoning/browser evidence and `aq4_bundle` publishes and validates a fresh
complete `ullm.generic_reasoning_release_bundle.v1`. A successful outcome
requires all six.

The AQ4 browser evidence, release evidence and validator reports, immutable
promotion copies, and bundle v1 are distinct paths below the AQ4 bundle
parent. All authorized outputs, the exact AQ4 backup, both source roots, and
the AQ4 worker are pairwise non-overlapping where required. Existing output
or symlink leaves, symlinked parents, noncanonical paths, source-root output,
and path aliases are rejected.

`rollback` contains the fresh exact AQ4 backup path and the SHA-256 values of
the fixed systemd unit and environment file. `prior_outcome` is null for the
first authorization or an exact immutable path/SHA-256 reference to the
preceding failed outcome with the same AQ4/SQ8 lineage. It never grants an
automatic retry.

The authorization is strict canonical JSON with one trailing LF, mode 0444,
link count one, root ownership in production, bounded stable-read identity,
and atomic no-replace publication.

## 2. One-shot claim

The claim schema is
`ullm.served_model.v2_cross_model_campaign_claim.v2` with exactly:

```text
schema_version
authorization_id
authorization_path
authorization_sha256
claimed_at
attempt
max_attempts
```

Its path is derived only as:

```text
/var/lib/ullm/served-model-campaign-claims/<authorization-sha256>.claim.json
```

The locked transaction publishes the root-owned, mode-0444, nlink-1,
canonical claim with atomic no-replace semantics before its first
operational side effect. An existing destination means consumed. Every
preflight, switch, campaign failure, interruption, restoration result, or
outcome-publication error leaves it consumed. The standalone claim CLI is
non-operational.

Claims and their authorization/outcome/recovery records remain loadable after
`expires_at` for audit and recovery. They cannot be used to start another
campaign.

## 3. Source-bound exact-six transaction

The runner accepts no caller-provided command vector. It derives
`ullm.served_model.v2_cross_model_campaign_plan.v2` from the pinned source and
authorization. The plan fixes production paths, service and secret paths,
content-addressed helper images, OpenWebUI image identity, model identities,
source tools, run IDs, output paths, and v2 active-binding arguments.

Both the SQ8 and AQ4 campaign source roots are sealed standalone Git clones,
not linked worktrees and not the operator's writable development checkout.
In production, every source member and the in-tree `.git` directory is
root-owned, free of symlinks, special files, hard-linked regular files,
POSIX ACLs, object alternates, and group/world write permission. Every
non-sticky path ancestor is likewise protected against non-root replacement.
The transaction inventories the complete source and Git metadata before any
Git query, uses a fixed non-mutating Git environment, and requires the same
stable filesystem fingerprint before and after every command. The runner
itself must be invoked from the same sealed SQ8 source named by
`--source-root`; pointing a writable runner at a separately sealed tree is
not sufficient.

Both served-model manifests must also resolve to sealed runtime closures.
Every worker executable, promotion receipt/evidence input, tokenizer member,
product/package manifest, and package payload reachable by either manifest
must be root-owned below protected non-user-writable ancestry, regular or
directory as appropriate, free of symlinks, POSIX ACLs, hard-linked regular
files, and group/world write permission. Preflight inventories those closures
and every command boundary revalidates the same metadata/content seal. A
read-only filename below a user-owned or writable parent is not immutable.

The 2026-07-24 production AQ4 bootstrap manifest points into user-owned
`/home` and product trees and therefore intentionally fails this admission.
Before issuing an exact-six authorization, operators must complete a separate
reviewed AQ4-to-AQ4 runtime-hardening promotion and make its new protected
manifest the authorization's `before`. Existing AQ4 evidence cannot simply
be copied because it binds the old absolute runtime paths.

The transaction holds the served-model activation lock across:

1. claim, read-only preflight, and immutable exact-AQ4 backup;
2. atomic SQ8 activation and candidate reconciliation/checks;
3. fresh `sq8_full`, `reasoning_release`, and `reasoning_browser`;
4. unconditional exact AQ4 restoration and reverse reconciliation;
5. fresh `aq4_reasoning_release`, `aq4_reasoning_browser`, and complete AQ4
   bundle v1;
6. final exact-AQ4 checks, structured live proof, immutable outcome
   publication, and lock close.

Switch and restoration use a pinned parent descriptor,
exact-current `renameat2(RENAME_EXCHANGE)`, inode/byte verification, and
directory `fsync`. Commands run under a Linux child subreaper. Timeout,
signal, or command failure terminates and reaps descendants before
restoration. `expires_at` bounds every evidence-producing command in both the
SQ8 and restored-AQ4 campaign phases, but never shortens restoration or final
proof.

Only evidence producers run as the fixed service identity
`uid=1000,gid=1000` with an exact supplementary-group allowlist. Root retains
the claim, lock, active-manifest exchange, reconciliation, validation,
adoption, publication, restoration, proof, and outcome boundaries. Each
producer receives a fresh transaction-private staging pathname plus the exact
authorization/claim hashes, campaign stage, and sealed SQ8 source root. The
producer must keep the CLI final path as its authorization and lineage
identity; the private pathname is only its physical publication target.

No producer may publish directly to an authorized final path. This also
contains legacy AQ4 producers whose historical publication uses replacing
rename. After all producer descendants are reaped, root descriptor-walks the
private tree, rejects symlinks, special files, hard links, cross-device
entries, unexpected owners/modes/layout, excessive counts/bytes, or a leaked
staging pathname, then changes ownership, freezes, validates, and publishes
the authorized file or directory with kernel-enforced no-replace semantics.
A raced final path is preserved and fails the consumed transaction.

After candidate reconciliation and readiness checks, the transaction performs
at least 900 seconds of monitored candidate stabilization before starting
`sq8_full`. Throughout that interval it repeatedly re-pins the authorization,
claim, sealed source and candidate runtime, exact `active.json` bytes, and the
unchanged service/gateway/worker epoch. Any drift or authorization deadline
shortfall fails before the full campaign. The `sq8_full` command may use up to
six hours, but its timeout is always capped by the authorization time
remaining; the authorization must be issued with enough margin for
stabilization, all six campaigns, restoration, and final proof.

The transaction re-pins source/tree, the scope-relevant complete runtime
closures, candidate, AQ4 worker and promotion source pair, unit, environment,
secrets, authorization, and claim between commands. Candidate-active stages
require both candidate and AQ4 rollback closures. Once exact AQ4 restoration
has committed, reverse reconciliation, fresh AQ4 evidence, final proof, and
standalone recovery require only the AQ4 and shared operation/credential
closures; loss of an already-displaced SQ8 candidate cannot prevent safe AQ4
recovery. It verifies the running OpenWebUI container image ID after compose
reconciliation and immediately before and after both browser campaigns.

SQ8 browser evidence uses
`ullm.openwebui.reasoning_browser_smoke.v5`. It retains v4 active-manifest
lineage, records the Playwright runner as `browser_image`, and records exact
before/after OpenWebUI server observations (`container_id`, `image_id`,
configured image, container name, running state, PID, and `started_at`).
Validator v3 requires both observations to be identical and to equal the
fixed server identity, and the bundle-v2 validator cross-binds them to this
authorization. Historical v4 evidence cannot satisfy bundle v2.

The command child receives a fixed minimal environment. Ambient loader,
Python import, Docker configuration, Git configuration, and PATH overrides
are not inherited from the invoking shell. Source-owned Python tools use the
fixed `/usr/bin/python3.12 -I -S -B` prefix. The two ROCm vendor CLIs require
their root-owned script directory for sibling imports and therefore use the
separate fixed `/usr/bin/python3.12 -E -S -B` prefix; `/opt/rocm-7.2.1`, its
Python modules/native libraries, `/usr/bin/ps`, and `/bin/sh` are an explicit
root-owned OS/ROCm TCB. Reviewed top-level executables are opened and invoked
through sealed descriptors. The sealed root-owned source/runtime ancestry is
the trust boundary for interpreter imports and reviewed descendant scripts.

ELF interpreters/shared libraries, the Python standard library/native
extensions, Git/Docker/systemd internals, Docker daemon and digest-pinned
container runtime, and the fixed root-owned OS paths above are also explicit
TCB rather than recursively byte-inventoried campaign artifacts. The service
identity's reviewed supplementary groups include host operational groups,
including Docker; an actor that can independently exercise those credentials
is root-equivalent and is outside this local TOCTOU model. Supporting hostile
same-credential processes would require a dedicated unprivileged producer
identity plus a constrained root broker, not additional file hashes.

The fixed gateway API credential is
`/etc/ullm/openai-api-key`, `uid=0,gid=1000,mode=0640`. The fixed browser-login
JWT is `/run/ullm-campaign-secrets/openwebui-session.jwt`,
`uid=0,gid=1000,mode=0640`, below a `uid=0,gid=1000,mode=0750` directory.
This directory is not nested below a service-user-writable parent. Both
credentials are stable-read, sealed, and re-pinned without recording their
bytes or pathname-derived staging details in public evidence.

## 4. Active-manifest observations and campaign outputs

The outcome records candidate observations in this exact successful order:

```text
candidate_activation
candidate_reconciliation
candidate_checks
sq8_full:before
sq8_full:after
reasoning_release:before
reasoning_release:after
reasoning_browser:before
reasoning_browser:after
```

Each observation stable-reads the actual production `active.json` and the
frozen candidate and requires byte equality.

After restoration it records AQ4 observations in this exact successful order:

```text
aq4_reasoning_release:before
aq4_reasoning_release:after
aq4_reasoning_browser:before
aq4_reasoning_browser:after
aq4_bundle:before
aq4_bundle:after
```

Each requires the actual production `active.json` bytes to equal the exact
pre-window AQ4 bytes. A successful outcome contains every observation.

Each campaign result binds its authorized run ID/final path, file or
directory kind, exact file-tree inventory SHA-256, artifact count, total
bytes, and selected evidence hashes. SQ8 outputs additionally bind included
candidate bytes and campaign-lineage v2. The fresh AQ4 raw campaign, browser
evidence, validator reports, and bundle v1 are independently schema-checked
and immutable.

## 5. Outcome and restoration

The outcome schema is
`ullm.served_model.v2_cross_model_campaign_outcome.v2`. Its exact root fields
are:

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
aq4_observations
candidate_observations
campaigns
restoration
```

`stages` has exactly:

```text
claim
lock
preflight
backup
candidate_activation
candidate_reconciliation
candidate_checks
sq8_full
reasoning_release
reasoning_browser
aq4_restore
reverse_reconciliation
aq4_reasoning_release
aq4_reasoning_browser
aq4_bundle
final_checks
```

A durable outcome contains no `pending` state. `succeeded_restored` requires
every stage and all six campaign outputs to pass. `failed_restored` requires
an identified failed stage plus complete exact-AQ4 restoration.
`failed_restore` means that complete restoration could not be proved and
authorizes no bundle or activation.

`restoration` binds expected, displaced, and observed manifest hashes; exact
byte equality; reverse reconciliation and final checks; AQ4 model/format and
worker hash; and a structured
`ullm.served_model.v2_cross_model_restoration_proof.v1`. The proof binds the
authorization/claim, active bytes, service and boot epoch, gateway/worker
PID/PPID/starttime/live executable, and stable HTTP 200 Gateway/OpenWebUI
health and exact-AQ4 model listings.

The outcome is canonical, root-owned, mode 0444, nlink 1, and atomically
published without replacement at:

```text
/var/lib/ullm/served-model-campaign-outcomes/<authorization-sha256>.outcome.json
```

## 6. Locked recovery

The recovery receipt schema is
`ullm.served_model.v2_cross_model_campaign_recovery.v2`. Recovery is
read-only unless the operator selects execution and confirms the exact
authorization SHA-256. It works after authorization expiry and does not
require the historical AQ4 source or campaign outputs to remain available.

Recovery loads the consumed claim, exact immutable backup, pinned
unit/environment, the sealed recovery command source, and the only safe active
states. Candidate identity is recognized from the authorization and current
active bytes; the original candidate pathname, SQ8 runtime, SQ8 campaign
outputs, and historical AQ4 campaign source are not recovery dependencies. It
never blindly overwrites an unknown active entry. Under the same activation
lock it restores exact AQ4, reverse-reconciles, performs final checks/live
proof, and publishes one immutable recovery receipt. `restored` requires the
same byte, worker, service, and HTTP proof as a restored outcome; otherwise
the receipt is `failed_restore`.

Recovery reconstructs and seals the backup-AQ4 runtime closure plus shared
unit, environment, credential, and rollback-operation artifacts before any
reconciliation command. An absent, mutable, or empty required seal is a hard
failure. Restoring manifest bytes alone never permits execution of an
unsealed AQ4 worker or payload.

## 7. Bundle and final activation boundary

The successful transaction itself produces the fresh AQ4 complete bundle v1.
Afterward, the operator assembles and independently validates the SQ8
`ullm.generic_reasoning_release_bundle.v2` from the three fresh SQ8 campaign
outputs and final SQ8 promotion pair.

`ullm.served_model.final_activation_plan.v2` is a separate, default-read-only
boundary. It reloads and inventories all six successful campaign outputs,
derives the AQ4 bundle path only from the outcome, validates the fresh AQ4
bundle v1 first, validates the SQ8 bundle v2 second, and pins exact rollback
and candidate bytes. Activation requires the exact plan SHA-256 and literal
confirmation. Later rollback uses the plan-bound locked rollback route and
does not reopen the campaign authorization, claim, outcome, SQ8 bundle, SQ8
candidate pathname, or SQ8 runtime. It requires the immutable successful
activation outcome, exact currently-active SQ8 bytes, sealed AQ4 rollback
closure, and reviewed rollback operations. Thus loss or corruption of the
completed campaign registry cannot strand an activated system on SQ8.

Implementation and private/mock/CPU validation do not authorize live
execution. A production claim, either fresh GPU campaign, final bundle, or
activation additionally requires reviewed final artifacts and a real private
OpenWebUI browser-login session JWT. None is created or executed by this
contract.
