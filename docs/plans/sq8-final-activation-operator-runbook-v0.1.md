# SQ8_0 final activation and exact AQ4_0 rollback runbook v0.1

Status: implementation-only preparation. **Do not execute the activation or
rollback commands during the current work.**

This runbook applies only to the independent Qwen3-14B-FP8 `SQ8_0` served
model. It does not apply to the historical AQ4 SQ8 overlay.

## Admission boundary

Do not prepare or execute a final plan until all of the following exist with
their two deliberately distinct source lineages exactly bound. The SQ8
worker, candidate, promotion pair, three SQ8 campaigns, and bundle v2 use the
same clean SQ8 release commit/tree. The restored-AQ4 campaigns and fresh AQ4
bundle v1 use the authorization's exact `before.promotion_source_commit` and
detached AQ4 source tree; they must not be relabeled as SQ8-source artifacts.

1. the final sealed SQ8 worker and build receipt;
2. an immutable, consumed cross-model campaign authorization;
3. its `succeeded_restored` immutable outcome, including all six complete
   `sq8_full`, `reasoning_release`, `reasoning_browser`,
   `aq4_reasoning_release`, `aq4_reasoning_browser`, and `aq4_bundle` output
   inventories and a live AQ4 restoration proof;
4. the exact read-only AQ4 manifest backup named by that authorization;
5. the fresh complete, gate-eligible AQ4
   `ullm.generic_reasoning_release_bundle.v1` at the exact path and hash in
   the successful outcome;
6. a complete, gate-eligible SQ8
   `ullm.generic_reasoning_release_bundle.v2`;
7. the frozen SQ8 candidate manifest, systemd unit, and environment file whose
   hashes equal the authorization and bundle; and
8. separate root-owned, non-writable, standalone Git clones for the exact SQ8
   and AQ4 campaign commits/trees, with the campaign runner invoked from the
   sealed SQ8 clone itself; and
9. root-owned sealed runtime closures for both manifests, including every
   worker/legacy-engine executable, promotion input, tokenizer member,
   product/package manifest, and package payload, below protected ancestry;
   and
10. a human-reviewed operations document as described below.

The fresh browser campaigns require a real private OpenWebUI browser-session
JWT and the full campaigns require the production GPU/service window. Those
runs are outside this implementation task.

The existing development checkout, the preserved
`uLLM-sq8-manifest-candidate-release-ee62d04e` baseline, and the historical
AQ4 detached worktree are evidence/build inputs only. They do not satisfy the
campaign source seal. Before authorization issuance, create new standalone
clones under a root-owned, non-group/world-writable operations directory with
`git clone --no-hardlinks`, detach the exact commits, remove no files, and
confirm clean trees. Do not use `git worktree add`: an external `.git` file
or object alternate is rejected. The cross-model runner and recovery runner
must be executed by absolute path from that sealed SQ8 clone and must receive
that same clone as `--source-root`.

The final-plan preparation, activation/preflight, and rollback wrappers must
also be invoked from that same root-owned sealed SQ8 clone. Their execution
source is derived only from the loaded module path; there is no CLI or plan
field that can redirect it. Before any local validator runs, the core requires
an internal `.git` directory, no object alternate or linked worktree, detached
and clean HEAD, protected root-owned ancestry, and an exact match between the
loaded source commit/tree and the plan's existing SQ8 `source` lineage. It
re-pins that source before and after every activation, restoration, rollback,
and outcome-publication boundary. The wrappers reject relative paths,
development checkouts, direct shebang execution, and every interpreter form
other than the exact absolute command shown below.

The current AQ4 bootstrap manifest
`5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a`
also does not satisfy runtime admission: its worker, promotion inputs,
tokenizer, and 1,044 package payloads are below user-owned/writable `/home`
and product trees. Do not issue the exact-six authorization from that
baseline. First complete a separately reviewed and authorized AQ4-to-AQ4
runtime-hardening promotion: no-hardlink root-stage the complete closure
outside `/home`, collect fresh path-bound AQ4 promotion evidence/receipt,
freeze a new manifest, and activate it through its own locked rollback/live
proof route. That future prerequisite requires a GPU/service window and is
not executed by this runbook. The exact-six `before`, backup, fresh AQ4
campaigns, and final rollback must all bind the resulting hardened manifest.

The 2026-07-24 read-only follow-up in
`journal/2026/07/24/aq4-bootstrap-closure-audit.md` confirms the live content
identity, reconstructs only the historical core release-evidence slot, and
records the concrete bundle/runtime-closure gaps. The unexecuted root-owned
SQ8 worker staging commands and corresponding AQ4 asset audit are in
`docs/plans/sq8-aq4-root-owned-staging-runbook-v0.1.md`. Neither document
waives this AQ4-to-AQ4 prerequisite.

## Reviewed operations document

The final tools do not accept command JSON on their command line. The five
operational stages are instead fixed in a separately reviewed, canonical JSON
file, made read-only (`0444`), owned by root, and linked exactly once:

```json
{
  "schema_version": "ullm.served_model.final_activation_operations.v2",
  "review_id": "REVIEWED-ID",
  "reviewed_at": "YYYY-MM-DDTHH:MM:SSZ",
  "reviewed_by": "HUMAN-REVIEWER",
  "timeout_seconds": 300,
  "active_window_timeout_seconds": 1800,
  "live_proofs": {
    "candidate_live_health": {
      "path": "/ABSOLUTE/final-sq8-live-proof.json",
      "service_unit": "ullm-openai.service",
      "gateway_executable_sha256": "64-lowercase-hex",
      "endpoint_urls": {
        "gateway_healthz": "http://127.0.0.1:PORT/healthz",
        "gateway_readyz": "http://127.0.0.1:PORT/readyz",
        "gateway_models": "http://127.0.0.1:PORT/v1/models",
        "openwebui_health": "http://127.0.0.1:PORT/health",
        "openwebui_models": "http://127.0.0.1:PORT/api/models"
      }
    },
    "rollback_live_health": {
      "path": "/ABSOLUTE/final-aq4-live-proof.json",
      "service_unit": "ullm-openai.service",
      "gateway_executable_sha256": "64-lowercase-hex",
      "endpoint_urls": {
        "gateway_healthz": "http://127.0.0.1:PORT/healthz",
        "gateway_readyz": "http://127.0.0.1:PORT/readyz",
        "gateway_models": "http://127.0.0.1:PORT/v1/models",
        "openwebui_health": "http://127.0.0.1:PORT/health",
        "openwebui_models": "http://127.0.0.1:PORT/api/models"
      }
    },
    "recovery_live_health": {
      "path": "/ABSOLUTE/final-aq4-recovery-live-proof.json",
      "service_unit": "ullm-openai.service",
      "gateway_executable_sha256": "64-lowercase-hex",
      "endpoint_urls": {
        "gateway_healthz": "http://127.0.0.1:PORT/healthz",
        "gateway_readyz": "http://127.0.0.1:PORT/readyz",
        "gateway_models": "http://127.0.0.1:PORT/v1/models",
        "openwebui_health": "http://127.0.0.1:PORT/health",
        "openwebui_models": "http://127.0.0.1:PORT/api/models"
      }
    }
  },
  "stages": {
    "candidate_reconciliation": [
      {
        "argv": ["/absolute/path/to/reviewed-candidate-reconcile"],
        "executable_sha256": "64-lowercase-hex"
      }
    ],
    "candidate_live_health": [
      {
        "argv": ["/absolute/path/to/reviewed-sq8-live-health"],
        "executable_sha256": "64-lowercase-hex"
      }
    ],
    "reverse_reconciliation": [
      {
        "argv": ["/absolute/path/to/reviewed-aq4-reconcile"],
        "executable_sha256": "64-lowercase-hex"
      }
    ],
    "rollback_live_health": [
      {
        "argv": ["/absolute/path/to/reviewed-aq4-live-health"],
        "executable_sha256": "64-lowercase-hex"
      }
    ],
    "recovery_live_health": [
      {
        "argv": ["/absolute/path/to/reviewed-aq4-recovery-live-health"],
        "executable_sha256": "64-lowercase-hex"
      }
    ]
  }
}
```

Each `argv[0]` must be a direct executable with the recorded hash. Shell,
interpreter, shebang-script, `env`, privilege-wrapper, and PATH-resolved
commands are rejected. Executables must be root-owned, single-link, executable,
and not group/world writable.
The core opens each executable without following symlinks, verifies its
metadata and bytes, and executes that already-open descriptor. It also becomes
a Linux child subreaper while each command runs and fails the stage after
terminating any descendant that escapes the command's owned process group.
The reviewed executables must themselves contain the fixed service
reconciliation and live health policy. Do not put credentials, a JWT, shell
source, or a command string in this document.

`timeout_seconds` is the per-command limit.
`active_window_timeout_seconds` is the monotonic limit for the complete
candidate-active sequence (or a manual rollback) and must be at least the
per-command limit and no more than 7200 seconds. A failed activation receives
a separate bounded recovery window so AQ4 restoration is still attempted
after the candidate window expires.

Every endpoint URL must be credential-free HTTP with an explicit port and a
`localhost`, loopback, private, or link-local IP host. Redirects are rejected.
The model endpoints are protected independently by credentials read only at
execution time from these fixed, root-owned, read-only, single-link files:

- Gateway API key: `/etc/ullm/openai-api-key`
- OpenWebUI browser-session JWT:
  `/run/ullm-campaign-secrets/openwebui-session.jwt`

Neither secret is accepted in the operations document, plan, proof, or
outcome. The OpenWebUI file must contain the real browser-login session JWT;
an API key is not a substitute. **That JWT is not currently available, so
activation execution remains blocked even after this runbook and scripts are
prepared.**

Read-only preflight seals both credential files and reports
`credential_seals_ready`. If either file is absent, malformed, mutable, or
below unsafe ancestry, it reports `ready: false` and exits nonzero. Execution
repeats the seal under the activation lock before it publishes an intent or
changes `active.json`.

Both files must be `uid=0,gid=1000`, mode `0640`, and single-link. The JWT
parent `/run/ullm-campaign-secrets` must itself be
`uid=0,gid=1000`, mode `0750`, and must not be nested below a service-user
writable parent.

Exit status alone is not a live-health proof. Each health executable must
publish its designated path as one canonical, root-owned, single-link `0444`
`ullm.served_model.final_activation_live_proof.v1` document. The runner passes
the fresh `ULLM_FINAL_ACTIVATION_EPOCH` and proof destination in its minimal
environment. The proof must bind that epoch and the exact plan hash; active
manifest path/hash, model ID, format ID, worker protocol/hash; service state
and boot ID; gateway/worker PID, PPID, starttime and executable hashes; and
HTTP 200 results for Gateway health/ready/models and OpenWebUI health/models.
Both model endpoints must expose exactly the plan-bound model. The core
requires `captured_at` to fall inside the live stage, then independently
re-observes the fixed service through `systemctl show`, `/proc`, the kernel
boot ID, and all five HTTP endpoints. It requires
`/etc/systemd/system/ullm-openai.service`,
`/etc/ullm/openai-gateway-manifest.env`, the reported service main PID, the
gateway/worker process epoch, and the exact model listing to remain stable
across the probes. The complete proof document and its immutable-file
reference are embedded in the activation or rollback outcome, so a later
manual rollback does not depend on the original proof pathname still
existing. An executable such as `/usr/bin/true` cannot make a health stage
pass.

The AQ4 health executable may reuse the collection routines in
`tools/served_model_aq4_restoration_proof.py`; it must still publish the
final-plan envelope above so the fresh activation epoch and plan identity are
present.

## Prepare the immutable plan

Run this only after the admission boundary is satisfied. All paths must be
absolute. The outcome parent directories must already exist, be root-owned,
and not be group/world writable.

```text
sudo -- /usr/bin/python3.12 -I -S -B /ABSOLUTE/ROOT-OWNED-SEALED-SQ8-SOURCE/tools/prepare-served-model-final-activation.py \
  --plan-id SQ8-FINAL-PLAN-ID \
  --authorization /ABSOLUTE/CAMPAIGN-AUTHORIZATION.json \
  --candidate-manifest /ABSOLUTE/SQ8-CANDIDATE.json \
  --active-manifest /etc/ullm/served-models/active.json \
  --rollback-manifest /ABSOLUTE/AQ4-BACKUP.json \
  --release-bundle /ABSOLUTE/generic-release-bundle-v2.json \
  --systemd-unit /etc/systemd/system/ullm-openai.service \
  --environment-file /etc/ullm/openai-gateway-manifest.env \
  --reviewed-operations /ABSOLUTE/reviewed-final-operations.json \
  --activation-intent /ABSOLUTE/final-activation-intent.json \
  --activation-outcome /ABSOLUTE/final-activation-outcome.json \
  --activation-recovery /ABSOLUTE/final-activation-recovery.json \
  --rollback-outcome /ABSOLUTE/final-rollback-outcome.json \
  --output /ABSOLUTE/final-activation-plan.json
```

Preparation writes only the new
`ullm.served_model.final_activation_plan.v3`. The v3 revision is intentionally
incompatible with plan v2: it binds separate intent, activation outcome,
recovery success, rollback outcome, and three live-proof paths. It rejects an
existing destination. It
revalidates the campaign claim/outcome and re-inventories all six campaign
outputs. It derives the AQ4 bundle path only from the successful outcome,
validates that fresh complete bundle v1 and its raw/browser/promotion
components first, then recomputes SQ8 bundle-v2 validation and binds its
independently recomputed SQ8 campaign inventories and exact claim to the same
outcome. It also proves the actual active bytes still equal the exact AQ4
backup. It inventories and seals the complete AQ4 and SQ8 runtime closures;
missing or empty seals, user-owned ancestry, ACLs, symlinks, hardlinks, and
writable entries are rejected. The active path, service unit, unit file, and
environment file are fixed by production policy; alternate same-byte copies
are rejected. Record the printed plan SHA-256 in the human review record.

## Read-only final preflight

The runner is preflight-only by default:

```text
sudo -- /usr/bin/python3.12 -I -S -B /ABSOLUTE/ROOT-OWNED-SEALED-SQ8-SOURCE/tools/run-served-model-final-activation.py \
  --plan /ABSOLUTE/final-activation-plan.json
```

This command must report `ready: true`, `credential_seals_ready: true`,
`active_manifest_changed: false`, and `commands_executed: false`. It still
performs the expensive complete-bundle validation and campaign-output
re-inventory. It does not acquire a campaign authorization, alter
`active.json`, or run an operation executable.

## Final activation (future operator window only)

After Claude and the user jointly review the exact plan hash and approve the
live window, the only executable form is:

```text
sudo -- /usr/bin/python3.12 -I -S -B /ABSOLUTE/ROOT-OWNED-SEALED-SQ8-SOURCE/tools/run-served-model-final-activation.py \
  --plan /ABSOLUTE/final-activation-plan.json \
  --execute \
  --confirm-plan-sha256 EXACT-PRINTED-PLAN-SHA256 \
  --confirmation ACTIVATE_SQ8_0_FROM_RESTORED_AQ4
```

The SHA-256 and literal confirmation are checked again by the core execution
API, not only by the CLI. The runner reacquires
`.active.json.activation.lock`, reopens the same plan under that lock, and
requires its path, inode identity, bytes, SHA-256, and active target to equal
the pre-lock confirmed snapshot. It repeats preflight and seals both live
credential files under the lock. It then publishes a plan-bound, root-owned,
single-link `0444` `ullm.served_model.final_activation_intent.v1` with
`renameat2(RENAME_NOREPLACE)`, exact re-open, and parent-directory `fsync`.
Only after that durable intent exists does it use Linux
`renameat2(RENAME_EXCHANGE)` with inode/byte comparison to make an
exact-current compare-and-swap of `active.json`; a racing entry is detected
and the exchange is reverted when safely possible. The active directory is
fsynced.

It then runs the two fixed SQ8 stages, verifies candidate bytes again, and
publishes the no-replace read-only
`ullm.served_model.final_activation_outcome.v2`. Publication stages a
single-link file, commits it with `renameat2(RENAME_NOREPLACE)`, exactly
reopens it, and fsyncs the parent; there is no two-hardlink crash window.
That receipt is the success commit boundary. No source recheck or other
fallible action occurs between publication and committing the termination
guard, so a post-publication fault cannot restore AQ4 behind an `activated`
receipt. Before and after each
command the runner re-pins the plan, candidate, rollback, both
AQ4-bundle-v1 and SQ8-bundle-v2 validations, unit, environment,
operations/executables, campaign outcome, and all six campaign inventories.
It also rechecks every sealed worker, tokenizer, product/package payload, and
promotion input immediately before and after each reviewed stage. It compares
the active entry's exact bytes through the already-open directory descriptor.
All stage timeouts are capped by the single monotonic candidate-active
deadline.

Any failure after the replacement attempts exact AQ4 byte restoration while
the same lock remains held, then runs the fixed reverse reconciliation and AQ4
live-health stages. A failure is never reported as safely restored merely
because the AQ4 bytes are present; reconciliation and live health must also
pass.

## Crash or failed-restore recovery

Use this mode only when the durable intent exists and one of these exact
authorities is present:

- no activation outcome (a SIGKILL, power loss, or equivalent crash);
- an activation outcome whose status is `failed_restore`; or
- a successful activation followed by `rollback_incomplete`, including an
  interrupted manual rollback that already restored exact AQ4 bytes.

A normal successful activation with SQ8 still active is not admitted here;
use the ordinary rollback route below.

Read-only recovery preflight:

```text
sudo -- /usr/bin/python3.12 -I -S -B /ABSOLUTE/ROOT-OWNED-SEALED-SQ8-SOURCE/tools/rollback-served-model.py \
  --plan /ABSOLUTE/final-activation-plan.json \
  --recover-failed-activation
```

After reviewing the same plan hash and recovery authority:

```text
sudo -- /usr/bin/python3.12 -I -S -B /ABSOLUTE/ROOT-OWNED-SEALED-SQ8-SOURCE/tools/rollback-served-model.py \
  --plan /ABSOLUTE/final-activation-plan.json \
  --recover-failed-activation \
  --execute \
  --confirm-plan-sha256 EXACT-PRINTED-PLAN-SHA256 \
  --confirmation RECOVER_FAILED_SQ8_0_TO_EXACT_AQ4
```

The recovery core reacquires the same activation lock, revalidates the intent
and any failed activation/rollback receipt, seals both credentials before any
swap, accepts only exact candidate SQ8 or exact rollback AQ4 bytes, restores
AQ4 when needed, then runs reviewed reverse reconciliation and the dedicated
recovery live-health stage. Its successful
`ullm.served_model.final_activation_recovery.v1` receipt is immutable and
one-shot.

If one recovery attempt fails, it does not consume the successful receipt
path. It writes an immutable failure audit and live proof at paths derived
from the plan-bound recovery bases plus a 256-bit attempt ID. A later reviewed
invocation can retry under the same lock; all prior failure audits remain
preserved. Never delete an intent, outcome, recovery audit, or proof to make a
retry pass.

## Later operator rollback

The rollback tool accepts only a plan with an immutable successful activation
outcome and exact candidate bytes currently active. Its default is also
read-only:

```text
sudo -- /usr/bin/python3.12 -I -S -B /ABSOLUTE/ROOT-OWNED-SEALED-SQ8-SOURCE/tools/rollback-served-model.py \
  --plan /ABSOLUTE/final-activation-plan.json
```

After reviewing that preflight and the same plan hash:

```text
sudo -- /usr/bin/python3.12 -I -S -B /ABSOLUTE/ROOT-OWNED-SEALED-SQ8-SOURCE/tools/rollback-served-model.py \
  --plan /ABSOLUTE/final-activation-plan.json \
  --execute \
  --confirm-plan-sha256 EXACT-PRINTED-PLAN-SHA256 \
  --confirmation ROLLBACK_SQ8_0_TO_EXACT_AQ4
```

Rollback uses the same core-level plan confirmation, locked plan identity
check, monotonic deadline, and exact-current exchange. It validates the
embedded successful SQ8 proof from the activation outcome and does not need
to reopen its original proof file. It publishes the plan-bound immutable
rollback outcome. If AQ4 bytes are restored but reverse reconciliation or
live health fails, the outcome is `rollback_incomplete` and records byte
equality separately; it never claims a healthy rollback. That immutable
incomplete outcome authorizes the explicit recovery route above, so a failed
rollback cannot strand exact SQ8 or AQ4 without a plan-bound retry path. A
successfully published `rolled_back` receipt is also the commit boundary: no
fallible source recheck follows it.

Never replace `active.json` manually with `install`, `cp`, or an unreviewed
script. Never bypass the complete bundle, immutable campaign outcome, plan
hash confirmation, or locked rollback route.
