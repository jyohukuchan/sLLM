# SQ8_0 final activation and exact AQ4_0 rollback runbook v0.1

Status: implementation-only preparation. **Do not execute the activation or
rollback commands during the current work.**

This runbook applies only to the independent Qwen3-14B-FP8 `SQ8_0` served
model. It does not apply to the historical AQ4 SQ8 overlay.

## Admission boundary

Do not prepare or execute a final plan until all of the following exist from
the same clean release commit and identity:

1. the final sealed SQ8 worker and build receipt;
2. an immutable, consumed cross-model campaign authorization;
3. its `succeeded_restored` immutable outcome, including the complete
   `sq8_full`, `reasoning_release`, and `reasoning_browser` output inventories
   and a live AQ4 restoration proof;
4. the exact read-only AQ4 manifest backup named by that authorization;
5. a complete, gate-eligible
   `ullm.generic_reasoning_release_bundle.v2`;
6. the frozen SQ8 candidate manifest, systemd unit, and environment file whose
   hashes equal the authorization and bundle; and
7. a human-reviewed operations document as described below.

The fresh browser campaigns require a real private OpenWebUI browser-session
JWT and the full campaigns require the production GPU/service window. Those
runs are outside this implementation task.

## Reviewed operations document

The final tools do not accept command JSON on their command line. The four
operational stages are instead fixed in a separately reviewed, canonical JSON
file, made read-only (`0444`), owned by root, and linked exactly once:

```json
{
  "schema_version": "ullm.served_model.final_activation_operations.v1",
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
  `/run/ullm/sq8-v2-cross-model-openwebui-session.jwt`

Neither secret is accepted in the operations document, plan, proof, or
outcome. The OpenWebUI file must contain the real browser-login session JWT;
an API key is not a substitute. **That JWT is not currently available, so
activation execution remains blocked even after this runbook and scripts are
prepared.**

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
sudo tools/prepare-served-model-final-activation.py \
  --plan-id SQ8-FINAL-PLAN-ID \
  --authorization /ABSOLUTE/CAMPAIGN-AUTHORIZATION.json \
  --candidate-manifest /ABSOLUTE/SQ8-CANDIDATE.json \
  --active-manifest /etc/ullm/served-models/active.json \
  --rollback-manifest /ABSOLUTE/AQ4-BACKUP.json \
  --release-bundle /ABSOLUTE/generic-release-bundle-v2.json \
  --systemd-unit /etc/systemd/system/ullm-openai.service \
  --environment-file /etc/ullm/openai-gateway-manifest.env \
  --reviewed-operations /ABSOLUTE/reviewed-final-operations.json \
  --activation-outcome /ABSOLUTE/final-activation-outcome.json \
  --rollback-outcome /ABSOLUTE/final-rollback-outcome.json \
  --output /ABSOLUTE/final-activation-plan.json
```

Preparation writes only the new plan. It rejects an existing destination. It
revalidates the campaign claim/outcome, re-inventories all three campaign
outputs, recomputes bundle v2 validation, binds its independently recomputed
`reasoning_release_campaign` inventory and exact claim to the successful
transaction outcome, and proves the actual active bytes still equal the exact
AQ4 backup. The active path, service unit, unit file, and environment file are
fixed by production policy; alternate same-byte copies are rejected. Record
the printed plan SHA-256 in the human review record.

## Read-only final preflight

The runner is preflight-only by default:

```text
sudo tools/run-served-model-final-activation.py \
  --plan /ABSOLUTE/final-activation-plan.json
```

This command must report `ready: true`,
`active_manifest_changed: false`, and `commands_executed: false`. It still
performs the expensive complete-bundle validation and campaign-output
re-inventory. It does not acquire a campaign authorization, alter
`active.json`, or run an operation executable.

## Final activation (future operator window only)

After Claude and the user jointly review the exact plan hash and approve the
live window, the only executable form is:

```text
sudo tools/run-served-model-final-activation.py \
  --plan /ABSOLUTE/final-activation-plan.json \
  --execute \
  --confirm-plan-sha256 EXACT-PRINTED-PLAN-SHA256 \
  --confirmation ACTIVATE_SQ8_0_FROM_RESTORED_AQ4
```

The SHA-256 and literal confirmation are checked again by the core execution
API, not only by the CLI. The runner reacquires
`.active.json.activation.lock`, reopens the same plan under that lock, and
requires its path, inode identity, bytes, SHA-256, and active target to equal
the pre-lock confirmed snapshot. It repeats preflight under the lock, then
uses Linux `renameat2(RENAME_EXCHANGE)` with inode/byte comparison to make an
exact-current compare-and-swap of `active.json`; a racing entry is detected
and the exchange is reverted when safely possible. The active directory is
fsynced.

It then runs the two fixed SQ8 stages, verifies candidate bytes again, and
publishes the no-replace read-only activation outcome. Before and after each
command the runner re-pins the plan, candidate, rollback, bundle validation,
unit, environment, operations/executables, campaign outcome and all three
campaign inventories, and compares the active entry's exact bytes through the
already-open directory descriptor. All stage timeouts are capped by the
single monotonic candidate-active deadline.

Any failure after the replacement attempts exact AQ4 byte restoration while
the same lock remains held, then runs the fixed reverse reconciliation and AQ4
live-health stages. A failure is never reported as safely restored merely
because the AQ4 bytes are present; reconciliation and live health must also
pass.

## Later operator rollback

The rollback tool accepts only a plan with an immutable successful activation
outcome and exact candidate bytes currently active. Its default is also
read-only:

```text
sudo tools/rollback-served-model.py \
  --plan /ABSOLUTE/final-activation-plan.json
```

After reviewing that preflight and the same plan hash:

```text
sudo tools/rollback-served-model.py \
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
equality separately; it never claims a healthy rollback.

Never replace `active.json` manually with `install`, `cp`, or an unreviewed
script. Never bypass the complete bundle, immutable campaign outcome, plan
hash confirmation, or locked rollback route.
