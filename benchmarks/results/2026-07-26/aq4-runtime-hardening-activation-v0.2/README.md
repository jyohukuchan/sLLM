# AQ4 runtime-hardening activation readiness v0.2

This directory records corrective preparation only. No command invoked activation or `--execute`, and `/etc/ullm/served-models/active.json` remained SHA-256 `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a`.

The final immutable plan is `/opt/ullm/aq4-runtime-hardening-v0.1/activation-v0.2-r3/activation-plan.json`, SHA-256 `0e12fe09ad4d00578ee74f1bcc730a6b401e63a6fc91bb1d237346251e8f81f8`.

- `preflight-before-isolated-r3.json` is the final plan's initial read-only state: only the required isolated-worker receipt was missing.
- `isolated-candidate-worker-preflight-r3-summary.json` is a credential-free summary of the immutable receipt from the R9700 (`gfx1201`) candidate-worker readiness check.
- `read-only-preflight-r3.json` is the final normal read-only plan check: `ready: true`, no blockers, and `production_activation_performed: false`.
- `sealed-artifacts-r3.json` records the final source/launcher/plan seals and the service state after preparation.
- `preflight-before-isolated.json` and `reviewed-operations.json` preserve the first new sealed plan, which was superseded before worker start when safe key-presence inspection showed that gateway MainPID does not carry worker HIP/manifest bindings. `*-r2.json` preserves the second prepared plan, superseded before activation to retain successful endpoint states if the readiness deadline expires. Neither was executed or wrote the active manifest.

`reviewed-operations-r3.json` is the source copy of the final root-owned immutable reviewed-operations document. No operations document contains credential values.
