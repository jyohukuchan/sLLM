# AQ4_0 bootstrap closure audit v0.1

This directory contains read-only, retrospective evidence for the exact
AQ4_0 manifest that was active on 2026-07-24.  It is not an activation
authorization, a replacement for fresh campaign evidence, or a claim that
the historical differing-worker bootstrap was a bundle-gated activation.

Fixed live reference:

- manifest: `/etc/ullm/served-models/active.json`
- manifest SHA-256:
  `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a`
- promotion source commit:
  `0cd760568e197e1adb4c4df3d6149591a912f709`
- worker SHA-256:
  `1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b`

`release-evidence-retrospective.json` was assembled from the preserved ten
HTTP/SSE cases and ten lifecycle events by the exact clean, detached
promotion-source commit above.  The current validator independently accepts
it as complete and gate eligible.  The independently published report is
`release-evidence-retrospective-validation.json`.  Their SHA-256 values are
recorded in `EVIDENCE-SHA256SUMS`.  This closes only the core
release-evidence slot of historical bundle v1.

The following still do not exist for the current AQ4_0 identity:

- browser evidence and its validator report;
- a complete `ullm.generic_reasoning_release_bundle.v1`; and
- a normal bundle-gated activation record.

The active manifest also points to a user-owned runtime closure below
`/home`.  Its worker, promotion inputs, tokenizer, package payloads, and
pathname ancestry fail the root-owned runtime seal required by the final SQ8
activation policy.  Moving to protected absolute paths changes the manifest
identity, so these retrospective artifacts must not be reused for that future
hardened manifest.

See
`journal/2026/07/24/aq4-bootstrap-closure-audit.md` for the complete findings,
commands, and remediation boundary.
