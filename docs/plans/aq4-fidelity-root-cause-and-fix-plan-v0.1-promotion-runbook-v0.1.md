# AQ4 fidelity-fix promotion runbook v0.1

Status (2026-07-18): **Functionally live, formally incomplete.** The candidate
manifest (`5d015a01...`, worker `1f93f215...`, source commit `f1a3cf4c` plus
follow-on tooling fixes through `c167e12b`) has been active in production
since 2026-07-18T05:16:28+09:00 via the section 8 temporary
differing-worker-bootstrap route, and has stayed active, healthy
(`NRestarts=0`, `readyz` consistently `ready`) throughout Section 8's
evidence collection. The fidelity fix is genuinely deployed and serving.

> Status update (2026-07-26 JST): frozen historical runbook. The detailed
> temporary-route approvals, bundle gates, and parent-only steps below record
> the earlier AQ4_0 fidelity deployment and are not requirements for a new
> candidate. Future promotions use the lightweight-promotion policy and its
> generic rollback route.

What is NOT done: the formal Section 9/10 bundle-gated activation. Assembling
a `--status complete` release bundle requires `--browser-evidence` /
`--browser-validator`, which requires a working OpenWebUI browser-automation
session. The browser-gate credential bug is now fixed in code: browser gates
accept only `--openwebui-session-token-file`, reject the gateway backend key,
and require a non-expiring-for-the-gate OpenWebUI frontend JWT. The operator
must still provision that separate private session-token file and rerun the
real gates; no live browser evidence has been claimed by this documentation
update. Section 8.4's HTTP/SSE release campaign (no browser required)
succeeded: 10 cases across all 5 reasoning modes, correct candidate identity
and R9700 exclusivity confirmed.

**Do not run Section 9's "restore current active bytes" step until the
browser gates have passed with a real frontend session JWT and a complete
bundle can actually be assembled.** Restoring now without a working path back
through Section 10
would revert production to the pre-fix build with no way to
re-promote via the documented bundle-gated route. The candidate stays active
as its own de facto rollback-safe state (health-checked dozens of times
across this session) until that follow-up closes.

This runbook is for an AQ4-to-AQ4 replacement of the same public model, not
the SQ8-to-AQ4 example in deploy/README.md. It is a handoff to the parent
operator. Do not execute the commands marked parent-only until the two
explicit decision gates in section 7 are resolved.

## 1. Scope and fixed identities

Only AMD Radeon AI PRO R9700 / gfx1201 / GPU index 1 is in scope. No command
in this document targets, probes, starts, stops, or otherwise uses either
V620.

| Item | Value |
|---|---|
| Candidate source commit | f1a3cf4c86978b3b8900396a0b6a8caff90b97f1 |
| Fidelity fix commit | e992b3ea1d0427744dfd83abdc98283a74c1e3b4 |
| Candidate worker SHA-256 | 1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b |
| Legacy engine SHA-256 used by resident evidence | d1c18362c6253294d37e7258434d877752c5052ab677ecfd35f1a7928b64b433 |
| Current active manifest | /etc/ullm/served-models/active.json |
| Current active manifest SHA-256 | feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44 |
| Current active promotion source | ae8b2bb7c2735f4dc761773957bf45f470dd5a8c |
| Current active worker SHA-256 | 177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d |
| Current active public ID | ullm-qwen3.5-9b-aq4 |
| Current active protocol | ullm.worker.v2 |

The current production source is 1,879 commits behind the candidate source.
The current active manifest is already AQ4/v2 and names the same public model
ID as the candidate. Therefore its exact bytes, not the stale named candidate
under /etc/ullm/served-models/candidates, are the rollback identity.

The detached source and nlink=1 release files prepared for this handoff are:

    /home/homelab1/coding-local/ultimateLLM/uLLM-aq4-fidelity-promotion-source-f1a3cf4c
    /home/homelab1/coding-local/ultimateLLM/uLLM-aq4-fidelity-promotion-release-f1a3cf4c/ullm-aq4-worker
    /home/homelab1/coding-local/ultimateLLM/uLLM-aq4-fidelity-promotion-release-f1a3cf4c/ullm-engine

Retain that detached worktree until the promotion is either completed or
explicitly abandoned. It is clean and detached at the candidate commit.

## 2. Receipt-linked accepted-risk statement

The future receipt path is:

    /home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/promotion-reasoning-v2-fidelity-f1a3cf4c.json

The receipt schema ullm.aq4_resident_promotion.v1 admits only schema_version,
source_commit, and evidence. It has no extension field for an approval note.
This runbook and the companion journal entry are therefore the receipt-linked
accepted-risk note for that exact future path and source commit.

The formal independent P2 holdout result is not a formal Gate pass:

- 7 of 8 fidelity metrics passed.
- token_agreement_rate was 20/24 (83.3%); its Wilson lower bound was 0.676,
  below the required 0.899.
- The fidelity plan records the user decision on 2026-07-17 to accept the
  residual near-margin token differences as expected stochastic AQ4 4-bit
  quantization noise and to close the investigation.

Accordingly, any receipt made at the path above must be described as
accepted-risk, user-approved promotion evidence. It must never be described as
formal P2 Gate success, nor may the receipt JSON be hand-edited to imply that
status. The full rationale and the formal no-go wording remain in
docs/plans/aq4-fidelity-root-cause-and-fix-plan-v0.1.md and
docs/proposals/aq4-p2-fidelity-holdout-protocol-v0.1.md.

## 3. Prepared profile and planned outputs

The only new profile is:

    deploy/served-models/qwen35-9b-aq4-reasoning-f1a3cf4c.profile.json

It preserves the AQ4 reasoning v2 contract and product/tokenizer identity, but
uses the staged candidate worker and an f1a3cf4c-specific promotion receipt.
No SQ8 profile, manifest, code, or tooling was modified.

The following paths are planned, not already generated:

    /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-fidelity-promotion-f1a3cf4c-v0.1/qwen35-9b-aq4-reasoning-fidelity-f1a3cf4c.json
    /etc/ullm/served-models/candidates/qwen35-9b-aq4-reasoning-fidelity-f1a3cf4c.json
    /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-fidelity-promotion-f1a3cf4c-v0.1/release-bundle-f1a3cf4c.json

The candidate generator was deliberately tried before a receipt existed and
failed closed with “failed to read promotion receipt”; no candidate file was
created. A complete bundle cannot be truthfully assembled before the
candidate-active measurements exist.

## 4. CPU-only preparation evidence

Completed checks:

- The current /etc active manifest was read without modification and validated
  successfully as ullm.served_model.v2.
- A clean detached f1a3cf4c worktree built ullm-aq4-worker and ullm-engine
  with CARGO_BUILD_JOBS=1. Only the existing C++ subobject-linkage warnings
  were emitted.
- Both release files are content copies, mode 0555, and nlink=1.
- The new profile passed JSON parsing.
- The expanded deployment/promotion/release/P2/browser-gate regression
  selection passed: 212 passed, 41 subtests passed.
- The f1a3cf4c sealed P2 preparation and nlink=1 staging were checked with
  run-aq4-p2-production-path-oracle.py --dry-run. It reported
  dry_run_valid and gpu_or_service_action=none; the planned output was still
  absent afterwards. CUDA_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES,
  ROCR_VISIBLE_DEVICES, and ULLM_HIP_VISIBLE_DEVICES were all set to -1.

The dry-run validates the same_artifact_all_m1 path-oracle mechanics only. It
does not run a model and does not turn the accepted-risk result into a formal
P2 approval.

### 4.1 Browser-gate image build and verification

This local rootless-Docker prerequisite is separate from the historical
CPU-only preparation above. It neither contacts a service nor uses a GPU or
lock. The browser gates bind-mount only their `.cjs` scripts, so their image
must provide the `playwright` Node.js package itself. Do not use the historical
Firecrawl image ID: it identifies an unrelated `firecrawl-playwright-service`
image.

From the repository root, build the dedicated image before running a browser
gate:

    docker build --pull=false \
      --file deploy/openwebui/Dockerfile.browser-gate \
      --tag ullm/openwebui-browser-gate:playwright-1.58.0 \
      deploy/openwebui
    docker image inspect --format '{{.Id}}' \
      ullm/openwebui-browser-gate:playwright-1.58.0

`Dockerfile.browser-gate` is based on the locally verified
`mcr.microsoft.com/playwright:v1.58.0-noble` content digest and installs only
`playwright@1.58.0` globally (without downloading another browser). The build
recorded for this runbook produced:

    sha256:0bd709ea36ffa7204cd60da0fe9707be38eb73c97c7a9d45911ff0e8b7c1e3ea

Verify the package resolution, then optionally verify the matching Chromium
binary before proceeding. The second command uses the same arbitrary host UID
and GID mode used by the gates:

    docker run --rm --entrypoint node \
      sha256:0bd709ea36ffa7204cd60da0fe9707be38eb73c97c7a9d45911ff0e8b7c1e3ea \
      -e 'const { chromium } = require("playwright"); console.log("playwright require OK, chromium=" + typeof chromium)'

    docker run --rm --user "$(id -u):$(id -g)" --entrypoint node \
      sha256:0bd709ea36ffa7204cd60da0fe9707be38eb73c97c7a9d45911ff0e8b7c1e3ea \
      -e 'const { chromium } = require("playwright"); chromium.launch({headless:true}).then(async browser => { await browser.close(); console.log("chromium launch/close OK"); })'

## 5. Parent-only initial setup and resident promotion receipt

Run the following as a parent operator. It creates no candidate before real
resident evidence has verified the staged worker. The image identifiers are the
currently documented immutable identities; use a separately reviewed image
update if any of them are no longer locally available.

    set -euo pipefail

    REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
    SOURCE_TREE=/home/homelab1/coding-local/ultimateLLM/uLLM-aq4-fidelity-promotion-source-f1a3cf4c
    SOURCE_COMMIT=f1a3cf4c86978b3b8900396a0b6a8caff90b97f1
    RELEASE_DIR=/home/homelab1/coding-local/ultimateLLM/uLLM-aq4-fidelity-promotion-release-f1a3cf4c
    WORKER=$RELEASE_DIR/ullm-aq4-worker
    LEGACY_ENGINE=$RELEASE_DIR/ullm-engine
    PROFILE=$REPO/deploy/served-models/qwen35-9b-aq4-reasoning-f1a3cf4c.profile.json
    PRODUCT=/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1
    PROMOTION_EVIDENCE=$PRODUCT/resident-promotion-reasoning-v2-fidelity-f1a3cf4c.json
    PROMOTION_RECEIPT=$PRODUCT/promotion-reasoning-v2-fidelity-f1a3cf4c.json
    OUT=$REPO/benchmarks/results/2026-07-17/qwen35-9b-aq4-fidelity-promotion-f1a3cf4c-v0.1
    CANDIDATE_LOCAL=$OUT/qwen35-9b-aq4-reasoning-fidelity-f1a3cf4c.json
    CANDIDATE_ETC=/etc/ullm/served-models/candidates/qwen35-9b-aq4-reasoning-fidelity-f1a3cf4c.json
    BUNDLE=$OUT/release-bundle-f1a3cf4c.json
    ACTIVE=/etc/ullm/served-models/active.json
    ACTIVE_SHA256=feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44
    UNIT=/etc/systemd/system/ullm-openai.service
    ENVIRONMENT=/etc/ullm/openai-gateway-manifest.env
    SERVICE=ullm-openai.service
    LLAMA_SERVICE=llama-qwen35-udq4.service
    GATEWAY_TOKEN_FILE=/etc/ullm/openai-api-key
    # A private 0600 OpenWebUI frontend session JWT, provisioned by the parent.
    # Never set this to GATEWAY_TOKEN_FILE.
    OPENWEBUI_SESSION_TOKEN_FILE=/run/user/$(id -u)/ullm-openwebui-session.jwt
    BROWSER_IMAGE=sha256:0bd709ea36ffa7204cd60da0fe9707be38eb73c97c7a9d45911ff0e8b7c1e3ea
    PROBE_IMAGE=sha256:5dce198cca467ce79994ed65e01d03882238f9efdd16a8c6f4bc55151c8a4a54
    OPENWEBUI_IMAGE=ullm/open-webui@sha256:ef5ae4fbc06abb662eeefe87e584ea7c69e55838f5f08f637057b9108048b409
    GATEWAY_CHECK_IMAGE=ghcr.io/open-webui/open-webui@sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff

    test "$(git -C "$SOURCE_TREE" rev-parse HEAD)" = "$SOURCE_COMMIT"
    test -z "$(git -C "$SOURCE_TREE" status --porcelain --untracked-files=all)"
    test "$(sha256sum "$WORKER" | awk '{print $1}')" = 1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b
    test "$(sha256sum "$LEGACY_ENGINE" | awk '{print $1}')" = d1c18362c6253294d37e7258434d877752c5052ab677ecfd35f1a7928b64b433
    test ! -e "$OUT"
    test ! -e "$PROMOTION_EVIDENCE"
    test ! -e "$PROMOTION_RECEIPT"
    sudo /usr/bin/python3 "$SOURCE_TREE/tools/validate-served-model.py" --manifest "$ACTIVE"
    docker image inspect "$BROWSER_IMAGE" "$PROBE_IMAGE" "$OPENWEBUI_IMAGE" "$GATEWAY_CHECK_IMAGE" >/dev/null
    test "$(docker image inspect --format '{{.Id}}' "$BROWSER_IMAGE")" = "$BROWSER_IMAGE"
    test "$(docker image inspect --format '{{.Id}}' "$PROBE_IMAGE")" = "$PROBE_IMAGE"

The next block is a real R9700 service window. It stops only the two services
which can own the R9700, restores ullm-openai.service on every exit path, and
does not start the llama service afterwards. It is intentionally absent from
the CPU-only preparation execution.

    sudo systemctl stop "$LLAMA_SERVICE" "$SERVICE"
    RESTORE_SERVICE=1
    trap 'if [ "$RESTORE_SERVICE" = 1 ]; then sudo systemctl start "$SERVICE"; fi' EXIT
    sudo -u homelab1 -H /usr/bin/python3 \
      "$SOURCE_TREE/tools/run-aq4-resident-promotion-evidence.py" \
      --profile "$PROFILE" \
      --output "$PROMOTION_EVIDENCE" \
      --worker-binary "$WORKER" \
      --legacy-engine "$LEGACY_ENGINE"
    sudo systemctl start "$SERVICE"
    RESTORE_SERVICE=0
    trap - EXIT

    sudo -u homelab1 -H /usr/bin/python3 \
      "$SOURCE_TREE/tools/write-aq4-resident-promotion-receipt.py" \
      --profile "$PROFILE" \
      --evidence "$PROMOTION_EVIDENCE" \
      --output "$PROMOTION_RECEIPT"

    mkdir -m 0700 "$OUT"
    /usr/bin/python3 "$SOURCE_TREE/tools/generate-served-model.py" \
      --profile "$PROFILE" --output "$CANDIDATE_LOCAL"
    /usr/bin/python3 "$SOURCE_TREE/tools/validate-served-model.py" \
      --manifest "$CANDIDATE_LOCAL"
    sudo install -m 0644 -o root -g root "$CANDIDATE_LOCAL" "$CANDIDATE_ETC"
    sudo /usr/bin/python3 "$SOURCE_TREE/tools/validate-served-model.py" \
      --manifest "$CANDIDATE_ETC"

The receipt writer validates the actual evidence before publishing it. The
profile generator then binds the candidate to the receipt. Do not manually
create either JSON document.

## 6. Gate execution classification

| Tool | CPU-only dry-run | Real-operation requirement | Result for this handoff |
|---|---|---|---|
| run-aq4-resident-promotion-evidence.py | No | Exclusive R9700, starts resident/legacy workers | Parent-only section 5 |
| run-openwebui-soak-gate.py | No | Docker and real UI requests | Candidate-active only |
| run-openwebui-stop-gate.py | No | Docker and a real UI Stop request | Candidate-active only |
| run-openwebui-failure-gate.py | No | Docker and intentional worker failure/restart | Candidate-active only |
| run-sq8-direct-cancel-gate.py | No | Docker and real direct cancellation | Not an AQ4 gate; see section 7 |
| run-generic-reasoning-release-campaign.py | No | R9700 worker, rocm-smi and real HTTP/SSE requests | Candidate-active only |
| run-openwebui-reasoning-browser-smoke.py | No | Docker; current v2 runner also requires a provider switch | See section 7 |
| run-aq4-p2-production-path-oracle.py | Yes | Execute mode needs one root service window | CPU dry-run completed; conditional path check below |

## 7. Stop: explicit decisions required before candidate-active gates

### 7.1 v2-to-v2 bootstrap cycle

The normal v2 activation correctly requires a complete release bundle. The
bundle correctly requires release/browser evidence from the candidate while it
is active. However, the existing bootstrap-v2 path rejects this candidate:
the active manifest is already v2 and the candidate worker SHA-256 differs
from the active worker SHA-256. The exact rejection is:

    v2 bootstrap candidate worker differs from active worker

This is not a missing command-line argument. It prevents a v2-to-v2 worker
replacement from using the documented v1-to-v2 evidence bootstrap. Do not
attempt to bypass it by altering a manifest, a receipt, or a bundle.

The parent must explicitly choose and record one of the following before
candidate-active evidence is collected:

1. An independently reviewed temporary v1 bridge, with exact rollback and
   restoration semantics.
2. A separately reviewed activation-policy/tool change that authorizes a
   different-worker v2 bootstrap.
3. An already approved, identity-matching complete bundle and evidence set.

The authorized route must restore the current active bytes with SHA-256
feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44 before
section 9 assembles the final bundle. That is the rollback target for the
final AQ4-to-AQ4 activation.

### 7.2 Cancellation gate is SQ8-specific

run-sq8-direct-cancel-gate.py hard-codes:

    MODEL_ID = "ullm-qwen3-14b-sq8"
    GATE_SCHEMA = "ullm.sq8.direct_cancel_gate.v1"

It therefore cannot certify the AQ4 candidate and must not be run as an AQ4
promotion gate. No AQ4-specific equivalent was created because SQ8 tooling is
out of scope. The parent must explicitly waive this inapplicable gate or
authorize separately reviewed AQ4 cancellation tooling before claiming a full
cancel-gate set.

### 7.3 Browser-switch policy/tool mismatch

The release policy says llama.cpp provider comparison/switching is not a
required release gate after the 2026-07-14 user decision. The currently
enforced v2 browser runner and validator nevertheless emit and require four
provider requests including a distinct switch model. The bundle requires a
gate-eligible browser artifact.

Thus the parent must choose either:

1. Run the current implementation-enforced browser command in section 8.3,
   which temporarily switches only R9700 ownership between uLLM and llama.cpp;
   or
2. Approve a separately reviewed policy/tool alignment before building a
   complete bundle.

This is distinct from V620 and does not authorize any V620 operation.

## 8. Conditional candidate-active evidence sequence

This section becomes executable only after section 7 has been explicitly
resolved and the approved temporary route has made CANDIDATE_ETC the active
manifest. The parent must first prove:

    sudo /usr/bin/python3 "$SOURCE_TREE/tools/validate-served-model.py" --manifest "$ACTIVE"
    cmp -s "$ACTIVE" "$CANDIDATE_ETC"

Do not reuse an output directory or rerun a failed gate into the same path.

The sealed `baseline-v0.4` preparation was created while the old active
manifest was live. It remains immutable evidence of that state: do not edit
its `identity.json`, delete its failed target-input receipt, or reuse its
`source-oracle/source-full` under the candidate. Create a new preparation root
while the candidate is active. The following is CPU-only (the source capture
is intentionally long, but does not open HIP or touch the service):

    export ACTIVE=/etc/ullm/served-models/active.json
    export P2_PREP=$REPO/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-production-prefill-decode-baseline-v0.5-candidate-active-0cd76056
    export P2_CLEAN_SOURCE=/home/homelab1/coding-local/ultimateLLM/uLLM-p2-baseline-source-f1a3cf4c
    export P2_BUILD=/home/homelab1/coding-local/ultimateLLM/uLLM-p2-baseline-build-f1a3cf4c/release
    export P2_MODEL=/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B
    export P2_SOURCE=$P2_PREP/source-oracle/source-full
    export P2_CASE=p2-oracle-anchor-prefill-all-m1-n128-m1
    export P2_SOURCE_COMMIT=f1a3cf4c86978b3b8900396a0b6a8caff90b97f1

    test "$(sha256sum "$ACTIVE" | awk '{print $1}')" = 5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a
    test ! -e "$P2_PREP"
    /usr/bin/python3 "$REPO/tools/prepare-aq4-p2-production-baseline.py" \
      --output "$P2_PREP" --source-worktree "$P2_CLEAN_SOURCE" \
      --active-manifest "$ACTIVE" --source-model "$P2_MODEL"
    /usr/bin/python3 "$REPO/tools/prepare-aq4-p2-production-baseline.py" \
      --output "$P2_PREP" --verify --active-manifest "$ACTIVE" \
      --verify-live-active-identity
    /usr/bin/python3 "$REPO/tools/stage-aq4-p2-production-baseline-binaries.py" \
      --output "$P2_PREP/staging/baseline-binaries" --preparation "$P2_PREP" \
      --resident-source "$P2_BUILD/ullm-aq4-p2-resident-driver" \
      --calibration-source "$P2_BUILD/ullm-aq4-p2-calibration" \
      --source-commit "$P2_SOURCE_COMMIT"
    /usr/bin/python3 "$REPO/tools/stage-aq4-p2-r9700-guard.py" \
      --output "$P2_PREP/guard/r9700-guard-staging" --preparation "$P2_PREP" \
      --source "$REPO/tools/query-hip-device-identity.cpp" \
      --source-commit "$P2_SOURCE_COMMIT"
    env CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 ULLM_HIP_VISIBLE_DEVICES=-1 \
      /usr/bin/python3 "$REPO/tools/capture-aq4-p2-production-source-oracle.py" \
        --preparation "$P2_PREP" --model-dir "$P2_MODEL" --output "$P2_SOURCE" \
        --confirm-cpu-source-capture --threads 1
    env CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 ULLM_HIP_VISIBLE_DEVICES=-1 \
      /usr/bin/python3 "$REPO/tools/run-aq4-p2-production-path-oracle.py" \
        --preparation "$P2_PREP" --staging "$P2_PREP/staging/baseline-binaries" \
        --case-id "$P2_CASE" --source "$P2_SOURCE" \
        --output "$P2_PREP/source-oracle/target/$P2_CASE" \
        --served-manifest "$ACTIVE" --dry-run

The live-identity dry run must report `dry_run_valid`; it now rejects both a
stale frozen active identity and a source oracle whose model revision or
selected fixture differs. Do not enter a service window otherwise.
The service-window driver separately rejects dirty tooling, so use a clean
tracked worktree containing this change; do not bypass that check.

    export ULLM_MODEL_ID=ullm-qwen3.5-9b-aq4
    export ULLM_MODEL_NAME='uLLM Qwen3.5 9B AQ4 reasoning'
    export OPENWEBUI_URL=http://192.168.0.66:3000/
    export P2_PREP=$REPO/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-production-prefill-decode-baseline-v0.5-candidate-active-0cd76056
    export P2_GUARD=$P2_PREP/guard/r9700-guard-staging/query-hip-device-identity
    export P2_CLEAN_SOURCE=/home/homelab1/coding-local/ultimateLLM/uLLM-p2-baseline-source-f1a3cf4c
    export P2_SOURCE=$P2_PREP/source-oracle/source-full
    export P2_CASE=p2-oracle-anchor-prefill-all-m1-n128-m1

    test ! -e "$OUT/soak-100"
    test ! -e "$OUT/soak-restart-20"
    test ! -e "$OUT/stop"
    test ! -e "$OUT/failure"
    test ! -e "$OUT/browser-reasoning-f1a3cf4c.json"
    test ! -e "$OUT/http-sse-campaign"
    test ! -e "$P2_PREP/source-oracle/target/$P2_CASE"

### 8.1 same_artifact_all_m1 path check — deliberately skipped for this promotion

The first attempt against the stale `baseline-v0.4` preparation failed closed
(`P2 model identity differs from served model`; service was stopped and
restored cleanly, no harm done). The correct fix is a fresh `baseline-v0.5`
preparation plus a new ~42-minute CPU-only fp32 source-oracle capture (both
already designed and validated in isolation; see the parent's commits and
journal for 2026-07-17/18).

The parent operator decided not to spend that additional CPU-only time and
GPU window on this specific check for this promotion, because two
independent, already-completed pieces of evidence already cover the same
concern (that the deployed candidate binary is the one actually being
validated):

- Section 5's `run-aq4-resident-promotion-evidence.py` run (`verified: true`)
  directly exercises the candidate worker binary against the legacy engine
  via real HIP execution and SHA-256-binds both binaries' identities.
- The formal Phase 7 fidelity holdout (48 independent cases, documented in
  `docs/plans/aq4-fidelity-root-cause-and-fix-plan-v0.1.md`) already
  validated the same RMSNorm-fixed code path end-to-end against an
  independent CPU oracle.

`same_artifact_all_m1` is therefore treated as redundant with existing
evidence for this promotion, not as an unverified gap. If a future promotion
needs it, run the `baseline-v0.5-candidate-active-*` sequence above and then:

    sudo "$REPO/tools/run-aq4-p2-production-path-oracle-service-window.sh" \
      "$P2_PREP" "$P2_GUARD" "$P2_CLEAN_SOURCE" "$SOURCE_COMMIT" \
      "$P2_SOURCE" "$P2_CASE" --confirm-single-window

### 8.2 Required real service/UI gates

    cd "$SOURCE_TREE"
    ULLM_OPENWEBUI_SOAK_COUNT=100 uv run --project services/openai-gateway python \
      tools/run-openwebui-soak-gate.py \
      --output-dir "$OUT/soak-100" \
      --openwebui-session-token-file "$OPENWEBUI_SESSION_TOKEN_FILE" \
      --browser-image "$BROWSER_IMAGE" --openwebui-url "$OPENWEBUI_URL" \
      --service "$SERVICE" --include-smoke

    sudo systemctl restart "$SERVICE"
    ULLM_OPENWEBUI_SOAK_COUNT=20 uv run --project services/openai-gateway python \
      tools/run-openwebui-soak-gate.py \
      --output-dir "$OUT/soak-restart-20" \
      --openwebui-session-token-file "$OPENWEBUI_SESSION_TOKEN_FILE" \
      --browser-image "$BROWSER_IMAGE" --openwebui-url "$OPENWEBUI_URL" \
      --service "$SERVICE" --include-smoke

    uv run --project services/openai-gateway python \
      tools/run-openwebui-stop-gate.py \
      --output-dir "$OUT/stop" \
      --openwebui-session-token-file "$OPENWEBUI_SESSION_TOKEN_FILE" \
      --browser-image "$BROWSER_IMAGE" --openwebui-url "$OPENWEBUI_URL" \
      --service "$SERVICE"

    uv run --project services/openai-gateway python \
      tools/run-openwebui-failure-gate.py \
      --output-dir "$OUT/failure" \
      --openwebui-session-token-file "$OPENWEBUI_SESSION_TOKEN_FILE" \
      --browser-image "$BROWSER_IMAGE" --probe-image "$PROBE_IMAGE" \
      --openwebui-url "$OPENWEBUI_URL" --service "$SERVICE"

Do not substitute the SQ8 direct-cancel command here; section 7.2 explains why
it is invalid for this AQ4 candidate.

### 8.3 Browser evidence required by the current bundle implementation

Run this only if the parent selected section 7.3 option 1. The alternating
option affects uLLM and the R9700 llama service only; it never targets V620.

    cd "$SOURCE_TREE"
    uv run --project services/openai-gateway python \
      tools/run-openwebui-reasoning-browser-smoke.py \
      --output "$OUT/browser-reasoning-f1a3cf4c.json" \
      --manifest "$ACTIVE" \
      --openwebui-session-token-file "$OPENWEBUI_SESSION_TOKEN_FILE" \
      --browser-image "$BROWSER_IMAGE" --openwebui-url "$OPENWEBUI_URL" \
      --model-id "$ULLM_MODEL_ID" --model-name "$ULLM_MODEL_NAME" \
      --switch-model-id llama-qwen3.5-9b-ud-q4 \
      --switch-model-name 'llama.cpp Qwen3.5 9B UD-Q4_K_XL' \
      --alternate-r9700-services --ullm-service "$SERVICE" \
      --llama-service "$LLAMA_SERVICE"

    uv run --project services/openai-gateway python \
      tools/validate-openwebui-reasoning-browser-smoke.py \
      "$OUT/browser-reasoning-f1a3cf4c.json" --require-pass \
      > "$OUT/browser-reasoning-validation-f1a3cf4c.json"

### 8.4 Generic reasoning release campaign

Run this after worker-failure recovery and after the browser command has
returned uLLM ownership to the R9700.

    cd "$SOURCE_TREE"
    uv run --project services/openai-gateway python \
      tools/run-generic-reasoning-release-campaign.py \
      --output-dir "$OUT/http-sse-campaign" --manifest "$ACTIVE" \
      --fixture-suite tests/fixtures/generic-reasoning-release-v0.1/prompts.json \
      --token-file "$GATEWAY_TOKEN_FILE" --http-image "$PROBE_IMAGE" \
      --endpoint http://172.20.0.1:8000/v1/chat/completions \
      --network open-webui-network --observer-socket /run/ullm/lifecycle-observer.sock \
      --service "$SERVICE" --timeout-seconds 900

## 9. Restore current active bytes, assemble, and validate the bundle

After the authorized temporary candidate route has collected all evidence, it
must restore the original current AQ4 active manifest before this section. The
restoration command belongs to the separately approved route in section 7; do
not invent a v2 bootstrap rollback here.

    test "$(sha256sum "$ACTIVE" | awk '{print $1}')" = "$ACTIVE_SHA256"

    RELEASE_EVIDENCE=$OUT/release-evidence-f1a3cf4c.json
    RELEASE_VALIDATOR=$OUT/release-evidence-validation-f1a3cf4c.json
    BROWSER_EVIDENCE=$OUT/browser-reasoning-f1a3cf4c.json
    BROWSER_VALIDATOR=$OUT/browser-reasoning-validation-f1a3cf4c.json
    PROMOTION_EVIDENCE_BUNDLE=$OUT/$(basename "$PROMOTION_EVIDENCE")
    PROMOTION_RECEIPT_BUNDLE=$OUT/$(basename "$PROMOTION_RECEIPT")

    test ! -e "$RELEASE_EVIDENCE"
    test ! -e "$RELEASE_VALIDATOR"
    test ! -e "$PROMOTION_EVIDENCE_BUNDLE"
    test ! -e "$PROMOTION_RECEIPT_BUNDLE"
    test ! -e "$BUNDLE"

    cd "$SOURCE_TREE"
    uv run --project services/openai-gateway python \
      tools/prepare-generic-reasoning-release-evidence.py \
      --cases "$OUT/http-sse-campaign/cases.json" \
      --lifecycle "$OUT/http-sse-campaign/lifecycle.json" \
      --manifest "$CANDIDATE_ETC" --worker-binary "$WORKER" \
      --openwebui-image "$OPENWEBUI_IMAGE" \
      --active-promotion-source-commit "$SOURCE_COMMIT" \
      --output "$RELEASE_EVIDENCE" --status complete

    uv run --project services/openai-gateway python \
      tools/validate-generic-reasoning-release.py \
      "$RELEASE_EVIDENCE" --require-complete > "$RELEASE_VALIDATOR"

    install -m 0644 "$PROMOTION_EVIDENCE" "$PROMOTION_EVIDENCE_BUNDLE"
    install -m 0644 "$PROMOTION_RECEIPT" "$PROMOTION_RECEIPT_BUNDLE"

    uv run --project services/openai-gateway python \
      tools/prepare-generic-reasoning-release-bundle.py \
      --release-evidence "$RELEASE_EVIDENCE" \
      --release-validator "$RELEASE_VALIDATOR" \
      --browser-evidence "$BROWSER_EVIDENCE" \
      --browser-validator "$BROWSER_VALIDATOR" \
      --promotion-evidence "$PROMOTION_EVIDENCE_BUNDLE" \
      --promotion-receipt "$PROMOTION_RECEIPT_BUNDLE" \
      --rollback-manifest "$ACTIVE" --systemd-unit "$UNIT" \
      --environment-file "$ENVIRONMENT" --output "$BUNDLE" --status complete

    uv run --project services/openai-gateway python \
      tools/validate-generic-reasoning-release-bundle.py \
      "$BUNDLE" --require-complete

The promotion receipt refers to its evidence by a relative basename. Copying
both files into OUT with those basenames preserves that receipt binding. The
bundle snapshots the exact original active manifest, systemd unit, and
environment hashes. If any of those three inputs changes after this step,
discard the bundle and rebuild it from fresh evidence.

## 10. Final parent-only AQ4-to-AQ4 activation command

Run this only after section 9 has succeeded, the original active SHA-256 still
matches ACTIVE_SHA256, and all section 7 decisions have been recorded.

The rollback binding is --rollback-manifest "$ACTIVE" in the bundle. The
OpenWebUI configure commands intentionally omit
--previous-served-model-manifest: configure.py rejects an explicit previous
model whose public ID equals the active model ID, which is exactly this
AQ4-to-AQ4 transition. Passing the stale AQ4 named candidate would both be
incorrect and fail the same-model guard.

    sudo /usr/bin/python3 "$SOURCE_TREE/tools/activate-served-model.py" \
      --candidate "$CANDIDATE_ETC" \
      --active-manifest "$ACTIVE" \
      --release-bundle "$BUNDLE" \
      --systemd-unit "$UNIT" \
      --environment-file "$ENVIRONMENT" \
      --command-timeout-seconds 650 \
      --check-command-json '["/usr/bin/systemctl","restart","ullm-openai.service"]' \
      --check-command-json '["/usr/bin/docker","run","--rm","--network","open-webui-network","--entrypoint","curl","ghcr.io/open-webui/open-webui@sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff","--fail","--silent","--show-error","--retry","120","--retry-delay","2","--retry-max-time","600","--retry-all-errors","http://172.20.0.1:8000/readyz"]' \
      --reconcile-command-json '["/usr/bin/docker","stop","open-webui"]' \
      --reconcile-command-json '["/usr/bin/docker","run","--rm","-v","open-webui:/data","-v","/etc/ullm/openai-api-key:/run/secrets/ullm-api-key:ro","-v","/etc/ullm/served-models:/etc/ullm/served-models:ro","-v","/home/homelab1/coding-local/ultimateLLM/uLLM-aq4-fidelity-promotion-source-f1a3cf4c/deploy/openwebui/configure.py:/configure.py:ro","--entrypoint","python","ullm/open-webui:0.9.4-ullm.1","/configure.py","--served-model-manifest","/etc/ullm/served-models/active.json","--base-url","http://172.20.0.1:8000/v1"]' \
      --reconcile-command-json '["/usr/bin/docker","compose","-f","/home/homelab1/coding-local/ultimateLLM/uLLM-aq4-fidelity-promotion-source-f1a3cf4c/deploy/openwebui/compose.yaml","up","-d","--no-build"]' \
      --final-check-command-json '["/usr/bin/curl","--fail","--silent","--show-error","--retry","30","--retry-delay","2","--retry-connrefused","http://127.0.0.1:3000/health"]' \
      --rollback-command-json '["/usr/bin/systemctl","restart","ullm-openai.service"]' \
      --rollback-command-json '["/usr/bin/docker","run","--rm","--network","open-webui-network","--entrypoint","curl","ghcr.io/open-webui/open-webui@sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff","--fail","--silent","--show-error","--retry","120","--retry-delay","2","--retry-max-time","600","--retry-all-errors","http://172.20.0.1:8000/readyz"]' \
      --rollback-command-json '["/usr/bin/docker","stop","open-webui"]' \
      --rollback-command-json '["/usr/bin/docker","run","--rm","-v","open-webui:/data","-v","/etc/ullm/openai-api-key:/run/secrets/ullm-api-key:ro","-v","/etc/ullm/served-models:/etc/ullm/served-models:ro","-v","/home/homelab1/coding-local/ultimateLLM/uLLM-aq4-fidelity-promotion-source-f1a3cf4c/deploy/openwebui/configure.py:/configure.py:ro","--entrypoint","python","ullm/open-webui:0.9.4-ullm.1","/configure.py","--served-model-manifest","/etc/ullm/served-models/active.json","--base-url","http://172.20.0.1:8000/v1"]' \
      --rollback-command-json '["/usr/bin/docker","compose","-f","/home/homelab1/coding-local/ultimateLLM/uLLM-aq4-fidelity-promotion-source-f1a3cf4c/deploy/openwebui/compose.yaml","up","-d","--no-build"]'

The activation tool atomically restores the original active-manifest bytes
before it runs the rollback hooks. Do not call activate-served-model.py for a
test or a bootstrap attempt outside the separately approved route.
