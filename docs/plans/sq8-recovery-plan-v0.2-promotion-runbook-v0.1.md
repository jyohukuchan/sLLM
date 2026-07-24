# SQ8_0 Qwen3-14B-FP8 production promotion runbook v0.1

Status (2026-07-24): **v2 implementation is prepared; cutover is not
authorized yet.** The
immutable SQ8_0 product artifact and the complete 2026-07-12 historical
campaign have both been revalidated, but that campaign is bound to an
unavailable historical worker binary, `ullm.worker.v1`, and a legacy worker
invocation. It is not evidence for the current manifest-capable SQ8 worker.
The previously documented v1 activation alternative is closed by the user's
v2 decision. Sections 12 through 14 retain the investigation/design history;
section 15 records the implemented state. This runbook still stops before an
`active.json` change because the final clean worker build and fresh,
identity-bound candidate-active campaigns have not completed.

This concerns the independent `SQ8_0` FP8 E4M3 format for
`Qwen/Qwen3-14B-FP8`. It is not AQ4_0 with FP8 applied to selected tensors.
Keep this promotion distinct from AQ4_0におけるFP8の部分適用.

Every command that starts, stops, restarts, or observes the production
service, R9700, or OpenWebUI is marked **parent-only**. It was deliberately
not run while preparing this handoff. Nothing in this document targets V620.

## 1. Scope and fixed identities

| Item | Fixed value or current observation |
|---|---|
| Target public ID | `ullm-qwen3-14b-sq8` |
| Target format | `SQ8_0`, implementation `qwen3_sq8_rdna4_v1` |
| Upstream model | `Qwen/Qwen3-14B-FP8` at `9a283b4a5efbc09ce247e0ae5b02b744739e525a` |
| Product root | `/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1` |
| Product receipt SHA-256 | `f2d7b25d40adb29dae225626a87129dc5495c1251ee45563bcf4e6660cca8016` |
| Artifact manifest SHA-256 | `23977f4e9bed4bac4cc64c177c35d7f83355861426bf32027a69cf7a241552e2` |
| Artifact content SHA-256 | `2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147` |
| Artifact selection | 280 tensor pairs; 13,213,670,400 payload bytes; 561 files |
| Package manifest SHA-256 | `c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb` |
| Package payload | 163 files; 3,112,499,200 bytes |
| Historical full-campaign source commit | `f647a8aa8a78ddf0846b92fa700b8fa5a0995887` |
| Historical worker SHA-256 | `145a5351db3957130200276314853e394d0fd206a69e2eab260c01141411b950` |
| Current SQ8 worker SHA-256 | `3ecc7915265e574e146f8c3d8d5de405459bdcb12c035f50e56e948da37d0454` |
| Current SQ8 worker size | 7,707,400 bytes |
| Current active manifest | `/etc/ullm/served-models/active.json` |
| Current active manifest SHA-256 | `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a` |
| Current active format / protocol | `AQ4_0` / `ullm.worker.v2` |
| Current active promotion source | `0cd760568e197e1adb4c4df3d6149591a912f709` |

The exact current active bytes are the rollback target. Do not use a stale
named candidate as a substitute for those bytes.

## 2. Immutable artifact and 2026-07-12 evidence

The product artifact itself is complete and read-only. Re-run the full hash
check before any new candidate is made:

```bash
set -euo pipefail

REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
PRODUCT=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1

cd "$REPO"
python3 tools/validate-sq8-product-promotion.py "$PRODUCT"
```

It must report `verified: true`, `full_payloads: true`, and the hashes in
section 1. A metadata-only pass is not a substitute for this full payload
hashing pass.

The complete historical campaign is retained here:

```text
/home/homelab1/datapool/ullm/sq8-openwebui-product-20260712-v0.1
```

Its `SHA256SUMS` has SHA-256
`7ae6a710fb3b83dcfe1c7a71f1699a1f01927b330dd4b7d01e9ea7409be60a45`.
The repository mirror at
`benchmarks/results/2026-07-12/sq8-openwebui-product-20260712-v0.1` is a
thin mirror: it omits `raw-session-results.jsonl` and
`service-journal.raw.jsonl`. It must not be used as a standalone release
input, even though its remaining files match.

Validate the complete archive and reproduce the independent decision in an
isolated copy. `release-validation.json` is a validator output rather than an
input, so retain the old one outside the copied bundle before recomputing it.

```bash
set -euo pipefail

REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
HISTORICAL_COMMIT=f647a8aa8a78ddf0846b92fa700b8fa5a0995887
HISTORICAL_WORKER_SHA=145a5351db3957130200276314853e394d0fd206a69e2eab260c01141411b950
HISTORICAL_BUNDLE=/home/homelab1/datapool/ullm/sq8-openwebui-product-20260712-v0.1

( cd "$HISTORICAL_BUNDLE" && sha256sum -c SHA256SUMS )

SOURCE_TREE=$(mktemp -d /tmp/ullm-sq8-historical-source-XXXXXX)
rmdir "$SOURCE_TREE"
git -C "$REPO" worktree add --detach "$SOURCE_TREE" "$HISTORICAL_COMMIT"

VALIDATION_ROOT=$(mktemp -d /tmp/ullm-sq8-historical-validation-XXXXXX)
cp -a --reflink=auto "$HISTORICAL_BUNDLE"/. "$VALIDATION_ROOT/bundle"
mv "$VALIDATION_ROOT/bundle/release-validation.json" \
  "$VALIDATION_ROOT/release-validation.preexisting.json"

python3 "$REPO/tools/validate-sq8-openwebui-release.py" \
  --expected-commit "$HISTORICAL_COMMIT" \
  --expected-worker-binary-sha256 "$HISTORICAL_WORKER_SHA" \
  --repo-root "$SOURCE_TREE" \
  "$VALIDATION_ROOT/bundle"

test "$(sha256sum "$VALIDATION_ROOT/bundle/release-validation.json" | awk '{print $1}')" = \
  4261908bb53881a7737f54251235284d7c4069659d65851a0c023447d0cf8e08
```

The expected independent result is `release_status: complete`, including 21
successful browser requests, five cancellation phases, 100 normal plus 20
restart resource requests, 72 latency requests, 610 resource samples, and
the planned service restart count transition 7 to 8.

### 2.1 Why the historical complete result cannot activate today

The 2026-07-12 evidence is trustworthy for its own immutable identities. It
does not bind the current candidate because all of the following differ:

- The recorded worker is `145a...b950`; that binary is no longer available in
  the staged-worker locations. The current worker is `3ecc...d045`.
- At `f647a8...`, `ullm-sq8-worker` accepted only legacy
  `--artifact PATH --package PATH` startup. The current worker additionally
  implements `--served-model-manifest PATH`, which the current service uses.
- The historical environment records the legacy worker invocation, while the
  current candidate profile uses the served-model-manifest invocation.

Rebuilding or locating `145a...b950` would preserve archival evidence only;
it would not make that old worker compatible with the present manifest-mode
service. Do not overwrite the current worker with it.

## 3. Current candidate dry-run and the fixed v2 decision

The existing profile is:

```text
deploy/served-models/qwen3-14b-sq8.profile.json
```

It currently emits `ullm.served_model.v1`, `ullm.worker.v1`, and records the
short product-plan commit `dfc63de` (full commit
`dfc63de724f945517d398791f23c273fdb1c484a`). A dry-run generated manifest
was valid but had the following distinct identity:

```text
manifest SHA-256: 0b5f7fe4fa201c799d2834ba7aee8efa5b76cc8d26e7e8b221e1dd72ed0a97b3
worker SHA-256:   3ecc7915265e574e146f8c3d8d5de405459bdcb12c035f50e56e948da37d0454
```

Reproduce that non-installing dry-run as follows. Its output must remain
outside `/etc/ullm/served-models`.

```bash
set -euo pipefail

REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
DRY_RUN_ROOT=$(mktemp -d /tmp/ullm-sq8-manifest-dryrun-XXXXXX)
CANDIDATE="$DRY_RUN_ROOT/qwen3-14b-sq8.json"

cd "$REPO"
python3 tools/generate-served-model.py \
  --profile deploy/served-models/qwen3-14b-sq8.profile.json \
  --output "$CANDIDATE"
python3 tools/validate-served-model.py --manifest "$CANDIDATE"
sha256sum "$CANDIDATE"
```

Do not install that current v1 candidate or treat the historical campaign as
its release bundle. The selected route is:

1. Generate `ullm.served_model.v2` with `ullm.worker.v2` and a validated Qwen3
   reasoning contract.
2. Bind a full 40-character source commit, the exact release worker, the
   immutable SQ8 product, and the candidate's semantic profile through an SQ8
   serving promotion receipt/evidence pair; bind the later final manifest
   bytes through the campaigns and bundle v2.
3. Collect a fresh SQ8 full campaign and the generic reasoning/browser
   campaigns while that exact candidate is active.
4. Assemble and independently validate a complete generic-reasoning release
   bundle using an SQ8-aware promotion validator.
5. Perform the final switch only through normal `--release-bundle`
   activation.

The profile file will continue to use the wrapper schema
`ullm.served_model.profile.v1`; AQ4 v2 profiles do too. In this repository,
“served-model v2” means that the generated manifest is
`ullm.served_model.v2`, the worker protocol is `ullm.worker.v2`, and the
required top-level `reasoning` object is present. It does **not** mean
inventing `ullm.served_model.profile.v2`.

The old v1 route is retained below only as historical context. It is not an
authorized fallback if a v2 gate fails.

## 4. Browser-gate session authentication

There are two different credentials, and they must remain separate:

| Purpose | File contract | Permitted use |
|---|---|---|
| Gateway OpenAI API authentication | `/etc/ullm/openai-api-key`, root:`homelab1`, mode `0640` | API contract, direct-cancel, latency, and `/v1` checks |
| OpenWebUI frontend login | parent-provisioned private file, execution user, mode `0600` | Browser soak, Stop, failure, and browser smoke only |

The second file must contain a real OpenWebUI frontend session JWT. It is not
the OpenWebUI signing key and it is never `/etc/ullm/openai-api-key`. The
updated runners reject a non-JWT value, require an `alg` header and integer
`exp`, and reject a token too close to expiry for each gate. They do not print
the token. OpenWebUI itself remains the authority that validates its signature
and session claims.

The exposed interfaces are now:

```text
tools/run-openwebui-soak-gate.py --openwebui-session-token-file PATH
tools/run-openwebui-stop-gate.py --openwebui-session-token-file PATH
tools/run-openwebui-failure-gate.py --openwebui-session-token-file PATH
tools/run-openwebui-reasoning-browser-smoke.py --openwebui-session-token-file PATH
tools/run-sq8-full-openwebui-campaign.py --openwebui-session-token-file PATH
```

The browser containers receive that file only as
`/run/secrets/openwebui-session-token` through
`OPENWEBUI_SESSION_TOKEN_FILE`. The gateway API key is not mounted there.

Before a real campaign, the parent may check only metadata, without printing
the JWT:

```bash
test "$OPENWEBUI_SESSION_TOKEN_FILE" != /etc/ullm/openai-api-key
test -f "$OPENWEBUI_SESSION_TOKEN_FILE"
test "$(stat -c '%a' "$OPENWEBUI_SESSION_TOKEN_FILE")" = 600
test "$(stat -c '%U' "$OPENWEBUI_SESSION_TOKEN_FILE")" = "$(id -un)"
```

## 5. Required SQ8 campaign gates

The current SQ8 campaign implementation is itself v1-bound:
`tools/sq8_full_campaign_identity.py` fixes the protocol to
`ullm.worker.v1`, validates `ullm.sq8_product_promotion.v1`, and does not bind
the bytes of an active served-model v2 manifest. The command later in this
section is therefore a target invocation after the section 12 changes, not a
command that is executable against the present source.

Run the full orchestrator rather than composing a partial set of browser
gates. Its fixed order binds the same candidate identity through all of these
components:

| Stage | Gate and independent evidence |
|---|---|
| API contract | `run-sq8-api-contract-gate.py` and its ingestor |
| Browser smoke and soak | `run-openwebui-soak-gate.py --include-smoke` and `sq8_openwebui_gate_ingest.py` |
| Direct cancellation | `run-sq8-direct-cancel-gate.py` and its ingestor |
| Browser Stop/recovery | `run-openwebui-stop-gate.py` and `sq8_openwebui_stop_gate_ingest.py` |
| Post-header failure/recovery | `run-openwebui-failure-gate.py` and its ingestor |
| Latency | `run-sq8-http-latency-gate.py` and its ingestor |
| Resource stability | normal and restart collectors, then independent campaign validation |

`sq8_openwebui_stop_gate_ingest.py` does not authenticate a browser. It
independently reconstructs the Stop evidence bundle after the gate has run;
therefore it cannot compensate for a wrong session token.

After the v2 implementation has passed CPU-only tests and the parent has
prepared a clean detached source tree, the section 12.7 locked
temporary-window wrapper installs the exact v2 candidate and invokes the
campaign under its consumed authorization claim. The target child invocation
is:

```bash
set -euo pipefail

SOURCE_TREE=/absolute/clean/sq8-release-source
SOURCE_COMMIT=<approved-40-character-commit-including-the-session-auth-fix>
WORKER=/absolute/staged/ullm-sq8-worker
WORKER_SHA=<SHA-256-of-WORKER>
CANDIDATE_ETC=/etc/ullm/served-models/candidates/qwen3-14b-sq8-<release>.json
CANDIDATE_SHA=<SHA-256-of-CANDIDATE_ETC>
ACTIVE=/etc/ullm/served-models/active.json
AUTHORIZATION_CLAIM=/absolute/fixed-registry/authorization-claim.json
RUN_ID=sq8-openwebui-production-<date>-v0.1
OUT=/absolute/new-empty-release-directory
GATEWAY_API_KEY_FILE=/etc/ullm/openai-api-key
OPENWEBUI_SESSION_TOKEN_FILE=/absolute/private/openwebui-session.jwt

test "$(git -C "$SOURCE_TREE" rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git -C "$SOURCE_TREE" status --porcelain --untracked-files=all)"
test "$(sha256sum "$WORKER" | awk '{print $1}')" = "$WORKER_SHA"
test "$(sha256sum "$CANDIDATE_ETC" | awk '{print $1}')" = "$CANDIDATE_SHA"
test ! -e "$OUT"

cd "$SOURCE_TREE"
python3 tools/run-sq8-full-openwebui-campaign.py \
  --execute \
  --expected-commit "$SOURCE_COMMIT" \
  --expected-worker-binary-sha256 "$WORKER_SHA" \
  --candidate-served-model-manifest "$CANDIDATE_ETC" \
  --active-served-model-manifest "$ACTIVE" \
  --expected-served-model-manifest-sha256 "$CANDIDATE_SHA" \
  --campaign-authorization-claim "$AUTHORIZATION_CLAIM" \
  --run-id "$RUN_ID" \
  --final-path "$OUT" \
  --api-key-file "$GATEWAY_API_KEY_FILE" \
  --openwebui-session-token-file "$OPENWEBUI_SESSION_TOKEN_FILE"

python3 tools/validate-sq8-openwebui-release.py \
  --expected-commit "$SOURCE_COMMIT" \
  --expected-worker-binary-sha256 "$WORKER_SHA" \
  --expected-served-model-manifest-sha256 "$CANDIDATE_SHA" \
  --campaign-authorization-claim "$AUTHORIZATION_CLAIM" \
  --repo-root "$SOURCE_TREE" \
  "$OUT"
```

The candidate/active manifest and authorization-claim arguments above are
required implementation targets from section 12; they are not accepted by
current HEAD yet. The campaign must not be launched outside the encompassing
temporary-window transaction.

This is a real service/GPU/OpenWebUI operation: it intentionally exercises
Stop and post-header failure/recovery. It is parent-only and was not run for
this handoff. It must finish with `release_status: complete`; any incomplete,
identity-mismatched, or expired-session result invalidates the candidate.

## 6. Parent-only `active.json` activation and rollback

This section is executable only after all v2 admission items in section 14
and a fresh, identity-matching complete campaign. It documents the actual
switch mechanics so that the operator does not hand-edit `active.json`.

First freeze the current rollback bytes and verify the candidate outside the
active path:

```bash
set -euo pipefail

ACTIVE=/etc/ullm/served-models/active.json
ACTIVE_SHA256=5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a
CANDIDATE_ETC=/etc/ullm/served-models/candidates/qwen3-14b-sq8-<release>.json
CANDIDATE_SHA=<SHA-256-of-CANDIDATE_ETC>
BACKUP=/etc/ullm/served-models/active.json.before-sq8-<release>
UNIT=/etc/systemd/system/ullm-openai.service
ENVIRONMENT_FILE=/etc/ullm/openai-gateway-manifest.env
ROLLBACK_OUTCOME=/absolute/new-rollback-outcome.json

test "$(sha256sum "$ACTIVE" | awk '{print $1}')" = "$ACTIVE_SHA256"
test "$(sha256sum "$CANDIDATE_ETC" | awk '{print $1}')" = "$CANDIDATE_SHA"
sudo /usr/bin/python3 "$SOURCE_TREE/tools/validate-served-model.py" \
  --manifest "$CANDIDATE_ETC"
test ! -e "$BACKUP"
sudo /usr/bin/python3 "$SOURCE_TREE/tools/freeze-served-model-manifest.py" \
  --source "$ACTIVE" \
  --expected-sha256 "$ACTIVE_SHA256" \
  --output "$BACKUP"
test "$(sha256sum "$BACKUP" | awk '{print $1}')" = "$ACTIVE_SHA256"
```

`freeze-served-model-manifest.py` is a required no-clobber implementation
target from section 12.8, not a current-HEAD interface. It must stable-read,
atomically publish mode-0444/nlink-1 bytes, and refuse an existing output.

The following v1 command is archived context from the earlier investigation.
It demonstrates the activation hooks and atomic rollback shape, but it is
**not authorized for SQ8 promotion** and must not be run. The service restart
and OpenWebUI reconciliation shown here are intentionally parent-only.
Replace image identities only after a separate identity check.

```bash
OPENWEBUI_IMAGE=ullm/open-webui:0.9.4-ullm.1
OPENWEBUI_BASE_IMAGE=ghcr.io/open-webui/open-webui@sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff

sudo /usr/bin/python3 "$SOURCE_TREE/tools/activate-served-model.py" \
  --candidate "$CANDIDATE_ETC" \
  --active-manifest "$ACTIVE" \
  --command-timeout-seconds 650 \
  --check-command-json '["/usr/bin/systemctl","restart","ullm-openai.service"]' \
  --check-command-json "[\"/usr/bin/docker\",\"run\",\"--rm\",\"--network\",\"open-webui-network\",\"--entrypoint\",\"curl\",\"$OPENWEBUI_BASE_IMAGE\",\"--fail\",\"--silent\",\"--show-error\",\"--retry\",\"120\",\"--retry-delay\",\"2\",\"--retry-max-time\",\"600\",\"--retry-all-errors\",\"http://172.20.0.1:8000/readyz\"]" \
  --reconcile-command-json '["/usr/bin/docker","stop","open-webui"]' \
  --reconcile-command-json "[\"/usr/bin/docker\",\"run\",\"--rm\",\"-v\",\"open-webui:/data\",\"-v\",\"/etc/ullm/openai-api-key:/run/secrets/ullm-api-key:ro\",\"-v\",\"/etc/ullm/served-models:/etc/ullm/served-models:ro\",\"-v\",\"$SOURCE_TREE/deploy/openwebui/configure.py:/configure.py:ro\",\"--entrypoint\",\"python\",\"$OPENWEBUI_IMAGE\",\"/configure.py\",\"--served-model-manifest\",\"/etc/ullm/served-models/active.json\",\"--base-url\",\"http://172.20.0.1:8000/v1\"]" \
  --reconcile-command-json "[\"/usr/bin/docker\",\"compose\",\"-f\",\"$SOURCE_TREE/deploy/openwebui/compose.yaml\",\"up\",\"-d\",\"--no-build\"]" \
  --final-check-command-json '["/usr/bin/curl","--fail","--silent","--show-error","--retry","30","--retry-delay","2","--retry-connrefused","http://127.0.0.1:3000/health"]' \
  --rollback-command-json '["/usr/bin/systemctl","restart","ullm-openai.service"]'
```

For the selected v2 SQ8 route, the final invocation must add the independently
validated `--release-bundle "$BUNDLE"`, `--systemd-unit`, and
`--environment-file` arguments required by `activate-served-model.py`. Do not
use the archived v1 command or `--bootstrap-v2` for final promotion.

If any activation check or reconciliation command fails, the activation tool
restores the exact previous active bytes before running rollback hooks. A
later operator rollback after completed activation must use the new locked
rollback subcommand from section 13, not `install`, `cp`, or an editor. Its
target interface is:

```bash
sudo /usr/bin/python3 "$SOURCE_TREE/tools/rollback-served-model.py" \
  --active-manifest "$ACTIVE" \
  --expected-current-sha256 "$CANDIDATE_SHA" \
  --rollback-manifest "$BACKUP" \
  --expected-rollback-sha256 "$ACTIVE_SHA256" \
  --systemd-unit "$UNIT" \
  --environment-file "$ENVIRONMENT_FILE" \
  --outcome "$ROLLBACK_OUTCOME"
# Supply the reviewed reverse-reconciliation and final-check command options.
```

That subcommand must take the activation lock, validate expected-current and
backup bytes, atomically replace plus fsync, reconcile OpenWebUI to AQ4,
verify gateway/UI/model identity, and publish an immutable outcome. Unit and
environment hashes are preconditions; current activation does not modify or
restore those files.

## 7. Pre- and post-activation health checks

Before the switch, record these without modifying the active manifest:

```bash
sudo /usr/bin/python3 "$SOURCE_TREE/tools/validate-served-model.py" --manifest "$ACTIVE"
sha256sum "$ACTIVE"
docker run --rm --network open-webui-network --entrypoint curl \
  "$OPENWEBUI_BASE_IMAGE" --fail --silent --show-error \
  http://172.20.0.1:8000/readyz
curl --fail --silent --show-error http://127.0.0.1:3000/health
```

After a successful activation, all of the following must hold before the
window is closed:

1. `cmp -s "$ACTIVE" "$CANDIDATE_ETC"` succeeds and
   `sha256sum "$ACTIVE"` equals `$CANDIDATE_SHA`; then
   `validate-served-model.py --manifest "$ACTIVE"` reports `format_id: SQ8_0`,
   public ID `ullm-qwen3-14b-sq8`, and the exact fresh candidate worker
   SHA-256. The summary alone is insufficient because it omits the complete
   promotion and reasoning identity.
2. Gateway `/readyz` and OpenWebUI `/health` both return success after the
   reconciliation commands complete.
3. An authenticated `/v1/models` check made with the gateway API credential
   lists `ullm-qwen3-14b-sq8`; use an approved secret-safe client, not an
   inline shell substitution that exposes the credential.
4. A real OpenWebUI browser check selects the SQ8_0 model and produces the
   expected model identity. It must use the separate session JWT, not the
   gateway API key.
5. The full campaign output and independent validator still bind the active
   manifest, worker, product artifact, source commit, and OpenWebUI image.

Any failure in these checks requires the rollback procedure above. Do not
leave a mixed active-manifest/OpenWebUI configuration state.

## 8. Dry-run record for this handoff

Completed without changing `/etc/ullm/served-models/active.json`, restarting
`ullm-openai.service`, invoking systemd, or using a GPU:

- The active AQ4_0 v2 manifest was read and validated; its hash is recorded
  in section 1.
- `validate-sq8-product-promotion.py` completed full artifact and package
  payload hashing successfully.
- The datapool full 2026-07-12 archive passed all 19 `SHA256SUMS` entries.
  A clean detached `f647a8...` source checkout reproduced the independent
  `complete` decision and the exact validator SHA-256
  `4261908bb53881a7737f54251235284d7c4069659d65851a0c023447d0cf8e08`.
- The historical repository mirror was correctly rejected as incomplete due
  to its two omitted raw files.
- A temporary current v1 candidate manifest was generated and validated; it
  proves current product/profile mechanics but also proves the worker and
  manifest identity mismatch described in section 2.1.
- On a private temporary copy only, `activate-served-model.py` successfully
  performed the v2-to-v1 SQ8_0 manifest replacement and validated the result.
  A separate forced failing check (`/usr/bin/false`) returned nonzero and
  restored the copied AQ4 active bytes exactly. In both cases the real active
  manifest remained SHA-256
  `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a`.
- Browser-gate unit tests cover rejection of a gateway API-key-shaped value
  and acceptance of a structurally valid, unexpired session JWT. No real
  session credential was read or created.
- The historical formal Stop/failure pilots remain pinned to their old gate
  and browser source SHA-256 values. The source-binding tests now explicitly
  require a fresh pilot instead of treating those immutable old pilots as
  validation of this new gate source.

The remaining actions are the v2 implementation and review work below,
followed by parent-controlled candidate-active service execution. The safe
dry-run result is therefore **not ready for production cutover**.

## 9. What AQ4_0 production actually uses as “v2”

This section records the current code and live-data path so that “follow AQ4
v2” has an exact meaning.

### 9.1 Live manifest and runtime enforcement

`/etc/ullm/served-models/active.json` is SHA-256
`5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a`.
It is byte-identical to the AQ4 fidelity candidate under
`benchmarks/results/2026-07-17/qwen35-9b-aq4-fidelity-promotion-f1a3cf4c-v0.1`.
Its relevant contract is:

| Field | Live AQ4 value |
|---|---|
| `schema_version` | `ullm.served_model.v2` |
| `public.id` | `ullm-qwen3.5-9b-aq4` |
| `format.format_id` | `AQ4_0` |
| `worker.protocol` | `ullm.worker.v2` |
| `worker.binary_sha256` | `1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b` |
| `promotion.source_commit` | `0cd760568e197e1adb4c4df3d6149591a912f709` |
| promotion receipt SHA-256 | `1b36fc880bf1510185eaad7887c9aed33f69df223036271e4bfba4bb43f16e8b` |
| reasoning dialect | `qwen3.5-thinking-v1` |

The systemd preflight calls `tools/validate-served-model.py`. The Python
gateway loader then strictly parses the same manifest and launches the exact
worker path and environment. The Rust served-model loader inside
`ullm-aq4-worker` parses it again and verifies the running executable. Both
runtime loaders live-hash the external promotion receipt. They do not parse
the AQ4 evidence or prove that the worker was built from
`promotion.source_commit`; those semantic checks occur in
`tools/generate-served-model.py`, and the deployed executable is ultimately
bound by its byte hash.

The live source value therefore identifies only the promotion-tool checkout
HEAD used by the evidence runner; that runner did not attest a clean worktree,
tree identity, or worker build provenance. The detached worker path and its
SHA-256 are the actual deployed-worker identity. SQ8 v2 must improve this
boundary by making the SQ8 promotion evidence reference a worker build
receipt that binds the clean source commit/tree to the resulting worker
SHA-256.

The live AQ4 manifest is also a frozen 30-environment-guard contract. Current
HEAD's AQ4 profile and Rust source contain 36 guards after later commits.
Regenerating from the current profile cannot reproduce the live manifest and
worker identity unchanged; SQ8 must always compare against the frozen live
bytes rather than treating current AQ4 source as their reconstruction.

### 9.2 Normal formal v2 authorization

The intended v2 promotion path is:

```text
profile materialization and semantic evidence checks
  -> pre-receipt promotion evidence
  -> immutable promotion receipt
  -> generated, strict, validated served_model.v2 candidate
  -> candidate-active release + browser evidence
  -> independently recomputed validator reports
  -> complete generic-reasoning release bundle
  -> activate-served-model.py --release-bundle
```

`activate-served-model.py` validates that the complete bundle binds:

- the candidate manifest SHA-256 and candidate worker SHA-256;
- the full promotion source commit;
- the exact old active-manifest SHA-256;
- the systemd unit and environment-file SHA-256 values;
- generic reasoning release evidence and its recomputed validator report;
- browser reasoning evidence and its recomputed validator report; and
- promotion evidence and its immutable receipt through the bundle validator.

The generic bundle envelope is `ullm.generic_reasoning_release_bundle.v1`.
Its exact root fields are `schema_version`, `status`,
`production_activation_performed`, `source_commit`,
`active_promotion_source_commit`, `identity`, `artifacts`, and
`rollback_target`. `identity` binds `manifest_sha256`,
`worker_binary_sha256`, `tokenizer_sha256`, and a content-addressed
`openwebui_image`. `artifacts` has the six fixed references
`release_evidence`, `release_validator`, `browser_evidence`,
`browser_validator`, `promotion_evidence`, and `promotion_receipt`.
`rollback_target` binds `manifest_sha256`, `systemd_unit_sha256`, and
`environment_sha256`.

The envelope and activation transaction are format-independent. The current
promotion sub-validator is not: `_validate_promotion()` in
`tools/validate-generic-reasoning-release-bundle.py` accepts only
`ullm.aq4_resident_promotion_evidence.v1` and
`ullm.aq4_resident_promotion.v1`. SQ8 can reuse its identity, rollback, and
independent-validation design, but section 12.6 requires a versioned envelope
extension as well as an SQ8 promotion branch because the six v1 artifact
slots cannot carry the separate SQ8 full campaign.

Two current-main gaps must not be mistaken for existing authorization:

- the activator matches the candidate manifest, worker, and promotion source
  to bundle identity, while the bundle validator separately validates its
  promotion evidence/receipt by shared source and worker values. Neither
  compares the candidate's `promotion.receipt` path/`receipt_sha256` directly
  with the bundle's `promotion_receipt` component; and
- the published browser evidence/validator does not retain manifest or worker
  hashes. Generic release evidence and bundle identity bind those values, but
  current browser evidence does not independently cross-bind them.

The SQ8 bundle revision in section 12.6 must close both gaps rather than
claiming that current AQ4 already does.

### 9.3 How the current AQ4 bytes reached production

The current AQ4 promotion is explicitly “functionally live, formally
incomplete” in
`docs/plans/aq4-fidelity-root-cause-and-fix-plan-v0.1-promotion-runbook-v0.1.md`.
It did not finish the complete-bundle route above. It used the temporary
v2-to-v2 differing-worker bootstrap introduced by commit
`9c9a6f2972d09ec74a85c58498840f5b7fcee304`:

```text
--bootstrap-v2
--authorize-differing-worker-v2-bootstrap
--authorization-note ...
```

That path requires both active and candidate manifests to be v2, requires the
same `public.id`, and normally requires the same worker SHA-256. The explicit
differing-worker flag relaxes only the worker equality check. It writes a
root-only audit sidecar with schema
`ullm.served_model.v2_differing_worker_bootstrap_authorization.v1` and these
semantic fields:

- authorization note and purpose
  `temporary_candidate_active_evidence_collection_only`;
- required follow-up: restore the original active manifest, then use
  bundle-gated activation;
- old and candidate manifest SHA-256 values; and
- old and candidate worker SHA-256 values.

The sidecar is not consumed by the gateway or worker. No reader treats it as
runtime authorization, and no copy exists adjacent to the live
`active.json`. Three observed records (`v1`, `v3`, and `v4`) remain beside
benchmark-result backups as root-only audit files. They are audit records for
privileged activation attempts, not success receipts.
Its sidecar is published no-replace, but the backup uses an absence check
followed by `os.replace`; together they provide only per-path replay
resistance, not a global one-shot guarantee. A new pathname permits another
attempt. The sidecar survives failures after the transaction enters the
switch/rollback stage, while pre-switch cleanup removes it. It therefore does
not, by itself, prove that post-switch hooks succeeded.

This escape hatch cannot authorize AQ4-to-SQ8 candidate-active evidence:
`activate-served-model.py` rejects the different model ID before reaching the
differing-worker exception. The separately named, narrower cross-model
campaign authorization and encompassing transaction in section 12.7 are
required. They are never the final promotion path.

### 9.4 Rollback behavior

For activation of either served-model schema version, the activation tool
snapshots the current active bytes, atomically replaces the manifest, and runs
check, reconciliation, and final-check hooks. On an exception it restores the
exact snapshot before running rollback hooks. The stronger v2 property is
that a normal complete bundle additionally commits to the rollback manifest,
systemd unit, and environment hashes before activation. The current audit
sidecar is not a rollback authorization and is not a substitute for that
bundle binding. The unit and environment files are hashed preconditions; the
tool does not modify or restore their contents.

## 10. Historical “SQ8 authorization lineage v2” classification

The similarly named history is **not** the served-model v2 mechanism above.
It belongs to an old Qwen3.5 AQ4_0 implementation in which exactly 48 QKV/Z
tensors were replaced with SQ8_0 data. That feature was formerly called the
“SQ8 overlay”; it is unrelated to the independently served
`Qwen/Qwen3-14B-FP8` `SQ8_0` target in this runbook.

### 10.1 Repository and implementation evidence

The relevant commits include:

| Commit | Subject |
|---|---|
| `0cd6b9a0` | Bind formal SQ8 authorization lineage |
| `823ba441` | Normalize authorized SQ8 worker identity |
| `390409fc` | Validate SQ8 authorization in Rust served models |
| `62cbf66e` | Generalize SQ8 authorization lineage v2 |
| `b3106eb8` | Define SQ8 v1 to v2 lineage migration |
| `fce80f66` | Support SQ8 authorization lineage v2 in Rust loader |
| `6ad51ac5` | Support append-only SQ8 lineage successors |

They are on a divergent side history rooted at `48f55b7c`, represented by
refs such as `p2-sq8-overlay-main-integration`; they are not ancestors of
current HEAD. Current main does not contain their spec, helper, promotion
builder, or loader fields. The feature itself began at `a97050f4` with the
subject `feat(aq4): admit exact SQ8 QKV Z overlay`.

At `6ad51ac5`, the implementation is named
`qwen35_aq4_sq8_linear_qkv_z_overlay_v1`, the base format is `AQ4_0`, the
overlay format is `SQ8_0`, the worker is `/ullm-aq4-worker`, and the generator
requires an exact 48-tensor topology. The lineage helper hard-codes schemas
such as
`ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1`,
`ullm.qwen35_aq4_sq8_overlay_independent_audit.v1`, and
`ullm.qwen35_aq4_sq8_overlay_promotion.v1`, plus
`sq8-promotion-<64 hex>` request IDs. The Python and Rust validators hard-code
the same audit family. “Generalize” in `62cbf66e` introduced a generic-looking
fixed-field v2 envelope, typed relations, and an initial predecessor-prefix
model; later commits made the v1 migration and two-entry successors exact.
All receipt schemas and semantic validation remained overlay-specific. It did
not create a universal served-model authorization policy.

### 10.2 Overlay lineage v1 versus v2

| Property | Overlay lineage v1 | Overlay lineage v2 |
|---|---|---|
| Root | Exact `schema_version`, `disposition`, `source`, `entries` | Same plus a live-bound `predecessor` reference |
| Entries | Exactly six ordered, heterogeneous audit records | Homogeneous exact fields: `sequence`, `relation`, `path`, `sha256`, `schema_version`, `status`, `request_id`, `source_commit` |
| Seed meaning | One implementation-ready capture audit (`actual: not_executed`); two capture no-GOs; two consumed actual failures; one restore no-GO | First migration normalizes those six, appends a predecessor-source actual failure, then a new-source current GO |
| Successor | No append-only successor contract | Exact predecessor prefix plus exactly two entries: prior-source actual failure, then new-source current GO |
| Minimum topology | Fixed historical six | At least one current GO, two capture no-GOs, one restore no-GO, and three actual failures; terminal entry is the only current-source GO |
| Reference | Input path, runtime path, whole-document SHA-256, entries SHA-256 | Adds entry count and current-implementation audit path/SHA-256 |
| Reuse | Every entry explicitly has `reusable_as_runtime_authorization: false`; v2-capable tooling always treats v1 as diagnostic-only | V2 preserves predecessor entries as history and selects the final `implementation_ready_current` entry, bound to the current source, as the current implementation audit |
| File rules | Canonical absolute, regular non-symlink, read-only, nlink=1, bounded/stable read and live rehash | Same, plus predecessor and current audit live validation |

The overlay runner had a one-attempt-per-invocation/output policy, not a
durable one-shot authorization. Actual execution required explicit paired
flags, an allowed `max_attempts=1`, a per-invocation `actual_run_count`
transition from zero to one, and a fresh caller-selected output. The Gate or
candidate was never consumed; another fresh output could be used for another
invocation. Its rollback restored an overlay maintenance service/default
runtime and checked a new epoch and health. That is different from atomic
`active.json` rollback.

No overlay schema, fixed historical entry set, request namespace, AQ4 worker
identity, or 48-tensor rule may be copied into independent SQ8_0. Concepts
that remain useful are create-new immutable receipts, canonical file checks,
live hashes, typed relations, and append-only predecessor validation if a
future served-model-wide authorization log is deliberately designed. Such a
log is not required to make SQ8 comply with the current served-model v2
release scheme.

## 11. Current served-model v1 versus v2

This is the comparison that governs the present promotion.

| Property | `ullm.served_model.v1` | `ullm.served_model.v2` |
|---|---|---|
| Profile wrapper | `ullm.served_model.profile.v1` | Also `ullm.served_model.profile.v1` |
| Exact manifest root | `schema_version`, `public`, `generation`, `format`, `tokenizer`, `worker`, `product`, `promotion` | Same exact fields plus required `reasoning` |
| Worker protocol | Must be `ullm.worker.v1` | Must be `ullm.worker.v2` |
| Reasoning | Forbidden | Required exact contract: `enabled_by_default`, `dialect_id`, start/end/forced-end token IDs, `initial_phase`, `eos_policy`, effort budgets, max budget, reserved answer tokens, history policy |
| Promotion manifest object | Exact `source_commit`, `receipt`, `receipt_sha256` | Same exact three fields |
| Runtime validation | Python gateway and Rust worker strict parsing, receipt live hash, worker self-identity | Same plus reasoning/protocol alignment and reasoning contract validation |
| Normal activation gate | No release bundle is required by `activate-served-model.py` | A complete validated release bundle is mandatory unless the explicit temporary bootstrap path is used |
| Candidate-active bootstrap | Not the v2 bootstrap contract | `--bootstrap-v2`; current active-v2 case requires same model ID and, absent an extra explicit flag, same worker hash |
| One-shot behavior | No scheme-wide consumed marker | Current bootstrap has a no-replace sidecar plus a path-scoped backup absence check/replace; neither consumes an authorization, and fresh paths permit another run |
| Automatic rollback | Atomic restore is available when the activation tool is used | Same atomic restore, plus the formal bundle pre-binds old manifest, unit, and environment hashes |
| Final authorization evidence | None enforced by the activator; historical SQ8 campaign evidence is an out-of-band convention | Promotion evidence/receipt + release/browser evidence + recomputed reports + complete bundle |

`tools/generate-served-model.py` makes the generation choice mechanically:
a profile with `reasoning` must name `ullm.worker.v2`; worker v2 without
`reasoning` is rejected; a reasoning profile emits served-model v2, otherwise
served-model v1. `tools/validate-served-model.py`, the Python loader, and the
Rust loader already accept a format-independent SQ8_0 v2 manifest. No new
top-level `authorization_lineage` field should be added to that strict
manifest for this promotion.

The v2 manifest schema itself does not require a 40-character promotion
source. Current loaders accept bounded text there. Full-commit enforcement
must come from the new SQ8 evidence/generator checks and from bootstrap/bundle
preflight.

## 12. Target SQ8_0 v2 contract and code-change matrix

### 12.1 Served-model profile and manifest

Update `deploy/served-models/qwen3-14b-sq8.profile.json` as follows:

- keep `schema_version: ullm.served_model.profile.v1`;
- change `worker.protocol` to `ullm.worker.v2`;
- point `worker.binary` to the separately rebuilt, immutable staged worker
  and let the generator bind its exact SHA-256;
- add a Qwen3 reasoning object;
- point `promotion.receipt` to the new SQ8 serving-promotion receipt;
- set `source_commit_from_receipt` to `["source_commit"]`;
- set `required_schema_version` to
  `ullm.sq8_serving_promotion.v1`;
- add the existing evidence path/SHA selectors used by the AQ4 profile; and
- change artifact `content_sha256_from_receipt` to
  `["product", "artifact_content_sha256"]` in the new receipt.

Read-only tokenizer inspection established the Qwen3 vocabulary size
`151936`, thinking start token `151667`, thinking end token `151668`, and the
existing EOS IDs `151645` and `151643`. The initial contract to test is:

```json
{
  "enabled_by_default": false,
  "dialect_id": "qwen3-thinking-v1",
  "start_token_ids": [151667],
  "end_token_ids": [151668],
  "forced_end_token_ids": [151668],
  "initial_phase": "reasoning",
  "eos_policy": "close",
  "effort_budgets": {"low": 32, "medium": 128, "high": 256},
  "max_budget_tokens": 256,
  "reserved_answer_tokens": 1,
  "history_reasoning_policy": "omit"
}
```

These token IDs are fixed tokenizer facts. The dialect and budget policy were
subsequently ratified after CPU tests covered disabled, zero-budget, bounded,
forced-close, EOS-close, unbounded, history-omission, answer-reservation,
cancellation, and accounting behavior. This paragraph is the original design
input; the ratified v0.2 contracts are authoritative.

### 12.2 SQ8 serving-promotion receipt

Do not overwrite the immutable product `promotion.json`. Its
`ullm.sq8_product_promotion.v1` schema and short `plan_commit` prove the old
artifact-copy operation, not the source provenance or worker-build identity
of the new served worker.
Create a separate receipt in the same immutable product root with this exact
root shape:

```json
{
  "schema_version": "ullm.sq8_serving_promotion.v1",
  "source_commit": "<40 lowercase hex>",
  "evidence": {"path": "<safe relative path>", "sha256": "<64 lowercase hex>"},
  "product": {
    "receipt": {"path": "promotion.json", "sha256": "<64 lowercase hex>"},
    "artifact_manifest_sha256": "<64 lowercase hex>",
    "artifact_content_sha256": "<64 lowercase hex>",
    "package_manifest_sha256": "<64 lowercase hex>"
  }
}
```

The writer must validate all referenced live files and use a race-safe,
atomic no-replace publication primitive (`link`, `renameat2(RENAME_NOREPLACE)`,
or an equivalent `O_EXCL` design). It must refuse an existing destination,
publish a stable mode-0444/nlink-1 file, rehash it after publication, and
never edit it. The profile points to this receipt; the generated manifest
continues to expose only the existing strict three-field `promotion` object
and therefore remains compatible with both runtime loaders.

### 12.3 SQ8 serving-promotion evidence

Add an `ullm.sq8_serving_promotion_evidence.v1` validator, runner, and
create-new receipt writer, parallel in role but not schema-copied from AQ4.
The evidence must bind:

- `verified: true` and `production_receipt_written: false`;
- a full 40-character source commit, Git tree identity, clean detached
  worktree result, and source-file hashes used by the build/evidence tools;
- a separate worker build receipt, exact staged worker path/SHA-256,
  `ullm.worker.v2`, and nlink=1/read-only file checks;
- the profile path/SHA-256 and a pre-receipt ephemeral manifest that exactly
  matches its public, generation, format, tokenizer, worker, product, and
  reasoning semantics;
- the old product receipt path/SHA-256, artifact manifest SHA-256, artifact
  content SHA-256, package manifest SHA-256, and a full-payload
  `validate-sq8-product-promotion.py` result; and
- CPU-safe v2 protocol cases for disabled reasoning, budget zero, all named
  efforts, unbounded reasoning, forced close, EOS close, answer reservation,
  usage accounting, cancellation, and clean reset.

Do not claim a clean shutdown of the real staged SQ8 backend in this
pre-receipt CPU evidence: launching that binary loads the GPU backend. Real
startup/shutdown/reset behavior belongs to the controlled candidate-active
campaign and its model-campaign evidence.

As with AQ4, pre-receipt evidence cannot have the byte hash of the final
manifest because the final manifest hashes the receipt that does not yet
exist. It must bind the ephemeral semantic identity. The final candidate
manifest SHA-256 is bound later by the fresh release/browser evidence and the
complete bundle.

### 12.4 Generator and runtime loaders

Refactor `_validate_aq4_evidence()` in
`tools/generate-served-model.py` into a strict promotion-evidence dispatch
keyed by `promotion.required_schema_version` and target format:

- preserve the AQ4 validator byte-for-byte in an AQ4 branch;
- add an SQ8 branch for the two schemas above and the SQ8 artifact binding;
- reject every unknown schema/format pairing; and
- add positive and mutation tests for source, worker, product, profile,
  reasoning, evidence path, and every SHA-256 binding.

No structural change is expected in
`services/openai-gateway/src/ullm_openai_gateway/served_model.py`,
`crates/ullm-engine/src/served_model.rs`, or
`tools/validate-served-model.py`: their served-model v2 and SQ8 startup
handling is already format-independent. Add SQ8 v2 positive/negative fixtures
to prove that statement and keep their exact-field fail-closed behavior.

### 12.5 SQ8 worker reasoning implementation

The SQ8 protocol parser already accepts worker v2 reasoning commands, and
`ullm-sq8-worker --served-model-manifest` already uses the common Rust
served-model loader. That is necessary but not sufficient. Current defaults
set `reasoning: None`; `sq8_serving_runtime.rs` returns
`reasoning_usage: None`; and `sq8_worker_runtime.rs` rejects a reasoning
request at release if usage was not supplied.

There is a second protocol gap: the decoder accepts both v1 and v2
discriminators, while `decode_with_profile()` does not enforce the loaded
profile's worker schema. A v2-profile SQ8 worker could therefore accept v1
commands without the explicit launcher compatibility mode required by the
now-ratified protocol v0.2 contract. The implementation enforces exact
profile-schema equality for
`generate`, `cancel`, and `shutdown`; any v1 compatibility mode must be
separately explicit and disabled in the production v2 profile. Add
wrong-discriminator and mixed-request negative tests.

Implement and test the actual reasoning phase, budget/forced-close transition,
answer reservation, and usage reporting across:

- `crates/ullm-engine/src/sq8_worker_protocol.rs`;
- `crates/ullm-engine/src/sq8_worker_runtime.rs`;
- `crates/ullm-engine/src/sq8_worker_backend.rs`; and
- `crates/ullm-engine/src/sq8_serving_runtime.rs`.

This work is separate from rebuilding the candidate binary. A newly rebuilt
binary is not v2-admissible until its behavior and build receipt satisfy these
tests.

Also migrate `tools/run-sq8-worker-acceptance.py` and
`tools/validate-sq8-worker-acceptance.py`, both of which currently hard-code
`ullm.worker.v1`, and add v2 acceptance fixtures without changing validation
of archived v1 outputs.

### 12.6 Campaign and release-bundle compatibility

The existing six-slot `ullm.generic_reasoning_release_bundle.v1` cannot carry
the SQ8 full campaign. Its promotion evidence is necessarily pre-receipt and
predates the candidate, while its exact `artifacts` object has no
model-campaign slot. Attaching post-candidate campaign hashes to promotion
evidence would create an impossible chronology. Do not reinterpret bundle v1
or hide the full campaign behind one of its existing fields.

Add `ullm.generic_reasoning_release_bundle.v2` with the same exact root,
identity, status, source, and rollback fields as v1, but an exact nine-field
`artifacts` object:

```text
release_evidence
release_validator
browser_evidence
browser_validator
promotion_evidence
promotion_receipt
model_campaign_manifest
model_campaign_evidence
model_campaign_validator
```

For SQ8, `model_campaign_manifest` binds the campaign's `SHA256SUMS`,
`model_campaign_evidence` is
`ullm.sq8.full_campaign.model_identity.v2`, and
`model_campaign_validator` is the independently recomputed
`ullm.sq8.openwebui_release.validation.v2` report. The campaign manifest must
cover an exact copy of the frozen candidate manifest as well as every
raw/derived campaign component. The bundle validator must rehash all manifest
entries with safe path/size/stable-read rules, invoke the SQ8 campaign
validator in a no-publish mode or isolated copy, and compare the entire
recomputed report.

Bundle v2 must also close two current-main AQ4 gaps:

- require the candidate manifest's live `promotion.receipt_sha256` to equal
  the bundled `promotion_receipt` component SHA-256, then require its schema,
  source, evidence reference, and worker identity to match the SQ8 promotion
  branch; and
- retain `ullm.openwebui.reasoning_browser_smoke.v3` as the identity-bearing
  intermediate schema and v4 as the lineage-only archive, then publish
  `ullm.openwebui.reasoning_browser_smoke.v5` for bundle v2. V5 keeps exact
  candidate/claim/run/output/actual-active lineage, separates the fixed
  Playwright runner image from the OpenWebUI server identity, and records
  exact server observations immediately before and after the browser gate.

`activate-served-model.py` must dispatch the independently validated bundle
schema explicitly: existing AQ4 remains on bundle v1 with unchanged
validation, while independent SQ8 requires bundle v2. In sections 12 through
14, “AQ4 reuse” always means the current-main served-model-v2/release-bundle
tooling in section 9, never the divergent SQ8-overlay lineage.

The following changes preserve other served models while adding that path:

| Component | Required change | Compatibility rule |
|---|---|---|
| `run-generic-reasoning-release-campaign.py` | Derive the expected process basename from the validated manifest instead of hard-coding `ullm-aq4-worker`; retain `/proc/<pid>/exe` SHA-256 binding | AQ4 still resolves to `ullm-aq4-worker`; SQ8 resolves to `ullm-sq8-worker`; all other owners fail |
| `run-openwebui-reasoning-browser-smoke.py` and its validator | Make the same manifest-derived process identity change and publish the new identity-bearing browser evidence | Do not accept an unbound caller-provided process name; retain old browser-schema validation only for historical/AQ4 inputs |
| `validate-generic-reasoning-release-bundle.py` | Preserve exact bundle-v1/AQ4 behavior; add exact bundle-v2 fields, SQ8 promotion dispatch, browser identity cross-check, and recomputed model-campaign validation | Reject mixed versions, extra/missing slots, wrong schema pairs, receipt mismatch, and identity mismatch |
| `prepare-generic-reasoning-release-bundle.py` | Add explicit bundle-v2 model-campaign inputs and copy/hash them after their independent validation | It is a packager, not the authority for an unvalidated SQ8 promotion or campaign |
| SQ8 full-campaign prepare/identity/production/validator modules | Add a new v2 evidence schema that binds the exact active candidate manifest SHA-256 and worker v2 protocol | Retain read-only validation of historical v1 archives; never reinterpret them as v2 evidence |
| `run-sq8-full-openwebui-campaign.py` | Require frozen candidate and actual active manifest paths, expected candidate SHA-256, and the claimed authorization; propagate them to every stage | All stage evidence must share source, manifest bytes, worker, product, model, image, claim, run ID, and output identity |

Passing a candidate pathname and expected digest is insufficient to prove what
systemd is serving. Before every campaign epoch/stage, each SQ8 and generic
reasoning/browser runner must stable-read the actual systemd-selected
`/etc/ullm/served-models/active.json`, compare its bytes and SHA-256 to the
frozen candidate, and retain that observation. The independent validator must
parse the candidate copy from the campaign bundle instead of trusting a
caller-supplied hash.

The fresh release set consists of both:

1. the expanded SQ8 full campaign (API contract, browser soak, direct cancel,
   Stop/recovery, post-header failure/recovery, latency, normal/restart
   resources); and
2. the generic reasoning release campaign plus browser reasoning smoke for
   disabled, budgets 32/128/256, and unbounded modes.

The complete release bundle, not the earlier pre-receipt promotion evidence,
must hash the final validated campaign outputs. Historical 2026-07-12
evidence remains archival and cannot satisfy any new identity slot.

### 12.7 Temporary cross-model v2 campaign window and durable one-shot

The exact-three `.v1` shape in this subsection is the original unissued
design draft. It is preserved as investigation history and is superseded by
the exact-six authorization/claim/outcome/recovery `.v2` contract in
`docs/specs/served-model-v2-cross-model-campaign-authorization-v0.2.md` and
the implementation snapshot in section 15. No live operator may issue or
claim the `.v1` shape below.

Candidate-active browser evidence requires a controlled temporary AQ4-v2 to
SQ8-v2 switch. Do not broaden
`--authorize-differing-worker-v2-bootstrap`, whose same-model rule is useful,
and do not treat a caller-invented attempt ID as authorization. A new ID/path
alone repeats the weakness of both the current AQ4 sidecar and the historical
overlay Gate.

Before execution, an operator must pre-issue an immutable, reviewed
authorization with this exact semantic shape:

```text
schema_version = ullm.served_model.v2_cross_model_campaign_authorization.v1
authorization_id
issued_at
expires_at
max_attempts = 1
authorization_note
purpose = temporary_candidate_active_evidence_collection_only
required_final_route = restore_exact_aq4_then_bundle_v2_activation
source = {commit, tree}
before = {model_id, format_id, manifest_sha256, worker_binary_sha256,
          promotion_source_commit}
candidate = {model_id, format_id, manifest_sha256, worker_protocol,
             worker_binary_sha256, promotion_source_commit,
             promotion_receipt_sha256}
campaigns = {
  sq8_full: {run_id, final_path},
  reasoning_release: {run_id, final_path},
  reasoning_browser: {run_id, final_path}
}
rollback = {backup_path, systemd_unit_sha256, environment_sha256}
prior_outcome = null or {path, sha256}
```

The authorization must be canonical, root-owned, mode 0444, nlink=1,
non-symlink, unexpired, and published with atomic no-replace semantics. A
machine-controlled claim path is derived from the authorization file hash in
a fixed registry, not supplied by the caller. The temporary-window tool must
atomically create that claim **before its first operational side effect**.
The claim consumes `max_attempts=1` even if any later preflight, activation,
campaign, validation, restoration, or outcome-publication step fails.

Both activation and all three campaign runners must require and revalidate
the same claim. Their source commit, active/candidate hashes, run IDs, final
paths, and output identities must exactly equal the authorization. A retry
requires a new operator-issued authorization whose `prior_outcome` binds the
previous immutable outcome; a caller cannot retry by choosing a new backup,
claim, or campaign pathname.

Implement this as a locked temporary-window wrapper/subcommand, not as a
successful bootstrap followed by an unrelated campaign:

1. validate and claim the authorization;
2. require the existing inactive-service precondition before the switch;
3. acquire and retain the activation lock for the whole temporary window;
4. verify both manifests as v2 and allow only
   `ullm-qwen3.5-9b-aq4`/`AQ4_0` to
   `ullm-qwen3-14b-sq8`/`SQ8_0`;
5. stable-read and bind active/candidate manifest, worker,
   promotion-receipt/source, unit, and environment hashes;
6. create an exact immutable AQ4 backup, atomically activate SQ8, reconcile,
   and verify the candidate identity;
7. run and independently validate both fresh campaign families under the
   claim; and
8. in an unconditional `finally`/signal-safe path, atomically restore the
   exact AQ4 bytes, reconcile OpenWebUI back to AQ4, and verify gateway,
   OpenWebUI, `/v1/models`, worker, and manifest identity.

The current activator restores only when one of its own hooks fails. Without
this encompassing transaction, a successful bootstrap followed by campaign
failure leaves SQ8 active. Manual `install` is not an acceptable restoration:
the wrapper must use the activation lock, expected-current and backup hashes,
atomic replace plus directory fsync, reverse reconciliation, and final health
checks.

Always publish a distinct immutable outcome keyed to the claim, recording the
authorization/claim hashes, stage results, campaign manifest/report hashes,
candidate-active observations, exact restored manifest hash, reverse
reconciliation, and final health/model checks. Authorization and claim remain
consumed even if outcome publication itself fails. These records are
machine-enforced operational authorization for the wrapper and campaigns;
they are not new gateway/worker manifest fields.

Only after the wrapper proves exact AQ4 restoration may bundle v2 be
assembled against that rollback target. Final SQ8 activation then uses
`--release-bundle`, never the temporary-window claim or bootstrap flag.

### 12.8 Specification and immutable-publication prerequisites

`docs/specs/served-model-manifest-v0.2.md`,
`docs/specs/sq8-worker-protocol-v0.2.md`,
`docs/specs/sq8-worker-build-release-v0.2.md`,
`docs/specs/sq8-serving-session-v0.2.md`,
`docs/specs/sq8-worker-acceptance-v0.3.md`, and
`docs/specs/sq8-openwebui-release-v0.2.md` are now ratified production
contracts. Their new schemas coexist with read-only validation of archived
v1 evidence; no frozen v0.1 document was reinterpreted.

Receipt, evidence, authorization, claim, outcome, final candidate, and bundle
publication all require atomic no-replace plus stable rehash. In particular,
`generate-served-model.py` currently publishes with `os.replace` and can
overwrite an existing output. Generation may still write a temporary
candidate, but release freezing needs a separate no-clobber immutable
publication step that produces mode-0444/nlink-1 bytes.

## 13. Ordered v2 implementation plan

The implementation order is dependency-sensitive:

1. **Ingest the reconstruction result as a baseline.** Receive the separately
   rebuilt worker/build record, capture its source/tree/path/size/hash, and use
   it to diagnose reproducibility. It is not the final release worker because
   steps 2 through 8 change worker-affecting v2 source.
2. **Implement SQ8 reasoning and exact protocol selection.** Add
   state/accounting behavior, reject v1 commands under the v2 profile unless
   an explicit compatibility mode is selected, and add CPU tests for all
   close, budget, cancellation, reset, usage, and wrong-schema cases.
3. **Ratify the v2 contracts.** Version the SQ8 manifest, worker, acceptance,
   serving-session, and release specifications; migrate the acceptance
   producer/validator while preserving archived v1 validation.
4. **Add SQ8 serving evidence/receipt tooling.** Implement strict schemas,
   build-receipt and live-hash checks, full product validation, pre-receipt
   CPU evidence, atomic no-replace publication, and mutation tests.
5. **Generalize manifest generation.** Add exact current-main AQ4/SQ8
   promotion dispatch, update the SQ8 profile to worker v2/reasoning/new
   receipt selectors, and generate a temporary candidate. Prove Python and
   Rust loaders accept it and reject protocol/reasoning/receipt mutations.
6. **Upgrade all candidate-active gates.** Derive process identity from the
   validated manifest, add identity-bearing browser evidence, introduce the
   SQ8 full-campaign v2 schema, compare actual active bytes to the candidate
   before every stage, include the candidate bytes in the campaign, and
   preserve all historical validator branches.
7. **Add bundle v2.** Preserve exact bundle-v1/AQ4 behavior; add the three
   model-campaign slots, SQ8 promotion dispatch, candidate-receipt cross-check,
   browser identity cross-check, independent full-campaign recomputation, and
   mixed-version/mutation rejection tests.
8. **Implement and review the durable campaign authorization and transaction.**
   Add pre-issued authorization, fixed-registry atomic claim, encompassing
   candidate-window wrapper, unconditional AQ4 restoration/reconciliation,
   immutable outcome, and a locked post-success rollback subcommand. Test
   success, every forced-failure boundary, interruption, replay, expiry, and
   exact-byte restore entirely on private manifest copies.
9. **Build and freeze the final candidate from one release commit.** After all
   worker-affecting v2/authentication changes land, build a fresh worker and
   build receipt from that same clean detached commit. Publish new
   mode-0444/nlink-1 evidence, receipt, worker, profile snapshot, candidate,
   authorization, and bundle inputs with no-clobber semantics. Do not reuse
   either the baseline rebuilt worker or the earlier `0b5f...` v1 dry-run.
10. **Parent-only candidate-active window.** Snapshot the exact AQ4 rollback
    bytes, atomically claim the reviewed authorization once, run both fresh
    campaign families inside the locked wrapper, independently validate them,
    and require verified AQ4 reverse reconciliation on exit.
11. **Assemble the complete bundle v2.** Bind the final candidate manifest,
    worker, full source commit, promotion pair, fresh validated campaigns,
    content-addressed OpenWebUI image, and restored AQ4/unit/environment
    rollback hashes. Require `status: complete`.
12. **Parent-only final promotion.** Invoke
    `activate-served-model.py --release-bundle ...`; run all post-activation
    byte-identity/health/browser checks. Any failed activation transaction
    restores exact old bytes before rollback hooks; any later operator
    rollback uses the new locked rollback subcommand, never manual `install`.

Steps 1 through 9 are implementation or CPU-only preparation. Steps 10 and
12 are the only service/GPU/systemd windows and were not performed during
this investigation.

## 14. Admission checklist and remaining human decisions

Cutover remains fail-closed until every item below is true:

- [ ] The separate worker reconstruction is recorded as a baseline, and a
      fresh complete `sq8-worker-build-release-v0.2.md` directory is produced
      from the same clean release commit after every worker-affecting v2
      change. The complete directory is byte-copied and validated at its final
      staged path before any absolute-path profile or promotion artifact is
      generated.
- [ ] Qwen3 reasoning dialect and budget semantics are approved and all Rust,
      gateway, generator, and mutation tests pass.
- [x] The v2 manifest/worker/session/acceptance/release specs are ratified,
      and a v2-profile worker rejects v1 commands unless an explicit
      non-production compatibility mode is selected.
- [ ] The SQ8 serving promotion evidence/receipt pair exists and independently
      validates.
- [ ] The generated candidate is served-model v2/worker v2, has the exact
      SQ8_0 product and worker, and validates in Python and Rust.
- [ ] AQ4 regression proves promotion dispatch and bundle validation are
      unchanged for existing served models.
- [ ] Bundle v2 includes and independently recomputes the full SQ8 campaign,
      cross-binds candidate promotion receipt and browser identity, and rejects
      every v1/v2 or AQ4/SQ8 schema mix.
- [ ] A separate AQ4-to-AQ4 runtime-hardening promotion has replaced the
      current user-owned bootstrap closure with a root-owned protected
      closure, fresh promotion pair, new frozen manifest, and immutable
      locked-transaction outcome.
- [ ] A human pre-issues the narrow AQ4-v2 to SQ8-v2 campaign authorization;
      its fixed-registry claim is atomically consumed exactly once.
- [ ] The exact SQ8 and AQ4 campaign commits are separately materialized as
      root-owned, non-writable standalone clones with in-tree `.git`
      directories, no hardlinks/alternates/ACLs/symlinks, and protected
      ancestry. The runner itself is invoked from the sealed SQ8 clone.
- [ ] Both candidate and rollback manifests resolve only to root-owned sealed
      runtime closures; worker/engine binaries, promotion inputs, tokenizers,
      product/package manifests, and all payloads remain repinnable.
- [ ] A real, private, sufficiently unexpired OpenWebUI frontend session JWT
      is provisioned for the parent-only browser gates.
- [ ] The three fresh candidate-active SQ8 campaigns and three fresh
      restored-AQ4 campaigns are complete. Each phase binds its exact active
      bytes, claim, run/output, worker, source, model, promotion, and image
      identities without conflating the distinct AQ4 and SQ8 source lineages.
- [ ] The encompassing wrapper restores the exact AQ4 rollback bytes,
      reconciles OpenWebUI back to AQ4, and publishes a successful immutable
      outcome before complete-bundle assembly.
- [ ] The independent bundle validator reports complete; final activation
      uses bundle v2 through `--release-bundle` and no bootstrap/v1 exception.
- [ ] Post-activation `active.json` bytes equal the frozen candidate exactly;
      any later rollback uses the locked rollback subcommand.

The remaining human decisions are operational: review the final clean
worker/build receipt and exact candidate/rollback/run/output/expiry
identities; pre-issue the narrow campaign authorization; then review the
complete bundle-v2 and final-activation plan hashes before execution.

OpenWebUI session-token provisioning is also an operator prerequisite, but it
does not change the v2 design. No decision is needed about the historical
overlay lineage: it is definitively out of scope and must not be imported.

## 15. Implementation snapshot before live holds

This section supersedes present-tense implementation gaps above while
preserving the investigation history. It applies only to the independent
Qwen3-14B-FP8 `SQ8_0`, never the historical Qwen3.5 `AQ4_0` partial-FP8
overlay.

### 15.1 Nine-step status

| Step | Status |
|---|---|
| 1. Reasoning/accounting and worker-v2 equality | **Complete.** Exact v2 command discriminators and disabled, 0/32/128/256/unbounded, close, reservation, cancellation, publish, reset, and accounting paths are implemented. |
| 2. V2 specifications | **Complete.** Manifest v0.2, worker v0.2, worker build release v0.2, session v0.2, acceptance v0.3, and OpenWebUI release v0.2 are ratified; archived v1 validation remains separate. |
| 3. SQ8 promotion and AQ4/SQ8 dispatch | **Implementation complete; final artifacts wait for step 4.** SQ8 evidence/receipt validation and no-replace writers coexist with the unchanged AQ4 branch. |
| 4. Final clean worker rebuild | **Pending at this documentation snapshot.** The earlier `uLLM-sq8-manifest-candidate-release-ee62d04e` is only a preserved baseline. No final path or hash is asserted here. |
| 5. Per-stage active-byte binding | **Complete.** Ordered observations compare the actual active bytes to SQ8 before/after all three SQ8 campaigns and to exact restored AQ4 before/after all three AQ4 campaigns. Candidate copies, claims, run IDs, and outputs are validator-bound. |
| 6. Generic bundle v2 | **Implementation complete; live complete bundle pending.** The SQ8 nine-slot path coexists with unchanged AQ4 bundle v1. A successful transaction produces the fresh AQ4 bundle v1; the later SQ8 bundle v2 awaits fresh GPU/JWT output. |
| 7. One-shot authorization | **Implementation complete; not issued or claimed in production.** Schema v2 authorizes exactly six distinct campaigns, fixed AQ4/SQ8 lineages and separate fixed browser/server images, fixed registries, one atomic claim, one deadline, immutable outcome, and predecessor binding. Production admission additionally requires sealed source clones and runtime closures. |
| 8. Locked campaign transaction | **Implementation complete in private/mock tests; production held.** Plan v2 runs the three SQ8 campaigns, unconditionally restores/reconciles AQ4, then runs fresh AQ4 reasoning/browser/bundle-v1 stages under the same lock/deadline. Legacy outputs use no-replace staging; source and complete runtime closures are recursively sealed/repinned; each browser stage is bracketed by fixed-server verification; structured restoration proof and locked recovery remain mandatory. The current AQ4 bootstrap closure is intentionally inadmissible until a separate AQ4 hardening promotion. |
| 9. Final activation procedure | **Prepared, never executed.** Plan v2 re-inventories all six outputs, validates fresh AQ4 bundle v1 before SQ8 bundle v2, seals both activation runtime closures, and retains default read-only preflight, exact plan-hash/literal confirmation, failure restoration, and later locked rollback. |

The final worker must be built only after all implementation commits, from one
clean detached commit/tree. No production `active.json`, worker, systemd
configuration, service, GPU, or V620 state was changed for this snapshot.

### 15.2 Exact release schemas and layouts

`ullm.generic_reasoning_release_evidence.v2` adds
`ullm.served_model.campaign_lineage.v2` to the v1 evidence. Its immutable
mode-`0555` output directory contains exactly:

```text
active-manifest-binding.json
active-manifest-observations.jsonl
candidate-served-model.json
cases.json
lifecycle.json
resource-samples.jsonl
summary.json
```

The observations are `preflight`, stream/nonstream for disabled,
budget-32/128/256 and unbounded, then `final`.

`ullm.openwebui.reasoning_browser_smoke.v5` retains the v4 source/identity
and campaign-lineage payload, adds the distinct Playwright `browser_image`,
and binds identical before/after observations of the fixed OpenWebUI server
container ID, image ID, configured image, container name, running state, PID,
and start timestamp. Its immutable mode-`0555` output
contains exactly:

```text
active-manifest-binding.json
active-manifest-observations.jsonl
browser-evidence.json
candidate-served-model.json
```

Its observations are `preflight`, `browser-launch`, `browser-complete`,
`validation`, and `publication`. Both schemas bind the immutable claim,
authorized run/output, included candidate bytes, and every actual-active byte
comparison.

`ullm.generic_reasoning_release_bundle.v2` has exactly these nine artifact
slots:

```text
release_evidence
release_validator
browser_evidence
browser_validator
promotion_evidence
promotion_receipt
model_campaign_manifest
model_campaign_evidence
model_campaign_validator
```

It requires generic evidence/validator v2, browser v5/validator v3, SQ8 full
campaign identity/validation v2, the candidate's live SQ8 promotion receipt,
one source/candidate/worker/tokenizer/image/claim lineage, and exact rollback
manifest/unit/environment hashes. The validator recomputes generic, browser,
promotion, and isolated no-publish full-campaign validation. Bundle v1 remains
the exact six-slot AQ4 path. Validator code is authoritative for every exact
root-field set and size/stable-read bound.

### 15.3 Transaction and final-route boundary

The campaign runner accepts no caller command vectors. It derives
`ullm.served_model.v2_cross_model_campaign_plan.v2`, and the v2 claim is its
first mutation. `expires_at` caps all SQ8 and restored-AQ4 evidence commands
but not restoration. The held activation lock, pinned directory descriptor,
`renameat2(RENAME_EXCHANGE)`, displaced inode/byte checks, directory `fsync`,
subreaper containment, and signal deferral extend through all six campaigns,
AQ4 restoration, structured live proof, immutable outcome, and lock close.
After candidate reconciliation/readiness, a monitored stabilization gate
requires at least 900 seconds of unchanged authorization/claim, source,
candidate/runtime, exact active bytes, and service/gateway/worker epoch before
`sq8_full`. The full campaign's fixed timeout maximum is 21,600 seconds and
is still capped by the authorization time remaining. The fixed OpenWebUI
image is checked against the running container after reconciliation and
around each browser campaign.

Only the evidence-producing subprocesses drop to the fixed service identity
and exact supplementary-group allowlist. They write a randomized
service-owned transaction-private staging tree, never an authorized final
path. After every descendant has been reaped, root descriptor-walks and
adopts the tree, rejects symlinks, special/hard-linked/cross-device entries,
unexpected ownership/layout/size, or a leaked staging path, freezes and
validates it, then publishes without replacement. Root retains claim, lock,
manifest exchange, validation, restoration, proof, and outcome ownership.
This boundary also contains the separately sealed, current transaction-aware
AQ4 producers whose native publication primitive can replace an existing
target. The AQ4 campaign/hardening source must not be the historical
`qwen35-9b-aq4-fidelity-promotion-f1a3cf4c` source/route; that lineage lacks
the current transaction and Docker-lease contracts. The separate
default-read-only recovery CLI can use a consumed claim after expiry without
the selected AQ4 campaign source or campaign outputs.

The transaction never root-executes tools from the writable development or
historical worktree. Production preflight accepts only standalone SQ8 and AQ4
Git roots whose complete worktree and `.git` metadata are root-owned,
symlink/ACL/alternate/hardlink-free, and not group/world writable through
their ancestry. It captures a stable recursive fingerprint, rechecks it
around fixed, non-mutating Git queries and every campaign command, and strips
ambient loader/Python/Docker/Git configuration from command environments.
The run/recovery CLI must itself come from the sealed SQ8 root. Recovery still
does not depend on the old AQ4 source after expiry: it seals the SQ8 recovery
code it actually executes, the exact AQ4 backup runtime, and the shared
unit/environment/credential/rollback-operation closure. It deliberately does
not require the displaced SQ8 candidate or candidate runtime.

The same admission applies to the complete runtime closure reachable from
both manifests: worker and legacy engine executables, promotion
receipt/evidence, tokenizer members, product/package manifests, and every
package payload. Every entry and replace-capable ancestor must be root-owned
and protected; seals are rechecked at command boundaries and in standalone
recovery. The current AQ4 bootstrap manifest hash
`5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a`
does not pass: its worker, receipt/evidence, 1,044 package leaves (about
7.70 GB), and four tokenizer files live below user-owned/writable trees.
Failing before claim is the required behavior.

Therefore a separate reviewed AQ4-to-AQ4 runtime-hardening promotion is a
live prerequisite, not part of the exact-six window. It must root-stage the
complete closure outside `/home`, create fresh AQ4 promotion evidence and
receipt for the new absolute paths, freeze a new AQ4 manifest, and activate it
through its own locked authorization/outcome/rollback route. The historical
receipt cannot be copied because it binds the old worker, product, and
manifest paths. The exact-six authorization must name this new hardened AQ4
manifest as `before`.

Source-owned Python commands use the canonical
`/usr/bin/python3.12 -I -S -B` prefix and a minimal import/configuration
environment. The fixed ROCm vendor scripts use
`/usr/bin/python3.12 -E -S -B` because their root-owned sibling imports are
required. Root-owned OS/Python/ROCm libraries, ELF loaders and shared
libraries, Git/Docker/systemd internals, the Docker daemon/container runtime,
`/bin/sh`, and `/usr/bin/ps` are explicit reviewed TCB, rather than falsely
described as recursively campaign-sealed artifacts. The browser-session JWT
is fixed at `/run/ullm-campaign-secrets/openwebui-session.jwt`, owned
`uid=0,gid=1000`, mode `0640`, below an `uid=0,gid=1000`, mode `0750`
directory; the API key remains `/etc/ullm/openai-api-key` with the same file
ownership and mode. Neither secret value enters public evidence.

Final activation is a separate boundary. Plan preparation requires a
`succeeded_restored` outcome, unchanged inventories for all six campaigns, a
fresh gate-eligible AQ4 bundle v1 derived from that outcome, a gate-eligible
complete SQ8 bundle v2, exact AQ4 rollback bytes, and reviewed direct
executables. Its runner source is itself a module-derived, clean detached
standalone root-owned Git seal; the plan pins that source commit/tree and
rechecks the in-memory seal at every stage. Activation and later rollback are
read-only unless the exact plan SHA-256 and respectively
`ACTIVATE_SQ8_0_FROM_RESTORED_AQ4` or
`ROLLBACK_SQ8_0_TO_EXACT_AQ4` are supplied. Neither was run here.

### 15.4 Live holds and human order

Still prohibited or unavailable: production `active.json` writes;
`ullm-openai.service` start/stop/restart; systemd changes; production GPU
campaign or GPU worker launch; V620 access; obtaining/using the real
OpenWebUI browser-login JWT; production claim; and activation/rollback.

Human/operator order:

1. prepare and execute the separately reviewed AQ4-to-AQ4 runtime-hardening
   promotion; this future step requires its own GPU/service window and must
   finish with a protected AQ4 runtime closure and immutable outcome;
2. build/freeze the complete worker-release-v2 directory from one clean
   detached commit; byte-copy and validate the whole release at its final
   protected staged path without rewriting its receipt or seal;
3. against that staged worker path and one explicitly supplied current sealed
   source root, create in order the profile, ephemeral manifest, CPU report,
   distinct promotion evidence/receipt pair, and final candidate;
4. root-stage the rest of the complete SQ8 runtime closure, create separate
   root-owned standalone sealed clones for the exact SQ8 and current
   transaction-aware AQ4 campaign commits (not the historical
   `f1a3cf4c` AQ4 source/route), and complete read-only campaign preflight
   from the sealed SQ8 clone;
5. provision the real browser-session JWT and pre-issue the exact,
   sufficiently unexpired authorization;
6. enter the separately approved maintenance window, explicitly stop
   `ullm-openai.service`, and read-only verify that every fixed
   `inactive_services` member is inactive **before** invoking any command
   that claims the authorization; this is a future operator action and was
   not executed while preparing this runbook;
7. only after that inactive-state verification, invoke the locked
   transaction; its first mutation atomically claims the authorization, then
   execute the three fresh SQ8 campaigns, exact AQ4 restore/reconciliation,
   and the three fresh AQ4 campaigns (including complete AQ4 bundle v1) in one
   locked window; require immutable `succeeded_restored`;
8. assemble and independently validate the SQ8 `status: complete` bundle v2,
   retaining the outcome-bound AQ4 bundle v1 as a separate admission input;
9. prepare/review the final plan and read-only preflight; and
10. execute activation only after explicit Claude+user approval, using the
   plan-bound rollback route if rollback is later needed.

## 16. Final audit delta

This section supersedes the admission and final-activation details in
sections 15.1 and 15.3 where they refer to plan v2. It also supersedes the
ordering of steps 1 and 2 in section 15.4: this task fixes the final SQ8
release immediately after the last clean source commit, before the future
live AQ4-to-AQ4 hardening window. It does not authorize a live campaign or
activation.

### 16.1 Final nine-step implementation state

| Step | Final implementation state |
|---|---|
| 1. Qwen3 reasoning/accounting and worker-v2 equality | **Complete on CPU.** Exact worker-v2 discriminators, one-token Qwen3 delimiters, budget/forced-close/history/answer-reservation/cancellation/reset behavior, and both release counters are enforced. GPU worker startup remains a live hold. |
| 2. V2 specifications | **Complete.** Manifest, worker, build release, session, acceptance, SQ8 OpenWebUI release, authorization, bundle, and final-route contracts are versioned and fail closed. |
| 3. Promotion evidence and dispatch | **Complete.** Final AQ4/SQ8 worker-v2 generation requires the exact format selector. Dedicated promotion-only ephemeral APIs/schemas cannot be admitted as final SQ8 promotion receipts. The selectorless compatibility exception is limited to the historical SQ8 worker-v1 input. |
| 4. Clean worker rebuild | **Prepared here; performed only after the final documentation commit.** The build uses a new detached clean clone and new release/build paths. The old `uLLM-sq8-manifest-candidate-release-ee62d04e` remains a baseline only. |
| 5. Actual active bytes per stage | **Complete.** The generic-reasoning, browser, and SQ8-full active-binding evidence validators require the production path `/etc/ullm/served-models/active.json`; the transaction checks its exact bytes against the expected candidate or restored AQ4 state at every boundary. |
| 6. Generic bundle v2 | **Complete and coexisting with v1.** Bundle v2 derives the claim only from the fixed registry/authorization hash and validates the three SQ8 campaigns plus final promotion lineage. AQ4 bundle v1 remains a separate unchanged admission input. Live complete bundles await fresh campaigns. |
| 7. One-shot authorization | **Complete; not used live.** New campaign admission uses in-window `load_live_claim` with bound inputs. Archival bundle/outcome/recovery audit and restore/recovery repinning may load the expired consumed claim, but cannot authorize new work. Claim/outcome publication uses `renameat2(RENAME_NOREPLACE)`, parent `fsync`, and exact single-link re-open. |
| 8. Locked transaction | **Complete in private/mock tests; not run live.** All v2 producers require transaction-private staging. Preflight, pre-switch, and repin perform full final SQ8 receipt/evidence/manifest validation. Protected root wrappers reject ambient interpreter/import state before loading campaign code. AQ4 restore and recovery retain only the archival authority needed after expiry. |
| 9. Final activation route | **Prepared as plan v3; never executed.** Durable intent precedes the swap; credential seals precede intent; success receipt publication is the commit boundary. Explicit recovery covers an intent-only crash, `failed_restore`, and `rollback_incomplete`, without consuming the success pathname on a failed attempt. |

### 16.2 Admission and publication invariants added by the final audit

Normal worker-v2 generation is fail closed: AQ4 requires the final AQ4
selector and SQ8 requires the final SQ8 selector. Promotion preparation calls
separate non-CLI ephemeral generation functions with exact AQ4/SQ8 schemas;
the SQ8 transaction's candidate validator always runs complete final
promotion validation and generator-binding validation at preflight, before
the switch, and on every relevant repin.

All v2 evidence producers require their transaction-supplied staging
directory. Each new campaign stage reloads the consumed authorization through
`load_live_claim`, rechecking the current validity window and bound inputs.
Post-campaign bundle/outcome/recovery audit and restore/recovery repinning are
deliberately different: they may treat the claim as archival evidence, derive
its pathname and required UID from the fixed production policy, and never
authorize a new campaign mutation.

The campaign issue/claim/run/recovery wrappers perform their exact
root-owned `/usr/bin/python3.12 -I -S -B` and protected-source checks before
local imports. Immutable claim, outcome, recovery, AQ4 backup, bundle, intent,
activation, rollback, and recovery-success publication uses a one-name
`renameat2(RENAME_NOREPLACE)` commit, parent-directory `fsync`, and
format-specific exact post-commit validation. Policy-owned authorization and
final receipts also verify owner; the bundle verifies exact bytes, mode, and
`nlink == 1`. There is no temporary hard-link state.

### 16.3 Final activation v3 and crash recovery

`ullm.served_model.final_activation_plan.v3` binds operations v2, separate
intent, activation-outcome, rollback-outcome, recovery-success, and dedicated
recovery-proof destinations. Read-only preflight is not ready when either
live credential cannot be sealed. Execution reacquires the same activation
lock, repeats credential sealing, publishes
`ullm.served_model.final_activation_intent.v1` durably, and only then performs
the exact-current active-manifest exchange.

`ullm.served_model.final_activation_outcome.v2` publication is the successful
activation commit boundary. No fallible source recheck follows it before the
termination guard is committed. A crash after intent with no activation
outcome, an outcome with `failed_restore`, or an interrupted/incomplete
manual rollback is handled only by:

```text
sudo -- /usr/bin/python3.12 -I -S -B /ABSOLUTE/ROOT-OWNED-SEALED-SQ8-SOURCE/tools/rollback-served-model.py \
  --plan /ABSOLUTE/final-activation-plan.json \
  --recover-failed-activation \
  --execute \
  --confirm-plan-sha256 EXACT-PRINTED-PLAN-SHA256 \
  --confirmation RECOVER_FAILED_SQ8_0_TO_EXACT_AQ4
```

The recovery route accepts only exact plan-bound SQ8 candidate bytes or exact
AQ4 rollback bytes, restores AQ4 when necessary, and requires reverse
reconciliation plus its dedicated AQ4 live proof. A failed recovery writes
an immutable attempt-specific audit/proof derived from the plan-bound base
and a random 256-bit attempt ID. It does not occupy the one-shot success
receipt, so a later explicitly reviewed invocation can retry.

The final focused CPU/private/mock regressions passed for dispatch/promotion,
bundle v1/v2, authorization/producers, locked transaction, final
activation/recovery, SQ8 full campaign contracts, additional AQ4
served-model compatibility, and the OpenAI gateway. No production
`active.json`, service, systemd, Docker container, GPU, V620, JWT, campaign,
activation, rollback, or recovery action was performed.
