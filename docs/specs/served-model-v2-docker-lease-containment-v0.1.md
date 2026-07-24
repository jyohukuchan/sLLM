# Served-model v2 Docker lease containment v0.1

Status: implementation contract for the AQ4_0/SQ8_0 cross-model campaign.

This contract covers transient containers created during one consumed
`ullm.served_model.v2_cross_model_campaign_claim.v2` transaction. It does not
authorize a campaign, change `active.json`, or replace the existing
served-model activation and restoration contracts.

## Lease identity and trusted executables

The transaction derives exactly one Docker lease from the immutable claim
snapshot:

```text
com.ultimatellm.served-model-campaign.claim=<claim snapshot SHA-256>
```

The label is exported as `ULLM_CAMPAIGN_DOCKER_LEASE_LABEL`. The source-bound
wrapper path is exported as `ULLM_CAMPAIGN_DOCKER`.

`tools/ullm-campaign-docker` is the only Docker executable admitted to
campaign producers. It injects the exact lease label into `docker run`,
`docker create`, `docker container run`, and `docker container create`.
Caller-provided `--label`, `--label-file`, and Docker global options are
rejected. Attached and clustered short-label forms such as `-lKEY=value` and
`-itlKEY=value` are rejected as well, so Docker option ordering cannot
replace the injected lease. Other Docker subcommands are forwarded without
inventing a label.

The fixed command plan binds the wrapper in both the candidate and AQ4
routes, including the historical detached AQ4 source producers. The
transaction runtime seal pins and revalidates:

- the source-bound wrapper;
- `/usr/bin/python3.12`, including the wrapper's isolated shebang
  interpreter;
- the wrapper backend `/usr/bin/docker`;
- every other existing command executable and source seal.

A missing, route-local-only, changed, non-executable, or direct
`/usr/bin/docker` command binding fails preflight.

Validation is per command, not merely per candidate/AQ4 side. `--docker=...`
is forbidden; `--docker` may occur once and its following argument must be
the exact wrapper. An alternate executable or argument whose basename is
`docker` is forbidden. The SQ8 full, generic reasoning, browser reasoning,
and OpenWebUI image-verifier producers each require their own exact wrapper
binding. Direct wrapper operations such as `run` and `compose` require the
canonical `/usr/bin/python3.12 -I -S -B <source wrapper>` prefix.

## Propagation through the SQ8 full campaign

The top-level full-campaign CLI accepts `--docker`. In v2 mode it must equal
the canonical `ULLM_CAMPAIGN_DOCKER` path, and the claim label must have the
exact key and a lowercase 64-hex value.

The backend passes the wrapper explicitly to all six gate processes and
passes both lease environment variables into their fixed child environment.
The API-contract, direct-cancel, and latency gates verify that their
collector's Docker binding matches the argument. The combined, stop, and
failure gates pass the wrapper to their existing Docker command builders.
The collector's HTTP client and readiness helper containers, and the
operational preflight's Docker inspect and readiness helper, use the same
binding.

The operational gateway readiness probe does not require `sudo`, `nsenter`,
or `docker exec`. It runs a mount-free helper container with:

- `--rm --pull=never`;
- `--network=container:<exact OpenWebUI 64-hex container ID>`;
- UID/GID `1000:1000`;
- read-only root, all capabilities dropped, and
  `no-new-privileges`;
- fixed PID, memory, and no-exec tmpfs bounds;
- the pinned OpenWebUI image and one fixed Python GET source;
- no credential, host, or output mounts.

The surrounding operational preflight still compares the OpenWebUI
container ID, PID, image, network, restart count, and start epoch before and
after the probe.

## Admission, cleanup, and proof

After the authorization is atomically claimed, preflight queries the Docker
daemon for all containers carrying the exact label. The inventory must be
empty before `active.json` can be switched.

Every fixed command completion path, including exception and timeout paths,
runs root-owned lease cleanup before control returns to campaign logic.
Each inventory is bounded to at most 256 unique full container IDs. Every
non-empty inventory is removed with one batched `docker container rm
--force <all IDs>` call, so timeout cost never multiplies per container.

An empty inventory is not immediately accepted. The daemon must report at
least three consecutive empty inventories spanning the full two-second
command-termination grace, with 250 ms bounded polls. A container which
appears after an earlier empty result is removed, resets the quiet interval,
and must be followed by a new full quiet interval. The same quiescence rule
applies to the non-mutating preflight zero check.

Each Docker control call has a timeout of at most 30 seconds and is capped by
the remaining cleanup deadline. One settle attempt has a fixed 95-second
wall-clock deadline and a 512-poll hard bound. A slow daemon or continuous
late creation therefore fails closed instead of extending restoration
without bound.

The transaction performs another cleanup and zero proof before any AQ4
restore attempt. It performs a final zero proof before the live AQ4
restoration proof and durable outcome. A cleanup-control error is
`CommandContainmentLost`; it prevents both `succeeded_restored` and
`failed_restored`. If containment was lost after candidate activation, the
transaction does not overwrite the active slot and publishes
`failed_restore` for the locked recovery route.

The recovery route uses the same claim-derived label. It removes and proves
zero stale containers before replacing the active manifest, cleans after
each reverse/final command, and proves zero again before the live AQ4 proof.
A recovery receipt can say `restored` only when that final zero proof
succeeds.

## Persistent OpenWebUI lifecycle boundary

The lease owns transient helper, HTTP-client, and browser containers. It
does not own or remove the persistent `open-webui` Compose service.
`docker compose up -d` is a source-bound reconciliation operation rather
than a transient producer.

Lease-zero is therefore not used as proof of persistent-route correctness.
After candidate reconciliation, the transaction waits for the fixed
gateway readiness result and verifies the pinned OpenWebUI image. After
restoring exact AQ4 bytes it repeats reconciliation, readiness and image
verification, rechecks exact `active.json`, and validates the live
service/gateway/worker restoration proof. A failed or interrupted
reconciliation cannot bypass these checks. If the transaction cannot prove
the route, it cannot publish a restored status.

This boundary assumes the Docker daemon and the sealed Docker CLI honor
completed reconciliation requests. A daemon integrity failure is outside
the campaign authorization model and remains a host incident.

## Operational prerequisites and exclusions

The production service must be in the transaction's required inactive
maintenance state before execution begins. The readiness helper removes the
former dynamic passwordless `sudo`/`nsenter` prerequisite.

Tests for this contract use fake Docker runners and fake subprocesses only.
They must not change the production `active.json`, contact the production
Docker daemon, operate systemd, use a GPU, or consume an OpenWebUI session
JWT.
