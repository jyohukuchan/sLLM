# SQ8_0 worker release staging and AQ4_0 production-asset audit v0.1

Status: the SQ8_0 root-owned staging was completed on 2026-07-24 and
read-only reverified on 2026-07-26.  **The SQ8 staging portion of the
root-operation block was executed; AQ4_0 root operations remain unexecuted.**
Neither a served-model manifest activation nor a final activation was performed.

This runbook stages only the sealed SQ8_0 v2 worker release.  It does not
activate a manifest, change `/etc`, restart a service, collect campaign
evidence, use a GPU, or handle login/JWT material.  The current AQ4_0 active
manifest and every existing SQ8/AQ4 candidate remain immutable inputs.

## Execution record

On 2026-07-24, the sealed SQ8_0 v2 worker release was published below
`/opt/ullm/releases` as a root-owned, no-hardlink staging copy.  This was
the SQ8-only staging operation; it did not activate a manifest, change an AQ4
asset, or perform an AQ4 root operation.

The 2026-07-26 read-only re-verification observed the following:

| Check | Observed result |
|---|---|
| staged path | `/opt/ullm/releases/uLLM-sq8-v2-final-worker-release-3bc9078d` |
| protected ancestry | `/opt`, `/opt/ullm`, and `/opt/ullm/releases` are root:root mode `0755`; the release is root:root mode `0555`, nlink 2 |
| member metadata | five metadata files are root:root mode `0444`, nlink 1; `ullm-sq8-worker` is root:root mode `0555`, nlink 1 |
| `sha256sum -c SHA256SUMS` | README, build provenance, build receipt, and worker all reported `OK` |
| `SEALED.json` bindings | source commit `3bc9078d1ca5a49aad060d667aac19e2aa53ee86`, source tree `bd95c4f65168b05f4ed572a7f89e35be23ede975`, worker `0b9989c26e656123addef15ffbf96b1aadf866a6eca06f02af8cab9bccb18a83`, provenance `d4a123210ea9680e115f2af1ea8e2285bf6a5c36a18c5db7b8ec779231a0c19d`, receipt `986708497df09d4d7998f79c0e5fe29a0a69c8c37aa7ed2e28643c16faf69cd3`, and sums `ded0a829ef8ab67a19883b454621131a63ff036afe1181d931a5e39d1cd548c5` all matched the staged bytes |
| source equality | `diff -r` against the user-owned release exited 0 with no output; corresponding staged and source members have distinct inodes |

The active manifest SHA-256 was observed as
`5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a`
during that re-verification.  This is an observation only: this staging did
not replace `/etc/ullm/served-models/active.json`.

AQ4_0 remains an audit failure requiring its separately reviewed
AQ4-to-AQ4 runtime-hardening promotion.  No AQ4 copy, ownership change,
staging, manifest activation, or other root operation has been performed by
this runbook.

## 1. Read-only findings

### 1.1 SQ8_0 final worker release

Source:

```text
/home/homelab1/coding-local/ultimateLLM/uLLM-sq8-v2-final-worker-release-3bc9078d
```

The release is a complete relocatable
`ullm.sq8_worker_release_seal.v2`:

| Item | Value |
|---|---|
| source commit | `3bc9078d1ca5a49aad060d667aac19e2aa53ee86` |
| source tree | `bd95c4f65168b05f4ed572a7f89e35be23ede975` |
| worker SHA-256 | `0b9989c26e656123addef15ffbf96b1aadf866a6eca06f02af8cab9bccb18a83` |
| build receipt SHA-256 | `986708497df09d4d7998f79c0e5fe29a0a69c8c37aa7ed2e28643c16faf69cd3` |
| build provenance SHA-256 | `d4a123210ea9680e115f2af1ea8e2285bf6a5c36a18c5db7b8ec779231a0c19d` |
| seal SHA-256 | `e01c18593606e173dc154a584feb68d511e224f418ebf2b700aaf377fd171381` |
| `SHA256SUMS` SHA-256 | `ded0a829ef8ab67a19883b454621131a63ff036afe1181d931a5e39d1cd548c5` |
| README SHA-256 | `b3e2157a02105d1ff7e8771ee0b51bea76d128f1e4ec5e086a8e369d817cf07b` |

The source directory has the exact six-member set, mode `0555`; the worker is
mode `0555`, nlink 1; and the five metadata members are mode `0444`, nlink 1.
Both offline validation and live-source validation against the clean detached
`uLLM-sq8-v2-final-source-3bc9078d` checkout passed.

The release is not yet a production runtime seal.  It and its pathname
ancestry are owned by UID 1000 below `/home`, and intermediate directories
are mode `0775`.  Chowning the leaf in place would not protect the ancestry
and would mutate the preserved candidate.  A no-hardlink copy below protected
root-owned ancestry is required.

The older
`uLLM-sq8-manifest-candidate-release-ee62d04e` is not the final release.  It
contains an `ullm.worker.v1` worker with a different hash and a served-model
manifest that embeds its historical absolute path.  Do not copy, rewrite, or
reuse it for the final SQ8_0 identity.

### 1.2 AQ4_0 production assets

There is no compliant root-owned AQ4 release convention to mirror.  The live
manifest itself is root:root `0644`, but it names:

- a user-owned AQ4 release below user-owned `/home` ancestry;
- a user-owned writable product root, promotion receipt, and promotion
  evidence;
- 1,044 user-owned mode-`0664` package payloads plus their manifest;
- a product-root symlink, `artifact -> package`; and
- a user-owned writable tokenizer root.

The exact final-activation runtime seal rejects this closure.  It scans the
entire product root, not only `package/`, so the symlink and unrelated
historical entries are in scope.  The AQ4 source checkout is also a linked
worktree with a `.git` pointer file, not a root-owned standalone clone.

The AQ4 result is therefore an audit failure, not a permission-fix request.
Do not chown/chmod these live source assets in place and do not copy the whole
product root.  AQ4 needs its own path-changing, separately reviewed
AQ4-to-AQ4 runtime-hardening promotion with:

1. a purpose-built minimal product tree containing the package and fresh
   path-bound promotion evidence/receipt, with no symlink;
2. a protected worker/legacy-engine release;
3. a protected tokenizer tree;
4. a standalone no-hardlink source clone at the exact AQ4 promotion commit;
5. a newly frozen manifest naming those absolute paths;
6. a locked activation/restoration and live proof; and
7. fresh AQ4 release/browser campaigns and complete bundle v1.

Existing evidence binds the old paths and manifest hash and cannot be copied
as fresh hardening evidence.

## 2. Selected and staged SQ8 destination

The completed destination is:

```text
/opt/ullm/releases/uLLM-sq8-v2-final-worker-release-3bc9078d
```

`/opt` is root:root `0755` on the root ext4 filesystem.  Before the
2026-07-24 staging, `/opt/ullm` was absent; the completed ancestry now has
the protected root:root `0755` state recorded above.  The policy requires
protected root-owned ancestry rather than a magic pathname; this destination
follows the repository's `/opt/ullm` runtime examples while leaving
`/var/lib/ullm` for the fixed campaign claim/outcome registries.

The publication was an exact byte copy, not a hardlink, reflink, symlink, or
in-place ownership change.  It built a root-owned temporary directory under
the final parent, validated the complete v2 release and runtime tree, and then
used a same-directory no-clobber rename.

Staging this directory closes only the SQ8 worker-release portion.  The final
SQ8 product, tokenizer, promotion pair, candidate manifest, and root-owned
standalone source clone remain separate prerequisites.

## 3. Executed SQ8 root-operation block (archival)

The following SQ8-only commands are the archival 2026-07-24 staging procedure
that was executed.  They are **not** a current action list: the absence
preconditions for `/opt/ullm` and the destination are now intentionally false,
so do not rerun or adapt this block.  It did not use or stop the service and
finished by proving that the AQ4 active-manifest hash did not change.  It did
not authorize or perform an AQ4 root operation.

```bash
set -euo pipefail
export LC_ALL=C

SQ8_RELEASE_SOURCE=/home/homelab1/coding-local/ultimateLLM/uLLM-sq8-v2-final-worker-release-3bc9078d
SQ8_BUILD_SOURCE=/home/homelab1/coding-local/ultimateLLM/uLLM-sq8-v2-final-source-3bc9078d
SQ8_RELEASE_PARENT=/opt/ullm/releases
SQ8_RELEASE_DESTINATION=/opt/ullm/releases/uLLM-sq8-v2-final-worker-release-3bc9078d
SQ8_ACTIVE_BEFORE=$(/usr/bin/sha256sum /etc/ullm/served-models/active.json)
SQ8_ACTIVE_BEFORE=${SQ8_ACTIVE_BEFORE%% *}

test "$SQ8_ACTIVE_BEFORE" = 5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a
test "$(/usr/bin/stat -c '%u:%g:%a' /opt)" = 0:0:755
test ! -e /opt/ullm
test ! -L /opt/ullm
test ! -e "$SQ8_RELEASE_DESTINATION"
test ! -L "$SQ8_RELEASE_DESTINATION"

test "$(/usr/bin/git -C "$SQ8_BUILD_SOURCE" rev-parse HEAD)" = 3bc9078d1ca5a49aad060d667aac19e2aa53ee86
test "$(/usr/bin/git -C "$SQ8_BUILD_SOURCE" rev-parse 'HEAD^{tree}')" = bd95c4f65168b05f4ed572a7f89e35be23ede975
test -z "$(/usr/bin/git -C "$SQ8_BUILD_SOURCE" status --porcelain=v1 --untracked-files=all)"
test -z "$(/usr/bin/git -C "$SQ8_BUILD_SOURCE" symbolic-ref -q HEAD || true)"

test "$(/usr/bin/sha256sum "$SQ8_RELEASE_SOURCE/README.md" | /usr/bin/cut -d' ' -f1)" = b3e2157a02105d1ff7e8771ee0b51bea76d128f1e4ec5e086a8e369d817cf07b
test "$(/usr/bin/sha256sum "$SQ8_RELEASE_SOURCE/SHA256SUMS" | /usr/bin/cut -d' ' -f1)" = ded0a829ef8ab67a19883b454621131a63ff036afe1181d931a5e39d1cd548c5
test "$(/usr/bin/sha256sum "$SQ8_RELEASE_SOURCE/SEALED.json" | /usr/bin/cut -d' ' -f1)" = e01c18593606e173dc154a584feb68d511e224f418ebf2b700aaf377fd171381
test "$(/usr/bin/sha256sum "$SQ8_RELEASE_SOURCE/build-provenance.json" | /usr/bin/cut -d' ' -f1)" = d4a123210ea9680e115f2af1ea8e2285bf6a5c36a18c5db7b8ec779231a0c19d
test "$(/usr/bin/sha256sum "$SQ8_RELEASE_SOURCE/build-receipt.json" | /usr/bin/cut -d' ' -f1)" = 986708497df09d4d7998f79c0e5fe29a0a69c8c37aa7ed2e28643c16faf69cd3
test "$(/usr/bin/sha256sum "$SQ8_RELEASE_SOURCE/ullm-sq8-worker" | /usr/bin/cut -d' ' -f1)" = 0b9989c26e656123addef15ffbf96b1aadf866a6eca06f02af8cab9bccb18a83

(
  cd "$SQ8_RELEASE_SOURCE"
  /usr/bin/sha256sum -c SHA256SUMS
)

sudo -- /usr/bin/mkdir --mode=0755 /opt/ullm
sudo -- /usr/bin/mkdir --mode=0755 "$SQ8_RELEASE_PARENT"

test "$(/usr/bin/stat -c '%u:%g:%a' /opt/ullm)" = 0:0:755
test "$(/usr/bin/stat -c '%u:%g:%a' "$SQ8_RELEASE_PARENT")" = 0:0:755
if /usr/bin/getfacl -R -s -p /opt/ullm | /usr/bin/grep -q .; then
  echo "unexpected ACL below /opt/ullm" >&2
  exit 1
fi

SQ8_RELEASE_STAGE=$(sudo -- /usr/bin/mktemp \
  --directory \
  --tmpdir="$SQ8_RELEASE_PARENT" \
  '.uLLM-sq8-v2-final-worker-release-3bc9078d.stage.XXXXXXXX')
case "$SQ8_RELEASE_STAGE" in
  /opt/ullm/releases/.uLLM-sq8-v2-final-worker-release-3bc9078d.stage.*) ;;
  *) echo "unexpected SQ8 stage path" >&2; exit 1 ;;
esac

for SQ8_MEMBER in README.md SHA256SUMS SEALED.json build-provenance.json build-receipt.json; do
  sudo -- /usr/bin/install \
    --owner=root \
    --group=root \
    --mode=0444 \
    -- "$SQ8_RELEASE_SOURCE/$SQ8_MEMBER" "$SQ8_RELEASE_STAGE/$SQ8_MEMBER"
done
sudo -- /usr/bin/install \
  --owner=root \
  --group=root \
  --mode=0555 \
  -- "$SQ8_RELEASE_SOURCE/ullm-sq8-worker" "$SQ8_RELEASE_STAGE/ullm-sq8-worker"
sudo -- /usr/bin/chmod 0555 -- "$SQ8_RELEASE_STAGE"

test "$(/usr/bin/find "$SQ8_RELEASE_STAGE" -mindepth 1 -maxdepth 1 -type f -printf '.' | /usr/bin/wc -c)" -eq 6
test -z "$(/usr/bin/find "$SQ8_RELEASE_STAGE" -mindepth 1 -maxdepth 1 ! -type f -print)"
test "$(/usr/bin/stat -c '%u:%g:%a' "$SQ8_RELEASE_STAGE")" = 0:0:555

for SQ8_MEMBER in README.md SHA256SUMS SEALED.json build-provenance.json build-receipt.json; do
  test "$(/usr/bin/stat -c '%u:%g:%a:%h' "$SQ8_RELEASE_STAGE/$SQ8_MEMBER")" = 0:0:444:1
done
test "$(/usr/bin/stat -c '%u:%g:%a:%h' "$SQ8_RELEASE_STAGE/ullm-sq8-worker")" = 0:0:555:1

if /usr/bin/getfacl -R -s -p "$SQ8_RELEASE_STAGE" | /usr/bin/grep -q .; then
  echo "unexpected ACL in SQ8 stage" >&2
  exit 1
fi
if /usr/sbin/getcap -r "$SQ8_RELEASE_STAGE" | /usr/bin/grep -q .; then
  echo "unexpected file capability in SQ8 stage" >&2
  exit 1
fi

(
  cd "$SQ8_RELEASE_STAGE"
  /usr/bin/sha256sum -c SHA256SUMS
)

/usr/bin/python3.12 -I -S -B -c '
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import served_model_campaign_runtime_seal as runtime_seal
import sq8_serving_promotion as promotion
release = promotion.validate_build_release(
    Path(sys.argv[2]),
    verify_live_source=True,
    source_root=Path(sys.argv[3]),
)
tree = runtime_seal.capture_runtime_tree_seal(
    Path(sys.argv[2]),
    label="staged SQ8 worker release",
    required_uid=0,
)
print(json.dumps({
    "runtime_tree_sha256": tree.fingerprint_sha256,
    "source_commit": release["receipt"]["source"]["commit"],
    "worker_sha256": release["worker_sha256"],
}, sort_keys=True))
' "$SQ8_BUILD_SOURCE/tools" "$SQ8_RELEASE_STAGE" "$SQ8_BUILD_SOURCE"

sudo -- /usr/bin/sync -f "$SQ8_RELEASE_STAGE"
sudo -- /usr/bin/mv \
  --no-clobber \
  --no-target-directory \
  -- "$SQ8_RELEASE_STAGE" "$SQ8_RELEASE_DESTINATION"
sudo -- /usr/bin/sync -f "$SQ8_RELEASE_PARENT"

test ! -e "$SQ8_RELEASE_STAGE"
test -d "$SQ8_RELEASE_DESTINATION"
test ! -L "$SQ8_RELEASE_DESTINATION"

for SQ8_MEMBER in README.md SHA256SUMS SEALED.json build-provenance.json build-receipt.json ullm-sq8-worker; do
  /usr/bin/cmp -s -- "$SQ8_RELEASE_SOURCE/$SQ8_MEMBER" "$SQ8_RELEASE_DESTINATION/$SQ8_MEMBER"
done

/usr/bin/python3.12 -I -S -B -c '
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import served_model_campaign_runtime_seal as runtime_seal
import sq8_serving_promotion as promotion
release = promotion.validate_build_release(
    Path(sys.argv[2]),
    verify_live_source=True,
    source_root=Path(sys.argv[3]),
)
tree = runtime_seal.capture_runtime_tree_seal(
    Path(sys.argv[2]),
    label="published SQ8 worker release",
    required_uid=0,
)
print(json.dumps({
    "runtime_tree_sha256": tree.fingerprint_sha256,
    "source_commit": release["receipt"]["source"]["commit"],
    "worker_sha256": release["worker_sha256"],
}, sort_keys=True))
' "$SQ8_BUILD_SOURCE/tools" "$SQ8_RELEASE_DESTINATION" "$SQ8_BUILD_SOURCE"

SQ8_ACTIVE_AFTER=$(/usr/bin/sha256sum /etc/ullm/served-models/active.json)
SQ8_ACTIVE_AFTER=${SQ8_ACTIVE_AFTER%% *}
test "$SQ8_ACTIVE_AFTER" = "$SQ8_ACTIVE_BEFORE"
```

Expected validator identity:

```json
{
  "source_commit": "3bc9078d1ca5a49aad060d667aac19e2aa53ee86",
  "worker_sha256": "0b9989c26e656123addef15ffbf96b1aadf866a6eca06f02af8cab9bccb18a83"
}
```

The runtime-tree fingerprint is intentionally observed from the new inodes;
it is not precomputed from the user-owned source release.

## 4. Failure handling

This staging has no active/service rollback because it changes neither.

- Before the final rename, retain the exact root-owned stage for inspection
  if any check fails.  Do not broadly remove `/opt/ullm`.
- `mv --no-clobber` can report success while declining to overwrite on some
  implementations; the immediate `test ! -e "$SQ8_RELEASE_STAGE"` makes that
  case fail closed.
- If post-publication validation fails, stop before generating a profile,
  promotion evidence, or manifest.  Review the exact path and quarantine only
  that directory under a new no-replace name if removal is later authorized.
- Once a profile, promotion artifact, or manifest names the final absolute
  path, never relocate, overwrite, or delete it.

Do not use `cp -a`, `cp -l`, `--reflink`, a symlink, or `chown` on the
preserved source release.

## 5. Remaining root-only checks, not staging mutations

The successful differing-worker bootstrap sidecar is not readable by the
normal operator account.  Claude should perform this separate read-only
confirmation:

```bash
AQ4_BOOTSTRAP_ROOT=/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-fidelity-promotion-f1a3cf4c-v0.1
for AQ4_ATTEMPT in 1 3 4; do
  sudo -- /usr/bin/stat -Lc '%A %a %U:%G %h %s %n' \
    "$AQ4_BOOTSTRAP_ROOT/bootstrap-backup-temp-activation-v$AQ4_ATTEMPT.json.authorization.json"
  sudo -- /usr/bin/sha256sum \
    "$AQ4_BOOTSTRAP_ROOT/bootstrap-backup-temp-activation-v$AQ4_ATTEMPT.json.authorization.json"
  sudo -- /usr/bin/jq -cS . \
    "$AQ4_BOOTSTRAP_ROOT/bootstrap-backup-temp-activation-v$AQ4_ATTEMPT.json.authorization.json"
done
```

Expected old/candidate manifest hashes are `feb3190d...` and `5d015a01...`;
expected old/candidate worker hashes are `177f3106...` and `1f93f215...`.
This confirms sidecar fields only.  It does not make those sidecars immutable
or prove a successful normal activation.

No AQ4 copy/chown command is authorized by this runbook.  The minimal
product/tokenizer layout and the fresh promotion-evidence destination must be
fixed in the separately reviewed AQ4-to-AQ4 hardening plan before root copies
are made.
