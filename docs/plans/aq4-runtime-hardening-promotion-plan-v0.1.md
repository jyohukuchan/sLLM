# AQ4_0 runtime hardening promotion plan v0.1

Status: Phase 1–5 preparation was completed. One Phase 6 activation attempt on 2026-07-26 failed at candidate live proof before the candidate gateway reached readiness; the exact original active-manifest bytes were restored, but the route did not complete rollback live proof and therefore did not declare a healthy rollback. Direct diagnosis later proved that the candidate worker and its minimal closures load correctly. The current service is observed active/running on the original manifest, but this consumed plan is not retry authority. No further activation, authorization consumption, or replacement of <code>/etc/ullm/served-models/active.json</code> is authorized by this document.

## Goal

Promote the currently live AQ4_0 runtime from its user-owned, path-contaminated closure into a new, root-owned runtime closure under <code>/opt/ullm</code>, without changing the AQ4 worker bits, model behavior, generation contract, or performance configuration.

The promoted runtime must have:

- a bit-identical worker at a protected path;
- a purpose-minimized product and tokenizer closure whose complete declared trees pass the runtime seal;
- a standalone, root-owned source clone at the exact promotion commit;
- newly collected promotion evidence and receipt bound to the new protected paths;
- a newly generated and frozen manifest that names only protected absolute paths; and
- a dedicated locked AQ4-to-AQ4 activation and rollback route, followed by fresh AQ4 release/browser evidence and bundle v1.

This is intentionally a runtime-hardening promotion, not the P3 performance optimization. The live worker SHA-256, all 30 required worker environment flags, public API contract, generation defaults, format, and reasoning contract must remain unchanged. P3 work, including the six WMMA/kernel requirement flags, belongs to the separate dependency and must not be introduced here.

## Success Criteria

1. Before final activation, the candidate worker SHA-256 is exactly <code>1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b</code>, its byte length is 4,223,912, and it is a root-owned single-link executable in the protected AQ4 release.
2. The candidate product root contains only the retained AQ4 package closure: five directories, 1,045 regular files, and no symlinks. Its package manifest remains byte-identical to the current package manifest and has SHA-256 <code>a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad</code>.
3. The candidate tokenizer root contains only the five explicitly listed tokenizer/template files; all are root-owned, single-link regular files, and the four manifest-declared hashes plus the chat-template hash match the live contract.
4. Every ancestor from <code>/opt</code> through each protected AQ4 leaf is owned by root, is not group/world writable, has no disallowed ACL/capability, and all runtime-sealed trees contain no symlink, special file, or ordinary hardlink.
5. The AQ4 promotion source is a clean detached standalone clone at <code>0cd760568e197e1adb4c4df3d6149591a912f709</code>; it is not a linked worktree and has no alternates, symlinks, ACLs, hardlinks, or group/world-writable content.
6. Promotion evidence and receipt are newly collected from the protected worker/product/tokenizer/source paths. No old AQ4 evidence, receipt, or manifest hash is copied, rebased, or cited as candidate evidence.
7. The frozen candidate manifest differs from the current active manifest only in the documented path-bound fields and fresh receipt hash. It contains no <code>/home/</code> path.
8. A reviewed AQ4-to-AQ4 locked activation route verifies runtime and source seals, preserves exact rollback bytes, atomically swaps only after explicit human approval, and produces durable candidate or rollback live proof.
9. The current active manifest bytes, SHA-256, systemd unit, environment file, SQ8 assets, <code>llama-qwen35-udq4.service</code>, and <code>gdm3</code> are not modified as part of planning or preparation. Final activation remains the sole operation that may replace active-manifest bytes.
10. After a successful hardening activation, a fresh AQ4 generic release campaign, browser campaign, and complete bundle v1 are collected from the hardened live runtime before any later workflow relies on them.

## Non-Goals

- Do not rebuild, optimize, patch, or otherwise alter <code>ullm-aq4-worker</code>.
- Do not introduce P3 performance settings, including the six AQ4/WMMA/paged-GQA requirements present only in the workspace P3 profile.
- Do not change the AQ4 model ID, public API shape, model format, reasoning configuration, generation defaults, GPU identity contract, or the existing 30 live required worker environment flags.
- Do not claim that old AQ4 promotion evidence, receipt, browser evidence, release evidence, or bundle can validate the protected closure. Their absolute paths and manifest hash bind them to the old closure.
- Do not place AQ4 material in <code>/opt/ullm/releases</code>; that namespace is already the SQ8 release namespace.
- Do not modify, restart, enable, or otherwise operate <code>llama-qwen35-udq4.service</code>; it remains disabled and inactive. Do not start <code>gdm3</code>; it remains inactive.
- Do not treat the historical active-manifest SHA-256 <code>feb3190d...</code> or historical worker SHA-256 <code>177f3106...</code> as a rollback target for this promotion.
- Do not activate a candidate merely because it passes a file seal. Activation requires the dedicated locked control route, a service window, and an explicit human approval immediately before the active-manifest byte swap.

## Confirmed Inputs and Fixed Invariants

The following values were rechecked from the live machine and are fixed inputs to this plan.

| Item | Confirmed value | Planning consequence |
| --- | --- | --- |
| Current active manifest | <code>/etc/ullm/served-models/active.json</code>, SHA-256 <code>5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a</code>, root:root 0644 | Capture these exact bytes immediately before an eventual activation. They are the only rollback bytes for this operation. |
| AQ4 promotion source commit | <code>0cd760568e197e1adb4c4df3d6149591a912f709</code> | The source clone and fresh receipt must record exactly this commit. |
| Live worker | <code>/home/homelab1/coding-local/ultimateLLM/uLLM-aq4-fidelity-promotion-release-f1a3cf4c/ullm-aq4-worker</code>, SHA-256 <code>1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b</code> | Copy byte-for-byte; do not rebuild. |
| Live legacy engine | Same release directory, <code>ullm-engine</code>, SHA-256 <code>d1c18362c6253294d37e7258434d877752c5052ab677ecfd35f1a7928b64b433</code> | Retain only for fresh resident-versus-legacy promotion evidence. It is not a served-manifest field. |
| Current promotion receipt | Old product-root path, SHA-256 <code>1b36fc880bf1510185eaad7887c9aed33f69df223036271e4bfba4bb43f16e8b</code> | Historical reference only; do not copy or reuse. |
| Current product root | <code>/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1</code> | It is not a viable sealed root and must not be named by the new manifest. |
| Current tokenizer root | <code>/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B</code> | Copy only the verified tokenizer/template closure, not model data or cache. |
| Systemd unit | <code>/etc/systemd/system/ullm-openai.service</code>, SHA-256 <code>f0239713b16b3bf31cfd12a98f506e77e55af9b31abf58352f4e437e1cdee552</code> | Recheck in activation preflight and while locked; drift aborts the operation. |
| Gateway environment file | <code>/etc/ullm/openai-gateway-manifest.env</code>, SHA-256 <code>68dd3a027fa86aaa8f5649bf55f34c32b818afb49a9e35e272f5dc6a1e5fb835</code> | Recheck in activation preflight and while locked; drift aborts the operation. |

The worker was originally built at <code>f1a3cf4c86978b3b8900396a0b6a8caff90b97f1</code>, while promotion provenance is the fixed commit above. This plan preserves that distinction: build provenance explains the existing bit-identical worker; it must not replace <code>promotion.source_commit</code>.

The current active manifest has 30 required worker environment flags. The workspace AQ4 profile contains six additional P3 flags and therefore must not be used as the hardening input profile. The future candidate profile is mechanically derived from the live manifest instead.

## Protected Ancestry and Layout

The AQ4 hardening root is deliberately separate from SQ8's existing <code>/opt/ullm/releases</code> namespace.

    /opt/ullm/aq4-runtime-hardening-v0.1/
      .staging/
      releases/aq4-fidelity-f1a3cf4c/
        ullm-aq4-worker
        ullm-engine
      products/qwen35-9b-aq4-package-a790a033f57d/
        package/
          manifest.json
          codebooks/
          tensors/
          passthrough/
      tokenizers/qwen35-9b-qwen2tokenizer-a4aee8afcf2e/
        merges.txt
        tokenizer.json
        tokenizer_config.json
        vocab.json
        chat_template.jinja
      sources/aq4-promotion-0cd760568e197/
      control-source/manifest-freezer-f71bb2e534b/
      control-source/aq4-hardening-activation-d11085c4e119/
      promotion/
        aq4-fidelity-profile.json
        promotion-evidence.json
        promotion-evidence-protected-path-binding.json
        promotion-receipt.json
      manifests/
        aq4-hardened-draft.json
        aq4-hardened-frozen.json
      activation/
        rollback-active-5d015a013dcf70ce.json
        reviewed-operations.json
        operation-bin/
        operation-source/
        activation-plan.json
        activation-intent.json
        outcome.json
        proofs/
      campaigns/
        release-bundle-v1/

The dedicated control-source commit is fixed to <code>d11085c4e119361cf0dca78e6cbe81cafcb9af6b</code>. Its protected clone at the 12-character-prefix path above is sealed, detached, clean, and bound by the plan to this full commit and tree <code>c41bf38138dad3a1091f6c2e45de835b713ef0b3</code>. It remains a separately sealed standalone control-tool source, not the AQ4 promotion source and not a substitute for it.

| Class | Required ownership and mode after finalization | Additional invariant |
| --- | --- | --- |
| <code>/opt</code>, <code>/opt/ullm</code>, and each AQ4 hardening namespace ancestor | root:root, 0755 or stricter non-writable-by-group/other directory mode | Existing SQ8 descendants remain read-only for this work. |
| Final closure directories | root:root 0555 | No ACL, capability, symlink, or special file. |
| Worker and legacy engine | root:root 0555 | One regular-file link each, exact expected SHA-256. |
| Package/tokenizer/source regular data | root:root 0444 | One regular-file link; no ACL/capability. |
| Frozen manifest, receipt, evidence, activation records, and proof documents | root:root 0444 | No-replace publication and link count one. |
| Staging directories | root:root 0700 or 0750 while incomplete | Must be unique, must not overlap a final path, and must never be mistaken for a sealable final closure. |

No AQ4 final path may be a symlink. The source path, control-source paths, product root, tokenizer root, worker release, evidence directory, and frozen manifest directory must all be direct children under the AQ4 root shown above. SQ8 source lineage remains under its existing SQ8-specific paths and is not read as input except for read-only environmental comparison.

## Product and Tokenizer Closure

### Minimal product closure

The current product root has nine directories, 1,167 regular files, and one <code>artifact -&gt; package</code> symlink. Its historical files and <code>artifacts/</code> directory are in scope whenever the old root is declared, which is why merely copying the worker or <code>package/</code> beneath the old root cannot pass the runtime seal.

The new product root contains exactly the following retained closure:

| Retain | File count | Logical byte count | Reason |
| --- | ---: | ---: | --- |
| <code>package/manifest.json</code> | 1 | 687,595 | The served manifest binds this package manifest's SHA-256. |
| <code>package/codebooks/</code> | 13 | 832 | Payload required by the package manifest. |
| <code>package/tensors/</code> | 512 | 4,684,644,352 | Quantized tensor index/scale payloads required by the package manifest. |
| <code>package/passthrough/</code> | 519 | 3,015,539,680 | Raw passthrough payloads required by the package manifest. |
| Total | 1,045 | 7,700,872,459 | Complete AQ4 package closure. |

The retained destination tree is five directories: the product root plus <code>package</code>, <code>codebooks</code>, <code>tensors</code>, and <code>passthrough</code>. It has zero symlinks. The old <code>artifact -&gt; package</code> link is intentionally not recreated: the current served manifest declares <code>product.artifact: null</code>, and its package manifest path is <code>package/manifest.json</code>. Nothing in the retained runtime contract needs that alias.

Discard from the destination product root:

- the entire old <code>artifacts/</code> directory;
- every historical promotion receipt/evidence JSON file;
- historical active/candidate manifest copies and all other historical top-level sidecars; and
- the <code>artifact</code> symlink itself.

Do not edit the retained package manifest to make paths look new. Package payload locations are relative to the package root, and the runtime loader resolves them under the declared product/package closure. Changing that manifest would change the bound package SHA-256 and would cease to be a pure relocation.

The source package's apparent allocation is lower because sparse content is present, but capacity admission must use its 7,700,872,459 logical bytes. Before any copy, require at least 17,179,869,184 bytes (16 GiB) available on the target filesystem after checking <code>stat -f</code>; this reserves space for a staged package, release, tokenizer, source clone, and verification overhead. The observed availability during planning was approximately 2.78 TB, but it must be checked again at execution time.

Copy method requirements:

1. Refuse the source if it has a symlink, special file, or non-single-link regular file inside the selected package closure.
2. Create a unique root-owned staging directory under the destination parent, never under the final product root.
3. Use an explicit recursive copy such as <code>rsync --recursive --times --no-perms --no-owner --no-group</code> from the selected <code>package/</code> only. Do not use <code>cp -a</code>, <code>rsync -a</code>, <code>-H</code>, <code>-l</code>, <code>--copy-links</code>, <code>--reflink</code>, a filesystem snapshot, or any hardlink-preserving/reflink-preserving method.
4. Set final ownership and modes only after content verification; then run a checksum dry-run, count verification, manifest <code>cmp</code>, hash verification, no-symlink/no-hardlink scan, ACL/capability scan, and full runtime-tree seal.
5. Flush the completed stage, publish by a no-clobber rename into the final path, and never auto-delete an interrupted stage. An interrupted stage is quarantined for diagnosis and must not be sealed or reused without a fresh verification pass.

### Tokenizer closure

The old tokenizer root contains 34 regular files and four directories, including source model shards and a Hugging Face cache. The serving contract needs only the tokenizer implementation assets and template:

| Retain | Live SHA-256 / binding | Why retain |
| --- | --- | --- |
| <code>merges.txt</code> | Manifest-declared hash | Required tokenizer vocabulary merge input. |
| <code>tokenizer.json</code> | Manifest-declared hash | Required tokenizer serialization. |
| <code>tokenizer_config.json</code> | Manifest-declared hash | Required tokenizer configuration. |
| <code>vocab.json</code> | Manifest-declared hash | Required tokenizer vocabulary. |
| <code>chat_template.jinja</code> | SHA-256 <code>a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715</code> | The active contract independently binds this template. Retaining it avoids <code>AutoTokenizer.from_pretrained</code> template-precedence ambiguity. |

The first four files total 22,900,710 logical bytes; with <code>chat_template.jinja</code>, the retained tokenizer closure is 22,908,466 logical bytes. The 29 omitted files include the four source <code>model.safetensors</code> shards, model/config/index/preprocessor/video/readme/license metadata, <code>.gitattributes</code>, and the 18 <code>.cache/huggingface</code> entries. No model shard or cache directory belongs in the destination.

The new tokenizer root has exactly these five files, no subdirectory, no symlink, no cache, root:root 0444 leaves, and a root:root 0555 root directory. The candidate manifest continues to list only the four existing <code>tokenizer.files</code> members and their unchanged hashes; it binds the fifth through its unchanged chat-template SHA-256 field.

## Working Hypotheses

1. **Resolved by commit <code>d11085c4e119361cf0dca78e6cbe81cafcb9af6b</code>.** The dedicated route is <code>tools/aq4_runtime_hardening_activation.py</code> with prepare/run/rollback wrappers and schema <code>ullm.aq4_runtime_hardening_activation_plan.v1</code>. It is not an SQ8 wrapper or import path. A later Phase 3-style root-sealed standalone clone at <code>control-source/aq4-hardening-activation-d11085c4e119/</code> is still required before it can prepare a production plan.
2. The candidate profile can be generated mechanically from the live active manifest without changing behavior. It must copy <code>public</code>, <code>generation</code>, <code>format</code>, <code>reasoning</code>, tokenizer metadata/template options and four file names, worker protocol/arguments/identity and all 30 environment flags, and product package metadata. It changes only its path-bound fields before receipt generation.
3. The current promotion-evidence and receipt tools can run as root from the protected standalone promotion source, writing root-owned output, because their content checks are already AQ4 resident-versus-legacy checks. They must be run with <code>python3 -B</code> so no bytecode is written into the sealed source clone.
4. The existing manifest freezer is not present at the AQ4 promotion commit. It must run from a distinct, independently sealed control-source clone at <code>f71bb2e534b12bbf0ab37e716da1090c485ab733</code>, where <code>tools/freeze-served-model-manifest.py</code> exists. That control clone does not change the promotion source commit recorded in the receipt.
5. **Resolved by commit <code>d11085c4e119361cf0dca78e6cbe81cafcb9af6b</code>.** The bundle v1 preparer now uses owner-bound <code>renameat2(RENAME_NOREPLACE)</code> publication, file/parent <code>fsync</code>, root CLI default <code>--required-uid 0</code>, and mode <code>0444</code>/nlink-one postchecks. The validator has <code>--require-immutable-publication --required-uid 0</code> for the mandatory post-publication re-read. This does not make a future bundle valid before Phase 7 supplies fresh hardened-runtime inputs.
6. The current active AQ4 runtime can be used as an exact emergency rollback byte target only while its old paths continue to pass the legacy operational checks. Its old closure is known not to be root-sealed; therefore a restored byte match alone is insufficient to claim a healthy rollback. Live proof and legacy asset/hash availability are required.

Each hypothesis is a deliberate gate, not permission to improvise. Resolve it through the Next Actions before beginning the corresponding execution phase.

## Phase Breakdown

### Phase 0 — Establish a non-mutating admission record

Purpose: record the execution-time inputs before allocating any protected path.

1. Read and hash the active manifest. Require its bytes and SHA-256 to still equal <code>5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a</code> at planning admission. If it has legitimately changed, stop and write a superseding plan; do not silently adopt it.
2. Read/hash the service unit and gateway environment file. Require the two fixed SHA-256 values in Confirmed Inputs.
3. Record <code>systemctl</code> state for <code>ullm-openai.service</code>, <code>llama-qwen35-udq4.service</code>, and <code>gdm3</code>. The latter two must remain disabled/inactive and inactive, respectively.
4. Rehash the old worker, legacy engine, old package manifest, four tokenizer members, and <code>chat_template.jinja</code>. Record byte sizes, owner/mode/link count, mount/device information, and all planned target paths.
5. Read the complete selected source/product/tokenizer trees to confirm the counts in this plan. This is inspection only; do not fix their modes or ownership.

Exit criterion: an immutable root-owned admission record has the current values and explicitly says that no copy, evidence collection, freeze, or activation has occurred. Any input drift returns to plan review, not a copy retry.

GPU or service window: neither.

### Phase 1 — Create and verify protected ancestry

Purpose: establish only empty, protected AQ4 namespace parents, separate from SQ8.

1. Verify <code>/opt</code> and <code>/opt/ullm</code> ancestry is root-owned and not group/world writable. Read <code>/opt/ullm/releases</code> only to confirm it remains untouched.
2. Check free capacity using the destination filesystem, enforcing the 16 GiB admission threshold before any staging directory is created.
3. Create the AQ4-specific parent <code>/opt/ullm/aq4-runtime-hardening-v0.1</code> and the direct category parents from the fixed layout. Do not create or modify anything below <code>/opt/ullm/releases</code>.
4. Verify every new ancestor is root:root, not group/world writable, not a symlink, and has no unexpected ACL/capability. Record device/inode data to ensure later stages did not cross filesystem boundaries unexpectedly.
5. Reserve a unique root-owned staging name for each future leaf. Refuse a preexisting final path; never overwrite it.

Exit criterion: only empty namespace parents/staging reservations exist; no candidate runtime content, evidence, or manifest has been published.

GPU or service window: neither.

### Phase 2 — Build minimal root-owned runtime closures

Purpose: copy the exact runtime inputs without copying their historical ancestry.

For all three closures, the copy primitive must create ordinary independent destination files: no hardlink, no reflink, no symlink traversal, and no archive-preservation shortcut. In particular, do not use <code>cp -a</code>, <code>rsync -a</code>, <code>-H</code>, <code>-l</code>, <code>--copy-links</code>, or <code>--reflink</code> for worker, engine, product, or tokenizer content.

#### 2A. Worker release

Copy exactly two regular files from the existing release into:

    /opt/ullm/aq4-runtime-hardening-v0.1/releases/aq4-fidelity-f1a3cf4c/

The destination contains only <code>ullm-aq4-worker</code> and <code>ullm-engine</code>. Require byte-for-byte checks against the two hashes in Confirmed Inputs, size checks, root:root 0555, nlink 1, no ACL/capability, and no symlink. Do not retain build logs, parent worktree files, or an old release directory wrapper.

#### 2B. Product

Perform exactly the selected <code>package/</code> copy described in Product and Tokenizer Closure, then publish it as:

    /opt/ullm/aq4-runtime-hardening-v0.1/products/qwen35-9b-aq4-package-a790a033f57d/

Require five directories, 1,045 regular files, zero symlinks, root:root final modes, the unchanged package manifest SHA-256, and the exact 7,700,872,459 logical-byte total. Run the full recursive runtime product-tree seal against this new root, not merely against <code>package/manifest.json</code>.

#### 2C. Tokenizer

Copy only the five files in the tokenizer table into:

    /opt/ullm/aq4-runtime-hardening-v0.1/tokenizers/qwen35-9b-qwen2tokenizer-a4aee8afcf2e/

Require five regular files, no subdirectories/symlinks, root:root 0444, nlink 1, no ACL/capability, the unchanged four declared file hashes, and the unchanged template hash. Run the full recursive runtime tokenizer-tree seal against this new root.

Exit criterion: all three closures are separately sealed and publish records establish no-hardlink/no-reflink provenance. A failure leaves its unique stage unpromoted for inspection; it never replaces a final path and does not trigger broad deletion.

GPU or service window: neither.

### Phase 3 — Create standalone promotion and control source clones

Purpose: replace the linked AQ4 worktree with a sealed standalone source and prepare only the separately versioned tools needed later.

The source input currently has a <code>.git</code> pointer file into a parent worktree and ignored hardlinked build output. It is unsuitable as a sealed source. The exact inspected input is:

    /home/homelab1/coding-local/ultimateLLM/uLLM-aq4-fidelity-promotion-source-f1a3cf4c

It was detached at the required commit and clean when inspected, but that must be rechecked in Phase 0. Clone it into a unique root-owned staging directory below <code>/opt/ullm/aq4-runtime-hardening-v0.1/.staging/sources/</code>, using the following operational shape:

    sudo -- /usr/bin/git clone --no-hardlinks --no-checkout -- "$AQ4_SOURCE_INPUT" "$AQ4_SOURCE_STAGE"
    sudo -- /usr/bin/git -C "$AQ4_SOURCE_STAGE" checkout --detach 0cd760568e197e1adb4c4df3d6149591a912f709

Do not use <code>git worktree add</code>, a file-tree copy, <code>--shared</code>, <code>--reference</code>, an object alternate, or a network-only clone that cannot prove the chosen source input. Then set final ownership/modes and publish it at:

    /opt/ullm/aq4-runtime-hardening-v0.1/sources/aq4-promotion-0cd760568e197/

Reject the promotion source if any of the following is true:

- <code>.git</code> is a regular pointer file instead of a directory;
- <code>git rev-parse HEAD</code> is not the exact commit, the clone is not detached, its tree is not <code>bd372761d7e79b3d5db2b533cdd3fdfa77f125c2</code>, or <code>git status --porcelain</code> is nonempty;
- <code>.git/objects/info/alternates</code> exists or Git reports a shared/alternate object arrangement;
- any source entry is a symlink, special file, ordinary hardlink, has a POSIX ACL/capability, or is group/world writable; or
- any ancestor under the AQ4 root fails protected-ancestry checks.

Create the manifest-freezer control source by the same standalone-clone and sealing method at:

    /opt/ullm/aq4-runtime-hardening-v0.1/control-source/manifest-freezer-f71bb2e534b/

It must be detached at <code>f71bb2e534b12bbf0ab37e716da1090c485ab733</code> and contain <code>tools/freeze-served-model-manifest.py</code>. The activation-control source has the same seal requirements and is pinned to <code>d11085c4e119361cf0dca78e6cbe81cafcb9af6b</code>; its final protected clone is now sealed at the layout path above.

Exit criterion: source seals pass for the promotion source and manifest-freezer control source. The promotion source commit remains <code>0cd76056...</code> everywhere that records AQ4 promotion provenance.

GPU or service window: neither.

### Phase 4 — Generate fresh path-bound promotion evidence, receipt, and frozen candidate manifest

Purpose: create a candidate whose only semantic difference from live AQ4 is the protected closure and its newly bound evidence/receipt.

#### 4A. Build the candidate profile from the live manifest

Construct a profile from the execution-time active manifest rather than from the workspace AQ4 profile. The profile must preserve:

- <code>public</code>, <code>generation</code>, <code>format</code>, and <code>reasoning</code> exactly;
- tokenizer transformer version, class, four declared file names, and template options;
- worker protocol, arguments, identity, and exactly the 30 existing required environment entries;
- product <code>artifact: null</code>, package manifest path <code>package/manifest.json</code>, and package manifest binding; and
- promotion source commit <code>0cd760568e197e1adb4c4df3d6149591a912f709</code>.

Before receipt creation, change only the following profile fields:

| Field | New value |
| --- | --- |
| <code>tokenizer.root</code> | <code>/opt/ullm/aq4-runtime-hardening-v0.1/tokenizers/qwen35-9b-qwen2tokenizer-a4aee8afcf2e</code> |
| <code>worker.binary</code> | <code>/opt/ullm/aq4-runtime-hardening-v0.1/releases/aq4-fidelity-f1a3cf4c/ullm-aq4-worker</code> |
| <code>product.root</code> | <code>/opt/ullm/aq4-runtime-hardening-v0.1/products/qwen35-9b-aq4-package-a790a033f57d</code> |
| <code>promotion.receipt</code> | <code>/opt/ullm/aq4-runtime-hardening-v0.1/promotion/promotion-receipt.json</code> |

The profile does not carry a precomputed worker hash or receipt hash; the generator derives/binds them. Diff the profile's preserved contract against the active manifest before evidence execution. Reject any of these P3-only environment keys, any other unexpected environment key, or any changed preserved field:

- <code>ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_GROUP8_KERNEL</code>
- <code>ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_KERNEL</code>
- <code>ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_KERNEL</code>
- <code>ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_RAGGED_M_KERNEL</code>
- <code>ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_RAGGED_M_KERNEL</code>
- <code>ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_WMMA_KERNEL</code>

#### 4B. Collect fresh promotion evidence

Run the exact AQ4 tools from these absolute paths in the sealed promotion source:

    /opt/ullm/aq4-runtime-hardening-v0.1/sources/aq4-promotion-0cd760568e197/tools/run-aq4-resident-promotion-evidence.py
    /opt/ullm/aq4-runtime-hardening-v0.1/sources/aq4-promotion-0cd760568e197/tools/write-aq4-resident-promotion-receipt.py
    /opt/ullm/aq4-runtime-hardening-v0.1/sources/aq4-promotion-0cd760568e197/tools/generate-served-model.py
    /opt/ullm/aq4-runtime-hardening-v0.1/sources/aq4-promotion-0cd760568e197/tools/validate-served-model.py

Invoke Python as <code>/usr/bin/python3 -B</code>. Run the evidence tool as root so output owned under the protected AQ4 root is root-owned. The evidence run must name the new protected worker, legacy engine, product, tokenizer, and promotion source paths in its output. It must use GPU index 1 and prove the resident worker against the retained legacy engine; it must not use a rebuilt worker.

This is the first GPU/service-maintenance phase:

1. Confirm no process owns VRAM on R9700/GPU index 1 using the approved ROCm inspection.
2. Stop only <code>ullm-openai.service</code> for the evidence window. Issue no start/enable command for <code>llama-qwen35-udq4.service</code>; keep it disabled/inactive. Do not operate <code>gdm3</code>.
3. Use a trap/finally path that restarts only <code>ullm-openai.service</code> if the evidence operation aborts.
4. Validate the evidence before receipt writing, then write the receipt in the same protected <code>promotion/</code> directory so its sibling evidence reference is path-valid.
5. Publish evidence and receipt as root:root 0444, nlink 1, no-replace files only after their content validators pass.

#### 4C. Generate, validate, and freeze the manifest

Generate a new v2 candidate manifest from the protected profile and fresh receipt. Validate it with the sealed promotion-source <code>generate-served-model.py</code> and <code>validate-served-model.py</code>, then use the sealed manifest-freezer control source's:

    /opt/ullm/aq4-runtime-hardening-v0.1/control-source/manifest-freezer-f71bb2e534b/tools/freeze-served-model-manifest.py

The freezer receives a staged candidate, its exact expected SHA-256, and an absent final output path. It writes:

    /opt/ullm/aq4-runtime-hardening-v0.1/manifests/aq4-hardened-frozen.json

Require root:root 0444, nlink 1, no ACL/capability, and a successful post-freeze validation. The generated manifest must differ from the current active manifest only in:

1. <code>tokenizer.root</code>;
2. <code>worker.binary</code>;
3. <code>product.root</code>;
4. <code>promotion.receipt</code>; and
5. <code>promotion.receipt_sha256</code>.

Its overall SHA-256 necessarily changes. The following must remain exactly the same: model/public contract, generation, format, reasoning, tokenizer file hashes/template hash/options, worker SHA-256/arguments/all 30 environment entries/identity, <code>product.artifact</code>, package path and package-manifest SHA-256, and promotion source commit. Search the frozen document for <code>/home/</code>; any match is a hard failure.

Exit criterion: fresh evidence, receipt, and frozen candidate exist at protected paths, validate with their intended tools, and have not been activated.

Execution record (2026-07-26 JST): the profile was mechanically derived from live <code>active.json</code>, has exactly 30 unique guard flags in live order, and has no P3-only key. Fresh evidence on R9700 / <code>gfx1201</code> / GPU index <code>1</code> passed both raw exact-token comparisons and clean-shutdown checks. Evidence SHA-256 is <code>4a604453abb6c7a672731d2b17d3333e471d6c5239b4fed1f6b338fe19a19adb</code>; the fresh receipt SHA-256 is <code>99ead62f6d5d6062690d78431dbb888949e100bf8951c55f9ff16c71545f1f24</code>; and protected-path binding SHA-256 is <code>e1b6158cddfab37b84afc2b85351a109d4530af7c4668adb932e5b94532ebe2b</code>. The freezer published <code>manifests/aq4-hardened-frozen.json</code> at SHA-256 <code>c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4</code>. Its only differences from live are the five fields listed above, and it has no <code>/home/</code> reference. No active-manifest bytes were replaced.

GPU or service window: GPU index 1 and an <code>ullm-openai.service</code> maintenance window are required for evidence collection only. Manifest generation/freezing itself needs neither.

### Phase 5 — Implement and review the dedicated AQ4-to-AQ4 locked activation control route

Purpose: close the known gap before any activation attempt. This is operational control tooling, not a worker/model/P3 code change.

Implementation is complete in <code>d11085c4e119361cf0dca78e6cbe81cafcb9af6b</code>; the detailed contract is <code>docs/specs/aq4-runtime-hardening-activation-v0.1.md</code>. The final decisions are:

| Decision | Implemented value |
| --- | --- |
| Plan schema | <code>ullm.aq4_runtime_hardening_activation_plan.v1</code> |
| Control-source pin | clean detached standalone clone at exact commit <code>d11085c4e119361cf0dca78e6cbe81cafcb9af6b</code>; plan binds its tree and all four route-tool hashes |
| Swap primitive | pinned active-parent dirfd, candidate-byte staging inode, <code>renameat2(RENAME_EXCHANGE)</code>, file/parent <code>fsync</code>; candidate frozen file is never renamed |
| Intent/outcome | canonical root-owned <code>0444</code>, nlink-one documents published by <code>renameat2(RENAME_NOREPLACE)</code>; success outcome is the commit boundary |
| Recovery | same lock, literal confirmation and exact plan SHA; failed recovery/rollback attempts go to unique immutable audits without consuming success receipt paths |
| Bundle v1 | owner-bound no-replace v1 publication plus immutable validator mode, as described in Working Hypothesis 5 |

CPU/private-copy/mock fault tests cover pre-intent unit drift and stale plan hash, SIGKILL after intent before swap, SIGKILL after swap, post-rename fault recognition, candidate live-proof failure and exact restore, duplicate execution, failed-recovery retry/audit separation, concurrent lock acquisition, and receipt publication fault after commit. The dedicated test also asserts that the route does not reference the SQ8 final route, <code>llama-qwen35-udq4.service</code>, or <code>gdm3</code>. No GPU was used.

Historical record: the pre-plan report at <code>benchmarks/results/2026-07-26/aq4-runtime-hardening-activation-v0.1/read-only-preplan-control-source-pin.json</code> correctly reported <code>ready: false</code> before Phase 4 created the candidate manifest, immutable rollback copy, plan, reviewed operations, credential seal set, and pinned control-source clone. It is superseded for current readiness by <code>benchmarks/results/2026-07-26/aq4-runtime-hardening-phase4/read-only-preflight.json</code>: the plan-bound default preflight is <code>ready: true</code>, has no blockers, and did not execute activation.

The reviewed route must be stored in the separately sealed activation control source and must provide these capabilities:

1. A root-owned, immutable activation plan schema, such as <code>ullm.aq4_runtime_hardening_activation_plan.v1</code>, that records the promotion source seal/tree/commit, frozen candidate SHA/path, candidate runtime seals, exact legacy active bytes/SHA, systemd/env hashes, executable hashes, proof destinations, and operation epoch.
2. A default non-mutating preflight that re-seals all candidate inputs, confirms no path/inode/hash drift, validates the frozen manifest, and reports <code>ready: true</code> only when it is safe to request approval.
3. A dedicated lock at or equivalent to <code>/etc/ullm/served-models/.active.json.activation.lock</code>. Under that lock, recheck the exact active bytes, activation plan hash, protected candidate paths, unit/env hashes, and source/runtime seals before a mutation.
4. Durable no-replace activation intent written and fsynced before the swap; an atomic compare-and-swap/rename operation for active-manifest bytes; and a durable outcome record. It must never mutate the frozen candidate file.
5. Checked gateway reconciliation and candidate live proof. The proof must bind the activation-plan hash/epoch and record active-manifest exact bytes/hash, model ID, worker path/hash, systemd state, boot ID, PID/PPID/starttime/executable hashes, and all five live endpoints: gateway health, gateway ready, gateway models, OpenWebUI health, and OpenWebUI models.
6. Automatic failure handling under the same lock that restores the exact just-captured pre-activation active bytes, reconciles the gateway, and writes rollback live proof. “Bytes restored” alone is not a successful rollback.
7. A durable recovery action for a crash or incomplete restore, and a later manual rollback action that requires the activation-plan hash, an explicit confirmation string, exact candidate-active bytes, and the saved rollback bytes.
8. A fail-closed legacy-asset check before rollback proof: the old worker/product/tokenizer/receipt must still be reachable and match their recorded legacy hashes. If they do not, record the condition and require incident recovery rather than falsely claim healthy rollback.

The implementation review must include fault injection for pre-intent failure, post-intent/pre-swap failure, post-swap/restart failure, live-proof failure, duplicate execution, stale plan hash, unit/env drift, and concurrent lock acquisition. It must demonstrate that it does not invoke the SQ8 final-activation route and that it never starts <code>llama-qwen35-udq4.service</code> or <code>gdm3</code>.

Exit criterion: the reviewed code, separately sealed control source, complete immutable activation plan, reviewed operations, credential seal set, and plan-bound <code>ready: true</code> preflight now exist. No active-manifest byte was changed in this phase. **Even at <code>ready: true</code>, this is not execution authority: the Phase 6 human approval gate remains mandatory.**

GPU or service window: code review/preflight needs neither. The eventual execute/rollback subcommand needs a service-maintenance window and normal gateway worker startup, but no P3 performance trial.

### Phase 6 — Human-gated locked activation and rollback readiness

Purpose: perform the only operation that may replace <code>active.json</code>, after all artifacts and preflight are complete.

1. Phase 4 preparation has already published the root-owned immutable rollback copy:

       /opt/ullm/aq4-runtime-hardening-v0.1/activation/rollback-active-5d015a013dcf70ce.json

   It equals the then-current active bytes and has SHA-256 <code>5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a</code>. Immediately before execution, re-read <code>active.json</code> and require an exact byte/hash match with this sealed plan input; do not overwrite the published rollback copy. If it differs, stop and prepare a superseding reviewed plan rather than adapting this one in place.
2. Run the dedicated route's non-mutating preflight and inspect its immutable plan, candidate seal report, rollback target, unit/env hashes, and output destinations.
3. Schedule the narrow service-maintenance window. The only routine service action is the gateway operation controlled by the dedicated route. Keep <code>llama-qwen35-udq4.service</code> disabled/inactive and <code>gdm3</code> inactive.
4. **ここで停止する。人間が、表示された activation-plan SHA-256・rollback SHA-256・候補 manifest SHA-256・service window を確認し、明示的に承認するまで、<code>/etc/ullm/served-models/active.json</code> の置換を一切実行しない。**
5. Only after that approval, invoke the dedicated locked execute action with the exact plan SHA-256 and required confirmation string.
6. Accept activation only if the route writes immutable outcome/candidate-proof records and every candidate live-proof check passes. On any failure, require its same-lock rollback procedure and inspect rollback live proof before reopening normal operation.

Exit criterion: either protected AQ4 is active with validated live proof, or the exact pre-activation bytes are restored with validated rollback live proof. No ambiguous state may be handed off.

GPU or service window: required for the locked execute/rollback operation. It is a correctness/reconciliation window, not a P3 benchmark or tuning window.

### Phase 6 execution record — readiness-proof failure (2026-07-26)

This section supersedes the future-tense execution wording above for plan <code>72140ff475b29e28f4ab6685459a344939bc54fcd12aa4f0b7c44cd7a8753194</code>. It is an incident record, not permission to repeat the operation.

- The immutable outcome records <code>status: failed_restore</code>, <code>failure_stage: candidate_live_proof</code>, candidate manifest SHA-256 <code>c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4</code>, and observed/restored active SHA-256 <code>5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a</code>. The subsequent recovery audit records only <code>error: ActivationError</code>.
- The journal establishes that the candidate service was reported started at 10:59:50.004 JST and was stopped at 10:59:50.286 JST: 282 ms later. It never emitted <code>Started server process</code>, <code>Application startup complete</code>, or a candidate <code>worker_fatal</code> event. For comparison, the restored gateway first emitted <code>Started server process</code> 464 ms after systemd reported it started and completed application startup roughly 2.8 seconds later.
- The 11:00:27 JST <code>unexpected worker stdout EOF</code> belongs to the already restored live gateway process and follows systemd's 11:00:27.613 stop request. It is shutdown-induced stdout closure, not a candidate model-load crash. The earlier 10:59:49 EOF likewise belongs to the pre-existing gateway being stopped for the activation attempt.
- Direct worker controls as user <code>homelab1</code>, in the gateway working directory with the service environment and all 30 <code>ULLM_REQUIRE_HIP_*</code> guards, loaded both candidate and live manifests. Both wrote the same ready record; their full stdout and 192-line / 223,428-byte stderr files are byte-identical. The candidate stderr contains only <code>ullm.backend_operation.load.v1</code> traces, with no error/fatal/panic text. Both workers stayed alive until the deliberate 120-second test timeout.
- The candidate product has all 1,045 source-package members, and all 1,044 package-manifest payload references resolve within it. The 122 omitted source-product files are history/SQ8 sidecars outside <code>package/</code>. The five tokenizer files load through the actual gateway tokenizer contract. The absent <code>artifact -&gt; package</code> symlink is correct because AQ4 manifests declare <code>product.artifact: null</code> and the AQ4 worker rejects an artifact directory.

Root cause: the sealed route's reconciliation operation returns as soon as systemd says <code>active/running</code>. Its immediately following observation performs one un-retried pass over gateway health/ready/models and OpenWebUI health/models. It ran before the candidate gateway had started its server process, so the live observation failed and the route stopped the candidate to restore the original manifest. This is an activation-control readiness race, not a worker, permission, closure, tokenizer, symlink, path-length, or mount-boundary failure.

The audit cannot identify which individual endpoint call failed: the operation catches all inner exceptions and prints only a generic message, while the activation wrapper records an <code>ActivationError</code> class name but discards the child stderr/cause. The outcome's <code>candidate_live_proof: not_run</code> stage value does not contradict <code>failure_stage: candidate_live_proof</code>; exception cleanup converts an unfinished <code>pending</code> stage to <code>not_run</code> before publishing the outcome.

The required correction is implemented by the corrective-preparation record below. The protected candidate closure and its manifest were not rebuilt or changed. This consumed plan must not be retried.

Full diagnosis evidence is stored in <code>benchmarks/results/2026-07-26/aq4-activation-failure-diagnosis/</code>.

### Phase 6 corrective preparation record — readiness v0.2 (2026-07-26)

This is preparation only, not activation authority. It creates a fresh source seal and a fresh immutable plan while preserving the consumed v0.1 plan, its outcome, and its audits unchanged.

- The final sealed control source is detached commit <code>af7298bad50cfc7b8166c5505aaaffe0e9ad465f</code>, tree <code>d6738b269d30605f1f36edb8fc06b4d698085f88</code>, at <code>/opt/ullm/aq4-runtime-hardening-v0.1/control-source/aq4-hardening-activation-af7298bad50c/</code>. It seals all five dedicated route files, including the operation payload, and has no Git alternates. The root-only launcher binds that payload by SHA-256.
- The first new source seal (<code>05014a8c…</code>) produced an immutable preliminary plan but its isolated preflight stopped before worker start. Safe key-presence inspection established that systemd's MainPID is the gateway, whereas HIP guards and the served-manifest binding belong to its worker child. Commit <code>389a58f…</code> corrected that input binding and produced an unexecuted r2 plan; commit <code>af7298ba…</code> additionally preserves already successful endpoint states if the readiness deadline expires. The earlier plans are retained as immutable non-executed evidence; none changed <code>active.json</code>. The final source reads only the manifest-bound worker child's whitelisted environment, never copies its full environment or credentials.
- The final immutable plan is <code>/opt/ullm/aq4-runtime-hardening-v0.1/activation-v0.2-r3/activation-plan.json</code>, SHA-256 <code>0e12fe09ad4d00578ee74f1bcc730a6b401e63a6fc91bb1d237346251e8f81f8</code>. It reuses the existing frozen candidate SHA-256 <code>c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4</code> and exact rollback SHA-256 <code>5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a</code>; it does not recreate the closure, products, releases, tokenizers, or promotion source.
- Both reconciliation and live observation use the same bounded readiness contract: a 120-second deadline, at most 15 attempts, 0.5/1/2/4-second exponential delays then an 8-second cap, and two consecutive coherent observations of the same worker PID/starttime identity. The full idle delay budget is 87.5 seconds, leaving bounded probe time inside 120 seconds. This is deliberately much larger than the diagnosed roughly 3-second gateway startup and 4.8-second direct candidate worker-ready observation, while still failing closed rather than waiting indefinitely. Reconciliation operations receive a 240-second outer limit (90-second systemd action + 120-second readiness + margin); observe and isolated-worker operations receive 150 seconds.
- A coherent pass now requires stable systemd and worker identity, the active manifest unchanged across the pass, the actual worker command and worker environment bound to that manifest, and successful status/model-ID checks for all five gateway/OpenWebUI endpoints. The same wait contract is used for candidate, rollback, and recovery reconciliation/proof paths. A timeout or any incoherent pass is a failure that enters the existing rollback/recovery path.
- Failed candidate or rollback live proof now publishes a unique immutable audit with explicit <code>stage_status: failed</code>, timestamps, sanitized return code/stderr/cause, stderr/stdout digests, and every endpoint's success/failure state. The audit accepts only fixed cause codes, discards stdout content, redacts bearer/API-key/token/JWT/session/secret/password forms, and bounds retained stderr to 16 KiB. CPU/mock tests inject API-key/JWT-shaped stderr and assert that neither secret is published.
- The strengthened preflight first requires a separately immutable candidate-isolated-worker receipt. It launches only the candidate worker as the gateway user, from the gateway working directory, with a whitelisted offline/cache/HIP/lock environment and all 30 manifest guard flags copied from the current manifest-bound worker; it does not restart the service or replace the active manifest. The final successful receipt reports <code>gfx1201</code>, profile <code>rdna4_aq4_resident</code>, ready after 3,195 ms, and deliberate SIGTERM cleanup (<code>returncode: -15</code>).
- The final normal read-only preflight reports <code>ready: true</code>, <code>blockers: []</code>, and <code>production_activation_performed: false</code>. Its initial preflight correctly reported only the missing isolated-worker receipt. The evidence is under <code>benchmarks/results/2026-07-26/aq4-runtime-hardening-activation-v0.2/</code>. No service restart was issued in this corrective preparation; the service remained active/running with the same MainPID and <code>active.json</code> SHA-256.

The human gate remains unchanged: do not invoke <code>--execute</code>, swap <code>active.json</code>, or treat this ready report as execution approval. A human must separately approve the final plan SHA-256 and a service-maintenance window immediately before any activation.

### Phase 7 — Fresh AQ4 campaign and complete bundle v1

Purpose: replace old path-bound post-promotion evidence only after the hardened AQ4 manifest is live and proved.

This phase is a mandatory follow-on, not an admission prerequisite for the hardening activation. Putting it before activation would be circular because its release/browser outputs are active-manifest-bound. It is required before a later workflow may rely on AQ4 as a fresh prerequisite.

Use fresh outputs rooted below:

    /opt/ullm/aq4-runtime-hardening-v0.1/campaigns/release-bundle-v1/

Run and validate, against the hardened active manifest only, using these absolute paths in the sealed AQ4 promotion source:

    /opt/ullm/aq4-runtime-hardening-v0.1/sources/aq4-promotion-0cd760568e197/tools/run-generic-reasoning-release-campaign.py
    /opt/ullm/aq4-runtime-hardening-v0.1/sources/aq4-promotion-0cd760568e197/tools/validate-generic-reasoning-release.py
    /opt/ullm/aq4-runtime-hardening-v0.1/sources/aq4-promotion-0cd760568e197/tools/run-openwebui-reasoning-browser-smoke.py
    /opt/ullm/aq4-runtime-hardening-v0.1/sources/aq4-promotion-0cd760568e197/tools/validate-openwebui-reasoning-browser-smoke.py
    /opt/ullm/aq4-runtime-hardening-v0.1/sources/aq4-promotion-0cd760568e197/tools/prepare-generic-reasoning-release-evidence.py
    /opt/ullm/aq4-runtime-hardening-v0.1/sources/aq4-promotion-0cd760568e197/tools/prepare-generic-reasoning-release-bundle.py
    /opt/ullm/aq4-runtime-hardening-v0.1/sources/aq4-promotion-0cd760568e197/tools/validate-generic-reasoning-release-bundle.py

The generic campaign requires GPU index 1 and the running hardened gateway. The browser campaign requires the live frontend, an authorized OpenWebUI session/JWT, and its Docker/front-end dependencies. It is not satisfied by an old browser artifact. Do not add P3 flags to the running manifest or campaign environment.

The v1 bundle preparer requires all six components below its bundle output directory. Place the freshly generated release evidence/validation, browser evidence/validation, and byte copies of the newly generated protected promotion evidence/receipt in the same release-bundle workspace before preparation. This is allowed only for the new pair; historical pair reuse remains forbidden. The receipt's sibling-evidence relationship must remain valid in the bundle layout.

Before treating <code>bundle-v1.json</code> as durable downstream input, use the resolved Working Hypothesis 5 route: invoke the v1 preparer as root with <code>--required-uid 0</code>, then require root:root 0444, nlink 1, no ACL/capability, and <code>validate-generic-reasoning-release-bundle.py --require-immutable-publication --required-uid 0</code>. This route is implemented, but its first real bundle remains Phase 7 work after hardened AQ4 candidate live proof.

Exit criterion: fresh campaign/browser outputs and complete immutable bundle v1 are available and validate against the hardened active manifest. They are clearly marked post-hardening evidence, not authorization for the already-completed activation.

GPU or service window: GPU plus running gateway for the generic campaign; frontend/session dependencies plus live service for browser smoke; CPU/storage only for final bundle assembly after all inputs have stabilized.

## Decision Tree

| Checkpoint | If it passes | If it fails |
| --- | --- | --- |
| Phase 0 active/unit/env snapshot | Continue to protected-ancestry admission. | Stop before creating content. If live inputs drifted, write a superseding plan with the new facts; do not reinterpret old evidence. |
| Phase 1 ancestry/capacity | Create unique stages. | Do not copy. Correct only the empty AQ4 namespace under a reviewed operation, or choose a new protected target; never touch SQ8 paths. |
| Worker copy/hash/seal | Continue with product/tokenizer closure. | Quarantine the unique stage. Recopy from verified input; never rebuild or substitute a worker. |
| Product count/hash/full seal | Continue with tokenizer closure. | Return to product-selection verification. Do not add old root history, symlink, or artifact alias to satisfy a missing file. |
| Tokenizer five-file seal | Continue with source clone. | Return to tokenizer member selection/hash verification. Do not copy model shards/cache as a shortcut. |
| Promotion source clone seal | Continue to evidence. | Destroy no final path; inspect/recreate a new stage. Never seal a linked worktree or source with alternates. |
| GPU exclusivity / service window | Collect fresh promotion evidence. | Do not stop unrelated services or compete for GPU. Reschedule the window. |
| Fresh evidence/receipt validation | Generate/freeze candidate. | Keep the old active runtime. Fix only evidence inputs/tooling; old evidence cannot bridge the gap. |
| Candidate manifest diff/freeze/seal | Build activation plan. | Return to profile generation. Any unexpected field change, <code>/home</code> reference, or P3 flag is a hard stop. |
| Dedicated activation-route implementation/review | Run non-mutating preflight. | No activation. Complete a separately reviewed control-tool task; do not use SQ8 or generic bootstrap routes. |
| Locked preflight / human approval | Execute once under lock. | Without readiness or explicit human approval, stop with the old active manifest untouched. |
| Candidate live proof | Mark hardened AQ4 active and begin Phase 7. | Route must restore exact rollback bytes under the same lock and obtain rollback live proof; if that proof fails, enter incident recovery rather than making a health claim. |
| Fresh campaign/browser/bundle v1 | Publish downstream-ready AQ4 evidence. | Keep hardened AQ4 live if live proof remains healthy; repair campaign/bundle tooling or prerequisites separately. Do not reactivate old evidence. |

## Risks

| Risk | Prevention | Required response |
| --- | --- | --- |
| 7.7 GB product copy is interrupted | Unique staging, capacity gate, stage hashes/counts, flush before no-clobber publish. | Leave the incomplete stage for inspection; do not seal, activate, or auto-delete it. Start a new verified stage when ready. |
| Destination capacity is exhausted or sparse files expand | Gate on 16 GiB free using logical package bytes, not <code>du</code> allocation alone. | Stop before copy. Expand/free only an explicitly reviewed non-SQ8 target; never delete broadly under <code>/opt</code>. |
| Old evidence/receipt is accidentally reused | Use new paths/names and reject old manifest SHA/path references in validators and review. | Discard the candidate evidence workflow and recollect fresh evidence. |
| Symlink removal produces package mismatch | Retain the exact <code>package/</code> closure and keep <code>product.artifact: null</code>; validate full product tree and package hash. | Return to closure selection. Do not recreate <code>artifact -&gt; package</code> in the final product root. |
| A copy has hidden hardlinks/reflinks/ACL/capabilities | Explicit no-hardlink/no-reflink copy method plus post-copy link/ACL/capability scans and recursive runtime seal. | Reject the stage and recreate it with the compliant copy method. |
| Standalone clone secretly shares objects or is a linked worktree | Require <code>git clone --no-hardlinks</code>, directory <code>.git</code>, no alternates, detached commit/tree checks. | Reject it; create a fresh clone without shared/reference/worktree options. |
| Rollback target is lost or stale | Capture exact current active bytes immediately before execute; bind their SHA to the locked plan and preserve an immutable copy. | Do not execute if capture differs from fixed current SHA. If post-execute rollback cannot prove legacy health, enter incident recovery. |
| Unit/environment drift changes behavior during activation | Hash both at admission, preflight, and under lock. | Abort before swap and regenerate the plan after review. |
| SQ8 assets are accidentally incorporated or modified | Use only the dedicated AQ4 root; prohibit writes beneath <code>/opt/ullm/releases</code> and SQ8 source/campaign paths. | Stop, audit the write scope, and remediate under a separate SQ8-aware incident workflow. |
| P3 optimization leaks into hardening | Derive profile from live active manifest and assert exact 30 flags. | Reject candidate/profile and rebuild it mechanically; do not benchmark or tune in this task. |
| Service recovery starts comparison services | Route/service scripts explicitly allow restart of <code>ullm-openai.service</code> only. | Stop and audit service state; return <code>llama-qwen35-udq4.service</code> to disabled/inactive and keep <code>gdm3</code> inactive before proceeding. |
| Bundle v1 looks valid but is mutable | Use the implemented root-owner no-replace publisher and immutable-validator mode; rerun validation after final mode/ownership. | Do not use it downstream until the Phase 7 bundle has passed this publication and validation sequence. |

## Next Actions

1. Do not invoke activation, rollback, or recovery again from plan <code>72140ff475b29e28f4ab6685459a344939bc54fcd12aa4f0b7c44cd7a8753194</code>. Preserve its immutable outcome/audit as incident evidence.
2. Treat plan <code>0e12fe09ad4d00578ee74f1bcc730a6b401e63a6fc91bb1d237346251e8f81f8</code> as prepared but unexecuted. Its <code>ready: true</code> report and isolated-worker receipt are not permission to invoke <code>--execute</code>.
3. Before any future human-gated attempt, repeat the plan-bound read-only preflight and inspect active-manifest, unit, environment, worker, product, tokenizer, source, credential seals, service state, and the isolated-worker receipt. If any bound input (including a credential seal) has drifted, stop and prepare another reviewed plan instead of adapting this one in place.
4. Only a new explicit approval, a service-maintenance window, and a successful candidate live proof can unlock Phase 7. Until then, do not collect or publish post-hardening campaign/browser/bundle evidence.
