# AQ4_0 runtime hardening Phase 1–3

Phase 1–3 only: protected ancestry, minimal runtime closures, and standalone source clones. Phase 4 evidence/receipt/profile/manifest work, Phase 5 control-route work, activation, service control, and GPU use were not performed.

## Published protected layout

- `/opt/ullm/aq4-runtime-hardening-v0.1/releases/aq4-fidelity-f1a3cf4c/`
- `/opt/ullm/aq4-runtime-hardening-v0.1/products/qwen35-9b-aq4-package-a790a033f57d/`
- `/opt/ullm/aq4-runtime-hardening-v0.1/tokenizers/qwen35-9b-qwen2tokenizer-a4aee8afcf2e/`
- `/opt/ullm/aq4-runtime-hardening-v0.1/sources/aq4-promotion-0cd760568e197/`
- `/opt/ullm/aq4-runtime-hardening-v0.1/control-source/manifest-freezer-f71bb2e534b/`

All final closure trees are root-owned. Final directories are `0555`; worker/legacy-engine leaves are `0555`; product, tokenizer, and source leaves are `0444`.

## Verification summary

- Live `active.json`, unit, and gateway environment hashes remain the planned values. The live manifest still contains exactly 30 required worker flags and no P3-only key.
- The protected worker is `cmp`-identical to the live worker: SHA-256 `1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b`, 4,223,912 bytes, and an independent inode.
- The product closure contains 1,045 regular files, five directories, and 7,700,872,459 logical bytes. Its manifest SHA-256 is `a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad`; all 1,045 destination hashes match the captured source ledger.
- The tokenizer closure contains only the five planned files; all five `cmp`/SHA-256 checks pass.
- The promotion source is detached at `0cd760568e197e1adb4c4df3d6149591a912f709` with tree `bd372761d7e79b3d5db2b533cdd3fdfa77f125c2`. The freezer control source is detached at `f71bb2e534b12bbf0ab37e716da1090c485ab733` with tree `3cd6e0ace792a070192c11af39e7f83c45de8d0a`; it contains `tools/freeze-served-model-manifest.py`.
- Every final closure and protected ancestor scanned with zero symlink, special-file, ordinary-hardlink, group/world-write, extended/default ACL, and capability violations. `git fsck --no-dangling` passes for both standalone clones.

`closure-members.tsv` records SHA-256, owner, mode, nlink, size, device, and inode for all 24,634 final closure members; directories have `sha256=-`. Its SHA-256 is `ac59ca737563a89d36765bf894972c872bd5cda695b08a500c4f96eb2d49dc4a`.

The source clone needs `core.filemode=false` to preserve a clean Git status after the required `0444` mode seal makes tracked executable files non-executable. This setting is local to each protected clone and is noted for the plan review.

## Deliberate non-completion

The Phase 4+ directories are present but empty. Therefore the complete runtime seal/activation readiness is intentionally `NOT_READY_EXPECTED`: fresh evidence, receipt, candidate profile, frozen manifest, activation plan, and reviewed activation control source do not exist yet.

An initial promotion-source staging clone became Git-dirty solely from the `0555/0444` seal while `core.filemode=true`. It remains quarantined at `/opt/ullm/aq4-runtime-hardening-v0.1/.staging/sources/aq4-promotion-0cd760568e197.20260725T193039Z.03ovtM` and was not published or reused. The final clone was newly created and passed all checks.

The start-of-task service snapshot showed `ullm-openai.service` inactive; the final read-only snapshot showed it active. No service command was issued in this task; the source of that external state change is unconfirmed.

See the TSV/JSON records in this directory for the complete audit trail. `hashes.sha256` covers the result files.
