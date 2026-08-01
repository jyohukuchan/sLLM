# uLLM project reset

## Objective

Preserve the current Git history and all local files, publish outstanding source work, then make the remote `main` tree empty and leave `/home/homelab1/coding-local/ultimateLLM` as an empty local directory.

## Preflight state

- Local `main`: `bc4a5754b43eafbd4606f76650b1791ab908ef3d`
- Remote `main`: `84153dd9ca1bca67491b67c8af494a6c6d16c6a7`
- Local `main` is 103 commits ahead and 0 behind.
- The main worktree has 18 tracked modifications.
- Untracked data contains at least 711,758 files and about 93 GB, including regular-Git-ineligible multi-GB generated artifacts.
- The complete workspace is about 155 GB.
- Local-only branches and linked worktrees require separate preservation.
- `ullm-openai.service` directly runs from this checkout and must be stopped before removal.

## Safety sequence

1. Stop services and writers that use the checkout.
2. Copy the workspace and external dirty worktrees to the local ZFS backup area with ownership and metadata preserved.
3. Verify the backup with a second rsync pass and inventory records.
4. Commit selected source changes and dirty linked-worktree tracked changes.
5. Save a Git bundle and publish pre-reset archive refs.
6. Push the preserved `main`, then commit and push an empty tree.
7. Verify the remote SHA/tree and remove the local workspace contents.
