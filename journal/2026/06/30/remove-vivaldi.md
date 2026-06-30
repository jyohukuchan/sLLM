# Remove Vivaldi apt source

## Done

- Removed `vivaldi-stable`.
- Removed `/etc/apt/sources.list.d/vivaldi.list`.
- Removed `/etc/apt/sources.list.d/vivaldi.sources`.
- Removed `/usr/share/keyrings/vivaldi-16BD9233.gpg`.
- Re-ran `apt-get update` successfully.

## Notes

- The first `apt-get update` attempt hit a transient apt lock after purge, but the lock cleared and the retry succeeded.
- No Vivaldi package or apt source file remained after removal.
