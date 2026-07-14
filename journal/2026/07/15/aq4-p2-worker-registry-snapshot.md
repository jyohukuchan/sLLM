# AQ4 P2 worker registry snapshot hardening

## 前回の要点

benchmark workerはtyped production caseとstartup registryのID↔SHAを照合していたが、registry fileのraw bytes identityと読み取りFDをproduction水準では固定していなかった。

## 今回の変更点

- CLIにregistry file bytesの期待SHA-256を必須化した。
- registry JSONにcanonical `registry_sha256` self-hashを追加した。
- absolute/no-parent、全ancestor/leaf非symlink、`O_NOFOLLOW`、regular/nlink=1を必須化した。
- device/inode/mode/size/mtime/ctime/link countをopen前後とread後のFD/pathで照合した。
- 1回だけopenした同じFDのbyte列からraw SHA、strict parse、self-hashを検証し、再openを行わない。
- rename、同一size rewriteとmtime復元、hardlink、leaf/ancestor symlinkの負例を追加した。

## 次の行動

target/bin/lib/check/fmtを完走し、他agentのbinding/launcher差分を含めず独立コミットにする。
