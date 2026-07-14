# P2 active fidelity gate attempt 1

## 前回の要点

active fidelity gateは、v32 source artifact、plan/cases、active identity、binary receiptを固定し、mock preflightを通過していた。GPU/serviceの本番実行は未実施だった。

## 今回の変更点

commit `c64f2b79290f9aeecabfddaf21efe35a1d62af89` のgateでattempt 1を開始した。gate SHA256は `1accf99f7ad27fffe1413dac5276de385d9a047d437405175ed5f21d994702e9` である。固定入力preflightは通過したが、`systemctl stop` と復旧時の `systemctl start` が `Interactive authentication required (homelab1)` で失敗し、gateはexit 90となった。

GPU captureへは到達せず、`RUN_STARTED`、output、metrics、service-stopped markerは作成されなかった。失敗後の読み取り確認では、serviceはactive/running、MainPID `647551`（`ullm-openai-gat*`）、`NRestarts=0`だった。RuntimeDirectoryは directory mode 750 uid/gid 1000 nlink 2、runtime lockは regular empty mode 600 uid/gid 1000 nlink 1で ownerは `647551`。active/package/worker SHAはそれぞれ `feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`、`a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad`、`177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d`で不変だった。

## 次の行動

gateのsystemd呼び出しを固定配列 `SYSTEMCTL=(sudo -n -- systemctl)` に統一する。実行者は同一PTYで事前に `sudo -v` を行うが、パスワードはコマンド、環境、ファイル、ログへ保存しない。GPU/service操作をこの修正作業中に再実行しない。
