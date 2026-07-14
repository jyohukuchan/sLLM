# p2 GPU attempt3 production run

## 前回の要点

- attempt3 template の preflight/mock は通過済みで、attempt2 との差分は専用 paths と output 内 SHA256 verifier だけだった。
- production run は独立許可まで保留していた。

## 今回の変更点

- commit `57d07f2e8d610bf9e8e23778eb71daf1f244c500` の freeze script を env unset で一度だけ実行した。
- rc=0、wall=11.80s。output の SHA256SUMS は 3/3 OK、3 rows×35 stages、missing stages なし。
- source-v2 比較では 3 行すべて `decoder_layer:0` が first mismatch。greedy は `[41330,16,15]`。
- observer は 6 samples、card2 最大 41% GPU use・7,439,523,840 B VRAM。service は active/running、NRestarts=0、healthz OK、lock owner は MainPID と一致した。
- attempt1/2 の raw SHA は不変、attempt3 は専用 residue のみを保持した。

## 次の行動

- raw evidence の独立確認後、attempt3 output/log/markers とこの journal を証拠 commit に保存する。
- 失敗再試行や追加 GPU/service 操作は実施しない。

検証済み: output 内 `sha256sum -c`、専用 analyzer rc=0、service/health/lock read-only checks。未実施: attempt3 再実行、追加 GPU run。
