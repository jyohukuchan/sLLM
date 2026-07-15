# P3 profile actual-v12 実行記録

## 前回の要点

- ready-v14 は commit `39af01d5dca7c76eb53fdbffc59dc976a2d24e6c`、operator-v12 実行ツールは commit `ad327b427a0cd4eed73078296316257f314b72c1` で封印済みだった。
- operator-command-v12 は commit `2185ac90f7188402c60280e87b8eded3cbfc65e8` で封印され、exact argv は 10 引数、`shell=false`、最大実行回数は 1 回だった。
- 最終直前監査では fresh output 9 件が不在で、SQ8 の親・cargo・test は 0、サービスは `active/running`、GPU 所有者は既存 worker のみだった。

## 今回の変更点

- 同一 PTY 内で sudo cache を確立し、封印済み exact argv を 1 回だけ実行した。開始時刻は `1784145178632798169 ns`、終了時刻は `1784145263570451066 ns`、経過時間は `84937652897 ns`、return code は `1` だった。invocation は `1/1`、retry は `0`、`shell=false` を維持した。
- 失敗地点は maintenance の `profile-capture`、launcher の `runner` だった。直接原因は rocprof capture が `hipMemcpyAsync` を未知の転送・同期 HIP API と判定したことだった。capture は起動済みで、runtime と trace を含む失敗証拠を保持した。
- outer-finally の finalizer は 1 回で成功し、状態を `failed_immutable_evidence_preserved_restore_passed` として封印した。復旧分類は `outer_finally_restored_new_epoch`、restore は 6 poll・`14936290949 ns` で成功した。
- posthealth は `ullm-openai.service` が `active/running`、MainPID `1956991`、worker PID `1957364`、NRestarts `0` だった。AMD SMI と KFD の対象 GPU 所有者はいずれも worker `1957364` のみで、lock は busy、対象残留プロセスは 0 だった。
- package integrity は 7,700,872,459 bytes・1,045 files の full content hash を exact 1 回だけ実行し、SHA-256 `a24774432d3f0b7f175dc761ef9a53df1fed901dd02f825e8542b17181f004b1` で合格した。tree metadata も full hash の前後で不変だった。
- maintenance-v10、execute-evidence-v9、runtime-v9、capture-v9、operator-result-v12、actual-audit-v12 の 6 ルートで `SHA256SUMS` を検証した。全ルートは mode `0555`、全 member は mode `0444`・nlink `1` だった。
- 各 `SHA256SUMS` の SHA-256 は順に `5f5109a69466a8beaf5d825b68bfafde7a38a256e98f35ed8853b2254d592e46`、`dba8056b72ccacb88c454087a422b133c4fab2932e3bb2bd6dd405972f78c4d4`、`a661cb54edce7e7ba7a2badefe5c974020d9a4a88fb63072771cc1e4f4b110ff`、`7ce99040c5f54f4867f5a4cc175b2d25b4761691474810305088e07ca4eab251`、`d35c26e9e8b229a39952ec895e62a5b8e6575ed6c859cee1b761f61b9fa86901`、`bf2d23422a807e61b4828a49d093e52ab6cb508904ba2b3359ff030315eee074` だった。
- actual audit の audit SHA-256 は `1f9317280d426cbc989814a56b3c3a8341c6359d7ec197cf8a18d910a6b8bdc9`、capture failure SHA-256 は `20e55d0b95ffa067f07aa0bd36857714aaf4beb82b58a03c9bc18bd7c9e1cc57` だった。
- `validate-actual` は合格した。pytest は 34 件中 33 件が合格し、1 件は actual-v12 の生成後に旧 actual-v11 状態を `partial or mixed` と読む事後状態依存の失敗だった。封印済み実行ツールと証拠は変更していない。
- 秘密情報は stdout、stderr、JSON に記録されず、audit の `secret_material_recorded` は false だった。文字列一致 2 件は capture の `agent_info.csv` にある CPU 製品名であり、sudo の資格情報ではないことを伏字付き文脈で確認した。それ以外の証拠 member に一致はなかった。

## 次の行動

- actual-v12 は retry 禁止のため再実行しない。
- 今回の 6 証拠ルートと本 journal は同じ commit に保存した。commit 後の archive と Git object の byte equality は 36 ファイル・mismatch 0、SUMS・mode・nlink・`validate-actual`・package exact1・残留 0・restore・posthealth はすべて合格した。
- rocprof capture の `hipMemcpyAsync` 分類互換性を、今回の immutable failure evidence を入力として別作業で修正する。
