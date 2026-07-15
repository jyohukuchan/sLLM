# SQ8正式認可系譜スキーマ

## 前回の要点

- 実装GO、過去のcapture No-Go 2件、消費済みactual failure 2件、restore No-Go 1件は別々の監査・失敗receiptに残っていた。
- 旧builderは過去failureの一部だけを任意の診断lineageとして扱い、正式認可入力全体を固定request identityへ束縛していなかった。

## 今回の変更点

- exact 6 entriesの`ullm.sq8_authorization_lineage_input.v1`を追加した。順序、relation、typed field、source receiptのlive SHAと内容、重複pathを検査する。
- manifestと全sourceはcanonical absolute path、regular 0444、single-link、non-symlinkを必須にし、unknown/duplicate/missing key、順序変更、path alias、symlink、hardlink、mode/hash driftをfail closedにした。
- manifestのinput path/SHA-256とentries digestをpromotion request IDへ束縛し、runtime copy/refをprofile、prepared/actual receipt、served manifest、build receipt、Gate、runner snapshot、SHA256SUMSへ伝播した。
- actual authorizationは新manifestを必須にした。旧schemaは未認可診断だけを許し、actual authorizationには再利用できない。
- builder materialization終端とrunner dry-run/execute前後でexternal manifest、runtime copy、全sourceを再hashする。

## 検証

- lineage/builder/receipt/runner/generator関連: 133 passed。
- 別worktreeの既存deployment profile絶対pathに依存する既存1件だけは対象外。
- GPU、service、sudo操作: 0。

## 次の行動

1. 実装commitを固定し、そのcommit identityを持つexact 6-entry manifestをcreate-newで生成する。
2. 新manifest付きfresh unauthorized runtimeをcreate-newでmaterializeする。
3. 新runtimeの独立監査がGOになるまでactual authorizationとGPU/service実行を禁止する。
