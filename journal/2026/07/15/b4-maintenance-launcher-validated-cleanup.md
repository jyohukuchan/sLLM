# B4 maintenance launcher-validated cleanup

## 前回の要点

profile capture callback の raw outcome は、launcher が schema を検証する前から maintenance から参照できていた。launcher が schema を拒否して validated `profile_capture` / `profile_diagnostics` を保存しない場合にも、maintenance が raw lifecycle state へ fallback して lock substrate cleanup の判断に使う経路が残っていた。

## 今回の変更点

- cleanup authority を launcher evidence 内の validated `profile_capture` と `profile_diagnostics` に限定した。
- capture 実行後に validated evidence が欠ける場合、runner、children、cleanup state を `unknown` として記録し、lock cleanup 関数を呼ばない。
- raw callback outcome は `diagnostic_only_unvalidated` と明示した診断欄だけに保存する。
- lock substrate を保持しても、outer service restore は従来どおり必ず試行する。
- schema rejection と、raw outcome が `cleanup_passed=true`、`children=[]`、`runner_finished=true` を偽装する場合の integration test を追加した。

## 検証

- `python3 -m py_compile tools/run-aq4-p2-resident-smoke-maintenance.py tests/test_aq4_p2_resident_smoke_maintenance.py`: pass
- 新規および cleanup / restore 関連: 10 passed
- artifact と並行変更中の capture API fixture を除く maintenance tests: 120 passed, 15 deselected
- `git diff --check`: pass

全件実行では、self-hash 付き canonical artifact に依存する 11 件は作業中の source hash と artifact pin が一致しない。さらに共有 worktree で並行変更中の capture API に既存 fixture が追随していない 3 件が失敗した。担当外の artifact、launcher、capture source は変更していない。

## 次の行動

launcher が validated lifecycle evidence を保存する成功・既知失敗経路は従来の cleanup contract を維持する。artifact 再生成と capture fixture の更新後に全件を再実行する。
